import os
import time
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils import CrossEntropyLoss2d
from models import reactive_net, reinforcement_net
from scipy import ndimage
import matplotlib.pyplot as plt
import copy
from apex import amp


class Trainer(object):
    def __init__(self, method, future_reward_discount,
                 load_snapshot, snapshot_file, force_cpu):

        self.method = method
        # Check if CUDA can be used
        if torch.cuda.is_available() and not force_cpu:
            print("CUDA detected. Running with GPU acceleration.")
            self.use_cuda = True
        elif force_cpu:
            print("CUDA detected, but overriding with option '--cpu'. Running with only CPU.")
            self.use_cuda = False
        else:
            print("CUDA is *NOT* detected. Running with only CPU.")
            self.use_cuda = False

        # Fully convolutional classification network for supervised learning
        if self.method == 'reactive':
            self.model = reactive_net(self.use_cuda)
            
            # Initialize classification loss
            suction_num_classes = 3 # 0-suction, 1-failed suction, 2- no loss
            suction_class_weights = torch.ones(suction_num_classes)
            suction_class_weights[suction_num_classes - 1] = 0
            if self.use_cuda:
                self.suction_criterion = CrossEntropyLoss2d(suction_class_weights.cuda()).cuda()
            else:
                self.suction_criterion = CrossEntropyLoss2d(suction_class_weights)
            
            grasp_num_classes = 3 # 0 - grasp, 1 - failed grasp, 2 - no loss
            grasp_class_weights = torch.ones(grasp_num_classes)
            grasp_class_weights[grasp_num_classes - 1] = 0
            if self.use_cuda:
                self.grasp_criterion = CrossEntropyLoss2d(grasp_class_weights.cuda()).cuda()
            else:
                self.grasp_criterion = CrossEntropyLoss2d(grasp_class_weights)
                
            gs_num_classes = 3 # 0 - gs, 1 - failed gs, 2 - no loss
            gs_class_weights = torch.ones(gs_num_classes)
            gs_class_weights[gs_num_classes - 1] = 0
            if self.use_cuda:
                self.gs_criterion = CrossEntropyLoss2d(gs_class_weights.cuda()).cuda()
            else:
                self.gs_criterion = CrossEntropyLoss2d(gs_class_weights) 
                
            # # Load pre-trained model
            if load_snapshot:
                self.model.load_state_dict(torch.load(snapshot_file))
                print('Pre-trained model snapshot loaded from: %s' % (snapshot_file))

            # Convert model from CPU to GPU
            if self.use_cuda:
                self.model = self.model.cuda()
                               
        # Fully convolutional Q network for deep reinforcement learning
        elif self.method == 'reinforcement':
            self.model = reinforcement_net(self.use_cuda)
            self.model_target = copy.deepcopy(self.model)
            self.model_target.load_state_dict(self.model.state_dict())
            
            self.future_reward_discount = future_reward_discount

            # Initialize Huber loss
            self.criterion = torch.nn.SmoothL1Loss(reduction='none') # Huber loss
            if self.use_cuda:
                self.criterion = self.criterion.cuda()

            # # Load pre-trained model
            if load_snapshot:
                self.model.load_state_dict(torch.load(snapshot_file))
                print('Pre-trained model snapshot loaded from: %s' % (snapshot_file))

            # Convert model from CPU to GPU
            if self.use_cuda:
                self.model = self.model.cuda()
                self.model_target = self.model_target.cuda()

        # Set model to training mode
        self.model.train()

        # Initialize optimizer
        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-4, momentum=0.9, weight_decay=2e-5)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4, betas=(0.9,0.999),eps=1e-8, weight_decay=0)
        
        self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level="O0")
        self.iteration = 0

        # Initialize lists to save execution info and RL variables
        self.executed_action_log = []
        self.label_value_log = []
        self.reward_value_log = []
        self.predicted_value_log = []
        self.use_heuristic_log = []
        self.is_exploit_log = []
        self.clearance_log = []        
        self.grasping_type_log = []
        self.episode_success_log = []
        self.training_loss_log = []


    # Pre-load execution info and RL variables
    def preload(self, transitions_directory):
        self.executed_action_log = np.loadtxt(os.path.join(transitions_directory, 'executed-action.log.txt'), delimiter=' ')
        self.iteration = self.executed_action_log.shape[0] - 2
        self.executed_action_log = self.executed_action_log[0:self.iteration,:]
        self.executed_action_log = self.executed_action_log.tolist()
        self.label_value_log = np.loadtxt(os.path.join(transitions_directory, 'label-value.log.txt'), delimiter=' ')
        self.label_value_log = self.label_value_log[0:self.iteration]
        self.label_value_log.shape = (self.iteration,1)
        self.label_value_log = self.label_value_log.tolist()
        self.predicted_value_log = np.loadtxt(os.path.join(transitions_directory, 'predicted-value.log.txt'), delimiter=' ')
        self.predicted_value_log = self.predicted_value_log[0:self.iteration]
        self.predicted_value_log.shape = (self.iteration,1)
        self.predicted_value_log = self.predicted_value_log.tolist()
        self.reward_value_log = np.loadtxt(os.path.join(transitions_directory, 'reward-value.log.txt'), delimiter=' ')
        self.reward_value_log = self.reward_value_log[0:self.iteration]
        self.reward_value_log.shape = (self.iteration,1)
        self.reward_value_log = self.reward_value_log.tolist()
        self.use_heuristic_log = np.loadtxt(os.path.join(transitions_directory, 'use-heuristic.log.txt'), delimiter=' ')
        self.use_heuristic_log = self.use_heuristic_log[0:self.iteration]
        self.use_heuristic_log.shape = (self.iteration,1)
        self.use_heuristic_log = self.use_heuristic_log.tolist()
        self.is_exploit_log = np.loadtxt(os.path.join(transitions_directory, 'is-exploit.log.txt'), delimiter=' ')
        self.is_exploit_log = self.is_exploit_log[0:self.iteration]
        self.is_exploit_log.shape = (self.iteration,1)
        self.is_exploit_log = self.is_exploit_log.tolist()
        self.clearance_log = np.loadtxt(os.path.join(transitions_directory, 'clearance.log.txt'), delimiter=' ')
        self.clearance_log.shape = (self.clearance_log.shape[0],1)
        self.clearance_log = self.clearance_log.tolist()
        
        self.grasping_type_log = np.loadtxt(os.path.join(transitions_directory, 'grasping_type.log.txt'), delimiter=' ')
        self.grasping_type_log = self.grasping_type_log[0:self.iteration]
        self.grasping_type_log.shape = (self.iteration,1)
        self.grasping_type_log = self.grasping_type_log.tolist()
        
        self.episode_success_log = np.loadtxt(os.path.join(transitions_directory, 'episode_success.log.txt'), delimiter=' ')
        self.episode_success_log = self.episode_success_log[0:self.iteration,:]
        self.episode_success_log = self.episode_success_log.tolist()
        
        self.training_loss_log = np.loadtxt(os.path.join(transitions_directory, 'training_loss.log.txt'), delimiter=' ')
        self.training_loss_log = self.training_loss_log[0:self.iteration,:]
        self.training_loss_log = self.training_loss_log.tolist()


    # Compute forward pass through model to compute affordances/Q
    def forward(self, depth_heightmap, m_depth_heightmap, style=0, is_volatile=False, is_target=False, specific_rotation=-1):

        # Apply 2x scale to input heightmaps
        depth_heightmap_2x = ndimage.zoom(depth_heightmap, zoom=[2,2], order=0)
        m_depth_heightmap_2x = ndimage.zoom(m_depth_heightmap, zoom=[2,2], order=0) 

        # Add extra padding (to handle rotations inside network)
        diag_length = float(depth_heightmap_2x.shape[0]) * np.sqrt(2)
        diag_length = np.ceil(diag_length/32)*32
        padding_width = int((diag_length - depth_heightmap_2x.shape[0])/2)
        depth_heightmap_2x =  np.pad(depth_heightmap_2x, padding_width, 'constant', constant_values=0)
        m_depth_heightmap_2x =  np.pad(m_depth_heightmap_2x, padding_width, 'constant', constant_values=0)

        # Pre-process depth image (normalize)
        image_mean = [0.0, 0.0, 0.0]
        image_std = [0.0, 0.0, 0.0]
        depth_heightmap_2x.shape = (depth_heightmap_2x.shape[0], depth_heightmap_2x.shape[1], 1)
        input_depth_image = np.concatenate((depth_heightmap_2x, depth_heightmap_2x, depth_heightmap_2x), axis=2)        
        m_depth_heightmap_2x.shape = (m_depth_heightmap_2x.shape[0], m_depth_heightmap_2x.shape[1], 1)
        m_input_depth_image = np.concatenate((m_depth_heightmap_2x, m_depth_heightmap_2x, m_depth_heightmap_2x), axis=2)
        
        for c in range(3):
            input_depth_image[:,:,c] = (input_depth_image[:,:,c] - image_mean[c])/image_std[c]
            m_input_depth_image[:,:,c] = (m_input_depth_image[:,:,c] - image_mean[c])/image_std[c]

        # Construct minibatch of size 1 (b,c,h,w)
        input_depth_image.shape = (input_depth_image.shape[0], input_depth_image.shape[1], input_depth_image.shape[2], 1)
        input_depth_data = torch.from_numpy(input_depth_image.astype(np.float32)).permute(3,2,0,1)
        m_input_depth_image.shape = (m_input_depth_image.shape[0], m_input_depth_image.shape[1], m_input_depth_image.shape[2], 1)
        m_input_depth_data = torch.from_numpy(m_input_depth_image.astype(np.float32)).permute(3,2,0,1)
        
        # Pass input data through model        
        if self.method == 'reactive':
            output_prob = self.model.forward(input_depth_data, m_input_depth_data, style, is_volatile, specific_rotation)
            output_prob = output_prob[0].view([1,3,1,1])
            output_predictions = np.zeros(len(output_prob))            
            for rotate_idx in range(len(output_prob)):
                output_predictions = F.softmax(output_prob, dim=1).cpu().data.numpy()[0,0,0]      
        elif self.method == 'reinforcement':
            if is_target:
                Q_prob = self.model_target.forward(input_depth_data, m_input_depth_data, style, is_volatile, specific_rotation)
            else:
                Q_prob = self.model.forward(input_depth_data, m_input_depth_data, style, is_volatile, specific_rotation)                    
            output_predictions = np.zeros(len(Q_prob))            
            for rotate_idx in range(len(Q_prob)):
                output_predictions[rotate_idx] = Q_prob[rotate_idx].cpu().data.numpy()                                                       
        
        return output_predictions             


    def get_label_value(self, primitive_action, objects_number,
                        suction_success, grasp_success, gs_success, 
                        depth_heightmap,mask_depth,objects_mask,
                        bestg_id,bests_id,bestgs_g_id,bestgs_s_id,
                        exploit_action,bestg_conf, bests_conf, bestgs_conf):
                        
        if self.method == 'reactive':
            # Compute label value
            label_value = 0
            if primitive_action == 'suction':
                success_value = suction_success
                if not suction_success:
                    label_value = 1
            elif primitive_action == 'grasp':
                success_value = grasp_success
                if not grasp_success:
                    label_value = 1
            elif primitive_action == 'grasp_then_suction':
                success_value = gs_success
                if gs_success == 2.5:# or gs_success == 0.5:
                    label_value = 0
                else:
                    label_value = 1
            print('Label value: %d' % (label_value))
            return label_value,success_value
                
        elif self.method == 'reinforcement':
            # Compute current reward            
            current_reward = 0
            if primitive_action == 'suction':
                current_reward = suction_success
            elif primitive_action == 'grasp':
                current_reward = grasp_success
            elif primitive_action == 'grasp_then_suction':
                current_reward = gs_success
            # Compute future reward
            if suction_success ==0 and grasp_success ==0 and gs_success ==0:
                future_reward = 0
            elif (objects_number == 1 and suction_success == 1) or (objects_number == 1 and grasp_success == 1) or (objects_number == 2 and gs_success == 2.5):
                future_reward = 0
            else:
                # Q(s', a, w)
                #Q_suction_next, Q_grasp_next, _ = self.forward(next_color_heightmap, next_depth_heightmap, g_then_s = False, is_volatile=True, is_target=False)                
                # Q_target(s', a, w')
                #Qtar_suc_next, Qtar_gra_next, _ = self.forward(next_color_heightmap, next_depth_heightmap, g_then_s = False, is_volatile=True, is_target=True)
                # Q_target(s', argmax(Q(s', a, w)), w')
                                  
                if exploit_action == 'grasp':
                    d_heightmap_mask = depth_heightmap * mask_depth[bestg_id[0]]
                    future_reward = self.forward(depth_heightmap, d_heightmap_mask, style=0, is_volatile=True, is_target=True,specific_rotation = bestg_id[1])
                    future_reward = future_reward[0]                                    
                elif exploit_action == 'suction':
                    d_heightmap_mask = depth_heightmap * mask_depth[bests_id[0]]
                    future_reward = self.forward(depth_heightmap, d_heightmap_mask, style=1, is_volatile=True, is_target=True,specific_rotation = bests_id[1])
                    future_reward = future_reward[0]                              
                elif exploit_action == 'grasp_then_suction':
                    d_heightmap_mask = depth_heightmap * (mask_depth[bestgs_g_id[0]] + mask_depth[bestgs_s_id[0]])
                    future_reward = self.forward(depth_heightmap, d_heightmap_mask, style=2, is_volatile=True, is_target=True, specific_rotation = bestgs_g_id[1])
                    future_reward = future_reward[0]                    
            expected_reward = current_reward + self.future_reward_discount * future_reward
            print('Expected reward: %f + %f x %f = %f' % (current_reward, self.future_reward_discount, future_reward, expected_reward))

        return expected_reward, current_reward


    # Compute labels and backpropagate
    def backprop(self, depth_heightmap, primitive_action, 
                 bestg_id, bests_id, bestgs_g_id, bestgs_s_id, 
                 label_value, objects_mask, sro_best, gro_best, bestgs_num):
        
        if self.method == 'reactive':        
            # Compute labels
            label = np.zeros((1,1,1))           
            label[0,0,0] = label_value
            # Compute loss and backward pass
            mask_depth = objects_mask.copy()            
            objects_mask.shape = (objects_mask.shape[0], objects_mask.shape[1], objects_mask.shape[2], 1)
            self.optimizer.zero_grad()
            loss_value = 0
            if primitive_action == 'grasp':
                # Do forward pass with specified rotation (to save gradients)
                d_heightmap_mask = depth_heightmap * mask_depth[bestg_id[0]]
                gra_conf = self.forward(depth_heightmap, d_heightmap_mask, style = 0, is_volatile=False, is_target=False, specific_rotation = bestg_id[1])
                if self.use_cuda:
                    loss = self.grasp_criterion(self.model.gra_prob[0].view([1,3,1,1]), Variable(torch.from_numpy(label).long().cuda()))
                else:
                    loss = self.grasp_criterion(self.model.gra_prob[0].view([1,3,1,1]), Variable(torch.from_numpy(label).long()))                
                loss = loss.sum()
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
                loss_value = loss.cpu().data.numpy()
            
            elif primitive_action == 'suction':
                # Do forward pass with specified rotation (to save gradients)
                d_heightmap_mask = depth_heightmap * mask_depth[bests_id[0]]
                suc_conf = self.forward(depth_heightmap, d_heightmap_mask, style = 1, is_volatile=False, is_target=False, specific_rotation = bests_id[1])
                if self.use_cuda:
                    loss = self.suction_criterion(self.model.suc_prob[0].view([1,3,1,1]), Variable(torch.from_numpy(label).long().cuda()))
                else:
                    loss = self.suction_criterion(self.model.suc_prob[0].view([1,3,1,1]), Variable(torch.from_numpy(label).long()))                
                loss = loss.sum()
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
                loss_value = loss.cpu().data.numpy()            
            
            elif primitive_action == 'grasp_then_suction':
                # Do forward pass with specified rotation (to save gradients)
                d_heightmap_mask = depth_heightmap * (mask_depth[bestgs_g_id[0]] + mask_depth[bestgs_s_id[0]])
                gs_conf = self.forward(depth_heightmap, d_heightmap_mask, style = 2, is_volatile=False, is_target=False, specific_rotation = bestgs_g_id[1])
                if self.use_cuda:
                    loss = self.gs_criterion(self.model.gs_prob[0].view([1,3,1,1]), Variable(torch.from_numpy(label).long().cuda()))
                else:
                    loss = self.gs_criterion(self.model.gs_prob[0].view([1,3,1,1]), Variable(torch.from_numpy(label).long()))               
                loss = loss.sum()
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
                loss_value = loss.cpu().data.numpy()  

            print('Training loss: %f' % (loss_value))
            self.optimizer.step()
            return loss_value
                
        elif self.method == 'reinforcement':                        
            mask_depth = objects_mask.copy()            
            objects_mask.shape = (objects_mask.shape[0], objects_mask.shape[1], objects_mask.shape[2], 1)            
            # Compute loss and backward pass
            self.optimizer.zero_grad()
            loss_value = 0
            if primitive_action == 'grasp':
                # Do forward pass with specified rotation (to save gradients)
                d_heightmap_mask = depth_heightmap * mask_depth[bestg_id[0]]
                gra_conf = self.forward(depth_heightmap, d_heightmap_mask, style = 0, is_volatile=False, is_target=False, specific_rotation = bestg_id[1])
                if self.use_cuda:
                    if abs(self.model.gra_prob[0,0,0,0]-label_value) < 1:
                        loss = 0.5*((self.model.gra_prob[0,0,0,0]-label_value)**2)
                    else:
                        loss = abs(self.model.gra_prob[0,0,0,0]-label_value) - 0.5              
                loss = loss.sum()
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
                loss_value = loss.cpu().data.numpy()

            elif primitive_action == 'suction':
                # Do forward pass with specified rotation (to save gradients)
                d_heightmap_mask = depth_heightmap * mask_depth[bests_id[0]]
                suc_conf = self.forward(depth_heightmap, d_heightmap_mask, style = 1, is_volatile=False, is_target=False, specific_rotation = bests_id[1])
                if self.use_cuda:
                    if abs(self.model.suc_prob[0,0,0,0]-label_value) < 1:
                        loss = 0.5*((self.model.suc_prob[0,0,0,0]-label_value)**2)
                    else:
                        loss = abs(self.model.suc_prob[0,0,0,0]-label_value) - 0.5            
                loss = loss.sum()
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
                loss_value = loss.cpu().data.numpy()

            elif primitive_action == 'grasp_then_suction':                
                # Do forward pass with specified rotation (to save gradients)
                d_heightmap_mask = depth_heightmap * (mask_depth[bestgs_g_id[0]] + mask_depth[bestgs_s_id[0]])
                gs_conf = self.forward(depth_heightmap, d_heightmap_mask, style = 2, is_volatile=False, is_target=False, specific_rotation = bestgs_g_id[1])
                if self.use_cuda:
                    if abs(self.model.gs_prob[0,0,0,0]-label_value) < 1:
                        loss = 0.5*((self.model.gs_prob[0,0,0,0]-label_value)**2)
                    else:
                        loss = abs(self.model.gs_prob[0,0,0,0]-label_value) - 0.5                
                loss = loss.sum()
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
                loss_value = loss.cpu().data.numpy()                        
            
            print('Training loss: %f' % (loss_value))
            self.optimizer.step()
            return loss_value
            






