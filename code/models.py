#!/usr/bin/env python

from collections import OrderedDict
import numpy as np
from scipy import ndimage 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
import matplotlib.pyplot as plt
import time


class reactive_net(nn.Module):

    def __init__(self, use_cuda): # , snapshot=None
        super(reactive_net, self).__init__()
        self.use_cuda = use_cuda

        # Initialize network trunks with DenseNet pre-trained on ImageNet
        self.suction_depth_trunk = torchvision.models.densenet.densenet121(pretrained=True)
        self.grasp_depth_trunk = torchvision.models.densenet.densenet121(pretrained=True)
        self.gs_depth_trunk = torchvision.models.densenet.densenet121(pretrained=True)
        self.gnum_rotations = 1
        self.snum_rotations = 1
        # reactive network architecture for suction
        self.suctionnet_val = nn.Sequential(OrderedDict([
            ('suction-val-norm0', nn.BatchNorm2d(2048)),
            ('suction-val-relu0', nn.ReLU(inplace=True)),
            ('suction-val-conv0', nn.Conv2d(2048, 64, kernel_size=1, stride=1, bias=False)),
            ('suction-val-norm1', nn.BatchNorm2d(64)),
            ('suction-val-relu1', nn.ReLU(inplace=True)),
            ('suction-val-conv1', nn.Conv2d(64, 3, kernel_size=20, stride=1, bias=False))
        ]))
                
        # reactive network architecture for grasping
        self.graspnet_val = nn.Sequential(OrderedDict([
            ('grasp-val-norm0', nn.BatchNorm2d(2048)),
            ('grasp-val-relu0', nn.ReLU(inplace=True)),
            ('grasp-val-conv0', nn.Conv2d(2048, 64, kernel_size=1, stride=1, bias=False)),
            ('grasp-val-norm1', nn.BatchNorm2d(64)),
            ('grasp-val-relu1', nn.ReLU(inplace=True)),
            ('grasp-val-conv1', nn.Conv2d(64, 3, kernel_size=20, stride=1, bias=False))
        ]))
                
        # reactive network architecture for grasping_then_suction
        self.gsnet_val = nn.Sequential(OrderedDict([
            ('grasp-val-norm0', nn.BatchNorm2d(2048)),
            ('grasp-val-relu0', nn.ReLU(inplace=True)),
            ('grasp-val-conv0', nn.Conv2d(2048, 64, kernel_size=1, stride=1, bias=False)),
            ('grasp-val-norm1', nn.BatchNorm2d(64)),
            ('grasp-val-relu1', nn.ReLU(inplace=True)),
            ('grasp-val-conv1', nn.Conv2d(64, 3, kernel_size=20, stride=1, bias=False))
        ]))
                    
        # Initialize network weights
        for m in self.named_modules():
            if 'suction-' in m[0] or 'grasp-' in m[0] or 'gs-' in m[0]:
                if isinstance(m[1], nn.Conv2d):
                    nn.init.kaiming_normal_(m[1].weight.data)
                elif isinstance(m[1], nn.BatchNorm2d):
                    m[1].weight.data.fill_(1)
                    m[1].bias.data.zero_()

        # Initialize output variable (for backprop)
        self.gra_prob = []
        self.suc_prob = []
        self.gs_prob = []


    def forward(self, input_depth_data, m_input_depth_data, style=0, is_volatile=False, specific_rotation=-1):

        if is_volatile and specific_rotation == -1:
            with torch.no_grad():
                gra_prob = []
                suc_prob = []
                gs_prob = []

                if style == 0:
                    # Apply rotations to images
                    for rotate_idx in range(self.gnum_rotations):
                        rotate_theta = np.radians(rotate_idx*(360/self.gnum_rotations))    
                        # Compute sample grid for rotation BEFORE neural network
                        affine_mat_before = np.asarray([[np.cos(-rotate_theta), np.sin(-rotate_theta), 0],[-np.sin(-rotate_theta), np.cos(-rotate_theta), 0]])
                        affine_mat_before.shape = (2,3,1)
                        affine_mat_before = torch.from_numpy(affine_mat_before).permute(2,0,1).float()
                        if self.use_cuda:
                            flow_grid_before = F.affine_grid(Variable(affine_mat_before, requires_grad=False).cuda(), input_depth_data.size(), align_corners=True)                       
                        # Rotate images clockwise
                        if self.use_cuda:
                            with torch.no_grad():
                                rotate_depth = F.grid_sample(Variable(input_depth_data).cuda(), flow_grid_before, mode='nearest', align_corners=True)                       
                        # Compute intermediate features     
                        interm_grasp_depth_feat = self.grasp_depth_trunk.features(rotate_depth)                        
                        m_interm_grasp_depth_feat = self.grasp_depth_trunk.features(m_input_depth_data.cuda())
                        interm_grasp_feat = torch.cat((interm_grasp_depth_feat, m_interm_grasp_depth_feat), dim=1)                           
                        grasp_Q = self.graspnet_val(interm_grasp_feat)                                    
                        gra_prob.append(grasp_Q)
                                            
                    return gra_prob

                elif style == 1:
                    # Apply rotations to images
                    for rotate_idx in range(self.snum_rotations):
                        rotate_theta = np.radians(rotate_idx*(360/self.snum_rotations))    
                        # Compute sample grid for rotation BEFORE neural network
                        affine_mat_before = np.asarray([[np.cos(-rotate_theta), np.sin(-rotate_theta), 0],[-np.sin(-rotate_theta), np.cos(-rotate_theta), 0]])
                        affine_mat_before.shape = (2,3,1)
                        affine_mat_before = torch.from_numpy(affine_mat_before).permute(2,0,1).float()
                        if self.use_cuda:
                            flow_grid_before = F.affine_grid(Variable(affine_mat_before, requires_grad=False).cuda(), input_depth_data.size(), align_corners=True)                        
                        # Rotate images clockwise
                        if self.use_cuda:
                            with torch.no_grad():
                                rotate_depth = F.grid_sample(Variable(input_depth_data).cuda(), flow_grid_before, mode='nearest', align_corners=True)                       
                        # Compute intermediate features
                        interm_suction_depth_feat = self.suction_depth_trunk.features(rotate_depth)                        
                        m_interm_suction_depth_feat = self.suction_depth_trunk.features(m_input_depth_data.cuda())                        
                        interm_suction_feat = torch.cat((interm_suction_depth_feat, m_interm_suction_depth_feat), dim=1)                          
                        suction_Q = self.suctionnet_val(interm_suction_feat)
                        suc_prob.append(suction_Q)
                                     
                    return suc_prob                 
                
                elif style == 2:
                    # Apply rotations to images
                    rotate_idx = 0
                    rotate_theta = np.radians(rotate_idx*(360/self.gnum_rotations))    
                    # Compute sample grid for rotation BEFORE neural network
                    affine_mat_before = np.asarray([[np.cos(-rotate_theta), np.sin(-rotate_theta), 0],[-np.sin(-rotate_theta), np.cos(-rotate_theta), 0]])
                    affine_mat_before.shape = (2,3,1)
                    affine_mat_before = torch.from_numpy(affine_mat_before).permute(2,0,1).float()
                    if self.use_cuda:
                        flow_grid_before = F.affine_grid(Variable(affine_mat_before, requires_grad=False).cuda(), input_depth_data.size(), align_corners=True)                      
                    # Rotate images clockwise
                    if self.use_cuda:
                        with torch.no_grad():
                            rotate_depth = F.grid_sample(Variable(input_depth_data).cuda(), flow_grid_before, mode='nearest', align_corners=True)                   
                    # Compute intermediate features
                    interm_gs_depth_feat = self.gs_depth_trunk.features(rotate_depth)
                    m_interm_gs_depth_feat = self.gs_depth_trunk.features(m_input_depth_data.cuda())                  
                    interm_gs_feat = torch.cat((interm_gs_depth_feat, m_interm_gs_depth_feat), dim=1)                                         
                    gs_Q = self.suctionnet_val(interm_gs_feat)
                    gs_prob.append(gs_Q)                    

                    return gs_prob                                                                

        elif is_volatile and specific_rotation != -1:
            with torch.no_grad():
                gra_prob = []
                suc_prob = []
                gs_prob = []

                if style == 0:
                    rotate_idx = specific_rotation
                    rotate_theta = np.radians(rotate_idx*(360/self.gnum_rotations))                    
                    # Compute sample grid for rotation BEFORE branches
                    affine_mat_before = np.asarray([[np.cos(-rotate_theta), np.sin(-rotate_theta), 0],[-np.sin(-rotate_theta), np.cos(-rotate_theta), 0]])
                    affine_mat_before.shape = (2,3,1)
                    affine_mat_before = torch.from_numpy(affine_mat_before).permute(2,0,1).float()
                    if self.use_cuda:
                        flow_grid_before = F.affine_grid(Variable(affine_mat_before, requires_grad=False).cuda(), input_depth_data.size(), align_corners=True)                    
                    # Rotate images clockwise
                    if self.use_cuda:
                        with torch.no_grad():
                            rotate_depth = F.grid_sample(Variable(input_depth_data, requires_grad=False).cuda(), flow_grid_before, mode='nearest', align_corners=True)                    
                    # Compute intermediate features                    
                    interm_grasp_depth_feat = self.grasp_depth_trunk.features(rotate_depth)
                    # m_interm_grasp_color_feat = self.grasp_color_trunk.features(m_input_color_data.cuda())
                    m_interm_grasp_depth_feat = self.grasp_depth_trunk.features(m_input_depth_data.cuda())                    
                    interm_grasp_feat = torch.cat((interm_grasp_depth_feat, m_interm_grasp_depth_feat), dim=1)                    
                    grasp_Q = self.graspnet_val(interm_grasp_feat)                        
                    gra_prob = grasp_Q    
                
                    return gra_prob
            
                elif style == 1:
                    rotate_idx = specific_rotation
                    rotate_theta = np.radians(rotate_idx*(360/self.gnum_rotations))                        
                    # Compute sample grid for rotation BEFORE branches
                    affine_mat_before = np.asarray([[np.cos(-rotate_theta), np.sin(-rotate_theta), 0],[-np.sin(-rotate_theta), np.cos(-rotate_theta), 0]])
                    affine_mat_before.shape = (2,3,1)
                    affine_mat_before = torch.from_numpy(affine_mat_before).permute(2,0,1).float()
                    if self.use_cuda:
                        flow_grid_before = F.affine_grid(Variable(affine_mat_before, requires_grad=False).cuda(), input_depth_data.size(), align_corners=True)                        
                    # Rotate images clockwise
                    if self.use_cuda:
                        with torch.no_grad():
                           rotate_depth = F.grid_sample(Variable(input_depth_data, requires_grad=False).cuda(), flow_grid_before, mode='nearest', align_corners=True)                       
                    # Compute intermediate features
                    interm_suction_depth_feat = self.suction_depth_trunk.features(rotate_depth)
                    m_interm_suction_depth_feat = self.suction_depth_trunk.features(m_input_depth_data.cuda())                                        
                    interm_suction_feat = torch.cat((interm_suction_depth_feat, m_interm_suction_depth_feat), dim=1)                 
                    suction_Q = self.suctionnet_val(interm_suction_feat)                       
                    suc_prob = suction_Q
                
                    return suc_prob
            
                elif style == 2:
                    # Apply rotations to images
                    rotate_idx = 0
                    rotate_theta = np.radians(rotate_idx*(360/self.gnum_rotations))    
                    # Compute sample grid for rotation BEFORE neural network
                    affine_mat_before = np.asarray([[np.cos(-rotate_theta), np.sin(-rotate_theta), 0],[-np.sin(-rotate_theta), np.cos(-rotate_theta), 0]])
                    affine_mat_before.shape = (2,3,1)
                    affine_mat_before = torch.from_numpy(affine_mat_before).permute(2,0,1).float()
                    if self.use_cuda:
                        flow_grid_before = F.affine_grid(Variable(affine_mat_before, requires_grad=False).cuda(), input_depth_data.size(), align_corners=True)                    
                    # Rotate images clockwise
                    if self.use_cuda:
                        with torch.no_grad():
                            rotate_depth = F.grid_sample(Variable(input_depth_data).cuda(), flow_grid_before, mode='nearest', align_corners=True)                    
                    # Compute intermediate features
                    interm_gs_depth_feat = self.gs_depth_trunk.features(rotate_depth)
                    m_interm_gs_depth_feat = self.gs_depth_trunk.features(m_input_depth_data.cuda())                   
                    interm_gs_feat = torch.cat((interm_gs_depth_feat, m_interm_gs_depth_feat), dim=1)                       
                    gs_Q = self.suctionnet_val(interm_gs_feat)
                    gs_prob = gs_Q
   
                    return gs_prob
            
        
        else:
            self.gra_prob = []
            self.suc_prob = []
            self.gs_prob = []
            
            if style == 0:                       
                # Apply rotations to intermediate features
                # for rotate_idx in range(self.num_rotations):
                rotate_idx = specific_rotation
                rotate_theta = np.radians(rotate_idx*(360/self.gnum_rotations))    
                # Compute sample grid for rotation BEFORE branches
                affine_mat_before = np.asarray([[np.cos(-rotate_theta), np.sin(-rotate_theta), 0],[-np.sin(-rotate_theta), np.cos(-rotate_theta), 0]])
                affine_mat_before.shape = (2,3,1)
                affine_mat_before = torch.from_numpy(affine_mat_before).permute(2,0,1).float()
                if self.use_cuda:
                    flow_grid_before = F.affine_grid(Variable(affine_mat_before, requires_grad=False).cuda(), input_depth_data.size(), align_corners=True)                
                # Rotate images clockwise
                if self.use_cuda:
                    rotate_depth = F.grid_sample(Variable(input_depth_data, requires_grad=False).cuda(), flow_grid_before, mode='nearest', align_corners=True)                
                # Compute intermediate features                
                interm_grasp_depth_feat = self.grasp_depth_trunk.features(rotate_depth)
                m_interm_grasp_depth_feat = self.grasp_depth_trunk.features(m_input_depth_data.cuda())                
                interm_grasp_feat = torch.cat((interm_grasp_depth_feat, m_interm_grasp_depth_feat), dim=1)                               
                grasp_Q = self.graspnet_val(interm_grasp_feat)                        
                self.gra_prob = grasp_Q    
               
                return self.gra_prob
            
            elif style == 1:
                # Apply rotations to intermediate features
                # for rotate_idx in range(self.num_rotations):
                rotate_idx = specific_rotation
                rotate_theta = np.radians(rotate_idx*(360/self.gnum_rotations))  
                # Compute sample grid for rotation BEFORE branches
                affine_mat_before = np.asarray([[np.cos(-rotate_theta), np.sin(-rotate_theta), 0],[-np.sin(-rotate_theta), np.cos(-rotate_theta), 0]])
                affine_mat_before.shape = (2,3,1)
                affine_mat_before = torch.from_numpy(affine_mat_before).permute(2,0,1).float()
                if self.use_cuda:
                    flow_grid_before = F.affine_grid(Variable(affine_mat_before, requires_grad=False).cuda(), input_depth_data.size(), align_corners=True)              
                # Rotate images clockwise
                if self.use_cuda:
                    rotate_depth = F.grid_sample(Variable(input_depth_data, requires_grad=False).cuda(), flow_grid_before, mode='nearest', align_corners=True)                
                # Compute intermediate features
                interm_suction_depth_feat = self.suction_depth_trunk.features(rotate_depth)
                m_interm_suction_depth_feat = self.suction_depth_trunk.features(m_input_depth_data.cuda())                                
                interm_suction_feat = torch.cat((interm_suction_depth_feat, m_interm_suction_depth_feat), dim=1)             
                suction_Q = self.suctionnet_val(interm_suction_feat)
                self.suc_prob = suction_Q    
                
                return self.suc_prob
            
            elif style == 2:
                # Apply rotations to intermediate features
                # for rotate_idx in range(self.num_rotations):
                rotate_idx = 0
                rotate_theta = np.radians(rotate_idx*(360/self.gnum_rotations))    
                # Compute sample grid for rotation BEFORE branches
                affine_mat_before = np.asarray([[np.cos(-rotate_theta), np.sin(-rotate_theta), 0],[-np.sin(-rotate_theta), np.cos(-rotate_theta), 0]])
                affine_mat_before.shape = (2,3,1)
                affine_mat_before = torch.from_numpy(affine_mat_before).permute(2,0,1).float()
                if self.use_cuda:
                    flow_grid_before = F.affine_grid(Variable(affine_mat_before, requires_grad=False).cuda(), input_depth_data.size(), align_corners=True)                
                # Rotate images clockwise
                if self.use_cuda:
                    rotate_depth = F.grid_sample(Variable(input_depth_data, requires_grad=False).cuda(), flow_grid_before, mode='nearest', align_corners=True)               
                # Compute intermediate features            
                interm_gs_depth_feat = self.gs_depth_trunk.features(rotate_depth)
                m_interm_gs_depth_feat = self.gs_depth_trunk.features(m_input_depth_data.cuda())               
                interm_gs_feat = torch.cat((interm_gs_depth_feat, m_interm_gs_depth_feat), dim=1)                               
                gs_Q = self.suctionnet_val(interm_gs_feat)
                self.gs_prob = gs_Q          
                                
                return self.gs_prob
                



class reinforcement_net(nn.Module):

    def __init__(self, use_cuda): # , snapshot=None
        super(reinforcement_net, self).__init__()
        self.use_cuda = use_cuda

        # Initialize network trunks with DenseNet pre-trained on ImageNet
        self.suction_depth_trunk = torchvision.models.densenet.densenet121(pretrained=True)
        self.grasp_depth_trunk = torchvision.models.densenet.densenet121(pretrained=True)
        self.gs_depth_trunk = torchvision.models.densenet.densenet121(pretrained=True)

        self.gnum_rotations = 1
        self.snum_rotations = 1

        # dueling network architecture for suction
        self.suctionnet_val = nn.Sequential(OrderedDict([
            ('suction-val-norm0', nn.BatchNorm2d(2048)),
            ('suction-val-relu0', nn.ReLU(inplace=True)),
            ('suction-val-conv0', nn.Conv2d(2048, 64, kernel_size=1, stride=1, bias=False)),
            ('suction-val-norm1', nn.BatchNorm2d(64)),
            ('suction-val-relu1', nn.ReLU(inplace=True)),
            ('suction-val-conv1', nn.Conv2d(64, 1, kernel_size=20, stride=1, bias=False))
        ]))                              
        
        # dueling network architecture for grasping
        self.graspnet_val = nn.Sequential(OrderedDict([
            ('grasp-val-norm0', nn.BatchNorm2d(2048)),
            ('grasp-val-relu0', nn.ReLU(inplace=True)),
            ('grasp-val-conv0', nn.Conv2d(2048, 64, kernel_size=1, stride=1, bias=False)),
            ('grasp-val-norm1', nn.BatchNorm2d(64)),
            ('grasp-val-relu1', nn.ReLU(inplace=True)),
            ('grasp-val-conv1', nn.Conv2d(64, 1, kernel_size=20, stride=1, bias=False))
        ]))
                               
        # network architecture for grasping_then_suction
        self.gsnet_val = nn.Sequential(OrderedDict([
            ('grasp-val-norm0', nn.BatchNorm2d(2048)),
            ('grasp-val-relu0', nn.ReLU(inplace=True)),
            ('grasp-val-conv0', nn.Conv2d(2048, 64, kernel_size=1, stride=1, bias=False)),
            ('grasp-val-norm1', nn.BatchNorm2d(64)),
            ('grasp-val-relu1', nn.ReLU(inplace=True)),
            ('grasp-val-conv1', nn.Conv2d(64, 1, kernel_size=20, stride=1, bias=False))
        ]))
                       
            
        # Initialize network weights
        for m in self.named_modules():
            if 'suction-' in m[0] or 'grasp-' in m[0] or 'gs-' in m[0]:
                if isinstance(m[1], nn.Conv2d):
                    nn.init.kaiming_normal_(m[1].weight.data)
                elif isinstance(m[1], nn.BatchNorm2d):
                    m[1].weight.data.fill_(1)
                    m[1].bias.data.zero_()

        # Initialize output variable (for backprop)
        self.gra_prob = []
        self.suc_prob = []
        self.gs_prob = []


    def forward(self, input_depth_data, m_input_depth_data, style=0, is_volatile=False, specific_rotation=-1):

        if is_volatile and specific_rotation == -1:
            with torch.no_grad():
                gra_prob = []
                suc_prob = []
                gs_prob = []

                if style == 0:
                    # Apply rotations to images
                    for rotate_idx in range(self.gnum_rotations):
                        rotate_theta = np.radians(rotate_idx*(360/self.gnum_rotations))    
                        # Compute sample grid for rotation BEFORE neural network
                        affine_mat_before = np.asarray([[np.cos(-rotate_theta), np.sin(-rotate_theta), 0],[-np.sin(-rotate_theta), np.cos(-rotate_theta), 0]])
                        affine_mat_before.shape = (2,3,1)
                        affine_mat_before = torch.from_numpy(affine_mat_before).permute(2,0,1).float()
                        if self.use_cuda:
                            flow_grid_before = F.affine_grid(Variable(affine_mat_before, requires_grad=False).cuda(), input_depth_data.size(), align_corners=True)                       
                        # Rotate images clockwise
                        if self.use_cuda:
                            with torch.no_grad():
                                rotate_depth = F.grid_sample(Variable(input_depth_data).cuda(), flow_grid_before, mode='nearest', align_corners=True)                       
                        # Compute intermediate features
                        interm_grasp_depth_feat = self.grasp_depth_trunk.features(rotate_depth)
                        m_interm_grasp_depth_feat = self.grasp_depth_trunk.features(m_input_depth_data.cuda())
                        interm_grasp_feat = torch.cat((interm_grasp_depth_feat, m_interm_grasp_depth_feat), dim=1)                            
                        grasp_Q = self.graspnet_val(interm_grasp_feat)                                    
                        gra_prob.append(grasp_Q)
                                            
                    return gra_prob

                elif style == 1:
                    # Apply rotations to images
                    for rotate_idx in range(self.snum_rotations):
                        rotate_theta = np.radians(rotate_idx*(360/self.snum_rotations))    
                        # Compute sample grid for rotation BEFORE neural network
                        affine_mat_before = np.asarray([[np.cos(-rotate_theta), np.sin(-rotate_theta), 0],[-np.sin(-rotate_theta), np.cos(-rotate_theta), 0]])
                        affine_mat_before.shape = (2,3,1)
                        affine_mat_before = torch.from_numpy(affine_mat_before).permute(2,0,1).float()
                        if self.use_cuda:
                            flow_grid_before = F.affine_grid(Variable(affine_mat_before, requires_grad=False).cuda(), input_depth_data.size(), align_corners=True)                        
                        # Rotate images clockwise
                        if self.use_cuda:
                            with torch.no_grad():
                                rotate_depth = F.grid_sample(Variable(input_depth_data).cuda(), flow_grid_before, mode='nearest', align_corners=True)                       
                        # Compute intermediate features
                        interm_suction_depth_feat = self.suction_depth_trunk.features(rotate_depth)
                        m_interm_suction_depth_feat = self.suction_depth_trunk.features(m_input_depth_data.cuda())                        
                        interm_suction_feat = torch.cat((interm_suction_depth_feat, m_interm_suction_depth_feat), dim=1)                          
                        suction_Q = self.suctionnet_val(interm_suction_feat)
                        suc_prob.append(suction_Q)
                 
                    
                    return suc_prob                 
                
                elif style == 2:
                    # Apply rotations to images
                    rotate_idx = 0
                    rotate_theta = np.radians(rotate_idx*(360/self.gnum_rotations))    
                    # Compute sample grid for rotation BEFORE neural network
                    affine_mat_before = np.asarray([[np.cos(-rotate_theta), np.sin(-rotate_theta), 0],[-np.sin(-rotate_theta), np.cos(-rotate_theta), 0]])
                    affine_mat_before.shape = (2,3,1)
                    affine_mat_before = torch.from_numpy(affine_mat_before).permute(2,0,1).float()
                    if self.use_cuda:
                        flow_grid_before = F.affine_grid(Variable(affine_mat_before, requires_grad=False).cuda(), input_depth_data.size(), align_corners=True)                      
                    # Rotate images clockwise
                    if self.use_cuda:
                        with torch.no_grad():
                            rotate_depth = F.grid_sample(Variable(input_depth_data).cuda(), flow_grid_before, mode='nearest', align_corners=True)                   
                    # Compute intermediate features
                    interm_gs_depth_feat = self.gs_depth_trunk.features(rotate_depth)
                    m_interm_gs_depth_feat = self.gs_depth_trunk.features(m_input_depth_data.cuda())                  
                    interm_gs_feat = torch.cat((interm_gs_depth_feat, m_interm_gs_depth_feat), dim=1)                     
                    gs_Q = self.suctionnet_val(interm_gs_feat)
                    gs_prob.append(gs_Q)                    

                    return gs_prob                                                                

        elif is_volatile and specific_rotation != -1:
            with torch.no_grad():
                gra_prob = []
                suc_prob = []
                gs_prob = []

                if style == 0:
                    rotate_idx = specific_rotation
                    rotate_theta = np.radians(rotate_idx*(360/self.gnum_rotations))                    
                    # Compute sample grid for rotation BEFORE branches
                    affine_mat_before = np.asarray([[np.cos(-rotate_theta), np.sin(-rotate_theta), 0],[-np.sin(-rotate_theta), np.cos(-rotate_theta), 0]])
                    affine_mat_before.shape = (2,3,1)
                    affine_mat_before = torch.from_numpy(affine_mat_before).permute(2,0,1).float()
                    if self.use_cuda:
                        flow_grid_before = F.affine_grid(Variable(affine_mat_before, requires_grad=False).cuda(), input_depth_data.size(), align_corners=True)                    
                    # Rotate images clockwise
                    if self.use_cuda:
                        with torch.no_grad():
                            rotate_depth = F.grid_sample(Variable(input_depth_data, requires_grad=False).cuda(), flow_grid_before, mode='nearest', align_corners=True)                    
                    # Compute intermediate features
                    interm_grasp_depth_feat = self.grasp_depth_trunk.features(rotate_depth)
                    m_interm_grasp_depth_feat = self.grasp_depth_trunk.features(m_input_depth_data.cuda())                    
                    interm_grasp_feat = torch.cat((interm_grasp_depth_feat, m_interm_grasp_depth_feat), dim=1)                    
                    grasp_Q = self.graspnet_val(interm_grasp_feat)                        
                    gra_prob = grasp_Q    
                
                    return gra_prob
            
                elif style == 1:
                    rotate_idx = specific_rotation
                    rotate_theta = np.radians(rotate_idx*(360/self.gnum_rotations))                        
                    # Compute sample grid for rotation BEFORE branches
                    affine_mat_before = np.asarray([[np.cos(-rotate_theta), np.sin(-rotate_theta), 0],[-np.sin(-rotate_theta), np.cos(-rotate_theta), 0]])
                    affine_mat_before.shape = (2,3,1)
                    affine_mat_before = torch.from_numpy(affine_mat_before).permute(2,0,1).float()
                    if self.use_cuda:
                        flow_grid_before = F.affine_grid(Variable(affine_mat_before, requires_grad=False).cuda(), input_depth_data.size(), align_corners=True)                        
                    # Rotate images clockwise
                    if self.use_cuda:
                        with torch.no_grad():
                            rotate_depth = F.grid_sample(Variable(input_depth_data, requires_grad=False).cuda(), flow_grid_before, mode='nearest', align_corners=True)                       
                    # Compute intermediate features
                    interm_suction_depth_feat = self.suction_depth_trunk.features(rotate_depth)
                    m_interm_suction_depth_feat = self.suction_depth_trunk.features(m_input_depth_data.cuda())                                        
                    interm_suction_feat = torch.cat((interm_suction_depth_feat, m_interm_suction_depth_feat), dim=1)                  
                    suction_Q = self.suctionnet_val(interm_suction_feat)                       
                    suc_prob = suction_Q
                
                    return suc_prob
            
                elif style == 2:
                    # Apply rotations to images
                    rotate_idx = 0
                    rotate_theta = np.radians(rotate_idx*(360/self.gnum_rotations))    
                    # Compute sample grid for rotation BEFORE neural network
                    affine_mat_before = np.asarray([[np.cos(-rotate_theta), np.sin(-rotate_theta), 0],[-np.sin(-rotate_theta), np.cos(-rotate_theta), 0]])
                    affine_mat_before.shape = (2,3,1)
                    affine_mat_before = torch.from_numpy(affine_mat_before).permute(2,0,1).float()
                    if self.use_cuda:
                        flow_grid_before = F.affine_grid(Variable(affine_mat_before, requires_grad=False).cuda(), input_depth_data.size(), align_corners=True)                    
                    # Rotate images clockwise
                    if self.use_cuda:
                        with torch.no_grad():
                            rotate_depth = F.grid_sample(Variable(input_depth_data).cuda(), flow_grid_before, mode='nearest', align_corners=True)                    
                    # Compute intermediate features
                    interm_gs_depth_feat = self.gs_depth_trunk.features(rotate_depth)
                    m_interm_gs_depth_feat = self.gs_depth_trunk.features(m_input_depth_data.cuda())                   
                    interm_gs_feat = torch.cat((interm_gs_depth_feat, m_interm_gs_depth_feat), dim=1)                       
                    gs_Q = self.suctionnet_val(interm_gs_feat)
                    gs_prob = gs_Q
   
                    return gs_prob
            
        
        else:
            self.gra_prob = []
            self.suc_prob = []
            self.gs_prob = []
            
            if style == 0:                       
                # Apply rotations to intermediate features
                # for rotate_idx in range(self.num_rotations):
                rotate_idx = specific_rotation
                rotate_theta = np.radians(rotate_idx*(360/self.gnum_rotations))    
                # Compute sample grid for rotation BEFORE branches
                affine_mat_before = np.asarray([[np.cos(-rotate_theta), np.sin(-rotate_theta), 0],[-np.sin(-rotate_theta), np.cos(-rotate_theta), 0]])
                affine_mat_before.shape = (2,3,1)
                affine_mat_before = torch.from_numpy(affine_mat_before).permute(2,0,1).float()
                if self.use_cuda:
                    flow_grid_before = F.affine_grid(Variable(affine_mat_before, requires_grad=False).cuda(), input_depth_data.size(), align_corners=True)                
                # Rotate images clockwise
                if self.use_cuda:
                    rotate_depth = F.grid_sample(Variable(input_depth_data, requires_grad=False).cuda(), flow_grid_before, mode='nearest', align_corners=True)                
                # Compute intermediate features
                interm_grasp_depth_feat = self.grasp_depth_trunk.features(rotate_depth)
                m_interm_grasp_depth_feat = self.grasp_depth_trunk.features(m_input_depth_data.cuda())                
                interm_grasp_feat = torch.cat((interm_grasp_depth_feat, m_interm_grasp_depth_feat), dim=1)                               
                grasp_Q = self.graspnet_val(interm_grasp_feat)                        
                self.gra_prob = grasp_Q    
               
                return self.gra_prob
            
            elif style == 1:
                # Apply rotations to intermediate features
                # for rotate_idx in range(self.num_rotations):
                rotate_idx = specific_rotation
                rotate_theta = np.radians(rotate_idx*(360/self.gnum_rotations))  
                # Compute sample grid for rotation BEFORE branches
                affine_mat_before = np.asarray([[np.cos(-rotate_theta), np.sin(-rotate_theta), 0],[-np.sin(-rotate_theta), np.cos(-rotate_theta), 0]])
                affine_mat_before.shape = (2,3,1)
                affine_mat_before = torch.from_numpy(affine_mat_before).permute(2,0,1).float()
                if self.use_cuda:
                    flow_grid_before = F.affine_grid(Variable(affine_mat_before, requires_grad=False).cuda(), input_depth_data.size(), align_corners=True)              
                # Rotate images clockwise
                if self.use_cuda:
                    rotate_depth = F.grid_sample(Variable(input_depth_data, requires_grad=False).cuda(), flow_grid_before, mode='nearest', align_corners=True)                
                # Compute intermediate features
                interm_suction_depth_feat = self.suction_depth_trunk.features(rotate_depth)
                m_interm_suction_depth_feat = self.suction_depth_trunk.features(m_input_depth_data.cuda())                                
                interm_suction_feat = torch.cat((interm_suction_depth_feat, m_interm_suction_depth_feat), dim=1)              
                suction_Q = self.suctionnet_val(interm_suction_feat)
                self.suc_prob = suction_Q    
                
                return self.suc_prob
            
            elif style == 2:
                # Apply rotations to intermediate features
                # for rotate_idx in range(self.num_rotations):
                rotate_idx = 0
                rotate_theta = np.radians(rotate_idx*(360/self.gnum_rotations))    
                # Compute sample grid for rotation BEFORE branches
                affine_mat_before = np.asarray([[np.cos(-rotate_theta), np.sin(-rotate_theta), 0],[-np.sin(-rotate_theta), np.cos(-rotate_theta), 0]])
                affine_mat_before.shape = (2,3,1)
                affine_mat_before = torch.from_numpy(affine_mat_before).permute(2,0,1).float()
                if self.use_cuda:
                    flow_grid_before = F.affine_grid(Variable(affine_mat_before, requires_grad=False).cuda(), input_depth_data.size(), align_corners=True)                
                # Rotate images clockwise
                if self.use_cuda:
                    rotate_depth = F.grid_sample(Variable(input_depth_data, requires_grad=False).cuda(), flow_grid_before, mode='nearest', align_corners=True)               
                # Compute intermediate features
                interm_gs_depth_feat = self.gs_depth_trunk.features(rotate_depth)
                m_interm_gs_depth_feat = self.gs_depth_trunk.features(m_input_depth_data.cuda())               
                interm_gs_feat = torch.cat((interm_gs_depth_feat, m_interm_gs_depth_feat), dim=1)                               
                gs_Q = self.suctionnet_val(interm_gs_feat)
                self.gs_prob = gs_Q          
                
                
                return self.gs_prob
                
