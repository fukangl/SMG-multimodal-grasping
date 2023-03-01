#!/usr/bin/env python

import math
import time
import os
import random
import threading
import argparse
import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
import cv2
from collections import namedtuple
import torch
from torch.autograd import Variable
from robot import Robot
from trainer import Trainer
from logger import Logger
import utils
import masks
from scipy import ndimage


def main(args):


    # --------------- Setup options ---------------
    is_sim = args.is_sim # Run in simulation?
    obj_mesh_dir = os.path.abspath(args.obj_mesh_dir) if is_sim else None # Directory containing 3D mesh files (.obj) of objects to be added to simulation
    num_obj = args.num_obj if is_sim else None # Number of objects to add to simulation
    workspace_limits = np.asarray([[-0.724, -0.276], [-0.224, 0.224], [-0.0001, 0.4]]) # Cols: min max, Rows: x y z (define workspace limits in robot coordinates)
    heightmap_resolution = args.heightmap_resolution # Meters per pixel of heightmap
    force_cpu = args.force_cpu
    is_cluttered = args.is_cluttered # lightly-cluttered or highly-cluttered

    # ------------- Algorithm options -------------
    method = args.method # dueling DQN or Reactive
    is_ets = args.is_ets # run with Enveloping_then_sucking action?
    is_pe = args.is_pe # run with orientation preenveloping?
    is_oo = args.is_oo # run with orientation optimization?
    future_reward_discount = args.future_reward_discount
    explore_rate_decay = args.explore_rate_decay
    
    # -------------- Training or Testing options --------------
    is_testing = args.is_testing
    training_episode = args.training_episode
    testing_episode = args.testing_episode
    step_lm = args.step
    target_update_freq = args.target_update_freq # frequency to update target Q network


    # ------ Pre-loading and logging options ------
    load_snapshot = args.load_snapshot # Load pre-trained snapshot of model?
    snapshot_file = os.path.abspath(args.snapshot_file)  if load_snapshot else None
    continue_logging = args.continue_logging # Continue logging from previous session
    logging_directory = os.path.abspath(args.logging_directory) if continue_logging else os.path.abspath('logs')
    

    # Initialize pick-and-place system (camera and robot)
    robot = Robot(is_sim, obj_mesh_dir, num_obj, workspace_limits,
                  is_cluttered,is_testing)

    # Initialize trainer
    trainer = Trainer(method, future_reward_discount,
                      load_snapshot, snapshot_file, force_cpu)

    # Initialize data logger
    logger = Logger(continue_logging, logging_directory)
    logger.save_camera_info(robot.cam_intrinsics, robot.cam_pose, robot.cam_depth_scale) # Save camera intrinsics and pose
    logger.save_heightmap_info(workspace_limits, heightmap_resolution) # Save heightmap parameters

    # Find last executed iteration of pre-loaded log, and load execution info and RL variables
    if continue_logging:
        trainer.preload(logger.transitions_directory)

    # Initialize variables for heuristic bootstrapping and exploration probability
    no_change_count = [0,0] if not is_testing else [0,0]
    explore_prob = 0.5 if not is_testing else 0.0

    # Quick hack for nonlocal memory between threads in Python 2
    nonlocal_variables = {'primitive_action' : None,
                          'best_pix_ind' : None,
                          'suction_success' : 0,
                          'grasp_success' : 0,
                          'gs_success' :0} 

    getvalue_variables = {'primitive_action' : None,
                          'best_pix_ind' : None,
                          'best_gs_conf' : 0}
        

    for episode in range(testing_episode if is_testing else training_episode):
        print('\n%s iteration: %d' % ('Testing' if is_testing else 'Training', trainer.iteration))
        episode_iter = 0
        episode_succ = 0
        # initialize the simulation or real-world experiment
        if is_sim: 
            robot.check_sim()
            robot.restart_sim()
            robot.add_objects()
        else:
            robot.restart_real()    
        if is_testing: # re-load original weights (before test run)
            trainer.model.load_state_dict(torch.load(snapshot_file))        
        # Train or Test
        for step in range(step_lm):                        
            # Get latest RGB-D image
            color_img, depth_img = robot.get_camera_data()            
            depth_img = depth_img * robot.cam_depth_scale # Apply depth scale from calibration    
            # Get heightmap from RGB-D image (by re-projecting 3D point cloud)
            color_heightmap, depth_heightmap, color_448, depth_448, A_htor = utils.get_heightmap(color_img, depth_img, robot.cam_intrinsics, robot.cam_pose, workspace_limits, heightmap_resolution)            
            valid_depth_heightmap = depth_heightmap.copy()
            valid_depth_heightmap[np.isnan(valid_depth_heightmap)] = 0                                                           
            # Save RGB-D images and RGB-D heightmaps
            logger.save_images(trainer.iteration, color_img, depth_img, '0')
            #logger.save_heightmaps(trainer.iteration, color_heightmap, valid_depth_heightmap, '0')            
            objects_mask_448, objects_mask, objects_number, object_boxes, masks_cter, box_mask_cors = masks.instance_segmentation(color_448,trainer.iteration)
            # Reset simulation or pause real-world experiment if table is empty or IK problem
            tip_position, target_position = robot.get_t_position(ttt=True)
            tt_distance = ((tip_position[0]-target_position[0])**2 + (tip_position[1]-target_position[1])**2 + (tip_position[2]-target_position[2])**2)            
            if objects_number == 0 or (is_sim and (no_change_count[0] + no_change_count[1]) > 3 and  prev_suction_success ==0 and  prev_grasp_success ==0 and  prev_gs_success ==0) or tt_distance > 0.01 or episode_succ >= 10:
                no_change_count = [0,0]
                trainer.clearance_log.append([trainer.iteration])
                logger.write_to_log('clearance', trainer.clearance_log)               
                trainer.episode_success_log.append([episode, episode_iter, episode_succ]) # 1 - grasp
                logger.write_to_log('episode_success', trainer.episode_success_log)               
                if is_sim:
                    if tt_distance > 0.01:
                        print('IK problem.')
                    else:
                        print('Not enough objects in view! Repositioning objects.')
                    break
                else:
                    print('Not enough stuff on the table! Flipping over bin of objects.')
                    break
                
            mask_depth = objects_mask.copy()
            mask_depth_448 = objects_mask_448.copy()            
            objects_mask.shape = (objects_mask.shape[0], objects_mask.shape[1], objects_mask.shape[2], 1)
            gro_num, sro_num = 1, 1
            gra_conf, suc_conf = np.zeros((objects_number,gro_num)), np.zeros((objects_number,sro_num))                         
            gs_conf = np.zeros((objects_number,objects_number))             
            gnu_best, snu_best = np.zeros(objects_number), np.zeros(objects_number)
            gro_best, sro_best = np.zeros(objects_number), np.zeros(objects_number)
            mask_depth_a = mask_depth[0]
            mask_depth_a_448 = mask_depth_448[0]
            if objects_number > 1:
                for a_num in range(1,objects_number):
                    mask_depth_a = mask_depth_a + mask_depth[a_num]
                    mask_depth_a_448 = mask_depth_a_448 + mask_depth_448[a_num]
            valid_depth_heightmap_a = valid_depth_heightmap * mask_depth_a
            depth_a_448 = depth_448 * mask_depth_a_448                        
            valid_depth_heightmap_a_s = cv2.applyColorMap(cv2.convertScaleAbs(valid_depth_heightmap_a,alpha=255/ratio), color_num)         
            depth_a_s_448 = cv2.applyColorMap(cv2.convertScaleAbs(depth_a_448,alpha=255/ratio), color_num)
            logger.save_heightmaps(trainer.iteration, color_heightmap, valid_depth_heightmap_a_s, '0')
            logger.save_heightmaps_448(trainer.iteration, color_448, depth_a_s_448, '0')
            
            for num in range(objects_number):
                c_heightmap_mask = color_heightmap * objects_mask[num]
                d_heightmap_mask = valid_depth_heightmap_a * mask_depth[num]
                d_heightmap_mask_448 = depth_a_448 * mask_depth_448[num]                
                # Save RGB-D masks
                d_heightmap_mask_s = cv2.applyColorMap(cv2.convertScaleAbs(d_heightmap_mask_448,alpha=255/ratio), color_num)
                # logger.save_masks(trainer.iteration, num, num, c_heightmap_mask, d_heightmap_mask_s, '0')                                                                                
                gra_conf[num] = trainer.forward(valid_depth_heightmap_a, d_heightmap_mask, style = 0, is_volatile=True, is_target=False)
                suc_conf[num] = trainer.forward(valid_depth_heightmap_a, d_heightmap_mask, style = 1, is_volatile=True, is_target=False)                 
                gnu_best[num], snu_best[num] = np.max(gra_conf[num]), np.max(suc_conf[num]) 
                gro_best[num], sro_best[num] = np.unravel_index(np.argmax(gra_conf[num]), gra_conf[num].shape)[0], np.unravel_index(np.argmax(suc_conf[num]), suc_conf[num].shape)[0]

            bestg_conf, bests_conf = np.max(gra_conf), np.max(suc_conf)
            gro_best, sro_best = gro_best.astype(int), sro_best.astype(int)
            bestg_id = np.unravel_index(np.argmax(gra_conf), gra_conf.shape)
            bests_id = np.unravel_index(np.argmax(suc_conf), suc_conf.shape)                        
            bestg_pix = [bestg_id[1],masks_cter[bestg_id[0],1],masks_cter[bestg_id[0],0],0,0,0]
            bests_pix = [0,0,0,bests_id[1],masks_cter[bests_id[0],1],masks_cter[bests_id[0],0]]
            bestg_pix = np.array(bestg_pix).astype(int)
            bests_pix = np.array(bests_pix).astype(int)                                                
            bestgs_conf = 0
            bestgs_g_id,bestgs_s_id,bestgs_num = [],[],[]
            if is_ets:
                if objects_number > 1:
                    gs_conf[:,:] = -100.                
                    for g_num in range(objects_number):
                        for s_num in range(g_num+1, objects_number):
                            if s_num != g_num:
                                c_heightmap_mask = color_heightmap * (objects_mask[g_num] + objects_mask[s_num])
                                d_heightmap_mask = valid_depth_heightmap_a * (mask_depth[g_num] + mask_depth[s_num])                            
                                d_heightmap_mask_448 = depth_a_448 * (mask_depth_448[g_num] + mask_depth_448[s_num])
                                # Save RGB-D masks
                                d_heightmap_mask_s = cv2.applyColorMap(cv2.convertScaleAbs(d_heightmap_mask_448,alpha=255/ratio), color_num)
                                # logger.save_masks(trainer.iteration, g_num, s_num, c_heightmap_mask, d_heightmap_mask_s, '1')                                                        
                                gs_conf[g_num,s_num] = trainer.forward(valid_depth_heightmap_a, d_heightmap_mask, style = 2, is_volatile=True, is_target=False)                              

                    bestgs_conf = np.max(gs_conf)
                    bestgs_num = np.unravel_index(np.argmax(gs_conf), gs_conf.shape)
                    if gnu_best[bestgs_num[0]] > gnu_best[bestgs_num[1]]:
                        bestgs_g_id = [bestgs_num[0],gro_best[bestgs_num[0]]]
                        bestgs_s_id = [bestgs_num[1],sro_best[bestgs_num[1]]]
                    else:
                        bestgs_g_id = [bestgs_num[1],gro_best[bestgs_num[1]]]
                        bestgs_s_id = [bestgs_num[0],sro_best[bestgs_num[0]]]                                                                                        
                    bestgs_pix = [bestgs_g_id[1], masks_cter[bestgs_g_id[0],1], masks_cter[bestgs_g_id[0],0], bestgs_s_id[1], masks_cter[bestgs_s_id[0],1], masks_cter[bestgs_s_id[0],0]]
                    bestgs_pix = np.array(bestgs_pix).astype(int)
            
            nonlocal_variables['primitive_action'] = 'grasp'
            getvalue_variables['primitive_action'] = 'grasp'            
            if not is_ets or objects_number == 1:     
                if bests_conf > bestg_conf:
                    nonlocal_variables['primitive_action'] = 'suction'
                    getvalue_variables['primitive_action'] = 'suction'                    
                if not is_testing:
                    explore_actions = np.random.uniform() < explore_prob
                    if explore_actions: # Exploitation (do best action) vs exploration (do other action)
                        print('Strategy: explore (exploration probability: %f)' % (explore_prob))
                        nonlocal_variables['primitive_action'] = 'suction' if np.random.randint(0,2) == 0 else 'grasp'
                    else:
                        print('Strategy: exploit (exploration probability: %f)' % (explore_prob))                
         
            elif is_ets and objects_number > 1:# and trainer.iteration >= 200:
                if method == 'reactive':
                    if bests_conf > max(bestg_conf,2*bestgs_conf):
                        nonlocal_variables['primitive_action'] = 'suction'
                        getvalue_variables['primitive_action'] = 'suction'
                    elif 2*bestgs_conf > max(bests_conf,bestg_conf):
                        nonlocal_variables['primitive_action'] = 'grasp_then_suction'
                        getvalue_variables['primitive_action'] = 'grasp_then_suction'
                else:
                    if bests_conf > max(bestg_conf,bestgs_conf):
                        nonlocal_variables['primitive_action'] = 'suction'
                        getvalue_variables['primitive_action'] = 'suction'                                                                    
                    elif bestgs_conf > max(bests_conf,bestg_conf):
                        nonlocal_variables['primitive_action'] = 'grasp_then_suction'
                        getvalue_variables['primitive_action'] = 'grasp_then_suction'                    
                if not is_testing:
                    explore_actions = np.random.uniform() < explore_prob
                    if explore_actions: # Exploitation (do best action) vs exploration (do other action)
                        print('Strategy: explore (exploration probability: %f)' % (explore_prob))
                        action_random = np.random.randint(0,3)
                        nonlocal_variables['primitive_action'] = 'suction' if action_random == 0 else 'grasp' if  action_random == 1 else 'grasp_then_suction'                        
                    else:
                        print('Strategy: exploit (exploration probability: %f)' % (explore_prob))
            trainer.is_exploit_log.append([0 if explore_actions else 1])
            logger.write_to_log('is-exploit', trainer.is_exploit_log)
                   
            if nonlocal_variables['primitive_action'] == 'grasp':
                nonlocal_variables['best_pix_ind'] = bestg_pix 
                predicted_value = bestg_conf
                box_mask_cors_448 = (box_mask_cors*2).astype(int)
                object_boxes_448 = (object_boxes*2).astype(int)
                poly = np.array(box_mask_cors_448[bestg_id[0]],np.int32).reshape(-1,1,2)
                label_mask = color_448.copy()               
                logger.save_action_masks(trainer.iteration,label_mask,'0')               
                primitive_grasp_position, grasp_rotation_angle, grasp_open_distance = utils.get_best_grasp_angle(is_pe, box_mask_cors, bestg_id, A_htor, robot.cam_intrinsics, robot.cam_pose, depth_img)                                                               
                
                
            elif nonlocal_variables['primitive_action'] == 'suction':
                nonlocal_variables['best_pix_ind'] = bests_pix
                predicted_value = bests_conf                                                                                 
                box_mask_cors_448 = (box_mask_cors*2).astype(int)
                object_boxes_448 = (object_boxes*2).astype(int)
                poly = np.array(box_mask_cors_448[bests_id[0]],np.int32).reshape(-1,1,2)
                label_mask = color_448.copy()
                logger.save_action_masks(trainer.iteration,label_mask,'0') 
                primitive_suction_position, suction_rotation_angle = utils.get_best_suction_angle(is_oo, objects_number, masks_cter, box_mask_cors, bests_id, A_htor, robot.cam_intrinsics, robot.cam_pose, depth_img)                                            
                           
            elif nonlocal_variables['primitive_action'] == 'grasp_then_suction':
                nonlocal_variables['best_pix_ind'] = bestgs_pix                
                predicted_value = bestgs_conf
                box_mask_cors_448 = (box_mask_cors*2).astype(int)
                object_boxes_448 = (object_boxes*2).astype(int)
                poly_g = np.array(box_mask_cors_448[bestgs_g_id[0]],np.int32).reshape(-1,1,2)
                poly_s = np.array(box_mask_cors_448[bestgs_s_id[0]],np.int32).reshape(-1,1,2)
                label_mask = color_448.copy()
                logger.save_action_masks(trainer.iteration,label_mask,'0')
                primitive_grasp_position, grasp_rotation_angle, grasp_open_distance = utils.get_best_grasp_angle(is_pe, box_mask_cors, bestgs_g_id, A_htor, robot.cam_intrinsics, robot.cam_pose, depth_img)
                                
                # predicted suction angle after a successful grasping
                if objects_number > 2:
                    object_boxes_1,masks_cter_1,box_mask_cors_1 = [],[],[]
                    for i in range(objects_number):
                        if i != bestgs_g_id[0]:
                            object_boxes_1.append(object_boxes[i])
                            masks_cter_1.append(masks_cter[i])
                            box_mask_cors_1.append(box_mask_cors[i])                    
                    object_boxes_1 = np.asarray(object_boxes_1)
                    masks_cter_1 = np.asarray(masks_cter_1)
                    box_mask_cors_1 = np.asarray(box_mask_cors_1)
                    if bestgs_g_id[0] < bestgs_s_id[0]:
                        bestgs_s_id[0] -=1
                    primitive_suction_position, suction_rotation_angle_1 = utils.get_best_suction_angle(is_oo, objects_number-1, masks_cter_1, box_mask_cors_1, bestgs_s_id, A_htor, robot.cam_intrinsics, robot.cam_pose, depth_img)
                else:   
                    primitive_suction_position, suction_rotation_angle_1 = utils.get_best_suction_angle(is_oo, objects_number, masks_cter, box_mask_cors, bestgs_s_id, A_htor, robot.cam_intrinsics, robot.cam_pose, depth_img)
                if objects_number == 2:
                    suction_rotation_angle_1 = 0
              
            trainer.predicted_value_log.append(predicted_value)
            logger.write_to_log('predicted-value', trainer.predicted_value_log)
            print('predicted_value: %r' % (predicted_value))
            print('primitive_action: %r' % (nonlocal_variables['primitive_action']))

            # Run training iteration
            if 'prev_color_img' in locals():    
                # Detect suction or grasp success    
                if prev_primitive_action == 'suction':
                    if prev_suction_success !=0:
                        no_change_count[1] = 0
                    else:
                        no_change_count[1] += 1                        
                elif prev_primitive_action == 'grasp' or prev_primitive_action == 'grasp_then_suction':
                    if prev_grasp_success !=0 or prev_gs_success !=0:
                        no_change_count[0] = 0
                    else:
                        no_change_count[0] += 1
    
                if not is_testing:
                    # Compute training labels                                        
                    label_value, prev_reward_value = trainer.get_label_value(prev_primitive_action, prev_objects_number,
                                                                             prev_suction_success, prev_grasp_success, prev_gs_success, 
                                                                             valid_depth_heightmap_a,mask_depth,objects_mask,
                                                                             bestg_id,bests_id,bestgs_g_id,bestgs_s_id,
                                                                             getvalue_variables['primitive_action'], 
                                                                             bestg_conf, bests_conf, bestgs_conf)
                    trainer.label_value_log.append([label_value])
                    logger.write_to_log('label-value', trainer.label_value_log)
                    trainer.reward_value_log.append([prev_reward_value])
                    logger.write_to_log('reward-value', trainer.reward_value_log) 
                    
                    if prev_primitive_action == 'suction':
                       grasping_type = 0
                    elif prev_primitive_action == 'grasp':
                        grasping_type = 1
                    elif prev_primitive_action == 'grasp_then_suction':
                        grasping_type = 2
                    trainer.grasping_type_log.append([grasping_type])
                    logger.write_to_log('grasping_type', trainer.grasping_type_log)
                                   
                    # Backpropagate
                    loss_value = trainer.backprop(prev_valid_depth_heightmap_a, prev_primitive_action, 
                                                  prev_bestg_id, prev_bests_id, prev_bestgs_g_id, prev_bestgs_s_id, 
                                                  label_value, prev_objects_mask, prev_sro_best, prev_gro_best, prev_bestgs_num)

                    trainer.training_loss_log.append([trainer.iteration,loss_value])
                    logger.write_to_log('training_loss', trainer.training_loss_log)                 
                    # Adjust exploration probability                
                    explore_prob = max(0.5 * np.power(0.9998, trainer.iteration),0.1) if explore_rate_decay else 0.5

                # Save model snapshot
                if not is_testing:
                    logger.save_backup_model(trainer.model, method)
                    # update target Q netowrk
                    if method == 'reinforcement':
                        if trainer.iteration % target_update_freq == 0:
                            trainer.model_target.load_state_dict(trainer.model.state_dict())
                        if trainer.iteration % 50 == 0:
                            logger.save_model(trainer.iteration, trainer.model, method)
                        if trainer.use_cuda:
                            trainer.model = trainer.model.cuda()
                    elif method == 'reactive':
                        if trainer.iteration % 50 == 0:
                            logger.save_model(trainer.iteration, trainer.model, method)
                            if trainer.use_cuda:
                                trainer.model = trainer.model.cuda()
                                                               
            # Compute 3D position of pixel
            print('Action: %s at (%d, %d, %d, %d, %d, %d)' % (nonlocal_variables['primitive_action'], nonlocal_variables['best_pix_ind'][0], nonlocal_variables['best_pix_ind'][1], nonlocal_variables['best_pix_ind'][2],
                                                              nonlocal_variables['best_pix_ind'][3], nonlocal_variables['best_pix_ind'][4], nonlocal_variables['best_pix_ind'][5]))                                                      
            # Save executed primitive
            if nonlocal_variables['primitive_action'] == 'suction':
                trainer.executed_action_log.append([0, nonlocal_variables['best_pix_ind'][0], nonlocal_variables['best_pix_ind'][1], nonlocal_variables['best_pix_ind'][2],
                                                    nonlocal_variables['best_pix_ind'][3], nonlocal_variables['best_pix_ind'][4], nonlocal_variables['best_pix_ind'][5]]) # 0 - suction
            elif nonlocal_variables['primitive_action'] == 'grasp':
                trainer.executed_action_log.append([1, nonlocal_variables['best_pix_ind'][0], nonlocal_variables['best_pix_ind'][1], nonlocal_variables['best_pix_ind'][2],
                                                    nonlocal_variables['best_pix_ind'][3], nonlocal_variables['best_pix_ind'][4], nonlocal_variables['best_pix_ind'][5]]) # 1 - grasp
            elif nonlocal_variables['primitive_action'] == 'grasp_then_suction':
                trainer.executed_action_log.append([2, nonlocal_variables['best_pix_ind'][0], nonlocal_variables['best_pix_ind'][1], nonlocal_variables['best_pix_ind'][2],
                                                    nonlocal_variables['best_pix_ind'][3], nonlocal_variables['best_pix_ind'][4], nonlocal_variables['best_pix_ind'][5]]) # 2 - grasp_then_suction
            logger.write_to_log('executed-action', trainer.executed_action_log)
         
            # Initialize variables that influence reward
            nonlocal_variables['suction_success'] = 0
            nonlocal_variables['grasp_success'] = 0 
            nonlocal_variables['gs_success'] = 0
            # Execute primitive                                            
            if nonlocal_variables['primitive_action'] == 'grasp':
                nonlocal_variables['grasp_success'] = robot.grasp_first(grasp_open_distance, primitive_grasp_position, grasp_rotation_angle, workspace_limits)
                # nonlocal_variables['grasp_success'] = robot.grasp_first(primitive_position, best_rotation_angle, workspace_limits)
                print('Grasping successful: %r' % (nonlocal_variables['grasp_success']))
            
            elif nonlocal_variables['primitive_action'] == 'suction':
                nonlocal_variables['suction_success'] = robot.suction_first(primitive_suction_position, suction_rotation_angle, workspace_limits)
                # nonlocal_variables['suction_success'] = robot.suction_first(primitive_position, best_rotation_angle, workspace_limits)
                print('suction successful: %r' % (nonlocal_variables['suction_success']))
            
            elif nonlocal_variables['primitive_action'] == 'grasp_then_suction':
                nonlocal_variables['gs_success'] = robot.grasp_then_suction(grasp_open_distance, primitive_grasp_position, grasp_rotation_angle, primitive_suction_position, suction_rotation_angle_1, suction_rotation_angle_1, workspace_limits)                                   
                print('grasp_then_suction successful: %r' % (nonlocal_variables['gs_success']))
                                                
            # Save information for next training step
            prev_color_img = color_img.copy()
            prev_depth_img = depth_img.copy()
            # prev_color_heightmap = color_heightmap.copy()
            prev_depth_heightmap = depth_heightmap.copy()
            prev_valid_depth_heightmap_a = valid_depth_heightmap_a.copy()
            prev_suction_success = nonlocal_variables['suction_success']
            prev_grasp_success = nonlocal_variables['grasp_success']
            prev_gs_success = nonlocal_variables['gs_success']
            prev_primitive_action = nonlocal_variables['primitive_action']
            # prev_suction_predictions = suction_predictions.copy()
            # prev_grasp_predictions = grasp_predictions.copy()            
            prev_bestg_id = bestg_id
            prev_bests_id = bests_id
            prev_bestgs_g_id = bestgs_g_id
            prev_bestgs_s_id = bestgs_s_id            
            prev_objects_mask = mask_depth
            prev_sro_best = sro_best
            prev_gro_best = gro_best
            prev_bestgs_num = bestgs_num
            prev_objects_number = objects_number                        
            trainer.iteration += 1            
            episode_iter += 1
            if prev_suction_success != 0 or prev_grasp_success !=0 or prev_gs_success !=0:
                episode_succ += 1


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # --------------- Setup options ---------------
    parser.add_argument('--is_sim', dest='is_sim', action='store_true', default=False,                                      help='run in simulation?')
    parser.add_argument('--obj_mesh_dir', dest='obj_mesh_dir', action='store', default='datasets/training',                  help='directory containing 3D mesh files (.obj) of objects to be added to simulation. set to \'training\'  or \'testing\'')
    parser.add_argument('--num_obj', dest='num_obj', type=int, action='store', default=10,                                  help='number of objects to add to simulation')
    parser.add_argument('--heightmap_resolution', dest='heightmap_resolution', type=float, action='store', default=0.002,   help='meters per pixel of heightmap')
    parser.add_argument('--cpu', dest='force_cpu', action='store_true', default=False,                                      help='force code to run in CPU mode')
    parser.add_argument('--is_cluttered', dest='is_cluttered', action='store_true', default=False,                          help='set to \'lightly-cluttered\'  or \'highly-cluttered\'')
    
    # ------------- Algorithm options -------------
    parser.add_argument('--method', dest='method', action='store', default='reinforcement',                                 help='set to \'reactive\' (supervised learning) or \'reinforcement\'(DDQN)')
    parser.add_argument('--is_ets', dest='is_ets', action='store_true', default=False,                                      help='run with Enveloping_then_Sucking action?')
    parser.add_argument('--is_pe', dest='is_pe', action='store_true', default=False,                                        help='run with preenveloping?')
    parser.add_argument('--is_oo', dest='is_oo', action='store_true', default=False,                                        help='run with orientation optimization?')  
    parser.add_argument('--future_reward_discount', dest='future_reward_discount', type=float, action='store', default=0.5)
    parser.add_argument('--explore_rate_decay', dest='explore_rate_decay', action='store_true', default=False)
   
    # --------------Training or Testing options --------------
    parser.add_argument('--is_testing', dest='is_testing', action='store_true', default=False)
    parser.add_argument('--training_episode', dest='training_episode', type=int, action='store', default=800,               help='maximum number of training')
    parser.add_argument('--testing_episode', dest='testing_episode', type=int, action='store', default=300,                 help='maximum number of testing')
    parser.add_argument('--step', dest='step', type=int, action='store', default=20,                                        help='step limitation in an episode')
    parser.add_argument('--target_update_freq', dest='target_update_freq', type=int, action='store', default=10,            help='frequency to update target Q network')
    
    # ------ Pre-loading and logging options ------
    parser.add_argument('--load_snapshot', dest='load_snapshot', action='store_true', default=False,                        help='load pre-trained snapshot of model?')
    parser.add_argument('--snapshot_file', dest='snapshot_file', action='store')
    parser.add_argument('--continue_logging', dest='continue_logging', action='store_true', default=False,                  help='continue logging from previous session?')
    parser.add_argument('--logging_directory', dest='logging_directory', action='store',default='test-10-obj-01.txt')
    
    # Run main program with specified arguments
    args = parser.parse_args()
    main(args)
