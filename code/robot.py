import socket
import select
import struct
import time
import os,sys
import numpy as np
import utils
from simulation import vrep
import random
import math
from scipy.optimize import fsolve


class Robot(object):
    def __init__(self, is_sim, obj_mesh_dir, num_obj, workspace_limits,
                 tcp_host_ip, tcp_port, rtc_host_ip, rtc_port,
                 is_testing, test_preset_cases, test_preset_file):

        self.is_sim = is_sim
        self.workspace_limits = workspace_limits
        
        self.torspring_angle = np.deg2rad(110)
        self.H, self.D, self.finger_length, self.finger_width, self.finger_depth = (47+6+55)/1000, 116/1000, 118/1000, 28.77/1000, 18.29/1000
        self.D0 = self.D - self.finger_depth
        self.sucker_height = 10/1000
        
        # training & testing
        self.relieve_s_position = [-0.3,-0.6,0.3]
        self.relieve_s_position_1 = [-0.3,-0.6,0.2]        
        self.relieve_g_position = [-0.1,-0.6,0.3]
        self.relieve_g_position_1 = [-0.1,-0.6,0.2]       
        self.relieve_demo_position = [-0.5,0,0.3]
        
        # demo

        # If in simulation...
        if self.is_sim:
            # Read files in object mesh directory
            self.obj_mesh_dir_g = os.path.join(obj_mesh_dir,"grasping_suitable")
            self.obj_mesh_dir_s = os.path.join(obj_mesh_dir,"suction_suitable")
            #self.obj_mesh_dir_b = os.path.join(obj_mesh_dir,"both_suitable")            
            self.mesh_list_g = os.listdir(self.obj_mesh_dir_g)
            self.mesh_list_s = os.listdir(self.obj_mesh_dir_s)
            #self.mesh_list_b = os.listdir(self.obj_mesh_dir_b)            
            self.num_obj = num_obj    
            self.obj_index_g = []
            self.obj_index_s = []
            #self.obj_index_b = []
            for num_ml in range(len(self.mesh_list_g)):
                if os.path.splitext(self.mesh_list_g[num_ml])[1] == '.obj':
                    self.obj_index_g.append(num_ml)
            for num_ml in range(len(self.mesh_list_s)):
                if os.path.splitext(self.mesh_list_s[num_ml])[1] == '.obj':
                    self.obj_index_s.append(num_ml) 
            # for num_ml in range(len(self.mesh_list_b)):
            #     if os.path.splitext(self.mesh_list_b[num_ml])[1] == '.obj':
            #         self.obj_index_b.append(num_ml)    
                           
            self.drop_xx,self.drop_yy = np.meshgrid(np.linspace(0,2,3), np.linspace(0,3,4))
            self.drop_xx.shape = (12,1)
            self.drop_yy.shape = (12,1)
            self.drop_xx = self.workspace_limits[0][0] + (self.workspace_limits[0][1] - self.workspace_limits[0][0] - 0.28)/2 + 0.14*self.drop_xx +0.05
            self.drop_yy = self.workspace_limits[1][0] + (self.workspace_limits[1][1] - self.workspace_limits[1][0] - 0.3)/2 + 0.125*self.drop_yy - 0.025
            # self.drop_xx = self.workspace_limits[0][0] + (self.workspace_limits[0][1] - self.workspace_limits[0][0] - 0.28)/2 + 0.13*self.drop_xx + 0.055        
            # self.drop_yy = self.workspace_limits[1][0] + (self.workspace_limits[1][1] - self.workspace_limits[1][0] - 0.3)/2 + 0.13*self.drop_yy -0.01
            self.drop_xy = np.concatenate((self.drop_xx, self.drop_yy), axis=1)                       
            
            # video position
            # self.drop_xy = np.asarray([[-0.65,-0.05],[-0.62,-0.15],[-0.48,-0.1],[-0.41,0.02],[-0.47,0.1],[-0.6,0.15],[-0.67,0.05],[-0.36,-0.15],
            #                             [-0.55,0.0],[-0.35,0.17]])  
            # self.drop_xyz = np.asarray([[-5.9996e-01,-2.5018e-02,3.4975e-02],[-4.5271e-01,3.2271e-02,2.0474e-02],[-5.6702e-01,-1.3488e-01,3.2786e-02],[-2.7429e-01,-2.4782e-02,4.1487e-02],[-6.2439e-01,1.0108e-01,2.2544e-02],
            #                             [-2.9935e-01,1.0106e-01,1.1523e-02],[-3.9996e-01,-1.0002e-01,7.9780e-03],[-3.9996e-01,1.4998e-01,5.9780e-03],[-4.9997e-01,1.7498e-01,5.9779e-03]])
            
             
            # self.ori_xyz = np.asarray([[-1.7215e+02,-7.7090e-01,3.9700e+01],[-8.3007e+01,-3.4956e+01,-8.9956e+01],[9.0000e+01,3.0000e+01,8.9862e+01],[8.0943e+01,7.6333e+01,1.1022e+02],[-8.9850e+01,1.1290e+01,-9.0000e+01],
            #                             [-8.9850e+01,1.1290e+01,-9.0000e+01],[-9.0000e+01,-5.6261e+00,-9.0000e+01],[-9.0000e+01,+4.0000e+01,-9.0000e+01],[-9.0000e+01,+1.0000e+01,-9.0000e+01]])


            # Make sure to have the server side running in V-REP:
            # in a child script of a V-REP scene, add following command
            # to be executed just once, at simulation start:
            # simExtRemoteApiStart(19999)
            # then start simulation, and run this program.
            # IMPORTANT: for each successful call to simxStart, there
            # should be a corresponding call to simxFinish at the end!
            # MODIFY remoteApiConnections.txt

            # Connect to simulator
            vrep.simxFinish(-1) # Just in case, close all opened connections
            self.sim_client = vrep.simxStart('127.0.0.1', 19997, True, True, 5000, 5) # Connect to V-REP on port 19997
            if self.sim_client == -1:
                print('Failed to connect to simulation (V-REP remote API server). Exiting.')
                sys.exit()
            else:
                print('Connected to simulation.')               
                
                self.restart_sim()
            self.is_testing = is_testing
            self.test_preset_cases = test_preset_cases
            self.test_preset_file = test_preset_file

            # Setup virtual camera in simulation
            self.setup_sim_camera()

            # If testing, read object meshes and poses from test case file
            if self.is_testing and self.test_preset_cases:
                file = open(self.test_preset_file, 'r')
                file_content = file.readlines()
                self.test_obj_mesh_files = []
                self.test_obj_mesh_colors = []
                self.test_obj_positions = []
                self.test_obj_orientations = []
                for object_idx in range(self.num_obj):
                    file_content_curr_object = file_content[object_idx].split()
                    self.test_obj_mesh_files.append(os.path.join(self.obj_mesh_dir,file_content_curr_object[0]))
                    self.test_obj_mesh_colors.append([float(file_content_curr_object[1]),float(file_content_curr_object[2]),float(file_content_curr_object[3])])
                    self.test_obj_positions.append([float(file_content_curr_object[4]),float(file_content_curr_object[5]),float(file_content_curr_object[6])])
                    self.test_obj_orientations.append([float(file_content_curr_object[7]),float(file_content_curr_object[8]),float(file_content_curr_object[9])])
                file.close()
                self.obj_mesh_color = np.asarray(self.test_obj_mesh_colors)

            # Add objects to simulation environment
            #self.add_objects()

    def setup_sim_camera(self):

        # Get handle to camera
        sim_ret, self.cam_handle = vrep.simxGetObjectHandle(self.sim_client, 'Vision_sensor_persp', vrep.simx_opmode_blocking)
        # Get camera pose and intrinsics in simulation
        sim_ret, cam_position = vrep.simxGetObjectPosition(self.sim_client, self.cam_handle, -1, vrep.simx_opmode_blocking)
        sim_ret, cam_orientation = vrep.simxGetObjectOrientation(self.sim_client, self.cam_handle, -1, vrep.simx_opmode_blocking)
        cam_trans = np.eye(4,4)
        cam_trans[0:3,3] = np.asarray(cam_position)
        cam_orientation = [-cam_orientation[0], -cam_orientation[1], -cam_orientation[2]]
        cam_rotm = np.eye(4,4)
        cam_rotm[0:3,0:3] = np.linalg.inv(utils.euler2rotm(cam_orientation))
        self.cam_pose = np.dot(cam_trans, cam_rotm) # Compute rigid transformation representating camera pose
        self.cam_intrinsics = np.asarray([[618.62, 0, 320], [0, 618.62, 240], [0, 0, 1]])
        self.cam_depth_scale = 1
        # Get background image
        self.bg_color_img, self.bg_depth_img = self.get_camera_data()
        self.bg_depth_img = self.bg_depth_img * self.cam_depth_scale


    def add_objects(self):

        # Add each object to robot workspace at x,y location and orientation (random or pre-loaded)
        self.object_handles = []
        self.un_object_handles = []
        self.obj_mesh = []                
        # Randomly choose objects to add to scene
        self.num_obj_g = np.random.randint(6)
        if self.num_obj_g > 0:
            self.num_obj_s = np.random.randint(6)
        else:
            self.num_obj_s = max(np.random.randint(6),1)
        #self.num_obj_b = 1 + np.random.randint(3)
        
        self.obj_mesh_ind_g = np.random.randint(0, len(self.obj_index_g), size=self.num_obj_g)
        self.obj_mesh_ind_s = np.random.randint(0, len(self.obj_index_s), size=self.num_obj_s)
        #self.obj_mesh_ind_b = np.random.randint(0, len(self.obj_index_b), size=self.num_obj_b)
        #print(self.obj_mesh_ind_s)
        # print('self.mesh_list_g',self.mesh_list_g)
        # print('self.mesh_list_s',self.mesh_list_s)
        # print('self.obj_index_g',self.obj_index_g)
        # print('self.obj_index_s',self.obj_index_s)
        # time.sleep(10)
        # self.obj_mesh_ind_g = [5,3,4,7,2]
        # self.obj_mesh_ind_s = [5,3,0,2]
        
        for i in range(len(self.obj_mesh_ind_g)):
            self.obj_mesh.append(os.path.join(self.obj_mesh_dir_g, self.mesh_list_g[self.obj_index_g[self.obj_mesh_ind_g[i]]]))
        for i in range(len(self.obj_mesh_ind_s)):
            self.obj_mesh.append(os.path.join(self.obj_mesh_dir_s, self.mesh_list_s[self.obj_index_s[self.obj_mesh_ind_s[i]]]))
        # for i in range(len(self.obj_mesh_ind_b)):
        #     self.obj_mesh.append(os.path.join(self.obj_mesh_dir_b, self.mesh_list_b[self.obj_index_b[self.obj_mesh_ind_b[i]]]))        
        # print('self.obj_mesh')
        # print(self.obj_mesh)
          
        self.drop_id = random.sample(range(0,10),len(self.obj_mesh))
        # print('self.drop_id')
        # print(self.drop_id)
        
        for object_idx in range(len(self.obj_mesh)):
            curr_mesh_file = self.obj_mesh[object_idx]                                   
            # print('curr_mesh_file')
            # print(curr_mesh_file)
            if self.is_testing and self.test_preset_cases:
                curr_mesh_file = self.test_obj_mesh_files[object_idx]
            curr_shape_name = 'shape_%02d' % object_idx
            
            # drop_x = (self.workspace_limits[0][1] - self.workspace_limits[0][0] - 0.2) * np.random.random_sample() + self.workspace_limits[0][0] + 0.1
            # drop_y = (self.workspace_limits[1][1] - self.workspace_limits[1][0] - 0.2) * np.random.random_sample() + self.workspace_limits[1][0] + 0.1            
            xy = self.drop_xy[self.drop_id[object_idx]]
            #xy = self.drop_xy[object_idx]                        
            drop_x = xy[0]
            drop_y = xy[1]                        
            object_position = [drop_x, drop_y, 0.08]            
            #object_position = [-0.5, 0, 0.2]            
            #object_orientation = [2*np.pi*np.random.random_sample(), 2*np.pi*np.random.random_sample(), 2*np.pi*np.random.random_sample()]
            orienid_1  = random.sample([-0.5,0.5], 1)
            orienid_2  = 80*random.uniform(-1, 1)                                     
            object_orientation = [orienid_1[0]*np.pi, orienid_2, orienid_1[0]*np.pi]                        

            # video demo                                    
            # object_position = [self.drop_xyz[object_idx][0],self.drop_xyz[object_idx][1],self.drop_xyz[object_idx][2]] 
            # object_orientation = [np.deg2rad(self.ori_xyz[object_idx][0]),np.deg2rad(self.ori_xyz[object_idx][1]),np.deg2rad(self.ori_xyz[object_idx][2])] 
            
            
            if self.is_testing and self.test_preset_cases:
                object_position = [self.test_obj_positions[object_idx][0], self.test_obj_positions[object_idx][1], self.test_obj_positions[object_idx][2]]
                object_orientation = [self.test_obj_orientations[object_idx][0], self.test_obj_orientations[object_idx][1], self.test_obj_orientations[object_idx][2]]
            ret_resp,ret_ints,ret_floats,ret_strings,ret_buffer = vrep.simxCallScriptFunction(self.sim_client, 'remoteApiCommandServer',vrep.sim_scripttype_childscript,'importShape',[0,0,255,0], object_position + object_orientation, [curr_mesh_file, curr_shape_name,], bytearray(), vrep.simx_opmode_blocking)
                           
            if ret_resp == 8:
                print('Failed to add new objects to simulation. Please restart.')
                sys.exit()
            
            curr_shape_handle = ret_ints[0]
            self.object_handles.append(curr_shape_handle)
            if not (self.is_testing and self.test_preset_cases):
                time.sleep(0.5)        
        self.grasp_handle = self.object_handles[:len(self.obj_mesh_ind_g)]
        self.suction_handle = self.object_handles[len(self.obj_mesh_ind_g):]
        self.un_object_handles = self.object_handles
        self.grasped_handles = []
        self.sucked_handles = []
        self.successful_handles = []
        self.prev_obj_positions = []
        self.obj_positions = []


    def restart_sim(self):
        
        self.upperhandle = [[0],[0],[0],[0]]
        self.sgripperhandle = [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]
        self.fsensorhandle = [[0],[0],[0],[0]]
        self.spadhandle = [[0],[0],[0],[0]]
        self.spadsensorhandle = [[0],[0],[0],[0]]
        self.sdummyhandle = [[0,0],[0,0],[0,0],[0,0]]
        self.spadlinkhandle = [[0],[0],[0],[0]]
        self.suction_tip_handle = [[0],[0],[0],[0]]
            
        sim_ret, self.upperhandle[0][0] = vrep.simxGetObjectHandle(self.sim_client, 'jointS_1', vrep.simx_opmode_blocking)
        sim_ret, self.upperhandle[1][0] = vrep.simxGetObjectHandle(self.sim_client, 'jointS_2', vrep.simx_opmode_blocking)
        sim_ret, self.upperhandle[2][0] = vrep.simxGetObjectHandle(self.sim_client, 'jointS_3', vrep.simx_opmode_blocking)
        sim_ret, self.upperhandle[3][0] = vrep.simxGetObjectHandle(self.sim_client, 'jointS_4', vrep.simx_opmode_blocking)
        
        sim_ret, self.sgripperhandle[0][0] = vrep.simxGetObjectHandle(self.sim_client, 'jointA_1', vrep.simx_opmode_blocking)
        sim_ret, self.sgripperhandle[1][0] = vrep.simxGetObjectHandle(self.sim_client, 'jointA_2', vrep.simx_opmode_blocking)
        sim_ret, self.sgripperhandle[2][0] = vrep.simxGetObjectHandle(self.sim_client, 'jointA_3', vrep.simx_opmode_blocking)
        sim_ret, self.sgripperhandle[3][0] = vrep.simxGetObjectHandle(self.sim_client, 'jointA_4', vrep.simx_opmode_blocking)
        sim_ret, self.sgripperhandle[0][1] = vrep.simxGetObjectHandle(self.sim_client, 'jointB_1', vrep.simx_opmode_blocking)
        sim_ret, self.sgripperhandle[1][1] = vrep.simxGetObjectHandle(self.sim_client, 'jointB_2', vrep.simx_opmode_blocking)
        sim_ret, self.sgripperhandle[2][1] = vrep.simxGetObjectHandle(self.sim_client, 'jointB_3', vrep.simx_opmode_blocking)
        sim_ret, self.sgripperhandle[3][1] = vrep.simxGetObjectHandle(self.sim_client, 'jointB_4', vrep.simx_opmode_blocking)
        sim_ret, self.sgripperhandle[0][2] = vrep.simxGetObjectHandle(self.sim_client, 'jointC_1', vrep.simx_opmode_blocking)
        sim_ret, self.sgripperhandle[1][2] = vrep.simxGetObjectHandle(self.sim_client, 'jointC_2', vrep.simx_opmode_blocking)
        sim_ret, self.sgripperhandle[2][2] = vrep.simxGetObjectHandle(self.sim_client, 'jointC_3', vrep.simx_opmode_blocking)
        sim_ret, self.sgripperhandle[3][2] = vrep.simxGetObjectHandle(self.sim_client, 'jointC_4', vrep.simx_opmode_blocking)
        sim_ret, self.sgripperhandle[0][3] = vrep.simxGetObjectHandle(self.sim_client, 'jointD_1', vrep.simx_opmode_blocking)
        sim_ret, self.sgripperhandle[1][3] = vrep.simxGetObjectHandle(self.sim_client, 'jointD_2', vrep.simx_opmode_blocking)
        sim_ret, self.sgripperhandle[2][3] = vrep.simxGetObjectHandle(self.sim_client, 'jointD_3', vrep.simx_opmode_blocking)
        sim_ret, self.sgripperhandle[3][3] = vrep.simxGetObjectHandle(self.sim_client, 'jointD_4', vrep.simx_opmode_blocking)

        # sim_ret, self.fsensorhandle[0][0] = vrep.simxGetObjectHandle(self.sim_client, 'Force_sensor1', vrep.simx_opmode_blocking)
        # sim_ret, self.fsensorhandle[1][0] = vrep.simxGetObjectHandle(self.sim_client, 'Force_sensor2', vrep.simx_opmode_blocking)
        # sim_ret, self.fsensorhandle[2][0] = vrep.simxGetObjectHandle(self.sim_client, 'Force_sensor3', vrep.simx_opmode_blocking)
        # sim_ret, self.fsensorhandle[3][0] = vrep.simxGetObjectHandle(self.sim_client, 'Force_sensor4', vrep.simx_opmode_blocking)
        
        # sim_ret, self.spadhandle[0][0] = vrep.simxGetObjectHandle(self.sim_client, 'suctionPad1', vrep.simx_opmode_blocking)
        # sim_ret, self.spadhandle[1][0] = vrep.simxGetObjectHandle(self.sim_client, 'suctionPad2', vrep.simx_opmode_blocking)
        # sim_ret, self.spadhandle[2][0] = vrep.simxGetObjectHandle(self.sim_client, 'suctionPad3', vrep.simx_opmode_blocking)
        # sim_ret, self.spadhandle[3][0] = vrep.simxGetObjectHandle(self.sim_client, 'suctionPad4', vrep.simx_opmode_blocking)
        
        # sim_ret, self.spadsensorhandle[0][0] = vrep.simxGetObjectHandle(self.sim_client, 'suctionPadSensor11', vrep.simx_opmode_blocking)
        # sim_ret, self.spadsensorhandle[1][0] = vrep.simxGetObjectHandle(self.sim_client, 'suctionPadSensor12', vrep.simx_opmode_blocking)
        # sim_ret, self.spadsensorhandle[2][0] = vrep.simxGetObjectHandle(self.sim_client, 'suctionPadSensor13', vrep.simx_opmode_blocking)
        # sim_ret, self.spadsensorhandle[3][0] = vrep.simxGetObjectHandle(self.sim_client, 'suctionPadSensor14', vrep.simx_opmode_blocking)
                               
        # sim_ret, self.sdummyhandle[0][0] = vrep.simxGetObjectHandle(self.sim_client, 'suctionPadLoopClosureDummy11', vrep.simx_opmode_blocking)   
        # sim_ret, self.sdummyhandle[0][1] = vrep.simxGetObjectHandle(self.sim_client, 'suctionPadLoopClosureDummy12', vrep.simx_opmode_blocking)
        # sim_ret, self.sdummyhandle[1][0] = vrep.simxGetObjectHandle(self.sim_client, 'suctionPadLoopClosureDummy21', vrep.simx_opmode_blocking)   
        # sim_ret, self.sdummyhandle[1][1] = vrep.simxGetObjectHandle(self.sim_client, 'suctionPadLoopClosureDummy22', vrep.simx_opmode_blocking)
        # sim_ret, self.sdummyhandle[2][0] = vrep.simxGetObjectHandle(self.sim_client, 'suctionPadLoopClosureDummy31', vrep.simx_opmode_blocking)   
        # sim_ret, self.sdummyhandle[2][1] = vrep.simxGetObjectHandle(self.sim_client, 'suctionPadLoopClosureDummy32', vrep.simx_opmode_blocking)
        # sim_ret, self.sdummyhandle[3][0] = vrep.simxGetObjectHandle(self.sim_client, 'suctionPadLoopClosureDummy41', vrep.simx_opmode_blocking)   
        # sim_ret, self.sdummyhandle[3][1] = vrep.simxGetObjectHandle(self.sim_client, 'suctionPadLoopClosureDummy42', vrep.simx_opmode_blocking)
        
        # sim_ret, self.spadlinkhandle[0][0] = vrep.simxGetObjectHandle(self.sim_client, 'suctionPadLink1', vrep.simx_opmode_blocking)
        # sim_ret, self.spadlinkhandle[1][0] = vrep.simxGetObjectHandle(self.sim_client, 'suctionPadLink2', vrep.simx_opmode_blocking)
        # sim_ret, self.spadlinkhandle[2][0] = vrep.simxGetObjectHandle(self.sim_client, 'suctionPadLink3', vrep.simx_opmode_blocking)
        # sim_ret, self.spadlinkhandle[3][0] = vrep.simxGetObjectHandle(self.sim_client, 'suctionPadLink4', vrep.simx_opmode_blocking)                
        
        sim_ret, self.suction_tip_handle[0][0] = vrep.simxGetObjectHandle(self.sim_client,'suction_tip1',vrep.simx_opmode_blocking)
        sim_ret, self.suction_tip_handle[1][0] = vrep.simxGetObjectHandle(self.sim_client,'suction_tip2',vrep.simx_opmode_blocking)
        sim_ret, self.suction_tip_handle[2][0] = vrep.simxGetObjectHandle(self.sim_client,'suction_tip3',vrep.simx_opmode_blocking)
        sim_ret, self.suction_tip_handle[3][0] = vrep.simxGetObjectHandle(self.sim_client,'suction_tip4',vrep.simx_opmode_blocking)
        
        sim_ret, self.grasp_target_handle = vrep.simxGetObjectHandle(self.sim_client,'gs_target',vrep.simx_opmode_blocking)
        sim_ret, self.grasp_tip_handle = vrep.simxGetObjectHandle(self.sim_client,'grasp_tip',vrep.simx_opmode_blocking)
        
                                       

        vrep.simxSetObjectPosition(self.sim_client, self.grasp_target_handle, -1, (-0.5,0,0.3), vrep.simx_opmode_blocking)
        vrep.simxStopSimulation(self.sim_client, vrep.simx_opmode_blocking)
        vrep.simxStartSimulation(self.sim_client, vrep.simx_opmode_blocking)
        time.sleep(1)
        sim_ret, gripper_position = vrep.simxGetObjectPosition(self.sim_client, self.grasp_tip_handle, -1, vrep.simx_opmode_blocking)
        while gripper_position[2] > 0.4: # V-REP bug requiring multiple starts and stops to restart
            vrep.simxStopSimulation(self.sim_client, vrep.simx_opmode_blocking)
            vrep.simxStartSimulation(self.sim_client, vrep.simx_opmode_blocking)
            time.sleep(0.5)
            sim_ret, gripper_position = vrep.simxGetObjectPosition(self.sim_client, self.grasp_tip_handle, -1, vrep.simx_opmode_blocking)


    def check_sim(self):

        # Check if simulation is stable by checking if gripper is within workspace
        sim_ret, gripper_position = vrep.simxGetObjectPosition(self.sim_client, self.grasp_tip_handle, -1, vrep.simx_opmode_blocking)
        sim_ok = gripper_position[0] > self.workspace_limits[0][0] - 0.1 and gripper_position[0] < self.workspace_limits[0][1] + 0.1 and gripper_position[1] > self.workspace_limits[1][0] - 0.1 and gripper_position[1] < self.workspace_limits[1][1] + 0.1 and gripper_position[2] > self.workspace_limits[2][0] and gripper_position[2] < self.workspace_limits[2][1]
        if not sim_ok:
            print('Simulation unstable. Restarting environment.')
            self.restart_sim()
            self.add_objects()


    def get_camera_data(self):

        if self.is_sim:

            # Get color image from simulation
            sim_ret, resolution, raw_image = vrep.simxGetVisionSensorImage(self.sim_client, self.cam_handle, 0, vrep.simx_opmode_blocking)
            color_img = np.asarray(raw_image)
            color_img.shape = (resolution[1], resolution[0], 3)
            color_img = color_img.astype(np.float)/255
            color_img[color_img < 0] += 1
            color_img *= 255
            color_img = np.fliplr(color_img)
            color_img = color_img.astype(np.uint8)

            # Get depth image from simulation
            sim_ret, resolution, depth_buffer = vrep.simxGetVisionSensorDepthBuffer(self.sim_client, self.cam_handle, vrep.simx_opmode_blocking)
            depth_img = np.asarray(depth_buffer)
            depth_img.shape = (resolution[1], resolution[0])
            depth_img = np.fliplr(depth_img)
            zNear = 0.01
            zFar = 10
            depth_img = depth_img * (zFar - zNear) + zNear

        return color_img, depth_img


    # def get_obj_positions(self):

    #     obj_positions = []
    #     for object_handle in self.object_handles:
    #         sim_ret, object_position = vrep.simxGetObjectPosition(self.sim_client, object_handle, -1, vrep.simx_opmode_blocking)
    #         obj_positions.append(object_position)

    #     return obj_positions


    def get_obj_positions(self,handles):

        obj_positions = []
        for object_handle in handles:
            sim_ret, object_position = vrep.simxGetObjectPosition(self.sim_client, object_handle, -1, vrep.simx_opmode_blocking)
            obj_positions.append(object_position)

        return obj_positions        

    def close_gripper(self, is_suction=1, rotate_angle_1=np.pi/2.5, rotate_angle_2=np.pi/8, asynch=False):

        if self.is_sim:                       
            if is_suction == 1:            
                angle_mag = rotate_angle_1/4
                angle_step = angle_mag
                for step_iter in range(1,int(angle_mag/angle_step)+1):
                    deg = step_iter*angle_step
                    for i in range(4):
                        for j in range(4):
                            vrep.simxSetJointPosition(self.sim_client, self.sgripperhandle[j][i], deg, vrep.simx_opmode_blocking)                  
            
            elif is_suction == 0:
                # grasp_force = np.zeros((4,3))
                # grasp_sumforce = np.zeros(4)
                # exit_called = False
                if rotate_angle_1 > 0:
                    angle_mag = rotate_angle_1/4
                    angle_step = angle_mag/2
                    for step_iter in range(1,int(angle_mag/angle_step)+1):
                        deg = step_iter*angle_step
                        for i in range(4):
                            for j in range(4):
                                vrep.simxSetJointPosition(self.sim_client, self.sgripperhandle[j][i], deg, vrep.simx_opmode_blocking)
                
            elif is_suction == -1:                
                # exit_called = False
                angle_mag = max(rotate_angle_2/4,0.001)
                angle_step = angle_mag/4
                for step_iter in range(1,int(angle_mag/angle_step)+1):
                    deg = step_iter*angle_step + rotate_angle_1/4
                    for i in range(4):
                        for j in range(4):
                            vrep.simxSetJointPosition(self.sim_client, self.sgripperhandle[j][i], deg, vrep.simx_opmode_blocking)
                
                      
                        # for force_num in range(4):
                        #     _,_, grasp_force[force_num],_ = vrep.simxReadForceSensor(self.sim_client, self.fsensorhandle[force_num][0], vrep.simx_opmode_blocking)
                        #     grasp_sumforce[force_num] = math.sqrt(grasp_force[force_num][0]**2 + grasp_force[force_num][1]**2 + grasp_force[force_num][2]**2)
                        # # if grasp_sumforce[0]>50 or grasp_sumforce[1]>50 or grasp_sumforce[2]>50 or grasp_sumforce[3]>50:
                        # if (grasp_sumforce[0]>50 and grasp_sumforce[2]>50) and (grasp_sumforce[1]>50 and grasp_sumforce[3]>50):
                        #     exit_called = True
                        #     break                       
                    # if exit_called == True:
                    #     break                    
                            
            # for step_iter in range(1,3):
            #     deg = step_iter*deg_step
            #     for i in range(4):
            #         vrep.simxSetJointPosition(self.sim_client, self.upperhandle[i][0], deg, vrep.simx_opmode_blocking)   
       

    def pre_rotate_angle(self, distance):                 
        if distance >= (self.D0 + 2*self.finger_length*np.sin(self.torspring_angle-np.pi/2))/math.sqrt(2):
            rotate_angle = 0
        else:
            distance = max(distance*math.sqrt(2),0.03)
            #print('distance',distance)
            def func(x):
                return [self.D0 - 2*self.finger_length*(np.cos(self.torspring_angle - np.pi/2) - np.sin(x[0]))/(self.torspring_angle - x[0]) - distance*x[1],
                        x[1]-1]   
            root = fsolve(func, [np.pi/100,1])
            #print('root[0]',root[0])
            rotate_angle = self.torspring_angle - root[0]
            
        return rotate_angle
    
    def open_gripper(self, asynch=False):

        if self.is_sim:            
            open_deg = 0
            for i in range(3,-1,-1):
                for j in range(3,-1,-1):
                    vrep.simxSetJointPosition(self.sim_client, self.sgripperhandle[j][i], open_deg, vrep.simx_opmode_blocking)            

    def move_to(self, tool_position, tool_orientation, suction_id):

        if self.is_sim:
            if suction_id != -1 and suction_id != -2:                                                                                              
                sim_ret, grasp_target_position = vrep.simxGetObjectPosition(self.sim_client, self.grasp_target_handle,-1,vrep.simx_opmode_blocking)
                sim_ret, suction_tip_position = vrep.simxGetObjectPosition(self.sim_client, self.suction_tip_handle[suction_id][0],-1,vrep.simx_opmode_blocking)                                                             
                move_direction = np.asarray([tool_position[0] - suction_tip_position[0], tool_position[1] - suction_tip_position[1], tool_position[2] - suction_tip_position[2]])
                grasp_tool_position = grasp_target_position + move_direction                               
                move_magnitude = np.linalg.norm(move_direction*0.7)
                move_step = 0.01*0.7*move_direction/move_magnitude
                num_move_steps = int(np.floor(move_magnitude/0.01))                
    
                for step_iter in range(num_move_steps):
                    vrep.simxSetObjectPosition(self.sim_client,self.grasp_target_handle,-1,(grasp_target_position[0] + move_step[0], grasp_target_position[1] + move_step[1], grasp_target_position[2] + move_step[2]),vrep.simx_opmode_blocking)
                    sim_ret, grasp_target_position = vrep.simxGetObjectPosition(self.sim_client,self.grasp_target_handle,-1,vrep.simx_opmode_blocking)                                    
                 
                sim_ret, grasp_target_position = vrep.simxGetObjectPosition(self.sim_client, self.grasp_target_handle,-1,vrep.simx_opmode_blocking)
                sim_ret, suction_tip_position = vrep.simxGetObjectPosition(self.sim_client, self.suction_tip_handle[suction_id][0],-1,vrep.simx_opmode_blocking)               
                move_direction = np.asarray([tool_position[0] - suction_tip_position[0], tool_position[1] - suction_tip_position[1], tool_position[2] - suction_tip_position[2]])
                move_magnitude = np.linalg.norm(move_direction)
                move_step = 0.002*move_direction/move_magnitude
                num_move_steps = int(np.floor(move_magnitude/0.002))                    
    
                for step_iter in range(num_move_steps):
                    vrep.simxSetObjectPosition(self.sim_client,self.grasp_target_handle,-1,(grasp_target_position[0] + move_step[0], grasp_target_position[1] + move_step[1], grasp_target_position[2] + move_step[2]),vrep.simx_opmode_blocking)
                    sim_ret, grasp_target_position = vrep.simxGetObjectPosition(self.sim_client,self.grasp_target_handle,-1,vrep.simx_opmode_blocking)                                                                                                                                                                                                                                             
                vrep.simxSetObjectPosition(self.sim_client,self.grasp_target_handle,-1,(grasp_tool_position[0],grasp_tool_position[1],grasp_tool_position[2]),vrep.simx_opmode_blocking)
               

                
            elif suction_id == -1:
                
                sim_ret, grasp_target_position = vrep.simxGetObjectPosition(self.sim_client, self.grasp_target_handle,-1,vrep.simx_opmode_blocking)                            
                move_direction = np.asarray([tool_position[0] - grasp_target_position[0], tool_position[1] - grasp_target_position[1], tool_position[2] - grasp_target_position[2]])                
                move_magnitude = np.linalg.norm(move_direction*0.7)
                move_step = 0.01*0.7*move_direction/move_magnitude
                num_move_steps = int(np.floor(move_magnitude/0.01))
   
                for step_iter in range(num_move_steps):
                    vrep.simxSetObjectPosition(self.sim_client,self.grasp_target_handle,-1,(grasp_target_position[0] + move_step[0], grasp_target_position[1] + move_step[1], grasp_target_position[2] + move_step[2]),vrep.simx_opmode_blocking)
                    sim_ret, grasp_target_position = vrep.simxGetObjectPosition(self.sim_client,self.grasp_target_handle,-1,vrep.simx_opmode_blocking)
                    
                sim_ret, grasp_target_position = vrep.simxGetObjectPosition(self.sim_client, self.grasp_target_handle,-1,vrep.simx_opmode_blocking)            
                move_direction = np.asarray([tool_position[0] - grasp_target_position[0], tool_position[1] - grasp_target_position[1], tool_position[2] - grasp_target_position[2]])
                move_magnitude = np.linalg.norm(move_direction)
                move_step = 0.002*move_direction/move_magnitude
                num_move_steps = int(np.floor(move_magnitude/0.002))                
    
                for step_iter in range(num_move_steps):
                    vrep.simxSetObjectPosition(self.sim_client,self.grasp_target_handle,-1,(grasp_target_position[0] + move_step[0], grasp_target_position[1] + move_step[1], grasp_target_position[2] + move_step[2]),vrep.simx_opmode_blocking)
                    sim_ret, grasp_target_position = vrep.simxGetObjectPosition(self.sim_client,self.grasp_target_handle,-1,vrep.simx_opmode_blocking)                                                                    
                
                vrep.simxSetObjectPosition(self.sim_client,self.grasp_target_handle,-1,(tool_position[0],tool_position[1],tool_position[2]),vrep.simx_opmode_blocking)


            elif suction_id == -2:                
                # sim_ret, grasp_target_position = vrep.simxGetObjectPosition(self.sim_client, self.grasp_target_handle,-1,vrep.simx_opmode_blocking)            
                # move_direction = np.asarray([tool_position[0] - grasp_target_position[0], tool_position[1] - grasp_target_position[1], tool_position[2] - grasp_target_position[2]])
                # move_magnitude = np.linalg.norm(move_direction)
                # move_step = 0.02*move_direction/move_magnitude
                # num_move_steps = int(np.floor(move_magnitude/0.02))
    
                # for step_iter in range(num_move_steps):
                #     vrep.simxSetObjectPosition(self.sim_client,self.grasp_target_handle,-1,(grasp_target_position[0] + move_step[0], grasp_target_position[1] + move_step[1], grasp_target_position[2] + move_step[2]),vrep.simx_opmode_blocking)
                #     sim_ret, grasp_target_position = vrep.simxGetObjectPosition(self.sim_client,self.grasp_target_handle,-1,vrep.simx_opmode_blocking)                                                
                # vrep.simxSetObjectPosition(self.sim_client,self.grasp_target_handle,-1,(tool_position[0],tool_position[1],tool_position[2]),vrep.simx_opmode_blocking)
                
                sim_ret, grasp_target_position = vrep.simxGetObjectPosition(self.sim_client, self.grasp_target_handle,-1,vrep.simx_opmode_blocking)                            
                move_direction = np.asarray([tool_position[0] - grasp_target_position[0], tool_position[1] - grasp_target_position[1], tool_position[2] - grasp_target_position[2]])                
                move_magnitude = np.linalg.norm(move_direction*0.5)
                move_step = 0.002*0.5*move_direction/move_magnitude
                num_move_steps = int(np.floor(move_magnitude/0.002))
   
                for step_iter in range(num_move_steps):
                    vrep.simxSetObjectPosition(self.sim_client,self.grasp_target_handle,-1,(grasp_target_position[0] + move_step[0], grasp_target_position[1] + move_step[1], grasp_target_position[2] + move_step[2]),vrep.simx_opmode_blocking)
                    sim_ret, grasp_target_position = vrep.simxGetObjectPosition(self.sim_client,self.grasp_target_handle,-1,vrep.simx_opmode_blocking)
                    
                sim_ret, grasp_target_position = vrep.simxGetObjectPosition(self.sim_client, self.grasp_target_handle,-1,vrep.simx_opmode_blocking)            
                move_direction = np.asarray([tool_position[0] - grasp_target_position[0], tool_position[1] - grasp_target_position[1], tool_position[2] - grasp_target_position[2]])
                move_magnitude = np.linalg.norm(move_direction)
                move_step = 0.01*move_direction/move_magnitude
                num_move_steps = int(np.floor(move_magnitude/0.01))                
    
                for step_iter in range(num_move_steps):
                    vrep.simxSetObjectPosition(self.sim_client,self.grasp_target_handle,-1,(grasp_target_position[0] + move_step[0], grasp_target_position[1] + move_step[1], grasp_target_position[2] + move_step[2]),vrep.simx_opmode_blocking)
                    sim_ret, grasp_target_position = vrep.simxGetObjectPosition(self.sim_client,self.grasp_target_handle,-1,vrep.simx_opmode_blocking)                                                                    
                
                vrep.simxSetObjectPosition(self.sim_client,self.grasp_target_handle,-1,(tool_position[0],tool_position[1],tool_position[2]),vrep.simx_opmode_blocking)

    
    def check_grasp_success(self):        
            
        object_positions = np.asarray(self.get_obj_positions(self.object_handles))
        object_positions_z = object_positions[:,2]            
        positions_sorted = np.argsort(object_positions_z)
        num_1 = positions_sorted[-1]
        position_1 = object_positions_z[num_1]
        if position_1 > 0.11:           
            # training
            # self.grasped_handles.append(self.un_object_handles[num_1])
            vrep.simxSetObjectPosition(self.sim_client, self.object_handles[num_1], -1,(-0.5, 0.5 + 0.1*float(num_1), 0.1),vrep.simx_opmode_blocking)
            self.successful_handles.append(self.object_handles[num_1])
            return 1
        else:
            return 0                
                                           
        #     grasp_force = np.zeros((4,3))
        #     grasp_sumforce = np.zeros(4)            
        #     for force_num in range(4):
        #         _,_, grasp_force[force_num],_ = vrep.simxReadForceSensor(self.sim_client, self.fsensorhandle[force_num][0], vrep.simx_opmode_blocking)
        #         grasp_sumforce[force_num] = math.sqrt(grasp_force[force_num][0]**2 + grasp_force[force_num][1]**2 + grasp_force[force_num][2]**2)
        #     if grasp_sumforce[0]>1 or grasp_sumforce[1]>1 or grasp_sumforce[2]>1 or grasp_sumforce[3]>1:
        #         return 1
        #     else:
        #         return 0  
        
        
    
    def suction_active(self, is_active, suction_id=-1):
        
        if is_active:
            float_number = float(len(self.object_handles)) + 0.1
            if suction_id == 0:                
                ret_1,ret_2,ret_3,ret_4,ret_5 = vrep.simxCallScriptFunction(self.sim_client, 'suctionPad1',vrep.sim_scripttype_childscript,'active_true',self.object_handles, [float_number], [], bytearray(), vrep.simx_opmode_blocking)
            if suction_id == 1:                
                ret_1,ret_2,ret_3,ret_4,ret_5 = vrep.simxCallScriptFunction(self.sim_client, 'suctionPad2',vrep.sim_scripttype_childscript,'active_true',self.object_handles, [float_number], [], bytearray(), vrep.simx_opmode_blocking)
            if suction_id == 2:                
                ret_1,ret_2,ret_3,ret_4,ret_5 = vrep.simxCallScriptFunction(self.sim_client, 'suctionPad3',vrep.sim_scripttype_childscript,'active_true',self.object_handles, [float_number], [], bytearray(), vrep.simx_opmode_blocking)
            if suction_id == 3:                
                ret_1,ret_2,ret_3,ret_4,ret_5 = vrep.simxCallScriptFunction(self.sim_client, 'suctionPad4',vrep.sim_scripttype_childscript,'active_true',self.object_handles, [float_number], [], bytearray(), vrep.simx_opmode_blocking)
            return ret_2[0],ret_2[1]
        else:
            vrep.simxCallScriptFunction(self.sim_client, 'suctionPad1',vrep.sim_scripttype_childscript,'active_false',[], [], [], bytearray(), vrep.simx_opmode_blocking)
            vrep.simxCallScriptFunction(self.sim_client, 'suctionPad2',vrep.sim_scripttype_childscript,'active_false',[], [], [], bytearray(), vrep.simx_opmode_blocking)
            vrep.simxCallScriptFunction(self.sim_client, 'suctionPad3',vrep.sim_scripttype_childscript,'active_false',[], [], [], bytearray(), vrep.simx_opmode_blocking)
            vrep.simxCallScriptFunction(self.sim_client, 'suctionPad4',vrep.sim_scripttype_childscript,'active_false',[], [], [], bytearray(), vrep.simx_opmode_blocking)                    
                
    def get_t_position(self, ttt):
        if ttt:
            sim_ret, p_t = vrep.simxGetObjectPosition(self.sim_client, self.grasp_tip_handle, -1, vrep.simx_opmode_blocking)
            sim_ret, p_g = vrep.simxGetObjectPosition(self.sim_client, self.grasp_target_handle, -1, vrep.simx_opmode_blocking)
            return p_t, p_g
    
    def grasp_then_suction(self, grasp_open_distance, primitive_grasp_position, grasp_rotation_angle, primitive_suction_position, suction_rotation_angle_0, suction_rotation_angle_1, workspace_limits):
        grasp_then_suction = 0
        grasp_success = 0
        suction_success = 0
        grasp_success_1, finger_angle = self.grasp(grasp_open_distance, primitive_grasp_position, grasp_rotation_angle, workspace_limits)
        if grasp_success_1:
            suction_success_1 = self.suction(primitive_suction_position, suction_rotation_angle_1, workspace_limits, 0)
        else:
            suction_success_1 = self.suction(primitive_suction_position, suction_rotation_angle_0, workspace_limits, finger_angle)
            
        object_positions = np.asarray(self.get_obj_positions(self.object_handles))
        object_positions_z = object_positions[:,2]        
        positions_sorted = np.argsort(object_positions_z)
        num_1 = positions_sorted[-1]
        num_2 = positions_sorted[-2]
        position_1 = object_positions_z[num_1]
        position_2 = object_positions_z[num_2]
        handle_1 = self.object_handles[num_1]
        handle_2 = self.object_handles[num_2]                      
                    
        
        # training & testing
        self.suction_active(is_active=False,suction_id=-1)        
        if position_1 > 0.11:
            vrep.simxSetObjectPosition(self.sim_client, handle_1, -1,(-0.5, 0.5 + 0.1*float(num_1), 0.1),vrep.simx_opmode_blocking)
            self.successful_handles.append(handle_1)
        if position_2 > 0.11:
            vrep.simxSetObjectPosition(self.sim_client, handle_2, -1,(-0.5, 0.5 + 0.1*float(num_2), 0.1),vrep.simx_opmode_blocking)                                    
            self.successful_handles.append(handle_2)
        #self.un_object_handles = list(set(self.object_handles).difference(set(self.successful_handles)))
        self.open_gripper()                
        # self.move_to([-0.5,0,0.25], None, is_suction=-1)
        if suction_success_1:
            suction_success = 1
        if grasp_success_1:
            grasp_success = 1
        
        # # video demo
        # if suction_success_1:
        #     suction_success = 1
        #     self.move_to(self.relieve_s_position, None, suction_id=-2) 
 
        #     self.suction_active(is_active=False,suction_id=-1)
        #     if position_2 > 0.07:
        #         # orienid_1  = random.sample([-0.5,0.5], 1)
        #         # orienid_2  = 80*random.uniform(-1, 1)                                     
        #         # object_orientation = [orienid_1[0]*np.pi, orienid_2, orienid_1[0]*np.pi]
        #         vrep.simxSetObjectPosition(self.sim_client, handle_2, -1,(-0.35+0.005*float(num_2),-0.65+0.005*float(num_2),0.03),vrep.simx_opmode_blocking)
        #         #vrep.simxSetObjectOrientation(self.sim_client, handle_2, -1, object_orientation, vrep.simx_opmode_blocking)
            
        #     if grasp_success_1:
        #         grasp_success = 1
        #         self.move_to(self.relieve_g_position, None, suction_id=-2)
        #     self.open_gripper()
        # else:
        #     if grasp_success_1:
        #         grasp_success = 1
        #         self.move_to(self.relieve_g_position, None, suction_id=-2)                
        #     self.suction_active(is_active=False,suction_id=-1)
        #     self.open_gripper()                           
                
        
        # reset the object if its "z" < 0
        for un_number in range(len(object_positions_z)):                                                            
            if object_positions[un_number,2] < 0.001:                
                vrep.simxSetObjectPosition(self.sim_client, self.object_handles[un_number], -1,(object_positions[un_number,0], object_positions[un_number,1], 0.08), vrep.simx_opmode_blocking)          
                time.sleep(0.5)
    
        if grasp_success == 1:
            if suction_success == 0:                                             
                grasp_then_suction = 0.5
            elif suction_success == 1:
                grasp_then_suction = 2.5
        elif grasp_success == 0:
            if suction_success == 1:
                grasp_then_suction = 0.5
    
        return grasp_then_suction
    
    
    def grasp_first(self, grasp_open_distance, primitive_grasp_position, grasp_rotation_angle, workspace_limits):                                            
        grasp_first_success = 0
        grasp_success_1 = 0        
        grasp_success_1,_ = self.grasp(grasp_open_distance, primitive_grasp_position, grasp_rotation_angle, workspace_limits)            
        
        object_positions = np.asarray(self.get_obj_positions(self.object_handles))
        object_positions_z = object_positions[:,2]        
        positions_sorted = np.argsort(object_positions_z)
        num_1 = positions_sorted[-1]
       # num_2 = positions_sorted[-2]
        position_1 = object_positions_z[num_1]
        #position_2 = object_positions_z[num_2]
        handle_1 = self.object_handles[num_1]
       # handle_2 = self.object_handles[num_2]
        
       
        # training & testing
        self.suction_active(is_active=False,suction_id=-1)        
        if position_1 > 0.11:
            vrep.simxSetObjectPosition(self.sim_client, handle_1, -1,(-0.5, 0.5 + 0.1*float(num_1), 0.1),vrep.simx_opmode_blocking)
            self.successful_handles.append(handle_1)
        # if position_2 > 0.11:
        #     vrep.simxSetObjectPosition(self.sim_client, handle_2, -1,(-0.5, 0.5 + 0.1*float(num_2), 0.1),vrep.simx_opmode_blocking)                                    
        #     self.successful_handles.append(handle_2)
        #self.un_object_handles = list(set(self.object_handles).difference(set(self.successful_handles)))
        
        
        if grasp_success_1:
            grasp_first_success = 1         
            # # video demo
            # self.move_to(self.relieve_g_position, None, suction_id=-2)                  
         
        self.open_gripper()
        self.suction_active(is_active=False,suction_id=-1)                        
        # reset the object if its "z" < 0
        for un_number in range(len(object_positions_z)):                                                            
            if object_positions[un_number,2] < 0.001:                
                vrep.simxSetObjectPosition(self.sim_client, self.object_handles[un_number], -1,(object_positions[un_number,0], object_positions[un_number,1], 0.08), vrep.simx_opmode_blocking)          
                time.sleep(0.5)                        
       
        return grasp_first_success
    
    def suction_first(self, primitive_suction_position, suction_rotation_angle, workspace_limits):                                            
                
        suction_first_success = 0        
        suction_success_1 = 0
        
        # Ensure gripper is closed                    
        suction_success_1 = self.suction(primitive_suction_position, suction_rotation_angle, workspace_limits, 1)              
        object_positions = np.asarray(self.get_obj_positions(self.object_handles))
        object_positions_z = object_positions[:,2]        
        
        positions_sorted = np.argsort(object_positions_z)
        num_1 = positions_sorted[-1]
#        num_2 = positions_sorted[-2]
        position_1 = object_positions_z[num_1]
#        position_2 = object_positions_z[num_2]
        handle_1 = self.object_handles[num_1]
#        handle_2 = self.object_handles[num_2]
                        
        
        # training & testing
        self.suction_active(is_active=False,suction_id=-1)        
        if position_1 > 0.11:
            vrep.simxSetObjectPosition(self.sim_client, handle_1, -1,(-0.5, 0.5 + 0.1*float(num_1), 0.1),vrep.simx_opmode_blocking)
            self.successful_handles.append(handle_1)
        # if position_2 > 0.11:
        #     vrep.simxSetObjectPosition(self.sim_client, handle_2, -1,(-0.5, 0.5 + 0.1*float(num_2), 0.1),vrep.simx_opmode_blocking)                                    
        #     self.successful_handles.append(handle_2)
        #self.un_object_handles = list(set(self.object_handles).difference(set(self.successful_handles)))
        
        if suction_success_1:
            suction_first_success = 1
            # # video demo
            # self.move_to(self.relieve_s_position, None, suction_id=-2)                                  
        self.suction_active(is_active=False,suction_id=-1)
        # if position_1 > 0.07:
        #     vrep.simxSetObjectPosition(self.sim_client, handle_1, -1,(-0.35+0.005*float(num_1),-0.65+0.005*float(num_1),0.02),vrep.simx_opmode_blocking)

                                                       
        self.open_gripper()
        for un_number in range(len(object_positions_z)):                                                 
            if object_positions[un_number,2] < 0.001:                
                vrep.simxSetObjectPosition(self.sim_client, self.object_handles[un_number], -1,(object_positions[un_number,0], object_positions[un_number,1], 0.08), vrep.simx_opmode_blocking)          
                time.sleep(0.5)
            
        return suction_first_success

    def grasp(self, grasp_open_distance, position, rotation_angle, workspace_limits):

        if self.is_sim:            
            # Compute tool orientation from heightmap rotation angle
            if rotation_angle <= np.pi/2:
                tool_rotation_angle = rotation_angle - np.pi/4
                
            else:
                tool_rotation_angle = rotation_angle - 3*np.pi/4            
            print('rotation_angle',rotation_angle)
            print('tool_rotation_angle',tool_rotation_angle)
                        
            # Avoid collision with floor
            position = np.asarray(position).copy()
            if position[2] < 0.018:
                position[2] = max(position[2] - 0.075, workspace_limits[2][0]) # training
            else:
                position[2] = max(position[2] - 0.175, workspace_limits[2][0]-0.01) # video demo
            
            # Move gripper to location above grasp target
            grasp_location_margin = 0.2
            # sim_ret, grasp_target_handle = vrep.simxGetObjectHandle(self.sim_client,'grasp_target',vrep.simx_opmode_blocking)
            location_above_grasp_target = (position[0], position[1], position[2] + grasp_location_margin)            
            # Compute gripper position and linear movement increments
            tool_position = location_above_grasp_target
            sim_ret, grasp_target_position = vrep.simxGetObjectPosition(self.sim_client, self.grasp_target_handle,-1,vrep.simx_opmode_blocking)
            move_direction = np.asarray([tool_position[0] - grasp_target_position[0], tool_position[1] - grasp_target_position[1], tool_position[2] - grasp_target_position[2]])
            move_magnitude = np.linalg.norm(move_direction)
            move_step = 0.02*move_direction/move_magnitude
            num_move_steps = int(np.floor(move_direction[0]/move_step[0]))
            # Compute gripper orientation and rotation increments
            sim_ret, gripper_orientation = vrep.simxGetObjectOrientation(self.sim_client, self.grasp_target_handle, -1, vrep.simx_opmode_blocking)
            rotation_step = 0.3 if (tool_rotation_angle - gripper_orientation[2] > 0) else -0.3
            num_rotation_steps = int(np.floor((tool_rotation_angle - gripper_orientation[2])/rotation_step))
            # Simultaneously move and rotate gripper
            for step_iter in range(max(num_move_steps, num_rotation_steps)):
                vrep.simxSetObjectPosition(self.sim_client,self.grasp_target_handle,-1,(grasp_target_position[0] + move_step[0]*min(step_iter,num_move_steps), grasp_target_position[1] + move_step[1]*min(step_iter,num_move_steps), grasp_target_position[2] + move_step[2]*min(step_iter,num_move_steps)),vrep.simx_opmode_blocking)
                vrep.simxSetObjectOrientation(self.sim_client, self.grasp_target_handle, -1, (0, 0, gripper_orientation[2] + rotation_step*min(step_iter,num_rotation_steps)), vrep.simx_opmode_blocking)
            vrep.simxSetObjectPosition(self.sim_client,self.grasp_target_handle,-1,(tool_position[0],tool_position[1],tool_position[2]),vrep.simx_opmode_blocking)
            vrep.simxSetObjectOrientation(self.sim_client, self.grasp_target_handle, -1, (0, 0, tool_rotation_angle), vrep.simx_opmode_blocking)
            # Ensure gripper is open
            self.open_gripper()
            
            # pre_close the gripper            
            print('grasp_open_distance',grasp_open_distance) 
            if grasp_open_distance < 0.08:
                pre_rotate_angle = self.pre_rotate_angle(grasp_open_distance*1.3)
            elif grasp_open_distance > 0.89:
                pre_rotate_angle = self.pre_rotate_angle(grasp_open_distance*1.2)
            else:
                pre_rotate_angle = self.pre_rotate_angle(grasp_open_distance*1.2)
            
            self.close_gripper(is_suction=0,rotate_angle_1=pre_rotate_angle)           
            # Approach grasp target
            self.move_to(position, None, suction_id=-1)            
            # Close gripper to grasp target
            if grasp_open_distance < 0.08:
                self.close_gripper(is_suction=-1, rotate_angle_1=pre_rotate_angle, rotate_angle_2=np.pi/2.5 - pre_rotate_angle)
                rotate_angle = np.pi/2.5
            elif grasp_open_distance > 0.89:
                self.close_gripper(is_suction=-1, rotate_angle_1=pre_rotate_angle, rotate_angle_2=np.pi/4.5)
                rotate_angle = pre_rotate_angle + np.pi/5
            else:
                self.close_gripper(is_suction=-1, rotate_angle_1=pre_rotate_angle, rotate_angle_2=np.pi/4.5)
                rotate_angle = pre_rotate_angle + np.pi/5
            # Move gripper to location above grasp target
            self.move_to(location_above_grasp_target, None, suction_id=-2)                        
            # Check if grasp is successful            
            grasp_success = self.check_grasp_success()
            # print('grasp_success: %r' % (grasp_success))                      

        return grasp_success, rotate_angle


    def suction(self, position, heightmap_rotation_angle, workspace_limits, finger_angle):
        print('finger_angle',finger_angle)
        if finger_angle == 1:
            self.close_gripper(is_suction = 1)
        elif finger_angle != 1 and finger_angle != 0:
            self.close_gripper(is_suction = -1, rotate_angle_1 = finger_angle, rotate_angle_2 = np.pi/2.5 - finger_angle)
               
        #time.sleep(10)
        if self.is_sim:                                                                                                                  
            # Compute tool orientation from heightmap rotation angle            
            if heightmap_rotation_angle < np.pi/4:
                self.suction_used_id = 0
                tool_rotation_angle = heightmap_rotation_angle
            elif heightmap_rotation_angle < 3*np.pi/4:
                self.suction_used_id = 1
                tool_rotation_angle = heightmap_rotation_angle - np.pi/2
            elif heightmap_rotation_angle < 5*np.pi/4:
                self.suction_used_id = 2
                tool_rotation_angle = heightmap_rotation_angle - np.pi
            elif heightmap_rotation_angle < 7*np.pi/4:
                self.suction_used_id = 3
                tool_rotation_angle = heightmap_rotation_angle - 3*np.pi/2
            else:
                self.suction_used_id = 0
                tool_rotation_angle = heightmap_rotation_angle - 2*np.pi
                                      
            position = np.asarray(position).copy()                        
            suction_point_margin = 0.2
            location_above_suction_point = (position[0], position[1], position[2] + suction_point_margin)

            # Compute gripper position and linear movement increments
            tool_position = location_above_suction_point
            sim_ret, grasp_target_position = vrep.simxGetObjectPosition(self.sim_client, self.grasp_target_handle,-1,vrep.simx_opmode_blocking)
            move_direction = np.asarray([tool_position[0] - grasp_target_position[0], tool_position[1] - grasp_target_position[1], tool_position[2] - grasp_target_position[2]])
            move_magnitude = np.linalg.norm(move_direction)
            move_step = 0.02*move_direction/move_magnitude
            num_move_steps = int(np.floor(move_direction[0]/move_step[0]))
            # Compute gripper orientation and rotation increments
            sim_ret, grasp_orientation = vrep.simxGetObjectOrientation(self.sim_client, self.grasp_target_handle, -1, vrep.simx_opmode_blocking)
            rotation_step = 0.1 if (tool_rotation_angle - grasp_orientation[2] > 0) else -0.1
            num_rotation_steps = int(np.floor((tool_rotation_angle - grasp_orientation[2])/rotation_step))
         
            sim_ret, suction_orientation = vrep.simxGetObjectOrientation(self.sim_client, self.suction_tip_handle[self.suction_used_id][0], self.grasp_target_handle, vrep.simx_opmode_blocking)           
            self.ro_angle = np.zeros((3,))
            if self.suction_used_id == 0:
                self.ro_angle[1] = np.pi/2 - suction_orientation[1]
            elif self.suction_used_id == 1:
                self.ro_angle[0] = -np.pi/2 - suction_orientation[0]
            elif self.suction_used_id == 2:
                self.ro_angle[1] = -np.pi/2 - suction_orientation[1]
            elif self.suction_used_id == 3:
                self.ro_angle[0] = np.pi/2 - suction_orientation[0]            
            self.ro_steps = 10
           
            # Simultaneously move and rotate gripper
            for step_iter in range(num_move_steps):
                vrep.simxSetObjectPosition(self.sim_client,self.grasp_target_handle,-1,(grasp_target_position[0] + move_step[0]*step_iter, grasp_target_position[1] + move_step[1]*step_iter, grasp_target_position[2] + move_step[2]*step_iter),vrep.simx_opmode_blocking)              
            vrep.simxSetObjectPosition(self.sim_client,self.grasp_target_handle,-1,(tool_position[0],tool_position[1],tool_position[2]),vrep.simx_opmode_blocking)            
            
            #time.sleep(10)
            for step_iter in range(num_rotation_steps):                
                vrep.simxSetObjectOrientation(self.sim_client, self.grasp_target_handle, -1, (0, 0, grasp_orientation[2] + rotation_step*step_iter), vrep.simx_opmode_blocking)           
            vrep.simxSetObjectOrientation(self.sim_client, self.grasp_target_handle, -1, (0, 0, tool_rotation_angle), vrep.simx_opmode_blocking)
                        
            sim_ret, grasp_orientation = vrep.simxGetObjectOrientation(self.sim_client, self.grasp_target_handle, self.suction_tip_handle[self.suction_used_id][0], vrep.simx_opmode_blocking)            
            
            for ro_iter in range(self.ro_steps):
                vrep.simxSetObjectOrientation(self.sim_client,self.grasp_target_handle,self.suction_tip_handle[self.suction_used_id][0],(grasp_orientation[0] + self.ro_angle[0]/self.ro_steps, grasp_orientation[1] + self.ro_angle[1]/self.ro_steps, grasp_orientation[2] + self.ro_angle[2]/self.ro_steps),vrep.simx_opmode_blocking)
            #vrep.simxSetObjectOrientation(self.sim_client,self.grasp_target_handle,self.suction_tip_handle[0][0],(grasp_orientation[0] + ro_angle,0,0),vrep.simx_opmode_blocking)                                                   
            #time.sleep(10)                        

            # # Rotate the suction to parallel to the plate            
            # tip_rotation_angle = np.pi/2 - tip_angle
            # self.ro_steps = 20
            # sim_ret, grasp_orientation = vrep.simxGetObjectOrientation(self.sim_client, self.grasp_target_handle, self.suction_tip_handle[0][0], vrep.simx_opmode_blocking)                                               
            # # for ro_iter in range(self.ro_steps):
            # #     vrep.simxSetObjectOrientation(self.sim_client,self.grasp_target_handle,self.suction_tip_handle[0][0],(grasp_orientation[0] + tip_rotation_angle/self.ro_steps,0,0),vrep.simx_opmode_blocking)
            # vrep.simxSetObjectOrientation(self.sim_client,self.grasp_target_handle,self.suction_tip_handle[0][0],(grasp_orientation[0],0,0),vrep.simx_opmode_blocking)                                                   
            
          
            # sim, suction_orientation = vrep.simxGetObjectOrientation(self.sim_client, self.suction_tip_handle[0][0], -1, vrep.simx_opmode_blocking)
            # tip_rotation_angle = heightmap_rotation_angle - suction_orientation
            # # print('suction_orientation',suction_orientation)
            # # print('tip_rotation_angle',tip_rotation_angle)                                                            
            # self.ro_steps = 20
            # sim_ret, grasp_orientation = vrep.simxGetObjectOrientation(self.sim_client, self.grasp_target_handle, -1, vrep.simx_opmode_blocking)                                               
            # print('grasp_orientation',grasp_orientation)
            # # for ro_iter in range(self.ro_steps):
            # #     vrep.simxSetObjectOrientation(self.sim_client,self.grasp_target_handle,self.suction_tip_handle[0][0],(grasp_orientation[0] + tip_rotation_angle/self.ro_steps,0,0),vrep.simx_opmode_blocking)
            # vrep.simxSetObjectOrientation(self.sim_client,self.grasp_target_handle,-1,(grasp_orientation[0] + tip_rotation_angle[0], grasp_orientation[1] + tip_rotation_angle[1], grasp_orientation[2] + 0),vrep.simx_opmode_blocking)                                                   
            # # time.sleep(2)
            # # for ro_iter in range(self.ro_steps):
            # #     vrep.simxSetObjectOrientation(self.sim_client,self.grasp_target_handle,-1,(grasp_orientation[0] + tip_rotation_angle[0], grasp_orientation[1] + tip_rotation_angle[1], grasp_orientation[2] + 1.5/self.ro_steps),vrep.simx_opmode_blocking) 
            # sim_ret, grasp_orientation = vrep.simxGetObjectOrientation(self.sim_client, self.grasp_target_handle, self.suction_tip_handle[0][0], vrep.simx_opmode_blocking)                                               
            # print('grasp_orientation',grasp_orientation)
            # sim, suction_orientation = vrep.simxGetObjectOrientation(self.sim_client, self.suction_tip_handle[0][0], -1, vrep.simx_opmode_blocking)
            # print('suction_orientation',suction_orientation)
                                    
            
            #time.sleep(10)
            self.suction_active(is_active=False, suction_id=-1)            
            # Approach suction point
            self.move_to(position, None, suction_id=self.suction_used_id)
            # sucking the target
            suction_success,sucction_handle = self.suction_active(is_active=True, suction_id = self.suction_used_id)
            # print('suction_success: %r' % (suction_success))
            if sucction_handle !=0:
                self.sucked_handles.append(sucction_handle)
             
            # # re-sucking
            # if not suction_success:
            #     time.sleep(3)
            #     motion_theta = np.radians(random.randrange(1,360))
            #     position0 = [0.005*np.cos(motion_theta),0.005*np.sin(motion_theta),0]
            #     self.move_to(position+position0, None, is_suction=True)
            #     self.suction_active(is_active=True)
            #time.sleep(10)
            # Move gripper to location above suction target
            self.move_to(location_above_suction_point, None, suction_id=-2)           
            # Compute gripper orientation and rotation increments
            
            sim_ret, grasp_orientation = vrep.simxGetObjectOrientation(self.sim_client, self.grasp_target_handle, -1, vrep.simx_opmode_blocking)            
            rotation_step0 = grasp_orientation[0]/self.ro_steps
            rotation_step1 = grasp_orientation[1]/self.ro_steps
            # rotate gripper
            for step_iter in range(self.ro_steps):                
                vrep.simxSetObjectOrientation(self.sim_client, self.grasp_target_handle, -1, (grasp_orientation[0]-step_iter*rotation_step0, grasp_orientation[1]-step_iter*rotation_step1, grasp_orientation[2]), vrep.simx_opmode_blocking)            
            vrep.simxSetObjectOrientation(self.sim_client, self.grasp_target_handle, -1, (0, 0, grasp_orientation[2]), vrep.simx_opmode_blocking)

        return suction_success





