import struct
import math
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import time
import matplotlib.pyplot as plt

def get_pointcloud(color_img, depth_img, camera_intrinsics): 

    # Get depth image size
    im_h = depth_img.shape[0]
    im_w = depth_img.shape[1]
    # Project depth into 3D point cloud in camera coordinates
    pix_x,pix_y = np.meshgrid(np.linspace(0,im_w-1,im_w), np.linspace(0,im_h-1,im_h))
    cam_pts_x = np.multiply(pix_x-camera_intrinsics[0][2],depth_img/camera_intrinsics[0][0])    
    cam_pts_y = np.multiply(pix_y-camera_intrinsics[1][2],depth_img/camera_intrinsics[1][1])    
    cam_pts_z = depth_img.copy()   
    cam_pts_x.shape = (im_h*im_w,1)
    cam_pts_y.shape = (im_h*im_w,1)
    cam_pts_z.shape = (im_h*im_w,1)
    # Reshape image into colors for 3D point cloud
    rgb_pts_r = color_img[:,:,0]
    rgb_pts_g = color_img[:,:,1]
    rgb_pts_b = color_img[:,:,2]
    rgb_pts_r.shape = (im_h*im_w,1)
    rgb_pts_g.shape = (im_h*im_w,1)
    rgb_pts_b.shape = (im_h*im_w,1)
    cam_pts = np.concatenate((cam_pts_x, cam_pts_y, cam_pts_z), axis=1)
    rgb_pts = np.concatenate((rgb_pts_r, rgb_pts_g, rgb_pts_b), axis=1)

    return cam_pts, rgb_pts


def get_heightmap(color_img, depth_img, cam_intrinsics, cam_pose, workspace_limits, heightmap_resolution):

    # Compute heightmap size
    heightmap_size = (224,224)
    colormask_size = (448,448)
    color_h, color_w = color_img.shape[:2]
    # Get 3D point cloud from RGB-D images
    surface_pts, color_pts = get_pointcloud(color_img, depth_img, cam_intrinsics)
    # Transform 3D point cloud from camera coordinates to robot coordinates
    surface_pts = np.transpose(np.dot(cam_pose[0:3,0:3],np.transpose(surface_pts)) + np.tile(cam_pose[0:3,3:],(1,surface_pts.shape[0])))    
    # sim
    x1,x2,x3,x4 = (110,110,510,510)
    y1,y2,y3,y4 = (0,400,400,0)    
    # real-experiment
    # x1,x2,x3,x4 = (105,105,455,455)
    # y1,y2,y3,y4 = (80,430,430,80)
        
    src = np.array([[x1,y1],  [x2,y2],  [x3,y3], [x4,y4]], np.float32)
    dst_heightmap = np.array([[0,0],[0,heightmap_size[0]],[heightmap_size[1],heightmap_size[0]], [heightmap_size[1],0]], np.float32)
    dst_mask = np.array([[0,0],[0,colormask_size[0]],[colormask_size[1],colormask_size[0]],[colormask_size[1],0]], np.float32)
    A_heightmap = cv2.getPerspectiveTransform(src, dst_heightmap)
    A_mask = cv2.getPerspectiveTransform(src, dst_mask)
    worldcor_depthimg = surface_pts[:,2]
    worldcor_depthimg.shape = (480,640)
    color_heightmap = cv2.warpPerspective(color_img, A_heightmap, heightmap_size)
    depth_heightmap = cv2.warpPerspective(worldcor_depthimg, A_heightmap, heightmap_size)
    color_mask = cv2.warpPerspective(color_img, A_mask, colormask_size)
    depth_mask = cv2.warpPerspective(worldcor_depthimg, A_mask, colormask_size)
    A_htor = cv2.getPerspectiveTransform(dst_heightmap, src)

    return color_heightmap, depth_heightmap, color_mask, depth_mask, A_htor

def global_position(pix_mask_position, A_htor, cam_intrinsics, cam_pose, depth_img):

    pix_mask_x = int((pix_mask_position[2]*A_htor[0,0]+pix_mask_position[1]*A_htor[0,1]+A_htor[0,2])/(pix_mask_position[2]*A_htor[2,0]+pix_mask_position[1]*A_htor[2,1]+A_htor[2,2]))
    pix_mask_y = int((pix_mask_position[2]*A_htor[1,0]+pix_mask_position[1]*A_htor[1,1]+A_htor[1,2])/(pix_mask_position[2]*A_htor[2,0]+pix_mask_position[1]*A_htor[2,1]+A_htor[2,2]))
    cam_pts_z = depth_img[pix_mask_y][pix_mask_x]
    cam_pts_x = np.multiply(pix_mask_x-cam_intrinsics[0][2],cam_pts_z/cam_intrinsics[0][0])    
    cam_pts_y = np.multiply(pix_mask_y-cam_intrinsics[1][2],cam_pts_z/cam_intrinsics[1][1]) 
    cam_pts = np.asarray([[cam_pts_x, cam_pts_y, cam_pts_z]])
    robot_cor = np.transpose(np.dot(cam_pose[0:3,0:3],np.transpose(cam_pts)) + np.tile(cam_pose[0:3,3:],(1,cam_pts.shape[0])))
    robot_cor.shape = (3)

    return robot_cor


# Save a 3D point cloud to a binary .ply file
def pcwrite(xyz_pts, filename, rgb_pts=None):
    assert xyz_pts.shape[1] == 3, 'input XYZ points should be an Nx3 matrix'
    if rgb_pts is None:
        rgb_pts = np.ones(xyz_pts.shape).astype(np.uint8)*255
    assert xyz_pts.shape == rgb_pts.shape, 'input RGB colors should be Nx3 matrix and same size as input XYZ points'

    # Write header for .ply file
    pc_file = open(filename, 'wb')
    pc_file.write(bytearray('ply\n', 'utf8'))
    pc_file.write(bytearray('format binary_little_endian 1.0\n', 'utf8'))
    pc_file.write(bytearray(('element vertex %d\n' % xyz_pts.shape[0]), 'utf8'))
    pc_file.write(bytearray('property float x\n', 'utf8'))
    pc_file.write(bytearray('property float y\n', 'utf8'))
    pc_file.write(bytearray('property float z\n', 'utf8'))
    pc_file.write(bytearray('property uchar red\n', 'utf8'))
    pc_file.write(bytearray('property uchar green\n', 'utf8'))
    pc_file.write(bytearray('property uchar blue\n', 'utf8'))
    pc_file.write(bytearray('end_header\n', 'utf8'))

    # Write 3D points to .ply file
    for i in range(xyz_pts.shape[0]):
        pc_file.write(bytearray(struct.pack("fffccc",xyz_pts[i][0],xyz_pts[i][1],xyz_pts[i][2],rgb_pts[i][0].tostring(),rgb_pts[i][1].tostring(),rgb_pts[i][2].tostring())))
    pc_file.close()


def get_affordance_vis(grasp_affordances, input_images, num_rotations, best_pix_ind):
    vis = None
    for vis_row in range(num_rotations/4):
        tmp_row_vis = None
        for vis_col in range(4):
            rotate_idx = vis_row*4+vis_col
            affordance_vis = grasp_affordances[rotate_idx,:,:]
            affordance_vis[affordance_vis < 0] = 0 # assume probability
            # affordance_vis = np.divide(affordance_vis, np.max(affordance_vis))
            affordance_vis[affordance_vis > 1] = 1 # assume probability
            affordance_vis.shape = (grasp_affordances.shape[1], grasp_affordances.shape[2])
            affordance_vis = cv2.applyColorMap((affordance_vis*255).astype(np.uint8), cv2.COLORMAP_JET)
            input_image_vis = (input_images[rotate_idx,:,:,:]*255).astype(np.uint8)
            input_image_vis = cv2.resize(input_image_vis, (0,0), fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
            affordance_vis = (0.5*cv2.cvtColor(input_image_vis, cv2.COLOR_RGB2BGR) + 0.5*affordance_vis).astype(np.uint8)
            if rotate_idx == best_pix_ind[0]:
                affordance_vis = cv2.circle(affordance_vis, (int(best_pix_ind[2]), int(best_pix_ind[1])), 7, (0,0,255), 2)
            if tmp_row_vis is None:
                tmp_row_vis = affordance_vis
            else:
                tmp_row_vis = np.concatenate((tmp_row_vis,affordance_vis), axis=1)
        if vis is None:
            vis = tmp_row_vis
        else:
            vis = np.concatenate((vis,tmp_row_vis), axis=0)

    return vis


def get_difference(color_heightmap, color_space, bg_color_heightmap):

    color_space = np.concatenate((color_space, np.asarray([[0.0, 0.0, 0.0]])), axis=0)
    color_space.shape = (color_space.shape[0], 1, 1, color_space.shape[1])
    color_space = np.tile(color_space, (1, color_heightmap.shape[0], color_heightmap.shape[1], 1))

    # Normalize color heightmaps
    color_heightmap = color_heightmap.astype(float)/255.0
    color_heightmap.shape = (1, color_heightmap.shape[0], color_heightmap.shape[1], color_heightmap.shape[2])
    color_heightmap = np.tile(color_heightmap, (color_space.shape[0], 1, 1, 1))
    bg_color_heightmap = bg_color_heightmap.astype(float)/255.0
    bg_color_heightmap.shape = (1, bg_color_heightmap.shape[0], bg_color_heightmap.shape[1], bg_color_heightmap.shape[2])
    bg_color_heightmap = np.tile(bg_color_heightmap, (color_space.shape[0], 1, 1, 1))

    # Compute nearest neighbor distances to key colors
    key_color_dist = np.sqrt(np.sum(np.power(color_heightmap - color_space,2), axis=3))
    # key_color_dist_prob = F.softmax(Variable(torch.from_numpy(key_color_dist), volatile=True), dim=0).data.numpy()

    bg_key_color_dist = np.sqrt(np.sum(np.power(bg_color_heightmap - color_space,2), axis=3))
    # bg_key_color_dist_prob = F.softmax(Variable(torch.from_numpy(bg_key_color_dist), volatile=True), dim=0).data.numpy()

    key_color_match = np.argmin(key_color_dist, axis=0)
    bg_key_color_match = np.argmin(bg_key_color_dist, axis=0)
    key_color_match[key_color_match == color_space.shape[0] - 1] = color_space.shape[0] + 1
    bg_key_color_match[bg_key_color_match == color_space.shape[0] - 1] = color_space.shape[0] + 2

    return np.sum(key_color_match == bg_key_color_match).astype(float)/np.sum(bg_key_color_match < color_space.shape[0]).astype(float)


# Get rotation matrix from euler angles
def euler2rotm(theta):
    R_x = np.array([[1,         0,                  0                   ],
                    [0,         math.cos(theta[0]), -math.sin(theta[0]) ],
                    [0,         math.sin(theta[0]), math.cos(theta[0])  ]
                    ])
    R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  ],
                    [0,                     1,      0                   ],
                    [-math.sin(theta[1]),   0,      math.cos(theta[1])  ]
                    ])         
    R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
                    [math.sin(theta[2]),    math.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])            
    R = np.dot(R_z, np.dot( R_y, R_x ))
    return R


# Checks if a matrix is a valid rotation matrix.
def isRotm(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6
 
 
# Calculates rotation matrix to euler angles
def rotm2euler(R) :
 
    assert(isRotm(R))

    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    singular = sy < 1e-6

    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
 
    return np.array([x, y, z])


def angle2rotm(angle, axis, point=None):
    # Copyright (c) 2006-2018, Christoph Gohlke

    sina = math.sin(angle)
    cosa = math.cos(angle)
    axis = axis/np.linalg.norm(axis)

    # Rotation matrix around unit vector
    R = np.diag([cosa, cosa, cosa])
    R += np.outer(axis, axis) * (1.0 - cosa)
    axis *= sina
    R += np.array([[ 0.0,     -axis[2],  axis[1]],
                      [ axis[2], 0.0,      -axis[0]],
                      [-axis[1], axis[0],  0.0]])
    M = np.identity(4)
    M[:3, :3] = R
    if point is not None:

        # Rotation not around origin
        point = np.array(point[:3], dtype=np.float64, copy=False)
        M[:3, 3] = point - np.dot(R, point)
    return M


def rotm2angle(R):
    # From: euclideanspace.com

    epsilon = 0.01 # Margin to allow for rounding errors
    epsilon2 = 0.1 # Margin to distinguish between 0 and 180 degrees

    assert(isRotm(R))

    if ((abs(R[0][1]-R[1][0])< epsilon) and (abs(R[0][2]-R[2][0])< epsilon) and (abs(R[1][2]-R[2][1])< epsilon)):
        # Singularity found
        # First check for identity matrix which must have +1 for all terms in leading diagonaland zero in other terms
        if ((abs(R[0][1]+R[1][0]) < epsilon2) and (abs(R[0][2]+R[2][0]) < epsilon2) and (abs(R[1][2]+R[2][1]) < epsilon2) and (abs(R[0][0]+R[1][1]+R[2][2]-3) < epsilon2)):
            # this singularity is identity matrix so angle = 0
            return [0,1,0,0] # zero angle, arbitrary axis

        # Otherwise this singularity is angle = 180
        angle = np.pi
        xx = (R[0][0]+1)/2
        yy = (R[1][1]+1)/2
        zz = (R[2][2]+1)/2
        xy = (R[0][1]+R[1][0])/4
        xz = (R[0][2]+R[2][0])/4
        yz = (R[1][2]+R[2][1])/4
        if ((xx > yy) and (xx > zz)): # R[0][0] is the largest diagonal term
            if (xx< epsilon):
                x = 0
                y = 0.7071
                z = 0.7071
            else:
                x = np.sqrt(xx)
                y = xy/x
                z = xz/x
        elif (yy > zz): # R[1][1] is the largest diagonal term
            if (yy< epsilon):
                x = 0.7071
                y = 0
                z = 0.7071
            else:
                y = np.sqrt(yy)
                x = xy/y
                z = yz/y
        else: # R[2][2] is the largest diagonal term so base result on this
            if (zz< epsilon):
                x = 0.7071
                y = 0.7071
                z = 0
            else:
                z = np.sqrt(zz)
                x = xz/z
                y = yz/z
        return [angle,x,y,z] # Return 180 deg rotation

    # As we have reached here there are no singularities so we can handle normally
    s = np.sqrt((R[2][1] - R[1][2])*(R[2][1] - R[1][2]) + (R[0][2] - R[2][0])*(R[0][2] - R[2][0]) + (R[1][0] - R[0][1])*(R[1][0] - R[0][1])) # used to normalise
    if (abs(s) < 0.001):
        s = 1 

    # Prevent divide by zero, should not happen if matrix is orthogonal and should be
    # Caught by singularity test above, but I've left it in just in case
    angle = np.arccos(( R[0][0] + R[1][1] + R[2][2] - 1)/2)
    x = (R[2][1] - R[1][2])/s
    y = (R[0][2] - R[2][0])/s
    z = (R[1][0] - R[0][1])/s
    return [angle,x,y,z]


# Cross entropy loss for 2D outputs
class CrossEntropyLoss2d(nn.Module):

    def __init__(self, weight=None, size_average=True):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss2d(weight, size_average)

    def forward(self, inputs, targets):
        return self.nll_loss(F.log_softmax(inputs, dim=1), targets)


def get_best_grasp_angle(is_pe, box_mask_cors, bestg_id, A_htor, cam_intrinsics, cam_pose, depth_img):
    # grasp_center
    grasp_center_pix = [0,
                        (box_mask_cors[bestg_id[0]][0][1] + box_mask_cors[bestg_id[0]][1][1] + box_mask_cors[bestg_id[0]][2][1] + box_mask_cors[bestg_id[0]][3][1])/4,
                        (box_mask_cors[bestg_id[0]][0][0] + box_mask_cors[bestg_id[0]][1][0] + box_mask_cors[bestg_id[0]][2][0] + box_mask_cors[bestg_id[0]][3][0])/4]
    grasp_center_pix = np.array(grasp_center_pix).astype(int)
    grasp_center_cor = global_position(grasp_center_pix, A_htor, cam_intrinsics, cam_pose, depth_img)
    
    
    # grasp angle and open distance        
    grasp_rotation_angle = 0
    grasp_open_distance = 2. # larger than the threshold
    if is_pe:
        box_glob_g_cors, g_angle_selected_box = np.zeros((4,3)), np.zeros((2,3))                                        
        for i in range(4):
            box_pix_g = [0,box_mask_cors[bestg_id[0]][i][1],box_mask_cors[bestg_id[0]][i][0]]
            box_pix_g = np.array(box_pix_g).astype(int)                    
            box_glob_g_cors[i] = global_position(box_pix_g, A_htor, cam_intrinsics, cam_pose, depth_img)                       
        
        g_id_distance_01 = math.sqrt((box_glob_g_cors[0][0] - box_glob_g_cors[1][0])**2 + (box_glob_g_cors[0][1] - box_glob_g_cors[1][1])**2)    
        g_id_distance_12 = math.sqrt((box_glob_g_cors[2][0] - box_glob_g_cors[1][0])**2 + (box_glob_g_cors[2][1] - box_glob_g_cors[1][1])**2)      
        
        if g_id_distance_01 > g_id_distance_12:
            grasp_open_distance = g_id_distance_12 *min(1.2,g_id_distance_01/g_id_distance_12)
            if box_glob_g_cors[0][1] == box_glob_g_cors[1][1]:
                g_angle_selected_box[0][2] = 0
            elif box_glob_g_cors[0][1] > box_glob_g_cors[1][1]:
                g_angle_selected_box[0][2] = math.acos((box_glob_g_cors[0][0]-box_glob_g_cors[1][0])/g_id_distance_01)
            else:
                g_angle_selected_box[0][2] = math.acos((box_glob_g_cors[1][0]-box_glob_g_cors[0][0])/g_id_distance_01)    
            
        else:
            grasp_open_distance = g_id_distance_01 *min(1.2,g_id_distance_12/g_id_distance_01)
            if box_glob_g_cors[2][1] == box_glob_g_cors[1][1]:
                g_angle_selected_box[0][2] = 0
            elif box_glob_g_cors[2][1] > box_glob_g_cors[1][1]:
                g_angle_selected_box[0][2] = math.acos((box_glob_g_cors[2][0]-box_glob_g_cors[1][0])/g_id_distance_12)
            else:
                g_angle_selected_box[0][2] = math.acos((box_glob_g_cors[1][0]-box_glob_g_cors[2][0])/g_id_distance_12)
                
        grasp_rotation_angle = g_angle_selected_box[0][2]
        
        # ax = plt.subplot(111,projection='polar')
        # ax.set_theta_offset(np.pi/2)
        # ax.set_thetagrids(np.arange(0.,360.,30.))                
        # selected_r = np.arange(0,1,0.01)
        # selected_t = np.ones((100,))*g_angle_selected_box[0][2]
        # ax.plot(selected_t, selected_r, linewidth=3, color='red')
        # plt.show() 

    return grasp_center_cor, grasp_rotation_angle, grasp_open_distance



def get_best_suction_angle(is_oo, objects_number, masks_cter, box_mask_cors, bests_id, A_htor, cam_intrinsics, cam_pose, depth_img):
                
    # robot cors of the sucking center
    suction_center_pix = [0,
                        (box_mask_cors[bests_id[0]][0][1] +box_mask_cors[bests_id[0]][1][1]+box_mask_cors[bests_id[0]][2][1]+box_mask_cors[bests_id[0]][3][1])/4,
                        (box_mask_cors[bests_id[0]][0][0] +box_mask_cors[bests_id[0]][1][0]+box_mask_cors[bests_id[0]][2][0]+box_mask_cors[bests_id[0]][3][0])/4]
    suction_center_pix = np.array(suction_center_pix).astype(int)
    suction_center_cor = global_position(suction_center_pix, A_htor, cam_intrinsics, cam_pose, depth_img)        
    
    suction_rotation_angle = 0
    if is_oo:   
        angle_val = np.ones((360,))
        object_val = np.ones((objects_number,3))
        box_glob_s_cors, center_glob_s_cors, height_s_aver, distance_s_aver = np.zeros((objects_number,4,3)),np.zeros((objects_number,3)),np.zeros((objects_number)),np.zeros((objects_number))
        # average height/distance of each object
        for i in range(objects_number):
            center_glob_s_cors[i] = global_position([0,masks_cter[i,1],masks_cter[i,0]], A_htor, cam_intrinsics, cam_pose, depth_img)                                        
            for j in range(4):
                box_pix_s = [0,box_mask_cors[i][j][1],box_mask_cors[i][j][0]]
                box_pix_s = np.array(box_pix_s).astype(int)                    
                box_glob_s_cors[i][j] = global_position(box_pix_s, A_htor, cam_intrinsics, cam_pose, depth_img)               
        for i in range(objects_number):
            height_s_aver[i] = max(center_glob_s_cors[i][2], box_glob_s_cors[i][0][2], box_glob_s_cors[i][1][2], box_glob_s_cors[i][2][2], box_glob_s_cors[i][3][2])
            distance_s_aver[i] = math.sqrt((center_glob_s_cors[i][0] - center_glob_s_cors[bests_id[0]][0])**2 + 
                                           (center_glob_s_cors[i][1] - center_glob_s_cors[bests_id[0]][1])**2)                
       
        # open angle 
        for object_id in range(objects_number):
            if object_id != bests_id[0]:
                angle_point = np.zeros((4))
                box_points = np.zeros((4,2))
                box_points[0] = [box_mask_cors[object_id][0][0],box_mask_cors[object_id][0][1]]
                box_points[1] = [box_mask_cors[object_id][1][0],box_mask_cors[object_id][1][1]]
                box_points[2] = [box_mask_cors[object_id][2][0],box_mask_cors[object_id][2][1]]
                box_points[3] = [box_mask_cors[object_id][3][0],box_mask_cors[object_id][3][1]]
                for points_num in range(4):                        
                    if box_points[points_num][0] == masks_cter[bests_id[0]][0]:
                        if box_points[points_num][1] > masks_cter[bests_id[0]][1]:
                            angle_point[points_num] = np.pi
                        else:
                            angle_point[points_num] = 0
                    if box_points[points_num][1] == masks_cter[bests_id[0]][1]:
                        if box_points[points_num][0] < masks_cter[bests_id[0]][0]:
                            angle_point[points_num] = np.pi/2
                        else:
                            angle_point[points_num] = 3*np.pi/2
                    if box_points[points_num][0] < masks_cter[bests_id[0]][0]:
                        if box_points[points_num][1] < masks_cter[bests_id[0]][1]:
                            angle_point[points_num] = math.atan((masks_cter[bests_id[0]][0] - box_points[points_num][0])/(masks_cter[bests_id[0]][1] - box_points[points_num][1]))
                        elif box_points[points_num][1] > masks_cter[bests_id[0]][1]:
                            angle_point[points_num] = np.pi/2 + math.atan((box_points[points_num][1] - masks_cter[bests_id[0]][1])/(masks_cter[bests_id[0]][0] - box_points[points_num][0]))                   
                    if box_points[points_num][0] > masks_cter[bests_id[0]][0]:
                        if box_points[points_num][1] < masks_cter[bests_id[0]][1]:
                            angle_point[points_num] = 3*np.pi/2 + math.atan((masks_cter[bests_id[0]][1] - box_points[points_num][1])/(box_points[points_num][0] - masks_cter[bests_id[0]][0]))
                        elif box_points[points_num][1] > masks_cter[bests_id[0]][1]:
                            angle_point[points_num] = np.pi + math.atan((box_points[points_num][0] - masks_cter[bests_id[0]][0])/(box_points[points_num][1] - masks_cter[bests_id[0]][1]))               
                angle_max = 0
                for i in range(3):
                    for j in range(i+1,4,1):
                        angle_diff = min(abs(angle_point[i] - angle_point[j]), 2*np.pi - abs(angle_point[i] - angle_point[j]))
                        if angle_diff > angle_max:
                            angle_max = angle_diff
                            object_val[object_id][0] = min(angle_point[i], angle_point[j])
                            object_val[object_id][1] = max(angle_point[i], angle_point[j])                          
                               
        
        for i in range(objects_number):
            h_aver = max(0.,height_s_aver[i]-height_s_aver[bests_id[0]])
            d_aver = max(0.001, distance_s_aver[i])
            object_val[i][2] = math.exp(-h_aver/d_aver)
        
        for i in range(objects_number):
            if i != bests_id[0] and object_val[i][2] != 1.:
                angle_0, angle_1 = int(180*object_val[i][0]/np.pi), int(180*object_val[i][1]/np.pi)
                if abs(object_val[i][0] - object_val[i][1]) <= np.pi:
                   for angle_id in range(angle_0,angle_1):
                       angle_val[angle_id] = angle_val[angle_id] * object_val[i][2]
                else:
                    for angle_id in range(angle_0):
                        angle_val[angle_id] = angle_val[angle_id] * object_val[i][2]
                    for angle_id in range(angle_1,360):
                        angle_val[angle_id] = angle_val[angle_id] * object_val[i][2]                                                 
        start_id, end_id, value_id, value_box, angle_box = 0, 0, angle_val[0], [], []
        for i in range(len(angle_val)):
            if angle_val[i] != value_id:
                end_id = i-1
                value_box.append(value_id)
                angle_box.append([start_id, end_id])
                value_id, start_id = angle_val[i], i
            if i == (len(angle_val)-1) and start_id != i:
                value_box.append(value_id)
                angle_box.append([start_id, i])                
        value_threshold, angle_threshold = 0.95, 45                    
        a_object_val, a_angle_val, a_value_box, a_angle_box = object_val.copy(), angle_val.copy(), value_box.copy(), angle_box.copy()
        object_val_set = list(set(a_object_val[:,2]))
        object_val_set.append(1.)
        object_val_set = list(set(object_val_set))
        pre_sorted = list(np.argsort(object_val_set))
        pre_sorted.reverse()
        s_angle_selected = []
        for val_id in range(len(object_val_set)):
            if min(a_value_box) >= value_threshold:
                s_angle_selected.append(0.)                    
            if len(s_angle_selected) == 0:
                value_selected, s_angle_selected_box = 1., []                       
                if a_angle_val[1] == a_angle_val[359] and a_value_box[0] >= value_selected:
                    angle_left = a_angle_box[0][1]
                    angle_right = a_angle_box[len(a_angle_box)-1][1] - a_angle_box[len(a_angle_box)-1][0]
                    if (angle_left + angle_right) >= angle_threshold:
                        if angle_left > angle_right:
                            #s_angle_selected.append(min(90, angle_left - int((angle_left + angle_right)/2)))
                            s_angle_selected.append(angle_left - int((angle_left + angle_right)/2))
                        else:
                            #s_angle_selected.append(max(270, a_angle_box[len(a_angle_box)-1][0] + int((angle_left + angle_right)/2)))
                            s_angle_selected.append(a_angle_box[len(a_angle_box)-1][0] + int((angle_left + angle_right)/2))
                if len(s_angle_selected) == 0:
                    for i in range(len(a_value_box)):
                        if a_value_box[i] >= value_selected and (a_angle_box[i][1] - a_angle_box[i][0]) >= angle_threshold:
                            s_angle_selected_box.append([a_angle_box[i][0], a_angle_box[i][1], a_angle_box[i][1] - a_angle_box[i][0], int((a_angle_box[i][0] + a_angle_box[i][1])/2)])
                    if len(s_angle_selected_box) > 0:
                        s_angle_selected_box = np.asarray(s_angle_selected_box)
                        angle_sorted = list(np.argsort(s_angle_selected_box[:,2]))
                        angle_sorted.reverse()
                        
                        s_angle_selected.append(s_angle_selected_box[angle_sorted[0]][3])
                                                                
                        # for ang_num in range(len(angle_sorted)):
                        #     sorted_num = angle_sorted[ang_num]
                        #     if s_angle_selected_box[sorted_num][3] < 135 or s_angle_selected_box[sorted_num][3] > 225:
                        #         s_angle_selected.append(s_angle_selected_box[sorted_num][3])
                        #     elif s_angle_selected_box[sorted_num][0] < 112.5 or s_angle_selected_box[sorted_num][1] > 247.5:
                        #         if (180 - s_angle_selected_box[sorted_num][0]) > (s_angle_selected_box[sorted_num][1] - 180):
                        #             s_angle_selected.append(max(90, s_angle_selected_box[sorted_num][0] + int(22.5)))
                        #         else:
                        #             s_angle_selected.append(min(270, s_angle_selected_box[sorted_num][1] - int(22.5)))
                        #     if len(s_angle_selected) != 0:
                        #         break
                        # if len(s_angle_selected) == 0:
                        #     slected_angle, selected_id = abs(s_angle_selected_box[0][3] - 180), 0
                        #     for ang_num in range(len(s_angle_selected_box)):
                        #         if abs(s_angle_selected_box[ang_num][3] - 180) > slected_angle:
                        #             slected_angle, selected_id = abs(s_angle_selected_box[ang_num][3] - 180), ang_num
                        #     s_angle_selected.append(s_angle_selected_box[selected_id][3])
                    
                    if len(s_angle_selected) == 0:
                        for obj_num in range(objects_number):
                            if abs(a_object_val[obj_num][2] - object_val_set[pre_sorted[val_id+1]]) < 0.001:
                                a_object_val[obj_num][2] = 1.
                        
                        a_angle_val = np.ones((360,))
                        for i in range(objects_number):
                            if i != bests_id[0] and a_object_val[i][2] !=1.:
                                angle_0, angle_1 = int(180*a_object_val[i][0]/np.pi), int(180*a_object_val[i][1]/np.pi)
                                if abs(a_object_val[i][0] - a_object_val[i][1]) <= np.pi:
                                   for angle_id in range(angle_0,angle_1):
                                       a_angle_val[angle_id] = a_angle_val[angle_id] * a_object_val[i][2]
                                else:
                                    for angle_id in range(angle_0):
                                        a_angle_val[angle_id] = a_angle_val[angle_id] * a_object_val[i][2]
                                    for angle_id in range(angle_1,360):
                                        a_angle_val[angle_id] = a_angle_val[angle_id] * a_object_val[i][2]    
                        a_start_id, a_end_id, a_value_id, a_value_box, a_angle_box = 0, 0, a_angle_val[0], [], []
                        for i in range(len(a_angle_val)):
                            if a_angle_val[i] != a_value_id:
                                a_end_id = i-1
                                a_value_box.append(a_value_id)
                                a_angle_box.append([a_start_id, a_end_id])
                                a_value_id, a_start_id = a_angle_val[i], i
                            if i == (len(a_angle_val)-1) and a_start_id != i:
                                a_value_box.append(a_value_id)
                                a_angle_box.append([a_start_id, i])           
            if len(s_angle_selected) != 0:
                break                                         
        suction_rotation_angle = s_angle_selected[0]
        
        
        
        
        # plt.figure(dpi=500)
        # theta = np.linspace(0.0, 2*np.pi, 360, endpoint=False)
        # radii = angle_val*10
        # width = np.pi/180
        # ax = plt.subplot(111,projection='polar')
        # bars = ax.bar(theta,radii,width=width,bottom=10.)
        # ax.set_theta_offset(np.pi)
        # ax.set_thetagrids(np.arange(0.,360.,45.))
        
        # print('s_angle_selected',s_angle_selected)
        # selected_r = np.arange(0,20,0.1)
        # selected_t = np.pi*np.ones((200,))*s_angle_selected[0]/180
        # ax.plot(selected_t, selected_r, linewidth=3, color='red')
        
        # for r, bar in zip(radii,bars):
        #     bar.set_facecolor(plt.cm.viridis(r/11))
        #     bar.set_alpha(0.5)
        
        # plt.grid(ls='--')
        # #plt.grid(axis='y')
        # plt.yticks([])
        # plt.tick_params(axis='x',colors='red')   
        # plt.show() 
    
       
        # box_glob_s_cors, box_glob_s_norv = np.zeros((4,3)), np.zeros((4,3))
        # euler_angles = np.zeros((4,3))
        # for i in range(4):                    
        #     box_pix_s = [0,box_mask_cors[bests_id[0]][i][1],box_mask_cors[bests_id[0]][i][0]]
        #     box_pix_s = np.array(box_pix_s).astype(int)                    
        #     box_glob_s_cors[i] = utils.global_position(box_pix_s, A_htor, robot.cam_intrinsics, robot.cam_pose, depth_img)
        # box_glob_s_norv[0] = [(box_glob_s_cors[3][1]-box_glob_s_cors[0][1])*(box_glob_s_cors[1][2]-box_glob_s_cors[0][2]) - (box_glob_s_cors[1][1]-box_glob_s_cors[0][1])*(box_glob_s_cors[3][2]-box_glob_s_cors[0][2]),
        #                       (box_glob_s_cors[3][2]-box_glob_s_cors[0][2])*(box_glob_s_cors[1][0]-box_glob_s_cors[0][0]) - (box_glob_s_cors[1][2]-box_glob_s_cors[0][2])*(box_glob_s_cors[3][0]-box_glob_s_cors[0][0]),
        #                       (box_glob_s_cors[3][0]-box_glob_s_cors[0][0])*(box_glob_s_cors[1][1]-box_glob_s_cors[0][1]) - (box_glob_s_cors[1][0]-box_glob_s_cors[0][0])*(box_glob_s_cors[3][1]-box_glob_s_cors[0][1])]
        # box_glob_s_norv[1] = [(box_glob_s_cors[0][1]-box_glob_s_cors[1][1])*(box_glob_s_cors[2][2]-box_glob_s_cors[1][2]) - (box_glob_s_cors[2][1]-box_glob_s_cors[1][1])*(box_glob_s_cors[0][2]-box_glob_s_cors[1][2]),
        #                       (box_glob_s_cors[0][2]-box_glob_s_cors[1][2])*(box_glob_s_cors[2][0]-box_glob_s_cors[1][0]) - (box_glob_s_cors[2][2]-box_glob_s_cors[1][2])*(box_glob_s_cors[0][0]-box_glob_s_cors[1][0]),
        #                       (box_glob_s_cors[0][0]-box_glob_s_cors[1][0])*(box_glob_s_cors[2][1]-box_glob_s_cors[1][1]) - (box_glob_s_cors[2][0]-box_glob_s_cors[1][0])*(box_glob_s_cors[0][1]-box_glob_s_cors[1][1])]
        # box_glob_s_norv[2] = [(box_glob_s_cors[1][1]-box_glob_s_cors[2][1])*(box_glob_s_cors[3][2]-box_glob_s_cors[2][2]) - (box_glob_s_cors[3][1]-box_glob_s_cors[2][1])*(box_glob_s_cors[1][2]-box_glob_s_cors[2][2]),
        #                       (box_glob_s_cors[1][2]-box_glob_s_cors[2][2])*(box_glob_s_cors[3][0]-box_glob_s_cors[2][0]) - (box_glob_s_cors[3][2]-box_glob_s_cors[2][2])*(box_glob_s_cors[1][0]-box_glob_s_cors[2][0]),
        #                       (box_glob_s_cors[1][0]-box_glob_s_cors[2][0])*(box_glob_s_cors[3][1]-box_glob_s_cors[2][1]) - (box_glob_s_cors[3][0]-box_glob_s_cors[2][0])*(box_glob_s_cors[1][1]-box_glob_s_cors[2][1])]
        # box_glob_s_norv[3] = [(box_glob_s_cors[2][1]-box_glob_s_cors[3][1])*(box_glob_s_cors[0][2]-box_glob_s_cors[3][2]) - (box_glob_s_cors[0][1]-box_glob_s_cors[3][1])*(box_glob_s_cors[2][2]-box_glob_s_cors[3][2]),
        #                       (box_glob_s_cors[2][2]-box_glob_s_cors[3][2])*(box_glob_s_cors[0][0]-box_glob_s_cors[3][0]) - (box_glob_s_cors[0][2]-box_glob_s_cors[3][2])*(box_glob_s_cors[2][0]-box_glob_s_cors[3][0]),
        #                       (box_glob_s_cors[2][0]-box_glob_s_cors[3][0])*(box_glob_s_cors[0][1]-box_glob_s_cors[3][1]) - (box_glob_s_cors[0][0]-box_glob_s_cors[3][0])*(box_glob_s_cors[2][1]-box_glob_s_cors[3][1])]
        
        # for i in range(4):
        #     box_glob_s_norv[i] = box_glob_s_norv[i]/math.sqrt(box_glob_s_norv[i][0]**2 + box_glob_s_norv[i][1]**2 + box_glob_s_norv[i][2]**2)
        # for i in range(4):
        #     if box_glob_s_norv[i][0] == 1 or box_glob_s_norv[i][0] == -1:
        #         box_glob_s_norv[i][0] = box_glob_s_norv[i][0]*0.999
        #     if box_glob_s_norv[i][1] == 0 and box_glob_s_norv[i][2] < 0:
        #         box_glob_s_norv[i][2] = box_glob_s_norv[i][2]*0.999
        #     if box_glob_s_norv[i][1] >= 0:
        #         euler_angles[i][0] = math.acos(box_glob_s_norv[i][2]/math.sqrt(1-box_glob_s_norv[i][0]**2))
        #     else:
        #         if box_glob_s_norv[i][2] >= 0:
        #             euler_angles[i][0] = math.asin(box_glob_s_norv[i][1]/math.sqrt(1-box_glob_s_norv[i][0]**2))
        #         else:
        #             euler_angles[i][0] = -math.acos(box_glob_s_norv[i][2]/math.sqrt(1-box_glob_s_norv[i][0]**2))
        #     euler_angles[i][1] = math.asin(box_glob_s_norv[i][0])                                        
        # print('box_glob_s_cors', box_glob_s_cors)
        # print('box_glob_s_norv', box_glob_s_norv)
        # print('euler_angles', euler_angles)                                
        # suction_rotation_angle = euler_angles[0]

    return suction_center_cor, np.deg2rad(suction_rotation_angle)





