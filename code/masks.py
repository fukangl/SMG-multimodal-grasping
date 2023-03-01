from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision
import torch 
import numpy as np
import cv2
import random
import NMS
import os
import time
 
model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
model.eval()
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]
def get_prediction(color_heightmap, threshold):
    w,h = color_heightmap.shape[0],color_heightmap.shape[1]
    area = w*h
    # img = Image.open(img_path)
    transform = T.Compose([T.ToTensor()])
    img = transform(color_heightmap)
    pred = model([img])
    #print(pred[0]['boxes'])           
    pred_score = list(pred[0]['scores'].detach().numpy())    
    pred_t = [pred_score.index(x) for x in pred_score if x>threshold]

    if len(pred_t)==0:
       masks_initial = None
       masks = None
       pred_boxes = None 
       pred_class = None
       number = 0
    else:
        
        #pred[0]['masks'] = F.interpolate(pred[0]['masks'],size=[224, 224], mode="bilinear",align_corners=True)                                
        masks_224 = F.interpolate(pred[0]['masks'],size=[224, 224], mode="bilinear",align_corners=True) 
        pred_m = pred_t[-1]
        masks_initial = (pred[0]['masks']>0.5).squeeze().detach().cpu().numpy()
        masks = (masks_224).squeeze().detach().cpu().numpy()
        if len(masks_initial.shape) == 2:
            masks_initial = np.expand_dims(masks_initial,axis=0)
        if len(masks.shape) == 2:
            masks = np.expand_dims(masks,axis=0)                   
        pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].numpy())]       
        pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().numpy())]        
        masks_initial = masks_initial[:pred_m+1]
        masks = masks[:pred_m+1]
        
        pred_boxes = pred_boxes[:pred_m+1]
        pred_class = pred_class[:pred_m+1]
        pred_score = pred_score[:pred_m+1]
        pred_boxes = np.array(pred_boxes)
                 
        for i in range(2):
            for j in range(2):
                pred_boxes[:,i,j] = pred_boxes[:,i,j]/2
        area = area/4                        
        keep = NMS.py_cpu_nms(pred_boxes, pred_score, 0.40, area/60, area/5)       
        masks_initial = masks_initial[keep]
        masks = masks[keep]
        
        pred_boxes = pred_boxes[keep]
        pclass = []
        for i in range(len(keep)):
            pclass.append(pred_class[keep[i]])
        pred_class = pclass

        number = len(masks)

    return masks_initial, masks, pred_boxes, pred_class, number
 
def random_colour_masks(image):
    colours = [[0, 255, 0],[0, 0, 255],[255, 0, 0],[0, 255, 255],[255, 255, 0],[255, 0, 255],[80, 70, 180],[250, 80, 190],[245, 145, 50],[70, 150, 250],[50, 190, 190]]
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)
    r[image == 1], g[image == 1], b[image == 1] = colours[random.randrange(0,10)]
    coloured_mask = np.stack([r, g,b], axis=2)
    
    return coloured_mask
 
def instance_segmentation(color_heightmap,iteration, threshold=0.01, rect_th=2, text_size=0.5, text_th=1):    
    masks_initial, masks, boxes, pred_cls, number = get_prediction(color_heightmap, threshold)
    print('objects_number: %r' % (number))
    
    if number == 0:
        masks_cter = np.zeros((2,2))
        box_mask_cors = np.zeros((2,4,2))
        return masks_initial, masks, number, boxes, masks_cter, box_mask_cors
    else:
        box_mask_cors = np.zeros((len(masks),4,2))        
        # for i in range(len(masks)):
        #     for i0 in range(int(boxes[i][0][0])+2,224,1):
        #         valid_ind0 = np.argwhere(masks[i][:,i0]==True)
        #         if len(valid_ind0) != 0:
        #             #print('valid_ind0',valid_ind0[0],valid_ind0[-1])
        #             #box_mask_cors[i][0] = [i0,valid_ind0[int(len(valid_ind0)/2)]]
        #             box_mask_cors[i][1] = [i0,valid_ind0[0]]
        #             box_mask_cors[i][0] = [i0,valid_ind0[-1]]
        #             break
        #     for i1 in range(int(boxes[i][0][1])+2,224,1):
        #         valid_ind1 = np.argwhere(masks[i][i1,:]==True)
        #         if len(valid_ind1) != 0:
        #             #box_mask_cors[i][1] = [valid_ind1[int(len(valid_ind1)/2)],i1]
        #             box_mask_cors[i][2] = [valid_ind1[0],i1]
        #             box_mask_cors[i][3] = [valid_ind1[-1],i1]
        #             break
        #     for i2 in range(int(boxes[i][1][0]-1)-2,0,-1):
        #         valid_ind2 = np.argwhere(masks[i][:,i2]==True)
        #         if len(valid_ind2) != 0:
        #             #box_mask_cors[i][2] = [i2,valid_ind2[int(len(valid_ind2)/2)]]
        #             box_mask_cors[i][4] = [i2,valid_ind2[0]]
        #             box_mask_cors[i][5] = [i2,valid_ind2[-1]]
        #             break
        #     for i3 in range(int(boxes[i][1][1]-1)-2,0,-1):
        #         valid_ind3 = np.argwhere(masks[i][i3,:]==True)
        #         if len(valid_ind3) != 0:
        #             #box_mask_cors[i][3] = [valid_ind3[int(len(valid_ind3)/2)],i3]
        #             box_mask_cors[i][7] = [valid_ind3[0],i3]
        #             box_mask_cors[i][6] = [valid_ind3[-1],i3]
        #             break 
        img = cv2.cvtColor(color_heightmap, cv2.COLOR_BGR2RGB)        
        img = img[::2,::2,:]
        masks_copy = masks.copy()
        masks_copy = np.array(masks_copy*10,dtype=np.uint8)
        for i in range(len(masks)):
            ret, thresh = cv2.threshold(masks_copy[i],0,255,cv2.THRESH_BINARY)
            contours,hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                x,y,w,h = cv2.boundingRect(c)
                rect = cv2.minAreaRect(c)
                box_mask_cors[i] = cv2.boxPoints(rect).astype(int)       
            
            rgb_mask = random_colour_masks(masks[i])            
            img = cv2.addWeighted(img, 1, rgb_mask, 0.1, 1)            
            cv2.rectangle(img, (boxes[i,0,0],boxes[i,0,1]), (boxes[i,1,0],boxes[i,1,1]),color=(0, 255, 0), thickness=rect_th)
            
            # poly = np.array(box_mask_cors[8],np.int32).reshape(-1,1,2)
            # cv2.polylines(img, [poly], True, color=(225, 225, 0), thickness=2)
            # cv2.putText(img,pred_cls[i], (boxes[i,0,0],boxes[i,0,1]), cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0),thickness=text_th)            
                               
        masks_cter = np.zeros((number,2))
        for i in range(number):
            masks_cter[i] = [(box_mask_cors[i,0,0] + box_mask_cors[i,1,0] + box_mask_cors[i,2,0] + box_mask_cors[i,3,0])/4,
                             (box_mask_cors[i,0,1] + box_mask_cors[i,1,1] + box_mask_cors[i,2,1] + box_mask_cors[i,3,1])/4]
        masks_cter = masks_cter.astype(int)
                
        
        plt.figure(figsize=(30,30))
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
        plt.show()
        # cv2.imwrite(os.path.join('/home/alienfit/grasping/logs/mask_img', '%06d.color.png' % (iteration)), img)
        
        print(np.max(masks_initial))
        
        
        return masks_initial, masks, number, boxes, masks_cter, box_mask_cors







