
import time
import os,sys
import numpy as np
import random 
import math

def py_cpu_nms(boxes, pred_score, co_thresh, min_area,max_area):
    x1 = boxes[:,0,0] 
    y1 = boxes[:,0,1] 
    x2 = boxes[:,1,0] 
    y2 = boxes[:,1,1]
    order=[]
    for i in range(len(pred_score)):
        x11 = boxes[i,0,0] 
        y11 = boxes[i,0,1] 
        x22 = boxes[i,1,0] 
        y22 = boxes[i,1,1]
        area = (x22 - x11) * (y22 - y11)
        if area > min_area and area < max_area:
            order.append(i)
    
    areas = (x2 - x1 + 1) * (y2 - y1 + 1) 

    # order = np.array(scores).argsort()[::-1]
    order = np.array(order)
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]]) 
        yy1 = np.maximum(y1[i], y1[order[1:]]) 
        xx2 = np.minimum(x2[i], x2[order[1:]]) 
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1) 
        h = np.maximum(0.0, yy2 - yy1 + 1) 
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= co_thresh)[0] 
        order = order[inds + 1]
        
    # ignore the boxes where the area bellow 1000
    # print('keep')
    # print(keep)
    # keep0 = []
    # for num in range(len(keep)):
    #     x1 = boxes[keep[num],0,0] 
    #     y1 = boxes[keep[num],0,1] 
    #     x2 = boxes[keep[num],1,0] 
    #     y2 = boxes[keep[num],1,1] 
    #     area = (x2 - x1) * (y2 - y1)
    #     print(area)
    #     if area >1100:
    #         #print(keep0)
    #         keep0.append(keep[num])
    #         #print(keep0)   
    # print('keep0')
    # print(keep0)
    return keep




