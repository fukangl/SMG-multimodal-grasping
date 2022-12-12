
import time
import os,sys
import numpy as np
import random 
import math

def py_cpu_nms(boxes, pred_score, co_thresh, min_area,max_area):
    #x1、y1、x2、y2、以及score赋值
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
    
    #scores = pred_score
    #print(scores)
    #每一个检测框的面积 
    areas = (x2 - x1 + 1) * (y2 - y1 + 1) 
    #按照score置信度降序排序
    #order = np.array(scores).argsort()[::-1]
    order = np.array(order)
    keep = [] #保留的结果框集合
    while order.size > 0:
        i = order[0]
        keep.append(i) #保留该类剩余box中得分最高的一个
        # print("keep")
        # print(keep)
        #得到相交区域,左上及右下
        xx1 = np.maximum(x1[i], x1[order[1:]]) 
        # print(xx1)
        yy1 = np.maximum(y1[i], y1[order[1:]]) 
        xx2 = np.minimum(x2[i], x2[order[1:]]) 
        yy2 = np.minimum(y2[i], y2[order[1:]])
        #计算相交的面积,不重叠时面积为0
        w = np.maximum(0.0, xx2 - xx1 + 1) 
        h = np.maximum(0.0, yy2 - yy1 + 1) 
        inter = w * h
        #计算IoU：重叠面积 /（面积1+面积2-重叠面积） 
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        #保留IoU小于阈值的box 
        inds = np.where(ovr <= co_thresh)[0] 
        order = order[inds + 1] #因为ovr数组的长度比order数组少一个,所以这里要将所有下标后移一位
        
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




