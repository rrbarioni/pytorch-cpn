import os
import numpy as np
import json
import random
import math
import cv2
import skimage
import skimage.transform

import torch
import torch.utils.data as data

from utils.osutils import *
from utils.imutils import *
from utils.transforms import *

class MscocoMulti(data.Dataset):
    def __init__(self, cfg, train=True):
        self.img_folder = cfg.img_path
        self.inp_res = cfg.data_shape
        self.out_res = cfg.output_shape
        self.pixel_means = cfg.pixel_means
        self.num_class = cfg.num_class
        self.cfg = cfg
        self.bbox_extend_factor = cfg.bbox_extend_factor
        with open(cfg.gt_path) as anno_file:   
            self.anno = json.load(anno_file)

    def augmentationCropImage(self, img, bbox, joints=None):  
        height, width = self.inp_res[0], self.inp_res[1]
        bbox = np.array(bbox).reshape(4, ).astype(np.float32)
        add = max(img.shape[0], img.shape[1])
        mean_value = self.pixel_means
        bimg = cv2.copyMakeBorder(img, add, add, add, add, borderType=cv2.BORDER_CONSTANT, value=mean_value.tolist())
        objcenter = np.array([(bbox[0] + bbox[2]) / 2., (bbox[1] + bbox[3]) / 2.])      
        bbox += add
        objcenter += add
        crop_width = (bbox[2] - bbox[0]) * (1 + self.bbox_extend_factor[0] * 2)
        crop_height = (bbox[3] - bbox[1]) * (1 + self.bbox_extend_factor[1] * 2)
        if crop_height / height > crop_width / width:
            crop_size = crop_height
            min_shape = height
        else:
            crop_size = crop_width
            min_shape = width  

        crop_size = min(crop_size, objcenter[0] / width * min_shape * 2. - 1.)
        crop_size = min(crop_size, (bimg.shape[1] - objcenter[0]) / width * min_shape * 2. - 1)
        crop_size = min(crop_size, objcenter[1] / height * min_shape * 2. - 1.)
        crop_size = min(crop_size, (bimg.shape[0] - objcenter[1]) / height * min_shape * 2. - 1)

        min_x = int(objcenter[0] - crop_size / 2. / min_shape * width)
        max_x = int(objcenter[0] + crop_size / 2. / min_shape * width)
        min_y = int(objcenter[1] - crop_size / 2. / min_shape * height)
        max_y = int(objcenter[1] + crop_size / 2. / min_shape * height)                               

        x_ratio = float(width) / (max_x - min_x)
        y_ratio = float(height) / (max_y - min_y)

        img = cv2.resize(bimg[min_y:max_y, min_x:max_x, :], (width, height))  
        details = np.asarray([min_x - add, min_y - add, max_x - add, max_y - add]).astype(np.float)

        return img, details



    def data_augmentation(self, img, label, operation):
        height, width = img.shape[0], img.shape[1]
        center = (width / 2., height / 2.)
        n = label.shape[0]
        affrat = random.uniform(self.scale_factor[0], self.scale_factor[1])
        
        halfl_w = min(width - center[0], (width - center[0]) / 1.25 * affrat)
        halfl_h = min(height - center[1], (height - center[1]) / 1.25 * affrat)
        img = skimage.transform.resize(img[int(center[1] - halfl_h): int(center[1] + halfl_h + 1),
                             int(center[0] - halfl_w): int(center[0] + halfl_w + 1)], (height, width))
        for i in range(n):
            label[i][0] = (label[i][0] - center[0]) / halfl_w * (width - center[0]) + center[0]
            label[i][1] = (label[i][1] - center[1]) / halfl_h * (height - center[1]) + center[1]
            label[i][2] *= (
            (label[i][0] >= 0) & (label[i][0] < width) & (label[i][1] >= 0) & (label[i][1] < height))

        # flip augmentation
        if operation == 1:
            img = cv2.flip(img, 1)
            cod = []
            allc = []
            for i in range(n):
                x, y = label[i][0], label[i][1]
                if x >= 0:
                    x = width - 1 - x
                cod.append((x, y, label[i][2]))
            # **** the joint index depends on the dataset ****    
            for (q, w) in self.symmetry:
                cod[q], cod[w] = cod[w], cod[q]
            for i in range(n):
                allc.append(cod[i][0])
                allc.append(cod[i][1])
                allc.append(cod[i][2])
            label = np.array(allc).reshape(n, 3)

        # rotated augmentation
        if operation > 1:      
            angle = random.uniform(0, self.rot_factor)
            if random.randint(0, 1):
                angle *= -1
            rotMat = cv2.getRotationMatrix2D(center, angle, 1.0)
            img = cv2.warpAffine(img, rotMat, (width, height))
            
            allc = []
            for i in range(n):
                x, y = label[i][0], label[i][1]
                v = label[i][2]
                coor = np.array([x, y])
                if x >= 0 and y >= 0:
                    R = rotMat[:, : 2]
                    W = np.array([rotMat[0][2], rotMat[1][2]])
                    coor = np.dot(R, coor) + W
                allc.append(int(coor[0]))
                allc.append(int(coor[1]))
                v *= ((coor[0] >= 0) & (coor[0] < width) & (coor[1] >= 0) & (coor[1] < height))
                allc.append(int(v))
            label = np.array(allc).reshape(n, 3).astype(np.int)
        return img, label

  
    def __getitem__(self, index):
        a = self.anno[index]
        image_name = a['imgInfo']['img_paths']
        img_path = os.path.join(self.img_folder, image_name)
        gt_bbox = a['unit']['GT_bbox']

        # image = scipy.misc.imread(img_path, mode='RGB')
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image, details = self.augmentationCropImage(image, gt_bbox)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        meta = {'index' : index, 'imgID' : a['imgInfo']['imgID'], 
        'GT_bbox' : np.array([gt_bbox[0], gt_bbox[1], gt_bbox[2], gt_bbox[3]]), 
        'img_path' : img_path, 'augmentation_details' : details}

        meta['det_scores'] = a['score']
        return image, meta

    def __len__(self):
        return len(self.anno)


