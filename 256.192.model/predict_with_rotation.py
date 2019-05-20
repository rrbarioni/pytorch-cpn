import os
import sys
sys.path.insert(0, '..')
# os.environ['CUDA_VISIBLE_DEVICES']=''

import re
import math
import time
import cv2
import numpy as np
import imutils

import torch

from test_config import cfg
from utils.imutils import *
from utils.transforms import *

colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0],
    [170, 255, 0], [85, 255, 0], [0, 255, 0], [0, 255, 85], [0, 255, 170],
    [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
    [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

keypoints_pairs = [(0,1), (0,2), (1,3), (2,4), (0,5), (0,6), (5,7), (6,8),
    (7,9), (8,10), (0,11), (0,12), (11,13), (12,14), (13,15), (14,16)]

rotation_rate = 45

def load_flattened_model():
    def checkpoint_state_dict_to_model_state_dict_keys(checkpoint_state_dict):
        def ckpt_to_model_key(k):
            # replace all "." with "_" (except the first and the last one ".")
            dot_occurences = [c.start() for c in re.finditer('\.', k)][1:-1]
            k = list(k)
            for di in dot_occurences:
                k[di] = '_'
            k = ''.join(k)
                
            return k
    
        checkpoint_state_dict_cpy = checkpoint_state_dict.copy()
        for k in checkpoint_state_dict.keys():
            new_k = ckpt_to_model_key(k)
            checkpoint_state_dict_cpy[new_k] = checkpoint_state_dict_cpy.pop(k)
            
        return checkpoint_state_dict_cpy
    
    checkpoint_file = os.path.join('checkpoint', 'CPN50_256x192.pth.tar')
    checkpoint = torch.load(checkpoint_file)
    checkpoint['state_dict'] = checkpoint_state_dict_to_model_state_dict_keys(
        checkpoint['state_dict'])

    model = network.__dict__[cfg.model](cfg.output_shape, cfg.num_class)
    model = torch.nn.DataParallel(model)
    model.load_state_dict(checkpoint['state_dict'])
    model.cuda()
    model.eval()

    model = list(model.modules())[1]

    return model

def load_model():
    checkpoint_file = os.path.join('checkpoint', 'CPN50_256x192.pth.tar')
    checkpoint = torch.load(checkpoint_file)

    model = network.__dict__[cfg.model](cfg.output_shape, cfg.num_class,
        pretrained=False)
    model = torch.nn.DataParallel(model)
    model.cuda()
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    model = list(model.modules())[1]

    return model

def apply_image_rotation(mat, rotation):
    mat = imutils.rotate_bound(mat, rotation)
    
    return mat
    
def revert_heatmap_rotation(mat, rotation):
    _, mat_h, mat_w = mat.shape
    
    for (i, joint_mat) in enumerate(mat):
        joint_mat = imutils.rotate_bound(joint_mat, -rotation)
        joint_mat = cv2.resize(joint_mat, (mat_w, mat_h))
        mat[i] = joint_mat
    
    return mat

def get_heatmaps(model, input_image, rotation):
    image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    image = apply_image_rotation(image, rotation)
    image = cv2.resize(image, (cfg.data_shape[1], cfg.data_shape[0]))
    img = im_to_torch(image)
    img = color_normalize(img, cfg.pixel_means)
    img.unsqueeze_(0)

    with torch.no_grad():
        input_var = torch.autograd.Variable(img.cuda())
        global_outputs, refine_output = model(input_var)
        score_map = refine_output.data.cpu()
        score_map = score_map.numpy()
        single_map = score_map[0]
        
        single_map = revert_heatmap_rotation(single_map, rotation)

    return single_map

def predict(model, input_image):
    single_map_list = [
        get_heatmaps(model, image, r)
        for r in range(0, 360, rotation_rate)]

def canvas_with_skeleton(canvas, keypoints):
    for i in range(len(keypoints_pairs)):
        cur_canvas = canvas.copy()
        
        ki1, ki2 = keypoints_pairs[i]
        y1, x1 = keypoints[ki1]
        y2, x2 = keypoints[ki2]
        X = np.array([x1, x2])
        Y = np.array([y1, y2])
        mX = np.mean(X)
        mY = np.mean(Y)
        length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
        polygon = cv2.ellipse2Poly(
            (int(mY), int(mX)),
            (int(length / 2), 4),
            int(angle), 0, 360, 1)
        cv2.fillConvexPoly(cur_canvas, polygon, colors[i])
        canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)
        
    for i in range(len(keypoints)):
        cv2.circle(canvas, keypoints[i], 4, colors[i], thickness=-1)

    return canvas

from flattened_network import network
model = load_flattened_model()
# from networks import network
# model = load_model()

out_video_frames = []
cap = cv2.VideoCapture('sample1.mp4')
while(True):
    
    _, img = cap.read()

    t = time.time()
    keypoints = predict(model, img)
    t = time.time() - t
    
    canvas = canvas_with_skeleton(img, keypoints)
    canvas = cv2.resize(canvas, None, fx=2, fy=2)
    # cv2.putText(canvas, 'fps: %s' % (1/t), (60, 60), cv2.FONT_HERSHEY_SIMPLEX,
    #     2, 255)
    # cv2.imshow('frame', canvas)
    out_video_frames.append(canvas)
    
    '''
    t = time.time()
    _, img = cap.read()
    img = cv2.resize(img, None, fx=2, fy=2)
    t = time.time() - t

    cv2.putText(img, 'fps: %s' % (1/(t + 0.00000001)), (60, 60),
        cv2.FONT_HERSHEY_SIMPLEX, 2, 255)
    cv2.imshow('frame', img)
    '''

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()