import os
import sys
sys.path.insert(0, '..')

import cv2
import numpy as np
import imutils
np.set_printoptions(suppress=True)

import torch

from test_config import cfg
from utils.imutils import *
from utils.transforms import *

class Predict:
    @staticmethod
    def predict(model, input_image):
        image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
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

            keypoints = []
            for p in range(cfg.num_class): 
                single_map[p] /= np.amax(single_map[p])
                border = 10
                dr = np.zeros((
                    cfg.output_shape[0] + 2 * border,
                    cfg.output_shape[1] + 2 * border))
                dr[border:-border, border:-border] = single_map[p].copy()
                dr = cv2.GaussianBlur(dr, (21, 21), 0)
                lb = dr.argmax()
                y, x = np.unravel_index(lb, dr.shape)
                dr[y, x] = 0
                lb = dr.argmax()
                py, px = np.unravel_index(lb, dr.shape)
                y -= border
                x -= border
                py -= border + y
                px -= border + x
                ln = (px ** 2 + py ** 2) ** 0.5
                delta = 0.25
                if ln > 1e-3:
                    x += delta * px / ln
                    y += delta * py / ln
                x = max(0, min(x, cfg.output_shape[1] - 1))
                y = max(0, min(y, cfg.output_shape[0] - 1))
                resy = (cfg.data_shape[0] / cfg.output_shape[0]) * y + 2
                resx = (cfg.data_shape[1] / cfg.output_shape[1]) * x + 2
                resy = resy * input_image.shape[0] / cfg.data_shape[0]
                resx = resx * input_image.shape[1] / cfg.data_shape[1]
                resy = int(round(resy))
                resx = int(round(resx))
                keypoints.append((resx, resy))

        return keypoints

    @staticmethod
    def predict_val(model, i, inputs, meta):
        single_result_dict_list = []
        with torch.no_grad():
            input_var = torch.autograd.Variable(inputs.cuda())
            global_outputs, refine_output = model(input_var)
            score_map = refine_output.data.cpu()
            score_map = score_map.numpy()

            ids = meta['imgID'].numpy()
            det_scores = meta['det_scores']
            details = meta['augmentation_details']
            for b in range(inputs.size(0)):
                single_result_dict = {}
                single_result = []
                
                single_map = score_map[b]
                r0 = single_map.copy()
                r0 /= 255
                r0 += 0.5
                v_score = np.zeros(cfg.num_class)
                for p in range(cfg.num_class): 
                    single_map[p] /= np.amax(single_map[p])
                    border = 10
                    dr = np.zeros((
                        cfg.output_shape[0] + 2*border,
                        cfg.output_shape[1]+2*border))
                    dr[border:-border, border:-border] = single_map[p].copy()
                    dr = cv2.GaussianBlur(dr, (21, 21), 0)
                    lb = dr.argmax()
                    y, x = np.unravel_index(lb, dr.shape)
                    dr[y, x] = 0
                    lb = dr.argmax()
                    py, px = np.unravel_index(lb, dr.shape)
                    y -= border
                    x -= border
                    py -= border + y
                    px -= border + x
                    ln = (px ** 2 + py ** 2) ** 0.5
                    delta = 0.25
                    if ln > 1e-3:
                        x += delta * px / ln
                        y += delta * py / ln
                    x = max(0, min(x, cfg.output_shape[1] - 1))
                    y = max(0, min(y, cfg.output_shape[0] - 1))
                    resy = float((4 * y + 2) / cfg.data_shape[0] * \
                        (details[b][3] - details[b][1]) + details[b][1])
                    resx = float((4 * x + 2) / cfg.data_shape[1] * \
                        (details[b][2] - details[b][0]) + details[b][0])
                    v_score[p] = float(
                        r0[p, int(round(y)+1e-10), int(round(x)+1e-10)])
                    single_result.append(resx)
                    single_result.append(resy)
                    single_result.append(1)   
                if len(single_result) != 0:
                    single_result_dict['image_id'] = int(ids[b])
                    single_result_dict['category_id'] = 1
                    single_result_dict['keypoints'] = single_result
                    single_result_dict['score'] = \
                        float(det_scores[b]) * v_score.mean()
                    single_result_dict_list.append(single_result_dict)

        return single_result_dict_list

class PredictWithRotation:
    @staticmethod
    def apply_image_rotation(mat, rotation):
        mat = imutils.rotate_bound(mat, rotation)
        
        return mat
        
    @staticmethod
    def revert_heatmap_rotation(mat, rotation):
        _, mat_h, mat_w = mat.shape
        
        for (i, joint_mat) in enumerate(mat):
            joint_mat = imutils.rotate_bound(joint_mat, -rotation)
            joint_mat = cv2.resize(joint_mat, (mat_w, mat_h))
            mat[i] = joint_mat
        
        return mat

    @staticmethod
    def get_heatmaps(model, input_image, rotation):
        image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        image = PredictWithRotation.apply_image_rotation(image, rotation)
        image = cv2.resize(image, (cfg.data_shape[1], cfg.data_shape[0]))
        img = im_to_torch(image)
        img = color_normalize(img, cfg.pixel_means)
        img.unsqueeze_(0)

        with torch.no_grad():
            input_var = torch.autograd.Variable(img.cuda())
            _, refine_output = model(input_var)
            single_map = refine_output.data.cpu().numpy()[0]
            
            single_map = PredictWithRotation.revert_heatmap_rotation(
                single_map, rotation)

        return single_map

    @staticmethod
    def get_keypoints(single_map, input_image, rotation):
        keypoints = []
        for p in range(cfg.num_class): 
            max_conf = np.amax(single_map[p])
            single_map[p] /= max_conf
            border = 10
            dr = np.zeros((
                cfg.output_shape[0] + 2 * border,
                cfg.output_shape[1] + 2 * border))
            dr[border:-border, border:-border] = single_map[p].copy()
            dr = cv2.GaussianBlur(dr, (21, 21), 0)
            lb = dr.argmax()
            y, x = np.unravel_index(lb, dr.shape)
            dr[y, x] = 0
            lb = dr.argmax()
            py, px = np.unravel_index(lb, dr.shape)
            y -= border
            x -= border
            py -= border + y
            px -= border + x
            ln = (px ** 2 + py ** 2) ** 0.5
            delta = 0.25
            if ln > 1e-3:
                x += delta * px / ln
                y += delta * py / ln
            x = max(0, min(x, cfg.output_shape[1] - 1))
            y = max(0, min(y, cfg.output_shape[0] - 1))
            resy = (cfg.data_shape[0] / cfg.output_shape[0]) * y + 2
            resx = (cfg.data_shape[1] / cfg.output_shape[1]) * x + 2
            resy = resy * input_image.shape[0] / cfg.data_shape[0]
            resx = resx * input_image.shape[1] / cfg.data_shape[1]
            resy = int(round(resy))
            resx = int(round(resx))
            keypoints.append((p, rotation, resx, resy, max_conf))

        return keypoints

    @staticmethod
    def predict(model, input_image, rotation_rate):
        rotations_list = list(range(0, 360, rotation_rate))
        single_map_list = np.array([
            (PredictWithRotation.get_heatmaps(model, input_image, r), r)
            for r in rotations_list])

        keypoints_list = []
        for sm, r in single_map_list:
            keypoints_list += PredictWithRotation.get_keypoints(
                sm, input_image, r)

        keypoints_selection_method = PredictWithRotation.keypoints_selection_v1
        keypoints = keypoints_selection_method(keypoints_list)

        return keypoints

    @staticmethod
    def keypoints_selection_v1(keypoints_list):
        '''
        from a list of keypoints from different image rotations,
        does a weighted average of the keypoints, considering its confidence
        '''
        keypoints_list = np.array(keypoints_list)
        keypoints = []
        
        grouped_keypoints_list = np.array([
            [k for k in keypoints_list if k[0] == i]
            for i in np.unique(keypoints_list[:,0])])
        
        for keypoint_list in grouped_keypoints_list:
            kx = 0
            ky = 0
            kw_sum = (keypoint_list[:,4] ** 3).sum()
            for _, _, x, y, w in keypoint_list:
                fw = w ** 3
                kx += (x * fw)
                ky += (y * fw)
            kx /= kw_sum
            ky /= kw_sum
            
            kx = int(round(kx))
            ky = int(round(ky))
            
            keypoints.append((kx, ky))

        return keypoints

    @staticmethod
    def predict_val(model, inputs, meta, rotation_rate):
        input_image = inputs.cpu().numpy()[0]
        keypoints = PredictWithRotation.predict(model, input_image,
            rotation_rate)
        keypoints = [k for t in [(x, y, 1) for (x, y) in keypoints] for k in t]
        single_result = keypoints

        single_result_dict = {}

        ids = meta['imgID'].numpy()[0]
        det_scores = meta['det_scores'][0]
        details = meta['augmentation_details'][0]

        single_result_dict['image_id'] = int(ids)
        single_result_dict['category_id'] = 1
        single_result_dict['keypoints'] = single_result
        single_result_dict['score'] = 1

        single_result_dict_list = [single_result_dict]

        return single_result_dict_list
