import os
import sys
sys.path.insert(0, '..')

import re
import math
import time
import cv2
import numpy as np

import torch

from test_config import cfg
from flattened_network import network

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
    # model.cuda()
    model.cpu()
    model.eval()

    model = list(model.modules())[1]

    return model

model = load_flattened_model()

from prunning import prunner
model2 = prunner.prune_layer(model, 0, 1)

from summary import summary
summary(model2, (3, 256, 192))

import hiddenlayer as hl
hl.build_graph(model, torch.zeros([1, 3, 256, 192]))
