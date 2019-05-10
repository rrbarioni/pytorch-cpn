import os
import sys
sys.path.insert(0, '..')

import math
import time
import cv2
import numpy as np

from pytorch2keras.converter import pytorch_to_keras

import torch

from test_config import cfg
from networks import network
from utils.imutils import *
from utils.transforms import *

def load_model():
    checkpoint_file = os.path.join('checkpoint', 'CPN50_256x192.pth.tar')
    checkpoint = torch.load(checkpoint_file)

    model = network.__dict__[cfg.model](cfg.output_shape, cfg.num_class,
        pretrained=False)
    model = torch.nn.DataParallel(model)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    
    model = list(model.modules())[1]

    return model, checkpoint

model, checkpoint = load_model()

input_np = np.random.uniform(0, 1, (1,3) + cfg.data_shape)
input_var = torch.autograd.Variable(torch.FloatTensor(input_np))

k_model = pytorch_to_keras(model, input_var, [(3,) + cfg.data_shape], verbose=True)  

'''
from merged_network import network
model = network.CPN(cfg.output_shape, cfg.num_class)
model.cuda()
global_outputs, refine_output = model(input_var)
'''