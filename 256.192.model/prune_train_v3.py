import os
import sys
sys.path.insert(0, '..')
os.environ['CUDA_VISIBLE_DEVICES']='0'

import argparse
import time
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(suppress=True)

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torchvision.datasets as datasets

from config import cfg
from utils.logger import Logger
from utils.evaluation import accuracy, AverageMeter, final_preds
from utils.misc import save_model, adjust_learning_rate
from utils.osutils import mkdir_p, isfile, isdir, join
from utils.transforms import fliplr, flip_back
from networks import network 
from dataloader.mscocoMulti import MscocoMulti
from funcs import *

def number_of_parameters(model):
    parameters = list(model.parameters())
    parameters = [p.cpu().detach().numpy() for p in parameters]
    
    n = sum([np.prod(p.shape) for p in parameters])
    
    return n
    
def prune_model(model, percentage):
    def l2_norm(k):
        k = k.flatten()
        l2_norm = sum([v ** 2 for v in k]) ** (1/2)
        
        return l2_norm
    
    parameters = list(model.parameters())
    parameters = [p.cpu().detach().numpy() for p in parameters]
    
    l2_list = []
    for i in range(len(parameters)):
        print('%s out of %s' % (i, len(parameters)))
        p = parameters[i]
        if p.ndim == 4:
            s = p.shape
            for j in range(s[0]):
                for k in range(s[1]):
                    l2_list.append((i, j, k, l2_norm(p[j][k])))
    
    return l2_list

def load_model():
    checkpoint_file = os.path.join('checkpoint', 'CPN50_256x192.pth.tar')
    checkpoint = torch.load(checkpoint_file)

    model = network.__dict__[cfg.model](cfg.output_shape, cfg.num_class,
        pretrained=False)
    model = torch.nn.DataParallel(model).cuda()
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    return model

# model = load_model()
