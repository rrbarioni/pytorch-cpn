import os
import sys
sys.path.insert(0, '..')

import re
import math
import time
import cv2
import numpy as np

# from pytorch2keras.converter import pytorch_to_keras

import torch

from test_config import cfg
from flattened_network import network
from utils.imutils import *
from utils.transforms import *

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

def prune_conv_layer(model, layer_index, filter_index):
    def replace_layers(model, i, indexes, layers):
        if i in indexes:
            return layers[indexes.index(i)]
        return model[i]

    # _, conv = model.features._modules.items()[layer_index]
    # _, conv = list(dict(model._modules).items())[layer_index]
    _, conv = list(model._modules.items())[layer_index]
    next_conv = None
    offset = 1

    # while layer_index + offset <  len(model.features._modules.items()):
    # while layer_index + offset < len(list(dict(model._modules).items())):
    while layer_index + offset < len(list(model._modules.items())):
        # res =  model.features._modules.items()[layer_index+offset]
        # res =  list(dict(model._modules).items())[layer_index + offset]
        res = list(model._modules.items())[layer_index + offset]
        if isinstance(res[1], torch.nn.modules.conv.Conv2d):
            next_name, next_conv = res
            break
        offset = offset + 1
    
    new_conv = \
        torch.nn.Conv2d(in_channels=conv.in_channels, 
            out_channels=conv.out_channels - 1,
            kernel_size=conv.kernel_size, 
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
            bias=conv.bias)

    old_weights = conv.weight.data.cpu().numpy()
    new_weights = new_conv.weight.data.cpu().numpy()

    new_weights[:filter_index,:,:,:] = old_weights[:filter_index,:,:,:]
    new_weights[filter_index:,:,:,:] = old_weights[filter_index + 1:,:,:,:]
    new_conv.weight.data = torch.from_numpy(new_weights).cuda()

    if conv.bias is not None:
        bias_numpy = conv.bias.data.cpu().numpy()

        bias = np.zeros(shape=(bias_numpy.shape[0] - 1), dtype=np.float32)
        bias[:filter_index] = bias_numpy[:filter_index]
        bias[filter_index:] = bias_numpy[filter_index + 1:]
        new_conv.bias.data = torch.from_numpy(bias).cuda()

    if not next_conv is None:
        next_new_conv = \
            torch.nn.Conv2d(in_channels=next_conv.in_channels - 1,
                out_channels=next_conv.out_channels,
                kernel_size=next_conv.kernel_size,
                stride=next_conv.stride,
                padding=next_conv.padding,
                dilation=next_conv.dilation,
                groups=next_conv.groups,
                bias=next_conv.bias)

        old_weights = next_conv.weight.data.cpu().numpy()
        new_weights = next_new_conv.weight.data.cpu().numpy()

        new_weights[:,:filter_index,:,:] = old_weights[:,:filter_index,:,:]
        new_weights[:,filter_index:,:,:] = old_weights[:,filter_index + 1:,:,:]
        next_new_conv.weight.data = torch.from_numpy(new_weights).cuda()

        if next_new_conv.bias is not None:
            next_new_conv.bias.data = next_conv.bias.data

    if not next_conv is None:
        
        features = torch.nn.Sequential(
            *(replace_layers(
                # model.features, i, [layer_index, layer_index+offset],
                # list(model.modules()), i, [layer_index, layer_index + offset],
                list(model._modules.items()), i,
                [layer_index, layer_index + offset], [new_conv, next_new_conv]) \
                # for i, _ in enumerate(model.features)))
                # for i, _ in enumerate(model.modules())))
                for i, _ in enumerate(list(model._modules.items()))))
        '''
        replace_layers_list = []
        indexes = [layer_index, layer_index + offset]
        layers = [new_conv, next_new_conv]
        for i, _ in enumerate(list(model._modules.items())):
            if i in indexes:
                curr_layer = layers[indexes.index(i)]
            else:
                # curr_layer = list(model.modules())[i]
                curr_layer =  list(model._modules.items())[i][1]
            replace_layers_list.append(curr_layer)
        features = torch.nn.Sequential(*replace_layers_list)
        '''

        # del model.features
        del model
        del conv

        # model.features = features
        model = features

    '''
    else:
        # Prunning the last conv layer. This affects the first linear layer of the
        # classifier.
        # model.features = torch.nn.Sequential(
        model = torch.nn.Sequential(
                # *(replace_layers(model.features, i, [layer_index], \
                *(replace_layers(list(model.modules()), i, [layer_index], \
                    # [new_conv]) for i, _ in enumerate(model.features)))
                    [new_conv]) for i, _ in enumerate(model.modules())))
        layer_index = 0
        old_linear_layer = None
        for _, module in model.classifier._modules.items():
            if isinstance(module, torch.nn.Linear):
                old_linear_layer = module
                break
            layer_index = layer_index  + 1

        if old_linear_layer is None:
            raise BaseException('No linear layer found in classifier')
        params_per_input_channel = \
            old_linear_layer.in_features / conv.out_channels

        new_linear_layer = \
            torch.nn.Linear(
                old_linear_layer.in_features - params_per_input_channel, 
                old_linear_layer.out_features)
        
        old_weights = old_linear_layer.weight.data.cpu().numpy()
        new_weights = new_linear_layer.weight.data.cpu().numpy()        

        new_weights[:, : filter_index * params_per_input_channel] = \
            old_weights[:, : filter_index * params_per_input_channel]
        new_weights[:, filter_index * params_per_input_channel :] = \
            old_weights[:, (filter_index + 1) * params_per_input_channel :]
        
        new_linear_layer.bias.data = old_linear_layer.bias.data

        new_linear_layer.weight.data = torch.from_numpy(new_weights).cuda()

        classifier = torch.nn.Sequential(
            *(replace_layers(model.classifier, i, [layer_index], \
                [new_linear_layer]) for i, _ in enumerate(model.classifier)))

        del model.classifier
        del next_conv
        del conv
        model.classifier = classifier

    return model
    '''

model = load_flattened_model()
model2 = prune_conv_layer(model, 0, 1)
