import os
import sys
import argparse
import time
from operator import itemgetter
from heapq import nsmallest
sys.path.insert(0, '..')

import cv2
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

import torchvision
import torchvision.datasets as datasets
from torchvision import models

from config import cfg as cfg
from test_config import cfg as test_cfg
from utils.logger import Logger
from utils.evaluation import accuracy, AverageMeter, final_preds
from utils.misc import save_model, adjust_learning_rate
from utils.osutils import mkdir_p, isfile, isdir, join
from utils.transforms import fliplr, flip_back
from networks import network 
from dataloader.mscocoMulti import MscocoMulti
 
def replace_layers(model, i, indexes, layers):
    if i in indexes:
        return layers[indexes.index(i)]
    return model[i]

def prune_conv_layer(model, layer_index, filter_index):
    _, conv = model.features._modules.items()[layer_index]
    next_conv = None
    offset = 1

    while layer_index + offset <  len(model.features._modules.items()):
        res =  model.features._modules.items()[layer_index+offset]
        if isinstance(res[1], torch.nn.modules.conv.Conv2d):
            next_name, next_conv = res
            break
        offset = offset + 1
    
    new_conv = \
        torch.nn.Conv2d(in_channels = conv.in_channels, \
            out_channels = conv.out_channels - 1,
            kernel_size = conv.kernel_size, \
            stride = conv.stride,
            padding = conv.padding,
            dilation = conv.dilation,
            groups = conv.groups,
            bias = conv.bias)

    old_weights = conv.weight.data.cpu().numpy()
    new_weights = new_conv.weight.data.cpu().numpy()

    new_weights[: filter_index, :, :, :] = \
        old_weights[: filter_index, :, :, :]
    new_weights[filter_index : , :, :, :] = \
        old_weights[filter_index + 1 :, :, :, :]
    new_conv.weight.data = torch.from_numpy(new_weights).cuda()

    bias_numpy = conv.bias.data.cpu().numpy()

    bias = np.zeros(shape = (bias_numpy.shape[0] - 1), dtype = np.float32)
    bias[:filter_index] = bias_numpy[:filter_index]
    bias[filter_index : ] = bias_numpy[filter_index + 1 :]
    new_conv.bias.data = torch.from_numpy(bias).cuda()

    if not next_conv is None:
        next_new_conv = \
            torch.nn.Conv2d(in_channels = next_conv.in_channels - 1,\
                out_channels =  next_conv.out_channels, \
                kernel_size = next_conv.kernel_size, \
                stride = next_conv.stride,
                padding = next_conv.padding,
                dilation = next_conv.dilation,
                groups = next_conv.groups,
                bias = next_conv.bias)

        old_weights = next_conv.weight.data.cpu().numpy()
        new_weights = next_new_conv.weight.data.cpu().numpy()

        new_weights[:, : filter_index, :, :] = \
            old_weights[:, : filter_index, :, :]
        new_weights[:, filter_index : , :, :] = \
            old_weights[:, filter_index + 1 :, :, :]
        next_new_conv.weight.data = torch.from_numpy(new_weights).cuda()

        next_new_conv.bias.data = next_conv.bias.data

    if not next_conv is None:
        features = torch.nn.Sequential(
            *(replace_layers(model.features, i, \
                [layer_index, layer_index+offset], \
                [new_conv, next_new_conv])
            for i, _ in enumerate(model.features)))
        del model.features
        del conv

        model.features = features

    else:
        # Prunning the last conv layer. This affects the first linear layer of
        # the classifier.
        model.features = torch.nn.Sequential(
                *(replace_layers(model.features, i, [layer_index], \
                    [new_conv]) for i, _ in enumerate(model.features)))
        layer_index = 0
        old_linear_layer = None
        for _, module in model.classifier._modules.items():
            if isinstance(module, torch.nn.Linear):
                old_linear_layer = module
                break
            layer_index = layer_index  + 1

        if old_linear_layer is None:
            raise BaseException('No linear laye found in classifier')
        params_per_input_channel = \
            old_linear_layer.in_features / conv.out_channels

        new_linear_layer = torch.nn.Linear(
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

def get_modules(model):
    modules = [m for m in list(model.modules()) if len(list(m.children())) == 0]
    modules = list(enumerate(modules))

    return modules

class FilterPrunner:
    def __init__(self, model):
        self.model = model
        self.reset()
    
    def reset(self):
        # self.activations = []
        # self.gradients = []
        # self.grad_index = 0
        # self.activation_to_layer = {}
        self.filter_ranks = {}

    def forward(self, x):
        self.activations = []
        self.gradients = []
        self.grad_index = 0
        self.activation_to_layer = {}

        activation_index = 0
        # for layer, (name, module) in \
        #   enumerate(self.model.features._modules.items()):
        for layer, module in get_modules(self.model):
            x = module(x)
            if isinstance(module, torch.nn.modules.conv.Conv2d):
                x.register_hook(self.compute_rank)
                self.activations.append(x)
                self.activation_to_layer[activation_index] = layer
                activation_index += 1

        return self.model.classifier(x.view(x.size(0), -1))

    def compute_rank(self, grad):
        activation_index = len(self.activations) - self.grad_index - 1
        activation = self.activations[activation_index]
        values = \
            torch.sum((activation * grad), dim = 0).\
                sum(dim=2).sum(dim=3)[0, :, 0, 0].data
        
        # Normalize the rank by the filter dimensions
        values = values / \
            (activation.size(0) * activation.size(2) * activation.size(3))

        if activation_index not in self.filter_ranks:
            self.filter_ranks[activation_index] = \
                torch.FloatTensor(activation.size(1)).zero_().cuda()

        self.filter_ranks[activation_index] += values
        self.grad_index += 1

    def lowest_ranking_filters(self, num):
        data = []
        for i in sorted(self.filter_ranks.keys()):
            for j in range(self.filter_ranks[i].size(0)):
                data.append(
                    (self.activation_to_layer[i], j, self.filter_ranks[i][j]))

        return nsmallest(num, data, itemgetter(2))

    def normalize_ranks_per_layer(self):
        for i in self.filter_ranks:
            v = torch.abs(self.filter_ranks[i])
            v = v / np.sqrt(torch.sum(v * v))
            self.filter_ranks[i] = v.cpu()

    def get_prunning_plan(self, num_filters_to_prune):
        filters_to_prune = self.lowest_ranking_filters(num_filters_to_prune)

        # After each of the k filters are prunned,
        # the filter index of the next filters change since the model is
        #   smaller.
        filters_to_prune_per_layer = {}
        for (l, f, _) in filters_to_prune:
            if l not in filters_to_prune_per_layer:
                filters_to_prune_per_layer[l] = []
            filters_to_prune_per_layer[l].append(f)

        for l in filters_to_prune_per_layer:
            filters_to_prune_per_layer[l] = \
                sorted(filters_to_prune_per_layer[l])
            for i in range(len(filters_to_prune_per_layer[l])):
                filters_to_prune_per_layer[l][i] = \
                    filters_to_prune_per_layer[l][i] - i

        filters_to_prune = []
        for l in filters_to_prune_per_layer:
            for i in filters_to_prune_per_layer[l]:
                filters_to_prune.append((l, i))

        return filters_to_prune

class PrunningFineTuner_CPN:
    def __init__(self, args):
        self.args = args

        self.model = network.__dict__[cfg.model](
            cfg.output_shape, cfg.num_class, pretrained=True)
        self.model = torch.nn.DataParallel(self.model).cuda()

        self.criterion1 = torch.nn.MSELoss().cuda()
        self.criterion2 = torch.nn.MSELoss(reduce=False).cuda()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.lr,
            weight_decay=cfg.weight_decay)

        if self.args.resume:
            if isfile(args.resume):
                print('=> loading checkpoint %s' % self.args.resume)
                checkpoint = torch.load(self.args.resume)
                pretrained_dict = checkpoint['state_dict']
                self.model.load_state_dict(pretrained_dict)
                self.args.start_epoch = checkpoint['epoch']
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                print('=> loaded checkpoint %s (epoch %s)' % \
                    (self.args.resume, checkpoint['epoch']))
                self.logger = Logger(join(self.args.checkpoint, 'log.txt'),
                    resume=True)
            else:
                print('=> no checkpoint found at %s' % self.args.resume)
        else:        
            self.logger = Logger(join(self.args.checkpoint, 'log.txt'))
            self.logger.set_names(['Epoch', 'LR', 'Train Loss'])

        cudnn.benchmark = True
        print('    Total params: %.2fMB' % (sum(p.numel() \
            for p in self.model.parameters()) / (1024 * 1024) * 4))

        self.train_loader = torch.utils.data.DataLoader(
            MscocoMulti(cfg),
            batch_size=cfg.batch_size*self.args.num_gpus, shuffle=True,
            num_workers=self.args.workers, pin_memory=True) 

        self.prunner = FilterPrunner(self.model)

    def train(self):
        for epoch in range(self.args.start_epoch, self.args.epochs):
            lr = adjust_learning_rate(self.optimizer, epoch, cfg.lr_dec_epoch,
                cfg.lr_gamma)
            print('\nEpoch: %d | LR: %.8f' % (epoch + 1, lr))

            train_loss = self.train_epoch()
            print('train_loss: ',train_loss)

            self.logger.append([epoch + 1, lr, train_loss])

            save_model({
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'optimizer' : self.optimizer.state_dict(),
            }, checkpoint=self.args.checkpoint)

    def train_epoch(self, rank_filters):
        # prepare for refine loss
        def ohkm(loss, top_k):
            ohkm_loss = 0.
            for i in range(loss.size()[0]):
                sub_loss = loss[i]
                topk_val, topk_idx = torch.topk(sub_loss, k=top_k, dim=0,
                    sorted=False)
                tmp_loss = torch.gather(sub_loss, 0, topk_idx)
                ohkm_loss += torch.sum(tmp_loss) / top_k
            ohkm_loss /= loss.size()[0]
            return ohkm_loss

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()

        # switch to train mode
        self.model.train()

        for i, (inputs, targets, valid, meta) in enumerate(self.train_loader):
            input_var = torch.autograd.Variable(inputs.cuda())

            target15, target11, target9, target7 = targets
            refine_target_var = torch.autograd.Variable(
                target7.cuda(async=True))
            valid_var = torch.autograd.Variable(valid.cuda(async=True))

            # compute output
            if not rank_filters:
                global_outputs, refine_output = self.model(input_var)
            else:
                global_outputs, refine_output = self.prunner.forward(input_var)
            score_map = refine_output.data.cpu()

            loss = 0.
            global_loss_record = 0.
            refine_loss_record = 0.
            # comput global loss and refine loss
            for global_output, label in zip(global_outputs, targets):
                num_points = global_output.size()[1]
                global_label = label * (valid > 1.1).type(
                    torch.FloatTensor).view(-1, num_points, 1, 1)
                global_loss = self.criterion1(
                    global_output, torch.autograd.Variable(
                        global_label.cuda(async=True))) / 2
                loss += global_loss
                global_loss_record += global_loss.data.item()
            refine_loss = self.criterion2(refine_output, refine_target_var)
            refine_loss = refine_loss.mean(dim=3).mean(dim=2)
            refine_loss *= (valid_var > 0.1).type(torch.cuda.FloatTensor)
            refine_loss = ohkm(refine_loss, 8)
            loss += refine_loss
            refine_loss_record = refine_loss.data.item()

            # record loss
            losses.update(loss.data.item(), inputs.size(0))

            # compute gradient and do Optimization step
            self.optimizer.zero_grad()
            loss.backward()
            if not rank_filters:
                self.optimizer.step()

            if(i % 100 == 0 and i != 0):
                print('iteration %s | loss: %s, global loss: %s, \
                    refine loss: %s, avg loss: %s' % \
                    (i, loss.data.item(), global_loss_record, 
                        refine_loss_record, losses.avg))

        return losses.avg

    def get_candidates_to_prune(self, num_filters_to_prune):
        self.prunner.reset()
        self.train_epoch(rank_filters=True)
        self.prunner.normalize_ranks_per_layer()

        return self.prunner.get_prunning_plan(num_filters_to_prune)
        
    def total_num_filters(self):
        filters = 0
        for _, module in get_modules(self.model):
            if isinstance(module, torch.nn.modules.conv.Conv2d):
                filters = filters + module.out_channels

        return filters

    def prune(self):
        self.model.train()

        # Make sure all the layers are trainable
        # for param in self.model.features.parameters():
        for param in self.model.parameters():
            param.requires_grad = True

        number_of_filters = self.total_num_filters()
        num_filters_to_prune_per_iteration = 512
        iterations = int(
            float(number_of_filters) / num_filters_to_prune_per_iteration)

        iterations = int(iterations * 2.0 / 3)

        print('Number of prunning iterations to reduce 67% filters', iterations)

        for _ in range(iterations):
            print('Ranking filters.. ')
            prune_targets = self.get_candidates_to_prune(
                num_filters_to_prune_per_iteration)
            layers_prunned = {}
            for layer_index, filter_index in prune_targets:
                if layer_index not in layers_prunned:
                    layers_prunned[layer_index] = 0
                layers_prunned[layer_index] = layers_prunned[layer_index] + 1 

            print('Layers that will be prunned', layers_prunned)
            print('Prunning filters.. ')
            self.model = self.model.cpu()
            for layer_index, filter_index in prune_targets:
                self.model = prune_conv_layer(
                    self.model, layer_index, filter_index)

            self.model = model.cuda()

            message = str(
                100 * float(self.total_num_filters()) / number_of_filters) + '%'
            print('Filters prunned', str(message))
            print('Fine tuning to recover from prunning iteration.')
            self.train()


        print('Finished. Going to fine tune the model a bit more')
        self.train()
        torch.save(self.model.state_dict(), 'model_prunned')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch CPN Prune-Training')
    parser.add_argument('-j', '--workers', default=12, type=int, metavar='N',
                        help='number of data loading workers (default: 12)')
    parser.add_argument('-g', '--num_gpus', default=1, type=int, metavar='N',
                        help='number of GPU to use (default: 1)')    
    parser.add_argument('--epochs', default=32, type=int, metavar='N',
                        help='number of total epochs to run (default: 32)')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')
    # parser.add_argument('--resume', default='', type=str, metavar='PATH',
    #                     help='path to latest checkpoint')

    args = parser.parse_args()
    args.resume = os.path.join('checkpoint', 'CPN50_256x192.pth.tar')

    fine_tuner = PrunningFineTuner_CPN(args)
    fine_tuner.prune()
