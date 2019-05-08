import math

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

class CPN(nn.Module):
    def __init__(self, output_shape, num_class):
        super(CPN, self).__init__()
        self.channel_settings = [2048, 1024, 512, 256]
        
        # resnet
        self.resnet_inplanes = 64
        self.resnet_layers = [3, 4, 6, 3]
        self.resnet_planes = [64, 128, 256, 512]
        self.resnet_strides = [1, 2, 2, 2]
        self.resnet_block_expansion = 4

        self.resnet_conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2,
            padding=3, bias=False)
        self.resnet_bn1 = nn.BatchNorm2d(64)
        self.resnet_relu = nn.ReLU(inplace=True)
        self.resnet_maxpool = nn.MaxPool2d(kernel_size=3, stride=2,
            padding=1)

        for i in range(len(self.resnet_layers)):
            inplanes = self.resnet_inplanes
            planes = self.resnet_planes[i]
            blocks = self.resnet_layers[i]
            stride = self.resnet_strides[i]
            block_expansion = self.resnet_block_expansion

            exec('self.resnet_layer%s_0_conv1 = nn.Conv2d(inplanes, planes, \
                kernel_size=1, bias=False)' % (i+1))
            exec('self.resnet_layer%s_0_bn1 = nn.BatchNorm2d(planes)' % (i+1))
            exec('self.resnet_layer%s_0_conv2 = nn.Conv2d(planes, planes, \
                kernel_size=3, stride=stride, padding=1, bias=False)' % (i+1))
            exec('self.resnet_layer%s_0_bn2 = nn.BatchNorm2d(planes)' % (i+1))
            exec('self.resnet_layer%s_0_conv3 = nn.Conv2d(planes, planes * 4, \
                kernel_size=1, bias=False)' % (i+1))
            exec('self.resnet_layer%s_0_bn3 = nn.BatchNorm2d( planes * 4)' % \
                (i+1))
            exec('self.resnet_layer%s_0_relu = nn.ReLU(inplace=True)' % (i+1))
            exec('self.resnet_layer%s_0_downsample_0 = nn.Conv2d(inplanes, \
                planes * block_expansion, kernel_size=1, stride=stride, \
                bias=False)' % (i+1))
            exec('self.resnet_layer%s_0_downsample_1 = nn.BatchNorm2d( \
                planes * block_expansion)' % (i+1))

            self.inplanes = planes * block_expansion
            for b in range(1, blocks):
                exec('self.resnet_layer%s_%s_conv1 = nn.Conv2d(inplanes, \
                    planes, kernel_size=1, bias=False)' % (i+1, b))
                exec('self.resnet_layer%s_%s_bn1 = nn.BatchNorm2d(planes)' % \
                    (i+1, b))
                exec('self.resnet_layer%s_%s_conv2 = nn.Conv2d(planes, planes, \
                    kernel_size=3, stride=1, padding=1, bias=False)' % (i+1, b))
                exec('self.resnet_layer%s_%s_bn2 = nn.BatchNorm2d(planes)' % \
                    (i+1, b))
                exec('self.resnet_layer%s_%s_conv3 = nn.Conv2d(planes, \
                    planes * 4, kernel_size=1, bias=False)' % (i+1, b))
                exec('self.resnet_layer%s_%s_bn3 = nn.BatchNorm2d(planes * 4)' \
                    % (i+1, b))
                exec('self.resnet_layer%s_%s_relu = nn.ReLU(inplace=True)' % \
                    (i+1, b))

        # global_net
        self.global_net_output_shape = output_shape
        self.global_net_num_class = num_class

        for i in range(len(self.channel_settings)):
            input_size = self.channel_settings[i]
            output_shape = self.global_net_output_shape
            num_class = self.global_net_num_class
            exec('self.global_net_laterals_%s_0 = nn.Conv2d(input_size, 256, \
                kernel_size=1, stride=1, bias=False)' % i)
            exec('self.global_net_laterals_%s_1 = nn.BatchNorm2d(256)' % i)
            exec('self.global_net_laterals_%s_2 = nn.ReLU(inplace=True)' % i)

            exec('self.global_net_predict_%s_0 = nn.Conv2d(256, 256, \
                kernel_size=1, stride=1, bias=False)' % i)
            exec('self.global_net_predict_%s_1 = nn.BatchNorm2d(256)' % i)
            exec('self.global_net_predict_%s_2 = nn.ReLU(inplace=True)' % i)
            exec('self.global_net_predict_%s_3 = nn.Conv2d(256, num_class, \
                kernel_size=3, stride=1, padding=1, bias=False)' % i)
            exec('self.global_net_predict_%s_4 = nn.Upsample( \
                size=output_shape, mode=\'bilinear\', align_corners=True)' % i)
            exec('self.global_net_predict_%s_5 = nn.BatchNorm2d(num_class)' % i)

            if i != len(self.channel_settings) - 1:
                exec('self.global_net_upsamples_%s_0 = nn.Upsample( \
                    scale_factor=2, mode=\'bilinear\', align_corners=True)' % i)
                exec('self.global_net_upsamples_%s_1 = nn.Conv2d(256, 256, \
                    kernel_size=1, stride=1, bias=False)' % i)
                exec('self.global_net_upsamples_%s_2 = nn.BatchNorm2d(256)' % i)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    
        # refine_net
        self.refine_net_lateral_channel = self.channel_settings[-1]
        self.refine_net_out_shape = output_shape
        self.refine_net_num_class = num_class
        self.refine_net_num_cascade = 4

        for i in range(self.refine_net_num_cascade):
            input_channel = self.refine_net_lateral_channel
            num = self.refine_net_num_cascade-i-1
            output_shape = self.refine_net_out_shape
            for j in range(num):
                inplanes = input_channel
                planes = 128
                stride = 1
                exec('self.refine_net_cascade_%s_%s_conv1 = nn.Conv2d( \
                    inplanes, planes, kernel_size=1, bias=False)' % (i, j))
                exec('self.refine_net_cascade_%s_%s_bn1 = nn.BatchNorm2d( \
                    planes)' % (i, j))
                exec('self.refine_net_cascade_%s_%s_conv2 = nn.Conv2d(planes, \
                    planes, kernel_size=3, stride=stride, padding=1, \
                    bias=False)' % (i, j))
                exec('self.refine_net_cascade_%s_%s_bn2 = nn.BatchNorm2d( \
                    planes)' % (i, j))
                exec('self.refine_net_cascade_%s_%s_conv3 = nn.Conv2d(planes, \
                    planes * 2, kernel_size=1, bias=False)' % (i, j))
                exec('self.refine_net_cascade_%s_%s_bn3 = nn.BatchNorm2d( \
                    planes)' % (i, j))
                exec('self.refine_net_cascade_%s_%s_relu = nn.ReLU( \
                    inplace=True)' % (i, j))
                exec('self.refine_net_cascade_%s_%s_downsample_0 = nn.Conv2d( \
                    inplanes, planes * 2, kernel_size=1, stride=stride, \
                    bias=False)' % (i, j))
                exec('self.refine_net_cascade_%s_%s_downsample_1 = \
                    nn.BatchNorm2d(planes * 2)' % (i, j))
            exec('self.refine_net_cascade_%s_%s = nn.Upsample( \
                size=output_shape, mode=\'bilinear\', align_corners=True)' % \
                (i, num))

        input_channel = 4 * self.refine_net_lateral_channel
        num_class = self.refine_net_num_class
        inplanes = input_channel
        planes = 128
        stride = 1
        self.refine_net_final_predict_0_conv1 = nn.Conv2d(inplanes, planes,
            kernel_size=1, bias=False)
        self.refine_net_final_predict_0_bn1 = nn.BatchNorm2d(planes)
        self.refine_net_final_predict_0_conv2 = nn.Conv2d(planes, planes,
            kernel_size=3, stride=stride, padding=1, bias=False)
        self.refine_net_final_predict_0_bn2 = nn.BatchNorm2d(planes)
        self.refine_net_final_predict_0_conv3 = nn.Conv2d(planes, planes * 2,
            kernel_size=1, bias=False)
        self.refine_net_final_predict_0_bn3 = nn.BatchNorm2d(planes * 2)
        self.refine_net_final_predict_0_relu = nn.ReLU(inplace=True)
        self.refine_net_final_predict_0_downsample_0 = nn.Conv2d(inplanes,
            planes * 2, kernel_size=1, stride=stride, bias=False)
        self.refine_net_final_predict_0_downsample_1 = nn.BatchNorm2d(
            planes * 2)
        self.refine_net_final_predict_1 = nn.Conv2d(256, num_class,
            kernel_size=3, stride=1, padding=1, bias=False)
        self.refine_net_final_predict_2 = nn.BatchNorm2d(num_class)

    def forward(self, x):
        # resnet
        x = self.resnet_conv1(x)
        x = self.resnet_bn1(x)
        x = self.resnet_relu(x)
        x = self.resnet_maxpool(x)

        for i in range(len(self.resnet_layers)):
            residual = x
            exec('out = self.resnet_layer%s_0_conv1(x)' % (i+1))
            exec('out = self.resnet_layer%s_0_bn1(out)' % (i+1))
            exec('out = self.resnet_layer%s_0_conv2(out)' % (i+1))
            exec('out = self.resnet_layer%s_0_bn2(out)' % (i+1))
            exec('out = self.resnet_layer%s_0_conv3(out)' % (i+1))
            exec('out = self.resnet_layer%s_0_bn3(out)' % (i+1))
            exec('residual = self.resnet_layer%s_0_downsample_0(x)' % (i+1))
            exec('residual = self.resnet_layer%s_0_downsample_1(residual)' % \
                (i+1))
            out += residual
            exec('out = self.resnet_layer%s_0_relu(out)' % (i+1))
            x = out

            for b in range(1, blocks):
                residual = x
                exec('out = self.resnet_layer%s_%s_conv1(x)' % (i+1, b))
                exec('out = self.resnet_layer%s_%s_bn1(out)' % (i+1, b))
                exec('out = self.resnet_layer%s_%s_conv2(out)' % (i+1, b))
                exec('out = self.resnet_layer%s_%s_bn2(out)' % (i+1, b))
                exec('out = self.resnet_layer%s_%s_conv3(out)' % (i+1, b))
                exec('out = self.resnet_layer%s_%s_bn3(out)' % (i+1, b))
                out += residual
                exec('out = self.resnet_layer%s_%s_relu(out)' % (i+1, b))
                x = out

        # global_net
        global_fms, global_outs = [], []
        for i in range(len(self.channel_settings)):
            if i == 0:
                exec('feature = self.global_net_laterals_%s_0(x[i])' % i)
                exec('feature = self.global_net_laterals_%s_1(feature)' % i)
                exec('feature = self.global_net_laterals_%s_2(feature)' % i)
            else:
                exec('feature = self.global_net_laterals_%s_0(x[i]) + up' % i)
                exec('feature = self.global_net_laterals_%s_1(feature) + up' % \
                    i)
                exec('feature = self.global_net_laterals_%s_2(feature) + up' % \
                    i)
            global_fms.append(feature)
            if i != len(self.channel_settings) - 1:
                exec('up = self.global_net_upsamples_%s_0(feature)' % i)
                exec('up = self.global_net_upsamples_%s_1(up)' % i)
                exec('up = self.global_net_upsamples_%s_2(up)' % i)
            exec('feature = self.global_net_predict_%s_0(feature)' % i)
            exec('feature = self.global_net_predict_%s_1(feature)' % i)
            exec('feature = self.global_net_predict_%s_2(feature)' % i)
            exec('feature = self.global_net_predict_%s_3(feature)' % i)
            exec('feature = self.global_net_predict_%s_4(feature)' % i)
            exec('feature = self.global_net_predict_%s_5(feature)' % i)
            global_outs.append(feature)
        x = global_fms

        # refine_net
        refine_fms = []
        for i in range(self.refine_net_num_cascade):
            num = self.refine_net_num_cascade-i-1
            for j in range(num):
                exec('x_i = self.refine_net_cascade_%s_%s_conv1(x[i])' % (i, j))
                exec('x_i = self.refine_net_cascade_%s_%s_bn1(x_i)' % (i, j))
                exec('x_i = self.refine_net_cascade_%s_%s_conv2(x_i)' % (i, j))
                exec('x_i = self.refine_net_cascade_%s_%s_bn2(x_i)' % (i, j))
                exec('x_i = self.refine_net_cascade_%s_%s_conv3(x_i)' % (i, j))
                exec('x_i = self.refine_net_cascade_%s_%s_bn3(x_i)' % (i, j))
                exec('x_i = self.refine_net_cascade_%s_%s_relu(x_i)' % (i, j))
                exec('x_i = self.refine_net_cascade_%s_%s_downsample_0(x_i)' % \
                    (i, j))
                exec('x_i = self.refine_net_cascade_%s_%s_downsample_1(x_i)' % \
                    (i, j))
            exec('x_i = self.refine_net_cascade_%s_%s(x_i)' % (i, j))
            refine_fms.append(x_i)

        out = torch.cat(refine_fms, dim=1)
        out = self.refine_net_final_predict_0_conv1(out)
        out = self.refine_net_final_predict_0_bn1(out)
        out = self.refine_net_final_predict_0_conv2(out)
        out = self.refine_net_final_predict_0_bn2(out)
        out = self.refine_net_final_predict_0_conv3(out)
        out = self.refine_net_final_predict_0_bn3(out)
        out = self.refine_net_final_predict_0_relu(out)
        out = self.refine_net_final_predict_0_downsample_0(out)
        out = self.refine_net_final_predict_0_downsample_1(out)
        out = self.refine_net_final_predict_1(out)
        out = self.refine_net_final_predict_2(out)

        return x
