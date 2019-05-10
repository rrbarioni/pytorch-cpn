import math

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

class CPN50(nn.Module):
    def __init__(self, output_shape, num_class):
        super(CPN50, self).__init__()
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

            self.resnet_inplanes = planes * block_expansion
            for b in range(1, blocks):
                inplanes = self.resnet_inplanes
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
                    planes * 2)' % (i, j))
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
        self.x = x
        self.x = self.resnet_conv1(self.x)
        self.x = self.resnet_bn1(self.x)
        self.x = self.resnet_relu(self.x)
        self.x = self.resnet_maxpool(self.x)

        self.x0 = self.x
        for i in range(len(self.resnet_layers)):
            blocks = self.resnet_layers[i]
            exec('self.residual = self.x%s' % (i))
            exec('self.out = self.resnet_layer%s_0_conv1(self.x%s)' % (i+1, i))
            exec('self.out = self.resnet_layer%s_0_bn1(self.out)' % (i+1))
            exec('self.out = self.resnet_layer%s_0_relu(self.out)' % (i+1))

            exec('self.out = self.resnet_layer%s_0_conv2(self.out)' % (i+1))
            exec('self.out = self.resnet_layer%s_0_bn2(self.out)' % (i+1))
            exec('self.out = self.resnet_layer%s_0_relu(self.out)' % (i+1))

            exec('self.out = self.resnet_layer%s_0_conv3(self.out)' % (i+1))
            exec('self.out = self.resnet_layer%s_0_bn3(self.out)' % (i+1))

            exec('self.residual = self.resnet_layer%s_0_downsample_0( \
                self.x%s)' % (i+1, i))
            exec('self.residual = self.resnet_layer%s_0_downsample_1( \
                self.residual)' % (i+1))

            self.out += self.residual
            exec('self.out = self.resnet_layer%s_0_relu(self.out)' % (i+1))
            exec('self.x%s = self.out' % (i+1))

            for b in range(1, blocks):
                exec('self.residual = self.x%s' % (i+1))
                exec('self.out = self.resnet_layer%s_%s_conv1(self.x%s)' % \
                    (i+1, b, i+1))
                exec('self.out = self.resnet_layer%s_%s_bn1(self.out)' % \
                    (i+1, b))
                exec('self.out = self.resnet_layer%s_%s_relu(self.out)' % \
                    (i+1, b))

                exec('self.out = self.resnet_layer%s_%s_conv2(self.out)' % \
                    (i+1, b))
                exec('self.out = self.resnet_layer%s_%s_bn2(self.out)' % \
                    (i+1, b))
                exec('self.out = self.resnet_layer%s_%s_relu(self.out)' % \
                    (i+1, b))

                exec('self.out = self.resnet_layer%s_%s_conv3(self.out)' % \
                    (i+1, b))
                exec('self.out = self.resnet_layer%s_%s_bn3(self.out)' % \
                    (i+1, b))

                self.out += self.residual
                exec('self.out = self.resnet_layer%s_%s_relu(self.out)' % \
                    (i+1, b))
                exec('self.x%s = self.out' % (i+1))
        self.x = [self.x4, self.x3, self.x2, self.x1]

        # global_net
        self.global_fms, self.global_outs = [], []
        for i in range(len(self.channel_settings)):
            if i == 0:
                exec('self.feature = self.global_net_laterals_%s_0(self.x[i])' \
                    % i)
                exec('self.feature = self.global_net_laterals_%s_1( \
                    self.feature)' % i)
                exec('self.feature = self.global_net_laterals_%s_2( \
                    self.feature)' % i)
            else:
                exec('self.feature = self.global_net_laterals_%s_0( \
                    self.x[i]) + self.up' % i)
                exec('self.feature = self.global_net_laterals_%s_1( \
                    self.feature) + self.up' % i)
                exec('self.feature = self.global_net_laterals_%s_2( \
                    self.feature) + self.up' % i)
            self.global_fms.append(self.feature)
            if i != len(self.channel_settings) - 1:
                exec('self.up = self.global_net_upsamples_%s_0(self.feature)' \
                    % i)
                exec('self.up = self.global_net_upsamples_%s_1(self.up)' % i)
                exec('self.up = self.global_net_upsamples_%s_2(self.up)' % i)
            exec('self.feature = self.global_net_predict_%s_0(self.feature)' % \
                    i)
            exec('self.feature = self.global_net_predict_%s_1(self.feature)' % \
                    i)
            exec('self.feature = self.global_net_predict_%s_2(self.feature)' % \
                    i)
            exec('self.feature = self.global_net_predict_%s_3(self.feature)' % \
                    i)
            exec('self.feature = self.global_net_predict_%s_4(self.feature)' % \
                    i)
            exec('self.feature = self.global_net_predict_%s_5(self.feature)' % \
                    i)
            self.global_outs.append(self.feature)
        self.x = self.global_fms

        # refine_net
        self.refine_fms = []
        for i in range(self.refine_net_num_cascade):
            num = self.refine_net_num_cascade-i-1
            for j in range(num):
                self.residual = self.x[i]
                exec('self.out = self.refine_net_cascade_%s_%s_conv1( \
                    self.x[i])' % (i, j))
                exec('self.out = self.refine_net_cascade_%s_%s_bn1(self.out)' \
                    % (i, j))
                exec('self.out = self.refine_net_cascade_%s_%s_relu(self.out)' \
                    % (i, j))

                exec('self.out = self.refine_net_cascade_%s_%s_conv2( \
                    self.out)' % (i, j))
                exec('self.out = self.refine_net_cascade_%s_%s_bn2(self.out)' \
                    % (i, j))
                exec('self.out = self.refine_net_cascade_%s_%s_relu(self.out)' \
                    % (i, j))

                exec('self.out = self.refine_net_cascade_%s_%s_conv3( \
                    self.out)' % (i, j))
                exec('self.out = self.refine_net_cascade_%s_%s_bn3(self.out)' \
                    % (i, j))

                exec('self.residual = \
                    self.refine_net_cascade_%s_%s_downsample_0(self.x[i])' % \
                        (i, j))
                exec('self.residual = \
                    self.refine_net_cascade_%s_%s_downsample_1(self.residual)' \
                        % (i, j))

                self.out += self.residual
                exec('self.out = self.refine_net_cascade_%s_%s_relu(self.out)' \
                    % (i, j))
                self.x[i] = self.out

            exec('self.x[i] = self.refine_net_cascade_%s_%s(self.x[i])' \
                % (i, num))
            self.refine_fms.append(self.x[i])

        self.x = torch.cat(self.refine_fms, dim=1)

        self.residual = self.x
        self.out = self.refine_net_final_predict_0_conv1(self.x)
        self.out = self.refine_net_final_predict_0_bn1(self.out)
        self.out = self.refine_net_final_predict_0_relu(self.out)

        self.out = self.refine_net_final_predict_0_conv2(self.out)
        self.out = self.refine_net_final_predict_0_bn2(self.out)
        self.out = self.refine_net_final_predict_0_relu(self.out)

        self.out = self.refine_net_final_predict_0_conv3(self.out)
        self.out = self.refine_net_final_predict_0_bn3(self.out)

        self.residual = self.refine_net_final_predict_0_downsample_0(self.x)
        self.residual = self.refine_net_final_predict_0_downsample_1( \
            self.residual)
        self.out += self.residual
        self.out = self.refine_net_final_predict_0_relu(self.out)

        self.out = self.refine_net_final_predict_1(self.out)
        self.out = self.refine_net_final_predict_2(self.out)
        self.refine_out = self.out

        return self.global_outs, self.refine_out
