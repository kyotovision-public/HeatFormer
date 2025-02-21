# ------------------------------------------------------------------------------
# multiview.pose3d.torch
# Copyright (c) 2018-present Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Chunyu Wang (chnuwa@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging

import torch
import torch.nn as nn
import torchvision.transforms.functional as F

from einops import rearrange

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class PoseResNet(nn.Module):

    def __init__(self, block, layers, cfg, **kwargs):
        self.inplanes = 64
        self.deconv_with_bias = cfg.POSE_RESNET.DECONV_WITH_BIAS

        super(PoseResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # used for deconv layers
        self.deconv_layers = self._make_deconv_layer(
            cfg.POSE_RESNET.NUM_DECONV_LAYERS,
            cfg.POSE_RESNET.NUM_DECONV_FILTERS,
            cfg.POSE_RESNET.NUM_DECONV_KERNELS,
        )

        self.final_layer = nn.Conv2d(
            in_channels=cfg.POSE_RESNET.NUM_DECONV_FILTERS[-1],
            # out_channels=cfg.DATASET.NUM_JOINTS,
            out_channels=33,
            kernel_size=cfg.POSE_RESNET.FINAL_CONV_KERNEL,
            stride=1,
            padding=1 if cfg.POSE_RESNET.FINAL_CONV_KERNEL == 3 else 0
        )

        self.cfg = cfg

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        """(batch, chanel, h, w) h=w=384"""
        x = self.conv1(x)    # (batch 64 h/2 w/2)
        x = self.bn1(x)      # (batch 64 h/2 w/2)
        x = self.relu(x)     # (batch 64 h/2 w/2)
        x = self.maxpool(x)  # (batch 64 h/4 w/4)

        x = self.layer1(x)   # (batch 256 h/4 w/4)
        x = self.layer2(x)   # (batch 512 h/8 w/8)
        x = self.layer3(x)   # (batch 1024 h/16 w/16)
        x = self.layer4(x)   # (batch 2048 h/32 w/32)

        img_feat = x

        x = self.deconv_layers(x)
        x_final_feature = x
        x = self.final_layer(x)

        return x, x_final_feature, img_feat
    
    def forward_fusion(self, heatmap, inp, type):
        """
        heatmap : (batch * n_view n_joints h w)
        mid_feat : (batch * n_view C h w)
        """
        # normalize heatmap [0, 1]
        heatmap_max = heatmap.max(dim=-1)[0].max(dim=-1)[0]  # (batch * n_view n_joints)
        heatmap_min = heatmap.min(dim=-1)[0].min(dim=-1)[0]  # (batch * n_view n_joints)
        
        # normalize
        normalized_heatmap = (heatmap - heatmap_min[..., None, None]) / ((heatmap_max - heatmap_min)[..., None, None]) # (batch * n_view n_joints h w)
        
        # kp fuse
        fused_heatmap = normalized_heatmap.max(dim=1)[0] # (batch * n_view h w)

        # heatmap resize
        inp_size = self.cfg.DATASET.IMG_SIZE
        fuse_heatmap_resize = F.resize(img=fused_heatmap, size=(inp_size, inp_size)) # (batch * n_view inp_h inp_w)

        if type == 'normal':
            fuse_feat = fuse_heatmap_resize.unsqueeze(1) + inp        # (batch * n_view 3 inp_h inp_w)
        elif type == 'average':
            fuse_feat = (fuse_heatmap_resize.unsqueeze(1) + inp) / 2  # (batch * n_view 3 inp_h inp_w)
        elif type == 'rescale':
            inp_min = inp.min(dim=-1)[0].min(dim=-1)[0]               # (batch * n_view 3)
            inp_max = inp.max(dim=-1)[0].max(dim=-1)[0]               # (batch * n_view 3)

            # [0,1] -> [inp_min, inp_max]
            fuse_heatmap_ = fuse_heatmap_resize.unsqueeze(1).repeat(1, 3, 1, 1)    # (batch * n_view 3 inp_h inp_w)
            fuse_heatmap_resize_rescale = fuse_heatmap_ * (inp_max - inp_min)[..., None, None] + inp_min[..., None, None] # (batch * n_view 3 inp_h inp_w)
            fuse_feat = (fuse_heatmap_resize_rescale + inp) / 2

        _, _, img_feat = self.forward(fuse_feat)

        return img_feat

    def init_weights(self, pretrained=''):
        if os.path.isfile(pretrained):
            pretrained_state_dict = torch.load(pretrained)
            logger.info('=> loading pretrained model {}'.format(pretrained))

            logger.info('=> delete last 1x1 conv weights and bias !')
            if 'final_layer.weight' in pretrained_state_dict:
                del pretrained_state_dict['final_layer.weight']
            if 'final_layer.bias' in pretrained_state_dict:
                del pretrained_state_dict['final_layer.bias']

            self.load_state_dict(pretrained_state_dict, strict=False)

            logger.info('=> init deconv weights from normal distribution')
            for name, m in self.deconv_layers.named_modules():
                if isinstance(m, nn.ConvTranspose2d):
                    logger.info('=> init {}.weight as normal(0, 0.001)'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.normal_(m.weight, std=0.001)
                    if self.deconv_with_bias:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    logger.info('=> init {}.weight as 1'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            logger.info('=> init final conv weights from normal distribution')
            for m in self.final_layer.modules():
                if isinstance(m, nn.Conv2d):
                    # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    logger.info('=> init {}.weight as normal(0, 0.001)'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.normal_(m.weight, std=0.001)
                    nn.init.constant_(m.bias, 0)
        else:
            logger.info('=> init weights from normal distribution')
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    nn.init.normal_(m.weight, std=0.001)
                    # nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.ConvTranspose2d):
                    nn.init.normal_(m.weight, std=0.001)
                    if self.deconv_with_bias:
                        nn.init.constant_(m.bias, 0)

resnet_spec = {18: (BasicBlock, [2, 2, 2, 2]),
               34: (BasicBlock, [3, 4, 6, 3]),
               50: (Bottleneck, [3, 4, 6, 3]),
               101: (Bottleneck, [3, 4, 23, 3]),
               152: (Bottleneck, [3, 8, 36, 3])}


def get_pose_net(cfg, **kwargs):
    num_layers = cfg.POSE_RESNET.NUM_LAYERS

    block_class, layers = resnet_spec[num_layers]

    model = PoseResNet(block_class, layers, cfg, **kwargs)

    return model
