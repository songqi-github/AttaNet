#!/usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn


class ConvBNReLU(nn.Module):

    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1,
                 dilation=1, groups=1, bias=False):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(
                in_chan, out_chan, kernel_size=ks, stride=stride,
                padding=padding, dilation=dilation,
                groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        feat = self.conv(x)
        feat = self.bn(feat)
        feat = self.relu(feat)
        return feat


class Branch(nn.Module):

    def __init__(self):
        super(Branch, self).__init__()
        self.S1_1 = ConvBNReLU(3, 32, ks=7, stride=2, padding=3)
        self.S1_2 = ConvBNReLU(32, 64, 3, stride=2)
        self.S2_1 = ConvBNReLU(64, 64, 3, stride=2)
        self.S2_2 = ConvBNReLU(64, 64, 3, stride=1)
        self.S2_3 = ConvBNReLU(64, 64, 3, stride=1)
        self.S3_1 = ConvBNReLU(64, 128, 3, stride=2)
        self.S3_2 = ConvBNReLU(128, 128, 3, stride=1)
        self.S3_3 = ConvBNReLU(128, 128, 3, stride=1)

    def forward(self, x):
        feat = self.S1_1(x)
        feat = self.S1_2(feat)
        feat = self.S2_1(feat)
        feat = self.S2_2(feat)
        feat8 = self.S2_3(feat)
        feat = self.S3_1(feat8)
        feat = self.S3_2(feat)
        feat16 = self.S3_3(feat)
        return feat8, feat16, feat


class Backbone(nn.Module):
    def __init__(self):
        super(Backbone, self).__init__()
        self.branch = Branch()
        self.init_weight()

    def forward(self, x):
        feat_d = self.branch(x)
        return feat_d

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params

