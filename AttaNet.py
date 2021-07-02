#!/usr/bin/python
# -*- encoding: utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F

from backbone import ResNet18
from torch.nn import BatchNorm2d


class ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, *args, **kwargs):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan,
                              out_chan,
                              kernel_size=ks,
                              stride=stride,
                              padding=padding,
                              bias=False)
        self.bn = BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)


class AttaNetOutput(nn.Module):
    def __init__(self, in_chan, mid_chan, n_classes, *args, **kwargs):
        super(AttaNetOutput, self).__init__()
        self.conv = ConvBNReLU(in_chan, mid_chan, ks=3, stride=1, padding=1)
        self.dropout = nn.Dropout2d(0.1)
        self.conv_out = nn.Conv2d(mid_chan, n_classes, kernel_size=1, bias=False)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.dropout(x)
        x = self.conv_out(x)
        return x

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
            elif isinstance(module, BatchNorm2d):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


class StripAttentionModule(nn.Module):
    def __init__(self, in_chan, out_chan, *args, **kwargs):
        super(StripAttentionModule, self).__init__()
        self.conv1 = ConvBNReLU(in_chan, 64, ks=1, stride=1, padding=0)
        self.conv2 = ConvBNReLU(in_chan, 64, ks=1, stride=1, padding=0)
        self.conv3 = ConvBNReLU(in_chan, out_chan, ks=1, stride=1, padding=0)
        self.softmax = nn.Softmax(dim=1)

        self.init_weight()

    def forward(self, x):
        q = self.conv1(x)
        batchsize, c_middle, h, w = q.size()
        q = F.avg_pool2d(q, [h, 1])
        q = q.view(batchsize, c_middle, -1).permute(0, 2, 1)

        k = self.conv2(x)
        k = k.view(batchsize, c_middle, -1)
        attention_map = torch.bmm(q, k)
        attention_map = self.softmax(attention_map)

        v = self.conv3(x)
        c_out = v.size()[1]
        v = F.avg_pool2d(v, [h, 1])
        v = v.view(batchsize, c_out, -1)

        augmented_feature_map = torch.bmm(v, attention_map)
        augmented_feature_map = augmented_feature_map.view(batchsize, c_out, h, w)
        out = x + augmented_feature_map
        return out

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
            elif isinstance(module, BatchNorm2d):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


class AttentionFusionModule(nn.Module):
    def __init__(self, in_chan, out_chan, *args, **kwargs):
        super(AttentionFusionModule, self).__init__()
        self.conv = ConvBNReLU(in_chan, out_chan, ks=1, stride=1, padding=0)
        self.conv_atten = nn.Conv2d(out_chan, out_chan, kernel_size=1, bias=False)
        self.bn_atten = BatchNorm2d(out_chan)
        self.sigmoid_atten = nn.Sigmoid()

        self.init_weight()

    def forward(self, feat16, feat32):
        feat32_up = F.interpolate(feat32, feat16.size()[2:], mode='nearest')
        fcat = torch.cat([feat16, feat32_up], dim=1)
        feat = self.conv(fcat)

        atten = F.avg_pool2d(feat, feat.size()[2:])
        atten = self.conv_atten(atten)
        atten = self.bn_atten(atten)
        atten = self.sigmoid_atten(atten)
        return atten

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
            elif isinstance(module, BatchNorm2d):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


class AttaNetHead(nn.Module):
    def __init__(self, *args, **kwargs):
        super(AttaNetHead, self).__init__()
        self.resnet = ResNet18()
        self.afm = AttentionFusionModule(640, 128)
        self.conv_head32 = ConvBNReLU(512, 128, ks=3, stride=1, padding=1)
        self.conv_head16 = ConvBNReLU(256, 128, ks=3, stride=1, padding=1)
        self.conv_head1 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
        self.sam = StripAttentionModule(128, 128)
        self.conv_head2 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
        self.init_weight()

    def forward(self, x):
        feat8, feat16, feat32 = self.resnet(x)
        h8, w8 = feat8.size()[2:]
        h16, w16 = feat16.size()[2:]

        # Attention Fusion Module
        feat16 = self.conv_head16(feat16)
        atten = self.afm(feat16, feat32)
        feat32 = self.conv_head32(feat32)
        feat32 = torch.mul(feat32, atten)
        feat32_up = F.interpolate(feat32, (h16, w16), mode='nearest')
        feat16 = torch.mul(feat16, (1 - atten))
        feat16_sum = feat16 + feat32_up

        # feature smoothness
        feat16_sum = self.conv_head1(feat16_sum)

        # Strip Attention Module
        feat16_sum = self.sam(feat16_sum)
        feat16_up = F.interpolate(feat16_sum, (h8, w8), mode='nearest')
        feat16_up = self.conv_head2(feat16_up)

        return feat16_up, feat32_up, feat8

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
            elif isinstance(module, BatchNorm2d):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


class AttaNet(nn.Module):
    def __init__(self, n_classes, *args, **kwargs):
        super(AttaNet, self).__init__()
        self.head = AttaNetHead()
        self.conv_out = AttaNetOutput(128, 128, n_classes)
        self.conv_out1 = AttaNetOutput(128, 64, n_classes)
        self.conv_out2 = AttaNetOutput(128, 64, n_classes)
        self.init_weight()

    def forward(self, x):
        h, w = x.size()[2:]
        out, auxout1, auxout2 = self.head(x)

        feat_out = self.conv_out(out)
        feat_aux1 = self.conv_out1(auxout1)
        feat_aux2 = self.conv_out2(auxout2)

        feat_out = F.interpolate(feat_out, (h, w), mode='bilinear', align_corners=True)
        feat_aux1 = F.interpolate(feat_aux1, (h, w), mode='bilinear', align_corners=True)
        feat_aux2 = F.interpolate(feat_aux2, (h, w), mode='bilinear', align_corners=True)
        return feat_out, feat_aux1, feat_aux2

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = [], [], [], []
        for name, child in self.named_children():
            child_wd_params, child_nowd_params = child.get_params()
            if isinstance(child, AttaNetOutput):
                lr_mul_wd_params += child_wd_params
                lr_mul_nowd_params += child_nowd_params
            else:
                wd_params += child_wd_params
                nowd_params += child_nowd_params
        return wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params


