# -*- coding: utf-8 -*-
# @Author  : LG

from torch import nn
import torch
from .layer import Conv_Bn_Actication, UpSample

from torch.nn import functional as F


class SppBlock(nn.Module):
    def __init__(self):
        super(SppBlock, self).__init__()
        ######
        self.conv1 = Conv_Bn_Actication(1024, 512, 1, 1, 'leaky')
        self.conv2 = Conv_Bn_Actication(512, 1024, 1, 1, 'leaky')
        self.conv3 = Conv_Bn_Actication(1024, 512, 1, 1, 'leaky')

        ### SPP ###
        self.pool1 = nn.MaxPool2d(5, 1, padding=2)
        self.pool2 = nn.MaxPool2d(9, 1, padding=4)
        self.pool3 = nn.MaxPool2d(13, 1, padding=6)

        ###
        self.conv4 = Conv_Bn_Actication(2048, 512, 1, 1, 'leaky')
        self.conv5 = Conv_Bn_Actication(512, 1024, 3, 1, 'leaky')
        self.conv6 = Conv_Bn_Actication(1024, 512, 1, 1, 'leaky')

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = torch.cat([x, self.pool1(x), self.pool2(x), self.pool3(x)], dim=1)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)

        return x


class UpSampleBlock(nn.Module):
    def __init__(self, in_channel): # 512 、 256
        super(UpSampleBlock, self).__init__()

        self.conv1 = Conv_Bn_Actication(in_channel, int(in_channel/2), 1, 1, 'leaky')
        self.upsample = UpSample(stride=2)

        self.conv2 = Conv_Bn_Actication(in_channel, int(in_channel/2), 1, 1, 'leaky')

        self.conv3 = Conv_Bn_Actication(in_channel, int(in_channel/2), 1, 1, 'leaky')
        self.conv4 = Conv_Bn_Actication(int(in_channel/2), in_channel, 3, 1, 'leaky')
        self.conv5 = Conv_Bn_Actication(in_channel, int(in_channel/2), 1, 1, 'leaky')
        self.conv6 = Conv_Bn_Actication(int(in_channel/2), in_channel, 3, 1, 'leaky')
        self.conv7 = Conv_Bn_Actication(in_channel, int(in_channel/2), 1, 1, 'leaky')

    def forward(self, x:torch.Tensor, x1:torch.Tensor) -> torch.Tensor:
        assert x.size(-1) > x1.size(-1)
        x1 = self.conv1(x1)
        x1 = self.upsample(x1)

        x = self.conv2(x)

        x = torch.cat([x, x1], dim=1)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)

        return x


class DownSampleBlock(nn.Module):
    def __init__(self, in_channel):
        super(DownSampleBlock, self).__init__()
        self.conv1 = Conv_Bn_Actication(in_channel, int(in_channel*2), 3, 2, 'leaky')

        self.conv2 = Conv_Bn_Actication(int(in_channel*4), int(in_channel*2), 1, 1, 'leaky')
        self.conv3 = Conv_Bn_Actication(int(in_channel*2), int(in_channel*4), 3, 1, 'leaky')
        self.conv4 = Conv_Bn_Actication(int(in_channel*4), int(in_channel*2), 1, 1, 'leaky')
        self.conv5 = Conv_Bn_Actication(int(in_channel*2), int(in_channel*4), 3, 1, 'leaky')
        self.conv6 = Conv_Bn_Actication(int(in_channel*4), int(in_channel*2), 1, 1, 'leaky')

    def forward(self, x, x1):
        assert x.size(-1) > x1.size(-1) # 这里对输入的俩层特征图做一些限制
        assert x.size(1)*2 == x1.size(1)

        x = self.conv1(x)

        x = torch.cat([x, x1], dim=1)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        return x


class Branch(nn.Module):
    def __init__(self, in_channel, num_classes=21, num_anchor=3):
        super(Branch, self).__init__()
        self.conv1 = Conv_Bn_Actication(in_channel, int(in_channel*2), 3, 1, 'leaky')
        self.conv2 = Conv_Bn_Actication(int(in_channel*2), (num_classes+4)*num_anchor, 1, 1, 'linear')

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x