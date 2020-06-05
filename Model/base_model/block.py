# -*- coding: utf-8 -*-
# @Author  : LG

import torch
from torch import nn
from .layer import Conv_Bn_Actication


class ShortCut(nn.Module):
    def __init__(self, in_channel, mid_channel, out_channel):
        super(ShortCut, self).__init__()

        self.conv1 = Conv_Bn_Actication(in_channel, mid_channel, 1, 1, 'mish')
        self.conv2 = Conv_Bn_Actication(mid_channel, out_channel, 3, 1, 'mish')

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x_copy = x
        x = self.conv1(x)
        x = self.conv2(x)
        return x+x_copy


class Block(nn.Module):
    def __init__(self, in_channel, downsample_num):
        super(Block, self).__init__()
        self.conv1 = Conv_Bn_Actication(in_channel, in_channel*2, 3, 2, 'mish')

        self.conv2 = Conv_Bn_Actication(in_channel*2, in_channel, 1, 1, 'mish')
        self.shortcuts = nn.ModuleList()
        for i in range(downsample_num):
            self.shortcuts.append(ShortCut(in_channel, in_channel, in_channel))
        self.conv4 = Conv_Bn_Actication(in_channel, in_channel, 1, 1, 'mish')

        self.conv5 = Conv_Bn_Actication(in_channel*2, in_channel, 1, 1, 'mish')
        self.conv6 = Conv_Bn_Actication(in_channel*2, in_channel*2, 1, 1, 'mish')

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x1 = self.conv1(x)

        x = self.conv2(x1)
        for shortcut in self.shortcuts:
            x = shortcut(x)
        x = self.conv4(x)

        x = torch.cat((self.conv5(x1), self.conv4(x)), dim=1)
        x = self.conv6(x)
        return x


# 第一个downsample结构与其他结构有些许区别，这里单独做
class Block1(nn.Module):
    def __init__(self):
        super(Block1, self).__init__()
        self.conv1 = Conv_Bn_Actication(32, 64, 3, 2, 'mish')
        self.conv2 = Conv_Bn_Actication(64, 64, 1, 1, 'mish')
        self.shortcut = ShortCut(64, 32, 64)
        self.conv4 = Conv_Bn_Actication(64, 64, 1, 1, 'mish')
        self.conv5 = Conv_Bn_Actication(64, 64, 1, 1, 'mish')
        self.conv6 = Conv_Bn_Actication(128, 64, 1, 1, 'mish')

    def forward(self, x:torch.Tensor) -> torch.Tensor:

        x1 = self.conv1(x)

        x = self.conv2(x1)
        x = self.shortcut(x)
        x = self.conv4(x)

        x = torch.cat((self.conv5(x1), self.conv4(x)), dim=1)
        x = self.conv6(x)
        return x


