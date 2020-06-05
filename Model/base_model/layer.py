# -*- coding: utf-8 -*-
# @Author  : LG

from torch import nn
import torch
from torch.nn import functional as F
import math

class Conv_Bn_Actication(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, activation='mish', bias=False, bn=True):
        super(Conv_Bn_Actication, self).__init__()

        padding = math.floor(kernel_size / 2)

        self.layers = nn.ModuleList()
        self.layers.append(
            nn.Conv2d(in_channels=in_channel,
                      out_channels=out_channel,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=padding,
                      bias=bias)
        )
        if bn:
            self.layers.append(
                nn.BatchNorm2d(out_channel)
            )
        if activation =='mish':
            self.layers.append(
                Mish()
            )
        elif activation == 'leaky':
            self.layers.append(
                nn.LeakyReLU(0.1, inplace=True)
            )
        elif activation == 'linear':
            pass
        else:
            raise ValueError('Activation only support mish or leaky, but get {}'.format(activation))

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class UpSample(nn.Module):
    def __init__(self, stride=2):
        super(UpSample, self).__init__()
        self.stride = stride

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=self.stride)
        return x


class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        # x = x * nn.Tanh(nn.Softplus(x))
        x = x * (torch.tanh(F.softplus(x)))
        return x