# -*- coding: utf-8 -*-
# @Author  : LG

from .base_model.block import Block, Block1
from .base_model.layer import Conv_Bn_Actication
from .base_model.extra import SppBlock, UpSampleBlock, DownSampleBlock
from torch import nn
import torch
from .struct.predictor import Predictor

class YOLOV4(nn.Module):
    def __init__(self):
        super(YOLOV4, self).__init__()
        self.conv1 = Conv_Bn_Actication(3, 32, 1, 1, 'mish')
        self.block1 = Block1()
        self.block2 = Block(in_channel=64, downsample_num=2)
        self.block3 = Block(in_channel=128, downsample_num=8)
        self.block4 = Block(in_channel=256, downsample_num=8)
        self.block5 = Block(in_channel=512, downsample_num=4)

        self.spp = SppBlock()
        self.upsample1 = UpSampleBlock(in_channel=512)
        self.upsample2 = UpSampleBlock(in_channel=256)

        self.downsample1 = DownSampleBlock(in_channel=128)
        self.downsample2 = DownSampleBlock(in_channel=256)

        self.predictor = Predictor(in_channels=(128, 256, 512))

    def forward(self, x:torch.Tensor):
        x = self.conv1(x)

        x = self.block1(x)
        x = self.block2(x)
        x3 = self.block3(x)
        x4 = self.block4(x3)
        x5 = self.block5(x4)

        x5 = self.spp(x5)
        x4 = self.upsample1(x4, x5)
        x3 = self.upsample2(x3, x4)

        x4 = self.downsample1(x3, x4)
        x5 = self.downsample2(x4, x5)

        out = self.predictor((x3, x4, x5))
        return out

