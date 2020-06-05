# -*- coding: utf-8 -*-
# @Author  : LG

from torch import nn
import torch
from ..base_model.extra import Branch

class Predictor(nn.Module):
    def __init__(self, in_channels:tuple):
        super(Predictor, self).__init__()
        self.layers = nn.ModuleList()
        for in_channel in in_channels:
            self.layers.append(Branch(in_channel=in_channel, num_classes=21, num_anchor=3))

    def forward(self, features:tuple) -> torch.Tensor:
        assert features[0].size(-1) > features[-1].size(-1)
        assert len(features) == len(self.layers)
        outs = []
        for branch, feature in zip(self.layers, features):
            out = branch(feature)       # [B, 255, 76, 76]
            out = out.permute(0, 2, 3, 1).view(out.size(0), -1, out.size(1))
            outs.append(out)

        outs = torch.cat(outs, dim=1)
        outs = outs.view(outs.size(0), -1, 21+4)
        return outs
