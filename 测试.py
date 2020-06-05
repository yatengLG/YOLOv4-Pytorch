# -*- coding: utf-8 -*-
# @Author  : LG

from Model.yolo import YOLOV4
import torch
import numpy as np
from Data.Dataset_VOC import vocdataset
from Configs import _C as cfg
from Data import transfrom,targettransform

# train_dataset=vocdataset(cfg, is_train=True, transform=transfrom(cfg,is_train=True),
#                          target_transform=targettransform(cfg))


model = YOLOV4()
model.to('cuda')

x = torch.ones(size=(1,3,608,608))
x = x.to('cuda')

out = model(x)
print(out.size())