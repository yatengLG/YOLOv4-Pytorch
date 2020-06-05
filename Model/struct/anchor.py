# -*- coding: utf-8 -*-
# @Author  : LG

import torch
from Utils.Boxs_op import center_form_to_corner_form, corner_form_to_center_form
class Anchor():

    def __init__(self, clip=False):
        self.features_sizes = [76, 38, 19]
        self.anchors = [[[12, 16], [19, 36], [40, 28]],
                        [[36, 75], [76, 55], [72, 146]],
                        [[142, 110], [192, 243], [459, 401]]]
        self.image_size = 608
        self.clip = clip
    def __call__(self):
        priors = []
        for k, (features_size) in enumerate(self.features_sizes):
            for i in range(features_size):
                for j in range(features_size):
                    cx = (i+0.5) / features_size
                    cy = (j+0.5) / features_size
                    for anchor in self.anchors[k]:
                        cw = anchor[0]/self.image_size
                        ch = anchor[1]/self.image_size

                        priors.append([cx, cy, cw, ch])

        priors = torch.tensor(priors)
        if self.clip:   # 对超出图像范围的框体进行截断
            priors = center_form_to_corner_form(priors) # 截断时,先转为 [xmin, ymin, xmin, xmax]形式
            priors.clamp_(max=1, min=0)
            priors = corner_form_to_center_form(priors)

        return priors

if __name__ == '__main__':
    priors = Anchor()()
    for prior in priors:
        print(prior)
    print(len(priors))