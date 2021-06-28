import os
import numpy as np

import torch.nn as nn
import torch.nn.init as init
import torch
from option import args


def make_model(args, parent=False):
    return DnCNN_qp(args)


class DnCNN_qp(nn.Module):
    def __init__(self, args, n_channels=64, image_channels=5):
        super(DnCNN_qp, self).__init__()
        kernel_size = 3
        padding = 1
        layers = []
        depth =args.dncnn_depth

        layers.append(nn.Conv2d(in_channels=image_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=True))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(depth-2):
            layers.append(nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(n_channels, eps=0.0001, momentum=0.95))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=n_channels, out_channels=3, kernel_size=kernel_size, padding=padding, bias=False))

        self.dncnn_qp = nn.Sequential(*layers)
        self._initialize_weights()

    def forward(self, x):
        y = x
        out = self.dncnn_qp(x)
        out = out + y[:, 0:3, :, :]
        return out


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.orthogonal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

def downSamplingCb(x):
    upLeftCb = x[:, 1, ::2, ::2]
    downRightCb = x[:, 1, 1::2, 1::2]
    return (upLeftCb + downRightCb + 1) >> 1

def downSamplingCr(x):
    upLeftCr = x[:, 2, :2, ::2]
    downRightCr = x[:, 2, 1::2, 1::2, 2]
    return (upLeftCr + downRightCr + 1) >> 1

# def DownSamplingLuma(lumaPic):
#     pos00 = lumaPic[::2, ::2, :]
#     pos10 = lumaPic[1::2, ::2, :]
#     return (pos00 + pos10 + 1) >> 1


if __name__ == '__main__':
    from help_func.my_torchsummary import summary

    model = DnCNN_qp(args)
    summary(model, (5, 96, 96), device='cpu')