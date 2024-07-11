import math
import torch

import mindspore.nn as nn
from mindspore import Tensor, context
import mindspore.ops as ops
from mindspore.common import dtype as mstype
from mindspore import Parameter
from mindspore import Tensor

import mindspore.numpy as mnp

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size,pad_mode='pad', padding=(kernel_size//2), has_bias=bias)


class MeanShift(nn.Cell):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__()
        self.rgb_range = rgb_range
        self.sign = sign
        std = Tensor(rgb_std, dtype=mnp.float32)
        mean = Tensor(rgb_mean, dtype=mnp.float32)

        weight = mnp.eye(3).reshape(3, 3, 1, 1) / std.reshape(3, 1, 1, 1)
        bias = sign * rgb_range * mean / std

        self.weight = Parameter(weight, requires_grad=False)
        self.bias = Parameter(bias, requires_grad=False)

        self.conv = nn.Conv2d(3, 3, kernel_size=1, has_bias=True, weight_init=self.weight, bias_init=self.bias)
        self.conv.weight.requires_grad = False
        self.conv.bias.requires_grad = False

    def construct(self, x):
        return self.conv(x)


