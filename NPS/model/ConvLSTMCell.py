__author__ = 'yunbo'

import torch
import torch.nn as nn
from model.common import ConvND

class ConvLSTMCell(nn.Module):
    def __init__(self, in_channel, num_hidden, width, kernel_size, stride, layer_norm, dim=2, periodic=False):
        super(ConvLSTMCell, self).__init__()

        self.num_hidden = num_hidden
        # self._forget_bias = 1.0
        self.conv = nn.Sequential(
            ConvND(in_channel + num_hidden, num_hidden * 4, kernel_size=kernel_size, stride=stride, dim=dim, periodic=periodic),
            #nn.LayerNorm([num_hidden * 4, width, width])
            nn.GroupNorm(1,num_hidden * 4, affine=True)
        )



    def forward(self, x, h, c):
        combined = torch.cat([x, h], dim=1)

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.num_hidden, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next
