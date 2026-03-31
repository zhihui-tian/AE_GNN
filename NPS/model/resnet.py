__author__ = 'Fei Zhou'

import torch
import torch.nn as nn
from model.common import ConvND, NNop


class resblock(nn.Module):
    def __init__(self, n_feat, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), dim=2, periodic=False):
        super(resblock, self).__init__()
        m = []
        for i in range(2):
            if bn: m.append(NNop[(dim,'bnorm')](n_feat))
#            if i == 0: m.append(act)
            if act is not None: m.append(act)
            m.append(ConvND(n_feat, n_feat, kernel_size, dim, periodic=periodic, bias=bias))
        self.body = nn.Sequential(*m)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res

class resnet(nn.Module):
    def __init__(self, in_feats, num_hidden, kernel, dim=2, periodic=False, bn=True, *args):
        super(resnet, self).__init__()
        # assert configs.kernel_size == 3:
        m = [ConvND(in_feats, num_hidden[0], kernel, dim, periodic, *args)]
        for i, nhid in enumerate(num_hidden):
            m.append(resblock(nhid, kernel, bn=bn, dim=dim, periodic=periodic))

        self.body = torch.nn.Sequential(*m)

    def forward(self, x_gen):
        # x_gen = xs_gen
        return self.body(x_gen)

