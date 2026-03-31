__author__ = 'yunbo'

import torch
import torch.nn as nn
# from model.common import ConvND
from .SpatioTemporalLSTMCell import SpatioTemporalLSTMCell
from ._base import _base_ConvRNN


class predrnn_v1(_base_ConvRNN):
    def init_layers(self):
        print("Initializing PredRNN_v1")
        cell_list = []
        for i in range(self.num_layers):
            in_channel = self.nfeat_in if i == 0 else self.num_hidden[i-1]
            cell_list.append(
                SpatioTemporalLSTMCell(in_channel, self.num_hidden[i], self.args.kernel_size,
                                       self.args.stride, True, dim=self.dim, periodic=self.periodic)
            )
        self.cell_list = nn.ModuleList(cell_list)

    def initHidden(self, x):
        x_shape = x.shape
        self.h_t = []
        self.c_t = []
        for i in range(self.num_layers):
            zeros = torch.zeros(x_shape[0], self.num_hidden[i], *x_shape[2:], device=x.device)
            self.h_t.append(zeros)
            self.c_t.append(zeros)
        self.memory = torch.zeros(x_shape[0], self.num_hidden[0], *x_shape[2:], device=x.device)

    def forward(self, x, reset=False, **kwx):
        if reset:
            self.initHidden(x) # init Hidden at each forward start

        for i in range(self.num_layers):
            self.h_t[i], self.c_t[i], self.memory = self.cell_list[i](self.h_t[i - 1] if i>0 else x, self.h_t[i], self.c_t[i], self.memory)
        return self.h_t[self.num_layers-1]
