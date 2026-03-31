__author__ = 'Fei Zhou'

import torch
import torch.nn as nn
import model
from model.common import ConvND
from model.ConvLSTMCell import ConvLSTMCell
from model.convlstm import convlstm
from model.resnet import resnet, resblock

def make_model(args, parent=False):
    return resnetlstm(len(args.num_hidden), args.num_hidden, args)

class ResNetLSTMCell(ConvLSTMCell):
    def __init__(self, in_channel, num_hidden, width, kernel_size, stride, bn=False, dim=2, periodic=False):
        # super(ResNetLSTMCell, self).__init__(in_channel, num_hidden, width, kernel_size, stride, bn, dim=dim, periodic=periodic)
        nn.Module.__init__(self)
        self.num_hidden = num_hidden[0]
        self.conv = resnet(in_channel + num_hidden[0], [num_hidden[0]*4]*len(num_hidden), kernel_size, dim=dim, periodic=periodic, bn=bn)
        # self.conv = nn.Sequential(
        #     ConvND(in_channel + num_hidden, num_hidden * 4, kernel_size=kernel_size, stride=stride, dim=dim, periodic=periodic),
        #     #nn.LayerNorm([num_hidden * 4, width, width])
        # )


class resnetlstm(convlstm):
    def init_layers(self):
        print("resnetlstm init")
        cell_list = []
        for i in range(1):
            in_channel = self.frame_channel if i == 0 else self.num_hidden[i-1]
            cell_list.append(
                ResNetLSTMCell(in_channel, self.num_hidden, self.shape, self.args.kernel_size,
                  self.args.stride, bn=self.args.bn, dim=self.dim, periodic=self.periodic)
            )
        self.cell_list = nn.ModuleList(cell_list)
        self.num_layers = 1
