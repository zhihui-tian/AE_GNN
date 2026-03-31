__author__ = 'Fei Zhou'

import torch
import torch.nn as nn

from model.simple_diffusion import SimpleNNDiffusion
roll = torch.roll

# def noise_loss(args, type=None):
#     if type == 'SimpleNNDiffusion':
#         return SimpleNNDiffusion(args)
#     else:
#         raise ValueError(f'ERROR unknown noise loss {type}')
def make_model(args):
    return ToyNNDiffusion(args)

class ToyNNDiffusion(SimpleNNDiffusion):
    class _toy_correlation(nn.Module):
        def forward(self, x):
            return nn.functional.relu(1-x[:,0:1]**2) * nn.functional.relu(1-x[:,1:2]**2)*(0.03**2)

    def _init_model(self, configs):
        self.Hrrp = ToyNNDiffusion._toy_correlation()

