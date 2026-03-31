import numpy as np
import torch
import torch.nn as nn

from model.common import TwoLayerNet, roll, trace, laplacian_roll, gradient, divergence #, GradientNorm, Laplacian
from .stochastic_evolution import StochPhaseEvolution

def make_model(args, parent=False):
    return StochasticCHE(args)

#conv = NNop[(dim,'conv')](channel // reduction, channel, 1, padding=0, bias=True)

class CHE(nn.Module):
    def __init__(self, nfeat, dim, periodic):
        super(CHE, self).__init__()
        self.dim = dim
        self.periodic = periodic
        self.freeE = TwoLayerNet(1, nfeat, 1)
        self.mobility = TwoLayerNet(1, nfeat, 1)
        self.stiffness = TwoLayerNet(1, nfeat, 1)
        self.m1=[self.freeE, self.mobility, self.stiffness]

    def forward(self, x_gen):
        freeE = self.freeE(x_gen)
        mobility = self.mobility(x_gen)
        stiffness = self.stiffness(x_gen)
        lapl_c = laplacian_roll(x_gen, 1, self.dim, 'pt')
        mu = freeE - stiffness*lapl_c
        return divergence(mobility*gradient(mu, self.dim, self.periodic), self.dim, self.periodic)
        # return mobility*laplacian_roll(mu, 1, self.dim, 'pt')
        # return laplacian_roll(mu, 1, self.dim, 'pt')


class StochasticCHE(StochPhaseEvolution):
    def init_mean(self):
        self.G0 = CHE(configs.n_feats, self.dim, self.periodic)
        self.m1 = self.G0.m1
