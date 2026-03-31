__author__ = 'Fei Zhou'

import torch
import torch.nn as nn
from model.common import laplacian_roll, Laplacian_Conv
roll = torch.roll

def make_model(args, parent=False):
    return toy_che(args)

# NOTE: channel first
# def laplacian(a):
#     return laplacian_roll(a, lvl=2, dim=2, type='pt')
# def laplacian(a):
#     return roll(a,1,2) + roll(a,-1,2) + roll(a,1,3) + roll(a,-1,3) -4*a

class toy_che(nn.Module):
    def __init__(self, options):
        super(toy_che, self).__init__()
        assert options.dim==2, 'ERROR toy che requires 2D'
        self.D = 0.01
        self.laplacian = Laplacian_Conv(dim=2) #laplacian
        self.register_parameter('dummy', nn.Parameter(torch.tensor(1., requires_grad=True)))

    def forward(self, a):
        return a + self.laplacian(a**3-a-self.laplacian(a))* self.D + 0.0*self.dummy
