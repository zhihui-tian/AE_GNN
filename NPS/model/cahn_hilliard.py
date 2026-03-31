__author__ = 'Fei Zhou'

import torch
import torch.nn as nn
import torch.nn.functional as F
from NPS.model.common import CNN, Laplacian_Conv, Divergence_Gradient_symm, TwoPointDiffusivity, Gradient_Conv, Divergence_Conv
roll = torch.roll

def make_model(args, parent=False):
    return cahn_hilliard_equation(args)

class cahn_hilliard_equation(nn.Module):
    def __init__(self, configs):
        super(cahn_hilliard_equation, self).__init__()
        self.dim=configs.dim
        self.mode = 'circular' if configs.periodic else 'zeros'
        import ast
        setting = ast.literal_eval(configs.model_setting)
        self.mobility_type = int(setting.get('mobility_type','1'))
        offset = 1
        if self.mobility_type == 2: # mobility(a, b) rather than (mobility(a)+mobility(b))/2
            print(f'WARNING: likely overly complicated mobility(c1,c2) and overkill. debug only')
            self.mobility = TwoPointDiffusivity(configs.n_colors, configs.num_hidden, 1, configs.kernel_size, dim=self.dim, activation=configs.act, periodic=configs.periodic)
        else:
            self.mobility = CNN(configs.n_colors, configs.num_hidden, 1, configs.kernel_size, dim=self.dim, activation=configs.act, periodic=configs.periodic)
            if self.mobility_type == -1: # mobility(a), asymmetric, therefore non-conservative. Only for debugging
                print(f'WARNING: NOT strictly conservative! debug only')
                self.gradient = Gradient_Conv(dim=self.dim, periodic=configs.periodic, offset=offset)
                self.divergence = Divergence_Conv(dim=self.dim, periodic=configs.periodic, offset=offset)
        self.stiffness= CNN(configs.n_colors, configs.num_hidden, 1, configs.kernel_size, dim=self.dim, activation=configs.act, periodic=configs.periodic)
        self.chem_pot = CNN(configs.n_colors, configs.num_hidden, 1, configs.kernel_size, dim=self.dim, activation=configs.act, periodic=configs.periodic)
        self.laplacian = Laplacian_Conv(dim=self.dim, periodic=configs.periodic)
        self.div_grad = Divergence_Gradient_symm(dim=self.dim, periodic=configs.periodic, offset=offset)

    def forward(self, c):
        mu = self.chem_pot(c) - self.stiffness(c)*self.laplacian(c[:,:1])
        if self.mobility_type == 2:
            print('TBD')
            # return self.div_grad(self.mobility(c,one_point_test=False), mu, a_is_vector=True)
        elif self.mobility_type == -1:
            return self.divergence(self.mobility(c)*self.gradient(mu))
        else: # (mobility(a)+mobility(b))/2
            return self.div_grad(self.mobility(c), mu)

