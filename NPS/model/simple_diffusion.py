__author__ = 'Fei Zhou'

import torch
import torch.nn as nn

import model
from model.common import MLP
roll = torch.roll

# def noise_loss(args, type=None):
#     if type == 'SimpleNNDiffusion':
#         return SimpleNNDiffusion(args)
#     else:
#         raise ValueError(f'ERROR unknown noise loss {type}')
def make_model(args):
    return SimpleNNDiffusion(args)

class SimpleNNDiffusion(model.NPSModel):
    def __init__(self, configs):
        super(SimpleNNDiffusion, self).__init__(configs)
        assert self.periodic, ValueError('ERROR SimpleNNDiffusion requires periodic boundary condition')
        self._init_model(configs)

    def _init_model(self, configs):
        # self.Hrrp = TwoLayerNet(2, configs.stoch_n_feats, self.out_feats, symmetric=True, conv=True, dim=self.dim, activation='relu')
        self.Hrrp = MLP(2, configs.stoch_hidden, self.out_feats, symmetric=True, dim=self.dim, squared=False)

    def calc_var(self, c0, ij_only=False):
        omega_ij = -torch.cat([self.Hrrp(torch.cat((c0, roll(c0, 1,i+2)),1)) for i in range(self.dim)], 1)
        if ij_only:
            return omega_ij
        omega_ij2= -torch.cat([self.Hrrp(torch.cat((c0, roll(c0,-1,i+2)),1)) for i in range(self.dim)], 1)
        # omega_ij2= torch.stack([roll(omega_ij[0], -1, i+1) for i in range(self.dim)], 0)
        omega_ii = -torch.sum(omega_ij, 1, keepdim=True)-torch.sum(omega_ij2, 1, keepdim=True)
        for i in range(self.dim):
            if c0.shape[i+2]==2:
                # i.e: a,b,a,b so the correlation should be
                omega_ij[:,i]+= omega_ij2[:,i]
        return omega_ii, omega_ij

    def forward(self, c0):
        omega_ij = self.calc_var(c0, True)
        noise = torch.empty_like(omega_ij).normal_() * torch.sqrt(nn.functional.relu(-omega_ij))
        H = torch.zeros_like(c0)
        for i in range(self.dim):
            H+= noise[:,i:i+1] - roll(noise[:,i:i+1],-1,i+2)
        return H

    def omega_scatter(self, c0, err):
        omega_ii, omega_ij = self.calc_var(c0)
        # omega = torch.cat([omega_ij, omega_ii], 1)
        omega = omega_ij
        err_ii = err**2
        err_ij = torch.cat([err * roll(err,1,i+2) for i in range(self.dim)], 1)
        # scatter_mat = torch.cat([err_ij, err_ii], 1)
        scatter_mat = err_ij
        return omega, scatter_mat

    def visualize(self, fname='noise_corr'):
        import numpy as np
        cgrid = np.linspace(-1.1, 1.1, 56)
        cgrid= np.array(np.meshgrid(cgrid, cgrid))
        shape2d = cgrid.shape[1:]
        cgrid = np.transpose(cgrid.reshape(2,-1)).reshape((-1,2) + ((1,)*self.dim))
        c0 = torch.tensor(cgrid,device='cuda').float()
        res_list=[]
        for g in [self.Hrrp]:
            y = g(c0).reshape(shape2d)
            res_list.append(y.cpu().detach().numpy())
        # pd_noise = self.Hrrp(c0.repeat_interleave(2,-1))
        # pd_noise = pd_noise.cpu().detach().numpy().ravel()
        np.save(fname, res_list[0])

