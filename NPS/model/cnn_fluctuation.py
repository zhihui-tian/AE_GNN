__author__ = 'Fei Zhou'

import torch
import torch.nn as nn
from torch.nn import functional as F

import model
from model.common import CNN
roll = torch.roll

def make_model(args):
    return CNNFluctuation(args)

class CNNFluctuation(model.NPSModel):
    def __init__(self, configs):
        super(CNNFluctuation, self).__init__(configs)
        self.args = configs
        import numpy as np
        self.numNN = configs.stoch_numNN
        self.skiponsite = configs.stoch_skiponsite
        assert self.numNN<=configs.dim, 'CNN fluctuation implemented for stoch_numNN<=dim (up to 3rd NN)'
        # dr defined relative to the (1,1) padded array
        if configs.dim == 2:
            dr_table = [(1-np.eye(2)).astype(int).tolist(),
                    (1-np.array([[1,1],[1,-1]])).astype(int).tolist()]
        elif configs.dim == 3:
            dr_table = [(1-np.eye(3)).astype(int).tolist(),
                    (1-np.vstack([1-np.eye(3),[[0,1,-1],[1,0,-1],[1,-1,0]]])).astype(int).tolist(),
                    (1-np.array([[1,1,1],[1,1,-1],[1,-1,1],[1,-1,-1]])).astype(int).tolist()]
        else:
            raise 'CNN fluctuation implemented for 2d, 3d only'
        dr = []
        for n in range(self.numNN):
            dr+= dr_table[n]
        self.drs = dr
        self.drs_inv = 2-np.array(self.drs)
        self.nNB ={n:len(dr_table[n-1]) for n in range(1, configs.dim+1)}
        self.nfluc = np.sum([self.nNB[n+1] for n in range(self.numNN)])
        n_out = configs.n_colors*((0 if self.skiponsite else 1)+self.nfluc)
        self.omega_shape     = (-1, configs.n_colors, self.nfluc) + tuple(configs.frame_shape)
        self.omega_shape_pad = (-1, configs.n_colors, self.nfluc) + tuple(np.array(configs.frame_shape)+2)
        print(f'debug omea shape {self.omega_shape} n out {n_out} fluc{self.nfluc} nNB {self.nNB} drs {self.drs} numNN {self.numNN}')
        # Hrrp output channels: onsite for channel 1, ..., nfluc for channel 1, ...
        self.Hrrp = CNN(configs.n_colors, configs.stoch_hidden, n_out, configs.stoch_kernel, dim=self.dim, activation=configs.stoch_act, periodic=configs.periodic)
        # self.relu = nn.ReLU()

    def calc_var(self, c0):
        omega = self.Hrrp(c0)
        # if ij_only:
        #     return omega_ij
        # omega_ij2= -torch.cat([self.Hrrp(torch.cat((c0, roll(c0,-1,i+2)),1)) for i in range(self.dim)], 1)
        # omega_ij2= torch.stack([roll(omega_ij[0], -1, i+1) for i in range(self.dim)], 0)
        # omega_ii = -torch.sum(omega_ij, 1, keepdim=True)-torch.sum(omega_ij2, 1, keepdim=True)
        # for i in range(self.dim):
        #     if c0.shape[i+2]==2:
        #         # i.e: a,b,a,b so the correlation should be
        #         omega_ij[:,i]+= omega_ij2[:,i]
        return omega

    def forward(self, c0):
        omega = self.calc_var(c0)
        if self.skiponsite:
            omega_ij = omega
        else:
            omega_ii = omega[:,:self.args.n_colors]
            omega_ij = omega[:,self.args.n_colors:]
        noise = torch.empty_like(omega_ij).normal_() * torch.sqrt(F.relu(-omega_ij))
        noise_pad = F.pad(noise, (1,)*(2*self.dim), mode='circular')
        noise = noise.view(*self.omega_shape)
        omega_ij = omega_ij.view(*self.omega_shape)
        noise_pad = noise_pad.view(*self.omega_shape_pad)
        # print(f'debug c0{c0.shape} noise{noise.shape} omegaij {omega_ij.shape} noise_pad {noise_pad.shape}')
        H = torch.zeros_like(c0)
        l = self.args.frame_shape
        for i, dr in enumerate(self.drs_inv):
            if self.dim ==2:
                H += noise[:,:,i] - noise_pad[:,:,i,dr[0]:dr[0]+l[0],dr[1]:dr[1]+l[1]]
            elif self.dim ==3:
                H += noise[:,:,i] - noise_pad[:,:,i,dr[0]:dr[0]+l[0],dr[1]:dr[1]+l[1],dr[2]:dr[2]+l[2]]
        if self.skiponsite:
            return H

        # onsite term is omega_ii minus diffusion term omega_ij
        for i in range(self.args.n_colors):
            if True or self.args.stoch_onsite[i]:
                # print(f'debug ii {omega_ii.shape} ij {omega_ij.shape}', omega_ii[:,i].shape, torch.sum(omega_ij[:,i],1).shape)
                omega_ii[:,i] -= torch.sum(omega_ij[:,i],1)
                o_pad = F.pad(omega_ij[:,i], (1,)*(2*self.dim), mode='circular')
                for j, dr in enumerate(self.drs_inv):
                    if self.dim ==2:
                        omega_ii[:,i] -= o_pad[:,j,dr[0]:dr[0]+l[0],dr[1]:dr[1]+l[1]]
                    elif self.dim ==3:
                        omega_ii[:,i] -= o_pad[:,j,dr[0]:dr[0]+l[0],dr[1]:dr[1]+l[1],dr[2]:dr[2]+l[2]]
                H[:,i] += torch.empty_like(omega_ii[:,i]).normal_() * torch.sqrt(F.relu(omega_ii[:,i]))
        return H

    def scatter_matix(self, err):
        err_ij = [] if self.skiponsite else [err**2]
        err_pad = F.pad(err, (1,)*(2*self.dim), mode='circular')
        l = self.args.frame_shape
        for dr in self.drs:
            if self.dim ==2:
                err_ij.append(err * err_pad[:,:,dr[0]:dr[0]+l[0],dr[1]:dr[1]+l[1]])
            elif self.dim ==3:
                err_ij.append(err * err_pad[:,:,dr[0]:dr[0]+l[0],dr[1]:dr[1]+l[1],dr[2]:dr[2]+l[2]])
        scatter_mat = torch.cat(err_ij, 1)
        return scatter_mat


    def omega_scatter(self, c0, err):
        omega_ij = self.calc_var(c0)
        # omega = torch.cat([omega_ij, omega_ii], 1)
        omega = omega_ij
        return omega, self.scatter_matix(err)

    def visualize(self, fname='noise_corr'):
        pass
        # import numpy as np
        # cgrid0 = np.linspace(-1.1, 1.1, 56)
        # cgrid= np.array(np.meshgrid(cgrid0, cgrid0))
        # shape2d = cgrid.shape[1:]
        # cgrid = np.repeat(np.transpose(cgrid.reshape(2,-1)),25,1).reshape((-1,2) + ((5,)*self.dim))
        # c0 = torch.tensor(cgrid,device='cuda').float()
        # # res_list=[]
        # # for g in [self.Hrrp]:
        #     # y = g(c0).reshape(shape2d)
        #     # res_list.append(y.cpu().detach().numpy())
        # # pd_noise = self.Hrrp(c0.repeat_interleave(2,-1))
        # # pd_noise = pd_noise.cpu().detach().numpy().ravel()
        # # np.save(fname, res_list[0])
        # np.save(fname, self.Hrrp(c0).reshape(shape2d).cpu().detach().numpy())

