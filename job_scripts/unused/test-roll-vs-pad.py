
import torch
import torch.nn as nn
roll = torch.roll
from torch.nn import functional as F

NL=50
c = torch.rand(100, 4, NL,NL,NL, device='cuda')

rd2=((1,1,2,1),(0,1,2,1),(0,1,1,1),(1,1,2,-1),(0,1,2,-1),(0,1,1,-1))
rd3=((0,1,1,1,2,1),(0,1,1,1,2,-1),(0,1,1,-1,2,1),(0,1,1,-1,2,-1))
def scatter_matix(err):
    err_ii = err**2
    err_ij = []
    if 1<=3:
        err_ij += [err * roll(err,1,i+2) for i in range(3)]
    if 2<=3:
        err_ij += [err * roll(roll(err,d[1],d[0]+2),d[3],d[2]+2) for d in rd2]
    if 3<=3:
        err_ij += [err * roll(roll(roll(err,d[1],d[0]+2),d[3],d[2]+2),d[5],d[4]+2) for d in rd3]
    # if 3<=3:
    #     err_ij += [err * roll(err,1,i+2) for i in range(3)]
    # err_ij = torch.cat([err * roll(err,1,i+2) for i in range(self.dim)], 1)
    # scatter_mat = torch.cat([err_ij, err_ii], 1)
    scatter_mat = torch.cat(err_ij, 1)
    return scatter_mat

import numpy as np
dr1= (1-np.eye(3)).astype(int).tolist()
# dr2= np.eye(3).astype(int).tolist() + [[1,0,2],[0,1,2],[0,2,1]]
dr2= (1-np.vstack([1-np.eye(3),[[0,1,-1],[1,0,-1],[1,-1,0]]])).astype(int).tolist()
dr3= (1-np.array([[1,1,1],[1,1,-1],[1,-1,1],[1,-1,-1]])).astype(int).tolist()
def scatter_matix_pad(err):
    err_ii = err**2
    err_ij = []
    err_pad = F.pad(err, (1,)*6, mode='circular')
    if 1<=3:
        err_ij += [err * err_pad[:,:,dr[0]:dr[0]+NL,dr[1]:dr[1]+NL,dr[2]:dr[2]+NL] for dr in dr1]
    if 2<=3:
        err_ij += [err * err_pad[:,:,dr[0]:dr[0]+NL,dr[1]:dr[1]+NL,dr[2]:dr[2]+NL] for dr in dr2]
    if 3<=3:
        err_ij += [err * err_pad[:,:,dr[0]:dr[0]+NL,dr[1]:dr[1]+NL,dr[2]:dr[2]+NL] for dr in dr3]
    # if 3<=3:
    #     err_ij += [err * roll(err,1,i+2) for i in range(3)]
    # err_ij = torch.cat([err * roll(err,1,i+2) for i in range(self.dim)], 1)
    # scatter_mat = torch.cat([err_ij, err_ii], 1)
    scatter_mat = torch.cat(err_ij, 1)
    return scatter_mat

import time
t0 = time.perf_counter()
for i in range(1000):
    x1=scatter_matix(c)
t1 = time.perf_counter()
print('timing with roll', t1-t0)

t0 = time.perf_counter()
for i in range(1000):
    x2=scatter_matix_pad(c)
t1 = time.perf_counter()
print('timing with pad', t1-t0)

print('comparing', torch.norm(x1-x2))

