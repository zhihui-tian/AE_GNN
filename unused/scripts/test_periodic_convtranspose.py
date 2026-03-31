import numpy as np
import matplotlib
# from matplotlib import animation
# from matplotlib import tri as mtri
import matplotlib.pyplot as plt
from NPS.model.common import ConvND, ConvTransposeND, my_activations

import torch
import torch.nn as nn


def upconv(nin, nout, stride, periodic=False, kernel_size=3):
        if (stride ==2):
            output_padding = 1
        else:
            output_padding = 0
        return ConvTransposeND(in_channels=nin,out_channels=nout,kernel_size=kernel_size, dim=2, periodic=periodic, stride=stride, bias=False)

xy= [np.arange(128)/128 for _ in range(2)]
xy_grid = np.array(np.meshgrid(*xy)).transpose(np.roll(np.arange(len(xy)+1),-1))
# print(xy)
print(xy_grid.shape)
fig, axs = plt.subplots(3,6)
# print(axs)
axs[0,0].matshow(xy_grid[...,0])
# axs[1].matshow(xy_grid[...,1])
a = np.sin(np.matmul(xy_grid, 2*np.pi*np.array([2,1]))) + 5
ap = torch.tensor(a[None,None,:],requires_grad=False).float()
axs[0,1].matshow(a)
axs[0,2].matshow(np.tile(a, (2,2)))
axs[0,3].axis('off')

with torch.no_grad():
    for i,k in enumerate((1, 3)):
        results = []
        weight = torch.tensor(np.random.randn(1,1,k,k)).float()
        # bias = torch.tensor([0]).float()
        S = 3
        samesize_no_pbc = upconv(1, 1, stride=1, kernel_size=k, periodic=False); samesize_no_pbc.weight = nn.Parameter(weight)
        samesize_pbc = upconv(1, 1, stride=1, kernel_size=k, periodic=True); samesize_pbc.conv.weight = nn.Parameter(weight)
        up_no_pbc = upconv(1, 1, stride=S, kernel_size=k, periodic=False); up_no_pbc.weight = nn.Parameter(weight)
        up_pbc = upconv(1, 1, stride=S, kernel_size=k, periodic=True); up_pbc.conv.weight = nn.Parameter(weight)
        print(samesize_no_pbc, samesize_pbc, samesize_no_pbc.weight, samesize_pbc.conv.weight)
        ax= axs[1+i,0]; z=samesize_no_pbc(ap)[0,0]; ax.matshow(torch.tile(z,(2,2))); ax.set_title(f'k{k} samesize_no_pbc {list(z.shape)}'); results.append(z)
        ax= axs[1+i,1]; z=samesize_pbc(ap)[0,0]; ax.matshow(torch.tile(z,(2,2))); ax.set_title(f'k{k} samesize_pbc {list(z.shape)}'); results.append(z)
        axs[1+i,2].matshow(torch.tile(results[-1]-results[-2],(2,2)))
        ax= axs[1+i,3]; z=up_no_pbc(ap)[0,0]; ax.matshow(torch.tile(z,(2,2))); ax.set_title(f'k{k} up_no_pbc {list(z.shape)}'); results.append(z)
        ax= axs[1+i,4]; z=up_pbc(ap)[0,0]; ax.matshow(torch.tile(z,(2,2))); ax.set_title(f'k{k} up_pbc {list(z.shape)}'); results.append(z)
        axs[1+i,5].matshow(torch.tile(results[-1]-results[-2],(2,2)))
        print(f'k={k} s=1 diff {np.linalg.norm(results[0]-results[1])}\n   results {results[0][:3,:3]}\n   results {results[1][:3,:3]} {a[:3,:3]}')
        print(f'k={k} s={S} diff {np.linalg.norm(results[2]-results[3])}\n   results {results[2][:3,:3]}\n   results {results[3][:3,:3]}')
    print(results[-2],'\n', results[-1])
    plt.show()
