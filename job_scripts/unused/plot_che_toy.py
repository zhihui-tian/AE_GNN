#!/bin/env python
import torch
import numpy as np
import matplotlib.pyplot as plt

import matplotlib
save = False
matplotlib.use('Agg' if save else 'TkAgg')
plt.rcParams.update({'font.size': 6})
dir=''

import ast
#plot_opt = ast.literal_eval(args.visualization_setting)
plot_opt = {"cmin":-1,"cmax":1,"Tmin":0,"Tmax":0.26}
cgrid = np.linspace(plot_opt.get('cmin',-0.05), plot_opt.get('cmax',1.05), 101)
tgrid = np.linspace(plot_opt.get('Tmin',1.55), plot_opt.get('Tmax',10.05), 150)
grid= np.array(np.meshgrid(cgrid, tgrid))
shape2d = grid.shape[1:]
# print(f'debug shape2d {shape2d}')
# xnp = np.transpose(grid.reshape(2,-1)).reshape((-1,2) + ((1,)*args.dim))
# x = torch.tensor(xnp,device='cuda').float()
x=(grid[0],grid[1])
fig, axs = plt.subplots(2, 3, figsize=(6, 9))
plt.setp(axs[-1, :], xlabel='c')
plt.setp(axs[:,  0], ylabel='T')
# def mobility(c, T): return (c+1)*(1-c)*(10*T+100*T**2)
# def stiffness(c, T): return 1 + 2*T*(1-c**2/4)
# def chem_pot(c, T): return c**3-c + T*np.log((c+1)/(1-c))
import sys, os
sys.path.append(sys.path[0])
from generate_SPDE import mobility, stiffness, chem_pot
nets = ([chem_pot, 'chem_pot'], [stiffness, 'stiffness'], [mobility, 'mobility'])
for i, (f, name) in enumerate(nets):
    ax=axs[0,i]
    y = f(*x).reshape(shape2d)#.cpu().detach().numpy()
    if name=='chem_pot':
        y-= y[:,50:51]
    np.save(dir+name, y)
    im=ax.imshow(y, extent=[cgrid[0], cgrid[-1], tgrid[0], tgrid[-1]], aspect='auto')
    ax.title.set_text(name)
    fig.colorbar(im, ax=ax)

    ax=axs[1,i]
    for it in range(6,150,28):
        t = tgrid[it]
        ax.plot(cgrid, y[it], label=f'T={t:.2f}')
    ax.legend()
if save:
    plt.savefig(dir+'plot_che.pdf')
else:
    plt.show()
