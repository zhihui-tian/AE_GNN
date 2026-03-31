#!/bin/env python
import torch
import numpy as np
import matplotlib.pyplot as plt

import sys, os
sys.path.append(os.path.join(os.path.dirname(sys.path[0]),'..'))
from NPS import utility, model
from NPS.option import args
import matplotlib
save = True
matplotlib.use('Agg' if save else 'TkAgg')
plt.rcParams.update({'font.size': 6})

dir='experiment/'+args.jobid+'/'
checkpoint = utility.checkpoint(args)
model = model.Model(args, checkpoint).get_model().deter.dxdt
print(model)

import ast
plot_opt = ast.literal_eval(args.visualization_setting)
cgrid = np.linspace(plot_opt.get('cmin',-0.05), plot_opt.get('cmax',1.05), 101)
tgrid = np.linspace(plot_opt.get('Tmin',2), plot_opt.get('Tmax',7), 151)
grid= np.array(np.meshgrid(cgrid, tgrid))
shape2d = grid.shape[1:]
# print(f'debug shape2d {shape2d}')
xnp = np.transpose(grid.reshape(2,-1)).reshape((-1,2) + ((1,)*args.dim))
x = torch.tensor(xnp,device='cuda').float()
fig, axs = plt.subplots(2, 3, figsize=(6, 9))
plt.setp(axs[-1, :], xlabel='c')
plt.setp(axs[:,  0], ylabel='T')
nets = ([model.mobility, 'mobility'], [model.chem_pot, 'chem_pot'], [model.stiffness, 'stiffness'])
for i, (f, name) in enumerate(nets):
    ax=axs[0,i]
    y = f(x).reshape(shape2d).cpu().detach().numpy()
    if name=='chem_pot':
        y-= y[:,50:51]
    np.save(dir+name, y)
    if name=='mobility':
        scaling_fac = tgrid[:,None]
    if True:
        y = y*scaling_fac/y[:, 75:76]
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
