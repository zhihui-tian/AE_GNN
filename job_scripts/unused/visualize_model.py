#!/bin/env python
import torch
import numpy as np
import matplotlib.pyplot as plt

import sys, os
sys.path.append(os.path.join(os.path.dirname(sys.path[0]),''))
import utility
import model
from option import args
import ast
import matplotlib
save = True
matplotlib.use('Agg' if save else 'TkAgg')
plt.rcParams.update({'font.size': 6})

plot_opt = ast.literal_eval(args.visualization_setting)
dir='experiment/'+args.jobid+'/'
checkpoint = utility.checkpoint(args)
model = model.Model(args, checkpoint).get_model().deter.dxdt
print(model)

xlabel = plot_opt['xlabel']
nets = [[getattr(model,f), f] for f in plot_opt['func']] # "func":["chem_pot", "mobility", "stiffness"]
x_range = [np.linspace(*rng) for rng in plot_opt['xgrid']] # "xgrid":[[-0.05,1.05,101],[1.55,7.05,151]]
xgrid= np.array(np.meshgrid(*x_range, indexing='ij'))
shapeNd = xgrid.shape[1:]
xdim = xgrid.shape[0]
xnp = np.transpose(xgrid.reshape(xdim,-1)).reshape((-1,xdim) + ((1,)*args.dim))
x = torch.tensor(xnp,device='cuda').float()
for i, (f, name) in enumerate(nets):
    y = f(x).reshape(shapeNd).cpu().detach().numpy()
    np.save(dir+name, y)
#     im=ax.imshow(y, extent=[cgrid[0], cgrid[-1], tgrid[0], tgrid[-1]], aspect='auto')
#     ax.title.set_text(name)
#     fig.colorbar(im, ax=ax)

#     ax=axs[1,i]
#     for it in range(6,150,28):
#         t = tgrid[it]
#         ax.plot(cgrid, y[it], label=f'T={t:.2f}')
#     ax.legend()
# if save:
#     plt.savefig(dir+'plot_che.pdf')
# else:
#     plt.show()
