#!/bin/env python
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import argparse
from NPS_common.utils import load_array, str2slice


def add_identity(axes, *line_args, **line_kwargs):
    identity, = axes.plot([], [], *line_args, **line_kwargs)
    def callback(axes):
        low_x, high_x = axes.get_xlim()
        low_y, high_y = axes.get_ylim()
        low = max(low_x, low_y)
        high = min(high_x, high_y)
        identity.set_data([low, high], [low, high])
    callback(axes)
    axes.callbacks.connect('xlim_changed', callback)
    axes.callbacks.connect('ylim_changed', callback)
    return axes


parser = argparse.ArgumentParser()
parser.add_argument("gt", type=str, help="gt data file")
parser.add_argument("pd", type=str, help="pd data file")
parser.add_argument("-c", default=':', type=str2slice, help="which channel to show, default all ':', e.g. '0:2'")
parser.add_argument("-o", default='', help="save as image")
options = parser.parse_args()
if options.o: matplotlib.use('Agg')

gt = load_array(options.gt)
pd = load_array(options.pd)
print('gt', gt.shape, 'pd', pd.shape)
if options.c is not None:
    gt = gt[...,options.c]
    pd = pd[...,options.c]
fig, axs = plt.subplots(1, 2, figsize=(12, 6))
axs[0].scatter(gt.ravel(), pd.ravel(), s=2)
add_identity(axs[0], color='r', ls='--')
axs[1].hist((pd-gt).ravel(), bins=100)
axs[1].set_yscale('log')
try:
    from scipy import stats
    res = stats.linregress(gt.ravel(), pd.ravel())
    axs[0].text(0.05, 0.95, f'$R^2$= {res.rvalue**2:.4f} rmse= {np.mean((pd-gt)**2):.3g} mae= {np.mean(np.abs(pd-gt)):.3g}', transform=axs[0].transAxes)
    #axs[0].text(0.05, 0.95, f'$R^2$= {res.rvalue**2:.4f} ')#std= {np.std(pd-gt):.3g} mae= {np.mean(np.abs(pd-gt)):.3g}', transform=axs[0].transAxes)
except:
    pass

if options.o:
    plt.savefig(options.o, fig=fig, writer='imagemagick')
else:
    plt.show()
