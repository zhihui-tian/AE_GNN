#!/bin/env python
import numpy as np
from scipy.stats import gaussian_kde
from scipy.stats import norm
# from pylab import plot,show,hist
import matplotlib.pyplot as plt
import argparse

from NPS_common.utils import load_array

parser = argparse.ArgumentParser()
parser.add_argument("--ichannel", "-i", type=str, default='-1', help="which channels, -1 for all ")
parser.add_argument("data", help="data file(s) (.npy or .npz)", nargs='+')
parser.add_argument("--bins", type=int, default=100, help="no. bins")
options = parser.parse_args()
options.ichannel = slice(0, None, 1) if options.ichannel=='-1' else 'TBD' #np.array(list(map(int, options.ichannel.split(','))))

dat_all = [load_array(x)[..., options.ichannel] for x in options.data]
nchannel = dat_all[0].shape[-1]
nplot= len(options.data)
fig, axs = plt.subplots(nrows=nplot, ncols=nchannel, figsize=(nchannel*8, nplot*5))
if nplot == 1:
    axs = [axs]

for i in range(nplot):
    for j in range(nchannel):
        ax = axs[i][j]
        dat = dat_all[i][...,j].ravel()
        ax.hist(dat, bins=options.bins); ax.set_yscale('log')
        ax.text(0.05, 0.95, f'mean= {np.mean(dat):.3g} std= {np.std(dat):.3g}', transform=ax.transAxes)

plt.show()


# # obtaining the pdf (my_pdf is a function!)
# my_pdf = gaussian_kde(samp)

# # plotting the result
# x = np.linspace(samp.min(), samp.max(),100)
# plot(x,my_pdf(x),'r') # distribution function
# hist(samp, bins=50, density=1,alpha=.3) # histogram
# show()
