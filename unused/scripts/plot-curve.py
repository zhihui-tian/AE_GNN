#!/bin/env python
import numpy as np
import glob, os
import matplotlib.pyplot as plt
import argparse

markers=['-','--','-.',':','.',',','o','v','^','<','>','1','2','3','4','s','p','*','h','H','+','x','D','d','|','_']
colors=['b','g','r','c','m','y','k']#,'wb]
styles=[i+j for j in markers for i in colors]

parser = argparse.ArgumentParser()
# parser.add_argument("-o", default='', help="save as gif")
parser.add_argument("data", help="data file(s)", nargs='+')
parser.add_argument("--toy", action='store_true', help="plot toy model mode", default=False)
options = parser.parse_args()

def plot_for_toy():
    fig, ax = plt.subplots(1,2, figsize=(15,5))
    for irun, run in enumerate(options.data):
        fs = glob.glob(f'pf_noise_{run}*.txt')
        dat = np.stack(list(map(np.loadtxt, fs)))
        print(f'No. of epochs = {run} dat size {dat.shape}')

        cgrid=dat[0,0]
        if dat.shape[1]>=5:
            pf_GT = dat[0,1]
            noise_GT = dat[0,3]
            pf_PD = dat[:,2]
            noise_PD = dat[:,4]
        else:
            try:
                gt_dat = np.loadtxt('GT.txt')
                pf_GT = gt_dat[1]
                noise_GT = gt_dat[2]
            except:
                pf_GT = None; noise_GT = None
            pf_PD = dat[:,1]
            noise_PD = dat[:,2]
        pf_PD_mean = np.mean(pf_PD, axis=0)
        noise_PD_mean = np.mean(noise_PD, axis=0)
        # print('debug', pf_GT.shape, noise_GT.shape, gt_dat.shape)

        if irun==0 and (pf_GT is not None) and (noise_GT is not None):
            ax[0].plot(cgrid, pf_GT, 'b.', label='PF GT')
            ax[1].plot(cgrid, noise_GT, 'b.', label='noise GT')
        if len(fs) > 1:
            pf_PD_std = np.std(pf_PD, axis=0)
            noise_PD_std = np.std(noise_PD, axis=0)
            ax[0].fill_between(cgrid, pf_PD_mean+pf_PD_std, pf_PD_mean-pf_PD_std, facecolor='red', alpha=0.5)
            ax[1].fill_between(cgrid, noise_PD_mean+noise_PD_std, noise_PD_mean-noise_PD_std, facecolor='red', alpha=0.5)
        # ax[0].plot(cgrid, pf_PD_mean, 'r-', label=run)
        # ax[1].plot(cgrid, noise_PD_mean, 'r.', label=run)
        ax[0].plot(cgrid, pf_PD_mean, label=run)
        ax[1].plot(cgrid, noise_PD_mean, label=run)

    ax[0].set_title('phase field equation')
    ax[1].set_title('thermal fluctuation')
    ax[0].legend()
    ax[1].legend()


def plot_curve(options):
    data = np.stack(list(map(np.loadtxt, options.data)))
    nrun = data.shape[0]
    nline = data.shape[1]-1
    fig, ax = plt.subplots(1,nline, figsize=(15,5))
    cgrid=data[0,0]
    for il in range(nline):
        for irun in range(nrun):
            ax[il].plot(cgrid, data[irun,il+1], styles[irun], label=f'{irun}')
        ax[il].legend()


if options.toy:
    plot_for_toy()
else:
    plot_curve(options)

plt.show()
