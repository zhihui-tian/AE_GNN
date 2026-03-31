#!/bin/env python

#import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import argparse
from scipy.fftpack import fftn, ifftn

parser = argparse.ArgumentParser()
parser.add_argument("array", help="input .npy array")
parser.add_argument("smooth_dim", help="which dims to smooth, e.g. 0,1,2")
parser.add_argument("cutoff", type=float,  help="fft grid radius cutoff")
#parser.add_argument("--n_in", type=int, default=1, help="number of inputs")
options = parser.parse_args()

smooth_dim = tuple(map(int, options.smooth_dim.split(',')))
a = np.load(options.array)
a_fft = fftn(a,axes=smooth_dim)
ijk = np.array(np.meshgrid(*(np.arange(a.shape[i]) for i in smooth_dim))).T
ijk_img = np.array(np.meshgrid(*(np.arange(a.shape[i],0,-1) for i in smooth_dim))).T
mask = np.where(np.linalg.norm(np.minimum(ijk, ijk_img),axis=len(smooth_dim)) < options.cutoff, 1, 0)
mask = mask.reshape([a.shape[i] if i in smooth_dim else 1 for i in range(a.ndim)])
a_smooth = ifftn(a_fft*mask,axes=smooth_dim).real.astype(a.dtype)
np.save("smooth.npy", a_smooth)
