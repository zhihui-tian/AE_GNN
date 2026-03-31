import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch

import ase.io
# from ase.visualize import view
from ase.neighborlist import neighbor_list
from NPS.model.common import vector_pbc
from NPS_common.io_utils import co, temp_txt_file

# import sys, os; sys.path.append(os.path.join(sys.path[0], '../../..'))

from glob    import glob
from pathlib import Path


import argparse

parser = argparse.ArgumentParser()
parser.add_argument("data", help="data file(s) (.npy or .npz)", nargs='+')
parser.add_argument("-o", default='', help="save image")
options = parser.parse_args()
if options.o: matplotlib.use('Agg')

s0 = ase.io.read(options.data[0])
types = np.unique(s0.get_atomic_numbers())
if len(s0.get_positions()[s0.get_atomic_numbers()==types[0]]) > len(s0.get_positions()[s0.get_atomic_numbers()==types[1]]):
    types = types[::-1]
s1_all = open(options.data[1], "r").readlines()
nTrj = int(co(f"grep TIMESTEP {options.data[1]} | wc -l"))
nL = len(s1_all) // nTrj
err_rate = [[], []]
err_dist = [[], []]
err_dmax = [[], []]
for i in range(0, len(s1_all), nL):
    # print(s1_all[i], s1_all[i].__class__)
    # s1_f = io.BytesIO(bytes('\n'.join(s1_all[i:i+nL]), 'utf-8'))
    # s1_f = io.StringIO(str('\n'.join(s1_all[i:i+nL])))
    with temp_txt_file(''.join(s1_all[i:i+nL])) as s1_f:
        s1 = ase.io.read(s1_f)
        # print(i, s1_f, s1)
# print(s0, s1)
# print(s0.get_positions())
# print(s0.get_atomic_numbers())
    cell = s0.get_cell().cellpar()[:3]
    # print(f'types {types} cell {cell}')
    for idx, t in enumerate(types):
        p0 = s0.get_positions()[s0.get_atomic_numbers()==t]
        p1 = s1.get_positions()[s1.get_atomic_numbers()==t]
        # print(p0.shape, p1.shape)
        assert np.all(np.array(p0.shape) == np.array(p1.shape))
        disp = p1[:, None] - p0[None, :]
        disp = vector_pbc(torch.from_numpy(disp).reshape(-1, 3), cell).reshape(len(p0), -1, 3)
        dist = np.linalg.norm(disp, axis=-1)
        imin = np.argmin(dist, axis=1)
        min_dist = np.min(dist, axis=1)
        mismatch_rate = 1-len(np.unique(imin)) / len(p0)
        err_rate[idx].append(mismatch_rate)
        err_dist[idx].append(np.mean(min_dist))
        err_dmax[idx].append(np.max(min_dist))
    # print(np.min(dist, axis=1))
    # min_disp = disp_pbc[imin]
    # min_dist = np.linalg.norm(min_disp, axis=1)
        # print(f'min dist mean {np.mean(min_dist)} max {np.max(min_dist)} mismatch {mismatch_rate}')
    # avg_disp = np.mean(min_disp, axis=0)
    # print(avg_disp)
    # dist_before = np.min(np.linalg.norm(disp_pbc, axis=-1), axis=1)
    # disp_pbc -= avg_disp
    # print(np.min(np.linalg.norm(disp_pbc, axis=-1), axis=1))
    # print(np.min(np.linalg.norm(disp_pbc, axis=-1), axis=1) - dist_before)
    # print(np.min(np.linalg.norm(disp, axis=-1), axis=1))
    # axs[idx, 0].scatter(np.min(np.linalg.norm(disp, axis=-1), axis=1), np.min(np.linalg.norm(disp_pbc, axis=-1), axis=1))
    # axs[idx, 1].hist(np.min(np.linalg.norm(disp, axis=-1), axis=1), alpha=0.5, bins=50, label='no pbc')
    # axs[idx, 1].hist(np.min(np.linalg.norm(disp_pbc, axis=-1), axis=1), alpha=0.5, bins=50, label='pbc')
fig, axs = plt.subplots(1, 2)
X = np.arange(nTrj)
c= ['r', 'b']
elements = ['Si', 'O']
for idx in range(2):
    axs[0].plot(X, err_rate[idx], c[idx])#, label=f'{idx} mismatch rate')
    axs[1].plot(X, err_dist[idx], c[idx]+'-', label=f'{elements[idx]} avg. dist')
    axs[1].plot(X, err_dmax[idx], c[idx]+':', label=f'{elements[idx]} max. dist')
    fig.legend()
axs[0].title.set_text('mismatch rate')
axs[1].title.set_text('distance')
fig.suptitle(os.path.split(options.data[0])[1])

if options.o:
    plt.savefig(options.o)
else:
    plt.show()


# ******************
# running example
# for f in A2B_cF24_227_c_a A2B_hP12_194_cg_f A2B_hP9_152_c_a A2B_hP9_180_j_c A2B_hP9_181_j_c A2B_mC144_9_24a_12a A2B_mC48_15_ae3f_2f A2B_oC24_20_abc_c A2B_tP12_92_b_a A2B_tP36_96_3b_ab A2B_tP6_136_f_a ; do python ~/lassen-space/NPS/scripts/compare_structure.py ~/amsdnn/phase_classification/data/unlabeled/SiO2/references/$f.lammpstrj  ~/amsdnn/phase_classification/data/unlabeled/SiO2/nvt/$f/out.lammpstrj  -o tmp_$f.png; done
# convert tmp_A2B_*.png -gravity center -append nvt.png
# convert npt.png nvt.png +append out.png

