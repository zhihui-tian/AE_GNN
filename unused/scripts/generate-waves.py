#!/bin/env python
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("nmode", help="number of modes (1 or 2 or 1,2)")
parser.add_argument("--data_size", help="\"Ns Nt Nx Ny [Nz for 3D]\"")
parser.add_argument('--periodic', action='store_true', default=False)
parser.add_argument('--save_t', action='store_true', default=False)
options = parser.parse_args()

def f(modes, Nt, Nspace, save_t=False, sharpness=2.3):
    xy= [np.arange(Ni)/Ni for Ni in Nspace]
    xy_grid = np.array(np.meshgrid(*xy)).transpose(np.roll(np.arange(len(xy)+1),-1))
    val=np.array([np.sum([np.sin(phase + np.dot(xy_grid, k) + t*omega)*magnitude*np.exp(-t*decay)#/(1+t*decay)
                                   for phase, k, omega, decay, magnitude in modes], axis=0) 
                           for t in np.arange(Nt)/Nt])
    val = val[...,None]
    if save_t:
        # print( ((np.arange(Nt)).reshape((-1,)+(1,)*(len(Nspace)+1))))
        val = np.concatenate([val, np.ones_like(val)*((np.arange(Nt)/Nt).reshape((-1,)+(1,)*(len(Nspace)+1)))], -1)
    return val
    # return xy, (val-np.min(val))/(np.max(val)-np.min(val))  # (np.tanh(val*sharpness)+1)/2

def kv_commensurate(dim):
    while True:
        kv = 2*np.pi*np.random.choice([0,4,3,2,1,-4,-3,-2,-1], dim)
        if np.linalg.norm(kv) > 0:
            break
    return kv

Ninput = list(map(int, options.data_size.split()))
dim = len(Ninput) - 2
Ns=Ninput[0]
Nt=Ninput[1]
Nspace=Ninput[2:]
nchannel = 2 if options.save_t else 1
Ninput.append(nchannel)
alldat=np.zeros(Ninput, dtype=np.float32)
for i in range(Ns):
    nmode=np.random.choice(list(map(int, options.nmode.split(','))))
    assert nmode in (1,2), 'Error nmode %d should be 1 or 2'%(nmode)
    modes=[]
    for imode in range(nmode):
        if options.periodic:
            if imode == 0:
                kv = kv_commensurate(dim)
            else:
                while True:
                    kv2 = kv_commensurate(dim)
                    if np.abs(kv.dot(kv2)) < 0.4*np.linalg.norm(kv)*np.linalg.norm(kv2):
                        kv = kv2
                        break
        else:
            k=2*np.pi/np.random.uniform(0.3, 0.6)
            if imode == 0:
                kvec=np.random.normal(0, 1, dim); kvec/=np.linalg.norm(kvec)
            else:
                while True:
                    kv2=np.random.normal(0, 1, dim); kv2/=np.linalg.norm(kv2)
                    if np.abs(kvec.dot(kv2)) < 0.5:
                        kvec = kv2
                        break
            kv = k*kvec
        phase=np.random.uniform(0, 2*np.pi)
        omega=2*np.pi/np.random.uniform(0.7, 2.1)
        decay=2*np.pi/np.random.uniform(1.5, 6)
        magnitude=np.random.uniform(1, 1)
        modes.append([phase, kv, omega, decay, magnitude])
    #print('debug i', i, modes)
    alldat[i]=f(modes, Nt, Nspace, options.save_t, 1.7)
np.save('output.npy', alldat)
