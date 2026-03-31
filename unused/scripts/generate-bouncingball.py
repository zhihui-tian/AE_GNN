#!/bin/env python
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("nmode", help="number of modes (1 or 2 or 1,2)")
parser.add_argument("--data_size", help="\"Ns Nt Nx Ny [Nz for 3D]\"")
options = parser.parse_args()

def bouncing_ball(modes, Nt, Nspace, sharpness=2.3):
    def _bounce(p, k, p0=0, p1=1):
#        print('debu bounce', p, p0, p1)
#        print(np.maximum(0,p0-p) , np.maximum(0,p-p1)) 
        return p+ 2*np.maximum(0,p0-p) - 2*np.maximum(0,p-p1), k*(1-2*(np.heaviside(p0-p,1)+np.heaviside(p-p1,1)))
    dim=len(Nspace)
    xy= [np.arange(Ni)/Ni for Ni in Nspace]
    xy_grid = np.array(np.meshgrid(*xy)).transpose(np.roll(np.arange(len(xy)+1),-1))
    val= np.zeros([Nt] + Nspace)
    for k0, x0, r0, magnitude in modes:
        x=x0.reshape([1]*dim + [-1]); r=r0; k=k0/Nt;
        for t in range(Nt):
            x, k= _bounce(x+k, k, r, 1-r)
            #print('debug t x', t, k0,x0, r0, x)
            val[t]+= np.exp(-np.linalg.norm(xy_grid-x,axis=-1)**2/(2*r**2))* magnitude
    return xy, (val-np.min(val))/(np.max(val)-np.min(val))  # (np.tanh(val*sharpness)+1)/2

Ninput = list(map(int, options.data_size.split()))
dim = len(Ninput) - 2
Ns=Ninput[0]
Nt=Ninput[1]
Nspace=Ninput[2:]
alldat=np.zeros(Ninput, dtype=np.float32)
for i in range(Ns):
    nmode=np.random.choice(list(map(int, options.nmode.split(','))))
    assert nmode in (1,2), 'Error nmode %d should be 1 or 2'%(nmode)
    modes=[]
    for imode in range(nmode):
        k=2*np.pi/np.random.uniform(1, 2)
        if imode == 0:
            kvec=np.random.normal(0, 1, dim); kvec/=np.linalg.norm(kvec)
        else:
            while True:
                kv2=np.random.normal(0, 1, dim); kv2/=np.linalg.norm(kv2)
                if np.abs(kvec.dot(kv2)) < 0.5:
                    kvec = kv2
                    break
        x0= np.random.uniform(0.2, 0.8, 3)
        r0= np.random.uniform(0.11, 0.16)
        magnitude=np.random.uniform(1, 1)
        modes.append([kvec*k, x0, r0, magnitude])
    #print('debug i', i, modes)
    alldat[i]=bouncing_ball(modes, Nt, Nspace, 1.7)[-1]
np.save('output.npy', alldat)
