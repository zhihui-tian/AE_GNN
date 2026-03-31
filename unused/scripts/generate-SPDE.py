#!/bin/env python
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data_size", help="\"Ns Nt Nx Ny [Nz for 3D]\"")
parser.add_argument("--method", help="laplacian or laplacian_sq or che or general_che")
parser.add_argument("--T", default=1.0, type=float, help="temperature for general_che")
parser.add_argument('--saveT', action='store_true', help='add T as second channel')
parser.add_argument("--D", default=0.1, type=float, help="diffusivity")
parser.add_argument("--tskip", default=1, type=int, help="output every tskip steps")
parser.add_argument("--tskip_noise", default=1, type=int, help="adding noise every (default 1: every step)")
parser.add_argument("--tbegin", default=0, type=int, help="output from this step")
parser.add_argument("-o", default='output.npy', help="output.npy")
parser.add_argument("--c0", default='', help="if specified starting from this .npy file")
parser.add_argument("--noise", default=0, type=float, help="noise term")
parser.add_argument("--dc0", default=0.001, type=float, help="initial fluctuation")
parser.add_argument("--cmin", default=-0.5, type=float, help="cmin")
parser.add_argument("--cmax", default= 0.5, type=float, help="cmax")
parser.add_argument('--delta', action='store_true', help='output delta (useful when fitting delta with 2-frame seqences)')

def laplacian(a):
    return np.roll(a,1,axis=0) + np.roll(a,-1,axis=0) + np.roll(a,1,axis=1) + np.roll(a,-1,axis=1) -4*a

def divergence_gradient(a, b):
    """
    div(a * grad(b))
    """
    return a*(np.roll(b,-1,0)-b) - np.roll(a,1,0)*(b-np.roll(b,1,0)) + a*(np.roll(b,-1,1)-b) - np.roll(a,1,1)*(b-np.roll(b,1,1))

def divergence_gradient_symm(a, b):
    """
    div(a * grad(b))
    """
    return ((np.roll(a,-1,0)+a)/2)*(np.roll(b,-1,0)-b) - ((a+np.roll(a,1,0))/2)*(b-np.roll(b,1,0)) + ((np.roll(a,-1,1)+a)/2)*(np.roll(b,-1,1)-b) - ((a+np.roll(a,1,1))/2)*(b-np.roll(b,1,1))

def PDE(modes, Nt, Nspace, method='laplacian', D=0.1, noise=0, cmin=-0.5, cmax=0.5, tskip_noise=1, c0=None, delta=False, dc0=0.001, T=1, saveT=False):
    dim=len(Nspace)
    xy= [np.arange(Ni)/Ni for Ni in Nspace]
    xy_grid = np.array(np.meshgrid(*xy)).transpose(np.roll(np.arange(len(xy)+1),-1))
    val= np.zeros([Nt] + list(Nspace))
    for t in range(Nt):
        if t == 0:
            if c0 is None:
                for x0, r0, magnitude in modes:
                    x=x0.reshape([1]*dim + [-1])
                    val[t]+= np.exp(-np.linalg.norm(xy_grid-x,axis=-1)**2/(2*r0**2))* magnitude
                if method in ['che', 'general_che']:
                    val[t] = np.random.uniform(-dc0, dc0, Nspace) + np.random.uniform(cmin,cmax)
                    if T>0 and (method=='general_che'):
                        val[t] = np.clip(val[t], -0.996, 0.996)
            else:
                val[0] = c0
        else:
            if method=='laplacian':
                dx = laplacian(val[t-1])
            elif method=='laplacian_sq':
                dx = laplacian(laplacian(val[t-1]))
            elif method=='che':
                dx = laplacian(np.power(val[t-1],3)-val[t-1]-laplacian(val[t-1]))
            elif method=='general_che':
                dx = divergence_gradient(mobility(val[t-1], T), chem_pot(val[t-1], T) - stiffness(val[t-1], T)*laplacian(val[t-1]))
            else:
                raise 'ERROR unknown method '+method
            val[t]= val[t-1] + D*dx if not delta else D*dx
            if noise != 0:
                # val[t]+= diffusion_noise(val[t-1], dim)*noise
                if t%tskip_noise==0:
                    val[t]+= diffusion_noise(val[t-tskip_noise], dim, T)*noise
    return np.stack([val, np.full_like(val, T)],-1) if saveT else val[...,None]

def mobility(c, T): return (c+1)*(1-c)*(10*T+100*T**2)

def stiffness(c, T): return 1 + 2*T*(1-c**2/4)

def chem_pot(c, T): return c**3-c + T*np.log((c+1)/(1-c))

# def S_ij_func(a,b): return np.maximum(np.abs(1-a**2), 0) * np.maximum(np.abs(1-b**2), 0)
def S_ij_func(a,b): return np.maximum(1-a**2, 0) * np.maximum(1-b**2, 0)

def diffusion_noise(c0, dim, T=1.0):
    H_ij = np.stack([S_ij_func(c0, np.roll(c0,1,axis=i))*T for i in range(dim)], -1)
    noise = np.random.randn(*(H_ij.shape)) * np.sqrt(H_ij)
    H= np.sum([noise[...,i] - np.roll(noise[...,i],-1,axis=i) for i in range(dim)], axis=0)
    return H

if __name__ == "__main__":
    options = parser.parse_args()
    Ninput = list(map(int, options.data_size.split()))
    dim = len(Ninput) - 2
    Ns=Ninput[0]
    Nt=Ninput[1]
    Nspace=Ninput[2:]
    alldat=np.zeros(Ninput + [2 if options.saveT else 1], dtype=np.float32)
    c0=np.load(options.c0) if options.c0 else [None]*Ns
    for i in range(Ns):
        nmode = 4
        modes=[]
        for imode in range(nmode):
            x0= np.random.uniform(0.3, 0.7, dim)
            r0= np.random.uniform(0.04, 0.1)
            magnitude=np.random.uniform(1, 2)
            modes.append([x0, r0, magnitude])
        #print('debug i', i, modes)
        alldat[i]= PDE(modes, Nt, Nspace, method=options.method, D=options.D, 
          dc0=options.dc0,
          cmin=options.cmin, cmax=options.cmax, noise=options.noise, tskip_noise=options.tskip_noise, c0=c0[i], delta=options.delta,
          T=options.T, saveT=options.saveT)
    np.save(options.o, alldat[:,options.tbegin::options.tskip])

# without noise: 
# $> parallel  python generate-SPDE.py --data_size '"5 10000 128 128"' --method che --D 0.01 --tskip 100 --tbegin 2800 -o ::: `seq 70`
# with noise: 
# $> parallel  python generate-SPDE.py --data_size '"10 10000 128 128"' --method che --D 0.01 --tskip 100 --tbegin 2800 --noise 0.01 --cmin -0.55 --cmax 0.55   -o ::: `seq 35`
# with noise tstep=1: 
# $> parallel  python generate-SPDE.py --data_size '"10 1000 64 64"' --method che --D 0.01 --tskip 1 --tbegin 2800 --noise 0.01 --cmin -0.55 --cmax 0.55   -o ::: `seq 35`
