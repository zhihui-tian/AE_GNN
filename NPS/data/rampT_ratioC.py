__author__ = 'Fei Zhou'

import numpy as np
from .longclip import longclip

class rampT_ratioC(longclip):
    def __init__(self, args, datf, train=True, name='typ'):
        super().__init__(args, datf, train, name)
        # set kinetic energy to average of the clip
        T_profile = np.mean(self.flat[:,2,...].reshape((self.nclip, self.clip_len)+tuple(self.flat.shape[2:])), 
          axis=tuple(range(2,2+self.dim)))
        x = np.arange(self.clip_len)
        A = np.stack([np.ones_like(x), x],-1)
        solution = np.linalg.lstsq(A, T_profile.T, rcond=None)[0]
        self.flat[:,2,...] = A.dot(solution).T.reshape((-1,)+((1,)*self.dim))
        self.flat[:,0,...] /= (self.flat[:,0,...]+self.flat[:,1,...])
        self.flat = self.flat[:,0:3:2]
