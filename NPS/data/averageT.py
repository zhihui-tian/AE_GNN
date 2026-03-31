__author__ = 'Fei Zhou'

import numpy as np
from .longclip import longclip

class averageT(longclip):
    def __init__(self, args, datf, train=True, name='typ'):
        super(averageT, self).__init__(args, datf, train, name)
        # set kinetic energy to average of the clip
        self.flat[:,2,...] = np.mean(self.flat[:,2,...].reshape((self.nclip, self.clip_len)+tuple(self.flat.shape[2:])), 
          axis=tuple(range(1,2+self.dim))).repeat(self.clip_len).reshape((-1,)+((1,)*self.dim))
        self.flat = self.flat[:,0:3:2]
