__author__ = 'Fei Zhou'

import numpy as np
from .longclip import longclip

class ratioC(longclip):
    def __init__(self, args, datf, train=True, name='typ'):
      """
      Assuming input channels: rhoA, rhoB, KE, PE
      """
      super().__init__(args, datf, train, name)
      self.flat[:,0,...] /= (self.flat[:,0,...]+self.flat[:,1,...])
      self.flat = self.flat[:,0:3:2]
