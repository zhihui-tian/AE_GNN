#!/bin/env python
import numpy as np
import subprocess
import io

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("chunk_file", help="e.g. chunk.dat")
parser.add_argument("col", help="comma separated list of columns (starting 1) e.g. 5,-1")
parser.add_argument("factor", help="comma separated list of factors for corresponding column e.g. 0.1,0.01")
parser.add_argument("-o", default='out.npy', help="e.g. out.npy")
options = parser.parse_args()
options.col = list(map(int,options.col.split(',')))
options.factor = np.array(list(map(float, options.factor.split(','))))

def co(instr, split=False):
    out=subprocess.Popen(instr, stdout=subprocess.PIPE, shell=True, universal_newlines=True).communicate()[0]
    return out.split('\n') if split else out

shape=np.loadtxt(io.StringIO(co("awk 'NR==5 {print 0.5/$2,0.5/$3,0.5/$4}' %s"%options.chunk_file))).astype(int)
arr = np.loadtxt(io.StringIO(co("grep '^  [0-9]' %s |awk '{print %s}'"%(options.chunk_file, 
  ",".join(["$%d"%s if s>=0 else "$(NF-%d)"%(-s-1) for s in options.col])))))
print('debug', arr.shape, options.col, options.factor, arr[:5])
print('debug command', ("grep '^  [0-9]' %s |awk '{print %s}'"%(options.chunk_file, 
  ",".join(["$%d"%s if s>=0 else "$NF-%d"%(-s-1) for s in options.col]))))
arr = arr.reshape((-1, len(options.col))) * options.factor[None,:]
print('debug shape', [-1] + shape.tolist() + [len(options.col)])
arr = arr.reshape([-1] + shape.tolist() + [len(options.col)])
np.save(options.o, arr.astype('float32'))
