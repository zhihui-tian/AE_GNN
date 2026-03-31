#!/bin/env python3
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--insize", type=int, default=256, help="save array size")
parser.add_argument("--outsize", type=int, default=32, help="save array size")
parser.add_argument("data", help="data file(s) (.bin)", nargs='+')

options = parser.parse_args()

def load_data(fname, insize=None):
    if fname[-4:]=='.bin':
        return np.fromfile(fname, dtype=np.float).astype(np.float32).reshape([-1]+ [options.insize]*3)
    dat=np.load(fname)
    if fname[-4:]=='.npz':
        return dat['input_raw_data']
    else:
        return dat

nplot= len(options.data)
data=[]
for i in range(nplot):
    data.append( load_data(options.data[i]).astype('float32'))
    if data[i].shape[-1]<=3:
        data[i] = np.transpose(data[i], np.roll(np.arange(data[i].ndim),1).tolist())[0]
    data[i]=data[i].reshape((-1,)+data[i].shape[-3:])
    if data[i].shape[-3:] != [options.outsize]*3:
        print(i, [-1] + [j for ii in range(3) for j in (options.outsize, data[i].shape[ii+1]//options.outsize)])
        data[i] = data[i].reshape([-1] + [j for ii in range(3) for j in (options.outsize, data[i].shape[ii+1]//options.outsize)])
        #data[i]=np.transpose(data[i], (0, 1, 3, 5, 2,4,6))
        #data[i]=data[i].reshape((len(data[i]))+(options.outsize)*3+(-1))
        data[i]=np.mean(data[i],axis=(2,4,6))[0]

    print(options.data[i], 'value range', np.min(data[i]), np.max(data[i]))
data = np.array(data).reshape([-1]+[options.outsize]*3)
print(data.shape, len(data))
np.save('converted.npy', data)
