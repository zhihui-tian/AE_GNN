#!/bin/env python3
import numpy as np
from NPS_common.utils import load_array, save_array
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--insize", type=int, default=-1, help="input array size")
parser.add_argument("--nchannel", type=int, default=1, help="nchannel")
parser.add_argument("--dim", type=int, default=3, help="dimension of each frame (default 3d data)")
parser.add_argument("--outsize", type=int, default=-1, help="save array size. If not < insize, downsize with average")
parser.add_argument("--tave", type=int, default=1, help="no. of time steps to average")
parser.add_argument("--lowpass_cut", type=float, default=-0.1, help="for averaging/smoothing, cutoff freq. of lowpass filter")
parser.add_argument("--dataset", action="store_true", default=False, help="Only collecting data (ignore all options except --ichannel")
parser.add_argument("--cat", action="store_true", default=False, help="concatenate rather than stack together")
parser.add_argument("--cat_slice", type=str, default='slice(None)', help="slice after reading each data point, e.g. slice(None) or slice(None,None,20)")
parser.add_argument("--ichannel", default="", help="only keep certain channel in the last dimension (e.g. 0 removing last dimension or 0:1 keeping dimension)")
parser.add_argument("--dtype", default="auto", help="save data type. auto to preserve input type")
parser.add_argument("-o", default="output.npy", help="output file (default output.npy)")
parser.add_argument("data", help="data file(s) (.bin)", nargs='+')

options = parser.parse_args()
if (not options.dataset) and options.dtype=="auto":
    options.dtype="float32"
options.cat_slice = eval(options.cat_slice)

def load_data(fname, insize=-1, outsize=-1, dim=3, nchannel=1, ochannel=-1,):
    root_func={2: np.sqrt, 3:np.cbrt}
    if fname[-4:]=='.bin':
        dat = np.fromfile(fname, dtype=np.float)
        in_L = int(root_func[dim](len(dat)/nchannel)) if insize==-1 else insize
        input_size = [in_L]*dim
#        print('debug in size', input_size, nchannel)
        dat = dat.reshape(input_size + [nchannel], order='F')
        #print(i, [-1] + [j for ii in range(3) for j in (options.outsize, data[i].shape[ii+1]//options.outsize)])
        out_L = in_L if outsize==-1 else outsize
        if out_L != in_L:
            dat = downsize_average(dat, out_L)
        return dat
    return load_array(fname)


def downsize_average(arr, out_L):
    arr = arr.reshape([-1] + [j for i in range(3) for j in (out_L, arr.shape[i+1]//out_L)])
        #data[i]=np.transpose(data[i], (0, 1, 3, 5, 2,4,6))
        #data[i]=data[i].reshape((len(data[i]))+(options.outsize)*3+(-1))
    return np.mean(arr, axis=(2,4,6))
    
def smoothing(arr, tave, cutoff=0.125):
    if cutoff < 0:
        return arr
    from scipy import signal
    nseq, ndiscard = np.divmod(len(arr), tave)
    if ndiscard > 0:
        print("WARNING: discarding %d frames to perform time averaging"%ndiscard)
    lowpass = signal.sosfiltfilt(signal.butter(4, cutoff, output='sos'), arr, axis=0)
    return lowpass[tave-1::tave]
#    return np.mean(arr[:nseq*tave].reshape((nseq, tave) + arr.shape[-(arr.ndim-1):]), axis=1)

def to_series(flist, typ='auto', dataset=False, insize=-1, outsize=-1, dim=3, tave=1, nchannel=1, ichannel="", cutoff=0.1, cat=False):
    nplot= len(flist)
    if nplot <=0:
        return
    if cat:
        return np.concatenate([load_data(f)[options.cat_slice] for f in flist],0)
    arr1 = load_data(flist[0], insize=insize, outsize=outsize, dim=dim, nchannel=nchannel)
    type = arr1.dtype if typ == 'auto' else typ
    # a=np.zeros((nplot,) + arr1.shape, dtype=arr1.dtype)
    a = []
    for i, fn in enumerate(flist):
        if dataset:
            a.append(load_data(fn))
        else:
            a.append(load_data(fn, insize=insize, outsize=outsize, dim=dim, nchannel=nchannel))
        print(fn, 'value range', np.min(a[i]), np.max(a[i]))
    if not dataset:
        a = smoothing(a, tave, cutoff=cutoff)
    if ichannel:
        c=ichannel.split(':')
        a=a[..., int(c[0]) if len(c)==1 else int(c[0]):int(c[1])]
    a = np.concatenate(a)
    if cat:
        a=a.reshape((-1,)+a.shape[2:])
    return a.astype(type)

save_array(options.o, to_series(options.data, typ=options.dtype, dataset=options.dataset,
  insize=options.insize, dim=options.dim, tave=options.tave, nchannel=options.nchannel, ichannel=options.ichannel, cutoff=options.lowpass_cut, cat=options.cat))
