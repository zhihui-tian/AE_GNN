#!/bin/env python
import numpy as np
import os.path
import argparse
from NPS_common.utils import load_array, save_array

parser = argparse.ArgumentParser()
parser.add_argument("data", help="npy file")
parser.add_argument("--validation", type=float, help="fraction to split into validation set. <=0: skip", default=0.2)
parser.add_argument("--test", type=float, help="fraction to split into test set. split<=0: skip", default=0.2)
parser.add_argument("--shuffle", action='store_true', help="whether to shuffle data", default=False)
parser.add_argument("--suffix", default='.npy', choices=('.npy', '.npz', '.sp.npz'))
# parser.add_argument("--in_len", type=int, help="Number of input sequence for npz", default=10)
# parser.add_argument("--out_len", type=int, help="Number of output sequence for npz", default=10)
parser.add_argument("--rescale_range", help="whether to shift values to min,max", default='')
parser.add_argument("--clip_range", help="clip values to min,max", default='')
options = parser.parse_args()

# def clips_from_data(dat, seq_len, input_seq_len):
#     nsteps, nchannel, nx, ny = dat.shape[:4]
#     nclips = nsteps//seq_len
#     #clips=np.vstack([np.arange(0,nsteps,input_seq_len), np.full((nclips*2), input_seq_len)])
#     clips=np.zeros((nclips, 2, 2))
#     clips[:,0,0] = np.arange(0, nsteps, seq_len)
#     clips[:,1,0] = np.arange(input_seq_len, nsteps, seq_len)
#     clips[:,0,1] = input_seq_len
#     clips[:,1,1] = seq_len-input_seq_len
#     clips=clips.astype(np.int32).transpose((1,0,2))
#     return clips


def simple_augment_video(vid, iop=None):
    """" vid should be [nframe, nx, ny, nchannel] iop from 1-8 if nx==ny, or 1-4 """
    if iop is None:
        iop= np.random.choice(8) if vid.shape[1]==vid.shape[2] else np.random.choice(4)
    v = vid if iop<=4 else np.array(np.transpose(vid, (0, 2, 1, 3)))
    iop= (iop-1)
    if iop==0:
        return v
    elif iop==1:
        return np.flip(v, 1)
    elif iop==2:
        return np.flip(v, 2)
    else:
        return np.flip(v, (1,2))


# seq_len=options.in_len + options.out_len
# input_seq_len=options.in_len
# output_seq_len=options.out_len
# Nchannel=1

alldat = load_array(options.data)
Ns, Nt= alldat.shape[:2]
# Nsize= list(alldat.shape[2:])
# nframes= Ns*Nt
# nclips = nframes//seq_len
# dims=np.array([[Nchannel]+Nsize], dtype=np.int32)
# print("dims",dims)

if options.shuffle:
    np.random.shuffle(alldat)
if options.rescale_range:
    if options.clip_range:
        crange=list(map(float, options.clip_range.split(',')))
        alldat= np.clip(alldat, crange[0], crange[1])
    m0, m1 = list(map(float, options.rescale_range.split(',')))
    v0 = np.amin(alldat)
    v1 = np.amax(alldat)
    alldat = (alldat-v0)*(m1-m0)/(v1-v0) + m0

nvalid = max(int(Ns*options.validation), 0)
ntest = max(int(Ns*options.test), 0)

save_array('train'+options.suffix, alldat[:(Ns-nvalid-ntest)])
save_array('valid'+options.suffix, alldat[(Ns-nvalid-ntest):(Ns-ntest)])
if ntest > 0: save_array('test'+options.suffix, alldat[(Ns-ntest):])
    # alldat=alldat.reshape((-1,1, Nx, Ny))
    # ntot=alldat.shape[0]
    # ntest=int(Ns*options.split)*Nt
    # print("all input_raw_data", alldat.shape)

    # if ntest<=0 or ntest>ntot:
    #     np.savez_compressed("data.npz", dims=dims, clips=clips_from_data(alldat, seq_len, input_seq_len), input_raw_data=alldat)
    #     print("all %d saved to data.npz"%(ntot))
    # else:
    #     dat=alldat[:-ntest]
    #     np.savez_compressed("train.npz", dims=dims, clips=clips_from_data(dat, seq_len, input_seq_len), input_raw_data=dat)
    #     dat=alldat[-ntest:]
    #     np.savez_compressed("valid.npz", dims=dims, clips=clips_from_data(dat, seq_len, input_seq_len), input_raw_data=dat)
    #     print("saved %d to train.npz and %d to valid.npz"%(ntot-ntest, ntest))
