import numpy as np
import os, glob
import argparse
from NPS_common.utils import save_array

frac_train = 0.85
parser = argparse.ArgumentParser()      
parser.add_argument('dirs', nargs='*')
parser.add_argument('-o', type=str, default='out.npy')
args = parser.parse_args()

dat = []
for dir in args.dirs:
    print(f'reading {dir}')
    obstacles = np.loadtxt(f'{dir}/obstacles.dat')
    frames = []
    for f in sorted(glob.glob(f'{dir}/disloc_*.dat'), key=lambda x: int(os.path.basename(x)[:-4].split('_')[-1])):
        frame = np.loadtxt(f)
        frame_int = np.loadtxt(f.replace('disloc_', 'dislocint_'))
        frames.append(np.stack((frame, frame_int, obstacles),-1).astype('float32'))
    frames = np.stack(frames,0)
    dat.append(frames)
dat = np.stack(dat, 0)
save_array(args.o, dat)
