#!/bin/env python
#from IPython.display import HTML
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import argparse
from NPS_common.utils import load_array

DIM=2
parser = argparse.ArgumentParser()
parser.add_argument("--tskip", type=int, default=1, help="time skip")
parser.add_argument("--tbegin", type=int, default=0, help="time begin")
parser.add_argument("--tend", type=int, default=-999, help="time end")
parser.add_argument("-o", default='', help="save as gif")
parser.add_argument("data", help="data file(s) (.npy or .npz)", nargs='+')
parser.add_argument("--range", default='', help="set to override automatic range, e.g. when it explodes")
parser.add_argument("--ichannel", "-i", "-c", default='0', help="which channel to show, default 0 (single channel), 0:2 or 1:4 RGB multi-channel")
parser.add_argument("--rv", action='store_true', help="invert RGB color")
parser.add_argument("--channel_index", type=int, default=-1, help="channel position, -1 for channel last (default) (-999 means no channel)")
parser.add_argument("--delay", type=int, default=25, help="Delay between frames")
parser.add_argument("--interp", type=str, default='antialiased', help="Interpolation method: antialiased, nearest, etc")
options = parser.parse_args()
if options.o: matplotlib.use('Agg')
options.ichannel = list(map(int, options.ichannel.split(':')))
if len(options.ichannel) == 1:
    options.ichannel = slice(options.ichannel[0], options.ichannel[0]+1)
    use_RGB = False
else:
    options.ichannel = slice(*options.ichannel)#eval(f'slice({options.ichannel})')
    use_RGB = True

nplot= len(options.data)
fig, axs = plt.subplots(1, nplot, figsize=(nplot*4, 4))

def run_animation(anim, fig):
    anim_running = True

    def onClick(event):
        nonlocal anim_running
        if anim_running:
            anim.event_source.stop()
            anim_running = False
        else:
            anim.event_source.start()
            anim_running = True

    fig.canvas.mpl_connect('button_press_event', onClick)

data=[]
dat_minmax = []
for i in range(nplot):
    data.append(load_array(options.data[i]).astype('float32'))
    if options.channel_index == -999:
        data[i] = data[i][...,None]
    elif options.channel_index > -1:
        new_ax = list(range(0,options.channel_index))+list(range(options.channel_index+1,data[i].ndim))+[options.channel_index]
        data[i] = np.transpose(data[i], new_ax)
    assert data[i].ndim>=DIM+2 # with channels
    data[i]=data[i][...,options.ichannel][...,:3]
    if data[i].shape[-1] == 2:
        data[i] = np.concatenate((data[i], np.zeros(data[i].shape[:-1]+(3-data[i].shape[-1],))), -1)
    if (data[i].shape[-1] == 3) and options.rv:
        data[i] = 1 - data[i]
    if options.tend==-999:
        data[i]=data[i].reshape((-1,)+data[i].shape[-(DIM+1):])[options.tbegin::options.tskip]
    else:
        data[i]=data[i].reshape((-1,)+data[i].shape[-(DIM+1):])[options.tbegin:options.tend:options.tskip]
    # dat_minmax.append([np.amin(data[i],(0,1,2)), np.amax(data[i],(0,1,2))])
    dat_minmax.append([np.amin(data[i]), np.amax(data[i])])
    print(options.data[i], 'value range', dat_minmax[i])
dat_minmax=np.array(dat_minmax)
if options.range:
    allmin = float(options.range.split(',')[0])
    allmax = float(options.range.split(',')[1])
else:
    allmin=np.amin(dat_minmax[:,0])
    allmax=np.amax(dat_minmax[:,1])

ims=[]
for i in range(nplot):
    ax = axs if nplot==1 else axs[i]
    # note using global vmin, vmax
    ims.append(ax.imshow(data[i][0,:,:], cmap=plt.get_cmap('hot'), vmin=allmin, vmax=allmax, interpolation=options.interp))
#        fig.colorbar(ims[i], ax=ax)
    ax.set_xlim((0, data[i].shape[-2]))
    ax.set_ylim((0, data[i].shape[-3]))
    ax.set_title('0')

# animation function. This is called sequentially
def animate(t):
    for i in range(nplot):
        ims[i].set_data(data[i][t])
        print('step t', t)
        ax = axs if nplot==1 else axs[i]
        ax.set_title(str(t))
    return ims

# call the animator. blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, animate, frames=len(data[0]), interval=options.delay, blit=True)

run_animation(anim, fig)

if options.o:
    anim.save(options.o, writer='imagemagick', fps=6)
else:
    plt.show()
