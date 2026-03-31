import matplotlib.pyplot as plt; import numpy as np; from matplotlib import cm

a=np.load('output.npy')
n=len(a)

coords = []
def onclick(event):
    global ix, iy
    ix, iy = event.xdata, event.ydata
    print('x = %d, y = %d'%(ix, iy), f' event.xdata, event.ydata {event.xdata} {event.ydata}')

    global coords
    coords.append((ix, iy))

    # if len(coords) == 2:
    #     fig.canvas.mpl_disconnect(cid)
    return coords

selected = []
def onpick(event):
    for i1 in range(5):
        for j1 in range(5):
            if event.artist == ax[i1,j1]:
                print(f'igroup {igroup} i1 j1 {i1} {j1} selected {igroup*25+i1*5+j1}')
                selected.append(igroup*25+i1*5+j1)

for i in range(len(a)//25):
    igroup = i
    fig, ax=plt.subplots(5,5, figsize=(15,15))
    for i1 in range(5):
        for j1 in range(5):
            ax[i1,j1].imshow(a[igroup*25+i1*5+j1,...,0])
            ax[i1,j1].set_xticks([])
            ax[i1,j1].set_yticks([])
            ax[i1,j1].set_picker(True)
    fig.canvas.mpl_connect('pick_event', onpick)
    fig.tight_layout()
    plt.show()

np.savetxt('selected.txt', selected)
np.save('selected.npy', a[selected])

