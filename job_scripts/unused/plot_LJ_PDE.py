import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
import subprocess, glob

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("dirs", nargs='+', help="run dirs")
parser.add_argument('--dat', action='store_true')
parser.add_argument('--shift', action='store_true')
parser.add_argument('--align', action='store_true')
options = parser.parse_args()

dirs = sorted(options.dirs)
#dirs=sorted(glob.glob("../experiment/LJ20cT.che_CT_n16,16,16_atanh_lr2e-3_symm*/"))
Ndir=len(dirs)
#dir="../experiment/LJ20cT.che_CT_n16,16,16_atanh_lr2e-3_symm"
c_grid = np.linspace(0,1,101)
T_grid = np.linspace(2,7,101)
# subprocess.check_output("""python `grep main.py _DIR_/config.txt|tail -n1 |sed 's:main.py:scripts/visualize_model.py:;s:=\({[^}]*}\):="\1":'` --visualization_setting='{"xlabel":"c,T","func":["chem_pot","stiffness","mobility"],"xgrid":[[0,1,101],[3,10.5,101]]}'""".replace('_DIR_',dir), shell=True)
if options.dat:
    for d in dirs:
        print('generating data in', d)
        subprocess.check_output("""grep main.py _DIR_/config.txt|tail -n1 |sed 's:.*main.py::;s:=\({[^}]*}\):=\\1:'|xargs python NPS/scripts/visualize_model.py --visualization_setting='{"xlabel":"c,T","func":["chem_pot","stiffness","mobility"],"xgrid":[[0,1,101],[3,10.5,101]]}' """.replace('_DIR_',d), shell=True)

names = ["chem_pot", "stiffness", "mobility"]
display_names = [r'$\mu$', r'$\lambda$', r'$M$']
# ys = [np.load(f'{dir}/{name}.npy') for name in names]
print('loading data')
y_all = np.array([[np.load(f'{d}/{name}.npy') for name in names] for d in dirs])
sign = np.sign(y_all[:,1,50,50])
print('setting sign')
y_all *= sign[:,None,None,None]
if options.shift:
    # shift chemical potential to zero at c=0.5
    y_all[:,0] -= y_all[:,0,50:51,:]
else:
    y_all[:,0] -= y_all[:,0,50:51,0:1]
if options.align:
    print('aligning with chemical potential')
    align_with = 0 # with chem_pot
    A_align=y_all[:,align_with].reshape((Ndir,-1))
    b_align=y_all[0,align_with].reshape(-1)
    scale = np.dot(A_align, b_align)/np.sum(A_align**2,axis=1)
    # y_all[:,0:1] *= scale[:,None,None,None]
    # y_all[:,1:2] *= scale[:,None,None,None]
    # y_all[:,2:3] /= scale[:,None,None,None]
    print('scale', scale)
    for i in range(Ndir): print(i, y_all[i,0,:3,:3])

colors = plt.cm.coolwarm(np.linspace(0,1,101))
highlighted= [0, 30, 60, 90]
# plt.setp(axs[:,  0], ylabel='T')
with PdfPages('plot_LJ_PDE.pdf') as pdf:
    for idir in range(Ndir):
        fig, axs = plt.subplots(1, 3, figsize=(12,4))
        fig.suptitle(dirs[idir])
        axs=np.array(axs).reshape((1,3))
        plt.setp(axs[-1, :], xlabel=r'$c$')
        ys = y_all[idir]
        for i, name in enumerate(names):
            ax = axs[0,i]
            y = ys[i]
            # if name=='mobility':
            #     scaling_fac = tgrid[:,None]
            # if True:
            #     y = y*scaling_fac/y[:, 75:76]
            ax.set_ylabel(display_names[i])
            for it in sorted(list(range(0,91,10))):
                if it in highlighted:
                    ax.plot(c_grid, y[:,it], color=colors[it], label=f'T={T_grid[it]:.1f}', linewidth=1.6)
                else:
                    ax.plot(c_grid, y[:,it], color=colors[it], linewidth=0.8)
            ax.legend()
        plt.tight_layout(pad=0.1, w_pad=0.3, h_pad=0.0)
        # plt.tight_layout(pad=0)
        if True:
            pdf.savefig()#plt.savefig('plot_LJ_PDE.pdf')
        else:
            plt.show()
        plt.close(fig)
