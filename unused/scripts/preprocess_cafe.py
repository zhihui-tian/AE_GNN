import numpy as np
# import os, glob
import argparse
import re
import pickle
import torch
from torch.nn import functional as F
from NPS_common.utils import save_array, repr_simple_graph

# from MeshGraphNets.amr_pointwise import amr_pointwise_v2 as amr_pointwise
# from MeshGraphNets.common import array2graph
# def snapshot2graph(x_in, flag, args, mesher, buffer=2, g_global=False):
#     dim = 3
#     print(x_in.shape)
#     diff_as_flag = False
#     if diff_as_flag:
#     # print(f'debug x_in {x_in.shape}')
#         input = F.pad(flag, (buffer,)*2*dim, 'circular')
#         # print('input', input.shape, 'x_in', x_in.shape)
#         max = F.max_pool3d(input, 2*buffer+1, stride=1)
#         min = F.max_pool3d(-input, 2*buffer+1, stride=1)
#         print('in to graph shape', (max+min)[...,None].shape, 'max', max.shape, np.bincount((max+min).reshape(-1)))
#         x_in = (max+min)
#     g = array2graph(flag, args.dim, periodic=args.periodic, time_dim=True)
#     if g_global:
#         g_gl = g['aux_vars']
#         del g_gl['node_type_dense']
#         return g_gl
#     del g['xtime']
#     print('g before meshing', repr_simple_graph(g))
#     g = mesher.remesh(g)
#     print('g after  meshing', repr_simple_graph(g))
#     # print('g after meshing', [(k, v.shape if hasattr(v, 'shape') else ([(k1,v1.shape if hasattr(v1, 'shape') else v1) for k1,v1 in v.items()] if isinstance(v, dict) else v)) for k,v in g.items()])
#     return g


def read_cafe_sequence(flist, args):
    # mesher = amr_pointwise(args.dim, periodic=args.periodic, 
    #           nfeat_fix=0, threshold=args.amr_threshold, buffer=args.amr_buffer)
    dat = []
    #  OUTPUT columns
    ## state, size, T, 3_euler_angles
    col = [7, 14, 6, 8,9,10]
    col_all = np.array(col + [3,4,5])
    for fn in flist:
        print(f'reading {fn}')
        raw = np.loadtxt(fn)[:, col_all].astype(np.float32)
        cell = [len(np.unique(raw[:,-3+i])) for i in range(3)]
        raw = raw[:,:-3]
        # cell = (raw[-1][3:6]*100).astype(int) + 1
        # print(cell)
        # raw = torch.from_numpy(raw).float()
        # print(raw.shape)
        raw[:, 0] += 1 # 0: inactive, 1=fluid, 2=mushy, 3=solid
        raw[np.where(raw[:,0]==3), 1] = 1e10
        raw[np.where(raw[:,0]<=1), 1] = -1e10
        raw[:, 1] = np.clip(raw[:,1]/args.size_scale, -1, 1)
        raw[:, 2] -= args.T0 # shift T to reference T0
        # g = snapshot2graph(raw, (raw[...,0]==1.).float()[None,None,...,None], args, mesher)
        # print('g', g)
        # dat.append(g)

        raw = raw.reshape(cell[::-1]+[-1]).astype(np.float32)
        raw = np.swapaxes(raw, 0, 2)
        if args.data_slice:
            raw = eval(f'lambda x: x[{args.data_slice}]')(raw)
        dat.append(raw)
    return dat


def main():
    print('Make sure files are ordered correctly!!!')
    parser = argparse.ArgumentParser()      
    parser.add_argument('files', nargs='*')
    # parser.add_argument('-c', type=str, default=':', help="':' to select all, or e.g. '3,1,-1'")
    # parser.add_argument('--amr_column', type=int, default=0, help='which column for meshing threshold')
    # parser.add_argument('--amr_buffer', type=int, default=2, help='buffer')
    # parser.add_argument('--amr_threshold', type=float, default=1e-3, help='threshold of removing mesh')
    parser.add_argument('--size_scale', type=float, default=0.015, help='size scaling factor')
    parser.add_argument('--T0', type=float, default=926.0, help='reference Temperature')
    parser.add_argument('--data_slice', default='', help='Slice input data, default no slicing, Example: ":50" limits training sequences to 50 (to study training vs dataset size); "...,:1" ignores channels after 1st. THIS CHANGES DATA ITSELF')
    parser.add_argument('-o', type=str, default='out.npy')
    args = parser.parse_args()
    args.files.sort(key=lambda f: int(re.sub('\D', '', f)))
    args.dim = 3; args.periodic = True
    # args.c = slice(None) if args.c==':' else np.array(map(int, args.c.split(',')))
    seq = read_cafe_sequence(args.files, args)
    save_array(args.o, np.array([seq]))
    # print('seq', np.array([seq]))
    # with open( args.o, 'wb') as fp:
    #     pickle.dump(np.array([seq]), fp)

## FINAL RUN for isotropic T data
# python scripts/preprocess_cafe.py `/bin/ls -v ~/data/cafe_gnn/ALE3DCAFE_PANDAS_iso_40/*.dat` --data_slice='...,:78,:'
# python -c "import numpy as np; d=np.load('train.npy'); valid = np.stack([d[0,j:j+10] for j in (200,250,300,350)]); np.save('valid.npy', valid)"
# python scripts/preprocess_cafe.py `/bin/ls -v ~/data/cafe_gnn/MINI/ALE3DCAFE_PANDAS_iso_40/*.dat` --data_slice='...,:78,:'

if __name__ == '__main__':
    main()

## header
# head -n1  /usr/WS1/cafe_gnn/ALE3DCAFE_PANDAS_iso_40/iso_40.588.dat |sed 's/\t/\n/g;s/# //' |cat -n
#
# f=/usr/WS1/cafe_gnn/ALE3DCAFE_PANDAS_gau2_13/gau2_13.59.dat; for i in `seq 11`; do echo -n $i `head -n1 $f |awk '{print \$('$i'+1)}'` "  "; awk "NR>1 {print \$$i}" $f|sort |uniq -c|head -n4 |tr '\n' ' '; echo; done 
# for f in `/bin/ls -v /usr/WS1/cafe_gnn/ALE3DCAFE_PANDAS_gau2_13/gau*.dat`; do awk 'NR>1 {if ($8>0){printf("    A%d %g %g %g", $8, $4*100,$5*100,$6*100)}}' $f|awk '{print (NF/4), "    ", $0  }'|sed 's/    /\n/g'; done  > tmp.xyz
# f=/usr/WS1/cafe_gnn/ALE3DCAFE_PANDAS_gnn_40/gnn_40.21.dat; for i in `seq 11`; do echo -n $i `head -n1 $f |awk '{print \$('$i'+1)}'` "  "; awk "NR>1 {print \$$i}" $f|sort |uniq -c |sort -gr|head -n8 |tr '\n' ' '; echo; done 
# for f in `/bin/ls -v /usr/WS1/cafe_gnn/ALE3DCAFE_PANDAS_gnn*/*.dat`; do awk 'NR>1 {if ($8>0){printf("    A%d %g %g %g %g %g %g", $8, $4*100,$5*100,$6*100, $9/6.29,$10/6.29,$11/6.29)}}' $f|awk '{print (NF/7), "    ", $0  }'|sed 's/    /\n/g'; done  > tmp.xyz
#
## count number of cells in each type
# for f in `/bin/ls -v /usr/WS1/cafe_gnn/ALE3DCAFE_PANDAS_iso_40/*.dat`; do 
#   awk 'BEGIN {count[-1]=0;count[0]=0;count[1]=0;count[2]=0;} (NR>1) {count[$8]++} END{ for (var in count) {printf("%s %s ",var,count[var])} print ""}' $f
# done > tmpcount.txt
#
## T profile
# tail -n1 `/bin/ls -v /usr/WS1/cafe_gnn/ALE3DCAFE_PANDAS_iso_40/*.dat` |grep -v '^ *$' |grep -v 'dat' |awk '{print $7}' > tmpT.txt
# p 'tmpcount.txt' u 2 w l t 'cell -1', 'tmpcount.txt' u 4 w l t 'cell 0 (liquid)', 'tmpcount.txt' u 6 w l t 'cell 1 (mushy)', 'tmpcount.txt' u 8 w l t 'cell 2 (solid)', 'tmpT.txt' u (($1-919)*70000) w l t '(T-919)*70000'


    # # print(np.meshgrid(*cell_size, indexing='ij'))
    # print(np.transpose(np.mgrid[0:cell[0], 0:cell[1], 0:cell[2]], (3,2,1,0)).reshape((3,-1)))
    # print(np.reshape(np.array(np.meshgrid(*cell_size, indexing='ij'))/100,(3,-1)).T)
    # assert np.allclose(np.reshape(np.array(np.meshgrid(*cell_size, indexing='ij'))/100,(3,-1)).T, raw[:, 3:6])

#     frames = np.stack(frames,0)
#     dat.append(frames)
# dat = np.stack(dat, 0)
