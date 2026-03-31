import numpy as np
import torch
from torch_geometric.data import Data
import os

# import sys, os; sys.path.append(os.path.join(sys.path[0], '../../..'))

from glob    import glob
from pathlib import Path


def read_hubbard1band_output(fy):
    fx = fy[:-4]+'.in'
    # print(f'debug reading {fx}')
    with open(fx) as f:
        lines = f.readlines()
    nnode = int(lines[0])
    node_feat = torch.tensor([np.fromstring(l, sep=' ') for l in lines[1:1+nnode]])
    nedge = int(lines[1+nnode])
    edge_index = torch.tensor([np.fromstring(l, dtype=int, sep=' ') for l in lines[2+nnode:2+nnode+nedge]])-1
    edge_index = torch.cat((edge_index, torch.flip(edge_index,[1])))
    edge_feat = torch.tensor([np.fromstring(l, sep=' ') for l in lines[2+nnode+nedge:]])
    edge_feat = torch.cat((edge_feat, edge_feat))
    # print(f'nnode {nnode} ned {nedge} node_feat ed_f {edge_feat} {lines[1:1+nnode]} {[l.split() for l in lines[1:1+nnode]]} {[np.fromstring(l,sep=" ") for l in lines[1:1+nnode]]}')

    with open(fy) as f:
        lines = f.readlines()
    global_y = torch.tensor([[float(lines[0])]])
    node_y = torch.tensor([np.fromstring(l, sep=' ') for l in lines[1:1+nnode]])
    # print(f'debug node_feat, edge_index, edge_feat, global_y, node_y {[node_feat, edge_index, edge_feat, global_y, node_y]}')
    return node_feat, edge_index, edge_feat, global_y, node_y

def load_hubbard1band(paths, dtype=torch.float32, cache_data=True, filter=""):
    dirs = sorted(glob(paths))
        # try:
        #     dataset = torch.load(cache_ds)
        #     return dataset
        # except:
        #     pass

    dataset = []
    for d in dirs:
        ds_dir = []
        cache_ds = f'{d}/cache.pt'
        if cache_data:
            try:
                ds_dir = torch.load(cache_ds)
                dataset += ds_dir
                print(f'  loaded cached dataset from {cache_ds}')
                continue
            except:
                pass
        print('  processing', d)
        for fname in sorted(glob(d+'/*.out')):
            node_feat, edge_index, edge_feat, global_y, node_y = read_hubbard1band_output(fname)
            data = Data(
                node_features=node_feat.type(dtype),
                edge_index=edge_index.T,
                edge_features=edge_feat.type(dtype),
                global_y = global_y.type(dtype), node_y = node_y.type(dtype)
            )
            ds_dir.append(data)
            # print(len(ds_dir), fname)
        dataset += ds_dir
        if cache_data:
            torch.save(ds_dir, cache_ds)
            # print(f'Caching dataset to {cache_ds}')
    print(f'Total data: {len(dataset)}')
    if filter == "":
        pass
    elif filter== "-no_node_y":
        for data in dataset:
            data.node_y = data.node_y[:,:0]
    elif filter== "-sum_node_y":
        for data in dataset:
            data.node_y = torch.sum(data.node_y, 1, keepdim=True)
    else:
        raise ValueError(f'Unknown filter {filter}')
    return dataset

if __name__ == '__main__':
    for filter in ("", "-no_node_y", "-sum_node_y"):
        ds = load_hubbard1band('/g/g90/zhou6/lassen-space/data/hubbard-1band/3/linear/2_1_*_0.5/', cache_data=False, filter=filter)
        print(len(ds), ds[:5])
        from torch_geometric.loader import DataLoader
        loader = DataLoader(ds, batch_size=32, shuffle=True)
        for j, d in enumerate(loader):
            print(j, d)
            if j==0:
                print(d.edge_index)
            if j>3:
                break


