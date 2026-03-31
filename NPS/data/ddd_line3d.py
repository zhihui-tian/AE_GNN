__author__ = 'Fei Zhou'

import numpy as np
import os, glob
# from NPS_common.utils import load_array_auto
from NPS.model.common import vector_pbc
import pickle

import torch
import torch.nn as nn
from torch_geometric.data import Data
from MeshGraphNets import common
import ase
from ase.neighborlist import neighbor_list


def register_args(parser):
    parser.add_argument('--tskip', type=int, default=1, help='skip time steps')
    parser.add_argument('--normalize', type=int, default=1, help='normalize x and y')
    parser.add_argument('--dist_scale', type=float, default=1., help='normalize distance')
    parser.add_argument('--force_scale', type=float, default=1., help='normalize force')
    parser.add_argument('--node_y', type=str, default='velocity', choices=('velocity', 'force', 'vel_inst'))
    parser.add_argument('--nonlink_cutoff', type=float, default=0., help='interaction cutoff for non-linked nodes')
    # parser.add_argument('--var_stress', type=float, default=None, help='Scaling factor of stress supplied')

def post_process_args(args):
    pass

def process_ddd3d_datafile(f, dim=3, tskip=1, cutoff=0., periodic=True):
    """ each frame contains 4 arrays (and there are 1000 frames per npz file). 
* the first array is the 3D box, 
* the second is the nodes array, 
    ** The fourth column of the node array (previously the pinned flag), is now a flag that is 1 if the node is a physical node (meaning a node connected to 3 or more links), or 0 if it is a discretization node (2 connections).
* the third is the links array, 
    **Columns 3-5 of the links array are the Burgers vectors of the dislocation lines, which define their behavior. In the 2D forest system only one dislocation was present so we could ignore it. In 3D however we will need to account for it, as we are now dealing with several dislocations of different types.
* the fourth is the applied stress tensor. 
"""
    assert dim == 3
    print('reading ddd trajectory', f)
    dat = np.load(f, allow_pickle=True)['arr_0'][1:]
    box = torch.from_numpy(dat[0, 0].astype(np.float32))
    # origin = box[:,0]
    cell_len = torch.from_numpy(np.diff(box).reshape((-1, dim)))
    # print(f'debug cell eln {cell_len}')
    traj = []

    for t in range(0, len(dat), tskip):
        nodes = torch.from_numpy(dat[t, 1][:,:dim].astype(np.float32))
        node_flag = torch.from_numpy(dat[t, 1][:,3].astype(np.int64))
        stress = torch.from_numpy(dat[t, 3])
        links = dat[t, 2][:,:2].astype(int)
        link_burgers = dat[t, 2][:,2:]
        assert link_burgers.shape[1] == 3
        edge_index = torch.from_numpy(links)
        edge_index = torch.cat((edge_index, torch.flip(edge_index,[1])))
        edge_vec = vector_pbc(nodes[edge_index[:,1]] - nodes[edge_index[:,0]], cell_len).float()
        edge_features = torch.cat([
            torch.from_numpy(np.concatenate((link_burgers, link_burgers))).float(),
            edge_vec,
            torch.linalg.norm(edge_vec, dim=-1, keepdim=True)], dim=-1)
        if cutoff > 0:
            nnode = nodes.shape[0]
            tmp_cell = ase.Atoms(positions=np.pad(nodes, ((0,0),(0,3-dim))), cell=np.pad(cell_len[0],(0,3-dim), constant_values=100), pbc=periodic)
            all_src, all_dst = neighbor_list("ij", tmp_cell, cutoff, self_interaction=False)
            edge_ij = edge_index[:,0]*nnode + edge_index[:,1]
            all_ij = all_src*nnode + all_dst
            # combined = torch.cat((edge_ij, nonlink_ij))
            # uniques, counts = combined.unique(return_counts=True)
            # nonlink_ij = uniques[counts == 1]
            nonlink_ij = np.setdiff1d(all_ij, edge_ij.numpy())
            nonlink_src, nonlink_dst = np.divmod(nonlink_ij, nnode)
            nonlink_index = torch.stack((torch.from_numpy(nonlink_src), torch.from_numpy(nonlink_dst)), 1)
            nonlink_index = torch.cat((nonlink_index, torch.flip(nonlink_index,[1])))
            nonlink_vec = vector_pbc(nodes[nonlink_index[:,1]] - nodes[nonlink_index[:,0]], cell_len).float()
            nonlink_features =  torch.cat([
                torch.zeros_like(nonlink_vec),
                nonlink_vec,
                torch.linalg.norm(nonlink_vec, dim=-1, keepdim=True)], dim=-1)
            edge_index = torch.cat((edge_index, nonlink_index))
            edge_features = torch.cat((edge_features, nonlink_features))
        # print(f'bond length ', links.shape)# torch.linalg.norm(edge_vec, dim=-1))
        force = torch.from_numpy(dat[t, 1][:,4:4+dim].astype(np.float32))
        velocity = torch.from_numpy(dat[t, 1][:,7:7+dim].astype(np.float32))
        vel_inst = torch.from_numpy(dat[t, 1][:,10:10+dim].astype(np.float32)) if dat[t, 1].shape[1]>10 else None
        # node_type = torch.ones(len(nodes), dtype=int)
        node_type = node_flag
        type_embed_dim = common.NodeType.SIZE 
        # node_features = torch.cat((nn.functional.one_hot(node_type, type_embed_dim).float(), force), -1)
        node_features = nn.functional.one_hot(node_type, type_embed_dim).float()
        graph = Data(#'box':box, 
                cell_len=cell_len,
                node_features=node_features,
                edge_index=edge_index.T,
                edge_features=edge_features,
                vel_inst = vel_inst,
                global_y=None, node_y=None, force=force, velocity=velocity)
        # for k, v in graph.items():
        #     print(k, v.shape)
        traj.append(graph)
    return traj



class ddd_line3d:
    def __init__(self, args, datf):
        self.args = args
        self.dim = args.dim
        self.statistics = {'dist_scale': torch.tensor(args.dist_scale).float(), 'force_scale': torch.tensor(args.force_scale).float()}
        # self.statistics['var_stress'] = None if args.var_stress is None else torch.tensor(args.var_stress).float()
        data_dir = args.data
        node_y = args.node_y # force or velocity
        cache = f'{data_dir}/lines_cache_skip{args.tskip}_cut{args.nonlink_cutoff}.pkl'
        # if os.path.exists(cache):
        try:
            self.flat = pickle.load(open(cache, "rb"))
            print(f'Loaded cache {cache}')
        except:
            dataset = []
            for f in sorted(glob.glob(data_dir+'/lines_*.npz')):
                try:
                    dataset += process_ddd3d_datafile(f, dim=args.dim, tskip=args.tskip, cutoff=args.nonlink_cutoff, periodic=args.periodic)
                except:
                    print('failed to process', f)
                    pass
            self.flat = dataset
            print(f'Saving cache {cache}')
            with open(cache, 'wb') as fp:
                pickle.dump(self.flat, fp)
        # self.start_pos = np.arange(len(dataset))
        for d in self.flat:
            d['node_y'] = d[node_y].clone()
            if node_y in ('velocity', 'vel_inst'):
                # append force to node features
                d['node_features'] = torch.cat((d['node_features'], d['force']/args.force_scale), -1)
            del d['force'], d['velocity'], d['vel_inst']
        if args.normalize:
            y_list = []
            for d in self.flat:
                y_list.append(torch.stack(torch.var_mean(d['node_y'], dim=0, unbiased=False, keepdim=True)))
            var, mean = torch.mean(torch.stack(y_list, 0), dim=0)
            std = torch.sqrt(var).clamp(min=1e-10)
            self.statistics['std_y'] = std
            self.statistics['mean_y'] = mean
            print(f'Dataset normalized with dist, std, mean: {self.statistics}')
            for d in self.flat:
                d['edge_features'][:,(-self.dim-1):] /= args.dist_scale
                d['node_y'] = (d['node_y']-mean)/std
        else:
            self.std_mean_y = (1, 0)

    # def scale_back_y(self, y):
    #     return (y+self.var_mean[1])*self.var_mean[0]

    def __getitem__(self, i):
        # j = self.start_pos[i]
        return self.flat[i]

    def __len__(self):
        return len(self.flat)

if __name__ == '__main__':
    import matplotlib; matplotlib.use('TkAgg'); import matplotlib.pyplot as plt; import seaborn as sns;
    from collections import namedtuple

    d = process_ddd3d_datafile('/g/g90/zhou6/amsdnn/3D_bulk/dataset1/lines_10.npz')
    args = {'dim':3, 'data':'/g/g90/zhou6/amsdnn/3D_bulk/dataset1/', 'tskip':1, 'dist_scale':20000, 'force_scale':1e9, 'normalize':1, 'node_y':'velocity', 'nonlink_cutoff':4.5, 'periodic':True}
    args = namedtuple('Data_args_init', args.keys())(*args.values())
    ds = ddd_line3d(args, None)

    # if True:
    #     plt.show()

