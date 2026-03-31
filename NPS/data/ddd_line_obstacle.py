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


def register_args(parser):
    parser.add_argument('--tskip', type=int, default=1, help='skip time steps')
    parser.add_argument('--normalize', type=int, default=0, help='normalize x and y')
    parser.add_argument('--dist_scale', type=float, default=1., help='normalize distance')
    parser.add_argument('--node_y', type=str, default='velocity', choices=('velocity', 'force', 'f+v'))
    parser.add_argument('--var_stress', type=float, default=None, help='Scaling factor of stress supplied')
    parser.add_argument('--stress_lrange', type=int, default=0, help='append long-range stress to edge features')

def post_process_args(args):
    pass

def process_ddd_datafile(f, dim=2, tskip=10, var_stress=None):
    # sort_idx=1
    print('reading ddd trajectory', f)
    dat = np.load(f, allow_pickle=True)['arr_0'][1:]
    box = torch.from_numpy(dat[0, 0].astype(np.float32))
    # origin = box[:,0]
    cell_len = torch.from_numpy(np.diff(box).reshape((-1, dim)))
    # print(f'debug cell eln {cell_len}')
    obstacles = torch.from_numpy(dat[0, 3].astype(np.float32))
    # n_node = len(nodes[0])
    traj = []
    # inode = torch.arange(n_node)
    # edges = torch.stack((inode, torch.roll(inode,1,0)),-1)
    for t in range(0, len(dat), tskip):
        nodes = torch.from_numpy(dat[t, 1][:,:dim].astype(np.float32))
        links = dat[t, 2][:,:2].astype(int)
        # strong assertions to check validity of links
        # assert np.min(links)>=0 and np.max(links)<len(nodes) and len(links)==len(nodes), \
        #     ValueError(f'ERROR in DDD traj: {f} t= {t} node {nodes.shape} links {links.shape} min/max {np.min(links)} {np.max(links)}\nLinks: {links}')
        if not (np.min(links)>=0 and np.max(links)<len(nodes) and len(links)==len(nodes)):
            print(f'ERROR in DDD traj: {f} t= {t} node {nodes.shape} links {links.shape} min/max {np.min(links)} {np.max(links)}\nLinks: {links}')
            continue
        edge_index = torch.from_numpy(links)
        edge_index = torch.cat((edge_index, torch.flip(edge_index,[1])))
        edge_vec = vector_pbc(nodes[edge_index[:,1]] - nodes[edge_index[:,0]], cell_len).float()
        edge_features = torch.cat([
            edge_vec,
            torch.linalg.norm(edge_vec, dim=-1, keepdim=True)], dim=-1)
        # print(f'bond length ', links.shape)# torch.linalg.norm(edge_vec, dim=-1))
        force = torch.from_numpy(dat[t, 1][:,4:4+dim].astype(np.float32))
        velocity = torch.from_numpy(dat[t, 1][:,7:7+dim].astype(np.float32))
        if False: # get around bug in data
            pinned = torch.from_numpy(dat[t, 1][:,3].astype(np.int64))
        else:
            pinned = ((velocity.abs().sum(1))==0.0).long()
        # node_type = torch.ones(len(nodes), dtype=int)
        node_type = pinned
        type_embed_dim = int(var_stress is None) + common.NodeType.SIZE # 2
        node_features = nn.functional.one_hot(node_type, type_embed_dim).float()
        stress_lr = None
        if var_stress is not None:
            stress = torch.from_numpy((dat[t, 4][0:1]/var_stress).astype(np.float32)).expand(len(nodes)).reshape((-1, 1))
            node_features = torch.cat((stress, node_features), -1)
            if dat[t, 2].shape[1] > 5:
                stress_lr = torch.from_numpy(dat[t, 2][:,5:6]/var_stress).float()
        graph = Data(#'box':box, 
                obstacles=obstacles,
                cell_len=cell_len,
                node_features=node_features,
                edge_index=edge_index.T,
                edge_features=edge_features,
                stress_lrange=stress_lr,
                global_y=None, node_y=None, force=force, velocity=velocity)
        # for k, v in graph.items():
        #     print(k, v.shape)
        traj.append(graph)
    return traj



class ddd_line_obstacle:
    def __init__(self, args, datf):
        self.args = args
        self.dim = args.dim
        self.statistics = {'dist_scale': torch.tensor(args.dist_scale).float()}
        self.statistics['var_stress'] = None if args.var_stress is None else torch.tensor(args.var_stress).float()
        data_dir = datf
        node_y = args.node_y # force or velocity
        cache = f'{data_dir}/lines_cache_skip{args.tskip}_y{node_y}.pkl'
        # if os.path.exists(cache):
        try:
            self.flat = pickle.load(open(cache, "rb"))
            print(f'Loaded cache {cache}')
        except:
            dataset = []
            for f in sorted(glob.glob(data_dir+'/lines_*.npz')):
                try:
                    dataset += process_ddd_datafile(f, dim=args.dim, tskip=args.tskip, var_stress=args.var_stress)
                except:
                    print('failed to process', f)
                    pass
            self.flat = dataset
            print(f'Saving cache {cache}')
            with open(cache, 'wb') as fp:
                pickle.dump(self.flat, fp)
        # self.start_pos = np.arange(len(dataset))
        for d in self.flat:
            if node_y == 'f+v':
                d['node_y'] = torch.cat((d['force'], d['velocity']), -1)
            else:
                d['node_y'] = d[node_y].clone()
            del d['force'], d['velocity']
            if args.stress_lrange:
                # append long range stress to edge features
                d['edge_features'] = torch.cat((d['edge_features'], d['stress_lrange'].expand(-1,2).T.reshape((-1, 1))), -1)
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
                if node_y == 'f+v':
                    # correct bug about scaling stress in edge_features
                    d['edge_features'][:,:3] /= args.dist_scale
                    # print(f'debug mean {mean} stdy {std} edge fet', d['edge_features'][:3], 'node feat', d['node_features'][:3])
                else:
                    d['edge_features'] /= args.dist_scale
                d['node_y'] = (d['node_y']-mean)/std
        else:
            self.std_mean_y = (1, 0)

    def scale_back_y(self, y):
        return (y+self.var_mean[1])*self.var_mean[0]

    def __getitem__(self, i):
        # j = self.start_pos[i]
        return self.flat[i]

    def __len__(self):
        return len(self.flat)

if __name__ == '__main__':
    import matplotlib; matplotlib.use('TkAgg'); import matplotlib.pyplot as plt; import seaborn as sns;
    from collections import namedtuple

    d = process_ddd_datafile('/g/g90/zhou6/amsdnn/2D_forest/dataset5/lines_0.npz')
    args = {'dim':2, 'data':'/g/g90/zhou6/amsdnn/2D_forest/dataset5/', 'tskip':10, 'dist_scale':20000, 'normalize':1, 'node_y':'velocity'}
    args = namedtuple('Data_args_init', args.keys())(*args.values())
    ds = ddd_line_obstacle(args, None)

    # if True:
    #     plt.show()

