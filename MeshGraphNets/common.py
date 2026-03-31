# Lint as: python3
# pytorch port 
# ============================================================================
"""Commonly used data structures and functions."""

import enum
import torch
import numpy as np

class NodeType(enum.IntEnum):
    NORMAL = 0
    OBSTACLE = 1
    AIRFOIL = 2
    HANDLE = 3
    INFLOW = 4
    OUTFLOW = 5
    WALL_BOUNDARY = 6
    SIZE = 9


def triangles_to_edges(faces, unique_op=True):
    """Computes mesh edges from triangles."""
    if faces.shape[-1] == 3:
    # collect edges from triangles
        edges = torch.cat([faces[:, 0:2],
                                         faces[:, 1:3],
                                         torch.stack([faces[:, 2], faces[:, 0]], axis=1)], dim=0)
    elif faces.shape[-1] == 2:
    # collect edges from a SINGLE edge rather than triangle
        edges = faces[:, 0:2]
    else:
        raise ValueError(f'ERROR triangles_to_edges expects 2 or 3 nodes per face')
    # those edges are sometimes duplicated (within the mesh) and sometimes
    # single (at the mesh boundary).
    # sort & pack edges as single tf.int64
    if unique_op:
        receivers = torch.min(edges, dim=1)[0]
        senders = torch.max(edges, dim=1)[0]
        packed_edges = tf.bitcast(tf.stack([senders, receivers], dim=1), torch.int64)
        # remove duplicates and unpack
        unique_edges = tf.bitcast(tf.unique(packed_edges)[0], torch.int32)
        senders, receivers = tf.unstack(unique_edges, axis=1)
    else:
        receivers = edges[:,0]
        senders = edges[:,1]
    # create two-way connectivity
    return (torch.cat([senders, receivers], axis=0),
                    torch.cat([receivers, senders], axis=0))

def array2graph(frame, dim, periodic=True, device='cpu', time_dim=False):
    # frame shape: [batch, frame_shape, feature] without time dimension, otherwise [b, time, frame_shape, feature]
    graph = shape2graph(frame.shape[-dim-1:-1], frame.shape[0], periodic, device=device)
    if time_dim:
        graph['xtime'] = frame.permute((0,)+tuple(range(2,2+dim+1))+(1,)).reshape((-1,frame.shape[-1],frame.shape[1])).to(device)
        graph['x'] = frame.reshape((-1,frame.shape[-1]*frame.shape[1])).to(device)
    else:
        graph['x'] = frame.reshape((-1,frame.shape[-1])).to(device)
    graph['x_dense'] = graph['x']
    graph['x_idx'] = torch.arange(len(graph['x']))
    graph['x_fix'] = None
    return graph

import functools
@functools.lru_cache()
def shape2graph(shape, batch=1, periodic=True, device='cpu'):
    dim = len(shape)
    # print(f'debug frame {shape} {shape} {dim}')
    # n_node = np.prod(shape) * batch
    # mesh_pos = np.tile(np.stack(np.meshgrid(*[np.arange(i) for i in shape], indexing='ij'), axis=-1),(batch,1)+(1,)*dim).reshape((-1, dim))
    mesh_pos_1 = np.stack(np.meshgrid(*[np.arange(i) for i in shape], indexing='ij'), axis=-1).reshape((-1, dim))
    mesh_pos = np.tile(mesh_pos_1, (batch,1))
    # print(f'debug xyz {xyz} {mesh_pos}')
    if periodic:
        corner = mesh_pos_1
    else:
        corner = np.stack(np.meshgrid(*[np.arange(i-1) for i in shape], indexing='ij'), axis=-1).reshape((-1, dim))
    partitions = np.concatenate((np.eye(dim), -np.eye(dim)))
    partitions = np.stack([np.zeros_like(partitions), partitions], 1)[None,...]
    edge_index = (corner[:,None,None,:] + partitions) % shape
    edge_index = np.dot(edge_index, np.cumprod((1,)+shape[:0:-1])[::-1]).astype('int64').reshape(-1,2)
    edge_index = np.tile(edge_index[None,...], (batch,1,1)) + (np.arange(batch)*len(mesh_pos_1))[:,None,None]
    edge_index = edge_index.reshape(-1,2)

    graph = {}
    graph['mesh_pos'] = torch.tensor(mesh_pos, dtype=torch.float32, device=device)
    graph['node_type'] = torch.zeros((len(mesh_pos),1), dtype=torch.int64, device=device)
    graph['edge_index'] = torch.tensor(edge_index, dtype=torch.int64, device=device)
    graph['lattice'] = torch.tensor(np.diag(shape), dtype=torch.float32, device=device)
    graph['inv_lattice'] = torch.tensor(np.diag(1/np.array(shape)), dtype=torch.float32, device=device)
    graph['batch_index'] = torch.arange(batch).repeat_interleave(len(mesh_pos_1))[:,None].to(device)
    graph['aux_vars'] = {'batch_cell_len': torch.tensor([batch]+list(shape)).float()[None,:].to(device),
      'batch': batch, 'shape':tuple(shape), 'n_dense': len(graph['mesh_pos']),
      'bwh':(batch,)+tuple(shape),
      'mesh_pos_dense':graph['mesh_pos'], 'node_type_dense':graph['node_type'],
      'batch_index_dense':graph['batch_index'], 'edge_index_dense':graph['edge_index'],
      'ijk2int': torch.tensor(np.cumprod([1]+list(shape)[::-1])[::-1].copy(), device=device, dtype=torch.int64)}
    # print(f'debug',graph['mesh_pos'].shape, graph['node_type'].shape, graph['edge_index'].shape, graph['lattice'].shape)
    return graph


def output_by_method(cur_x, x_update, method=None):
    if method is None:
        return x_update
    else:
        outs = []
        for i,m in enumerate(method):
            if m == 'id':
                out = x_update[...,i]
            elif m == 'fix':
                out = cur_x[...,i]
            elif m == 'res':
                out = x_update[...,i] + cur_x[...,i]
            elif m == 'sigmoid':
                out = torch.sigmoid(x_update[...,i])
            else:
                raise ValueError(f'unknown output channel method {m}')
            outs.append(out)
        return torch.stack(outs, -1)
