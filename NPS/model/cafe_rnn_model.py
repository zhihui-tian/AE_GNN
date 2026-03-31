# https://github.com/vincent-leguen/PhyDNet

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import confusion_matrix
from NPS.model.common import diff_conv
from NPS_common.pt_utils import euler_angles_to_matrix
from NPS_common.utils import grid_points

from PhyDNet.models import ConvLSTM
from PhyDNet.convlstm import ConvLSTMEncDec

def get_rot_matrix(eulers, euler_convention):
    return torch.where(((eulers+1).abs()<1e-6).all(dim=-1,keepdim=True), torch.zeros((*eulers.shape[:-1], 9), dtype=eulers.dtype, device=eulers.device),
              euler_angles_to_matrix(eulers, euler_convention).reshape(*eulers.shape[:-1],9))

def encode_cafe(x, n_state=4, encode_euler=False, euler_convention='ZXZ'):
    encode_id = (n_state == 4)
    node_type = nn.functional.one_hot(x[...,0].long(), n_state) if encode_id else x[..., 0:1]
    euler_enc = get_rot_matrix(x[...,1:4], euler_convention) if encode_euler else x[...,1:4]
    fields = x[..., 4:] # may need to append T(t+1)!!!
    return torch.cat([node_type, euler_enc, fields], dim=-1)

def encode_rotmat(x):
    return encode_cafe(torch.from_numpy(x), n_state=1, encode_euler=True).numpy()


def register_args(parser):
    parser.add_argument('--evolver', type=str, default='convlstm')
    parser.add_argument('--filter_nb', type=int, default=3, help='filter size when searching for capturing source from neighboring cells')
    parser.add_argument('--euler_convention', default='ZXZ')
    parser.add_argument('--n_state', type=int, default=4, help='state embedding')
    parser.add_argument('--n_euler', type=int, default=3, choices=(3, 9), help='euler embedding')
    parser.add_argument('--n_field', type=int, default=1, help='no. of fields')
    parser.add_argument('--n_field_ext', type=int, default=1, help='no. of external fields')
    # parser.add_argument('--last_act', type=str, default='sigmoid', help='Activation at end. "" to disable')
    parser.add_argument('--encdec_unet', type=int, default=0, help='add a unet connection between encoder and decoder')
    # parser.add_argument('--dx', type=int, default=0, help='x(t+1)=x(t)+F(x(t))')

def post_process_args(args):
    assert not args.channel_first, ValueError(f'Channel last please')
    assert args.ngram == 1
    assert args.RNN or (args.n_in==args.ngram), ValueError('Set n_in=ngram when disabling RNN')
    assert args.dim ==3
    assert not args.periodic
    assert args.loss_from_model

def make_model(args):
    if args.evolver == 'convlstm':
        convcell =  ConvLSTM(input_dim=args.nfeat_hid, hidden_dims=[args.nfeat_hid] * args.n_mpassing, n_layers=args.n_mpassing, kernel_size=args.kernel_size, dim=args.dim, periodic=args.periodic)
        nfeat_in  = args.n_state+args.n_euler+args.n_field+args.n_field_ext
        nfeat_out = args.n_state+args.n_euler+args.n_field
        evolver = ConvLSTMEncDec(convcell, nfeat_in=nfeat_in, nfeat_out=nfeat_out, nfeat_hid=args.nfeat_hid, encdec_unet=bool(args.encdec_unet), dx=True, dim=args.dim, periodic=args.periodic, last_act='')
    else:
        raise ValueError('Name of network unknown %s' % args.rnn_model_name)
    return Cafe_RNN(evolver, nfeat_in=args.nfeat_in, nfeat_out=args.nfeat_out, dim=args.dim, periodic=args.periodic, filter_nb=args.filter_nb,
      euler_convention=args.euler_convention, n_state=args.n_state, n_euler=args.n_euler, n_field=args.n_field, n_field_ext=args.n_field_ext)

        
class Cafe_RNN(torch.nn.Module):
    # All data are channel last except calling an RNN
    def __init__(self, evolver, nfeat_in=6, nfeat_out=5, dim=3, periodic=False, filter_nb=3, euler_convention='XYZ', n_state=4, n_euler=3, n_field=1, n_field_ext=1):
        super().__init__()
        self.evolver = evolver
        self.nfeat_in = nfeat_in
        self.nfeat_out = nfeat_out
        self.dim = dim
        self.filter_nb = filter_nb
        self.periodic = periodic
        self.euler_convention = euler_convention
        self.n_state = n_state
        self.n_euler = n_euler
        self.n_field = n_field # size
        self.n_field_ext = n_field_ext # T
        self.n_embedding = self.n_state + self.n_euler + self.n_field
        self.diff_conv = diff_conv(dim, filter_nb, self.n_euler, periodic=periodic)
        buffer = filter_nb//2
        self.buffer = buffer
        indices_neighbor = torch.tensor(grid_points([2*buffer]*dim)-buffer).reshape(-1,dim) # all points made from [-1,0,1]
        # neighbors only, i.e. remove 0,0,0
        indices_neighbor = indices_neighbor[indices_neighbor.abs().sum(dim=1)>0]
        ## pad one space for batch index
        indices_neighbor = torch.cat((torch.zeros_like(indices_neighbor[:,:1]), indices_neighbor), 1) # shape(1, -1, 1+dim) # , device='CUDA'
        self.register_buffer('indices_neighbor', indices_neighbor[None, ...].requires_grad_(False)) #,persistent=False  #in_channel, out_channel,, device='cuda')
        self.padding = (buffer,)*self.dim*2
        self.mode = ('circular' if periodic else 'constant')

    def get_rot(self, eulers):
        if self.n_euler == 3:
            return eulers
        else:
        # print('eulers', eulers.shape, euler_angles_to_matrix(eulers, self.euler_convention).shape)
            return get_rot_matrix(eulers, self.euler_convention)

    def encode(self, x):
        return encode_cafe(x, n_state=self.n_state, encode_euler=(self.n_embedding==9), euler_convention=self.euler_convention)

    def decode(self, x, out): # out_cf in the Channel First format
        """Final output: state id (1), euler (3), size(1)"""
        state0 = x[...,0].long()
        flag_inactive = state0==0
        state1 = torch.argmax(out[..., 0:self.n_state], axis=-1)
        # print(f'debug x {x.shape} out {out.shape} state0 {state0.shape} state1 {state1.shape} flag {flag_inactive.shape}')
        state1[flag_inactive] = 0 # keep inactive

        field1 = out[..., -self.n_field:]
        # hard coded: inactive/liquid has size= -1, solid 1, mushy in (0,1)
        field1[state1<=1] = -1
        # field1[state1==2].clamp_(0, 1) # WARNING: in place clamp_ NOT working here!!
        field1[state1==2] = field1[state1==2].clamp(0, 0.92)
        field1[state1==3] = 1

        euler1 = x[...,1:4] ## copy from t=0 Euler angles
        euler1[state1<=1] = -1
        flag_solidify = torch.logical_and(state0<=1, state1>1)
        if torch.any(flag_solidify):
            # for mushy or solid cells, inherit Euler angles unless it was liquid phase at t=0
            # print(f'debug  flag_solidify {flag_solidify.shape}, {flag_solidify} {torch.nonzero(flag_solidify).shape}')
            # rot1 = out[flag_solidify, self.n_state:self.n_state+ self.n_euler]
            rot1 = out[flag_solidify][:, self.n_state:self.n_state+self.n_euler]
            # print(f'debug  rot1 {rot1.shape}, {rot1}')

            # ijk1 = torch.cat((inputs['batch_index'], inputs['mesh_pos'].int()), -1)[flag_solidify]
            ijk1 = torch.nonzero(flag_solidify)
            # padding
            ijk1[:, 1:] += self.buffer
            euler0_pad = F.pad(x[...,1:4].permute(0,4,1,2,3), self.padding, mode=self.mode).permute(0,2,3,4,1)

            # print(f'debug ijk1 {ijk1.shape} {ijk1} \n rot1 {rot1.shape}, {rot1} \n nbs {self.indices_neighbor.shape} {self.indices_neighbor}')
            # ijk1_nbs = ((ijk1[:,None,:] + self.indices_neighbor).reshape((-1, self.dim+1)) % inputs['aux_vars']['batch_cell_len']).long()
            ijk1_nbs = (ijk1[:,None,:] + self.indices_neighbor).reshape((-1, 1+self.dim)).long()
            # ijk1_nbs = torch.inner(ijk1_nbs.float(), inputs['aux_vars']['ijk2int'].float()).long()
            # print(f'debug ijk {ijk1.shape} ijk1_nbs {ijk1_nbs.shape} euler0_pad {euler0_pad.shape} euler0_pad[tuple(i for i in ijk1_nbs.T)] {euler0_pad[tuple(i for i in ijk1_nbs.T)].shape}')
            rot0_nbs = euler0_pad[tuple(i for i in ijk1_nbs.T)]
            # print(f'debug rot0_nbs {rot0_nbs.shape} self.get_rot(rot0_nbs) {self.get_rot(rot0_nbs).shape} euler0_pad {euler0_pad.shape} ijk1 {ijk1.shape} rot1 {rot1.shape}')
            # print(f'debug rot0_nbs {rot0_nbs.shape} {rot0_nbs}')
            rot0_nbs = self.get_rot(rot0_nbs).reshape([ijk1.shape[0], -1, 3])
            # print(f'debug rot0_nbs {rot0_nbs.shape} {rot0_nbs}')
            dist = (rot0_nbs - rot1[:,None]).square().sum(dim=-1)
            imin = torch.argmin(dist, dim=1)[:,None,None].expand(-1,-1,3)
            # print(f'debug dist {dist.shape} {dist}', torch.argmin(dist, dim=1).shape, torch.argmin(dist, dim=1))
            # idx_min = ijk1_nbs[:,torch.argmin(dist, dim=1)]
            euler1[flag_solidify] = torch.gather(rot0_nbs, 1, imin).squeeze(1) #inputs['x_dense'][idx_min,1:4]
            # print(f'debug e {euler1.shape} field {field1.shape} state {state1.shape}')
            # rot0 = graph.node_features[self.n_state:self.n_state+9]
            # euler1 = torch.zeros((out.shape[0], 3), device=out.device, dtype=out.dtype)
        # print(f'debug state1 {state1.shape} euler {euler1.shape} field1 {field1.shape}')
        return torch.cat([state1[...,None], euler1, field1], dim=-1)

    def forward(self, x, reset=False, target=None, criterion=None, mask=None, **kwx):
        ### input to RNN is limited to: cell state, oct_size, T
        ### hidden2 represents the octahedron at the site
        x_encoded = self.encode(x)
        out = self.evolver(x_encoded.permute(0,4,1,2,3), reset).permute(0,2,3,4,1)
        y = self.decode(x, out)
        if target is None:
            return y, (0,)
        else:
            # slice_mask = ???
            # state1 = torch.argmax(out[:,:self.n_state], axis=1)
            state0 = x[..., 0].long()
            state1 = y[..., 0].long()

            # tgt = target[inputs['x_idx']]
            # slice_mask = slice_mask.repeat(inputs['aux_vars']['batch']) if slice_mask is not None else torch.ones(tgt.shape[0], dtype=torch.bool, device=tgt.device)
            if mask is None: mask = torch.ones_like(state0, dtype=torch.bool)
            # loss_state = F.cross_entropy(out[:,:self.n_state], tgt[:, 0].long())
            # mask = 
            # print(f'debug out {out.shape} x {x.shape} target {target.shape} x_encoded {x_encoded.shape}')
            state1gt = target[..., 0].long()
            important = ((state0 != state1) + (state0 != state1gt) + (state0==2) + (state1==2) + (state1gt==2)).reshape(-1)
            unimportant = important.logical_not()
            state1_prob = out[..., :self.n_state].reshape((-1, self.n_state))
            state1gt = state1gt.reshape(-1)
            loss_state1 = F.cross_entropy(state1_prob[important], state1gt[important])
            loss_state2 = F.cross_entropy(state1_prob[unimportant], state1gt[unimportant])
            mushy_flag  = torch.logical_and(state1 == 2, mask)
            mushy_solid = torch.logical_and(state1 >= 2, mask)
            loss_euler = criterion(y[mushy_solid][..., 1:4], self.get_rot(target[mushy_solid][..., 1:4]))
            loss_field = criterion(y[mushy_flag ][..., 4:5], target[mushy_flag][..., 4:5])
        return y, (loss_state1.nan_to_num(), loss_state2.nan_to_num(), loss_euler.nan_to_num(), loss_field.nan_to_num())

    def analyze(self, y, target):
        state1 = y[..., 0].int()#.long().reshape(-1).detach().cpu()
        state1gt = target[..., 0].int()#.long().reshape(-1).detach().cpu()
        print('confusion', confusion_matrix(state1gt, state1, labels=list(range(4)))[1:,1:].reshape(-1))



