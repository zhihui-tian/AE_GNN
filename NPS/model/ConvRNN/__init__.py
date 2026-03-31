__author__ = 'yunbo'

import torch
import torch.nn as nn
from model.common import CNN
from model.autoencoder import make_autoencoder

def register_args(parser):
    parser.add_argument('--lat_model', type=str, default='predrnn_v1', help='Which ConvRNN model')
    parser.add_argument('--autoencoder', type=str, default='None+pw1', help='autoencoder')
    parser.add_argument('--encdec_unet', type=int, default=0, help='add a unet connection between encoder and decoder')
    parser.add_argument('--dx', type=int, default=0, help='x(t+1)=x(t)+F(x(t))')

def post_process_args(args):
    assert args.channel_first, ValueError(f'CNN based model demands channel first')
    assert args.ngram == 1
    assert args.RNN or (args.n_in==args.ngram), ValueError('Set n_in=ngram when disabling RNN')
    if isinstance(args.nfeat_hid, int):
        args.nfeat_hid = [args.nfeat_hid] * args.n_mpassing


def make_model(args):
    nf_lat_in = args.nfeat_hid[0]
    if args.lat_model == 'predrnn_v1':
        from .predrnn_v1 import predrnn_v1
        rnn = predrnn_v1(args.n_mpassing, args.nfeat_hid, args.nfeat_in, args.nfeat_out, args.dim, args.periodic, args)
        nf_lat_in = args.nfeat_in
    elif args.lat_model == 'predrnn_v2':
        raise NotImplementedError()
        from .predrnn_v2 import predrnn_v2
        rnn = predrnn_v2(args.n_mpassing, args.nfeat_hid, args.nfeat_in, args.nfeat_out, args)
    elif args.lat_model == 'convlstm':
        raise NotImplementedError()
        from .convlstm import convlstm
        rnn = convlstm(args.n_mpassing, args.nfeat_hid, args.nfeat_in, args.nfeat_out, args)
    return ConvRNN(rnn, args.autoencoder, args.nfeat_in, nf_lat_in, args.nfeat_hid[-1], args.nfeat_out, args.dim, args.periodic, args.dx, args.encdec_unet, args)

class ConvRNN(nn.Module):
    def __init__(self, rnn, autoencoder, nfeat_in, nfeat_in_rnn, nfeat_out_rnn, nfeat_out, dim=2, periodic=False, dx=0, unet=0, args=None):
        super().__init__()
        self.rnn = rnn
        self.args = args
        self.autoencoder = make_autoencoder(autoencoder, nfeat_in, nfeat_in_rnn, nfeat_out_rnn, nfeat_out, dim=dim, periodic=periodic, args=args)
        self.nfeat_in = nfeat_in
        self.nfeat_out = nfeat_out
        self.dim = dim
        self.periodic = periodic
        self.dx = dx
        self.encdec_unet = CNN(nfeat_in_rnn+nfeat_out_rnn, [], nfeat_out_rnn, 1, activation=args.act, dim=dim, periodic=periodic, last_bias=False) if unet else None

    def forward(self, x, reset=False, **kwx):
        x1 = self.autoencoder(x)
        x2 = self.rnn(x1, reset=reset, **kwx)
        if self.encdec_unet is not None:
            x2 = self.encdec_unet(torch.cat([x2, x1], dim=1))
        # output_image = torch.sigmoid( self.decoder_D(concat ))
        x2 = self.autoencoder(x2, False)
        if self.dx:
            x2 += x[:, :self.nfeat_out]
        return x2

