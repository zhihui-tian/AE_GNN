# Lint as: python3
# pytorch port 

# ============================================================================
"""Model for neural phase simulation."""

import torch
import torch.nn as nn
from model.common import CNN


def make_autoencoder(autoencoder, nfeat_in, nfeat_in_lat, nfeat_out_lat, nfeat_out, dim=2, periodic=False, args=None):
    if autoencoder == 'rev2wae':
        nfeat_lat = args.nfeat_in * 2**(len(args.nstrides_2wae) * args.dim)
        assert nfeat_lat == nfeat_in_lat and nfeat_lat == nfeat_out_lat
        from CrevNet.crevnet import autoencoder as AE
        assert args.nfeat_autoencoder == args.nfeat_in * 2**(len(args.nstrides_2wae) * args.dim)
        return AE(nBlocks=args.nblocks_2wae, nStrides=args.nstrides_2wae,
                        nChannels=None, init_ds=2,
                        dropout_rate=0., affineBN=True, in_shape=(args.nfeat_in,)+args.frame_shape,
                        mult=4, dim=args.dim, periodic=args.periodic)
    elif '+' in autoencoder:
        enc, dec = list(map(lambda x: x.strip(), autoencoder.split('+')))
        return CNN_encoderdecoder(enc, dec, nfeat_in, nfeat_in_lat, nfeat_out_lat, nfeat_out, dim, periodic, args)
    else:
        raise ValueError(f'Unknown autoencoder type {autoencoder}')


class CNN_encoderdecoder(nn.Module):
    """Dimension reduction with autoencoder. in_shape=(B,W,H(,D),C), out_shape=(B,w,h(,d),c)"""
    def __init__(self, enc, dec, nfeat_in, nfeat_in_lat, nfeat_out_lat, nfeat_out, dim, periodic, args):
        # print(f'debug', enc, dec, nfeat_in, nfeat_in_lat, nfeat_out_lat, nfeat_out, dim, periodic, args)
        super().__init__()
        ae = []
        for i, typ in enumerate((enc, dec)):
            nfi, nf, nfo = (nfeat_in, nfeat_in_lat, nfeat_in_lat) if i==0 else (nfeat_out_lat, nfeat_out_lat, nfeat_out)
            if typ in ('', 'None'):
                assert nfi == nfo
                f = nn.Identity()
            elif typ.startswith('pw'):
                ## point wise
                nlayer = int(typ[2:])
                f = CNN(nfi, [nf]*nlayer, nfo, 1, activation=args.act, dim=dim, periodic=periodic, last_bias=False)
            elif type.startswith('CNN'):
                raise NotImplementedError()
            else:
                raise ValueError(f'Unknown enc/decoder type {typ}')
            ae.append(f)
        self.encoder, self.decoder = ae

    def forward(self, x, encode=True):
        if encode:
            return self.encoder(x)
        else:
            return self.decoder(x)

