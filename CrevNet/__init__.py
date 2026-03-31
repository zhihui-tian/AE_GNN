import torch
import torch.nn as nn
from torch.autograd import Variable
import random
# from . import utils
# from . import data_utils
import numpy as np
# from tqdm import trange


def register_args(parser):
    parser.add_argument('--rnn_size', type=int, default=512, help='dimensionality of hidden layer')
    # parser.add_argument('--posterior_rnn_layers', type=int, default=2, help='number of layers')
    parser.add_argument('--predictor_rnn_layers', type=int, default=6, help='number of layers')
    # parser.add_argument('--gap', type=int, default=1, help='number of timesteps')
    # parser.add_argument('--z_dim', type=int, default=512, help='dimensionality of z_t')
    parser.add_argument('--g_dim', type=int, default=512,
                        help='dimensionality of encoder output vector and decoder input vector')
    parser.add_argument('--nblocks_2wae', type=str, default='2,2,2,2', help='number of blocks in 2way autoencoder')
    parser.add_argument('--nstrides_2wae', type=str, default='1,2,2,2', help='number of strides in 2way autoencoder')

def post_process_args(args):
    assert args.channel_first, ValueError(f'Conv nets in torch requires channel first')
    assert args.ngram == 1
    assert args.RNN or (args.n_in==args.ngram), ValueError('Set n_in=ngram when disabling RNN')
    args.nblocks_2wae = tuple(map(int, args.nblocks_2wae.split(',')))
    args.nstrides_2wae = tuple(map(int, args.nstrides_2wae.split(',')))


class CrevNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        # dtype = torch.cuda.FloatTensor
        # opt.data_type = 'sequence'
        from . import crevnet as model

        self.frame_predictor = model.zig_rev_predictor(args.rnn_size,  args.rnn_size, args.g_dim, \
          args.predictor_rnn_layers, 'lstm', [x//16 for x in args.frame_shape], \
          dim=args.dim, periodic=args.periodic)
        self.encoder = model.autoencoder(nBlocks=args.nblocks_2wae, nStrides=args.nstrides_2wae,
                            nChannels=None, init_ds=2,
                            dropout_rate=0., affineBN=True, in_shape=(args.nfeat_in,)+args.frame_shape,
                            mult=4, dim=args.dim, periodic=args.periodic)
        print(f'debug predictor\n{self.frame_predictor}\n encoder\n{self.encoder}')

    def reset(self, x):
        args = self.args
        # self.memo = Variable(torch.zeros(x.shape[0], args.rnn_size, *[w//16 for w in args.frame_shape]).cuda())
        self.memo = (torch.zeros(x.shape[0], args.rnn_size, *[w//16 for w in args.frame_shape]).cuda())
        self.frame_predictor.init_hidden(x)

    def forward(self, x, reset=False, **kwx):
        x_in = x.to('cuda')
        if reset:
            self.reset(x)
        h = self.encoder(x_in)
        h_pred, self.memo, _ = self.frame_predictor((h, self.memo))
        return self.encoder(h_pred, False)


def make_model(args):
    return CrevNet(args)


def make_trainer(args, loader, model, loss, checkpoint):
    return CrevNetTrainer(args, loader, model, loss, checkpoint)

from NPS.trainer import Trainer
class CrevNetTrainer(Trainer):
    def train_batch(self, x, criterion=None, epoch=1):
        model = self.model.get_model()
        teacher_forcing_ratio = np.maximum(0 , 1 - epoch * 0.003) 
        args = self.args
        n_in  = args.n_in
        n_out = args.n_out
        x_in = x[:,:n_in]
        if args.noise > 0:
            x_in += args.noise* torch.randn_like(x_in)
        self.optimizer.zero_grad()
        loss = 0

        for ei in range(n_in-1): 
            y= self.model(x_in[:, ei], (ei==0))
            loss += criterion(y, x_in[:,ei+1])

        decoder_input = x_in[:,-1] # first decoder input = last image of input sequence
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False 
        for di in range(n_out):
            y = self.model(decoder_input, reset=(not args.RNN) or ((n_in==1) and (di==0)))
            target = x[:,n_in+di]
            loss += criterion(y, target)
            if use_teacher_forcing:
                decoder_input = target # Teacher forcing
            else:
                decoder_input = y

        loss.backward()
        self.optimizer.step()
        return loss.item() / n_out


def get_training_batch():
    while True:
        for sequence in train_loader:
            batch = sequence.transpose_(0, 1).to('cuda') # data_utils.normalize_data(opt, dtype, sequence)
            yield batch


training_batch_generator = get_training_batch()


def get_testing_batch():
    # while True:
    for sequence in test_loader:
        batch = sequence.transpose_(0, 1).to('cuda') # data_utils.normalize_data(opt, dtype, sequence)
        yield batch


testing_batch_generator = get_testing_batch()
