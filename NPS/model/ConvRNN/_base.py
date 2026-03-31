import torch
import torch.nn as nn

class _base_ConvRNN(nn.Module):
    def __init__(self, num_layers, num_hidden, nfeat_in, nfeat_out, dim, periodic, args=None):
        super().__init__()
        self.args = args
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.nfeat_in = nfeat_in
        self.nfeat_out = nfeat_out
        self.dim = dim
        self.periodic = periodic
        self.init_layers()

    def init_layers(self):
        pass

