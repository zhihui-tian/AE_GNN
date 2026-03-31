import numpy as np
import sys
# from torchsummary import summary
import torch
from torchinfo import summary
import pickle

device = torch.device('cuda')
G= pickle.load(open(sys.argv[1], 'rb'))['G_ema'].to(device)

# G = legacy.load_network_pkl(dnnlib.util.open_url(sys.argv[1]))['G_ema'].to(device) # type: ignore
print(f'G.c_dim {G.c_dim} G.z_dim {G.z_dim}')
print(G)
# summary(G)

summary(G,input_size=[(1,512,),(1,1)])

