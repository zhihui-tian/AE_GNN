import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data", default='40000', help="data file, [:,(y,x)]")
parser.add_argument("--dimy", type=int, default=1, help="y dimension (rest of a row in data is x)")
parser.add_argument("--noise", type=float, default=0.0, help="noise")

parser.add_argument("--nhidden", default='128,128', help="n hidden")
parser.add_argument("--act", default='relu', help="activation")
parser.add_argument("--dropout", type=float, default=0.0, help="add dropout (if positive)")

parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
parser.add_argument("--epochs", type=int, default=100, help="epochs")
parser.add_argument("--minibatch", type=int, default=10000, help="minibatch size")
options = parser.parse_args()

def S_ij_func(a,b): return np.maximum(1-a**2, 0) * np.maximum(1-b**2, 0) #* (0.03**2)

def get_dat(nsample, noise=0, dimy=1):
    if isinstance(nsample, str):
        dat = np.load(nsample)
        c = dat[:,dimy:]
        y = dat[:,:dimy]
    else:
        c = np.random.uniform(-1.05, 1.05, (nsample, 2))
        y = S_ij_func(c[:,0:1], c[:, 1:2])
        if noise > 0:
            y+= np.random.uniform(-1,1,y.shape)*noise
    return c, y


dat_x, dat_y = get_dat(options.data if os.path.exists(options.data) else int(options.data), options.noise, dimy=options.dimy)
dat_y0 = S_ij_func(dat_x[:,0:1], dat_x[:, 1:2])
nsample = len(dat_y)
n_in = dat_x.shape[1]
n_out = dat_y.shape[1]

# print('debug dat', dat_x[:3], dat_y[:3])

cgrid = np.linspace(-1.1, 1.1, 56)
cgrid= np.array(np.meshgrid(cgrid, cgrid))
shape2d = cgrid.shape[1:]
cgrid = np.transpose(cgrid.reshape(2,-1))#.reshape((-1,2))
c0 = torch.tensor(cgrid,device='cuda').float()


epochs = options.epochs
print_every = 10

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
nGPU = torch.cuda.device_count()
idx = np.arange(nsample)

# def init_weights(m):
#     if type(m) == nn.Linear:
#         torch.nn.init.normal_(m.weight, 0, 0.1)
#         try:
#             m.bias.data.fill_(0.001)
#         except:
#             pass

class my_act(nn.Module):
    def forward(self, x):
        return 1/(1+x**2) #torch.exp(-x**2)

nfeat=list(map(int, options.nhidden.split(',')))
try:
    # import model
    from NPS.model.common import MLP
    model = MLP(n_in, nfeat, n_out, activation=options.act, conv=False, dropout=options.dropout)
    print('model: from MLP', nfeat)
except:    
    model = nn.Sequential(
        nn.Linear(n_in,  nfeat[0]), nn.ReLU(), 
        nn.Linear(nfeat[0], nfeat[1]), nn.ReLU(),
        nn.Linear(nfeat[1], 1, bias=False))
print('model:', model)
# model.apply(init_weights)
pmodel = torch.nn.DataParallel(model)
model = pmodel.module
pmodel.to(device)
criterion = nn.MSELoss()

def train_model():
    optimizer = torch.optim.Adam(pmodel.parameters(), lr=options.lr)
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1/(epoch/500+1))
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.8)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=40, verbose=True)
    for t in range(0, epochs):
        # print('debug t=',t)
        np.random.shuffle(idx)
        for idat, idx_this in enumerate(idx.reshape((-1,options.minibatch*nGPU))):
            x = torch.tensor(dat_x[idx_this], device=device).float()
            y = torch.tensor(dat_y[idx_this], device=device).float()
            # print('debug idx', idx_this, '\nx y', x.shape, y.shape, x[:3], y[:3]);            exit()
            # y_pred, omega, scatter_mat= toy_model(x)
            # loss = lossf(y_pred, x[:,1], omega, scatter_mat)
            y_pred = pmodel(x)
            loss = criterion(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step(loss)
        if t % print_every == 0 or t==epochs-1:
            print(t, loss.item())
            # np.save('noise_corr', pmodel(c0).cpu().detach().numpy().reshape(shape2d))
    # print('debug model weight before exit', model.m1[0].state_dict())
    # print('RMSE(final)', np.sqrt(np.mean((pmodel(torch.tensor(dat_x, device=device).float()).cpu().detach().numpy()-dat_y0)**2)))
    print('RMSE(final)', np.sqrt(loss.item()))
    np.save('gt-pd.npy', np.concatenate([y.detach().cpu().numpy(), y_pred.detach().cpu().numpy()],1))
    return model

train_model()
#calculator_array.py '[plt.plot(np.diag( noise_corr.npy )),plt.plot(np.diag(np.flip( noise_corr.npy ,0))),plt.plot(np.diag( noise_corr.npy [0])),plt.plot(np.diag( noise_corr.npy [:,0])), plt.show()]'
