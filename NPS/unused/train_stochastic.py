import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import models
import utility
import data
from option import args
from models.common import laplacian_roll as Laplacian

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
nGPU = torch.cuda.device_count()
os.makedirs(args.dir, exist_ok=True)

loader = data.Data(args)
checkpoint = utility.checkpoint(args)
# model = models.Model(args, checkpoint).model
model = models.make_model(args, checkpoint)
# print('debug model weight before', model.m1[0].state_dict())

def load_model(model, save_file, optimizer=None, loss_hist=None, epoch=None):
    # try:
    #     model.load_state_dict(torch.load(ckpt), strict=True)
    #     print('Loaded weights from', os.path.join(args.dir, 'model_latest.pt'))
    #     # print('debug model weight after', model.m1[0].state_dict())
    #     success = True
    # except:
    #     print('Failed to load', os.path.join(args.dir, 'model_latest.pt'))
    #     success = False
    #     pass
    success=checkpoint.load(model, save_file, optimizer, loss_hist, epoch)
    if nGPU > 1:
        print("Using", nGPU, "GPUs")
        pmodel = torch.nn.DataParallel(model)
        model = pmodel.module
    else:
        pmodel = model
    pmodel.to(device)
    # print('debug model weight before training', model.m1[0].state_dict())
    return pmodel, model, success

def train_model(model, args, loader):
    save_file = os.path.join(args.dir, 'model_latest.tar')
    loss_hist = []
    epoch_start = [0]
    pmodel, model, _ = load_model(model, save_file)#, optimizer, loss_hist, epoch_start)
    optimizer = torch.optim.Adam(pmodel.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1/(epoch/500+1))
    checkpoint.load(None, save_file, optimizer, loss_hist, epoch_start)

    test(pmodel, args, loader.test, args.dir+'/valid')
    for t in range(epoch_start[0], args.epochs):
        # print('debug t=',t)
        loader.train.shuffle()
        for idat, x_np in enumerate(loader.train.sample(args.minibatch_size*nGPU)):
            # x_np = gen_train_dat(r, dim, nx, 4, args.minibatch_size*nGPU)
            x = torch.tensor(x_np, device=device).float()
            # y_pred, omega, scatter_mat= toy_model(x)
            # loss = lossf(y_pred, x[:,1], omega, scatter_mat)
            loss = torch.mean(pmodel(x))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
        if t % args.print_every == 0 or t==args.epochs-1:
            print(t, loss.item())
        if t % args.test_every == 0 or t==args.epochs-1:
            model.visualize(args.dir+'/plots.txt')
            loss_hist.append(test(pmodel, args, loader.test, args.dir+'/valid'))
            checkpoint.save(model, optimizer, loss_hist, t)
            #torch.save({
            #  'model_state_dict':model.state_dict(),
            #  'optimizer_state_dict':optimizer.state_dict(),
            #  'epoch':t,
            #  'loss': loss_hist
            #}, save_file)
            # print('debug toy_model.train', toy_model.train)
            model.train()
    # print('debug model weight before exit', model.m1[0].state_dict())
    return model

def predict(model, args, dat, noise=False):
    path = args.dir+'/predict/'
    save_file = os.path.join(args.dir, 'model_latest.tar')
    pmodel, model, loaded = load_model(model, save_file)
    if not loaded:
        raise ValueError('failed to load '+save_file)
    test(pmodel, args, dat, path, noise)

def test(model, args, dat, path, noise=False):
    except_time=(0,)+tuple(range(2,args.dim+3))
    model.eval()
    # if not os.path.exists(args.file_test): return
    errs=[]
    e_baseline=[]
    if not os.path.exists(path): os.makedirs(path, exist_ok=True)
    for idat, x_np in enumerate(dat.sample(args.minibatch_size*nGPU)):
        x = torch.tensor(x_np, device=device).float()
        x_pd = model(x,'forecast', x.shape[1], noise)
        # print(f'debug x pd {x_pd.shape} {x_pd.__class__}')
        errs.append(torch.mean((x[:,args.input_length:]-x_pd[:,args.input_length:])**2, except_time).detach().cpu().numpy())
        e_baseline.append(torch.mean((x[:,args.input_length:]-x[:,args.input_length-1:args.input_length])**2, except_time).detach().cpu().numpy())
        x_pd = x_pd.detach().cpu().numpy()
        dir=path+'/%d'%idat
        os.makedirs(dir, exist_ok=True)
        np.save(dir+'/gt.npy', x_np)
        # print(f'debug x_np0 min max {np.min(x_np[:,0])} {np.max(x_np[:,0])}')
        # print(f'debug x_np1 min max {np.min(x_np[:,1])} {np.max(x_np[:,1])}')
        # print(f'debug x_pd0 min max {np.min(x_pd[:,0])} {np.max(x_pd[:,0])}')
        # print(f'debug x_pd1 min max {np.min(x_pd[:,1])} {np.max(x_pd[:,1])}')
        np.save(dir+'/pd.npy', x_pd)
    print("error MSE", np.sqrt(np.mean(errs)), np.sqrt(np.mean(e_baseline)))
    print('err per frame', np.sqrt(np.mean(errs,0)), np.sqrt(np.mean(e_baseline,0)))
    return np.sqrt(np.mean(errs,0))


if __name__ == "__main__":
    if args.test_only:
        predict(model, args, loader.test, args.noise)
    else:
        train_model(model, args, loader)
    # torch.save(model.state_dict(), os.path.join(args.dir, 'model_latest.pt'))
