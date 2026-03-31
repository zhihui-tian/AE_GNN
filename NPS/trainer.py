import os, sys, time
import math
# from decimal import Decimal

from . import utility

import torch
# from torch.autograd import Variable
#from tqdm import tqdm
import numpy as np
from NPS_common.utils import a1line
from .data.data_augment import data_augment_operator
# from codecarbon import track_emissions
# from codecarbon import EmissionsTracker

def make_trainer(args, loader, model, loss, checkpoint):
    return Trainer(args, loader, model, loss, checkpoint)

def memory_diagnostics(verbose=False, reset_stats=True, suffix=''):
    ndevices=torch.cuda.device_count()
    for d in range(ndevices):
        if verbose:
            dname=torch.cuda.get_device_name(d)
            dprop=torch.cuda.get_device_properties(d)
            a=torch.cuda.max_memory_allocated(d)
            r=torch.cuda.max_memory_reserved(d)
            t=dprop.total_memory
            print('Device: {} = {:.3f}/{:.3f} GB. Total = {:.3f} GB {}'.format(dname, a/1e9, r/1e9, t/1e9, suffix))
        if reset_stats: torch.cuda.reset_peak_memory_stats(device=d)

class Trainer():
    def __init__(self, args, loader, model, loss, ckp):
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        self.args = args
        self.dim = args.dim

        self.ckp = ckp
        self.loader_train = loader.train
        self.loader_test = loader.test
        self.model = model
        self.loss = loss
        self.augment_op = data_augment_operator(args)
        self.optimizer = utility.make_optimizer(args, self.model) if args.mode == 'train' else None
        self.scheduler = utility.make_scheduler(args, self.optimizer) if args.mode == 'train' else None
        # if self.scheduler is not None: self.scheduler.last_epoch = len(self.loss.log)+1
        # self.epoch_start = len(self.loss.log)+1 if self.loss is not None else 0
        self.epoch_start = 0
        self.device = torch.device('cpu' if self.args.cpu else 'cuda')
        # self.nsample_train = args.batch*args.n_GPUs
        self.loss_valid_log, self.loss_train_log = [], []
        self.loss_valid_min = np.inf
        self.ich = 1 if args.channel_first else -1

        misc_file =  os.path.join(ckp.dir, 'opt.pt')
        if self.optimizer is not None and os.path.exists(misc_file): #self.args.load != '.':
            try:
                print(f'Loading optimizer from {misc_file}')
                ckpt = torch.load(misc_file)
                self.optimizer.load_state_dict(ckpt['optimizer_state'])
                # print(f'epoch st {self.epoch_start} log {self.loss.log}')
                self.scheduler.load_state_dict(ckpt['scheduler_state'])
                self.epoch_start = ckpt['epoch']
                print(f'Restored optimizer, scheduler, epoch')
                self.loss_valid_log, self.loss_train_log = torch.load(os.path.join(ckp.dir, 'loss_log.pt'))
                print(f'Restored loss log (train/valid)')
                if len(self.loss_valid_log): self.loss_valid_min = np.min(self.loss_valid_log)
            except:
                print(f'Failed to load optimizer from {misc_file}')
            # if args.scheduler != 'plateau':
            #     for _ in range(len(ckp.log)): self.scheduler.step()

        self.ckp.write_log('Commandline ' + ' '.join(sys.argv))
        self.train_generator = self.get_training_batch()
        if args.epoch_size == -1: args.epoch_size = len(self.loader_train)
        # if args.n_traj_out == -1: args.n_traj_out = len(self.loader_test) * args.batch 

    def get_training_batch(self):
        while True:
            for batch in self.loader_train:
                yield(self.augment_op(batch))
                # if self.pointgroup.nops > 1:
                #     batch = self.pointgroup.random_op()(batch)
                # yield batch

    # def get_arg_value(name, default=None):
    #     for arg in sys.argv:
    #         if arg.startswith(f"--{name}="):
    #             return arg.split("=", 1)[1]
    #     return default
    # gnnmodel = get_arg_value("gnnmodel", default="unknown")
    # nstrides = get_arg_value("nstrides_2wae", default="NA")
    # project_name = f"{gnnmodel}_{nstrides}"

    # @track_emissions(country_iso_code="USA", project_name=args.gnnmodel)

    def train(self):
        # while not self.terminate():
        #     self.train_epoch()
        #     self.test()
        project_name = f"{self.args.gnnmodel}_{self.args.nstrides_2wae}".replace('/', '_').replace(',', '-')
        # tracker = EmissionsTracker(
        #     project_name=project_name
        # )
        # tracker.start()

        for epoch in range(self.epoch_start+1, self.args.nepoch+1):
            t0 = time.time()
            loss_epoch = 0
            loss_epoch_item = []
            self.model.train()
            print(f'Epoch {epoch}')
            #for pg in self.scheduler.optimizer.param_groups:
            #    pg['lr']=1e-3
            print('Epoch {}: lr: {}'.format(epoch, *[pg['lr'] for pg in self.scheduler.optimizer.param_groups]))
            for i in range(1, self.args.epoch_size+1):
                x = next(self.train_generator).to(self.device)

                # print(f'debug x in train {x.shape} epo {epoch} {i} opt.epoch_size {opt.epoch_size} opt.niter {opt.niter}')
                # print(f'debug x in train {x.shape} epo {epoch} {i}')
                if i ==1:
                    memory_diagnostics(verbose=True, suffix='After Dataloader Load')   ##
                    
                loss = self.train_batch(x, epoch=epoch)
                # single loss, or (loss, (loss_items))
                if isinstance(loss, tuple):
                    if len(loss[1]): loss_epoch_item.append(loss[1])
                    loss = loss[0]
                # print(f'debug got loss {loss}')
                loss_epoch += loss
                if i % self.args.print_freq == 0:
                    print(f'  {i} loss_batch {loss:7.3e} {a1line(loss_epoch_item[-1]) if loss_epoch_item else ""} averaged {loss_epoch/i:7.3e}')
                    memory_diagnostics(verbose=self.args.log_mem_loss, reset_stats=False, suffix='Loss')
            loss_epoch /= self.args.epoch_size
            if loss_epoch_item: loss_epoch_item = np.mean(np.array(loss_epoch_item), 0)
            memory_diagnostics(verbose=self.args.log_mem_train, reset_stats=(self.args.log_mem_train or self.args.log_mem_valid), suffix='Train')
            tmp = time.time(); t_train = tmp-t0; t0 = tmp
            with torch.no_grad():
                self.model.eval()
                loss_valid = self.evaluate(False, epoch=epoch)
                print(f'{epoch} {time.strftime("%H:%M:%S", time.localtime())} Train_loss: {loss_epoch:7.3e} {a1line(loss_epoch_item)} Valid_loss,mse,mae: {a1line(loss_valid)} train/test_time {t_train:6.2e} {time.time()-t0:6.2e}')
            memory_diagnostics(verbose=self.args.log_mem_valid, reset_stats=(self.args.log_mem_train or self.args.log_mem_valid), suffix='Valid')

            self.loss_train_log.append(loss_epoch)
            self.loss_valid_log.append(loss_valid[0])
            is_best = False
            if self.loss_valid_min > loss_valid[0]:
                self.loss_valid_min = loss_valid[0]
                is_best = True
            if self.args.scheduler == 'plateau':
                self.scheduler.step(loss_valid[0])
            else:
                self.scheduler.step()

            # save the model
            if True:
                self.ckp.save(self, epoch, is_best=is_best, loss_log=(self.loss_train_log, self.loss_valid_log))
            # if epoch % 1 == 0:
                # torch.save({
                #     'model': model,
                #     'optimizer': optimizer,
                #     'opt': opt},
                #     '%/model.pt' % (opt.log_dir))
        # tracker.stop()

    def evaluate(self, predict_only, epoch=0):
        # t0 = time.time()
        args = self.args
        n_in  = args.n_in_test
        n_out = args.n_out_test
        ngram = args.ngram
        mse_detail = []
        mae_detail = []
        losses = []
        loss_item = None
        self.model.eval()
        n_pd = 0
        pd_all = []
        gt_all = []
        sp = 3 if args.channel_first else 2
        # print(f'debug>> predict only is {predict_only}, n_pd is {n_pd} and args.n_traj_out is{args.n_traj_out}')
        # print(f'len(loader_train): {len(self.loader_train)}')
        print(f'type(loader_train): {type(self.loader_train)}')
        print(f'len(loader_test): {len(self.loader_test)}')

        # t0 = time.time()
        with torch.no_grad():

            x_init = next(iter(self.loader_test))   ### """WARM UP!"
            pd = self.evaluate_batch(x_init.to(self.device), predict_only)
            t0 = time.time()

            for i, x in enumerate(self.loader_test):            # for training or inference rollout
            # x = next(iter(self.loader_test))                      # for statistical plot, get variance of result from same initial condition
            # for i in range(6):

                print(f'loader_test loop times{i}')
                print(f'x shape is {x.shape}')
                if predict_only and (n_pd >= args.n_traj_out):
                    break
                # pd = self.evaluate_batch(x.to(self.device)).detach().cpu()

                if args.predict_ff and predict_only:
                #Feed forward prediction to reduce memory
                    #for s in range(min(x.shape[0], args.n_traj_out)):
                        #print('Predicting from sample {}'.format(s))
                        #print('Loaded:', x.shape, s, ngram, n_in)
                        #xi=x[s:s+1,n_in-ngram:n_in]
                        xi=x[:,n_in-ngram:n_in]
                        
                        gt_all.append(xi)
                        jprev=0
                        for j in range(0,self.args.n_out_predict,self.args.n_out_test):
                            # print(f'[Variable]: args.n_out_predict:{self.args.n_out_predict}')
                            # print(f'[Variable]: args.n_out_test:{self.args.n_out_test}')
                            xp = self.evaluate_batch(xi.to(self.device), predict_only)
                            print(f'time for {i} batch:{time.time()-t0:7.3e}')
                            # print(f'[Variable]:xp:{xp.shape}')
                            if isinstance(xp, tuple):
                                loss_item = xp[1]
                                xp = xp[0]
                            xp = xp.detach().cpu()
                            xi=torch.cat([xi, xp], dim=1)
                            xi = xi[:,-(n_in - ngram + 1):]
                            if self.args.clip_step_predict < self.args.n_out_valid:
                                jpinds = np.floor(np.arange(0, self.args.n_out_test, self.args.clip_step_predict)).astype(int)
                            else:
                                if j % self.args.clip_step_predict < self.args.n_out_test:
                                    jpinds = [j%self.args.clip_step_predict]
                                else:
                                    jpinds = []
                            for jp in jpinds:
                                #print('{}) Predicted {}->{}, Appending {}'.format(i, j, j+self.args.n_out_test, j+jp))
                                xs = xp[:,jp,:]
                                xs = xs[:, None, :]
                                if j == 0 and jp == 0: 
                                    pd = xs 
                                    print('Setting new pd', pd.shape)
                                else: 
                                    pd = torch.cat([pd, xs], dim = 1)
                                    #print('Appending to pd', pd.shape)
                                #print(xp.shape, xs.shape, pd.shape)
                            memory_diagnostics(verbose=True, reset_stats=True, suffix='Predict')
                            jprev = j 
                        #if s==0: 
                        #    pd = pds 
                        #    print('Setting new pd', pd.shape)
                        #else: 
                        #    pd = torch.cat([pd, pds], dim = 0)
                        #    print('Appending to pd', pd.shape)

                else:
                      pd = self.evaluate_batch(x.to(self.device), predict_only)
                      #   print('[Debug]: actually run else branch')
                      print(f'[Variable]: shape of x is : {x.shape}')
                      print(f'[Variable]: shape of pd is : {pd.shape}')
                      
                      print(f'time for {i} batch:{time.time()-t0:7.3e}')
                      if isinstance(pd, tuple):
                          loss_item = pd[1]
                          pd = pd[0]

                    #   memory_diagnostics(verbose=True, reset_stats=True, suffix='Predict')  ###
                      pd = pd.detach().cpu()

                      if args.mode == 'predict':
                        x0 = x[:, 0:1, ...]             ### only in inference
                        pd = torch.cat([x0, pd], dim=1) 
                        print(f'model mode is {args.mode}')   

                      
                n_pd += len(pd)
                pd_all.append(pd)

                if not predict_only:
                    gt = x[:, n_in:n_in+n_out]
                    gt_all.append(gt)
                    mse_detail.append(torch.mean((pd-gt)**2 , axis=tuple(range(sp,sp+args.dim))))
                    mae_detail.append(torch.mean(torch.abs(pd-gt) , axis=tuple(range(sp,sp+args.dim))))
                    losses.append([self.loss(pd, gt)] if (loss_item is None) or (loss_item==[]) else loss_item)

        # print(f'debug pd {pd_all[-1].shape} {len(pd_all)}')
        print(f'Predicted data time before cat {time.time()-t0:7.3e}')
        pd_all = torch.cat(pd_all)
        pd_all_size = pd_all.shape
        if args.n_traj_out > 0:
            pd_all = pd_all[:args.n_traj_out]
        print(f'Predicted data of size {pd_all_size} time {time.time()-t0:7.3e}')

        if args.infer_mode =='original':
            args.pd_file = f'{self.args.dir}/pd.npy'
        else:
            args.pd_file = f'{self.args.dir}/pd_latent.npy'
        
        print('Saving PD file to {}'.format(self.args.pd_file))
        
        np.save(f'{self.args.pd_file}', utility.to_channellast(pd_all, self.dim) if self.args.channel_first else pd_all)
        gt_all = torch.cat(gt_all)
        if args.n_traj_out > 0:
            gt_all = gt_all[:args.n_traj_out]
        print('Saving GT file to {}'.format(self.args.gt_file))
        np.save(f'{self.args.gt_file}', utility.to_channellast(gt_all, self.dim) if self.args.channel_first else gt_all)
        if predict_only:
            print(f'Predicted data of size {pd_all_size} time {time.time()-t0:7.3e}')
            return
        else:
            mse_detail = np.concatenate(mse_detail, 0)
            mae_detail = np.concatenate(mae_detail, 0)
            mse, mae = np.mean(mse_detail), np.mean(mae_detail)
            print(f'valid per time/channel/seq: mse {a1line(np.mean(mse_detail,axis=(0,2)))} / {a1line(np.mean(mse_detail,axis=(0,1)))} / {a1line(np.mean(mse_detail,axis=(1,2)))}')
            print(f'valid per time/channel/seq: mae {a1line(np.mean(mae_detail,axis=(0,2)))} / {a1line(np.mean(mae_detail,axis=(0,1)))} / {a1line(np.mean(mae_detail,axis=(1,2)))}')
            try:
                self.model.get_model().analyze(pd_all.reshape(-1, pd_all.shape[-1]), gt_all.reshape(-1, gt_all.shape[-1]))
            except:
                pass
            return tuple(np.mean(losses, 0)) + (mse, mae)

    def validate(self):
        return self.evaluate(False)

    def predict(self):
        return self.evaluate(True)

    def model_y_loss(self, x_in, target, loss_from_model=False, reset=True):
        """The model may either directly return a prediction, or prediction, loss and itemized losses"""
        y = self.model(x_in, reset=reset, target=target if loss_from_model else None, criterion=self.loss, mask=None)
        if target is None:
            return y, 0, []
        if isinstance(y, tuple):
            y, loss_step_item = y
            loss_step = self.step_loss(loss_step_item)
            loss_step_item = [x.item() for x in loss_step_item]
        else:
            loss_step = self.loss(y, target)
            loss_step_item = []
        return y, loss_step, loss_step_item

    def step_loss(self, loss_step_item):
        if self.args.loss_wt:
            return sum([loss_step_item[i]*self.args.loss_wt[i] for i in range(len(loss_step_item))])
        else:
            return sum(loss_step_item)

    def train_batch(self, x, epoch=1):
        args = self.args
        n_in  = args.n_in
        n_out = args.n_out
        RNN = args.RNN
        loss = 0
        loss_item = []
        self.optimizer.zero_grad()
        for ei in range(n_in-1):
            target = x[:, ei+1]
            tgt = target[:, :self.args.nfeat_out] if args.channel_first else target[..., :self.args.nfeat_out]
            y, step_loss, step_loss_item = self.model_y_loss(x[:,ei], tgt, reset=(ei==0), loss_from_model=args.loss_from_model)
            if step_loss_item: loss_item.append(step_loss_item)
            loss += step_loss
        x_in = x[:,n_in-1]
        for di in range(n_out):
            target = x[:,n_in+di]
            tgt = target[:, :self.args.nfeat_out] if args.channel_first else target[..., :self.args.nfeat_out]
            y, step_loss, step_loss_item = self.model_y_loss(x_in, tgt, reset=(not RNN) or ((n_in==1) and (di==0)), loss_from_model=args.loss_from_model)
            if step_loss_item: loss_item.append(step_loss_item)
            loss += step_loss
            if args.nfeat_in>args.nfeat_out: y = torch.cat([y, target[:, args.nfeat_out:] if args.channel_first else target[..., args.nfeat_out:]], self.ich)
            if di < n_out-1: x_in = self.get_scheduled_input(y, target, epoch)
        loss.backward()
        self.optimizer.step()
        return loss.item() / n_out, np.mean(loss_item, 0) if loss_item else []

    def evaluate_batch(self, x, predict_only=False):
        args = self.args
        n_in  = args.n_in_test
        n_out = args.n_out_test
        RNN = self.args.RNN
        traj = []
        loss = 0
        loss_item = []
        for ei in range(n_in-1):
            target = None if predict_only else x[:, ei+1]
            tgt = None if predict_only else (target[:, :self.args.nfeat_out] if args.channel_first else target[..., :self.args.nfeat_out])
            y, step_loss, step_loss_item = self.model_y_loss(x[:,ei], tgt, reset=(ei==0), loss_from_model=args.loss_from_model)
            if not predict_only:
                if step_loss_item: loss_item.append(step_loss_item)
                loss += step_loss
        x_in = x[:,n_in-1]
        for di in range(n_out):
            target = None if predict_only else x[:,n_in+di]
            tgt = None if predict_only else (target[:, :self.args.nfeat_out] if args.channel_first else target[..., :self.args.nfeat_out])
            y, step_loss, step_loss_item = self.model_y_loss(x_in, tgt, reset=(not RNN) or ((n_in==1) and (di==0)), loss_from_model=args.loss_from_model)
            if not predict_only:
                if step_loss_item: loss_item.append(step_loss_item)
                loss += step_loss
            if args.nfeat_in>args.nfeat_out: y = torch.cat([y, target[:, args.nfeat_out:] if args.channel_first else target[..., args.nfeat_out:]], self.ich)
            traj.append(y.detach())
            x_in = y
        return torch.stack(traj, 1), np.mean(loss_item, 0) if loss_item else []

    def get_scheduled_input(self, y, target, epoch):
        flag = self.get_real_input_flag(epoch)
        if flag == 1:
            x_in = target
        elif flag == 0:
            x_in = y
        else:
            "TBD"
        return x_in

    def get_real_input_flag(self, itr=1):
        args = self.args
        if args.rnn_scheduled_sampling == 'reverse':
            real_input_flag = reserve_schedule_sampling_exp(itr, args)
        elif args.rnn_scheduled_sampling == 'decrease':
            real_input_flag = schedule_sampling(itr, args)
        elif args.rnn_scheduled_sampling == 'GT':
            real_input_flag = 1
        elif args.rnn_scheduled_sampling == 'PD':
            real_input_flag = 0
        else:
            raise ValueError(f'unknown rnn_scheduled_sampling {args.rnn_scheduled_sampling}')
        return real_input_flag



def reserve_schedule_sampling_exp(itr, args):
    if itr < args.r_sampling_step_1:
        r_eta = 0.5
    elif itr < args.r_sampling_step_2:
        r_eta = 1.0 - 0.5 * math.exp(-float(itr - args.r_sampling_step_1) / args.r_exp_alpha)
    else:
        r_eta = 1.0

    if itr < args.r_sampling_step_1:
        eta = 0.5
    elif itr < args.r_sampling_step_2:
        eta = 0.5 - (0.5 / (args.r_sampling_step_2 - args.r_sampling_step_1)) * (itr - args.r_sampling_step_1)
    else:
        eta = 0.0

    r_random_flip = np.random.random_sample(
        (args.batch, args.n_in - 1))
    r_true_token = (r_random_flip < r_eta)

    random_flip = np.random.random_sample(
        (args.batch, args.n_out - 1))
    true_token = (random_flip < eta)

    ones = np.ones(  (*[x // args.patch_size for x in args.frame_shape],
                    args.patch_size ** args.dim * args.nfeat_in))
    zeros = np.zeros((*[x // args.patch_size for x in args.frame_shape],
                    args.patch_size ** args.dim * args.nfeat_in))

    real_input_flag = []
    for i in range(args.batch):
        for j in range(args.n_in + args.n_out - 2):
            if j < args.n_in - 1:
                if r_true_token[i, j]:
                    real_input_flag.append(ones)
                else:
                    real_input_flag.append(zeros)
            else:
                if true_token[i, j - (args.n_in - 1)]:
                    real_input_flag.append(ones)
                else:
                    real_input_flag.append(zeros)

    real_input_flag = np.array(real_input_flag)
    real_input_flag = np.reshape(real_input_flag,
                                 (args.batch,
                                  args.n_in + args.n_out - 2,
                                  *[x // args.patch_size for x in args.frame_shape],
                                  args.patch_size ** 2 * args.nfeat_in))
    return real_input_flag


def schedule_sampling(itr, args):
    zeros = np.zeros((args.batch,
                      args.n_out - 1,
                      *[x // args.patch_size for x in args.frame_shape],
                      args.patch_size ** 2 * args.nfeat_in))
    if not args.scheduled_sampling:
        return 0.0, zeros

    eta = np.max(0, args.sampling_start_value - itr*args.sampling_changing_rate)
    random_flip = np.random.random_sample(
        (args.batch, args.n_out - 1))
    true_token = (random_flip < eta)
    ones = np.ones((*[x // args.patch_size for x in args.frame_shape],
                    args.patch_size ** args.dim * args.nfeat_in))
    zeros = np.zeros((*[x // args.patch_size for x in args.frame_shape],
                      args.patch_size ** args.dim * args.nfeat_in))
    real_input_flag = []
    for i in range(args.batch):
        for j in range(args.n_out - 1):
            if true_token[i, j]:
                real_input_flag.append(ones)
            else:
                real_input_flag.append(zeros)
    real_input_flag = np.array(real_input_flag)
    real_input_flag = np.reshape(real_input_flag,
                                 (args.batch,
                                  args.n_out - 1,
                                  *[x // args.patch_size for x in args.frame_shape],
                                  args.patch_size ** 2 * args.nfeat_in))
    return real_input_flag



# import os, sys, time
# import math
# # from decimal import Decimal

# from . import utility

# import torch
# # from torch.autograd import Variable
# #from tqdm import tqdm
# import numpy as np
# from .data.pointgroup import PointGroup
# from .data.noise_operator import noise_operator
# from NPS_common.utils import a1line

# from torch.profiler import profile, record_function, ProfilerActivity

# def make_trainer(args, loader, model, loss, checkpoint):
#     return Trainer(args, loader, model, loss, checkpoint)

# def memory_diagnostics():
#     ndevices=torch.cuda.device_count()
#     for d in range(ndevices):
#         dname=torch.cuda.get_device_name(d)
#         dprop=torch.cuda.get_device_properties(d)
#         #memusage=torch.cuda.memory_usage(d)
#         a=torch.cuda.max_memory_allocated(d)
#         r=torch.cuda.max_memory_reserved(d)
#         t=dprop.total_memory
#         print('Device: {} = {:.2f}/{:.2f} GB. Total = {:.2f} GB'.format(dname, a/1e9, r/1e9, t/1e9))
#         #print(torch.cuda.memory_summary())

# class Trainer():
#     def __init__(self, args, loader, model, loss, ckp):
#         print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
#         self.args = args
#         self.dim = args.dim

#         self.ckp = ckp
#         self.loader_train = loader.train
#         self.loader_test = loader.test
#         self.model = model
#         self.loss = loss
#         self.pointgroup = PointGroup(args.pointgroup, self.dim, args.channel_first)
#         self.noise_op = noise_operator(args.noise_op)
#         self.optimizer = utility.make_optimizer(args, self.model) if args.mode == 'train' else None
#         self.scheduler = utility.make_scheduler(args, self.optimizer) if args.mode == 'train' else None
#         # if self.scheduler is not None: self.scheduler.last_epoch = len(self.loss.log)+1
#         # self.epoch_start = len(self.loss.log)+1 if self.loss is not None else 0
#         self.epoch_start = 0
#         self.device = torch.device('cpu' if self.args.cpu else 'cuda')
#         # self.nsample_train = args.batch*args.n_GPUs
#         self.loss_valid_log, self.loss_train_log = [], []
#         self.loss_valid_min = np.inf

#         misc_file =  os.path.join(ckp.dir, 'opt.pt')
#         if self.optimizer is not None and os.path.exists(misc_file): #self.args.load != '.':
#             try:
#                 print(f'Loading optimizer from {misc_file}')
#                 ckpt = torch.load(misc_file)
#                 self.optimizer.load_state_dict(ckpt['optimizer_state'])
#                 # print(f'epoch st {self.epoch_start} log {self.loss.log}')
#                 self.scheduler.load_state_dict(ckpt['scheduler_state'])
#                 self.epoch_start = ckpt['epoch']
#                 print(f'Restored optimizer, scheduler, epoch')
#                 self.loss_valid_log, self.loss_train_log = torch.load(os.path.join(ckp.dir, 'loss_log.pt'))
#                 print(f'Restored loss log (train/valid)')
#                 if len(self.loss_valid_log): self.loss_valid_min = np.min(self.loss_valid_log)
#             except:
#                 print(f'Failed to load optimizer from {misc_file}')
#             # if args.scheduler != 'plateau':
#             #     for _ in range(len(ckp.log)): self.scheduler.step()

#         self.ckp.write_log('Commandline ' + ' '.join(sys.argv))
#         self.train_generator = self.get_training_batch()
#         if args.epoch_size == -1: args.epoch_size = len(self.loader_train)
#         if args.n_traj_out == -1: args.n_traj_out = len(self.loader_test) * args.batch 

#     def get_training_batch(self):
#         while True:
#             for batch in self.loader_train:
#                 if self.pointgroup.nops > 1:
#                     batch = self.pointgroup.random_op()(batch)
#                 yield batch

#     def train(self):
#         # while not self.terminate():
#         #     self.train_epoch()
#         #     self.test()
#         print('STARTING TRAINING')
#         print(self.epoch_start+1, self.args.nepoch, list(range(self.epoch_start+1, self.args.nepoch+1)))
#         for epoch in range(self.epoch_start+1, self.args.nepoch+1):
#             print(f'STARTING EPOCH {epoch}')
#             t0 = time.time()
#             loss_epoch = 0
#             #with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True, record_shapes=True) as prof:		
#             self.model.train()
#             #print('Done training?', self.args.epoch_size)
#             for i in range(1, self.args.epoch_size+1):
#                 #print('hi', i, self.train_generator, self.device)
#                 #Might need to get some debug tools
#                 #print(self.loader_train)
#                 #print(len(self.loader_train))
#                 #for s in self.loader_train: break #Terrible replacement for next 
#                 #print(s.shape)
#                 #x=s.to(self.device)
#                 #print(x)
#                 #print(x.shape)
#                 #print(next(self.loader_train))
#                 #x = next(self.loader_train).to(self.device)
#                 x = next(self.train_generator).to(self.device)
#                 #print(f'debug x in train {x.shape} epo {epoch} {i} opt.epoch_size {opt.epoch_size} opt.niter {opt.niter}')
#                 loss = self.train_batch(x, criterion=self.loss, epoch=epoch)
#                 #print(f'debug got loss {loss}')
#                 loss_epoch += loss
#                 if i % self.args.print_freq == 0:
#                     print(f'  {i} loss_batch {loss:7.3e} averaged {loss_epoch/i:7.3e}')
#             loss_epoch /= self.args.epoch_size
#             tmp = time.time(); t_train = tmp-t0; t0 = tmp
#             with torch.no_grad():
#                 memory_diagnostics()
#                 self.model.eval()
#                 loss_valid = self.evaluate(False, epoch=epoch)
#                 print(f'{epoch} {time.strftime("%H:%M:%S", time.localtime())} Train_loss: {loss_epoch:7.3e} Valid_loss,mse,mae: {a1line(loss_valid)} train/test_time {t_train:6.2e} {time.time()-t0:6.2e}')
#             self.loss_train_log.append(loss_epoch)
#             self.loss_valid_log.append(loss_valid[0])
#             is_best = False
#             if self.loss_valid_min > loss_valid[0]:
#                 self.loss_valid_min = loss_valid[0]
#                 is_best = True
#             if self.args.scheduler == 'plateau':
#                 self.scheduler.step(loss_valid[0])
#             else:
#                 self.scheduler.step()

#             # save the model
#             if True:
#                 self.ckp.save(self, epoch, is_best=is_best, loss_log=(self.loss_train_log, self.loss_valid_log))
#             # if epoch % 1 == 0:
#                 # torch.save({
#                 #     'model': model,
#                 #     'optimizer': optimizer,
#                 #     'opt': opt},
#                 #     '%/model.pt' % (opt.log_dir))

#     def evaluate(self, predict_only, epoch=0):
#         t0 = time.time()
#         args = self.args
#         n_in  = args.n_in_test
#         n_out = args.n_out_test
#         mse_detail = []
#         mae_detail = []
#         losses = []
#         self.model.eval()
#         n_pd = 0
#         pd_all = []
#         gt_all = []
#         sp = 3 if args.channel_first else 2
#         with torch.no_grad():
#             #print('loader_test', len(self.loader_test), list(self.loader_test))
#             for i, x in enumerate(self.loader_test):
#                 if predict_only and (n_pd >= args.n_traj_out):
#                     break
#                 #print('n_pd', n_pd)
#                 pd = self.evaluate_batch(x.to(self.device)).detach().cpu()
#                 n_pd += len(pd)
#                 pd_all.append(pd)
#                 if not predict_only:
#                     gt = x[:, n_in:n_in+n_out]
#                     gt_all.append(gt)
#                     mse_detail.append(torch.mean((pd-gt)**2 , axis=tuple(range(sp,sp+args.dim))))
#                     mae_detail.append(torch.mean(torch.abs(pd-gt) , axis=tuple(range(sp,sp+args.dim))))
#                     losses.append(self.loss(pd, gt))

#         # print(f'debug pd {pd_all[-1].shape} {len(pd_all)}')
#         pd_all = torch.cat(pd_all)
#         pd_all_size = pd_all.shape
#         if args.n_traj_out > 0:
#             pd_all = pd_all[:args.n_traj_out]
#         if predict_only:
#             pd_all= pd_all[:,range(0, pd_all_size[1], args.clip_step_predict),...]
#         if isinstance(args.pd_file, type(None)): args.pd_file = f'{self.args.dir}/pd.npy'
#         np.save(args.pd_file, utility.to_channellast(pd_all, self.dim) if self.args.channel_first else pd_all)


#         if predict_only:
#             print(f'Predicted data of size {pd_all_size} time {time.time()-t0:7.3e}')
#             print(f'Saving predictions of size {pd_all.shape} to {args.pd_file}')
#             return
#         else:
#             gt_all = torch.cat(gt_all)
#             if args.n_traj_out > 0:
#                 gt_all = gt_all[:args.n_traj_out]
#             if isinstance(args.gt_file, type(None)): args.gt_file = f'{self.args.dir}/gt.npy'
#             np.save(args.gt_file, utility.to_channellast(gt_all, self.dim) if self.args.channel_first else gt_all)
#             mse_detail = np.concatenate(mse_detail, 0)
#             mae_detail = np.concatenate(mae_detail, 0)
#             mse, mae = np.mean(mse_detail), np.mean(mae_detail)
#             print(f'valid per time/channel/seq: mse {a1line(np.mean(mse_detail,axis=(0,2)))} / {a1line(np.mean(mse_detail,axis=(0,1)))} / {a1line(np.mean(mse_detail,axis=(1,2)))}')
#             print(f'valid per time/channel/seq: mae {a1line(np.mean(mae_detail,axis=(0,2)))} / {a1line(np.mean(mae_detail,axis=(0,1)))} / {a1line(np.mean(mae_detail,axis=(1,2)))}')
#             return np.mean(losses), mse, mae

#     def validate(self):
#         return self.evaluate(False)

#     def predict(self):
#         return self.evaluate(True)

#     def train_batch(self, x, criterion=None, epoch=1):
#         args = self.args
#         n_in  = args.n_in
#         n_out = args.n_out
#         RNN = args.RNN
#         loss = 0
#         self.optimizer.zero_grad()
#         for ei in range(n_in-1):
#             y = self.model(x[:,ei], reset=(ei==0) )
#             loss += self.loss(y, x[:,ei+1,:,:,:])
#         use_teacher_forcing = True # if random.random() < teacher_forcing_ratio else False
#         x_in = x[:,n_in-1]
#         for di in range(n_out):
#             y = self.model(x_in, reset=(not RNN) or ((n_in==1) and (di==0)))
#             target = x[:,n_in+di,:,:,:]
#             loss += criterion(y, target)
#             if use_teacher_forcing:
#                 x_in = target # Teacher forcing
#             else:
#                 x_in = y
#         loss.backward()
#         self.optimizer.step()
#         return loss.item() / n_out

#     def evaluate_batch(self, x):
#         args = self.args
#         n_in  = args.n_in_test
#         n_out = args.n_out_test
#         RNN = self.args.RNN
#         traj = []
#         for ei in range(n_in-1):
#             y = self.model(x[:,ei], reset=(ei==0))
#         x_in = x[:,n_in-1]
#         for di in range(n_out):
#             y = self.model(x_in, reset=(not RNN) or ((n_in==1) and (di==0)))
#             traj.append(y.detach())
#             x_in = y
#         return torch.stack(traj, 1)


