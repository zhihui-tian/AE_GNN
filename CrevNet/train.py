import scipy.signal
import torch
import torch.optim as optim
import torch.nn as nn
import argparse
import os
import random
from torch.autograd import Variable
from torch.utils.data import DataLoader
# from . import utils
from . import data_utils
import numpy as np
# from tqdm import trange

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='train')
parser.add_argument('--dim', type=int, default=2, help='dimension')
parser.add_argument('--periodic', action='store_true', help='periodic boundary condition')
parser.add_argument('--noise', type=float, default=0, help='noise')
parser.add_argument('--n_rollout', type=int, default=10, help='no. of rollouts')
parser.add_argument('--n_predict', type=int, default=-1, help='no. of predicted frames')
# parser.add_argument('--n_in', type=int, default=10, help='no. of input frames')
# parser.add_argument('--n_out', type=int, default=10, help='no. of output frames')
parser.add_argument('--n_out_test', type=int, default=-1, help='n_out for test_loader')
parser.add_argument('--clip_step', type=int, default=1, help='steps between adjacent starting positions in a long sequence')
parser.add_argument('--clip_step_test', type=int, default=-1, help='clip_step for test_loader')

parser.add_argument('--lr', default=0.0002, type=float, help='learning rate')
parser.add_argument('--beta1', default=0.9, type=float, help='momentum term for adam')
parser.add_argument('--batch_size', default=1, type=int, help='batch size')
parser.add_argument('--log_dir', default='logs', help='base directory to save logs')
parser.add_argument('--model_dir', default='', help='base directory to save logs')
parser.add_argument('--name', default='', help='identifier for directory')
parser.add_argument('--data', default='', help='root directory for data')
parser.add_argument('--optimizer', default='adam', help='optimizer to train with')
parser.add_argument('--niter', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--seed', default=1, type=int, help='manual seed')
parser.add_argument('--epoch_size', type=int, default=2000, help='epoch size')
parser.add_argument('--image_width', type=int, default=496, help='the height / width of the input image to network')
parser.add_argument('--image_height', type=int, default=448, help='the height / width of the input image to network')
parser.add_argument('--frame_shape', default='64,64', help='the height / width of the input image to network')
parser.add_argument('--channels', default=4, type=int)
parser.add_argument('--dataset', default='longclip', help='dataset to train with')
parser.add_argument('--n_past', type=int, default=5, help='number of frames to condition on')
parser.add_argument('--n_future', type=int, default=3, help='number of frames to predict')
# parser.add_argument('--n_eval', type=int, default=8, help='number of frames to predict at eval time')
parser.add_argument('--rnn_size', type=int, default=512, help='dimensionality of hidden layer')
# parser.add_argument('--posterior_rnn_layers', type=int, default=2, help='number of layers')
parser.add_argument('--predictor_rnn_layers', type=int, default=6, help='number of layers')
# parser.add_argument('--gap', type=int, default=1, help='number of timesteps')
# parser.add_argument('--z_dim', type=int, default=512, help='dimensionality of z_t')
parser.add_argument('--g_dim', type=int, default=512,
                    help='dimensionality of encoder output vector and decoder input vector')
parser.add_argument('--beta', type=float, default=0.0001, help='weighting on KL to prior')
parser.add_argument('--data_threads', type=int, default=0, help='number of data loading threads')
parser.add_argument('--loss', default='L2', help='L2 or L1')
parser.add_argument('--RNN', type=int, default=1, help='1 = recurrent (default), 0=feedforward (empty memory)')

opt = parser.parse_args()
opt.frame_shape = list(map(int, filter(bool, opt.frame_shape.split(','))))
args = opt
args.n_in = args.n_past
args.n_out = args.n_future




if opt.model_dir != '':
    try:
        saved_model = torch.load('%s/model.pth' % opt.model_dir)
        # optimizer = opt.optimizer
        # model_dir = opt.model_dir
        # opt = saved_model['opt']
        # opt.optimizer = optimizer
        # opt.model_dir = model_dir
    except:
        pass
    # opt.log_dir = '%s/continued' % opt.log_dir
    opt.log_dir = opt.model_dir
else:
    name = f'model_city_trial={opt.frame_shape[0]}x{opt.frame_shape[1]}-rnn_size={opt.rnn_size}-predictor-posterior-rnn_layers={opt.predictor_rnn_layers}-{opt.posterior_rnn_layers}'+\
     f'-n_past={opt.n_past}-n_future={opt.n_future}-lr={opt.lr:.7f}-g_dim={opt.g_dim}-z_dim={opt.z_dim}-beta={opt.beta:.7f}{opt.name}'
    if opt.dataset == 'smmnist':
        opt.log_dir = '%s/%s-%d/%s' % (opt.log_dir, opt.dataset, opt.num_digits, name)
    else:
        opt.log_dir = '%s/%s/%s' % (opt.log_dir, opt.dataset, name)

os.makedirs('%s/gen/' % opt.log_dir, exist_ok=True)
os.makedirs('%s/plots/' % opt.log_dir, exist_ok=True)


opt.max_step = opt.n_past+opt.n_future
print("Random Seed: ", opt.seed)
random.seed(opt.seed)
torch.manual_seed(opt.seed)
torch.cuda.manual_seed_all(opt.seed)
dtype = torch.cuda.FloatTensor
opt.data_type = 'sequence'

# ---------------- load the models  ----------------

print(opt)

# ---------------- optimizers ----------------
opt.optimizer = optim.Adam


from . import crevnet as model

frame_predictor = model.zig_rev_predictor(opt.rnn_size,  opt.rnn_size, opt.g_dim, opt.predictor_rnn_layers,opt.batch_size,'lstm',[x//16 for x in args.frame_shape], dim=opt.dim, periodic=opt.periodic)
encoder = model.autoencoder(nBlocks=[2,2,2,2], nStrides=[1, 2, 2, 2],
                    nChannels=None, init_ds=2,
                    dropout_rate=0., affineBN=True, in_shape=[opt.channels]+opt.frame_shape,
                    mult=4, dim=opt.dim, periodic=opt.periodic)
print(f'debug predictor\n{frame_predictor}\n encoder\n{encoder}')


frame_predictor_optimizer = opt.optimizer(frame_predictor.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
encoder_optimizer = opt.optimizer(encoder.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

try:
    checkpoint = torch.load(f'{opt.log_dir}/model.pth')
    encoder = checkpoint['encoder']
    frame_predictor = checkpoint['frame_predictor']
    encoder_optimizer = checkpoint['encoder_optimizer']
    frame_predictor_optimizer = checkpoint['frame_predictor_optimizer']
    print(f"restored model and optimizer from {opt.log_dir}/model.pth")
except:
    pass

# --------- loss functions ------------------------------------
mse_criterion = nn.MSELoss() if args.loss=='L2' else nn.L1Loss()

# --------- transfer to gpu ------------------------------------
frame_predictor.cuda()
encoder.cuda()
mse_criterion.cuda()

# --------- load a dataset ------------------------------------
# train_data, test_data = data_utils.load_dataset(opt)
if args.dataset == 'longclip':
    import NPS.data
    from collections import namedtuple
    if False: #args.file_test:
        file_test = args.file_test
    else:
        file_test = {"eval":"valid","train":"valid","predict":"test"}[args.mode]
        file_test = f'{args.data}/{file_test}.npy'
    data_args = {'datatype_train':'longclip', 'datatype_test':'longclip',
      'file_train':f'{args.data}/train.npy', 'file_test':file_test,
      'minibatch_size':args.batch_size, 'cpu':False, 'n_threads':1, 'test_only':False,
      'dim':args.dim, 'data_slice':'', 'data_filter':'', 'data_preprocess':'',
      'channel_first':True, 'space_CG':False, 'frame_shape':args.frame_shape, 'time_CG':1,
      'total_length_test':args.n_in+(args.n_out if args.n_out_test<0 else args.n_out_test),
      'clip_step_test':args.clip_step if args.clip_step_test<0 else args.clip_step_test,
      'total_length':args.n_in+args.n_out, 'frame_step':1, 'clip_step':args.clip_step, 'i_in_out':False, 'n_in':args.n_in}
    data_args = namedtuple('Data_args_init', data_args.keys())(*data_args.values())
    loader = NPS.data.Data(data_args)
    print(f'debug training DS shape {loader.train.flat.shape} {loader.train.start_pos.shape} test {loader.test.flat.shape} {loader.test.start_pos.shape}')

train_loader = DataLoader(loader.train,
                          num_workers=2,
                          batch_size=opt.batch_size,
                          shuffle=True,
                          drop_last=True,
                          pin_memory=False)
test_loader = DataLoader(loader.test,
                         num_workers=0,#opt.data_threads,
                         batch_size=opt.batch_size,
                         shuffle=False,
                         drop_last=True)#,
                        #  pin_memory=False)
print(f'debug loader {loader} train {loader.train} {train_loader} test {loader.test} {test_loader}')


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

def plot(testing_batch_generator, epoch):
    # nsample = 1
    predictions = []
    # gt_seq = [x[i][:,:3].cpu() for i in range(len(x))]
    target = []
    mse_detail = []
    mae_detail = []
    # for s in range(nsample):
        # for i in range(10):
        #     x = next(testing_batch_generator)
    for i, x in enumerate(test_loader):
        x = x.transpose_(0, 1).to('cuda')
        # print(f'debug testing {i} {x.shape}')
        n_tgt = x.shape[0]
        batch = x.shape[1]
        prediction = []
        frame_predictor.hidden = frame_predictor.init_hidden()
        memo = Variable(torch.zeros(batch, opt.rnn_size, *[w//16 for w in args.frame_shape]).cuda())

        x_in = x[0]
        prediction.append(x_in.cpu())
        target.append(x.transpose(0,1).cpu())
        for i in range(1, n_tgt):
            h = encoder(x_in)
            if args.RNN == 0:
                memo.zero_()
                frame_predictor.hidden = frame_predictor.init_hidden()
            if i < opt.n_past:
                _, memo,_ = frame_predictor((h,memo))
                x_in = x[i]
                # predictions.append(x_in)
            else:
                # print(f'debug h {h[0].shape} {h[1].shape} memo {memo.shape}')
                h_pred, memo,_ = frame_predictor((h, memo))
                x_in =encoder(h_pred,False).detach()
            prediction.append(x_in.cpu())
        predictions.append(torch.stack(prediction,1))
    # target = torch.cat(target, 0).numpy()
    target = np.concatenate(target, 0)
    # predictions = np.swapaxes(np.array(np.array(torch.stack(gen_seq[0]).cpu())),0,1)
    predictions = torch.cat(predictions,0).numpy()
    # print(f'tgt {target.shape} pred {predictions.shape}')
    # if p:
    #     # print(f'gen seq {len(gen_seq)} {len(gen_seq[0])} {np.array(torch.stack(gen_seq[0]).cpu()).shape}')
    #     # print(f'gt seq {len(gt_seq)} {len(gt_seq[0])} {gt_seq.shape}')
    #     np.save(f'{opt.log_dir}/gen/gt{epoch}.npy', target)
    #     np.save(f'{opt.log_dir}/gen/pd{epoch}.npy', predictions)
    np.save(f'{opt.log_dir}/gen/gt.npy', target)
    np.save(f'{opt.log_dir}/gen/pd.npy', predictions)

    # print(f'debug gt_seq {gt_seq.shape} n_in {opt.n_past} ntgt {n_tgt} gen {len(gen_seq[0])} {gen_seq[0][0].shape}')
    # for t in range(opt.n_past, n_tgt):
    #     for i in range(opt.batch_size):
    #         mse += torch.mean((gt_seq[t][i][:,:,:].data.cpu() - gen_seq[0][t][i][:,:,:].data.cpu()) ** 2)
    mse_detail= (np.mean((predictions[:,1:]-target[:,1:])**2 , axis=tuple(range(3,3+args.dim))))
    mae_detail= (np.mean(np.abs(predictions[:,1:]-target[:,1:]) , axis=tuple(range(3,3+args.dim))))
    print(f'Eval per time/channel: mse {np.mean(mse_detail,axis=(0,2))} {np.mean(mse_detail,axis=(0,1))} mae {np.mean(mae_detail,axis=(0,2))} {np.mean(mae_detail,axis=(0,1))}')
    mse = np.mean(mse_detail)
    return mse

    to_plot = []
    gifs = [[] for t in range(n_tgt)]
    nrow = min(opt.batch_size, 10)

    for i in range(nrow):
        row = []
        for t in range(n_tgt):
            row.append(gt_seq[t][i])
        to_plot.append(row)

        s_list = [0]
        for ss in range(len(s_list)):
            s = s_list[ss]
            row = []
            for t in range(n_tgt):
                row.append(gen_seq[s][t][i])
            to_plot.append(row)
        for t in range(n_tgt):
            row = []
            row.append(gt_seq[t][i])
            for ss in range(len(s_list)):
                s = s_list[ss]
                row.append(gen_seq[s][t][i])
            gifs[t].append(row)

    if False:
        fname = '%s/gen/sample_%d.png' % (opt.log_dir, epoch)
        data_utils.save_tensors_image(fname, to_plot)

        fname = '%s/gen/sample_%d.gif' % (opt.log_dir, epoch)
        data_utils.save_gif(fname, gifs)
    return mse


# --------- training funtions ------------------------------------
def train(x, *arg):
    frame_predictor.zero_grad()
    encoder.zero_grad()
    frame_predictor.hidden = frame_predictor.init_hidden()

    mse = 0
    memo = Variable(torch.zeros(opt.batch_size, opt.rnn_size, *[x//16 for x in args.frame_shape]).cuda())


    for i in range(1, opt.n_past + opt.n_future):
        h = encoder(x[i - 1] + args.noise* torch.randn_like(x[i - 1]), True)
        # print(f'debug h {h[0].shape} {h[1].shape} memo {memo.shape}')
        if args.RNN == 0:
            memo.zero_()
            frame_predictor.hidden = frame_predictor.init_hidden()
        h_pred,memo,_ = frame_predictor((h,memo))
        x_pred = encoder(h_pred, False)
        mse +=  (mse_criterion(x_pred, x[i]))

    loss = mse
    loss.backward()

    frame_predictor_optimizer.step()
    encoder_optimizer.step()

    return mse.data.cpu().numpy() / (opt.n_past + opt.n_future)



# --------- training loop ------------------------------------
scheduler = ReduceLROnPlateau(encoder_optimizer, mode='min', patience=10,factor=0.5,verbose=True)
for epoch in range(opt.niter):
    frame_predictor.train()
    encoder.train()
    epoch_mse = 0

    for i in range(opt.epoch_size):
        x = next(training_batch_generator)
        # print(f'debug x in train {x.shape} epo {epoch} {i} opt.epoch_size {opt.epoch_size} opt.niter {opt.niter}')
        mse = train(x,epoch)
        epoch_mse += mse

    with torch.no_grad():
        frame_predictor.eval()
        encoder.eval()

        eval = plot(testing_batch_generator, epoch)
        scheduler_enc.step(eval)
        # for i in range(10):
        #     x = next(testing_batch_generator)
        # for i, x in enumerate(testing_batch_generator):
        #     ssim = 
        #     eval += ssim

        print('[%02d] mse loss: %.7f| ssim eval: %.7f(%d)' % (
            epoch, epoch_mse / opt.epoch_size, eval / 360.0, epoch * opt.epoch_size * opt.batch_size))


    # save the model
    if epoch % 1 == 0:
        torch.save({
            'encoder': encoder,
            'frame_predictor': frame_predictor,
            'encoder_optimizer': encoder_optimizer,
            'frame_predictor_optimizer': frame_predictor_optimizer,
            'opt': opt},
            '%s/model.pth' % (opt.log_dir))
