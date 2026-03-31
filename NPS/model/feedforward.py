__author__ = 'Fei Zhou'

import torch
import torch.nn as nn

from NPS.model.common import ConvND, NNop, CNN
from NPS.model.common import shift_to_const
from NPS import model

def make_model(args, parent=False):
    return feedforward(args)


class feedforward(model.NPSModel):
    def __init__(self, configs):
        super(feedforward, self).__init__(configs)
        self.args = configs
        self.num_hidden = configs.num_hidden
        kernel_size = configs.kernel_size
        self.bn = configs.bn
        self.add_c0 = configs.add_c0
        self.input_single_frame = True
        net = configs.ff_model
        assert self.ngram <= self.input_length, "ERROR: expecting input %d >= n_net %d"%(self.input_length, self.ngram)
        self.input_shape=[-1, self.ngram*self.in_feats] + self.frame_shape
        print("constructing feedforward net %s in=%d tot=%d n_net=%d dim=%d hidden=%s"%
              (net, self.input_length, self.total_length, self.ngram, self.dim, str(configs.num_hidden)))
        # initializer=0.001
        # rescale = configs.rescale

        # initializer = torch.random_uniform_initializer(-initializer,initializer)

        if net == "cnn":
            self.dxdt = CNN(self.input_shape[1], self.num_hidden[:-1], self.num_hidden[-1], kernel_size, dim=self.dim, periodic=self.periodic, activation=configs.act)
        # elif net == "cahn_hilliard":
        #     from .cahn_hilliard import che
        #     self.dxdt = che
        elif net[:6] == "resnet":
            if net == 'resnet':
                from model.resnet import resnet
                self.dxdt = resnet(self.input_shape[1], self.num_hidden, kernel_size, self.dim, self.periodic, self.bn)
            # elif net == 'resnet_bottle':
            #     self.dxdt = resnet_bottle
            elif net == 'resnetv2':
                from model.resnetv2 import resnet
                self.dxdt = resnet(self.input_shape[1], self.num_hidden, kernel_size, self.dim, self.periodic, self.bn)
            else:
                raise ValueError('unknown resnet options', net)
        else:
            self.dxdt = model.make_model(configs, None, net)

        self.tail = ConvND(configs.num_hidden[-1], self.out_feats, 1, self.dim, bias=False) if not configs.no_putback_conv else None
        self.noise_func = None
#            if rescale and (rescale != 'none'):
#                x_gen = rescale01(x_gen, rescale)


    def add_term(self, noise_func):
        self.noise_func = noise_func

    def forward(self, images, mask_true, input_length=1, total_length=2, noise=False, train=False, pred_length=-1):
        """
        if train, return deterministic prediction, noise correlation prediction, and scatter matrix
        if predicting, return deterministic+stochastic prediction
        """
        if pred_length == -1:
            pred_length=total_length-input_length
        # print('debug in feedforward,', images.shape, mask_true, input_length, total_length, noise, train)
        if self.noise_func is None:
            noise=False
        gen_images = []
        gen_mean = []
        c_mean = torch.mean(images[:,0], tuple(range(2,images.ndim-1)), True) if self.args.any_conserved else None
        self.conserved = torch.Tensor(self.args.conserved).float()[None,:].cuda() # delete this line when upgrading pytorch
        # shape = images.get_shape().as_list()

        for t in range(self.ngram, input_length+pred_length):
            # print(f'debug t {t} in shape {self.input_shape} single frame {self.input_single_frame}')
            if t <= input_length:
                # x_gen = images[:,t+self.ngram-1]
                xs_gen = images[:,t-self.ngram:t]
            else:
                if self.training and mask_true is not None and (t<total_length):
                    x_gen = mask_true[:,t-input_length-1]*images[:,t] + (1-mask_true[:,t-input_length-1])*x_gen
                if self.ngram > 1:
                    xs_gen = torch.cat([xs_gen[:,1:], torch.unsqueeze(x_gen, 1)], 1)
                else:
                    xs_gen = torch.unsqueeze(x_gen, 1)
            x0 = xs_gen[:, -1]
            x_gen = self.dxdt(xs_gen.view(*self.input_shape) if self.input_single_frame else xs_gen)
            #print('debug x_gen', x_gen)
            #for i in range(num_layers):
            #    x_gen = dxdt(xs_gen, i, num_hidden[i], filter_size, initializer, conv)
            if self.tail is not None: x_gen = self.tail(x_gen)
            if self.add_c0:
                x_gen += x0[:,:self.args.n_colors_out]
            # if rescale and (rescale != 'none'):
            #     x_gen = rescale01(x_gen, rescale)
            if self.args.any_conserved: x_gen = shift_to_const(x_gen, c_mean, self.conserved)
            ####################################### this is end of mean prediction #############
            if train: gen_mean.append(x_gen)
            if noise:
                x_gen_full = x_gen + self.noise_func(x0)
            else:
                x_gen_full = x_gen
            if train or (t>=self.args.tbegin_test and t%self.args.tskip_test==0):
                gen_images.append(x_gen_full)
            x_gen = x_gen_full

        gen_images = torch.stack(gen_images, dim=1)
        if train: gen_mean = torch.stack(gen_mean, dim=1)
        # [batch_size, seq_length, height, width, channels]
        # gen_images = tf.transpose(gen_images, [1,0] + list(range(2,dim+3)))
        if train:
            if noise:
                # note: only the omega and scatter matrices of the last frame was returned for simplicity
                return (gen_images,gen_mean) + self.noise_func.omega_scatter(x0, gen_mean[:,-1]-images[:, total_length-1])
            else:
                return (gen_images,gen_mean, torch.zeros_like(x0),torch.zeros_like(x0))
        else:
            return gen_images
        # loss = 0
        # if l2loss > 0:
        #     loss += tf.nn.l2_loss(gen_images - images[:,1:]) * l2loss
        # if l1loss > 0:
        #     loss += tf.nn.l1_loss(gen_images - images[:,1:]) * l1loss
        # return [gen_images[:,-(seq_length-input_length):], loss]

