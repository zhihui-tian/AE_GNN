__author__ = 'yunbo'

import torch
import torch.nn as nn
import model
from model.common import ConvND
from model.SpatioTemporalLSTMCell import SpatioTemporalLSTMCell
from model.convlstm import convlstm

def make_model(args, parent=False):
    return predrnn(len(args.num_hidden), args.num_hidden, args)

class predrnn(convlstm):
    def init_layers(self):
        print("convlstm initial")
        cell_list = []
        for i in range(self.num_layers):
            in_channel = self.frame_channel if i == 0 else self.num_hidden[i-1]
            cell_list.append(
                SpatioTemporalLSTMCell(in_channel, self.num_hidden[i], self.shape, self.args.kernel_size,
                                       self.args.stride, self.args.layer_norm, dim=self.dim, periodic=self.periodic)
            )
        self.cell_list = nn.ModuleList(cell_list)


    def forward(self, images, mask_true, input_length=1, total_length=2, noise=False, train=False, pred_length=-1):
        batch = images.shape[0]
        shape = self.shape

        h_t = []
        c_t = []
        # c_t_block = []

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

        for i in range(self.num_layers):
            zeros = torch.zeros([batch, self.num_hidden[i]]+shape).to(images.device)
            h_t.append(zeros)
            c_t.append(zeros)

        memory = torch.zeros([batch, self.num_hidden[i]]+shape).to(images.device)

        for t in range(input_length+pred_length-1):
            if t < input_length:
                net = images[:,t]
            else:
                if self.training and mask_true is not None and (t<total_length):
                    net = mask_true[:,t-input_length]*images[:,t] + (1-mask_true[:,t-input_length])*x_gen
                else:
                    net = x_gen
            x0 = net
            h_t[0], c_t[0], memory = self.cell_list[0](net, h_t[0], c_t[0], memory)
            # c_t_block.append(c_t[0].detach())

            for i in range(1, self.num_layers):
                h_t[i], c_t[i], memory = self.cell_list[i](h_t[i - 1], h_t[i], c_t[i], memory)

            x_gen = self.tail(h_t[self.num_layers-1])
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

        # [length, batch, channel, height, width] -> [batch, length, height, width, channel]
        # next_frames = torch.stack(next_frames, dim=1)#.permute(1, 0, 3, 4, 2).contiguous()
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
