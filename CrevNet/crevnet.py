import torch
import torch.nn as nn
# import torch.nn.functional as F
from torch.autograd import Variable
# from .utils import split, injective_pad, psi
import torch.backends.cudnn as cudnn
import functools
from torch.nn import init
import random
from itertools import accumulate
from NPS.model.common import ConvND, ConvTransposeND, my_activations


class psi(nn.Module):
    def __init__(self, block_size, dim=2):
        super(psi, self).__init__()
        self.block_size = block_size
        self.block_size_sq = block_size*block_size
        self.block_size_3 = block_size**3
        self.dim = dim

    def inverse(self, inpt):
        if self.dim == 2:
            bl, bl_sq = self.block_size, self.block_size_sq
            bs, d, h, w = inpt.shape
            return inpt.view(bs, bl, bl, int(d // bl_sq), h, w).permute(0, 3, 4, 1, 5, 2).reshape(bs, -1, h * bl, w * bl)
        elif self.dim == 3:
            bl, bl_sq, bl_3 = self.block_size, self.block_size_sq, self.block_size_3
            bs, d, h, w, d3 = inpt.shape
            return inpt.view(bs, bl, bl, bl, int(d // bl_3), h, w, d3).permute(0,4,5,1,6,2,7,3).reshape(bs, -1, h * bl, w * bl, d3*bl)
        else:
            raise ValueError(f'dim must be 2 or 3 but got {self.dim}')

    def forward(self, inpt):
        if self.dim == 2:
            bl, bl_sq = self.block_size, self.block_size_sq
            # print(f'debug psi input forward {inpt.shape}')
            bs, d, new_h, new_w = inpt.shape[0], inpt.shape[1], int(inpt.shape[2] // bl), int(inpt.shape[3] // bl)
            if inpt.shape[2] % bl > 0 or  inpt.shape[3] % bl > 0:
                raise ValueError('Sample cannot be divided evenly by block size. Make sure the spatial dimensions can be divided evenly for all strides. Shape: {}, Block: {}'.format(inpt.shape, bl))

            #print(inpt.shape,(bs, d, new_h, bl, new_w, bl), (bs, d * bl_sq, new_h, new_w))
            #print(inpt.view(bs, d, new_h, bl, new_w, bl).permute(0, 3, 5, 1, 2, 4).shape, (bs, d, new_h, bl, new_w, bl), (bs, d * bl_sq, new_h, new_w))
            return inpt.view(bs, d, new_h, bl, new_w, bl).permute(0, 3, 5, 1, 2, 4).reshape(bs, d * bl_sq, new_h, new_w)
        elif self.dim == 3:
            bl, bl_sq, bl_3 = self.block_size, self.block_size_sq, self.block_size_3
            bs, d, new_h, new_w, new_d3 = inpt.shape[0], inpt.shape[1], int(inpt.shape[2] // bl), int(inpt.shape[3] // bl), int(inpt.shape[4] // bl)
            return inpt.view(bs, d, new_h, bl, new_w, bl, new_d3, bl).permute(0,3,5,7,1,2,4,6).reshape(bs, -1, new_h, new_w, new_d3)
        else:
            raise ValueError(f'dim must be 2 or 3 but got {self.dim}')


class irevnet_block(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, first=False, dropout_rate=0.,
                 affineBN=True, mult=4, dim=2, periodic=False):
        """ buid invertible bottleneck block """
        super(irevnet_block, self).__init__()
        # self.first = first
        print(f'debug in_ch {in_ch} out_ch {out_ch}')
        self.stride = stride
        self.psi = psi(stride, dim=dim)
        layers = []
        if not first:
            layers.append(nn.GroupNorm(1,in_ch//2, affine=affineBN))
            layers.append(nn.ReLU(inplace=True))
        if int(out_ch//mult)==0:
            ch = 1
        else:
            ch =int(out_ch//mult)
        layers.append(ConvND(in_ch//2, ch, kernel_size=3,
                      stride=self.stride, bias=False, dim=dim, periodic=periodic))
        layers.append(nn.GroupNorm(1,ch, affine=affineBN))
        layers.append(nn.ReLU(inplace=True))
        layers.append(ConvND(ch, ch,
                      kernel_size=3, bias=False, dim=dim, periodic=periodic))
        layers.append(nn.Dropout(p=dropout_rate))
        layers.append(nn.GroupNorm(1,ch, affine=affineBN))
        layers.append(nn.ReLU(inplace=True))
        layers.append(ConvND(ch, out_ch, kernel_size=3,
                      bias=False, dim=dim, periodic=periodic))
        self.bottleneck_block = nn.Sequential(*layers)

    def forward(self, x):
        """ bijective or injective block forward """
        x1 = x[0]
        x2 = x[1]
        # print(f'debug input irev {x1.shape} {x2.shape}')
        Fx2 = self.bottleneck_block(x2)
        if self.stride == 2:
            x1 = self.psi.forward(x1)
            x2 = self.psi.forward(x2)
        y1 = Fx2 + x1
        # print(f'debug output irev {x2.shape} {y1.shape}')
        return (x2, y1)

    def inverse(self, x):
        """ bijective or injecitve block inverse """
        x2, y1 = x[0], x[1]
        if self.stride == 2:
            x2 = self.psi.inverse(x2)
        Fx2 = - self.bottleneck_block(x2)
        x1 = Fx2 + y1
        if self.stride == 2:
            x1 = self.psi.inverse(x1)
        x = (x1, x2)
        return x

def merge(x1, x2):
    return torch.cat((x1, x2), 1)


class autoencoder(nn.Module):
    def __init__(self, nBlocks, nStrides, nChannels=None, init_ds=2,
                 dropout_rate=0., affineBN=True, in_shape=None, mult=4,it=False, dim=2, periodic=False):
        super(autoencoder, self).__init__()
        #Reshaped volume number of channels increases with dimension.
        #This works for the default value of init_ds=2 in 3D. Might break for other values.

        block=int(2**dim)

        self.ds = in_shape[2]//2**(nStrides.count(2)+init_ds//2)
        self.init_ds = init_ds
        if init_ds == 1:
            self.in_ch = in_shape[0]
        else:
            #self.in_ch = in_shape[0] * 2**self.init_ds
            self.in_ch = in_shape[0] * block 
        self.nBlocks = nBlocks
        self.first = True
        self.it = it
        # print('')
        # print(' == Building iRevNet %d == ' % (sum(nBlocks) * 3))
        if not nChannels:
            nChannels = [self.in_ch//2, self.in_ch//2 * block,
                         self.in_ch//2 * block**2, self.in_ch//2 * block**3]

        self.init_psi = psi(self.init_ds, dim=dim)
        self.stack = self.irevnet_stack(irevnet_block, nChannels, nBlocks,
                                        nStrides, dropout_rate=dropout_rate,
                                        affineBN=affineBN, in_ch=self.in_ch,
                                        mult=mult, dim=dim, periodic=periodic)

    def irevnet_stack(self, _block, nChannels, nBlocks, nStrides, dropout_rate,
                      affineBN, in_ch, mult, dim=2, periodic=False):
        """ Create stack of irevnet blocks """
        block_list = nn.ModuleList()
        strides = []
        channels = []
        print(f'debug constructor input nChannels, nBlocks, nStrides {nChannels}, {nBlocks}, {nStrides}')
        for channel, depth, stride in zip(nChannels, nBlocks, nStrides):
            strides = strides + ([stride] + [1]*(depth-1))
            channels = channels + ([channel]*depth)
        print(f'debug channels, strides {channels}, {strides}')
        for channel, stride in zip(channels, strides):
            print(f'  debug in_ch {in_ch} channel, stride {channel}, {stride}')
            block_list.append(_block(in_ch, channel, stride,
                                     first=self.first,
                                     dropout_rate=dropout_rate,
                                     affineBN=affineBN, mult=mult, dim=dim, periodic=periodic))
            in_ch = 2 * channel
            self.first = False
        return block_list

    def forward(self, input, is_predict = True):

        # print(f'debug ae input class {self.it} input {input.__class__}')
        # print(f'debug ae mode {is_predict} input it {self.it} input {input[0].shape} {input[1].shape} {input.__class__}', 'tuple' if isinstance(input, tuple) else input.shape)# {input.shape}')
        if is_predict:
            n = self.in_ch // 2
            if self.it:
                input = merge(input[0], input[1])
            if self.init_ds != 0:
                x = self.init_psi.forward(input)
            else:
                x = input

            # print(f"[AE] Input shape: {x.shape}")    #
            out = (x[:, :n], x[:, n:])
            for block in self.stack:
                out = block.forward(out)
            x = out
            # print(f"[AE] Final x2 spatial shape: {out[0].shape}")   #
        else:
            out = input
            for i in range(len(self.stack)):
                out = self.stack[-1 - i].inverse(out)
            out = merge(out[0], out[1])
            if self.init_ds != 0:
                x = self.init_psi.inverse(out)
            else:
                x = out
            if self.it:
                n = self.in_ch // 2
                x = (x[:, :n, :, :], x[:, n:, :, :])
        # print(f'debug ae output x {x[0].shape} {x[1].shape}')
        return x



class STConvLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, memo_size, dim=2, periodic=False):
        super(STConvLSTMCell,self).__init__()
        self.KERNEL_SIZE = 3
        self.PADDING = self.KERNEL_SIZE // 2
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.memo_size = memo_size
        self.in_gate = ConvND(input_size + hidden_size + hidden_size, hidden_size, self.KERNEL_SIZE, dim=dim, periodic=periodic)
        self.remember_gate = ConvND(input_size + hidden_size+ hidden_size, hidden_size, self.KERNEL_SIZE, dim=dim, periodic=periodic)
        self.cell_gate = ConvND(input_size + hidden_size+ hidden_size, hidden_size, self.KERNEL_SIZE, dim=dim, periodic=periodic)

        self.in_gate1 = ConvND(input_size + memo_size+ hidden_size, memo_size, self.KERNEL_SIZE, dim=dim, periodic=periodic)
        self.remember_gate1 = ConvND(input_size + memo_size+ hidden_size, memo_size, self.KERNEL_SIZE, dim=dim, periodic=periodic)
        self.cell_gate1 = ConvND(input_size + memo_size+ hidden_size, memo_size, self.KERNEL_SIZE, dim=dim, periodic=periodic)

        self.w1 = ConvND(hidden_size + memo_size, hidden_size, 1, dim=dim)
        self.out_gate = ConvND(input_size + hidden_size +hidden_size+memo_size, hidden_size, self.KERNEL_SIZE, dim=dim, periodic=periodic)


    def forward(self, input, prev_state):
        input_,prev_memo = input
        # print(f'debug ST in: hidden_size {self.hidden_size} input_ {len(input)} {input[0].shape} prev {prev_state[0].shape} {prev_state[1].shape}')
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]

        # generate empty prev_state, if None is provided
        if prev_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            prev_state = (
                Variable(torch.zeros(state_size)).cuda(),
                Variable(torch.zeros(state_size)).cuda()
            )

        prev_hidden, prev_cell = prev_state

        # data size is [batch, channel, height, width]
        # print(f'debug hidden_size {self.hidden_size} input_ {len(input)} {input[0].shape} prev {prev_hidden.shape} {prev_cell.shape}')
        stacked_inputs = torch.cat((input_, prev_hidden,prev_cell), 1)

        # print(f'debug stacked_inputs {stacked_inputs.shape} input_, prev_hidden,prev_cell {input_.shape} {prev_hidden.shape} {prev_cell.shape} prev_memo {prev_memo.shape} ')
        in_gate = torch.sigmoid(self.in_gate(stacked_inputs))
        remember_gate = torch.sigmoid(self.remember_gate(stacked_inputs))
        cell_gate = torch.tanh(self.cell_gate(stacked_inputs))

        # compute current cell and hidden state
        cell = (remember_gate * prev_cell) + (in_gate * cell_gate)

        stacked_inputs1 = torch.cat((input_, prev_memo,cell), 1)

        in_gate1 = torch.sigmoid(self.in_gate1(stacked_inputs1))
        remember_gate1 = torch.sigmoid(self.remember_gate1(stacked_inputs1))
        cell_gate1 = torch.tanh(self.cell_gate1(stacked_inputs1))



        memo = (remember_gate1 * prev_memo) + (in_gate1 * cell_gate1)

        out_gate = torch.sigmoid(self.out_gate(torch.cat((input_, prev_hidden,cell,memo), 1)))
        hidden = out_gate * torch.tanh(self.w1(torch.cat((cell,memo),1)))
        #print(hidden.size())
        return (hidden, cell),memo




class zig_rev_predictor(nn.Module):
    # def __init__(self, input_size, hidden_size,output_size,n_layers,batch_size, type = 'lstm', frame_shape=(8,8), dim=2, periodic=False):
    def __init__(self, input_size, hidden_size, output_size, n_layers, type = 'lstm', frame_shape=(8,8), dim=2, periodic=False):
        super(zig_rev_predictor, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        # self.batch_size = batch_size
        self.type = type
        self.frame_shape = frame_shape

        self.convlstm = nn.ModuleList(
                [STConvLSTMCell(input_size if i == 0 else hidden_size, hidden_size,hidden_size) for i in
                 range(self.n_layers)])

        self.att = nn.ModuleList([nn.Sequential(ConvND(self.hidden_size, self.hidden_size, 1, dim=dim),
                                                nn.GroupNorm(1,self.hidden_size,affine=True),
                                                nn.ReLU(),
                                                ConvND(self.hidden_size, self.hidden_size, 3, dim=dim, periodic=periodic),
                                                nn.GroupNorm(1, self.hidden_size, affine=True),
                                                nn.ReLU(),
                                                ConvND(self.hidden_size, self.hidden_size, 1, dim=dim),
                                                nn.Sigmoid()
                                                ) for i in range(self.n_layers)])

        # self.hidden = self.init_hidden()
        # self.init_hidden()
        # self.prev_hidden = self.hidden

    def init_hidden(self, x):
        self.hidden = []
        batch_size = x.size(0)

        for i in range(self.n_layers):
            self.hidden.append((Variable(torch.zeros(batch_size, self.hidden_size, *self.frame_shape).cuda()),
                           Variable(torch.zeros(batch_size, self.hidden_size, *self.frame_shape).cuda())))

    # def copy(self,hidden):
    #     self.prev_hidden=[]
    #     if self.type == 'residual':
    #         for i in range(self.n_layers):
    #             self.prev_hidden.append((0,(hidden[i][1][0].clone(),
    #                               hidden[i][1][1].clone())))

    def forward(self, input):
        input_, memo = input
        x1, x2 = input_
        mask = []
        for i in range(self.n_layers):
            out = self.convlstm[i]((x1,memo), self.hidden[i])
            self.hidden[i] = out[0]
            memo = out[1]
            g = self.att[i](self.hidden[i][0])
            mask.append(g)
            x2 = (1 - g) * x2 + g * self.hidden[i][0]
            x1, x2 = x2, x1

        return (x1,x2),memo,mask




