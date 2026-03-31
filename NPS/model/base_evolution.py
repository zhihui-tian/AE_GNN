import numpy as np
import torch
import torch.nn as nn

# from model.common import TwoLayerNet, roll, trace, laplacian_roll, gradient, divergence #, GradientNorm, Laplacian
from NPS import model

def make_model(args):
    return BaseEvolution(args)

#conv = NNop[(dim,'conv')](channel // reduction, channel, 1, padding=0, bias=True)

## Deterministic part, i.e. the phase field equation
class BaseEvolution(nn.Module):
    def __init__(self, configs):
        super(BaseEvolution, self).__init__()
        # import ast
        # self.setting = ast.literal_eval(configs.model_setting)
        self.dim=configs.dim
        self.periodic = configs.periodic
        self.loss = configs.loss #'1*L1'
        self.ngram = configs.ngram
        self.args = configs
        # self.feedforward = configs.feedforward # evaluate with no memory
        stoch = model.make_model(configs, None, configs.model_stoch) if configs.noise else None
        self.deter = model.make_model(configs, None, configs.model_deter)
        self.deter.add_term(stoch)

    def forward(self, c, mask, input_length=1, total_length=2, noise=False, train=False, pred_length=-1):
        """
        If predicting, simply output one clip of frames.
        If training, output clip, clip of mean predictions only, noise correlation, and scatter matrix
        noise = false completely turns off noise
        """
        if train and (pred_length==-1): pred_length = total_length-input_length
        return self.deter(c, mask, input_length, total_length, noise, train=train, pred_length=pred_length)

    def calc_mean(self, c):
        c0 = c[:,0]
        return c0+dc

    def forecast_1frame(self, c, noise=True):
        c_det =self.calc_mean(c)
        if not noise:
            return c_det
        omega_ii, omega_ij = self.calc_var(c)
        noise = self.model_stoch(c)
        H = 0
        for i in range(self.dim):
            H+= noise[...,i:i+1] - roll(noise[...,i:i+1],-1,axis=i+1)
        return c_det + H

    def forecast(self, c, input_length, total_length, noise=True):
        # res = torch.empty([c.shape[0]]+[nstep]+list(c.shape[2:]), dtype=c.dtype, device=c.device)
        cshape = c.shape
        assert (c[0] >= input_length) and (input_length>=1), ValueError(f'Supplied input len {c[0]} must be >= {input_length}')
        cshape[1] = total_length
        res = torch.empty(*cshape, dtype=c.dtype)
        res[:,:input_length] = c[:,:input_length]
        res[:,input_length:] = self.deter(c, noise)
        return res
        # if self.feedforward:
        #     for i in range(input_length, total_length):
        #         res[:, i] = self.forecast_1frame(res[:,i-1:i], noise=noise)
        #     return res
        # else:
        #     return self.deter


    def visualize(self, fname):
        cgrid=np.arange(-1.1, 1.1, 0.01)
        c0 = torch.tensor(cgrid[None,:,None],device='cuda').float()
        pd_pf_list=[]
        for g in self.m1:
            pd_pf = g(c0)
            pd_pf = pd_pf.cpu().detach().numpy().ravel()
            # pd_pf-= pd_pf[0]
            pd_pf_list.append(pd_pf)
        pd_noise = self.Hrrp(c0.repeat_interleave(2,-1))
        pd_noise = pd_noise.cpu().detach().numpy().ravel()
        np.savetxt(fname, np.stack([cgrid, *pd_pf_list, pd_noise]))


class SqErr_cov(nn.Module):
    @staticmethod
    def forward(ctx, x, cov):
        """
         sum over batch index i
         sum_i x[i]^T . cov[i] . x[i]
        """
        conv_inv = torch.inverse(cov)
        ctx.save_for_backward(conv_inv)
        N_cov = cov.shape[1]
        N_batch = cov.shape[0]
        diff = (y-G).view(N_batch,-1)
        nll = 0.5*torch.logdet(cov) + 0.5*torch.tr()
        return


sqerr_cov = SqErr_cov()


class CovarianceNLL_Loss_func(torch.autograd.Function):
    """
        y = c_1 - c_0
        G = phase field eq
        shape = (n_batch, n_c==1(for now), spatial_dims)
        omega = covariance matrix, (n_batch, N_cov, N_cov)
        where N_cov = n_c * (No. of spatial points - conserved)
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
    """
    @staticmethod
    def forward(y, x, ome, sca, conserved):
        raise 'ERROR full NLL loss with covariance matrix NOT implemented yet'
        return torch.nn.functional.mse_loss(y, x) + torch.nn.functional.mse_loss(ome, sca)
        # inv_ome = torch.cholesky_inverse(omega)
        # det = torch.det(omega)
        # ctx.save_for_backward(scatter_mat, omega)
        # return 0.5*(torch.logdet(omega) + (scatter_mat * inv_ome))

    # @staticmethod
    # def backward(ctx, grad_output):
    #     input, label = ctx.saved_tensors
    #     # my code
    #     return grad_input, None

class CovarianceNLL_Loss(nn.Module):
    def __init__(self, conserved=False):
        super(CovarianceNLL_Loss, self).__init__()
        self.conserved = conserved

    def forward(self, y, x, ome, sca):
        return CovarianceNLL_Loss_func.forward(y, x, ome, sca, self.conserved)
        # conv_inv = torch.inverse(cov)
        # ctx.save_for_backward(conv_inv)
        # N_cov = cov.shape[1]
        # N_batch = cov.shape[0]
        # diff = (y-G).view(N_batch,-1)
        # nll = 0.5*torch.logdet(cov) + 0.5*sqerr_cov(diff, cov)
        # return nll



class NNCovarianceNLL_Loss_func(torch.autograd.Function):
    @staticmethod
    def forward(y, x, ome, sca, conserved):
        om = ome.view(-1,4)
        sm = sca.view(-1,4)
        detO = om[:,0]*om[:,3]-om[:,1]*om[:,2]
        loss = detO + (sm[:,0]*om[:,3] + sm[:,3]*om[:,0] - sm[:,1]*om[:,1] - sm[:,2]*om[:,2])/detO
        return 0.5*loss.mean()


class NNCovarianceNLL_Loss(nn.Module):
    def __init__(self, conserved=False):
        super(NNCovarianceNLL_Loss, self).__init__()
        self.conserved = conserved

    def forward(self, y, x, ome, sca):
        return NNCovarianceNLL_Loss_func.forward(y, x, ome, sca, self.conserved)
        # conv_inv = torch.inverse(cov)
        # ctx.save_for_backward(conv_inv)
        # N_cov = cov.shape[1]
        # N_batch = cov.shape[0]
        # diff = (y-G).view(N_batch,-1)
        # nll = 0.5*torch.logdet(cov) + 0.5*sqerr_cov(diff, cov)
        # return nll
