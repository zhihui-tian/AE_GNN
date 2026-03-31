import numpy as np
import torch
import torch.nn as nn

from model.common import TwoLayerNet, roll, trace, laplacian_roll, gradient, divergence #, GradientNorm, Laplacian


def make_model(args):
    return StochEvolution(args)

#conv = NNop[(dim,'conv')](channel // reduction, channel, 1, padding=0, bias=True)

## Deterministic part, i.e. the phase field equation
class StochPhaseEvolution(nn.Module):
    def __init__(self, configs):
        super(StochPhaseEvolution, self).__init__()
        import ast
        self.setting = ast.literal_eval(configs.model_setting)
        self.dim=configs.dim
        self.periodic = configs.periodic
        self.loss = configs.loss #'1*L1'
        self.args = configs
        self.init_mean()
        self.init_noise()

    def init_mean(self):
        print(f'Stoch Phase Evolution model, setting={self.setting} dim={self.dim}')
        self.G0 = TwoLayerNet(1,configs.n_feats,1)
        self.m1=[self.G0]
        self.G1 = TwoLayerNet(1,configs.n_feats,1)
        self.G2 = TwoLayerNet(1,configs.n_feats,1)

    def init_noise(self):
        self.Hrrp = TwoLayerNet(2,configs.n_feats,1, symmetric=True)
        # self.Hrrp_ = TwoLayerNet(2,configs.n_feats,1, symmetric=True)
        # self.Hrrp = lambda x: self.Hrrp_(x)**2
        # self.sparse_idx = torch.LongTensor(torch.from_numpy(StochPhaseFeildEq.get_NN(configs.frame_shape)), requires_grad=False)
        # self.register_buffer('sparse_index', self.sparse_idx)

    @staticmethod
    def get_NN(shape):
        dim = len(shape)
        nTot = np.prod(shape)
        latt = np.arange(nTot).reshape(shape)
        nn = np.stack([np.stack([latt, latt if i==0 else np.roll(latt, 1 if i<=dim else -1, axis=(i-1)%dim) ],-1) for i in range(2*dim+1)], -2).reshape((-1,2))
        return nn

    def forward(self, c, calc='loss', nstep=0, noise=False):
        if calc == 'loss':
            return self.loss_func(c)
        elif calc == 'forecast':
            return self.forecast(c, nstep, noise)

    def calc_mean(self, c):
        c0 = c[:,0]
        dc = 0
        if int(self.setting['G0']):
            dc += laplacian_roll(self.G0(c0), 1, self.dim, 'pt')
        if int(self.setting['G1']):
            grad_norm = GradientNorm(c0)
            dc += self.G1(grad_norm) 
        if int(self.setting['G2']):
            lapl = Laplacian(c0)
            dc += self.G2(lapl)
        return c0+dc

    def calc_var(self, c):
        c0 = c[:,0]
        omega_ij = -torch.cat([self.Hrrp(torch.cat((c0, roll(c0, 1,axis=i+1)),-1)) for i in range(self.dim)], -1)
        omega_ij2= -torch.cat([self.Hrrp(torch.cat((c0, roll(c0,-1,axis=i+1)),-1)) for i in range(self.dim)], -1)
        # omega_ij2= torch.stack([roll(omega_ij[0], -1, axis=i+1) for i in range(self.dim)], 0)
        omega_ii = -torch.sum(omega_ij, -1, keepdim=True)-torch.sum(omega_ij2, -1, keepdim=True)
        for i in range(self.dim):
            if c0.shape[i+1]==2:
                # i.e: a,b,a,b so the correlation should be
                omega_ij[...,i]+= omega_ij2[...,i]
        return omega_ii, omega_ij

    def calc_mean_var(self, c):
        return (self.calc_mean(c),) + self.calc_var(c)

    def loss_func(self, c):
        c_det, omega_ii, omega_ij = self.calc_mean_var(c)
        err = c[:,1]-c_det
        err_ii = err**2
        err_ij = torch.cat([err * roll(err,1,axis=i+1) for i in range(self.dim)], -1)
        if self.loss == 'NNCovarianceNLL':
            omega = torch.cat([omega_ii.unsqueeze(-1).repeat_interleave(self.dim,-2),
                omega_ij.unsqueeze(-1).repeat_interleave(2,-1),
                torch.stack([roll(omega_ii, 1, axis=i+1) for i in range(self.dim)], -2)], -1).reshape(nbatch,-1,4)
            scatter_mat = torch.cat([err_ii.unsqueeze(-1).repeat_interleave(self.dim,-2),
                err_ij.unsqueeze(-1).repeat_interleave(2,-1),
                torch.stack([roll(err_ii, 1, axis=i+1) for i in range(self.dim)], -2)], -1).reshape(nbatch,-1,4)
            loss = NNCovarianceNLL_Loss(c_det, c[:,1], omega, scatter_mat)

            # print(f'debug assembled omega {omega} scatter {scatter_mat}')
            # sel = torch.randperm(omega.shape[1])[:int(0.05*omega.shape[1])]
            # sel = torch.rand(omega.shape[1]).ge(0.99)
            #sel = (torch.cuda.FloatTensor(int(omega.shape[1]*0.05)).uniform_()*omega.shape[1]).type(torch.cuda.LongTensor)
            # print('debug sel', sel, torch.cuda.FloatTensor(int(omega.shape[1]*0.05)).uniform_().type(torch.LongTensor))
            #omega = torch.index_select(omega, 1, sel)
            #scatter_mat = torch.index_select(scatter_mat, 1, sel)
            # print(f'debug scatter_mat {scatter_mat.shape}, ii {err_ii.shape}, ij {err_ij.shape}')
        elif self.loss == 'CovarianceNLL':
            # omega = torch.diag_embed(omega_ii.view(n_batch,-1))
            # omega = [torch.sparse.FloatTensor(self.sparse_idx, [0]], torch.Size([nTot,nTot])).to_dense() for ib in range(c0.shape[0])]
            # print(f'debug omega {omega.shape}, ii {omega_ii.shape}, ij {omega_ij.shape}')
            scatter_mat = torch.cat([err_ii.unsqueeze(-1).repeat_interleave(self.dim,-2),
                err_ij.unsqueeze(-1).repeat_interleave(2,-1),
                torch.stack([roll(err_ii, 1, axis=i) for i in range(self.dim)], err_ii.dim()-1)], -1)
            loss = NNCovarianceNLL_Loss(c_det, c[:,1], omega, scatter_mat)
           # print(f'debug scatter_mat {scatter_mat.shape}, ii {err_ii.shape}, ij {err_ij.shape}')
        elif self.loss in ('L2', '1*L2', "MSE", 'L1', '1*L1'):
            omega = torch.cat([omega_ii, omega_ij], -1)
            scatter_mat = torch.cat([err_ii, err_ij], -1)
            lossf_mse = torch.nn.MSELoss(reduction='mean') if self.loss in ('L2', '1*L2', "MSE") else torch.nn.L1Loss(reduction='mean')
            loss = lossf_mse(c_det, c[:,1]) + self.args.f1*lossf_mse(omega, scatter_mat)
        else:
            raise ValueError('ERROR unknown loss '+self.loss)
        return loss


    def forecast_1frame(self, c, noise=True):
        c_det =self.calc_mean(c)
        if not noise:
            return c_det
        omega_ii, omega_ij = self.calc_var(c)
        noise = torch.empty_like(omega_ij).normal_() * torch.sqrt((-omega_ij).abs())
        H = 0
        for i in range(self.dim):
            H+= noise[...,i:i+1] - roll(noise[...,i:i+1],-1,axis=i+1)
        return c_det + H

    def forecast(self, c, nstep, noise=True):
        # res = torch.empty([c.shape[0]]+[nstep]+list(c.shape[2:]), dtype=c.dtype, device=c.device)
        res = torch.empty_like(c)
        res[:,0] = c[:,0]
        for i in range(nstep-1):
            res[:, i+1] = self.forecast_1frame(res[:,i:], noise=noise)
        return res


    def visualize(self, fname):
        cgrid=np.arange(0, 1, 0.01)
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
