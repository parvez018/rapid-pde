import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import sys


from init_pfields import initialize_nucleus_v1
from diff_ops import DifferentialOp, LaplacianOp
from parameters import param


import math
import os


# param.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = param.device
# torch.set_default_tensor_type('torch.cuda.FloatTensor')
# device  = torch.device("cpu")



from set_seed import seed_torch
seed_torch(546762)



def log_with_mask(mat, eps=param.eps):
        mask = (mat < eps).detach()
        mat = mat.masked_fill(mask=mask, value=eps)
        return torch.log(mat)


def fix_deviations(mat, lb=0.0, ub=1.0):
        # below_range = ((mat<lb)).sum()
        # above_range = ((mat>ub)).sum()
        # out_of_range = above_range + below_range
        # if out_of_range>0:
        #         print("no. of out-of-range points:",out_of_range,below_range,above_range)
        mat.masked_fill_(torch.ge(mat, ub).detach(), ub)
        mat.masked_fill_(torch.le(mat, lb).detach(), lb)
        return mat


class IrradiationSingleTimestep(nn.Module):
        def __init__(self, normalize=False):
                super(IrradiationSingleTimestep, self).__init__()
                self.energy_v0 = nn.Parameter(torch.randn(1)*5 + 0.1, requires_grad=True)       # E_v^f
                self.energy_i0 = nn.Parameter(torch.randn(1)*5 + 0.1, requires_grad=True)       # E_i^f
                self.kBT0 = nn.Parameter(torch.randn(1)*5 + 0.1, requires_grad=True)
                self.kappa_v0 = nn.Parameter(torch.randn(1) / 2.0 + 0.1, requires_grad=True)
                self.kappa_i0 = nn.Parameter(torch.randn(1) / 2.0 + 0.1, requires_grad=True)
                self.kappa_eta0 = nn.Parameter(torch.randn(1) / 2.0 + 0.1, requires_grad=True)
                #self.r_bulk = nn.Parameter(torch.randn(1) / 10.0 + 0.1, requires_grad=True)
                # self.r_bulk0 = nn.Parameter(torch.tensor(5.0-0.001), requires_grad=False)

                #self.r_surf = nn.Parameter(torch.randn(1) / 10.0 + 0.1, requires_grad=True)
                # self.r_surf0 = nn.Parameter(torch.tensor(10.0-0.001), requires_grad=False)

                # self.r_surf_base = nn.Parameter(torch.tensor(1.0), requires_grad=False)
                # self.p_casc0 = nn.Parameter(torch.tensor(0.01-0.001), requires_grad=False)
                # self.bias0 = nn.Parameter(torch.tensor(0.3-0.001), requires_grad=False)
                # self.vg0 = nn.Parameter(torch.tensor(0.01-0.001), requires_grad=False)
                # self.p_casc = nn.Parameter(torch.randn(1) / 10.0 + 0.1, requires_grad=True)
                #self.bias = nn.Parameter(torch.randn(1) / 10.0 + 0.1, requires_grad=True)      # B
                # self.vg = nn.Parameter(torch.randn(1) / 10.0 + 0.1, requires_grad=True)
                self.diff_v0 = nn.Parameter(torch.randn(1) / 2.0 + 0.1, requires_grad=True)    # D_v
                self.diff_i0 = nn.Parameter(torch.randn(1) / 2.0 + 0.1, requires_grad=True)    # D_i
                self.L0 = nn.Parameter(torch.randn(1) / 2.0 + 0.1, requires_grad=True)


                # self.fluct_norm = param.fluct_norm
                self.dt = param.dt
                self.dx = param.dx
                self.dy = param.dy
                self.lap = LaplacianOp()
                self.normalize = normalize

        def init_params(self, params):
                self.energy_v0.data = torch.tensor([params[0]]) # 8
                self.energy_i0.data = torch.tensor([params[1]]) # 8
                self.kBT0.data = torch.tensor([params[2]])      # 5.0
                self.kappa_v0.data = torch.tensor([params[3]])  # 1.0
                self.kappa_i0.data = torch.tensor([params[4]])  # 1.0
                self.kappa_eta0.data = torch.tensor([params[5]])  # 1.0
                # self.r_bulk0.data = torch.tensor([params[6]])   # 5.0
                # self.r_surf0.data = torch.tensor([params[7]])   # 10.0
                # self.p_casc0.data = torch.tensor([params[8]])   # 0.01
                # self.bias0.data = torch.tensor([params[9]])     # 0.3
                # self.vg0.data = torch.tensor([params[10]])      # 0.01
                self.diff_v0.data = torch.tensor([params[11]])  # 1.0
                self.diff_i0.data = torch.tensor([params[12]])  # 1.0
                self.L0.data = torch.tensor([params[13]])         #1.0


        def print_params(self, batch_loss=0.0):
                print('%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.8f'
                      % (self.energy_v0.item(),
                         self.energy_i0.item(),
                         self.kBT0.item(),
                         self.kappa_v0.item(),
                         self.kappa_i0.item(),
                         self.kappa_eta0.item(),
                         self.r_bulk0.item(),
                         self.r_surf0.item(),
                         self.p_casc0.item(),
                         self.bias0.item(),
                         self.vg0.item(),
                         self.diff_v0.item(),
                         self.diff_i0.item(),
                         self.L0.item(),
                         batch_loss))

        def source_term(self, eta, p_casc, vg):
                R1 = torch.rand_like(eta)
                R2 = torch.rand_like(eta)
                mask = (eta >= 0.8) | (R1 > p_casc)
                result = R2*vg
                result.masked_fill_(mask=mask, value=0.0)
                return result

        def fluctuation(self, c):
                '''
                dimension of c?
                randn_like(c) => returns a vector of size equal to size(c),
                filled with random number drawn for normal distribution(mean=0,var=1)
                fluct_norm => given as param argument
                '''
                return torch.randn_like(c) / self.fluct_norm

        def forward(self, cv, ci, eta):
                '''
                dimensions of cv,ci,eta??
                '''
                energy_v = torch.abs(self.energy_v0) + 0.001
                energy_i = torch.abs(self.energy_i0) + 0.001
                kBT = torch.abs(self.kBT0) + 0.001
                kappa_v = torch.abs(self.kappa_v0) + 0.001
                kappa_i = torch.abs(self.kappa_i0) + 0.001
                kappa_eta = torch.abs(self.kappa_eta0) + 0.001
                # r_bulk = torch.abs(self.r_bulk0) + 0.001
                # r_surf = torch.abs(self.r_surf0) + 0.001


                # p_casc = torch.abs(self.p_casc0) + 0.001
                # bias = torch.abs(self.bias0) + 0.001
                # vg = torch.abs(self.vg0) + 0.001
                diff_v = torch.abs(self.diff_v0) + 0.001
                diff_i = torch.abs(self.diff_i0) + 0.001
                L = torch.abs(self.L0) + 0.001


                lap_cv = self.lap(cv, dx=self.dx, dy=self.dy)
                lap_ci = self.lap(ci, dx=self.dx, dy=self.dy)
                lap_eta = self.lap(eta, dx=self.dx, dy=self.dy)
                h = (eta-1)**2
                j = eta**2


                fs = energy_v * cv + energy_i * ci + kBT * (cv*log_with_mask(cv) + ci*log_with_mask(ci) + (1 - cv - ci) * log_with_mask(1 - cv - ci))
                # fs_mask = (1 - cv - ci < param.eps)
                # fs.masked_fill_(mask=fs_mask, value=0.0)
                dfs_dcv = energy_v + kBT*(log_with_mask(cv) - log_with_mask(1-cv-ci))
                dfs_dci = energy_i + kBT*(log_with_mask(ci) - log_with_mask(1-cv-ci))
                # dfs_dcv.masked_fill_(mask=fs_mask, value=0.0)
                # dfs_dci.masked_fill_(mask=fs_mask, value=0.0)


                fv = (cv-1)**2 + ci**2
                dfv_dcv = 2*(cv-1)
                dfv_dci = 2*ci


                dF_dcv = h * dfs_dcv + j * dfv_dcv - kappa_v * lap_cv
                dF_dci = h * dfs_dci + j * dfv_dci - kappa_i * lap_ci
                dF_deta = param.N * (fs * 2 * (eta-1) + fv * 2 * eta - kappa_eta * lap_eta)


                mv = diff_v * cv / kBT
                mi = diff_i * ci / kBT


                # update
                cv_new = cv + self.dt * (mv * self.lap(dF_dcv) )
                ci_new = ci + self.dt * (mi * self.lap(dF_dci) )
                eta_new = eta + self.dt * (-L * dF_deta )

                
                # print("fixing cv")
                fix_deviations(cv_new)
                
                # print("fixing ci")
                fix_deviations(ci_new)
                
                # print("fixing eta")
                fix_deviations(eta_new)


                # print("L.grad=",L.grad)
                return cv_new, ci_new, eta_new




if __name__=='__main__':
        Nx = 32
        Ny = 32

        cv, ci, eta = initialize_nucleus_v1(r=10.0,Nx=Nx,Ny=Ny)
        model = IrradiationSingleTimestep()

        for i in range(1):
                cv1,ci1,eta1 = model(cv,ci,eta)
                cv = cv1
                ci = ci1
                eta = eta1


        ref = 1.0
        total_loss = torch.sum((ref-cv)**2+(ref-ci)**2+(ref-eta)**2)

        print("Loss=",total_loss.item())
        model.zero_grad()
        total_loss.backward()

        print("energy_v0.grad=",model.energy_v0.grad,model.energy_v0)
        print("energy_i0.grad=",model.energy_i0.grad,model.energy_i0)
        print("kBT0.grad=",model.kBT0.grad,model.kBT0)
        print("L0.grad=",model.L0.grad,model.L0)