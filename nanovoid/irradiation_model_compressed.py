import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import sys


from init_pfields import initialize_nucleus_v1
from diff_ops import DifferentialOp, LaplacianOp
from parameters import param
from feature_extraction import get_features


import math
import os



from set_seed import seed_torch
# seed_torch(546762)


device = param.device

def log_with_mask(mat, eps=param.eps):
        mask = (mat < eps).detach()
        mat = mat.masked_fill(mask=mask, value=eps)
        return torch.log(mat)


def fix_deviations(mat, lb=0.0, ub=1.0):
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
                self.diff_v0 = nn.Parameter(torch.randn(1) / 2.0 + 0.1, requires_grad=True)    # D_v
                self.diff_i0 = nn.Parameter(torch.randn(1) / 2.0 + 0.1, requires_grad=True)    # D_i
                self.L0 = nn.Parameter(torch.randn(1) *20 + 0.1, requires_grad=True)


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
                self.diff_v0.data = torch.tensor([params[11]])  # 1.0
                self.diff_i0.data = torch.tensor([params[12]])  # 1.0
                self.L0.data = torch.tensor([params[13]])         #1.0


        def print_params(self, batch_loss=0.0):
                print('%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.8f'
                      % (self.energy_v0.item(),
                         self.energy_i0.item(),
                         self.kBT0.item(),
                         self.kappa_v0.item(),
                         self.kappa_i0.item(),
                         self.kappa_eta0.item(),
                         self.diff_v0.item(),
                         self.diff_i0.item(),
                         self.L0.item(),
                         batch_loss))


        def forward(self, features):
                '''
                input: features=tensor of size: batch x total_features x feature_size
                output: 3 tensors cv_delta,ci_delta,eta_delta
                        cv_delta = tensor of size: batch x feature_size
                        same as above for ci_delta and eta_delta
                '''
                # print("features.size=",features.size())
                # print("features[:,0].size=",features[:,11].size())
                
                # squeezed = False
                # if features.dim()==2:
                #         features = features.unsqueeze(0)
                #         squeezed = True
                
                
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
                
                batch_size,total_features,*tmp = features.size()
                all_cv_delta = None
                all_ci_delta = None
                all_eta_delta = None
                all_deltas = None
                                
                
                
                cv_feature_coeffs = [diff_v*energy_v/kBT,diff_v,-diff_v,diff_v/kBT,-diff_v*kappa_v/kBT]
                cv_delta = torch.zeros_like(features[:,0])
                for find in range(len(cv_feature_coeffs)):
                        cv_delta += cv_feature_coeffs[find] * features[:,find]
                
                cv_delta *= self.dt
                
                # print("\n\nfeatures[:,0].size=",features[:,0].size(),"\n\n")
                # print("cv_delta.size=",cv_delta.size())
                # sys.exit(1)
                
                
                
                ci_feature_coeffs = [diff_i*energy_i/kBT,diff_i,-diff_i,diff_i/kBT,-kappa_i*diff_i/kBT]
                ci_feature_start_ind = 5
                ci_delta = torch.zeros_like(features[:,0])
                for find in range(len(ci_feature_coeffs)):
                        ci_delta += ci_feature_coeffs[find] * features[:,find+ci_feature_start_ind]
                
                ci_delta *= self.dt
                
                
                eta_feature_coeffs = [energy_v*2,energy_i*2,kBT*2,1,-kappa_eta]
                eta_feature_start_ind = 10
                eta_delta = torch.zeros_like(features[:,0])
                for find in range(len(eta_feature_coeffs)):
                        eta_delta += eta_feature_coeffs[find]*features[:,find+eta_feature_start_ind]

                eta_delta *= param.N*(-L)*self.dt
                # eta_delta *= (-L)                
                # eta_delta *= self.dt
                
                # all_cv_delta = cv + cv_delta                
                # all_ci_delta = ci + ci_delta
                # all_eta_delta = eta + eta_delta
                
                
                # fix_deviations(all_cv_delta)
                # fix_deviations(all_ci_delta)
                # fix_deviations(all_eta_delta)
                
                # cv_delta = all_cv_delta - cv
                # ci_delta = all_ci_delta - ci
                # eta_delta = all_eta_delta - eta
                
                
                # return all_cv_delta, all_ci_delta, all_eta_delta
                return cv_delta, ci_delta, eta_delta




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