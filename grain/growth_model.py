import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader



import sys

from diff_ops import LaplacianOp

import math
import os

import parameters as param

def fix_deviations(mat, lb=0.000, ub=1.000):
        # out_of_range = ((mat<0.5) | (mat>1.0)).sum()
        # print("no. of out-of-range:",out_of_range)
        mat.masked_fill_(torch.ge(mat, ub).detach(), ub)
        mat.masked_fill_(torch.le(mat, lb).detach(), lb)
        return mat

class GrainGrowthSingleTimestep(nn.Module):
        def __init__(self, normalize=False):
                super(GrainGrowthSingleTimestep, self).__init__()
                self.ngrain = param.n_grain
                self.A = torch.tensor(param.A)
                self.B = torch.tensor(param.B)
                # self.L = nn.Parameter(torch.tensor(param.L),requires_grad=True)
                # self.kappa = nn.Parameter(torch.tensor(param.kappa),requires_grad=True)
                
                self.L = nn.Parameter(torch.randn(1)*5 + 0.1,requires_grad=True)
                self.kappa = nn.Parameter(torch.randn(1)*5 + 0.1,requires_grad=True)
                
                self.lap = LaplacianOp()
                self.dx = param.dx
                self.dy = param.dy
                self.dtime = param.dtime
                
        def init_params(self,L,kappa):
            self.L.data = torch.tensor([L])
            self.kappa.data = torch.tensor([kappa])
            
        def print_params(self):
            for name,value in self.named_parameters():
                if value.requires_grad:
                    print(name,value)
        
        def print_grads(self):
            print()
            for ps in self.parameters():
                if ps.requires_grad:
                    print(ps,ps.grad)
            print()
        
        def forward(self,batch_etas):
            # print("batch_etas.size=",batch_etas.size())
            total_dim = batch_etas.dim()
            # print("total dimenstions",total_dim)
            
            
            is_batch = False
            if total_dim==4:
                is_batch = True
            else:
                batch_etas = batch_etas.unsqueeze(0)
            
            
            n_grain = list(batch_etas.size())[-3]
            # print("n_grain",n_grain)
            
            
            all_new_etas = None
            for etas in batch_etas:
                # print("etas.size=",etas.size())
                sum_eta_2 = etas[0]**2
                for i in range(1, n_grain):
                    sum_eta_2 += etas[i]**2
                
                etas_new = torch.zeros_like(etas)
                absL = torch.abs(self.L)
                absKappa = torch.abs(self.kappa)
                for i in range(0, n_grain):
                    dfdeta = -self.A*etas[i] + self.B*(etas[i])**3 
                    
                    sq_sum = sum_eta_2-(etas[i])**2
                    dfdeta += 2*etas[i]*sq_sum
                    
                    lap_eta = self.lap(etas[i],self.dx,self.dy)
                    
                    term1 = dfdeta
                    # term2 = lap_eta
                    term1 = term1-absKappa*lap_eta
                    etas_new[i] = etas[i] - self.dtime*absL*(term1)

                fix_deviations(etas_new)
                etas1 = etas_new.unsqueeze(0)
                if all_new_etas is None:
                    all_new_etas = etas1
                else:
                    all_new_etas = torch.cat((all_new_etas,etas1),0)
            
            final_etas = all_new_etas
            # print("final_etas.size=",final_etas.size())
            if not is_batch:
                final_etas = final_etas.squeeze()
            return final_etas