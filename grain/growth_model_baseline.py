import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader



import sys

from diff_ops import LaplacianOp
from feature_extraction import get_features

import math
import os

import parameters as param

def fix_deviations(mat, lb=0.000, ub=1.000):
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
            self.L.data = torch.tensor(L)
            self.kappa.data = torch.tensor(kappa)
            
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
        
        def forward(self,batch_features):
            # print("FORWARD-->")
            # print("batch_features.size=",batch_features.size())
            # total_dim = batch_etas.dim()
            # # print("total dimenstions",total_dim)
            
            
            # is_batch = False
            # if total_dim==4:
            #     is_batch = True
            # else:
            #     batch_etas = batch_etas.unsqueeze(0)
            

            
            
            n_frames,n_grain,n_features,*feat_dim = batch_features.size()
            deltas = torch.zeros(n_frames,n_grain,*feat_dim)
            
            absL = torch.abs(self.L)
            absKappa = torch.abs(self.kappa)
            
            for f in range(n_frames):
                # etas = batch_etas[f]
                # sum_eta_2 = etas[0]**2
                # for i in range(1, n_grain):
                #     sum_eta_2 += etas[i]**2
                
                # etas_new = torch.zeros_like(etas)
                
                for i in range(0, n_grain):
                    # dfdeta = -self.A*etas[i] + self.B*(etas[i])**3 
                    # sq_sum = sum_eta_2-(etas[i])**2
                    # dfdeta += 2*etas[i]*sq_sum
                    
                    # lap_eta = self.lap(etas[i],self.dx,self.dy)
                    dfdeta = batch_features[f][i][0]
                    lap_eta = batch_features[f][i][1]
                    
                    term1 = dfdeta
                    term1 = term1-absKappa*lap_eta
                    # changes = -self.dtime*absL*term1
                     
                    deltas[f][i] = -self.dtime*absL*term1
                    
                    # enew = etas[i] + deltas[f][i]
                    # deltas[f][i] = enew - etas[i]
            
            # print("deltas.size=",deltas.size())
            return deltas
    

