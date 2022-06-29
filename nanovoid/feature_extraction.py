import numpy as np
import imageio
import torch
import torchvision
import os

from diff_ops import LaplacianOp



import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


from datetime import datetime

from torchvision.utils import save_image

from parameters import param


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
device = param.device

def log_with_mask(mat, eps=param.eps):
    mask = (mat < eps).detach()
    mat = mat.masked_fill(mask=mask, value=eps)
    return torch.log(mat)

def project(projector,vec):
    flat_vec = vec.clone().flatten()
    prod = torch.matmul(projector,flat_vec)
    return prod
    

def _get_pfields(all_data):
    cv = None
    ci = None
    eta = None
    for item in all_data:
        current_cv = item['cv'].unsqueeze(0)
        current_ci = item['ci'].unsqueeze(0)
        current_eta = item['eta'].unsqueeze(0)
        if cv is None:
            cv = current_cv
            ci = current_ci
            eta = current_eta
        else:
            cv = torch.cat((cv,current_cv),0)
            ci = torch.cat((ci,current_ci),0)
            eta = torch.cat((eta,current_eta),0)
    
    return cv,ci,eta
    
    
def get_features(all_data):
    
    cv,ci,eta = _get_pfields(all_data)
    cv = cv.to(device)
    ci = ci.to(device)
    eta = eta.to(device)
    print("cv.size=",cv.size())
    print("ci.size=",ci.size())
    print("eta.size=",eta.size())
    
    lap = LaplacianOp()
    n_frames, xdim, ydim = cv.size()
    
    
    feature_per_frame = 15
    batch_features = torch.zeros(n_frames,feature_per_frame,xdim,ydim)
    
    print("batch_features[:][0].size=",batch_features[:,0].size())
    

    # single_cv = cv
    # single_ci = ci[f]
    # single_eta = eta[f]
    
    h = (eta-1)
    h2 = (eta-1)**2
    lap_h2 = lap(h2)
    j = eta**2
    log_cv = log_with_mask(cv)
    log_ci = log_with_mask(ci)
    log_cvi = log_with_mask(1-cv-ci)
                
    batch_features[:,0] = lap_h2 * cv
    batch_features[:,1] = lap(h2*log_cv) * cv
    batch_features[:,2] = lap(h2*log_cvi) * cv
    batch_features[:,3] = lap(j*2*(cv-1)) * cv
    batch_features[:,4] = lap(lap(cv)) * cv
    
    
    batch_features[:,5] = ci * lap_h2
    batch_features[:,6] = lap(h2*log_ci) * ci
    batch_features[:,7] = lap(h2*log_cvi) * ci
    batch_features[:,8] = lap(j*2*(ci)) * ci
    batch_features[:,9] = lap(lap(ci)) * ci
    
    batch_features[:,10] = cv*(eta-1) 
    batch_features[:,11] = ci*(eta-1) 
    batch_features[:,12] = (cv*log_with_mask(cv) + ci*log_with_mask(ci) + (1 - cv - ci) * log_with_mask(1 - cv - ci)) * (eta-1) 
    batch_features[:,13] = ((cv-1)**2 + ci**2)*2*eta  
    batch_features[:,14] = lap(eta)
    
    
    
    return batch_features
        