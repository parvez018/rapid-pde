import numpy as np
import imageio
import torch
import torchvision
import os

from diff_ops import LaplacianOp



# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D


from datetime import datetime

from torchvision.utils import save_image

import parameters



def project(projector,vec):
    flat_vec = vec.clone().flatten()
    prod = torch.matmul(projector,flat_vec)
    return prod
    


def get_features(batch_etas):
    
    dx = parameters.dx
    dy = parameters.dy
    
    A = parameters.A
    B = parameters.B
    
    lap = LaplacianOp()
    
    n_frames, n_grain, xdim, ydim = batch_etas.size()
    print("n_frames,n_grain,xdim,ydim",n_frames,n_grain,xdim,ydim)
    
    feature_per_frame = 2
    batch_features = torch.zeros(n_frames,n_grain,feature_per_frame,xdim,ydim)
    
    for f in range(n_frames):
    # for etas in batch_etas:
        etas = batch_etas[f]
        sum_eta_2 = etas[0]**2
        for i in range(1, n_grain):
            sum_eta_2 += etas[i]**2
        
        for i in range(0, n_grain):
            dfdeta = -etas[i] + (etas[i])**3 
            
            sq_sum = sum_eta_2-(etas[i])**2
            dfdeta += 2*etas[i]*sq_sum
            
            lap_eta = lap(etas[i],dx,dy)
            
            feature1 = torch.clone(dfdeta)
            feature2 = torch.clone(lap_eta)
            batch_features[f][i][0] = feature1
            batch_features[f][i][1] = feature2
    
    print("batch_feature_size=",np.shape(batch_features))
    return batch_features
    
    
    
if __name__=='__main__':
    output_dir = 'output/'
    # filename_pkl = 'data/all_data_Nx64_dtime0.05.pkl'
    filename_pkl = "data/all_data_Nx64_dtime0.05_L5.0_k0.1.pkl"


    all_data = torch.load(filename_pkl)
    all_data = all_data[:5]
    
    n_frames = len(all_data)
    deltas = torch.zeros(n_frames-1,*all_data[0]['etas'].size())
    print("deltas.size=",deltas.size())
    
    n_grain,xdim,ydim = all_data[0]['etas'].size()
    print("n_grain",n_grain)
    etas = torch.zeros(n_frames,n_grain,xdim,ydim)
    etas[0] = all_data[0]['etas']
    
    for i in range(n_frames-1):
        deltas[i] = all_data[i+1]['etas']-all_data[i]['etas']
        etas[i+1] = all_data[i+1]['etas']
        print("nonzero deltas at frame",i,torch.count_nonzero(deltas[i]))
    

    xydim = 4096
    
    reduction_factor = 0.2
    reduced_dim = int(reduction_factor*xydim)
    
    
    projection_matrix = np.random.randint(low=1,high=50,size=(reduced_dim,xydim))
    
    projection_matrix = torch.tensor(projection_matrix)*1.0
    
    compressed_deltas = torch.zeros(n_frames-1,n_grain,reduced_dim)

    for i in range(n_frames-1):
        for j in range(n_grain):
            compressed_deltas[i][j] = project(projection_matrix,deltas[i][j])
            
            dnorm = np.linalg.norm(deltas[i][j])
            cdnorm = np.linalg.norm(compressed_deltas[i][j])
            print("frame",i,"grain",j,"dnorm",dnorm,"cdnorm",cdnorm)
    
    
    features = get_features(etas[:n_frames-1])
    
    print("original vector dimension",xydim)
    print("reduced vector dimension",reduced_dim)