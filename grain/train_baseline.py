import numpy as np
# import imageio
import torch
import torch.nn as nn

# import torchvision
# import os


from datetime import datetime
from scipy import sparse

# from torchvision.utils import save_image

import parameters
from feature_extraction import get_features
from torch.utils.data import Dataset, DataLoader


from growth_model_comp import GrainGrowthSingleTimestep
from utils import get_cmd_inputs

from set_seed import seed_torch

device = torch.device("cpu")

class GrainDataset(Dataset):
    def __init__(self, filename_pkl, projection_matrix=None):
        super(GrainDataset, self).__init__()
        self.all_data = torch.load(filename_pkl)
        
        self.all_data = self.all_data[:1000]
        
        self.n_frames = len(self.all_data)
        self.n_grain,self.xdim,self.ydim = self.all_data[0]['etas'].size()

        skip_step = 1
        self.skip_step = skip_step

        self.cnt = len(self.all_data) - skip_step
        
        self.features = get_features(self.get_etas())
        print("features.size=",self.features.size())
        
        self.deltas = self.get_deltas()
        print("deltas.size=",self.deltas.size())

    
    def get_deltas(self):
        deltas = None
        for i in range(self.cnt):
            change = self.all_data[i+1]['etas']-self.all_data[i]['etas']
            change = change.unsqueeze(0)
            if deltas is None:
                deltas = change
            else:
                deltas = torch.cat((deltas,change))
        return deltas
        
    def get_etas(self):
        etas = None
        for data in self.all_data:
            cureta = data['etas'].unsqueeze(0)
            if etas is None:
                etas = cureta
            else:
                etas = torch.cat((etas,cureta),0)
        return etas


    def __getitem__(self, index):
        return {
                'features':self.features[index],
                'deltas':self.deltas[index]
                }
    
    def __len__(self):
        return self.cnt


if __name__=='__main__':
    # seed_torch(1888)
    
    
    
    output_dir = 'output/'
    
    skip_step, batch_size, epochs, dims, compression, rid = get_cmd_inputs()
    skip_step = 1
    


    filename_pkl = "data/all_data_Nx%d_dtime0.05_L8.0_k0.1.pkl"%(dims)
    trained_modelname = "model/baseline_Nx%d_dtime0.05_batch%r.model"%(dims,batch_size)
    

    lr = 0.01
    
    time0 = datetime.now()
    
    # prepare dataset
    dataset = GrainDataset(filename_pkl)
    loader = DataLoader(dataset,batch_size=batch_size,shuffle=False)


    # initialize model
    grain_model = GrainGrowthSingleTimestep()
    grain_model = grain_model.to(device)
    
    
    print("Initial model parameters:")
    grain_model.print_params()
    initial_modelname = "initial_model.model"
    torch.save(grain_model.state_dict(),initial_modelname)
    torch.save(grain_model.state_dict(),trained_modelname)
    
    # initialize model to be learned, its parameters
    mse = nn.MSELoss(reduction='sum').to(device)
    optimizer = torch.optim.Adam(grain_model.parameters(), lr=lr)
    minloss = -1


    nprint = 50
    epsilon = 1e-30
    
    for istep in range(epochs):
        # print("Epoch",ep)
        total_loss = 0
        for batch in loader:
            gt_deltas = batch['deltas'].to(device)
            batch_features = batch['features'].to(device)
            
        
            deltas = grain_model(batch_features)
            
            
            batch_loss = torch.sum((gt_deltas-deltas)**2)
            
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            
            # print("loss=",batch_loss.item())
            total_loss += batch_loss.item()
        
        # print("loss",total_loss)
        if total_loss<minloss or minloss<0:
            minloss = total_loss
            torch.save(grain_model.state_dict(),trained_modelname)
        
        if istep%nprint==0:
            print("Step",istep)
            print("minloss",minloss)
        
        if minloss<epsilon:
            print("Exiting after epoch",istep)
            break   
    
    
    timeN = datetime.now()
    compute_time = (timeN-time0).total_seconds()
    print('Compute Time: %10f\n'%compute_time)
    
    
    print("Final loss",minloss)
    
    print("Initial model")
    print(torch.load(initial_modelname))
    print("Learned model")
    print(torch.load(trained_modelname))
    
    print("datafile:",filename_pkl)
    print("learned model:",trained_modelname)
    print("epochs:",epochs)
    print("batch size:",batch_size)