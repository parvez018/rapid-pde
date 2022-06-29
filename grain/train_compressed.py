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

# from compress_delta import project
# from growth_model import GrainGrowthSingleTimestep
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

def batch_project_etas(projector,batch_etas):
    all_prod = None
    n_grain = 2
    for etas in batch_etas:
        eta_proj = None
        for g in range(n_grain):
            
            flat_vec = etas[g].clone().flatten()
            prod = torch.matmul(projector,flat_vec) # for dense matrix from torch.rand
            
            # prod = projector.dot(flat_vec) # for sparse coo matrix from scipy.sparse
            # prod = torch.tensor(prod)
            
            prod = prod.unsqueeze(0)
            if eta_proj is None:
                eta_proj = prod
            else:
                eta_proj = torch.cat((eta_proj,prod))
        eta_proj = eta_proj.unsqueeze(0)
        if all_prod is None:
            all_prod = eta_proj
        else:
            all_prod = torch.cat((all_prod,eta_proj),0)
    print("original.size=",batch_etas.size())
    print("projection.size=",all_prod.size())
    return all_prod

def batch_project_features(projector,batch_features):
    all_prod = None
    n_grain = 2
    n_features = 2
    for features in batch_features:
        grain_feat_proj = None
        for g in range(n_grain):
            feat_proj = None
            for f in range(n_features):
            
                flat_vec = features[g][f].clone().flatten()
                prod = torch.matmul(projector,flat_vec)
                
                # prod = projector.dot(flat_vec) # for sparse coo matrix from scipy.sparse
                # prod = torch.tensor(prod)
                
                prod = prod.unsqueeze(0)
                if feat_proj is None:
                    feat_proj = prod
                else:
                    feat_proj = torch.cat((feat_proj,prod))
            
            feat_proj = feat_proj.unsqueeze(0)
            if grain_feat_proj is None:
                grain_feat_proj = feat_proj
            else:
                grain_feat_proj = torch.cat((grain_feat_proj,feat_proj))
        
        grain_feat_proj = grain_feat_proj.unsqueeze(0)
        if all_prod is None:
            all_prod = grain_feat_proj
        else:
            all_prod = torch.cat((all_prod,grain_feat_proj),0)
    # print("original.size=",batch_features.size())
    # print("projection.size=",all_prod.size())
    return all_prod

class GrainDataset(Dataset):
    def __init__(self, filename_pkl,compression=0.005):
        super(GrainDataset, self).__init__()
        # self.all_data = torch.load(os.path.join(data_path, filename_pkl))
        # self.projector = projection_matrix
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
        
        # initialize projection matrix
        xydim = self.xdim*self.ydim
        reduction_factor = compression
        reduced_dim = int(reduction_factor*xydim)
        self.projection_matrix = np.random.randint(low=1,high=50,size=(reduced_dim,xydim))
        self.projection_matrix = torch.tensor(self.projection_matrix)*1.0
        
        # k = 0.2
        # self.projection_matrix = sparse.random(reduced_dim, xydim, density=k, data_rvs=np.random.random_sample)*50
        
        
        self.comp_deltas = batch_project_etas(self.projection_matrix,self.deltas)
        self.comp_features = batch_project_features(self.projection_matrix,self.features)
        
        print("compressed_feature size:",self.comp_features.size())

    
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
                # 'etas':self.all_data[index]['etas'],
                # 'etas_ref':self.all_data[index+self.skip_step]['etas'],
                # 'features':self.features[index],
                # 'deltas':self.deltas[index],
                'cfeatures':self.comp_features[index],
                'cdeltas':self.comp_deltas[index]
                }
    
    def __len__(self):
        return self.cnt


if __name__=='__main__':
    # seed_torch(1888)
    
    
    
    output_dir = 'output/'
    
    # skip_step, batch_size, epochs = get_cmd_inputs()
    # skip_step = 1

    # filename_pkl = "data/all_data_Nx800_dtime0.05_L8.0_k0.1.pkl"
    # trained_modelname = "model/trained_Nx800_dtime0.05_batch%r.model"%(batch_size)
    
    skip_step, batch_size, epochs, dims, compression, rid = get_cmd_inputs()
    skip_step = 1
    


    filename_pkl = "data/all_data_Nx%d_dtime0.05_L8.0_k0.1.pkl"%(dims)
    trained_modelname = "model/trained_Nx%d_dtime0.05_batch%r_comp%r_rid%d.model"%(dims,batch_size,compression,rid)
    
    
    lr = 0.01
    
    time0 = datetime.now()
    
    # prepare dataset
    dataset = GrainDataset(filename_pkl,compression)
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
    
    # initialize projection matrix
    xydim = dataset.xdim*dataset.ydim
    reduction_factor = 0.01
    reduced_dim = int(reduction_factor*xydim)
    
    
    
    
    # projection_matrix = np.random.randint(low=1,high=50,size=(reduced_dim,xydim))
    # projection_matrix = torch.tensor(projection_matrix)*1.0
    
    nprint = 50
    epsilon = 1e-30
    
    for istep in range(epochs):
        # print("Epoch",ep)
        total_loss = 0
        for batch in loader:
            # etas = batch['etas'].to(device)
            # etas_ref = batch['etas_ref'].to(device)
            # comp_gt_deltas = batch['deltas'].to(device)
            # comp_batch_features = batch['features'].to(device)
            
            # comp_gt_deltas = batch_project_etas(projection_matrix,gt_deltas)
            # comp_batch_features = batch_project_features(projection_matrix,batch_features)
            
            comp_gt_deltas = batch['cdeltas'].to(device)
            comp_batch_features = batch['cfeatures'].to(device)
            
            deltas = grain_model(comp_batch_features)
            
            
            # batch_loss = torch.sum((gt_deltas-deltas)**2)
            batch_loss = torch.sum((comp_gt_deltas-deltas)**2)
            
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