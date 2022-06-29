import torch
import torch.nn as nn
import sys
import argparse
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
import numpy as np


from irradiation_model_compressed import IrradiationSingleTimestep


from feature_extraction import get_features
from delta_extraction import get_deltas

from compression import compress_deltas,compress_features
from parameters import param


import math
import os
import shutil

device = param.device
torch.set_num_threads(1)


seed = 4321

from set_seed import seed_torch


class IrradiationVideoDataset(Dataset):
    def __init__(self, data_path, filename_pkl,  skip_step=1):
        super(IrradiationVideoDataset, self).__init__()
        # self.all_data = torch.load(os.path.join(data_path, filename_pkl),map_location=device)
        self.all_data = torch.load(os.path.join(data_path, filename_pkl))
        self.all_data = self.all_data[:1000]
        self.skip_step = skip_step
        self.start_skip = 0
        self.cnt = len(self.all_data) - skip_step

        
        self.deltas = get_deltas(self.all_data).type(torch.float).to(device)
        self.features = get_features(self.all_data).type(torch.float).to(device)
        
        
        print("self.deltas.size=",self.deltas.size())
        print("self.features.size=",self.features.size())
        


    def __getitem__(self, index):
        # index = self.all_data[idx]['step']
        return {
            'deltas':self.deltas[index],
            'features': self.features[index]
        }


    def __len__(self):
        return self.cnt


def get_cmd_inputs():
    if len(sys.argv)<6:
        print("python %s skip_step batch_size epochs dims rid"%(sys.argv[0]))
        sys.exit(1)
    return int(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3]),int(sys.argv[4]),int(sys.argv[5])
    

def compare_parameters(trained_modelname,gt_modelname):
    print("Learned parameters:")
    learned_model = IrradiationSingleTimestep()
    learned_model.load_state_dict(torch.load(trained_modelname,map_location=device))
    learned_model.eval()
    print(learned_model.state_dict())
    
    print("Ground truth parameters:")
    gt_model = IrradiationSingleTimestep()
    gt_model.load_state_dict(torch.load(gt_modelname,map_location=device))
    gt_model.eval()
    print(gt_model.state_dict())
    
    



if __name__=='__main__':
    seed_torch(125478)

    data_path = "data/"
    
    # filename_pkl = "void_comp_Nx128_step10000_dt0.01_dx1.pkl"
    # trained_modelname = "models/comp_irrd_Nx128.model"
    # gt_modelname = "models/gt_model_comp_Nx128_step10000_dt0.01_dx1.pkl"



    
    # skip_step, batch_size, epoch = get_cmd_inputs()
    # skip_step = 1
    
    param.nstep = 5000
   
    
    skip_step, batch_size, epoch, param.Nx, rid = get_cmd_inputs()
    skip_step = 1
    
    filename_pkl = 'void_comp_Nx%d_step%d_dt%r_dx%r.pkl'%(param.Nx,param.nstep,param.dt,param.dx)
    trained_modelname = "models/baseline_irrd_Nx%d_r%d.model"%(param.Nx,rid)
    gt_modelname = "models/gt_model_comp_Nx%d_step%d_dt%r_dx%r.pkl"%(param.Nx,param.nstep,param.dt,param.dx)
    
    nprint = 50
        
    time0 = datetime.now()
    
    
    dataset = IrradiationVideoDataset(data_path,filename_pkl)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    ts_model = IrradiationSingleTimestep()
    # ts_model.load_state_dict(torch.load(gt_modelname))
    ts_model = ts_model.to(device)
    
    mse = nn.MSELoss(reduction='sum').to(device)

    lr = 0.1
    optimizer = torch.optim.Adam(ts_model.parameters(), lr=lr)

    minloss = -1
    lambda_cv = 1
    lambda_ci = 1
    lambda_eta = 1

    for istep in range(epoch):
        loss = 0.0
        total_size = 0
        
        for batch in loader:
            features = batch['features'].to(device)
            gt_deltas = batch['deltas'].to(device)
            
            cv_gt_deltas = gt_deltas[:,0]
            ci_gt_deltas = gt_deltas[:,1]
            eta_gt_deltas = gt_deltas[:,2]
            
            cv_delta, ci_delta, eta_delta = ts_model(features)
            
            
            cv_batch_loss = lambda_cv * mse(cv_delta,cv_gt_deltas)
            ci_batch_loss = lambda_ci * mse(ci_delta, ci_gt_deltas)
            eta_batch_loss = lambda_eta * mse(eta_delta, eta_gt_deltas)
            
            
            batch_loss = cv_batch_loss + ci_batch_loss + eta_batch_loss
                       

            optimizer.zero_grad()

            batch_loss.backward()

            optimizer.step()

            # this_size = cv.size(0)
            loss += batch_loss.item()
            # total_size += this_size
            # ts_model.print_params(loss / total_size)
            
            
            # print('(cv_ref-cv_new) =', mse(cv_ref, cv_delta).data.cpu().numpy(), \
            #         '(ci_ref-ci_new) =', mse(ci_ref, ci_delta).data.cpu().numpy(), \
            #         '(eta_ref-eta_new) =', mse(eta_ref, eta_delta).data.cpu().numpy())


        # loss /= total_size
        
        
        if loss < minloss or minloss<0:
            minloss = loss
            # print('Get minimal loss')
            torch.save(ts_model.state_dict(), trained_modelname)
        
        
        if istep % nprint == 0:
            print("Epoch:",istep,"current minloss:",minloss)
            print('current loss:', loss)
            


    timeN = datetime.now()
    compute_time = (timeN-time0).total_seconds()
    
    
    
    compare_parameters(trained_modelname,gt_modelname)
    
    
    print('Compute Time: %10f\n'%compute_time)
    
    print("trained model:",trained_modelname)
    print("trained from data:",filename_pkl)
    