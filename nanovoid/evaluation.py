import torch
from utils import get_cmd_inputs

from irradiation_model import IrradiationSingleTimestep
from torch import linalg as LA

from parameters import param


device = param.device


if __name__=="__main__":
    skip_step, batch_size, epochs, dims, compression, rid = get_cmd_inputs()
    skip_step = 100
    
    param.nstep = 5000
    param.Nx = dims
    
        
    filename_pkl = 'data/void_comp_Nx%d_step%d_dt%r_dx%r.pkl'%(dims,param.nstep,param.dt,param.dx)
    trained_modelname = "models/comp_irrd_Nx%d_c%r_rid%d.model"%(param.Nx,compression,rid)
    gt_modelname = "models/gt_model_comp_Nx%d_step%d_dt%r_dx%r.pkl"%(param.Nx,param.nstep,param.dt,param.dx)
    
    
    # for baseline model evaluation
    # trained_modelname = "models/baseline_irrd_Nx%d.model"%(dims)
    
    
    ts_model = IrradiationSingleTimestep().to(device)
    ts_model.load_state_dict(torch.load(trained_modelname))
    ts_model.eval()
    
    print("ts_model",ts_model.state_dict())
    
    
    all_data = torch.load(filename_pkl)
    all_data = all_data[1500:1800]
    total = 100
    diff_norm = 0.0
    with torch.no_grad():
        for i in range(total):
            cv = all_data[i]['cv'].to(device)
            ci = all_data[i]['ci'].to(device)
            eta = all_data[i]['eta'].to(device)
            for istep in range(skip_step):
                cv_new, ci_new, eta_new = ts_model(cv, ci, eta)
                del cv
                del ci
                del eta
                cv, ci, eta = cv_new, ci_new, eta_new
            gt_eta = all_data[i+skip_step]['eta']
            # print("gt_eta.size",gt_eta.size())
            # print("eta.size",eta.size())
            diff_norm += LA.norm(eta-gt_eta)
    
    avg_diff_norm = diff_norm/total
    print("File:",filename_pkl)
    print("Model:",trained_modelname)
    print("Error for eta:",avg_diff_norm)