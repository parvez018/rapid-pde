import torch
from utils import get_cmd_inputs

from growth_model import GrainGrowthSingleTimestep
from torch import linalg as LA
import parameters


if __name__=="__main__":
    skip_step, batch_size, epochs, dims, compression, rid = get_cmd_inputs()
    skip_step = 100
    L = 8.0
    kappa = 0.1
    
    filename_pkl = "data/all_data_Nx%d_dtime0.05_L8.0_k0.1.pkl"%(dims)
    # trained_modelname = "model/trained_Nx%d_dtime0.05_batch%r_comp%r.model"%(dims,batch_size,compression)
    trained_modelname = "model/trained_Nx%d_dtime0.05_batch%r_comp%r_rid%d.model"%(dims,batch_size,compression,rid)
    gt_modelname = 'model/gt_model_Nx%d_dtime%r_L%r_k%r.model'%(dims,parameters.dtime,L,kappa)
    
    
    # for baseline model evaluation
    # trained_modelname = "model/baseline_Nx%d_dtime0.05_batch%r.model"%(dims,batch_size)
    
    
    grain_model = GrainGrowthSingleTimestep()
    grain_model.load_state_dict(torch.load(trained_modelname))
    grain_model.eval()
    
    print("grain_model",grain_model.state_dict())
    
    
    all_data = torch.load(filename_pkl)
    all_data = all_data[1500:1800]
    total = 100
    diff_norm = 0.0
    with torch.no_grad():
        for i in range(total):
            etas = all_data[i]['etas']
            for istep in range(skip_step):
                etas_new = grain_model(etas)
                del etas
                etas = etas_new
            gt_etas = all_data[i+skip_step]['etas']
            # print("gt_etas.size",gt_etas.size())
            # print("etas.size",etas.size())
            diff_norm += LA.norm(etas-gt_etas)
    
    avg_diff_norm = diff_norm/total
    print("File:",filename_pkl)
    print("Model:",trained_modelname)
    print("Error:",avg_diff_norm)
# load data 1500:2000
# load trained model

# for frame i=1500-1600,
# simulate for 100 steps starting from i
# compare with ground truth frame = L2 norm difference average 