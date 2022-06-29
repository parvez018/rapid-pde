from datetime import datetime
import parameters
import imageio
from scipy import ndimage, misc
import torch
import csv
import numpy as np
import sys



from growth_model import GrainGrowthSingleTimestep


def read_init_file(filename):
    with open(filename, 'r') as f:
        csv_reader = csv.reader(f)
        first_line = next(csv_reader)
        Nx = int(first_line[0])
        Ny = int(first_line[1])
        n_grain = int(first_line[2])
        etas = []
        for pg in range(n_grain):
            mtx = []
            for i in range(Nx):
                line_str = next(csv_reader)
                line = [float(itm) for itm in line_str]
                mtx.append(line)
            etas.append(np.array(mtx))
    f.close()
    etas = torch.tensor(etas,dtype=torch.float)
    print("input file:",filename)
    print("input from file etas.size=",etas.size())
    return (Nx, Ny, n_grain, etas)




if __name__=='__main__':
    output_dir = "output/"
        
    # input_filename = 'grain_growth_init.out'
    input_filename = 'grain_growth_init.out'
    modelname = "model/gt_model_Nx100_dtime0.05_L8.0_k0.1.model"
    # modelname = "model/trained_Nx100_dtime0.05_batch%r.model"%(batch_size)
    
    
    L = 8.0
    kappa = 0.1
    batch_size = 1
    compression = 0.005
    dims = 100
    rid = 101
    
    compression = float(sys.argv[1])
    rid = int(sys.argv[2])
    
    # modelname = "model/trained_Nx%d_dtime0.05_batch%r_comp%r_rid%d.model"%(dims,batch_size,compression,rid)
    
    # modelname = 'model/gt_model_Nx%d_dtime%r_L%r_k%r.model'%(dims,parameters.dtime,L,kappa)
    
    
    # modelname = trained_modelname
    # modelname = gt_modelname
    
    
    output_filename = "trained_d%d_c%r_etas.pt"%(dims,compression)
    Nx,Ny,n_grain,etas = read_init_file(input_filename)
    Nxy = Nx*Ny
    
    
    time0 = datetime.now()

    # nstep = parameters.nstep
    # nprint = parameters.nprint
    nprint = 100

    nstep = 4100
    
    
    grain_model = GrainGrowthSingleTimestep()
    model = torch.load(modelname)
    print("model",model)
    grain_model.load_state_dict(torch.load(modelname))
    # grain_model = torch.load(modelname)
    grain_model.eval()
    

    all_eta = None
    
    
    ttime = 0
    dtime = parameters.dtime


    with torch.no_grad():
        for istep in  range(nstep):
            ttime = ttime +dtime
            etas_new = grain_model(etas)
            
            etas1 = etas.unsqueeze(0)
            if istep%nprint==0:
                print("Step",istep)
                if all_eta is None:
                    all_eta = etas1
                else:
                    all_eta = torch.cat((all_eta,etas1),0)
            
            etas = etas_new
    
    print("all_eta=",all_eta.size())

    timeN = datetime.now()
    compute_time = (timeN-time0).total_seconds()
    print('Compute Time: %10f\n'%compute_time)
    torch.save(all_eta,output_dir+output_filename)
    
    print("Output saved as",output_dir+output_filename)


