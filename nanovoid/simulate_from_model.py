import numpy as np
# import cv2
import imageio
import torch
import torchvision
import os
import sys


from parameters import param
from irradiation_model import IrradiationSingleTimestep
from set_seed import seed_torch

from init_pfields import initialize_nucleus_v1,initialize_nucleus_compressed_learning

seed_torch(546762)


device = param.device


output_path = 'data/'
model_path = 'models/'



def gen_data():
	
		Nx = 100
		compression = 0.0005
		rid = 101
		# modelname = "models/comp_irrd_Nx%d.model"%(Nx)
		# modelname = "models/baseline_irrd_Nx%d.model"%(Nx)
		# modelname = "models/gt_model_comp_Nx%d_step5000_dt0.01_dx1.pkl"%(Nx)
		modelname = "models/comp_irrd_Nx%d_c%r_rid%d.model"%(Nx,compression,rid)
		
		# output_video = "output/etas_sim_from_gt_d%d.mp4"%(Nx)
		# output_video = "output/etas_sim_from_base_d%d_c%r.mp4"%(Nx,compression)
		output_video = "output/etas_sim_from_comp_d%d_c%r.mp4"%(Nx,compression)
		
		param.Nx = 100
		param.Ny = 100
		param.nstep = 20000
		
		cv, ci, eta = initialize_nucleus_compressed_learning(r=param.Nx/3,Nx=param.Nx,Ny=param.Ny)
		
		cv = cv.to(device)
		ci = ci.to(device)
		eta = eta.to(device)
		
		all_etas = eta.clone().unsqueeze(0)
		
		ttime = 0
		ts_model = IrradiationSingleTimestep().to(device)
		
		
		save_frequency = 5000
		with torch.no_grad():
				ts_model.load_state_dict(torch.load(modelname))			

				ts_model.eval()
				all_data = []
				for step in range(param.nstep):
					cv_new, ci_new, eta_new = ts_model(cv, ci, eta)
					del cv
					del ci
					del eta
					cv, ci, eta = cv_new, ci_new, eta_new
					
					if step % save_frequency == 0:
						all_data.append({'step': step, 'cv': cv, 'ci': ci, 'eta': eta})
						all_etas = torch.cat((all_etas,eta.unsqueeze(0)),0)
						

					if step % param.nprint == 0:
						print('Step %d, time %.2f' % (step, ttime))

					ttime += param.dt
				
				
				all_etas = all_etas*128.0 + 100
				all_etas = all_etas.unsqueeze(-1).repeat([1,1,1,3])
				
				if all_etas.is_cuda:
					all_etas = all_etas.cpu()

				torchvision.io.write_video(output_video,all_etas,fps=15)

				print("videofile saved as:",output_video)


if __name__ == '__main__':
	gen_data()
