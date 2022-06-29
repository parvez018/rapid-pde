import torch
import os

class Param:
                pass

param = Param()

param.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


torch.manual_seed(1234)
torch.cuda.manual_seed(1234)

param.min_write_val = 1e-10
param.lr = 1e-1
param.lr2 = 1e-3
param.grad_clip_val = 1e10

param.Nx = 512
param.Ny = 512
param.dx = 1
param.dy = 1
param.nstep = 50
param.nprint = 1
param.dt = 2e-2
param.eps = 1e-6
param.N = 1


def initialize_nucleus_v0(r=7):
	cv = torch.ones(param.Nx, param.Ny, dtype=torch.float) * 0.122
	ci = torch.ones(param.Nx, param.Ny, dtype=torch.float) * 6.9e-4
	eta = torch.zeros(param.Nx, param.Ny, dtype=torch.float)
	for i in range(param.Nx):
		for j in range(param.Ny):
			if (i-param.Nx/2)**2 + (j-param.Ny/2)**2 < r**2:
				cv[i, j] = 1.0
				eta[i, j] = 1.0
	return cv, ci, eta


def initialize_nucleus_v11(r=50.0,Nx=32,Ny=32):
	cv = torch.ones(Nx,Ny, dtype=torch.float) * 6.9e-2
	ci = torch.ones(Nx,Ny, dtype=torch.float) * 0.122
	eta = torch.zeros(Nx,Ny, dtype=torch.float)
	for i in range(Nx):
		for j in range(Ny):
			if (i-Nx/2)**2 + (j-Ny/2)**2 < r**2:
				cv[i, j] = 1.0
				ci[i, j] = 0.0
				eta[i, j] = 1.0
	return cv, ci, eta


def initialize_nucleus_v1(r=50.0,Nx=32,Ny=32):
	cv = torch.ones(Nx,Ny, dtype=torch.float) * 6.9e-2
	ci = torch.ones(Nx,Ny, dtype=torch.float) * 0.122
	eta = torch.ones(Nx,Ny, dtype=torch.float) * 1e-1
	for i in range(Nx):
		for j in range(Ny):
			if (i-Nx/2)**2 + (j-Ny/2)**2 < r**2:
				cv[i, j] = 1.0
				ci[i, j] = 0.0
				eta[i, j] = 1.0
	cv.requires_grad_(True)
	ci.requires_grad_(True)
	eta.requires_grad_(True)
	return cv, ci, eta



def initialize_nucleus_v2(r1=20.0, x1=32, y1=32, r2=10.0, x2=96.0, y2=96.0):
	cv = torch.ones(param.Nx, param.Ny, dtype=torch.float) * 6.9e-2
	ci = torch.ones(param.Nx, param.Ny, dtype=torch.float) * 0.122
	eta = torch.zeros(param.Nx, param.Ny, dtype=torch.float)
	for i in range(param.Nx):
		for j in range(param.Ny):
			if (i-x1)**2 + (j-y1)**2 < r1**2:
				cv[i, j] = 1.0
				ci[i, j] = 0.0
				eta[i, j] = 1.0
			if (i-x2)**2 + (j-y2)**2 < r2**2:
				cv[i, j] = 1.0
				ci[i, j] = 0.0
				eta[i, j] = 1.0
	return cv, ci, eta


def fix_deviations(mat, lb=0.0, ub=1.0):
		mat.masked_fill_(torch.ge(mat, ub).detach(), ub)
		mat.masked_fill_(torch.le(mat, lb).detach(), lb)
		return mat


def initialize_nucleus_compressed_learning(r=50.0,Nx=32,Ny=32):
	cv = torch.ones(Nx,Ny, dtype=torch.float) * 1e-5
	ci = torch.ones(Nx,Ny, dtype=torch.float) * 1e-5
	eta = torch.ones(Nx,Ny, dtype=torch.float) * 1e-5
	for i in range(Nx):
		for j in range(Ny):
			if (i-Nx/2)**2 + (j-Ny/2)**2 < r**2:
				cv[i, j] = 1.0
				ci[i, j] = 0.0
				eta[i, j] = 1.0
	cv.requires_grad_(True)
	ci.requires_grad_(True)
	eta.requires_grad_(True)
	return cv, ci, eta
