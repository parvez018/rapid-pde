import torch


class SimulationParameters():
    pass
    
param = SimulationParameters()


param.Nx = 128
param.Ny = 128
param.dx = 1
param.dy = 1
param.nstep = 2000
param.nprint = 100
param.dt = 1e-2
param.eps = 1e-6
param.N = 1

# param.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
param.device = torch.device("cpu")