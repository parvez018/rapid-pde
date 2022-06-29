import math
import torch
import argparse
import csv
import numpy as np

import parameters as param


# np.random.seed(2021)


def save_out_file(etas, output_file, Nx, Ny):
    with open(output_file, 'w') as f:
        f.write(str(Nx)+','+str(Ny)+','+str(len(etas))+'\n')
        csv_writer = csv.writer(f)
        for eta in etas:
            for i in range(Nx):
                csv_writer.writerow(eta[i])
    f.close()
    print("saved eta size=",np.shape(etas))
    print("saved eta file:",output_file)

def dis(i1, j1, i2, j2):
    return math.sqrt((i1-i2)**2 + (j1-j2)**2)

def prepare_grain_v1(Nx, Ny, n_grain):
    assert n_grain == 2
    eta1 = np.ones((Nx, Ny))
    eta2 = np.zeros((Nx, Ny))
    radius = Nx/4
    for i in range(Nx):
        for j in range(Ny):
            if dis(i,j,Nx/2,Ny/2) < radius:
                eta1[i,j] = 0.0
                eta2[i,j] = 1.0
    return [eta1, eta2]


def prepare_grain_v_debug_multistep(Nx, Ny, n_grain):
    assert n_grain == 3
    eta1 = np.ones((Nx, Ny))
    eta2 = np.zeros((Nx, Ny))
    eta3 = np.zeros((Nx, Ny))
    for i in range(Nx):
        for j in range(Ny):
            if dis(i,j,Nx/2,Ny/2) < Nx/8:
                eta1[i,j] = 0.3
                eta2[i,j] = 0.5
                eta3[i,j] = 0.8
    return [eta1, eta2, eta3]

def prepare_grain_v2(Nx, Ny, n_grain):
    assert n_grain == 2
    eta1 = np.zeros((Nx, Ny))
    eta2 = np.ones((Nx, Ny))
    for i in range(Nx):
        for j in range(Ny):
            if dis(i,j,Nx/2,Ny/2) < Nx/4:
                eta1[i,j] = 1.0
                eta2[i,j] = 0.0
            elif dis(i,j,Nx/2,Ny/2) < 3*Nx/8:
                eta1[i,j] = 0.5
                eta2[i,j] = 0.5
    return [eta1, eta2]

def prepare_grain_v3(Nx, Ny, n_grain):
    assert n_grain == 2
    eta1 = np.zeros((Nx, Ny))
    eta2 = np.ones((Nx, Ny))
    for i in range(Nx):
        for j in range(Ny):
            if i > Nx / 2 + Nx / 8 * math.sin(2*math.pi*20 / Ny * j):
                eta1[i,j] = 1.0
                eta2[i,j] = 0.0
                
    return [eta1, eta2]

def prepare_grain_v4(Nx, Ny, n_grain):
    assert n_grain == 3
    etas = []
    for i in range(n_grain):
        etas.append(np.zeros((Nx,Ny)))
    # eta1 = np.zeros((Nx, Ny))
    # eta2 = np.zeros((Nx, Ny))
    # eta3 = np.zeros((Nx, Ny))
    
    centers = np.random.randint(Nx,size=(3,2))
    centers = [ [Nx*0.25,Ny*0.5], [Nx*0.75,Ny*0.25], [Nx*0.8,Ny*0.8] ]
    print(centers)
    for i in range(Nx):
        for j in range(Ny):
            distances = [dis(i,j,x,y) for [x,y] in centers]
            idx = distances.index(min(distances))
            etas[idx][i,j] = 1.0
                
    return etas
    

##### Main Program

# parser = argparse.ArgumentParser(description='Generate Initial Condition for Grain Growth.')
# parser.add_argument('--output', help='output file.',default='grain_growth_init.out')

# args = parser.parse_args()

output_file = 'grain_growth_init.out'

# initialize eta variables
# param.Nx = param.Ny = 500
param.n_grain = 3
etas = prepare_grain_v4(param.Nx, param.Ny, param.n_grain)
# etas = prepare_grain_v_debug_multistep(param.Nx, param.Ny, param.n_grain)


save_out_file(etas, output_file, param.Nx, param.Ny)

