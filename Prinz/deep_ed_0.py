import torch
import torch.nn as nn
from torch.autograd import Variable, grad, backward
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import torch.utils.data as Data
from math import pi,inf,log
import copy

from pyemma.plots import scatter_contour
from pyemma.msm import MSM,markov_model
from scipy import linalg
from approximate_diffusion_models import OneDimensionalModel

import potential_function
from networks import Net_P,Net_G
import EarlyStopping

all_trajs=np.load('data/traj.npy')
all_trajs_val=np.load('data/traj_val.npy')

beta=1.

lb=-1.
ub=1.
grid_num=100
delta_t=0.01
diffusion_model=OneDimensionalModel(potential_function,beta,lb,ub,grid_num,delta_t)

tau=5


    
state_num=4
noise_dim=4

partition_mem=np.empty([3,diffusion_model.center_list.shape[0],state_num])
K_0_mem=np.empty([3,state_num,state_num])
its_0_mem=np.empty([3,3])
transition_density_0_mem=np.empty([3,diffusion_model.center_list.shape[0],diffusion_model.center_list.shape[0]])
stationary_density_0_mem=np.empty([3,diffusion_model.center_list.shape[0]])

for kk in range(3):
    traj=all_trajs[kk]
    traj_val=all_trajs_val[kk]

    P=Net_P(1,state_num)
    G=Net_G(1,state_num,noise_dim)

    P.train()
    G.train()

    batch_size = 100
    LR = 1e-3           # learning rate for generator

    X_mem=torch.from_numpy(traj[:-tau]).float()
    Y_mem=torch.from_numpy(traj[tau:]).float()
    X_val=Variable(torch.from_numpy(traj_val[:-tau]).float())
    Y_val=Variable(torch.from_numpy(traj_val[tau:]).float())
    data_size=X_mem.shape[0]
    data_size_val=traj_val.shape[0]-tau

    opt_P = torch.optim.Adam(P.parameters(),lr=LR)
    opt_G = torch.optim.Adam(G.parameters(),lr=LR)
    stopper=EarlyStopping(5)
    for epoch in range(200):
        idx_mem_0=torch.randperm(data_size)
        idx=0
        while True:
            actual_batch_size=min(batch_size,data_size-idx)
            if actual_batch_size<=0:
                break
            X=Variable(X_mem[idx_mem_0[idx:idx+actual_batch_size]])
            Y=Variable(Y_mem[idx_mem_0[idx:idx+actual_batch_size]])
            idx+=actual_batch_size
            O = P(X)
            M = F.softmax(O,dim=1)
            B0 = torch.eye(state_num)[(state_num-torch.sum(torch.cumsum(M,dim=1)>=Variable(torch.rand(actual_batch_size)).unsqueeze(1),1).long()).data].float()
            B1 = torch.eye(state_num)[(state_num-torch.sum(torch.cumsum(M,dim=1)>=Variable(torch.rand(actual_batch_size)).unsqueeze(1),1).long()).data].float()       
            R0 = Variable(torch.cat((B0,torch.randn(actual_batch_size, noise_dim)),1))
            R1 = Variable(torch.cat((B1,torch.randn(actual_batch_size, noise_dim)),1))
            Y0 = G(R0)
            Y1 = G(R1)
            D = torch.abs(Y0-Y)+torch.abs(Y1-Y)-torch.abs(Y0-Y1)

            G_loss = torch.mean(D)
            opt_G.zero_grad()
            G_loss.backward()
            opt_G.step()
            
            opt_P.zero_grad()
            O.backward((B0+B1-2*M.data)*D.data/(actual_batch_size+0.))
            opt_P.step()

        P.eval()
        G.eval()
        O=P(X_val)
        M = F.softmax(O,dim=1)
        B0 = torch.eye(state_num)[(state_num-torch.sum(torch.cumsum(M,dim=1)>=Variable(torch.rand(data_size_val)).unsqueeze(1),1).long()).data].float()
        B1 = torch.eye(state_num)[(state_num-torch.sum(torch.cumsum(M,dim=1)>=Variable(torch.rand(data_size_val)).unsqueeze(1),1).long()).data].float()       
        R0 = Variable(torch.cat((B0,torch.randn(data_size_val, noise_dim)),1))
        R1 = Variable(torch.cat((B1,torch.randn(data_size_val, noise_dim)),1))
        Y0 = G(R0)
        Y1 = G(R1)
        D = torch.abs(Y0-Y_val)+torch.abs(Y1-Y_val)-torch.abs(Y0-Y1)
        loss_val=(torch.mean(D)).data[0]
        P.train()
        G.train()
        print(epoch,loss_val)
        if stopper.read_validation_result([P,G],loss_val):
            break

    P,G=stopper.get_best_model()

    LR=1e-5
    P.eval()
    opt_G = torch.optim.Adam(G.parameters(),lr=LR)
    stopper=EarlyStopping(5)
    stopper.read_validation_result(G,loss_val)
    M_mem = F.softmax(P(Variable(X_mem)),dim=1).data
    M_val = F.softmax(P(X_val),dim=1)
    for epoch in range(200):
        idx_mem_0=torch.randperm(data_size)
        idx=0
        while True:
            actual_batch_size=min(batch_size,data_size-idx)
            if actual_batch_size<=0:
                break
            M=Variable(M_mem[idx_mem_0[idx:idx+actual_batch_size]])
            Y=Variable(Y_mem[idx_mem_0[idx:idx+actual_batch_size]])
            idx+=actual_batch_size
            B0 = torch.eye(state_num)[(state_num-torch.sum(torch.cumsum(M,dim=1)>=Variable(torch.rand(actual_batch_size)).unsqueeze(1),1).long()).data].float()
            B1 = torch.eye(state_num)[(state_num-torch.sum(torch.cumsum(M,dim=1)>=Variable(torch.rand(actual_batch_size)).unsqueeze(1),1).long()).data].float()       
            R0 = Variable(torch.cat((B0,torch.randn(actual_batch_size, noise_dim)),1))
            R1 = Variable(torch.cat((B1,torch.randn(actual_batch_size, noise_dim)),1))
            Y0 = G(R0)
            Y1 = G(R1)
            D = torch.abs(Y0-Y)+torch.abs(Y1-Y)-torch.abs(Y0-Y1)
        
            G_loss = torch.mean(D)
            opt_G.zero_grad()
            G_loss.backward()
            opt_G.step()
        
        G.eval()
        B0 = torch.eye(state_num)[(state_num-torch.sum(torch.cumsum(M_val,dim=1)>=Variable(torch.rand(data_size_val)).unsqueeze(1),1).long()).data].float()
        B1 = torch.eye(state_num)[(state_num-torch.sum(torch.cumsum(M_val,dim=1)>=Variable(torch.rand(data_size_val)).unsqueeze(1),1).long()).data].float()       
        R0 = Variable(torch.cat((B0,torch.randn(data_size_val, noise_dim)),1))
        R1 = Variable(torch.cat((B1,torch.randn(data_size_val, noise_dim)),1))
        Y0 = G(R0)
        Y1 = G(R1)
        D = torch.abs(Y0-Y_val)+torch.abs(Y1-Y_val)-torch.abs(Y0-Y1)
        loss_val=(torch.mean(D)).data[0]
        G.train()
        print(epoch,loss_val)
        if stopper.read_validation_result(G,loss_val):
            break
    
    G=stopper.get_best_model()

    torch.save(P.state_dict(), 'data/ed/P_params_traj_'+str(kk)+'_tau_'+str(tau)+'.pkl')
    torch.save(G.state_dict(), 'data/ed/G_params_traj_'+str(kk)+'_tau_'+str(tau)+'.pkl')

    P.load_state_dict(torch.load('data/ed/P_params_traj_'+str(kk)+'_tau_'+str(tau)+'.pkl'))
    G.load_state_dict(torch.load('data/ed/G_params_traj_'+str(kk)+'_tau_'+str(tau)+'.pkl'))

    P.eval()
    G.eval()

    xx=Variable(torch.from_numpy(diffusion_model.center_list.reshape(-1,1)).float())
    pp=(F.softmax(P(xx),1)).data.numpy()
    partition_mem[kk]=pp
    
    TEST_BATCH_SIZE=10000
    sample_mem=np.empty([TEST_BATCH_SIZE,state_num])
    for idx in range(state_num):
        B=torch.zeros([TEST_BATCH_SIZE,state_num])
        B[:,idx]=1
        R=Variable(torch.cat((B,torch.randn(TEST_BATCH_SIZE, noise_dim)),1))
        sample_mem[:,idx]=G(R).data.numpy().reshape(-1)
        
    K=np.empty([state_num,state_num])
    for idx in range(state_num):
        GR=Variable(torch.from_numpy(sample_mem[:,idx].reshape(-1,1)).float())
        K[idx,:]=torch.mean(F.softmax(P(GR),1),0).data.numpy()
    K=K/K.sum(1)[:,np.newaxis]
    
    its=-tau*delta_t/np.log(sorted(np.absolute(np.linalg.eigvals(K)), key=lambda x:np.absolute(x),reverse=True)[1:4])
    its_0_mem[kk]=its
    
    print(its)
    print(diffusion_model.its[1:4])
    
    hist_mem=np.empty([diffusion_model.center_list.shape[0],state_num])
    for i in range(state_num):
        hist_mem[:,i]=np.histogram(sample_mem[:,i],bins=grid_num,range=(lb,ub),density=True,)[0]
        hist_mem[:,i]/=hist_mem[:,i].sum()

    transition_density=pp.dot(hist_mem.T)
    model=markov_model(K)
    stationary_density=model.stationary_distribution.dot(hist_mem.T)

    transition_density_0_mem[kk]=transition_density
    stationary_density_0_mem[kk]=stationary_density

np.save('data/ed/partition_mem',partition_mem)
np.save('data/ed/K_0_mem',K_0_mem)
np.save('data/ed/its_0_mem',its_0_mem)
np.save('data/ed/transition_density_0_mem',transition_density_0_mem)
np.save('data/ed/stationary_density_0_mem',stationary_density_0_mem)

for kk in range(3):
    plt.figure()
    plt.plot(partition_mem[kk])
    plt.figure()
    plt.plot(stationary_density_0_mem[kk])
    plt.figure()
    plt.contourf(transition_density_0_mem[kk])