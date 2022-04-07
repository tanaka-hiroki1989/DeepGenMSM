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
from networks import Net_P,Net_G

all_trajs=np.load('data/traj.npy')
all_trajs_val=np.load('data/traj_val.npy')

beta=1.

lb=-1.
ub=1.
grid_num=100
delta_t=0.01
diffusion_model=OneDimensionalModel(potential_function,beta,lb,ub,grid_num,delta_t)

tau=5

def log_sum_exp(value, dim=None, keepdim=False):
    """Numerically stable implementation of the operation

    value.exp().sum(dim, keepdim).log()
    """
    # TODO: torch.max(value, dim=None) threw an error at time of writing
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(torch.exp(value0),
                                       dim=dim, keepdim=keepdim))
    else:
        m = torch.max(value)
        sum_exp = torch.sum(torch.exp(value - m))
    return m + torch.log(sum_exp)

state_num=4

partition_mem=np.empty([3,diffusion_model.center_list.shape[0],state_num])
K_0_mem=np.empty([3,state_num,state_num])
its_0_mem=np.empty([3,3])
transition_density_0_mem=np.empty([3,diffusion_model.center_list.shape[0],diffusion_model.center_list.shape[0]])
stationary_density_0_mem=np.empty([3,diffusion_model.center_list.shape[0]])

for kk in range(3):
    traj=all_trajs[kk]
    traj_val=all_trajs_val[kk]

    P=Net_P(1,state_num)
    G=Net_G(1,state_num)

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
    '''
    opt = torch.optim.Adam(list(P.parameters())+list(G.parameters()),lr=LR)
    stopper=EarlyStopping(5)
    for epoch in range(200):
        idx_mem_0=torch.randperm(data_size)
        idx=0
        while True:
            actual_batch_size=min(batch_size,data_size-idx)
            if actual_batch_size<=0:
                break
            X_0=Variable(X_mem[idx_mem_0[idx:idx+actual_batch_size]])
            Y_0=Variable(Y_mem[idx_mem_0[idx:idx+actual_batch_size]])
            idx+=actual_batch_size
            log_Chi_0=P(X_0)
            log_Gamma_0=G(Y_0)
            log_Gamma_0=log_Gamma_0-log_sum_exp(log_Gamma_0,0)+log(actual_batch_size+0.)
            ll=log_sum_exp(log_Chi_0+log_Gamma_0,1)
            loss=-torch.mean(ll)
            opt.zero_grad()
            backward(loss)
            opt.step()
    
        P.eval()
        G.eval()
        log_Chi_val=P(X_val)
        log_Gamma_val=G(Y_val)
        log_Gamma_val=log_Gamma_val-log_sum_exp(log_Gamma_val,0)+log(data_size_val+0.)
        ll=log_sum_exp(log_Chi_val+log_Gamma_val,1)
        loss_val=-torch.sum(ll).data[0]
        print(epoch,loss_val)
        P.train()
        G.train()
        if stopper.read_validation_result([P,G],loss_val):
            break

    P,G=stopper.get_best_model()

    LR=1e-5
    opt = torch.optim.Adam(list(G.parameters()),lr=LR)
    stopper=EarlyStopping(5)
    stopper.read_validation_result(G,loss_val)
    P.eval()
    G.train()
    log_Chi=P(Variable(X_mem)).data
    log_Chi_val=P(X_val)
    for epoch in range(200):
        idx_mem_0=torch.randperm(data_size)
        idx=0
        print(epoch)
        while True:
            actual_batch_size=min(batch_size,data_size-idx)
            if actual_batch_size<=0:
                break
            Y_0=Variable(Y_mem[idx_mem_0[idx:idx+actual_batch_size]])
            log_Chi_0=Variable(log_Chi[idx_mem_0[idx:idx+actual_batch_size]])
            idx+=actual_batch_size
            log_Gamma_0=G(Y_0)
            log_Gamma_0=log_Gamma_0-log_sum_exp(log_Gamma_0,0)+log(actual_batch_size+0.)
            ll=log_sum_exp(log_Chi_0+log_Gamma_0,1)
            loss=-torch.mean(ll)
            opt.zero_grad()
            backward(loss)
            opt.step()

        G.eval()
        Gamma_val=G(Y_val)
        Gamma_val=Gamma_val/torch.mean(Gamma_val,0)
        log_Gamma_val=G(Y_val)
        log_Gamma_val=log_Gamma_val-log_sum_exp(log_Gamma_val,0)+log(data_size_val+0.)
        ll=log_sum_exp(log_Chi_val+log_Gamma_val,1)
        loss_val=-torch.sum(ll).data[0]
        G.train()
        print(epoch,loss_val)
        if stopper.read_validation_result(G,loss_val):
            break
    G=stopper.get_best_model()


    torch.save(P.state_dict(), 'data/ml/P_params_traj_'+str(kk)+'_tau_'+str(tau)+'.pkl')
    torch.save(G.state_dict(), 'data/ml/G_params_traj_'+str(kk)+'_tau_'+str(tau)+'.pkl')
    '''
    P.load_state_dict(torch.load('data/ml/P_params_traj_'+str(kk)+'_tau_'+str(tau)+'.pkl'))
    G.load_state_dict(torch.load('data/ml/G_params_traj_'+str(kk)+'_tau_'+str(tau)+'.pkl'))

    P.eval()
    G.eval()

    xx=Variable(torch.from_numpy(diffusion_model.center_list.reshape(-1,1)).float())
    pp=(torch.exp(P(xx))).data.numpy()
    partition_mem[kk]=pp
    
    Chi_1=torch.exp(P(Variable(Y_mem)))
    log_Gamma=G(Variable(Y_mem))
    Gamma=torch.exp(log_Gamma-log_sum_exp(log_Gamma))
    Gamma=Gamma/torch.mean(Gamma,0)
    K=torch.mm(torch.t(Gamma),Chi_1).data.numpy()/data_size
    K=K/K.sum(1)[:,np.newaxis]
    K_0_mem[kk]=K
    its=-tau*delta_t/np.log(sorted(np.absolute(np.linalg.eigvals(K)), key=lambda x:np.absolute(x),reverse=True)[1:4])
    its_0_mem[kk]=its
    
    print(its)
    print(diffusion_model.its[1:4])
    
    hist_mem=np.empty([diffusion_model.center_list.shape[0],state_num])
    for i in range(state_num):
        hist_mem[:,i]=np.histogram(traj[tau:].reshape(-1),bins=grid_num,range=(lb,ub),density=True,weights=Gamma[:,i].data.numpy().reshape(-1))[0]
        hist_mem[:,i]/=hist_mem[:,i].sum()

    transition_density=pp.dot(hist_mem.T)
    model=markov_model(K)
    stationary_density=model.stationary_distribution.dot(hist_mem.T)

    transition_density_0_mem[kk]=transition_density
    stationary_density_0_mem[kk]=stationary_density

np.save('data/ml/partition_mem',partition_mem)
np.save('data/ml/K_0_mem',K_0_mem)
np.save('data/ml/its_0_mem',its_0_mem)
np.save('data/ml/transition_density_0_mem',transition_density_0_mem)
np.save('data/ml/stationary_density_0_mem',stationary_density_0_mem)
