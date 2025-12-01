#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Nov Sep 22 2024
Modified on Wed Nov 05 2025
@author: rea nkhumise
"""

#import libraries
import numpy as np
import math
from numpy import random
from numpy import linalg as LA
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.collections import LineCollection
from itertools import product

from copy import deepcopy
from collections import defaultdict
import pandas as pd
import seaborn as sns
import ot
import argparse
import time
import os
from os.path import dirname, abspath, join
import sys
from sklearn.preprocessing import StandardScaler
from  sklearn.model_selection import KFold
import gymnasium as gym
import gymnasium_robotics

#pytorch
import torch
import torch.nn as nn
import torch.optim as optima
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.distributions import Normal, kl_divergence
from torch.distributions.multivariate_normal import MultivariateNormal

#tianshou
from tianshou.policy import DDPGPolicy,SACPolicy,TD3Policy
from tianshou.utils.net.common import Net, get_dict_state_decorator
from tianshou.utils.net.continuous import Actor,ActorProb,Critic

#GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print('device: ', device)

#set directory/location
this_dir = dirname(__file__)
agent_dir = abspath(join(this_dir,'..')) 
sys.path.insert(0,agent_dir)

parser = argparse.ArgumentParser()
parser.add_argument("--n_count", type=int, default=6)
parser.add_argument("--n_iteration", type=int, default=1)
args = parser.parse_args()

##tasks
task = 'Reach' #'Push', 'Slide'
algo = 'sac' #'ddpg' #'td3'   
cnt = 5 
encoding_dim = 25 #


#-------------------------------
# AutoEncoder 
#-------------------------------

# Generic Multi-layer-pereceptron
class MLP(nn.Module):
    def __init__(self,input_dim,hidden_dims):
        super(MLP, self).__init__()
        num_hidden_layer = 1 #2

        if num_hidden_layer == 1:
            self.net = nn.Sequential(
                nn.Linear(input_dim,hidden_dims[0]),
                nn.ReLU(),)
            self.out_dim = hidden_dims[0]
        else:
            self.net = nn.Sequential(
                nn.Linear(input_dim,hidden_dims[0]),
                nn.ReLU(),
                nn.Linear(hidden_dims[0],hidden_dims[1]),
                nn.ReLU(),)
            self.out_dim = hidden_dims[1]

    def forward(self,x):
        return self.net(x)
    
#Conditional VAE
class cVAE(nn.Module):
    def __init__(self, x_dim, y_dim, z_dim=8, hidden=[256,256]):
        super(cVAE, self).__init__()
        self.x_dim = x_dim #actions
        self.y_dim = y_dim #states
        self.z_dim = z_dim #latent_variables

        ##ENCODER
        #encoder q(z|x,y) = N(m,v)
        enc_in = x_dim + y_dim #[actions,states]
        self.encoder_net = MLP(enc_in,hidden)
        self.encoder_mu = nn.Linear(self.encoder_net.out_dim,z_dim) #mean_estimation
        self.encoder_logvar = nn.Linear(self.encoder_net.out_dim,z_dim) #log_variance_estimation

        #DECODER
        dec_in = z_dim + x_dim #[latent,actions]
        self.decoder_net = MLP(dec_in,hidden)
        
        #assuming Gaussian likelihood (i.e. p_(y|x,z) = N(m_l,v_l) )
        self.decoder_mu = nn.Linear(self.decoder_net.out_dim,y_dim) #mean_estimation
        self.decoder_logvar = nn.Linear(self.decoder_net.out_dim,y_dim) #log_variance_estimation

    #Gaussian_parameters: posterior
    def encode(self,x,y):
        xy = torch.cat([x,y],dim=-1)
        h = self.encoder_net(xy)
        mu = self.encoder_mu(h)
        logvar = self.encoder_logvar(h)
        return mu, logvar
    
    #Gaussian_parameters: likelihood
    def decode(self,x,z):
        xz = torch.cat([x,z],dim=-1)
        h = self.decoder_net(xz)
        mu = self.decoder_mu(h)
        logvar = self.decoder_logvar(h) 
        return mu, logvar
    
    def reparameterize(self,mean,log_variance):
        sigma = torch.exp(0.5*log_variance) #log(std) = 0.5*log(var)
        epsilon = torch.randn_like(sigma) #e ~ N(0,I)
        return mean + epsilon*sigma
    
    def forward(self,x,y): #[actions,states]
        #encode
        mu_enc, logvar_enc = self.encode(x,y)

        #latent
        z = self.reparameterize(mu_enc,logvar_enc) #z = mu + e*sig
        
        #decode
        mu_dec, logvar_dec = self.decode(x,z)
        return mu_enc, logvar_enc, mu_dec, logvar_dec, z
    
def loss_function(model,x,y,beta=0.05): #[actions,states]
    mu_enc, logvar_enc, mu_dec, logvar_dec, z = model(x,y)

    ##reconstruction_loss
    #if likelihood/Decoder ( p_(y|x,z) ) is Gaussian
    # print('mu_dec: ', mu_dec.shape)
    # l1 = - F.mse_loss(y,mu_dec)
    # print('l1: ', l1)
    # l1 = - ((y-mu_dec)**2).sum(dim=1) #.mean() #mean of per-sample
    l1 = 0.5*((y-mu_dec)**2).sum(dim=1) # .mean()
    # print('l1: ', l1)
    # print('l1: ', l1.mean())
    # print('l1: ', l1)
    # xxx

    ##regularisation: KL[ q_(z|x,y) || N(0,I)  ]
    sigma_enc = torch.exp(0.5*logvar_enc) #log(std) = 0.5*log(var)
    posterior_dist = Normal(mu_enc,sigma_enc) #approx. posterior
    prior_dist = Normal(torch.zeros_like(mu_enc),torch.ones_like(sigma_enc)) #prior
    
    l2 = kl_divergence(posterior_dist,prior_dist).sum(dim=1)
    # print('l2: ', l2)
    # print('l2: ', l2.mean())
    # xxx
    
    ##loss = -ELBO
    loss = beta*l2 + l1
    # print('loss: ', loss)
    # print('loss: ', loss.mean())
    # print('l2.mean() + l1.mean(): ', l2.mean() + l1.mean())
    # xxx
    return loss.mean()

#load and unpack dataset
def dataset(steps,file_location):
    """
    Extracts states and actions datasets, i.e. policy datasets
    Then outputs normalised states and actions samples

    Args:
        steps: int, policy number
        file_location: str, policy location
    """
    ## load_data
    file_name = f'sample_{steps}.npy'
    file_path = abspath(join(this_dir,file_location,file_name)) #file_path: location + name
    data = np.load(file_path, allow_pickle=True)

    ## trajectories
    x1 = data[0] #states_in_dataset
    y1 = data[1] #actions_in_dataset
    tx1 = np.asarray(data[5]) #len(states)_of_runs
    ty1 = np.asarray(data[5]) #len(actions)_of_runs
    # print('x1: ', len(x1))
    # print('y1: ', len(y1))
    # print('tx1: ', tx1)
    # print('tx2: ', ty1)
    # xxx

    x,y = [],[]
    # iterate_each_rollout
    for i in range(len(tx1)-1):
        states_ = x1[tx1[i]:tx1[i+1]] #states_per_rollout
        actions_ = y1[ty1[i]:ty1[i+1]] #actions_per_rollout

        #create state & action datasets
        x.extend(states_) #shape (N,a)
        y.extend(actions_) #shape (N,b)

    states = np.array(x)
    actions = np.array(y)

    #normalisation (to make values unitless) 
    scaler_state = StandardScaler() # x' = (x - u)/s [u: mean, s: std]
    scaler_action = StandardScaler() # x' = (x - u)/s [u: mean, s: std]
    
    # scaler_state = RobustScaler()
    # scaler_action = RobustScaler()
    states = scaler_state.fit_transform(states)
    actions = scaler_action.fit_transform(actions)
    # print('states_min/max: ', states.min(), states.max())
    # print('actions_min/max: ', actions.min(), actions.max())
    return states, actions #normalised values


##### testing #### 
# n_samples = 1000
# batch_size = 56 #256
dim_x = 4 #2 #4 #actions
dim_y = 10 #2 #0 #states
dim_z = 6 #10 #7 #14



iteration = args.n_iteration
file_location = f'{algo}/{task.lower()}/samples_{cnt}/iter_{iteration}'
steps = 0 #105 #0 #105

states, actions = dataset(steps,file_location)

#data
dim_y = states.shape[1]
dim_x = actions.shape[1]
dim_z = 7 #14
batch_size = 512 #6 #256
n_samples = 20000 #states.shape[0]
# n_samples = 100#0

#convert to torch        
t_states = torch.tensor(states,dtype=torch.float32)
t_actions = torch.tensor(actions,dtype=torch.float32)
# print('t_states: ',t_states.shape)
# print('t_actions: ',t_actions.shape)
# xxx

# #prepare 80%/20% for training/testing
# num_data = int(n_samples*.8) #80%
# # print('num_data: ', num_data)

# #data splitting
# train_y = t_states[:num_data]
# train_x = t_actions[:num_data]
# test_y = t_states[num_data:]
# test_x = t_actions[num_data:]
# # print('train_y: ',train_y)
# # print('train_x: ',train_x)
# # print('test_y: ',test_y)
# # print('test_x: ',test_x)

# #Split loader
# train_loader =  DataLoader( 
#                     TensorDataset(
#                         train_y,
#                         train_x),
#                     batch_size=batch_size,#512
#                     shuffle=True
#                     )

# test_loader =  DataLoader( 
#                     TensorDataset(
#                         test_y,
#                         test_x),
#                     batch_size=batch_size,#512
#                     shuffle=True
#                     )

#params
epochs = 50



k = 10
kf = KFold(n_splits=k,shuffle=True,random_state=42)
fold_losses = []
for fold, (train_idx,test_idx) in enumerate(kf.split(t_actions)):
    print(f'\n=== Fold {fold+1}/{k} ===')

    #data splitting
    train_x, train_y = t_actions[train_idx], t_states[train_idx]
    test_x, test_y = t_actions[test_idx], t_states[test_idx]
    # print('train_y: ',train_y)
    # print('train_x: ',train_x)
    # print('test_y: ',test_y)
    # print('test_x: ',test_x)

    #Split loader
    train_loader =  DataLoader(TensorDataset(train_y,train_x),
                        batch_size=batch_size,#512
                        shuffle=True)

    test_loader =  DataLoader(TensorDataset(test_y,test_x),
                        batch_size=batch_size,#512
                        shuffle=True)


    #creat_model
    model = cVAE(x_dim=dim_x, y_dim=dim_y, z_dim=dim_z, hidden=[32,32])
    model.to(device)

    #create_optimizer
    optimiser = torch.optim.Adam(
        model.parameters(),lr=1e-3,weight_decay=1e-5)


    # best_test_loss = np.inf
    # best_epoch = 0
    # best_model = deepcopy(model.state_dict())
    loss_dict = dict(train=[],test=[])

    for epoch in range(1,epochs+1):
        # print('epoch: ', epoch)
        #train
        model.train()
        train_loss = 0.0
        for states_b, actions_b in train_loader: #(train_y,train_x) [states,action]
            states_b,actions_b  = states_b.to(device),actions_b.to(device) 
            loss = loss_function(model,actions_b,states_b) 
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            train_loss += loss.item()
        # train_loss /= len(train_loader) #train_states.shape[0] #length_total_training_dataset
        # print('train_len' , len(train_loader))
        # print('train_loss: ',train_loss)
        # loss_dict['train'].append(train_loss)

    #validate
    test_loss = 0.0
    model.eval()
    with torch.no_grad():
        for states_b,actions_b in test_loader:
            states_b,actions_b  = states_b.to(device),actions_b.to(device) 
            loss_1 = loss_function(model,actions_b,states_b)
            test_loss += loss_1.item() #* states_b.size(0) #batch_size
    test_loss /= len(test_loader) #test_states.shape[0] #length_total_training_dataset
    # print('test_loss: ',test_loss)
    # print('test_len' ,len(test_loader))
    # loss_dict['test'].append(test_loss)

        #save best model for use at the end
        # if test_loss < best_test_loss:
        #     best_epoch = epoch
        #     best_test_loss = test_loss
            # best_model = deepcopy(model.state_dict())

    fold_losses.append(test_loss)

    print(f'Fold {fold+1} validation loss: {test_loss:.6f}')

print('nCross-validation losses per fold:')
for i, L in enumerate(fold_losses):
    print(f' Fold {i+1}: {L:.6f}')

print(f"\n Mean CV loss: {torch.tensor(fold_losses).mean().item():.6f}")
print(f"\n Mean CV loss: {torch.tensor(fold_losses).std().item():.6f}")
xxxx

