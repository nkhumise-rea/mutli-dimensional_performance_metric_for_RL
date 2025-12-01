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

        #xxxxx


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
    
def loss_function(model,x,y,beta=0.01): #[actions,states]
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



##### testing #### 
n_samples = 100#0
dim_x = 4 #2 #4 #actions
dim_y = 10 #2 #0 #states
dim_z = 7 #14
batch_size = 256

#data
# X = torch.randn(n_samples,dim_x) #torch.randn(n_samples,dim_x)
# X = torch.ones_like(X)
# Y = torch.randn(n_samples,dim_y) 
# XY = torch.randn(n_samples,dim_y+dim_x)
# X = XY[:,:dim_x]
# Y = XY[:,dim_x:]
# print('X: ', X)
# print('Y: ', Y)

loc = torch.zeros(dim_x+dim_y)
scale = torch.rand(dim_x+dim_y,dim_x+dim_y)
sigma = torch.matmul(scale, scale.T)
sigma = torch.linalg.cholesky(sigma)

sigma2 = torch.matmul(scale*2, 2*scale.T)
sigma2 = torch.linalg.cholesky(sigma2)

mvn = MultivariateNormal(loc, scale_tril=sigma).rsample((n_samples,)) #torch.diag(scale)
mvn2 = MultivariateNormal(loc, scale_tril=sigma2).rsample((n_samples,)) #torch.diag(scale)
# print(mvn.shape)
# xxx
X = mvn[:,:dim_x]
Y = mvn[:,dim_x:]
X2 = mvn2[:,:dim_x]
Y2 = mvn2[:,dim_x:]

#prepare 80%/20% for training/testing
num_data = int(n_samples*.8) #80%
print('num_data: ',num_data)

#data splitting
train_y = Y[:num_data]
train_x = X[:num_data]
test_y = Y[num_data:]
test_x = X[num_data:]
# print('train_y: ',train_y)
# print('train_x: ',train_x)
# print('test_y: ',test_y)
# print('test_x: ',test_x)

#Split loader
train_loader =  DataLoader( 
                    TensorDataset(
                        train_y,
                        train_x),
                    batch_size=batch_size,#512
                    shuffle=True
                    )

test_loader =  DataLoader( 
                    TensorDataset(
                        test_y,
                        test_x),
                    batch_size=batch_size,#512
                    shuffle=True
                    )

#params
epochs = 150

#creat_model
model = cVAE(x_dim=dim_x, y_dim=dim_y, z_dim=dim_z, hidden=[32,32])
model.to(device)

#create_optimizer
optimiser = torch.optim.Adam(
    model.parameters(),lr=1e-2,weight_decay=1e-5)

best_test_loss = np.inf
best_epoch = 0
best_model = deepcopy(model.state_dict())
loss_dict = dict(train=[],test=[])
for epoch in range(1,epochs+1):
    print('epoch: ', epoch)
    #train
    model.train()
    train_loss = 0.0
    for states_b, actions_b in train_loader: #(train_y,train_x) [states,action]
        states_b,actions_b  = states_b.to(device),actions_b.to(device) 

        #loss_function(model,x,y): #[actions,states]
        loss = loss_function(model,actions_b,states_b) 
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        train_loss += loss.item()
    train_loss /= len(train_loader) #train_states.shape[0] #length_total_training_dataset
    # print('train_len' , len(train_loader))
    # print('train_loss: ',train_loss)
    loss_dict['train'].append(train_loss)

    #validate
    test_loss = 0.0
    with torch.no_grad():
        model.eval()
        for states_b,actions_b in test_loader:
            states_b,actions_b  = states_b.to(device),actions_b.to(device) 
            loss_1 = loss_function(model,actions_b,states_b)
            test_loss += loss_1.item() #* states_b.size(0) #batch_size
        test_loss /= len(test_loader) #test_states.shape[0] #length_total_training_dataset
        # print('test_loss: ',test_loss)
        # print('test_len' ,len(test_loader))
        loss_dict['test'].append(test_loss)

    #save best model for use at the end
    if test_loss < best_test_loss:
        best_epoch = epoch
        best_test_loss = test_loss
        # best_model = deepcopy(model.state_dict())

# print('test_loss: ', test_loss) 
print(f'best_test_loss: {best_test_loss}')

print(f'best_epoch: {best_epoch} | best_test_loss: {best_test_loss}')
plt.plot(loss_dict['train'], label='Train')
plt.plot(loss_dict['test'], label='Validation')
plt.xlabel('Epochs',fontweight='bold')
plt.ylabel('Loss',fontweight='bold')
plt.legend()
# plt.title('Loss curves')
plt.show()

xxxx

