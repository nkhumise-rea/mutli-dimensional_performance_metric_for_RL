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

#tianshou
from tianshou.policy import DDPGPolicy,SACPolicy,TD3Policy
from tianshou.utils.net.common import Net, get_dict_state_decorator
from tianshou.utils.net.continuous import Actor,ActorProb,Critic

#GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print('device: ', device)
# xxxx

parser = argparse.ArgumentParser()
parser.add_argument("--n_count", type=int, default=5)
parser.add_argument("--n_iteration", type=int, default=5)
args = parser.parse_args()

#set directory/location
this_dir = dirname(__file__)
agent_dir = abspath(join(this_dir,'..')) 
sys.path.insert(0,agent_dir)

##RL_Parameters
lr_actor = 1e-2
lr_critic = 1e-3
exploration_noise = 0.2 #ref: Plappert (2018)

##SAC
lr_sac = 1e-3
alpha = 0.2

##TD3
lr_actor_td3 = 1e-3 #3e-4 #ref: mujoco_td3.py
lr_critic_td3 = 1e-3 #3e-4 #ref: mujoco_td3.py

##ALL
steps_per_episode = 100 #100 #50
hidden_sizes_RL = [256,256,256] #ref: Plappert (2018)
tau = 50e-3 #ref: Plappert (2018)
gamma = 0.99
num_episodes = 500
render = None

##tasks
task = 'Reach' #'Push', 'Slide'
algo = 'ddpg' #'td3' #'sac'   
cnt = 5 
encoding_dim = 25 #

print(f'algo: {algo}, task: {task}')

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
    def __init__(self, x_dim, y_dim, z_dim=6, hidden=[32,32]):
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
        return mu_enc, logvar_enc, mu_dec #, logvar_dec, z

#Conditional VAE loss_function
def loss_function(model,x,y,beta=0.1): #[actions,states]
    mu_enc, logvar_enc, mu_dec = model(x,y)

    ##reconstruction_loss
    #if likelihood/Decoder ( p_(y|x,z) ) is Gaussian
    l1 = 0.5*((y-mu_dec)**2).sum(dim=1) # .mean()
    # print('l1: ', l1.mean())
    # print('l1: ', l1)

    ##regularisation: KL[ q_(z|x,y) || N(0,I)  ]
    sigma_enc = torch.exp(0.5*logvar_enc) #log(std) = 0.5*log(var)
    posterior_dist = Normal(mu_enc,sigma_enc) #approx. posterior
    prior_dist = Normal(torch.zeros_like(mu_enc),torch.ones_like(sigma_enc)) #prior
    
    l2 = kl_divergence(posterior_dist,prior_dist).sum(dim=1)
    # print('l2: ', l2)
    # print('l2: ', l2.mean())
    
    ##loss = -ELBO
    loss = beta*l2 + l1
    # print('loss: ', loss)
    # print('loss: ', loss.mean())
    # print('l2.mean() + l1.mean(): ', l2.mean() + l1.mean())

    return loss.mean()

class setting():
    def __init__(self):
        
        ##Environment
        head = "human" #"rgb_array" # 
        if task == 'Reach':
            self.env = gym.make('FetchReach-v4', render_mode=head,max_episode_steps=steps_per_episode)
        elif task == 'Push':
            self.env = gym.make('FetchPush-v4', render_mode=head, max_episode_steps=steps_per_episode)
            # env = gym.make('FetchPushDense-v4', render_mode=head, max_episode_steps=steps_per_episode)
        else:
            self.env = gym.make('FetchSlide-v4', render_mode=head, max_episode_steps=steps_per_episode)
            # env = gym.make('FetchSlideDense-v4', render_mode=head, max_episode_steps=100)

        #env shapes
        state_shape = {
            'observation' : self.env.observation_space['observation'].shape[0],
            'achieved_goal' : self.env.observation_space['achieved_goal'].shape[0],
            'desired_goal' : self.env.observation_space['desired_goal'].shape[0],
            }
        self.action_shape = self.env.action_space.shape[0] 
        self.max_action = self.env.action_space.high[0]
        self.sigma = self.max_action*exploration_noise
        
        self.dict_state_dec, self.flat_state_shape = get_dict_state_decorator (
            state_shape = state_shape,
            keys = ['observation','achieved_goal','desired_goal'],
            )

        self.num_states = state_shape['observation']
        self.num_actions = self.env.action_space.shape[0]
        
    # verify_that_final_policy_converged
    def verify_policy_converged(self,iteration=0):
        """
        Prints mean return and standard deviation over N trials
        """
        self.iteration = 5 #iteration #process_iteration_num

        #NN architecture
        num_states = self.num_states
        num_actions = self.num_actions

        #file_location
        file_location = f'{algo}/{task.lower()}/count_{cnt}/iter_{self.iteration}'
        print('file_location: ', file_location)
        # xxxx
        # print(os.listdir(file_location))
        # print(sorted(os.listdir(file_location)))
        # print(len(os.listdir(file_location)))
        # xxx

        step = 400 #1200 #len(os.listdir(file_location))-1
        print('step: ', step)

        # retrieve policy_model
        if algo == 'ddpg':
            self.pi_net = self.retrieve_policy_model_ddpg(
                                            num_actions, 
                                            hidden_sizes_RL,  
                                            step,
                                            file_location
                                            )
        elif algo == 'sac':
            self.pi_net = self.retrieve_policy_model_sac(
                                            num_actions, 
                                            hidden_sizes_RL,  
                                            step,
                                            file_location
                                            )
        elif algo == 'td3':
            self.pi_net = self.retrieve_policy_model_td3(
                                            num_actions, 
                                            hidden_sizes_RL,  
                                            step,
                                            file_location
                                            )
        else:
            print('Algorithm not registered')

        #evaluate_converged_policy
        self.pol_evals()

        return 

    # policy_data_generation
    def policy_data_generation(self,iteration=0):
        """
        Saves policy data as .npy file
        """
        self.iteration = args.n_iteration #iteration #process_iteration_num

        #NN architecture
        num_states = self.num_states
        num_actions = self.num_actions

        #file_location
        file_location = f'{algo}/{task.lower()}/count_{cnt}/iter_{self.iteration}'
        print('file_location: ', file_location)
        # xxxx
        # print(os.listdir(file_location))
        # print(sorted(os.listdir(file_location)))
        # print(len(os.listdir(file_location)))
        # xxx
        for step in range(len(os.listdir(file_location))-1):
            print('step: ', step)
            # retrieve policy_model
            if algo == 'ddpg':
                self.pi_net = self.retrieve_policy_model_ddpg(
                                                num_actions, 
                                                hidden_sizes_RL,  
                                                step,
                                                file_location
                                                )
            elif algo == 'sac':
                self.pi_net = self.retrieve_policy_model_sac(
                                                num_actions, 
                                                hidden_sizes_RL,  
                                                step,
                                                file_location
                                                )
            elif algo == 'td3':
                self.pi_net = self.retrieve_policy_model_td3(
                                                num_actions, 
                                                hidden_sizes_RL,  
                                                step,
                                                file_location
                                                )
            else:
                print('Algorithm not registered')

            # generate_roll-outs
            self.pol_rollouts(step)
        return 
    
    # policy_path_generation_in_policy_space
    def policy_path_generation(self,iteration=0):
        """
        Outputs and saves Wasserstein distances across successive 
        policies and between the final policy and 
        successive policies during training

        Args:
            iteration: int, 
        """
        self.iteration = args.n_iteration #iteration #process_iteration_num

        # execution  
        file_location = f'{algo}/{task.lower()}/samples_{cnt}/iter_{self.iteration}'        
        final_policy_data_num = len(os.listdir(file_location)) - 1 #optimal_policy
        initial_policy_data_num = 0 #initial_policy
        # print(file_location)
        # print('final_policy_data_num: ', final_policy_data_num)
        # print('initial_policy_data_num: ', initial_policy_data_num)

        # Wasserstein distance: W(pi_{0},pi_{N}) 
        start_time_outer = time.time()
        #-----------------------------------------
        # """        
        direct_distance = self.OTDD(
            initial_policy_data_num,
            final_policy_data_num,
            file_location)
        # """
        to_final_distance_list = [direct_distance] #[] #
        # to_final_distance_list = [None]
        # print('to_final_distance_list: ', to_final_distance_list)
        #-----------------------------------------
        # end_time_outer = time.time()
        # duration_outer = end_time_outer - start_time_outer
        # print(f' duration_outer: {duration_outer/60} min')

        # Wasserstein distance: \sum W(pi_{k},pi_{k+1})
        policy_data_num_first = 0
        successive_distance_list = []
        frequency = int((final_policy_data_num+1)/40) #50

        start_step = 1
        # for update in range(1,len(os.listdir(file_location))-1):
        for update in range(start_step,len(os.listdir(file_location)),frequency):
            print('update: ', update)
            policy_data_num_second = update #assign

            #OTDD between successive policies
            successive_distance = self.OTDD(
                policy_data_num_first,
                policy_data_num_second,
                file_location)    
            successive_distance_list.append(successive_distance) 

            #OTDD between final policy and others
            to_final_distance = self.OTDD(
                policy_data_num_second,
                final_policy_data_num,
                file_location)
            to_final_distance_list.append(to_final_distance) 

            #save policy trajectory datapoints
            policy_trajectory_data = np.array([
                successive_distance_list,
                to_final_distance_list
                ],dtype=object)
            self.save_policy_trajectories_per_update(policy_trajectory_data,update)
            
            #update policy number
            policy_data_num_first = policy_data_num_second   

        end_time_outer = time.time()
        duration_outer = end_time_outer - start_time_outer
        print(f' duration_outer: {duration_outer/60} min')
        return        

    #load and unpack dataset
    def dataset(self,steps,file_location):
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
    
    #action-to-action wasserstein distance
    def inner_wasserstein(self,
                          a1,a2,
                          model1,model2,
                          n_projections=500):
        """
        Outputs Wasserstein distance between action pairs (a1,a2)

        Args:
            a1: array, action sample from policy 1 
            a2: array, action sample from policy 2
        """
        states_given_a1 = self.sample_p_x_given_y(a1,model1) # s ~ P(S|a1)
        states_given_a2 = self.sample_p_x_given_y(a2,model2) # s ~ Q(S|a2)
        # print('states_given_a1: \n', states_given_a1)
        # print('states_given_a2: \n', states_given_a2)

        #rule-of-thumb (200 - 500 should be fine) 
        n_projections = max(100,4*states_given_a1.shape[1])
        # print('n_projections: ', n_projections)
  
        #wasserstein_distance
        W_sw = ot.sliced.sliced_wasserstein_distance(
            states_given_a1,
            states_given_a2,
            n_projections=n_projections,
            p=1, #p-wasserstein
            seed=42
        )
        # print('W_sw: ', W_sw)
        return W_sw

    #Optimal_Transport_Dataset_Distance
    def OTDD(self,steps_i,steps_j,file_location): #using_finite_episodes
        """
        Outputs Wasserstein distance between two policy dataset
        D_i = {(s_k,a_k)}_{k>=1} and D_j = {(s_k,a_k)}_{k>=1}

        Args:
            steps_i: int, policy number/address i 
            steps_j: int, policy number/address j
        """        
        # print('steps_i: ', steps_i)
        # print('steps_j: ', steps_j)

        #extract states anc actions samples
        states_i, actions_i = self.dataset(steps_i,file_location)
        states_j, actions_j = self.dataset(steps_j,file_location)

        #probability distribution of states given actions
        model_i = self.CVAE(states_i,actions_i)
        model_j = self.CVAE(states_j,actions_j)

        #mini-batch stochastic OT
        n_batches = 3 #5 #10
        num_samples = actions_i.shape[0] #states_i.shape[0]
        size_batch = 128 #256, 512
        wasserstein = 0

        # print('start loop')
        # start_time_outer = time.time()
        m = np.zeros([size_batch,size_batch]) #cost_matric
        for n_ in range(n_batches):
            idx = np.random.choice(num_samples,size_batch,replace=False)
            idy = np.random.choice(num_samples,size_batch,replace=False)
            states_i_batch = states_i[idx]
            actions_i_batch = actions_i[idx]
            states_j_batch = states_j[idy]
            actions_j_batch = actions_j[idy]

            # start_time_inner = time.time()
            for id, i in enumerate(zip(states_i_batch,actions_i_batch)):
                for jd,j in enumerate(zip(states_j_batch,actions_j_batch)):
                    d_a = self.inner_wasserstein(i[1], #action_i
                                                 j[1], #action_j
                                                 model_i,
                                                 model_j)
                    # print('d_a: ', d_a)
                    if d_a > 1000:
                        # print('d_a: ', d_a)
                        # print(f'i[1]: {i[1]} and j[1]: {j[1]}')
                        print('excessive action-to-action distance!')
                        xxx
                    d_s = LA.norm(i[0] - j[0]) #state_i - state_j
                    
                    m[id,jd] = d_a + d_s
            # print('m: ', m.shape)
            a = b = np.ones(size_batch)/size_batch
            # batch_wasserstein = ot.sinkhorn2(a,b,m,reg=0.1)
            # print('batch_wasserstein: ', batch_wasserstein)

            val = ot.emd2(a,b,M=m) #OT matrix
            # print('val: ', val)
            
            # end_time_inner = time.time()
            # duration_inner = end_time_inner - start_time_inner
            # print(f'{n_}. duration_inner: {duration_inner/60} min')
            # xxxx
            wasserstein += val #batch_wasserstein 
        wasserstein /= n_batches
        # print('wasserstein: ', wasserstein)

        # end_time_outer = time.time()
        # duration_outer = end_time_outer - start_time_outer
        # print(f' duration_outer: {duration_outer/60} min')
        # xxx
        return wasserstein 

    #conditional distribution Q(States | Actions)
    def CVAE(self,states,actions,batch_size=512,epochs=25,lr=1e-3):
        """
        Outputs probability distribution of states given
        action i.e. Q(S|A)

        Args:
            states: array, state samples from policy
            action: array, action samples from policy 
        """
        dim_states = states.shape[1]
        dim_actions = actions.shape[1]
        num_samples = 20000 #states.shape[0]

        #convert to torch        
        t_states = torch.tensor(states,dtype=torch.float32)
        t_actions = torch.tensor(actions,dtype=torch.float32)
        # print('t_states: ',t_states.shape)
        # print('t_actions: ',t_actions.shape)

        #prepare 80%/20% for training/testing
        num_data = int(num_samples*.8) #80%
        # print('num_data: ', num_data)

        #data splitting
        train_states,train_actions = t_states[:num_data],t_actions[:num_data]
        test_states,test_actions = t_states[num_data:],t_actions[num_data:]
        # print('train_states: ',train_states.shape)
        # print('test_states: ',test_states.shape)
        # print('train_actions: ',train_actions.shape)
        # print('test_actions: ',test_actions.shape)


        #Split loader
        train_loader =  DataLoader(TensorDataset(train_states,train_actions),
                            batch_size=batch_size,#512
                            shuffle=True)

        test_loader =  DataLoader(TensorDataset(test_states,test_actions),
                            batch_size=batch_size,#512
                            shuffle=True)
        
        #create model
        model = cVAE(x_dim=dim_actions,y_dim=dim_states)
        
        #create optimiser
        optimiser = torch.optim.Adam(model.parameters(),
                                     lr=lr,
                                     weight_decay=1e-5)
        model.to(device)

        #params
        best_test_loss = np.inf
        best_epoch = 0
        best_model = deepcopy(model.state_dict())
        loss_dict = dict(train=[], test=[])
        for epoch in range(1,epochs+1):
            # print('epoch: ', epoch)
            #train
            model.train()
            train_loss = 0.0
            for states_b,actions_b in train_loader:
                states_b,actions_b  = states_b.to(device),actions_b.to(device) 
                loss = loss_function(model,actions_b,states_b) 
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
                train_loss += loss.item() #* states_b.size(0) #batch_size
            train_loss /= len(train_loader) #train_states.shape[0] #length_total_training_dataset
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
            loss_dict['test'].append(test_loss)

            #save best model for use at the end
            if test_loss < best_test_loss:
                best_epoch = epoch
                best_test_loss = test_loss
                best_model = deepcopy(model.state_dict())
        
        error_rate = (np.array(loss_dict['test']).min()-np.array(loss_dict['train']).min())*100/(np.array(loss_dict['train']).max()-np.array(loss_dict['train']).min())
        # print('test_loss: ', test_loss) 
        # print(f'best_test_loss: {best_test_loss}')
        # print(f'error_rate: {error_rate}')

        """    
        #visualization of loss        
        #print loss every n epoches
            # if(epoch)%10==0:
            #     # print(f"Epoch {epoch+1:02d}/{epochs}, NLL: {train_loss:.4f}")
            #     print(f'{epoch}: Train loss: {train_loss:1.4f}')
            #     print(f'{epoch}: Test loss: {test_loss:1.4f}')

        print(f'best_epoch: {best_epoch} | best_test_loss: {best_test_loss}')
        plt.plot(loss_dict['train'], label='Train')
        plt.plot(loss_dict['test'], label='Validation')
        plt.xlabel('Epochs',fontweight='bold')
        plt.ylabel('Loss',fontweight='bold')
        plt.legend()
        # plt.title('Loss curves')
        plt.show()
        # """

        # xxx

        model.load_state_dict(best_model)   
        return model # Q(X|Y) - conditional distribution

    #samples from conditional distribution
    def sample_p_x_given_y(self,y_query,model,n_samples=500): #1024
        """
        Sample from conditional model q_theta(x|y)

        Args:
            n_samples: int, number of samples to draw per y_query
        """
        model.eval()
        y = torch.tensor(y_query,dtype=torch.float32).to(device)
        y_q = y.unsqueeze(0) #.repeat(n_samples,1) #n_samples

        # sample z ~ p(z) (standard normal)
        with torch.no_grad():
            z = torch.randn(n_samples, model.z_dim).to(device)
            y_expand = y_q.expand(n_samples,-1)
            mu_x, logvar_x = model.decode(z,y_expand)

            # sample x ~ N(mu_x,diag(var))
            std_x = torch.exp(0.5*logvar_x)
            eta = torch.randn_like(std_x) #torch.randn_like(mu_x)
            x_samples = mu_x + eta*std_x
            x_ = x_samples.squeeze(1).cpu().numpy()   
            # print('x_: ', x_.shape)
            # print('min/max: ', x_.min(), x_.max())
            # print('mean/std: ', x_.mean(), x_.std())
            # print('=================================')
            # x
        return x_ #np.array(x_,dtype='float64')

    #~~~~~~~~~~~~~~Saving Policy Trajectories ~~~~~~~~~~~~~~~~~~~~

    # saving_policy_trajectories (fast_Track)
    def save_policy_trajectories_per_update(self,meta_data,update):
        meta_data = np.asarray( meta_data, dtype=object) #data_to_save

        file_name = f'traj_till{update}'
        file_location = f'{algo}/{task.lower()}/path_{cnt}/iter_{self.iteration}/dim_{encoding_dim}_100k'
        
        os.makedirs(join(this_dir,file_location), exist_ok=True) #create_file_location_if_nonexistent
        file_path = abspath(join(this_dir,file_location,file_name)) #file_path: location + name
        np.save(file_path,meta_data) #save_command
        return

    #~~~~~~~~~~~~~~ Retrieving Policies & Saving Samples ~~~~~~~~~~~~~~~~~~~~

    # retrieve policy model
    def retrieve_policy_model_ddpg(self,
                            num_actions, 
                            num_hidden, 
                            steps,
                            file_location):
        
        # steps = 1998
        file_name = f'actorDS_{steps}.pth'     
        file_path = abspath(join(this_dir,file_location,file_name)) #file_path: location + name

        #declare model
        #--preprocessing networks
        net_a = self.dict_state_dec(Net)(
            self.flat_state_shape,
            hidden_sizes=num_hidden, 
            device=device)

        #--Actor, 2 x Critics networks
        actor = self.dict_state_dec(Actor)( #deterministic actor
            net_a,
            num_actions,
            max_action = self.max_action,
            device = device).to(device)
        
        net_c = self.dict_state_dec(Net)(
            self.flat_state_shape,
            num_actions, 
            hidden_sizes=num_hidden, 
            concat = True, 
            device=device)

        #--Actor, 2 x Critics networks
        actor = self.dict_state_dec(Actor)( #deterministic actor
            net_a,
            num_actions,
            max_action = self.max_action,
            device = device).to(device)

        critic = self.dict_state_dec(Critic)(net_c,device = device).to(device)
        actor_optim = torch.optim.Adam(actor.parameters(),lr=lr_actor)
        critic_optim = torch.optim.Adam(critic.parameters(),lr=lr_critic)

        policy = DDPGPolicy(
            actor,
            actor_optim,
            critic,
            critic_optim,
            tau=tau,
            gamma=gamma,
            exploration_noise=None,
            estimation_step=1,
            )

        ##load_trained_policy
        policy.load_state_dict(torch.load(file_path, map_location=torch.device('cpu')))                
        policy.eval()
        return policy

    def retrieve_policy_model_sac(self,
                            num_actions, 
                            num_hidden, 
                            steps,
                            file_location):
        
        # steps = 1998
        file_name = f'actorDS_{steps}.pth'     
        file_path = abspath(join(this_dir,file_location,file_name)) #file_path: location + name

        #--preprocessing networks
        net_a = self.dict_state_dec(Net)(
            self.flat_state_shape,
            hidden_sizes=num_hidden, 
            device=device)

        net_c1 = self.dict_state_dec(Net)(
            self.flat_state_shape,
            num_actions, 
            hidden_sizes=num_hidden,
            concat = True, 
            device=device)

        net_c2 = self.dict_state_dec(Net)(
            self.flat_state_shape,
            self.action_shape, 
            hidden_sizes=num_hidden,
            concat = True, 
            device=device)

        #--Actor, 2 x Critics networks
        actor = self.dict_state_dec(ActorProb)(
            net_a,
            num_actions,
            max_action = self.max_action,
            device = device,
            unbounded = True,
            conditioned_sigma = True).to(device)

        critic1 = self.dict_state_dec(Critic)(net_c1,device = device).to(device)
        critic2 = self.dict_state_dec(Critic)(net_c2,device = device).to(device)

        policy = SACPolicy(
                actor=actor,
                actor_optim=None,
                critic1=critic1,
                critic1_optim=None,
                critic2=critic2,
                critic2_optim=None,
                tau=tau,
                gamma=gamma,
                alpha=alpha,
            )

        ##load_trained_policy
        policy.load_state_dict(torch.load(file_path, map_location=torch.device('cpu')))                
        policy.eval()
        return policy
   
    def retrieve_policy_model_td3(self,
                            num_actions, 
                            num_hidden, 
                            steps,
                            file_location):
        
        # steps = 1998
        file_name = f'actorDS_{steps}.pth'     
        file_path = abspath(join(this_dir,file_location,file_name)) #file_path: location + name

        #--preprocessing networks
        net_a = self.dict_state_dec(Net)(
            self.flat_state_shape,
            hidden_sizes=num_hidden, 
            device=device)

        net_c1 = self.dict_state_dec(Net)(
            self.flat_state_shape,
            num_actions, 
            hidden_sizes=num_hidden,
            concat = True, 
            device=device)

        net_c2 = self.dict_state_dec(Net)(
            self.flat_state_shape,
            num_actions, 
            hidden_sizes=num_hidden,
            concat = True, 
            device=device)

        #--Actor, 2 x Critics networks
        actor = self.dict_state_dec(Actor)(
            net_a,
            num_actions,
            max_action = self.max_action,
            device = device).to(device)
            
        critic1 = self.dict_state_dec(Critic)(net_c1,device = device).to(device)
        critic2 = self.dict_state_dec(Critic)(net_c2,device = device).to(device)
        actor_optim = torch.optim.Adam(actor.parameters(),lr=lr_actor_td3)
        critic1_optim = torch.optim.Adam(critic1.parameters(),lr=lr_critic_td3)
        critic2_optim = torch.optim.Adam(critic2.parameters(),lr=lr_critic_td3)

        policy = TD3Policy(    
            actor=actor,
            actor_optim=actor_optim,
            critic1=critic1,
            critic1_optim=critic1_optim,
            critic2=critic2,
            critic2_optim=critic2_optim,
            tau=tau,
            gamma=gamma,
            exploration_noise=None,
            estimation_step=1,
            )

        ##load_trained_policy
        policy.load_state_dict(torch.load(file_path, map_location=torch.device('cpu')))                
        policy.eval()
        return policy

    #preprocessing_Observations
    def preprocesses_obs(self,obs_dict):
        parts = []
        for _, val in obs_dict.items():
            # print('val: ', val)
            parts.append(val)
        obs_vec = np.concatenate(parts,axis=0)
        # print('obs_vec: ', obs_vec)

        ############# OTDD ############
        # states/observations use for OTDD
        otdd_obs = obs_vec[:self.num_states].tolist()
        # print(otdd_obs)

        return torch.as_tensor(obs_vec,dtype=torch.float32).unsqueeze(0), otdd_obs

    #trained_agent (for: evaluation)
    def learnt_agent(self):
        ##Environment
        head = "rgb_array" #"human" # 
        if task == 'Reach':
            self.env_test = gym.make('FetchReach-v4', render_mode=head,max_episode_steps=steps_per_episode)
        elif task == 'Push':
            self.env_test = gym.make('FetchPush-v4', render_mode=head, max_episode_steps=steps_per_episode)
            # env = gym.make('FetchPushDense-v4', render_mode=head, max_episode_steps=steps_per_episode)
        else:
            self.env_test = gym.make('FetchSlide-v4', render_mode=head, max_episode_steps=steps_per_episode)
            # env = gym.make('FetchSlideDense-v4', render_mode=head, max_episode_steps=100)

        rew_list, act_list, st_list, trj = [],[],[],[]
        t_done = False
        ep_return = 0 # episode return
        t_obs, _ = self.env_test.reset() #reset environment
        # print(t_obs)
        # xx
        
        t_state,otdd_state = self.preprocesses_obs(t_obs)
        # print('batch2: ', t_state)
        # print('otdd_state: ', otdd_state)
        # xxx

        while not t_done:
            t_action = self.select_action(t_state)  
            # print('t_action: ', t_action)

            t_next_obs, t_reward, t_terminated, t_truncated, _ = self.env_test.step(t_action)
            # print('t_next_obs: ', t_next_obs)

            t_done = t_terminated or t_truncated #done_function (new_approach)
            ep_return += t_reward
            t_next_state,otdd_next_state = self.preprocesses_obs(t_next_obs)
            # print('batch2: ', t_next_state)
            # print('otdd_next_state: ', otdd_next_state)
            # xx
            
            #experience_buffer_data
            st_list.append(otdd_state) #stt_list
            rew_list.append(t_reward) #rew_list

            if t_done:
                trj = deepcopy(st_list)
                trj.append(otdd_next_state)

            act_list.append(t_action.tolist())
            t_state = t_next_state
            otdd_state = otdd_next_state
        
        self.env_test.close() #shut_env

        # results_display
        # print('returns: ', ep_return)        
        # print('st_list: ', st_list)
        # print('trj: ', trj)
        # print('act_list: ', act_list)
        # print('rew_list: ', rew_list)
        # xxx

        return rew_list , act_list, ep_return, st_list, trj

    # policy_rollouts
    def pol_rollouts(self,steps):
        """        
        Outputs list of states, actions, rewards, len(states), 
        len(actions)

        Args:
            steps: int, policy number 
        """
        rew_list, act_list, stt_list, ret_list = [],[],[],[]
        act_du,stt_du = [0],[0]

        while act_du[-1] <= 100000: #5000 #num_decisions
            rews,acts,ret,_,trj = self.learnt_agent() 
            rew_list += rews #immediate_rewards
            act_list += acts #actions
            stt_list += trj #states
            ret_list.append(ret) #returns
            act_du.append(len(act_list)) #size_action_list
            stt_du.append(len(stt_list)) #size_state_list

        meta_data = [stt_list,act_list,rew_list,ret_list,stt_du,act_du]
        # print('meta_data: ', meta_data)
        self.save_state_action_samples(meta_data,steps) #save_state-action_samples
        return 
    
    # saving_state-action_samples
    def save_state_action_samples(self,meta_data,steps):
        meta_data = np.asarray( meta_data, dtype=object) #data_to_save

        file_name = f'sample_{steps}'
        # file_location = f'{algo}/{task.lower()}/samples_{cnt}/iter_{self.iteration}/dim_{encoding_dim}'
        file_location = f'{algo}/{task.lower()}/samples_{cnt}/iter_{self.iteration}'
        # print('file_location: ', file_location)
        
        os.makedirs(join(this_dir,file_location), exist_ok=True) #create_file_location_if_nonexistent
        file_path = abspath(join(this_dir,file_location,file_name)) #file_path: location + name
        # print('file_path: ', file_path)

        np.save(file_path,meta_data) #save_command
        return

    #action_selection_mechanism   
    def select_action(self,batch):
        action = self.get_action_deterministically(batch)
        return action

    #greedy_action
    def get_action_deterministically(self,batch):
        with torch.no_grad():
            if algo == 'sac':
                action = self.pi_net.actor(batch)[0][0].detach().cpu().numpy()[0]
            else:
                action = self.pi_net.actor(batch)[0].detach().cpu().numpy()[0]
        # print(action)
        return action

    #preprocessing_Observations_for_converged_policy
    def preprocesses_obs_converged(self,obs_dict):
        parts = []
        for _, val in obs_dict.items():
            # print('val: ', val)
            parts.append(val)
        obs_vec = np.concatenate(parts,axis=0)
        # print('obs_vec: ', obs_vec)
        return torch.as_tensor(obs_vec,dtype=torch.float32).unsqueeze(0)

    #converged_agent (for: evaluation)
    def converged_agent(self):
        ##Environment
        head = "rgb_array" #"human" # 
        self.env_test = gym.make('FetchReach-v4', 
                                 render_mode=head,
                                 max_episode_steps=steps_per_episode)

        # print('steps_per_episode: ', steps_per_episode)

        t_done = False
        ep_return = 0 # episode return
        t_obs, _ = self.env_test.reset() #reset environment
        # print(t_obs)
        # xx
        
        t_state = self.preprocesses_obs_converged(t_obs)
        # print('batch2: ', t_state)
        # print('otdd_state: ', otdd_state)
        # xxx

        while not t_done:
            t_action = self.select_action(t_state)  
            # print('t_action: ', t_action)

            t_next_obs, t_reward, t_terminated, t_truncated, _ = self.env_test.step(t_action)
            # print('t_next_obs: ', t_next_obs)

            t_done = t_terminated or t_truncated #done_function (new_approach)
            ep_return += t_reward
            t_next_state = self.preprocesses_obs_converged(t_next_obs)
            # print('batch2: ', t_next_state)
            # print('otdd_next_state: ', otdd_next_state)
            # xx
            
            t_state = t_next_state
        
        self.env_test.close() #shut_env

        # results_display
        # print('returns: ', ep_return)        
        # print('st_list: ', st_list)
        # print('trj: ', trj)
        # print('act_list: ', act_list)
        # print('rew_list: ', rew_list)
        # xxx

        return ep_return       

    # policy_evaluations
    def pol_evals(self):
        """        
        Outputs mean return and standard deviation over N trials 
        """
        ep_return_list = []
        for _ in range(50): #5000 #num_decisions
            ep_return = self.converged_agent() 
            ep_return_list.append(ep_return)
        
        # print('ep_return_list: ',ep_return_list)
        ep_return_list = np.array(ep_return_list)
        # print('ep_return_list: ',ep_return_list)

        return_mean = ep_return_list.mean()
        return_std = ep_return_list.std()

        print(f'{return_mean} +/- {return_std}')
        return
    
    
    #~~~~~~~~~~~~~~ ########################### ~~~~~~~~~~~~~~~~~~~~

    #~~~~~~~~~~~~~~ Results and Visualisation ~~~~~~~~~~~~~~~~~~~~

    def policy_path_evaluation(self,iteration=2):
        
        # data_extraction
        self.iteration = 2 #5 #iteration
        num_update = 641 #391
        file_name = f'traj_till{num_update}.npy'
        file_location = f'{algo}/{task.lower()}/path_{cnt}/iter_{self.iteration}/dim_{encoding_dim}_100k'
        file_path = abspath(join(this_dir,file_location,file_name)) #file_path: location + name
        data = np.load(file_path, allow_pickle=True)   
        # print('data: \n', data)
        
        # data_category
        p = 6 #float_precision (decimal_places)
        chords = np.round(data[0],p) #y_k (stepwise distance)
        radii = np.round(data[1],p) #x_k (distance_to_optimal)
        chords = np.append(chords,radii[-1])
        # print('successive_distance_list: \n', len(chords))
        # print('to_final_distance_list: \n', len(radii))
        # xx

        direct_path = radii[0] #geodesic_distance
        # print('direct_path: ', direct_path)

        # data_collection
        curve_dis = np.round(sum(chords),p) # sum_{y_k}
        radii_diff = self.successive_diffs(radii,p) # x_k - x_{k+1}
        
        # =======================================
        # Tabular Data
        # =======================================
        #distance-to-optimal reduced
        positive_indices = np.where(radii_diff > 0) # improving transition
        positive_indices = positive_indices[0]

        # non_positive_chords = np.array([ chords[i] for i in non_positive_indices])
        positive_chords = np.array([ chords[i] for i in positive_indices])

        # wasted_effort = non_positive_chords.sum() #paid_to_further_from_goal
        usedful_effort = positive_chords.sum() #paid_to_closer_to_goal
        # print('wasted_effort: ', wasted_effort)
        # print('usedful_effort: ', usedful_effort)

        print('ALGO: ', algo)
        print('ESL: ', np.round(curve_dis/direct_path,3)) #effort_of_sequential_learning
        print('OMR: ', np.round(usedful_effort/curve_dis,3)) #optimal_movement_ratio
        print('UC: ', num_update)
        # =======================================
        xxx

        # data_plotting
        plt.figure(figsize=(16, 10))

        ## Radius behaviour (1) 
        plt.subplot(3,2,1)
        plt.plot(radii/max(radii),'g')
        plt.plot(self.running_avg(radii/max(radii),0.69),'r')
        plt.ylabel('relative dis_to_opt',fontweight='bold')
        plt.xlabel('updates',fontweight='bold')
        plt.title(f'Radius_Behaviour [{algo}]',fontweight='bold')

        ## Chord behaviour (2) 
        plt.subplot(3,2,2)
        plt.plot(chords/max(chords),'b')
        plt.plot(self.running_avg(chords/max(chords),0.69),'r')
        plt.ylabel('relative stepwise_dis',fontweight='bold')
        plt.xlabel('updates',fontweight='bold')
        plt.title(f'Chord_Behaviour [{algo}]',fontweight='bold')

        # plt.show()
        # xxx

        ## Chords vs Radius (3)
        plt.subplot(3,2,3)
        opt_Xs = radii 
        opt_Ys = chords 
        opt_time = np.arange(1,1999,50) #len(opt_Xs))
        opt_time = np.append(opt_time,2001) #1999
        # print('opt_Xs: ', len(opt_Xs))
        # print('opt_Ys: ', len(opt_Ys))
        # print('opt_time: ', opt_time)
        # print('opt_time: ', len(opt_time))
        # xxx
    
        #coloured_segments
        points = np.array([opt_Xs,opt_Ys]).T.reshape(-1,1,2)
        segments = np.concatenate([points[:-1],points[1:]],axis=1)
        lc = LineCollection(segments,
                            cmap='plasma',
                            norm=plt.Normalize(opt_time.min(),opt_time.max()),
                            alpha=0.2)
        lc.set_array(opt_time)
        plt.gca().add_collection(lc) 

        #plot
        scatter = plt.scatter(opt_Xs,opt_Ys,c=opt_time,
                                cmap='plasma',
                                edgecolor='k'
                                ) #cmap='viridis'
        cbar = plt.colorbar(scatter)
        cbar.set_label('Updates',fontweight='bold')                
        plt.xlabel('Radius',fontweight='bold')
        plt.ylabel('Chord',fontweight='bold')
        plt.title(f'Radius_vs_Chord [{algo}]',fontweight='bold')

        # plt.show()
        # xxx

        ## Chord vs Changing Radii (4)
        plt.subplot(3,2,4)
        XX = radii_diff
        YY = chords[:-1] #[:num_eval-1]
        tt = np.arange(1,1999,50) #len(XX))
        # print('XX: ', len(XX))
        # print('YY: ', len(YY))
        # print('tt: ', tt)
        # print('len(tt): ', len(tt))

        #coloured_segments
        points = np.array([XX,YY]).T.reshape(-1,1,2)
        segments = np.concatenate([points[:-1],points[1:]],axis=1)
        lc = LineCollection(segments,cmap='plasma',norm=plt.Normalize(tt.min(),tt.max()),alpha=0.2)
        lc.set_array(tt)
        plt.gca().add_collection(lc)

        #plots
        scat = plt.scatter(XX,YY,c=tt,cmap='plasma',edgecolor='k') #cmap='viridis'
        cbar = plt.colorbar(scat)
        cbar.set_label('Updates',fontweight='bold')

        plt.ylabel('Chord',fontweight='bold')
        plt.xlabel('$\delta$[Radii]',fontweight='bold')
        plt.title(f'$\delta$[Radii]_vs_Chords [{algo}]',fontweight='bold')
        # plt.xlim(-.7,.7)

        plt.vlines(x = 0, #convergence_line
            ymin=min(YY),
            ymax=max(YY), 
            colors='black', 
            ls=':',)

        plt.tight_layout()
                
        ## 3D Policy Trajectory Visualization (5)
        Xs = radii/max(radii) #radii 
        Ys = chords/max(chords) #chords 
        self._3Dplots(Xs,Ys,name=f'{algo}') 

        plt.show()
        return

    # running average function
    def running_avg(self,score_col,beta = 0.99):
        cum_score = []
        run_score = score_col[0]
        for score in score_col:
            run_score = beta*run_score + (1.0 - beta)*score
            cum_score.append(run_score)
        return cum_score
    
    # 3Dplots (chords vs radii vs time)
    def _3Dplots(self,Xs,Ys,name='',fig=None,pos=None):
        time = self.update_steps #np.arange(1,2051,50) #len(Xs))
        # print('time: ', time)
        # print('len(time): ', len(time))

        if fig == None:
            fig = plt.figure()
            ax = fig.add_subplot(111,projection='3d')
        else:
            ax = fig.add_subplot(pos,projection='3d')

        scatter = ax.scatter(Xs,Ys,
                             time,
                             c=time,
                             cmap='plasma',
                            #  edgecolor='k'
                             ) #cmap='viridis'
        ax.plot(Xs,Ys,time,c = 'k',alpha=1)

        cbar = plt.colorbar(scatter)
        cbar.set_label('#updates',fontweight='bold',fontsize=16)
        cbar.set_ticks([])

        # plt.show()

        ax.set_zlabel('#updates',fontweight='bold',fontsize=16)
        ax.set_xlabel('distance_to_optimal',fontweight='bold',fontsize=16)
        ax.set_ylabel('stepwise_distance',fontweight='bold',fontsize=16)
        # ax.set_xlabel('distance-to-optimal',fontweight='bold')
        # ax.set_ylabel('stepwise-distance',fontweight='bold')
        
        ax.view_init(elev=20.,azim=-120)
        
        ax.set_xlim(0,4.0) #observed across all algos
        ax.set_ylim(0,4.0) #observed across all algos

        # ax.set_xlim(0,max(Xs))
        # ax.set_ylim(0,max(Ys))
        # print(f'x_max: {max(Xs)}, y_max: {max(Ys)}')

        ax.invert_zaxis()
        ax.set_title(f'{name}',fontweight='bold',fontsize=16)
        ax.tick_params(labelsize=14)
        ax.ticklabel_format(axis='z',style='sci',scilimits=[0,2])
        # ax.set_visible(False)
        plt.tight_layout()      
        return  
    
    #successive_differences [radii (distance_to_optimal)]
    def successive_diffs(self,x_col,p): #radii (dis_to_optim)
        #x_col[:-1] -> {0:N-1} : x_k
        #x_col[1:] -> {1:N} : x_{k+1}
        x_k_1 = x_col[1:] #x_{k+1}
        x_k = x_col[:-1] #x_{k}
        x_diff = np.round(x_k - x_k_1,p) # x_k - x_{k+1}#
        
        return x_diff 
    
    #successive_area (areal_velocity)
    def successive_area(self,y_col,x_col,p):

        #x_col[:-1] -> {0:N-1} : x_k
        #x_col[1:] -> {1:N} : x_{k+1}
        #y_col[:-1] -> {0:N-1} : y_k
        x_k = x_col[:-1] #a
        x_k_1 = x_col[1:] #b
        y_k = y_col[:-1] #c
        eta = 1e-4 #small value to avoid zero division
        
        #compute areas
        s_k = (x_k+x_k_1+y_k)*0.5 # s = (a+b+c)/2

        d = s_k - y_k
        e = s_k - x_k_1
        f = s_k - y_k

        d = np.where(np.abs(d)>eta,d,0)        
        e = np.where(np.abs(e)>eta,e,0)
        f = np.where(np.abs(f)>eta,f,0)

        area_k = np.round((s_k*d*e*f)**0.5,p) #A = (s(s-a)(s-b)(s-c))^.5
        return area_k
    
    #temporal OMR
    def temporal_OMR(self,chords,radii,p):

        # print('len(chords): ', len(chords))
        T = int(len(chords)*.1)
        # print('T: ', T)
        # xxx
        segments = 12
        div = 1 #len(chords)//segments
        omr,ups  = [],[]
        for k in range(len(chords)-T+1):
            # if k%div == 0 and k <= (len(chords)-T) :
            # if k <= (len(chords)-T) :
            #     # print(i)

                # data_collection
                n_chords = chords[k:]
                n_radii = radii[k:]
                # curve_dis = np.round(sum(n_chords),p) # sum_{y_k} #[i:]
                radii_diff = self.successive_diffs(n_radii,p) # x_k - x_{k+1}
            
                #distance-to-optimal not reduced
                non_positive_indices = np.where(radii_diff <= 0) # non-improving transition 
                non_positive_indices = non_positive_indices[0]

                #distance-to-optimal reduced
                positive_indices = np.where(radii_diff > 0) # improving transition
                # print(positive_indices)
                # print(positive_indices[0])
                # xxx
                
                positive_indices = positive_indices[0]
                
                # test = len(positive_indices)+len(non_positive_indices)
                # print(len(positive_indices), len(non_positive_indices), len(positive_indices)/test )
                # # xxx

                non_positive_chords = np.array([ n_chords[i] for i in non_positive_indices])
                positive_chords = np.array([ n_chords[i] for i in positive_indices])

                wasted_effort = non_positive_chords.sum() #paid_to_further_from_goal
                usedful_effort = positive_chords.sum() #paid_to_closer_to_goal
                # print('wasted_effort: ', wasted_effort)
                # print('usedful_effort: ', usedful_effort)
                total_travel = wasted_effort + usedful_effort

                # print(curve_dis - total_travel)

                # print('OMR: ', np.round(usedful_effort/total_travel,3))
                # omr.append(usedful_effort/curve_dis)
                omr.append(usedful_effort/total_travel)               
                ups.append(k*self.sampling_factor)
                # print('k: ', k)
                # print('k*c: ', k*2500)
        # print('omr: ', omr)
        # print('ups: ', ups)
        # xxx
        return omr, ups 
    
    # saving plots
    def save_images_for_paper(self,iteration=0):
        # data_extraction
        self.iteration = 5 #2 #iteration
        num_update = 391 #641 #
        file_name = f'traj_till{num_update}.npy'
        file_location = f'{algo}/{task.lower()}/path_{cnt}/iter_{self.iteration}/dim_{encoding_dim}_100k'
        file_path = abspath(join(this_dir,file_location,file_name)) #file_path: location + name
        data = np.load(file_path, allow_pickle=True)   
        # print('data: \n', data)
        
        # data_category
        p = 6 #float_precision (decimal_places)
        chords = np.round(data[0],p) #y_k (stepwise distance)
        radii = np.round(data[1],p) #x_k (distance_to_optimal)
        chords = np.append(chords,radii[-1])
        print('chords: \n', len(chords))
        print('radii: \n', len(radii))


        self.sampling_factor = 1000 #1600 #2500
        total_num_updates = 40000 #65000 #100000
        self.update_steps = [i for i in range(0,total_num_updates+self.sampling_factor,self.sampling_factor)]
        print('update_steps: ', self.update_steps)
        print('update_steps: ', len(self.update_steps))
        # xxx

        direct_path = radii[0] #geodesic_distance
        # print('direct_path: ', direct_path)

                
        # data_collection
        curve_dis = np.round(sum(chords),p) # sum_{y_k}
        radii_diff = self.successive_diffs(radii,p) # x_k - x_{k+1}
        areal = self.successive_area(chords,radii,p)
        # print('areal: ', areal)
        
        OMR,updates = self.temporal_OMR(chords,radii,p)
        # print('curve_dis: ', curve_dis)
        # print('radii_diff: \n', radii_diff)

        
        # =======================================
        # Thesis Plots
        # =======================================
        ## Radius behaviour (1) 
        plt.figure(figsize=(6,3))
        plt.plot(self.update_steps,radii,'g',alpha=1.0)
        # plt.plot(self.running_avg(radii,0.69),'r')
        plt.ylabel('distance_to_optimal',fontweight='bold',fontsize=16)
        plt.xlabel('#updates',fontweight='bold',fontsize=16)
        plt.ticklabel_format(axis='x', style='sci', scilimits=[0,2])
        plt.ylim(0,4)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.tight_layout()
        file_name2 = f'{algo}_2optimal.png'
        file_path2 = abspath(join(this_dir,'exploration/reach',file_name2)) #file_path: location + name
        plt.savefig(file_path2, format='png') 

        ## Chord behaviour (2)
        plt.figure(figsize=(6,3))
        plt.plot(self.update_steps,chords,'b',alpha=1.0)
        # plt.plot(self.running_avg(chords,0.69),'r')
        plt.ylabel('stepwise_distance',fontweight='bold',fontsize=16)
        plt.xlabel('#updates',fontweight='bold',fontsize=16)
        plt.ticklabel_format(axis='x', style='sci', scilimits=[0,2])
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.ylim(0,4)      
        plt.tight_layout()
        file_name3 = f'{algo}_stepwise.png'
        file_path3 = abspath(join(this_dir,'exploration/reach',file_name3)) #file_path: location + name
        plt.savefig(file_path3, format='png') 


        """        
        ## Areal Velocity behaviour (3) 
        plt.figure(figsize=(6,3))
        plt.plot(update_steps,areal,'k',alpha=1.0)
        # plt.plot(self.running_avg(areal,0.69),'r')
        plt.ylabel('areal_velocity',fontweight='bold',fontsize=16)
        plt.xlabel('updates',fontweight='bold',fontsize=16)
        # plt.ylim(0,0.16)
        plt.tight_layout()
        """

        ## OMR behaviour (4) 
        plt.figure(figsize=(6,3))
        plt.plot(updates,OMR,'k')
        plt.ylabel('OMR(k)',fontweight='bold',fontsize=16)
        plt.xlabel('#updates',fontweight='bold',fontsize=16)
        plt.ticklabel_format(axis='x', style='sci', scilimits=[0,2])
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.ylim(0.0,1.2)
        plt.tight_layout()
        file_name4 = f'{algo}_OMR.png'
        file_path4 = abspath(join(this_dir,'exploration/reach',file_name4)) #file_path: location + name
        plt.savefig(file_path4, format='png') 
        
        # plt.show()
        # xxxx

        ## 3D Policy Trajectory Visualization (5)
        Xs = radii  #radii/max(radii)
        Ys = chords  #chords/max(chords) 
        self._3Dplots(Xs,Ys,name=f'{algo}') 
        file_name5 = f'{algo}_3D.png'
        file_path5 = abspath(join(this_dir,'exploration/reach',file_name5)) #file_path: location + name
        plt.savefig(file_path5, format='png') 

        plt.show()
        return
    
    #~~~~~~~~~~~~~~ ########################### ~~~~~~~~~~~~~~~~~~~~

#Execution
if __name__ == '__main__':
    agent = setting() 
    # agent.main()

    """
    #Verify_final_policy_converged 
    agent.verify_policy_converged()
    # """
    
    """ 
    #Generation_of_state-action_samples  
    agent.policy_data_generation() 
    # """

    """    
    #Generation_of_OTDD_path
    agent.policy_path_generation()
    # """

    # """    
    #Evaluation_of_OTDD_path
    agent.policy_path_evaluation()
    # """

    """    
    # Save Policy_trajectory file (pdf)
    agent.save_images_for_paper()
    # """

