#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 2024
Modified on Sat Oct 25 2025
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
from sklearn.preprocessing import StandardScaler
import joblib
import seaborn as sns
import ot
import time
import os
import argparse
from os.path import dirname, abspath, join
import sys
from collections import deque

#gym
import gymnasium as gym
import gymnasium_robotics

#pytorch
import torch
import torch.nn as nn
import torch.optim as optima
import torch.nn.functional as F

#tianshou
from torch.utils.tensorboard import SummaryWriter
from tianshou.data import Batch
from tianshou.policy import DDPGPolicy,SACPolicy,TD3Policy
from tianshou.trainer import offpolicy_trainer
from tianshou.utils.net.common import Net, get_dict_state_decorator
from tianshou.utils.net.continuous import Actor,ActorProb,Critic
from tianshou.utils import TensorboardLogger
from tianshou.env import DummyVectorEnv #dnt
from tianshou.exploration import GaussianNoise, OUNoise

#GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#set directory/location
this_dir = dirname(__file__)
agent_dir = abspath(join(this_dir,'..')) 
sys.path.insert(0,agent_dir)


##Autoencoder_Parameters
hidden_sizes = [256,64] #[256,128]


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
algo = 'ddpg' #'td3' #'sac' #'ddpg' #'td3'   
cnt = 0 
encoding_dim = 2

class Autoencoder(nn.Module):
    def __init__(self,num_states,encoding_dim,num_hidden_l1,num_hidden_l2):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(num_states,num_hidden_l1),
            nn.ReLU(),
            nn.Linear(num_hidden_l1,num_hidden_l2),
            nn.ReLU(),
            nn.Linear(num_hidden_l2,encoding_dim),
            nn.Tanh(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim,num_hidden_l2),
            nn.ReLU(),
            nn.Linear(num_hidden_l2,num_hidden_l1),
            nn.ReLU(),
            nn.Linear(num_hidden_l1,num_states),
            # nn.Tanh(),
        )

    def forward(self,state):
        latent = self.encoder(state)
        estimated_state = self.decoder(latent)
        return latent, estimated_state

class setting():
    def __init__(self, 
                 rew_setting = 1, #[1:dense, 0:sparse]
                 n_eps = 500,
                 ):
        
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
        # self.itr = iteration
        # print('self.num_states: ', self.num_states)

        #Rounding_up_digits
        self.p = 5
        
        self.num_bins_action = 4 #discretization_bins #12
        self.bins_action = {
            '0': np.linspace(-1,1,self.num_bins_action+1),
            '1': np.linspace(-1,1,self.num_bins_action+1),
            '2': np.linspace(-1,1,self.num_bins_action+1),
            '3': np.linspace(-1,1,self.num_bins_action+1)
            }
        # print('self.bins_action: ', self.bins_action)

        self.discrete_actions = {
           '0': np.array([ round((self.bins_action['0'][i]+self.bins_action['0'][i+1])/2,self.p) for i in range(self.num_bins_action) ]),
           '1': np.array([ round((self.bins_action['1'][i]+self.bins_action['1'][i+1])/2,self.p) for i in range(self.num_bins_action) ]),
           '2': np.array([ round((self.bins_action['2'][i]+self.bins_action['2'][i+1])/2,self.p) for i in range(self.num_bins_action) ]),
           '3': np.array([ round((self.bins_action['3'][i]+self.bins_action['3'][i+1])/2,self.p) for i in range(self.num_bins_action) ])
        }
        # print('self.discrete_actions: ', self.discrete_actions)

        self.num_bins = 10 #discretization_bins #12
        self.bins_states, self.discrete_states = {},{}
        for k in range(encoding_dim):
            self.bins_states.update({f'{k}': np.linspace(-1,1,self.num_bins+1)})
            self.discrete_states.update({f'{k}': np.array([ round((self.bins_states[f'{k}'][i]+self.bins_states[f'{k}'][i+1])/2,self.p) for i in range(self.num_bins) ])})
        # print('self.bins_states: ', self.bins_states)
        # print('self.discrete_states: ', self.discrete_states)


        self.discrete_state_space = np.array([v for v in product(*self.discrete_states.values())]) #state_space
        self.discrete_action_space = np.array([v for v in product(*self.discrete_actions.values())]) #action_space
        self.discrete_actions_pairs = [(i,j) for i,_ in enumerate(self.discrete_action_space) for j,_ in enumerate(self.discrete_action_space) ] #action_pairs
        # print('self.discrete_state_space : ', self.discrete_state_space )
        # print('self.discrete_state_space.shape: ', self.discrete_state_space.shape)
        # print(self.discrete_state_space.reshape(-1,2))
        # print('self.discrete_action_space : ', self.discrete_action_space )
        # print('self.discrete_action_space.shape: ', self.discrete_action_space.shape)
        # print(self.discrete_action_space.reshape(-1,4))
        # print('self.discrete_actions_pairs: ', len(self.discrete_actions_pairs))
        # xxx

    # policy_data_generation
    def policy_data_generation(self,iteration=0):
        self.iteration = iteration #process_iteration_num

        #NN architecture
        num_states = self.num_states
        num_actions = self.num_actions

        #file_location
        file_location = f'{algo}/{task.lower()}/count_{cnt}/iter_{self.iteration}'
        # print('file_location: ', file_location)
        # xxxx
        # print(os.listdir(file_location))
        # print(sorted(os.listdir(file_location)))
        # print(len(os.listdir(file_location)))
        # xxx
        for step in range(len(os.listdir(file_location))-1):
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
            # """

        return 
    

    #~~~~~~~~~~~~~~ Retrieving & Saving ~~~~~~~~~~~~~~~~~~~~

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
            tau = tau,
            gamma = gamma,
            exploration_noise = GaussianNoise(sigma=self.sigma), #OUNoise(),
            estimation_step = 1,
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
        actor_optim = torch.optim.Adam(actor.parameters(),lr=lr_sac)
        critic1_optim = torch.optim.Adam(critic1.parameters(),lr=lr_sac)
        critic2_optim = torch.optim.Adam(critic2.parameters(),lr=lr_sac)

        policy = SACPolicy(
            actor,
            actor_optim,
            critic1,
            critic1_optim,
            critic2,
            critic2_optim,
            tau = tau,
            gamma = gamma,
            alpha = alpha,
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
            exploration_noise=GaussianNoise(sigma=self.sigma), #OUNoise(),
            estimation_step=1,
            )

        ##load_trained_policy
        policy.load_state_dict(torch.load(file_path, map_location=torch.device('cpu')))                
        policy.eval()
        return policy
   

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
        
        t_state,latent_state = self.preprocesses_obs(t_obs)
        # print('batch2: ', t_state)
        # print('latent_state: ', latent_state)
        # xxx

        while not t_done:
            t_action = self.select_action(t_state)  
            # print('t_action: ', t_action)

            t_next_obs, t_reward, t_terminated, t_truncated, _ = self.env_test.step(t_action)
            # print('t_next_obs: ', t_next_obs)

            t_done = t_terminated or t_truncated #done_function (new_approach)
            ep_return += t_reward
            t_next_state,latent_next_state = self.preprocesses_obs(t_next_obs)
            # print('batch2: ', t_next_state)
            # print('latent_next_state: ', latent_next_state)
            # xx
            
            #experience_buffer_data
            st_list.append(latent_state.tolist()) #stt_list
            rew_list.append(t_reward) #rew_list

            if t_done:
                trj = deepcopy(st_list)
                trj.append(latent_next_state.tolist())

            act_list.append(t_action.tolist())
            t_state = t_next_state
            latent_state = latent_next_state
        
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
        rew_list, act_list, stt_list, ret_list = [],[],[],[]
        act_du,stt_du = [0],[0]

        while act_du[-1] <= 5000: #num_decisions
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
        file_location = f'{algo}/{task.lower()}/samples_{cnt}/iter_{self.iteration}/dim_{encoding_dim}'
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
        # print(batch)
        # action = self.pi_net.actor(batch) #.detach().cpu().numpy()[0]
        # print(action)
        # xxxx

        with torch.no_grad():
            if algo == 'sac':
                action = self.pi_net.actor(batch)[0][0].detach().cpu().numpy()[0]
            else:
                action = self.pi_net.actor(batch)[0].detach().cpu().numpy()[0]
        # print(action)
        # xxx
        # with torch.no_grad():
        #     action = self.pi_net.actor(batch)[0].detach().cpu().numpy()[0]
        return action

    #preprocessing_Observations
    def preprocesses_obs(self,obs_dict):
        parts = []
        ae_parts = obs_dict['observation'] #state_4_autoencoder
        # print('ae_parts: ', ae_parts)

        for _, val in obs_dict.items():
            # print('val: ', val)
            parts.append(val)
        obs_vec = np.concatenate(parts,axis=0)
        # print('obs_vec: ', obs_vec)

        ####### autoencoder #######
        X = [ae_parts]
        # print('X: \n', X)
        # xxx

        #load_saved_scaler_parameters
        file_location = 'algos_tasks/autoencoder_training_data'
        scaler = joblib.load(f'{file_location}/dim_{encoding_dim}/scaler.pkl')
        X_scaled = scaler.fit_transform(X)
        # print('X_scaled: \n', X_scaled)
        X_tensor = torch.FloatTensor(X_scaled)
        # print('X_tensor: \n', X_tensor)

        autoencoder = Autoencoder(
                            self.num_states,
                            encoding_dim,
                            hidden_sizes[0],
                            hidden_sizes[1])
        file_path = f"{file_location}/dim_{encoding_dim}/encoder_tuned.pth"
        autoencoder.load_state_dict(torch.load(file_path, map_location=torch.device('cpu')))
        autoencoder.eval()

        with torch.no_grad():
            z_new,_ = autoencoder(X_tensor)
            # print('z_new: ', z_new.cpu().numpy()[0])


        # print(torch.as_tensor(obs_vec,dtype=torch.float32).unsqueeze(0))
        # print(z_new.cpu().numpy()[0])
        # xxx
        
        return torch.as_tensor(obs_vec,dtype=torch.float32).unsqueeze(0), z_new.cpu().numpy()[0]

    # occupancy_measure_generation
    def occupancy_generation_fast(self,iteration=0):
        self.iteration = iteration #process_iteration_num

        # """
        # execution  
        file_location = f'{algo}/{task.lower()}/samples_{cnt}/iter_{self.iteration}/dim_{encoding_dim}'
        final_policy_data_num = len(os.listdir(file_location)) - 1 #optimal_policy
        initial_policy_data_num = 0 #initial_policy
        # print(file_location)
        print('final_policy_data_num: ', final_policy_data_num)
        print('initial_policy_data_num: ', initial_policy_data_num)
        # xxxx

        # initial_policy & final_policy occupancy_measures
        final_policy_occup,_ = self.occupancy_measure_finite_timeless_fast(
                        final_policy_data_num,
                        file_location
                        )
        # print('final_policy_occup: ', final_policy_occup)
        # print('final_states_visits: \n', final_states_visits)
        # xxxx
        initial_policy_occup,_ = self.occupancy_measure_finite_timeless_fast( #initl_states_visits
                        initial_policy_data_num,
                        file_location
                        )
        # print('initial_policy_occup: ', initial_policy_occup)

        direct_dis = self.OTDD(final_policy_occup,initial_policy_occup)
        print('direct_dis: ', direct_dis)
        # print('lip_con: ', lip_con)
        xx
        # max_lip_con = lip_con # maximum lipschitz constant
        succ_dis_col,to_fin_dis_col,to_ini_dis_col = [],[],[]
        # print('max_lip_con_start: ', max_lip_con)
        for update in range(len(os.listdir(file_location))-1):

            # measure from data retrieved 
            policy_occup_1,_ = self.occupancy_measure_finite_timeless_fast(
                            update,
                            file_location
                            )
            
            policy_occup_2,_ = self.occupancy_measure_finite_timeless_fast(
                            update+1,
                            file_location
                            )

            # #between successive 
            successive_dis = self.OTDD(policy_occup_1,policy_occup_2)
            succ_dis_col.append(successive_dis)
            
            # #between final(optimal)_&_others
            to_final_dis = self.OTDD(policy_occup_1,final_policy_occup)
            to_fin_dis_col.append(to_final_dis)
            
            xxxx
            max_lip_con = 0 #max(lip_con_1,lip_con_2,max_lip_con)

            if update % 500 == 0:
                #cumulative data
                policy_trajectory_data = np.asarray([
                    direct_dis, #geodesic_distance
                    max_lip_con, #lipschitz_constant
                    succ_dis_col, #y_k (stepwise distance)
                    to_fin_dis_col, #x_k (distance_to_optimal)
                    ],dtype=object) #dtype=object
            
                self.save_policy_trajectories_per_update(policy_trajectory_data,update)


    #~~~~~~~~~~~~~~ Value Functions & Occupancy Measures ~~~~~~~~~~~~~~~~~~~~

    # Occupancy Measure for stationary Policy in Finite MDP (faster version)
    def occupancy_measure_finite_timeless_fast(self,steps,file_location): #using_finite_episodes
        
        ## load_data
        file_name = f'sample_{steps}.npy'
        file_path = abspath(join(this_dir,file_location,file_name)) #file_path: location + name
        data = np.load(file_path, allow_pickle=True)
        # print('data: ', data)
        # xxx

        ## trajectories
        x1 = data[0] #states_in_dataset
        y1 = data[1] #actions_in_dataset
        tx1 = np.asarray(data[5]) #len(states)_of_runs
        ty1 = np.asarray(data[5]) #len(actions)_of_runs


        # print('x1: ', x1)
        # print('y1: ', len(y1))

        # print('tx1: ', tx1)
        # print('tx2: ', ty1)

        #################### corrupted files ####################
        # check the dimension of every entry that it is 2

        for idx, dlist in enumerate(x1):
            # print(len(dlist))
            if len(dlist) > 2:
                print(idx)
                xxx

        #########################################################
        # xxx

        states_dict,actions_dict = {},{}

        # iterate_each_rollout
        for i in range(len(tx1)-1):
            states = x1[tx1[i]:tx1[i+1]] #states_per_rollout
            actions = y1[ty1[i]:ty1[i+1]] #actions_per_rollout
            states_dict[i], actions_dict[i] = states, actions

        # xxx
        steps_counts = defaultdict(int)
        pair_counts = defaultdict(int)
        probs = defaultdict(int) #probabilities [occupancy_measure]
        
        # iterate_each_episode_rollout
        for k in actions_dict.keys():
            states = states_dict[k] #states_per_episode_rollout
            actions = actions_dict[k] #actions_per_episode_rollout
            steps_counts[k] = len(actions) #num_decisions_per_episode

            dis_states, dis_actions = self.partitions_fast(states,actions) #state_action_discretization
            
            for j in zip(dis_states,dis_actions): 
                pair_counts[j] += 1
            # print('states: ', states[0])
        
        total_counts = sum(pair_counts.values())
        avg_eps_length = sum(steps_counts.values())/len(steps_counts)
        # print('total_count: ', total_counts)
        # xxxx
        #calculate_probability_distribution
        for j in pair_counts:
                # probs[j] = round(pair_counts[j]/total_counts,self.p)
                probs[j] = pair_counts[j]/total_counts
        
        return probs, avg_eps_length #, policy_state_visits
    
    # state_action_discretization
    def partitions_fast(self,states,actions):
        
        actions = np.array(actions)
        # print(actions)
        # xxx
        act = {
            '0': actions[:,0], #column1
            '1': actions[:,1], #column2
            '2': actions[:,2], #column3
            '3': actions[:,3] #column4
        }

        bins_action_counts_,_ = np.histogramdd(
            [act['0'], 
            act['1'], 
            act['2'],
            act['3']],
            bins=[self.bins_action['0'], 
                  self.bins_action['1'],
                  self.bins_action['2'], 
                  self.bins_action['3']
                  ]
        )    

        # print(bins_action_counts_.shape)
        # print(bins_action_counts_[-1,-1,-1,-1])
        # cnt = np.sum(np.all(np.isclose(states, [0.6, 0.07]), axis=1))

        # print(bins_action_counts_.shape[0])
        # print(bins_action_counts_.shape[1])
        # print(bins_action_counts_.shape[2])
        # print(bins_action_counts_.shape[3])
        # xxx

        dis_actions = [
            (self.discrete_actions['0'][i], 
             self.discrete_actions['1'][j],
             self.discrete_actions['2'][k],
             self.discrete_actions['3'][l])
            for i in range(bins_action_counts_.shape[0])
            for j in range(bins_action_counts_.shape[1])
            for k in range(bins_action_counts_.shape[2])
            for l in range(bins_action_counts_.shape[3])
            for _ in range(int(bins_action_counts_[i,j,k,l]))
            ]   
        # print(dis_actions)

        states = np.array(states)
        # print(states)
        stat = {
            '0': states[:,0], #column1
            '1': states[:,1], #column2
            # '2': actions[:,2], #column3
            # '3': actions[:,3] #column4
        }

        bins_state_counts_,_ = np.histogramdd(
            [stat['0'], 
            stat['1'], 
            # stat['2'],
            # stat['3']
            ],
            bins=[self.bins_states['0'], 
                  self.bins_states['1'],
                #   self.bins_action['2'], 
                #   self.bins_action['3']
                  ]
        )    

        # print(bins_state_counts_.shape)
        # print(bins_state_counts_[-1,-1])
        cnt = np.sum(np.all(np.isclose(states, [0.20808282,0.23740117]), axis=1))
        bins_state_counts_[-1, -1] += cnt

        dis_states = [
            (self.discrete_states['0'][i], 
             self.discrete_states['1'][j],
            #  self.discrete_states['2'][k],
            #  self.discrete_states['3'][l]
             )
            for i in range(bins_state_counts_.shape[0])
            for j in range(bins_state_counts_.shape[1])
            # for k in range(bins_state_counts_.shape[2])
            # for l in range(bins_state_counts_.shape[3])
            for _ in range(int(bins_state_counts_[i,j]))
            ]   
        # print(dis_states)
        return dis_states, dis_actions #, bins_pos_vel_counts

    #~~~~~~~~~~~~~~ OTDD ~~~~~~~~~~~~~~~~~~~~

    # Wasserstein: OTDD
    def OTDD(self,occu1,occu2):
        p_sGa1 = self.prob_states_given_actions(occu1)
        p_sGa2 = self.prob_states_given_actions(occu2)

        # print('p_sGa1: ', p_sGa1)
        # print('p_sGa2: ', p_sGa2)
        # print("=======================================")
        # xxxx

        action_dis = defaultdict(int) 
        # print(self.y_pairs)

        # action_to_action distance
        for i in self.discrete_actions_pairs:
            # print(i)
            # xxx
            action_dis[i] = self.inner_wasserstein(p_sGa1[i[0]],p_sGa2[i[1]])
        # print('action_dis: \n', action_dis)
        # xxxx

        # state-action-pair_to_state-action-pair distance
        otdd = self.outer_wasserstein(occu1,occu2,action_dis)
        print('otdd: ', otdd)
        xxx
        return otdd #, lip_con
    
    # Mapping actions -> state_distributions
    def prob_states_given_actions(self,occup):
        p_sGa = defaultdict(lambda: defaultdict(int)) #probabilities [states | actions = a]
        p_a = defaultdict(int) #probabilities [actions]



        # print('self.discrete_states[0]: ', self.discrete_states['0'])
        
        # print('self.discrete_action_space: ', self.discrete_action_space)
        # xxx

        # p(A)
        for k,_ in enumerate(self.discrete_action_space):
            for i in self.discrete_states['0']:
                for j in self.discrete_states['1']:
                    # for k in self.discrete_states['2']:
                    #     for l in self.discrete_states['3']:
                    p_a[k] += occup[((i,j),k)]
        # print('p_a: \n', p_a)
        # xxxx

        # p(S|A)
        for k,_ in enumerate(self.discrete_action_space):
            for i in self.discrete_states['0']:
                for j in self.discrete_states['1']:
                    if occup[((i,j),k)] == 0: 
                        p_sGa[k][(i,j)] = 0
                    else:
                        p_sGa[k][(i,j)] = occup[((i,j),k)]/p_a[k]
        # print('p_sGa: \n', p_sGa)
        # print('p_sGa.size: \n', len(p_sGa))
        # print("=======================================")
        # xxx
        return p_sGa

    # Inner_Wasserstein
    def inner_wasserstein(self,p_sGa,q_sGa):
        z = self.discrete_state_space
        CostMatrix = ot.dist(z,z, metric='euclidean') #cost matrix: 'cityblock' 
        # print('CostMatrix: ',CostMatrix)
        # xxxx
                
        P = np.asarray(list(p_sGa.values())) #prob_dis
        Q = np.asarray(list(q_sGa.values())) #prob_dis

        # print('P: \n', P)
        # print('Q: \n', Q)
        # print("=======================================")
        # xxxx

        if P.sum() != 0 and Q.sum() != 0:
            val = ot.emd2(  P, #A_distribution 
                            Q, #B_distribution
                            M = CostMatrix, #cost_matrix pre-processing
                            numItermax=int(1e6)
                            ) #OT matrix
        else:
            val = None
        # print('val: ', val)
        # xxx

        return val
    
    # Outer_Wasserstein
    def outer_wasserstein(self,occu1,occu2,action_dis):
        z1 = list(occu1.keys())  #state-action space
        z2 = list(occu2.keys())  #state-action space
        CostMatrix = self.outer_cost(z1,z2,action_dis) #cost matrix
        print('CostMatrix: ',CostMatrix)
        print('CostMatrix: ',len(CostMatrix))
        xxx
       
        P = np.asarray(list(occu1.values())) #prob_dis #self.p
        Q = np.asarray(list(occu2.values()))#prob_dis

        # P = np.asarray(list(occu1.values())) #prob_dis
        # Q = np.asarray(list(occu2.values())) #prob_dis
        
        # print('P: ', P)
        # print('Q: ', Q)

        # print('P.sum(): ', P.sum())
        # print('Q.sum(): ', Q.sum())

        if P.sum() != 0 and Q.sum() != 0:
            val = ot.emd2(  P, #A_distribution 
                            Q, #B_distribution
                            M = CostMatrix, #cost_matrix pre-processing
                            numItermax=int(1e6)
                            ) #OT matrix
        else:
            val = None
        # print('val: ', val)
        # print('lip_con: ', lip_con)
        # xxx

        return val #, lip_con
    
    # Outer_cost_metric
    def outer_cost(self,z1,z2,action_dis):        
        len_z1 = len(z1) 
        len_z2 = len(z2)  
        m = np.zeros([len_z1,len_z2])
        # lip_con = 0
        for idx, i in enumerate(z1):
            for jdx, j in enumerate(z2):
                vector = np.asarray(i[0]) - np.asarray(j[0])

                dis_states = LA.norm(vector) #'euclidean'
                dis_actions = action_dis[ i[1], j[1]]
                # print('dis_states: ', dis_states)
                # print('dis_actions: ', dis_actions)

                # rew_diff = np.abs(self.rewards_function(i[0],i[1]) - self.rewards_function(j[0],j[1]) ) #rewards_difference
                # print('rew_diff: ', rew_diff)

                if dis_actions == None:
                    m[idx][jdx] = 0
                else: 
                    dis_state_action_pairs = dis_states + dis_actions
                    m[idx][jdx] = dis_state_action_pairs

                    # # Lipschitz_constant_calculator
                    # if dis_state_action_pairs > 1e-4: #d_sa != 0 (prevent extreme values)
                    #     """
                    #     d_sa > 1e-4: ensures numerical stability, 
                    #     focusing on pairs with meaningful information
                    #     """
                    #     ratio = rew_diff/dis_state_action_pairs
                    #     if ratio > lip_con:
                    #         lip_con = ratio
        # print('m: ', m) 
        # xxx
        # print('lip_con: ', lip_con) 
        return m #, lip_con
    

    # MountainCar_rewards_model
    def rewards_function(self,state,action):
        rew = -0.1*(action**2)
        if state[0] >= 0.45:
            rew = 100
        # print('rew: ', rew)
        # xxx
        return rew


#Execution
if __name__ == '__main__':
    agent = setting( rew_setting=1, #[rew_setting, num_eps]
                      n_eps=100,
                      ) 
    # agent.main()
    
    # """ 
    #Generation_of_state-action_samples  
    agent.policy_data_generation() 
    # """

    # """ 
    #Generation_of_policy_trajectory   
    # agent.occupancy_generation_fast(iteration=0)
    #"""