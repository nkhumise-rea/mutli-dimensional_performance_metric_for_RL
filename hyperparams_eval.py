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
import time
import os
import argparse
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

parser = argparse.ArgumentParser()
parser.add_argument("--n_count", type=int, default=0)
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

##varying_parameters_
#(DDPG-&-TD3)
lr_actor_list = [1e-4, 1e-3, 1e-2] 
lr_critic_list = [1e-4, 1e-3, 1e-2] 
#(SAC)
alpha_list = [0.1, 0.2, 0.5]
lr_list = [1e-4, 1e-3, 1e-2] 


##ALL
hidden_sizes_RL = [256,256,256] #ref: Plappert (2018)
tau = 50e-3 #ref: Plappert (2018)
gamma = 0.99
num_episodes = 500
render = None

##tasks
task = 'Reach' #'Push', 'Slide'
algo = 'sac' #'td3' #'ddpg' 
cnt = 0 
encoding_dim = 25 #

print(f'algo: {algo}, task: {task}')

class setting():
    def __init__(self,steps_per_episode=100):

        ##Environment
        head = "human" #"rgb_array" # 
        self.env = gym.make('FetchReach-v4', 
                            render_mode=head,
                            max_episode_steps=steps_per_episode)

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
        
    # policy_evaluation
    def policy_evaluation(self):
        """
        Outputs Wasserstein distance between action pairs (a1,a2)

        Args:
            a1: array, action sample from policy 1 
            a2: array, action sample from policy 2
        """
        #NN architecture
        num_states = self.num_states
        num_actions = self.num_actions

        #file_location       
        file_location = f'torch_models/hyperparams/{task}/{algo}/maxSteps_{steps_per_episode}'
        
        # print('file_location: ', file_location)
        # xxxx
        # print(os.listdir(file_location))
        # print(sorted(os.listdir(file_location)))
        # print(len(os.listdir(file_location)))
        # xxx
        if algo == 'sac':
            hyperparams = [ (alpha,lr) for alpha in alpha_list for lr in lr_list]
        else:
            hyperparams = [ (lr_actor,lr_critic) for lr_actor in lr_actor_list for lr_critic in lr_critic_list]
        # print('hyperparams: ' , hyperparams)

        best_per_env_mean = 0 #-np.inf
        for num_hyp, params in enumerate(hyperparams):
            num_hyp += 1
            # retrieve policy_model
            if algo == 'ddpg':
                self.pi_net = self.retrieve_policy_model_ddpg(
                                                num_actions, 
                                                hidden_sizes_RL,  
                                                num_hyp,
                                                params,
                                                file_location
                                                )
            elif algo == 'sac':
                self.pi_net = self.retrieve_policy_model_sac(
                                                num_actions, 
                                                hidden_sizes_RL,  
                                                num_hyp,
                                                params,
                                                file_location
                                                )
            elif algo == 'td3':
                self.pi_net = self.retrieve_policy_model_td3(
                                                num_actions, 
                                                hidden_sizes_RL,  
                                                num_hyp,
                                                params,
                                                file_location
                                                )
            else:
                print('Algorithm not registered')

            #normalise score
            return_mean, _ = self.pol_rollouts()
            # print(f'hyp: {num_hyp}-{params} | mean: {return_mean:.6f}, std: {return_std:.6f}')
            
            if return_mean > best_per_env_mean:
                best_per_env_mean = return_mean
                # best_per_env_std = return_std
                best_num_hyp = num_hyp
                best_hyp_param = params


            #cross_environment
            hyperparams_cross[num_hyp].append(return_mean)


        #per_environment
        hyperparams_per[steps_per_episode] = (best_num_hyp,best_per_env_mean)

        # print(f'Best_per_env: Horizon: {steps_per_episode} -> hyp: {best_num_hyp}-{best_hyp_param} | mean: {best_per_env_mean:.6f}, std: {best_per_env_std:.6f}')
        # print(f'Best_per_env: Horizon: {steps_per_episode} -> hyp: {best_num_hyp}-{best_hyp_param} | mean: {best_per_env_mean:.6f}')
        # print('hyperparams_per: ', hyperparams_per)
        # print('hyperparams_cross: ', hyperparams_cross)
        # xxx
        return hyperparams_per, hyperparams_cross, hyperparams
           
    # policy_rollouts
    def pol_rollouts(self):
        """        
        Outputs mean return and standard deviation over N trials 
        """
        ep_return_list = []
        for _ in range(50): #5000 #num_decisions
            ep_return = self.learnt_agent() 
            ep_return_list.append(ep_return)
        
        # min-max normalisation
        min_return = -steps_per_episode #lowest_return_value
        max_return  = 0 #highest_return_value

        # print('ep_return_list: ',ep_return_list)
        ep_return_list = (np.array(ep_return_list) - min_return)/(max_return - min_return) #normalising_returns
        # print('ep_return_list: ',ep_return_list)

        return_mean = ep_return_list.mean()
        return_std = ep_return_list.std()
        # print('return_mean: ', return_mean)
        # print('return_std: ', return_std)
        # xx
        # print('meta_data: ', meta_data)
        # self.save_state_action_samples(meta_data,steps) #save_state-action_samples
        return return_mean,return_std
    
    #trained_agent (for: evaluation)
    def learnt_agent(self):
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
        
        t_state = self.preprocesses_obs(t_obs)
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
            t_next_state = self.preprocesses_obs(t_next_obs)
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
    
    #preprocessing_Observations
    def preprocesses_obs(self,obs_dict):
        parts = []
        for _, val in obs_dict.items():
            # print('val: ', val)
            parts.append(val)
        obs_vec = np.concatenate(parts,axis=0)
        # print('obs_vec: ', obs_vec)
        return torch.as_tensor(obs_vec,dtype=torch.float32).unsqueeze(0)


    #~~~~~~~~~~~~~~ Retrieving Policies & Saving Samples ~~~~~~~~~~~~~~~~~~~~

    # retrieve policy model
    def retrieve_policy_model_ddpg(self,
                            num_actions, 
                            num_hidden, 
                            num_hyp,
                            params,
                            file_location):
        
        #assign_hyperparameters
        # print('params: ', params)
        lr_actor = params[0]
        lr_critic = params[1]

        # steps = 1998
        file_name = f"{algo}_hypSet_{num_hyp}_iter_{args.n_count}.pth"  
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
                            num_hyp,
                            params,
                            file_location):
        
        #assign_hyperparameters
        # print('params: ', params)
        alpha = params[0]
        lr_sac = params[1]

        # steps = 1998
        file_name = f"{algo}_hypSet_{num_hyp}_iter_{args.n_count}.pth"  
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
                            num_hyp,
                            params,
                            file_location):
        
        #assign_hyperparameters
        # print('params: ', params)
        lr_actor_td3 = params[0]
        lr_critic_td3 = params[1]
        
        # steps = 1998
        file_name = f"{algo}_hypSet_{num_hyp}_iter_{args.n_count}.pth"  
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
    
    #~~~~~~~~~~~~~~ ########################### ~~~~~~~~~~~~~~~~~~~~




#Execution
if __name__ == '__main__':
    
    # """
    ##
    start_time = time.time()

    #declare_dictionaries_for_scores
    hyperparams_cross = defaultdict(list)
    hyperparams_per = {}

        
    #Horizon variants
    steps_per_episode_list = [25,50,75,100,125]
    for steps_per_episode in steps_per_episode_list:
        agent = setting(steps_per_episode) 
        per_env, cross_env, hyp_params = agent.policy_evaluation()
    print('per_env: ', per_env)
    print('cross_env: ', cross_env)
    print('hyp_params: ', hyp_params)
    xxx

    #per-env_tuned_score
    avg_max_score = 0.0
    for horizon_id, hyp_id_score in per_env.items():
        # print('horizon_id: ', horizon_id)
        # print('hyp_id_score: ', hyp_id_score)
        hyp_id, max_hyp_score = hyp_id_score
        # print('hyp_id: ', hyp_id)
        # print('max_hyp_score: ',max_hyp_score)
        avg_max_score += max_hyp_score
    per_env_avg_max_score = avg_max_score/len(steps_per_episode_list)
    print('per_env_avg_max_score: ', per_env_avg_max_score)

    #cross-env_tuned_score
    best_avg_score = 0.0
    for hyp_id, hyp_score_list in cross_env.items():
        # print('hyp_id: ', hyp_id)
        # print('hyp_score_list: ', hyp_score_list)
        scores = np.array(hyp_score_list)
        avg_score = scores.mean()
        # print('avg_score: ', avg_score)
        if best_avg_score < avg_score:
            best_avg_score = avg_score
            best_hyp_id = hyp_id
    cross_env_max_avg_score = best_avg_score
    print(f'best_hyp: {best_hyp_id}-{hyp_params[best_hyp_id-1]} | cross_env_max_avg_score: {cross_env_max_avg_score} ')

    #hyperparameter_sensitivity
    sensitivity = per_env_avg_max_score - cross_env_max_avg_score
    print(f'algo: {algo} -> sensitivty: {sensitivity}')

    end_time = time.time()
    duration = end_time - start_time
    print(f'duration: {duration/60} min')
    #"""

    # steps_per_episode = 25 #100
    # agent = setting(steps_per_episode) 

    """
    #Evaluate_learned_policy
    agent.policy_evaluation()
    #"""

       