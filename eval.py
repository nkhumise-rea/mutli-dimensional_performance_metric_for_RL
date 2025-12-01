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
import seaborn as sns
import ot
import time
import os
import argparse
from os.path import dirname, abspath, join
import sys
from collections import deque

#gym
import gym

#pytorch
import torch
import torch.nn as nn
import torch.optim as optima
import torch.nn.functional as F

#GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#set directory/location
this_dir = dirname(__file__)
agent_dir = abspath(join(this_dir,'..')) 
sys.path.insert(0,agent_dir)

class setting():
    def __init__(self, 
                 rew_setting = 1, #[1:dense, 0:sparse]
                 n_eps = 500,
                 ):
        
        self.algo = 'ddpg'
        self.env = gym.make('MountainCarContinuous-v0') 

        self.num_states = self.env.observation_space.shape[0]
        self.num_actions = self.env.action_space.shape[0]
        # self.itr = iteration
        # print('self.num_states: ', self.num_states)

        #Rounding_up_digits
        self.p = 5
        
        self.num_bins_action = 4 #discretization_bins #12
        self.bins_action = np.linspace(-1,1,self.num_bins_action+1)
        self.discrete_actions = np.array([ round((self.bins_action[i]+self.bins_action[i+1])/2,self.p) for i in range(self.num_bins_action) ])
        # print('self.bins_action: ', self.bins_action)
        # print('self.discrete_actions: ', self.discrete_actions)

        ######################################################
        ################### Fetch: Actions ###################
        """
        self.bins_action = {
            '0': np.linspace(-1,1,self.num_bins_action+1),
            '1': np.linspace(-1,1,self.num_bins_action+1),
            '2': np.linspace(-1,1,self.num_bins_action+1),
            '3': np.linspace(-1,1,self.num_bins_action+1)
            }
        
        print('self.bins_action: ', self.bins_action)

        self.discrete_actions = {
           '0': np.array([ round((self.bins_action['0'][i]+self.bins_action['0'][i+1])/2,self.p) for i in range(self.num_bins_action) ]),
           '1': np.array([ round((self.bins_action['1'][i]+self.bins_action['1'][i+1])/2,self.p) for i in range(self.num_bins_action) ]),
           '2': np.array([ round((self.bins_action['2'][i]+self.bins_action['2'][i+1])/2,self.p) for i in range(self.num_bins_action) ]),
           '3': np.array([ round((self.bins_action['3'][i]+self.bins_action['3'][i+1])/2,self.p) for i in range(self.num_bins_action) ])
        }

        print('self.discrete_actions: ', self.discrete_actions)
        """

        ######################################################

        self.num_bins = 10 #discretization_bins #12
        self.bins_pos = np.linspace(-1.2,0.6,self.num_bins+1)
        self.bins_vel = np.linspace(-0.07,0.07,self.num_bins+1)
        self.discrete_pos = np.array([ round((self.bins_pos[i]+self.bins_pos[i+1])/2,self.p) for i in range(self.num_bins) ])
        self.discrete_vel = np.array([ round((self.bins_vel[i]+self.bins_vel[i+1])/2,self.p) for i in range(self.num_bins) ])

        # self.discrete_actions = np.array([ (self.bins_action[i]+self.bins_action[i+1])/2 for i in range(self.num_bins) ])
        # self.discrete_pos = np.array([ (self.bins_pos[i]+self.bins_pos[i+1])/2 for i in range(self.num_bins) ])
        # self.discrete_vel = np.array([ (self.bins_vel[i]+self.bins_vel[i+1])/2 for i in range(self.num_bins) ])

        self.discrete_states = np.array([(i,j) for i in self.discrete_pos for j in self.discrete_vel ]) #state_space
        self.discrete_actions_pairs = [(i,j) for i in self.discrete_actions for j in self.discrete_actions ] #action_pairs
        # print('self.discrete_states: ', self.discrete_states.shape)
        # print(self.discrete_states.reshape(-1,2))

        ######################################################
        ################### Fetch: States ####################
        """
        self.bins_states = {
            '0': np.linspace(-1,1,self.num_bins+1),
            '1': np.linspace(-1,1,self.num_bins+1),
            # '2': np.linspace(-1,1,self.num_bins+1),
            # '3': np.linspace(-1,1,self.num_bins+1)
            }
        # print('self.bins_states: ', self.bins_states)

        self.discrete_states = {
           '0': np.array([ round((self.bins_states['0'][i]+self.bins_states['0'][i+1])/2,self.p) for i in range(self.num_bins) ]),
           '1': np.array([ round((self.bins_states['1'][i]+self.bins_states['1'][i+1])/2,self.p) for i in range(self.num_bins) ]),
        #    '2': np.array([ round((self.bins_states['2'][i]+self.bins_states['2'][i+1])/2,self.p) for i in range(self.num_bins) ]),
        #    '3': np.array([ round((self.bins_states['3'][i]+self.bins_states['3'][i+1])/2,self.p) for i in range(self.num_bins) ])
        }
        # print('self.discrete_states: ', self.discrete_states)

        self.discrete_state_space = np.array([v for v in product(*self.discrete_states.values())]) #state_space
        # print('self.discrete_state_space : ', self.discrete_state_space )
        # print('self.discrete_state_space.shape: ', self.discrete_state_space.shape)
        # print(self.discrete_state_space.reshape(-1,2))
        """

        ######################################################

    # policy_data_generation
    def policy_data_generation(self,iteration=0):
        self.iteration = iteration #process_iteration_num

        #NN architecture
        num_states = self.num_states
        num_actions = self.num_actions
        num_hidden_l1 = 64 
        num_hidden_l2 = 64 

        #file_location
        # file_location = f'ddpg_data/models/iter_{self.iteration}'
        file_location = f'/mnt/parscratch/users/acp21rmn/Mountain_Car_OTDD/algo_data/ddpg/models/iter_{self.iteration}'

        # print(os.listdir(file_location))
        # print(sorted(os.listdir(file_location)))
        # print(len(os.listdir(file_location)))

        for step in range(len(os.listdir(file_location))):
        # for step in [26589]: #range(23620,len(os.listdir(file_location))-1):
            # print('step: ', step)
    
            #  """            
            # retrieve policy_model
            self.pi_net = self.retrieve_policy_model(
                                                num_states, 
                                                num_actions, 
                                                num_hidden_l1, 
                                                num_hidden_l2,
                                                step,
                                                file_location)

            # generate_roll-outs
            self.pol_rollouts(step)
            # """

        return 

    # occupancy_measure_generation
    def occupancy_generation(self,iteration=0):
        self.iteration = iteration #process_iteration_num

        # """
        # execution        
        file_location = f'/mnt/parscratch/users/acp21rmn/Mountain_Car_OTDD/algo_data/ddpg/samples/iter_{self.iteration}'
        final_policy_data_num = len(os.listdir(file_location)) - 1 #optimal_policy
        initial_policy_data_num = 0 #initial_policy
        #"""

        """        
        #testing
        file_location = f'ddpg_data/samples/iter_{self.iteration}'
        final_policy_data_num = 23619 #optimal_policy
        initial_policy_data_num = 0 #initial_policy
        # print('final_policy_data_num: ', final_policy_data_num)
        #"""

        # initial_policy & final_policy occupancy_measures
        final_policy_occup,_,final_states_visits = self.occupancy_measure_finite_timeless(
                        final_policy_data_num,
                        file_location
                        )
        # print('final_policy_occup: ', final_policy_occup)
        # print('final_states_visits: \n', final_states_visits)

        initial_policy_occup,_,_= self.occupancy_measure_finite_timeless( #initl_states_visits
                        initial_policy_data_num,
                        file_location
                        )
        # print('initial_policy_occup: ', initial_policy_occup)

        # self.plot_state_visits(final_states_visits)
        # self.plot_state_visits(initl_states_visits)

        trajctory_states_visits = np.zeros_like(final_states_visits)
        # print('trajctory_states_visits: ', trajctory_states_visits)

        direct_dis,lip_con = self.OTDD(final_policy_occup,initial_policy_occup)
        # print('direct_dis: ', direct_dis)
        # print('lip_con: ', lip_con)

        max_lip_con = lip_con # maximum lipschitz constant
        succ_dis_col,to_fin_dis_col,to_ini_dis_col = [],[],[]
        # print('max_lip_con_start: ', max_lip_con)
        for update in range(len(os.listdir(file_location))-1):

            # measure from data retrieved 
            policy_occup_1,_,policy1_states_visits = self.occupancy_measure_finite_timeless(
                            update,
                            file_location
                            )
            
            policy_occup_2,_,policy2_states_visits = self.occupancy_measure_finite_timeless(
                            update+1,
                            file_location
                            )
            
            trajctory_states_visits += (policy1_states_visits + policy2_states_visits)
            # print(trajctory_states_visits)
            # xxx

            # #between successive 
            successive_dis,lip_con_1 = self.OTDD(policy_occup_1,policy_occup_2)
            succ_dis_col.append(successive_dis)
            
            # #between final(optimal)_&_others
            to_final_dis,lip_con_2 = self.OTDD(policy_occup_1,final_policy_occup)
            to_fin_dis_col.append(to_final_dis)

            # #between initial_&_others
            to_initial_dis,lip_con_3 = self.OTDD(policy_occup_2,initial_policy_occup)
            to_ini_dis_col.append(to_initial_dis)
            
            # print('lip_con_1: ', lip_con_1)
            # print('lip_con_2: ', lip_con_2)
            # print('lip_con_3: ', lip_con_3)
            # xxx

            max_lip_con = max(lip_con_1,lip_con_2,lip_con_3,max_lip_con)

            # print('max_lip_con: ', max_lip_con)
        
        # print(trajctory_states_visits)
        # trajctory_states_visits /= trajctory_states_visits.sum()
        # self.plot_state_visits(100*trajctory_states_visits)
        # xxx

        policy_trajectory_data = np.asarray([
            direct_dis, #geodesic_distance
            max_lip_con, #lipschitz_constant
            succ_dis_col, #y_k (stepwise distance)
            to_fin_dis_col, #x_k (distance_to_optimal)
            to_ini_dis_col, # (distance_from_initial)
            trajctory_states_visits, #state_visitaion_frequencies
            ],dtype=object) #dtype=object
        
        self.save_policy_trajectories(policy_trajectory_data)

        """  
        ## testing_visualization      
        plt.figure(figsize=(12, 9))

        ## collection_of_successive_distances 
        plt.subplot(2,2,1)
        plt.plot(succ_dis_col,color='green',alpha=0.2)
        plt.plot(self.running_avg(succ_dis_col,0.9),color='orchid') 
        plt.title('stepwise')
        plt.ylabel('distance')
        plt.xlabel('updates')

        ## collection_of_distances-to-optimal 
        plt.subplot(2,2,2)
        plt.plot(to_fin_dis_col,color='green',alpha=0.2)
        plt.plot(self.running_avg(to_fin_dis_col,0.9),color='orchid')
        plt.title('dis_to_optimal')
        plt.ylabel('distance')
        plt.xlabel('updates')

        ## collection_of_distances-to-initial
        plt.subplot(2,2,3)
        plt.plot(to_ini_dis_col,color='green',alpha=0.2)
        plt.plot(self.running_avg(to_ini_dis_col,0.9),color='orchid')
        plt.title('dis_from_initial')
        plt.ylabel('distance')
        plt.xlabel('updates') 

        plt.tight_layout()        
        plt.show()
        """

        return
    
    # occupancy_measure_generation_fast(version)
    def occupancy_generation_fast(self,iteration=0):
        self.iteration = iteration #process_iteration_num

        # """
        # execution        
        file_location = f'/mnt/parscratch/users/acp21rmn/Mountain_Car_OTDD/algo_data/ddpg/samples/iter_{self.iteration}'
        final_policy_data_num = len(os.listdir(file_location)) - 1 #optimal_policy
        initial_policy_data_num = 0 #initial_policy
        #"""

        """        
        #testing
        file_location = f'ddpg_data/samples/iter_{self.iteration}'
        final_policy_data_num = 23619 #optimal_policy
        initial_policy_data_num = 0 #initial_policy
        # print('final_policy_data_num: ', final_policy_data_num)
        #"""

        # initial_policy & final_policy occupancy_measures
        final_policy_occup,_ = self.occupancy_measure_finite_timeless_fast(
                        final_policy_data_num,
                        file_location
                        )
        # print('final_policy_occup: ', final_policy_occup)
        # print('final_states_visits: \n', final_states_visits)

        initial_policy_occup,_ = self.occupancy_measure_finite_timeless_fast( #initl_states_visits
                        initial_policy_data_num,
                        file_location
                        )
        # print('initial_policy_occup: ', initial_policy_occup)

        # self.plot_state_visits(final_states_visits)
        # self.plot_state_visits(initl_states_visits)

        direct_dis,lip_con = self.OTDD(final_policy_occup,initial_policy_occup)
        # print('direct_dis: ', direct_dis)
        # print('lip_con: ', lip_con)

        max_lip_con = lip_con # maximum lipschitz constant
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
            successive_dis,lip_con_1 = self.OTDD(policy_occup_1,policy_occup_2)
            succ_dis_col.append(successive_dis)
            
            # #between final(optimal)_&_others
            to_final_dis,lip_con_2 = self.OTDD(policy_occup_1,final_policy_occup)
            to_fin_dis_col.append(to_final_dis)
            

            max_lip_con = max(lip_con_1,lip_con_2,max_lip_con)

            if update % 500 == 0:
                #cumulative data
                policy_trajectory_data = np.asarray([
                    direct_dis, #geodesic_distance
                    max_lip_con, #lipschitz_constant
                    succ_dis_col, #y_k (stepwise distance)
                    to_fin_dis_col, #x_k (distance_to_optimal)
                    ],dtype=object) #dtype=object
            
                self.save_policy_trajectories_per_update(policy_trajectory_data,update)

        """        
        policy_trajectory_data = np.asarray([
            direct_dis, #geodesic_distance
            max_lip_con, #lipschitz_constant
            succ_dis_col, #y_k (stepwise distance)
            to_fin_dis_col, #x_k (distance_to_optimal)
            ],dtype=object) #dtype=object
        
        self.save_policy_trajectories(policy_trajectory_data)
        # """

        """  
        ## testing_visualization      
        plt.figure(figsize=(12, 9))

        ## collection_of_successive_distances 
        plt.subplot(2,2,1)
        plt.plot(succ_dis_col,color='green',alpha=0.2)
        plt.plot(self.running_avg(succ_dis_col,0.9),color='orchid') 
        plt.title('stepwise')
        plt.ylabel('distance')
        plt.xlabel('updates')

        ## collection_of_distances-to-optimal 
        plt.subplot(2,2,2)
        plt.plot(to_fin_dis_col,color='green',alpha=0.2)
        plt.plot(self.running_avg(to_fin_dis_col,0.9),color='orchid')
        plt.title('dis_to_optimal')
        plt.ylabel('distance')
        plt.xlabel('updates')

        ## collection_of_distances-to-initial
        plt.subplot(2,2,3)
        plt.plot(to_ini_dis_col,color='green',alpha=0.2)
        plt.plot(self.running_avg(to_ini_dis_col,0.9),color='orchid')
        plt.title('dis_from_initial')
        plt.ylabel('distance')
        plt.xlabel('updates') 

        plt.tight_layout()        
        plt.show()
        """

        return
    
    #~~~~~~~~~~~~~~ Policy Evolution Plots ~~~~~~~~~~~~~~~~~~

    # (single)_policy_trajectory_evaluation
    def policy_trajectory_evaluation(self,iteration=0):
        self.iteration = iteration #process_iteration_num

        max_iterations = [23000,41500,0,29000,18500]
        num_updates = max_iterations[self.iteration]
        # num_updates = 18500 #[23000, 18500]
        file_name = f'traj_till{num_updates}.npy'
        file_location = f'ddpg_data/trajectory/iter_{self.iteration}' #file_location        
        file_path = abspath(join(this_dir,file_location,file_name)) #file_path: location + name
        data = np.load(file_path, allow_pickle=True) 

        # updates_file_location = f'policy_data/DQN/set_{problem_setting}/iter_{iteration}'
        # num_updates = len(os.listdir(updates_file_location)) - 1 #optimal_policy

        # print('data: ', data)
        # xxxx

        # data_extraction
        p = 6 #float_precision (decimal_places)

        direct_dis = np.round(data[0],p) #geodesic_distance
        max_lip_con = np.round(data[1],p) #lipschitz_constant_candidate
        chords = np.round(data[2] ,p) #y_k (stepwise distance) data[2]
        radii = np.round(data[3] ,p) #x_k (distance_to_optimal) data[3]

        # direct_dis, #geodesic_distance
        # max_lip_con, #lipschitz_constant
        # succ_dis_col, #y_k (stepwise distance)
        # to_fin_dis_col, #x_k (distance_to_optimal)

        # print('direct_dis: ', direct_dis)
        # print('max_lip_con: ', max_lip_con)
        # print('chords: \n', chords)
        # print('radii: \n', radii)
        # xxxx

        # data_collection
        curve_dis = np.round(sum(chords),p) # sum_{y_k}
        radii_diff = self.successive_diffs(radii,p) # x_k - x_{k+1}
      
        #distance-to-optimal not reduced
        # non_positive_indices = np.where(radii_diff <= 0) # non-improving transition 
        # non_positive_indices = non_positive_indices[0]

        #distance-to-optimal reduced
        positive_indices = np.where(radii_diff > 0) # improving transition
        positive_indices = positive_indices[0]

        # non_positive_chords = np.array([ chords[i] for i in non_positive_indices])
        positive_chords = np.array([ chords[i] for i in positive_indices])

        # wasted_effort = non_positive_chords.sum() #paid_to_further_from_goal
        usedful_effort = positive_chords.sum() #paid_to_closer_to_goal
        # print('wasted_effort: ', wasted_effort)
        # print('usedful_effort: ', usedful_effort)

        print('ALGO: ', self.algo)
        print('ESL: ', np.round(curve_dis/direct_dis,3)) #effort_of_sequential_learning
        print('OMR: ', np.round(usedful_effort/curve_dis,3)) #optimal_movement_ratio
        print('UC: ', num_updates)
            
        return

    # (single)_policy_evolution_plot
    def policy_evolution_plot(self,iteration=0):
        self.iteration = iteration #process_iteration_num

        max_iterations = [23000,29500,0,29000,18500]
        num_updates = max_iterations[self.iteration]
        # num_updates = 18500 #[23000, 18500]
        file_name = f'traj_till{num_updates}.npy'
        file_location = f'ddpg_data/trajectory/iter_{self.iteration}' #file_location        
        file_path = abspath(join(this_dir,file_location,file_name)) #file_path: location + name
        data = np.load(file_path, allow_pickle=True) 

        """        
        # print('data: ', data)

        # data_extraction
        p = 6 #float_precision (decimal_places)

        direct_dis = np.round(data[0],p) #geodesic_distance
        max_lip_con = np.round(data[1],p) #lipschitz_constant_candidate
        # chords = np.round(data[2],p) #y_k (stepwise distance)
        # radii = np.round(data[3],p) #x_k (distance_to_optimal)

        print('direct_dis: ', direct_dis)
        print('max_lip_con: ', max_lip_con)
        # print('chords: \n', chords)
        # print('radii: \n', radii)

        # direct_dis, #geodesic_distance
        # max_lip_con, #lipschitz_constant
        # succ_dis_col, #y_k (stepwise distance)
        # to_fin_dis_col, #x_k (distance_to_optimal)
        xxxx
        # """

        
        # data_extraction
        p = 6 #float_precision (decimal_places)

        chords = np.round(data[2],p) #y_k (stepwise distance)
        radii = np.round(data[3],p) #x_k (distance_to_optimal)

        # data_collection
        radii_diff = self.successive_diffs(radii,p) # x_k - x_{k+1}

        # plots
        plt.figure(figsize=(16, 10))

        ## Radius behaviour (1) 
        plt.subplot(3,2,1)
        plt.plot(radii/max(radii),'g')
        plt.plot(self.running_avg(radii/max(radii),0.99),'r')
        plt.ylabel('relative dis_to_opt',fontweight='bold')
        plt.xlabel('updates',fontweight='bold')
        plt.title(f'Radius_Behaviour [{self.algo}]',fontweight='bold')

        ## Chord behaviour (2) 
        plt.subplot(3,2,2)
        plt.plot(chords/max(chords),'b')
        plt.plot(self.running_avg(chords/max(chords),0.99),'r')
        plt.ylabel('relative stepwise_dis',fontweight='bold')
        plt.xlabel('updates',fontweight='bold')
        plt.title(f'Chord_Behaviour [{self.algo}]',fontweight='bold')

        ## Chords vs Radius (3)
        plt.subplot(3,2,3)
        opt_Xs = radii 
        opt_Ys = chords 
        opt_time = np.arange(len(opt_Xs))

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
        plt.title(f'Radius_vs_Chord [{self.algo}]',fontweight='bold')

        ## Chord vs Changing Radii (4)
        plt.subplot(3,2,4)
        XX = radii_diff
        YY = chords[:-1] #[:num_eval-1]
        tt = np.arange(len(XX))

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
        plt.title(f'$\delta$[Radii]_vs_Chords [{self.algo}]',fontweight='bold')
        plt.xlim(-.7,.7)

        plt.vlines(x = 0, #convergence_line
            ymin=min(YY),
            ymax=max(YY), 
            colors='black', 
            ls=':',)

        plt.tight_layout()
        
        
        ## 3D Policy Trajectory Visualization (5)
        Xs = radii/max(radii) #radii 
        Ys = chords/max(chords) #chords 
        self._3Dplots(Xs,Ys,name=f'{self.algo}') 

        plt.show()
        return
    
    # saving plots
    def save_images_for_paper(self,iteration=0):
        self.iteration = iteration #process_iteration_num

        max_iterations = [23000,29500,0,29000,18500]
        num_updates = max_iterations[self.iteration]
        # num_updates = 18500 #[23000, 18500]
        file_name = f'traj_till{num_updates}.npy'
        file_location = f'ddpg_data/trajectory/iter_{self.iteration}' #file_location        
        file_path = abspath(join(this_dir,file_location,file_name)) #file_path: location + name
        data = np.load(file_path, allow_pickle=True) 
        
        # data_extraction
        p = 10 #float_precision (decimal_places)

        chords = np.round(data[2],p) #y_k (stepwise distance)
        radii = np.round(data[3],p) #x_k (distance_to_optimal)

        # data_collection
        # radii_diff = self.successive_diffs(radii,p) # x_k - x_{k+1}
        areal = self.successive_area(chords,radii,p)
        OMR,updates = self.temporal_OMR(chords,radii,p)

        # plots
        # plt.figure(figsize=(16, 10))

        file_location = f'ddpg_data/images/iter_{self.iteration}' #file_location
        os.makedirs(join(this_dir,file_location), exist_ok=True) #create_file_location_if_nonexistent

        ## Radius behaviour (1) 
        # plt.subplot(3,2,1)
        plt.figure(figsize=(6,3))
        # plt.plot(radii/max(radii),'g',alpha=0.2)
        # plt.plot(self.running_avg(radii/max(radii),0.99),'r')
        plt.plot(radii,'g',alpha=0.2)
        plt.plot(self.running_avg(radii,0.99),'r')
        plt.ylabel('distance_to_optimal',fontweight='bold',fontsize=16)
        plt.xlabel('#updates',fontweight='bold',fontsize=16)
        plt.ylim(0,.85)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.tight_layout()
        file_name1 = f'to_opt.pdf'
        file_path1 = abspath(join(this_dir,file_location,file_name1)) #file_path: location + name
        plt.savefig(file_path1, format='pdf')

        ## Chord behaviour (2) 
        plt.figure(figsize=(6,3))
        # plt.plot(chords/max(chords),'b',alpha=0.2)
        # plt.plot(self.running_avg(chords/max(chords),0.99),'r')
        plt.plot(chords,'b',alpha=0.2)
        plt.plot(self.running_avg(chords,0.99),'r')
        plt.ylabel('stepwise_distance',fontweight='bold',fontsize=16)
        plt.xlabel('#updates',fontweight='bold',fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.ylim(0,.5)      
        plt.tight_layout()
        file_name2 = f'stepwise.pdf'
        file_path2 = abspath(join(this_dir,file_location,file_name2)) #file_path: location + name
        plt.savefig(file_path2, format='pdf')


        ## Areal Velocity behaviour (3) 
        plt.figure(figsize=(6,3))
        plt.plot(areal,'k',alpha=0.2)
        plt.plot(self.running_avg(areal,0.99),'r')
        # plt.plot(areal/max(areal),'k',alpha=0.2)
        # plt.plot(self.running_avg(areal/max(areal),0.99),'r')
        plt.ylabel('areal_velocity',fontweight='bold')
        plt.xlabel('updates',fontweight='bold')
        plt.ylim(0,0.16)
        plt.tight_layout()
        file_name3 = f'areal.pdf'
        file_path3 = abspath(join(this_dir,file_location,file_name3)) #file_path: location + name
        plt.savefig(file_path3, format='pdf')

        ## OMR behaviour (4) 
        plt.figure(figsize=(6,3))
        plt.plot(updates,OMR,'k')
        plt.ylabel('OMR(k)',fontweight='bold',fontsize=16)
        plt.xlabel('#updates',fontweight='bold',fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.ylim(0.4,0.6)
        plt.tight_layout()
        file_name4 = f'OMR.pdf'
        file_path4 = abspath(join(this_dir,file_location,file_name4)) #file_path: location + name
        plt.savefig(file_path4, format='pdf')

        ## 3D Policy Trajectory Visualization (5)
        Xs = radii  #radii/max(radii)
        Ys = chords  #chords/max(chords) 
        self._3Dplots(Xs,Ys,name=f'{self.algo}') 
        file_name5 = f'3D.pdf'
        file_path5 = abspath(join(this_dir,file_location,file_name5)) #file_path: location + name
        plt.savefig(file_path5, format='pdf') 

        plt.show()
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
        
        segments = 12
        div = 5 #len(chords)//segments
        omr,ups  = [],[]
        for k in range(len(chords)):
            if k%div == 0 and k <= (len(chords)-100) :
                # print(i)

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

                # print('OMR: ', np.round(usedful_effort/curve_dis,3))
                # omr.append(usedful_effort/curve_dis)
                omr.append(usedful_effort/total_travel)
                ups.append(k)
 
        return omr, ups 

    # 3Dplots (chords vs radii vs time)
    def _3Dplots(self,Xs,Ys,name='',fig=None,pos=None):
        time = np.arange(len(Xs))

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

        ax.set_zlabel('#updates',fontweight='bold',fontsize=16)
        ax.set_xlabel('distance_to_optimal',fontweight='bold',fontsize=16)
        ax.set_ylabel('stepwise_distance',fontweight='bold',fontsize=16)
        # ax.set_xlabel('distance-to-optimal',fontweight='bold')
        # ax.set_ylabel('stepwise-distance',fontweight='bold')

        ax.view_init(elev=20.,azim=-120)
        ax.set_xlim(0,max(Xs))
        ax.set_ylim(0,max(Ys))
        ax.set_zlim(0,max(time))

        ax.set_title(f'{name}',fontweight='bold',fontsize=16)
        ax.invert_zaxis()
        ax.set_xlim(0,.85)
        ax.set_ylim(0,.5)
        ax.tick_params(labelsize=14)
        # ax.set_visible(False)
        ax.ticklabel_format(axis='z',style='sci',scilimits=[0,2])

        plt.tight_layout()      
        return  
   
    #~~~~~~~~~~~~~~ OTDD ~~~~~~~~~~~~~~~~~~~~

    # Wasserstein: OTDD
    def OTDD(self,occu1,occu2):
        p_sGa1 = self.prob_states_given_actions(occu1)
        p_sGa2 = self.prob_states_given_actions(occu2)

        # print('p_sGa1: ', p_sGa1)
        # print('p_sGa2: ', p_sGa2)
        # print("=======================================")

        action_dis = defaultdict(int) 
        # print(self.y_pairs)

        # action_to_action distance
        for i in self.discrete_actions_pairs:
            action_dis[i] = self.inner_wasserstein(p_sGa1[i[0]],p_sGa2[i[1]])
        # print('action_dis: \n', action_dis)

        # state-action-pair_to_state-action-pair distance
        otdd, lip_con = self.outer_wasserstein(occu1,occu2,action_dis)
        # print('otdd: ', otdd)
        # print('lip_con: ', lip_con)
        # xxx
        return otdd, lip_con
    
    # Mapping actions -> state_distributions
    def prob_states_given_actions(self,occup):
        p_sGa = defaultdict(lambda: defaultdict(int)) #probabilities [states | actions = a]
        p_a = defaultdict(int) #probabilities [actions]

        # p(A)
        for k in self.discrete_actions:
            for i in self.discrete_pos:
                for j in self.discrete_vel:
                    p_a[k] += occup[((i,j),k)]
        # print('p_a: \n', p_a)

        # p(S|A)
        for k in self.discrete_actions:
            for i in self.discrete_pos:
                for j in self.discrete_vel:
                    if occup[((i,j),k)] == 0: 
                        p_sGa[k][(i,j)] = 0
                    else:
                        p_sGa[k][(i,j)] = occup[((i,j),k)]/p_a[k]
        # print('p_sGa: \n', p_sGa)
        # print('p_sGa.size: \n', len(p_sGa))
        # print("=======================================")

        return p_sGa

    # Inner_Wasserstein
    def inner_wasserstein(self,p_sGa,q_sGa):
        z = self.discrete_states
        CostMatrix = ot.dist(z,z, metric='cityblock') #cost matrix: 'euclidean' 
        # print('CostMatrix: ',CostMatrix)
                
        P = np.asarray(list(p_sGa.values())) #prob_dis
        Q = np.asarray(list(q_sGa.values())) #prob_dis

        # print('P: \n', P)
        # print('Q: \n', Q)
        # print("=======================================")

        if P.sum() != 0 and Q.sum() != 0:
            val = ot.emd2(  P, #A_distribution 
                            Q, #B_distribution
                            M = CostMatrix, #cost_matrix pre-processing
                            numItermax=int(1e6)
                            ) #OT matrix
        else:
            val = None
        # print('val: ', val)

        return val
    
    # Outer_Wasserstein
    def outer_wasserstein(self,occu1,occu2,action_dis):
        z1 = list(occu1.keys())  #state-action space
        z2 = list(occu2.keys())  #state-action space
        CostMatrix, lip_con = self.outer_cost(z1,z2,action_dis) #cost matrix
        # print('CostMatrix: ',CostMatrix)
        # print('CostMatrix: ',len(CostMatrix))
       
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

        return val, lip_con
    
    # Outer_cost_metric
    def outer_cost(self,z1,z2,action_dis):        
        len_z1 = len(z1) 
        len_z2 = len(z2)  
        m = np.zeros([len_z1,len_z2])
        lip_con = 0
        for idx, i in enumerate(z1):
            for jdx, j in enumerate(z2):
                vector = np.asarray(i[0]) - np.asarray(j[0])

                dis_states = LA.norm( vector, 1) #'cityblock'
                dis_actions = action_dis[ i[1], j[1]]
                # print('dis_states: ', dis_states)
                # print('dis_actions: ', dis_actions)

                rew_diff = np.abs(self.rewards_function(i[0],i[1]) - self.rewards_function(j[0],j[1]) ) #rewards_difference
                # print('rew_diff: ', rew_diff)

                if dis_actions == None:
                    m[idx][jdx] = 0
                else: 
                    dis_state_action_pairs = dis_states + dis_actions
                    m[idx][jdx] = dis_state_action_pairs

                    # Lipschitz_constant_calculator
                    if dis_state_action_pairs > 1e-4: #d_sa != 0 (prevent extreme values)
                        """
                        d_sa > 1e-4: ensures numerical stability, 
                        focusing on pairs with meaningful information
                        """
                        ratio = rew_diff/dis_state_action_pairs
                        if ratio > lip_con:
                            lip_con = ratio
        # print('m: ', m) 
        # print('lip_con: ', lip_con) 
        return m, lip_con


    #~~~~~~~~~~~~~~ Sanity Testing ~~~~~~~~~~~~~~~~~~~ 

    # occupancy_measuring_testing && value_functions
    def occupancy_testing(self,iteration=0):
        self.iteration = iteration #process_iteration_num

        #NN architecture
        # num_states = self.num_states
        # num_actions = self.num_actions
        # num_hidden_l1 = 64 
        # num_hidden_l2 = 64 

        #file_location
        file_location = f'ddpg_data/samples/iter_{self.iteration}'

        optimal_policy_data_num = 0 #0 #23619 #4608 #len(os.listdir(file_location)) - 1
        print('optimal_policy_data_num: ', optimal_policy_data_num)

        value_optimal_1 = self.state_value_function_finite(
                                  optimal_policy_data_num,
                                  file_location
                                  )
        print('value_optimal_1: ', value_optimal_1)
               
        occupancy_optimal, avg_eps_length_optimal,_ = self.occupancy_measure_finite_timeless(
                                  optimal_policy_data_num,
                                  file_location
                                  )
        print('occupancy_optimal: ', occupancy_optimal)
        print('occupancy_optimal[1]: ', occupancy_optimal[2])

        value_optimal_2 = self.value_function_occupancy_measure(occupancy_optimal, avg_eps_length_optimal)
        print('value_optimal_2: ', value_optimal_2)
        print(np.abs(value_optimal_1 - value_optimal_2))
        
        xxx
        value_error = []
        avg_length = []
        value_col = []
        value_relative_error = []
        for update in range(len(os.listdir(file_location)) - 1): #executes_optimal_policy_model

            # retrieve policy_model_value
            value_1 = self.state_value_function_finite(
                                    update,
                                    file_location
                                    )
            # print('value_1: ', value_1)

            occupancy, avg_eps_length,_ = self.occupancy_measure_finite_timeless(
                                    update,
                                    file_location
                                    )
            
            value_2 = self.value_function_occupancy_measure(occupancy, avg_eps_length)
            # print('value_2: ', value_2)

            value_col.append(value_1)
            # value_error.append(np.abs(value_1 - value_2))
            avg_length.append(avg_eps_length)
            value_relative_error.append(100*np.abs(value_1 - value_2)/value_1)

        
        ## collection_of_values
        plt.plot(value_col)
        plt.title('Values')
        plt.ylabel('values')
        plt.xlabel('updates')
        plt.show()        
              
        ## collection_of_value_errors      
        # plt.plot(value_error)
        # plt.title('Value Error')
        # plt.ylabel('error')
        # plt.xlabel('updates')
        # plt.show()

        ## collection_of_episode_lengths 
        plt.plot(avg_length)
        plt.title('avg episode length')
        plt.ylabel('length')
        plt.xlabel('updates')
        plt.show()

        ## collection_of_relative_value_errors 
        plt.plot(value_relative_error)
        plt.title('Relative Value Error')
        plt.ylabel('rel. error')
        plt.xlabel('updates')
        plt.show()

        return

    # testing_policy_performance
    def performance_test(self,iteration=0):
        self.iteration = iteration #process_iteration_num

        #NN architecture
        num_states = self.num_states
        num_actions = self.num_actions
        num_hidden_l1 = 64 
        num_hidden_l2 = 64 

        #file_location
        file_location = f'ddpg_data/models/iter_{self.iteration}'

        # """     
        ## single_model_testing   
        step = 54246 #6588 #16589 #26589 
        n = 5 #num_tests

        for _ in range(n): #test_n_times
            # retrieve policy_model
            self.pi_net = self.retrieve_policy_model(
                                                num_states, 
                                                num_actions, 
                                                num_hidden_l1, 
                                                num_hidden_l2,
                                                step,
                                                file_location)
            # Visualize_performance
            _,act,ret,stt,trj = self.learnt_agent()
            print('act: ', act)
            print('stt: ', stt)
            print('ret: ', ret)
            self.show_state_trajectory(trj)
        # """
        
        """
        ## multiple_model_testing
        for step in [731, 732, 733, 734, 735, 736, 737, 738, 739, 740, 741, 742, 743]:

            n = 5 #num_tests
            for _ in range(n): #test_n_times
                # retrieve policy_model
                self.pi_net = self.retrieve_policy_model(
                                                    num_states, 
                                                    num_actions, 
                                                    num_hidden_l1, 
                                                    num_hidden_l2,
                                                    step,
                                                    file_location)
                # Visualize_performance
                _,act,ret,stt,trj = self.learnt_agent()
                print('act: ', act)
                print('stt: ', stt)
                print('ret: ', ret)
                self.show_state_trajectory(trj)
        # """
        print('complete!')
        return 

    #~~~~~~~~~~~~~~ Agent Evaluation ~~~~~~~~~~~~~~~~~~~

    #illustration_of_state_trajectory
    def show_state_trajectory(self,trj):
        path = np.array(trj)
        # print('trj[0]: \n', path[0:10])
        # print('trj: \n', path.shape)
        # print('trj[c1]: \n', path[:,0])

        x =  path[:,0] #positions
        y = path[:,1] #velocities
        plt.plot(x,y)

        plt.plot(x[0],y[0],'^g')
        plt.plot(x[-1],y[-1],'or')

        plt.xlabel('positions')
        plt.ylabel('velocities')
        plt.title('Phase Plot')
        plt.show()

        return

    #trained_agent (for: evaluation)
    def learnt_agent(self):
        self.env_test = gym.make('MountainCarContinuous-v0')  #start_env
        
        rew_list, act_list, st_list, trj = [],[],[],[]
        # Xpos_list = []
        t_done = False
        ep_return = 0 # episode return

        t_obs,_ = self.env_test.reset() #reset environment
        t_state = torch.tensor(t_obs).float().unsqueeze(0)
        
        while not t_done:
            t_action = self.select_action(t_state, True)  
            t_next_obs, t_reward, t_terminated, t_truncated, _ = self.env_test.step(t_action)
            t_done = t_terminated or t_truncated #done_function (new_approach)
            ep_return += t_reward
            t_next_state = torch.tensor(t_next_obs).float().unsqueeze(0)

            """
            t_pos = t_next_obs[0] #position_along: x-axis
            if t_pos >= 0.45:
                print('t_pos: ', t_pos)
            Xpos_list.append(t_pos)
            """
            
            #experience_buffer_data
            st_list.append(t_state.detach().cpu().numpy()[0]) #stt_list
            rew_list.append(t_reward) #rew_list

            if t_done:
                trj = deepcopy(st_list)
                trj.append( t_next_state.detach().cpu().numpy()[0] )

            act_list.append(t_action[0])
            t_state = t_next_state
        
        self.env_test.close() #shut_env

        # results_display
        # print('returns: ', ep_return)
        # self.angular_plot(Xpos_list,0)

        return rew_list , act_list, ep_return, st_list, trj
  
    #action_selection_mechanism   
    def select_action(self, state, evaluation_episode=False):
        if evaluation_episode:
            action = self.get_action_deterministically(state)
        else:
            action = self.get_action_stochastically(state)
        return action

    #stochastic_action
    def get_action_stochastically(self, state):
        action = self.pi_net(state).detach().cpu().numpy()[0]  

        #from: https://github.com/amuta/DDPG-MountainCarContinuous-v0/blob/master/README.md      
        # action = np.clip(action*.2 + self.noise(),-1,1)
        action = np.clip(action + self.noise(),-1,1)

        """        
        action += self.noise()*max(0,self.epsilon)
        action += self.noise() 
        action = np.clip(action,-1,1)
        """

        return action

    #greedy_action
    def get_action_deterministically(self, state):
        action = self.pi_net(state).detach().cpu().numpy()[0]
        return action
    
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
        # xx

        self.save_state_action_samples(meta_data,steps) #save_state-action_samples
        return 

    #~~~~~~~~~~~~~~ Value Functions & Occupancy Measures ~~~~~~~~~~~~~~~~~~~~

    # Occupancy Measure for stationary Policy in Finite MDP (faster version)
    def occupancy_measure_finite_timeless_fast(self,steps,file_location): #using_finite_episodes
        
        ## load_data
        file_name = f'sample_{steps}.npy'
        file_path = abspath(join(this_dir,file_location,file_name)) #file_path: location + name
        data = np.load(file_path, allow_pickle=True)

        ## trajectories
        x1 = data[0] #states_in_dataset
        y1 = data[1] #actions_in_dataset
        tx1 = np.asarray(data[4]) #len(states)_of_runs
        ty1 = np.asarray(data[5]) #len(actions)_of_runs

        states_dict,actions_dict = {},{}

        # iterate_each_rollout
        for i in range(len(tx1)-1):
            states = x1[tx1[i]:tx1[i+1]] #states_per_rollout
            actions = y1[ty1[i]:ty1[i+1]] #actions_per_rollout
            states_dict[i], actions_dict[i] = states, actions

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
        total_counts = sum(pair_counts.values())
        avg_eps_length = sum(steps_counts.values())/len(steps_counts)
            
        #calculate_probability_distribution
        for j in pair_counts:
                # probs[j] = round(pair_counts[j]/total_counts,self.p)
                probs[j] = pair_counts[j]/total_counts
        
        return probs, avg_eps_length #, policy_state_visits
    
    # state_action_discretization
    def partitions_fast(self,states,actions):
        
        # actions
        bins_action_counts = np.zeros(self.num_bins_action)
        for action in actions:
            for i in range(self.num_bins_action):
                if self.bins_action[i] <= action < self.bins_action[i+1]:
                    bins_action_counts[i] += 1 
                    break
        bins_action_counts[-1] += np.sum(actions == 1) # add action = 1 to last bin
        # print('bins_action_counts: ', bins_action_counts)

        dis_actions = []
        for value, repeat in zip(self.discrete_actions,bins_action_counts):
            if repeat != 0:
                dis_actions.extend([value]*int(repeat))

        # states
        bins_pos_vel_counts = np.zeros([self.num_bins,self.num_bins])
        states = np.array(states)
        pos =  states[:,0] #positions
        vel = states[:,1] #velocities

        for po,ve in zip(pos,vel):
            for i in range(self.num_bins):
                if self.bins_pos[i] <= po < self.bins_pos[i+1]:
                    for j in range(self.num_bins):
                        if self.bins_vel[j] <= ve < self.bins_vel[j+1]:
                            bins_pos_vel_counts[i][j] += 1
                            break
        cnt = len([ 1 for i in states if (i == [0.6,0.07]).all() ]) # add state = [0.6,0.07] to last bin
        bins_pos_vel_counts[-1][-1] += cnt
        # print('bins_pos_vel_counts: ', bins_pos_vel_counts)

        dis_states = []
        for i in range(len(bins_pos_vel_counts)):
            for j in range(len(bins_pos_vel_counts)):
                value = (self.discrete_pos[i],self.discrete_vel[j])
                repeat = bins_pos_vel_counts[i][j]
                if repeat != 0:
                    dis_states.extend([ value ]*int(repeat))
        return dis_states, dis_actions #, bins_pos_vel_counts

    # Occupancy Measure for stationary Policy in Finite MDP
    def occupancy_measure_finite_timeless(self,steps,file_location): #using_finite_episodes
        
        ## load_data
        file_name = f'sample_{steps}.npy'
        file_path = abspath(join(this_dir,file_location,file_name)) #file_path: location + name
        data = np.load(file_path, allow_pickle=True)

        ## trajectories
        x1 = data[0] #states_in_dataset
        y1 = data[1] #actions_in_dataset
        tx1 = np.asarray(data[4]) #len(states)_of_runs
        ty1 = np.asarray(data[5]) #len(actions)_of_runs

        states_dict,actions_dict = {},{}

        # iterate_each_rollout
        for i in range(len(tx1)-1):
            states = x1[tx1[i]:tx1[i+1]] #states_per_rollout
            actions = y1[ty1[i]:ty1[i+1]] #actions_per_rollout
            states_dict[i], actions_dict[i] = states, actions

        steps_counts = defaultdict(int)
        pair_counts = defaultdict(int)
        probs = defaultdict(int) #probabilities [occupancy_measure]

        # iterate_each_episode_rollout
        for k in actions_dict.keys():
            states = states_dict[k] #states_per_episode_rollout
            actions = actions_dict[k] #actions_per_episode_rollout
            steps_counts[k] = len(actions) #num_decisions_per_episode

            dis_states, dis_actions, policy_state_visits = self.partitions(states,actions) #state_action_discretization
            
            for j in zip(dis_states,dis_actions): 
                pair_counts[j] += 1
        total_counts = sum(pair_counts.values())
        avg_eps_length = sum(steps_counts.values())/len(steps_counts)
            
        #calculate_probability_distribution
        for j in pair_counts:
                # probs[j] = round(pair_counts[j]/total_counts,self.p)
                probs[j] = pair_counts[j]/total_counts
        
        return probs, avg_eps_length, policy_state_visits

    # state_action_discretization
    def partitions(self,states,actions):
        
        # actions
        bins_action_counts = np.zeros(self.num_bins)
        for action in actions:
            for i in range(self.num_bins):
                if self.bins_action[i] <= action < self.bins_action[i+1]:
                    bins_action_counts[i] += 1 
                    break
        bins_action_counts[-1] += np.sum(actions == 1) # add action = 1 to last bin
        # print('bins_action_counts: ', bins_action_counts)

        dis_actions = []
        for value, repeat in zip(self.discrete_actions,bins_action_counts):
            if repeat != 0:
                dis_actions.extend([value]*int(repeat))

        # states
        bins_pos_vel_counts = np.zeros([self.num_bins,self.num_bins])
        states = np.array(states)
        pos =  states[:,0] #positions
        vel = states[:,1] #velocities

        for po,ve in zip(pos,vel):
            for i in range(self.num_bins):
                if self.bins_pos[i] <= po < self.bins_pos[i+1]:
                    for j in range(self.num_bins):
                        if self.bins_vel[j] <= ve < self.bins_vel[j+1]:
                            bins_pos_vel_counts[i][j] += 1
                            break
        cnt = len([ 1 for i in states if (i == [0.6,0.07]).all() ]) # add state = [0.6,0.07] to last bin
        bins_pos_vel_counts[-1][-1] += cnt
        # print('bins_pos_vel_counts: ', bins_pos_vel_counts)

        dis_states = []
        for i in range(len(bins_pos_vel_counts)):
            for j in range(len(bins_pos_vel_counts)):
                value = (self.discrete_pos[i],self.discrete_vel[j])
                repeat = bins_pos_vel_counts[i][j]
                if repeat != 0:
                    dis_states.extend([ value ]*int(repeat))
        return dis_states, dis_actions, bins_pos_vel_counts

    # state_visits_plot
    def plot_state_visits(self,visits):
        visits_distr = visits/visits.sum()
        sns.heatmap(100*visits_distr,
            cbar=True,
            linewidths=0.01,
            xticklabels=np.round(self.discrete_pos,2),
            yticklabels=np.round(self.discrete_vel,2),
            # annot=True,
            # fmt=".2f"
            )   
        # plt.tick_params(labeltop=True, labelbottom=False, bottom=False, top=True)   
        plt.title(f'State_Visitation (%)')
        plt.xlabel('discrete positions')
        plt.ylabel('discrete velocities')

        plt.show()
        return
    
    # Finite value_function
    def state_value_function_finite(self,steps,file_location): #using_finite_episodes
        
        ## load_data
        file_name = f'sample_{steps}.npy'
        file_path = abspath(join(this_dir,file_location,file_name)) #file_path: location + name
        data = np.load(file_path, allow_pickle=True)

        ## trajectories
        x1 = data[0] #states_in_dataset
        y1 = data[1] #actions_in_dataset
        tx1 = np.asarray(data[4]) #len(states)_of_runs
        ty1 = np.asarray(data[5]) #len(actions)_of_runs

        states_dict,actions_dict = {},{}

        # iterate_each_rollout
        for i in range(len(tx1)-1):
            states = x1[tx1[i]:tx1[i+1]] #states_per_rollout
            actions = y1[ty1[i]:ty1[i+1]] #actions_per_rollout
            states_dict[i], actions_dict[i] = states, actions

        gamma = 0.99 #discount factor
        avg_cum_rew = 0
        cum_rew_list = []
        for k in states_dict.keys():
            cum_rew = 0
            for _,(n_state,n_action) in enumerate(zip(states_dict[k],actions_dict[k])):
                reward = self.rewards_function(n_state,n_action)
                cum_rew += reward #(gamma**step)*reward 
            cum_rew_list.append(cum_rew)
        avg_cum_rew = np.mean(cum_rew_list)

        # print('avg_cum_rew: ', avg_cum_rew)
        # xxx
        return avg_cum_rew
    
    # MountainCar_rewards_model
    def rewards_function(self,state,action):
        rew = -0.1*(action**2)
        if state[0] >= 0.45:
            rew = 100
        # print('rew: ', rew)
        # xxx
        return rew
    
    # Finite value_function (using Occupancy Measure)
    def value_function_occupancy_measure(self, probs, avg_eps_length):
        cum_product = 0
        for i in self.discrete_pos:
            for j in self.discrete_vel:
               for k in self.discrete_actions:
                    cum_product += self.rewards_function([i,j],k)*probs[((i,j),k)]
        value  = avg_eps_length*cum_product
        return value

    #~~~~~~~~~~~~~~ Retrieving & Saving ~~~~~~~~~~~~~~~~~~~~

    # saving_policy_trajectories (fast_Track)
    def save_policy_trajectories_per_update(self,meta_data,update):
        meta_data = np.asarray( meta_data, dtype=object) #data_to_save

        file_name = f'traj_till{update}'
        # file_location = f'ddpg_data/paths/iter_{self.iteration}' #file_location

        file_location = f'/mnt/parscratch/users/acp21rmn/Mountain_Car_OTDD/algo_data/ddpg/cums_reduced/iter_{self.iteration}' #file_location
        # /mnt/parscratch/users/acp21rmn/Mountain_Car_OTDD/algo_data/ddpg/cums/iter_

        os.makedirs(join(this_dir,file_location), exist_ok=True) #create_file_location_if_nonexistent
        file_path = abspath(join(this_dir,file_location,file_name)) #file_path: location + name
        np.save(file_path,meta_data) #save_command
        return

    # saving_policy_trajectories
    def save_policy_trajectories(self,meta_data):
        meta_data = np.asarray( meta_data, dtype=object) #data_to_save

        file_name = f'traj_{self.iteration}'
        # file_location = f'ddpg_data/paths/iter_{self.iteration}' #file_location

        file_location = f'/mnt/parscratch/users/acp21rmn/Mountain_Car_OTDD/algo_data/ddpg/paths/iter_{self.iteration}' #file_location
        # /mnt/parscratch/users/acp21rmn/Mountain_Car_OTDD/algo_data/ddpg/paths/iter_

        os.makedirs(join(this_dir,file_location), exist_ok=True) #create_file_location_if_nonexistent
        file_path = abspath(join(this_dir,file_location,file_name)) #file_path: location + name
        np.save(file_path,meta_data) #save_command
        return

    # saving_state-action_samples
    def save_state_action_samples(self,meta_data,steps):
        meta_data = np.asarray( meta_data, dtype=object) #data_to_save

        file_name = f'sample_{steps}'
        # file_location = f'ddpg_data/samples/iter_{self.iteration}' #file_location

        file_location = f'/mnt/parscratch/users/acp21rmn/Mountain_Car_OTDD/algo_data/ddpg/samples/iter_{self.iteration}' #file_location
        # /mnt/parscratch/users/acp21rmn/Mountain_Car_OTDD/algo_data/ddpg/samples/iter_

        os.makedirs(join(this_dir,file_location), exist_ok=True) #create_file_location_if_nonexistent
        file_path = abspath(join(this_dir,file_location,file_name)) #file_path: location + name
        np.save(file_path,meta_data) #save_command
        return

    # retrieve policy model
    def retrieve_policy_model(self,
                            num_states, 
                            num_actions, 
                            num_hidden_l1, 
                            num_hidden_l2,
                            steps,
                            file_location):
        
        file_name = f'actor_{steps}.pth'     
        file_path = abspath(join(this_dir,file_location,file_name)) #file_path: location + name

        #declare model
        pi_net = Actor(
                        num_states,
                        num_actions,
                        num_hidden_l1,
                        num_hidden_l2).to(device)     
        
        pi_net.load_state_dict(torch.load(file_path))            
        return pi_net


    #~~~~~~~~~~~~~~ Plotting ~~~~~~~~~~~~~~~~~~~~

    # plotting returns vs episodes
    def plot_returns(self, score_col, con_eps):
        # plt.figure(figsize=(8,15))
        plt.title('Return Plot')
        plt.plot(score_col, '-*',color='orchid',alpha=0.2)
        plt.plot(self.running_avg(score_col,0.99), '-*',color='red')
        plt.plot(self.rolling(score_col,100), '-^',color='green')
        plt.vlines(x = con_eps,
                   ymin=min(score_col),
                   ymax=max(score_col), 
                   colors='black', 
                   ls=':',
                #    label='Convergence'
                   )
        plt.xlabel('episodes')
        plt.ylabel('returns')
        plt.show()

    # running average function
    def running_avg(self,score_col,beta = 0.99):
        cum_score = []
        run_score = score_col[0]
        for score in score_col:
            run_score = beta*run_score + (1.0 - beta)*score
            cum_score.append(run_score)
        return cum_score

    # rolling window avg function
    def rolling(self,dataset, window=100): #20
        N = window #episode collection
        cumsum = [0] #list of cumulative sum
        moving_aves = [] #list of moving averages
        for i, x in enumerate(dataset, 1): # i: tracks index-values & x:tracks list values
            cumsum.append(cumsum[i-1] + x) #cumulative sum
            if i>=N:
                moving_ave = (cumsum[i] - cumsum[i-N])/N
                moving_aves.append(moving_ave)
        return moving_aves


#Execution
if __name__ == '__main__':
    agent = setting( rew_setting=1, #[rew_setting, num_eps]
                      n_eps=100,
                      ) 
    # agent.main()
    
    """ 
    #Generation_of_state-action_samples   
    for i in range(3,5): #2
        agent.policy_data_generation(iteration=i,
                                    )
    #"""

    """ 
    #Generation_of_policy_trajectory   
    agent.occupancy_generation_fast(iteration=4,
                                    )
    #"""

    """ 
    #Generation_of_policy_trajectory   
    agent.occupancy_generation(iteration=0,
                                    )
    #"""
    
    """
    #Performance_evaluation
    agent.performance_test(iteration=3)

    # """

    """
    #Occupancy_measuring_testing && value_functions
    agent.occupancy_testing(iteration=0)
    # """

    """
    # Analysis Policy_trajectory
    agent.policy_trajectory_evaluation(iteration=0)
    # """

    """
    # Visualizing Policy_trajectory
    agent.policy_evolution_plot(iteration=0)
    # """

    # """
    # Save Policy_trajectory file (pdf)
    agent.save_images_for_paper(iteration=0) #paper iteration = 0
    # """