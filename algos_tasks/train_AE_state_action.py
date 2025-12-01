import gymnasium as gym
import gymnasium_robotics
import matplotlib.pyplot as plt
import time
import os

import argparse
import sys
from os.path import join, abspath

from copy import copy, deepcopy
import numpy as np
from collections import namedtuple, deque
import matplotlib.pyplot as plt
import time

#ML
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

#pytorch
import torch
import torch.nn as nn
import torch.optim as optima
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.utils.data import Dataset, DataLoader


device = "cuda" if torch.cuda.is_available() else "cpu"


parser = argparse.ArgumentParser()
parser.add_argument("--n_count", type=int, default=55)
parser.add_argument("--encoding_dim", type=int, default=4)
args = parser.parse_args()

this_dir = os.getcwd()  # or wherever your root is
sys.path.insert(0, "..")
gym.register_envs(gymnasium_robotics)

##Parameters
hidden_sizes = [256,128] 
batch_size = 256 #56 
seed = 0
steps_per_episode = 100 #100 #50
task = 'Reach' #'Push', 'Slide'
num_eps = 1000

##Autoencoder_Parameters
encoding_dim = args.encoding_dim #latent_state_dimension
action_encoding_dim = 2
lr = 0.003 #learning_rate
num_epochs = 1000 #training_epochs

file_location = 'autoencoder_training_data'

##Environment
head = "rgb_array" #"human" 
if task == 'Reach':
    env = gym.make('FetchReach-v4', render_mode=head,max_episode_steps=steps_per_episode)
elif task == 'Push':
    env = gym.make('FetchPush-v4', render_mode=head, max_episode_steps=steps_per_episode)
    # env = gym.make('FetchPushDense-v4', render_mode=head, max_episode_steps=steps_per_episode)
else:
    env = gym.make('FetchSlide-v4', render_mode=head, max_episode_steps=steps_per_episode)
    # env = gym.make('FetchSlideDense-v4', render_mode=head, max_episode_steps=100)

#env shapes
state_shape = {
    'observation' : env.observation_space['observation'].shape[0],
    'achieved_goal' : env.observation_space['achieved_goal'].shape[0],
    'desired_goal' : env.observation_space['desired_goal'].shape[0],
    }
action_shape = env.action_space.shape[0] 

print(action_shape)
xxx
max_action = env.action_space.high[0]
obs = env.reset()

class StateAE(nn.Module):
    def __init__(self,num_states,encoding_dim,num_hidden_l1,num_hidden_l2):
        super(StateAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(num_states,num_hidden_l1),
            nn.ReLU(),
            nn.Linear(num_hidden_l1,num_hidden_l2),
            nn.ReLU(),
            nn.Linear(num_hidden_l2,encoding_dim),
            # nn.Tanh(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim,num_hidden_l2),
            nn.ReLU(),
            nn.Linear(num_hidden_l2,num_hidden_l1),
            nn.ReLU(),
            nn.Linear(num_hidden_l1,num_states),
            # nn.Tanh(),
        )

    # def forward(self,state):
    #     latent = self.encoder(state)
    #     estimated_state = self.decoder(latent)
    #     return latent, estimated_state
    
    def encode(self,state): return self.encoder(state)
    def decode(self,z_state): return self.decoder(z_state)

    
class ActionAE(nn.Module):
    def __init__(self,num_actions,action_encoding_dim,num_hidden_l1,num_hidden_l2):
        super(ActionAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(num_actions,num_hidden_l1),
            nn.ReLU(),
            nn.Linear(num_hidden_l1,num_hidden_l2),
            nn.ReLU(),
            nn.Linear(num_hidden_l2,action_encoding_dim),
            # nn.Tanh(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(action_encoding_dim,num_hidden_l2),
            nn.ReLU(),
            nn.Linear(num_hidden_l2,num_hidden_l1),
            nn.ReLU(),
            nn.Linear(num_hidden_l1,num_actions),
            # nn.Tanh(),
        )

    # def forward(self,action):
    #     latent = self.encoder(action)
    #     estimated_action = self.decoder(latent)
    #     return latent, estimated_action
    
    def encode(self,action): return self.encoder(action)
    def decode(self,z_state): return self.decoder(z_state)
    

class LatentDynamics(nn.Module):
    def __init__(self,encoding_dim,action_encoding_dim,num_hidden_l1,num_hidden_l2):
        super(LatentDynamics, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(encoding_dim+action_encoding_dim,num_hidden_l2),
            nn.ReLU(),
            nn.Linear(num_hidden_l2,num_hidden_l1),
            nn.ReLU(),
            nn.Linear(num_hidden_l1,encoding_dim),
        )
    def forward(self,z_s,z_a):
        return self.net(torch.cat([z_s,z_a],dim=-1))

class RANDOM_AGENT():
    def __init__(self):
        self.env = env
        self.num_episodes = num_eps #50 #total number of episodes
        self.duration = None 
        self.count = 0

    #print_model_weights
    def print_model(self,model):
        #for p in model.parameters(): #w/t names
        #    print(p.data)
        for n,p in model.named_parameters(): #w/ names
            # print(n)
            print(p.data)

    ## Policy 
    def output_action(self):
        action = self.env.action_space.sample() #default, low=0, high=1
        return action #.detach().cpu().numpy() #[0]

    def evaluate(self):
        #start_time = time.time()
        
        ## Configurations
        #hyperameters
        num_episodes = self.num_episodes
        print_every = 10
        steps = 0

        stored_states,stored_actions = [],[]
        for episode in range(num_episodes):
            done = False
            steps = 0 
            obs = self.env.reset()
            # print('obs: ', obs)
            # print('obs: ', obs[0]['observation'])

            state = obs[0]['observation']
            # print('state: ', state)

            while not done:
                stored_states.append(state.tolist())
                steps += 1

                action = self.output_action()
                stored_actions.append(action.tolist())
                # print('action: ', action)
                # xxxx
              
                ##Batch(obs=i, act=i, rew=i, terminated=0, truncated=0, obs_next=i + 1, info={})
                next_obs,_, done, _, _ = self.env.step(action)
                # print('next_obs: ', next_obs)

                if steps == steps_per_episode:
                    done = True
                
                next_state = next_obs['observation']
                # print('next_state: ', next_state)

                state = next_state

        # print('stored_states: ', stored_states)
        # print('stored_actions: ', stored_actions)
        # print('stored_states: ', pd.DataFrame(stored_states))
        # xxx

        dataset_states = pd.DataFrame(stored_states)
        dataset_states.to_csv(f'{file_location}/dim_{encoding_dim}/data_states.csv',index=False)

        dataset_actions = pd.DataFrame(stored_actions)
        dataset_actions.to_csv(f'{file_location}/dim_{encoding_dim}/data_actions.csv',index=False)
        return stored_states,stored_actions #pd.DataFrame(stored_states) #dataset_size = num_eps*steps_per_episode
    

class Processing():
    def __init__(self,data):
        i = 0
        # data = torch.tensor(data, dtype=torch.float32)
        # print('data: ', data)

        dataset = self.normalisation(data)
        # print('dataset: ', dataset)
        # xxx

        self.loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=True)
        
    def normalisation(self,data):
        X = data
        # print('X: \n', X)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        # print('X_scaled: \n', X_scaled)

        X_tensor = torch.FloatTensor(X_scaled)
        # print('X_tensor: \n', X_tensor)

        #save_scaler_parameters
        joblib.dump(scaler,f'{file_location}/dim_{encoding_dim}/scaler.pkl')
        return X_tensor
    

    def compute_loss(self,state_ae,action_ae,f_dyn,batch):
        s_t,a_t,s_tp1 = batch

        #encode
        z_state = state_ae.encode(s_t)
        z_action = action_ae.encode(a_t)
        z_next_state = state_ae.encode(s_tp1)

        #decode for reconstruction
        est_state = state_ae.decode(z_state)
        est_action = state_ae.decode(z_action)

        #Losses
        recon_state = F.mse_loss(est_state,s_t)
        recon_action = F.mse_loss(est_action,a_t)
        align = self.latent_alignment_loss(z_state,z_action)
        dyn = self.latent_dynamics_loss(f_dyn,z_state,z_action,z_next_state)

        total_loss = recon_state + recon_action + 0.1*align + 0.1*dyn
        return total_loss, dict(recon_state=recon_state.item(),
                                recon_action=recon_action.item(),
                                align=align.item(),
                                dyn=dyn.item(),
                                )
    
    #alignment_loss: couples embedding directly - ensures co-occurring
    #(s,a) pairs sit near each other
    def latent_alignment_loss(self,z_s,z_a):
        #cosine similarity encourages smooth mapping
        #enforces directional alignment in latent space
        cos_sim = F.cosine_similarity(z_s,z_a, dim=-1)
        return (1 - cos_sim).mean()
    
    #dynamics_loss: latent transitions coherent with original dynamics
    def latent_dynamics_loss(self,f_dyn,z_s,z_a,z_s_next):
        pred = f_dyn(z_s,z_a)
        return F.mse_loss(pred,z_s_next)

    def train(self):

        #model_params
        num_states = state_shape['observation'] #self.env.observation_space.shape[0]
        num_hidden_l1 = hidden_sizes[0]  
        num_hidden_l2 =  hidden_sizes[1]  
        num_actions = action_shape
        
        # print('num_states: ', num_states)
        # print('num_hidden_l1: ',num_hidden_l1)
        # print('num_hidden_l2: ',num_hidden_l2)


        #declare model
        # self.model = Autoencoder(num_states,
        #               encoding_dim,
        #               num_hidden_l1,
        #               num_hidden_l2).to(device)
        
        state_ae = StateAE(num_states,
                           encoding_dim,
                           num_hidden_l1,
                           num_hidden_l2).to(device)

        action_ae = ActionAE(num_actions,
                             action_encoding_dim,
                             num_hidden_l1,
                             num_hidden_l2).to(device)

        f_dyn = LatentDynamics(encoding_dim,
                               action_encoding_dim,
                               num_hidden_l1,
                               num_hidden_l2).to(device)
        
        criterion  = nn.MSELoss()
        optimizer = optima.Adam(
            list(state_ae.parameters()) +
            list(action_ae.parameters()) +
            list(f_dyn.parameters()),lr=lr)

        # self.print_model(self.model) #print model_weights
        # print('new_game')

        losses = []
        #training the autoencoder
        for epoch in range(num_epochs): 
            for states in self.loader:
                
                # print('states: ', states)
                states = states.to(device)

                loss, logs = self.compute_loss(state_ae,action_ae,f_dyn,batch)
                losses.append(loss.item())
                # print('loss: ', loss)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.6f}")

        # self.learning_plot(losses)
        # self.save_model(self.model)
        print('Complete')

    def save_model(self,model):
        file_name = f"dim_{encoding_dim}/encoder_.pth"
        os.makedirs(join(this_dir, file_location), exist_ok=True)
        file_path = abspath(join(this_dir, file_location, file_name))
        
        # save the actor (policy) network
        torch.save(model.state_dict(), file_path)

    def learning_plot(self,losses):
        # plt.style.use('fivethirtyeight')
        plt.figure(figsize=(8,5))
        plt.plot(losses,label='Loss')
        plt.xlabel('Updates')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Loss Curve for Autoencoder')
        # plt.show()
        plt.savefig(f'{file_location}/dim_{encoding_dim}/learning_curve', dpi=300, bbox_inches='tight')


if __name__ == '__main__':  
    agent = RANDOM_AGENT() # head=[0:GUI, 1:DIRECT]
    
    stored_states,stored_actions = agent.evaluate()
    # print('stored_states: \n', stored_states)
    # print('stored_actions: \n', stored_actions)

    xxx
    mod = Processing(stored_states)
    mod.train()
