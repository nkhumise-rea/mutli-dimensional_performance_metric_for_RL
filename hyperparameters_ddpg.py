import gymnasium as gym
import gymnasium_robotics
import matplotlib.pyplot as plt
import time
import os
import numpy as np
import torch
from numpy import random
import argparse
import sys
from os.path import join, abspath
from torch.utils.tensorboard import SummaryWriter
from tianshou.data import Collector, HERReplayBuffer
from tianshou.policy import DDPGPolicy
from tianshou.trainer import offpolicy_trainer
from tianshou.utils.net.common import Net, get_dict_state_decorator
from tianshou.utils.net.continuous import Actor, Critic
from tianshou.utils import TensorboardLogger
from tianshou.env import DummyVectorEnv #dnt
from tianshou.exploration import GaussianNoise, OUNoise
device = "cuda" if torch.cuda.is_available() else "cpu"

print('device: ', device)

parser = argparse.ArgumentParser()
parser.add_argument("--n_count", type=int, default=0)
parser.add_argument("--n_iteration", type=int, default=1)
args = parser.parse_args()
 
sys.path.insert(0, "..")
gym.register_envs(gymnasium_robotics)

##constant_parameters
num_envs = 1
hidden_sizes = [256,256,256] #ref: Plappert (2018)
tau = 50e-3 #ref: Plappert (2018) [Reach]
gamma = 0.99
batch_size = 256 #ref: Plappert (2018) [Reach]
buffer_size = 1e6
num_steps_collect = 10000 #25000
seed = 0

##varying_parameters
lr_actor_list = [1e-4, 1e-3, 1e-2] 
lr_critic_list = [1e-4, 1e-3, 1e-2] 
# exploration_noise_list = [0.2]
exploration_noise = 0.2

##tasks
task = 'Reach' #'Push', 'Slide'
step_per_epoch = 10000 #10k[Reach]
algo = 'ddpg'   

print(f'algo: DDPG, task: {task}')

##Horizon variants
steps_per_episode_list = [25,50,75,100,125]

##initialising_env
head = "rgb_array" #"human" # 
env = gym.make('FetchReach-v4',render_mode=head)

#env shapes
state_shape = {
    'observation' : env.observation_space['observation'].shape[0],
    'achieved_goal' : env.observation_space['achieved_goal'].shape[0],
    'desired_goal' : env.observation_space['desired_goal'].shape[0],
    }
action_shape = env.action_space.shape[0] 
max_action = env.action_space.high[0]
obs = env.reset()

dict_state_dec, flat_state_shape = get_dict_state_decorator (
    state_shape = state_shape,
    keys = ['observation','achieved_goal','desired_goal'],
    )

##training_loop
for steps_per_episode in steps_per_episode_list:
    start_time = time.time()
    env = gym.make('FetchReach-v4',
                   render_mode=head,
                   max_episode_steps=steps_per_episode)  
    num_hyp = 1
    for lr_actor in lr_actor_list:
        for lr_critic in lr_critic_list:
            # """   
            ##model
            #--preprocessing networks
            net_a = dict_state_dec(Net)(
                flat_state_shape,
                hidden_sizes=hidden_sizes, 
                device=device)

            net_c = dict_state_dec(Net)(
                flat_state_shape,
                action_shape, 
                hidden_sizes=hidden_sizes,
                concat = True, 
                device=device)

            #--Actor, 2 x Critics networks
            actor = dict_state_dec(Actor)( #deterministic actor
                net_a,
                action_shape,
                max_action = max_action,
                device = device).to(device)

            critic = dict_state_dec(Critic)(net_c,device = device).to(device)
            actor_optim = torch.optim.Adam(actor.parameters(),lr=lr_actor)
            critic_optim = torch.optim.Adam(critic.parameters(),lr=lr_critic)
            sigma = max_action*exploration_noise

            policy = DDPGPolicy(
                actor,
                actor_optim,
                critic,
                critic_optim,
                tau = tau,
                gamma = gamma,
                exploration_noise = GaussianNoise(sigma=sigma), #OUNoise(),
                estimation_step = 1,
                )

            ##buffer
            num_step_episode = steps_per_episode #100 #No. steps per episode
            def compute_reward_fn(achieved_goal, desired_goal):
                return env.unwrapped.compute_reward(achieved_goal, desired_goal, info={})

            buf = HERReplayBuffer(
                                size = buffer_size,
                                compute_reward_fn = compute_reward_fn,
                                horizon = num_step_episode,
                                future_k = 8, 
                                )

            ##collectors
            train_collector = Collector(policy,env,buf,exploration_noise=True)
            test_collector = Collector(policy,env)
            train_collector.collect(
                n_step = num_steps_collect,
                random = True,
                )

            #prep before training
            train_collector.reset()
            test_collector.reset()
            env.reset()
            buf.reset()

            ##
            this_dir = os.getcwd()  # or wherever your root is
            log_path = os.path.join(f"Hyperparams/{task.lower()}/{algo}/maxSteps_{steps_per_episode}/hyp_{num_hyp}/iter_{args.n_count}")
            writer = SummaryWriter(log_path) 
            train_interval = 1
            test_interval = 1
            update_interval = 1

            ##logger
            logger = TensorboardLogger(
                                        writer,
                                        train_interval=train_interval,
                                        test_interval=test_interval, 
                                        )

            def save_best_fn(policy):
                file_name = f"ddpg_hypSet_{num_hyp}_iter_{args.n_count}.pth"
                file_location = f'torch_models/hyperparams/{task}/{algo}/maxSteps_{steps_per_episode}'
                os.makedirs(join(this_dir, file_location), exist_ok=True)
                file_path = abspath(join(this_dir, file_location, file_name))
                
                # save the actor (policy) network
                torch.save(policy.state_dict(), file_path)
                return

            result_f = offpolicy_trainer(
                    policy,
                    train_collector,
                    test_collector,
                    max_epoch=10, #train_steps = max_epoch*step_per_epoch 
                    step_per_epoch = step_per_epoch, #20k,10k[Reach] - 100k[Push]
                    step_per_collect = 1,
                    episode_per_test = 1,
                    batch_size = batch_size,
                    save_best_fn = save_best_fn,
                    logger = logger,
                    update_per_step = 1, #[Reach]
                    test_in_train = False,
                    show_progress = True,
                    )
            # """
            
            # print('Hyp_set#: ', num_hyp) 
            num_hyp += 1

    end_time = time.time()
    duration = end_time - start_time
    print(f'duration: {duration/60} min')

