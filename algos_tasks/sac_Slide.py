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
from tianshou.data import Collector, HERReplayBuffer, HERVectorReplayBuffer
from tianshou.policy import SACPolicy
from tianshou.trainer import offpolicy_trainer
from tianshou.utils.net.common import Net, get_dict_state_decorator
from tianshou.utils.net.continuous import ActorProb, Critic
from tianshou.utils import TensorboardLogger
from tianshou.env import DummyVectorEnv, SubprocVectorEnv #dnt
device = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser()
parser.add_argument("--n_count", type=int, default=1020)
parser.add_argument("--n_iteration", type=int, default=0)
args = parser.parse_args()
 
sys.path.insert(0, "..")
gym.register_envs(gymnasium_robotics)

##Parameters
num_envs = 8
hidden_sizes = [256,256,256] #ref: Plappert (2018) 
# hidden_sizes = [256,256] #ref: ChatGPT 
tau = 5e-3 #ref: ChatGPT
gamma = 0.99
alpha = 0.2
lr_alpha = 5e-4 #ref: ChatGPT
lr = 3e-4 #ref: ChatGPT
batch_size = 256 #ref: Plappert (2018) / ChatGPT
buffer_size = 1e6
num_steps_collect = 25000
seed = 0
task = 'Slide' #'Push' #'Reach' 

##Environment
head = "rgb_array" #"human" # 
if task == 'Reach':
    steps_per_episode = 100
    env = gym.make('FetchReach-v4', render_mode=head,max_episode_steps=steps_per_episode)
elif task == 'Push':
    steps_per_episode = 50
    env = gym.make('FetchPush-v4', render_mode=head, max_episode_steps=steps_per_episode)
    # env = gym.make('FetchPushDense-v4', render_mode=head, max_episode_steps=steps_per_episode)
else:
    steps_per_episode = 100 #50
    env = gym.make('FetchSlide-v4', render_mode=head, max_episode_steps=steps_per_episode)
    # env = gym.make('FetchSlideDense-v4', render_mode=head, max_episode_steps=100)
# env = DummyVectorEnv([lambda: env for _ in range(num_envs)])
envs = SubprocVectorEnv([lambda: env for _ in range(num_envs)])

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

##model
#--preprocessing networks
net_a = dict_state_dec(Net)(
    flat_state_shape,
    hidden_sizes=hidden_sizes, 
    device=device)

net_c1 = dict_state_dec(Net)(
    flat_state_shape,
    action_shape, 
    hidden_sizes=hidden_sizes,
    concat = True, 
    device=device)

net_c2 = dict_state_dec(Net)(
    flat_state_shape,
    action_shape, 
    hidden_sizes=hidden_sizes,
    concat = True, 
    device=device)

#--Actor, 2 x Critics networks
actor = dict_state_dec(ActorProb)(
    net_a,
    action_shape,
    max_action = max_action,
    device = device,
    unbounded = True,
    conditioned_sigma = True).to(device)

critic1 = dict_state_dec(Critic)(net_c1,device = device).to(device)
critic2 = dict_state_dec(Critic)(net_c2,device = device).to(device)
actor_optim = torch.optim.Adam(actor.parameters(),lr=lr)
critic1_optim = torch.optim.Adam(critic1.parameters(),lr=lr)
critic2_optim = torch.optim.Adam(critic2.parameters(),lr=lr)


log_alpha = torch.zeros(1, requires_grad=True).to(device) #ref: ChatGPT
alpha_optim = torch.optim.Adam([log_alpha], lr=lr_alpha) #ref: ChatGPT

policy = SACPolicy(
    actor,
    actor_optim,
    critic1,
    critic1_optim,
    critic2,
    critic2_optim,
    tau = tau,
    gamma = gamma,
    # alpha = alpha,
    alpha=(-action_shape,log_alpha,alpha_optim)
    )

start_time = time.time()

##buffer
num_step_episode = steps_per_episode #100 #No. steps per episode
def compute_reward_fn(achieved_goal, desired_goal):
    # return env.compute_reward(achieved_goal, desired_goal, info={})
    return env.unwrapped.compute_reward(achieved_goal, desired_goal, info={})

# buf = HERReplayBuffer(
#                     size = buffer_size,
#                     compute_reward_fn = compute_reward_fn,
#                     horizon = num_step_episode,
#                     future_k = 8,
#                       )

buf = HERVectorReplayBuffer(
                    total_size=buffer_size,
                    buffer_num=num_envs,
                    compute_reward_fn=compute_reward_fn,
                    horizon=num_step_episode,
                    future_k=8,
)

##collectors
train_collector = Collector(policy,envs,buf,
                            exploration_noise=True,
                            # n_step=num_step_episode,
                            )
test_collector = Collector(policy,envs)
train_collector.collect(
    n_step = num_steps_collect,
    random = True,
    )

#prep before training
train_collector.reset()
test_collector.reset()
env.reset()
buf.reset()

#"""
this_dir = os.getcwd()  # or wherever your root is

##Training loop_3
log_path = os.path.join("../runs/{}/sac_{}".format(task,args.n_count))
writer = SummaryWriter(log_path) 
train_interval = 1
test_interval = 1
update_interval = 1

##logger
#now = datetime.datetime.now().strftime("%y%m%d-%H%M%S")   
logger = TensorboardLogger(
                            writer,
                            train_interval=train_interval,
                            test_interval=test_interval, 
                            )

def save_best_fn(policy):
    file_name = "sac_{}.pth".format(args.n_count)
    file_location = f'../torch_models/{task}'
    os.makedirs(join(this_dir, file_location), exist_ok=True)
    file_path = abspath(join(this_dir, file_location, file_name))
    
    # save the actor (policy) network
    torch.save(policy.state_dict(), file_path)

def stop_fn(mean_rewards):
    if mean_rewards > -4.0:
        return True
    else:
        return False

def policy_model_saver(policy, step, epoch):
    M = 20
    if step % M == 0:
        file_name = f"actor_{step}.pth"
        file_location = f'../sac/{task}/count_{args.n_count}/iter_{args.n_iteration}'
        os.makedirs(join(this_dir, file_location), exist_ok=True)
        file_path = abspath(join(this_dir, file_location, file_name))
        
        # save the actor (policy) network
        torch.save(policy.state_dict(), file_path)
        # print(f"Saved policy weights at step {step} -> {file_path}") 

test_fn = lambda epoch, env_step: policy_model_saver(policy, env_step,epoch)

result_f = offpolicy_trainer(
        policy,
        train_collector,
        test_collector,
        max_epoch = 10, #max_epoch*step_per_epoch = 1e6
        step_per_epoch = 100000, #ref: ChatGPT 
        step_per_collect = 1,
        episode_per_test = 1,
        batch_size = batch_size,
        save_best_fn = save_best_fn,

        #stop_fn = stop_fn,
        logger = logger,
        # train_fn = test_fn,

        update_per_step = 4, #ref: ChatGPT #gradient_updates_per_step
        test_in_train = False,
        show_progress = True,
        )

end_time = time.time()
duration = end_time - start_time
print('duration: ', duration)

