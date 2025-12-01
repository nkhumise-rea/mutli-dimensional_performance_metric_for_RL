import gymnasium as gym
import gymnasium_robotics
import matplotlib.pyplot as plt
import time
import os
import numpy as np
import torch
from torch.distributions import Distribution, Normal, Independent
from numpy import random
import argparse
import sys
from torch.utils.tensorboard import SummaryWriter
from tianshou.data import Collector, HERReplayBuffer
from tianshou.policy import PPOPolicy
from tianshou.trainer import onpolicy_trainer
from tianshou.utils.net.common import Net, get_dict_state_decorator, ActorCritic
from tianshou.utils.net.continuous import ActorProb, Critic #ContinuousCritic #ContinuousActorProbabilistic, 
from tianshou.utils import TensorboardLogger
from tianshou.env import DummyVectorEnv #dnt
device = "cuda" if torch.cuda.is_available() else "cpu"

import torch.nn.functional as F

class NormalizedNet(Net):
    def forward(self, obs, state=None, info={}):
        # Flatten dict observation (if not already flattened by dict_state_dec)
        
        # print('obs: ', obs)
        # print('state: ', state)
        # xxx
        x, state = super().forward(obs, state, info)
        # Normalize along feature dimension

        # print('x: ', x)
        # print('state: ', state)
        x = (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + 1e-8)
        return x, state


parser = argparse.ArgumentParser()
parser.add_argument("--n_count", type=int, default=0, help="number of bins")
args = parser.parse_args()
 
sys.path.insert(0, "..")
gym.register_envs(gymnasium_robotics)

##Parameters
num_envs = 1
hidden_sizes = [256,256,256] #ref: Plappert (2018)
tau = 50e-3 #ref: Plappert (2018)
gamma = 0.99
lr = 1e-5
exploration_noise = 0.2 #ref: Plappert (2018)
batch_size = 256 #ref: Plappert (2018)
buffer_size = 1e6
num_steps_collect = 25000
seed = 0

##Environment
head = "rgb_array" #"human" #
env = gym.make('FetchReach-v4', render_mode=head, max_episode_steps=100)
# env = gym.make('FetchPush-v4', render_mode=head)
# env = DummyVectorEnv([lambda: env for _ in range(num_envs)])

#env shapes
state_shape = {
    'observation' : env.observation_space['observation'].shape[0],
    'achieved_goal' : env.observation_space['achieved_goal'].shape[0],
    'desired_goal' : env.observation_space['desired_goal'].shape[0],
    }
action_shape = env.action_space.shape[0] 
action_space = env.action_space
observation_space = env.observation_space['observation']

max_action = env.action_space.high[0]
obs = env.reset()

dict_state_dec, flat_state_shape = get_dict_state_decorator (
    state_shape = state_shape,
    keys = ['observation','achieved_goal','desired_goal'],
    )

##model
#--preprocessing networks
# Net.log_std.data.fill_(-0.5)
net_a = dict_state_dec(Net)(
    flat_state_shape,
    hidden_sizes=hidden_sizes, 
    activation=torch.nn.Tanh,
    device=device)

net_c = dict_state_dec(Net)(
    flat_state_shape,
    hidden_sizes=hidden_sizes,
    activation=torch.nn.Tanh,
    device=device)

#--Actor, 2 x Critics networks
actor = dict_state_dec(ActorProb)(
# actor = ActorProb(
    net_a,
    action_shape,
    max_action = max_action,
    device = device,
    unbounded = True,
    conditioned_sigma = True
    ).to(device)
critic = dict_state_dec(Critic)(net_c,device = device).to(device)
actor_critic = ActorCritic(actor,critic)
##################################################################
# Orthogonal init
# torch.nn.init.constant_(actor.sigma_param, -0.5)
for m in actor_critic.modules():
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.orthogonal_(m.weight,gain=np.sqrt(2))
        torch.nn.init.zeros_(m.bias)

for m in actor.mu.modules():
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.zeros_(m.bias)  
        m.weight.data.copy_(0.01 * m.weight.data)

# for m in actor.modules():
#     if isinstance(m, torch.nn.Linear):
#         torch.nn.init.orthogonal_(m.weight)
#         torch.nn.init.constant_(m.bias)

# for m in critic.modules():
#     if isinstance(m, torch.nn.Linear):
#         torch.nn.init.orthogonal_(m.weight)
#         torch.nn.init.constant_(m.bias)

# Safe initial std
# actor.log_std.data.fill_(-0.5)  # std ~ 0.6
##################################################################
optim = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=lr)
# optim = torch.optim.Adam(actor.parameters(),lr=lr)

# def dist_fn(mean, std):
#     std = torch.clamp(std, min=1e-6, max=1.0)
#     return Independent(Normal(mean, std), 1)

def dist_fn(mean: tuple[torch.Tensor], std: tuple[torch.Tensor]) -> Distribution:
    # print('mean.shape: ', mean.shape)
    # print('std.shape: ', std.shape)

    # print('mean: ', mean)
    # print('std: ', std)
    loc = torch.clamp(mean, -10, 10)
    scale = torch.clamp(std, min=1e-6, max=1.0)
    print('loc: ', loc)
    print('scale: ', scale)
    return Independent(Normal(loc, scale), 1)

policy = PPOPolicy(
    actor,
    critic,
    optim,
    dist_fn = dist_fn, #for continuous control
    action_space = action_space,
    discount_factor = gamma,
    reward_normalization = True,
    action_scaling = True,
    # action_bound = 'tanh',
    # observation_space = observation_space,
    # max_grad_norm = 0.5,
    )

start_time = time.time()

##buffer
num_step_episode = 100 #No. steps per episode
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

#"""
##Training loop_3
log_path = os.path.join("./runs/Reach/ppo_HER_{}".format(args.n_count))
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
    torch.save(policy.state_dict(),"./torch_models/Reach/ppo_HER_{}.pth".format(args.n_count))

def stop_fn(mean_rewards):
    if mean_rewards > -4.0:
        return True
    else:
        return False

result_f = onpolicy_trainer(
    policy,
    train_collector,
    test_collector,
    max_epoch = 20,
    step_per_epoch = 10000, 
    repeat_per_collect = 4,
    step_per_collect = 1,
    episode_per_test = 1,
    batch_size = batch_size,
    save_best_fn = save_best_fn,
    #stop_fn = stop_fn,
    logger = logger,
    update_per_step = 1,
    test_in_train = False,
    show_progress = True,
)

end_time = time.time()
duration = end_time - start_time
print('duration: ', duration)