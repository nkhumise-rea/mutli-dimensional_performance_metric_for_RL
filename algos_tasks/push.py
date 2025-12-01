import pybullet as p
import pybullet_data
import numpy as np
import time
import gymnasium as gym
import gymnasium_robotics


gym.register_envs(gymnasium_robotics)

env = gym.make('FetchPush-v4', render_mode="human")
# env = gym.make('FetchPushDense-v4', render_mode="human")

obs, info = env.reset()

achieved_goal = np.array([1, 0, 1])
desired_goal = np.array([0, 0, 0])

print(obs["achieved_goal"])
# xxx
# r = env.compute_reward(obs["achieved_goal"], obs["desired_goal"], info)

r = env.unwrapped.compute_reward(obs["achieved_goal"], obs["desired_goal"], info)

print(r)
env.close()
xxx
for _ in range(100000000):
    action = env.action_space.sample()

    print('action: ', action)
    obs, reward, done, trunc, info = env.step(action)
    env.render()
env.close()


