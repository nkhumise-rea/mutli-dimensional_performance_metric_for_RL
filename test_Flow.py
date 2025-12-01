import torch
from nflows.distributions import StandardNormal
from nflows.transforms import MaskedAffineAutoregressiveTransform, CompositeTransform
from nflows.flows import Flow

# =========================================================
# 0. Initialisation
# =========================================================
import numpy as np
from os.path import dirname, abspath, join
import sys
#set directory/location
this_dir = dirname(__file__)
agent_dir = abspath(join(this_dir,'..')) 
sys.path.insert(0,agent_dir)

##tasks
task = 'Reach' #'Push', 'Slide'
algo = 'ddpg' #'td3' #'sac' #'ddpg' #'td3'   
cnt = 0 
encoding_dim = 25 #
iteration = 0

# =========================================================
# 1. Prepare your data
# =========================================================
## load_data
steps = 1800
file_name = f'sample_{steps}.npy'
file_location = f'{algo}/{task.lower()}/samples_{cnt}/iter_{iteration}/dim_{encoding_dim}'

file_path = abspath(join(this_dir,file_location,file_name)) #file_path: location + name
data = np.load(file_path, allow_pickle=True)
# print('data: ', data)
# xxx

## trajectories
x1 = data[0] #states_in_dataset
y1 = data[1] #actions_in_dataset
tx1 = np.asarray(data[5]) #len(states)_of_runs
ty1 = np.asarray(data[5]) #len(actions)_of_runs


# print('x1: ', len(x1))
# print('y1: ', len(y1))

# print('tx1: ', tx1)
# print('tx2: ', ty1)

x,y = [],[]
# iterate_each_rollout
for i in range(len(tx1)-1):
    states = x1[tx1[i]:tx1[i+1]] #states_per_rollout
    actions = y1[ty1[i]:ty1[i+1]] #actions_per_rollout

    #create state & action datasets
    x.extend(states) #shape (N,a)
    y.extend(actions) #shape (N,b)

print('x: ',np.array(x).shape)
print('y: ',np.array(y).shape)

#create state-action dataset
xy = np.hstack([x,y])


# =========================================================
# 2. FLOW::
# =========================================================
Z = torch.tensor(xy, dtype=torch.float32)

base_dist = StandardNormal(shape=[Z.shape[1]])
transforms = [MaskedAffineAutoregressiveTransform(features=Z.shape[1], hidden_features=128) for _ in range(5)]
flow = CompositeTransform(transforms)

flow_model = Flow(flow, base_dist)

optimizer = torch.optim.Adam(flow_model.parameters(), lr=1e-3)
for epoch in range(1000):
    optimizer.zero_grad()
    loss = -flow_model.log_prob(Z).mean()
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1:02d}/{1000}, NLL: {loss:.4f}")



# Evaluate density
p_z = torch.exp(flow_model.log_prob(Z))
print(p_z)





