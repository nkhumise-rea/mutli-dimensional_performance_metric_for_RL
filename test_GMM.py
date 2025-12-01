# =========================================================
# Empirical Joint Density Estimation p(x, y) via GMM
# =========================================================

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from nflows.distributions import StandardNormal
from nflows.transforms import (
    MaskedAffineAutoregressiveTransform,
    ReversePermutation,
    BatchNorm,
    CompositeTransform,
)
from nflows.flows import Flow
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

#normalize
scaler = StandardScaler()
xy_scaled = scaler.fit_transform(xy) #shape (N,a+b)
gmm = GaussianMixture(
            n_components=3, #12 
            covariance_type='full', 
            reg_covar=1e-5,  # <- stabilizes small determinants
            # random_state=0
            ).fit(xy_scaled)

# print(gmm.means_)

# print(xy[0])
# xxx 

for i, cov in enumerate(gmm.covariances_):
    print(i, np.linalg.det(cov))

print('BIC: ')
for k in range(2, 15):
    gmm = GaussianMixture(n_components=k, covariance_type='full', reg_covar=1e-5).fit(xy_scaled)
    print(k, gmm.bic(xy_scaled))


# Evaluate density
# print(gmm.predict([xy[0],xy[50],xy[50]]))
print(xy_scaled[:3])
print(np.exp((gmm.score_samples(xy_scaled))))

xxx
p_z = np.exp(gmm.score_samples(xy))  # shape (N,)