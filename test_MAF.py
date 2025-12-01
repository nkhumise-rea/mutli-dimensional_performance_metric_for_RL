# =========================================================
# Empirical Joint Density Estimation p(x, y) via MAF (nflows)
# =========================================================

import torch
import matplotlib.pyplot as plt
import copy
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
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

#GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
# Suppose x ∈ ℝ^a and y ∈ ℝ^b
# Replace this with your own dataset
# N, a, b = 5000, 7, 7    # Example: total 15D
# x = torch.randn(N, a)
# y = 2 * torch.randn(N, b) + 0.5
# xy = torch.cat([x, y], dim=1)  # shape (N, a+b)
# dim = xy.shape[1]

## load_data
steps = 100 #1900
file_name = f'sample_{steps}.npy'
file_location = f'{algo}/{task.lower()}/samples_{cnt}/iter_{iteration}/dim_{encoding_dim}_100k'

file_path = abspath(join(this_dir,file_location,file_name)) #file_path: location + name
data = np.load(file_path, allow_pickle=True)
# print('data: ', data)
# xxx

## trajectories
x1 = data[0] #states_in_dataset
y1 = data[1] #actions_in_dataset
tx1 = np.asarray(data[5]) #len(states)_of_runs
ty1 = np.asarray(data[5]) #len(actions)_of_runs


print('x1: ', len(x1))
print('y1: ', len(y1))

print('tx1: ', tx1)
print('tx2: ', ty1)
# xxx

x,y = [],[]
# iterate_each_rollout
for i in range(len(tx1)-1):
    states_ = x1[tx1[i]:tx1[i+1]] #states_per_rollout
    actions_ = y1[ty1[i]:ty1[i+1]] #actions_per_rollout
    # states_dict[i], actions_dict[i] = states, actions

    #create state & action datasets
    x.extend(states_) #shape (N,a)
    y.extend(actions_) #shape (N,b)

states = np.array(x)
actions = np.array(y)
# t_states = torch.tensor(states,dtype=torch.float32)
# t_actions = torch.tensor(actions,dtype=torch.float32)
print('states: ',states.shape)
print('actions: ',actions.shape)
# xxx

dim_states = states.shape[1]
dim_actions = actions.shape[1]
num_samples = states.shape[0]
print('dim: ',dim_states) #shape (N,a+b)
print('dim: ',dim_actions) #shape (N,a+b)

#create state-action dataset
state_actions = np.hstack([states,actions]) #shape (N,a+b)
print('state_actions: ',state_actions.shape) #shape (N,a+b)

dim = state_actions.shape[1]
print('dim: ',dim) #shape (N,a+b)

state_actions_ = torch.tensor(state_actions,dtype=torch.float32)
print('state_actions_: ',state_actions_.shape) #shape (N,a+b)
# xx
# =========================================================
# 2. Normalize (important for stable training)
# =========================================================
"""
scaler = StandardScaler()
xy_scaled = torch.tensor(scaler.fit_transform(state_actions_), dtype=torch.float32)
dim = xy_scaled.shape[1]

dataset = TensorDataset(xy_scaled)
loader = DataLoader(dataset, batch_size=256, shuffle=True)
"""

#Normalize
scaler_state = StandardScaler()
scaler_action = StandardScaler()
# s1 = scaler.fit_transform(states)
# print('s1: ', s1.shape)
# print(s1.dtype)

# print("scaler.n_features_in_ =", scaler.n_features_in_)

# s2 = scaler.inverse_transform(s1[:3])
# print('s2: ', s2.shape)
# print(s2.dtype)
# xxx




t_states = torch.tensor(
    scaler_state.fit_transform(states),
    dtype=torch.float32)
t_actions = torch.tensor(
    scaler_action.fit_transform(actions),
    dtype=torch.float32)
print('t_states: ',t_states.shape)
print('t_actions: ',t_actions.shape)

#prepare 80%/20% for training/testing
num_data = int(num_samples*.8) #80%
print('num_data: ', num_data)

#data splitting
train_states = t_states[:num_data]
train_actions = t_actions[:num_data]

test_states = t_states[num_data:]
test_actions = t_actions[num_data:]
print('train_states: ',train_states.shape)
print('test_states: ',test_states.shape)
print('train_actions: ',train_actions.shape)
print('test_actions: ',test_actions.shape)

#Split loader
train_loader =  DataLoader( 
                    TensorDataset(
                        train_states,
                        train_actions),
                    batch_size=512,
                    # shuffle=True
                    )

test_loader =  DataLoader( 
                    TensorDataset(
                        test_states,
                        test_actions),
                    batch_size=512, #256
                    # shuffle=True
                    )

# print('len(loader.dataset):', len(loader.dataset))
# print('num_samples: ', num_samples)
# print(xy_scaled.shape)
# xxx
# =========================================================
# 3. Define Masked Autoregressive Flow (MAF)
# =========================================================
def create_maf_flow(dim_states, dim_actions, num_flows=5, hidden_features=128, num_blocks=2):
    """Construct a Masked Autoregressive Flow with optional normalization and permutation."""
    transforms = []
    for _ in range(num_flows):
        transforms.append(
            MaskedAffineAutoregressiveTransform(
                features=dim_states,
                hidden_features=hidden_features,
                context_features=dim_actions,
                num_blocks=num_blocks,
                # use_residual_blocks=True,
                # random_mask=False,
                activation=nn.ReLU()
            )
        )
        # Normalization + permutation improve expressiveness
        # transforms.append(BatchNorm(features=dim))
        transforms.append(ReversePermutation(features=dim_states))
    transform = CompositeTransform(transforms)
    base_dist = StandardNormal([dim_states])
    return Flow(transform, base_dist)

# Create flow model
flow = create_maf_flow(dim_states=dim_states, 
                       dim_actions=dim_actions,
                       num_flows=5, 
                       hidden_features=128, 
                       num_blocks=2)
flow.to(device)

# print('here!!!')
# xxx
# =========================================================
# 4. Train the flow
# =========================================================
optimizer = torch.optim.Adam(flow.parameters(), 
                             lr=5e-4,#1e-3,
                             weight_decay=1e-6
                             )
epochs = 1#00 #200 #100 #50
best_test_loss = np.inf
best_epoch = 0
best_model = copy.deepcopy(flow.state_dict())

# for epoch in range(epochs):
#     total_loss = 0.0
#     for (batch,) in loader:
#         loss = -flow.log_prob(batch).mean()  # negative log-likelihood
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item() * len(batch)
#     print(f"Epoch {epoch+1:02d}/{epochs}, NLL: {total_loss/len(dataset):.4f}")

loss_dict = dict(train=[], test=[])
for epoch in range(epochs):

    #train
    train_loss = 0.0
    flow.train()
    for states_b,actions_b in train_loader:
        states_b,actions_b  = states_b.to(device),actions_b.to(device) 
        loss = -flow.log_prob(inputs=states_b,
                              context=actions_b
                              ).mean()  # negative log-likelihood
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item() #* states_b.size(0) #batch_size
        train_loss /= len(train_loader) #train_states.shape[0] #length_total_training_dataset
    loss_dict['train'].append(train_loss)

    # if(epoch+1)%1==0:
    #     print(f"Epoch {epoch+1:02d}/{epochs}, NLL: {train_loss:.4f}")

    #validate
    test_loss = 0.0
    with torch.no_grad():
        flow.eval()
        for states_b,actions_b in test_loader:
            states_b,actions_b  = states_b.to(device),actions_b.to(device) 
            loss_1 = -flow.log_prob(inputs=states_b,
                                context=actions_b
                                ).mean() # negative log-likelihood
            test_loss += loss_1.item() #* states_b.size(0) #batch_size
            test_loss /= len(test_loader) #test_states.shape[0] #length_total_training_dataset
        loss_dict['test'].append(test_loss)

    #save best model for use at the end
    if test_loss < best_test_loss:
        best_epoch = epoch
        best_test_loss = test_loss
        best_model = copy.deepcopy(flow.state_dict())
    
    #print loss every n epoches
    if(epoch)%10==0:
        # print(f"Epoch {epoch+1:02d}/{epochs}, NLL: {train_loss:.4f}")
        print(f'{epoch}: Train loss: {train_loss:1.4f}')
        print(f'{epoch}: Test loss: {test_loss:1.4f}')

print(f'best_epoch: {best_epoch} | best_test_loss: {best_test_loss}')
plt.plot(loss_dict['train'], label='Train')
plt.plot(loss_dict['test'], label='Validation')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss curves')
# plt.show()

# xxx

# =========================================================
# 5. Evaluate the learned density p(x,y)
# =========================================================

flow.load_state_dict(best_model)
flow.eval()
def sample_qy(y_query,n_samples=4): #1024
    """
    Sample from conditional flow q_theta(x|y)

    Args:
        n_samples: int, number of samples to draw per y_query
    """
    y = torch.tensor(y_query,dtype=torch.float32).to(device)
    y_con = y.unsqueeze(0).repeat(n_samples,1) #n_samples
    with torch.no_grad():
        # flow.eval()
        x_samples = flow.sample(
            num_samples=1, 
            context=y_con) 
    x_ = x_samples.squeeze(1).cpu().numpy()   
    # print('x_: ', x_)
    # return x_ #normalised_samples
    return x_

action = actions[10]

print(action)
print('aciton: ', action.shape)


states_given_action = sample_qy(action)
print('states_given_action: ', states_given_action.shape)
print('states_given_action: ', states_given_action)

# print("scaler.n_features_in_ =", scaler_state.n_features_in_)
# print("states_given_action.shape =", states_given_action.shape)

# xxx

states_given_action2 = scaler_state.inverse_transform(states_given_action)
print('states_given_action: ', states_given_action2.shape)
print('states_given_action: ', states_given_action2)
# print('Sample mean:')
# print('E[S|a] =', states_given_action.mean(axis=1))

# ============================================
# generation.py -> Normalising Flows Script
# ============================================
#conditional distribution Q(States | Actions)
def conditional_dist(self,states,actions):
    """
    Outputs probability distribution of states given
    action i.e. Q(S|A)

    Args:
        states: array, state samples from policy
        action: array, action samples from policy 
    """
    dim_states = states.shape[1]
    dim_actions = actions.shape[1]
    num_samples = states.shape[0]
    # print('dim: ',dim_states) #shape (N,a+b)
    # print('dim: ',dim_actions) #shape (N,a+b)

    # =========================================================
    # 2. Prepare training 
    # =========================================================
    # #normalisation
    # self.scaler_state = StandardScaler() # x' = (x - u)/s [u: mean, s: std]
    # self.scaler_action = StandardScaler() # x' = (x - u)/s [u: mean, s: std]        
    # t_states = torch.tensor(
    #     self.scaler_state.fit_transform(states),
    #                         dtype=torch.float32)
    # t_actions = torch.tensor(
    #     self.scaler_action.fit_transform(actions),
    #                          dtype=torch.float32)

    #convert to torch        
    t_states = torch.tensor(states,dtype=torch.float32)
    t_actions = torch.tensor(actions,dtype=torch.float32)
    # print('t_states: ',t_states.shape)
    # print('t_actions: ',t_actions.shape)

    #prepare 80%/20% for training/testing
    num_data = int(num_samples*.8) #80%
    # print('num_data: ', num_data)

    #data splitting
    train_states = t_states[:num_data]
    train_actions = t_actions[:num_data]

    test_states = t_states[num_data:]
    test_actions = t_actions[num_data:]
    # print('train_states: ',train_states.shape)
    # print('test_states: ',test_states.shape)
    # print('train_actions: ',train_actions.shape)
    # print('test_actions: ',test_actions.shape)

    #Split loader
    train_loader =  DataLoader( 
                        TensorDataset(
                            train_states,
                            train_actions),
                        batch_size=256,#512
                        shuffle=True
                        )

    test_loader =  DataLoader( 
                        TensorDataset(
                            test_states,
                            test_actions),
                        batch_size=256,#512
                        shuffle=True
                        )

    # =========================================================
    # 3. Define Masked Autoregressive Flow (MAF)
    # =========================================================
    def create_maf_flow(dim_states, dim_actions, num_flows=5, hidden_features=128, num_blocks=2):
        torch.manual_seed(0)
        """Construct a Masked Autoregressive Flow with optional normalization and permutation."""
        transforms = []
        for _ in range(num_flows):
            transforms.append(
                MaskedAffineAutoregressiveTransform(
                    features=dim_states,
                    hidden_features=hidden_features,
                    context_features=dim_actions,
                    num_blocks=num_blocks,
                    # activation=nn.Tanh() #bounded activation
                    activation=nn.ReLU()
                    # activation=nn.ELU()
                )
            )
            # Normalization + permutation improve expressiveness
            transforms.append(BatchNorm(features=dim_states)) #normalize activations
            transforms.append(ReversePermutation(features=dim_states))
        transform = CompositeTransform(transforms)
        base_dist = StandardNormal([dim_states])
        return Flow(transform, base_dist)

    # Create flow model
    flow = create_maf_flow(dim_states=dim_states, 
                        dim_actions=dim_actions,
                        num_flows=3, #5, 
                        hidden_features=32, #64, #128, 
                        num_blocks=1#2
                        )
    flow.to(device) #flow = prob_states_given_action: Q(S|a) 

    # =========================================================
    # 4. Train the flow
    # =========================================================
    optimizer = torch.optim.Adam(flow.parameters(), 
                                lr=1e-5, #1e-4
                                # weight_decay=1e-6
                                )
    epochs = 50
    best_test_loss = np.inf
    best_epoch = 0
    best_model = deepcopy(flow.state_dict())

    loss_dict = dict(train=[], test=[])
    for epoch in range(epochs):

        #train
        train_loss = 0.0
        flow.train()
        for states_b,actions_b in train_loader:
            states_b,actions_b  = states_b.to(device),actions_b.to(device) 
            loss = -flow.log_prob(inputs=states_b,
                                context=actions_b
                                ).mean()  # negative log-likelihood
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            torch.nn.utils.clip_grad_norm_(flow.parameters(), max_norm=1.0) # Clip gradients to a max norm of 1.0

            train_loss += loss.item() #* states_b.size(0) #batch_size
            train_loss /= len(train_loader) #train_states.shape[0] #length_total_training_dataset
        loss_dict['train'].append(train_loss)

        #validate
        test_loss = 0.0
        with torch.no_grad():
            flow.eval()
            for states_b,actions_b in test_loader:
                states_b,actions_b  = states_b.to(device),actions_b.to(device) 
                loss_1 = -flow.log_prob(inputs=states_b,
                                    context=actions_b
                                    ).mean() # negative log-likelihood
                test_loss += loss_1.item() #* states_b.size(0) #batch_size
                test_loss /= len(test_loader) #test_states.shape[0] #length_total_training_dataset
            loss_dict['test'].append(test_loss)

        #save best model for use at the end
        if test_loss < best_test_loss:
            best_epoch = epoch
            best_test_loss = test_loss
            best_model = deepcopy(flow.state_dict())

    print('test_loss: ', test_loss) 
    # """    
    #visualization of loss        
    #print loss every n epoches
        # if(epoch)%10==0:
        #     # print(f"Epoch {epoch+1:02d}/{epochs}, NLL: {train_loss:.4f}")
        #     print(f'{epoch}: Train loss: {train_loss:1.4f}')
        #     print(f'{epoch}: Test loss: {test_loss:1.4f}')

    print(f'best_epoch: {best_epoch} | best_test_loss: {best_test_loss}')
    plt.plot(loss_dict['train'], label='Train')
    plt.plot(loss_dict['test'], label='Validation')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss curves')
    plt.show()
    # """

    # =========================================================
    # 5. Load best estimated density p(X|Y)
    # flow is prob_states_given_action: Q(S|a)  
    # =========================================================
    flow.load_state_dict(best_model)     
    return flow # Q(X|Y) - conditional distribution

#samples from conditional distribution
def sample_qy(self,y_query,flow,n_samples=5000): #1024
    """
    Sample from conditional flow q_theta(x|y)

    Args:
        n_samples: int, number of samples to draw per y_query
    """
    flow.eval()
    y = torch.tensor(y_query,dtype=torch.float32).to(device)
    y_con = y.unsqueeze(0).repeat(n_samples,1) #n_samples
    with torch.no_grad():
        x_samples = flow.sample(
            num_samples=1, 
            context=y_con) 
    x_ = x_samples.squeeze(1).cpu().numpy()   
    # print('x_: ', x_.shape)
    # print('min/max: ', x_.min(), x_.max())
    # print('mean/std: ', x_.mean(), x_.std())
    # print('=================================')
    # xxx
    # x
    return x_ #np.array(x_,dtype='float64')
