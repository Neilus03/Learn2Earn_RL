import torch

#HYPERPARAMETERS
lr = 0.0001
gamma = 0.99
batch_size = 32
replay_memory_size = 10000
target_update = 10
epsilon_start = 1.0
epsilon_final = 0.01
epsilon_decay = 30000
num_episodes = 1000
max_steps = 1000
save_checkpoint_every = 10
render = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")