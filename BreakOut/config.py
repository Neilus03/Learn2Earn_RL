import torch

#HYPERPARAMETERS
lr = 0.0003
gamma = 0.99
batch_size = 64
replay_memory_size = 10000
target_update = 10
epsilon_start = 1.0
epsilon_final = 0.01
epsilon_decay = 30000
num_episodes = 1000000
max_steps = 10000000

#FLAGS
pretrained = True
log_to_wandb = False
free_checkpoint_memory = False

#DEVICE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#CHECKPOINTS
checkpoint_dir = '/home/ndelafuente/Learn2Earn_RL/BreakOut/checkpoints'
