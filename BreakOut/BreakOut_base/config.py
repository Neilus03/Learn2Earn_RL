import torch

#HYPERPARAMETERS
lr = 0.003
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
pretrained = False # Set to True to load a checkpoint
log_to_wandb = True # Set to True to log metrics to wandb
free_checkpoint_memory = False # Set to True to remove previous checkpoints
render = False # Set to True to render the environment
negative_reward = False # Set to True to use negative rewards

#MODELS
attention_model = True # Set to True to use the attention model, False to use the regular model

#DEVICE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#CHECKPOINTS
checkpoint_dir = '/home/ndelafuente/Learn2Earn_RL/BreakOut/checkpoints'
