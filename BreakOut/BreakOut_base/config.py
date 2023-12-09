import torch

#HYPERPARAMETERS
lr = 3e-4
gamma = 0.99
batch_size = 64
replay_memory_size = 10000
target_update = 10 # Update the target network every 10 episodes 
epsilon_start = 1
epsilon_final = 0.01
epsilon_decay = 1000 # Decay the epsilon value every 10 episodes by using an exponential decay
num_episodes = 10000
max_steps = 10000

#FLAGS
pretrained = False # Set to True to load a checkpoint
log_to_wandb = True # Set to True to log metrics to wandb
free_checkpoint_memory = False # Set to True to remove previous checkpoints
render = False # Set to True to render the environment
negative_reward = False # Set to True to use negative rewards

#MODELS
attention_model = False # Set to True to use the attention model, False to use the regular model

#DEVICE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#CHECKPOINTS
checkpoint_dir = '/home/ndelafuente/Downloads/Learn2Earn_RL/BreakOut/BreakOut_base/checkpoints'
