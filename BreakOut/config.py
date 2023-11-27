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
num_episodes = 10000
max_steps = 10000

#FLAGS
pretrained = False
log_to_wandb = False

#DEVICE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#CHECKPOINTS
checkpoint_dir = r'C:\Users\neild\OneDrive\Documentos\ARTIFICIAL INTELLIGENCE (UAB)\3º\1st semester\ML Paradigms\Learn2Earn_RL-1\BreakOut\checkpoints'
