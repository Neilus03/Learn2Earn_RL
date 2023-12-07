from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from stable_baselines3.common.atari_wrappers import AtariWrapper
import gymnasium as gym
import torch
import config
import wandb
from wandb.integration.sb3 import WandbCallback
from utils import make_env, unzip_file, CustomWandbCallback, RewardLogger
import os
from stable_baselines3.common.utils import get_latest_run_id
import tensorboard as tb



log_dir = os.path.join('/home/ndelafuente/Learn2Earn_RL/BreakOut/BreakOut_sb3_A2C/log_dir', 'a2c_breakout')
os.makedirs(log_dir, exist_ok=True)

#check_freq is the frequency at which the callback is called, in this case, the callback is called every 1000 timesteps
check_freq = 2000
#save_path is the path where the best model will be saved
save_path = '/home/ndelafuente/Learn2Earn_RL/BreakOut/BreakOut_sb3_A2C/a2c_Breakout_1M_save_path'
#Create the folder where the best model will be saved if it does not exist
os.makedirs(save_path, exist_ok=True)
#Create the callback that logs the mean reward of the last 100 episodes to wandb
custom_callback = CustomWandbCallback(check_freq, save_path)



'''
Set up loging to wandb
'''
#Set wandb to log the training process
wandb.init(project="breakout-a2c", entity= "ai42", sync_tensorboard=True)
#wandb_callback is a callback that logs the training process to wandb, this is done because wandb.watch() does not work with sb3
wandb_callback = WandbCallback()

'''
Set up the environment
'''
# Create multiple environments and wrap them correctly
env = make_atari_env("BreakoutNoFrameskip-v4", n_envs=4, seed=0)
env = VecFrameStack(env, n_stack=4)

'''
Set up the model
'''
model = A2C("CnnPolicy", env, verbose=1, tensorboard_log=log_dir)
if config.pretrained:
    model = A2C.load("/home/ndelafuente/Learn2Earn_RL/BreakOut/BreakOut_sb3_A2C/a2c_Breakout_1M.zip", env=env, verbose=1, tensorboard_log=log_dir)

'''
Train the model and save it
'''
#model.learn will train the model for 1e6 timesteps, timestep is the number of actions taken by the agent, 
# in a game like breakout, the agent takes an action every frame, then the number of timesteps is the number of frames,
# which is the number of frames in 1 game multiplied by the number of games played.
#The average number of frames in 1 game is 1000, so 1e6 timesteps is 1000 games more or less.
#log_interval is the number of timesteps between each log, in this case, the training process will be logged every 100 timesteps.
#callback is a callback that logs the training process to wandb, this is done because wandb.watch() does not work with sb3
model.learn(total_timesteps=1e6, log_interval=100, callback=[wandb_callback, custom_callback])
#Save the model
model.save("a2c_Breakout_1M")

# Close the environment and finish the logging
env.close()
wandb.finish()
