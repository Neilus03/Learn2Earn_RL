from stable_baselines3 import PPO
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

wandb.init(project="breakout-PPO-test", entity= "ai42", sync_tensorboard=True)

"""
Test the model
"""

#Unzip the file a2c_Breakout_1M.zip and store the unzipped files in the folder a2c_Breakout_unzipped
unzip_file('PPO_Breakout_1M.zip', 'PPO_Breakout_1M_unzipped') 
           
#We start with a single environment for Breakout with render mode set to human
env = make_env("BreakoutNoFrameskip-v4") 
#We then wrap the environment with the DummyVecEnv wrapper which converts the environment to a single vectorized environment
env = DummyVecEnv([env]) # Output shape: (1, 84, 84)
#Finally, we wrap the environment with the VecFrameStack wrapper which stacks the observations over the last 4 frames
env = VecFrameStack(env, n_stack=4) # Output shape: (4, 84, 84)



# Create the model
model = PPO("CnnPolicy", env, verbose=1)

# Load the model components, including the policy network and the value network
model.policy.load_state_dict(torch.load("/home/ndelafuente/Learn2Earn_RL/BreakOut/BreakOut_sb3_PPO/PPO_Breakout_1M_unzipped/policy.pth"))
model.policy.optimizer.load_state_dict(torch.load("/home/ndelafuente/Learn2Earn_RL/BreakOut/BreakOut_sb3_PPO/PPO_Breakout_1M_unzipped/policy.optimizer.pth"))

# Visualize the gameplay
num_episodes = 100

# Run the episodes and render the gameplay
for episode in range(num_episodes):
    # Reset the environment and stack the initial state 4 times
    obs = env.reset()#obs = np.stack([obs] * 4, axis=0)  # Initial state stack
    done = False
    episode_reward = 0
    while not done:
        # Take an action in the environment according to the policy of the trained agent
        action, _ = model.predict(obs)
        # Take the action in the environment and store the results in the variables
        obs, reward, done, info = env.step(action) # obs shape: (1, 84, 84), reward shape: (1,), done shape: (1,), info shape: (1,) 
        # Update the total reward
        episode_reward += reward[0]
        # Render the environment to visualize the gameplay of the trained agent
        env.render()
    # Log the total reward of the episode to wandb
    wandb.log({'test_episode_reward': episode_reward, 'test_episode': episode})

# Close the environment and finish the logging
env.close()
wandb.finish()
