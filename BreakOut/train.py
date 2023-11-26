from agent import Agent
from env import BreakoutEnvWrapper
from replay_memory import ReplayMemory
from utils import get_epsilon, update_state_stack, log_clear_video_directory
from checkpoint import save_checkpoint, load_checkpoint
import config

import matplotlib.pyplot as plt
from IPython import display
import numpy as np
import wandb
import os
import gymnasium as gym
import pygame

# Initialize wandb for logging metrics
wandb.init(project="breakout-dqn1", entity="ai42")

# Initialize pygame for rendering the Breakout game (this isn't helping currently)
pygame.init()

# Define a video directory for logging videos of the agent playing Breakout
video_dir = 'C:\Users\neild\OneDrive\Documentos\ARTIFICIAL INTELLIGENCE (UAB)\3º\1st semester\ML Paradigms\Learn2Earn_RL-1\BreakOut' # Set this to your preferred directory
os.makedirs(video_dir, exist_ok=True) # Make the video directory if it doesn't exist

def train():
    '''
    Description:
            Train the agent in the Breakout environment using the DQN algorithm.
            This is done by interacting with the environment, collecting experiences,
            and training the agent using the collected experiences.
    '''
    # Initialize the environment and the agent
    env = BreakoutEnvWrapper() # Create the Breakout environment
    env = gym.wrappers.RecordVideo(env, video_folder=video_dir, episode_trigger=lambda episode_id: True) # Record videos of the agent playing Breakout
    agent = Agent(state_space=(4, 84, 84), 
                  action_space=env.action_space, 
                  replay_memory_capacity=config.replay_memory_size, 
                  batch_size=config.batch_size,
                  gamma=config.gamma, 
                  lr=config.lr, 
                  target_update=config.target_update, 
                  device=config.device)

    # Initialize replay memory with a capacity of replay_memory_size
    replay_memory = ReplayMemory(config.replay_memory_size)

    # Begin training loop over episodes
    for episode in range(config.num_episodes): 
        # Initialize the state by resetting the environment
        state = env.reset() 
        
        # Initialize the stack of states by stacking the same state 4 times, once the state is
        # updated, the stack will be updated too, so it wont be the same state 4 times anymore
        state = np.stack([state] * 4, axis=0) # shape: (4, 84, 84)

        # Begin episode loop over steps
        for step in range(config.max_steps): 
            # Select an action using the agent's policy
            action = agent.select_action(state, get_epsilon(episode)) # Epsilon-greedy policy, where epsilon is decayed over episodes
            
            # Take the selected action in the environment and get the next state, reward, and other info
            next_state, reward, terminated, truncated, info = env.step(action) 
            
            # Update the stack of states by stacking the next state on top of the stack of states
            next_state_stack = update_state_stack(state, next_state) # shape: (4, 84, 84)
            
            # Push a new experience to the replay memory, overwriting the oldest experience if the memory is full
            replay_memory.push(state, action, reward, next_state_stack, terminated, truncated)

            # Update the state to the next stack of states
            state = next_state_stack

            # Train the agent using a random batch of experiences from the replay memory 
            agent.learn()

            # Check if the episode is terminated or truncated, if so, break the episode loop
            if terminated or truncated:
                break
        
        # Update the target network every target_update episodes 
        if episode % config.target_update == 0:
            agent.update_target_network()

        # Save checkpoints every few episodes or at the end of training
        if episode % config.save_checkpoint_every == 0 or episode == config.num_episodes - 1:
            save_checkpoint(agent, replay_memory, 'checkpoint.pth')

        # Render the Breakout game (partida) on screen (currently failing for some reason)
        if config.render: #For now this is set to False because it is failing to render the game on screen
           log_clear_video_directory(video_dir) # Clear the video directory and log the most recent video file to wandb
            
        # Get the total reward of all the episodes in the replay memory
        main_reward = np.sum([experience.reward for experience in replay_memory.memory])  
        
        # Log the episode number, the number of steps in the episode, and the total reward of the episode
        wandb.log({"Episode": episode, "Steps": step, "Main Reward": main_reward})

        
    env.close() # Close the environment

if __name__ == "__main__":
    train()
