from agent import Agent
from env import BreakoutEnvWrapper
from replay_memory import ReplayMemory
from utils import get_epsilon, update_state_stack, log_clear_video_directory
from checkpoint import save_checkpoint, load_checkpoint
import config

import matplotlib.pyplot as plt
import numpy as np
import wandb
import os
import gymnasium as gym
# Initialize wandb for logging metrics
wandb.init(project="breakout-dqn1", entity="ai42")

# Define a video directory
video_dir = '/home/ndelafuente/Desktop/Learn2Earn_RL/LunarLander/Breakout/Videos1' # Set this to your preferred directory
os.makedirs(video_dir, exist_ok=True)

# Wrap your environment to record videos


def train():
    '''
    Description:
            Train the agent in the Breakout environment using the DQN algorithm.
            This is done by interacting with the environment, collecting experiences,
            and training the agent using the collected experiences.
    '''
    # Initialize the environment and the agent
    env = BreakoutEnvWrapper()
    env = gym.wrappers.RecordVideo(env, video_folder=video_dir)

    
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
            next_state_stack = update_state_stack(state, next_state)

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
        if config.render:
            env.render()
        env.render()
        log_clear_video_directory(video_dir) # Clear the video directory and log the most recent video file to wandb
          
        # Get the total reward of the episode
        reward = np.sum([experience.reward for experience in replay_memory.memory])  
        
        # Log the episode number, the number of steps in the episode, and the total reward of the episode
        wandb.log({"Episode": episode, "Steps": step, "Total Reward": reward})
        plt.figure(figsize=(10, 7))
        plt.bar(episode, reward)
        plt.title("Total Reward per Episode")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.savefig("reward.png")
        wandb.Image("reward.png")
    env.close() # Close the environment

if __name__ == "__main__":
    train()
