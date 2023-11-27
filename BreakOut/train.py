from agent import Agent
from env import BreakoutEnvWrapper
from replay_memory import ReplayMemory
from utils import get_epsilon, update_state_stack
from checkpoint import save_checkpoint, load_last_checkpoint, remove_previous_checkpoints
import config

import numpy as np
import wandb
import gymnasium as gym
import pygame

# Initialize wandb for logging metrics
if config.log_to_wandb:
    wandb.init(project="breakout-dqn1", entity="ai42")

# Initialize pygame for rendering the Breakout game (this isn't helping currently)
pygame.init()

# Define a video directory for logging videos of the agent playing Breakout
#video_dir = r'C:\Users\neild\OneDrive\Escritorio\Breakout_videos' # Set this to your preferred directory
#os.makedirs(video_dir, exist_ok=True) # Make the video directory if it doesn't exist

losses = [] # Initialize a list to store the losses of the training steps

def train():
    '''
    Description:
            Train the agent in the Breakout environment using the DQN algorithm.
            This is done by interacting with the environment, collecting experiences,
            and training the agent using the collected experiences.
    '''
    # Initialize the environment and the agent
    env = BreakoutEnvWrapper() # Create the Breakout environment
    #the line below seems to be not needed
    #env = gym.wrappers.RecordVideo(env, video_folder=video_dir, episode_trigger=lambda episode_id: True) # Record videos of the agent playing Breakout
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

    # Load a checkpoint if pretrained is True
    if config.pretrained:
        load_last_checkpoint(agent, replay_memory, config.checkpoint_dir)
        print("Checkpoint loaded successfully")

    best_reward = -np.inf # Initialize the best reward to -infinity 
    
    # Begin training loop over episodes
    for episode in range(config.num_episodes): 
        # Initialize the state by resetting the environment
        state = env.reset() 
        
        # Initialize the stack of states by stacking the same state 4 times, once the state is
        # updated, the stack will be updated too, so it wont be the same state 4 times anymore
        state = np.stack([state] * 4, axis=0) # shape: (4, 84, 84)
        
        # Initialize the episode reward to 0
        episode_reward = 0 
        
        # Initialize the number of lives to 5, because the agent starts with 5 lives
        # each time the agent loses a life, the reward of the current step will be set to -1
        current_lives = 5 
        
        # Begin episode loop over steps
        for step in range(config.max_steps): 
            # Select an action using the agent's policy
            action = agent.select_action(state, get_epsilon(episode)) # Epsilon-greedy policy, where epsilon is decayed over episodes
            
            # Take the selected action in the environment and get the next state, reward, and other info
            next_state, reward, terminated, truncated, info = env.step(action) 
            
            # Check if the agent lost a life, if so, set the reward to -1
            if info['lives'] < current_lives: # Check if the agent lost a life
                reward = -1 # If the agent lost a life, set the reward to -1
                current_lives = info['lives']
            
            # Update the episode reward by adding the reward of the current step
            episode_reward += reward
            
            # Update the stack of states by stacking the next state on top of the stack of states
            next_state_stack = update_state_stack(state, next_state) # shape: (4, 84, 84)
            
            # Push a new experience to the replay memory, overwriting the oldest experience if the memory is full
            replay_memory.push(state, action, reward, next_state_stack, terminated, truncated)

            # Update the state to the next stack of states
            state = next_state_stack

            # Train the agent using a random batch of experiences from the replay memory and save the loss of the training step
            loss = agent.learn() # This is done by using the agent's learn() method, which returns the loss of the training step while training the agent

            # Append the loss of the training step to the losses list
            losses.append(loss)
            
            # Check if the episode is terminated or truncated, if so, break the episode loop
            if terminated or truncated:
                break
        
        # Update the target network every target_update episodes 
        if episode % config.target_update == 0:
            agent.update_target_network()

        # Save checkpoints if the episode reward is better than the best reward so far
        if episode_reward > best_reward:
            save_checkpoint(agent, replay_memory, config.checkpoint_dir, f'checkpoint{episode}.pth')
            print(f"Checkpoint saved at episode {episode} with reward {episode_reward}")
            remove_previous_checkpoints(config.checkpoint_dir)
            best_reward = episode_reward
        
        # Render the Breakout game (partida) on screen (currently failing for some reason)
        
        #SEEMS TO WORK WITHOUT THE LINE BELOW
        #log_clear_video_directory(video_dir) # Clear the video directory and log the most recent video file to wandb
            
        # Get the total acumulated reward of all the episodes in the replay memory and log it to wandb
        total_reward = np.sum([experience.reward for experience in replay_memory.memory]) 
        ''' 
            remember that the replay memory is a list of experiences, this
            experiences are objects of the class Experience defined in replay_memory.py 
            and have the attributes state, action, reward, next_state, terminated, and truncated
            so we can get the reward of each experience by using the attribute reward
            and then we can sum all the rewards of all the experiences in the replay memory
            to get the total reward of all the episodes in the replay memory, the capacity of the
            replay memory is replay_memory_size, so the total reward of all the episodes in the
            replay memory is the total reward of the last replay_memory_size episodes
        '''
    
        # Log the episode number, the number of steps in the episode, the total reward of the episode and the loss of the training step to wandb
        if config.log_to_wandb:
            wandb.log({"Episode": episode, "Steps": step, "Episode Reward": episode_reward, "Main Reward": total_reward, "Losses": losses[-1]})

        
    env.close() # Close the environment
    if config.log_to_wandb:
        wandb.finish() # Finish the wandb run
if __name__ == "__main__":
    train()
