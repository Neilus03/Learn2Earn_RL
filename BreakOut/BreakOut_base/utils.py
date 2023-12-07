import numpy as np
import math
import config
import random
import os
import wandb

def get_epsilon(episode):
    '''
    Description:
        Get the epsilon value for the current episode.
        This is done by using an epsilon decay over episodes.
        The epsilon decay is defined in the config.py file.
    
    Parameters:
        episode (int): The current episode.
    
    Returns:
        epsilon (float): The epsilon value for the current episode.
    '''
    # Define an epsilon decay over episodes
    epsilon = config.epsilon_final + (config.epsilon_start - config.epsilon_final) * \
              math.exp(-1. * episode / config.epsilon_decay)
    return epsilon

def update_state_stack(state_stack, new_state):
    '''
    Description:
        Update the stack of states by removing the oldest state and adding the new state at the end.
        This is done by concatenating the state_stack without the oldest state and the new state.
        This is useful because we need to stack the states to feed them to the DQN, and we need to update the stack
        so that the DQN can learn from the most recent states.
    '''
    # Update and maintain the stack of states by removing the oldest state and adding the new state at the end
    updated_stack = np.concatenate([state_stack[1:], np.expand_dims(new_state, 0)], axis=0) 
    return updated_stack

def prepopulate_replay_memory(env, replay_memory, prepopulate_steps, action_space):
    '''
    Description:
        Prepopulate the replay memory with random experiences. This is done to avoid the problem of
        the agent learning from a small number of experiences at the beginning of the training.
        If the agent learns from a small number of experiences at the beginning of the training,
        it will not learn well because it will not have enough experiences to learn from.
        
    Parameters:
        env (gym environment): The environment.
        replay_memory (ReplayMemory object): The replay memory.
        prepopulate_steps (int): The number of experiences to prepopulate the replay memory.
        action_space (int): The number of possible actions.
        
    Returns:
        None, it populates the replay memory with random experiences in place.
    '''
    # Reset the environment and stack the initial state 4 times
    state = env.reset()
    state = np.stack([state] * 4, axis=0)  # Initial state stack
    
    # Prepopulate the replay memory with random experiences
    for _ in range(prepopulate_steps):
        
        # Select a random action
        action = random.randrange(action_space)
        # Perform the action and observe the next state, reward, terminated and truncated
        next_state, reward, terminated, truncated, info = env.step(action)
        
        # Stack the next state and update the state stack
        next_state_stack = update_state_stack(state, next_state)
        
        # Push the experience to the replay memory for populating it
        replay_memory.push(state, action, reward, next_state_stack, terminated, truncated)
        
        # Update the state for the next iteration
        state = next_state_stack
            
    print(f"Prepopulated with {len(replay_memory.memory)} experiences.")
    
