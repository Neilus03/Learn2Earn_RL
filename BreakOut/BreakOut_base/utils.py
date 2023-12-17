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
    # Define an epsilon decay over episodes, which is used to decay the epsilon value over episodes
    
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
    
    parameters:
        state_stack (numpy array): The stack of states with shape (4, 84, 84).
        new_state (numpy array): The new state with shape (84, 84).
    
    Returns:
        updated_stack (numpy array): The updated stack of states with shape (4, 84, 84).
    '''
    # Update and maintain the stack of states by removing the oldest state and adding the new state at the end
    updated_stack = np.concatenate([state_stack[1:], np.expand_dims(new_state, 0)], axis=0) 
    return updated_stack # Returns a new array with the new state at the end with shape (4, 84, 84)

