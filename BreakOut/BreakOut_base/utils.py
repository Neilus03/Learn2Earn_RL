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
    return updated_stack # Returns a new array with the new state at the end with shape (4, 84, 84)

'''
# Tests to populate the replay memory
if __name__ == "__main__":
    # Example usage of the ReplayMemory class:
    from env import BreakoutEnvWrapper
    from replay_memory import ReplayMemory
    env = BreakoutEnvWrapper()
    replay_memory = ReplayMemory(capacity=10000)
    replay_memory.prepopulate_replay_memory(env=env, replay_memory=replay_memory, prepopulate_steps=100, action_space=env.action_space)
    env.close()
    print("Memory size:", len(replay_memory.memory))
    print("Push count:", replay_memory.push_count)
    print("Memory:", replay_memory.memory)
'''