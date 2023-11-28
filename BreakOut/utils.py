import numpy as np
import math
import config
import os
import wandb

def get_epsilon(episode):
    # Define an epsilon decay over episodes
    epsilon = config.epsilon_final + (config.epsilon_start - config.epsilon_final) * \
              math.exp(-1. * episode / config.epsilon_decay)
    return epsilon

def update_state_stack(state_stack, new_state):
    # Update and maintain the stack of states
    updated_stack = np.concatenate([state_stack[1:], np.expand_dims(new_state, 0)], axis=0)
    return updated_stack
