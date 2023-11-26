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

# Function to clear the video directory
def log_clear_video_directory(video_dir):
    '''
    This function clears the video directory of all video files
    and logs the most recent video file to wandb.
    '''  
    # Get list of video files in video directory 
    video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
    
    # After each episode, log the video file to wandb
    if video_files:
        last_video_file = video_files[-1]  # Assuming the last file in the list is the most recent
        wandb.log({"episode_video": wandb.Video(os.path.join(video_dir, last_video_file), fps=4, format="mp4")})
    
    # Delete all video files in video directory
    for file in video_files:
        os.remove(os.path.join(video_dir, file))