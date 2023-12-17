import argparse
import gymnasium as gym
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from AC_model import AC_Policy

def main():
    parser = argparse.ArgumentParser(description='Test the Actor-Critic model')
    parser.add_argument('--model-path', type=str, required=True, help='Path to the trained model')
    parser.add_argument('--env', type=str, default='CartPole-v1', choices=['CartPole-v1', 'MountainCar-v0', 'Acrobot-v1'],
                        help='Environment to use (default: CartPole-v1)') # usage: python train.py --env CartPole-v1
    args = parser.parse_args()
    
    # Create the environment and wrap it to record a video
    env = gym.make(args.env, render_mode="human")
    
    # Load the trained model
    
    model = AC_Policy(env.observation_space.shape[0], 128, env.action_space.n)
    model.load_state_dict(torch.load(args.model_path))
    model.eval()

    state, _info = env.reset()
    done = False
    while not done:
        state = np.array(state)
        state = torch.from_numpy(state).float()
        action_probs, _ = model(state)
        action = torch.argmax(action_probs).item()
        state, _reward, terminated, truncated, _info = env.step(action)
        done = terminated or truncated
        env.render()

    env.close()

if __name__ == '__main__':
    main()