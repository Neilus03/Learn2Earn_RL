import argparse
import gymnasium as gym
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
#from gymnasium.wrappers import Monitor
from AC_model import AC_Policy

def main():
    parser = argparse.ArgumentParser(description='Test the Actor-Critic model')
    parser.add_argument('--model-path', type=str, required=True, help='Path to the trained model')
    args = parser.parse_args()

    # Load the trained model
    model = AC_Policy(4, 128, 2)
    model.load_state_dict(torch.load(args.model_path))
    model.eval()

    # Create the environment and wrap it to record a video
    env = gym.make('CartPole-v1', render_mode="human")
    #env = Monitor(env, './video', force=True)

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