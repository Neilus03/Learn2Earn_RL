import wandb
from AC_model import AC_Policy, select_action, finish_episode
import gymnasium as gym
import argparse
import torch
from itertools import count
from collections import namedtuple
import torch.optim as optim
import numpy as np

def main():

    running_reward = 10

    # run infinitely many episodes
    for i_episode in count(1):

        # reset environment and episode reward
        state, _ = env.reset()
        ep_reward = 0

        # for each episode, only run 9999 steps so that we don't
        # infinite loop while learning
        for t in range(1, 10000):

            # select action from policy
            action = select_action(state, model)

            # take the action
            state, reward, done, _, _ = env.step(action)

            if args.render:
                env.render()

            model.rewards.append(reward)
            ep_reward += reward
            if done:
                break

        # update cumulative reward
        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward

        # perform backprop
        finish_episode(model, optimizer, args.gamma)

        # Log metrics to wandb
        if args.wandb:
            wandb.log({'Episode': i_episode, 'Last Reward': ep_reward, 'Average Reward': running_reward})

        # log results
        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                  i_episode, ep_reward, running_reward))

        # Save model every args.save_interval episodes
        if i_episode % args.save_interval == 0:
            torch.save(model.state_dict(), f'{args.env}_actor_critic_model_{i_episode}.pth')

        # check if we have "solved" the cart pole problem
        if running_reward > env.spec.reward_threshold:
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, t))
            
            # save the model if wandb flag is set to true
            if args.wandb:
                torch.save(model.state_dict(), 'solved_'+args.env+'_actor_critic_model.pth')
            break


if __name__ == '__main__':

    # The following arguments are to set the environment, gamma parameter,
    # seed parameter, render (boole), log interval of status, wandb (bool)
    # save interval to save the model

    parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
    
    parser.add_argument('--env', type=str, default='CartPole-v1', choices=['CartPole-v1', 'MountainCar-v0', 'Acrobot-v1'],
                        help='Environment to use (default: CartPole-v1)') # usage: python train.py --env CartPole-v1

    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor (default: 0.99)') # usage: python train.py --gamma 0.99
    
    parser.add_argument('--lr', type=float, default=3e-2, metavar='G',
                        help='learning rate (default: 3e-2)') # usage: python train.py --lr 2e-3

    parser.add_argument('--seed', type=int, default=543, metavar='N',
                        help='random seed (default: 543)') # usage: python train.py --seed 123

    parser.add_argument('--render', action='store_true',
                        help='render the environment') # usage: python train.py --render

    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='interval between training status logs (default: 10)') # usage: python train.py --log-interval 10

    parser.add_argument('--wandb', action='store_true', help='enable wandb logging') # usage: python train.py --wandb

    parser.add_argument('--save-interval', type=int, default=100, metavar='N',
                        help='interval between saving model (default: 100)') # usage: python train.py 

    args = parser.parse_args()

    SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])


    env = gym.make(args.env, render_mode="human") if args.render else gym.make(args.env)
    env.reset(seed=args.seed)
    torch.manual_seed(args.seed)

    # if wandb flag is set then 
    if args.wandb:
        wandb.init(project='AC_agent', entity='ai42', name='AC_agent1'+args.env)

    # Model initialization
    model = AC_Policy(input_dim=env.observation_space.shape[0], hidden_dim=128, output_dim=env.action_space.n)

    # Optimizer initialization with Adam 
    optimizer = optim.Adam(model.parameters(), args.lr)

    main()
