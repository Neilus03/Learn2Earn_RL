import numpy as np
from itertools import count
from collections import namedtuple
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

# Tuples of actions with log probability and value of state
SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

class AC_Policy(nn.Module):
    """
    implements both actor and critic in one model
    """
    def __init__(self, input_dim=4, hidden_dim=128, output_dim=2):
        super(AC_Policy, self).__init__()
        self.affine1 = nn.Linear(input_dim, hidden_dim) # input_dim is the number of inputs from the observation space

        # actor's layer
        self.action_head = nn.Linear(hidden_dim, output_dim) # output_dim is the number of actions from the action space

        # critic's layer
        self.value_head = nn.Linear(hidden_dim, 1) # 1 is the value of the action

        # action & reward buffer
        self.saved_actions = []
        self.rewards = []


    def forward(self, x):
        """
        forward of both actor and critic
        """
        x = F.relu(self.affine1(x))

        # actor: choses action to take from state s_t
        # by returning probability of each action
        action_prob = F.softmax(self.action_head(x), dim=-1)

        # critic: evaluates being in the state s_t
        state_values = self.value_head(x)

        # return values for both actor and critic as a tuple of 2 values:
        # 1. a list with the probability of each action over the action space
        # 2. the value from state s_t
        return action_prob, state_values

def select_action(state, model):
    state = torch.from_numpy(state).float()
    probs, state_value = model(state)

    # create a categorical distribution over the list of probabilities of actions
    m = Categorical(probs)

    # and sample an action using the distribution
    action = m.sample()

    # save to action buffer
    model.saved_actions.append(SavedAction(m.log_prob(action), state_value))

    # the action to take (left or right)
    return action.item()


def finish_episode(model, optimizer, gamma=0.99):
    """
    Training code. Calculates actor and critic loss and performs backprop.
    """
    eps = np.finfo(np.float32).eps.item()
    R = 0
    saved_actions = model.saved_actions
    policy_losses = [] # list to save actor (policy) loss
    value_losses = [] # list to save critic (value) loss
    returns = [] # list to save the true values

    # calculate the true value using rewards returned from the environment
    for r in model.rewards[::-1]:
        # calculate the discounted value
        R = r + gamma * R
        returns.insert(0, R)

    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)

    for (log_prob, value), R in zip(saved_actions, returns):
        advantage = R - value.item()

        # calculate actor (policy) loss
        policy_losses.append(-log_prob * advantage)

        # calculate critic (value) loss using L1 smooth loss
        value_losses.append(F.smooth_l1_loss(value, torch.tensor([R])))

    # reset gradients
    optimizer.zero_grad()

    # sum up all the values of policy_losses and value_losses
    loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()

    # perform backprop
    loss.backward()
    optimizer.step()

    # reset rewards and action buffer 
    del model.rewards[:]
    del model.saved_actions[:]

