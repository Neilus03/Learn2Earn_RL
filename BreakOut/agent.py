import torch
import torch.optim as optim
import torch.nn.functional as F
import random
from DQN_model import DeepQNetwork
from replay_memory import ReplayMemory, Experience
from env import BreakoutEnvWrapper
import numpy as np


class Agent:
    def __init__(self, state_space, action_space, replay_memory_capacity, batch_size, gamma, lr, target_update, device):
        '''
        initialize the agent.
        '''
        self.state_space = state_space # shape of the input image (F, 84, 84) in Breakout, where F is the number of stacked frames
        self.action_space = action_space # The number of possible actions
        self.replay_memory = ReplayMemory(replay_memory_capacity) # Create the replay memory with the specified capacity
        self.batch_size = batch_size # The size of the batch of experiences sampled from the replay memory
        self.gamma = gamma # The discount factor
        self.lr = lr # The learning rate of the optimizer
        self.target_update = target_update # The number of episodes between updating the target network
        self.device = device # The device to use for training the DQN (cpu or gpu)
        self.policy_net = DeepQNetwork(state_space[1], state_space[2], action_space).to(device) # The policy network used to select actions
        self.target_net = DeepQNetwork(state_space[1], state_space[2], action_space).to(device) # The target network used to calculate the Q-values and stabilize the training
        self.target_net.load_state_dict(self.policy_net.state_dict()) # Initialize the weights of the target network to be the same as the policy network
        self.target_net.eval()  # Target net is not trained, so we set it to evaluation mode
        self.optimizer = optim.Adam(params=self.policy_net.parameters(), lr=lr) # The optimizer used to train the policy network
        '''
        Note: the target network is a copy of the policy network, that serves as a reference for the Q-values, 
        and it is updated every target_update episodes. This is done to stabilize the training of the DQN by
        preventing the target Q-values from changing too rapidly and causing oscillations or divergence.
        '''
        
    
    def select_action(self, state, epsilon):
        '''
        Description:
            Select an action using an epsilon-greedy policy.     
        Parameters:
            state (numpy.ndarray): The current state of the environment.
            epsilon (float): The probability of selecting a random action.
        Output:
            action (int): The selected action.
        '''
        if random.random() > epsilon: # Select the action with the highest Q-value	
            with torch.no_grad(): # Disable gradient calculation because we are not training the DQN, and we don't want to waste memory
                state = torch.tensor([state], device=self.device, dtype=torch.float32) # Convert the state to a torch tensor
                q_values = self.policy_net(state) # Get the Q-values from the policy network for the current state, which is the output of the network
                action = q_values.max(1)[1].view(1, 1).item() # Select the action with the highest Q-value.
                         #here we use the max() method to get the highest Q-value, and the item() method to get the value of the tensor as a python number
                         #the view() method is used to reshape the tensor to the shape of the action space, which is (1, 1) in this case
        else:
            # Select a random action
            action = random.randrange(self.action_space)

        return action # Return the selected action

    def learn(self):
        '''
        Description:
            Train the policy network using a batch of experiences sampled from the replay memory, 
            and update the target network if the push_count is a multiple of target_update.
        '''
        # Check if the replay memory can provide a batch of experiences for training based on the batch size
        if self.replay_memory.can_provide_sample(self.batch_size):
            
            # Sample a batch of experiences from the replay memory for training
            experiences = self.replay_memory.sample(self.batch_size)
            
            # Transpose the batch of experiences to get:
            batch = Experience(*zip(*experiences))

            states = torch.tensor(batch.state, device=self.device, dtype=torch.float32) # a batch of states
            actions = torch.tensor(batch.action, device=self.device, dtype=torch.int64) # a batch of actions
            rewards = torch.tensor(batch.reward, device=self.device, dtype=torch.float32) # a batch of rewards
            next_states = torch.tensor(batch.next_state, device=self.device, dtype=torch.float32) # a batch of next states
            terminateds = torch.tensor(batch.terminated, device=self.device, dtype=torch.bool) # a batch of terminated flags
            truncateds = torch.tensor(batch.truncated, device=self.device, dtype=torch.bool) # a batch of truncated flags
            dones = terminateds | truncateds # dones is True if the episode is terminated or truncated, otherwise it is False, this is done with a logical or (|)
            
            # Calculate the Q-values for the current states using the policy network
            current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(-1)) # gather() is used to select the Q-values of the actions taken in the current states
            
            # Calculate the Q-values for the next states using the target network
            next_q_values = self.target_net(next_states).max(1)[0].detach() # detach() is used to prevent the gradients from flowing into the target network
            
            '''
            Now it comes a key (but tricky) part of the DQN algorithm, which is calculating the expected Q-values.
            This is done by using the Bellman equation, which is:
            Q(s, a) = r + (gamma * max(Q(s', a')) * (1 - done))
            where:
                s is the current state
                a is the action taken in the current state
                r is the reward for taking the action in the current state
                s' is the next state
                a' is the action taken in the next state
                gamma is the discount factor
                done is True if the episode is terminated, otherwise it is False
                formula source: https://www.youtube.com/watch?v=0Ey02HT_1Ho and https://www.wikiwand.com/en/Q-learning
            '''
            #In this case, we adapt the formula to the case of a batch of experiences:
            expected_q_values = rewards + (self.gamma * next_q_values * (1 - dones))
            # where:
                # rewards is a batch of rewards
                # self.gamma is the discount factor
                # next_q_values is a batch of Q-values for the next states
                # dones is a batch of (terminated or truncated) flags

            # Calculate the loss between the current Q-values and the expected Q-values
            loss = F.mse_loss(current_q_values, expected_q_values.unsqueeze(1)) #I use the mean squared error loss function as in the paper "Playing Atari with Deep Reinforcement Learning"
            # set the optimizer gradients to zero before backpropagation
            self.optimizer.zero_grad()
            # backpropagation
            loss.backward()
            # update the weights of the policy network
            self.optimizer.step()


    def update_target_network(self):
        '''
        Description:
            Update the weights of the target network to be the same as the policy network.
            
        Note: the target network is a copy of the policy network, that serves as a reference for the Q-values,
        and it is updated every target_update episodes. This is done to stabilize the training of the DQN by
        preventing the target Q-values from changing too rapidly and causing oscillations or divergence.
        '''
        if self.replay_memory.push_count % self.target_update == 0: # We don't do this every episode because it is expensive
            self.target_net.load_state_dict(self.policy_net.state_dict()) # Update the weights of the target network to be the same as the policy network

# Example usage of the Agent class:
if __name__ == "__main__":
    # Initialize the Breakout environment wrapper
    env_wrapper = BreakoutEnvWrapper()
    
    # Define the state space dimensions (stack size, image height, image width)
    state_space_dims = (4, 84, 84)
    
    # Initialize the agent with the specified parameters
    agent = Agent(
        state_space=state_space_dims,
        action_space=env_wrapper.action_space,
        replay_memory_capacity=10000,
        batch_size=32,
        gamma=0.99,
        lr=0.0001,
        target_update=10,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    # Process the initial environment state (stacking four initial frames)
    initial_state = env_wrapper.reset()
    state_stack = [initial_state] * 4  # Stack 4 initial states for the first state

    # Run through some steps to populate the replay memory
    for _ in range(10):
        stacked_state = np.stack(state_stack, axis=0)
        print(f"Stacked state shape: {stacked_state.shape}")
        print(f"Stacked state: {stacked_state}")
        
        action = agent.select_action(stacked_state, epsilon=0.1)
        
        print(f"Action: {action}")
        
        next_state, reward, terminated, truncated, info = env_wrapper.step(action)
        
        print(f"Next state shape: {next_state.shape}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}, Info: {info}")
        
        # Update the state stack with the new frame
        state_stack.pop(0)  # Remove the oldest frame
        state_stack.append(next_state)  # Add the new frame

        # Convert the stacked state to a format suitable for the replay memory
        next_stacked_state = np.stack(state_stack, axis=0)

        # Save the transition in the replay memory
        agent.replay_memory.push(stacked_state, action, reward, next_stacked_state, terminated, truncated)

        if terminated or truncated:
            # Start a new episode if the last one is terminated or truncated
            state = env_wrapper.reset()
            state_stack = [initial_state] * 4  # Reset the state stack

    # Close the environment after populating replay memory
    env_wrapper.env.close()
