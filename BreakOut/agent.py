import torch
import torch.optim as optim
import torch.nn.functional as F
import random
from DQN_model import DeepQNetwork
from replay_memory import ReplayMemory, Experience
from env import BreakoutEnvWrapper


class Agent:
    def __init__(self, state_space, action_space, replay_memory_capacity, batch_size, gamma, lr, target_update, device):
        '''
        initialize the agent.
        '''
        self.state_space = state_space # shape of the input image (F, 84, 84) in Breakout, where F is the number of stacked frames
        print("state_space:", state_space)
        self.action_space = action_space # The number of possible actions
        self.replay_memory = ReplayMemory(replay_memory_capacity) # Create the replay memory with the specified capacity
        self.batch_size = batch_size # The size of the batch of experiences sampled from the replay memory
        self.gamma = gamma # The discount factor
        self.lr = lr # The learning rate of the optimizer
        self.target_update = target_update # The number of episodes between updating the target network
        self.device = device # The device to use for training the DQN (cpu or gpu)
        self.policy_net = DeepQNetwork(state_space[1], state_space[2], action_space).to(device) # The policy network
        self.target_net = DeepQNetwork(state_space[1], state_space[2], action_space).to(device) # The target network
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
            terminateds = torch.tensor(batch.terminated, device=self.device, dtype=torch.uint8) # a batch of terminated flags
            truncateds = torch.tensor(batch.truncated, device=self.device, dtype=torch.uint8) # a batch of truncated flags
            dones = terminateds or truncateds # dones is True if the episode is terminated or truncated, otherwise it is False
            
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
        if self.push_count % self.target_update == 0: # We don't do this every episode because it is expensive
            self.target_net.load_state_dict(self.policy_net.state_dict()) # Update the weights of the target network to be the same as the policy network

# Example usage:

if __name__ == "__main__":
    agent = Agent(state_space=4, action_space=4, replay_memory_capacity=10000, batch_size=32, gamma=0.99, lr=0.0001, target_update=10, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    print(agent.policy_net)
    print("Number of parameters:", sum(p.numel() for p in agent.policy_net.parameters() if p.requires_grad))
    print("\nNumber of layers:", len(list(agent.policy_net.parameters())))
    print("\nNumber of convolutional layers:", len(list(agent.policy_net.conv1.parameters())) + len(list(agent.policy_net.conv2.parameters())))
    print("\nNumber of fully connected layers:", len(list(agent.policy_net.fc1.parameters())) + len(list(agent.policy_net.fc2.parameters())))
    print("\nNumber of neurons in the first fully connected layer:", len(list(agent.policy_net.fc1.parameters())))
    print("\nNumber of neurons in the second fully connected layer:", len(list(agent.policy_net.fc2.parameters())))
    print("\nNumber of neurons in the convolutional layers:", len(list(agent.policy_net.conv1.parameters())) + len(list(agent.policy_net.conv2.parameters())))
    print("\nNumber of parameters in the optimizer:", len(list(agent.optimizer.state_dict()["state"].keys())))
    print("\nNumber of parameters in the replay memory:", len(agent.replay_memory.memory))
    print("\nNumber of parameters in the batch:", len(agent.replay_memory.sample(32)))
    print("\nNumber of parameters in the experience:", len(agent.replay_memory.sample(32)[0]))
    print("\nNumber of parameters in the state:", len(agent.replay_memory.sample(32)[0].state))
    print("\nNumber of parameters in the action:", len(agent.replay_memory.sample(32)[0].action))
    print("\nNumber of parameters in the reward:", len(agent.replay_memory.sample(32)[0].reward))
    print("\nNumber of parameters in the next state:", len(agent.replay_memory.sample(32)[0].next_state))
    print("\nNumber of parameters in the terminated flag:", len(agent.replay_memory.sample(32)[0].terminated))
    print("\nNumber of parameters in the truncated flag:", len(agent.replay_memory.sample(32)[0].truncated))
    print("\nNumber of parameters in the dones flag:", len(agent.replay_memory.sample(32)[0].terminated or agent.replay_memory.sample(32)[0].truncated))
    print("\nNumber of parameters in the states tensor:", len(torch.tensor(agent.replay_memory.sample(32)[0].state, device=agent.device, dtype=torch.float32)))
    print("\nNumber of parameters in the actions tensor:", len(torch.tensor(agent.replay_memory.sample(32)[0].action, device=agent.device, dtype=torch.int64)))
    print("\nNumber of parameters in the rewards tensor:", len(torch.tensor(agent.replay_memory.sample(32)[0].reward, device=agent.device, dtype=torch.float32)))
    print("\nNumber of parameters in the next states tensor:", len(torch.tensor(agent.replay_memory.sample(32)[0].next_state, device=agent.device, dtype=torch.float32)))
    print("\nNumber of parameters in the terminateds tensor:", len(torch.tensor(agent.replay_memory.sample(32)[0].terminated, device=agent.device, dtype=torch.uint8)))
    print("\nNumber of parameters in the truncateds tensor:", len(torch.tensor(agent.replay_memory.sample(32)[0].truncated, device=agent.device, dtype=torch.uint8)))
    print("\nNumber of parameters in the dones tensor:", len(torch.tensor(agent.replay_memory.sample(32)[0].terminated or agent.replay_memory.sample(32)[0].truncated, device=agent.device, dtype=torch.uint8)))
    print("\nNumber of parameters in the current q values:", len(agent.policy_net(torch.tensor(agent.replay_memory.sample(32)[0].state, device=agent.device, dtype=torch.float32)).gather(1, torch.tensor(agent.replay_memory.sample(32)[0].action, device=agent.device, dtype=torch.int64).unsqueeze(-1))))
    print("\nNumber of parameters in the next q values:", len(agent.target_net(torch.tensor(agent.replay_memory.sample(32)[0].next_state, device=agent.device, dtype=torch.float32)).max(1)[0].detach()))
    print("\nNumber of parameters in the expected q values:", len(torch.tensor(agent.replay_memory.sample(32)[0].reward, device=agent.device, dtype=torch.float32) + (agent.gamma * agent.target_net(torch.tensor(agent.replay_memory.sample(32)[0].next_state, device=agent.device, dtype=torch.float32)).max(1)[0].detach() * (1 - (torch.tensor(agent.replay_memory.sample(32)[0].terminated, device=agent.device, dtype=torch.uint8) or torch.tensor(agent.replay_memory.sample(32)[0].truncated, device=agent.device, dtype=torch.uint8))))))
    print("\nNumber of parameters in the loss:", len(F.mse_loss(agent.policy_net(torch.tensor(agent.replay_memory.sample(32)[0].state, device=agent.device, dtype=torch.float32)).gather(1, torch.tensor(agent.replay_memory.sample(32)[0].action, device=agent.device, dtype=torch.int64).unsqueeze(-1)), torch.tensor(agent.replay_memory.sample(32)[0].reward, device=agent.device, dtype=torch.float32) + (agent.gamma * agent.target_net(torch.tensor(agent.replay_memory.sample(32)[0].next_state, device=agent.device, dtype=torch.float32)).max(1)[0].detach() * (1 - (torch.tensor(agent.replay_memory.sample(32)[0].terminated, device=agent.device, dtype=torch.uint8) or torch.tensor(agent.replay_memory.sample(32)[0].truncated, device=agent.device, dtype=torch.uint8)))))))
    print("\nNumber of parameters in the optimizer gradients:", len(agent.optimizer.zero_grad()))
    print("\nNumber of parameters in the optimizer weights:", len(agent.optimizer.step()))
    print("\nNumber of parameters in the target network weights:", len(agent.target_net.load_state_dict(agent.policy_net.state_dict())))
    print("\nNumber of parameters in the target network:", len(agent.target_net))
    
