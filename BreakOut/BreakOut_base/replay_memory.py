import random 
from collections import namedtuple
from utils import update_state_stack
import numpy as np

# Define a named tuple to store experiences for better access to the data fields
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'terminated', 'truncated'])

class ReplayMemory:
    ''' 
    A cyclic buffer of bounded size that holds the transitions observed recently. 
    This is impotant for training DQN because the samples are highly correlated, and therefore not independent.
    With the replay memory we can break this correlation and therefore improve the training by
    sampling random batches of transitions from the replay memory, as I understood from Jordi's class.
    It also implements a .sample() method for selecting a random batch of transitions for training.
    '''
    def __init__(self, capacity):
        '''
        initialize the replay memory with a maximum capacity, and an empty list to store the experiences
        '''        
        self.capacity = capacity
        self.memory = []
        self.push_count = 0

    def push(self, state, action, reward, next_state, terminated, truncated):
        '''
        Push a new experience to the replay memory if it is not full, otherwise overwrite the oldest experience.
        we do this by using the modulo operator to get the remainder of the division of the push_count by the capacity.
        This needs to be done because we don't want to increase the size of the memory list, we want to keep it fixed.
        '''
        #Check if the memory is full comparing the length of the memory list to the capacity
        if len(self.memory) < self.capacity: # If the memory is not full, append a new experience
            self.memory.append(None) # Append a new empty experience to the memory list, it is None because we don't have any experience yet
        
        else:#if memory is full, overwrite the oldest experience
            self.memory[self.push_count % self.capacity] = Experience(state, action, reward, next_state, terminated, truncated) #Overwrite the oldest experience with the new experience
        
        # Increase the push_count by 1 when each new experience is added to the memory list 
        # whatever the case is (whether the memory is full or not)
        self.push_count += 1

    def sample(self, batch_size):
        '''
        Sample a random batch of experiences from the replay memory by using the
        random.sample() method which returns a list of random samples from the memory list.
        This is done for getting a random batch of experiences for training the DQN.
        '''
        if self.can_provide_sample(batch_size): # Check if the memory list is full enough to provide a random batch of experiences for training
            #return a sample of batch_size number of experiences from the memory list but without None experiences
            return random.sample([e for e in self.memory if e is not None], batch_size) # sample because is more efficient than using random.choice()
        else:
            return None # If the memory list is not full enough, return None because we can't provide a random batch of experiences for training
        
    def can_provide_sample(self, batch_size): 
        '''
        Check if the memory list is full enough to provide a random batch of experiences for training, and make sure that there are at
        least batch_size non None experiences.
        Because we can't train the DQN if the memory list is not full enough, and training it with Nones wouldnt be useful either.
        At the first episodes the memory list is not full enough, so we need to wait until it is full enough.
        Then we can start training the DQN. Luckily, the memory list fills up quickly.
        '''
        # Check if the memory list is full enough to provide a random batch of experiences for training
        # and make sure that there are at least batch_size non None experiences in the memory list
        return len(self.memory) >= batch_size and self.memory.count(None) < len(self.memory) - batch_size
    
    def prepopulate_replay_memory(self, env, batch_size, action_space):
        '''
        Description:
            Prepopulate the replay memory with random experiences. This is done to avoid the problem of
            the agent learning from a small number of experiences at the beginning of the training.
            If the agent learns from a small number of experiences at the beginning of the training,
            it will not learn well because it will not have enough experiences to learn from.
            
        Parameters:
            env (gym environment): The environment.
            replay_memory (ReplayMemory object): The replay memory.
            prepopulate_steps (int): The number of experiences to prepopulate the replay memory.
            action_space (int): The number of possible actions.
            
        Returns:
            None, it populates the replay memory with random experiences in place.
        '''
        # Reset the environment and stack the initial state 4 times
        state = env.reset()
        state = np.stack([state] * 4, axis=0)  # Initial state stack
        
        # Prepopulate the replay memory with random experiences
        while self.can_provide_sample(batch_size) == False: # Check if the replay memory can provide a random batch of experiences for training
            
            # Select a random action
            action = random.randrange(action_space)
            # Perform the action and observe the next state, reward, terminated and truncated
            next_state, reward, terminated, truncated, info = env.step(action)
            
            # Stack the next state and update the state stack
            next_state_stack = update_state_stack(state, next_state) 
            
            # Push the experience to the replay memory for populating it
            self.push(state, action, reward, next_state_stack, terminated, truncated)
            
            # Update the state for the next iteration
            state = next_state_stack
                
        print(f"Prepopulated with {len(self.memory)} experiences.")


        
        

# Example usage of the ReplayMemory class:
if __name__ == '__main__':
    memory = ReplayMemory(capacity=10) # Create a replay memory with a capacity of 10000 experiences
    print("Initial memory size:", len(memory.memory)) # Print the initial size of the memory list
    print("Initial push count:", memory.push_count) # Print the initial push count

    for i in range(1, 30): 
        memory.push(i, i, i, i, i, i) # Push a new experience to the memory list
        print("Memory size:", len(memory.memory)) # Print the size of the memory list
        print("Push count:", memory.push_count) # Print the push count
        print("Memory:", memory.memory) # Print the memory list
        print("Sample:", memory.sample(3)) # Print a random sample of 3 experiences from the memory list
        print("Can provide sample:", memory.can_provide_sample(3)) # Print whether the memory list is full enough to provide
                                                                #  a random sample of 3 experiences for training
        print()
