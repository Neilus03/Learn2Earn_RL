import random
from collections import namedtuple

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
        
        #if memory is full, overwrite the oldest experience
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
            return random.sample(self.memory, batch_size) # sample because is more efficient than using random.choice()

    def can_provide_sample(self, batch_size): 
        '''
        Check if the memory list is full enough to provide a random batch of experiences for training.
        Because we can't train the DQN if the memory list is not full enough.
        At the first episodes the memory list is not full enough, so we need to wait until it is full enough.
        Then we can start training the DQN. Luckily, the memory list fills up quickly.
        '''
        # If the length of the memory list is greater than or equal
        # to the batch size, return True, otherwise return False
        return len(self.memory) >= batch_size 



# Example usage of the ReplayMemory class:
if __name__ == '__main__':
    memory = ReplayMemory(capacity=10000) # Create a replay memory with a capacity of 10000 experiences
    print("Initial memory size:", len(memory.memory)) # Print the initial size of the memory list
    print("Initial push count:", memory.push_count) # Print the initial push count

    for i in range(1, 10): 
        memory.push(i, i, i, i, i, i) # Push a new experience to the memory list
        print("Memory size:", len(memory.memory)) # Print the size of the memory list
        print("Push count:", memory.push_count) # Print the push count
        print("Memory:", memory.memory) # Print the memory list
        print("Sample:", memory.sample(3)) # Print a random sample of 3 experiences from the memory list
        print("Can provide sample:", memory.can_provide_sample(3)) # Print whether the memory list is full enough to provide
                                                                   #  a random sample of 3 experiences for training
        print()
