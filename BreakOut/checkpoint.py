import torch

def save_checkpoint(agent, memory, filename):
    '''
    Save the agent's DQN and the replay memory to a checkpoint file.
    '''
    checkpoint = {
        'state_dict': agent.policy_net.state_dict(),
        'optimizer': agent.optimizer.state_dict(),
        'memory': memory.memory,
        'push_count': memory.push_count
    }
    torch.save(checkpoint, filename)
    
def load_checkpoint(agent, memory, filename):
    '''
    Load the agent's DQN and the replay memory from a checkpoint file.
    '''
    checkpoint = torch.load(filename)
    agent.policy_net.load_state_dict(checkpoint['state_dict'])
    agent.optimizer.load_state_dict(checkpoint['optimizer'])
    memory.memory = checkpoint['memory']
    memory.push_count = checkpoint['push_count']
    