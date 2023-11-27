import torch
import os


def save_checkpoint(agent, memory, checkpoint_dir, filename):
    '''
    Save the agent's DQN and the replay memory to a checkpoint file.
    '''
    # Ensure checkpoint directory exists.
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        'state_dict': agent.policy_net.state_dict(),
        'optimizer': agent.optimizer.state_dict(),
        'memory': memory.memory,
        'push_count': memory.push_count
    }
    torch.save(checkpoint, os.path.join(checkpoint_dir, filename))
    
def load_last_checkpoint(agent, memory, checkpoint_dir):
    '''
    Load the last checkpoint file into the agent's DQN and the replay memory.
    '''
    
    # Get list of checkpoint files in the current directory
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint')]
    # Get the last checkpoint file
    last_checkpoint_file = checkpoint_files[-1]
    # Load the last checkpoint file into the agent's DQN and the replay memory
    checkpoint = torch.load(os.path.join(checkpoint_dir, last_checkpoint_file))

    #Update the agent´s policy net based on the checkpoint 
    agent.policy_net.load_state_dict(checkpoint['state_dict'])
    
    #Update the agent´s optimizer based on the checkpoint
    agent.optimizer.load_state_dict(checkpoint['optimizer'])
    
    #Update the replay memory based on the checkpoint
    memory.memory = checkpoint['memory']
    
    #Update the push count of the replay memory based on the checkpoint
    memory.push_count = checkpoint['push_count']
    
    
def remove_previous_checkpoints(checkpoint_dir):
    '''
    Remove the previous checkpoints, all but the last checkpoint.
    '''
    # Get list of checkpoint files in the current directory
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint')]
    # Remove all but the last checkpoint file
    for file in checkpoint_files[:-1]:
        os.remove(os.path.join(checkpoint_dir, file))
