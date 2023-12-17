import torch
import os


def save_checkpoint(agent, memory, checkpoint_dir, filename):
    '''
    Description:
        Save the agent's DQN and the replay memory into a checkpoint file.
    Arguments:
        agent: the agent to be saved
        memory: the replay memory to be saved
        checkpoint_dir: the directory where the checkpoint file will be saved
        filename: the name of the checkpoint file
    Output:
        None, but the checkpoint file will be saved in the checkpoint_dir
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
    Description:
        Load the last checkpoint file into the agent's DQN and the replay memory.
    Arguments:
        agent: the agent to be updated
        memory: the replay memory to be updated
        checkpoint_dir: the directory where the checkpoint file is saved
    Output:
        None, but the agent's DQN and the replay memory will be updated
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
    Description:
        Remove all the checkpoint files except the last one for 
        efficient memory usage.
    Arguments:
        checkpoint_dir: the directory where the checkpoint files are saved
    Output:
        None, but the checkpoint files will be removed
    '''
    # Get list of checkpoint files in the current directory
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint')]

    # remove the checkpoints with lower index than the last checkpoint
    for checkpoint_file in checkpoint_files[:-1]:
        os.remove(os.path.join(checkpoint_dir, checkpoint_file))
    
