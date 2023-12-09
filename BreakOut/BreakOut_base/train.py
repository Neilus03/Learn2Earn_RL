from agent import Agent
from env import BreakoutEnvWrapper
from replay_memory import ReplayMemory
from utils import get_epsilon, update_state_stack
from checkpoint import save_checkpoint, load_last_checkpoint, remove_previous_checkpoints
import config


import numpy as np
import wandb
import gymnasium as gym

# Initialize wandb for logging metrics
if config.log_to_wandb:
    wandb.init(project="breakout-dqn1", entity="ai42")


def train():
    '''
    Description:
            Train the agent in the Breakout environment using the DQN algorithm.
            This is done by interacting with the environment, collecting experiences,
            and training the agent using the collected experiences.
    '''
    # Initialize the environment and the agent
    env = BreakoutEnvWrapper()# Create the Breakout environment
 
    agent = Agent(state_space=(4, 84, 84), 
                  action_space=env.action_space, 
                  replay_memory_capacity=config.replay_memory_size, 
                  batch_size=config.batch_size,
                  gamma=config.gamma, 
                  lr=config.lr, 
                  target_update=config.target_update, 
                  device=config.device)

    # Initialize replay memory with a capacity of replay_memory_size
    replay_memory = ReplayMemory(config.replay_memory_size)

    # Prepopulate the replay memory with replay_memory_size experiences to start training the agent 
    #prepopulate_replay_memory(env, replay_memory, config.replay_memory_size, agent.action_space)
    
    # Load a checkpoint if pretrained is True
    if config.pretrained:
        load_last_checkpoint(agent, replay_memory, config.checkpoint_dir)
        print("Checkpoint loaded successfully")

    best_reward = -np.inf # Initialize the best reward to -infinity 
    last_5_episodes_rewards = [-5] * 5 # Initialize a list to store the rewards of the last 5 episodes and set all the rewards to 0 initially
    
    # Initialize a list to store the avg losses of the episodes
    losses = []
    
    # Begin training loop over episodes
    for episode in range(config.num_episodes): 
        # Initialize the state by resetting the environment
        state = env.reset() 
        
        # Initialize the stack of states by stacking the same state 4 times, once the state is
        # updated, the stack will be updated too, so it wont be the same state 4 times anymore
        state = np.stack([state] * 4, axis=0) # shape: (4, 84, 84)
        
        # Initialize the episode reward to 0
        episode_reward = 0 
        
        # Initialize the number of lives to 5, because the agent starts with 5 lives
        # each time the agent loses a life, the reward of the current step will be set to -1
        current_lives = 5 
        
        #Track episode losses
        episode_losses = [] # Initialize a list to store the losses of the training steps of the current episode
        
        # Begin episode loop over steps
        for step in range(config.max_steps): 
            # Select an action using the agent's policy
            action = agent.select_action(state, get_epsilon(episode)) # Epsilon-greedy policy, where epsilon is decayed over episodes
            
            # Take the selected action in the environment and get the next state, reward, and other info
            next_state, reward, terminated, truncated, info = env.step(action) 
            
            # Check if the agent lost a life, if so, set the reward to -1
            if config.negative_reward:
                if info['lives'] < current_lives: # Check if the agent lost a life
                    reward = -1 # If the agent lost a life, set the reward to -1
                    current_lives = info['lives'] # Update the number of lives to the current number of lives
            
            # Update the episode reward by adding the reward of the current step
            episode_reward += reward
            
            # Update the stack of states by stacking the next state on top of the stack of states
            next_state_stack = update_state_stack(state, next_state) # shape: (4, 84, 84)
            
            # Push a new experience to the replay memory, overwriting the oldest experience if the memory is full
            replay_memory.push(state, action, reward, next_state_stack, terminated, truncated)

            # Update the state to the next stack of states
            state = next_state_stack

            # Check if the replay memory can provide a random batch of experiences for training, if so, train the agent
            if replay_memory.can_provide_sample(agent.batch_size):
                # Train the agent using a random batch of experiences from the replay memory and save the loss of the training step
                # This is done by using the agent's learn() method, which returns the loss of the training step while training the agent
                loss = agent.learn(env)
                

                if loss is not None:# Check if the loss is not None, if so, append the loss to the episode_losses list
                    episode_losses.append(loss)# Append the loss of the training step to the losses list
            else:
                print("Waiting for replay memory to fill up...")
            
            # Check if the episode is terminated or truncated, if so, break the episode loop
            if terminated or truncated:
                break
            
        print(f"Episode: {episode} Episode Reward: {episode_reward} Episode Loss: {np.mean(episode_losses)}")
        
        # Update the target network every target_update episodes 
        if episode % config.target_update == 0:
            print("Updating target network...")
            agent.update_target_network()


        #Average the losses of the training steps of the current episode
        avg_loss = np.mean(episode_losses) if episode_losses else None# Get the mean of the losses of the training steps of the current episode
        losses.append(avg_loss) # Append the mean loss of the training steps of the current episode to the losses list
        
        # Append the episode reward to the last_5_episodes_rewards list
        last_5_episodes_rewards.append(episode_reward)
        
        #Get the mean reward of the last 5 episodes 
        last_5_episodes_mean_reward = sum(last_5_episodes_rewards[-5:]) / 5 
         
         
        # Save checkpoints if the episode reward is better than the best reward so far
        if last_5_episodes_mean_reward > best_reward:       
            # Save the agent's DQN and the replay memory to a checkpoint file 
            save_checkpoint(agent, replay_memory, config.checkpoint_dir, f'checkpoint{episode}.pth')
            
            # Print the episode number and the episode reward if checkpoint is saved
            print(f"Checkpoint saved at episode {episode} with reward {episode_reward}")
            
            # Update the best reward to the episode reward
            best_reward = episode_reward
            
            # Remove previous checkpoints if free_checkpoint_memory is True
            if config.free_checkpoint_memory:
                remove_previous_checkpoints(config.checkpoint_dir)
            
            
        # Log the episode number, the number of steps in the episode, the total reward of the episode and the loss of the training step to wandb
        if config.log_to_wandb:
            wandb.log({"Episode": episode,
                       "Steps": step,
                       "Episode Reward": episode_reward,
                       "Average Loss": avg_loss })

        if episode %1000 == 0:
            print(f"Episode: {episode} Episode Reward: {episode_reward}")
        
    env.close() # Close the environment
    if config.log_to_wandb:
        wandb.finish() # Finish the wandb run
if __name__ == "__main__":
    train()
