# Breakout DQN Reinforcement Learning Project

This directory holds the implementation of a Deep Q-Network (DQN) with and without attention mechanisms for playing the game Breakout. The implementation includes the training routine, environment wrappers, and utilities for memory replay and model checkpointing.

## Project Structure

- `agent.py`: Defines the agent that interacts with the environment, including decision-making and learning mechanisms.
- `checkpoint.py`: Functions for saving and loading the state of the agent and memory.
- `config.py`: Configuration file containing hyperparameters and flags for the experiment.
- `DQN_attention_model.py`: Defines the DQN with a self-attention mechanism.
- `DQN_model.py`: Defines the standard DQN model.
- `env.py`: Environment wrapper for the Breakout game that preprocesses the states for the agent.
- `replay_memory.py`: Implements the replay memory to store transitions for training the DQN.
- `train.py`: The main script to start the training process of the agent.
- `utils.py`: Utility functions, including epsilon decay for exploration and experiment logging.

## Installation

Ensure you have Python 3.x installed and the following dependencies:

```bash
pip install torch gymnasium[extra] wandb numpy
```

## Training
To train the agent, run:

```bash
python train.py
```

Adjust hyperparameters in config.py as needed. For example, you can choose between a standard DQN and a DQN with an attention mechanism.

## Testing
Testing routines are embedded within the training script, where the agent's performance is periodically evaluated.
