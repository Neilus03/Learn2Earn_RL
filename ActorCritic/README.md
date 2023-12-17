# Overview
This Python script trains an actor-critic model using PyTorch, suitable for reinforcement learning environments provided by Gymnasium. The script integrates with Weights & Biases (wandb) for logging and tracking experiments. It supports multiple environments, configurable training parameters, and offers options for rendering, logging, and model saving.

# Execution and usage of arguments

To run the script, use the following command:

```
python train.py [arguments]
```

## Arguments

- **--env**: Specify the Gymnasium environment (Default: CartPole-v1). Supported options: CartPole-v1, MountainCar-v0, Acrobot-v1.
Usage: --env CartPole-v1
- **--gamma**: Discount factor for future rewards (Default: 0.99).
Usage: --gamma 0.99
- **--lr**: Learning rate for the optimizer (Default: 3e-2).
Usage: --lr 2e-3
- **--seed**: Random seed for reproducibility (Default: 543).
Usage: --seed 123
- **--render**: Render the environment during training. This is a flag without value. It is a boolean flag. If you ommit it is set to false
Usage: --render
- **--log-interval**: Interval between logging training status (Default: 10).
Usage: --log-interval 10
- **--wandb**: Enable Weights & Biases logging. This is a flag without value. If it is given, you should change the code to put your credential from your wandb account
Usage: --wandb
- **--save-interval**: Interval between saving the model (Default: 100). The interval is the number of episodes
Usage: --save-interval 100

## Example Command

```
python train.py --env MountainCar-v0 --gamma 0.95 --lr 1e-2 --seed 42 --log-interval 20 --save-interval 50 --wandb
```
This command trains the model on MountainCar-v0 environment with a learning rate of 1e-2, gamma value of 0.95, random seed 42, logs every 20 episodes, saves the model every 50 episodes, and enables wandb logging.
