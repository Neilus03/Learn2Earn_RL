import gymnasium as gym
import numpy as np
import cv2


class BreakoutEnvWrapper:
    """ 
    A wrapper for the Gymnasium Breakout environment that preprocesses 
    the images and simplifies interaction with the environment.
    """

    def __init__(self, env_name='ALE/Breakout-v5',  max_episode_steps=1000):
        self.env = gym.make(env_name) # Create the environment
        self.env._max_episode_steps = max_episode_steps # Set the maximum number of steps in an episode
        self.render_mode = 'rgb_array' # Set the render mode to rgb_array
        self.metadata = self.env.metadata # Get the metadata of the underlying gymnasium environment
        self.metadata['render_fps'] = 30
        
        # Get the action space, color space and needed shape of the environment
        self.action_space = self.env.action_space.n # The number of possible actions in the environment (4 in Breakout)
        self.color_space = cv2.COLOR_RGB2GRAY  # Color space to convert the image to (grayscale) as in the paper "Playing Atari with Deep Reinforcement Learning"
        self.resize_shape = (84, 84)  # Target resize shape for the image (84x84) as in the paper "Playing Atari with Deep Reinforcement Learning"
        
        
    def preprocess(self, observation):
        """
        Preprocess the raw image data from the environment by converting 
        it to grayscale, resizing it, and normalizing it.
        """
        # Convert to grayscale, resize, and normalize the image
        gray = cv2.cvtColor(observation, self.color_space) # Convert the image to grayscale
        resized = cv2.resize(gray, self.resize_shape) # Resize the image
        normalized = resized / 255.0 # Normalize the image (values between 0 and 1)
        return normalized.astype(np.float32)  # Return the image as a float32


    def reset(self, **kwargs):
        """
        Resets the environment and preprocesses the initial observation.
        Now handles additional keyword arguments for compatibility with Gymnasium's RecordVideo wrapper.
        """
        observation, info = self.env.reset(**kwargs) # Get the initial observation from the environment
        return self.preprocess(observation)  # Preprocess the observation (change color space, resize, normalize) and return it


    def step(self, action):
        """
        Takes an action in the environment and preprocesses the new observation,
        which will be fed to the agent at the next timestep.
        """
        observation, reward, terminated, truncated, info = self.env.step(action) # Take an action in the environment and get the new observation, reward, and other info
        return self.preprocess(observation), reward, terminated, truncated, info # Preprocess the observation (change color space, resize, normalize) and return it, along with the reward, and other info


    def render(self, mode='rgb_array'):
        """ Renders the environment on screen. """
        self.env.render() # Render the environment on screen (currently failing for some reason)

    def close(self):
        """ Close the environment. """
        self.env.close() # Close the environment



# Test the wrapper by running this file directly
if __name__ == "__main__":
    env_wrapper = BreakoutEnvWrapper()
    state = env_wrapper.reset()
    print("Initial state shape:", state.shape)

    for _ in range(10):
        action = env_wrapper.env.action_space.sample()
        next_state, reward, terminated, truncated, info = env_wrapper.step(action)
        print(f"Next state shape: {next_state.shape}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}, Info: {info}")

        # Attempt to render the environment on screen
        #env_wrapper.render() #This is failing now, I have to check why it is failing

        if terminated or truncated:
            print("Episode finished after 10 steps.")
            break

    env_wrapper.env.close()
