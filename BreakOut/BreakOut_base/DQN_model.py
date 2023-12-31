import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepQNetwork(nn.Module):
    """ Deep Q-Network using Convolutional Neural Network. """
    
    def __init__(self, image_height, image_width, num_actions):
        super(DeepQNetwork, self).__init__() # Call the constructor of the parent class
        
        # The convolutional layers below are used to extract features from the input image.
        # Transforming the input image of shape (4, 84, 84) to a feature map of shape (32, 9, 9).
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=8, stride=4) # 4 input channels because of the 4 stacked frames. shape after conv1: (16, 20, 20)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2) # 32 output channels because of the 32 filters. shape after conv2: (32, 9, 9)
        
        
        # Calculate the size of the output from the last convolutional layer.
        self.feature_size = self._feature_size(image_height, image_width) # 32 * 9 * 9 = 2592
        
        # Fully connected layers
        self.fc1 = nn.Linear(in_features=self.feature_size, out_features=256) # 256 neurons in the first FC layer
        self.fc2 = nn.Linear(in_features=256, out_features=num_actions) # 4 neurons in the second FC layer (one for each action)

    def forward(self, x):
        """ Forward pass through the network."""
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1) # Flatten the tensor for the FC layer
        x = F.relu(self.fc1(x))
        return self.fc2(x) #Output shape: [batch_size, num_actions]
    
    def _feature_size(self, height, width):
        """ Calculate the size of the features after convolution layers. """
        x = torch.zeros(1, 4, height, width) # Create a tensor with the shape of the input image
        x = F.relu(self.conv1(x)) # Apply the first convolutional layer
        x = F.relu(self.conv2(x)) # Apply the second convolutional layer
        return x.view(1, -1).size(1) # Flatten the tensor and return the size of the output tensor
        
        #For a 84x84 input image the output size is 2592.

'''
Below is the code for the instantiation of the DQN model as an example.
'''

# Example instantiation:
if __name__ == "__main__":
    dqn = DeepQNetwork(image_height=84, image_width=84, num_actions=4)
    print(dqn)
    print("Number of parameters:", sum(p.numel() for p in dqn.parameters() if p.requires_grad))
    
