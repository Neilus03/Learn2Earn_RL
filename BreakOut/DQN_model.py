import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepQNetwork(nn.Module):
    """ Deep Q-Network using Convolutional Neural Network. """
    
    def __init__(self, image_height, image_width, num_actions = 4):
        """
        Initialize the DQN.
        
        Parameters:
        image_height (int): The height of the input image.
        image_width (int): The width of the input image. 
                    
            #In the paper "Playing Atari with Deep Reinforcement Learning" the input image is 84x84 pixels.
        
        num_actions (int): The number of possible actions, in Breakout it is 4. #
                           
            #Actions in the Breakout environment: [0: do nothing, 1: fire (start the game), 2: move right, 3: move left]
        """
        super(DeepQNetwork, self).__init__() # Call the constructor of the parent class
        
        
        # The convolutional layers below are used to extract features from the input image.
        # Transforming the input image of shape (4, 84, 84) to a feature map of shape (32, 9, 9).
    
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=8, stride=4) # 4 input channels because of the 4 stacked frames. shape after conv1: (16, 20, 20)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2) # 32 output channels because of the 32 filters. shape after conv2: (32, 9, 9)
        
        # Fully connected layers
        # Calculate the size of the output from the last convolutional layer.
        self.feature_size = self._feature_size(image_height, image_width) # 32 * 9 * 9 = 2592
        self.fc1 = nn.Linear(in_features=self.feature_size, out_features=256) # 256 neurons in the first FC layer
        self.fc2 = nn.Linear(in_features=256, out_features=num_actions) # 4 neurons in the second FC layer (one for each action)

    def forward(self, x):
        """ Forward pass through the network."""
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1) # Flatten the tensor for the FC layer
        x = F.relu(self.fc1(x))
        return self.fc2(x)
    
    def _feature_size(self, height, width):
        """ Calculate the size of the features after convolution layers. """
        x = torch.zeros(1, 4, height, width) # Create a tensor with the shape of the input image
        x = F.relu(self.conv1(x)) # Apply the first convolutional layer
        x = F.relu(self.conv2(x)) # Apply the second convolutional layer
        return x.view(1, -1).size(1) # Flatten the tensor and return the size of the output tensor
        
        #For a 84x84 input image the output size is 2592.

# Example instantiation:
# dqn = DeepQNetwork(image_height=84, image_width=84, num_actions=4)
if __name__ == "__main__":
    dqn = DeepQNetwork(image_height=84, image_width=84, num_actions=4)
    print(dqn)
    print("Number of parameters:", sum(p.numel() for p in dqn.parameters() if p.requires_grad))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
'''
class SelfAttention(nn.Module):
    """
    Self attention layer for the convolutional layers.
    Self attention will be used to calculate the importance of each part of the feature map.
    And then the feature map will be multiplied by the importance.
    """
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        
        # Calculate the query, key, and value matrices using 1x1 convolutions 
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        
        # Initialize the gamma parameter
        self.gamma = nn.Parameter(torch.zeros(1))
        
        # Initialize the softmax layer
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        """
        Forward pass through the self attention layer.
        
        Parameters:
        x (tensor): The input tensor.
        """
        # Calculate the query, key, and value matrices
        proj_query = self.query_conv(x).view(x.size(0), -1, x.size(2) * x.size(3)).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(x.size(0), -1, x.size(2) * x.size(3))
        proj_value = self.value_conv(x).view(x.size(0), -1, x.size(2) * x.size(3))
        
        # Calculate the energy
        energy = torch.bmm(proj_query, proj_key)
        
        # Calculate the attention
        attention = self.softmax(energy)
        
        # Calculate the output
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(x.size(0), x.size(1), x.size(2), x.size(3))
        
        # Calculate the output after the self attention layer
        out = self.gamma * out + x
        return out
'''