import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, feature_dim):
        super(SelfAttention, self).__init__()
        # This is a very simple self-attention mechanism that only uses a single linear layer.
        # It works by applying a linear layer to each feature and then applying softmax to the result.
        # The softmax function will give a weight to each feature, which will be used to multiply the feature.
        # This will make the network focus on the most important features.
        #Here we understand a feature as a column of the feature map.
        self.attention = nn.Linear(feature_dim, 1)

    def forward(self, x):
        # Apply attention to each feature and get a weight for each feature.
        attention_weights = F.softmax(self.attention(x), dim=1) # dim=1 because we want to apply softmax to each feature (column)
        
        # Multiply each feature by its attention weight and return the result.
        return x * attention_weights.expand_as(x) # expand_as(x) because we want to multiply each feature by its attention weight

class DeepQNetwork(nn.Module):
    """ Deep Q-Network with added self-attention to focus on the paddle and ball. """

    def __init__(self, image_height, image_width, num_actions=4):
        super(DeepQNetwork, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2)
        
        # Attention layer
        self.feature_size = self._feature_size(image_height, image_width)
        self.attention = SelfAttention(self.feature_size)

        # Fully connected layers
        self.fc1 = nn.Linear(in_features=self.feature_size, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=num_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        
        # Flatten the features and apply attention
        x = x.view(x.size(0), -1)
        x = self.attention(x)
        
        x = F.relu(self.fc1(x))
        return self.fc2(x)
    
    def _feature_size(self, height, width):
        x = torch.zeros(1, 4, height, width)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x.view(1, -1).size(1)

# Example instantiation:
if __name__ == "__main__":
    dqn = DeepQNetwork(image_height=84, image_width=84, num_actions=4)
    print(dqn)
    print("Number of parameters:", sum(p.numel() for p in dqn.parameters() if p.requires_grad))
