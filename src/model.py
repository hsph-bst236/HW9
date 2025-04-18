import torch
from torch import nn
import numpy as np
from torch.distributions import Categorical

    
class MlpPolicy(nn.Module):
    """
    Policy network with MLP feature extractor
    
    This network has two components:
    1. A policy network that outputs a distribution over actions
    2. A value network that estimates the value function
    
    Both networks use two hidden layers [64, 64] with Tanh activation.
    Note that the input shape is a 3D tensor (num_frames, height, width).
    """
    def __init__(self, input_shape: tuple[int, int, int], action_size: int):
        super().__init__()
        hidden_size = 64
        # TODO: [part a]
        ### YOUR CODE HERE ###
        pass
        ### END YOUR CODE ###
    
    def forward(self, x) -> tuple[Categorical, torch.Tensor]: 
        """
        Forward pass through the policy network
        
        Args:
            x: Input observation tensor
            
        Returns:
            pi: Categorical distribution over actions
            value: Value function estimate
        """
        # TODO: [part a] 
        ### YOUR CODE HERE ###
        pass
        ### END YOUR CODE ###
    
    def predict(self, x, deterministic=False):
        with torch.no_grad():
            pi, value = self(x)
            if deterministic:
                action = pi.probs.argmax(dim=-1).cpu().numpy()[0]
            else:
                action = pi.sample().cpu().numpy()[0]
            return action, value
        
def load_model(path, env):
    '''
    Load a model from a path
    '''
    checkpoint = torch.load(path)
    
    # Determine shape of observations
    obs_shape = env.env.observation_space.shape
    obs_shape = [obs_shape[2], obs_shape[0], obs_shape[1]]  # (4, 13, 16)
    action_size = env.env.action_space.n
    
    model = MlpPolicy(obs_shape, action_size).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    
    
    # Load model state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # Set to evaluation mode
    
    return model



