"""
Neural network models for reinforcement learning agents.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional

class MLP(nn.Module):
    """
    Multi-layer perceptron with optional layer normalization.
    """
    def __init__(self, input_dim, output_dim, hidden_sizes, activation=nn.ReLU,
                 output_activation=None, layer_norm=False):
        """
        Initialize the MLP.
        
        Args:
            input_dim (int): Input dimension
            output_dim (int): Output dimension
            hidden_sizes (list): List of hidden layer sizes
            activation: Activation function to use
            output_activation: Output activation function (None for no activation)
            layer_norm (bool): Whether to use layer normalization
        """
        super(MLP, self).__init__()
        
        # Create layer sizes
        layer_sizes = [input_dim] + hidden_sizes + [output_dim]
        
        # Create layers
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            
            # Add layer normalization if specified (except for output layer)
            if layer_norm and i < len(layer_sizes) - 2:
                layers.append(nn.LayerNorm(layer_sizes[i + 1]))
            
            # Add activation (except for output layer)
            if i < len(layer_sizes) - 2:
                layers.append(activation())
            elif output_activation is not None:
                layers.append(output_activation())
        
        # Create sequential model
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass through the MLP.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output tensor
        """
        return self.layers(x)


class PolicyNetwork(nn.Module):
    """
    Policy network for actor-critic methods.
    Supports both discrete and continuous action spaces.
    """
    def __init__(self, state_dim, action_dim, hidden_sizes=[64, 64], discrete=True):
        """
        Initialize the policy network.
        
        Args:
            state_dim (int): State dimension
            action_dim (int): Action dimension (num actions for discrete, action vector size for continuous)
            hidden_sizes (list): List of hidden layer sizes
            discrete (bool): Whether the action space is discrete
        """
        super(PolicyNetwork, self).__init__()
        
        self.discrete = discrete
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        if discrete:
            # For discrete actions, output is logits for each action
            self.network = MLP(
                input_dim=state_dim,
                output_dim=action_dim,
                hidden_sizes=hidden_sizes,
                activation=nn.ReLU
            )
        else:
            # For continuous actions, output mean and log_std
            self.network = MLP(
                input_dim=state_dim,
                output_dim=action_dim,
                hidden_sizes=hidden_sizes,
                activation=nn.ReLU
            )
            # Separate log_std parameter
            self.log_std = nn.Parameter(torch.zeros(action_dim))
    
    def forward(self, state):
        """
        Forward pass through the policy network.
        
        Args:
            state (torch.Tensor): State tensor
            
        Returns:
            torch.distributions.Distribution: Action distribution
        """
        if self.discrete:
            # Get action logits and create categorical distribution
            logits = self.network(state)
            return torch.distributions.Categorical(logits=logits)
        else:
            # Get action mean and create normal distribution
            mean = self.network(state)
            std = torch.exp(self.log_std)
            return torch.distributions.Normal(mean, std)
    
    def get_action(self, state, deterministic=False):
        """
        Get an action from the policy.
        
        Args:
            state (torch.Tensor): State tensor
            deterministic (bool): Whether to take the deterministic action
            
        Returns:
            tuple: (action, log_prob)
        """
        # Get action distribution
        dist = self.forward(state)
        
        # Get action and log probability
        if deterministic:
            if self.discrete:
                action = torch.argmax(dist.probs, dim=-1)
            else:
                action = dist.mean
            log_prob = dist.log_prob(action)
        else:
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
            # Sum log probs for continuous actions
            if not self.discrete:
                log_prob = log_prob.sum(dim=-1)
        
        return action, log_prob


class ValueNetwork(nn.Module):
    """
    Value network for actor-critic methods.
    """
    def __init__(self, state_dim, hidden_sizes=[64, 64]):
        """
        Initialize the value network.
        
        Args:
            state_dim (int): State dimension
            hidden_sizes (list): List of hidden layer sizes
        """
        super(ValueNetwork, self).__init__()
        
        self.network = MLP(
            input_dim=state_dim,
            output_dim=1,
            hidden_sizes=hidden_sizes,
            activation=nn.ReLU
        )
    
    def forward(self, state):
        """
        Forward pass through the value network.
        
        Args:
            state (torch.Tensor): State tensor
            
        Returns:
            torch.Tensor: Value prediction
        """
        return self.network(state)


class QNetwork(nn.Module):
    """
    Q-network for value-based methods like DQN.
    """
    def __init__(self, state_dim, action_dim, hidden_sizes=[64, 64]):
        """
        Initialize the Q-network.
        
        Args:
            state_dim (int): State dimension
            action_dim (int): Action dimension (number of discrete actions)
            hidden_sizes (list): List of hidden layer sizes
        """
        super(QNetwork, self).__init__()
        
        self.network = MLP(
            input_dim=state_dim,
            output_dim=action_dim,
            hidden_sizes=hidden_sizes,
            activation=nn.ReLU
        )
    
    def forward(self, state):
        """
        Forward pass through the Q-network.
        
        Args:
            state (torch.Tensor): State tensor
            
        Returns:
            torch.Tensor: Q-values for each action
        """
        return self.network(state)


class DuelingQNetwork(nn.Module):
    """
    Dueling Q-network for value-based methods with advantage decomposition.
    """
    def __init__(self, state_dim, action_dim, hidden_sizes=[64, 64]):
        """
        Initialize the dueling Q-network.
        
        Args:
            state_dim (int): State dimension
            action_dim (int): Action dimension (number of discrete actions)
            hidden_sizes (list): List of hidden layer sizes
        """
        super(DuelingQNetwork, self).__init__()
        
        # Shared feature extractor
        self.feature_extractor = MLP(
            input_dim=state_dim,
            output_dim=hidden_sizes[-1],
            hidden_sizes=hidden_sizes[:-1],
            activation=nn.ReLU
        )
        
        # Value stream
        self.value_stream = nn.Linear(hidden_sizes[-1], 1)
        
        # Advantage stream
        self.advantage_stream = nn.Linear(hidden_sizes[-1], action_dim)
    
    def forward(self, state):
        """
        Forward pass through the dueling Q-network.
        
        Args:
            state (torch.Tensor): State tensor
            
        Returns:
            torch.Tensor: Q-values for each action
        """
        # Extract features
        features = self.feature_extractor(state)
        
        # Get value and advantage
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        # Combine value and advantages
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a')))
        return value + (advantages - advantages.mean(dim=1, keepdim=True))


class BaseNetwork(nn.Module):
    """Base class for neural networks with state dict conversion."""
    
    def load_state_dict(self, state_dict: dict):
        """
        Load state dict with key conversion for backward compatibility.
        
        Args:
            state_dict: State dict to load
        """
        # Convert keys from old format to new format
        new_state_dict = {}
        
        for old_key, param in state_dict.items():
            # Try to convert old key format to new format
            if old_key.startswith('net.'):
                new_key = old_key
            elif 'network.layers' in old_key:
                # Extract layer number
                layer_num = old_key.split('.')[2]
                new_key = old_key.replace(f'network.layers.{layer_num}', f'net.{layer_num}')
            else:
                new_key = old_key.replace('network.', 'net.')
            
            new_state_dict[new_key] = param
        
        # Load converted state dict
        super().load_state_dict(new_state_dict)


class Actor(BaseNetwork):
    """
    Actor network for PPO agent.
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_sizes: List[int]):
        """
        Initialize actor network.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_sizes: List of hidden layer sizes
        """
        super().__init__()
        
        # Store network parameters
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_sizes = hidden_sizes
        
        # Create network layers
        layers = []
        prev_size = state_dim
        
        for size in hidden_sizes:
            layers.append(nn.Linear(prev_size, size))
            layers.append(nn.ReLU())
            prev_size = size
        
        layers.append(nn.Linear(prev_size, action_dim))
        
        # Create sequential network
        self.net = nn.Sequential(*layers)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through network.
        
        Args:
            state: Input state
            
        Returns:
            Action probabilities
        """
        # Get logits
        logits = self.net(state)
        
        # Clamp logits to prevent numerical instability
        logits = torch.clamp(logits, -20.0, 20.0)
        
        # Return probabilities with stability measures
        probs = F.softmax(logits, dim=-1)
        
        # Ensure no zeros (which could lead to NaNs later in division)
        probs = torch.clamp(probs, min=1e-8, max=1.0)
        
        # Renormalize
        probs = probs / probs.sum(dim=-1, keepdim=True)
        
        return probs


class Critic(BaseNetwork):
    """
    Critic network for PPO agent.
    """
    
    def __init__(self, state_dim: int, hidden_sizes: List[int]):
        """
        Initialize critic network.
        
        Args:
            state_dim: Dimension of state space
            hidden_sizes: List of hidden layer sizes
        """
        super().__init__()
        
        # Store network parameters
        self.state_dim = state_dim
        self.hidden_sizes = hidden_sizes
        
        # Create network layers
        layers = []
        prev_size = state_dim
        
        for size in hidden_sizes:
            layers.append(nn.Linear(prev_size, size))
            layers.append(nn.ReLU())
            prev_size = size
        
        layers.append(nn.Linear(prev_size, 1))
        
        # Create sequential network
        self.net = nn.Sequential(*layers)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through network.
        
        Args:
            state: Input state
            
        Returns:
            State value
        """
        return self.net(state) 