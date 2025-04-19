"""
Random agent for baseline comparison.
"""

import numpy as np

class RandomAgent:
    """
    Random agent that selects actions uniformly at random.
    Used as a baseline for comparison.
    """
    def __init__(self, action_dim, discrete=True, **kwargs):
        """
        Initialize the random agent.
        
        Args:
            action_dim (int): Dimension of the action space
            discrete (bool): Whether the action space is discrete
            **kwargs: Additional arguments (unused)
        """
        self.action_dim = action_dim
        self.discrete = discrete
    
    def select_action(self, state, deterministic=False):
        """
        Select a random action.
        
        Args:
            state: Current state (unused)
            deterministic (bool): Whether to be deterministic (unused)
            
        Returns:
            int or numpy.ndarray: Random action
        """
        if self.discrete:
            return np.random.randint(0, self.action_dim)
        else:
            # For continuous actions, sample from [-1, 1]
            return 2.0 * np.random.random(self.action_dim) - 1.0
    
    def update(self, state, action, reward, next_state, done):
        """
        Update the agent (no-op for random agent).
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
            
        Returns:
            dict: Empty dictionary
        """
        return {}
    
    def save(self, path):
        """
        Save the agent (no-op for random agent).
        
        Args:
            path (str): Path to save the model
        """
        pass
    
    @classmethod
    def load(cls, path, device=None):
        """
        Load an agent (returns a new instance for random agent).
        
        Args:
            path (str): Path to load the model from (unused)
            device (str): Device to load the model to (unused)
            
        Returns:
            RandomAgent: New random agent
        """
        return cls(action_dim=4, discrete=True)  # Default to discrete with 4 actions
    
    def train(self):
        """Set agent to training mode (no-op for random agent)."""
        pass
    
    def eval(self):
        """Set agent to evaluation mode (no-op for random agent)."""
        pass
    
    def get_hyperparameters(self):
        """
        Get agent hyperparameters.
        
        Returns:
            dict: Dictionary of hyperparameters
        """
        return {
            "action_dim": self.action_dim,
            "discrete": self.discrete,
            "type": "random"
        } 