"""
Wrapper for PPO agent to match the interface expected by the training loop.
"""

import torch
import numpy as np
from src.agents.ppo_agent import PPOAgent

class PPOAgentWrapper:
    """
    Wrapper around PPOAgent to match the interface expected by the training loop.
    """
    def __init__(self, state_dim, action_dim, discrete=True, device="cpu", **kwargs):
        """
        Initialize PPO agent wrapper.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            discrete: Whether action space is discrete
            device: Device to use for tensors
            **kwargs: Additional arguments for PPOAgent
        """
        self.agent = PPOAgent(state_dim, action_dim, discrete, device, **kwargs)
        self.device = device
        self.buffer = []
        self.update_frequency = kwargs.get("update_frequency", 128)  # How often to do PPO updates
        self.steps_since_update = 0
        
    def select_action(self, state):
        """Wrapper around agent's select_action method."""
        action = self.agent.select_action(state)
        return action
    
    def update(self, state, action, reward, next_state, done):
        """
        Collect experience and periodically update the policy.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
            
        Returns:
            dict: Update metrics if an update was performed, else None
        """
        # Compute log probability and value for the state-action pair
        with torch.no_grad():
            # Convert state to tensor if needed
            if not isinstance(state, torch.Tensor):
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            else:
                state_tensor = state.unsqueeze(0) if state.dim() == 1 else state
                
            # Get log probability and value
            probs = self.agent.actor(state_tensor)
            if self.agent.discrete:
                log_prob = torch.log(probs.squeeze(0)[action] + 1e-10)
            else:
                # Continuous case (not fully implemented here)
                log_prob = torch.tensor([0.0], device=self.device)  # Placeholder
                
            value = self.agent.critic(state_tensor).squeeze()
        
        # Add experience to buffer with log_prob and value
        self.agent.buffer.add(state, action, reward, next_state, done, log_prob, value)
        self.steps_since_update += 1
        
        # Only update periodically and if we have enough data
        if self.steps_since_update >= self.update_frequency and self.agent.buffer.size >= self.agent.batch_size:
            self.steps_since_update = 0
            
            # Call update and handle None return
            update_result = self.agent.update()
            return update_result if update_result is not None else {}
        
        return None
    
    def save(self, path):
        """Wrapper around agent's save method."""
        self.agent.save(path)
    
    def load(self, path):
        """Wrapper around agent's load method."""
        self.agent.load(path)
    
    def get_hyperparameters(self):
        """Wrapper around agent's get_hyperparameters method."""
        return self.agent.get_hyperparameters() if hasattr(self.agent, "get_hyperparameters") else {} 