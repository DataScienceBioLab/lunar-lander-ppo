"""
Buffer modules for storing agent experiences.
Includes implementations for standard replay buffer and prioritized replay buffer.
"""

import numpy as np
import torch
from collections import deque
import random
from typing import Dict, List, Tuple

class ReplayBuffer:
    """
    Standard replay buffer for off-policy RL algorithms.
    """
    def __init__(self, capacity, device="cpu"):
        """
        Initialize a replay buffer.
        
        Args:
            capacity (int): Maximum buffer capacity
            device (str): Device to store tensors on
        """
        self.buffer = deque(maxlen=capacity)
        self.device = device
    
    def add(self, state, action, reward, next_state, done):
        """
        Add a transition to the buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
        """
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """
        Sample a batch of transitions from the buffer.
        
        Args:
            batch_size (int): Number of transitions to sample
            
        Returns:
            tuple: Batch of transitions (states, actions, rewards, next_states, dones)
        """
        states, actions, rewards, next_states, dones = zip(*random.sample(self.buffer, batch_size))
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).unsqueeze(1).to(self.device)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        """
        Get the current size of the buffer.
        
        Returns:
            int: Current buffer size
        """
        return len(self.buffer)


class PrioritizedReplayBuffer:
    """
    Prioritized replay buffer for off-policy RL algorithms.
    Uses sum-tree data structure for efficient sampling based on priorities.
    """
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001, device="cpu"):
        """
        Initialize a prioritized replay buffer.
        
        Args:
            capacity (int): Maximum buffer capacity
            alpha (float): Priority exponent parameter
            beta (float): Importance sampling exponent parameter
            beta_increment (float): Beta increment per sampling
            device (str): Device to store tensors on
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.device = device
        
        # Initialize buffer
        self.buffer = []
        self.position = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.max_priority = 1.0
    
    def add(self, state, action, reward, next_state, done):
        """
        Add a transition to the buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
        """
        # Store experience
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)
        
        # Set max priority for new experience
        self.priorities[self.position] = self.max_priority
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        """
        Sample a batch of transitions from the buffer based on priorities.
        
        Args:
            batch_size (int): Number of transitions to sample
            
        Returns:
            tuple: Batch of transitions (states, actions, rewards, next_states, dones, weights, indices)
        """
        if len(self.buffer) < self.capacity:
            priorities = self.priorities[:len(self.buffer)]
        else:
            priorities = self.priorities
        
        # Calculate sampling probabilities
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        # Sample indices based on probabilities
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        
        # Get experiences at indices
        states, actions, rewards, next_states, dones = zip(*[self.buffer[idx] for idx in indices])
        
        # Calculate importance sampling weights
        weights = (len(self.buffer) * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()
        
        # Increment beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).unsqueeze(1).to(self.device)
        weights = torch.FloatTensor(weights).unsqueeze(1).to(self.device)
        
        return states, actions, rewards, next_states, dones, weights, indices
    
    def update_priorities(self, indices, priorities):
        """
        Update priorities of transitions.
        
        Args:
            indices (list): Indices of transitions to update
            priorities (list): New priorities
        """
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)
    
    def __len__(self):
        """
        Get the current size of the buffer.
        
        Returns:
            int: Current buffer size
        """
        return len(self.buffer)


class RolloutBuffer:
    """Buffer for storing rollout information."""
    
    def __init__(self, capacity: int, device: str = "cpu"):
        """
        Initialize buffer.
        
        Args:
            capacity: Maximum size of buffer
            device: Device to store tensors on
        """
        self.capacity = capacity
        self.device = device
        self.reset()
    
    def add(self, state: torch.Tensor, action: torch.Tensor, 
            reward: float, next_state: torch.Tensor, done: bool,
            log_prob: torch.Tensor = None, value: torch.Tensor = None):
        """
        Add a transition to the buffer.
        
        Args:
            state: Current state (torch.Tensor)
            action: Action taken (torch.Tensor)
            reward: Reward received (float)
            next_state: Next state (torch.Tensor)
            done: Whether the episode is done (bool)
            log_prob: Log probability of action (torch.Tensor)
            value: Value of state (torch.Tensor)
        """
        # Convert to tensors if they're not already
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).to(self.device)
        if not isinstance(action, torch.Tensor):
            action = torch.LongTensor([action]).to(self.device)
        if not isinstance(reward, torch.Tensor):
            reward = torch.FloatTensor([reward]).to(self.device)
        if not isinstance(next_state, torch.Tensor):
            next_state = torch.FloatTensor(next_state).to(self.device)
        if not isinstance(done, torch.Tensor):
            done = torch.FloatTensor([float(done)]).to(self.device)
        
        # Append to buffers
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
        
        # Add log_prob and value if provided
        if log_prob is not None:
            self.log_probs.append(log_prob)
        if value is not None:
            self.values.append(value)
        
        self.size += 1
    
    def get(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get all transitions in the buffer.
        
        Returns:
            Tuple of (states, actions, rewards, next_states, dones) as tensors
        """
        # Stack all experiences
        states = torch.stack(self.states)
        actions = torch.stack(self.actions)
        rewards = torch.stack(self.rewards)
        next_states = torch.stack(self.next_states)
        dones = torch.stack(self.dones)
        
        return states, actions, rewards, next_states, dones
    
    def get_ppo_data(self):
        """
        Get PPO-specific data from the buffer.
        
        Returns:
            Tuple of (states, actions, log_probs, rewards, next_states, dones, values) as tensors
        """
        # Stack all experiences if available
        if len(self.states) == 0:
            return None, None, None, None, None, None, None
            
        states = torch.stack(self.states)
        actions = torch.stack(self.actions)
        rewards = torch.stack(self.rewards)
        next_states = torch.stack(self.next_states)
        dones = torch.stack(self.dones)
        
        # Stack log_probs and values if they exist
        log_probs = torch.stack(self.log_probs) if self.log_probs else None
        values = torch.stack(self.values) if self.values else None
        
        return states, actions, log_probs, rewards, next_states, dones, values
    
    def reset(self):
        """Reset the buffer."""
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.log_probs = []
        self.values = []
        self.size = 0
    
    def is_full(self) -> bool:
        """
        Check if the buffer is full.
        
        Returns:
            bool: Whether the buffer is full
        """
        return self.size >= self.capacity 