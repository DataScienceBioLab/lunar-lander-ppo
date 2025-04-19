"""
Proximal Policy Optimization (PPO) agent implementation.
"""

import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
from src.agents.networks import Actor, Critic
from src.utils.buffer import RolloutBuffer

class PPOAgent:
    """
    Proximal Policy Optimization agent.
    """
    def __init__(
        self,
        state_dim,
        action_dim,
        discrete=True,
        device="cpu",
        **kwargs
    ):
        """
        Initialize PPO agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            discrete: Whether action space is discrete
            device: Device to use for tensors
            **kwargs: Additional arguments
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.discrete = discrete
        self.device = device
        
        # Get hyperparameters
        self.lr = kwargs.get("learning_rate", 1e-4)  # Reduced learning rate for more stable learning
        self.gamma = kwargs.get("gamma", 0.99)
        self.gae_lambda = kwargs.get("gae_lambda", 0.95)
        self.clip_param = kwargs.get("clip_param", 0.1)  # Reduced for more conservative updates
        self.value_coef = kwargs.get("value_coef", 0.5)
        self.entropy_coef = kwargs.get("entropy_coef", 0.01)
        self.max_grad_norm = kwargs.get("max_grad_norm", 0.5)
        self.num_epochs = kwargs.get("num_epochs", 20)  # Increased for more training per batch
        self.batch_size = kwargs.get("batch_size", 64)
        self.buffer_size = kwargs.get("buffer_size", 2048)
        self.normalize_advantage = kwargs.get("normalize_advantage", True)
        
        # Create networks with larger capacity
        actor_hidden_sizes = kwargs.get("actor_hidden_sizes", [256, 256])  # Increased network size
        critic_hidden_sizes = kwargs.get("critic_hidden_sizes", [256, 256])
        
        self.actor = Actor(state_dim, action_dim, actor_hidden_sizes).to(device)
        self.critic = Critic(state_dim, critic_hidden_sizes).to(device)
        
        # Create optimizer with weight decay for regularization
        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=self.lr,
            weight_decay=1e-5  # Added weight decay
        )
        
        # Create buffer
        self.buffer = RolloutBuffer(
            capacity=self.buffer_size,
            device=self.device
        )
        
        # Initialize step counters
        self.total_steps = 0
        self.optimization_steps = 0
    
    def select_action(self, state, deterministic=False):
        """
        Select an action given a state.
        
        Args:
            state: Current state
            deterministic: Whether to select action deterministically
            
        Returns:
            Selected action
        """
        # Convert state to tensor if needed and move to correct device
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).to(self.device)
        else:
            state = state.to(self.device)
        
        # Add batch dimension if needed
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        # Get action probabilities
        with torch.no_grad():
            probs = self.actor(state)
            
            # Check for NaN values and fix if needed
            if torch.isnan(probs).any():
                # If we have NaNs, use a uniform distribution instead
                probs = torch.ones_like(probs) / self.action_dim
        
        # Select action
        if deterministic:
            action = torch.argmax(probs, dim=-1)
        else:
            try:
                dist = torch.distributions.Categorical(probs)
                action = dist.sample()
            except Exception as e:
                # Fallback to uniform random action if distribution fails
                print(f"Warning: Distribution sampling failed: {e}. Using uniform random action.")
                action = torch.randint(0, self.action_dim, (1,)).to(self.device)
        
        return action.item()
    
    def update(self, timesteps_per_batch=1000, env=None):
        """
        Collect trajectories and update the agent's policy and value function.
        
        Args:
            timesteps_per_batch: Number of timesteps to collect per batch
            env: Environment to collect trajectories from
            
        Returns:
            dict: Dictionary of update metrics
        """
        if env is not None:
            self._collect_trajectories(env, timesteps_per_batch)
        
        # Skip update if buffer is empty or not enough data
        if len(self.buffer.states) < self.batch_size:
            return None
        
        # Compute returns and advantages
        states, actions, old_log_probs, returns, advantages = self._compute_gae()
        
        # Skip update if there's not enough data (might happen if buffer was reset)
        if len(states) == 0:
            return None
        
        # Prepare for mini-batch updates
        batch_size = states.size(0)
        mini_batch_size = batch_size // 4  # Use smaller mini-batches for better stability
        
        metrics = {}
        
        # Update policy for multiple epochs
        for epoch in range(self.num_epochs):
            # Create mini-batches
            batch_size = min(self.batch_size, states.size(0))
            indices = torch.randperm(states.size(0))
            
            for start_idx in range(0, states.size(0), batch_size):
                # Get mini-batch
                end_idx = min(start_idx + batch_size, states.size(0))
                batch_indices = indices[start_idx:end_idx]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]
                
                # Get current probabilities
                probs = self.actor(batch_states)
                dist = torch.distributions.Categorical(probs)
                log_probs = dist.log_prob(batch_actions.squeeze())
                entropy = dist.entropy().mean()
                
                # Get current value predictions
                values = self.critic(batch_states).squeeze()
                
                # Compute ratio
                ratio = torch.exp(log_probs - batch_old_log_probs.squeeze())
                
                # Compute surrogate losses
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * batch_advantages
                
                # Compute losses
                actor_loss = -torch.min(surr1, surr2).mean()
                value_loss = self.value_coef * F.mse_loss(values, batch_returns.squeeze())
                entropy_loss = -self.entropy_coef * entropy
                
                # Compute total loss
                total_loss = actor_loss + value_loss + entropy_loss
                
                # Optimize
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                # Update metrics
                metrics = {
                    "policy_loss": actor_loss.item(),
                    "value_loss": value_loss.item(),
                    "entropy": entropy.item(),
                    "total_loss": total_loss.item(),
                }
        
        # Clear buffer
        self.buffer.reset()
        
        return metrics
    
    def _collect_trajectories(self, env, timesteps_per_batch):
        """
        Collect trajectories from the environment.
        
        Args:
            env: Environment to collect trajectories from
            timesteps_per_batch: Number of timesteps to collect
        """
        self.buffer.reset()
        
        state, _ = env.reset()
        done = False
        episode_rewards = []
        episode_length = 0
        episode_reward = 0
        
        for _ in range(timesteps_per_batch):
            # Convert state to tensor
            state_tensor = torch.FloatTensor(state).to(self.device)
            
            # Get action probabilities
            with torch.no_grad():
                probs = self.actor(state_tensor.unsqueeze(0))
                dist = torch.distributions.Categorical(probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                value = self.critic(state_tensor.unsqueeze(0)).squeeze()
            
            # Take action in environment
            next_state, reward, done, truncated, info = env.step(action.item())
            done = done or truncated
            
            # Store transition
            self.buffer.add(
                state=state_tensor,
                action=action,
                reward=reward,
                next_state=torch.FloatTensor(next_state).to(self.device),
                done=done,
                log_prob=log_prob,
                value=value
            )
            
            # Update state
            state = next_state
            episode_reward += reward
            episode_length += 1
            
            # If episode is done, reset environment
            if done:
                episode_rewards.append(episode_reward)
                state, _ = env.reset()
                episode_reward = 0
                episode_length = 0
                done = False
        
        # Return metrics about the collected trajectories
        return {
            "mean_reward": np.mean(episode_rewards) if episode_rewards else 0.0,
            "num_episodes": len(episode_rewards),
        }
    
    def _compute_gae(self):
        """
        Compute Generalized Advantage Estimation for the collected experiences.
        
        Returns:
            states: Batch of states
            actions: Batch of actions
            old_log_probs: Batch of log probabilities
            returns: Batch of returns
            advantages: Batch of advantages
        """
        # Get data from buffer
        states, actions, log_probs, rewards, next_states, dones, values = self.buffer.get_ppo_data()
        
        # Check if buffer is empty
        if states is None or len(self.buffer.states) == 0:
            # Return empty tensors if buffer is empty
            empty = torch.tensor([], device=self.device)
            return empty, empty, empty, empty, empty
        
        # Make sure all data is in tensor form
        if not isinstance(rewards, torch.Tensor):
            rewards = torch.cat(rewards) if isinstance(rewards[0], torch.Tensor) else torch.FloatTensor(rewards).to(self.device)
        rewards = rewards.squeeze() # Ensure rewards is 1D [BatchSize]
        
        if not isinstance(dones, torch.Tensor):
            dones = torch.cat(dones) if isinstance(dones[0], torch.Tensor) else torch.FloatTensor(dones).to(self.device)
        dones = dones.squeeze() # Ensure dones is 1D [BatchSize]
        
        if not isinstance(values, torch.Tensor):
            values = torch.cat(values) if isinstance(values[0], torch.Tensor) else torch.FloatTensor(values).to(self.device)
        values = values.squeeze() # Ensure values is 1D [BatchSize]
        
        if not isinstance(log_probs, torch.Tensor):
            log_probs = torch.cat(log_probs) if isinstance(log_probs[0], torch.Tensor) else torch.FloatTensor(log_probs).to(self.device)
        log_probs = log_probs.squeeze() # Ensure log_probs is 1D [BatchSize]
        
        # Compute returns and advantages
        returns = torch.zeros_like(rewards)
        advantages = torch.zeros_like(rewards)
        
        # Get last value
        with torch.no_grad():
            last_value = self.critic(states[-1].unsqueeze(0)).squeeze() if states.size(0) > 0 else torch.zeros((), device=self.device)
        
        # Compute returns and advantages using GAE
        gae = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t]
                next_value = last_value
            else:
                next_non_terminal = 1.0 - dones[t]
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            returns[t] = gae + values[t]
            advantages[t] = gae
        
        # Normalize advantages
        if self.normalize_advantage and advantages.size(0) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return states, actions, log_probs, returns, advantages
    
    def save(self, path):
        """
        Save the agent model.
        
        Args:
            path (str): Path to save the model
        """
        torch.save({
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "discrete": self.discrete,
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "actor_hidden_sizes": self.actor.hidden_sizes,
            "critic_hidden_sizes": self.critic.hidden_sizes,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "total_steps": self.total_steps
        }, path)
    
    @classmethod
    def load(cls, path, device="cpu"):
        """
        Load agent from file.
        
        Args:
            path: Path to load agent from
            device: Device to load agent to
            
        Returns:
            PPOAgent: Loaded agent
        """
        try:
            # First try with weights_only=True (default in PyTorch 2.6+)
            checkpoint = torch.load(path, map_location=device)
        except Exception as e:
            # If that fails, try with weights_only=False (for PyTorch 2.6+)
            print(f"Warning: Error loading with default settings. Trying with weights_only=False: {e}")
            checkpoint = torch.load(path, map_location=device, weights_only=False)
        
        # Create agent
        agent = cls(
            state_dim=checkpoint["state_dim"],
            action_dim=checkpoint["action_dim"],
            discrete=checkpoint["discrete"],
            device=device,
            actor_hidden_sizes=checkpoint["actor_hidden_sizes"],
            critic_hidden_sizes=checkpoint["critic_hidden_sizes"]
        )
        
        # Move networks to device first
        agent.actor = agent.actor.to(device)
        agent.critic = agent.critic.to(device)
        
        # Load state dicts (already on correct device from map_location)
        agent.actor.load_state_dict(checkpoint["actor_state_dict"])
        agent.critic.load_state_dict(checkpoint["critic_state_dict"])
        
        # Try to load optimizer state if compatible
        try:
            # Move optimizer state to device
            optimizer_state = checkpoint["optimizer_state_dict"]
            for state in optimizer_state['state'].values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
            agent.optimizer.load_state_dict(optimizer_state)
        except (ValueError, KeyError) as e:
            print(f"Warning: Could not load optimizer state from {path}: {e}")
        
        agent.total_steps = checkpoint.get("total_steps", 0)
        
        return agent
    
    def train(self):
        """Set agent to training mode."""
        self.actor.train()
        self.critic.train()
    
    def eval(self):
        """Set agent to evaluation mode."""
        self.actor.eval()
        self.critic.eval()
        
    def get_hyperparameters(self):
        """
        Get agent hyperparameters.
        
        Returns:
            dict: Dictionary of hyperparameters
        """
        return {
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "discrete": self.discrete,
            "learning_rate": self.lr,
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
            "clip_param": self.clip_param,
            "value_coef": self.value_coef,
            "entropy_coef": self.entropy_coef,
            "max_grad_norm": self.max_grad_norm,
            "actor_hidden_sizes": self.actor.hidden_sizes,
            "critic_hidden_sizes": self.critic.hidden_sizes,
            "num_epochs": self.num_epochs,
            "batch_size": self.batch_size,
            "normalize_advantage": self.normalize_advantage,
            "total_steps": self.total_steps
        } 