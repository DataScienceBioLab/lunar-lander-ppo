"""
Goal-conditioned environment wrapper for LunarLander.

This wrapper converts the LunarLander environment into a goal-conditioned
environment compatible with Hindsight Experience Replay (HER).
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class GoalLunarLanderWrapper(gym.Wrapper):
    """
    Wrapper that converts LunarLander into a goal-conditioned environment.
    
    In this wrapper:
    - The goal is to land at a specific (x, y) location
    - The observation space is augmented with goal information
    - Rewards can be sparse or dense based on distance to goal
    """
    
    def __init__(self, env, sparse_reward=False, distance_threshold=0.3):
        """
        Initialize the wrapper.
        
        Args:
            env: The environment to wrap
            sparse_reward: Whether to use sparse rewards (0 for success, -1 otherwise)
            distance_threshold: Distance threshold for success
        """
        super().__init__(env)
        
        # Original observation space dimensions
        self.orig_obs_dim = env.observation_space.shape[0]
        
        # Goal dimension (x, y coordinates)
        self.goal_dim = 2
        
        # Parameters
        self.sparse_reward = sparse_reward
        self.distance_threshold = distance_threshold
        
        # Define goal bounds (valid landing area on the ground)
        self.goal_low = np.array([-1.0, 0.0])  # Left side of the ground, at ground level
        self.goal_high = np.array([1.0, 0.0])  # Right side of the ground, at ground level
        
        # Define new observation space
        self.observation_space = spaces.Dict({
            "observation": spaces.Box(
                low=env.observation_space.low,
                high=env.observation_space.high,
                dtype=np.float32
            ),
            "achieved_goal": spaces.Box(
                low=np.array([-np.inf, -np.inf]),
                high=np.array([np.inf, np.inf]),
                dtype=np.float32
            ),
            "desired_goal": spaces.Box(
                low=self.goal_low,
                high=self.goal_high,
                dtype=np.float32
            )
        })
        
        # Keep action space the same
        self.action_space = env.action_space
        
        # Cache the current goal
        self.current_goal = None
    
    def reset(self, **kwargs):
        """Reset the environment and sample a new goal."""
        obs, info = self.env.reset(**kwargs)
        
        # Sample a random goal on the ground
        goal = self._sample_goal()
        self.current_goal = goal
        
        # Get the achieved goal (current x, y position)
        achieved_goal = self._get_achieved_goal(obs)
        
        # Construct the new observation
        obs_dict = {
            "observation": obs,
            "achieved_goal": achieved_goal,
            "desired_goal": goal
        }
        
        return obs_dict, info
    
    def step(self, action):
        """Take a step in the environment."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Get the achieved goal
        achieved_goal = self._get_achieved_goal(obs)
        
        # Compute reward based on the achieved goal and desired goal
        goal_reward = self.compute_reward(achieved_goal, self.current_goal, info)
        
        # Get success information
        distance = np.linalg.norm(achieved_goal - self.current_goal)
        is_success = distance < self.distance_threshold
        
        # Add success and distance information to info
        info["is_success"] = float(is_success)
        info["distance"] = distance
        
        # Only terminate based on the original environment
        # We don't want to terminate just because we reached the goal
        
        # Construct the new observation
        obs_dict = {
            "observation": obs,
            "achieved_goal": achieved_goal,
            "desired_goal": self.current_goal
        }
        
        # If we're in sparse reward mode, use the goal reward
        # Otherwise, combine the original reward with goal reward
        if self.sparse_reward:
            reward = goal_reward
        else:
            # Add a scaled goal reward to the original reward
            reward = reward + goal_reward
        
        return obs_dict, reward, terminated, truncated, info
    
    def _sample_goal(self):
        """Sample a random goal in the valid landing area."""
        # Sample a random x position between -1 and 1
        x = np.random.uniform(self.goal_low[0], self.goal_high[0])
        # Fix y to be at ground level (0)
        y = 0.0
        
        return np.array([x, y], dtype=np.float32)
    
    def _get_achieved_goal(self, obs):
        """Extract the achieved goal from the observation."""
        # Extract position (x, y) from the observation
        # In LunarLander, first two dimensions are position
        return obs[:2].copy()
    
    def compute_reward(self, achieved_goal, desired_goal, info=None):
        """Compute the reward for achieving a given goal."""
        # Compute distance to goal
        distance = np.linalg.norm(achieved_goal - desired_goal)
        
        if self.sparse_reward:
            # Sparse reward: 0 for success, -1 otherwise
            return 0.0 if distance < self.distance_threshold else -1.0
        else:
            # Dense reward: negative distance to goal
            # Adding a bonus for being very close to encourage precision
            if distance < self.distance_threshold:
                return 1.0  # Bonus for success
            else:
                return -distance  # Negative distance as reward
    
    def get_normalized_goal(self, goal):
        """Normalize goal to be between -1 and 1."""
        return 2.0 * (goal - self.goal_low) / (self.goal_high - self.goal_low) - 1.0 