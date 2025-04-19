"""
Sparse reward version of LunarLander for Hindsight Experience Replay.
"""
import gymnasium as gym
import numpy as np
from gymnasium import spaces

class LunarLanderSparse(gym.Wrapper):
    """
    A wrapper for LunarLander-v2 that provides a sparse reward structure.
    Only gives a non-negative reward when the lander successfully lands.
    
    Also tracks the goal position (landing pad) and achieved position (lander position).
    """
    
    def __init__(self, render_mode=None):
        env = gym.make("LunarLander-v2", render_mode=render_mode)
        super().__init__(env)
        
        # Define the goal dimension (x, y position of landing pad)
        self.goal_dim = 2
        
        # Landing pad is at coordinates (0, 0)
        self.goal = np.array([0.0, 0.0])
        
        # Define thresholds for successful landing
        self.position_threshold = 0.3  # Distance from landing pad
        self.velocity_threshold = 0.5  # Maximum velocity for safe landing
        self.angle_threshold = 0.2     # Maximum angle deviation
        
        # Combine state and goal for the observation space
        state_dim = env.observation_space.shape[0]
        self.state_dim = state_dim
        
        # Observation space includes state and goal
        self.observation_space = spaces.Dict({
            'observation': env.observation_space,
            'desired_goal': spaces.Box(low=-np.inf, high=np.inf, shape=(self.goal_dim,), dtype=np.float32),
            'achieved_goal': spaces.Box(low=-np.inf, high=np.inf, shape=(self.goal_dim,), dtype=np.float32),
        })
        
    def reset(self, **kwargs):
        state, info = self.env.reset(**kwargs)
        
        # Extract achieved goal (x, y position of the lander)
        achieved_goal = np.array([state[0], state[1]])
        
        # Prepare the observation dictionary
        obs = {
            'observation': state,
            'desired_goal': self.goal.copy(),
            'achieved_goal': achieved_goal,
        }
        
        return obs, info
    
    def step(self, action):
        # Take a step in the environment
        next_state, reward, terminated, truncated, info = self.env.reset(**kwargs)
        
        # Extract the lander's position (achieved goal)
        achieved_goal = np.array([next_state[0], next_state[1]])
        
        # Calculate sparse reward
        sparse_reward = self._compute_sparse_reward(next_state, achieved_goal)
        
        # Prepare the observation dictionary
        obs = {
            'observation': next_state,
            'desired_goal': self.goal.copy(),
            'achieved_goal': achieved_goal,
        }
        
        # Store original dense reward in info
        info['original_reward'] = reward
        
        return obs, sparse_reward, terminated, truncated, info
    
    def _compute_sparse_reward(self, state, achieved_goal):
        """
        Compute sparse reward based on landing criteria.
        Returns 0 for successful landing, -1 otherwise.
        """
        x, y = state[0], state[1]
        vx, vy = state[2], state[3]
        angle = state[4]
        
        # Check if landing conditions are met
        position_ok = np.linalg.norm(achieved_goal - self.goal) < self.position_threshold
        velocity_ok = np.sqrt(vx**2 + vy**2) < self.velocity_threshold
        angle_ok = abs(angle) < self.angle_threshold
        
        # Successful landing gets 0 reward (sparse positive), everything else -1
        if position_ok and velocity_ok and angle_ok:
            return 0.0  # Sparse positive reward (successful landing)
        else:
            return -1.0  # Negative reward for all other states
    
    def compute_reward(self, achieved_goal, desired_goal, info):
        """
        External reward calculation for HER.
        This is used by HER to recompute rewards with different goals.
        """
        # Distance between achieved goal and desired goal
        dist = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        
        # Sparse reward: 0 if close enough, -1 otherwise
        return (dist < self.position_threshold).astype(np.float32) * 0.0 - 1.0
    
    def get_state_goal(self, observation):
        """
        Combine state and goal for input to the agent.
        """
        return np.concatenate([observation['observation'], observation['desired_goal']]) 