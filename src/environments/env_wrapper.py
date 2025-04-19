#!/usr/bin/env python
"""
Environment wrapper for reinforcement learning.
Handles environment creation, observation normalization, and reward shaping.
"""

import gymnasium as gym
import numpy as np
import torch
from gymnasium.wrappers import RecordVideo, RecordEpisodeStatistics, NormalizeObservation, ClipAction
from gymnasium.wrappers.monitoring.video_recorder import VideoRecorder
from typing import Dict, Tuple, Any, Optional
import math
import time # Import time for unique video naming
import os # Import os for path manipulation

class EnvWrapper(gym.Wrapper):
    """
    Wrapper for Gymnasium environments with additional functionality.
    """
    
    def __init__(
        self,
        env_name="LunarLander-v2",
        seed=None,
        norm_obs=True,
        clip_obs=10.0,
        norm_reward=False,
        clip_reward=10.0,
        gamma=0.99,
        epsilon=1e-8,
        record_video=False,
        video_folder="videos",
        record_freq=100,
        device="cpu",
        # --- Reward Shaping & Strategy Params ---
        reward_strategy="progressive", # "progressive" or "multiplicative"
        # Progressive Params (used if reward_strategy='progressive')
        landing_reward_scale=15.0,
        velocity_penalty_scale=0.2,
        angle_penalty_scale=0.5,
        distance_penalty_scale=0.0,
        fuel_penalty_scale=0.0,
        safe_landing_bonus=300.0,
        crash_penalty_scale=50.0,
        min_flight_steps=30,
        # Multiplicative Params (used if reward_strategy='multiplicative')
        base_landing_reward=1000.0,
        safety_multiplier_type="exp",
        accuracy_multiplier_type="gaussian",
        efficiency_multiplier_type="none",
        timeout_reward=0.0,
        # Shared Thresholds (used by both or within multiplicative functions)
        landing_velocity_threshold=0.6,
        landing_angle_threshold=0.25,
        landing_position_threshold=0.2, # Mostly for multiplicative accuracy
        fuel_efficiency_target=50.0, # Mostly for multiplicative efficiency
        **kwargs
    ):
        """
        Initialize the environment wrapper.
        
        Args:
            env_name (str): Name of the environment to create
            seed (int, optional): Random seed for the environment
            norm_obs (bool): Whether to normalize observations
            clip_obs (float): Observation clipping range if norm_obs is True
            norm_reward (bool): Whether to normalize rewards
            clip_reward (float): Reward clipping range if norm_reward is True
            gamma (float): Discount factor for return normalization
            epsilon (float): Small constant for numerical stability in normalization
            record_video (bool): Whether to record video of episodes
            video_folder (str): Directory to save recorded videos
            record_freq (int): Frequency of video recording
            device (str): Device for PyTorch tensors
            reward_strategy (str): Reward strategy to use ("progressive" or "multiplicative")
            landing_reward_scale (float): Scale factor for landing rewards
            velocity_penalty_scale (float): Scale factor for velocity penalties
            angle_penalty_scale (float): Scale factor for angle penalties
            distance_penalty_scale (float): Scale factor for distance penalties
            fuel_penalty_scale (float): Scale factor for fuel usage penalties
            safe_landing_bonus (float): One-time bonus for safe landing (low velocity, upright)
            crash_penalty_scale (float): Scale factor for crash penalties
            min_flight_steps (int): Minimum steps before landing bonus is applied
            base_landing_reward (float): Base landing reward for multiplicative strategy
            safety_multiplier_type (str): Type of multiplier for safety
            accuracy_multiplier_type (str): Type of multiplier for accuracy
            efficiency_multiplier_type (str): Type of multiplier for efficiency
            timeout_reward (float): Reward for timeout
            landing_velocity_threshold (float): Max velocity for safe landing
            landing_angle_threshold (float): Max angle for safe landing
            landing_position_threshold (float): Threshold for position in multiplicative strategy
            fuel_efficiency_target (float): Target fuel efficiency for multiplicative strategy
        """
        # Create environment
        render_mode = "rgb_array" if record_video else None
        try:
            self.env = gym.make(env_name, render_mode=render_mode, **kwargs)
        except TypeError as e:
            print(f"Warning: Environment {env_name} might not accept all kwargs: {e}. Trying without...")
            # Filter common kwargs if needed, or just try without
            kwargs_filtered = {k:v for k,v in kwargs.items() if k not in ['render_mode']} # Example filter
            self.env = gym.make(env_name, render_mode=render_mode, **kwargs_filtered)
        
        super().__init__(self.env)
        
        # Store the initial seed explicitly
        if seed is None:
            seed = np.random.randint(0, 1e6)
        self._initial_seed = seed # Use _initial_seed
        
        # Seed base RNGs
        np.random.seed(self._initial_seed)
        torch.manual_seed(self._initial_seed)
        # Seed the action space for consistent sampling 
        self.env.action_space.seed(self._initial_seed)
        
        # Get environment info
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.is_discrete = isinstance(self.action_space, gym.spaces.Discrete)
        
        # Get dimensions
        self.state_dim = self.observation_space.shape[0]
        self.action_dim = (
            self.action_space.n if self.is_discrete 
            else self.action_space.shape[0]
        )
        
        # Set up observation normalization
        self.norm_obs = norm_obs
        self.clip_obs = clip_obs
        self.obs_mean = torch.zeros(self.state_dim, dtype=torch.float32).to(device)
        self.obs_var = torch.ones(self.state_dim, dtype=torch.float32).to(device) # Use variance for Welford
        self.obs_count = epsilon # Initialize count with epsilon to avoid division by zero
        
        # Set up device
        self.device = device
        self.obs_mean = self.obs_mean.to(self.device)
        self.obs_var = self.obs_var.to(self.device)
        
        # Reward shaping parameters
        self.landing_reward_scale = landing_reward_scale
        self.velocity_penalty_scale = velocity_penalty_scale
        self.angle_penalty_scale = angle_penalty_scale
        self.distance_penalty_scale = distance_penalty_scale
        self.fuel_penalty_scale = fuel_penalty_scale
        self.safe_landing_bonus = safe_landing_bonus
        self.crash_penalty_scale = crash_penalty_scale
        self.min_flight_steps = min_flight_steps
        self.landing_velocity_threshold = landing_velocity_threshold
        self.landing_angle_threshold = landing_angle_threshold
        
        # Episode tracking
        self.current_step = 0
        self.has_landed = False
        self.landing_pos = None
        self.landing_awarded = False
        
        # Dynamic reward parameters
        self.episode_count = 0
        self.best_reward = float("-inf")
        self.reward_threshold = -200  # Initial threshold for good performance
        
        # Initialize reward tracking
        self.episode_reward = 0
        self.episode_length = 0
        self.total_steps = 0
        
        # Store environment info
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n if isinstance(self.env.action_space, gym.spaces.Discrete) else self.env.action_space.shape[0]
        self.is_discrete = isinstance(self.env.action_space, gym.spaces.Discrete)
        
        # Store configuration
        self.norm_obs = norm_obs
        self.clip_obs = clip_obs
        
        # Apply wrappers
        self.env = RecordEpisodeStatistics(self.env)
        
        if norm_obs:
            self.env = NormalizeObservation(self.env)
            self.env = gym.wrappers.TransformObservation(self.env, lambda obs: np.clip(
                obs, -clip_obs, clip_obs))
        
        # Reward strategy
        self.reward_strategy = reward_strategy
        self.base_landing_reward = base_landing_reward
        self.safety_multiplier_type = safety_multiplier_type
        self.accuracy_multiplier_type = accuracy_multiplier_type
        self.efficiency_multiplier_type = efficiency_multiplier_type
        self.timeout_reward = timeout_reward
        self.landing_position_threshold = landing_position_threshold
        self.fuel_efficiency_target = fuel_efficiency_target
        
        # Reward normalization
        self.norm_reward = norm_reward
        self.clip_reward = clip_reward
        self.ret_mean = torch.zeros(1, dtype=torch.float32).to(device)
        self.ret_var = torch.ones(1, dtype=torch.float32).to(device) # Use variance
        self.ret_count = epsilon
        self.current_return = torch.zeros(1, dtype=torch.float32).to(device)
        self.gamma = gamma
        self.epsilon = epsilon
        
        # Video Recording Setup
        self.record_video = record_video
        self.video_folder = video_folder
        self.record_freq = record_freq
        if self.record_video:
            self._setup_video_recording()
        
    def _setup_video_recording(self):
        """Sets up video recording wrapper."""
        if not os.path.exists(self.video_folder):
            os.makedirs(self.video_folder)
            print(f"Created video directory: {self.video_folder}")
        # Trigger video recording based on episode count
        self.env = RecordVideo(
            self.env,
            video_folder=self.video_folder,
            episode_trigger=lambda ep: ep % self.record_freq == 0,
            name_prefix=f"rl-video-{time.strftime('%Y%m%d-%H%M%S')}" # Unique name
        )
        # Skip setting render_mode as it causes issues with newer gymnasium versions
        # self.env.render_mode = 'rgb_array'
        
    def reset(self, seed=None, options=None):
        """Reset the environment and return the initial state."""
        # Determine the seed for this specific reset call
        # Use the provided seed if available, otherwise derive from the initial seed and episode count
        current_seed = seed if seed is not None else self.episode_count + self._initial_seed
        
        # Reset the underlying environment with the determined seed
        obs, info = self.env.reset(seed=current_seed, options=options) 
        
        # Reset episode tracking
        self.current_step = 0
        self.has_landed = False
        self.landing_pos = None
        self.landing_awarded = False
        
        # Reset internal step counter and fuel tracker
        self._current_step = 0
        self._total_fuel_used = 0.0
        self.current_return = torch.zeros(1, dtype=torch.float32).to(self.device)
        self.episode_count += 1 # Increment episode count *after* using it for seed
        
        if self.norm_obs:
            obs = self._get_normalized_observation(obs)
        
        return obs, info
    
    def step(self, action):
        """
        Take a step in the environment.
        
        Args:
            action: Action to take
            
        Returns:
            state: Next state
            reward: Reward received
            done: Whether the episode is done
            truncated: Whether the episode was truncated
            info: Additional information
        """
        # Convert to numpy array if necessary
        if isinstance(action, torch.Tensor):
            processed_action = action.cpu().numpy()
        else:
            processed_action = action
        
        # Ensure action is in the correct format for the environment
        if self.is_discrete:
            # Handle scalar tensor or numpy array
            processed_action = int(processed_action.item()) if isinstance(processed_action, (torch.Tensor, np.ndarray)) and np.prod(processed_action.shape) == 1 else int(processed_action)
        
        # Take step in environment
        next_state, reward, terminated, truncated, info = self.env.step(processed_action)
        
        # Increment step counter
        self.current_step += 1
        
        # Calculate fuel used this step
        fuel_this_step = 0
        if self.is_discrete and "LunarLander" in self.env.spec.id:
            # Fuel usage constants for LunarLander actions
            if processed_action == 1: fuel_this_step = 0.05 # Orientation thrusters less fuel (adjust as needed)
            if processed_action == 2: fuel_this_step = 0.15 # Main engine more fuel (adjust as needed)
            if processed_action == 3: fuel_this_step = 0.05 # Orientation thrusters less fuel
        self._total_fuel_used += fuel_this_step
        
        # --- Apply Reward Shaping / Terminal Rewards based on Strategy --- 
        original_reward = reward # Store original reward from env
        shaped_reward = 0.0 # Initialize shaping/terminal adjustment
        state = next_state
        x_pos, y_pos, vx, vy, angle, ang_vel, leg1_contact, leg2_contact = state
        
        # --- Apply per-step shaping only for progressive strategy --- 
        if self.reward_strategy == "progressive":
            # ... (progressive shaping penalty calculation remains the same) ...
            shaping_penalty = self._shape_reward(reward, state, processed_action, terminated)
            shaped_reward += shaping_penalty 

        # --- Calculate Terminal Rewards --- 
        if terminated or truncated:
            is_crashed = terminated and reward <= -99 # Crash typically gives -100
            is_landed = bool(leg1_contact and leg2_contact)
            is_timeout = truncated and not (is_crashed or is_landed)
            
            # Define minimum positive reward for any non-crash landing
            min_landing_reward = 10.0 

            if is_crashed:
                terminal_reward_adjustment = -self.crash_penalty_scale 
            elif is_landed and self._current_step >= self.min_flight_steps:
                # Successfully Landed (after min steps)
                final_vel = np.sqrt(vx**2 + vy**2)
                final_angle = abs(angle)
                final_pos_off_center = abs(x_pos)

                if self.reward_strategy == "progressive":
                    # Calculate additive bonus/penalties
                    # Start with the bonus
                    terminal_reward_adjustment = self.safe_landing_bonus
                    # Apply penalties
                    terminal_reward_adjustment -= final_vel * self.velocity_penalty_scale * 10 
                    terminal_reward_adjustment -= final_angle * self.angle_penalty_scale * 10
                    if self.distance_penalty_scale == 0.0:
                         terminal_reward_adjustment -= final_pos_off_center * 5.0 
                    # Ensure at least a minimum positive reward for landing
                    terminal_reward_adjustment = max(min_landing_reward, terminal_reward_adjustment)

                elif self.reward_strategy == "multiplicative":
                    # Calculate multiplicative reward
                    safety_mult_vel = self._calculate_multiplier(final_vel, self.landing_velocity_threshold, self.safety_multiplier_type, lower_is_better=True)
                    safety_mult_ang = self._calculate_multiplier(final_angle, self.landing_angle_threshold, self.safety_multiplier_type, lower_is_better=True)
                    accuracy_mult = self._calculate_multiplier(final_pos_off_center, self.landing_position_threshold, self.accuracy_multiplier_type, lower_is_better=True)
                    efficiency_mult = self._calculate_multiplier(self._total_fuel_used, self.fuel_efficiency_target, self.efficiency_multiplier_type, lower_is_better=True)
                    
                    combined_multiplier = safety_mult_vel * safety_mult_ang * accuracy_mult * efficiency_mult
                    base_reward_val = self.base_landing_reward * combined_multiplier
                    
                    # Ensure at least a minimum positive reward for landing (e.g., 1% of base or flat minimum)
                    # min_mult_reward = max(min_landing_reward, 0.01 * self.base_landing_reward)
                    terminal_reward_adjustment = max(min_landing_reward, base_reward_val)
                else:
                    terminal_reward_adjustment = min_landing_reward # Unknown strategy, give minimum landing reward

            elif is_timeout:
                terminal_reward_adjustment = self.timeout_reward
            else: # Landed too early or other terminal state (not crash/timeout)
                 # If it landed (legs down) but too early, give minimum landing reward
                terminal_reward_adjustment = min_landing_reward if is_landed else 0.0 

            shaped_reward += terminal_reward_adjustment 

        final_reward = original_reward + shaped_reward
        
        # Normalize observations if enabled
        if self.norm_obs:
            next_state = self._get_normalized_observation(next_state)
        
        return next_state, final_reward, terminated, truncated, info
    
    def _shape_reward(self, reward, state, action, done):
        """
        Apply reward shaping based on the current state and action.
        
        For Stage 1+: Extreme focus on safe landing with minimal penalties.
        """
        shaped_reward = reward
        
        # Extract state components
        x, y = state[0], state[1]  # position
        vx, vy = state[2], state[3]  # velocity
        angle = state[4]  # angle
        angular_vel = state[5]  # angular velocity
        left_contact, right_contact = state[6], state[7]  # leg contact points
        
        # Track if we've landed (either leg touching ground)
        if left_contact or right_contact:
            self.has_landed = True
            self.landing_pos = (x, y)
        
        # Phase 1: Give extra reward just for being close to ground
        if y < 0.5:  # When close to ground level
            shaped_reward += 0.5 * (0.5 - y)  # Small bonus for approaching ground
        
        # Apply landing reward scaling
        if abs(x) < 0.2:  # landed on pad
            shaped_reward *= self.landing_reward_scale
        
        # Greatly reduce all penalties when approaching landing
        penalty_reduction = 1.0
        if y < 1.0:  # When approaching landing
            # Progressively reduce penalties as we get closer to the ground
            penalty_reduction = max(0.2, y)
        
        # Penalty for excessive velocity (but greatly reduced)
        velocity_penalty = self.velocity_penalty_scale * penalty_reduction * (vx**2 + vy**2)
        shaped_reward -= velocity_penalty
        
        # Penalty for bad angle (but greatly reduced)
        angle_penalty = self.angle_penalty_scale * penalty_reduction * (angle**2)
        shaped_reward -= angle_penalty
        
        # Minimal penalty for distance from center
        distance_penalty = self.distance_penalty_scale * penalty_reduction * (abs(x))
        shaped_reward -= distance_penalty
        
        # Apply fuel penalty (disabled in Stage 1)
        if self.fuel_penalty_scale > 0 and action > 0:  # Using fuel
            shaped_reward -= self.fuel_penalty_scale
        
        # Special case: One-time bonus for safe landing
        # Only if episode is done, lander is upright, velocity is low, and we haven't awarded it yet
        if (done and self.has_landed and self.current_step >= self.min_flight_steps and
            not self.landing_awarded and 
            abs(angle) < self.landing_angle_threshold and 
            abs(vy) < self.landing_velocity_threshold and
            abs(vx) < self.landing_velocity_threshold):
            shaped_reward += self.safe_landing_bonus
            self.landing_awarded = True
        
        # Partial landing bonus - give some credit for almost landing well
        if done and self.has_landed and not self.landing_awarded:
            # Calculate how close we were to a good landing
            velocity_factor = max(0, 1 - (np.sqrt(vx**2 + vy**2) / (self.landing_velocity_threshold * 2)))
            angle_factor = max(0, 1 - (abs(angle) / (self.landing_angle_threshold * 2)))
            
            # If we were at least 50% close to the thresholds, give partial bonus
            if velocity_factor > 0.5 and angle_factor > 0.5:
                partial_bonus = self.safe_landing_bonus * 0.2 * (velocity_factor + angle_factor) / 2
                shaped_reward += partial_bonus
        
        # Special case: Penalty for crashing (high velocity or bad angle on contact)
        if done and self.has_landed and not self.landing_awarded:
            # Lower penalties when we're close to landing well
            crash_velocity = max(0, abs(vy) - self.landing_velocity_threshold)
            crash_angle = max(0, abs(angle) - self.landing_angle_threshold)
            
            # Reduce crash penalty to avoid excessive negative rewards
            crash_penalty = self.crash_penalty_scale * (
                min(3.0, crash_velocity) + 
                min(3.0, crash_angle)
            )
            shaped_reward -= crash_penalty
        
        return shaped_reward
    
    def _get_normalized_observation(self, obs):
        """Normalize observation using current running stats."""
        obs_tensor = torch.FloatTensor(obs).to(self.device)
        # Var = M2 / (count - 1), Std = sqrt(Var)
        variance = self.obs_var / max(1.0, self.obs_count - 1) # Avoid division by zero if count is 1
        std_dev = torch.sqrt(variance) + self.epsilon
        normalized_obs = (obs_tensor - self.obs_mean) / std_dev
        
        if self.clip_obs:
            normalized_obs = torch.clamp(normalized_obs, -self.clip_obs, self.clip_obs)
        return normalized_obs.cpu().numpy()
    
    def close(self):
        """Close environment."""
        self.env.close()
    
    def render(self):
        """Render environment."""
        return self.env.render()
    
    def get_metrics(self) -> Dict[str, float]:
        """
        Get current episode metrics.
        
        Returns:
            Dictionary of metrics
        """
        return {
            "episode_reward": self.episode_reward,
            "episode_length": self.episode_length,
            "total_steps": self.total_steps
        }
    
    def get_random_action(self):
        """
        Get a random action valid for the environment.
        
        Returns:
            numpy.ndarray or int: Random action
        """
        return self.env.action_space.sample()
    
    def preprocess_state(self, state):
        """
        Preprocess a state to be fed to a neural network.
        
        Args:
            state (numpy.ndarray): State to preprocess
            
        Returns:
            torch.Tensor: Preprocessed state
        """
        return torch.FloatTensor(state).unsqueeze(0).to(self.device)
    
    def record_episode(self, agent, episode_length=1000, filepath=None):
        """
        Record a video of an episode with the given agent.
        
        Args:
            agent: Agent to use for action selection
            episode_length (int): Maximum episode length
            filepath (str, optional): Path to save the video
            
        Returns:
            float: Total episode reward
        """
        # Create recorder if filepath is provided
        if filepath:
            recorder = VideoRecorder(self.env, filepath)
        
        # Reset environment
        state = self.reset()
        total_reward = 0
        done = False
        truncated = False
        
        # Run episode
        for _ in range(episode_length):
            if filepath:
                recorder.capture_frame()
            
            # Get action from agent
            action = agent.select_action(state)
            
            # Take step in environment
            next_state, reward, done, truncated, _ = self.step(action)
            
            # Update state and reward
            state = next_state
            total_reward += reward
            
            # Break if episode is done
            if done or truncated:
                break
        
        # Close recorder
        if filepath:
            recorder.close()
        
        return total_reward

    def _calculate_multiplier(self, value, threshold, m_type, lower_is_better=True):
        """Helper to calculate reward multipliers for multiplicative strategy."""
        if m_type == "none" or m_type is None:
            return 1.0

        # Ensure threshold is positive for relative calculation if needed
        safe_threshold = threshold + 1e-8 if threshold == 0 else threshold

        if lower_is_better:
            excess = max(0, value - threshold)
            relative_excess = excess / abs(safe_threshold)
        else: # Higher is better
            shortfall = max(0, threshold - value)
            relative_excess = shortfall / abs(safe_threshold)
            
        multiplier = 0.0
        if m_type == "linear":
            multiplier = max(0.0, 1.0 - relative_excess) # Linear decay from 1 to 0
        elif m_type == "exp":
            decay_rate = 5.0 # How quickly it decays (tuneable?)
            multiplier = math.exp(-decay_rate * relative_excess)
        elif m_type == "gaussian":
            sigma = 0.5 # Width of the peak (tuneable?)
            multiplier = math.exp(-(relative_excess**2) / (2 * sigma**2))
        elif m_type == "step":
            multiplier = 1.0 if value <= threshold else 0.0 # Step function
        else:
             print(f"Warning: Unknown multiplier type '{m_type}'. Defaulting to 1.0")
             multiplier = 1.0
             
        return max(0.0, min(1.0, multiplier)) 