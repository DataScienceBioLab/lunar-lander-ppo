"""
Configuration module for RL experiments.
Contains default hyperparameters and environment settings.
"""

import torch
import os
import yaml
from typing import Dict, Any

# Environment settings
ENV_CONFIG = {
    "lunarlander": {
        "id": "LunarLander-v2",
        "norm_obs": True,
        "clip_obs": 10.0,
        "render_mode": "rgb_array",
        # Stage 1 landing-focused reward parameters - Updated for even more focus on landing
        "landing_reward_scale": 15.0,            # Increased further
        "velocity_penalty_scale": 0.1,           # Greatly reduced
        "angle_penalty_scale": 0.3,              # Greatly reduced
        "distance_penalty_scale": 0.05,          # Greatly reduced
        "fuel_penalty_scale": 0.0,               # Still zero for Stage 1
        "safe_landing_bonus": 500.0,             # Dramatically increased
        "crash_penalty_scale": 50.0,             # Reduced to avoid excessive penalties
        "min_flight_steps": 20,                  # Reduced to allow quicker landings
        "landing_velocity_threshold": 1.0,       # More permissive velocity threshold
        "landing_angle_threshold": 0.6           # More permissive angle threshold
    },
    "cartpole": {  # Fallback environment
        "id": "CartPole-v1",
        "norm_obs": True,
        "clip_obs": 10.0,
        "render_mode": "rgb_array"
    }
}

# PPO hyperparameters
PPO_CONFIG = {
    "learning_rate": 3e-4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_param": 0.2,
    "value_coef": 0.5,
    "entropy_coef": 0.01,
    "max_grad_norm": 0.5,
    "num_epochs": 10,
    "batch_size": 64,
    "buffer_size": 2048,
    "normalize_advantage": True,
    "actor_hidden_sizes": [64, 64],
    "critic_hidden_sizes": [64, 64]
}

# DQN hyperparameters
DQN_CONFIG = {
    "hidden_sizes": [64, 64],
    "learning_rate": 1e-3,
    "gamma": 0.99,
    "buffer_size": 100000,
    "batch_size": 64,
    "target_update_freq": 1000,
    "eps_start": 1.0,
    "eps_end": 0.01,
    "eps_decay": 0.995,
    "double_q": True,
    "prioritized_replay": True,
    "alpha": 0.6,  # Prioritized replay alpha
    "beta": 0.4    # Prioritized replay beta
}

# A2C hyperparameters
A2C_CONFIG = {
    "actor_hidden_sizes": [64, 64],
    "critic_hidden_sizes": [64, 64],
    "learning_rate": 7e-4,
    "gamma": 0.99,
    "entropy_coef": 0.01,
    "value_coef": 0.5,
    "max_grad_norm": 0.5,
    "num_steps": 5
}

# Training settings
TRAINING_CONFIG = {
    "max_episodes": 2000,
    "max_steps": 1000,
    "seed": 42,
    "save_freq": 100,
    "log_freq": 10,
    "eval_freq": 100,
    "eval_episodes": 10,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "use_gpu": torch.cuda.is_available()
}

# Logging settings
LOGGING_CONFIG = {
    "log_dir": "data/logs",
    "models_dir": "models",
    "results_dir": "results"
}

# Dictionary holding different strategy configurations for OVERNIGHT RUN 4
STRATEGY_CONFIGS = {
    
    # --- Run 4: Multiplicative V0 - Safety Only Focus ---
    "run4_mult_v0_b500_sExp_lr1e4_5k": {
        "env_params": {
            "reward_strategy": "multiplicative", 
            "base_landing_reward": 500.0, # Moderate base reward
            "safety_multiplier_type": "exp", # Focus on safety
            "accuracy_multiplier_type": "none", # IGNORE ACCURACY
            "efficiency_multiplier_type": "none",# IGNORE EFFICIENCY
            "crash_penalty_scale": 100.0, 
            "timeout_reward": 0.0,
            "landing_velocity_threshold": 0.6, # Use these in safety multiplier
            "landing_angle_threshold": 0.25,
            # Position/Fuel thresholds not used when multipliers are 'none'
            # "landing_position_threshold": 0.2, 
            # "fuel_efficiency_target": 50.0, 
        },
        "ppo_params": {"learning_rate": 0.0001}, # Slightly higher LR
        "training_params": {"max_episodes": 5000} # 5k episodes
    },
    "run4_mult_v0_b1k_sExp_lr5e5_5k": {
        "env_params": {
            "reward_strategy": "multiplicative", 
            "base_landing_reward": 1000.0, # Higher base reward
            "safety_multiplier_type": "exp", 
            "accuracy_multiplier_type": "none", 
            "efficiency_multiplier_type": "none",
            "crash_penalty_scale": 100.0, 
            "timeout_reward": 0.0,
            "landing_velocity_threshold": 0.6, 
            "landing_angle_threshold": 0.25,
        },
        "ppo_params": {"learning_rate": 0.00005}, # Default LR from successful runs
        "training_params": {"max_episodes": 5000}
    },
    "run4_mult_v0_b500_sLin_lr1e4_5k": { # Test Linear safety
        "env_params": {
            "reward_strategy": "multiplicative", 
            "base_landing_reward": 500.0, 
            "safety_multiplier_type": "linear", # Linear safety
            "accuracy_multiplier_type": "none", 
            "efficiency_multiplier_type": "none",
            "crash_penalty_scale": 100.0, 
            "timeout_reward": 0.0,
            "landing_velocity_threshold": 0.6, 
            "landing_angle_threshold": 0.25,
        },
        "ppo_params": {"learning_rate": 0.0001},
        "training_params": {"max_episodes": 5000}
    },
    "run4_mult_v0_b1k_sLin_lr5e5_5k": { # Test Linear safety
        "env_params": {
            "reward_strategy": "multiplicative", 
            "base_landing_reward": 1000.0, 
            "safety_multiplier_type": "linear", 
            "accuracy_multiplier_type": "none", 
            "efficiency_multiplier_type": "none",
            "crash_penalty_scale": 100.0, 
            "timeout_reward": 0.0,
            "landing_velocity_threshold": 0.6, 
            "landing_angle_threshold": 0.25,
        },
        "ppo_params": {"learning_rate": 0.00005},
        "training_params": {"max_episodes": 5000}
    },
}

def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    Load configuration from a YAML file or return default config.
    
    Args:
        config_path: Path to config file (optional)
        
    Returns:
        Dictionary containing configuration
    """
    default_config = {
        # Environment settings
        "env_name": "LunarLander-v2",
        "max_episodes": 2000,
        "max_steps": 1000,
        "eval_freq": 100,
        "eval_episodes": 10,
        "seed": 42,
        
        # PPO parameters
        "learning_rate": 3e-4,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_param": 0.2,
        "value_coef": 0.5,
        "entropy_coef": 0.01,
        "max_grad_norm": 0.5,
        "num_epochs": 10,
        "batch_size": 64,
        "buffer_size": 2048,
        
        # Network architecture
        "actor_hidden_sizes": [64, 64],
        "critic_hidden_sizes": [64, 64],
        
        # Reward shaping parameters - Updated for Stage 1+: Extreme focus on landing
        "landing_reward_scale": 15.0,            # Increased further
        "velocity_penalty_scale": 0.1,           # Greatly reduced
        "angle_penalty_scale": 0.3,              # Greatly reduced
        "distance_penalty_scale": 0.05,          # Greatly reduced
        "fuel_penalty_scale": 0.0,               # Still zero for Stage 1
        "safe_landing_bonus": 500.0,             # Dramatically increased
        "crash_penalty_scale": 50.0,             # Reduced to avoid excessive penalties
        "min_flight_steps": 20,                  # Reduced to allow quicker landings
        
        # Landing criteria - More permissive for Stage 1
        "landing_velocity_threshold": 1.0,       # More permissive velocity threshold
        "landing_angle_threshold": 0.6,          # More permissive angle threshold
        "landing_position_threshold": 0.3,       # More permissive position threshold
    }
    
    if config_path and os.path.exists(config_path):
        with open(config_path, "r") as f:
            loaded_config = yaml.safe_load(f)
            default_config.update(loaded_config)
    
    return default_config 