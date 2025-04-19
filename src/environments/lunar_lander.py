"""
LunarLander environment utilities.
"""

import gymnasium as gym
from src.environments.env_wrapper import EnvWrapper

def create_lunar_lander_env(
    reward_config="standard",
    enforce_soft_landing=False,
    velocity_threshold=0.5,
    enforce_landing_zone=False,
    landing_zone_width=1.0,
    render_mode=None
):
    """Create a LunarLander environment with specified parameters.
    
    Args:
        reward_config: Reward configuration (standard, sparse, very_sparse)
        enforce_soft_landing: Whether to enforce soft landing constraint
        velocity_threshold: Maximum velocity for soft landing
        enforce_landing_zone: Whether to enforce landing zone constraint
        landing_zone_width: Width of landing zone
        render_mode: Render mode (None, 'human', 'rgb_array')
        
    Returns:
        EnvWrapper environment for LunarLander
    """
    # Configure wrapper parameters
    env_kwargs = {
        "env_name": "LunarLander-v2",
        "landing_velocity_threshold": velocity_threshold,
        "landing_position_threshold": landing_zone_width / 2,  # Convert width to threshold
        "render_mode": render_mode
    }
    
    # Map reward_config to appropriate settings
    if reward_config == "sparse":
        env_kwargs["reward_strategy"] = "multiplicative"
        env_kwargs["landing_reward_scale"] = 10.0
        env_kwargs["velocity_penalty_scale"] = 0.1
        env_kwargs["distance_penalty_scale"] = 0.0
    elif reward_config == "very_sparse":
        env_kwargs["reward_strategy"] = "multiplicative"
        env_kwargs["landing_reward_scale"] = 20.0
        env_kwargs["velocity_penalty_scale"] = 0.0
        env_kwargs["distance_penalty_scale"] = 0.0
    else:  # standard
        env_kwargs["reward_strategy"] = "progressive"
        env_kwargs["landing_reward_scale"] = 15.0
        env_kwargs["velocity_penalty_scale"] = 0.2
        env_kwargs["distance_penalty_scale"] = 0.0
    
    # Apply soft landing constraint
    if enforce_soft_landing:
        env_kwargs["landing_velocity_threshold"] = velocity_threshold
        env_kwargs["velocity_penalty_scale"] *= 2.0  # Increase penalty for velocity
    
    # Apply landing zone constraint
    if enforce_landing_zone:
        env_kwargs["landing_position_threshold"] = landing_zone_width / 2
        env_kwargs["distance_penalty_scale"] = 0.2  # Add penalty for distance
    
    # Create a wrapped environment
    wrapped_env = EnvWrapper(**env_kwargs)
    
    return wrapped_env 