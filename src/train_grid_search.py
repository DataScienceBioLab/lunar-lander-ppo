import os
import random
import numpy as np
import torch
from datetime import datetime
from typing import Dict, Any
import argparse
import sys
import time

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.train import train
from src.utils.config import load_config, PPO_CONFIG, ENV_CONFIG
from src.utils.logger import Logger
from src.environments.env_wrapper import EnvWrapper

def sample_hyperparameters() -> Dict[str, Any]:
    """Sample hyperparameters from predefined ranges."""
    return {
        # PPO parameters
        "learning_rate": 10 ** random.uniform(-4.5, -3.5),  # 3e-5 to 3e-4
        "gamma": random.uniform(0.99, 0.999),
        "gae_lambda": random.uniform(0.9, 0.99),
        "clip_param": random.uniform(0.1, 0.3),
        "value_coef": random.uniform(0.3, 0.7),
        "entropy_coef": random.uniform(0.001, 0.02),
        "max_grad_norm": random.uniform(0.3, 0.7),
        "num_epochs": random.randint(10, 30),
        "batch_size": random.choice([32, 64, 128]),
        "buffer_size": random.choice([1024, 2048, 4096]),
        
        # Network architecture
        "actor_hidden_sizes": [random.choice([128, 256, 512]) for _ in range(2)],
        "critic_hidden_sizes": [random.choice([128, 256, 512]) for _ in range(2)],
        
        # Stage 1 Reward shaping parameters
        "landing_reward_scale": random.uniform(8.0, 15.0),  # Heavily incentivize landing on pad
        "velocity_penalty_scale": random.uniform(0.2, 0.5),  # Reduce velocity penalties
        "angle_penalty_scale": random.uniform(0.5, 1.0),     # Moderate angle penalties
        "distance_penalty_scale": random.uniform(0.05, 0.2), # Minimal distance penalties
        "fuel_penalty_scale": 0.0,                           # No fuel penalty for Stage 1
        "safe_landing_bonus": random.uniform(150.0, 250.0),  # Large bonus for safe landing
        "crash_penalty_scale": random.uniform(50.0, 150.0),  # Significant penalty for crashing
        "min_flight_steps": random.randint(20, 40),         # Minimum flight time
        "landing_velocity_threshold": random.uniform(0.4, 0.6),  # Allow higher landing velocity
        "landing_angle_threshold": random.uniform(0.25, 0.35),  # Allow larger landing angle
    }

def run_trial(config: Dict[str, Any], trial_num: int, total_trials: int) -> float:
    """Run a single trial with given hyperparameters."""
    # Create experiment name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"ppo_grid_search_trial{trial_num}_{timestamp}"
    
    # Update config with sampled hyperparameters
    hyperparams = sample_hyperparameters()
    
    # Update PPO config with sampled hyperparameters
    ppo_config = PPO_CONFIG.copy()
    for key in ["learning_rate", "gamma", "gae_lambda", "clip_param", "value_coef", 
                "entropy_coef", "max_grad_norm", "num_epochs", "batch_size", "buffer_size",
                "actor_hidden_sizes", "critic_hidden_sizes"]:
        ppo_config[key] = hyperparams[key]
    
    # Update environment config with sampled hyperparameters
    env_config = ENV_CONFIG["lunarlander"].copy()
    for key in ["landing_reward_scale", "velocity_penalty_scale", "angle_penalty_scale",
                "distance_penalty_scale", "fuel_penalty_scale", "safe_landing_bonus",
                "crash_penalty_scale", "min_flight_steps", "landing_velocity_threshold",
                "landing_angle_threshold"]:
        if key in hyperparams:
            env_config[key] = hyperparams[key]
    
    # Update main config
    config.update({
        "ppo": ppo_config,
        "env": env_config
    })
    
    # Initialize logger
    logger = Logger(exp_name, config)
    logger.info(f"Starting trial {trial_num}/{total_trials}")
    logger.info("Hyperparameters:")
    for key, value in hyperparams.items():
        logger.info(f"  {key}: {value}")
    
    try:
        # Create args namespace
        args = argparse.Namespace(
            env="lunarlander",
            algo="ppo",
            max_episodes=config["max_episodes"],
            max_steps=None,
            seed=config["seed"],
            no_cuda=False,
            save_dir=None,
            exp_name=exp_name,
            verbose=False,
            config=config  # Pass the entire config
        )
        
        # Run training
        results = train(args)
        best_reward = results["best_reward"]
        
        logger.info(f"Trial {trial_num} completed with best reward: {best_reward}")
        return best_reward
        
    except Exception as e:
        logger.error(f"Trial {trial_num} failed: {str(e)}")
        return float("-inf")

def main():
    parser = argparse.ArgumentParser(description="Run stochastic grid search for PPO parameters")
    parser.add_argument("--trials", type=int, default=20, help="Number of trials to run")
    parser.add_argument("--episodes", type=int, default=2000, help="Number of episodes per trial")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Load base config
    config = load_config()
    config["max_episodes"] = args.episodes
    config["seed"] = args.seed
    
    # Run trials
    best_reward = float("-inf")
    best_params = None
    
    for trial in range(1, args.trials + 1):
        reward = run_trial(config, trial, args.trials)
        
        if reward > best_reward:
            best_reward = reward
            best_params = config.copy()
            
            # Save best parameters
            with open("best_params.txt", "w") as f:
                f.write(f"Best reward: {best_reward}\n")
                f.write("Parameters:\n")
                for key, value in best_params.items():
                    f.write(f"{key}: {value}\n")
    
    print(f"\nGrid search completed!")
    print(f"Best reward: {best_reward}")
    print("Best parameters saved to best_params.txt")

if __name__ == "__main__":
    main() 