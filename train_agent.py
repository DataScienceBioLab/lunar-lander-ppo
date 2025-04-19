#!/usr/bin/env python3
"""
Main training script for lunar lander PPO agent.
"""

import argparse
import os
from pathlib import Path
import time

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train a PPO agent for Lunar Lander.")
    
    # Training parameters
    parser.add_argument("--episodes", type=int, default=2000, 
                      help="Number of episodes to train")
    parser.add_argument("--batch_size", type=int, default=2048, 
                      help="PPO batch size")
    parser.add_argument("--lr", type=float, default=3e-4, 
                      help="Learning rate")
    
    # Reward shaping parameters
    parser.add_argument("--landing_reward_scale", type=float, default=10.0,
                      help="Scaling factor for landing rewards")
    parser.add_argument("--velocity_penalty_scale", type=float, default=0.1,
                      help="Scaling factor for velocity penalties")
    parser.add_argument("--angle_penalty_scale", type=float, default=0.2,
                      help="Scaling factor for angle penalties")
    parser.add_argument("--distance_penalty_scale", type=float, default=0.1,
                      help="Scaling factor for distance penalties")
    parser.add_argument("--fuel_penalty_scale", type=float, default=0.0,
                      help="Scaling factor for fuel penalties")
    
    # Logging and checkpointing
    parser.add_argument("--log_dir", type=str, default="data/logs",
                      help="Directory for tensorboard logs")
    parser.add_argument("--checkpoint_dir", type=str, default="models",
                      help="Directory for saving model checkpoints")
    parser.add_argument("--eval_freq", type=int, default=100,
                      help="Frequency of evaluation during training")
    parser.add_argument("--save_freq", type=int, default=100,
                      help="Frequency of model saving")
    
    # Rendering and visualization
    parser.add_argument("--render", action="store_true",
                      help="Render environment during training")
                      
    return parser.parse_args()

def main():
    """Main training function."""
    args = parse_args()
    
    # Create directories if they don't exist
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Import here to avoid circular imports
    from src.train import train
    from src.utils.config import ENV_CONFIG, PPO_CONFIG, TRAINING_CONFIG
    
    # Set up arguments for train function
    train_args = argparse.Namespace()
    train_args.algo = "ppo"
    train_args.env = "lunarlander"
    train_args.max_episodes = args.episodes
    train_args.seed = None
    train_args.no_cuda = False
    train_args.save_dir = args.checkpoint_dir
    train_args.exp_name = f"ppo_lunarlander_{int(time.time())}"
    train_args.verbose = True
    train_args.record_video = args.render
    train_args.reward_strategy = None
    
    # Set up clean environment parameters
    # The train function will extract env_name from args.env
    env_params = {
        "landing_reward_scale": args.landing_reward_scale,
        "velocity_penalty_scale": args.velocity_penalty_scale,
        "angle_penalty_scale": args.angle_penalty_scale,
        "distance_penalty_scale": args.distance_penalty_scale,
        "fuel_penalty_scale": args.fuel_penalty_scale,
    }
    
    # Update args.env to match what we want to use
    train_args.env = "lunarlander"
    
    # Set up agent parameters
    agent_params = PPO_CONFIG.copy()
    agent_params.update({
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
    })
    
    # Set up training parameters
    training_params = TRAINING_CONFIG.copy()
    training_params.update({
        "max_episodes": args.episodes,
        "save_freq": args.save_freq,
        "log_freq": args.eval_freq,
    })
    
    # Set up logging parameters
    logging_params = {
        "log_dir": args.log_dir,
        "models_dir": args.checkpoint_dir,
    }
    
    # Run training
    train(train_args, env_params, agent_params, training_params, logging_params)

if __name__ == "__main__":
    main() 