#!/usr/bin/env python3
"""
Main training script for lunar lander PPO agent.
"""

import argparse
import os
from pathlib import Path

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
    from src.train import train_ppo
    
    # Run training
    train_ppo(
        episodes=args.episodes,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        landing_reward_scale=args.landing_reward_scale,
        velocity_penalty_scale=args.velocity_penalty_scale,
        angle_penalty_scale=args.angle_penalty_scale,
        distance_penalty_scale=args.distance_penalty_scale,
        fuel_penalty_scale=args.fuel_penalty_scale,
        log_dir=args.log_dir,
        checkpoint_dir=args.checkpoint_dir,
        eval_freq=args.eval_freq,
        save_freq=args.save_freq,
        render=args.render
    )

if __name__ == "__main__":
    main() 