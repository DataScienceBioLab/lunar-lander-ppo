#!/usr/bin/env python3
"""
Script to evaluate a trained lunar lander PPO agent.
"""

import argparse
import os
import torch
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate a trained PPO agent for Lunar Lander.")
    
    # Model parameters
    parser.add_argument("--model_path", type=str, required=True,
                      help="Path to the trained model file")
    
    # Evaluation parameters
    parser.add_argument("--num_episodes", type=int, default=30,
                      help="Number of episodes to evaluate")
    parser.add_argument("--seed", type=int, default=42,
                      help="Random seed for reproducibility")
    
    # Rendering and visualization
    parser.add_argument("--render", action="store_true",
                      help="Render environment during evaluation")
    parser.add_argument("--save_video", action="store_true",
                      help="Save a video of the agent's performance")
    parser.add_argument("--video_dir", type=str, default="videos",
                      help="Directory to save videos")
    
    # Results
    parser.add_argument("--results_dir", type=str, default="results",
                      help="Directory to save evaluation results")
    
    return parser.parse_args()

def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Create directories if they don't exist
    os.makedirs(args.results_dir, exist_ok=True)
    if args.save_video:
        os.makedirs(args.video_dir, exist_ok=True)
    
    # Import agent and environment
    from src.agents.ppo_agent import PPOAgent
    from src.environments.env_wrapper import EnvWrapper
    
    # Set up environment
    env = EnvWrapper(
        env_name="LunarLander-v2",
        seed=args.seed,
        landing_reward_scale=10.0,
        velocity_penalty_scale=0.1,
        angle_penalty_scale=0.2,
        distance_penalty_scale=0.1,
        fuel_penalty_scale=0.0,
        record_video=args.save_video,
        video_folder=args.video_dir if args.save_video else None
    )
    
    # Get state and action dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # Create agent
    agent = PPOAgent(state_dim, action_dim)
    
    # Load model
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at {args.model_path}")
        return
    
    agent.load(args.model_path)
    print(f"Loaded model from {args.model_path}")
    
    # Evaluate
    rewards = []
    episode_lengths = []
    landing_successes = 0
    
    for i in range(args.num_episodes):
        state, _ = env.reset()
        done = False
        truncated = False
        episode_reward = 0
        episode_length = 0
        
        while not (done or truncated):
            if args.render:
                env.render()
            
            # Select action
            action = agent.select_action(state, evaluate=True)
            
            # Take step in environment
            next_state, reward, done, truncated, info = env.step(action)
            
            # Update state and counters
            state = next_state
            episode_reward += reward
            episode_length += 1
        
        # Record results
        rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        if env.landing_success:
            landing_successes += 1
        
        print(f"Episode {i+1}/{args.num_episodes}: " + 
              f"Reward = {episode_reward:.2f}, " + 
              f"Length = {episode_length}, " + 
              f"Success = {env.landing_success}")
    
    # Calculate statistics
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    success_rate = landing_successes / args.num_episodes * 100
    mean_length = np.mean(episode_lengths)
    
    # Print summary
    print("\nEvaluation Results:")
    print(f"Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"Success Rate: {success_rate:.1f}%")
    print(f"Mean Episode Length: {mean_length:.1f}")
    
    # Save results
    plt.figure(figsize=(10, 6))
    plt.bar(range(args.num_episodes), rewards)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title(f'Evaluation Rewards (Success Rate: {success_rate:.1f}%)')
    plt.tight_layout()
    plt.savefig(os.path.join(args.results_dir, 'evaluation_rewards.png'))
    print(f"Saved results to {args.results_dir}")
    
    if args.save_video:
        print(f"Videos saved to {args.video_dir}")

if __name__ == "__main__":
    main() 