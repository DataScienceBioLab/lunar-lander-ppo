#!/usr/bin/env python3
"""
Test script for evaluating a trained lunar lander PPO agent.
"""

import argparse
import numpy as np
import torch
import os
import gymnasium as gym
from datetime import datetime

from src.agents.ppo_agent import PPOAgent
from src.environments.env_wrapper import EnvWrapper

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test a trained PPO agent for Lunar Lander.")
    
    # Model parameters
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to the model to test")
    
    # Testing parameters
    parser.add_argument("--num_episodes", type=int, default=10,
                        help="Number of episodes to test")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    # Environment parameters
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
    
    # Video recording
    parser.add_argument("--record_video", action="store_true",
                      help="Record video of agent's performance")
    parser.add_argument("--video_dir", type=str, default="videos",
                      help="Directory to save videos")
                      
    return parser.parse_args()

def test_agent(model_path, num_episodes, env_params, seed=42, record_video=False, video_dir="videos"):
    """Test a trained agent."""
    # Set random seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        # Try loading with new PyTorch behavior
        agent = PPOAgent.load(model_path, device=device)
    except Exception as e:
        print(f"Warning: Could not load with weights_only=True. Trying with weights_only=False...")
        # Monkey patch the PPOAgent.load method to use weights_only=False
        original_load = PPOAgent.load
        def patched_load(path, device="cpu"):
            checkpoint = torch.load(path, map_location=device, weights_only=False)
            state_dict = checkpoint["model_state_dict"]
            hyperparams = checkpoint.get("hyperparams", {})
            
            # Create a new agent with the hyperparameters
            agent = PPOAgent(
                state_dim=hyperparams.get("state_dim", 8),
                action_dim=hyperparams.get("action_dim", 4),
                discrete=hyperparams.get("discrete", True),
                **{k: v for k, v in hyperparams.items() if k not in ["state_dim", "action_dim", "discrete"]}
            )
            agent.to(device)
            agent.load_state_dict(state_dict)
            agent.eval()
            return agent
        
        # Replace the load method
        PPOAgent.load = patched_load
        agent = PPOAgent.load(model_path, device=device)
    
    agent.eval()  # Set to evaluation mode
    
    # Create environment with video recording capability if requested
    env_params_with_recording = env_params.copy()
    if record_video:
        env_params_with_recording.update({
            "record_video": True,
            "video_folder": video_dir,
            "record_freq": 1  # Record every episode
        })
    
    env = EnvWrapper(
        env_name="LunarLander-v2",
        seed=seed,
        device=device,
        **env_params_with_recording
    )
    
    # Test the agent
    rewards = []
    success_count = 0
    velocity_at_landing = []
    angle_at_landing = []
    landing_position = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            # Select action
            with torch.no_grad():
                action = agent.select_action(state, deterministic=True)
            
            # Take action
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            state = next_state
            
            # Record landing metrics if the episode ends
            if done and info.get("landed", False):
                velocity_at_landing.append(info.get("landing_velocity", 0.0))
                angle_at_landing.append(info.get("landing_angle", 0.0))
                landing_position.append(info.get("landing_position", [0.0, 0.0]))
        
        # Check if landing was successful
        if info.get("landed", False):
            success_count += 1
        
        rewards.append(episode_reward)
        print(f"Episode {episode+1}/{num_episodes}: Reward = {episode_reward:.2f}, Success = {info.get('landed', False)}")
    
    # Print summary
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    success_rate = success_count / num_episodes * 100
    
    print("\n--- Test Results ---")
    print(f"Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"Success Rate: {success_rate:.2f}%")
    print(f"Total Episodes: {num_episodes}")
    
    # Print landing metrics if any landings were successful
    if success_count > 0:
        print("\n--- Landing Metrics ---")
        print(f"Mean Landing Velocity: {np.mean(velocity_at_landing):.2f} ± {np.std(velocity_at_landing):.2f}")
        print(f"Mean Landing Angle: {np.mean(angle_at_landing):.2f} ± {np.std(angle_at_landing):.2f}")
        print(f"Mean Landing Position X: {np.mean([p[0] for p in landing_position]):.2f}")
        print(f"Mean Landing Position Y: {np.mean([p[1] for p in landing_position]):.2f}")
    
    env.close()
    
    if record_video:
        print(f"\nVideos saved to: {video_dir}")
    
    return {
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "success_rate": success_rate,
        "rewards": rewards
    }

def main():
    """Main function."""
    args = parse_args()
    
    # If no model path is provided, try to find the most recent model
    if args.model_path is None:
        model_dirs = [d for d in os.listdir("models") if os.path.isdir(os.path.join("models", d))]
        if not model_dirs:
            print("No model directories found.")
            return
        
        latest_dir = max(model_dirs, key=lambda d: os.path.getctime(os.path.join("models", d)))
        
        model_files = [f for f in os.listdir(os.path.join("models", latest_dir)) 
                      if f.endswith(".pt") and os.path.isfile(os.path.join("models", latest_dir, f))]
        if not model_files:
            print(f"No model files found in {latest_dir}.")
            return
        
        # Prioritize best model if available, otherwise use the final model
        if "model_best.pt" in model_files:
            args.model_path = os.path.join("models", latest_dir, "model_best.pt")
        else:
            args.model_path = os.path.join("models", latest_dir, "model_final.pt")
        
        print(f"Using model: {args.model_path}")
    
    # Set up environment parameters
    env_params = {
        "landing_reward_scale": args.landing_reward_scale,
        "velocity_penalty_scale": args.velocity_penalty_scale,
        "angle_penalty_scale": args.angle_penalty_scale,
        "distance_penalty_scale": args.distance_penalty_scale,
        "fuel_penalty_scale": args.fuel_penalty_scale,
    }
    
    # Test the agent
    test_agent(
        model_path=args.model_path,
        num_episodes=args.num_episodes,
        env_params=env_params,
        seed=args.seed,
        record_video=args.record_video,
        video_dir=args.video_dir
    )

if __name__ == "__main__":
    main() 