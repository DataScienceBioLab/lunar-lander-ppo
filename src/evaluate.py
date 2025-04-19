"""
Evaluation script for testing trained RL agents.
"""

import os
import argparse
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.environments.env_wrapper import EnvWrapper
from src.agents.ppo_agent import PPOAgent
from src.utils.config import ENV_CONFIG, TRAINING_CONFIG, LOGGING_CONFIG

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate an RL agent")
    
    # Model
    parser.add_argument("--model", type=str, required=True,
                        help="Path to the model file")
    
    # Environment
    parser.add_argument("--env", type=str, default="lunarlander", 
                        choices=["lunarlander", "cartpole"],
                        help="Environment to use")
    
    # Evaluation
    parser.add_argument("--episodes", type=int, default=10,
                        help="Number of evaluation episodes")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed")
    parser.add_argument("--deterministic", action="store_true",
                        help="Use deterministic actions")
    
    # Output
    parser.add_argument("--render", action="store_true",
                        help="Render environment")
    parser.add_argument("--save_video", action="store_true",
                        help="Save video of evaluation episodes")
    parser.add_argument("--video_dir", type=str, default="videos",
                        help="Directory to save videos")
    
    # GPU
    parser.add_argument("--no_cuda", action="store_true",
                        help="Disable CUDA acceleration")
    
    args = parser.parse_args()
    return args

def evaluate(args):
    """
    Evaluate an agent on the specified environment.
    
    Args:
        args: Command line arguments
        
    Returns:
        dict: Evaluation metrics
    """
    # Set up device
    use_cuda = torch.cuda.is_available() and not args.no_cuda
    device = torch.device("cuda" if use_cuda else "cpu")
    
    # Set up environment
    env_config = ENV_CONFIG[args.env].copy()
    
    # Set random seed
    seed = args.seed if args.seed is not None else TRAINING_CONFIG["seed"]
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    # Create environment
    render_mode = "human" if args.render else "rgb_array" if args.save_video else None
    env = EnvWrapper(
        env_name=env_config["id"],
        seed=seed,
        norm_obs=env_config["norm_obs"],
        clip_obs=env_config["clip_obs"],
        record_video=args.save_video,
        video_folder=args.video_dir,
        record_freq=1,  # Record every episode
        device=device
    )
    
    # Load agent
    agent = PPOAgent.load(args.model, device=device)
    agent.eval()  # Set to evaluation mode
    
    # Run evaluation episodes
    rewards = []
    lengths = []
    
    for episode in tqdm(range(args.episodes), desc="Evaluating"):
        state = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        truncated = False
        
        # Record video if requested
        if args.save_video and episode == 0:
            video_path = os.path.join(args.video_dir, f"eval_episode_{episode}.mp4")
            os.makedirs(os.path.dirname(video_path), exist_ok=True)
            
            # Only record first episode to save space
            record_episode = True
        else:
            record_episode = False
            video_path = None
        
        # Episode loop
        while not (done or truncated):
            # Select action
            action = agent.select_action(state, deterministic=args.deterministic)
            
            # Take step in environment
            next_state, reward, done, truncated, _ = env.step(action)
            
            # Update metrics
            episode_reward += reward
            episode_length += 1
            
            # Update state
            state = next_state
            
            # Render if requested
            if args.render:
                env.render()
                time.sleep(0.02)  # Small delay for visualization
        
        # Log episode results
        rewards.append(episode_reward)
        lengths.append(episode_length)
        
        # Print episode results
        print(f"Episode {episode+1}/{args.episodes}: Reward = {episode_reward:.2f}, Length = {episode_length}")
    
    # Calculate statistics
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    min_reward = np.min(rewards)
    max_reward = np.max(rewards)
    mean_length = np.mean(lengths)
    
    # Print summary statistics
    print("\nEvaluation Summary:")
    print(f"Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"Min/Max Reward: {min_reward:.2f} / {max_reward:.2f}")
    print(f"Mean Episode Length: {mean_length:.2f}")
    
    # Plot reward distribution
    plt.figure(figsize=(10, 6))
    plt.hist(rewards, bins=10, alpha=0.7)
    plt.axvline(mean_reward, color='r', linestyle='--', label=f'Mean: {mean_reward:.2f}')
    plt.xlabel("Reward")
    plt.ylabel("Frequency")
    plt.title("Reward Distribution")
    plt.legend()
    
    # Save plot
    plot_path = os.path.join(LOGGING_CONFIG["results_dir"], "eval_reward_distribution.png")
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path)
    
    # Close environment
    env.close()
    
    return {
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "min_reward": min_reward,
        "max_reward": max_reward,
        "mean_length": mean_length,
        "all_rewards": rewards,
        "all_lengths": lengths
    }

if __name__ == "__main__":
    args = parse_args()
    results = evaluate(args) 