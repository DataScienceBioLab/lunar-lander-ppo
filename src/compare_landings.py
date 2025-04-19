#!/usr/bin/env python
"""
Script to generate comparisons of successful and unsuccessful landings.
"""

import os
import sys
import argparse
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from collections import deque
import gymnasium as gym

# Add the project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.environments.env_wrapper import EnvWrapper
from src.agents.ppo_agent import PPOAgent
from src.utils.config import ENV_CONFIG

def parse_args():
    parser = argparse.ArgumentParser(description="Generate comparison of landings")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model to evaluate")
    parser.add_argument("--num_episodes", type=int, default=100, help="Number of episodes to sample from")
    parser.add_argument("--save_dir", type=str, default="results/landing_comparison", help="Directory to save results")
    parser.add_argument("--seed", type=int, default=789, help="Random seed")
    return parser.parse_args()

def record_episode_data(env, agent, episode_num):
    """Record a single episode and collect data about the landing."""
    state, _ = env.reset()
    done = False
    truncated = False
    total_reward = 0
    steps = 0
    
    # Data collectors
    positions = []
    velocities = []
    angles = []
    
    while not done and not truncated:
        # Get action from agent
        action = agent.select_action(state, deterministic=True)
        
        # Step environment
        next_state, reward, done, truncated, info = env.step(action)
        
        # Update state and collect data
        state = next_state
        total_reward += reward
        steps += 1
        
        # Extract position, velocity, and angle
        x_pos, y_pos = state[0], state[1]
        x_vel, y_vel = state[2], state[3]
        angle = state[4]
        
        # Store data
        positions.append(x_pos)
        velocities.append(np.sqrt(x_vel**2 + y_vel**2))  # Magnitude of velocity
        angles.append(angle)
    
    # Determine if landing was successful
    # In LunarLander-v2, landing is successful if:
    # 1. The lander is between the flags (approximately -0.2 to 0.2)
    # 2. The vertical velocity is small (approximately < 0.5)
    # 3. The angle is close to upright (approximately < 0.2 radians)
    landing_position = state[0]  # x-position
    landing_velocity = np.sqrt(state[2]**2 + state[3]**2)  # Magnitude of velocity
    landing_angle = abs(state[4])  # Absolute angle
    
    success = (
        abs(landing_position) < 0.2 and  # Between flags
        landing_velocity < 0.5 and  # Soft landing
        landing_angle < 0.2  # Upright
    )
    
    # Return collected data
    return {
        'episode_num': episode_num,
        'success': success,
        'reward': total_reward,
        'steps': steps,
        'landing_position': landing_position,
        'landing_velocity': landing_velocity,
        'landing_angle': landing_angle,
        'positions': positions,
        'velocities': velocities,
        'angles': angles
    }

def create_landing_comparison(
    successful_episodes,
    failed_episodes,
    save_dir,
    filename="landing_comparison.png"
):
    """
    Create a static comparison image with plots showing the difference between 
    successful and failed landings.
    
    Args:
        successful_episodes: List of successful episode data
        failed_episodes: List of failed episode data
        save_dir: Directory to save the plot
        filename: Filename to save the plot
    """
    if not successful_episodes or not failed_episodes:
        print("Not enough data to create comparison")
        return
    
    # Define landing thresholds
    landing_velocity_threshold = 0.5  # Approximate threshold based on OpenAI Gym LunarLander
    landing_angle_threshold = 0.2     # Approximate threshold based on OpenAI Gym LunarLander
    
    fig, axs = plt.subplots(3, 2, figsize=(15, 15))
    
    # Plot positions
    ax_success_pos = axs[0, 0]
    ax_failed_pos = axs[0, 1]
    
    # Plot velocities
    ax_success_vel = axs[1, 0]
    ax_failed_vel = axs[1, 1]
    
    # Plot angles
    ax_success_angle = axs[2, 0]
    ax_failed_angle = axs[2, 1]
    
    # Plot successful landings
    for i, episode in enumerate(successful_episodes):
        time_steps = np.arange(len(episode['positions']))
        ax_success_pos.plot(time_steps, episode['positions'], label=f"Ep {episode['episode_num']}")
        ax_success_vel.plot(time_steps, episode['velocities'], label=f"Ep {episode['episode_num']}")
        ax_success_angle.plot(time_steps, episode['angles'], label=f"Ep {episode['episode_num']}")
    
    # Plot failed landings
    for i, episode in enumerate(failed_episodes[:5]):  # Limit to 5 failed episodes for clarity
        time_steps = np.arange(len(episode['positions']))
        ax_failed_pos.plot(time_steps, episode['positions'], label=f"Ep {episode['episode_num']}")
        ax_failed_vel.plot(time_steps, episode['velocities'], label=f"Ep {episode['episode_num']}")
        ax_failed_angle.plot(time_steps, episode['angles'], label=f"Ep {episode['episode_num']}")
    
    # Add thresholds and labels
    ax_success_vel.axhline(y=landing_velocity_threshold, color='green', linestyle='--', 
                          label='Safe Velocity')
    ax_failed_vel.axhline(y=landing_velocity_threshold, color='green', linestyle='--', 
                         label='Safe Velocity')
    
    ax_success_angle.axhline(y=landing_angle_threshold, color='green', linestyle='--', 
                            label='Safe Angle')
    ax_success_angle.axhline(y=-landing_angle_threshold, color='green', linestyle='--')
    ax_failed_angle.axhline(y=landing_angle_threshold, color='green', linestyle='--', 
                           label='Safe Angle')
    ax_failed_angle.axhline(y=-landing_angle_threshold, color='green', linestyle='--')
    
    # Add labels and titles
    ax_success_pos.set_title("Successful Landings - Position")
    ax_failed_pos.set_title("Failed Landings - Position")
    ax_success_vel.set_title("Successful Landings - Velocity")
    ax_failed_vel.set_title("Failed Landings - Velocity")
    ax_success_angle.set_title("Successful Landings - Angle")
    ax_failed_angle.set_title("Failed Landings - Angle")
    
    for ax in axs.flat:
        ax.set_xlabel("Time Steps")
        ax.legend()
    
    axs[0, 0].set_ylabel("Position")
    axs[0, 1].set_ylabel("Position")
    axs[1, 0].set_ylabel("Velocity")
    axs[1, 1].set_ylabel("Velocity")
    axs[2, 0].set_ylabel("Angle")
    axs[2, 1].set_ylabel("Angle")
    
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path)
    plt.close()
    print(f"Saved landing comparison to {save_path}")

def create_landing_summary(episodes, save_dir, filename="landing_summary.png", title="Landing Analysis"):
    """Create a summary visualization of multiple landing episodes."""
    if not episodes:
        print("No episodes to summarize")
        return
    
    # Set up figure
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle(title, fontsize=16)
    
    # Create grid for plots
    gs = fig.add_gridspec(2, 3)
    
    # Position plot
    ax_pos = fig.add_subplot(gs[0, 0])
    ax_vel = fig.add_subplot(gs[0, 1])
    ax_angle = fig.add_subplot(gs[0, 2])
    
    # Metrics summary
    ax_metrics = fig.add_subplot(gs[1, :])
    
    # Plot all episode data
    for i, episode in enumerate(episodes):
        color = plt.cm.viridis(i / len(episodes))
        
        # Get episode data
        timesteps = np.arange(len(episode['positions']))
        
        # Plot position, velocity, angle
        ax_pos.plot(timesteps, episode['positions'], color=color, 
                   label=f"Ep {episode['episode_num']}", alpha=0.7)
        ax_vel.plot(timesteps, episode['velocities'], color=color, 
                   label=f"Ep {episode['episode_num']}", alpha=0.7)
        ax_angle.plot(timesteps, episode['angles'], color=color, 
                     label=f"Ep {episode['episode_num']}", alpha=0.7)
    
    # Add safe thresholds
    landing_velocity_threshold = 0.5  # Approximate threshold
    landing_angle_threshold = 0.2     # Approximate threshold
    
    ax_vel.axhline(y=landing_velocity_threshold, color='green', linestyle='--', 
                  label='Safe Velocity', alpha=0.7)
    
    ax_angle.axhline(y=landing_angle_threshold, color='green', linestyle='--', 
                    label='Safe Angle', alpha=0.7)
    ax_angle.axhline(y=-landing_angle_threshold, color='green', linestyle='--', alpha=0.7)
    
    # Set axis labels
    ax_pos.set_title("Position over Time")
    ax_pos.set_xlabel("Time Steps")
    ax_pos.set_ylabel("Position")
    ax_pos.grid(True, alpha=0.3)
    
    ax_vel.set_title("Velocity over Time")
    ax_vel.set_xlabel("Time Steps")
    ax_vel.set_ylabel("Velocity")
    ax_vel.grid(True, alpha=0.3)
    
    ax_angle.set_title("Angle over Time")
    ax_angle.set_xlabel("Time Steps")
    ax_angle.set_ylabel("Angle (rad)")
    ax_angle.grid(True, alpha=0.3)
    
    # Create a metrics table
    metrics_data = []
    for episode in episodes:
        metrics_data.append([
            f"Ep {episode['episode_num']}",
            f"{episode['reward']:.1f}",
            f"{episode['landing_position']:.2f}",
            f"{episode['landing_velocity']:.2f}",
            f"{episode['landing_angle']:.2f}",
            f"{episode['steps']}"
        ])
    
    column_labels = ['Episode', 'Reward', 'Position', 'Velocity', 'Angle', 'Steps']
    ax_metrics.axis('tight')
    ax_metrics.axis('off')
    table = ax_metrics.table(
        cellText=metrics_data,
        colLabels=column_labels,
        loc='center',
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    ax_metrics.set_title("Landing Metrics")
    
    # Add summary statistics to the plot
    mean_reward = np.mean([episode['reward'] for episode in episodes])
    mean_pos = np.mean([abs(episode['landing_position']) for episode in episodes])
    mean_vel = np.mean([episode['landing_velocity'] for episode in episodes])
    mean_angle = np.mean([episode['landing_angle'] for episode in episodes])
    
    stats_text = (
        f"Mean Reward: {mean_reward:.1f}\n"
        f"Mean |Position|: {mean_pos:.2f}\n"
        f"Mean Velocity: {mean_vel:.2f}\n"
        f"Mean Angle: {mean_angle:.2f}\n"
    )
    
    # Add text for stats
    fig.text(0.02, 0.02, stats_text, fontsize=10)
    
    # Add legends
    handles, labels = ax_pos.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.99, 0.99))
    
    # Save figure
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path)
    plt.close()
    print(f"Saved landing summary to {save_path}")

def main(args):
    """Main entry point for the script."""
    # Load agent model from checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model from {args.model_path}")
    model = PPOAgent.load(args.model_path, device)
    
    # Run episodes and collect data
    print(f"Running {args.num_episodes} episodes to collect data...")
    successful_episodes = []
    failed_episodes = []
    
    env = gym.make("LunarLander-v2", render_mode="rgb_array")
    
    for episode in range(1, args.num_episodes + 1):
        episode_data = record_episode_data(env, model, episode)
        
        if episode_data['success']:
            successful_episodes.append(episode_data)
            print(f"Episode {episode}: SUCCESS - Reward: {episode_data['reward']:.1f}, Position: {episode_data['landing_position']:.2f}, Velocity: {episode_data['landing_velocity']:.2f}, Angle: {episode_data['landing_angle']:.2f}")
        else:
            failed_episodes.append(episode_data)
            print(f"Episode {episode}: FAILED - Reward: {episode_data['reward']:.1f}, Position: {episode_data['landing_position']:.2f}, Velocity: {episode_data['landing_velocity']:.2f}, Angle: {episode_data['landing_angle']:.2f}")
    
    env.close()
    
    print(f"Collected {len(successful_episodes)} successful landings and {len(failed_episodes)} failed landings")
    
    # Create comparison visualizations
    if successful_episodes and failed_episodes:
        create_landing_comparison(
            successful_episodes,
            failed_episodes,
            args.save_dir,
            filename="landing_comparison.png"
        )
        
        # Create summary for all successful landings
        if successful_episodes:
            create_landing_summary(
                successful_episodes, 
                args.save_dir, 
                "successful_landings_summary.png", 
                title="Successful Landings Analysis"
            )
        
        # Create summary for a sample of failed landings
        if failed_episodes:
            create_landing_summary(
                failed_episodes[:min(10, len(failed_episodes))],
                args.save_dir,
                "failed_landings_summary.png",
                title="Failed Landings Analysis"
            )
    else:
        print("Not enough data to create comparisons")

if __name__ == "__main__":
    args = parse_args()
    main(args) 