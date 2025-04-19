"""
Visualization utilities for RL experiments.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from matplotlib.animation import FuncAnimation, FFMpegWriter
import pandas as pd
import seaborn as sns

def plot_training_curves(log_dir, save_dir=None):
    """
    Plot training curves from metrics data.
    
    Args:
        log_dir (str): Directory containing metrics.csv
        save_dir (str, optional): Directory to save plots
        
    Returns:
        dict: Dictionary of figure objects
    """
    # Load metrics
    metrics_file = os.path.join(log_dir, "metrics.csv")
    if not os.path.exists(metrics_file):
        raise FileNotFoundError(f"Metrics file not found: {metrics_file}")
    
    metrics_df = pd.read_csv(metrics_file)
    
    # Create directory to save plots if provided
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    figures = {}
    
    # Plot rewards
    fig, ax = plt.subplots(figsize=(10, 6))
    if "train_rewards" in metrics_df.columns:
        ax.plot(metrics_df["train_rewards"], label="Train")
    if "eval_rewards" in metrics_df.columns:
        ax.plot(metrics_df["eval_rewards"], label="Eval")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title("Episode Rewards")
    ax.legend()
    ax.grid(alpha=0.3)
    
    figures["rewards"] = fig
    if save_dir:
        fig.savefig(os.path.join(save_dir, "rewards.png"))
    
    # Plot episode lengths
    fig, ax = plt.subplots(figsize=(10, 6))
    if "train_lengths" in metrics_df.columns:
        ax.plot(metrics_df["train_lengths"], label="Train")
    if "eval_lengths" in metrics_df.columns:
        ax.plot(metrics_df["eval_lengths"], label="Eval")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Length")
    ax.set_title("Episode Lengths")
    ax.legend()
    ax.grid(alpha=0.3)
    
    figures["lengths"] = fig
    if save_dir:
        fig.savefig(os.path.join(save_dir, "lengths.png"))
    
    # Plot losses if available
    if "train_losses" in metrics_df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(metrics_df["train_losses"])
        ax.set_xlabel("Update Step")
        ax.set_ylabel("Loss")
        ax.set_title("Training Loss")
        ax.grid(alpha=0.3)
        
        figures["loss"] = fig
        if save_dir:
            fig.savefig(os.path.join(save_dir, "loss.png"))
    
    # Plot learning rates if available
    if "learning_rates" in metrics_df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(metrics_df["learning_rates"])
        ax.set_xlabel("Update Step")
        ax.set_ylabel("Learning Rate")
        ax.set_title("Learning Rate Schedule")
        ax.grid(alpha=0.3)
        
        figures["lr"] = fig
        if save_dir:
            fig.savefig(os.path.join(save_dir, "learning_rate.png"))
    
    # Plot combined metrics
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    if "train_rewards" in metrics_df.columns:
        sns.lineplot(x=metrics_df.index, y=metrics_df["train_rewards"], 
                   label="Rewards", ax=axes[0])
    if "train_lengths" in metrics_df.columns:
        sns.lineplot(x=metrics_df.index, y=metrics_df["train_lengths"], 
                   label="Lengths", ax=axes[1])
    
    # Add smoothed curves
    if "train_rewards" in metrics_df.columns:
        window_size = min(25, len(metrics_df) // 10 + 1)
        smoothed_rewards = metrics_df["train_rewards"].rolling(window=window_size, center=True).mean()
        sns.lineplot(x=metrics_df.index, y=smoothed_rewards, 
                   label=f"Rewards (Smoothed, window={window_size})", 
                   ax=axes[0], color="red", alpha=0.7)
    
    axes[0].set_title("Training Progress")
    axes[0].set_ylabel("Reward")
    axes[0].grid(alpha=0.3)
    axes[0].legend()
    
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Episode Length")
    axes[1].grid(alpha=0.3)
    axes[1].legend()
    
    plt.tight_layout()
    
    figures["combined"] = fig
    if save_dir:
        fig.savefig(os.path.join(save_dir, "combined.png"))
    
    return figures

def plot_value_heatmap(agent, env, resolution=20, save_path=None):
    """
    Plot a heatmap of state values for a 2D state space.
    
    Args:
        agent: Agent with a value function
        env: Environment wrapper
        resolution (int): Resolution of the grid
        save_path (str, optional): Path to save the plot
        
    Returns:
        matplotlib.figure.Figure: Figure object
    """
    # Only works for 2D state spaces
    if env.state_dim != 2:
        raise ValueError("Value heatmap only works for 2D state spaces")
    
    # Get state bounds
    if hasattr(env, "observation_space"):
        low = env.observation_space.low[:2]
        high = env.observation_space.high[:2]
    else:
        # Use defaults if bounds not available
        low = np.array([-1, -1])
        high = np.array([1, 1])
    
    # Create grid
    x = np.linspace(low[0], high[0], resolution)
    y = np.linspace(low[1], high[1], resolution)
    X, Y = np.meshgrid(x, y)
    
    # Evaluate value function for each grid point
    values = np.zeros((resolution, resolution))
    for i in range(resolution):
        for j in range(resolution):
            state = np.array([X[i, j], Y[i, j]])
            if hasattr(agent, "critic"):
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
                values[i, j] = agent.critic(state_tensor).item()
            else:
                # If agent doesn't have a critic, use a dummy value
                values[i, j] = 0.0
    
    # Plot heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(values, origin="lower", extent=[low[0], high[0], low[1], high[1]])
    plt.colorbar(im, ax=ax, label="Value")
    ax.set_xlabel("State Dimension 1")
    ax.set_ylabel("State Dimension 2")
    ax.set_title("Value Function Heatmap")
    
    if save_path:
        plt.savefig(save_path)
    
    return fig

def compare_training_curves(log_dirs, labels=None, save_path=None):
    """
    Compare training curves from multiple runs.
    
    Args:
        log_dirs (list): List of log directories
        labels (list, optional): List of labels for each run
        save_path (str, optional): Path to save the plot
        
    Returns:
        matplotlib.figure.Figure: Figure object
    """
    # Create default labels if not provided
    if labels is None:
        labels = [f"Run {i+1}" for i in range(len(log_dirs))]
    
    # Load metrics from each log directory
    dfs = []
    for log_dir in log_dirs:
        metrics_file = os.path.join(log_dir, "metrics.csv")
        if not os.path.exists(metrics_file):
            raise FileNotFoundError(f"Metrics file not found: {metrics_file}")
        df = pd.read_csv(metrics_file)
        dfs.append(df)
    
    # Create plot
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Plot rewards
    for i, (df, label) in enumerate(zip(dfs, labels)):
        if "train_rewards" in df.columns:
            axes[0].plot(df["train_rewards"], label=label, alpha=0.5)
            
            # Add smoothed curve
            window_size = min(25, len(df) // 10 + 1)
            smoothed_rewards = df["train_rewards"].rolling(window=window_size, center=True).mean()
            axes[0].plot(smoothed_rewards, label=f"{label} (Smoothed)", alpha=0.8)
    
    # Plot returns
    for i, (df, label) in enumerate(zip(dfs, labels)):
        if "train_lengths" in df.columns:
            axes[1].plot(df["train_lengths"], label=label, alpha=0.5)
    
    axes[0].set_title("Training Rewards Comparison")
    axes[0].set_ylabel("Reward")
    axes[0].grid(alpha=0.3)
    axes[0].legend()
    
    axes[1].set_title("Episode Lengths Comparison")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Length")
    axes[1].grid(alpha=0.3)
    axes[1].legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    return fig

def create_rollout_animation(env, agent, max_steps=1000, save_path=None):
    """
    Create an animation of an agent's rollout.
    
    Args:
        env: Environment wrapper
        agent: Agent to use for action selection
        max_steps (int): Maximum number of steps to simulate
        save_path (str, optional): Path to save the animation
        
    Returns:
        matplotlib.animation.FuncAnimation: Animation object
    """
    # Reset environment
    state = env.reset()
    frames = []
    
    # Run episode
    done = False
    truncated = False
    step = 0
    
    while not (done or truncated) and step < max_steps:
        # Render frame
        frame = env.render()
        frames.append(frame)
        
        # Select action and step environment
        action = agent.select_action(state)
        state, _, done, truncated, _ = env.step(action)
        
        step += 1
    
    # Create figure and animation
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axis("off")
    
    img = ax.imshow(frames[0])
    
    def init():
        img.set_data(frames[0])
        return (img,)
    
    def animate(i):
        img.set_data(frames[i])
        return (img,)
    
    anim = FuncAnimation(fig, animate, init_func=init, frames=len(frames), 
                        interval=50, blit=True)
    
    if save_path:
        writer = FFMpegWriter(fps=30)
        anim.save(save_path, writer=writer)
    
    return anim 