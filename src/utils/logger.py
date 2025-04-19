"""
Logger module for tracking training progress and visualizing results.
"""

import os
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import logging
from typing import Dict, Any

class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles NumPy types and torch.device."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, torch.device):
            return str(obj)
        return super(NumpyEncoder, self).default(obj)

class Logger:
    """
    Logger class for tracking experiment metrics and results.
    """
    def __init__(self, exp_name: str, config: Dict[str, Any], log_dir="data/logs"):
        """
        Initialize the logger.
        
        Args:
            exp_name (str): Name of the experiment
            config (dict): Configuration dictionary
            log_dir (str): Directory to save logs
        """
        # Create timestamp and experiment directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_name = f"{exp_name}_{timestamp}"
        self.log_dir = os.path.join(log_dir, self.exp_name)
        
        # Create directories if they don't exist
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Initialize tensorboard writer
        self.writer = SummaryWriter(self.log_dir)
        
        # Save configuration
        with open(os.path.join(self.log_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=4, cls=NumpyEncoder)
        
        # Initialize metrics dictionary
        self.metrics = {
            "step": [],
            "train_rewards": [],
            "train_lengths": [],
            "eval_rewards": [],
            "eval_lengths": [],
            "train_losses": [],
            "learning_rates": [],
            "timestamps": [],
            "rewards": [],
            "lengths": [],
            "policy_loss": [],
            "value_loss": [],
            "entropy": []
        }
        
        # Initialize start time
        self.start_time = time.time()
        self.last_log_time = self.start_time
        
        # Setup logging
        self.logger = logging.getLogger(self.exp_name)
        self.logger.setLevel(logging.INFO)
        
        # File handler
        fh = logging.FileHandler(os.path.join(self.log_dir, "experiment.log"))
        fh.setLevel(logging.INFO)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        # Add handlers
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
        
        # Log initialization
        self.info("Logger initialized. Experiment: " + self.exp_name)
        self.info("Logs will be saved to: " + self.log_dir)
    
    def info(self, msg: str):
        """Log info message."""
        self.logger.info(msg)
    
    def error(self, msg: str):
        """Log error message."""
        self.logger.error(msg)
    
    def log_step(self, step, rewards, lengths, loss=None, value_loss=None, entropy=None, lr=None):
        """
        Log metrics for a single training step/episode.
        
        Args:
            step (int): Current training step or episode number.
            rewards (float): Total reward for the episode.
            lengths (int): Length of the episode.
            loss (float, optional): Policy loss.
            value_loss (float, optional): Value function loss.
            entropy (float, optional): Policy entropy.
            lr (float, optional): Learning rate.
        """
        if self.writer:
            self.writer.add_scalar('Train/Reward', rewards, step)
            self.writer.add_scalar('Train/Episode Length', lengths, step)
            if loss is not None:
                self.writer.add_scalar('Train/Policy Loss', loss, step)
            if value_loss is not None:
                self.writer.add_scalar('Train/Value Loss', value_loss, step)
            if entropy is not None:
                self.writer.add_scalar('Train/Entropy', entropy, step)
            if lr is not None:
                self.writer.add_scalar('Train/Learning Rate', lr, step)
            
            # Flush writer periodically or based on condition
            # self.writer.flush()

        # Store metrics internally if needed for later saving
        # Ensure these lists exist if they weren't initialized for some reason
        if "rewards" not in self.metrics: self.metrics["rewards"] = []
        if "lengths" not in self.metrics: self.metrics["lengths"] = []
        if "policy_loss" not in self.metrics: self.metrics["policy_loss"] = []
        if "value_loss" not in self.metrics: self.metrics["value_loss"] = []
        if "entropy" not in self.metrics: self.metrics["entropy"] = []
        if "learning_rate" not in self.metrics: self.metrics["learning_rate"] = []
            
        self.metrics["rewards"].append(rewards)
        self.metrics["lengths"].append(lengths)
        if loss is not None:
            self.metrics["policy_loss"].append(loss)
        if value_loss is not None:
            self.metrics["value_loss"].append(value_loss)
        if entropy is not None:
            self.metrics["entropy"].append(entropy)
        if lr is not None:
            self.metrics["learning_rate"].append(lr)
        
    def log_evaluation(self, step, mean_reward, std_reward, success_rate=None):
        # Ensure eval lists exist
        if "eval_rewards" not in self.metrics: self.metrics["eval_rewards"] = []
        if "eval_lengths" not in self.metrics: self.metrics["eval_lengths"] = []
        if "eval_success_rate" not in self.metrics: self.metrics["eval_success_rate"] = []
            
        self.metrics["eval_rewards"].append(mean_reward)
        self.metrics["eval_lengths"].append(0) # Placeholder if lengths aren't tracked
        if success_rate is not None:
            self.metrics["eval_success_rate"].append(success_rate)
            
        if self.writer:
            self.writer.add_scalar('Eval/Mean Reward', mean_reward, step)
            self.writer.add_scalar('Eval/Std Reward', std_reward, step)
            if success_rate is not None:
                 self.writer.add_scalar('Eval/Success Rate', success_rate, step)
            # self.writer.flush()
        
    def save_metrics(self):
        """Save collected metrics to a JSON file."""
        # Ensure lists have consistent lengths if some metrics weren't always logged
        max_len = 0
        if self.metrics:
             list_lengths = [len(v) for v in self.metrics.values() if isinstance(v, list)]
             if list_lengths:
                  max_len = max(list_lengths)
                  
        padded_metrics = {}
        for key, value in self.metrics.items():
            if isinstance(value, list):
                # Pad shorter lists with NaN or last value
                if len(value) < max_len:
                    padding = [np.nan] * (max_len - len(value))
                    padded_metrics[key] = value + padding
                else:
                    padded_metrics[key] = value
            else:
                 padded_metrics[key] = value # Keep non-list items as is
                 
        path = os.path.join(self.log_dir, 'training_metrics.json')
        try:
            # Custom encoder to handle NaN for JSON
            class NpEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, np.integer):
                        return int(obj)
                    if isinstance(obj, np.floating):
                        return float(obj)
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    if np.isnan(obj):
                        return None # Represent NaN as null in JSON
                    return super(NpEncoder, self).default(obj)
                    
            with open(path, 'w') as f:
                json.dump(padded_metrics, f, indent=4, cls=NpEncoder)
            print(f"Saved training metrics to {path}")
        except Exception as e:
            print(f"Error saving metrics to {path}: {e}")
            
    def _generate_plots(self):
        """
        Generate plots from metrics.
        """
        # Plot training and evaluation rewards
        plt.figure(figsize=(10, 6))
        if any(r is not None for r in self.metrics["train_rewards"]):
            # Filter out None values
            steps = [s for s, r in zip(self.metrics["step"], self.metrics["train_rewards"]) if r is not None]
            rewards = [r for r in self.metrics["train_rewards"] if r is not None]
            plt.plot(steps, rewards, label="Train Rewards")
        if any(r is not None for r in self.metrics["eval_rewards"]):
            # Filter out None values
            steps = [s for s, r in zip(self.metrics["step"], self.metrics["eval_rewards"]) if r is not None]
            rewards = [r for r in self.metrics["eval_rewards"] if r is not None]
            plt.plot(steps, rewards, label="Eval Rewards")
        plt.xlabel("Episodes")
        plt.ylabel("Reward")
        plt.legend()
        plt.title("Training and Evaluation Rewards")
        plt.savefig(os.path.join(self.log_dir, "rewards.png"))
        plt.close()
        
        # Plot training loss if available
        if any(l is not None for l in self.metrics["train_losses"]):
            plt.figure(figsize=(10, 6))
            # Filter out None values
            steps = [s for s, l in zip(self.metrics["step"], self.metrics["train_losses"]) if l is not None]
            losses = [l for l in self.metrics["train_losses"] if l is not None]
            plt.plot(steps, losses)
            plt.xlabel("Steps")
            plt.ylabel("Loss")
            plt.title("Training Loss")
            plt.savefig(os.path.join(self.log_dir, "loss.png"))
            plt.close()
    
    def close(self):
        """
        Close the logger, save metrics, and generate final plots.
        """
        self.save_metrics()
        self.writer.close()
        
        total_time = time.time() - self.start_time
        print(f"Experiment completed in {total_time:.2f} seconds")
        print(f"Logs saved to: {self.log_dir}")
        
        return self.log_dir 