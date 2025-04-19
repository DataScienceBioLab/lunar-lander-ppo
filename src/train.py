"""
Training script for reinforcement learning agents.
"""

import os
import sys
import argparse
import time
import numpy as np
import torch
import gymnasium as gym
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add the current directory to the Python path
sys.path.insert(0, os.path.abspath('.'))

from src.environments.env_wrapper import EnvWrapper
from src.agents.ppo_agent import PPOAgent
from src.agents.ppo_agent_wrapper import PPOAgentWrapper
from src.agents.random_agent import RandomAgent
from src.utils.logger import Logger
from src.utils.config import (
    ENV_CONFIG, PPO_CONFIG, TRAINING_CONFIG, LOGGING_CONFIG,
    STRATEGY_CONFIGS
)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train an RL agent")
    
    # Environment
    parser.add_argument("--env", type=str, default="lunarlander", 
                        choices=["lunarlander", "cartpole"],
                        help="Environment to use")
    
    # Algorithm
    parser.add_argument("--algo", type=str, default="ppo", 
                        choices=["ppo", "dqn", "a2c", "random"],
                        help="Algorithm to use")
    
    # Training
    parser.add_argument("--max_episodes", type=int, default=None,
                        help="Maximum number of episodes")
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Maximum number of steps")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed")
    
    # GPU
    parser.add_argument("--no_cuda", action="store_true",
                        help="Disable CUDA acceleration")
    
    # Saving
    parser.add_argument("--save_dir", type=str, default=None,
                        help="Directory to save model and results")
    parser.add_argument("--exp_name", type=str, default=None,
                        help="Experiment name")
    
    # Logging
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose output")
    parser.add_argument("--record_video", action="store_true",
                        help="Record video of agent performance")
    
    parser.add_argument("--reward_strategy", type=str, default=None,
                        help="Name of the reward strategy from config.py to use (e.g., 'progressive_v3')")
    
    args = parser.parse_args()
    return args

def train(args, env_params, agent_params, training_params, logging_params):
    """
    Train an agent on the specified environment using provided params.
    
    Args:
        args: Command line arguments (for algo, seed, device, exp_name etc.)
        env_params: Dictionary of environment parameters.
        agent_params: Dictionary of agent hyperparameters.
        training_params: Dictionary of training parameters.
        logging_params: Dictionary of logging parameters.
    """
    # Set up device
    use_cuda = torch.cuda.is_available() and not args.no_cuda
    device = torch.device("cuda" if use_cuda else "cpu")
    
    # Set random seed
    seed = args.seed if args.seed is not None else training_params.get("seed", 42)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    print(f"Using seed: {seed}")
    
    # Create environment using provided parameters
    print("Creating environment with params:", env_params)
    env = EnvWrapper(
        env_name=env_params.get("id", ENV_CONFIG.get(args.env, {}).get("id")),
        seed=seed,
        device=device,
        **env_params # Pass all env params from the strategy config
    )
    
    # Create agent using provided parameters
    print("Creating agent with params:", agent_params)
    if args.algo == "ppo":
        # Combine default PPO config with strategy-specific overrides
        final_agent_params = PPO_CONFIG.copy()
        final_agent_params.update(agent_params)
        agent = PPOAgentWrapper(
            state_dim=env.state_dim,
            action_dim=env.action_dim,
            discrete=env.is_discrete,
            device=device,
            **final_agent_params
        )
    elif args.algo == "random":
        agent = RandomAgent(
            action_dim=env.action_dim,
            discrete=env.is_discrete
        )
    else:
        raise NotImplementedError(f"Algorithm {args.algo} not implemented yet")
    
    # Create logger
    # Use specific log/model dirs for overnight runs
    log_dir = logging_params.get("log_dir", "data/logs")
    models_dir = logging_params.get("models_dir", "models")
    exp_name = args.exp_name or f"{args.algo}_{args.env}_{args.reward_strategy or 'default'}_s{seed}"
    
    logger = Logger(
        exp_name=exp_name,
        config={
            "environment": env_params,
            "agent": agent.get_hyperparameters() if hasattr(agent, "get_hyperparameters") else {"type": args.algo},
            "training": training_params,
            "cli_args": vars(args)
        },
        log_dir=log_dir
    )
    print(f"Logging to: {logger.log_dir}")
    
    # --- Training loop (mostly unchanged, uses training_params) ---
    episode = 0
    total_steps = 0
    best_reward = -float("inf")
    max_episodes = training_params.get("max_episodes", 2000)
    log_freq = training_params.get("log_freq", 10)
    save_freq = training_params.get("save_freq", 100)
    save_dir = os.path.join(models_dir, exp_name)
    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving models to: {save_dir}")

    rewards_history = [] # Use a more descriptive name
    lengths_history = []
    
    pbar = tqdm(total=max_episodes, desc=f"Training {exp_name}")
    start_time = time.time()
    
    while episode < max_episodes:
        state, _ = env.reset() # Use obs, info = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        truncated = False
        loss_info = {}
        
        while not (done or truncated):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated # Combine termination flags
            
            update_info = agent.update(state, action, reward, next_state, done)
            
            episode_reward += reward
            episode_length += 1
            total_steps += 1
            state = next_state
            
            if update_info:
                loss_info = update_info
            
            # Check for max total steps if defined
            # max_total_steps = training_params.get("max_total_steps")
            # if max_total_steps and total_steps >= max_total_steps:
            #     break 
        
        rewards_history.append(episode_reward)
        lengths_history.append(episode_length)
        
        logger.log_step(
            step=episode,
            rewards=episode_reward,
            lengths=episode_length,
            loss=loss_info.get("actor_loss"), # Log specific losses if needed
            value_loss=loss_info.get("value_loss"),
            entropy=loss_info.get("entropy"),
            # Use agent_params which includes potential overrides
            lr=agent_params.get("learning_rate", PPO_CONFIG.get("learning_rate")) 
        )
        
        # Save best model
        current_avg_reward = np.mean(rewards_history[-10:]) # Evaluate based on recent average
        if episode > 10 and current_avg_reward > best_reward and hasattr(agent, "save"):
            print(f"\nNew best reward: {current_avg_reward:.2f} (previously {best_reward:.2f}) at episode {episode}. Saving best model...")
            best_reward = current_avg_reward
            best_model_path = os.path.join(save_dir, f"model_best.pt")
            agent.save(best_model_path)
        
        # Save model periodically
        if episode % save_freq == 0 and episode > 0 and hasattr(agent, "save"):
            print(f"\nSaving checkpoint model at episode {episode}...")
            ckpt_model_path = os.path.join(save_dir, f"model_{episode}.pt")
            agent.save(ckpt_model_path)
        
        pbar.update(1)
        if args.verbose or episode % log_freq == 0:
            pbar.set_postfix({
                "reward": f"{episode_reward:.2f}",
                "avg_rew (10ep)": f"{current_avg_reward:.2f}",
                "length": episode_length,
                "steps": total_steps
            })
            
        episode += 1
        # if max_total_steps and total_steps >= max_total_steps:
        #     break

    pbar.close()
    training_time = time.time() - start_time
    print(f"Training completed for {exp_name} in {training_time:.2f} seconds")
    print(f"Total episodes: {episode}, Total steps: {total_steps}")
    print(f"Best average reward (10 ep): {best_reward:.2f}")
    
    # Save final model
    final_model_path = None
    if hasattr(agent, "save"):
        final_model_path = os.path.join(save_dir, f"model_final.pt")
        agent.save(final_model_path)
        print(f"Saved final model to {final_model_path}")

    logger.save_metrics() 
    logger.close()
    env.close()
    
    # Plotting (Consider moving this to a separate analysis script)
    # ... (plotting code remains similar, using rewards_history, lengths_history) ...
    
    return {
        "best_avg_reward": best_reward,
        "final_model_path": final_model_path,
        "training_time": training_time,
        "exp_name": exp_name
    }

def main():
    args = parse_args()
    
    # Determine which configuration to use
    if args.reward_strategy and args.reward_strategy in STRATEGY_CONFIGS:
        print(f"Using strategy: {args.reward_strategy}")
        strategy_config = STRATEGY_CONFIGS[args.reward_strategy]
        env_params = strategy_config.get("env_params", {})
        agent_params = strategy_config.get("ppo_params", {}) # Assuming PPO for now
        training_params = strategy_config.get("training_params", {})
    else:
        # Fallback to default configuration if no strategy or invalid strategy specified
        print("Warning: No valid reward strategy specified via --reward_strategy. Using default config.")
        env_params = ENV_CONFIG.get(args.env, {})
        agent_params = PPO_CONFIG.copy()
        training_params = TRAINING_CONFIG.copy()

    # Override specific training parameters from command line if provided
    if args.max_episodes:
        training_params["max_episodes"] = args.max_episodes
    if args.seed:
        training_params["seed"] = args.seed # Allow overriding strategy seed
        
    # Define specific logging/saving directories for these runs
    logging_params = {
        "log_dir": "data/overnight_logs",
        "models_dir": "overnight_models"
    }
        
    # --- Pass loaded parameters to the train function --- 
    results = train(args, env_params, agent_params, training_params, logging_params)
    
    print("\n--- Training Run Finished --- ")
    print(f"Experiment: {results.get('exp_name')}")
    print(f"Best Average Reward: {results.get('best_avg_reward'):.2f}")
    print(f"Training Time: {results.get('training_time'):.2f}s")
    print(f"Final model saved to: {results.get('final_model_path')}")
    print("----------------------------\n")

if __name__ == "__main__":
    main()