#!/usr/bin/env python
"""
Helper utilities for Streamlit applications.
Contains functions to help with common Streamlit warnings and issues.
"""

import logging
import warnings
import streamlit as st
import numpy as np
import torch

# Suppress warning messages
def suppress_warnings():
    """Suppress common warning messages in Streamlit."""
    # Suppress Streamlit's warning for torch module path
    logging.getLogger("streamlit.watcher.local_sources_watcher").setLevel(logging.ERROR)
    
    # Add safe globals for torch serialization
    torch.serialization.add_safe_globals([np.int64, np.float64, np.bool_])
    
    # Suppress PyTorch warnings
    warnings.filterwarnings("ignore", message="You are using `torch.load` with")
    
    # Suppress other common warnings
    warnings.filterwarnings("ignore", message=".*macro_block_size.*")
    warnings.filterwarnings("ignore", message=".*Trying to unpickle.*")

# Custom JSON encoder for NumPy and PyTorch types
class TensorEncoder:
    """Utility class to encode NumPy arrays and PyTorch tensors for JSON."""
    
    @staticmethod
    def encode_tensor(obj):
        """Convert tensor or ndarray to list for JSON serialization."""
        if isinstance(obj, (np.ndarray, np.number)):
            return obj.tolist()
        elif isinstance(obj, (torch.Tensor,)):
            return obj.cpu().numpy().tolist()
        elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

# Set custom theme for visualizations
def set_matplotlib_style():
    """Set consistent matplotlib style for Streamlit apps."""
    import matplotlib.pyplot as plt
    
    plt.style.use('dark_background')
    
    # Custom settings
    plt.rcParams['figure.facecolor'] = '#0E1117'
    plt.rcParams['axes.facecolor'] = '#262730'
    plt.rcParams['axes.labelcolor'] = 'white'
    plt.rcParams['text.color'] = 'white'
    plt.rcParams['axes.edgecolor'] = 'white'
    plt.rcParams['xtick.color'] = 'white'
    plt.rcParams['ytick.color'] = 'white'
    plt.rcParams['grid.color'] = '#4F4F4F'
    plt.rcParams['figure.autolayout'] = True

# Cache torch model loading
@st.cache_resource
def load_model(model_path, model_class):
    """
    Load a PyTorch model with caching to improve performance.
    
    Args:
        model_path: Path to the model file
        model_class: Class to load (e.g., PPOAgent)
        
    Returns:
        Loaded model
    """
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model_class.load(model_path, device=device)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Cache environment creation
@st.cache_resource
def create_environment(env_name, seed=42, render_mode="rgb_array"):
    """
    Create and cache a gym environment to improve performance.
    
    Args:
        env_name: Name of the environment
        seed: Random seed
        render_mode: Rendering mode
        
    Returns:
        Created environment
    """
    from src.environments.env_wrapper import EnvWrapper
    
    try:
        env = EnvWrapper(env_name=env_name, seed=seed, render_mode=render_mode)
        return env
    except Exception as e:
        st.error(f"Error creating environment: {e}")
        return None 