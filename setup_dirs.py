#!/usr/bin/env python3
"""
Setup script to create necessary directories for the lunar-lander-ppo project.
Run this script after cloning the repository to ensure all required directories exist.
"""

import os
import shutil
from pathlib import Path

# Define the directory structure
DIRS = [
    "src/agents",
    "src/environments",
    "src/utils",
    "models",
    "results",
    "videos",
    "docs/images",
    "data/logs",
]

def create_directory(dir_path):
    """Create directory if it doesn't exist."""
    path = Path(dir_path)
    if not path.exists():
        path.mkdir(parents=True)
        print(f"Created directory: {dir_path}")
    else:
        print(f"Directory already exists: {dir_path}")

def create_gitkeep(dir_path):
    """Create .gitkeep file in directory."""
    gitkeep = Path(dir_path) / ".gitkeep"
    if not gitkeep.exists():
        gitkeep.touch()
        print(f"Created .gitkeep in: {dir_path}")

def setup_directories():
    """Set up all required directories for the project."""
    # Get the project root directory
    root_dir = Path(__file__).parent.absolute()
    
    # Create each directory
    for dir_path in DIRS:
        full_path = root_dir / dir_path
        create_directory(full_path)
        create_gitkeep(full_path)
    
    print("\nDirectory setup complete!")
    print("You can now run 'pip install -e .' to install the package in development mode.")

if __name__ == "__main__":
    setup_directories() 