#!/usr/bin/env python3
"""
Script to copy relevant files from the existing project directory to the lunar-lander-ppo structure.
This script copies only the necessary files while excluding personal information and unfinished work.
"""

import os
import shutil
from pathlib import Path
import argparse
import re

def parse_args():
    parser = argparse.ArgumentParser(description="Copy project files to lunar-lander-ppo directory")
    parser.add_argument("--source", type=str, default="/home/southgate/Developement/reinforced",
                        help="Source directory containing the original project")
    parser.add_argument("--dest", type=str, default=".",
                        help="Destination directory (lunar-lander-ppo)")
    return parser.parse_args()

def ensure_dir_exists(path):
    """Ensure directory exists."""
    Path(path).mkdir(parents=True, exist_ok=True)

def copy_file(src, dest):
    """Copy file and report action."""
    if not os.path.exists(src):
        print(f"Warning: Source file does not exist: {src}")
        return False
    
    ensure_dir_exists(os.path.dirname(dest))
    shutil.copy2(src, dest)
    print(f"Copied: {src} -> {dest}")
    return True

def copy_directory(src, dest, exclude_patterns=None):
    """Copy directory contents, excluding patterns."""
    if exclude_patterns is None:
        exclude_patterns = []
        
    if not os.path.exists(src):
        print(f"Warning: Source directory does not exist: {src}")
        return False
    
    ensure_dir_exists(dest)
    
    # Compile regex patterns for exclusion
    exclude_regex = [re.compile(pattern) for pattern in exclude_patterns]
    
    # Walk the source directory
    for root, dirs, files in os.walk(src):
        # Calculate relative path
        rel_path = os.path.relpath(root, src)
        if rel_path == '.':
            rel_path = ''
            
        # Create corresponding directory in destination
        dest_dir = os.path.join(dest, rel_path)
        ensure_dir_exists(dest_dir)
        
        # Copy files
        for file in files:
            src_file = os.path.join(root, file)
            dest_file = os.path.join(dest_dir, file)
            
            # Check exclusion patterns
            skip = False
            for pattern in exclude_regex:
                if pattern.search(src_file):
                    skip = True
                    break
                    
            if not skip:
                copy_file(src_file, dest_file)
    
    return True

def main():
    args = parse_args()
    source_dir = args.source
    dest_dir = args.dest
    
    # Files to copy (source -> destination)
    files_to_copy = {
        # Main source files
        os.path.join(source_dir, "src", "train.py"): 
            os.path.join(dest_dir, "src", "train.py"),
        os.path.join(source_dir, "src", "evaluate.py"): 
            os.path.join(dest_dir, "src", "evaluate.py"),
        os.path.join(source_dir, "src", "train_grid_search.py"): 
            os.path.join(dest_dir, "src", "train_grid_search.py"),
        os.path.join(source_dir, "src", "compare_landings.py"): 
            os.path.join(dest_dir, "src", "compare_landings.py"),
            
        # Documentation
        os.path.join(source_dir, "docs", "experiment_summary.md"): 
            os.path.join(dest_dir, "docs", "experiment_summary.md"),
        os.path.join(source_dir, "docs", "reward_gaming_analysis.md"): 
            os.path.join(dest_dir, "docs", "reward_gaming_analysis.md"),
        os.path.join(source_dir, "docs", "landing_behavior_analysis.md"): 
            os.path.join(dest_dir, "docs", "landing_behavior_analysis.md"),
    }
    
    # Directories to copy recursively (source -> destination, exclude_patterns)
    dirs_to_copy = {
        # Copy agent implementations
        os.path.join(source_dir, "src", "agents"): (
            os.path.join(dest_dir, "src", "agents"),
            ["sac_agent.py", "td3_her.py", "td3_her_agent.py"]  # Exclude unfinished agents
        ),
        
        # Copy environment wrappers
        os.path.join(source_dir, "src", "environments"): (
            os.path.join(dest_dir, "src", "environments"),
            []
        ),
        
        # Copy utilities
        os.path.join(source_dir, "src", "utils"): (
            os.path.join(dest_dir, "src", "utils"),
            []
        ),
        
        # Copy report assets for documentation
        os.path.join(source_dir, "report_assets", "images"): (
            os.path.join(dest_dir, "docs", "images"),
            []
        ),
        
        # Copy selected result visualizations
        os.path.join(source_dir, "results"): (
            os.path.join(dest_dir, "results"),
            ["model_comparison", "landing_analysis"]  # Only include relevant results
        ),
    }
    
    # Copy individual files
    for src, dest in files_to_copy.items():
        copy_file(src, dest)
    
    # Copy directories
    for src, (dest, exclude) in dirs_to_copy.items():
        copy_directory(src, dest, exclude)
    
    # Create __init__.py files to make directories into proper Python packages
    python_package_dirs = [
        os.path.join(dest_dir, "src"),
        os.path.join(dest_dir, "src", "agents"),
        os.path.join(dest_dir, "src", "environments"),
        os.path.join(dest_dir, "src", "utils"),
    ]
    
    for pkg_dir in python_package_dirs:
        init_file = os.path.join(pkg_dir, "__init__.py")
        if not os.path.exists(init_file):
            with open(init_file, 'w') as f:
                pass  # Create empty __init__.py file
            print(f"Created: {init_file}")
    
    print("\nFile copying complete!")
    print("Remember to review the files and remove any remaining personal information")
    print("Next steps:")
    print("1. Run the setup_dirs.py script to ensure all directories exist")
    print("2. Install the package in development mode: pip install -e .")
    print("3. Test that everything works by running: python train_agent.py --episodes 10")

if __name__ == "__main__":
    main() 