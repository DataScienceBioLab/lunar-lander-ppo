#!/bin/bash
# Script to initialize and push lunar-lander-ppo repository to GitHub

# Check if GitHub username is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <github_username>"
    echo "Example: $0 kevmok"
    exit 1
fi

GITHUB_USERNAME=$1
REPO_NAME="lunar-lander-ppo"
GITHUB_REPO="https://github.com/$GITHUB_USERNAME/$REPO_NAME.git"

echo "Setting up Git repository..."

# Initialize git repository
git init

# Add all files
git add .

# Create initial commit
git commit -m "Initial commit of Lunar Lander PPO implementation"

# Set up GitHub as remote
echo "Setting up GitHub remote..."
git remote add origin $GITHUB_REPO

echo ""
echo "Repository is ready to be pushed to GitHub."
echo "Before pushing, make sure to:"
echo "1. Create a new repository named '$REPO_NAME' on GitHub"
echo "2. If GitHub repo already exists, run: git push -u origin main"
echo ""
echo "To create the GitHub repository from command line, you can use GitHub CLI:"
echo "gh repo create $REPO_NAME --public --description \"Proximal Policy Optimization implementation for the LunarLander-v2 environment\""
echo ""
echo "After creating the repository, push with:"
echo "git push -u origin main" 