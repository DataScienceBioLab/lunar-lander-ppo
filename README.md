# Lunar Lander PPO Implementation

This repository contains a Proximal Policy Optimization (PPO) implementation for the LunarLander-v2 environment from Gymnasium. The project focuses on reward function design and optimization for reinforcement learning agents.

![Lunar Lander Visualization](docs/images/landing_comparison.png)

## Project Overview

The lunar landing problem requires an agent to control thrusters to safely land a spacecraft on a designated landing pad. This project explores:

1. **Reward Gaming**: How agents can exploit reward structures, leading to unintended behaviors
2. **Progressive Reward Shaping**: A staged approach to reward design, focusing on one goal at a time
3. **Parameter Optimization**: Systematic grid search of reward parameters to improve landing performance

### Key Findings

- Default reward parameters led to 0% landing success rate despite high rewards
- Agents learned to "game" the reward system by hovering indefinitely
- Increasing landing rewards (10x) while reducing velocity penalties improved success rate to 26.7%
- Progressive reward design approach is more effective than simultaneous optimization

## Repository Structure

```
lunar-lander-ppo/
├── src/
│   ├── agents/                 # Agent implementations
│   │   ├── ppo_agent.py        # PPO agent implementation
│   │   ├── random_agent.py     # Random baseline agent
│   │   └── networks.py         # Neural network architectures
│   ├── environments/           # Environment wrappers
│   │   └── env_wrapper.py      # Customized environment with reward shaping
│   ├── utils/                  # Utility functions
│   │   ├── buffer.py           # Replay buffer implementation
│   │   ├── logger.py           # Logging utilities  
│   │   └── config.py           # Configuration utilities
│   ├── train.py                # Main training script
│   ├── train_grid_search.py    # Grid search for reward parameters
│   ├── evaluate.py             # Evaluation scripts
│   └── compare_landings.py     # Visualize and compare landing behaviors
├── docs/                       # Documentation
│   ├── experiment_summary.md   # Summary of experiments and results
│   ├── reward_gaming_analysis.md # Analysis of reward gaming behavior
│   ├── images/                 # Documentation images
│   └── ...
├── results/                    # Result visualizations and data
├── requirements.txt            # Project dependencies
└── README.md                   # This file
```

## Installation

### System Requirements

Before installing the Python dependencies, you'll need the following system packages:

#### Ubuntu/Debian
```bash
# Install SWIG (required for Box2D)
sudo apt-get update
sudo apt-get install -y swig

# For rendering support
sudo apt-get install -y xvfb python3-opengl ffmpeg
```

#### macOS
```bash
# Using Homebrew
brew update
brew install swig
```

#### Windows
```bash
# Using Chocolatey
choco install swig
```

### Python Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/lunar-lander-ppo.git
cd lunar-lander-ppo
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the package and dependencies:
```bash
pip install -e .
```

This will install all required dependencies including:
- gymnasium[box2d] - With Box2D support for LunarLander
- numpy
- torch
- matplotlib
- pandas
- tqdm
- tensorboard
- PyYAML
- moviepy
- pillow
- scikit-learn

### Troubleshooting

If you encounter issues with Box2D installation:

1. Make sure SWIG is installed and in your PATH
2. You can try installing Box2D dependencies separately:
```bash
pip install box2d-py swig
```

3. For rendering issues, you might need to install additional libraries depending on your system.

#### PyTorch 2.6+ Compatibility

If you're using PyTorch 2.6 or higher, you might see warnings about `weights_only` when loading models. This is due to a change in PyTorch 2.6 that defaults to `weights_only=True`. The code includes workarounds for this issue, but if you encounter problems, you can:

```bash
# Install a specific version of PyTorch compatible with older saved models
pip install torch==2.0.0
```

or update models to be compatible with the new PyTorch loading mechanism.

## Usage

### Training

Train a PPO agent using the main training script:
```bash
python train_agent.py
```

Train with custom reward parameters and training settings:
```bash
python train_agent.py --episodes 3000 --batch_size 4096 --lr 2e-4 --landing_reward_scale 10.0 --velocity_penalty_scale 0.1 --fuel_penalty_scale 0.0
```

Common training parameters:
- `--episodes`: Number of episodes to train (default: 2000)
- `--batch_size`: PPO batch size (default: 2048)
- `--lr`: Learning rate (default: 3e-4)
- `--landing_reward_scale`: Scale landing rewards (default: 10.0)
- `--velocity_penalty_scale`: Scale velocity penalties (default: 0.1)
- `--angle_penalty_scale`: Scale angle penalties (default: 0.2)
- `--distance_penalty_scale`: Scale distance penalties (default: 0.1)
- `--fuel_penalty_scale`: Scale fuel usage penalties (default: 0.0)
- `--eval_freq`: Frequency of evaluation during training (default: 100 episodes)
- `--save_freq`: Frequency of model saving (default: 100 episodes)
- `--render`: Enable environment rendering during training

Run a grid search for optimal parameters:
```bash
python src/train_grid_search.py
```

The training process automatically saves models to the `models` directory and logs to `data/logs` for visualization with TensorBoard.

### Testing/Evaluation

Test a trained model using the test script:
```bash
python test_agent.py --model_path models/your_run/model_best.pt --num_episodes 50
```

Test with custom reward parameters:
```bash
python test_agent.py --model_path models/your_run/model_best.pt --num_episodes 20 --landing_reward_scale 10.0 --velocity_penalty_scale 0.1
```

Record videos of agent performance:
```bash
python test_agent.py --model_path models/your_run/model_best.pt --record_video --video_dir videos/test_run
```

Common testing parameters:
- `--model_path`: Path to the trained model file
- `--num_episodes`: Number of episodes to test (default: 10)
- `--seed`: Random seed for reproducibility (default: 42)
- `--record_video`: Enable video recording of agent performance
- `--video_dir`: Directory to save recorded videos (default: "videos")

If no model path is provided, the test script will automatically locate and use the most recent model.

Generate comparison videos of different landings:
```bash
python src/compare_landings.py --model_path models/best_model.pt --num_episodes 20
```

### Complete Training and Testing Workflow

A typical workflow involves:

1. Train an agent with default or custom parameters:
   ```bash
   python train_agent.py --episodes 2000 --landing_reward_scale 10.0 --velocity_penalty_scale 0.1
   ```

2. Test the trained agent and record performance:
   ```bash
   python test_agent.py --record_video --num_episodes 20
   ```

3. Analyze results and adjust reward parameters as needed
   
4. Retrain with adjusted parameters and compare results

## Reward Function Design

The project explores different reward function designs to promote successful landings:

### Default Reward Structure
- Landing reward: +100 points
- Crash penalty: -100 points
- Engine usage penalty: -0.3 points per frame (main engine)
- Small rewards for position and orientation

### Optimized Reward Structure (Stage 1: Landing Focus)
- Landing reward scale: 10.0 (10x increase)
- Velocity penalty scale: 0.1 (reduced from 0.3)
- Fuel penalty scale: 0.0 (removed completely)
- This configuration achieved a 26.7% landing success rate

## Performance Results

| Reward Configuration | Landing Success Rate | Average Reward |
|----------------------|----------------------|----------------|
| Default              | 0%                   | ~200           |
| Trial 12 (optimized) | 26.7%                | ~230           |
| Trial 19             | ~20%                 | [value]        |
| Trial 10             | ~18%                 | [value]        |

## Future Work

The repository includes plans for continued improvement:

1. **Stage 2 (Landing Precision)**: Add position-based rewards for precise landing
2. **Stage 3 (Efficiency Optimization)**: Reintroduce fuel penalties after mastering landing
3. **Alternative Algorithms**: Implementations of SAC and TD3+HER for comparison

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenAI Gym/Gymnasium for the LunarLander-v2 environment
- PyTorch team for their deep learning framework
- Michigan State University Machine Learning Course 