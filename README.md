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

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training

Train a PPO agent with default parameters:
```bash
python src/train.py
```

Train with custom reward parameters:
```bash
python src/train.py --landing_reward_scale 10.0 --velocity_penalty_scale 0.1 --fuel_penalty_scale 0.0
```

Run a grid search for optimal parameters:
```bash
python src/train_grid_search.py
```

### Evaluation

Evaluate a trained model:
```bash
python src/evaluate.py --model_path models/best_model.pt
```

Generate comparison videos of different landings:
```bash
python src/compare_landings.py --model_path models/best_model.pt --num_episodes 20
```

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