# PPO Experiment Summary

## Grid Search Experiments

### Top Performing Models
1. **Trial 12**: Best overall performance
   - Landing success rate: 26.7%
   - Average reward: ~230.0
   - Key parameters:
     - Landing reward scale: 10.0
     - Velocity penalty scale: 0.1
     - Fuel penalty scale: 0.0

2. **Trial 19**: Second best performance
   - Landing success rate: ~20%
   - Key parameters: [TBD]

3. **Trial 10**: Third best performance
   - Landing success rate: ~18%
   - Key parameters: [TBD]

### Key Parameter Insights
- **Landing reward scale**: Values around 10.0 significantly outperformed the default (1.7)
- **Velocity penalty scale**: Reducing from 0.3 to 0.1 encouraged more landing attempts
- **Fuel penalty scale**: Removing fuel penalties (0.0) allowed for more thruster usage during landing

## Reward Structure Evolution

### Original Reward Structure
- Default LunarLander-v2 rewards
- Led to hovering behavior with 0% successful landings

### Stage 1: Landing Focus
- Increased landing reward (10.0)
- Reduced velocity penalties (0.1)
- Removed fuel penalties (0.0)
- Results: ~26.7% successful landing rate

### Stage 2: Landing Precision
- [Not fully implemented in final project]
- Planned approach: Build on successful landing model and add position-based rewards

### Stage 3: Efficiency Optimization
- [Not fully implemented in final project]
- Planned approach: Add back reduced fuel penalties after mastering landing

## Visual Behavior Analysis

### Key Observations
1. **Hover Strategy**
   - Early models learned to hover indefinitely
   - Agent maximized rewards by avoiding penalties without landing

2. **Landing Approach Improvements**
   - Best models learned upright orientation
   - Controlled horizontal velocity
   - Applied main engine to reduce vertical velocity before touchdown

3. **Failure Modes**
   - Excessive horizontal velocity at touchdown
   - Late activation of landing thrusters
   - Poor orientation control

## Hardware and Training Statistics
- Training environment: [Your hardware specs]
- Average training time per trial: [Time]
- Total number of environment steps: [Steps]
- PPO batch size: 2048
- PPO epochs: 10
- Total experiments run: 20 grid search trials + additional Stage 1 focus runs 