# Brachiosaurus Environment

Quadrupedal sauropod locomotion and food-reaching environment using MuJoCo and Gymnasium.

## Overview

Brachiosaurus is a massive quadrupedal herbivore, notable for having front legs longer than its rear legs (giraffe-like posture). This environment trains the agent to walk with a coordinated four-legged gait and reach elevated food sources using its long articulated neck.

## Model Specifications

| Property | Value |
|----------|-------|
| Torso height | 2.0m (simulated) |
| Total mass | ~205 kg (simulated) |
| Joints | 28 (1 free + 27 hinge) |
| Actuators | 22 (6 neck + 16 legs) |
| Observation dim | 75 |
| Action dim | 22 |

### Body Structure
- **Torso**: Barrel-shaped body (~200kg total mass)
- **Neck**: 4 articulated segments + head with nasal crest (pitch + yaw control)
- **Front legs**: Longer than rear (shoulder height ~2.2m), 3 joints each (hip pitch/roll, knee, ankle)
- **Rear legs**: Shorter (hip height ~1.8m), same joint structure
- **Tail**: 4 passive segments for counterbalance

### Reward Components
- **Forward velocity** - Movement toward food target
- **Alive bonus** - Survival reward
- **Energy penalty** - Penalizes excessive actuator use
- **Gait stability** - Penalizes excessive torso angular velocity
- **Food reach bonus** - Large reward when head reaches food

## Curriculum Learning

### Stage 1: Balance (500K steps)
Learn to maintain a stable four-legged stance.

### Stage 2: Locomotion (1M steps)
Learn coordinated quadrupedal walking toward a target.

### Stage 3: Food Reach (2M steps)
Walk to food and extend neck to reach elevated food sources.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run environment tests
python -m pytest tests/ -v

# Train stage 1 (balance)
python scripts/train_sb3.py train --stage 1 --timesteps 500000

# View the model (requires display)
python scripts/view_model.py
```

## Files

```
brachiosaurus/
├── assets/
│   └── brachiosaurus.xml      # MuJoCo MJCF model
├── envs/
│   ├── __init__.py
│   └── brachio_env.py          # Gymnasium environment
├── scripts/
│   ├── train_sb3.py            # SB3 PPO training with curriculum
│   ├── test_env.py             # Environment validation script
│   └── view_model.py           # MuJoCo passive viewer
├── tests/
│   ├── conftest.py
│   └── test_brachio_env.py     # Pytest test suite
├── requirements.txt
└── README.md
```
