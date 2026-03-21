# Brachiosaurus Environment

Quadrupedal sauropod locomotion and food-reaching environment using MuJoCo and Gymnasium.

## Overview

Brachiosaurus is a massive quadrupedal herbivore, notable for having front legs longer than its rear legs (giraffe-like posture). This environment trains the agent to walk with a coordinated four-legged gait and reach elevated food sources using its long articulated neck.

## Model Specifications

| Property | Value |
|----------|-------|
| Torso height | 2.0m (simulated) |
| Total mass | ~205 kg (simulated) |
| Joints | 32 (1 free + 31 hinge) |
| Actuators | 26 (6 neck + 20 legs) |
| Observation dim | 83 |
| Action dim | 26 |

### Body Structure
- **Torso**: Barrel-shaped body (~200kg total mass)
- **Neck**: 4 articulated segments + head with nasal crest (pitch + yaw control)
- **Front legs**: Longer than rear (shoulder height ~2.2m), 5 joints each (hip pitch/roll, knee, ankle, toe) with semi-digitigrade metacarpal segment
- **Rear legs**: Shorter (hip height ~1.8m), same joint structure with semi-digitigrade metatarsal segment
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
# Install from the repository root
pip install -e ".[all]"

# Run environment tests
python -m pytest environments/brachiosaurus/tests/ -v

# Train stage 1 (balance)
python scripts/train_sb3.py train --stage 1 --timesteps 1000000

# View the model (requires display)
python scripts/view_model.py
```

## Files

```
brachiosaurus/
├── assets/
│   └── brachiosaurus.xml       # MuJoCo MJCF model
├── envs/
│   ├── __init__.py
│   └── brachio_env.py           # Gymnasium environment
├── scripts/
│   ├── train_sb3.py             # SB3 PPO training with curriculum
│   ├── test_env.py              # Environment validation script
│   └── view_model.py            # MuJoCo passive viewer
├── tests/
│   ├── test_brachio_env.py      # Species-specific env tests
│   ├── test_brachio_rewards.py  # Species-specific reward tests
│   └── test_static_balance.py   # Static balance tests
└── README.md
```
