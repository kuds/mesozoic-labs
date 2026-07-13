# Brachiosaurus Environment

Quadrupedal dinosaur-inspired locomotion and target-reaching environment using MuJoCo and Gymnasium.

## Overview

This Brachiosaurus-inspired model has longer front legs than rear legs and a long articulated neck. The environment trains the agent to walk with a coordinated four-legged gait and move its head toward an elevated food target. "Food reach" succeeds when the head tip enters a configured distance threshold around the target; it does not require contact between physical food and head geoms.

## Generated Specifications, Curriculum, and Results

The authoritative public dimensions, current stage budgets, success criterion, and provenance-labelled historical
results are in the generator-managed [Brachiosaurus catalog entry](../../README.md#brachiosaurus). They are derived
from the species manifest, executable environment, compiled MJCF, current TOML stage configs, and result summaries.

## Implementation Notes

### Body Structure
- **Torso**: Simplified barrel-shaped body
- **Neck**: 4 articulated segments + head with nasal crest (pitch + yaw control)
- **Front legs**: Longer than rear (shoulder height ~2.2m), 5 joints each (hip pitch/roll, knee, ankle, toe) with semi-digitigrade metacarpal segment
- **Rear legs**: Shorter (hip height ~1.8m), same joint structure with semi-digitigrade metatarsal segment
- **Tail**: Segmented counterbalance with actuated controls

### Reward Components
- **Forward velocity** - Movement toward food target
- **Alive bonus** - Survival reward
- **Energy penalty** - Penalizes excessive actuator use
- **Gait stability** - Penalizes excessive torso angular velocity
- **Food reach bonus** - Large reward when the head tip enters the configured distance threshold around food

## Quick Start

```bash
# Install from the repository root
pip install -e ".[all]"

# Run environment tests
python -m pytest environments/brachiosaurus/tests/ -v

# Train stage 1 using its current TOML-configured budget
python environments/brachiosaurus/scripts/train_sb3.py train --stage 1

# View the model (requires display)
python environments/brachiosaurus/scripts/view_model.py
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
