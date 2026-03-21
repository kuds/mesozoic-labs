# Tyrannosaurus Rex Environment

A large bipedal predator locomotion and bite-attack environment built with MuJoCo and Gymnasium.

## Overview

Tyrannosaurus Rex is a massive bipedal carnivore with a huge skull, powerful jaws, and vestigial forelimbs. This environment trains the agent to sprint toward prey and deliver a bite with its head, using its heavy tail as a counterbalance.

## Model Specifications

| Property | Value |
|----------|-------|
| Pelvis height | ~0.9m (simulated) |
| Joints | 28 (1 free + 25 hinge + 2 ball) |
| Actuators | 21 (3 neck/head + 14 legs + 4 tail) |
| Observation dim | 83 |
| Action dim | 21 |

### Body Structure
- **Head/Neck**: Massive skull with bite geom, 3 actuated joints (neck pitch/yaw, head pitch)
- **Legs**: Powerful digitigrade legs with 7 joints each (hip pitch/roll, knee, ankle, toe d2/d3/d4)
- **Tail**: 5 segments, 4 actuated (pitch 1, yaw 1, pitch 2, pitch 3), heavy counterbalance to skull

### Reward Components
- **Forward velocity** - Movement toward prey target
- **Alive bonus** - Survival reward
- **Energy penalty** - Penalizes excessive actuator use
- **Tail stability** - Penalizes tail angular velocity
- **Bite bonus** - Large reward when head contacts prey
- **Approach shaping** - Reward for closing distance to prey

## Curriculum Learning

### Stage 1: Balance (500K steps)
Learn to maintain a stable bipedal stance.

### Stage 2: Locomotion (1M steps)
Learn to walk/run forward toward a target.

### Stage 3: Bite (2M steps)
Sprint to prey and deliver a bite attack with the head.

## Quick Start

```bash
# Install from the repository root
pip install -e ".[all]"

# Run environment tests
python -m pytest environments/trex/tests/ -v

# Train stage 1 (balance)
python scripts/train_sb3.py train --stage 1 --timesteps 1000000

# View the model (requires display)
python scripts/view_model.py
```

## Environment Details

### Observation Space (dim=83)
| Component | Dimensions | Description |
|-----------|------------|-------------|
| Joint positions | 33 | All qpos except root freejoint (25 hinge + 2×4 ball) |
| Joint velocities | 31 | All qvel except root freejoint (25 hinge + 2×3 ball) |
| Pelvis quaternion | 4 | Orientation from framequat sensor |
| Pelvis gyro | 3 | Angular velocity |
| Pelvis linear vel | 3 | Linear velocity |
| Pelvis accel | 3 | Accelerometer |
| Foot contacts | 2 | Left/right foot touch (on central digit 3) |
| Prey direction | 3 | Unit vector to prey |
| Prey distance | 1 | Scalar distance |

### Action Space (dim=21)
All actions normalized to [-1, 1], scaled to actuator control ranges (3 neck/head + 14 legs + 4 tail).

### Termination Conditions
- Pelvis height outside healthy range (0.5m–1.6m)
- Excessive tilt angle
- Nosedive (forward pitch exceeds natural lean + threshold)
- Head/torso/tail contacts ground
- Bite success (head contacts prey)
- Episode length > max_episode_steps

## Files

```
trex/
├── assets/
│   └── trex.xml                # MuJoCo MJCF model
├── envs/
│   ├── __init__.py
│   └── trex_env.py             # Gymnasium environment
├── scripts/
│   ├── view_model.py           # MuJoCo passive viewer
│   ├── test_actuators.py       # Test joint movements
│   ├── test_env.py             # Environment validation script
│   └── train_sb3.py            # SB3 PPO training with curriculum
├── tests/
│   ├── test_trex_env.py        # Species-specific env tests
│   ├── test_trex_rewards.py    # Species-specific reward tests
│   └── test_static_balance.py  # Static balance tests
└── README.md
```

Hyperparameter configs are at `configs/trex/` in the repo root.
