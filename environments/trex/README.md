# Tyrannosaurus Rex Environment

A large dinosaur-inspired bipedal locomotion and head-contact environment built with MuJoCo and Gymnasium.

## Overview

This Tyrannosaurus-inspired model has a heavy head, vestigial forelimbs, and a tail used as a counterbalance. The Stage 3 task retains the name "Bite," but the implemented success event is contact between a fixed `head_bite` geom and the prey. The model has neck/head actuators but no articulated or actuated jaw, so the metric should be read as a head-contact proxy rather than simulated biting biomechanics.

## Generated Specifications, Curriculum, and Results

The authoritative public dimensions, current stage budgets, success criterion, and provenance-labelled historical
results are in the generator-managed [T-Rex catalog entry](../../README.md#t-rex). They are derived from the species
manifest, executable environment, compiled MJCF, current TOML stage configs, and result summaries.

## Implementation Notes

### Body Structure
- **Head/Neck**: Heavy head with a fixed contact geom, 3 actuated joints (neck pitch/yaw, head pitch), and no jaw joint
- **Legs**: Powerful digitigrade legs with 7 joints each (hip pitch/roll, knee, ankle, toe d2/d3/d4); the 4 proximal joints are actuated, the 3 toe digits ride passive springs
- **Tail**: 5 segments, 4 actuated (pitch 1, yaw 1, pitch 2, pitch 3), heavy counterbalance to skull

### Reward Components
- **Forward velocity** - Movement toward prey target
- **Alive bonus** - Survival reward
- **Energy penalty** - Penalizes excessive actuator use
- **Tail stability** - Penalizes tail angular velocity
- **Bite bonus** - Large reward when the fixed head contact geom touches prey
- **Approach shaping** - Reward for closing distance to prey

## Quick Start

```bash
# Install from the repository root
pip install -e ".[all]"

# Run environment tests
python -m pytest environments/trex/tests/ -v

# Train stage 1 using its current TOML-configured budget
python environments/trex/scripts/train_sb3.py train --stage 1

# Train the recovery stage (stage 1b): the stance task plus scheduled
# 165.5 N / 0.20 s external pushes derived from the plant itself
# (configs/trex/recovery.toml). Warm-start from a certified stance
# checkpoint; runs as a non-advancing pilot until the recovery gate's
# thresholds are calibrated and frozen by the gate resolver.
python environments/trex/scripts/train_sb3.py train --stage recovery --load <stance-checkpoint>.zip --load-mode initialize_next_stage

# View the model (requires display)
python environments/trex/scripts/view_model.py
```

## Environment Details

Observation and action totals are generated in the catalog entry linked above. The source of the observation layout and
action-to-actuator mapping is `envs/trex_env.py`; actions are normalized to
[-1, 1] residuals, where zero commands the complete XML `home` control and
the endpoints retain access to the full actuator ranges.

### Termination Conditions
- Pelvis height outside healthy range (0.5m–1.6m)
- Excessive tilt angle
- Nosedive (forward pitch exceeds natural lean + threshold)
- Head/torso/tail contacts ground
- Bite success (fixed `head_bite` geom contacts prey; no jaw articulation)
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
