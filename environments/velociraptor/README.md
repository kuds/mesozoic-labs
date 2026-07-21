# Velociraptor MuJoCo Project

A bipedal dinosaur locomotion and predatory strike environment built with MuJoCo and Gymnasium.

## Generated Specifications, Curriculum, and Results

The authoritative public dimensions, current stage budgets, success criterion, and provenance-labelled historical
results are in the generator-managed [Velociraptor catalog entry](../../README.md#velociraptor). They are derived from
the species manifest, executable environment, compiled MJCF, current TOML stage configs, and result summaries.

## Project Structure

```
velociraptor/
├── assets/
│   └── raptor.xml              # MJCF model definition
├── envs/
│   ├── __init__.py
│   └── raptor_env.py           # Gymnasium environment
├── scripts/
│   ├── view_model.py           # Passive viewer for MJCF iteration
│   ├── test_actuators.py       # Test joint movements
│   ├── test_env.py             # Verify environment works
│   └── train_sb3.py            # Training with Stable-Baselines3
├── tests/
│   ├── test_raptor_env.py      # Species-specific env tests
│   ├── test_raptor_rewards.py  # Species-specific reward tests
│   └── test_static_balance.py  # Static balance tests
└── README.md
```

Hyperparameter configs are at `configs/velociraptor/` in the repo root.

## Installation

```bash
# Install from the repository root
pip install -e ".[all]"
```

## Quick Start

Run the commands in this section from the species directory:

```bash
cd environments/velociraptor
```

### 1. View the Model

First, verify the MJCF loads correctly:

```bash
python scripts/view_model.py
```

This opens a passive viewer. Check that:
- The raptor settles into a stable crouch
- No body parts explode or clip through each other
- The tail oscillates briefly then stabilizes

### 2. Test Actuators

See all joints move through their ranges:

```bash
python scripts/test_actuators.py
```

### 3. Test Environment

Run the environment test suite:

```bash
python scripts/test_env.py
python scripts/test_env.py --render  # With visualization
```

### 4. Train with Curriculum Learning

Run all three stages using the current TOML-configured budgets:

```bash
python scripts/train_sb3.py curriculum --algorithm ppo
```

### 5. Evaluate Trained Policy

```bash
python scripts/train_sb3.py eval logs/<run_dir>/models/stage3_final.zip
```

## Environment Details

Observation and action totals are generated in the catalog entry linked above. The source of the observation layout and
action-to-actuator mapping is `envs/raptor_env.py`. Actions are normalized residuals around the named XML `home`
keyframe: zero commands the standing pose, while -1 and +1 still reach each actuator's lower and upper limits through
piecewise-linear interpolation. The evidence and compatibility rationale are in the
[Stage-1 basin investigation](../../docs/investigations/VELOCIRAPTOR_STAGE1_BASIN_INVESTIGATION.md).

### Reward Components

Reward weights vary by stage. The `[env]` section of each
`configs/velociraptor/stage*.toml` file is authoritative; this README does not
copy numeric weights. Components include locomotion, survival, posture, energy,
tail stability, approach shaping, target contact, and fall penalties. Posture
shaping is direction-aware and centred on the raptor's natural forward lean;
absolute tilt remains the safety signal for termination.

### Termination Conditions
- Pelvis height < 0.25m (fallen)
- Pelvis height > 1.0m (launched into air)
- Torso contacts ground
- Episode length > max_episode_steps

## Tuning Guide

### MJCF Model (`assets/raptor.xml`)

**If the raptor falls immediately:**
- Increase `damping` on leg joints
- Adjust initial pose (qpos0) to more stable crouch
- Check CoM is over the feet

**If movements are jerky:**
- Reduce actuator `kp` gains
- Increase `damping`
- Reduce control frequency (increase `frame_skip`)

**If the tail flops around:**
- Increase tail joint `stiffness` and `damping`
- Reduce tail joint `range`

### Reward Weights (`envs/raptor_env.py`)

**If it doesn't learn to walk:**
- Increase `forward_vel_weight`
- Decrease `energy_penalty_weight`
- Check that alive_bonus isn't dominating

**If it walks but falls a lot:**
- Inspect the episode-length distribution and termination mix before changing weights
- Check whether `forward_z` tracks the natural lean; do not reward world-vertical posture for this morphology
- Lower reset noise only if a controlled probe isolates reset perturbations as the failure source

**If it ignores the prey:**
- Add proximity reward (bonus for getting closer)
- Reduce `prey_distance_range` to spawn prey closer

## JAX/MJX Backend

The repository includes an experimental shared JAX/MJX PPO path. See the
[JAX/MJX training guide](../../website/docs/training/jax.md) for its current
scope, installation extra, and backend-parity limitations.

## Troubleshooting

**"No module named 'envs'"**
Run scripts from the species directory:
`cd environments/velociraptor && python scripts/test_env.py`.

**Viewer doesn't open**
Install a display backend: `pip install glfw` or run with `MUJOCO_GL=egl` for headless.

**Training is slow**
- Use more parallel envs: `--n-envs 8`
- Use subprocess vectorization: `--subproc`
- Reduce evaluation frequency: `--eval-freq 50000`

**NaN in observations**
- Physics is exploding; reduce timestep or actuator gains
- Check for division by zero in reward computation
