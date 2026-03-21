# Velociraptor MuJoCo Project

A bipedal dinosaur locomotion and predatory strike environment built with MuJoCo and Gymnasium.

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

Training proceeds in three stages:

**Stage 1: Balance** (learn to stand without falling)
```bash
python scripts/train_sb3.py train --stage 1 --timesteps 1000000
```

**Stage 2: Locomotion** (learn to walk/run forward)
```bash
python scripts/train_sb3.py train --stage 2 --timesteps 2000000 \
    --load logs/<stage1_dir>/models/stage1_final.zip
```

**Stage 3: Strike** (sprint and attack prey)
```bash
python scripts/train_sb3.py train --stage 3 --timesteps 3000000 \
    --load logs/<stage2_dir>/models/stage2_final.zip
```

### 5. Evaluate Trained Policy

```bash
python scripts/train_sb3.py eval logs/<run_dir>/models/stage3_final.zip
```

## Environment Details

### Observation Space (dim=67)
| Component | Dimensions | Description |
|-----------|------------|-------------|
| Joint positions | 24 | All qpos except root freejoint (24 hinge) |
| Joint velocities | 24 | All qvel except root freejoint (24 hinge) |
| Pelvis quaternion | 4 | Orientation from framequat sensor |
| Pelvis gyro | 3 | Angular velocity |
| Pelvis linear vel | 3 | Linear velocity |
| Pelvis accel | 3 | Accelerometer |
| Foot contacts | 2 | Left/right foot touch (on central digit 3) |
| Prey direction | 3 | Unit vector to prey |
| Prey distance | 1 | Scalar distance |

### Action Space (dim=22)
All actions normalized to [-1, 1], scaled to actuator control ranges (14 legs + 4 tail + 2 sickle claws + 2 arms).

### Reward Components

| Component | Weight (Stage 3) | Description |
|-----------|------------------|-------------|
| Forward velocity | 1.0 | Reward for moving +X |
| Alive bonus | 0.1 | Per-step survival bonus |
| Energy penalty | -0.001 | Penalize large actions |
| Tail stability | -0.05 | Penalize tail angular velocity |
| Strike bonus | +500 | Claw contacts prey |
| Approach shaping | 0.5 | Reward closing distance to prey |
| Fall penalty | -100 | Episode termination |

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
- Increase `alive_bonus` in early training
- Add a reward for maintaining upright orientation

**If it ignores the prey:**
- Add proximity reward (bonus for getting closer)
- Reduce `prey_distance_range` to spawn prey closer

## Phase 5: Migration to JAX/MJX

Once you have a working policy from SB3, port to MJX for faster training:

```bash
# Install JAX stack
pip install "jax[cuda12]" mujoco-mjx brax
```

The port involves:
1. Replace `gymnasium.Env` → `brax.envs.base.PipelineEnv`
2. Replace NumPy → JAX arrays
3. Replace `mujoco.mj_step` → `mjx.step`
4. Use `brax.training.agents.ppo.train`

See the [Brax documentation](https://github.com/google/brax) and [MuJoCo Playground examples](https://github.com/google-deepmind/mujoco_playground) for reference.

## Troubleshooting

**"No module named 'envs'"**
Run scripts from the project root: `cd velociraptor && python scripts/test_env.py`

**Viewer doesn't open**
Install a display backend: `pip install glfw` or run with `MUJOCO_GL=egl` for headless.

**Training is slow**
- Use more parallel envs: `--n-envs 8`
- Use subprocess vectorization: `--subproc`
- Reduce evaluation frequency: `--eval-freq 50000`

**NaN in observations**
- Physics is exploding; reduce timestep or actuator gains
- Check for division by zero in reward computation
