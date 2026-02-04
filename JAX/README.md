# Velociraptor MuJoCo Project

A bipedal dinosaur locomotion and predatory strike environment built with MuJoCo and Gymnasium.

## Project Structure

```
velociraptor/
├── assets/
│   └── raptor.xml          # MJCF model definition
├── envs/
│   ├── __init__.py
│   └── raptor_env.py       # Gymnasium environment
├── scripts/
│   ├── view_model.py       # Passive viewer for MJCF iteration
│   ├── test_actuators.py   # Test joint movements
│   ├── test_env.py         # Verify environment works
│   └── train_sb3.py        # Training with Stable-Baselines3
├── configs/                 # (future) hyperparameter configs
├── logs/                    # Training logs (created during training)
├── models/                  # Saved models (created during training)
├── requirements.txt
└── README.md
```

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt
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
python scripts/train_sb3.py train --stage 1 --timesteps 500000
```

**Stage 2: Locomotion** (learn to walk/run forward)
```bash
python scripts/train_sb3.py train --stage 2 --timesteps 1000000 \
    --load logs/<stage1_dir>/models/stage1_final.zip
```

**Stage 3: Strike** (sprint and attack prey)
```bash
python scripts/train_sb3.py train --stage 3 --timesteps 2000000 \
    --load logs/<stage2_dir>/models/stage2_final.zip
```

### 5. Evaluate Trained Policy

```bash
python scripts/train_sb3.py eval logs/<run_dir>/models/stage3_final.zip
```

## Environment Details

### Observation Space (dim=51)
| Component | Dimensions | Description |
|-----------|------------|-------------|
| Joint positions | 20 | All joints except root freejoint |
| Joint velocities | 19 | All joints except root freejoint |
| Pelvis quaternion | 4 | Orientation |
| Pelvis gyro | 3 | Angular velocity |
| Pelvis linear vel | 3 | Linear velocity |
| Pelvis accel | 3 | Accelerometer |
| Foot contacts | 2 | Left/right foot touch |
| Prey direction | 3 | Unit vector to prey |
| Prey distance | 1 | Scalar distance |

### Action Space (dim=12)
All actions normalized to [-1, 1], scaled to actuator control ranges.

| Index | Actuator | Type |
|-------|----------|------|
| 0-4 | Right leg (hip pitch/roll, knee, ankle, toe) | Position |
| 5 | Right sickle claw | Motor |
| 6-10 | Left leg (hip pitch/roll, knee, ankle, toe) | Position |
| 11 | Left sickle claw | Motor |

### Reward Components

| Component | Weight (Stage 3) | Description |
|-----------|------------------|-------------|
| Forward velocity | 1.0 | Reward for moving +X |
| Alive bonus | 0.1 | Per-step survival bonus |
| Energy penalty | -0.001 | Penalize large actions |
| Tail stability | -0.02 | Penalize tail angular velocity |
| Strike bonus | +500 | Claw contacts prey |
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
