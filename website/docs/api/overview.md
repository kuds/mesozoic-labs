---
sidebar_position: 1
---

# API Overview

Reference documentation for the Mesozoic Labs environments.

## Gymnasium Registration

All environments are registered with the `MesozoicLabs` namespace:

```python
import gymnasium as gym

# Import to trigger registration
import environments.velociraptor.envs.raptor_env  # noqa: F401

env = gym.make("MesozoicLabs/Raptor-v0")
```

Available environment IDs:
- `MesozoicLabs/Raptor-v0` - Velociraptor
- `MesozoicLabs/Brachio-v0` - Brachiosaurus
- `MesozoicLabs/TRex-v0` - T-Rex

## Velociraptor Environment

The main environment for velociraptor locomotion and predatory strike training.

```python
from environments.velociraptor.envs.raptor_env import RaptorEnv

env = RaptorEnv(
    render_mode="human",       # "human" or "rgb_array"
    frame_skip=5,              # Action repeat steps
    max_episode_steps=1000,
    forward_vel_weight=1.0,    # Reward for forward movement
    alive_bonus=0.1,           # Bonus for staying upright
    strike_bonus=500.0,        # Reward for claw-prey contact
)

observation, info = env.reset(seed=42)
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)
```

### Observation Space (69 dimensions)

| Component | Dims | Description |
|-----------|------|-------------|
| Joint positions | 26 | All qpos excluding root freejoint (18 hinge + 2x4 ball) |
| Joint velocities | 24 | All qvel excluding root freejoint (18 hinge + 2x3 ball) |
| Pelvis orientation | 4 | Quaternion from framequat sensor |
| Pelvis angular velocity | 3 | Gyroscope reading |
| Pelvis linear velocity | 3 | Root body velocity |
| Pelvis acceleration | 3 | Accelerometer reading |
| Foot contact | 2 | Left/right touch sensors |
| Prey direction | 3 | Unit vector toward prey |
| Prey distance | 1 | Scalar distance to prey |

### Action Space (12 dimensions)

Continuous actions in `[-1, 1]`, scaled to actuator control ranges:
- Right leg: hip pitch, hip roll, knee, ankle, toe (5)
- Right sickle claw (1)
- Left leg: hip pitch, hip roll, knee, ankle, toe (5)
- Left sickle claw (1)

### Reward Components

| Component | Weight | Description |
|-----------|--------|-------------|
| `forward_vel_weight` | 1.0 | Reward proportional to forward velocity |
| `alive_bonus` | 0.1 | Per-step survival bonus |
| `energy_penalty_weight` | 0.001 | Penalizes large actions |
| `tail_stability_weight` | 0.05 | Penalizes tail angular velocity |
| `strike_bonus` | 500.0 | Bonus when sickle claw contacts prey |
| `strike_approach_weight` | 0.5 | Reward for closing distance to prey |
| `fall_penalty` | -100.0 | Penalty on termination from falling |

## T-Rex Environment

Large bipedal predator with bite-attack behavior.

```python
from environments.trex.envs.trex_env import TRexEnv

env = TRexEnv(
    render_mode="human",
    bite_bonus=500.0,          # Reward for jaw-prey contact
    bite_approach_weight=0.5,  # Reward for closing distance
)
```

- **Observation:** 77 dimensions
- **Action:** 14 dimensions (3 neck/head + 1 jaw + 5 per leg)

## Brachiosaurus Environment

Quadrupedal sauropod with food-reaching behavior.

```python
from environments.brachiosaurus.envs.brachio_env import BrachioEnv

env = BrachioEnv(render_mode="human")
```

- **Observation:** 75 dimensions
- **Action:** 22 dimensions (6 neck + 16 legs)

## Training with Stable-Baselines3

```python
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from environments.velociraptor.envs.raptor_env import RaptorEnv

def make_env():
    env = RaptorEnv(forward_vel_weight=0.0, alive_bonus=1.0)
    return Monitor(env)

vec_env = DummyVecEnv([make_env])
vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)

model = PPO("MlpPolicy", vec_env, learning_rate=3e-4)
model.learn(total_timesteps=1_000_000, progress_bar=True)
model.save("raptor_stage1")
```

Or use the included training script with curriculum learning:

```bash
cd environments/velociraptor
python scripts/train_sb3.py train --stage 1 --timesteps 1000000
python scripts/train_sb3.py train --stage 2 --timesteps 1000000 --load models/stage1_final.zip
python scripts/train_sb3.py eval models/stage2_final.zip --stage 2
```
