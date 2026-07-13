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
    strike_bonus=10.0,         # Reward for claw-prey contact
)

observation, info = env.reset(seed=42)
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)
```

### Observation and Action Spaces

Observations combine joint state, root-body sensors, foot contacts, and
target-relative features. Actions are continuous values in `[-1, 1]`, scaled to
the actuator control ranges in the MJCF. The exact observation and action
dimensions are derived from the environment and compiled model and displayed on
the generated [Velociraptor model page](/docs/models/velociraptor).

### Reward Components

These are constructor defaults for the illustrative environment above, not the
current curriculum values. Training loads stage-specific overrides from the
TOML configs.

| Component | Constructor Default | Description |
|-----------|--------|-------------|
| `forward_vel_weight` | 1.0 | Reward proportional to forward velocity |
| `alive_bonus` | 0.1 | Per-step survival bonus |
| `energy_penalty_weight` | 0.001 | Penalizes large actions |
| `tail_stability_weight` | 0.05 | Penalizes tail angular velocity |
| `strike_bonus` | 10.0 | Bonus when sickle claw contacts prey |
| `strike_approach_weight` | 1.0 | Reward for closing distance to prey |
| `fall_penalty` | -100.0 | Penalty on termination from falling |

## T-Rex Environment

Large dinosaur-inspired bipedal model with a fixed head-contact task. The
Stage 3 name "Bite" is retained in configs and metric keys, but the model has no
articulated jaw: success means the fixed `head_bite` geom contacted the prey.

```python
from environments.trex.envs.trex_env import TRexEnv

env = TRexEnv(
    render_mode="human",
    bite_bonus=10.0,           # Reward for fixed head-geom/prey contact
    bite_approach_weight=1.0,  # Reward for closing distance
)
```

The generated [T-Rex model page](/docs/models/trex) is the authoritative public
source for its current observation and action dimensions.

## Brachiosaurus Environment

Quadrupedal dinosaur-inspired model with a distance-based food-target task.
"Food reached" means the head tip moved within the configured threshold; it
does not require physical contact with a food geom.

```python
from environments.brachiosaurus.envs.brachio_env import BrachioEnv

env = BrachioEnv(render_mode="human")
```

The generated [Brachiosaurus model page](/docs/models/brachiosaurus) is the
authoritative public source for its current observation and action dimensions.

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
# Illustrative standalone run, not a committed curriculum-stage budget.
model.learn(total_timesteps=10_000, progress_bar=True)
model.save("raptor_stage1")
```

Or use the included training script. The `curriculum` command runs all three stages in a single call — each stage loads its own hyperparameters from the TOML config automatically:

```bash
cd environments/velociraptor

# Full 3-stage curriculum (recommended) — one command, stages 1-3
python scripts/train_sb3.py curriculum --algorithm ppo --n-envs 4

# Or control stages individually; each command reads its budget from TOML
python scripts/train_sb3.py train --stage 1 --algorithm ppo
python scripts/train_sb3.py train --stage 2 --algorithm ppo \
  --load logs/stage1/models/stage1_final.zip
python scripts/train_sb3.py eval logs/stage2/models/stage2_final.zip --algorithm ppo --stage 2

# Use SAC instead
python scripts/train_sb3.py curriculum --algorithm sac

# Override hyperparameters without editing TOML files
python scripts/train_sb3.py train --stage 1 \
  --override ppo.learning_rate=1e-3 env.alive_bonus=3.0

# Write outputs to a specific directory (e.g. GCS mount for cloud training)
python scripts/train_sb3.py curriculum --output-dir /mnt/gcs/training/velociraptor
```

## Diagnostic Metrics

The `LocomotionMetrics` class (from `environments.shared.metrics`) computes
diagnostic metrics during evaluation. Call `record_step(info, reward)` each step, then
`compute()` at episode end.

```python
from environments.shared.metrics import LocomotionMetrics

metrics = LocomotionMetrics()
obs, info = env.reset()
for _ in range(1000):
    action = model.predict(obs)[0]
    obs, reward, terminated, truncated, info = env.step(action)
    metrics.record_step(info, reward)
    if terminated or truncated:
        break

report = metrics.compute()
agg = LocomotionMetrics.aggregate_episodes([report])
```

### Locomotion Health Metrics

| Metric key | Description | Species |
|---|---|---|
| `mean_forward_velocity` | Mean forward speed (m/s) | All |
| `cost_of_transport` | Energy per unit distance per unit weight | All |
| `mean_pelvis_height` | Mean pelvis z-position (m) | T-Rex, Velociraptor |
| `gait_symmetry` | Left/right stride symmetry ∈ [0, 1] | All |
| `stride_frequency` | Mean step frequency (Hz) | All |
| `mean_tilt_angle` | Mean body tilt angle (rad) | All |
| `termination_reason` | Reason episode ended | All |

### Behavioral Metrics

| Metric key | Description | Species |
|---|---|---|
| `mean_heading_alignment` | cos θ alignment toward prey ∈ [-1, 1] | T-Rex, Velociraptor |
| `mean_contact_asymmetry` | Left/right contact imbalance ∈ [0, 1] | All |
| `success_rate` | Episode-level indicator: 1 if any success event fired, otherwise 0 | All |
| `min_prey_distance` | Minimum distance reached to prey (m) | T-Rex, Velociraptor |
| `min_prey_distance` | Minimum distance reached to food (m) | Brachiosaurus |

### Species-Specific Key Mapping

| Metric | T-Rex | Velociraptor | Brachiosaurus |
|---|---|---|---|
| Height key | `pelvis_height` | `pelvis_height` | `pelvis_height` (aliased from torso) |
| Success key | `bite_success` | `strike_success` | `food_reached` |
| Prey/food key | `prey_distance` | `prey_distance` | `prey_distance` (aliased from head_food_distance) |
| Heading | ✓ | ✓ | N/A |

The key names are stable API labels, not biomechanical claims. In particular,
`bite_success` is fixed head-geom contact and `food_reached` is head-tip
proximity to the target.

The `evaluate` command in each `train_sb3.py` script prints all metrics above grouped
into **Core Performance**, **Velocity**, **Gait Quality**, **Balance**, and a species-specific
**Hunting / Food Reaching** section.
