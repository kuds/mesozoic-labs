---
sidebar_position: 1
---

# API Overview

Reference documentation for the Mesozoic Labs environments.

## Gymnasium Registration

All environments are registered with the `MesozoicLabs` namespace, one per
model, as `MesozoicLabs/<Species>-<Embodiment>-v0`:

```python
import gymnasium as gym

# Registers every model of the species manifest
import environments  # noqa: F401

env = gym.make("MesozoicLabs/VelociraptorMongoliensis-Anatomical-v0")
```

Available environment IDs:

- `MesozoicLabs/VelociraptorMongoliensis-Anatomical-v0` - Velociraptor Mongoliensis (Anatomical)
- `MesozoicLabs/TyrannosaurusRex-Anatomical-v0` - Tyrannosaurus Rex (Anatomical)
- `MesozoicLabs/BrachiosaurusAltithorax-Anatomical-v0` - Brachiosaurus Altithorax (Anatomical)
- `MesozoicLabs/DibothrosuchusElaphros-Anatomical-v0` - Dibothrosuchus Elaphros (Anatomical)
- `MesozoicLabs/CompsognathusLongipes-Anatomical-v0` - Compsognathus Longipes (Anatomical)
- `MesozoicLabs/CompsognathusLongipes-Robot-v0` - Compsognathus Longipes (Robot), the robot prototype
- `MesozoicLabs/AllosaurusFragilis-Anatomical-v0` - Allosaurus Fragilis (Anatomical)

`import environments` registers all seven at once, from `configs/species_manifest.toml`.
Each class is `<Species><Embodiment>Env` in `environments.<species>.<embodiment>.env`,
a subclass of the species base class `<Species>Env` in `environments.<species>.base_env`.

## Velociraptor Environment

The main environment for velociraptor stance and locomotion training.

The hunt tasks (the velociraptor strike, the T-Rex head-contact bite, the
Brachiosaurus food reach, the Dibothrosuchus snap) are gone: decision D-D59
removed every hunt node, and with it their reward terms, success terminations,
info keys and constructor arguments (`strike_*`, `bite_*`, `food_reach_*`,
`food_approach_weight`, `food_head_proximity_weight`, `snap_*`), which these
classes now refuse with a `TypeError`. None of the four declares a success key,
so its evaluations record no success rate. The prey or food target itself is
still in the scene and the observation, and leaves with the next
policy-interface revision.

```python
from environments.velociraptor_mongoliensis.anatomical.env import VelociraptorMongoliensisAnatomicalEnv

env = VelociraptorMongoliensisAnatomicalEnv(
    render_mode="human",       # "human" or "rgb_array"
    frame_skip=5,              # Action repeat steps
    max_episode_steps=1000,
    forward_vel_weight=1.0,    # Reward for forward movement
    alive_bonus=0.1,           # Bonus for staying upright
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
the generated [Velociraptor model page](/docs/models/velociraptor_mongoliensis/anatomical).

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
| `fall_penalty` | -100.0 | Penalty on termination from falling |

## T-Rex Environment

Large dinosaur-inspired bipedal model. Its former hunt, a fixed head-contact
"bite" (the model has no articulated jaw), was removed by decision D-D59 with
its `bite_*` arguments; the `head_bite` geom it used leaves with the prey at
the next policy-interface revision.

```python
from environments.tyrannosaurus_rex.anatomical.env import TyrannosaurusRexAnatomicalEnv

env = TyrannosaurusRexAnatomicalEnv(render_mode="human")
```

The generated [T-Rex model page](/docs/models/tyrannosaurus_rex/anatomical) is the authoritative public
source for its current observation and action dimensions.

## Brachiosaurus Environment

Quadrupedal dinosaur-inspired model. Its former hunt, a distance-based food
reach (the head tip within a threshold of the food, with no contact), was
removed by decision D-D59 with its `food_reach_*`, `food_approach_weight` and
`food_head_proximity_weight` arguments.

```python
from environments.brachiosaurus_altithorax.anatomical.env import BrachiosaurusAltithoraxAnatomicalEnv

env = BrachiosaurusAltithoraxAnatomicalEnv(render_mode="human")
```

The generated [Brachiosaurus model page](/docs/models/brachiosaurus_altithorax/anatomical) is the
authoritative public source for its current observation and action dimensions.

## Training with Stable-Baselines3

```python
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from environments.velociraptor_mongoliensis.anatomical.env import VelociraptorMongoliensisAnatomicalEnv

def make_env():
    env = VelociraptorMongoliensisAnatomicalEnv(forward_vel_weight=0.0, alive_bonus=1.0)
    return Monitor(env)

vec_env = DummyVecEnv([make_env])
vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)

model = PPO("MlpPolicy", vec_env, learning_rate=3e-4)
# Illustrative standalone run, not a committed curriculum-stage budget.
model.learn(total_timesteps=10_000, progress_bar=True)
model.save("velociraptor_mongoliensis_anatomical_stance")
```

Or use the shared training CLI from the repository root, which takes the model as
`--species <species> --embodiment <embodiment>` (both required). The `curriculum` command runs the model's advancing stages, stance then locomotion, in a single call — each stage loads its own hyperparameters from the TOML config automatically:

```bash
# The full curriculum (recommended) — one command, stance then locomotion
python -m environments.shared.train --species velociraptor_mongoliensis --embodiment anatomical \
  curriculum --algorithm ppo --n-envs 4

# Or control stages individually; each command reads its budget from TOML
python -m environments.shared.train --species velociraptor_mongoliensis --embodiment anatomical \
  train --stage 1 --algorithm ppo
python -m environments.shared.train --species velociraptor_mongoliensis --embodiment anatomical \
  train --stage 2 --algorithm ppo --load logs/stage1/models/stage1_final.zip
python -m environments.shared.train --species velociraptor_mongoliensis --embodiment anatomical \
  eval logs/stage2/models/stage2_final.zip --algorithm ppo --stage 2

# Use SAC instead
python -m environments.shared.train --species velociraptor_mongoliensis --embodiment anatomical \
  curriculum --algorithm sac

# Override hyperparameters without editing TOML files
python -m environments.shared.train --species velociraptor_mongoliensis --embodiment anatomical \
  train --stage 1 --override ppo.learning_rate=1e-3 env.alive_bonus=3.0

# Write outputs to a specific directory, a fresh one per run (the curriculum
# refuses a stage directory that already records a stage)
python -m environments.shared.train --species velociraptor_mongoliensis --embodiment anatomical \
  curriculum --output-dir logs/<new_run>
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
| `success_rate` | Episode-level indicator: 1 if any of the model's success keys fired, otherwise 0; absent for a model that declares none | Allosaurus (`course_complete`) and the Compsognathus pair (`target_success`) |

### Species-Specific Key Mapping

| Metric | T-Rex | Velociraptor | Brachiosaurus |
|---|---|---|---|
| Height key | `pelvis_height` | `pelvis_height` | `pelvis_height` (aliased from torso) |
| Success keys (`SUCCESS_KEYS`) | none | none | none |
| Heading | ✓ | ✓ | N/A |

`LocomotionMetrics(success_keys=...)` takes a model's `SUCCESS_KEYS`
(`SpeciesConfig.success_keys`); with none it records no `success_rate`. The
hunt's success keys (`bite_success`, `strike_success`, `food_reached`,
`snap_success`) and its `prey_distance` and time-to-target metrics were
removed with the hunt (decision D-D59).

The shared CLI's `eval` command prints all metrics above grouped
into **Core Performance**, **Velocity**, **Gait Quality** and **Balance**, with
a success rate only for a model that declares success keys.
