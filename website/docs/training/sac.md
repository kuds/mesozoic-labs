---
sidebar_position: 2
---

# SAC Training

Soft Actor-Critic (SAC) is an off-policy algorithm that optimizes a stochastic policy with entropy regularization.

## Overview

SAC is known for:

- Sample efficiency
- Automatic temperature tuning
- Stable exploration

## Basic Usage

```python
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from environments.velociraptor.envs.raptor_env import RaptorEnv

def make_env():
    env = RaptorEnv(forward_vel_weight=1.0, alive_bonus=0.1)
    return Monitor(env)

vec_env = DummyVecEnv([make_env])
vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)

model = SAC("MlpPolicy", vec_env, learning_rate=3e-4, buffer_size=1_000_000)
# Illustrative standalone run, not a committed curriculum-stage budget.
model.learn(total_timesteps=10_000, progress_bar=True)
model.save("raptor_sac")
```

Or use the included training script:

```bash
cd environments/velociraptor

# Single stage; the current budget comes from its TOML config
python scripts/train_sb3.py train --stage 1 --algorithm sac

# Full 3-stage curriculum in one command (per-stage hyperparameters applied automatically)
python scripts/train_sb3.py curriculum --algorithm sac
```

## SAC Hyperparameters

SAC settings vary by species and curriculum stage. The authoritative values are
the `[sac]` sections in `configs/<species>/stage*.toml`; copied defaults here
would quickly become stale. The main fields are `learning_rate`, `batch_size`,
`gamma`, `tau`, `ent_coef`, `buffer_size`, `train_freq`, and `gradient_steps`.

## 3-Stage Curriculum

SAC training follows the same curriculum as PPO:

1. **Stage 1 — Balance**: Stand upright without falling (`forward_vel_weight=0`, high `alive_bonus`)
2. **Stage 2 — Locomotion**: Walk and run forward (increase `forward_vel_weight`, add gait rewards)
3. **Stage 3 — Behavior**: Species-specific task (strike for Velociraptor, a fixed head-contact "bite" proxy for T-Rex, and a head-tip distance-based food-reach proxy for Brachiosaurus)

These task names are configuration labels. T-Rex has no articulated jaw, and
Brachiosaurus success does not require physical food contact.

Stage transitions are automated by the `CurriculumManager` using the thresholds
in each stage's TOML config. Current stage budgets and gates are shown on the
[generated model pages](/docs/models/velociraptor).

## Published Results

The generated [Velociraptor model page](/docs/models/velociraptor) displays the
available SAC summary alongside its provenance status. That historical,
unverified run is not a controlled comparison with PPO and does not establish a
general performance or training-time advantage.
