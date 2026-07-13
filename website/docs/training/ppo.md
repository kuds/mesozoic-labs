---
sidebar_position: 1
---

# PPO Training

Proximal Policy Optimization (PPO) is a policy gradient method for reinforcement learning.

## Overview

PPO is known for:

- Stable training
- Good sample efficiency
- Easy hyperparameter tuning

## Basic Usage

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

Or use the included training script with curriculum learning:

```bash
cd environments/velociraptor

# Single stage; the current budget comes from its TOML config
python scripts/train_sb3.py train --stage 1 --algorithm ppo

# Full 3-stage curriculum in one command (per-stage hyperparameters applied automatically)
python scripts/train_sb3.py curriculum --algorithm ppo
```

## PPO Hyperparameters

PPO settings vary by species and curriculum stage. The authoritative values are
the `[ppo]` sections in `configs/<species>/stage*.toml`; copied defaults here
would quickly become stale. The main fields are `learning_rate`, `n_steps`,
`batch_size`, `n_epochs`, `gamma`, `gae_lambda`, `clip_range`, and `ent_coef`.

## 3-Stage Curriculum

PPO training follows the same curriculum stages as SAC:

1. **Stage 1 — Balance**: Stand upright without falling (`forward_vel_weight=0`, high `alive_bonus`)
2. **Stage 2 — Locomotion**: Walk and run forward (increase `forward_vel_weight`, add gait rewards)
3. **Stage 3 — Behavior**: Species-specific task (strike for Velociraptor, a fixed head-contact "bite" proxy for T-Rex, and a head-tip distance-based food-reach proxy for Brachiosaurus)

These task names are configuration labels. T-Rex has no articulated jaw, and
Brachiosaurus success does not require physical food contact.

Stage transitions are automated by the `CurriculumManager` using the thresholds
in each stage's TOML config. Current stage budgets and gates are shown on the
[generated model pages](/docs/models/velociraptor).

## Published Results

Published PPO summaries are displayed on the
[Velociraptor](/docs/models/velociraptor), [T-Rex](/docs/models/trex), and
[Brachiosaurus](/docs/models/brachiosaurus) pages from the generated catalog.
The available summaries are historical and unverified, so they do not establish
that PPO is faster, slower, better, or worse than SAC under controlled conditions.
