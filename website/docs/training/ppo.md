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
model.learn(total_timesteps=500_000, progress_bar=True)
model.save("raptor_stage1")
```

Or use the included training script with curriculum learning:

```bash
cd environments/velociraptor
python scripts/train_sb3.py train --stage 1 --timesteps 500000
```

## Results

| Model | Steps | Avg Reward | Time |
|-------|-------|------------|------|
| Basic Dinosaur | 2.6M | 319.94 | 1:29:43 |

:::note Coming Soon
Detailed PPO documentation is under development.
:::
