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
model.learn(total_timesteps=1_000_000, progress_bar=True)
model.save("raptor_stage1")
```

Or use the included training script with curriculum learning:

```bash
cd environments/velociraptor

# Single stage
python scripts/train_sb3.py train --stage 1 --algorithm ppo --timesteps 1000000

# Full 3-stage curriculum in one command (per-stage hyperparameters applied automatically)
python scripts/train_sb3.py curriculum --algorithm ppo
```

## PPO Hyperparameters

These defaults are defined in the per-species TOML config files under `configs/`.

| Parameter | Velociraptor | T-Rex | Brachiosaurus | Description |
|-----------|-------------|-------|---------------|-------------|
| learning_rate | 3e-4 | 3e-4 | 3e-4 | Network learning rate |
| n_steps | 4096 | 2048 | 2048 | Steps per rollout |
| batch_size | 64 | 64 | 64 | Minibatch size |
| n_epochs | 10 | 10 | 10 | Epochs per update |
| gamma | 0.99 | 0.99 | 0.99 | Discount factor |
| gae_lambda | 0.95 | 0.95 | 0.95 | GAE lambda |
| clip_range | 0.2 | 0.2 | 0.2 | PPO clip range |
| ent_coef | 0.03 | 0.01 | 0.01 | Entropy coefficient |

The Velociraptor uses a higher `ent_coef` (0.03) and larger rollout buffer (`n_steps=4096`) compared to the other species.

## 3-Stage Curriculum

PPO training follows the same curriculum stages as SAC:

1. **Stage 1 — Balance**: Stand upright without falling (`forward_vel_weight=0`, high `alive_bonus`)
2. **Stage 2 — Locomotion**: Walk and run forward (increase `forward_vel_weight`, add gait rewards)
3. **Stage 3 — Behavior**: Species-specific task (strike for Velociraptor, bite for T-Rex, food reach for Brachiosaurus)

Stage transitions are automated by the `CurriculumManager` when the agent achieves the threshold reward for 3 consecutive evaluations.

## Results

| Species | Steps | Avg Reward | Time |
|---------|-------|------------|------|
| Velociraptor | 2.6M | 319.94 | 1:29:43 |

PPO trains faster per step but achieves lower final reward compared to SAC for dinosaur locomotion tasks. See the [SAC page](/docs/training/sac) for comparison.
