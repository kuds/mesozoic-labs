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
model.learn(total_timesteps=3_600_000, progress_bar=True)
model.save("raptor_sac")
```

Or use the included training script:

```bash
cd environments/velociraptor

# Single stage
python scripts/train_sb3.py train --stage 1 --algorithm sac --timesteps 1000000

# Full 3-stage curriculum in one command (per-stage hyperparameters applied automatically)
python scripts/train_sb3.py curriculum --algorithm sac
```

## SAC Hyperparameters

These defaults are defined in the per-species TOML config files under `configs/`.

| Parameter | Value | Description |
|-----------|-------|-------------|
| learning_rate | 3e-4 | Network learning rate |
| batch_size | 256 | Training batch size |
| gamma | 0.99 | Discount factor |
| tau | 0.005 | Soft update coefficient |
| ent_coef | auto | Automatic entropy tuning |
| buffer_size | 1M | Replay buffer size |

SAC hyperparameters are consistent across all three species. The `ent_coef="auto"` setting lets SAC automatically tune its entropy coefficient during training.

## 3-Stage Curriculum

SAC training follows the same curriculum as PPO:

1. **Stage 1 — Balance**: Stand upright without falling (`forward_vel_weight=0`, high `alive_bonus`)
2. **Stage 2 — Locomotion**: Walk and run forward (increase `forward_vel_weight`, add gait rewards)
3. **Stage 3 — Behavior**: Species-specific task (strike for Velociraptor, bite for T-Rex, food reach for Brachiosaurus)

Stage transitions are automated by the `CurriculumManager` when the agent achieves the threshold reward for 3 consecutive evaluations.

## Results

| Species | Steps | Avg Reward | Time |
|---------|-------|------------|------|
| Velociraptor | 3.6M | 3091.31 | 4:36:59 |

SAC significantly outperforms PPO for dinosaur locomotion tasks, achieving ~10x higher reward at the cost of longer training time. The replay buffer enables more efficient use of each experience sample.
