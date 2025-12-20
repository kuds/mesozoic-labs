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
from mesozoic import DinoEnv, PPOAgent

env = DinoEnv("trex")
agent = PPOAgent(env)

agent.train(
    total_steps=2_600_000,
    learning_rate=3e-4,
    clip_ratio=0.2
)
```

## Results

| Model | Steps | Avg Reward | Time |
|-------|-------|------------|------|
| Basic Dinosaur | 2.6M | 319.94 | 1:29:43 |

:::note Coming Soon
Detailed PPO documentation is under development.
:::
