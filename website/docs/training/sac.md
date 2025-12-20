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
from mesozoic import DinoEnv, SACAgent

env = DinoEnv("trex")
agent = SACAgent(env)

agent.train(
    total_steps=3_600_000,
    learning_rate=3e-4,
    buffer_size=1_000_000
)
```

## Results

| Model | Steps | Avg Reward | Time |
|-------|-------|------------|------|
| Basic Dinosaur | 3.6M | 3091.31 | 4:36:59 |

SAC significantly outperforms PPO for dinosaur locomotion tasks.

:::note Coming Soon
Detailed SAC documentation is under development.
:::
