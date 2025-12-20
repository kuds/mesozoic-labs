---
sidebar_position: 1
---

# API Overview

Reference documentation for the Mesozoic Labs Python API.

## Core Classes

### DinoEnv

The main environment class for dinosaur simulation.

```python
from mesozoic import DinoEnv

env = DinoEnv(
    model="trex",           # Dinosaur model name
    render_mode="human",    # "human" or "rgb_array"
    frame_skip=4            # Action repeat
)

observation, info = env.reset()
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)
```

### Agents

```python
from mesozoic import PPOAgent, SACAgent

# PPO Agent
ppo = PPOAgent(env, learning_rate=3e-4)
ppo.train(total_steps=1_000_000)

# SAC Agent
sac = SACAgent(env, learning_rate=3e-4)
sac.train(total_steps=1_000_000)
```

:::note Coming Soon
Full API reference is under development.
:::
