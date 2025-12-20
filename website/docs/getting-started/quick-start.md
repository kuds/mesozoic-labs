---
sidebar_position: 2
---

# Quick Start

Train your first robotic dinosaur in minutes.

## Basic Training Loop

```python
from mesozoic import DinoEnv, SACAgent

# Initialize environment with T-Rex model
env = DinoEnv("trex")

# Create SAC agent
agent = SACAgent(
    observation_space=env.observation_space,
    action_space=env.action_space,
    learning_rate=3e-4
)

# Train the dinosaur to walk
agent.train(total_steps=1_000_000)

# Save the trained model
agent.save("trex_walker.pt")
```

## Visualize Results

```python
# Render the trained dinosaur
env.render_episode(agent, save_video="trex_walking.mp4")
```

:::note Coming Soon
More detailed quick start guide is under development.
:::
