---
sidebar_position: 3
---

# Hyperparameters

Guide to tuning hyperparameters for optimal training.

## Common Parameters

| Parameter | PPO Default | SAC Default | Description |
|-----------|-------------|-------------|-------------|
| learning_rate | 3e-4 | 3e-4 | Network learning rate |
| batch_size | 64 | 256 | Training batch size |
| gamma | 0.99 | 0.99 | Discount factor |
| buffer_size | N/A | 1M | Replay buffer size |

## Tips

1. **Start with defaults** - Our defaults work well for most dinosaur models
2. **Increase steps** - More training usually helps
3. **Monitor rewards** - Watch for plateaus
4. **Use GPU** - Training is 10x faster

:::note Coming Soon
Detailed hyperparameter tuning guide is under development.
:::
