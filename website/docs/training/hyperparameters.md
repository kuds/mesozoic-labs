---
sidebar_position: 3
---

# Hyperparameters

Guide to tuning hyperparameters for optimal training.

## Config Files

All hyperparameters are defined in TOML config files under `configs/<species>/`. Each species has three stage configs:

```
configs/
├── velociraptor/
│   ├── stage1_balance.toml
│   ├── stage2_locomotion.toml
│   └── stage3_strike.toml
├── trex/
│   ├── stage1_balance.toml
│   ├── stage2_locomotion.toml
│   └── stage3_bite.toml
└── brachiosaurus/
    ├── stage1_balance.toml
    ├── stage2_locomotion.toml
    └── stage3_food_reach.toml
```

Each TOML file contains `[ppo]`, `[sac]`, `[env]`, and `[curriculum]` sections.

## PPO Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| learning_rate | 3e-4 | Network learning rate |
| n_steps | 2048-4096 | Steps per rollout buffer (Velociraptor uses 4096) |
| batch_size | 64 | Minibatch size for gradient updates |
| n_epochs | 10 | Number of epochs per PPO update |
| gamma | 0.99 | Discount factor for future rewards |
| gae_lambda | 0.95 | GAE lambda for advantage estimation |
| clip_range | 0.2 | PPO surrogate objective clip range |
| ent_coef | 0.01-0.03 | Entropy bonus coefficient (Velociraptor uses 0.03) |

## SAC Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| learning_rate | 3e-4 | Network learning rate |
| batch_size | 256 | Training batch size |
| gamma | 0.99 | Discount factor |
| tau | 0.005 | Soft target update coefficient |
| ent_coef | auto | Entropy coefficient (auto-tuned) |
| buffer_size | 1M | Experience replay buffer size |

## Environment Reward Weights

Each training stage emphasizes different reward components:

| Weight | Stage 1 (Balance) | Stage 2 (Locomotion) | Stage 3 (Behavior) |
|--------|-------------------|----------------------|--------------------|
| alive_bonus | High (1.0-5.0) | Moderate | Low |
| forward_vel_weight | 0.0 | High | Moderate |
| energy_penalty_weight | 0.0005 | 0.0005 | 0.0005 |
| behavior_weight* | 0.0 | 0.0 | High |

*Behavior weight varies by species: `strike_bonus` (Velociraptor), `bite_bonus` (T-Rex), `food_reach_bonus` (Brachiosaurus).

## Curriculum Thresholds

Stage transitions are controlled by the `[curriculum]` section in each config:

| Parameter | Default | Description |
|-----------|---------|-------------|
| timesteps | 1000000 | Maximum timesteps per stage |
| min_avg_reward | 50.0 | Minimum average reward to advance |
| min_avg_episode_length | 400 | Minimum average episode length |
| required_consecutive | 3 | Consecutive evaluations above threshold |

## Tips

1. **Start with defaults** — The TOML configs are tuned for each species
2. **Increase timesteps** — More training usually helps, especially for Stage 3
3. **Monitor with W&B** — Use `--wandb` flag to track per-component rewards
4. **Use GPU** — Training is significantly faster with CUDA
5. **SAC vs PPO** — SAC achieves higher final reward; PPO trains faster per step
