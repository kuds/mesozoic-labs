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

## Per-Stage Hyperparameters

**Each stage has its own `[ppo]` and `[sac]` sections.** When the `curriculum` command advances to the next stage it re-initialises the model with that stage's hyperparameters automatically. The values follow a deliberate progression across all three species:

| Hyperparameter | Stage 1 (Balance) | Stage 2 (Locomotion) | Stage 3 (Behavior) | Rationale |
|---|---|---|---|---|
| `learning_rate` | `3e-4` | `1e-4` | `5e-5` | Coarser search early, fine-tune for complex behaviour |
| `ent_coef` (PPO) | `0.005–0.01` | `0.005–0.01` | `0.001` | High exploration for balance, exploit for strike/bite/food |
| `clip_range` (PPO) | `0.2` | `0.2` | `0.1` | Conservative updates for sparse terminal rewards |
| `gamma` | `0.99–0.998` | `0.99` | `0.995` | More farsighted when rewards are sparse |
| `n_steps` (PPO) | `2048–4096` | `2048–4096` | `4096` | Larger rollout buffer for complex behaviour |
| `batch_size` | `64–256` | `128–256` | `256` | Larger batches for later stages |

The `env` reward weights also shift significantly between stages — that is the core mechanic of curriculum learning:

| Reward component | Stage 1 | Stage 2 | Stage 3 |
|---|---|---|---|
| `alive_bonus` | High (1.0–2.0) | Moderate (0.5–1.0) | Low (0.1) |
| `forward_vel_weight` | `0.0` | High (1.0–2.0) | Moderate (1.0) |
| Behavior bonus (strike/bite/food) | `0.0` | `0.0` | High (10–500) |

## PPO Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| learning_rate | 3e-4 | Network learning rate (3e-4 → 1e-4 → 5e-5 across stages) |
| learning_rate_end | *(optional)* | If set, enables a linear decay schedule from `learning_rate` to this value |
| n_steps | 2048–4096 | Steps per rollout buffer (larger in later stages) |
| batch_size | 64–256 | Minibatch size for gradient updates |
| n_epochs | 10 | Number of epochs per PPO update |
| gamma | 0.99–0.998 | Discount factor (higher in stage 3 for sparse rewards) |
| gae_lambda | 0.95 | GAE lambda for advantage estimation |
| clip_range | 0.2 → 0.1 | PPO surrogate objective clip range (tightened in stage 3) |
| ent_coef | 0.001–0.01 | Entropy bonus (higher early, lower for fine-grained behaviour) |

## SAC Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| learning_rate | 3e-4 → 1e-4 → 5e-5 | Network learning rate (same per-stage progression as PPO) |
| batch_size | 256 | Training batch size |
| gamma | 0.99–0.995 | Discount factor |
| tau | 0.005 | Soft target update coefficient |
| ent_coef | `"auto"` | Entropy coefficient (auto-tuned throughout) |

## Curriculum Thresholds

Stage transitions are controlled by the `[curriculum]` section in each config:

| Parameter | Default | Description |
|-----------|---------|-------------|
| timesteps | 4000000–8000000 | Maximum timesteps per stage before auto-advancing |
| min_avg_reward | 50–100 | Minimum average reward to advance early |
| min_avg_episode_length | 300–800 | Minimum average episode length to advance early |
| required_consecutive | 3 | Consecutive evaluations above threshold |

Both `min_avg_reward` and `min_avg_episode_length` must be exceeded for `required_consecutive` evaluations before a stage advances early. If the `timesteps` budget runs out first, the stage advances anyway.

## Overriding Hyperparameters from the CLI

Use `--override` to change TOML values without editing files — useful for hyperparameter sweeps. Keys use dot notation; values are auto-cast to `int`, `float`, or `str`:

```bash
# Override learning rate and entropy coefficient for all stages
python scripts/train_sb3.py train --stage 1 \
  --override ppo.learning_rate=1e-3 ppo.ent_coef=0.02 env.alive_bonus=5.0

# Works with curriculum too — applies to ALL stages
python scripts/train_sb3.py curriculum \
  --override ppo.learning_rate=2e-4
```

Supported key prefixes:

| Prefix | Overrides |
|--------|-----------|
| `ppo.X` | `ppo_kwargs[X]` |
| `sac.X` | `sac_kwargs[X]` |
| `env.X` | `env_kwargs[X]` (reward weights, episode settings) |

For stage-scoped overrides within a curriculum run, prefix with the stage number: `1.ppo.learning_rate=3e-4 2.ppo.learning_rate=1e-4`. Plain `section.key=value` still applies to all stages.

> **Systematic sweeps:** To try many combinations automatically, use the Vertex AI HPT sweep tool. See [Hyperparameter Sweeps](sweeps.md).

## Tips

1. **Start with defaults** — The TOML configs are tuned for each species
2. **Use `--algorithm sac`** — SAC achieves higher final reward; PPO trains faster per step
3. **Monitor with W&B** — Use `--wandb` to track per-component rewards across stages
4. **Use GPU** — Training is significantly faster with CUDA
5. **Increase timesteps for stage 3** — The sparse terminal reward (strike/bite/food) often needs more samples to converge
