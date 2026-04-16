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

## Tuning Playbook

Use this section as a symptom-driven reference when a run is misbehaving. Each entry lists the most effective knobs first. Adjust one group at a time so you can attribute outcomes to changes.

### Stage 1 — Balance

| Symptom | Most likely cause | What to change |
|---|---|---|
| Agent falls immediately (ep length ~30–80, reward < 50) | Over-aggressive actions or wobbly initial pose | Lower `ppo.learning_rate` (try `3e-5`), raise `env.posture_weight`, raise `env.nosedive_weight` |
| Agent hops / drifts across the arena to stay "alive" | `alive_bonus` too large relative to drift/spin penalties | Lower `env.alive_bonus` to `1.0–1.75`, raise `env.drift_penalty_weight` and `env.speed_penalty_weight` |
| Spins in place to maintain upright torso | Spin is cheaper than true balance | Raise `env.spin_penalty_weight` to `0.1+`, add small `env.heading_weight` (`0.1`) |
| Reward plateaus at a mediocre value, no further progress | Entropy too low or LR too high for fine-tuning | Lower LR to `3e-5`, raise `ppo.ent_coef` to `0.01`, or switch to cosine schedule via `learning_rate_end` |
| Reward is highly variable across seeds | Init noise dominates | Lower `env.reset_noise_scale` (default `0.05`, try `0.02`) |
| Jerky, unstable joint motion | Smoothness under-weighted | Raise `env.smoothness_weight` and `env.energy_penalty_weight` |

### Stage 2 — Locomotion

| Symptom | Most likely cause | What to change |
|---|---|---|
| Forward velocity stuck near zero | `forward_vel_weight` too low or `alive_bonus` dominant | Raise `env.forward_vel_weight` to `1.0–2.0`, reduce `env.alive_bonus` |
| Crab-walks sideways toward target | No lateral or heading constraint | Set `env.lateral_penalty_weight` to `0.1`, raise `env.heading_weight` |
| Walks forward but falls frequently | Balance reward zeroed too aggressively | Keep `env.posture_weight ≥ 0.3`, re-enable mild `env.alive_bonus` (`0.3–0.5`) |
| Unrealistic "ice-skating" gait | Symmetry and smoothness under-weighted | Raise `env.gait_symmetry_weight`, raise `env.smoothness_weight` |
| Max forward speed exceeds physical reasonableness | Reward uncapped | Set `env.forward_vel_max` (e.g. `3.0` for raptor, `1.5` for brachio) |

### Stage 3 — Behavior (Strike / Bite / Food Reach)

| Symptom | Most likely cause | What to change |
|---|---|---|
| Never triggers terminal event (strike/bite/food) | Sparse-reward exploration stalled | Raise `ppo.ent_coef` to `0.005–0.01`, widen `ppo.clip_range` to `0.15`, tighten `env.prey_distance_range` so the target spawns closer |
| Lingers near target without triggering | Proximity bonuses rewarding hovering | Zero `env.strike_proximity_weight` (or `env.food_proximity_weight`), keep `env.*_bonus` high (`1000`+) and `*_approach_weight` as the only gradient |
| Forgets locomotion during Stage 3 warm-up | Reward schedule shift too abrupt | Use `curriculum.warmup_timesteps = 300000`, `curriculum.ramp_timesteps = 500000`, `curriculum.warmup_clip_range = 0.02` to anneal changes |
| Learns the behavior but then regresses | Over-entropy or critic drift | Lower `ppo.ent_coef` after convergence, narrow `ppo.clip_range` to `0.1` |
| `strike_bonus` signal not dominating | Discounted future alive-reward too large | Ensure `env.alive_bonus = 0` in stage 3 and that `strike_bonus >> gamma^H · per_step_reward` |

### Algorithm-specific notes

**PPO.** The most impactful lever is the `learning_rate` / `learning_rate_end` pair — roughly 10× lower than the "textbook" `3e-4` produces the most consistent runs in this repo. `n_steps × n_envs` must be divisible by `batch_size`; otherwise SB3 silently drops samples. `clip_range` should narrow across the curriculum (`0.2 → 0.15 → 0.1`) as policies become more specialized.

**SAC.** `ent_coef = "auto"` handles exploration automatically — prefer it over a fixed float. The `train_freq=16, gradient_steps=8` ratio (2:1 gradient-to-env) smooths the learning curve at modest compute cost. Set `buffer_size` proportional to stage length (300K for Stage 1, 1M for Stage 3).

**JAX / MJX.** `num_envs × rollout_len` is the effective rollout buffer. The default `2048 × 64` ≈ 131K is aggressive — reduce to `1024 × 64` if GPU memory is tight. `warmup_updates` and `ramp_updates` in `[jax]` mirror the `warmup_timesteps` / `ramp_timesteps` knobs from `[curriculum]` but in update-count units.

### A minimal tuning workflow

1. **Baseline.** Run the stage with committed defaults for 2–3 seeds. Record best reward, mean episode length, and any behavioral metrics (success rate, velocity).
2. **Diagnose.** If the run fails, match symptoms against the tables above. Do not change more than one group of knobs per run.
3. **Narrow.** For promising directions, launch a Ray Tune sweep over 3–5 candidate values using `notebooks/ray_tune_sweep.ipynb`. Use the ASHA scheduler to prune early.
4. **Promote.** Commit the winning values back to the TOML with a trailing comment explaining why (see existing configs for the house style — e.g. `# Setting 4 sweep: ...`).
5. **Regress-test.** Re-run the full curriculum end-to-end on the winning config before committing. A Stage 1 change often degrades Stage 3.

For systematic multi-parameter sweeps, see [Hyperparameter Sweeps](sweeps.md).
