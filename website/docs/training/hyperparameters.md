---
sidebar_position: 3
---

# Hyperparameters

Guide to tuning hyperparameters for training experiments.

## Config Files

All hyperparameters are defined in TOML config files under `configs/<species>/`. Each species has three stage configs:

```
configs/
├── velociraptor/
│   ├── stance.toml
│   ├── locomotion.toml
│   └── stage3_strike.toml
├── trex/
│   ├── stance.toml
│   ├── locomotion.toml
│   └── behavior.toml
└── brachiosaurus/
    ├── stance.toml
    ├── locomotion.toml
    └── stage3_food_reach.toml
```

Each TOML file contains `[ppo]`, `[sac]`, `[env]`, and `[curriculum]` sections.

## Per-Stage Hyperparameters

**Each stage has its own `[ppo]` and `[sac]` sections.** When the
`curriculum` command advances, it loads the next TOML file and re-initialises
the model with that stage's settings. Values differ across species and change
as experiments evolve, so the stage TOML files are the authoritative source.

The broad curriculum intent is balance first, locomotion second, and a
species-specific simulator task third. Reward weights shift with that intent;
do not infer current values from a copied range in this guide. The generated
[model pages](/docs/models/velociraptor) show the current stage names, budgets,
and advancement gates read from the same configs.

## PPO Parameters

| Parameter | Description |
|-----------|-------------|
| `learning_rate` | Initial network learning rate |
| `learning_rate_end` | Optional end value for a linear learning-rate schedule |
| `n_steps` | Steps collected per rollout buffer |
| `batch_size` | Minibatch size for gradient updates |
| `n_epochs` | Optimisation epochs per PPO update |
| `gamma` | Reward discount factor |
| `gae_lambda` | Generalized advantage-estimation factor |
| `clip_range` | PPO surrogate-objective clipping range |
| `ent_coef` / `ent_coef_end` | Initial and optional scheduled entropy coefficients |

## SAC Parameters

| Parameter | Description |
|-----------|-------------|
| `learning_rate` | Network learning rate |
| `batch_size` | Replay-sample batch size |
| `gamma` | Reward discount factor |
| `tau` | Soft target-update coefficient |
| `ent_coef` | Entropy coefficient or automatic-tuning mode |
| `buffer_size` | Replay-buffer capacity |
| `train_freq` | Environment-data collection frequency between updates |
| `gradient_steps` | Gradient updates per training interval |

## Curriculum Thresholds

Stage transitions are controlled by the `[curriculum]` section in each config:

| Parameter | Description |
|-----------|-------------|
| `timesteps` | Maximum configured stage budget before advancing |
| `min_avg_reward` | Minimum evaluation-window mean reward for early advancement |
| `min_avg_episode_length` | Minimum evaluation-window mean episode length for early advancement |
| `min_avg_forward_vel` | Optional minimum mean forward velocity; enabled when greater than zero |
| `min_success_rate` | Optional minimum episode success rate; enabled when greater than zero |
| `min_eval_episodes` | Minimum episodes required in an evaluation window; currently defaults to 10 in `StageThreshold` |
| `required_consecutive` | Consecutive evaluations that must satisfy every enabled criterion |

For the SB3 curriculum, every enabled criterion must pass in the same evaluation
window, the window must contain at least 10 episodes (the current
`StageThreshold.min_eval_episodes` implementation default), and this result must
repeat for `required_consecutive` evaluations before the stage advances early.
If the `timesteps` budget runs out first, the stage advances anyway. Consult the
generated model pages or the TOML files for current values rather than relying
on a static table here.

JAX/MJX does not currently reproduce that decision loop. The CLI curriculum
trains the configured stage and performs one reward-only gate afterward. The JAX
notebook evaluation helper checks the enabled reward, episode-length,
forward-velocity, and success-rate thresholds once, but does not require
consecutive passes. See [JAX/MJX Training](jax.md#three-stage-task-sequence).

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

> **Systematic sweeps:** Use `notebooks/ray_tune_sweep.ipynb` for a Colab/Google
> Drive workflow, or see [Hyperparameter Sweeps](sweeps.md) for the Vertex AI
> workflow.

## Tips

1. **Start from the committed config** — Treat it as a reproducible baseline, not a validated optimum
2. **Compare algorithms under the same protocol** — The published historical runs do not provide a controlled PPO-versus-SAC comparison
3. **Monitor with W&B** — Use `--wandb` to track per-component rewards across stages
4. **Measure throughput** — Compare CPU and GPU performance for your algorithm and parallel-environment count
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

The names "Bite" and "Food Reach" refer to simulation success proxies. In the
Gym/SB3 environments, T-Rex uses contact from a fixed head geom and has no
articulated jaw; Brachiosaurus uses a head-tip distance threshold and does not
require physical food contact. MJX uses site-distance thresholds for both
T-Rex and Velociraptor instead of their Gym/SB3 contact checks; see
[JAX/MJX Training](jax.md#three-stage-task-sequence).

| Symptom | Most likely cause | What to change |
|---|---|---|
| Never triggers terminal event (strike/bite/food) | Sparse-reward exploration stalled | Raise `ppo.ent_coef` to `0.005–0.01`, widen `ppo.clip_range` to `0.15`, tighten `env.prey_distance_range` so the target spawns closer |
| Lingers near target without triggering | Proximity bonuses rewarding hovering | Zero `env.strike_proximity_weight` (or `env.food_head_proximity_weight`), keep the terminal bonus dominant per the current stage config, and use `*_approach_weight` as the approach gradient |
| Forgets locomotion during Stage 3 warm-up | Reward schedule shift too abrupt | Use `curriculum.warmup_timesteps = 300000`, `curriculum.ramp_timesteps = 500000`, `curriculum.warmup_clip_range = 0.02` to anneal changes |
| Learns the behavior but then regresses | Over-entropy or critic drift | Lower `ppo.ent_coef` after convergence, narrow `ppo.clip_range` to `0.1` |
| `strike_bonus` signal not dominating | Discounted future alive-reward too large | Ensure `env.alive_bonus = 0` in stage 3 and that `strike_bonus >> gamma^H · per_step_reward` |

### Algorithm-specific notes

**PPO.** Treat the `learning_rate` / `learning_rate_end` schedule as a primary
tuning lever. Choose `n_steps`, `n_envs`, and `batch_size` together so rollout
batches divide cleanly into minibatches; Stable-Baselines3 warns when they do
not. Tune `clip_range` and entropy against the stage's observed stability rather
than assuming one fixed progression across species.

**SAC.** Automatic entropy tuning is available through `ent_coef = "auto"`.
Tune `train_freq`, `gradient_steps`, and `buffer_size` together: their useful
values depend on the species, stage, parallel-environment count, and memory
budget.

**JAX / MJX.** `num_envs × rollout_len` determines the effective rollout
buffer and is a major memory lever. Reduce one or both values if device memory
is tight. `warmup_updates` and `ramp_updates` in `[jax]` mirror the
`warmup_timesteps` / `ramp_timesteps` knobs from `[curriculum]`, but use update
counts. The per-stage `[jax]` section is authoritative; see the
[current config-key reference](jax.md#ppo-hyperparameters-jax) rather than
copying values from a guide.

### A minimal tuning workflow

1. **Baseline.** Run the stage with committed defaults for 2–3 seeds. Record best reward, mean episode length, and any behavioral metrics (success rate, velocity).
2. **Diagnose.** If the run fails, match symptoms against the tables above. Do not change more than one group of knobs per run.
3. **Narrow.** For promising directions, launch a Ray Tune sweep over 3–5 candidate values using `notebooks/ray_tune_sweep.ipynb`. Use the ASHA scheduler to prune early.
4. **Promote.** Commit the winning values back to the TOML with a trailing comment explaining why (see existing configs for the house style — e.g. `# Setting 4 sweep: ...`).
5. **Regress-test.** Re-run the full curriculum end-to-end on the winning config before committing. A Stage 1 change often degrades Stage 3.

For systematic multi-parameter sweeps, see [Hyperparameter Sweeps](sweeps.md).
