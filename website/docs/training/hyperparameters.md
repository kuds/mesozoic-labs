---
sidebar_position: 3
---

# Hyperparameters

Guide to tuning hyperparameters for training experiments.

## Config Files

All hyperparameters are defined in TOML config files under `configs/<species>/<embodiment>/`, one directory per model (a species and an embodiment, `anatomical` or `robot`). Every model commits a `stages.toml` manifest (schema `mesozoic.stage-manifest/v2`) that names its stage config files and declares the recipe graph over them. Each stage config is named by its stage, with no number. The anatomical Velociraptor, Brachiosaurus and Dibothrosuchus have three stage configs; the anatomical T-Rex and both Compsognathus models have four:

```
configs/
├── velociraptor_mongoliensis/
│   └── anatomical/
│       ├── stance.toml
│       ├── locomotion.toml
│       ├── strike.toml
│       └── stages.toml
├── tyrannosaurus_rex/
│   └── anatomical/
│       ├── stance.toml
│       ├── recovery.toml
│       ├── locomotion.toml
│       ├── behavior.toml
│       └── stages.toml
├── brachiosaurus_altithorax/
│   └── anatomical/
│       ├── stance.toml
│       ├── locomotion.toml
│       ├── food_reach.toml
│       └── stages.toml
├── dibothrosuchus_elaphros/
│   └── anatomical/
│       ├── stance.toml
│       ├── locomotion.toml
│       ├── snap.toml
│       └── stages.toml
└── compsognathus_longipes/
    ├── anatomical/
    │   ├── stance.toml
    │   ├── recovery.toml
    │   ├── locomotion.toml
    │   ├── behavior.toml
    │   └── stages.toml
    └── robot/
        ├── stance.toml
        ├── recovery.toml
        ├── locomotion.toml
        ├── behavior.toml
        └── stages.toml
```

Each model directory also holds its behavior recipes under `behaviors/`, and the
Compsognathus models their `recovery_calibration.json`.

Each stage's config has `[stage]`, `[env]` and `[curriculum]` sections plus the algorithm sections it supports: `[ppo]` and `[sac]` for every stage except the T-Rex `recovery`, which is PPO-only. A stage TOML file declares these tables itself or inherits any of them but `[curriculum]` from another stage of its model through a top-level key written before the first table, `extends = { stage = "stance", tables = ["stage", "env", "ppo"] }`: each listed table is the parent's, with the file's own keys overriding in place or appended, one level deep (a nested table such as `[ppo.policy_kwargs]` is replaced whole), and a parent that itself extends is refused. Each `recovery.toml` extends its `stance.toml` this way; the T-Rex one does not list `sac`. The loader rejects any other table, `[jax]` included: the JAX/MJX backend was retired (decision D-D17). `stages.toml` carries no hyperparameters. It records the manifest schema and, per `[[stages]]` entry in manifest order: the stage `id`, its `config` file, an optional `legacy_number` (how integer stage references resolve), `warm_start_from` (the id of an earlier entry the node initialises from; absent means root), `deliverable = true` (its certified checkpoint is a published policy) and a `recipe` label (`stand`, `recover`, `walk` or `hunt`, never a stage id; a label resolves to its deepest deliverable in manifest order). Recipes are derived from the edges, never declared in a second table. Ids are an open vocabulary matching `^[a-z][a-z0-9_]*$`, with `stance`, `recovery`, `locomotion` and `behavior` reserved. See [Behavior Recipes](recipes.md#the-stage-manifest).

## Per-Stage Hyperparameters

**Each stage has its own `[ppo]` and `[sac]` sections**, declared in its TOML
file or inherited through `extends`; the T-Rex `recovery` has no `[sac]`. When
the `curriculum` command advances, it loads the next TOML file and
re-initialises the algorithm with that stage's settings; the next node's
weights come from its declared `warm_start_from` parent's handoff checkpoint,
loaded under `initialize_next_stage`. Values differ across models and change
as experiments evolve, so the stage TOML files are the authoritative source.

Every trained node records a `hyperparameters_sha256` in the `run` block of
its `stage_config.json`: a digest over the stage's `[ppo]` or `[sac]` table
plus its `warmup_` / `ramp_` shaping keys, independent of key order and
untouched by env kwargs or gate thresholds. A `label` is recorded beside it
when `--label TEXT` (`train` or `curriculum`) or the notebook's `RUN_LABEL` is
set. Both are copied into the run's `provenance.deliverables` records and
tagged onto the W&B run as `hp:<12 hex>` and `label:<text>`, which is how two
variants of one node are told apart later.

The broad curriculum intent is balance first, locomotion second, and a
species-specific simulator task third. Reward weights shift with that intent;
do not infer current values from a copied range in this guide. The generated
[model pages](/docs/models/velociraptor_mongoliensis/anatomical) show the current stage names, budgets,
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
| `symmetry` (+ `symmetry_kappa`, `symmetry_sigma_floor`, `symmetry_mirror_critic`) | Optional mirror-symmetric PPO, off unless set; see [Mirror Symmetry](mirror-symmetry.md) |

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
| `min_avg_forward_vel` | Optional minimum mean forward velocity; enabled when greater than zero (`reward_and_length/v1` only) |
| `min_success_rate` | Optional minimum episode success rate; enabled when greater than zero (`reward_and_length/v1` only; retired for the T-Rex hunt, which is gated on the bound below) |
| `min_success_lcb` | `task_success/v1` only: the one-sided 95% Clopper-Pearson lower bound on per-episode task success must clear this bar (T-Rex hunt: 0.5, provisional — 20/30 clears it, 19/30 does not); `min_avg_reward` is then a collapse rail, never the gate |
| `min_eval_episodes` | Minimum episodes required in an evaluation window; defaults to 10 in `StageThreshold` except for `task_success/v1`, where it is REQUIRED (the bound's power is a function of the declared n; T-Rex hunt: 30) |
| `required_consecutive` | Consecutive evaluations that must satisfy every enabled criterion |

For the SB3 curriculum, every enabled criterion must pass in the same evaluation
window, the window must contain at least `min_eval_episodes` episodes (10 by
default, the `StageThreshold.min_eval_episodes` implementation default; the
declared value for a `task_success/v1` stage), and this result must
repeat for `required_consecutive` evaluations before the stage advances early.
If the `timesteps` budget runs out first, the node's `gate_verdict.json`
records a failure and the `curriculum` command stops before the next node
rather than training it from an uncertified parent. Consult the generated
model pages or the TOML files for current values rather than relying on a
static table here.

## Overriding Hyperparameters from the CLI

Use `--override` to change TOML values without editing files — useful for hyperparameter sweeps. Keys use dot notation; values are auto-cast to `int`, `float`, or `str`:

```bash
# Override learning rate and entropy coefficient for all stages
python -m environments.shared.train --species velociraptor_mongoliensis --embodiment anatomical \
  train --stage 1 --override ppo.learning_rate=1e-3 ppo.ent_coef=0.02 env.alive_bonus=5.0

# Works with curriculum too — applies to ALL stages
python -m environments.shared.train --species velociraptor_mongoliensis --embodiment anatomical \
  curriculum --override ppo.learning_rate=2e-4
```

Supported key prefixes:

| Prefix | Overrides |
|--------|-----------|
| `ppo.X` | `ppo_kwargs[X]` |
| `sac.X` | `sac_kwargs[X]` |
| `env.X` | `env_kwargs[X]` (reward weights, episode settings) |

For stage-scoped overrides, prefix with the stage's legacy number or its id: `1.ppo.learning_rate=3e-4 2.ppo.learning_rate=1e-4`, or `recovery.ppo.learning_rate=1e-4` on a `train --stage recovery` run. Plain `section.key=value` still applies to all stages.

### Experimenting on a node that already passes

When a run is trunked (`curriculum --trunk-from`, or the notebook's
`TRUNK_FROM`), whether an edited node is reused or retrained depends on which
block you edited. The task digest a verdict is bound to covers the effective
`[env]` block, the plant and the push schedule — not `[ppo]` / `[sac]`, the
`timesteps` budget, or the gate thresholds.

1. **An algorithm-block edit** (`[ppo]`, `[sac]`, or a `warmup_` / `ramp_`
   shaping key) leaves the digest unchanged, so a trunked run reuses the old
   certified checkpoint unless the node is the run's target or
   `--retrain-from <id>` / `RETRAIN_FROM` covers it. The reuse prints a warning
   naming the differing dotted keys (`ppo.learning_rate`,
   `shaping.warmup_timesteps`) and pointing at the retrain knob; it is never a
   refusal.
2. **An `[env]` edit** changes the digest: the old checkpoint is refused by the
   reuse rule and the node and everything below it retrain.
3. **A gate-threshold edit** leaves the digest unchanged but changes the
   gate digest the verdict records (`gate_sha256`, decision D-A22): reuse
   rule 7 refuses the old verdict naming the thresholds that differ, and a
   trunked run then trains the node itself. The remedy is a re-judge under
   the current gate — the notebook JUDGE branch / `generate_stage_artifacts`,
   or `backfill_gate_verdict.py --force --gate current` for a
   `reward_and_length/v1` directory — never a retrain.

A variant is a new run directory: writing into a stage directory that already
holds `stage_config.json` or `gate_verdict.json` is refused unless the load is
an explicit same-stage resume. It is also a trunk for everything it holds:
`--trunk-from` follows a run's own stage directories first and, for a node
it only reused, its `ancestors/<stage_id>/` record to the run that certified
the node, so a variant that reused stance hands stance on to a later run as
the original run's checkpoint. See
[Behavior Recipes](recipes.md#experiments-on-a-node-that-already-passes).

## Tips

1. **Start from the committed config** — Treat it as a reproducible baseline, not a validated optimum
2. **Compare algorithms under the same protocol** — The published historical runs do not provide a controlled PPO-versus-SAC comparison
3. **Monitor with W&B** — Use `--wandb` to track per-component rewards across stages
4. **Watch a run mid-training** — Every stage directory holds `progress.json`, rewritten after each evaluation: steps against the budget, steps per second and the projected finish, the latest evaluation and how its episodes ended, the posture against the stage's bars (centre of pressure, sole tilt and corner lift, per foot), the checkpoints so far and any warnings. On Colab, open it from Drive at `MyDrive/mesozoic-labs/logs/<species>/<embodiment>/<algorithm>/<run>/<stage_dir>/progress.json` while the notebook trains; the full per-evaluation series are in `gate_progress.npz` beside it. A line starting `PostureDrift` in the training output means that on two evaluations in a row, half or more of the evaluation episodes that reached the horizon stood over the stage's fore-aft centre-of-pressure bar; training continues. An evaluation in which no episode reached the horizon measures no posture and breaks the streak, so a seed whose episodes all fall gets no `PostureDrift` line; its falls show in `progress.json`'s `evaluation.falls`. A line starting `PostureTrend` comes earlier: on two evaluations in a row, at least a tenth of those episodes stood over the bar, and the feet's signed centre of pressure is moving toward the edge it is on. It is a prompt to look at the run, not a sign the stage will fail, and training continues. Beside `progress.json`, `eval_episodes.csv` has one row per evaluation episode (its return, how it ended, its posture, balance and floor forces) and `train_episodes.csv` one row per training episode, so the falls the policy learns from can be read too. When the stage ends, `figures/gate_progress.png` plots the evaluations' reward, centre of pressure, over-bar fractions and leg offsets against steps; to plot a run mid-training, download its stage directory and run `python environments/shared/scripts/plot_gate_progress.py <stage_dir>`, which writes the figure to the current directory, never into the run. The schema is in `docs/TRAINING_ARTIFACT_PRESENTATION.md` in the repository
5. **Measure throughput** — Compare CPU and GPU performance for your algorithm and parallel-environment count
6. **Increase timesteps for stage 3** — The sparse terminal reward (strike/bite/food) often needs more samples to converge

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
| Max forward speed exceeds physical reasonableness | Reward uncapped | Set `env.forward_vel_max` (e.g. `3.0` for the Velociraptor, `1.5` for the Brachiosaurus) |

### Stage 3 — Behavior (Strike / Bite / Food Reach)

The names "Bite" and "Food Reach" refer to simulation success proxies. In the
Gym/SB3 environments, T-Rex uses contact from a fixed head geom and has no
articulated jaw; Brachiosaurus uses a head-tip distance threshold and does not
require physical food contact.

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

### A minimal tuning workflow

1. **Baseline.** Run the stage with committed defaults for 2–3 seeds. Record best reward, mean episode length, and any behavioral metrics (success rate, velocity).
2. **Diagnose.** If the run fails, match symptoms against the tables above. Do not change more than one group of knobs per run.
3. **Narrow.** For promising directions, train 3–5 candidate values as separate runs, each changing one value with `--override` (see [Overriding Hyperparameters from the CLI](#overriding-hyperparameters-from-the-cli)), and compare them with the baseline under the same protocol.
4. **Promote.** Commit the winning values back to the TOML with a trailing comment explaining why (see existing configs for the house style — e.g. `# Setting 4 sweep: ...`). The edit moves the stage's digests, so regenerate the committed digest snapshot in the same commit with `python -m environments.shared.harnesses.digest_snapshot --block-optional-backends --write` (CI's plant-contract job checks it).
5. **Regress-test.** Re-run everything below the edited node on the certified trunk before committing: `curriculum --trunk-from <certified run> --retrain-from <edited node> --output-dir <new run>` retrains that node and every node after it. An `[env]` edit retrains from that node down automatically, since it changes the task digest. A stance change often degrades the behavior node. The regress-test run reuses the trunk's ancestors above the edited node and can pass them on: a later run trunked from it resolves each reused node through its ancestor record to the run that certified it, one machine-visible run directory away, so later behaviors can trunk from the regress-test run directly.
