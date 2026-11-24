---
sidebar_position: 3
---

# Hyperparameters

Guide to tuning hyperparameters for training experiments.

## Config Files

All hyperparameters are defined in TOML config files under `configs/<species>/<embodiment>/`, one directory per model (a species and an embodiment, `anatomical` or `robot`). Every model commits a `stages.toml` manifest (schema `mesozoic.stage-manifest/v2`) that names its stage config files and declares the recipe graph over them. Each stage config is named by its stage, with no number. The anatomical Velociraptor, Brachiosaurus, Dibothrosuchus and Allosaurus have two stage configs; the anatomical T-Rex and both Compsognathus models have three (the hunt stage that followed locomotion on every model but Allosaurus was removed by decision D-D59):

```
configs/
├── velociraptor_mongoliensis/
│   └── anatomical/
│       ├── stance.toml
│       ├── locomotion.toml
│       └── stages.toml
├── tyrannosaurus_rex/
│   └── anatomical/
│       ├── stance.toml
│       ├── recovery.toml
│       ├── locomotion.toml
│       └── stages.toml
├── brachiosaurus_altithorax/
│   └── anatomical/
│       ├── stance.toml
│       ├── locomotion.toml
│       └── stages.toml
├── dibothrosuchus_elaphros/
│   └── anatomical/
│       ├── stance.toml
│       ├── locomotion.toml
│       └── stages.toml
├── compsognathus_longipes/
│   ├── anatomical/
│   │   ├── stance.toml
│   │   ├── recovery.toml
│   │   ├── locomotion.toml
│   │   └── stages.toml
│   └── robot/
│       ├── stance.toml
│       ├── recovery.toml
│       ├── locomotion.toml
│       └── stages.toml
└── allosaurus_fragilis/
    └── anatomical/
        ├── stance.toml
        ├── locomotion.toml
        └── stages.toml
```

Each model directory also holds its behavior recipes under `behaviors/`, and the
Compsognathus models their `recovery_calibration.json`.

Each stage's config has `[stage]`, `[env]` and `[curriculum]` sections plus the algorithm sections it supports: `[ppo]` and `[sac]` for every stage except the T-Rex `recovery`, which is PPO-only. A stage may also declare a `[commands]` table, its live command task (see [Commands](#commands)); no committed stage declares one yet. A stage TOML file declares these tables itself or inherits any of them but `[curriculum]` from another stage of its model through a top-level key written before the first table, `extends = { stage = "stance", tables = ["stage", "env", "ppo"] }`: each listed table is the parent's, with the file's own keys overriding in place or appended, one level deep (a nested table such as `[ppo.policy_kwargs]` is replaced whole), and a parent that itself extends is refused. Each `recovery.toml` extends its `stance.toml` this way; the T-Rex one does not list `sac`. The loader rejects any other table, `[jax]` included: the JAX/MJX backend was retired (decision D-D17). `stages.toml` carries no hyperparameters. It records the manifest schema and, per `[[stages]]` entry in manifest order: the stage `id`, its `config` file, an optional `legacy_number` (how integer stage references resolve), `warm_start_from` (the id of an earlier entry the node initialises from; absent means root), `deliverable = true` (its certified checkpoint is a published policy) and a `recipe` label (`stance`, `recover` or `walk`; a label equals a stage id only when it resolves to that stage, as `stance` does, `stand`, the stance label until decision D-D38, is refused naming `stance`, and `hunt`, removed with the hunt nodes by decision D-D59, is refused naming `walk`; a label resolves to its deepest deliverable in manifest order). Recipes are derived from the edges, never declared in a second table. Ids are an open vocabulary matching `^[a-z][a-z0-9_]*$`, with `stance`, `recovery`, `locomotion` and `behavior` reserved. See [Behavior Recipes](recipes.md#the-stage-manifest).

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
set. Both are copied into the run's `provenance.deliverables` records, which
is how two variants of one node are told apart later.

The broad curriculum intent is balance first, locomotion second, and a
species-specific simulator task third. Reward weights shift with that intent;
do not infer current values from a copied range in this guide. The generated
[model pages](/docs/models/velociraptor_mongoliensis/anatomical) show the current stage names, budgets,
and advancement gates read from the same configs.

## Commands

Every observation ends with a 3-wide body-relative command (forward speed,
lateral speed and yaw rate, each pre-scaled to [-1, 1]). A stage without a
`[commands]` table runs `command_mode = "none"`: the command is constant zero,
nothing is drawn for it and nothing scores it. Every committed stage runs this
way today. A stage turns its command on with a `[commands]` table (decision
D-D60), for example:

```toml
[commands]
mode = "heading_and_speed"       # or "heading", which holds cruise_speed
cruise_speed = 0.5
speed_range = [0.25, 0.5]
stop_probability = 0.15          # drawn at reset: with no turns and a long switch interval, a whole-episode stand
turn_increment_max = 0.0
switch_interval_s = 100.0
initial_heading_offset_max = 0.26
tracking_weight = 2.0
```

- **Keys.** `mode` is required and must be a live mode; every other key is a
  field of `DirectionCommandConfig` (`environments/shared/direction_commands.py`).
  An unknown key, a value the config refuses and `"heading"` with a speed range
  or stops are refused, naming the file. `[env]` never names `command_mode` or
  `command_config`. The stance never declares the table, its own or inherited:
  its six-heading certificate and the mirror refuse a live command. `extends`
  inherits `[commands]` key by key, like `[env]`, so a node that widens its
  parent's command distribution states only the keys it changes.
- **Task.** The table becomes the stage's `command_mode` and `command_config`
  env kwargs. `stage_config.json` records every field, and the task
  fingerprint gains a `command` section (the command controller's manifest),
  which training checks against the env it builds. A `"none"` stage's
  fingerprint has no `command` section, so adding the table to a stage is a
  new task.
- **Reward.** While the command is live, the environment adds
  `tracking_weight` times a Gaussian of the body-frame velocity error and the
  yaw-rate error (`reward_tracking`, included in `reward_total`), with the
  sigmas `tracking_velocity_sigma` and `tracking_yaw_rate_sigma`. The step's
  `info` carries the command state, the tracking metrics judged against the
  five `tracking_*_tolerance` fields, `heading_error_rad` and
  `tracking_dwell_s` (consecutive seconds in tolerance, restarted by each new
  command). An unset speed sigma or tolerance scales with `speed_scale`.
- **Randomness.** Each episode's command schedule is seeded from its own
  stream (the reset seed, the episode index since that seed and a fixed
  domain word), never from the environment's reset stream, so a live stage's
  joint, velocity and push draws equal those of the same stage without
  commands. `initial_heading_offset_max` (radians, at most π) turns the first
  requested heading by a uniform draw from that stream; 0 draws nothing.
- **Normalization.** The trainer (PPO and SAC, the command line and the
  notebook) holds the command slice of the `VecNormalize` statistics at mean 0
  and variance 1 in every stage, so a normalised command is the raw command
  (while a live stage trains, each step's observation is normalised just
  before the hold, so it can be about `n_envs / count` away), and every
  sidecar it saves carries that identity. SAC's replay buffer stores the raw
  observations, terminal ones included. A sidecar whose slice is not at
  identity (one saved before this rule) is refused by a live stage and
  re-imposed, with a warning, by a stage without commands, whose zero command
  normalises to zero either way.
- **Warm starts.** A live stage warm-started from a parent that trained
  without commands (a stance) starts with its policy's command input weights
  zeroed, and the policy is proven unchanged on zero commands and blind to
  live ones before the first update. A live parent keeps its command weights
  and statistics, so a node that widens its parent's command distribution
  continues it exactly.

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
| `min_eval_episodes` | Minimum episodes required in an evaluation window; defaults to 10 in `StageThreshold`, and required by the floor-truth stance gates (the bound's power is a function of the declared n) |
| `required_consecutive` | Consecutive evaluations that must satisfy every enabled criterion |

The task-success gates are gone: `min_success_rate`, the `task_success/v1`
gate kind and its `min_success_lcb` bar were removed with the hunt stages, their
only users (decision D-D59). A config or a recorded stage that still names one
is refused with a message naming that decision.

For the SB3 curriculum, every enabled criterion must pass in the same evaluation
window, the window must contain at least `min_eval_episodes` episodes (10 by
default, the `StageThreshold.min_eval_episodes` implementation default), and
this result must repeat for `required_consecutive` evaluations before the
stage advances early.
If the `timesteps` budget runs out first, the node's `gate_verdict.json`
records a failure and the `curriculum` command stops before the next node
rather than training it from an uncertified parent. Consult the generated
model pages or the TOML files for current values rather than relying on a
static table here.

### Command-tracking gate (`command_tracking/v1`)

A stage with a live command (a `[commands]` table, see [Commands](#commands))
is judged by the `command_tracking/v1` gate kind (decision D-D61). No
committed stage declares it yet: the walk node switches to it in the next
phase, and the later follow nodes use it too. The gate is judged once, after
the stage, on a fixed panel of deterministic episodes of the handoff
checkpoint. The in-training curriculum manager refuses the kind, so such a
stage never advances early and always trains its full `timesteps` budget.
The string key `profile` selects the rules:

- **`walk`**, the walk node's. Each panel episode belongs to the family its
  reset command puts it in: **cruise** (a non-zero speed) or **stand** (an
  exactly zero command for the whole episode). A cruise episode succeeds
  when it reaches the horizon, settles its initial heading error within
  `1 s + |error| / yaw_rate_max` and holds it for 1 s, keeps
  `min_cruise_tracking_fraction` of its steps in tolerance after that
  settle, makes `min_cruise_speed_ratio` of the requested speed along the
  requested heading after 2 s, and passes the gait clause below. A stand
  episode succeeds when it reaches the horizon, settles its stop and holds
  the stop criterion (speed and yaw rate within the stop tolerances) for
  `min_stand_hold_fraction` of its steps after 1 s. A family with fewer
  episodes than its minimum fails the gate, and so does an episode whose
  command changes mid-episode.
- **`follow`**, the follow nodes': the behavior certificate's statistics
  (survival and success bounds; the tracking, event, stop and turn settle
  fractions; the stop and turn exposure minimums), the gait clause, and
  optionally a paired comparison with a reference policy (the walker) on
  the same seeds, which fails closed when the reference panel is missing.
  It pairs follow success on the follow panel, so it bounds how much worse
  than the walker the follower follows; it is not a test that straight
  walking was kept.

Every bar is a required `[curriculum]` key with no default in code, so the
gate digest covers every number a verdict depends on. A block is checked
when the config loads: an unknown key, the other profile's key or a value
outside its range stops the run. Every starting value below is
**provisional**, to be frozen from the first pilot's panel; a verdict reached
under a provisional bar is judged again, not retrained, when the bar moves.

| Walk key | Starting value | Meaning |
|---|---|---|
| `profile` | `"walk"` | Selects the walk profile |
| `min_cruise_episodes` | 20 | Cruise episodes on the panel |
| `min_stand_episodes` | 5 | Stand episodes on the panel |
| `min_cruise_survival_lcb` | 0.80 | Lower 95% bound (exact Clopper-Pearson) on the cruise episodes reaching the horizon; at 20 episodes only 20 of 20 clears it |
| `min_cruise_success_lcb` | 0.60 | Lower 95% bound on the cruise episodes that succeed; at 20 episodes 17 of 20 clears it |
| `min_cruise_tracking_fraction` | 0.60 | A cruise episode's in-tolerance fraction after its heading settle |
| `min_cruise_speed_ratio` | 0.50 | Mean speed along the requested heading after 2 s, per requested speed; a placeholder looser than the design draft's 0.9, since the tracking bar already bounds the speed error |
| `min_stand_hold_fraction` | 0.60 | A stand episode's in-tolerance (stop criterion) fraction after 1 s; at 0.60 a stand may move for 40% of those steps, so the pilot is to freeze it near 1 |
| `min_stand_success_fraction` | 0.60 | Fraction of stand episodes that succeed |
| `max_stand_falls` | 0 | Stand episodes that may end before the horizon |

| Follow key | Starting value | Meaning |
|---|---|---|
| `profile` | `"follow"` | Selects the follow profile |
| `min_follow_episodes` | 20 | Valid episodes on the panel |
| `min_follow_horizon_s` | 20.0 | An episode with a shorter horizon is not a valid sample |
| `min_follow_survival_lcb` | 0.80 | Lower 95% bound on the episodes reaching the horizon |
| `min_follow_success_lcb` | 0.60 | Lower 95% bound on the episodes that succeed |
| `min_follow_tracking_fraction` | 0.60 | An episode's in-tolerance fraction |
| `min_follow_event_settle_fraction` | 0.60 | An episode's fraction of command events that settle |
| `min_follow_stop_settle_fraction` | 0.60 | Per episode, and over the episodes that meet a stop |
| `min_follow_turn_settle_fraction` | 0.60 | Per episode, and over the episodes that meet a turn |
| `min_follow_command_episodes` | 5 | Episodes that must meet a stop (a turn) when the schedule draws them |
| `min_follow_course_fraction` and `follow_course_distance_m` | 0.50, the species' course distance | Optional, declared together: radial progress per course distance |
| `min_follow_reference_delta_lcb` | none | Optional: lower bound on the mean per-seed success difference against the reference policy |

The **gait clause** is required in both profiles, so an episode that tracks
the command is also judged on its footfalls, not on the pelvis alone. The
panel records every episode's floor contacts, and from 1 s on reduces them
to alternation, simultaneous touchdowns, hop flight, lead swaps, limb duty
and foot-on-foot and phantom support (stand episodes are not judged on
gait). A lead swap counts only once the feet's fore-aft offset has passed
3% of the leg length, so a two-footed hop with its feet side by side reads
no lead swaps even when its feet land more than 20 ms apart (which reads as
alternating footfalls). `gait_limbs` names the animal and the bars it must
declare; an unmeasured metric fails its bar.

| Gait key | Biped (`gait_limbs = 2`) | Quadruped (`gait_limbs = 4`) |
|---|---|---|
| `min_alternation_index` | required, 0.75 | refused |
| `max_simultaneous_fraction` | required, 0.15 | optional |
| `max_hop_flight_fraction` | required, 0.15 | optional |
| `min_lead_swaps_per_stride` | required, 1.0 | optional |
| `max_foot_contact_fraction` | required, 0.02 | optional |
| `max_phantom_support_fraction` | required, 0.05 | optional |
| `max_flight_fraction` | optional, 0.05 where the commanded speed is below Froude 0.5 | required, 0.05 |
| `min_limb_duty` | optional | required, 0.20 |
| `min_pair_alternation_index` | refused | required, 0.75 (the lower of the fore and hind pairs) |

The bipeds are the velociraptor, the T-Rex, both Compsognathus models and the
Allosaurus; the quadrupeds are the Brachiosaurus and the Dibothrosuchus, whose
lateral-sequence or trot match is measured and reported but not judged.

What the clause does not refuse: a hop that lands both feet together
fails, as does a slide with every foot down (it makes no footfall), but a
staggered hop whose feet swap their lead past the 3% band is refused only by
the optional flight cap, and a skid broken by brief alternating lifts is not
refused at all (slip is reported, not judged). On a quadruped, a bound whose
two fore feet (and two hind feet) land more than 20 ms apart with no flight
reads as alternating pairs and clears the three required floors; only the
optional `min_lead_swaps_per_stride` refuses it. The stand family's gait is
not judged, so stepping in place within the stop tolerances passes it; the
stand episodes' recorded alternation and touchdown rate show it. Each
episode record carries the left/right phase (about 0.5 on a walk, near 0 or
1 on a hop) and the stance-foot slip speed for review. The
in-tolerance test is the environment's own, at the tolerances the stage's
`[commands]` table resolves; the optional keys `gate_velocity_tolerance`,
`gate_yaw_rate_tolerance`, `gate_heading_tolerance`,
`gate_stop_speed_tolerance` and `gate_stop_yaw_rate_tolerance` replace one
for the gate only.

After the stage, these files appear in the stage directory:

- `models/handoff_choice.json`: the handoff checkpoint, chosen by the gate
  rather than by reward. `robust_best_model`, `best_model` and the final
  checkpoint are each rolled on a selection panel (seeds from 2042) and
  ranked by passing, then by the profile's success bound. The certificate is
  rolled on other seeds (from 3042), so the choice cannot pick a lucky draw
  of the certification panel; when a rare stand family would make the two
  panels share a seed, no choice is made and the stage fails its gate.
- `command_tracking_report.json`, `command_tracking_report.txt` and
  `command_tracking_panel_selected.csv`: the certification panel, one record
  per episode with its classification. The verdict is derived again from
  those records, and publication derives it again from the CSV.
- `figures/command_tracking.png`: commanded against achieved speed, yaw rate
  and heading error over the panel's first cruise and first stand episode
  (rolled again from `command_tracking_trace.csv`, the same episodes the gate
  judged), with the tolerances, the judged windows and the settle deadline,
  and the tracking fraction over training (the training steps' mean
  in-tolerance flag, which `diagnostics.npz` records for a stage with a live
  command) against the profile's tracking bar.
- `figures/command_gate_progress.png`: each gated statistic of every
  candidate checkpoint on the selection panel, in the order training saved
  them, and of the certified handoff, against its bar, beside the number of
  certification episodes that failed each criterion. Its title is the
  stage's verdict.

The figures belong to the judgement that drew them: figures and a trace an
earlier judgement left are removed first, and none are drawn when the panel
could not be rolled. The command-line `curriculum` command writes the same
report and figures after each such node's budget.

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

A `[commands]` key has no override prefix (`commands.X` is refused as an unknown section): a stage's command task is edited in its TOML.

For stage-scoped overrides, prefix with the stage's legacy number or its id: `1.ppo.learning_rate=3e-4 2.ppo.learning_rate=1e-4`, or `recovery.ppo.learning_rate=1e-4` on a `train --stage recovery` run. Plain `section.key=value` still applies to all stages.

### Experimenting on a node that already passes

When a run is trunked (`curriculum --trunk-from`, or the notebook's
`TRUNK_FROM`), whether an edited node is reused or retrained depends on which
block you edited. The task digest a verdict is bound to covers the effective
`[env]` block, the `[commands]` table, the plant and the push schedule — not
`[ppo]` / `[sac]`, the `timesteps` budget, or the gate thresholds.

1. **An algorithm-block edit** (`[ppo]`, `[sac]`, or a `warmup_` / `ramp_`
   shaping key) leaves the digest unchanged, so a trunked run reuses the old
   certified checkpoint unless the node is the run's target or
   `--retrain-from <id>` / `RETRAIN_FROM` covers it. The reuse prints a warning
   naming the differing dotted keys (`ppo.learning_rate`,
   `shaping.warmup_timesteps`) and pointing at the retrain knob; it is never a
   refusal.
2. **An `[env]` or `[commands]` edit** changes the digest: the old checkpoint is
   refused by the reuse rule and the node and everything below it retrain.
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
3. **Monitor with TensorBoard** — Every stage logs its per-component rewards as `diagnostics/*` scalars; point `tensorboard --logdir` at the run directory
4. **Watch a run mid-training** — Every stage directory holds `progress.json`, rewritten after each evaluation: steps against the budget, steps per second and the projected finish, the latest evaluation and how its episodes ended, the posture against the stage's bars (centre of pressure, sole tilt and corner lift, per foot), the checkpoints so far and any warnings. On Colab, open it from Drive at `MyDrive/mesozoic-labs/logs/<species>/<embodiment>/<algorithm>/<run>/<stage_dir>/progress.json` while the notebook trains; the full per-evaluation series are in `gate_progress.npz` beside it. A line starting `PostureDrift` in the training output means that on two evaluations in a row, half or more of the evaluation episodes that reached the horizon stood over the stage's fore-aft centre-of-pressure bar; training continues. An evaluation in which no episode reached the horizon measures no posture and breaks the streak, so a seed whose episodes all fall gets no `PostureDrift` line; its falls show in `progress.json`'s `evaluation.falls`. A line starting `PostureTrend` comes earlier: on two evaluations in a row, at least a tenth of those episodes stood over the bar, and the feet's signed centre of pressure is moving toward the edge it is on. It is a prompt to look at the run, not a sign the stage will fail, and training continues. Beside `progress.json`, `eval_episodes.csv` has one row per evaluation episode (its return, how it ended, its posture, balance and floor forces) and `train_episodes.csv` one row per training episode, so the falls the policy learns from can be read too. When the stage ends, `figures/gate_progress.png` plots the evaluations' reward, centre of pressure, over-bar fractions and leg offsets against steps; to plot a run mid-training, download its stage directory and run `python environments/shared/scripts/plot_gate_progress.py <stage_dir>`, which writes the figure to the current directory, never into the run. The stage's training history from those two ledgers (its feet, balance, asymmetry, saturation and per-actuator policy spread over the training) is in `figures/training_history/training_history.html`, also written when the stage ends. The schema is in `docs/TRAINING_ARTIFACT_PRESENTATION.md` in the repository
5. **Measure throughput** — Compare CPU and GPU performance for your algorithm and parallel-environment count

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
| Crab-walks sideways | No lateral or heading constraint | Set `env.lateral_penalty_weight` to `0.1`, raise `env.heading_weight` |
| Walks forward but falls frequently | Balance reward zeroed too aggressively | Keep `env.posture_weight ≥ 0.3`, re-enable mild `env.alive_bonus` (`0.3–0.5`) |
| Unrealistic "ice-skating" gait | Symmetry and smoothness under-weighted | Raise `env.gait_symmetry_weight`, raise `env.smoothness_weight` |
| Max forward speed exceeds physical reasonableness | Reward uncapped | Set `env.forward_vel_max` (e.g. `3.0` for the Velociraptor, `1.5` for the Brachiosaurus) |

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
5. **Regress-test.** Re-run everything below the edited node on the certified trunk before committing: `curriculum --trunk-from <certified run> --retrain-from <edited node> --output-dir <new run>` retrains that node and every node after it. An `[env]` edit retrains from that node down automatically, since it changes the task digest. A stance change often degrades the nodes below it. The regress-test run reuses the trunk's ancestors above the edited node and can pass them on: a later run trunked from it resolves each reused node through its ancestor record to the run that certified it, one machine-visible run directory away, so later behaviors can trunk from the regress-test run directly.
