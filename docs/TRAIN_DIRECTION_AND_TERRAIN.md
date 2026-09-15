# Direction following and randomized terrain pilots

These opt-in T. rex experiments add desired-heading/speed commands and gentle
randomized ground to the current r13 walker. They run on SB3/PPO and retain the
64-input, 15-action layout. They are pilot tasks with separate artifact identities;
the canonical locomotion, hunting, and MJX environments remain unchanged.

## What is implemented

- A desired world heading and speed become body-relative forward speed and a
  bounded turning rate. The lateral command starts at zero. Commands change on a
  seeded schedule or through `env.set_direction(heading_radians, speed_m_per_s)`.
- Direction-only, direction-and-speed, terrain-contact adaptation, gentle-terrain,
  and combined recipes.
  Turns begin within ±30°, with a 0.3 rad/s turning cap. Speed recipes include
  stops; zero speed masks the heading objective and checks actual motion.
- A finite 70 × 70 m heightfield in the recipes, with smooth grades capped at 3° and nominal
  15 mm roughness. Each run gets a new recorded seed by default. Each episode
  receives a small variation of that run's course. Explicit seeds replay it.
- A 3 m flat spawn radius contains the whole animal, including its tail. Terrain
  recipes mix in 25% original-plane episodes. Pelvis/head/skull checks use local ground
  height. Neck contact is detected in a separate collision-only model, since the
  canonical neck cannot support the animal physically.
- Preparation of the current walker zeros only the newly active command input
  columns and matching optimizer moments, preserves other weights/statistics,
  and verifies initial action/value equivalence. Command normalization always
  passes through its fixed scales. Adaptation between behavior recipes retains
  the command weights already learned.

This first increment covers smooth solid terrain. Steps, large obstacles,
slippery patches, adaptive difficulty promotion, terrain look-ahead sensing, and
the main training notebook's behavior selector are follow-on work. The recipes
are runnable through the dedicated command below; their budgets are proposals,
not evidence that the behaviors have been learned.

## Run a short training check

Install the training dependencies from the repository root:

```bash
pip install -e '.[train,test]'
```

Use a canonical current-interface T. rex **locomotion** PPO checkpoint and its
matching, identity-tagged normalization file. The selected walker from run
`20260914_123816` is the reviewed starting point. Neither artifact is included
in the repository.

```bash
python -m environments.trex.scripts.train_behaviors \
  --config configs/behavior_pilots/trex_follow_direction.toml \
  --checkpoint /path/to/walker.zip \
  --vecnormalize /path/to/walker_vecnormalize.pkl \
  --output /path/to/runs/follow-smoke \
  --steps 4096 --eval-episodes 5 --seed 42
```

The output directory must be new or empty. This performs one 4,096-step rollout
for the reviewed walker, updates PPO, saves a matched bundle, and evaluates it.
PPO rounds requested training steps up to complete rollouts; `run.json` records
both requested and actual additional steps. A short check proves the pipeline
works; use the full recipe budget for an actual learning pilot.

Omit `--steps` to use the selected recipe's proposed budget. Omit `--seed` for a
fresh random run seed; it is printed and saved. `--eval-only` performs no learning.
The runner uses CPU SB3 and reports measured training time; it submits no cloud
job. It preserves the parent's network and rollout settings, uses a constant
learning rate of 5e-5, and limits PPO clipping to 0.02 for the first 100,000
adaptation steps before restoring 0.2. The warmup anchor survives exact-task
resume. Reward normalization adapts during training and is disabled for scoring.

## Train the next behavior

| Recipe | Purpose | Proposed additional steps |
|---|---|---:|
| `trex_follow_direction.toml` | Turn while retaining cruise speed | 3M |
| `trex_follow_direction_speed.toml` | Vary speed, stop and resume | 3M |
| `trex_terrain_contact.toml` | Adapt the gait to flat-heightfield foot contacts | 300k |
| `trex_gentle_terrain.toml` | Maintain a heading across randomized gentle ground | 2M |
| `trex_combined_terrain.toml` | Change heading/speed on gentle ground | 3M |

Start terrain from the learned follower, after checking its flat-ground behavior.
First use the contact-adaptation recipe: a paired preflight found that the
prepared original walker survived 5/5 ten-second original-plane trials but
0/5 flat-heightfield trials. A heightfield changes foot contacts even with all
heights zero. Measure and train that transition before adding slopes. Keep
original-plane retention episodes as a separate control.
Use `--adapt` when changing to a compatible recipe. This preserves trained
command connections instead of zeroing them a second time:

```bash
python -m environments.trex.scripts.train_behaviors \
  --config configs/behavior_pilots/trex_terrain_contact.toml \
  --checkpoint /path/to/runs/follow/model.zip \
  --vecnormalize /path/to/runs/follow/vecnormalize.pkl \
  --output /path/to/runs/terrain-contact --adapt
```

Use `--resume` with the same recipe to continue the exact task. With no `--steps`,
resume trains only the remaining recipe budget. An explicit `--steps` means
**additional** steps for this invocation. Resume checks PPO settings as well as
the environment; an explicit short run does not reduce the recipe's stage target.
Resume and adaptation verify the
saved `bundle.json` hashes before loading. Configuration changes that alter
command scaling, the heading adapter, control timing, the animal, or source
semantics are rejected. Terrain parameters and compatible command schedules can
change through explicit adaptation.

Matched snapshots are saved about every 100,000 steps after completed updates,
under `checkpoints/`; `latest_checkpoint.json` identifies the newest one. A
keyboard interruption also saves a bundle. Resume starts fresh episodes; it
preserves learning progress but does not promise identical interrupted physics
trajectories or random-number state.

## Reproduce a terrain episode

Every reset is recorded in `training_episodes.jsonl`. Each terrain manifest
contains the generator configuration, run seed, episode index, measured maximum
grade, height range, and a hash of the actual float32 height samples.

```python
from environments.shared.terrain import TerrainConfig, generate_terrain

terrain = generate_terrain(
    TerrainConfig(**saved_manifest["config"]),
    run_seed=saved_manifest["run_seed"],
    episode_index=saved_manifest["episode_index"],
)
assert terrain.manifest()["samples_sha256"] == saved_manifest["samples_sha256"]
```

`TRexBehaviorEnv.reset(seed=S)` repeats the complete reset and resets its episode
counter. `reset()` advances the episode stream. Terrain and command randomness
use independent streams, so command sampling does not change the map. Run seed
and reset seed both contribute to the course seed. Evaluation uses a separate
seed stream and saves each map's replay recipe.

The surface query uses MuJoCo's triangular interpolation, verified against
vertical rays. It does not use bilinear interpolation. MuJoCo's general
`mj_geomDistance` is unsuitable for heightfield clearance; the environment
explicitly refuses that generic diagnostic on terrain. Spawn placement uses
the canonical plane under the flat apron and verifies actual hfield contacts.

## Read the results

The output bundle includes:

- `model.zip`, `vecnormalize.pkl`, and `bundle.json`: matched artifacts and hashes.
- `run.json`: resolved environment identity, source hashes, training settings,
  starting checkpoint, actual step count, duration, and evaluation summary.
- `progress.csv`: PPO training diagnostics.
- `training_episodes.jsonl`: every actual training/evaluation reset recipe.
- Evaluation summaries and per-episode command/step logs with target heading,
  emitted commands, measured speed/turning, stopping, and terrain information.

Judge heading acquisition, speed error, sustained stopping, full-horizon
survival, and terrain exposure together. Radial distance alone is only a course
diagnostic: walking the wrong direction, falling after progress, or remaining on
the spawn apron does not demonstrate the requested behavior. Planned commands
that never occur because of an early fall remain visible in evaluation.

Keep the original walker frozen and compare paired flat-ground evaluations.
Run left/right turns and unseen terrain seeds before increasing slope or
roughness. The pilot evaluator reports measurements; it does not mint a
canonical gate pass or publish a trained capability.
