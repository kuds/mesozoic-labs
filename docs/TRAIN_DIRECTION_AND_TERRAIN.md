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
  15 mm roughness in the original sloped template. Localized bumps, shallow
  depressions, and mixed ground have separate templates with nominal 20 mm
  features on an otherwise level surface. Each run gets a new recorded seed by default. Each episode
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
MJX training are follow-on work. The recipes are available in the SB3 notebook
and through the dedicated command below; their budgets are proposals,
not evidence that the behaviors have been learned.

## Use the SB3 training notebook

Open [`notebooks/sb3_training.ipynb`](../notebooks/sb3_training.ipynb). Until
PR #540 is merged, open the notebook from `codex/direction-and-random-terrain`
and set its setup parameter `REPO_REF` to that branch so Colab installs the
matching code. After merge, the default `main` is appropriate.

In the configuration form, select **Tyrannosaurus Rex**, `ALGORITHM = "ppo"`,
and one of these `BEHAVIOR` values:

| Behavior parameter | Experiment |
|---|---|
| `follow_direction` | Fixed-speed heading changes |
| `follow_direction_speed` | Heading, speed, stop and restart |
| `terrain_contact` | Flat-heightfield contact adaptation |
| `sloped_terrain` | The original gently sloped ground |
| `bumps_terrain` | Random localized bumps |
| `depressions_terrain` | Random shallow depressions |
| `mixed_terrain` | Bumps and depressions together |
| `combined_terrain` | Heading/speed changes on gently sloped ground |

The existing `stand`, `walk`, `hunt`, and explicit stage-ID selections retain
the notebook's canonical chain workflow. The new pilots use the dedicated
behavior runner and produce experimental behavior bundles.

Set `PILOT_CHECKPOINT` and `PILOT_VECNORMALIZE` to the matching saved pair.
Choose the loading mode explicitly:

- `PILOT_LOAD_MODE = "prepare"` starts from the current canonical locomotion
  walker and activates its command inputs.
- `"resume"` continues the same behavior recipe from a matched behavior bundle.
- `"adapt"` transfers a learned behavior to a compatible next recipe.

`PILOT_SEED = None` gives each run a fresh recorded seed; an integer repeats
the course. `PILOT_STEPS = None` uses the recipe budget, or its remaining budget
on resume. An explicit step count is additional training. `QUICK_TEST = True`
uses 4,096 steps when no explicit step count is set. `PILOT_EVAL_ONLY = True`
evaluates without training.
Repeating an unchanged selection retains its run ID and seed and refuses to
overwrite existing output. To start another run with the same settings in the
same runtime, set a new `PILOT_RUN_ID`. A new selection or runtime with automatic
IDs/seeds gets a fresh run.

Pilot execution uses one environment on CPU and inherits the checkpoint's PPO
network and rollout settings; canonical `N_ENVS` does not change those settings.
Canonical trunk/widen/retrain controls do not apply to pilots.

The notebook saves pilot outputs under
`logs/trex/ppo/behavior_pilots/<behavior>/<run-id>/`, on Google Drive when mounted
or locally otherwise. `PILOT_RECORD_VIDEO = True` saves and displays each
evaluation video with both heat maps. `PILOT_EVAL_EPISODES` and
`PILOT_VIDEO_FPS` control evaluation count and video rate. New output directories
prevent accidental replacement of previous evidence. Notebook evaluation,
replay, and cleanup use the behavior outputs rather than canonical gate reports.

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
  --config configs/trex/behavior_pilots/trex_follow_direction.toml \
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
| `trex_bumps_terrain.toml` | Cross scattered smooth bumps on level ground | 2M |
| `trex_depressions_terrain.toml` | Cross shallow solid depressions | 2M |
| `trex_mixed_terrain.toml` | Cross a mixture of bumps and depressions | 2M |
| `trex_combined_terrain.toml` | Change heading/speed on gentle ground | 3M |

### Terrain templates and randomization

Templates specify the kind and scale of ground, while seeds vary its layout.
The original `sloped` template combines a broad random grade with small sinusoidal
ripples. On a full-map heat map, its large overall elevation range can hide the
centimetre-scale ripples. The new `bumps`, `depressions`, and `mixed` templates
have no broad slope, so their localized features define the difficulty.

Feature templates start with a nominal 2 cm height or depth and randomized
0.7–1.2 m radii. These are broad, shallow features suitable for the first contact
adaptation pilots. Depressions have solid bottoms; they are not gaps through the
ground. Their positions, sizes, and amplitudes vary by run, with smaller changes
between episodes. The spawn apron stays flat. The final physical surface is
limited to the configured maximum grade; its measured heights and grade are
recorded because overlapping features or grade limiting can change the nominal
amplitude.

Keep each template as a separate evaluation slice so falls on depressions are
not hidden by good results on bumps. After contact adaptation, compare the
individual templates, then the mixed template, before raising feature height or
adding sharper ground. Reserve fixed seeds for repeatable comparisons and use
unseen seeds to check whether the behavior transfers to new layouts. The recipes
do not automatically promote difficulty or establish that any template is mastered.

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
  --config configs/trex/behavior_pilots/trex_terrain_contact.toml \
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

### Save video replays and their terrain maps

Install `pip install -e '.[train,viz]'` and add `--record-video` to the training
command. Each evaluation video automatically saves both terrain heat maps beside
it. `--video-fps` sets the playback rate (25 by default); it does not change the
simulation timestep. The bundled encoder does not require a system FFmpeg install.

To record a previously saved behavior bundle from the same compatible source and
recipe, run evaluation without further training:

```bash
python -m environments.trex.scripts.train_behaviors \
  --config configs/trex/behavior_pilots/trex_mixed_terrain.toml \
  --checkpoint /path/to/runs/mixed/model.zip \
  --vecnormalize /path/to/runs/mixed/vecnormalize.pkl \
  --output /path/to/replays/mixed \
  --eval-only --resume --record-video --eval-episodes 3 --seed 42
```

Each scored episode receives a directory such as
`replays/episode_000_seed_12345/` containing:

- `replay.mp4`: the exact trajectory used for that episode's evaluation.
- `terrain_full_map.png`: the entire physical map with the actual path.
- `terrain_local_map.png`: an 8 m close-up around the final position, with a
  local color range to make small height changes visible. Heights are labelled
  in centimetres; world coordinates are in metres.
- `terrain_and_path.npz`: the exact physics heightfield samples, height grid,
  coordinates, trajectory and frame timing.
- `manifest.json`: template, seeds, terrain hash, episode outcome and file hashes.

The recorder captures the terminal pose and frame before the vectorized
environment resets. It copies the physical heightfield rather than generating
a second map for display, and checks that its sample hash matches the episode's
terrain record. A replay directory is published only once all its files are
complete. Requested video or map failures fail the export visibly.

Original-plane retention episodes save explicitly labelled
`flat_plane_full_map.png` and `flat_plane_local_map.png` reference maps. These
are not presented as sampled heightfields. Video recording remains optional;
requesting it always includes the matching maps.

The output bundle includes:

- `model.zip`, `vecnormalize.pkl`, and `bundle.json`: matched artifacts and hashes.
- `run.json`: resolved environment identity, source hashes, training settings,
  starting checkpoint, actual step count, duration, and evaluation summary.
- `progress.csv`: PPO training diagnostics.
- `training_episodes.jsonl`: every actual training/evaluation reset recipe.
- Evaluation summaries and per-episode command/step logs with target heading,
  emitted commands, measured speed/turning, stopping, and terrain information.
- When `--record-video` is enabled, per-episode replay directories with video,
  terrain maps and exact saved terrain/path data.

Judge heading acquisition, speed error, sustained stopping, full-horizon
survival, and terrain exposure together. Radial distance alone is only a course
diagnostic: walking the wrong direction, falling after progress, or remaining on
the spawn apron does not demonstrate the requested behavior. Planned commands
that never occur because of an early fall remain visible in evaluation.

Keep the original walker frozen and compare paired flat-ground evaluations.
Run left/right turns and unseen terrain seeds before increasing slope or
roughness. The pilot evaluator reports measurements; it does not mint a
canonical gate pass or publish a trained capability.
