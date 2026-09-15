# Direction following and randomized terrain

Direction following and difficult-terrain training are supported SB3/PPO behaviors
for all six registered species: Velociraptor, Tyrannosaurus Rex, Brachiosaurus,
Dibothrosuchus, Compsognathus, and Compsognathus Robot. Every species has the same
eleven behavior selections and a complete set of TOML recipes under
`configs/<species>/behaviors/`.

These recipes activate the existing three command inputs while preserving the
species' observation and action dimensions. Terrain scenes retain the animal's
model and replace its floor. A trained locomotion checkpoint and its matched
normalization file provide the starting gait. Supported training does not mean a
new policy has already learned the behavior: saved evaluations report measured
performance, separately from canonical locomotion certification.

## Use the SB3 training notebook

Open [`notebooks/sb3_training.ipynb`](../notebooks/sb3_training.ipynb) from the
repository revision containing this feature, and use the same `REPO_REF` in its
setup form. Select any registered **SPECIES**, set `ALGORITHM = "ppo"`, and choose
one of these **BEHAVIOR** values:

For general terrain training, use **`difficult_terrain`** or
**`follow_direction_difficult_terrain`**. Each trains one policy across all five
ground families, with a new family and randomized course selected at reset.

| Behavior / TOML filename | Training task | Default additional steps |
|---|---|---:|
| **`difficult_terrain`** | Straight locomotion across flat ground, slopes, bumps, depressions, and mixed terrain | 3M |
| **`follow_direction_difficult_terrain`** | Follow heading/speed commands across all five terrain families | 3M |
| `follow_direction` | Change heading while retaining cruise speed | 3M |
| `follow_direction_speed` | Change heading and speed; stop and restart | 3M |
| `terrain_contact` | Adapt foot contacts to a flat heightfield | 300k |
| `sloped_terrain` | Walk across randomized gentle slopes | 2M |
| `bumps_terrain` | Cross scattered smooth bumps | 2M |
| `depressions_terrain` | Cross shallow solid depressions | 2M |
| `mixed_terrain` | Cross both bumps and depressions | 2M |
| `combined_terrain` | Follow heading and speed commands on slopes | 3M |
| `combined_mixed_terrain` | Follow heading and speed commands over bumps and depressions | 3M |

The nine individual presets remain useful for focused comparisons and staged
adaptation. The two general terrain behaviors combine those terrain families in
one training run.

Each filename ends in `.toml`; for example,
[`configs/compsognathus/behaviors/follow_direction_difficult_terrain.toml`](../configs/compsognathus/behaviors/follow_direction_difficult_terrain.toml).
The existing `stand`, `walk`, `hunt`, and explicit stage-ID selections retain the
canonical curriculum workflow.

Set `BEHAVIOR_CHECKPOINT` and `BEHAVIOR_VECNORMALIZE` to the matching saved pair,
then choose the loading mode explicitly:

- `BEHAVIOR_LOAD_MODE = "prepare"` starts from that species' current canonical
  **locomotion** PPO checkpoint and activates its reserved command inputs.
- `"resume"` continues the same behavior recipe from a matched behavior bundle.
- `"adapt"` transfers a learned behavior to a compatible next recipe for the same
  species, retaining learned command connections and recording its parent.

`BEHAVIOR_SEED = None` gives each run a fresh recorded seed. An integer repeats
its course. `BEHAVIOR_STEPS = None` uses the recipe budget, or the remaining
budget on resume; an explicit step count requests additional training.
`QUICK_TEST = True` uses 4,096 steps if no explicit step count is set.
`BEHAVIOR_EVAL_ONLY = True` scores without training.

An unchanged selection retains its run ID and seed and refuses to overwrite
existing output. To repeat the same settings in one runtime, set a new
`BEHAVIOR_RUN_ID`. Automatic IDs and seeds are refreshed for a new selection or
runtime. Behavior execution uses one CPU environment and inherits the parent
checkpoint's PPO network and rollout settings. `N_ENVS`, `TRUNK_FROM`,
`WIDEN_FROM`, and `RETRAIN_FROM` belong to the canonical curriculum workflow;
clear the latter three for these behavior runs.

The notebook saves results under
`logs/<species>/ppo/behaviors/<behavior>/<run-id>/`, on Google Drive when mounted
or locally otherwise. `BEHAVIOR_RECORD_VIDEO = True` saves and displays each
scored video with both terrain heat maps. `BEHAVIOR_EVAL_EPISODES` controls the
number of episodes and `BEHAVIOR_VIDEO_FPS` controls video frame rate.
Use at least five evaluation episodes for the general terrain behaviors to cover
every enabled family. Results show each family's episode count, survival, falls,
and command tracking; fewer episodes explicitly mark coverage as incomplete.
Five episodes provide only one trial per family, which checks coverage rather
than establishing competence. Use 25 or 50 episodes for a more useful comparison,
including unseen seeds, and inspect the individual outcomes.

## Species-specific recipe settings

Speeds start from each species' locomotion target. Terrain sizes follow the
actual simulated animal, including the tail; the two Compsognathus models have
much smaller ground features and slower commands than T. rex. Each terrain map
fits a 25-second straight cruise and the animal's full extent. The Compsognathus
recipes use 1,250 control steps; the other species use 2,500, giving the same
25-second horizon at their respective control rates.

| Species ID | Cruise speed (m/s) | Bump height / depression depth (mm) | Feature radius (m) | Flat spawn radius (m) | Map width (m) |
|---|---:|---:|---|---:|---:|
| `velociraptor` | 2.00 | 10 | 0.45–0.70 | 1.50 | 110 |
| `trex` | 1.05 | 20 | 0.70–1.20 | 3.00 | 70 |
| `brachiosaurus` | 0.75 | 25 | 0.90–1.50 | 3.50 | 50 |
| `dibothrosuchus` | 0.90 | 6 | 0.32–0.50 | 1.25 | 60 |
| `compsognathus` | 0.08 | 5 | 0.18–0.32 | 0.80 | 8 |
| `compsognathus_robot` | 0.04 | 4 | 0.15–0.25 | 0.50 | 5 |

These are initial training settings, not measured competence thresholds.
Feature heights are nominal maxima: overlapping features and the 3° grade cap
can change the final physical height. Maps record their measured height range,
maximum grade, and exact sample hash. Heightfield resolution and smooth feature
radii are specified in each TOML. Command normalization scales stay fixed across
one species' recipes so adaptation preserves the meaning of learned inputs.

Every recipe declares `[behavior] species`, its stable behavior `name`, and
`parent = "locomotion"`. The runner resolves that parent through the species'
`stages.toml`, loads its locomotion environment settings, and checks the supplied
checkpoint's species and stage. Parentage is explicit; the notebook does not
choose a checkpoint automatically or add these behaviors to the canonical
certification curriculum.

## Run training from the command line

Install training dependencies from the repository root:

```bash
pip install -e '.[train,test]'
```

Use a current-interface PPO locomotion checkpoint and its matching,
identity-tagged normalization file for the selected species. Checkpoints are not
included in the repository.

```bash
python -m environments.shared.train_behaviors \
  --species compsognathus \
  --recipe configs/compsognathus/behaviors/follow_direction.toml \
  --checkpoint /path/to/compsognathus/walker.zip \
  --vecnormalize /path/to/compsognathus/walker_vecnormalize.pkl \
  --output /path/to/runs/compsognathus-follow \
  --steps 4096 --eval-episodes 5 --seed 42
```

The output directory must be new or empty. PPO completes full rollouts, so actual
steps can exceed the requested count; `run.json` records both. A short check
validates execution, while the recipe budget provides a starting training budget.
Omit `--seed` for a fresh run seed, saved in the results; omit `--steps` to use the
recipe budget. `--eval-only` performs no learning. No cloud job is submitted.

The recipes use a constant learning rate of 5e-5 and PPO clipping of 0.02 for the
first 100,000 adaptation steps, then 0.2. Exact-task resume retains that warmup
anchor. Reward normalization adapts during training and is disabled for scoring.
Preparation zeros only newly activated command columns and their optimizer
moments, preserving other weights and statistics and checking initial
policy-action/value equivalence. Resume and adaptation verify bundle hashes.

## Follow directions on difficult terrain

A practical sequence is locomotion → `follow_direction` →
`follow_direction_speed` → `terrain_contact` → `difficult_terrain` →
`follow_direction_difficult_terrain`. Check performance at each step; this
sequence is a training recommendation, not an automatic promotion rule. Use the
individual template presets when diagnosing a particular kind of ground.

Start terrain work with `terrain_contact`. Even a zero-height heightfield changes
foot contacts compared with the original plane. The general terrain recipes
include original-plane episodes through their terrain sampler. Focused terrain
presets retain their existing 25% original-plane episodes. Compare slopes, bumps,
and depressions separately as well as assessing the combined training run.

Use `--adapt` when transferring a behavior checkpoint to another compatible
recipe. For example, after contact adaptation and general terrain training:

```bash
python -m environments.shared.train_behaviors \
  --species compsognathus \
  --recipe configs/compsognathus/behaviors/follow_direction_difficult_terrain.toml \
  --checkpoint /path/to/runs/compsognathus-terrain/model.zip \
  --vecnormalize /path/to/runs/compsognathus-terrain/vecnormalize.pkl \
  --output /path/to/runs/compsognathus-combined --adapt
```

Use `--resume` for the same recipe. Without `--steps`, resume trains only the
remaining recipe budget; explicit steps are additional. Compatible command
schedules and terrain parameters can change through adaptation, while changes
to the animal, command scaling, control timing, or source semantics are rejected.
Matched snapshots are saved about every 100,000 steps after completed updates;
`latest_checkpoint.json` identifies the newest one. Keyboard interruption also
saves a bundle. Resume preserves learning progress and starts fresh episodes.

## Terrain templates and randomization

The two general terrain recipes configure a weighted, balanced sampler:

```toml
[terrain_sampler]
flat = 1
sloped = 1
bumps = 1
depressions = 1
mixed = 1
```

Each shuffled five-episode block visits every family once with these defaults.
Positive integer weights repeat a family that many times per block; zero disables
it. The sum must be between 1 and 1,000. The original plane is the `flat` family;
it is distinct from the flat-heightfield `terrain_contact` adaptation preset.
`env.flat_probability` is zero in sampler recipes because the sampler already
controls flat-ground coverage.

At reset, the sampler chooses the episode's family and creates its seeded course.
**The family and the physical surface stay fixed throughout that episode.**
Commands can change during the episode, but the ground does not change underneath
the animal. The next reset advances the seeded family schedule and course
variation. The TOML's `[terrain]` section supplies the selected species' common
map dimensions, smoothness limits, spawn apron, and feature sizes.

Templates define the kind and scale of ground; seeds change its layout. The
`sloped` template combines a broad grade with small smooth ripples. `bumps`,
`depressions`, and `mixed` have a level base and localized smooth features.
Depressions have solid bottoms: they are shallow bowls, not open gaps.

Each run gets its own course. Episodes vary that layout slightly, including
feature positions, sizes, and amplitudes, while preserving a flat spawn apron.
Terrain and command randomness use independent streams. Explicit reset seeds
repeat complete resets; resets without a seed advance the episode stream.
Evaluation uses a separate seed stream. Reserve fixed seeds for comparisons and
unseen seeds to check whether a learned behavior transfers to new layouts.
For sampler recipes, evaluation visits enabled families in a balanced sequence
and reports each family separately. Missing families are listed as unevaluated,
with no invented survival or tracking result. The family counts and coverage flag
make small evaluation budgets visible.

Every reset is recorded in `training_episodes.jsonl`. Rebuild an episode from its
saved terrain manifest:

```python
from environments.shared.terrain import TerrainConfig, generate_terrain

terrain = generate_terrain(
    TerrainConfig(**saved_manifest["config"]),
    run_seed=saved_manifest["run_seed"],
    episode_index=saved_manifest["episode_index"],
)
assert terrain.manifest()["samples_sha256"] == saved_manifest["samples_sha256"]
```

Surface queries use MuJoCo's triangular heightfield interpolation. Terrain-aware
clearance and contact checks use the animal's local ground height. This release
covers smooth solid terrain; steps, large obstacles, slippery patches, automatic
difficulty promotion, terrain look-ahead sensing, and MJX training are not part
of these recipes.

## Save video replays and their heat maps

Install `pip install -e '.[train,viz]'` and add `--record-video`. To replay a saved
behavior bundle, use the same compatible recipe and source revision with
`--eval-only --resume --record-video --eval-episodes 3 --seed 42`, writing to a
new output directory. `--video-fps` changes playback rate, not physics timing.
The bundled encoder does not require a system FFmpeg installation.

Each scored episode saves a directory such as `replays/episode_000_seed_12345/`:

- `replay.mp4`: the exact trajectory used for that episode's evaluation.
- `terrain_full_map.png`: the entire physical map with the actual path.
- `terrain_local_map.png`: a close-up with a local color range for small features.
- `terrain_and_path.npz`: exact physical height samples, trajectory, and timing.
- `manifest.json`: template, seeds, sample hash, episode outcome, and file hashes.

The terminal frame is captured before automatic reset. Maps use the physical
heightfield that the animal crossed. Exports check the sample hash and publish a
replay directory only after all files are complete; requested video or map
failures fail visibly. Original-plane episodes receive clearly labelled
`flat_plane_full_map.png` and `flat_plane_local_map.png` reference maps.

To view a downloaded or moved run, keep `run.json` and its complete `replays/`
directory together and call
`display_notebook_behavior(Path("/path/to/saved-run"))` from
`environments.shared.behavior_notebook`. Media filenames resolve from the saved
episode manifests, including indexes that still contain another machine's
original paths.

## Read the results

The output includes matched `model.zip`, `vecnormalize.pkl`, and `bundle.json`,
along with `run.json`, `progress.csv`, reset manifests, and per-episode command
and step logs. Results retain the starting checkpoint, species and parent
identity, resolved settings, requested and actual steps, duration, and evaluation
measurements. Video recording adds the matched replay directories above.

Assess heading acquisition, speed error, sustained stopping, full-horizon
survival, and exposure to the terrain together. Radial distance is a course
progress diagnostic, not proof of success: the animal may walk the wrong way,
fall after progressing, or stay on the flat apron. Planned commands missed after
an early fall remain visible. Compare flat-ground retention separately from
terrain exposure and keep a frozen parent for paired checks.

### Existing T. rex runs

Historical TOMLs under `configs/trex/behavior_pilots/` remain available for their
original saved run definitions. New training uses `configs/trex/behaviors/` and
the shared runner. Do not edit old manifests or rename old recipe contents to
force a resume: bundle identities guard the meaning of the saved evidence.
