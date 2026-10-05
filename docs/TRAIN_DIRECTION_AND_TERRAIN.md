# Direction following and randomized terrain

**Status (2026-09-23): pilot pipeline, command line only, evaluation-only
outputs.** The behaviors below train through a separate runner
(`python -m environments.shared.train_behaviors`) beside the canonical chain
loop; their bundles are not certified deliverables and are not reused as
training parents (decision D-D9). The SB3 training notebook has no
direction/terrain path since the consolidation's notebook-only PR-12 slice
(decision D-D13): its `BEHAVIOR` takes `stand`, `walk`, `hunt` or a stage id, and
its `BEHAVIOR_*` knobs are gone. The final goal (every species
follows a direction on difficult terrain) is reached by re-homing
`follow_direction` and `follow_direction_difficult_terrain` as manifest nodes
under `locomotion`, trained by `train_base` and judged by a registered gate kind
([CONSOLIDATION_PLAN_2026_09.md](CONSOLIDATION_PLAN_2026_09.md) PR-9..PR-13;
goal decisions G1-G4 in [BEHAVIOR_RECIPES_PLAN.md](BEHAVIOR_RECIPES_PLAN.md)
§6.2). Until then this page is the operator guide for the pilots; every pilot
session, in every load mode and for evaluation, takes an explicit pair (the
certified library left with consolidation PR-5), so give explicit
`--checkpoint` / `--vecnormalize` paths (for the certified trex walker `20260914_123816`, the handoff pair that
`logs/trex/ppo/20260914_123816/03_locomotion/gate_verdict.json` names in its
`checkpoint` and `normalization` fields, under that node's `models/`).

Direction following and difficult-terrain training are supported SB3/PPO behaviors
for all six registered species: Velociraptor, Tyrannosaurus Rex, Brachiosaurus,
Dibothrosuchus, Compsognathus, and Compsognathus Robot. Every species has the same
eleven behavior recipes, a complete set of TOML files under
`configs/<species>/behaviors/`.

These recipes activate the existing three command inputs while preserving the
species' observation and action dimensions. Terrain scenes retain the animal's
model and replace its floor. A trained locomotion checkpoint and its matched
normalization file provide the starting gait. Supported training does not mean a
new policy has already learned the behavior: saved evaluations report measured
performance, separately from canonical locomotion certification.

## Terrain and the training notebook (status 2026-10-05)

On 2026-10-04 the maintainer deferred terrain work until the walking gait is
fixed. Every curriculum stage trains on the MuJoCo plane, and no stage TOML
sets terrain; only the behavior recipes do (see the status above).
On 2026-10-05 the maintainer paused the consolidation after PR-10: PR-11
onward waits until one or two species walk well (the consolidation plan's
status table).

The node order is `stance` (the notebook's `stand`), `locomotion` (its
`walk`), `follow_direction`, then `follow_direction_difficult_terrain`
(decision G1 in [BEHAVIOR_RECIPES_PLAN.md](BEHAVIOR_RECIPES_PLAN.md) §6.2). On
trex and the two Compsognathus models, `stand` also trains `recovery`, a side
branch that `walk` does not use. It is second in their manifests, so their
walker is node `03_locomotion` (`02_locomotion` elsewhere).
`difficult_terrain` is an optional diagnostic sibling, not a step. The
sequence under "Follow directions on difficult terrain" is advice for
command-line pilots, not this chain.

PR-11 will add the last two nodes, which `BEHAVIOR = "follow"` will select
with no new cell. Their gate will be `none/v1`, which always refuses, and the
chain loop halts at a refused gate. So the chain will stop after
`follow_direction` until PR-13 registers `terrain_command/v1` (D-D6). The
cleanup plan's decision 12 (recommended, not taken) would land it right after
PR-11.

The blocker is heightfield contact. On 2026-09-25 three certified walkers
(trex, velociraptor, compsognathus) finished every plane episode. On a flat
heightfield they finished only 1/13, 0/13 and 0/13 (the HIGH terrain blocker
in [KNOWN_ISSUES.md](KNOWN_ISSUES.md)). Velociraptor's walker also runs at
3.64–3.66 m/s against its recipes' 2.0 m/s cruise, so its maps are too small.
The cleanup plan's decisions 11 (investigate heightfield contact first) and 13
(re-derive recipe speeds and map sizes from each walker) are recommended, not
taken. Until both are decided, no terrain pilot should start
([CLEANUP_PLAN_2026_09.md](CLEANUP_PLAN_2026_09.md) §2 and §5.1, item 6).

From Colab today, a scratch cell run after the setup cell can evaluate a
certified walker. The paths in its node's `gate_verdict.json` are relative to
the node. The recipe names its species, so `--species` is optional. `OUT` must
be new or empty; the run writes only there, never under `logs/`, and its
output is evaluation-only. Dropping `--eval-only` trains a pilot, which the
blocker above says to hold.

```python
import json, sys
from pathlib import Path
from google.colab import drive
drive.mount("/content/drive")
WALKER = Path("/content/drive/MyDrive/mesozoic-labs/logs/trex/ppo/<run_id>/03_locomotion")
verdict = json.loads((WALKER / "gate_verdict.json").read_text())
assert verdict["passed"], "not a certified walker"
CKPT, NORM = WALKER / verdict["checkpoint"], WALKER / verdict["normalization"]
RECIPE = "/content/trex_gentle_contact.toml"  # any recipe file, such as the one below
OUT = "/content/drive/MyDrive/mesozoic-labs/behavior_pilots/trex_gentle_1"
!cd /content/mesozoic-labs && {sys.executable} -m environments.shared.train_behaviors --recipe "{RECIPE}" --checkpoint "{CKPT}" --vecnormalize "{NORM}" --output "{OUT}" --eval-only --eval-episodes 25 --seed 1
```

For a gentle start once terrain resumes, begin with `terrain_contact`, as
"Follow directions on difficult terrain" advises. Copy trex's
`terrain_contact.toml` recipe outside `configs/`: CI checks every recipe there
against a committed digest, and the setup cell refuses to change `REPO_REF` on
a checkout with local edits. In its `[terrain_sampler]`, set `flat = 3` and
`terrain_contact = 1`, leaving the other four at 0. Each shuffled block of
four then has three plane episodes and one on the flat heightfield. Evaluation
alternates the two families whatever their weights, so 25 episodes score 13 on
the plane and 12 on the heightfield. For a little shaped ground next, do the
same with `bumps_terrain.toml` (`flat = 3`, `bumps = 1`; optionally
`feature_height = 0.01` in `[terrain]`, half the bump height).

## The eleven behaviors

Each behavior is a recipe that the command-line runner below takes as
`--recipe configs/<species>/behaviors/<behavior>.toml`, for any registered
species, with PPO.

For general terrain training, use **`difficult_terrain`** or
**`follow_direction_difficult_terrain`**. Each trains one policy across five of
the six terrain families (all but `terrain_contact`), with a new family and
randomized course selected at reset.

| Behavior / TOML filename | Training task | Default additional steps |
|---|---|---:|
| **`difficult_terrain`** | Straight locomotion across flat ground, slopes, bumps, depressions, and mixed terrain | 3M |
| **`follow_direction_difficult_terrain`** | Follow heading/speed commands across flat ground, slopes, bumps, depressions, and mixed terrain | 3M |
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
None of these names is a notebook `BEHAVIOR` value: the notebook's `stand`,
`walk`, `hunt` and stage-ID selections are the canonical curriculum only, until
PR-11 adds the direction and terrain nodes to the species' manifests.

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
checkpoint's species and stage. Parentage is explicit: the runner takes its
source only from an explicit `--checkpoint` / `--vecnormalize` pair, in every
load mode and for evaluation (it refuses a missing one), and it never adds these
behaviors to the canonical certification curriculum.

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
Omit `--seed` for a fresh run seed, saved in the results (an integer repeats the
course); omit `--steps` to use the recipe budget, or the remaining budget on
resume; `--steps 4096` is a quick pipeline check. `--eval-only` performs no
learning. No cloud job is submitted. A run uses one CPU environment and inherits
the parent checkpoint's PPO network and rollout settings.

The loading mode follows from the flags, always with the matching saved pair in
`--checkpoint` / `--vecnormalize`:

- no flag (preparation) starts from that species' current canonical
  **locomotion** PPO checkpoint and activates its reserved command inputs;
- `--resume` continues the same behavior recipe from a matched behavior bundle;
- `--adapt` transfers a learned behavior to a compatible next recipe for the same
  species, retaining learned command connections and recording its parent.

Use at least five evaluation episodes (`--eval-episodes`, default 5) for the
general terrain behaviors to cover every enabled family. Results show each
family's episode count, survival, falls, and command tracking; fewer episodes
explicitly mark coverage as incomplete. Five episodes provide only one trial per
family, which checks coverage rather than establishing competence. Use 25 or 50
episodes for a more useful comparison, including unseen seeds, and inspect the
individual outcomes.

Runs made from the notebook before the PR-12 slice wrote to
`logs/<species>/ppo/behaviors/<behavior>/<run-id>/` (on Google Drive when it
was mounted, otherwise in that checkout's `logs/`). None is on Drive: nothing
was trained beyond 4,096-step smoke runs
([CLEANUP_PLAN_2026_09.md](CLEANUP_PLAN_2026_09.md) §5.2). Any such output is
evaluation-only pilot output (decision D-D9); leave it in place.

The recipes use a constant learning rate of 5e-5 and PPO clipping of 0.02 for the
first 100,000 adaptation steps, then 0.2. Exact-task resume retains that warmup
anchor. Reward normalization adapts during training and is disabled for scoring.
Preparation zeros only newly activated command columns and their optimizer
moments, preserving other weights and statistics and checking initial
policy-action/value equivalence. The three command inputs' statistics restart
at mean 0 / variance 1, so commands enter the policy at their own scale from
the first step, and then update like every other input (decision D-D3). The
normalization file is a plain SB3 `VecNormalize`. A bundle trained before
consolidation PR-8 neither resumes nor adapts: `--resume` and `--adapt`
refuse it, and its normalization file, which names
the deleted `BehaviorVecNormalize` class, is refused wherever a behavior
loader reads it. Evaluate it at the commit that trained it (`run.json`'s
`git_commit`). Resume and adaptation verify bundle hashes. Since
consolidation PR-9 a behavior checkpoint carries the canonical stamps: the
plant identity, the recipe env's task fingerprint (`run.json` and
`bundle.json` record it as `task_fingerprint`) and the preparation report
as its task lineage. `--resume` requires the same fingerprint; `--adapt`
the same species, plant, command mode, scales and adapter, and the same
versioned implementation (the fingerprint's stage, `command-terrain/v2`,
not source-file hashes). A bundle saved earlier records no task
fingerprint: `--resume` and `--adapt` refuse it by name (its
`mesozoic.behavior-bundle/v1` schema), and the certification panel as
another task.

## Follow directions on difficult terrain

A practical sequence is locomotion → `follow_direction` →
`follow_direction_speed` → `terrain_contact` → `difficult_terrain` →
`follow_direction_difficult_terrain`. Check performance at each step; this
sequence is a training recommendation, not an automatic promotion rule. Use the
individual template presets when diagnosing a particular kind of ground.

Start terrain work with `terrain_contact`. Even a zero-height heightfield changes
foot contacts compared with the original plane. Every terrain recipe sets its
original-plane episodes in its terrain sampler: the focused presets, including
`terrain_contact`, run one plane episode in each shuffled block of four, and the
general recipes one in each block of five. Compare slopes, bumps, and
depressions separately as well as assessing the combined training run.

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
to the animal, command scaling or adapter, control timing, or the versioned
implementation are rejected.
Matched snapshots are saved about every 100,000 steps after completed updates;
`latest_checkpoint.json` identifies the newest one. Keyboard interruption also
saves a bundle. Resume preserves learning progress and starts fresh episodes.

## Terrain templates and randomization

Every terrain recipe configures a weighted, balanced sampler. The two general
terrain recipes use:

```toml
[terrain_sampler]
flat = 1
sloped = 1
bumps = 1
depressions = 1
mixed = 1
terrain_contact = 0
```

Each shuffled five-episode block visits every enabled family once with these
weights. Positive integer weights repeat a family that many times per block; zero
disables it. A table states all six families, and the sum must be between 1 and
1,000. The original plane is the `flat` family. `terrain_contact` is a
zero-height heightfield on the recipe's map, which still changes foot contacts
compared with the plane; the other families are the gentle templates. Each
focused preset uses `flat = 1` with its own family at 3 (the `terrain_contact`
preset: `terrain_contact = 3`). A recipe with `[terrain]` and no
`[terrain_sampler]` runs every episode on that terrain. `env.flat_probability`
is retired, and a recipe that sets it is refused: `flat = 1` beside the
terrain's own family at 3 is the former 0.25.

At reset, the sampler chooses the episode's family and creates its seeded course.
**The family and the physical surface stay fixed throughout that episode.**
Commands can change during the episode, but the ground does not change underneath
the animal. The next reset advances the seeded family schedule and course
variation. The TOML's `[terrain]` section supplies the selected species' common
map dimensions, smoothness limits, spawn apron, and feature sizes; each family
sets its own surface on that map.

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
Evaluation visits each recipe's enabled families in a balanced sequence (the
plane alone for `follow_direction` and `follow_direction_speed`, which run
without terrain) and reports each family separately. Missing families are
listed as unevaluated, with no invented survival or tracking result. The
family counts and coverage flag make small evaluation budgets visible.

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

To inspect a downloaded or moved run, keep `run.json` and its complete
`replays/` directory together: each `replays/<episode>/manifest.json` names its
`replay.mp4` and map images relative to that directory, so a moved run stays
readable even where the run-level index still records another machine's
original paths. (The notebook's display helper left with the PR-12 slice.)

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

The eight `[pilot]` recipes that #540 shipped under `configs/trex/behavior_pilots/`
left with consolidation PR-6 (2026-09-20), together with the `[pilot]` recipe
dialect and the `environments/trex/scripts/train_behaviors.py` entry point. Six
of them were the `configs/trex/behaviors/` recipes with the defaults left
unwritten (`trex_gentle_terrain` was `sloped_terrain`); `trex_combined_terrain`
and `trex_follow_direction_speed` commanded a minimum speed of 0.5 where the
supported recipes command 0.525 (half the 1.05 cruise speed). A 2026-09-15 pilot
run stays self-describing, because its `run.json` `recipe` and `bundle.json`
`training_recipe` hold the recipe inline, and its outputs are evaluation-only
(decision D-D9); no recipe in the repository reproduces it for `--resume` or
`--adapt`. T. rex training uses `configs/trex/behaviors/` and
`python -m environments.shared.train_behaviors`. Do not edit old manifests or
rename old recipe contents to force a resume: bundle identities guard the
meaning of the saved evidence.

Consolidation PR-7 (#556) deleted the separate T. rex behavior class
(`environments/trex/envs/behavior_env.py`); T. rex now uses the same behavior
environment as every species. Its bundles carried the
`mesozoic.trex-command-terrain/v1` identity, which `--adapt` no longer accepts,
and every species' identity hashed the behavior sources that PR-7 changed. A
behavior bundle trained before PR-7, for any species, therefore cannot be
resumed, adapted or re-paneled; keep it as evaluation-only evidence (D-D9).
Since PR-9 the identity is the recipe env's task fingerprint, and every
bundle saved before PR-9 is refused by name.
