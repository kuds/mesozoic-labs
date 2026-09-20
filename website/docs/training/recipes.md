---
sidebar_position: 0
---

# Behavior Recipes

Every species trains a small set of named behaviors — stand, walk and hunt —
and each one is its own policy: certified against its own gate, published on
its own, and built by reusing the certified trunk beneath it and training
only what is missing above it. This page owns that vocabulary and workflow;
the [PPO](ppo.md), [SAC](sac.md), [hyperparameter](hyperparameters.md) and
[Vertex AI](vertex-ai.md) pages link here instead of re-explaining it.

## One trunk, one leaf per behavior

The training tree is not a ladder whose last rung is the only product. It is
a small directed graph of certified checkpoints: one shared trunk (stand,
walk) and a leaf per behavior (hunt). Every node is judged on its own, every
deliverable is published on its own, and no run redoes work an earlier run
already certified.

| Term | Meaning |
|---|---|
| **Node** | One stage TOML (`[env]`, `[curriculum]`, `[ppo]`, `[sac]`, `[jax]` and a `gate_kind`); one training run with one checkpoint and one gate verdict. Today's "stage". |
| **Edge** (`warm_start_from`) | The declared parent a node initialises from, under `initialize_next_stage`, with lineage recorded. |
| **Trunk** | The shared certified chain every behavior builds on: stance → (recovery) → locomotion. |
| **Leaf** | A node with no children: today `behavior` (hunt). |
| **Deliverable** | A node whose certified checkpoint is a published policy (`deliverable = true`). |
| **Recipe** | A deliverable plus its ancestor chain, *derived* from the edges; the per-entry `recipe` label groups nodes under a behavior name. "walk" = stance → locomotion; "hunt" = stance → locomotion → behavior; "stand" = stance → recovery where a recovery node exists. |
| **Certified** | A property of one checkpoint: the node's own gate passed *and* every ancestor's gate passed, with every piece of evidence hash-bound to the checkpoint it describes. |

The hunting node keeps its id `behavior` for every species; its recipe
label is `hunt`, and the task name (strike, bite proxy, food reach, snap,
target reach) comes from the stage TOML's `[stage] name`.

Direction-following and terrain traversal exist today as a separate pilot
pipeline (see `docs/TRAIN_DIRECTION_AND_TERRAIN.md` in the repository) and
are planned as manifest nodes under `locomotion` — `follow_direction` and
`follow_direction_difficult_terrain`, with their own gate kind — so the leaf
set on this page will grow; nothing on this page changes until then.

## The stage manifest

Each species declares its graph in `configs/<species>/stages.toml` under
schema `mesozoic.stage-manifest/v2`. Per `[[stages]]` entry:

| Key | Meaning |
|---|---|
| `id` | The stage's identity; an open vocabulary matching `^[a-z][a-z0-9_]*$`. The four ids `stance`, `recovery`, `locomotion` and `behavior` are reserved and keep their meaning everywhere. |
| `config` | The stage TOML file, relative to the species config directory. |
| `legacy_number` | The integer the stage was known as before manifests existed (1, 2, 3); absent on a stage that is addressed by id only. |
| `warm_start_from` | The id of an **earlier** entry this node initialises from; absent means root. A self or forward reference is fatal. |
| `deliverable` | `true` when the node's certified checkpoint is a published policy. A v2 manifest with no deliverable is fatal. |
| `recipe` | The behavior label the node belongs to. A label resolves to its deepest deliverable in manifest order. |

Recipes are read off the edges, never declared in a second table, so adding a
behavior is adding a node with an edge. The committed T-Rex manifest:

```toml
schema = "mesozoic.stage-manifest/v2"

[[stages]]                       # quiet stance
id = "stance"
config = "stance.toml"
legacy_number = 1
recipe = "stand"
deliverable = true               # stance_quality/v1; certified 2/3 seeds

[[stages]]                       # balance under pushes
id = "recovery"
config = "recovery.toml"
warm_start_from = "stance"
recipe = "stand"
deliverable = true               # recovery_quality/v1, judged against a per-run frozen resolution

[[stages]]
id = "locomotion"
config = "locomotion.toml"
legacy_number = 2
warm_start_from = "stance"       # the 2026-08-23 lineage rule, now declared;
recipe = "walk"                  # flips to "recovery" once a recovery run passes (plan A2)
deliverable = true

[[stages]]                       # id stays "behavior"; the label is the recipe
id = "behavior"
config = "behavior.toml"
legacy_number = 3
warm_start_from = "locomotion"
recipe = "hunt"
deliverable = true
```

Compsognathus and the Compsognathus robot commit the same four-node layout.
Velociraptor, Brachiosaurus and Dibothrosuchus commit three nodes — `stance`
(`stand`), `locomotion` (`walk`, warm-started from stance) and `behavior`
(`hunt`, warm-started from locomotion) — over their existing
`stage1_*.toml`, `stage2_*.toml` and `stage3_*.toml` files. Every node in
every committed manifest is a deliverable.

Two consequences of label resolution: on a species with a recovery node,
`"stand"` resolves to `recovery` (the deepest `stand` deliverable), so the
stance node alone is addressed by its id `"stance"`; and `"hunt"` resolves to
`behavior` for every species, which is why the notebook's default works
everywhere. Inside a run, stage directories are named `{position:02d}_{id}`
(`01_stance`, `02_locomotion`, `03_behavior` for the three-node species;
`01_stance`, `02_recovery`, `03_locomotion`, `04_behavior` on the T-Rex), and
file labels keep `stage1`, `stage2`, `stage3` for the legacy-numbered stages
and the bare id otherwise (`recovery_final.zip`).

## What one run does

A run names a target: `BEHAVIOR` in the notebook, or `--target BEHAVIOR` on
the command line (default: the last advancing, legacy-numbered stage). It
resolves the target's ancestor chain from the manifest and walks it
root-first. At each node it does exactly one of three things:

1. **Reuse** a certified checkpoint that already exists — in this run, or,
   for an ancestor, in the trunk run — when the reuse rule below holds. A
   cross-run ancestor is recorded under `ancestors/<stage_id>/` (its JSON
   records copied, never its checkpoint pair) and loaded from where it
   lives, by the CLI and the notebook alike.
   The trunk run is the one `TRUNK_FROM` / `--trunk-from` names, or, under
   `"auto"` (the notebook default, decision D-A25), the run beside this one
   (under the species/algorithm log directory in the notebook; the siblings
   of the run directory on the command line) whose certified ancestors cover
   the most of the chain root-first, newest on a tie. The selection prints
   the run it chose, the replication each reused node rests on, the runs it
   refused with the rule that refused them (the first 20; the selection
   object holds every run scanned), and any run whose root passed under an
   older policy interface as a `WIDEN_FROM` candidate.
2. **Judge** a node that was trained but never gated (the notebook only:
   its final checkpoint exists but `gate_verdict.json` does not, because the
   resume cell spent its budget).
3. **Train** it otherwise, warm-started from its parent's handoff checkpoint
   and VecNormalize sidecar along the declared edge, then judge its gate and
   write `gate_verdict.json` beside the handoff.

The target node is never reused across runs: it is what the run exists to
certify, and an earlier run's certified target is that run's deliverable,
published from there. A node trains only on a parent that passed its gate; a
failed node stops the chain, and nothing is trained from scratch silently. In
the notebook the bundle is written before the verdict is enforced, so a
failed hunt still publishes the certified nodes this run trained beneath it.
Ancestors reused from a trunk are not this run's deliverables: they stay
published by the run that certified them (D-A18) and appear here only under
`provenance.ancestors`, so a trunked run whose target fails has nothing
certified to publish.

## The reuse rule

`environments/shared/ancestors.py` owns the rule; `curriculum --trunk-from`
and the notebook loop both apply it, refusing on the first failure with a
reason naming it:

1. the candidate run has a stage directory for the node, in any layout
   generation — or, holding only `ancestors/<stage_id>/ancestor.json` for it
   (it reused the node itself), the record is followed to the run that
   certified the node: its `source_run_dir` as recorded, else the run of the
   same name beside the candidate (Colab, Drive and bucket layouts keep runs
   side by side), else a refusal naming both paths. The source must pass
   every rule below, and its handoff pair must still hash to the digests the
   record bound the reuse to; at most eight records are followed and a
   record pointing back at a run already on the path is refused (D-A23);
2. that directory carries a `gate_verdict.json` that passed, hashes both
   files of its handoff pair, and judged this node's id;
3. the verdict's `task_sha256` equals the fingerprint derived from the
   current stage config — exact equality, the same check `resume_same_stage`
   applies;
4. the chain: a non-root candidate's recorded `parent_checkpoint_sha256`
   must equal the digest of the checkpoint resolved for its declared parent
   in this run, so a certified walk is reused only on top of the very stance
   it was trained from; a root candidate must not have entered from a parent
   at all. Reuse is therefore root-first, and once a node is trained in this
   run none of its descendants are looked up;
5. the handoff pair the directory selects now re-hashes to the verdict's
   digests, so a checkpoint rewritten after judging is refused;
6. the checkpoint's recorded plant identity validates against the current
   plant, with no legacy allowance;
7. the gate (decision D-A22; evaluated between rules 3 and 4, before the
   chain, the hashing and the plant): the verdict's `gate_sha256` — the
   digest of the gate it was judged under (`gate`: kind, schema version and
   only the thresholds the kind consumes, numerics normalised so `100` and
   `100.0` are one gate) — equals the digest of the node's current
   `[curriculum]` block. A verdict without the field was judged before D-A22
   and is refused until re-judged; a differing digest is refused naming every
   threshold that differs (`min_avg_reward: judged at 1940.0, configured
   2100.0 now`). Only the gate the verdict was judged under counts, never
   the block the directory trained under: a directory re-judged under an
   edited gate is reusable under that gate (edit a threshold, re-judge, never
   retrain). The re-judge paths are the notebook's JUDGE branch
   (`generate_stage_artifacts`, for a directory holding no verdict — remove
   a refused one first) and `backfill_gate_verdict.py --force [--gate
   current]` below.

A candidate the rule refuses is not an error: `curriculum --trunk-from` and
the notebook loop log the refusal and train the node in the new run, so
inventory the trunk's verdicts before the first trunked run.

Two runs that both certified stance produced two different checkpoints; a
walk descends from exactly one of them, and ids never stand in for digests.

Trunks compose. A run that reused stance from an earlier
trunk holds no stance directory, only `ancestors/stance/`; asked to follow
records (`find_certified_ancestor(..., follow_records=True)`, which
`curriculum --trunk-from` and the notebook's trunk candidate pass), rule 1 resolves through that record to the
run that certified stance, one machine-visible run directory away, and
applies every rule there. A later run trunked from it therefore reuses stance
from the original run — its `ancestors/stance/ancestor.json` and walk's
`parent_run_id` name the original, never the middle run — and reuses the walk
the middle run trained itself. The log names the path taken (`Following the
ancestor record in <middle> to run <original>`). A source rewritten and
re-judged since the record was made is refused: the record binds the reuse to
one checkpoint pair. The record names the source by absolute path, so one
made from `--trunk-from logs/<run>` follows from any working directory; when
the recorded path is gone, the run of the same name beside the middle run is
tried (a moved `LOG_BASE`), else the refusal names both paths. The notebook's
chain loop follows records for the `TRUNK_DIR` candidate only: it tries this
run's own `RUN_DIR` first and takes a hit there as this run's own node, so a
record in `RUN_DIR` — the reuse an earlier pass made from the trunk — is never
followed, or the trunk's stance would re-enter as trained here.

On reuse the child run writes `ancestors/<stage_id>/`: `ancestor.json` plus
verbatim copies of the ancestor stage's `gate_verdict.json`,
`stage_config.json`, `task_fingerprint.json` and `plant_identity.json` —
small records only, never the checkpoint pair — and the child node's lineage
names the ancestor's run as `parent_run_id`.

A stage directory judged before Phase A has no verdict file and is refused by
rule 2 until it is re-judged from its recorded evidence:

```bash
python -m environments.shared.scripts.backfill_gate_verdict logs/<run>/<stage_dir>
```

`--species` and `--stage` override what the directory's `stage_config.json`
records, and `--force` re-derives over an existing verdict — the path for a
verdict written before D-A22, which records no `gate_sha256` and is refused
by rule 7. `--gate recorded` (the default) judges under the block
`stage_config.json` recorded and digests it; `--gate current` judges under
the checkout's block for the stage and digests that one, the re-judge path
after a threshold edit for a `reward_and_length/v1` directory, whose episode
rows are re-aggregated under the new floors. A `stance_quality/v1` verdict is
read off `stance_gate_report.json` and re-derives nothing, so the tool refuses
a report whose recorded `thresholds` are not the ones being judged under
(under either `--gate`): a stance directory whose rail moved is re-judged
through the notebook's JUDGE branch, which measures a fresh panel. The tool
refuses when the evidence is missing, and cannot backfill
`recovery_quality/v1`.

## In the notebook

`notebooks/sb3_training.ipynb` exposes six knobs in its configuration cell,
committed as:

```python
BEHAVIOR = "hunt"  # a recipe label ("stand" | "walk" | "hunt") or a deliverable's stage id
TRUNK_FROM = "auto"  # "auto" (D-A25): the sibling run covering the most of the chain; a run id pins one; "" trains every node here
WIDEN_FROM = ""  # optional earlier run (id or absolute path) whose certified ROOT handoff is widened to this checkout's policy interface into RUN_DIR before the chain runs (BEHAVIOR_RECIPES_PLAN §4.6 Phase C)
WIDEN_MAX_REVISION_GAP = 1  # how many policy-interface revisions behind WIDEN_FROM's parent may be (D-C17); 1 = the Phase C bump alone; the two certified trex stance parents (r11) need 2 because r11 → r12 was fingerprint-only
RETRAIN_FROM = ""  # optional chain node to train here with every node below it (empty = off; D-A19)
RUN_LABEL = ""  # optional free-text label recorded beside each trained node's hyperparameter digest (D-A21)
```

`WIDEN_FROM` names an earlier run whose certified root checkpoint was trained
behind this checkout's policy interface: the widen cell after the chain
resolution widens its handoff pair into this run's root stage directory (zero
columns for the new command dims, never re-trained) and the chain loop
re-judges it; a widen session sets `SEED` to that run's seed before the storage
cell runs (decisions D-C13/D-C14). `WIDEN_MAX_REVISION_GAP` bounds how many
policy-interface revisions behind that parent may be (default `1`, the Phase C
bump alone). The widen tool still checks the physics digest, `nq`/`nv`/`nu`,
`action_dim` and the observation widths whatever the bound, so a larger value
only crosses fingerprint-only bumps, and a parent further behind than the
bound is refused with the gap and the tool's `--max-revision-gap` /
`max_revision_gap` bound named; the two certified trex
stance parents are r11 archives, two revisions behind r13, and need `2`
(decision D-C17).

`TRUNK_FROM` defaults to `"auto"`: once the chain is resolved, the resolve
cell selects the run under `<LOG_BASE>/<species>/<algorithm>/` whose
certified ancestors cover the most of the chain root-first (newest on a tie)
and prints the choice, what it rests on and every refusal; `""` turns reuse
off. A pinned `TRUNK_FROM` is a run id, resolved under the same directory,
or an absolute path to a run directory. It must be a finished bundle (a run
with a `provenance.json`) of the same species, algorithm and backend, and it
must not be this run — certified nodes of an earlier run come in through
`TRUNK_FROM`, never by pointing `RUN_ID` at that run. `RETRAIN_FROM` must name
a node on the chosen behavior's chain.

The setup cell resolves the chain once: `TARGET_NODE =
MANIFEST.resolve_behavior(BEHAVIOR)` and `CHAIN =
MANIFEST.chain_for(TARGET_NODE.id)`. One `# ===== BEHAVIOR CHAIN LOOP =====`
cell then walks `CHAIN` root-first and, per node, reuses, judges or trains
as described above. Reuse looks in this run first, then — for ancestors only
— in the trunk. For a frozen-null gate kind (`recovery_quality/v1`, the
`FROZEN_NULL_GATE_KINDS` set) the gate's thresholds and null panels freeze
**before** the node trains and the policy panel rolls after, on the same
frozen seeds. Every trained or judged node writes its artifacts, the training
summary and the run bundle before its verdict is enforced; the runtime is
then released and the loop raises. A reuse that would ignore an edit to the
node's algorithm block prints a warning naming the differing keys and points
at `RETRAIN_FROM`. A stage directory that already records a node is refused,
so a new variant is a new `RUN_ID`.

**Hunt on a walk trunk.** An earlier run `20260901_120000` certified stance
and locomotion for the Velociraptor under PPO. In a fresh session:

```python
BEHAVIOR = "hunt"
TRUNK_FROM = "20260901_120000"
```

The chain is `stance -> locomotion -> behavior`. The loop reuses the trunk's
stance if its verdict passed, its task digest and plant match, and it entered
from no parent; reuses the trunk's locomotion only if it was trained from
that very stance checkpoint; then trains `behavior` here, warm-started from
the reused locomotion handoff. The run directory gains `ancestors/stance/`
and `ancestors/locomotion/`, the trained `03_behavior/` records
`parent_run_id`, and the bundle publishes `behavior` as its one deliverable
(`target_deliverable = "behavior"`), with `provenance.ancestors` carrying
stance and locomotion — certified through the trunk's verdicts and still
published by the trunk run.

**Stand on the T-Rex.** `"stand"` resolves to the deepest `stand`
deliverable, which on a species with a recovery node is `recovery`:

```python
BEHAVIOR = "stand"
```

The chain is `stance -> recovery`. Stance trains (or is reused), then the
recovery resolution is frozen from the stance handoff, recovery trains, the
frozen panel is rolled, and the `recovery_quality/v1` verdict is enforced
exactly like any other node's. To train the stance node alone, set
`BEHAVIOR = "stance"`. The old `RUN_RECOVERY_STAGE` switch no longer exists.

Two escape hatches remain. The manual single-node cell (`MANUAL_NODE`) trains
one node outside the chain and records its verdict without enforcing it; it
never feeds the chain. The resume cell (`RESUME_STAGE`) continues an
interrupted node from its newest intact periodic checkpoint under
`resume_same_stage`; the re-saved `stage_config.json` keeps the edge's
lineage keys and records the continued-from checkpoint under
`resume_load_path` / `resume_checkpoint_sha256`, so a resumed-then-judged
node still chains by digest and stays reusable.

## On the command line

The `curriculum` command runs the species' advancing stages in manifest
order; each node warm-starts from its declared parent's handoff checkpoint
and VecNormalize sidecar, a node whose parent has no certified checkpoint
stops the run, and every trained node writes `gate_verdict.json` from the
in-training manager's verdict. `--target BEHAVIOR` names the behavior the
run certifies — a recipe label (`walk`) or a deliverable's stage id
(`locomotion`), resolved as the notebook's `BEHAVIOR` knob resolves them, or
a legacy number (`2`), resolved as `--stage` resolves it (the notebook takes
no number) — and the run walks that target's chain and stops there, so
`--target walk` certifies a walk-only run; the default is the last advancing
stage, the whole ladder. The target is never reused from a trunk, whichever
node it is. The chain must be the advancing ladder up to the target, because
the command-line curriculum judges with the integer-keyed `CurriculumManager`,
advanced once per node walked: `--target stand` on the T-Rex or Compsognathus,
whose `stand` chain runs through the non-advancing recovery node, is refused
before any directory is written, and that chain is trained through the
notebook (`BEHAVIOR = "stand"`); a chain that skips a ladder stage (an edge
rewired past it) is refused the same way, since the manager would judge the
node after the gap against the skipped stage's thresholds. `--retrain-from`
must name a node on the target's chain.

```bash
cd environments/velociraptor

# Reuse an earlier run's certified stance and locomotion; train behavior here
python scripts/train_sb3.py curriculum --algorithm ppo \
  --trunk-from logs/<earlier_run> --output-dir logs/<new_run>

# A walk-only run: reuse the certified stance, train locomotion here as the target, stop
python scripts/train_sb3.py curriculum --algorithm ppo --target walk \
  --trunk-from logs/<earlier_run> --output-dir logs/<walk_run>

# Reuse only the certified stance; retrain locomotion and behavior here
python scripts/train_sb3.py curriculum --algorithm ppo \
  --trunk-from logs/<earlier_run> --retrain-from locomotion --output-dir logs/<new_run>

# Tag every trained node (recorded beside its hyperparameter digest; W&B tag label:<TEXT>)
python scripts/train_sb3.py curriculum --algorithm ppo --label "lr-3e-4"

# One node by hand, entering from its declared parent's handoff
python scripts/train_sb3.py train --stage locomotion --algorithm ppo \
  --load logs/<run>/01_stance/models/robust_best_model.zip \
  --load-mode initialize_next_stage

# The non-advancing recovery node (T-Rex, Compsognathus): never part of the CLI curriculum
python scripts/train_sb3.py train --stage recovery --algorithm ppo \
  --load logs/<run>/01_stance/models/robust_best_model.zip \
  --load-mode initialize_next_stage
```

`--stage` accepts a legacy number or a stage id; `--retrain-from` accepts
either too. `--load` takes the checkpoint (`.zip` or its stem) and finds its
VecNormalize sidecar beside it as `<stem>_vecnorm.pkl`; the handoff the
curriculum promotes is `robust_best_model`, else `best_model`. The same flags
are exposed by the shared entry point:
`python -m environments.shared.train --species <species> curriculum ...`.

A single-node `train` run is unjudged: `train()` records the stage config,
lineage and checkpoint pair but writes no `gate_verdict.json` (the
curriculum, `generate_stage_artifacts` and the backfill tool are what write
one), and it writes into its own `--output-dir` — by default
`logs/<species>/<stage_dir>_<timestamp>/` — not into a `<run>/01_stance/`
layout. A hand-chained ladder therefore cannot be passed as `--trunk-from`
until each stage directory has been re-judged from its recorded evidence
with the backfill tool above.

Refusals:

- `--retrain-from` without `--trunk-from` is a usage error (without a trunk
  every stage is trained in this run already), as is naming a non-advancing
  or unknown stage. Both are raised before the run directory is created.
- `train --load-mode initialize_next_stage` refuses a checkpoint whose
  recorded stage is not this node's declared `warm_start_from` parent. The
  default `--load-mode resume_same_stage` requires an exact task match, so a
  parent's checkpoint cannot be loaded under it.
- A stage directory that already holds `stage_config.json` or
  `gate_verdict.json` is refused unless the load is an explicit
  `--load <checkpoint> --load-mode resume_same_stage`. `train` checks this
  before it writes anything; the curriculum checks each node when the walk
  reaches it, before that stage directory is written to — by then the run
  directory, its `plant_identity.json` and any earlier node already exist.
  The curriculum has no resume mode, so it always needs a fresh
  `--output-dir`.
- Non-advancing nodes (no legacy number — recovery) are skipped by the CLI
  curriculum with a log line in Phase A and trained on their own with
  `train --stage recovery`.

A CLI curriculum run writes `curriculum_results.csv` and a `gate_verdict.json`
per trained node but no `provenance.json` or `summary.json`; the result
bundle described below is written by the notebook. A CLI run can still serve
as a later CLI run's `--trunk-from` (it is named by its directory), and a
notebook run can serve as a CLI trunk. A pinned notebook `TRUNK_FROM`
requires a run with a `provenance.json`; `"auto"` judges siblings by their
stage records and refuses one whose `provenance.json` names another species,
algorithm or backend.

## What a run directory holds

After a notebook run that reused a trunk's stance, a T-Rex run directory
looks like this:

```
20260912_trex_ppo/
├── ancestors/
│   └── stance/                 reused from run 20260901: ancestor.json + copies of its
│       ├── ancestor.json         gate_verdict, stage_config, task_fingerprint, plant_identity
│       └── ...                   (never the checkpoint)
├── 03_locomotion/
│   ├── models/robust_best_model.zip + _vecnorm.pkl
│   ├── stage_config.json       run block: load_mode = initialize_next_stage,
│   │                             parent_checkpoint_sha256 = the stance digest, parent_run_id
│   └── gate_verdict.json       passed, hash-bound to the handoff pair above
├── 04_behavior/                the target: always trained here
│   └── ...
├── provenance.json             deliverables {locomotion, behavior}, each with certified;
│                               ancestors {stance} — reused, still published by run 20260901
└── summary.json                written whenever anything certified; status complete | partial
```

The `run` block of each `stage_config.json` records:

| Key | Meaning |
|---|---|
| `load_path`, `load_mode`, `parent_checkpoint_sha256`, `parent_task_sha256` | Where the node's initial weights came from; a from-scratch root writes none of them. |
| `parent_run_id` | Present only when the parent was a certified ancestor reused from another run. |
| `resume_load_path`, `resume_checkpoint_sha256` | The periodic checkpoint a same-stage resume continued from; the edge keys above are kept. |
| `hyperparameters_sha256` | A digest over the stage's `[ppo]` or `[sac]` block plus its `warmup_` / `ramp_` shaping keys, key-order independent, untouched by env kwargs or gate thresholds. Always written. |
| `label` | The free-text label from `--label` / `RUN_LABEL`, when one was given. |
| `duration_seconds` | How long the stage actually trained, accumulated across every session that trained it. Written by the notebook's `train_stage` on every exit (D-A15); absent on CLI runs. |

`gate_verdict.json` (schema `mesozoic.gate-verdict/v1`) records the species,
stage and stage id, the gate kind and schema version, `passed` and the list
of failures, the handoff checkpoint and its normalization sidecar with their
digests, the `task_sha256` it was judged on, when it was judged and
`judged_by` (the post-stage judge, the in-training manager, or `backfill`).

## Publication per deliverable

Result schema v4 (`provenance.json` and `summary.json`; v2 and v3 still
read):

- `provenance.deliverables` maps each deliverable the run holds to
  `model_path`, `model_hash`, `normalization_hash`, `gate_kind`, `certified`
  and `replication`, plus the optional `hyperparameters_sha256` and `label`
  copied from the stage's run block, and `certification_seeds` and
  `provisional`, which the bundle writer derives — the stage's declared
  `[curriculum]` bar (default 1) and `replication.count < certification_seeds`
  — rather than copying from the run block. `provenance.ancestors` is the
  summary-side projection of `ancestors/<stage_id>/`.
- `replication` is writer-recorded (decision D-B10). When the notebook's
  publication cell runs, `environments.shared.replication.discover_replicates_for_run`
  looks through the sibling runs under `LOG_BASE/<species>/<algo>/` for the
  same recipe on another seed — a passed, reusable `gate_verdict.json` for
  the node with equal `task_sha256` and `gate_sha256`, a `stage_config.json`
  with the same plant identity and `hyperparameters_sha256` (derived from
  the recorded blocks when a sibling saved before that field existed), and
  a different training seed (decision D-B16) — and `replication.runs` lists
  this run first, then those replicates. A sibling whose verdict has no
  `gate_sha256` is skipped until re-backfilled (rule 7). A deliverable with
  fewer runs than its stage's `certification_seeds` is `provisional` (trex
  stance declares 2; once a stance bundle is published the catalog and the
  species page render it as `1 run of 2 seeds; provisional`, re-deriving
  the label from the current config — no trex stance bundle is committed
  yet; see `docs/KNOWN_ISSUES.md`). A replicate that certifies later is counted by re-running the
  publication cell of the run that should count it, with the sibling
  present: a `partial` bundle is rebuilt, and a `complete` bundle
  regenerates its derived artifacts when the replication record is its
  only change — anything else on a complete bundle is still refused as
  immutable.
- `target_deliverable` names the node the run aimed at and
  `primary_deliverable` the published model — the target when certified,
  else the deepest certified deliverable; `selected_model_path` is the
  primary's.
- Bundle status is `complete` when the target is present and certified and
  every present deliverable is certified, `partial` when at least one
  deliverable is certified, and `failed` otherwise. `summary.json` is written
  whenever something is certified, so a failed hunt publishes the certified
  walk it trained, a walk-only run is a complete bundle when walk was its
  target, and a stance-only run targeting hunt is `partial`, never
  `complete`. Deliverables are the nodes trained or judged in this run; a
  reused ancestor is listed under `provenance.ancestors` only, so a trunked
  run whose target fails is `failed` and writes no `summary.json` — its
  trunk is already published by the run that certified it.
- `validate_result_bundle(..., require_publishable=True)` requires at least
  one certified deliverable; `require_complete=True` keeps meaning
  `complete`. The audit reports `canonical-partial` for a bundle with at
  least one certified deliverable.

The species catalog (schema 4) and `configs/species_manifest.toml` (schema 2)
follow: stage rows carry `deliverable`, `warm_start_from` and `recipe`; the
README SPECIES table shows Recipe and Warm-start-from columns (a generated
block, never hand-edited); result rows list one entry per deliverable
headlined by its gate kind, with stance and recovery headline values
rendered as null until a later phase exports per-stage gate metrics
(decision D-B15); and stage
videos are keyed by stage id. The four historical ladder summaries publish
`deliverables = []` and are never relabelled certified.

## Experiments on a node that already passes

Certification belongs to a checkpoint for a task. The task digest covers
the species, stage, backend, plant, the effective `[env]` block (every
reward weight, episode length, reset noise) and the push schedule. It does
not cover the `[ppo]` / `[sac]` block, the timesteps budget, or the gate
thresholds. That one fact decides what an edit does to a trunked run:

| You edit | Task digest | What a trunked run does |
|---|---|---|
| Algorithm block (`[ppo]`, `[sac]`, `warmup_` / `ramp_` shaping) | unchanged | Reuses the old certified checkpoint unless the node is the run's target or `--retrain-from` / `RETRAIN_FROM` covers it. The reuse prints a warning naming the differing keys (`ppo.learning_rate`, `shaping.warmup_timesteps`) and pointing at the retrain knob — never a refusal. |
| Environment block (`[env]`, pushes, plant) | changes | Rule 3 refuses the old checkpoint; the node and everything below it retrain. Older checkpoints stay valid in the runs that certified them. |
| Gate thresholds | unchanged | Rule 7 refuses the old verdict, naming the thresholds that differ (`gate_sha256` no longer matches). Re-judge the directory under the current gate — the notebook JUDGE branch / `generate_stage_artifacts`, or `backfill_gate_verdict.py --force --gate current` for a `reward_and_length/v1` directory — never retrain; until then a trunked run trains the node itself. |

Two variants of one node are two runs: writing into a stage directory that
already records a node is refused, so give each variant a fresh
`--output-dir` / `RUN_ID`. Nothing ranks variants — compare their bundles.
To try new algorithm values on a node the trunk already certified, retrain
from it:

```bash
python scripts/train_sb3.py curriculum --algorithm ppo \
  --trunk-from logs/<certified_run> --retrain-from locomotion --output-dir logs/<variant_run>
```

The variant reuses stance as an ancestor and trains locomotion and behavior
here. A later hunt trunked from `<variant_run>` reuses the variant's
locomotion and, through the variant's `ancestors/stance/` record, the stance
that `<certified_run>` certified (rule 1 above): the winning locomotion
carries into later hunts without retraining anything, provided the original
run is still visible from the machine — as recorded, or beside the variant
under the same log base.

Vertex AI sweep trials are single-stage `train()` runs wrapped by the
`trial` subcommand, which judges the stage afterwards through
`generate_stage_artifacts` (a `gate_verdict.json` beside the trial's
handoff); the Ray Tune worker runs its own SB3 loop and judges the same way,
best-effort. Neither trial directory is a run directory — its `models/` sit
directly under `stageN/<trial_id>/`, so rule 1 finds no stage directory for
the node — and trials are never reusable as ancestors (plan A6). Promote a
winning configuration by committing it to the TOML and training a fresh
curriculum or a trunked `--retrain-from` run. See
[Hyperparameter Sweeps](sweeps.md).

## What stays on the ladder in Phase A

- The JAX/MJX runner (`jax_curriculum.run_curriculum`) walks the advancing
  stages by number, carrying parameters and normalization statistics forward;
  the JAX notebook walks them by `CURRENT_STAGE`. Neither reads
  `warm_start_from`, writes `gate_verdict.json` or publishes deliverables,
  so a JAX run cannot serve as a trunk. See [JAX/MJX Training](jax.md).
- Sweeps stay trunk-only and integer-keyed (`stage1` / `stage2` / `stage3`
  in the search-space files and the collector), and their trials are never
  reusable as ancestors.
- The in-training `CurriculumManager` stays integer-keyed: it judges the
  advancing nodes during training, and semantic-id nodes (recovery) are
  judged after the stage.
- Stance and recovery deliverables headline a metric name with a null value
  until per-stage gate metrics are exported into the summary (a later phase,
  decision D-B15). A verdict records the gate it was judged under (`gate` /
  `gate_sha256`, decision D-A22), but the deliverable's summary does not
  surface it yet. The hunt deliverable is the exception: `task_success/v1`
  exports `selected_model_success_lcb` (with the count and panel size) into
  the stage row, judged from the selected checkpoint's `evaluation_selected.csv`.
- A CLI-certified hunt (`curriculum --target hunt`) is judged IN-TRAINING on
  the last EvalCallback panel (the manager's bound over its successes,
  recorded as `success_count` / `n_success_samples` in the verdict's
  `stage_result`) and writes no `evaluation_selected.csv`, so it cannot be
  backfilled or published without the notebook's post-stage judge; treat it
  as a training-time signal, not a certification.
