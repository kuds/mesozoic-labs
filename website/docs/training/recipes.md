---
sidebar_position: 0
---

# Behavior Recipes

Every model trains a small set of named behaviors — stance and walk, and
recover on the models with a recovery stage — and each one is its own
policy: certified against its own gate, published on
its own, and built by reusing the certified trunk beneath it and training
only what is missing above it. This page owns that vocabulary and workflow;
the [PPO](ppo.md), [SAC](sac.md) and [hyperparameter](hyperparameters.md)
pages link here instead of re-explaining it.

## One trunk, one leaf per behavior

The training tree is not a ladder whose last rung is the only product. It is
a small directed graph of certified checkpoints: one shared trunk (stance,
walk) and a leaf per behavior built on it. Every node is judged on its own, every
deliverable is published on its own, and no run redoes work an earlier run
already certified.

| Term | Meaning |
|---|---|
| **Node** | One stage TOML (`[env]`, `[curriculum]`, `[ppo]`, `[sac]` and a `gate_kind`); one training run with one checkpoint and one gate verdict. Today's "stage". |
| **Edge** (`warm_start_from`) | The declared parent a node initialises from, under `initialize_next_stage`, with lineage recorded. |
| **Trunk** | The shared certified chain every behavior builds on: stance → (recovery) → locomotion. |
| **Leaf** | A node with no children, such as `locomotion` on every model, or `recovery` on the models that have it. |
| **Deliverable** | A node whose certified checkpoint is a published policy (`deliverable = true`). |
| **Recipe** | A deliverable plus its ancestor chain, *derived* from the edges; the per-entry `recipe` label groups nodes under a behavior name. "stance" = stance alone, on every model; "recover" = stance → recovery where a recovery node exists; "walk" = stance → locomotion. |
| **Certified** | A property of one checkpoint: the node's own gate passed *and* every ancestor's gate passed, with every piece of evidence hash-bound to the checkpoint it describes. |

Until decision D-D59 (2026-11-23) six models had a hunting leaf after walk:
the node `behavior` (historical stage 3), labelled `hunt`, whose task (strike,
bite proxy, food reach, snap, target reach) came from its stage TOML. That
node is removed (the prey it targeted left the scene with the policy-interface
revision that followed under the same decision), so every chain now ends at walk;
`"hunt"` is refused naming `"walk"` wherever a label is taken, and the id
`behavior` and legacy number 3 stay reserved, resolving to no stage.

Direction-following and terrain traversal exist today as a separate pilot
pipeline (see `docs/TRAIN_DIRECTION_AND_TERRAIN.md` in the repository) and
are planned as manifest nodes under `locomotion` — `follow_direction` and
`follow_direction_difficult_terrain`, judged by the `follow` profile of the
`command_tracking/v1` gate kind — so the leaf set on this page will grow;
nothing on this page changes until then.
Follow-direction, driven by a commanded heading and speed, is the path after
walk (decision D-D59). The machinery those nodes need exists since decision
D-D60: a stage TOML may declare a `[commands]` table; every environment scores
a live command with a tracking reward and draws its schedule from a random
stream of its own; and every stage holds the command inputs' normalization at
identity, so a node that widens its parent's commands continues it exactly
([Hyperparameters](hyperparameters.md#commands)). Since decision D-D61 the
gate kind for a live command exists too: `command_tracking/v1`, whose `walk`
profile judges the walk node on a panel of whole-episode cruise and stand
commands and whose `follow` profile judges the follow nodes, each with a gait
clause, and whose handoff is the checkpoint the gate itself prefers
([Hyperparameters](hyperparameters.md#curriculum-thresholds)).

**The walk is the first live-command stage** (decision D-D62). On every
model, `locomotion` follows a narrow command distribution: each episode draws
one request at reset and keeps it, a forward cruise between half and all of
the model's cruise speed along a heading up to 15 degrees off the spawn
heading, or, in about 15% of episodes, an exactly zero command, a
whole-episode stand. Its reward is the tracking term beside the forward
walk's alive, posture, effort and gait terms; a stand block pays the
stance's weights at a zero command, so zero means stand still; and a cruise
ramp raises the training commands from a quarter of the end speeds over the
first 2M steps. It is judged after training by the `walk` profile, whose
bars are provisional (decision D-D62 (16)). Since decision D-D64 the
tracking term and the gate score a moving command on a one-stride trailing
mean (`[commands] tracking_window_s`), so at a command it matches every
reference walk in the repository clears the in-tolerance fraction and the
speed ratio (the T-Rex and Allosaurus fail `heading_settle` on their
scripted starts), though no controller has yet passed a committed walk
panel. Its handoff is the pair the gate prefers
(`models/handoff_choice.json`). The
stance and recovery run no command, and their task and gate digests did not
move. Because a walk warm-starts from a stance with its command inputs
zeroed, the stance it starts from must be one trained on the current plant
interface; and since a recipe of the pilot pipeline starts only from a
command-free walker, the pilot `follow_direction` recipes are superseded by
the follow nodes planned to continue the walk. No committed config trains a
command-free walker on the current interface (every walk declares a
`[commands]` table, and `--override commands.*` is refused), so until a
follow or terrain node with a command exists, or a command-free walk config
is added, the terrain recipes have no parent a committed path produces.

During training, the walk's evaluation env is reseeded before every
evaluation, so each evaluation rolls the same commands and the same number of
stand episodes, and `robust_best_model` scores the cruise and the stand
episodes apart (each family's mean minus its standard deviation, weighted
0.85 / 0.15 by the stop probability): a stand episode earns several times a
cruise episode, and pooled, that gap would read as risk.

## The stage manifest

Each model declares its graph in `configs/<species>/<embodiment>/stages.toml` under
schema `mesozoic.stage-manifest/v2`. Per `[[stages]]` entry:

| Key | Meaning |
|---|---|
| `id` | The stage's identity; an open vocabulary matching `^[a-z][a-z0-9_]*$`. The four ids `stance`, `recovery`, `locomotion` and `behavior` are reserved and keep their meaning everywhere. |
| `config` | The stage TOML file, relative to the model's config directory. |
| `legacy_number` | The integer the stage was known as before manifests existed (1, 2; 3 was the removed hunt); absent on a stage that is addressed by id only. |
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
recipe = "stance"
deliverable = true               # the six-heading certificate since D-D32 (v2 since physics r8, D-D24); no r10 seed yet (bar 2)

[[stages]]                       # balance under pushes
id = "recovery"
config = "recovery.toml"
warm_start_from = "stance"
recipe = "recover"
deliverable = true               # recovery_quality/v1, judged against a per-run frozen resolution

[[stages]]
id = "locomotion"
config = "locomotion.toml"
legacy_number = 2
warm_start_from = "stance"       # the 2026-08-23 lineage rule, now declared;
recipe = "walk"                  # flips to "recovery" once a recovery run passes (plan A2)
deliverable = true
```

The anatomical Compsognathus and the Compsognathus robot commit the same
three-node layout, labels included. The anatomical Velociraptor, Brachiosaurus,
Dibothrosuchus and Allosaurus commit two nodes — `stance` (labelled `stance`)
and `locomotion` (`walk`, warm-started from stance) — over `stance.toml` and
`locomotion.toml`; every stage file is named by its stage, with no number.
Every node in every committed manifest is a deliverable.

The labels are `stance`, `recover` and `walk`. Each names one node
on every model that has it, so label resolution never surprises: `"stance"`
resolves to `stance` on every model (it is both the stance node's id and its
label), `"recover"` to `recovery` on the T-Rex and both Compsognathus models
(chain `stance -> recovery`) and to nothing elsewhere, and `"walk"` to
`locomotion` on every model. A label may equal a stage id only when it resolves to that very
stage, as `stance` does; the loader refuses any other overlap, so a label
never shadows a different node's id. Until decision D-D35 (2026-10-16) the
recovery node was labelled `stand` too, so `"stand"` resolved to `recovery`
(the deepest `stand` deliverable) on those three models and trained stance
then push recovery there but stance alone elsewhere; the push-robust standing
policy those three publish is now the `recover` deliverable. The stance node
itself was labelled `stand` until decision D-D38 (2026-10-18) gave the
standing behavior one name, `stance`, everywhere: `"stand"` is no alias, and
every place that takes a label (the notebook's `BEHAVIOR`, `curriculum
--target`, a manifest's `recipe`, the species manifest's deliverables)
refuses it with `'stand' was renamed to 'stance' (decision D-D38); use
'stance'`. The hunt label is refused the same way since decision D-D59, with
`'hunt' was removed with every node it labelled (decision D-D59); use 'walk',
the behavior those nodes were trained from`. Inside a run, stage directories
are named `{position:02d}_{id}` (`01_stance`, `02_locomotion` for the two-node
models; `01_stance`, `02_recovery`, `03_locomotion` on the T-Rex and both
Compsognathus models; an older run may also hold `03_behavior` or
`04_behavior`), and file labels keep `stage1`, `stage2` for the
legacy-numbered stages and the bare id otherwise (`recovery_final.zip`).

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
   `"auto"` (the notebook default, decision D-A25), the run (in the notebook,
   one of the certification tree `<LOG_BASE>/<species>/<embodiment>/<algorithm>/`,
   whatever `RUN_KIND` says; on the command line, a sibling of the run
   directory) whose certified ancestors cover
   the most of the chain root-first, the greatest run directory name on a
   tie (the newest timestamp id; a custom `RUN_ID` such as `my_run` sorts
   after every timestamp). The selection prints
   the run it chose, the replication each reused node rests on, the runs it
   refused with the rule that refused them (the first 20; the selection
   object holds every run scanned), and any run whose root passed under an
   older policy interface as a candidate for the command-line widen tool
   (`widen_checkpoint --max-revision-gap N`).
2. **Judge** a node that was trained but never gated (the notebook only:
   its final checkpoint exists but `gate_verdict.json` does not, because the
   resume cell finished its budget, the runtime stopped before its verdict,
   or the command-line widen tool wrote a widened root into the run), even
   when a trunk run certifies it: the loop does not consult the trunk for an
   ancestor its run holds trained but unjudged, and judges such an ancestor
   only on the parent resolved here (cleanup decision 6 (b); a run that
   already holds the trunk's copy of it as an `ancestors/` record keeps that
   record, and a root widened beside such a record is refused, naming a new
   run id). The target, and any node in a session without a trunk, get no
   such parent check (see `docs/KNOWN_ISSUES.md`).
3. **Train** it otherwise, warm-started from its parent's handoff checkpoint
   and VecNormalize sidecar along the declared edge, then judge its gate and
   write `gate_verdict.json` beside the handoff.

The target node is never reused across runs: it is what the run exists to
certify, and an earlier run's certified target is that run's deliverable,
published from there. A node trains only on a parent that passed its gate; a
failed node stops the chain, and nothing is trained from scratch silently. In
the notebook the bundle is written before the verdict is enforced, so a
failed walk still publishes the certified nodes this run trained beneath it.
Ancestors reused from a trunk are not this run's deliverables: they stay
published by the run that certified them (D-A18) and appear here only under
`provenance.ancestors`, so a trunked run whose target fails has nothing
certified to publish.

### Before a walk: the warm-start noise scan

A walk node (`locomotion`, recipe `walk`) warm-starts PPO from its parent's
handoff and keeps the parent's learned action std, `exp(log_std)`. With too
little noise the walk explores nothing: the T-Rex's walking pilot 1 stood
still on all three seeds from a parent of std 0.095. With too much, the
parent falls at once: walking pilot 3's parent stood in 9 of 120 episodes at
a uniform `log_std` of −1.5. `[curriculum] warm_start_log_std` in the walk's
TOML resets every actuator's `log_std` at the warm start. No committed walk
stage declares it.

`environments/shared/scripts/warm_start_noise_scan.py` measures how much
noise a parent tolerates in the walk's own env, for any model, before its
first walk:

```bash
python -m environments.shared.scripts.warm_start_noise_scan \
  --species velociraptor_mongoliensis --embodiment anatomical \
  --stage-dir logs/<run>/01_stance --jobs 4 --out-dir scratch/noise_scan
```

It loads the stage's handoff pair (or `--model` and `--vecnorm`), checked
against the current plant, into the walk stage's env, built from its
committed config with the parent's VecNormalize statistics frozen. The env
and the policy are set as a fresh walk starts training. Every walk is a
live command walk, so the env draws its commands from the cruise ramp's
start: the walk's own commands at 0.25 of their speeds, each reset seed's
own draw (on the default seeds 3042–3081, three of them, 3052, 3060 and
3074, draw a stand on every committed walk, against the walk's 15% stop
probability). As training does, the scan zeroes the stance's weights on the
command inputs, which never trained, and checks that the policy then
ignores the command. So the commands change the return, never an action,
and the stand counts measure the policy the walk starts from. Two reference
settings come first: the policy's mean action with no noise, and the
parent's own learned `log_std`. Each further setting writes one value
into every actuator's `log_std`, in memory only. By default the grid has
two passes. The first rounds the parent's largest learned `log_std` to a
multiple of 0.125 and runs from 0.5 below it to 1.0 above it in steps of
0.25, 7 values. The second adds one value 0.125 above the largest that stood
every episode, as walking pilot 3 added −1.875 between its −2.0 and −1.75
(`--log-std` names other values, rolled as given). Each noisy setting rolls
seeds 3042–3081 in three noise streams, 120 episodes, with actions sampled
from the policy's own distribution, up to the walk's episode length. An
episode stands when it reaches that horizon without a termination. The tool
prints a table and writes it with a JSON file into `--out-dir`. The table
gives the stand counts per stream, the causes of the early ends, and the
lift-offs and slide of the standing episodes. Below it is the largest grid
value at which every episode stood. If a smaller value fell, the line also
names the largest value with every smaller value standing too. While it
runs, `warm_start_noise_scan.partial.json` holds the settings finished so
far, so an interrupted scan keeps what it measured; the scan creates it
before its first setting rolls, so a second scan started into the same
directory at the same time is refused before it rolls. The tool refuses an
output directory inside a run or one that already holds a scan, finished,
running or interrupted, and it changes no config. The default scan is at
most 1.12 million control steps on the T-Rex, about 26 minutes of one CPU
on the machine it was measured on; a Colab CPU runtime is slower, and
`--jobs N` divides the time. Run it as a session of its own once the
parent's run is certified and published, not inside a training session:
only a certified stance is a walk's parent, and the scan needs no GPU. An
11M-step T-Rex stance trains in about 21h11m–21h31m on an L4, and setup,
its certificate and its bundle take about 50 minutes more, so a training
session ends with about 1.6 to 2 hours of Colab's 24-hour limit left:
margin for a slow session, not time for a scan whose rate on a Colab CPU
has not been measured.

Whether to declare `warm_start_log_std` from a scan is the maintainer's
decision, per model. A value does not carry from one model or plant
revision to another: walking pilot 3's −2.0 was measured on a physics-r10
T-Rex parent, and r11 changed the T-Rex legs.

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
   current]` below. Neither writes into a run whose bundle is `complete`,
   which is immutable: such a verdict stays as published, and when rule 7
   refuses it a trunked run trains the node itself (decision D-B8 as
   amended 2026-11-24).

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
followed, or the trunk's stance would re-enter as trained here. It consults
`TRUNK_DIR` for every ancestor `RETRAIN_FROM` does not cover, except a node
`RUN_DIR` holds trained but unjudged, which it judges here instead (or
refuses as an interrupted node, or as trained on another parent than the one
resolved here); for a node the run already holds as an `ancestors/` record
the trunk is still consulted (a root widened into the run beside that record
is refused instead).

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

`--species` with `--embodiment` names the model of a directory whose
`stage_config.json` does not name it whole (a pre-D-D33 directory records no
`embodiment`; an override that disagrees with a whole record is refused),
`--stage` overrides the stage it records, and `--force` re-derives over an
existing verdict — the path for a verdict written before D-A22, which records
no `gate_sha256` and is refused by rule 7. `--gate recorded` (the default) judges under the block
`stage_config.json` recorded and digests it; `--gate current` judges under
the checkout's block for the stage and digests that one, the re-judge path
after a threshold edit for a `reward_and_length/v1` directory, whose episode
rows are re-aggregated under the new floors. A `stance_quality/v1` verdict is
read off `stance_gate_report.json` and re-derives nothing, so the tool refuses
a report whose recorded `thresholds` are not the ones being judged under
(under either `--gate`): a stance directory whose rail moved is re-judged
through the notebook's JUDGE branch, which measures a fresh panel. The tool
refuses when the evidence is missing, and cannot backfill
`recovery_quality/v1`. It refuses, before it writes anything and `--force` or
not, a stage directory whose run holds a `complete` bundle: that bundle is
immutable, and a verdict added or rewritten in it would stop its manifest
verifying.

## In the notebook

`notebooks/sb3_training.ipynb` exposes six knobs in its configuration cell,
committed as:

```python
BEHAVIOR = "auto"  # the model's final deliverable, a recipe label, or a deliverable's stage id
TRUNK_FROM = "auto"  # "auto" (D-A25): the run in this model's certification tree (<algo>/, whatever RUN_KIND says) whose certified ancestors cover the most of the chain, the greatest run directory name on a tie (the newest timestamp id), chosen in the resolve cell and printed; a run id (resolved in <algo>/) or a path pins one (a screen pins a screen as ../<algo>_screen/<id>; a certification run refuses one under <algo>_screen/ or <algo>_quick_test/); "" trains every node here
RETRAIN_FROM = ""  # optional chain node to train here with every node below it (empty = off; D-A19)
RUN_LABEL = ""  # optional free-text label recorded beside each trained node's hyperparameter digest (D-A21)
RUN_ID = ""  # "" = the run this runtime's storage cell resolved last (a fresh timestamped run on the first pass or after SPECIES, EMBODIMENT, ALGORITHM, RUN_KIND or QUICK_TEST changed); a new id starts a fresh run (letters, digits, _ and - only; refused if, in this ALGORITHM's trees, it names only another model's run: SPECIES and EMBODIMENT pick the model); an existing run id re-enters that run to finish an interrupted or partial run (section 5)
RUN_KIND = "certify"  # "certify" = a run that may certify, under <algo>/, the tree TRUNK_FROM = "auto" and seed replication scan; "screen" = a branch screen, pilot or other experiment, under <algo>_screen/: never a trunk or a seed replicate of a certification run, though it reuses certified ancestors; set it again to re-enter or resume a screen; QUICK_TEST wins (<algo>_quick_test/)
```

`RUN_ID` names the run directory under `<LOG_BASE>/<species>/<embodiment>/<algorithm>/`,
the certification tree, or under the tree `RUN_KIND` or `QUICK_TEST` moves it
to: `<algorithm>_screen/` under `RUN_KIND = "screen"` (a branch screen, a
pilot or another experiment), and `<algorithm>_quick_test/` when `QUICK_TEST`
is on, whatever `RUN_KIND` says. Trunk selection and seed-replicate discovery
never reach either, so a 50,000-step quick test, or a screen whose task, gate,
plant and recipe digests equal a landed recipe's, is never selected as a trunk
or counted as a seed replicate of a certification run (the runs of one of
those trees may count one another). The folder is the run's only record of
its kind (`environments/shared/run_tree.py`), and the Drive summary notebook
reads it into its `run_kind` column. A run id names one run of the model,
whatever its tree: a fresh runtime resets `RUN_KIND` and `QUICK_TEST`, so a
screen is re-entered or resumed with `RUN_KIND = "screen"` set again, and the
storage cell refuses, before anything is written, a `RUN_ID` that names no run
in the selected tree but a run in another. A screen pins a screen trunk as
`../<algorithm>_screen/<id>` or by its absolute path, since a bare `TRUNK_FROM`
id resolves in `<algorithm>/`. The tree covers trunk selection, replicate
discovery and the pinned trunk only: nothing refuses a certification run that
loads, resumes or widens a screen's checkpoint (the manual node's
`MANUAL_LOAD_PATH`, the widen tool, a command-line `--load`), so a
certification run never trains from one. A screen's branch needs a notebook
with the knob: on a branch cut before 2026-11-24 the storage cell ignores
`RUN_KIND` and writes the run into `<algorithm>/`, so bring the change onto
the branch first (merge `main`, or apply `environments/shared/run_tree.py` and
the storage cell) and check that the storage cell's `Run directory:` line
shows `<algorithm>_screen/` before training.
`""` keeps the run this runtime's storage cell resolved last in the tree that
`SPECIES`, `EMBODIMENT`, `ALGORITHM`, `RUN_KIND` and `QUICK_TEST` select and mints a fresh timestamped
run when there is none there (the first pass in a runtime, or after one of
those changed),
a new id starts a fresh run, and an existing run's id re-enters that run in
place, to resume an interrupted node or finish a `partial` run; the storage
cell prints whether it minted a new run or re-entered one. To start a fresh
run in a runtime that already resolved one, set a new id (a timestamp such
as `"20260924_120000"`, the format the storage cell mints). `SPECIES` and
`EMBODIMENT` select the model, never the id (the maintainer's decisions of
2026-11-26). Before anything is written, the storage cell refuses an
explicit id that names no run of the model but a run of another model under
`<LOG_BASE>`, in the trees of this `ALGORITHM`, naming every run of that id
and the knobs that reach it; and it refuses a new id with characters other
than letters, digits, `_` and `-` (a run the tree already holds is
re-entered whatever its characters, with a note). Every explicit id prints
a banner naming the model, its species and embodiment, the algorithm, the
run tree, whether the run is new or re-entered, and every other run of that
id under `<LOG_BASE>`, of any model and either algorithm (a run of the id
under the other algorithm alone is named, not refused). These rules cover
the notebook's `RUN_ID` only: a run directory named on the command line
(`--output-dir`, or `widen_checkpoint --to-stage-dir` below) is not checked,
so name it by the same rule. A run whose bundle
is `complete` is immutable: a session that would judge or train a node into
it is refused before anything is trained or written (in the resolve cell, or
in the chain loop for what the resolve cell cannot predict, such as a node
held only as an `ancestors/` record that the trunk no longer certifies), and
the refusal names the remedy — a fresh
`RUN_ID` with `TRUNK_FROM` set to the complete run, which reuses its
certified nodes across runs.

A checkpoint trained behind this checkout's policy interface is widened on the
command line into a new run, one the notebook has not opened yet (`python -m
environments.shared.scripts.widen_checkpoint ... --to-stage-dir
<LOG_BASE>/<species>/<embodiment>/<algorithm>/<new run id>/01_stance`, with
`<algorithm>_screen/` in place of `<algorithm>/` for a screen, where `<new run id>`
is a new timestamp id `YYYYMMDD_HHMMSS` that no run uses yet, with
`--max-revision-gap N` for a parent more than one revision behind, decision
D-C17, and `--label` when the session sets `RUN_LABEL`; on Colab, the three
steps in the tool's module docstring: the notebook's section 1, a scratch
cell that mounts Drive, never the storage cell, then the tool from
`/content/mesozoic-labs`), which the notebook then re-enters
with `RUN_ID` set to that id, `SEED` set to the parent's seed and
`TRUNK_FROM = ""`: the storage cell refuses any other seed before it writes
anything (D-C14), and the chain loop judges the widened root before it
consults any trunk (decisions D-C13, D-D14, and the cleanup's decision
6 (b)). The notebook's `WIDEN_FROM` / `WIDEN_MAX_REVISION_GAP` knobs and widen
cell were removed once the two pending widen sessions had run.

`TRUNK_FROM` defaults to `"auto"`: once the chain is resolved, the resolve
cell selects the run under `<LOG_BASE>/<species>/<embodiment>/<algorithm>/` (the
certification tree, whatever `RUN_KIND` says, so a screen builds on certified
ancestors) whose certified ancestors cover the most of the chain root-first (the greatest run
directory name on a tie, the newest timestamp id)
and prints the choice, what it rests on and every refusal; `""` turns reuse
off. Whatever the trunk, the resolve cell then records it in the run's
`trunk_run.json`, as the `TRUNK_FROM` value that reproduces it (cleanup
decision 4 (a)): never in a `complete` run, and never once the run holds
an `ancestors/` record, where another trunk than the recorded one is
warned about (the trunk those records came through when the cells ran in
order; see `docs/KNOWN_ISSUES.md`) and a missing file is never guessed. A
pinned `TRUNK_FROM` is a run id, resolved under the same directory, or an
absolute path to a run directory. A certification run's trunk is a
certification run: one under `<algorithm>_screen/` or `<algorithm>_quick_test/`
is refused before anything is written, while a screen or a quick test may pin
either. It must be a run whose
`provenance.json` names the same species, embodiment, algorithm and backend, whatever its
bundle status (a node is reused only when the reuse rule certifies it), and it
must not be this run — certified nodes of an earlier run come in through
`TRUNK_FROM`, never by pointing `RUN_ID` at that run. `RETRAIN_FROM` must name
a node on the chosen behavior's chain.

The setup cell resolves `"auto"` to the last deliverable in the model manifest.
This selects `walk` on every model. Explicit
recipe labels and stage IDs go through `MANIFEST.resolve_behavior(BEHAVIOR)`.
The cell obtains the chain with `MANIFEST.chain_for(TARGET_NODE.id)`. One `# ===== BEHAVIOR CHAIN LOOP =====`
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

**Publishing a run the loop does not write.** A node the loop reuses from
this run's own records writes nothing, so a session that reuses every node
writes no bundle: a run re-entered after a runtime ended between its last
verdict and its bundle (a stance node's probes run last and are report only),
or a `partial` run re-entered to count a replicate that certified later. The
opt-in publish cell after the loop (`PUBLISH_RUN = True`; the maintainer's
decision of 2026-11-26) writes the training summary and the bundle from the
run's own records with the loop's two calls, checks the bundle and prints its
status and what it certified. Set `RUN_ID` to the run and every other knob as
its session had them (`ALGORITHM`, `SEED` and `N_ENVS` too, and `BEHAVIOR`
above all: it sets the chain and the target). Then run every cell above
section 5, the publish cell, and last the auto-disconnect cell (section 9),
which in Colab flushes Drive's buffered writes before it releases the
runtime (with `AUTO_DISCONNECT = False`, `drive.flush_and_unmount()` does
the flush by hand). Skip the loop: it raises on a failed
node, which most `partial` runs hold, and trains any node the run lacks. Run
all works too when every node of the chain passed. The cell trains, judges and
probes nothing and changes no stage file. It walks the chain as the loop does,
an ancestor's `ancestors/` record standing for it, and stops after the first
failed verdict, where the loop halted. Before it writes anything it refuses a
`complete` run (already published and immutable), a `BEHAVIOR` whose target is
not the one the run's `provenance.json` records (publishing would retarget the
run), a run that holds no node of the model (usually `RUN_ID = ""` in a fresh
runtime, where the storage cell opens a new run), a node on the chain, up to
the first failed verdict, that is neither judged in the run (a
`gate_verdict.json` in its stage directory) nor recorded under `ancestors/`
(the loop judges or trains it first), a node judged in the run that the
bundle would leave out, and a node whose verdict passed but whose stage no
longer hands on the pair the verdict hashes (the loop's reuse refuses it too;
on the command-driven walk, the pair its handoff choice,
`models/handoff_choice.json`, names). A judged node's results re-enter the
bundle from its verdict, as the loop's reuse does (the walk's from its
`command_tracking/v1` verdict), and a missing report-only file, such as a
probe the cut-off skipped, does not block it.

**Walk on a stance trunk.** An earlier run `20260901_120000` certified
stance for the Velociraptor under PPO. In a fresh session:

```python
BEHAVIOR = "walk"
TRUNK_FROM = "20260901_120000"
```

The chain is `stance -> locomotion`. The loop reuses the trunk's stance if
its verdict passed, its task digest and plant match, and it entered from no
parent; then trains `locomotion` here, warm-started from the reused stance
handoff (a walk warm-started from a stance begins blind to its command and
learns to read it), and judges it after its budget on its `command_tracking/v1`
panel. The run directory gains `ancestors/stance/`, the trained
`02_locomotion/` records `parent_run_id`, and the bundle publishes
`locomotion` as its one deliverable (`target_deliverable = "2"`, its stage
key), with `provenance.ancestors` carrying stance — certified through the
trunk's verdict and still published by the trunk run.

**Recover on the T-Rex.** `"recover"` resolves to `recovery`, on the T-Rex
and both Compsognathus models:

```python
BEHAVIOR = "recover"
```

The chain is `stance -> recovery`. Stance trains (or is reused), then the
recovery resolution is frozen from the stance handoff, recovery trains, the
frozen panel is rolled, and the `recovery_quality/v1` verdict is enforced
exactly like any other node's. `BEHAVIOR = "stance"` trains the stance node
alone, on these models as on every other. The old
`RUN_RECOVERY_STAGE` switch no longer exists.

Two escape hatches remain. The manual single-node cell (`MANUAL_NODE`) trains
one node outside the chain and records its verdict without enforcing it; it
never feeds the chain. It refuses, before it writes anything, a run whose
bundle is `complete` and a node that already holds a gate verdict, whatever
`MANUAL_LOAD_MODE` says: a judged node is never trained further in place,
so a new attempt is a fresh `RUN_ID`. The resume cell (`RESUME_STAGE`)
continues an interrupted node from its newest intact periodic checkpoint under
`resume_same_stage` and evaluates nothing (the loop's JUDGE branch evaluates
the node from disk); the re-saved `stage_config.json` keeps the load lineage
the record holds (an entered node's edge, or a root's first load) and
records the continued-from checkpoint under `resume_load_path` /
`resume_checkpoint_sha256`, so a resumed-then-judged node still chains by
digest and stays reusable. The record also keeps the first session's
`git_commit`, `library_version`, `run.timesteps` and `run.label`, and each
resume session's commit, library version, timesteps and continued-from
checkpoint are appended to `run.resume_sessions` once its model has loaded
(a resume refused or failed before training adds none). Only a PPO node is
resumed: SAC restarts its step counter on a load (its replay buffer is not
saved), so the cell refuses an `ALGORITHM` other than `"ppo"`, and `train`
refuses such a load into a directory holding the node's records. The trainer
refuses, before it writes anything, a checkpoint trained on another task, a
resume whose `hyperparameters_sha256` differs from the node's record, naming
the keys, and a `RUN_LABEL` other than the recorded label (an empty one keeps
it). A node first trained at a commit from before 2026-11-24 resumes at
`main` like any other: these checks compare it with its record, and the
re-saved record keeps its first session. The cell warns
when `BEHAVIOR` is `"auto"`, when the run's `provenance.json` records another
target deliverable than `BEHAVIOR` resolves to, and when the checkout's
commit is not the one the node records; it trains the same node either way,
so to keep the run's target, interrupt the run before the chain loop starts,
set `BEHAVIOR` and Run all again. It sits ahead of the chain loop, so a
resume is: leave `REPO_REF` at `main` (pinning the `git_commit` the node's
`stage_config.json` records works only for a commit that has
`environments/shared/run_tree.py`, which the storage cell imports), set
`RUN_ID` to the interrupted run and
`RESUME_STAGE` to its node, with `RUN_LABEL` and `RETRAIN_FROM` empty,
`RUN_KIND`, `QUICK_TEST`, `CRITIC_PHASE` and the other recipe knobs as the
interrupted session had them, and `TRUNK_FROM` set to the value the run's `trunk_run.json` records
(unless the run's `ancestors/` records came through another trunk than the
file names: then remove the file and pin by the nearest ancestor record, as
`docs/KNOWN_ISSUES.md` describes; a run opened before that file existed is
pinned by hand to the run the interrupted session resolved, `""` only when
it trained every node itself), then Run all; the loop judges the node after
the resume trains it, before it consults any trunk, so a node trained here
although that trunk run certifies it is resumed the same way.
Once the run holds an `ancestors/` record, the resume cell refuses a resume
under another trunk than `trunk_run.json` names, and a resume of a node the
run holds as such a record (the record stays that node in the run: the loop
takes it while a trunk certifies the node, and a bundle refuses a node both
reused and trained here). A node that already holds
`gate_verdict.json` or an intact final checkpoint pair is never retrained: the
cell trains nothing and the loop reuses, refuses or judges it (decision
D-D16).

## On the command line

The `curriculum` command runs the model's advancing stages in manifest
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
advanced once per node walked: `--target recover` on the T-Rex or
Compsognathus, whose chain runs through the non-advancing recovery node, is
refused before any directory is written, and that chain is trained through the
notebook (`BEHAVIOR = "recover"`), while `--target stance` is a stance-only run
on every model; a chain that skips a ladder stage (an edge
rewired past it) is refused the same way, since the manager would judge the
node after the gap against the skipped stage's thresholds. `--retrain-from`
must name a node on the target's chain.

```bash
# Reuse an earlier run's certified stance; train locomotion here
python -m environments.shared.train --species velociraptor_mongoliensis --embodiment anatomical \
  curriculum --algorithm ppo --trunk-from logs/<earlier_run> --output-dir logs/<new_run>

# A stance-only run: train the stance here as the target, stop
python -m environments.shared.train --species velociraptor_mongoliensis --embodiment anatomical \
  curriculum --algorithm ppo --target stance --output-dir logs/<stance_run>

# Tag every trained node (recorded beside its hyperparameter digest)
python -m environments.shared.train --species velociraptor_mongoliensis --embodiment anatomical \
  curriculum --algorithm ppo --label "lr-3e-4"

# One node by hand, entering from its declared parent's handoff
python -m environments.shared.train --species velociraptor_mongoliensis --embodiment anatomical \
  train --stage locomotion --algorithm ppo \
  --load logs/<run>/01_stance/models/robust_best_model.zip \
  --load-mode initialize_next_stage

# The non-advancing recovery node (T-Rex, Compsognathus): never part of the CLI curriculum
python -m environments.shared.train --species tyrannosaurus_rex --embodiment anatomical \
  train --stage recovery --algorithm ppo \
  --load logs/<run>/01_stance/models/robust_best_model.zip \
  --load-mode initialize_next_stage
```

`--stage` accepts a legacy number or a stage id; `--retrain-from` accepts
either too. `--load` takes the checkpoint (`.zip` or its stem) and finds its
VecNormalize sidecar beside it as `<stem>_vecnorm.pkl`; the handoff the
curriculum promotes is `robust_best_model`, else `best_model`. Every model
trains through this one entry point,
`python -m environments.shared.train --species <species> --embodiment <embodiment> <subcommand> ...`,
and both identity flags are required.

A single-node `train` run is unjudged: `train()` records the stage config,
lineage and checkpoint pair but writes no `gate_verdict.json` (the
curriculum, `generate_stage_artifacts` and the backfill tool are what write
one), and it writes into its own `--output-dir` — by default
`logs/<species>/<embodiment>/<stage_dir>_<timestamp>/` under the repository
root — not into a `<run>/01_stance/`
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
  `--load <checkpoint> --load-mode resume_same_stage` of a stage that holds no
  `gate_verdict.json`: a judged node is never trained further in place, under
  any load mode (decision D-A20 as amended 2026-11-24). `train` checks this
  before it writes anything; the curriculum checks each node when the walk
  reaches it, before that stage directory is written to — by then the run
  directory, its `plant_identity.json` and any earlier node already exist.
  The curriculum has no resume mode, so it always needs a fresh
  `--output-dir`.
- A same-stage resume is refused, before `train` writes anything, when it
  cannot continue the stage the directory records: a `--load` that is not a
  continuation into a directory that already holds that stage's training
  records (`evaluations.npz`, an episode ledger, its periodic or final
  checkpoints, a best-model pair), that is a SAC load (its replay buffer is
  not saved, so its step counter restarts) or a curated PPO checkpoint
  (`best_model.zip`, `<label>_final.zip`: no step count in the name); a
  checkpoint trained on another task; and any resume whose
  `hyperparameters_sha256` or `--label` differs from the directory's
  `stage_config.json`. Such a `--load` into a fresh directory (the default
  without `--output-dir`) trains.
- A same-stage resume of a live-command stage (the walk) under a `--seed`
  other than the run's is refused before `train` writes anything (decision
  D-D62): its in-training evaluation panel is drawn from that seed. The seed
  is read from the directory's `stage_config.json`, else from the one beside
  the checkpoint, so the refusal holds for a resume into a fresh directory
  too. Stance and recovery (`command_mode = "none"`) are not refused.
- Non-advancing nodes (no legacy number — recovery) are skipped by the CLI
  curriculum with a log line in Phase A and trained on their own with
  `train --stage recovery`.

A CLI curriculum run writes `curriculum_results.csv` and a `gate_verdict.json`
per trained node but no `provenance.json` or `summary.json`; the result
bundle described below is written by the notebook. A CLI run can still serve
as a later CLI run's `--trunk-from` (it is named by its directory), and a
notebook run can serve as a CLI trunk. A pinned notebook `TRUNK_FROM`
requires a run with a `provenance.json`; `"auto"` judges siblings by their
stage records and refuses one whose `provenance.json` names another species
or embodiment, algorithm or backend.

## What a run directory holds

After a notebook walk run that reused a trunk's stance, a T-Rex run directory
(under `<LOG_BASE>/tyrannosaurus_rex/anatomical/ppo/`) looks like this:

```
20260912_120000/
├── provenance.json             deliverables {locomotion}, with certified;
│                               ancestors {stance} — reused, still published by run 20260901_120000
├── plant_identity.json, trunk_run.json, zero_action_baseline.json,
│   collected_results.csv, training_summary.txt
├── summary.json                written whenever anything certified; status complete | partial
├── artifact_manifest.json      the bundle's status and every file's hash; written last
├── ancestors/
│   └── stance/                 reused from run 20260901_120000: ancestor.json + copies of its
│       ├── ancestor.json         gate_verdict, stage_config, task_fingerprint, plant_identity
│       └── ...                   (never the checkpoint)
└── 03_locomotion/              NN is the node's manifest position; 02 is recovery, off this chain
    ├── models/                 the robust_best_model, best_model and final pairs (.zip + _vecnorm.pkl)
    │                             and handoff_choice.json, the pair the gated statistic chose (D-D61)
    ├── stage_config.json       run block: load_mode = initialize_next_stage,
    │                             parent_checkpoint_sha256 = the stance digest, parent_run_id
    └── gate_verdict.json       passed, hash-bound to the handoff pair above; the target,
                                  always trained here
```

A run that trained its stance holds `01_stance/` instead of
`ancestors/stance/`. A run can hold both when its own stance verdict failed or
no longer passes the reuse rule and the chain loop then reused the trunk's
certified stance: the record stands for the node, and the bundle records
stance as reused (published by the trunk run), never as trained in this run.
[docs/DRIVE_LAYOUT.md](https://github.com/kuds/mesozoic-labs/blob/main/docs/DRIVE_LAYOUT.md)
maps the whole tree under `<LOG_BASE>`, from the model directory down to each
stage directory's files, with what writes and what reads each one.

The `run` block of each `stage_config.json` records:

| Key | Meaning |
|---|---|
| `load_path`, `load_mode`, `parent_checkpoint_sha256`, `parent_task_sha256` | Where the node's initial weights came from; a from-scratch root writes none of them. |
| `parent_run_id` | Present only when the parent was a certified ancestor reused from another run. |
| `resume_load_path`, `resume_checkpoint_sha256` | The periodic checkpoint the latest same-stage resume continued from, when the record already held a load: the keys above stay that load (an entered node's edge, or a root's first load). A root trained from scratch records its first resume's load under the keys above instead. |
| `resume_sessions` | One entry per same-stage resume into this directory that reached training, oldest first: its `git_commit`, `library_version`, `timesteps` (the steps it was set to train), `resume_load_path`, `resume_checkpoint_sha256` and `resumed_at`. The top-level `git_commit` and `library_version` and the run block's `timesteps` and `label` stay the first session's. |
| `hyperparameters_sha256` | A digest over the stage's `[ppo]` or `[sac]` block plus its `warmup_` / `ramp_` shaping keys, key-order independent, untouched by env kwargs, gate thresholds and the budget. Always written; a same-stage resume under another digest is refused. |
| `label` | The free-text label from `--label` / `RUN_LABEL`, when one was given. A same-stage resume keeps it; another non-empty label is refused. |
| `duration_seconds` | Seconds from `train_base.train`'s start (for a `curriculum` node, from the point the node is trained rather than reused) to the stage's final save, summed over the sessions that reached a final save in this stage directory: a same-stage resume into the same directory adds its session (D-A15). A notebook session stopped before its final save records nothing, so an interrupted-then-resumed node reports the resumed session only; a CLI resume into a fresh `--output-dir` records its own session. Written by the CLI `train` and `curriculum` subcommands (every node a curriculum trains, one a Ctrl-C ended included) and the notebook's `train_stage`. `curriculum_results.csv`'s `training_duration_seconds` is the time `learn()` alone took. |

`gate_verdict.json` (schema `mesozoic.gate-verdict/v2`) records the species
and embodiment, stage and stage id, the gate kind and schema version, `passed` and the list
of failures, the handoff checkpoint and its normalization sidecar with their
digests, the `task_sha256` it was judged on, when it was judged and
`judged_by` (the post-stage judge, the in-training manager, or `backfill`).

## Publication per deliverable

Result schema v5 (`summary.json`, v4's per-deliverable shape with the
`embodiment` field; `provenance.json` is provenance schema 2). The readers
accept result schema 5 alone: decision D-D63 retired result schema 2, the
integer-stage form of the summaries committed until D-D59, which the tree
at commit `6c9982d81c8a` (version 0.4.0.dev2, before D-D59 removed the
stage 3 they record) still reads:

- `provenance.deliverables` maps each deliverable the run holds to
  `model_path`, `model_hash`, `normalization_hash`, `gate_kind`, `certified`
  and `replication`, plus the optional `hyperparameters_sha256` and `label`
  copied from the stage's run block, and `certification_seeds` and
  `provisional`, which the bundle writer derives — the stage's declared
  `[curriculum]` bar (default 1) and `replication.count < certification_seeds`
  — rather than copying from the run block. `provenance.ancestors` is the
  summary-side projection of `ancestors/<stage_id>/`.
- `replication` is writer-recorded (decision D-B10). When the notebook
  writes the run's bundle, `environments.shared.replication.discover_replicates_for_run`
  looks through the sibling runs under `LOG_BASE/<species>/<embodiment>/<algo>/` (for a
  screen or a quick test, its own `<algo>_screen/` or `<algo>_quick_test/` tree, so
  neither is ever a replicate of a certification run) for the
  same recipe on another seed — a passed, reusable `gate_verdict.json` for
  the node with equal `task_sha256` and `gate_sha256`, a `stage_config.json`
  with the same plant identity and `hyperparameters_sha256` (derived from
  the recorded blocks when a sibling saved before that field existed), and
  a different training seed (decision D-B16) — and `replication.runs` lists
  this run first, then those replicates. A sibling whose verdict has no
  `gate_sha256` is skipped until re-backfilled (rule 7). A deliverable with
  fewer runs than its stage's `certification_seeds` is `provisional` (the T-Rex
  stance declares 2; once a stance bundle is published the catalog and the
  model page render it as `1 run of 2 seeds; provisional`, re-deriving
  the label from the current config — no T-Rex stance bundle is committed
  yet; see `docs/KNOWN_ISSUES.md`). A replicate that certifies later is counted when the
  bundle of the run that should count it is written again with the sibling
  present: a `partial` bundle is rebuilt, and a `complete` bundle
  regenerates its derived artifacts when the replication record is its
  only change — anything else on a complete bundle is still refused as
  immutable. In the SB3 notebook the publish cell writes a `partial` run's
  bundle again, a chain stopped at a failed node included
  (`PUBLISH_RUN = True`, [In the notebook](#in-the-notebook)); it refuses a
  `complete` run, whose record a hand-typed
  `save_run_bundle(chain_results(), model_id=MODEL_ID)` call regenerates
  once the chain loop has re-entered the run, in a session run up to the
  loop rather than Run all, which ends in the notebook's auto-disconnect.
- `target_deliverable` names the node the run aimed at and
  `primary_deliverable` the published model — the target when certified,
  else the deepest certified deliverable; `selected_model_path` is the
  primary's.
- Bundle status is `complete` when the target is present and certified and
  every present deliverable is certified, `partial` when at least one
  deliverable is certified, and `failed` otherwise. `summary.json` is written
  whenever something is certified, so a failed walk publishes the certified
  stance it trained, a stance-only run is a complete bundle when stance was
  its target, and a run that targeted walk but certified only stance is
  `partial`, never `complete`. Deliverables are the nodes trained or judged in this run; a
  reused ancestor is listed under `provenance.ancestors` only, so a trunked
  run whose target fails is `failed` and writes no `summary.json` — its
  trunk is already published by the run that certified it.
- `validate_result_bundle(..., require_publishable=True)` requires at least
  one certified deliverable; `require_complete=True` keeps meaning
  `complete`. The audit reports `canonical-partial` for a bundle with at
  least one certified deliverable.

The species catalog (schema 5 today) and `configs/species_manifest.toml`
(schema 3 today) follow: stage rows carry `deliverable`, `warm_start_from` and `recipe`; the
generated species catalog page (`docs/SPECIES_CATALOG.md`) and each model
page's stage table show Recipe and Warm-start-from columns (generated,
never hand-edited); result rows list one entry per deliverable
headlined by its gate kind, with stance and recovery headline values
rendered as null until a later phase exports per-stage gate metrics
(decision D-B15); and stage
videos are keyed by stage id. A result that certifies no deliverable is
never published. A historical ladder summary published `deliverables = []`
and was never relabelled certified; the four committed ones, whose last
stage was a hunt, were deleted by decision D-D59 (the last commit holding
them is named in `results/README.md`), and decision D-D63 retired their
schema.

## Experiments on a node that already passes

Certification belongs to a checkpoint for a task. The task digest covers
the species and embodiment, stage, backend, plant, the effective `[env]` block (every
reward weight, episode length, reset noise) and the push schedule. It does
not cover the `[ppo]` / `[sac]` block, the timesteps budget, or the gate
thresholds. That one fact decides what an edit does to a trunked run:

| You edit | Task digest | What a trunked run does |
|---|---|---|
| Algorithm block (`[ppo]`, `[sac]`, `warmup_` / `ramp_` shaping) | unchanged | Reuses the old certified checkpoint unless the node is the run's target or `--retrain-from` / `RETRAIN_FROM` covers it. The reuse prints a warning naming the differing keys (`ppo.learning_rate`, `shaping.warmup_timesteps`) and pointing at the retrain knob — never a refusal. |
| Environment block (`[env]`, pushes, plant) | changes | Rule 3 refuses the old checkpoint; the node and everything below it retrain. Older checkpoints stay valid in the runs that certified them. |
| Gate thresholds | unchanged | Rule 7 refuses the old verdict, naming the thresholds that differ (`gate_sha256` no longer matches). Re-judge the directory under the current gate — the notebook JUDGE branch / `generate_stage_artifacts`, or `backfill_gate_verdict.py --force --gate current` for a `reward_and_length/v1` directory — never retrain; until then a trunked run trains the node itself. A run whose bundle is `complete` is never re-judged in place, so for its verdict the trunked run's training is the only route. |

Two variants of one node are two runs: writing into a stage directory that
already records a node is refused, so give each variant a fresh
`--output-dir` / `RUN_ID`. Nothing ranks variants — compare their bundles.
To try new algorithm values on a node the trunk already certified, retrain
from it:

```bash
python -m environments.shared.train --species velociraptor_mongoliensis --embodiment anatomical \
  curriculum --algorithm ppo --trunk-from logs/<certified_run> --retrain-from locomotion \
  --output-dir logs/<variant_run>
```

The variant reuses stance as an ancestor and trains locomotion here. A
later run of a node built on locomotion, trunked from `<variant_run>`, reuses
the variant's locomotion and, through the variant's `ancestors/stance/`
record, the stance that `<certified_run>` certified (rule 1 above): the
winning locomotion carries into later runs without retraining anything,
provided the original run is still visible from the machine — as recorded,
or beside the variant under the same log base.

## What stays on the ladder in Phase A

- The in-training `CurriculumManager` stays integer-keyed: it judges the
  advancing nodes during training, and semantic-id nodes (recovery) are
  judged after the stage.
- Stance and recovery deliverables headline a metric name with a null value
  until per-stage gate metrics are exported into the summary (a later phase,
  decision D-B15). A verdict records the gate it was judged under (`gate` /
  `gate_sha256`, decision D-A22), but the deliverable's summary does not
  surface it yet. The hunt deliverable was the exception, exporting its
  `task_success/v1` bound into the stage row, until decision D-D59 removed
  the hunt and, with it, that gate kind and its export.
