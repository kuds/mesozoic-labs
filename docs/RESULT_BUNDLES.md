# Result Bundles

Mesozoic Labs training runs are produced in Google Colab, persisted to Google
Drive, and promoted into `results/` later. A result bundle preserves the
evidence needed to make that hand-off reproducible.

## Two validation levels

- **Drive validation** accepts partial and gate-failed curricula. These runs
  retain their configs, checkpoints, raw evaluation episodes, and provenance
  even though they may not be publishable.
- **Promotion validation** (`validate_result_bundle(..., require_publishable=True)`)
  requires at least one *certified deliverable*, a selected checkpoint (and
  SB3 VecNormalize sidecar) for every recorded stage, every resolved stage
  config with plant identity, recorded selected- and terminal-policy
  evaluation episodes for every recorded stage, a canonical `summary.json`,
  matching CSV metrics, and valid artifact hashes.

Publication is **per deliverable** (result schema v4, and v5 since decision
D-D33 added the `embodiment` field;
[BEHAVIOR_RECIPES_PLAN §4.3](BEHAVIOR_RECIPES_PLAN.md)). The model's stage
manifest flags which nodes are deliverables and which node each one
warm-starts from. A deliverable is *certified* when its own gate passed and
every `warm_start_from` ancestor is present — trained in the bundle, or
carried by an `ancestors/<stage_id>/` record of a node reused from another run
— with a passed gate. Absence never reads as a pass.

Every bundle save names a **target deliverable** (`save_result_bundle(...,
target_deliverable=...)`; the manifest's last deliverable by default, the
notebook's `BEHAVIOR` node in the chain loop). The bundle status is:

| status | meaning |
|---|---|
| `complete` | the target is present and certified **and** every deliverable present in the run is certified — the bundle is then immutable |
| `partial` | at least one deliverable is certified, but not the target or not every present one — a stance-only run targeting walk, a run whose walk failed above a stance it trained and certified |
| `failed` | no deliverable is certified — a failed root, a trunk whose gate failed below every leaf, or a trunked run whose only trained node (its target) failed |

`summary.json` is written whenever at least one deliverable is certified, so a
failed leaf still publishes the certified trunk trained in the same run (a
trunk reused from another run is recorded under `provenance.ancestors`, not
republished) and a walk-only run writes a valid bundle. The **primary deliverable** — `provenance.selected_model_path`
/ `model_hash` and the headline `final_avg_reward` — is the target when it is
certified, else the deepest certified deliverable in manifest order; it is
always a certified deliverable, never a failed leaf.

`require_complete=True` (the notebook's completion cell) demands
`bundle_status == "complete"`, audited as `canonical-valid`;
`require_publishable=True` (the audit, the bundle writer, the species
catalog, the notebook's gate-failure branch) also accepts a partial bundle
with its summary, audited as `canonical-partial`. A partial bundle must not be
presented as a complete result. Run `20260720_203454` is a concrete
failed-but-auditable example; its role in the Velociraptor diagnosis is
documented in the
[Stage-1 basin investigation](investigations/VELOCIRAPTOR_STAGE1_BASIN_INVESTIGATION.md).

Under a v1 stage manifest (one deliverable — the last advancing node — with
edges to the previous advancing node) these rules collapse to the historical
"every advancing stage present and passed / terminal = last advancing stage"
rule. These are the only rules: the readers accept schema 5 alone (decision
D-D63 retired schema 2, which had its own).

## Canonical layout

Human-facing labels and the shared stance/recovery replay layout are
specified in [Training artifact names and stance replays](TRAINING_ARTIFACT_PRESENTATION.md),
and [DRIVE_LAYOUT.md](DRIVE_LAYOUT.md) maps every file of a run with its
writer and its readers.

A node trained or judged in the run has its stage directory, and a node
reused from another run has an `ancestors/<stage_id>/` record. The bundle
writer refuses a bundle that records one node both ways
(`reporting/bundles.py`, `result_schema.py`). A Tyrannosaurus rex walk run
that trained its stance:

```text
<run-id>/
├── provenance.json
├── plant_identity.json
├── trunk_run.json               # the trunk the notebook session resolved (mesozoic.trunk-run/v1)
├── zero_action_baseline.json    # the zero-action cell's copy of this model's statue baseline
├── collected_results.csv
├── summary.json                 # written whenever >= 1 deliverable is certified
├── training_summary.txt
├── artifact_manifest.json       # status marker (complete / partial / failed); written last
├── 01_stance/                   # NN_<id>: NN is the node's manifest position (stage{N} before 2026-08-20)
│   ├── stage_config.json
│   ├── gate_verdict.json        # per-node verdict, hash-bound to the handoff pair
│   ├── evaluation_final.csv
│   ├── evaluation_selected.csv
│   └── models/
└── 03_locomotion/               # the walk, the last stage since D-D59; 02 is recovery, off this chain
```

The same walk with its stance reused from the trunk run holds the same run
files and an `ancestors/stance/` record:

```text
<run-id>/
├── ...
├── ancestors/                   # nodes reused from another run — records, never checkpoints
│   └── stance/
│       ├── ancestor.json        # mesozoic.ancestor-record/v1
│       ├── gate_verdict.json    # verbatim copies of the ancestor stage's files
│       ├── stage_config.json
│       ├── task_fingerprint.json
│       └── plant_identity.json
└── 03_locomotion/
```

A run written before decision D-D59 removed the hunt nodes may also hold a
`04_behavior/` stage directory, the hunt its chain ended at; no current stage
resolves to it.

A run can also hold both for one node. When the run's own verdict of a node
below the target failed or no longer passes the reuse rule, the chain loop
reuses the trunk's certified node and records it beside the run's own stage
directory. The record then stands for the node: the bundle records the node
as reused, and the stage directory stays, hashed by the manifest but not
published (`tests/test_publish_cell.py`,
`test_an_ancestor_record_stands_for_the_node_beside_a_verdict_of_the_runs_own`).
The run's report shows the node as reused and names the stage directory as
not published ([DRIVE_LAYOUT.md](DRIVE_LAYOUT.md#the-reports-tree)).

`gate_verdict.json` (`mesozoic.gate-verdict/v2`) lives in the stage
directory root beside `stage_config.json` and records the node's verdict with
the SHA-256 of the handoff checkpoint and its VecNormalize sidecar; it is what
makes a node reusable as an ancestor by a later run. Pre-Phase-A bundles have
none, and the audit does not require it. A run judged before Phase A
therefore cannot serve as a `--trunk-from` / `TRUNK_FROM` trunk until its
stage directories are re-judged — by `generate_stage_artifacts`, or by
`python -m environments.shared.scripts.backfill_gate_verdict <stage_dir>`,
which re-derives a `stance_quality/v1` or `reward_and_length/v1` verdict
from the evidence the directory already holds (`judged_by = "backfill"`),
refuses a recorded block that names a gate kind or key retired with the hunt
nodes (`task_success/v1`, `min_success_rate` and the rest; decision D-D59),
re-derives a `stance_quality/v2` verdict only
from a `stance_gate_report.json` the judge admits for the stage's gate,
measurement definition, handoff pair and recorded task and can re-derive
from its own rows (decision D-D23), re-derives a `stance_quality_heading/v1`
verdict only from a `stance_heading_report.json` the judge admits the same
way and for its protocol, six-heading grid and certification seeds (decision
D-D32; a directory holding only a v2 report is refused), re-derives a
`command_tracking/v1` verdict only from a `command_tracking_report.json` the
judge admits for the stage's gate, handoff pair, handoff choice, task and
command (decision D-D61) and for the current panel protocol
(`command_panel/v2` since decision D-D64, which refuses a v1 report), refuses
to invent one from missing evidence, and cannot backfill
`recovery_quality/v1` (re-judge that through the notebook chain, which rolls
the panel). It also refuses, `--force` or not and before it writes anything,
a stage directory whose run holds a `complete` bundle, which a new or
rewritten verdict would stop verifying. The verdict also records the gate it
was judged under —
`gate` (kind, schema version and the thresholds the kind consumes) and
`gate_sha256`, its digest (decision D-A22) — which reuse rule 7 compares with
the reusing run's gate; a verdict written before D-A22 lacks both and is not
a trunk until re-backfilled with `--force` (`--gate current` after a threshold
edit, for a `reward_and_length/v1` directory) — never, in a run whose bundle
is `complete`, whose verdicts stay as published (decision D-B8 as amended
2026-11-24) — and a copied verdict whose
`gate_sha256` is not the digest of its `gate` block fails the audit.

`ancestors/<stage_id>/ancestor.json` records `{schema, stage_id, stage_key,
parent_run_id, source_run_dir, source_stage_dir, handoff {name, model_path,
model_sha256, normalization_path, normalization_sha256}, task_sha256,
judged_by, reused_at}`. The reader
(`result_bundle.load_ancestor_records`) fails closed on an undeclared stage
id, a missing file, a verdict whose hashes disagree with the record's, or a
task fingerprint that disagrees between `ancestor.json`, `gate_verdict.json`
and `stage_config.json`. A stage that warm-started from a reused ancestor
records `parent_run_id` in its `stage_config.json` run block, and the audit
binds that lineage to the record's checkpoint hash. A same-stage resume keeps
the load lineage its record holds (the edge keys of a node that entered from
its parent, or a root's first load) and adds `resume_load_path` /
`resume_checkpoint_sha256` (`config.RESUME_LINEAGE_KEYS`) for the periodic
checkpoint it continued from, so a resumed-then-judged node still chains by
digest; a root trained from scratch records its first resume's load under the
lineage keys. Every same-stage resume into a directory whose
`stage_config.json` is readable keeps the record's identity: the top-level
`git_commit` and `library_version` and the run block's `timesteps` and
`label` stay the first session's, and each resume session that reached
training (its model loaded) appends its commit, library version, timesteps,
continued-from checkpoint and UTC time to `run.resume_sessions`
(`config.RESUME_SESSIONS_KEY`, `config.record_resume_session`; before
2026-11-24 a resume rewrote all four with its own, the remaining budget as
`timesteps`, and a root's lineage with its own load). A resume whose
`hyperparameters_sha256`, or whose non-empty label, differs from the
record's is refused before anything is written
(`config.refuse_resume_recipe_change`), as are a checkpoint trained on
another task and a load that cannot continue the run (a SAC load, or a
curated PPO checkpoint) into a stage directory that already holds the
stage's training records. The run block also always records
`hyperparameters_sha256` and, when set, a `label`; `train_base.train` (the
CLI `train` subcommand and the notebook's `train_stage`) and the CLI
`curriculum` (each node it trains) add `duration_seconds` at the stage's final
save (decision D-A15): the training time of the sessions that reached a final
save in this stage directory. A same-stage resume into the same directory adds
its session; a session stopped before its final save records nothing, so an
interrupted-then-resumed notebook node reports the resumed session only, and a
CLI resume into a fresh directory records its own session. A curriculum node
never resumes, and a Ctrl-C saves it, so it records the one session that
trained it.

A stage directory written by `environments/shared/scripts/widen_checkpoint.py`
(BEHAVIOR_RECIPES_PLAN §4.6 "Widening instead of retraining"; decisions D-C8,
D-C9) is a ROOT that never trained: its run block carries the parent's `seed`
/ `n_envs` / `timesteps` (and `duration_seconds` when the parent stage
directory recorded one), `hyperparameters_sha256`, an optional `label`, and
the eight `config.WIDEN_LINEAGE_KEYS` —
`widened_from_path`, `widened_from_checkpoint_sha256`,
`widened_from_normalization_sha256`, `widened_from_task_sha256`,
`widened_from_policy_interface_sha256`,
`widened_from_policy_interface_revision`, `widened_from_run_id` (the parent
run's provenance `run_id`, else its directory name; in the explicit
`--model` / `--vecnorm` form the `--parent-run-id` given, else `null`) and
`widened_by` (tool version and commit) — and NONE of the
`LOAD_LINEAGE_KEYS`. The chain loop JUDGES such a root, so its stage result
carries the 30-episode final evaluation and the selected-checkpoint panel but
no `evaluations.npz` curve: `summary.json` records its `best_eval_reward` /
`best_eval_std` / `best_eval_step` as `null`, and the `provenance.deliverables`
record copies `widened_from_run_id` from the run block, which is the one case
`validate_result_summary` accepts a null canonical `best_eval_reward` for (the
curve is the parent run's; a root that trained here still needs its own).
The `LOAD_LINEAGE_KEYS` rule:
`ancestors._check_chain` refuses a root that entered under
`initialize_next_stage`, and the audit binds `parent_run_id` to an
`ancestors/` record, so the widen keys are provenance the audit ignores and
no reader consumes. The same parent digests sit in the archive's
`mesozoic_widen_lineage` attribute; both artifacts are re-stamped with the
current plant identity and the stage's current task fingerprint (the
parent's `mesozoic_task_lineage`, when present, is kept). `widen_report.json`
beside `stage_config.json` (`mesozoic.widen-report/v2`; since 2026-09-19 also
`schedule_members_restated`, a member-name to `repr` map of the parent's
cloudpickled schedule members the tool re-stated so the widened archive
carries no interpreter-specific bytecode, and `schedule_members_source`,
which names where the values came from: the parent's recorded
`hyperparameters` block, else the current stage config's algorithm block;
the map is empty when the parent stored them by reference) records the parent
paths and digests, the from/to observation widths, the padded tensors and
their columns, the inserted-at map (SAC critics), the optimizer members
padded, the max |padded column| (0.0), the max action delta on zero and on
probe commands over the seeded 200-step rollout, the inherited
`num_timesteps`, the commit and the library versions. It also records
`revision_gap` (how many interface revisions the widening crossed, `current
- parent`, or `null` for an `--allow-legacy-plant` parent that carries no
identity; the same value sits in the archive's `mesozoic_widen_lineage`)
and `max_revision_gap` (the `--max-revision-gap` bound it was admitted
under, default 1; decision D-C17 — a gap above 1 is only reachable across
fingerprint-only bumps, since every width and physics field is still
gated). The tool writes the
parent's own handoff name plus byte-identical `<stage_label>_final.*` copies
(so the notebook JUDGE branch fires) and never writes `gate_verdict.json`,
`provenance.json`, `gate_resolution.json`, `evaluations.npz`, `metrics.json`
or periodic checkpoints — the widened node is re-paneled before it certifies.

Completed bundles are immutable. Start a new run ID rather than replacing a
checkpoint, stage result, seed role, or other captured experiment setting. The
SB3 notebook enforces this before training: a session whose `RUN_ID` names a
run with a `complete` bundle and which would judge or train a node there is
refused at the end of the resolve cell, before anything is written, with the
remedy (a fresh `RUN_ID` whose `TRUNK_FROM` names that run); the chain loop
refuses, before the write, what the resolve cell cannot predict without the
reuse rule (a node held only as an `ancestors/` record that the trunk no
longer certifies, or a trunk's copy recorded into a run that lacks the
record), and the manual and resume cells refuse the same write
(`environments.shared.result_bundle.reentry`);
the zero-action cell leaves such a run's `zero_action_baseline.json` as sealed.
The resolve cell also records the trunk run the session resolved in the run's
`trunk_run.json` (`mesozoic.trunk-run/v1`: `{"schema", "trunk_from"}`, the
`TRUNK_FROM` value that reproduces it: `""`, a run id or an absolute run
directory; no timestamp, so a session on the same trunk writes nothing;
`environments.shared.result_bundle.trunk_record`). It never writes it into a
`complete` run, and never over the trunk the run's `ancestors/` records came
through, when the resolve cell ran before the chain loop that recorded them (a
chain loop run after a `TRUNK_FROM` change without the resolve cell can record
from another trunk; [KNOWN_ISSUES.md](KNOWN_ISSUES.md)): once the run holds
such a record the file is fixed, a session under another trunk is warned, and
the resume cell refuses to resume under it (cleanup ROW-4/6, decision 4 (a)).
It is a plain run file, declared and hashed by the next bundle write like any
other.
Repeating an identical export is a write-free no-op, so a transient Drive
failure cannot remove the completion marker from an already valid bundle. A
partial marker — a chain whose target has not certified yet — is rebuilt
over by the next node's save, which is how the notebook's per-node saves grow
one bundle.

A completion marker that no longer verifies is not the end of the run when
only the artifacts the export itself regenerates (`summary.json`,
`collected_results.csv`, `provenance.json`, `plant_identity.json`) disagree
with it — an export interrupted mid-rewrite, say: the next export rebuilds
them in place, and only after the rebuilt bundle has passed every check does
it replace the marker. If a certified artifact — a checkpoint or its
VecNormalize sidecar, evaluation evidence, a stage config, a gate verdict, an
ancestor record, `trunk_run.json` (a fact about the run, which the export
does not regenerate) — changed after publication, the export refuses and
names the file; that bundle needs a new run ID.

## Source of truth

`summary.json` is the canonical public result. `collected_results.csv` is
generated from the same normalized stage values, with resolved
hyperparameters added for analysis. README and website tables are generated
from the summary.

Do not manually copy metrics among JSON, CSV, and documentation. The bundle
validator treats disagreements as conflicts.

## Provenance captured at run time

`provenance.json` is initialized before training and records:

- run ID, species and embodiment (two fields; provenance schema 2), algorithm,
  and backend;
- repository URL, exact commit, dirty state, and patch hash when dirty;
- training and evaluation seed roles, and the `certification_panel` role
  (decision D-B17): the start seed of the registered 40-seed panel block
  (`constants.PUBLICATION_SEED_START`, 3042) that `stance_panel_selected.csv`
  rows and a recovery stage's frozen `gate_resolution.json` are bound to at
  publication — row `i` must carry `panel_seed == role + i`, and the
  resolution's `decision_procedure.panel_seed_start` must equal the role; a
  recorded `stance_quality/v1` or `stance_quality/v2` PASS in a bundle whose
  provenance lacks the role is refused (a v2 pass is also re-derived from the
  panel's floor-truth rows, which must record the certified checkpoint's
  digest, the certified sidecar's when there is one, the task the stage's
  `stage_config.json` records, and this checkout's measurement version and
  definition, and must hold the column of every metric a declared criterion
  reads: a panel rolled before a later metric existed is refused by name
  under a gate that declares its key, and re-rolled), and so is a role
  naming any block other than the registered one (the rows are checked
  against the role, so the role itself is pinned).  A recorded
  `stance_quality_heading/v1` PASS (decision D-D32) is re-derived from
  `stance_heading_panel_selected.csv` alone under the same bindings plus the
  gate digest of the block the stage records: its seed block must start at
  the role, and its policy rows (and, where a statue-relative criterion is
  declared, its statue rows) must be the complete, unique grid of those seeds
  at the six headings, each heading's stamped statue reference what its
  statue rows reduce to.  A recorded `command_tracking/v1` PASS (decision
  D-D61) is re-derived from `command_tracking_panel_selected.csv` alone (one
  row per episode record, every row stamped with the profile, the
  measurement, the scored pair's digests, the task and gate digests, the
  panel's first seed, horizon and control step): the rows must record the
  certified pair, the task and gate digest of the block the stage records,
  the stage's horizon and control step, and be exactly the panel the stage's
  thresholds and command plan from the role, in order; a follow profile's
  reference rows must be the parent checkpoint the stage entered on, on the
  same seeds.  The stage directory also holds `command_tracking_report.json`
  and its text form, which the post-stage judge admits, and
  `models/handoff_choice.json`, the handoff chosen by the gated statistic on
  a selection panel from seed 2042 that shares no seed with the
  certification panel (the judge refuses a choice whose recorded selection
  plan does), which `select_handoff_checkpoint` honors.
  `command_tracking_trace.csv` (the step rows of the chart episodes, rolled
  again after the verdict, each stamped with the checkpoint digest it was
  rolled on) and
  `figures/command_tracking.png` and `figures/command_gate_progress.png`
  (drawn into a local staging directory and copied into `figures/` with
  `atomic_copy` on the notebook and CLI paths alike; a figure whose draw
  raises or is interrupted has its partial file removed before the copy,
  and a runtime killed mid-draw copies nothing, so a half-written png never
  lands) are report-only: publication neither binds nor reads them.
  The role is bound per evidence file, not through
  `evaluation_protocols` (its name deliberately lacks "evaluation"), and it
  may equal the publication seed;
- deterministic evaluation protocols and episode counts;
- Python, platform, and dependency versions;
- hardware and parallel environment count;
- plant identity;
- a `sessions` record — one entry per process that touched the run: the
  creating session's `started_at`, then a `resumed_at` entry for every later
  session, carrying an `environment_drift` map when that session's
  environment differed from the original capture.

The bundle writer adds the **finalization** fields (result schema v4 onward):

- every stage's selected checkpoint hash, matching SB3 VecNormalize path/hash,
  and resolved-config hashes (`selected_checkpoints`, `config_hash`);
- `deliverables` — `{stage_key: {model_path, model_hash, normalization_hash,
  gate_kind, certified, replication: {count, runs: [{run_id, training_seed}]},
  certification_seeds, provisional}}` for every deliverable present in the
  run, plus the optional `hyperparameters_sha256` and `label` copied from
  the stage's `stage_config.json` run block (decision D-A21).
  `replication.runs` has exactly `count` entries with distinct run ids and
  distinct training seeds, THIS run first, then its replicates — the sibling
  runs under the same `LOG_BASE/<species>/<embodiment>/<algo>/` directory that certified
  the SAME recipe on another seed (decision D-B16; plan §4.5; a screen the SB3
  notebook trains with `RUN_KIND = "screen"` sits in `<algo>_screen/` and a
  quick test in `<algo>_quick_test/`, so neither is a sibling of a certification
  run, though the runs of one of those trees may count one another): a sibling
  counts when its stage directory for the node holds a passed, reusable
  `gate_verdict.json` with equal `task_sha256` and equal `gate_sha256`, its
  `stage_config.json` records the same plant identity and the same
  `hyperparameters_sha256` (recorded in its run block, else derived from its
  recorded algorithm, hyperparameters block and shaping keys — exactly what
  `hyperparameters_sha256` hashes — so a pre-D-A21 sibling is never skipped
  for the missing field alone), and its run block's `seed` differs.  A sibling whose
  verdict records no `gate_sha256` (judged before decision D-A22) is skipped
  until re-judged with `backfill_gate_verdict.py --force` (never, for a
  sibling whose bundle is `complete`), consistent with reuse rule 7.  A run
  that never counts (the maintainer's stance-landing
  decision of 2026-11-06: the nine r10 T. rex stance screen runs, which trained
  the landed stance recipe on branch commits, listed in
  `environments/shared/run_exclusions.py` by those commits, their run ids and
  their checkpoints, read from `provenance.json`, the stage directory's
  `stage_config.json` and its `gate_verdict.json`; a run whose stage loaded,
  resumed or widened one of their checkpoints is one too) is never counted as a
  replicate and counts no replicate of its own; it is never a reusable ancestor
  (reuse rule 9 in `environments/shared/ancestors.py`), a widening parent or a
  promotable result (the audit and the species catalog refuse its provenance,
  and a result whose provenance records one of them as an ancestor).
  `environments.shared.replication.discover_replicates_for_run`
  finds them and the SB3 notebook passes them as `save_result_bundle(replicates=...)`.
  The count is recorded at publication (decision D-B10): a replicate that finishes
  later is counted when the run's bundle is written again with the
  sibling present, which rebuilds a `partial` bundle and regenerates a
  `complete` bundle's derived artifacts (`provenance.json`, `summary.json`,
  the CSV, the manifest) when the replication record is the ONLY thing that
  changed — every certified artifact and every other result of a complete
  bundle stays immutable, and a re-run that would change anything else is
  still refused.  In the SB3 notebook a session that reuses every node
  writes no bundle, so a `partial` run is written again by its publish cell
  (`PUBLISH_RUN = True`, [Colab workflow](#colab-workflow)), a chain stopped
  at a failed node included; that cell refuses a `complete` run, so a
  complete bundle's replication record is regenerated only by a hand-typed
  `save_run_bundle(chain_results(), model_id=MODEL_ID)` call once the chain
  loop has re-entered the run, in a session run up to the chain loop rather
  than Run all, which ends in the notebook's auto-disconnect.  A key naming
  a deliverable the run holds outside the
  published chain is dropped (its node has no record in this bundle); a
  key naming no deliverable of the model is refused.
  `certification_seeds` is the stage's declared `[curriculum]` value
  (default 1; the T-Rex stance 2) and `provisional` is
  `count < certification_seeds` (decision D-B11) — the audit refuses a
  record whose first run is not this run, whose `certification_seeds` is
  not the stage config's, or whose label disagrees with its count, and the
  catalog re-derives `provisional` from the CURRENT config so a raised bar
  relabels a committed bundle without republishing it;
- `ancestors` — `{stage_key: {run_id, model_hash, normalization_hash,
  gate_kind, passed, task_sha256}}`, the projection of the on-disk
  `ancestors/` records;
- `target_deliverable` and `primary_deliverable` (stage keys, the primary
  `null` when nothing is certified), with `selected_model_path` /
  `model_hash` pointing at the primary.

These are finalization fields, not identity: re-running the notebook's setup
cell under the same run ID with a different `BEHAVIOR` is not a rejected run.

Dirty source trees remain auditable as partial or failed runs, but cannot be
promoted. Commit the intended source state and start a new run ID.

Missing historical values are never guessed. Legacy Drive directories remain
readable but are labelled `legacy-unverified` or `legacy-conflict`.

## Colab workflow

The SB3 notebook calls the shared bundle functions. A new run ID starts an
independent experiment. SB3's chain runs in one ordered notebook session
because its stage objects are kept in memory; the bundle is saved after every
node the chain loop trains or judges, growing from `partial` to `complete`
when the target certifies.

A node the chain loop reuses from the run's own records writes nothing, so a
session that reuses every node writes no bundle. The notebook's publish cell,
after the chain loop (section 6, `PUBLISH_RUN = True`; the maintainer's
decision of 2026-11-26), writes the training summary and the bundle from the
run's own records with the chain loop's two calls, then validates the bundle
and prints its status and what it certified. It is for a run cut off after
its last verdict but before its bundle (a stance node's probes run last and
are report only, so a runtime lost during them leaves the verdict, the stage
summary, figures and replays, and no `summary.json`, `collected_results.csv`
or `artifact_manifest.json`), and for a `partial` run re-entered to count a
replicate that certified later. A `partial` run usually holds a failed node,
on which the chain loop raises, so the cell is run without it: every cell
above section 5, then the publish cell, with `BEHAVIOR` and the other knobs
as the run's session had them, and last the auto-disconnect cell (section
9), which in Colab flushes Drive's buffered writes before it releases the
runtime (with `AUTO_DISCONNECT = False`, `drive.flush_and_unmount()` does
the flush by hand).

It trains, judges and probes nothing and changes no stage file. It walks the
chain as the chain loop does: root first, an ancestor's `ancestors/<id>/`
record standing for it (the target never, D-A18), and stopping after the
first failed verdict, where the chain loop halted after its write. Every
other node it meets on the chain must hold a `gate_verdict.json` in its stage
directory (found as the reuse rule finds one, `stage_dir_candidates`), whose
`stage_result` re-enters the bundle, as the chain loop's reuse of a node of
this run does. Before it writes anything it refuses a `complete` run (it is
published and immutable; the cell's message says how a late replicate is
counted into one), a `BEHAVIOR` whose target differs from the
`target_deliverable` the run's `provenance.json` records (it would retarget
the run, and a shallower target seals it `complete` without the nodes below),
a run that holds no node of the model, neither a stage directory nor an
`ancestors/` record (usually `RUN_ID = ""` in a fresh runtime, where the
storage cell opens a new run), a node on the chain that is neither judged nor
recorded (the chain loop judges
or trains it first), a node judged in the run that the bundle would leave
out (off the chain, or below the failed node), and a node whose verdict
passed but whose stage directory no longer hands on the pair the verdict
hashes (`select_handoff_checkpoint`, as the chain loop's reuse checks it; on
the command-driven walk the pair its handoff choice,
`models/handoff_choice.json`, names, decision D-D61). The walk node
re-enters from its `command_tracking/v1` verdict, and the writer re-derives
its pass from `command_tracking_panel_selected.csv`. A run that never wrote a
bundle records no target, so for it the cell trusts `BEHAVIOR`. A missing
report-only file, such as a probe the cut-off skipped, does not block it. A
stage result recorded under another root (a run copied off Drive) is found
again from its verdict's stage-directory-relative, hash-bound pair, and each
node's control step, which a verdict does not record, comes from a bare env of
its task, so the summary's sim time is right for the Compsognathus pair (0.02
s). A bundle write the writer refuses puts the summary back. In a new
runtime the storage cell appends its session to a `partial` run's
`provenance.json`, which the verify cell then refuses
([KNOWN_ISSUES.md](KNOWN_ISSUES.md)); the publish cell rebuilds the bundle
over it.

`publication_gate_passed` is the promotion decision recomputed from the fixed
publication evaluation and the frozen stage thresholds. A recorded pass must
be reproducible from the evidence; a recorded failure is bound to its
evidence but is not re-judged, and can never be certified. Training-time
`required_consecutive` settings describe chronological evaluation batches and
are not inferred from episodes in the publication evaluation.

Reusing a run in a fresh Colab session revalidates the run's identity:
species, embodiment, algorithm, backend, seed roles, evaluation protocols and seeds,
episode and parallel-environment counts, plant identity, and the run ID must
match the captured provenance exactly, and the initializer refuses a mismatch
instead of combining stages from different experiments. Environment fields —
Python, platform, and dependency versions, the Git commit/dirty state and
patch hash, and hardware — may drift between sessions: the resume is accepted,
the top-level provenance keeps the values the first session captured, and the
drift is recorded (with a logged warning) on that session's entry in the
per-session `sessions` record. For faithful reproduction, still prefer
checking out the captured `repository_commit` and reinstalling that revision
before continuing.

The Google Drive summary notebook audits canonical bundles first and uses its
older CSV/text/NPZ reconstruction only as a historical fallback; it reads
`summary.json` for both `canonical-valid` and `canonical-partial` bundles.
Before promotion, run strict validation, on a run of the certification tree
`<algo>/` only: a run's tree is its only record of its kind
(`environments/shared/run_tree.py`; nothing in `provenance.json` names it, as
decision D-D16 (b) left quick tests), a copy promoted into `results/` loses
it, and the Drive summary notebook's strict promotion cell refuses a run under
`<algo>_screen/` or `<algo>_quick_test/` before it validates:

```python
from environments.shared.result_bundle import validate_result_bundle

validate_result_bundle(RUN_DIR, require_complete=True)       # the target and every present deliverable certified
validate_result_bundle(RUN_DIR, require_publishable=True)    # at least one certified deliverable, summarised
```

The audit statuses are `canonical-valid` (summary + complete manifest),
`canonical-partial` (summary + partial manifest), `partial` / `failed` (no
summary), and `canonical-conflict` (any error — including a summary whose
`bundle_status` disagrees with the manifest, or a summary under a failed
manifest).

The manifest hashes files present when the exporter runs, and promotion rejects
undeclared files. If artifacts are added manually, regenerate the manifest
before promotion. The SB3 notebook flushes Google Drive before an automatic
runtime disconnect, and in Colab without Drive (`USE_GOOGLE_DRIVE = False`,
whose runs live only on the VM's disk) it never disconnects automatically
(`notebook_runtime.disconnect_runtime`): the release reads the storage
cell's `RUNS_ON_DRIVE`, which stays False for the runtime once a run went to
the VM's disk, never the knob. The report tools refuse an output directory
inside a run (`result_bundle.refuse_output_inside_run`:
`stance_gate_report.py --out-dir`, `episode_trace_report.py --out-dir`,
`warm_start_noise_scan.py --out-dir`,
`harnesses/recovery_offdist_panel.py --out`). `widen_checkpoint.py` refuses a
target in a run whose bundle is `complete`, and
`harnesses/freeze_recovery_gate.py` refuses such a run and a stage directory
that holds `gate_verdict.json`, `--replace` or not.

## The cross-run table and offline judgements

Each run's records stay in its own folder. The Google Drive summary notebook
reads every model's runs into one table, one row per stage of each training
run (`source` `training`, with the stage's gate verdict in `stage_passed`) and
per trial of each legacy sweep (`sweep` or `sweep-analysis`), and its section 9
saves the table on Drive as `logs/runs_summary.csv`, each model's rows as
`logs/<species>/<embodiment>/runs_summary.csv` and each model's legacy sweep
rows as `logs/<species>/<embodiment>/sweeps/sweeps_summary.csv`. It reads
only the model directories the species manifest lists. The March 2026 sweeps
and every run written before decision D-D33 sit under the retired species ids
(`logs/<old id>/`), which it skips, so those sweeps give the table no rows and
no `sweeps_summary.csv` ([DRIVE_LAYOUT.md](DRIVE_LAYOUT.md#generations)).

Before saving, it adds the rows of the repository's offline-judgement ledger,
`results/offline_judgements.csv`, as `source` `offline`: gate certificates and
pilot judgements of a checkpoint made outside training, by someone who cannot
write to the run's Drive folder, and recorded by pull request instead
([results/README.md](../results/README.md#offline-judgements-offline_judgementscsv)
documents its columns). The ledger is read from the notebook's own checkout
(on Colab, `REPO_REF`, `main` by default), so a row added on `main` appears
the next time the notebook runs on Colab. A row records a judgement only: it
certifies no run and promotes nothing.

`environments.shared.reporting.runs_summary` holds what the notebook calls:
`load_offline_judgements` validates the ledger; `merge_runs_summary` appends
its rows after the notebook's (a `PASS` verdict reads `stage_passed` true, a
`FAIL` false and any other verdict empty, and a row's `run_path` is its
`run_dir`); and `write_runs_summary` writes the union of every row's columns,
with a missing cell empty: the identity columns first, then the ledger's own
when the file holds an offline row, then every other column sorted, then the
metric columns of `collected_results.csv`. Each file is written to a
temporary file beside it and moved over it with `os.replace`, as
`progress.json` is, so a runtime reclaimed mid-write leaves the previous
table, never half of one. Each time the table has rows, the notebook rewrites
the combined file and the files of the models in the table whole, and keeps no
dated copies; a model with offline rows but no folder on Drive gets its
`logs/<species>/<embodiment>/` folder, to hold its `runs_summary.csv`. It
writes no table when the logs directory does not exist (Drive not mounted; it
never creates that directory) or when the table has no rows, and then leaves
any earlier table as it was; a model no longer in the table also keeps its
earlier file. Without the ledger file it saves the table with no offline rows.
Whenever the logs directory exists, rows or none, the same cell also rebuilds
each model's `reports/` tree (below).

The table is a view, rebuilt from the run folders and the ledger each time
the notebook saves it, and nothing reads it back. It is not a promotion record:
a run is promoted only through its bundle, as above. Each model's `reports/`
tree beside it is the other derived view, one row per run and a page per run
([DRIVE_LAYOUT.md](DRIVE_LAYOUT.md#the-reports-tree)).

## Public result paths

The public algorithm remains `PPO` or `SAC`; backend identity is separate:

```text
results/<species>/<embodiment>/ppo/summary.json       # SB3 PPO
results/<species>/<embodiment>/sac/summary.json       # SB3 SAC
results/<species>/<embodiment>/jax_ppo/summary.json   # JAX/MJX PPO, recorded before D-D17
```

The validator still files a `jax-mjx` summary under `jax_<algorithm>/`, so a
JAX run recorded before D-D17 retired that backend (cleanup PR-B) is never
mislabeled as a different reinforcement-learning algorithm; nothing writes one
now, and `results/` holds only Stable-Baselines3 results (none at present:
decision D-D59 deleted the four historical summaries with the hunt nodes their
last stage trained; `results/README.md` names the last commit that holds them). The paths are
unchanged by schema v4; the summary carries the deliverables map. Decision
D-D33 added the `<embodiment>` level and made `species` and `embodiment` two
required fields of every summary (result schema 5; the CSV gains an
`embodiment` column). The readers accept schema 5 alone. A summary of
schema 3 or 4 names a species ID that no longer exists and is refused.
Schema 2, the integer-stage form of the four summaries committed until
D-D59 (`git show 6c9982d81c8a:results/<path>`), is refused too since
decision D-D63, by a message that names that commit as the tree that
still reads them (version 0.4.0.dev2). It names the commit, not the
version: a later tree that still declares 0.4.0.dev2 came after D-D59
removed the stage 3 they record, and refuses them. `results/` holds no
summary at present, and no writer has emitted schema 2 since schema 3
replaced it.
