# Drive layout

The tree the notebooks write on Google Drive, and what writes and reads each
part of it. This is a living reference (the maintainer's decision of
2026-11-26): change it in the same pull request as a writer or a reader.
[RESULT_BUNDLES.md](RESULT_BUNDLES.md) specifies what a bundle must hold and
how it is validated; this page says where things are.

`LOG_BASE` is the root. The SB3 notebook's storage cell
(`notebooks/sb3_training.ipynb`, section 3) sets it to
`/content/drive/MyDrive/mesozoic-labs/logs` when it runs in Colab with
`USE_GOOGLE_DRIVE = True`, and to the checkout's `logs/` otherwise: on
Colab with `USE_GOOGLE_DRIVE = False` that is the VM's own disk, which Colab
deletes with the runtime. The Drive summary notebook
(`notebooks/google_drive_summary.ipynb`) reads the same folder as
`LOGS_DIR`. "Read by people" below means that no code reads the file back.

```text
<LOG_BASE>/
├── runs_summary.csv                    every model's rows, from the Drive summary notebook
├── <species>/<embodiment>/             one directory per model of configs/species_manifest.toml
│   ├── runs_summary.csv                this model's rows
│   ├── zero_action_baselines/          <stamp>.json per model scored, <stamp>.txt tables
│   ├── reports/                        derived from the runs (run reports and index), never inside a run
│   ├── <algo>/<run_id>/                the certification tree (RUN_KIND = "certify")
│   ├── <algo>_screen/<run_id>/         screens, pilots, experiments (RUN_KIND = "screen")
│   ├── <algo>_quick_test/<run_id>/     pipeline checks (QUICK_TEST = True)
│   └── sweeps/sweeps_summary.csv       only for a legacy sweep found under the model
└── <old id>/                           everything written before decision D-D33: not read
```

`<algo>` is the lowercase `ALGORITHM`, `ppo` or `sac`
(`environments/shared/run_tree.py`, `run_tree_name`).

## The model level

| Item | Writer | Readers |
|---|---|---|
| `<LOG_BASE>/runs_summary.csv` | The Drive summary notebook, section 9 (`reporting.runs_summary.write_runs_summary`), replaced whole each time the table has rows | Read by people; nothing reads it back |
| `<species>/<embodiment>/runs_summary.csv` | The same cell, one file per model in the table (a model with offline rows but no folder gets its folder) | Read by people |
| `<species>/<embodiment>/sweeps/sweeps_summary.csv` | The same cell, only when the model has legacy sweep rows. The March 2026 sweeps sit under the retired ids (below), so they give it none | Read by people |
| `zero_action_baselines/<stamp>.json` | The SB3 notebook's zero-action cell (`environments/shared/scripts/zero_action_baseline.py`, `preflight`): one record per model scored (`BASELINE_MODELS`), written only when the baseline changed apart from its time; `<stamp>` is `YYYYmmdd_HHMMSS` | The same cell, which compares the newest record before it writes; people. Training reads the run's copy (run level) |
| `zero_action_baselines/<stamp>.txt` | The same cell: the table of every model scored, beside the trained model's records, written only when it changed | The same cell (the newest table); people |
| `reports/` | `environments/shared/scripts/run_report.py`, through `environments/shared/reporting/run_index.py` (next section). The SB3 notebook rebuilds a run's folder and the model's index after every `save_run_bundle` and in the verify cell (`write_run_report_quietly`, best-effort), and the Drive summary notebook's section 9 rebuilds the tree of every model (`write_model_reports`); a file whose text is unchanged is not rewritten | People; the SB3 storage cell prints the first lines of `INDEX.md`. The Drive summary's walker skips it |
| `<algo>/`, `<algo>_screen/`, `<algo>_quick_test/` | The storage cell creates the run directory in the tree `RUN_KIND` and `QUICK_TEST` select (`run_tree.run_tree_name`) | `TRUNK_FROM = "auto"` scans `<algo>/` only (`ancestors.select_trunk`); seed-replicate discovery scans a run's own siblings (`replication.discover_replicates`); the Drive summary walks the model directory and reads each run's tree into `run_kind` (`run_tree.run_kind_of`) |

The three run trees are the only record of a run's kind: nothing in
`provenance.json`, a stage record or a digest names it.

## The reports tree

`<LOG_BASE>/<species>/<embodiment>/reports/` sits beside the model's run
trees ("reports and index", the maintainer's decision of 2026-11-26). It is
derived from them, never written inside a run and never hashed or declared by
a manifest, so removing it loses nothing. `run_report.py` refuses an output
inside a run (`result_bundle.refuse_output_inside_run`) and writes only what
changed:

- It owns `report.md` and `index.json` in each run's report folder, and
  `INDEX.md` and `index.json` in `reports/`. A file whose new text equals the
  old is never rewritten, so an unchanged run costs Drive no write.
- A run's folder whose text changed is rebuilt whole: built in a hidden
  folder beside it, then renamed over it, so a build that fails keeps the old
  folder. Anything else the folder holds is copied into every rebuild, and
  the tool prints its name. Each index file is replaced atomically.
- Report-only output of your own, such as a probe rolled again, belongs in a
  folder of `reports/` that is no run's report folder, which the tool never
  touches: `reports/<tree>/<run_id>.probes/`, where a missing-probe warning's
  commands write.
- A report folder whose run is gone is left as it is and no longer indexed.

```text
reports/
├── INDEX.md                          one row per run of the model, grouped by state, newest first
├── index.json                        the same rows (schema "mesozoic.model-index/v1")
├── <algo>[_screen|_quick_test]/<run_id>/
│   ├── report.md                     the one page for a run
│   └── index.json                    every file of the run (schema "mesozoic.run-index/v1")
├── <algo>[_screen|_quick_test]/<run_id>.probes/   your own report-only output; never touched
└── <run name>/                       the folder of a run directly in the model directory
```

A run's folder mirrors the run's path under the model directory
(`run_index.report_dir_for`). A run directly in the model directory, a flat
`<algo>_<YYYYMMDD_HHMMSS>/` run from before 2026-03-31 or a command-line
`curriculum_<timestamp>/` run, has its folder at `reports/<run name>/` and no
tree: `INDEX.md` lists it under the tree `(model directory)`. The index finds
the runs of each tree and those directly in the model directory
(`run_index.discover_runs`); it skips `reports/`, `sweeps/`,
`zero_action_baselines/` and hidden directories. Each page names the run's
layout, the generation of its stage directories: `NN_<stage_id>` (from
2026-08-20), `stage<N>`, `bare <stage_id>` (a bare reserved id such as
`stance/`) or `none` (no stage directory).

- **`INDEX.md`** lists each run's id, tree, state, certified deliverables,
  whether they are provisional, replication count and handoff, with a link
  to its `report.md`. A run without a sealed bundle (no
  `artifact_manifest.json`, or one that cannot be read) certifies nothing
  here, whatever its `provenance.json` records: its row reads `none
  published`, with the nodes whose verdict passed. A run of `<algo>/` that
  `run_exclusions.excluded_run_reason` names (the nine r10 stance screens,
  and runs built on them) reads `never counts`, and a stranded run (below)
  reads `stranded`. A row reads less than a page and carries no warning
  count.
- **`report.md`** has a state line; for an unfinished run of a notebook
  tree that a notebook session can re-enter (neither stranded nor refused
  re-entry, below), the step that finishes it (`run_index.NEXT_STEPS`: the
  publish cell, the chain loop, the RESUME cell, or restoring the manifest;
  a judged run without a bundle that also holds a node trained and not
  judged gets that node's chain-loop or RESUME step, since the publish cell
  refuses the run until the node is judged); a table per node
  with its id, verdict, gate kind and handoff, or `reused from <trunk
  run>/<stage dir>`, linked to the trunk run's report once that report
  exists; the certificate and probe one-liners, read from the small `.txt`
  reports (`stance_heading_report.txt`, `stance_gate_report.txt`, and the
  walk's `command_tracking_report.txt` with each family's success count),
  never from the large JSON; the paths of its figures, training history and
  replays; and warnings. A warning names a file no role covers; the declared
  probes a stance stage lacks (`stage_artifacts.declared_stance_probes`),
  expected only on a stage whose verdict `generate_stage_artifacts` wrote (its
  `judged_by`), one warning per stage with a command per probe (a stage the
  command-line curriculum judged gets a note instead: its judge rolls no
  probe; any other stage expects none); the handoff checkpoint and
  normalization that a verdict or an ancestor record names and that no longer
  exist (an ancestor's are also looked for in the trunk run beside the run,
  as the reuse rule looks); and a record that cannot be read, with the step
  that repairs it.
- **A stranded run** is one whose recorded plant (its `provenance.json`'s,
  else a `plant_identity.json`) is not the model's current plant, as the
  committed `configs/plant_manifest.generated.json` gives it
  (`run_index.plant_drift`, comparing the fields
  `plant_contract.validate_recorded_identity` compares). None of its
  checkpoints loads on this checkout, and a notebook session refuses to
  re-enter it. Its page says which fields differ and names no next step
  (when its `artifact_manifest.json` cannot be read, one: restoring the
  manifest so the record reads again, `run_index.RECORD_STEP`, which
  re-enters nothing), its warnings name no command that re-judges a stage
  or rolls a probe again, its certified deliverables read as a record of
  what passed then, and its row says `stranded`. Every run written before
  decision D-D59 is stranded: all seven plants moved and every checkpoint of
  every model was stranded once (`configs/plant_versions.toml`, note 24).
  Where the plant contract cannot be imported (its package imports MuJoCo),
  the current plant is unknown: no run reads as stranded, and each page
  says the plant cannot be read.
- **A run refused re-entry** is one that is not stranded but whose
  `provenance.json` records another plant identity than the one a new run
  records (`run_index.reentry_drift`): a change to the plant's appearance or
  source files alone (its visual revision and digest, its source-closure
  digest) moves no field a checkpoint depends on, so its checkpoints still
  load and a later run can still reuse its passed nodes, but the storage
  cell records the whole identity and `result_bundle.initialize_result_bundle`
  refuses a run whose `provenance.json` records another. Its page names the
  fields and, as for a stranded run, no next step but restoring an
  unreadable manifest.
- **The run's `index.json`** lists every file with its path, stage, role and
  size. Its `sha256` is copied from `artifact_manifest.json`, with
  `hash_source` `"manifest"`, and never computed again; for a run without a
  manifest the size comes from the listing and `sha256` is `null`, with
  `hash_source` `"none"`. The roles (`run_index.ROLES`, assigned by
  `run_index.ARTIFACT_ROLES`):

| Role | Files |
|---|---|
| `bundle` | `collected_results.csv`, `summary.json`, `artifact_manifest.json` |
| `provenance` | `provenance.json`, `trunk_run.json` |
| `fingerprint` | `plant_identity.json` (run and stage), `task_fingerprint.json` |
| `config` | `stage_config.json`, and `gate_resolution.json`, frozen before the node trains |
| `verdict` | `gate_verdict.json` |
| `handoff` | `models/handoff_choice.json`, the pair a `command_tracking/v1` walk hands on |
| `certificate` | `stance_gate_report.*`, `stance_heading_report.*` and `command_tracking_report.*` |
| `evidence` | `evaluation_selected.csv` (a hunt's `task_success/v1` evidence too, in a run written before decision D-D59), `evaluation_final.csv`, the stance and command-tracking panel CSVs and `recovery_*.csv` |
| `ledger` | what training writes as it goes (the stage table's ledgers, `progress.json`, `hop_watch.json`), and `curriculum_results.csv` |
| `baseline` | `zero_action_baseline.json` |
| `checkpoint`, `normalization` | `models/`: the policies, and their `_vecnorm.pkl` and `_vecnormalize_<steps>_steps.pkl` statistics |
| `tensorboard`, `figure`, `training_history`, `replay` | `tensorboard/`, `figures/`, `figures/training_history/`, `replays/`, and the figures and replays of a stage root written before `replays/` existed |
| `trace` | `command_tracking_trace.csv`, the step rows `figures/command_tracking.png` is drawn from |
| `probe` | the stance probes |
| `summary_text` | `training_summary.txt`, `stage_summary.txt` |
| `ancestor_record` | everything under `ancestors/<stage_id>/` |
| `report_only` | sidecars no reader names: `handoff_in_training.json`, `widen_report.json`, and the retired `stage_result.json` and `k3_panel.*` files |
| `unclassified` | a name no writer writes, which the page names in a warning |

A run's state comes from its files. `run_index.py` defines each one, and
`INDEX.md` groups the runs in this order (`run_index.RUN_STATES`):

| State | The run holds |
|---|---|
| `complete` | An `artifact_manifest.json` recording `complete`: the target and every deliverable it records are certified |
| `partial` | A manifest recording `partial`: at least one deliverable it records is certified, and its target or another deliverable it records is not |
| `failed` | A manifest recording `failed`: no deliverable it records is certified, and the run has no `summary.json` |
| `judged_no_bundle` | No manifest, and a stage directory holding `gate_verdict.json` |
| `post_training_cut` | No manifest, a stage directory holding evaluation CSVs (`evaluation_selected.csv` or `evaluation_final.csv`), and no stage holding a verdict |
| `training_cut` | No manifest, a stage directory holding a checkpoint in `models/`, and no stage holding an evaluation CSV or a verdict |
| `bundle_unreadable` | An `artifact_manifest.json` that cannot be read or records another status, so the bundle's status is unknown |
| `empty` | Nothing trained: no stage directory holds a checkpoint. At most the records written before training, or the `ancestors/` records of reused nodes |

After these groups `INDEX.md` adds `unreadable` (`run_index.INDEX_STATES`),
which is no run state: a run whose records raised something unforeseen while
the index read them, its row showing the error.

The first three and `bundle_unreadable` come from
`result_bundle.read_bundle_status`; the others, for a run without a manifest,
from its most advanced stage directory. The states describe files, not
history: a run trained before verdicts existed (`stage<N>/` directories)
reads as `training_cut` or `post_training_cut`, and its page says so.

## The run level

A run directory is `<LOG_BASE>/<species>/<embodiment>/<tree>/<run_id>/`. A
node trained or judged in the run has a stage directory, and a node reused
from another run has an `ancestors/<stage_id>/` record. A run can hold both
for one node: when the run's own verdict of a node below the target failed or
no longer passes the reuse rule, the chain loop reuses the trunk's certified
node and records it beside the stage directory. The record then stands for
the node. The bundle records the node as reused, and the stage directory
stays, hashed by the manifest but not published
(`tests/test_publish_cell.py`,
`test_an_ancestor_record_stands_for_the_node_beside_a_verdict_of_the_runs_own`).
The run's report shows the node as reused, linked to the trunk run's report,
and names the stage directory as not published; it reads nothing else from
it. What the bundle writer refuses is a bundle that records one node both as
trained in the run and as reused (`reporting/bundles.py`, `result_schema.py`).

```text
<run_id>/
├── provenance.json
├── plant_identity.json
├── trunk_run.json
├── zero_action_baseline.json
├── summary.json                 only when at least one deliverable is certified
├── collected_results.csv
├── training_summary.txt
├── artifact_manifest.json       written last
├── ancestors/<stage_id>/        a node reused from the trunk run: records only
└── NN_<stage_id>/               a node trained or judged in this run
```

| Item | Writer | Readers |
|---|---|---|
| `provenance.json` | The storage cell when the run opens (`result_bundle.initialize_result_bundle`); a later session is checked against it and recorded in its `sessions` record; the bundle writer adds the finalization fields (`reporting.bundles.save_result_bundle`, `result_bundle.update_provenance`) | The storage cell (a pinned `TRUNK_FROM` must name the same model, algorithm and backend); trunk selection; the publish cell (the recorded `target_deliverable`); the audit (`result_bundle/audit.py`); the Drive summary; the run index (`reports/`), which also compares its `plant_identity` with the current plant; the report tools, which take the nearest directory holding it as the run and refuse an output there (`result_bundle.run_directory` and `refuse_output_inside_run`, or `plot_gate_progress.run_directory`) |
| `plant_identity.json` | The bundle writer, at every save | The audit (it must match `provenance.json`); the reuse rule, for a stage directory without its own (`ancestors.py`); the run index, for a run whose `provenance.json` records no plant |
| `trunk_run.json` | The resolve cell (`result_bundle.trunk_record.record_trunk_run`): the `TRUNK_FROM` value that reproduces the trunk; never written into a `complete` run, and fixed once the run holds an `ancestors/` record | The resolve cell (it warns of another trunk); the RESUME cell (`result_bundle.reentry.refuse_trunk_other_than_recorded`); the run index (`reports/`) |
| `zero_action_baseline.json` | The zero-action cell: a copy of the trained model's record, rewritten only when the baseline changed apart from its time, and never written into a `complete` run | `curriculum/baseline_watch.py` (the training-time warning), `progress.json`'s `zero_action_reward` (`curriculum/progress_report.py`) and the statue line of `figures/gate_progress.png` (`visualization.plot_gate_progress`) |
| `summary.json` | The bundle writer, whenever at least one deliverable is certified | The audit; the Drive summary; a promoted copy under `results/`, which the species catalog reads |
| `collected_results.csv` | The bundle writer, at every save | The audit (`result_bundle.compare_summary_to_csv`); the Drive summary |
| `training_summary.txt` | `reporting.text_summaries.write_training_summary`, from the chain loop after every node it trains or judges, from the manual node cell (section 7) and from the publish cell | Read by people |
| `artifact_manifest.json` | The bundle writer, last, once the bundle has passed validation: the status (`complete`, `partial` or `failed`) and the hash of every file under the run | `result_bundle.read_bundle_status`, which every refusal to write into a `complete` run reads (the resolve cell, the chain loop, the manual, resume, zero-action and publish cells, `backfill_gate_verdict.py`, `widen_checkpoint.py`, `harnesses/freeze_recovery_gate.py`); the bundle writer's next save; the audit; the Drive summary; the run index (`reports/`) |
| `ancestors/<stage_id>/` | `ancestors.record_ancestor` (the chain loop, `curriculum --trunk-from`): `ancestor.json` and copies of the ancestor stage's `gate_verdict.json`, `stage_config.json`, `task_fingerprint.json` and `plant_identity.json`, never a checkpoint | `result_bundle.load_ancestor_records` (the bundle writer and the audit); the reuse rule of a later run trunked from this one, which follows the record to the run that certified the node; the publish cell; `result_bundle/reentry.py`; the run index (`reports/`, `ancestor.json`), which links the node to the trunk run's report |

The bundle is saved after every node the chain loop or the manual node cell
trains or judges, and records `complete`, `partial` or `failed` by what is
certified so far (`result_schema.bundle_status_for`). The publish cell
(`PUBLISH_RUN = True`) writes `training_summary.txt` and the bundle for a run
that has none yet, such as one cut off during a stance node's probes.

Two Tyrannosaurus rex walk runs, one that reused stance from its trunk and
one that trained it:

```text
<run_id>/                        <run_id>/
├── ancestors/stance/            ├── 01_stance/
└── 03_locomotion/               └── 03_locomotion/
```

A run written before decision D-D59 removed the hunt nodes may end at a hunt
instead: `03_behavior/` or `04_behavior/` beside these (`stage3/` before
2026-08-20). The stage level below describes those directories.

## The stage level

A stage directory is `NN_<stage_id>/`, where `NN` is the node's position in
the model's stage manifest (`configs/<species>/<embodiment>/stages.toml`,
`stage_manifest.stage_dirname`), not its place in the chain. Every model's
chain ends at the walk (decision D-D59 removed the hunt nodes). The
two-node models (Velociraptor, Brachiosaurus, Dibothrosuchus, Allosaurus)
write `01_stance/` and `02_locomotion/`. The three-node models (T. rex and
both Compsognathus models) put `02_recovery/` second, which only the
`recover` chain walks, so their walk chain writes `01_stance/` and
`03_locomotion/`. On all seven the walk follows a live command (decision
D-D62): each episode draws one heading-and-speed command at reset
(`command_mode = "heading_and_speed"`, the stage's `[commands]` table), a
forward cruise or, on about 15% of episodes, an exactly zero command to
stand still. It is judged once, after training, by `command_tracking/v1`
with the `walk` profile (decision D-D61) on the handoff its gated statistic
chose, whose files the table below names. Readers find a stage under any
name it was ever written under (`stage_manifest.stage_dir_candidates`).

A run written before decision D-D59 can also hold a hunt stage, the
`behavior` node (legacy stage 3) that followed locomotion: `03_behavior/` on
the Velociraptor, Brachiosaurus and Dibothrosuchus, `04_behavior/` on the
T. rex and both Compsognathus models (`stage3/` before 2026-08-20). Its
walk was judged by `reward_and_length/v1`, without a command, and its hunt
by `reward_and_length/v1` with a `min_success_rate` floor or, on the T. rex,
by the `task_success/v1` gate (a lower confidence bound on the success
rate), whose evidence is the hunt's `evaluation_selected.csv`. Decision
D-D59 retired that gate kind and the floor with the nodes. No manifest
declares the node now: the run index still reads the directory as the
`behavior` stage, after the declared ones, and says no chain walks or
reuses it, and `backfill_gate_verdict.py` refuses a recorded block that
names a retired gate kind or key. Such a run is stranded (above) in any
case.

`<label>` below is the stage's file label (`stage_manifest.stage_label`):
`stage<N>` for a numbered stage, its id (`recovery`) otherwise. The groups
say what a file is for and are not the run index's roles (the role table
above): the index files `gate_resolution.json`, grouped here with the
recovery evidence, as `config`, since it is frozen before the node trains,
and `hop_watch.json`, a sidecar here, as `ledger`, since training writes it.

| Group | Files | Writer | Readers |
|---|---|---|---|
| Verdict | `gate_verdict.json` | `generate_stage_artifacts` (`reporting/stage_artifacts.py`); the CLI `curriculum` (`train_base.train_curriculum`), judged by its in-training manager or, on a `stance_quality/v2` or `stance_quality_heading/v1` stance and on the walk, by its post-training judge; `scripts/backfill_gate_verdict.py` | The reuse rule (`ancestors.find_certified_ancestor`); seed replication; the chain loop and the publish cell, whose bundle takes the node's result from it; the audit; the run index (`reports/`) |
| Evidence | `evaluation_selected.csv`, `evaluation_final.csv` | `reporting.csv_output.save_evaluation_episodes`, from `stage_artifacts.evaluate_stage_checkpoints` (the notebook's evaluation of the selected and final checkpoints). Until decision D-D59 a hunt's `task_success/v1` stage also had `evaluation_selected.csv` written by `train_base`'s post-training panel and by `generate_stage_artifacts` before the verdict, both removed with the gate; the runs that trained a hunt keep the file | The bundle writer and the audit, by fixed path (`result_bundle/evidence.py`) |
| Evidence (recovery) | `gate_resolution.json`, `recovery_episodes_<controller>.csv`, `recovery_shoves_<controller>.csv` | `harnesses/freeze_recovery_gate.py` before training; `recovery_evaluation.write_recovery_evidence` after | `gate_resolution.json`: the recovery gate (`curriculum/gate_resolver.py`) and the audit (`result_bundle/evidence.py`); the CSVs are read by people |
| Certificate | `stance_gate_report.json`, `stance_gate_report.txt`, `stance_panel_selected.csv` (stance gates v1 and v2); `stance_heading_report.json`, `stance_heading_report.txt`, `stance_heading_panel_selected.csv` (`stance_quality_heading/v1`) | `generate_stage_artifacts`, before the verdict, and on a `stance_quality/v2` or `stance_quality_heading/v1` stance the CLI `curriculum`'s post-training judge (`stage_artifacts._write_stance_gate_report`, `reporting/stance_report.py`, `curriculum/stance_heading_gate.py`) | The gate judge (`reporting.gates.evaluate_stage_gate`); `backfill_gate_verdict.py`; the audit, which re-derives a v2 or heading PASS from the panel CSV; the `.txt` files are read by people and by `reports/` |
| Certificate (walk) | `command_tracking_report.json`, `command_tracking_report.txt`, `command_tracking_panel_selected.csv` (`command_tracking/v1`, decision D-D61): the cruise and stand panel rolled on the chosen handoff | `generate_stage_artifacts`, before the verdict, and the CLI `curriculum`'s post-training judge (`stage_artifacts._write_command_tracking_report`, `reporting/command_tracking_report.py`) | The gate judge (`reporting.gates.evaluate_stage_gate`), which admits the report; `backfill_gate_verdict.py`; the bundle writer and the audit, which re-derive a PASS from the panel CSV (`result_bundle/evidence.py`); the `.txt` file is read by people and by `reports/` |
| Handoff choice | `models/handoff_choice.json` (`mesozoic.handoff-choice/v1`): the pair the gated statistic chose on a selection panel from seed 2042, among `robust_best_model`, `best_model` and the final pair | `stage_artifacts.ensure_command_handoff_choice` (`command_tracking_report.choose_command_handoff`, `checkpoints.write_handoff_choice`): at the end of training (`train_base.py`), and in `stage_artifacts.evaluate_stage_checkpoints` when the stage holds no valid choice under its current gate | `curriculum.checkpoints.select_handoff_checkpoint`, so every reader of the handoff (the replays, the judge, the next node, reuse, the publish cell, `widen_checkpoint.py`) follows the one choice; the command-tracking judge, which refuses a handoff that is not the choice |
| Configs | `stage_config.json` | `config.save_stage_config` when training starts (a same-stage resume saves it again, keeping the first session's record); `widen_checkpoint.py` for a widened root | The bundle writer (the config hash) and the audit; the reuse rule; seed replication; the RESUME cell; the Drive summary's older-run reader; the training history; the run index (`reports/`) |
| Fingerprints | `task_fingerprint.json`, `plant_identity.json` | `config.save_stage_config` | The reuse rule, and the ancestor record that copies them; the training history (the plant check) |
| Ledgers | `evaluations.npz` (SB3's `EvalCallback`), `diagnostics.npz` (`diagnostics.py`; a resume copies the earlier file aside first as `diagnostics.prior_<UTC>.npz`, never deleted), `gate_progress.npz` and `eval_episodes.csv` (`eval_diagnostics.py`), `train_episodes.csv` (`rollout_ledger.py`), `policy_probe.npz` (`policy_probe.py`, PPO), `metrics.json` (`train_base.py`, the post-training panels) | Training | Checkpoint selection (`curriculum/checkpoints.py`); the figures; the training history; `stage_artifacts.build_stage_results_from_eval_data`; the Drive summary's older-run reader (`evaluations.npz`) |
| Progress | `progress.json` | `curriculum/progress_report.py`, during training | The training history (its checkpoint record and last step); people, mid-run |
| Summary | `stage_summary.txt` | `reporting.text_summaries.write_stage_summary`, after the verdict | The Drive summary's older-run reader |
| Probes | `stance_gate_probe_filtered`, `_constant`, `_release`, `_impulse` and `stance_heading_probe`, each `.txt` and `.json`, for each `stance_probe_*` key the stage declares | `generate_stage_artifacts`, last of all (`_run_stance_probes`) | Report only: the heading-probe handoff warning reads `stance_heading_probe.json`, and the run's report warns of a declared probe that is missing on a stage whose verdict `generate_stage_artifacts` wrote (the command-line curriculum's judge rolls none) |
| Sidecars | `handoff_in_training.json`; `hop_watch.json` (only when the hop watch trips); `widen_report.json` (a widened root) | `handoff_in_training.json`: the certificate writers, beside the certificate (`stage_artifacts._write_handoff_sidecar`), so `generate_stage_artifacts` and the CLI `curriculum`'s post-training judge (a `stance_quality/v2` or `stance_quality_heading/v1` stance, and the walk); `curriculum/hop_watch.py`, during training; `widen_checkpoint.py` | Read by people |
| Trace | `command_tracking_trace.csv`: the step rows of the walk's chart episodes, rolled again after the verdict | `stage_artifacts.write_command_tracking_figures` (`reporting/command_tracking_trace.py`), from `generate_stage_artifacts` and the CLI `curriculum`'s post-training judge, which first removes the trace and figures an earlier judgement left (Retention, below) | `visualization.plot_command_tracking`; report only: nothing judges, binds or publishes it |
| Retired | `k3_panel.csv`, `k3_panel.json` and the `k3_panel.prior_<stamp>.*` copies a resume set aside (the in-run K3 noise panel, decision D-D48, retired by the maintainer's decision of 2026-11-26); `stage_result.json` (the JAX stage writer, until decision D-D17); a hunt's whole stage directory (decision D-D59, above) | No writer now; the runs that ran them keep them, such as the three `stance6_r10LTSB` screens | Nothing reads them but the run index |
| `models/` | `<label>_<steps>_steps.zip` and `<label>_vecnormalize_<steps>_steps.pkl` (periodic), `best_model`, `robust_best_model` and `<label>_final`, each `.zip` with its `_vecnorm.pkl`; on the walk also `handoff_choice.json` (above) | Training (`train_base.py`; retention below) | The handoff selection (`curriculum.checkpoints.select_handoff_checkpoint`: the walk's recorded choice, else `robust_best_model`, then `best_model`); the next node; the RESUME cell; the bundle (the selected pair); the training history |
| `tensorboard/` | TensorBoard event files | `tb_sync.py`, copied from local scratch during training | TensorBoard |
| `figures/` | `training_curves.png`, `locomotion_health.png`, `behavioral_metrics.png`, `foot_contacts.png`, `stance_diagnostics.png`, `gate_progress.png`; on the walk also `command_tracking.png` and `command_gate_progress.png` | `generate_stage_artifacts` (`visualization.py`; the walk's two through `stage_artifacts.write_command_tracking_figures`, after the verdict, which the CLI `curriculum`'s post-training judge also calls) | Report only |
| `figures/training_history/` | `01_outcomes.png` to `20_gait_ledger.png`, `training_history.html`, `training_history.json` | `generate_stage_artifacts` (`reporting/training_history.py`) | Report only |
| `replays/` | `<species>_<embodiment>_<algo>_<label>_selected.mp4` and `_final.mp4`; on a stance or recovery stage also a `_<view>.mp4` per camera view and a `_stance.csv` | `generate_stage_artifacts` (`evaluation.record_stage_video`) | `reporting.stage_layout.iter_replay_files` (the notebook's replay cell) |

`generate_stage_artifacts` writes in this order: the certificate (on the
walk, the command-tracking report, rolled on the pair the handoff choice
names), the verdict, `stage_summary.txt`, the figures (on the walk, the
trace and its two figures among them) and training history, the replays,
then the probes. Until decision D-D59 a hunt's `task_success/v1` evidence
came between the certificate and the verdict. The notebook writes
`training_summary.txt` and the bundle after it returns, so a runtime lost
during the probes leaves a stage with a verdict that no bundle records yet (a
run with no earlier bundle is then in the state `judged_no_bundle`).

## Generations

| Written | Where | Stage directories | Read today |
|---|---|---|---|
| Before 2026-03-31 | Flat runs `logs/<old id>/<algo>_<YYYYMMDD_HHMMSS>/`, and the Ray Tune sweeps of March 2026 in `logs/<old id>/sweeps/stage<N>_<algo>_<YYYYMMDD_HHMMSS>/` | `stage<N>/` | No |
| 2026-03-31 to 2026-10-11 | `logs/<old id>/<algo>/<run_id>/`, with quick tests in `logs/<old id>/<algo>_quick_test/<run_id>/` from 2026-09-25 | `stage<N>/` (a bare id for a semantic stage) until 2026-08-20, then `NN_<stage_id>/` | No |
| From 2026-10-11 | `logs/<species>/<embodiment>/<algo>/<run_id>/` and `<algo>_quick_test/<run_id>/`, with `<algo>_screen/<run_id>/` from 2026-11-24 | `NN_<stage_id>/` | Yes |

`<old id>` is a species id that decision D-D33 retired on 2026-10-11, such as
`trex`. Those trees stay on Drive as they are. Current code reads none of
them: the Drive summary lists them as skipped, `TRUNK_FROM = "auto"` and
replication scan only the current trees, a pinned `TRUNK_FROM` must record
this model's species and embodiment, and the plant contract refuses their
checkpoints. To read or rerun a pre-rename run, check out `f40c11d3`
(CHANGELOG, decision D-D33). The Drive summary's readers of the flat and
sweep layouts remain, and read such a folder only if it is placed under a
model directory.

The current trees are read, but every run in them written before decision
D-D59 is stranded: the plant contract refuses its checkpoints, the reuse
rule refuses its nodes, and its report and index row say so (the reports
tree, above). Its stage directories may include a hunt's `03_behavior/` or
`04_behavior/` (the stage level, above).

Screens and pilots trained before 2026-11-24 sit in `<algo>/`:
`environments/shared/run_exclusions.py` keeps the nine r10 stance screens
(and runs built on them) out of every count, and the walking pilots
(`pilot*_walk`) still read as certification runs.

## Other trees

- **Quick tests**, `<algo>_quick_test/` (`QUICK_TEST = True`, whatever
  `RUN_KIND` says): 50,000 steps a node, the same run layout. Never a trunk
  or a seed replicate of a real run; quick tests may count one another.
- **Screens and pilots**, `<algo>_screen/` (`RUN_KIND = "screen"`): a branch
  screen, a pilot or another experiment trained through the notebook, the
  same run layout. Never a trunk or a seed replicate of a certification run;
  screens may count one another, and a screen still reuses certified
  ancestors from `<algo>/`.
- **Command-line behavior pilots** (`environments.shared.train_behaviors`)
  write only into their `--output`, which has no default and must be new or
  empty. Nothing refuses a path under `logs/`; the operator guide's example
  is `MyDrive/mesozoic-labs/behavior_pilots/<name>/`
  ([TRAIN_DIRECTION_AND_TERRAIN.md](TRAIN_DIRECTION_AND_TERRAIN.md)).
- **Scratch.** The report tools write to the `--out-dir` or `--out` they are
  given and refuse one inside a run: `stance_gate_report.py`,
  `episode_trace_report.py`, `warm_start_noise_scan.py` and
  `harnesses/recovery_offdist_panel.py` through
  `result_bundle.refuse_output_inside_run`, and `plot_gate_progress.py` and
  `plot_training_history.py` through their own check
  (`plot_gate_progress.run_directory`). Their examples write under a
  `scratch/` directory or `/tmp/`; no code names a scratch tree on Drive.
- **The command line.** `python -m environments.shared.train ... train` writes
  one stage directory: its `--output-dir` (or `--log-dir`), by default
  `logs/<species>/<embodiment>/<NN_stage_id>_<timestamp>/` in the checkout.
  `... curriculum` writes a run directory, by default
  `logs/<species>/<embodiment>/curriculum_<timestamp>/`, holding
  `plant_identity.json`, `curriculum_results.csv`, `ancestors/` and its stage
  directories, but no `provenance.json` or bundle.

## Run ids

The storage cell names the run directory from `RUN_ID` and applies every
rule below before anything is written. The knob, the memo, the directory-name
rule and the re-entry of an existing id are decision D-D15's; the refusal of
an id of another tree came with the screen tree on 2026-11-24 (`run_tree.py`);
the characters of a new id, the refusal of another model's id and the banner
are the maintainer's decisions of 2026-11-26.

- `RUN_ID = ""` keeps the run the storage cell resolved last in this runtime,
  while it is in the tree this session selects, and otherwise mints a new id
  from the time, `YYYYmmdd_HHMMSS`.
- An id is a directory name: no path, no surrounding whitespace, not `.` or
  `..`.
- A new id uses only letters, digits, `_` and `-`. A run the tree already
  holds is re-entered whatever its characters.
- An existing id re-enters that run, to finish an interrupted node or a
  `partial` run, under the identity its `provenance.json` records (model,
  algorithm, seeds, plant). A `complete` run takes no new node: a new
  variant is a new id, trunked from it.
- One id names one run of a model, whatever its tree: an id that names no run
  in the selected tree but a run in another of the model's trees for
  `ALGORITHM` is refused (`run_tree.refuse_run_id_of_another_tree`).
- `SPECIES` and `EMBODIMENT` select the model, never the id: an id that names
  no run of this model in its trees for `ALGORITHM` but a run of another model
  in that model's trees is refused, naming every run of the id.
- An explicit id prints a banner naming the model, the algorithm, the run
  tree and its kind, whether the run is new or re-entered, and every other
  run of that id under `LOG_BASE`, of any model and either algorithm.

A run directory named on the command line (`--output-dir`,
`widen_checkpoint --to-stage-dir`) is held to none of these.

## The placement rule

The maintainer's decision of 2026-11-26. A stage directory's root holds only
files that some reader names by a fixed path, and the ledgers. Report-only
output goes in `figures/` or `replays/`, or outside the run in a folder of
the `reports/` tree that is no run's report folder
(`reports/<tree>/<run_id>.probes/`). A new file at a stage root needs a
reader that names it by a fixed path. The
rule moves nothing: files already at a stage root stay there, and no manifest
rule changes.

## Retention

No retention policy is set. Nothing deletes a run tree, on Drive or locally,
and the Drive summary notebook never moves or deletes a run's folder; outside
the runs, `run_report.py` replaces only what it builds under `reports/`: a
run's report folder whose text changed, with everything else the old folder
held copied into the new one, and the two index files. It also removes the
hidden temporaries a build lost with its runtime left there. Inside a run,
two deletions remove files a writer did not just write. One is checkpoint
pruning in a stage's `models/`:
`CheckpointRetentionCallback` keeps the newest `max_checkpoints` periodic
checkpoints (5 by default) and, with `checkpoint_archive_every_timesteps`,
one per interval (`curriculum/checkpoints.py`,
`prune_periodic_checkpoints`); `best_model`, `robust_best_model` and
`<label>_final` are never pruned
([TRAINING_ARTIFACT_PRESENTATION.md](TRAINING_ARTIFACT_PRESENTATION.md#onset-checkpoints-the-archive-rule)).
The other is on the walk (`command_tracking/v1`): each
judgement of the node first removes the `command_tracking_trace.csv` and the
two command figures (`figures/command_tracking.png` and
`figures/command_gate_progress.png`) an earlier judgement left
(`stage_artifacts.remove_stale_command_tracking_outputs`, from
`generate_stage_artifacts` before its figure block, whether or not graphs
are drawn, and from `stage_artifacts.write_command_tracking_figures` before
it draws), so a walk re-judged with graphs off holds none of the three until
it is judged again with them on. Otherwise a writer only replaces the files
it writes and removes what it left behind itself: the `.tmp` leftovers of an
interrupted JSON write, which the bundle writer clears before it hashes
(`result_bundle/manifest.py`), the temporaries a TensorBoard copy stranded
in `tensorboard/` (`tb_sync.py`), and the partial output of a widening that
failed (`widen_checkpoint.py`).
