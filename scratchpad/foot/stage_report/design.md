# Stage report card: implementation design

The maintainer's request (2026-10-17): "as part of the training artifacts, a summary view of the information after
each behavior trained instead of having to recreate it". This document designs it from main `2e734e29` (#627's
merge, read-only checkout `S/fix/main_ro`) and from the telemetry branch `claude/stance-balance-telemetry`
(read-only checkout `S/fix/balance_telemetry`, head `71df0723` "Prove in the suite that the evaluation-side
telemetry changes no training" when it was read; the branch is being rewritten, so commits are named below by
subject, not by hash). `S` = `/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad`.
Paths are relative to the repository root unless absolute. Line numbers are the branch's unless marked "main".

The build is **stacked on the telemetry branch, with a soft dependency**: every branch-only name is imported inside
a `try` with main's behaviour as the fallback (section 13), so the card can be rebased onto main unchanged if the
telemetry PR stalls. The reading code accepts files that lack every branch field (all runs written before the
branch lands) and files from main's #620/#622/#626 era.

This revision follows an adversarial review (section 19). The largest change: **the replay code is not touched**.
The foot load map comes from one dedicated, unrendered episode on its own env (section 4), not from the rendered
replay's recorder.

---

## 0. Decisions in one page

| # | What | Where (new code) | Written | When |
|---|---|---|---|---|
| a | `stage_report.pdf`: a two-page A4 portrait report per stage, which Google Drive's preview displays | `reporting/stage_report.py` (collect, JSON, publish), `reporting/stage_report_render.py` (draw), `reporting/stage_report_bars.py` (bars per gate kind) | `<stage_dir>/stage_report.pdf` | After the gate verdict, the stage summary and the staged figures and replays; before the stance probes and before the run's bundle is sealed |
| b | `stage_report.json` (`mesozoic.stage-report/v1`): the same numbers, plus a flat `ledger` block for the results ledger | same | `<stage_dir>/stage_report.json` | same |
| c | `run_report.pdf`: one page with one line per node | `reporting/stage_report.py:write_run_report` | `<run_dir>/run_report.pdf` | Before every `save_run_bundle` the chain loop or the manual cell makes, beside `training_summary.txt`; so the last one before sealing is the run-end report |
| d | Inline display in the notebook | `notebook_runtime.display_stage_report`, `display_run_report` | nothing (PNG bytes in memory) | Chain loop and manual cell, after `display_stage_videos`. Also a new report-cards cell in section 8, before "Verify the result bundle" |
| e | Foot-load episode: one deterministic, **unrendered** episode of the handoff pair at the replay seed, on its own env, with the floor-truth recorder, inside `preserve_global_rng()` | `gait/foot_load.py` (outline, reduction), `evaluation.roll_foot_load_episode` (new function; `record_stage_video` unchanged) | `<stage_dir>/stage_report_foot_load.json` (`mesozoic.foot-load/v1`) | Step 5b, when the caller allows simulation: `generate_stage_artifacts(record_videos=True)` (the notebook), the CLI by default (not on an interrupted node), the script with `--rollout` |
| f | Backfill of a card a reclaim cost: a node of THIS run that holds a verdict but no card | `reporting.stage_report.ensure_stage_reports`, called by the notebook helper `write_run_report()` | the card files | Only immediately before a `save_run_bundle` call, so the save declares them; never into a complete bundle, never into another run's directory |
| g | Regenerate any stage's report, old runs included, outside the run | `scripts/stage_report.py STAGE_DIR [--out DIR] [--in-place] [--rollout] [--run-report RUN_DIR]` | `--out`, default `./stage_reports/<run_id>/<stage_dir_name>/`; any resolved output inside a run directory is refused, the default included | on demand |
| h | Bundle and audit | no manifest code change; `result_bundle/audit.py` gets optional consistency **warnings** | – | – |

Report-only. No change to training, any evaluation result, the gate verdict, checkpoint selection, the RNG streams or
any digest. No config key, no new package (matplotlib's `PdfPages` only). **`record_stage_video`,
`_attach_pad_recorder`, `_record_stage_replays` and the replays' stance CSVs are unchanged**, so no existing replay
attaches a recorder it did not attach before. The only extra simulation is (e): one episode per stage, after the
verdict, on a separate env that nothing else steps, RNG-neutral (rule 1.3), whose failure costs the map only.

Prototype measurements, from real run files (section 11): **73 KB and 2.0 s** for a two-page card from an 18-row
`gate_progress.npz` with a six-heading certificate. **106 KB and 1.6 s** when the same series are scaled to a
12M-step run (234 evaluation rows). The first matplotlib import costs a further 0.5-1.6 s. Reading the per-actuator
std from a checkpoint zip takes 6 ms and leaves the numpy and torch RNG states unchanged. The review rasterised the
prototype at 100 dpi (`S/foot/stage_report/review/p1_100dpi-1.png`): the heading table's failing-criteria cell runs
off the page and 6-7 pt text is barely legible inline; sections 5 and 8 add the fixes.

---

## 1. Ground rules

1. **Report-only.** The report reads files and writes its own files. Nothing reads them back to make a decision:
   not the gate, not reuse, not the bundle writer, not the chain loop. `generate_stage_artifacts` returns the same
   `stage_results` object with the same keys and values. The report never adds a key to it, because `stage_results`
   feeds `collected_results.csv`, `summary.json` and the verdict's `stage_result` projection.
2. **A failure costs the report only.** Each data block, each panel and each file has its own
   `try`/`except Exception`. A failure logs one warning and records itself in the JSON's `errors` list. The entry
   points (`write_stage_report`, `write_run_report`, `ensure_stage_reports`, `roll_foot_load_episode`'s caller)
   never raise an `Exception`. That includes an exception from the floor-truth recorder's substep hook, which runs
   unguarded inside `mj_step`'s loop (base_env.py:1302-1305; recorder.py:512-554 guards only the balance capture):
   on the dedicated episode's env it ends that episode and costs the map, nothing else. `KeyboardInterrupt`
   propagates, as everywhere else; even then no partial file is published (rule 4).
3. **No RNG on any training or evaluation path.** The report reads only, except episode (e), which runs inside
   `preserve_global_rng()` (new `environments/shared/rng_guard.py`). It snapshots and restores Python's `random`,
   `np.random`, torch's CPU generator and, **only when `torch.cuda.is_initialized()`**, every CUDA generator
   (`get_rng_state_all` would otherwise initialise CUDA itself). It is needed twice over: building the policy in
   `load_sb3_model` draws from torch's global generator before the state dict is loaded, and SB3's `load()` runs
   `_setup_model` -> `set_random_seed(saved seed)` for any checkpoint that saved a seed (evaluation.py:673-678),
   which reseeds all three global generators. The episode loads with SB3's default device, as the replay does
   (stage_artifacts.py:2171 passes none), so it rolls the same actions as the video on a GPU runtime too. The env is
   seeded only by `reset(seed=replay_seed(seed, stage))` on its own instance. Training is not exposed either way
   (`make_env`'s `_init` reseeds every global generator, train_base.py:200-201), but the stance probes, which run after
   the card, load policies and would see a leaked draw; the guard and a test (section 15) rule that out.
4. **Atomic, local-first writes, private until complete.** `staged_artifacts` publishes its staging tree in a
   `finally` (stage_layout.py:154-168), so a `KeyboardInterrupt` during rendering would publish whatever the staging
   tree holds. The card is therefore rendered into a **private** temporary directory outside the staging tree; the
   PDF is moved into the staging tree only after `PdfPages.close()` has returned, and the JSON only after
   `json.dumps(..., allow_nan=False)` has succeeded. Publication is `publish_tree`'s `file_io.atomic_copy` per file
   (file_io.py:45-66). If the runtime dies mid-copy it leaves `.<name>.<random>.tmp`, which
   `result_bundle.manifest._discard_stale_temporaries` deletes before hashing (manifest.py:39, 50-65).
5. **Never into a sealed run.** Every writer takes the run directory from its caller (the notebook's `RUN_DIR`, the
   CLI's output root) and also walks up from the stage directory to every ancestor holding `artifact_manifest.json`.
   For each it calls `result_bundle.read_bundle_status`. If one reads `"complete"`, or raises because the manifest is
   unreadable (manifest.py:68-90), the writer skips with one log line that names `scripts/stage_report.py` for
   regenerating outside the run. The notebook already refuses every judge or train into a complete run before it
   gets this far (reentry.py:143-210; cell 20's `refuse_write_into_complete_run`). This is defence in depth for the
   CLI pointed into a notebook run, the backfill and the script.
6. **House rules from #626's review.** JSON is read through `file_io.read_json_object` (the certificate and
   `gate_verdict.json` included; the verdict's schema is checked by `read_gate_verdict`). Files are written with
   `file_io`'s atomic writers. Tests use binomial model IDs. The one new call-time read of
   `policy_loading._PERIODIC_CHECKPOINT_RE` (section 3.6) is added to `test_policy_loading.py`'s table of call-time
   readers (`CALL_TIME_READS`, test_policy_loading.py:668-682). `roll_foot_load_episode` reads `_ensure_sb3`
   inside its body, so that table's `evaluation.py` entry becomes `{"_ensure_sb3": 3, "_resolve_vecnorm_sidecar": 1}`.
7. **The card restates the record; it does not judge.** The verdict box shows `gate_verdict.json`'s `passed` and
   `failures` (or `stage_results`' on a path that wrote no verdict). The bar table is a read-only re-reading of the
   inputs the judge used. When the table and the recorded verdict disagree, the card says so and the recorded
   verdict stands (section 6.3).
8. **No replay code changes.** `evaluation.record_stage_video`, `_attach_pad_recorder`, `_merge_pad_columns` and
   `stage_artifacts._record_stage_replays` keep their bytes. `replays/` keeps holding only videos and stance CSVs
   (test_stage_layout.py:224-227 pins its exact listing). This keeps the telemetry design's rule 1.5 ("No model
   attaches a recorder it did not attach before", telemetry design.md:73-77) for every existing env.
9. **Nothing a complete bundle may legitimately change after sealing is shown.** `save_result_bundle` regenerates a
   complete bundle's `summary.json` when only its replication record changed (bundles.py:719-735). The cards
   therefore never show `replication`, `certification_seeds` or `provisional`; a sealed PDF cannot follow them.

---

## 2. Where the report is generated

### 2.1 The shared entry point: `generate_stage_artifacts`

`reporting/stage_artifacts.py:generate_stage_artifacts` is the one function the notebook calls for every node it
trains or judges (the only caller outside tests; the CLI does not call it). Its order today (branch
stage_artifacts.py:1892-2099; main 1777-1969):

1. `_write_stance_gate_report` (1956): the certificate (on the branch, followed by `_write_handoff_sidecar`);
2. `_write_task_success_evidence`;
3. `_apply_stage_gate` (1978): the verdict onto `stage_results` and `gate_verdict.json`;
4. `text_summaries.write_stage_summary` (1991): `stage_summary.txt`;
5. `with stage_layout.staged_artifacts(stage_dir)` (1999): the figures, then `_record_stage_replays` (the videos and
   the per-frame stance CSVs), published to the stage directory when the block exits;
6. `_run_stance_probes` (2090): the most expensive step, run dead last on purpose.

New step **5b**, between 5 and 6:

```python
    if (generate_graphs if stage_report is None else stage_report):
        _write_stage_report_card(            # never raises; see section 10
            species_cfg=species_cfg, stage_config=stage_config, stage=stage, algorithm=algorithm,
            stage_dir=stage_dir, seed=seed, stage_results=stage_results, context=report_context,
            allow_episode=record_videos,
        )
```

New keyword-only parameters: `stage_report: bool | None = None` and `report_context: str = "notebook"`.
`None` follows `generate_graphs`: the card is a rendered, best-effort artifact like the figures, and the existing
callers that pass `generate_graphs=False` (test_compsognathus_longipes_training.py:440-451, :571) keep their behaviour
and their runtime. The foot-load episode follows `record_videos`, the existing "simulate replays" switch, so a caller
that asked for no simulation gets none. The context only labels the card: `"notebook"`, `"manual"`, `"cli_train"`,
`"cli_curriculum"`, `"backfill"` or `"script"`.

Why step 5b:
- **After step 3.** The card states the verdict, and the verdict must not depend on the card. The JSON records the
  sha256 of the `gate_verdict.json` it summarised.
- **After step 5.** The files section lists what was actually published: a file that failed to publish is skipped
  by `publish_tree` (stage_layout.py:186-199), and the card checks each file exists before listing it. (The map no
  longer depends on the replays.)
- **Before step 6.** The probes cost about 300 episodes on the T. rex. A runtime lost during them should cost the
  probes, as it does today, and not the card as well. The card lists the probes the stage declares and names the
  files they write afterwards (`stance_*_probe.*`). It does not summarise them; that is deferred (section 18).
- **The window it leaves.** A reclaim between step 3 and 5b leaves a node with a verdict and no card. The next
  session reuses that node in place (cell 20's same-run REUSE branch `continue`s before any artifact step,
  sb3_training.ipynb line 1257), so without (f) it would never get one. Section 2.3's backfill closes it.

The ordering pins `test_the_certificate_runs_before_the_verdict_and_the_probes_after`
(test_reporting_stage_artifacts.py:1308) and `test_the_verdict_is_recorded_before_any_probe_runs`
(test_stance_gate_report.py:3163) gain
`source.index("staged_artifacts(") < source.index("_write_stage_report_card(") < source.index("_run_stance_probes(")`.

### 2.2 Every path that trains or judges a node

| Path | Who calls the report | Notes |
|---|---|---|
| Notebook chain, TRAIN branch | `generate_stage_artifacts` (cell 20) | default context |
| Notebook chain, JUDGE branch (a node the RESUME cell finished, or a root widened on the CLI) | `generate_stage_artifacts` (cell 20) | The header shows the resume and the widen lineage (section 3.1). The curves mark the resumed step, parsed from the resume checkpoint's name with `_PERIODIC_CHECKPOINT_RE`. The duration is the accumulated `run.duration_seconds` |
| Notebook chain, REUSE branch, same run | none at reuse time | The node's card from its first session is already in the stage directory. If a reclaim cost it, (f) writes it before the next bundle save (2.3) |
| Notebook chain, REUSE branch, cross-run | none, ever, into the source run | The card lives in the run that certified it; the run report reads it there (section 7). Nothing is written into another run's directory |
| Notebook manual cell (the walking pilots, D-D36) | `generate_stage_artifacts(..., report_context="manual")` (cell 22) | The banner reads "manual node: the verdict is recorded, never enforced; not a chain step". The card is written even when the manual cell's bundle save is refused (`ResultBundleError`), as the pilots' were; it is then as undeclared as the rest of that stage directory (2.3) |
| Notebook RESUME cell | none (`evaluate=False`; nothing is judged) | The JUDGE branch that follows writes the card |
| CLI `train` (`cli.py` `train` subcommand, `train_base.train`) | `train()`, after `_report_hpt_metrics` and the env closes (train_base.py:1742-1762) | New `train(..., write_stage_report: bool \| None = None)`. `None` follows `report_metrics`, which the notebook sets to `False` (sb3_training.ipynb line 731), so the notebook never gets a second card from `train()`. CLI flag `--no-stage-report`. `train()` writes no gate verdict, so the verdict box reads NOT JUDGED (section 6.3). A single CLI stage belongs to no run: `run.run_id` is null and no run report is written |
| CLI `curriculum` (`train_base.train_curriculum`) | per node, after `write_gate_verdict` and `_record_stage_result`, before `if interrupted: break` (train_base.py:2961-3028) | context `"cli_curriculum"`. An interrupted node (Ctrl-C) gets a card with NOT JUDGED and **no** foot-load episode, so the exit stays prompt. `run_report.pdf` in `base_dir` is rewritten after every node |
| `scripts/backfill_gate_verdict.py` | none | It rewrites `gate_verdict.json` only. An older card then summarises another verdict. The JSON's `gate_verdict_sha256` shows that; the audit warns (section 12) and the ledger prefers the verdict (section 6.5) |
| `scripts/stage_report.py` (new) | the script | Outside the run by default (section 2.4) |
| `train_behaviors.py` (direction and terrain behavior bundles) | none | Out of scope: a separate trainer with its own `mesozoic.behavior-bundle/v2` bundle, "separate from canonical gates" (train_behaviors.py:1-4); the telemetry design defers it too (§12). The script can render a card for any of its stage-shaped directories |

### 2.3 The run report, the backfill and the bundle in the notebook

Today the chain loop runs (cell 20): `generate_stage_artifacts` → `display_stage_videos` → the two inline plot calls
→ `write_training_summary(RUN_DIR, chain_results())` → `save_run_bundle(chain_results(), ...)` → enforce the verdict
(sb3_training.ipynb lines 1382-1407). The pin `test_gate_failure_writes_the_bundle_then_halts`
(test_sb3_notebook_pins.py:2573) fixes `write_training_summary < save_run_bundle < gate check`.

New order:

```text
generate_stage_artifacts(...)          # writes the stage card (5b)
display_stage_videos(stage_dir)
display_stage_report(stage_dir)        # new: PNG in memory, writes nothing
plot_training_curves(...); plot_diagnostics_graphs(...)   # unchanged
write_training_summary(RUN_DIR, chain_results())
write_run_report()                     # new notebook helper (cell 15): backfill (f), then run_report.pdf
save_run_bundle(chain_results(), model_id=MODEL_ID)       # hashes every file in RUN_DIR: the cards are declared
<enforce the verdict>
```

The manual cell gets the same two lines, inside its existing `try` around the bundle write (lines 1529-1535).
Because the cards are written before `save_run_bundle`, `build_artifact_manifest` hashes them like every other file
in the run (manifest.py:113-122). Nothing is written after sealing: the section 8 display cell and
`display_stage_report` draw in memory only. A pin test checks the file listing before and after, as
`test_the_curves_cell_writes_nothing_into_the_sealed_bundle` (test_sb3_notebook_pins.py:3795) does for the curves
cell.

**`write_run_report()`** (cell 15, never raises):
1. **Nodes.** One `RunReportNode` per entry of `MANIFEST.stages` that is in `CHAIN` or `NODE_RESULTS`, in manifest
   order. This run's own nodes come from `completed_stages` (it holds every node trained, judged or same-run reused
   in this session, with its stage directory). The current node is **not** yet in `NODE_HANDOFF` at this point (cell
   20 sets it at line 1409, after the bundle save at 1401) and a failed node never enters it, so `NODE_HANDOFF` is read
   only for cross-run ancestors (`run_id` not `None`, `stage_dir` in the source run). A chain node in neither is
   `not_reached`, with this run's `stage_dirname` directory. Fields: `stage_id`, `stage_dir`,
   `origin` (`"this_run"`, `"reused"` or `"not_reached"`), `source_run_id` (for `"reused"`), `manual` (not in `CHAIN`).
2. **Backfill (f).** `reporting.ensure_stage_reports(RUN_DIR, nodes, ...)`: for each `this_run` node whose stage
   directory is under `RUN_DIR`, holds `gate_verdict.json` and has no `stage_report.json`, it writes the card with
   context `"backfill"`, the foot-load episode included. It refuses (rule 1.5) when the bundle is complete. It runs
   only here, i.e. only immediately before a `save_run_bundle` call, so the save that follows declares what it
   wrote. A session that only reuses nodes never reaches a save and writes nothing: a card written then would be an
   undeclared file under the partial manifest, and section 8's "Verify the result bundle" cell
   (`validate_result_bundle(RUN_DIR, ...)`, line 1666; `reject_unlisted=True` by default, audit.py:657) would fail.
3. **The page.** `reporting.write_run_report(RUN_DIR, model_id=MODEL_ID, algorithm=ALGORITHM, seed=SEED,
   nodes=..., target=TARGET_NODE.id)`.

**Parity with `training_summary.txt`.** Both run-root files are rewritten just before each save. When the save is
refused (the manual cell's `ResultBundleError`, or a preflight failure in the chain loop), both now disagree with the
partial manifest until the next successful save, exactly as `training_summary.txt` alone does today; the refused
node's stage directory is undeclared in that case anyway. No new exposure.

### 2.4 The regenerate script

`python -m environments.shared.scripts.stage_report STAGE_DIR [--out DIR] [--in-place] [--rollout] [--run-report RUN_DIR]`

- By default it writes to `./stage_reports/<run_id>/<stage_dir_name>/` in the current directory.
- **Any resolved output directory inside a run directory is refused, the default included**, as the branch's
  `scripts/plot_gate_progress.py` does for its default (plot_gate_progress.py:45-58: the run directory is the nearest
  ancestor of `STAGE_DIR` holding `provenance.json`). A user who `cd`s into a run and runs the script with no `--out`
  would otherwise drop undeclared files into it, which fails `validate_result_bundle` and the drive-summary notebook's
  strict promotion validation (google_drive_summary.ipynb cell 22).
- `--in-place` writes into `STAGE_DIR` itself. It is refused when that run (or any ancestor) holds an
  `artifact_manifest.json` of **any** status: a file added to a published run is undeclared. The walking-pilot runs
  (provenance, no manifest) and CLI output directories are allowed.
- `--rollout` allows the foot-load episode. It needs SB3 and MuJoCo; without the flag the script reads files only,
  and the map falls back to the stance CSV (section 4.5).
- Old runs therefore get a card on demand, outside the run unless the run has never been bundled.

---

## 3. What the report reads

Every input is optional. A missing or unreadable input leaves its block `null`, its panel shows a one-line note
saying what is missing, and `errors` names it if the read raised. Paths are relative to the stage directory.

### 3.1 Identity and header

| Field | Source, in order of preference |
|---|---|
| model, display name | `species_cfg.model_id` (`ModelId`), `species_names.model_display_name`; script: `stage_config.json` `species`/`embodiment` |
| stage reference, id, display name | the `stage` argument, `stage_manifest.resolve_stage_key(model_id, stage).id`, `stage_display_name(stage, stage_config["name"])` |
| algorithm | the argument; `stage_config.json` `algorithm` |
| run id | `<run_dir>/provenance.json` `run_id` (read_json_object), else the run directory's name (as `progress_report._run_id`). Null for a CLI `train` stage (no run) |
| seed, n_envs, session timesteps | `stage_config.json` `run.seed`, `run.n_envs`, `run.timesteps` |
| commit, library version | `stage_config.json` `git_commit`, `library_version`. Each session re-saves `stage_config.json` (config.py:940-972), so these are **the last session's**; the header says "commit (last session)" |
| label, hyperparameters digest | `run.label`, `run.hyperparameters_sha256` |
| parent checkpoint, load mode | Only when `run.load_mode == "initialize_next_stage"`: `run.load_path`, `run.parent_checkpoint_sha256`, `run.parent_task_sha256`, `run.parent_run_id`. A root that was resumed records its own periodic checkpoint in the same keys with `load_mode = "resume_same_stage"` (config.py:961-971, the `else` branch); the card shows that as "resumed from its own checkpoint", never as a parent. Without either: "none: a root trained from scratch" |
| resumed from | `run.resume_load_path`, `run.resume_checkpoint_sha256` (a node that entered from its parent and was then resumed, config.py:964-969). Only the **last** resume is recorded; the header says "last resumed from" |
| widened from | `run.widened_from_*` (`config.WIDEN_LINEAGE_KEYS`, config.py:607-612) for a root widened on the CLI |
| budget | `stage_config.json` `curriculum.timesteps` (the declared budget), and `run.timesteps` (this session's) |
| trained steps | the final checkpoint's `num_timesteps` (`task_fingerprint.read_checkpoint_attribute(models/<label>_final.zip, "num_timesteps")`), else `stage_results["timesteps"]` |
| duration | `run.duration_seconds` (D-A15: the node's whole time over all sessions, config.py:458-510), else `stage_results["duration_seconds"]` |
| task, plant | `task_fingerprint.json` `task_sha256` (or `stage_config.json` `task_fingerprint`); `plant_identity.json` `plant_version`/digest |
| date | the report's own `generated_at` (UTC), and the verdict's `judged_at`. No file modification times: Drive does not keep them reliably |

### 3.2 Verdict

| Field | Source |
|---|---|
| passed, failures, gate kind and version, judged_by, judged_at, gate_sha256, checkpoint_sha256 | `result_bundle.read_gate_verdict(stage_dir)`. On a `GateVerdictError` (a malformed verdict), `verdict.source = "unreadable"` with the reason |
| without a verdict file | `stage_results["publication_gate_passed"]` and `["gate_failures"]` (`source: "stage_results"`); else NOT JUDGED |
| gate_verdict_sha256 | `result_bundle.hashing.sha256_file(gate_verdict.json)`, the function the manifest uses, so the audit can compare the two strings (section 12) |
| thresholds | the verdict's `gate.thresholds` (the gate it was judged under, D-A22), else `stage_config["curriculum_kwargs"]` |
| bars | per gate kind, section 6.3 |

### 3.3 Evidence per gate kind (read-only)

| Kind | File | Fields read |
|---|---|---|
| `stance_quality_heading/v1` | `stance_heading_report.json` | `schema`; `passed`; `failures`; `result.{n_clean_seeds, n_seeds, clean_seed_lcb, seed_failures, pooled_episodes}`; `thresholds`; `handoff.checkpoint_sha256`; `headings[*].{yaw_deg, result.{n_clean, n_episodes, clean_lcb, full_horizon_fraction, mean_reward, statue_mean_reward, criterion_failures, hop_or_fall_episodes, passed}, metrics.{reward_mean, reward_std, episode_length_mean, bilateral_support_duty}, statue.reference.{mean_reward, n_full_horizon, n_episodes}, episode_evidence[*].<criterion metric>}` |
| `stance_quality/v2` | `stance_gate_report.json` | `result` (StanceV2Result.as_dict), `thresholds`, `statue.reference`, `episode_evidence[*]`, `handoff` |
| `stance_quality/v1` | `stance_gate_report.json` | `passed`, `failures`, `metrics`, `thresholds`, as `render_stance_gate_report` prints them |
| `task_success/v1` | `evaluation_selected.csv`, through `reporting.gates.task_success_statistics(stage_dir)` | `best_model_success_count`, `best_model_n_episodes`, `best_model_success_lcb`, `best_model_reward`, `best_model_length` |
| `recovery_quality/v1` | `recovery_episodes_<controller>.csv`, `gate_resolution.json` | successes / episodes on the frozen panel. The decision procedure is not re-derived; the recorded failures are shown |
| `reward_and_length/v1` | `stage_results`, else `evaluation_selected.csv` and `metrics.json` | the same key precedence as `_reward_and_length_stage_gate` (gates.py:1560): `best_model_reward` → `best_eval_reward` → `mean_reward`, and so on |

A certificate or report is used only when its `handoff.checkpoint_sha256` equals the verdict's `checkpoint_sha256`.
Otherwise the card prints "the certificate on disk is for another checkpoint" and shows no bars from it.

### 3.4 Key numbers

| Number | Source and fallback |
|---|---|
| best in-training evaluation | `evaluations.npz`: the argmax of the per-evaluation mean, its std, step and episode count (as `build_stage_results_from_eval_data`) |
| final in-training evaluation | `evaluations.npz`, last row |
| selected-checkpoint panel (30 episodes) | `evaluation_selected.csv` (`reward`, `length`, `mean_forward_velocity`, `distance_traveled`, `task_success`, `checkpoint_sha256`), else `stage_results["best_model_*"]`, else `metrics.json` (CLI) |
| final-checkpoint panel | `evaluation_final.csv`, else `stage_results["mean_reward"]`, and so on |
| zero-action statue reward | in order: the certificate's 0° statue reference; the v2 report's `statue.reference.mean_reward`; `<run_dir>/zero_action_baseline.json`, only when its top-level `stage` resolves to this stage (`baseline_watch.read_zero_action_baseline`). Otherwise null, so a stance statue is never compared with a locomotion reward. Ratio = selected-panel mean / statue |
| clean count and lower bound | heading: `n_clean_seeds/n_seeds`, `clean_seed_lcb`. v2: `n_clean/n_episodes`, `clean_lcb`. task_success: `k/n`, `best_model_success_lcb`. Otherwise null. The bar beside it is `min_clean_stance_lcb` or `min_success_lcb`. The in-training screen (branch) is shown apart from it and labelled "screen, not a verdict" |
| falls | selected and final panels: a termination-reason column when the CSV has one, else episodes with `length < horizon` (horizon = `env_kwargs.max_episode_steps`, default 1000). Labelled "ended early" on kinds whose episodes can also end on success (task_success). Certificate: `sum(n_episodes - n_full_horizon)` over headings. Last in-training evaluation: the `termination_<cause>` counts in `gate_progress.npz` (main since #626). Training rollouts: `diagnostics.npz` `rollout_falls` (telemetry item 4, not yet built) |
| speed | the selected and final panels' `mean_forward_velocity` (mean ± std) and `distance_traveled` |
| handoff | `select_handoff_checkpoint(models/)` gives name, path and vecnorm. `sha256_file` gives both digests. The step and score come from `handoff_in_training.json` (branch, stance stages) when present. Otherwise the report computes them itself for any kind: `read_checkpoint_attribute(zip, "num_timesteps")` and the matching `evaluations.npz` row (mean, std ddof 0, episodes). It does **not** call the branch's private `_handoff_in_training` (stage_artifacts.py:357), so the branch's code is not changed and the card does not depend on it |
| robust_best / best / final steps | `read_checkpoint_attribute(<name>.zip, "num_timesteps")` for each. `progress.json` `checkpoints.{robust_best_model_timesteps, best_model_timesteps}` is used as a cross-check, and a disagreement is noted |
| kept periodic checkpoints | `progress.json` `checkpoints.periodic_timesteps` (telemetry item 8, not yet built), else a listing of `models/` by `_PERIODIC_CHECKPOINT_RE` |
| advisories | `progress.json` `warnings[*].{source, timesteps, message}` and `warnings_total`. progress.json keeps the newest 10, and the card says so |

### 3.5 Posture over training (`gate_progress.npz`)

Each panel is drawn from whichever of its keys are present. A missing key is not an error. Grouping uses the
branch's `visualization._gate_progress_groups` and `_OVER_BAR_LINES` when importable (pure functions of key names and
constants; visualization.py:812-829), else an equivalent local table pinned against them by a test.

| Panel | main keys (#620/#622/#626) | branch keys | absent on |
|---|---|---|---|
| signed CoP per foot | `<label>_sole_cop_fore_aft`, `<label>_foot_cop_fore_aft`; bars from `max_sole_cop_fore_aft`, `max_foot_cop_fore_aft` | – | Velociraptor, Brachiosaurus (no box pad) |
| toe and heel margins | `<label>_toe_margin_nm`, `<label>_heel_margin_nm` | – | every model but the T. rex. A box-soled model without a span shows the roll rails instead: `<label>_sole_tilt_deg`, `<label>_sole_corner_lift_m` |
| digit share | `<label>_digit_share` | – | soles without digits |
| over-bar fractions and falls | `max_*_over_bar_fraction`, with PostureDrift's 0.5 line; falls per evaluation from `termination_<cause>` (every cause except `truncated`) on a twin axis | PostureTrend's 0.1 line (`_OVER_BAR_LINES`) | – |
| clean count per evaluation | fallback: `full_horizon_fraction` | `screen_v2_clean_count / screen_v2_episodes`; `screen_v2_clean_lcb` against `min_clean_stance_lcb` | stages without the screen |
| per-actuator policy std | checkpoint zips (section 3.6), else `diagnostics.npz` `algo_std` (the mean std) as one line | `policy_probe.npz` `std_per_actuator` against `update_timesteps` (telemetry item 3, not yet built) | – |
| whole-body balance | – | `com_fore_aft_mean_m`, `xcom_toe_margin_min_m`, `xcom_heel_margin_min_m`, with 0 | main runs; models without a box sole |

The x axis is `timesteps / 1e6`. Vertical marks on every panel: the handoff step (green dashed), the resume step (grey
dotted), and the advisories (ticks coloured by `source`). `diagnostics.npz` does not continue across a resume (the
known gap the telemetry design keeps, §12), so the training-rollout curves of a resumed node start at the resume;
the caption says so when `resume_load_path` or a `resume_same_stage` load is recorded.

### 3.6 Per-actuator std without the policy probe

For each of `robust_best_model.zip`, `best_model.zip`, `<label>_final.zip` and at most 12 kept periodic checkpoints
(evenly spaced by step), the report reads only the `policy.pth` member with `zipfile`. It loads it with
`torch.load(io.BytesIO(...), map_location="cpu", weights_only=True)` and takes `exp(state["log_std"])`. The step is
the zip's `num_timesteps`. Measured: 6 ms per zip, with numpy and torch RNG states unchanged (S/foot/stage_report/proto).
**The reads stop after 10 s of wall clock** (a Drive FUSE mount may fetch a whole zip on first access, and a JUDGE
session in a fresh runtime has none of them cached); the panel then says "std from N of M checkpoints" and the JSON
records `policy_std.read`/`policy_std.available`. The three named checkpoints are read first. It is skipped when torch
is not importable (the offline script on a laptop without torch) or when a policy has no `log_std` (SAC). Actuator
labels come from the foot-load sidecar's `actuator_names` (section 4.2), else from the certificate's
`headings[0].action.per_actuator[*].joint`, else the indices.

### 3.7 Species and embodiment fallbacks

| Model | Box sole | Span (toe/heel lines) | Digits | Recorder today (unchanged) | Card, page 2 |
|---|---|---|---|---|---|
| T. rex anatomical | plantar pad | yes | d2-d4, hinged | stance and recovery replays | every panel. Full map: pad, digits, ankle, heel and toe lines |
| Compsognathus anatomical | plantar pad | no | d2-d4 (hinged where the morphology finds the MTP joint) | stance and recovery | CoP, roll rails, digit share. Map with digits |
| Compsognathus robot | sole box | no | no | stance and recovery | CoP, roll rails. Map: the sole box only |
| Dibothrosuchus | box per foot (4) | no | no | stance and recovery | CoP, roll rails. A 2×2 map grid |
| Velociraptor | none (capsules: d3, d4, metatarsus) | no | – | none | no CoP panels. The map shows the support capsules and their loads (section 4) with no CoP path or bars |
| Brachiosaurus | ellipsoid per foot (4) | no | no | none | no CoP panels. A 2×2 map grid of ellipsoid outlines with loads |

Where a model has no box pad, the CoP panel gives its space to the panels that exist. A panel is never drawn empty.
On every model the map comes from the dedicated episode (section 4), which attaches the recorder to its own new env
only; the "Recorder today" column stays as it is.

---

## 4. The foot load map: one dedicated episode

### 4.1 Why not the replay's recorder

The first draft extended the rendered replay's floor-truth recorder to every model and stage and shared its trace.
The review rejected that:
- **It can cost the videos.** The recorder's substep hook runs inside `mj_step`'s loop with no guard
  (base_env.py:1302-1305; recorder.py:512-554 guards only the balance capture). `record_stage_video` does not guard
  `render_env.step` (evaluation.py:375), and `_record_stage_replays` wraps both replays in one `try`
  (stage_artifacts.py:2127-2231). A hook exception on an env that never had a recorder (Velociraptor, Brachiosaurus,
  every locomotion and behaviour replay) would log "Video recording failed", lose the selected video and skip the
  final replay.
- **It can cost the stance CSV's pad columns.** `per_geom=True` adds an unguarded reduction inside `_reduce`
  (recorder.py:811-815). `_merge_pad_columns` handles any `end_episode` failure by dropping every pad column
  (evaluation.py:488-511).
- **It breaks two invariants.** The telemetry design's rule 1.5 (no model attaches a recorder it did not attach
  before) and `replays/` holding only videos and stance CSVs (test_stage_layout.py:224-227).

A dedicated episode costs one more unrendered episode per stage (section 11) and touches none of that code.

### 4.2 The outline geometry

`gait/foot_load.py:foot_outline(env, morphology) -> dict` runs on the dedicated episode's env before its reset, on a
**scratch `MjData`** of the env's model. It is set to the reset keyframe: `_reset_keyframe_id`, validated against
`model.nkey`, with `qpos0` when the model has no keyframe. Then `mj_resetDataKeyframe` and `mj_forward` run on the
scratch data, exactly as `Morphology.from_env` chooses the sole axes (morphology.py:384-391). The env's own `data` is
never touched, and no RNG is drawn.

Per foot it records the foot frame:
- with a sole: the origin is the sole geom's centre. x is the sole's signed forward axis (`sole_forward_axis`,
  `sole_forward_sign`). y points to the animal's left (`sole_lateral_axis`, `sole_lateral_sign`).
- without a sole: the origin is the reference site, x the root's +x projected onto the horizontal, y to the left.

In that frame it records:
- `half_size` (fore, lateral) of a box sole;
- `ankle_x` (`ankle_forward_m`), and the `heel_x`/`toe_x` lines where the foot declares a span;
- one outline polygon per geom of the limb's `leg_geoms`. The polygon is the convex hull of the geom projected onto
  the frame's xy plane: a box from its 8 corners, a capsule or cylinder from 2×16 points on its end circles, a
  sphere or ellipsoid from 64 surface points. At most 40 vertices each, rounded to 0.1 mm;
- each hinged digit's MTP point and tip point (the axis `support_load_pos` is measured along: the recorder's
  `_digit_axis_ends`).

The outline is drawn at the home pose; the caption says so. It also records `actuator_names` (the model's actuator
names, for the std panel's labels).

### 4.3 The sidecar: `mesozoic.foot-load/v1`

`<stage_dir>/stage_report_foot_load.json`, beside the card (not under `replays/`):

```json
{
  "schema": "mesozoic.foot-load/v1",
  "source": "replay_seed_episode",      // or "stance_csv" / "foot_contact_columns" when no episode ran (4.5)
  "species": "...", "embodiment": "...", "stage": 1, "label": "selected",
  "checkpoint": "robust_best_model.zip", "checkpoint_sha256": "sha256:...", "normalization_sha256": "sha256:...",
  "seed": 3043, "device": "cuda", "control_dt": 0.01, "max_steps": 1000,
  "steps": 1000, "terminated": false, "truncated": true, "termination_reason": "",
  "window_start": 200,                 // the stage's settle_steps (0 when undeclared); means are over [window_start, T)
  "body_weight_n": 840.9,
  "actuator_names": ["neck_pitch", "..."],
  "bars": {"max_sole_cop_fore_aft": 0.8, "max_foot_cop_fore_aft": null},
  "feet": [{
    "label": "r", "sole": "r_plantar_geom", "half_size": [0.07, 0.05], "ankle_x": -0.14, "heel_x": -0.07, "toe_x": 0.07,
    "limb_load_n": 412.1, "limb_load_bw": 0.49, "offsupport_load_n": 0.0,
    "geoms": [
      {"name": "r_plantar_geom", "role": "sole", "outline": [[x, y], ...], "load_n": 263.0, "share_of_limb": 0.64,
       "loaded_fraction": 1.0},
      {"name": "r_toe_d3_geom", "role": "digit", "outline": [...], "load_n": 74.1, "share_of_limb": 0.18,
       "loaded_fraction": 0.98, "mtp": [x, y], "tip": [x, y], "load_pos_mean": 0.62},
      {"name": "r_metatarsus_geom", "role": "other_leg", "outline": [...], "load_n": 3.2, "share_of_limb": 0.008}
    ],
    "pad_cop": {"fore": [...T...], "outer": [...T...], "fore_mean": 0.41, "fore_max_abs": 0.66, "outer_mean": -0.05},
    "foot_cop": {"x": [...T...], "mean": 0.01, "std": 0.004}
  }],
  "error": null
}
```

- `share_of_limb` is the geom's mean force divided by `mean(leg_floor_mean)`.
- An off-support leg geom is listed only if its mean force is above 0.5 N. Support geoms are always listed.
- The per-step series are kept at full length, rounded to 5 significant digits, with `null` for NaN
  (`allow_nan=False`). That is about 50 KB for a biped and 100 KB for a quadruped over 1,000 steps.
- A foot without a sole has no `pad_cop`, `half_size` or `bars`.
- The name matches no reader's glob: not `iter_replay_files` (`replays/` only, `.mp4`/`.csv`), not
  `result_bundle/evidence.py`'s `*_steps.*` (evidence.py:350), not `reentry.py`'s `*/stage_config.json`.

### 4.4 The episode: `evaluation.roll_foot_load_episode`

A new function; nothing existing is refactored.

- **The replay's episode.** The handoff pair (`select_handoff_checkpoint`), `load_sb3_model` with SB3's default
  device (as `_record_stage_replays` does), `load_vecnormalize` with `training = False` and `norm_reward = False`,
  `env_class(**env_kwargs)` with no render mode, `reset(seed=replay_seed(seed, stage))`, `model.predict(...,
  deterministic=True)` on `normalize_obs(obs)`, until terminated, truncated or **`max_steps = 1000`**, the default of
  `record_stage_video(max_steps=1000)` (evaluation.py:252). A test pins the constant against that default, not
  against `max_episode_steps`, which differs on some stages. The recorder (`SubstepContactRecorder(env, morphology,
  per_geom=True)`) is attached after the reset, as the replay's pad recorder is (evaluation.py:356-365).
- **Same actions as the video, pinned.** A test runs `record_stage_video` (mediapy stubbed, a stub deterministic
  policy, T. rex stance and one locomotion stage) and this function on the same seed and compares the per-step
  actions and rewards bit for bit. If a model's render mode ever changed its physics, the test fails and the caption
  would have to stop claiming "the replay's episode".
- **Its own env and generator.** A fresh instance seeded only by that `reset`; the recorder detached and the env and
  the VecNormalize closed in `finally`.
- **RNG-neutral.** The whole call, model load included, runs inside `preserve_global_rng()` (rule 1.3). If the
  telemetry PR's item 3 lands a guard first, the build reuses it.
- **Bounded and isolated.** One episode: 1,000 control steps on the T. rex. The recorder's own measurement is
  1.15 s → 1.53 s + 44 ms of reduction per 1,000-step T. rex episode (recorder.py:23-30); with the policy forward
  pass, the model load and the env build, an estimated 3-6 s on Colab CPU (the PR records the measured time). It is
  skipped, with the reason in the map panel, when there is no handoff pair, when `Morphology.from_env` fails, on an
  interrupted CLI node, or when the caller disallowed simulation. A hook or reduction exception ends it and costs
  the map only (rule 1.2).
- It writes `stage_report_foot_load.json` through the card's private-then-staged publication (rule 1.4).

### 4.5 Without the episode (older runs, the script without `--rollout`, a failed episode)

The map degrades in this order:
1. **The stance CSV.** `replays/*_selected_stance.csv`, with #620/#622's pad columns
   (`<label>_sole_cop_fore_aft`, `<label>_digit_share`, `<label>_<digit>_load_share`, `<label>_<digit>_load_pos`,
   `<label>_foot_cop_fore_aft`). This gives load **shares** per pad and digit, and the fore-aft CoP over time. It
   is drawn on a labelled **schematic** (a unit pad with stub digits): the CSV carries no geometry or lateral CoP.
2. **The env's foot-contact columns.** The CSV's `<x>_foot_contact` columns alone give one bar per foot.
3. **Nothing.** One line: "no floor-truth contact data for this stage", with the reason.

The script (2.4) can roll the episode for an old run with `--rollout`.

---

## 5. Page layout

A4 portrait, 8.27 × 11.69 in, 0.4 in margins. Text is 6-7 pt (titles 7.5 pt, the stage title 12 pt).
`rc_context({"pdf.fonttype": 42})` keeps text selectable in Drive's preview. Colours use matplotlib's tab10 and
viridis; PASS and FAIL are spelled out, never shown by colour alone. Matplotlib `Figure` objects are made without
pyplot, so neither the notebook's inline backend nor pyplot's figure registry is touched. `PdfPages(...,
metadata={"Title": ..., "Subject": "mesozoic.stage-report/v1", "CreationDate": None})`.

**Text never leaves its box.** Every table cell and free-text line is fitted to its width before drawing (measured
with the renderer's text extent, then cut with "…" and "+N more"); the full text stays in the JSON, the certificate
and `gate_verdict.json`. The prototype's heading table showed the failure this prevents (the failing-criteria column
ran off the page). A test renders the widest fixtures and asserts every text artist's window extent lies inside its
axes or the page.

### 5.1 Page 1: "Report card" (heights in inches, top to bottom)

| Block | Height | Content |
|---|---|---|
| Title | 0.45 | "<Model> - <stage display name> (<stage id>, stage <ref>) - <ALGO>". The manual-node banner when `context == "manual"`; "backfilled after a lost session" when `context == "backfill"` |
| Provenance | 0.75 | Four monospace lines. (1) run id · seed · commit (8 hex, last session) · library version · generated (UTC) · context. (2) Parent: load path name, sha256 (12 hex) and load mode, or "resumed from its own checkpoint", or "none: a root"; the parent run id when present; widened-from when present. (3) Last-resumed-from line, when present. (4) Trained / declared budget (this session's) · duration · task sha (12 hex) |
| Verdict | 2.3 | **Left (1.7 in):** a PASS / FAIL / NOT JUDGED box (green / red / grey) with the gate kind, `judged_by` and `judged_at`. **Right:** the bar table (section 6.3) with columns bar, value, threshold, margin, met. For the six-heading certificate: a top row for the seeds clean at every heading, then one row per heading (yaw, clean n/N, LCB, full-horizon fraction, reward / statue, failing criteria, PASS/FAIL). Then one line with the six tightest criterion margins, failing criteria first. Under the box, the first three `failures` (fitted) and "+N more in gate_verdict.json" |
| Key numbers | 1.5 | Two columns of six. Best in-training evaluation (mean ± std @ step). Final in-training evaluation. Selected panel (mean ± std, N, k ended early). Final panel. Statue reward and ratio. Clean k/n, LCB (bar). Falls (certificate · selected · final · last evaluation · rollouts). Forward speed (selected · final). Handoff (name @ step, selection score, step_agrees). Steps of robust_best / best / final. Kept checkpoints. Advisories (total, by source) |
| Training curves | 4.2 | 2×2, the same four series as `visualization.plot_training_curves` (visualization.py:89-210), drawn by the card's own drawer onto its axes: evaluation reward with a std band, the statue line and `min_avg_reward`; episode length with the horizon; tilt and forward speed from the training rollouts (`diagnostics.npz`). The handoff, resume and final steps are marked on all four. A parity test compares the plotted arrays with `plot_training_curves`' lines. `visualization.py` is not changed |
| Files | 1.0 | Relative to the run directory, existing files only. Handoff zip and vecnorm; final; the replays (selected/final, front/side); certificate or report text; evidence CSVs; `figures/*.png`; `progress.json`; `gate_progress.npz`; `eval_episodes.csv` (branch); the foot-load sidecar; `stage_report.json`. The probes the stage declares are named as "written after this card" |
| Footer | 0.2 | "page 1/2 · mesozoic.stage-report/v1 · report only: gate_verdict.json is the record" |

### 5.2 Page 2: "Posture and feet"

| Block | Height | Content |
|---|---|---|
| Title | 0.35 | "Posture over training and the foot load map" |
| Posture panels | 5.4 | Up to 7 stacked panels with a shared x axis, each 0.65-0.9 in tall depending on how many are present (section 3.5). Legends use 5.5 pt, up to 4 columns |
| Foot load map | 3.8 | One axes per foot, equal aspect, in the foot frame (x forward to the right, y left up). A biped gets two side by side, 3.6 × 3.4 in each; a quadruped gets a 2×2 grid of 3.6 × 1.7 in. Each geom outline is filled by its mean normal force on one colour scale shared across feet, with a colourbar in N and % of body weight. A hinged digit gets a dot at its mean load position, sized by its load. The pad CoP path is a thin line coloured by time, with its mean marked ×. The pad centre is marked +. Dotted red lines mark `± max_sole_cop_fore_aft × half-length`, and the whole-foot bar where declared. The whole-foot CoP is a mean ± std bracket on the centre line; the declared heel, toe and ankle lines are grey. Each title reads e.g. "r: limb 412 N (49% BW); pad 64%, d2 9%, d3 18%, d4 9%; off-support 0 N". The caption names the checkpoint, the seed, the window, the source ("the selected replay's seed, rolled without rendering") and "outline at the home pose" |
| Footer | 0.2 | page 2/2 |

The card stays at two pages. With no `gate_progress.npz` (a stage that recorded none), the posture block shows the
`diagnostics.npz` reward components instead (the `plot_diagnostics_graphs` series), or a note.

### 5.3 `run_report.pdf` (one page, A4 landscape)

- **Header:** model, run id, seed, algorithm, behavior and target, commit, generated (UTC). "bundle: sealed after
  this report" (the card cannot know the new status). No replication or provisional label (rule 1.9).
- **Table, one row per node in manifest order:** node, origin (this run / reused from <run_id> / not reached;
  "manual" flagged), gate kind, verdict, key bar (value / threshold), best evaluation reward, selected reward /
  statue, ended early, speed, handoff @ step, trained / budget, duration, and the card's path relative to the run
  (or "in run <run_id>" for a reused node).
- **Below the table:** small multiples of each node's evaluation reward curve (one row, up to 6 nodes), the total
  duration, and the count of advisories.
- **Sources:** a row comes from the node's `stage_report.json`. Without one, it comes from `gate_verdict.json`
  and `evaluations.npz` alone, marked "no card". A cross-run reused node reads its source run's card through
  `NODE_HANDOFF[id]["stage_dir"]` (read-only). When that is unreachable it reads `ancestors/<id>/gate_verdict.json`,
  marked "reused from <run_id>".

---

## 6. `stage_report.json`: `mesozoic.stage-report/v1`

### 6.1 Versioning

The rule matches `progress.json`'s on the telemetry branch (docs/TRAINING_ARTIFACT_PRESENTATION.md): the schema
string changes when a key is removed or renamed, or changes meaning, units or type. Adding a key does not change it.
Floats are kept to 6 significant digits (`progress_report._number`'s rule). NaN and infinity become `null`
(`allow_nan=False`). It is written with `sort_keys=True` and `indent=1`. Expected size: 15-30 KB; the budget is
64 KB, and nothing per episode or per step is stored in it. It holds no absolute path: directories are named
relative to the run directory, so the file is the same wherever Drive is mounted.

### 6.2 Fields

```text
schema                    "mesozoic.stage-report/v1"
generated_at              ISO UTC, seconds, "Z"
generated_by              "reporting.stage_report.write_stage_report"
library_version           config.get_library_version()   (the card's code, not the training session's)
context                   "notebook" | "manual" | "cli_train" | "cli_curriculum" | "backfill" | "script"
species, embodiment, model                    ModelId record, display name
stage, stage_id, stage_dir (relative to the run), stage_name, algorithm
run {run_id|null, run_dir_name, seed, n_envs, git_commit, library_version, label, hyperparameters_sha256,
     task_sha256, plant_version,
     load_mode, load_path, parent_checkpoint_sha256, parent_task_sha256, parent_run_id, resumed_root: bool,
     resume_load_path, resume_checkpoint_sha256, resume_timesteps,
     widened_from_path, widened_from_checkpoint_sha256, widened_from_run_id,
     budget_timesteps, session_timesteps, trained_timesteps, duration_seconds}
verdict {source: "gate_verdict.json"|"stage_results"|"unreadable"|null, gate_kind, gate_schema_version,
         passed: bool|null, failures: [str], judged_by, judged_at, gate_sha256, gate_verdict_sha256,
         checkpoint_sha256, thresholds: {...},
         bars: [{name, value, threshold, direction: "min"|"max", margin, met: bool|null, unit, source, detail}],
         bars_agree_with_verdict: bool|null,
         certificate: null | {file, schema, n_clean_seeds, n_seeds, clean_seed_lcb, bar,
                              headings: [{yaw_deg, n_clean, n_episodes, clean_lcb, full_horizon_fraction,
                                          mean_reward, statue_mean_reward, statue_ratio, hop_or_fall_episodes,
                                          criterion_failures: {key: count}, passed}],
                              criteria: [{key, metric, direction, bar, worst, margin, failing}]},
         note: str|null}                      # e.g. "certificate is for another checkpoint", "manual node"
numbers {best_eval {mean_reward, std_reward, timesteps, episodes},
         final_eval {...},
         selected_panel {checkpoint, episodes, mean_reward, std_reward, mean_length, ended_early,
                         mean_forward_velocity, std_forward_velocity, mean_distance, success_rate, evaluation_seed},
         final_panel {...},
         statue {mean_reward, source, ratio_selected},
         clean {n_clean, n, lcb, bar, source},
         screen_last {episodes, clean_count, clean_lcb, timesteps} | null      # branch: in-training screen
         falls {certificate {ended_early, episodes}, selected_panel {...}, final_panel {...},
                last_evaluation {timesteps, causes: {cause: count}}, training_rollouts {episodes, falls} | null},
         handoff {name, zip, vecnorm, sha256, normalization_sha256, timesteps, mean_reward, std_reward,
                  selection_rule, selection_score, first_scoring_timesteps, step_agrees, source},
         checkpoints {robust_best_model {timesteps, sha256}, best_model {...}, final {...},
                      periodic_kept: [int], progress_json_agrees: bool|null},
         advisories {total, by_source: {source: count}, kept: [{source, timesteps, message}]}}
posture {last {timesteps, values: {key: float}}, at_handoff {timesteps, values: {...}}}
        # values: every gate_progress.npz scalar in the posture whitelist present in the file:
        # *_over_bar_fraction, max_*_mean, <label>_{sole_cop_fore_aft, foot_cop_fore_aft, digit_share,
        # toe_margin_nm, heel_margin_nm, sole_tilt_deg, sole_corner_lift_m}, leg_dc_*, full_horizon_fraction,
        # unsupported_duty, termination_*; branch: screen_v2_*, leg_q_*, BALANCE_SCALAR_KEYS, FORCE_KEYS,
        # <label>_cone_use_max
policy_std {source: "policy_probe.npz"|"checkpoints"|"diagnostics.algo_std"|null, actuator_names,
            read: int, available: int,
            at: [{name, timesteps, std: [float]}]}            # at most 15 entries; the series stays in the files
foot_load {source: "replay_seed_episode"|"stance_csv"|"foot_contact_columns"|null, file,
           checkpoint_sha256, seed, window_start, steps, skipped_reason: str|null,
           feet: [{label, limb_load_n, limb_load_bw, offsupport_load_n,
                   geoms: [{name, role, load_n, share_of_limb, loaded_fraction, load_pos_mean}],
                   pad_cop_fore_mean, pad_cop_fore_max_abs, pad_cop_outer_mean, foot_cop_mean}]}
files {checkpoints: [...], replays: [...], figures: [...], evidence: [...], reports: [...], records: [...]}
      # paths relative to the run directory, existing files only
pdf {rendered: bool, pages: int|null, bytes: int|null, error: str|null}
      # "rendered" means rendered and closed locally; publication of each file is publish_tree's (it may skip one)
errors [{block, error}]                          # "TypeError: ..." per failed block or panel
ledger {...}                                     # 6.4
```

No `replication`, `certification_seeds` or `provisional` field (rule 1.9).

### 6.3 Bars per gate kind (`reporting/stage_report_bars.py`)

All bar extractors are pure functions of the files and `stage_results`. **None of them is called by
`evaluate_stage_gate`, and `gates.py` is not changed.** Each bar has `margin = value - threshold` for a `min` bar
and `threshold - value` for a `max` bar, so a positive margin means the bar is met.

| Kind | Bars |
|---|---|
| `reward_and_length/v1` | Each declared criterion: `min_avg_reward`, `min_avg_episode_length`, `min_avg_forward_vel` (when > 0), `min_success_rate` (when > 0). Values use the judge's key precedence (gates.py:1576-1617); `source` names the key used |
| `stance_quality/v2` | `clean_lcb ≥ min_clean_stance_lcb` (the count beside it), `full_horizon_fraction ≥ min_full_horizon_fraction`, `mean_reward ≥ min_avg_reward`, `mean_reward / statue_mean_reward ≥ min_avg_reward_statue_ratio`, `hop_or_fall_episodes ≤ max_hop_or_fall_episodes` (each when declared). Then each declared `EPISODE_CRITERIA` row `(key, metric, direction)`: `worst` is the min or max of `episode_evidence[*][metric]` over the panel's episodes, `failing` is `result.criterion_failures[key]` |
| `stance_quality_heading/v1` | The certificate statistic first: `clean_seed_lcb ≥ min_clean_stance_lcb` with `n_clean_seeds/n_seeds`. Then the v2 rows for each heading (`headings[*].result`). The criteria rows are pooled over every heading's `episode_evidence` (worst value; `failing` summed) |
| `stance_quality/v1` | The report's own `metrics` against `thresholds`, the rows `render_stance_gate_report` prints |
| `task_success/v1` | `best_model_success_lcb ≥ min_success_lcb` (with k/n), and the reward rail where declared |
| `recovery_quality/v1` | successes / episodes on the frozen panel (detail only; `met = null`), plus the recorded failures |
| `none/v1`, unknown, absent | none. The recorded failure text only |

`bars_agree_with_verdict` is `all(bar.met)` compared with `verdict.passed`, for the kinds whose bars cover the whole
decision (`reward_and_length/v1`, `stance_quality/v2`, `stance_quality_heading/v1`, `task_success/v1`). It is `null`
for the rest. On disagreement the card prints, in the verdict box, "the bar table disagrees with the recorded
verdict; the verdict stands", and the log carries one warning. Section 15 has a test that keeps the two in
agreement on synthetic cases.

### 6.4 The `ledger` block: flat, for the results ledger

The parallel results-ledger PR (`results/offline_judgements.csv`, and the drive-summary notebook's
`logs/runs_summary.csv`, google_drive_summary.ipynb cells 23-24) reads `<stage_dir>/stage_report.json` when present and
takes the `ledger` block as columns. Every value is a scalar or null. The names are stable under v1:

```text
report_schema, report_generated_at, gate_verdict_sha256, species, embodiment, stage, stage_id, run_id, seed,
algorithm, git_commit, context, gate_kind, passed, judged_by, failures_count,
n_clean, n_panel, clean_lcb, clean_bar, statue_ratio, statue_reward,
best_eval_reward, best_eval_timesteps, final_eval_reward, selected_reward, selected_reward_std, final_panel_reward,
ended_early_selected, episodes_selected, ended_early_certificate, episodes_certificate,
forward_speed_selected, distance_selected, success_rate_selected,
handoff, handoff_timesteps, handoff_sha256, robust_best_timesteps, best_timesteps, final_timesteps,
trained_timesteps, budget_timesteps, duration_seconds, load_mode, parent_checkpoint_sha256, parent_run_id,
resumed, max_sole_cop_fore_aft_over_bar_fraction_last, max_foot_cop_fore_aft_over_bar_fraction_last,
max_sole_tilt_deg_over_bar_fraction_last, screen_clean_lcb_last, advisories_total, foot_load_source,
pad_cop_fore_mean_<label>, digit_share_<label>, limb_load_bw_<label>
```

The per-foot columns vary with the model's foot labels (`r`/`l`, or the quadrupeds' four); a combined CSV takes
their union.

### 6.5 How the ledger reads it

- `read_stage_report(stage_dir) -> dict | None` uses `read_json_object`. It accepts only a `schema` starting with
  `"mesozoic.stage-report/v1"` and returns `None` otherwise. The block is plain JSON, so a ledger that lands before
  this PR can parse it with `json` and the schema check alone; whichever PR lands second wires the call.
- **Staleness.** When the stage holds a `gate_verdict.json` whose sha256 differs from `ledger.gate_verdict_sha256`,
  the report summarised an older verdict (a backfill or re-judge since). The ledger then sets
  `report_stale = True` and takes `passed` from the verdict.
- **Missing card.** A stage without `stage_report.json` (every run before this PR) keeps the ledger's existing
  parse path (the drive summary notebook's `check_stage_passed` and `summary.json` routing).
- The library exports `ledger_row(record)` and `read_stage_report`. The drive-summary notebook already checks out
  the repository (`test_sb3_notebook_pins.py:4059`,
  `test_the_drive_summary_checks_out_repo_ref_with_the_sb3_notebooks_git_block`), so it can import them.

---

## 7. `run_report.pdf`

`reporting.stage_report.write_run_report(run_dir, *, model_id, algorithm, seed, nodes, target=None, context)` builds
section 5.3's page and publishes it with an atomic copy, rendered privately first (rule 1.4). It writes no JSON: its
data is each node's `stage_report.json`, and the ledger reads those directly.

| Caller | When |
|---|---|
| notebook chain loop and manual cell | inside `write_run_report()`, after the backfill, after `write_training_summary`, before `save_run_bundle`, after every trained or judged node. So the run report is always current up to the last node, including a node that failed (the bundle is written before the halt) |
| CLI `train_curriculum` | after each node's card, and once after the loop, in `base_dir` |
| CLI `train` | none: a single stage, no run |
| script | `--run-report RUN_DIR`, written to `--out` (refused inside a run, 2.4) |

A re-run of a complete run reuses every node and calls neither writer (the REUSE branch `continue`s before the
bundle save, sb3_training.ipynb line 1257), and every writer refuses a complete bundle (rule 1.5). So
`_REGENERATED_ARTIFACTS` (bundles.py:32-34) stays as it is, and the complete-marker rebuild rule (bundles.py:540-567:
only regenerated artifacts may disagree) never meets a card that changed after sealing.

---

## 8. Inline display in the notebook

`notebook_runtime.display_stage_report(stage_dir)`:
- reads `stage_report.json`, calls `stage_report_render.render_stage_report(record, stage_dir)`, which returns the
  two `Figure` objects (the same function the PDF writer uses), saves each to PNG bytes in memory at **150 dpi**
  and shows them with `IPython.display.display(Image(data=..., width=900))`. A PNG at the inline backend's default
  dpi makes 6-7 pt text 6-9 px tall (the review's 100 dpi raster); `Image` also does not depend on the inline
  backend's figure formatter;
- writes nothing;
- without a card, prints "No stage report in <stage_dir> (see the log above)";
- catches and prints any exception: a display must not stop "Run all".

`display_run_report(run_dir, nodes)` does the same for the run page.

Notebook edits (`notebooks/sb3_training.ipynb`):
- **Cell 15:** import the two display functions, `write_run_report` and `ensure_stage_reports`; add the
  `write_run_report()` helper (2.3).
- **Cell 20 (chain loop):** `display_stage_report(stage_dir)` after `display_stage_videos(stage_dir)` (line 1394);
  `write_run_report()` between `write_training_summary(...)` and `save_run_bundle(...)` (lines 1400-1401).
- **Cell 22 (manual):** `report_context="manual"` on its `generate_stage_artifacts` call;
  `display_stage_report(_manual_dir)` after line 1520; `write_run_report()` inside the existing `try`, before
  `save_run_bundle`.
- **Section 8:** a new markdown cell, "### Report cards", and a code cell, placed **before** "Verify the result
  bundle" (cell 29/30), that calls `display_stage_report` for each `(stage_ref, stage_dir)` in `completed_stages`
  and for each cross-run ancestor's `NODE_HANDOFF[id]["stage_dir"]` (read-only), then `display_run_report(RUN_DIR,
  ...)`. The markdown says the cards were saved and declared before the bundle was sealed, and that this cell only
  redraws them.
- **Section 6 markdown:** one sentence naming `stage_report.pdf`, `stage_report.json` and `run_report.pdf`.

The existing inline curve calls stay. The card adds the verdict, key numbers, posture and feet. The locomotion-health
and behavioural-metrics figures are not on the card. A later change could drop the duplicated curves; this PR does not.

---

## 9. Dependencies

- `matplotlib` (already a training dependency; `visualization` imports it lazily), `matplotlib.backends.backend_pdf.PdfPages`,
  `matplotlib.figure.Figure`, `matplotlib.patches`. It is imported lazily inside the render module only. `stage_report.py`
  (collect, JSON, `ledger_row`, `read_stage_report`) imports neither matplotlib nor mujoco nor torch, so the ledger
  can use it with numpy alone.
- numpy, json, zipfile, io (stdlib).
- torch: optional, only for section 3.6, guarded by `ImportError`.
- mujoco and SB3: only in `gait/foot_load.py` and `evaluation.roll_foot_load_episode`, imported inside the function.
- IPython: only in `notebook_runtime`'s two display functions, inside the function.
- No new package; `pyproject.toml` is unchanged.

---

## 10. Failure isolation

| Failure | Effect | Never |
|---|---|---|
| A data block raises (an unreadable certificate, a malformed npz) | that block is `null`, `errors += {block, error}`, its panel shows "unavailable: <type>", one warning | another block, the verdict, `stage_results` |
| A panel raises during drawing | a text placeholder in that panel's axes; the PDF still renders | the other panels |
| `PdfPages` or `savefig` raises | no PDF; the JSON says `pdf.rendered = false` with the error; the private temp file is discarded | a partial PDF in the stage directory |
| `KeyboardInterrupt` while rendering | it propagates; the private directory is removed; `staged_artifacts`' `finally` publishes only what was complete when moved in (rule 1.4) | a partial PDF published by the staging `finally` |
| JSON serialisation raises | no JSON; the PDF is still published if it rendered | a partial JSON |
| The publish copy fails (Drive) | `publish_tree` logs and skips that file (stage_layout.py:186-199); the other card file may still land | an exception; a `.tmp` that survives hashing (it is discarded, manifest.py:50-65) |
| The foot-load episode raises (no handoff, plant mismatch, MuJoCo, the recorder hook inside `mj_step`, the reduction) | the map panel says why; `foot_load.skipped_reason` | the replays, the stance CSV, a leaked RNG draw (the restore runs in `finally`), an open env |
| The per-actuator std reads are slow (Drive) | they stop at 10 s; "std from N of M checkpoints" | an unbounded card time |
| A reclaim between the verdict and the card | the node has no card; (f) writes it before the next bundle save in a later session | a card written into a complete bundle, or written where no save follows |
| The run is sealed (`complete`) or a manifest is unreadable | the writer skips with one log line | a write |
| `display_stage_report` raises | the cell prints it and continues | a stopped "Run all" |

**The bundle never declares a missing file.** The manifest is built from the files on disk at sealing
(`build_artifact_manifest` hashes `run_path.rglob("*")`, manifest.py:113). A card that failed is simply absent, so it
is not declared. A card that exists is declared and hashed. A file is never declared and then deleted: the writers
only create or replace atomically, and they write nothing after `save_run_bundle`.

`_write_stage_report_card` in `stage_artifacts.py` is itself one `try`/`except Exception` around
`stage_report.write_stage_report`. That costs the card and never the stage, as `_record_stage_replays` and the
graph block already behave.

---

## 11. Size and time budget

Measured with `S/foot/stage_report/proto/proto_render.py`, a prototype that is not the build. It reads real files:
the r10 screen's `progress/s60/{evaluations,gate_progress,diagnostics}.npz` and the r9 arm L s42 certificate
`foot/next_signals/howclose/cert_r9armL_s42_6291456_optarmL/stance_heading_report.json` (1.8 MB). It renders the two
A4 pages with `PdfPages` (Agg, 4 CPU container):

| Case | Evaluations | PDF | Render | Read |
|---|---|---|---|---|
| as recorded | 18 rows | 73,065 B | 2.01 s | 0.06 s |
| series scaled ×13 (a 12M-step run at a 50,000-step evaluation cadence) | 234 rows | 106,094 B | 1.57 s | 0.03 s |

matplotlib's first import cost 0.5-1.6 s. Reading `log_std` from a checkpoint zip took 6 ms, with no RNG change.
poppler (`pdfinfo`/`pdftoppm`) warns "Mismatch between font type and embedded font file" on the Type 42 fonts; the
pages render. The PR's proof run opens the card in Drive's preview; if text renders wrongly there, the build drops
`pdf.fonttype` to matplotlib's default (Type 3), which Drive also renders.

| Artifact | Expected | Budget (warn above) |
|---|---|---|
| `stage_report.pdf` | 80-250 KB | 1 MB |
| `stage_report.json` | 15-30 KB | 64 KB |
| `stage_report_foot_load.json` | 50 KB biped, 100 KB quadruped (one per stage) | 256 KB |
| `run_report.pdf` | 50-150 KB | 512 KB |
| card wall time, excluding the episode | 3-6 s, plus Drive reads of the certificate (1.8 MB) and of `policy.pth` members (capped at 10 s) | 30 s |
| foot-load episode | one unrendered episode: an estimated 3-6 s for the T. rex (section 4.4) | one horizon |
| inline display | one more render of the two figures to PNG, about 2-3 s | – |
| backfill (f) | the card and its episode per backfilled node; normally none | – |

Per stage on Colab: about 10-15 s against hours of training and the probes' ~300 episodes. Drive traffic per stage:
three new files of a few hundred KB, written once, against the roughly 30 checkpoint and vecnorm files the stage
already holds. The run report is rewritten once per node. The PR body records the numbers measured on the
QUICK-size proof run and on a full T. rex stance stage.

---

## 12. The result bundle manifest and audit

- **Declared and hashed, with no code change.** `build_artifact_manifest` lists every file under the run directory
  (manifest.py:113-122). `stage_report.pdf`, `stage_report.json`, `stage_report_foot_load.json` and
  `run_report.pdf` are declared and hashed when present at sealing, like `stage_summary.txt` and
  `training_summary.txt`.
- **Optional for old runs.** None of them joins `audit_result_bundle`'s `required_paths` (audit.py:388-423). A
  bundle without them audits exactly as before, and the drive-summary notebook's strict promotion validation
  (`validate_result_bundle(require_complete=True)`, google_drive_summary.ipynb cell 22) passes unchanged on every old
  run. Promotion copies `summary.json` and `collected_results.csv` into `results/` (results/README.md), so no card
  reaches the repository. `collected_results.csv`, `summary.json` and the verdict's `stage_result` do not change,
  because `stage_results` does not (rule 1.1). `compare_summary_to_csv` is unaffected. `config_hash` hashes an
  explicit list (bundles.py:442), not the stage directory.
- **Immutability.** A complete bundle is never written into (rule 1.5). `_REGENERATED_ARTIFACTS` is unchanged, so a
  card that changed after publication is caught like any other certified artifact. The replication-only
  regeneration of a complete bundle (bundles.py:719-735) leaves the cards as they are, and the cards show nothing it
  can change (rule 1.9).
- **Idempotent re-publication.** A partial bundle is rebuilt over freely (bundles.py:526-531); a new card or a
  rewritten `run_report.pdf` is declared by that rebuild. A complete bundle that still verifies returns early when
  the results are identical (bundles.py:719-742), and nothing here wrote into it.
- **New audit warnings, never errors** (`result_bundle/audit.py`, after the manifest checks). For each declared
  `<stage_dir>/stage_report.json`:
  - it is not readable, or its `schema` does not start with `mesozoic.stage-report/` → warning;
  - the stage's `gate_verdict.json` is declared and its manifest hash differs from the card's
    `verdict.gate_verdict_sha256` → warning "stage_report.json summarises an older gate verdict (stale card)";
  - the card's `verdict.passed` differs from the verdict's `passed` → warning.

  A card is a summary, not evidence, so it can never fail a bundle. `result_bundle` must not import `reporting`
  (gate_verdict.py's module note), so the audit reads the card with `json` and checks the two fields itself.

---

## 13. Telemetry-branch fields used (stacking list)

Read on `claude/stance-balance-telemetry` at `71df0723`. "Commit" is the branch commit's subject.

**Soft dependency.** Every branch-only code name below is imported inside `try`/`except ImportError` with main's
behaviour as the fallback; every branch-only file field is optional. The card PR therefore has no hard dependency on
the branch, which is still being rewritten. It does not modify any branch function (the first draft's public alias
of `_handoff_in_training` is dropped, section 3.4).

**Present on the branch:**

| Field | File | Branch commit | Used for | Fallback without it |
|---|---|---|---|---|
| `screen_v2_episodes`, `screen_v2_clean_count`, `screen_v2_clean_lcb` | `gate_progress.npz` | "Keep a ledger of every evaluation episode and screen it on the stance gate's own bars" | clean count per evaluation (page 2); `numbers.screen_last`; ledger `screen_clean_lcb_last` | `full_horizon_fraction` |
| `screen_v2_yaw0_bars_met`, `screen_v2_hop_or_fall_episodes`, `screen_v2_fail_<criterion>_count` | `gate_progress.npz` | same | JSON `posture` snapshot only | absent |
| `com_fore_aft_mean_m`, `xcom_toe_margin_min_m`, `xcom_heel_margin_min_m` | `gate_progress.npz` | "Record whole-body balance and floor forces in the floor-truth recorder, opt-in" (wired by the ledger commit) | balance panel (page 2) | panel omitted |
| every `gait.balance_telemetry.BALANCE_SCALAR_KEYS` and `FORCE_KEYS` key; `<label>_cone_use_max` | `gate_progress.npz` | same | JSON `posture` snapshot only | absent |
| `leg_dc_<l>_<r>_mean_<joint>_deg`, `leg_q_<l>_minus_<r>_<joint>_deg`, `leg_q_<l>_<r>_mean_<joint>_deg` | `gate_progress.npz` | ledger commit, balance commit | JSON `posture` snapshot only | main's `leg_dc_<l>_minus_<r>_<joint>_deg` |
| `selected`, `timesteps`, `mean_reward`, `std_reward`, `episodes`, `selection_rule`, `selection_score`, `first_scoring_timesteps`, `step_agrees`, `error` | `handoff_in_training.json` (`stage_artifacts.HANDOFF_SIDECAR_FILENAME`) | "Record the handoff's in-training step and score beside the stance certificate" | `numbers.handoff`; the handoff mark on the curves | the card's own computation (section 3.4), for any kind |
| `warnings[*]` with `source == "posture_trend"` | `progress.json` | "Warn when the signed centre of pressure trends toward the edge it stands over" | advisory ticks; the 0.1 PostureTrend line on the over-bar panel | main's sources only (plateau, baseline, hop, posture drift) |
| `figures/gate_progress.png`, `eval_episodes.csv` | stage directory | "Plot gate_progress.npz at stage end and on demand"; ledger commit | listed in the files section only | not listed |
| `visualization._gate_progress_groups`, `_OVER_BAR_LINES` | code | "Plot gate_progress.npz at stage end and on demand" | panel grouping and the drift/trend lines on page 2 (imported, not refactored) | a local table pinned against them when they exist |

**Not read:** `actuator_force_frac_per_actuator`, `full_horizon_term_*`, `fall_term_*`, the Hof, shear and moment-arm
keys, `diagnostics.npz` `reward_action_jerk`, `progress.json` `evaluation.screen_v2` / `balance` / `forces` (the card
reads `gate_progress.npz` directly), and the ledger's per-episode columns.

**Designed in the telemetry design but not on the branch at `71df0723`** (items 3, 4 and 8). The card reads them
when present. Until then it uses the fallback, and nothing waits for them:

| Field | File | Telemetry item | Used for | Fallback |
|---|---|---|---|---|
| `update_timesteps`, `std_per_actuator` | `policy_probe.npz` | 3 | the per-actuator std panel | checkpoint `log_std` (3.6), then `diagnostics.npz` `algo_std` |
| `rollout_episodes`, `rollout_falls` | `diagnostics.npz` | 4 | `numbers.falls.training_rollouts` | null |
| `train_episodes.csv` | stage directory | 4 | files section | – |
| `checkpoints.periodic_timesteps` | `progress.json` | 8 | `numbers.checkpoints.periodic_kept` | listing `models/` by `_PERIODIC_CHECKPOINT_RE` |

**Constants imported, so names cannot drift:** `eval_diagnostics.TERMINATION_KEY_PREFIX` and
`curriculum.progress_report.PROGRESS_FILENAME` (main); `eval_diagnostics.SCREEN_KEY_PREFIX`,
`gait.balance_telemetry.BALANCE_SCALAR_KEYS`, `FORCE_KEYS`, `LEG_Q_PREFIX`, `stage_artifacts.HANDOFF_SIDECAR_FILENAME`
(branch, soft). The item 3/4/8 names are string constants in `stage_report.py` until those items land. A test then
pins them against the producing modules; it is skipped while those modules do not exist.

---

## 14. Code changes, file by file

| File | Change |
|---|---|
| `environments/shared/reporting/stage_report.py` (new) | Schema and file-name constants; `collect_stage_report`; `write_stage_report` (rule 1.5 check, collect, the optional episode, render through `stage_report_render` into a private directory, publish through `staged_artifacts`); `ensure_stage_reports` (2.3); `read_stage_report`; `ledger_row`; `RunReportNode`; `write_run_report`. No matplotlib, mujoco or torch import at module level |
| `environments/shared/reporting/stage_report_render.py` (new) | `render_stage_report(record, stage_dir) -> list[Figure]`, `render_run_report(...)`; the panels, including the card's own training-curve drawer; the text-fitting helper; each panel in its own guard |
| `environments/shared/reporting/stage_report_bars.py` (new) | The per-kind bar extractors of 6.3 |
| `environments/shared/reporting/stage_artifacts.py` | `generate_stage_artifacts(..., stage_report=None, report_context="notebook")`; `_write_stage_report_card` (step 5b). **`_record_stage_replays` and the branch's `_handoff_in_training` are unchanged** |
| `environments/shared/reporting/__init__.py` | export `write_stage_report`, `ensure_stage_reports`, `write_run_report`, `read_stage_report`, `ledger_row` |
| `environments/shared/visualization.py` | **unchanged** |
| `environments/shared/gait/foot_load.py` (new) | `FOOT_LOAD_SCHEMA`, `REPLAY_MAX_STEPS = 1000`, `foot_outline(env, morphology)`, `foot_load_summary(trace, morphology, outline, *, window_start, bars)`, `write_foot_load(path, payload)` |
| `environments/shared/evaluation.py` | **one new function only**, `roll_foot_load_episode(...)` (section 4.4). `record_stage_video`, `_attach_pad_recorder`, `_close_pad_step` and `_merge_pad_columns` keep their bytes |
| `environments/shared/rng_guard.py` (new) | `preserve_global_rng()` |
| `environments/shared/train_base.py` | `train(..., write_stage_report=None)`, called after the env closes; `train_curriculum(..., write_stage_report=True)`, a card per node (no episode on an interrupted node) and the run report |
| `environments/shared/cli.py` | `--no-stage-report` for `train` and `curriculum` |
| `environments/shared/notebook_runtime.py` | `display_stage_report`, `display_run_report` |
| `environments/shared/scripts/stage_report.py` (new) | section 2.4 |
| `environments/shared/result_bundle/audit.py` | the three optional warnings of section 12 |
| `environments/shared/reporting/stage_layout.py` | docstring layout tree only: `stage_report.pdf`, `stage_report.json`, `stage_report_foot_load.json` at the stage root |
| `notebooks/sb3_training.ipynb` | section 8 |

---

## 15. Tests

New `environments/shared/tests/test_stage_report.py` (normal job; matplotlib is already a test dependency through
`test_visualization.py`). Its fixtures are synthetic stage directories: `evaluations.npz`; a `gate_progress.npz`
with main's keys only and one with the branch's keys added; `diagnostics.npz`; `stage_config.json` (one with a
parent edge, one resumed root, one widened root); `gate_verdict.json` through `write_gate_verdict`; a small
six-heading certificate fixture built with the report's own builders; evaluation CSVs; a foot-load sidecar.

- **JSON.** Exact field set and types per fixture. `allow_nan=False`. NaN becomes null. No absolute path anywhere in
  the file. Deterministic given the files (two writes equal except `generated_at`). Size under 64 KB on the largest
  fixture. No `replication`/`provisional` key.
- **Header lineage.** A resumed root shows "resumed from its own checkpoint" and no parent; a node with a parent edge
  that was resumed shows both; a widened root shows its source.
- **Degradation.**
  - An empty stage directory gives a JSON with null blocks and a two-page PDF of placeholders.
  - Main-only `gate_progress.npz` draws no screen or balance panel; branch keys add exactly those.
  - With every branch import made to raise `ImportError`, the card renders from main's fields (the soft dependency).
  - A Velociraptor-shaped directory (no pad keys) gives no CoP panels and the capsule map. A Dibothrosuchus-shaped
    one gives four feet in a 2×2 grid. A Compsognathus robot gives a sole-only map.
  - A non-stance kind gives no certificate block.
- **Layout.** On the widest fixtures (six headings with every criterion failing; a quadruped; long node ids), every
  text artist's window extent lies inside its axes or the page.
- **Bars.**
  - `reward_and_length/v1`: the values and the key precedence match `_reward_and_length_stage_gate`'s inputs on a
    grid of `stage_results`. `all(met) == evaluate_stage_gate(...)[0]` for every grid case.
  - v2 and heading: on fixtures built from the real certificate's structure, the bar rows equal the certificate's
    `result` numbers, and `bars_agree_with_verdict` holds.
  - A certificate for another checkpoint gives no bars and the note.
- **Curves parity.** The card's four training-curve series equal the `Line2D` data `plot_training_curves` draws for
  the same stage directory.
- **Ledger.** `ledger_row` keys equal the documented list. The staleness rule: a changed `gate_verdict.json` makes
  `read_stage_report` plus the ledger helper report `report_stale`.
- **Failure isolation**, each injected separately with monkeypatch: a raising block, panel, `PdfPages`,
  `json.dumps`, `atomic_copy`, the episode.
  - `write_stage_report` never raises.
  - The files that could still be written are written; no `.tmp` is left in the stage directory.
  - `errors` names the failure.
- **Interrupt while rendering.** A `KeyboardInterrupt` raised from inside `PdfPages.savefig` propagates, and after
  `staged_artifacts`' `finally` the stage directory holds no `stage_report.pdf` (and the previous one, if any, is
  intact).
- **Sealed run.** A run directory with a `complete` manifest gets no file, from the writer, the backfill or the
  script. One with an unreadable manifest gets no file. A CLI output directory inside a complete notebook run gets
  none (the ancestor walk of rule 1.5).
- **RNG neutrality.** `random`, `np.random` and `torch` states (and CUDA's when initialised; skipped on CPU CI) are
  equal before and after `write_stage_report`, with and without the episode. The episode uses a stub policy and the
  T. rex env (mujoco job). The same is pinned for `preserve_global_rng` around a deliberate draw and around an SB3
  load of a checkpoint that saved a seed. `preserve_global_rng` does not initialise CUDA.
- **Pages and budget.** Exactly 2 pages (`PdfPages.get_pagecount`) on every fixture, including six headings and a
  quadruped. The PDF is under 1 MB on the 234-row fixture.
- **Std read cap.** With a patched clock, the checkpoint reads stop at the cap and the JSON records `read < available`.

`test_stage_artifacts_generation.py` / `test_reporting_stage_artifacts.py`:
- **Order.** The step-5b pin of 2.1, in both ordering tests.
- **stage_results unchanged.** It is `==` with and without the card. `gate_verdict.json` is equal in every field but
  `judged_at`.
- **A raising report** (monkeypatched `write_stage_report`) leaves every other artifact and the return value
  unchanged.
- **Defaults.** `stage_report=None` writes a card when `generate_graphs` is true and none when it is false;
  `record_videos=False` writes a card with no episode.
- **Replays untouched.** `_record_stage_replays` calls `record_stage_video` with exactly today's keyword set (the
  existing producer test gains that assertion); `test_stage_layout.py:224-227`'s exact `replays/` listing still holds
  with the card on.

`test_gait_foot_load.py` (new, mujoco job):
- **Reduction.** On a recorded T. rex statue episode (seed 3042, zero action), each support geom's `load_n` equals
  the window mean of `trace.support_floor_mean`. The geoms of a foot plus `offsupport_load_n` sum to `limb_load_n`
  within 1e-9 relative. The outline polygons contain the geom centres. The pad `half_size` equals the MJCF's.
- **Episode equals the replay.** `roll_foot_load_episode` and `record_stage_video` (mediapy stubbed) on the same stub
  policy and seed give bit-identical per-step actions and rewards: T. rex stance, one locomotion stage, one
  Brachiosaurus stage. `REPLAY_MAX_STEPS` equals `record_stage_video`'s `max_steps` default.
- **Hook failure.** A recorder whose hook raises on step 10 ends the episode with `skipped_reason` set; the env is
  closed, the RNG states are restored, and nothing else is written.
- **All six models.** The episode runs and writes a sidecar (or a stated skip) on every model's stance stage.

`test_result_bundle_manifest.py` / `test_result_bundle_audit.py`:
- A bundle saved with cards present declares and hashes them and verifies.
- A bundle without cards (an old run) audits unchanged, and `validate_result_bundle(require_complete=True)` passes on
  it as before.
- A failed card (absent file) leaves no missing declared file.
- A stale card gives a warning and no error.
- `collected_results.csv` and `summary.json` are byte-identical with and without the cards.
- A complete bundle re-published with only a replication change leaves every card byte-identical.

`test_sb3_notebook_pins.py`:
- **Order.** `write_training_summary < write_run_report < save_run_bundle < gate check`, in the chain loop and in
  the manual cell. `display_stage_report` follows `display_stage_videos`.
- **Backfill placement.** `ensure_stage_reports` is called only inside `write_run_report()`, and `write_run_report()`
  only immediately before `save_run_bundle`; the REUSE branch calls neither. Executed on a fixture run: a same-run
  node with a verdict and no card gets one; a cross-run ancestor's directory gets nothing; a reuse-only session
  writes nothing.
- **No writes after sealing.** The section 8 report-cards cell, executed as `test_the_curves_cell_writes_nothing_into_the_sealed_bundle`
  executes the curves cell, writes no file (directory listing before and after), and sits before the verify cell.
- The manual cell passes `report_context="manual"`.

`test_policy_loading.py`: in `CALL_TIME_READS` (test_policy_loading.py:668-682),
`environments/shared/reporting/stage_report.py: {"_PERIODIC_CHECKPOINT_RE": 1}`, and `evaluation.py`'s `_ensure_sb3`
count from 2 to 3 for `roll_foot_load_episode`.

`test_curriculum_cli_end_to_end.py` / `test_train_wiring.py`:
- CLI `train` writes a card by default (run id null, no run report) and none with `--no-stage-report`.
- The notebook's `train()` call (`report_metrics=False`) writes none.
- `train_curriculum` writes a card per node and `run_report.pdf`; an interrupted node's card reads NOT JUDGED and
  has no episode.

Script test: the default `--out` is refused when the current directory is inside a run, and allowed outside.
`--in-place` is refused under a manifest of any status and allowed without one. `--rollout` is required for the
episode.

**Inertness proof the PR records** (as the telemetry PR's section 10). `digest_snapshot --check` moves 0 lines;
`species_catalog --check` and `plant_contract --check` are current. Two QUICK-size notebook-path runs of the same
seed, `stage_report=False` against the default, give identical `evaluations.npz`, `gate_progress.npz`,
`evaluation_*.csv`, `gate_verdict.json` (minus `judged_at`), replay stance CSVs, probe outputs and checkpoint zips.
The only extra files are the cards, the foot-load sidecars and `run_report.pdf`.

---

## 16. Docs and records

- **`docs/TRAINING_ARTIFACT_PRESENTATION.md`**, new section "Stage report card":
  - the two pages and what each panel reads (sections 3 and 5 in short);
  - the fallbacks per model;
  - the `mesozoic.stage-report/v1` schema and the additive rule;
  - the `ledger` block and the staleness rule;
  - `mesozoic.foot-load/v1`, and why the map comes from a dedicated episode, not the replay (section 4.1);
  - the episode's RNG guard;
  - `run_report.pdf` and the backfill;
  - the regenerate script;
  - "a card restates the record".
- **`docs/RESULT_BUNDLES.md`:**
  - in the canonical layout tree, `run_report.pdf` at the run root and `stage_report.{pdf,json}`,
    `stage_report_foot_load.json` in a stage;
  - one paragraph: optional, declared and hashed when present, written before sealing, never into a complete
    bundle, warnings only in the audit.
- **`reporting/stage_layout.py`** module docstring: the layout tree.
- **`website/docs/training/hyperparameters.md`** tip 4 ("Watch a run mid-training"): one sentence. After each node,
  `stage_report.pdf` in the stage directory is the summary; it opens in Drive's preview. `run_report.pdf` in the run
  directory lists every node.
- **The notebook's** section 6 and section 8 markdown (section 8 above).
- **`CHANGELOG.md` [Unreleased]:** an Added entry in #620's style, no PR number.
- **`docs/CONSOLIDATION_PLAN_2026_09.md`:**
  - a row after the telemetry PR's: "Per-stage report card and run report (outside this sequence)", status
    **Carried out**, report-only, at the maintainer's request of 2026-10-17, ending "The consolidation stays paused.";
  - it records the landing of the PR before it, as the house convention does.
- `test_landing_records.py` must pass.

---

## 17. Commit plan (stacked on the telemetry branch, soft dependency)

1. `rng_guard.preserve_global_rng` and its tests.
2. `gait/foot_load.py` (outline and reduction) and `test_gait_foot_load.py`'s reduction tests.
3. `evaluation.roll_foot_load_episode` (new function only), with the episode-equals-replay, hook-failure and
   six-model tests.
4. `reporting/stage_report_bars.py` and the bar tests, including the agreement grid against `evaluate_stage_gate`.
5. `reporting/stage_report.py` and `stage_report_render.py`, with `test_stage_report.py` (layout, curves parity,
   interrupt, soft dependency).
6. Wiring: `generate_stage_artifacts` step 5b, `train`, `train_curriculum`, `cli.py`, `notebook_runtime`, and the
   pins.
7. The notebook cells (`write_run_report()` with the backfill, displays, section 8 cell) and the notebook pins.
8. `result_bundle/audit.py` warnings and the bundle tests.
9. `scripts/stage_report.py` and its tests.
10. Docs and records (16).

Each commit's suite passes. The whole PR keeps `digest_snapshot --check` at 0 moved lines.

---

## 18. Deferred, open questions and risks

**Deferred:**
- A one-line summary of each probe on the card. The probes run after it (2.1). A later change can have
  `_run_stance_probes` append to the JSON, which would need a second write and is not worth it now.
- Copying an ancestor's `stage_report.json` into `ancestors/<id>/` (the record schema is untouched here).
- The PDF's byte determinism (`CreationDate` is already omitted). Tests pin the JSON instead.
- Retiring the notebook's duplicated inline curves.
- Cards for `train_behaviors.py` behaviour bundles (2.2).
- A foot map per heading for the six-heading certificate (the map shows the selected replay's seed at yaw 0).

**Open for the maintainer:**
1. **Page orientation.** A4 portrait for the card, landscape for the run report? Letter is an alternative; Drive's
   preview shows either.
2. **The episode on the Velociraptor and the Brachiosaurus.** The map's episode attaches the floor-truth recorder to
   a new, report-only env on those two models (no existing env gains a recorder, rule 1.8). The alternative is no
   map on those models. Proposed: attach, since it costs one episode and touches nothing else.
3. **Backfill (f).** Proposed on. It writes only before a bundle save and only into this run's own stage
   directories; the alternative is a run whose interrupted node never has a card in the bundle.

**Risks:**
- *The card disagrees with the verdict.* Guarded by rule 1.7, the agreement tests and the staleness hash. The verdict
  file remains the only record that reuse and the bundle read.
- *The episode differs from the video.* Pinned by the bit-identity test on three models; a model where render mode
  changes the physics would fail it, and the caption would then drop "the replay's episode".
- *Drive latency* on the certificate read (1.8 MB) and the `policy.pth` members. Capped at 10 s; measured in the PR.
- *matplotlib version drift in the table layout.* The layout test checks text extents, not pixels; the page-count and
  opens-cleanly checks stay.
- *Type 42 fonts in Drive's preview* (poppler's warning on the prototype). Checked on the proof run; Type 3 is the
  fallback.
- *The telemetry branch is rewritten.* Fields are named by constant, not by commit hash, imports are soft, and
  section 13's table is re-checked against the branch head when the build starts.

---

## 19. Review

An adversarial review of the first draft against main `2e734e29` and the branch at `71df0723`. Each item: what was
wrong, the evidence, what changed.

### 19.1 Changed

1. **The replay recorder extension could lose the replay videos and the stance CSV's pad columns** (blocking). The
   recorder's hook is unguarded inside `mj_step` (base_env.py:1302-1305; recorder.py:512-554), `render_env.step` is
   unguarded in `record_stage_video` (evaluation.py:375), and both replays share one `try` in `_record_stage_replays`
   (stage_artifacts.py:2127-2231): a hook exception on a model or stage that never had a recorder would cost the
   selected video and skip the final one. `per_geom=True` adds an unguarded reduction (recorder.py:811-815) whose
   failure `_merge_pad_columns` turns into "no pad columns" (evaluation.py:488-511). It also broke the telemetry
   design's rule 1.5 and `replays/`'s pinned listing (test_stage_layout.py:224-227). **Changed:** the replay code is
   untouched (rule 1.8); the map comes from one dedicated unrendered episode on its own env (section 4), whose failure
   costs only the map; the sidecar moved to `<stage_dir>/stage_report_foot_load.json`; the `_attach_floor_recorder`
   refactor, the shared trace and `collect_foot_load` are gone; the first draft's fallback (f) became the only source.
2. **The fallback episode was not the replay's episode.** It ran "to `max_episode_steps`" while the replay stops at
   `max_steps=1000` (evaluation.py:252). **Changed:** `REPLAY_MAX_STEPS = 1000`, pinned against that default, and a
   bit-identity test of per-step actions and rewards against `record_stage_video`.
3. **RNG guard details.** SB3's `load()` reseeds Python, numpy and torch from a checkpoint's saved seed
   (evaluation.py:673-678), a stronger reason for the guard than the policy initialiser alone. Snapshotting CUDA
   unconditionally would initialise it. Forcing the episode onto the CPU would make it diverge from a GPU replay.
   **Changed:** rule 1.3 states all three; CUDA only when initialised; SB3's default device, as the replay.
4. **A `KeyboardInterrupt` during rendering could publish a partial PDF.** `staged_artifacts` publishes its staging
   tree in a `finally` (stage_layout.py:154-168), so "rendered into a local temporary directory" was not enough if that
   directory was the staging tree. **Changed:** rule 1.4 renders privately and moves complete files into staging; a
   test injects the interrupt.
5. **A reclaim between the verdict and the card left a node with no card, forever.** Same-run reuse `continue`s
   before any artifact step (sb3_training.ipynb line 1257), and the script's `--in-place` is refused under the partial
   manifest the run then holds. **Changed:** backfill (f) inside `write_run_report()`, only immediately before a bundle
   save so the save declares it, only into this run's own stage directories, never into a complete bundle (2.3).
6. **The script's default output could land inside a run.** The first draft refused only an explicit `--out` (0, row
   g) while 2.4 defaulted to the current directory. **Changed:** every resolved output, default included, is refused
   inside a run directory, as plot_gate_progress.py:45-58 already does; default `./stage_reports/<run_id>/<stage>/`.
7. **The run-report helper could not find the current node.** It read stage directories from `NODE_HANDOFF`, which
   the chain loop fills only after the bundle save (line 1409 vs 1401) and never for a failed node. **Changed:** this
   run's nodes come from `completed_stages`; `NODE_HANDOFF` only for cross-run ancestors; `origin` is an enum with a
   separate `source_run_id`.
8. **The header misread a resumed root's own checkpoint as its parent.** A root resume records its periodic
   checkpoint under `load_path`/`parent_checkpoint_sha256` with `load_mode = "resume_same_stage"` (config.py:961-971);
   only a node with a parent edge gets `resume_*` keys (config.py:964-969), and only the last resume is kept; `git_commit`
   is the last session's (config.py:940-950). **Changed:** section 3.1 reads the parent only under
   `initialize_next_stage`, says "resumed from its own checkpoint", "last resumed from", "commit (last session)", and
   shows the widen lineage (config.py:607-612).
9. **Inline display was illegible and depended on the inline formatter.** The 100 dpi raster of the prototype
   (`S/foot/stage_report/review/p1_100dpi-1.png`) shows 6-7 pt text at 6-9 px and the heading table's failing column
   running off the page. **Changed:** inline PNG at 150 dpi through `IPython.display.Image` (section 8); a
   text-fitting rule and a text-extent test (section 5).
10. **Existing tests and callers would gain a card and an episode silently.** `stage_report=True` by default would
    run in every existing `generate_stage_artifacts` test (test_compsognathus_longipes_training.py:440-451, the
    stage-layout tests). **Changed:** `stage_report=None` follows `generate_graphs`, the episode follows
    `record_videos`.
11. **Coupling to a branch in flux.** The draft refactored the branch's `plot_gate_progress` into a shared drawer and
    aliased its private `_handoff_in_training` publicly. **Changed:** `visualization.py` and the branch's functions are
    unchanged; the card draws its own curves with a parity test, imports branch names softly (section 13), and computes
    the handoff step itself.
12. **The per-actuator std reads were unbounded on Drive.** Up to 15 zips read through FUSE in a fresh JUDGE runtime.
    **Changed:** a 10 s cap, named checkpoints first, `read`/`available` in the JSON (3.6).
13. **Post-sealing changes a sealed PDF cannot follow.** A complete bundle's replication record may be regenerated
    after sealing (bundles.py:719-735). **Changed:** rule 1.9; the cards show no replication or provisional label.
14. **Interrupted CLI nodes.** A Ctrl-C ended node would have waited for an episode. **Changed:** no episode on an
    interrupted node (2.2). A single CLI `train` stage has no run: run id null, no run report.
15. **Scope gap.** `train_behaviors.py` behaviour bundles were not mentioned. **Changed:** listed as out of scope with
    the reason (2.2, 18).
16. **Wrong citations.** The ordering pin is test_reporting_stage_artifacts.py:1308 (plus test_stance_gate_report.py:3163),
    not test_stage_artifacts_generation.py; `generate_stage_artifacts` is at 1892-2099 on the branch (1777-1969 on
    main). `test_policy_loading.py`'s `CALL_TIME_READS` (668-682) must also count the new function's `_ensure_sb3` read
    in `evaluation.py`. **Changed** throughout. The JSON also lost its absolute paths (6.1) and `pdf.rendered`'s meaning
    is stated.

### 19.2 Checked and kept

- **Declared files.** Cards are written before `save_run_bundle`, and `build_artifact_manifest` hashes every file
  present (manifest.py:113-122): declared, hashed, never declared-then-missing. Kept.
- **Re-entry over a sealed manifest.** The notebook refuses judge/train sessions into a complete run before any write
  (reentry.py:143-210; cell 20's `refuse_write_into_complete_run`), the REUSE branch writes nothing, and the writers
  refuse `complete` themselves. Kept, with the ancestor walk added for the CLI and script (rule 1.5).
- **Promotion of old runs.** `required_paths` (audit.py:388-423) does not change; strict validation
  (google_drive_summary.ipynb cell 22) is unaffected; promotion copies only `summary.json` and `collected_results.csv`.
  Kept.
- **Training, evaluation, verdict, checkpoint selection, digest.** `stage_results` is untouched (rule 1.1); the card
  runs after `_apply_stage_gate` and before the probes; no config key; training reseeds every global generator in
  `make_env` (train_base.py:200-201), so even a leaked draw could not reach training. Kept.
- **Temporaries.** `atomic_copy`'s `.<name>.<random>.tmp` matches the manifest's discard pattern (file_io.py:56;
  manifest.py:39). Kept.

### 19.3 Rejected

- **Moving the card before the figures and replays** to shrink the reclaim window. Rejected: the files section would
  list files that do not exist yet; the backfill closes the window instead.
- **Backfilling on a reuse-only session.** Rejected: with no bundle save after it, the card is undeclared under the
  partial manifest and the "Verify the result bundle" cell fails (`reject_unlisted=True`, audit.py:657, notebook line
  1666).
- **Adding the cards to `_REGENERATED_ARTIFACTS`.** Rejected: it would let a changed card be re-hashed silently into a
  complete bundle, which is what the marker exists to catch (bundles.py:540-567).
- **Requiring the cards in the audit.** Rejected: every existing run would fail promotion.
- **Forcing `device="cpu"` for the episode.** Rejected: on a GPU runtime the replay ran on CUDA, and a CPU forward pass
  can roll a different episode from the video; the guard handles CUDA instead.
- **Embedding `figures/training_curves.png` as an image** instead of drawing vector curves. Rejected: the 14 × 10 in
  figure scaled into 8 × 4 in leaves its text unreadable; the card draws the same arrays (parity test).
- **Showing the probes on the card** by running the card after them. Rejected (as in the draft): a reclaim during
  ~300 probe episodes would then cost the card too.
- **A per-geom recorder on the existing stance replays only** (keeping the replay path for box-soled models). Rejected:
  it still puts `per_geom`'s unguarded reduction in front of the pad columns, and two map sources would need two sets
  of tests for one picture.

## 20. Maintainer decisions (2026-10-18)

1. Page size: A4 (card portrait, run report landscape).
2. Velociraptor and Brachiosaurus: yes, the foot map's one report-only episode attaches the floor-truth recorder to a
   new, report-only env on those models (no existing env or replay changes).
3. Backfill: on, as specified in (f): only before a bundle save, only into this run's own stage directories.
Build: stacked on the telemetry branch once its PR is ready; its own PR.
