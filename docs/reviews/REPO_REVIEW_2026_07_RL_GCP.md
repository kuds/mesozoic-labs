# RL Pipeline + Google Cloud Review — July 2026

> **Archived review.** Still-open items from this review are tracked in
> [`docs/KNOWN_ISSUES.md`](../KNOWN_ISSUES.md); this document is kept as the
> historical record with full context.

Scope: the Google Cloud integration (Vertex AI job submission/orchestration,
GCS artifact handling, TensorBoard sync, docs, packaging), a fresh delta pass
over the SB3 and JAX/MJX training stacks (relative to
`docs/reviews/REPO_REVIEW_2026_06.md` — nothing from that review is re-reported), the
sweep/reporting infrastructure, and all four Jupyter notebooks.

> **Status:** items marked **FIXED** were fixed on this branch in the same
> change set as this document, with regression tests where cheap. Items
> marked **OPEN** are verified findings left for follow-up (mostly larger
> design changes).

Severity legend matches the June review: **CRITICAL** = corrupts results or
breaks a documented workflow, **HIGH** = wrong results in common cases,
**MEDIUM** = edge cases / robustness, **LOW** = cosmetic / QoL.

---

## 1. Google Cloud integration

### Fixed

- **HIGH (docs / broken workflow)** — `website/docs/training/vertex-ai.md`
  told users to run `python environments/shared/scripts/sweep.py launch-all`
  (no such file — the tool is a package invoked as
  `python -m environments.shared.scripts.sweep`) with
  `--search-space-file configs/sweep_ppo.json` (the real files are
  per-species: `configs/<species>/sweep_ppo.json`). Both commands in the
  "Running Long Sweeps from a GCE VM" walkthrough failed as written; the same
  stale config paths appeared throughout `sweeps.md`. **FIXED** (docs now use
  `python -m …` and the species defaults).
- **MEDIUM (robustness)** — `--wandb` could kill a Vertex job at startup:
  `init_wandb` was unguarded, so a missing/invalid `WANDB_API_KEY` on a
  headless worker raised out of `wandb.init` — while `vertex-ai.md` promised
  the flag is "silently ignored". **FIXED** — `wandb.init` failures now log a
  warning and training continues (`wandb_integration.py`).
- **MEDIUM (logging)** — W&B split runs on relaunch: the June fix added
  `id=… , resume="allow"`, but the generated id was never persisted, so a
  preempted/restarted job always got a fresh timestamped id (split run).
  **FIXED** — the id is persisted to `<run_dir>/wandb_run_id.txt` and reused
  when the same output dir is re-used (`init_wandb(run_dir=…)`).
- **MEDIUM (hang)** — `submit.py:_wait_for_job`: a failing state-refresh
  `continue`d past the timeout check, so a persistent API/auth failure
  polled forever even with `--stage-timeout` set. **FIXED** (timeout now
  honored inside the refresh-failure branch).
- **MEDIUM (logging / artifacts)** — TensorBoard events were buffered locally
  and synced to GCS only at stage end, so a preempted spot VM (explicitly
  recommended by the docs) lost the entire stage's TB logs. **FIXED** —
  `PeriodicTbSyncCallback` (tb_sync.py) flushes the buffer on the checkpoint
  cadence; `_sync_tb_to_gcs` grew a `cleanup=False` mode for mid-training
  syncs.
- **MEDIUM (packaging)** — `google-cloud-aiplatform` / `google-cloud-storage`
  are imported across ~8 modules but were declared nowhere; the Docker image
  installed `[train,viz]` only, so in-container GCS-client fallbacks silently
  degraded. **FIXED** — new `gcp` extra in `pyproject.toml`, installed in the
  Dockerfile and included in `[all]`.
- **MEDIUM (artifacts)** — `upload_curriculum_artifacts` didn't upload
  `stage_config.json`, `metrics.json`, `evaluations.npz`, or
  `diagnostics.npz` — `metrics.json` is exactly what `sweep collect-results`
  consumes, so a curriculum run uploaded via this path couldn't be
  post-processed from GCS. **FIXED** (sidecars now uploaded; one shared GCS
  client per batch instead of one client per file).
- **LOW (bug)** — `scripts/setup_vertex_ai.sh` stripped a `gs://` prefix only
  in the bucket-creation prompt, not the job-submission prompt (yielding
  `/gcs/gs://bucket/...` output paths), and accepted arbitrary stage numbers.
  **FIXED** (strip + validation).
- **LOW (robustness)** — `results.py:_load_trial_metrics`'s GCS fallback only
  caught `ImportError/JSONDecodeError/OSError`; google-api-core errors
  (`Forbidden`, `DefaultCredentialsError`, …) crashed the whole collection
  run. **FIXED** (broad catch per trial).
- **LOW (logging)** — "HPT metric reported" logged even when
  `cloudml-hypertune` wasn't installed; "W&B run initialized." logged even
  when init returned `None`. **FIXED**.
- **LOW (docs)** — `vertex-ai.md` had two `## 9.` sections, and the Spot-VM
  example set only `restart_job_on_worker_restart=True` (which does **not**
  provision Spot capacity — users would pay on-demand prices believing they
  were on Spot). **FIXED** (renumbered; example now sets
  `scheduling_strategy=SPOT`).
- **LOW (docs)** — `state.py` docstring said state uploads to
  `_sweep_state.json`; the actual key is `_sweep_state_<algorithm>.json`.
  **FIXED**.

### Open

- **LOW** — `_is_retryable_gcp_error` (submit.py) treats the name
  `GoogleAPICallError` — the SDK's generic fallback class — as retryable.
  Harmless in practice (mapped permanent errors have specific subclasses),
  but worth a comment or removal.
- **LOW (artifacts)** — Vertex job submissions log the console URL but don't
  record the job resource name/URL into any durable artifact
  (`stage_config.json` / a submit-time manifest next to the results); see
  §4 "artifact opportunities".
- **LOW** — the JAX `TrainingCSVLogger` opens the CSV directly on the output
  path and flushes per update; on a `/gcs` FUSE mount that is one network
  write per update (the exact pattern `tb_sync` exists to avoid). Buffer
  locally when `_is_gcs_path(path)`. (Resume truncation is fixed — see §3.)

## 2. SB3 training stack (delta vs June)

### Fixed

- **HIGH (perf)** — `EvalCollapseEarlyStopCallback._on_step` re-opened and
  parsed `evaluations.npz` on **every training step** (~1.5M reads per 6M-step
  stage; a GCS FUSE network read on Vertex). **FIXED** — it now reads
  `EvalCallback.evaluations_results` in memory; no file I/O at all.
- **HIGH (perf, SAC)** — `DiagnosticsCallback` rewrote the entire
  `diagnostics.npz` history up to 3× per rollout-end; SAC fires rollout-end
  every `train_freq` steps, making this O(n²) I/O (hundreds of GB of
  cumulative writes on a 6M-step SAC stage). **FIXED** — one throttled save
  per rollout-end (default 60 s interval) + a final flush in
  `_on_training_end`, with a regression test.
- **MEDIUM (robustness)** — `train()` ran the ~80-episode post-training eval
  *before* saving the final model, and the `eval_policy` call was unguarded —
  a crash (or second Ctrl-C) during eval lost both the final checkpoint and
  `metrics.json`. **FIXED** — final save happens first; `eval_policy` is
  guarded so `metrics.json` is always written.
- **LOW (logging)** — `WandbCallback` gated logging with
  `num_timesteps % log_freq != 0`; `num_timesteps` advances by `n_envs`, so
  any `n_envs` not dividing `log_freq` logged rarely or never. **FIXED**
  (elapsed-steps gate).
- **LOW (metrics)** — `LocomotionMetrics` used `-1.0` as the
  "target never reached" sentinel for `time_to_target`; `aggregate_episodes`
  only filters non-finite values, so the sentinel was averaged into
  `mean_time_to_target`. **FIXED** — NaN sentinel (auto-filtered) + a new
  `target_reached` fraction; the eval CLI prints both.
- **LOW (perf)** — `RewardRampCallback` throttled updates by value
  quantisation, broadcasting an `env_method` RPC to every SubprocVecEnv
  worker every few dozen steps on long ramps despite a comment claiming
  "every 10k steps". **FIXED** (timestep-bucket gate).
- **LOW (cleanup)** — dead module-level numpy import in `train_base.py`;
  a comment now documents that the `algo_*` diagnostics series lag one
  rollout (SB3 fires `on_rollout_end` before that iteration's `train()`).

### Open

- **LOW** — `curriculum.py` `_read_latest_eval`: the
  `successes.shape[0] == n_evals` guard permanently discards npz successes if
  SB3 started recording them one eval late (silently falls back to the
  supplementary sample).
- **Artifact opportunities** (see §4): `train_curriculum` never calls
  `generate_stage_artifacts` (no videos/summaries on the CLI path), never
  runs the quality eval, and `CurriculumManager.summary()` — the full gate
  history — is computed but never persisted. `WandbCallback` is always built
  without a `video_env`, so its video recording is dead code, and
  `wandb_run.finish()` runs before final metrics exist.

## 3. JAX/MJX stack (delta vs June)

### Fixed

- **CRITICAL** — the JAX curriculum gate compared the trainer's **per-step**
  `mean_reward` (~0.5–2 for a good policy) against the TOML episode-level
  `min_avg_reward` (100.0), so `run_curriculum` still stopped after stage 1
  every time — the June fix corrected the config key but not the metric
  semantics. **FIXED** — `JaxTrainer.train` now reports
  `mean_episode_return` (tail of `episode_return_history`) and
  `check_stage_gate` gates on it, with regression tests using realistic
  trainer metrics.
- **HIGH** — `MJXDinoEnv.__init__` merged TOML kwargs **into the live species
  registry dict**, permanently baking one stage's config into every env
  constructed later in the process (e.g. trex stage-1 anti-locomotion
  penalties leaking into stage 2). **FIXED** (registry entry + its
  `reward_weights` dict are copied before merging).
- **HIGH** — trex registry still declared the legacy `bite_approach_weight`
  key, which (being un-canonicalized) shadowed the TOML-configured
  `approach_weight` in the step function's legacy-first lookup — trex JAX
  training always used approach weight 1.0 regardless of TOML. **FIXED**
  (registry key renamed to `approach_weight`; the lookup collapsed to the
  canonical key).
- **HIGH** — `run_curriculum` carried policy params across stages but reset
  observation-normalization stats each stage, feeding stage-2/3 policies
  arbitrarily re-scaled inputs. **FIXED** — `train_jax` accepts/returns
  `obs_stats` (now returns a 3-tuple) and the curriculum threads them
  through, mirroring the SB3 path's `obs_rms` carry.
- **HIGH** — the notebook/functional trainer path configured
  `vf_clip_range` (and printed it) but `scan_ppo_epochs` never passed it to
  `ppo_update`, so value clipping was silently off. **FIXED** (closure
  constant passed through).
- **MEDIUM (logging)** — `ppo_loss` never exposed the total loss
  (`jax.grad(has_aux=True)` discards it), so `CSVLoggingHook` wrote
  `total_loss=0.0` and `StabilityHook`'s loss watchdog could never fire.
  **FIXED** (`total_loss` added to the loss info dict).

### Fixed in the follow-up JAX pass (same branch)

- **HIGH (eval)** — `jax_setup.make_obs_fn` hardcoded
  `target_pos=jnp.zeros(3)`, so during `evaluate_policy_cpu` /
  `record_training_video` the policy was told the target is at the world
  origin while success detection used the real prey/food body — stage-3
  gates and videos evaluated a policy pursuing the wrong goal. **FIXED** —
  the obs function resolves the model's target body and feeds its position
  (the same body success detection uses); videos get this for free via the
  shared obs fn. (Remaining nuance: training randomizes the virtual target
  3–8 m ahead; eval uses the model's fixed target placement.)
- **MEDIUM (eval fidelity)** — `run_stage_evaluation`'s reward/termination
  config diverged from the training env. **FIXED** — `setup_species` merges
  the species-registry default weights under the TOML (same merge
  `MJXDinoEnv` performs); `evaluate_policy_cpu` now supplies per-step
  `target_pos` / `prev_target_distance` / `prev_action` / `forward_ref_2d` /
  `success_site_positions` to the reward closure, which also carries
  `forward_vel_max`, `dt`, `fall_penalty`, and the species' success bonus;
  and the TOML `nosedive_termination_threshold` reaches `EvalConfig`. The
  eval loop's `prev_action` is now updated *after* the reward call (it was
  compared against itself). Verified end-to-end with a CPU smoke run
  (stage-3 velociraptor reward changes when the parity kwargs engage).
- **MEDIUM (CLI fidelity)** — single-stage CLI dropped tuned TOML `[jax]`
  keys, and `vf_coef` was ignored on all CLI paths. **FIXED** — the CLI now
  passes `minibatch_size`, `warmup_*`, `ramp_*`, and `num_envs`; `train_jax`
  grew a `vf_coef` parameter wired into `PPOConfig`, the curriculum key map,
  and the CLI (trex's tuned 0.25 now applies).
- **MEDIUM (artifacts)** — the JAX CLI path saved nothing by default and
  `--curriculum` ignored `--checkpoint-dir`. **FIXED** — with
  `--checkpoint-dir`, headless runs now write a per-update training CSV
  (`<species>_s<N>_training_log.csv`), rotating + final checkpoints
  (params/obs_rms/opt_state via `CheckpointHook`), and a
  `<species>_s<N>_best.pkl` best-episode-return snapshot; the curriculum
  branch forwards the checkpoint dir to every stage (stage-prefixed
  filenames keep them apart).
- **MEDIUM (logging)** — JAX eval detected success but never recorded it.
  **FIXED** — `EvalResults.successes` / `mean_success_rate`, surfaced in
  `stage_results` and the eval summary, and `check_stage_gate` now enforces
  the TOML `min_success_rate` (stage 3) and `min_avg_forward_vel` (stage 2)
  thresholds — full gate parity with the SB3 `CurriculumManager`.
- **LOW** — `fall_rate` counted time-limit truncations as falls (both
  trainer paths now count natural terminations only); `TrainingCSVLogger`
  grew an `append` mode used on notebook resume (no more truncated logs);
  brachio's 4 foot sensors now split right/left by parity instead of
  lumping 3 of 4 feet as "left"; `jax_setup` docstrings corrected
  (no more `setup_training`, 4-tuple return documented);
  `except (ImportError, Exception)` in `jax_viz.py` replaced with
  `except Exception`; `jax.md` no longer documents the nonexistent
  `wandb_project` / `WandbHook` and now shows the real
  checkpoint/CSV/best-model artifacts and `restore_train_state` resume flow.

### Still open (JAX)

- **LOW** — eval evaluates against the model's fixed target placement
  rather than randomized target distances like training; consider sampling
  target positions per eval episode for distribution parity.
- **LOW** — two same-named `check_stage_gate` functions with different
  signatures (`jax_eval` — eval-results based, now 4-criteria — vs
  `jax_curriculum` — metrics-dict based); worth renaming one.
- **LOW** — per-step eval reward diagnostics (`diag_reward_components`)
  still decompose forward velocity in world-X rather than the
  agent-to-target frame the total reward now uses; display-only.

## 4. Sweep / reporting infrastructure (delta vs June)

### Fixed

- **HIGH** — `_reconnect_or_collect_partial` wrapped the entire reconnect
  flow (including the long `_wait_for_job` poll) in a blanket
  `except Exception`, swallowing `TimeoutError` (fire-and-forget/timeout
  resume flows) and `_SweepJobFailed` — the caller then submitted a
  **duplicate HPT job** while the original was still running, or replaced a
  failed job without collecting its partial trials (overwriting trial dirs).
  **FIXED** — only the initial lookup can fall through; a reconnect timeout
  exits cleanly (state already saved), and a failure during the poll routes
  through partial-trial recovery.
- **HIGH** — the Ray Tune path's curriculum gate read only the
  `best_mean_forward_vel` / `best_mean_success_rate` aliases, which the Ray
  results-DataFrame path never produces — every stage-2/3 Ray trial was
  written with `stage_passed=False`. Also `NaN` (ASHA-pruned trials)
  silently *passed* threshold comparisons. **FIXED** —
  `_evaluate_curriculum_gate` now accepts both alias sets and treats
  NaN as missing.
- **MEDIUM** — `ray_orchestration._quick_rank_trials` crashed with
  `AttributeError: species_name` (`SpeciesConfig`'s field is `species`).
  **FIXED**.
- **MEDIUM** — contradictory `n_envs` precedence: `launch` could never apply
  the search-space file's `n_envs` (argparse default 4 looked "explicitly
  set"), while `launch-all` let the file override an explicit CLI flag.
  **FIXED** — both parsers default `--n-envs` to `None` and resolve
  CLI > file > 4.
- **MEDIUM (drift)** — the June `success_keys` fix was applied to
  `velociraptor/scripts/train_sb3.py` but not to the duplicate definition in
  `species_registry.py` (used by the whole Ray path). **FIXED** (registry
  updated to `["strike_success"]`); consolidating the two sources of truth
  remains open.
- **LOW (reproducibility)** — since seeds vary per trial, sweep CSVs no
  longer contained enough information to reproduce a trial. **FIXED** — both
  `metrics.json` writers now record the effective `seed` plus run identity
  (`species`, `algorithm`, `library_version`), which the extra-key
  passthrough carries into collected CSVs.
- **LOW (artifacts)** — sweep graphs were uploaded to fixed GCS keys, so a
  later stage/algorithm sweep overwrote the previous graphs. **FIXED** (keys
  now prefixed with the CSV stem: `species_algorithm_stageN`).

### Open

- **MEDIUM (cleanup)** — `ray_orchestration.py` (759 lines) has **zero
  callers**: `ray_tune_sweep.ipynb` re-implements all of it inline, and the
  drift is already real (the `species_name` crash existed only in the module
  copy). Either wire the notebook to the module or delete the module.
- **LOW** — quality scoring silently scores different metric sets on
  different paths (`cost_of_transport` / `vel_consistency` are weighted in
  `configs/quality_scoring.toml` but exported only by the notebook path);
  one crashed trial missing a metric drops that metric for the entire set.
- **LOW** — `ray_search_space.load_resume_settings` reads a `gpu_model` key
  that `save_search_space` never writes; `seed=0` is unstorable (falsy
  check).
- **LOW (docs)** — `results/README.md` documents per-stage GIFs and a CSV
  schema that half the shipped result files don't match.

### Artifact opportunities (not implemented; recommended)

1. **Vertex-side sweep manifest** for parity with the Ray path's
   `save_search_space` snapshot: write the resolved search space, image URI,
   machine type, and per-stage settings to
   `gs://<bucket>/sweeps/<species>/stage<N>/` at submit time (include the
   HPT job resource name/console URL).
2. **`best_trial_config.json` per stage on the Vertex path** (the Ray path
   exports one) and a **model manifest** (`model_manifest.json`) next to
   each exported `best_model.zip` linking checkpoint → trial id,
   hyperparameters, seed, eval metrics, library version, VecNormalize file.
3. **Curriculum gate history**: dump `CurriculumManager.summary()` to
   `curriculum_state.json` after each stage; record the achieved gate
   metrics (not just thresholds) in `curriculum_results.csv`.
4. **CLI curriculum parity with the notebook**: call
   `generate_stage_artifacts` (video + summaries + plots) per stage in
   `train_curriculum`, and run the quality eval so `metrics.json` exists on
   that path too.
5. **W&B**: pass a render-capable eval env into `WandbCallback` (its video
   path is currently dead code), log final metrics before `finish()`, and
   upload `best_model.zip` + `best_model_vecnorm.pkl` as W&B Artifacts.
6. **Run identity in CSVs**: add sweep-run timestamps / `resume_run` to
   collected rows so results merged across resume cycles are
   distinguishable.

## 5. Jupyter notebooks

- **HIGH (bug)** — `ray_tune_sweep.ipynb`, "save search space" cell: shipped
  with a literal newline inside an f-string
  (`print(f"⏎Search space saved…")`) — a `SyntaxError` that crashes the cell
  (and with it, the search-space record-keeping step). **FIXED** (escaped to
  `\n`).
- **MEDIUM (drift risk)** — `ray_tune_sweep.ipynb` re-implements the whole
  `ray_orchestration.py` module inline (tuner setup, discover/rank, parallel
  eval, best-trial export). See §4 Open — the two copies have already
  diverged once.
- **LOW** — the notebook's `EVAL_EPISODES` knob does not affect the in-trial
  eval episode count (trials use `train_trial`'s default of 30) — a reader
  would expect it to.
- **LOW (reproducibility)** — all notebooks `pip install` unpinned latest
  `mujoco` / `stable-baselines3` / `jax` / `ray[tune]`; a breaking upstream
  release changes behavior between sessions. Consider pinning known-good
  versions in the install cells.
- **LOW (docs)** — markdown cells still reference `configs/sweep_ppo.json`
  (the code correctly resolves `configs/<species>/sweep_<algo>.json`).
- Otherwise healthy: every `from environments…` import in all four notebooks
  resolves against current code, and an AST pass over all calls into the
  shared library found no signature drift (`sb3_training`, `jax_training`,
  and `google_drive_summary` are clean; `google_drive_summary` correctly
  reads the `"curriculum"` key that `save_stage_config` writes).

## 6. Suggested follow-ups (priority order)

*(Items 1, 2, and 4 of the original list — the JAX eval target bug, JAX CLI
artifacts/success recording, and eval reward parity — were fixed in the
follow-up JAX pass on this branch; see §3.)*

1. Consolidate `ray_orchestration.py` with the notebook (§4/§5) and the
   duplicate `SpeciesConfig` definitions (registry vs `train_sb3.py`).
2. The artifact opportunities in §4 — the Vertex manifest and model manifest
   are small and make sweeps auditable end-to-end.
3. Remaining JAX polish (§3 "Still open"): randomized eval targets,
   `check_stage_gate` naming, world-X eval diagnostics.
