# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] — Reproducible Runs & Velociraptor Stage-1 Diagnosis (v0.3.3)

### Changed
- **Velociraptor knee actuators raised to 1.5×kp** (breaking — plant change,
  physics revision 1 → 2): the knee measured 0 % clip at the moderate
  2.5 Hz/0.8-amplitude contract gait but 30–46 % at sprint-like 3–4 Hz
  full-amplitude excitation while still capped at 0.8×kp (`forcerange`
  ±145 on kp=180) — the same clipped-torque signature that collapsed
  stage-2 locomotion at the hips. At ±270 the same sprint excitation
  measures 0 % knee clip; the moderate-regime contract and the
  home-control no-saturation guarantee are unchanged, and a new
  sprint-regime contract test pins the knee headroom.

- **Eval-collapse early-stop is now variance-robust, gate-floored, and
  loosened for stage 1**: the backstop's peak is the per-evaluation robust
  score (mean − std) instead of the raw mean — run `20260720_203454` was
  aborted at 1.1M/6M steps because a single 50k evaluation of
  261.79 ± 261.72 (robust score 0.07) set a raw-mean kill threshold that
  every later normal-dip evaluation "violated" — and detection only arms
  once the robust peak clears the stage's `min_avg_reward` curriculum gate
  (overridable via `collapse_peak_floor`), so the pre-convergence grind
  can never register as a collapse. All three stage-1 TOMLs also gain the
  explicit `collapse_min_evals = 20` / `collapse_patience = 10` /
  `collapse_drop_fraction = 0.5` overrides stage 2 already had: run
  `20260721_004731`'s healthy 2.2–2.3M transition turbulence accumulated
  2 of 8 kill-drops under the old stage-1 defaults.

### Fixed
- **`metrics.json` now describes the promoted checkpoint**: the post-stage
  quality/velocity/success evaluation loaded SB3's mean-reward
  `best_model` while next-stage training loads `robust_best_model` when
  present, so the sidecar could describe a policy the curriculum never
  promoted (in run `20260720_203454`, the degenerate lucky-peak 50k
  checkpoint). Both paths now share one selection helper
  (`robust_best_model` → `best_model`, each only with its matched
  VecNormalize stats), and the sidecar records which checkpoint it
  evaluated as `quality_eval_checkpoint`.
- **Velociraptor foot touch sensors were dead during normal stance**
  (breaking — policy-interface revision 2 → 3, visual revision 1 → 2): the
  raptor is digitigrade and a lying toe capsule contacts the floor near its
  ends, but the r=0.02 touch sites sat at the toe midpoint, so both sensors
  — and the two foot-contact observation dims fed from them — read 0 while
  standing (verified: six active floor contacts, zero sensor signal). The
  sites now envelop the whole toe_d3 capsule. Fixing them exposed a second
  defect: the adjacent toe digits (d3/d4) interpenetrate at rest and the
  solver held a constant ~300 N phantom repulsion between them, which the
  enlarged sites would have summed even airborne; new contact excludes
  remove it. Verified post-fix: ~37 N per foot at settled stance, exactly
  0 N airborne, 48/50 standing-probe survival unchanged, and regression
  tests pin stance/airborne sensor behavior. The T-Rex has the same
  dead-sensor defect (recorded in KNOWN_ISSUES, not yet fixed).
  Follow-up (visual revision 2 → 3): the enlarged sites rendered as large
  gray balls on the feet in run videos (first visible in run
  `20260721_141523`'s stage-1 video); they now live in site group 4,
  which default rendering hides, so they no longer draw — toggle site
  group 4 in an interactive viewer to visualize the touch volumes.

### Added
- **Canonical result bundles for Colab/Google Drive training**: schema-v2
  summaries, runtime-captured provenance, immutable completed bundles,
  selected/terminal evaluation evidence, plant/config/model/VecNormalize
  hashes, partial-versus-promotion validation, idempotent manifests, and
  strict exporter tests now make failed runs auditable without presenting
  them as publishable results.
- **Velociraptor Stage-1 basin investigation**: records the pre-training
  50-seed physics probe, Commit A's A-only PPO run (`20260720_203454`), the
  natural-lean diagnosis, Commit B's reward geometry, targeted backend-routing
  lessons, and the falsifiable next-run plan.

### Changed
- **Velociraptor actions are residuals around the XML `home` controls**:
  `-1`/`0`/`+1` map to actuator minimum/home/maximum through a piecewise-linear
  transform. This is a breaking policy-interface revision (1 → 2); older PPO,
  SAC, and JAX checkpoints remain historical and cannot be resumed as current
  policies. XML physics and actuator reach are unchanged.
- **Velociraptor posture shaping targets its natural forward pitch** (0.35
  rad) instead of world vertical. The yaw-invariant, direction- and roll-aware
  squared-chord formulation keeps finite JAX gradients at the exact target;
  T-Rex and Brachiosaurus retain vertical posture targets.

### Fixed
- **Targeted natural-posture consistency across training and evaluation
  paths**: the shared NumPy/JAX primitive, direct MJX, JAX total and detailed
  reward paths, CPU evaluation, diagnostics, and runtime `natural_pitch`
  overrides resolve the same reward-only target. Absolute-tilt termination and
  the existing nosedive boundary remain unchanged, and the JAX notebook now
  binds reward functions only after environment creation. Comprehensive
  fixed-state, all-component SB3↔JAX parity remains open.

## [Unreleased] — RL/GCP Review Fixes & Model Physics (v0.3.2)

### Changed
- **Plateau diagnostics now follow deterministic stage-gate evaluations**:
  rollout return warnings (which fired at different rates for PPO and SAC and
  repeated every rollout) are replaced by a warn-once/re-arm state machine.
  It monitors the currently blocking configured gate — balance duration,
  forward velocity, task success, then reward — and explains that training
  continues independently. SB3 evaluation now reports task success as `N/A`
  when the current stage has no success-rate gate, while preserving genuine
  Stage 3 `0%` results and success arrays.
- **Trex and brachiosaurus gait actuator headroom raised to 1.5×kp**
  (breaking — plant change): the same static-only forcerange sizing that
  broke velociraptor stage-2 measured 44–50 % hip / ~31 % ankle / 10–14 %
  knee gait-cycle clipping on trex (2.5 Hz) and 20–33 % hip clipping on
  brachiosaurus *at a 1.5 Hz walk*. Raised the clipped groups only (trex
  hip pitch/knee/ankle; brachio's four hip pitches) — post-fix clipping
  ≤0.5 %, root-z divergence vs an unbounded plant ≤2 mm. New
  `test_actuator_bounds.py` for both species pins the contract, built on
  shared helpers (`environments/shared/tests/actuator_bounds_helpers.py`);
  the velociraptor test now uses the same helpers. A species-generic
  saturation report lives at
  `environments/shared/scripts/actuator_saturation_report.py` (the
  velociraptor script delegates to it). Policies trained on the old caps
  will not transfer.
- **Entropy decay enabled for velociraptor stage 1** (`ent_coef_end =
  0.001`): run `20260711_235303` — which cleared stages 1–3 and set the
  stage-2 record (2705.93 ± 7.84 @ 2.75 m/s) — showed stage 1 destabilizing
  late (peak 1524 @ 2.45M → 385 @ 3.95M, bimodal falls) under constant
  `ent_coef`, the same pathology entropy decay fixed in stage 2. Staged
  (commented) `ent_coef_end` suggestions added to the trex and
  brachiosaurus stage 1/2 TOMLs for their next runs.
- **Raptor hip-pitch/ankle actuator headroom raised to 1.5×kp** (breaking —
  plant change): the blanket 0.8×kp `forcerange` from the July 9 model
  hardening clipped 34–40 % of hip and 22–25 % of ankle torque during a
  moderate gait cycle and collapsed stage-2 locomotion twice (runs
  `20260709_185946` and `20260711_165924`, bitwise-identical failures). At
  1.5×kp the same excitation measures 0 % clipping and zero pelvis-z
  divergence from an unbounded plant; `test_actuator_bounds.py` now pins
  the per-group ratios and asserts the gait regime stays *unclipped*.
  Velociraptor policies trained on the 0.8×kp plant will not transfer.
- **Velociraptor stage-2 config hardened against the fragile-fast-gait
  collapse**: `ent_coef_end = 0.001` (entropy decay — action std grew
  1.18→1.49 in both collapses), `forward_vel_max` 3.0 → 2.5 (saturate the
  speed incentive just past the 2.0 m/s gate), `fall_penalty` −50 → −150
  (late-fall hedge; the corrected break-even math is in
  `docs/investigations/STAGE2_RECOMMENDATIONS.md` §2.1).
- **Docs reorganized**: dated run analyses and reward investigations moved
  to `docs/investigations/` (TRAINING_REVIEW, TRAINING_REVIEW_JAX_STAGE1,
  REWARD_DISCREPANCY_INVESTIGATION, REWARD_SCALE_REDESIGN,
  STAGE2_INVESTIGATION, STAGE2_RECOMMENDATIONS); `docs/README.md` now maps
  the four doc categories (living reference / plans / investigations /
  reviews) and their lifecycles.
- **Python 3.11+ required; Python 3.13 supported** (breaking): dropped
  Python 3.10 support (`requires-python >= 3.11`), added 3.13 to the
  trove classifiers and the CI test matrix (3.11 + 3.13), and bumped the
  Docker base image to `python:3.13-slim`.  The `tomli` conditional
  dependency and import fallbacks are gone — stdlib `tomllib` is always
  available on 3.11+.  Ruff now targets py311; mypy targets 3.12 because
  numpy ≥ 2.3 stubs use PEP 695 syntax that mypy rejects for older targets.
- **Homepage light mode actually renders light**: the landing page palette
  (`index.module.css`) gained a `[data-theme='light']` variant — until now
  every section was hardcoded dark, so toggling the theme changed almost
  nothing.  Accent colors are darkened for WCAG AA contrast on light
  surfaces, the curriculum/feature icon colors follow the theme, and the
  forced-dark homepage navbar override (plus its logo-swap hack) in
  `custom.css` is removed.  The footer now follows the theme in both modes
  (Infima's `.footer--dark` element-level variables had silently overridden
  the intended colors, so it rendered `#303846` even in dark mode).  The
  code terminal mock-up intentionally stays dark in both themes.

### Fixed
- **Trex/brachio stage TOMLs: restored `[ppo.policy_kwargs]` headers lost
  in the entropy-decay staging** (PR #436): the dropped headers put
  `net_arch` into the top-level `[ppo]` table, crashing `PPO.__init__()`
  at stage start (trex run `20260712_185931`). New config regression
  tests now pin the structure for every species/stage: `net_arch` only
  under `policy_kwargs`, `[ppo]` keys validated against the PPO
  constructor signature plus harness schedule keys, and `[env]` keys
  validated against each species' env constructor signature.
- **JAX pipeline correctness** (July 2026 review, PR #426): curriculum gate
  now compares episode-level return against the TOML threshold (it compared
  per-step reward and never advanced past stage 1); observation-normalization
  stats carry across curriculum stages; `MJXDinoEnv` no longer mutates the
  live species registry (stage config leaked across stages); trex TOML
  `approach_weight` was shadowed by a legacy registry key; `vf_clip_range`
  now reaches the fused PPO update; CPU evaluation observes the real target
  body (was hardcoded to the world origin) and its reward includes the same
  approach/proximity/success/fall components as training; eval records
  per-episode success and gates on TOML `min_avg_forward_vel` /
  `min_success_rate`
- **SB3 pipeline**: eval-collapse callback re-read `evaluations.npz` every
  training step (a GCS network read on Vertex); `diagnostics.npz` was fully
  rewritten up to 3× per rollout-end (O(n²) I/O, severe for SAC); final
  model now saved before the post-training eval; `time_to_target` no longer
  averages a `-1` sentinel into the mean
- **Sweeps / Google Cloud**: sweep resume no longer submits duplicate Vertex
  HPT jobs after a swallowed timeout; Ray Tune stage-2/3 curriculum gates no
  longer fail unconditionally (metric-alias mismatch; NaN treated as
  missing); `--wandb` with a missing API key no longer kills headless
  training and relaunches resume the same W&B run; TensorBoard events sync
  to GCS periodically (spot-VM preemption no longer loses a stage's logs);
  `_wait_for_job` honors `--stage-timeout` during persistent API failures;
  effective per-trial seed and run identity recorded in `metrics.json`
- `ray_tune_sweep.ipynb` shipped with a `SyntaxError` in the
  save-search-space cell
- Broken sweep commands in `vertex-ai.md`; stale `configs/sweep_*.json`
  paths; nonexistent `wandb_project`/`WandbHook` in `jax.md`

### Changed
- **Brachiosaurus model physics (breaking for trained policies)**: leg
  springs now reference stance angles with stronger, force-bounded servos —
  the model can now statically stand at torso z≈1.13 (previously collapsed
  to z≈0.70, below the z=1.0 alive floor, even when commanded straight).
  Pre-existing brachiosaurus checkpoints will not transfer.
- All three models use `integrator="implicitfast"` and bounded `forcerange`
  on position actuators (raptor/trex bounds sized to leave settle behavior
  unchanged)
- Headless JAX runs with `--checkpoint-dir` now produce a per-update
  training CSV, rotating + final checkpoints, and a best-episode-return
  snapshot; `--curriculum` forwards the checkpoint dir per stage; the
  single-stage CLI passes previously-dropped TOML `[jax]` keys
  (`minibatch_size`, `warmup_*`, `ramp_*`, `num_envs`, `vf_coef`)
- New `gcp` extra (`google-cloud-aiplatform`, `google-cloud-storage`),
  installed in the Docker image
- Code reviews consolidated: `docs/KNOWN_ISSUES.md` is the living list of
  open findings; dated reviews archived under `docs/reviews/`

## [Unreleased] — Codebase Consolidation & Training Results (v0.3.0)

### Added
- Velociraptor SAC training results — all 3 stages passed (90.0% strike success, 22M steps, 22:59:18)
- Brachiosaurus PPO training results — Stages 1-2 passed, Stage 3 (food_reach) at 16.7% success (target: 50%)
- Updated training review with Brachiosaurus Stage 2/3 analysis and Velociraptor SAC comparison
- `environments/shared/train_base.py` — shared training logic with `SpeciesConfig` dataclass (~1,100 lines), reducing each species' `train_sb3.py` from ~950 lines to ~44 lines
- `environments/shared/test_env_base.py` — shared test utilities for environment validation (~214 lines)
- `environments/shared/constants.py` — centralized simulation-wide constants (sensor layout, VecNormalize defaults, physics defaults)
- `environments/shared/tests/reward_test_helpers.py` — reusable reward assertion functions for cross-species test consistency
- Expanded T-Rex reward tests (nosedive, height, heading, spin, drift, backward velocity)
- Expanded Brachiosaurus reward tests (gait instability, speed penalty, food reach threshold)
- Consolidated training notebooks into single parameterized `notebooks/training.ipynb`
- Algorithm-specific training diagnostics persisted to `diagnostics.npz` for **both PPO and SAC** (previously only on TensorBoard): captured SB3 `train/*` metrics (PPO `clip_fraction`/`approx_kl`/`explained_variance`/`entropy_loss`, SAC `critic_loss`/`actor_loss`/`ent_coef`) under `algo_*` keys
- `diagnostics/action_saturation` and `diagnostics/action_abs_max` — fraction of action components pinned at the control limit and the peak magnitude, computed for both PPO and SAC (SAC previously had no action logging at all)
- `diagnostics/grad_norm` — PPO post-update (clipped) gradient norm, surfacing when the clip ceiling binds or the policy stalls
- `explained_variance` in the JAX PPO loss info dict, per-update CSV log, and console output (value-function fit diagnostic)

### Changed
- Species training scripts (`train_sb3.py`) are now thin wrappers around shared `train_base.py`
- `DiagnosticsCallback` plateau detection now averages every episode that completed during a rollout (accumulated per-step) instead of only episodes that happened to end on the rollout's final step — the old sampling was biased and sparse
- JAX PPO clamps the Gaussian policy's log-std to `[-5, 2]` (applied identically in `sample_action` and `ppo_loss`, preserving the importance ratio) so a collapsing/exploding policy can no longer drive the action std to 0/inf and NaN the log-prob and entropy terms
- Species test scripts use shared `test_env_base.py` utilities
- `BaseDinoEnv` now provides concrete helper methods for common reward computations, gait symmetry, and termination checks
- Raptor, T-Rex, and Brachio reward tests refactored to use shared helpers
- `CookieResetButton` now uses shared `resetConsentStatus()` instead of duplicating localStorage logic

### Removed
- ~3,000+ lines of duplicated code across training scripts, test utilities, and environment methods
- Unused `getConsentStatus()` export from CookieConsent component

## [0.2.0] - 2026-02-09

### Added
- TOML-based configuration for curriculum stage reward weights and hyperparameters (`configs/` directory)
- Config loader utility (`environments/shared/config.py`) with `load_stage_config()` and `load_all_stages()`
- Gymnasium namespace registration (`MesozoicLabs/Raptor-v0`, `MesozoicLabs/Brachio-v0`, `MesozoicLabs/TRex-v0`)
- Pre-commit hooks with Ruff (format + lint), mypy, and standard checks
- `pytest-cov` for test coverage reporting with 70% threshold
- `CONTRIBUTING.md` with development workflow guidelines
- `CHANGELOG.md` (this file)
- GitHub issue templates (bug report, feature request, new species proposal)
- Reward function unit tests and curriculum stage transition tests
- Package metadata in `pyproject.toml` (authors, license, classifiers, URLs)
- `mypy` and `ruff` configuration in `pyproject.toml`
- `CurriculumManager` class for automated multi-stage training (`environments/shared/curriculum.py`)
- `CurriculumCallback` SB3 callback that monitors evaluation and stops training when advancement thresholds are met
- `thresholds_from_configs()` helper to extract curriculum thresholds from TOML configs
- `[curriculum]` section in all TOML stage configs with `timesteps`, `min_avg_reward`, `min_avg_episode_length`, and `required_consecutive` fields
- `curriculum` subcommand in all three species' training scripts for automated end-to-end 3-stage training
- `LocomotionMetrics` class with gait symmetry, cost of transport, stride frequency, and time-to-target (`environments/shared/metrics.py`)
- `WandbCallback` for SB3 with per-component reward logging and config snapshots (`environments/shared/wandb_integration.py`)
- `WandbCallback` video recording of evaluation episodes (`video_env`, `video_freq` parameters)
- `wandb` added to `[train]` optional dependencies

### Changed
- Training scripts now load stage configs from TOML files instead of hardcoded dictionaries
- Config loader now parses `[curriculum]` section from TOML files into `curriculum_kwargs`
- Bumped version from 0.1.0 to 0.2.0
- Dev dependencies expanded: `pytest-cov`, `mypy`, `ruff`, `pre-commit`
- All `print()` calls in training scripts replaced with `logging` module
- Gymnasium environments auto-register on `import environments` (no longer requires `register_all()`)

## [0.1.0] - 2025-01-01

### Added
- Initial release
- Velociraptor bipedal locomotion environment with sickle claw strike
- Brachiosaurus quadrupedal locomotion environment with food reaching
- T-Rex bipedal locomotion environment with jaw bite
- 3-stage curriculum learning (balance, locomotion, behavior)
- PPO and SAC training via Stable-Baselines3
- MuJoCo MJCF models for all three species
- Colab-ready training notebooks
- Docusaurus documentation site
- GitHub Actions CI with per-species test jobs
