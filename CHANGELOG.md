# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] — RL/GCP Review Fixes & Model Physics (v0.3.2)

### Changed
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
