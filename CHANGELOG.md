# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] — Codebase Consolidation (v0.3.0)

### Added
- `environments/shared/train_base.py` — shared training logic with `SpeciesConfig` dataclass (~1,100 lines), reducing each species' `train_sb3.py` from ~950 lines to ~44 lines
- `environments/shared/test_env_base.py` — shared test utilities for environment validation (~214 lines)
- `environments/shared/constants.py` — centralized simulation-wide constants (sensor layout, VecNormalize defaults, physics defaults)
- `environments/shared/tests/reward_test_helpers.py` — reusable reward assertion functions for cross-species test consistency
- Expanded T-Rex reward tests (nosedive, height, heading, spin, drift, backward velocity)
- Expanded Brachiosaurus reward tests (gait instability, speed penalty, food reach threshold)
- Consolidated training notebooks into single parameterized `notebooks/training.ipynb`

### Changed
- Species training scripts (`train_sb3.py`) are now thin wrappers around shared `train_base.py`
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
