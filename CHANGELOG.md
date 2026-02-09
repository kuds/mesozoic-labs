# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - Unreleased

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
- `LocomotionMetrics` class with gait symmetry, cost of transport, stride frequency, and time-to-target (`environments/shared/metrics.py`)
- `WandbCallback` for SB3 with per-component reward logging and config snapshots (`environments/shared/wandb_integration.py`)
- `wandb` added to `[train]` optional dependencies

### Changed
- Training scripts now load stage configs from TOML files instead of hardcoded dictionaries
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
