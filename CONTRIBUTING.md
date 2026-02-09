# Contributing to Mesozoic Labs

Thanks for your interest in contributing! This document covers the development
workflow, code standards, and how to submit changes.

## Development Setup

```bash
git clone https://github.com/kuds/mesozoic-labs.git
cd mesozoic-labs

python -m venv venv
source venv/bin/activate

# Install with all development dependencies
pip install -e ".[all]"

# Install pre-commit hooks
pre-commit install
```

## Code Style

We use **Ruff** for linting and formatting (configured in `pyproject.toml`):

```bash
# Check for issues
ruff check environments/

# Auto-fix issues
ruff check --fix environments/

# Format code
ruff format environments/
```

We use **mypy** for static type checking:

```bash
mypy environments/
```

Pre-commit hooks run both automatically on `git commit`.

## Running Tests

```bash
# Run all tests
pytest

# Run tests for a specific species
pytest environments/velociraptor/tests/ -v

# Run with coverage
pytest --cov=environments --cov-report=term-missing
```

All tests must pass before submitting a PR. We target 70%+ code coverage.

## Adding a New Species

The project is designed to make adding new dinosaur species straightforward.
Follow this checklist:

1. **Create the directory structure:**
   ```
   environments/<species>/
   ├── __init__.py
   ├── assets/<species>.xml      # MuJoCo MJCF model
   ├── envs/
   │   ├── __init__.py
   │   └── <species>_env.py      # Gymnasium environment
   ├── scripts/
   │   ├── train_sb3.py          # Training script
   │   ├── view_model.py         # Model viewer
   │   └── test_env.py           # Quick env validation
   └── tests/
       ├── __init__.py
       ├── conftest.py            # Copy from existing species
       └── test_<species>_env.py  # Pytest suite
   ```

2. **Create the MJCF model** (`assets/<species>.xml`):
   - Define the body hierarchy, joints, actuators, and sensors
   - Include a mocap body for the prey/food target
   - Add touch sensors on feet and relevant contact geoms

3. **Implement the environment** (`envs/<species>_env.py`):
   - Subclass `BaseDinoEnv` from `environments.shared.base_env`
   - Implement the five abstract methods: `_cache_ids`, `_get_obs`,
     `_get_reward_info`, `_is_terminated`, `_spawn_target`
   - Register with Gymnasium using the `MesozoicLabs/<Species>-v0` namespace

4. **Add curriculum configs** (`configs/<species>/`):
   - Create `stage1_balance.toml`, `stage2_locomotion.toml`, `stage3_<behavior>.toml`
   - Follow the TOML structure from an existing species

5. **Write tests** (`tests/test_<species>_env.py`):
   - Copy the test structure from an existing species
   - Verify observation/action space shapes, reward components, determinism

6. **Update CI** (`.github/workflows/python-ci.yml`):
   - Add a `test-<species>` job following the existing pattern

7. **Update pyproject.toml**:
   - Add the test path to `[tool.pytest.ini_options]`
   - Add the Gymnasium entry point

## Pull Request Process

1. Create a feature branch from `main`
2. Make your changes with clear, focused commits
3. Ensure all tests pass and pre-commit hooks are clean
4. Open a PR with a description of what changed and why
5. Link any related issues

## Reporting Issues

Use the GitHub issue templates:
- **Bug Report**: For environment crashes, training failures, or incorrect behavior
- **Feature Request**: For new capabilities or improvements
- **New Species**: For proposing a new dinosaur species
