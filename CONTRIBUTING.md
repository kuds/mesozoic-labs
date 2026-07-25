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

> **Note:** The training and test scripts use shared base modules in
> `environments/shared/`. Species-specific scripts are thin wrappers around
> this shared infrastructure. See `docs/CODE_CONSOLIDATION.md` for the consolidation
> plan and architecture details.

1. **Create the directory structure:**
   ```
   environments/<species>/
   ├── __init__.py
   ├── assets/<species>.xml      # MuJoCo MJCF model
   ├── envs/
   │   ├── __init__.py
   │   └── <species>_env.py      # Gymnasium environment
   ├── scripts/
   │   ├── train_sb3.py          # Training script (wraps shared base)
   │   ├── view_model.py         # Model viewer
   │   └── test_env.py           # Quick env validation (wraps shared base)
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
   - Add the species to `environments/__init__.py` and
     `environments/shared/species_registry.py`

4. **Register the MJX plant** (`mjx_config.py`):
   - Call `register_species_mjx` with the sensor layout, root `body_ids`
     (`"pelvis"` for bipeds, `"torso"` for quadrupeds — the shared observation
     builder dispatches on this), termination heights, and stage-3 success
     sites
   - Add the species to the model-path and module maps in
     `environments/shared/mjx_env.py` and `environments/shared/jax_training.py`

5. **Add curriculum configs** (`configs/<species>/`):
   - Create `stage1_balance.toml`, `stage2_locomotion.toml`, `stage3_<behavior>.toml`
   - Follow the TOML structure from an existing species
   - Calibrate `reset_noise_scale` for stage 1 against
     `python environments/shared/scripts/zero_action_baseline.py <species> --sweep-noise`:
     a level at which a do-nothing policy reaches the full horizon in nearly
     every episode makes the stage reward a statue
   - Add `sweep_ppo.json` and `sweep_sac.json` if the species will be swept

6. **Write tests** (`tests/test_<species>_env.py`):
   - Use the shared test utilities in `environments/shared/`
   - Verify observation/action space shapes, reward components, determinism
   - Extend `environments/shared/tests/test_species_integration.py` and
     `test_mjcf_assets.py` with the new species' dimensions

7. **Declare the plant revisions** (`configs/plant_versions.toml`):
   - Add a `[plants.<species>]` block starting every revision at 1, with the
     species' `observation_schema`
   - Run `python -m environments.shared.plant_contract --write` to regenerate
     `configs/plant_manifest.generated.json`. If a shared interface function
     changed, the generator will refuse until the affected species' revisions
     are bumped

8. **Add the public catalog entry** (`configs/species_manifest.toml`):
   - Add presentation metadata, the environment entry point, and the MJCF path
   - Declare success semantics for every supported backend and the applicable
     training-notebook IDs
   - Declare only existing, provenance-labelled result summaries or stage videos
   - Run `python -m environments.shared.species_catalog` to regenerate the
     README blocks and `website/src/data/species.generated.json`
   - Run `python -m environments.shared.species_catalog --check` to verify that
     generated data and declared artifacts are current

9. **Update CI** (`.github/workflows/python-ci.yml`):
   - Add a `test-<species>` job following the existing pattern

10. **Update pyproject.toml**:
    - Add the test path to `[tool.pytest.ini_options]`
    - Add the Gymnasium entry point

11. **Add the documentation** (`environments/<species>/README.md`,
    `website/docs/models/<species>.mdx`):
    - Link the new model page from `website/sidebars.ts`, `website/docs/intro.md`,
      and export the species from `website/src/data/species.ts`

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
