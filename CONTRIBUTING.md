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

CI runs mypy twice: in the lint job, without Stable-Baselines3 or torch, and
in the SB3 job, with both installed (`stable-baselines3[extra]==2.9.0` and the
CPU `torch==2.13.0` wheel, on Python 3.12), where it must report no errors. The
SB3 job is the authority. The `.[all]` install above only approximates it: it
leaves SB3 and torch unpinned and on Python 3.11 resolves an older numpy, and
either can change what mypy reports. To reproduce the SB3
job's check, use Python 3.12 and a fresh environment:

```bash
pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.13.0"
pip install -e ".[train,test,viz]" "stable-baselines3[extra]==2.9.0" "mypy==2.3.1"
mypy environments/ --ignore-missing-imports
```

Pre-commit hooks run both automatically on `git commit`. The `dev` extra, the
hooks and CI pin one ruff version and one mypy version
(`environments/shared/tests/test_ci_tool_pins.py` keeps them in agreement),
and the ruff hooks cover `environments/`, as CI's ruff does. No hook touches
the digest data files (the plant MJCF sources and meshes, the recipe TOMLs, the
plant manifests, `plant_versions.toml` and the recovery calibrations): a
whitespace fix there would move a digest. The Python modules whose bytes a
digest hashes stay under the hooks; CI's pinned ruff keeps them from changing,
and any edit to them moves the behavior identities anyway. The one exception is
the frozen MJX interface core (decision D-D17: `mjx_env.py`, `jax_setup.py`,
`mjx_utils.py` and `obs_functions.py` in `environments/shared/`, and the four
`mjx_config.py` registrations): four species' policy-interface digests hash its
function tokens, so `pyproject.toml`'s `[tool.ruff]` `extend-exclude`, with
`force-exclude`, keeps every ruff run off it (the hooks, CI, and a file named
on the command line), and `test_plant_contract_frozen_mjx.py` pins the list.
Never edit those files.

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

4. **Declare the species SB3-only** (`envs/<species>_env.py`):
   - Set the class attribute `supported_training_backends = ("stable-baselines3",)`,
     as `CompsognathusEnv` does. Stable-Baselines3 is the only training backend
     (decision D-D17); without the attribute the plant contract treats the
     species as dual-backend and fails with "cannot import MJX plant
     registration"
   - Do not add an `mjx_config.py`: the four that exist belong to the frozen
     MJX interface core, which the policy-interface digests of T-Rex,
     Velociraptor, Brachiosaurus and Dibothrosuchus hash and which is never
     edited or extended (`docs/PLANT_CONTRACT.md`, "Backend parity and runtime
     binding")

5. **Add curriculum configs** (`configs/<species>/`):
   - Create `stage1_balance.toml`, `stage2_locomotion.toml`, `stage3_<behavior>.toml`
   - Follow the TOML structure from an existing species
   - Calibrate `reset_noise_scale` for stage 1 against
     `python environments/shared/scripts/zero_action_baseline.py <species> --sweep-noise`:
     a level at which a do-nothing policy reaches the full horizon in nearly
     every episode makes the stage reward a statue

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
   - Set `training_backends = ["stable-baselines3"]` and
     `training_notebooks = ["sb3_training"]`; `plant_contract/manifest.py`
     raises when `training_backends` differs from the environment's
     `supported_training_backends` (step 4)
   - Declare success semantics for `stable-baselines3`
   - Declare only existing, provenance-labelled result summaries or stage videos
   - Run `python -m environments.shared.species_catalog` to regenerate the
     README blocks and `website/src/data/species.generated.json`
   - Run `python -m environments.shared.species_catalog --check` to verify that
     generated data and declared artifacts are current
   - Add the species to `REWARD_SPECIES` in
     `environments/shared/harnesses/digest_snapshot.py` (its roll amplitude and
     the MJCF element its target sits on); without it the reward section
     prints ERROR lines and `--write` refuses; also add its behavior stage's
     success reason to `_SUCCESS` and the termination reasons its state probes
     reach to `_POSE_REASONS` in
     `environments/shared/tests/test_plant_contract_digest_snapshot.py`
   - Once its stages and behavior recipes are in place, run
     `python -m environments.shared.harnesses.digest_snapshot --block-optional-backends --write`
     to add the species' lines to `configs/digest_snapshot.generated.txt`

9. **Update CI** (`.github/workflows/python-ci.yml`):
   - Add a `test-<species>` job following the existing pattern

10. **Update pyproject.toml**:
    - Add the test path to `[tool.pytest.ini_options]`

11. **Add the documentation** (`environments/<species>/README.md`,
    `website/docs/models/<species>.mdx`):
    - Link the new model page from `website/sidebars.ts`, `website/docs/intro.md`,
      and export the species from `website/src/data/species.ts`

## Pull Request Process

1. Create a feature branch from `main`
2. Make your changes with clear, focused commits
3. Ensure all tests pass and pre-commit hooks are clean
4. If the change moves a digest on purpose (a plant, the policy interface, a stage
   config, a recovery calibration, a behavior recipe, the environment code it
   hashes, or reward, info or termination code), regenerate the digest snapshot from the repository root with the
   canonical MuJoCo (`configs/plant_versions.toml`) and commit
   `configs/digest_snapshot.generated.txt` in the same PR:
   `python -m environments.shared.harnesses.digest_snapshot --block-optional-backends --write`.
   CI's plant-contract job fails on any digest line that moved without it; a
   change that claims to move no digest leaves the file unchanged. A refactor
   that claims to move no number also runs the harness's `--exact` on the base
   and the head, on one machine, and shows an empty `diff` (see its `--help`)
5. Open a PR with a description of what changed and why
6. Link any related issues

## Reporting Issues

Use the GitHub issue templates:
- **Bug Report**: For environment crashes, training failures, or incorrect behavior
- **Feature Request**: For new capabilities or improvements
- **New Species**: For proposing a new dinosaur species
