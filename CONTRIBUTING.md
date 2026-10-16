# Contributing to Mesozoic Labs

Thanks for your interest in contributing! This document covers the development
workflow, code standards, and how to submit changes.

## Repository Layout

```
mesozoic-labs/
├── environments/
│   ├── <species>/        # one package per species, named by its binomial (docs/SPECIES_NAMING.md)
│   │   ├── base_env.py   # the species base class <Species>Env, shared by its embodiments
│   │   ├── <embodiment>/ # one model: anatomical, or a robot of the species
│   │   │   ├── env.py    # <Species><Embodiment>Env and the embodiment's body data
│   │   │   └── assets/   # MuJoCo MJCF model <species>_<embodiment>.xml
│   │   ├── scripts/      # view_model.py and species tools
│   │   ├── tests/
│   │   └── README.md     # with a Models table, one column per embodiment
│   └── shared/           # base environment, training CLI, stage manifests, curriculum and gates,
│                         # plant contract, result bundles, species catalog generator, harnesses
│                         # and the shared tests
├── configs/              # species_manifest.toml, the plant-contract, certification and digest
│                         # files, and configs/<species>/<embodiment>/ per model with its
│                         # stages.toml, stage TOMLs and behavior recipes
├── notebooks/            # the training and Drive summary notebooks
├── results/              # curated run summaries and GIFs (see results/README.md)
├── docs/                 # plans, records, living reference and the generated species catalog
│                         # (map: docs/README.md)
└── website/              # the documentation site (Docusaurus)
```

A model is the pair (species, embodiment): `environments/<species>/` holds
species-level material and `environments/<species>/<embodiment>/` holds one
model. Every identifier is derived from the pair by the one rule of
[docs/SPECIES_NAMING.md](docs/SPECIES_NAMING.md).

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
whitespace fix there would move a digest. Every Python module stays under the
hooks: no digest hashes a Python file's bytes, and the policy-interface digests
hash the tokens of named functions only (not comments, docstrings or blank
lines), which CI's pinned ruff keeps from changing. The one exception is
the frozen MJX interface core (decision D-D17: `mjx_env.py`, `jax_setup.py`,
`mjx_utils.py` and `obs_functions.py` in `environments/shared/`, and the two
`<species>/anatomical/mjx_config.py` registrations): two models' policy-interface digests hash its
function tokens, so `pyproject.toml`'s `[tool.ruff]` `extend-exclude`, with
`force-exclude`, keeps every ruff run off it (the hooks, CI, and a file named
on the command line), and `test_plant_contract_frozen_mjx.py` pins the list.
Never edit those files.

## Running Tests

```bash
# Run all tests
pytest

# Run tests for a specific species
pytest environments/velociraptor_mongoliensis/tests/ -v

# Run with coverage
pytest --cov=environments --cov-report=term-missing
```

After changing a model, a stage config, a manifest entry, a notebook path, a
video or a result summary, regenerate the generated public data (the root
README's species and notebook tables, `docs/SPECIES_CATALOG.md` and
`website/src/data/species.generated.json`) and verify it, as CI does:

```bash
python -m environments.shared.species_catalog
python -m environments.shared.species_catalog --check
```

All tests must pass before submitting a PR. We target 70%+ code coverage;
CI's `coverage` job gates the union of every test job at `fail_under = 70`.
Coverage counts all of `environments/` except tests and the hand-run or
one-off files that `[tool.coverage.run] omit` in `pyproject.toml` names one
by one, each with its reason, so a new script or harness counts unless you
add an entry. An entry names a file that no test imports by name and that
feeds no gate (`environments/shared/tests/test_coverage_config.py` checks
the first and holds the certification code and the statue baselines in);
`harnesses/digest_snapshot.py` is the one exception, checked by its own CI
step.

## Adding a species

The project is designed to make adding new dinosaur species straightforward. A
new species starts with its first embodiment, normally `anatomical`; a robot of
an existing species is the next section. Every name below follows
[docs/SPECIES_NAMING.md](docs/SPECIES_NAMING.md): `<species>` is the binomial in
snake_case, and `<Species>` its PascalCase form. Follow this checklist:

> **Note:** The training and test scripts use shared base modules in
> `environments/shared/`; every model trains through the shared CLI,
> `python -m environments.shared.train --species <species> --embodiment <embodiment> ...`.
> See `docs/CODE_CONSOLIDATION.md` for the consolidation plan and architecture
> details.

1. **Create the directory structure:**
   ```
   environments/<species>/
   ├── __init__.py
   ├── README.md                 # with a Models table, one column per embodiment
   ├── base_env.py               # <Species>Env, the species base class
   ├── anatomical/
   │   ├── __init__.py
   │   ├── env.py                # <Species>AnatomicalEnv(<Species>Env)
   │   └── assets/<species>_anatomical.xml   # MuJoCo MJCF model
   ├── scripts/
   │   ├── __init__.py
   │   ├── view_model.py         # Model viewer (required --embodiment)
   │   └── test_env.py           # Quick env validation (wraps shared base)
   └── tests/
       ├── __init__.py
       └── test_anatomical_env.py  # Pytest suite
   ```
   Every Python directory gets an `__init__.py` (coverage skips a package
   without one); data directories get none.

2. **Create the MJCF model** (`anatomical/assets/<species>_anatomical.xml`, with
   `<mujoco model="<species>_anatomical">`):
   - Define the body hierarchy, joints, actuators, and sensors
   - Include a mocap body for the prey/food target
   - Add touch sensors on feet and relevant contact geoms

3. **Implement the environment:**
   - In `base_env.py`, subclass `BaseDinoEnv` from `environments.shared.base_env`
     as `<Species>Env` and implement the five abstract methods: `_cache_ids`,
     `_get_obs`, `_get_reward_info`, `_is_terminated`, `_spawn_target`. Put
     every method the plant contract hashes here, on the species base, so a
     later robot of the species does not move the anatomical model's
     policy-interface digest
   - Give the species base the class attributes the species registry reads:
     `HEIGHT_LABEL`, `BEHAVIOR_SECTION_LABEL` and `SUCCESS_KEYS`
   - In `anatomical/env.py`, subclass it as `<Species>AnatomicalEnv` with the
     embodiment's body data: `embodiment = "anatomical"`, `MODEL_PATH` built
     with `ModelId("<species>", "anatomical").asset_path(...)`, `FOOT_SUPPORT`
     (each foot's support geoms, sole and foot site, which the floor-truth gait
     measurement reads, and optionally its `ankle`, `heel_line` and
     `toe_line`, all three or none: the ankle hinge and the support span the
     whole-foot centre of pressure is measured on) and `REWARD_PROBE` (the
     digest harness's reward-roll amplitude and the MJCF element its success
     probe moves the target onto)
   - Do not call `gym.register`: `environments/__init__.py` registers every
     model of the species manifest as `MesozoicLabs/<Species>-Anatomical-v0`,
     and the species registry imports the manifest's `env_entrypoint`

4. **Declare the species SB3-only** (`base_env.py`):
   - Set the class attribute `supported_training_backends = ("stable-baselines3",)`,
     as `CompsognathusLongipesEnv` does. Stable-Baselines3 is the only training backend
     (decision D-D17); without the attribute the plant contract treats the
     model as dual-backend and fails with "cannot import MJX plant
     registration"
   - Do not add an `mjx_config.py`: the two that exist belong to the frozen
     MJX interface core, which the policy-interface digests of the anatomical
     Brachiosaurus and Dibothrosuchus hash and which is never
     edited or extended (`docs/PLANT_CONTRACT.md`, "Backend parity and runtime
     binding")
   - A biped can take the heading-free observation schema
     `bipedal-target-heading-free/v1`, as the T-Rex and the anatomical
     Compsognathus do: build its root block with
     `environments/shared/heading_observation.heading_free_root_segments`
     and declare the schema in `configs/plant_versions.toml`
     (`docs/PLANT_CONTRACT.md`, "The heading-free observation schema")

5. **Add curriculum configs** (`configs/<species>/anatomical/`):
   - Create `stance.toml`, `locomotion.toml` and the skill stage
     (`strike.toml`, `snap.toml`, `food_reach.toml` or `behavior.toml`), each
     named by its stage with no number, plus `stages.toml` and the eleven
     behavior recipes under `behaviors/`, each with
     `[behavior] species = "<species>"` and `embodiment = "anatomical"`
   - Follow the TOML structure from an existing species
   - Calibrate `reset_noise_scale` for stage 1 against
     `python environments/shared/scripts/zero_action_baseline.py --model-id <species> anatomical --sweep-noise`:
     a level at which a do-nothing policy reaches the full horizon in nearly
     every episode makes the stage reward a statue

6. **Write tests** (`environments/<species>/tests/test_anatomical_env.py`):
   - Use the shared test utilities in `environments/shared/`
   - Verify observation/action space shapes, reward components, determinism
   - Add the new model to the per-model tables of the shared tests
     (`test_species_integration.py`, `test_mjcf_assets.py`,
     `test_plant_contract_digest_snapshot.py`'s `_SUCCESS` and
     `_POSE_REASONS`, and the others): each table asserts that it covers
     every model of the manifest, or exempts it explicitly, so a missing
     entry fails by name

7. **Declare the plant revisions** (`configs/plant_versions.toml`):
   - Add a `[plants.<species>.anatomical]` table starting every revision at 1,
     with the model's `observation_schema`, and a numbered note
   - Run `python -m environments.shared.plant_contract --write` to regenerate
     `configs/plant_manifest.generated.json`. If a shared interface function
     changed, the generator will refuse until the affected models' revisions
     are bumped

8. **Add the manifest entry** (`configs/species_manifest.toml`):
   - Add `[species.<species>]` with its `display_order` and `display_name`, and
     `[species.<species>.embodiments.anatomical]` with the presentation
     metadata, the environment entry point
     (`environments.<species>.anatomical.env:<Species>AnatomicalEnv`) and the
     MJCF path
   - Set `training_backends = ["stable-baselines3"]` and
     `training_notebooks = ["sb3_training"]`; `plant_contract/manifest.py`
     raises when `training_backends` differs from the environment's
     `supported_training_backends` (step 4)
   - Declare success semantics for `stable-baselines3`
   - Declare only existing, provenance-labelled result summaries or stage videos
   - Run `python -m environments.shared.species_catalog` to regenerate the
     README blocks, `docs/SPECIES_CATALOG.md` and
     `website/src/data/species.generated.json`
   - Run `python -m environments.shared.species_catalog --check` to verify that
     generated data and declared artifacts are current
   - Once its stages and behavior recipes are in place, run
     `python -m environments.shared.harnesses.digest_snapshot --block-optional-backends --write`
     to add the model's lines to `configs/digest_snapshot.generated.txt`; the
     reward section reads the class's `REWARD_PROBE` (step 3)

9. **Update CI** (`.github/workflows/python-ci.yml`):
   - Add `environments/<species>/tests/` to the `species` suite's `tests` in
     the `test` job's matrix (`include`); the species directories run in one
     pytest run per Python version

10. **Update pyproject.toml**:
    - Add the test path to `testpaths` in `[tool.pytest.ini_options]`
      (`environments/shared/tests/test_ci_tool_pins.py` checks that the CI
      matrix runs every `testpaths` directory exactly once)

11. **Add the documentation** (`environments/<species>/README.md`,
    `website/docs/models/<species>/`):
    - A `_category_.json` holding `{"label": "<display name>", "position": <display_order>}`
      and the model page `anatomical.mdx`, which renders
      `requireModel('<species>', 'anatomical')` from `website/src/data/species.ts`
      and sets `sidebar_label: Anatomical`; the Models sidebar is
      autogenerated, so `website/sidebars.ts` needs no edit
    - Add the model to the model list in `website/docs/intro.md`, its row to
      the model table in `docs/SPECIES_NAMING.md` and its Gymnasium ID to the
      list in `website/docs/api/overview.md` (contract tests compare the three
      lists and the `_category_.json` with the manifest), and update the
      model counts those pages state

## Adding an embodiment (a robot)

A robot of an existing species is a new embodiment of that species, not a new
species: it shares the species' base class, task, rewards, stage graph and
docs, and adds its own body. It takes exactly these, and nothing in
`environments/shared/` beyond its per-model test tables (step 7), CI,
`pyproject.toml`, pre-commit, the website sidebar or the notebooks (the
notebook's `EMBODIMENT` dropdown already offers `robot`):

1. **The package** `environments/<species>/robot/`:
   - `__init__.py`
   - `env.py` with `<Species>RobotEnv(<Species>Env)`: `embodiment = "robot"`,
     `MODEL_PATH`, `FOOT_SUPPORT` and `REWARD_PROBE`, as in step 3 above
   - `assets/<species>_robot.xml` (with `assets/meshes/` if it uses meshes),
     and `references/` for third-party sources, with their licence notice
     beside the files it covers
2. **The configs** `configs/<species>/robot/`: the stage files and
   `stages.toml`, and every behavior recipe of the species with
   `embodiment = "robot"`. Write each file whole: never `extends` another
   embodiment's TOML (`extends` resolves only inside its own directory).
3. **The manifest table** `[species.<species>.embodiments.robot]`, with its own
   `training_backends = ["stable-baselines3"]`.
4. **The plant table** `[plants.<species>.robot]` in
   `configs/plant_versions.toml`, at revisions 1 / 1 / 1, with a numbered
   note that records which hardware revision `physics_revision` 1 is. A later
   hardware revision is a physics revision of this embodiment, not a new one.
5. **A recovery calibration** `configs/<species>/robot/recovery_calibration.json`,
   when the stage graph has a recovery stage, measured with
   `python -m environments.shared.scripts.calibrate_recovery --species <species> --embodiment robot --output <file>`
   ([the Compsognathus calibration guide](environments/compsognathus_longipes/RECOVERY_CALIBRATION.md)).
6. **The documentation** (contract tests compare the three model lists with
   the manifest):
   - `website/docs/models/<species>/robot.mdx` and a robot column in the
     species README's Models table
   - a link to the model page in the model list of `website/docs/intro.md`
   - the model's row in the model table of `docs/SPECIES_NAMING.md`, and its
     Gymnasium ID in the list in `website/docs/api/overview.md`
   - the model counts `intro.md` and `api/overview.md` state ("Six Models of
     Five Species", "registers all six")
7. **The tests:** the entries the per-model test tables ask for when their
   coverage assertions name the new model, and robot-specific tests in
   `environments/<species>/tests/` (already in CI and `testpaths`).
8. **Regenerate:** `python -m environments.shared.plant_contract --write`,
   `python -m environments.shared.harnesses.digest_snapshot --block-optional-backends --write`
   and `python -m environments.shared.species_catalog`.

Code that robots share (servo and IMU models, system identification, and the
ROS 2 bridge behind an optional extra) belongs in one shared package, never
per species (decision D-D33). None of it exists yet.

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
