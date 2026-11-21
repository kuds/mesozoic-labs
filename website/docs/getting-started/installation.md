---
sidebar_position: 1
---

# Installation

Get started with Mesozoic Labs by setting up your development environment.

## Prerequisites

- Python 3.11+ (tested on 3.11–3.13)
- CUDA-compatible GPU (recommended for training, not required)

## Local Install

```bash
# Clone the repository
git clone https://github.com/kuds/mesozoic-labs.git
cd mesozoic-labs

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package with SB3 training dependencies
pip install -e ".[train]"

# Or install all optional dependencies (training, visualization, dev tools)
pip install -e ".[all]"
```

On Windows, the repository forces LF line endings on checkout (`.gitattributes`), because the plant contract hashes the checked-out MJCF and TOML bytes. A clone made before LF was forced (November 2026) must be re-cloned or re-checked out, or training and evaluation refuse with "generated plant manifest is stale". To re-check out, commit or stash your local edits first, since `git reset --hard` discards them, then run `git rm --cached -r . && git reset --hard`.

## Verify Installation

```bash
# View a model (requires display)
python environments/velociraptor_mongoliensis/scripts/view_model.py --embodiment anatomical

# Run environment tests
pytest environments/velociraptor_mongoliensis/tests/ -v
```

## Google Colab

For the easiest setup, use the pre-configured Google Colab notebooks in the `notebooks/` directory. The training notebook selects the model with two dropdowns, `SPECIES` and `EMBODIMENT`, rather than separate files for each model, and it handles dependency installation automatically.

Available notebooks:

- `notebooks/sb3_training.ipynb` - Trains one behavior (`BEHAVIOR = "hunt"` by default; `"stance"`, `"recover"`, `"walk"` or a deliverable's stage id) with SB3. Select `stance` or `walk` for Allosaurus. To reuse certified ancestors, set `TRUNK_FROM` as described in [Behavior Recipes](/docs/training/recipes)
- `notebooks/google_drive_summary.ipynb` - Training-run summaries and comparisons

## Dependencies

Core requirements (from `pyproject.toml`):

| Package | Version | Purpose |
|---------|---------|---------|
| mujoco | == 3.10.0 | Physics simulation (pinned; the plant contract requires this exact version) |
| gymnasium | >= 0.29.0 | RL environment API |
| numpy | >= 1.24.0 | Numerical computing |

Optional training dependencies (`pip install -e ".[train]"`):

| Package | Version | Purpose |
|---------|---------|---------|
| torch | >= 2.6 | Neural networks (2.6 fixes a `torch.load` safety bypass) |
| stable-baselines3 | >= 2.9.0 | RL algorithms (PPO, SAC) |
| wandb | >= 0.16.0 | Experiment tracking |
