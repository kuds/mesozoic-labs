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

## Verify Installation

```bash
# View a model (requires display)
python environments/velociraptor/scripts/view_model.py

# Run environment tests
pytest environments/velociraptor/tests/ -v
```

## Google Colab

For the easiest setup, use the pre-configured Google Colab notebooks in the `notebooks/` directory. The training notebook uses a species selector rather than separate files for each species, and it handles dependency installation automatically.

Available notebooks:

- `notebooks/sb3_training.ipynb` - Trains one behavior (`BEHAVIOR = "hunt"` by default; `"stand"`, `"walk"` or a deliverable's stage id) for any of the six species with SB3, reusing a trunk run's certified ancestors through `TRUNK_FROM`; see [Behavior Recipes](/docs/training/recipes)
- `notebooks/google_drive_summary.ipynb` - Training-run summaries and comparisons

## Dependencies

Core requirements (from `pyproject.toml`):

| Package | Version | Purpose |
|---------|---------|---------|
| mujoco | >= 3.0.0 | Physics simulation |
| gymnasium | >= 0.29.0 | RL environment API |
| numpy | >= 1.24.0 | Numerical computing |

Optional training dependencies (`pip install -e ".[train]"`):

| Package | Version | Purpose |
|---------|---------|---------|
| stable-baselines3 | >= 2.2.0 | RL algorithms (PPO, SAC) |
| wandb | >= 0.16.0 | Experiment tracking |
