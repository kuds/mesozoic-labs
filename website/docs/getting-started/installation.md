---
sidebar_position: 1
---

# Installation

Get started with Mesozoic Labs by setting up your development environment.

## Prerequisites

- Python 3.11+ (tested on 3.11–3.13)
- CUDA-compatible GPU (recommended for training, not required)
- [Docker](https://docs.docker.com/get-docker/) (optional, recommended for reproducible training)

## Docker (Recommended)

The repo ships a `Dockerfile` that bundles MuJoCo, Stable-Baselines3, and all training dependencies into a single image — no manual dependency management required.

```bash
# Build the image (from the repo root)
docker build -t mesozoic-labs:latest .

# Verify the image works with a quick smoke-test
docker run --rm mesozoic-labs:latest \
  environments/velociraptor/scripts/train_sb3.py \
  train --stage 1 --timesteps 1000 --n-envs 1

# Train with GPU and write outputs to your local machine
docker run --rm --gpus all \
  -v "$(pwd)/outputs:/app/outputs" \
  mesozoic-labs:latest \
  environments/velociraptor/scripts/train_sb3.py \
  curriculum --algorithm ppo --n-envs 4 --output-dir /app/outputs/velociraptor
```

The Dockerfile sets `MUJOCO_GL=osmesa` for headless rendering — no display is needed.

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

# Install JAX/MJX for GPU-accelerated training
pip install -e ".[jax]"

# Or install all optional dependencies (training, JAX, visualization, dev tools)
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

For the easiest setup, use the pre-configured Google Colab notebooks in the `notebooks/` directory. The training notebooks use a species selector rather than separate files for each species, and they handle dependency installation automatically.

Available notebooks:

- `notebooks/sb3_training.ipynb` - Unified SB3 curriculum training for T-Rex, Velociraptor, and Brachiosaurus
- `notebooks/jax_training.ipynb` - Unified JAX/MJX training for all three species on an NVIDIA GPU
- `notebooks/ray_tune_sweep.ipynb` - Ray Tune hyperparameter sweeps
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

Optional JAX dependencies (`pip install -e ".[jax]"`):

| Package | Version | Purpose |
|---------|---------|---------|
| mujoco-mjx | >= 3.0.0 | GPU-accelerated MuJoCo simulation |
| jax[cuda12] | >= 0.4.20 | JAX with CUDA support |
| flax | >= 0.8.0 | Neural network library for JAX |
| optax | >= 0.1.7 | Gradient-based optimization for JAX |
