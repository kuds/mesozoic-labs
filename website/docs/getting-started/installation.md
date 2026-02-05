---
sidebar_position: 1
---

# Installation

Get started with Mesozoic Labs by setting up your development environment.

## Prerequisites

- Python 3.9+
- CUDA-compatible GPU (recommended for training, not required)

## Local Install

```bash
# Clone the repository
git clone https://github.com/kuds/mesozoic-labs.git
cd mesozoic-labs

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies for the velociraptor environment
pip install -r environments/velociraptor/requirements.txt

# (Optional) Install test dependencies
pip install pytest
```

## Verify Installation

```bash
cd environments/velociraptor
python scripts/view_model.py     # Opens interactive model viewer
pytest tests/ -v                 # Runs environment tests
```

## Google Colab

For the easiest setup, use the pre-configured Google Colab notebooks in the `notebooks/` directory. These handle all dependency installation automatically.

- [PPO Training Notebook](https://colab.research.google.com/github/kuds/apex/blob/main/%5BApex%5D%20Proximal%20Policy%20Optimization%20(PPO).ipynb)
- [SAC Training Notebook](https://colab.research.google.com/github/kuds/apex/blob/main/%5BApex%5D%20Soft%20Actor-Critic%20(SAC).ipynb)

## Dependencies

Core requirements (see `environments/velociraptor/requirements.txt`):

| Package | Version | Purpose |
|---------|---------|---------|
| mujoco | >= 3.0.0 | Physics simulation |
| gymnasium | >= 0.29.0 | RL environment API |
| stable-baselines3 | >= 2.2.0 | RL algorithms (PPO, SAC) |
| numpy | >= 1.24.0 | Numerical computing |
