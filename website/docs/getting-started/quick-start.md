---
sidebar_position: 2
---

# Quick Start

Train your first robotic dinosaur in minutes.

## Option 1: Google Colab (Easiest)

Open one of the pre-configured notebooks in the `notebooks/` directory:

- `notebooks/velociraptor_training.ipynb` - Velociraptor 3-stage curriculum
- `notebooks/trex_training.ipynb` - T-Rex 3-stage curriculum
- `notebooks/brachiosaurus_training.ipynb` - Brachiosaurus 3-stage curriculum

Each notebook handles dependency installation automatically.

## Option 2: Local Setup

```bash
# Clone and setup
git clone https://github.com/kuds/mesozoic-labs.git
cd mesozoic-labs

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package with training dependencies
pip install -e ".[train]"
```

### View the Model

```bash
cd environments/velociraptor
python scripts/view_model.py
```

### Train with Curriculum Learning

The velociraptor uses 3-stage curriculum learning:

```bash
# Stage 1: Learn to stand and balance
python scripts/train_sb3.py train --stage 1 --timesteps 1000000

# Stage 2: Learn to walk/run (loads Stage 1 weights)
python scripts/train_sb3.py train --stage 2 --timesteps 2000000 \
  --load logs/<stage1_dir>/models/stage1_final.zip

# Stage 3: Sprint and strike prey
python scripts/train_sb3.py train --stage 3 --timesteps 3000000 \
  --load logs/<stage2_dir>/models/stage2_final.zip
```

### Evaluate a Trained Policy

```bash
python scripts/train_sb3.py eval logs/<stage_dir>/models/stage1_final.zip
```

### Run Tests

```bash
pytest tests/ -v
```

## Basic Training Loop (Python)

```python
import gymnasium as gym

# Registers MesozoicLabs environments
import environments.velociraptor.envs.raptor_env  # noqa: F401

env = gym.make("MesozoicLabs/Raptor-v0")

obs, info = env.reset(seed=42)
for step in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()

env.close()
```
