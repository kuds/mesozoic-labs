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

## Option 2: Docker (Recommended for Reproducibility)

The repo ships a ready-to-use `Dockerfile` that bundles MuJoCo, Stable-Baselines3, and all training dependencies.

```bash
# Build the image
docker build -t mesozoic-labs:latest .

# Test it with a quick 1000-step run (no GPU needed)
docker run --rm mesozoic-labs:latest \
  environments/velociraptor/scripts/train_sb3.py \
  train --stage 1 --algorithm ppo --timesteps 1000 --n-envs 1

# Full curriculum (all 3 stages) with GPU
docker run --rm --gpus all \
  -v "$(pwd)/outputs:/app/outputs" \
  mesozoic-labs:latest \
  environments/velociraptor/scripts/train_sb3.py \
  curriculum --algorithm ppo --n-envs 4 --output-dir /app/outputs/velociraptor
```

The `--output-dir` flag writes all checkpoints and logs to the mounted host directory.

## Option 3: Local Setup

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

The `curriculum` command runs all three stages in a single call. Each stage automatically loads its own hyperparameters from the TOML config when it starts:

```bash
# Full 3-stage curriculum — one command, all stages handled automatically
python scripts/train_sb3.py curriculum --algorithm ppo

# Or control stages manually
python scripts/train_sb3.py train --stage 1 --algorithm ppo --timesteps 1000000
python scripts/train_sb3.py train --stage 2 --algorithm ppo --timesteps 2000000 \
  --load logs/<stage1_dir>/models/stage1_final.zip
python scripts/train_sb3.py train --stage 3 --algorithm ppo --timesteps 3000000 \
  --load logs/<stage2_dir>/models/stage2_final.zip
```

### Evaluate a Trained Policy

```bash
python scripts/train_sb3.py eval logs/<stage_dir>/models/stage1_final.zip --algorithm ppo
```

### Override Hyperparameters

```bash
# Try a different learning rate without editing the TOML files
python scripts/train_sb3.py train --stage 1 \
  --override ppo.learning_rate=1e-3 env.alive_bonus=3.0
```

### Run Tests

```bash
pytest -v
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
