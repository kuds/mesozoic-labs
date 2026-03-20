---
license: mit
language:
- en
library_name: stable-baselines3
tags:
- reinforcement-learning
- mujoco
- locomotion
- robotics
- curriculum-learning
- dinosaurs
- gymnasium
model-index:
- name: PPO-Velociraptor
  results:
  - task:
      type: reinforcement-learning
      name: reinforcement-learning
    dataset:
      name: MesozoicLabs/Raptor-v0
      type: MesozoicLabs/Raptor-v0
    metrics:
    - type: mean_reward
      value: 1366.19 +/- 76.29
      name: mean_reward
      verified: false
    - type: success_rate
      value: 93.3%
      name: strike_success_rate
      verified: false
- name: PPO-TRex
  results:
  - task:
      type: reinforcement-learning
      name: reinforcement-learning
    dataset:
      name: MesozoicLabs/TRex-v0
      type: MesozoicLabs/TRex-v0
    metrics:
    - type: mean_reward
      value: 1294.28 +/- 67.19
      name: mean_reward
      verified: false
    - type: success_rate
      value: 96.7%
      name: bite_success_rate
      verified: false
---

# **PPO** Agents for Robotic Dinosaur Locomotion — **Mesozoic Labs**

![Trained PPO Agent](/results/velociraptor/ppo/stage1_balance.gif)

This repository contains **PPO** (Proximal Policy Optimization) agents trained to control robotic dinosaurs in MuJoCo physics simulation. Each species is trained using a 3-stage curriculum learning approach.

- [GitHub Repository](https://github.com/kuds/mesozoic-labs)
- [Documentation](https://mesozoiclabs.com)
- [Blog: From Zero to Dino-Roar](https://www.findingtheta.com/blog/from-zero-to-dino-roar-teaching-a-t-rex-to-walk-with-mujoco-and-reinforcement-learning)

## Species & Training Results

### Velociraptor (PPO) — All 3 stages passed | 22M steps | 11:25:15 total

A bipedal predator with sickle claws, trained on 3 curriculum stages:

| Stage | Name | Best Reward | Avg Forward Vel | Success Rate | Time |
|-------|------|-------------|-----------------|--------------|------|
| 1 | Balance | 1964.43 +/- 27.39 | 0.11 m/s | — | 2:57:25 |
| 2 | Locomotion | 2678.68 +/- 4.07 | 3.47 m/s | — | 4:35:55 |
| 3 | Strike | 1366.19 +/- 76.29 | 2.02 m/s | 93.3% | 3:51:54 |

### T-Rex (PPO) — All 3 stages passed | 22M steps | 13:02:32 total

A large bipedal predator with powerful jaws, trained on 3 curriculum stages:

| Stage | Name | Best Reward | Avg Forward Vel | Success Rate | Time |
|-------|------|-------------|-----------------|--------------|------|
| 1 | Balance | 3008.66 +/- 7.62 | 0.02 m/s | — | 3:35:24 |
| 2 | Locomotion | 1936.01 +/- 13.12 | 3.47 m/s | — | 5:17:18 |
| 3 | Bite | 1294.28 +/- 67.19 | 1.68 m/s | 96.7% | 4:09:49 |

### Brachiosaurus (PPO) — In progress

A quadrupedal sauropod herbivore with a long neck for reaching elevated food sources.

## Training Details

- **Algorithm:** PPO (Proximal Policy Optimization) via [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3)
- **Physics Engine:** [MuJoCo](https://mujoco.org/) (>= 3.0)
- **Environment Framework:** [Gymnasium](https://gymnasium.farama.org/) (>= 0.29)
- **Hardware:** Google Colab L4 GPU
- **Seed:** 42
- **Parallel Envs:** 4
- **Curriculum:** 3-stage progressive training (Balance → Locomotion → Species-specific task)

## Environment Details

| Species | Observation Dims | Action Dims | Gymnasium ID |
|---------|-----------------|-------------|--------------|
| Velociraptor | 67 | 22 | `MesozoicLabs/Raptor-v0` |
| T-Rex | 83 | 21 | `MesozoicLabs/TRex-v0` |
| Brachiosaurus | 75 | 22 | `MesozoicLabs/Brachio-v0` |

## Usage

### Installation

```bash
git clone https://github.com/kuds/mesozoic-labs.git
cd mesozoic-labs

python -m venv venv
source venv/bin/activate

# Install with training dependencies
pip install -e ".[train]"
```

### Loading a Trained Model

```python
from stable_baselines3 import PPO
import gymnasium as gym

# Register Mesozoic Labs environments
import environments

# Load the trained model (e.g., velociraptor stage 3)
model = PPO.load("path/to/best_model.zip")

# Create the environment
env = gym.make("MesozoicLabs/Raptor-v0", render_mode="human")

# Run the trained agent
obs, info = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
env.close()
```

### Training from Scratch

```bash
# Full 3-stage curriculum for velociraptor
cd environments/velociraptor
python scripts/train_sb3.py curriculum --algorithm ppo

# Single stage training
python scripts/train_sb3.py train --stage 1 --timesteps 6000000 --n-envs 4
```

### Loading from Hugging Face Hub

```bash
pip install huggingface_hub
```

```python
from huggingface_hub import hf_hub_download
from stable_baselines3 import PPO
import gymnasium as gym
import environments

# Download the model from the Hub
model_path = hf_hub_download(
    repo_id="kuds/mesozoic-labs",
    filename="results/velociraptor/ppo/best_model.zip"
)

# Load the model
model = PPO.load(model_path)

# Create the environment
env = gym.make("MesozoicLabs/Raptor-v0", render_mode="human")

# Run the trained agent
obs, info = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
env.close()
```

### Docker

```bash
# Build
docker build -t mesozoic-labs:latest .

# Quick smoke-test
docker run --rm mesozoic-labs:latest \
  environments/velociraptor/scripts/train_sb3.py \
  train --stage 1 --timesteps 1000 --n-envs 1

# Full curriculum with GPU
docker run --rm --gpus all \
  -v "$(pwd)/outputs:/app/outputs" \
  mesozoic-labs:latest \
  environments/velociraptor/scripts/train_sb3.py \
  curriculum --algorithm ppo --n-envs 4 --output-dir /app/outputs/velociraptor
```

## Notebooks

| Notebook | Description |
|----------|-------------|
| `notebooks/training.ipynb` | Unified 3-stage curriculum training for all species (Colab-ready) |
| `notebooks/jax_trex_training.ipynb` | JAX/MJX T-Rex training for GPU acceleration (Colab-ready) |
| `notebooks/ray_tune_sweep.ipynb` | Ray Tune hyperparameter sweep with ASHA early stopping (Colab-ready) |
| `notebooks/google_drive_summary.ipynb` | Training runs summary and comparison across all species (Colab-ready) |

## Citation

```bibtex
@misc{mesozoic-labs,
  author = {Mesozoic Labs Contributors},
  title = {Mesozoic Labs: Robotic Dinosaur Locomotion with Reinforcement Learning},
  year = {2026},
  publisher = {GitHub / Hugging Face},
  url = {https://github.com/kuds/mesozoic-labs}
}
```

## License

MIT License
