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
- gymnasium
model-index:
- name: PPO-Velociraptor-Locomotion
  results:
  - task:
      type: reinforcement-learning
      name: reinforcement-learning
    dataset:
      name: MesozoicLabs/Raptor-v0
      type: MesozoicLabs/Raptor-v0
    metrics:
    - type: mean_reward
      value: 2678.68 +/- 4.07
      name: mean_reward
      verified: false
    - type: mean_forward_velocity
      value: 3.47 m/s
      name: mean_forward_velocity
      verified: false
---

# **PPO** Agent playing **MesozoicLabs/Raptor-v0** (Locomotion)

![Trained PPO Agent](/results/velociraptor/ppo/stage1_balance.gif)

This is a trained **PPO** (Proximal Policy Optimization) agent that controls a bipedal velociraptor in MuJoCo physics simulation, learning to walk and run forward.

- [GitHub Repository](https://github.com/kuds/mesozoic-labs)
- [Documentation](https://mesozoiclabs.com)
- [Blog: From Zero to Dino-Roar](https://www.findingtheta.com/blog/from-zero-to-dino-roar-teaching-a-t-rex-to-walk-with-mujoco-and-reinforcement-learning)

## Training Results

The agent is trained using a 2-stage curriculum (Balance → Locomotion) over 14M total timesteps on a Google Colab L4 GPU:

| Stage | Name | Best Reward | Avg Forward Vel | Time |
|-------|------|-------------|-----------------|------|
| 1 | Balance | 1964.43 +/- 27.39 | 0.11 m/s | 2:57:25 |
| 2 | Locomotion | 2678.68 +/- 4.07 | 3.47 m/s | 4:35:55 |

## Training Details

- **Algorithm:** PPO via [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3)
- **Physics Engine:** [MuJoCo](https://mujoco.org/) (>= 3.0)
- **Environment Framework:** [Gymnasium](https://gymnasium.farama.org/) (>= 0.29)
- **Hardware:** Google Colab L4 GPU
- **Seed:** 42
- **Parallel Envs:** 4
- **Total Timesteps:** 14M (6M balance + 8M locomotion)

## Environment

| Feature | Details |
|---------|---------|
| Gymnasium ID | `MesozoicLabs/Raptor-v0` |
| Observation | 67 dims (joints, pelvis, prey tracking) |
| Action | 22 dims (legs, claws, tail, arms) |
| MJCF Model | `environments/velociraptor/assets/raptor.xml` |

## Usage

Then, you can load the model using the following Python code:

```python
from stable_baselines3 import PPO
import gymnasium as gym

# Register Mesozoic Labs environments
import environments

# Load the trained model
model = PPO.load("best_model.zip")

# Create the environment
env = gym.make("MesozoicLabs/Raptor-v0", render_mode="human")

# Enjoy the trained agent
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
cd environments/velociraptor

# Stage 1: Balance
python scripts/train_sb3.py train --stage 1 --timesteps 6000000 --n-envs 4

# Stage 2: Locomotion
python scripts/train_sb3.py train --stage 2 --timesteps 8000000 --n-envs 4
```

### Hugging Face Hub

You can also use the Hugging Face Hub to load the model. First, you need to install the Hugging Face Hub library:

```bash
pip install huggingface_hub
```

Then, you can load the model from the hub using the following code:

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

# Enjoy the trained agent
obs, info = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
env.close()
```

## License

MIT License
