# Mesozoic Labs

Robotic dinosaur locomotion research using reinforcement learning and MuJoCo physics simulation.

![Trained PPO Agent](/results/velociraptor/ppo/stage1_balance.gif)

## Overview

Mesozoic Labs is a research project exploring bipedal and quadrupedal locomotion in robotic dinosaurs. We use MuJoCo for realistic physics simulation and train agents with algorithms like PPO and SAC.

**Goals:**
- Develop realistic locomotion controllers for various dinosaur species
- Explore predatory behaviors (hunting, striking, pack coordination)
- Create transferable policies for robotic applications
- Experiment with JAX/MJX for high-performance training

## Repository Structure

```
mesozoic-labs/
├── environments/              # Dinosaur training environments
│   ├── velociraptor/          # Velociraptor (bipedal predator with sickle claws)
│   │   ├── assets/            # MJCF model files
│   │   ├── envs/              # Gymnasium environments
│   │   ├── scripts/           # Training & utility scripts
│   │   ├── tests/             # Pytest test suite
│   │   └── README.md
│   ├── brachiosaurus/         # Brachiosaurus (quadrupedal sauropod)
│   │   ├── assets/            # MJCF model files
│   │   ├── envs/              # Gymnasium environments
│   │   ├── scripts/           # Training & utility scripts
│   │   ├── tests/             # Pytest test suite
│   │   └── README.md
│   ├── trex/                  # T-Rex (large bipedal predator)
│   │   ├── assets/            # MJCF model files
│   │   ├── envs/              # Gymnasium environments
│   │   ├── scripts/           # Training & utility scripts
│   │   └── tests/             # Pytest test suite
│   └── shared/                # Shared base classes and utilities
│       ├── base_env.py        # BaseDinoEnv abstract class
│       ├── config.py          # TOML configuration loading
│       ├── curriculum.py      # Curriculum learning manager
│       ├── train_base.py      # Shared SB3 training infrastructure
│       ├── species_registry.py # Species configuration registry
│       ├── metrics.py         # Locomotion evaluation metrics
│       ├── wandb_integration.py # W&B experiment tracking
│       ├── mjx_env.py         # JAX/MJX batched environment
│       ├── jax_ppo.py         # JAX-native PPO implementation
│       ├── jax_training.py    # JAX training loop
│       └── tests/             # Shared utility tests
├── configs/                   # TOML hyperparameter configs per species/stage
├── notebooks/                 # Jupyter notebooks for experiments
│   ├── sb3_training.ipynb
│   ├── jax_training.ipynb
│   ├── ray_tune_sweep.ipynb
│   └── google_drive_summary.ipynb
├── website/                   # Documentation site (Docusaurus)
└── results/                   # Training results (GIFs + collected_results.csv per species/algorithm)
```

## Environments

### Velociraptor
**Status:** Active development

A bipedal predator with distinctive sickle claws, trained using 3-stage curriculum learning:
1. **Balance** - Learn to stand without falling
2. **Locomotion** - Walk and run forward
3. **Strike** - Sprint and attack prey with claws

| Feature | Details |
|---------|---------|
| Observation | 67 dims (joints, pelvis, prey tracking) |
| Action | 22 dims (legs, claws, tail, arms) |
| Model | `environments/velociraptor/assets/raptor.xml` |

[Full documentation →](environments/velociraptor/README.md)

### Brachiosaurus
**Status:** Active development

A quadrupedal sauropod herbivore with a long neck for reaching elevated food sources. The first quadrupedal species in the project, featuring columnar elephant-like legs and characteristic longer front legs.

Trained using 3-stage curriculum learning:
1. **Balance** - Stable quadrupedal stance
2. **Locomotion** - Coordinated four-legged walking
3. **Food Reach** - Walk to food and reach with neck

| Feature | Details |
|---------|---------|
| Observation | 83 dims (joints, torso, food tracking) |
| Action | 30 dims (6 neck + 24 leg controls) |
| Model | `environments/brachiosaurus/assets/brachiosaurus.xml` |

[Full documentation →](environments/brachiosaurus/README.md)

### T-Rex
**Status:** Active development

Large bipedal predator with a massive skull, powerful jaws, and vestigial forelimbs. Hunts by sprinting toward prey and delivering a bite.

Trained using 3-stage curriculum learning:
1. **Balance** - Stable bipedal stance
2. **Locomotion** - Walk and run toward prey
3. **Hunting** - Sprint and bite prey with jaws

| Feature | Details |
|---------|---------|
| Observation | 83 dims (joints, pelvis, prey tracking) |
| Action | 21 dims (3 neck/head + 7 per leg + 4 tail) |
| Model | `environments/trex/assets/trex.xml` |

### Planned Species
- Deinonychus (pack hunter)
- Compsognathus (small, fast biped)
- Stegosaurus (armored quadrupedal defender)

## Quick Start

```bash
# Clone and setup
git clone https://github.com/kuds/mesozoic-labs.git
cd mesozoic-labs

python -m venv venv
source venv/bin/activate

# Install the package with training dependencies
pip install -e ".[train]"

# View the velociraptor model
python environments/velociraptor/scripts/view_model.py

# Full 3-stage curriculum — one command, all stages handled automatically
# (each stage loads its own hyperparameters from the TOML config)
cd environments/velociraptor
python scripts/train_sb3.py curriculum --algorithm ppo
```

## Docker

The repo ships a `Dockerfile` that bundles MuJoCo, Stable-Baselines3, and all training dependencies:

```bash
# Build
docker build -t mesozoic-labs:latest .

# Quick smoke-test (no GPU needed)
docker run --rm mesozoic-labs:latest \
  environments/velociraptor/scripts/train_sb3.py \
  train --stage 1 --timesteps 1000 --n-envs 1

# Full curriculum with GPU, writing outputs to local disk
docker run --rm --gpus all \
  -v "$(pwd)/outputs:/app/outputs" \
  mesozoic-labs:latest \
  environments/velociraptor/scripts/train_sb3.py \
  curriculum --algorithm ppo --n-envs 4 --output-dir /app/outputs/velociraptor
```

See [Vertex AI training docs](website/docs/training/vertex-ai.md) for cloud deployment.

## Training Results

Hardware: Google Colab L4 GPU

### Velociraptor (PPO) — All 3 stages passed | 22M steps | 11:25:15 total

| Stage | Name | Best Reward | Avg Fwd Vel | Success Rate | Time |
|-------|------|-------------|-------------|--------------|------|
| 1 | Balance | 1964.43 | 0.11 m/s | — | 2:57:25 |
| 2 | Locomotion | 2678.68 | 3.47 m/s | — | 4:35:55 |
| 3 | Strike | 1366.19 | 2.02 m/s | 93.3% | 3:51:54 |

### Velociraptor (SAC) — All 3 stages passed | 22M steps | 22:59:18 total

| Stage | Name | Best Reward | Avg Fwd Vel | Success Rate | Time |
|-------|------|-------------|-------------|--------------|------|
| 1 | Balance | 970.19 | -0.64 m/s | — | 5:08:59 |
| 2 | Locomotion | 2078.62 | 2.91 m/s | — | 8:36:12 |
| 3 | Strike | 1195.43 | 1.63 m/s | 90.0% | 9:14:06 |

### T-Rex (PPO) — All 3 stages passed | 22M steps | 13:02:32 total

| Stage | Name | Best Reward | Avg Fwd Vel | Success Rate | Time |
|-------|------|-------------|-------------|--------------|------|
| 1 | Balance | 3008.66 | 0.02 m/s | — | 3:35:24 |
| 2 | Locomotion | 1936.01 | 3.47 m/s | — | 5:17:18 |
| 3 | Bite | 1294.28 | 1.68 m/s | 96.7% | 4:09:49 |

### Brachiosaurus (PPO) — Stages 1-2 passed, Stage 3 in progress | 30M steps | 15:59:39 total

| Stage | Name | Best Reward | Avg Fwd Vel | Success Rate | Time |
|-------|------|-------------|-------------|--------------|------|
| 1 | Balance | 3002.52 | 0.02 m/s | — | 3:46:42 |
| 2 | Locomotion | 4176.95 | 1.12 m/s | — | 8:18:51 |
| 3 | Food Reach | 732.20 | 0.52 m/s | 16.7% (target: 50%) | 3:54:06 |

## Notebooks

| Notebook | Description |
|----------|-------------|
| `notebooks/sb3_training.ipynb` | Unified 3-stage curriculum training for all species (Colab-ready) |
| `notebooks/jax_training.ipynb` | JAX/MJX training for all species with GPU acceleration (Colab-ready) |
| `notebooks/ray_tune_sweep.ipynb` | Ray Tune hyperparameter sweep with ASHA early stopping (Colab-ready) |
| `notebooks/google_drive_summary.ipynb` | Training runs summary and comparison across all species (Colab-ready) |

## Roadmap

- [x] Complete velociraptor 3-stage training (PPO, 93.3% strike success)
- [x] Complete velociraptor 3-stage training (SAC, 90.0% strike success)
- [x] Complete T-Rex 3-stage training (PPO, 96.7% bite success)
- [-] Complete brachiosaurus 3-stage training (Stages 1-2 passed, Stage 3 food_reach at 16.7% vs 50% target)
- [-] SAC training for T-Rex (velociraptor SAC complete)
- [ ] Domain randomization (friction, damping, gravity, actuator strength, external pushes, observation noise)
- [ ] Terrain adaptation (uneven ground, obstacles)
- [-] JAX/MJX migration for faster training (PPO pipeline complete, SAC pending)
- [ ] Multi-agent pack hunting scenarios
- [ ] Sim-to-real transfer experiments

See [docs/ROADMAP.md](docs/ROADMAP.md) for the full phased timeline, milestones, and dependency graph.

## Resources

- **Documentation:** [mesozoiclabs.com](https://mesozoiclabs.com)
- **Blog:** [From Zero to Dino-Roar](https://www.findingtheta.com/blog/from-zero-to-dino-roar-teaching-a-t-rex-to-walk-with-mujoco-and-reinforcement-learning)

## Development

```bash
# Install with all dev dependencies
pip install -e ".[all]"

# Run tests
pytest

# Lint and type check
ruff check environments/
mypy environments/
```

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Citation

If you use Mesozoic Labs in your research, please cite:

```bibtex
@software{mesozoic_labs,
  title     = {Mesozoic Labs: Dinosaur Locomotion via Reinforcement Learning},
  author    = {Michael Kudlaty},
  year      = {2025},
  url       = {https://github.com/kuds/mesozoic-labs},
  license   = {MIT}
}
```

## License

MIT License
