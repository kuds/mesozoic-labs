# Mesozoic Labs

Robotic dinosaur locomotion research using reinforcement learning and MuJoCo physics simulation.

![Trained SAC Agent](/Images/sac_apex.gif)

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
│       ├── metrics.py         # Training metrics tracking
│       └── wandb_integration.py
├── configs/                   # TOML hyperparameter configs per species/stage
├── notebooks/                 # Jupyter notebooks for experiments
│   ├── velociraptor_training.ipynb
│   ├── brachiosaurus_training.ipynb
│   ├── trex_training.ipynb
│   └── jax_trex_training.ipynb
├── website/                   # Documentation site (Docusaurus)
└── Images/                    # Training visualizations
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
| Observation | 69 dims (joints, pelvis, prey tracking) |
| Action | 12 dims (leg + claw controls) |
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
| Observation | 75 dims (joints, torso, food tracking) |
| Action | 22 dims (6 neck + 16 leg controls) |
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
| Observation | 77 dims (joints, pelvis, prey tracking) |
| Action | 14 dims (3 neck/head + 1 jaw + 5 per leg) |
| Model | `environments/trex/assets/trex.xml` |

### Planned Species
- Deinonychus (pack hunter)
- Allosaurus (large theropod)
- Compsognathus (small, fast biped)

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

# Train stage 1 (balance)
cd environments/velociraptor
python scripts/train_sb3.py train --stage 1 --timesteps 500000
```

## Training Results

Hardware: Google Colab T4 GPU

| Dinosaur       | Algorithm | Avg Reward | Training Time | Steps     |
|----------------|-----------|------------|---------------|-----------|
| Basic Dinosaur | PPO       | 319.94     | 1:29:43       | 2,600,000 |
| Basic Dinosaur | SAC       | 3091.31    | 4:36:59       | 3,600,000 |
| T-Rex          | PPO       | -          | -             | 5,000,000 |
| T-Rex          | SAC       | -          | -             | 5,000,000 |
| Velociraptor   | PPO       | -          | -             | 5,000,000 |
| Velociraptor   | SAC       | -          | -             | 5,000,000 |
| Brachiosaurus  | PPO       | -          | -             | 3,500,000 |

## Notebooks

| Notebook | Description |
|----------|-------------|
| `notebooks/velociraptor_training.ipynb` | Velociraptor 3-stage curriculum training (Colab-ready) |
| `notebooks/brachiosaurus_training.ipynb` | Brachiosaurus 3-stage curriculum training (Colab-ready) |
| `notebooks/trex_training.ipynb` | T-Rex 3-stage curriculum training (Colab-ready) |
| `notebooks/jax_trex_training.ipynb` | JAX/MJX T-Rex training for TPU acceleration (Colab-ready) |

## Roadmap

- [ ] Complete velociraptor 3-stage training
- [ ] Complete brachiosaurus 3-stage training
- [ ] T-Rex environment and training
- [ ] JAX/MJX migration for faster training
- [ ] Multi-agent pack hunting scenarios
- [ ] Terrain adaptation (uneven ground, obstacles)
- [ ] Sim-to-real transfer experiments

See [ROADMAP.md](ROADMAP.md) for the full phased timeline, milestones, and dependency graph.

## Resources

- **Documentation:** [mesozoiclabs.com](https://mesozoiclabs.com)
- **Blog:** [From Zero to Dino-Roar](https://www.findingtheta.com/blog/from-zero-to-dino-roar-teaching-a-t-rex-to-walk-with-mujoco-and-reinforcement-learning)

## Contributing

Contributions welcome! Open an issue or PR.

## License

MIT License
