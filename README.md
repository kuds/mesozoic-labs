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
│   │   └── README.md
│   ├── brachiosaurus/          # Brachiosaurus (quadrupedal sauropod)
│   │   ├── assets/            # MJCF model files
│   │   ├── envs/              # Gymnasium environments
│   │   ├── scripts/           # Training & utility scripts
│   │   └── tests/             # Pytest test suite
│   ├── trex/                  # T-Rex (large bipedal predator)
│   │   └── trex.xml           # MJCF model
│   └── [future]/              # More species coming...
├── notebooks/                 # Jupyter notebooks for experiments
│   ├── ppo_training.ipynb
│   └── sac_training.ipynb
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
**Status:** Model ready, training planned

Large bipedal predator with humanoid-based locomotion.

| Feature | Details |
|---------|---------|
| Model | `environments/trex/trex.xml` |

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

# Install dependencies for velociraptor
pip install -r environments/velociraptor/requirements.txt

# View the model
cd environments/velociraptor
python scripts/view_model.py

# Train stage 1
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
| `notebooks/ppo_training.ipynb` | PPO training experiments (Colab-ready) |
| `notebooks/sac_training.ipynb` | SAC training experiments (Colab-ready) |

## Roadmap

- [ ] Complete velociraptor 3-stage training
- [ ] Complete brachiosaurus 3-stage training
- [ ] T-Rex environment and training
- [ ] JAX/MJX migration for faster training
- [ ] Multi-agent pack hunting scenarios
- [ ] Terrain adaptation (uneven ground, obstacles)
- [ ] Sim-to-real transfer experiments

## Resources

- **Documentation:** [mesozoiclabs.com](https://mesozoiclabs.com)
- **Blog:** [From Zero to Dino-Roar](https://www.findingtheta.com/blog/from-zero-to-dino-roar-teaching-a-t-rex-to-walk-with-mujoco-and-reinforcement-learning)

## Contributing

Contributions welcome! Open an issue or PR.

## License

MIT License
