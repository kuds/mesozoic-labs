---
sidebar_position: 1
slug: /
---

# Welcome to Mesozoic Labs

**Mesozoic Labs** is an open-source research platform for dinosaur-inspired locomotion using physics simulation and reinforcement learning.

## What is Mesozoic Labs?

We use articulated MuJoCo models to study how reinforcement-learning agents can learn balance, locomotion, and species-inspired tasks. These models are engineering abstractions inspired by dinosaur morphology; they have not been validated as accurate reconstructions of dinosaur anatomy. Hardware transfer is a future research direction, and no sim-to-real or physical-robot results are currently published.

## Key Features

- **Physics-Based Models** - Articulated MuJoCo models inspired by dinosaur morphology
- **Reinforcement Learning** - PPO and SAC algorithms for training locomotion
- **Behavior Recipes** - Each model declares its supported training stages in `configs/<species>/<embodiment>/stages.toml`. The intermediate T. rex trains stance and command-driven locomotion.
- **Eight Models of Five Species** - Five anatomical models, the Compsognathus robot, and the mini and intermediate T. rex robots. Each model has a distinct species and embodiment pair.
- **Sim-to-Real Roadmap** - Planned hardware, system-identification, and transfer experiments; not yet validated
- **Open Source** - Fully open codebase for research and education

## Species Catalog and Published Results

The model pages render observation and action dimensions, compiled-model facts,
current curriculum stages, and published result summaries from the generated
species catalog. Each stage row shows its recipe label (stance, recover, walk or hunt),
whether it is a published deliverable, and the parent it warm-starts from;
published results list one row per deliverable, headlined by the metric its
gate kind certifies:

- [Velociraptor Mongoliensis (Anatomical)](/docs/models/velociraptor_mongoliensis/anatomical)
- [Tyrannosaurus Rex (Anatomical)](/docs/models/tyrannosaurus_rex/anatomical)
- [Tyrannosaurus Rex (Robot)](/docs/models/tyrannosaurus_rex/robot)
- [Tyrannosaurus Rex (Intermediate)](/docs/models/tyrannosaurus_rex/intermediate)
- [Brachiosaurus Altithorax (Anatomical)](/docs/models/brachiosaurus_altithorax/anatomical)
- [Dibothrosuchus Elaphros (Anatomical)](/docs/models/dibothrosuchus_elaphros/anatomical)
- [Compsognathus Longipes (Anatomical)](/docs/models/compsognathus_longipes/anatomical)
- [Compsognathus Longipes (Robot)](/docs/models/compsognathus_longipes/robot)

The published result summaries describe historical ladder runs with no
per-deliverable records. They are marked unverified because the original
repository commit, model hash, and config hash were not recorded; they should
not be treated as controlled algorithm comparisons or as results from the
current model revision and configs.

## Quick Links

- [Getting Started](/docs/getting-started/installation) - Set up your development environment
- [Models](/docs/models/tyrannosaurus_rex/anatomical) - Explore available species models
- [Behavior Recipes](/docs/training/recipes) - Train stance, walk and hunt on one certified trunk
- [Training](/docs/training/ppo) - Learn how to train your own dinosaur
- [GitHub](https://github.com/kuds/mesozoic-labs) - View the source code

## Project Status

The project is actively under development. Core infrastructure includes
automated curriculum training, W&B experiment tracking, and evaluation metrics.
See the repository roadmap for planned work, including hardware and sim-to-real
experiments that have not yet been validated.
