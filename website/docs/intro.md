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
- **Behavior Recipes** - Every species trains stand, walk and hunt as separately certified, separately published policies on one shared trunk, declared in `configs/<species>/stages.toml`
- **JAX/MJX Integration** - Batched, GPU-oriented PPO training for T-Rex, Velociraptor, Brachiosaurus, and Dibothrosuchus
- **Six Species** - T-Rex, Velociraptor, Brachiosaurus, Dibothrosuchus, Compsognathus, and the Compsognathus robot
- **Sim-to-Real Roadmap** - Planned hardware, system-identification, and transfer experiments; not yet validated
- **Open Source** - Fully open codebase for research and education

## Species Catalog and Published Results

The model pages render observation and action dimensions, compiled-model facts,
current curriculum stages, and published result summaries from the generated
species catalog. Each stage row shows its recipe label (stand, walk or hunt),
whether it is a published deliverable, and the parent it warm-starts from;
published results list one row per deliverable, headlined by the metric its
gate kind certifies:

- [Velociraptor](/docs/models/velociraptor)
- [T-Rex](/docs/models/trex)
- [Brachiosaurus](/docs/models/brachiosaurus)
- [Dibothrosuchus](/docs/models/dibothrosuchus)
- [Compsognathus](/docs/models/compsognathus)
- [Compsognathus (Robot)](/docs/models/compsognathus_robot)

The published result summaries describe historical ladder runs with no
per-deliverable records. They are marked unverified because the original
repository commit, model hash, and config hash were not recorded; they should
not be treated as controlled algorithm comparisons or as results from the
current model revision and configs.

## Quick Links

- [Getting Started](/docs/getting-started/installation) - Set up your development environment
- [Models](/docs/models/trex) - Explore available species models
- [Behavior Recipes](/docs/training/recipes) - Train stand, walk and hunt on one certified trunk
- [Training](/docs/training/ppo) - Learn how to train your own dinosaur
- [JAX/MJX Training](/docs/training/jax) - GPU-accelerated training with JAX
- [GitHub](https://github.com/kuds/mesozoic-labs) - View the source code

## Project Status

The project is actively under development. Core infrastructure includes
automated curriculum training, W&B experiment tracking, and evaluation metrics.
See the repository roadmap for planned work, including hardware and sim-to-real
experiments that have not yet been validated.
