---
sidebar_position: 1
slug: /
---

# Welcome to Mesozoic Labs

**Mesozoic Labs** designs physical robots inspired by Mesozoic animals. We start with anatomy, then develop robot models where a build is feasible. All current movement trials take place in simulation.

## What is Mesozoic Labs?

These docs cover the simulation tools behind the [animal collection](/animals). We use articulated MuJoCo models to study balance, locomotion and species-specific behaviors. The anatomical models simplify body proportions and joints for these experiments. The [Triceratops project](/animals/triceratops) shows how an anatomical study informs a robot design.

## Key Features

- **Physics-Based Models** - Articulated MuJoCo models inspired by dinosaur morphology
- **Reinforcement Learning** - PPO and SAC algorithms for training locomotion
- **Behavior Recipes** - Models train the behaviors in their own stage graph, including stance and walk, declared in `configs/<species>/<embodiment>/stages.toml`
- **Seven Models of Six Species** - Anatomical models of T-Rex, Velociraptor, Brachiosaurus, Dibothrosuchus, Compsognathus and Allosaurus, plus the Compsognathus robot. Each model has a species and an embodiment (`anatomical` or `robot`)
- **Sim-to-Real Roadmap** - Planned hardware, system-identification, and transfer experiments; not yet validated
- **Open Source** - Fully open codebase for research and education

## Species Catalog and Published Results

The model pages render observation and action dimensions, compiled-model facts,
current curriculum stages, and published result summaries from the generated
species catalog. Each stage row shows its recipe label (stance, recover or walk),
whether it is a published deliverable, and the parent it warm-starts from;
published results list one row per deliverable, headlined by the metric its
gate kind certifies:

- [Velociraptor Mongoliensis (Anatomical)](/docs/models/velociraptor_mongoliensis/anatomical)
- [Tyrannosaurus Rex (Anatomical)](/docs/models/tyrannosaurus_rex/anatomical)
- [Brachiosaurus Altithorax (Anatomical)](/docs/models/brachiosaurus_altithorax/anatomical)
- [Dibothrosuchus Elaphros (Anatomical)](/docs/models/dibothrosuchus_elaphros/anatomical)
- [Compsognathus Longipes (Anatomical)](/docs/models/compsognathus_longipes/anatomical)
- [Compsognathus Longipes (Robot)](/docs/models/compsognathus_longipes/robot)
- [Allosaurus Fragilis (Anatomical)](/docs/models/allosaurus_fragilis/anatomical)

The published result summaries describe historical ladder runs with no
per-deliverable records. They are marked unverified because the original
repository commit, model hash, and config hash were not recorded; they should
not be treated as controlled algorithm comparisons or as results from the
current model revision and configs.

## Quick Links

- [Getting Started](/docs/getting-started/installation) - Set up your development environment
- [Animals](/animals) - Explore anatomical models, robot designs and movement studies
- [Behavior Recipes](/docs/training/recipes) - Train stance, recovery and walk on one certified trunk
- [Training](/docs/training/ppo) - Learn how to train your own dinosaur
- [GitHub](https://github.com/kuds/mesozoic-labs) - View the source code

## Project Status

The project is actively under development. Core infrastructure includes
automated curriculum training, TensorBoard logging, and evaluation metrics.
See the repository roadmap for planned work, including hardware and sim-to-real
experiments that have not yet been validated.
