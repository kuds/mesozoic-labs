---
sidebar_position: 1
slug: /
---

# Welcome to Mesozoic Labs

**Mesozoic Labs** is an open-source platform for building robotic dinosaurs using physics simulation and reinforcement learning.

## What is Mesozoic Labs?

We're bringing prehistoric creatures back to life—as robots. Using state-of-the-art reinforcement learning algorithms and accurate physics simulations, we train virtual dinosaurs to walk, run, and move naturally. Then we transfer that learned behavior to real robotic hardware.

## Key Features

- **Accurate Physics Models** - MuJoCo-based simulations of dinosaur anatomy
- **Reinforcement Learning** - PPO and SAC algorithms for training locomotion
- **Multiple Species** - T-Rex, Velociraptor, Brachiosaurus, and more
- **Sim-to-Real** - Transfer learned behaviors to physical robots
- **Open Source** - Fully open codebase for research and education

## Current Results

| Species | Algorithm | Avg Reward | Training Steps |
|---------|-----------|------------|----------------|
| Velociraptor | SAC | 3091.31 | 3.6M |
| Velociraptor | PPO | 319.94 | 2.6M |

All results use 3-stage curriculum learning (balance → locomotion → behavior).

## Quick Links

- [Getting Started](/docs/getting-started/installation) - Set up your development environment
- [Models](/docs/models/trex) - Explore available dinosaur models
- [Training](/docs/training/sac) - Learn how to train your own dinosaur
- [GitHub](https://github.com/kuds/mesozoic-labs) - View the source code

## Project Status

The project is actively under development (currently on Phase 1 — v0.3.0). Core infrastructure is in place including automated curriculum training, W&B experiment tracking, and evaluation metrics. Star us on [GitHub](https://github.com/kuds/mesozoic-labs) to follow our progress!
