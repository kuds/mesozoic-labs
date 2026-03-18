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

| Species | Algorithm | Stage | Best Eval Reward | Avg Fwd Vel | Training Steps | Time |
|---------|-----------|-------|------------------|-------------|----------------|------|
| T-Rex | PPO | 1 — Balance | 3008.66 +/- 7.62 | 0.02 m/s | 6.0M | 3h 35m |
| T-Rex | PPO | 2 — Locomotion | 1936.01 +/- 13.12 | 3.47 m/s | 8.0M | 5h 17m |
| T-Rex | PPO | 3 — Bite | 1294.28 +/- 67.19 | 1.68 m/s | 8.0M | 4h 10m |
| Velociraptor | PPO | 1 — Balance | 1964.43 +/- 27.39 | 0.11 m/s | 6.0M | 2h 57m |
| Velociraptor | PPO | 2 — Locomotion | 2678.68 +/- 4.07 | 3.47 m/s | 8.0M | 4h 36m |
| Velociraptor | PPO | 3 — Strike | 1366.19 +/- 76.29 | 2.02 m/s | 8.0M | 3h 52m |
| Basic Dinosaur | PPO | Full curriculum | 319.94 | — | 2.6M | 1h 30m |
| Basic Dinosaur | SAC | Full curriculum | 3091.31 | — | 3.6M | — |

All results use 3-stage curriculum learning (balance → locomotion → behavior).

## Quick Links

- [Getting Started](/docs/getting-started/installation) - Set up your development environment
- [Models](/docs/models/trex) - Explore available dinosaur models
- [Training](/docs/training/sac) - Learn how to train your own dinosaur
- [GitHub](https://github.com/kuds/mesozoic-labs) - View the source code

## Project Status

The project is actively under development (currently on Phase 1 — v0.3.0). Core infrastructure is in place including automated curriculum training, W&B experiment tracking, and evaluation metrics. Star us on [GitHub](https://github.com/kuds/mesozoic-labs) to follow our progress!
