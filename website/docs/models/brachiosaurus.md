---
sidebar_position: 3
---

# Brachiosaurus Model

A massive quadrupedal sauropod herbivore optimized for stable four-legged locomotion and food reaching.

## Specifications

| Property | Value |
|----------|-------|
| Height | 2.0m torso, ~4m head reach (simulated) |
| Weight | 205kg (simulated) |
| Joints | 28 |
| Actuators | 22 |
| Observation dim | 75 |
| Action dim | 22 |

## Features

- First quadrupedal species in the project
- Articulated 4-segment neck with head control (6 actuators)
- Front legs longer than rear (characteristic giraffe-like posture)
- Columnar elephant-like legs for stable support
- Food-reaching behavior using neck articulation
- 3-stage curriculum learning (balance, locomotion, food reach)

## Unique Characteristics

Brachiosaurus differs from all other species in the project:

- **Quadrupedal gait** instead of bipedal
- **Herbivore behavior** (food reaching) instead of predatory (hunting/striking)
- **22 actuators** (6 neck + 16 legs) - most complex action space
- **75-dim observation** - largest observation space

## Training Stages

1. **Balance** - Stand on all four legs without falling
2. **Locomotion** - Coordinated quadrupedal walking
3. **Food Reach** - Navigate to food and extend neck

## Diagnostic Metrics (Stage 3)

After each training stage the `eval` command reports:

| Metric | Description |
|--------|-------------|
| `mean_forward_velocity` | Average forward speed (m/s) |
| `gait_symmetry` | Left/right stride symmetry ∈ [0, 1] |
| `stride_frequency` | Step frequency (Hz) |
| `cost_of_transport` | Energy efficiency (lower is better) |
| `mean_pelvis_height` | Torso height stability (m) |
| `success_rate` | Fraction of steps with food-reached event |
| `min_prey_distance` | Closest approach to food (m, `head_food_distance`) |

## Usage

```bash
cd environments/brachiosaurus

# View the model
python scripts/view_model.py

# Train
python scripts/train_sb3.py train --stage 1 --timesteps 1000000
```
