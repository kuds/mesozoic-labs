---
sidebar_position: 1
---

# T-Rex Model

The Tyrannosaurus Rex is our flagship dinosaur model, featuring accurate bipedal locomotion physics.

## Specifications

| Property | Value |
|----------|-------|
| Height | 4.5m (simulated) |
| Weight | 8000kg (simulated) |
| Joints | 18 |
| Actuators | 12 |
| Observation Dim | 67 |
| Action Dim | 12 |

## Anatomy

The T-Rex model includes:
- **Head** - Mass distribution for balance
- **Torso** - Main body with realistic proportions
- **Arms** - Small forelimbs (cosmetic)
- **Legs** - Powerful hind limbs with 3 joints each
- **Tail** - Counterbalance for bipedal stance

## Training Results

Using SAC algorithm with 3.6M training steps:
- Average Reward: 3091.31
- Training Time: ~4.5 hours (T4 GPU)

:::note Coming Soon
Detailed model documentation is under development.
:::
