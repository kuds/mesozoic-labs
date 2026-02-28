---
sidebar_position: 2
---

# Velociraptor Model

A bipedal predator with distinctive sickle claws, trained for locomotion and predatory strike behavior.

## Specifications

| Property | Value |
|----------|-------|
| Hip height | ~0.5m (simulated) |
| Total mass | ~15 kg (simulated, scaled) |
| Hinge joints | 24 |
| Actuators | 22 |
| Observation dim | 67 |
| Action dim | 22 |

## Anatomy

- **Torso** - Elongated body with simplified head/neck
- **Legs** - Digitigrade hind limbs (hip pitch/roll, knee, ankle, 2 toe digits per leg)
- **Sickle claws** - Retractable claw on digit 2 of each foot (the weapon)
- **Arms** - Stub forelimbs with shoulder pitch/roll actuators
- **Tail** - 5-segment counterbalance (4 actuated segments)

## Action Space (22 dims)

| Index | Actuator | Type |
|-------|----------|------|
| 0-5 | Right leg (hip pitch/roll, knee, ankle, toe d3/d4) | Position |
| 6 | Right sickle claw | Motor |
| 7-12 | Left leg (hip pitch/roll, knee, ankle, toe d3/d4) | Position |
| 13 | Left sickle claw | Motor |
| 14-17 | Tail (pitch 1, yaw 1, pitch 2, pitch 3) | Position |
| 18-19 | Right arm (shoulder pitch/roll) | Position |
| 20-21 | Left arm (shoulder pitch/roll) | Position |

## Training Stages

1. **Balance** - Learn to stand without falling
2. **Locomotion** - Walk and run forward
3. **Strike** - Sprint and attack prey with sickle claws

## Diagnostic Metrics (Stage 3)

After each training stage the `eval` command reports:

| Metric | Description |
|--------|-------------|
| `mean_forward_velocity` | Average forward speed (m/s) |
| `gait_symmetry` | Left/right stride symmetry ∈ [0, 1] |
| `stride_frequency` | Step frequency (Hz) |
| `cost_of_transport` | Energy efficiency (lower is better) |
| `mean_pelvis_height` | Upright stability (m) |
| `mean_heading_alignment` | cos θ toward prey ∈ [-1, 1] |
| `success_rate` | Fraction of steps with sickle-claw strike |
| `min_prey_distance` | Closest approach to prey (m) |

## Usage

```bash
cd environments/velociraptor

# View the model
python scripts/view_model.py

# Train stage 1
python scripts/train_sb3.py train --stage 1 --timesteps 1000000
```
