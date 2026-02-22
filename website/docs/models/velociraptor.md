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
| Hinge joints | 20 |
| Ball joints | 2 (shoulders, passive) |
| Actuators | 17 |
| Observation dim | 73 |
| Action dim | 17 |

## Anatomy

- **Torso** - Elongated body with simplified head/neck
- **Legs** - Digitigrade hind limbs (hip pitch/roll, knee, ankle, 2 toe digits per leg)
- **Sickle claws** - Retractable claw on digit 2 of each foot (the weapon)
- **Arms** - Stub forelimbs (passive, not actuated)
- **Tail** - 5-segment stiff counterbalance

## Action Space (17 dims)

| Index | Actuator | Type |
|-------|----------|------|
| 0-5 | Right leg (hip pitch/roll, knee, ankle, toe d3/d4) | Position |
| 6 | Right sickle claw | Motor |
| 7-12 | Left leg (hip pitch/roll, knee, ankle, toe d3/d4) | Position |
| 13 | Left sickle claw | Motor |
| 14-16 | Tail (pitch 1, yaw 1, pitch 2) | Position |

## Training Stages

1. **Balance** - Learn to stand without falling
2. **Locomotion** - Walk and run forward
3. **Strike** - Sprint and attack prey with sickle claws

## Usage

```bash
cd environments/velociraptor

# View the model
python scripts/view_model.py

# Train stage 1
python scripts/train_sb3.py train --stage 1 --timesteps 1000000
```
