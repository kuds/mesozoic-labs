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
| Hinge joints | 18 |
| Ball joints | 2 (shoulders, passive) |
| Actuators | 12 |
| Observation dim | 69 |
| Action dim | 12 |

## Anatomy

- **Torso** - Elongated body with simplified head/neck
- **Legs** - Digitigrade hind limbs (hip pitch/roll, knee, ankle, toe per leg)
- **Sickle claws** - Retractable claw on digit 2 of each foot (the weapon)
- **Arms** - Stub forelimbs (passive, not actuated)
- **Tail** - 5-segment stiff counterbalance

## Action Space (12 dims)

| Index | Actuator | Type |
|-------|----------|------|
| 0-4 | Right leg (hip pitch/roll, knee, ankle, toe) | Position |
| 5 | Right sickle claw | Motor |
| 6-10 | Left leg (hip pitch/roll, knee, ankle, toe) | Position |
| 11 | Left sickle claw | Motor |

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
