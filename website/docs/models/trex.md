---
sidebar_position: 1
---

# T-Rex Model

The Tyrannosaurus Rex is a large bipedal predator with a massive skull, powerful jaws, and vestigial forelimbs. It hunts by sprinting toward prey and delivering a bite.

## Specifications

| Property | Value |
|----------|-------|
| Hip height | ~1.1m (simulated) |
| Total mass | ~80 kg (simulated, scaled) |
| Hinge joints | 22 |
| Ball joints | 2 (shoulders, passive) |
| Actuators | 14 |
| Observation dim | 77 |
| Action dim | 14 |

## Anatomy

The T-Rex model includes:
- **Torso** - Forward-leaning body (~30 deg from horizontal) with ribcage and belly
- **Neck + Skull** - Short muscular neck with massive elongated skull and brow ridges
- **Jaw** - Articulated lower mandible with bite contact geom
- **Legs** - Powerful digitigrade hind limbs (hip pitch/roll, knee, ankle, toe per leg)
- **Arms** - Tiny vestigial forelimbs with 2-fingered hands (passive, not actuated)
- **Tail** - 5-segment heavy counterbalance to skull

## Action Space (14 dims)

| Index | Actuator | Type |
|-------|----------|------|
| 0-2 | Neck pitch, neck yaw, head pitch | Position |
| 3 | Jaw | Motor |
| 4-8 | Right leg (hip pitch/roll, knee, ankle, toe) | Position |
| 9-13 | Left leg (hip pitch/roll, knee, ankle, toe) | Position |

## Training Stages

1. **Balance** - Stable bipedal stance
2. **Locomotion** - Walk and run toward prey
3. **Hunting** - Sprint and bite prey with jaws

## Usage

```bash
cd environments/trex

# View the model
python scripts/view_model.py

# Train stage 1
python scripts/train_sb3.py train --stage 1 --timesteps 1000000
```
