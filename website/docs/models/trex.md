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
| Hinge joints | 26 |
| Ball joints | 2 (shoulders, passive) |
| Actuators | 18 |
| Observation dim | 85 |
| Action dim | 18 |

## Anatomy

The T-Rex model includes:
- **Torso** - Forward-leaning body (~30 deg from horizontal) with ribcage and belly
- **Neck + Skull** - Short muscular neck with massive elongated skull and brow ridges
- **Jaw** - Articulated lower mandible with bite contact geom
- **Legs** - Powerful digitigrade hind limbs (hip pitch/roll, knee, ankle, 3 toe digits per leg)
- **Arms** - Tiny vestigial forelimbs with 2-fingered hands (passive, not actuated)
- **Tail** - 5-segment heavy counterbalance to skull

## Action Space (18 dims)

| Index | Actuator | Type |
|-------|----------|------|
| 0-2 | Neck pitch, neck yaw, head pitch | Position |
| 3 | Jaw | Motor |
| 4-10 | Right leg (hip pitch/roll, knee, ankle, toe d2/d3/d4) | Position |
| 11-17 | Left leg (hip pitch/roll, knee, ankle, toe d2/d3/d4) | Position |

## Training Stages

1. **Balance** - Stable bipedal stance
2. **Locomotion** - Walk and run toward prey
3. **Hunting** - Sprint and bite prey with jaws

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
| `success_rate` | Fraction of steps with jaw-bite event |
| `min_prey_distance` | Closest approach to prey (m) |

## Usage

```bash
cd environments/trex

# View the model
python scripts/view_model.py

# Train stage 1
python scripts/train_sb3.py train --stage 1 --timesteps 1000000
```
