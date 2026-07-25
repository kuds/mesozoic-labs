# Dibothrosuchus elaphros Environment

A small, gracile crocodylomorph locomotion and snout-contact environment built with MuJoCo and Gymnasium.

## Overview

*Dibothrosuchus elaphros* is a sphenosuchian crocodylomorph from the Early Jurassic Lower Lufeng Formation of Yunnan,
and it is the first non-dinosaur in this repository. It is the counterexample to the usual crocodilian picture: a
long-legged terrestrial animal that carried its limbs under its body in an erect, parasagittal posture rather than in
a sprawl, with hindlimbs noticeably longer than its forelimbs. The species name means "light" or "nimble"; the genus
name means "two-trough crocodile", after the paired paramedian rows of dorsal osteoderms.

The Stage 3 task is named "Snap", but the implemented success event is contact between a fixed `snout_snap` geom and
the prey. The model has neck and head actuators but no articulated or actuated jaw, so the metric should be read as a
snout-contact proxy rather than simulated bite biomechanics. The prey sphere is 0.06 m across — the smallest stage-3
target in the repository — which makes the aiming gradients, not the approach, the hard part of the stage.

The model is a research abstraction, not a validated reconstruction.

## Generated Specifications, Curriculum, and Results

The authoritative public dimensions, current stage budgets, success criterion, and provenance-labelled historical
results are in the generator-managed [Dibothrosuchus catalog entry](../../README.md#dibothrosuchus). They are derived
from the species manifest, executable environment, compiled MJCF, current TOML stage configs, and result summaries.

No training runs have been published for this species yet, so the catalog entry carries no results table.

## Implementation Notes

### Body Structure
- **Trunk**: A level root frame with a nose-down sloping capsule, a low-slung belly, and two paramedian rows of
  collision-disabled dorsal scutes (6 pairs) that carry mass but never collide
- **Neck/Skull**: Long slender neck (pitch, yaw), a narrow crested skull (head pitch), a sagittal crest, and an
  elongate snout carrying the fixed snap contact geom
- **Legs**: Four erect limbs, 5 joints each (hip pitch/roll, knee, ankle, toe) ending in a flat pad. The acetabulum
  sits 22 mm below the trunk frame and the glenoid 72 mm below it, so the forelimb chain spans 0.242 m of ground
  clearance against the hindlimb's 0.292 m
- **Tail**: 4 segments, 3 actuated (pitch 1, yaw 1, pitch 2, pitch 3 — segment 4 is passive), carried clear of the
  ground in the high walk

### Home Pose
Unlike the other species, the stance geometry is authored into the body offsets rather than into joint references:
**every hinge joint is 0 at the home keyframe**, and so is the home control vector. Holding that control for 1500
steps leaves the trunk at 0.3129 m (0.6 mm of settle) with forward_z +0.003 and tilt 0.003 rad, so `natural_pitch` is
0 and both the posture and nosedive rewards keep their world-vertical reference. The four pads carry the full 84.9 N
of weight, split 71/29 rear/front.

### Reward Components
- **Forward velocity** — Movement toward the prey target
- **Alive bonus** — Survival reward
- **Energy penalty** — Penalizes excessive actuator use
- **Gait stability** — Penalizes trunk angular velocity
- **Gait symmetry** — Rewards diagonal-pair (FR+RL / FL+RR) touchdown alternation
- **Tail stability** — Penalizes tail-tip angular velocity
- **Snap bonus** — Large reward when the fixed snout contact geom touches prey
- **Approach shaping** — Reward for closing **snout**-to-prey distance. Measured from the snout rather than the trunk:
  the 0.16 m skull rides a mobile neck, so trunk-referenced shaping would keep paying out after the snout had
  already overshot
- **Snout proximity** — Continuous last-mile aiming gradient for snout placement

## Quick Start

```bash
# Install from the repository root
pip install -e ".[all]"

# Run environment tests
python -m pytest environments/dibothrosuchus/tests/ -v

# Train stage 1 using its current TOML-configured budget
python environments/dibothrosuchus/scripts/train_sb3.py train --stage 1

# Or through the shared entry point
python -m environments.shared.train --species dibothrosuchus train --stage 1

# View the model (requires display)
python environments/dibothrosuchus/scripts/view_model.py
```

## Environment Details

Observation and action totals are generated in the catalog entry linked above. The source of the observation layout
and action-to-actuator mapping is `envs/dibothrosuchus_env.py`; actions are normalized to [-1, 1] residuals, where
zero commands the complete XML `home` control and the endpoints retain access to the full actuator ranges.

### Termination Conditions
- Trunk height outside the healthy range (0.18 m–0.55 m by default; the lower bound is the sprawl-collapse gate)
- Excessive tilt angle
- Nosedive (forward pitch exceeds the measured level neutral plus a threshold)
- Snout tip below 0.04 m (catches snout-propping that geom contact detection can miss)
- Skull, snout, trunk, belly, or distal tail contacts the ground
- Snap success (fixed `snout_snap` geom contacts prey; no jaw articulation)
- Episode length > max_episode_steps

### Plant Characterization
The zero-action baseline
(`python environments/shared/scripts/zero_action_baseline.py dibothrosuchus --sweep-noise`) is what calibrates the
Stage 1 reset noise: at `reset_noise_scale=0.05` a do-nothing policy reaches the full horizon in 100% of episodes, so
the stage would reward a statue. The committed 0.14 leaves 62%, with failures dominated by real balance losses.

Actuator `forcerange` is sized from the repository's documented stage-2 failure mode: hip pitch, knee, and ankle carry
1.5x-kp headroom on **every** leg (a 1.24 m animal strides fast relative to its body length, so all three stance
joints see gait-scale torque), and everything else keeps a 0.7x-kp spike cap. Measured clipping is 0.0% under
home-control settling, a 1.5 Hz walk, and a 2.5 Hz trot, and 0.5% worst-case under sprint-like 4 Hz full-amplitude
excitation. Run `python environments/shared/scripts/actuator_saturation_report.py dibothrosuchus` for the current
per-actuator numbers.

## Files

```
dibothrosuchus/
├── assets/
│   └── dibothrosuchus.xml          # MuJoCo MJCF model
├── envs/
│   ├── __init__.py
│   └── dibothrosuchus_env.py       # Gymnasium environment
├── mjx_config.py                   # JAX/MJX species registration
├── scripts/
│   ├── view_model.py               # MuJoCo passive viewer
│   ├── test_actuators.py           # Test joint movements
│   ├── test_env.py                 # Environment validation script
│   └── train_sb3.py                # SB3 PPO training with curriculum
├── tests/
│   ├── test_dibothrosuchus_env.py      # Species-specific env tests
│   ├── test_dibothrosuchus_rewards.py  # Species-specific reward tests
│   ├── test_static_balance.py          # Home-pose and mass-distribution tests
│   └── test_actuator_bounds.py         # Bounded-actuator plant characterization
└── README.md
```

Hyperparameter configs are at `configs/dibothrosuchus/` in the repo root.
