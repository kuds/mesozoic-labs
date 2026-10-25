# Tyrannosaurus Rex Environment

A large dinosaur-inspired bipedal locomotion and head-contact environment built with MuJoCo and Gymnasium.

## Overview

This Tyrannosaurus-inspired model has a heavy head, vestigial forelimbs, and a tail used as a counterbalance. The Stage 3 task retains the name "Bite," but the implemented success event is contact between a fixed `head_bite` geom and the prey. The model has neck/head actuators but no articulated or actuated jaw, so the metric should be read as a head-contact proxy rather than simulated biting biomechanics.

## Generated Specifications, Curriculum, and Results

The authoritative public dimensions, current stage budgets, success criterion, and provenance-labelled historical
results are in the generator-managed [Tyrannosaurus Rex (Anatomical) catalog entry](../../docs/SPECIES_CATALOG.md#tyrannosaurus_rex-anatomical). They are derived from the species
manifest, executable environment, compiled MJCF, current TOML stage configs, and result summaries.

## Models

The species has one model, the pair (species, embodiment) of
[docs/SPECIES_NAMING.md](../../docs/SPECIES_NAMING.md):

| | Anatomical |
|---|---|
| Species / embodiment | `tyrannosaurus_rex` / `anatomical` |
| Environment class | `TyrannosaurusRexAnatomicalEnv` (`anatomical/env.py`), on the species base `TyrannosaurusRexEnv` (`base_env.py`) |
| Gymnasium ID | `MesozoicLabs/TyrannosaurusRex-Anatomical-v0` |
| MJCF | `anatomical/assets/tyrannosaurus_rex_anatomical.xml` |
| Configs | `configs/tyrannosaurus_rex/anatomical/` |
| Catalog | [Tyrannosaurus Rex (Anatomical)](../../docs/SPECIES_CATALOG.md#tyrannosaurus_rex-anatomical) |

## Implementation Notes

### Body Structure
- **Head/Neck**: Heavy head with a fixed contact geom, 3 actuated joints (neck pitch/yaw, head pitch), and no jaw joint
- **Legs**: Powerful digitigrade legs with 7 joints each (hip pitch/roll, knee, ankle, toe d2/d3/d4); the 4 proximal joints are actuated, the 3 toe digits ride passive springs. Since physics r8 (`configs/plant_versions.toml` note 13) the hip-roll servos are kp 600 / ±480 N·m, stiff enough that both legs together out-resist the body's lateral sway (2 × (kp + 40) = 1280 against m·g·h = 729 N·m/rad on physics r10, 1.76×; 704.6 on r9 and 704.4 on r8); at kp 150 the zero-action statue stood on a rolled pad in some episodes
- **Contacts**: an elliptic friction cone since physics r9 (note 16), so the box plantar pads, whose contact frames stay on the world axes, grip the same in every direction and every world heading is physically identical; under MuJoCo's default pyramidal cone a 45° turn changed the physics slightly
- **Tail**: 5 segments, 4 actuated (pitch 1, yaw 1, pitch 2, pitch 3), heavy counterbalance to skull. Since physics r10 (`configs/plant_versions.toml` note 20, decision D-D44) the three pitch springs are preloaded (`springref` 52.5 / 25.9 / 10.4°), so the passive tail holds the keyframe's level pose with the servos near zero; through r9 the springs rested at zero and the zero-action statue settled `tail_1` on its −12° ventral stop, `tail_2` just above its own. The stance's `tail_home_pose` term targets the settled tail (`TyrannosaurusRexEnv._TAIL_SETTLED_QPOS`, the keyframe zeros to within 0.2 mrad on r10)
- **Observation**: heading-free since policy interface r14 (`bipedal-target-heading-free/v1`, decision D-D31, `configs/plant_versions.toml` note 18): projected gravity in place of the world pelvis quaternion, the pelvis velocity in the gravity-aligned heading frame (forward, left, up) and the prey direction in pelvis coordinates, all from the free root's quaternion (`environments/shared/heading_observation.py`), beside the joint state, the site-frame gyro and accelerometer, the two foot-contact sums, the prey distance and the trailing command; no channel reads the world heading. The T-Rex is SB3-only from r14 (its frozen MJX registration was deleted), and every r13 checkpoint is refused

### Reward Components
- **Forward velocity** - Movement toward prey target
- **Alive bonus** - Survival reward
- **Energy penalty** - Penalizes excessive actuator use
- **Tail stability** - Penalizes tail angular velocity
- **Bite bonus** - Large reward when the fixed head contact geom touches prey
- **Approach shaping** - Reward for closing distance to prey
- **Stance shaping** (stance and recovery) - neck posture against the statue's settled pose (`neck_posture_reference = "settled"`), level plantar pads (`foot_flatness_weight`) and a steady stance width (`stance_width_weight`; since D-D27 centred on the animal's own width at the end of the settle, `stance_width_reference = "settled"`), both foot terms paid only on a loaded foot (`foot_terms_min_support_force`), a floor-impact and an airborne-substep penalty over the whole episode (`floor_impact_weight`, `airborne_substep_weight`), and smoothness, jerk and saturation priced on the policy's own command rather than the 10 Hz-filtered one (`action_penalty_source = "raw"`); each a `TyrannosaurusRexEnv` kwarg that is inert at its default and set in `configs/tyrannosaurus_rex/anatomical/stance.toml`. Two further options are off by default and set by no stage: the standing lateral-lean bonus (`lateral_lean_weight`, `lateral_lean_tolerance_m`) and `leg_home_pose_reference = "settled"`; see `website/docs/training/mirror-symmetry.md`

### Stance gate
The stance stage certifies under the six-heading certificate `stance_quality_heading/v1` (decision D-D32), which
applies the `stance_quality/v2` decision (decision D-D24; its bars revised by D-D27 and D-D28) at each of six
whole-scene headings. Under v2 each
episode of the 40-episode panel (seeds 3042-3081) is classified on floor truth, the floor's normal force under each
leg on every physics substep, and the stance passes when the one-sided 95% lower bound on clean episodes reaches 0.80
(37 of 40). An episode is clean when it reaches the horizon with no hop or stomp in the settle window, both feet down
and loaded, no drift or chatter, no actuator held at its limit, flat pads (tilt, corner lift and contact points)
pushed on along their length rather than on an edge (the pad's fore-aft centre of pressure, D-D28), at most one
re-plant in the settle and no more than 25° of turning; and at most one of the 40 may hop for the whole episode or
fall. The bars and their measured provenance are in `configs/tyrannosaurus_rex/anatomical/stance.toml`. The stance stage always spawns at
the same heading, so the certificate rolls the panel again with the whole scene (the animal, the prey and the heading
reward's reference direction) turned −90°, −45°, 45°, 90° and 180°, each heading against the statue rolled at it, and
passes only when every heading passes v2 and the lower bound on seeds clean at every heading reaches 0.80: the seed-52
stance `20261009_155723`, which passed v2 and falls when turned −45° or −90°, fails it. The heading-free observations
the maintainer chose on 2026-10-07 are carried out too (policy interface r14, above), so a whole-scene turn no longer
changes what the policy sees, and a heading that fails where 0° passes points at rounding the policy's closed loop
amplifies (`docs/KNOWN_ISSUES.md`). The verdict is the post-stage `stance_heading_report.json` (and its grid,
`stance_heading_panel_selected.csv`) on the handoff checkpoint pair, never an in-training evaluation; it costs 12
panels, about 13 minutes. It replaces the report-only heading probe (D-D27), which the stance no longer declares. The
first two
physics-r8 stances (`20261006_185343`, `20261006_185704`) both failed this gate; what they did and what decision D-D27
changed in response is §9 of
[`docs/investigations/STANCE_HACK_AUDIT_2026_10.md`](../../docs/investigations/STANCE_HACK_AUDIT_2026_10.md), and the
calibration of D-D28's centre-of-pressure bar is its §10.

## Quick Start

```bash
# Install from the repository root
pip install -e ".[all]"

# Run environment tests
python -m pytest environments/tyrannosaurus_rex/tests/ -v

# Train stage 1 using its current TOML-configured budget
python -m environments.shared.train --species tyrannosaurus_rex --embodiment anatomical train --stage 1

# Train the hunt chain (stance -> locomotion -> behavior), reusing an earlier
# run's certified trunk and training only what is missing above it
python -m environments.shared.train --species tyrannosaurus_rex --embodiment anatomical \
    curriculum --trunk-from <earlier-run-dir> --output-dir <new-run-dir>

# Train the recovery stage — the node of the `recover` recipe, a published
# deliverable judged post-stage against the frozen recovery_quality/v1 gate
# (configs/tyrannosaurus_rex/anatomical/recovery.toml, frozen 2026-08-28): the stance task plus scheduled
# 165.5 N / 0.20 s external pushes derived from the plant itself. Warm-start from
# a certified stance checkpoint: the load is refused unless the checkpoint records
# stance, recovery's declared parent. The `curriculum` command skips this
# non-advancing node; the notebook runs it with BEHAVIOR = "recover".
python -m environments.shared.train --species tyrannosaurus_rex --embodiment anatomical \
    train --stage recovery --load <stance-checkpoint>.zip --load-mode initialize_next_stage

# View the model (requires display)
python environments/tyrannosaurus_rex/scripts/view_model.py --embodiment anatomical
```

Behaviors, their warm-start edges (`configs/tyrannosaurus_rex/anatomical/stages.toml`) and what a run
reuses are described in [docs/BEHAVIOR_RECIPES_PLAN.md](../../docs/BEHAVIOR_RECIPES_PLAN.md);
the notebook equivalent is `BEHAVIOR` / `TRUNK_FROM` in
[notebooks/sb3_training.ipynb](../../notebooks/sb3_training.ipynb).

For the opt-in PPO direction-following and randomized gentle-terrain pilots,
see [Train direction and terrain](../../docs/TRAIN_DIRECTION_AND_TERRAIN.md).
These use a dedicated runner and separate behavior artifacts.

## Environment Details

Observation and action totals are generated in the catalog entry linked above. The source of the observation layout and
action-to-actuator mapping is `environments/tyrannosaurus_rex/base_env.py`; actions are normalized to
[-1, 1] residuals, where zero commands the complete XML `home` control and
the endpoints retain access to the full actuator ranges.

### Termination Conditions
- Pelvis height outside healthy range (0.5m–1.6m)
- Excessive tilt angle
- Nosedive (forward pitch exceeds natural lean + threshold)
- Head/torso/tail contacts ground
- Bite success (fixed `head_bite` geom contacts prey; no jaw articulation)
- Episode length > max_episode_steps

## Files

```
tyrannosaurus_rex/
├── __init__.py
├── base_env.py                 # TyrannosaurusRexEnv, the species base class
├── anatomical/
│   ├── __init__.py
│   ├── env.py                  # TyrannosaurusRexAnatomicalEnv and its body data
│   └── assets/
│       └── tyrannosaurus_rex_anatomical.xml   # MuJoCo MJCF model
├── scripts/
│   ├── view_model.py           # MuJoCo passive viewer
│   ├── test_actuators.py       # Test joint movements
│   └── test_env.py             # Environment validation script
├── tests/
│   ├── test_anatomical_env.py        # Model-specific env tests
│   ├── test_anatomical_rewards.py    # Model-specific reward tests
│   ├── test_anatomical_actuator_bounds.py  # Actuator bound tests
│   └── test_anatomical_static_balance.py  # Static balance tests
├── requirements.txt
└── README.md
```

Hyperparameter configs are at `configs/tyrannosaurus_rex/anatomical/` in the repo root
(`stance.toml`, `recovery.toml`, `locomotion.toml`, `behavior.toml`, `stages.toml` and `behaviors/`).
