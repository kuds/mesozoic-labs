# Tyrannosaurus Rex Environment

A large dinosaur-inspired bipedal locomotion and head-contact environment built with MuJoCo and Gymnasium.

## Overview

This Tyrannosaurus-inspired model has a heavy head, vestigial forelimbs, and a tail used as a counterbalance. The Stage 3 task retains the name "Bite," but the implemented success event is contact between a fixed `head_bite` geom and the prey. The model has neck/head actuators but no articulated or actuated jaw, so the metric should be read as a head-contact proxy rather than simulated biting biomechanics.

## Generated Specifications, Curriculum, and Results

The authoritative public dimensions, current stage budgets, success criterion, and provenance-labelled historical
results are in the generator-managed [Tyrannosaurus Rex catalog entry](../../docs/SPECIES_CATALOG.md#t-rex). They are derived from the species
manifest, executable environment, compiled MJCF, current TOML stage configs, and result summaries.

## Implementation Notes

### Body Structure
- **Head/Neck**: Heavy head with a fixed contact geom, 3 actuated joints (neck pitch/yaw, head pitch), and no jaw joint
- **Legs**: Powerful digitigrade legs with 7 joints each (hip pitch/roll, knee, ankle, toe d2/d3/d4); the 4 proximal joints are actuated, the 3 toe digits ride passive springs. Since physics r8 (`configs/plant_versions.toml` note 13) the hip-roll servos are kp 600 / ±480 N·m, stiff enough that both legs together out-resist the body's lateral sway (2 × (kp + 40) = 1280 against m·g·h = 704 N·m/rad); at kp 150 the zero-action statue stood on a rolled pad in some episodes
- **Tail**: 5 segments, 4 actuated (pitch 1, yaw 1, pitch 2, pitch 3), heavy counterbalance to skull

### Reward Components
- **Forward velocity** - Movement toward prey target
- **Alive bonus** - Survival reward
- **Energy penalty** - Penalizes excessive actuator use
- **Tail stability** - Penalizes tail angular velocity
- **Bite bonus** - Large reward when the fixed head contact geom touches prey
- **Approach shaping** - Reward for closing distance to prey
- **Stance shaping** (stance and recovery) - neck posture against the statue's settled pose (`neck_posture_reference = "settled"`), level plantar pads (`foot_flatness_weight`) and a steady stance width (`stance_width_weight`; since D-D27 centred on the animal's own width at the end of the settle, `stance_width_reference = "settled"`), both foot terms paid only on a loaded foot (`foot_terms_min_support_force`), a floor-impact and an airborne-substep penalty over the whole episode (`floor_impact_weight`, `airborne_substep_weight`), and smoothness, jerk and saturation priced on the policy's own command rather than the 10 Hz-filtered one (`action_penalty_source = "raw"`); each a `TRexEnv` kwarg that is inert at its default and set in `configs/trex/stance.toml`

### Stance gate
The stance stage certifies under `stance_quality/v2` (decision D-D24; its bars revised by D-D27 and D-D28): each
episode of the 40-episode panel (seeds 3042-3081) is classified on floor truth, the floor's normal force under each
leg on every physics substep, and the stance passes when the one-sided 95% lower bound on clean episodes reaches 0.80
(37 of 40). An episode is clean when it reaches the horizon with no hop or stomp in the settle window, both feet down
and loaded, no drift or chatter, no actuator held at its limit, flat pads (tilt, corner lift and contact points)
pushed on along their length rather than on an edge (the pad's fore-aft centre of pressure, D-D28), at most one
re-plant in the settle and no more than 25° of turning; and at most one of the 40 may hop for the whole episode or
fall. The bars and their measured provenance are in `configs/trex/stance.toml`. The verdict is the post-stage
`stance_gate_report.json` on the handoff checkpoint pair, never an in-training evaluation. Beside it the report writes
a heading probe (`stance_heading_probe.txt`: the policy and the statue spawned turned by ±45° and ±90°), which is
report only: the stance stage always spawns at the same heading, and the heading-free observations the maintainer
chose on 2026-10-07, which make the probe a gate check, are a later PR (`docs/KNOWN_ISSUES.md`). The first two
physics-r8 stances (`20261006_185343`, `20261006_185704`) both failed this gate; what they did and what decision D-D27
changed in response is §9 of
[`docs/investigations/STANCE_HACK_AUDIT_2026_10.md`](../../docs/investigations/STANCE_HACK_AUDIT_2026_10.md), and the
calibration of D-D28's centre-of-pressure bar is its §10.

## Quick Start

```bash
# Install from the repository root
pip install -e ".[all]"

# Run environment tests
python -m pytest environments/trex/tests/ -v

# Train stage 1 using its current TOML-configured budget
python environments/trex/scripts/train_sb3.py train --stage 1

# Train the hunt chain (stance -> locomotion -> behavior), reusing an earlier
# run's certified trunk and training only what is missing above it
python environments/trex/scripts/train_sb3.py curriculum --trunk-from <earlier-run-dir> --output-dir <new-run-dir>

# Train the recovery stage — the second node of the `stand` recipe, a published
# deliverable judged post-stage against the frozen recovery_quality/v1 gate
# (configs/trex/recovery.toml, frozen 2026-08-28): the stance task plus scheduled
# 165.5 N / 0.20 s external pushes derived from the plant itself. Warm-start from
# a certified stance checkpoint: the load is refused unless the checkpoint records
# stance, recovery's declared parent. The `curriculum` command skips this
# non-advancing node; the notebook runs it with BEHAVIOR = "stand".
python environments/trex/scripts/train_sb3.py train --stage recovery --load <stance-checkpoint>.zip --load-mode initialize_next_stage

# View the model (requires display)
python environments/trex/scripts/view_model.py
```

Behaviors, their warm-start edges (`configs/trex/stages.toml`) and what a run
reuses are described in [docs/BEHAVIOR_RECIPES_PLAN.md](../../docs/BEHAVIOR_RECIPES_PLAN.md);
the notebook equivalent is `BEHAVIOR` / `TRUNK_FROM` in
[notebooks/sb3_training.ipynb](../../notebooks/sb3_training.ipynb).

For the opt-in PPO direction-following and randomized gentle-terrain pilots,
see [Train direction and terrain](../../docs/TRAIN_DIRECTION_AND_TERRAIN.md).
These use a dedicated runner and separate behavior artifacts.

## Environment Details

Observation and action totals are generated in the catalog entry linked above. The source of the observation layout and
action-to-actuator mapping is `envs/trex_env.py`; actions are normalized to
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
trex/
├── assets/
│   └── trex.xml                # MuJoCo MJCF model
├── envs/
│   ├── __init__.py
│   └── trex_env.py             # Gymnasium environment
├── scripts/
│   ├── view_model.py           # MuJoCo passive viewer
│   ├── test_actuators.py       # Test joint movements
│   ├── test_env.py             # Environment validation script
│   └── train_sb3.py            # SB3 PPO training with curriculum
├── tests/
│   ├── test_trex_env.py        # Species-specific env tests
│   ├── test_trex_rewards.py    # Species-specific reward tests
│   └── test_static_balance.py  # Static balance tests
└── README.md
```

Hyperparameter configs are at `configs/trex/` in the repo root.
