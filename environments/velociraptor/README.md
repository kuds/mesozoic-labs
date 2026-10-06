# Velociraptor Mongoliensis Environment

A bipedal dinosaur locomotion and predatory strike environment built with MuJoCo and Gymnasium.

## Generated Specifications, Curriculum, and Results

The authoritative public dimensions, current stage budgets, success criterion, and provenance-labelled historical
results are in the generator-managed [Velociraptor Mongoliensis catalog entry](../../docs/SPECIES_CATALOG.md#velociraptor). They are derived from
the species manifest, executable environment, compiled MJCF, current TOML stage configs, and result summaries.

## Project Structure

```
velociraptor/
├── assets/
│   └── raptor.xml              # MJCF model definition
├── envs/
│   ├── __init__.py
│   └── raptor_env.py           # Gymnasium environment
├── scripts/
│   ├── view_model.py           # Passive viewer for MJCF iteration
│   ├── test_actuators.py       # Test joint movements
│   ├── test_env.py             # Verify environment works
│   └── train_sb3.py            # Training with Stable-Baselines3
├── tests/
│   ├── test_raptor_env.py      # Species-specific env tests
│   ├── test_raptor_rewards.py  # Species-specific reward tests
│   └── test_static_balance.py  # Static balance tests
└── README.md
```

Hyperparameter configs are at `configs/velociraptor/` in the repo root.

## Installation

```bash
# Install from the repository root
pip install -e ".[all]"
```

## Quick Start

Run the commands in this section from the species directory:

```bash
cd environments/velociraptor
```

### 1. View the Model

First, verify the MJCF loads correctly:

```bash
python scripts/view_model.py
```

This opens a passive viewer. Check that:
- The raptor settles into a stable stance, toes flat on the floor
- No body parts explode or clip through each other
- The tail oscillates briefly then stabilizes

### 2. Test Actuators

See all joints move through their ranges:

```bash
python scripts/test_actuators.py
```

### 3. Test Environment

Run the environment test suite:

```bash
python scripts/test_env.py
python scripts/test_env.py --render  # With visualization
```

### 4. Train with Curriculum Learning

Run the three advancing stages — each warm-started from its declared parent
(`configs/velociraptor/stages.toml`: stance → locomotion → behavior, all
published deliverables) — using the current TOML-configured budgets:

```bash
python scripts/train_sb3.py curriculum --algorithm ppo
```

To reuse the certified stance and locomotion of an earlier run and train only
the strike leaf (the notebook's `TRUNK_FROM`):

```bash
python scripts/train_sb3.py curriculum --algorithm ppo --trunk-from logs/<earlier_run> --output-dir logs/<new_run>
```

See [docs/BEHAVIOR_RECIPES_PLAN.md](../../docs/BEHAVIOR_RECIPES_PLAN.md) for
the behavior recipes and the reuse rule.

### 5. Evaluate Trained Policy

```bash
python scripts/train_sb3.py eval logs/<run_dir>/03_behavior/models/stage3_final.zip
```

## Environment Details

Observation and action totals are generated in the catalog entry linked above. The source of the observation layout and
action-to-actuator mapping is `envs/raptor_env.py`. Actions are normalized residuals around the named XML `home`
keyframe: zero commands the standing pose, while -1 and +1 still reach each actuator's lower and upper limits through
piecewise-linear interpolation. The evidence and compatibility rationale are in the
[Stage-1 basin investigation](../../docs/investigations/VELOCIRAPTOR_STAGE1_BASIN_INVESTIGATION.md).

Since physics r3 (`configs/plant_versions.toml` note 14, decision D-D25) the home keyframe is the standing equilibrium:
every leg spring is anchored at it (`springref`), the ankle and toes put the toes flat at 0.5 mm of floor contact, and
the home controls carry the gravity preload, so the nominal leg servos hold the stance (0.2-5.8% of their forcerange)
with or without the springs, and the statue settles at the 0.35 rad natural lean. Each foot's contact observation is
the sum of three touch sensors, on the middle toe, the outer toe and the metatarsus, and equals the floor force under
the foot. Because the frozen MJX registration cannot mirror that sum, the velociraptor is SB3-only since its policy
interface revision 11; it has no `mjx_config.py`.

### Stance gate
The stance stage certifies under `stance_quality/v2` (decision D-D25): each episode of the 40-episode panel (seeds
3042-3081) is classified on floor truth, the floor's normal force under each leg on every physics substep, and the
stance passes when the one-sided 95% lower bound on clean episodes reaches 0.80 (37 of 40). An episode is clean when
it reaches the horizon with no hop or stomp in the settle window, both feet down and loaded, every support geom of
each foot (both toes and the metatarsal head) loaded, no splay, drift or chatter and no actuator held at its limit; the
bars and their measured provenance are in `configs/velociraptor/stage1_balance.toml`. The verdict is the post-stage
`stance_gate_report.json` on the handoff checkpoint pair, never an in-training evaluation.

### Reward Components

Reward weights vary by stage. The `[env]` section of each
`configs/velociraptor/stage*.toml` file is authoritative; this README does not
copy numeric weights. Components include locomotion, survival, posture, energy,
tail stability, approach shaping, target contact, and fall penalties. Posture
shaping is direction-aware and centred on the raptor's natural forward lean;
absolute tilt remains the safety signal for termination. The stance stage also
pays bilateral support (the weaker-loaded foot), retention of the home leg pose,
and charges commands parked at their limits and command chatter (action jerk),
with part of the alive bonus conditioned on support; each is a `RaptorEnv` kwarg
that is inert at its default and set in `configs/velociraptor/stage1_balance.toml`.

### Termination Conditions
- Pelvis height < 0.25m (fallen)
- Pelvis height > 1.0m (launched into air)
- Torso contacts ground
- Episode length > max_episode_steps

## Tuning Guide

### MJCF Model (`assets/raptor.xml`)

**If the raptor falls immediately:**
- Increase `damping` on leg joints
- Adjust initial pose (qpos0) to more stable crouch
- Check CoM is over the feet

**If movements are jerky:**
- Reduce actuator `kp` gains
- Increase `damping`
- Reduce control frequency (increase `frame_skip`)

**If the tail flops around:**
- Increase tail joint `stiffness` and `damping`
- Reduce tail joint `range`

### Reward Weights (`envs/raptor_env.py`)

**If it doesn't learn to walk:**
- Increase `forward_vel_weight`
- Decrease `energy_penalty_weight`
- Check that alive_bonus isn't dominating

**If it walks but falls a lot:**
- Inspect the episode-length distribution and termination mix before changing weights
- Check whether `forward_z` tracks the natural lean; do not reward world-vertical posture for this morphology
- Lower reset noise only if a controlled probe isolates reset perturbations as the failure source

**If it ignores the prey:**
- Add proximity reward (bonus for getting closer)
- Reduce `prey_distance_range` to spawn prey closer

## Troubleshooting

**"No module named 'envs'"**
Run scripts from the species directory:
`cd environments/velociraptor && python scripts/test_env.py`.

**Viewer doesn't open**
Install a display backend: `pip install glfw` or run with `MUJOCO_GL=egl` for headless.

**Training is slow**
- Use more parallel envs: `--n-envs 8`
- Use subprocess vectorization: `--subproc`
- Reduce evaluation frequency: `--eval-freq 50000`

**NaN in observations**
- Physics is exploding; reduce timestep or actuator gains
- Check for division by zero in reward computation
