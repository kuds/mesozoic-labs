# Compsognathus longipes

*Compsognathus longipes* was a small, long-tailed bipedal theropod. The species has two models: an anatomical
reconstruction scaled from a fossil, and the Rev B robot, a twelve-servo biped with a fixed head and an unpowered
tail. Both are for learning to stand, to recover from pushes and to walk. They are research abstractions: the
anatomical model is not a validated model of how the animal moved, and the robot has not been built.

## Models

| | Anatomical | Robot |
|---|---|---|
| Species / embodiment | `compsognathus_longipes` / `anatomical` | `compsognathus_longipes` / `robot` |
| Gymnasium ID | `MesozoicLabs/CompsognathusLongipes-Anatomical-v0` | `MesozoicLabs/CompsognathusLongipes-Robot-v0` |
| Environment | [`CompsognathusLongipesAnatomicalEnv`](anatomical/env.py) | [`CompsognathusLongipesRobotEnv`](robot/env.py) |
| MuJoCo model | [`compsognathus_longipes_anatomical.xml`](anatomical/assets/compsognathus_longipes_anatomical.xml) | [`compsognathus_longipes_robot.xml`](robot/assets/compsognathus_longipes_robot.xml) |
| Stage configs | [`configs/compsognathus_longipes/anatomical/stages.toml`](../../configs/compsognathus_longipes/anatomical/stages.toml) | [`configs/compsognathus_longipes/robot/stages.toml`](../../configs/compsognathus_longipes/robot/stages.toml) |
| Catalog entry | [Compsognathus longipes (Anatomical)](../../docs/SPECIES_CATALOG.md#compsognathus_longipes-anatomical) | [Compsognathus longipes (Robot)](../../docs/SPECIES_CATALOG.md#compsognathus_longipes-robot) |
| Model page | [mesozoiclabs.com/docs/models/compsognathus_longipes/anatomical](https://mesozoiclabs.com/docs/models/compsognathus_longipes/anatomical) | [mesozoiclabs.com/docs/models/compsognathus_longipes/robot](https://mesozoiclabs.com/docs/models/compsognathus_longipes/robot) |

Dimensions, stage budgets, gates and published results are generated in the catalog entry; this page does not repeat
them.

## Anatomical review

- **Proportions.** Bone lengths follow the French specimen from Canjuers (CNJ79), scaled to a 95 mm femur. The
  smaller Bavarian holotype has proportionally longer lower legs and is not fitted. The 1 kg mass, the full tail
  length, the soft-tissue outlines and the standing posture are estimates.
- **Joints.** Fourteen position servos drive hip pitch, hip roll, knee, ankle and the metatarsophalangeal (MTP) toe
  joint of each leg, neck pitch, jaw pitch and two axes at the tail base. They apply bounded net joint moments, not
  modeled muscles. The `home` keyframe carries a gravity preload, so zero action holds the standing pose.
- **Feet.** Each foot stands on three separate digits (II, III and IV) and a cushion under the metatarsal heads; a
  short hallux sits above the floor. The digits bend together at the MTP joint, and the pads share one rigid foot body
  with its touch sensor. A passive digital flexor across the ankle and MTP stretches under load and recoils.
- **Tail, neck and arms.** The tail has two driven base axes and three passive distal hinges sprung against gravity.
  The skull has a jaw on its own joint. The folded arms hold a fixed pose.
- **Known limits.** The toes cannot curl, grip or flex one at a time. The trunk is light and the calves heavy for
  their drawn volumes. The center of mass lies outside either foot alone, so lifting a foot needs a weight shift
  first.

The full review, with the fossil measurements, joint torques, a muscle-force check, static balance and sources, is
[ANATOMY_AND_BALANCE.md](ANATOMY_AND_BALANCE.md).

## Reference walk

This species has no reference-walk script, and neither model has a demonstrated walking gait on its current plant. The
[gait audit](../../docs/investigations/GAIT_AUDIT_2026_09.md) covers the earlier certified walkers: an anatomical one
on physics r1, invalidated by later revisions, and a robot policy that hops. What exists is standing validation. Run
it from the repository root:

```bash
python -m environments.compsognathus_longipes.scripts.validate_models --output /tmp/compsognathus_longipes-preflight.json
python -m environments.compsognathus_longipes.scripts.static_balance --embodiment anatomical --output /tmp/compsognathus-balance.json
```

`validate_models` runs the 26-trial standing protocol on both models: ten seconds at fixed joint targets, ten starts
from perturbed joint angles, a 15% mass-growth case and a motors-off control that must fall. It writes a JSON report
with an overall `passed` verdict and needs the `test` extra for SciPy. `static_balance` checks force and torque
balance at `home` and again after the model settles. The committed results are
[`data/preflight_anatomical_r3.json`](data/preflight_anatomical_r3.json) and
[`data/static_balance_r3.json`](data/static_balance_r3.json); the anatomy review and the
[robot design record](../../docs/reviews/COMPSOGNATHUS_ROBOT_DESIGN.md#validation-scope) explain them.

## Run the model

From the repository root:

```bash
pip install -e .              # the environments: MuJoCo, Gymnasium and NumPy
pip install -e ".[train]"     # adds Stable-Baselines3 and PyTorch, for training
pip install -e ".[test]"      # adds pytest and SciPy, for the tests and validate_models

python -m environments.compsognathus_longipes.scripts.view_model --embodiment anatomical
python -m environments.compsognathus_longipes.scripts.view_model --embodiment robot
pytest environments/compsognathus_longipes/tests
```

`view_model` opens MuJoCo's interactive viewer at `home`, holds the home joint targets and steps real physics.
`--embodiment` has no default; `--motors-off` disables actuation, and `--mass-scale 1.15` scales body masses and
inertias with the motor limits unchanged. It needs a display; on macOS, run it with `mjpython` instead of `python`.
With the `viz` extra (`pip install -e ".[viz]"`), `--compare`, `--snapshot` and `--video` render images and videos
instead, headless under `MUJOCO_GL=egl` or `MUJOCO_GL=osmesa`; `--head-camera` views or renders through the 640 × 480
snout camera, and `--help` shows examples.

The species has no smoke-test script of its own, so `tests/test_training_env.py` stands in for one: it checks both
Gymnasium environments' contract, resets and steps of every stage across seeds, a full episode at zero action,
rejection of invalid actions, a scene and a task without a target, and the spawn heading as the forward reference.

## Training

Both models train from the repository root through the shared command:

```bash
python -m environments.shared.train --species compsognathus_longipes --embodiment anatomical curriculum --algorithm ppo
python -m environments.shared.train --species compsognathus_longipes --embodiment robot curriculum --algorithm ppo
```

It trains stance, then locomotion, each warm-started from its parent in `stages.toml` and run with the budget of its
stage config. The target reaching that followed locomotion was removed with every hunt node (decision D-D59). To
reuse the certified stance of an earlier run and train only what is missing:

```bash
python -m environments.shared.train --species compsognathus_longipes --embodiment anatomical \
    curriculum --algorithm ppo --trunk-from <earlier_run_dir> --output-dir <new_run_dir>
```

In [`notebooks/sb3_training.ipynb`](../../notebooks/sb3_training.ipynb), set `SPECIES = "compsognathus_longipes"`,
`EMBODIMENT` to `"anatomical"` or `"robot"` and `BEHAVIOR` to `"stance"`, `"recover"` or `"walk"`. The
[behavior recipes guide](https://mesozoiclabs.com/docs/training/recipes) covers stopping at a behavior, retraining a
stage and the notebook settings; the [quick start](https://mesozoiclabs.com/docs/getting-started/quick-start) shows
how to evaluate a checkpoint.

**These policies read simulator state, not the head camera.** Both models' policies are MLPs over joint state, pelvis
motion and foot forces taken from the simulator (the pelvis velocity among them, which the hardware cannot measure
directly), so they cannot be deployed as-is on the robot. `env.render_head_camera()` returns the 640 × 480 RGB camera image separately. State estimation, sensor noise,
control latency, motor calibration and sim-to-real validation are future work.

Species notes:

- **Recovery.** `recovery` holds the certified stance against horizontal pushes calibrated for each model. Nothing
  warm-starts from it, so `curriculum` skips it; train it in the notebook with `BEHAVIOR = "recover"`, which reuses a
  certified stance first (one behavior a session, D-D79: with none to reuse the session stops before it writes
  anything). [RECOVERY_CALIBRATION.md](RECOVERY_CALIBRATION.md) gives the push, the judge and
  the pilot targets.
- **Stance.** The anatomical stance passes on a stance-quality check of the floor forces under each foot, rolled at
  six headings after the stage ends, not on reward. Its bars still admit a soft two-foot bounce, a foot lifted for
  less than half a control step and a fast weight shuttle, so count a trained stance's lift-offs by hand before using
  it ([KNOWN_ISSUES](../../docs/KNOWN_ISSUES.md); the amendment in
  [TRAINING_RECIPE_REVIEW.md](TRAINING_RECIPE_REVIEW.md) says how). The robot's stance passes on full-horizon and
  unsupported-duty bars.
- **Exploration noise.** Keep the anatomical stance recipe's `log_std_init` at `-2.0`: the standing model survives that
  initial noise but falls under noise not far above it
  ([`tests/test_anatomical_noise_tolerance.py`](tests/test_anatomical_noise_tolerance.py)). Re-measure after any
  change to the MJCF or the action mapping.
- **Recipe.** PPO uses a small-update recipe (a low, decaying learning rate, KL early stopping and entropy decay) that
  the [training recipe review](TRAINING_RECIPE_REVIEW.md) explains. SAC runs on the same budgets but is not validated,
  and automatic reward-collapse stopping stays off for this species.
- **Two plants.** Each model has its own plant identity, so a checkpoint from one is refused by the other.

## Task and rewards

The trained tasks are the stance, recovery from pushes and the walk. The target reach, the hunt that followed the
walk, was removed by decision D-D59, and with it the target marker, its terms and its kwargs: the scene holds the
animal and the floor, and the species declares no success event. Under the stance and recovery, forward, sideways and
heading are measured against the episode's reference heading, the spawn's own forward.

**The walk follows a live command** (decision D-D62); the stance and recovery run no command. Each walk episode draws
one request at reset and holds it for the whole 20 s episode: a forward cruise at 0.04-0.08 m/s (the robot 0.02-0.04
m/s) along a heading up to 0.26 rad (15 degrees) off the spawn heading, or, in about 15% of episodes, an exactly zero
command, which asks the animal to stand still. The command is the observation's last three values. A Gaussian term
pays tracking the commanded velocity and yaw rate (weight 2.5); it subsumes the forward, heading and sideways terms,
which the walk sets to 0, and the terms that read a forward direction measure along the commanded heading. On a zero
command a stand block (`[commands.stand_weights]`, 16 weights on the anatomical model, 6 on the robot) pays the
stance's reward weights, so zero means stand still, not march in place. The commanded speeds ramp from 0.25 of their
end values to the full range over the first 2M training steps. After training the walk is judged on its
`command_tracking/v1` panel: the cruise family's survival and success bounds, the stand family's hold and the biped
gait clause with its flight cap. The bars are provisional, and no controller has yet been shown to clear them; the
tracking width (0.02 m/s, the robot 0.01 m/s) lies under the estimated lateral sway of a stepping gait on these
plants, and neither has a walking controller to measure it (decision D-D62 (16)). The tracking term and the gate score
a moving command on the trailing mean of the last 1.22 s (the robot 1.50 s), a stride estimated from the leg length at
half the cruise speed (`[commands] tracking_window_s`, decision D-D64). That only partly relieves the risk: on a
pendulum model of the sway, the windowed error fits the tolerance for about half of the plausible strides at the
anatomical model's home support width, and for about one in seven on the robot's (decision D-D64 (7)).

Both observations read no world heading (`bipedal-heading-free/v1`): the joint positions and velocities, projected
gravity, the pelvis gyro, the pelvis velocity in a gravity-aligned heading frame, the pelvis accelerometer, one force
channel per foot and the command, 51 values on the anatomical model and 41 on the robot. The anatomical model's foot
channels are its touch sensors. The robot's read each sole's normal force against the floor: its soles collide with
each other, and its touch sensors count a sole resting on the other foot as support. Its reward still reads the touch
sensors ([KNOWN_ISSUES](../../docs/KNOWN_ISSUES.md)). Checkpoints of the anatomical policy interface r6 or earlier, and
of the robot's r3 or earlier, no longer load. Actions are offsets from the `home`
keyframe: zero holds the standing pose, and -1 and +1 reach each servo's limits. The anatomical model's ten leg
actions are softened near home (a soft-cubic residual); its neck, jaw and tail, and every robot servo, map linearly.
The docstrings of [`base_env.py`](base_env.py) and [`anatomical/env.py`](anatomical/env.py) give the layout and
both action mappings.

The main reward groups:

- **Survival and posture:** an alive bonus paid while the feet carry weight, upright posture, pelvis height, the home
  joint pose and a fall penalty.
- **Command tracking** (walk): the Gaussian of the velocity and yaw-rate errors against the command.
- **Progress:** forward velocity and heading alignment, which the walk sets to 0.
- **Stability:** drift, sideways-velocity and spin penalties, and an optional gait-symmetry term.
- **Stance quality** (anatomical stance and recovery, and the anatomical walk's zero command): support under both
  feet, level pads and the settled stance width on loaded feet, the home leg pose and every support geom loaded, and
  penalties on floor impacts, airborne physics substeps, commands held at their limits and action jerk.
- **Effort:** energy and action smoothness.

The weights are the `CompsognathusLongipesEnv` constructor defaults in `base_env.py` and, for the stance-quality
terms, the `CompsognathusLongipesAnatomicalEnv` defaults in `anatomical/env.py`, where they are off. The `[env]` table
of each stage TOML under `configs/compsognathus_longipes/<embodiment>/` overrides them. An episode ends on a
non-finite state, a pelvis too low or too high, excessive tilt or any body part but the feet touching the floor. The
thresholds are in `base_env.py`.

## Robot embodiment

The robot model preserves the Rev B design of September 6, 2026: a twelve-servo leg mechanism built on material from
the Open Duck Mini v2, under a restyled casing. Each leg has hip yaw, hip roll, hip pitch, knee, ankle pitch and ankle
roll servos; the candidates are two Feetech STS3250 hip-roll servos and ten STS3215 servos. The head and tail are
fixed and unpowered. The robot carries one forward-facing RGB camera in the snout, an IMU in the core, joint encoders
and optional foot-force sensing. `sensors.onboard_readings` returns what the hardware could measure, without simulator
ground truth. The 1.5856 kg mass is an allocation, and the servo gains and torque caps are simulation assumptions, not
identified hardware.

The full design record, with the mass budget, part candidates, sensor layout, actuation and contact assumptions,
validation results and sources, is
[docs/reviews/COMPSOGNATHUS_ROBOT_DESIGN.md](../../docs/reviews/COMPSOGNATHUS_ROBOT_DESIGN.md). The hardware
notes, with the sizes tried for every robot concept and the parts and costs of those priced so far, are in
[docs/hardware/](../../docs/hardware/README.md).

## Files

| Path | Contents |
|---|---|
| [`base_env.py`](base_env.py) | `CompsognathusLongipesEnv`, the species base: task, heading-free observation, action mapping, rewards, termination and the head camera |
| [`anatomical/env.py`](anatomical/env.py) | `CompsognathusLongipesAnatomicalEnv`: the soft-cubic leg actions and the stance-quality terms |
| [`anatomical/assets/`](anatomical/assets/) | The anatomical MJCF |
| [`robot/env.py`](robot/env.py) | `CompsognathusLongipesRobotEnv`, with its floor-contact foot channels |
| [`robot/assets/`](robot/assets/) | The robot MJCF and its leg and casing meshes |
| [`robot/references/`](robot/references/) | The original Rev B MJCF, the Open Duck Mini license and its attribution notice |
| [`model.py`](model.py) | Model-only helpers for the viewer and validation: loading, motors off, visual bounds and floor contacts |
| [`sensors.py`](sensors.py) | `onboard_readings`, the robot's measurable IMU, encoder and foot-force signals |
| [`data/`](data/) | Model parameters, the sensor layout, preflight, static-balance, recovery-calibration and training-validation records, and figures |
| [`scripts/`](scripts/) | The viewer (`view_model.py`), the standing validation (`validate_models.py`), static balance (`static_balance.py`), the model generators (`build_models.py`, which rewrites both MJCFs, and its helpers), a casing-overlap screen (`validate_shells.py`), a style render (`render_style_comparison.py`) and an early-PPO-update probe (`probe_ppo_updates.py`) |
| [`tests/`](tests/) | Model, environment, reward, noise-tolerance, foot, balance, static-balance, stance-quality, push-recovery, sensor and style tests |
| [`ANATOMY_AND_BALANCE.md`](ANATOMY_AND_BALANCE.md), [`RECOVERY_CALIBRATION.md`](RECOVERY_CALIBRATION.md), [`TRAINING_RECIPE_REVIEW.md`](TRAINING_RECIPE_REVIEW.md), [`TRAINING_VALIDATION.md`](TRAINING_VALIDATION.md) | The species' review and training records (see Further reading) |

## Further reading

- [Anatomical review](ANATOMY_AND_BALANCE.md): the fossil measurements, the earlier and revised anatomy, joint torques
  and muscle forces, and static balance.
- [Robot design record](../../docs/reviews/COMPSOGNATHUS_ROBOT_DESIGN.md): the Rev B mass budget, servos, sensors,
  actuation and contact assumptions, and the standing validation of both models.
- [Push-recovery calibration](RECOVERY_CALIBRATION.md): the push, the physical judge, the measured fixed-command
  baselines and the pilot targets for both models.
- [Training recipe review](TRAINING_RECIPE_REVIEW.md): why the PPO settings and stage budgets are what they are, the
  next full-run protocol, and the amendment for the anatomical stance gate.
- [Training validation](TRAINING_VALIDATION.md): the SB3 integration checks and the anatomical physics r2 measurements.
- [Stance-hack audit, 2026-10](../../docs/investigations/STANCE_HACK_AUDIT_2026_10.md): the floor-truth stance gate,
  and in §11 its adoption on the anatomical model.
- [Heading-free observations](../../docs/HEADING_INVARIANCE_DESIGN_2026_10.md): the design of the heading-free
  observation, first on the anatomical model, and of the six-heading stance check.
- [Direction following and terrain](../../docs/TRAIN_DIRECTION_AND_TERRAIN.md): the pilot behaviors in
  `configs/compsognathus_longipes/<embodiment>/behaviors/`, trained by
  `python -m environments.shared.train_behaviors`.
- [Plant revisions](../../configs/plant_versions.toml): what changed in each physics, policy-interface and visual
  revision of both models, tracked under the [plant contract](../../docs/PLANT_CONTRACT.md).
