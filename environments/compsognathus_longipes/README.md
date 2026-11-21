# Compsognathus Longipes

*Compsognathus longipes* was a small, long-tailed bipedal theropod. The species has two models: an anatomical
reconstruction scaled from a fossil, and the Rev B robot, a twelve-servo biped with a fixed head and an unpowered
tail. Both are for learning to stand, to recover from pushes, to walk, and to reach a target marker and stop beside
it. They are research abstractions: the anatomical model is not a validated model of how the animal moved, and the
robot has not been built.

## Models

| | Anatomical | Robot |
|---|---|---|
| Species / embodiment | `compsognathus_longipes` / `anatomical` | `compsognathus_longipes` / `robot` |
| Gymnasium ID | `MesozoicLabs/CompsognathusLongipes-Anatomical-v0` | `MesozoicLabs/CompsognathusLongipes-Robot-v0` |
| Environment | [`CompsognathusLongipesAnatomicalEnv`](anatomical/env.py) | [`CompsognathusLongipesRobotEnv`](robot/env.py) |
| MuJoCo model | [`compsognathus_longipes_anatomical.xml`](anatomical/assets/compsognathus_longipes_anatomical.xml) | [`compsognathus_longipes_robot.xml`](robot/assets/compsognathus_longipes_robot.xml) |
| Stage configs | [`configs/compsognathus_longipes/anatomical/stages.toml`](../../configs/compsognathus_longipes/anatomical/stages.toml) | [`configs/compsognathus_longipes/robot/stages.toml`](../../configs/compsognathus_longipes/robot/stages.toml) |
| Catalog entry | [Compsognathus Longipes (Anatomical)](../../docs/SPECIES_CATALOG.md#compsognathus_longipes-anatomical) | [Compsognathus Longipes (Robot)](../../docs/SPECIES_CATALOG.md#compsognathus_longipes-robot) |
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

This species has no reference-walk script, and neither model has a demonstrated walking gait on its current plant.
The [gait audit](../../docs/investigations/GAIT_AUDIT_2026_09.md) records the earlier certified walkers: an
anatomical walker on physics r1, invalidated by the later anatomical revisions, and a robot locomotion policy that
hops. What exists is standing validation. Run it from the repository root:

```bash
python -m environments.compsognathus_longipes.scripts.validate_models --output /tmp/compsognathus_longipes-preflight.json
python -m environments.compsognathus_longipes.scripts.static_balance --embodiment anatomical --output /tmp/compsognathus-balance.json
```

`validate_models` runs the 26-trial standing protocol on both models: ten seconds at fixed joint targets, ten starts
from perturbed joint angles, a 15% mass-growth case and a motors-off control that must fall. It writes a JSON report
of every trial with an overall `passed` verdict; it needs the `test` extra for SciPy. `static_balance` checks force
and torque balance at the authored `home` pose and again after the model settles, and writes each foot contact, the
center-of-mass projection and each joint's load. The committed results are
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
`--embodiment` has no default. `--motors-off` disables actuation, and `--mass-scale 1.15` scales body masses and
inertias with the motor limits unchanged. The viewer needs a display; on macOS, run it with `mjpython` instead of
`python`. With the `viz` extra (`pip install -e ".[viz]"`) it also renders pictures and videos:

```bash
python -m environments.compsognathus_longipes.scripts.view_model --compare /tmp/models.png
python -m environments.compsognathus_longipes.scripts.view_model --embodiment robot --head-camera --snapshot /tmp/onboard.png
python -m environments.compsognathus_longipes.scripts.view_model --embodiment robot --video /tmp/standing.mp4 --seconds 5
```

`--compare` renders both models side by side and takes no `--embodiment`; `--head-camera` renders the onboard camera.
Headless rendering needs `MUJOCO_GL=egl` or `MUJOCO_GL=osmesa`; model loading and the physics tests need neither.

This species has no `test_env` smoke script. `tests/test_training_env.py` checks both Gymnasium environments: the
Gymnasium contract, resets and steps of every stage across seeds, a full episode at zero action, rejection of invalid
actions and slow, upright target arrival. After editing a model
generator in `scripts/`, `python -m environments.compsognathus_longipes.scripts.build_models` rewrites both MJCFs.

## Training

Both models train from the repository root through the shared command:

```bash
python -m environments.shared.train --species compsognathus_longipes --embodiment anatomical curriculum --algorithm ppo
python -m environments.shared.train --species compsognathus_longipes --embodiment robot curriculum --algorithm ppo
```

It trains stance, then locomotion, then target reaching, each warm-started from its parent in `stages.toml` and run
with the budget of its stage config. To reuse the certified stance and walk of an earlier run and train only what is
missing:

```bash
python -m environments.shared.train --species compsognathus_longipes --embodiment anatomical \
    curriculum --algorithm ppo --trunk-from <earlier_run_dir> --output-dir <new_run_dir>
```

In [`notebooks/sb3_training.ipynb`](../../notebooks/sb3_training.ipynb), set `SPECIES = "compsognathus_longipes"`,
`EMBODIMENT` to `"anatomical"` or `"robot"` and `BEHAVIOR` to `"stance"`, `"recover"`, `"walk"` or `"hunt"`. The
[behavior recipes guide](https://mesozoiclabs.com/docs/training/recipes) covers stopping at a behavior, retraining a
stage and the notebook settings; the [quick start](https://mesozoiclabs.com/docs/getting-started/quick-start) shows
how to evaluate a checkpoint.

**These policies read simulator state, not the head camera.** Both models' policies are MLPs over joint state, pelvis
motion, foot forces and the target's position taken from the simulator, so they cannot be deployed as-is on the
robot. `env.render_head_camera()` returns the 640 × 480 RGB camera image separately. State estimation, sensor noise,
control latency, motor calibration and sim-to-real validation are future work.

Species notes:

- **Recovery.** `recovery` holds the certified stance against horizontal pushes calibrated separately for each
  model. Nothing warm-starts from it, so `curriculum` skips it. Train it in the notebook with
  `BEHAVIOR = "recover"`: the notebook reuses a certified stance from `RUN_DIR` or `TRUNK_FROM` (or trains one),
  freezes the paired null panels before recovery trains, then rolls the policy panel and enforces the recovery
  verdict. [RECOVERY_CALIBRATION.md](RECOVERY_CALIBRATION.md) gives the push, the judge and the pilot targets.
- **Stance.** The anatomical stance passes on a stance-quality check of the floor forces under each foot, rolled at
  six headings after the stage ends (12 panels, about 21 minutes), not on reward. Its bars still admit a soft
  two-foot bounce, a foot lifted for less than half a control step and a fast weight shuttle, so count a trained
  stance's lift-offs by hand before using it ([KNOWN_ISSUES](../../docs/KNOWN_ISSUES.md); the amendment in
  [TRAINING_RECIPE_REVIEW.md](TRAINING_RECIPE_REVIEW.md) says how). The robot's stance passes on full-horizon and
  unsupported-duty bars; its report also writes a heading probe (`stance_heading_probe.txt`) that warns without
  blocking.
- **Exploration noise.** Keep the anatomical stance recipe's `log_std_init` at −2.0. The standing model survives
  PPO's initial action noise but falls under noise not far above it
  ([`tests/test_anatomical_noise_tolerance.py`](tests/test_anatomical_noise_tolerance.py)), so watch the learned
  standard deviation and re-measure after any change to the MJCF or the action mapping.
- **Zero-action baseline.** Zero action already stands. Measure it before reading a return improvement:

  ```bash
  python -m environments.shared.scripts.zero_action_baseline \
      --model-id compsognathus_longipes anatomical --model-id compsognathus_longipes robot --episodes 20
  ```

- **Recipe.** PPO uses a small-update recipe (a low, decaying learning rate, KL early stopping and entropy decay);
  SAC runs on the same budgets, but its learning is not validated. Automatic reward-collapse stopping stays off for
  this species. The notebook trains each stage's full budget before its publication gate; `curriculum` can advance
  early after consecutive passing evaluations.
- **Checkpoints.** SB3 only. Each model has its own plant identity, so a checkpoint from one is refused by the
  other. Keep each checkpoint with its matched VecNormalize file.
- **Integration test.** `pytest environments/shared/tests/test_compsognathus_longipes_training.py` runs short CPU PPO
  and SAC trainings of both models through the shared trainer's stages, the checkpoint handoffs and the notebook's
  training function. It checks the plumbing, not learned behavior.

## Task and rewards

A target marker spawns ahead of the animal at a random distance and side offset. The task is to reach it and stop:
the upright pelvis must enter the target radius in the ground plane at or below a speed limit, with no body part but
the feet on the floor. The environment reports `target_success` and ends the episode. The marker is not touched or
bitten; the robot's head is fixed.

The observation holds the joint positions and velocities, the pelvis gyro and accelerometer, the foot forces, the
target's direction and distance, and a three-value command segment. The command is zero in the stage configs; only
the direction-following and terrain pilots under `configs/compsognathus_longipes/<embodiment>/behaviors/` set it. The
anatomical observation reads no world heading: it uses projected gravity, the pelvis velocity in a gravity-aligned
heading frame and the target direction in pelvis coordinates. The robot reads the world-frame pelvis orientation,
velocity and target direction. Actions are offsets from the `home` keyframe: zero holds the standing pose, and -1 and
+1 reach each servo's limits. The anatomical model's ten leg actions are softened near home (a soft-cubic residual);
its neck, jaw and tail, and every robot servo, map linearly. The docstrings of [`base_env.py`](base_env.py) and
[`anatomical/env.py`](anatomical/env.py) give both layouts.

The main reward groups:

- **Survival and posture:** an alive bonus paid while the feet carry weight, upright posture, pelvis height, the home
  joint pose and a fall penalty.
- **Progress:** forward velocity and progress toward the target, heading alignment and a target-reach bonus.
- **Stability:** drift, sideways-velocity and spin penalties, and an optional gait-symmetry term.
- **Effort:** energy and action smoothness.
- **Stance quality** (anatomical stance and recovery): support under both feet, level pads and the settled stance
  width on loaded feet, the home leg pose and every support geom loaded, and penalties on floor impacts, airborne
  physics substeps, commands held at their limits and action jerk.

The weights are the `CompsognathusLongipesEnv` constructor defaults in `base_env.py` and, for the stance-quality terms,
the `CompsognathusLongipesAnatomicalEnv` defaults in `anatomical/env.py`, where they are off. The `[env]` table of each
stage TOML under `configs/compsognathus_longipes/<embodiment>/` overrides them. An episode ends on a non-finite state,
a pelvis too low or too high, excessive tilt, any body part but the feet touching the floor, or a target reached.
Time limits are truncations.

## Robot embodiment

The robot model preserves the Rev B design of September 6, 2026: a twelve-servo leg mechanism built on material from
the Open Duck Mini v2, under a restyled casing. Each leg has hip yaw, hip roll, hip pitch, knee, ankle pitch and ankle
roll servos; the candidates are two Feetech STS3250 hip-roll servos and ten STS3215 servos. The head and tail are
fixed and unpowered. The robot carries one forward-facing RGB camera in the snout, an IMU in the core, joint encoders
and optional foot-force sensing. `sensors.onboard_readings` returns what the hardware could measure, without
simulator ground truth. The 1.5856 kg mass is an allocation, and the servo gains and torque caps are simulation
assumptions, not identified hardware. `robot/references/` keeps the original Rev B MJCF and the Open Duck Mini
license.

The full design record, with the mass budget, part candidates, sensor layout, actuation and contact assumptions,
validation results and sources, is
[docs/reviews/COMPSOGNATHUS_ROBOT_DESIGN.md](../../docs/reviews/COMPSOGNATHUS_ROBOT_DESIGN.md). Hardware
feasibility and cost plans are in [docs/hardware/](../../docs/hardware/README.md).

## Files

| Path | Contents |
|---|---|
| [`base_env.py`](base_env.py) | `CompsognathusLongipesEnv`, the species base: task, observation, action mapping, rewards, termination and the head camera |
| [`anatomical/env.py`](anatomical/env.py) | `CompsognathusLongipesAnatomicalEnv`: the soft-cubic leg actions, the heading-free observation and the stance-quality terms |
| [`anatomical/assets/`](anatomical/assets/) | The anatomical MJCF |
| [`robot/env.py`](robot/env.py) | `CompsognathusLongipesRobotEnv` |
| [`robot/assets/`](robot/assets/) | The robot MJCF and its leg and casing meshes |
| [`robot/references/`](robot/references/) | The original Rev B MJCF, the Open Duck Mini license and its attribution notice |
| [`model.py`](model.py) | Model-only helpers for the viewer and validation: loading, motors off, visual bounds and floor contacts |
| [`sensors.py`](sensors.py) | `onboard_readings`, the robot's measurable IMU, encoder and foot-force signals |
| [`data/`](data/) | Model parameters, the sensor layout, preflight, static-balance, recovery-calibration and training-validation records, and figures |
| [`scripts/`](scripts/) | The model generators (`build_models.py` and its helpers), the viewer, the standing validation, static balance, a casing-overlap screen, a style render and an early-PPO-update probe |
| [`tests/`](tests/) | Model, environment, reward, noise-tolerance, foot, balance, static-balance, stance-quality, push-recovery, sensor and style tests |
| [`ANATOMY_AND_BALANCE.md`](ANATOMY_AND_BALANCE.md), [`RECOVERY_CALIBRATION.md`](RECOVERY_CALIBRATION.md), [`TRAINING_RECIPE_REVIEW.md`](TRAINING_RECIPE_REVIEW.md), [`TRAINING_VALIDATION.md`](TRAINING_VALIDATION.md) | The species' review and training records (see Further reading) |

## Further reading

- [Anatomy and standing balance](ANATOMY_AND_BALANCE.md): the fossil measurements, the earlier and revised anatomy,
  joint torques and muscle forces, and static balance.
- [Robot design record](../../docs/reviews/COMPSOGNATHUS_ROBOT_DESIGN.md): the Rev B mass budget, servos, sensors,
  actuation and contact assumptions, and the standing validation of both models.
- [Push-recovery calibration](RECOVERY_CALIBRATION.md): the push, the physical judge, the measured fixed-command
  baselines and the pilot targets for both models.
- [Training recipe review](TRAINING_RECIPE_REVIEW.md): why the PPO settings and stage budgets are what they are, the
  next full-run protocol, and the amendment for the anatomical stance gate.
- [Training validation](TRAINING_VALIDATION.md): the SB3 integration checks and the anatomical physics r2 measurements.
- [Stance-hack audit](../../docs/investigations/STANCE_HACK_AUDIT_2026_10.md): the floor-truth stance gate, and in §11
  its adoption on the anatomical model.
- [Heading-free observations](../../docs/HEADING_INVARIANCE_DESIGN_2026_10.md): the design of the anatomical
  observation and of the six-heading stance check.
- [Plant revision notes](../../configs/plant_versions.toml): what changed in each physics and policy-interface
  revision of both models.
- [Direction following and terrain](../../docs/TRAIN_DIRECTION_AND_TERRAIN.md): the pilot behaviors in
  `configs/compsognathus_longipes/<embodiment>/behaviors/`.
