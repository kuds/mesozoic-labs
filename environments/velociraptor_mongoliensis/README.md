# Velociraptor Mongoliensis

A life-size reconstruction of *Velociraptor mongoliensis*, a small feathered theropod. It walks on two weight-bearing
toes per foot and carries a raised sickle claw on each foot. The model is for learning to stand and to walk. It is a
research abstraction built on measured bones, not a validated model of how the animal moved.

## Models

| | Anatomical |
|---|---|
| Species / embodiment | `velociraptor_mongoliensis` / `anatomical` |
| Gymnasium ID | `MesozoicLabs/VelociraptorMongoliensis-Anatomical-v0` |
| Environment | [`VelociraptorMongoliensisAnatomicalEnv`](anatomical/env.py) |
| MuJoCo model | [`velociraptor_mongoliensis_anatomical.xml`](anatomical/assets/velociraptor_mongoliensis_anatomical.xml) |
| Stage configs | [`configs/velociraptor_mongoliensis/anatomical/stages.toml`](../../configs/velociraptor_mongoliensis/anatomical/stages.toml) |
| Catalog entry | [Velociraptor Mongoliensis (Anatomical)](../../docs/SPECIES_CATALOG.md#velociraptor_mongoliensis-anatomical) |
| Model page | [mesozoiclabs.com/docs/models/velociraptor_mongoliensis/anatomical](https://mesozoiclabs.com/docs/models/velociraptor_mongoliensis/anatomical) |

Dimensions, stage budgets, gates and published results are generated in the catalog entry; this page does not repeat
them.

## Anatomical review

- **Proportions.** The femur, tibia and metatarsus follow the measured specimen IGM 100/986, set in a crouched,
  bird-like leg posture under a level trunk. The skull follows GIN 100/25, so the leg and skull come from different
  animals. The mass distribution is reconstructed.
- **Joints.** Position servos drive neck pitch and yaw, head pitch, each leg's hip pitch, hip roll, knee and ankle,
  each sickle claw, and four tail joints. The home keyframe is the standing equilibrium: the leg springs rest there and
  the home controls carry the body's weight, so zero action holds the stance.
- **Feet.** Each foot stands on a rounded pad under the metatarsophalangeal (MTP) joint and on digits III and IV,
  which are passive sprung hinges. Digit II rides raised on its own sprung hinge. Its claw is a force-limited servo
  that only the prey collides with. The pad, digit III and digit IV each have a touch sensor and their own
  observation channel. The friction cone is elliptic, as on every species.
- **Tail, neck and arms.** Spring preloads carry the weight of the five-segment tail, and its distal end is rigid. The
  neck and skull are separate bodies; the skull collides with the floor and the neck does not. The feathered arms are
  welded to the trunk and do not collide.
- **Known limits.** The ankle bends only fore and aft, and the toes do not flex joint by joint. In the reference walk
  the feet land farther apart across than they step forward, unlike the narrow trackways of related animals. The
  plumage, color and soft-tissue outlines are reconstruction choices.

The full review, with specimen comparisons, standing balance, walking measurements and sources, is
[docs/reviews/VELOCIRAPTOR_ANATOMICAL_REFERENCE.md](../../docs/reviews/VELOCIRAPTOR_ANATOMICAL_REFERENCE.md).

## Reference walk

A scripted controller ([`scripts/reference_walk.py`](scripts/reference_walk.py)) shows that the model can stand and
walk under its own servos. It sends normalized actions through the ordinary Gym step, uses the measured pelvis motion
for feedback, plans alternate footholds and shifts support to the stance leg. MuJoCo computes the root motion, the
passive toes and every ground contact. Run it from the repository root:

```bash
python -m environments.velociraptor_mongoliensis.scripts.audit_motion --mode stand
python -m environments.velociraptor_mongoliensis.scripts.audit_motion --mode walk --seconds 35
```

`--mode stand` holds the home controls on level ground. `--mode walk` walks slowly straight ahead. Each command prints
a JSON report: touchdowns, stance duty, stride, swing clearance and contact slip per foot, the share of floor load on
the pad, digit III and digit IV, and the motion of the pelvis, head and tail. The stand report also gives how far the
center of mass lies inside the loaded contacts. `--seeds` and `--noise` repeat a trial from perturbed starts, and
`--skin-clearance` measures the rendered foot surface against the floor. The review lists the exact runs behind its
tables, and [`data/reference_validation.json`](data/reference_validation.json) holds its measured data.
[`tests/test_reference_walk.py`](tests/test_reference_walk.py) checks the walk,
[`tests/test_com_stance_quality.py`](tests/test_com_stance_quality.py) checks standing balance, and the
[walking replay](../../website/static/videos/velociraptor_mongoliensis_anatomical_reference_walk.mp4) shows the walk
with its measured foot contacts.

## Run the model

From the repository root:

```bash
pip install -e .              # the environments: MuJoCo, Gymnasium and NumPy
pip install -e ".[train]"     # adds Stable-Baselines3 and PyTorch, for training
pip install -e ".[test]"      # adds pytest, for the tests

python -m environments.velociraptor_mongoliensis.scripts.view_model --embodiment anatomical
python -m environments.velociraptor_mongoliensis.scripts.test_actuators --embodiment anatomical
python -m environments.velociraptor_mongoliensis.scripts.test_env --embodiment anatomical
pytest environments/velociraptor_mongoliensis/tests
```

`view_model` prints the model's bodies, joints and actuators and opens MuJoCo's interactive viewer, and
`test_actuators` opens the viewer with every actuator sweeping through its range. Both need a display; on macOS, run
them with `mjpython` instead of `python`. `test_env` is a smoke test: it resets and steps the environment and checks
the reward components, determinism and observation bounds over short random rollouts. `--render`, `--episodes` and
`--steps` adjust it.

## Training

Every model trains from the repository root through one command:

```bash
python -m environments.shared.train --species velociraptor_mongoliensis --embodiment anatomical curriculum --algorithm ppo
```

It trains stance, then locomotion, each warm-started from its parent in `stages.toml` and run with the budget of its
stage config. The strike that followed locomotion was removed with every hunt node (decision D-D59). To reuse the
certified stance of an earlier run and train only what is missing:

```bash
python -m environments.shared.train --species velociraptor_mongoliensis --embodiment anatomical \
    curriculum --algorithm ppo --trunk-from <earlier_run_dir> --output-dir <new_run_dir>
```

In [`notebooks/sb3_training.ipynb`](../../notebooks/sb3_training.ipynb), set `SPECIES = "velociraptor_mongoliensis"`,
`EMBODIMENT = "anatomical"` and `BEHAVIOR` to `"stance"` or `"walk"`. The
[behavior recipes guide](https://mesozoiclabs.com/docs/training/recipes) covers stopping at a behavior, retraining a
stage and the notebook settings; the [quick start](https://mesozoiclabs.com/docs/getting-started/quick-start) shows
how to evaluate a checkpoint.

Species notes:

- **Stance.** The stance passes on a stance-quality check of the floor forces under each foot, rolled at six headings
  after the stage ends, not on reward. The catalog entry lists its bars.
- **Measured constants.** Several stance constants are measured on the zero-action model: the settled pitch that the
  posture term centers on (`natural_pitch`), the settled tail and neck poses, and the starting action noise.
  `log_std_init = -2.30` in `stance.toml` gives σ 0.10, the edge of what the standing model survives: it holds 40/40
  episodes up to σ 0.09 and 38/40 at σ 0.10 (`tests/test_anatomical_noise_tolerance.py`). Re-measure them after any
  change to the MJCF.

## Task and rewards

The trained tasks are the stance and the walk; the strike, the hunt that followed the walk, was removed by decision
D-D59 and no stage trains it. Until the environment change that removes them, a prey sphere still spawns ahead of the
animal at a random distance and side offset, the observation still carries its direction and distance, and the strike
terms below are still in the code, set to zero by every stage config. The environment reports `strike_success` when a
sickle-claw geom contacts the prey while the strike reward is on; that contact ends the episode.

The observation reads no world heading: it uses projected gravity, the root velocity in a gravity-aligned heading frame
and the prey direction in pelvis coordinates, beside the joint state, the pelvis gyro and accelerometer, one channel per
foot touch sensor, the prey distance and the command. Actions are offsets from the MJCF `home` keyframe: zero holds the
standing pose, and -1 and +1 reach each actuator's limits. The docstring of [`base_env.py`](base_env.py) gives both
layouts.

The main reward groups:

- **Survival and posture:** an alive bonus, a posture term centered on the settled pitch, an optional nosedive
  penalty, and a fall penalty.
- **Progress:** forward velocity toward the prey, with optional backward, drift, sideways, spin and speed penalties.
- **Strike** (zero in every stage config since D-D59): approach shaping, optional proximity terms, and a bonus on
  claw contact.
- **Stance quality** (stance stage only): support under both feet, the home leg, tail and neck poses, and penalties
  on floor impacts, airborne physics substeps, action jerk and commands held at their limits.
- **Effort:** energy, action smoothness and tail stability.

The weights are the `VelociraptorMongoliensisEnv` constructor defaults in `base_env.py`, overridden by the `[env]`
table of each stage TOML in
[`configs/velociraptor_mongoliensis/anatomical/`](../../configs/velociraptor_mongoliensis/anatomical/). An episode
ends on a fall or launch, excessive tilt or a nosedive, the head tip or skull dropping too low, the trunk, skull or
distal tail touching the floor, or a strike. The thresholds are in `base_env.py`.

## Files

| Path | Contents |
|---|---|
| [`base_env.py`](base_env.py) | `VelociraptorMongoliensisEnv`, the species base: task, observation, action mapping, rewards and termination |
| [`anatomical/env.py`](anatomical/env.py) | `VelociraptorMongoliensisAnatomicalEnv`, the registered model |
| [`anatomical/assets/`](anatomical/assets/) | The MJCF model |
| [`data/`](data/) | The reference trials' measured data and the review's figures |
| [`scripts/`](scripts/) | The viewer (`view_model.py`), the actuator sweep (`test_actuators.py`), the smoke test (`test_env.py`), the reference walk and motion audit (`reference_walk.py`, `audit_motion.py`), an actuator saturation report (`actuator_saturation_report.py`) and the builder of the rendered surface (`build_integument.py`) |
| [`tests/`](tests/) | Environment, reward, actuator-bound, static-balance, mirror-symmetry, noise-tolerance, stance-quality, viewer, motion-audit and reference-walk tests |

## Further reading

- [Anatomical review](../../docs/reviews/VELOCIRAPTOR_ANATOMICAL_REFERENCE.md): specimen comparisons, standing balance
  and the reference walk's measurements.
- [Plant review](../../docs/reviews/VELOCIRAPTOR_PLANT_REVIEW.md): the anatomy and mechanics findings on the earlier
  model, with a status note on which are closed.
- [Stage 1 basin investigation](../../docs/investigations/VELOCIRAPTOR_STAGE1_BASIN_INVESTIGATION.md): why actions
  are offsets from the home keyframe and the posture term is centered on the natural lean.
- [Heading-free observations](../../docs/HEADING_INVARIANCE_DESIGN_2026_10.md): the design of the observation and of
  the six-heading stance check.
- [Direction following and terrain](../../docs/TRAIN_DIRECTION_AND_TERRAIN.md): the pilot behaviors in
  `configs/velociraptor_mongoliensis/anatomical/behaviors/`, trained by
  `python -m environments.shared.train_behaviors`.
- [Plant revisions](../../configs/plant_versions.toml): what changed in each physics, policy-interface and visual
  revision of the model, tracked under the [plant contract](../../docs/PLANT_CONTRACT.md).
