# Dibothrosuchus elaphros

*Dibothrosuchus elaphros* was a small, lightly built crocodylomorph from the Early Jurassic of Yunnan, China. The
model is a quadruped with an upright stance, gracile limbs, hindlimbs longer than its forelimbs, and a mobile neck,
head and tail. It is for learning to stand, to walk, and to snap: closing on small prey and touching it with the
snout. It is a research abstraction: the jaws form one rigid body, and the mass and soft tissue are modeling choices.

## Models

| | Anatomical |
|---|---|
| Species / embodiment | `dibothrosuchus_elaphros` / `anatomical` |
| Gymnasium ID | `MesozoicLabs/DibothrosuchusElaphros-Anatomical-v0` |
| Environment | [`DibothrosuchusElaphrosAnatomicalEnv`](anatomical/env.py) |
| MuJoCo model | [`dibothrosuchus_elaphros_anatomical.xml`](anatomical/assets/dibothrosuchus_elaphros_anatomical.xml) |
| Stage configs | [`configs/dibothrosuchus_elaphros/anatomical/stages.toml`](../../configs/dibothrosuchus_elaphros/anatomical/stages.toml) |
| Catalog entry | [Dibothrosuchus elaphros (Anatomical)](../../docs/SPECIES_CATALOG.md#dibothrosuchus_elaphros-anatomical) |
| Model page | [mesozoiclabs.com/docs/models/dibothrosuchus_elaphros/anatomical](https://mesozoiclabs.com/docs/models/dibothrosuchus_elaphros/anatomical) |

Dimensions, stage budgets, gates and published results are generated in the catalog entry; this page does not repeat
them.

## Anatomical review

- **Proportions.** The humerus and radius match measurements of the specimen IVPP V7907, and the skull matches a
  published cranial length. The snout-to-tail span is close to the published whole-animal estimate; the tail is
  reconstructed.
- **Joints.** A free trunk carries position servos at neck pitch, neck yaw and head pitch; at each leg's hip pitch,
  hip roll, elbow or knee, wrist or ankle, and toe; and at the first three tail segments. The last tail joint is a
  passive spring. The elbows sit behind the shoulders and the hind knees ahead of the hips, so each joint folds in its
  anatomical direction.
- **Feet.** Each forefoot has four compact digital rays on one toe hinge beside a small central sole, and the fossil
  hand supports the absence of a fifth digit. Each hindfoot has four supporting rays and a small, raised fifth ray,
  arranged by comparison with *Terrestrisuchus*. The sole and each digital ray are separate floor contacts, and one
  touch sensor per foot feeds the observation.
- **Head and surface.** A tapered trunk and tail, a distinct lower jaw, lateral eyes and nares, low frontal ridges and
  paired dorsal osteoderms give the surface recognizable landmarks. The snap uses a fixed contact geom at the snout.
- **Known limits.** The incomplete hindlimb leaves the hindfoot's dimensions uncertain. The walk's footfall sequence,
  cadence, tail carriage and toe-off timing are assumptions; living crocodylians use several footfall patterns. The
  reference walk covers one slow, straight walk on level ground.

The full review, with measurements, sources, standing balance, the walking trials, actuator loads and the termination
conditions, is
[docs/reviews/DIBOTHROSUCHUS_ANATOMICAL_REFERENCE.md](../../docs/reviews/DIBOTHROSUCHUS_ANATOMICAL_REFERENCE.md).

## Reference walk

A scripted controller ([`scripts/reference_walk.py`](scripts/reference_walk.py)) shows that the model can walk under
its own servos. It places the left hind foot, left forefoot, right hind foot and right forefoot in
sequence, each foot rolling forward over its digits before lifting, and it sends ordinary normalized actions to a
freely moving body. Run it from the repository root:

```bash
python -m environments.dibothrosuchus_elaphros.scripts.reference_walk
python -m environments.dibothrosuchus_elaphros.scripts.reference_walk --strides 36 --seed 3 --reset-noise 0.05
```

By default the walk runs nine strides from a neutral start and measures the six after startup. It prints a JSON
report: touchdown order, forward speed, stride period and length, duty factor, swing clearance and loaded heel rise
per foot, the share of load on the digits, track width, the fewest feet loaded at once, airborne substeps, contact
slip, and the motion of the trunk, snout and tail tip. `--strides` sets the total number of strides, and `--seed` with
`--reset-noise` starts from a perturbed pose.

[`tests/test_reference_walk.py`](tests/test_reference_walk.py) checks the walk,
[`tests/test_com_stance_quality.py`](tests/test_com_stance_quality.py) checks standing balance, and the
[walking replay](../../website/static/videos/dibothrosuchus_elaphros_anatomical_reference_walk.mp4) shows the default
run with its measured contacts.

## Run the model

From the repository root:

```bash
pip install -e .              # the environments: MuJoCo, Gymnasium and NumPy
pip install -e ".[train]"     # adds Stable-Baselines3 and PyTorch, for training
pip install -e ".[test]"      # adds pytest, for the tests

python -m environments.dibothrosuchus_elaphros.scripts.view_model --embodiment anatomical
python -m environments.dibothrosuchus_elaphros.scripts.test_actuators --embodiment anatomical
python -m environments.dibothrosuchus_elaphros.scripts.test_env --embodiment anatomical
pytest environments/dibothrosuchus_elaphros/tests
```

`view_model` prints the model's bodies, joints and actuators and opens MuJoCo's interactive viewer, and
`test_actuators` opens the viewer with every actuator sweeping through its range. Both need a display; on macOS, run
them with `mjpython` instead of `python`. `test_env` is a smoke test: it resets and steps the environment and checks
the reward components, determinism and observation bounds over short random rollouts. `--render`, `--episodes` and
`--steps` adjust it.

## Training

Every model trains from the repository root through one command:

```bash
python -m environments.shared.train --species dibothrosuchus_elaphros --embodiment anatomical curriculum --algorithm ppo
```

It trains stance, then locomotion, then the snap, each warm-started from its parent in `stages.toml` and run with the
budget of its stage config. To reuse the certified stance and walk of an earlier run and train only what is missing:

```bash
python -m environments.shared.train --species dibothrosuchus_elaphros --embodiment anatomical \
    curriculum --algorithm ppo --trunk-from <earlier_run_dir> --output-dir <new_run_dir>
```

In [`notebooks/sb3_training.ipynb`](../../notebooks/sb3_training.ipynb), set `SPECIES = "dibothrosuchus_elaphros"`,
`EMBODIMENT = "anatomical"` and `BEHAVIOR` to `"stance"`, `"walk"` or `"hunt"`. The
[behavior recipes guide](https://mesozoiclabs.com/docs/training/recipes) covers stopping at a behavior, retraining a
stage and the notebook settings; the [quick start](https://mesozoiclabs.com/docs/getting-started/quick-start) shows
how to evaluate a checkpoint.

Species notes:

- **Published results.** No trained policy has been published for this model yet. The motion on this page and on the
  model page comes from the scripted reference controller.
- **Older checkpoints.** Checkpoints from policy interface r9 or earlier fail the plant check and cannot seed a warm
  start or `--trunk-from` (`configs/plant_versions.toml`, note 23).

## Task and rewards

A small prey sphere spawns ahead of the animal at reset, just above the ground. The task is to touch it with the
fixed snout geom `snout_snap`; the model has no articulated jaw. The environment reports `snap_success` and, when
the snap bonus is enabled (the snap stage), ends the episode.

The observation holds the joint positions and velocities, the trunk orientation quaternion, gyro, linear velocity and
accelerometer, one touch channel per foot, the prey direction and distance, and the command. Actions are offsets from
the MJCF `home` controls: zero holds the standing pose, and -1 and +1 reach each actuator's limits. The docstring of
[`base_env.py`](base_env.py) gives both layouts.

The main reward groups:

- **Survival and posture:** an alive bonus, a fall penalty, and penalties on tilt and nosedive with a trunk-height
  term.
- **Progress:** forward velocity and heading toward the prey, with penalties on backward, sideways and drifting
  motion, spin, speed above a threshold and standing idle.
- **Snap:** a large bonus on contact, shaping on the snout-to-prey distance and a snout-proximity term for the last
  approach. The shaping is measured from the snout, not the trunk: the skull rides a mobile neck, so trunk-referenced
  shaping would keep paying after the snout had overshot the prey.
- **Gait quality:** penalties on trunk and tail-tip angular velocity and on jerky actions, and a reward for
  alternating diagonal foot pairs (front right with rear left, front left with rear right).
- **Effort:** an energy penalty on actuator use.

The weights are the `DibothrosuchusElaphrosEnv` constructor defaults in `base_env.py`, where several terms are off,
overridden by the `[env]` table of each stage TOML in
[`configs/dibothrosuchus_elaphros/anatomical/`](../../configs/dibothrosuchus_elaphros/anatomical/). An episode ends
when the trunk height leaves its healthy range (the lower bound catches a collapse into a sprawl), the trunk tilts or
pitches nose-down too far, the snout tip drops to the floor, the skull, snout, trunk, belly or distal tail touches the
floor, or the snap succeeds. The thresholds are in `base_env.py`, and the review lists them.

## Files

| Path | Contents |
|---|---|
| [`base_env.py`](base_env.py) | `DibothrosuchusElaphrosEnv`, the species base: task, observation, action mapping, rewards and termination |
| [`anatomical/env.py`](anatomical/env.py) | `DibothrosuchusElaphrosAnatomicalEnv`, the registered model, with its foot contact geoms |
| [`anatomical/assets/`](anatomical/assets/) | The MJCF model |
| [`data/`](data/) | The review's figure |
| [`scripts/`](scripts/) | The viewer (`view_model.py`), the actuator sweep (`test_actuators.py`), the smoke test (`test_env.py`) and the reference walk (`reference_walk.py`) |
| [`tests/`](tests/) | Environment, reward, actuator-bound, static-balance, stance-quality and reference-walk tests |

## Further reading

- [Anatomical review](../../docs/reviews/DIBOTHROSUCHUS_ANATOMICAL_REFERENCE.md): proportions, feet, standing balance,
  the reference walk's measurements, actuator loads and termination conditions.
- [Gait audit, 2026-09](../../docs/investigations/GAIT_AUDIT_2026_09.md): on an earlier plant, the certified stance
  and the failed walk were both the zero-action statue, and a re-run skidded on three legs.
- [Stance-hack audit, 2026-10](../../docs/investigations/STANCE_HACK_AUDIT_2026_10.md), §3.5: the statue and the
  three-legged stance of the certified checkpoints on an earlier plant.
- [Hardware notes](../../docs/hardware/PARTS_AND_COSTS.md#dibothrosuchus-inspired-quadruped): a custom
  Dibothrosuchus-inspired quadruped, a paper concept of July 2026, as the near-$10k tier of the running-robot plans.
- [Direction following and terrain](../../docs/TRAIN_DIRECTION_AND_TERRAIN.md): the pilot behaviors in
  `configs/dibothrosuchus_elaphros/anatomical/behaviors/`, trained by `python -m environments.shared.train_behaviors`.
- [Plant revisions](../../configs/plant_versions.toml): what changed in each physics, policy-interface and visual
  revision of the model, tracked under the [plant contract](../../docs/PLANT_CONTRACT.md).
