# Dibothrosuchus elaphros

*Dibothrosuchus elaphros* was a small, lightly built crocodylomorph from the Early Jurassic of Yunnan, China. The
model is a quadruped with an upright stance, gracile limbs, hindlimbs longer than its forelimbs, and a mobile neck,
head and tail. It is for learning to stand and to walk. It is a research abstraction: the jaws form one rigid body,
and the mass and soft tissue are modeling choices.

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
  paired dorsal osteoderms give the surface recognizable landmarks. The fixed snout contact box the snap used was
  removed with the prey in physics r4 (decision D-D59); the snout-tip site stays for the snout-prop termination.
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

It trains stance, then locomotion, each warm-started from its parent in `stages.toml` and run with the budget of its
stage config. The snap that followed locomotion was removed with every hunt node (decision D-D59). To reuse the
certified stance of an earlier run and train only what is missing:

```bash
python -m environments.shared.train --species dibothrosuchus_elaphros --embodiment anatomical \
    curriculum --algorithm ppo --trunk-from <earlier_run_dir> --output-dir <new_run_dir>
```

In [`notebooks/sb3_training.ipynb`](../../notebooks/sb3_training.ipynb), set `SPECIES = "dibothrosuchus_elaphros"`,
`EMBODIMENT = "anatomical"` and `BEHAVIOR` to `"stance"` or `"walk"`. The
[behavior recipes guide](https://mesozoiclabs.com/docs/training/recipes) covers stopping at a behavior, retraining a
stage and the notebook settings; the [quick start](https://mesozoiclabs.com/docs/getting-started/quick-start) shows
how to evaluate a checkpoint.

Species notes:

- **Published results.** No trained policy has been published for this model yet. The motion on this page and on the
  model page comes from the scripted reference controller.
- **Older checkpoints.** Checkpoints from policy interface r10 or earlier fail the plant check and cannot seed a warm
  start or `--trunk-from`: r10 declared the model SB3-only (note 23), and r11 removed the prey and made the observation
  heading-free (`configs/plant_versions.toml`).

## Task and rewards

The trained tasks are the stance and the walk; the snap, the hunt that followed the walk, was removed by decision
D-D59. Its reward terms, its `snap_success` termination and its constructor arguments are gone from the code, and the
model declares no success key. The prey sphere, its observation channels and the fixed snout box `snout_snap` the
snap used went with it in policy interface r11 (physics r4), so the scene holds only the animal and the floor. Under
the stance the forward and heading terms measure along the heading the animal spawns with.

**The walk follows a live command** (decision D-D62); the stance runs no command. Each walk episode draws one request
at reset and holds it for the whole 10 s episode: a forward cruise at 0.45-0.9 m/s along a heading up to 0.26 rad (15
degrees) off the spawn heading, or, in about 15% of episodes, an exactly zero command, which asks the animal to stand
still. The command is the observation's last three values. A Gaussian term pays tracking the commanded velocity and
yaw rate (weight 2.5); it subsumes the forward, heading and sideways terms, which the walk sets to 0, and the terms
that read a forward direction measure along the commanded heading. On a zero command a stand block
(`[commands.stand_weights]`, 11 weights) pays the stance's reward weights, so zero means stand still, not march in
place. The commanded speeds ramp from 0.25 of their end values to the full range over the first 2M training steps. The
standing-idle penalty is not paid on a zero command. After training the walk is judged on its `command_tracking/v1`
panel: the cruise family's survival and success bounds, the stand family's hold and the quadruped gait clause (pair
alternation, limb duty, the flight cap) and the lead-swap bar. The bars are provisional, and no controller has yet
been shown to clear them; no controller in the repository walks inside the commanded speed range (the reference walk
reaches 0.16 m/s), and the gait-symmetry term pays a standing statue its whole weight (decision D-D62 (16)).

The observation reads no world heading: it uses the trunk's projected gravity and its velocity in a gravity-aligned
heading frame, beside the joint positions and velocities, the trunk gyro and accelerometer, one touch channel per foot
and the command. Before that revision it read the world-frame trunk quaternion and velocity and the prey direction and
distance. Actions are offsets from the MJCF `home` controls: zero holds the standing pose, and -1 and +1 reach each
actuator's limits. The docstring of [`base_env.py`](base_env.py) gives both layouts.

The main reward groups:

- **Survival and posture:** an alive bonus, a fall penalty, and penalties on tilt and nosedive with a trunk-height
  term.
- **Command tracking** (walk): the Gaussian of the velocity and yaw-rate errors against the command.
- **Progress:** forward velocity and heading along the spawn heading (the commanded heading on the walk), with
  penalties on backward, sideways and drifting motion, spin, speed above a threshold and standing idle; the walk sets
  the forward, heading and sideways weights to 0.
- **Gait quality:** penalties on trunk and tail-tip angular velocity and on jerky actions, and a reward for
  alternating diagonal foot pairs (front right with rear left, front left with rear right).
- **Effort:** an energy penalty on actuator use.

The weights are the `DibothrosuchusElaphrosEnv` constructor defaults in `base_env.py`, where several terms are off,
overridden by the `[env]` table of each stage TOML in
[`configs/dibothrosuchus_elaphros/anatomical/`](../../configs/dibothrosuchus_elaphros/anatomical/). An episode ends
when the trunk height leaves its healthy range (the lower bound catches a collapse into a sprawl), the trunk tilts or
pitches nose-down too far, the snout tip drops to the floor, or the skull, snout, trunk, belly or distal tail touches
the floor. The thresholds are in `base_env.py`, and the review lists them.

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
