# Brachiosaurus Altithorax

A sauropod-inspired quadruped with a long, articulated neck, a tapering tail and shoulders that stand higher than its
hips. It walks with a slow four-beat, lateral-sequence gait that keeps at least two feet on the ground. The model is
for learning to stand and to walk. It is a research abstraction at model scale, not a validated model of how the
animal moved.

## Models

| | Anatomical |
|---|---|
| Species / embodiment | `brachiosaurus_altithorax` / `anatomical` |
| Gymnasium ID | `MesozoicLabs/BrachiosaurusAltithorax-Anatomical-v0` |
| Environment | [`BrachiosaurusAltithoraxAnatomicalEnv`](anatomical/env.py) |
| MuJoCo model | [`brachiosaurus_altithorax_anatomical.xml`](anatomical/assets/brachiosaurus_altithorax_anatomical.xml) |
| Stage configs | [`configs/brachiosaurus_altithorax/anatomical/stages.toml`](../../configs/brachiosaurus_altithorax/anatomical/stages.toml) |
| Catalog entry | [Brachiosaurus Altithorax (Anatomical)](../../docs/SPECIES_CATALOG.md#brachiosaurus_altithorax-anatomical) |
| Model page | [mesozoiclabs.com/docs/models/brachiosaurus_altithorax/anatomical](https://mesozoiclabs.com/docs/models/brachiosaurus_altithorax/anatomical) |

Dimensions, stage budgets, gates and published results are generated in the catalog entry; this page does not repeat
them.

## Anatomical review

- **Proportions.** The humerus and femur are nearly equal in length, as the holotype supports. The humerus, radius
  and metacarpus follow the ratio of the referred Potter Creek material, so the forelimb gains its extra height
  through the forearm and metacarpus. The chest deepens toward the shoulders. The skull outline, browsing angle,
  distal tail and skin contours draw on other brachiosaurids. Loads and speeds apply at model scale.
- **Joints.** A free torso carries position servos at four neck pitch joints, a neck yaw joint and head pitch; at each
  leg's hip pitch, hip roll, elbow or knee, wrist or ankle, and toe; and at the first three tail segments. The last
  tail joint is a passive spring. The elbows and hind knees fold in opposite directions.
- **Feet.** The compact forefoot carries weight through a near-vertical metacarpus. The longer hindfoot has an
  inclined metatarsus, a fleshy heel and three claws on the inner digits. On level ground each foot stands on one
  smooth sole ellipsoid, sized differently for forefoot and hindfoot. The claws and sculpted soft tissue are visual
  only, so the model cannot show individual toe pressures.
- **Neck and tail.** The neck holds the head up under the neutral command, and the snout lowers for browsing and
  recovers when the command is released. The tail tapers from a fleshy base.
- **Known limits.** The reference walk covers one slow speed on flat ground. A policy that stands still can earn a
  high locomotion reward, so reward alone does not show walking. The published training results come from earlier
  versions of the model.

The full review, with measurements, sources, standing balance, the walking trials and browsing, is
[docs/reviews/BRACHIOSAURUS_ANATOMICAL_REFERENCE.md](../../docs/reviews/BRACHIOSAURUS_ANATOMICAL_REFERENCE.md).

## Reference walk

A scripted controller ([`scripts/reference_walk.py`](scripts/reference_walk.py)) shows that the model can walk under
its own servos. It asks for a lateral-sequence walk with diagonal couplets (left hind, left fore, right
hind, right fore) and sends ordinary normalized actions; the joint servos, gravity and floor contact determine the
motion. Run it from the repository root:

```bash
python -m environments.brachiosaurus_altithorax.scripts.reference_walk
python -m environments.brachiosaurus_altithorax.scripts.reference_walk --seed 3 --reset-noise 0.05
```

The walk runs nine strides and measures the six after startup. It prints a JSON report: touchdown order, forward
speed, stride period and length, duty factor, swing clearance and loaded elbow and knee flexion per foot, the fewest
feet loaded at once, airborne substeps, contact slip, and the torso's bob, pitch and roll. The first command starts
from a noise-free reset; `--seed` and `--reset-noise` start from a perturbed pose.
[`tests/test_reference_walk.py`](tests/test_reference_walk.py) checks the walk,
[`tests/test_com_stance_quality.py`](tests/test_com_stance_quality.py) checks standing balance, and the
[walking replay](../../website/static/videos/brachiosaurus_altithorax_anatomical_reference_walk.mp4) shows the
noise-free run with its measured contacts.

## Run the model

From the repository root:

```bash
pip install -e .              # the environments: MuJoCo, Gymnasium and NumPy
pip install -e ".[train]"     # adds Stable-Baselines3 and PyTorch, for training
pip install -e ".[test]"      # adds pytest, for the tests

python -m environments.brachiosaurus_altithorax.scripts.view_model --embodiment anatomical
python -m environments.brachiosaurus_altithorax.scripts.test_env --embodiment anatomical
pytest environments/brachiosaurus_altithorax/tests
```

`view_model` prints the model's bodies, joints and actuators and opens MuJoCo's interactive viewer. It needs a
display; on macOS, run it with `mjpython` instead of `python`. `test_env` is a smoke test: it resets and steps the
environment and checks the reward components, determinism and observation bounds over short random rollouts.
`--render`, `--episodes` and `--steps` adjust it.

## Training

Every model trains from the repository root through one command:

```bash
python -m environments.shared.train --species brachiosaurus_altithorax --embodiment anatomical curriculum --algorithm ppo
```

It trains stance, then locomotion, each warm-started from its parent in `stages.toml` and run with the budget of its
stage config. The food reach that followed locomotion was removed with every hunt node (decision D-D59). To reuse the
certified stance of an earlier run and train only what is missing:

```bash
python -m environments.shared.train --species brachiosaurus_altithorax --embodiment anatomical \
    curriculum --algorithm ppo --trunk-from <earlier_run_dir> --output-dir <new_run_dir>
```

In [`notebooks/sb3_training.ipynb`](../../notebooks/sb3_training.ipynb), set `SPECIES = "brachiosaurus_altithorax"`,
`EMBODIMENT = "anatomical"` and `BEHAVIOR` to `"stance"` or `"walk"`. The
[behavior recipes guide](https://mesozoiclabs.com/docs/training/recipes) covers stopping at a behavior, retraining a
stage and the notebook settings; the [quick start](https://mesozoiclabs.com/docs/getting-started/quick-start) shows
how to evaluate a checkpoint.

Species notes:

- **Published clips.** The published training clips predate physics revision 6, so a policy for the current model
  must be trained and evaluated again. The 2026-07-18 result summary they came with was deleted with the hunt nodes
  (decision D-D59; `results/README.md`). Checkpoints from policy interface r11 or earlier fail the plant
  check and cannot seed a warm start or `--trunk-from`: r11 declared the model SB3-only (note 23), and r12 removed the
  food and made the observation heading-free (`configs/plant_versions.toml`).
- **Reference walk speed.** The reference walk (about 0.35 m/s) sits at the slow end of the walk's commanded cruise
  range (0.375-0.75 m/s), and it fails the walk gate's tracking bars even at a command it matches. It shows what the
  plant can do mechanically, not a gait that would pass the gate.

## Task and rewards

The trained tasks are the stance and the walk; the food reach, the hunt that followed the walk, was removed by
decision D-D59. Its reward terms, its `food_reached` termination and its constructor arguments (with the replay-camera
widening they switched on) are gone from the code, and the model declares no success key. The food target and its
observation channels went with it in policy interface r12 (physics r7), so the scene holds only the animal and the
floor. Under the stance the forward and heading terms measure along the heading the animal spawns with. The head-tip
site and its position sensor stay.

**The walk follows a live command** (decision D-D62); the stance runs no command. Each walk episode draws one request
at reset and holds it for the whole 10 s episode: a forward cruise at 0.375-0.75 m/s along a heading up to 0.26 rad
(15 degrees) off the spawn heading, or, in about 15% of episodes, an exactly zero command, which asks the animal to
stand still. The command is the observation's last three values. A Gaussian term pays tracking the commanded velocity
and yaw rate (weight 2.5); it subsumes the forward, heading and sideways terms, which the walk sets to 0, and the
terms that read a forward direction measure along the commanded heading. On a zero command a stand block
(`[commands.stand_weights]`, 10 weights) pays the stance's reward weights, so zero means stand still, not march in
place. The commanded speeds ramp from 0.25 of their end values to the full range over the first 2M training steps. The
standing-idle penalty is not paid on a zero command. After training the walk is judged on its `command_tracking/v1`
panel: the cruise family's survival and success bounds, the stand family's hold and the quadruped gait clause (pair
alternation, limb duty, the flight cap) and the lead-swap bar. The bars are provisional, and no controller has yet
been shown to clear them; the reference walk reads tracking quality 0.32 a step at a command it matches, the
gait-symmetry term pays a standing statue its whole weight, and the reference walk's hind-pair alternation is close to
its bar (decision D-D62 (16)).

The observation reads no world heading: it uses the torso's projected gravity and its velocity in a gravity-aligned
heading frame, beside the joint positions and velocities, the torso gyro and accelerometer, one touch channel per foot
and the command. Before that revision it read the world-frame torso quaternion and velocity and the food direction and
distance. Actions are offsets from the MJCF `home` controls: zero holds the standing pose, and -1 and +1 reach each
actuator's limits. The docstring of [`base_env.py`](base_env.py) gives both layouts.

The main reward groups:

- **Survival and posture:** an alive bonus, a fall penalty, and penalties on tilt and nosedive with a torso-height
  term.
- **Command tracking** (walk): the Gaussian of the velocity and yaw-rate errors against the command.
- **Progress:** forward velocity and heading along the spawn heading (the commanded heading on the walk), with
  penalties on backward, sideways and drifting motion, spin, speed above a threshold and standing idle; the walk sets
  the forward, heading and sideways weights to 0.
- **Gait quality:** penalties on torso and tail-tip angular velocity and on jerky actions, and a reward for
  alternating diagonal foot pairs.
- **Effort:** an energy penalty on actuator use.

The weights are the `BrachiosaurusAltithoraxEnv` constructor defaults in `base_env.py`, where most terms are off,
overridden by the `[env]` table of each stage TOML in
[`configs/brachiosaurus_altithorax/anatomical/`](../../configs/brachiosaurus_altithorax/anatomical/). An episode ends
when the torso height leaves its healthy range or the torso tilts too far, or when the torso, belly, head or distal
tail touches the floor. The thresholds are in `base_env.py`.

## Files

| Path | Contents |
|---|---|
| [`base_env.py`](base_env.py) | `BrachiosaurusAltithoraxEnv`, the species base: task, observation, action mapping, rewards and termination |
| [`anatomical/env.py`](anatomical/env.py) | `BrachiosaurusAltithoraxAnatomicalEnv`, the registered model, with its foot contact geoms |
| [`anatomical/assets/`](anatomical/assets/) | The MJCF model |
| [`data/`](data/) | The review's figure |
| [`scripts/`](scripts/) | The viewer (`view_model.py`), the smoke test (`test_env.py`), the reference walk (`reference_walk.py`) and the soft-tissue mesh builder (`axial_surface.py` with `limb_surface.py`) that rewrites the skin meshes in the MJCF |
| [`tests/`](tests/) | Environment, reward, actuator-bound, static-balance, stance-quality and reference-walk tests |

## Further reading

- [Anatomical review](../../docs/reviews/BRACHIOSAURUS_ANATOMICAL_REFERENCE.md): proportions, feet, standing balance,
  the reference walk's measurements, browsing and limits.
- [Gait audit, 2026-09](../../docs/investigations/GAIT_AUDIT_2026_09.md): replays of the July walker, a neck-pumping
  bound on an earlier plant, and why the locomotion reward pays a standing statue.
- [Stance-hack audit, 2026-10](../../docs/investigations/STANCE_HACK_AUDIT_2026_10.md), §3.6: the zero-action statue on
  an earlier plant.
- [Training review](../../docs/investigations/TRAINING_REVIEW.md): the first Brachiosaurus runs (February–March 2026),
  with a stage-2 review.
- [Direction following and terrain](../../docs/TRAIN_DIRECTION_AND_TERRAIN.md): the pilot behaviors in
  `configs/brachiosaurus_altithorax/anatomical/behaviors/`, trained by
  `python -m environments.shared.train_behaviors`.
- [Plant revisions](../../configs/plant_versions.toml): what changed in each physics, policy-interface and visual
  revision of the model, tracked under the [plant contract](../../docs/PLANT_CONTRACT.md).
