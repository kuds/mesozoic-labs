# Brachiosaurus Altithorax

A sauropod-inspired quadruped with a long, articulated neck, a tapering tail and shoulders that stand higher than its
hips. It walks with a slow four-beat, lateral-sequence gait that keeps at least two feet on the ground. The model is
for learning to stand, to walk, and to reach food: bringing the head tip to an elevated food target. It is a research
abstraction at model scale, not a validated model of how the animal moved.

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

It trains stance, then locomotion, then food reach, each warm-started from its parent in `stages.toml` and run with
the budget of its stage config. To reuse the certified stance and walk of an earlier run and train only what is
missing:

```bash
python -m environments.shared.train --species brachiosaurus_altithorax --embodiment anatomical \
    curriculum --algorithm ppo --trunk-from <earlier_run_dir> --output-dir <new_run_dir>
```

In [`notebooks/sb3_training.ipynb`](../../notebooks/sb3_training.ipynb), set `SPECIES = "brachiosaurus_altithorax"`,
`EMBODIMENT = "anatomical"` and `BEHAVIOR` to `"stance"`, `"walk"` or `"hunt"`. The
[behavior recipes guide](https://mesozoiclabs.com/docs/training/recipes) covers stopping at a behavior, retraining a
stage and the notebook settings; the [quick start](https://mesozoiclabs.com/docs/getting-started/quick-start) shows
how to evaluate a checkpoint.

Species notes:

- **Published results.** The published results and training clips predate physics revision 6, so a policy for the
  current model must be trained and evaluated again. Checkpoints from policy interface r10 or earlier fail the plant
  check and cannot seed a warm start or `--trunk-from` (`configs/plant_versions.toml`, note 23).
- **Reference walk speed.** The reference walk is slower than the locomotion gate's minimum speed. It shows what the
  plant can do mechanically, not a gait that would pass the gate.

## Task and rewards

The species task is food reach. At reset a food target appears ahead of and above the animal. Success is the
head-tip site coming within a configured distance of the food; it is a distance test, and no food geom touches the
head. The environment reports it as `food_reached` and, when the food-reach bonus is enabled (the food-reach stage),
ends the episode.

The observation holds the joint positions and velocities, the torso orientation quaternion, gyro, linear velocity and
accelerometer, one touch channel per foot, the food direction and distance, and the command. Actions are residuals in
[-1, 1] around the MJCF `home` controls, so zero action holds the home pose. The docstring of
[`base_env.py`](base_env.py) gives both layouts.

The main reward groups:

- **Survival and posture:** an alive bonus, a fall penalty, and penalties on tilt and nosedive with a torso-height
  term.
- **Progress:** forward velocity and heading toward the food, with penalties on backward, sideways and drifting
  motion, spin, speed above a threshold and standing idle.
- **Gait quality:** penalties on torso and tail-tip angular velocity and on jerky actions, and a reward for
  alternating diagonal foot pairs.
- **Effort:** an energy penalty on actuator use.
- **Food reach:** a large bonus on success, shaping on the head-to-food distance and a head-proximity term.

The weights are the `BrachiosaurusAltithoraxEnv` constructor defaults in `base_env.py`, where most terms are off,
overridden by the `[env]` tables of `stance.toml`, `locomotion.toml` and `food_reach.toml` in
[`configs/brachiosaurus_altithorax/anatomical/`](../../configs/brachiosaurus_altithorax/anatomical/). An episode ends
when the torso height leaves its healthy range or the torso tilts too far, when the torso, belly, head or distal tail
touches the floor, or when the food is reached. The thresholds are in `base_env.py`.

## Files

| Path | Contents |
|---|---|
| [`base_env.py`](base_env.py) | `BrachiosaurusAltithoraxEnv`, the species base: task, observation, action mapping, rewards and termination |
| [`anatomical/env.py`](anatomical/env.py) | `BrachiosaurusAltithoraxAnatomicalEnv`, the registered model, with its foot contact geoms |
| [`anatomical/assets/`](anatomical/assets/) | The MJCF model |
| [`data/`](data/) | The review's figure |
| [`scripts/`](scripts/) | The viewer (`view_model.py`), the smoke test (`test_env.py`), the reference walk (`reference_walk.py`) and the soft-tissue mesh builder (`axial_surface.py` with `limb_surface.py`) that rewrites the skin meshes in the MJCF |
| [`tests/`](tests/) | Environment, reward, actuator-bound, static-balance, standing-balance and reference-walk tests |

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
