# Allosaurus fragilis

A reconstruction of *Allosaurus fragilis*, a large Late Jurassic theropod, at adult size. It stands and walks on two
digitigrade feet, each with three forward weight-bearing toes, a pad under the toe joints and a raised inner toe. The
model is for standing and for walking straight ahead on level ground. It is a research abstraction:
fossils fix the segment lengths and foot structure, while the soft tissue, muscle torques and walking speed are
reconstruction choices.

## Models

| | Anatomical |
|---|---|
| Species / embodiment | `allosaurus_fragilis` / `anatomical` |
| Gymnasium ID | `MesozoicLabs/AllosaurusFragilis-Anatomical-v0` |
| Environment | [`AllosaurusFragilisAnatomicalEnv`](anatomical/env.py) |
| MuJoCo model | [`allosaurus_fragilis_anatomical.xml`](anatomical/assets/allosaurus_fragilis_anatomical.xml) |
| Stage configs | [`configs/allosaurus_fragilis/anatomical/stages.toml`](../../configs/allosaurus_fragilis/anatomical/stages.toml) |
| Catalog entry | [Allosaurus fragilis (Anatomical)](../../docs/SPECIES_CATALOG.md#allosaurus_fragilis-anatomical) |
| Model page | [mesozoiclabs.com/docs/models/allosaurus_fragilis/anatomical](https://mesozoiclabs.com/docs/models/allosaurus_fragilis/anatomical) |

Dimensions, stage budgets, gates and published results are generated in the catalog entry; this page does not repeat
them.

## Anatomical review

- **Proportions.** The femur, tibia and metatarsus, measured between joint centers, are within 2% of the adult
  specimen UUVP 6000r. The tail makes up about half the length, and the whole animal is slightly shorter than
  published reconstructions of USNM 4734 and UUVP 6000. The standing center of mass lies just ahead of and below the
  hip, near a published genus-level estimate.
- **Joints.** A free pelvis carries torque motors at each leg's hip roll, hip pitch, knee, ankle and toe joint. Two
  tail yaw joints move passively against springs and dampers. The head, neck and arms move with the trunk.
- **Feet.** Digits II, III and IV fan forward, with III the longest, and each has its own contact surface. A pad under
  the metatarsophalangeal joints shares their load. The three forward toes pitch together on one joint. The hallux
  sits high on the inner metatarsus. One touch sensor per foot measures its floor load, and the friction cone is
  elliptic, as on every species.
- **Head, arms and tail.** A paired crest rises ahead of the eyes, the three-fingered hands face inward, and the tail
  tapers from a muscular root. Color, pattern, lip outline and muscle contours are interpretive.
- **Known limits.** In the reference walk the track is wide for the short step, the trunk stays unusually level, and
  the outer digit takes the largest share of the load. The grouped toes limit how weight rolls from one digit to the
  next. The tests cover slow, straight travel on level ground only.

The full review, with measurements, floor-contact results, sources and limits, is
[docs/reviews/ALLOSAURUS_ANATOMICAL_REFERENCE.md](../../docs/reviews/ALLOSAURUS_ANATOMICAL_REFERENCE.md).

## Reference walk

A scripted controller ([`scripts/reference_walk.py`](scripts/reference_walk.py)) shows that the model can stand and
walk under its own motors. It reads the live body state, shifts weight to the stance foot and aims the swing foot at
the next foothold, and it sends only joint torques. Gravity and floor contact determine the motion. Run it from the
repository root:

```bash
python -m environments.allosaurus_fragilis.scripts.audit_motion --mode stand --seconds 20
python -m environments.allosaurus_fragilis.scripts.audit_motion --mode walk --seconds 30
```

The audit starts from the MJCF `home` keyframe and steps the MJCF directly. `--mode stand` balances in place;
`--mode walk` walks straight ahead. Each command prints a JSON report: touchdowns, duty, stride and swing clearance
per foot, the floor load on each pad and digit, forward travel and speed, step length and track width, double support
and time with no foot loaded, contact slip, peak floor load, and the motion of the pelvis, head and tail. `--warmup`
sets how many initial seconds the report leaves out, `--trace FILE.npz` saves the poses, foot positions and loads for
video replay, and `--skin-clearance` measures the visible skin above the floor. The review lists the exact runs behind
its tables, and [`data/reference_validation.json`](data/reference_validation.json) holds its measured data.
[`tests/test_reference_walk.py`](tests/test_reference_walk.py) checks the stand and the walk, and the
[walking replay](../../website/static/videos/allosaurus_fragilis_anatomical_reference_walk.mp4) shows the walk with
its foot placement and measured foot loads.

## Run the model

From the repository root:

```bash
pip install -e .              # the environments: MuJoCo, Gymnasium and NumPy
pip install -e ".[train]"     # adds Stable-Baselines3 and PyTorch, for training
pip install -e ".[test]"      # adds pytest, for the tests

python -m mujoco.viewer --mjcf=environments/allosaurus_fragilis/anatomical/assets/allosaurus_fragilis_anatomical.xml
pytest environments/allosaurus_fragilis/tests
```

The species has no viewer or smoke-test script of its own, so MuJoCo's standalone viewer opens the model. It needs a
display; on macOS, run it with `mjpython` instead of `python`. The tests stand in for a smoke test.
`tests/test_environment.py` resets and steps the registered environment, checks that zero action gives the home
torques and -1 and +1 the motor limits, and checks that only a supported arrival completes the walking course.
`tests/test_reference_walk.py` checks that the reference controller balances and walks the registered environment
through its motors, and `tests/test_anatomy.py` checks the leg lengths, the foot layout and the visible tissue.

## Training

Every model trains from the repository root through one command:

```bash
python -m environments.shared.train --species allosaurus_fragilis --embodiment anatomical curriculum --algorithm ppo
```

It trains stance, then locomotion, each warm-started from its parent in `stages.toml` and run with the budget of its
stage config. To reuse the certified stance of an earlier run and train only the walk:

```bash
python -m environments.shared.train --species allosaurus_fragilis --embodiment anatomical \
    curriculum --algorithm ppo --trunk-from <earlier_run_dir> --output-dir <new_run_dir>
```

In [`notebooks/sb3_training.ipynb`](../../notebooks/sb3_training.ipynb), set `SPECIES = "allosaurus_fragilis"`,
`EMBODIMENT = "anatomical"` and `BEHAVIOR` to `"stance"` or `"walk"`. The
[behavior recipes guide](https://mesozoiclabs.com/docs/training/recipes) covers stopping at a behavior, retraining a
stage and the notebook settings; the [quick start](https://mesozoiclabs.com/docs/getting-started/quick-start) shows
how to evaluate a checkpoint.

Species notes:

- **Published results.** No trained policy has been published for this model yet. The motion on this page and on the
  model page comes from the scripted reference controller.

## Task and rewards

The stance stage holds an upright posture on the feet. The locomotion stage, the walk, follows a live command
(decision D-D62); the stance runs none. Each walk episode draws one request at reset and holds it for the whole 30 s
episode: a forward cruise at 0.175-0.35 m/s along a heading up to 0.26 rad (15 degrees) off the spawn heading, or, in
about 15% of episodes, an exactly zero command, which asks the animal to stand still. A Gaussian term pays tracking
the commanded velocity and yaw rate (weight 2.5) and replaces the forward term, which the walk sets to 0; on a zero
command a stand block pays the stance's drift hold. The commanded speeds ramp from 0.25 of their end values to the
full range over the first 2M training steps. A cruise episode has an 8 m course along its commanded heading at reset:
the environment reports `course_complete` on the step the pelvis has advanced 7.65 m along it (the course length less
a 0.35 m tolerance) while at least one foot bears weight, and `success` from then on. It is an event, not a
termination, and no gate criterion; a stand episode has no course. After training the walk is judged on its
`command_tracking/v1` panel: the cruise family's survival and success bounds, the stand family's hold and the biped
gait clause with its flight cap. The bars are provisional, and no controller has yet been shown to clear them: the
reference walk sways within its stride, so at a command it matches it reads tracking quality 0.14 a step and fails the
tracking bars (decision D-D62 (16)). Under command_mode "none" the course runs along the spawn heading while the
forward weight is positive and ends the episode. The finish is a line across the heading, so a sideways offset neither
prevents nor causes it. Until policy interface r2 an invisible target sat 8 m ahead and the course ended within 0.35 m
of it; the scene now holds only the animal and the floor. The stance deliverable is judged on supported standing
instead; the catalog entry gives its definition.

The observation reads no world heading: it uses projected gravity and the root velocity in a gravity-aligned heading
frame, beside the joint state, the pelvis gyro and accelerometer, one load channel per foot and the command, 41 values
in all. Policy interface r1 also read the target direction in pelvis coordinates and its distance (45 values), so its
checkpoints do not load. Actions are offsets from the MJCF `home` torques: zero applies the home torques, and -1 and
+1 reach each motor's limits.

The main reward groups:

- **Support and posture:** an alive bonus and an upright-posture term at standing height, both paid only while a foot
  bears weight, and a fall penalty.
- **Command tracking** (walk): the Gaussian of the velocity and yaw-rate errors against the command.
- **Progress:** forward velocity along the spawn heading (the commanded heading on the walk), which the walk sets to
  0.
- **Drift** (stance, and the walk's zero command): a penalty on horizontal displacement from the starting point.
- **Effort:** an energy penalty on the squared action.

The weights are the `AllosaurusFragilisEnv` constructor defaults in [`base_env.py`](base_env.py), overridden by the
`[env]` table of each stage TOML in
[`configs/allosaurus_fragilis/anatomical/`](../../configs/allosaurus_fragilis/anatomical/). An episode ends on a
non-finite state, a pelvis height out of range or excessive tilt, any body part other than the feet touching the floor,
or, under command_mode "none", a completed course. The thresholds are in `base_env.py`.

## Files

| Path | Contents |
|---|---|
| [`base_env.py`](base_env.py) | `AllosaurusFragilisEnv`, the species base: task, observation, action mapping, rewards and termination |
| [`anatomical/env.py`](anatomical/env.py) | `AllosaurusFragilisAnatomicalEnv`, the registered model, with its foot contact geoms |
| [`anatomical/assets/`](anatomical/assets/) | The MJCF model and its skin texture |
| [`data/`](data/) | The reference trials' measured data and the review's figures |
| [`scripts/`](scripts/) | The reference walk and motion audit (`reference_walk.py`, `audit_motion.py`) and the builder of the MJCF and its skin (`build_anatomy.py`, with `surface.py`) |
| [`tests/`](tests/) | Anatomy, environment and reference-walk tests |

## Further reading

- [Anatomical review](../../docs/reviews/ALLOSAURUS_ANATOMICAL_REFERENCE.md): proportions, standing balance, the
  reference walk's measurements and the limits of the reconstruction.
- [Direction following and terrain](../../docs/TRAIN_DIRECTION_AND_TERRAIN.md): the pilot behaviors in
  `configs/allosaurus_fragilis/anatomical/behaviors/`, trained by `python -m environments.shared.train_behaviors`.
- [Plant revisions](../../configs/plant_versions.toml): what changed in each physics, policy-interface and visual
  revision of the model, tracked under the [plant contract](../../docs/PLANT_CONTRACT.md).
