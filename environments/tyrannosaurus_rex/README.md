# Tyrannosaurus rex

A scaled-down reconstruction of an adult *Tyrannosaurus rex*: a heavy biped with a broad skull, two-fingered arms and
a long counterbalancing tail. It stands on a plantar pad and three forward toes per foot. The model is for learning to
stand, to recover from pushes and to walk. It is a research abstraction, not a validated account of the animal's
mechanics.

## Models

| | Anatomical |
|---|---|
| Species / embodiment | `tyrannosaurus_rex` / `anatomical` |
| Gymnasium ID | `MesozoicLabs/TyrannosaurusRex-Anatomical-v0` |
| Environment | [`TyrannosaurusRexAnatomicalEnv`](anatomical/env.py) |
| MuJoCo model | [`tyrannosaurus_rex_anatomical.xml`](anatomical/assets/tyrannosaurus_rex_anatomical.xml) |
| Stage configs | [`configs/tyrannosaurus_rex/anatomical/stages.toml`](../../configs/tyrannosaurus_rex/anatomical/stages.toml) |
| Catalog entry | [Tyrannosaurus rex (Anatomical)](../../docs/SPECIES_CATALOG.md#tyrannosaurus_rex-anatomical) |
| Model page | [mesozoiclabs.com/docs/models/tyrannosaurus_rex/anatomical](https://mesozoiclabs.com/docs/models/tyrannosaurus_rex/anatomical) |

Dimensions, stage budgets, gates and published results are generated in the catalog entry; this page does not repeat
them.

## Anatomical review

- **Proportions.** The femur and tibia, measured between joint centers, follow the ratio of the adult specimens
  CM 9380 and MOR 555. The metatarsus is longer relative to the femur than in those adults. The whole animal is
  scaled down to simulation size, and its mass distribution is a modeling choice.
- **Joints.** Position servos drive neck pitch and yaw, head pitch, each leg's hip pitch, hip roll, knee and ankle,
  and four tail joints. The three forward toes of each foot are passive sprung hinges. The hip-roll servos are stiff
  enough that the two legs together resist the body's sideways sway.
- **Feet.** Each foot rests on a rigid plantar pad and three hinged forward digits, each with its own touch volume.
  Digit III is the longest. The small raised hallux and the claw sheaths are visual only. The friction cone is
  elliptic, so the pads grip the same way at every heading.
- **Tail, head and arms.** Spring preloads hold the tail level with its servos near zero. Its three proximal bodies
  articulate and the distal pair moves together. The arms are fixed to the trunk. The head-contact task uses a fixed
  snout geom.
- **Known limits.** The reference walk lands its feet wide apart compared with the near-inline steps of tyrannosaurid
  trackways, the pelvis rolls markedly as weight shifts, and the pad still carries most of the load late in stance.
  With zero action the model falls under the standard push schedule.

The full review, with landmark comparisons, balance and walking measurements, PPO transfer trials and sources, is
[docs/reviews/TREX_ANATOMICAL_REVIEW_2026_11.md](../../docs/reviews/TREX_ANATOMICAL_REVIEW_2026_11.md).

## Reference walk

A scripted controller ([`scripts/reference_walk.py`](scripts/reference_walk.py)) shows that the model can stand and
walk under its own servos. It sends the same normalized joint commands a policy uses, places each swing foot with
inverse kinematics in a private workspace, and uses the measured foot load and body motion to shift weight. MuJoCo
computes the root motion, the passive toes and every ground contact. Run it from the repository root:

```bash
python -m environments.tyrannosaurus_rex.scripts.audit_motion --mode stand --seconds 20 --warmup 9
python -m environments.tyrannosaurus_rex.scripts.audit_motion --mode walk --seconds 30 --warmup 4
```

`--mode stand` holds the home controls on level ground. `--mode walk` walks straight ahead. Each command prints a JSON
report: touchdowns, stance duty, stride, swing clearance and contact slip per foot, the share of floor load on the pad
and toes, and the motion of the pelvis, head and tail. The stand report also gives how far the center of mass lies
inside the loaded contacts. `--noise` and `--seeds` repeat a trial from perturbed starts, `--actuators` adds force and
joint-limit margins, and `--trace FILE.npz` saves the first seed's poses, foot forces and clearances. The review gives
the exact runs behind its tables, and the command that replays archived PPO walking checkpoints on the physics r11
model ([`scripts/replay_walk.py`](scripts/replay_walk.py), which runs in a checkout of `3368abe` or earlier: the
current environment refuses the archived configs' bite and prey arguments and observes 59 values, not their 63). [`tests/test_reference_walk.py`](tests/test_reference_walk.py)
checks the walk, [`tests/test_com_stance_quality.py`](tests/test_com_stance_quality.py) checks standing balance, and
the [walking replay](../../website/static/videos/tyrannosaurus_rex_anatomical_reference_walk.mp4) shows a noisy
walking trial with a close-up of the feet and their measured loads.

## Run the model

From the repository root:

```bash
pip install -e .              # the environments: MuJoCo, Gymnasium and NumPy
pip install -e ".[train]"     # adds Stable-Baselines3 and PyTorch, for training
pip install -e ".[test]"      # adds pytest, for the tests

python -m environments.tyrannosaurus_rex.scripts.view_model --embodiment anatomical
python -m environments.tyrannosaurus_rex.scripts.test_actuators --embodiment anatomical
python -m environments.tyrannosaurus_rex.scripts.test_env --embodiment anatomical
pytest environments/tyrannosaurus_rex/tests
```

`view_model` prints the model's bodies, joints and actuators and opens MuJoCo's interactive viewer, and
`test_actuators` opens the viewer with every actuator sweeping through its range. Both need a display; on macOS, run
them with `mjpython` instead of `python`. `test_env` is a smoke test: it resets and steps the environment and checks
the reward components, determinism and observation bounds over short random rollouts. `--render`, `--episodes` and
`--steps` adjust it.

## Training

Every model trains from the repository root through one command:

```bash
python -m environments.shared.train --species tyrannosaurus_rex --embodiment anatomical curriculum --algorithm ppo
```

It trains stance, then locomotion, each warm-started from its parent in `stages.toml` and run with the budget of its
stage config. The bite task that followed locomotion was removed with every hunt node (decision D-D59). To reuse the
certified stance of an earlier run and train only what is missing:

```bash
python -m environments.shared.train --species tyrannosaurus_rex --embodiment anatomical \
    curriculum --algorithm ppo --trunk-from <earlier_run_dir> --output-dir <new_run_dir>
```

In [`notebooks/sb3_training.ipynb`](../../notebooks/sb3_training.ipynb), set `SPECIES = "tyrannosaurus_rex"`,
`EMBODIMENT = "anatomical"` and `BEHAVIOR` to `"stance"`, `"recover"` or `"walk"`. The
[behavior recipes guide](https://mesozoiclabs.com/docs/training/recipes) covers stopping at a behavior, retraining a
stage and the notebook settings; the [quick start](https://mesozoiclabs.com/docs/getting-started/quick-start) shows
how to evaluate a checkpoint.

Species notes:

- **Recovery stage.** `recovery` holds the certified stance against scheduled pushes sized from the plant itself. It
  is a published behavior, judged after the stage against a frozen recovery gate (`recovery.toml`). Nothing
  warm-starts from it, so `curriculum` skips it. Train it on its own from the stance's handoff checkpoint (the one its
  `gate_verdict.json` names); the load is refused unless that checkpoint is a stance. It trains with PPO only. In the
  notebook, `BEHAVIOR = "recover"` trains stance and then recovery.

  ```bash
  python -m environments.shared.train --species tyrannosaurus_rex --embodiment anatomical \
      train --stage recovery --algorithm ppo --load <stance_checkpoint>.zip --load-mode initialize_next_stage
  ```

- **Stance.** The stance trains with mirror-symmetric PPO and extra stance terms (sideways lean, the settled leg pose,
  front-to-back foot stagger and each foot's center of pressure); see
  [mirror symmetry](https://mesozoiclabs.com/docs/training/mirror-symmetry). It passes on a stance-quality check of
  the floor forces under each foot, rolled at six headings after the stage ends, not on reward. The catalog entry
  lists its bars.

## Task and rewards

The trained tasks are the stance, recovery from pushes and the walk; the bite, the hunt that followed the walk, was
removed by decision D-D59. Its reward terms, its `bite_success` termination and its constructor arguments are gone from
the code, and the model declares no success key. The scene holds no prey: the prey sphere and the head-contact box the
bite read were removed with it in physics r12 and policy interface r17, and checkpoints of policy interface r16 or
earlier no longer load. Under the stance and recovery the forward, drift and heading terms measure from the spawn: the
pelvis's starting position and the unit forward of its starting orientation.

**The walk follows a live command** (decision D-D62); the stance and recovery run no command. Each walk episode draws
one request at reset and holds it for the whole 10 s episode: a forward cruise at 0.525-1.05 m/s along a heading up to
0.26 rad (15 degrees) off the spawn heading, or, in about 15% of episodes, an exactly zero command, which asks the
animal to stand still. The command is the observation's last three values. A Gaussian term pays tracking the commanded
velocity and yaw rate (weight 2.5); it subsumes the forward, heading and sideways terms, which the walk sets to 0, and
the terms that read a forward direction measure along the commanded heading. On a zero command a stand block
(`[commands.stand_weights]`, 26 weights; the standing lean, foot stagger and center-of-pressure terms are paid through
it) pays the stance's reward weights, so zero means stand still, not march in place. The commanded speeds ramp from
0.25 of their end values to the full range over the first 2M training steps. After training the walk is judged on its
`command_tracking/v1` panel: the cruise family's survival and success bounds, the stand family's hold and the biped
gait clause with its flight cap (Froude 0.12 at cruise). The bars are provisional, and no controller has yet passed
the committed panel (decision D-D62 (16)). The reference walk sways within its stride, so the tracking term and the gate
score a moving command on the trailing mean of the last 1.12 s, about one stride (`[commands] tracking_window_s`,
decision D-D64). At a command it matches (0.633 m/s) the reference walk then reads tracking quality 0.84 (0.13 a step
on its own), clears the in-tolerance fraction and the speed ratio, and fails `heading_settle` on 20 of 20, on its
scripted start: a centred stride mean of its speed reaches the speed tolerance at about 1.95 s, so the gate's trailing
window first reads in tolerance at about 2.5 s, after the 2.12 s allowance. A quicker start passes every episode.

The observation reads no world heading: it uses projected gravity and the root velocity in a gravity-aligned heading
frame, beside the joint state, the pelvis gyro and accelerometer, one load channel per foot (pad and toes) and the
command, last. Actions are offsets from the MJCF `home`
controls: zero holds the standing pose, and -1 and +1 reach each actuator's limits. The docstring of
[`base_env.py`](base_env.py) gives both layouts.

The main reward groups:

- **Survival and posture:** an alive bonus, posture, nosedive and height terms, head clearance and neck posture, and a
  fall penalty.
- **Command tracking** (walk): the Gaussian of the velocity and yaw-rate errors against the command.
- **Progress:** forward velocity along the spawn's forward direction (the commanded heading on the walk), with
  optional backward, drift, sideways, spin and speed penalties; the walk sets the forward, heading and sideways
  weights to 0.
- **Stance quality** (stance and recovery, and the walk's zero command): support and load balance between the feet,
  the home leg and tail poses, flat soles and a steady stance width, sideways lean, foot stagger, each foot's center
  of pressure, and penalties on floor impacts and airborne physics substeps.
- **Effort:** energy, smoothness, jerk and actuator saturation, and tail stability.

The weights are the `TyrannosaurusRexEnv` constructor defaults in `base_env.py`, where most stance terms are off,
overridden by the `[env]` table of each stage TOML in
[`configs/tyrannosaurus_rex/anatomical/`](../../configs/tyrannosaurus_rex/anatomical/); recovery inherits the stance's
`[env]`. An episode ends on a fall or launch, excessive tilt or a nosedive, the snout tip or skull dropping too low, or the
head, torso or tail touching the floor. The thresholds are in `base_env.py`.

## Files

| Path | Contents |
|---|---|
| [`base_env.py`](base_env.py) | `TyrannosaurusRexEnv`, the species base: task, observation, action mapping, rewards and termination |
| [`anatomical/env.py`](anatomical/env.py) | `TyrannosaurusRexAnatomicalEnv`, the registered model |
| [`anatomical/assets/`](anatomical/assets/) | The MJCF model |
| [`data/`](data/) | The review's measurements (`anatomical_review.json`) |
| [`scripts/`](scripts/) | The viewer (`view_model.py`), the actuator sweep (`test_actuators.py`), the smoke test (`test_env.py`), the reference walk and motion audit (`reference_walk.py`, `audit_motion.py`), the PPO transfer replay (`replay_walk.py`) and the builders of the rendered surface (`build_integument.py`, `hindlimb_surface.py`) |
| [`tests/`](tests/) | Environment, reward, actuator-bound, static-balance, lean and settled-leg, stance-stagger, center-of-pressure, stance-quality and reference-walk tests |

## Further reading

- [Anatomical review](../../docs/reviews/TREX_ANATOMICAL_REVIEW_2026_11.md): proportions, balance, the reference walk
  and the transfer of archived PPO checkpoints.
- [Stance-hack audit, 2026-10](../../docs/investigations/STANCE_HACK_AUDIT_2026_10.md): why the stance is judged on
  floor forces; §9 and §10 cover the first stances it judged and the center-of-pressure bar.
- [T. rex stance record](../../docs/investigations/TREX_STANCE_SYMMETRY_AND_FOOT_2026_10.md): the symmetry, foot and
  tail work behind the current stance recipe.
- [Recovery stage, first runs](../../docs/investigations/TREX_RECOVERY_STAGE_FIRST_RUNS_2026_08.md) and the
  [stance and recovery split](../../docs/STAGE1_SPLIT_PLAN.md): the push schedule and the recovery gate.
- [Direction following and terrain](../../docs/TRAIN_DIRECTION_AND_TERRAIN.md): the pilot behaviors in
  `configs/tyrannosaurus_rex/anatomical/behaviors/`, trained by `python -m environments.shared.train_behaviors`.
- [Plant revisions](../../configs/plant_versions.toml): what changed in each physics, policy-interface and visual
  revision of the model, tracked under the [plant contract](../../docs/PLANT_CONTRACT.md).
