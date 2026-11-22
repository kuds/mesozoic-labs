# Tyrannosaurus Rex

A scaled-down reconstruction of an adult *Tyrannosaurus rex*: a heavy biped with a broad skull, two-fingered arms and
a long counterbalancing tail. It stands on a plantar pad and three forward toes per foot. The model is for learning to
stand, to recover from pushes, to walk, and to reach prey and touch it with the snout. The jaw is fixed, so the "bite"
is a head-contact proxy, and the model is a research abstraction, not a validated account of the animal's mechanics.

## Models

| | Anatomical |
|---|---|
| Species / embodiment | `tyrannosaurus_rex` / `anatomical` |
| Gymnasium ID | `MesozoicLabs/TyrannosaurusRex-Anatomical-v0` |
| Environment | [`TyrannosaurusRexAnatomicalEnv`](anatomical/env.py) |
| MuJoCo model | [`tyrannosaurus_rex_anatomical.xml`](anatomical/assets/tyrannosaurus_rex_anatomical.xml) |
| Stage configs | [`configs/tyrannosaurus_rex/anatomical/stages.toml`](../../configs/tyrannosaurus_rex/anatomical/stages.toml) |
| Catalog entry | [Tyrannosaurus Rex (Anatomical)](../../docs/SPECIES_CATALOG.md#tyrannosaurus_rex-anatomical) |
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
the exact runs behind its tables, and the command that replays archived PPO walking checkpoints on this model
([`scripts/replay_walk.py`](scripts/replay_walk.py)). [`tests/test_reference_walk.py`](tests/test_reference_walk.py)
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

It trains stance, then locomotion, then the bite task, each warm-started from its parent in `stages.toml` and run
with the budget of its stage config. To reuse the certified stance and walk of an earlier run and train only what is
missing:

```bash
python -m environments.shared.train --species tyrannosaurus_rex --embodiment anatomical \
    curriculum --algorithm ppo --trunk-from <earlier_run_dir> --output-dir <new_run_dir>
```

In [`notebooks/sb3_training.ipynb`](../../notebooks/sb3_training.ipynb), set `SPECIES = "tyrannosaurus_rex"`,
`EMBODIMENT = "anatomical"` and `BEHAVIOR` to `"stance"`, `"recover"`, `"walk"` or `"hunt"`. The
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

A prey sphere spawns ahead of the animal at a random distance and side offset. The task is to reach it and touch it
with the fixed head-contact geom. The environment reports `bite_success` on that contact, while the bite reward is on;
the contact ends the episode.

The observation reads no world heading: it uses projected gravity, the root velocity in a gravity-aligned heading frame
and the prey direction in pelvis coordinates, beside the joint state, the pelvis gyro and accelerometer, one load
channel per foot (pad and toes), the prey distance and the command. Actions are offsets from the MJCF `home`
controls: zero holds the standing pose, and -1 and +1 reach each actuator's limits. The docstring of
[`base_env.py`](base_env.py) gives both layouts.

The main reward groups:

- **Survival and posture:** an alive bonus, posture, nosedive and height terms, head clearance and neck posture, and a
  fall penalty.
- **Progress:** forward velocity toward the prey, with optional backward, drift, sideways, spin and speed penalties.
- **Bite:** approach shaping, head-proximity shaping, and a bonus when the head-contact geom touches the prey.
- **Stance quality** (stance and recovery): support and load balance between the feet, the home leg and tail poses,
  flat soles and a steady stance width, sideways lean, foot stagger, each foot's center of pressure, and penalties on
  floor impacts and airborne physics substeps.
- **Effort:** energy, smoothness, jerk and actuator saturation, and tail stability.

The weights are the `TyrannosaurusRexEnv` constructor defaults in `base_env.py`, where most stance terms are off,
overridden by the `[env]` table of each stage TOML in
[`configs/tyrannosaurus_rex/anatomical/`](../../configs/tyrannosaurus_rex/anatomical/); recovery inherits the stance's
`[env]`. An episode ends on a fall or launch, excessive tilt or a nosedive, the snout tip or skull dropping too low, the
head, torso or tail touching the floor, or a bite. The thresholds are in `base_env.py`.

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
