# Compsognathus push-recovery calibration

This extension makes the anatomical and robot models runnable through the
SB3 notebook's opt-in recovery stage. It calibrates finite horizontal pushes
and a physical recovery judge separately for the two plants. It does not
establish that a trained policy has learned recovery, and it makes no hardware
claim. The robot's head and tail remain fixed and unpowered.

## What is implemented

- Both environments expose the shared deterministic push scheduler. Pushes
  remain disabled in stance, locomotion, and target reaching.
- Each manifest inserts semantic `recovery` after `stance`; legacy stages 1,
  2, and 3 retain their meanings. Recovery remains a non-advancing pilot.
- Recovery mirrors the stance environment except for five disturbance
  settings. PPO uses the revised small-update recipe, a 3M-step experimental
  allowance, entropy decay to zero over 2M steps, and 100k-step transition
  warm-up. SAC retains its configured optimizer and notebook transition behavior.
- The notebook supports PPO and SAC checkpoint inference, freezes the null
  comparisons before recovery training, and saves per-episode and per-shove
  CSV evidence for the nulls and the evaluated policy.
- Each new species uses its own calibration JSON. The frozen evaluation
  identity includes the measured height reference, safe set, control clock,
  and profile hash. A changed plant, task, or profile requires recalibration;
  missing or inconsistent evidence refuses a verdict.

## Disturbance definition

The existing shared scheduler derives a nominal capture-velocity scale:

\[
v_c=r\sqrt{g/h},\qquad J=\alpha m v_c,\qquad F=J/T.
\]

Here `m` is the root subtree mass, `h` is its home-keyframe center-of-mass
height, and `r` is the minimum side margin to the axis-aligned bounding box
of home floor contacts. This is an approximate inverted-pendulum scaling;
it is not an exact convex-hull margin or a guaranteed recoverability limit.
The same multiple can have different measured difficulty on the two models.
The capture-point assumptions and the distinction between ankle, momentum,
and stepping responses are discussed in [Stephens (2007), Eq. 4](https://www.cs.cmu.edu/~bstephe1/papers/humanoids07.pdf).

The force acts at a massive body's center of mass, which need not be the
whole animal's center of mass. The anatomical model uses its pelvis. The
robot's free-root pelvis is a massless frame, so the scheduler uses the
heaviest massive body rigidly welded to it: the core, at a nominal height
of 0.288 m. The previous massless-root inertial point was 0.15 m higher and
produced an artificial tipping lever. The robot was recalibrated after this
correction. Massive-root species, including Tyrannosaurus Rex, retain their
previous target and trajectories. Contact and position-servo forces act during
the pulse, so `J/m` is a nominal free-body velocity increment, not the
measured velocity after a shove. See the [MuJoCo applied-force documentation](https://mujoco.readthedocs.io/en/stable/programming/simulation.html).

Both plants use a 0.02-second control step. The 0.20-second force pulse is
exactly ten control steps, avoiding rounding-induced impulse errors. Pushes
are horizontal with seeded random directions and a 2.0-second interval with
requested 0.25-second jitter (quantized to 12 steps, or 0.24 seconds).
Dynamics tests verify the force at every physics substep,
its integrated impulse, and clearing outside the scheduled window.

## Calibration and qualification are separate

The calibration script first measures quiet fixed-command trajectories,
then screens push strengths using a separate seed block. It freezes a
selected setting before evaluating fresh held-out seeds. Publication seeds
3042–3081 are reserved for the later trained-policy gate. Pairing initial
states and disturbances across controllers reduces uncontrolled variation;
separating selection from evaluation prevents choosing a setting on its own
reported holdout. This follows the experiment-design principles in
[Patterson et al. (2024), sections 4.2 and 4.4](https://jmlr.org/papers/volume25/23-0183/23-0183.pdf).

Zero action commands the authored home pose through position servos. It is
a fixed-command **PD control baseline**, not an unpowered robot. A second
fixed brace tests whether a modest static pose change explains the result.
These controls calibrate task difficulty and a feasible reference posture;
they do not replace an evaluation of a learned stance or recovery policy.

The initial safe-set thresholds combine measured quiet variation with
explicit engineering tolerance floors. They are provisional physical
acceptance criteria, not estimates of a learned policy's capability. The
full profile, quiet quantiles, force sweep, selected-controller outcomes,
and holdout measurements are recorded with the executable calibration.

## Measured physical calibration

The completed run contains **1,184 episodes, 725,899 control steps, and
6,248 judged shoves** across quiet measurements, strength selection, and
held-out nominal/stress panels. Each selected strength was screened on
32 seeds before the separate 40-seed holdout was evaluated.

| Quantity | Anatomical Compsognathus | Robot Compsognathus |
|---|---:|---:|
| Simulated mass | 1.0000 kg | 1.5856 kg |
| Home subtree COM height | 0.22835 m | 0.19633 m |
| Home contact AABB margin | 0.01960 m | 0.03497 m |
| Selected capture-velocity multiple | 1.32 | 1.20 |
| Push force, for 0.20 s | 0.84809 N | 2.35151 N |
| Nominal impulse | 0.16962 N s | 0.47030 N s |
| Force application body | Pelvis | Core |
| Measured root-height reference | 0.24447 m | 0.21806 m |
| Selection zero-command episode successes | 13/32 | 12/32 |
| Held-out zero-command episode successes | 27/40 | 7/40 |
| Held-out small fixed-brace episode successes | 31/40 | 15/40 |
| Held-out zero-command full-horizon survival | 28/40 | 20/40 |
| Held-out small fixed-brace full-horizon survival | 32/40 | 40/40 |
| Stress zero-command / fixed-brace successes | 1/40 / 3/40 | 1/40 / 1/40 |

The calibration brace commands +0.005 at both hip-pitch actuators and
−0.005 at both ankle-pitch actuators, with all other actions zero. It is a
predefined small static offset, separate from the stance-derived brace that
will be frozen from the user's actual checkpoint. The stress panel uses
1.25 times the selected force and a 1.9-second interval.

Both quiet controllers survived all eight seeds per model and occupied the
safe set throughout the measured post-settle samples. The selected safe-set
tolerances are height error ≤0.005 m, tilt ≤0.08 rad, and planar speed
≤0.08 m/s. All three came from the declared engineering floors rather than
the smaller measured quiet p99.9 variation. The height reference is a
settled root position, distinct from the subtree COM used for force scaling.

The anatomical holdout was easier than its selection panel, so the selected
force is not a precise estimate of a 50% failure threshold. Its fixed-command
baselines remain strong; beating both frozen nulls is essential to show a
control benefit. The robot's small brace survived every nominal episode but
recovered every shove in only 15 of 40, demonstrating why survival alone
does not qualify recovery. Neither result measures a learned policy.

Seed blocks are disjoint: quiet 1042–1049, selection 2042–2073, nominal
holdout 7042–7081, and stress 8042–8081. The future learned-stance brace uses
6042–6045; publication seeds 3042–3081 were not used in physical calibration.

## Pre-registered pilot targets

An episode succeeds only if it reaches the full horizon and every judged
shove recovers: re-enter the physical safe set within 40 control steps
(0.8 seconds) after pulse end and remain there for 20 consecutive steps
(0.4 seconds). A full-horizon episode without any judged shove cannot
qualify. The safe set checks root height relative to a fixed measured
reference, tilt, and planar speed. Per-step bilateral force is not required;
foot support is still measured and body-floor contact terminates an episode.

The initial targets are explicit engineering choices for a pilot:

| Criterion | Target |
|---|---:|
| Registered policy/null panel | 40 episodes, seeds 3042–3081 |
| One-sided 95% lower bound on episode success | At least 0.50 |
| One-sided 95% lower bound on paired success improvement over each required null | At least 0.10 |

The first condition requires at least 26 successes out of 40; the paired
condition can demand more, depending on which episodes the null succeeds on.
These targets have not been demonstrated by a trained Compsognathus policy.
They are not copied from Tyrannosaurus Rex's measured 0.30/0.20 thresholds.
Both zero command and the frozen stance-derived brace are required paired
comparisons for Compsognathus. A fixed pose that merely beats zero command
cannot pass by matching the brace; the policy must improve on both.
The historical Tyrannosaurus Rex gate retains its original single-null rule.

## Running the pilot

1. Use this revision in a fresh SB3 notebook runtime and select
   **Compsognathus Longipes** or **Compsognathus Longipes (Robot)**.
2. Train or load a passing stance policy first. Preserve its policy and matching
   VecNormalize checkpoint. Balance unit tests alone do not satisfy this step.
   Newly exposed, implicitly disabled push defaults preserve the exact task
   identity of existing Compsognathus quiet-stage checkpoints; changed task
   settings still refuse same-stage resume.
   Brace extraction requires the source policy to survive each two-second
   quiet reference rollout and occupy the safe set for at least 95% of
   samples after the one-second settling period. A mismatch stops before
   recovery training and requires revisiting the reference calibration.
3. Set `RUN_RECOVERY_STAGE = True`. The existing recovery cell freezes the
   same-task null panels, then trains from the stance checkpoint and evaluates
   the selected recovery checkpoint against that frozen judge.
4. Inspect the gate verdict, episode/shove CSVs, and video together. A short
   smoke run may fail; it is not a qualification experiment. Repeat full runs
   with additional training seeds and evaluate different shove magnitudes
   and timing before claiming robustness outside the declared schedule.

Recovery results join the run bundle. Locomotion still starts from stance;
this pilot does not automatically replace its starting checkpoint. If the
stance policy cannot occupy the measured reference posture, revisit the
calibration before training recovery, and freeze a new experiment rather
than changing the judge after seeing recovery results.

## Reproduction and validation

The calibration script and measurements live in
`scripts/calibrate_recovery.py` and `data/recovery_calibration_v1.json`.
The runtime profiles are in `configs/compsognathus/recovery_calibration.json`
and `configs/compsognathus_robot/recovery_calibration.json`.

From the repository root, with the training dependencies installed:

```bash
python -m environments.compsognathus.scripts.calibrate_recovery \
  --output environments/compsognathus/data/recovery_calibration_v1.json
```

The artifact records dependency versions and hashes of the calibration and
physics sources, with all episode and shove rows. Its working-tree head
identifies the parent commit; source hashes identify the measured uncommitted
implementation included in this change. A zero-second recorded re-entry
delay means the first judged sample after the pulse already met the safe
set, not instantaneous physical recovery.

Validation includes actual force/impulse tests, default-behavior regressions,
PPO/SAC stance-to-recovery updates and checkpoint reloads, notebook execution,
and refusal of missing, stale, tampered, or undersized calibration evidence.
Full training and trained-policy qualification remain future experiments.
