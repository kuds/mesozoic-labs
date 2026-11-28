# Triceratops 60 cm: reinforcement-learning setup (simulation only)

The training setup for a walking policy of the 60 cm, four-legged Triceratops museum robot, in MuJoCo, with
stable-baselines3 PPO. **Status: built and tested locally on short runs only. No policy has been trained to walk;
nothing here shows that the robot walks.** The reward weights, action scales and filter are untuned starting
points. Colab itself is untested; the first Colab run is a one-hour pipeline test (decision D16), not training.

## Files

| File | What it is |
|---|---|
| `trike_env.py` | The Gymnasium environment `TrikeEnv`, its settings `TrikeEnvConfig` (one dataclass, every range and weight with its default) and the pure reward function `reward_terms`. |
| `train.py` | PPO training: SubprocVecEnv (or DummyVecEnv), VecNormalize, atomic checkpoint pairs, `--resume`, settings kept per run, a deterministic evaluation during training, `run_record.json`. |
| `evaluate.py` | N episodes of a checkpoint (or a zero-action baseline) with randomisation on, the delay fixed at 10 ms (and optionally 0 ms), and a sweep of fixed commands (0-0.27 m/s, the 0.25 m/s target reported on its own); JSON out. |
| `trike60_training.ipynb` | The Colab notebook (8 cells), set up for the pipeline test: parameters, a shallow fetch of one exact commit, Python and pinned installs, Drive, training (always resuming), evaluation, runtime release. |
| `requirements-colab.txt` | The pins the notebook installs (torch as the CPU build). |
| `COLAB_STEPS.md` | The owner's step sheet for the pipeline test (draft; the commit to fill in). |
| `trike_checkpoints.py` | Checkpoint pairs: the check that tells a broken file from an I/O error, the scan (newest intact pair, re-reads on I/O errors), atomic saves (non-finite parameters set aside), setting broken files aside, atomic JSON. |
| `gait_metrics.py` | The gait and drive measures of an evaluation episode (logged every physics step): RMS torque / continuous rating per drive, time at the force limit (nominal and episode limit), stance slip ratio, debounced touchdowns, power, drift and heading change; the definitions of `analysis/gait_pipe1.py`. |
| `steer_eval.py` | Steering evaluation (decision D20): a checkpoint on its request schedule and on scripted courses, with the measures of `telemetry/steering.py` and a top view per course. |
| `demo_course.py` | Heat, battery and loads over a museum demo (decision D21): walking and standing simulated in segments, then a 20 min demo loop run through the thermal and battery models; bearing loads and foot forces per segment. |
| `telemetry/` | Generic training and hardware telemetry (decision D21; plan `PLAN_STEERING_AND_TELEMETRY.md`), robot-agnostic so a T. rex or larger Triceratops env can use it: `health.py` (training health, `train_health.jsonl`), `checks.py` (periodic checks at named points), `health_dashboard.py` (their charts), `steering.py` (direction-following measures from the shared contract's step fields), `thermal.py` (drive winding temperature, a first-order model calibrated to the maker's temperature-rise figure), `power.py` (battery runtime, pack currents, braking surplus), `loads.py` (joint bearing loads and foot forces from MuJoCo). Tests: `tests/test_telemetry.py`, on a toy env. |
| `tests/test_trike_env.py`, `tests/conftest.py` | 56 pytest tests, **run by hand** (they are not in the repository's `testpaths` or CI). |

## How to run locally

A Python 3.12-3.14 venv with the pinned packages (torch from the CPU index):

    python -m venv venv
    venv/bin/pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
    venv/bin/pip install mujoco==3.10.0 numpy==2.5.5 gymnasium==1.4.0 stable-baselines3==2.9.2 cloudpickle==3.1.2 pytest

From the repository root (the scripts put `rl/` and the repository root on `sys.path` themselves):

    venv/bin/python hardware/triceratops_build/rl/train.py --out /tmp/trike_run --total-steps 2000000 --n-envs 4
    venv/bin/python hardware/triceratops_build/rl/train.py --out /tmp/trike_run --total-steps 2000000 --n-envs 4 --resume
    venv/bin/python hardware/triceratops_build/rl/evaluate.py --run-dir /tmp/trike_run --episodes 10 --delays 10 0 --sweep
    venv/bin/python hardware/triceratops_build/rl/evaluate.py --policy zero --model trike60_C_closed --episodes 3 --sweep
    venv/bin/python -m pytest hardware/triceratops_build/rl/tests          # about 2.5 minutes, by hand

`--total-steps` is the run's total, not an increment. `--model` takes `trike60_C_closed` (default),
`trike60_C_closed_ankleroll`, `trike60_C_closed_capsule` (the capsule sole, D19) or a path. `--env-config FILE.json` overrides any `TrikeEnvConfig` field.

**Run folder rules (train.py).** Without `--resume`, a folder that already holds checkpoints is refused. With
`--resume`: no checkpoint starts the run at step 0 (unless the record forbids it, below); an intact pair continues it;
if every checkpoint is *structurally* broken
(missing sidecar, not a zip, missing SB3 members, a pickle that does not load) they are moved to
`checkpoints/broken_<time>/` (kept, never deleted), recorded in `run_record.json`, and the run starts again from step
0. An **I/O error** while reading them (an `OSError` such as EIO from a Drive mount, including on the folder
itself: only "no such folder" counts as no folder) is not a verdict on the files: the whole folder is read again (`--io-retries` 3 times, `--io-retry-wait` 20 s apart) and if the error persists the
run is refused with exit 2 and nothing moved. (The shared `checkpoint_pair_problem` reports both kinds alike, so
`rl/` uses its own check, `trike_checkpoints.pair_problem`.) **A step-0 start never overwrites earlier saves:** every
session records each save in `run_record.json` as it is made (`saves`, `last_checkpoint`) and how far it got
(`steps_reached`, updated at every save and progress line), written atomically, so a session that is killed (a lost
runtime) still leaves that record; a quarantine is recorded the moment its files are moved. When the folder lists
no resumable checkpoint but, since the run's last quarantine, a session recorded saves that are nowhere under the
checkpoint folder (`broken_*/` and `nonfinite/` included), train.py refuses (exit 2) instead of starting at step 0
and overwriting saves that a storage merely failed to show; if all of them are non-finite it refuses with "there is
no finite checkpoint to resume from" (a diverged run: use a new run folder with other settings); a session in the
older record format (no `saves` list) must have recorded an end and stayed under the first checkpoint interval. A
session killed before its first save, or right after a quarantine, leaves the run free to start again.
**Every refusal comes before any change**: nothing is moved, removed or written by a refused run. A run keeps its settings (model, env config, PPO settings, envs, seed, total steps, penalty ramp, evaluation):
a resume with other values is refused, or runs with `--allow-changes`, which logs every change in the session's
entry and applies it (PPO settings are passed to the load). The policy's shape (`--net`, `--log-std-init`) never
changes. Changing `--total-steps` moves the learning rate (linear in steps / total), so a run is extended under a new
run folder instead. `run_record.json` records what each session really used (`ppo_in_use` is read back from the
loaded model); with `--allow-changes` a new `--gamma` is also set on the loaded VecNormalize (its reward normaliser
discounts with its own copy). Every JSON file and every checkpoint is written to a temporary name and renamed; an
unreadable `run_record.json` is moved aside, not a dead end. A pair whose parameters are not finite (a diverged run)
is saved under `checkpoints/nonfinite/`, never a resume candidate, and the session ends with exit code 1.

**Stopping.** The first SIGINT or SIGTERM ends training and saves a final pair (both signals are then ignored until
it is written); exit code 130. SubprocVecEnv workers ignore SIGINT; the main process closes them. A Jupyter interrupt
signals the kernel's whole process group, so the notebook starts `train.py` in its own session and forwards exactly
one SIGINT; it then keeps showing train.py's lines, ignoring further clicks, until train.py has exited, and only then
flushes Drive (cell 5 also flushes after any non-zero exit; the flush itself ignores clicks). `--exit-with-parent
[PID]` (Linux, `PR_SET_PDEATHSIG`; the notebook passes its kernel's PID) turns the death of the notebook kernel into a
SIGTERM, so `train.py` never trains on as an orphan; it checks `prctl`'s result and refuses to start (exit 3) if the
parent is already gone. Output to a closed pipe does not stop the save.

## Design

**Standalone env, not a `BaseDinoEnv` subclass.** `environments/shared/base_env.py` sizes the action space to all
`model.nu` actuators (16 here, of which the policy drives 12), writes `data.ctrl` once per control step (the servo
delay needs a write every physics step), and its `reset` gives no hook between loading the keyframe and settling
where per-episode model randomisation belongs. Fitting those would mean overriding `step` and most of `reset`, and
the base class carries the shared plant-contract and fingerprint conventions of the registered species. Shared code
is imported, not copied: `LinearSchedule`, `newest_intact_periodic_pair` and `checkpoint_pair_problem`,
`load_sb3_model`, and in the notebook `disconnect_runtime` (with an inline fallback if the import fails in a kernel
whose packages cell 3 just replaced). No manifest or registry entry.

**Model.** `../sim/generated/trike60_C_closed.xml` by default (model revision 3, 12.69 kg, ankle springs
2.03 N·m/rad). Everything is read from the file, so a regenerated model needs no code change while the actuator,
sensor, joint and geom names stay. Reset uses the `home` keyframe (all ctrl 0 = neutral targets); the soles sit
exactly on the floor there, so reset adds uniform +-0.02 rad to the 12 leg joints and then places the root so the
lowest sole corner is 2 mm above the floor.

**Control: 50 Hz** (`frame_skip` 20 at the model's 1 ms step). The 0-10 ms servo delay is far below one control
period, the drives are stiff position servos (kp 598.5) that track between updates, and 50 Hz halves the policy
evaluations per simulated second, which matters on CPU. Configurable.

**Actions.** 12 numbers in [-1, 1], order `LEG_ACTUATORS` = `{left,right}_{fore,hind}_{roll,pitch,flexion}` (fore
before hind: left_fore, right_fore, left_hind, right_hind). They pass a **first-order low-pass filter at 4 Hz**
(`action_filter_hz`; 0 turns it off), then become residual targets `neutral + filtered * scale`, scale 0.15 rad
(roll), 0.40 rad (pitch), 0.40 rad (flexion), clipped to each ctrlrange. The filter is `y += alpha * (a - y)` once
per control step with alpha = 1 - exp(-2 pi 4 Hz x 20 ms) = 0.395. Computed from that: a 1.5 Hz gait passes at 0.94
of its amplitude but about **29 ms late (three times the 10 ms servo delay)**; step-to-step jitter (25 Hz) passes at
0.25 of its amplitude, and white noise at 0.50 of its standard deviation. Measured with random actions (std 0.37,
command fixed at 0.25 m/s, 4 episodes) it cuts the robot's drift from 0.080 to 0.017 m/s and the energy penalty from
0.35 to 0.08 per step, so exploration noise alone cannot carry the robot. It is part of the policy interface: **the
robot must run this identical filter at the identical 50 Hz control rate**, and its lag adds to the servo delay.
Neck and tail servos are held at 0.

**Servo delay.** A per-physics-step FIFO (`deque(maxlen=k+1)`) of the 12 leg targets: each physics step appends the
current target and writes the oldest into `data.ctrl`, so a target reaches `data.ctrl` exactly k physics steps (k ms)
after it was set (tested for k = 0, 3, 10). k is drawn uniformly from 0-10 per episode, or fixed by
`delay_steps_fixed` (evaluation: 10), or by `reset(options={"delay_steps": k})`.

**Domain randomisation** (per episode; the model is restored from saved nominal copies, then scaled, then
`mj_setConst`; with `domain_randomization=False` it is bit-identical to the file's model, tested):

| Quantity | Range | How |
|---|---|---|
| Sliding friction, floor and 4 soles | 0.6-1.0 | one draw, set on all five (MuJoCo takes the max of a pair, so the contact gets exactly this) |
| Body masses | x0.9-1.1 | independent per body; inertia scaled by the same factor |
| Actuator kp | x0.9-1.1 | per actuator; `gainprm[0]` and `biasprm[1]` together (kv unchanged) |
| Actuator force limit | x0.85-1.0 | per actuator, `forcerange` |

The draws are made at every reset whether randomisation is on or off, so the rest of the episode's random stream
does not shift when it is switched.

**Observation** (68 floats by default; `env.obs_layout` names the slots, `env.obs_slots(obs)` splits one; a test checks every
slot against the quantity it names): 12 leg joint angles (relative to neutral) and 12 joint speeds; 4 passive ankle
pitch angles (the model has jointpos sensors on them; the variant's roll ankles have none and are not observed);
gravity direction in the body frame (from the IMU quaternion); `heading_error_sin_cos_ESTIMATE`; gyro;
`base_lin_vel_body_PRIVILEGED` (from `qvel`: no sensor on the robot gives it; `obs_privileged_lin_vel=False` removes
it); 4 foot touch loads / nominal weight; `previous_action_filtered` (the filter state, i.e. what is commanded) and
`previous_action_raw` (the policy's last output, which the action-rate penalty compares the next one with, so that
penalty depends only on what the policy sees); the commanded speed (m/s); and, only if `phase_clock_hz` > 0, a gait
clock (sin/cos; no reward uses it). The clock is **off by default** since the pipe1 gait analysis (19 Nov 2026): that
policy stepped at 2.39 Hz and ignored its 1.5 Hz clock. Runs recorded with the clock (trike60-pipe1: 1.5 Hz, 70
floats) keep it: `evaluate.py --run-dir` and `train.py --resume` rebuild the env from the run's `run_record.json`,
whose env config names it, and fields added since take their defaults (tested).

Observation noise (Gaussian std, every observation; `observation_noise=False` turns it all off): joint angle
0.005 rad, joint speed 0.05 rad/s, gyro 0.02 rad/s, quaternion 0.01 per component (renormalised, about 1 deg), base
velocity 0.02 m/s, foot loads 0.02 of body weight (about 2.5 N). **Heading:** the parts list has no magnetometer, so
on the robot the heading is a gyro-integrated estimate that drifts. The heading observation is therefore the noisy
quaternion's yaw **plus a random walk** of 0.01 rad/sqrt(s) (`heading_drift_rw`; about 2.6 deg std after 20 s;
reset to 0 each episode; 0 turns it off). The heading reward uses the true yaw.

**Task and reward** (per 20 ms step; every term is in `info` as `r_*`; `reward_terms` is a pure function, tested
for its shape and the sign of every term): command drawn per episode uniformly from [0, 0.27] m/s (10 % of episodes
exactly 0); heading = the yaw at reset. Forward and lateral speed are the chassis displacement over the control step
along and across that heading.

| Term | Weight | Form |
|---|---|---|
| speed tracking | 1.5 | exp(-(v_fwd - v_cmd)^2 / 0.06^2) (sigma 22 % of the command range) |
| speed tracking, linear | -0.8 | \|v_fwd - v_cmd\| (a slope far from the command, where the exp term is flat) |
| alive | 0.3 | constant |
| upright | -1.0 | gx^2 + gy^2 (gravity in body frame) |
| height | -20 | (z - z_standing)^2 (z_standing measured at init: 0.5 s standing on the nominal model) |
| lateral | -2.0 | v_lat^2 |
| yaw rate | -0.1 | yaw_rate^2 |
| heading | -0.5 | heading_error^2 (true yaw) |
| energy | -0.001 x ramp | mean over the step of sum \|torque x joint speed\| of the 12 leg drives (W) |
| action rate | -0.04 | sum (a - a_prev)^2 of the raw policy outputs |
| foot slip | -1.0 x ramp | sum over feet in contact (> 2 N) of the sole's horizontal speed^2 (0.5 until pipe1; doubled rather than normalised by stance, so a sliding stance costs twice as much whatever its length) |
| drive heat | -0.3 x late | mean over the 12 leg drives and the step's 20 physics steps of (torque / continuous rating)^2; ratings at the joint from the mass ledger: X4 roll and pitch 7.5, X6 front elbow 10.8, X4 hind knee 6.75 N·m (tested against the ledger). Standing still scores about 0.06; the pipe1 gait 0.76, with exploration noise 0.99; every drive exactly at its rating, 1 |
| torque saturation | -2.0 x late | share of the 12 drives x 20 physics steps at >= 95 % of the episode's randomised force limit (the limit the drive would hit). Pipe1 gait 3.0 %, with noise 7.1 %; standing 0 |
| air time | +1.0 (gain), x late (shortfall) | at each debounced touchdown, d = min(air time, 0.5 s) - 0.20 s, summed over feet, only when v_cmd > 0.05 m/s: + d when positive, late x d when negative. 0.20 s: at 0.25 m/s a 10-12 cm stride takes about 0.45 s and a trot's swing (duty about 0.6) about 0.18 s; pipe1's real swings were 140-215 ms. The 0.5 s cap keeps long hops from paying more. A foot's first landing after reset (the soles start 2 mm up) is not a touchdown |
| diagonal symmetry | 0 (off) | when v_cmd > 0.05: +1 for a trot (diagonal pairs together), -1 for a bound (front pair and hind pair each together), 0 for a pace, a pronk or all four down |
| termination | -5 | once |

**Debounced contact.** A foot counts as lifted only after its load has stayed at or below 2 N for 10 ms
(`contact_debounce_s`). The ~2 ms breaks inside pipe1's stances are a sub-millimetre contact-model rebound (the
review of the gait analysis: sole lowest corner median 0.03 mm above the floor, carrying about 0 N), not steps; the
debounce merges them into the stance. They are **not penalised** (that would train against the contact solver).

**Two schedules** (train.py sets both in every env before each rollout, records both in the run's settings, so a
resume keeps them, and prints both on the progress line; both are in `info`):

- the **penalty scale** (`RAMPED_TERMS`: energy, slip): 0.1 at step 0, linear to 1.0 at 2,000,000 steps
  (`--penalty-ramp-start`, `--penalty-ramp-steps`), the schedule that trained pipe1;
- the **late scale** (`LATE_TERMS`: heat, saturation, and the negative part of the air-time term): 0 until 1,000,000
  steps, linear to 1.0 at 3,000,000 (`--late-ramp-from`, `--late-ramp-to`).

Why: the review of the first heat and saturation reward (applied with energy and slip from 0.3 at step 0) found that
it discouraged exploration. The initial policy's noise paid heat, saturation, slip and short-swing costs at once.
Its probes stood still or barely walked at 300,000 steps, and the review's 1.25 M-step probes confirmed the slow start.
The late schedule lets a gait form first, and the costs that shape it arrive afterwards.

**Reward per step at three stages of training** (measured; the review's method: seeds 100000-100002, delay 10 ms,
randomisation on; pipe1's policy, loaded with the gait analysis's safe loaders, in an env with pipe1's observation and
today's reward; "noise" is the initial policy's, std 0.2 around zero; "pipe1 + noise", pipe1's gait with that noise;
no falls in any case):

| Stage | Standing | Noise | pipe1 | pipe1 + noise | pipe1 + noise - standing | noise - standing |
|---|---|---|---|---|---|---|
| **0.25 m/s**, step 0 (penalty 0.10, late 0.00) | +0.100 | +0.078 | +1.553 | +1.175 | +1.075 | -0.022 |
| 1.5 M (penalty 0.78, late 0.25) | +0.095 | -0.118 | +1.401 | +0.882 | +0.787 | -0.213 |
| >= 3 M (penalty 1.00, late 1.00) | +0.081 | -0.467 | +1.149 | +0.446 | **+0.366** | -0.548 |
| **0.15 m/s**, step 0 | +0.183 | +0.375 | +1.562 | +1.191 | +1.008 | +0.192 |
| 1.5 M | +0.178 | +0.172 | +1.444 | +0.925 | +0.747 | -0.006 |
| >= 3 M | +0.163 | -0.182 | +1.238 | +0.510 | **+0.347** | -0.346 |

At step 0 exploration noise earns about what standing does (-0.022 at 0.25 m/s, +0.192 at 0.15 m/s). At full scale a
noisy walking policy still beats standing by 0.37 (0.25 m/s) and 0.35 (0.15 m/s) per step. The weights were chosen for
that from the same measurement, linear in the weights. The table's three rows per command were also measured directly
with the final defaults. At the first version's weights (heat 0.5, saturation 5.0) pipe1 + noise fell 0.03 below
standing at full scale. With saturation 2.0 alone it led by 0.18, and with saturation 1.0 by 0.25. Heat 0.3 with
saturation 2.0 meets the 0.3 margin, so those are the defaults. At full scale, noise around zero costs heat -0.22,
saturation -0.08, slip -0.10, energy -0.03 and air-time shortfall -0.13 per step.

Standing never pays less than falling at once (the step reward stays positive), so the policy has no reason to end
episodes early.

**Termination:** tilt over 35 deg, chassis below 0.6 x standing height, any geom other than the four soles touching
the floor (checked every physics step), or a non-finite or diverging simulation. **Truncation:** 20 s (1,000 steps).

**Determinism:** every random draw comes from the env's `np_random`; the same reset seed gives identical
observations (tested over 50 steps with randomisation and noise on).

## Rough terrain, pushes, low friction and a drift cost (run 3, decision D18)

Added 19 Nov 2026 after the run2b analysis (`analysis/REPORT_RUN2B.md`: on steps, ramps and slippery floors that
policy slowed sharply and drifted sideways up to 1.3 m in 18 s; it had only ever seen a flat floor). **Every new
setting defaults to off**, and with them off the environment is bit-identical to the one before (checked against the
previous commit's `trike_env.py`, both models, randomisation on and off), so pipe1's and run2b's recorded configs
load and measure as they did. `configs/run3_terrain.json` turns them on; the notebook's `ENV_PRESET` passes it to
train.py as `--env-config`.

- **Terrain** (`terrain_boxes`, `terrain_*`): a pool of boxes, each the only geom of its own massless mocap body
  with the floor's contact settings, laid out at every reset along the robot's heading from 0.4-0.8 m ahead of the
  chassis: flat (30 % of episodes), a run of small steps (35 %: plateaus 0.3-1.0 m long, each 1-4 cm up or down,
  never below the floor or above 12 cm), or a ramp up (2.5-10°, 0.8-2.0 m), a plateau (0.4-1.5 m) and a ramp down
  (35 %). The boxes span 2 m to each side. Why mocap bodies: a box on the world body cannot be moved at runtime,
  because MuJoCo fixes the world body's collision bounds when the model is compiled (tried: a box moved and resized
  at runtime was simply passed through). Mocap boxes, with their size and bounds updated, are hit exactly. The policy
  does not see the terrain (no camera or height map in the observation); it must cope by feel.
- **Height on terrain:** the height reward and the fall test use the chassis height above the terrain under it
  (the episode's profile), not above the floor, so climbing a ramp costs nothing.
- **Pushes** (`push_*`): in half the episodes, one sideways push on the chassis at a random time between 3 and 17 s,
  0.1 s long, of impulse robot mass x dv with |dv| up to 1.0 m/s free-body (run2b fell at 1.25 m/s in the sweep).
- **Low friction** (`friction_low_hard`): the friction draw's lower end moves from 0.6 to 0.3.
- **Schedule:** terrain heights and slopes, push sizes and the friction low end all scale with one terrain scale,
  0 until 1 M steps and full at 4 M (`--terrain-ramp-from`, `--terrain-ramp-to`), so the policy first learns to
  walk on the flat as run2b did. Since 19 Nov 2026 train.py sets all three schedules' scales before learning
  starts: SB3 resets the envs before any callback runs, so until then each session's first episode in every env
  ran at scale 1 (full penalties, and full terrain). This applies to every config, so a run2b-config run at this
  commit no longer reproduces run2b's first episodes exactly (a negligible difference).
- **Drift cost** (`w_lateral_drift` 100, `lateral_drift_tau_s` 1 s): x (sideways speed along the reset heading's
  left axis, low-passed at 1 s)^2. It prices a steady sideways drift and leaves the trot's step-to-step sway almost
  free (w_lateral already prices that). On run2b's policy it would cost 0.004 per step on the nominal floor and 0.09
  at friction 0.2 (4 episodes each at 0.25 m/s, measured with this env), against a tracking reward of 1.2-1.45 per
  step there; at w 40 the slippery case would have cost only 0.04, too little to matter.
- **The capsule foot and the slip cost** (D19, 22 Nov 2026; `slip_at_contact`): run 3 trains on the capsule-sole
  model `trike60_C_closed_capsule` (the foot study's `capsule_r24`, byte for byte; `sim/REPORT_FEET60.md`). The slip
  cost takes each loaded sole's speed at the mean point of its contacts instead of at the sole's origin, so a sole
  that rolls without sliding is not charged for rolling. In the frozen model-based trot (two episodes per foot,
  friction 1.0) the cost at the capsule's origin is 3.6x the cost at its contacts, against 1.1-1.2x for the box
  soles. The ratio is smaller on slippery floors and on terrain (1.3x at friction 0.3; 1.1x with terrain and pushes
  at full scale, with run2b's policy on the capsule): the option matters most on grippy flat floors. Default off, so
  earlier runs load and train as before. Training-time evaluations still take slip at the sole's origin.
- **Evaluations stay on the flat floor:** train.py's evaluations during training, evaluate.py, the gait report and
  the robustness sweep turn terrain, pushes and the low-friction end off (`flat_floor`), so run 3's numbers compare
  with run2b's directly; the sweep adds its own steps, ramps and pushes.
- **Cost:** about 3 % fewer env steps per second (378 against 390 in one process).
- **Local probe (development, one seed, not judged; 19 Nov 2026):**
  - Setup: the run-3 preset, 3 envs, seed 0, 1.5 M steps on a compressed schedule (terrain from 0.3 M to 1.2 M,
    late penalties from 0.3 M to 0.9 M, energy and slip ramped over 0.6 M).
  - The probe was stopped by a time limit at 1,000,002 steps; train.py saved and recorded the stop, and it resumed
    from that pair.
  - Flat-floor evaluations (one deterministic episode per command, 10 ms delay): no falls at any of them; at 0.25
    m/s, 0.239 m/s at 0.5 M, 0.235 at 0.75 M, 0.236 at 1.25 M and 0.237 at 1.5 M (terrain at full scale from 1.2 M).
  - Training rollouts kept their full 20 s length up to full terrain, and averaged 993 of 1000 steps at the end (a
    few falls on terrain).
  - The worst drive in the final flat evaluation was at 1.20 of its rating; run2b's evaluations peaked at 1.14 near
    1.5 M before their late penalties brought it under 1.0.
  - It shows that walking survives the terrain coming in; nothing about the final quality of a 10 M-step run.

## Steering: follow a requested direction (decision D20; built 23 Nov 2026, not yet trained)

`command_mode` in `TrikeEnvConfig`. `"none"` (the default) is the straight task, unchanged: the regression digest
holds, and no draw or observation is added. `"heading"` and `"heading_and_speed"` use the repository's shared
direction-command contract (`environments/shared/direction_commands.py`, `command_frame.py`), imported and never
edited.
- **Requests:** a seeded schedule, independent of the robot's actions, asks for a world heading and a speed. In
  `"heading"` the speed is 0.25 m/s, or a stop; in `"heading_and_speed"` it is drawn from 0.1-0.27 m/s. Requests
  change every 3-5 s, by up to 60°.
- **The policy's command:** a bounded proportional adapter turns each request into a body-frame command: forward
  speed (slowed while turning), 0 sideways, and a turn rate up to 0.4 rad/s. The adapter works from the robot's own
  heading estimate, the true yaw plus the gyro drift. The policy sees the command as the shared three-value segment,
  scaled to [-1, 1] and appended last (71 inputs instead of 68).
- **Turning on the spot** (this env's extension): the shared contract stops turning at zero speed. With
  `cmd_turn_in_place`, a stop whose requested heading differs from the estimate by more than 5° turns on the spot,
  at the same bounded rate.
- **Reward** (revised after the reviews of 23 Nov 2026, before any steering run):
  - speeds are taken along the robot's own heading, not the reset heading;
  - `r_track` is the shared contract's `gaussian_tracking_reward` times `w_track`: one Gaussian of the forward speed
    error (`track_sigma`, as the straight task's) and the turn-rate error averaged over about one stride (low-passed
    with `track_yaw_tau_s`, 0.5 s; `track_yaw_sigma`, 0.2 rad/s). Being one Gaussian, standing still while asked to
    walk or turn earns about nothing (a separate turn bonus, the first design, paid standing still). The trot's
    step-to-step sideways sway and yaw wobble stay out of it, priced per step by `r_lateral` and `r_yaw_rate` as in
    the straight task: with the per-step sideways speed and turn rate in it, two trained straight gaits (run2b,
    pipe1) scored 0.86 and 0.41 per step instead of their straight-task 1.44 and 1.32; as built, 1.42 and 1.29
    with a zero turn command. In the steering env on a straight request they score 0.90 and 0.99: they sit about
    5° off heading and cannot see the adapter's 0.10-0.14 rad/s corrections, as policies never trained on a turn
    command would. The averaged turn-rate error is not in the observation, so r_track depends on about 0.5 s of
    history the critic cannot see; after a request has been ignored, credit takes about 0.7 s to recover once the
    robot tracks. Worth knowing if steering training is noisy;
  - `r_yaw_rate` penalises the error against the commanded turn rate;
  - `r_heading` is 0, because holding the heading is the adapter's job;
  - the stepping terms (`r_air_time`, `r_diagonal`) count when the command asks to walk or to turn faster than
    `air_time_min_turn_rate` (0.1 rad/s), so turning on the spot is a stepping task too.
  Every other term is unchanged; `STEERING_TERMS` is empty.
- **When a command applies:** the observation a step returns already shows the command the next step is judged by;
  `set_command_target` returns a fresh observation with the new request, to act on in place of the last one.
  Turning on the spot starts beyond 5° of heading error and stops within 2.5° (hysteresis), and the heading-error
  input is zero at a stop with nothing to turn to.
- **Evaluation options:**
  - `options={"command_speed": v}` holds the reset heading at v, so the straight-line tools (gait report, sweep,
    train.py's evaluations, evaluate.py) work on a steering policy;
  - `options={"command_target": (heading offset, speed)}` holds another request;
  - `set_command_target(heading, speed)` is an operator's request mid-episode.
- **`steer_eval.py`:** a steering policy on its own request schedule, and on scripted courses (90° turns while
  walking, a stop with a 170° turn on the spot, standing; 170°, because a half turn sits on the angle wrap where
  noise picks the direction). It reports the generic measures of `telemetry/steering.py` with the 60 cm tolerances
  (0.05 m/s, 0.1 rad/s, 10°):
  - speed and turn-rate tracking judged on 0.5 s means (about one stride) after the first second, since the per-step
    values mostly read the gait's own sway (the per-step errors are kept too);
  - per request: the heading error 3 s after it and the settle time; a turn that ends in a fall counts as failed;
  - turn radius over the steps turning the commanded way, net drift per turn on the spot, and speed while standing.
  Each course gets a top view and a heading-against-time chart.
- **`evaluate.py` on a steering run:** each episode walks one straight request (a seeded speed, or a stop), so its
  numbers compare with the straight task's in distribution; the speed is drawn from its own seeded generator, so
  episode by episode the speeds differ from a straight run's at the same seed. Speeds above `cmd_speed_scale` are
  refused before anything runs. The steering behaviour itself is `steer_eval.py`'s.
- **Checks:** with `--checks-every`, a steering run's checks add the courses `turns_walking` and `turn_on_the_spot`,
  about 10 s more per check; the checks chart gives them a third panel (heading error 3 s after a request).
- **Provenance:** the gait report records the shared direction-command module's sha256 beside the run commit's, so a
  run's exported `trike_env.py` used with a newer shared module shows.
- **Not yet decided:** the settings above are starting points, set before any steering run. Run 3's terrain
  (`configs/run3_terrain.json`) is laid along the reset heading, so a robot that turns walks off it; the steering
  presets have none (D28, flat for now). The presets and the training additions are in the next section; the rules
  come in a pre-registration after the dev runs.

## The steering run's training additions (D26, D29; built and reviewed 26 Nov 2026, not yet trained)

Each is off by default: the straight task and run 3's preset are unchanged, and both regression digests hold.

- **The hottest drive's heat cost, `r_heat_peak` (D29),** a late term (x late scale, like `r_heat`):
  - each drive's (torque / continuous rating)^2, averaged over the step's physics steps;
  - from `heat_peak_start_s` (2 s, where F2's window starts) on, low-passed with `heat_peak_tau_s` (4 s), starting
    from `heat_peak_margin`^2 (`heat_peak_update`). A hot drive's average rises from the margin towards its own
    level; a cool one tends below the margin, with the 4 s average rippling with the stride;
  - cost = `w_heat_peak` x late scale x max(0, sqrt(the largest of the 12) - `heat_peak_margin`), margin 0.8
    (against a heat rule near 0.85: in D22's 128 mm target cases, the hottest single episode sat 0.035-0.057 above
    the 5-episode figure; `analytics/results/fore128-s*/`);
  - over 2-20 s, on gaits well over the margin (the D22 policies with action noise, and walking at 0.25 m/s), the
    cost comes to 0.80-0.82 of F2's own price (900 steps x the excess of F2's 2-20 s RMS); the env's cost matches
    the offline figure. Near or under the margin it charges more than F2 would: 1.26 of the price on the coolest
    group (walking at 0.15 m/s, price 0.015 per step), and about 0.005 per step (up to 0.016) on episodes whose F2
    is under 0.8, where F2 charges nothing. The cause is the stride ripple of a 4 s average and the maximum over
    12 drives, not the start;
  - the average itself weighs every step's heat alike. In the cost, the hinge and the maximum make heat early in
    the window on a cool or not-hottest drive count more than later heat (the burst figures below). While the
    hottest drive's average is over the margin, a step's heat on it reaches the return (gamma 0.99) with a third of
    the weight of an unfiltered per-step cost (alpha / (1 - 0.99 (1 - alpha)), tau 4 s). The average is not
    observed;
  - `info["heat_peak_rms"]`, the hottest drive's average (0 before 2 s), is there only when the term is on.
- **How the average was chosen** (both changes after independent reviews, before any run;
  `analysis/steer_heat_average.py`, results in `analysis/results/steer1/heat_average/`). Seven candidates were
  compared on the drive heat of the four D22 128 mm policies (72 episodes, no falls):
  - The first version started at 0 at every reset. Over a 20 s episode it priced a drive at 0.85-1.0 of its rating
    at 38-61 % of its steady cost (the first review).
  - The second was bias-corrected from 2 s. Just after 2 s it is then the mean of a few 20 ms steps, and the cost
    takes the largest of 12 such means, so it followed the stride, not the drive's RMS (the re-check). One step's
    heat at 2.02 s weighed 4.43 in the return against 0.34 at 15 s. Episodes under the margin paid 0.018 per step,
    79 % of it in 2-4 s. A 0.5 s burst at twice the rating cost 1.80-1.87 times as much at 3 s as at 10 s on the
    hottest drive, and 1.83-2.37 times on the second-hottest.
  - Starting at the margin (adopted): every step weighs 0.33 in the average; 0.005 per step on episodes under the
    margin (48 % of that small cost in 2-4 s); a burst at 3 s costs 1.13-1.24 times one at 10 s on the hottest
    drive and 1.10-1.48 times on the second-hottest. Only 4-5 % of the cost falls in 2-4 s on hot gaits: that is
    its start-up deficit (the average climbs from the margin), the same cause as its 0.8 share of the price, which
    the weight absorbs.
  - Bias correction with n floored at 200 steps came close (0.83-0.89 of the price) but weights early steps up to
    1.6 times later ones and gives burst ratios of 1.27-1.99. A floor of 50 or 100 steps, or pricing from 4 s,
    kept more of the early weight.
  - Tested: stride-like inputs (square waves of 1.0-1.8 s period) under the margin (RMS 0.75) cost at most 0.06 per
    step and under 0.005 after 8 s; over it (RMS 0.95) they cost 0.7-0.9 of F2's price, under 8 % of it in 2-4 s.
    Started in stance, the bias-corrected version fails these (0.26 on the first step, 1.16-1.25 of the price,
    20-23 % in 2-4 s); started in swing it would pass, so the test runs both phases.
- **Whole-robot mass and payload (D26),** `mass_scale_whole_range` and `payload_kg_range`:
  - two uniforms drawn after every older draw, so a seed's delay, command, friction and per-body factors are as
    without them;
  - under randomisation, every body's mass and inertia x one factor, then a payload on the chassis at its centre of
    mass, with the chassis inertia scaled by the same mass ratio (as the sweep's payload);
  - with the presets' x0.9-1.1 and 0-1 kg, the training robot's total mass spans x0.864-1.230 of nominal (mean
    1.038, 5-95 % 0.928-1.151; 2000 resets), against x0.930-1.073 from the per-body draws alone;
  - the constants are recomputed on a scratch copy of the data, so the reset state is kept: the keyframe, the leg
    joint noise and the 2 mm sole clearance. The review found that the first version, recomputing them on the live
    data, reset every episode to the model's spring pose: no leg noise, soles at 0 mm and the neck outside its
    range. Fixed and tested: with factors that change no mass, the reset state and a 200-step trajectory are those of
    the config without them, bit for bit.
- **Measuring:** the evaluations and the analysis tools turn both draws off (`flat_floor`, `gait_pipe1.flat_floor`,
  `sweep.FLAT_FLOOR`). A randomised measurement draws the per-body factors only, as for every earlier run, so the
  target case compares with F20; the heavier robots are the sweep's mass and payload points. The sweep reads the
  training ranges from the run record for its shading and its report line: delay, friction, per-body and whole-robot
  mass, payload, terrain and pushes.
- **The weight** (measured; `analysis/steer_heat_weight.py`, results in `analysis/results/steer1/heat_weight/`).
  The README's method above, on the four D22 128 mm policies in their own env with the heat cost added. The margin
  is the reward per step a walking gait with action noise earns over standing (standing pays no `r_heat_peak`; its
  hottest drive's average is 0.46-0.51 at 20 s). Each cell: the worst of the four policies' 3-seed means, +- the
  largest standard error of the four; no falls in any of the 744 episodes. The gaits are the D22 policies' final
  (10 M) ones with noise added; their logged tracking reward (`train_health.jsonl`, the mean of the rollouts within
  100 k steps of 3 M, and of the last 200 k steps) was 1.20-1.29 per step at 3 M against 1.27-1.34 at 10 M, so the
  3 M gaits walked nearly as well. 0.25 m/s,
  full scale (late scale 1, as from 3 M steps):

  | Action noise (std) | hottest drive's average at 20 s | w 0 | w 1 | w 2 | w 3 |
  |---|---|---|---|---|---|
  | 0.2 (the initial policy's) | 1.18-1.26 | +0.35 +-0.03 | +0.02 +-0.04 | -0.30 +-0.06 | -0.62 +-0.07 |
  | 0.175 | 1.15-1.21 | +0.50 +-0.07 | +0.20 +-0.07 | -0.09 +-0.08 | -0.39 +-0.09 |
  | 0.15 | 1.10-1.16 | +0.66 +-0.03 | +0.38 +-0.04 | +0.11 +-0.05 | -0.16 +-0.07 |
  | **0.125 (D22 at 3 M)** | 1.06-1.10 | +0.74 +-0.05 | **+0.52 +-0.06** | **+0.29 +-0.07** | +0.06 +-0.09 |
  | 0.11 (D22 at 4-5 M) | 1.00-1.07 | +0.85 +-0.04 | +0.64 +-0.05 | +0.44 +-0.06 | +0.24 +-0.07 |
  | 0 (deterministic) | 0.87-0.96 | +1.13 +-0.05 | +1.03 +-0.05 | +0.92 +-0.06 | +0.80 +-0.07 |

  At 0.15 m/s and noise 0.125: +0.54, +0.31 and +0.09 at weights 1, 2 and 3. The whole robot x1.1 at noise 0.125:
  +0.54 and +0.27 at weights 1 and 2 (0.25 m/s), +0.57 and +0.31 (0.15 m/s). At 2 M steps (late scale 0.5, the
  D22 noise there, 0.147): +0.68 at weight 1 and +0.54-0.55 at weight 2, both speeds.
  - Exploration noise makes the hottest drive hotter, so the cost falls mostly on the noise. The 0.3 margin rule
    above was set at the initial noise, 0.2; at that noise the D22 policies already miss it at 0.15 m/s with no heat
    cost (+0.22). So it is now applied where it binds: at the noise there is when the cost reaches full weight (3 M
    steps; at 2 M, at half weight, the margins are larger). That noise was measured in the D22 runs
    (`analysis/d22_action_std.py`, `analysis/results/steer1/d22_action_std.json`): 0.178-0.181 at 1 M, 0.145-0.148
    at 2 M, **0.123-0.128 at 3 M**, 0.104-0.111 at 5 M and 0.083-0.091 at 10 M, the four 128 mm runs within 0.008
    of each other. They used the same schedules but trained the straight task with run 3's terrain, 0-10 ms delays,
    no whole-robot draws and no `r_heat_peak` (which also prices noise), so their noise is a guide only.
  - At that noise, weight 1 keeps +0.52-0.57. Weight 2 is on the line: +0.29 and +0.31 at the two nominal speeds,
    +0.27 and +0.31 for the robot x1.1; at noise 0.12 it keeps +0.37 and +0.43, at 0.13 +0.25 and +0.34 (the cells
    scatter by about +-0.05 between neighbouring noise levels). Weight 3 does not (+0.01-0.09). The dev runs use
    **1 and 2** (`configs/steer1_dev_heat1.json`, `steer1_dev_heat2.json`): one well inside the rule and one on its
    line, so that a heat effect has room to show. Earlier choices, before the D22 noise logs were read (changed
    before any run): 1 and 3, then 0.5 and 1 on an assumed 0.15-0.175 at 3 M.
  - Weight 2 needs its 3 M noise at or under 0.124, the bottom of the D22 range (0.1230-0.1282): at 0.15 it keeps
    +0.11 (+0.04 for the robot x1.1), and at 0.175 it falls below standing (-0.09). The judged weight is picked by a rule fixed before the dev runs
    (`runs/trike60-steer1-dev_STEPS.md`, section 5): among other measures, each dev run's own noise at 3 M against a
    limit read off these tables (0.124 for weight 2, 0.158 for weight 1).
  - Pricing every drive over the margin, rather than the hottest, costs 2.4-4.1 times as much for the policies with
    noise 0.125-0.2 (2.1 at 0.11, up to 4.3 for the robot x1.1, 4.4 for noise alone); at weight 1, noise 0.125, 0.25 m/s: 0.67
    against 0.21 per step, a margin of +0.08 against +0.52 (`heat_weight_margins_every_drive*`, priced at the same
    late scale). D29's hottest drive is kept.
  - The margins' standard errors are 0.01-0.10 on 3 seeds; figures within about 0.1 of each other are not told
    apart. These are straight-task policies; the steering task's tracking reward differs, so this guides the dev
    runs and predicts nothing.
- **The presets** (`configs/steer1_dev_heat1.json`, `steer1_dev_heat2.json`): speed and heading commands
  (`heading_and_speed`); flat floor (D28); pushes in half the episodes, up to 1.0 m/s; the slippery end at 0.3; the
  drift cost and slip at the contact as in run 3; servo delay 0-20 ms; the whole-robot factor x0.9-1.1 and a 0-1 kg
  payload; today's `w_heat` 0.3 kept, plus `r_heat_peak` at 1 or 2 (margin 0.8, tau 4 s, from 2 s). The model is
  `trike60_C_closed_capsule_fore128` (D25); the notebook refuses a steer1 preset with another MODEL. The presets set
  no schedule. The learning rate follows TOTAL_STEPS; the late ramp (1 M to 3 M) and the push and friction ramp
  (1 M to 4 M) are fixed in steps, so a run shorter than 3 M never prices heat at full weight (hence D31: the dev
  runs use the 10 M schedule and are read at 5 M).
  - The 20 ms stress point is the top of the trained delay range (drawn in 1/21 of the episodes, 15-20 ms in 6/21),
    as 10 ms was the top of 0-10 ms before; widening to 25 ms was considered and not adopted, so that the trained
    range does not grow beyond what the stress point needs.
- **During training,** the evaluations' per-drive RMS over rating (`eval_during_training.jsonl`) shows the hottest
  drive; `train_health.jsonl` shows `r_heat_peak`.

## Heat, battery and loads over a demo (decision D21; `demo_course.py`)

These are simulation results plus model estimates, not measurements. `python demo_course.py --out DIR
[--ckpt-dir RUN]` runs two stages.

**1. Segments, simulated.** One 20 s episode per activity: walking at 0.25 m/s, walking at 0.15 m/s, and standing.
Each is on the nominal model with sensor noise and a 10 ms delay. The walker is the model-based trot on the capsule
feet, or a checkpoint. Each segment records, every 1 ms:
- each drive's torque and speed;
- the electrical estimate (copper loss, the drives' net draw on a shared bus, the braking surplus);
- the joint bearing loads and the foot forces.

**2. The demo, modelled.** A loop (default 90 s walking, 30 s standing) is repeated for one pack's runtime (or
`--minutes`), from 25 °C in open air (`--room-c` sets another room: D48 checks at 35 °C, `analysis/REPORT_HEAT_35C.md`),
through two models:
- **winding temperature per drive:** a first-order model calibrated so the maker's thermal-equilibrium torque (7.5
  N·m for the X4-36-L; not its 10.5 N·m rated torque) reaches 100 °C at 90 min. The X6-60 has no published table,
  so the X6-50-L's (12 N·m) stands in, as `sim/data/drives.json` records. The thermal time constant is given at 4,
  8 and 15 min: fitted with this model to the maker's overload figures, 6.5-15.3 min (X4-36-L, 10.5 N·m to 100 °C in
  5.4-12.4 min), 4.6-4.7 min (X4-36-L, 21 N·m for 44 s) and 8.1 min (X6-50-L, 20 N·m to 100 °C in 4 min 9 s);
  calibrated at the X4-36-L's 0 rpm equilibrium torque instead, 4.4-10.0 and 3.5 min. Each drive also gets the
  middle of its settled temperature cycle when the loop runs without a break;
- **battery:** 146.5 Wh, 80 % usable, 12S, 44.4 V nominal (12 × 3.7 V) and 42.0 V at low charge, plus a 30 W hotel
  load (also 25 and 40 W). Leg drives only, copper loss at 25 °C, no driver, converter, iron or gearbox losses: see
  `telemetry/power.py`.
A segment that falls makes the demo INVALID in the JSON, the printout and the chart. Not covered: turns, starts and
stops (the segments are straight and the first 2 s are not measured), and back-to-back demos beyond the
"without a break" figure.

**Reading of 23 Nov 2026** (model-based trot, capsule feet; re-run after the review; simulation plus estimates, not a
judged result). From `demo_course.py`'s printout:
- **Drive draw:** 45.0 W walking at 0.25 m/s, 46.1 W at 0.15 m/s, 10.1 W standing; 1 ms peaks of 407 W and 597 W.
- **Heat:** over one pack (106 min), the hottest windings are the pitch drives, peaking at 50.6-51.6 °C over the
  4-15 min range. Running the loop without a break, their temperature cycles around about 50 °C, the peaks about
  1 °C higher. That is far from 100 °C, but museum-shell airflow is not modelled.
- **Battery:** 106 min per pack at 66 W mean (115 min at a 25 W hotel load, 92 min at 40 W). Pack current at low
  charge: mean 1.6 A, RMS 2.1 A; peak 10.4 A on the default loop, and 14.9 A in the 0.15 m/s segment, which the
  default loop leaves out.
- **Braking:** the drives' net surplus peaks at 10 W on the loop (25 W in the 0.15 m/s segment), below the 30 W
  hotel load, which takes it; the largest single braking event is 0.05 J on the loop and 0.14 J at 0.15 m/s. The
  power board's ideal diodes keep it from the packs.
- **Loads** (the largest over the three segments, all at 0.15 m/s):
  - joint reaction force: front elbow (right_fore_flexion), 99th percentile 2.23 body weights, maximum 3.34;
  - bending: front pitch, 99th percentile 25.9 N·m (right_fore_pitch), maximum 39.1 N·m (left_fore_pitch);
  - front foot (right_fore_sole): up to 3.26 body weights.
  These feed the bench plan. Every 60 cm leg body has one joint. A body with several joints (the T. rex hips) is
  treated as MuJoCo builds it, a serial gimbal: each bearing's bending includes the other drive's torque, as it
  does in such a joint (tested against an explicit gimbal); a parallel-actuated joint would need its own model.

## What training prints and writes

Every `--log-every` steps (100,000): steps, rate, the penalty scale, and the **training rollouts' (stochastic
actions)** recent episodes. Those numbers come from exploration and can look like walking while the policy's mean
stands still (the review's 300k-step probe did exactly that). The honest signal is the **deterministic evaluation**
every `--eval-every` steps (250,000): one episode at each of 0, 0.15 and 0.25 m/s, policy mean, randomisation on,
delay fixed at 10 ms, fixed seeds; two printed lines (the speeds; then the drive, slip and step figures at the highest
command) and one line in `eval_during_training.jsonl` with every measure for every command. Every
`--checkpoint-every` steps (250,000): `Saved checkpoint pair at <n> steps: ...`.

**Gait and drive measures** (`gait_metrics.py`, also in `evaluate.py`'s JSON for every episode and block): the
definitions of `analysis/gait_pipe1.py` (window t >= 2 s; a test applies that script's `measures` to the same logged
episode and compares), logged every physics step:

- per leg drive (12): RMS torque and RMS / continuous rating; the share of physics steps at >= 99 % of its NOMINAL
  force limit (21 / 21 / 37.8 / 18.9 N·m) and, separately labelled, of the EPISODE'S randomised limit (0.85-1.0 x);
- per drive group (roll, pitch, front elbow, hind knee): the worst joint's RMS / rating and its name, and the group's
  at-limit shares;
- per foot: stance slip ratio (sole horizontal travel while loaded / chassis forward travel), debounced touchdowns
  per second (and the raw count at 2 N, as the analysis counted it);
- mean leg power and cost of transport; sideways drift (m) and heading change (deg) over the window.

**Training health** (`telemetry/health.py`, since 23 Nov 2026). It is always on and only reads: a test, and the
review's bit-for-bit comparison with commit efd1e1c7, show training is the same with and without it. It writes one
line per rollout in `train_health.jsonl`, with the update that followed it, and records:
- SB3's own diagnostics: approx. KL, clip fraction, entropy, explained variance, value and policy losses, learning
  rate, action std;
- the gradient norm after PPO's clipping;
- the per-step mean of every `r_*` reward term and `reward_total` (train.py passes these prefixes; the module's
  default is `reward_` only, because in the species envs `r_` also means "right");
- how many episodes ended, and why: `termination`, or `time_limit` for a truncation, or `unlabelled`;
- VecNormalize's observation and return variances.

**Periodic checks** (`telemetry/checks.py`, off unless `--checks-every N`): every N steps, two deterministic episodes
(seeds 300000-300001) at each robustness-sweep point named by `--checks`. The points are `step_3cm`, `ramp_8deg`,
`friction_0.3` and `push_1.0`, on the run's own config at the nominal model, 0.25 m/s and 10 ms. Each check writes
one line in `checks_during_training.jsonl` and prints one line. These are progress readings, not a verdict.
- **Cost:** training pauses while a check runs. A check took 16-19 s with a standing policy and 26 s with a walking
  one on this machine's 4 slow cores. At Colab's ~690 env steps/s, a check every 1 M steps adds about 2 % to a run,
  and every 250,000 steps about 7 %. Use 1,000,000 unless a run needs finer readings.
- **Monitoring, not a setting:** the checks are not among the settings a run keeps (each session's arguments are
  recorded), so a resume may turn them on or off.
- **Failures:** a point whose episode fails is recorded as an error, with a warning; training goes on.
- **Refused before start:** a model outside `sim/generated` (the sweep rebuilds the model from there by name), and
  `TRIKE_ENV_DIR` being set.

**Charts:** `python telemetry/health_dashboard.py --label NAME --health RUN/train_health.jsonl --checks
RUN/checks_during_training.jsonl --out DIR` draws `training_health_NAME.png` and `training_checks_NAME.png`.

## What was tested (locally, CPU, 18 Nov 2026)

`python -m pytest hardware/triceratops_build/rl/tests` (by hand): `56 passed in 160.01s (0:02:40)`. They cover:

- **Env:** shapes and finiteness; the reward is the sum of its terms; `check_env`; zero action stands 2 s; the delay
  FIFO is exact for k = 0, 3, 10; the action filter is first order and cuts alternating jitter.
- **Randomisation and noise:** the delay is drawn per episode or fixed; targets are clipped; randomisation stays
  within its ranges and is restored bit-exactly; same seed, same first 50 observations.
- **Observation:** every slot's content (the raw previous action too, and that the action-rate penalty equals the
  one computed from the observed raw action); its signs (forward speed, roll, yaw); foot-load noise of the configured
  size; the heading drift is a random walk, and off when configured off.
- **Reward:** the tracking terms' shape (peak, value at one sigma, monotone, standing below moving for every command
  from 0.05 m/s, standing still positive); the sign of every term over 500 random inputs (heat and saturation never
  positive; air time either sign, zero below 0.05 m/s); the two schedules: the penalty scale touching exactly energy
  and slip, the late scale exactly heat, saturation and the air-time shortfall (never the gain), their defaults (0.1 to
  1.0 over 2,000,000 steps; 0 until 1,000,000 then 1.0 at 3,000,000) and their values along the way; in the env, a
  late scale of 0 zeroes heat and saturation (still measured) and leaves only the air-time gain; train.py setting both
  in every env (the progress line's values at 512 and 1,024 steps) and a resume refusing a changed late schedule; the
  diagonal score; termination on tilt, body contact
  and height; truncation; the ankle-roll model.
- **Heat, saturation, steps:** the continuous ratings equal the mass ledger's `joint_thermal_eq_nm` for both models;
  the env's heat index and saturated share equal the values computed by hand from the torques logged every physics
  step (with hard pitch and flexion targets that do saturate), and standing still costs under 0.05 per step; the
  debounce rule on a hand-made trace (2 and 9 ms gaps stay in the stance, 10 ms and longer are steps, and a foot's
  first landing after reset is not one); the env's incremental touchdown count and air-time gain and shortfall (with
  the 0.5 s cap) equal the same rule applied to its logged loads, and the air-time reward is zero at a zero command; the evaluation's measures equal `analysis/gait_pipe1.py`'s `measures` on the same
  logged episode (speed, drift, power, raw touchdowns, slip ratios, RMS / rating per group, at-limit shares against the
  nominal and the episode limit); the clock off by default (68 floats) and an older run's config (clock 1.5 Hz, without
  the new fields) rebuilding its 70-float env.
- **train.py, stop and resume:**
  - a SIGINT stop, then refusals without `--resume` and of changed settings and of a changed policy shape;
  - a resume with the step count carried over;
  - save lines at 1,024, 2,048 and 3,072 steps and periodic deterministic evaluations, each logged with the full
    drive, slip and step measures and the second printed line;
  - `ppo_in_use` equal to the recorded settings;
  - an "already complete" finished run;
  - `--allow-changes` logged, with the learning-rate warning;
  - a truncated newest pair skipped;
  - SIGINT and SIGTERM, each sent twice, saving an intact pair with no temporary files left;
  - **the notebook's `run_step` under a process-group interrupt** (two trials), and **the parent killed** (SIGKILL)
    under `--exit-with-parent`, each saving an intact pair with the session recorded;
  - broken-only checkpoints moved aside and the run restarted, with a truncated `run_record.json` moved aside;
  - **a simulated EIO** while reading an intact run: `pair_problem` raises instead of calling the pair broken, a
    transient error is re-read and the pair found, a persistent one makes train.py refuse (exit 2) with every file
    byte-identical and no `broken_*` folder;
  - a record naming saves the folder no longer lists: refused, nothing created;
  - **the reviewer's overwrite scenario**: a session SIGKILLed right after its 1024-step save (its record lists the
    save and 1024 steps reached); resumed through a wrapper where `os.path.isdir()` answers False for the folder,
    train.py resumes from that save, whose bytes stay unchanged; with `os.stat` of the folder failing with EIO it is
    re-read and refused; with the folder gone it is refused, nothing created; a session killed at 512 steps, before
    its first save, may start again; the guard's rules (missing saves, non-finite saves only, older records, a
    quarantine as a session field or as the run's own entry) one by one;
  - **the reviewer's H3**: the only save broken, the resume moves it aside and is killed at "Starting a new run",
    before its own session is recorded; the quarantine is already in `run_record.json`, and the next resume starts
    again instead of being refused for good;
  - every periodic save and the final one listed in the session's record, `steps_reached` equal to `steps_end`;
  - refusals before any change: a refused resume leaves a `.tmp` file and broken checkpoints where they were;
  - non-finite parameters saved under `nonfinite/` and never found by the scan;
  - `--exit-with-parent` with a parent that is already gone: refused, exit 3, nothing created;
  - `--allow-changes --gamma 0.98`: the saved VecNormalize carries 0.98;
  - **the notebook's own cell 5** (with cell 2's `run_step`) on a stand-in train.py whose save takes 2 s: stop
    clicked twice, the second click ignored, Drive flushed only after the save (and after a failed exit too).
- **evaluate.py:** the sweep and the target block, with the gait summary in the JSON and the TARGET gait line; the model taken from the run record (a run trained on the
  ankle-roll model is evaluated on it); VecNormalize in eval mode (training False, norm_reward False, statistics
  unchanged after 20 steps); refusal without a model; an unreadable `run_record.json` (refused without `--model`,
  evaluated with it).
- **Notebook:** the seven code cells run outside Colab (a tiny run trains; re-running cell 5 finds it complete;
  evaluation writes its JSON with the sweep); the parameter cell refusing a branch name and an unknown MODEL;
  cell 3's Python check (3.11 refused, 3.13 accepted); cell 2's shallow fetch of one commit from a local bare copy
  of this repository (one commit, no history, clean, repeatable).

Also run by hand, from copies of the reviewer's own scripts: `stop_test.py` (4 of 4 trials an intact pair and
`status=interrupted`), `stop_test2.py` (one SIGINT to train.py's whole process group: intact pair, exit 130), double
SIGINTs 0, 10, 50 and 200 ms apart (intact pairs, no temporary files), `double_stop.py` and `double_stop3.py` (the
second click ignored; the flush point reached only after the save: 2 of 2 runs, with `pgrep` narrowed to the
script's own path), `parent_kill.py` (train.py gone after 1 s, intact pair, status interrupted), `signal_storm.py`
(all four patterns: exit 130, intact pair, no traceback) and `env_recheck.py` (FIFO exact for k = 0-25 with and
without the filter; the reward figures above).

**Throughput** (this machine: 4 vCPU Intel Xeon @ 2.10 GHz, `os.cpu_count()` = 4; env steps per second during PPO
training, `steps / wall time of model.learn`; default PPO settings, 1 torch thread):

| Setup | env steps/s |
|---|---|
| 1 env, DummyVecEnv | 392-401 (three runs) |
| 2 envs, SubprocVecEnv | 535 |
| 4 envs, SubprocVecEnv | 732-757 (two 300k-step probes with the periodic evaluations; about 800 in the first version) |
| **8 envs on these 4 vCPUs, SubprocVecEnv (the pipeline test's setting)** | **932** (200k steps, periodic evaluation off, machine otherwise idle); the reviewer saw 921-932, and 901-921 with the evaluations on |
| raw env stepping, 1 process | 588-629 |
| raw env stepping, 4 processes at once | about 300 each |

At about 900 steps/s (8 envs, evaluations included), 2 million steps (the pipeline test) take about 37 minutes on
this machine; COLAB_STEPS.md uses the same figure. More envs than vCPUs is faster here because the main process's
per-step work overlaps with the workers'. The PPO update is about 5 % of the time; collecting rollouts dominates. More torch threads made it much slower (187 steps/s at 4 threads with 4 envs),
so `--torch-threads` defaults to 1. On this VM four processes step at about half the single-process speed each, so
Colab's rate is known only when the first progress lines print it.

**A 1,000,000-step probe with today's defaults** (two schedules, heat 0.3, saturation 2.0; `trike60_C_closed_ankleroll`,
8 envs on these 4 vCPUs, seed 0; one seed, so evidence, not a result; 808 env steps/s over 1,247 s). The late scale
stays 0 throughout (its ramp starts at 1,000,000 steps), so this shows the exploration-friendly start only. Periodic
deterministic evaluation (one 20 s episode per command, randomisation on, delay 10 ms):

| Steps | 0 m/s | 0.15 m/s | 0.25 m/s | At 0.25: RMS / rating roll, pitch, elbow, knee | at >= 99 % nominal | slip ratio | touchdowns/s | power | drift, heading |
|---|---|---|---|---|---|---|---|---|---|
| 250k | 0.003 | 0.008 | 0.009 | 0.24, 0.47, 0.46, 0.76 | 0 % | (little travel) | 0.4-4.4 | 3 W | +0.01 m, +5.6° |
| 500k | 0.002 | 0.019 | 0.163 | 0.56, 1.06, 0.91, 0.96 | 0 % | 0.35-0.62 | 2.2-3.9 | 31 W | -0.02 m, +0.7° |
| 750k | 0.002 | 0.102 | 0.072 | 0.37, 0.92, 0.70, 0.69 | 0 % | 0.53-1.30 | 1.1-8.0 | 14 W | -0.24 m, -17.3° |
| 1M | 0.004 | 0.042 | 0.228 | 0.53, 1.16, 1.03, 0.90 | 0 % | 0.29-0.42 | 4.3-5.9 | 41 W | -0.28 m, -5.4° |

`evaluate.py --sweep` on the final checkpoint (3 episodes per command): 0.001 / 0.105 / 0.229 m/s at 0 / 0.15 /
0.25 m/s, no falls. At 0.25 m/s:

- RMS / rating (worst joint): roll 0.59, pitch 1.27 (right fore), elbow 1.11 (left fore), knee 0.97;
- at >= 99 % of the nominal limit: 0.1 / 0 / 0 / 0.9 % (roll, pitch, elbow, knee);
- slip ratio 0.24-0.37; 3.9-5.4 debounced touchdowns per foot per second; 42 W;
- sideways drift -0.13 to -0.78 m over 18 s, heading -6.9° at worst.

So by 1 M steps a gait reaches the target speed again, and its heat is about pipe1's. That is expected before the late
costs begin. Whether the late schedule then lowers the pitch and elbow loads without losing the gait needs a run past
3 M steps. The 0.15 m/s command is tracked poorly (0.105) and the drift is large.

**Two 300k-step probes with the first version of the heat and saturation reward** (heat 0.5, saturation 5.0, both
ramped with energy and slip from 0.3 to 1.0 over 500,000 steps; air time, slip 1.0, clock off; 8 envs on these 4 vCPUs, seed 0, deterministic evaluation at 10 ms with randomisation on; one seed
each, so evidence, not a result; `evaluate.py --sweep`, 3 episodes per command, on the final checkpoints):

| Model | 0.15 m/s | 0.25 m/s | falls | RMS / rating at 0.25 (worst) | at >= 99 % nominal limit | slip ratio | touchdowns per foot per s | power |
|---|---|---|---|---|---|---|---|---|
| `trike60_C_closed` | 0.009 m/s | 0.015 m/s | 0 | knee 0.67 (left hind) | 0 % | 0.63-0.90 (on very little travel) | 0.7-5.7 | 3 W |
| `trike60_C_closed_ankleroll` | 0.033 m/s | 0.094 m/s | 0 | pitch 0.92 (left hind) | 0-0.3 % | 0.26-0.34 | 2.7-4.4 | 11 W |

Both start more slowly than the earlier reward's probe below (0.239 m/s at 300k on the closed model): the training
rollouts moved (0.08-0.09 m/s at a mean command of 0.12) but the deterministic policies, under costs already at 0.3-0.7
of full strength, mostly stood (closed) or began to walk (roll ankle; 0.124 m/s at its 300k evaluation). Whether this
is only a slower start or a standstill that persists needs a longer run; if it persists, the levers are the ramp's
start, `w_heat` and `w_saturation`; the review then measured why (above) and the two schedules replaced it. Rates
during these probes: 1,062 env steps/s (closed,
mostly standing) and 745 (roll ankle, walking, four more joints).

**Two 300k-step probes with the reward before 19 Nov 2026** (4 envs, seed 0; one seed each, so evidence, not a
result). With the initial action noise of the first version (`--log-std-init -1.0`, std 0.37) the training
rollouts moved at 0.12 m/s against a mean command of 0.12 m/s, but the deterministic evaluation at 300k steps moved
at 0.015-0.017 m/s at 0.15 and 0.25 m/s: the policy still moved through its noise (its std had barely shrunk, 0.365).
With `--log-std-init -1.6` (std 0.2, now the default) the deterministic evaluation at 300k steps gave 0.137 m/s at
0.15 and 0.239 m/s at 0.25, and `evaluate.py --sweep` (3 episodes per command, randomisation on, 10 ms) gave
-0.001, 0.064, 0.217, 0.242 and 0.246 m/s at 0, 0.10, 0.20, 0.25 and 0.27 m/s, no falls. At 0.25 m/s every foot
lifts (airborne in 27-51 % of control steps), the leg power averages 40 W, and the soles slip a lot (the summed
squared horizontal speed of the stance soles averages 0.18 m^2/s^2). That is the chassis moving with the feet
stepping and sliding after 300k steps; it is not shown to be a gait, and nothing about its quality was examined.

## What was NOT tested

- **Colab itself.** The notebook ran only outside Colab, where fetching the commit, pip, the Drive mount, the Drive
  flush and remount after a stop, and the runtime release are skipped. The shallow fetch was tested against a local
  bare copy (GitHub serves a commit by its hash, but that was not tried from here). Whether the CPU-index torch wheel
  installs on Colab's Python, how many vCPUs High-RAM gives, how far behind Drive's background upload runs (what a
  disconnect loses), and whether a run survives a closed tab are all unseen; the pipeline test (COLAB_STEPS.md)
  exists to see them.
- **Any real training.** The longest run was 300,000 steps. Whether PPO learns to walk with these rewards, how many
  steps it needs, and whether the gait respects the drives' thermal limits are unknown.
- The reward weights, action scales, filter, network size and PPO settings are untuned.
- Rendering and videos (the env has no `render`); GPU training; the ankle-roll variant beyond loading, standing, a
  1,024-step training run and one 300k-step probe.
- Transfer to the real robot: the privileged base velocity, the delay model, the heading drift and the randomisation
  ranges are simulation choices, not measured properties of the hardware.
- Two runtimes writing one run folder at once (one runtime per run, as the step sheet says): there is no lock.

## Notes

- Checkpoints are written straight into the Drive folder. A save cut short by a disconnect is skipped on resume by
  `newest_intact_periodic_pair`, which falls back to the previous intact pair (tested with a truncated zip).
- A resumed session re-seeds the envs with `seed + steps`, so it does not replay the first session's episodes.
- Old checkpoints are not pruned: about 2 MB per pair, one pair per 250,000 steps.
