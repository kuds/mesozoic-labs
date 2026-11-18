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
| `tests/test_trike_env.py`, `tests/conftest.py` | 35 pytest tests, **run by hand** (they are not in the repository's `testpaths` or CI). |

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
    venv/bin/python -m pytest hardware/triceratops_build/rl/tests          # about 2.5 minutes

`--total-steps` is the run's total, not an increment. `--model` takes `trike60_C_closed` (default),
`trike60_C_closed_ankleroll` or a path. `--env-config FILE.json` overrides any `TrikeEnvConfig` field.

**Run folder rules (train.py).** Without `--resume`, a folder that already holds checkpoints is refused. With
`--resume`: no checkpoint starts the run; an intact pair continues it; if every checkpoint is broken they are moved
to `checkpoints/broken_<time>/` (kept, never deleted), recorded in `run_record.json`, and the run starts again from
step 0. A run keeps its settings (model, env config, PPO settings, envs, seed, total steps, penalty ramp, evaluation):
a resume with other values is refused, or runs with `--allow-changes`, which logs every change in the session's
entry and applies it (PPO settings are passed to the load). The policy's shape (`--net`, `--log-std-init`) never
changes. Changing `--total-steps` moves the learning rate (linear in steps / total), so a run is extended under a new
run folder instead. `run_record.json` records what each session really used (`ppo_in_use` is read back from the
loaded model). Every JSON file and every checkpoint is written to a temporary name and renamed; an unreadable
`run_record.json` is moved aside, not a dead end.

**Stopping.** The first SIGINT or SIGTERM ends training and saves a final pair (both signals are then ignored until
it is written); exit code 130. SubprocVecEnv workers ignore SIGINT; the main process closes them. A Jupyter interrupt
signals the kernel's whole process group, so the notebook starts `train.py` in its own session and forwards exactly
one SIGINT. `--exit-with-parent` (Linux, `PR_SET_PDEATHSIG`; the notebook sets it) turns the death of the notebook
kernel into a SIGTERM, so `train.py` never trains on as an orphan; output to a closed pipe no longer stops the save.

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
(roll), 0.40 rad (pitch), 0.40 rad (flexion), clipped to each ctrlrange. The filter passes a 1.5 Hz gait at 94 %
amplitude but cuts 25 Hz step-to-step jitter to about a sixth: measured with random actions (std 0.37, about the
initial policy's noise) it cuts the robot's drift from 0.080 to 0.017 m/s and the leg power from 0.35 to 0.08 reward
units per step, so exploration noise alone cannot carry the robot. It is part of the policy interface: the robot
must run the same filter. Neck and tail servos are held at 0.

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

**Observation** (58 floats; `env.obs_layout` names the slots, `env.obs_slots(obs)` splits one; a test checks every
slot against the quantity it names): 12 leg joint angles (relative to neutral) and 12 joint speeds; 4 passive ankle
pitch angles (the model has jointpos sensors on them; the variant's roll ankles have none and are not observed);
gravity direction in the body frame (from the IMU quaternion); `heading_error_sin_cos_ESTIMATE`; gyro;
`base_lin_vel_body_PRIVILEGED` (from `qvel`: no sensor on the robot gives it; `obs_privileged_lin_vel=False` removes
it); 4 foot touch loads / nominal weight; `previous_action_filtered` (the filter state, i.e. what is commanded); the
commanded speed (m/s); a 1.5 Hz gait clock (sin/cos; a feature only, no reward uses it).

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
| foot slip | -0.5 x ramp | sum over feet in contact (> 2 N) of the sole's horizontal speed^2 |
| diagonal symmetry | 0 (off) | when v_cmd > 0.05: +1 for a trot (diagonal pairs together), -1 for a bound (front pair and hind pair each together), 0 for a pace, a pronk or all four down |
| termination | -5 | once |

**Penalty ramp:** train.py scales the energy and slip penalties from 0.1 at step 0 to 1.0 at 2,000,000 steps
(`--penalty-ramp-steps`, `--penalty-ramp-start`), set in every env before each rollout, so early exploration is not
punished into standing still. Measured with this config (one 20 s episode each, nominal settings): standing still
earns 1.80 per step at a 0 m/s command, 0.31 at 0.10, 0.14 at 0.20, 0.10 at 0.25 and 0.08 at 0.27 m/s, against about
1.8 minus the penalties for moving at the commanded speed; random actions with std 0.37 earn -0.20 per step at full
penalties (-0.01 at the ramp's start). Standing never pays less than falling at once (the step reward stays positive),
so the policy has no reason to end episodes early.

**Termination:** tilt over 35 deg, chassis below 0.6 x standing height, any geom other than the four soles touching
the floor (checked every physics step), or a non-finite or diverging simulation. **Truncation:** 20 s (1,000 steps).

**Determinism:** every random draw comes from the env's `np_random`; the same reset seed gives identical
observations (tested over 50 steps with randomisation and noise on).

## What training prints and writes

Every `--log-every` steps (100,000): steps, rate, the penalty scale, and the **training rollouts' (stochastic
actions)** recent episodes. Those numbers come from exploration and can look like walking while the policy's mean
stands still (the review's 300k-step probe did exactly that). The honest signal is the **deterministic evaluation**
every `--eval-every` steps (250,000): one episode at each of 0, 0.15 and 0.25 m/s, policy mean, randomisation on,
delay fixed at 10 ms, fixed seeds; one printed line and one line in `eval_during_training.jsonl`. Every
`--checkpoint-every` steps (250,000): `Saved checkpoint pair at <n> steps: ...`.

## What was tested (locally, CPU, 18 Nov 2026)

`python -m pytest hardware/triceratops_build/rl/tests` (by hand): `35 passed in 134.68s (0:02:14)`. They cover:

- **Env:** shapes and finiteness; the reward is the sum of its terms; `check_env`; zero action stands 2 s; the delay
  FIFO is exact for k = 0, 3, 10; the action filter is first order and cuts alternating jitter.
- **Randomisation and noise:** the delay is drawn per episode or fixed; targets are clipped; randomisation stays
  within its ranges and is restored bit-exactly; same seed, same first 50 observations.
- **Observation:** every slot's content; its signs (forward speed, roll, yaw); foot-load noise of the configured
  size; the heading drift is a random walk, and off when configured off.
- **Reward:** the tracking terms' shape (peak, value at one sigma, monotone, standing below moving for every command
  from 0.05 m/s, standing still positive); the sign of every term over 500 random inputs; the penalty ramp touching
  only energy and slip; the diagonal score; termination on tilt, body contact and height; truncation; the ankle-roll
  model.
- **train.py, stop and resume:**
  - a SIGINT stop, then refusals without `--resume` and of changed settings and of a changed policy shape;
  - a resume with the step count carried over;
  - save lines at 1,024, 2,048 and 3,072 steps and periodic deterministic evaluations, each logged;
  - `ppo_in_use` equal to the recorded settings;
  - an "already complete" finished run;
  - `--allow-changes` logged, with the learning-rate warning;
  - a truncated newest pair skipped;
  - SIGINT and SIGTERM, each sent twice, saving an intact pair with no temporary files left;
  - **the notebook's `run_step` under a process-group interrupt** (two trials), and **the parent killed** (SIGKILL)
    under `--exit-with-parent`, each saving an intact pair with the session recorded;
  - broken-only checkpoints moved aside and the run restarted, with a truncated `run_record.json` moved aside.
- **evaluate.py:** the sweep and the target block; the model taken from the run record (a run trained on the
  ankle-roll model is evaluated on it); VecNormalize in eval mode (training False, norm_reward False, statistics
  unchanged after 20 steps); refusal without a model.
- **Notebook:** the seven code cells run outside Colab (a tiny run trains; re-running cell 5 finds it complete;
  evaluation writes its JSON with the sweep); the parameter cell refusing a branch name and an unknown MODEL;
  cell 3's Python check (3.11 refused, 3.13 accepted); cell 2's shallow fetch of one commit from a local bare copy
  of this repository (one commit, no history, clean, repeatable).

Also run by hand with the reviewer's own scripts: `stop_test.py` (4 of 4 trials an intact pair and
`status=interrupted`), `stop_test2.py` (one SIGINT to train.py's whole process group: intact pair, exit 130), and
double SIGINTs 0, 10, 50 and 200 ms apart (intact pairs, no temporary files).

**Throughput** (this machine: 4 vCPU Intel Xeon @ 2.10 GHz, `os.cpu_count()` = 4; env steps per second during PPO
training, `steps / wall time of model.learn`; default PPO settings, 1 torch thread):

| Setup | env steps/s |
|---|---|
| 1 env, DummyVecEnv | 392-401 (three runs) |
| 2 envs, SubprocVecEnv | 535 |
| 4 envs, SubprocVecEnv | about 800 before this revision; 732-757 now, with the filter, the drift and the deterministic evaluations (two 300k-step probes, machine otherwise idle) |
| raw env stepping, 1 process | 588-629 |
| raw env stepping, 4 processes at once | about 300 each |

At about 800 steps/s, 2 million steps (the pipeline test) take about 42 minutes. The PPO update is about 5 % of the
time; collecting rollouts dominates. More torch threads made it much slower (187 steps/s at 4 threads with 4 envs),
so `--torch-threads` defaults to 1. On this VM four processes step at about half the single-process speed each, so
Colab's rate is known only when the first progress lines print it.

**Two 300k-step probes** (4 envs, seed 0, this revision's reward, filter and ramp; one seed each, so evidence, not a
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
- Rendering and videos (the env has no `render`); GPU training; the ankle-roll variant beyond loading, standing and
  a 1,024-step training run.
- Transfer to the real robot: the privileged base velocity, the delay model, the heading drift and the randomisation
  ranges are simulation choices, not measured properties of the hardware.
- Two runtimes writing one run folder at once (one runtime per run, as the step sheet says): there is no lock.

## Notes

- Checkpoints are written straight into the Drive folder. A save cut short by a disconnect is skipped on resume by
  `newest_intact_periodic_pair`, which falls back to the previous intact pair (tested with a truncated zip).
- A resumed session re-seeds the envs with `seed + steps`, so it does not replay the first session's episodes.
- Old checkpoints are not pruned: about 2 MB per pair, one pair per 250,000 steps.
