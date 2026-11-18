# Triceratops 60 cm: reinforcement-learning setup (simulation only)

The training setup for a walking policy of the 60 cm, four-legged Triceratops museum robot, in MuJoCo, with
stable-baselines3 PPO. **Status: built and tested locally on short runs only. No policy has been trained; nothing
here shows that the robot walks.** The reward weights are first guesses, not tuned. Colab itself is untested.

## Files

| File | What it is |
|---|---|
| `trike_env.py` | The Gymnasium environment `TrikeEnv` and its settings `TrikeEnvConfig` (one dataclass, every range and weight with its default). |
| `train.py` | PPO training: SubprocVecEnv (or DummyVecEnv), VecNormalize, periodic checkpoint pairs, `--resume` from the newest intact pair, a `run_record.json`. |
| `evaluate.py` | Runs N episodes of a checkpoint (or a zero-action baseline) with randomisation on and the servo delay fixed at 10 ms (and optionally 0 ms); writes JSON. |
| `trike60_training.ipynb` | The Colab notebook (8 cells): parameters, checkout of an exact commit, pinned installs, Drive, training (auto-resume), evaluation, runtime release. |
| `requirements-colab.txt` | The pins the notebook installs (torch as the CPU build). |
| `COLAB_STEPS.md` | The owner's step sheet for Colab (draft; values to fill in). |
| `tests/test_trike_env.py`, `tests/conftest.py` | pytest suite (20 tests). |

## How to run locally

A Python 3.13 venv with the pinned packages (torch from the CPU index):

    python -m venv venv
    venv/bin/pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
    venv/bin/pip install mujoco==3.10.0 numpy==2.5.5 gymnasium==1.4.0 stable-baselines3==2.9.2 cloudpickle==3.1.2 pytest

From the repository root (the scripts put `rl/` and the repository root on `sys.path` themselves):

    venv/bin/python hardware/triceratops_build/rl/train.py --out /tmp/trike_run --total-steps 2000000 --n-envs 4
    venv/bin/python hardware/triceratops_build/rl/train.py --out /tmp/trike_run --total-steps 4000000 --n-envs 4 --resume
    venv/bin/python hardware/triceratops_build/rl/evaluate.py --run-dir /tmp/trike_run --episodes 10 --delays 10 0
    venv/bin/python hardware/triceratops_build/rl/evaluate.py --policy zero --episodes 3 --command 0.25
    venv/bin/python -m pytest hardware/triceratops_build/rl/tests

`--total-steps` is the run's total, not an increment. Without `--resume`, `train.py` refuses a folder that already
holds checkpoints. `--model hardware/triceratops_build/sim/generated/trike60_C_closed_ankleroll.xml` trains on the
passive-roll-ankle variant (same observation and action layout). `--env-config FILE.json` overrides any
`TrikeEnvConfig` field. Ctrl-C or SIGTERM saves a checkpoint pair and exits with code 130.

## Design

**Standalone env, not a `BaseDinoEnv` subclass.** `environments/shared/base_env.py` sizes the action space to all
`model.nu` actuators (16 here, of which the policy drives 12), writes `data.ctrl` once per control step (the servo
delay needs a write every physics step), and its `reset` gives no hook between loading the keyframe and settling
where per-episode model randomisation belongs. Fitting those would mean overriding `step` and most of `reset`, and
the base class carries the shared plant-contract and fingerprint conventions of the registered species. A ~500-line
standalone `gymnasium.Env` was clearer. Shared code is imported, not copied: `LinearSchedule`
(`environments/shared/curriculum/schedules.py`), `newest_intact_periodic_pair`
(`environments/shared/curriculum/checkpoints.py`), `load_sb3_model` (`environments/shared/policy_loading.py`) and,
in the notebook, `disconnect_runtime` (`environments/shared/notebook_runtime.py`, with an inline fallback if the
import fails in a kernel whose packages cell 3 just replaced). No manifest or registry entry.

**Model.** `../sim/generated/trike60_C_closed.xml` by default (revision 2, 12.69 kg; everything is read from the
file, so a regenerated model needs no code change as long as the actuator, sensor, joint and geom names stay).
Reset uses the `home` keyframe (all ctrl 0 = neutral targets). The soles sit exactly on the floor at the keyframe,
so reset adds uniform +-0.02 rad noise to the 12 leg joints and then places the root so the lowest sole corner is
2 mm above the floor.

**Control: 50 Hz** (`frame_skip` 20 at the model's 1 ms step). The 0-10 ms servo delay is far below one control
period, the drives are stiff position servos (kp 598.5) that track between updates, and the scripted crawl used
100 Hz; 50 Hz halves the policy evaluations per simulated second, which matters on CPU. Configurable.

**Actions.** 12 numbers in [-1, 1], order `LEG_ACTUATORS` = `{left,right}_{fore,hind}_{roll,pitch,flexion}`
(fore before hind as in the XML: left_fore, right_fore, left_hind, right_hind). Residual targets
`neutral + a * scale`, scale 0.15 rad (roll), 0.40 rad (pitch), 0.40 rad (flexion), then clipped to each
ctrlrange. Neck and tail servos are held at 0.

**Servo delay.** A per-physics-step FIFO (`deque(maxlen=k+1)`) of the 12 leg targets: each physics step appends the
current target and writes the oldest into `data.ctrl`, so a new target reaches `data.ctrl` exactly k physics steps
(k ms) later (tested for k = 0, 3, 10). k is drawn uniformly from 0-10 per episode, or fixed by
`delay_steps_fixed` (evaluation: 10), or by `reset(options={"delay_steps": k})`.

**Domain randomisation** (per episode; the model is restored from saved nominal copies, then scaled, then
`mj_setConst`; with `domain_randomization=False` it is bit-identical to the file's model, tested):

| Quantity | Range | How |
|---|---|---|
| Sliding friction, floor and 4 soles | 0.6-1.0 | one draw, set on all five (MuJoCo takes the max of a pair, so the contact gets exactly this) |
| Body masses | x0.9-1.1 | independent per body; inertia scaled by the same factor |
| Actuator kp | x0.9-1.1 | per actuator; `gainprm[0]` and `biasprm[1]` together (kv unchanged) |
| Actuator force limit | x0.85-1.0 | per actuator, `forcerange` |
| Observation noise (std) | joint pos 0.005 rad, joint vel 0.05 rad/s, gyro 0.02 rad/s, quaternion 0.01 (renormalised), base velocity 0.02 m/s | Gaussian, every observation; `observation_noise=False` turns it off |

The draws are made at every reset whether randomisation is on or off, so the rest of the episode's random stream
(delay, command, reset noise) does not shift when it is switched.

**Observation** (58 floats; `env.obs_layout` lists it): 12 leg joint angles (relative to neutral) and 12 joint
speeds; 4 passive ankle pitch angles (the model has jointpos sensors on them; the roll ankles of the variant are not
observed, having no sensor); gravity direction in the body frame (from the IMU quaternion); heading error as
sin/cos; gyro; **base linear velocity in the body frame from `qvel` (privileged: no sensor on the robot gives it;
`obs_privileged_lin_vel=False` removes it)**; 4 foot touch loads / nominal weight; previous action; commanded speed
(m/s); a 1.5 Hz gait clock (sin/cos; a feature only, no reward uses it).

**Task and reward** (per 20 ms step; every term is in `info` as `r_*`): command drawn per episode uniformly from
[0, 0.27] m/s (10 % of episodes exactly 0); heading = the yaw at reset. Forward and lateral speed are the chassis
displacement over the control step along and across that heading.

| Term | Weight | Form |
|---|---|---|
| speed tracking | 1.5 | exp(-(v_fwd - v_cmd)^2 / 0.12^2) |
| alive | 0.2 | constant |
| upright | -1.0 | gx^2 + gy^2 (gravity in body frame) |
| height | -20 | (z - z_standing)^2 (z_standing measured at init: 0.5 s standing on the nominal model) |
| lateral | -2.0 | v_lat^2 |
| yaw rate | -0.1 | yaw_rate^2 |
| heading | -0.5 | heading_error^2 |
| energy | -0.001 | mean over the step of sum \|torque x joint speed\| of the 12 leg drives (W) |
| action rate | -0.02 | sum (a - a_prev)^2 |
| foot slip | -0.5 | sum over feet in contact (> 2 N) of the sole's horizontal speed^2 |
| diagonal symmetry | 0 (off) | +1 for trot-like contacts, -1 for pace/bound, only when v_cmd > 0.05 |
| termination | -5 | once |

Measured scale (one 20 s episode each, nominal settings): standing still (zero action) earns 1.70 per step at a
0 m/s command, 0.62 at 0.135 m/s and 0.21 at 0.27 m/s; random actions with std 0.37 (about the initial policy's
noise, `log_std_init` -1) earn about -0.23 per step (energy and slip dominate). So walking at the commanded speed
(up to 1.7) pays more than standing, and standing more than flailing. These weights are a starting point for
tuning, nothing more.

**Termination:** tilt over 35 deg, chassis below 0.6 x standing height, any geom other than the four soles touching
the floor (checked every physics step), or a non-finite or diverging simulation. **Truncation:** 20 s (1,000 steps).

**Determinism:** every random draw comes from the env's `np_random`; the same reset seed gives identical
observations (tested over 50 steps with randomisation and noise on).

## What was tested (locally, CPU, 18 Nov 2026)

`pytest hardware/triceratops_build/rl/tests`: `20 passed in 68.65s` (Python 3.13.18, the versions below).
They cover: reset/step shapes and finiteness, reward = sum of terms; gymnasium's `check_env`; zero action stands
2 s (randomisation off, delay 0); the delay FIFO exact for k = 0, 3, 10; delay drawn per episode, fixed when set;
targets clipped to ctrlrange; randomisation within its ranges and restored bit-exactly to nominal when off
(including the `mj_setConst` constants); same seed same first 50 observations; termination on tilt, body contact
and height; truncation; the roll-ankle model; a train.py smoke run (2,048 steps, 2 envs) writing checkpoint pairs,
refusing to overwrite, resuming with the step count carried over, skipping a truncated newest checkpoint, finding a
finished run complete; SIGINT and SIGTERM mid-training saving a pair (exit 130); evaluate.py writing its JSON; the
notebook's seven code cells executed outside Colab (a tiny run trains, re-running the training cell resumes it and
finds it complete, evaluation writes its JSON); the parameter cell refusing a branch name.

Throughput (this machine: 4 vCPU Intel Xeon @ 2.10 GHz, `os.cpu_count()` = 4), env steps per second during PPO
training, measured as `steps / wall time of model.learn` in `run_record.json` (default PPO settings: n_steps 1024,
batch 2048, 5 epochs, [256, 256] ELU MLPs, 1 torch thread):

| Setup | env steps/s |
|---|---|
| 1 env, DummyVecEnv | 392-401 (three runs, 31k-41k steps each) |
| 2 envs, SubprocVecEnv | 535 (one run, 41k steps) |
| 4 envs, SubprocVecEnv | about 800 (798 over a 82k-step run; 815 over the first 80 s of a second run, before another job shared the machine) |
| raw env stepping with random actions, no learning: 1 process | 588-629 |
| raw env stepping, 4 processes at once | about 300 each (about 1,200 together) |

At 800 steps/s that is about 2.9 million steps per hour (20 million in about 7 h) on this machine. The PPO update
itself is about 5 % of the time; collecting rollouts dominates (an env step costs about 1.3 ms: 20 MuJoCo steps of
about 50 us plus about 0.3 ms of Python; the policy's forward pass and SB3's bookkeeping come on top). More torch
threads made it much slower (187 steps/s at 4 threads with 4 envs: the threads compete with the env workers), so
`--torch-threads` defaults to 1. On this VM four processes step at about half the single-process speed each, so how
the rate scales on Colab's processors is not known until the first Colab run prints it (COLAB_STEPS.md 3.6 (f)).

## What was NOT tested

- **Colab itself.** The notebook ran only outside Colab, where the clone/checkout, pip installs, the Drive mount and
  the runtime release are skipped. Whether the CPU-index torch wheel installs on Colab's Python, how pip's
  dependency warnings look there, whether Drive writes of checkpoints are reliable over hours, and whether a run
  survives a closed tab are all unseen.
- **Any real training.** The longest run was about 100,000 steps. Whether PPO learns to walk with these rewards,
  how many steps it needs, and whether the gait respects the drives' thermal limits are unknown.
- The reward weights, action scales, network size and PPO settings are untuned.
- Rendering and videos (the env has no `render`); GPU training; the ankle-roll variant beyond loading and standing.
- Transfer to the real robot: the privileged base velocity, the simple delay model and the randomisation ranges are
  simulation choices, not measured properties of the hardware.

## Notes

- Checkpoints are written straight into the Drive folder. A save cut short by a disconnect is skipped on resume by
  `newest_intact_periodic_pair`, which falls back to the previous intact pair (tested locally with a truncated zip).
- A resumed session re-seeds the envs with `seed + steps`, so it does not replay the first session's episodes; the
  env config must match the run's first session (`run_record.json`) or `train.py` refuses.
- Old checkpoints are not pruned: about 2 MB per pair, one pair per 250,000 steps.
