# Pre-registration: run 3, terrain training on two seeds (written before either run starts)

20 Nov 2026. Decision D18. Simulation only. Rules, seeds, budgets and stop rules are fixed here, before training.
Any rule changed after a result will be labelled "changed after a result" and its first wording kept.

**Disclosure.** The rules below were written after the run2b analysis (`REPORT_RUN2B.md`, F14), and their thresholds
were chosen with run2b's weaknesses in view: its edges on steps and ramps, its sideways drift, and its falls from a
1.25 m/s push. Run 3 itself has not been run.

## What is trained

- **Code:** the reviewed commit that `runs/trike60-run3_STEPS.md` names (the notebook runs exactly that commit).
- **Setup:**
  - Model: `trike60_C_closed_ankleroll.xml` (revision 3).
  - Environment: `configs/run3_terrain.json` (ENV_PRESET `run3_terrain`): terrain boxes, pushes, friction low end
    0.3 and the drift cost, all on the terrain schedule (0 until 1 M steps, full at 4 M).
  - Everything else as run2b: reward, the penalty and late schedules, PPO settings, 8 envs, evaluations every
    250,000 steps on the flat floor.
- **Seeds and budget:** two runs, `trike60-run3-s0` (SEED 0) and `trike60-run3-s1` (SEED 1), 10,000,000 steps each.
  There are no other seeds and no reruns to improve a result.
- **Stop rules:**
  - A session that stops (Colab disconnect, the stop button) is resumed under the same RUN_ID.
  - A run that cannot be resumed (pipeline failure, non-finite parameters) is restarted once under a new RUN_ID with
    the same seed. It is reported, and the first attempt's files are kept.
  - A run is never extended or shortened.
- **What is judged:** the final checkpoint of each run (the last pair train.py writes at 10,0xx,xxx steps). It is not
  chosen among checkpoints.

## How it is measured

The tools at the run commit, with their definitions unchanged. Both tools measure on the flat floor (`flat_floor`); the
sweep adds its own steps, ramps and pushes.

- **Gait report** (`rl/analytics/gait_report.py`):
  - 0.25 m/s target case;
  - 5 episodes, seeds 100000-100004;
  - randomisation on, delay 10 ms, deterministic actions.
- **Robustness sweep** (`rl/robustness/sweep.py`):
  - nominal model, observation noise on;
  - 8 episodes per point, seeds 200000-200007;
  - 0.25 m/s unless stated.
- **Model export:** the run commit's `rl/` exported through TRIKE_ENV_DIR, as for run2b.

## Rules (each judged per seed)

"Nominal speed" is that seed's own speed at the sweep's nominal point. Drift is the sweep's `drift_abs_mean_m`: the
mean over 8 episodes of the absolute sideways displacement over 18 s.

| Rule | Measure | Pass when | run2b, for reference |
|---|---|---|---|
| F1 flat speed | Gait report, 0.25 m/s target case | Mean speed at least 0.225 m/s, and no falls in 5 episodes | 0.245 m/s, 0 falls |
| F2 flat heat | Same case | Every joint's RMS torque / continuous rating under 1.0 (mean of episodes) | worst 0.72 |
| T1 steps | Sweep, 3 cm step up | No fall in 8, and speed at least 80 % of nominal | 0 falls, 46 % of nominal |
| T2 ramps | Sweep, 8° ramp up | No fall in 8, and speed at least 80 % of nominal | 0 falls, 66 % |
| T3 straight on terrain | Sweep, steps 2 and 3 cm, ramps 5 and 8° | Drift at most 0.30 m at each of the four points | 1.13, 1.15, 0.66, 0.87 m |
| T4 slippery floor | Sweep, friction 0.3 | Drift at most 0.20 m, and speed at least 80 % of nominal | 0.32 m, 95 % |
| T5 pushes | Sweep, sideways push, free-body dv | No fall at -1.0 or +1.0 m/s; at most 2 of 8 falls at each of -1.25 and +1.25 m/s | 0 and 0; 4 and 7 of 8 |

## Reading

- A rule **passes** when both seeds pass it, **fails** when neither does, and is **mixed** when one does.
- The setup "handles small steps, ramps and pushes without losing flat walking" (simulation, this model) only if
  every rule passes on both seeds. Otherwise the report names the rules that pass, fail or are mixed.
- Everything else from the two tools is reported against run2b, as in `REPORT_RUN2B.md`, without a verdict. That
  includes the full gait report, every sweep point, and the time at the torque limit.
- A verdict here is about this simulation and this setup. It says nothing measured about the robot.

## Not judged here

- The policy observes the simulator's true velocity (`obs_privileged_lin_vel`).
- The terrain is blind to the policy.
- Training ranges and sweep ranges overlap in part (steps up to 4 cm and pushes up to 1.0 m/s in training), so T1,
  T2 and the 1.0 m/s part of T5 test trained conditions. Only the 1.25 m/s pushes and friction below 0.3 test beyond
  them.
