# Robustness sweep of a trained walking policy (simulation)

`sweep.py` takes one trained policy for the 60 cm Triceratops and changes one thing at a time to see where its walk
breaks: servo delay, floor friction, mass, drive strength, sideways pushes, and a small step or ramp. Everything here is
**simulation only**: the model's masses are estimates. A sweep of one policy describes that policy. It is a report,
not a verdict on the robot or on the method.

Results for the first policy, trike60-pipe1, are in `results/pipe1/`: `summary_pipe1.md` (read this first),
`sweep_pipe1.json.gz` (every episode; gzip), and one chart per sweep, `pipe1_<sweep>.png`.

## How to run

```
python hardware/triceratops_build/rl/robustness/sweep.py --ckpt-dir DIR --run NAME [--seeds 8] [--workers 2]
python hardware/triceratops_build/rl/robustness/sweep.py --from-json results/NAME/sweep_NAME.json  # re-summarise
python hardware/triceratops_build/rl/robustness/charts.py results/NAME/sweep_NAME.json   # redraw the charts only
python -m pytest hardware/triceratops_build/rl/robustness/tests -q
```

`DIR` holds the run's checkpoint zip (`*_<steps>_steps.zip`, the newest is used), its VecNormalize pickle
(`*vecnormalize_<steps>_steps.pkl`) and `run_record.json`. Results go to `results/NAME/` unless `--out` says otherwise.
`--sweeps delay friction mass drive push terrain` runs a subset; `--no-repro` skips the reproduction check;
`--no-charts` skips the PNGs. `--from-json` rebuilds the summaries, the Markdown and the charts from the episodes in
a finished run's JSON without simulating. Needs mujoco 3.10, numpy, torch (CPU), stable-baselines3, gymnasium and matplotlib.

The pipe1 run: 70 points x 8 seeds plus 5 reproduction episodes (565 episodes of 20 s), about 13 minutes on 2 worker
processes of a 4-vCPU machine (about 2.5 s of CPU per episode). Each worker uses one torch thread.

**Loading the checkpoint.** Nothing stored in the checkpoint is executed. The policy is rebuilt here and its weights
load with `torch.load(weights_only=True)`; the observation statistics come through a restricted unpickler that accepts
plain data only. Both loaders are `load_obs_stats` and `build_policy` from `../analysis/gait_pipe1.py`. Do not replace
them with `PPO.load` or `pickle.load`.

**Environment version.** The environment comes through `gait_pipe1.py`: `rl/trike_env.py`, or the copy of `rl/` that
the environment variable `TRIKE_ENV_DIR` points at (for example `git archive <run commit> hardware/triceratops_build/rl
environments/shared | tar -x -C DIR`, then `TRIKE_ENV_DIR=DIR/hardware/triceratops_build/rl`; a steering run, D20,
also needs the run commit's `environments/shared`, which the export brings along, and the gait report records the
hash of the shared direction-command module it imported beside the run commit's). Before the sweep, `sweep.py` re-runs gait_pipe1's target case (randomisation on, delay 10 ms, seeds
100000-100004, 0.25 m/s) and compares each episode's speed with `../analysis/results/gait_pipe1_2026-11-19.json`.
The JSON records the file's hash and whether the speeds are identical. For pipe1 they were identical to the last
digit with the current `rl/trike_env.py` (and with the export of the run's commit, 6aa3a679), so the current file was
used. Only reward terms and defaults changed since; the run record restores pipe1's settings (70-float observation,
1.5 Hz phase clock).

## What a point is

Every point runs the same seeds (200000, 200001, ...) for 20 s with deterministic actions (the policy's mean). Every
factor but the swept one is at its **nominal** value: the model as built (no domain randomisation), servo delay 10 ms
(the analysis's value and the top of the training range), command 0.25 m/s. Observation noise and the heading drift
stay on, as in training. Seeds still change the reset pose, the observation noise and the heading drift. Points that
share all settings are run once: the nominal point is in every sweep.

The nominal point is therefore the **nominal model with observation noise on**. The gait report's "nominal model" case
(`../analysis/REPORT_GAIT_PIPE1.md`) has the noise off, and its target case has randomisation on, so neither is
directly comparable with this sweep's nominal point (pipe1: 0.239 m/s here, 0.240 and 0.238 m/s there).

| Sweep | Values | How it is applied | Training range |
|---|---|---|---|
| Servo delay | 0, 2, 5, 10, 15, 20 ms; 30, 40, 60 ms beyond the requested set; at 0.25 and 0.15 m/s | `reset(options={"delay_steps": k})`: the env's per-physics-step FIFO, k ms | 0-10 ms |
| Friction | 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2; 0.15 and 0.2 beyond; the model's own as nominal | sliding friction (`geom_friction[:, 0]`) of the floor and the four soles set to the value, as the env's randomisation does (MuJoCo takes the larger of the pair, so the contact gets this value). Nominal: floor 0.8, soles 0.85, so 0.85 in contact | 0.6-1.0 |
| Mass | every body x0.8, 0.9, 1.0, 1.1, 1.2; x1.4, 1.6 beyond | `body_mass` and `body_inertia` of every body but the world x the factor, then `mj_setConst`: the env's randomisation with one factor for all bodies (a test checks they match) | x0.9-1.1 per body |
| Payload | +0.5, +1.0 kg; +2, +3 kg beyond | added to the chassis at its centre of mass, the chassis inertia scaled by the same mass ratio (spread like the chassis) | none |
| Force limit | x0.7, 0.85, 1.0; x0.4, 0.55 beyond | `actuator_forcerange` of the 12 leg drives x the factor | x0.85-1.0 |
| Position gain kp | x0.8, 1.0, 1.2; x0.4, 0.6, 1.5 beyond | `gainprm[0]` and `biasprm[1]` of the 12 leg drives x the factor (the env's rule) | x0.9-1.1 |
| Sideways push | 0.1-0.5 m/s each way; 0.75, 1.0, 1.25, 1.5 m/s beyond | a horizontal force on the chassis through `xfrc_applied` for the physics steps from t = 8.0 to 8.1 s (100 steps), along the chassis's left axis at the onset (negative: to the right). Size = robot mass x dv / 0.1 s, so dv is the velocity a free body would gain. The summary and chart quote each push three ways: dv, the impulse (N s), and the velocity change the centre of mass actually got; the feet absorb the rest | no pushes |
| Step up | 1, 2 cm; 3 cm beyond | a box added to the world in memory (MjSpec, written to a temporary file; nothing in the repository changes), its edge 0.6 m ahead of the chassis start, the floor's contact settings | flat floor |
| Ramp up | 3, 5 deg; 8 deg beyond | a long tilted box whose top face starts at floor level 0.6 m ahead | flat floor |

The force-limit and kp sweeps change the 12 leg drives only: the head and tail servos stay nominal (the env's
randomisation scales them too). The env restores the saved nominal model at every reset before the point is applied
(`PointEnv._apply_factors`), so nothing carries over between points; a test checks this array by array.

The points beyond the requested set were added after a one-seed trial run found no fall or 20 % speed drop anywhere in
the requested ranges, to find where the policy does break; the summary marks them.

## Measures

Per episode, over the window from t = 2 s to the end (gait_pipe1's `measures`, so the definitions are the gait
analysis's):

- **Fall**: the episode ended early. The env's rules: tilt over 35 deg, chassis below 0.6 of its standing height, a
  body part other than a sole touching the floor, or an unstable simulation. With a step or ramp, a body part other
  than a sole touching the added box counts too.
- **Forward speed**: chassis travel along its heading at reset over the window, divided by the window's length.
  **Speed error**: the mean speed minus the command.
- **Sideways drift**: chassis travel across that heading over the window (reported as |drift|).
- **Slip ratio**: per foot, the sole's horizontal travel while loaded (over 2 N) divided by the chassis's forward
  travel; the foot with the highest mean is reported. pipe1 slides its feet even at the nominal friction (about a
  quarter of the distance travelled, as the gait report found), so speed alone says little about grip.
- **Heat**: per leg drive, RMS torque over the window divided by its continuous rating at the joint (X4 roll and
  pitch 7.5 N·m, X6 front elbow 10.8 N·m, X4 hind knee through its 1:1 crank 6.75 N·m). Reported: the joint with the
  highest mean over episodes. RMS over 18 s stands in for steady walking; it is not a thermal simulation.
- **Time at the nominal torque limit**: per drive, the fraction of 1 ms steps at or above 99 % of the model's nominal
  limit (21 / 21 / 37.8 / 18.9 N·m). **At this point's limit**: the same against the point's limit, which differs only
  in the force-limit sweep (a drive at x0.7 cannot reach the nominal limit). Reported: the **worst single joint**
  (highest mean over episodes). The gait report's figures are per joint and per drive group (for example its 3.0 % for
  the pitch group on the nominal model is over the four pitch joints), so they are not the same number.
- **Push** (in the push sweep): the velocity change the push actually gave the robot's centre of mass along the push
  (centre-of-mass speed over the 10 ms after the push, minus the same seed's unpushed episode's at that time); the
  **extra sideways displacement**, the largest of (the chassis's sideways travel along the push minus the same seed's
  unpushed episode's) within 3 s of the onset (the two episodes are identical until the push); the **recovery time**,
  from the onset to the first time after the push from which the chassis's 0.5 s-centred mean sideways speed stays
  within 0.05 m/s for 1 s (0.1 s, the push's own length, is the floor). The unpushed episodes, measured the same way,
  set the noise floor: the gait's own sway occasionally exceeds the threshold, and the summary states their worst
  value; recovery times below it mean nothing. The largest body roll after the onset.

Per point: falls out of N; the other measures are means over the episodes that did **not** fall (the JSON keeps every
episode).

**Where it breaks**: walking outward from the nominal value on each side, the first point with a fall or a mean
speed more than 20 % below the nominal point's at the same command. "None" means no such point in the tested range.

## What this does not show

- It is a simulation of a model whose masses, friction and drive behaviour are estimates; contact is MuJoCo's soft
  contact model. A real robot will break somewhere else.
- One policy, a 2-million-step pipeline test. The figures describe trike60-pipe1, not what a better-trained policy or
  the model-based walk would do.
- One factor at a time around a nominal model: no combinations (a heavy robot on a slippery floor with a long delay),
  and no randomisation of the other factors.
- 8 seeds per point: a fall rate of 0/8 is not a proof of zero; a single fall in 8 is a real but rough signal.
- Deterministic actions; one command (two for the delay sweep); the push only sideways, only at t = 8 s, only at the
  chassis; the terrain only upward and only ahead.
- Heat is RMS torque against a continuous rating, not a thermal model; "at the limit" is a fraction of time, not
  damage.

## Files

- `sweep.py`: the sweep (point definitions, factor application, push, terrain, measures, summaries, Markdown).
- `charts.py`: one PNG per sweep (small multiples sharing the factor on x; the training range shaded).
- `tests/test_robustness.py`: the push gives the intended velocity change to a free body and lasts exactly 0.1 s in
  an episode; mass, friction, kp and force-limit scaling match the env's randomisation; the payload lands on the
  chassis only; the nominal model is restored exactly between points; the step and ramp have the intended shape and
  the floor's contact settings; the break rule.
- `results/<run>/`: `sweep_<run>.json.gz` (gzip; settings, hashes, the reproduction check, per-point summaries rounded to 6
  significant figures, and every episode on one line), `summary_<run>.md`, `<run>_<sweep>.png`.
