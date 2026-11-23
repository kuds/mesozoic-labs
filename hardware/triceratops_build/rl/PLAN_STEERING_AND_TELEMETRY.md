# Plan: steering and training telemetry for the 60 cm build (decisions D20, D21)

23 Nov 2026. A plan, not a result. Nothing here changes run 3: both seeds train at commit efd1e1c7, and run 3 is
measured with that commit's `rl/` (`analysis/PREREG_RUN3.md`).

## What the owner chose

- **D20:**
  - "Follow the direction" means speed and heading commands that can change at any time, including turning on the
    spot. An operator or a path follower supplies them.
  - Build it now; train it after run 3's results.
  - Steering comes first. Real-robot sensing (no simulator-only velocity) is the run after.
- **D21:** add the following, generic where possible, so a T. rex or a larger Triceratops build can reuse them:
  - heat and battery over a demo;
  - training health;
  - terrain and turning checks during training;
  - loads for the hardware.

## Reused, not rewritten: the repository's shared contracts

`environments/shared/` already defines direction following for every species (`docs/TRAIN_DIRECTION_AND_TERRAIN.md`,
`docs/BEHAVIOR_RECIPES_PLAN.md`). The build env imports these modules and **never edits them**: they serve every
species, the T. rex included, and the T. rex is out of bounds.
- `direction_commands.py`: `DirectionCommandConfig` and `DirectionCommandController`, a seeded schedule of requested
  world headings and speeds that does not depend on the robot's actions. Also `tracking_metrics` and
  `gaussian_tracking_reward`.
- `command_frame.py`: the command segment. Three inputs (`v_x_cmd`, `v_y_cmd`, `yaw_rate_cmd`), pre-scaled to
  [-1, 1] and appended last to the observation. Mode `"none"` is byte-inert.
- `diagnostics.py`: a per-component reward-breakdown callback. *Checked for fit (23 Nov), not used:*
  - it lists the species envs' reward names explicitly (`reward_*`), and the build env's terms are `r_*`;
  - it writes TensorBoard and an `.npz` beside the stage, where this pipeline writes JSON lines to Drive;
  - it imports the stance diagnostics of the species envs.
  `telemetry/health.py` does the same job generically. It finds terms by prefix, reads either termination key, and
  writes plain JSON lines.

The build env stays standalone (not a `BaseDinoEnv`; `README.md` says why). It adopts the command contract with the
same names, ranges and modes, so a policy interface built here matches the repository's.

## New and generic: `rl/telemetry/`

Robot-agnostic modules, with no Triceratops constant inside. Each takes a MuJoCo model and a drive table in the
`sim/data/drives.json` schema (a joint-to-drive map, maker figures, sources), and is tested on its own. They are
not wired into any T. rex code.

| Module | What it gives | Inputs |
|---|---|---|
| `health.py` | An SB3 callback writing `train_health.jsonl`: the learning algorithm's own diagnostics (approx. KL, clip fraction, entropy, action std, value loss, explained variance, learning rate, gradient norm), per-rollout means of every reward term (by prefix: `reward_` by default, `r_` added by the build), and termination causes. A training-dashboard panel reads it. | SB3 logger, step `info` |
| `checks.py` | Periodic checks during training, every N steps: named points (a step, a push, a slippery floor, a turn), a few deterministic episodes each, through a robot-supplied `run_point`, one line each in `checks_during_training.jsonl`. Not judged: the final checkpoint is what is judged. | `run_point`, point list |
| `thermal.py` | A first-order winding-temperature model per drive, driven by the copper loss of its torque: the temperature over a long episode, the time to a limit, and cool-down. Calibrated from the maker's temperature-rise figure (`thermal_eq_nm`: 90 min to 100 °C, open air). The time constant is not published as such; as built, it is a 4-15 min range fitted to the maker's overload figures, labelled as such. | drive table, torque trace |
| `power.py` | Per drive and pack: electrical power (mechanical plus copper loss, with regeneration shown separately), peak and RMS pack current at the pack voltage, energy per metre, and runtime for a demo duty mix. Generalises the gait report's power section. | drive table, pack (Wh, V, usable share), hotel load |
| `loads.py` | Distributions of joint torque and joint reaction force (bearing loads), and foot impact forces, per joint and per foot, for the bench plan and part sizing. | MuJoCo model and data per step |

A **demo course**, a scripted command schedule (walk, turn 90°, stop, turn on the spot, walk), runs any policy or
controller for 15-20 simulated minutes. It reports heat, battery and loads in one place.

*As built (`demo_course.py`, 23 Nov, revised after its review):* straight segments only (walking at 0.25 and 0.15
m/s, standing), each simulated for 20 s and measured from 2 s, then a walk-stand loop modelled for one pack's runtime
(106 min with the trot). Done as the plan's `power.py` row asks: RMS pack current and energy per metre per segment;
added: the largest braking event, runtime over the hotel-load range, the temperature cycle without a break. Not done yet:
- turns and turning on the spot (they wait for a steering policy; the trot walks straight);
- starts and stops (the first 2 s of each segment are not measured, about 10 transitions per 20 min);
- joint-torque distributions (the heat model takes each drive's mean squared torque only);
- foot impact rates (only percentiles and maxima of the contact forces are kept);
- the head and tail servos in the battery figure.
The thermal time constant is no longer an open guess: the maker's overload figures fit about 4-15 min (`thermal.py`).

## Steering in the build env (inert by default)

- **Command modes:** `command_mode` of `"none"` (today's task, byte-identical, the regression digest holds),
  `"heading"` or `"heading_and_speed"`, with one `DirectionCommandConfig` scaled to 60 cm. The cruise speed is
  0.25 m/s. The turn rates and heading-change sizes are set from what the model-based trot and run 3 show.
- **Observation:** the shared 3-input command segment, appended last. Under `"none"` the observation is unchanged.
- **Reward** (as built, revised after the reviews of 23 Nov): one shared Gaussian of the forward speed error and the
  stride-averaged turn-rate error, so standing still while asked to move earns about nothing, while the trot's
  step-to-step sway and wobble stay priced per step as in the straight task. The
  requested-heading term planned here was dropped before any run (`r_heading` is 0): the command adapter closes the
  heading loop. The stepping terms also count when the command asks to turn. The heat, torque-limit, slip and drift
  costs stay.
- **Evaluation measures:**
  - speed and turn-rate tracking error;
  - heading error a set time after each change;
  - turning radius;
  - settle time after a change;
  - turning on the spot: rate and drift.
  As built, these are `steer_eval.py`'s schedule and courses and the checks' course points; the gait report and the
  sweep stay straight-line (a steering policy walks them on a straight request).
- **Training:** a new preset, rules pre-registered after run 3's results, and step sheets. Run 3's terrain is laid
  along the reset heading, so the steering preset needs terrain around the start, or none. The notebook gains the
  preset in its list. It may also gain an option to start from an earlier run's policy (train.py has none today);
  that is decided with the pre-registration.

## Order of work

Each step has tests, and its own independent review before the next relies on it.
1. Training health (`health.py`) and periodic checks (`checks.py`). The steering run needs both to be watched.
2. Steering in the env, its measures, and the turning cases in the gait report and sweep.
3. Heat and battery over the demo course (`thermal.py`, `power.py`).
4. Loads (`loads.py`).
5. After run 3's results: the steering run's pre-registration, preset, notebook update and step sheets.

## Limits

- The drive data are maker figures, not measured. The thermal time constant and the copper-loss inputs are
  assumptions until the bench measures them.
- The model's masses are estimates (F10, F12). Everything trained before the bench is a method to retrain with
  measured values, not the final policy.
