# Pre-registration: the model-based trot at 60 cm (task 5, D10), written before any judged run

20 Nov 2026. Simulation only. The controller's parameters, the conditions, the seeds, the rules and the stop rules
are fixed here. Any rule changed after a judged result will be labelled "changed after a result", with its first
wording kept.

**Disclosure.** The rules were written after development on episode seeds 1-9 (`DEV_LOG.md`). Those seeds showed:
- 0.233-0.239 m/s at the 0.25 m/s command averaged over the whole episode, no falls. That average includes the
  0.5 s stand and the 1 s ramp; from 2 s on, as the rules measure, one seed gave 0.248 m/s;
- worst joint at 0.78 of its continuous rating;
- slip ratio up to 0.17;
- the worst joint at its torque limit in up to 2.2 % of steps in one episode.

The thresholds below are the ones used for the learned policies where one exists (speed, heat, slip; `rl/analysis/PREREG_RUN3.md`,
`rl/analysis/PLAN_RUN2B.md`), not fitted to these results. The torque-limit measure is reported, not judged, for the
reason given below.

## What is judged

- **Controller:** `walk/walk60.py`, `TrotController` with `WalkParams()` defaults at the commit that adds this
  file. No parameter changes after it.
- **Plant:** `walk60.plant_config()`.
  - Model: the training env (`rl/trike_env.py`) on `trike60_C_closed_ankleroll.xml` (revision 3).
  - Control: 100 Hz. Action filter off (it belongs to the learned policy's interface).
  - Servo delay: 10 ms (D9).
  - Observation noise and the heading estimate's drift: on.
  - Domain randomisation: on (mass, inertia, kp, force limits), with the floor friction fixed per condition.
- **Command:** 0.25 m/s (the museum target at 60 cm). Episodes last 20 s and are measured from 2 s with
  `rl/gait_metrics.py`.
- **Judged conditions** (10 episodes each, seeds 101-110):

  | Condition | Floor friction | Contact |
  |---|---|---|
  | J1 grippy | 1.0 | the model's own (impratio 1) |
  | J2 slippery | 0.4 | the model's own |
  | J3 grippy, stiffer contact | 1.0 | impratio 10 |
  | J4 slippery, stiffer contact | 0.4 | impratio 10 |

- **Reported, not judged** (seeds 101-105):
  - J1 and J2 at 0 ms delay;
  - J1 at the 0.15 m/s command;
  - standing (0 m/s) on J1;
  - the nominal model with no randomisation or noise.
- **Tool:** `walk/judge_walk60.py`, committed with this file. It writes `walk/results/walk60_<date>/`.
- **Stop rules:**
  - One judged pass of the four conditions. No reruns to improve a result.
  - If a fault in the harness (not the controller) is found after the run, the fix is reviewed and all four
    conditions are run again on new seeds 201-210, labelled as a rerun. Both results are kept.

## Rules (each judged per condition, over its 10 episodes)

| Rule | Pass when |
|---|---|
| W1 falls | No fall in 10 episodes |
| W2 speed | Mean forward speed at least 0.225 m/s (90 % of the command), and no single episode under 0.20 m/s |
| W3 heat | Every joint's RMS torque / continuous rating under 1.0 in every episode |
| W4 slip | Every foot's slip ratio (mean of the 10 episodes) at most 0.25 |

**Reading:**
- The walk **passes** (simulation, this model, this controller) when W1-W4 pass in all four judged conditions.
- Otherwise the report names the rules and conditions that fail.
- This verdict is for one controller across ten episode seeds per condition. The learned policies are the second
  controller, compared in the report without a joint verdict.

## Reported without a rule

- **Time at the torque limit:** per joint, against the episode's own (randomised) limit and the nominal one, with
  the 1 % flag used for the learned policies. It is not judged: the model clamps each drive at 70 % of its maker
  peak (21 of 30 N·m for the X4, 37.8 of 54 for the X6 elbow), so brief time at the model's clamp means the
  controller asked for more than the model allows, not more than the drive can give.
- **Straightness:** heading change and sideways drift. They follow the simulated drift of the gyro-integrated heading
  estimate (development: up to 8.5° in 17.5 s), which no controller here can see.
- **Other measures:** joint power, touchdowns per second, and the inverse-kinematics error.

## Not shown

- Simulation of a model whose masses and contact are estimates.
- No terrain, pushes or turning.
- The 100 Hz control rate is an assumption about the robot's controller.
