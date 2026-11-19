# Development log: the model-based trot (task 5, D10)

19 Nov 2026. Simulation only. Development used episode seeds 1-9; judged seeds are kept for the pre-registered
runs (`PREREG_WALK60.md`). Plant: `walk60.plant_config`: the training env on the roll-ankle model, 100 Hz control,
action filter off, delay 10 ms unless stated, randomisation and observation noise on unless stated. Measures:
`rl/gait_metrics.py`, the same measures as the learned policies' training evaluations. The command is 0.25 m/s
unless stated.

**Speeds in this log** are the env's whole-episode average (`episode_forward_speed`), which includes the 0.5 s stand
and the 1 s speed ramp; the gait measures and the pre-registered rules use the window from 2 s on, about 0.01-0.015
m/s higher (seed 9, row 11's grippy case: 0.235 whole-episode, 0.248 from 2 s, found when the judging script was
tried on that seed). Loads, slip and power are from 2 s on throughout.

| # | Change | Episodes | Result | Kept? |
|---|---|---|---|---|
| 1 | First version: trot, period 0.60 s, duty 0.55; foot targets in a frame levelled with the measured roll and pitch; stance roll/pitch correction; Raibert touchdown; leg odometry | seed 1, nominal model, no noise | Fell (tilt) at 5.3 s after a growing roll oscillation of up to ±15°. A target below the chassis in a level frame is met at any roll, so roll was held only by the weak correction | no |
| 2 | Foot targets fixed in the chassis frame (the stiff servos hold the chassis parallel to the ground under the stance feet) | seed 1, nominal; seeds 1-3 randomised with noise | Nominal: 20 s, 0.231 m/s, worst drive 0.89 of its rating, slip ratio 0.09-0.15. Randomised with noise: 0.14-0.19 m/s, slip 0.23-0.53, hind knees 1.03-1.08 | yes |
| 3 | Isolated noise from randomisation | seeds 1-2 | Noise only: 0.12-0.15 m/s, slip up to 0.58. Randomisation only: 0.23 m/s, slip about 0.1. The roll and pitch reading's noise (about 1° per sample) shook the stance legs by 1-3 mm at 100 Hz | - |
| 4 | Complementary attitude filter (gyro integrated, pulled to the gravity reading with a 0.2 s time constant) | seeds 1-2, noise only; seeds 1-3 in five cases | Noise only: 0.23 m/s, slip 0.09-0.16. Without the roll and pitch correction at all: 0.245 m/s, similar loads. Randomised, grippy 1.0, slippery 0.4, 0 ms delay, impratio 10: 0.22-0.25 m/s, no falls, worst drive 0.92 | yes |
| 5 | Heading: traced the estimated against the true heading | seeds 1-2 | The controller holds its heading estimate within about 1.5°. The true heading follows the estimate's simulated gyro drift (up to 8.5° in 17.5 s on seed 2). Sideways drift from the start line comes from it and is not observable without an outside reference | - |
| 6 | Gait sweep: press 0 / 6 mm, duty 0.6, period 0.7 / 0.5 s, lift 25 mm, leg drop 0.27 m | seeds 1-2 | Period 0.7 s was best (worst drive 0.74, 30 W). Front feet still registered about 3 touchdowns per stride | period later |
| 7 | Traced the front foot's load | seed 1, nominal, period 0.7 | The front foot struck the floor late in its swing (before the clock's stance) at 2.5-3 body weights; for a moment no foot carried load | - |
| 8 | Swing ends its arc at 90 % of the swing; swing aims at the ground estimated from roll and pitch; a foot loaded early becomes a stance foot | seeds 1-2 | One fall (period 0.7). Aiming at the attitude-estimated ground is positive feedback: a nose-down pitch makes the front feet land on shorter legs | no (ground estimate) |
| 9 | Each part on its own | seeds 1-2 | Early landing switch: slower (0.20 m/s), more slip, because it cuts the stride short. Arc ending at 90 % alone: 0.227 m/s, worst drive 0.82, 35 W | arc at 90 % only |
| 10 | Period 0.7 s with the arc at 90 %, no early landing switch; compared with 0.6 s | seeds 4-6, randomised and friction 0.4 | 0.7 s: 0.234-0.239 m/s, worst drive 0.78, slip 0.08-0.17, 24-27 W, 1.5-2.2 touchdowns per second (the stride is 1.43 Hz). 0.6 s: 0.224-0.240 m/s, worst 0.83, 1.8-5.1 touchdowns per second | 0.7 s |
| 11 | Frozen parameters across cases | seeds 7-9 | Grippy 1.0, 0 ms delay, impratio 10, 0.15 m/s, standing (0 m/s), 0.27 m/s: no falls. 0.233-0.236 m/s at 0.25 m/s; 0.142 m/s at 0.15; 0.25 m/s at 0.27; worst drive 0.78; "at the nominal limit 0.8 % or less" as printed, but that is a drive-group mean (see below); slip 0.05-0.11 | frozen |

**Frozen:** `WalkParams()` defaults as of the commit that adds `PREREG_WALK60.md`.

**Corrections from the independent review of the judged run (19 Nov 2026):**
- **Torque-limit figures (rows 10-11).** The torque-limit numbers in rows 10-11 come from `gait_metrics.one_line`,
  which prints each drive group's at-limit time averaged over its four joints. Rerun with the frozen code (seeds
  6-9), the worst single joint, the right fore pitch, was at its episode's limit in 2.5-3.1 % of steps (3.0 % at
  the nominal limit on seed 8, friction 1.0).
- **Row 10's power and touchdowns** are per-case means. Per episode they were 23-28 W, with front feet up to
  2.9 touchdowns per second (seed 6).
- **The `walk60.py` docstring** (its third paragraph) and the comment above the landing settings still describe
  the level-frame option and the ground-estimate and early-landing options. The frozen defaults are the body frame
  and both options off. The file is left unchanged so that its hash matches the judged run's record.

**Known limits seen in development:**
- Leg odometry cannot see foot slip, so the Raibert term does not correct speed lost to slip.
- Heading and sideways drift follow the simulated gyro drift.
- Control at 100 Hz, while the learned policies run at 50 Hz: an assumption about the robot's controller, to be
  checked on the bench.
