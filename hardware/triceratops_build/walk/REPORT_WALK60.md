# The model-based trot at 60 cm (task 5, D10): judged result (simulation)

19 Nov 2026. Controller `walk/walk60.py`. Development log `DEV_LOG.md`. Rules `PREREG_WALK60.md`, committed before the
judged run. Tool output `results/walk60_2026-11-19/` (`judge_walk60.py --workers 2` at commit 0382ce56, clean;
114.5 s; the worker count is from the command line and is not recorded in the output). Simulation of a model whose
masses and contact are estimates. Revised after the independent review (list at the end); the verdict is unchanged
and the judged run reproduced bit for bit.

## Answer first

- **PASS: every rule in all four judged conditions.** The conditions are a grippy floor (friction 1.0) and a
  slippery one (0.4), each under the model's contact and a stiffer contact setting. Each was run for 10 episodes
  (seeds 101-110), with 10 ms servo delay, sensor noise and randomisation on, at the 0.25 m/s command. Results over
  the 40 episodes:
  - **W1:** no falls.
  - **W2:** 0.247-0.254 m/s (rule: mean at least 0.225, none under 0.20).
  - **W3:** worst joint at 0.86 of its continuous rating (rule: under 1.0 in every episode).
  - **W4:** worst foot slip ratio 0.16 (rule: at most 0.25).
- **The drives have margin.** The worst joints are the hind pitch drives on the grippy floor: 0.72-0.82 of rating
  as a mean, 0.86 in the worst episode under stiffer contact. On the slippery floor the worst are the hind knees:
  0.72-0.73 mean, 0.78 worst.
  - The X4 hind knees stay at or under 0.78 of their rating in every condition, against the budget's average-load
    estimate of 1.00 (F12). The learned policy run2b gave 0.72 (0.77 in its worst episode).
  - That makes two controllers, in simulation and on flat floor at 0.25 m/s, that load the hind knees to about
    0.7-0.8 of rating. F12's warning describes neither, though neither settles the hardware.
- **Power:** 24-30 W of joint mechanical power, against run2b's 22 W.
- **What it does not do well (reported, no rule):**
  - **Extra front-foot touchdowns on grippy floors.** The front feet record 3.0-3.2 touchdowns per second on
    average (up to 3.9 in one episode) against a stride of 1.43 per second, so about two per stride. In the three
    grippy-floor episodes the reviewer instrumented, about a quarter (11-33 % per episode) came late in the foot's
    own stance, while the other diagonal pair landed; almost all the rest came around the foot's own landing. On
    slippery floors it is 1.5 per second on average, with single episodes up to 1.9. At the 0.15 m/s command it is
    worse: 4.4-4.7 per second on average, 5.9 at worst (about 3 per stride). There the extra touchdowns are
    re-touches just after landing (about two thirds, in two instrumented episodes), not more in late stance.
  - **Time at the torque limit, against the 1 % flag used for the learned policies.**
    - On the nominal model, where randomised limits cannot hide it, the right fore pitch drive sits at its limit in
      2.9 % of steps and the left fore pitch in 1.9 %, in every episode. Run2b's worst joint on its nominal model
      was at 0.9 %.
    - With randomisation, measured against each episode's own limit: the front pitch drives are over 1 % in 9-10
      of 10 episodes on grippy floors (up to 3.5 %); the hind knees in 2-3 of 10 episodes on every floor (up to
      1.5 %, the highest on a grippy floor).
    - In the instrumented episodes the front pitch time came at landing, the hind knees' at lift-off and mid-swing.
    - Measured against the nominal limit with randomisation on, the worst is 0.6 %. That measure cannot see a
      drive whose limit was drawn lower (0.85-1.0 of nominal), so it understates.
    - The model clamps each drive at 70 % of its maker's peak, so this is time at the model's clamp, not beyond
      what the drive can give.
  - **Straightness.** Heading change is up to 5.9° and sideways drift up to 0.25 m over 18 s. Both follow the
    simulated drift of the gyro-integrated heading estimate, which the controller cannot see.
- **What it means.** In simulation, a 60 cm Triceratops with this build's drives can trot at the museum's 0.25 m/s
  under 10 ms delay and sensor noise without a learned policy. It needs no velocity sensor beyond leg odometry and
  runs under every drive's continuous rating. The walk is a candidate for the museum renders and a baseline for the
  learned policies.

## Judged conditions (10 episodes each, seeds 101-110)

| Condition | Falls | Speed, mean (min), m/s | Worst joint RMS / rating (worst episode) | Worst foot slip ratio | Rules |
|---|---|---|---|---|---|
| J1 grippy (1.0), the model's contact | 0 | 0.248 (0.247) | 0.77 left hind pitch | 0.09 | pass |
| J2 slippery (0.4), the model's contact | 0 | 0.252 (0.250) | 0.78 right hind knee | 0.16 | pass |
| J3 grippy (1.0), stiffer contact (impratio 10) | 0 | 0.251 (0.249) | 0.86 left hind pitch | 0.08 | pass |
| J4 slippery (0.4), stiffer contact | 0 | 0.252 (0.251) | 0.77 right hind knee | 0.16 | pass |

Per condition (from `results.json`):

| | J1 | J2 | J3 | J4 |
|---|---|---|---|---|
| Highest mean RMS / rating, the four worst joints | hind pitch L 0.72, R 0.70; right fore pitch 0.67; left hind knee 0.65 | hind knees R 0.73, L 0.72; fore elbows 0.54, 0.53 | hind pitch L 0.82, R 0.76; fore pitch R 0.75, L 0.71 | hind knees 0.72, 0.72; fore elbows 0.54, 0.53 |
| Touchdowns per s, LF / RF / LH / RH | 3.0 / 3.1 / 1.5 / 1.5 | 1.5 / 1.5 / 1.6 / 1.6 | 3.2 / 3.2 / 1.7 / 1.6 | 1.5 / 1.5 / 1.7 / 1.6 |
| Slip ratio, LF / RF / LH / RH | 0.09 / 0.09 / 0.06 / 0.07 | 0.16 / 0.16 / 0.10 / 0.11 | 0.08 / 0.08 / 0.06 / 0.06 | 0.15 / 0.16 / 0.11 / 0.11 |
| Touchdowns per s, worst episode, front / hind | 3.8 / 1.6 | 1.9 / 1.8 | 3.9 / 1.9 | 1.7 / 2.1 |
| Slip ratio, worst single episode and foot | 0.11 | 0.20 | 0.09 | 0.19 |
| At the episode's torque limit, worst joint, mean (worst episode) | right fore pitch 2.5 % (3.4 %) | left hind knee 0.7 % (1.4 %) | right fore pitch 2.6 % (3.5 %) | left hind knee 0.7 % (1.5 %) |
| Joint-episodes over the 1 % flag, of 120 (joints) | 23 (both fore pitch, both hind knees) | 5 (hind knees) | 27 (both fore pitch, both hind pitch, both hind knees) | 5 (hind knees) |
| Joint mechanical power | 28 W | 24 W | 30 W | 24 W |

## Reported cases (5 episodes each, seeds 101-105, no rule)

| Case | Speed, m/s | Falls | Worst RMS / rating | Front touchdowns per s, mean (worst episode) | At the episode's limit, worst joint, mean (worst) | Power |
|---|---|---|---|---|---|---|
| J1 at 0 ms delay | 0.246-0.250 | 0 | 0.75 | 2.7-2.9 (3.3) | right fore pitch 2.6 % (3.2 %) | 28 W |
| J2 at 0 ms delay | 0.250-0.253 | 0 | 0.76 | 1.4-1.5 (1.5); left hind 1.9 | left hind knee 0.8 % (1.3 %) | 24 W |
| J1 at the 0.15 m/s command | 0.151-0.152 | 0 | 0.70 | 4.4-4.7 (5.9) | left fore pitch 2.4 % (2.9 %) | 20 W |
| J1 standing (0 m/s) | 0.000 | 0 | 0.52 | 0 | 0 % | 1 W |
| Nominal model, no randomisation or noise | 0.245-0.246 | 0 | 0.70 | 3.1-3.5 (3.6) | right fore pitch 2.9 % (2.9 %), the same at the nominal limit | 26 W |

The delay barely matters between 0 and 10 ms.

## Against the learned policy (run2b; not like for like)

| | Model-based trot (J1, grippy) | Learned, run2b (gait report, 0.25 m/s) |
|---|---|---|
| Floor and randomisation | friction 1.0; mass, kp and force limits randomised | friction 0.6-1.0, mass, kp and force limits randomised |
| Joint target range around neutral, roll / pitch / flexion | ±0.45 / 0.70 / 0.85 rad (the joint ranges clip first) | ±0.15 / 0.40 / 0.40 rad (the policy's action scales) |
| Control rate | 100 Hz | 50 Hz, 4 Hz action filter |
| Velocity input | leg odometry | the simulator's true velocity (privileged) |
| Speed | 0.248 m/s | 0.245 m/s |
| Stride | 1.43 Hz (the 0.70 s period), about 17 cm (speed / stride rate) | 1.22 Hz, 20 cm (measured) |
| Worst joint RMS / rating, mean (worst) | 0.72 (0.77), left hind pitch | 0.72 (0.77), left hind knee |
| Slip ratio | 0.06-0.09 | 0.05-0.09 |
| Joint mechanical power | 28 W | 22 W |
| Sideways drift, worst, 18 s | 0.25 m | 0.23 m |
| Front-foot touchdowns per s | 3.0 / 3.1 | 1.24 / 1.23 |
| At the torque limit on the nominal model, worst joint | 2.9 % (right fore pitch) | 0.9 % (right fore pitch) |

The two reach similar heat and slip by different gaits. The model-based trot is clearly worse on time at the torque
limit and on front-foot touchdowns, uses more power (shorter, quicker strides), and has no privileged input. It has
not been run through the robustness sweep (steps, ramps, pushes), so its edges are not known; the learned
policies' are.

## Robustness sweep (reported, not judged; plan `PLAN_SWEEP_WALK60.md`, written first)

The learned policies' sweep (`rl/robustness/sweep.py`: same points, physics changes, push, terrain, measures and
break rule) with the frozen trot as the source of the actions. Nominal model, observation noise on, no
randomisation, 8 episodes per point (seeds 200000-200007), 0.25 m/s unless stated. Tool output
`results/sweep_walk60/` (`walk/sweep_walk60.py` at 088ca493, clean, 560 episodes, 4 workers, 6.9 min). One trial
episode per terrain point was run before the plan's commit, to check the driver. The charts' captions keep the
sweep's wording for policies ("one policy", "trained on a flat floor"); here they mean this controller.

"Break": the first point outward from nominal with a fall, or a speed more than 20 % below the nominal point's.

| Sweep | Trot (walk60) | Learned, run2b | Learned, pipe1 |
|---|---|---|---|
| Delay at 0.25 m/s (to 60 ms) | **none** (0.243 m/s at 60 ms) | 60 ms (0.171 m/s, -31 %) | 60 ms (-26 %) |
| Delay at 0.15 m/s (to 60 ms) | none (0.124 m/s at 60 ms, -17 %) | 60 ms (-29 %) | none |
| Friction 0.15-1.2 | none | none | none |
| Mass x0.8-1.6, payload to +3 kg | none | none | none |
| Force limit x0.4-1, kp x0.4-1.5 | none | none | none |
| Sideways push, free-body dv | falls at ±1.5 m/s (8/8 each side); **none at ±1.25** | falls at ±1.25 (4/8 right, 7/8 left) | falls at ±1.25 (4/8 right, 1/8 left) |
| Step up | 3 cm: 0.193 m/s, **-22 %** | 3 cm: -54 % | 3 cm: -39 % |
| Ramp up | **5°: 0.179 m/s, -27 %** (3°: -18 %; 8°: -48 %) | 8°: -34 % (5°: -12 %) | 8°: -21 % (5°: -13 %) |

Beyond the breaks (mean of 8 episodes; trot / run2b):

| Point | Speed, m/s | Sideways drift | Worst joint RMS / rating | Worst joint at the nominal limit |
|---|---|---|---|---|
| Nominal | 0.246 / 0.246 | 0.12 / 0.08 m | 0.68 / 0.72 | 2.6 / 0.9 % |
| Friction 0.15 | 0.260 / 0.223 | **0.13 / 0.78 m** | 0.80 / 0.76 | 0.1 / 2.2 % |
| Delay 60 ms | 0.243 / 0.171 | 0.10 / 0.08 m | 0.74 / 1.19 | 3.7 / 10.2 % |
| Mass x1.6 | 0.247 / 0.241 | 0.13 / 0.18 m | 0.88 / 1.06 | 3.6 / 4.5 % |
| Step 3 cm | 0.193 / 0.114 | **0.11 / 1.15 m** | 0.94 / 1.40 | 5.2 / 13.0 % |
| Ramp 5° | 0.179 / 0.218 | 0.21 / 0.66 m | 0.99 / 0.72 | 6.5 / 2.8 % |
| Ramp 8° | 0.129 / 0.162 | 0.42 / 0.87 m | **1.09** / 0.78 | 7.4 / 2.5 % |
| Push 1.25 m/s left | 0.230 / 0.227 (no falls / 7 of 8 fell) | 0.12 / 0.09 m | 0.73 / 0.72 | 2.8 / 0.9 % |

**Reading:**
- On almost every axis the trot's edges lie further out than the learned policies':
  - **Delay:** it holds speed to 60 ms.
  - **Slippery floors:** it holds its line where run2b drifted.
  - **Pushes:** it survives the 1.25 m/s push that toppled both policies.
  - **Steps:** it loses less speed on a 3 cm step.
- **Ramps are its weak side.** It slows more than either policy from 3° on and breaks at 5°. On an 8° ramp its worst
  joint (a pitch drive) runs above its continuous rating (1.09) and at its limit 7.4 % of the time. Its fixed timing
  and chassis-locked leg targets do not adapt to a slope; the roll and pitch correction is the only help.
- **Torque limit:** across the sweep it spends more time at the nominal torque limit than run2b (2-4 % at most
  points against about 1 %), as in the judged run.
- **Drift:** about 0.12 m throughout, from the simulated gyro drift, where run2b's drift ranged from 0.05 to 1.15 m.
- **Not a verdict.** These are controllers as built, differing in control rate, action range, velocity input and
  gait. One controller per side; simulation.

## What this does not show

- The judged rules cover the flat floor only; steps, ramps and pushes are in the reported sweep above. No turns.
- One controller with frozen parameters; the verdict covers ten episode seeds per condition.
- Simulation of a model whose masses and contact are estimates.
- The 100 Hz control rate is an assumption about the robot's controller.
- Leg odometry cannot see foot slip. It is unbiased but noisy (in one instrumented episode the review found an RMS
  error of 0.058 m/s), so the Raibert speed term mostly reacts to noise, moving touchdowns by about ±5 mm.
- Heading and drift follow the simulated gyro drift. Holding a line over a museum floor needs an outside reference
  (vision or markers) on the robot.

## Next (proposals)

1. *Done: the robustness sweep above.* Next for the walk: slope adaptation (lean the stance targets with the
   measured pitch on a ramp, or adjust the timing) to lift the ramp edge, judged against the same sweep.
2. Soften the front feet on grippy floors: the landing (the time at the front pitch drives' limit, and the
   re-touches just after landing, which are worse at 0.15 m/s) and the extra touchdowns late in stance.
3. Use this walk for the museum render clips: it is deterministic and robust in these conditions.

## Revised after the independent review (19 Nov 2026; the verdict and every judged number unchanged)

- Time at the torque limit was understated. The first version quoted 0.6 % at the nominal limit, which randomised
  limits mask, and gave neither the nominal model's 2.9 % nor the 1 % flag counts. Both are now in, and the run2b
  comparison gained this row and the touchdowns row.
- The front feet's extra touchdowns were called "a bounce at landing". The reviewer's instrumented grippy-floor
  episodes show about a quarter (11-33 % per episode) coming late in stance; the 0.15 m/s case, worse at landing,
  is now reported.
- Per-episode worst values (slip, touchdowns) and the action-range difference from the learned policy were added.
- The summary tool's RESULTS.md gives only the worst joint and no touchdowns, though the pre-registration lists
  per-joint limit time against the 1 % flag and touchdowns: this report gives them from `results.json`.
