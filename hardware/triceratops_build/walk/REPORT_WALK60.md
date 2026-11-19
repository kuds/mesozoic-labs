# The model-based trot at 60 cm (task 5, D10): judged result (simulation)

19 Nov 2026. Controller `walk/walk60.py`. Development log `DEV_LOG.md`. Rules `PREREG_WALK60.md`, committed before the
judged run. Tool output `results/walk60_2026-11-19/` (`judge_walk60.py` at commit 0382ce56, clean; 114 s on 2
processes). Simulation of a model whose masses and contact are estimates.

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
  - **Front-foot bounce on grippy floors.** The front feet record about 3 touchdowns per second against a stride of
    1.43 per second: they bounce once at landing. On the slippery floor it is 1.5 per second, with no bounce.
  - **Time at the torque limit.** The front pitch drives sit at their randomised torque limit in up to 3.4-3.5 % of
    steps on grippy floors, at landing; the hind knees reach it in 1.1-1.5 % on slippery floors. At the nominal
    limit the worst is 0.6 %.
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
| At the episode's torque limit, worst joint | right fore pitch 3.4 % | left hind knee 1.4 % | right fore pitch 3.5 % | left hind knee 1.5 % |
| Joint mechanical power | 28 W | 24 W | 30 W | 24 W |

## Reported cases (5 episodes each, seeds 101-105, no rule)

| Case | Speed, m/s | Falls | Worst RMS / rating | Power |
|---|---|---|---|---|
| J1 at 0 ms delay | 0.246-0.250 | 0 | 0.75 | 28 W |
| J2 at 0 ms delay | 0.250-0.253 | 0 | 0.76 | 24 W |
| J1 at the 0.15 m/s command | 0.151-0.152 | 0 | 0.70 | 20 W |
| J1 standing (0 m/s) | 0.000 | 0 | 0.52 | 1 W |
| Nominal model, no randomisation or noise | 0.245-0.246 | 0 | 0.70 | 26 W |

The delay barely matters between 0 and 10 ms.

## Against the learned policy (run2b; not like for like)

| | Model-based trot (J1, grippy) | Learned, run2b (gait report, 0.25 m/s) |
|---|---|---|
| Floor and randomisation | friction 1.0; mass, kp and force limits randomised | friction 0.6-1.0 randomised |
| Control rate | 100 Hz | 50 Hz, 4 Hz action filter |
| Velocity input | leg odometry | the simulator's true velocity (privileged) |
| Speed | 0.248 m/s | 0.245 m/s |
| Stride | 1.43 Hz (the 0.70 s period), about 17 cm (speed / stride rate) | 1.22 Hz, 20 cm (measured) |
| Worst joint RMS / rating, mean (worst) | 0.72 (0.77), left hind pitch | 0.72 (0.77), left hind knee |
| Slip ratio | 0.06-0.09 | 0.05-0.09 |
| Joint mechanical power | 28 W | 22 W |
| Sideways drift, worst, 18 s | 0.25 m | 0.23 m |

The two reach similar loads and slip by different gaits. The model-based trot uses more power (shorter, quicker
strides) and has no privileged input. It has not been run through the robustness sweep (steps, ramps, pushes),
so its edges are not known; the learned policies' are.

## What this does not show

- Flat floor only: no steps, ramps, pushes or turns.
- One controller with frozen parameters; the verdict covers ten episode seeds per condition.
- Simulation of a model whose masses and contact are estimates.
- The 100 Hz control rate is an assumption about the robot's controller.
- Leg odometry cannot see foot slip.
- Heading and drift follow the simulated gyro drift. Holding a line over a museum floor needs an outside reference
  (vision or markers) on the robot.

## Next (proposals)

1. Run the walk through the robustness sweep (delay, friction, mass, payload, force limit, kp, pushes, steps,
   ramps), to set its edges against the learned policies'.
2. Soften the front-foot landing on grippy floors (the remaining bounce and time at the torque limit).
3. Use this walk for the museum render clips: it is deterministic and robust in these conditions.
