# Model-based trot, judged runs (SIMULATION)

Written by `walk/judge_walk60.py` at commit 0382ce56; rules from `walk/PREREG_WALK60.md`. Command 0.25 m/s, delay 10 ms unless stated, randomisation and noise on unless stated. Judged seeds 101-110.

**Verdict: PASS** (W1-W4 in all four judged conditions).

| Condition | W1 falls | W2 speed, mean (min), m/s | W3 worst RMS / rating | W4 worst foot slip ratio | All |
|---|---|---|---|---|---|
| J1 grippy (friction 1.0), the model's contact | 0 pass | 0.248 (0.247) pass | 0.77 pass | 0.09 pass | PASS |
| J2 slippery (friction 0.4), the model's contact | 0 pass | 0.252 (0.250) pass | 0.78 pass | 0.16 pass | PASS |
| J3 grippy (friction 1.0), stiffer contact (impratio 10) | 0 pass | 0.251 (0.249) pass | 0.86 pass | 0.08 pass | PASS |
| J4 slippery (friction 0.4), stiffer contact (impratio 10) | 0 pass | 0.252 (0.251) pass | 0.77 pass | 0.16 pass | PASS |

## Reported (no rule)

| Case | Speed, m/s (per episode) | Falls | Worst RMS / rating (joint) | At episode limit, worst joint | At nominal limit, worst joint | Heading change, worst | Drift, worst | Power | IK error |
|---|---|---|---|---|---|---|---|---|---|
| J1 grippy (friction 1.0), the model's contact | 0.250, 0.247, 0.248, 0.248, 0.249, 0.247, 0.248, 0.248, 0.249, 0.248 | 0 | 0.77 (left_hind_pitch) | 3.4 % (right_fore_pitch) | 0.5 % (right_hind_flexion) | 5.6° | 0.25 m | 28 W | 0.02 mm |
| J2 slippery (friction 0.4), the model's contact | 0.252, 0.250, 0.253, 0.251, 0.253, 0.253, 0.252, 0.253, 0.252, 0.251 | 0 | 0.78 (right_hind_flexion) | 1.4 % (left_hind_flexion) | 0.6 % (right_hind_flexion) | 5.8° | 0.23 m | 24 W | 0.02 mm |
| J3 grippy (friction 1.0), stiffer contact (impratio 10) | 0.250, 0.249, 0.252, 0.251, 0.252, 0.250, 0.250, 0.251, 0.252, 0.249 | 0 | 0.86 (left_hind_pitch) | 3.5 % (right_fore_pitch) | 0.6 % (right_hind_flexion) | 5.9° | 0.23 m | 30 W | 0.02 mm |
| J4 slippery (friction 0.4), stiffer contact (impratio 10) | 0.253, 0.251, 0.254, 0.252, 0.254, 0.253, 0.251, 0.253, 0.252, 0.251 | 0 | 0.77 (right_hind_flexion) | 1.5 % (left_hind_flexion) | 0.6 % (right_hind_flexion) | 5.6° | 0.21 m | 24 W | 0.02 mm |
| R1 J1 at 0 ms delay | 0.250, 0.246, 0.249, 0.247, 0.250 | 0 | 0.75 (left_hind_pitch) | 3.2 % (right_fore_pitch) | 0.0 % (right_hind_roll) | 5.1° | 0.19 m | 28 W | 0.02 mm |
| R2 J2 at 0 ms delay | 0.252, 0.250, 0.253, 0.251, 0.253 | 0 | 0.76 (right_hind_flexion) | 1.3 % (left_hind_flexion) | 0.0 % (right_hind_roll) | 5.8° | 0.21 m | 24 W | 0.02 mm |
| R3 J1 at the 0.15 m/s command | 0.152, 0.151, 0.151, 0.151, 0.151 | 0 | 0.70 (right_hind_flexion) | 2.9 % (right_fore_pitch) | 0.0 % (right_hind_roll) | 4.5° | 0.11 m | 20 W | 0.02 mm |
| R4 J1 standing (0 m/s) | -0.000, -0.000, 0.000, 0.000, -0.000 | 0 | 0.52 (right_hind_flexion) | 0.0 % (right_hind_roll) | 0.0 % (right_hind_roll) | 0.0° | 0.00 m | 1 W | 0.02 mm |
| R5 nominal model, no randomisation or noise | 0.246, 0.245, 0.245, 0.245, 0.245 | 0 | 0.70 (right_fore_pitch) | 2.9 % (right_fore_pitch) | 2.9 % (right_fore_pitch) | 0.6° | 0.04 m | 26 W | 0.02 mm |

Time at the torque limit is reported against the 1 % flag used for the learned policies; it is not judged (PREREG_WALK60.md). Heading and drift follow the simulated gyro drift. Simulation only.
