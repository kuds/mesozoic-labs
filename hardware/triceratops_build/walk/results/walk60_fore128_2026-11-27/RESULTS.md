# Model-based trot on `trike60_C_closed_capsule_fore128.xml`: results (SIMULATION)

Written by `walk/fore128_walk60.py summarise`. Model `trike60_C_closed_capsule_fore128.xml`. Every number here is simulation, of a model whose masses and contact are estimates; nothing is measured. Reported, not judged: no pre-registration covers this model. PREREG_WALK60.md's W1-W4 were written for the 188 mm roll-ankle model (trike60_C_closed_ankleroll.xml); shown beside the measurements, not judged. PREREG_WALK60.md judged only J1-J4; its thresholds on every other case here are for orientation only.

- Speed: the mean forward speed from 2 s to the end of the 20 s episode, m/s, over the episodes that did not fall. In the sweep the push comes at 8 s, so a slower gait after it is averaged with the 6 s before it, and the sweep's break rule (a fall, or a mean more than 20 % below the nominal point's) uses this mean. Where `fore128_slow_gait.py` measured the same episodes (checked seed by seed: falls and 2-20 s speeds for sweep rows, whole-episode and 2-20 s speeds for judged conditions), the 10-20 s speed and the larger hind knee's load over 10-20 s are shown beside the rows. Slow: under 0.225 m/s over 10-20 s. Slow and hot: slow, and the larger hind knee's RMS torque over 10-20 s at or over its continuous rating (1.0); the cut-offs are `fore128_slow_gait.py`'s, not a pre-registration's.
- Drive load: RMS torque / continuous rating at the joint (roll and pitch 7.5, front elbow 10.8, hind knee 6.75 N m: rl/trike_env.py cont_torque_*; desk ratings).
- Peak drive speed: the largest |joint speed| over the episodes that did not fall, from t = 2 s, rpm, per group roll / pitch / front elbow / hind knee, with the largest 99th percentile in brackets. The budget's allowances (desk): roll 83, pitch 83, front elbow 108, hind knee 83 rpm (X4 83 rpm: DECISION_LOG.md E1 (rated-speed cap above 24 V); X6 elbow 108 rpm: 0.70 x 176 rpm no-load at 48 V x 42/48 (budget/size_budget.py DRIVES, sim/CRAWL_REPORT.md); both as rl/analysis/gait_pipe1.py RPM_ALLOW holds them).
- Torque-limit time: the share of 1 ms physics steps with a drive's torque at or above 99 % of its limit (the model clamps each drive at 70 % of the maker's peak).

## PREREG_WALK60's conditions, reported only on this model, seeds 101-110 (commit aa150f89, dirty)

Randomisation and sensor noise on, 10 episodes per condition.

| Condition | Falls | Speed mean (min), m/s | Episodes under 90 % of the command | Worst drive RMS / rating, worst episode (joint) | Episodes with a drive at or over 1.0 | Worst joint's mean RMS / rating | Heading change, worst episode | Sideways drift, worst episode | Slip ratio, worst foot's mean (at the sole origin / at the contacts) | Peak drive speed, rpm | Steps at the episode's own torque limit, worst joint-episode | Hind touchdowns per s, worst episode | PREREG_WALK60's thresholds (written for the 188 mm roll-ankle model) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| J1 grippy (friction 1.0), the model's contact | 0/10 | 0.239 (0.201) | 2 | 1.04 (left_hind_flexion) | 1 | 0.72 (left_hind_pitch) | 7.3 deg | 0.27 m | 0.12 / 0.06 | 17 (11) / 57 (48) / 52 (44) / 55 (41) | 7.6 % (left_hind_flexion) | 5.1 | W1 met (0 falls); W2 met (0.239, min 0.201); W3 **not met** (1.04); W4 met (0.12) |
| J2 slippery (friction 0.4), the model's contact | 0/10 | 0.256 (0.253) | 0 | 0.83 (left_hind_flexion) | 0 | 0.75 (right_hind_flexion) | 6.0 deg | 0.30 m | 0.11 / 0.09 | 15 (8) / 53 (47) / 48 (43) / 51 (40) | 1.4 % (left_hind_flexion) | 3.1 | W1 met (0 falls); W2 met (0.256, min 0.253); W3 met (0.83); W4 met (0.11) |
| J3 grippy (friction 1.0), stiffer contact (impratio 10) | 0/10 | 0.221 (0.193) | 6 | 1.10 (left_hind_flexion) | 3 | 0.87 (left_hind_flexion) | 6.0 deg | 0.30 m | 0.13 / 0.07 | 20 (11) / 56 (47) / 52 (43) / 55 (39) | 9.4 % (left_hind_flexion) | 5.4 | W1 met (0 falls); W2 **not met** (0.221, min 0.193); W3 **not met** (1.10); W4 met (0.13) |
| J4 slippery (friction 0.4), stiffer contact (impratio 10) | 0/10 | 0.256 (0.252) | 0 | 0.83 (left_hind_flexion) | 0 | 0.75 (right_hind_flexion) | 5.8 deg | 0.29 m | 0.13 / 0.10 | 14 (8) / 54 (46) / 48 (43) / 50 (40) | 1.5 % (left_hind_flexion) | 3.7 | W1 met (0 falls); W2 met (0.256, min 0.252); W3 met (0.83); W4 met (0.13) |
| R1 J1 at 0 ms delay | 0/10 | 0.251 (0.248) | 0 | 0.77 (left_hind_flexion) | 0 | 0.67 (left_hind_flexion) | 5.5 deg | 0.27 m | 0.09 / not run | 9 (5) / 59 (49) / 50 (44) / 55 (41) | 1.9 % (right_fore_pitch) | 2.9 | orientation only: W1 met (0 falls); W2 met (0.251, min 0.248); W3 met (0.77); W4 met (0.09) |
| R2 J2 at 0 ms delay | 0/10 | 0.257 (0.255) | 0 | 0.81 (left_hind_flexion) | 0 | 0.72 (right_hind_flexion) | 5.8 deg | 0.26 m | 0.10 / not run | 12 (7) / 56 (48) / 49 (44) / 51 (41) | 1.4 % (left_hind_flexion) | 2.6 | orientation only: W1 met (0 falls); W2 met (0.257, min 0.255); W3 met (0.81); W4 met (0.10) |
| R3 J1 at the 0.15 m/s command | 0/10 | 0.139 (0.137) | 0 | 0.70 (right_fore_pitch) | 0 | 0.68 (right_fore_pitch) | 5.8 deg | 0.13 m | 0.18 / not run | 12 (6) / 54 (41) / 48 (42) / 47 (37) | 3.2 % (left_fore_pitch) | 1.6 | none (not at the 0.25 m/s command) |
| R4 J1 standing (0 m/s) | 0/10 | -0.000 (-0.000) |  | 0.53 (right_hind_flexion) | 0 | 0.49 (right_hind_flexion) | 0.0 deg | 0.00 m |  / not run | 1 (0) / 2 (1) / 4 (2) / 3 (2) | 0.0 % (right_hind_roll) | 0.0 | none (not at the 0.25 m/s command) |
| R5 nominal model, no randomisation or noise | 0/10 | 0.250 (0.250) | 0 | 0.70 (left_hind_pitch) | 0 | 0.69 (right_hind_pitch) | 1.2 deg | 0.01 m | 0.11 / not run | 3 (2) / 56 (47) / 51 (43) / 52 (33) | 0.7 % (right_fore_pitch) | 1.5 | orientation only: W1 met (0 falls); W2 met (0.250, min 0.250); W3 met (0.70); W4 met (0.11) |
| J1d16 J1 at 16 ms delay | 0/10 | 0.225 (0.194) | 6 | 1.09 (right_hind_flexion) | 2 | 0.84 (right_hind_flexion) | 7.6 deg | 0.33 m | 0.13 / not run | 15 (12) / 57 (48) / 51 (43) / 53 (40) | 9.1 % (left_hind_flexion) | 5.4 | orientation only: W1 met (0 falls); W2 **not met** (0.225, min 0.194); W3 **not met** (1.09); W4 met (0.13) |
| J1d17 J1 at 17 ms delay | 0/10 | 0.231 (0.195) | 4 | 1.11 (right_hind_flexion) | 2 | 0.79 (left_hind_flexion) | 7.1 deg | 0.29 m | 0.13 / not run | 15 (12) / 57 (48) / 51 (43) / 52 (41) | 10.2 % (left_hind_flexion) | 5.8 | orientation only: W1 met (0 falls); W2 **not met** (0.231, min 0.195); W3 **not met** (1.11); W4 met (0.13) |
| J1d18 J1 at 18 ms delay | 0/10 | 0.246 (0.242) | 0 | 0.79 (left_hind_pitch) | 0 | 0.75 (left_hind_pitch) | 6.2 deg | 0.24 m | 0.12 / not run | 13 (5) / 57 (48) / 50 (43) / 53 (41) | 2.9 % (right_fore_pitch) | 2.1 | orientation only: W1 met (0 falls); W2 met (0.246, min 0.242); W3 met (0.79); W4 met (0.12) |
| J1d19 J1 at 19 ms delay | 0/10 | 0.233 (0.217) | 4 | 0.87 (left_fore_pitch) | 0 | 0.80 (right_hind_pitch) | 5.4 deg | 0.23 m | 0.15 / not run | 12 (8) / 61 (55) / 54 (46) / 55 (42) | 5.1 % (left_fore_pitch) | 2.8 | orientation only: W1 met (0 falls); W2 met (0.233, min 0.217); W3 met (0.87); W4 met (0.15) |
| J1d20 J1 at 20 ms delay | 0/10 | 0.244 (0.237) | 0 | 0.80 (right_hind_pitch) | 0 | 0.75 (left_hind_pitch) | 4.6 deg | 0.20 m | 0.13 / not run | 7 (4) / 56 (48) / 52 (44) / 54 (41) | 3.3 % (right_fore_pitch) | 1.6 | orientation only: W1 met (0 falls); W2 met (0.244, min 0.237); W3 met (0.80); W4 met (0.13) |
| J1d30 J1 at 30 ms delay | 0/10 | 0.227 (0.217) | 4 | 0.90 (right_fore_pitch) | 0 | 0.84 (right_fore_pitch) | 5.4 deg | 0.21 m | 0.16 / not run | 10 (7) / 79 (55) / 56 (46) / 56 (42) | 5.4 % (right_fore_pitch) | 2.6 | orientation only: W1 met (0 falls); W2 met (0.227, min 0.217); W3 met (0.90); W4 met (0.16) |
| J1d40 J1 at 40 ms delay | 0/10 | 0.224 (0.203) | 5 | 0.99 (left_hind_flexion) | 0 | 0.86 (right_fore_pitch) | 6.5 deg | 0.20 m | 0.16 / not run | 16 (11) / 96 (53) / 55 (45) / 69 (42) | 8.3 % (right_hind_flexion) | 4.3 | orientation only: W1 met (0 falls); W2 **not met** (0.224, min 0.203); W3 met (0.99); W4 met (0.16) |
| J1d60 J1 at 60 ms delay | 0/10 | 0.232 (0.202) | 2 | 0.97 (left_hind_flexion) | 0 | 0.84 (right_fore_pitch) | 12.3 deg | 0.25 m | 0.16 / not run | 25 (11) / 166 (48) / 112 (45) / 69 (40) | 7.6 % (left_hind_flexion) | 4.8 | orientation only: W1 met (0 falls); W2 met (0.232, min 0.202); W3 met (0.97); W4 met (0.16) |

### 10-20 s windows of the same episodes, seeds 101-110 (`capsule_fore128_judged_101-110.json`)

| Condition | Slow over 10-20 s / upright | Slow and hot (hind knee at or over 1.0) | 10-20 s speed, range, m/s | Larger hind knee RMS / rating over 10-20 s, range | Slow episodes: 10-20 s speed; hind knee; mean pitch (range), deg (+ nose down) | Whole-episode and 2-20 s speeds match the stored episodes |
|---|---|---|---|---|---|---|
| J1 | 2/10 | 2 | 0.185-0.250 | 0.61-1.12 | 0.185-0.190; 1.11-1.12; -1.38 (-1.41 to -1.35) | yes |
| J2 | 0/10 | 0 | 0.254-0.259 | 0.68-0.82 |  | yes |
| J3 | 8/10 | 7 | 0.183-0.251 | 0.54-1.15 | 0.183-0.223; 0.88-1.15; -1.27 (-1.39 to -0.75) | yes |
| J4 | 0/10 | 0 | 0.253-0.259 | 0.73-0.83 |  | yes |
| R1 | 0/10 | 0 | 0.248-0.253 | 0.61-0.77 |  | yes |
| R2 | 0/10 | 0 | 0.255-0.258 | 0.67-0.82 |  | yes |
| R5 | 0/10 | 0 | 0.250-0.251 | 0.61-0.62 |  | yes |
| J1d16 | 6/10 | 6 | 0.178-0.251 | 0.61-1.16 | 0.178-0.203; 1.05-1.16; -1.37 (-1.43 to -1.32) | yes |
| J1d17 | 4/10 | 4 | 0.176-0.253 | 0.61-1.16 | 0.176-0.200; 1.09-1.16; -1.38 (-1.44 to -1.32) | yes |
| J1d18 | 0/10 | 0 | 0.242-0.248 | 0.59-0.73 |  | yes |
| J1d19 | 4/10 | 0 | 0.215-0.245 | 0.60-0.75 | 0.215-0.222; 0.67-0.75; 0.77 (0.71 to 0.84) | yes |
| J1d20 | 0/10 | 0 | 0.235-0.246 | 0.61-0.68 |  | yes |
| J1d30 | 5/10 | 0 | 0.212-0.243 | 0.63-0.77 | 0.212-0.223; 0.64-0.77; 0.74 (0.66 to 0.80) | yes |
| J1d40 | 6/10 | 1 | 0.180-0.235 | 0.62-1.16 | 0.180-0.225; 0.67-1.16; 0.41 (-1.36 to 0.82) | yes |
| J1d60 | 1/10 | 0 | 0.221-0.243 | 0.61-0.92 | 0.221-0.221; 0.92-0.92; -0.32 (-0.32 to -0.32) | yes |

## PREREG_WALK60's conditions, reported only on this model, seeds 201-210 (commit aa150f89, dirty)

Randomisation and sensor noise on, 10 episodes per condition.

| Condition | Falls | Speed mean (min), m/s | Episodes under 90 % of the command | Worst drive RMS / rating, worst episode (joint) | Episodes with a drive at or over 1.0 | Worst joint's mean RMS / rating | Heading change, worst episode | Sideways drift, worst episode | Slip ratio, worst foot's mean (at the sole origin / at the contacts) | Peak drive speed, rpm | Steps at the episode's own torque limit, worst joint-episode | Hind touchdowns per s, worst episode | PREREG_WALK60's thresholds (written for the 188 mm roll-ankle model) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| J1 grippy (friction 1.0), the model's contact | 0/10 | 0.244 (0.222) | 1 | 0.95 (right_hind_flexion) | 0 | 0.71 (right_hind_pitch) | 5.9 deg | 0.17 m | 0.11 / 0.05 | 18 (11) / 57 (48) / 51 (44) / 55 (41) | 5.5 % (right_hind_flexion) | 3.9 | W1 met (0 falls); W2 met (0.244, min 0.222); W3 met (0.95); W4 met (0.11) |
| J2 slippery (friction 0.4), the model's contact | 0/10 | 0.257 (0.254) | 0 | 0.85 (right_hind_flexion) | 0 | 0.77 (right_hind_flexion) | 5.5 deg | 0.16 m | 0.11 / 0.09 | 17 (7) / 55 (47) / 49 (43) / 50 (41) | 1.6 % (left_hind_flexion) | 2.8 | W1 met (0 falls); W2 met (0.257, min 0.254); W3 met (0.85); W4 met (0.11) |
| J3 grippy (friction 1.0), stiffer contact (impratio 10) | 0/10 | 0.217 (0.190) | 6 | 1.13 (right_hind_flexion) | 5 | 0.91 (left_hind_flexion) | 17.9 deg | 0.31 m | 0.15 / 0.08 | 27 (16) / 62 (48) / 56 (45) / 69 (41) | 11.0 % (left_hind_flexion) | 5.6 | W1 met (0 falls); W2 **not met** (0.217, min 0.190); W3 **not met** (1.13); W4 met (0.15) |
| J4 slippery (friction 0.4), stiffer contact (impratio 10) | 0/10 | 0.257 (0.253) | 0 | 0.84 (right_hind_flexion) | 0 | 0.77 (right_hind_flexion) | 6.2 deg | 0.17 m | 0.12 / 0.09 | 14 (8) / 55 (47) / 48 (43) / 50 (41) | 1.6 % (left_hind_flexion) | 3.2 | W1 met (0 falls); W2 met (0.257, min 0.253); W3 met (0.84); W4 met (0.12) |
| R1 J1 at 0 ms delay | 0/10 | 0.250 (0.235) | 0 | 0.86 (left_hind_flexion) | 0 | 0.69 (left_hind_flexion) | 6.0 deg | 0.17 m | 0.09 / not run | 17 (8) / 58 (48) / 50 (44) / 56 (41) | 3.1 % (right_hind_flexion) | 3.9 | orientation only: W1 met (0 falls); W2 met (0.250, min 0.235); W3 met (0.86); W4 met (0.09) |
| R2 J2 at 0 ms delay | 0/10 | 0.257 (0.256) | 0 | 0.84 (right_hind_flexion) | 0 | 0.75 (right_hind_flexion) | 5.6 deg | 0.16 m | 0.10 / not run | 12 (7) / 55 (48) / 49 (44) / 50 (41) | 1.5 % (left_hind_flexion) | 2.7 | orientation only: W1 met (0 falls); W2 met (0.257, min 0.256); W3 met (0.84); W4 met (0.10) |
| R3 J1 at the 0.15 m/s command | 0/10 | 0.140 (0.138) | 0 | 0.74 (left_hind_flexion) | 0 | 0.70 (right_fore_pitch) | 4.8 deg | 0.10 m | 0.18 / not run | 12 (6) / 51 (42) / 48 (42) / 48 (38) | 3.4 % (right_fore_pitch) | 1.6 | none (not at the 0.25 m/s command) |
| R4 J1 standing (0 m/s) | 0/10 | 0.000 (-0.000) |  | 0.53 (left_hind_flexion) | 0 | 0.50 (left_hind_flexion) | 0.0 deg | 0.00 m |  / not run | 1 (0) / 2 (1) / 4 (2) / 3 (2) | 0.0 % (right_hind_roll) | 0.0 | none (not at the 0.25 m/s command) |
| R5 nominal model, no randomisation or noise | 0/10 | 0.250 (0.250) | 0 | 0.69 (left_hind_pitch) | 0 | 0.69 (right_hind_pitch) | 1.1 deg | 0.01 m | 0.11 / not run | 3 (2) / 56 (46) / 51 (43) / 52 (33) | 0.7 % (right_fore_pitch) | 1.4 | orientation only: W1 met (0 falls); W2 met (0.250, min 0.250); W3 met (0.69); W4 met (0.11) |
| J1d16 J1 at 16 ms delay | 0/10 | 0.226 (0.191) | 4 | 1.10 (right_hind_flexion) | 4 | 0.83 (left_hind_flexion) | 8.9 deg | 0.18 m | 0.13 / not run | 19 (12) / 55 (47) / 52 (43) / 55 (41) | 10.3 % (left_hind_flexion) | 5.7 | orientation only: W1 met (0 falls); W2 **not met** (0.226, min 0.191); W3 **not met** (1.10); W4 met (0.13) |
| J1d17 J1 at 17 ms delay | 0/10 | 0.233 (0.185) | 3 | 1.15 (right_hind_flexion) | 2 | 0.79 (left_hind_flexion) | 10.7 deg | 0.14 m | 0.12 / not run | 18 (12) / 56 (48) / 53 (43) / 55 (41) | 11.7 % (right_hind_flexion) | 5.3 | orientation only: W1 met (0 falls); W2 **not met** (0.233, min 0.185); W3 **not met** (1.15); W4 met (0.12) |
| J1d18 J1 at 18 ms delay | 0/10 | 0.233 (0.182) | 2 | 1.16 (right_hind_flexion) | 2 | 0.77 (right_hind_pitch) | 5.0 deg | 0.17 m | 0.13 / not run | 19 (12) / 56 (48) / 52 (44) / 55 (41) | 12.2 % (right_hind_flexion) | 5.4 | orientation only: W1 met (0 falls); W2 **not met** (0.233, min 0.182); W3 **not met** (1.16); W4 met (0.13) |
| J1d19 J1 at 19 ms delay | 0/10 | 0.234 (0.210) | 3 | 0.89 (left_fore_pitch) | 0 | 0.80 (left_hind_pitch) | 6.3 deg | 0.19 m | 0.15 / not run | 12 (9) / 73 (56) / 57 (47) / 56 (42) | 5.2 % (left_fore_pitch) | 2.8 | orientation only: W1 met (0 falls); W2 met (0.234, min 0.210); W3 met (0.89); W4 met (0.15) |
| J1d20 J1 at 20 ms delay | 0/10 | 0.242 (0.234) | 0 | 0.80 (right_hind_pitch) | 0 | 0.76 (right_hind_pitch) | 5.1 deg | 0.17 m | 0.13 / not run | 11 (6) / 64 (51) / 55 (44) / 54 (41) | 3.3 % (right_fore_pitch) | 2.1 | orientation only: W1 met (0 falls); W2 met (0.242, min 0.234); W3 met (0.80); W4 met (0.13) |
| J1d30 J1 at 30 ms delay | 0/10 | 0.226 (0.186) | 2 | 1.15 (left_hind_flexion) | 1 | 0.84 (right_fore_pitch) | 30.3 deg | 0.36 m | 0.16 / not run | 35 (20) / 78 (55) / 61 (49) / 66 (41) | 10.3 % (left_hind_flexion) | 6.2 | orientation only: W1 met (0 falls); W2 **not met** (0.226, min 0.186); W3 **not met** (1.15); W4 met (0.16) |
| J1d40 J1 at 40 ms delay | 0/10 | 0.222 (0.195) | 3 | 1.22 (left_hind_flexion) | 3 | 0.87 (right_hind_pitch) | 13.3 deg | 0.26 m | 0.16 / not run | 18 (14) / 92 (52) / 60 (45) / 65 (42) | 11.8 % (left_hind_flexion) | 6.3 | orientation only: W1 met (0 falls); W2 **not met** (0.222, min 0.195); W3 **not met** (1.22); W4 met (0.16) |
| J1d60 J1 at 60 ms delay | 0/10 | 0.230 (0.197) | 2 | 1.08 (right_hind_flexion) | 1 | 0.85 (right_fore_pitch) | 12.3 deg | 0.16 m | 0.16 / not run | 25 (10) / 120 (49) / 87 (45) / 77 (42) | 10.1 % (left_hind_flexion) | 5.8 | orientation only: W1 met (0 falls); W2 **not met** (0.230, min 0.197); W3 **not met** (1.08); W4 met (0.16) |

### 10-20 s windows of the same episodes, seeds 201-210 (`capsule_fore128_judged_201-210.json`)

| Condition | Slow over 10-20 s / upright | Slow and hot (hind knee at or over 1.0) | 10-20 s speed, range, m/s | Larger hind knee RMS / rating over 10-20 s, range | Slow episodes: 10-20 s speed; hind knee; mean pitch (range), deg (+ nose down) | Whole-episode and 2-20 s speeds match the stored episodes |
|---|---|---|---|---|---|---|
| J1 | 2/10 | 1 | 0.200-0.251 | 0.61-1.09 | 0.200-0.214; 0.98-1.09; -1.12 (-1.23 to -1.00) | yes |
| J2 | 0/10 | 0 | 0.254-0.259 | 0.75-0.85 |  | yes |
| J3 | 7/10 | 6 | 0.183-0.252 | 0.54-1.17 | 0.183-0.225; 0.90-1.17; -1.26 (-1.39 to -0.67) | yes |
| J4 | 0/10 | 0 | 0.253-0.259 | 0.70-0.85 |  | yes |
| R1 | 1/10 | 0 | 0.219-0.253 | 0.64-0.96 | 0.219-0.219; 0.96-0.96; -0.99 (-0.99 to -0.99) | yes |
| R2 | 0/10 | 0 | 0.256-0.258 | 0.72-0.85 |  | yes |
| R5 | 0/10 | 0 | 0.250-0.250 | 0.61-0.62 |  | yes |
| J1d16 | 5/10 | 4 | 0.178-0.250 | 0.65-1.17 | 0.178-0.220; 0.95-1.17; -1.30 (-1.44 to -0.92) | yes |
| J1d17 | 3/10 | 3 | 0.176-0.251 | 0.64-1.19 | 0.176-0.188; 1.10-1.19; -1.38 (-1.42 to -1.34) | yes |
| J1d18 | 3/10 | 3 | 0.168-0.252 | 0.65-1.21 | 0.168-0.207; 1.01-1.21; -1.32 (-1.46 to -1.11) | yes |
| J1d19 | 3/10 | 0 | 0.209-0.245 | 0.63-0.75 | 0.209-0.224; 0.66-0.75; 0.78 (0.69 to 0.89) | yes |
| J1d20 | 0/10 | 0 | 0.238-0.245 | 0.62-0.69 |  | yes |
| J1d30 | 2/10 | 1 | 0.184-0.242 | 0.63-1.16 | 0.184-0.219; 0.75-1.16; -0.29 (-1.29 to 0.72) | yes |
| J1d40 | 4/10 | 3 | 0.197-0.235 | 0.64-1.24 | 0.197-0.224; 0.64-1.24; -0.51 (-1.38 to 0.73) | yes |
| J1d60 | 2/10 | 1 | 0.189-0.241 | 0.66-1.09 | 0.189-0.221; 0.84-1.09; -0.56 (-1.09 to -0.04) | yes |

## Sweep `walk60_fore128_fore_aft_200000`: fore-aft pushes, seeds 200000-200009 (commit aa150f89, dirty; 190 episodes, 5.4 min)

Nominal model, observation noise on, no randomisation, 10 ms delay and 0.25 m/s unless the row says otherwise; 10 episodes per point. A push lasts 0.1 s from 8 s; its free-body dv is its impulse over the robot's whole mass.

### fore-aft push (along the forward axis at onset; + forward, - back) (the sweep's break rule: low: none; high: none)

| Push, free-body dv, m/s (+ forward, - back) | Falls | Speed mean 2-20 s (min), m/s | Episodes under 90 % of the command | Sideways drift, worst, m | Heading change, worst, deg | Worst drive RMS / rating: worst joint's mean (worst episode; joint) | Episodes with a drive at or over 1.0 | Peak drive speed, rpm | Steps at the nominal torque limit, worst joint's mean | Extra travel along the push, mean (max), m | 10-20 s: slow / upright | 10-20 s: slow and hot | 10-20 s speed, range, m/s | 10-20 s larger hind knee RMS / rating, range |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| -1.5 | 0/10 | 0.223 (0.207) | 6 | 0.14 | 7.9 | 0.82 (0.94; right_hind_flexion) | 0 | 21 (10) / 64 (46) / 67 (43) / 67 (33) | 2.3 % | 0.203 (0.272) | 6/10 | 6 | 0.192-0.252 | 0.62-1.05 |
| -1.25 | 0/10 | 0.212 (0.208) | 10 | 0.17 | 7.9 | 0.92 (0.93; right_hind_flexion) | 0 | 23 (10) / 57 (44) / 64 (43) / 72 (33) | 3.6 % | 0.202 (0.234) | 10/10 | 10 | 0.191-0.200 | 1.03-1.05 |
| -1 | 0/10 | 0.214 (0.209) | 10 | 0.23 | 4.9 | 0.92 (0.95; right_hind_flexion) | 0 | 22 (11) / 57 (44) / 58 (44) / 58 (33) | 3.9 % | 0.180 (0.220) | 10/10 | 10 | 0.191-0.200 | 1.02-1.05 |
| -0.75 | 0/10 | 0.224 (0.211) | 7 | 0.18 | 8.1 | 0.86 (0.93; right_hind_flexion) | 0 | 24 (11) / 57 (46) / 56 (43) / 53 (33) | 2.9 % | 0.120 (0.185) | 7/10 | 7 | 0.191-0.252 | 0.63-1.05 |
| -0.5 | 0/10 | 0.213 (0.211) | 10 | 0.11 | 5.6 | 0.93 (0.94; left_hind_flexion) | 0 | 22 (12) / 57 (44) / 51 (43) / 66 (33) | 4.1 % | 0.181 (0.200) | 10/10 | 10 | 0.192-0.199 | 1.03-1.05 |
| -0.4 | 0/10 | 0.214 (0.211) | 10 | 0.20 | 8.1 | 0.92 (0.93; left_hind_flexion) | 0 | 20 (11) / 57 (44) / 51 (43) / 53 (33) | 3.9 % | 0.153 (0.181) | 10/10 | 10 | 0.192-0.199 | 1.03-1.05 |
| -0.3 | 0/10 | 0.215 (0.213) | 10 | 0.18 | 10.4 | 0.91 (0.93; left_hind_flexion) | 0 | 20 (10) / 57 (44) / 51 (43) / 53 (32) | 3.8 % | 0.138 (0.162) | 10/10 | 10 | 0.190-0.199 | 1.03-1.05 |
| -0.2 | 0/10 | 0.235 (0.218) | 5 | 0.23 | 10.0 | 0.78 (0.91; left_hind_flexion) | 0 | 20 (10) / 57 (46) / 52 (43) / 53 (33) | 1.8 % | 0.033 (0.083) | 5/10 | 5 | 0.194-0.253 | 0.62-1.04 |
| -0.1 | 0/10 | 0.250 (0.243) | 0 | 0.25 | 8.3 | 0.67 (0.73; left_hind_pitch) | 0 | 11 (7) / 57 (46) / 51 (43) / 53 (33) | 0.6 % | 0.004 (0.007) | 0/10 | 0 | 0.236-0.253 | 0.62-0.80 |
| 0 | 0/10 | 0.251 (0.250) | 0 | 0.26 | 8.4 | 0.68 (0.70; left_hind_pitch) | 0 | 6 (3) / 57 (46) / 51 (43) / 53 (33) | 0.7 % |  | 0/10 | 0 | 0.250-0.253 | 0.62-0.70 |
| 0.1 | 0/10 | 0.251 (0.250) | 0 | 0.25 | 8.2 | 0.67 (0.69; left_hind_pitch) | 0 | 7 (3) / 57 (46) / 51 (43) / 53 (33) | 0.7 % | 0.003 (0.005) | 0/10 | 0 | 0.250-0.253 | 0.62-0.67 |
| 0.2 | 0/10 | 0.251 (0.250) | 0 | 0.24 | 8.3 | 0.67 (0.69; left_hind_pitch) | 0 | 6 (3) / 57 (46) / 51 (43) / 53 (33) | 0.7 % | 0.005 (0.007) | 0/10 | 0 | 0.250-0.254 | 0.62-0.68 |
| 0.3 | 0/10 | 0.251 (0.250) | 0 | 0.25 | 8.4 | 0.66 (0.70; left_hind_pitch) | 0 | 7 (3) / 57 (46) / 51 (43) / 53 (33) | 0.7 % | 0.007 (0.007) | 0/10 | 0 | 0.250-0.256 | 0.61-0.75 |
| 0.4 | 0/10 | 0.250 (0.249) | 0 | 0.24 | 8.4 | 0.68 (0.69; left_hind_pitch) | 0 | 9 (4) / 57 (47) / 53 (43) / 53 (33) | 0.8 % | 0.010 (0.010) | 0/10 | 0 | 0.249-0.253 | 0.62-0.67 |
| 0.5 | 0/10 | 0.247 (0.230) | 0 | 0.24 | 8.5 | 0.69 (0.82; left_hind_pitch) | 0 | 19 (8) / 58 (47) / 54 (43) / 53 (33) | 0.8 % | 0.014 (0.014) | 1/10 | 0 | 0.216-0.252 | 0.62-0.95 |
| 0.75 | 0/10 | 0.250 (0.249) | 0 | 0.26 | 8.3 | 0.67 (0.73; right_hind_pitch) | 0 | 14 (4) / 63 (47) / 59 (43) / 53 (33) | 0.8 % | 0.028 (0.029) | 0/10 | 0 | 0.250-0.255 | 0.62-0.76 |
| 1 | 0/10 | 0.245 (0.220) | 2 | 0.24 | 8.3 | 0.71 (0.91; left_hind_flexion) | 0 | 20 (10) / 65 (47) / 65 (43) / 53 (33) | 0.9 % | 0.048 (0.050) | 2/10 | 2 | 0.198-0.254 | 0.62-1.04 |
| 1.25 | 0/10 | 0.244 (0.224) | 1 | 0.24 | 9.6 | 0.76 (0.91; left_hind_flexion) | 0 | 24 (9) / 89 (47) / 93 (43) / 53 (33) | 1.4 % | 0.118 (0.125) | 4/10 | 4 | 0.195-0.252 | 0.62-1.04 |
| 1.5 | 0/10 | 0.256 (0.228) | 0 | 0.26 | 8.3 | 0.69 (0.91; left_hind_pitch) | 0 | 22 (10) / 109 (47) / 112 (43) / 53 (33) | 1.1 % | 0.180 (0.196) | 1/10 | 1 | 0.195-0.253 | 0.61-1.05 |

## Sweep `walk60_fore128_fore_aft_300000`: fore-aft pushes, seeds 300000-300009 (commit aa150f89, dirty; 190 episodes, 3.0 min)

Nominal model, observation noise on, no randomisation, 10 ms delay and 0.25 m/s unless the row says otherwise; 10 episodes per point. A push lasts 0.1 s from 8 s; its free-body dv is its impulse over the robot's whole mass.

### fore-aft push (along the forward axis at onset; + forward, - back) (the sweep's break rule: low: none; high: none)

| Push, free-body dv, m/s (+ forward, - back) | Falls | Speed mean 2-20 s (min), m/s | Episodes under 90 % of the command | Sideways drift, worst, m | Heading change, worst, deg | Worst drive RMS / rating: worst joint's mean (worst episode; joint) | Episodes with a drive at or over 1.0 | Peak drive speed, rpm | Steps at the nominal torque limit, worst joint's mean | Extra travel along the push, mean (max), m | 10-20 s: slow / upright | 10-20 s: slow and hot | 10-20 s speed, range, m/s | 10-20 s larger hind knee RMS / rating, range |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| -1.5 | 0/10 | 0.223 (0.208) | 6 | 0.24 | 6.2 | 0.82 (0.94; right_hind_flexion) | 0 | 20 (10) / 66 (46) / 72 (43) / 71 (34) | 2.3 % | 0.216 (0.260) | 6/10 | 6 | 0.193-0.253 | 0.62-1.05 |
| -1.25 | 0/10 | 0.215 (0.209) | 9 | 0.23 | 6.0 | 0.90 (0.94; right_hind_flexion) | 0 | 21 (11) / 57 (46) / 64 (43) / 73 (33) | 3.4 % | 0.199 (0.245) | 9/10 | 9 | 0.193-0.253 | 0.71-1.05 |
| -1 | 0/10 | 0.213 (0.211) | 10 | 0.20 | 7.5 | 0.93 (0.94; right_hind_flexion) | 0 | 22 (12) / 57 (44) / 60 (43) / 57 (33) | 3.9 % | 0.184 (0.211) | 10/10 | 10 | 0.191-0.198 | 1.03-1.05 |
| -0.75 | 0/10 | 0.220 (0.212) | 8 | 0.11 | 5.8 | 0.89 (0.94; right_hind_flexion) | 0 | 23 (12) / 57 (46) / 56 (43) / 60 (33) | 3.3 % | 0.131 (0.189) | 9/10 | 8 | 0.193-0.251 | 0.65-1.06 |
| -0.5 | 0/10 | 0.213 (0.211) | 10 | 0.24 | 5.7 | 0.93 (0.95; left_hind_flexion) | 0 | 25 (11) / 57 (44) / 54 (43) / 63 (33) | 4.2 % | 0.178 (0.192) | 10/10 | 10 | 0.192-0.199 | 1.03-1.05 |
| -0.4 | 0/10 | 0.215 (0.210) | 10 | 0.33 | 8.6 | 0.92 (0.95; left_hind_flexion) | 0 | 23 (12) / 57 (44) / 51 (43) / 53 (33) | 3.9 % | 0.154 (0.177) | 10/10 | 10 | 0.191-0.200 | 1.03-1.07 |
| -0.3 | 0/10 | 0.216 (0.214) | 10 | 0.17 | 5.4 | 0.92 (0.94; right_hind_flexion) | 0 | 20 (10) / 57 (44) / 51 (43) / 53 (33) | 3.7 % | 0.130 (0.167) | 10/10 | 10 | 0.193-0.199 | 1.03-1.05 |
| -0.2 | 0/10 | 0.241 (0.216) | 3 | 0.17 | 4.7 | 0.73 (0.93; right_hind_flexion) | 0 | 20 (10) / 57 (46) / 52 (43) / 53 (33) | 1.1 % | 0.042 (0.141) | 3/10 | 3 | 0.192-0.253 | 0.62-1.06 |
| -0.1 | 0/10 | 0.245 (0.217) | 2 | 0.25 | 5.8 | 0.70 (0.91; left_hind_flexion) | 0 | 20 (10) / 57 (46) / 51 (43) / 53 (33) | 0.8 % | 0.022 (0.114) | 2/10 | 2 | 0.197-0.253 | 0.62-1.04 |
| 0 | 0/10 | 0.251 (0.250) | 0 | 0.25 | 3.8 | 0.67 (0.69; left_hind_pitch) | 0 | 7 (4) / 57 (46) / 51 (43) / 53 (33) | 0.7 % |  | 0/10 | 0 | 0.250-0.254 | 0.62-0.70 |
| 0.1 | 0/10 | 0.249 (0.238) | 0 | 0.25 | 4.1 | 0.67 (0.79; left_hind_pitch) | 0 | 12 (8) / 57 (46) / 51 (43) / 53 (33) | 0.7 % | 0.004 (0.014) | 0/10 | 0 | 0.226-0.252 | 0.62-0.88 |
| 0.2 | 0/10 | 0.248 (0.224) | 1 | 0.24 | 4.0 | 0.67 (0.87; left_hind_pitch) | 0 | 20 (9) / 57 (47) / 51 (43) / 53 (33) | 0.7 % | 0.005 (0.009) | 1/10 | 1 | 0.203-0.254 | 0.62-1.02 |
| 0.3 | 0/10 | 0.251 (0.250) | 0 | 0.25 | 3.9 | 0.67 (0.69; left_hind_pitch) | 0 | 8 (4) / 57 (46) / 51 (43) / 53 (33) | 0.7 % | 0.007 (0.008) | 0/10 | 0 | 0.250-0.254 | 0.62-0.71 |
| 0.4 | 0/10 | 0.250 (0.248) | 0 | 0.24 | 4.1 | 0.67 (0.69; left_hind_pitch) | 0 | 9 (4) / 57 (47) / 53 (43) / 53 (33) | 0.8 % | 0.010 (0.010) | 0/10 | 0 | 0.250-0.252 | 0.62-0.66 |
| 0.5 | 0/10 | 0.250 (0.248) | 0 | 0.25 | 3.9 | 0.68 (0.70; left_hind_pitch) | 0 | 10 (4) / 58 (47) / 54 (43) / 53 (33) | 0.8 % | 0.014 (0.014) | 0/10 | 0 | 0.250-0.254 | 0.62-0.71 |
| 0.75 | 0/10 | 0.251 (0.249) | 0 | 0.25 | 4.0 | 0.67 (0.69; right_hind_pitch) | 0 | 15 (4) / 62 (47) / 59 (43) / 53 (33) | 0.8 % | 0.028 (0.030) | 0/10 | 0 | 0.249-0.253 | 0.62-0.71 |
| 1 | 0/10 | 0.249 (0.223) | 1 | 0.26 | 4.1 | 0.69 (0.89; left_hind_pitch) | 0 | 20 (9) / 68 (47) / 64 (43) / 62 (33) | 0.8 % | 0.048 (0.051) | 1/10 | 1 | 0.199-0.253 | 0.62-1.03 |
| 1.25 | 0/10 | 0.253 (0.225) | 0 | 0.24 | 4.1 | 0.68 (0.91; left_hind_pitch) | 0 | 21 (9) / 98 (47) / 94 (43) / 53 (33) | 0.9 % | 0.117 (0.130) | 1/10 | 1 | 0.199-0.254 | 0.62-1.03 |
| 1.5 | 0/10 | 0.250 (0.226) | 0 | 0.28 | 3.9 | 0.72 (0.93; left_hind_flexion) | 0 | 20 (10) / 112 (48) / 112 (43) / 56 (33) | 1.2 % | 0.178 (0.188) | 3/10 | 3 | 0.193-0.252 | 0.62-1.05 |

## Sweep `walk60_fore128_sideways_200000`: sideways pushes, seeds 200000-200009 (commit aa150f89, dirty; 720 episodes, 21.2 min)

Nominal model, observation noise on, no randomisation, 10 ms delay and 0.25 m/s unless the row says otherwise; 10 episodes per point. A push lasts 0.1 s from 8 s; its free-body dv is its impulse over the robot's whole mass.

### delay, 0.25 m/s (the sweep's break rule: low: none; high: none)

| Delay, ms | Falls | Speed mean 2-20 s (min), m/s | Episodes under 90 % of the command | Sideways drift, worst, m | Heading change, worst, deg | Worst drive RMS / rating: worst joint's mean (worst episode; joint) | Episodes with a drive at or over 1.0 | Peak drive speed, rpm | Steps at the nominal torque limit, worst joint's mean | 10-20 s: slow / upright | 10-20 s: slow and hot | 10-20 s speed, range, m/s | 10-20 s larger hind knee RMS / rating, range |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0/10 | 0.253 (0.251) | 0 | 0.26 | 8.0 | 0.65 (0.66; left_hind_pitch) | 0 | 8 (3) / 58 (47) / 49 (43) / 52 (33) | 0.6 % | | | | |
| 2 | 0/10 | 0.253 (0.253) | 0 | 0.25 | 8.1 | 0.65 (0.67; right_hind_flexion) | 0 | 9 (3) / 57 (46) / 49 (42) / 52 (33) | 0.5 % | | | | |
| 5 | 0/10 | 0.251 (0.229) | 0 | 0.20 | 8.3 | 0.74 (0.88; right_hind_flexion) | 0 | 12 (9) / 55 (45) / 50 (42) / 52 (33) | 0.4 % | | | | |
| 10 | 0/10 | 0.251 (0.250) | 0 | 0.26 | 8.4 | 0.68 (0.70; left_hind_pitch) | 0 | 6 (3) / 57 (46) / 51 (43) / 53 (33) | 0.7 % | | | | |
| 15 | 0/10 | 0.243 (0.207) | 1 | 0.26 | 7.8 | 0.75 (1.00; right_hind_flexion) | 0 | 21 (11) / 55 (45) / 51 (43) / 52 (33) | 1.1 % | 2/10 | 1 | 0.186-0.255 | 0.67-1.09 |
| 16 | 0/10 | 0.223 (0.187) | 5 | 0.19 | 8.2 | 0.88 (1.08; left_hind_flexion) | 4 | 23 (12) / 56 (45) / 53 (43) / 51 (34) | 3.3 % | 5/10 | 5 | 0.182-0.252 | 0.69-1.10 |
| 17 | 0/10 | 0.235 (0.185) | 3 | 0.20 | 8.3 | 0.80 (1.10; left_hind_flexion) | 2 | 23 (13) / 55 (45) / 53 (45) / 51 (35) | 2.0 % | 3/10 | 3 | 0.179-0.253 | 0.61-1.13 |
| 18 | 0/10 | 0.219 (0.185) | 5 | 0.20 | 12.8 | 0.87 (1.10; left_hind_flexion) | 5 | 24 (14) / 54 (45) / 54 (45) / 51 (34) | 3.6 % | 6/10 | 5 | 0.178-0.254 | 0.62-1.13 |
| 19 | 0/10 | 0.247 (0.245) | 0 | 0.24 | 8.7 | 0.75 (0.77; left_hind_pitch) | 0 | 7 (3) / 58 (47) / 50 (43) / 51 (33) | 1.4 % | 0/10 | 0 | 0.244-0.247 | 0.61-0.63 |
| 20 | 0/10 | 0.243 (0.211) | 1 | 0.24 | 8.7 | 0.73 (1.00; left_hind_pitch) | 1 | 22 (10) / 57 (47) / 51 (43) / 52 (33) | 1.2 % | 1/10 | 1 | 0.183-0.248 | 0.62-1.11 |
| 30 | 0/10 | 0.222 (0.189) | 4 | 0.24 | 14.7 | 0.82 (1.12; right_hind_flexion) | 4 | 26 (15) / 57 (47) / 56 (46) / 59 (37) | 3.2 % | 4/10 | 4 | 0.183-0.243 | 0.63-1.16 |
| 40 | 0/10 | 0.218 (0.188) | 4 | 0.28 | 20.5 | 0.85 (1.16; right_hind_flexion) | 4 | 14 (10) / 70 (48) / 57 (44) / 69 (39) | 4.0 % | 4/10 | 4 | 0.185-0.236 | 0.64-1.18 |
| 60 | 0/10 | 0.230 (0.190) | 3 | 0.28 | 8.2 | 0.80 (1.09; right_hind_pitch) | 1 | 29 (10) / 125 (48) / 65 (44) / 68 (39) | 2.1 % | 1/10 | 1 | 0.197-0.244 | 0.64-1.12 |

### delay, 0.15 m/s (the sweep's break rule: low: none; high: none)

| Delay, ms | Falls | Speed mean 2-20 s (min), m/s | Episodes under 90 % of the command | Sideways drift, worst, m | Heading change, worst, deg | Worst drive RMS / rating: worst joint's mean (worst episode; joint) | Episodes with a drive at or over 1.0 | Peak drive speed, rpm | Steps at the nominal torque limit, worst joint's mean |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0/10 | 0.145 (0.145) | 0 | 0.16 | 7.8 | 0.67 (0.68; right_fore_pitch) | 0 | 11 (6) / 52 (40) / 47 (41) / 35 (31) | 0.7 % |
| 2 | 0/10 | 0.146 (0.145) | 0 | 0.15 | 7.0 | 0.65 (0.67; right_fore_pitch) | 0 | 11 (6) / 51 (40) / 49 (41) / 34 (31) | 0.6 % |
| 5 | 0/10 | 0.147 (0.147) | 0 | 0.15 | 7.8 | 0.63 (0.64; left_hind_flexion) | 0 | 11 (5) / 54 (39) / 51 (40) / 35 (31) | 0.4 % |
| 10 | 0/10 | 0.142 (0.140) | 0 | 0.15 | 7.7 | 0.70 (0.70; right_fore_pitch) | 0 | 12 (6) / 64 (40) / 54 (40) / 48 (31) | 1.2 % |
| 15 | 0/10 | 0.142 (0.141) | 0 | 0.14 | 8.7 | 0.69 (0.70; right_fore_pitch) | 0 | 13 (6) / 67 (39) / 56 (40) / 34 (31) | 1.1 % |
| 16 | 0/10 | 0.142 (0.142) | 0 | 0.15 | 8.9 | 0.69 (0.69; right_fore_pitch) | 0 | 12 (5) / 69 (39) / 54 (40) / 34 (31) | 1.2 % |
| 17 | 0/10 | 0.142 (0.141) | 0 | 0.14 | 8.2 | 0.69 (0.70; right_fore_pitch) | 0 | 12 (6) / 69 (39) / 56 (40) / 34 (31) | 1.1 % |
| 18 | 0/10 | 0.141 (0.139) | 0 | 0.14 | 8.0 | 0.71 (0.72; right_fore_pitch) | 0 | 13 (6) / 72 (39) / 58 (40) / 34 (31) | 1.4 % |
| 19 | 0/10 | 0.136 (0.135) | 0 | 0.14 | 5.3 | 0.74 (0.75; right_fore_pitch) | 0 | 13 (6) / 80 (40) / 58 (41) / 34 (31) | 1.9 % |
| 20 | 0/10 | 0.137 (0.136) | 0 | 0.14 | 7.3 | 0.73 (0.74; right_fore_pitch) | 0 | 12 (6) / 79 (39) / 58 (40) / 35 (31) | 1.6 % |
| 30 | 0/10 | 0.136 (0.135) | 0 | 0.14 | 5.8 | 0.75 (0.76; left_fore_pitch) | 0 | 11 (5) / 71 (39) / 57 (40) / 34 (30) | 1.9 % |
| 40 | 0/10 | 0.135 (0.132) | 4 | 0.14 | 9.9 | 0.78 (0.79; right_fore_pitch) | 0 | 13 (6) / 80 (40) / 56 (40) / 35 (30) | 2.6 % |
| 60 | 0/10 | 0.130 (0.116) | 9 | 0.31 | 14.8 | 0.88 (0.97; right_fore_pitch) | 0 | 31 (12) / 125 (67) / 70 (50) / 66 (43) | 4.7 % |

### friction (the sweep's break rule: low: none; high: none)

| Floor friction | Falls | Speed mean 2-20 s (min), m/s | Episodes under 90 % of the command | Sideways drift, worst, m | Heading change, worst, deg | Worst drive RMS / rating: worst joint's mean (worst episode; joint) | Episodes with a drive at or over 1.0 | Peak drive speed, rpm | Steps at the nominal torque limit, worst joint's mean | 10-20 s: slow / upright | 10-20 s: slow and hot | 10-20 s speed, range, m/s | 10-20 s larger hind knee RMS / rating, range |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.15 | 0/10 | 0.258 (0.257) | 0 | 0.24 | 7.3 | 0.80 (0.80; right_hind_flexion) | 0 | 8 (4) / 55 (48) / 47 (43) / 50 (33) | 0.7 % | | | | |
| 0.2 | 0/10 | 0.257 (0.256) | 0 | 0.28 | 7.4 | 0.78 (0.78; right_hind_flexion) | 0 | 10 (5) / 55 (48) / 48 (43) / 50 (33) | 0.7 % | | | | |
| 0.3 | 0/10 | 0.256 (0.255) | 0 | 0.29 | 7.7 | 0.74 (0.76; right_hind_flexion) | 0 | 12 (5) / 53 (47) / 47 (42) / 50 (33) | 0.6 % | | | | |
| 0.4 | 0/10 | 0.257 (0.255) | 0 | 0.28 | 7.6 | 0.74 (0.76; right_hind_flexion) | 0 | 12 (6) / 53 (46) / 48 (42) / 49 (33) | 0.5 % | | | | |
| 0.5 | 0/10 | 0.257 (0.256) | 0 | 0.26 | 7.6 | 0.73 (0.75; right_hind_flexion) | 0 | 12 (7) / 53 (45) / 47 (42) / 49 (33) | 0.5 % | | | | |
| 0.6 | 0/10 | 0.256 (0.255) | 0 | 0.26 | 7.7 | 0.73 (0.75; right_hind_flexion) | 0 | 10 (5) / 54 (45) / 47 (42) / 50 (33) | 0.5 % | | | | |
| 0.8 | 0/10 | 0.251 (0.239) | 0 | 0.27 | 8.0 | 0.68 (0.77; right_hind_flexion) | 0 | 12 (8) / 57 (46) / 51 (43) / 52 (33) | 0.6 % | 0/10 | 0 | 0.229-0.255 | 0.63-0.87 |
| the model's own | 0/10 | 0.251 (0.250) | 0 | 0.26 | 8.4 | 0.68 (0.70; left_hind_pitch) | 0 | 6 (3) / 57 (46) / 51 (43) / 53 (33) | 0.7 % | | | | |
| 1 | 0/10 | 0.248 (0.247) | 0 | 0.24 | 7.8 | 0.71 (0.72; right_hind_pitch) | 0 | 4 (2) / 56 (47) / 51 (43) / 52 (33) | 0.8 % | | | | |
| 1.2 | 0/10 | 0.247 (0.245) | 0 | 0.28 | 7.3 | 0.72 (0.75; right_hind_pitch) | 0 | 6 (3) / 57 (47) / 50 (43) / 52 (33) | 0.9 % | | | | |

### mass, every body (the sweep's break rule: low: none; high: none)

| Mass, x every body | Falls | Speed mean 2-20 s (min), m/s | Episodes under 90 % of the command | Sideways drift, worst, m | Heading change, worst, deg | Worst drive RMS / rating: worst joint's mean (worst episode; joint) | Episodes with a drive at or over 1.0 | Peak drive speed, rpm | Steps at the nominal torque limit, worst joint's mean | 10-20 s: slow / upright | 10-20 s: slow and hot | 10-20 s speed, range, m/s | 10-20 s larger hind knee RMS / rating, range |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.8 | 0/10 | 0.249 (0.249) | 0 | 0.26 | 7.7 | 0.61 (0.62; right_fore_pitch) | 0 | 6 (3) / 59 (48) / 51 (43) / 57 (33) | 0.8 % | 0/10 | 0 | 0.248-0.251 | 0.56-0.58 |
| 0.9 | 0/10 | 0.250 (0.249) | 0 | 0.24 | 8.0 | 0.63 (0.65; left_hind_pitch) | 0 | 6 (3) / 58 (47) / 52 (43) / 54 (33) | 0.7 % | 0/10 | 0 | 0.249-0.255 | 0.59-0.69 |
| 1 | 0/10 | 0.251 (0.250) | 0 | 0.26 | 8.4 | 0.68 (0.70; left_hind_pitch) | 0 | 6 (3) / 57 (46) / 51 (43) / 53 (33) | 0.7 % | | | | |
| 1.1 | 0/10 | 0.252 (0.251) | 0 | 0.27 | 8.2 | 0.70 (0.75; right_hind_pitch) | 0 | 7 (4) / 57 (46) / 51 (42) / 51 (33) | 0.6 % | | | | |
| 1.2 | 0/10 | 0.250 (0.243) | 0 | 0.26 | 8.4 | 0.77 (0.94; right_hind_flexion) | 0 | 19 (8) / 56 (45) / 50 (42) / 50 (33) | 0.5 % | 0/10 | 0 | 0.234-0.254 | 0.68-1.00 |
| 1.4 | 0/10 | 0.247 (0.225) | 1 | 0.20 | 8.0 | 0.93 (1.13; right_hind_flexion) | 3 | 20 (10) / 56 (45) / 49 (42) / 48 (33) | 1.4 % | 2/10 | 2 | 0.211-0.254 | 0.75-1.28 |
| 1.6 | 0/10 | 0.238 (0.219) | 2 | 0.22 | 8.2 | 1.12 (1.32; right_hind_flexion) | 8 | 21 (11) / 54 (45) / 49 (42) / 52 (33) | 4.1 % | 4/10 | 4 | 0.204-0.252 | 0.88-1.41 |

### payload on the chassis (the sweep's break rule: high: none)

| Payload on the chassis, kg | Falls | Speed mean 2-20 s (min), m/s | Episodes under 90 % of the command | Sideways drift, worst, m | Heading change, worst, deg | Worst drive RMS / rating: worst joint's mean (worst episode; joint) | Episodes with a drive at or over 1.0 | Peak drive speed, rpm | Steps at the nominal torque limit, worst joint's mean | 10-20 s: slow / upright | 10-20 s: slow and hot | 10-20 s speed, range, m/s | 10-20 s larger hind knee RMS / rating, range |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0/10 | 0.251 (0.250) | 0 | 0.26 | 8.4 | 0.68 (0.70; left_hind_pitch) | 0 | 6 (3) / 57 (46) / 51 (43) / 53 (33) | 0.7 % | | | | |
| 0.5 | 0/10 | 0.240 (0.206) | 2 | 0.18 | 8.2 | 0.76 (1.02; left_hind_flexion) | 2 | 20 (10) / 58 (46) / 51 (43) / 53 (33) | 1.4 % | 3/10 | 3 | 0.192-0.254 | 0.63-1.07 |
| 1 | 0/10 | 0.236 (0.199) | 3 | 0.21 | 9.6 | 0.82 (1.08; left_hind_flexion) | 2 | 21 (11) / 57 (46) / 51 (43) / 53 (33) | 2.1 % | 4/10 | 4 | 0.196-0.253 | 0.64-1.10 |
| 2 | 0/10 | 0.219 (0.201) | 7 | 0.20 | 8.4 | 1.04 (1.14; left_hind_flexion) | 8 | 22 (11) / 57 (45) / 51 (43) / 52 (33) | 4.9 % | 9/10 | 9 | 0.197-0.236 | 0.96-1.17 |
| 3 | 0/10 | 0.213 (0.208) | 9 | 0.22 | 8.3 | 1.15 (1.18; right_hind_flexion) | 10 | 22 (11) / 56 (43) / 51 (43) / 52 (32) | 6.6 % | 10/10 | 10 | 0.202-0.207 | 1.19-1.22 |

### force limit (the sweep's break rule: low: none)

| Force limit, x nominal | Falls | Speed mean 2-20 s (min), m/s | Episodes under 90 % of the command | Sideways drift, worst, m | Heading change, worst, deg | Worst drive RMS / rating: worst joint's mean (worst episode; joint) | Episodes with a drive at or over 1.0 | Peak drive speed, rpm | Steps at the nominal torque limit, worst joint's mean | 10-20 s: slow / upright | 10-20 s: slow and hot | 10-20 s speed, range, m/s | 10-20 s larger hind knee RMS / rating, range |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.4 | 0/10 | 0.244 (0.211) | 1 | 0.26 | 7.6 | 0.61 (0.69; left_hind_flexion) | 0 | 13 (5) / 55 (46) / 78 (46) / 41 (35) | 0.0 % | 3/10 | 0 | 0.177-0.255 | 0.55-0.79 |
| 0.55 | 0/10 | 0.252 (0.251) | 0 | 0.27 | 7.6 | 0.61 (0.67; right_hind_pitch) | 0 | 9 (4) / 55 (47) / 60 (49) / 42 (36) | 0.0 % | | | | |
| 0.7 | 0/10 | 0.251 (0.251) | 0 | 0.27 | 8.3 | 0.65 (0.68; left_hind_pitch) | 0 | 7 (3) / 56 (47) / 63 (43) / 47 (38) | 0.0 % | 0/10 | 0 | 0.250-0.254 | 0.62-0.71 |
| 0.85 | 0/10 | 0.251 (0.250) | 0 | 0.27 | 7.9 | 0.67 (0.72; left_hind_flexion) | 0 | 7 (3) / 58 (47) / 51 (43) / 52 (38) | 0.0 % | | | | |
| 1 | 0/10 | 0.251 (0.250) | 0 | 0.26 | 8.4 | 0.68 (0.70; left_hind_pitch) | 0 | 6 (3) / 57 (46) / 51 (43) / 53 (33) | 0.7 % | | | | |

### position gain kp (the sweep's break rule: low: 0.8 (speed 0.191 m/s, 24 % below nominal); high: none)

| Position gain kp, x nominal | Falls | Speed mean 2-20 s (min), m/s | Episodes under 90 % of the command | Sideways drift, worst, m | Heading change, worst, deg | Worst drive RMS / rating: worst joint's mean (worst episode; joint) | Episodes with a drive at or over 1.0 | Peak drive speed, rpm | Steps at the nominal torque limit, worst joint's mean | 10-20 s: slow / upright | 10-20 s: slow and hot | 10-20 s speed, range, m/s | 10-20 s larger hind knee RMS / rating, range |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.4 | 0/10 | 0.177 (0.174) | 10 | 0.17 | 7.5 | 0.94 (0.95; right_hind_flexion) | 0 | 7 (5) / 39 (34) / 36 (33) / 34 (22) | 1.7 % | | | | |
| 0.6 | 0/10 | 0.177 (0.175) | 10 | 0.19 | 5.9 | 1.02 (1.04; left_hind_flexion) | 10 | 10 (8) / 48 (39) / 41 (38) / 43 (29) | 5.5 % | | | | |
| 0.8 | 0/10 | 0.191 (0.183) | 10 | 0.14 | 8.6 | 0.99 (1.04; left_hind_flexion) | 6 | 14 (10) / 53 (41) / 48 (41) / 37 (31) | 5.2 % | 10/10 | 10 | 0.176-0.180 | 1.03-1.07 |
| 1 | 0/10 | 0.251 (0.250) | 0 | 0.26 | 8.4 | 0.68 (0.70; left_hind_pitch) | 0 | 6 (3) / 57 (46) / 51 (43) / 53 (33) | 0.7 % | | | | |
| 1.2 | 0/10 | 0.251 (0.250) | 0 | 0.26 | 8.0 | 0.74 (0.76; left_hind_pitch) | 0 | 8 (3) / 60 (49) / 53 (45) / 56 (43) | 1.4 % | | | | |
| 1.5 | 0/10 | 0.250 (0.249) | 0 | 0.25 | 7.6 | 0.81 (0.83; left_hind_pitch) | 0 | 11 (5) / 65 (52) / 82 (49) / 55 (45) | 2.4 % | | | | |

### sideways push (the sweep's break rule: low: -1 (5/10 fell); high: 1.5 (10/10 fell))

| Push, free-body dv, m/s (+ left, - right) | Falls | Speed mean 2-20 s (min), m/s | Episodes under 90 % of the command | Sideways drift, worst, m | Heading change, worst, deg | Worst drive RMS / rating: worst joint's mean (worst episode; joint) | Episodes with a drive at or over 1.0 | Peak drive speed, rpm | Steps at the nominal torque limit, worst joint's mean | Extra travel along the push, mean (max), m | 10-20 s: slow / upright | 10-20 s: slow and hot | 10-20 s speed, range, m/s | 10-20 s larger hind knee RMS / rating, range |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| -1.5 | 10/10 | all fell | | | | | | | | | 0/0 | 0 |  |  |
| -1.25 | 0/10 | 0.222 (0.201) | 6 | 0.57 | 21.4 | 0.84 (0.96; right_hind_flexion) | 0 | 66 (31) / 130 (61) / 217 (72) / 236 (73) | 2.5 % | 0.201 (0.219) | 5/10 | 5 | 0.186-0.253 | 0.65-1.06 |
| -1 | 5/10 | 0.203 (0.182) | 5 | 0.26 | 5.1 | 0.89 (0.94; right_hind_flexion) | 0 | 51 (25) / 148 (53) / 316 (73) / 156 (48) | 3.5 % | 0.111 (0.118) | 5/5 | 2 | 0.190-0.218 | 0.97-1.09 |
| -0.75 | 0/10 | 0.238 (0.218) | 4 | 0.20 | 8.5 | 0.77 (0.91; right_hind_flexion) | 0 | 23 (12) / 64 (46) / 105 (43) / 86 (33) | 1.7 % | 0.068 (0.084) | 4/10 | 4 | 0.193-0.253 | 0.62-1.05 |
| -0.5 | 0/10 | 0.215 (0.212) | 10 | 0.18 | 8.6 | 0.93 (0.94; right_hind_flexion) | 0 | 27 (11) / 67 (44) / 56 (43) / 60 (32) | 3.9 % | 0.044 (0.056) | 10/10 | 10 | 0.192-0.200 | 1.03-1.04 |
| -0.4 | 0/10 | 0.212 (0.209) | 10 | 0.15 | 8.7 | 0.93 (0.94; left_hind_flexion) | 0 | 23 (12) / 57 (44) / 56 (44) / 53 (33) | 4.2 % | 0.031 (0.034) | 10/10 | 10 | 0.190-0.200 | 1.03-1.06 |
| -0.3 | 0/10 | 0.212 (0.210) | 10 | 0.18 | 9.3 | 0.93 (0.94; left_hind_flexion) | 0 | 22 (11) / 57 (44) / 51 (43) / 61 (33) | 4.1 % | 0.025 (0.030) | 10/10 | 10 | 0.191-0.199 | 1.03-1.06 |
| -0.2 | 0/10 | 0.215 (0.212) | 10 | 0.14 | 7.6 | 0.92 (0.93; left_hind_flexion) | 0 | 20 (10) / 57 (44) / 51 (43) / 53 (32) | 3.8 % | 0.027 (0.036) | 10/10 | 10 | 0.191-0.200 | 1.03-1.05 |
| -0.1 | 0/10 | 0.242 (0.217) | 3 | 0.17 | 8.6 | 0.73 (0.92; right_hind_flexion) | 0 | 19 (10) / 57 (46) / 51 (43) / 53 (33) | 1.1 % | 0.014 (0.025) | 3/10 | 3 | 0.193-0.255 | 0.62-1.04 |
| 0 | 0/10 | 0.251 (0.250) | 0 | 0.26 | 8.4 | 0.68 (0.70; left_hind_pitch) | 0 | 6 (3) / 57 (46) / 51 (43) / 53 (33) | 0.7 % |  | 0/10 | 0 | 0.250-0.253 | 0.62-0.70 |
| 0.1 | 0/10 | 0.251 (0.250) | 0 | 0.26 | 8.5 | 0.67 (0.70; left_hind_pitch) | 0 | 7 (3) / 57 (47) / 51 (43) / 53 (33) | 0.7 % | 0.009 (0.012) | 0/10 | 0 | 0.250-0.254 | 0.62-0.71 |
| 0.2 | 0/10 | 0.250 (0.249) | 0 | 0.25 | 8.4 | 0.68 (0.70; left_hind_pitch) | 0 | 10 (4) / 57 (47) / 53 (43) / 53 (33) | 0.8 % | 0.015 (0.015) | 0/10 | 0 | 0.249-0.253 | 0.61-0.67 |
| 0.3 | 0/10 | 0.247 (0.220) | 1 | 0.25 | 8.4 | 0.68 (0.89; left_hind_pitch) | 0 | 13 (9) / 69 (47) / 62 (43) / 53 (33) | 0.8 % | 0.020 (0.024) | 1/10 | 1 | 0.199-0.255 | 0.61-1.03 |
| 0.4 | 0/10 | 0.242 (0.218) | 2 | 0.24 | 8.4 | 0.70 (0.90; left_hind_flexion) | 0 | 20 (10) / 74 (47) / 68 (43) / 53 (33) | 0.9 % | 0.026 (0.034) | 2/10 | 2 | 0.196-0.252 | 0.62-1.04 |
| 0.5 | 0/10 | 0.248 (0.242) | 0 | 0.25 | 8.4 | 0.67 (0.78; right_hind_flexion) | 0 | 20 (7) / 88 (47) / 75 (43) / 58 (33) | 0.9 % | 0.029 (0.030) | 0/10 | 0 | 0.237-0.254 | 0.62-0.87 |
| 0.75 | 0/10 | 0.249 (0.243) | 0 | 0.22 | 8.2 | 0.69 (0.77; left_hind_pitch) | 0 | 35 (7) / 107 (47) / 87 (43) / 66 (33) | 0.8 % | 0.052 (0.056) | 0/10 | 0 | 0.239-0.253 | 0.62-0.85 |
| 1 | 0/10 | 0.213 (0.209) | 10 | 0.18 | 3.8 | 0.93 (0.94; left_hind_flexion) | 0 | 42 (14) / 124 (44) / 119 (45) / 95 (33) | 4.2 % | 0.098 (0.102) | 10/10 | 10 | 0.189-0.201 | 1.02-1.06 |
| 1.25 | 0/10 | 0.228 (0.209) | 4 | 0.26 | 8.4 | 0.81 (0.97; left_hind_flexion) | 0 | 50 (18) / 138 (48) / 231 (54) / 123 (40) | 2.6 % | 0.169 (0.173) | 4/10 | 4 | 0.193-0.252 | 0.64-1.05 |
| 1.5 | 10/10 | all fell | | | | | | | | | 0/0 | 0 |  |  |

## Sweep `walk60_fore128_sideways_300000`: sideways pushes, seeds 300000-300009 (commit aa150f89, dirty; 720 episodes, 14.9 min)

Nominal model, observation noise on, no randomisation, 10 ms delay and 0.25 m/s unless the row says otherwise; 10 episodes per point. A push lasts 0.1 s from 8 s; its free-body dv is its impulse over the robot's whole mass.

### delay, 0.25 m/s (the sweep's break rule: low: none; high: none)

| Delay, ms | Falls | Speed mean 2-20 s (min), m/s | Episodes under 90 % of the command | Sideways drift, worst, m | Heading change, worst, deg | Worst drive RMS / rating: worst joint's mean (worst episode; joint) | Episodes with a drive at or over 1.0 | Peak drive speed, rpm | Steps at the nominal torque limit, worst joint's mean | 10-20 s: slow / upright | 10-20 s: slow and hot | 10-20 s speed, range, m/s | 10-20 s larger hind knee RMS / rating, range |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0/10 | 0.253 (0.251) | 0 | 0.25 | 3.4 | 0.64 (0.67; right_hind_pitch) | 0 | 8 (3) / 56 (47) / 50 (43) / 52 (33) | 0.6 % | | | | |
| 2 | 0/10 | 0.253 (0.253) | 0 | 0.24 | 3.6 | 0.65 (0.68; right_hind_flexion) | 0 | 8 (3) / 56 (46) / 49 (42) / 53 (33) | 0.5 % | | | | |
| 5 | 0/10 | 0.254 (0.253) | 0 | 0.25 | 3.4 | 0.70 (0.75; right_hind_flexion) | 0 | 10 (5) / 55 (45) / 49 (42) / 52 (33) | 0.5 % | | | | |
| 10 | 0/10 | 0.251 (0.250) | 0 | 0.25 | 3.8 | 0.67 (0.69; left_hind_pitch) | 0 | 7 (4) / 57 (46) / 51 (43) / 53 (33) | 0.7 % | | | | |
| 15 | 0/10 | 0.232 (0.189) | 3 | 0.21 | 11.6 | 0.85 (1.07; right_hind_flexion) | 2 | 24 (13) / 55 (45) / 56 (44) / 52 (34) | 2.4 % | 5/10 | 4 | 0.181-0.256 | 0.62-1.10 |
| 16 | 0/10 | 0.216 (0.187) | 6 | 0.32 | 16.0 | 0.93 (1.09; left_hind_flexion) | 6 | 23 (14) / 55 (45) / 53 (45) / 52 (35) | 4.1 % | 7/10 | 6 | 0.178-0.256 | 0.62-1.12 |
| 17 | 0/10 | 0.202 (0.183) | 8 | 0.37 | 15.0 | 0.99 (1.10; left_hind_flexion) | 7 | 25 (15) / 55 (45) / 55 (46) / 52 (36) | 5.6 % | 8/10 | 8 | 0.180-0.242 | 0.84-1.11 |
| 18 | 0/10 | 0.215 (0.184) | 6 | 0.36 | 16.7 | 0.89 (1.10; left_hind_flexion) | 6 | 26 (16) / 54 (45) / 54 (46) / 52 (36) | 4.3 % | 6/10 | 6 | 0.177-0.250 | 0.62-1.12 |
| 19 | 0/10 | 0.240 (0.184) | 1 | 0.24 | 13.9 | 0.76 (1.09; left_hind_pitch) | 1 | 22 (14) / 58 (48) / 53 (46) / 51 (34) | 1.4 % | 1/10 | 1 | 0.181-0.248 | 0.61-1.10 |
| 20 | 0/10 | 0.230 (0.185) | 3 | 0.38 | 20.2 | 0.77 (1.10; left_hind_pitch) | 3 | 23 (16) / 57 (46) / 54 (47) / 52 (36) | 2.3 % | 3/10 | 3 | 0.180-0.248 | 0.62-1.12 |
| 30 | 0/10 | 0.237 (0.191) | 1 | 0.17 | 5.9 | 0.76 (1.09; left_hind_pitch) | 1 | 24 (10) / 56 (47) / 54 (43) / 51 (34) | 1.7 % | 1/10 | 1 | 0.184-0.245 | 0.63-1.12 |
| 40 | 0/10 | 0.221 (0.196) | 4 | 0.32 | 9.8 | 0.82 (1.13; left_hind_flexion) | 4 | 19 (10) / 68 (48) / 57 (45) / 71 (40) | 3.2 % | 4/10 | 3 | 0.197-0.238 | 0.65-1.12 |
| 60 | 0/10 | 0.240 (0.231) | 0 | 0.24 | 5.7 | 0.79 (0.82; right_fore_pitch) | 0 | 23 (10) / 113 (46) / 66 (43) / 63 (34) | 2.2 % | 1/10 | 0 | 0.220-0.243 | 0.64-0.91 |

### delay, 0.15 m/s (the sweep's break rule: low: none; high: none)

| Delay, ms | Falls | Speed mean 2-20 s (min), m/s | Episodes under 90 % of the command | Sideways drift, worst, m | Heading change, worst, deg | Worst drive RMS / rating: worst joint's mean (worst episode; joint) | Episodes with a drive at or over 1.0 | Peak drive speed, rpm | Steps at the nominal torque limit, worst joint's mean |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0/10 | 0.145 (0.145) | 0 | 0.15 | 4.8 | 0.67 (0.68; right_fore_pitch) | 0 | 11 (6) / 54 (40) / 48 (41) / 35 (31) | 0.7 % |
| 2 | 0/10 | 0.146 (0.145) | 0 | 0.16 | 3.4 | 0.66 (0.67; right_fore_pitch) | 0 | 11 (6) / 53 (40) / 49 (41) / 35 (31) | 0.6 % |
| 5 | 0/10 | 0.147 (0.147) | 0 | 0.15 | 4.2 | 0.63 (0.64; left_hind_flexion) | 0 | 11 (5) / 52 (39) / 49 (40) / 34 (31) | 0.4 % |
| 10 | 0/10 | 0.142 (0.141) | 0 | 0.15 | 3.7 | 0.70 (0.70; right_fore_pitch) | 0 | 14 (6) / 70 (40) / 58 (40) / 48 (31) | 1.2 % |
| 15 | 0/10 | 0.143 (0.142) | 0 | 0.15 | 4.0 | 0.68 (0.69; right_fore_pitch) | 0 | 12 (5) / 71 (39) / 53 (40) / 35 (31) | 1.0 % |
| 16 | 0/10 | 0.142 (0.142) | 0 | 0.14 | 3.9 | 0.69 (0.70; right_fore_pitch) | 0 | 11 (6) / 71 (39) / 53 (40) / 34 (31) | 1.1 % |
| 17 | 0/10 | 0.142 (0.141) | 0 | 0.16 | 3.8 | 0.69 (0.70; right_fore_pitch) | 0 | 12 (6) / 68 (39) / 55 (40) / 34 (31) | 1.2 % |
| 18 | 0/10 | 0.141 (0.140) | 0 | 0.15 | 4.2 | 0.71 (0.71; right_fore_pitch) | 0 | 12 (6) / 66 (39) / 57 (40) / 35 (31) | 1.4 % |
| 19 | 0/10 | 0.136 (0.135) | 0 | 0.14 | 5.4 | 0.75 (0.76; right_fore_pitch) | 0 | 12 (6) / 78 (40) / 59 (41) / 35 (31) | 1.9 % |
| 20 | 0/10 | 0.137 (0.136) | 0 | 0.16 | 3.3 | 0.73 (0.74; right_fore_pitch) | 0 | 12 (6) / 70 (39) / 58 (40) / 34 (31) | 1.6 % |
| 30 | 0/10 | 0.135 (0.132) | 1 | 0.13 | 6.8 | 0.75 (0.77; left_fore_pitch) | 0 | 12 (6) / 76 (39) / 57 (40) / 47 (30) | 1.9 % |
| 40 | 0/10 | 0.135 (0.132) | 5 | 0.13 | 6.9 | 0.77 (0.79; right_fore_pitch) | 0 | 13 (6) / 77 (40) / 57 (40) / 35 (30) | 2.5 % |
| 60 | 0/10 | 0.137 (0.119) | 5 | 0.17 | 8.5 | 0.85 (1.02; right_fore_pitch) | 1 | 35 (13) / 124 (66) / 64 (51) / 65 (43) | 4.1 % |

### friction (the sweep's break rule: low: none; high: none)

| Floor friction | Falls | Speed mean 2-20 s (min), m/s | Episodes under 90 % of the command | Sideways drift, worst, m | Heading change, worst, deg | Worst drive RMS / rating: worst joint's mean (worst episode; joint) | Episodes with a drive at or over 1.0 | Peak drive speed, rpm | Steps at the nominal torque limit, worst joint's mean | 10-20 s: slow / upright | 10-20 s: slow and hot | 10-20 s speed, range, m/s | 10-20 s larger hind knee RMS / rating, range |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.15 | 0/10 | 0.258 (0.257) | 0 | 0.19 | 3.4 | 0.80 (0.80; right_hind_flexion) | 0 | 8 (4) / 55 (48) / 47 (43) / 50 (33) | 0.7 % | | | | |
| 0.2 | 0/10 | 0.257 (0.255) | 0 | 0.22 | 3.5 | 0.78 (0.78; right_hind_flexion) | 0 | 11 (6) / 55 (48) / 47 (43) / 50 (33) | 0.7 % | | | | |
| 0.3 | 0/10 | 0.256 (0.256) | 0 | 0.26 | 3.5 | 0.74 (0.76; right_hind_flexion) | 0 | 14 (6) / 55 (47) / 47 (43) / 50 (33) | 0.6 % | | | | |
| 0.4 | 0/10 | 0.257 (0.256) | 0 | 0.24 | 3.6 | 0.73 (0.77; right_hind_flexion) | 0 | 13 (6) / 54 (46) / 48 (42) / 49 (33) | 0.5 % | | | | |
| 0.5 | 0/10 | 0.257 (0.256) | 0 | 0.24 | 3.3 | 0.74 (0.76; right_hind_flexion) | 0 | 13 (6) / 53 (45) / 48 (42) / 49 (33) | 0.5 % | | | | |
| 0.6 | 0/10 | 0.256 (0.253) | 0 | 0.23 | 3.8 | 0.72 (0.75; right_hind_flexion) | 0 | 11 (6) / 53 (45) / 47 (42) / 50 (33) | 0.5 % | | | | |
| 0.8 | 0/10 | 0.251 (0.236) | 0 | 0.26 | 3.9 | 0.66 (0.82; left_hind_flexion) | 0 | 13 (8) / 57 (46) / 51 (43) / 53 (33) | 0.6 % | 1/10 | 0 | 0.223-0.253 | 0.62-0.94 |
| the model's own | 0/10 | 0.251 (0.250) | 0 | 0.25 | 3.8 | 0.67 (0.69; left_hind_pitch) | 0 | 7 (4) / 57 (46) / 51 (43) / 53 (33) | 0.7 % | | | | |
| 1 | 0/10 | 0.248 (0.247) | 0 | 0.25 | 3.5 | 0.71 (0.72; right_hind_pitch) | 0 | 5 (3) / 56 (47) / 51 (43) / 52 (33) | 0.8 % | | | | |
| 1.2 | 0/10 | 0.246 (0.245) | 0 | 0.26 | 3.2 | 0.74 (0.76; right_hind_pitch) | 0 | 6 (3) / 56 (47) / 50 (43) / 52 (33) | 0.9 % | | | | |

### mass, every body (the sweep's break rule: low: none; high: none)

| Mass, x every body | Falls | Speed mean 2-20 s (min), m/s | Episodes under 90 % of the command | Sideways drift, worst, m | Heading change, worst, deg | Worst drive RMS / rating: worst joint's mean (worst episode; joint) | Episodes with a drive at or over 1.0 | Peak drive speed, rpm | Steps at the nominal torque limit, worst joint's mean | 10-20 s: slow / upright | 10-20 s: slow and hot | 10-20 s speed, range, m/s | 10-20 s larger hind knee RMS / rating, range |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.8 | 0/10 | 0.245 (0.206) | 1 | 0.24 | 3.6 | 0.62 (0.90; right_fore_pitch) | 0 | 11 (9) / 57 (48) / 54 (43) / 57 (33) | 0.7 % | 1/10 | 0 | 0.204-0.251 | 0.56-0.90 |
| 0.9 | 0/10 | 0.242 (0.204) | 2 | 0.25 | 3.7 | 0.66 (0.96; right_hind_flexion) | 0 | 12 (10) / 57 (47) / 52 (43) / 54 (33) | 0.7 % | 2/10 | 0 | 0.200-0.251 | 0.59-0.97 |
| 1 | 0/10 | 0.251 (0.250) | 0 | 0.25 | 3.8 | 0.67 (0.69; left_hind_pitch) | 0 | 7 (4) / 57 (46) / 51 (43) / 53 (33) | 0.7 % | | | | |
| 1.1 | 0/10 | 0.252 (0.250) | 0 | 0.26 | 4.0 | 0.70 (0.77; right_hind_pitch) | 0 | 7 (4) / 56 (46) / 51 (43) / 52 (33) | 0.7 % | | | | |
| 1.2 | 0/10 | 0.250 (0.243) | 0 | 0.23 | 3.9 | 0.74 (0.90; right_hind_pitch) | 0 | 19 (8) / 56 (45) / 50 (42) / 51 (33) | 0.6 % | 0/10 | 0 | 0.235-0.255 | 0.68-0.98 |
| 1.4 | 0/10 | 0.241 (0.220) | 2 | 0.23 | 4.8 | 0.99 (1.19; right_hind_flexion) | 4 | 21 (11) / 54 (45) / 49 (42) / 48 (33) | 2.6 % | 4/10 | 4 | 0.211-0.253 | 0.76-1.28 |
| 1.6 | 0/10 | 0.239 (0.217) | 1 | 0.23 | 4.1 | 1.11 (1.36; right_hind_flexion) | 8 | 19 (12) / 53 (45) / 49 (42) / 49 (33) | 3.6 % | 5/10 | 5 | 0.210-0.252 | 0.82-1.40 |

### payload on the chassis (the sweep's break rule: high: none)

| Payload on the chassis, kg | Falls | Speed mean 2-20 s (min), m/s | Episodes under 90 % of the command | Sideways drift, worst, m | Heading change, worst, deg | Worst drive RMS / rating: worst joint's mean (worst episode; joint) | Episodes with a drive at or over 1.0 | Peak drive speed, rpm | Steps at the nominal torque limit, worst joint's mean | 10-20 s: slow / upright | 10-20 s: slow and hot | 10-20 s speed, range, m/s | 10-20 s larger hind knee RMS / rating, range |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0/10 | 0.251 (0.250) | 0 | 0.25 | 3.8 | 0.67 (0.69; left_hind_pitch) | 0 | 7 (4) / 57 (46) / 51 (43) / 53 (33) | 0.7 % | | | | |
| 0.5 | 0/10 | 0.234 (0.199) | 4 | 0.26 | 3.9 | 0.80 (1.06; left_hind_flexion) | 2 | 20 (11) / 57 (46) / 52 (43) / 53 (33) | 2.1 % | 4/10 | 4 | 0.195-0.252 | 0.63-1.08 |
| 1 | 0/10 | 0.227 (0.199) | 6 | 0.25 | 4.0 | 0.90 (1.10; right_hind_flexion) | 3 | 26 (11) / 57 (46) / 52 (43) / 53 (33) | 3.2 % | 6/10 | 6 | 0.194-0.254 | 0.67-1.11 |
| 2 | 0/10 | 0.219 (0.205) | 8 | 0.23 | 5.6 | 1.05 (1.13; right_hind_flexion) | 8 | 22 (11) / 56 (45) / 51 (43) / 52 (33) | 4.8 % | 9/10 | 9 | 0.197-0.250 | 0.89-1.16 |
| 3 | 0/10 | 0.211 (0.205) | 10 | 0.21 | 4.1 | 1.16 (1.21; right_hind_flexion) | 10 | 22 (11) / 56 (42) / 51 (42) / 52 (31) | 7.0 % | 10/10 | 10 | 0.201-0.205 | 1.20-1.22 |

### force limit (the sweep's break rule: low: none)

| Force limit, x nominal | Falls | Speed mean 2-20 s (min), m/s | Episodes under 90 % of the command | Sideways drift, worst, m | Heading change, worst, deg | Worst drive RMS / rating: worst joint's mean (worst episode; joint) | Episodes with a drive at or over 1.0 | Peak drive speed, rpm | Steps at the nominal torque limit, worst joint's mean | 10-20 s: slow / upright | 10-20 s: slow and hot | 10-20 s speed, range, m/s | 10-20 s larger hind knee RMS / rating, range |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.4 | 0/10 | 0.241 (0.171) | 2 | 0.25 | 3.5 | 0.60 (0.81; right_hind_flexion) | 0 | 12 (5) / 52 (46) / 80 (48) / 41 (37) | 0.0 % | 2/10 | 0 | 0.157-0.255 | 0.55-0.83 |
| 0.55 | 0/10 | 0.252 (0.251) | 0 | 0.26 | 3.4 | 0.61 (0.64; right_hind_pitch) | 0 | 7 (3) / 53 (47) / 59 (49) / 42 (36) | 0.0 % | | | | |
| 0.7 | 0/10 | 0.249 (0.226) | 0 | 0.25 | 3.9 | 0.66 (0.86; right_hind_flexion) | 0 | 21 (11) / 56 (47) / 64 (43) / 47 (38) | 0.0 % | 1/10 | 0 | 0.210-0.254 | 0.62-0.95 |
| 0.85 | 0/10 | 0.251 (0.250) | 0 | 0.25 | 3.7 | 0.67 (0.72; left_hind_flexion) | 0 | 7 (3) / 56 (47) / 51 (43) / 52 (38) | 0.0 % | | | | |
| 1 | 0/10 | 0.251 (0.250) | 0 | 0.25 | 3.8 | 0.67 (0.69; left_hind_pitch) | 0 | 7 (4) / 57 (46) / 51 (43) / 53 (33) | 0.7 % | | | | |

### position gain kp (the sweep's break rule: low: 0.8 (speed 0.191 m/s, 24 % below nominal); high: none)

| Position gain kp, x nominal | Falls | Speed mean 2-20 s (min), m/s | Episodes under 90 % of the command | Sideways drift, worst, m | Heading change, worst, deg | Worst drive RMS / rating: worst joint's mean (worst episode; joint) | Episodes with a drive at or over 1.0 | Peak drive speed, rpm | Steps at the nominal torque limit, worst joint's mean | 10-20 s: slow / upright | 10-20 s: slow and hot | 10-20 s speed, range, m/s | 10-20 s larger hind knee RMS / rating, range |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.4 | 0/10 | 0.177 (0.175) | 10 | 0.15 | 4.0 | 0.94 (0.95; right_hind_flexion) | 0 | 7 (5) / 40 (34) / 36 (33) / 33 (22) | 1.7 % | | | | |
| 0.6 | 0/10 | 0.176 (0.175) | 10 | 0.12 | 4.8 | 1.02 (1.04; left_hind_flexion) | 10 | 10 (7) / 47 (39) / 42 (38) / 43 (29) | 5.4 % | | | | |
| 0.8 | 0/10 | 0.191 (0.184) | 10 | 0.22 | 9.7 | 0.98 (1.05; left_hind_flexion) | 5 | 15 (12) / 53 (42) / 47 (42) / 39 (32) | 5.2 % | 10/10 | 10 | 0.176-0.182 | 1.04-1.08 |
| 1 | 0/10 | 0.251 (0.250) | 0 | 0.25 | 3.8 | 0.67 (0.69; left_hind_pitch) | 0 | 7 (4) / 57 (46) / 51 (43) / 53 (33) | 0.7 % | | | | |
| 1.2 | 0/10 | 0.251 (0.250) | 0 | 0.24 | 3.8 | 0.73 (0.76; left_hind_pitch) | 0 | 7 (3) / 59 (50) / 55 (45) / 56 (44) | 1.4 % | | | | |
| 1.5 | 0/10 | 0.249 (0.248) | 0 | 0.25 | 3.2 | 0.81 (0.83; left_hind_pitch) | 0 | 11 (6) / 65 (53) / 82 (49) / 55 (45) | 2.4 % | | | | |

### sideways push (the sweep's break rule: low: -1 (5/10 fell); high: 1.25 (1/10 fell))

| Push, free-body dv, m/s (+ left, - right) | Falls | Speed mean 2-20 s (min), m/s | Episodes under 90 % of the command | Sideways drift, worst, m | Heading change, worst, deg | Worst drive RMS / rating: worst joint's mean (worst episode; joint) | Episodes with a drive at or over 1.0 | Peak drive speed, rpm | Steps at the nominal torque limit, worst joint's mean | Extra travel along the push, mean (max), m | 10-20 s: slow / upright | 10-20 s: slow and hot | 10-20 s speed, range, m/s | 10-20 s larger hind knee RMS / rating, range |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| -1.5 | 10/10 | all fell | | | | | | | | | 0/0 | 0 |  |  |
| -1.25 | 3/10 | 0.221 (0.198) | 4 | 0.53 | 12.7 | 0.86 (0.95; right_hind_flexion) | 0 | 61 (23) / 132 (51) / 254 (61) / 236 (58) | 2.8 % | 0.210 (0.266) | 4/7 | 4 | 0.189-0.253 | 0.64-1.06 |
| -1 | 5/10 | 0.230 (0.202) | 1 | 0.04 | 6.3 | 0.74 (0.90; right_hind_flexion) | 0 | 51 (19) / 144 (48) / 267 (49) / 151 (42) | 1.6 % | 0.118 (0.152) | 1/5 | 1 | 0.202-0.255 | 0.63-1.02 |
| -0.75 | 0/10 | 0.233 (0.220) | 6 | 0.21 | 5.5 | 0.81 (0.92; right_hind_flexion) | 0 | 23 (13) / 72 (46) / 108 (43) / 110 (34) | 2.2 % | 0.068 (0.082) | 6/10 | 6 | 0.193-0.254 | 0.62-1.05 |
| -0.5 | 0/10 | 0.218 (0.210) | 9 | 0.26 | 6.4 | 0.90 (0.95; right_hind_flexion) | 0 | 27 (11) / 67 (46) / 60 (43) / 73 (33) | 3.6 % | 0.045 (0.058) | 9/10 | 9 | 0.189-0.251 | 0.62-1.06 |
| -0.4 | 0/10 | 0.219 (0.209) | 8 | 0.26 | 8.1 | 0.88 (0.95; left_hind_flexion) | 0 | 26 (13) / 57 (46) / 56 (44) / 53 (33) | 3.6 % | 0.031 (0.035) | 8/10 | 8 | 0.188-0.254 | 0.64-1.07 |
| -0.3 | 0/10 | 0.213 (0.211) | 10 | 0.21 | 9.8 | 0.93 (0.95; left_hind_flexion) | 0 | 25 (12) / 57 (44) / 56 (43) / 64 (33) | 4.2 % | 0.024 (0.030) | 10/10 | 10 | 0.191-0.200 | 1.03-1.05 |
| -0.2 | 0/10 | 0.215 (0.214) | 10 | 0.22 | 5.4 | 0.92 (0.94; left_hind_flexion) | 0 | 20 (10) / 57 (44) / 51 (43) / 53 (32) | 3.9 % | 0.027 (0.034) | 10/10 | 10 | 0.193-0.200 | 1.03-1.05 |
| -0.1 | 0/10 | 0.234 (0.216) | 5 | 0.25 | 4.8 | 0.78 (0.93; right_hind_flexion) | 0 | 20 (10) / 57 (46) / 51 (43) / 53 (33) | 1.8 % | 0.019 (0.030) | 5/10 | 5 | 0.194-0.252 | 0.62-1.05 |
| 0 | 0/10 | 0.251 (0.250) | 0 | 0.25 | 3.8 | 0.67 (0.69; left_hind_pitch) | 0 | 7 (4) / 57 (46) / 51 (43) / 53 (33) | 0.7 % |  | 0/10 | 0 | 0.250-0.254 | 0.62-0.70 |
| 0.1 | 0/10 | 0.250 (0.238) | 0 | 0.25 | 4.1 | 0.67 (0.78; left_hind_pitch) | 0 | 20 (8) / 57 (46) / 51 (43) / 53 (33) | 0.7 % | 0.010 (0.016) | 0/10 | 0 | 0.227-0.254 | 0.62-0.86 |
| 0.2 | 0/10 | 0.247 (0.224) | 1 | 0.26 | 4.1 | 0.68 (0.88; left_hind_pitch) | 0 | 20 (9) / 57 (47) / 52 (43) / 53 (33) | 0.8 % | 0.015 (0.018) | 1/10 | 1 | 0.205-0.254 | 0.62-1.01 |
| 0.3 | 0/10 | 0.250 (0.249) | 0 | 0.24 | 4.1 | 0.67 (0.70; left_hind_pitch) | 0 | 13 (4) / 64 (47) / 62 (43) / 53 (33) | 0.9 % | 0.020 (0.024) | 0/10 | 0 | 0.250-0.254 | 0.62-0.71 |
| 0.4 | 0/10 | 0.250 (0.249) | 0 | 0.25 | 3.8 | 0.69 (0.72; right_hind_pitch) | 0 | 15 (5) / 82 (47) / 66 (43) / 53 (33) | 0.9 % | 0.024 (0.028) | 0/10 | 0 | 0.250-0.252 | 0.62-0.65 |
| 0.5 | 0/10 | 0.249 (0.248) | 0 | 0.24 | 3.9 | 0.68 (0.71; right_hind_pitch) | 0 | 19 (5) / 84 (47) / 75 (43) / 57 (33) | 0.9 % | 0.029 (0.033) | 0/10 | 0 | 0.250-0.251 | 0.62-0.65 |
| 0.75 | 0/10 | 0.246 (0.224) | 1 | 0.25 | 3.9 | 0.69 (0.87; left_hind_pitch) | 0 | 35 (9) / 107 (47) / 91 (43) / 66 (33) | 0.8 % | 0.052 (0.058) | 1/10 | 1 | 0.203-0.252 | 0.62-1.01 |
| 1 | 0/10 | 0.214 (0.208) | 10 | 0.15 | 4.2 | 0.93 (0.96; left_hind_flexion) | 0 | 42 (14) / 122 (44) / 119 (45) / 95 (33) | 4.1 % | 0.097 (0.105) | 10/10 | 10 | 0.189-0.200 | 1.03-1.07 |
| 1.25 | 1/10 | 0.219 (0.197) | 6 | 0.25 | 10.8 | 0.88 (0.96; left_hind_flexion) | 0 | 52 (17) / 134 (48) / 234 (53) / 122 (40) | 3.4 % | 0.170 (0.203) | 6/9 | 6 | 0.184-0.254 | 0.62-1.08 |
| 1.5 | 10/10 | all fell | | | | | | | | | 0/0 | 0 |  |  |

