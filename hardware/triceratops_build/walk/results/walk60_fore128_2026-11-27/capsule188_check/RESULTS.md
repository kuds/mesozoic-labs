# Model-based trot on `trike60_C_closed_capsule.xml`: results (SIMULATION)

Written by `walk/fore128_walk60.py summarise`. Model `trike60_C_closed_capsule.xml`. Every number here is simulation, of a model whose masses and contact are estimates; nothing is measured. Reported, not judged: no pre-registration covers this model. PREREG_WALK60.md's W1-W4 were written for the 188 mm roll-ankle model (trike60_C_closed_ankleroll.xml); shown beside the measurements, not judged. PREREG_WALK60.md judged only J1-J4; its thresholds on every other case here are for orientation only.

- Speed: the mean forward speed from 2 s to the end of the 20 s episode, m/s, over the episodes that did not fall. In the sweep the push comes at 8 s, so a slower gait after it is averaged with the 6 s before it, and the sweep's break rule (a fall, or a mean more than 20 % below the nominal point's) uses this mean. Where `fore128_slow_gait.py` measured the same episodes (checked seed by seed: falls and 2-20 s speeds for sweep rows, whole-episode and 2-20 s speeds for judged conditions), the 10-20 s speed and the larger hind knee's load over 10-20 s are shown beside the rows. Slow: under 0.225 m/s over 10-20 s. Slow and hot: slow, and the larger hind knee's RMS torque over 10-20 s at or over its continuous rating (1.0); the cut-offs are `fore128_slow_gait.py`'s, not a pre-registration's.
- Drive load: RMS torque / continuous rating at the joint (roll and pitch 7.5, front elbow 10.8, hind knee 6.75 N m: rl/trike_env.py cont_torque_*; desk ratings).
- Peak drive speed: the largest |joint speed| over the episodes that did not fall, from t = 2 s, rpm, per group roll / pitch / front elbow / hind knee, with the largest 99th percentile in brackets. The budget's allowances (desk): roll 83, pitch 83, front elbow 108, hind knee 83 rpm (X4 83 rpm: DECISION_LOG.md E1 (rated-speed cap above 24 V); X6 elbow 108 rpm: 0.70 x 176 rpm no-load at 48 V x 42/48 (budget/size_budget.py DRIVES, sim/CRAWL_REPORT.md); both as rl/analysis/gait_pipe1.py RPM_ALLOW holds them).
- Torque-limit time: the share of 1 ms physics steps with a drive's torque at or above 99 % of its limit (the model clamps each drive at 70 % of the maker's peak).

## PREREG_WALK60's conditions, reported only on this model, seeds 101-110 (commit aa150f89, dirty)

Randomisation and sensor noise on, 10 episodes per condition.

| Condition | Falls | Speed mean (min), m/s | Episodes under 90 % of the command | Worst drive RMS / rating, worst episode (joint) | Episodes with a drive at or over 1.0 | Worst joint's mean RMS / rating | Heading change, worst episode | Sideways drift, worst episode | Slip ratio, worst foot's mean (at the sole origin / at the contacts) | Peak drive speed, rpm | Steps at the episode's own torque limit, worst joint-episode | Hind touchdowns per s, worst episode | PREREG_WALK60's thresholds (written for the 188 mm roll-ankle model) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| J1 grippy (friction 1.0), the model's contact | 0/10 | 0.247 (0.244) | 0 | 0.77 (left_hind_pitch) | 0 | 0.72 (left_hind_pitch) | 5.5 deg | 0.21 m | 0.12 / 0.06 | 7 (4) / 40 (34) / 39 (35) / 53 (41) | 3.0 % (right_fore_pitch) | 1.7 | W1 met (0 falls); W2 met (0.247, min 0.244); W3 met (0.77); W4 met (0.12) |
| J2 slippery (friction 0.4), the model's contact | 0/10 | 0.256 (0.252) | 0 | 0.80 (left_hind_flexion) | 0 | 0.73 (right_hind_flexion) | 5.7 deg | 0.28 m | 0.12 / 0.09 | 12 (7) / 39 (32) / 37 (34) / 52 (41) | 1.5 % (left_hind_flexion) | 3.7 | W1 met (0 falls); W2 met (0.256, min 0.252); W3 met (0.80); W4 met (0.12) |
| J3 grippy (friction 1.0), stiffer contact (impratio 10) | 0/10 | 0.241 (0.193) | 2 | 1.04 (left_hind_flexion) | 2 | 0.82 (left_hind_pitch) | 20.7 deg | 0.23 m | 0.11 / 0.05 | 25 (17) / 40 (34) / 46 (35) / 57 (41) | 9.5 % (right_hind_flexion) | 6.1 | W1 met (0 falls); W2 **not met** (0.241, min 0.193); W3 **not met** (1.04); W4 met (0.11) |
| J4 slippery (friction 0.4), stiffer contact (impratio 10) | 0/10 | 0.256 (0.251) | 0 | 0.79 (left_hind_flexion) | 0 | 0.73 (right_hind_flexion) | 5.5 deg | 0.27 m | 0.13 / 0.10 | 12 (7) / 39 (32) / 38 (33) / 49 (40) | 1.5 % (left_hind_flexion) | 3.8 | W1 met (0 falls); W2 met (0.256, min 0.251); W3 met (0.79); W4 met (0.13) |
| R1 J1 at 0 ms delay | 0/10 | 0.247 (0.227) | 0 | 0.89 (right_hind_flexion) | 0 | 0.69 (right_hind_pitch) | 5.9 deg | 0.25 m | 0.10 / not run | 11 (8) / 41 (35) / 48 (35) / 54 (41) | 3.6 % (right_hind_flexion) | 3.8 | orientation only: W1 met (0 falls); W2 met (0.247, min 0.227); W3 met (0.89); W4 met (0.10) |
| R2 J2 at 0 ms delay | 0/10 | 0.257 (0.253) | 0 | 0.78 (left_hind_flexion) | 0 | 0.71 (right_hind_flexion) | 5.8 deg | 0.25 m | 0.10 / not run | 12 (6) / 39 (33) / 38 (34) / 51 (41) | 1.4 % (left_hind_flexion) | 2.4 | orientation only: W1 met (0 falls); W2 met (0.257, min 0.253); W3 met (0.78); W4 met (0.10) |
| R3 J1 at the 0.15 m/s command | 0/10 | 0.138 (0.134) | 2 | 0.78 (right_fore_pitch) | 0 | 0.71 (right_fore_pitch) | 6.7 deg | 0.11 m | 0.18 / not run | 15 (7) / 36 (28) / 40 (33) / 48 (37) | 4.6 % (right_fore_pitch) | 1.9 | none (not at the 0.25 m/s command) |
| R4 J1 standing (0 m/s) | 0/10 | 0.000 (-0.000) |  | 0.52 (right_hind_flexion) | 0 | 0.49 (right_hind_flexion) | 0.0 deg | 0.00 m |  / not run | 1 (0) / 1 (1) / 3 (2) / 3 (2) | 0.0 % (right_hind_roll) | 0.0 | none (not at the 0.25 m/s command) |
| R5 nominal model, no randomisation or noise | 0/10 | 0.249 (0.248) | 0 | 0.74 (left_hind_pitch) | 0 | 0.73 (left_hind_pitch) | 0.7 deg | 0.02 m | 0.12 / not run | 4 (3) / 40 (33) / 38 (34) / 52 (33) | 2.3 % (right_fore_pitch) | 1.4 | orientation only: W1 met (0 falls); W2 met (0.249, min 0.248); W3 met (0.74); W4 met (0.12) |

### 10-20 s windows of the same episodes, seeds 101-110 (`capsule_judged_101-110.json`)

| Condition | Slow over 10-20 s / upright | Slow and hot (hind knee at or over 1.0) | 10-20 s speed, range, m/s | Larger hind knee RMS / rating over 10-20 s, range | Slow episodes: 10-20 s speed; hind knee; mean pitch (range), deg (+ nose down) | Whole-episode and 2-20 s speeds match the stored episodes |
|---|---|---|---|---|---|---|
| J1 | 0/10 | 0 | 0.244-0.248 | 0.60-0.68 |  | yes |
| J3 | 2/10 | 2 | 0.188-0.255 | 0.56-1.07 | 0.188-0.190; 1.06-1.07; -1.27 (-1.31 to -1.23) | yes |

## PREREG_WALK60's conditions, reported only on this model, seeds 201-210 (commit aa150f89, dirty)

Randomisation and sensor noise on, 10 episodes per condition.

| Condition | Falls | Speed mean (min), m/s | Episodes under 90 % of the command | Worst drive RMS / rating, worst episode (joint) | Episodes with a drive at or over 1.0 | Worst joint's mean RMS / rating | Heading change, worst episode | Sideways drift, worst episode | Slip ratio, worst foot's mean (at the sole origin / at the contacts) | Peak drive speed, rpm | Steps at the episode's own torque limit, worst joint-episode | Hind touchdowns per s, worst episode | PREREG_WALK60's thresholds (written for the 188 mm roll-ankle model) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| J1 grippy (friction 1.0), the model's contact | 0/10 | 0.246 (0.246) | 0 | 0.77 (left_fore_pitch) | 0 | 0.73 (right_hind_pitch) | 4.7 deg | 0.17 m | 0.12 / 0.06 | 7 (4) / 41 (35) / 39 (35) / 55 (41) | 3.1 % (right_fore_pitch) | 1.5 | W1 met (0 falls); W2 met (0.246, min 0.246); W3 met (0.77); W4 met (0.12) |
| J2 slippery (friction 0.4), the model's contact | 0/10 | 0.257 (0.253) | 0 | 0.83 (right_hind_flexion) | 0 | 0.75 (right_hind_flexion) | 5.5 deg | 0.15 m | 0.11 / 0.09 | 13 (6) / 39 (33) / 38 (34) / 50 (41) | 1.5 % (left_hind_flexion) | 2.9 | W1 met (0 falls); W2 met (0.257, min 0.253); W3 met (0.83); W4 met (0.11) |
| J3 grippy (friction 1.0), stiffer contact (impratio 10) | 0/10 | 0.253 (0.252) | 0 | 0.91 (left_fore_pitch) | 0 | 0.84 (left_fore_pitch) | 4.8 deg | 0.17 m | 0.11 / 0.04 | 7 (4) / 43 (34) / 39 (35) / 56 (41) | 3.3 % (right_fore_pitch) | 1.5 | W1 met (0 falls); W2 met (0.253, min 0.252); W3 met (0.91); W4 met (0.11) |
| J4 slippery (friction 0.4), stiffer contact (impratio 10) | 0/10 | 0.257 (0.253) | 0 | 0.81 (right_hind_flexion) | 0 | 0.74 (right_hind_flexion) | 5.6 deg | 0.15 m | 0.12 / 0.09 | 12 (7) / 40 (32) / 38 (34) / 50 (41) | 1.6 % (left_hind_flexion) | 3.2 | W1 met (0 falls); W2 met (0.257, min 0.253); W3 met (0.81); W4 met (0.12) |
| R1 J1 at 0 ms delay | 0/10 | 0.247 (0.229) | 0 | 0.84 (left_hind_flexion) | 0 | 0.70 (left_hind_pitch) | 5.4 deg | 0.18 m | 0.10 / not run | 10 (7) / 41 (35) / 46 (35) / 54 (41) | 3.4 % (right_hind_flexion) | 3.1 | orientation only: W1 met (0 falls); W2 met (0.247, min 0.229); W3 met (0.84); W4 met (0.10) |
| R2 J2 at 0 ms delay | 0/10 | 0.257 (0.255) | 0 | 0.82 (right_hind_flexion) | 0 | 0.73 (right_hind_flexion) | 5.4 deg | 0.17 m | 0.10 / not run | 9 (5) / 38 (34) / 38 (34) / 50 (41) | 1.5 % (left_hind_flexion) | 2.4 | orientation only: W1 met (0 falls); W2 met (0.257, min 0.255); W3 met (0.82); W4 met (0.10) |
| R3 J1 at the 0.15 m/s command | 0/10 | 0.139 (0.136) | 0 | 0.79 (right_fore_pitch) | 0 | 0.72 (right_fore_pitch) | 4.3 deg | 0.10 m | 0.18 / not run | 9 (6) / 32 (28) / 39 (33) / 48 (38) | 5.1 % (right_fore_pitch) | 1.6 | none (not at the 0.25 m/s command) |
| R4 J1 standing (0 m/s) | 0/10 | 0.000 (-0.000) |  | 0.52 (left_hind_flexion) | 0 | 0.49 (left_hind_flexion) | 0.0 deg | 0.00 m |  / not run | 0 (0) / 1 (1) / 3 (2) / 3 (2) | 0.0 % (right_hind_roll) | 0.0 | none (not at the 0.25 m/s command) |
| R5 nominal model, no randomisation or noise | 0/10 | 0.249 (0.249) | 0 | 0.74 (left_hind_pitch) | 0 | 0.73 (left_hind_pitch) | 0.9 deg | 0.02 m | 0.12 / not run | 4 (3) / 41 (33) / 38 (34) / 52 (33) | 2.3 % (right_fore_pitch) | 1.4 | orientation only: W1 met (0 falls); W2 met (0.249, min 0.249); W3 met (0.74); W4 met (0.12) |

### 10-20 s windows of the same episodes, seeds 201-210 (`capsule_judged_201-210.json`)

| Condition | Slow over 10-20 s / upright | Slow and hot (hind knee at or over 1.0) | 10-20 s speed, range, m/s | Larger hind knee RMS / rating over 10-20 s, range | Slow episodes: 10-20 s speed; hind knee; mean pitch (range), deg (+ nose down) | Whole-episode and 2-20 s speeds match the stored episodes |
|---|---|---|---|---|---|---|
| J1 | 0/10 | 0 | 0.246-0.248 | 0.61-0.68 |  | yes |
| J3 | 0/10 | 0 | 0.252-0.254 | 0.56-0.63 |  | yes |

