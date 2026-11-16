# Today's walk against servo delay (set V0, body 75-M-H-10)

**simulation; stand-in controller fitted with no delay; a delay that is not the judged 10 ms; no verdict; added after the 75 cm results.** Made by `stage2/runs/2026-11-16-75cm-addon/delays_table.py` from the committed records; the
words and patterns are those of `STAGE2_75_ADDON_PLAN.md` A-6, written before any add-on record existed.
Records at 14, 16 and 20 ms: (P37: 14, 16 and 20 ms). Records at 12 and 18 ms and the slow walk at 8 ms:
the lead's addition, not among the delays the owner approved. The slow walk and the `R0` rows at 14, 16 and
20 ms are the lead's reading of "today's walk". Records at 0 to 10 ms are the registered records of the
75 cm panels. 10 ms is the judged delay; nothing here changes its verdict.

A record is **up** when at least 17 of its 20 runs stay up, **down** when at most 3 do, otherwise **mixed**.
Each cell reads: runs that stayed up · of those, runs that also covered the distance · runs that met every
limit · runs that met every limit but for A1 (the neck and tail servo planning limit). All of 20.

## 1. The 0.40 m/s walk

| Delay | Records | Fit 42, grippy | Fit 43, grippy | Fit 42, slippery | Fit 43, slippery | No feedback (`R0`): up, grippy · slippery | The delay reads |
|---|---|---|---|---|---|---|---|
| 0 ms | registered | 20 · 20 · 0 · 19 | 20 · 20 · 0 · 19 | 20 · 20 · 0 · 19 | 20 · 20 · 0 · 20 | 20 · 20 | up |
| 2 ms | registered | 20 · 20 · 0 · 20 | 20 · 20 · 0 · 20 | 20 · 20 · 0 · 20 | 20 · 20 · 0 · 19 | 18 · 19 | up |
| 4 ms | registered | 20 · 20 · 0 · 16 | 20 · 20 · 0 · 12 | 20 · 20 · 0 · 11 | 20 · 20 · 0 · 11 | 19 · 20 | up |
| 6 ms | registered | 20 · 20 · 0 · 0 | 20 · 20 · 0 · 0 | 20 · 20 · 0 · 1 | 20 · 20 · 0 · 0 | 20 · 20 | up |
| 8 ms | registered | 0 · 0 · 0 · 0 | 0 · 0 · 0 · 0 | 0 · 0 · 0 · 0 | 0 · 0 · 0 · 0 | 19 · 20 | down |
| 10 ms | registered | 0 · 0 · 0 · 0 | 1 · 0 · 0 · 0 | 0 · 0 · 0 · 0 | 3 · 0 · 0 · 0 | 20 · 20 | down |
| 12 ms | add-on A2 | 0 · 0 · 0 · 0 | 0 · 0 · 0 · 0 | 0 · 0 · 0 · 0 | 0 · 0 · 0 · 0 | 19 · 19 | down |
| 14 ms | add-on A1 | 1 · 0 · 0 · 0 | 3 · 0 · 0 · 0 | 0 · 0 · 0 · 0 | 0 · 0 · 0 · 0 | 20 · 19 | down |
| 16 ms | add-on A1 | 0 · 0 · 0 · 0 | 2 · 0 · 0 · 0 | 0 · 0 · 0 · 0 | 0 · 0 · 0 · 0 | 18 · 20 | down |
| 18 ms | add-on A2 | 6 · 0 · 0 · 0 | 18 · 0 · 0 · 0 | 4 · 0 · 0 · 0 | 19 · 0 · 0 · 0 | 20 · 19 | mixed |
| 20 ms | add-on A1 | 1 · 0 · 0 · 0 | 1 · 0 · 0 · 0 | 1 · 0 · 0 · 0 | 1 · 0 · 0 · 0 | 20 · 19 | down |

Neither phrase for the 0.40 m/s walk: no delay above 10 ms reads up and at least one is mixed.

The runner's word for the two fits together (H2), per floor and reading:

| Delay | Grippy, every limit | Grippy, but for A1 | Slippery, every limit | Slippery, but for A1 |
|---|---|---|---|---|
| 0 ms | does not pass at 0 ms | passes at 0 ms | does not pass at 0 ms | passes at 0 ms |
| 2 ms | does not pass at 2 ms | passes at 2 ms | does not pass at 2 ms | passes at 2 ms |
| 4 ms | does not pass at 4 ms | does not pass at 4 ms | does not pass at 4 ms | does not pass at 4 ms |
| 6 ms | does not pass at 6 ms | does not pass at 6 ms | does not pass at 6 ms | does not pass at 6 ms |
| 8 ms | does not pass at 8 ms | does not pass at 8 ms | does not pass at 8 ms | does not pass at 8 ms |
| 10 ms | judged delay: see BAR_TABLES.md | judged delay: see BAR_TABLES.md | judged delay: see BAR_TABLES.md | judged delay: see BAR_TABLES.md |
| 12 ms | does not pass at 12 ms | does not pass at 12 ms | does not pass at 12 ms | does not pass at 12 ms |
| 14 ms | does not pass at 14 ms | does not pass at 14 ms | does not pass at 14 ms | does not pass at 14 ms |
| 16 ms | does not pass at 16 ms | does not pass at 16 ms | does not pass at 16 ms | does not pass at 16 ms |
| 18 ms | does not pass at 18 ms | does not pass at 18 ms | does not pass at 18 ms | does not pass at 18 ms |
| 20 ms | does not pass at 20 ms | does not pass at 20 ms | does not pass at 20 ms | does not pass at 20 ms |

## 2. The slow walk (its own reading; not merged with section 1)

| Delay | Records | Fit 42 | Fit 43 | No feedback (`R0`): up | The delay reads |
|---|---|---|---|---|---|
| 0 ms | registered | 20 · 20 · 20 · 20 (Stage 1 speed 20) | 20 · 20 · 20 · 20 (Stage 1 speed 20) | - | up |
| 2 ms | - | - | - | - | no word (no record) |
| 4 ms | registered | 20 · 20 · 0 · 0 (Stage 1 speed 20) | 20 · 20 · 0 · 0 (Stage 1 speed 20) | - | up |
| 6 ms | registered | 20 · 20 · 0 · 0 (Stage 1 speed 20) | 20 · 20 · 0 · 0 (Stage 1 speed 20) | - | up |
| 8 ms | add-on A3 | 19 · 19 · 0 · 0 (Stage 1 speed 19) | 19 · 19 · 0 · 0 (Stage 1 speed 19) | - | up |
| 10 ms | registered | 2 · 0 · 0 · 0 (Stage 1 speed 0) | 1 · 0 · 0 · 0 (Stage 1 speed 0) | - | down |
| 12 ms | add-on A2 | 0 · 0 · 0 · 0 (Stage 1 speed 0) | 0 · 0 · 0 · 0 (Stage 1 speed 0) | 20 | down |
| 14 ms | add-on A1 | 1 · 0 · 0 · 0 (Stage 1 speed 0) | 3 · 0 · 0 · 0 (Stage 1 speed 0) | 20 | down |
| 16 ms | add-on A1 | 0 · 0 · 0 · 0 (Stage 1 speed 0) | 3 · 2 · 0 · 0 (Stage 1 speed 2) | 20 | down |
| 18 ms | add-on A2 | 10 · 0 · 0 · 0 (Stage 1 speed 0) | 18 · 18 · 0 · 0 (Stage 1 speed 18) | 20 | mixed |
| 20 ms | add-on A1 | 19 · 0 · 0 · 0 (Stage 1 speed 0) | 20 · 0 · 0 · 0 (Stage 1 speed 0) | 20 | up |

A window for the slow walk: the lowest delay above 10 ms that reads up is 20 ms. 20 ms: a window of staying up, not of walking.

`R0` on the slow walk is the lead's addition; it is printed and raises no flag.

## 3. The mirror estimate (a design estimate, not a result)

Prediction: the 0.40 m/s walk reads down at 12 ms and up at 14, 16, 18 and 20 ms.

| Delay | Predicted | Read |
|---|---|---|
| 12 ms | down | down |
| 14 ms | up | down |
| 16 ms | up | down |
| 18 ms | up | mixed |
| 20 ms | up | down |

**The five predictions are not met** on the delays that were run: 14 ms, 16 ms, 18 ms, 20 ms differ. No other centre is fitted after the result.

## 4. Where the falls come

Classified by the runs' `segment_reached` alone. Fall times are printed and classify nothing.

| Delay | Walk | Fit and floor | Fell | Where (runs) | Median fall time |
|---|---|---|---|---|---|
| 8 ms | 0.40 m/s | 42, grippy floor | 20 | slow stride 1, change of speed 11, 0.40 m/s walk itself 8 | 12.0 s |
| 8 ms | 0.40 m/s | 43, grippy floor | 20 | slow stride 1, change of speed 4, 0.40 m/s walk itself 15 | 13.1 s |
| 8 ms | 0.40 m/s | 42, slippery floor | 20 | slow stride 3, change of speed 14, 0.40 m/s walk itself 3 | 11.2 s |
| 8 ms | 0.40 m/s | 43, slippery floor | 20 | change of speed 10, 0.40 m/s walk itself 10 | 12.1 s |
| 8 ms | slow | 42, grippy floor | 1 | slow stride 1 | 6.5 s |
| 8 ms | slow | 43, grippy floor | 1 | slow stride 1 | 6.0 s |
| 10 ms | 0.40 m/s | 42, grippy floor | 20 | slow stride 4, change of speed 13, start from standing 3 | 7.9 s |
| 10 ms | 0.40 m/s | 43, grippy floor | 19 | slow stride 4, change of speed 8, start from standing 7 | 6.5 s |
| 10 ms | 0.40 m/s | 42, slippery floor | 20 | slow stride 4, change of speed 10, start from standing 6 | 6.6 s |
| 10 ms | 0.40 m/s | 43, slippery floor | 17 | change of speed 4, start from standing 13 | 4.6 s |
| 10 ms | slow | 42, grippy floor | 18 | slow stride 15, start from standing 3 | 7.7 s |
| 10 ms | slow | 43, grippy floor | 19 | slow stride 12, start from standing 7 | 6.5 s |
| 12 ms | 0.40 m/s | 42, grippy floor | 20 | slow stride 5, start from standing 15 | 4.8 s |
| 12 ms | 0.40 m/s | 43, grippy floor | 20 | change of speed 1, start from standing 19 | 4.7 s |
| 12 ms | 0.40 m/s | 42, slippery floor | 20 | slow stride 4, start from standing 16 | 4.8 s |
| 12 ms | 0.40 m/s | 43, slippery floor | 20 | slow stride 1, start from standing 19 | 4.7 s |
| 12 ms | slow | 42, grippy floor | 20 | slow stride 5, start from standing 15 | 4.8 s |
| 12 ms | slow | 43, grippy floor | 20 | slow stride 1, start from standing 19 | 4.7 s |
| 14 ms | 0.40 m/s | 42, grippy floor | 19 | slow stride 4, start from standing 15 | 4.6 s |
| 14 ms | 0.40 m/s | 43, grippy floor | 17 | slow stride 3, change of speed 3, start from standing 11 | 5.2 s |
| 14 ms | 0.40 m/s | 42, slippery floor | 20 | slow stride 1, start from standing 19 | 4.7 s |
| 14 ms | 0.40 m/s | 43, slippery floor | 20 | slow stride 2, change of speed 5, start from standing 13 | 5.2 s |
| 14 ms | slow | 42, grippy floor | 19 | slow stride 4, start from standing 15 | 4.6 s |
| 14 ms | slow | 43, grippy floor | 17 | slow stride 6, start from standing 11 | 5.2 s |
| 16 ms | 0.40 m/s | 42, grippy floor | 20 | start from standing 20 | 4.2 s |
| 16 ms | 0.40 m/s | 43, grippy floor | 18 | slow stride 3, change of speed 4, start from standing 11 | 4.7 s |
| 16 ms | 0.40 m/s | 42, slippery floor | 20 | slow stride 1, start from standing 19 | 4.2 s |
| 16 ms | 0.40 m/s | 43, slippery floor | 20 | slow stride 3, change of speed 4, start from standing 13 | 4.6 s |
| 16 ms | slow | 42, grippy floor | 20 | start from standing 20 | 4.2 s |
| 16 ms | slow | 43, grippy floor | 17 | slow stride 6, start from standing 11 | 4.7 s |
| 18 ms | 0.40 m/s | 42, grippy floor | 14 | change of speed 6, start from standing 8 | 4.1 s |
| 18 ms | 0.40 m/s | 43, grippy floor | 2 | start from standing 2 | 4.0 s |
| 18 ms | 0.40 m/s | 42, slippery floor | 16 | change of speed 1, start from standing 15 | 3.8 s |
| 18 ms | 0.40 m/s | 43, slippery floor | 1 | change of speed 1 | 11.1 s |
| 18 ms | slow | 42, grippy floor | 10 | slow stride 2, start from standing 8 | 3.9 s |
| 18 ms | slow | 43, grippy floor | 2 | start from standing 2 | 4.0 s |
| 20 ms | 0.40 m/s | 42, grippy floor | 19 | change of speed 1, start from standing 1, 0.40 m/s walk itself 17 | 14.4 s |
| 20 ms | 0.40 m/s | 43, grippy floor | 19 | change of speed 2, 0.40 m/s walk itself 17 | 15.7 s |
| 20 ms | 0.40 m/s | 42, slippery floor | 19 | change of speed 3, 0.40 m/s walk itself 16 | 14.6 s |
| 20 ms | 0.40 m/s | 43, slippery floor | 19 | change of speed 2, 0.40 m/s walk itself 17 | 14.4 s |
| 20 ms | slow | 42, grippy floor | 1 | start from standing 1 | 3.6 s |

## 5. Of the runs that stayed up (not judged)

| Delay | Walk | Row and floor | Up | Median speed | Median toe clearance (limit) | Failed limits (runs) | Largest drive speed ratio | Median peak ground force |
|---|---|---|---|---|---|---|---|---|
| 0 ms | 0.40 m/s | R2/42, grippy floor | 20 | 0.410 m/s | 24.2 mm (20.1) | drive_sustained 20, line 1 | 0.75 (left_knee_servo) | - |
| 0 ms | 0.40 m/s | R2/43, grippy floor | 20 | 0.411 m/s | 24.2 mm (20.1) | drive_sustained 20, line 1 | 0.75 (left_knee_servo) | - |
| 0 ms | 0.40 m/s | R0, grippy floor | 20 | 0.410 m/s | 24.5 mm (20.1) | drive_sustained 20, duty 20, along 1, no_flight 1, toe_clearance_own_swing 1 | 0.74 (left_knee_servo) | - |
| 0 ms | 0.40 m/s | R2/42, slippery floor | 20 | 0.409 m/s | 24.4 mm (20.1) | drive_sustained 20, line 1 | 0.75 (right_knee_servo) | - |
| 0 ms | 0.40 m/s | R2/43, slippery floor | 20 | 0.410 m/s | 24.3 mm (20.1) | drive_sustained 20 | 0.75 (left_knee_servo) | - |
| 0 ms | 0.40 m/s | R0, slippery floor | 20 | 0.410 m/s | 26.0 mm (20.1) | drive_sustained 20, duty 20 | 0.74 (left_knee_servo) | - |
| 0 ms | slow | R2/42, grippy floor | 20 | 0.250 m/s | 22.1 mm (20.1) | none | 0.53 (left_knee_servo) | 1.23 body weights |
| 0 ms | slow | R2/43, grippy floor | 20 | 0.250 m/s | 22.2 mm (20.1) | none | 0.52 (left_knee_servo) | 1.25 body weights |
| 2 ms | 0.40 m/s | R2/42, grippy floor | 20 | 0.412 m/s | 22.9 mm (20.1) | drive_sustained 20 | 0.76 (right_knee_servo) | - |
| 2 ms | 0.40 m/s | R2/43, grippy floor | 20 | 0.413 m/s | 23.1 mm (20.1) | drive_sustained 20 | 0.75 (right_knee_servo) | - |
| 2 ms | 0.40 m/s | R0, grippy floor | 18 | 0.410 m/s | 25.5 mm (20.1) | drive_sustained 18, duty 18, line 1 | 0.74 (left_knee_servo) | - |
| 2 ms | 0.40 m/s | R2/42, slippery floor | 20 | 0.412 m/s | 23.1 mm (20.1) | drive_sustained 20 | 0.76 (right_knee_servo) | - |
| 2 ms | 0.40 m/s | R2/43, slippery floor | 20 | 0.413 m/s | 23.2 mm (20.1) | drive_sustained 20, line 1 | 0.75 (left_knee_servo) | - |
| 2 ms | 0.40 m/s | R0, slippery floor | 19 | 0.410 m/s | 25.9 mm (20.1) | drive_sustained 19, duty 19, line 1, no_flight 1, toe_clearance_own_swing 1 | 0.74 (left_knee_servo) | - |
| 4 ms | 0.40 m/s | R2/42, grippy floor | 20 | 0.412 m/s | 20.7 mm (20.1) | drive_sustained 20, toe_clearance_own_swing 4 | 0.76 (left_knee_servo) | - |
| 4 ms | 0.40 m/s | R2/43, grippy floor | 20 | 0.414 m/s | 20.2 mm (20.1) | drive_sustained 20, toe_clearance_own_swing 7, line 1 | 0.75 (right_knee_servo) | - |
| 4 ms | 0.40 m/s | R0, grippy floor | 19 | 0.410 m/s | 24.9 mm (20.1) | drive_sustained 19, duty 19, no_flight 3, toe_clearance_own_swing 3, along 1 | 0.74 (left_knee_servo) | - |
| 4 ms | 0.40 m/s | R2/42, slippery floor | 20 | 0.412 m/s | 20.2 mm (20.1) | drive_sustained 20, toe_clearance_own_swing 9 | 0.76 (left_knee_servo) | - |
| 4 ms | 0.40 m/s | R2/43, slippery floor | 20 | 0.415 m/s | 20.1 mm (20.1) | drive_sustained 20, toe_clearance_own_swing 9 | 0.75 (right_knee_servo) | - |
| 4 ms | 0.40 m/s | R0, slippery floor | 20 | 0.410 m/s | 26.0 mm (20.1) | drive_sustained 20, duty 20 | 0.74 (left_knee_servo) | - |
| 4 ms | slow | R2/42, grippy floor | 20 | 0.252 m/s | 18.0 mm (20.1) | toe_clearance_own_swing 20 | 0.53 (left_knee_servo) | 1.32 body weights |
| 4 ms | slow | R2/43, grippy floor | 20 | 0.252 m/s | 17.9 mm (20.1) | toe_clearance_own_swing 20 | 0.52 (left_knee_servo) | 1.30 body weights |
| 6 ms | 0.40 m/s | R2/42, grippy floor | 20 | 0.409 m/s | 18.9 mm (20.1) | drive_sustained 20, toe_clearance_own_swing 20 | 0.76 (left_knee_servo) | - |
| 6 ms | 0.40 m/s | R2/43, grippy floor | 20 | 0.413 m/s | 18.3 mm (20.1) | drive_sustained 20, toe_clearance_own_swing 20 | 0.75 (right_knee_servo) | - |
| 6 ms | 0.40 m/s | R0, grippy floor | 20 | 0.410 m/s | 26.1 mm (20.1) | drive_sustained 20, duty 20 | 0.74 (left_knee_servo) | - |
| 6 ms | 0.40 m/s | R2/42, slippery floor | 20 | 0.409 m/s | 19.0 mm (20.1) | drive_sustained 20, toe_clearance_own_swing 19 | 0.75 (left_knee_servo) | - |
| 6 ms | 0.40 m/s | R2/43, slippery floor | 20 | 0.414 m/s | 18.3 mm (20.1) | drive_sustained 20, toe_clearance_own_swing 20 | 0.75 (left_knee_servo) | - |
| 6 ms | 0.40 m/s | R0, slippery floor | 20 | 0.410 m/s | 25.7 mm (20.1) | drive_sustained 20, duty 20, line 1 | 0.74 (left_knee_servo) | - |
| 6 ms | slow | R2/42, grippy floor | 20 | 0.250 m/s | 15.4 mm (20.1) | toe_clearance_own_swing 20 | 0.53 (right_ankle_b_servo) | 1.34 body weights |
| 6 ms | slow | R2/43, grippy floor | 20 | 0.251 m/s | 14.9 mm (20.1) | toe_clearance_own_swing 20 | 0.53 (right_ankle_b_servo) | 1.36 body weights |
| 8 ms | 0.40 m/s | R0, grippy floor | 19 | 0.410 m/s | 26.3 mm (20.1) | drive_sustained 19, duty 19, line 2 | 0.74 (left_knee_servo) | - |
| 8 ms | 0.40 m/s | R0, slippery floor | 20 | 0.410 m/s | 25.7 mm (20.1) | drive_sustained 20, duty 20, line 1 | 0.74 (left_knee_servo) | - |
| 8 ms | slow | R2/42, grippy floor | 19 | 0.248 m/s | 12.4 mm (20.1) | toe_clearance_own_swing 19, grf 3, line 1, stride 1 | 0.79 (right_ankle_a_servo) | 1.55 body weights |
| 8 ms | slow | R2/43, grippy floor | 19 | 0.250 m/s | 11.3 mm (20.1) | toe_clearance_own_swing 19, grf 2 | 0.68 (right_ankle_b_servo) | 1.47 body weights |
| 10 ms | 0.40 m/s | R2/43, grippy floor | 1 | 0.391 m/s | 4.3 mm (20.1) | along 1, drive_speed 1, drive_sustained 1, duty 1, toe_clearance_own_swing 1 | 1.35 (right_ankle_b_servo) | - |
| 10 ms | 0.40 m/s | R0, grippy floor | 20 | 0.410 m/s | 25.5 mm (20.1) | drive_sustained 20, duty 20, line 1, toe_clearance_own_swing 1 | 0.74 (left_knee_servo) | - |
| 10 ms | 0.40 m/s | R2/43, slippery floor | 3 | 0.387 m/s | 3.7 mm (20.1) | along 3, drive_speed 3, drive_sustained 3, duty 3, slip_median 3, toe_clearance_own_swing 3, line 1 | 1.36 (left_ankle_a_servo) | - |
| 10 ms | 0.40 m/s | R0, slippery floor | 20 | 0.410 m/s | 25.9 mm (20.1) | drive_sustained 20, duty 20 | 0.74 (left_knee_servo) | - |
| 10 ms | slow | R2/42, grippy floor | 2 | 0.238 m/s | 1.1 mm (20.1) | along 2, cadence 2, grf 2, stride 2, toe_clearance_own_swing 2, drive_speed 1, line 1 | 1.25 (left_ankle_a_servo) | 2.19 body weights |
| 10 ms | slow | R2/43, grippy floor | 1 | 0.232 m/s | 3.0 mm (20.1) | along 1, drive_speed 1, duty 1, grf 1, slip_median 1, toe_clearance_own_swing 1 | 1.26 (left_ankle_a_servo) | 3.17 body weights |
| 12 ms | 0.40 m/s | R0, grippy floor | 19 | 0.410 m/s | 25.2 mm (20.1) | drive_sustained 19, duty 19, line 1 | 0.74 (left_knee_servo) | - |
| 12 ms | 0.40 m/s | R0, slippery floor | 19 | 0.410 m/s | 25.8 mm (20.1) | drive_sustained 19, duty 19, line 1 | 0.74 (left_knee_servo) | - |
| 12 ms | slow | R0, grippy floor | 20 | 0.249 m/s | 23.6 mm (20.1) | none | 0.52 (left_knee_servo) | 1.28 body weights |
| 14 ms | 0.40 m/s | R2/42, grippy floor | 1 | 0.374 m/s | 4.3 mm (20.1) | along 1, drive_speed 1, drive_sustained 1, duty 1, line 1, toe_clearance_own_swing 1 | 1.27 (right_ankle_a_servo) | - |
| 14 ms | 0.40 m/s | R2/43, grippy floor | 3 | 0.389 m/s | 2.1 mm (20.1) | along 3, drive_speed 3, drive_sustained 3, duty 3, toe_clearance_own_swing 3, line 2 | 1.63 (right_ankle_a_servo) | - |
| 14 ms | 0.40 m/s | R0, grippy floor | 20 | 0.409 m/s | 24.9 mm (20.1) | drive_sustained 20, duty 20, line 2, toe_clearance_own_swing 2, no_flight 1 | 0.74 (left_knee_servo) | - |
| 14 ms | 0.40 m/s | R0, slippery floor | 19 | 0.410 m/s | 26.0 mm (20.1) | drive_sustained 19, duty 19 | 0.74 (left_knee_servo) | - |
| 14 ms | slow | R2/42, grippy floor | 1 | 0.225 m/s | 4.8 mm (20.1) | along 1, drive_speed 1, duty 1, grf 1, line 1, slip_median 1, toe_clearance_own_swing 1 | 1.28 (right_ankle_b_servo) | 2.88 body weights |
| 14 ms | slow | R2/43, grippy floor | 3 | 0.229 m/s | 3.2 mm (20.1) | along 3, drive_speed 3, duty 3, grf 3, slip_median 3, toe_clearance_own_swing 3, line 2, stride 2 | 1.63 (right_ankle_a_servo) | 2.87 body weights |
| 14 ms | slow | R0, grippy floor | 20 | 0.249 m/s | 23.7 mm (20.1) | none | 0.52 (left_knee_servo) | 1.28 body weights |
| 16 ms | 0.40 m/s | R2/43, grippy floor | 2 | 0.389 m/s | 1.4 mm (20.1) | along 2, drive_speed 2, drive_sustained 2, duty 2, toe_clearance_own_swing 2 | 1.26 (left_ankle_a_servo) | - |
| 16 ms | 0.40 m/s | R0, grippy floor | 18 | 0.409 m/s | 25.0 mm (20.1) | drive_sustained 18, duty 18 | 0.74 (left_knee_servo) | - |
| 16 ms | 0.40 m/s | R0, slippery floor | 20 | 0.410 m/s | 25.8 mm (20.1) | drive_sustained 20, duty 20 | 0.74 (left_knee_servo) | - |
| 16 ms | slow | R2/43, grippy floor | 3 | 0.245 m/s | 6.8 mm (20.1) | grf 3, toe_clearance_own_swing 3, drive_speed 2, stride 2, along 1, duty 1 | 1.85 (left_ankle_b_servo) | 2.26 body weights |
| 16 ms | slow | R0, grippy floor | 20 | 0.249 m/s | 23.8 mm (20.1) | none | 0.52 (left_knee_servo) | 1.28 body weights |
| 18 ms | 0.40 m/s | R2/42, grippy floor | 6 | 0.384 m/s | 2.8 mm (20.1) | along 6, drive_speed 6, drive_sustained 6, toe_clearance_own_swing 6, duty 3, line 1 | 1.78 (left_ankle_a_servo) | - |
| 18 ms | 0.40 m/s | R2/43, grippy floor | 18 | 0.390 m/s | 0.6 mm (20.1) | along 18, drive_speed 18, drive_sustained 18, toe_clearance_own_swing 18, duty 17, no_flight 3, line 1 | 1.90 (left_ankle_a_servo) | - |
| 18 ms | 0.40 m/s | R0, grippy floor | 20 | 0.409 m/s | 25.5 mm (20.1) | drive_sustained 20, duty 20 | 0.74 (left_knee_servo) | - |
| 18 ms | 0.40 m/s | R2/42, slippery floor | 4 | 0.379 m/s | 3.9 mm (20.1) | along 4, drive_speed 4, drive_sustained 4, toe_clearance_own_swing 4, duty 3, line 2 | 1.37 (right_ankle_b_servo) | - |
| 18 ms | 0.40 m/s | R2/43, slippery floor | 19 | 0.387 m/s | 0.7 mm (20.1) | along 19, drive_speed 19, drive_sustained 19, duty 19, toe_clearance_own_swing 19, line 4, no_flight 4 | 1.95 (left_ankle_a_servo) | - |
| 18 ms | 0.40 m/s | R0, slippery floor | 19 | 0.409 m/s | 26.1 mm (20.1) | drive_sustained 19, duty 19, line 1 | 0.74 (left_knee_servo) | - |
| 18 ms | slow | R2/42, grippy floor | 10 | 0.239 m/s | -2.0 mm (20.1) | along 10, drive_speed 10, grf 10, stride 10, toe_clearance_own_swing 10, cadence 8, no_flight 4, duty 2, pitch_peak 1, roll_peak 1, slip_median 1 | 1.69 (left_ankle_a_servo) | 2.71 body weights |
| 18 ms | slow | R2/43, grippy floor | 18 | 0.245 m/s | 14.4 mm (20.1) | cadence 18, grf 18, stride 18, toe_clearance_own_swing 18, drive_speed 4 | 1.34 (left_ankle_b_servo) | 1.90 body weights |
| 18 ms | slow | R0, grippy floor | 20 | 0.249 m/s | 23.6 mm (20.1) | none | 0.52 (left_knee_servo) | 1.28 body weights |
| 20 ms | 0.40 m/s | R2/42, grippy floor | 1 | 0.374 m/s | -3.2 mm (20.1) | along 1, drive_speed 1, drive_sustained 1, duty 1, line 1, no_flight 1, toe_clearance_own_swing 1 | 2.06 (right_ankle_b_servo) | - |
| 20 ms | 0.40 m/s | R2/43, grippy floor | 1 | 0.388 m/s | -1.8 mm (20.1) | along 1, drive_speed 1, drive_sustained 1, duty 1, toe_clearance_own_swing 1 | 1.69 (left_ankle_a_servo) | - |
| 20 ms | 0.40 m/s | R0, grippy floor | 20 | 0.409 m/s | 25.5 mm (20.1) | drive_sustained 20, duty 20 | 0.74 (left_knee_servo) | - |
| 20 ms | 0.40 m/s | R2/42, slippery floor | 1 | 0.346 m/s | -3.7 mm (20.1) | along 1, drive_speed 1, drive_sustained 1, duty 1, line 1, no_flight 1, pitch_peak 1, roll_peak 1, roll_rms 1, slip_worst 1, toe_clearance_own_swing 1 | 2.17 (right_ankle_b_servo) | - |
| 20 ms | 0.40 m/s | R2/43, slippery floor | 1 | 0.382 m/s | -2.9 mm (20.1) | along 1, drive_speed 1, drive_sustained 1, duty 1, toe_clearance_own_swing 1 | 2.05 (left_ankle_a_servo) | - |
| 20 ms | 0.40 m/s | R0, slippery floor | 19 | 0.410 m/s | 25.7 mm (20.1) | drive_sustained 19, duty 19, along 1, line 1, no_flight 1, toe_clearance_own_swing 1 | 0.74 (left_knee_servo) | - |
| 20 ms | slow | R2/42, grippy floor | 19 | 0.240 m/s | 13.2 mm (20.1) | along 19, drive_speed 19, grf 19, stride 19, toe_clearance_own_swing 19, cadence 18, duty 1, slip_median 1 | 1.98 (left_ankle_b_servo) | 2.38 body weights |
| 20 ms | slow | R2/43, grippy floor | 20 | 0.241 m/s | 12.5 mm (20.1) | along 20, drive_speed 20, grf 20, stride 20, toe_clearance_own_swing 20, cadence 18, duty 2, slip_median 2, line 1 | 1.99 (left_ankle_b_servo) | 2.04 body weights |
| 20 ms | slow | R0, grippy floor | 20 | 0.249 m/s | 23.7 mm (20.1) | none | 0.52 (left_knee_servo) | 1.28 body weights |

## 6. The expectations of A-6, beside the result

1. `R0` on the 0.40 m/s walk, expected 18 to 20 of 20 up at every delay: read 18 to 20. Flags (an `R0` record under 17): none. Every read record holds `latency_steps` = ms / 2: yes.
2. The slow walk under fit 43 at 20 ms, expected up: read up.
3. No expectation was stated for fit 42 at 20 ms, for 12 to 18 ms, or for the 0.40 m/s walk at 20 ms.
4. No add-on record was expected to meet every limit in 17 of 20: 5 do: stage2/runs/2026-11-16-75cm-addon/A/V0/75-M-H-10/seeds3042/eval/R0@12ms/walk/nominal.json, stage2/runs/2026-11-16-75cm-addon/A/V0/75-M-H-10/seeds3042/eval/R0@14ms/walk/nominal.json, stage2/runs/2026-11-16-75cm-addon/A/V0/75-M-H-10/seeds3042/eval/R0@16ms/walk/nominal.json, stage2/runs/2026-11-16-75cm-addon/A/V0/75-M-H-10/seeds3042/eval/R0@18ms/walk/nominal.json, stage2/runs/2026-11-16-75cm-addon/A/V0/75-M-H-10/seeds3042/eval/R0@20ms/walk/nominal.json.

## 7. The records

| Record | From | sha256 (first 12) |
|---|---|---|
| `stage2/runs/2026-11-15-75cm/V0/75-M-H-10/seeds3042/eval/R2/42@0ms/walk/nominal.json` | registered | `3d7b62aca965` |
| `stage2/runs/2026-11-15-75cm/V0/75-M-H-10/seeds3042/eval/R2/43@0ms/walk/nominal.json` | registered | `c59c10feb8ec` |
| `stage2/runs/2026-11-15-75cm/V0/75-M-H-10/seeds3042/eval/R0@0ms/walk/v0.4.json` | registered | `f5fcfa80df7b` |
| `stage2/runs/2026-11-15-75cm/V0/75-M-H-10/seeds3042/eval/R2/42@0ms/walk/v0.4.json` | registered | `779cef59a069` |
| `stage2/runs/2026-11-15-75cm/V0/75-M-H-10/seeds3042/eval/R2/43@0ms/walk/v0.4.json` | registered | `019b73464ae1` |
| `stage2/runs/2026-11-15-75cm/V0/75-M-H-10/seeds3042/eval/R0@0ms/walk/v0.4_mu0.54.json` | registered | `544a255a8b7d` |
| `stage2/runs/2026-11-15-75cm/V0/75-M-H-10/seeds3042/eval/R2/42@0ms/walk/v0.4_mu0.54.json` | registered | `ff89e5ae4228` |
| `stage2/runs/2026-11-15-75cm/V0/75-M-H-10/seeds3042/eval/R2/43@0ms/walk/v0.4_mu0.54.json` | registered | `fc0fad2857c8` |
| `stage2/runs/2026-11-15-75cm/V0/75-M-H-10/seeds3042/eval/R0@2ms/walk/v0.4.json` | registered | `f8efe06b9207` |
| `stage2/runs/2026-11-15-75cm/V0/75-M-H-10/seeds3042/eval/R2/42@2ms/walk/v0.4.json` | registered | `cc7093778e10` |
| `stage2/runs/2026-11-15-75cm/V0/75-M-H-10/seeds3042/eval/R2/43@2ms/walk/v0.4.json` | registered | `4fae530fddc8` |
| `stage2/runs/2026-11-15-75cm/V0/75-M-H-10/seeds3042/eval/R0@2ms/walk/v0.4_mu0.54.json` | registered | `2fbdc85c387c` |
| `stage2/runs/2026-11-15-75cm/V0/75-M-H-10/seeds3042/eval/R2/42@2ms/walk/v0.4_mu0.54.json` | registered | `99aee899e5e7` |
| `stage2/runs/2026-11-15-75cm/V0/75-M-H-10/seeds3042/eval/R2/43@2ms/walk/v0.4_mu0.54.json` | registered | `a7f40e8ff6e1` |
| `stage2/runs/2026-11-15-75cm/V0/75-M-H-10/seeds3042/eval/R2/42@4ms/walk/nominal.json` | registered | `f9c134b582bc` |
| `stage2/runs/2026-11-15-75cm/V0/75-M-H-10/seeds3042/eval/R2/43@4ms/walk/nominal.json` | registered | `cfa97596be2b` |
| `stage2/runs/2026-11-15-75cm/V0/75-M-H-10/seeds3042/eval/R0@4ms/walk/v0.4.json` | registered | `ea2b0790ee94` |
| `stage2/runs/2026-11-15-75cm/V0/75-M-H-10/seeds3042/eval/R2/42@4ms/walk/v0.4.json` | registered | `a9ddb697cca8` |
| `stage2/runs/2026-11-15-75cm/V0/75-M-H-10/seeds3042/eval/R2/43@4ms/walk/v0.4.json` | registered | `e0ec68e78e78` |
| `stage2/runs/2026-11-15-75cm/V0/75-M-H-10/seeds3042/eval/R0@4ms/walk/v0.4_mu0.54.json` | registered | `e23a5b04fdb8` |
| `stage2/runs/2026-11-15-75cm/V0/75-M-H-10/seeds3042/eval/R2/42@4ms/walk/v0.4_mu0.54.json` | registered | `2debf629baf5` |
| `stage2/runs/2026-11-15-75cm/V0/75-M-H-10/seeds3042/eval/R2/43@4ms/walk/v0.4_mu0.54.json` | registered | `42199b9eb50d` |
| `stage2/runs/2026-11-15-75cm/V0/75-M-H-10/seeds3042/eval/R2/42@6ms/walk/nominal.json` | registered | `1fb49bd93fdf` |
| `stage2/runs/2026-11-15-75cm/V0/75-M-H-10/seeds3042/eval/R2/43@6ms/walk/nominal.json` | registered | `0028fbcfd832` |
| `stage2/runs/2026-11-15-75cm/V0/75-M-H-10/seeds3042/eval/R0@6ms/walk/v0.4.json` | registered | `7a38b63331b7` |
| `stage2/runs/2026-11-15-75cm/V0/75-M-H-10/seeds3042/eval/R2/42@6ms/walk/v0.4.json` | registered | `57e86db3c002` |
| `stage2/runs/2026-11-15-75cm/V0/75-M-H-10/seeds3042/eval/R2/43@6ms/walk/v0.4.json` | registered | `4873b0b836b9` |
| `stage2/runs/2026-11-15-75cm/V0/75-M-H-10/seeds3042/eval/R0@6ms/walk/v0.4_mu0.54.json` | registered | `9e8e86e5dfbf` |
| `stage2/runs/2026-11-15-75cm/V0/75-M-H-10/seeds3042/eval/R2/42@6ms/walk/v0.4_mu0.54.json` | registered | `5dfa957eb5e8` |
| `stage2/runs/2026-11-15-75cm/V0/75-M-H-10/seeds3042/eval/R2/43@6ms/walk/v0.4_mu0.54.json` | registered | `22e9fd97d49d` |
| `stage2/runs/2026-11-16-75cm-addon/A/V0/75-M-H-10/seeds3042/eval/R2/42@8ms/walk/nominal.json` | add-on A3: the lead's addition, not among the delays the owner approved | `2557d1b2533d` |
| `stage2/runs/2026-11-16-75cm-addon/A/V0/75-M-H-10/seeds3042/eval/R2/43@8ms/walk/nominal.json` | add-on A3: the lead's addition, not among the delays the owner approved | `e680c3e34533` |
| `stage2/runs/2026-11-15-75cm/V0/75-M-H-10/seeds3042/eval/R0@8ms/walk/v0.4.json` | registered | `16e0db9b4af2` |
| `stage2/runs/2026-11-15-75cm/V0/75-M-H-10/seeds3042/eval/R2/42@8ms/walk/v0.4.json` | registered | `eda132e53021` |
| `stage2/runs/2026-11-15-75cm/V0/75-M-H-10/seeds3042/eval/R2/43@8ms/walk/v0.4.json` | registered | `37fb9ca583ba` |
| `stage2/runs/2026-11-15-75cm/V0/75-M-H-10/seeds3042/eval/R0@8ms/walk/v0.4_mu0.54.json` | registered | `b93e7d9e7b40` |
| `stage2/runs/2026-11-15-75cm/V0/75-M-H-10/seeds3042/eval/R2/42@8ms/walk/v0.4_mu0.54.json` | registered | `02b678c8f6d1` |
| `stage2/runs/2026-11-15-75cm/V0/75-M-H-10/seeds3042/eval/R2/43@8ms/walk/v0.4_mu0.54.json` | registered | `81c29ffde9a9` |
| `stage2/runs/2026-11-15-75cm/V0/75-M-H-10/seeds3042/eval/R2/42@10ms/walk/nominal.json` | registered | `6d1e503687f0` |
| `stage2/runs/2026-11-15-75cm/V0/75-M-H-10/seeds3042/eval/R2/43@10ms/walk/nominal.json` | registered | `a168f7ec8b11` |
| `stage2/runs/2026-11-15-75cm/V0/75-M-H-10/seeds3042/eval/R0@10ms/walk/v0.4.json` | registered | `3456b3007b3e` |
| `stage2/runs/2026-11-15-75cm/V0/75-M-H-10/seeds3042/eval/R2/42@10ms/walk/v0.4.json` | registered | `ebb76002e352` |
| `stage2/runs/2026-11-15-75cm/V0/75-M-H-10/seeds3042/eval/R2/43@10ms/walk/v0.4.json` | registered | `7dbe1f9edd80` |
| `stage2/runs/2026-11-15-75cm/V0/75-M-H-10/seeds3042/eval/R0@10ms/walk/v0.4_mu0.54.json` | registered | `74639eb1eb04` |
| `stage2/runs/2026-11-15-75cm/V0/75-M-H-10/seeds3042/eval/R2/42@10ms/walk/v0.4_mu0.54.json` | registered | `ddc6833725c1` |
| `stage2/runs/2026-11-15-75cm/V0/75-M-H-10/seeds3042/eval/R2/43@10ms/walk/v0.4_mu0.54.json` | registered | `bb155383f3b4` |
| `stage2/runs/2026-11-16-75cm-addon/A/V0/75-M-H-10/seeds3042/eval/R0@12ms/walk/nominal.json` | add-on A2: the lead's addition, not among the delays the owner approved | `bb77c1010095` |
| `stage2/runs/2026-11-16-75cm-addon/A/V0/75-M-H-10/seeds3042/eval/R2/42@12ms/walk/nominal.json` | add-on A2: the lead's addition, not among the delays the owner approved | `82ac4e7bb175` |
| `stage2/runs/2026-11-16-75cm-addon/A/V0/75-M-H-10/seeds3042/eval/R2/43@12ms/walk/nominal.json` | add-on A2: the lead's addition, not among the delays the owner approved | `25a544b74f32` |
| `stage2/runs/2026-11-16-75cm-addon/A/V0/75-M-H-10/seeds3042/eval/R0@12ms/walk/v0.4.json` | add-on A2: the lead's addition, not among the delays the owner approved | `9ccf1dd49d43` |
| `stage2/runs/2026-11-16-75cm-addon/A/V0/75-M-H-10/seeds3042/eval/R2/42@12ms/walk/v0.4.json` | add-on A2: the lead's addition, not among the delays the owner approved | `8702f689b172` |
| `stage2/runs/2026-11-16-75cm-addon/A/V0/75-M-H-10/seeds3042/eval/R2/43@12ms/walk/v0.4.json` | add-on A2: the lead's addition, not among the delays the owner approved | `5766aeba4bf8` |
| `stage2/runs/2026-11-16-75cm-addon/A/V0/75-M-H-10/seeds3042/eval/R0@12ms/walk/v0.4_mu0.54.json` | add-on A2: the lead's addition, not among the delays the owner approved | `f3528ce86e0f` |
| `stage2/runs/2026-11-16-75cm-addon/A/V0/75-M-H-10/seeds3042/eval/R2/42@12ms/walk/v0.4_mu0.54.json` | add-on A2: the lead's addition, not among the delays the owner approved | `f7ad508eddd0` |
| `stage2/runs/2026-11-16-75cm-addon/A/V0/75-M-H-10/seeds3042/eval/R2/43@12ms/walk/v0.4_mu0.54.json` | add-on A2: the lead's addition, not among the delays the owner approved | `4a47d422cb40` |
| `stage2/runs/2026-11-16-75cm-addon/A/V0/75-M-H-10/seeds3042/eval/R0@14ms/walk/nominal.json` | add-on A1: (P37: 14, 16 and 20 ms) | `dabd2a780546` |
| `stage2/runs/2026-11-16-75cm-addon/A/V0/75-M-H-10/seeds3042/eval/R2/42@14ms/walk/nominal.json` | add-on A1: (P37: 14, 16 and 20 ms) | `46d3f5d0d083` |
| `stage2/runs/2026-11-16-75cm-addon/A/V0/75-M-H-10/seeds3042/eval/R2/43@14ms/walk/nominal.json` | add-on A1: (P37: 14, 16 and 20 ms) | `0c1b813cf2f8` |
| `stage2/runs/2026-11-16-75cm-addon/A/V0/75-M-H-10/seeds3042/eval/R0@14ms/walk/v0.4.json` | add-on A1: (P37: 14, 16 and 20 ms) | `01b22cee73af` |
| `stage2/runs/2026-11-16-75cm-addon/A/V0/75-M-H-10/seeds3042/eval/R2/42@14ms/walk/v0.4.json` | add-on A1: (P37: 14, 16 and 20 ms) | `1c1487ffdf27` |
| `stage2/runs/2026-11-16-75cm-addon/A/V0/75-M-H-10/seeds3042/eval/R2/43@14ms/walk/v0.4.json` | add-on A1: (P37: 14, 16 and 20 ms) | `a4f3b6269704` |
| `stage2/runs/2026-11-16-75cm-addon/A/V0/75-M-H-10/seeds3042/eval/R0@14ms/walk/v0.4_mu0.54.json` | add-on A1: (P37: 14, 16 and 20 ms) | `a262735346fd` |
| `stage2/runs/2026-11-16-75cm-addon/A/V0/75-M-H-10/seeds3042/eval/R2/42@14ms/walk/v0.4_mu0.54.json` | add-on A1: (P37: 14, 16 and 20 ms) | `ec451d77049e` |
| `stage2/runs/2026-11-16-75cm-addon/A/V0/75-M-H-10/seeds3042/eval/R2/43@14ms/walk/v0.4_mu0.54.json` | add-on A1: (P37: 14, 16 and 20 ms) | `8c080acfc5a4` |
| `stage2/runs/2026-11-16-75cm-addon/A/V0/75-M-H-10/seeds3042/eval/R0@16ms/walk/nominal.json` | add-on A1: (P37: 14, 16 and 20 ms) | `ca390a1a5a90` |
| `stage2/runs/2026-11-16-75cm-addon/A/V0/75-M-H-10/seeds3042/eval/R2/42@16ms/walk/nominal.json` | add-on A1: (P37: 14, 16 and 20 ms) | `a1d649388540` |
| `stage2/runs/2026-11-16-75cm-addon/A/V0/75-M-H-10/seeds3042/eval/R2/43@16ms/walk/nominal.json` | add-on A1: (P37: 14, 16 and 20 ms) | `2a4f64ddef43` |
| `stage2/runs/2026-11-16-75cm-addon/A/V0/75-M-H-10/seeds3042/eval/R0@16ms/walk/v0.4.json` | add-on A1: (P37: 14, 16 and 20 ms) | `2f2783b3d86e` |
| `stage2/runs/2026-11-16-75cm-addon/A/V0/75-M-H-10/seeds3042/eval/R2/42@16ms/walk/v0.4.json` | add-on A1: (P37: 14, 16 and 20 ms) | `87208a544c65` |
| `stage2/runs/2026-11-16-75cm-addon/A/V0/75-M-H-10/seeds3042/eval/R2/43@16ms/walk/v0.4.json` | add-on A1: (P37: 14, 16 and 20 ms) | `6bd1ba0d0706` |
| `stage2/runs/2026-11-16-75cm-addon/A/V0/75-M-H-10/seeds3042/eval/R0@16ms/walk/v0.4_mu0.54.json` | add-on A1: (P37: 14, 16 and 20 ms) | `85f1b278fcba` |
| `stage2/runs/2026-11-16-75cm-addon/A/V0/75-M-H-10/seeds3042/eval/R2/42@16ms/walk/v0.4_mu0.54.json` | add-on A1: (P37: 14, 16 and 20 ms) | `f7cc316ba6fd` |
| `stage2/runs/2026-11-16-75cm-addon/A/V0/75-M-H-10/seeds3042/eval/R2/43@16ms/walk/v0.4_mu0.54.json` | add-on A1: (P37: 14, 16 and 20 ms) | `104156298676` |
| `stage2/runs/2026-11-16-75cm-addon/A/V0/75-M-H-10/seeds3042/eval/R0@18ms/walk/nominal.json` | add-on A2: the lead's addition, not among the delays the owner approved | `a87ffb9d8e1a` |
| `stage2/runs/2026-11-16-75cm-addon/A/V0/75-M-H-10/seeds3042/eval/R2/42@18ms/walk/nominal.json` | add-on A2: the lead's addition, not among the delays the owner approved | `5fc4b74806a9` |
| `stage2/runs/2026-11-16-75cm-addon/A/V0/75-M-H-10/seeds3042/eval/R2/43@18ms/walk/nominal.json` | add-on A2: the lead's addition, not among the delays the owner approved | `93d64ceffdc0` |
| `stage2/runs/2026-11-16-75cm-addon/A/V0/75-M-H-10/seeds3042/eval/R0@18ms/walk/v0.4.json` | add-on A2: the lead's addition, not among the delays the owner approved | `060e843ab8c1` |
| `stage2/runs/2026-11-16-75cm-addon/A/V0/75-M-H-10/seeds3042/eval/R2/42@18ms/walk/v0.4.json` | add-on A2: the lead's addition, not among the delays the owner approved | `dbed2ef96beb` |
| `stage2/runs/2026-11-16-75cm-addon/A/V0/75-M-H-10/seeds3042/eval/R2/43@18ms/walk/v0.4.json` | add-on A2: the lead's addition, not among the delays the owner approved | `75def28fcffc` |
| `stage2/runs/2026-11-16-75cm-addon/A/V0/75-M-H-10/seeds3042/eval/R0@18ms/walk/v0.4_mu0.54.json` | add-on A2: the lead's addition, not among the delays the owner approved | `ff85a1b1f2bd` |
| `stage2/runs/2026-11-16-75cm-addon/A/V0/75-M-H-10/seeds3042/eval/R2/42@18ms/walk/v0.4_mu0.54.json` | add-on A2: the lead's addition, not among the delays the owner approved | `f9c9b6646671` |
| `stage2/runs/2026-11-16-75cm-addon/A/V0/75-M-H-10/seeds3042/eval/R2/43@18ms/walk/v0.4_mu0.54.json` | add-on A2: the lead's addition, not among the delays the owner approved | `a6e87372228e` |
| `stage2/runs/2026-11-16-75cm-addon/A/V0/75-M-H-10/seeds3042/eval/R0@20ms/walk/nominal.json` | add-on A1: (P37: 14, 16 and 20 ms) | `44348d922322` |
| `stage2/runs/2026-11-16-75cm-addon/A/V0/75-M-H-10/seeds3042/eval/R2/42@20ms/walk/nominal.json` | add-on A1: (P37: 14, 16 and 20 ms) | `7475c3c6bb92` |
| `stage2/runs/2026-11-16-75cm-addon/A/V0/75-M-H-10/seeds3042/eval/R2/43@20ms/walk/nominal.json` | add-on A1: (P37: 14, 16 and 20 ms) | `9e3316da4979` |
| `stage2/runs/2026-11-16-75cm-addon/A/V0/75-M-H-10/seeds3042/eval/R0@20ms/walk/v0.4.json` | add-on A1: (P37: 14, 16 and 20 ms) | `0968f7f9c3ac` |
| `stage2/runs/2026-11-16-75cm-addon/A/V0/75-M-H-10/seeds3042/eval/R2/42@20ms/walk/v0.4.json` | add-on A1: (P37: 14, 16 and 20 ms) | `599d6b896371` |
| `stage2/runs/2026-11-16-75cm-addon/A/V0/75-M-H-10/seeds3042/eval/R2/43@20ms/walk/v0.4.json` | add-on A1: (P37: 14, 16 and 20 ms) | `16cf245b6f34` |
| `stage2/runs/2026-11-16-75cm-addon/A/V0/75-M-H-10/seeds3042/eval/R0@20ms/walk/v0.4_mu0.54.json` | add-on A1: (P37: 14, 16 and 20 ms) | `2ed417a71c2d` |
| `stage2/runs/2026-11-16-75cm-addon/A/V0/75-M-H-10/seeds3042/eval/R2/42@20ms/walk/v0.4_mu0.54.json` | add-on A1: (P37: 14, 16 and 20 ms) | `0cd0abb81ef7` |
| `stage2/runs/2026-11-16-75cm-addon/A/V0/75-M-H-10/seeds3042/eval/R2/43@20ms/walk/v0.4_mu0.54.json` | add-on A1: (P37: 14, 16 and 20 ms) | `0885945abbaa` |
