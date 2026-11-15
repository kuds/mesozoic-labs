# Digest of the 75 cm run's records (`2026-11-15-75cm`)

Simulation only; no number here was measured on hardware. Every number is read from the committed records of this run by `digest/digest.py` (standard library only). It is a digest for the written report: the full machine-made tables, with every label and every record path, are `BAR_TABLES.md`. A dash is a cell the registered order does not hold; "not run" is a record the runner wrote as not run.

Cells of the count tables: **episodes that stayed up / episodes that met every limit / episodes that met every limit but for A1** (the neck and tail servos' sustained load), each of 20, seeds 3042 to 3061. A record passes with 20 up and at least 17 (H1). Rows `R2/42` and `R2/43` are the two fitted balance controllers (fitted with no servo delay); `R0` has no balance feedback and gives no verdict.

## 1. The walk at 0.40 m/s, grippy floor (mu 0.9): counts by servo delay

| Set | Row | 0 ms | 2 ms | 4 ms | 6 ms | 8 ms | 10 ms |
|---|---|---|---|---|---|---|---|
| V0 (control: today's walk) | `R2/42` | 20 / 0 / 19 | 20 / 0 / 20 | 20 / 0 / 16 | 20 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| V0 (control: today's walk) | `R2/43` | 20 / 0 / 19 | 20 / 0 / 20 | 20 / 0 / 12 | 20 / 0 / 0 | 0 / 0 / 0 | 1 / 0 / 0 |
| V0 (control: today's walk) | `R0` | 20 / 0 / 0 | 18 / 0 / 0 | 19 / 0 / 0 | 20 / 0 / 0 | 19 / 0 / 0 | 20 / 0 / 0 |
| F25F9 (control: the pull request's walk) | `R2/42` | 20 / 0 / 0 | 20 / 0 / 0 | 20 / 0 / 0 | 15 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| F25F9 (control: the pull request's walk) | `R2/43` | 20 / 0 / 0 | 20 / 0 / 0 | 20 / 0 / 0 | 19 / 0 / 0 | 0 / 0 / 0 | 2 / 0 / 0 |
| F25F9 (control: the pull request's walk) | `R0` | 20 / 0 / 0 | 20 / 0 / 0 | 20 / 0 / 0 | 20 / 0 / 0 | 20 / 0 / 0 | 20 / 0 / 0 |
| F25F9cl (the candidate) | `R2/42` | 20 / 0 / 0 | 20 / 0 / 0 | 20 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| F25F9cl (the candidate) | `R2/43` | 20 / 0 / 0 | 20 / 0 / 0 | 20 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| F25F9cl (the candidate) | `R0` | 20 / 0 / 0 | 20 / 0 / 0 | 20 / 0 / 0 | 20 / 0 / 0 | 20 / 0 / 0 | 20 / 0 / 0 |

## 1. The walk at 0.40 m/s, slippery floor (mu 0.54): counts by servo delay

| Set | Row | 0 ms | 2 ms | 4 ms | 6 ms | 8 ms | 10 ms |
|---|---|---|---|---|---|---|---|
| V0 (control: today's walk) | `R2/42` | 20 / 0 / 19 | 20 / 0 / 20 | 20 / 0 / 11 | 20 / 0 / 1 | 0 / 0 / 0 | 0 / 0 / 0 |
| V0 (control: today's walk) | `R2/43` | 20 / 0 / 20 | 20 / 0 / 19 | 20 / 0 / 11 | 20 / 0 / 0 | 0 / 0 / 0 | 3 / 0 / 0 |
| V0 (control: today's walk) | `R0` | 20 / 0 / 0 | 19 / 0 / 0 | 20 / 0 / 0 | 20 / 0 / 0 | 20 / 0 / 0 | 20 / 0 / 0 |
| F25F9 (control: the pull request's walk) | `R2/42` | 20 / 0 / 0 | 20 / 0 / 0 | 20 / 0 / 0 | 11 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| F25F9 (control: the pull request's walk) | `R2/43` | 20 / 0 / 0 | 20 / 0 / 0 | 20 / 0 / 0 | 18 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| F25F9 (control: the pull request's walk) | `R0` | 20 / 0 / 0 | 20 / 0 / 0 | 20 / 0 / 0 | 20 / 0 / 0 | 20 / 0 / 0 | 20 / 0 / 0 |
| F25F9cl (the candidate) | `R2/42` | 20 / 0 / 0 | 20 / 0 / 0 | 20 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| F25F9cl (the candidate) | `R2/43` | 20 / 0 / 0 | 20 / 0 / 0 | 20 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| F25F9cl (the candidate) | `R0` | 20 / 0 / 0 | 20 / 0 / 0 | 20 / 0 / 0 | 20 / 0 / 0 | 20 / 0 / 0 | 20 / 0 / 0 |

## 2. The walk at 0.40 m/s, grippy floor: measured medians

Medians over the episodes that give the value (n in brackets). Speed over the last 10 s. Toe clearance as the bar judges it (limit 20.07 mm). Duty as the mean of the two feet (limits 0.55 to 0.65). Tail or neck servo: the larger sustained-load ratio of the two over the window (limit 1.0; this is A1). Leg servo: the largest leg drive's ratio (limit 1.0).

| Set | Row | Delay | Speed (m/s) | Toe clearance (mm) | Duty | Tail or neck servo ratio | Leg servo ratio | Failed terms (episodes of 20) |
|---|---|---|---|---|---|---|---|---|
| V0 | `R2/42` | 0 ms | 0.410 (20) | 24.2 (20) | 0.563 | 1.07 | 0.80 | `drive_sustained` 20, `line` 1 |
| V0 | `R2/42` | 2 ms | 0.412 (20) | 22.9 (20) | 0.573 | 1.05 | 0.80 | `drive_sustained` 20 |
| V0 | `R2/42` | 4 ms | 0.412 (20) | 20.7 (20) | 0.583 | 1.04 | 0.79 | `drive_sustained` 20, `toe_clearance_own_swing` 4 |
| V0 | `R2/42` | 6 ms | 0.409 (20) | 18.9 (20) | 0.583 | 1.02 | 0.79 | `drive_sustained` 20, `toe_clearance_own_swing` 20 |
| V0 | `R2/42` | 8 ms | - (0) | - (0) | - | - | - | `survived` 20 |
| V0 | `R2/42` | 10 ms | - (0) | - (0) | - | - | - | `survived` 20 |
| V0 | `R2/43` | 0 ms | 0.411 (20) | 24.2 (20) | 0.560 | 1.08 | 0.80 | `drive_sustained` 20, `line` 1 |
| V0 | `R2/43` | 2 ms | 0.413 (20) | 23.1 (20) | 0.571 | 1.06 | 0.80 | `drive_sustained` 20 |
| V0 | `R2/43` | 4 ms | 0.414 (20) | 20.2 (20) | 0.584 | 1.05 | 0.79 | `drive_sustained` 20, `toe_clearance_own_swing` 7, `line` 1 |
| V0 | `R2/43` | 6 ms | 0.413 (20) | 18.3 (20) | 0.582 | 1.03 | 0.80 | `drive_sustained` 20, `toe_clearance_own_swing` 20 |
| V0 | `R2/43` | 8 ms | - (0) | - (0) | - | - | - | `survived` 20 |
| V0 | `R2/43` | 10 ms | 0.391 (1) | 4.3 (1) | 0.526 | 1.19 | 0.93 | `survived` 19, `along` 1, `drive_speed` 1, `drive_sustained` 1, `duty` 1, `toe_clearance_own_swing` 1 |
| V0 | `R0` | 0 ms | 0.410 (20) | 24.5 (20) | 0.546 | 1.10 | 0.80 | `drive_sustained` 20, `duty` 20, `along` 1, `no_flight` 1, `toe_clearance_own_swing` 1 |
| V0 | `R0` | 2 ms | 0.410 (18) | 25.5 (18) | 0.545 | 1.10 | 0.81 | `drive_sustained` 18, `duty` 18, `survived` 2, `line` 1 |
| V0 | `R0` | 4 ms | 0.410 (19) | 24.9 (19) | 0.546 | 1.10 | 0.80 | `drive_sustained` 19, `duty` 19, `no_flight` 3, `toe_clearance_own_swing` 3, `along` 1, `survived` 1 |
| V0 | `R0` | 6 ms | 0.410 (20) | 26.1 (20) | 0.546 | 1.10 | 0.81 | `drive_sustained` 20, `duty` 20 |
| V0 | `R0` | 8 ms | 0.410 (19) | 26.3 (19) | 0.546 | 1.10 | 0.81 | `drive_sustained` 19, `duty` 19, `line` 2, `survived` 1 |
| V0 | `R0` | 10 ms | 0.410 (20) | 25.5 (20) | 0.546 | 1.10 | 0.81 | `drive_sustained` 20, `duty` 20, `line` 1, `toe_clearance_own_swing` 1 |
| F25F9 | `R2/42` | 0 ms | 0.413 (20) | 17.2 (20) | 0.559 | 0.69 | 0.75 | `duty` 20, `toe_clearance_own_swing` 20, `line` 1 |
| F25F9 | `R2/42` | 2 ms | 0.413 (20) | 14.6 (20) | 0.573 | 0.69 | 0.75 | `toe_clearance_own_swing` 20, `duty` 1, `line` 1 |
| F25F9 | `R2/42` | 4 ms | 0.412 (20) | 12.7 (20) | 0.581 | 0.69 | 0.75 | `toe_clearance_own_swing` 20, `line` 1 |
| F25F9 | `R2/42` | 6 ms | 0.410 (15) | 12.6 (15) | 0.575 | 0.69 | 0.76 | `toe_clearance_own_swing` 15, `survived` 5, `line` 1 |
| F25F9 | `R2/42` | 8 ms | - (0) | - (0) | - | - | - | `survived` 20 |
| F25F9 | `R2/42` | 10 ms | - (0) | - (0) | - | - | - | `survived` 20 |
| F25F9 | `R2/43` | 0 ms | 0.412 (20) | 17.8 (20) | 0.551 | 0.69 | 0.76 | `duty` 20, `toe_clearance_own_swing` 20, `line` 1 |
| F25F9 | `R2/43` | 2 ms | 0.415 (20) | 15.0 (20) | 0.567 | 0.69 | 0.75 | `toe_clearance_own_swing` 20, `duty` 18, `line` 1 |
| F25F9 | `R2/43` | 4 ms | 0.415 (20) | 12.0 (20) | 0.579 | 0.69 | 0.75 | `toe_clearance_own_swing` 20, `line` 1 |
| F25F9 | `R2/43` | 6 ms | 0.415 (19) | 11.8 (19) | 0.574 | 0.70 | 0.76 | `toe_clearance_own_swing` 19, `duty` 4, `line` 1, `survived` 1 |
| F25F9 | `R2/43` | 8 ms | - (0) | - (0) | - | - | - | `survived` 20 |
| F25F9 | `R2/43` | 10 ms | 0.391 (2) | 1.9 (2) | 0.523 | 0.90 | 0.99 | `survived` 18, `along` 2, `drive_speed` 2, `duty` 2, `slip_median` 2, `toe_clearance_own_swing` 2 |
| F25F9 | `R0` | 0 ms | 0.411 (20) | 16.5 (20) | 0.534 | 0.68 | 0.73 | `duty` 20, `toe_clearance_own_swing` 20, `line` 1 |
| F25F9 | `R0` | 2 ms | 0.411 (20) | 16.4 (20) | 0.535 | 0.68 | 0.73 | `duty` 20, `toe_clearance_own_swing` 20 |
| F25F9 | `R0` | 4 ms | 0.411 (20) | 16.8 (20) | 0.533 | 0.68 | 0.73 | `duty` 20, `toe_clearance_own_swing` 20, `line` 1 |
| F25F9 | `R0` | 6 ms | 0.412 (20) | 16.8 (20) | 0.535 | 0.68 | 0.73 | `duty` 20, `toe_clearance_own_swing` 20 |
| F25F9 | `R0` | 8 ms | 0.411 (20) | 17.0 (20) | 0.532 | 0.68 | 0.73 | `duty` 20, `toe_clearance_own_swing` 20, `line` 1, `no_flight` 1 |
| F25F9 | `R0` | 10 ms | 0.411 (20) | 16.7 (20) | 0.534 | 0.68 | 0.73 | `duty` 20, `toe_clearance_own_swing` 20, `line` 1 |
| F25F9cl | `R2/42` | 0 ms | 0.415 (20) | 20.0 (20) | 0.593 | 0.69 | 0.76 | `joint_range` 20, `toe_clearance_own_swing` 11, `line` 1 |
| F25F9cl | `R2/42` | 2 ms | 0.416 (20) | 17.0 (20) | 0.607 | 0.69 | 0.75 | `joint_range` 20, `toe_clearance_own_swing` 20 |
| F25F9cl | `R2/42` | 4 ms | 0.413 (20) | 13.7 (20) | 0.618 | 0.70 | 0.76 | `joint_range` 20, `toe_clearance_own_swing` 20, `line` 1 |
| F25F9cl | `R2/42` | 6 ms | - (0) | - (0) | - | - | - | `survived` 20 |
| F25F9cl | `R2/42` | 8 ms | - (0) | - (0) | - | - | - | `survived` 20 |
| F25F9cl | `R2/42` | 10 ms | - (0) | - (0) | - | - | - | `survived` 20 |
| F25F9cl | `R2/43` | 0 ms | 0.414 (20) | 20.6 (20) | 0.587 | 0.70 | 0.77 | `joint_range` 20, `toe_clearance_own_swing` 2 |
| F25F9cl | `R2/43` | 2 ms | 0.417 (20) | 16.8 (20) | 0.605 | 0.69 | 0.76 | `joint_range` 20, `toe_clearance_own_swing` 20, `line` 1 |
| F25F9cl | `R2/43` | 4 ms | 0.416 (20) | 13.0 (20) | 0.620 | 0.70 | 0.76 | `joint_range` 20, `toe_clearance_own_swing` 20, `line` 1 |
| F25F9cl | `R2/43` | 6 ms | - (0) | - (0) | - | - | - | `survived` 20 |
| F25F9cl | `R2/43` | 8 ms | - (0) | - (0) | - | - | - | `survived` 20 |
| F25F9cl | `R2/43` | 10 ms | - (0) | - (0) | - | - | - | `survived` 20 |
| F25F9cl | `R0` | 0 ms | 0.413 (20) | 21.9 (20) | 0.565 | 0.69 | 0.76 | `duty` 20, `joint_range` 20 |
| F25F9cl | `R0` | 2 ms | 0.413 (20) | 21.9 (20) | 0.564 | 0.69 | 0.76 | `duty` 20, `joint_range` 20 |
| F25F9cl | `R0` | 4 ms | 0.413 (20) | 21.8 (20) | 0.564 | 0.69 | 0.76 | `joint_range` 20, `duty` 17, `line` 1 |
| F25F9cl | `R0` | 6 ms | 0.413 (20) | 21.8 (20) | 0.565 | 0.69 | 0.76 | `joint_range` 20, `duty` 18, `line` 1, `toe_clearance_own_swing` 1 |
| F25F9cl | `R0` | 8 ms | 0.413 (20) | 21.6 (20) | 0.564 | 0.69 | 0.76 | `duty` 20, `joint_range` 20, `line` 1 |
| F25F9cl | `R0` | 10 ms | 0.413 (20) | 21.9 (20) | 0.564 | 0.69 | 0.76 | `joint_range` 20, `duty` 19, `line` 1 |

## 3. The speed ladder, grippy floor: counts by rung and delay

`walk/nominal` is the set's slow walk (V0 and the candidate: 0.243 m/s). The ladder is read at 0, 4, 6 and 10 ms under the two fitted rows; the 0.40 m/s rung is section 1's record.

| Set | Row | Delay | `nominal` | `v0.3` | `v0.35` | `v0.4` | `v0.45` | `v0.5` |
|---|---|---|---|---|---|---|---|---|
| V0 | `R2/42` | 0 ms | 20 / 20 / 20 | 20 / 20 / 20 | 20 / 0 / 19 | 20 / 0 / 19 | 20 / 0 / 0 | 20 / 0 / 1 |
| V0 | `R2/43` | 0 ms | 20 / 20 / 20 | 20 / 20 / 20 | 20 / 0 / 19 | 20 / 0 / 19 | 20 / 0 / 0 | 20 / 0 / 0 |
| V0 | `R2/42` | 4 ms | 20 / 0 / 0 | 20 / 17 / 17 | 20 / 0 / 20 | 20 / 0 / 16 | 20 / 0 / 17 | 20 / 0 / 15 |
| V0 | `R2/43` | 4 ms | 20 / 0 / 0 | 20 / 12 / 12 | 20 / 0 / 18 | 20 / 0 / 12 | 20 / 0 / 11 | 20 / 0 / 17 |
| V0 | `R2/42` | 6 ms | 20 / 0 / 0 | 20 / 0 / 0 | 20 / 0 / 0 | 20 / 0 / 0 | 20 / 0 / 3 | 14 / 0 / 2 |
| V0 | `R2/43` | 6 ms | 20 / 0 / 0 | 20 / 0 / 0 | 20 / 0 / 0 | 20 / 0 / 0 | 19 / 0 / 0 | 20 / 0 / 4 |
| V0 | `R2/42` | 10 ms | 2 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| V0 | `R2/43` | 10 ms | 1 / 0 / 0 | 1 / 0 / 0 | 1 / 0 / 0 | 1 / 0 / 0 | 1 / 0 / 0 | 1 / 0 / 0 |
| F25F9 | `R2/42` | 0 ms | - | - | - | 20 / 0 / 0 | - | not run |
| F25F9 | `R2/43` | 0 ms | - | - | - | 20 / 0 / 0 | - | not run |
| F25F9 | `R2/42` | 4 ms | - | - | - | 20 / 0 / 0 | - | - |
| F25F9 | `R2/43` | 4 ms | - | - | - | 20 / 0 / 0 | - | - |
| F25F9 | `R2/42` | 6 ms | - | - | - | 15 / 0 / 0 | - | - |
| F25F9 | `R2/43` | 6 ms | - | - | - | 19 / 0 / 0 | - | - |
| F25F9 | `R2/42` | 10 ms | - | - | - | 0 / 0 / 0 | - | - |
| F25F9 | `R2/43` | 10 ms | - | - | - | 2 / 0 / 0 | - | - |
| F25F9cl | `R2/42` | 0 ms | 20 / 0 / 0 | 20 / 0 / 0 | 20 / 0 / 0 | 20 / 0 / 0 | 20 / 0 / 0 | 20 / 0 / 0 |
| F25F9cl | `R2/43` | 0 ms | 20 / 0 / 0 | 20 / 0 / 0 | 20 / 0 / 0 | 20 / 0 / 0 | 20 / 0 / 0 | 20 / 0 / 0 |
| F25F9cl | `R2/42` | 4 ms | 20 / 0 / 0 | 20 / 0 / 0 | 20 / 0 / 0 | 20 / 0 / 0 | 20 / 0 / 0 | 20 / 0 / 0 |
| F25F9cl | `R2/43` | 4 ms | 20 / 0 / 0 | 20 / 0 / 0 | 20 / 0 / 0 | 20 / 0 / 0 | 20 / 0 / 0 | 20 / 0 / 0 |
| F25F9cl | `R2/42` | 6 ms | 20 / 0 / 0 | 19 / 0 / 0 | 14 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| F25F9cl | `R2/43` | 6 ms | 19 / 0 / 0 | 17 / 0 / 0 | 18 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| F25F9cl | `R2/42` | 10 ms | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| F25F9cl | `R2/43` | 10 ms | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |

## 4. Turns, nudges and stops: counts by delay

Each of these episodes walks at 0.40 m/s first; the command or push comes after about 13 s. Falls are split into before and after the command (b / a).

| Set | Condition | Row | 0 ms | 4 ms | 6 ms | 10 ms |
|---|---|---|---|---|---|---|
| V0 | `nudge/backward` | `R2/42` | 20 / 13 / 13 (falls 0 b / 0 a) | 14 / 13 / 13 (falls 0 b / 6 a) | 14 / 10 / 10 (falls 0 b / 6 a) | 0 / 0 / 0 (falls 20 b / 0 a) |
| V0 | `nudge/backward` | `R2/43` | 20 / 13 / 13 (falls 0 b / 0 a) | 12 / 10 / 10 (falls 0 b / 8 a) | 9 / 8 / 8 (falls 0 b / 11 a) | 1 / 0 / 0 (falls 19 b / 0 a) |
| V0 | `nudge/backward` | `R0` | 9 / 1 / 4 (falls 0 b / 11 a) | 8 / 2 / 2 (falls 0 b / 12 a) | 10 / 1 / 3 (falls 0 b / 10 a) | 10 / 3 / 4 (falls 0 b / 10 a) |
| V0 | `nudge/backward_mu0.54` | `R2/42` | 18 / 12 / 12 (falls 0 b / 2 a) | 13 / 13 / 13 (falls 0 b / 7 a) | 12 / 10 / 10 (falls 0 b / 8 a) | 0 / 0 / 0 (falls 20 b / 0 a) |
| V0 | `nudge/backward_mu0.54` | `R2/43` | 20 / 12 / 12 (falls 0 b / 0 a) | 15 / 11 / 11 (falls 0 b / 5 a) | 11 / 10 / 10 (falls 0 b / 9 a) | 1 / 0 / 0 (falls 17 b / 2 a) |
| V0 | `nudge/backward_mu0.54` | `R0` | 5 / 2 / 2 (falls 0 b / 15 a) | 5 / 0 / 0 (falls 0 b / 15 a) | 6 / 0 / 0 (falls 0 b / 14 a) | 9 / 4 / 4 (falls 0 b / 11 a) |
| V0 | `nudge/backward_x0.375` | `R2/42` | - | - | - | 0 / 0 / 0 (falls 20 b / 0 a) |
| V0 | `nudge/backward_x0.375` | `R2/43` | - | - | - | 0 / 0 / 0 (falls 19 b / 1 a) |
| V0 | `nudge/backward_x0.5` | `R2/42` | - | - | - | 0 / 0 / 0 (falls 20 b / 0 a) |
| V0 | `nudge/backward_x0.5` | `R2/43` | - | - | - | 0 / 0 / 0 (falls 19 b / 1 a) |
| V0 | `nudge/forward` | `R2/42` | 19 / 14 / 14 (falls 0 b / 1 a) | 20 / 19 / 19 (falls 0 b / 0 a) | 19 / 19 / 19 (falls 0 b / 1 a) | 0 / 0 / 0 (falls 20 b / 0 a) |
| V0 | `nudge/forward` | `R2/43` | 20 / 14 / 14 (falls 0 b / 0 a) | 14 / 12 / 12 (falls 0 b / 6 a) | 15 / 13 / 13 (falls 0 b / 5 a) | 0 / 0 / 0 (falls 19 b / 1 a) |
| V0 | `nudge/forward` | `R0` | 3 / 0 / 0 (falls 0 b / 17 a) | 1 / 0 / 0 (falls 0 b / 19 a) | 2 / 1 / 1 (falls 0 b / 18 a) | 2 / 0 / 0 (falls 0 b / 18 a) |
| V0 | `nudge/forward_mu0.54` | `R2/42` | 19 / 15 / 15 (falls 0 b / 1 a) | 19 / 19 / 19 (falls 0 b / 1 a) | 15 / 15 / 15 (falls 0 b / 5 a) | 0 / 0 / 0 (falls 20 b / 0 a) |
| V0 | `nudge/forward_mu0.54` | `R2/43` | 20 / 14 / 14 (falls 0 b / 0 a) | 13 / 11 / 11 (falls 0 b / 7 a) | 15 / 12 / 12 (falls 0 b / 5 a) | 1 / 0 / 0 (falls 17 b / 2 a) |
| V0 | `nudge/forward_mu0.54` | `R0` | 0 / 0 / 0 (falls 0 b / 20 a) | 1 / 1 / 1 (falls 0 b / 19 a) | 1 / 0 / 0 (falls 0 b / 19 a) | 4 / 0 / 0 (falls 0 b / 16 a) |
| V0 | `nudge/forward_x0.375` | `R2/42` | - | - | - | 0 / 0 / 0 (falls 20 b / 0 a) |
| V0 | `nudge/forward_x0.375` | `R2/43` | - | - | - | 0 / 0 / 0 (falls 19 b / 1 a) |
| V0 | `nudge/forward_x0.5` | `R2/42` | - | - | - | 0 / 0 / 0 (falls 20 b / 0 a) |
| V0 | `nudge/forward_x0.5` | `R2/43` | - | - | - | 0 / 0 / 0 (falls 19 b / 1 a) |
| V0 | `nudge/left` | `R2/42` | 20 / 15 / 15 (falls 0 b / 0 a) | 17 / 10 / 10 (falls 0 b / 3 a) | 12 / 5 / 5 (falls 0 b / 8 a) | 0 / 0 / 0 (falls 20 b / 0 a) |
| V0 | `nudge/left` | `R2/43` | 20 / 11 / 11 (falls 0 b / 0 a) | 14 / 6 / 6 (falls 0 b / 6 a) | 15 / 5 / 5 (falls 0 b / 5 a) | 1 / 0 / 0 (falls 19 b / 0 a) |
| V0 | `nudge/left` | `R0` | 13 / 5 / 9 (falls 0 b / 7 a) | 15 / 7 / 11 (falls 0 b / 5 a) | 12 / 6 / 10 (falls 0 b / 8 a) | 12 / 6 / 8 (falls 0 b / 8 a) |
| V0 | `nudge/left_mu0.54` | `R2/42` | 19 / 14 / 14 (falls 0 b / 1 a) | 13 / 10 / 10 (falls 0 b / 7 a) | 11 / 2 / 2 (falls 0 b / 9 a) | 0 / 0 / 0 (falls 20 b / 0 a) |
| V0 | `nudge/left_mu0.54` | `R2/43` | 17 / 11 / 11 (falls 0 b / 3 a) | 14 / 6 / 6 (falls 0 b / 6 a) | 14 / 6 / 6 (falls 0 b / 6 a) | 0 / 0 / 0 (falls 17 b / 3 a) |
| V0 | `nudge/left_mu0.54` | `R0` | 8 / 3 / 3 (falls 0 b / 12 a) | 9 / 5 / 5 (falls 0 b / 11 a) | 10 / 7 / 7 (falls 0 b / 10 a) | 12 / 6 / 6 (falls 0 b / 8 a) |
| V0 | `nudge/left_x0.375` | `R2/42` | - | - | - | 0 / 0 / 0 (falls 20 b / 0 a) |
| V0 | `nudge/left_x0.375` | `R2/43` | - | - | - | 1 / 0 / 0 (falls 19 b / 0 a) |
| V0 | `nudge/left_x0.5` | `R2/42` | - | - | - | 0 / 0 / 0 (falls 20 b / 0 a) |
| V0 | `nudge/left_x0.5` | `R2/43` | - | - | - | 0 / 0 / 0 (falls 19 b / 1 a) |
| V0 | `nudge/right` | `R2/42` | 19 / 13 / 13 (falls 0 b / 1 a) | 15 / 7 / 7 (falls 0 b / 5 a) | 13 / 3 / 3 (falls 0 b / 7 a) | 0 / 0 / 0 (falls 20 b / 0 a) |
| V0 | `nudge/right` | `R2/43` | 20 / 13 / 13 (falls 0 b / 0 a) | 15 / 9 / 9 (falls 0 b / 5 a) | 14 / 4 / 4 (falls 0 b / 6 a) | 0 / 0 / 0 (falls 19 b / 1 a) |
| V0 | `nudge/right` | `R0` | 17 / 4 / 9 (falls 0 b / 3 a) | 15 / 1 / 6 (falls 0 b / 5 a) | 16 / 6 / 10 (falls 0 b / 4 a) | 14 / 4 / 9 (falls 0 b / 6 a) |
| V0 | `nudge/right_mu0.54` | `R2/42` | 18 / 13 / 13 (falls 0 b / 2 a) | 13 / 7 / 7 (falls 0 b / 7 a) | 13 / 2 / 2 (falls 0 b / 7 a) | 0 / 0 / 0 (falls 20 b / 0 a) |
| V0 | `nudge/right_mu0.54` | `R2/43` | 20 / 13 / 13 (falls 0 b / 0 a) | 15 / 9 / 9 (falls 0 b / 5 a) | 15 / 7 / 7 (falls 0 b / 5 a) | 1 / 0 / 0 (falls 17 b / 2 a) |
| V0 | `nudge/right_mu0.54` | `R0` | 11 / 5 / 5 (falls 0 b / 9 a) | 11 / 7 / 7 (falls 0 b / 9 a) | 13 / 7 / 7 (falls 0 b / 7 a) | 12 / 5 / 5 (falls 0 b / 8 a) |
| V0 | `nudge/right_x0.375` | `R2/42` | - | - | - | 0 / 0 / 0 (falls 20 b / 0 a) |
| V0 | `nudge/right_x0.375` | `R2/43` | - | - | - | 0 / 0 / 0 (falls 19 b / 1 a) |
| V0 | `nudge/right_x0.5` | `R2/42` | - | - | - | 0 / 0 / 0 (falls 20 b / 0 a) |
| V0 | `nudge/right_x0.5` | `R2/43` | - | - | - | 0 / 0 / 0 (falls 19 b / 1 a) |
| V0 | `stop/one_step` | `R2/42` | - | - | - | 0 / 0 / 0 (falls 20 b / 0 a) |
| V0 | `stop/one_step` | `R2/43` | - | - | - | 1 / 0 / 0 (falls 19 b / 0 a) |
| V0 | `stop/two_step` | `R2/42` | 20 / 20 / 20 (falls 0 b / 0 a) | 20 / 20 / 20 (falls 0 b / 0 a) | 20 / 20 / 20 (falls 0 b / 0 a) | 0 / 0 / 0 (falls 20 b / 0 a) |
| V0 | `stop/two_step` | `R2/43` | 20 / 20 / 20 (falls 0 b / 0 a) | 20 / 20 / 20 (falls 0 b / 0 a) | 20 / 20 / 20 (falls 0 b / 0 a) | 1 / 0 / 0 (falls 19 b / 0 a) |
| V0 | `stop/two_step` | `R0` | 20 / 19 / 19 (falls 0 b / 0 a) | 20 / 18 / 18 (falls 0 b / 0 a) | 20 / 20 / 20 (falls 0 b / 0 a) | 20 / 20 / 20 (falls 0 b / 0 a) |
| V0 | `stop/two_step_mu0.54` | `R2/42` | 20 / 20 / 20 (falls 0 b / 0 a) | 20 / 20 / 20 (falls 0 b / 0 a) | 20 / 20 / 20 (falls 0 b / 0 a) | 0 / 0 / 0 (falls 20 b / 0 a) |
| V0 | `stop/two_step_mu0.54` | `R2/43` | 20 / 20 / 20 (falls 0 b / 0 a) | 20 / 20 / 20 (falls 0 b / 0 a) | 20 / 20 / 20 (falls 0 b / 0 a) | 3 / 0 / 0 (falls 17 b / 0 a) |
| V0 | `stop/two_step_mu0.54` | `R0` | 20 / 20 / 20 (falls 0 b / 0 a) | 20 / 20 / 20 (falls 0 b / 0 a) | 20 / 20 / 20 (falls 0 b / 0 a) | 20 / 20 / 20 (falls 0 b / 0 a) |
| V0 | `turn/q90_left` | `R2/42` | 20 / 0 / 0 (falls 0 b / 0 a) | 20 / 0 / 10 (falls 0 b / 0 a) | 20 / 0 / 1 (falls 0 b / 0 a) | 0 / 0 / 0 (falls 20 b / 0 a) |
| V0 | `turn/q90_left` | `R2/43` | 20 / 0 / 0 (falls 0 b / 0 a) | 20 / 0 / 7 (falls 0 b / 0 a) | 20 / 0 / 2 (falls 0 b / 0 a) | 0 / 0 / 0 (falls 19 b / 1 a) |
| V0 | `turn/q90_left` | `R0` | 20 / 0 / 0 (falls 0 b / 0 a) | 19 / 0 / 0 (falls 0 b / 1 a) | 20 / 0 / 0 (falls 0 b / 0 a) | 20 / 0 / 0 (falls 0 b / 0 a) |
| V0 | `turn/q90_left_mu0.54` | `R2/42` | 20 / 0 / 0 (falls 0 b / 0 a) | 20 / 0 / 15 (falls 0 b / 0 a) | 20 / 0 / 1 (falls 0 b / 0 a) | 0 / 0 / 0 (falls 20 b / 0 a) |
| V0 | `turn/q90_left_mu0.54` | `R2/43` | 19 / 0 / 0 (falls 0 b / 1 a) | 20 / 0 / 1 (falls 0 b / 0 a) | 20 / 0 / 0 (falls 0 b / 0 a) | 0 / 0 / 0 (falls 17 b / 3 a) |
| V0 | `turn/q90_left_mu0.54` | `R0` | 19 / 0 / 0 (falls 0 b / 1 a) | 19 / 0 / 0 (falls 0 b / 1 a) | 20 / 0 / 0 (falls 0 b / 0 a) | 20 / 0 / 0 (falls 0 b / 0 a) |
| V0 | `turn/q90_right` | `R2/42` | 20 / 0 / 0 (falls 0 b / 0 a) | 20 / 0 / 12 (falls 0 b / 0 a) | 20 / 0 / 3 (falls 0 b / 0 a) | 0 / 0 / 0 (falls 20 b / 0 a) |
| V0 | `turn/q90_right` | `R2/43` | 20 / 0 / 0 (falls 0 b / 0 a) | 20 / 0 / 9 (falls 0 b / 0 a) | 20 / 0 / 0 (falls 0 b / 0 a) | 0 / 0 / 0 (falls 19 b / 1 a) |
| V0 | `turn/q90_right` | `R0` | 19 / 0 / 0 (falls 0 b / 1 a) | 19 / 0 / 0 (falls 0 b / 1 a) | 20 / 0 / 0 (falls 0 b / 0 a) | 20 / 0 / 0 (falls 0 b / 0 a) |
| V0 | `turn/q90_right_mu0.54` | `R2/42` | 20 / 0 / 0 (falls 0 b / 0 a) | 20 / 0 / 5 (falls 0 b / 0 a) | 20 / 0 / 1 (falls 0 b / 0 a) | 0 / 0 / 0 (falls 20 b / 0 a) |
| V0 | `turn/q90_right_mu0.54` | `R2/43` | 19 / 0 / 0 (falls 0 b / 1 a) | 20 / 0 / 4 (falls 0 b / 0 a) | 20 / 0 / 0 (falls 0 b / 0 a) | 1 / 0 / 0 (falls 17 b / 2 a) |
| V0 | `turn/q90_right_mu0.54` | `R0` | 20 / 0 / 0 (falls 0 b / 0 a) | 20 / 0 / 0 (falls 0 b / 0 a) | 20 / 0 / 0 (falls 0 b / 0 a) | 19 / 0 / 0 (falls 0 b / 1 a) |
| V0 | `turn/t90_0.35_left` | `R2/42` | - | - | - | 0 / 0 / 0 (falls 20 b / 0 a) |
| V0 | `turn/t90_0.35_left` | `R2/43` | - | - | - | 1 / 0 / 0 (falls 19 b / 0 a) |
| V0 | `turn/t90_0.35_right` | `R2/42` | - | - | - | 0 / 0 / 0 (falls 20 b / 0 a) |
| V0 | `turn/t90_0.35_right` | `R2/43` | - | - | - | 1 / 0 / 0 (falls 19 b / 0 a) |
| V0 | `turn/t90_0.5_left` | `R2/42` | - | - | - | 0 / 0 / 0 (falls 20 b / 0 a) |
| V0 | `turn/t90_0.5_left` | `R2/43` | - | - | - | 1 / 0 / 0 (falls 19 b / 0 a) |
| V0 | `turn/t90_0.5_right` | `R2/42` | - | - | - | 0 / 0 / 0 (falls 20 b / 0 a) |
| V0 | `turn/t90_0.5_right` | `R2/43` | - | - | - | 1 / 0 / 0 (falls 19 b / 0 a) |
| V0 | `turn/t90_1_left` | `R2/42` | - | - | - | not run |
| V0 | `turn/t90_1_left` | `R2/43` | - | - | - | not run |
| V0 | `turn/t90_1_right` | `R2/42` | - | - | - | not run |
| V0 | `turn/t90_1_right` | `R2/43` | - | - | - | not run |
| F25F9cl | `nudge/backward` | `R2/42` | 19 / 0 / 0 (falls 0 b / 1 a) | 18 / 0 / 0 (falls 0 b / 2 a) | 0 / 0 / 0 (falls 19 b / 1 a) | 0 / 0 / 0 (falls 20 b / 0 a) |
| F25F9cl | `nudge/backward` | `R2/43` | 17 / 0 / 0 (falls 0 b / 3 a) | 12 / 0 / 0 (falls 0 b / 8 a) | 0 / 0 / 0 (falls 17 b / 3 a) | 0 / 0 / 0 (falls 20 b / 0 a) |
| F25F9cl | `nudge/backward` | `R0` | 9 / 0 / 0 (falls 0 b / 11 a) | 11 / 0 / 0 (falls 0 b / 9 a) | 8 / 0 / 0 (falls 0 b / 12 a) | 9 / 0 / 0 (falls 0 b / 11 a) |
| F25F9cl | `nudge/backward_mu0.54` | `R2/42` | 19 / 0 / 0 (falls 0 b / 1 a) | 20 / 0 / 0 (falls 0 b / 0 a) | 0 / 0 / 0 (falls 20 b / 0 a) | 0 / 0 / 0 (falls 20 b / 0 a) |
| F25F9cl | `nudge/backward_mu0.54` | `R2/43` | 17 / 0 / 0 (falls 0 b / 3 a) | 18 / 0 / 0 (falls 0 b / 2 a) | 0 / 0 / 0 (falls 18 b / 2 a) | 0 / 0 / 0 (falls 20 b / 0 a) |
| F25F9cl | `nudge/backward_mu0.54` | `R0` | 13 / 0 / 0 (falls 0 b / 7 a) | 9 / 0 / 0 (falls 0 b / 11 a) | 9 / 0 / 0 (falls 0 b / 11 a) | 9 / 0 / 0 (falls 0 b / 11 a) |
| F25F9cl | `nudge/backward_x0.375` | `R2/42` | - | - | - | 0 / 0 / 0 (falls 20 b / 0 a) |
| F25F9cl | `nudge/backward_x0.375` | `R2/43` | - | - | - | 0 / 0 / 0 (falls 20 b / 0 a) |
| F25F9cl | `nudge/backward_x0.5` | `R2/42` | - | - | - | 0 / 0 / 0 (falls 20 b / 0 a) |
| F25F9cl | `nudge/backward_x0.5` | `R2/43` | - | - | - | 0 / 0 / 0 (falls 20 b / 0 a) |
| F25F9cl | `nudge/forward` | `R2/42` | 20 / 0 / 0 (falls 0 b / 0 a) | 20 / 0 / 0 (falls 0 b / 0 a) | 0 / 0 / 0 (falls 19 b / 1 a) | 0 / 0 / 0 (falls 20 b / 0 a) |
| F25F9cl | `nudge/forward` | `R2/43` | 19 / 0 / 0 (falls 0 b / 1 a) | 19 / 0 / 0 (falls 0 b / 1 a) | 0 / 0 / 0 (falls 17 b / 3 a) | 0 / 0 / 0 (falls 20 b / 0 a) |
| F25F9cl | `nudge/forward` | `R0` | 7 / 0 / 0 (falls 0 b / 13 a) | 7 / 0 / 0 (falls 0 b / 13 a) | 8 / 0 / 0 (falls 0 b / 12 a) | 7 / 0 / 0 (falls 0 b / 13 a) |
| F25F9cl | `nudge/forward_mu0.54` | `R2/42` | 19 / 0 / 0 (falls 0 b / 1 a) | 20 / 0 / 0 (falls 0 b / 0 a) | 0 / 0 / 0 (falls 20 b / 0 a) | 0 / 0 / 0 (falls 20 b / 0 a) |
| F25F9cl | `nudge/forward_mu0.54` | `R2/43` | 20 / 0 / 0 (falls 0 b / 0 a) | 20 / 0 / 0 (falls 0 b / 0 a) | 0 / 0 / 0 (falls 18 b / 2 a) | 0 / 0 / 0 (falls 20 b / 0 a) |
| F25F9cl | `nudge/forward_mu0.54` | `R0` | 7 / 0 / 0 (falls 0 b / 13 a) | 10 / 0 / 0 (falls 0 b / 10 a) | 7 / 0 / 0 (falls 0 b / 13 a) | 7 / 0 / 0 (falls 0 b / 13 a) |
| F25F9cl | `nudge/forward_x0.375` | `R2/42` | - | - | - | 0 / 0 / 0 (falls 20 b / 0 a) |
| F25F9cl | `nudge/forward_x0.375` | `R2/43` | - | - | - | 0 / 0 / 0 (falls 20 b / 0 a) |
| F25F9cl | `nudge/forward_x0.5` | `R2/42` | - | - | - | 0 / 0 / 0 (falls 20 b / 0 a) |
| F25F9cl | `nudge/forward_x0.5` | `R2/43` | - | - | - | 0 / 0 / 0 (falls 20 b / 0 a) |
| F25F9cl | `nudge/left` | `R2/42` | 15 / 0 / 0 (falls 0 b / 5 a) | 12 / 0 / 0 (falls 0 b / 8 a) | 0 / 0 / 0 (falls 19 b / 1 a) | 0 / 0 / 0 (falls 20 b / 0 a) |
| F25F9cl | `nudge/left` | `R2/43` | 15 / 0 / 0 (falls 0 b / 5 a) | 12 / 0 / 0 (falls 0 b / 8 a) | 0 / 0 / 0 (falls 17 b / 3 a) | 0 / 0 / 0 (falls 20 b / 0 a) |
| F25F9cl | `nudge/left` | `R0` | 10 / 0 / 0 (falls 0 b / 10 a) | 9 / 0 / 0 (falls 0 b / 11 a) | 10 / 0 / 0 (falls 0 b / 10 a) | 7 / 0 / 0 (falls 0 b / 13 a) |
| F25F9cl | `nudge/left_mu0.54` | `R2/42` | 15 / 0 / 0 (falls 0 b / 5 a) | 11 / 0 / 0 (falls 0 b / 9 a) | 0 / 0 / 0 (falls 20 b / 0 a) | 0 / 0 / 0 (falls 20 b / 0 a) |
| F25F9cl | `nudge/left_mu0.54` | `R2/43` | 15 / 0 / 0 (falls 0 b / 5 a) | 12 / 0 / 0 (falls 0 b / 8 a) | 0 / 0 / 0 (falls 18 b / 2 a) | 0 / 0 / 0 (falls 20 b / 0 a) |
| F25F9cl | `nudge/left_mu0.54` | `R0` | 8 / 0 / 0 (falls 0 b / 12 a) | 10 / 0 / 0 (falls 0 b / 10 a) | 11 / 0 / 0 (falls 0 b / 9 a) | 8 / 0 / 0 (falls 0 b / 12 a) |
| F25F9cl | `nudge/left_x0.375` | `R2/42` | - | - | - | 0 / 0 / 0 (falls 20 b / 0 a) |
| F25F9cl | `nudge/left_x0.375` | `R2/43` | - | - | - | 0 / 0 / 0 (falls 20 b / 0 a) |
| F25F9cl | `nudge/left_x0.5` | `R2/42` | - | - | - | 0 / 0 / 0 (falls 20 b / 0 a) |
| F25F9cl | `nudge/left_x0.5` | `R2/43` | - | - | - | 0 / 0 / 0 (falls 20 b / 0 a) |
| F25F9cl | `nudge/right` | `R2/42` | 15 / 0 / 0 (falls 0 b / 5 a) | 6 / 0 / 0 (falls 0 b / 14 a) | 0 / 0 / 0 (falls 19 b / 1 a) | 0 / 0 / 0 (falls 20 b / 0 a) |
| F25F9cl | `nudge/right` | `R2/43` | 15 / 0 / 0 (falls 0 b / 5 a) | 11 / 0 / 0 (falls 0 b / 9 a) | 0 / 0 / 0 (falls 17 b / 3 a) | 0 / 0 / 0 (falls 20 b / 0 a) |
| F25F9cl | `nudge/right` | `R0` | 5 / 0 / 0 (falls 0 b / 15 a) | 4 / 0 / 0 (falls 0 b / 16 a) | 4 / 0 / 0 (falls 0 b / 16 a) | 6 / 0 / 0 (falls 0 b / 14 a) |
| F25F9cl | `nudge/right_mu0.54` | `R2/42` | 15 / 0 / 0 (falls 0 b / 5 a) | 6 / 0 / 0 (falls 0 b / 14 a) | 0 / 0 / 0 (falls 20 b / 0 a) | 0 / 0 / 0 (falls 20 b / 0 a) |
| F25F9cl | `nudge/right_mu0.54` | `R2/43` | 15 / 0 / 0 (falls 0 b / 5 a) | 11 / 0 / 0 (falls 0 b / 9 a) | 0 / 0 / 0 (falls 18 b / 2 a) | 0 / 0 / 0 (falls 20 b / 0 a) |
| F25F9cl | `nudge/right_mu0.54` | `R0` | 7 / 0 / 0 (falls 0 b / 13 a) | 8 / 0 / 0 (falls 0 b / 12 a) | 7 / 0 / 0 (falls 0 b / 13 a) | 8 / 0 / 0 (falls 0 b / 12 a) |
| F25F9cl | `nudge/right_x0.375` | `R2/42` | - | - | - | 0 / 0 / 0 (falls 20 b / 0 a) |
| F25F9cl | `nudge/right_x0.375` | `R2/43` | - | - | - | 0 / 0 / 0 (falls 20 b / 0 a) |
| F25F9cl | `nudge/right_x0.5` | `R2/42` | - | - | - | 0 / 0 / 0 (falls 20 b / 0 a) |
| F25F9cl | `nudge/right_x0.5` | `R2/43` | - | - | - | 0 / 0 / 0 (falls 20 b / 0 a) |
| F25F9cl | `stop/one_step` | `R2/42` | - | - | - | 0 / 0 / 0 (falls 20 b / 0 a) |
| F25F9cl | `stop/one_step` | `R2/43` | - | - | - | 0 / 0 / 0 (falls 20 b / 0 a) |
| F25F9cl | `stop/two_step` | `R2/42` | 20 / 0 / 0 (falls 0 b / 0 a) | 20 / 0 / 0 (falls 0 b / 0 a) | 1 / 0 / 0 (falls 19 b / 0 a) | 0 / 0 / 0 (falls 20 b / 0 a) |
| F25F9cl | `stop/two_step` | `R2/43` | 20 / 0 / 0 (falls 0 b / 0 a) | 20 / 0 / 0 (falls 0 b / 0 a) | 1 / 0 / 0 (falls 18 b / 1 a) | 0 / 0 / 0 (falls 20 b / 0 a) |
| F25F9cl | `stop/two_step` | `R0` | 20 / 0 / 0 (falls 0 b / 0 a) | 20 / 0 / 0 (falls 0 b / 0 a) | 20 / 0 / 0 (falls 0 b / 0 a) | 20 / 0 / 0 (falls 0 b / 0 a) |
| F25F9cl | `stop/two_step_mu0.54` | `R2/42` | 20 / 0 / 0 (falls 0 b / 0 a) | 20 / 0 / 0 (falls 0 b / 0 a) | 0 / 0 / 0 (falls 20 b / 0 a) | 0 / 0 / 0 (falls 20 b / 0 a) |
| F25F9cl | `stop/two_step_mu0.54` | `R2/43` | 20 / 0 / 0 (falls 0 b / 0 a) | 20 / 0 / 0 (falls 0 b / 0 a) | 2 / 0 / 0 (falls 18 b / 0 a) | 0 / 0 / 0 (falls 20 b / 0 a) |
| F25F9cl | `stop/two_step_mu0.54` | `R0` | 20 / 0 / 0 (falls 0 b / 0 a) | 20 / 0 / 0 (falls 0 b / 0 a) | 20 / 0 / 0 (falls 0 b / 0 a) | 20 / 0 / 0 (falls 0 b / 0 a) |
| F25F9cl | `turn/q90_left` | `R2/42` | 20 / 0 / 0 (falls 0 b / 0 a) | 20 / 0 / 0 (falls 0 b / 0 a) | 1 / 0 / 0 (falls 19 b / 0 a) | 0 / 0 / 0 (falls 20 b / 0 a) |
| F25F9cl | `turn/q90_left` | `R2/43` | 20 / 0 / 0 (falls 0 b / 0 a) | 19 / 0 / 0 (falls 0 b / 1 a) | 1 / 0 / 0 (falls 17 b / 2 a) | 0 / 0 / 0 (falls 20 b / 0 a) |
| F25F9cl | `turn/q90_left` | `R0` | 0 / 0 / 0 (falls 0 b / 20 a) | 0 / 0 / 0 (falls 0 b / 20 a) | 0 / 0 / 0 (falls 0 b / 20 a) | 0 / 0 / 0 (falls 0 b / 20 a) |
| F25F9cl | `turn/q90_left_mu0.54` | `R2/42` | 20 / 0 / 0 (falls 0 b / 0 a) | 20 / 0 / 0 (falls 0 b / 0 a) | 0 / 0 / 0 (falls 20 b / 0 a) | 0 / 0 / 0 (falls 20 b / 0 a) |
| F25F9cl | `turn/q90_left_mu0.54` | `R2/43` | 20 / 0 / 0 (falls 0 b / 0 a) | 20 / 0 / 0 (falls 0 b / 0 a) | 1 / 0 / 0 (falls 18 b / 1 a) | 0 / 0 / 0 (falls 20 b / 0 a) |
| F25F9cl | `turn/q90_left_mu0.54` | `R0` | 0 / 0 / 0 (falls 0 b / 20 a) | 0 / 0 / 0 (falls 0 b / 20 a) | 0 / 0 / 0 (falls 0 b / 20 a) | 0 / 0 / 0 (falls 0 b / 20 a) |
| F25F9cl | `turn/q90_right` | `R2/42` | 20 / 0 / 0 (falls 0 b / 0 a) | 19 / 0 / 0 (falls 0 b / 1 a) | 1 / 0 / 0 (falls 19 b / 0 a) | 0 / 0 / 0 (falls 20 b / 0 a) |
| F25F9cl | `turn/q90_right` | `R2/43` | 20 / 0 / 0 (falls 0 b / 0 a) | 19 / 0 / 0 (falls 0 b / 1 a) | 1 / 0 / 0 (falls 18 b / 1 a) | 0 / 0 / 0 (falls 20 b / 0 a) |
| F25F9cl | `turn/q90_right` | `R0` | 0 / 0 / 0 (falls 0 b / 20 a) | 0 / 0 / 0 (falls 0 b / 20 a) | 0 / 0 / 0 (falls 0 b / 20 a) | 0 / 0 / 0 (falls 0 b / 20 a) |
| F25F9cl | `turn/q90_right_mu0.54` | `R2/42` | 20 / 0 / 0 (falls 0 b / 0 a) | 20 / 0 / 0 (falls 0 b / 0 a) | 0 / 0 / 0 (falls 20 b / 0 a) | 0 / 0 / 0 (falls 20 b / 0 a) |
| F25F9cl | `turn/q90_right_mu0.54` | `R2/43` | 20 / 0 / 0 (falls 0 b / 0 a) | 20 / 0 / 0 (falls 0 b / 0 a) | 1 / 0 / 0 (falls 18 b / 1 a) | 0 / 0 / 0 (falls 20 b / 0 a) |
| F25F9cl | `turn/q90_right_mu0.54` | `R0` | 0 / 0 / 0 (falls 0 b / 20 a) | 0 / 0 / 0 (falls 0 b / 20 a) | 0 / 0 / 0 (falls 0 b / 20 a) | 0 / 0 / 0 (falls 0 b / 20 a) |
| F25F9cl | `turn/t90_0.35_left` | `R2/42` | - | - | - | 0 / 0 / 0 (falls 20 b / 0 a) |
| F25F9cl | `turn/t90_0.35_left` | `R2/43` | - | - | - | 0 / 0 / 0 (falls 20 b / 0 a) |
| F25F9cl | `turn/t90_0.35_right` | `R2/42` | - | - | - | 0 / 0 / 0 (falls 20 b / 0 a) |
| F25F9cl | `turn/t90_0.35_right` | `R2/43` | - | - | - | 0 / 0 / 0 (falls 20 b / 0 a) |
| F25F9cl | `turn/t90_0.5_left` | `R2/42` | - | - | - | 0 / 0 / 0 (falls 20 b / 0 a) |
| F25F9cl | `turn/t90_0.5_left` | `R2/43` | - | - | - | 0 / 0 / 0 (falls 20 b / 0 a) |
| F25F9cl | `turn/t90_0.5_right` | `R2/42` | - | - | - | 0 / 0 / 0 (falls 20 b / 0 a) |
| F25F9cl | `turn/t90_0.5_right` | `R2/43` | - | - | - | 0 / 0 / 0 (falls 20 b / 0 a) |
| F25F9cl | `turn/t90_1_left` | `R2/42` | - | - | - | not run |
| F25F9cl | `turn/t90_1_left` | `R2/43` | - | - | - | not run |
| F25F9cl | `turn/t90_1_right` | `R2/42` | - | - | - | not run |
| F25F9cl | `turn/t90_1_right` | `R2/43` | - | - | - | not run |

## 5. Turns: measured medians where the turn was reached

Heading error at the judged instant (limit 10 degrees), the angle the trunk turned (commanded 90), the time to come within 10 degrees, and the toe clearance over the turn's window (limit 20.07 mm).

| Set | Condition | Row | Delay | Up | Heading error (deg) | Turned (deg) | Time to within 10 deg (s) | Toe clearance (mm) | Failed terms (episodes of 20) |
|---|---|---|---|---|---|---|---|---|---|
| V0 | `turn/q90_left` | `R2/42` | 0 ms | 20 | -8.4 (20) | 82.5 | 1.98 | 22.2 (20) | `drive_sustained` 20, `duty` 20, `drive_speed` 8, `toe_clearance_own_swing` 8, `heading` 5, `no_flight` 2 |
| V0 | `turn/q90_left` | `R2/42` | 4 ms | 20 | -5.9 (20) | 85.1 | 1.92 | 20.4 (20) | `drive_sustained` 20, `toe_clearance_own_swing` 7, `duty` 3 |
| V0 | `turn/q90_left` | `R2/42` | 6 ms | 20 | -6.9 (20) | 84.4 | 1.93 | 19.6 (20) | `drive_sustained` 20, `toe_clearance_own_swing` 17, `duty` 10 |
| V0 | `turn/q90_left` | `R2/43` | 0 ms | 20 | -7.3 (20) | 82.6 | 1.94 | 22.0 (20) | `drive_sustained` 20, `duty` 20, `drive_speed` 5, `heading` 5, `toe_clearance_own_swing` 5, `no_flight` 1 |
| V0 | `turn/q90_left` | `R2/43` | 4 ms | 20 | -5.9 (20) | 85.0 | 1.91 | 19.6 (20) | `drive_sustained` 20, `toe_clearance_own_swing` 12, `duty` 1 |
| V0 | `turn/q90_left` | `R2/43` | 6 ms | 20 | -7.0 (20) | 84.4 | 1.93 | 18.4 (20) | `drive_sustained` 20, `toe_clearance_own_swing` 18, `drive_speed` 2, `duty` 2, `heading` 1 |
| V0 | `turn/q90_left` | `R0` | 0 ms | 20 | -10.4 (20) | 80.1 | 2.02 | 6.2 (20) | `drive_sustained` 20, `duty` 20, `no_flight` 20, `toe_clearance_own_swing` 20, `heading` 12 |
| V0 | `turn/q90_left` | `R0` | 4 ms | 19 | -10.5 (19) | 79.7 | 2.05 | 5.6 (19) | `drive_sustained` 19, `duty` 19, `no_flight` 19, `toe_clearance_own_swing` 19, `heading` 15, `survived` 1 |
| V0 | `turn/q90_left` | `R0` | 6 ms | 20 | -11.7 (20) | 79.6 | 2.04 | 5.7 (20) | `drive_sustained` 20, `duty` 20, `no_flight` 20, `toe_clearance_own_swing` 20, `heading` 15 |
| V0 | `turn/q90_left` | `R0` | 10 ms | 20 | -11.7 (20) | 79.5 | 2.05 | 6.0 (20) | `drive_sustained` 20, `duty` 20, `no_flight` 20, `toe_clearance_own_swing` 20, `heading` 18 |
| V0 | `turn/q90_left_mu0.54` | `R2/42` | 0 ms | 20 | -16.8 (20) | 73.5 | 2.12 | 6.4 (20) | `drive_sustained` 20, `duty` 20, `heading` 20, `toe_clearance_own_swing` 18, `drive_speed` 14, `no_flight` 8, `slip_median` 3 |
| V0 | `turn/q90_left_mu0.54` | `R2/42` | 4 ms | 20 | -8.1 (20) | 82.7 | 1.97 | 20.6 (20) | `drive_sustained` 20, `toe_clearance_own_swing` 5 |
| V0 | `turn/q90_left_mu0.54` | `R2/42` | 6 ms | 20 | -10.3 (20) | 81.2 | 2.01 | 19.8 (20) | `drive_sustained` 20, `toe_clearance_own_swing` 13, `heading` 10, `duty` 9 |
| V0 | `turn/q90_left_mu0.54` | `R2/43` | 0 ms | 19 | -14.3 (20) | 76.0 | 2.10 | 3.9 (20) | `drive_sustained` 19, `duty` 19, `heading` 19, `toe_clearance_own_swing` 19, `drive_speed` 16, `no_flight` 4, `slip_median` 3, `survived` 1 |
| V0 | `turn/q90_left_mu0.54` | `R2/43` | 4 ms | 20 | -7.6 (20) | 83.1 | 1.95 | 19.7 (20) | `drive_sustained` 20, `toe_clearance_own_swing` 16, `duty` 4, `heading` 3 |
| V0 | `turn/q90_left_mu0.54` | `R2/43` | 6 ms | 20 | -10.3 (20) | 80.8 | 2.01 | 18.4 (20) | `drive_sustained` 20, `toe_clearance_own_swing` 20, `heading` 11, `duty` 6 |
| V0 | `turn/q90_left_mu0.54` | `R0` | 0 ms | 19 | -18.0 (20) | 72.8 | 2.13 | 6.6 (20) | `drive_sustained` 19, `duty` 19, `heading` 19, `no_flight` 19, `toe_clearance_own_swing` 19, `survived` 1 |
| V0 | `turn/q90_left_mu0.54` | `R0` | 4 ms | 19 | -18.5 (20) | 72.7 | 2.14 | 6.7 (20) | `drive_sustained` 19, `duty` 19, `heading` 19, `no_flight` 19, `toe_clearance_own_swing` 19, `slip_median` 2, `drive_speed` 1, `survived` 1 |
| V0 | `turn/q90_left_mu0.54` | `R0` | 6 ms | 20 | -17.9 (20) | 72.4 | 2.14 | 6.3 (20) | `drive_sustained` 20, `duty` 20, `heading` 20, `no_flight` 20, `toe_clearance_own_swing` 20, `slip_median` 1 |
| V0 | `turn/q90_left_mu0.54` | `R0` | 10 ms | 20 | -18.6 (20) | 72.3 | 2.14 | 6.5 (20) | `drive_sustained` 20, `duty` 20, `heading` 20, `no_flight` 20, `toe_clearance_own_swing` 20, `drive_speed` 1, `slip_median` 1 |
| V0 | `turn/q90_right` | `R2/42` | 0 ms | 20 | 8.2 (20) | -81.6 | 1.97 | 14.2 (20) | `drive_sustained` 20, `duty` 20, `drive_speed` 10, `toe_clearance_own_swing` 10, `heading` 7, `no_flight` 3 |
| V0 | `turn/q90_right` | `R2/42` | 4 ms | 20 | 5.3 (20) | -84.8 | 1.91 | 20.4 (20) | `drive_sustained` 20, `toe_clearance_own_swing` 6, `duty` 4 |
| V0 | `turn/q90_right` | `R2/42` | 6 ms | 20 | 6.0 (20) | -84.2 | 1.92 | 19.4 (20) | `drive_sustained` 20, `toe_clearance_own_swing` 17, `duty` 10, `drive_speed` 2, `heading` 2 |
| V0 | `turn/q90_right` | `R2/43` | 0 ms | 20 | 7.4 (20) | -83.0 | 1.95 | 22.2 (20) | `drive_sustained` 20, `duty` 20, `heading` 5, `drive_speed` 3, `toe_clearance_own_swing` 3, `no_flight` 2 |
| V0 | `turn/q90_right` | `R2/43` | 4 ms | 20 | 5.4 (20) | -85.0 | 1.90 | 20.1 (20) | `drive_sustained` 20, `toe_clearance_own_swing` 11, `duty` 1 |
| V0 | `turn/q90_right` | `R2/43` | 6 ms | 20 | 5.0 (20) | -84.7 | 1.90 | 18.6 (20) | `drive_sustained` 20, `toe_clearance_own_swing` 20, `duty` 3 |
| V0 | `turn/q90_right` | `R0` | 0 ms | 19 | 10.7 (20) | -80.4 | 2.02 | 7.0 (19) | `drive_sustained` 19, `duty` 19, `no_flight` 19, `toe_clearance_own_swing` 19, `heading` 13, `survived` 1 |
| V0 | `turn/q90_right` | `R0` | 4 ms | 19 | 11.2 (19) | -80.0 | 2.03 | 6.5 (19) | `drive_sustained` 19, `duty` 19, `no_flight` 19, `toe_clearance_own_swing` 19, `heading` 14, `survived` 1 |
| V0 | `turn/q90_right` | `R0` | 6 ms | 20 | 10.8 (20) | -80.0 | 2.04 | 5.7 (20) | `drive_sustained` 20, `duty` 20, `no_flight` 20, `toe_clearance_own_swing` 20, `heading` 16 |
| V0 | `turn/q90_right` | `R0` | 10 ms | 20 | 10.6 (20) | -79.7 | 2.02 | 6.0 (20) | `drive_sustained` 20, `duty` 20, `no_flight` 20, `toe_clearance_own_swing` 20, `heading` 16 |
| V0 | `turn/q90_right_mu0.54` | `R2/42` | 0 ms | 20 | 16.3 (20) | -74.5 | 2.11 | 6.3 (20) | `drive_sustained` 20, `duty` 20, `heading` 20, `toe_clearance_own_swing` 19, `drive_speed` 14, `no_flight` 10, `slip_median` 3 |
| V0 | `turn/q90_right_mu0.54` | `R2/42` | 4 ms | 20 | 7.4 (20) | -82.3 | 1.96 | 20.6 (20) | `drive_sustained` 20, `toe_clearance_own_swing` 8, `duty` 6, `heading` 5 |
| V0 | `turn/q90_right_mu0.54` | `R2/42` | 6 ms | 20 | 9.9 (20) | -80.0 | 2.00 | 19.7 (20) | `drive_sustained` 20, `duty` 14, `toe_clearance_own_swing` 13, `heading` 9 |
| V0 | `turn/q90_right_mu0.54` | `R2/43` | 0 ms | 19 | 16.4 (20) | -74.3 | 2.12 | 6.0 (19) | `drive_sustained` 19, `duty` 19, `heading` 17, `toe_clearance_own_swing` 17, `drive_speed` 13, `no_flight` 11, `slip_median` 2, `survived` 1 |
| V0 | `turn/q90_right_mu0.54` | `R2/43` | 4 ms | 20 | 7.4 (20) | -82.7 | 1.95 | 19.5 (20) | `drive_sustained` 20, `toe_clearance_own_swing` 16, `duty` 4, `heading` 2 |
| V0 | `turn/q90_right_mu0.54` | `R2/43` | 6 ms | 20 | 9.0 (20) | -80.7 | 1.98 | 18.4 (20) | `drive_sustained` 20, `toe_clearance_own_swing` 19, `duty` 6, `heading` 5 |
| V0 | `turn/q90_right_mu0.54` | `R2/43` | 10 ms | 1 | 23.6 (2) | -75.2 | 4.57 | -2.1 (1) | `survived` 19, `drive_speed` 1, `drive_sustained` 1, `duty` 1, `heading` 1, `toe_clearance_own_swing` 1 |
| V0 | `turn/q90_right_mu0.54` | `R0` | 0 ms | 20 | 17.9 (20) | -72.6 | 2.13 | 7.6 (20) | `drive_sustained` 20, `duty` 20, `heading` 20, `no_flight` 20, `toe_clearance_own_swing` 20 |
| V0 | `turn/q90_right_mu0.54` | `R0` | 4 ms | 20 | 17.9 (20) | -72.6 | 2.13 | 6.5 (20) | `drive_sustained` 20, `duty` 20, `heading` 20, `no_flight` 20, `toe_clearance_own_swing` 20 |
| V0 | `turn/q90_right_mu0.54` | `R0` | 6 ms | 20 | 18.3 (20) | -72.4 | 2.13 | 6.2 (20) | `drive_sustained` 20, `duty` 20, `heading` 20, `no_flight` 20, `toe_clearance_own_swing` 20 |
| V0 | `turn/q90_right_mu0.54` | `R0` | 10 ms | 19 | 18.3 (20) | -72.0 | 2.15 | 6.5 (19) | `drive_sustained` 19, `duty` 19, `heading` 19, `no_flight` 19, `toe_clearance_own_swing` 19, `survived` 1 |
| V0 | `turn/t90_0.35_left` | `R2/43` | 10 ms | 1 | 1.9 (1) | 91.1 | 4.04 | 4.2 (1) | `survived` 19, `drive_speed` 1, `drive_sustained` 1, `duty` 1, `toe_clearance_own_swing` 1 |
| V0 | `turn/t90_0.35_right` | `R2/43` | 10 ms | 1 | 3.5 (1) | -90.7 | 4.22 | 4.5 (1) | `survived` 19, `drive_speed` 1, `drive_sustained` 1, `duty` 1, `toe_clearance_own_swing` 1 |
| V0 | `turn/t90_0.5_left` | `R2/43` | 10 ms | 1 | -3.8 (1) | 85.4 | 2.83 | 3.6 (1) | `survived` 19, `drive_speed` 1, `drive_sustained` 1, `duty` 1, `toe_clearance_own_swing` 1 |
| V0 | `turn/t90_0.5_right` | `R2/43` | 10 ms | 1 | 8.9 (1) | -85.3 | 3.11 | 3.9 (1) | `survived` 19, `drive_speed` 1, `drive_sustained` 1, `duty` 1, `toe_clearance_own_swing` 1 |
| F25F9cl | `turn/q90_left` | `R2/42` | 0 ms | 20 | -6.0 (20) | 84.8 | 1.93 | 15.6 (20) | `duty` 20, `toe_clearance_own_swing` 20 |
| F25F9cl | `turn/q90_left` | `R2/42` | 4 ms | 20 | -6.2 (20) | 84.8 | 1.93 | 13.9 (20) | `toe_clearance_own_swing` 20 |
| F25F9cl | `turn/q90_left` | `R2/42` | 6 ms | 1 | -8.4 (1) | 83.8 | 1.97 | 12.3 (1) | `survived` 19, `toe_clearance_own_swing` 1 |
| F25F9cl | `turn/q90_left` | `R2/43` | 0 ms | 20 | -5.3 (20) | 85.1 | 1.91 | 15.5 (20) | `duty` 20, `toe_clearance_own_swing` 20, `heading` 1 |
| F25F9cl | `turn/q90_left` | `R2/43` | 4 ms | 19 | -6.5 (20) | 84.6 | 1.94 | 13.6 (19) | `toe_clearance_own_swing` 19, `survived` 1 |
| F25F9cl | `turn/q90_left` | `R2/43` | 6 ms | 1 | -13.1 (1) | 84.3 | 2.06 | 12.6 (1) | `survived` 19, `heading` 1, `toe_clearance_own_swing` 1 |
| F25F9cl | `turn/q90_left` | `R0` | 0 ms | 0 | -7.7 (20) | 82.6 | - | - (0) | `survived` 20 |
| F25F9cl | `turn/q90_left` | `R0` | 4 ms | 0 | -8.1 (20) | 82.6 | - | - (0) | `survived` 20 |
| F25F9cl | `turn/q90_left` | `R0` | 6 ms | 0 | -9.1 (20) | 82.0 | - | - (0) | `survived` 20 |
| F25F9cl | `turn/q90_left` | `R0` | 10 ms | 0 | -8.6 (20) | 82.0 | - | - (0) | `survived` 20 |
| F25F9cl | `turn/q90_left_mu0.54` | `R2/42` | 0 ms | 20 | -7.0 (20) | 83.7 | 1.95 | 15.8 (20) | `duty` 20, `toe_clearance_own_swing` 20, `heading` 2 |
| F25F9cl | `turn/q90_left_mu0.54` | `R2/42` | 4 ms | 20 | -7.0 (20) | 83.9 | 1.95 | 13.9 (20) | `toe_clearance_own_swing` 20 |
| F25F9cl | `turn/q90_left_mu0.54` | `R2/43` | 0 ms | 20 | -6.4 (20) | 84.9 | 1.94 | 15.5 (20) | `duty` 20, `toe_clearance_own_swing` 20 |
| F25F9cl | `turn/q90_left_mu0.54` | `R2/43` | 4 ms | 20 | -7.4 (20) | 84.1 | 1.96 | 13.5 (20) | `toe_clearance_own_swing` 20, `heading` 1 |
| F25F9cl | `turn/q90_left_mu0.54` | `R2/43` | 6 ms | 1 | -12.2 (2) | 79.5 | - | 12.0 (1) | `survived` 19, `heading` 1, `toe_clearance_own_swing` 1 |
| F25F9cl | `turn/q90_left_mu0.54` | `R0` | 0 ms | 0 | -13.9 (20) | 77.4 | - | - (0) | `survived` 20 |
| F25F9cl | `turn/q90_left_mu0.54` | `R0` | 4 ms | 0 | -14.6 (20) | 76.1 | - | - (0) | `survived` 20 |
| F25F9cl | `turn/q90_left_mu0.54` | `R0` | 6 ms | 0 | -14.9 (20) | 76.3 | - | - (0) | `survived` 20 |
| F25F9cl | `turn/q90_left_mu0.54` | `R0` | 10 ms | 0 | -15.3 (20) | 75.8 | - | -14.8 (1) | `survived` 20 |
| F25F9cl | `turn/q90_right` | `R2/42` | 0 ms | 20 | 5.8 (20) | -84.9 | 1.92 | 15.8 (20) | `duty` 20, `toe_clearance_own_swing` 20, `heading` 1 |
| F25F9cl | `turn/q90_right` | `R2/42` | 4 ms | 19 | 5.7 (20) | -84.8 | 1.91 | 14.0 (19) | `toe_clearance_own_swing` 19, `survived` 1 |
| F25F9cl | `turn/q90_right` | `R2/42` | 6 ms | 1 | 5.1 (1) | -84.2 | 1.90 | 13.5 (1) | `survived` 19, `duty` 1, `toe_clearance_own_swing` 1 |
| F25F9cl | `turn/q90_right` | `R2/43` | 0 ms | 20 | 6.2 (20) | -84.8 | 1.93 | 15.6 (20) | `duty` 20, `toe_clearance_own_swing` 20, `heading` 1 |
| F25F9cl | `turn/q90_right` | `R2/43` | 4 ms | 19 | 6.1 (20) | -84.5 | 1.93 | 13.5 (19) | `toe_clearance_own_swing` 19, `survived` 1 |
| F25F9cl | `turn/q90_right` | `R2/43` | 6 ms | 1 | -0.6 (1) | -84.9 | 1.79 | 11.0 (1) | `survived` 19, `toe_clearance_own_swing` 1 |
| F25F9cl | `turn/q90_right` | `R0` | 0 ms | 0 | 7.9 (20) | -82.9 | - | - (0) | `survived` 20 |
| F25F9cl | `turn/q90_right` | `R0` | 4 ms | 0 | 8.1 (20) | -82.6 | - | - (0) | `survived` 20 |
| F25F9cl | `turn/q90_right` | `R0` | 6 ms | 0 | 8.0 (20) | -82.4 | - | - (0) | `survived` 20 |
| F25F9cl | `turn/q90_right` | `R0` | 10 ms | 0 | 9.1 (20) | -81.9 | - | - (0) | `survived` 20 |
| F25F9cl | `turn/q90_right_mu0.54` | `R2/42` | 0 ms | 20 | 7.5 (20) | -83.5 | 1.96 | 15.6 (20) | `toe_clearance_own_swing` 20, `duty` 19, `heading` 4 |
| F25F9cl | `turn/q90_right_mu0.54` | `R2/42` | 4 ms | 20 | 6.0 (20) | -84.1 | 1.93 | 14.0 (20) | `toe_clearance_own_swing` 20 |
| F25F9cl | `turn/q90_right_mu0.54` | `R2/43` | 0 ms | 20 | 6.8 (20) | -84.1 | 1.95 | 15.4 (20) | `duty` 20, `toe_clearance_own_swing` 20, `heading` 1 |
| F25F9cl | `turn/q90_right_mu0.54` | `R2/43` | 4 ms | 20 | 6.9 (20) | -83.6 | 1.95 | 13.5 (20) | `toe_clearance_own_swing` 20, `heading` 1 |
| F25F9cl | `turn/q90_right_mu0.54` | `R2/43` | 6 ms | 1 | 9.3 (2) | -81.4 | - | 10.7 (1) | `survived` 19, `heading` 1, `toe_clearance_own_swing` 1 |
| F25F9cl | `turn/q90_right_mu0.54` | `R0` | 0 ms | 0 | 14.9 (20) | -76.3 | - | - (0) | `survived` 20 |
| F25F9cl | `turn/q90_right_mu0.54` | `R0` | 4 ms | 0 | 14.7 (20) | -75.9 | - | - (0) | `survived` 20 |
| F25F9cl | `turn/q90_right_mu0.54` | `R0` | 6 ms | 0 | 14.6 (20) | -76.4 | - | - (0) | `survived` 20 |
| F25F9cl | `turn/q90_right_mu0.54` | `R0` | 10 ms | 0 | 15.0 (20) | -76.0 | - | - (0) | `survived` 20 |

## 6. Nudges and stops: measured

Nudges: episodes that came back onto the command within 3 s (of 20), and the median time for those. Stops: median touchdowns after the command, strides until forward motion stopped, and distance covered after the command.

| Set | Condition | Row | Delay | Up | Recovered | Median time back (s) | Touchdowns | Strides to stop | Distance (m) | Failed terms (episodes of 20) |
|---|---|---|---|---|---|---|---|---|---|---|
| V0 | `nudge/backward` | `R2/42` | 0 ms | 20 | 14 | 0.00 | - | - | - | `drive_speed` 7, `recovered` 6, `drive_sustained` 2 |
| V0 | `nudge/backward` | `R2/42` | 4 ms | 14 | 13 | 0.00 | - | - | - | `survived` 6, `drive_speed` 1, `recovered` 1 |
| V0 | `nudge/backward` | `R2/42` | 6 ms | 14 | 12 | 0.01 | - | - | - | `survived` 6, `drive_speed` 3, `recovered` 2 |
| V0 | `nudge/backward` | `R2/42` | 10 ms | 0 | 0 | - | - | - | - | `survived` 20 |
| V0 | `nudge/backward` | `R2/43` | 0 ms | 20 | 13 | 0.01 | - | - | - | `recovered` 7, `drive_speed` 6, `drive_sustained` 1 |
| V0 | `nudge/backward` | `R2/43` | 4 ms | 12 | 10 | 0.00 | - | - | - | `survived` 8, `drive_speed` 2, `recovered` 2 |
| V0 | `nudge/backward` | `R2/43` | 6 ms | 9 | 8 | 0.00 | - | - | - | `survived` 11, `drive_speed` 1, `recovered` 1 |
| V0 | `nudge/backward` | `R2/43` | 10 ms | 1 | 1 | 0.00 | - | - | - | `survived` 19, `drive_speed` 1, `drive_sustained` 1 |
| V0 | `nudge/backward` | `R0` | 0 ms | 9 | 4 | 0.71 | - | - | - | `survived` 11, `recovered` 5, `drive_sustained` 4 |
| V0 | `nudge/backward` | `R0` | 4 ms | 8 | 2 | 1.69 | - | - | - | `survived` 12, `recovered` 6, `drive_speed` 1, `drive_sustained` 1 |
| V0 | `nudge/backward` | `R0` | 6 ms | 10 | 3 | 0.49 | - | - | - | `survived` 10, `recovered` 7, `drive_sustained` 5 |
| V0 | `nudge/backward` | `R0` | 10 ms | 10 | 4 | 1.45 | - | - | - | `survived` 10, `recovered` 6, `drive_sustained` 3 |
| V0 | `nudge/backward_mu0.54` | `R2/42` | 0 ms | 18 | 12 | 0.00 | - | - | - | `recovered` 6, `drive_speed` 5, `survived` 2 |
| V0 | `nudge/backward_mu0.54` | `R2/42` | 4 ms | 13 | 13 | 0.00 | - | - | - | `survived` 7 |
| V0 | `nudge/backward_mu0.54` | `R2/42` | 6 ms | 12 | 10 | 0.01 | - | - | - | `survived` 8, `drive_speed` 2, `recovered` 2 |
| V0 | `nudge/backward_mu0.54` | `R2/42` | 10 ms | 0 | 0 | - | - | - | - | `survived` 20 |
| V0 | `nudge/backward_mu0.54` | `R2/43` | 0 ms | 20 | 13 | 0.00 | - | - | - | `drive_speed` 7, `recovered` 7 |
| V0 | `nudge/backward_mu0.54` | `R2/43` | 4 ms | 15 | 11 | 0.00 | - | - | - | `survived` 5, `drive_speed` 4, `recovered` 4 |
| V0 | `nudge/backward_mu0.54` | `R2/43` | 6 ms | 11 | 10 | 0.00 | - | - | - | `survived` 9, `recovered` 1 |
| V0 | `nudge/backward_mu0.54` | `R2/43` | 10 ms | 1 | 0 | - | - | - | - | `survived` 19, `drive_speed` 1, `drive_sustained` 1, `recovered` 1 |
| V0 | `nudge/backward_mu0.54` | `R0` | 0 ms | 5 | 2 | 1.94 | - | - | - | `survived` 15, `recovered` 3 |
| V0 | `nudge/backward_mu0.54` | `R0` | 4 ms | 5 | 0 | - | - | - | - | `survived` 15, `recovered` 5 |
| V0 | `nudge/backward_mu0.54` | `R0` | 6 ms | 6 | 0 | - | - | - | - | `survived` 14, `recovered` 6, `drive_speed` 1 |
| V0 | `nudge/backward_mu0.54` | `R0` | 10 ms | 9 | 4 | 1.11 | - | - | - | `survived` 11, `recovered` 5 |
| V0 | `nudge/backward_x0.375` | `R2/42` | 10 ms | 0 | 0 | - | - | - | - | `survived` 20 |
| V0 | `nudge/backward_x0.375` | `R2/43` | 10 ms | 0 | 0 | - | - | - | - | `survived` 20 |
| V0 | `nudge/backward_x0.5` | `R2/42` | 10 ms | 0 | 0 | - | - | - | - | `survived` 20 |
| V0 | `nudge/backward_x0.5` | `R2/43` | 10 ms | 0 | 0 | - | - | - | - | `survived` 20 |
| V0 | `nudge/forward` | `R2/42` | 0 ms | 19 | 16 | 0.01 | - | - | - | `drive_speed` 5, `recovered` 3, `drive_sustained` 2, `survived` 1 |
| V0 | `nudge/forward` | `R2/42` | 4 ms | 20 | 19 | 0.00 | - | - | - | `drive_speed` 1, `recovered` 1 |
| V0 | `nudge/forward` | `R2/42` | 6 ms | 19 | 19 | 0.00 | - | - | - | `survived` 1 |
| V0 | `nudge/forward` | `R2/42` | 10 ms | 0 | 0 | - | - | - | - | `survived` 20 |
| V0 | `nudge/forward` | `R2/43` | 0 ms | 20 | 18 | 0.01 | - | - | - | `drive_speed` 6, `drive_sustained` 3, `recovered` 2 |
| V0 | `nudge/forward` | `R2/43` | 4 ms | 14 | 13 | 0.00 | - | - | - | `survived` 6, `drive_speed` 2, `recovered` 1 |
| V0 | `nudge/forward` | `R2/43` | 6 ms | 15 | 15 | 0.00 | - | - | - | `survived` 5, `drive_speed` 2 |
| V0 | `nudge/forward` | `R2/43` | 10 ms | 0 | 0 | - | - | - | - | `survived` 20 |
| V0 | `nudge/forward` | `R0` | 0 ms | 3 | 0 | - | - | - | - | `survived` 17, `recovered` 3 |
| V0 | `nudge/forward` | `R0` | 4 ms | 1 | 0 | - | - | - | - | `survived` 19, `recovered` 1 |
| V0 | `nudge/forward` | `R0` | 6 ms | 2 | 1 | 2.92 | - | - | - | `survived` 18, `recovered` 1 |
| V0 | `nudge/forward` | `R0` | 10 ms | 2 | 0 | - | - | - | - | `survived` 18, `recovered` 2 |
| V0 | `nudge/forward_mu0.54` | `R2/42` | 0 ms | 19 | 17 | 0.01 | - | - | - | `drive_speed` 4, `recovered` 2, `survived` 1 |
| V0 | `nudge/forward_mu0.54` | `R2/42` | 4 ms | 19 | 19 | 0.00 | - | - | - | `survived` 1 |
| V0 | `nudge/forward_mu0.54` | `R2/42` | 6 ms | 15 | 15 | 0.00 | - | - | - | `survived` 5 |
| V0 | `nudge/forward_mu0.54` | `R2/42` | 10 ms | 0 | 0 | - | - | - | - | `survived` 20 |
| V0 | `nudge/forward_mu0.54` | `R2/43` | 0 ms | 20 | 16 | 0.01 | - | - | - | `drive_speed` 6, `recovered` 4 |
| V0 | `nudge/forward_mu0.54` | `R2/43` | 4 ms | 13 | 12 | 0.00 | - | - | - | `survived` 7, `drive_speed` 2, `recovered` 1 |
| V0 | `nudge/forward_mu0.54` | `R2/43` | 6 ms | 15 | 14 | 0.00 | - | - | - | `survived` 5, `drive_speed` 3, `recovered` 1 |
| V0 | `nudge/forward_mu0.54` | `R2/43` | 10 ms | 1 | 0 | - | - | - | - | `survived` 19, `drive_speed` 1, `drive_sustained` 1, `recovered` 1 |
| V0 | `nudge/forward_mu0.54` | `R0` | 0 ms | 0 | 0 | - | - | - | - | `survived` 20 |
| V0 | `nudge/forward_mu0.54` | `R0` | 4 ms | 1 | 1 | 1.95 | - | - | - | `survived` 19 |
| V0 | `nudge/forward_mu0.54` | `R0` | 6 ms | 1 | 0 | - | - | - | - | `survived` 19, `recovered` 1 |
| V0 | `nudge/forward_mu0.54` | `R0` | 10 ms | 4 | 0 | - | - | - | - | `survived` 16, `recovered` 4 |
| V0 | `nudge/forward_x0.375` | `R2/42` | 10 ms | 0 | 0 | - | - | - | - | `survived` 20 |
| V0 | `nudge/forward_x0.375` | `R2/43` | 10 ms | 0 | 0 | - | - | - | - | `survived` 20 |
| V0 | `nudge/forward_x0.5` | `R2/42` | 10 ms | 0 | 0 | - | - | - | - | `survived` 20 |
| V0 | `nudge/forward_x0.5` | `R2/43` | 10 ms | 0 | 0 | - | - | - | - | `survived` 20 |
| V0 | `nudge/left` | `R2/42` | 0 ms | 20 | 20 | 0.01 | - | - | - | `drive_speed` 5, `drive_sustained` 3 |
| V0 | `nudge/left` | `R2/42` | 4 ms | 17 | 16 | 0.01 | - | - | - | `drive_speed` 7, `survived` 3, `recovered` 1 |
| V0 | `nudge/left` | `R2/42` | 6 ms | 12 | 12 | 0.50 | - | - | - | `survived` 8, `drive_speed` 7 |
| V0 | `nudge/left` | `R2/42` | 10 ms | 0 | 0 | - | - | - | - | `survived` 20 |
| V0 | `nudge/left` | `R2/43` | 0 ms | 20 | 17 | 0.01 | - | - | - | `drive_speed` 9, `drive_sustained` 5, `recovered` 3 |
| V0 | `nudge/left` | `R2/43` | 4 ms | 14 | 13 | 0.01 | - | - | - | `drive_speed` 8, `survived` 6, `recovered` 1 |
| V0 | `nudge/left` | `R2/43` | 6 ms | 15 | 14 | 0.01 | - | - | - | `drive_speed` 10, `survived` 5, `recovered` 1 |
| V0 | `nudge/left` | `R2/43` | 10 ms | 1 | 1 | 0.00 | - | - | - | `survived` 19, `drive_speed` 1, `drive_sustained` 1 |
| V0 | `nudge/left` | `R0` | 0 ms | 13 | 9 | 0.47 | - | - | - | `survived` 7, `drive_sustained` 4, `recovered` 4 |
| V0 | `nudge/left` | `R0` | 4 ms | 15 | 11 | 0.01 | - | - | - | `survived` 5, `drive_sustained` 4, `recovered` 4 |
| V0 | `nudge/left` | `R0` | 6 ms | 12 | 10 | 0.46 | - | - | - | `survived` 8, `drive_sustained` 4, `recovered` 2 |
| V0 | `nudge/left` | `R0` | 10 ms | 12 | 8 | 0.29 | - | - | - | `survived` 8, `recovered` 4, `drive_sustained` 2 |
| V0 | `nudge/left_mu0.54` | `R2/42` | 0 ms | 19 | 17 | 0.01 | - | - | - | `drive_speed` 4, `recovered` 2, `survived` 1 |
| V0 | `nudge/left_mu0.54` | `R2/42` | 4 ms | 13 | 10 | 0.00 | - | - | - | `survived` 7, `drive_speed` 3, `recovered` 3 |
| V0 | `nudge/left_mu0.54` | `R2/42` | 6 ms | 11 | 2 | 0.00 | - | - | - | `drive_speed` 9, `recovered` 9, `survived` 9 |
| V0 | `nudge/left_mu0.54` | `R2/42` | 10 ms | 0 | 0 | - | - | - | - | `survived` 20 |
| V0 | `nudge/left_mu0.54` | `R2/43` | 0 ms | 17 | 12 | 0.00 | - | - | - | `drive_speed` 6, `recovered` 5, `survived` 3 |
| V0 | `nudge/left_mu0.54` | `R2/43` | 4 ms | 14 | 7 | 0.00 | - | - | - | `drive_speed` 8, `recovered` 7, `survived` 6 |
| V0 | `nudge/left_mu0.54` | `R2/43` | 6 ms | 14 | 12 | 0.01 | - | - | - | `drive_speed` 8, `survived` 6, `recovered` 2 |
| V0 | `nudge/left_mu0.54` | `R2/43` | 10 ms | 0 | 0 | - | - | - | - | `survived` 20 |
| V0 | `nudge/left_mu0.54` | `R0` | 0 ms | 8 | 3 | 0.63 | - | - | - | `survived` 12, `recovered` 5 |
| V0 | `nudge/left_mu0.54` | `R0` | 4 ms | 9 | 5 | 2.57 | - | - | - | `survived` 11, `recovered` 4 |
| V0 | `nudge/left_mu0.54` | `R0` | 6 ms | 10 | 7 | 1.71 | - | - | - | `survived` 10, `recovered` 3 |
| V0 | `nudge/left_mu0.54` | `R0` | 10 ms | 12 | 6 | 1.91 | - | - | - | `survived` 8, `recovered` 6 |
| V0 | `nudge/left_x0.375` | `R2/42` | 10 ms | 0 | 0 | - | - | - | - | `survived` 20 |
| V0 | `nudge/left_x0.375` | `R2/43` | 10 ms | 1 | 1 | 0.00 | - | - | - | `survived` 19, `drive_speed` 1, `drive_sustained` 1 |
| V0 | `nudge/left_x0.5` | `R2/42` | 10 ms | 0 | 0 | - | - | - | - | `survived` 20 |
| V0 | `nudge/left_x0.5` | `R2/43` | 10 ms | 0 | 0 | - | - | - | - | `survived` 20 |
| V0 | `nudge/right` | `R2/42` | 0 ms | 19 | 17 | 0.01 | - | - | - | `drive_speed` 6, `drive_sustained` 2, `recovered` 2, `survived` 1 |
| V0 | `nudge/right` | `R2/42` | 4 ms | 15 | 13 | 0.01 | - | - | - | `drive_speed` 8, `survived` 5, `recovered` 2 |
| V0 | `nudge/right` | `R2/42` | 6 ms | 13 | 12 | 0.33 | - | - | - | `drive_speed` 10, `survived` 7, `recovered` 1 |
| V0 | `nudge/right` | `R2/42` | 10 ms | 0 | 0 | - | - | - | - | `survived` 20 |
| V0 | `nudge/right` | `R2/43` | 0 ms | 20 | 19 | 0.01 | - | - | - | `drive_speed` 7, `drive_sustained` 6, `recovered` 1 |
| V0 | `nudge/right` | `R2/43` | 4 ms | 15 | 15 | 0.00 | - | - | - | `drive_speed` 6, `survived` 5 |
| V0 | `nudge/right` | `R2/43` | 6 ms | 14 | 13 | 0.00 | - | - | - | `drive_speed` 10, `survived` 6, `recovered` 1 |
| V0 | `nudge/right` | `R2/43` | 10 ms | 0 | 0 | - | - | - | - | `survived` 20 |
| V0 | `nudge/right` | `R0` | 0 ms | 17 | 9 | 1.88 | - | - | - | `recovered` 8, `drive_sustained` 6, `survived` 3 |
| V0 | `nudge/right` | `R0` | 4 ms | 15 | 6 | 0.72 | - | - | - | `recovered` 9, `drive_sustained` 8, `survived` 5 |
| V0 | `nudge/right` | `R0` | 6 ms | 16 | 10 | 1.72 | - | - | - | `drive_sustained` 6, `recovered` 6, `survived` 4 |
| V0 | `nudge/right` | `R0` | 10 ms | 14 | 9 | 0.01 | - | - | - | `drive_sustained` 6, `survived` 6, `recovered` 5 |
| V0 | `nudge/right_mu0.54` | `R2/42` | 0 ms | 18 | 14 | 0.01 | - | - | - | `drive_speed` 5, `recovered` 4, `survived` 2 |
| V0 | `nudge/right_mu0.54` | `R2/42` | 4 ms | 13 | 7 | 0.01 | - | - | - | `survived` 7, `drive_speed` 6, `recovered` 6 |
| V0 | `nudge/right_mu0.54` | `R2/42` | 6 ms | 13 | 2 | 0.00 | - | - | - | `drive_speed` 11, `recovered` 11, `survived` 7 |
| V0 | `nudge/right_mu0.54` | `R2/42` | 10 ms | 0 | 0 | - | - | - | - | `survived` 20 |
| V0 | `nudge/right_mu0.54` | `R2/43` | 0 ms | 20 | 18 | 0.01 | - | - | - | `drive_speed` 7, `recovered` 2 |
| V0 | `nudge/right_mu0.54` | `R2/43` | 4 ms | 15 | 14 | 0.01 | - | - | - | `drive_speed` 6, `survived` 5, `recovered` 1 |
| V0 | `nudge/right_mu0.54` | `R2/43` | 6 ms | 15 | 14 | 0.01 | - | - | - | `drive_speed` 8, `survived` 5, `recovered` 1 |
| V0 | `nudge/right_mu0.54` | `R2/43` | 10 ms | 1 | 0 | - | - | - | - | `survived` 19, `drive_speed` 1, `drive_sustained` 1, `recovered` 1 |
| V0 | `nudge/right_mu0.54` | `R0` | 0 ms | 11 | 5 | 1.70 | - | - | - | `survived` 9, `recovered` 6 |
| V0 | `nudge/right_mu0.54` | `R0` | 4 ms | 11 | 7 | 1.69 | - | - | - | `survived` 9, `recovered` 4 |
| V0 | `nudge/right_mu0.54` | `R0` | 6 ms | 13 | 7 | 1.36 | - | - | - | `survived` 7, `recovered` 6 |
| V0 | `nudge/right_mu0.54` | `R0` | 10 ms | 12 | 5 | 1.36 | - | - | - | `survived` 8, `recovered` 7 |
| V0 | `nudge/right_x0.375` | `R2/42` | 10 ms | 0 | 0 | - | - | - | - | `survived` 20 |
| V0 | `nudge/right_x0.375` | `R2/43` | 10 ms | 0 | 0 | - | - | - | - | `survived` 20 |
| V0 | `nudge/right_x0.5` | `R2/42` | 10 ms | 0 | 0 | - | - | - | - | `survived` 20 |
| V0 | `nudge/right_x0.5` | `R2/43` | 10 ms | 0 | 0 | - | - | - | - | `survived` 20 |
| V0 | `stop/one_step` | `R2/42` | 10 ms | 0 | - | - | - | - | - | `survived` 20 |
| V0 | `stop/one_step` | `R2/43` | 10 ms | 1 | - | - | 3 | 1.13 | 0.08 | `survived` 19, `drive_speed` 1, `touchdowns_after_command` 1 |
| V0 | `stop/two_step` | `R2/42` | 0 ms | 20 | - | - | 2 | 1.24 | 0.23 | none |
| V0 | `stop/two_step` | `R2/42` | 4 ms | 20 | - | - | 2 | 1.27 | 0.24 | none |
| V0 | `stop/two_step` | `R2/42` | 6 ms | 20 | - | - | 2 | 1.29 | 0.24 | none |
| V0 | `stop/two_step` | `R2/42` | 10 ms | 0 | - | - | - | - | - | `survived` 20 |
| V0 | `stop/two_step` | `R2/43` | 0 ms | 20 | - | - | 2 | 1.24 | 0.23 | none |
| V0 | `stop/two_step` | `R2/43` | 4 ms | 20 | - | - | 2 | 1.26 | 0.24 | none |
| V0 | `stop/two_step` | `R2/43` | 6 ms | 20 | - | - | 2 | 1.27 | 0.24 | none |
| V0 | `stop/two_step` | `R2/43` | 10 ms | 1 | - | - | 4 | 1.69 | 0.23 | `survived` 19, `drive_speed` 1, `touchdowns_after_command` 1 |
| V0 | `stop/two_step` | `R0` | 0 ms | 20 | - | - | 2 | 1.30 | 0.23 | `touchdowns_after_command` 1 |
| V0 | `stop/two_step` | `R0` | 4 ms | 20 | - | - | 2 | 1.32 | 0.24 | `touchdowns_after_command` 2 |
| V0 | `stop/two_step` | `R0` | 6 ms | 20 | - | - | 2 | 1.30 | 0.24 | none |
| V0 | `stop/two_step` | `R0` | 10 ms | 20 | - | - | 2 | 1.32 | 0.24 | none |
| V0 | `stop/two_step_mu0.54` | `R2/42` | 0 ms | 20 | - | - | 2 | 1.23 | 0.23 | none |
| V0 | `stop/two_step_mu0.54` | `R2/42` | 4 ms | 20 | - | - | 2 | 1.26 | 0.24 | none |
| V0 | `stop/two_step_mu0.54` | `R2/42` | 6 ms | 20 | - | - | 2 | 1.28 | 0.24 | none |
| V0 | `stop/two_step_mu0.54` | `R2/42` | 10 ms | 0 | - | - | - | - | - | `survived` 20 |
| V0 | `stop/two_step_mu0.54` | `R2/43` | 0 ms | 20 | - | - | 2 | 1.24 | 0.23 | none |
| V0 | `stop/two_step_mu0.54` | `R2/43` | 4 ms | 20 | - | - | 2 | 1.26 | 0.24 | none |
| V0 | `stop/two_step_mu0.54` | `R2/43` | 6 ms | 20 | - | - | 2 | 1.25 | 0.24 | none |
| V0 | `stop/two_step_mu0.54` | `R2/43` | 10 ms | 3 | - | - | 4 | 1.64 | 0.23 | `survived` 17, `drive_speed` 3, `touchdowns_after_command` 3 |
| V0 | `stop/two_step_mu0.54` | `R0` | 0 ms | 20 | - | - | 2 | 1.32 | 0.23 | none |
| V0 | `stop/two_step_mu0.54` | `R0` | 4 ms | 20 | - | - | 2 | 1.33 | 0.24 | none |
| V0 | `stop/two_step_mu0.54` | `R0` | 6 ms | 20 | - | - | 2 | 1.32 | 0.24 | none |
| V0 | `stop/two_step_mu0.54` | `R0` | 10 ms | 20 | - | - | 2 | 1.33 | 0.24 | none |
| F25F9cl | `nudge/backward` | `R2/42` | 0 ms | 19 | 19 | 0.00 | - | - | - | `joint_range` 19, `survived` 1 |
| F25F9cl | `nudge/backward` | `R2/42` | 4 ms | 18 | 18 | 0.00 | - | - | - | `joint_range` 18, `survived` 2 |
| F25F9cl | `nudge/backward` | `R2/42` | 6 ms | 0 | 0 | - | - | - | - | `survived` 20 |
| F25F9cl | `nudge/backward` | `R2/42` | 10 ms | 0 | 0 | - | - | - | - | `survived` 20 |
| F25F9cl | `nudge/backward` | `R2/43` | 0 ms | 17 | 17 | 0.00 | - | - | - | `joint_range` 17, `survived` 3 |
| F25F9cl | `nudge/backward` | `R2/43` | 4 ms | 12 | 12 | 0.00 | - | - | - | `joint_range` 12, `survived` 8 |
| F25F9cl | `nudge/backward` | `R2/43` | 6 ms | 0 | 0 | - | - | - | - | `survived` 20 |
| F25F9cl | `nudge/backward` | `R2/43` | 10 ms | 0 | 0 | - | - | - | - | `survived` 20 |
| F25F9cl | `nudge/backward` | `R0` | 0 ms | 9 | 9 | 0.00 | - | - | - | `survived` 11, `joint_range` 9 |
| F25F9cl | `nudge/backward` | `R0` | 4 ms | 11 | 11 | 0.00 | - | - | - | `joint_range` 11, `survived` 9 |
| F25F9cl | `nudge/backward` | `R0` | 6 ms | 8 | 8 | 0.00 | - | - | - | `survived` 12, `joint_range` 8 |
| F25F9cl | `nudge/backward` | `R0` | 10 ms | 9 | 9 | 0.00 | - | - | - | `survived` 11, `joint_range` 9 |
| F25F9cl | `nudge/backward_mu0.54` | `R2/42` | 0 ms | 19 | 19 | 0.00 | - | - | - | `joint_range` 19, `survived` 1 |
| F25F9cl | `nudge/backward_mu0.54` | `R2/42` | 4 ms | 20 | 20 | 0.00 | - | - | - | `joint_range` 20 |
| F25F9cl | `nudge/backward_mu0.54` | `R2/42` | 6 ms | 0 | 0 | - | - | - | - | `survived` 20 |
| F25F9cl | `nudge/backward_mu0.54` | `R2/42` | 10 ms | 0 | 0 | - | - | - | - | `survived` 20 |
| F25F9cl | `nudge/backward_mu0.54` | `R2/43` | 0 ms | 17 | 16 | 0.00 | - | - | - | `joint_range` 17, `survived` 3, `recovered` 1 |
| F25F9cl | `nudge/backward_mu0.54` | `R2/43` | 4 ms | 18 | 17 | 0.00 | - | - | - | `joint_range` 18, `survived` 2, `recovered` 1 |
| F25F9cl | `nudge/backward_mu0.54` | `R2/43` | 6 ms | 0 | 0 | - | - | - | - | `survived` 20 |
| F25F9cl | `nudge/backward_mu0.54` | `R2/43` | 10 ms | 0 | 0 | - | - | - | - | `survived` 20 |
| F25F9cl | `nudge/backward_mu0.54` | `R0` | 0 ms | 13 | 13 | 0.00 | - | - | - | `joint_range` 13, `survived` 7 |
| F25F9cl | `nudge/backward_mu0.54` | `R0` | 4 ms | 9 | 9 | 0.00 | - | - | - | `survived` 11, `joint_range` 9 |
| F25F9cl | `nudge/backward_mu0.54` | `R0` | 6 ms | 9 | 9 | 0.01 | - | - | - | `survived` 11, `joint_range` 9 |
| F25F9cl | `nudge/backward_mu0.54` | `R0` | 10 ms | 9 | 9 | 0.00 | - | - | - | `survived` 11, `joint_range` 9 |
| F25F9cl | `nudge/backward_x0.375` | `R2/42` | 10 ms | 0 | 0 | - | - | - | - | `survived` 20 |
| F25F9cl | `nudge/backward_x0.375` | `R2/43` | 10 ms | 0 | 0 | - | - | - | - | `survived` 20 |
| F25F9cl | `nudge/backward_x0.5` | `R2/42` | 10 ms | 0 | 0 | - | - | - | - | `survived` 20 |
| F25F9cl | `nudge/backward_x0.5` | `R2/43` | 10 ms | 0 | 0 | - | - | - | - | `survived` 20 |
| F25F9cl | `nudge/forward` | `R2/42` | 0 ms | 20 | 20 | 0.00 | - | - | - | `joint_range` 20 |
| F25F9cl | `nudge/forward` | `R2/42` | 4 ms | 20 | 20 | 0.00 | - | - | - | `joint_range` 20 |
| F25F9cl | `nudge/forward` | `R2/42` | 6 ms | 0 | 0 | - | - | - | - | `survived` 20 |
| F25F9cl | `nudge/forward` | `R2/42` | 10 ms | 0 | 0 | - | - | - | - | `survived` 20 |
| F25F9cl | `nudge/forward` | `R2/43` | 0 ms | 19 | 19 | 0.00 | - | - | - | `joint_range` 19, `survived` 1 |
| F25F9cl | `nudge/forward` | `R2/43` | 4 ms | 19 | 18 | 0.00 | - | - | - | `joint_range` 19, `recovered` 1, `survived` 1 |
| F25F9cl | `nudge/forward` | `R2/43` | 6 ms | 0 | 0 | - | - | - | - | `survived` 20 |
| F25F9cl | `nudge/forward` | `R2/43` | 10 ms | 0 | 0 | - | - | - | - | `survived` 20 |
| F25F9cl | `nudge/forward` | `R0` | 0 ms | 7 | 7 | 0.00 | - | - | - | `survived` 13, `joint_range` 7 |
| F25F9cl | `nudge/forward` | `R0` | 4 ms | 7 | 7 | 0.00 | - | - | - | `survived` 13, `joint_range` 7 |
| F25F9cl | `nudge/forward` | `R0` | 6 ms | 8 | 8 | 0.00 | - | - | - | `survived` 12, `joint_range` 8 |
| F25F9cl | `nudge/forward` | `R0` | 10 ms | 7 | 7 | 0.00 | - | - | - | `survived` 13, `joint_range` 7 |
| F25F9cl | `nudge/forward_mu0.54` | `R2/42` | 0 ms | 19 | 19 | 0.00 | - | - | - | `joint_range` 19, `survived` 1 |
| F25F9cl | `nudge/forward_mu0.54` | `R2/42` | 4 ms | 20 | 20 | 0.00 | - | - | - | `joint_range` 20 |
| F25F9cl | `nudge/forward_mu0.54` | `R2/42` | 6 ms | 0 | 0 | - | - | - | - | `survived` 20 |
| F25F9cl | `nudge/forward_mu0.54` | `R2/42` | 10 ms | 0 | 0 | - | - | - | - | `survived` 20 |
| F25F9cl | `nudge/forward_mu0.54` | `R2/43` | 0 ms | 20 | 20 | 0.00 | - | - | - | `joint_range` 20 |
| F25F9cl | `nudge/forward_mu0.54` | `R2/43` | 4 ms | 20 | 20 | 0.00 | - | - | - | `joint_range` 20 |
| F25F9cl | `nudge/forward_mu0.54` | `R2/43` | 6 ms | 0 | 0 | - | - | - | - | `survived` 20 |
| F25F9cl | `nudge/forward_mu0.54` | `R2/43` | 10 ms | 0 | 0 | - | - | - | - | `survived` 20 |
| F25F9cl | `nudge/forward_mu0.54` | `R0` | 0 ms | 7 | 7 | 0.00 | - | - | - | `survived` 13, `joint_range` 7 |
| F25F9cl | `nudge/forward_mu0.54` | `R0` | 4 ms | 10 | 10 | 0.00 | - | - | - | `joint_range` 10, `survived` 10 |
| F25F9cl | `nudge/forward_mu0.54` | `R0` | 6 ms | 7 | 7 | 0.00 | - | - | - | `survived` 13, `joint_range` 7 |
| F25F9cl | `nudge/forward_mu0.54` | `R0` | 10 ms | 7 | 7 | 0.01 | - | - | - | `survived` 13, `joint_range` 7 |
| F25F9cl | `nudge/forward_x0.375` | `R2/42` | 10 ms | 0 | 0 | - | - | - | - | `survived` 20 |
| F25F9cl | `nudge/forward_x0.375` | `R2/43` | 10 ms | 0 | 0 | - | - | - | - | `survived` 20 |
| F25F9cl | `nudge/forward_x0.5` | `R2/42` | 10 ms | 0 | 0 | - | - | - | - | `survived` 20 |
| F25F9cl | `nudge/forward_x0.5` | `R2/43` | 10 ms | 0 | 0 | - | - | - | - | `survived` 20 |
| F25F9cl | `nudge/left` | `R2/42` | 0 ms | 15 | 15 | 0.00 | - | - | - | `joint_range` 15, `survived` 5 |
| F25F9cl | `nudge/left` | `R2/42` | 4 ms | 12 | 11 | 0.00 | - | - | - | `joint_range` 12, `survived` 8, `recovered` 1 |
| F25F9cl | `nudge/left` | `R2/42` | 6 ms | 0 | 0 | - | - | - | - | `survived` 20 |
| F25F9cl | `nudge/left` | `R2/42` | 10 ms | 0 | 0 | - | - | - | - | `survived` 20 |
| F25F9cl | `nudge/left` | `R2/43` | 0 ms | 15 | 15 | 0.00 | - | - | - | `joint_range` 15, `survived` 5 |
| F25F9cl | `nudge/left` | `R2/43` | 4 ms | 12 | 12 | 0.00 | - | - | - | `joint_range` 12, `survived` 8 |
| F25F9cl | `nudge/left` | `R2/43` | 6 ms | 0 | 0 | - | - | - | - | `survived` 20 |
| F25F9cl | `nudge/left` | `R2/43` | 10 ms | 0 | 0 | - | - | - | - | `survived` 20 |
| F25F9cl | `nudge/left` | `R0` | 0 ms | 10 | 10 | 0.00 | - | - | - | `joint_range` 10, `survived` 10 |
| F25F9cl | `nudge/left` | `R0` | 4 ms | 9 | 9 | 0.00 | - | - | - | `survived` 11, `joint_range` 9 |
| F25F9cl | `nudge/left` | `R0` | 6 ms | 10 | 10 | 0.01 | - | - | - | `joint_range` 10, `survived` 10 |
| F25F9cl | `nudge/left` | `R0` | 10 ms | 7 | 6 | 0.00 | - | - | - | `survived` 13, `joint_range` 7, `recovered` 1 |
| F25F9cl | `nudge/left_mu0.54` | `R2/42` | 0 ms | 15 | 15 | 0.00 | - | - | - | `joint_range` 15, `survived` 5 |
| F25F9cl | `nudge/left_mu0.54` | `R2/42` | 4 ms | 11 | 11 | 0.00 | - | - | - | `joint_range` 11, `survived` 9 |
| F25F9cl | `nudge/left_mu0.54` | `R2/42` | 6 ms | 0 | 0 | - | - | - | - | `survived` 20 |
| F25F9cl | `nudge/left_mu0.54` | `R2/42` | 10 ms | 0 | 0 | - | - | - | - | `survived` 20 |
| F25F9cl | `nudge/left_mu0.54` | `R2/43` | 0 ms | 15 | 15 | 0.00 | - | - | - | `joint_range` 15, `survived` 5 |
| F25F9cl | `nudge/left_mu0.54` | `R2/43` | 4 ms | 12 | 12 | 0.00 | - | - | - | `joint_range` 12, `survived` 8 |
| F25F9cl | `nudge/left_mu0.54` | `R2/43` | 6 ms | 0 | 0 | - | - | - | - | `survived` 20 |
| F25F9cl | `nudge/left_mu0.54` | `R2/43` | 10 ms | 0 | 0 | - | - | - | - | `survived` 20 |
| F25F9cl | `nudge/left_mu0.54` | `R0` | 0 ms | 8 | 8 | 0.01 | - | - | - | `survived` 12, `joint_range` 8 |
| F25F9cl | `nudge/left_mu0.54` | `R0` | 4 ms | 10 | 10 | 0.01 | - | - | - | `joint_range` 10, `survived` 10 |
| F25F9cl | `nudge/left_mu0.54` | `R0` | 6 ms | 11 | 11 | 0.00 | - | - | - | `joint_range` 11, `survived` 9 |
| F25F9cl | `nudge/left_mu0.54` | `R0` | 10 ms | 8 | 8 | 0.01 | - | - | - | `survived` 12, `joint_range` 8 |
| F25F9cl | `nudge/left_x0.375` | `R2/42` | 10 ms | 0 | 0 | - | - | - | - | `survived` 20 |
| F25F9cl | `nudge/left_x0.375` | `R2/43` | 10 ms | 0 | 0 | - | - | - | - | `survived` 20 |
| F25F9cl | `nudge/left_x0.5` | `R2/42` | 10 ms | 0 | 0 | - | - | - | - | `survived` 20 |
| F25F9cl | `nudge/left_x0.5` | `R2/43` | 10 ms | 0 | 0 | - | - | - | - | `survived` 20 |
| F25F9cl | `nudge/right` | `R2/42` | 0 ms | 15 | 15 | 0.00 | - | - | - | `joint_range` 15, `survived` 5 |
| F25F9cl | `nudge/right` | `R2/42` | 4 ms | 6 | 6 | 0.00 | - | - | - | `survived` 14, `joint_range` 6 |
| F25F9cl | `nudge/right` | `R2/42` | 6 ms | 0 | 0 | - | - | - | - | `survived` 20 |
| F25F9cl | `nudge/right` | `R2/42` | 10 ms | 0 | 0 | - | - | - | - | `survived` 20 |
| F25F9cl | `nudge/right` | `R2/43` | 0 ms | 15 | 15 | 0.00 | - | - | - | `joint_range` 15, `survived` 5 |
| F25F9cl | `nudge/right` | `R2/43` | 4 ms | 11 | 11 | 0.00 | - | - | - | `joint_range` 11, `survived` 9, `drive_speed` 1 |
| F25F9cl | `nudge/right` | `R2/43` | 6 ms | 0 | 0 | - | - | - | - | `survived` 20 |
| F25F9cl | `nudge/right` | `R2/43` | 10 ms | 0 | 0 | - | - | - | - | `survived` 20 |
| F25F9cl | `nudge/right` | `R0` | 0 ms | 5 | 5 | 0.00 | - | - | - | `survived` 15, `joint_range` 5 |
| F25F9cl | `nudge/right` | `R0` | 4 ms | 4 | 4 | 0.00 | - | - | - | `survived` 16, `joint_range` 4 |
| F25F9cl | `nudge/right` | `R0` | 6 ms | 4 | 4 | 0.00 | - | - | - | `survived` 16, `joint_range` 4 |
| F25F9cl | `nudge/right` | `R0` | 10 ms | 6 | 6 | 0.00 | - | - | - | `survived` 14, `joint_range` 6 |
| F25F9cl | `nudge/right_mu0.54` | `R2/42` | 0 ms | 15 | 15 | 0.00 | - | - | - | `joint_range` 15, `survived` 5 |
| F25F9cl | `nudge/right_mu0.54` | `R2/42` | 4 ms | 6 | 6 | 0.00 | - | - | - | `survived` 14, `joint_range` 6 |
| F25F9cl | `nudge/right_mu0.54` | `R2/42` | 6 ms | 0 | 0 | - | - | - | - | `survived` 20 |
| F25F9cl | `nudge/right_mu0.54` | `R2/42` | 10 ms | 0 | 0 | - | - | - | - | `survived` 20 |
| F25F9cl | `nudge/right_mu0.54` | `R2/43` | 0 ms | 15 | 15 | 0.00 | - | - | - | `joint_range` 15, `survived` 5 |
| F25F9cl | `nudge/right_mu0.54` | `R2/43` | 4 ms | 11 | 11 | 0.00 | - | - | - | `joint_range` 11, `survived` 9 |
| F25F9cl | `nudge/right_mu0.54` | `R2/43` | 6 ms | 0 | 0 | - | - | - | - | `survived` 20 |
| F25F9cl | `nudge/right_mu0.54` | `R2/43` | 10 ms | 0 | 0 | - | - | - | - | `survived` 20 |
| F25F9cl | `nudge/right_mu0.54` | `R0` | 0 ms | 7 | 7 | 0.00 | - | - | - | `survived` 13, `joint_range` 7 |
| F25F9cl | `nudge/right_mu0.54` | `R0` | 4 ms | 8 | 8 | 0.00 | - | - | - | `survived` 12, `joint_range` 8 |
| F25F9cl | `nudge/right_mu0.54` | `R0` | 6 ms | 7 | 7 | 0.00 | - | - | - | `survived` 13, `joint_range` 7 |
| F25F9cl | `nudge/right_mu0.54` | `R0` | 10 ms | 8 | 8 | 0.00 | - | - | - | `survived` 12, `joint_range` 8 |
| F25F9cl | `nudge/right_x0.375` | `R2/42` | 10 ms | 0 | 0 | - | - | - | - | `survived` 20 |
| F25F9cl | `nudge/right_x0.375` | `R2/43` | 10 ms | 0 | 0 | - | - | - | - | `survived` 20 |
| F25F9cl | `nudge/right_x0.5` | `R2/42` | 10 ms | 0 | 0 | - | - | - | - | `survived` 20 |
| F25F9cl | `nudge/right_x0.5` | `R2/43` | 10 ms | 0 | 0 | - | - | - | - | `survived` 20 |
| F25F9cl | `stop/one_step` | `R2/42` | 10 ms | 0 | - | - | - | - | - | `survived` 20 |
| F25F9cl | `stop/one_step` | `R2/43` | 10 ms | 0 | - | - | - | - | - | `survived` 20 |
| F25F9cl | `stop/two_step` | `R2/42` | 0 ms | 20 | - | - | 2 | 1.18 | 0.27 | `joint_range` 20 |
| F25F9cl | `stop/two_step` | `R2/42` | 4 ms | 20 | - | - | 2 | 1.20 | 0.27 | `joint_range` 20 |
| F25F9cl | `stop/two_step` | `R2/42` | 6 ms | 1 | - | - | 2 | 1.24 | 0.27 | `survived` 19, `joint_range` 1 |
| F25F9cl | `stop/two_step` | `R2/42` | 10 ms | 0 | - | - | - | - | - | `survived` 20 |
| F25F9cl | `stop/two_step` | `R2/43` | 0 ms | 20 | - | - | 2 | 1.18 | 0.27 | `joint_range` 20 |
| F25F9cl | `stop/two_step` | `R2/43` | 4 ms | 20 | - | - | 2 | 1.19 | 0.27 | `joint_range` 20 |
| F25F9cl | `stop/two_step` | `R2/43` | 6 ms | 1 | - | - | 2 | 1.29 | 0.21 | `survived` 19, `joint_range` 1 |
| F25F9cl | `stop/two_step` | `R2/43` | 10 ms | 0 | - | - | - | - | - | `survived` 20 |
| F25F9cl | `stop/two_step` | `R0` | 0 ms | 20 | - | - | 2 | 1.27 | 0.27 | `joint_range` 20 |
| F25F9cl | `stop/two_step` | `R0` | 4 ms | 20 | - | - | 2 | 1.27 | 0.27 | `joint_range` 20 |
| F25F9cl | `stop/two_step` | `R0` | 6 ms | 20 | - | - | 2 | 1.27 | 0.27 | `joint_range` 20 |
| F25F9cl | `stop/two_step` | `R0` | 10 ms | 20 | - | - | 2 | 1.27 | 0.28 | `joint_range` 20 |
| F25F9cl | `stop/two_step_mu0.54` | `R2/42` | 0 ms | 20 | - | - | 2 | 1.18 | 0.27 | `joint_range` 20 |
| F25F9cl | `stop/two_step_mu0.54` | `R2/42` | 4 ms | 20 | - | - | 2 | 1.20 | 0.27 | `joint_range` 20 |
| F25F9cl | `stop/two_step_mu0.54` | `R2/42` | 6 ms | 0 | - | - | - | - | - | `survived` 20 |
| F25F9cl | `stop/two_step_mu0.54` | `R2/42` | 10 ms | 0 | - | - | - | - | - | `survived` 20 |
| F25F9cl | `stop/two_step_mu0.54` | `R2/43` | 0 ms | 20 | - | - | 2 | 1.18 | 0.27 | `joint_range` 20 |
| F25F9cl | `stop/two_step_mu0.54` | `R2/43` | 4 ms | 20 | - | - | 2 | 1.20 | 0.27 | `joint_range` 20 |
| F25F9cl | `stop/two_step_mu0.54` | `R2/43` | 6 ms | 2 | - | - | 4 | 2.30 | 0.27 | `survived` 18, `joint_range` 2, `touchdowns_after_command` 2 |
| F25F9cl | `stop/two_step_mu0.54` | `R2/43` | 10 ms | 0 | - | - | - | - | - | `survived` 20 |
| F25F9cl | `stop/two_step_mu0.54` | `R0` | 0 ms | 20 | - | - | 2 | 1.27 | 0.27 | `joint_range` 20 |
| F25F9cl | `stop/two_step_mu0.54` | `R0` | 4 ms | 20 | - | - | 2 | 1.27 | 0.27 | `joint_range` 20 |
| F25F9cl | `stop/two_step_mu0.54` | `R0` | 6 ms | 20 | - | - | 2 | 1.27 | 0.27 | `joint_range` 20 |
| F25F9cl | `stop/two_step_mu0.54` | `R0` | 10 ms | 20 | - | - | 2 | 1.28 | 0.28 | `joint_range` 20 |

## 7. Compute

| Task | Budget (core-hours) | Used (core-hours) | Units started | Units cut |
|---|---|---|---|---|
| 3 | 16 | 4.25 | 97 | 0 |
| 4 | 12 | 5.51 | 14 | 0 |

Records read: 592 (582 done, 10 written as not run).
