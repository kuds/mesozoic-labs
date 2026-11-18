# The first builds of the three 75 cm sets under the rules as amended on 15 Nov (morning)

Written by the lead on 15 Nov 2026. Prescribed motion only: these are the reference builder's own gates on the
references, read before any closed-loop episode on a Stage 2 set. Nothing here is frozen and nothing is a record;
the set directories are in the worktree `wt/wp-s` (the arrays are git-ignored). Made from each `set.json` by a
script, not typed. Limits: warm-up ZMP 2.0 mm and first-stride consistency 2.5 mm (A7, `join_command` and
`zmp_consistency`); ZMP after the first stride 4.0 mm; joint range 0.10 rad on every joint but the thigh twist
(A12, and A11's stride rule since the morning's amendment); thigh twist 0.85 of its range at the kinematic pass.

## V0 on 75-M-H-10

The set builder's build (commit f5a0bf5 of wt/wp-s), screened and verified.

- Built walks with an entry: model, rules, references.
- Built moves with an entry: start, nominal_to_v0.4, nominal_to_v0.3, nominal_to_v0.35, nominal_to_v0.45, nominal_to_v0.5, stop2_v0.4, stop1_v0.4, turn90_0.35_left, turn90_0.35_right, turn90_0.5_left, turn90_0.5_right, turn90_pi4_left, turn90_pi4_right, turn90_1_left, turn90_1_right.

| Item | Outcome | Warm-up ZMP (mm) | First-stride consistency (mm) | ZMP after (mm) | Tightest joint but the twist (rad) |
|---|---|---|---|---|---|
| `turn90_0.5_left` | no reference: fails join_command; no reference within the gates (entry gates: join_command) | 2.76 | 1.51 | 2.44 | 0.111 |
| `turn90_0.5_right` | no reference: fails join_command; no reference within the gates (entry gates: join_command) | 2.76 | 1.51 | 2.44 | 0.111 |
| `turn90_pi4_left` | no reference: fails join_command, zmp_consistency; no reference within the gates (entry gates: join_command, zmp_consistency) | 2.26 | 2.63 | 2.77 | 0.106 |
| `turn90_pi4_right` | no reference: fails join_command, zmp_consistency; no reference within the gates (entry gates: join_command, zmp_consistency) | 2.26 | 2.63 | 2.77 | 0.106 |
| `turn90_1_left` | no reference within the thigh twist and the leg drives: no stride factor of the grid passes the kinematic rule (thigh twist at most 0.85 of its range, | | | | |
| `turn90_1_right` | no reference within the thigh twist and the leg drives: no stride factor of the grid passes the kinematic rule (thigh twist at most 0.85 of its range, | | | | |

| Turn rate | Stride factor chosen | Factors tried: factor, thigh-twist use, tightest other joint (rad), IK ok |
|---|---|---|
| 0.35 | 1.0 | 1, 0.449, 0.122, yes |
| 0.5 | 0.9 | 1, 0.633, 0.090, yes; 0.95, 0.602, 0.099, yes; 0.9, 0.569, 0.107, yes |
| pi4 | 0.7 | 1, 1.000, 0.036, no; 0.95, 0.983, 0.047, yes; 0.9, 0.928, 0.058, yes; 0.85, 0.873, 0.069, yes; 0.8, 0.818, 0.080, yes; 0.75, 0.762, 0.091, yes; 0.7, 0.707, 0.101, yes |
| 1 | None | 1, 1.000, 0.001, no; 0.95, 1.000, 0.013, no; 0.9, 1.000, 0.026, no; 0.85, 1.000, 0.038, no; 0.8, 1.000, 0.051, no; 0.75, 0.994, 0.063, yes; 0.7, 0.920, 0.075, yes; 0.65, 0.847, 0.087, yes; 0.6, 0.775, 0.099, yes |

## F25F9cl on 75-M-H-10

The lead's build with the builder's commands; the build stopped at the start (A8).

- Built walks with an entry: model, rules, references.
- Built moves with an entry: start, nominal_to_v0.4, nominal_to_v0.3.

| Item | Outcome | Warm-up ZMP (mm) | First-stride consistency (mm) | ZMP after (mm) | Tightest joint but the twist (rad) |
|---|---|---|---|---|---|
| `start` | no reference: fails no entry gate (see the reason); no transition within the gates | not read | not read | not read | 0.095 |

| Turn rate | Stride factor chosen | Factors tried: factor, thigh-twist use, tightest other joint (rad), IK ok |
|---|---|---|

## F25F9cl, scratch on 75-M-H-10

A scratch build of some items past the stopped start, for information only.

- Built walks with an entry: model, rules, references.
- Built moves with an entry: stop2_v0.4, nominal_to_v0.4, stop1_v0.4, turn90_pi4_left, turn90_pi4_right, turn90_0.35_left, turn90_0.35_right.

| Item | Outcome | Warm-up ZMP (mm) | First-stride consistency (mm) | ZMP after (mm) | Tightest joint but the twist (rad) |
|---|---|---|---|---|---|
| `turn90_pi4_left` | no reference within the thigh twist and the leg drives: no stride factor of the grid passes the kinematic rule (thigh twist at most 0.85 of its range, | | | | |
| `turn90_pi4_right` | no reference within the thigh twist and the leg drives: no stride factor of the grid passes the kinematic rule (thigh twist at most 0.85 of its range, | | | | |
| `turn90_0.35_left` | no reference: fails join_command; no reference within the gates (entry gates: join_command) | 2.05 | 1.24 | 2.10 | 0.111 |
| `turn90_0.35_right` | no reference: fails join_command; no reference within the gates (entry gates: join_command) | 2.05 | 1.24 | 2.14 | 0.111 |

| Turn rate | Stride factor chosen | Factors tried: factor, thigh-twist use, tightest other joint (rad), IK ok |
|---|---|---|
| pi4 | None | 1, 1.000, -0.016, no; 0.95, 1.000, -0.003, no; 0.9, 1.000, 0.009, no; 0.85, 1.000, 0.022, no; 0.8, 1.000, 0.035, no; 0.75, 0.961, 0.048, yes; 0.7, 0.890, 0.060, yes; 0.65, 0.821, 0.073, yes; 0.6, 0.751, 0.085, yes |
| 0.35 | 0.85 | 1, 0.534, 0.083, yes; 0.95, 0.509, 0.092, yes; 0.9, 0.483, 0.100, yes; 0.85, 0.457, 0.108, yes |

## F25F9 on 75-M-H-10

The lead's build with the builder's commands.

- Built walks with an entry: model, rules, references.
- Built moves with an entry: start, nominal_to_v0.4, nominal_to_v0.3, nominal_to_v0.35, nominal_to_v0.45, nominal_to_v0.5, stop2_v0.4, stop1_v0.4, turn90_0.35_left, turn90_0.35_right, turn90_0.5_left, turn90_0.5_right, turn90_pi4_left, turn90_pi4_right, turn90_1_left, turn90_1_right.

| Item | Outcome | Warm-up ZMP (mm) | First-stride consistency (mm) | ZMP after (mm) | Tightest joint but the twist (rad) |
|---|---|---|---|---|---|
| `nominal` | dropped from the ladder: over clearance swing_clearance 0.0382: blocks the freeze (clearance swing_clearance); a walk that is not frozen is dropped fr | | | | |
| `v0.3` | dropped from the ladder: over clearance swing_clearance 0.0382: blocks the freeze (clearance swing_clearance); a walk that is not frozen is dropped fr | | | | |
| `v0.35` | dropped from the ladder: over clearance swing_clearance 0.0382: blocks the freeze (clearance swing_clearance); a walk that is not frozen is dropped fr | | | | |
| `start` | the walk nominal is not a rung of the set (nominal: dropped from the ladder: over clearance swing_clearance 0.0382: blocks the freeze (clearance swing | | | | |
| `nominal_to_v0.4` | the walk nominal is not a rung of the set (nominal: dropped from the ladder: over clearance swing_clearance 0.0382: blocks the freeze (clearance swing | | | | |
| `nominal_to_v0.3` | the walk nominal, v0.3 is not a rung of the set (nominal: dropped from the ladder: over clearance swing_clearance 0.0382: blocks the freeze (clearance | | | | |
| `nominal_to_v0.35` | the walk nominal, v0.35 is not a rung of the set (nominal: dropped from the ladder: over clearance swing_clearance 0.0382: blocks the freeze (clearanc | | | | |
| `nominal_to_v0.45` | the walk nominal is not a rung of the set (nominal: dropped from the ladder: over clearance swing_clearance 0.0382: blocks the freeze (clearance swing | | | | |
| `nominal_to_v0.5` | the walk nominal is not a rung of the set (nominal: dropped from the ladder: over clearance swing_clearance 0.0382: blocks the freeze (clearance swing | | | | |
| `stop2_v0.4` | no reference: fails join_command, zmp_consistency; no transition within the gates | 2.08 | 2.55 | 0.40 | 0.133 |
| `turn90_0.35_left` | no reference: fails join_command, zmp_consistency; no reference within the gates (entry gates: join_command, zmp_consistency) | 3.12 | 3.51 | 3.93 | 0.110 |
| `turn90_0.35_right` | no reference: fails join_command, zmp_consistency; no reference within the gates (entry gates: join_command, zmp_consistency) | 3.12 | 3.51 | 3.93 | 0.110 |
| `turn90_0.5_left` | no reference: fails join_command, zmp_consistency; no reference within the gates (entry gates: join_command, zmp_consistency) | 4.01 | 2.79 | 3.69 | 0.109 |
| `turn90_0.5_right` | no reference: fails join_command, zmp_consistency; no reference within the gates (entry gates: join_command, zmp_consistency) | 4.01 | 2.79 | 3.69 | 0.109 |
| `turn90_pi4_left` | no reference within the thigh twist and the leg drives: no stride factor of the grid passes the kinematic rule (thigh twist at most 0.85 of its range, | | | | |
| `turn90_pi4_right` | no reference within the thigh twist and the leg drives: no stride factor of the grid passes the kinematic rule (thigh twist at most 0.85 of its range, | | | | |
| `turn90_1_left` | no reference within the thigh twist and the leg drives: no stride factor of the grid passes the kinematic rule (thigh twist at most 0.85 of its range, | | | | |
| `turn90_1_right` | no reference within the thigh twist and the leg drives: no stride factor of the grid passes the kinematic rule (thigh twist at most 0.85 of its range, | | | | |

| Turn rate | Stride factor chosen | Factors tried: factor, thigh-twist use, tightest other joint (rad), IK ok |
|---|---|---|
| 0.35 | 0.9 | 1, 0.509, 0.091, yes; 0.95, 0.485, 0.099, yes; 0.9, 0.461, 0.107, yes |
| 0.5 | 0.75 | 1, 0.741, 0.056, yes; 0.95, 0.705, 0.065, yes; 0.9, 0.668, 0.075, yes; 0.85, 0.630, 0.085, yes; 0.8, 0.591, 0.095, yes; 0.75, 0.554, 0.105, yes |
| pi4 | None | 1, 1.000, -0.002, no; 0.95, 1.000, 0.010, no; 0.9, 1.000, 0.022, no; 0.85, 1.000, 0.034, no; 0.8, 0.965, 0.047, yes; 0.75, 0.899, 0.059, yes; 0.7, 0.834, 0.072, yes; 0.65, 0.769, 0.084, yes; 0.6, 0.704, 0.096, yes |
| 1 | None | 1, 1.000, -0.041, no; 0.95, 1.000, -0.027, no; 0.9, 1.000, -0.013, no; 0.85, 1.000, 0.000, no; 0.8, 1.000, 0.015, no; 0.75, 1.000, 0.029, no; 0.7, 1.000, 0.043, no; 0.65, 1.000, 0.057, no; 0.6, 0.916, 0.071, yes |

## What the lead reads from it

- The joint-range gate of 0.10 rad on moves blocks F25F9cl's start (hip roll 0.095 rad), which under A8 stops the set,
  and with A11's morning amendment leaves F25F9cl and F25F9 no stride factor for the quick turn. Stage 1's own
  transitions on this body reach 0.057 rad (the set screen's review), and decision 14 judges the closed loop against
  the reference's own smallest margin less 0.02 rad.
- The warm-up ZMP reads 1.35 mm at V0's own timing and 2.05 to 4.01 mm elsewhere, while the warm-up CoM stays within
  0.09 mm of the walk's: the reading follows the preview controller's grid and the walk, not a difference in the
  motion before the command. The limit (2.0 mm) and the first-stride limit (2.5 mm) were set on V0's prototypes.
- F25F9's walks at 0.35 m/s and below have a planned mid-swing pad clearance of 0.0382 H against 0.04 H, so they are
  dropped, and the set has no start from standing.
- Under the rules a turn left without a reference goes to the owner (A11). The lead put the choice to the owner on
  15 Nov; the answer and any amendment are recorded in `STAGE2_75_PLAN.md` and `DECISION_LOG.md`.
