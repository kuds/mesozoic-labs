# Stage 1 tables

Machine-made by `python -m designlab.stage1 report` at 2026-11-10T01:18:21+00:00 from the records in `stage1/runs/2026-11-08` (designlab/stage1/report.py; the rules are in stage1_tables.json, which is also each body's summary, plan section 7's <body>/summary.json, decision 34).  Missing records show as 'not run'.  The narrative report is STAGE1_PROBE.md.

## 60-N-T-10

### Verdicts by controller

| Controller | shown | provisional | body-limited | controller-limited | not attributed | fails by falls (R4: not attributed) | incomplete | not run |
|---|---|---|---|---|---|---|---|---|
| R2 (shown) | 14 | 0 | 0 | 6 | 0 | 0 | 0 | 0 |
| R3 (not judged: judged with R2) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| R4 | 0 | 3 | 10 | 7 | 0 | 2 | 0 | 0 |

Counts over the conditions gated for each controller's rows.
R4 has one seed (decisions 3 and 28): its verdicts are at most provisional (policy_rows.R4_VERDICT_RULE); its record on each gated condition: FAIL 19, pass 3.
60-N-T-10's R3 is judged with R2 (R3/42 stopped at 2015232 steps: G6b failed again after the fix (FIX_RULE) (decision 32); R3/43 stopped at 2015232 steps: the body's R3 is judged with R2 (train.R2_JUDGED_RULE, decision 38): its seed 42 run was stopped by FIX_RULE at 2015232 steps (G6b failed again after the fix), so this run cannot change a verdict; stopped to give the 75 cm runs the cores (decision 32)): the grid shows R2's verdicts (train.R2_JUDGED_RULE).

G7 pass: sway 0.1677 H (Stage 0 0.1677, plan 0.168), cadence 1.9612 steps/s (Stage 0 1.9612, plan 1.96).

### Verdict grid

Verdicts shown: R2: 60-N-T-10's R3 is judged with R2 (R3/42 stopped at 2015232 steps: G6b failed again after the fix (FIX_RULE) (decision 32); R3/43 stopped at 2015232 steps: the body's R3 is judged with R2 (train.R2_JUDGED_RULE, decision 38): its seed 42 run was stopped by FIX_RULE at 2015232 steps (G6b failed again after the fix), so this run cannot change a verdict; stopped to give the 75 cm runs the cores (decision 32); train.R2_JUDGED_RULE).  Cells: result, survived,succeeded of 20 (their one-sided 95 % Clopper-Pearson lower bounds); '(rep.)' reported, not gated, for that row; '*' interim (listed below each table); 'not run' no record; 'no stance' the floor has no settled stance for the body (panels.NOT_RUN_RULE); 'n/a' the row does not run there; R4's verdict is at most provisional (one seed), and its falls are never attributed (by design: decision 20's attribution is for R2 and R3).

#### Stand

| Condition | Gated for | R0 | R1 | R2/42 | R2/43 | R3/42 | R3/43 | R4/42 | R2 verdict (shown) | R3 verdict (not judged: stopped) | R4 verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| hard | all | pass 20,20 (0.86,0.86) | pass 20,20 (0.86,0.86) | pass 20,20 (0.86,0.86) | pass 20,20 (0.86,0.86) | not run | not run | pass 20,20 (0.86,0.86) | shown | not run | provisional |
| robot_floor_carpet | all | n/a | n/a | pass 20,20 (0.86,0.86) | pass 20,20 (0.86,0.86) | not run | not run | pass 20,20 (0.86,0.86) | shown | not run | provisional |
| low | all | n/a | n/a | pass 20,20 (0.86,0.86) | pass 20,20 (0.86,0.86) | not run | not run | pass 20,20 (0.86,0.86) | shown | not run | provisional |

#### Walk

| Condition | Gated for | R0 | R1 | R2/42 | R2/43 | R3/42 | R3/43 | R4/42 | R2 verdict (shown) | R3 verdict (not judged: stopped) | R4 verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| nominal | all | FAIL 3,3 (0.04,0.04) | FAIL 18,18 (0.72,0.72) | pass 20,18 (0.86,0.72) | pass 20,20 (0.86,0.86) | not run | not run | FAIL 20,0 (0.86,0.00) | shown | not run | body-limited |
| brisk | all | FAIL 0,0 (0.00,0.00) | FAIL 15,15 (0.54,0.54) | FAIL 17,17 (0.66,0.66) | FAIL 0,0 (0.00,0.00) | not run | not run | FAIL 20,0 (0.86,0.00) | controller-limited | not run | body-limited |
| fr05 | all | FAIL 0,0 (0.00,0.00) | FAIL 1,1 (0.00,0.00) | FAIL 1,0 (0.00,0.00) | FAIL 0,0 (0.00,0.00) | not run | not run | FAIL 20,0 (0.86,0.00) | controller-limited | not run | body-limited |
| nominal_mu0.54 | all | FAIL 5,5 (0.10,0.10) | FAIL 14,14 (0.49,0.49) | pass 20,19 (0.86,0.78) | pass 20,19 (0.86,0.78) | not run | not run | FAIL 20,0 (0.86,0.00) | shown | not run | body-limited |
| brisk_mu0.54 | none (reported) | FAIL 0,0 (0.00,0.00) (rep.) | FAIL 14,12 (0.49,0.39) (rep.) | FAIL 15,11 (0.54,0.35) (rep.) | FAIL 0,0 (0.00,0.00) (rep.) | not run | not run | FAIL 20,0 (0.86,0.00) (rep.) | controller-limited (rep.) | not run (rep.) | body-limited (rep.) |
| fr05_mu0.54 | none (reported) | FAIL 0,0 (0.00,0.00) (rep.) | FAIL 0,0 (0.00,0.00) (rep.) | FAIL 0,0 (0.00,0.00) (rep.) | FAIL 0,0 (0.00,0.00) (rep.) | not run | not run | FAIL 20,0 (0.86,0.00) (rep.) | controller-limited (rep.) | not run (rep.) | body-limited (rep.) |

- R0 walk/nominal: not attributed (falls; attribution is applied to R2, R3 rows) (by design)
- R1 walk/nominal: not attributed (falls; attribution is applied to R2, R3 rows) (by design)
- R4/42 walk/nominal: body-limited: slip_median, slip_worst; checkpoint 1253376
- R0 walk/brisk: not attributed (falls; attribution is applied to R2, R3 rows) (by design)
- R1 walk/brisk: not attributed (falls; attribution is applied to R2, R3 rows) (by design)
- R2/42 walk/brisk: controller-limited (decision 20; 3 falls: the sign test needs 5 for p < 0.05, so no limit was lifted; untestable candidates: cop_at_pad_edge_with_ankle_saturated)
- R2/43 walk/brisk: controller-limited (decision 20; untestable candidates: cop_at_pad_edge_with_ankle_saturated)
- R4/42 walk/brisk: body-limited: slip_median, slip_worst; checkpoint 1253376
- R0 walk/fr05: not attributed (falls; attribution is applied to R2, R3 rows) (by design)
- R1 walk/fr05: not attributed (falls; attribution is applied to R2, R3 rows) (by design)
- R2/42 walk/fr05: controller-limited (decision 20; untestable candidates: cop_at_pad_edge_with_ankle_saturated); also controller-limited
- R2/43 walk/fr05: controller-limited (decision 20; untestable candidates: cop_at_pad_edge_with_ankle_saturated)
- R4/42 walk/fr05: body-limited: slip_median, slip_worst; checkpoint 1253376
- R0 walk/nominal_mu0.54: not attributed (falls; attribution is applied to R2, R3 rows) (by design)
- R1 walk/nominal_mu0.54: not attributed (falls; attribution is applied to R2, R3 rows) (by design)
- R4/42 walk/nominal_mu0.54: body-limited: slip_median, slip_worst; checkpoint 1253376
- R0 walk/brisk_mu0.54: not attributed (falls; attribution is applied to R2, R3 rows) (by design)
- R1 walk/brisk_mu0.54: not attributed (falls; attribution is applied to R2, R3 rows) (by design); also controller-limited
- R2/42 walk/brisk_mu0.54: body-limited: slip; also controller-limited
- R2/43 walk/brisk_mu0.54: controller-limited (decision 20; untestable candidates: cop_at_pad_edge_with_ankle_saturated)
- R4/42 walk/brisk_mu0.54: body-limited: slip_median, slip_worst; checkpoint 1253376
- R0 walk/fr05_mu0.54: not attributed (falls; attribution is applied to R2, R3 rows) (by design)
- R1 walk/fr05_mu0.54: not attributed (falls; attribution is applied to R2, R3 rows) (by design)
- R2/42 walk/fr05_mu0.54: body-limited: slip
- R2/43 walk/fr05_mu0.54: controller-limited (decision 20; untestable candidates: cop_at_pad_edge_with_ankle_saturated)
- R4/42 walk/fr05_mu0.54: body-limited: slip_median, slip_worst; checkpoint 1253376

#### Turn

| Condition | Gated for | R0 | R1 | R2/42 | R2/43 | R3/42 | R3/43 | R4/42 | R2 verdict (shown) | R3 verdict (not judged: stopped) | R4 verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| walking_0.35_left | all | FAIL 2,2 (0.02,0.02) | FAIL 19,19 (0.78,0.78) | pass 20,19 (0.86,0.78) | pass 20,20 (0.86,0.86) | not run | not run | FAIL 20,0 (0.86,0.00) | shown | not run | body-limited |
| walking_0.35_right | all | FAIL 2,2 (0.02,0.02) | FAIL 18,17 (0.72,0.66) | pass 20,19 (0.86,0.78) | pass 20,20 (0.86,0.86) | not run | not run | FAIL 20,0 (0.86,0.00) | shown | not run | body-limited |
| in_place_0.50_left | all | pass 20,20 (0.86,0.86) | pass 20,20 (0.86,0.86) | pass 20,20 (0.86,0.86) | pass 20,20 (0.86,0.86) | not run | not run | FAIL 20,0 (0.86,0.00) | shown | not run | body-limited |
| in_place_0.50_right | all | pass 20,20 (0.86,0.86) | pass 20,20 (0.86,0.86) | pass 20,20 (0.86,0.86) | pass 20,20 (0.86,0.86) | not run | not run | FAIL 20,0 (0.86,0.00) | shown | not run | body-limited |
| walking_0.50_left | none (reported) | FAIL 1,1 (0.00,0.00) (rep.) | FAIL 19,15 (0.78,0.54) (rep.) | FAIL 20,16 (0.86,0.60) (rep.) | FAIL 20,14 (0.86,0.49) (rep.) | not run | not run | FAIL 20,0 (0.86,0.00) (rep.) | controller-limited (rep.) | not run (rep.) | body-limited (rep.) |
| walking_0.50_right | none (reported) | FAIL 1,1 (0.00,0.00) (rep.) | FAIL 19,12 (0.78,0.39) (rep.) | pass 20,17 (0.86,0.66) (rep.) | pass 20,18 (0.86,0.72) (rep.) | not run | not run | FAIL 20,0 (0.86,0.00) (rep.) | shown (rep.) | not run (rep.) | body-limited (rep.) |

- R0 turn/walking_0.35_left: not attributed (falls; attribution is applied to R2, R3 rows) (by design)
- R1 turn/walking_0.35_left: not attributed (falls; attribution is applied to R2, R3 rows) (by design)
- R4/42 turn/walking_0.35_left: body-limited: slip_median, slip_worst; checkpoint 1253376
- R0 turn/walking_0.35_right: not attributed (falls; attribution is applied to R2, R3 rows) (by design)
- R1 turn/walking_0.35_right: not attributed (falls; attribution is applied to R2, R3 rows) (by design); also controller-limited
- R4/42 turn/walking_0.35_right: body-limited: slip_median, slip_worst; checkpoint 1253376
- R4/42 turn/in_place_0.50_left: body-limited: slip_median, slip_worst; checkpoint 1253376
- R4/42 turn/in_place_0.50_right: body-limited: slip_median, slip_worst; checkpoint 1253376
- R0 turn/walking_0.50_left: not attributed (falls; attribution is applied to R2, R3 rows) (by design)
- R1 turn/walking_0.50_left: body-limited: joint_range; also not attributed (falls; attribution is applied to R2, R3 rows)
- R2/42 turn/walking_0.50_left: controller-limited
- R2/43 turn/walking_0.50_left: body-limited: joint_range
- R4/42 turn/walking_0.50_left: body-limited: slip_median, slip_worst; checkpoint 1253376
- R0 turn/walking_0.50_right: not attributed (falls; attribution is applied to R2, R3 rows) (by design)
- R1 turn/walking_0.50_right: body-limited: joint_range; also not attributed (falls; attribution is applied to R2, R3 rows)
- R4/42 turn/walking_0.50_right: body-limited: slip_median, slip_worst; checkpoint 1253376

#### Start and stop

| Condition | Gated for | R0 | R1 | R2/42 | R2/43 | R3/42 | R3/43 | R4/42 | R2 verdict (shown) | R3 verdict (not judged: stopped) | R4 verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| twice | all | FAIL 4,3 (0.07,0.04) | pass 20,20 (0.86,0.86) | pass 20,20 (0.86,0.86) | pass 20,20 (0.86,0.86) | not run | not run | FAIL 20,0 (0.86,0.00) | shown | not run | controller-limited |

- R0 start_stop/twice: not attributed (falls; attribution is applied to R2, R3 rows) (by design); also controller-limited
- R4/42 start_stop/twice: controller-limited; checkpoint 1253376

#### Push

| Condition | Gated for | R0 | R1 | R2/42 | R2/43 | R3/42 | R3/43 | R4/42 | R2 verdict (shown) | R3 verdict (not judged: stopped) | R4 verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| standing_forward | all | FAIL 6,0 (0.14,0.00) | pass 20,20 (0.86,0.86) | pass 20,20 (0.86,0.86) | pass 20,20 (0.86,0.86) | not run | not run | FAIL 0,0 (0.00,0.00) | shown | not run | fails by falls (R4: not attributed) |
| standing_backward | all | FAIL 20,0 (0.86,0.00) | pass 20,20 (0.86,0.86) | pass 20,20 (0.86,0.86) | pass 20,20 (0.86,0.86) | not run | not run | FAIL 20,1 (0.86,0.00) | shown | not run | controller-limited |
| standing_left | all | pass 20,20 (0.86,0.86) | pass 20,20 (0.86,0.86) | pass 20,20 (0.86,0.86) | pass 20,20 (0.86,0.86) | not run | not run | FAIL 20,2 (0.86,0.02) | shown | not run | controller-limited |
| standing_right | all | pass 20,20 (0.86,0.86) | pass 20,20 (0.86,0.86) | pass 20,20 (0.86,0.86) | pass 20,20 (0.86,0.86) | not run | not run | FAIL 20,2 (0.86,0.02) | shown | not run | controller-limited |
| walking_forward | all | FAIL 0,0 (0.00,0.00) | FAIL 2,1 (0.02,0.00) | FAIL 1,1 (0.00,0.00) | FAIL 1,1 (0.00,0.00) | not run | not run | FAIL 0,0 (0.00,0.00) | controller-limited | not run | fails by falls (R4: not attributed) |
| walking_backward | all | FAIL 0,0 (0.00,0.00) | FAIL 0,0 (0.00,0.00) | FAIL 0,0 (0.00,0.00) | FAIL 0,0 (0.00,0.00) | not run | not run | FAIL 19,0 (0.78,0.00) | controller-limited | not run | controller-limited |
| walking_left | all | FAIL 0,0 (0.00,0.00) | FAIL 1,1 (0.00,0.00) | FAIL 2,0 (0.02,0.00) | FAIL 0,0 (0.00,0.00) | not run | not run | FAIL 20,0 (0.86,0.00) | controller-limited | not run | controller-limited |
| walking_right | all | FAIL 0,0 (0.00,0.00) | FAIL 1,0 (0.00,0.00) | FAIL 2,0 (0.02,0.00) | FAIL 0,0 (0.00,0.00) | not run | not run | FAIL 20,0 (0.86,0.00) | controller-limited | not run | controller-limited |

- R0 push/standing_forward: not attributed (falls; attribution is applied to R2, R3 rows) (by design); also controller-limited
- R4/42 push/standing_forward: not attributed (falls; attribution is applied to R2, R3 rows) (by design); checkpoint 1253376
- R0 push/standing_backward: controller-limited
- R4/42 push/standing_backward: controller-limited; checkpoint 1253376
- R4/42 push/standing_left: controller-limited; checkpoint 1253376
- R4/42 push/standing_right: controller-limited; checkpoint 1253376
- R0 push/walking_forward: not attributed (falls; attribution is applied to R2, R3 rows) (by design)
- R1 push/walking_forward: not attributed (falls; attribution is applied to R2, R3 rows) (by design); also controller-limited
- R2/42 push/walking_forward: controller-limited (decision 20; untestable candidates: cop_at_pad_edge_with_ankle_saturated)
- R2/43 push/walking_forward: controller-limited (decision 20; untestable candidates: cop_at_pad_edge_with_ankle_saturated)
- R4/42 push/walking_forward: not attributed (falls; attribution is applied to R2, R3 rows) (by design); checkpoint 1253376
- R0 push/walking_backward: not attributed (falls; attribution is applied to R2, R3 rows) (by design)
- R1 push/walking_backward: not attributed (falls; attribution is applied to R2, R3 rows) (by design)
- R2/42 push/walking_backward: controller-limited (decision 20; untestable candidates: cop_at_pad_edge_with_ankle_saturated)
- R2/43 push/walking_backward: controller-limited (decision 20; untestable candidates: cop_at_pad_edge_with_ankle_saturated)
- R4/42 push/walking_backward: controller-limited; also not attributed (falls; attribution is applied to R2, R3 rows); checkpoint 1253376
- R0 push/walking_left: not attributed (falls; attribution is applied to R2, R3 rows) (by design)
- R1 push/walking_left: not attributed (falls; attribution is applied to R2, R3 rows) (by design)
- R2/42 push/walking_left: controller-limited (decision 20; untestable candidates: cop_at_pad_edge_with_ankle_saturated); also controller-limited
- R2/43 push/walking_left: controller-limited (decision 20; untestable candidates: cop_at_pad_edge_with_ankle_saturated)
- R4/42 push/walking_left: controller-limited; checkpoint 1253376
- R0 push/walking_right: not attributed (falls; attribution is applied to R2, R3 rows) (by design)
- R1 push/walking_right: not attributed (falls; attribution is applied to R2, R3 rows) (by design); also controller-limited
- R2/42 push/walking_right: body-limited: slip; also controller-limited
- R2/43 push/walking_right: controller-limited (decision 20; untestable candidates: cop_at_pad_edge_with_ankle_saturated)
- R4/42 push/walking_right: controller-limited; checkpoint 1253376

#### Carpet

| Condition | Gated for | R0 | R1 | R2/42 | R2/43 | R3/42 | R3/43 | R4/42 | R2 verdict (shown) | R3 verdict (not judged: stopped) | R4 verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| robot_floor_carpet | R3/42, R3/43, R4/42 | n/a | n/a | FAIL 0,0 (0.00,0.00) (rep.) | FAIL 0,0 (0.00,0.00) (rep.) | not run | not run | FAIL 20,0 (0.86,0.00) | controller-limited (rep.) | not run | body-limited |
| low | R3/42, R3/43, R4/42 | n/a | n/a | FAIL 0,0 (0.00,0.00) (rep.) | FAIL 0,0 (0.00,0.00) (rep.) | not run | not run | FAIL 20,0 (0.86,0.00) | controller-limited (rep.) | not run | body-limited |
| medium | none (reported) | n/a | n/a | no stance | no stance | not run | not run | no stance | not run (rep.) | not run (rep.) | not run (rep.) |
| high | none (reported) | n/a | n/a | no stance | no stance | not run | not run | no stance | not run (rep.) | not run (rep.) | not run (rep.) |

The tracker (R0-R2) was never fitted on carpet (decision 23: the fit and G5 ran on hard floor only, and R2 fell in every start tried on robot_floor_carpet and low pile): the carpet walk is reported for R0-R2, with its fall attribution, and gated for R3 and R4.

- R2/42 carpet/robot_floor_carpet: controller-limited (decision 20; untestable candidates: cop_at_pad_edge_with_ankle_saturated)
- R2/43 carpet/robot_floor_carpet: controller-limited (decision 20; untestable candidates: cop_at_pad_edge_with_ankle_saturated)
- R4/42 carpet/robot_floor_carpet: body-limited: slip_median, slip_worst; checkpoint 1253376
- R2/42 carpet/low: controller-limited (decision 20; untestable candidates: cop_at_pad_edge_with_ankle_saturated)
- R2/43 carpet/low: controller-limited (decision 20; untestable candidates: cop_at_pad_edge_with_ankle_saturated)
- R4/42 carpet/low: body-limited: slip_median, slip_worst; checkpoint 1253376

Provenance of each row (PROVENANCE_RULE):

- R0 = R0: feedforward + PD on the commanded clock (ROWS_RULE)
- R1 = R1: the tracker on the true state, the body-level G5 selection (seed42_gen20)
- R2/42 = R2: the tracker on the estimate, CMA-ES seed 42's G5 selection seed42_gen20
- R2/43 = R2: the tracker on the estimate, CMA-ES seed 43's G5 selection seed43_gen40
- R4/42 = R4: PPO seed 42's policy-only policy at 1,253,376 steps, the joint targets held by the servos alone (no tracker); checkpoint 1253376 steps (final)

### Looks (sway, head, tail)

Reported, not gated (LOOKS_RULE): sway against 0.12 H, head and tail amplitude against BEHAVIOR_SPEC's bands (head 0.05-0.15 rad, tail 0.1-0.2 rad).

| Condition | Row | Sway mean / max (H) | Head amplitude mean [range] (rad), in band | Tail amplitude mean [range] (rad), in band |
|---|---|---|---|---|
| walk/nominal | R0 | 0.183 / 0.200 | 0.051 [0.051-0.051], 3/3 | 0.120 [0.120-0.120], 3/3 |
| walk/nominal | R1 | 0.174 / 0.184 | 0.051 [0.051-0.051], 18/18 | 0.120 [0.120-0.120], 18/18 |
| walk/nominal | R2/42 | 0.176 / 0.187 | 0.051 [0.051-0.051], 20/20 | 0.120 [0.120-0.120], 20/20 |
| walk/nominal | R2/43 | 0.178 / 0.190 | 0.051 [0.051-0.052], 20/20 | 0.120 [0.120-0.121], 20/20 |
| walk/nominal | R4/42 | 0.008 / 0.012 | 0.001 [0.001-0.001], 0/20 | 0.001 [0.001-0.001], 0/20 |
| walk/brisk | R0 | - | - | - |
| walk/brisk | R1 | 0.129 / 0.158 | 0.052 [0.051-0.052], 15/15 | 0.121 [0.120-0.121], 15/15 |
| walk/brisk | R2/42 | 0.132 / 0.151 | 0.051 [0.051-0.052], 17/17 | 0.121 [0.121-0.121], 17/17 |
| walk/brisk | R2/43 | - | - | - |
| walk/brisk | R4/42 | 0.005 / 0.007 | 0.002 [0.002-0.002], 0/20 | 0.001 [0.001-0.001], 0/20 |
| walk/fr05 | R0 | - | - | - |
| walk/fr05 | R1 | 0.107 / 0.145 | 0.052 [0.052-0.052], 1/1 | 0.122 [0.122-0.122], 1/1 |
| walk/fr05 | R2/42 | 0.127 / 0.141 | 0.052 [0.052-0.052], 1/1 | 0.122 [0.122-0.122], 1/1 |
| walk/fr05 | R2/43 | - | - | - |
| walk/fr05 | R4/42 | 0.006 / 0.010 | 0.002 [0.002-0.002], 0/20 | 0.001 [0.001-0.001], 0/20 |
| walk/nominal_mu0.54 | R0 | 0.183 / 0.197 | 0.051 [0.051-0.051], 5/5 | 0.120 [0.120-0.121], 5/5 |
| walk/nominal_mu0.54 | R1 | 0.175 / 0.186 | 0.051 [0.051-0.051], 14/14 | 0.120 [0.120-0.120], 14/14 |
| walk/nominal_mu0.54 | R2/42 | 0.176 / 0.188 | 0.051 [0.051-0.051], 20/20 | 0.120 [0.120-0.121], 20/20 |
| walk/nominal_mu0.54 | R2/43 | 0.180 / 0.193 | 0.051 [0.051-0.051], 20/20 | 0.120 [0.120-0.121], 20/20 |
| walk/nominal_mu0.54 | R4/42 | 0.008 / 0.011 | 0.001 [0.001-0.001], 0/20 | 0.001 [0.001-0.001], 0/20 |
| walk/brisk_mu0.54 | R0 | - | - | - |
| walk/brisk_mu0.54 | R1 | 0.132 / 0.163 | 0.052 [0.051-0.053], 14/14 | 0.121 [0.121-0.122], 14/14 |
| walk/brisk_mu0.54 | R2/42 | 0.138 / 0.187 | 0.052 [0.051-0.052], 15/15 | 0.122 [0.121-0.123], 15/15 |
| walk/brisk_mu0.54 | R2/43 | - | - | - |
| walk/brisk_mu0.54 | R4/42 | 0.005 / 0.007 | 0.002 [0.002-0.002], 0/20 | 0.001 [0.001-0.001], 0/20 |
| walk/fr05_mu0.54 | R0 | - | - | - |
| walk/fr05_mu0.54 | R1 | - | - | - |
| walk/fr05_mu0.54 | R2/42 | - | - | - |
| walk/fr05_mu0.54 | R2/43 | - | - | - |
| walk/fr05_mu0.54 | R4/42 | 0.007 / 0.010 | 0.002 [0.002-0.002], 0/20 | 0.001 [0.001-0.001], 0/20 |
| carpet/robot_floor_carpet | R2/42 | - | - | - |
| carpet/robot_floor_carpet | R2/43 | - | - | - |
| carpet/robot_floor_carpet | R4/42 | 0.008 / 0.013 | 0.001 [0.001-0.001], 0/20 | 0.000 [0.000-0.001], 0/20 |
| carpet/low | R2/42 | - | - | - |
| carpet/low | R2/43 | - | - | - |
| carpet/low | R4/42 | 0.007 / 0.010 | 0.001 [0.001-0.002], 0/20 | 0.001 [0.000-0.001], 0/20 |

### Margin ledger

Cells: the threshold (the last value at which all 3 seeds survive and none fails a term it passed at nominal, or survive) and the nearest value that fails with the label there (falls labelled by lifting each candidate limit on the 3 seeds, ledger.LIFT_RULE); '*' the search is not finished or the record is flagged (listed below).

| Margin | Search | R2/42 works | R3/42 works | R2/42 survives | R3/42 survives | Stage 0 |
|---|---|---|---|---|---|---|
| Grip | walk_nominal | 0.7062 (fails 0.7; controller-limited) | not run | works to 0.1 | not run | 0.48 |
| Grip | turn_left | 0.3187 (fails 0.3125; body-limited: slip (lifted, all 3 seeds survive; untestable: cop_at_pad_edge_with_ankle_saturated)) | not run | 0.125 (fails 0.1187; body-limited: slip (lifted, all 3 seeds survive; untestable: cop_at_pad_edge_with_ankle_saturated)) | not run | 0.54 |
| Grip | turn_right | 0.4188 (fails 0.4125; controller-limited) | not run | 0.3063 (fails 0.3; body-limited: slip (lifted, all 3 seeds survive; untestable: cop_at_pad_edge_with_ankle_saturated)) | not run | 0.54 |
| Grip | walk_brisk | fails at nominal (body-limited: drive_torque, slip (lifted, all 3 seeds survive; untestable: cop_at_pad_edge_with_ankle_saturated)) [panels: FAIL 17,17; controller-limited (decision 20; 3 falls: the sign test needs 5 for p < 0.05, so no limit was lifted; untestable candidates: cop_at_pad_edge_with_ankle_saturated)] | not run | fails at nominal (body-limited: drive_torque, slip (lifted, all 3 seeds survive; untestable: cop_at_pad_edge_with_ankle_saturated)) [panels: FAIL 17,17; controller-limited (decision 20; 3 falls: the sign test needs 5 for p < 0.05, so no limit was lifted; untestable candidates: cop_at_pad_edge_with_ankle_saturated)] | not run | 0.75 |
| Grip | turn (worse of turn_left, turn_right) | 0.4188 (fails 0.4125; controller-limited) [turn_right] | not run | 0.3063 (fails 0.3; body-limited: slip (lifted, all 3 seeds survive; untestable: cop_at_pad_edge_with_ankle_saturated)) [turn_right] | not run | 0.54 |
| Push | standing_forward | 0.2587 (fails 0.2663; controller-limited (falls with no relaxable hardware limit active before the onset: nothing to lift, decision 20; untestable: cop_at_pad_edge_with_ankle_saturated)) | not run | 0.2587 (fails 0.2663; controller-limited (falls with no relaxable hardware limit active before the onset: nothing to lift, decision 20; untestable: cop_at_pad_edge_with_ankle_saturated)) | not run | 0.487 |
| Push | standing_backward | 0.2663 (fails 0.2739; controller-limited (falls with no relaxable hardware limit active before the onset: nothing to lift, decision 20; untestable: cop_at_pad_edge_with_ankle_saturated)) | not run | 0.2663 (fails 0.2739; controller-limited (falls with no relaxable hardware limit active before the onset: nothing to lift, decision 20; untestable: cop_at_pad_edge_with_ankle_saturated)) | not run | (forward 0.487) |
| Push | standing_left | 0.5479 (fails 0.5555; controller-limited (lifting slip saves none of the falls; untestable: cop_at_pad_edge_with_ankle_saturated)) | not run | 0.5479 (fails 0.5555; controller-limited (lifting slip saves none of the falls; untestable: cop_at_pad_edge_with_ankle_saturated)) | not run | (forward 0.487) |
| Push | standing_right | 0.5555 (fails 0.5631; controller-limited (lifting slip saves none of the falls; untestable: cop_at_pad_edge_with_ankle_saturated)) | not run | 0.5555 (fails 0.5631; controller-limited (lifting slip saves none of the falls; untestable: cop_at_pad_edge_with_ankle_saturated)) | not run | (forward 0.487) |
| Push | walking_forward | 0.1217 (fails 0.1294; body-limited: slip (lifted, all 3 seeds survive; untestable: cop_at_pad_edge_with_ankle_saturated)) | not run | 0.1217 (fails 0.1294; body-limited: slip (lifted, all 3 seeds survive; untestable: cop_at_pad_edge_with_ankle_saturated)) | not run | 0.487 |
| Push | walking_backward | 0.1522 (fails 0.1598; controller-limited (lifting drive_torque, slip saves none of the falls; untestable: cop_at_pad_edge_with_ankle_saturated)) | not run | 0.1522 (fails 0.1598; controller-limited (lifting drive_torque, slip saves none of the falls; untestable: cop_at_pad_edge_with_ankle_saturated)) | not run | (forward 0.487) |
| Push | walking_left | 0.06848 (fails 0.07609; body-limited: slip (lifted, all 3 seeds survive; untestable: cop_at_pad_edge_with_ankle_saturated)) | not run | 0.06848 (fails 0.07609; body-limited: slip (lifted, all 3 seeds survive; untestable: cop_at_pad_edge_with_ankle_saturated)) | not run | (forward 0.487) |
| Push | walking_right | 0.0837 (fails 0.09131; controller-limited (lifting slip saves none of the falls; untestable: cop_at_pad_edge_with_ankle_saturated)) | not run | 0.0837 (fails 0.09131; controller-limited (lifting slip saves none of the falls; untestable: cop_at_pad_edge_with_ankle_saturated)) | not run | (forward 0.487) |
| Stop | - | not run (plan deviation, decision 33) | not run (plan deviation, decision 33) | not run (plan deviation, decision 33) | not run (plan deviation, decision 33) | 1.25 |
| Drive speed | walk_fr05 | fails at nominal (controller-limited (lifting drive_torque, slip saves none of the falls; untestable: cop_at_pad_edge_with_ankle_saturated)) [panels: FAIL 1,0; controller-limited (decision 20; untestable candidates: cop_at_pad_edge_with_ankle_saturated)] | not run | fails at nominal (controller-limited (lifting drive_torque, slip saves none of the falls; untestable: cop_at_pad_edge_with_ankle_saturated)) [panels: FAIL 1,0; controller-limited (decision 20; untestable candidates: cop_at_pad_edge_with_ankle_saturated)] | not run | 0.93 (0.81 without the 15 % margin) |
| Drive torque | walk_nominal | 0.9238 (fails 0.918; controller-limited) | not run | 0.5312 (fails 0.5254; body-limited: slip (lifted, all 3 seeds survive; untestable: cop_at_pad_edge_with_ankle_saturated)) | not run | 0.76 (walk RMS 0.76) |
| Model doubt: servo kp | walk_nominal | 0.6777 (fails 0.6719; controller-limited) | not run | 0.5781 (fails 0.5723; controller-limited (falls with no relaxable hardware limit active before the onset: nothing to lift, decision 20; untestable: cop_at_pad_edge_with_ankle_saturated)) | not run | - |
| Model doubt: latency | walk_nominal | 2 steps (4 ms) (fails 3 steps (6 ms); controller-limited) | not run | 3 steps (6 ms) (fails 4 steps (8 ms); controller-limited (lifting drive_torque, slip saves none of the falls; untestable: cop_at_pad_edge_with_ankle_saturated)) | not run | - |
| Model doubt: trunk mass | walk_nominal | works to 1.3 (robot +13.0 %) | not run | works to 1.3 (robot +13.0 %) | not run | - |
| Model doubt: trunk mass | walk_nominal_trunk_x1.1 (trunk x1.1: robot +4.3 %) | works 3/3 | not run | survives 3/3 | not run | - |
| Model doubt: trunk mass | walk_nominal_robot_x1.1 (trunk x1.23: robot +10.0 %) | works 3/3 | not run | survives 3/3 | not run | - |
| Model doubt: pile damping | stand_low_down | works to 0.5 | not run | works to 0.5 | not run | - |
| Model doubt: pile damping | stand_low_up | works to 2 | not run | works to 2 | not run | - |
| Model doubt: pile damping | carpet_low_down | fails at nominal (controller-limited (lifting slip saves none of the falls; untestable: cop_at_pad_edge_with_ankle_saturated)) [panels: FAIL 0,0; controller-limited (decision 20; untestable candidates: cop_at_pad_edge_with_ankle_saturated)] | not run | fails at nominal (controller-limited (lifting slip saves none of the falls; untestable: cop_at_pad_edge_with_ankle_saturated)) [panels: FAIL 0,0; controller-limited (decision 20; untestable candidates: cop_at_pad_edge_with_ankle_saturated)] | not run | - |
| Model doubt: pile damping | carpet_low_up | fails at nominal (controller-limited (lifting slip saves none of the falls; untestable: cop_at_pad_edge_with_ankle_saturated)) [panels: FAIL 0,0; controller-limited (decision 20; untestable candidates: cop_at_pad_edge_with_ankle_saturated)] | not run | fails at nominal (controller-limited (lifting slip saves none of the falls; untestable: cop_at_pad_edge_with_ankle_saturated)) [panels: FAIL 0,0; controller-limited (decision 20; untestable candidates: cop_at_pad_edge_with_ankle_saturated)] | not run | - |
| Pack voltage | walk_nominal_25.2V | works 2/3 = nominal plant; outright fails | not run | survives 3/3 | not run | - |
| Pack voltage | walk_nominal_22.2V | works 2/3; outright fails | not run | survives 3/3 | not run | - |
| Pack voltage | walk_nominal_21V | works 2/3; outright fails | not run | survives 3/3 | not run | - |
| Pack voltage | walk_fr05_25.2V | fails 0/3 = nominal plant (controller-limited (lifting drive_torque, slip saves none of the falls; untestable: cop_at_pad_edge_with_ankle_saturated)) | not run | falls 0/3 | not run | - |
| Pack voltage | walk_fr05_22.2V | fails 0/3 (controller-limited (lifting drive_torque, slip saves none of the falls; untestable: cop_at_pad_edge_with_ankle_saturated)) | not run | falls 0/3 | not run | - |
| Pack voltage | walk_fr05_21V | fails 0/3 (controller-limited (lifting drive_torque, slip saves none of the falls; untestable: cop_at_pad_edge_with_ankle_saturated)) | not run | falls 0/3 | not run | - |

- R3 is judged with R2 for this body (R3/42 stopped at 2015232 steps: G6b failed again after the fix (FIX_RULE) (decision 32); R3/43 stopped at 2015232 steps: the body's R3 is judged with R2 (train.R2_JUDGED_RULE, decision 38): its seed 42 run was stopped by FIX_RULE at 2015232 steps (G6b failed again after the fix), so this run cannot change a verdict; stopped to give the 75 cm runs the cores (decision 32)): its ledger row is not run (ledger.ROWS_RULE)
- Grip: Stage 0: the lowest friction the round-two turning screen requires in single support, per body: nominal walk / 0.35 rad/s walking turn (the worse direction) / brisk walk (STAGE0_RULE)
- Stop: not run (plan deviation, decision 33): the deceleration a stop holds needs stop references that decelerate at other rates, and the run holds one stop transition (STOP_RULE)
- Drive speed: Stage 0: the knee's speed need at Fr 0.05 over its rated speed, including the screens' 15 % speed margin; without it 0.81 (0.81 / 0.83 / 0.81), which is what the ledger's limit compares (STAGE0_RULE)
- Drive torque: Stage 0: the worst drive's sustained-torque ratio, the larger of the nominal walk's RMS and the one-foot hold (ankle_a's one-foot hold); the nominal walk's RMS alone 0.76 (neck), which is what the ledger's limit compares (STAGE0_RULE)
- Pack voltage: Stage 0: Stage 0 used 24 V; at 22.2 V the knee's rated speed (8.04 rad/s) equals the Fr 0.05 requirement including the screens' 15 % speed margin (8.08-8.12 rad/s; the peak swing speed itself is about 7.0 rad/s)

### PPO runs

| Run | Status | Steps / budget | Last checkpoint | Judged checkpoint | G6a zero residual | G6a env steps/s | G6b | Fix; G6b after it | G6c | Job h | CPU h | run.json written |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R3/42 | stopped_g6b | 2015232 / 4005888 | 2015232 | 2506752 (final) | yes | 83 (FAIL: cut the steps equally for every body, decision 26) | FAIL at 1007616 (survival policy 14.14 s, zero 15.98 s): make one fix, then stop | from 1007616 (task_weights 1,1,1,1,1,2,2,1,1); FAIL at 2015232 (survival policy 15.16 s, zero 15.98 s): stop the run (FIX_RULE) | - | 17.72 (running) | running | 2026-11-10T00:12:29+00:00 |
| R3/43 | stopped_r2_judged | 2015232 / 4005888 | 2015232 | 2506752 (final) | yes | 80 (FAIL: cut the steps equally for every body, decision 26) | FAIL at 1007616 (survival policy 15.41 s, zero 15.98 s): make one fix, then stop | from 1007616 (task_weights 1,1,1,1,1,2,2,1,1); pass at 2015232 (survival policy 16.00 s, zero 15.98 s) | - | 19.54 (running) | running | 2026-11-10T00:12:29+00:00 |
| R4/42 | running | 1507328 / 4005888 | 1507328 | 1253376 (final) | - | 120 (FAIL: cut the steps equally for every body, decision 26) | - | - | - | 10.33 (running) | running | 2026-11-10T00:12:30+00:00 |

Elapsed PPO wall clock (the union of every run's sessions): 10.33 h; job-hours (their sum: the runs share the cores) 47.59 h.

### Fit and G5

G5 pass (seeds passing: [42, 43]); body-level selection (env.GAINS_RULE): CMA-ES seed 42, seed42_gen20 (passed).

| CMA-ES seed | Fit | Evaluations | Sigma | Selected | Walk | Walking turn | In-place turn | G5 |
|---|---|---|---|---|---|---|---|---|
| 42 | done | 720 | 0.0614 | seed42_gen20 | 18/20 (surv 20) | 17/20 (surv 20) | 20/20 (surv 20) | passes G5 |
| 43 | done | 720 | 0.0300 | seed43_gen40 | 20/20 (surv 20) | 19/20 (surv 20) | 20/20 (surv 20) | passes G5 |

### Core-hours per step

| Step | Wall h | Core-h | Basis |
|---|---|---|---|
| prepare | 0.14 | 0.14 | wall, one process |
| gates | 0.07 | 0.07 | wall, one process |
| fit | 5.22 | 5.22 | sessions' end - start x workers |
| g5 | 0.38 | 0.38 | tune.json seconds, one process assumed (workers not recorded): a lower bound |
| ppo | 10.33 (job-hours 47.59) | 19.10 | finished sessions' CPU, and for a running session its wall to the record's last write x the CPU per wall second of its first rollouts (an estimate); wall: the elapsed union of the sessions, job-hours beside |
| panels | 3.95 | 4.07 | episodes' and attributions' wall seconds (one process each): an upper bound on CPU |
| ledger | 0.27 | 0.31 | episodes' CPU seconds, each evaluation once (by its key) however many margins use it |
| total | | 29.28 | core-hours recorded so far |

## 60-N-H-10

### Verdicts by controller

| Controller | shown | provisional | body-limited | controller-limited | not attributed | fails by falls (R4: not attributed) | incomplete | not run |
|---|---|---|---|---|---|---|---|---|
| R2 (shown) | 14 | 1 | 0 | 5 | 0 | 0 | 0 | 0 |
| R3 (not judged: judged with R2) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| R4 | 0 | 6 | 10 | 4 | 0 | 2 | 0 | 0 |

Counts over the conditions gated for each controller's rows.
R4 has one seed (decisions 3 and 28): its verdicts are at most provisional (policy_rows.R4_VERDICT_RULE); its record on each gated condition: FAIL 16, pass 6.
60-N-H-10's R3 is judged with R2 (R3/42 stopped at 2015232 steps: G6b failed again after the fix (FIX_RULE) (decision 32); R3/43 stopped at 1753088 steps: the body's R3 is judged with R2 (train.R2_JUDGED_RULE, decision 38): its seed 42 run was stopped by FIX_RULE at 2015232 steps (G6b failed again after the fix), so this run cannot change a verdict; stopped to give the 75 cm runs the cores (decision 32)): the grid shows R2's verdicts (train.R2_JUDGED_RULE).

G7 pass: sway 0.1693 H (Stage 0 0.1693, plan 0.169), cadence 1.9692 steps/s (Stage 0 1.9691, plan 1.97).

### Verdict grid

Verdicts shown: R2: 60-N-H-10's R3 is judged with R2 (R3/42 stopped at 2015232 steps: G6b failed again after the fix (FIX_RULE) (decision 32); R3/43 stopped at 1753088 steps: the body's R3 is judged with R2 (train.R2_JUDGED_RULE, decision 38): its seed 42 run was stopped by FIX_RULE at 2015232 steps (G6b failed again after the fix), so this run cannot change a verdict; stopped to give the 75 cm runs the cores (decision 32); train.R2_JUDGED_RULE).  Cells: result, survived,succeeded of 20 (their one-sided 95 % Clopper-Pearson lower bounds); '(rep.)' reported, not gated, for that row; '*' interim (listed below each table); 'not run' no record; 'no stance' the floor has no settled stance for the body (panels.NOT_RUN_RULE); 'n/a' the row does not run there; R4's verdict is at most provisional (one seed), and its falls are never attributed (by design: decision 20's attribution is for R2 and R3).

#### Stand

| Condition | Gated for | R0 | R1 | R2/42 | R2/43 | R3/42 | R3/43 | R4/42 | R2 verdict (shown) | R3 verdict (not judged: stopped) | R4 verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| hard | all | pass 20,20 (0.86,0.86) | pass 20,20 (0.86,0.86) | pass 20,20 (0.86,0.86) | pass 20,20 (0.86,0.86) | not run | not run | pass 20,20 (0.86,0.86) | shown | not run | provisional |
| robot_floor_carpet | all | n/a | n/a | pass 20,20 (0.86,0.86) | pass 20,20 (0.86,0.86) | not run | not run | pass 20,20 (0.86,0.86) | shown | not run | provisional |
| low | all | n/a | n/a | pass 20,20 (0.86,0.86) | pass 20,20 (0.86,0.86) | not run | not run | pass 20,20 (0.86,0.86) | shown | not run | provisional |

#### Walk

| Condition | Gated for | R0 | R1 | R2/42 | R2/43 | R3/42 | R3/43 | R4/42 | R2 verdict (shown) | R3 verdict (not judged: stopped) | R4 verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| nominal | all | FAIL 1,1 (0.00,0.00) | FAIL 18,18 (0.72,0.72) | pass 20,18 (0.86,0.72) | pass 20,18 (0.86,0.72) | not run | not run | FAIL 20,0 (0.86,0.00) | shown | not run | body-limited |
| brisk | all | FAIL 0,0 (0.00,0.00) | FAIL 15,14 (0.54,0.49) | FAIL 5,5 (0.10,0.10) | pass 20,19 (0.86,0.78) | not run | not run | FAIL 20,0 (0.86,0.00) | provisional | not run | body-limited |
| fr05 | all | FAIL 0,0 (0.00,0.00) | FAIL 0,0 (0.00,0.00) | FAIL 0,0 (0.00,0.00) | FAIL 0,0 (0.00,0.00) | not run | not run | FAIL 20,0 (0.86,0.00) | controller-limited | not run | body-limited |
| nominal_mu0.54 | all | FAIL 7,7 (0.18,0.18) | FAIL 12,11 (0.39,0.35) | pass 20,18 (0.86,0.72) | pass 20,19 (0.86,0.78) | not run | not run | FAIL 20,0 (0.86,0.00) | shown | not run | body-limited |
| brisk_mu0.54 | none (reported) | FAIL 0,0 (0.00,0.00) (rep.) | FAIL 13,11 (0.44,0.35) (rep.) | FAIL 6,3 (0.14,0.04) (rep.) | FAIL 17,14 (0.66,0.49) (rep.) | not run | not run | FAIL 20,0 (0.86,0.00) (rep.) | controller-limited (rep.) | not run (rep.) | body-limited (rep.) |
| fr05_mu0.54 | none (reported) | FAIL 0,0 (0.00,0.00) (rep.) | FAIL 0,0 (0.00,0.00) (rep.) | FAIL 0,0 (0.00,0.00) (rep.) | FAIL 2,0 (0.02,0.00) (rep.) | not run | not run | FAIL 20,0 (0.86,0.00) (rep.) | controller-limited (rep.) | not run (rep.) | body-limited (rep.) |

- R0 walk/nominal: not attributed (falls; attribution is applied to R2, R3 rows) (by design)
- R1 walk/nominal: not attributed (falls; attribution is applied to R2, R3 rows) (by design)
- R4/42 walk/nominal: body-limited: slip_median, slip_worst; checkpoint 1253376
- R0 walk/brisk: not attributed (falls; attribution is applied to R2, R3 rows) (by design)
- R1 walk/brisk: not attributed (falls; attribution is applied to R2, R3 rows) (by design); also controller-limited
- R2/42 walk/brisk: body-limited: slip
- R4/42 walk/brisk: body-limited: slip_median, slip_worst; checkpoint 1253376
- R0 walk/fr05: not attributed (falls; attribution is applied to R2, R3 rows) (by design)
- R1 walk/fr05: not attributed (falls; attribution is applied to R2, R3 rows) (by design)
- R2/42 walk/fr05: controller-limited (decision 20; untestable candidates: cop_at_pad_edge_with_ankle_saturated)
- R2/43 walk/fr05: body-limited: slip
- R4/42 walk/fr05: body-limited: slip_median, slip_worst; checkpoint 1253376
- R0 walk/nominal_mu0.54: not attributed (falls; attribution is applied to R2, R3 rows) (by design)
- R1 walk/nominal_mu0.54: not attributed (falls; attribution is applied to R2, R3 rows) (by design); also controller-limited
- R4/42 walk/nominal_mu0.54: body-limited: slip_median, slip_worst; checkpoint 1253376
- R0 walk/brisk_mu0.54: not attributed (falls; attribution is applied to R2, R3 rows) (by design)
- R1 walk/brisk_mu0.54: not attributed (falls; attribution is applied to R2, R3 rows) (by design); also controller-limited
- R2/42 walk/brisk_mu0.54: body-limited: slip; also controller-limited
- R2/43 walk/brisk_mu0.54: controller-limited (decision 20; 3 falls: the sign test needs 5 for p < 0.05, so no limit was lifted; untestable candidates: cop_at_pad_edge_with_ankle_saturated); also controller-limited
- R4/42 walk/brisk_mu0.54: body-limited: slip_median, slip_worst; checkpoint 1253376
- R0 walk/fr05_mu0.54: not attributed (falls; attribution is applied to R2, R3 rows) (by design)
- R1 walk/fr05_mu0.54: not attributed (falls; attribution is applied to R2, R3 rows) (by design)
- R2/42 walk/fr05_mu0.54: controller-limited (decision 20; untestable candidates: cop_at_pad_edge_with_ankle_saturated)
- R2/43 walk/fr05_mu0.54: controller-limited (decision 20; untestable candidates: cop_at_pad_edge_with_ankle_saturated); also controller-limited
- R4/42 walk/fr05_mu0.54: body-limited: slip_median, slip_worst; checkpoint 1253376

#### Turn

| Condition | Gated for | R0 | R1 | R2/42 | R2/43 | R3/42 | R3/43 | R4/42 | R2 verdict (shown) | R3 verdict (not judged: stopped) | R4 verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| walking_0.35_left | all | FAIL 0,0 (0.00,0.00) | FAIL 12,11 (0.39,0.35) | pass 20,19 (0.86,0.78) | pass 20,19 (0.86,0.78) | not run | not run | FAIL 20,0 (0.86,0.00) | shown | not run | body-limited |
| walking_0.35_right | all | FAIL 2,2 (0.02,0.02) | FAIL 17,16 (0.66,0.60) | pass 20,19 (0.86,0.78) | pass 20,20 (0.86,0.86) | not run | not run | FAIL 20,0 (0.86,0.00) | shown | not run | body-limited |
| in_place_0.50_left | all | FAIL 19,19 (0.78,0.78) | pass 20,20 (0.86,0.86) | pass 20,20 (0.86,0.86) | pass 20,20 (0.86,0.86) | not run | not run | FAIL 20,0 (0.86,0.00) | shown | not run | body-limited |
| in_place_0.50_right | all | pass 20,20 (0.86,0.86) | pass 20,20 (0.86,0.86) | pass 20,20 (0.86,0.86) | pass 20,20 (0.86,0.86) | not run | not run | FAIL 20,0 (0.86,0.00) | shown | not run | body-limited |
| walking_0.50_left | none (reported) | FAIL 2,2 (0.02,0.02) (rep.) | FAIL 12,1 (0.39,0.00) (rep.) | FAIL 20,9 (0.86,0.26) (rep.) | FAIL 20,8 (0.86,0.22) (rep.) | not run | not run | FAIL 20,0 (0.86,0.00) (rep.) | body-limited (rep.) | not run (rep.) | body-limited (rep.) |
| walking_0.50_right | none (reported) | FAIL 1,1 (0.00,0.00) (rep.) | FAIL 14,5 (0.49,0.10) (rep.) | FAIL 20,8 (0.86,0.22) (rep.) | FAIL 20,8 (0.86,0.22) (rep.) | not run | not run | FAIL 20,0 (0.86,0.00) (rep.) | body-limited (rep.) | not run (rep.) | body-limited (rep.) |

- R0 turn/walking_0.35_left: not attributed (falls; attribution is applied to R2, R3 rows) (by design)
- R1 turn/walking_0.35_left: not attributed (falls; attribution is applied to R2, R3 rows) (by design); also body-limited: joint_range
- R4/42 turn/walking_0.35_left: body-limited: slip_median, slip_worst; checkpoint 1253376
- R0 turn/walking_0.35_right: not attributed (falls; attribution is applied to R2, R3 rows) (by design)
- R1 turn/walking_0.35_right: not attributed (falls; attribution is applied to R2, R3 rows) (by design); also controller-limited
- R4/42 turn/walking_0.35_right: body-limited: slip_median, slip_worst; checkpoint 1253376
- R0 turn/in_place_0.50_left: not attributed (falls; attribution is applied to R2, R3 rows) (by design)
- R4/42 turn/in_place_0.50_left: body-limited: slip_median, slip_worst; checkpoint 1253376
- R4/42 turn/in_place_0.50_right: body-limited: slip_median, slip_worst; checkpoint 1253376
- R0 turn/walking_0.50_left: not attributed (falls; attribution is applied to R2, R3 rows) (by design)
- R1 turn/walking_0.50_left: body-limited: joint_range; also not attributed (falls; attribution is applied to R2, R3 rows)
- R2/42 turn/walking_0.50_left: body-limited: joint_range
- R2/43 turn/walking_0.50_left: body-limited: joint_range
- R4/42 turn/walking_0.50_left: body-limited: slip_median, slip_worst; checkpoint 1253376
- R0 turn/walking_0.50_right: not attributed (falls; attribution is applied to R2, R3 rows) (by design)
- R1 turn/walking_0.50_right: body-limited: joint_range; also not attributed (falls; attribution is applied to R2, R3 rows)
- R2/42 turn/walking_0.50_right: body-limited: joint_range
- R2/43 turn/walking_0.50_right: body-limited: joint_range
- R4/42 turn/walking_0.50_right: body-limited: slip_median, slip_worst; checkpoint 1253376

#### Start and stop

| Condition | Gated for | R0 | R1 | R2/42 | R2/43 | R3/42 | R3/43 | R4/42 | R2 verdict (shown) | R3 verdict (not judged: stopped) | R4 verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| twice | all | FAIL 6,4 (0.14,0.07) | FAIL 15,15 (0.54,0.54) | pass 20,20 (0.86,0.86) | pass 20,20 (0.86,0.86) | not run | not run | FAIL 20,0 (0.86,0.00) | shown | not run | controller-limited |

- R0 start_stop/twice: not attributed (falls; attribution is applied to R2, R3 rows) (by design); also controller-limited
- R1 start_stop/twice: not attributed (falls; attribution is applied to R2, R3 rows) (by design)
- R4/42 start_stop/twice: controller-limited; checkpoint 1253376

#### Push

| Condition | Gated for | R0 | R1 | R2/42 | R2/43 | R3/42 | R3/43 | R4/42 | R2 verdict (shown) | R3 verdict (not judged: stopped) | R4 verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| standing_forward | all | FAIL 20,0 (0.86,0.00) | pass 20,20 (0.86,0.86) | pass 20,20 (0.86,0.86) | pass 20,20 (0.86,0.86) | not run | not run | FAIL 0,0 (0.00,0.00) | shown | not run | fails by falls (R4: not attributed) |
| standing_backward | all | FAIL 20,0 (0.86,0.00) | pass 20,20 (0.86,0.86) | pass 20,20 (0.86,0.86) | pass 20,20 (0.86,0.86) | not run | not run | pass 20,20 (0.86,0.86) | shown | not run | provisional |
| standing_left | all | pass 20,20 (0.86,0.86) | pass 20,20 (0.86,0.86) | pass 20,20 (0.86,0.86) | pass 20,20 (0.86,0.86) | not run | not run | pass 20,19 (0.86,0.78) | shown | not run | provisional |
| standing_right | all | pass 20,20 (0.86,0.86) | pass 20,20 (0.86,0.86) | pass 20,20 (0.86,0.86) | pass 20,20 (0.86,0.86) | not run | not run | pass 20,19 (0.86,0.78) | shown | not run | provisional |
| walking_forward | all | FAIL 0,0 (0.00,0.00) | FAIL 1,1 (0.00,0.00) | FAIL 2,1 (0.02,0.00) | FAIL 2,1 (0.02,0.00) | not run | not run | FAIL 0,0 (0.00,0.00) | controller-limited | not run | fails by falls (R4: not attributed) |
| walking_backward | all | FAIL 0,0 (0.00,0.00) | FAIL 0,0 (0.00,0.00) | FAIL 0,0 (0.00,0.00) | FAIL 0,0 (0.00,0.00) | not run | not run | FAIL 20,0 (0.86,0.00) | controller-limited | not run | controller-limited |
| walking_left | all | FAIL 0,0 (0.00,0.00) | FAIL 2,2 (0.02,0.02) | FAIL 2,2 (0.02,0.02) | FAIL 2,2 (0.02,0.02) | not run | not run | FAIL 20,0 (0.86,0.00) | controller-limited | not run | controller-limited |
| walking_right | all | FAIL 0,0 (0.00,0.00) | FAIL 5,5 (0.10,0.10) | FAIL 7,6 (0.18,0.14) | FAIL 7,7 (0.18,0.18) | not run | not run | FAIL 20,0 (0.86,0.00) | controller-limited | not run | controller-limited |

- R0 push/standing_forward: controller-limited
- R4/42 push/standing_forward: not attributed (falls; attribution is applied to R2, R3 rows) (by design); checkpoint 1253376
- R0 push/standing_backward: controller-limited
- R0 push/walking_forward: not attributed (falls; attribution is applied to R2, R3 rows) (by design)
- R1 push/walking_forward: not attributed (falls; attribution is applied to R2, R3 rows) (by design)
- R2/42 push/walking_forward: controller-limited (decision 20; untestable candidates: cop_at_pad_edge_with_ankle_saturated); also controller-limited
- R2/43 push/walking_forward: controller-limited (decision 20; untestable candidates: cop_at_pad_edge_with_ankle_saturated); also controller-limited
- R4/42 push/walking_forward: not attributed (falls; attribution is applied to R2, R3 rows) (by design); checkpoint 1253376
- R0 push/walking_backward: not attributed (falls; attribution is applied to R2, R3 rows) (by design)
- R1 push/walking_backward: not attributed (falls; attribution is applied to R2, R3 rows) (by design)
- R2/42 push/walking_backward: body-limited: slip
- R2/43 push/walking_backward: controller-limited (decision 20; untestable candidates: cop_at_pad_edge_with_ankle_saturated)
- R4/42 push/walking_backward: controller-limited; checkpoint 1253376
- R0 push/walking_left: not attributed (falls; attribution is applied to R2, R3 rows) (by design)
- R1 push/walking_left: not attributed (falls; attribution is applied to R2, R3 rows) (by design)
- R2/42 push/walking_left: body-limited: drive_torque
- R2/43 push/walking_left: controller-limited (decision 20; untestable candidates: cop_at_pad_edge_with_ankle_saturated)
- R4/42 push/walking_left: controller-limited; checkpoint 1253376
- R0 push/walking_right: not attributed (falls; attribution is applied to R2, R3 rows) (by design)
- R1 push/walking_right: not attributed (falls; attribution is applied to R2, R3 rows) (by design)
- R2/42 push/walking_right: controller-limited (decision 20; untestable candidates: cop_at_pad_edge_with_ankle_saturated); also controller-limited
- R2/43 push/walking_right: controller-limited (decision 20; untestable candidates: cop_at_pad_edge_with_ankle_saturated)
- R4/42 push/walking_right: controller-limited; checkpoint 1253376

#### Carpet

| Condition | Gated for | R0 | R1 | R2/42 | R2/43 | R3/42 | R3/43 | R4/42 | R2 verdict (shown) | R3 verdict (not judged: stopped) | R4 verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| robot_floor_carpet | R3/42, R3/43, R4/42 | n/a | n/a | FAIL 0,0 (0.00,0.00) (rep.) | FAIL 0,0 (0.00,0.00) (rep.) | not run | not run | FAIL 20,0 (0.86,0.00) | controller-limited (rep.) | not run | body-limited |
| low | R3/42, R3/43, R4/42 | n/a | n/a | FAIL 0,0 (0.00,0.00) (rep.) | FAIL 0,0 (0.00,0.00) (rep.) | not run | not run | FAIL 20,0 (0.86,0.00) | controller-limited (rep.) | not run | body-limited |
| medium | none (reported) | n/a | n/a | no stance | no stance | not run | not run | no stance | not run (rep.) | not run (rep.) | not run (rep.) |
| high | none (reported) | n/a | n/a | no stance | no stance | not run | not run | no stance | not run (rep.) | not run (rep.) | not run (rep.) |

The tracker (R0-R2) was never fitted on carpet (decision 23: the fit and G5 ran on hard floor only, and R2 fell in every start tried on robot_floor_carpet and low pile): the carpet walk is reported for R0-R2, with its fall attribution, and gated for R3 and R4.

- R2/42 carpet/robot_floor_carpet: controller-limited (decision 20; untestable candidates: cop_at_pad_edge_with_ankle_saturated)
- R2/43 carpet/robot_floor_carpet: controller-limited (decision 20; untestable candidates: cop_at_pad_edge_with_ankle_saturated)
- R4/42 carpet/robot_floor_carpet: body-limited: slip_median, slip_worst; checkpoint 1253376
- R2/42 carpet/low: controller-limited (decision 20; untestable candidates: cop_at_pad_edge_with_ankle_saturated)
- R2/43 carpet/low: controller-limited (decision 20; untestable candidates: cop_at_pad_edge_with_ankle_saturated)
- R4/42 carpet/low: body-limited: slip_median, slip_worst; checkpoint 1253376

Provenance of each row (PROVENANCE_RULE):

- R0 = R0: feedforward + PD on the commanded clock (ROWS_RULE)
- R1 = R1: the tracker on the true state, the body-level G5 selection (seed43_gen60)
- R2/42 = R2: the tracker on the estimate, CMA-ES seed 42's G5 selection seed42_gen60
- R2/43 = R2: the tracker on the estimate, CMA-ES seed 43's G5 selection seed43_gen60
- R4/42 = R4: PPO seed 42's policy-only policy at 1,253,376 steps, the joint targets held by the servos alone (no tracker); checkpoint 1253376 steps (final)

### Looks (sway, head, tail)

Reported, not gated (LOOKS_RULE): sway against 0.12 H, head and tail amplitude against BEHAVIOR_SPEC's bands (head 0.05-0.15 rad, tail 0.1-0.2 rad).

| Condition | Row | Sway mean / max (H) | Head amplitude mean [range] (rad), in band | Tail amplitude mean [range] (rad), in band |
|---|---|---|---|---|
| walk/nominal | R0 | 0.184 / 0.188 | 0.051 [0.051-0.051], 1/1 | 0.120 [0.120-0.120], 1/1 |
| walk/nominal | R1 | 0.179 / 0.195 | 0.051 [0.051-0.051], 18/18 | 0.120 [0.120-0.120], 18/18 |
| walk/nominal | R2/42 | 0.179 / 0.191 | 0.051 [0.051-0.051], 20/20 | 0.120 [0.120-0.121], 20/20 |
| walk/nominal | R2/43 | 0.178 / 0.188 | 0.051 [0.051-0.051], 20/20 | 0.120 [0.120-0.121], 20/20 |
| walk/nominal | R4/42 | 0.008 / 0.009 | 0.001 [0.001-0.002], 0/20 | 0.001 [0.001-0.001], 0/20 |
| walk/brisk | R0 | - | - | - |
| walk/brisk | R1 | 0.134 / 0.162 | 0.052 [0.051-0.053], 15/15 | 0.121 [0.121-0.121], 15/15 |
| walk/brisk | R2/42 | 0.136 / 0.148 | 0.052 [0.051-0.052], 5/5 | 0.121 [0.121-0.122], 5/5 |
| walk/brisk | R2/43 | 0.137 / 0.173 | 0.052 [0.051-0.052], 20/20 | 0.121 [0.121-0.121], 20/20 |
| walk/brisk | R4/42 | 0.008 / 0.010 | 0.001 [0.001-0.002], 0/20 | 0.001 [0.001-0.001], 0/20 |
| walk/fr05 | R0 | - | - | - |
| walk/fr05 | R1 | - | - | - |
| walk/fr05 | R2/42 | - | - | - |
| walk/fr05 | R2/43 | - | - | - |
| walk/fr05 | R4/42 | 0.007 / 0.012 | 0.001 [0.001-0.001], 0/20 | 0.001 [0.001-0.001], 0/20 |
| walk/nominal_mu0.54 | R0 | 0.184 / 0.193 | 0.051 [0.051-0.051], 7/7 | 0.120 [0.120-0.121], 7/7 |
| walk/nominal_mu0.54 | R1 | 0.183 / 0.246 | 0.051 [0.051-0.053], 12/12 | 0.120 [0.120-0.121], 12/12 |
| walk/nominal_mu0.54 | R2/42 | 0.180 / 0.191 | 0.051 [0.051-0.052], 20/20 | 0.121 [0.120-0.121], 20/20 |
| walk/nominal_mu0.54 | R2/43 | 0.179 / 0.189 | 0.051 [0.051-0.051], 20/20 | 0.120 [0.120-0.121], 20/20 |
| walk/nominal_mu0.54 | R4/42 | 0.008 / 0.009 | 0.001 [0.001-0.002], 0/20 | 0.001 [0.001-0.001], 0/20 |
| walk/brisk_mu0.54 | R0 | - | - | - |
| walk/brisk_mu0.54 | R1 | 0.136 / 0.185 | 0.052 [0.052-0.053], 13/13 | 0.121 [0.121-0.122], 13/13 |
| walk/brisk_mu0.54 | R2/42 | 0.146 / 0.195 | 0.052 [0.052-0.054], 6/6 | 0.122 [0.121-0.123], 6/6 |
| walk/brisk_mu0.54 | R2/43 | 0.141 / 0.189 | 0.052 [0.051-0.054], 17/17 | 0.122 [0.121-0.122], 17/17 |
| walk/brisk_mu0.54 | R4/42 | 0.008 / 0.010 | 0.001 [0.001-0.001], 0/20 | 0.001 [0.001-0.001], 0/20 |
| walk/fr05_mu0.54 | R0 | - | - | - |
| walk/fr05_mu0.54 | R1 | - | - | - |
| walk/fr05_mu0.54 | R2/42 | - | - | - |
| walk/fr05_mu0.54 | R2/43 | 0.139 / 0.178 | 0.052 [0.052-0.052], 2/2 | 0.123 [0.123-0.123], 2/2 |
| walk/fr05_mu0.54 | R4/42 | 0.007 / 0.012 | 0.001 [0.001-0.001], 0/20 | 0.001 [0.001-0.001], 0/20 |
| carpet/robot_floor_carpet | R2/42 | - | - | - |
| carpet/robot_floor_carpet | R2/43 | - | - | - |
| carpet/robot_floor_carpet | R4/42 | 0.014 / 0.017 | 0.002 [0.001-0.002], 0/20 | 0.001 [0.001-0.001], 0/20 |
| carpet/low | R2/42 | - | - | - |
| carpet/low | R2/43 | - | - | - |
| carpet/low | R4/42 | 0.006 / 0.008 | 0.001 [0.001-0.002], 0/20 | 0.001 [0.001-0.001], 0/20 |

### Margin ledger

Cells: the threshold (the last value at which all 3 seeds survive and none fails a term it passed at nominal, or survive) and the nearest value that fails with the label there (falls labelled by lifting each candidate limit on the 3 seeds, ledger.LIFT_RULE); '*' the search is not finished or the record is flagged (listed below).

| Margin | Search | R2/43 works | R3/42 works | R2/43 survives | R3/42 survives | Stage 0 |
|---|---|---|---|---|---|---|
| Grip | walk_nominal | 0.275 (fails 0.2688; controller-limited) | not run | 0.1125 (fails 0.1063; body-limited: slip (lifted, all 3 seeds survive; untestable: cop_at_pad_edge_with_ankle_saturated)) | not run | 0.47 |
| Grip | turn_left | 0.4437 (fails 0.4375; body-limited: joint_range) | not run | 0.275 (fails 0.2688; body-limited: slip (lifted, all 3 seeds survive; untestable: cop_at_pad_edge_with_ankle_saturated)) | not run | 0.53 |
| Grip | turn_right | 0.5687 (fails 0.5625; controller-limited) | not run | 0.3 (fails 0.2937; body-limited: slip (lifted, all 3 seeds survive; untestable: cop_at_pad_edge_with_ankle_saturated)) | not run | 0.53 |
| Grip | walk_brisk | 0.575 (fails 0.5687; body-limited: slip (lifted, all 3 seeds survive; untestable: cop_at_pad_edge_with_ankle_saturated)) | not run | 0.5312 (fails 0.525; body-limited: slip (lifted, all 3 seeds survive; untestable: cop_at_pad_edge_with_ankle_saturated)) | not run | 0.73 |
| Grip | turn (worse of turn_left, turn_right) | 0.5687 (fails 0.5625; controller-limited) [turn_right] | not run | 0.3 (fails 0.2937; body-limited: slip (lifted, all 3 seeds survive; untestable: cop_at_pad_edge_with_ankle_saturated)) [turn_right] | not run | 0.53 |
| Push | standing_forward | 0.2539 (fails 0.2614; controller-limited (falls with no relaxable hardware limit active before the onset: nothing to lift, decision 20; untestable: cop_at_pad_edge_with_ankle_saturated)) | not run | 0.2539 (fails 0.2614; controller-limited (falls with no relaxable hardware limit active before the onset: nothing to lift, decision 20; untestable: cop_at_pad_edge_with_ankle_saturated)) | not run | 0.478 |
| Push | standing_backward | 0.2614 (fails 0.2689; controller-limited (falls with no relaxable hardware limit active before the onset: nothing to lift, decision 20; untestable: cop_at_pad_edge_with_ankle_saturated)) | not run | 0.2614 (fails 0.2689; controller-limited (falls with no relaxable hardware limit active before the onset: nothing to lift, decision 20; untestable: cop_at_pad_edge_with_ankle_saturated)) | not run | (forward 0.478) |
| Push | standing_left | 0.5452 (fails 0.5527; controller-limited (falls with no relaxable hardware limit active before the onset: nothing to lift, decision 20; untestable: cop_at_pad_edge_with_ankle_saturated)) | not run | 0.5452 (fails 0.5527; controller-limited (falls with no relaxable hardware limit active before the onset: nothing to lift, decision 20; untestable: cop_at_pad_edge_with_ankle_saturated)) | not run | (forward 0.478) |
| Push | standing_right | 0.5527 (fails 0.5602; body-limited: slip (lifted, all 3 seeds survive; untestable: cop_at_pad_edge_with_ankle_saturated)) | not run | 0.5527 (fails 0.5602; body-limited: slip (lifted, all 3 seeds survive; untestable: cop_at_pad_edge_with_ankle_saturated)) | not run | (forward 0.478) |
| Push | walking_forward | 0.1195 (fails 0.127; controller-limited (falls with no relaxable hardware limit active before the onset: nothing to lift, decision 20; untestable: cop_at_pad_edge_with_ankle_saturated)) | not run | 0.1195 (fails 0.127; controller-limited (falls with no relaxable hardware limit active before the onset: nothing to lift, decision 20; untestable: cop_at_pad_edge_with_ankle_saturated)) | not run | 0.478 |
| Push | walking_backward | 0.1344 (fails 0.1419; body-limited: slip (lifted, all 3 seeds survive; untestable: cop_at_pad_edge_with_ankle_saturated)) | not run | 0.1344 (fails 0.1419; body-limited: slip (lifted, all 3 seeds survive; untestable: cop_at_pad_edge_with_ankle_saturated)) | not run | (forward 0.478) |
| Push | walking_left | 0.07469 (fails 0.08216; controller-limited (lifting slip saves none of the falls; untestable: cop_at_pad_edge_with_ankle_saturated)) | not run | 0.07469 (fails 0.08216; controller-limited (lifting slip saves none of the falls; untestable: cop_at_pad_edge_with_ankle_saturated)) | not run | (forward 0.478) |
| Push | walking_right | 0.07469 (fails 0.08216; controller-limited (falls with no relaxable hardware limit active before the onset: nothing to lift, decision 20; untestable: cop_at_pad_edge_with_ankle_saturated)) | not run | 0.07469 (fails 0.08216; controller-limited (falls with no relaxable hardware limit active before the onset: nothing to lift, decision 20; untestable: cop_at_pad_edge_with_ankle_saturated)) | not run | (forward 0.478) |
| Stop | - | not run (plan deviation, decision 33) | not run (plan deviation, decision 33) | not run (plan deviation, decision 33) | not run (plan deviation, decision 33) | 1.25 |
| Drive speed | walk_fr05 | fails at nominal (controller-limited (lifting drive_torque, slip saves none of the falls; untestable: cop_at_pad_edge_with_ankle_saturated)) [panels: FAIL 0,0; body-limited: slip] | not run | fails at nominal (controller-limited (lifting drive_torque, slip saves none of the falls; untestable: cop_at_pad_edge_with_ankle_saturated)) [panels: FAIL 0,0; body-limited: slip] | not run | 0.95 (0.83 without the 15 % margin) |
| Drive torque | walk_nominal | 0.8359 (fails 0.8301; body-limited: drive_sustained) | not run | 0.4785 (fails 0.4727; body-limited: drive_torque, slip (lifted, all 3 seeds survive; untestable: cop_at_pad_edge_with_ankle_saturated)) | not run | 0.76 (walk RMS 0.76) |
| Model doubt: servo kp | walk_nominal | 0.6719 (fails 0.666; controller-limited) | not run | 0.5781 (fails 0.5723; controller-limited (falls with no relaxable hardware limit active before the onset: nothing to lift, decision 20; untestable: cop_at_pad_edge_with_ankle_saturated)) | not run | - |
| Model doubt: latency | walk_nominal | 2 steps (4 ms) (fails 3 steps (6 ms); controller-limited) | not run | 4 steps (8 ms) (fails 5 steps (10 ms); controller-limited (lifting drive_torque, slip saves none of the falls; untestable: cop_at_pad_edge_with_ankle_saturated)) | not run | - |
| Model doubt: trunk mass | walk_nominal | works to 1.3 (robot +13.2 %) | not run | works to 1.3 (robot +13.2 %) | not run | - |
| Model doubt: trunk mass | walk_nominal_trunk_x1.1 (trunk x1.1: robot +4.4 %) | works 3/3 | not run | survives 3/3 | not run | - |
| Model doubt: trunk mass | walk_nominal_robot_x1.1 (trunk x1.227: robot +10.0 %) | works 3/3 | not run | survives 3/3 | not run | - |
| Model doubt: pile damping | stand_low_down | works to 0.5 | not run | works to 0.5 | not run | - |
| Model doubt: pile damping | stand_low_up | works to 2 | not run | works to 2 | not run | - |
| Model doubt: pile damping | carpet_low_down | fails at nominal (controller-limited (lifting slip saves none of the falls; untestable: cop_at_pad_edge_with_ankle_saturated)) [panels: FAIL 0,0; controller-limited (decision 20; untestable candidates: cop_at_pad_edge_with_ankle_saturated)] | not run | fails at nominal (controller-limited (lifting slip saves none of the falls; untestable: cop_at_pad_edge_with_ankle_saturated)) [panels: FAIL 0,0; controller-limited (decision 20; untestable candidates: cop_at_pad_edge_with_ankle_saturated)] | not run | - |
| Model doubt: pile damping | carpet_low_up | fails at nominal (controller-limited (lifting slip saves none of the falls; untestable: cop_at_pad_edge_with_ankle_saturated)) [panels: FAIL 0,0; controller-limited (decision 20; untestable candidates: cop_at_pad_edge_with_ankle_saturated)] | not run | fails at nominal (controller-limited (lifting slip saves none of the falls; untestable: cop_at_pad_edge_with_ankle_saturated)) [panels: FAIL 0,0; controller-limited (decision 20; untestable candidates: cop_at_pad_edge_with_ankle_saturated)] | not run | - |
| Pack voltage | walk_nominal_25.2V | works 3/3 = nominal plant | not run | survives 3/3 | not run | - |
| Pack voltage | walk_nominal_22.2V | works 3/3 | not run | survives 3/3 | not run | - |
| Pack voltage | walk_nominal_21V | works 3/3 | not run | survives 3/3 | not run | - |
| Pack voltage | walk_fr05_25.2V | fails 0/3 = nominal plant (controller-limited (lifting drive_torque, slip saves none of the falls; untestable: cop_at_pad_edge_with_ankle_saturated)) | not run | falls 0/3 | not run | - |
| Pack voltage | walk_fr05_22.2V | fails 0/3 (controller-limited (lifting drive_torque, envelope, slip saves none of the falls; untestable: cop_at_pad_edge_with_ankle_saturated)) | not run | falls 0/3 | not run | - |
| Pack voltage | walk_fr05_21V | fails 0/3 (controller-limited (lifting drive_torque, envelope, slip saves none of the falls; untestable: cop_at_pad_edge_with_ankle_saturated)) | not run | falls 0/3 | not run | - |

- R3 is judged with R2 for this body (R3/42 stopped at 2015232 steps: G6b failed again after the fix (FIX_RULE) (decision 32); R3/43 stopped at 1753088 steps: the body's R3 is judged with R2 (train.R2_JUDGED_RULE, decision 38): its seed 42 run was stopped by FIX_RULE at 2015232 steps (G6b failed again after the fix), so this run cannot change a verdict; stopped to give the 75 cm runs the cores (decision 32)): its ledger row is not run (ledger.ROWS_RULE)
- Grip: Stage 0: the lowest friction the round-two turning screen requires in single support, per body: nominal walk / 0.35 rad/s walking turn (the worse direction) / brisk walk (STAGE0_RULE)
- Stop: not run (plan deviation, decision 33): the deceleration a stop holds needs stop references that decelerate at other rates, and the run holds one stop transition (STOP_RULE)
- Drive speed: Stage 0: the knee's speed need at Fr 0.05 over its rated speed, including the screens' 15 % speed margin; without it 0.83 (0.81 / 0.83 / 0.81), which is what the ledger's limit compares (STAGE0_RULE)
- Drive torque: Stage 0: the worst drive's sustained-torque ratio, the larger of the nominal walk's RMS and the one-foot hold (ankle_a's one-foot hold); the nominal walk's RMS alone 0.76 (neck), which is what the ledger's limit compares (STAGE0_RULE)
- Pack voltage: Stage 0: Stage 0 used 24 V; at 22.2 V the knee's rated speed (8.04 rad/s) equals the Fr 0.05 requirement including the screens' 15 % speed margin (8.08-8.12 rad/s; the peak swing speed itself is about 7.0 rad/s)

### PPO runs

| Run | Status | Steps / budget | Last checkpoint | Judged checkpoint | G6a zero residual | G6a env steps/s | G6b | Fix; G6b after it | G6c | Job h | CPU h | run.json written |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R3/42 | stopped_g6b | 2015232 / 4005888 | 2015232 | 2506752 (final) | yes | 81 (FAIL: cut the steps equally for every body, decision 26) | FAIL at 1007616 (survival policy 15.03 s, zero 15.49 s): make one fix, then stop | from 1007616 (task_weights 1,1,1,1,1,2,2,1,1); FAIL at 2015232 (survival policy 14.30 s, zero 15.49 s): stop the run (FIX_RULE) | - | 17.74 (running) | running | 2026-11-10T00:12:29+00:00 |
| R3/43 | stopped_r2_judged | 1753088 / 4005888 | 1753088 | 2506752 (final) | yes | 83 (FAIL: cut the steps equally for every body, decision 26) | pass at 1007616 (survival policy 15.92 s, zero 15.49 s) | from 1007616 (task_weights 1,1,1,1,1,2,2,1,1); - | - | 19.54 (running) | running | 2026-11-10T00:12:29+00:00 |
| R4/42 | running | 1507328 / 4005888 | 1507328 | 1253376 (final) | - | 115 (FAIL: cut the steps equally for every body, decision 26) | - | - | - | 10.33 (running) | running | 2026-11-10T00:12:30+00:00 |

Elapsed PPO wall clock (the union of every run's sessions): 10.33 h; job-hours (their sum: the runs share the cores) 47.61 h.

### Fit and G5

G5 pass (seeds passing: [42, 43]); body-level selection (env.GAINS_RULE): CMA-ES seed 43, seed43_gen60 (passed).

| CMA-ES seed | Fit | Evaluations | Sigma | Selected | Walk | Walking turn | In-place turn | G5 |
|---|---|---|---|---|---|---|---|---|
| 42 | done | 720 | 0.0372 | seed42_gen60 | 18/20 (surv 20) | 20/20 (surv 20) | 19/20 (surv 20) | passes G5 |
| 43 | done | 720 | 0.0537 | seed43_gen60 | 19/20 (surv 20) | 18/20 (surv 20) | 19/20 (surv 20) | passes G5 |

### Core-hours per step

| Step | Wall h | Core-h | Basis |
|---|---|---|---|
| prepare | 0.14 | 0.14 | wall, one process |
| gates | 0.07 | 0.07 | wall, one process |
| fit | 5.26 | 5.26 | sessions' end - start x workers |
| g5 | 0.38 | 0.38 | tune.json seconds, one process assumed (workers not recorded): a lower bound |
| ppo | 10.33 (job-hours 47.61) | 19.04 | finished sessions' CPU, and for a running session its wall to the record's last write x the CPU per wall second of its first rollouts (an estimate); wall: the elapsed union of the sessions, job-hours beside |
| panels | 3.95 | 4.08 | episodes' and attributions' wall seconds (one process each): an upper bound on CPU |
| ledger | 0.31 | 0.35 | episodes' CPU seconds, each evaluation once (by its key) however many margins use it |
| total | | 29.32 | core-hours recorded so far |

## 75-M-H-10

### Verdicts by controller

| Controller | shown | provisional | body-limited | controller-limited | not attributed | fails by falls (R4: not attributed) | incomplete | not run |
|---|---|---|---|---|---|---|---|---|
| R2 | 14 | 0 | 3 | 3 | 0 | 0 | 0 | 0 |
| R3 (shown) | 12 | 2 | 5 | 3 | 0 | 0 | 0 | 0 |
| R4 | 0 | 6 | 10 | 5 | 0 | 1 | 0 | 0 |

Counts over the conditions gated for each controller's rows.
R4 has one seed (decisions 3 and 28): its verdicts are at most provisional (policy_rows.R4_VERDICT_RULE); its record on each gated condition: FAIL 16, pass 6.

G7 pass: sway 0.1641 H (Stage 0 0.1641, plan 0.164), cadence 1.7612 steps/s (Stage 0 1.7612, plan 1.76).

### Verdict grid

Verdicts shown: R3: the controller judged on every behaviour (plan section 2); G6c passed at 2506752 steps.  Cells: result, survived,succeeded of 20 (their one-sided 95 % Clopper-Pearson lower bounds); '(rep.)' reported, not gated, for that row; '*' interim (listed below each table); 'not run' no record; 'no stance' the floor has no settled stance for the body (panels.NOT_RUN_RULE); 'n/a' the row does not run there; R4's verdict is at most provisional (one seed), and its falls are never attributed (by design: decision 20's attribution is for R2 and R3).

#### Stand

| Condition | Gated for | R0 | R1 | R2/42 | R2/43 | R3/42 | R3/43 | R4/42 | R2 verdict | R3 verdict (shown) | R4 verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| hard | all | pass 20,20 (0.86,0.86) | pass 20,20 (0.86,0.86) | pass 20,20 (0.86,0.86) | pass 20,20 (0.86,0.86) | pass 20,20 (0.86,0.86) | pass 20,20 (0.86,0.86) | pass 20,20 (0.86,0.86) | shown | shown | provisional |
| robot_floor_carpet | all | n/a | n/a | pass 20,20 (0.86,0.86) | pass 20,20 (0.86,0.86) | pass 20,20 (0.86,0.86) | pass 20,20 (0.86,0.86) | pass 20,20 (0.86,0.86) | shown | shown | provisional |
| low | all | n/a | n/a | pass 20,20 (0.86,0.86) | pass 20,20 (0.86,0.86) | pass 20,20 (0.86,0.86) | pass 20,20 (0.86,0.86) | pass 20,20 (0.86,0.86) | shown | shown | provisional |

#### Walk

| Condition | Gated for | R0 | R1 | R2/42 | R2/43 | R3/42 | R3/43 | R4/42 | R2 verdict | R3 verdict (shown) | R4 verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| nominal | all | pass 20,20 (0.86,0.86) | pass 20,20 (0.86,0.86) | pass 20,20 (0.86,0.86) | pass 20,20 (0.86,0.86) | pass 20,20 (0.86,0.86) | pass 20,20 (0.86,0.86) | FAIL 20,0 (0.86,0.00) | shown | shown | body-limited |
| brisk | all | FAIL 20,0 (0.86,0.00) | FAIL 20,0 (0.86,0.00) | FAIL 20,0 (0.86,0.00) | FAIL 20,0 (0.86,0.00) | FAIL 20,0 (0.86,0.00) | FAIL 20,0 (0.86,0.00) | FAIL 20,0 (0.86,0.00) | body-limited | body-limited | body-limited |
| fr05 | all | FAIL 16,0 (0.60,0.00) | FAIL 20,0 (0.86,0.00) | FAIL 20,0 (0.86,0.00) | FAIL 20,0 (0.86,0.00) | FAIL 20,0 (0.86,0.00) | FAIL 20,0 (0.86,0.00) | FAIL 20,0 (0.86,0.00) | body-limited | body-limited | body-limited |
| nominal_mu0.54 | all | pass 20,20 (0.86,0.86) | pass 20,20 (0.86,0.86) | pass 20,20 (0.86,0.86) | pass 20,20 (0.86,0.86) | pass 20,20 (0.86,0.86) | pass 20,20 (0.86,0.86) | FAIL 20,0 (0.86,0.00) | shown | shown | body-limited |
| brisk_mu0.54 | none (reported) | FAIL 20,0 (0.86,0.00) (rep.) | FAIL 20,0 (0.86,0.00) (rep.) | FAIL 20,0 (0.86,0.00) (rep.) | FAIL 20,0 (0.86,0.00) (rep.) | FAIL 20,0 (0.86,0.00) (rep.) | FAIL 20,0 (0.86,0.00) (rep.) | FAIL 20,0 (0.86,0.00) (rep.) | body-limited (rep.) | body-limited (rep.) | body-limited (rep.) |
| fr05_mu0.54 | none (reported) | FAIL 16,0 (0.60,0.00) (rep.) | FAIL 20,0 (0.86,0.00) (rep.) | FAIL 20,0 (0.86,0.00) (rep.) | FAIL 20,0 (0.86,0.00) (rep.) | FAIL 20,0 (0.86,0.00) (rep.) | FAIL 18,0 (0.72,0.00) (rep.) | FAIL 20,0 (0.86,0.00) (rep.) | body-limited (rep.) | body-limited (rep.) | body-limited (rep.) |

- R4/42 walk/nominal: body-limited: slip_median, slip_worst; checkpoint 1253376
- R0 walk/brisk: body-limited: drive_sustained
- R1 walk/brisk: body-limited: drive_sustained
- R2/42 walk/brisk: body-limited: drive_sustained
- R2/43 walk/brisk: body-limited: drive_sustained
- R3/42 walk/brisk: body-limited: drive_sustained; checkpoint 2506752
- R3/43 walk/brisk: body-limited: drive_sustained; checkpoint 2506752
- R4/42 walk/brisk: body-limited: slip_median, slip_worst; checkpoint 1253376
- R0 walk/fr05: body-limited: drive_sustained; also not attributed (falls; attribution is applied to R2, R3 rows)
- R1 walk/fr05: body-limited: drive_sustained
- R2/42 walk/fr05: body-limited: drive_sustained
- R2/43 walk/fr05: body-limited: drive_sustained, drive_speed
- R3/42 walk/fr05: body-limited: drive_sustained; checkpoint 2506752
- R3/43 walk/fr05: body-limited: drive_sustained, drive_speed; checkpoint 2506752
- R4/42 walk/fr05: body-limited: slip_median, slip_worst; checkpoint 1253376
- R4/42 walk/nominal_mu0.54: body-limited: slip_median, slip_worst; checkpoint 1253376
- R0 walk/brisk_mu0.54: body-limited: drive_sustained
- R1 walk/brisk_mu0.54: body-limited: drive_sustained
- R2/42 walk/brisk_mu0.54: body-limited: drive_sustained
- R2/43 walk/brisk_mu0.54: body-limited: drive_sustained
- R3/42 walk/brisk_mu0.54: body-limited: drive_sustained; checkpoint 2506752
- R3/43 walk/brisk_mu0.54: body-limited: drive_sustained; checkpoint 2506752
- R4/42 walk/brisk_mu0.54: body-limited: slip_median, slip_worst; checkpoint 1253376
- R0 walk/fr05_mu0.54: body-limited: drive_sustained; also not attributed (falls; attribution is applied to R2, R3 rows)
- R1 walk/fr05_mu0.54: body-limited: drive_sustained
- R2/42 walk/fr05_mu0.54: body-limited: drive_sustained
- R2/43 walk/fr05_mu0.54: body-limited: drive_sustained, drive_speed, slip_median
- R3/42 walk/fr05_mu0.54: body-limited: drive_sustained, slip_median; checkpoint 2506752
- R3/43 walk/fr05_mu0.54: body-limited: drive_sustained, drive_speed, slip_median; also controller-limited (decision 20; 2 falls: the sign test needs 5 for p < 0.05, so no limit was lifted; untestable candidates: cop_at_pad_edge_with_ankle_saturated); checkpoint 2506752
- R4/42 walk/fr05_mu0.54: body-limited: slip_median, slip_worst; checkpoint 1253376

#### Turn

| Condition | Gated for | R0 | R1 | R2/42 | R2/43 | R3/42 | R3/43 | R4/42 | R2 verdict | R3 verdict (shown) | R4 verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| walking_0.35_left | all | pass 20,20 (0.86,0.86) | pass 20,20 (0.86,0.86) | pass 20,20 (0.86,0.86) | pass 20,20 (0.86,0.86) | pass 20,20 (0.86,0.86) | pass 20,20 (0.86,0.86) | FAIL 20,0 (0.86,0.00) | shown | shown | body-limited |
| walking_0.35_right | all | pass 20,20 (0.86,0.86) | pass 20,20 (0.86,0.86) | pass 20,20 (0.86,0.86) | pass 20,20 (0.86,0.86) | pass 20,20 (0.86,0.86) | FAIL 20,6 (0.86,0.14) | FAIL 20,0 (0.86,0.00) | shown | provisional | body-limited |
| in_place_0.50_left | all | pass 20,20 (0.86,0.86) | pass 20,20 (0.86,0.86) | pass 20,20 (0.86,0.86) | pass 20,20 (0.86,0.86) | pass 20,20 (0.86,0.86) | pass 20,20 (0.86,0.86) | FAIL 20,0 (0.86,0.00) | shown | shown | body-limited |
| in_place_0.50_right | all | pass 20,20 (0.86,0.86) | pass 20,20 (0.86,0.86) | pass 20,20 (0.86,0.86) | pass 20,20 (0.86,0.86) | pass 20,20 (0.86,0.86) | pass 20,20 (0.86,0.86) | FAIL 20,0 (0.86,0.00) | shown | shown | body-limited |
| walking_0.50_left | none (reported) | pass 20,20 (0.86,0.86) (rep.) | pass 20,20 (0.86,0.86) (rep.) | pass 20,20 (0.86,0.86) (rep.) | pass 20,20 (0.86,0.86) (rep.) | pass 20,20 (0.86,0.86) (rep.) | pass 20,20 (0.86,0.86) (rep.) | FAIL 20,0 (0.86,0.00) (rep.) | shown (rep.) | shown (rep.) | body-limited (rep.) |
| walking_0.50_right | none (reported) | pass 20,20 (0.86,0.86) (rep.) | pass 20,20 (0.86,0.86) (rep.) | pass 20,20 (0.86,0.86) (rep.) | pass 20,20 (0.86,0.86) (rep.) | pass 20,20 (0.86,0.86) (rep.) | pass 20,20 (0.86,0.86) (rep.) | FAIL 20,0 (0.86,0.00) (rep.) | shown (rep.) | shown (rep.) | body-limited (rep.) |

- R4/42 turn/walking_0.35_left: body-limited: slip_median, slip_worst; checkpoint 1253376
- R3/43 turn/walking_0.35_right: controller-limited; checkpoint 2506752
- R4/42 turn/walking_0.35_right: body-limited: slip_median, slip_worst; checkpoint 1253376
- R4/42 turn/in_place_0.50_left: body-limited: slip_median, slip_worst; checkpoint 1253376
- R4/42 turn/in_place_0.50_right: body-limited: slip_median, slip_worst; checkpoint 1253376
- R4/42 turn/walking_0.50_left: body-limited: slip_median, slip_worst; checkpoint 1253376
- R4/42 turn/walking_0.50_right: body-limited: slip_median, slip_worst; checkpoint 1253376

#### Start and stop

| Condition | Gated for | R0 | R1 | R2/42 | R2/43 | R3/42 | R3/43 | R4/42 | R2 verdict | R3 verdict (shown) | R4 verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| twice | all | pass 20,20 (0.86,0.86) | pass 20,20 (0.86,0.86) | pass 20,20 (0.86,0.86) | pass 20,20 (0.86,0.86) | pass 20,20 (0.86,0.86) | FAIL 20,11 (0.86,0.35) | FAIL 20,0 (0.86,0.00) | shown | provisional | controller-limited |

- R3/43 start_stop/twice: controller-limited; checkpoint 2506752
- R4/42 start_stop/twice: controller-limited; checkpoint 1253376

#### Push

| Condition | Gated for | R0 | R1 | R2/42 | R2/43 | R3/42 | R3/43 | R4/42 | R2 verdict | R3 verdict (shown) | R4 verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| standing_forward | all | FAIL 20,0 (0.86,0.00) | pass 20,20 (0.86,0.86) | pass 20,20 (0.86,0.86) | pass 20,20 (0.86,0.86) | pass 20,20 (0.86,0.86) | pass 20,20 (0.86,0.86) | FAIL 10,9 (0.30,0.26) | shown | shown | fails by falls (R4: not attributed) |
| standing_backward | all | FAIL 20,0 (0.86,0.00) | pass 20,20 (0.86,0.86) | pass 20,20 (0.86,0.86) | pass 20,20 (0.86,0.86) | pass 20,20 (0.86,0.86) | pass 20,20 (0.86,0.86) | pass 20,18 (0.86,0.72) | shown | shown | provisional |
| standing_left | all | pass 20,20 (0.86,0.86) | pass 20,20 (0.86,0.86) | pass 20,20 (0.86,0.86) | pass 20,20 (0.86,0.86) | pass 20,20 (0.86,0.86) | pass 20,20 (0.86,0.86) | pass 20,18 (0.86,0.72) | shown | shown | provisional |
| standing_right | all | pass 20,20 (0.86,0.86) | pass 20,20 (0.86,0.86) | pass 20,20 (0.86,0.86) | pass 20,20 (0.86,0.86) | pass 20,20 (0.86,0.86) | pass 20,20 (0.86,0.86) | pass 20,18 (0.86,0.72) | shown | shown | provisional |
| walking_forward | all | FAIL 1,0 (0.00,0.00) | FAIL 1,1 (0.00,0.00) | FAIL 1,0 (0.00,0.00) | FAIL 1,1 (0.00,0.00) | FAIL 2,1 (0.02,0.00) | FAIL 1,0 (0.00,0.00) | FAIL 17,0 (0.66,0.00) | controller-limited | body-limited | controller-limited |
| walking_backward | all | FAIL 0,0 (0.00,0.00) | FAIL 0,0 (0.00,0.00) | FAIL 0,0 (0.00,0.00) | FAIL 0,0 (0.00,0.00) | FAIL 0,0 (0.00,0.00) | FAIL 0,0 (0.00,0.00) | FAIL 20,0 (0.86,0.00) | body-limited | body-limited | controller-limited |
| walking_left | all | FAIL 0,0 (0.00,0.00) | FAIL 5,5 (0.10,0.10) | FAIL 6,6 (0.14,0.14) | FAIL 5,4 (0.10,0.07) | FAIL 8,6 (0.22,0.14) | FAIL 6,5 (0.14,0.10) | FAIL 20,0 (0.86,0.00) | controller-limited | controller-limited | controller-limited |
| walking_right | all | FAIL 0,0 (0.00,0.00) | FAIL 7,6 (0.18,0.14) | FAIL 8,4 (0.22,0.07) | FAIL 4,2 (0.07,0.02) | FAIL 7,7 (0.18,0.18) | FAIL 6,3 (0.14,0.04) | FAIL 20,0 (0.86,0.00) | controller-limited | controller-limited | controller-limited |

- R0 push/standing_forward: controller-limited
- R4/42 push/standing_forward: not attributed (falls; attribution is applied to R2, R3 rows) (by design); also controller-limited; checkpoint 1253376
- R0 push/standing_backward: controller-limited
- R0 push/walking_forward: not attributed (falls; attribution is applied to R2, R3 rows) (by design); also controller-limited
- R1 push/walking_forward: not attributed (falls; attribution is applied to R2, R3 rows) (by design)
- R2/42 push/walking_forward: controller-limited (decision 20; untestable candidates: cop_at_pad_edge_with_ankle_saturated); also controller-limited
- R2/43 push/walking_forward: controller-limited (decision 20; untestable candidates: cop_at_pad_edge_with_ankle_saturated)
- R3/42 push/walking_forward: body-limited: slip; also controller-limited; checkpoint 2506752
- R3/43 push/walking_forward: body-limited: slip; also controller-limited; checkpoint 2506752
- R4/42 push/walking_forward: controller-limited; also not attributed (falls; attribution is applied to R2, R3 rows); checkpoint 1253376
- R0 push/walking_backward: not attributed (falls; attribution is applied to R2, R3 rows) (by design)
- R1 push/walking_backward: not attributed (falls; attribution is applied to R2, R3 rows) (by design)
- R2/42 push/walking_backward: body-limited: slip
- R2/43 push/walking_backward: body-limited: slip
- R3/42 push/walking_backward: body-limited: slip; checkpoint 2506752
- R3/43 push/walking_backward: body-limited: drive_torque, slip; checkpoint 2506752
- R4/42 push/walking_backward: controller-limited; checkpoint 1253376
- R0 push/walking_left: not attributed (falls; attribution is applied to R2, R3 rows) (by design)
- R1 push/walking_left: not attributed (falls; attribution is applied to R2, R3 rows) (by design)
- R2/42 push/walking_left: controller-limited (decision 20; untestable candidates: cop_at_pad_edge_with_ankle_saturated)
- R2/43 push/walking_left: controller-limited (decision 20; untestable candidates: cop_at_pad_edge_with_ankle_saturated); also body-limited: drive_speed
- R3/42 push/walking_left: controller-limited (decision 20; untestable candidates: cop_at_pad_edge_with_ankle_saturated); also controller-limited; checkpoint 2506752
- R3/43 push/walking_left: controller-limited (decision 20; untestable candidates: cop_at_pad_edge_with_ankle_saturated); also controller-limited; checkpoint 2506752
- R4/42 push/walking_left: controller-limited; checkpoint 1253376
- R0 push/walking_right: not attributed (falls; attribution is applied to R2, R3 rows) (by design)
- R1 push/walking_right: not attributed (falls; attribution is applied to R2, R3 rows) (by design); also controller-limited
- R2/42 push/walking_right: body-limited: slip; also controller-limited
- R2/43 push/walking_right: controller-limited (decision 20; untestable candidates: cop_at_pad_edge_with_ankle_saturated); also controller-limited
- R3/42 push/walking_right: controller-limited (decision 20; untestable candidates: cop_at_pad_edge_with_ankle_saturated); checkpoint 2506752
- R3/43 push/walking_right: controller-limited (decision 20; untestable candidates: cop_at_pad_edge_with_ankle_saturated); also controller-limited; checkpoint 2506752
- R4/42 push/walking_right: controller-limited; checkpoint 1253376

#### Carpet

| Condition | Gated for | R0 | R1 | R2/42 | R2/43 | R3/42 | R3/43 | R4/42 | R2 verdict | R3 verdict (shown) | R4 verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| robot_floor_carpet | R3/42, R3/43, R4/42 | n/a | n/a | FAIL 0,0 (0.00,0.00) (rep.) | FAIL 0,0 (0.00,0.00) (rep.) | FAIL 0,0 (0.00,0.00) | FAIL 0,0 (0.00,0.00) | FAIL 20,0 (0.86,0.00) | body-limited (rep.) | body-limited | body-limited |
| low | R3/42, R3/43, R4/42 | n/a | n/a | FAIL 0,0 (0.00,0.00) (rep.) | FAIL 0,0 (0.00,0.00) (rep.) | FAIL 14,0 (0.49,0.00) | FAIL 0,0 (0.00,0.00) | FAIL 20,0 (0.86,0.00) | controller-limited (rep.) | controller-limited | body-limited |
| medium | none (reported) | n/a | n/a | FAIL 0,0 (0.00,0.00) (rep.) | FAIL 0,0 (0.00,0.00) (rep.) | FAIL 0,0 (0.00,0.00) (rep.) | FAIL 0,0 (0.00,0.00) (rep.) | FAIL 20,0 (0.86,0.00) (rep.) | controller-limited (rep.) | controller-limited (rep.) | body-limited (rep.) |
| high | none (reported) | n/a | n/a | FAIL 0,0 (0.00,0.00) (rep.) | FAIL 0,0 (0.00,0.00) (rep.) | FAIL 0,0 (0.00,0.00) (rep.) | FAIL 0,0 (0.00,0.00) (rep.) | FAIL 20,0 (0.86,0.00) (rep.) | controller-limited (rep.) | controller-limited (rep.) | body-limited (rep.) |

The tracker (R0-R2) was never fitted on carpet (decision 23: the fit and G5 ran on hard floor only, and R2 fell in every start tried on robot_floor_carpet and low pile): the carpet walk is reported for R0-R2, with its fall attribution, and gated for R3 and R4.

- R2/42 carpet/robot_floor_carpet: body-limited: slip
- R2/43 carpet/robot_floor_carpet: body-limited: slip
- R3/42 carpet/robot_floor_carpet: body-limited: slip; checkpoint 2506752
- R3/43 carpet/robot_floor_carpet: body-limited: slip; checkpoint 2506752
- R4/42 carpet/robot_floor_carpet: body-limited: slip_median, slip_worst; checkpoint 1253376
- R2/42 carpet/low: controller-limited (decision 20; untestable candidates: cop_at_pad_edge_with_ankle_saturated)
- R2/43 carpet/low: controller-limited (decision 20; untestable candidates: cop_at_pad_edge_with_ankle_saturated)
- R3/42 carpet/low: body-limited: slip_median; also controller-limited (decision 20; untestable candidates: cop_at_pad_edge_with_ankle_saturated); checkpoint 2506752
- R3/43 carpet/low: controller-limited (decision 20; untestable candidates: cop_at_pad_edge_with_ankle_saturated); checkpoint 2506752
- R4/42 carpet/low: body-limited: slip_median, slip_worst; checkpoint 1253376
- R2/42 carpet/medium: controller-limited (decision 20; untestable candidates: cop_at_pad_edge_with_ankle_saturated)
- R2/43 carpet/medium: controller-limited (decision 20; untestable candidates: cop_at_pad_edge_with_ankle_saturated)
- R3/42 carpet/medium: controller-limited (decision 20; untestable candidates: cop_at_pad_edge_with_ankle_saturated); checkpoint 2506752
- R3/43 carpet/medium: controller-limited (decision 20; untestable candidates: cop_at_pad_edge_with_ankle_saturated); checkpoint 2506752
- R4/42 carpet/medium: body-limited: slip_median, slip_worst; checkpoint 1253376
- R2/42 carpet/high: controller-limited (decision 20; untestable candidates: cop_at_pad_edge_with_ankle_saturated)
- R2/43 carpet/high: controller-limited (decision 20; untestable candidates: cop_at_pad_edge_with_ankle_saturated)
- R3/42 carpet/high: controller-limited (decision 20; untestable candidates: cop_at_pad_edge_with_ankle_saturated); checkpoint 2506752
- R3/43 carpet/high: controller-limited (decision 20; untestable candidates: cop_at_pad_edge_with_ankle_saturated); checkpoint 2506752
- R4/42 carpet/high: body-limited: slip_median, slip_worst; checkpoint 1253376

Provenance of each row (PROVENANCE_RULE):

- R0 = R0: feedforward + PD on the commanded clock (ROWS_RULE)
- R1 = R1: the tracker on the true state, the body-level G5 selection (seed43_gen40)
- R2/42 = R2: the tracker on the estimate, CMA-ES seed 42's G5 selection seed42_gen60
- R2/43 = R2: the tracker on the estimate, CMA-ES seed 43's G5 selection seed43_gen40
- R3/42 = R3: R2 (the tracker on the estimate) with the body-level G5 selection seed43_gen40 plus PPO seed 42's residual policy at 2,506,752 steps; checkpoint 2506752 steps (final)
- R3/43 = R3: R2 (the tracker on the estimate) with the body-level G5 selection seed43_gen40 plus PPO seed 43's residual policy at 2,506,752 steps; checkpoint 2506752 steps (final)
- R4/42 = R4: PPO seed 42's policy-only policy at 1,253,376 steps, the joint targets held by the servos alone (no tracker); checkpoint 1253376 steps (final)

### Looks (sway, head, tail)

Reported, not gated (LOOKS_RULE): sway against 0.12 H, head and tail amplitude against BEHAVIOR_SPEC's bands (head 0.05-0.15 rad, tail 0.1-0.2 rad).

| Condition | Row | Sway mean / max (H) | Head amplitude mean [range] (rad), in band | Tail amplitude mean [range] (rad), in band |
|---|---|---|---|---|
| walk/nominal | R0 | 0.164 / 0.170 | 0.050 [0.050-0.050], 20/20 | 0.118 [0.118-0.118], 20/20 |
| walk/nominal | R1 | 0.167 / 0.169 | 0.050 [0.050-0.050], 20/20 | 0.119 [0.119-0.119], 20/20 |
| walk/nominal | R2/42 | 0.170 / 0.173 | 0.051 [0.050-0.051], 20/20 | 0.119 [0.119-0.119], 20/20 |
| walk/nominal | R2/43 | 0.168 / 0.171 | 0.051 [0.051-0.051], 20/20 | 0.119 [0.119-0.119], 20/20 |
| walk/nominal | R3/42 | 0.164 / 0.171 | 0.054 [0.054-0.054], 20/20 | 0.121 [0.120-0.121], 20/20 |
| walk/nominal | R3/43 | 0.177 / 0.188 | 0.053 [0.052-0.053], 20/20 | 0.120 [0.120-0.120], 20/20 |
| walk/nominal | R4/42 | 0.004 / 0.006 | 0.002 [0.002-0.002], 0/20 | 0.001 [0.001-0.001], 0/20 |
| walk/brisk | R0 | 0.130 / 0.137 | 0.051 [0.051-0.051], 20/20 | 0.121 [0.121-0.121], 20/20 |
| walk/brisk | R1 | 0.122 / 0.128 | 0.051 [0.050-0.051], 20/20 | 0.120 [0.120-0.120], 20/20 |
| walk/brisk | R2/42 | 0.124 / 0.130 | 0.051 [0.051-0.051], 20/20 | 0.121 [0.120-0.121], 20/20 |
| walk/brisk | R2/43 | 0.123 / 0.128 | 0.051 [0.050-0.051], 20/20 | 0.121 [0.121-0.121], 20/20 |
| walk/brisk | R3/42 | 0.118 / 0.130 | 0.054 [0.054-0.055], 20/20 | 0.123 [0.123-0.124], 20/20 |
| walk/brisk | R3/43 | 0.145 / 0.174 | 0.053 [0.053-0.054], 20/20 | 0.121 [0.121-0.121], 20/20 |
| walk/brisk | R4/42 | 0.003 / 0.004 | 0.002 [0.002-0.002], 0/20 | 0.001 [0.001-0.001], 0/20 |
| walk/fr05 | R0 | 0.127 / 0.169 | 0.052 [0.052-0.052], 16/16 | 0.119 [0.117-0.123], 16/16 |
| walk/fr05 | R1 | 0.111 / 0.119 | 0.051 [0.051-0.051], 20/20 | 0.123 [0.123-0.124], 20/20 |
| walk/fr05 | R2/42 | 0.115 / 0.128 | 0.051 [0.051-0.051], 20/20 | 0.124 [0.123-0.124], 20/20 |
| walk/fr05 | R2/43 | 0.127 / 0.188 | 0.051 [0.051-0.052], 20/20 | 0.122 [0.120-0.123], 20/20 |
| walk/fr05 | R3/42 | 0.112 / 0.137 | 0.054 [0.054-0.054], 20/20 | 0.124 [0.123-0.124], 20/20 |
| walk/fr05 | R3/43 | 0.143 / 0.193 | 0.054 [0.053-0.054], 20/20 | 0.117 [0.116-0.119], 20/20 |
| walk/fr05 | R4/42 | 0.002 / 0.004 | 0.002 [0.002-0.002], 0/20 | 0.001 [0.001-0.001], 0/20 |
| walk/nominal_mu0.54 | R0 | 0.164 / 0.167 | 0.050 [0.050-0.050], 20/20 | 0.119 [0.119-0.119], 20/20 |
| walk/nominal_mu0.54 | R1 | 0.167 / 0.169 | 0.050 [0.050-0.050], 20/20 | 0.119 [0.119-0.119], 20/20 |
| walk/nominal_mu0.54 | R2/42 | 0.170 / 0.174 | 0.051 [0.051-0.051], 20/20 | 0.119 [0.119-0.119], 20/20 |
| walk/nominal_mu0.54 | R2/43 | 0.168 / 0.172 | 0.051 [0.051-0.051], 20/20 | 0.119 [0.119-0.119], 20/20 |
| walk/nominal_mu0.54 | R3/42 | 0.166 / 0.175 | 0.054 [0.054-0.054], 20/20 | 0.121 [0.121-0.121], 20/20 |
| walk/nominal_mu0.54 | R3/43 | 0.180 / 0.192 | 0.053 [0.052-0.053], 20/20 | 0.120 [0.120-0.120], 20/20 |
| walk/nominal_mu0.54 | R4/42 | 0.004 / 0.006 | 0.002 [0.002-0.002], 0/20 | 0.001 [0.001-0.001], 0/20 |
| walk/brisk_mu0.54 | R0 | 0.130 / 0.136 | 0.051 [0.051-0.051], 20/20 | 0.121 [0.121-0.121], 20/20 |
| walk/brisk_mu0.54 | R1 | 0.122 / 0.127 | 0.051 [0.050-0.051], 20/20 | 0.121 [0.121-0.121], 20/20 |
| walk/brisk_mu0.54 | R2/42 | 0.125 / 0.135 | 0.051 [0.051-0.051], 20/20 | 0.121 [0.121-0.121], 20/20 |
| walk/brisk_mu0.54 | R2/43 | 0.124 / 0.131 | 0.051 [0.051-0.051], 20/20 | 0.121 [0.121-0.121], 20/20 |
| walk/brisk_mu0.54 | R3/42 | 0.119 / 0.135 | 0.054 [0.054-0.055], 20/20 | 0.124 [0.123-0.124], 20/20 |
| walk/brisk_mu0.54 | R3/43 | 0.150 / 0.183 | 0.054 [0.053-0.054], 20/20 | 0.121 [0.121-0.122], 20/20 |
| walk/brisk_mu0.54 | R4/42 | 0.003 / 0.004 | 0.002 [0.002-0.002], 0/20 | 0.001 [0.001-0.001], 0/20 |
| walk/fr05_mu0.54 | R0 | 0.127 / 0.176 | 0.052 [0.051-0.052], 16/16 | 0.118 [0.117-0.124], 16/16 |
| walk/fr05_mu0.54 | R1 | 0.111 / 0.138 | 0.051 [0.051-0.051], 20/20 | 0.123 [0.123-0.123], 20/20 |
| walk/fr05_mu0.54 | R2/42 | 0.117 / 0.137 | 0.051 [0.051-0.051], 20/20 | 0.123 [0.123-0.123], 20/20 |
| walk/fr05_mu0.54 | R2/43 | 0.134 / 0.187 | 0.051 [0.051-0.052], 20/20 | 0.119 [0.117-0.122], 20/20 |
| walk/fr05_mu0.54 | R3/42 | 0.121 / 0.208 | 0.054 [0.054-0.055], 20/20 | 0.123 [0.123-0.124], 20/20 |
| walk/fr05_mu0.54 | R3/43 | 0.145 / 0.196 | 0.054 [0.053-0.054], 18/18 | 0.117 [0.116-0.118], 18/18 |
| walk/fr05_mu0.54 | R4/42 | 0.002 / 0.004 | 0.002 [0.002-0.002], 0/20 | 0.001 [0.001-0.001], 0/20 |
| carpet/robot_floor_carpet | R2/42 | - | - | - |
| carpet/robot_floor_carpet | R2/43 | - | - | - |
| carpet/robot_floor_carpet | R3/42 | - | - | - |
| carpet/robot_floor_carpet | R3/43 | - | - | - |
| carpet/robot_floor_carpet | R4/42 | 0.003 / 0.005 | 0.002 [0.002-0.002], 0/20 | 0.001 [0.000-0.001], 0/20 |
| carpet/low | R2/42 | - | - | - |
| carpet/low | R2/43 | - | - | - |
| carpet/low | R3/42 | 0.162 / 0.178 | 0.054 [0.054-0.054], 14/14 | 0.121 [0.121-0.122], 14/14 |
| carpet/low | R3/43 | - | - | - |
| carpet/low | R4/42 | 0.003 / 0.004 | 0.002 [0.002-0.002], 0/20 | 0.001 [0.001-0.001], 0/20 |
| carpet/medium | R2/42 | - | - | - |
| carpet/medium | R2/43 | - | - | - |
| carpet/medium | R3/42 | - | - | - |
| carpet/medium | R3/43 | - | - | - |
| carpet/medium | R4/42 | 0.003 / 0.005 | 0.002 [0.002-0.003], 0/20 | 0.001 [0.001-0.002], 0/20 |
| carpet/high | R2/42 | - | - | - |
| carpet/high | R2/43 | - | - | - |
| carpet/high | R3/42 | - | - | - |
| carpet/high | R3/43 | - | - | - |
| carpet/high | R4/42 | 0.003 / 0.005 | 0.002 [0.002-0.003], 0/20 | 0.001 [0.001-0.002], 0/20 |

### Margin ledger

Cells: the threshold (the last value at which all 3 seeds survive and none fails a term it passed at nominal, or survive) and the nearest value that fails with the label there (falls labelled by lifting each candidate limit on the 3 seeds, ledger.LIFT_RULE); '*' the search is not finished or the record is flagged (listed below).

| Margin | Search | R2/43 works | R3/42 works | R2/43 survives | R3/42 survives | Stage 0 |
|---|---|---|---|---|---|---|
| Grip | walk_nominal | 0.2063 (fails 0.2; body-limited: slip_median) | 0.2812 (fails 0.275; controller-limited) | works to 0.1 | works to 0.1 | 0.36 |
| Grip | turn_left | 0.2125 (fails 0.2063; body-limited: slip_median) | 0.325 (fails 0.3187; controller-limited) | works to 0.1 | works to 0.1 | 0.41 |
| Grip | turn_right | 0.2562 (fails 0.25; controller-limited) | 0.2188 (fails 0.2125; body-limited: slip_median) | works to 0.1 | works to 0.1 | 0.41 |
| Grip | walk_brisk | 0.2875 (fails 0.2812; controller-limited) | 0.3438 (fails 0.3375; controller-limited) | 0.1063 (fails 0.1; body-limited: slip (lifted, all 3 seeds survive; untestable: cop_at_pad_edge_with_ankle_saturated)) | 0.1063 (fails 0.1; body-limited: slip (lifted, all 3 seeds survive; untestable: cop_at_pad_edge_with_ankle_saturated)) | 0.56 |
| Grip | turn (worse of turn_left, turn_right) | 0.2562 (fails 0.25; controller-limited) [turn_right] | 0.325 (fails 0.3187; controller-limited) [turn_left] | works to 0.1 [turn_left] | works to 0.1 [turn_left] | 0.41 |
| Push | standing_forward | 0.3129 (fails 0.3221; controller-limited) | 0.3037 (fails 0.3129; controller-limited (lifting drive_torque, envelope saves none of the falls; untestable: cop_at_pad_edge_with_ankle_saturated)) | 0.3313 (fails 0.3405; controller-limited (falls with no relaxable hardware limit active before the onset: nothing to lift, decision 20; untestable: cop_at_pad_edge_with_ankle_saturated)) | 0.3037 (fails 0.3129; controller-limited (lifting drive_torque, envelope saves none of the falls; untestable: cop_at_pad_edge_with_ankle_saturated)) | 0.589 |
| Push | standing_backward | 0.3405 (fails 0.3497; controller-limited (lifting slip saves none of the falls; untestable: cop_at_pad_edge_with_ankle_saturated)) | 0.3497 (fails 0.3589; controller-limited (falls with no relaxable hardware limit active before the onset: nothing to lift, decision 20; untestable: cop_at_pad_edge_with_ankle_saturated)) | 0.3405 (fails 0.3497; controller-limited (lifting slip saves none of the falls; untestable: cop_at_pad_edge_with_ankle_saturated)) | 0.3497 (fails 0.3589; controller-limited (falls with no relaxable hardware limit active before the onset: nothing to lift, decision 20; untestable: cop_at_pad_edge_with_ankle_saturated)) | (forward 0.589) |
| Push | standing_left | 0.6166 (fails 0.6258; controller-limited) | 0.6258 (fails 0.635; controller-limited (lifting slip saves none of the falls; untestable: cop_at_pad_edge_with_ankle_saturated)) | 0.6258 (fails 0.635; controller-limited (lifting slip saves none of the falls; untestable: cop_at_pad_edge_with_ankle_saturated)) | 0.6258 (fails 0.635; controller-limited (lifting slip saves none of the falls; untestable: cop_at_pad_edge_with_ankle_saturated)) | (forward 0.589) |
| Push | standing_right | 0.6258 (fails 0.635; controller-limited) | 0.6166 (fails 0.6258; controller-limited) | 0.6442 (fails 0.6534; controller-limited (lifting drive_torque, envelope, slip saves none of the falls; untestable: cop_at_pad_edge_with_ankle_saturated)) | 0.6442 (fails 0.6534; controller-limited (lifting drive_torque, envelope, slip saves none of the falls; untestable: cop_at_pad_edge_with_ankle_saturated)) | (forward 0.589) |
| Push | walking_forward | 0.1749 (fails 0.1841; controller-limited (lifting drive_torque, slip saves none of the falls; untestable: cop_at_pad_edge_with_ankle_saturated)) | 0.1841 (fails 0.1933; controller-limited (falls with no relaxable hardware limit active before the onset: nothing to lift, decision 20; untestable: cop_at_pad_edge_with_ankle_saturated)) | 0.1749 (fails 0.1841; controller-limited (lifting drive_torque, slip saves none of the falls; untestable: cop_at_pad_edge_with_ankle_saturated)) | 0.1841 (fails 0.1933; controller-limited (falls with no relaxable hardware limit active before the onset: nothing to lift, decision 20; untestable: cop_at_pad_edge_with_ankle_saturated)) | 0.589 |
| Push | walking_backward | 0.1749 (fails 0.1841; controller-limited (lifting slip saves none of the falls; untestable: cop_at_pad_edge_with_ankle_saturated)) | 0.1657 (fails 0.1749; controller-limited) | 0.1749 (fails 0.1841; controller-limited (lifting slip saves none of the falls; untestable: cop_at_pad_edge_with_ankle_saturated)) | 0.1749 (fails 0.1841; controller-limited (lifting slip saves none of the falls; untestable: cop_at_pad_edge_with_ankle_saturated)) | (forward 0.589) |
| Push | walking_left | 0.1012 (fails 0.1104; controller-limited (lifting slip saves none of the falls; untestable: cop_at_pad_edge_with_ankle_saturated)) | 0.08283 (fails 0.09203; body-limited: drive_torque (lifted, all 3 seeds survive; untestable: cop_at_pad_edge_with_ankle_saturated)) | 0.1012 (fails 0.1104; controller-limited (lifting slip saves none of the falls; untestable: cop_at_pad_edge_with_ankle_saturated)) | 0.08283 (fails 0.09203; body-limited: drive_torque (lifted, all 3 seeds survive; untestable: cop_at_pad_edge_with_ankle_saturated)) | (forward 0.589) |
| Push | walking_right | 0.1196 (fails 0.1288; controller-limited (lifting slip saves none of the falls; untestable: cop_at_pad_edge_with_ankle_saturated)) | 0.1196 (fails 0.1288; body-limited: slip (lifted, all 3 seeds survive; untestable: cop_at_pad_edge_with_ankle_saturated)) | 0.1196 (fails 0.1288; controller-limited (lifting slip saves none of the falls; untestable: cop_at_pad_edge_with_ankle_saturated)) | 0.1196 (fails 0.1288; body-limited: slip (lifted, all 3 seeds survive; untestable: cop_at_pad_edge_with_ankle_saturated)) | (forward 0.589) |
| Stop | - | not run (plan deviation, decision 33) | not run (plan deviation, decision 33) | not run (plan deviation, decision 33) | not run (plan deviation, decision 33) | 1.41 |
| Drive speed | walk_fr05 | 0.4961 (fails 0.4902; controller-limited) | 0.8652 (fails 0.8594; body-limited: drive_speed) | 0.291 (fails 0.2852; body-limited: drive_torque, envelope (lifted, all 3 seeds survive; untestable: cop_at_pad_edge_with_ankle_saturated)) | 0.2617 (fails 0.2559; body-limited: drive_torque, envelope, slip (lifted, all 3 seeds survive; untestable: cop_at_pad_edge_with_ankle_saturated)) | 0.93 (0.81 without the 15 % margin) |
| Drive torque | walk_nominal | 0.877 (fails 0.8711; body-limited: drive_sustained) | 0.8594 (fails 0.8535; body-limited: drive_sustained) | 0.291 (fails 0.2852; body-limited: drive_torque (lifted, all 3 seeds survive; untestable: cop_at_pad_edge_with_ankle_saturated)) | 0.3613 (fails 0.3555; body-limited: slip (lifted, all 3 seeds survive; untestable: cop_at_pad_edge_with_ankle_saturated)) | 0.77 (walk RMS 0.66) |
| Model doubt: servo kp | walk_nominal | 0.7129 (fails 0.707; controller-limited) | 0.7188 (fails 0.7129; controller-limited) | 0.5547 (fails 0.5488; controller-limited (lifting drive_torque, slip saves none of the falls; untestable: cop_at_pad_edge_with_ankle_saturated)) | 0.5664 (fails 0.5605; controller-limited (lifting drive_torque, slip saves none of the falls; untestable: cop_at_pad_edge_with_ankle_saturated)) | - |
| Model doubt: latency | walk_nominal | 3 steps (6 ms) (fails 4 steps (8 ms); controller-limited) | 3 steps (6 ms) (fails 4 steps (8 ms); controller-limited) | works to 10 steps (20 ms) | works to 10 steps (20 ms) | - |
| Model doubt: trunk mass | walk_nominal | works to 1.3 (robot +12.4 %) | works to 1.3 (robot +12.4 %) | works to 1.3 (robot +12.4 %) | works to 1.3 (robot +12.4 %) | - |
| Model doubt: trunk mass | walk_nominal_trunk_x1.1 (trunk x1.1: robot +4.1 %) | works 3/3 | works 3/3 | survives 3/3 | survives 3/3 | - |
| Model doubt: trunk mass | walk_nominal_robot_x1.1 (trunk x1.241: robot +10.0 %) | works 3/3 | works 3/3 | survives 3/3 | survives 3/3 | - |
| Model doubt: pile damping | stand_low_down | works to 0.5 | not run | works to 0.5 | not run | - |
| Model doubt: pile damping | stand_low_up | works to 2 | not run | works to 2 | not run | - |
| Model doubt: pile damping | carpet_low_down | fails at nominal (controller-limited (lifting slip saves none of the falls; untestable: cop_at_pad_edge_with_ankle_saturated)) [panels: FAIL 0,0; controller-limited (decision 20; untestable candidates: cop_at_pad_edge_with_ankle_saturated)] | not run | fails at nominal (controller-limited (lifting slip saves none of the falls; untestable: cop_at_pad_edge_with_ankle_saturated)) [panels: FAIL 0,0; controller-limited (decision 20; untestable candidates: cop_at_pad_edge_with_ankle_saturated)] | not run | - |
| Model doubt: pile damping | carpet_low_up | fails at nominal (controller-limited (lifting slip saves none of the falls; untestable: cop_at_pad_edge_with_ankle_saturated)) [panels: FAIL 0,0; controller-limited (decision 20; untestable candidates: cop_at_pad_edge_with_ankle_saturated)] | not run | fails at nominal (controller-limited (lifting slip saves none of the falls; untestable: cop_at_pad_edge_with_ankle_saturated)) [panels: FAIL 0,0; controller-limited (decision 20; untestable candidates: cop_at_pad_edge_with_ankle_saturated)] | not run | - |
| Pack voltage | walk_nominal_25.2V | works 3/3 | works 3/3 | survives 3/3 | survives 3/3 | - |
| Pack voltage | walk_nominal_22.2V | works 3/3 | works 3/3 | survives 3/3 | survives 3/3 | - |
| Pack voltage | walk_nominal_21V | works 3/3 | works 3/3 | survives 3/3 | survives 3/3 | - |
| Pack voltage | walk_fr05_25.2V | works 0/3; outright fails | works 0/3; outright fails | survives 3/3 | survives 3/3 | - |
| Pack voltage | walk_fr05_22.2V | works 0/3; outright fails | works 0/3; outright fails | survives 3/3 | survives 3/3 | - |
| Pack voltage | walk_fr05_21V | fails 0/3 (controller-limited) | works 0/3; outright fails | survives 3/3 | survives 3/3 | - |

- Grip: Stage 0: the lowest friction the round-two turning screen requires in single support, per body: nominal walk / 0.35 rad/s walking turn (the worse direction) / brisk walk (STAGE0_RULE)
- Stop: not run (plan deviation, decision 33): the deceleration a stop holds needs stop references that decelerate at other rates, and the run holds one stop transition (STOP_RULE)
- Drive speed: Stage 0: the knee's speed need at Fr 0.05 over its rated speed, including the screens' 15 % speed margin; without it 0.81 (0.81 / 0.83 / 0.81), which is what the ledger's limit compares (STAGE0_RULE)
- Drive torque: Stage 0: the worst drive's sustained-torque ratio, the larger of the nominal walk's RMS and the one-foot hold (hip_yaw's one-foot hold); the nominal walk's RMS alone 0.66 (neck), which is what the ledger's limit compares (STAGE0_RULE)
- Pack voltage: Stage 0: Stage 0 used 24 V; at 22.2 V the knee's rated speed (8.04 rad/s) equals the Fr 0.05 requirement including the screens' 15 % speed margin (8.08-8.12 rad/s; the peak swing speed itself is about 7.0 rad/s)
- R3/42 ran checkpoint 2506752 (final)

### PPO runs

| Run | Status | Steps / budget | Last checkpoint | Judged checkpoint | G6a zero residual | G6a env steps/s | G6b | Fix; G6b after it | G6c | Job h | CPU h | run.json written |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R3/42 | running | 2506752 / 4005888 | 2506752 | 2506752 (final) | yes | 74 (FAIL: cut the steps equally for every body, decision 26) | FAIL at 1007616 (survival policy 17.99 s, zero 18.00 s): make one fix, then stop | from 1007616 (task_weights 1,1,1,1,1,2,2,1,1); pass at 2015232 (survival policy 18.07 s, zero 18.00 s) | pass at 2506752 (beats R2 on walk, walking_turn_left, walking_turn_right, in_place_turn_left, brisk) | 19.52 (running) | running | 2026-11-10T00:12:29+00:00 |
| R3/43 | running | 2506752 / 4005888 | 2506752 | 2506752 (final) | yes | 75 (FAIL: cut the steps equally for every body, decision 26) | FAIL at 1007616 (survival policy 16.94 s, zero 18.00 s): make one fix, then stop | from 1007616 (task_weights 1,1,1,1,1,2,2,1,1); FAIL at 2015232 (survival policy 18.00 s, zero 18.00 s): stop the run (FIX_RULE) | pass at 2506752 (beats R2 on walk, walking_turn_left, walking_turn_right, in_place_turn_right, brisk, carpet_walk on low) | 19.06 (running) | running | 2026-11-10T00:12:29+00:00 |
| R4/42 | running | 1253376 / 4005888 | 1253376 | 1253376 (final) | - | 100 (FAIL: cut the steps equally for every body, decision 26) | - | - | - | 10.33 (running) | running | 2026-11-10T00:12:30+00:00 |

Elapsed PPO wall clock (the union of every run's sessions): 10.33 h; job-hours (their sum: the runs share the cores) 48.91 h.

### Fit and G5

G5 pass (seeds passing: [42, 43]); body-level selection (env.GAINS_RULE): CMA-ES seed 43, seed43_gen40 (passed).

| CMA-ES seed | Fit | Evaluations | Sigma | Selected | Walk | Walking turn | In-place turn | G5 |
|---|---|---|---|---|---|---|---|---|
| 42 | done | 720 | 0.0145 | seed42_gen60 | 20/20 (surv 20) | 20/20 (surv 20) | 20/20 (surv 20) | passes G5 |
| 43 | done | 720 | 0.0361 | seed43_gen40 | 20/20 (surv 20) | 20/20 (surv 20) | 20/20 (surv 20) | passes G5 |

### Core-hours per step

| Step | Wall h | Core-h | Basis |
|---|---|---|---|
| prepare | 0.12 | 0.12 | wall, one process |
| gates | 0.09 | 0.09 | wall, one process |
| fit | 3.06 | 6.12 | sessions' end - start x workers |
| g5 | 0.39 | 0.39 | tune.json seconds, one process assumed (workers not recorded): a lower bound |
| ppo | 10.33 (job-hours 48.91) | 19.89 | finished sessions' CPU, and for a running session its wall to the record's last write x the CPU per wall second of its first rollouts (an estimate); wall: the elapsed union of the sessions, job-hours beside |
| panels | 6.11 | 6.13 | episodes' and attributions' wall seconds (one process each): an upper bound on CPU |
| ledger | 0.97 | 0.84 | episodes' CPU seconds, each evaluation once (by its key) however many margins use it |
| total | | 33.57 | core-hours recorded so far |

Core-hours recorded over the bodies reported: 92.17.
