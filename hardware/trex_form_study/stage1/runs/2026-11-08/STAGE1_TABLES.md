# Stage 1 tables

Machine-made by `python -m designlab.stage1 report` at 2026-11-09T22:26:13+00:00 from the records in `stage1/runs/2026-11-08` (designlab/stage1/report.py; the rules are in stage1_tables.json, which is also each body's summary, plan section 7's <body>/summary.json, decision 34).  Missing records show as 'not run'.  The narrative report is STAGE1_PROBE.md.

## 60-N-T-10

### Verdicts by controller

| Controller | shown | provisional | body-limited | controller-limited | not attributed | fails by falls (R4: not attributed) | incomplete | not run |
|---|---|---|---|---|---|---|---|---|
| R2 (shown) | 14 | 0 | 0 | 6 | 0 | 0 | 0 | 0 |
| R3 (not judged: judged with R2) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| R4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 22 |

Counts over the conditions gated for each controller's rows.
R4 has one seed (decisions 3 and 28): its verdicts are at most provisional (policy_rows.R4_VERDICT_RULE); its record on each gated condition: not run 22.
60-N-T-10's R3 is judged with R2 (R3/42 stopped at 2015232 steps: G6b failed again after the fix (FIX_RULE) (decision 32); R3/43 stopped at 2015232 steps: the body's R3 is judged with R2 (train.R2_JUDGED_RULE, decision 38): its seed 42 run was stopped by FIX_RULE at 2015232 steps (G6b failed again after the fix), so this run cannot change a verdict; stopped to give the 75 cm runs the cores (decision 32)): the grid shows R2's verdicts (train.R2_JUDGED_RULE).

G7 pass: sway 0.1677 H (Stage 0 0.1677, plan 0.168), cadence 1.9612 steps/s (Stage 0 1.9612, plan 1.96).

### Verdict grid

Verdicts shown: R2: 60-N-T-10's R3 is judged with R2 (R3/42 stopped at 2015232 steps: G6b failed again after the fix (FIX_RULE) (decision 32); R3/43 stopped at 2015232 steps: the body's R3 is judged with R2 (train.R2_JUDGED_RULE, decision 38): its seed 42 run was stopped by FIX_RULE at 2015232 steps (G6b failed again after the fix), so this run cannot change a verdict; stopped to give the 75 cm runs the cores (decision 32); train.R2_JUDGED_RULE).  Cells: result, survived,succeeded of 20 (their one-sided 95 % Clopper-Pearson lower bounds); '(rep.)' reported, not gated, for that row; '*' interim (listed below each table); 'not run' no record; 'no stance' the floor has no settled stance for the body (panels.NOT_RUN_RULE); 'n/a' the row does not run there; R4's verdict is at most provisional (one seed), and its falls are never attributed (by design: decision 20's attribution is for R2 and R3).

#### Stand

| Condition | Gated for | R0 | R1 | R2/42 | R2/43 | R3/42 | R3/43 | R4/42 | R2 verdict (shown) | R3 verdict (not judged: stopped) | R4 verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| hard | all | pass 20,20 (0.86,0.86) | pass 20,20 (0.86,0.86) | pass 20,20 (0.86,0.86) | pass 20,20 (0.86,0.86) | not run | not run | not run | shown | not run | not run |
| robot_floor_carpet | all | n/a | n/a | pass 20,20 (0.86,0.86) | pass 20,20 (0.86,0.86) | not run | not run | not run | shown | not run | not run |
| low | all | n/a | n/a | pass 20,20 (0.86,0.86) | pass 20,20 (0.86,0.86) | not run | not run | not run | shown | not run | not run |

#### Walk

| Condition | Gated for | R0 | R1 | R2/42 | R2/43 | R3/42 | R3/43 | R4/42 | R2 verdict (shown) | R3 verdict (not judged: stopped) | R4 verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| nominal | all | FAIL 3,3 (0.04,0.04) | FAIL 18,18 (0.72,0.72) | pass 20,18 (0.86,0.72) | pass 20,20 (0.86,0.86) | not run | not run | not run | shown | not run | not run |
| brisk | all | FAIL 0,0 (0.00,0.00) | FAIL 15,15 (0.54,0.54) | FAIL 17,17 (0.66,0.66) | FAIL 0,0 (0.00,0.00) | not run | not run | not run | controller-limited | not run | not run |
| fr05 | all | FAIL 0,0 (0.00,0.00) | FAIL 1,1 (0.00,0.00) | FAIL 1,0 (0.00,0.00) | FAIL 0,0 (0.00,0.00) | not run | not run | not run | controller-limited | not run | not run |
| nominal_mu0.54 | all | FAIL 5,5 (0.10,0.10) | FAIL 14,14 (0.49,0.49) | pass 20,19 (0.86,0.78) | pass 20,19 (0.86,0.78) | not run | not run | not run | shown | not run | not run |
| brisk_mu0.54 | none (reported) | FAIL 0,0 (0.00,0.00) (rep.) | FAIL 14,12 (0.49,0.39) (rep.) | FAIL 15,11 (0.54,0.35) (rep.) | FAIL 0,0 (0.00,0.00) (rep.) | not run | not run | not run | controller-limited (rep.) | not run (rep.) | not run (rep.) |
| fr05_mu0.54 | none (reported) | FAIL 0,0 (0.00,0.00) (rep.) | FAIL 0,0 (0.00,0.00) (rep.) | FAIL 0,0 (0.00,0.00) (rep.) | FAIL 0,0 (0.00,0.00) (rep.) | not run | not run | not run | controller-limited (rep.) | not run (rep.) | not run (rep.) |

- R0 walk/nominal: not attributed (falls; attribution is applied to R2, R3 rows) (by design)
- R1 walk/nominal: not attributed (falls; attribution is applied to R2, R3 rows) (by design)
- R0 walk/brisk: not attributed (falls; attribution is applied to R2, R3 rows) (by design)
- R1 walk/brisk: not attributed (falls; attribution is applied to R2, R3 rows) (by design)
- R2/42 walk/brisk: controller-limited (decision 20; 3 falls: the sign test needs 5 for p < 0.05, so no limit was lifted; untestable candidates: cop_at_pad_edge_with_ankle_saturated)
- R2/43 walk/brisk: controller-limited (decision 20; untestable candidates: cop_at_pad_edge_with_ankle_saturated)
- R0 walk/fr05: not attributed (falls; attribution is applied to R2, R3 rows) (by design)
- R1 walk/fr05: not attributed (falls; attribution is applied to R2, R3 rows) (by design)
- R2/42 walk/fr05: controller-limited (decision 20; untestable candidates: cop_at_pad_edge_with_ankle_saturated); also controller-limited
- R2/43 walk/fr05: controller-limited (decision 20; untestable candidates: cop_at_pad_edge_with_ankle_saturated)
- R0 walk/nominal_mu0.54: not attributed (falls; attribution is applied to R2, R3 rows) (by design)
- R1 walk/nominal_mu0.54: not attributed (falls; attribution is applied to R2, R3 rows) (by design)
- R0 walk/brisk_mu0.54: not attributed (falls; attribution is applied to R2, R3 rows) (by design)
- R1 walk/brisk_mu0.54: not attributed (falls; attribution is applied to R2, R3 rows) (by design); also controller-limited
- R2/42 walk/brisk_mu0.54: body-limited: slip; also controller-limited
- R2/43 walk/brisk_mu0.54: controller-limited (decision 20; untestable candidates: cop_at_pad_edge_with_ankle_saturated)
- R0 walk/fr05_mu0.54: not attributed (falls; attribution is applied to R2, R3 rows) (by design)
- R1 walk/fr05_mu0.54: not attributed (falls; attribution is applied to R2, R3 rows) (by design)
- R2/42 walk/fr05_mu0.54: body-limited: slip
- R2/43 walk/fr05_mu0.54: controller-limited (decision 20; untestable candidates: cop_at_pad_edge_with_ankle_saturated)

#### Turn

| Condition | Gated for | R0 | R1 | R2/42 | R2/43 | R3/42 | R3/43 | R4/42 | R2 verdict (shown) | R3 verdict (not judged: stopped) | R4 verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| walking_0.35_left | all | FAIL 2,2 (0.02,0.02) | FAIL 19,19 (0.78,0.78) | pass 20,19 (0.86,0.78) | pass 20,20 (0.86,0.86) | not run | not run | not run | shown | not run | not run |
| walking_0.35_right | all | FAIL 2,2 (0.02,0.02) | FAIL 18,17 (0.72,0.66) | pass 20,19 (0.86,0.78) | pass 20,20 (0.86,0.86) | not run | not run | not run | shown | not run | not run |
| in_place_0.50_left | all | not run | pass 20,20 (0.86,0.86) | pass 20,20 (0.86,0.86) | pass 20,20 (0.86,0.86) | not run | not run | not run | shown | not run | not run |
| in_place_0.50_right | all | not run | pass 20,20 (0.86,0.86) | pass 20,20 (0.86,0.86) | pass 20,20 (0.86,0.86) | not run | not run | not run | shown | not run | not run |
| walking_0.50_left | none (reported) | not run | FAIL 19,15 (0.78,0.54) (rep.) | FAIL 20,16 (0.86,0.60) (rep.) | FAIL 20,14 (0.86,0.49) (rep.) | not run | not run | not run | controller-limited (rep.) | not run (rep.) | not run (rep.) |
| walking_0.50_right | none (reported) | not run | FAIL 19,12 (0.78,0.39) (rep.) | pass 20,17 (0.86,0.66) (rep.) | pass 20,18 (0.86,0.72) (rep.) | not run | not run | not run | shown (rep.) | not run (rep.) | not run (rep.) |

- R0 turn/walking_0.35_left: not attributed (falls; attribution is applied to R2, R3 rows) (by design)
- R1 turn/walking_0.35_left: not attributed (falls; attribution is applied to R2, R3 rows) (by design)
- R0 turn/walking_0.35_right: not attributed (falls; attribution is applied to R2, R3 rows) (by design)
- R1 turn/walking_0.35_right: not attributed (falls; attribution is applied to R2, R3 rows) (by design); also controller-limited
- R1 turn/walking_0.50_left: body-limited: joint_range; also not attributed (falls; attribution is applied to R2, R3 rows)
- R2/42 turn/walking_0.50_left: controller-limited
- R2/43 turn/walking_0.50_left: body-limited: joint_range
- R1 turn/walking_0.50_right: body-limited: joint_range; also not attributed (falls; attribution is applied to R2, R3 rows)

#### Start and stop

| Condition | Gated for | R0 | R1 | R2/42 | R2/43 | R3/42 | R3/43 | R4/42 | R2 verdict (shown) | R3 verdict (not judged: stopped) | R4 verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| twice | all | not run | pass 20,20 (0.86,0.86) | pass 20,20 (0.86,0.86) | pass 20,20 (0.86,0.86) | not run | not run | not run | shown | not run | not run |

#### Push

| Condition | Gated for | R0 | R1 | R2/42 | R2/43 | R3/42 | R3/43 | R4/42 | R2 verdict (shown) | R3 verdict (not judged: stopped) | R4 verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| standing_forward | all | not run | pass 20,20 (0.86,0.86) | pass 20,20 (0.86,0.86) | pass 20,20 (0.86,0.86) | not run | not run | not run | shown | not run | not run |
| standing_backward | all | not run | pass 20,20 (0.86,0.86) | pass 20,20 (0.86,0.86) | pass 20,20 (0.86,0.86) | not run | not run | not run | shown | not run | not run |
| standing_left | all | not run | pass 20,20 (0.86,0.86) | pass 20,20 (0.86,0.86) | pass 20,20 (0.86,0.86) | not run | not run | not run | shown | not run | not run |
| standing_right | all | not run | pass 20,20 (0.86,0.86) | pass 20,20 (0.86,0.86) | pass 20,20 (0.86,0.86) | not run | not run | not run | shown | not run | not run |
| walking_forward | all | not run | FAIL 2,1 (0.02,0.00) | FAIL 1,1 (0.00,0.00) | FAIL 1,1 (0.00,0.00) | not run | not run | not run | controller-limited | not run | not run |
| walking_backward | all | not run | FAIL 0,0 (0.00,0.00) | FAIL 0,0 (0.00,0.00) | FAIL 0,0 (0.00,0.00) | not run | not run | not run | controller-limited | not run | not run |
| walking_left | all | not run | FAIL 1,1 (0.00,0.00) | FAIL 2,0 (0.02,0.00) | FAIL 0,0 (0.00,0.00) | not run | not run | not run | controller-limited | not run | not run |
| walking_right | all | not run | FAIL 1,0 (0.00,0.00) | FAIL 2,0 (0.02,0.00) | FAIL 0,0 (0.00,0.00) | not run | not run | not run | controller-limited | not run | not run |

- R1 push/walking_forward: not attributed (falls; attribution is applied to R2, R3 rows) (by design); also controller-limited
- R2/42 push/walking_forward: controller-limited (decision 20; untestable candidates: cop_at_pad_edge_with_ankle_saturated)
- R2/43 push/walking_forward: controller-limited (decision 20; untestable candidates: cop_at_pad_edge_with_ankle_saturated)
- R1 push/walking_backward: not attributed (falls; attribution is applied to R2, R3 rows) (by design)
- R2/42 push/walking_backward: controller-limited (decision 20; untestable candidates: cop_at_pad_edge_with_ankle_saturated)
- R2/43 push/walking_backward: controller-limited (decision 20; untestable candidates: cop_at_pad_edge_with_ankle_saturated)
- R1 push/walking_left: not attributed (falls; attribution is applied to R2, R3 rows) (by design)
- R2/42 push/walking_left: controller-limited (decision 20; untestable candidates: cop_at_pad_edge_with_ankle_saturated); also controller-limited
- R2/43 push/walking_left: controller-limited (decision 20; untestable candidates: cop_at_pad_edge_with_ankle_saturated)
- R1 push/walking_right: not attributed (falls; attribution is applied to R2, R3 rows) (by design); also controller-limited
- R2/42 push/walking_right: body-limited: slip; also controller-limited
- R2/43 push/walking_right: controller-limited (decision 20; untestable candidates: cop_at_pad_edge_with_ankle_saturated)

#### Carpet

| Condition | Gated for | R0 | R1 | R2/42 | R2/43 | R3/42 | R3/43 | R4/42 | R2 verdict (shown) | R3 verdict (not judged: stopped) | R4 verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| robot_floor_carpet | R3/42, R3/43, R4/42 | n/a | n/a | FAIL 0,0 (0.00,0.00) (rep.) | FAIL 0,0 (0.00,0.00) (rep.) | not run | not run | not run | controller-limited (rep.) | not run | not run |
| low | R3/42, R3/43, R4/42 | n/a | n/a | FAIL 0,0 (0.00,0.00) (rep.) | FAIL 0,0 (0.00,0.00) (rep.) | not run | not run | not run | controller-limited (rep.) | not run | not run |
| medium | none (reported) | n/a | n/a | no stance | no stance | not run | not run | not run | not run (rep.) | not run (rep.) | not run (rep.) |
| high | none (reported) | n/a | n/a | no stance | no stance | not run | not run | not run | not run (rep.) | not run (rep.) | not run (rep.) |

The tracker (R0-R2) was never fitted on carpet (decision 23: the fit and G5 ran on hard floor only, and R2 fell in every start tried on robot_floor_carpet and low pile): the carpet walk is reported for R0-R2, with its fall attribution, and gated for R3 and R4.

- R2/42 carpet/robot_floor_carpet: controller-limited (decision 20; untestable candidates: cop_at_pad_edge_with_ankle_saturated)
- R2/43 carpet/robot_floor_carpet: controller-limited (decision 20; untestable candidates: cop_at_pad_edge_with_ankle_saturated)
- R2/42 carpet/low: controller-limited (decision 20; untestable candidates: cop_at_pad_edge_with_ankle_saturated)
- R2/43 carpet/low: controller-limited (decision 20; untestable candidates: cop_at_pad_edge_with_ankle_saturated)

Provenance of each row (PROVENANCE_RULE):

- R0 = R0: feedforward + PD on the commanded clock (ROWS_RULE)
- R1 = R1: the tracker on the true state, the body-level G5 selection (seed42_gen20)
- R2/42 = R2: the tracker on the estimate, CMA-ES seed 42's G5 selection seed42_gen20
- R2/43 = R2: the tracker on the estimate, CMA-ES seed 43's G5 selection seed43_gen40

### Looks (sway, head, tail)

Reported, not gated (LOOKS_RULE): sway against 0.12 H, head and tail amplitude against BEHAVIOR_SPEC's bands (head 0.05-0.15 rad, tail 0.1-0.2 rad).

| Condition | Row | Sway mean / max (H) | Head amplitude mean [range] (rad), in band | Tail amplitude mean [range] (rad), in band |
|---|---|---|---|---|
| walk/nominal | R0 | 0.183 / 0.200 | 0.051 [0.051-0.051], 3/3 | 0.120 [0.120-0.120], 3/3 |
| walk/nominal | R1 | 0.174 / 0.184 | 0.051 [0.051-0.051], 18/18 | 0.120 [0.120-0.120], 18/18 |
| walk/nominal | R2/42 | 0.176 / 0.187 | 0.051 [0.051-0.051], 20/20 | 0.120 [0.120-0.120], 20/20 |
| walk/nominal | R2/43 | 0.178 / 0.190 | 0.051 [0.051-0.052], 20/20 | 0.120 [0.120-0.121], 20/20 |
| walk/brisk | R0 | - | - | - |
| walk/brisk | R1 | 0.129 / 0.158 | 0.052 [0.051-0.052], 15/15 | 0.121 [0.120-0.121], 15/15 |
| walk/brisk | R2/42 | 0.132 / 0.151 | 0.051 [0.051-0.052], 17/17 | 0.121 [0.121-0.121], 17/17 |
| walk/brisk | R2/43 | - | - | - |
| walk/fr05 | R0 | - | - | - |
| walk/fr05 | R1 | 0.107 / 0.145 | 0.052 [0.052-0.052], 1/1 | 0.122 [0.122-0.122], 1/1 |
| walk/fr05 | R2/42 | 0.127 / 0.141 | 0.052 [0.052-0.052], 1/1 | 0.122 [0.122-0.122], 1/1 |
| walk/fr05 | R2/43 | - | - | - |
| walk/nominal_mu0.54 | R0 | 0.183 / 0.197 | 0.051 [0.051-0.051], 5/5 | 0.120 [0.120-0.121], 5/5 |
| walk/nominal_mu0.54 | R1 | 0.175 / 0.186 | 0.051 [0.051-0.051], 14/14 | 0.120 [0.120-0.120], 14/14 |
| walk/nominal_mu0.54 | R2/42 | 0.176 / 0.188 | 0.051 [0.051-0.051], 20/20 | 0.120 [0.120-0.121], 20/20 |
| walk/nominal_mu0.54 | R2/43 | 0.180 / 0.193 | 0.051 [0.051-0.051], 20/20 | 0.120 [0.120-0.121], 20/20 |
| walk/brisk_mu0.54 | R0 | - | - | - |
| walk/brisk_mu0.54 | R1 | 0.132 / 0.163 | 0.052 [0.051-0.053], 14/14 | 0.121 [0.121-0.122], 14/14 |
| walk/brisk_mu0.54 | R2/42 | 0.138 / 0.187 | 0.052 [0.051-0.052], 15/15 | 0.122 [0.121-0.123], 15/15 |
| walk/brisk_mu0.54 | R2/43 | - | - | - |
| walk/fr05_mu0.54 | R0 | - | - | - |
| walk/fr05_mu0.54 | R1 | - | - | - |
| walk/fr05_mu0.54 | R2/42 | - | - | - |
| walk/fr05_mu0.54 | R2/43 | - | - | - |
| carpet/robot_floor_carpet | R2/42 | - | - | - |
| carpet/robot_floor_carpet | R2/43 | - | - | - |
| carpet/low | R2/42 | - | - | - |
| carpet/low | R2/43 | - | - | - |

### Margin ledger

Cells: the threshold (the last value at which all 3 seeds survive and none fails a term it passed at nominal, or survive) and the nearest value that fails with the label there (falls labelled by lifting each candidate limit on the 3 seeds, ledger.LIFT_RULE); '*' the search is not finished or the record is flagged (listed below).

| Margin | Search | R2/42 works | R3/42 works | R2/42 survives | R3/42 survives | Stage 0 |
|---|---|---|---|---|---|---|
| Grip | walk_nominal | incomplete * | not run | works to 0.1 * | not run | 0.48 |
| Grip | turn_left | incomplete * | not run | incomplete * | not run | 0.54 |
| Grip | turn_right | incomplete * | not run | incomplete * | not run | 0.54 |
| Grip | walk_brisk | fails at nominal (body-limited: drive_torque, slip (lifted, all 3 seeds survive; untestable: cop_at_pad_edge_with_ankle_saturated)) [panels: FAIL 17,17; controller-limited (decision 20; 3 falls: the sign test needs 5 for p < 0.05, so no limit was lifted; untestable candidates: cop_at_pad_edge_with_ankle_saturated)] * | not run | fails at nominal (body-limited: drive_torque, slip (lifted, all 3 seeds survive; untestable: cop_at_pad_edge_with_ankle_saturated)) [panels: FAIL 17,17; controller-limited (decision 20; 3 falls: the sign test needs 5 for p < 0.05, so no limit was lifted; untestable candidates: cop_at_pad_edge_with_ankle_saturated)] * | not run | 0.75 |
| Grip | turn (worse of turn_left, turn_right) | incomplete * | not run | incomplete * | not run | 0.54 |
| Push | standing_forward | not run | not run | not run | not run | 0.487 |
| Push | standing_backward | not run | not run | not run | not run | (forward 0.487) |
| Push | standing_left | not run | not run | not run | not run | (forward 0.487) |
| Push | standing_right | not run | not run | not run | not run | (forward 0.487) |
| Push | walking_forward | not run | not run | not run | not run | 0.487 |
| Push | walking_backward | not run | not run | not run | not run | (forward 0.487) |
| Push | walking_left | not run | not run | not run | not run | (forward 0.487) |
| Push | walking_right | not run | not run | not run | not run | (forward 0.487) |
| Stop | - | not run (plan deviation, decision 33) | not run (plan deviation, decision 33) | not run (plan deviation, decision 33) | not run (plan deviation, decision 33) | 1.25 |
| Drive speed | walk_fr05 | not run | not run | not run | not run | 0.93 (0.81 without the 15 % margin) |
| Drive torque | walk_nominal | not run | not run | not run | not run | 0.76 (walk RMS 0.76) |
| Model doubt: servo kp | walk_nominal | not run | not run | not run | not run | - |
| Model doubt: latency | walk_nominal | not run | not run | not run | not run | - |
| Model doubt: trunk mass | walk_nominal | not run | not run | not run | not run | - |
| Model doubt: trunk mass | walk_nominal_trunk_x1.1 | not run | not run | not run | not run | - |
| Model doubt: trunk mass | walk_nominal_robot_x1.1 | not run | not run | not run | not run | - |
| Model doubt: pile damping | stand_low_down | not run | not run | not run | not run | - |
| Model doubt: pile damping | stand_low_up | not run | not run | not run | not run | - |
| Model doubt: pile damping | carpet_low_down | not run | not run | not run | not run | - |
| Model doubt: pile damping | carpet_low_up | not run | not run | not run | not run | - |
| Pack voltage | walk_nominal_25.2V | not run | not run | not run | not run | - |
| Pack voltage | walk_nominal_22.2V | not run | not run | not run | not run | - |
| Pack voltage | walk_nominal_21V | not run | not run | not run | not run | - |
| Pack voltage | walk_fr05_25.2V | not run | not run | not run | not run | - |
| Pack voltage | walk_fr05_22.2V | not run | not run | not run | not run | - |
| Pack voltage | walk_fr05_21V | not run | not run | not run | not run | - |

- R3 is judged with R2 for this body (R3/42 stopped at 2015232 steps: G6b failed again after the fix (FIX_RULE) (decision 32); R3/43 stopped at 2015232 steps: the body's R3 is judged with R2 (train.R2_JUDGED_RULE, decision 38): its seed 42 run was stopped by FIX_RULE at 2015232 steps (G6b failed again after the fix), so this run cannot change a verdict; stopped to give the 75 cm runs the cores (decision 32)): its ledger row is not run (ledger.ROWS_RULE)
- Grip: Stage 0: the lowest friction the round-two turning screen requires in single support, per body: nominal walk / 0.35 rad/s walking turn (the worse direction) / brisk walk (STAGE0_RULE)
- Stop: not run (plan deviation, decision 33): the deceleration a stop holds needs stop references that decelerate at other rates, and the run holds one stop transition (STOP_RULE)
- Drive speed: Stage 0: the knee's speed need at Fr 0.05 over its rated speed, including the screens' 15 % speed margin; without it 0.81 (0.81 / 0.83 / 0.81), which is what the ledger's limit compares (STAGE0_RULE)
- Drive torque: Stage 0: the worst drive's sustained-torque ratio, the larger of the nominal walk's RMS and the one-foot hold (ankle_a's one-foot hold); the nominal walk's RMS alone 0.76 (neck), which is what the ledger's limit compares (STAGE0_RULE)
- Pack voltage: Stage 0: Stage 0 used 24 V; at 22.2 V the knee's rated speed (8.04 rad/s) equals the Fr 0.05 requirement including the screens' 15 % speed margin (8.08-8.12 rad/s; the peak swing speed itself is about 7.0 rad/s)

### PPO runs

| Run | Status | Steps / budget | Last checkpoint | Judged checkpoint | G6a zero residual | G6a env steps/s | G6b | Fix; G6b after it | G6c | Job h | CPU h | run.json written |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R3/42 | stopped_g6b | 2015232 / 4005888 | 2015232 | 1507328 (provisional) | yes | 83 (FAIL: cut the steps equally for every body, decision 26) | FAIL at 1007616 (survival policy 14.14 s, zero 15.98 s): make one fix, then stop | from 1007616 (task_weights 1,1,1,1,1,2,2,1,1); FAIL at 2015232 (survival policy 15.16 s, zero 15.98 s): stop the run (FIX_RULE) | - | 14.07 (running) | running | 2026-11-09T22:23:02+00:00 |
| R3/43 | stopped_r2_judged | 2015232 / 4005888 | 2015232 | 1507328 (provisional, not yet recorded) | yes | 80 (FAIL: cut the steps equally for every body, decision 26) | FAIL at 1007616 (survival policy 15.41 s, zero 15.98 s): make one fix, then stop | from 1007616 (task_weights 1,1,1,1,1,2,2,1,1); pass at 2015232 (survival policy 16.00 s, zero 15.98 s) | - | 14.17 (running) | running | 2026-11-09T22:24:58+00:00 |
| R4/42 | running | 1507328 / 4005888 | 1507328 | 1253376 (final) | - | 120 (FAIL: cut the steps equally for every body, decision 26) | - | - | - | 8.55 (running) | running | 2026-11-09T22:25:38+00:00 |

Elapsed PPO wall clock (the union of every run's sessions): 8.55 h; job-hours (their sum: the runs share the cores) 36.78 h.

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
| ppo | 8.55 (job-hours 36.78) | 14.83 | finished sessions' CPU, and for a running session its wall to the record's last write x the CPU per wall second of its first rollouts (an estimate); wall: the elapsed union of the sessions, job-hours beside |
| panels | 3.32 | 3.45 | episodes' and attributions' wall seconds (one process each): an upper bound on CPU |
| ledger | - | 0.04 | episodes' CPU seconds, each evaluation once (by its key) however many margins use it |
| total | | 24.13 | core-hours recorded so far |

## 60-N-H-10

### Verdicts by controller

| Controller | shown | provisional | body-limited | controller-limited | not attributed | fails by falls (R4: not attributed) | incomplete | not run |
|---|---|---|---|---|---|---|---|---|
| R2 (shown) | 14 | 1 | 0 | 5 | 0 | 0 | 0 | 0 |
| R3 (not judged: judged with R2) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| R4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 22 |

Counts over the conditions gated for each controller's rows.
R4 has one seed (decisions 3 and 28): its verdicts are at most provisional (policy_rows.R4_VERDICT_RULE); its record on each gated condition: not run 22.
60-N-H-10's R3 is judged with R2 (R3/42 stopped at 2015232 steps: G6b failed again after the fix (FIX_RULE) (decision 32); R3/43 stopped at 1753088 steps: the body's R3 is judged with R2 (train.R2_JUDGED_RULE, decision 38): its seed 42 run was stopped by FIX_RULE at 2015232 steps (G6b failed again after the fix), so this run cannot change a verdict; stopped to give the 75 cm runs the cores (decision 32)): the grid shows R2's verdicts (train.R2_JUDGED_RULE).

G7 pass: sway 0.1693 H (Stage 0 0.1693, plan 0.169), cadence 1.9692 steps/s (Stage 0 1.9691, plan 1.97).

### Verdict grid

Verdicts shown: R2: 60-N-H-10's R3 is judged with R2 (R3/42 stopped at 2015232 steps: G6b failed again after the fix (FIX_RULE) (decision 32); R3/43 stopped at 1753088 steps: the body's R3 is judged with R2 (train.R2_JUDGED_RULE, decision 38): its seed 42 run was stopped by FIX_RULE at 2015232 steps (G6b failed again after the fix), so this run cannot change a verdict; stopped to give the 75 cm runs the cores (decision 32); train.R2_JUDGED_RULE).  Cells: result, survived,succeeded of 20 (their one-sided 95 % Clopper-Pearson lower bounds); '(rep.)' reported, not gated, for that row; '*' interim (listed below each table); 'not run' no record; 'no stance' the floor has no settled stance for the body (panels.NOT_RUN_RULE); 'n/a' the row does not run there; R4's verdict is at most provisional (one seed), and its falls are never attributed (by design: decision 20's attribution is for R2 and R3).

#### Stand

| Condition | Gated for | R0 | R1 | R2/42 | R2/43 | R3/42 | R3/43 | R4/42 | R2 verdict (shown) | R3 verdict (not judged: stopped) | R4 verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| hard | all | pass 20,20 (0.86,0.86) | pass 20,20 (0.86,0.86) | pass 20,20 (0.86,0.86) | pass 20,20 (0.86,0.86) | not run | not run | not run | shown | not run | not run |
| robot_floor_carpet | all | n/a | n/a | pass 20,20 (0.86,0.86) | pass 20,20 (0.86,0.86) | not run | not run | not run | shown | not run | not run |
| low | all | n/a | n/a | pass 20,20 (0.86,0.86) | pass 20,20 (0.86,0.86) | not run | not run | not run | shown | not run | not run |

#### Walk

| Condition | Gated for | R0 | R1 | R2/42 | R2/43 | R3/42 | R3/43 | R4/42 | R2 verdict (shown) | R3 verdict (not judged: stopped) | R4 verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| nominal | all | FAIL 1,1 (0.00,0.00) | FAIL 18,18 (0.72,0.72) | pass 20,18 (0.86,0.72) | pass 20,18 (0.86,0.72) | not run | not run | not run | shown | not run | not run |
| brisk | all | FAIL 0,0 (0.00,0.00) | FAIL 15,14 (0.54,0.49) | FAIL 5,5 (0.10,0.10) | pass 20,19 (0.86,0.78) | not run | not run | not run | provisional | not run | not run |
| fr05 | all | FAIL 0,0 (0.00,0.00) | FAIL 0,0 (0.00,0.00) | FAIL 0,0 (0.00,0.00) | FAIL 0,0 (0.00,0.00) | not run | not run | not run | controller-limited | not run | not run |
| nominal_mu0.54 | all | FAIL 7,7 (0.18,0.18) | FAIL 12,11 (0.39,0.35) | pass 20,18 (0.86,0.72) | pass 20,19 (0.86,0.78) | not run | not run | not run | shown | not run | not run |
| brisk_mu0.54 | none (reported) | FAIL 0,0 (0.00,0.00) (rep.) | FAIL 13,11 (0.44,0.35) (rep.) | FAIL 6,3 (0.14,0.04) (rep.) | FAIL 17,14 (0.66,0.49) (rep.) | not run | not run | not run | controller-limited (rep.) | not run (rep.) | not run (rep.) |
| fr05_mu0.54 | none (reported) | FAIL 0,0 (0.00,0.00) (rep.) | FAIL 0,0 (0.00,0.00) (rep.) | FAIL 0,0 (0.00,0.00) (rep.) | FAIL 2,0 (0.02,0.00) (rep.) | not run | not run | not run | controller-limited (rep.) | not run (rep.) | not run (rep.) |

- R0 walk/nominal: not attributed (falls; attribution is applied to R2, R3 rows) (by design)
- R1 walk/nominal: not attributed (falls; attribution is applied to R2, R3 rows) (by design)
- R0 walk/brisk: not attributed (falls; attribution is applied to R2, R3 rows) (by design)
- R1 walk/brisk: not attributed (falls; attribution is applied to R2, R3 rows) (by design); also controller-limited
- R2/42 walk/brisk: body-limited: slip
- R0 walk/fr05: not attributed (falls; attribution is applied to R2, R3 rows) (by design)
- R1 walk/fr05: not attributed (falls; attribution is applied to R2, R3 rows) (by design)
- R2/42 walk/fr05: controller-limited (decision 20; untestable candidates: cop_at_pad_edge_with_ankle_saturated)
- R2/43 walk/fr05: body-limited: slip
- R0 walk/nominal_mu0.54: not attributed (falls; attribution is applied to R2, R3 rows) (by design)
- R1 walk/nominal_mu0.54: not attributed (falls; attribution is applied to R2, R3 rows) (by design); also controller-limited
- R0 walk/brisk_mu0.54: not attributed (falls; attribution is applied to R2, R3 rows) (by design)
- R1 walk/brisk_mu0.54: not attributed (falls; attribution is applied to R2, R3 rows) (by design); also controller-limited
- R2/42 walk/brisk_mu0.54: body-limited: slip; also controller-limited
- R2/43 walk/brisk_mu0.54: controller-limited (decision 20; 3 falls: the sign test needs 5 for p < 0.05, so no limit was lifted; untestable candidates: cop_at_pad_edge_with_ankle_saturated); also controller-limited
- R0 walk/fr05_mu0.54: not attributed (falls; attribution is applied to R2, R3 rows) (by design)
- R1 walk/fr05_mu0.54: not attributed (falls; attribution is applied to R2, R3 rows) (by design)
- R2/42 walk/fr05_mu0.54: controller-limited (decision 20; untestable candidates: cop_at_pad_edge_with_ankle_saturated)
- R2/43 walk/fr05_mu0.54: controller-limited (decision 20; untestable candidates: cop_at_pad_edge_with_ankle_saturated); also controller-limited

#### Turn

| Condition | Gated for | R0 | R1 | R2/42 | R2/43 | R3/42 | R3/43 | R4/42 | R2 verdict (shown) | R3 verdict (not judged: stopped) | R4 verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| walking_0.35_left | all | FAIL 0,0 (0.00,0.00) | FAIL 12,11 (0.39,0.35) | pass 20,19 (0.86,0.78) | pass 20,19 (0.86,0.78) | not run | not run | not run | shown | not run | not run |
| walking_0.35_right | all | FAIL 2,2 (0.02,0.02) | FAIL 17,16 (0.66,0.60) | pass 20,19 (0.86,0.78) | pass 20,20 (0.86,0.86) | not run | not run | not run | shown | not run | not run |
| in_place_0.50_left | all | not run | pass 20,20 (0.86,0.86) | pass 20,20 (0.86,0.86) | pass 20,20 (0.86,0.86) | not run | not run | not run | shown | not run | not run |
| in_place_0.50_right | all | not run | pass 20,20 (0.86,0.86) | pass 20,20 (0.86,0.86) | pass 20,20 (0.86,0.86) | not run | not run | not run | shown | not run | not run |
| walking_0.50_left | none (reported) | not run | FAIL 12,1 (0.39,0.00) (rep.) | FAIL 20,9 (0.86,0.26) (rep.) | FAIL 20,8 (0.86,0.22) (rep.) | not run | not run | not run | body-limited (rep.) | not run (rep.) | not run (rep.) |
| walking_0.50_right | none (reported) | not run | FAIL 14,5 (0.49,0.10) (rep.) | FAIL 20,8 (0.86,0.22) (rep.) | FAIL 20,8 (0.86,0.22) (rep.) | not run | not run | not run | body-limited (rep.) | not run (rep.) | not run (rep.) |

- R0 turn/walking_0.35_left: not attributed (falls; attribution is applied to R2, R3 rows) (by design)
- R1 turn/walking_0.35_left: not attributed (falls; attribution is applied to R2, R3 rows) (by design); also body-limited: joint_range
- R0 turn/walking_0.35_right: not attributed (falls; attribution is applied to R2, R3 rows) (by design)
- R1 turn/walking_0.35_right: not attributed (falls; attribution is applied to R2, R3 rows) (by design); also controller-limited
- R1 turn/walking_0.50_left: body-limited: joint_range; also not attributed (falls; attribution is applied to R2, R3 rows)
- R2/42 turn/walking_0.50_left: body-limited: joint_range
- R2/43 turn/walking_0.50_left: body-limited: joint_range
- R1 turn/walking_0.50_right: body-limited: joint_range; also not attributed (falls; attribution is applied to R2, R3 rows)
- R2/42 turn/walking_0.50_right: body-limited: joint_range
- R2/43 turn/walking_0.50_right: body-limited: joint_range

#### Start and stop

| Condition | Gated for | R0 | R1 | R2/42 | R2/43 | R3/42 | R3/43 | R4/42 | R2 verdict (shown) | R3 verdict (not judged: stopped) | R4 verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| twice | all | not run | FAIL 15,15 (0.54,0.54) | pass 20,20 (0.86,0.86) | pass 20,20 (0.86,0.86) | not run | not run | not run | shown | not run | not run |

- R1 start_stop/twice: not attributed (falls; attribution is applied to R2, R3 rows) (by design)

#### Push

| Condition | Gated for | R0 | R1 | R2/42 | R2/43 | R3/42 | R3/43 | R4/42 | R2 verdict (shown) | R3 verdict (not judged: stopped) | R4 verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| standing_forward | all | not run | pass 20,20 (0.86,0.86) | pass 20,20 (0.86,0.86) | pass 20,20 (0.86,0.86) | not run | not run | not run | shown | not run | not run |
| standing_backward | all | not run | pass 20,20 (0.86,0.86) | pass 20,20 (0.86,0.86) | pass 20,20 (0.86,0.86) | not run | not run | not run | shown | not run | not run |
| standing_left | all | not run | pass 20,20 (0.86,0.86) | pass 20,20 (0.86,0.86) | pass 20,20 (0.86,0.86) | not run | not run | not run | shown | not run | not run |
| standing_right | all | not run | pass 20,20 (0.86,0.86) | pass 20,20 (0.86,0.86) | pass 20,20 (0.86,0.86) | not run | not run | not run | shown | not run | not run |
| walking_forward | all | not run | FAIL 1,1 (0.00,0.00) | FAIL 2,1 (0.02,0.00) | FAIL 2,1 (0.02,0.00) | not run | not run | not run | controller-limited | not run | not run |
| walking_backward | all | not run | FAIL 0,0 (0.00,0.00) | FAIL 0,0 (0.00,0.00) | FAIL 0,0 (0.00,0.00) | not run | not run | not run | controller-limited | not run | not run |
| walking_left | all | not run | FAIL 2,2 (0.02,0.02) | FAIL 2,2 (0.02,0.02) | FAIL 2,2 (0.02,0.02) | not run | not run | not run | controller-limited | not run | not run |
| walking_right | all | not run | FAIL 5,5 (0.10,0.10) | FAIL 7,6 (0.18,0.14) | FAIL 7,7 (0.18,0.18) | not run | not run | not run | controller-limited | not run | not run |

- R1 push/walking_forward: not attributed (falls; attribution is applied to R2, R3 rows) (by design)
- R2/42 push/walking_forward: controller-limited (decision 20; untestable candidates: cop_at_pad_edge_with_ankle_saturated); also controller-limited
- R2/43 push/walking_forward: controller-limited (decision 20; untestable candidates: cop_at_pad_edge_with_ankle_saturated); also controller-limited
- R1 push/walking_backward: not attributed (falls; attribution is applied to R2, R3 rows) (by design)
- R2/42 push/walking_backward: body-limited: slip
- R2/43 push/walking_backward: controller-limited (decision 20; untestable candidates: cop_at_pad_edge_with_ankle_saturated)
- R1 push/walking_left: not attributed (falls; attribution is applied to R2, R3 rows) (by design)
- R2/42 push/walking_left: body-limited: drive_torque
- R2/43 push/walking_left: controller-limited (decision 20; untestable candidates: cop_at_pad_edge_with_ankle_saturated)
- R1 push/walking_right: not attributed (falls; attribution is applied to R2, R3 rows) (by design)
- R2/42 push/walking_right: controller-limited (decision 20; untestable candidates: cop_at_pad_edge_with_ankle_saturated); also controller-limited
- R2/43 push/walking_right: controller-limited (decision 20; untestable candidates: cop_at_pad_edge_with_ankle_saturated)

#### Carpet

| Condition | Gated for | R0 | R1 | R2/42 | R2/43 | R3/42 | R3/43 | R4/42 | R2 verdict (shown) | R3 verdict (not judged: stopped) | R4 verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| robot_floor_carpet | R3/42, R3/43, R4/42 | n/a | n/a | FAIL 0,0 (0.00,0.00) (rep.) | FAIL 0,0 (0.00,0.00) (rep.) | not run | not run | not run | controller-limited (rep.) | not run | not run |
| low | R3/42, R3/43, R4/42 | n/a | n/a | FAIL 0,0 (0.00,0.00) (rep.) | FAIL 0,0 (0.00,0.00) (rep.) | not run | not run | not run | controller-limited (rep.) | not run | not run |
| medium | none (reported) | n/a | n/a | no stance | no stance | not run | not run | not run | not run (rep.) | not run (rep.) | not run (rep.) |
| high | none (reported) | n/a | n/a | no stance | no stance | not run | not run | not run | not run (rep.) | not run (rep.) | not run (rep.) |

The tracker (R0-R2) was never fitted on carpet (decision 23: the fit and G5 ran on hard floor only, and R2 fell in every start tried on robot_floor_carpet and low pile): the carpet walk is reported for R0-R2, with its fall attribution, and gated for R3 and R4.

- R2/42 carpet/robot_floor_carpet: controller-limited (decision 20; untestable candidates: cop_at_pad_edge_with_ankle_saturated)
- R2/43 carpet/robot_floor_carpet: controller-limited (decision 20; untestable candidates: cop_at_pad_edge_with_ankle_saturated)
- R2/42 carpet/low: controller-limited (decision 20; untestable candidates: cop_at_pad_edge_with_ankle_saturated)
- R2/43 carpet/low: controller-limited (decision 20; untestable candidates: cop_at_pad_edge_with_ankle_saturated)

Provenance of each row (PROVENANCE_RULE):

- R0 = R0: feedforward + PD on the commanded clock (ROWS_RULE)
- R1 = R1: the tracker on the true state, the body-level G5 selection (seed43_gen60)
- R2/42 = R2: the tracker on the estimate, CMA-ES seed 42's G5 selection seed42_gen60
- R2/43 = R2: the tracker on the estimate, CMA-ES seed 43's G5 selection seed43_gen60

### Looks (sway, head, tail)

Reported, not gated (LOOKS_RULE): sway against 0.12 H, head and tail amplitude against BEHAVIOR_SPEC's bands (head 0.05-0.15 rad, tail 0.1-0.2 rad).

| Condition | Row | Sway mean / max (H) | Head amplitude mean [range] (rad), in band | Tail amplitude mean [range] (rad), in band |
|---|---|---|---|---|
| walk/nominal | R0 | 0.184 / 0.188 | 0.051 [0.051-0.051], 1/1 | 0.120 [0.120-0.120], 1/1 |
| walk/nominal | R1 | 0.179 / 0.195 | 0.051 [0.051-0.051], 18/18 | 0.120 [0.120-0.120], 18/18 |
| walk/nominal | R2/42 | 0.179 / 0.191 | 0.051 [0.051-0.051], 20/20 | 0.120 [0.120-0.121], 20/20 |
| walk/nominal | R2/43 | 0.178 / 0.188 | 0.051 [0.051-0.051], 20/20 | 0.120 [0.120-0.121], 20/20 |
| walk/brisk | R0 | - | - | - |
| walk/brisk | R1 | 0.134 / 0.162 | 0.052 [0.051-0.053], 15/15 | 0.121 [0.121-0.121], 15/15 |
| walk/brisk | R2/42 | 0.136 / 0.148 | 0.052 [0.051-0.052], 5/5 | 0.121 [0.121-0.122], 5/5 |
| walk/brisk | R2/43 | 0.137 / 0.173 | 0.052 [0.051-0.052], 20/20 | 0.121 [0.121-0.121], 20/20 |
| walk/fr05 | R0 | - | - | - |
| walk/fr05 | R1 | - | - | - |
| walk/fr05 | R2/42 | - | - | - |
| walk/fr05 | R2/43 | - | - | - |
| walk/nominal_mu0.54 | R0 | 0.184 / 0.193 | 0.051 [0.051-0.051], 7/7 | 0.120 [0.120-0.121], 7/7 |
| walk/nominal_mu0.54 | R1 | 0.183 / 0.246 | 0.051 [0.051-0.053], 12/12 | 0.120 [0.120-0.121], 12/12 |
| walk/nominal_mu0.54 | R2/42 | 0.180 / 0.191 | 0.051 [0.051-0.052], 20/20 | 0.121 [0.120-0.121], 20/20 |
| walk/nominal_mu0.54 | R2/43 | 0.179 / 0.189 | 0.051 [0.051-0.051], 20/20 | 0.120 [0.120-0.121], 20/20 |
| walk/brisk_mu0.54 | R0 | - | - | - |
| walk/brisk_mu0.54 | R1 | 0.136 / 0.185 | 0.052 [0.052-0.053], 13/13 | 0.121 [0.121-0.122], 13/13 |
| walk/brisk_mu0.54 | R2/42 | 0.146 / 0.195 | 0.052 [0.052-0.054], 6/6 | 0.122 [0.121-0.123], 6/6 |
| walk/brisk_mu0.54 | R2/43 | 0.141 / 0.189 | 0.052 [0.051-0.054], 17/17 | 0.122 [0.121-0.122], 17/17 |
| walk/fr05_mu0.54 | R0 | - | - | - |
| walk/fr05_mu0.54 | R1 | - | - | - |
| walk/fr05_mu0.54 | R2/42 | - | - | - |
| walk/fr05_mu0.54 | R2/43 | 0.139 / 0.178 | 0.052 [0.052-0.052], 2/2 | 0.123 [0.123-0.123], 2/2 |
| carpet/robot_floor_carpet | R2/42 | - | - | - |
| carpet/robot_floor_carpet | R2/43 | - | - | - |
| carpet/low | R2/42 | - | - | - |
| carpet/low | R2/43 | - | - | - |

### Margin ledger

Cells: the threshold (the last value at which all 3 seeds survive and none fails a term it passed at nominal, or survive) and the nearest value that fails with the label there (falls labelled by lifting each candidate limit on the 3 seeds, ledger.LIFT_RULE); '*' the search is not finished or the record is flagged (listed below).

| Margin | Search | R2/43 works | R3/42 works | R2/43 survives | R3/42 survives | Stage 0 |
|---|---|---|---|---|---|---|
| Grip | walk_nominal | incomplete * | not run | incomplete * | not run | 0.47 |
| Grip | turn_left | incomplete * | not run | incomplete * | not run | 0.53 |
| Grip | turn_right | incomplete * | not run | incomplete * | not run | 0.53 |
| Grip | walk_brisk | incomplete * | not run | incomplete * | not run | 0.73 |
| Grip | turn (worse of turn_left, turn_right) | incomplete * | not run | incomplete * | not run | 0.53 |
| Push | standing_forward | not run | not run | not run | not run | 0.478 |
| Push | standing_backward | not run | not run | not run | not run | (forward 0.478) |
| Push | standing_left | not run | not run | not run | not run | (forward 0.478) |
| Push | standing_right | not run | not run | not run | not run | (forward 0.478) |
| Push | walking_forward | not run | not run | not run | not run | 0.478 |
| Push | walking_backward | not run | not run | not run | not run | (forward 0.478) |
| Push | walking_left | not run | not run | not run | not run | (forward 0.478) |
| Push | walking_right | not run | not run | not run | not run | (forward 0.478) |
| Stop | - | not run (plan deviation, decision 33) | not run (plan deviation, decision 33) | not run (plan deviation, decision 33) | not run (plan deviation, decision 33) | 1.25 |
| Drive speed | walk_fr05 | not run | not run | not run | not run | 0.95 (0.83 without the 15 % margin) |
| Drive torque | walk_nominal | not run | not run | not run | not run | 0.76 (walk RMS 0.76) |
| Model doubt: servo kp | walk_nominal | not run | not run | not run | not run | - |
| Model doubt: latency | walk_nominal | not run | not run | not run | not run | - |
| Model doubt: trunk mass | walk_nominal | not run | not run | not run | not run | - |
| Model doubt: trunk mass | walk_nominal_trunk_x1.1 | not run | not run | not run | not run | - |
| Model doubt: trunk mass | walk_nominal_robot_x1.1 | not run | not run | not run | not run | - |
| Model doubt: pile damping | stand_low_down | not run | not run | not run | not run | - |
| Model doubt: pile damping | stand_low_up | not run | not run | not run | not run | - |
| Model doubt: pile damping | carpet_low_down | not run | not run | not run | not run | - |
| Model doubt: pile damping | carpet_low_up | not run | not run | not run | not run | - |
| Pack voltage | walk_nominal_25.2V | not run | not run | not run | not run | - |
| Pack voltage | walk_nominal_22.2V | not run | not run | not run | not run | - |
| Pack voltage | walk_nominal_21V | not run | not run | not run | not run | - |
| Pack voltage | walk_fr05_25.2V | not run | not run | not run | not run | - |
| Pack voltage | walk_fr05_22.2V | not run | not run | not run | not run | - |
| Pack voltage | walk_fr05_21V | not run | not run | not run | not run | - |

- R3 is judged with R2 for this body (R3/42 stopped at 2015232 steps: G6b failed again after the fix (FIX_RULE) (decision 32); R3/43 stopped at 1753088 steps: the body's R3 is judged with R2 (train.R2_JUDGED_RULE, decision 38): its seed 42 run was stopped by FIX_RULE at 2015232 steps (G6b failed again after the fix), so this run cannot change a verdict; stopped to give the 75 cm runs the cores (decision 32)): its ledger row is not run (ledger.ROWS_RULE)
- Grip: Stage 0: the lowest friction the round-two turning screen requires in single support, per body: nominal walk / 0.35 rad/s walking turn (the worse direction) / brisk walk (STAGE0_RULE)
- Stop: not run (plan deviation, decision 33): the deceleration a stop holds needs stop references that decelerate at other rates, and the run holds one stop transition (STOP_RULE)
- Drive speed: Stage 0: the knee's speed need at Fr 0.05 over its rated speed, including the screens' 15 % speed margin; without it 0.83 (0.81 / 0.83 / 0.81), which is what the ledger's limit compares (STAGE0_RULE)
- Drive torque: Stage 0: the worst drive's sustained-torque ratio, the larger of the nominal walk's RMS and the one-foot hold (ankle_a's one-foot hold); the nominal walk's RMS alone 0.76 (neck), which is what the ledger's limit compares (STAGE0_RULE)
- Pack voltage: Stage 0: Stage 0 used 24 V; at 22.2 V the knee's rated speed (8.04 rad/s) equals the Fr 0.05 requirement including the screens' 15 % speed margin (8.08-8.12 rad/s; the peak swing speed itself is about 7.0 rad/s)

### PPO runs

| Run | Status | Steps / budget | Last checkpoint | Judged checkpoint | G6a zero residual | G6a env steps/s | G6b | Fix; G6b after it | G6c | Job h | CPU h | run.json written |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R3/42 | stopped_g6b | 2015232 / 4005888 | 2015232 | 1507328 (provisional) | yes | 81 (FAIL: cut the steps equally for every body, decision 26) | FAIL at 1007616 (survival policy 15.03 s, zero 15.49 s): make one fix, then stop | from 1007616 (task_weights 1,1,1,1,1,2,2,1,1); FAIL at 2015232 (survival policy 14.30 s, zero 15.49 s): stop the run (FIX_RULE) | - | 14.13 (running) | running | 2026-11-09T22:24:13+00:00 |
| R3/43 | stopped_r2_judged | 1753088 / 4005888 | 1753088 | 1753088 (provisional, not yet recorded) | yes | 83 (FAIL: cut the steps equally for every body, decision 26) | pass at 1007616 (survival policy 15.92 s, zero 15.49 s) | from 1007616 (task_weights 1,1,1,1,1,2,2,1,1); - | - | 14.17 (running) | running | 2026-11-09T22:24:58+00:00 |
| R4/42 | running | 1507328 / 4005888 | 1507328 | 1253376 (final) | - | 115 (FAIL: cut the steps equally for every body, decision 26) | - | - | - | 8.55 (running) | running | 2026-11-09T22:25:38+00:00 |

Elapsed PPO wall clock (the union of every run's sessions): 8.55 h; job-hours (their sum: the runs share the cores) 36.85 h.

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
| ppo | 8.55 (job-hours 36.84) | 14.79 | finished sessions' CPU, and for a running session its wall to the record's last write x the CPU per wall second of its first rollouts (an estimate); wall: the elapsed union of the sessions, job-hours beside |
| panels | 3.31 | 3.44 | episodes' and attributions' wall seconds (one process each): an upper bound on CPU |
| ledger | - | 0.05 | episodes' CPU seconds, each evaluation once (by its key) however many margins use it |
| total | | 24.14 | core-hours recorded so far |

## 75-M-H-10

### Verdicts by controller

| Controller | shown | provisional | body-limited | controller-limited | not attributed | fails by falls (R4: not attributed) | incomplete | not run |
|---|---|---|---|---|---|---|---|---|
| R2 (shown) | 14 | 0 | 3 | 3 | 0 | 0 | 0 | 0 |
| R3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 22 |
| R4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 22 |

Counts over the conditions gated for each controller's rows.
R4 has one seed (decisions 3 and 28): its verdicts are at most provisional (policy_rows.R4_VERDICT_RULE); its record on each gated condition: not run 22.

G7 pass: sway 0.1641 H (Stage 0 0.1641, plan 0.164), cadence 1.7612 steps/s (Stage 0 1.7612, plan 1.76).

### Verdict grid

Verdicts shown: R2 (interim: R3 is the controller judged on every behaviour, plan section 2; R2 is judged only if G6c fails, plan section 3 step 7); G6c not reached (plan section 3 step 7: R3's verdicts stand only if seed 42 passes it).  Cells: result, survived,succeeded of 20 (their one-sided 95 % Clopper-Pearson lower bounds); '(rep.)' reported, not gated, for that row; '*' interim (listed below each table); 'not run' no record; 'no stance' the floor has no settled stance for the body (panels.NOT_RUN_RULE); 'n/a' the row does not run there; R4's verdict is at most provisional (one seed), and its falls are never attributed (by design: decision 20's attribution is for R2 and R3).

#### Stand

| Condition | Gated for | R0 | R1 | R2/42 | R2/43 | R3/42 | R3/43 | R4/42 | R2 verdict (shown) | R3 verdict | R4 verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| hard | all | not run | pass 20,20 (0.86,0.86) | pass 20,20 (0.86,0.86) | pass 20,20 (0.86,0.86) | not run | not run | not run | shown | not run | not run |
| robot_floor_carpet | all | n/a | n/a | pass 20,20 (0.86,0.86) | pass 20,20 (0.86,0.86) | not run | not run | not run | shown | not run | not run |
| low | all | n/a | n/a | pass 20,20 (0.86,0.86) | pass 20,20 (0.86,0.86) | not run | not run | not run | shown | not run | not run |

#### Walk

| Condition | Gated for | R0 | R1 | R2/42 | R2/43 | R3/42 | R3/43 | R4/42 | R2 verdict (shown) | R3 verdict | R4 verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| nominal | all | not run | not run | pass 20,20 (0.86,0.86) | pass 20,20 (0.86,0.86) | not run | not run | not run | shown | not run | not run |
| brisk | all | not run | not run | FAIL 20,0 (0.86,0.00) | FAIL 20,0 (0.86,0.00) | not run | not run | not run | body-limited | not run | not run |
| fr05 | all | not run | not run | FAIL 20,0 (0.86,0.00) | FAIL 20,0 (0.86,0.00) | not run | not run | not run | body-limited | not run | not run |
| nominal_mu0.54 | all | not run | not run | pass 20,20 (0.86,0.86) | pass 20,20 (0.86,0.86) | not run | not run | not run | shown | not run | not run |
| brisk_mu0.54 | none (reported) | not run | not run | FAIL 20,0 (0.86,0.00) (rep.) | FAIL 20,0 (0.86,0.00) (rep.) | not run | not run | not run | body-limited (rep.) | not run (rep.) | not run (rep.) |
| fr05_mu0.54 | none (reported) | not run | not run | FAIL 20,0 (0.86,0.00) (rep.) | FAIL 20,0 (0.86,0.00) (rep.) | not run | not run | not run | body-limited (rep.) | not run (rep.) | not run (rep.) |

- R2/42 walk/brisk: body-limited: drive_sustained
- R2/43 walk/brisk: body-limited: drive_sustained
- R2/42 walk/fr05: body-limited: drive_sustained
- R2/43 walk/fr05: body-limited: drive_sustained, drive_speed
- R2/42 walk/brisk_mu0.54: body-limited: drive_sustained
- R2/43 walk/brisk_mu0.54: body-limited: drive_sustained
- R2/42 walk/fr05_mu0.54: body-limited: drive_sustained
- R2/43 walk/fr05_mu0.54: body-limited: drive_sustained, drive_speed, slip_median

#### Turn

| Condition | Gated for | R0 | R1 | R2/42 | R2/43 | R3/42 | R3/43 | R4/42 | R2 verdict (shown) | R3 verdict | R4 verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| walking_0.35_left | all | not run | not run | pass 20,20 (0.86,0.86) | pass 20,20 (0.86,0.86) | not run | not run | not run | shown | not run | not run |
| walking_0.35_right | all | not run | not run | pass 20,20 (0.86,0.86) | pass 20,20 (0.86,0.86) | not run | not run | not run | shown | not run | not run |
| in_place_0.50_left | all | not run | not run | pass 20,20 (0.86,0.86) | pass 20,20 (0.86,0.86) | not run | not run | not run | shown | not run | not run |
| in_place_0.50_right | all | not run | not run | pass 20,20 (0.86,0.86) | pass 20,20 (0.86,0.86) | not run | not run | not run | shown | not run | not run |
| walking_0.50_left | none (reported) | not run | not run | pass 20,20 (0.86,0.86) (rep.) | pass 20,20 (0.86,0.86) (rep.) | not run | not run | not run | shown (rep.) | not run (rep.) | not run (rep.) |
| walking_0.50_right | none (reported) | not run | not run | pass 20,20 (0.86,0.86) (rep.) | pass 20,20 (0.86,0.86) (rep.) | not run | not run | not run | shown (rep.) | not run (rep.) | not run (rep.) |

#### Start and stop

| Condition | Gated for | R0 | R1 | R2/42 | R2/43 | R3/42 | R3/43 | R4/42 | R2 verdict (shown) | R3 verdict | R4 verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| twice | all | not run | not run | pass 20,20 (0.86,0.86) | pass 20,20 (0.86,0.86) | not run | not run | not run | shown | not run | not run |

#### Push

| Condition | Gated for | R0 | R1 | R2/42 | R2/43 | R3/42 | R3/43 | R4/42 | R2 verdict (shown) | R3 verdict | R4 verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| standing_forward | all | not run | not run | pass 20,20 (0.86,0.86) | pass 20,20 (0.86,0.86) | not run | not run | not run | shown | not run | not run |
| standing_backward | all | not run | not run | pass 20,20 (0.86,0.86) | pass 20,20 (0.86,0.86) | not run | not run | not run | shown | not run | not run |
| standing_left | all | not run | not run | pass 20,20 (0.86,0.86) | pass 20,20 (0.86,0.86) | not run | not run | not run | shown | not run | not run |
| standing_right | all | not run | not run | pass 20,20 (0.86,0.86) | pass 20,20 (0.86,0.86) | not run | not run | not run | shown | not run | not run |
| walking_forward | all | not run | not run | FAIL 1,0 (0.00,0.00) | FAIL 1,1 (0.00,0.00) | not run | not run | not run | controller-limited | not run | not run |
| walking_backward | all | not run | not run | FAIL 0,0 (0.00,0.00) | FAIL 0,0 (0.00,0.00) | not run | not run | not run | body-limited | not run | not run |
| walking_left | all | not run | not run | FAIL 6,6 (0.14,0.14) | FAIL 5,4 (0.10,0.07) | not run | not run | not run | controller-limited | not run | not run |
| walking_right | all | not run | not run | FAIL 8,4 (0.22,0.07) | FAIL 4,2 (0.07,0.02) | not run | not run | not run | controller-limited | not run | not run |

- R2/42 push/walking_forward: controller-limited (decision 20; untestable candidates: cop_at_pad_edge_with_ankle_saturated); also controller-limited
- R2/43 push/walking_forward: controller-limited (decision 20; untestable candidates: cop_at_pad_edge_with_ankle_saturated)
- R2/42 push/walking_backward: body-limited: slip
- R2/43 push/walking_backward: body-limited: slip
- R2/42 push/walking_left: controller-limited (decision 20; untestable candidates: cop_at_pad_edge_with_ankle_saturated)
- R2/43 push/walking_left: controller-limited (decision 20; untestable candidates: cop_at_pad_edge_with_ankle_saturated); also body-limited: drive_speed
- R2/42 push/walking_right: body-limited: slip; also controller-limited
- R2/43 push/walking_right: controller-limited (decision 20; untestable candidates: cop_at_pad_edge_with_ankle_saturated); also controller-limited

#### Carpet

| Condition | Gated for | R0 | R1 | R2/42 | R2/43 | R3/42 | R3/43 | R4/42 | R2 verdict (shown) | R3 verdict | R4 verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| robot_floor_carpet | R3/42, R3/43, R4/42 | n/a | n/a | FAIL 0,0 (0.00,0.00) (rep.) | FAIL 0,0 (0.00,0.00) (rep.) | not run | not run | not run | body-limited (rep.) | not run | not run |
| low | R3/42, R3/43, R4/42 | n/a | n/a | FAIL 0,0 (0.00,0.00) (rep.) | FAIL 0,0 (0.00,0.00) (rep.) | not run | not run | not run | controller-limited (rep.) | not run | not run |
| medium | none (reported) | n/a | n/a | FAIL 0,0 (0.00,0.00) (rep.) | FAIL 0,0 (0.00,0.00) (rep.) | not run | not run | not run | controller-limited (rep.) | not run (rep.) | not run (rep.) |
| high | none (reported) | n/a | n/a | FAIL 0,0 (0.00,0.00) (rep.) | FAIL 0,0 (0.00,0.00) (rep.) | not run | not run | not run | controller-limited (rep.) | not run (rep.) | not run (rep.) |

The tracker (R0-R2) was never fitted on carpet (decision 23: the fit and G5 ran on hard floor only, and R2 fell in every start tried on robot_floor_carpet and low pile): the carpet walk is reported for R0-R2, with its fall attribution, and gated for R3 and R4.

- R2/42 carpet/robot_floor_carpet: body-limited: slip
- R2/43 carpet/robot_floor_carpet: body-limited: slip
- R2/42 carpet/low: controller-limited (decision 20; untestable candidates: cop_at_pad_edge_with_ankle_saturated)
- R2/43 carpet/low: controller-limited (decision 20; untestable candidates: cop_at_pad_edge_with_ankle_saturated)
- R2/42 carpet/medium: controller-limited (decision 20; untestable candidates: cop_at_pad_edge_with_ankle_saturated)
- R2/43 carpet/medium: controller-limited (decision 20; untestable candidates: cop_at_pad_edge_with_ankle_saturated)
- R2/42 carpet/high: controller-limited (decision 20; untestable candidates: cop_at_pad_edge_with_ankle_saturated)
- R2/43 carpet/high: controller-limited (decision 20; untestable candidates: cop_at_pad_edge_with_ankle_saturated)

Provenance of each row (PROVENANCE_RULE):

- R1 = R1: the tracker on the true state, the body-level G5 selection (seed43_gen40)
- R2/42 = R2: the tracker on the estimate, CMA-ES seed 42's G5 selection seed42_gen60
- R2/43 = R2: the tracker on the estimate, CMA-ES seed 43's G5 selection seed43_gen40

### Looks (sway, head, tail)

Reported, not gated (LOOKS_RULE): sway against 0.12 H, head and tail amplitude against BEHAVIOR_SPEC's bands (head 0.05-0.15 rad, tail 0.1-0.2 rad).

| Condition | Row | Sway mean / max (H) | Head amplitude mean [range] (rad), in band | Tail amplitude mean [range] (rad), in band |
|---|---|---|---|---|
| walk/nominal | R2/42 | 0.170 / 0.173 | 0.051 [0.050-0.051], 20/20 | 0.119 [0.119-0.119], 20/20 |
| walk/nominal | R2/43 | 0.168 / 0.171 | 0.051 [0.051-0.051], 20/20 | 0.119 [0.119-0.119], 20/20 |
| walk/brisk | R2/42 | 0.124 / 0.130 | 0.051 [0.051-0.051], 20/20 | 0.121 [0.120-0.121], 20/20 |
| walk/brisk | R2/43 | 0.123 / 0.128 | 0.051 [0.050-0.051], 20/20 | 0.121 [0.121-0.121], 20/20 |
| walk/fr05 | R2/42 | 0.115 / 0.128 | 0.051 [0.051-0.051], 20/20 | 0.124 [0.123-0.124], 20/20 |
| walk/fr05 | R2/43 | 0.127 / 0.188 | 0.051 [0.051-0.052], 20/20 | 0.122 [0.120-0.123], 20/20 |
| walk/nominal_mu0.54 | R2/42 | 0.170 / 0.174 | 0.051 [0.051-0.051], 20/20 | 0.119 [0.119-0.119], 20/20 |
| walk/nominal_mu0.54 | R2/43 | 0.168 / 0.172 | 0.051 [0.051-0.051], 20/20 | 0.119 [0.119-0.119], 20/20 |
| walk/brisk_mu0.54 | R2/42 | 0.125 / 0.135 | 0.051 [0.051-0.051], 20/20 | 0.121 [0.121-0.121], 20/20 |
| walk/brisk_mu0.54 | R2/43 | 0.124 / 0.131 | 0.051 [0.051-0.051], 20/20 | 0.121 [0.121-0.121], 20/20 |
| walk/fr05_mu0.54 | R2/42 | 0.117 / 0.137 | 0.051 [0.051-0.051], 20/20 | 0.123 [0.123-0.123], 20/20 |
| walk/fr05_mu0.54 | R2/43 | 0.134 / 0.187 | 0.051 [0.051-0.052], 20/20 | 0.119 [0.117-0.122], 20/20 |
| carpet/robot_floor_carpet | R2/42 | - | - | - |
| carpet/robot_floor_carpet | R2/43 | - | - | - |
| carpet/low | R2/42 | - | - | - |
| carpet/low | R2/43 | - | - | - |
| carpet/medium | R2/42 | - | - | - |
| carpet/medium | R2/43 | - | - | - |
| carpet/high | R2/42 | - | - | - |
| carpet/high | R2/43 | - | - | - |

### Margin ledger

Cells: the threshold (the last value at which all 3 seeds survive and none fails a term it passed at nominal, or survive) and the nearest value that fails with the label there (falls labelled by lifting each candidate limit on the 3 seeds, ledger.LIFT_RULE); '*' the search is not finished or the record is flagged (listed below).

| Margin | Search | R2/43 works | R3/42 works | R2/43 survives | R3/42 survives | Stage 0 |
|---|---|---|---|---|---|---|
| Grip | walk_nominal | incomplete * | not run | works to 0.1 * | not run | 0.36 |
| Grip | turn_left | incomplete * | not run | works to 0.1 * | not run | 0.41 |
| Grip | turn_right | incomplete * | not run | works to 0.1 * | not run | 0.41 |
| Grip | walk_brisk | incomplete * | not run | incomplete * | not run | 0.56 |
| Grip | turn (worse of turn_left, turn_right) | incomplete * | not run | works to 0.1 [turn_left] * | not run | 0.41 |
| Push | standing_forward | not run | not run | not run | not run | 0.589 |
| Push | standing_backward | not run | not run | not run | not run | (forward 0.589) |
| Push | standing_left | not run | not run | not run | not run | (forward 0.589) |
| Push | standing_right | not run | not run | not run | not run | (forward 0.589) |
| Push | walking_forward | not run | not run | not run | not run | 0.589 |
| Push | walking_backward | not run | not run | not run | not run | (forward 0.589) |
| Push | walking_left | not run | not run | not run | not run | (forward 0.589) |
| Push | walking_right | not run | not run | not run | not run | (forward 0.589) |
| Stop | - | not run (plan deviation, decision 33) | not run (plan deviation, decision 33) | not run (plan deviation, decision 33) | not run (plan deviation, decision 33) | 1.41 |
| Drive speed | walk_fr05 | not run | not run | not run | not run | 0.93 (0.81 without the 15 % margin) |
| Drive torque | walk_nominal | not run | not run | not run | not run | 0.77 (walk RMS 0.66) |
| Model doubt: servo kp | walk_nominal | not run | not run | not run | not run | - |
| Model doubt: latency | walk_nominal | not run | not run | not run | not run | - |
| Model doubt: trunk mass | walk_nominal | not run | not run | not run | not run | - |
| Model doubt: trunk mass | walk_nominal_trunk_x1.1 | not run | not run | not run | not run | - |
| Model doubt: trunk mass | walk_nominal_robot_x1.1 | not run | not run | not run | not run | - |
| Model doubt: pile damping | stand_low_down | not run | not run | not run | not run | - |
| Model doubt: pile damping | stand_low_up | not run | not run | not run | not run | - |
| Model doubt: pile damping | carpet_low_down | not run | not run | not run | not run | - |
| Model doubt: pile damping | carpet_low_up | not run | not run | not run | not run | - |
| Pack voltage | walk_nominal_25.2V | not run | not run | not run | not run | - |
| Pack voltage | walk_nominal_22.2V | not run | not run | not run | not run | - |
| Pack voltage | walk_nominal_21V | not run | not run | not run | not run | - |
| Pack voltage | walk_fr05_25.2V | not run | not run | not run | not run | - |
| Pack voltage | walk_fr05_22.2V | not run | not run | not run | not run | - |
| Pack voltage | walk_fr05_21V | not run | not run | not run | not run | - |

- Grip: Stage 0: the lowest friction the round-two turning screen requires in single support, per body: nominal walk / 0.35 rad/s walking turn (the worse direction) / brisk walk (STAGE0_RULE)
- Stop: not run (plan deviation, decision 33): the deceleration a stop holds needs stop references that decelerate at other rates, and the run holds one stop transition (STOP_RULE)
- Drive speed: Stage 0: the knee's speed need at Fr 0.05 over its rated speed, including the screens' 15 % speed margin; without it 0.81 (0.81 / 0.83 / 0.81), which is what the ledger's limit compares (STAGE0_RULE)
- Drive torque: Stage 0: the worst drive's sustained-torque ratio, the larger of the nominal walk's RMS and the one-foot hold (hip_yaw's one-foot hold); the nominal walk's RMS alone 0.66 (neck), which is what the ledger's limit compares (STAGE0_RULE)
- Pack voltage: Stage 0: Stage 0 used 24 V; at 22.2 V the knee's rated speed (8.04 rad/s) equals the Fr 0.05 requirement including the screens' 15 % speed margin (8.08-8.12 rad/s; the peak swing speed itself is about 7.0 rad/s)

### PPO runs

| Run | Status | Steps / budget | Last checkpoint | Judged checkpoint | G6a zero residual | G6a env steps/s | G6b | Fix; G6b after it | G6c | Job h | CPU h | run.json written |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R3/42 | running | 1753088 / 4005888 | 1753088 | 1753088 (provisional, not yet recorded) | yes | 74 (FAIL: cut the steps equally for every body, decision 26) | FAIL at 1007616 (survival policy 17.99 s, zero 18.00 s): make one fix, then stop | from 1007616 (task_weights 1,1,1,1,1,2,2,1,1); - | - | 14.21 (running) | running | 2026-11-09T22:26:22+00:00 |
| R3/43 | running | 1753088 / 4005888 | 1753088 | 1753088 (provisional, not yet recorded) | yes | 75 (FAIL: cut the steps equally for every body, decision 26) | FAIL at 1007616 (survival policy 16.94 s, zero 18.00 s): make one fix, then stop | from 1007616 (task_weights 1,1,1,1,1,2,2,1,1); - | - | 14.19 (running) | running | 2026-11-09T22:25:56+00:00 |
| R4/42 | running | 1253376 / 4005888 | 1253376 | 1253376 (final) | - | 100 (FAIL: cut the steps equally for every body, decision 26) | - | - | - | 8.55 (running) | running | 2026-11-09T22:25:38+00:00 |

Elapsed PPO wall clock (the union of every run's sessions): 8.56 h; job-hours (their sum: the runs share the cores) 36.95 h.

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
| ppo | 8.56 (job-hours 36.95) | 14.64 | finished sessions' CPU, and for a running session its wall to the record's last write x the CPU per wall second of its first rollouts (an estimate); wall: the elapsed union of the sessions, job-hours beside |
| panels | 3.41 | 3.44 | episodes' and attributions' wall seconds (one process each): an upper bound on CPU |
| ledger | - | 0.04 | episodes' CPU seconds, each evaluation once (by its key) however many margins use it |
| total | | 24.83 | core-hours recorded so far |

Core-hours recorded over the bodies reported: 73.10.
