# Draft pre-registration: the 75 cm track's tasks 2-4 (references, the walk's panels, the rest of the bar)

Draft by the architect, 14 Nov 2026, for the lead to edit, number into the plan and commit **before any reference
is frozen and before any closed-loop episode is run**. Body `75-M-H-10`; everything is simulation. Names are the
final ones of `ROUND_A_DESIGN.md` (same directory). Every rule ends with "Check:", the record field a reviewer
compares with it.

Tags: **[plan]** as `STAGE2_PLAN.md` section 5 states it; **[S1]** as Stage 1's rule, unchanged; **[NEW]** a
rule this round adds; **[differs: plan]** or **[differs: S1]** where it departs, with the departure named;
**[amends ruling 5]** where it departs from the lead's ruling of 14 Nov 23:10 UTC (the lead decides; the
alternative is given). Constants marked "(prototype)" were set on scratch prototypes of V0 on `75-M-H-10` on
14 Nov, before any closed-loop run; none was set on a closed-loop result.

## A. Sets and references (task 2)

**A1. Sets.** [NEW] Three reference sets on `75-M-H-10`, each a directory `stage2/sets/<set>/75-M-H-10/` in Stage
1's run-directory layout: `V0` (control: today's walk), `F25F9` (control: the PR's `_tuned_f9_spec`, verbatim),
`F25F9cl` (candidate: that spec with duty 0.59 and lift 0.095 H at every speed). Check: `set.json` `set`, `role`;
each walk entry's `definition.spec` equals `sets.SETS[set].spec_of(body, H, v)`; `tuned_f9_spec` equals the PR's
function field by field (a test imports the PR's script).

**A2. Speeds.** [plan; JETSON_PLEM_REVIEW task 2] Walks `nominal` (Froude 0.012: 0.2431 m/s) and `v0.3`, `v0.35`,
`v0.4`, `v0.45`, `v0.5`. V0: `sizing.NOMINAL` at the nominal speed; `look.speed_spec` between Stage 1's walks;
above `envelope.FAST`'s speed (0.4961 m/s) FAST's stride, duty and lift at the higher Froude number, labelled
`new`. F25F9cl: every rung other than 0.40, 0.45 and 0.50 m/s labelled `new`. Check: entries' `label`.

**A3. The candidate's margin was set after results.** [NEW label; JETSON_PLEM_REVIEW task 3] Duty 0.59 and lift
0.095 H were chosen by the PR after its results on seeds 3042-3061. Every F25F9cl entry and record carries
`margin_rule: "duty and lift set after results on seeds 3042-3061 (JPS 2.3)"`; rule H4 (fresh seeds) follows from
it. Check: the field, in every entry and record of the set.

**A4. One builder; Stage 1's transitions reproduce exactly.** [NEW] Every start, change, stop and turn comes from
`designlab.stage2.moves`. For Stage 1's 18 transitions on both picks it regenerates each array file to the
sha256 in `stage1/runs/2026-11-08/<body>/transitions.json`. Check: `stage2/sets/legacy_check.json`: 36 of 36
`equal`, `moves_sha256` = the sha256 of the `moves.py` that built the sets. V0's `refs/nominal.npz` and
`transitions/start.npz` hash to Stage 1's files; V0's look-format `nominal` and `v0.4` hash to the look ladder's
V0 rungs (entries' `equals_stage1`, `equals_ladder`).

**A5. Starts and speed changes are Stage 1's construction.** [S1] A start is stand to the set's `nominal`; a
change is `nominal_to_v<v>`, one stride of ramp, built with foreknowledge as Stage 1's (`transitions.generate`'s
method); no bar reading is tied to their command. Check: `move.causal: false`, `ramps`.

**A6. A commanded move is causal.** [differs: S1 (decision 18's convention); lead's ruling 2] For a stop and a
turn the reference is the periodic walk, frame for frame, up to the command; the preview controller first sees
the command at that frame; the file begins there. The command instant on a track is the move's first frame. A
command may only be given at a lift-off. Alternative named and not used: Stage 1's convention, command = start
of the reference's own ramp, under which the reference's CoM leaves the walk 0.4-0.6 s before the "command"
(scratch: 0.58 s for the stop from 0.40 m/s). Decision 18's reading is kept in every stop record. Check:
`move.causal: true`, `move.command_phase` a multiple of 0.5, `join_command` (model CoM within 1.0 mm over the
warm-up stride and at the command frame, joints within 1e-3 rad at it, multibody ZMP within 2.0 mm up to two
frames before it; the ZMP at the command frame recorded, not gated).

**A7. Gates of every move.** [S1 for G3's limits; NEW for the rest; constants (prototype)] `ik_ok`; last stride
against the periodic walk (or the stand): CoM within 1.0 mm, multibody ZMP within 2.0 mm (G3); non-causal moves:
first stride the same, final ZMP error at most 4.0 mm; causal moves: `join_command` (A6), consistency error at
most 2.5 mm, the excused transient confined to the first stride after the command (it fades to zero over phases
[command + 0.5, command + 1.0]), ZMP error after that stride at most 4.0 mm, single-support CoP at least 10 mm
inside the pad. Check: `pass_`, `join_a` / `join_command`, `join_b`, `zmp_error_history_m`,
`zmp_consistency_history_m`, `post_window_zmp_error_m`, `cop_margin_single_min`, `move.rules`.

**A8. Fallbacks for starts, changes and stops.** [NEW] A move that fails A7 or a blocking gate of A12 is retried
in this order and no other: (1) a change with a ramp of 2 strides; (2) for a ladder rung, a change from the
neighbouring rung toward 0.40 m/s; (3) recorded "no transition within the gates". A start and the stop have no
fallback. A failed `start` or `nominal_to_v0.4` stops the set and goes to the lead. Check: entries' `fallback`,
`available`.

**A9. The two-touchdown stop, `stop2_v0.4`.** [differs: plan in the command's place; lead's ruling 3] Stage 1's
stop plan (the left foot lifts last and lands beside the right; speed ramp over 0.4 stride about `phi_c = 0.5 -
duty / 2`) commanded causally at the right foot's lift-off, half a stride before the left foot's last lift-off.
The reference has exactly two touchdowns after the command, its forward speed is under 5 % of the walk's within
1.5 strides of the command, and its 1 Hz low-passed forward acceleration stays within 1.5 m/s^2. Stated limit of
that last gate: a zero-phase order-2 Butterworth at 1 Hz reads at most 2.22 x dv, so a stop from 0.42 m/s cannot
read above 0.93 m/s^2; the unfiltered peak is recorded beside it. `stop1_v0.4`, Stage 1's one-step stop on the
same walk (not causal), is built as a labelled comparison. Check: `reference.touchdowns_after_command == 2`,
`reference.forward_stopped_strides <= 1.5`, `reference.accel_forward_lp1hz_peak_m_s2`,
`reference.accel_forward_raw_peak_m_s2`, `reference.stage1_stop`.

**A10. The quick turn, `turn90_pi4_left` and `turn90_pi4_right`.** [plan for the turn; NEW for its construction]
One causal move from `v0.4` to `v0.4` on the heading + 90 deg at pi/4 rad/s: yaw rate and stride period ramp
together from the command, hold, ramp back; path speed, duty and lift stay the walk's; the command is the
lift-off of the inside foot; left and right are mirror plans.
- Ramps: **0.375 s of reference time, for every set, body and rate** [amends ruling 5, which said half a stride:
  half a stride is 0.387 s for V0 and 0.410 s for F25F9cl, which puts the candidate's reference at 79.7 deg at
  2.0 s of its own clock (scratch)]. Check: `turn.ramp_s == 0.375`.
- Appendages: inside the turn the tail's and the neck's swing amplitudes are multiplied by `(T(p) / T_walk)^2`
  [NEW; lead's ruling 1; a look choice, for every set, look sets included]. Check: `turn.appendage_rule`,
  `move.rules` holds `S2A-appendage`.
- The reference itself: total heading 90 deg within 1e-9 rad; at the bar's 2.0 s on the nominal commanded clock
  (reference time 2.1 s) at least 80 deg [plan, G9]; at the slowest commanded clock (reference time 1.995 s) at
  least 80 deg [NEW margin]. Check: `reference.heading_at_bar_deg`, `reference.heading_at_slowest_clock_deg`,
  `turn.heading_total_rad`.

**A11. The stride rule.** [NEW; the plan's remedy "its reference takes quicker steps" made a rule; constants
(prototype)] Inside the turn the stride period is `f x T_walk` at the same path speed. `f` is the largest value
of 1.00, 0.95, ..., 0.60 for which the finite turn's kinematic pass (no ZMP correction), in both directions, has
`ik_ok` and uses at most 0.85 of the usable thigh-twist range (+-0.30 rad) on the worse leg. The reference built
with all passes must show `ik_ok` and at most 0.87; if not, `f` steps down once, at most twice in all. Outcomes:
- no `f` passes, or the built reference is over a leg drive's Stage 0 limit: no reference for that set and rate;
  the turn conditions get "not run" records with the numbers; it goes to the owner (lever 6 or lever 1). Nothing
  is designed around it [lead's ruling 4].
- over a neck or tail planning limit after the appendage rule: frozen with `a1_over: true` and read under
  reading 3 only [lead's ruling 1].
The ladder's other rates from 0.40 m/s (0.35, 0.5, 1.0 rad/s) use the same move and rule. Check: `set.json`
`stride_rules[rate]` (every `f` tried with both directions' readings, the chosen `f`, `steps_down`), each turn's
`range_use`, `a1_over`, `not_available`.

**A12. Stage 0 screen of every reference, and what blocks a freeze.** [plan 4 A.3; NEW table] Every walk and
move has a block in the set's `screen.json` with every gate the look ladder's screen applies (drive ratios cont,
peak, speed; passive neck and tail loads; grip need at mu 0.9 and 0.54; sway; accelerations; the clearance gates;
joint-range margins). A reference is not frozen when: a leg drive's ratio exceeds 1; a neck or tail drive's ratio
exceeds 1 on a walk, start, change or stop (on a turn: A11); `ik_ok` is false; the single-support CoP rule fails;
a planned swing's mid-swing pad clearance is under 0.04 H; an interference, component-overlap or tail-clearance
gate fails; a joint has under 0.10 rad of range (a turn's thigh twist: A11's 0.87 instead); a start, change or
stop exceeds 1.5 m/s^2 of 1 Hz forward acceleration. Reported only: grip need, sway, passive loads, a walk's or
turn's acceleration. A walk rung that is not frozen is dropped from the ladder with its numbers. Check:
`screen.json` `blocks[name].verdict.blocks_freeze`, `a1_over`; `set.json` `freezes[-1].manifest`.

**A13. Freezing.** [NEW] Panels run only on a frozen set. A freeze lists every entry's name, kind, file sha256 and
entry-subset sha256; a later freeze may add entries and never drops or changes one; a changed reference takes a
new name or a new set version. Arrays are built at accuracy r2 with one BLAS thread. Check: `freezes[]`; each
entry's `built_by.accuracy == "r2"`, `built_by.blas_threads == 1`; a bar record's `set_freeze.manifest_sha256`.

## B. Episodes, rows, conditions

**B1. Episodes.** [plan; S1 `EPISODE_RULE`] 20 per condition and row; settled stance with Stage 1's start noise;
nominal plant; commanded 5 % above (the track played at 1.05 x); hard floor. Seed blocks: `seeds3042`
(3042-3061, the study's seeds, the headline) and `seeds5042` (5042-5061, replication) [NEW block].

**B2. Rows.** [JETSON_PLEM_REVIEW task 3; NEW names] `R2/42@<ms>ms` and `R2/43@<ms>ms`: Stage 1's two fitted
tracker controllers with the gains of `stage1/runs/2026-11-08/75-M-H-10/tune.json` unchanged; `R0@<ms>ms`: the
panels' open-loop row. `<ms>` is a servo delay of k = ms / 2 whole physics steps for the whole episode (Stage
1b's delay queue, `speedfit.servo_delay`), k = 0..5 (0, 2, 4, 6, 8, 10 ms) [plan amendment S2-9: whole steps].
`R0@10ms` is R0L at the judged delay. Check: `controller.gains` against `tune.json` and its sha256,
`controller.latency_steps`.

**B3. Judged delay.** [plan S2-7; owner decision 4] 10 ms is judged; 0-8 ms are reported beside it; nothing is
read between delays.

**B4. Floors.** [plan] Every condition at mu 0.9 (the floor as compiled) and at mu 0.54 (`_mu0.54`).

**B5. The track's LIPM frequency.** [differs: S1, which uses a 0.03 H crouch for every reference] `omega` is
computed with the set's own crouch (V0 0.03 H; the F25 sets 0.05 H on this body: about 1.2 %). The controllers'
gains are not refitted. Check: records' `omega`, `crouch_frac`.

**B6. The 16 gated conditions per set** [plan's table] and their plans [NEW]. Lead-in of every item: `start`,
one stride of `nominal`, `nominal_to_v0.4`, then `v0.4`.
- `walk/v0.4`, `walk/v0.4_mu0.54`: the lead-in, then `v0.4` for the fewest strides that hold the judged last
  10 s (Stage 1's `WALK_RULE` plan).
- `turn/q90_left`, `turn/q90_right` (+ `_mu0.54`): the lead-in, 2 strides of `v0.4` (2.5 for the right turn),
  the turn, then `v0.4` to the end; the episode lasts 5.2 s after the command at the slowest clock.
- `nudge/forward`, `backward`, `left`, `right` (+ `_mu0.54`): the lead-in and `v0.4`; a 0.1 s pulse at the
  trunk's CoM of `m x 0.25 x 0.589 m/s / 0.1 s` in the commanded heading's frame, at 2 + u strides into the
  0.40 m/s walk, u uniform in [0, 1) from the seed.
- `stop/two_step` (+ `_mu0.54`): the lead-in, 2.5 strides of `v0.4`, `stop2_v0.4`, then a hold of at least 2.0 s
  after the planned rest.
Reported conditions: `walk/nominal`, `walk/v0.3`, `walk/v0.35`, `walk/v0.45`, `walk/v0.5` (the speed ladder;
`_mu0.54` only where H6 lists it), `turn/t90_<0.35|0.5|1>_<dir>`, `nudge/<dir>_x0.375`, `_x0.5`,
`stop/one_step`. Check: records' `plan`, `seconds`, `definition`.

**B7. Shared lead-in and seed slot.** [differs: S1, which gives turns their own seed slots] Every item takes the
walk's seed slot, so every item of a seed meets the robot in the state the walk item judged. The 16 conditions
are therefore not independent: a fall in the lead-in shows in all eight items of that seed and grip. Check:
episodes' `command_s`, `fell_before_command`; records' `falls.before_command`, `falls.after_command`.

## C. Clocks and readings

**C1. Clocks.** [NEW precision of the plan's "2.0 s after the command"] Reference clock: track time. Simulated
time. The bar's clock: simulated seconds since the command instant `t_c`, the simulated time at which the
reference clock reaches the move's first frame. Every judged time (2.0 s, 3 s, 10 s) is in simulated seconds. A
turn at rate w is read at `at_s = (pi/2) / |w|` after its command (2.0 s for the quick turn) and held 3 s
[lead's ruling 5]. A nudge's "command" is its onset. Check: episodes' `command_s`, `command_ref_s`.

**C2. Three readings, in the plan's order.** [plan] (1) *shown with the look*: not defined for these three sets
(they carry no look bands): the block is `null` with that reason. (2) *shown*: every limit. (3) *shown but for
A1*: every limit, with the sustained-drive term read on the leg drives only (every drive but the neck's and the
tail's). Each episode has `success` and `success_but_A1`; each record has counts, a one-sided 95 %
Clopper-Pearson lower bound, and a pass for each. The two are never pooled. Expectation stated before the runs,
from Stage 1's own records: at 0.35 m/s and above the tail servo alone exceeded its sustained limit in 20 of 20
episodes of this body, so reading 2 may fail everywhere for that one term. Check: `counts`, `counts_but_A1`,
`measures.bar.drives`.

**C3. Toe clearance over each foot's own swing.** [plan; NEW definition; amends ruling 5 in one clause] Contact
is Stage 1's (a foot's load above 2 % of weight, 20 ms debounce, per 2 ms step). An *air run* of a foot runs from
a lift-off step to the next touchdown step. It is a **swing** if any 100 Hz update inside it falls in the
reference's swing of that foot, whatever its length, or if it lasts at least `0.5 x` the shortest planned swing
of the track (on the simulated clock); otherwise it is an *air gap* (stance chatter), counted and not read
[amends ruling 5, whose rule made every air run under half a swing an air gap, so that a touchdown at 48 % of
the swing or a mid-swing stub left the judged reading]. A reference swing holding two or more swings is
*interrupted*. Each swing wholly inside the window is read as the lowest point of the foot (every foot geom
above the floor, Stage 1's `low`) over the middle half of its own duration: the value interpolated at 25 % and at
75 % of the swing and every update between. The term `toe_clearance_own_swing` passes when the smallest reading
is at least 0.04 H (20.07 mm), no reference swing in the window is interrupted, and each foot has at least its
number of reference swings less one. Stage 1's reading (on the reference's clock) is recorded beside it,
unchanged. Check: `measures.bar.toe_clearance_own_swing_m`, `own_swings`, `ref_swings`, `interrupted_swings`,
`air_gaps`; the trace's `swings`, `contact`, `low100`.

**C4. The walk item.** [plan; lead's ruling 5 for the line] Window: the last 10 s of a complete episode. `along`:
the CoM covers at least `10 s x` the rung's target speed along the commanded heading (4.0 m at 0.40 m/s).
`line`: the CoM stays within 0.25 m of the line along the commanded heading through its position at the
window's start (the deviation from the line through the episode's start is reported). Other terms, Stage 1's
limits [S1]: survived; duty 0.55-0.65 (0.55-0.72 below 0.30 m/s, owner decision 9); no flight; roll and pitch
4 deg RMS and 8 deg peak; `toe_clearance_own_swing` (C3); slip 5 mm median and 15 mm worst; sustained drive RMS
over the window at most its rating (C2); drive speed; joint range 0.10 rad. Reported, not judged: peak ground
reaction (Stage 1 drops it, cadence and stride for every walk but the nominal one), heading drift. Stage 1's
reading (`stage1`): `tune.measure`'s terms restricted to the panels' list for that walk. Check: `terms`,
`measures.bar`, `measures.stage1`.

**C5. The turn item.** [plan; lead's ruling 5 for the start heading] `heading`: the trunk's heading is within
10 deg of the commanded heading at the command + 90 deg at `t_c + at_s` and at every update for 3 s after. Time
to within 10 deg and the turn measured from the trunk's own heading at the command are reported. Over `[t_c,
t_c + at_s + 3 s]`: the walk's limits of C4 (without `along` and `line`), the joint range by decision 14 (at
least the reference track's own smallest margin less 0.02 rad, capped at 0.10 rad), and no leg interference
(50 Hz poses on the screened model). Peak ground reaction reported, as in the walk item [NEW: Stage 1 judged it
in its turns, which were from the nominal walk]. Check: `measures.bar.heading_error_at_deg`,
`heading_error_hold_max_deg`, `time_to_within_10deg_s`, `measures.legs`.

**C6. Nudges.** [plan; S1 decision 30] No fall; back on the command (0.9-1.1 x the commanded speed along the
heading and at most 0.1 x across, by stride means) within 3 s of the pulse's end, held to the planned end; drive
and range limits over the episode. Check: `measures.push`, `measures.recovery`.

**C7. The stop item.** [plan] `touchdowns_after_command`: at most two touchdowns (a foot's contact beginning)
with `t_c < t <= t_stop`, `t_stop` the first time from which the CoM's forward speed stays under 5 % of the
command (0.42 m/s) to the end of the trace; `stopped_by_end`; `stop_acceleration_1hz` at most 1.5 m/s^2 over two
strides from the command (A9's stated limit applies); drive and range limits over the episode. Reported: the
stopped time in seconds and strides (decision 18's quantity, now from a causal command), each counted
touchdown's foot and time, the last touchdown before the command and its distance to it (the left foot's
scheduled landing is 47-85 ms of reference time before the command, by set, so a late landing counts and is visible), the unfiltered
deceleration peak. Check: `measures.bar.touchdowns_after_command`, `touchdown_table`,
`last_touchdown_before_command`, `forward_stopped_s`.

**C8. An episode that cannot be read is an error.** [S1 `FAILED_EPISODE_RULE`, extended] An episode that raised,
or that did not fall and whose trace ends before its judged window, is run again; it is never a failed term.

## D. Verdicts, replication, budget (rules H1-H6)

**H1. Pass per record.** [S1 `PASS_RULE`] 20 of 20 up and at least 17 succeed, on the block's own 20 seeds, for
each reading. **H2. Per controller.** [plan] `shown` if both `R2/42` and `R2/43` pass at that delay,
`provisional` if one. The verdict at 10 ms is the judged one. **H3. The bar** [plan]: the 16 gated conditions at
10 ms; a condition without a reference is "not run" with its reason and is not a pass.

**H4. Replication on fresh seeds.** [NEW; lead's ruling 5] Every `seeds3042` record of F25F9cl (a row, a
condition, a delay) that passes in reading 2 or in reading 3 is run again on `seeds5042` with the same row. The
headline stays `seeds3042`; the replication is printed beside it and never pooled. Check: `block`,
`replication_of`.

**H5. Fall attribution.** [S1 decision 20, with a NEW citation rule] On R2 rows' records with 5 or more falls:
each candidate limit lifted over the same 20 seeds, sign test at 0.05. A turn, nudge or stop record with fewer
than 5 falls after its command cites the walk record of the same set, row, grip and block. A label is a hint,
not a proof. Check: `attribution`, `falls`.

**H6. Budget and order.** [lead's budget; NEW ledger] 12 core-hours for task 3 and 12 for task 4, counted as the
sum of episode wall times in `budget.json`. A tier starts only if the spent time plus its estimate (from the
committed `COST.md`, measured before the first tier) fits; a started tier finishes; attribution is checked per
record; whatever is not started is reported "cut by the pre-registered order". Order:
- Task 3: (1) `walk/v0.4` and `_mu0.54` at 10 ms on F25F9cl, F25F9, V0 under both R2 rows and R0, with
  attribution; (2) replication of (1); (3) the same at 0 ms; (4) 4 then 6 ms; (5) 2 then 8 ms; (6) the speed
  ladder at mu 0.9 on F25F9cl then V0 at 10 ms, then 0 ms; (7) the PR's remaining rows: F25F9 `walk/v0.5` at
  0 ms, F25F9cl `walk/v0.45_mu0.54` and `walk/v0.5_mu0.54` at 0 ms, F25F9cl `walk/v0.5` at 2 ms;
  (8) replication of (3)-(7); (9) the ladder at 4 then 6 ms; (10) attribution of the remaining records.
- Task 4, on F25F9cl then V0, the 14 turn, nudge and stop conditions: (1) 10 ms with attribution;
  (2) replication of (1); (3) 0 ms; (4) 4 ms; (5) 6 ms; (6) the reported ladders (turn rates, nudges at 0.375
  and 0.5 x the bound, `stop/one_step`) at 10 ms, mu 0.9; (7) replication of (3)-(5).
Check: `budget.json`; `done/` markers' order in the logs; the report's "cut" cells.

## E. Records and code (rules R1-R5)

**R1. Keys.** [S1 `RESUME_RULE`, narrowed] A record's key holds the row's description and sources, the condition,
plan, seconds, seeds, command, the body's plant inputs and library versions (`panels.key_context`), the sha256 of
each array file played and of the entry numbers the planner reads, and the source hashes of the panels' modules
and of `bar.py`, `readings.py`, `setio.py`, `speedfit.py`. **R2. No overwrite.** [differs: S1, whose `eval` reruns
and overwrites on a key change] A finished record whose key differs is never replaced silently: the run stops;
`--supersede` keeps the old record beside the new. **R3. Traces.** [S1 `TRACE_RULE`, extended] Stage 1's 50 Hz
trace plus the 500 Hz contact flags, both feet's 100 Hz low points and the per-swing table, with its sha256 in
the episode record. **R4. Code state A.** [NEW] The commit at which the three sets are frozen. Before it: the
legacy check (A4) and the closed-loop regression (`stage2/regress/regress.json`: committed Stage 1 and 1b
episodes re-run from the code at that commit reproduce their traces' sha256 and their episode records) both
pass; one Stage 1 edit is in (`control.Track.add_periodic` reads an optional `window_strides` key; without the
key the path is unchanged). From state A to the end of tasks 3-4 no module in a key changes; if one must, the
affected baseline rows are run again into a new directory. **R5. Reports.** [S1] Tables are machine-made from
the records, cite record paths, keep the PR's numbers in columns labelled as its claims, and never set a measured
value against a target.

## F. Not run in this round, by name

The turn and nudges from the nominal walk, carpet walks, stiffness x0.7, `robot_floor_carpet`; delays of 14, 16
and 20 ms; the look gates and reading 1; 0.55 m/s; the Jetson-mass row of JPS 2.3; a stop from 0.50 m/s; any
60 cm body. Each appears in the report as "not run".

## For the lead before committing this text

1. C3's air-gap clause and A10's ramp amend ruling 5; choose, and delete the tag or the amendment.
2. H6's order decides what the budgets cut (likely task 4's 6 ms tier, V0's 4 ms tier, the reported ladders,
   attribution of non-judged delays); confirm it or enlarge the budgets.
3. C2's expectation sentence commits to reading 2 as the headline even if the control fails it everywhere;
   confirm the wording.
4. B7 (shared seed slot) and C5 (ground reaction reported in the turn) depart from Stage 1; confirm each.
5. R4 moves the one Stage 1 edit to before state A; the alternative is no Stage 1 edit in this round.
