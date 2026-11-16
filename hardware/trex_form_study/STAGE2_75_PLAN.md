# Stage 2, the 75 cm track: rules fixed before the runs (tasks 2–4)

**Status (written 14 Nov 2026; settled 15 Nov 2026).** This text was written on 14 Nov 2026, and the points its
independent review left open were settled by the study's lead on 15 Nov 2026: both before any reference set is frozen
and before any Stage 2 closed-loop episode is run. Simulation only: it changes no body, and nothing is adopted without
the owner (P24). It carries out `JETSON_PLEM_REVIEW.md` section 5, tasks 2–4 (record the references; the official panels
for the walk; the rest of the bar), on the 75 cm body `75-M-H-10`, under the owner's decisions of 14 Nov 2026
(`DECISION_LOG.md` P30). It is the detailed rule text that `STAGE2_PLAN.md` section 9, rule S2-16, extends to every body
and reference set of Stage 2, but for the rules S2-16 names as the 75 cm round's own. Paths are relative to
`hardware/trex_form_study/`. A rule changed after a result is labelled so. The text was reviewed independently before
its commit. The lead's rulings of 15 Nov on the twelve points that review left stand in the rules they settle: in this
text each is marked "[settled 15 Nov, before any set was frozen]", and the wording of points 11 and 12 stands in
`STAGE2_PLAN.md` section 9, whose opening paragraph names it; the last section lists them. Between the commit that adds
this file and the first unit of task 3 the only closed-loop episodes run on a Stage 2 set are the build's checks: single
episodes on set V0 under `R0` rows, each listed in `stage2/runs/2026-11-15-75cm/dev_episodes.json` and none a record
(rule R4). Check: the commit that adds this file (15 Nov 2026) and the commit that adds the settled wording are
ancestors of the commit named in every `set.json` `freezes[].commit` and in every record's `sessions[].commit`;
`dev_episodes.json` (rule R4).

## What had been run when these rules were fixed

- No Stage 2 closed-loop episode on any new reference, no fit and no training run.
- Kinematic and Stage 0 scratch prototypes of references on `75-M-H-10` and `60-N-T-10`: thigh-twist screens of steady
  turns, a prototype causal stop and quick turn of today's 0.40 m/s walk, and their Stage 0 drive ratios. Constants
  marked "(prototype)" below were set on these. They are prescribed-motion results in simulation, not records.
- Reproductions of committed Stage 1 and 1b episodes from unchanged code, to check that this container reproduces the
  first session's simulations: 160 of Stage 1b's 10 ms turn episodes of `60-N-T-10` (`R2s/42@10ms` and `R2@10ms`, four
  turn conditions, 20 seeds each), each trace byte-identical to the sha256 in its committed episode record, and four of
  Stage 1's 8 Nov episodes of `75-M-H-10`, also byte for byte. These add no result.
- Known when these rules were fixed, and not run by this study: the PR's own closed-loop runs of its tuned walks on this
  body (simulation; "the PR" and JPS are explained under "Words, names and tags"). Its script recorded a 3-episode
  survival probe of the tuned walk with look feature F9's tail at 0.40 m/s under R2, on seeds other than the panels': 3,
  3, 3 and 1 of 3 stayed up at 0, 2, 6 and 10 ms (`stage2/jetson_plem/jetson_plem_study.json`). JPS 2.3 gives a
  20-episode table on seeds 3042–3061 under `R2/42` and `R2/43`, for which no code or record exists
  (`JETSON_PLEM_REVIEW.md` section 1): as the PR claims it, the candidate meets every limit at 0 and 2 ms and falls in
  every episode at 6 to 10 ms. Neither is a record of this study. Rules A3 and H4 answer the table, and C2 says what it
  leads one to expect.

The prototype numbers behind the rules and their "(prototype)" constants follow. All are scratch prototypes of 14 Nov on
`75-M-H-10`: prescribed motion in simulation, run while these rules were being designed. None is a record and none may
be cited as a result; task 2's official builds replace them. Beside today's walk (V0), whose stop and quick turn were
built in full, they hold passes on the candidate's walk (F25F9cl): its quick turn at plan level and as kinematic passes,
and its change to 0.40 m/s built in full. The abbreviations are explained under "Words, names and tags" below.

| Scratch prototype | Numbers (not records) | Bears on |
|---|---|---|
| Stage 1's one-step stop, built from V0's 0.40 m/s walk | its CoM leaves the continuing walk (by more than 1 mm) 0.58 s before Stage 1's "command"; forward deceleration at 1 Hz 0.64 m/s² | A6, A9 |
| A causal stop on the cart-table model | one step, commanded at the last lift-off: the ZMP would be 69 mm off its target (the pad reaches 63 mm); Stage 1's stop plan commanded at the right foot's lift-off: 8 mm, two touchdowns after the command | A9 |
| That two-touchdown stop built in full (6 corrections) | `ik_ok`; consistency error 18.9 → 0.67 mm; causal transient 9.2 mm; join at the command: CoM 0.006 mm; join to the stand: CoM 0.15 mm, ZMP 0.43 mm; touchdowns 0.371 s and 0.801 s after the command; forward speed below 5 % from 1.033 s, 1.200 strides; deceleration at 1 Hz 0.647 m/s² (0.691 at the 5 % fast clock), unfiltered 2.16 m/s² | A7, A9 |
| The quick turn at plan level, with ramps of half a stride | V0 (stride factor 0.8, ramp 0.3873 s): heading 80.74° at 2.0 s of reference time, 85.17° at the bar's 2.0 s, 80.51° with the clock held 5 % slow; F25F9cl (stride factor 0.65, ramp 0.4098 s): 79.67°, 84.15°, 79.45° | A10 |
| Kinematic passes of the quick turn at other ramp lengths | thigh-twist use: V0 0.818 at both ramps tried; F25F9cl 0.823, 0.822, 0.820 for ramps of 0.410, 0.385, 0.361 s, with the reference heading at the bar's 2.0 s 84.15°, 84.76°, 85.37° | A10 |
| The quick turn from V0's 0.40 m/s walk built in full, stride ×0.8, left and right, with ramps of half a stride (0.387 s) and before the appendage rule | `ik_ok`; thigh-twist use 0.817 on the worse (outside) leg and 0.765 on the other; consistency error 48 → 1.7 mm; causal transient 27.5 mm; join at the command: CoM 0.001 mm; last stride against the walk: CoM 0.10 mm, ZMP 1.42 mm; heading 90.000000° | A7, A11 |
| Kinematic passes of the quick turn at other stride factors | thigh-twist use: V0 ×0.85 0.875; ×0.9 and ×1.0 pinned, with IK failures; F25F9cl ×0.7 0.894, ×0.65 0.823, ×0.6 0.753. The stride rule would pick 0.80 for V0 and 0.65 for F25F9cl | A11 |
| The excused transient in the V0 right turn | ZMP term 27.6 mm in the first half stride, 6.0 mm in the second, then 2.0–3.7 mm at every lift-off (the walk itself: 1.85 mm); faded to zero over phases [command + 0.5, command + 1.0], the join reads ZMP 0.727 mm (from 1.42 mm) and CoM 0.105 mm | A7 |
| Stage 0 ratios of that V0 turn, before the appendage rule | hip-pitch speed 0.923, knee speed 0.925, every leg ratio below 1; tail peak 1.199 (the walk: 0.728); single-support CoP margin 24 mm (the walk: 45 mm); single-support grip need 0.90 (the walk: 0.63) | A7, A10, A11 |
| F25F9cl's quick turn at stride ×0.65, kinematic pass | hip-pitch speed ratio 0.968, knee 0.861, thigh-twist use 0.823 | A11 |
| The change from the nominal walk to 0.40 m/s, Stage 1's construction | V0: CoM 0.29 mm, ZMP 0.58 mm; F25F9cl: CoM 0.25 mm, ZMP 0.96 mm, `ik_ok` | A5, A7 |

What the table does not hold. No turn has been built in full with the registered 0.375 s ramp (A10) or with the
appendage rule: the full builds used ramps of half a stride, the kinematic passes ran at 0.410, 0.385 and 0.361 s, and
0.375 s was set on plan-level arithmetic of both sets (the reference's heading at the judged instant moves 24.8° per
second of ramp, which puts both references at or above 80.3° at the slowest clock). In a prototype of a causal move the
multibody ZMP at the command frame itself read 1.99 mm, its central difference straddling the command; A6 therefore
records that frame and gates the frames up to two before it. Scratch checks of code were also run (the new plan function
against Stage 1's on its 18 transitions; array files' hashes against Stage 1's and the look ladder's): they compare
arrays and add no result.

Four readings of committed Stage 1 records (simulation) also bear on the rules:

- Stage 1's 0.5 rad/s turn references use 0.847 of the thigh twist on this body, and `R2/42` and `R2/43` kept
  decision 14's range there in 20 of 20 episodes at 0 ms (A11's 0.85).
- Stage 1's own 36 transitions have a largest final ZMP error of 3.8 mm (A7's 4.0 mm).
- In this body's walks at 0.351 m/s and at Froude 0.05 under `R2/42` and `R2/43`, 20 of 20 episodes stayed up and none
  succeeded: the sustained-drive term failed in 20 of 20, with the tail servo the worst drive (C2's first expectation).
- In Stage 1's margin ledger (3 episodes per value) this body's nominal walk met every limit up to 6 ms of servo delay
  and fell in 3 of 3 episodes at 10 ms under `R2/43`, though it stayed up at 20 ms, the only larger delay tried
  (`DECISION_LOG.md` F26 and A17; C2's second expectation).

## In brief, for the owner

- **What is built (task 2).** Three sets of reference motions for the 75 cm body: `V0`, today's walk (a control);
  `F25F9`, the tuned walk with the counter-phased tail as pull request #668 defines it (a control); and `F25F9cl`, that
  walk with a little more duty and foot lift (the labelled candidate of P30; it is not chosen). Each set holds walks at
  the nominal speed and at 0.30–0.50 m/s, a start, speed changes, a stop and 90° turns to the left and to the right.
  Every motion is priced by the Stage 0 screen and frozen by its hash before any episode uses it.
- **What is run (tasks 3 and 4).** The study's official panels: 20 episodes per condition on its seeds 3042–3061, under
  Stage 1's two fitted balance controllers, unchanged. First the walk at 0.40 m/s on all three sets, with an open-loop
  row beside the controllers. Then the rest of the agility bar on the candidate and on today's walk: the 90° turn each
  way, a nudge from each of four sides, and the stop. As far as the budget reaches, also: the walk at the other speeds,
  slower and faster turns, harder nudges, Stage 1's one-step stop, the PR's remaining rows, and one control run of the
  candidate in which the controller's pendulum frequency is computed as Stage 1 computed it.
- **What runs before task 3.** On these three sets, only the build's checks: single episodes of today's walk (`V0`)
  under the open-loop row, each listed and none a record. The two controllers and the servo delay are smoke-tested only
  on Stage 1's own conditions, whose results are already published, and the rates behind the cost estimate are measured
  there too. No episode of the tuned walk or of the candidate, and no controller row on any of the three sets, runs
  before task 3.
- **Delays and grips.** The verdict is read at 10 ms of servo delay. The walk is also run at 0, 2, 4, 6 and 8 ms and the
  rest of the bar at 0, 4 and 6 ms, as far as the budget reaches; these are reported beside the verdict and never read
  as it: a cell at another delay gives its counts and reads "passes at 4 ms" (with its own delay), never "shown" or
  "provisional". Every condition of the bar runs on a grippy floor (μ 0.9) and on a slippery one (μ 0.54); the ladders
  run on the grippy floor.
- **What counts as shown.** A condition passes when all 20 episodes stay up and at least 17 meet every limit. It is
  "shown" when both controllers pass at 10 ms and "provisional" when one does. Each episode is read twice: against every
  limit (the headline), and against every limit but the neck and tail servos' sustained load ("shown but for A1"),
  printed beside it and never pooled. Two things are expected before the runs, from results already known (simulation).
  In Stage 1's records this body's tail servo was over its sustained limit in every episode of the walks at 0.35 m/s and
  faster under both controllers, so the headline reading may fail everywhere for that one reason. And in Stage 1's
  records this body's slow walk fell at 10 ms of delay under one of these controllers (3 episodes), while the PR's own
  unrecorded table has the candidate falling at 6 to 10 ms: the 10 ms verdicts may be "not shown" because the robot
  falls, whatever the walk.
- **The candidate's margin was chosen after results.** Its duty and lift were set after the pull request's results on
  the study's own seeds. So whatever the candidate passes under either controller is run again on 20 fresh
  seeds (5042–5061), as far as the budget reaches; the result is printed beside the headline and never pooled with it. A
  candidate's verdict is never printed alone: beside it stands "replicated", "not replicated" or "replication cut by the
  pre-registered order". A pass of the open-loop row is not run again (that row gives no verdict), and the two controls
  are not run on the fresh seeds.
- **New or stricter than Stage 1.** The command is causal: until the instant of a stop or turn command the reference is
  the plain walk, frame for frame (Stage 1's stop reference began to leave the walk about half a second before its
  "command"), and the bar's times are simulated seconds from that instant. Toe clearance is read over each foot's own
  swing, from its lift-off to its touchdown, instead of on the reference's clock. Drift is judged: over the walk's last
  10 s the robot must cover 4.0 m along the commanded heading and stay within 0.25 m of a straight line. One limit is
  looser than in Stage 1's turns: peak ground reaction is reported, not judged, as in the 0.40 m/s walk.
- **What may come back to you.** The quick turn needs quicker steps to stay inside the thigh twist's range (±0.30 rad).
  A set that has no quick turn within that range and the leg drives' limits gets no turn reference: its turn conditions
  are recorded "not run" with the numbers, nothing is designed around it, and the question comes to you (more
  thigh-twist range, lever 6, or faster leg drives, lever 1). A turn whose reference fails another of its gates is
  recorded "not run" in the same way and also comes to you; a turn has no other fallback. One look choice is made for
  every set: inside the turn's quicker steps the tail and the neck swing less, in proportion to the square of the step
  time.
- **Compute.** 16 core-hours for the walk's panels (task 3) and 12 for the rest of the bar (task 4), spent in a fixed
  order with the judged 10 ms runs first. They come on top of the plan's estimate for this CPU (about 80 core-hours,
  written before this track existed): with the sets and screens (about 4 core-hours) the track adds about 32 core-hours,
  an estimate. The order is run unit by unit (one set at one delay); the first unit that does not fit is cut with every
  unit after it, and whatever is cut is reported, by name, as "cut by the pre-registered order".
- **Nothing is adopted.** Everything here is simulation. The candidate is priced with the Stage 0 screen, run through
  the official panels and shown to you beside the look ladder with clips (S2-15; P30); it is not chosen, and V6, the
  look you picked on 14 Nov, stays the look reference until you decide otherwise. The last section lists how the study's
  lead settled, on 15 Nov and before the first set was frozen, the twelve points an independent review had left.

## Words, names and tags

- **Documents and numbers.** "plan" is `STAGE2_PLAN.md` and "S2-n" its rule n; "decision n" is `STAGE1_PLAN.md`'s; S1 is
  Stage 1. P24, P30, A1, A17, F25 and F26 are rows of `DECISION_LOG.md`. V6 is the look the owner picked on 14 Nov,
  every look feature (P30). "The PR" is pull request kuds/mesozoic-labs#668, reviewed in `JETSON_PLEM_REVIEW.md`; JPS is
  its report, `stage2/jetson_plem/JETSON_PLEM_AND_SPEED.md`, and JPS 2.3 that report's section 2.3, the closed-loop
  table that task 3 turns into records.
- **The body.** `75-M-H-10` is the 75 cm pick. H is its hip height (0.501854 m), the unit of every ratio. The "thigh
  twist" is the hip-yaw joint, usable over ±0.30 rad.
- **Motion.** CoM: centre of mass. ZMP: zero-moment point, the point on the floor about which the motion's tipping
  moment is zero; the "multibody ZMP" is computed from the whole model. CoP: centre of pressure under a foot. IK:
  inverse kinematics; `ik_ok` says it solved on every frame. LIPM: the linear inverted pendulum model, whose natural
  frequency is `omega`. The preview controller is Stage 1's planner of the CoM path, which looks ahead along the ZMP
  target. Duty: the share of a stride a foot spends on the ground. Froude number: v²/(gH). μ: the floor's friction
  coefficient (the grip). RMS: root mean square. "At 1 Hz": low-passed at 1 Hz by a zero-phase Butterworth filter of
  order 2. `T_walk` is the walk's stride period, `T(p)` the stride period at phase p of a move, and `dv` a change of
  speed.
- **Stages and gates.** Stage 0 is the screen of a prescribed motion with perfect tracking: drive ratios against the
  planning limits, grip need, clearances. Stage 1 and 1b are the study's earlier closed-loop rounds. G3 is Stage 1's
  gate on a transition's joins; G9 is the plan's validation gate (plan 4 B.3). "The bar" is the plan's agility bar (plan
  section 5). "The panels" are Stage 1's official evaluation (`designlab/stage1/panels.py`), whose named rules
  (`EPISODE_RULE`, `WALK_RULE`, `PASS_RULE` and others) are cited by name.
- **Controllers (rows).** R2 is Stage 1's balance tracker; `R2/42` and `R2/43` are its two fits (fit seeds 42 and 43).
  R0 is the panels' open-loop row: the reference's servo targets with no balance feedback. R0L is the plan's name for
  that open-loop row in Stage 2 (S2-3). `R2s/42@10ms` and `R2@10ms`, named in the list above, are Stage 1b's speed-fit
  rows on `60-N-T-10`; they are not the rows of B2.
- **Names with an F, an A, an R or an H.** F25 is the decision log's row F25, the tuned walk. The "F9" in the set names
  `F25F9` and `F25F9cl` is look feature F9, the counter-phased tail (`stage2/research/research_look.md` 3.1), not the
  decision log's row F9. A1 is the decision log's row A1, the sustained torque assumed for a drive; "shown but for A1"
  sets aside the neck and tail servos' sustained limit. This text's own rules are A1–A13, B1–B7, C1–C8, H1–H6 and R1–R5,
  each named at its head. Outside a rule's head, "A1" alone is the decision log's row, "H" alone is the hip height, and
  R0, R2, R5 and R6 are controllers (R5 is the plan's judged policy, R6 its policy-only ablation, and R2L the tracker
  refitted with delay in its mix; the last section names them). This text's rules of those names are then written
  "rule A1" and "rule R1" to "rule R5", and code and records that cite a rule of this text name the file with it
  ("STAGE2_75_PLAN rule R2").
- **Levers.** Lever 1 and lever 6 are two of the plan's hardware levers (plan 4 D): faster hip-pitch and knee drives,
  and more thigh-twist range.
- **Files.** sha256 is a file's hash. `r2` is the gait generator's accuracy level. BLAS is the linear-algebra library
  under numpy, held to one thread so that arrays reproduce bit for bit.
- **Tags.** Each rule carries at least one: **[plan]** as the plan states it (its section 5 unless another is named);
  **[S1]** Stage 1's rule, unchanged; **[NEW]** a rule this round adds; **[differs: plan]** or **[differs: S1]** where
  the rule departs, with the departure named. "Lead's ruling n" is ruling n that the study's lead gave on the design of
  these rules on 14 Nov 2026, before any run; the rulings are session notes, not in the repository, and each rule states
  its content in full. **[design]** marks a detail taken from that design (a session note of 14 Nov, written before any
  run) where the draft of these rules was silent; it is stated in full here. Constants marked "(prototype)" were set on
  the scratch prototypes tabled above (V0's stop and turn built in full; the candidate's turn at plan level and as
  kinematic passes), before any Stage 2 closed-loop run; none was set on a Stage 2 closed-loop result. Two also lean on
  committed Stage 1 records, listed after the table: A11's 0.85 (closed-loop records at 0 ms) and A7's 4.0 mm.
  **[settled 15 Nov, before any set was frozen]** stands directly before wording that the study's lead settled on
  15 Nov 2026, on one of the twelve points the independent review of this text had left for the lead; the wording so
  marked runs to the end of its paragraph or list item, or to that paragraph's "Check:". Those rulings are not the
  lead's rulings 1 to 5 of 14 Nov; they too are session notes, and the last section lists them by point. Every rule ends
  with "Check:", the record field or file a reviewer compares with it.

## A. Sets and references (task 2)

**A1. Sets.** [NEW] Three reference sets on `75-M-H-10`, each a directory `stage2/sets/<set>/75-M-H-10/` in Stage 1's
run-directory layout: `V0` (control: today's walk), `F25F9` (control: the PR's `_tuned_f9_spec`, verbatim), `F25F9cl`
(candidate: that spec with duty 0.59 and lift 0.095 H at every speed). Check: `set.json` `set`, `role`; each walk
entry's `definition.spec` equals `sets.SETS[set].spec_of(body, H, v)`; `tuned_f9_spec` equals the PR's function field by
field (a test imports the PR's script).

**A2. Speeds.** [plan; JETSON_PLEM_REVIEW task 2] Walks `nominal` (Froude 0.012: 0.2431 m/s) and `v0.3`, `v0.35`,
`v0.4`, `v0.45`, `v0.5`. V0: `sizing.NOMINAL` at the nominal speed; `look.speed_spec` between Stage 1's walks; above
`envelope.FAST`'s speed (0.4961 m/s) FAST's stride, duty and lift at the higher Froude number, labelled `new`. F25F9cl:
every rung other than 0.40, 0.45 and 0.50 m/s labelled `new`. Check: entries' `label`.

**A3. The candidate's margin was set after results.** [NEW label; JETSON_PLEM_REVIEW task 3] Duty 0.59 and lift 0.095 H
were chosen by the PR after its results on seeds 3042–3061. Every F25F9cl entry carries `margin_rule: "duty and lift set
after results on seeds 3042-3061 (JPS 2.3)"`, and every record of the set carries that label in its `rules`; rule H4
(fresh seeds) follows from it. Check: the entries' `margin_rule`; the records' `rules`.

**A4. One builder; Stage 1's transitions reproduce exactly.** [NEW; in place of the `transitions.generate` that
JETSON_PLEM_REVIEW task 2 names, which it reproduces exactly] Every start, change, stop and turn comes from
`designlab.stage2.moves`. For Stage 1's 18 transitions on both picks it regenerates each array file to the sha256 in
`stage1/runs/2026-11-08/<body>/transitions.json`. Check: `stage2/sets/legacy_check.json`: 36 of 36 `equal`,
`moves_sha256` = the sha256 of the `moves.py` that built the sets. V0's `refs/nominal.npz` and `transitions/start.npz`
hash to Stage 1's files; V0's look-format `nominal` and `v0.4` hash to the look ladder's V0 rungs (entries'
`equals_stage1`, `equals_ladder`).

**A5. Starts and speed changes are Stage 1's construction.** [S1] A start is stand to the set's `nominal`; a change is
`nominal_to_v<v>`, one stride of ramp, built with foreknowledge as Stage 1's (`transitions.generate`'s method); no bar
reading is tied to their command. Check: `move.causal: false`, `ramps`.

**A6. A commanded move is causal.** [differs: S1 (decision 18's convention); lead's ruling 2; constants (prototype)] For
a stop and a turn the reference is the periodic walk, frame for frame, up to the command; the preview controller first
sees the command at that frame; the file begins there. The command instant on a track is the move's first frame. A
command may only be given at a lift-off. Alternative named and not used: Stage 1's convention, command = start of the
reference's own ramp, under which the reference's CoM leaves the walk about half a second before the "command" (scratch
prototype: 0.58 s for the stop from 0.40 m/s). Decision 18's reading is kept in every stop record. Check: `move.causal:
true`, `move.command_phase` a multiple of 0.5, `join_command` (model CoM within 1.0 mm over the warm-up stride and at
the command frame, joints within 1e-3 rad at it, multibody ZMP within 2.0 mm up to two frames before it; the ZMP at the
command frame recorded, not gated).

**A7. Gates of every move.** [S1 for G3's limits; NEW for the rest; constants (prototype)] `ik_ok`; last stride against
the periodic walk (or the stand): CoM within 1.0 mm, multibody ZMP within 2.0 mm (G3); non-causal moves: first stride
the same, final ZMP error at most 4.0 mm; causal moves: `join_command` (A6), consistency error at most 2.5 mm, the
excused transient confined to the first stride after the command (it fades to zero over phases [command + 0.5, command +
1.0]), ZMP error after that stride at most 4.0 mm, single-support CoP at least 10 mm inside the pad. The consistency
error and both ZMP errors are read after the last correction pass; the number of passes and the gain are the walk's own
[design]. Check: `pass_`, `join_a` / `join_command`, `join_b`, `zmp_error_history_m`, `zmp_consistency_history_m`,
`post_window_zmp_error_m`, `cop_margin_single_min`, `move.rules`.

[changed 15 Nov after a result: the review's scratch builds of V0's stop and quick turn (prescribed motion); before any
set was frozen and before any closed-loop episode on a Stage 2 set] The two causal limits each read their own frames.
The consistency error is read over the frames before one stride after the command (phase < command + 1) and is at most
2.5 mm; the ZMP error over the frames from one stride after the command on is at most 4.0 mm. As first written the
2.5 mm limit was read over every output frame. After the fade those frames' consistency error is the ZMP error itself,
so the 4.0 mm limit could never fail and the frames after the transient were held tighter than Stage 1's own
transitions (4.0 mm; they end at 1.7 to 3.8 mm). Under the first wording V0's quick turn read 2.76 mm (2.04 mm inside
the first stride) and failed in both directions, while the candidate's timing passed at 2.03 mm; V0's stop read
1.62 mm. Neither constant moved. The all-frames value stays in every entry (`zmp_consistency_all_m`), gated by nothing.
Corrected in the same edit, with no limit moved: the reference's forward acceleration (the low-passed value that is
gated and the raw peak recorded beside it) is differentiated on the frames' own time grid, and the low-pass filter is
given the walk or the stand around the move as its history; the first build differentiated a resampled path, which
over-read the raw peak by a quarter to a third.

[changed 15 Nov after a result, by the owner's decision (`DECISION_LOG.md` P34): the first builds of the three sets, `stage2/design/checkpoint/SETS_FIRST_BUILD.md`; prescribed motion, before any set was frozen and before any closed-loop episode on a Stage 2 set] The ZMP readings of a causal move have one limit, Stage 1's 4.0 mm: the warm-up frames against the walk's
(`join_command`; it was 2.0 mm), the consistency error over the first stride after the command (it was 2.5 mm), and the
ZMP error after that stride (4.0 mm, unchanged). The CoM within 1.0 mm in the warm-up and at the command, and the joints
within 0.001 rad at the command, stand as written. Why: the two tighter limits were constants set on V0's prototypes at
V0's own timing. Across the three sets the warm-up ZMP read 1.35 mm at V0's own timing and 2.05 to 4.01 mm at other
walks and stride factors, while the warm-up CoM stayed within 0.09 mm of the walk's, so the reading follows the preview
controller's grid and not a difference in the motion before the command; the first-stride consistency read 1.24 to
3.51 mm. Under the tighter limits V0's quick turn, F25F9's two-touchdown stop and every turn built on the two tuned
sets had no reference. The readings stay in every entry as they were.

**A8. Fallbacks for starts, changes and stops.** [NEW] A move that fails A7 or a blocking gate of A12 is retried in this
order and no other: (1) a change with a ramp of 2 strides; (2) for a ladder rung, a change from the neighbouring rung
toward 0.40 m/s; (3) recorded "no transition within the gates". A start and the stop have no fallback, and
`nominal_to_v0.4` has step (1) only. A failed `start` or `nominal_to_v0.4` stops the set and goes to the lead. A failed
`stop2_v0.4` leaves the stop conditions "not run" with its numbers (H3), and a set whose `nominal` or `v0.4` walk is not
frozen (A12) has no lead-in, so its conditions are "not run" too [design: a condition that needs an absent reference
gets a "not run" record]. A gate or constant changed on such an account is a change after a result and is labelled so.
Check: entries' `fallback`, `available`.

[changed 15 Nov after a result, by the owner's decision (`DECISION_LOG.md` P34): the first builds of the three sets, `stage2/design/checkpoint/SETS_FIRST_BUILD.md`; prescribed motion, before any set was frozen and before any closed-loop episode on a Stage 2 set] A set whose `nominal` walk is not frozen but whose `v0.4` walk is starts from standing straight into its
0.40 m/s walk (`start_v0.4`), and its lead-in is that start followed by the 0.40 m/s walk; `set.json` and every record of
the set say so (`lead_in: "start_v0.4"`). This is an exception for such a set, labelled in every table that prints its
rows. If `start_v0.4` fails a gate the set is not run. It applies today to F25F9 alone: built as the pull request
specifies it, its walks at 0.35 m/s and below have a planned mid-swing pad clearance of 0.0382 H (19.2 mm) against
0.04 H (20.1 mm) and are dropped from its ladder, while its walks at 0.40 to 0.50 m/s are within every limit.

[made exact 15 Nov after a prescribed-motion result, the sets' independent review; before any set was frozen] Step (2)'s
change from the neighbouring rung is built with one stride of ramp, and step (1)'s two-stride ramp is not tried on it.
The text above did not say; the builder built it so before the result was known, and the other reading was not
tried. What it decides today: on F25F9 the changes `v0.4_to_v0.45` and `v0.45_to_v0.5` end with a ZMP error of 4.59
and 5.06 mm against A7's 4.0 mm, so its walks at 0.45 and 0.50 m/s are frozen but cannot be reached, and their ladder
rows are "not run" with those numbers. V0 and F25F9cl reach all six of their walks by the direct change.

**A9. The two-touchdown stop, `stop2_v0.4`.** [differs: plan in the command's place; lead's ruling 3] Stage 1's stop
plan (the left foot lifts last and lands beside the right; speed ramp over 0.4 stride about `phi_c = 0.5 - duty / 2`)
commanded causally at the right foot's lift-off, half a stride before the left foot's last lift-off. The reference has
exactly two touchdowns after the command, its forward speed is under 5 % of the walk's within 1.5 strides of the
command, and its 1 Hz low-passed forward acceleration stays within 1.5 m/s². Stated limit of that last gate: a
zero-phase order-2 Butterworth at 1 Hz reads at most 2.22 × dv, so a stop from 0.42 m/s cannot read above 0.93 m/s²; the
unfiltered peak is recorded beside it. `stop1_v0.4`, Stage 1's one-step stop on the same walk (not causal), is built as
a labelled comparison. Check: `reference.touchdowns_after_command == 2`, `reference.forward_stopped_strides <= 1.5`,
`reference.accel_forward_lp1hz_peak_m_s2`, `reference.accel_forward_raw_peak_m_s2`, `reference.stage1_stop`.

**A10. The quick turn, `turn90_pi4_left` and `turn90_pi4_right`.** [plan for the turn; NEW for its construction] One
causal move from `v0.4` to `v0.4` on the heading + 90° at π/4 rad/s: yaw rate and stride period ramp together from the
command, hold, ramp back; path speed, duty and lift stay the walk's; the command is the lift-off of the inside foot;
left and right are mirror plans.

- Ramps: **0.375 s of reference time, for every set, body and rate** [NEW; constant (prototype), set on plan-level
  arithmetic of V0 and of the candidate: a ramp of half a stride would last 0.387 s for V0 and 0.410 s for F25F9cl,
  which puts the candidate's reference at 79.7° at 2.0 s of its own clock (scratch prototype); by the same arithmetic
  0.375 s puts both references at or above 80.3° at the slowest clock. No turn has been built in full at 0.375 s].
  Check: `turn.ramp_s == 0.375`.
- Appendages: inside the turn the tail's and the neck's swing amplitudes are multiplied by `(T(p) / T_walk)^2` [NEW;
  lead's ruling 1; a look choice, for every set, look sets included]. Check: `turn.appendage_rule`, `move.rules` holds
  `S2A-appendage`.
- The reference itself: total heading 90° within 1e-9 rad; at the bar's 2.0 s on the nominal commanded clock (reference
  time 2.1 s) at least 80° [plan, G9]; at the slowest commanded clock (reference time 1.995 s) at least 80° [NEW
  margin]. Check: `reference.heading_at_bar_deg`, `reference.heading_at_slowest_clock_deg`, `turn.heading_total_rad`.

**A11. The stride rule.** [NEW; the plan's remedy "its reference takes quicker steps" made a rule; constants
(prototype)] Inside the turn the stride period is `f × T_walk` at the same path speed. `f` is the largest value of 1.00,
0.95, ..., 0.60 for which the finite turn's kinematic pass (no ZMP correction), in both directions, has `ik_ok` and uses
at most 0.85 of the usable thigh-twist range (±0.30 rad) on the worse leg. The reference built with all passes must show
`ik_ok` and at most 0.87; if not, `f` steps down once, at most twice in all. `f` is one value for both directions, and a
step down stays on the grid (none below 0.60). Outcomes:

- no `f` passes, or the built reference is over a leg drive's Stage 0 limit: no reference for that set and rate; the
  turn conditions get "not run" records with the numbers; it goes to the owner (lever 6 or lever 1). Nothing is designed
  around it [lead's ruling 4].
- over a neck or tail planning limit after the appendage rule (a sustained, peak or speed ratio above 1 at Stage 0;
  A12): frozen with `a1_over: true` and read under reading 3 only [lead's ruling 1]. Reading 2 therefore gives such a
  turn no pass. Reading 3 sets aside only the neck's and the tail's sustained term (C2): their speed is still judged in
  closed loop. [settled 15 Nov, before any set was frozen] A turn frozen with `a1_over`: its conditions run and both
  readings are recorded, but reading 2 gives them no verdict (the cell reads "not read: `a1_over`", which is not a
  pass); only reading 3 can pass.
- the built reference fails another gate (A7's; A10's total heading or either of its 80° readings; another blocking gate
  of A12): there is no further retry; it is written with `pass_: false` and not frozen [design], so there is no
  reference for that set and rate and the turn conditions get "not run" records with the numbers (H3). A ramp, grid or
  gate changed on that account is a change after a result and is labelled so.
  [settled 15 Nov, before any set was frozen] A turn left without a reference in this way: it goes to the owner. A turn
  has no other fallback: the ramp, the grid, the limits and the appendage rule are not changed for it except by an
  amendment labelled as made after that result.

The ladder's other rates from 0.40 m/s (0.35, 0.5, 1.0 rad/s) use the same move and rule. [noted 15 Nov,
before any set was frozen; plan-level arithmetic of the review, no rule changed] With the 0.375 s ramps the 1.0 rad/s
turn's heading at the slowest clock is 78.4° on V0 and 77.7° on the candidate's timing, below A10's 80°: by the third
outcome above it is built, written `pass_: false` with its numbers and not frozen, and the ladder reports that rate as
"no reference within the heading rule".

[changed 15 Nov after a result: the set screen's review read the design's scratch prototypes of V0's quick turn at
`f` 0.8 and found the inside leg's hip roll 0.086 rad from its limit, under A12's 0.10 rad, which by the third outcome
above would have left the turn without a reference; prescribed motion; before any set was frozen and before any
closed-loop episode on a Stage 2 set] The stride rule also reads A12's joint range: `f` is the largest grid value whose
kinematic pass, in both directions, has `ik_ok`, uses at most 0.85 of the thigh-twist range on the worse leg **and
leaves every other gated joint at least 0.10 rad of range**; the built reference must show `ik_ok`, at most 0.87 and
A12's 0.10 rad, and if not `f` steps down as above. No limit moved: A12's 0.10 rad stands on every joint but the twist,
and the quicker steps that were the plan's remedy for the twist are now also its remedy for the hip roll. The lead's
kinematic scan of V0's quick turn on this body (scratch) gives a hip-roll margin of 0.080, 0.091, 0.101, 0.111 and
0.121 rad at `f` 0.80, 0.75, 0.70, 0.65 and 0.60, the same in both directions to 0.0001 rad, with the twist's use
falling from 0.82 to 0.60. Expectation stated before the build, changing no rule: V0's quick turn is picked at `f`
0.70 instead of 0.80, so its strides inside the turn last 0.70 of the walk's and its drive speeds rise; A12's leg-drive
gate then decides whether it has a reference. Check: `stride_rules[rate]` holds, for every `f` tried and both
directions, the tightest joint other than the twist and its margin.

[changed 15 Nov after a result, by the owner's decision (`DECISION_LOG.md` P34): the first builds of the three sets, `stage2/design/checkpoint/SETS_FIRST_BUILD.md`; prescribed motion, before any set was frozen and before any closed-loop episode on a Stage 2 set] The amendment of the same morning is withdrawn as written: the stride rule does not read 0.10 rad. It reads A12's
joint range for a move, which is now 0.05 rad, and that binds on none of the three sets' turns at the factors the thigh
twist picks (V0 0.80, F25F9cl 0.65, F25F9 0.70 for the quick turn, with 0.072 to 0.086 rad of hip roll at the
kinematic pass). Why: with 0.10 rad V0's quick turn moved to factor 0.70, where it missed two gates, and the two tuned
sets had no factor at all (0.085 and 0.096 rad at factor 0.60). Check: `set.json`
`stride_rules[rate]` (every `f` tried with both directions' readings, the chosen `f`, `steps_down`), each turn's
`range_use`, `a1_over`, `not_available`; the report's reading 2 cell of a turn frozen with `a1_over`
("not read: `a1_over`").

[made exact 15 Nov after the sets' independent review; before any set was frozen; no built set is in this state] The
stride factor is one value for both directions, but after it is chosen each direction's built reference is judged on
its own gates. If one direction fails a gate under the third outcome and the other passes, the passing direction is
frozen and the failing one has no reference: its conditions are "not run" with the numbers, which is not a pass, and
it goes to the owner as above. On the three sets every rate has both directions alike.

**A12. Stage 0 screen of every reference, and what blocks a freeze.** [plan 4 A.3; NEW table] Every walk and move has a
block in the set's `screen.json` with every gate the look ladder's screen applies (drive ratios cont, peak, speed;
passive neck and tail loads; grip need at μ 0.9 and 0.54; sway; accelerations; the clearance gates; joint-range
margins). A reference is not frozen when: a leg drive's ratio exceeds 1; a neck or tail drive's ratio exceeds 1 on a
walk, start, change or stop (on a turn: A11); `ik_ok` is false; the single-support CoP rule fails; a planned swing's
mid-swing pad clearance is under 0.04 H; an interference, component-overlap or tail-clearance gate fails; a joint has
under 0.10 rad of range (a turn's thigh twist: A11's 0.87 instead); a start, change or stop exceeds 1.5 m/s² of 1 Hz
forward acceleration. Reported only: grip need, sway, passive loads, a walk's or turn's acceleration. A walk rung that
is not frozen is dropped from the ladder with its numbers. Check: each reference's block in `screen.json`:
`verdict.blocks_freeze`, `verdict.a1_over`; `set.json` `freezes[-1].manifest`.

[changed 15 Nov after a result, by the owner's decision (`DECISION_LOG.md` P34): the first builds of the three sets, `stage2/design/checkpoint/SETS_FIRST_BUILD.md`; prescribed motion, before any set was frozen and before any closed-loop episode on a Stage 2 set] The joint-range gate of a move (a start, change, stop or turn) is 0.05 rad on every joint but a turn's thigh
twist; a walk keeps 0.10 rad. Why: Stage 1's own transitions on this body come within 0.057 rad of a limit (10 of its
18 are under 0.10 rad) and ran in Stage 1's closed loop, where the joint range is judged against the reference's own
smallest margin less 0.02 rad (decision 14, unchanged in C4 and C5). With 0.10 rad F25F9cl's start (0.095 rad of hip
roll) was blocked, which stops the set (A8).

[noted 15 Nov after the sets' independent review; before any set was frozen; no gate or limit changed] Two readings of
a move's 1 Hz forward acceleration stand in the records. This rule's 1.5 m/s² gate reads the screen's own
(`screen.json` `accel.com_forward_lp1hz_peak`: the move's frames alone, the filter given no walk or stand around
them). A7's reading as amended is the builder's (`accel.com_forward_lp1hz_peak_entry`: the filter given the walk or
the stand around the move). On the three sets as built the larger of the two on any start, change or stop is
0.74 m/s² (F25F9's `stop2_v0.4`: 0.741 and 0.653), half the limit, so no verdict depends on which is read. The lead's
earlier ruling to make them one number in the three modules was not built before the freeze and is left for a later
code state.

**A13. Freezing.** [NEW] Panels run only on a frozen set. A freeze lists every entry's name, kind, file sha256 and
entry-subset sha256; a later freeze may add entries and never drops or changes one; a changed reference takes a new name
or a new set version. Arrays are built at accuracy r2 with one BLAS thread. [settled 15 Nov, before any set was frozen]
A reference added by a later freeze, a renamed reference or a new set version made after any episode of this round is a
change after a result: it is labelled so, with its reason, in `set.json` and in every record and table that uses it, and
its records replace no verdict of the registered set. Check: `freezes[]`; each entry's `built_by.accuracy == "r2"`,
`built_by.blas_threads == 1`; a bar record's `set_freeze.manifest_sha256`; the label "change after a result" and its
reason in `set.json` and in each record and table that uses such a reference.

## B. Episodes, rows, conditions

**B1. Episodes.** [plan; S1 `EPISODE_RULE`] 20 per condition and row; settled stance with Stage 1's start noise; nominal
plant; commanded 5 % above (the track played at 1.05 ×); hard floor. Seed blocks: `seeds3042` (3042–3061, the study's
seeds, the headline) and `seeds5042` (5042–5061, replication) [NEW block]. Check: records' `block`, `seeds`, `rules`.

**B2. Rows.** [JETSON_PLEM_REVIEW task 3; NEW names] `R2/42@<ms>ms` and `R2/43@<ms>ms`: Stage 1's two fitted tracker
controllers with the gains of `stage1/runs/2026-11-08/75-M-H-10/tune.json` unchanged; `R0@<ms>ms`: the panels' open-loop
row. `<ms>` is a servo delay of k = ms / 2 whole physics steps for the whole episode (Stage 1b's delay queue,
`speedfit.servo_delay`), k = 0..5 (0, 2, 4, 6, 8, 10 ms) [plan amendment S2-9: whole steps]. `R0@10ms` is R0L at the
judged delay. Check: `controller.gains` against `tune.json` and its sha256, `controller.latency_steps`.

**B3. Judged delay.** [plan S2-7; owner decision of 14 Nov, P30] 10 ms is judged; 0–8 ms are reported beside it; nothing
is read between delays. Task 3 (the walk) runs 0, 2, 4, 6, 8 and 10 ms; task 4 (the turn, the nudges and the stop) runs
0, 4, 6 and 10 ms (`JETSON_PLEM_REVIEW.md` task 4: "also 0 and 4 or 6 ms"); both as far as H6's budget reaches. Check:
records' `row` (`@10ms` for every verdict).

**B4. Floors.** [plan] Every gated condition of the bar at μ 0.9 (the floor as compiled) and at μ 0.54 (`_mu0.54`); the
ladders at μ 0.9, with `_mu0.54` only where H6 lists it. Check: records' `condition`, `definition`.

**B5. The track's LIPM frequency.** [differs: S1, which uses a 0.03 H crouch for every reference] `omega` is computed
with the set's own crouch (V0 0.03 H; the F25 sets 0.05 H on this body: about 1.2 %). The controllers' gains are not
refitted. One reported control runs with Stage 1's 0.03 H instead (H6, task 3, tier 3): 40 episodes in all, on
`seeds3042` [design: "on the same seeds"]. It gives no verdict and, being 40 episodes in all, is neither replicated nor
attributed; run in H6's order, it is a registered control and not a re-run in the sense of rule R2.
[settled 15 Nov, before any set was frozen] Its records live apart from the judged ones, under
`stage2/runs/2026-11-15-75cm/controls/omega_s1/`, and carry a `control` field that reads
"omega at Stage 1's 0.03 H crouch"; `omega` and `crouch_frac` enter the key of rule R1. Check: records' `omega`,
`crouch_frac`; the control's records' path and `control`; each record's `key` (rule R1).

**B6. The 16 gated conditions per set** [plan's table] and their plans [NEW]. Lead-in of every gated item: `start`, one
stride of `nominal`, `nominal_to_v0.4`, then `v0.4`.

- `walk/v0.4`, `walk/v0.4_mu0.54`: the lead-in, then `v0.4` for the fewest strides that hold the judged last 10 s
  (Stage 1's `WALK_RULE` plan).
- `turn/q90_left`, `turn/q90_right` (+ `_mu0.54`): the lead-in, 2 strides of `v0.4` (2.5 for the right turn), the turn,
  then `v0.4` to the end; the episode lasts 5.2 s after the command at the slowest clock.
- `nudge/forward`, `backward`, `left`, `right` (+ `_mu0.54`): the lead-in and `v0.4`; a 0.1 s pulse at the trunk's CoM
  of `m × 0.25 × 0.589 m/s / 0.1 s` in the commanded heading's frame, at 2 + u strides into the 0.40 m/s walk, u uniform
  in [0, 1) from the seed; the episode ends 5.1 s after the latest possible onset at the slowest clock (the 0.1 s pulse,
  3.0 s to recover, 2.0 s held) [design].
- `stop/two_step` (+ `_mu0.54`): the lead-in, 2.5 strides of `v0.4`, `stop2_v0.4`, then a hold of at least 2.0 s after
  the planned rest.

Reported conditions: `walk/nominal`, `walk/v0.3`, `walk/v0.35`, `walk/v0.45`, `walk/v0.5` (the speed ladder; `_mu0.54`
only where H6 lists it), `turn/t90_<0.35|0.5|1>_<dir>`, `nudge/<dir>_x0.375`, `_x0.5`, `stop/one_step`. A ladder rung
`walk/v<v>` has the walk item's plan with the change to that rung in its lead-in (`nominal_to_v<v>`, or A8's path
through 0.40 m/s) [design]. [settled 15 Nov, before any set was frozen] The plan of `walk/nominal` is `start`, then
`nominal` to the end, judged over its last 10 s. `stop/one_step` plays the non-causal `stop1_v0.4` (A9): it is read from
Stage 1's command (decision 18's convention), labelled so in its records. Check: records' `plan`, `seconds`,
`definition`; in `stop/one_step`'s records, the label of decision 18's command.

**B7. Shared lead-in and seed slot.** [differs: S1, which gives turns their own seed slots] Every item takes the walk's
seed slot, so every item of a seed meets the robot in the state the walk item judged. The 16 conditions are therefore
not independent: a fall in the lead-in shows in all eight items of that seed and grip. Check: episodes' `command_s`,
`fell_before_command`; records' `falls.before_command`, `falls.after_command`.

## C. Clocks and readings

**C1. Clocks.** [NEW precision of the plan's "2.0 s after the command"] Three clocks: the reference clock (track time);
simulated time; and the bar's clock, simulated seconds since the command instant `t_c`, the simulated time at which the
reference clock reaches the move's first frame. Every judged time (2.0 s, 3 s, 10 s) is in simulated seconds. A turn at
rate w is read at `at_s = (π/2) / |w|` after its command (2.0 s for the quick turn) and held 3 s [lead's ruling 5]. A
nudge's "command" is its onset. Check: episodes' `command_s`, `command_ref_s`.

**C2. Three readings, in the plan's order.** [plan for the readings; NEW for the headline] (1) *shown with the look*:
not defined for these three sets (they carry no look bands): the block is `null` with that reason. (2) *shown*: every
limit. (3) *shown but for A1*: every limit, with the sustained-drive term read on the leg drives only (every drive but
the neck's and the tail's). Each episode has `success` and `success_but_A1`; each record has counts, a one-sided 95 %
Clopper–Pearson lower bound (the exact binomial bound) and a pass for each. For these three sets reading 2 is the
headline, since the plan's headline (reading 1) is not defined for them, and reading 3 is printed beside it. The two are
never pooled. Expectation stated before the runs, from Stage 1's own records (simulation): in this body's walks at
0.351 m/s and at Froude 0.05 the sustained-drive term failed in 20 of 20 episodes under both controllers, with the tail
servo the worst drive (the records store only the worst drive), so reading 2 may fail everywhere for that one term. At
Froude 0.05 the duty term failed in 20 of 20 as well (`DECISION_LOG.md` F26), which reading 3 does not set aside. A
second expectation, from committed records (simulation): in Stage 1's margin ledger this body's nominal walk fell in
3 of 3 episodes at 10 ms of servo delay under `R2/43`, having met every limit up to 6 ms (`DECISION_LOG.md` F26 and
A17), and the controllers here are those fits, made with no delay; the PR's unrecorded table has the candidate falling
in every episode at 6 to 10 ms. The judged 10 ms conditions may therefore fail by falls on every set. Neither
expectation changes a rule. [settled 15 Nov, before any set was frozen] A turn frozen with `a1_over` (A11): its
conditions run and both readings are recorded, but reading 2 gives them no verdict (the cell reads
"not read: `a1_over`", which is not a pass); only reading 3 can pass. Check: `counts`, `counts_but_A1`, `counts_look`
(`null` with its reason), `measures.bar.drives`; the report's reading 2 cell of a turn frozen with `a1_over`.

**C3. Toe clearance over each foot's own swing.** [plan; NEW definition] Contact is Stage 1's (a foot's load above 2 %
of weight, 20 ms debounce, per 2 ms step). An *air run* of a foot runs from a lift-off step to the next touchdown step.
It is a **swing** if any 100 Hz update inside it falls in the reference's swing of that foot, whatever its length, or if
it lasts at least 0.5 × the shortest planned swing of the track (on the simulated clock: its reference seconds divided
by 1.05); otherwise it is an *air gap* (stance chatter), counted and not read [NEW: a plain rule that made every air run
under half a swing an air gap would let a touchdown at 48 % of the swing, or a mid-swing stub, leave the judged
reading]. A reference swing holding two or more swings is *interrupted*. Each swing wholly inside the window is read as
the lowest point of the foot (every foot geom above the floor, Stage 1's `low`) over the middle half of its own
duration: the value interpolated at 25 % and at 75 % of the swing and every update between. The term
`toe_clearance_own_swing` passes when the smallest reading is at least 0.04 H (20.07 mm), no reference swing in the
window is interrupted, and each foot has at least its number of reference swings less one. Stage 1's reading (on the
reference's clock) is recorded beside it, unchanged. The recorder's convention, that an update's time is the start of
the physics step just taken, is checked once on a real episode before the runs; if it is off by one physics step, the
lift-off and touchdown times shift by that step through one named constant, and the check's result is recorded [design].
Check: `measures.bar.toe_clearance_own_swing_m`, `own_swings`, `ref_swings`, `interrupted_swings`, `air_gaps`,
`air_gap_max_s`; the trace's `swings`, `contact`, `low100`.

[changed 15 Nov after a result: the review's reread of Stage 1's stock walks of this body under the reading above
(`walk/nominal`, `walk/brisk`, `walk/fr05`, seeds 3042–3044, hard floor, no servo delay; smoke on Stage 1's own run
directory, which R4 allows); before any set was frozen and before any closed-loop episode on a Stage 2 set] The middle
half is that of the **planned** swing, counted from the foot's own lift-off. A swing is read over
[`t_lo` + 0.25 `T_p`, `t_lo` + 0.75 `T_p`]: `t_lo` is its own lift-off, and `T_p` is the planned duration, on the
simulated clock, of the reference swing of that foot it overlaps the longest (the track's shortest planned swing if it
overlaps none). The interval is read to its end even if the foot has touched down before it, so an early touchdown
inside it fails the term; a swing whose interval runs past the window's end is neither read nor counted.

- Why: on these walks the 2 % load rule puts a foot's own touchdown 7 to 17 ms after the reference's (the foot rests
  within 1 mm of the floor for about 28 ms before its load crosses 2 %), which puts the 75 % edge of the own duration
  on the falling flank. As first written the reading was 0.010 to 0.017 H under Stage 1's on the same low points. It
  failed the `R0` row in six of six episodes (0.033 to 0.038 H) on walks Stage 1 passes (0.048 to 0.054 H); `R2/42`
  passed nine of nine with as little as 0.3 mm to spare.
- What stays: the limit (0.04 H), contact, the swing and air-gap rule and *interrupted*. The reading as first written
  stays in every record (`toe_clearance_own_duration_m`), reported and never judged; Stage 1's reading stays beside
  both.
- The count rule is made exact in the same edit: a reference swing counts toward a foot's number only if no window
  edge cuts the own swing that overlaps it. On synthetic traces the first wording failed a clean walk at 11 of 410
  window ends, where the window cut a foot's first and last swing.
- Expectation, stated before the runs and changing no rule: F25F9's planned reading at 0.40 m/s is 0.0405 H (half of
  its 0.081 H lift), 0.25 mm above the limit, so the control is expected to fail this term in most closed-loop episodes
  at 0.40 m/s. The candidate's planned reading is 0.0475 H.
  [corrected in place 15 Nov, after the sets' independent review and before any set was frozen: this sentence named V0
  where the numbers are F25F9's (V0's lift at 0.40 m/s is 0.11 H), and both numbers are Stage 0's mid-swing pad
  clearance (half the lift), not what this rule reads on the reference. The next item gives the references' own
  readings.]
- Expectation from the references themselves, stated before the runs and changing no rule [added 15 Nov after the
  sets' independent review; prescribed motion; before any set was frozen and before any closed-loop episode on a
  Stage 2 set]. The runner's kinematic check (`bar refcheck`: the reference played as if the robot tracked it exactly,
  no physics) reads this rule on each reference, against the limit of 0.04 H (20.07 mm):

  | Set | Walk at 0.40 m/s | Quick turn, left / right |
  |---|---|---|
  | V0 | 0.0503 H | 0.0456 / 0.0454 H |
  | F25F9cl | 0.0446 H | **0.0323 / 0.0316 H** |
  | F25F9 | **0.0393 H** | 0.0320 / 0.0313 H (not run in this round, H3) |

  So F25F9's walk item, and the candidate's four gated quick-turn conditions (left and right at μ 0.9 and 0.54), are
  expected to fail this term in every episode in which the robot tracks its reference, under readings 2 and 3, for a
  reason that is in the reference and not in the controller. On the candidate's turn the cause is the two swings that
  straddle the turn's 0.375 s ramps: the stride period changes inside the swing, so the middle half in time is not the
  middle half in phase (per swing on the left turn: the first swing after the command 0.0374 H at its 25 % edge, the
  steady-turn swings 0.0440 to 0.0458 H, the ramp-out swing 0.0313 H at its 75 % edge). A12's clearance gate, read by
  swing phase, gives 0.0484 H for that same turn and is not changed. On F25F9's walk the reading is every foot geom's
  over the time window, against Stage 0's pad reading of 0.0405 H at mid-swing.
  The lead's ruling: nothing is changed for it. No limit, reading, ramp, lift or stride factor moves; the sets are
  frozen with these references; each set's `refcheck.json` is committed beside it and cited by its freeze; and the
  report prints the reference's own reading beside the measured one wherever it prints these conditions. A change to
  how this rule reads a swing inside a ramp, to the lift inside a turn, or to the ramp would be a change after a
  result and is the owner's to decide after the panels (`DECISION_LOG.md` P35: the configuration is kept for now).
  Check: `stage2/sets/<set>/75-M-H-10/refcheck.json`; the set's latest freeze record.

The amended reading was built by the runner's fixer and checked by the lead against an implementation of his own,
written from this text, before any set was frozen: on 1,500 random synthetic traces (window cuts in 128, interrupted
swings in 929, air gaps in 969) the judged reading, the reading as first written, the swing and reference-swing counts,
the interrupted count, the air gaps and the count rule were equal in every one
(`stage2/design/lead_c3_check.py`, scratch). In the record the readings stand under `measures.bar.toe` for a walk or
a turn and under `measures.bar.toe_reported` for a nudge or a stop, which is where the Check line above is to be read.

**C4. The walk item.** [plan; lead's ruling 5 for the line] Window: the last 10 s of a complete episode. `along`: the
CoM covers at least 10 s × the rung's target speed along the commanded heading (4.0 m at 0.40 m/s [plan]; the same rule
at the ladder's other rungs is NEW). `line`: the CoM stays within 0.25 m of the line along the commanded heading through
its position at the window's start (the deviation from the line through the episode's start is reported). Other terms,
Stage 1's limits [S1]: survived; duty 0.55–0.65 (0.55–0.72 for a rung whose target speed is below 0.30 m/s, here
`walk/nominal` alone: owner decision of 14 Nov, P30; plan section 7, decision 2); no flight; roll and pitch 4° RMS and
8° peak; `toe_clearance_own_swing` (C3); slip 5 mm median and 15 mm worst; sustained drive RMS over the window at most
its rating (C2); drive speed; joint range 0.10 rad. Reported, not judged: peak ground reaction (Stage 1 drops it,
cadence and stride for every walk but the nominal one), heading drift. Stage 1's reading (`stage1`): `tune.measure`'s
terms restricted to the panels' list for that walk; it is printed for the comparison with JPS 2.3 and with Stage 1,
gives no verdict and triggers no replication (H2, H4). [settled 15 Nov, before any set was frozen] In `walk/nominal`
peak ground reaction, cadence and stride are judged as Stage 1 judges them. Check: `terms` (in `walk/nominal` with those
three), `measures.bar`, `measures.stage1`.

**C5. The turn item.** [plan; lead's ruling 5 for the start heading] `heading`: the trunk's heading is within 10° of the
commanded heading at the command + 90° at `t_c + at_s` and at every update for 3 s after. Time to within 10° and the
turn measured from the trunk's own heading at the command are reported. Over `[t_c, t_c + at_s + 3 s]`: the walk's
limits of C4 (without `along` and `line`), the joint range by decision 14 (at least the reference track's own smallest
margin less 0.02 rad, capped at 0.10 rad), and no leg interference (50 Hz poses on the screened model). Peak ground
reaction reported, as in the walk item [differs: S1, which judged it in its turns; those were from the nominal walk].
Check: `measures.bar.heading_error_at_deg`, `heading_error_hold_max_deg`, `time_to_within_10deg_s`, `measures.legs`.

[noted 15 Nov, before any set was frozen] The joint range and the leg gap are read over the window. The drives' peak
speed is read over the whole episode, because the recorder holds the episode's peak only; that is at least as strict as
the window, and the record says `drive_speed_span: "episode"`.

**C6. Nudges.** [plan; S1 decision 30] No fall; back on the command (0.9–1.1 × the commanded speed along the heading and
at most 0.1 × across, by stride means) within 3 s of the pulse's end, held to the planned end; drive and range limits
over the episode. Check: `measures.push`, `measures.recovery`.

**C7. The stop item.** [plan] `touchdowns_after_command`: at most two touchdowns (a foot's contact beginning) with
`t_c < t <= t_stop`, `t_stop` the first time from which the magnitude of the CoM's forward speed stays under 5 % of the
command (0.42 m/s) to the end of the trace; `stopped_by_end`; `stop_acceleration_1hz` at most 1.5 m/s² over two strides
from the command (A9's stated limit applies); drive and range limits over the episode. Reported: the stopped time in
seconds and strides (decision 18's quantity, now from a causal command), each counted touchdown's foot and time, the
last touchdown before the command and its distance to it (the left foot's scheduled landing is 50–89 ms of reference
time before the command, by set, which is 47–85 ms of simulated time at the commanded clock, so a late landing counts
and is visible), the unfiltered deceleration peak. Check: `measures.bar.touchdowns_after_command`, `touchdown_table`,
`last_touchdown_before_command`, `forward_stopped_s`.

**C8. An episode that cannot be read is an error.** [S1 `FAILED_EPISODE_RULE`, extended] An episode that raised, or that
did not fall and whose trace ends before its judged window, is run again; it is never a failed term. As Stage 1's rule
has it, the episode is recorded with its reason, the same seed is run again on the next run of the same key, and a
record that still holds such an episode is not judged (`pass_` null, verdict "incomplete") and is not a pass. No seed is
dropped or exchanged for another (B1). Check: episodes' `error`, `window_reached`; the record's `pass_`.

## H. Verdicts, replication, budget

**H1. Pass per record.** [S1 `PASS_RULE`] 20 of 20 up and at least 17 succeed, on the block's own 20 seeds, for each
reading; reading 2 of a turn frozen with `a1_over` is recorded and has no pass (H3). Check: `counts` and
`counts_but_A1`, each with its `pass_` (in `counts` of a turn frozen with `a1_over` the `pass_` is `null`, with the
reason "not read: `a1_over`").

**H2. Per controller.** [plan] `shown` if both `R2/42` and `R2/43` pass at that delay, `provisional` if one. The verdict
at 10 ms is the judged one. The word carries its reading: in reading 2 it is `shown` or `provisional`; in reading 3 it
is `shown but for A1` or `provisional but for A1` (the plan's name for that reading), never the bare word. No verdict
comes from an `R0` row, from Stage 1's reading (C4), from the replication block alone (H4) or from the omega control
(B5). A cell at another delay names its delay and is never the bar's verdict (B3).
[settled 15 Nov, before any set was frozen] At any other delay a cell gives the counts and `passes at <ms> ms` and never
either word. Check: the two rows' records at that delay; the block's `eval/summary.json`; the report's verdict cells and
its cells at the other delays.

**H3. The bar.** [plan] The 16 gated conditions at 10 ms; a condition without a reference is "not run" with its reason
and is not a pass. On F25F9, a control for the walk only, the 14 turn, nudge and stop conditions are not run in this
round (H6; the list "Not run in this round"). [settled 15 Nov, before any set was frozen] A turn frozen with `a1_over`
(A11): its conditions run and both readings are recorded, but reading 2 gives them no verdict (the cell reads
"not read: `a1_over`", which is not a pass); only reading 3 can pass. Check: the 16 conditions' records under each
`@10ms` row: `status` (`done` or `not run`), `reason`; the report's reading 2 cell of a turn frozen with `a1_over`.

**H4. Replication on fresh seeds.** [NEW; lead's ruling 5] Every `seeds3042` record of F25F9cl under an R2 row (a
condition, a delay) that passes in reading 2 or in reading 3 is run again on `seeds5042` with the same row, in H6's
order and as far as its budget reaches; the replication record is labelled with the reading that triggered it [design].
The headline stays `seeds3042`; the replication is printed beside it and never pooled, and a replication that H6 cut is
printed as cut. The controls are not run on `seeds5042`: the fresh block shows only whether the candidate's own pass
repeats, and the candidate is set beside a control on `seeds3042` alone. [settled 15 Nov, before any set was frozen] A
candidate's verdict is never printed alone: beside it stands "replicated" (the `seeds5042` record passes in the reading
that triggered it), "not replicated" (it fails) or "replication cut by the pre-registered order". A pass of an `R0` row
is not replicated: `R0` rows give no verdict (H2). Check: `block`, `replication_of`; the report's verdict cells: beside
each `shown` or `provisional` cell of the candidate's (in either reading), a replication word for each row's record that
passed; no `seeds5042` record under an `R0` row.

**H5. Fall attribution.** [S1 decision 20, with a NEW citation rule] On R2 rows' records with 5 or more falls: each
candidate limit lifted over the same 20 seeds, sign test at 0.05. A turn, nudge or stop record with fewer than 5 falls
after its command cites the walk record of the same set, row, grip and block. A label is a hint, not a proof.
Attribution runs where H6's order places it; a record whose attribution the budget cut reads "not attributed (budget,
pre-registered order)" [design]. Check: `attribution`, `falls`.

**H6. Budget and order.** [lead's budget; NEW ledger] 16 core-hours for task 3 and 12 for task 4, counted as the sum of
episode wall times in `budget.json`. What starts and what is cut is decided unit by unit, as the first paragraph after
the order settles it; this sentence was reworded with it on 15 Nov, before any set was frozen (on 14 Nov it read: "A
tier starts only if the spent time plus its estimate (from the committed `COST.md`, measured before the first tier)
fits; a started tier finishes; attribution is checked per record"). `COST.md` is measured before the first unit, a
started unit finishes, and whatever is not started is reported "cut by the pre-registered order". Every episode run for
a task counts, attribution's lifted episodes and C8's re-runs too; task 2's builds, screens and gates are in neither
budget. The run directory is `stage2/runs/2026-11-15-75cm/`, where `budget.json` and `COST.md` are committed [design].
The rows of each tier are those the cost estimate behind the budgets counted [design]. Order:

- Task 3: (1) `walk/v0.4` and `_mu0.54` at 10 ms on F25F9cl, F25F9, V0 under both R2 rows and R0, with attribution; (2)
  replication of (1); (3) the same at 0 ms (the conditions, sets and rows of (1); their attribution waits for tier 10),
  and at the end of this tier the omega control: F25F9cl `walk/v0.4` at μ 0.9 under `R2/42` at 10 ms and at 0 ms with
  Stage 1's 0.03 H crouch in `omega` (B5), 40 episodes, reported as a control; (4) 4 then 6 ms; (5) 2 then 8 ms; (6) the
  speed ladder at μ 0.9 on F25F9cl then V0 at 10 ms, then 0 ms, under both R2 rows; (7) the PR's remaining rows, under
  both R2 rows: F25F9 `walk/v0.5` at 0 ms, F25F9cl `walk/v0.45_mu0.54` and `walk/v0.5_mu0.54` at 0 ms, F25F9cl
  `walk/v0.5` at 2 ms; (8) replication of (3)–(7); (9) the ladder at 4 then 6 ms, under both R2 rows; (10) attribution
  of the remaining records, in the tiers' order.
- Task 4, on F25F9cl then V0, the 14 turn, nudge and stop conditions, under both R2 rows and R0: (1) 10 ms with
  attribution; (2) replication of (1); (3) 0 ms; (4) 4 ms; (5) 6 ms; (6) the reported ladders (turn rates, nudges at
  0.375 and 0.5 × the bound, `stop/one_step`) at 10 ms, μ 0.9, under both R2 rows; (7) replication of (3)–(5). The order
  holds no attribution after tier (1): the records of tiers (2)–(7) are not attributed in this round.

[settled 15 Nov, before any set was frozen] The order is run as a list of units: one set at one delay within a numbered
tier, sets and delays in the order written; the attribution of one record is a unit of its own. A unit starts only if
the task's spent time plus the unit's estimate fits; the estimate is the unit's planned simulated seconds (for a
replication or an attribution unit, those of the records that trigger it) times the rates of the committed `COST.md`.
The first unit that does not fit is cut with every unit after it: none is skipped to reach a cheaper one. `budget.json`
holds, per task, the units in the order started, each with its estimate, the core-hours spent before it, its start time
and the core-hours it used, and the list of cut units. `COST.md`'s rates are measured on Stage 1's own committed
conditions, not on any reference of the three sets.

[settled 15 Nov, before any set was frozen] The 16 + 12 core-hours come on top of `STAGE2_PLAN.md` section 6's estimate
for this CPU (about 80 core-hours), which was written before the 75 cm track existed; with the sets and screens (about
4 core-hours) the track adds about 32 core-hours, an estimate.

[settled 15 Nov, before any set was frozen] The omega control's records (task 3, tier 3; B5) live apart from the judged
ones, under `stage2/runs/2026-11-15-75cm/controls/omega_s1/`, and carry a `control` field that reads
"omega at Stage 1's 0.03 H crouch".

Check: `stage2/runs/2026-11-15-75cm/budget.json` (per task: the units in the order started, each with its estimate, the
core-hours spent before it, its start time and the core-hours it used; the list of cut units) and `COST.md` (its rates
and the Stage 1 conditions they were measured on); records' `sessions[].started_at` in the tiers' order; the report's
"cut" cells; the omega control's records' path, `control`, `omega`, `crouch_frac`. The chain's `done/` markers and logs
show the order on the machine that ran it; they are not committed and are no evidence for a reviewer.

## R. Records and code

**R1. Keys.** [S1 `RESUME_RULE`, narrowed] A record's key holds the row's description and sources, the condition, plan,
seconds, seeds, command, the body's plant inputs and library versions (`panels.key_context`), the sha256 of each array
file played and of the entry numbers the planner reads, and the source hashes of the panels' modules and of `bar.py`,
`readings.py`, `setio.py`, `speedfit.py`. [settled 15 Nov, before any set was frozen] `omega` and `crouch_frac` enter
the key of rule R1. Check: each record's `key`, with `omega` and `crouch_frac` in it.

**R2. No overwrite.** [differs: S1, whose `eval` reruns and overwrites on a key change] A finished record whose key
differs is never replaced silently: the run stops; `--supersede` keeps the old record beside the new.
[settled 15 Nov, before any set was frozen] The registered records are the first finished records under
`stage2/runs/2026-11-15-75cm/`. A record made later for the same set, row, condition, delay and block states its reason
in `supersedes.reason`, is labelled "re-run after a result" wherever it is printed, and the registered record's counts
and verdict stay printed beside it. Check: `<record>.superseded/` beside a superseded record; the later record's
`supersedes.reason`; the report's cells labelled "re-run after a result", each beside the registered record's counts and
verdict.

**R3. Traces.** [S1 `TRACE_RULE`, extended] Stage 1's 50 Hz trace plus the 500 Hz contact flags, both feet's 100 Hz low
points and the per-swing table, with its sha256 in the episode record. Traces are not committed, as in Stage 1: a
reviewer runs the episode again from its record's key and compares the trace's sha256 with the record's. Check:
episodes' `trace`; the trace's `contact`, `low100`, `swings`.

**R4. Code state A.** [NEW] The commit at which the three sets are frozen. Before it: the legacy check (A4) and the
closed-loop regression (`stage2/regress/regress.json`: committed Stage 1 and 1b episodes re-run from the code at that
commit reproduce their traces' sha256 and their episode records) both pass; the one Stage 1 edit of this round is in
(`control.Track.add_periodic` reads an optional `window_strides` key; without the key the path is unchanged); and the
Jetson body's code is in (`designlab/build.py`, `refine.py`, `power.py`, the Stage 1 registry). From state A to the end
of tasks 3–4 no module in a key changes; if one must, the affected baseline rows are run again into a new directory and
both are kept (S2-17); [settled 15 Nov, before any set was frozen] the state-A records stay the registered ones.

[recorded 15 Nov at the freeze; no rule changed] The three sets were frozen one per commit, because a freeze refuses a
tree that holds another set's uncommitted `set.json`: V0 at `f9be946`, F25F9 at `8eded6c`, F25F9cl at `2c9ecd0` (each
freeze's `commit` is the commit before the one that records it). **Code state A is `c377ba1`**, the commit that records
the third freeze. The four commits differ from one another only in the sets' `set.json`; no module, rule or other
record changes between them. Before it: the legacy check (36 of 36 at the moves.py of the sets) and the closed-loop
regression (256 of 256 episode runs, full tier, at `872cb12`; every commit from there to state A changes records,
the chain script, a citation file and these freezes, and no module in a key). Manifests: V0 `34a820e550b6…` (20
entries), F25F9 `1a5f606feed7…` (12), F25F9cl `cebd3d238b37…` (20). No flag was given to any freeze.

Episodes that the builders run to test the bar's code before task 3's first unit (the build's check episodes on set V0
under `R0` rows; the smoke of the R2 rows and of the servo-delay wrapper on Stage 1's own run directory; the check of
the recorder's convention in C3, within those limits; the rate measurement for `COST.md`, on Stage 1's own committed
conditions) are labelled smoke and leave no record: smoke output goes to scratch directories, and of the rate
measurement only the rates are kept. They count in neither budget and are reported nowhere as results, and no rule of
this text changes on their account except by a labelled amendment. The first of these two sentences was reworded on
15 Nov, before any set was frozen, so that it agrees with the settled wording that follows (on 14 Nov it began:
"Episodes that the builders run to test the bar's code before task 3's first tier (smoke runs on a built set, the check
of the recorder's convention in C3, the rate measurement for `COST.md`) are labelled smoke"). In the wording that
follows, "this commit" is the commit that adds this file (15 Nov 2026). [settled 15 Nov, before any set was frozen]
Between this commit and the first unit of task 3 the only closed-loop episodes run on a Stage 2 set are the build's
checks: single episodes on set V0 under `R0` rows. The R2 rows and the servo-delay wrapper are smoke-tested only on
Stage 1's own run directory (the read-only S1 view, whose conditions and results are already published in Stage 1's
records). No episode of F25F9 or F25F9cl and no R2 row on any of the three sets runs before task 3's first unit. Every
such check episode is listed in `stage2/runs/2026-11-15-75cm/dev_episodes.json` and is no record. `COST.md`'s rates are
measured on Stage 1's own committed conditions (H6). Check: `stage2/sets/legacy_check.json`,
`stage2/regress/regress.json`; `set.json` `freezes[].commit`; records' `key.sources`;
`stage2/runs/2026-11-15-75cm/dev_episodes.json`: every check episode, none of F25F9 or F25F9cl and none under an R2 row
on the three sets.

**R5. Reports.** [S1] Tables are machine-made from the records, cite record paths, keep the PR's numbers in columns
labelled as its claims, and never set a measured value against a target. A seed-by-seed table of task 3's records stands
beside JPS 2.3's cells (`JETSON_PLEM_REVIEW.md` task 3, "done when"); JPS's 0.496 m/s rows stand beside the 0.50 m/s
rung, and the table says so [design]. Check: `BAR_TABLES.md` and `bar_tables.json` regenerate from the records.

## Not run in this round, by name

The turn and nudges from the nominal walk, carpet walks, stiffness ×0.7, `robot_floor_carpet`; delays of 14, 16 and
20 ms; the turn, nudge and stop conditions at 2 and 8 ms (task 4 reads 10, 0, 4 and 6 ms; B3); the 14 turn, nudge and
stop conditions of F25F9, a control for the walk only (its stop and turn references are built and screened); `R0` rows
on the ladders and on the PR's remaining rows; Stage 1's twelve no-loss conditions (plan section 5); the look gates and
reading 1; 0.55 m/s; the Jetson-mass row of JPS 2.3; a stop from 0.50 m/s; any 60 cm body. Each appears in the report as
"not run", as does any other cell of JPS 2.3 with no counterpart in these runs.

## Settled by the lead on 15 Nov 2026, before the first frozen set

The independent review of this text left twelve points for the study's lead. The lead ruled on each on 15 Nov 2026,
before any reference set was frozen and before any Stage 2 closed-loop episode was run: they are rules fixed before
results. Each line below says what was decided and, after its last colon, in which rule the wording now stands: in this
text it is marked "[settled 15 Nov, before any set was frozen]"; in `STAGE2_PLAN.md` (points 11 and 12) section 9's
opening paragraph names it. The rulings themselves are session notes, not in the repository. A later answer is added as
an amendment with its date; one given after a result is labelled so.

1. Before task 3's first unit only the build's single check episodes on V0 under `R0` rows run, each listed: rule R4.
2. The budget is spent unit by unit; the first unit that does not fit is cut with every unit after it: H6.
3. The first finished records are the registered ones; what is made later is labelled so: rule R2, A13, rule R4.
4. The omega control's records live apart, and `omega` and `crouch_frac` enter the key: B5, H6, rule R1.
5. A turn frozen with `a1_over` runs; reading 2 gives it no verdict, and only reading 3 can pass: A11, C2, H3.
6. At any delay but 10 ms a cell gives the counts and `passes at <ms> ms`, never `shown` or `provisional`: H2.
7. A candidate's verdict is never printed without its replication status; a pass of an `R0` row is not replicated: H4.
8. A turn that fails a gate other than the stride rule's goes to the owner and has no other fallback: A11.
9. `stop/one_step` is read from Stage 1's command; `walk/nominal` is `start`, then `nominal` to the end, with peak
   ground reaction, cadence and stride judged as Stage 1 judges them: B6, C4.
10. The 16 + 12 core-hours come on top of the plan's estimate (about 80); the track adds about 32, an estimate: H6.
11. The rules that are the 75 cm round's own are named in the plan, and the judged controller's two seeds stand in place
    of `R2/42` and `R2/43`: `STAGE2_PLAN.md`, S2-16.
12. The plan's section 9 is reworded by the lead (the PR's runs, the delays read, a bench value off the 2 ms grid, the
    reading of S2-4's gain, G9's records, how the candidate is shown, what R5, R6 and R2L may edit): `STAGE2_PLAN.md`,
    the section's opening paragraph and its list of what had been run, S2-9, S2-10, S2-11, S2-15 and S2-17.

What else changed in this text on 15 Nov, also before any set was frozen. This section took the place of the list of
open points; the commit that adds this file holds that list, with the reviewers' proposed wording. The Status, the
brief, the Tags entry and the "Check:" lines follow the settled wording, and the clauses that had pointed to this
section as open are gone. One sentence of H2 moved within its paragraph, and rule R4 became two paragraphs with a gloss
on "this commit". Inside the marked wording a lead-in that names its subject, and a cross-reference to a rule of this
text, are the author's. Five rule sentences outside the marks were reworded or extended by the author so that the older
text agrees with the settled wording; they are not the lead's wording: H6's second sentence and rule R4's sentence on
the builders' episodes (each quotes its reading of 14 Nov), H1's clause on a turn frozen with `a1_over` (with the `null`
in its Check), H4's "under an R2 row" (with the replication words in its Check) and B5's clause on rule R2.

### Confirmed and added by the lead, 15 Nov 2026, before any set was frozen

The independent check of the settled text left the points below. Each is decided here, and the five author's sentences
named above are confirmed as rules. No reference set had been frozen and no Stage 2 closed-loop episode had been run.

1. H6: "a started unit finishes" is confirmed.
2. H1: reading 2 of a turn frozen with `a1_over` has `pass_: null`; it is no pass and is not replicated.
3. B5: the omega control is a control of its own (its key holds `omega` and `crouch_frac`); it is not a re-run under rule
   R2.
4. H4: a candidate's `shown` or `provisional` verdict prints with the replication status of each record that passed; a
   cell that is neither carries no replication word.
5. H5 and H6, where the attribution units stand: the attribution units of a tier that the order lists "with
   attribution" run directly after that tier's episode units, record by record in the order the records were made, and
   before the next tier starts; every other attribution unit of task 3 stands in its tier (10); task 4 attributes its
   tier (1) only. An attribution that is not run is reported "not attributed: cut by the pre-registered order".
6. H2 and the judged delay: "the judged delay" is 10 ms until a bench value exists. When one exists, the judged delay
   becomes the bench value's whole step (`STAGE2_PLAN.md` S2-9), the verdict words then apply at that delay, and the
   change is labelled with the bench record it rests on.
7. C1 and `stop/one_step`: the command instant is a move's first frame for every causal move; the one reported
   condition that plays a non-causal move, `stop/one_step`, is read from Stage 1's command (B6) and says so.
8. Rule R4, check episodes: a check is at most one episode (seed 3042) per condition and `R0` row on set `V0`; each is
   listed in `dev_episodes.json` with its reason. A13's "after any episode of this round" means a registered episode (a
   judged or reported record's); a freeze made after a check episode records `after_check_episodes: true` and the reason
   for the new freeze, and is not thereby "a change after a result".
9. `STAGE2_PLAN.md` S2-17's "the one Stage 1 edit of this round" means the one edit to a module that enters a record key
   (`control.py`); the plan's section 6 keeps its estimate of about 80 core-hours as written on 10 Nov, and the 75 cm
   track's about 32 core-hours come on top (H6).

## Amended on 15 Nov 2026 after the builds' independent reviews, before the first frozen set

Each is written into the rule it concerns, with its label. No closed-loop episode had run on a Stage 2 set.

1. A7 (changed after a prescribed-motion result): the consistency limit reads the first stride after the command, the
   4.0 mm limit the frames after it; the acceleration reading is differentiated on the frames' own grid.
2. A11 (noted, no change): the 1.0 rad/s ladder turn has no reference within the heading rule. A11 (changed after
   a prescribed-motion result): the stride rule also reads A12's 0.10 rad on every joint but the twist.
3. C3 (changed after a reread of Stage 1's stock walks): the middle half of the planned swing counted from the own
   lift-off; the count rule; the expectation for V0.
4. C5 (noted): the drives' peak speed is read over the whole episode.
5. Rule R4 and scratch sets: R4's limits protect the three registered sets. Closed-loop episodes of a Stage 2 move on a
   scratch or fixture set built from Stage 1's gaits are run only to test the runner, under `R0` rows and never under
   an `R2` row. Those run by the runner's builder and its two reviewers on 14–15 Nov are listed in summary form in
   `stage2/runs/2026-11-15-75cm/dev_episodes.json` under `scratch_sets`.
6. `control.Track.add_periodic` (R4's one Stage 1 edit): the stride period of a K-stride window is
   `(n // K) × step`, exact for every K, and a window whose frame count is no multiple of K is refused. Without the
   key the path is unchanged.
7. A12 (made exact, no limit moved): a gate whose reading is missing blocks the freeze (no single-support CoP reading,
   no joint-range margins, a rolling-contact split that did not solve); the appendage groups are the neck and the tail
   alone; a move's sustained ratio is the RMS over time, which on a walk's uniform frames is the same number.
8. By the owner's decision of 15 Nov (`DECISION_LOG.md` P34), after the first builds of the three sets
   (`stage2/design/checkpoint/SETS_FIRST_BUILD.md`): A7's ZMP readings of a causal move have one limit of 4.0 mm; A12's
   joint range for a move is 0.05 rad (a walk keeps 0.10); A11's stride rule reads that 0.05 rad, and item 2's change
   to it is withdrawn as written; A8: a set without a frozen `nominal` walk starts straight into its 0.40 m/s walk
   (F25F9). Each is labelled in its rule as changed after a result.
9. After the independent reviews of the sets as built under item 8 (prescribed motion; no set frozen, no closed-loop
   episode on a Stage 2 set), no limit or reading changed: C3's expectation sentence corrected in place (it named V0
   for F25F9's numbers) and the references' own C3 readings stated as an expectation before the runs (F25F9's walk
   0.0393 H, the candidate's quick turn 0.0323 / 0.0316 H, against 0.04 H); A8 made exact (step (2)'s change has one
   stride of ramp); A11 made exact (each direction of a turn is judged on its own built reference); A12 noted (two
   acceleration readings in the records, both far inside the limit).
10. After the independent review of the sets' fix round (two reviewers; before the freeze; prescribed motion only), no
   rule changed. (a) The chain's reference-check step: `bar refcheck` ends with exit 1 when a reference fails a term
   of its own item, which item 9 states as an expectation; the step now passes when the check's record equals, item
   for item and input for input, the one committed beside the set and cited by its freeze (`run_75cm.sh`, in no
   record's key). (b) The runner's `summary` output is its log: its row lines carry a set's lead-in label and its
   verdict lines do not. The tables of A8's "every table that prints its rows" are `barreport`'s, which label every
   line of such a set. (c) C3's "the report prints the reference's own reading beside the measured one" binds the
   tables too: `barreport` (in no record's key) reads the cited `refcheck.json` for it before the tables are made.
   (d) Left open, in no key and not needed today: a later freeze after `moves.py` has changed (the tie to the legacy
   check's `moves.py` should read the entries a freeze adds); a freeze does not yet check which `bar.py` made the
   reference check it cites, nor that the screen was made by today's `setscreen.py` (both hold on the three sets as
   frozen: checked by the reviewers).
11. [noted 15 Nov after a result: the chain's own walk records of the three frozen sets (tasks 3 and 4 still running);
   no rule, limit or reading changed] (a) **The candidate's walk item cannot pass as registered, whatever the
   controller.** C4 judges the joint range at 0.10 rad, and the runner reads it over the whole episode
   (`bar.py`, the walk item: `panels._limits_ok`, Stage 1's span). Every episode starts with the set's `start`, and
   F25F9cl's `start` comes within 0.095 rad of the hip-roll limit (the number P34's change of the move's *build* gate
   already names). Every surviving episode of every F25F9cl record so far reads 0.0950 to 0.0952 rad, under all
   three rows and at every delay, so its term `joint_range` fails 20 of 20 before the controller matters. This was
   not stated as an expectation before the runs; it should have been. The registered verdicts stand as the records
   hold them. The report also prints, under this label and as no verdict, the same records' counts with that one
   term set aside, so that the reader can see what else fails (V0's and F25F9's starts keep 0.127 and 0.136 rad and
   are not touched by this). Whether a later round judges a walk's joint range over its window (as C5's note does
   for a turn) or rebuilds the candidate's start is a rule change or a new set, for a later freeze and the owner.
   (b) **The tables' fix round** (`barreport`, in no key): the reviewer's finding 6 asked whether C3's "wherever it
   prints these conditions" needs more than the reference's column in the measured and failed-terms parts. Ruling:
   the column there, plus the label now on every line that prints a controller's verdict on such a condition, is
   what C3 asks; no verdict word is changed (that would change a rule).
12. [noted 16 Nov after the independent review of the report on these runs (`STAGE2_75_REPORT.md`); no rule, limit or
   reading changed; the lead's error] **A12's amendment of 15 Nov contradicts C4.** The amendment made for P34 gives
   as its reason that in the closed loop "the joint range is judged against the reference's own smallest margin less
   0.02 rad (decision 14, unchanged in C4 and C5)". That is true of C5 (the turn item) and false of C4 (the walk
   item): C4 says "joint range 0.10 rad", and the runner at code state A applies 0.10 rad over the whole episode to
   walks, nudges and stops and decision 14 to turns only (`bar.py`, `required`; `panels.LIMITS_RULE`). The owner's
   decision P34 was therefore taken on a sentence that was not true of the walk item. The registered counts follow
   C4 and the runner. By the amendment's sentence the candidate's limit would have been 0.075 rad against the
   0.0950 rad read, and the term would not have failed: that is arithmetic on the rule, not a record. Item 11(a)
   stands as written; this item adds why the expectation was missed. What to do about it is the owner's question
   (the report, section 3, question 3).
13. [noted 16 Nov 2026, after the panels and their add-on; no rule, limit or record of this round changes] The sentence
   near the top of this file, that until the first unit of task 3 "the only closed-loop episodes run on a Stage 2 set
   are the build's checks" (the development episodes of rule R4), names the 75 cm round's three sets: V0, F25F9 and
   F25F9cl. That is how the Track B rules draft reads it (its rule X1); the look sets are limited by their own gates.
   Task 3's first unit ran on 15 Nov 2026 and tasks 3 and 4 ended that day, so the sentence restricts nothing any
   more. On 16 Nov 2026 the lead said so for the training env's fix round: the env was compared with the bar's
   runner on the frozen set V0 under `R0` rows, as a check that leaves no record
   (`stage2/design/checkpoint/STATUS.md`, the update of 03:02 UTC; `stage2/design/TRACKB_RULINGS3_16NOV.md`, ruling 10).
