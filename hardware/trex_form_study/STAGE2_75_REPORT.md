# The 75 cm build in simulation: the first panels of the agility bar

15 Nov 2026. Body `75-M-H-10`. **Simulation only: nothing in this report was built or measured on hardware.** Nothing
is adopted by it (`DECISION_LOG.md` P24). Section 10 lists what is for the owner to decide, with draft rows.

**Status: draft of 15 Nov, 23:20 UTC, under independent review. The turning section (section 8 and item 7 of the answer) is
added when the reviewed turning figures are merged.**

Paths are relative to `hardware/trex_form_study/`.

**Four kinds of statement appear here and are kept apart.**

- **Simulated:** the robot model walking under a controller in the physics simulation. Every count and every measured
  median below is of this kind, read from the committed records under `stage2/runs/2026-11-15-75cm/`.
- **Planned:** a reference motion played without physics. The turning circle's planned numbers and "the reference's
  own toe clearance" are of this kind.
- **Design estimate:** arithmetic on catalogue facts, such as the control delay of a wiring layout. Not simulated and
  not measured.
- **Measured on hardware:** none exists yet.

**Where the numbers come from.** The rules were written before the runs: `STAGE2_75_PLAN.md`, code state `c377ba1`
(committed 19:42 UTC; the chain started at 19:44 UTC and ended at 23:09 UTC). The machine-made tables, with every
record path and every label, are `stage2/runs/2026-11-15-75cm/BAR_TABLES.md`. The tables in this report are copied by
script from a short digest of the same records, `stage2/runs/2026-11-15-75cm/digest/DIGEST.md`. Where anything here
differs from the records, the records decide.

## 1. The answer

1. **At the judged servo delay of 10 ms nothing is shown.** None of the bar's 16 conditions passes under either fitted
   balance controller, on any of the three walks, in either reading. The reason is the same everywhere: at 10 ms the
   robot falls while it is still walking straight. Under the fitted controllers 0 to 3 of 20 episodes stay up in every
   10 ms record of the bar's 16 conditions, and in the turn, push and stop records almost every fall comes before the
   command is given. At 8 ms, 0 of 20 stay up in all twelve walk records. The plan stated falls at 10 ms as an
   expectation before the runs (section 3).
2. **Under these controllers the walks hold to about 4 to 6 ms of delay.** At 0.40 m/s all three walks stay up 20 of
   20 at 0, 2 and 4 ms. At 6 ms today's walk still stays up 20 of 20, the pull request's walk 11 to 19 of 20, and the
   candidate 0 of 20.
3. **Today's walk (V0) is the only one with passes at any delay.**
   - At 0.24 and 0.30 m/s with no delay it meets every limit in 20 of 20 episodes under both controllers.
   - At 0.35 and 0.40 m/s it meets every limit but one, the tail servo's sustained load, which reads 1.00 to 1.05
     times its rating at 0.35 m/s and 1.02 to 1.08 at 0.40 m/s. With that one limit set aside it passes at 0 and
     2 ms at 0.40 m/s, and at 0 and 4 ms at 0.35 m/s (2 ms was not run at 0.35 m/s).
   - Its two-step stop meets every limit in 20 of 20 at 0, 4 and 6 ms, on both floors.
4. **The candidate walk (F25F9cl) passes nothing as registered.**
   - One term fails in every episode before the controller matters: its start-up move comes within 0.095 rad of the
     hip-roll limit, and the walk rule asks for 0.10 rad over the whole episode. This was not stated as an expectation
     before the runs and should have been (section 6).
   - Apart from that, its toe clearance is at the limit with no delay (20.0 and 20.6 mm against 20.07 mm) and under
     it with any delay (17 mm at 2 ms, 13 to 14 mm at 4 ms), and it falls in every episode from 6 ms at 0.40 m/s.
   - It does cure today's walk's failing term: its tail servo load reads 0.69 to 0.70 times the rating.
5. **The pull request's walk (F25F9) passes nothing.** Its toe clearance is under the limit at every delay (17 to
   18 mm with no delay), which the plan expected because the reference's own clearance, played without physics, is
   19.74 mm. Its 0.45 and 0.50 m/s walks could not be reached under this round's rules and were not run.
6. **The pull request's own numbers for the candidate reproduce under its own reading.** Read as the pull request
   read its results (Stage 1's reading), and with the start-up term set aside, the candidate meets every limit in 20
   of 20 at 0 and 2 ms at 0.40 and 0.50 m/s, as claimed. That is a labelled re-reading made after the result, not a
   verdict (section 6). It fails here mainly because this study's toe-clearance reading is stricter.
7. **Turning room.** [to be added when the reviewed views are merged: TURN_ANSWER]
8. **Compute.** The walk panels used 4.25 of their 16 core-hours and the turn, push and stop panels 5.51 of their 12.
   Nothing was cut. 592 records were written, 10 of them by the runner as "not run" with a reason.

## 2. What was run

**Three walks on one body.** Each is a frozen set of reference motions for `75-M-H-10`.

| Set | What it is | Role |
|---|---|---|
| `V0` | Today's walk (Stage 1's reference), with a 0.24 to 0.50 m/s speed ladder | control |
| `F25F9` | The walk of pull request #668, as its code builds it | control, for the walk only |
| `F25F9cl` | That walk with duty 0.59 and lift 0.095 H at every speed | candidate; its margin was set after results on these seeds |

**Three controller rows.** `R2/42` and `R2/43` are the same balance controller fitted twice (seeds 42 and 43) in
Stage 1, with no servo delay in the fit. A verdict needs both. `R0` has no balance feedback; it is reported and gives
no verdict.

**Servo delay.** The time between the controller's command and the change at the joint, in the model. The bar is
judged at 10 ms because no bench value exists (P30, P32). 0, 2, 4, 6 and 8 ms are reported beside it for the walk at
0.40 m/s; the speed ladder and the turns, pushes and stop are read at 0, 4, 6 and 10 ms.

**Floors.** Every condition of the bar runs on a grippy hard floor (friction 0.9) and a slippery one (0.54). The
speed ladder runs on the grippy floor.

**Pass rule.** A record is 20 episodes on seeds 3042 to 3061. It passes when all 20 stay up and at least 17 meet
every limit. Each episode is read twice: against every limit, and against every limit but the neck and tail servos'
sustained load (called A1 in the study). A condition is "shown" when both fitted rows pass at 10 ms.

**The bar's 16 conditions:** the walk at 0.40 m/s, a 90° turn to each side at π/4 rad/s (the quick turn), a push
from each of four sides (0.25 of the body's push bound, about 22 N for 0.1 s), and a two-step stop; each on both
floors.

**Budget.** 16 core-hours for the walk and ladder (task 3) and 12 for the turns, pushes and stop (task 4), fixed
before the runs, spent in a fixed order.

## 3. The walk at 0.40 m/s against servo delay

Cells: **episodes that stayed up / met every limit / met every limit but for A1**, each of 20.

Grippy floor (friction 0.9):

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

Slippery floor (friction 0.54):

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

What the table shows (simulated):

- **10 ms and 8 ms.** Under both fitted controllers every walk falls: 0 of 20 stay up at 8 ms in all twelve records,
  and 0 to 3 of 20 at 10 ms. With no balance feedback (`R0`) the same walks stay up 18 to 20 of 20 at every delay
  and meet no limit set.
- **Expected before the runs.** `STAGE2_75_PLAN.md` said, before any set was frozen, that "the 10 ms verdicts may be
  'not shown' because the robot falls, whatever the walk", from Stage 1's records of this body and from the pull
  request's own table. It also said the every-limit reading might fail everywhere on V0 for the tail servo alone.
  Both came true.
- **Where each walk stops holding.** Today's walk stays up through 6 ms and loses its toe clearance there. The pull
  request's walk begins to fall at 6 ms. The candidate falls in every episode at 6 ms.
- **The two floors give the same picture for the walk.** The largest difference between a cell above and its twin
  on the other floor is 5 episodes (V0, `R2/42`, 4 ms, but for A1: 16 against 11).

Measured medians on the grippy floor, fitted rows, 0 to 6 ms. Speed is the simulated median over the last 10 s (the
commanded speed is 0.40 m/s). Toe clearance is read as the bar judges it; its limit is 20.07 mm. Duty is the mean of
the two feet; its limits are 0.55 to 0.65. The servo columns are sustained load over rating; the limit is 1.0.

| Set | Row | Delay | Speed (m/s) | Toe clearance (mm) | Duty | Tail or neck servo ratio | Leg servo ratio | Failed terms (episodes of 20) |
|---|---|---|---|---|---|---|---|---|
| V0 | `R2/42` | 0 ms | 0.410 (20) | 24.2 (20) | 0.563 | 1.07 | 0.80 | `drive_sustained` 20, `line` 1 |
| V0 | `R2/42` | 2 ms | 0.412 (20) | 22.9 (20) | 0.573 | 1.05 | 0.80 | `drive_sustained` 20 |
| V0 | `R2/42` | 4 ms | 0.412 (20) | 20.7 (20) | 0.583 | 1.04 | 0.79 | `drive_sustained` 20, `toe_clearance_own_swing` 4 |
| V0 | `R2/42` | 6 ms | 0.409 (20) | 18.9 (20) | 0.583 | 1.02 | 0.79 | `drive_sustained` 20, `toe_clearance_own_swing` 20 |
| V0 | `R2/43` | 0 ms | 0.411 (20) | 24.2 (20) | 0.560 | 1.08 | 0.80 | `drive_sustained` 20, `line` 1 |
| V0 | `R2/43` | 2 ms | 0.413 (20) | 23.1 (20) | 0.571 | 1.06 | 0.80 | `drive_sustained` 20 |
| V0 | `R2/43` | 4 ms | 0.414 (20) | 20.2 (20) | 0.584 | 1.05 | 0.79 | `drive_sustained` 20, `toe_clearance_own_swing` 7, `line` 1 |
| V0 | `R2/43` | 6 ms | 0.413 (20) | 18.3 (20) | 0.582 | 1.03 | 0.80 | `drive_sustained` 20, `toe_clearance_own_swing` 20 |
| F25F9 | `R2/42` | 0 ms | 0.413 (20) | 17.2 (20) | 0.559 | 0.69 | 0.75 | `duty` 20, `toe_clearance_own_swing` 20, `line` 1 |
| F25F9 | `R2/42` | 2 ms | 0.413 (20) | 14.6 (20) | 0.573 | 0.69 | 0.75 | `toe_clearance_own_swing` 20, `duty` 1, `line` 1 |
| F25F9 | `R2/42` | 4 ms | 0.412 (20) | 12.7 (20) | 0.581 | 0.69 | 0.75 | `toe_clearance_own_swing` 20, `line` 1 |
| F25F9 | `R2/42` | 6 ms | 0.410 (15) | 12.6 (15) | 0.575 | 0.69 | 0.76 | `toe_clearance_own_swing` 15, `survived` 5, `line` 1 |
| F25F9 | `R2/43` | 0 ms | 0.412 (20) | 17.8 (20) | 0.551 | 0.69 | 0.76 | `duty` 20, `toe_clearance_own_swing` 20, `line` 1 |
| F25F9 | `R2/43` | 2 ms | 0.415 (20) | 15.0 (20) | 0.567 | 0.69 | 0.75 | `toe_clearance_own_swing` 20, `duty` 18, `line` 1 |
| F25F9 | `R2/43` | 4 ms | 0.415 (20) | 12.0 (20) | 0.579 | 0.69 | 0.75 | `toe_clearance_own_swing` 20, `line` 1 |
| F25F9 | `R2/43` | 6 ms | 0.415 (19) | 11.8 (19) | 0.574 | 0.70 | 0.76 | `toe_clearance_own_swing` 19, `duty` 4, `line` 1, `survived` 1 |
| F25F9cl | `R2/42` | 0 ms | 0.415 (20) | 20.0 (20) | 0.593 | 0.69 | 0.76 | `joint_range` 20, `toe_clearance_own_swing` 11, `line` 1 |
| F25F9cl | `R2/42` | 2 ms | 0.416 (20) | 17.0 (20) | 0.607 | 0.69 | 0.75 | `joint_range` 20, `toe_clearance_own_swing` 20 |
| F25F9cl | `R2/42` | 4 ms | 0.413 (20) | 13.7 (20) | 0.618 | 0.70 | 0.76 | `joint_range` 20, `toe_clearance_own_swing` 20, `line` 1 |
| F25F9cl | `R2/42` | 6 ms | - (0) | - (0) | - | - | - | `survived` 20 |
| F25F9cl | `R2/43` | 0 ms | 0.414 (20) | 20.6 (20) | 0.587 | 0.70 | 0.77 | `joint_range` 20, `toe_clearance_own_swing` 2 |
| F25F9cl | `R2/43` | 2 ms | 0.417 (20) | 16.8 (20) | 0.605 | 0.69 | 0.76 | `joint_range` 20, `toe_clearance_own_swing` 20, `line` 1 |
| F25F9cl | `R2/43` | 4 ms | 0.416 (20) | 13.0 (20) | 0.620 | 0.70 | 0.76 | `joint_range` 20, `toe_clearance_own_swing` 20, `line` 1 |
| F25F9cl | `R2/43` | 6 ms | - (0) | - (0) | - | - | - | `survived` 20 |

- **Toe clearance falls with delay on every walk**: on V0 from 24.2 mm with no delay to 18.3 to 18.9 mm at 6 ms; on
  F25F9 from 17.2 to 17.8 mm to 11.8 to 12.6 mm at 6 ms; on the candidate from 20.0 to 20.6 mm to 13.0 to 13.7 mm
  at 4 ms.
- **The tail servo is over its rating only on today's walk** (1.02 to 1.08 at 0.40 m/s). On both of the pull
  request's walks it reads 0.69 to 0.70.
- **Why the fall at 10 ms** (a hint from the runner's fall attribution, not a proof): on V0 and F25F9 the label is
  "controller-limited". On the candidate it is "body-limited: drive torque", meaning that lifting the leg drives'
  torque limit over the same 20 starts lengthened the time to the fall significantly. In the four 10 ms walk
  records of each set the candidate falls after a median of 3.1 to 3.2 strides (4.5 to 6.7 s into the episode),
  F25F9 after 3.7 to 3.8 strides and V0 after 4.2 to 7.4 strides.

## 4. Speed

The ladder on the grippy floor, fitted rows. `nominal` is each set's slow walk (0.243 m/s on V0 and the candidate).
Cells as in section 3.

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
| F25F9cl | `R2/42` | 0 ms | 20 / 0 / 0 | 20 / 0 / 0 | 20 / 0 / 0 | 20 / 0 / 0 | 20 / 0 / 0 | 20 / 0 / 0 |
| F25F9cl | `R2/43` | 0 ms | 20 / 0 / 0 | 20 / 0 / 0 | 20 / 0 / 0 | 20 / 0 / 0 | 20 / 0 / 0 | 20 / 0 / 0 |
| F25F9cl | `R2/42` | 4 ms | 20 / 0 / 0 | 20 / 0 / 0 | 20 / 0 / 0 | 20 / 0 / 0 | 20 / 0 / 0 | 20 / 0 / 0 |
| F25F9cl | `R2/43` | 4 ms | 20 / 0 / 0 | 20 / 0 / 0 | 20 / 0 / 0 | 20 / 0 / 0 | 20 / 0 / 0 | 20 / 0 / 0 |
| F25F9cl | `R2/42` | 6 ms | 20 / 0 / 0 | 19 / 0 / 0 | 14 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| F25F9cl | `R2/43` | 6 ms | 19 / 0 / 0 | 17 / 0 / 0 | 18 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| F25F9cl | `R2/42` | 10 ms | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| F25F9cl | `R2/43` | 10 ms | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |

- **Today's walk, no delay:** every limit is met in 20 of 20 at 0.24 and 0.30 m/s. At 0.35 and 0.40 m/s the only
  failing term is the tail servo (19 of 20 pass with it set aside). At 0.45 and 0.50 m/s the duty term fails as well.
- **Today's walk, 4 ms:** at 0.30 m/s 17 and 12 of 20 meet every limit, so one controller passes and one does not.
  With the tail servo set aside, 0.35 m/s passes under both (20 and 18).
- **Today's walk, 6 ms:** it stays up 20 of 20 from 0.24 to 0.40 m/s and meets no limit set: the toe clearance is
  under the limit in 20 of 20.
- **The candidate** stays up 20 of 20 at every speed at 0 and 4 ms and passes nothing as registered (section 6). At
  6 ms it stays up at 0.24 to 0.35 m/s (14 to 20 of 20) and falls in every episode at 0.40 m/s and faster.
- **At 10 ms** no rung of either set stays up in more than 2 of 20.

## 5. Turns, pushes and the stop

Each episode walks at 0.40 m/s first; the command or the push comes after about 13 s. Grippy floor; cells as in
section 3, with the falls split into before and after the command (b / a). The slippery floor, the reported turn
rates and the stronger pushes are in the digest and in `BAR_TABLES.md`.

| Set | Condition | Row | 0 ms | 4 ms | 6 ms | 10 ms |
|---|---|---|---|---|---|---|
| V0 | `nudge/backward` | `R2/42` | 20 / 13 / 13 (0 b / 0 a) | 14 / 13 / 13 (0 b / 6 a) | 14 / 10 / 10 (0 b / 6 a) | 0 / 0 / 0 (20 b / 0 a) |
| V0 | `nudge/backward` | `R2/43` | 20 / 13 / 13 (0 b / 0 a) | 12 / 10 / 10 (0 b / 8 a) | 9 / 8 / 8 (0 b / 11 a) | 1 / 0 / 0 (19 b / 0 a) |
| V0 | `nudge/backward` | `R0` | 9 / 1 / 4 (0 b / 11 a) | 8 / 2 / 2 (0 b / 12 a) | 10 / 1 / 3 (0 b / 10 a) | 10 / 3 / 4 (0 b / 10 a) |
| V0 | `nudge/forward` | `R2/42` | 19 / 14 / 14 (0 b / 1 a) | 20 / 19 / 19 (0 b / 0 a) | 19 / 19 / 19 (0 b / 1 a) | 0 / 0 / 0 (20 b / 0 a) |
| V0 | `nudge/forward` | `R2/43` | 20 / 14 / 14 (0 b / 0 a) | 14 / 12 / 12 (0 b / 6 a) | 15 / 13 / 13 (0 b / 5 a) | 0 / 0 / 0 (19 b / 1 a) |
| V0 | `nudge/forward` | `R0` | 3 / 0 / 0 (0 b / 17 a) | 1 / 0 / 0 (0 b / 19 a) | 2 / 1 / 1 (0 b / 18 a) | 2 / 0 / 0 (0 b / 18 a) |
| V0 | `nudge/left` | `R2/42` | 20 / 15 / 15 (0 b / 0 a) | 17 / 10 / 10 (0 b / 3 a) | 12 / 5 / 5 (0 b / 8 a) | 0 / 0 / 0 (20 b / 0 a) |
| V0 | `nudge/left` | `R2/43` | 20 / 11 / 11 (0 b / 0 a) | 14 / 6 / 6 (0 b / 6 a) | 15 / 5 / 5 (0 b / 5 a) | 1 / 0 / 0 (19 b / 0 a) |
| V0 | `nudge/left` | `R0` | 13 / 5 / 9 (0 b / 7 a) | 15 / 7 / 11 (0 b / 5 a) | 12 / 6 / 10 (0 b / 8 a) | 12 / 6 / 8 (0 b / 8 a) |
| V0 | `nudge/right` | `R2/42` | 19 / 13 / 13 (0 b / 1 a) | 15 / 7 / 7 (0 b / 5 a) | 13 / 3 / 3 (0 b / 7 a) | 0 / 0 / 0 (20 b / 0 a) |
| V0 | `nudge/right` | `R2/43` | 20 / 13 / 13 (0 b / 0 a) | 15 / 9 / 9 (0 b / 5 a) | 14 / 4 / 4 (0 b / 6 a) | 0 / 0 / 0 (19 b / 1 a) |
| V0 | `nudge/right` | `R0` | 17 / 4 / 9 (0 b / 3 a) | 15 / 1 / 6 (0 b / 5 a) | 16 / 6 / 10 (0 b / 4 a) | 14 / 4 / 9 (0 b / 6 a) |
| V0 | `stop/two_step` | `R2/42` | 20 / 20 / 20 (0 b / 0 a) | 20 / 20 / 20 (0 b / 0 a) | 20 / 20 / 20 (0 b / 0 a) | 0 / 0 / 0 (20 b / 0 a) |
| V0 | `stop/two_step` | `R2/43` | 20 / 20 / 20 (0 b / 0 a) | 20 / 20 / 20 (0 b / 0 a) | 20 / 20 / 20 (0 b / 0 a) | 1 / 0 / 0 (19 b / 0 a) |
| V0 | `stop/two_step` | `R0` | 20 / 19 / 19 (0 b / 0 a) | 20 / 18 / 18 (0 b / 0 a) | 20 / 20 / 20 (0 b / 0 a) | 20 / 20 / 20 (0 b / 0 a) |
| V0 | `turn/q90_left` | `R2/42` | 20 / 0 / 0 (0 b / 0 a) | 20 / 0 / 10 (0 b / 0 a) | 20 / 0 / 1 (0 b / 0 a) | 0 / 0 / 0 (20 b / 0 a) |
| V0 | `turn/q90_left` | `R2/43` | 20 / 0 / 0 (0 b / 0 a) | 20 / 0 / 7 (0 b / 0 a) | 20 / 0 / 2 (0 b / 0 a) | 0 / 0 / 0 (19 b / 1 a) |
| V0 | `turn/q90_left` | `R0` | 20 / 0 / 0 (0 b / 0 a) | 19 / 0 / 0 (0 b / 1 a) | 20 / 0 / 0 (0 b / 0 a) | 20 / 0 / 0 (0 b / 0 a) |
| V0 | `turn/q90_right` | `R2/42` | 20 / 0 / 0 (0 b / 0 a) | 20 / 0 / 12 (0 b / 0 a) | 20 / 0 / 3 (0 b / 0 a) | 0 / 0 / 0 (20 b / 0 a) |
| V0 | `turn/q90_right` | `R2/43` | 20 / 0 / 0 (0 b / 0 a) | 20 / 0 / 9 (0 b / 0 a) | 20 / 0 / 0 (0 b / 0 a) | 0 / 0 / 0 (19 b / 1 a) |
| V0 | `turn/q90_right` | `R0` | 19 / 0 / 0 (0 b / 1 a) | 19 / 0 / 0 (0 b / 1 a) | 20 / 0 / 0 (0 b / 0 a) | 20 / 0 / 0 (0 b / 0 a) |
| F25F9cl | `nudge/backward` | `R2/42` | 19 / 0 / 0 (0 b / 1 a) | 18 / 0 / 0 (0 b / 2 a) | 0 / 0 / 0 (19 b / 1 a) | 0 / 0 / 0 (20 b / 0 a) |
| F25F9cl | `nudge/backward` | `R2/43` | 17 / 0 / 0 (0 b / 3 a) | 12 / 0 / 0 (0 b / 8 a) | 0 / 0 / 0 (17 b / 3 a) | 0 / 0 / 0 (20 b / 0 a) |
| F25F9cl | `nudge/backward` | `R0` | 9 / 0 / 0 (0 b / 11 a) | 11 / 0 / 0 (0 b / 9 a) | 8 / 0 / 0 (0 b / 12 a) | 9 / 0 / 0 (0 b / 11 a) |
| F25F9cl | `nudge/forward` | `R2/42` | 20 / 0 / 0 (0 b / 0 a) | 20 / 0 / 0 (0 b / 0 a) | 0 / 0 / 0 (19 b / 1 a) | 0 / 0 / 0 (20 b / 0 a) |
| F25F9cl | `nudge/forward` | `R2/43` | 19 / 0 / 0 (0 b / 1 a) | 19 / 0 / 0 (0 b / 1 a) | 0 / 0 / 0 (17 b / 3 a) | 0 / 0 / 0 (20 b / 0 a) |
| F25F9cl | `nudge/forward` | `R0` | 7 / 0 / 0 (0 b / 13 a) | 7 / 0 / 0 (0 b / 13 a) | 8 / 0 / 0 (0 b / 12 a) | 7 / 0 / 0 (0 b / 13 a) |
| F25F9cl | `nudge/left` | `R2/42` | 15 / 0 / 0 (0 b / 5 a) | 12 / 0 / 0 (0 b / 8 a) | 0 / 0 / 0 (19 b / 1 a) | 0 / 0 / 0 (20 b / 0 a) |
| F25F9cl | `nudge/left` | `R2/43` | 15 / 0 / 0 (0 b / 5 a) | 12 / 0 / 0 (0 b / 8 a) | 0 / 0 / 0 (17 b / 3 a) | 0 / 0 / 0 (20 b / 0 a) |
| F25F9cl | `nudge/left` | `R0` | 10 / 0 / 0 (0 b / 10 a) | 9 / 0 / 0 (0 b / 11 a) | 10 / 0 / 0 (0 b / 10 a) | 7 / 0 / 0 (0 b / 13 a) |
| F25F9cl | `nudge/right` | `R2/42` | 15 / 0 / 0 (0 b / 5 a) | 6 / 0 / 0 (0 b / 14 a) | 0 / 0 / 0 (19 b / 1 a) | 0 / 0 / 0 (20 b / 0 a) |
| F25F9cl | `nudge/right` | `R2/43` | 15 / 0 / 0 (0 b / 5 a) | 11 / 0 / 0 (0 b / 9 a) | 0 / 0 / 0 (17 b / 3 a) | 0 / 0 / 0 (20 b / 0 a) |
| F25F9cl | `nudge/right` | `R0` | 5 / 0 / 0 (0 b / 15 a) | 4 / 0 / 0 (0 b / 16 a) | 4 / 0 / 0 (0 b / 16 a) | 6 / 0 / 0 (0 b / 14 a) |
| F25F9cl | `stop/two_step` | `R2/42` | 20 / 0 / 0 (0 b / 0 a) | 20 / 0 / 0 (0 b / 0 a) | 1 / 0 / 0 (19 b / 0 a) | 0 / 0 / 0 (20 b / 0 a) |
| F25F9cl | `stop/two_step` | `R2/43` | 20 / 0 / 0 (0 b / 0 a) | 20 / 0 / 0 (0 b / 0 a) | 1 / 0 / 0 (18 b / 1 a) | 0 / 0 / 0 (20 b / 0 a) |
| F25F9cl | `stop/two_step` | `R0` | 20 / 0 / 0 (0 b / 0 a) | 20 / 0 / 0 (0 b / 0 a) | 20 / 0 / 0 (0 b / 0 a) | 20 / 0 / 0 (0 b / 0 a) |
| F25F9cl | `turn/q90_left` | `R2/42` | 20 / 0 / 0 (0 b / 0 a) | 20 / 0 / 0 (0 b / 0 a) | 1 / 0 / 0 (19 b / 0 a) | 0 / 0 / 0 (20 b / 0 a) |
| F25F9cl | `turn/q90_left` | `R2/43` | 20 / 0 / 0 (0 b / 0 a) | 19 / 0 / 0 (0 b / 1 a) | 1 / 0 / 0 (17 b / 2 a) | 0 / 0 / 0 (20 b / 0 a) |
| F25F9cl | `turn/q90_left` | `R0` | 0 / 0 / 0 (0 b / 20 a) | 0 / 0 / 0 (0 b / 20 a) | 0 / 0 / 0 (0 b / 20 a) | 0 / 0 / 0 (0 b / 20 a) |
| F25F9cl | `turn/q90_right` | `R2/42` | 20 / 0 / 0 (0 b / 0 a) | 19 / 0 / 0 (0 b / 1 a) | 1 / 0 / 0 (19 b / 0 a) | 0 / 0 / 0 (20 b / 0 a) |
| F25F9cl | `turn/q90_right` | `R2/43` | 20 / 0 / 0 (0 b / 0 a) | 19 / 0 / 0 (0 b / 1 a) | 1 / 0 / 0 (18 b / 1 a) | 0 / 0 / 0 (20 b / 0 a) |
| F25F9cl | `turn/q90_right` | `R0` | 0 / 0 / 0 (0 b / 20 a) | 0 / 0 / 0 (0 b / 20 a) | 0 / 0 / 0 (0 b / 20 a) | 0 / 0 / 0 (0 b / 20 a) |

Simulated, today's walk (V0), fitted rows:

- **Stop.** The two-step stop meets every limit in 20 of 20 at 0, 4 and 6 ms on the grippy floor: two touchdowns
  after the command, forward motion over after a median of 1.24 to 1.29 strides, 0.23 to 0.24 m covered.
- **Quick turn.** It stays up 20 of 20 at 0, 4 and 6 ms and has turned a median of 82 to 85° at the judged instant,
  2 s after the command (commanded 90°; the heading limit is 10°). It passes in neither reading. With no delay the failing terms are
  the tail servo (20 of 20), duty (20 of 20), and in 3 to 10 episodes a drive's peak speed, toe clearance and
  heading. At 4 ms, with the tail servo set aside, 7 to 12 of 20 pass. This turn's reference is itself over the tail
  servo's limit, so only the reading without A1 can pass (plan, H3).
- **Pushes.** With no delay it stays up 19 or 20 of 20 for every direction, and 11 to 15 of 20 meet every limit: the
  failing terms are a drive's peak speed, "back on the command within 3 s" and, in 1 to 6 episodes, a servo's
  sustained load. At 4 and 6 ms 0 to 11 of 20 fall after the push, and 3 to 19 of 20 meet every limit.
- **At 10 ms** nothing is read: the robot falls before the command in 17 to 20 of 20.

Simulated, the candidate:

- With no delay it stays up through the stop (20 of 20), the quick turn (20 of 20) and forward and backward pushes
  (17 to 20 of 20); a sideways push fells 5 of 20, where today's walk loses at most 1.
- It passes nothing as registered. In the stop and the pushes the only failing term among the episodes that stayed
  up is the start-up move's joint range (section 6). In the turn the toe clearance fails in every episode (15.5 to
  15.8 mm against 20.07 mm with no delay), which the plan expected: the turn's reference itself reads 16.23 and
  15.84 mm when played without physics. With no delay the duty term fails in every episode of the turn as well.
- With no balance feedback (`R0`) the candidate falls in every episode of every quick turn, after the command;
  today's walk stays up 19 or 20 of 20 on the same turn.

## 6. The candidate, read again with one term set aside

**Label: a re-reading made on 15 Nov after the walk records were read (`STAGE2_75_PLAN.md`, closing amendments,
item 11). It is not a verdict and changes no rule.** The registered counts are in sections 3 to 5.

What happened. The walk rule judges the joint range at 0.10 rad over the whole episode. Every episode begins with the
set's start-up move from standing, and the candidate's start-up move comes within 0.095 rad of the hip-roll limit.
Every episode of the candidate that stayed up reads 0.0950 to 0.0952 rad, under all three rows and at every delay, so
the term fails 20 of 20 whatever the controller does. The plan already named the 0.095 rad when the owner relaxed the
*build* gate for start-up moves to 0.05 rad (P34), but nobody carried it over to the walk rule's closed-loop reading.
The lead should have stated it as an expectation before the runs. Today's walk and the pull request's walk keep 0.127
and 0.136 rad and are not touched by this.

The same records with that one term set aside (`stage2/runs/2026-11-15-75cm/rereads/JOINT_RANGE_ASIDE.md`; counts of
20, `R2/42` · `R2/43`):

| Walk (candidate) | Delay | Stayed up | Every limit, as registered | Every limit, joint range aside | Stage 1's reading, as recorded | Stage 1's reading, joint range aside |
|---|---|---|---|---|---|---|
| 0.24 m/s (`nominal`) | 0 ms | 20 · 20 | 0 · 0 | 0 · 0 | 0 · 0 | 0 · 0 |
| 0.24 m/s (`nominal`) | 4 ms | 20 · 20 | 0 · 0 | 0 · 0 | 0 · 0 | 0 · 0 |
| 0.24 m/s (`nominal`) | 6 ms | 20 · 19 | 0 · 0 | 0 · 0 | 0 · 0 | 0 · 0 |
| 0.30 m/s | 0 ms | 20 · 20 | 0 · 0 | 19 · 20 | 0 · 0 | 20 · 20 |
| 0.30 m/s | 4 ms | 20 · 20 | 0 · 0 | 0 · 0 | 0 · 0 | 19 · 19 |
| 0.30 m/s | 6 ms | 19 · 17 | 0 · 0 | 0 · 0 | 0 · 0 | 0 · 0 |
| 0.35 m/s | 0 ms | 20 · 20 | 0 · 0 | 18 · 18 | 0 · 0 | 20 · 20 |
| 0.35 m/s | 4 ms | 20 · 20 | 0 · 0 | 0 · 0 | 0 · 0 | 15 · 12 |
| 0.35 m/s | 6 ms | 14 · 18 | 0 · 0 | 0 · 0 | 0 · 0 | 0 · 0 |
| 0.40 m/s | 0 ms | 20 · 20 | 0 · 0 | 8 · 18 | 0 · 0 | 20 · 20 |
| 0.40 m/s | 2 ms | 20 · 20 | 0 · 0 | 0 · 0 | 0 · 0 | 20 · 20 |
| 0.40 m/s | 4 ms | 20 · 20 | 0 · 0 | 0 · 0 | 0 · 0 | 2 · 2 |
| 0.40 m/s | 6 ms | 0 · 0 | 0 · 0 | 0 · 0 | 0 · 0 | 0 · 0 |
| 0.45 m/s | 0 ms | 20 · 20 | 0 · 0 | 14 · 16 | 0 · 0 | 20 · 20 |
| 0.45 m/s | 4 ms | 20 · 20 | 0 · 0 | 0 · 0 | 0 · 0 | 3 · 0 |
| 0.45 m/s | 6 ms | 0 · 0 | 0 · 0 | 0 · 0 | 0 · 0 | 0 · 0 |
| 0.50 m/s | 0 ms | 20 · 20 | 0 · 0 | 3 · 15 | 0 · 0 | 20 · 20 |
| 0.50 m/s | 2 ms | 20 · 20 | 0 · 0 | 0 · 0 | 0 · 0 | 20 · 20 |
| 0.50 m/s | 4 ms | 20 · 20 | 0 · 0 | 0 · 0 | 0 · 0 | 1 · 0 |
| 0.50 m/s | 6 ms | 0 · 0 | 0 · 0 | 0 · 0 | 0 · 0 | 0 · 0 |
| 0.40 m/s, slippery floor | 0 ms | 20 · 20 | 0 · 0 | 10 · 17 | 0 · 0 | 20 · 20 |
| 0.40 m/s, slippery floor | 2 ms | 20 · 20 | 0 · 0 | 0 · 0 | 0 · 0 | 20 · 20 |
| 0.40 m/s, slippery floor | 4 ms | 20 · 20 | 0 · 0 | 0 · 0 | 0 · 0 | 1 · 3 |
| 0.40 m/s, slippery floor | 6 ms | 0 · 0 | 0 · 0 | 0 · 0 | 0 · 0 | 0 · 0 |
| 0.45 m/s, slippery floor | 0 ms | 20 · 20 | 0 · 0 | 11 · 18 | 0 · 0 | 20 · 20 |
| 0.50 m/s, slippery floor | 0 ms | 20 · 20 | 0 · 0 | 8 · 17 | 0 · 0 | 20 · 20 |

- **By this study's reading**, with the term aside, the candidate would meet "20 up and at least 17" under both
  controllers at 0.30 and 0.35 m/s with no delay, and nowhere else: at 0.40 m/s one controller has 8 of 20 because
  the toe clearance sits on the limit, and with any delay the toe clearance fails in every episode. Its slow walk
  (0.24 m/s) fails the stride and cadence terms in every episode, apart from all this.
- **By Stage 1's reading**, which is the one the pull request's report uses, the candidate meets every limit in 20
  of 20 at 0 ms from 0.30 to 0.50 m/s and at 2 ms at 0.40 and 0.50 m/s; at 4 ms it drops to 0 to 3 of 20 at 0.40 m/s
  and faster.
- **The difference between the two readings is the toe clearance.** Stage 1 reads the lowest point over the swing as
  the simulation ran it; this study's bar reads the middle half of the *planned* swing counted from the foot's own
  lift-off, which is stricter when the foot lifts late. On the candidate at 0.40 m/s with no delay the two readings
  give 23.8 to 24.0 mm and 20.0 to 20.6 mm.
- **Stop and pushes:** with the term aside the candidate's stop would read 20 of 20 at 0 and 4 ms, and its forward
  and backward pushes 17 to 20 of 20 at 0 ms. Its sideways pushes stay at 15 of 20 up, which is not a pass.

Nothing was replicated on fresh seeds: the rule replicates what the candidate passes, and it passed nothing.

## 7. The pull request's claims beside these records

The pull request's report (`stage2/jetson_plem/JETSON_PLEM_AND_SPEED.md`, section 2.3) has a table of results with
no code and no record behind it (`JETSON_PLEM_REVIEW.md`). `BAR_TABLES.md` section 6 sets each of its lines beside
this round's records, seed by seed. In short:

| Pull request's claim (its line) | This round's records (simulated) |
|---|---|
| Today's walk at 0.35 m/s, no delay: stays up 20 of 20, fails only the tail servo (92) | The same: 20 of 20 up; 0 of 20 meet every limit; 19 of 20 with the tail servo set aside |
| Today's walk at 0.50 m/s, no delay: stays up, fails tail servo, duty and ankle speed (93) | The same in kind: 20 of 20 up; the tail servo and duty fail under both controllers, a drive's peak speed under one |
| Its walk at 0.50 m/s, duty 0.55: one controller passes (94) | Not run: under this round's rules that walk cannot be reached from the set's 0.40 m/s walk |
| Candidate at 0.40, 0.45 and 0.50 m/s, no delay, both floors: 20 of 20 meet every limit (95) | Registered: 0 of 20 (the start-up term). By Stage 1's reading with that term aside: 20 of 20 in all six cells |
| Candidate at 2 ms, 0.40 and 0.50 m/s: 20 of 20 (98) | Registered: 0 of 20. By Stage 1's reading with the term aside: 20 of 20 in both cells |
| Candidate at 4 ms loses toe clearance; at 6 to 10 ms it falls 0 of 20 (99) | The same: at 4 ms 0 to 3 of 20 by Stage 1's reading with the term aside; at 6 ms and at 10 ms 0 of 20 stay up at 0.40 and 0.50 m/s |
| Two-step stop at 2 ms: 20 of 20 (100) | 2 ms was not run for the stop. At 0 and 4 ms: 20 of 20 up, the start-up term the only one failing |
| Pushes at 2 ms: forward 20 of 20 up, left 15 of 20 up (101) | 2 ms was not run for the pushes. At 0 ms: forward 20 and 19 up, left 15 and 15 up |
| 0.55 m/s; a body with the Jetson's added mass (96, 97) | Not run in this round |

So the pull request's table is largely reproduced **under the reading it used**. What this round adds is that the
same walk fails this study's stricter toe-clearance reading as soon as there is any delay, and that it falls at 6 ms
where today's walk still stands.

## 8. Turning room

[to be added when the reviewed views are merged: TURN_SECTION]

## 9. What this does not show

- **Nothing about hardware.** The servo delay of the real build is not measured. The design estimate for the
  baseline wiring (a microcontroller with two CAN buses) is 0.3 to 3.5 ms, which maps onto 2 to 4 ms of simulated
  delay, or 2 to 6 ms on the pessimistic reading of the drive manuals (`stage2/jetson75/LATENCY_AND_BUS.md`; an
  estimate, kind (d) there).
- **Only one kind of controller.** Both fitted rows are Stage 1's balance tracker, fitted with no delay. The policy
  to be trained on Colab (R5) has not been run on this body. A refit of the tracker with delay in its mix (R2L) is
  planned; on the 60 cm bodies such a refit still fell at 10 ms (`DECISION_LOG.md` F27).
- **Not run in this round, by plan:** the turns, pushes and stop at 2 and 8 ms; carpet; the look gates; the
  Jetson-mass body; 0.55 m/s; any 60 cm body. `BAR_TABLES.md` section 9 lists each.
- **The candidate's margin was set after results on these seeds** and nothing of it was replicated, because it
  passed nothing.
- **The 1 rad/s turn has no reference** on either set (the built motion is over a leg drive's limit or outside the
  thigh twist's range); its four records per set are written as "not run" with that reason.
- **Fall attribution is a hint.** The labels in section 3 come from a paired sign test over 20 starts and say
  nothing certain about cause.

## 10. For the owner

Nothing below is adopted. The lead's reading is an interpretation of the results above, not a result.

**The lead's reading.** The binding constraint on the 75 cm build in simulation is the control delay, not the choice
of walk. All three walks stand at 4 ms and fall at 8 ms under controllers fitted with no delay, and Stage 1b's
refit with delay on the 60 cm bodies did not move that. The design estimate for the baseline wiring sits at 2 to
4 ms, inside the range where today's walk holds. Between the walks neither wins outright: the pull request's walks
remove the tail servo's overload, and pay for it in toe clearance and in delay tolerance.

**Questions.**

1. **The judged delay.** The bar is judged at an assumed 10 ms until a bench value exists (P30), and the bench waits
   for a 75 cm build that works in simulation (P32). With controllers of this kind no walk passes at 10 ms, so
   each of the two is waiting for the other. The choices: (a) keep 10 ms and wait for the trained policy's results; (b) judge at the upper end of the
   design estimate (4 ms, or 6 ms to be cautious), labelled as an assumption, and keep reporting 10 ms; (c) bring the
   delay bench forward.
2. **Pull request #668.** Its walk results reproduce under its own reading (section 7); its control layout was
   reviewed separately (`JETSON_PLEM_REVIEW.md`). Whether to merge it, close it, or let this branch's records
   supersede its table is the owner's call (P30, decision 8).
3. **The tail servo at 0.35 m/s and faster.** Today's walk overloads it by 2 to 7 %. Options to test in simulation
   (P33): a smaller tail swing, a stronger servo, or a 0.30 m/s cruising pace.
4. **The candidate's start-up move.** Rebuilding it with more hip-roll margin, or reading a walk's joint range over
   its window as a turn's is read, would remove the term that fails by itself. Either is a new freeze or a rule
   change after a result. It would not change the 10 ms verdicts or the toe-clearance result.
5. **Carpet (P36).** No sourced carpet model could be written (`stage2/design/carpet_sources_15nov.md`). The two
   ways on are a bracket of floors with no verdict, or a short bench measurement.

**Draft decision-log rows (proposed, not adopted).**

| Row | Kind | Text |
|---|---|---|
| F28 (draft) | Finding | 75 cm agility bar, first panels (simulation, 15 Nov 2026; `stage2/runs/2026-11-15-75cm/`). At 10 ms of servo delay none of the 16 conditions is shown on `V0`, `F25F9` or `F25F9cl` under Stage 1's fitted tracker (fits 42 and 43, made with no delay): 0 to 3 of 20 episodes stay up, and 0 of 20 at 8 ms. At 0.40 m/s all three stay up 20 of 20 to 4 ms; V0 to 6 ms. V0 meets every limit at 0.24 and 0.30 m/s at 0 ms and every limit but A1 (tail servo, 1.00 to 1.08 of its rating) at 0.35 m/s (0 and 4 ms) and at 0.40 m/s (0 and 2 ms). F25F9cl passes nothing as registered; its start fails the walk rule's joint range by itself (0.095 against 0.10 rad). Budget used: 4.25 of 16 and 5.51 of 12 core-hours, nothing cut. |
| F29 (draft) | Finding | The candidate's results in pull request #668's table (section 2.3, lines 95, 98, 99) are reproduced under Stage 1's reading with the start-up joint-range term set aside (a re-reading after a result, no verdict): 20 of 20 at 0 and 2 ms at 0.40 and 0.50 m/s, 0 to 3 of 20 at 4 ms, falls at 6 ms. Under this study's toe-clearance reading the candidate is at the limit at 0 ms (20.0 to 20.6 mm against 20.07 mm) and under it from 2 ms. |
| E (draft) | Error | The lead did not state before the runs that F25F9cl's start (0.095 rad of hip-roll margin, known since P34) fails the walk item's 0.10 rad joint range over the episode. Found after the walk records; noted in the plan (closing amendments, item 11); no rule changed. |

## 11. Records, and how to make the tables again

- Records: `stage2/runs/2026-11-15-75cm/<set>/75-M-H-10/seeds3042/eval/<row>@<delay>ms/<behaviour>/<condition>.json`;
  the ledger `budget.json`; the rates `COST.md`.
- Machine-made tables: `python -m designlab.stage2.barreport --out stage2/runs/2026-11-15-75cm` writes
  `BAR_TABLES.md` and `bar_tables.json` there.
- Digest: `python stage2/runs/2026-11-15-75cm/digest/digest.py`.
- Re-reading: `python stage2/runs/2026-11-15-75cm/rereads/term_aside.py`.
- Turning figures and clips: `python -m designlab.stage2.barviews turnpath --set <set>` and `... clips ...`
  (`stage2/runs/2026-11-15-75cm/views/`).
- The sets as frozen: `stage2/sets/<set>/75-M-H-10/set.json`.
