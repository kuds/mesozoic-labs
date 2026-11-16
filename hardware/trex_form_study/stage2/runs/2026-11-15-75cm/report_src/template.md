# The 75 cm build in simulation: the first panels of the agility bar

15 Nov 2026. Body `75-M-H-10`. **Simulation only: nothing in this report was built or measured on hardware.** Nothing
is adopted by it. Section 3 lists what is for the owner to decide.

{{STATUS}}

Paths are relative to `hardware/trex_form_study/`. Every count below is read from the committed records under
`stage2/runs/2026-11-15-75cm/`; where anything here differs from the records, the records decide.

## 1. The short answer

*Words. **Delay**: the time between a command and the joint starting to move, in the model. **Today's walk**: the
study's current reference walk. **The pull request's walk**: an outside proposal for a faster walk. **The
candidate**: that walk with a small safety margin. **A run** lasts 14 to 27 s on a flat hard floor.*

**Will the 75 cm build walk?** In simulation, with today's walk and the balance controller carried over from
Stage 1:

- **It stays up when the delay is 6 ms or less.** At 0.24 to 0.40 m/s it stays up in 20 of 20 runs at every delay
  tried up to 6 ms (runs of 14 s at 0.24 m/s and about 21 s at the other speeds, no push). At 0.40 m/s it falls
  in every run at 8 ms.
- **At 10 ms, the delay the study assumes until one is measured, nothing passes and it hardly walks at all.** Of 20
  runs, 0 to 3 stay up. In the 0.40 m/s walk records (80 runs: two fits, two floors) 76 fall, all before the
  0.40 m/s walk begins: 64 in the start from standing or the speed-up and 12 in the first slow stride. Asked only
  to walk slowly (0.24 m/s), it stays up in 3 of 40 runs, and 27 of the 37 falls come during the slow walk itself.
- **The real delay is not measured.** The design estimate for the baseline wiring (a microcontroller with two CAN
  buses) is 2 to 4 ms, or 2 to 6 ms on the cautious reading. Every term in it is an estimate; none is measured.
  The cautious top end, 6 ms, is one 2 ms step below 8 ms, where every run falls.
- **Staying up is not the same as passing.** Under both fits of the controller, today's walk meets every limit at
  0.24 and 0.30 m/s with no delay. At 4 ms it does not: at 0.30 m/s one fit has 17 of 20 and the other 12, and 2 ms
  was not run at these speeds. The limit it misses there is toe clearance (how high the foot is above the floor in
  mid-step; limit 20 mm), read this study's own, stricter way.
- **The tail servo is over the study's planning limit at 0.35 m/s and faster.** At 0.35 and 0.40 m/s, at 0 to 4 ms,
  it is the one limit that fails in every run, at up to 1.11 times the planning limit for continuous load; at 6 ms
  toe clearance fails in every run as well. That limit is 0.57 of the maker's rated torque, an assumption no bench
  test has checked, and about half of the tail's load in the model comes from joint damping and friction values
  that have no source.
- **It is not yet robust.** No push test and no quick-turn test passes under both fits of the controller at any
  delay, not even with none. With no delay a light push (about 22 N for a tenth of a second) is survived in 19 or
  20 of 20 runs on the grippy floor; at 4 ms a sideways or backward push fells it in 3 to 8 of 20. It stays up
  through the quick turn in 19 or 20 of 20 at 0, 4 and 6 ms, but misses limits.
- **The stop works.** The two-step stop meets every limit in 20 of 20 runs at 0, 4 and 6 ms, on both floors.
- **This is the stand-in controller, and the next one is not a sure cure.** It was fitted with no delay. The plan's
  judged controller, a policy trained with delay, has not been trained or run on this body. Two earlier results of
  this study count against assuming it will work: in Stage 1 a learned add-on to this controller, trained on this
  body with 0 to 20 ms of random delay, also fell at 10 ms (three slow-walk runs; its learned part hardly acted),
  and one refit of the controller with delay, on the 60 cm bodies, still fell at 10 ms.

Runs that stayed up, of 20, walking at 0.40 m/s on the grippy floor (the two fits of the controller, side by side):

{{SMALL}}

**The two other walks.** The candidate and the pull request's walk pass nothing. Both carry a tail that swings in
counter-phase, and at 0.40 m/s their tail servo reads 0.40 to 0.48 of the planning limit, against 1.02 to 1.11 for
today's walk. Section 8 explains a fault in the rules, made by the lead, that fails the candidate on one limit
before its controller matters.

**Turning room.** {{TURN_ANSWER}}

**Compute.** The walk panels used 4.25 of their 16 core-hours and the turn, push and stop panels 5.51 of their 12.
Nothing was cut. 592 records were written under the three walks, and 2 records of a registered control.

**Added 16 Nov 2026: longer delays** (simulation; stand-in controller fitted with no delay; delays that are not
the judged 10 ms; no verdict; 14, 16 and 20 ms are the delays the owner approved (P37), 12 and 18 ms are the lead's
addition). Today's walk was then run at 12 to 20 ms in 2 ms steps (section 13). At 0.40 m/s no run covers the
distance its limit asks at any of these delays. Of 20 runs, 0 to 3 stay up at 12, 14, 16 and 20 ms; at 18 ms 6 and
4 stay up under one controller fit and 18 and 19 under the other. At 12 to 16 ms most falls come in the start from
standing, before any walking at 0.40 m/s. The slow walk stays up again at 20 ms (19 and 20 of 20), short of its
distance and outside its limits. Nothing above changes.

## 2. What this does not show

- **Nothing about hardware.** No delay, friction or gear play of the real drives is measured. The delay estimate is
  from `stage2/jetson75/LATENCY_AND_BUS.md`: 0.3 to 3.5 ms for a Teensy microcontroller with two CAN buses, under
  either computer, which maps onto 2 to 4 ms of simulated delay, or 2 to 6 ms if the drives' own loop is the slower
  of two published figures. Every term is the author's estimate. The pull request's own layout, the Jetson driving
  one USB-CAN channel directly, is estimated at up to 12 ms. The simulation holds the delay constant; rare late
  cycles are not modelled.
- **Only the stand-in controller.** Both controller rows are Stage 1's balance controller, fitted with no delay.
  The plan's judged controller is a policy trained with the delay drawn at random from 0 to 24 ms (`STAGE2_PLAN.md`
  section 4, as amended by S2-9). A refit of the stand-in with delay in its mix is also planned.
- **A simple servo model.** Ideal position servos behind a torque and speed limit and a delay. Every joint has one
  fixed damping and friction value (0.2 N·m·s/rad and 0.15 N·m), which has no source (`DECISION_LOG.md` A18); no
  gear play.
- **The tail servo's limit is a planning figure.** The "continuous load" limit is 0.57 of the maker's rated torque
  (`DECISION_LOG.md` A1). No thermal test of the servo has been run.
- **Not run in this round, by plan:** the turns, pushes and stop at 2 and 8 ms; the turn, push and stop tests of
  the pull request's walk; a stop from 0.50 m/s; delays above 10 ms; carpet; a softer servo (stiffness × 0.7); the
  checks on how the walk looks; the body with the Jetson's added mass; 0.55 m/s; Stage 1's twelve no-loss
  conditions; the turn and pushes from the slow walk; the rows with no balance feedback on the speed ladder; any
  60 cm body. `BAR_TABLES.md` section 9 lists each.
- **Ten records could not be run.** The fastest turn (57° per second) has no reference that fits the hip's range
  and the leg drives on either walk (8 records), and the pull request's walk cannot reach 0.50 m/s under this
  round's rules (2 records). The runner wrote each as "not run" with its reason.
- **The slower turns, stronger pushes and one-step stop read next to nothing.** They were run at 10 ms alone, where
  the robot falls before the command in 19 or 20 of 20.
- **The toe-clearance reading is this study's own**, changed after a result and labelled so in the plan before any
  walk was frozen. It reads lower than Stage 1's on every walk here under the two controller fits at 0 to 6 ms
  (section 5).

## 3. For the owner

Nothing below is adopted. Each question gives the choices, what each costs, the lead's pick and what happens if
there is no answer.

**Question 1. What to do about the judged delay.** The bar is judged at an assumed 10 ms until a bench value
exists, and the bench waits for a build that works in simulation. With the stand-in controller no walk passes at
10 ms.

- (a) Keep 10 ms and wait for the policy trained on rented GPUs (Colab). Nothing new to decide or run.
- (b) The same, and first refit the stand-in controller with delay on this body, here, in simulation: two fits,
  about 3 core-hours each by the plan's estimate, plus the panels. The plan already holds one such refit. The one
  earlier refit of this kind, on the 60 cm bodies, still fell at 10 ms; this body differs in that it walks with the
  balance feedback switched off.
- (c) The same, and first run today's walk at 14, 16 and 20 ms: minutes of compute, with the rules written first.
  In Stage 1 this body fell at 10 ms and stayed up at 20 ms, meeting no limit there (three runs of the slow walk),
  and 10 ms is one period of this controller's 100 Hz feedback. Nobody has explained it. These runs would show
  whether 10 ms is a special point for this controller.
- (d) Also read a second, labelled verdict at 4 ms or 6 ms. No compute cost. It reverses your decision that the
  other delays are never read as the verdict (P30). On today's records, of the bar's conditions only the stop
  would pass at 4 or 6 ms. At 2 ms the 0.40 m/s walk passes with the tail servo's limit left out, on both floors;
  the turns, pushes and stop were not run at 2 ms.
- (e) Bring the delay bench forward: about $2,000 of catalogue parts for its first run, about $2,400 with two
  Teensy boards and small parts; a 24 V bench supply is not priced. This is before the simulation has chosen a
  build.
- **Lead's pick: (c), then (b).** If there is no answer: (a).

**Question 2. The tail servo at 0.35 m/s and faster.** With no delay today's walk loads it at 5 % over the
planning limit at 0.35 m/s, 8 % at 0.40 m/s (10 to 11 % on the slippery floor), 12 to 13 % at 0.45 m/s and 19 to
27 % at 0.50 m/s. In the model nothing happens; on hardware a servo held over its continuous limit heats up, and
how fast is not measured. About half of the tail's load in the model comes from the unsourced joint damping and
friction: as planned (no physics), the load at 0.40 m/s is 0.56 of the limit without them and 0.98 with them
(0.27 N·m from the motion, 0.27 N·m from damping and friction alone, 0.48 N·m together;
`stage2/look/SCREEN.md`).

- (a) Cruise at 0.30 m/s. With no delay every limit is met, with the tail at 0.96 of the limit; at 4 ms one of the
  two fits passes. No cost.
- (b) Use the counter-phased tail, which is part of the look already chosen (V6) and is not in today's walk. As
  planned (no physics), V6's tail load on this body is 0.39 of the limit at 0.40 m/s against 0.98 for today's walk.
  Simulated, the two other walks of this round carry a counter-phased tail and read 0.40 to 0.48. V6 itself has not
  been simulated under a controller. No new cost: it comes with the look work already planned.
- (c) Test a smaller tail swing or a stronger tail servo in simulation. A few core-hours (the lead's guess).
- (d) A bench reading of the tail joint's friction and of the servo's heating. It would settle what the model
  cannot. It needs a tail servo (about $22) and its mount; the time is not estimated.
- **Lead's pick: (b).** If there is no answer: (b), since the look is already chosen.

**Question 3. Keep the candidate walk in the 75 cm study?** On these records today's walk is ahead on most counts:
it is the only walk with passes, it stays up at 6 ms where the candidate falls, it keeps 4 to 7 mm more toe
clearance at 0.40 m/s, and it takes sideways pushes better. The candidate is ahead on the tail load, on step timing
at 0.45 and 0.50 m/s with no delay, and on forward and backward pushes at 4 ms (13 falls of 160 runs against 40).

- (a) Run nothing more on it. Its clips are still shown beside the look work, as already agreed (P30). No cost.
- (b) Rebuild its first steps from standing with more hip room: a new freeze, and a few core-hours to run its
  panels again (the lead's guess).
- (c) Change how the joint range is read for walks, pushes and stops. No compute cost. A rule change after a
  result. It would also remove the contradiction the lead put into the rules (section 8).
- **Lead's pick: (a).** If there is no answer: nothing more is run on it.

**Question 4. Pull request #668.** Of the ten lines of its results table, two (the candidate at 0 and 2 ms) agree
with these records only under its own reading and with one limit set aside after the result; one more is close;
two are about today's walk and agree, one of them only in part; five were not run as claimed (section 9).

- (a) Leave it open as it is. No cost.
- (b) Close it as superseded by this branch, which already holds its walk as the sets `F25F9` and `F25F9cl` with
  its table checked line by line, and put a dated note at the top of its report here pointing to this one. No
  cost.
- (c) Merge it relabelled, with the same note at the top of its report. No cost.
- (d) Merge it by itself. No cost, but its report would then stand in the repository without these corrections.
- **Lead's pick: (b).** If there is no answer: (a).

**Question 5. Carpet.** You agreed to a carpet side-check after the main panels, recorded as a thin-carpet model
set from published figures (P36). A first lookup, which could open none of the primary documents, found no figure
for the foot pad's grip on such carpet or for how far it sinks (`stage2/design/carpet_sources_15nov.md`), so no
sourced model can be written yet. For scale: the study's existing carpet floor, whose values have no source,
felled every walking start in Stage 1.

- (a) Run today's walk on a softer and on a grippier floor as a range, with floor values that have no source and
  with no verdict. About 1 core-hour (the lead's guess).
- (b) A short bench measurement on a carpet sample (it needs the sample, a luggage scale and a ruler), which
  turns the range into one point.
- (c) Leave carpet until a controller passes on the hard floor. No cost.
- **Lead's pick: (a), at 0 to 4 ms of delay.** If there is no answer: nothing is run on carpet.

**The lead's reading of the results** (an interpretation, not a result).

- **What fails with delay is this controller's feedback loop, not the body or the walk.** With the balance
  feedback switched off the same walks stay up at 10 ms, though they cannot take a push. Whether another fit, or
  another kind of controller, cures it is not shown.
- **For a cure being possible:** in Stage 1 this body fell at 10 ms and stayed up at 20 ms (three runs of the slow
  walk, meeting no limit at 20 ms), and 10 ms is one period of this controller's 100 Hz feedback, so 10 ms may be
  a special point for it. **Against:** the learned add-on trained with delay on this body fell at 10 ms, the one
  refit with delay on the 60 cm bodies fell at 10 ms, and the plan dropped this controller as the judged one for
  that reason.
- **Delay is not the only shortfall.** With no delay today's walk passes 4 of the bar's 16 conditions: the walk on
  both floors (with the tail servo's limit left out) and the stop on both floors. The quick turn and the pushes do
  not pass at any delay, not even with none.
- **An outside reference, not a result of this study.** The open-source Microduck robot trains its walking policy
  through 15 to 30 ms of command delay drawn at random, with a detailed servo model and numbers checked on a servo
  bench (`stage2/research/microduck_delay_15nov.md`; read, not run). That is the remedy this study's plan already
  holds for its trained policy. It shows nothing about this body.

## 4. What was run, and the words used

**Four kinds of statement are kept apart.** *Simulated*: the robot model walking under a controller in the physics
simulation; every count and measured median here. *Planned*: a reference motion played without physics; the turning
circle's planned numbers and "the reference's own toe clearance". *Design estimate*: the author's term-by-term estimate from catalogue facts; nothing in it is measured.
*Measured on hardware*: none exists.

**Three walks on one body.** Each is a frozen set of reference motions for `75-M-H-10`.

| Set | What it is | Role |
|---|---|---|
| `V0` | Today's walk (Stage 1's reference), with a 0.24 to 0.50 m/s speed ladder | control |
| `F25F9` | The walk of pull request #668 (an outside proposal for a faster walk and a Jetson computer), as its code builds it | control, for the walk only |
| `F25F9cl` | That walk with each foot on the ground for 0.59 of the step cycle and a higher foot lift | candidate; its margin was set after results on these seeds |

**Three controller rows.** `R2/42` and `R2/43` are Stage 1's balance controller fitted twice, with no servo delay
in the fit. A verdict needs both. `R0` has no balance feedback; it is reported and gives no verdict.

**Every episode starts from standing.** On V0 and the candidate the robot starts, takes one slow stride, speeds up
and then walks at 0.40 m/s. On F25F9 it starts straight into 0.40 m/s. A turn, push or stop command comes after
about 14 s on today's walk and about 18 s on the candidate.

**Servo delay.** The time between the controller's command and the change at the joint, in the model. The bar is
judged at 10 ms (`DECISION_LOG.md` P30, P32). 0, 2, 4, 6 and 8 ms are reported beside it for the walk at 0.40 m/s;
the speed ladder and the turns, pushes and stop are read at 0, 4, 6 and 10 ms.

**Floors.** A grippy hard floor (friction 0.9) and a slippery one (0.54). The speed ladder runs on the grippy floor.

**Pass rule, written before the runs.** A record is 20 episodes on seeds 3042 to 3061. It passes when all 20 stay
up and at least 17 meet every limit. A condition is "shown" when both controller fits pass at 10 ms. A pass at any
other delay is reported and is no verdict.

**Two registered readings of every episode.** *Every limit*: speed, straightness, toe clearance, step timing (duty:
the share of each step cycle a foot is on the ground), slip, the drives' loads and speeds, joint range. *But for A1*:
every limit except the continuous-load limit of the neck and tail servos, which the study calls A1.

**The bar's 16 conditions:** the walk at 0.40 m/s, a 90° turn to each side at 45° per second (the quick turn), a
push from each of four sides (about 22 N, roughly 2 kg of force, for 0.1 s), and a two-step stop; each on both
floors.

**The rules and the tables.** `STAGE2_75_PLAN.md`, code state `c377ba1` (committed 19:42 UTC; the first episode
started at 19:44 UTC and the chain ended at 23:09 UTC). The machine-made tables with every record path and label are
`stage2/runs/2026-11-15-75cm/BAR_TABLES.md`. The tables below are copied by script from a digest of the same
records, `stage2/runs/2026-11-15-75cm/digest/DIGEST.md`.

## 5. The walk at 0.40 m/s against servo delay

Cells: **episodes that stayed up / met every limit / met every limit but for A1**, each of 20.

Grippy floor (friction 0.9):

{{WALK_09}}

Slippery floor (friction 0.54):

{{WALK_054}}

**Where the falls come.** The falls of the two controller fits on both floors together (80 episodes per cell), by
the part of the episode the robot had reached:

{{FALLS}}

What the tables show (simulated):

- **10 ms.** Under both controller fits 0 to 3 of 20 stay up, and no fall is in the 0.40 m/s walk itself: every one
  comes earlier. The candidate and the pull request's walk fall in their start from standing.
- **8 ms.** 0 of 20 stay up in all twelve records. Today's walk gets further: 36 of its 80 falls are in the
  0.40 m/s walk and 39 in the speed-up. The other two still fall mostly in the start.
- **6 ms.** Today's walk does not fall. The candidate falls in every episode: in the speed-up (51), in the
  0.40 m/s walk (28) or in the slow stride (1).
- **No balance feedback.** With `R0` the same walks stay up 18 to 20 of 20 at every delay and meet no limit set.
- **Expected before the runs.** The plan said, before any walk was frozen, that "the 10 ms verdicts may be 'not
  shown' because the robot falls, whatever the walk". That came true at 10 ms and at 8 ms. It also said the
  every-limit reading might fail on today's walk for the tail servo alone. That came true at 0.35 m/s at 0 and
  4 ms and at 0.40 m/s at 0 and 2 ms, where the tail servo is the only limit that fails in 20 of 20 (0.35 m/s was
  not run at 2 ms).
- **The pull request's walk is under the toe limit in every episode that stayed up, at every delay**, which the
  plan expected: its reference itself, played without physics, clears only 19.74 mm.
- **The two floors give the same picture for the walk.** The largest difference between a cell above and its twin
  on the other floor is 5 episodes (V0, `R2/42`, 4 ms, but for A1: 16 against 11).
- **The registered control** (40 episodes, no verdict): the candidate's walk under `R2/42` with the controller's
  pendulum frequency computed as Stage 1 computed it. At 10 ms 0 of 20 stay up, as in the registered record; at 0 ms
  20 stay up and 0 meet every limit.

Measured medians on the grippy floor, both controller fits, 0 to 6 ms. Speed is the simulated median over the last
10 s (the target is 0.40 m/s; the reference is played 5 % fast, at 0.42 m/s). Toe clearance is read as the bar
judges it, against a limit of 20.07 mm; Stage 1's reading of the same episodes stands beside it and is not judged.
Duty is the mean of the two feet; its limits are 0.55 to 0.65. The servo columns are continuous load over the
planning limit; the limit is 1.0. In the last column `survived` is a fall, `drive_sustained` the continuous-load
limit of a drive, `line` straightness, `along` the distance covered, `joint_range` the room left at the end of a
joint's travel, `toe_clearance_own_swing` the toe clearance, `duty` the step timing.

{{MEDIANS}}

- **Toe clearance falls with delay on every walk**: on V0 from 24.2 mm with no delay to 20.2 to 20.7 mm at 4 ms and
  18.3 to 18.9 mm at 6 ms; on F25F9 from 17.2 to 17.8 mm to 11.8 to 12.6 mm at 6 ms; on the candidate from 20.0 to
  20.6 mm to 13.0 to 13.7 mm at 4 ms.
- **Today's walk loses its passes at 4 and 6 ms on this study's toe reading.** At 4 ms the toe limit fails in 4 to 9
  of 20 episodes at 0.40 m/s (both floors), at 6 ms in 19 or 20. By Stage 1's reading of the same episodes the toe
  clears 22.8 to 26.9 mm at 4 and 6 ms (both floors).
- **The tail servo is over the planning limit only on today's walk** (1.02 to 1.08 at 0.40 m/s on the grippy floor,
  1.04 to 1.11 on the slippery floor, over 0 to 6 ms). On both of the pull request's walks, whose tail swings in
  counter-phase, the tail reads 0.40 to 0.48 and the neck, at 0.69 to 0.70, is the larger of the two. The neck reads
  0.70 on today's walk as well.
- **The runner's fall labels at 10 ms are hints, and one of them is weak.** On V0 and F25F9 the label is
  "controller-limited", with one possible cause (the foot's pad edge) that the simulation cannot test. On the
  candidate it is "body-limited: drive torque". The effect behind that label is one physics step: with the leg
  drives' torque limit lifted, the fall came a median of 0.002 s later and 0 of 20 stayed up, in all four records.
  The sign test calls that significant; it is no evidence that stronger drives would keep the candidate up. All the
  candidate's 10 ms falls are in its start from standing.
- **When the falls come at 10 ms.** The candidate falls after a median of 3.1 to 3.2 strides (a median of 4.6 to
  4.7 s into the episode), F25F9 after 3.7 to 3.8 strides and V0 after 4.2 to 7.4 strides.

## 6. Speed

The ladder on the grippy floor, both controller fits. `nominal` is each set's slow walk (0.243 m/s on V0 and the
candidate). Cells as in section 5.

{{LADDER}}

- **Today's walk, no delay:** every limit is met in 20 of 20 at 0.24 and 0.30 m/s. At 0.35 and 0.40 m/s the tail
  servo fails in 20 of 20 and one episode per record also leaves its line (19 of 20 pass but for A1). At 0.45 m/s
  the duty limit fails as well. At 0.50 m/s duty fails under both fits, and under `R2/43` the walk also misses its
  distance (11 of 20), its toe clearance (16) and an ankle drive's peak speed (16).
- **Today's walk, 4 ms:** the slow walk (0.24 m/s) fails toe clearance in 20 of 20 (median 18 mm). At 0.30 m/s 17
  and 12 of 20 meet every limit, so one fit passes and one does not. 0.35 m/s is the only rung that passes under
  both fits, and only but for A1 (20 and 18).
- **Today's walk, 6 ms:** it stays up 20 of 20 from 0.24 to 0.40 m/s and meets no limit set: the toe clearance is
  under the limit in 20 of 20.
- **By Stage 1's reading** of the same episodes, today's walk meets every limit in 20 of 20 at 0.24 and 0.30 m/s
  at 0, 4 and 6 ms. That reading is recorded for comparison and gives no verdict.
- **The candidate** stays up 20 of 20 at every speed at 0 and 4 ms and passes nothing as registered (section 8). At
  6 ms it stays up at 0.24 to 0.35 m/s (14 to 20 of 20) and falls in every episode at 0.40 m/s and faster.
- **At 10 ms the ladder says nothing about speed.** No rung of either set stays up in more than 2 of 20. The
  candidate falls in its start from standing in every episode. Today's walk falls in the start or the slow walk:
  on the slow rung 27 of its 37 falls are in the 0.24 m/s walk itself and 10 in the start, and on the faster rungs
  the falls come before the rung's own walk begins.

## 7. Turns, pushes and the stop

Each episode walks at 0.40 m/s first. Grippy floor, both controller fits; cells as in section 5, with the falls
split into before and after the command (b / a). "n.r." in the middle place means not read: today's walk's quick
turn is built from a reference whose planned tail torque is over the planning limit, so by the rules only its
reading but for A1 can pass. The slippery floor and the `R0` rows are in the digest and in `BAR_TABLES.md`.

{{REST}}

Simulated, today's walk (V0):

- **Stop.** The two-step stop meets every limit in 20 of 20 at 0, 4 and 6 ms on both floors: two touchdowns after
  the command, forward motion over after a median of 1.24 to 1.29 strides, 0.23 to 0.24 m covered. The stop's rule
  does not judge toe clearance.
- **Quick turn.** It stays up 20 of 20 at 0, 4 and 6 ms and has turned a median of 82 to 85° at the judged instant,
  2 s after the command (commanded 90°; the heading limit is 10°). It does not pass at any delay, not even with
  none. With no delay the failing
  limits are the tail servo (20 of 20), duty (20 of 20), in 3 to 10 episodes a drive's peak speed, toe clearance
  and heading, and in 1 to 3 a moment with both feet off the floor. At 4 ms 7 to 12 of 20 episodes meet the other
  limits but for A1, against the 17 the rule asks.
- **Pushes.** With no delay it stays up 19 or 20 of 20 for every direction, and 11 to 15 of 20 meet every limit, so
  no push passes (the one record that passes anywhere is the forward push at 4 ms under one fit, 20 up and 19): the failing limits are a drive's peak speed, "back on the commanded speed within 3 s" and, in 1
  to 6 episodes, a servo's continuous load. At 4 and 6 ms 0 to 11 of 20 fall after the push, and 3 to 19 of 20 meet
  every limit.
- **At 10 ms** nothing is read: the robot falls before the command in 19 or 20 of 20 (17 to 20 on the slippery
  floor).

Simulated, the candidate:

- With no delay it stays up through the stop (20 of 20), the quick turn (20 of 20) and forward and backward pushes
  (17 to 20 of 20); a sideways push fells 5 of 20, where today's walk loses at most 1.
- It passes nothing as registered. With no delay, in the stop and the pushes the only failing limit among the
  episodes that stayed up is the joint range of its first steps (section 8); at 4 ms three push episodes fail
  another limit too. In the turn the toe clearance fails in every episode (15.5 to 15.8 mm against 20.07 mm with no
  delay), which the plan expected: the turn's reference itself reads 16.23 and 15.84 mm when played without
  physics. With no delay the duty limit fails in every episode of the turn as well.
- With no balance feedback (`R0`) the candidate falls in every episode of every quick turn, after the command;
  today's walk stays up 19 or 20 of 20 on the same turn.

## 8. The candidate, counted again with one limit set aside

**Label: a count made on 15 Nov after the walk records were read (`STAGE2_75_PLAN.md`, closing amendments, item
11). It is not a verdict and changes no rule.** The registered counts are in sections 5 to 7.

**What happened.** The walk rule judges the joint range at 0.10 rad (5.7°) from the end of each joint's travel, and
the runner reads it over the whole episode for the walk, the pushes and the stop. Every episode begins with the
set's first steps from standing, and the candidate's first steps come within 0.095 rad (5.4°) of the end of the
hip's sideways travel. Every walk, push and stop episode of the candidate that stayed up reads 0.0950 to 0.0952 rad,
under all three rows and at every delay, so the limit fails whatever the controller does. (A turn's joint range is
read over the turn only, by another rule, and does not fail.) Today's walk and the pull request's walk keep 0.127
and 0.136 rad and are not touched by this.

**The rules contradict themselves here, and that is the lead's error.** When the owner relaxed the *build* gate for
start-up moves to 0.05 rad (P34), the lead's amendment to the plan (A12) gave as the reason that in the closed loop
"the joint range is judged against the reference's own smallest margin less 0.02 rad (decision 14, unchanged in C4
and C5)". That is wrong for C4, the walk rule: C4 says 0.10 rad, and the runner applies 0.10 rad to walks, pushes
and stops and decision 14 to turns only. So the owner's decision P34 was taken on a sentence that was not true of
the walk rule, and the candidate's failure on this limit was not stated as an expectation before the runs. The
registered counts follow C4 and the runner as frozen. By the sentence in A12 the limit would have been 0.075 rad and
would not have failed; that is arithmetic on the rule, not a record.

The same records with that one limit set aside (`stage2/runs/2026-11-15-75cm/rereads/JOINT_RANGE_ASIDE.md`; counts
of 20, `R2/42` · `R2/43`). "Stage 1's reading" is Stage 1's own list of limits for a walk, which the pull request's
report uses; it is recorded for comparison and gives no verdict.

{{ASIDE}}

- **By this study's reading**, with the limit aside, the candidate's walk would meet "20 up and at least 17" under
  both fits at 0.30 and 0.35 m/s with no delay, and at no other speed or delay: at 0.40 m/s one fit has 8 of 20
  because the toe clearance sits on the limit, and with any delay the toe clearance fails in every episode. Its
  slow walk (0.24 m/s) fails the stride and cadence limits in every episode, apart from all this.
- **By Stage 1's reading**, as recorded, the candidate reads 0 of 20 everywhere, because Stage 1's reading holds the
  same joint-range limit. With it aside the candidate meets every limit in 20 of 20 at 0 ms from 0.30 to 0.50 m/s
  and at 2 ms at 0.40 and 0.50 m/s; at 4 ms it drops to 0 to 3 of 20 at 0.40 m/s and faster.
- **Today's walk under the same Stage 1 reading**, for a fair comparison: 20 of 20 at 0.24 and 0.30 m/s at 0, 4 and
  6 ms; at 0.35 and 0.40 m/s on the grippy floor the tail servo is the only limit that fails, through 6 ms, where
  the candidate has fallen.
- **The difference between the two readings is the toe clearance.** Stage 1 reads the foot's low point over the
  middle half of each reference swing, on the reference's clock. This study's bar reads the middle half of the
  planned swing time counted from the foot's own lift-off. On the candidate at 0.40 m/s with no delay the two give
  23.8 to 24.0 mm and 20.0 to 20.6 mm.
- **Stop and pushes:** with the limit aside the candidate's stop would read 20 of 20 at 0 and 4 ms. Of its pushes
  only the forward push under one fit would meet the rule with no delay (20 up, 20 met); the others lose 1 to 5
  episodes to falls.

Nothing was replicated on fresh seeds: the rule replicates what the candidate passes, and it passed nothing.

## 9. The pull request's claims beside these records

The pull request's report (`stage2/jetson_plem/JETSON_PLEM_AND_SPEED.md`, section 2.3) has a table of ten lines.
Its candidate, added-mass, stop and push lines have no code and no record behind them; its two lines on today's walk
restate Stage 1's records (`JETSON_PLEM_REVIEW.md`). `BAR_TABLES.md` section 6 sets each line beside this round's
records, seed by seed. In short, with the right-hand column simulated here:

| Pull request's claim (its line) | This round's records |
|---|---|
| Today's walk at 0.35 m/s, no delay: stays up 20 of 20, fails only the tail servo (92) | Agrees: 20 of 20 up; 0 of 20 meet every limit; 19 of 20 but for A1 |
| Today's walk at 0.50 m/s, no delay: stays up, fails tail servo, duty and ankle speed (93) | Partly: 20 of 20 up; tail servo and duty fail under both fits; under one an ankle drive's peak speed (16 of 20), and also distance (11), toe clearance (16) and a moment off the floor (4), which the pull request does not list |
| Its walk at 0.50 m/s, duty 0.55: one controller passes (94) | Not run: under this round's rules that walk cannot be reached from the set's 0.40 m/s walk |
| Candidate at 0.40, 0.45 and 0.50 m/s, no delay, both floors: 20 of 20 meet every limit (95) | As registered: 0 of 20. By Stage 1's reading with the joint range set aside (after the result, no verdict): 20 of 20 in all six cells. By this study's reading with it set aside: no cell passes under both fits |
| Candidate at 2 ms, 0.40 and 0.50 m/s: 20 of 20 (98) | As registered: 0 of 20. By Stage 1's reading with the limit set aside: 20 of 20 in both cells |
| Candidate at 4 ms loses toe clearance (5 and 1 of 20); at 6 to 10 ms it falls 0 of 20 (99) | Close: by Stage 1's reading with the limit set aside, 1 and 0 of 20 at 0.50 m/s at 4 ms. At 6 ms and at 10 ms 0 of 20 stay up at 0.40 and 0.50 m/s |
| Two-step stop at 2 ms: 20 of 20 (100) | 2 ms was not run for the stop, nor a stop from 0.50 m/s. At 0 and 4 ms from 0.40 m/s: 20 of 20 up, the joint range the only limit failing |
| Pushes at 2 ms: forward 20 of 20 up, left 15 of 20 up (101) | 2 ms was not run for the pushes, nor pushes at 0.50 m/s. At 0 ms: forward 20 and 19 up, left 15 and 15 up |
| 0.55 m/s; a body with the Jetson's added mass (96, 97) | Not run in this round |

The pull request's rows at 0.496 m/s stand beside this round's 0.50 m/s rung.

**What this adds up to.** Of the ten lines, the two on today's walk agree with the records, one of them in part.
Two candidate lines (95, 98) agree only under Stage 1's reading with the joint-range limit set aside, which is a
count made after the result and no verdict; a third (99) is close. The stop and the pushes were run at other
delays. Three lines were not run. What this round adds: under this study's toe-clearance reading the candidate is
at the limit with no delay and under it from 2 ms, and it falls at 6 ms where today's walk still stands.

## 10. Turning room

{{TURN_SECTION}}

## 11. Draft decision-log rows (proposed, not adopted)

| Row | Kind | Text |
|---|---|---|
| F28 (draft) | Finding | 75 cm agility bar, first panels (simulation, 15 Nov 2026; `stage2/runs/2026-11-15-75cm/`). Under Stage 1's fitted tracker (fits 42 and 43, made with no delay) none of the 16 conditions is shown at 10 ms on `V0` or `F25F9cl`, and neither of the two walk conditions run on `F25F9`: 0 to 3 of 20 episodes stay up (0 of 20 at 8 ms in the walk records), and in the walk records every 10 ms fall comes before the 0.40 m/s walk begins (candidate and `F25F9`: in the start from standing; V0's slow walk alone also falls, 27 of 37 falls in the slow walk itself); in V0's turn, push and stop records 17 to 20 of 20 fall before the command. At 0.40 m/s all three stay up 20 of 20 to 4 ms; V0 to 6 ms. V0 meets every limit at 0.24 and 0.30 m/s at 0 ms, and every limit but A1 at 0.35 m/s (0 and 4 ms) and 0.40 m/s (0 and 2 ms, both floors), the tail servo at 1.01 to 1.05 and 1.05 to 1.11 of its planning limit in those records. V0's two-step stop passes at 0, 4 and 6 ms; no quick-turn or push condition passes at any delay. `F25F9cl` passes nothing as registered. Budget used: 4.25 of 16 and 5.51 of 12 core-hours, nothing cut. |
| F29 (draft) | Finding | Two of the candidate's lines in pull request #668's table (section 2.3, lines 95 and 98) agree with these records only under Stage 1's reading with the start-up joint-range limit set aside (a count made after the result, no verdict): 20 of 20 at 0 and 2 ms at 0.40 and 0.50 m/s. Line 99 is close (0 to 3 of 20 at 4 ms; falls at 6 ms). As recorded, Stage 1's reading gives 0 of 20. Under this study's toe-clearance reading the candidate is at the limit at 0 ms (20.0 to 20.6 mm against 20.07 mm) and under it from 2 ms. Its tail servo reads 0.40 to 0.45 of the planning limit at 0.40 m/s against 1.02 to 1.11 on V0. |
| E (draft) | Error | The lead's amendment to `STAGE2_75_PLAN.md` A12 (P34, 15 Nov, before the freeze) said the closed loop judges the joint range by decision 14, "unchanged in C4 and C5". That is wrong for C4, which says 0.10 rad and which the runner reads at 0.10 rad over the whole episode for walks, pushes and stops. `F25F9cl`'s start (0.095 rad of hip-roll margin) therefore fails that limit in every such episode whatever the controller, and this was not stated as an expectation before the runs. Found after the walk records; noted in the plan (closing amendments, item 11); no rule changed; the registered counts follow C4 and the runner. |

## 12. Records, and how to make the tables again

- Records: `stage2/runs/2026-11-15-75cm/<set>/75-M-H-10/seeds3042/eval/<row>@<delay>ms/<behaviour>/<condition>.json`;
  the control's records under `controls/omega_s1/`; the ledger `budget.json`; the rates `COST.md`.
- Machine-made tables: `python -m designlab.stage2.barreport --out stage2/runs/2026-11-15-75cm` writes
  `BAR_TABLES.md` and `bar_tables.json` there.
- Digest: `python stage2/runs/2026-11-15-75cm/digest/digest.py`.
- The count with one limit set aside: `python stage2/runs/2026-11-15-75cm/rereads/term_aside.py`.
- Turning figures and clips: `python -m designlab.stage2.barviews turnpath --set <set>` and `... clips ...`
  (`stage2/runs/2026-11-15-75cm/views/`).
- The sets as frozen: `stage2/sets/<set>/75-M-H-10/set.json`.

## 13. Added 16 Nov 2026: today's walk at longer delays

*Simulation; stand-in controller fitted with no delay; delays that are not the judged 10 ms; no verdict; added
after the 75 cm results. The owner asked for these runs (`DECISION_LOG.md` P37). Their rules were written and
independently reviewed before the first run (`STAGE2_75_ADDON_PLAN.md` section A). The records at 14, 16 and 20 ms
are the delays the owner approved (P37: 14, 16 and 20 ms). The records at 12 and 18 ms and the slow walk at 8 ms
are the lead's addition, not among the delays the owner approved. The slow walk and the no-feedback rows at 14, 16
and 20 ms are the lead's reading of "today's walk". The bar stays judged at 10 ms and no verdict of sections 1 to
12 changes.*

**The question.** Is 10 ms a special point for the stand-in controller, with the walk holding again at a longer
delay, or does delay simply get worse?

**The answer, in the words fixed before the runs.** A record is "up" when at least 17 of its 20 runs stay up,
"down" when at most 3 do, otherwise "mixed". A delay is up or down when all its records under the two controller
fits agree. Three answers were fixed in advance for each walk. "Delay simply gets worse": every delay run above
10 ms reads down. "A window": some delay above 10 ms reads up, so the falls around 10 ms sit between delays that
stay up. "Neither": no delay above 10 ms reads up and at least one is mixed.

- The 0.40 m/s walk: {{ADDON_PATTERN_FAST}}
- The slow walk (target 0.243 m/s): {{ADDON_PATTERN_SLOW}}
- The lead's design estimate, that the curve would mirror about 10 ms (12 ms like 8 ms, 14 like 6, and so on):
  {{ADDON_MIRROR}}

**In plain words.** At the longer delays that were run (12 to 20 ms in 2 ms steps), no run under the stand-in
controller met every limit. At 0.40 m/s no run covered the distance its limit asks; at 18 ms the robot stayed up in
18 and 19 of 20 runs under one of the two fits and in 6 and 4 under the other. The slow walk stayed up again at
20 ms (19 and 20 of 20) without covering its distance. These are counts, not a verdict.

Each cell below reads: runs that stayed up · of those, runs that also covered the distance the limit asks (10 s at
the target speed) · runs that met every limit · runs that met every limit but for A1 (the neck and tail servo
planning limit). All of 20. In the slow walk's table "Stage 1 speed" is the number of runs that also met Stage 1's
own speed limit; it decides nothing. Rows at 0 to 10 ms are the registered records of sections 5 and 6, apart from
the slow walk at 8 ms. For every pair of fits at 12 to 20 ms, on both walks and in both readings, the runner's own
word is "does not pass at that delay".

The 0.40 m/s walk:

{{ADDON_FAST}}

The slow walk:

{{ADDON_SLOW}}

What the tables show:

- **At 0.40 m/s no run covers the distance at any longer delay that was run.** At 12, 14, 16 and 20 ms 0 to 3 of
  20 runs stay up under both controller fits on both floors, as at 8 and 10 ms. At 18 ms the two fits differ: 6
  and 4 of 20 stay up under fit 42, and 18 and 19 of 20 under fit 43 (grippy and slippery floor). The runs that
  stay up under fit 43 at 18 ms cover a median of 3.90 and 3.87 m where the limit asks 4.00 m. Every one of them
  fails the drive speed limit (the largest reading is 1.90 to 1.95 times the limit, on an ankle drive) and toe
  clearance (median 0.6 to 0.7 mm against a limit of 20.1 mm), and their median peak ground force is 3.05 and
  3.00 body weights, against 1.46 and 1.50 under the same fit with no delay.
- **At 12, 14 and 16 ms these records say almost nothing about walking at 0.40 m/s itself.** Of 80 runs at each
  delay (two fits, two floors), 0, 4 and 2 reach the 0.40 m/s walk. Most falls come in the start from standing:
  11 to 20 runs in each record, with a median fall time of 4.2 to 5.2 s. At 18 ms 47 of 80 reach the 0.40 m/s
  walk and none of them falls. At 20 ms 71 reach it and 67 fall in it, with a median fall time per record of 14.4
  to 15.7 s into a 21 s run. On the grippy floor the slow walk's runs and the 0.40 m/s walk's runs are the same
  simulation until the first slow stride ends: at 12 to 16 ms, 110 of the 113 runs that fall in both have the same
  fall time, so the two tables largely count the same falls.
- **The slow walk holds 2 ms longer than was known.** At 8 ms it stays up in 19 of 20 runs under both fits and
  covers its distance. It fails toe clearance in all 19 (median 11.3 to 12.4 mm against 20.1 mm), so it meets
  every limit in none. At 10 to 16 ms 0 to 3 of 20 stay up.
- **At 18 and 20 ms the slow walk stays up again, outside its limits.** At 18 ms 10 of 20 runs stay up under
  fit 42 and 18 of 20 under fit 43. The 18 under fit 43 cover the distance and fail cadence, ground force, stride
  length and toe clearance; the 10 under fit 42 do not cover it. At 20 ms 19 and 20 of 20 stay up and none covers
  the distance: the medians are 2.40 and 2.41 m where the limit asks 2.43 m (0.240 to 0.241 m/s against 0.243 m/s).
  Every run at 20 ms also fails the drive speed limit (up to 1.98 to 1.99 times the limit, on an ankle drive), the
  ground force limit, stride length and toe clearance. No run at 18 or 20 ms meets every limit. The 20 ms result
  agrees with Stage 1's three runs at 20 ms (`DECISION_LOG.md` F26).
- **With the balance feedback off, the count that stays up does not move with delay.** The row with no feedback
  (`R0`) stays up in 18 to 20 of 20 runs at each delay run from 0 to 20 ms on the 0.40 m/s walk. There it meets
  every limit in no run at any delay: the duty limit and a servo's continuous-load limit fail in every run that
  stays up. On the slow walk it stays up in 20 of 20 and meets every limit in all 20, at each of 12, 14, 16, 18
  and 20 ms. That row was not run with pushes here, and a playback with no feedback cannot correct a disturbance.

**The expectations written before the runs** (rule A-6's own numbers).

- Expectation 1, the no-feedback row stays up in 18 to 20 of 20 at every delay on the 0.40 m/s walk: met. No flag
  was raised.
- Expectation 2, the slow walk under fit 43 at 20 ms reads up: met. It reads up, short of the distance.
- Expectation 3 stated that nothing could be expected for the other new records.
- Expectation 4, no add-on record meets every limit in 17 of 20: **not met as written.** Five records do, all of
  them the no-feedback row on the slow walk. The lead wrote this expectation with the controller's rows in mind
  and did not say so. Under the controller's rows no run meets every limit at 12 to 20 ms.
- The mirror estimate: not met. No other point of symmetry is tried after the result.

**What this means for the plan** (the lead's reading; an interpretation, not a result).

- These runs give no reason to judge at another delay. At 0.40 m/s the stand-in controller reads down at 8 to
  16 ms and at 20 ms, and mixed at 18 ms; at none of them does a run cover the distance. Delays above 20 ms and
  between the 2 ms steps were not run.
- Survival does not simply fall with delay either. At 18 ms runs stay up again under fit 43 (18 and 19 of 20 at
  0.40 m/s, 18 of 20 on the slow walk). At 20 ms the slow walk stays up under both fits (19 and 20 of 20) and the
  0.40 m/s walk does not (1 of 20 in each record). The one estimate the lead wrote down in advance for this is
  refuted.
- The falls come with the feedback on: the same motion played with no feedback stays up in 18 to 20 of 20 runs at
  every delay that was run. The lead reads this as the feedback acting late; these runs do not test that. A
  controller trained or fitted with delay is the remedy the lead takes from it. It is not shown to work: a learned
  add-on trained with 0 to 20 ms of delay on this body (three runs; `designlab/stage1/env.py:235`) and a refit
  with delay on the 60 cm bodies both still fell at 10 ms (`DECISION_LOG.md` F26, F27).
- For the buy gate (`BUY_GATE_DRAFT.md`, a draft): with this controller, today's walk at 0.40 m/s stays up in 20
  of 20 runs at 0, 2, 4 and 6 ms and in none at 8 ms. Staying up is not passing: at 6 ms it fails toe clearance in
  19 or 20 of 20 runs. The bench's measurement of the real delay matters as before.

**Compute and records.** All six units ran and none was cut: 47 records, 940 runs, 0.495 of the section's
1.0 core-hour (`stage2/runs/2026-11-16-75cm-addon/A/ADDON_LEDGER.json`). The records are under
`stage2/runs/2026-11-16-75cm-addon/A/`. The tables, with where the falls come, the measures of the runs that
stayed up and every record's path, are `stage2/runs/2026-11-16-75cm-addon/DELAYS_TABLE.md`, made by
`python stage2/runs/2026-11-16-75cm-addon/delays_table.py`. In the tables here the column "Records" is reworded by
the report's assembler; the cells are that file's. The reviewer of this section found four cases that the table
script would settle beyond or apart from rule A-6's text (an unread `R0` record, a window without 8 and 10 ms
down, a delay with no word in the mirror result, `R0` falls left out of the table of falls); none occurs in these
records (`stage2/design/addon/review_sec13.md`, finding 11).
