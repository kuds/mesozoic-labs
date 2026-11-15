# The 75 cm build in simulation: the first panels of the agility bar

15 Nov 2026. Body `75-M-H-10`. **Simulation only: nothing in this report was built or measured on hardware.** Nothing
is adopted by it. Section 3 lists what is for the owner to decide.

**Status: revised on 15 Nov, 23:45 UTC, after three independent reviews (every number recomputed from the records; every claim checked against the rules and sources; a plain reading as the owner). A check of this revision is under way. The turning section (section 10 and the line in section 1) is added when the reviewed turning figures are merged.**

Paths are relative to `hardware/trex_form_study/`. Every count below is read from the committed records under
`stage2/runs/2026-11-15-75cm/`; where anything here differs from the records, the records decide.

## 1. The short answer

**Will the 75 cm build walk?** In simulation, with today's walk and the balance controller carried over from
Stage 1:

- **It walks without falling when the control delay is 6 ms or less.** At 0.24 to 0.40 m/s it stays up in 20 of 20
  runs at every delay tried up to 6 ms. At 0.40 m/s it falls in every run at 8 ms and in 17 to 20 of 20 at 10 ms.
- **At 10 ms, the delay the study assumes until one is measured, nothing passes.** The robot falls before it even
  reaches its 0.40 m/s walk: while starting from standing, in its first slow stride, or while speeding up.
- **The real delay is not measured.** The design estimate for the planned wiring is 2 to 4 ms, or 2 to 6 ms on the
  cautious reading. It is arithmetic, not a measurement.
- **Staying up is not the same as passing.** Today's walk meets every limit at 0.24 and 0.30 m/s only with no delay
  at all. At 0.35 and 0.40 m/s its tail servo runs at up to 1.11 times its continuous rating. From 4 ms its toe
  clearance (how high the foot is above the floor in mid-step) drops to the 20 mm limit and below.
- **It is not yet robust.** No push test and no quick-turn test passes at any delay. With no delay a light push
  (about 22 N for a tenth of a second) is survived in 19 or 20 of 20 runs on the grippy floor; at 4 ms a sideways
  or backward push fells it in 3 to 8 of 20.
- **The stop works.** The two-step stop meets every limit in 20 of 20 runs at 0, 4 and 6 ms, on both floors.
- **This is the stand-in controller.** It was fitted with no delay. The controller the bar is meant to judge, a
  policy trained with delay, has not been trained or run on this body yet.

Runs that stayed up, of 20, walking at 0.40 m/s on the grippy floor (the two fits of the controller, side by side):

| Walk | 0 ms | 2 ms | 4 ms | 6 ms | 8 ms | 10 ms |
|---|---|---|---|---|---|---|
| Today's walk | 20 · 20 | 20 · 20 | 20 · 20 | 20 · 20 | 0 · 0 | 0 · 1 |
| The pull request's walk | 20 · 20 | 20 · 20 | 20 · 20 | 15 · 19 | 0 · 0 | 0 · 2 |
| The candidate walk | 20 · 20 | 20 · 20 | 20 · 20 | 0 · 0 | 0 · 0 | 0 · 0 |

**The two other walks do worse.** The candidate walk and the pull request's walk pass nothing. They do cut the tail
servo's load to 0.40 to 0.48 of its rating, which is the one limit today's walk fails at 0.35 and 0.40 m/s. Section 8
explains a fault in the rules, made by the lead, that fails the candidate on one limit before its controller matters.

**Turning room.** [to be added when the reviewed views are merged: TURN_ANSWER]

**Compute.** The walk panels used 4.25 of their 16 core-hours and the turn, push and stop panels 5.51 of their 12.
Nothing was cut. 592 records were written under the three walks (10 of them by the runner as "not run", with a
reason) and 2 records of a registered control.

## 2. What this does not show

- **Nothing about hardware.** No delay, friction or gear play of the real drives is measured. The delay estimate is
  from `stage2/jetson75/LATENCY_AND_BUS.md`: 0.3 to 3.5 ms for a microcontroller with two CAN buses, which maps
  onto 2 to 4 ms of simulated delay, or 2 to 6 ms if the drives' own loop is the slower of two published figures.
  It leaves out timing jitter.
- **Only the stand-in controller.** Both controller rows are Stage 1's balance controller, fitted with no delay.
  The plan's judged controller is a policy trained with the delay drawn at random from 0 to 24 ms (`STAGE2_PLAN.md`
  section 4). A refit of the stand-in with delay in its mix is also planned. One such refit, on the 60 cm bodies
  (one fit), still fell at 10 ms (`DECISION_LOG.md` F27); it has not been tried on this body, which unlike the
  60 cm bodies walks with the balance feedback switched off.
- **A simple servo model.** Ideal position servos behind a torque and speed limit and a delay; no friction model
  and no gear play.
- **Not run in this round, by plan:** the turns, pushes and stop at 2 and 8 ms; the turn, push and stop tests of
  the pull request's walk; a stop from 0.50 m/s; delays above 10 ms; carpet; the look gates; the body with the
  Jetson's added mass; 0.55 m/s; Stage 1's twelve no-loss conditions; the turn and pushes from the slow walk; any
  60 cm body. `BAR_TABLES.md` section 9 lists each.
- **The slower turns, stronger pushes and one-step stop read nothing.** They were run at 10 ms alone, where the
  robot falls before the command.
- **The toe-clearance reading is this study's own**, changed after a result and labelled so in the plan before any
  walk was frozen. It reads lower than Stage 1's on every walk here (section 5).

## 3. For the owner

Nothing below is adopted. Each question gives the choices, what each costs, the lead's pick and what happens if
there is no answer.

**Question 1. What to do about the judged delay.** The bar is judged at an assumed 10 ms until a bench value
exists, and the bench waits for a build that works in simulation. With the stand-in controller no walk passes at
10 ms.

- (a) Keep 10 ms and wait for the policy trained on Colab. Nothing new to decide or run.
- (b) The same, and first refit the stand-in controller with delay on this body, here, in simulation. It is the
  quickest test of whether 10 ms is within reach of this kind of controller: two fits, rules written first, a few
  core-hours.
- (c) Also read a second, labelled verdict at 4 ms or 6 ms, as an assumption from the design estimate. On today's
  records the 0.40 m/s walk, the quick turn and the pushes would still not pass there (only the stop would), and it
  changes a rule after a result.
- (d) Bring the delay bench forward. Its first run needs about $2,000 of catalogue parts ($2,400 with the
  controller boards), before the simulation has chosen a build.
- **Lead's pick: (b).** If there is no answer: (a).

**Question 2. The tail servo at 0.35 m/s and faster.** With no delay today's walk loads it at 5 % over its
continuous rating at 0.35 m/s, 8 % at 0.40 m/s (10 to 11 % on the slippery floor), 12 to 13 % at 0.45 m/s and 19 to
28 % at 0.50 m/s. That is heat over minutes, not a fall.

- (a) Cruise at 0.30 m/s. With no delay every limit is met; at 4 ms one of the two controller fits passes.
- (b) Wait for the look the owner already chose (V6), whose tail swings in counter-phase. It is not in today's
  walk as run here, and its tail load on this body has not been read. It comes with the look work already planned.
- (c) Test a smaller tail swing or a stronger tail servo in simulation.
- **Lead's pick: (b), then (c) if the load is still over.** If there is no answer: (b).

**Question 3. Keep the candidate walk in the 75 cm study?** On these records today's walk is ahead on everything
but the tail servo's load (section 8).

- (a) No. Record what it showed and stop spending on it.
- (b) Yes, and rebuild its first steps from standing with more hip room. A new freeze, and a few core-hours to
  run its panels again.
- (c) Yes, and change how the joint range is read for walks, pushes and stops. A rule change after a result, and
  the second easing for the same move (P34 was the first).
- **Lead's pick: (a).** If there is no answer: nothing more is run on it.

**Question 4. Pull request #668.** Two of its table's ten lines (the candidate at 0 and 2 ms) agree with these
records only under its own reading and with one limit set aside after the result; one more is close; two match
today's walk; five were not run as claimed (section 9).

- (a) Leave it open as it is.
- (b) Close it with a pointer to this report. Its walk already lives in this branch as the sets `F25F9` and
  `F25F9cl`, with its table checked line by line.
- (c) Merge it by itself. Its report would then stand in the repository without these corrections.
- **Lead's pick: (b), with a dated note at the top of its report in this branch pointing here.** If there is no
  answer: (a).

**Question 5. Carpet.** No sourced carpet model could be written (`stage2/design/carpet_sources_15nov.md`).

- (a) Run today's walk on a softer and on a grippier floor model as a range, with no verdict. About 1 core-hour.
- (b) A short bench measurement on a carpet sample (a luggage scale and a ruler), which turns the range into one
  point.
- (c) Leave carpet until a controller passes on the hard floor.
- **Lead's pick: (a) at 0 to 4 ms of delay.** If there is no answer: (a), as already asked for (P36).

**The lead's reading of the results** (an interpretation, not a result).

- **What limits the build in these runs is how much delay this controller tolerates: 4 to 6 ms.** Whether that is
  the delay as such or this particular fit is not shown. Three things point at the fit. With the balance feedback
  switched off the same walks stay up at 10 ms (but cannot take a push). The 10 ms falls come in the start from
  standing and the speed-up, not in the walk. And in Stage 1 this body fell at 10 ms and stayed up at 20 ms (three
  runs of the slow walk; `DECISION_LOG.md` F26).
- **Delay is not the only shortfall.** With no delay today's walk passes 4 of the bar's 16 conditions: the walk on
  both floors (with the tail servo's limit left out) and the stop on both floors. The quick turn and the pushes
  pass at no delay.
- **Today's walk is ahead on everything but the tail servo's load.** It is the only walk with passes, it stays up
  at 6 ms where the candidate falls, it keeps about 4 mm more toe clearance, and it takes sideways pushes better.
- **An outside reference, not a result of this study.** The open-source Microduck robot trains its walking policy
  through 15 to 30 ms of command delay drawn at random, with a detailed servo model and numbers checked on a servo
  bench (`stage2/research/microduck_delay_15nov.md`; read, not run). That is the remedy this study's plan already
  holds for its trained policy. It suggests that 10 ms need not rule out walking; it shows nothing about this body.

## 4. What was run, and the words used

**Four kinds of statement are kept apart.** *Simulated*: the robot model walking under a controller in the physics
simulation; every count and measured median here. *Planned*: a reference motion played without physics; the turning
circle's planned numbers and "the reference's own toe clearance". *Design estimate*: arithmetic on catalogue facts.
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

**Where the falls come.** The falls of the two controller fits on both floors together (80 episodes per cell), by
the part of the episode the robot had reached:

| Set | Delay | Falls of 80 | Where (falls) |
|---|---|---|---|
| V0 | 6 ms | 0 | - |
| V0 | 8 ms | 80 | the slow stride after the start: 5; the change of speed to 0.40 m/s: 39; the 0.40 m/s walk itself: 36 |
| V0 | 10 ms | 76 | the start from standing: 29; the slow stride after the start: 12; the change of speed to 0.40 m/s: 35 |
| F25F9 | 6 ms | 17 | the start from standing (straight into 0.40 m/s): 7; the 0.40 m/s walk itself: 10 |
| F25F9 | 8 ms | 80 | the start from standing (straight into 0.40 m/s): 72; the 0.40 m/s walk itself: 8 |
| F25F9 | 10 ms | 78 | the start from standing (straight into 0.40 m/s): 78 |
| F25F9cl | 6 ms | 80 | the slow stride after the start: 1; the change of speed to 0.40 m/s: 51; the 0.40 m/s walk itself: 28 |
| F25F9cl | 8 ms | 80 | the start from standing: 73; the slow stride after the start: 5; the change of speed to 0.40 m/s: 2 |
| F25F9cl | 10 ms | 80 | the start from standing: 80 |

What the tables show (simulated):

- **10 ms.** Under both controller fits 0 to 3 of 20 stay up, and no fall is in the 0.40 m/s walk itself: every one
  comes earlier. The candidate and the pull request's walk fall in their start from standing.
- **8 ms.** 0 of 20 stay up in all twelve records. Today's walk gets further: 36 of its 80 falls are in the
  0.40 m/s walk and 39 in the speed-up. The other two still fall mostly in the start.
- **6 ms.** Today's walk does not fall. The candidate falls in every episode, in the speed-up (51) or in the
  0.40 m/s walk (28).
- **No balance feedback.** With `R0` the same walks stay up 18 to 20 of 20 at every delay and meet no limit set.
- **Expected before the runs.** The plan said, before any walk was frozen, that "the 10 ms verdicts may be 'not
  shown' because the robot falls, whatever the walk". That came true at 10 ms and at 8 ms. It also said the
  every-limit reading might fail on today's walk for the tail servo alone. That came true at 0.35 and 0.40 m/s at
  0 and 2 ms, where the tail servo is the only limit that fails in 20 of 20.
- **The two floors give the same picture for the walk.** The largest difference between a cell above and its twin
  on the other floor is 5 episodes (V0, `R2/42`, 4 ms, but for A1: 16 against 11).
- **The registered control** (40 episodes, no verdict): the candidate's walk under `R2/42` with the controller's
  pendulum frequency computed as Stage 1 computed it. At 10 ms 0 of 20 stay up, as in the registered record; at 0 ms
  20 stay up and 0 meet every limit.

Measured medians on the grippy floor, both controller fits, 0 to 6 ms. Speed is the simulated median over the last
10 s (the target is 0.40 m/s; the reference is played 5 % fast, at 0.42 m/s). Toe clearance is read as the bar
judges it, against a limit of 20.07 mm; Stage 1's reading of the same episodes stands beside it and is not judged.
Duty is the mean of the two feet; its limits are 0.55 to 0.65. The servo columns are continuous load over rating;
the limit is 1.0.

| Set | Row | Delay | Speed (m/s) | Toe clearance (mm) | Toe clearance, Stage 1's reading (mm) | Duty | Tail servo | Neck servo | Leg servo | Failed terms (episodes of 20) |
|---|---|---|---|---|---|---|---|---|---|---|
| V0 | `R2/42` | 0 ms | 0.410 (20) | 24.2 (20) | 27.1 | 0.563 | 1.07 | 0.70 | 0.80 | `drive_sustained` 20, `line` 1 |
| V0 | `R2/42` | 2 ms | 0.412 (20) | 22.9 (20) | 27.2 | 0.573 | 1.05 | 0.70 | 0.80 | `drive_sustained` 20 |
| V0 | `R2/42` | 4 ms | 0.412 (20) | 20.7 (20) | 25.2 | 0.583 | 1.04 | 0.70 | 0.79 | `drive_sustained` 20, `toe_clearance_own_swing` 4 |
| V0 | `R2/42` | 6 ms | 0.409 (20) | 18.9 (20) | 23.0 | 0.583 | 1.02 | 0.70 | 0.79 | `drive_sustained` 20, `toe_clearance_own_swing` 20 |
| V0 | `R2/43` | 0 ms | 0.411 (20) | 24.2 (20) | 28.7 | 0.560 | 1.08 | 0.70 | 0.80 | `drive_sustained` 20, `line` 1 |
| V0 | `R2/43` | 2 ms | 0.413 (20) | 23.1 (20) | 30.8 | 0.571 | 1.06 | 0.70 | 0.80 | `drive_sustained` 20 |
| V0 | `R2/43` | 4 ms | 0.414 (20) | 20.2 (20) | 26.9 | 0.584 | 1.05 | 0.70 | 0.79 | `drive_sustained` 20, `toe_clearance_own_swing` 7, `line` 1 |
| V0 | `R2/43` | 6 ms | 0.413 (20) | 18.3 (20) | 23.4 | 0.582 | 1.03 | 0.71 | 0.80 | `drive_sustained` 20, `toe_clearance_own_swing` 20 |
| F25F9 | `R2/42` | 0 ms | 0.413 (20) | 17.2 (20) | 20.9 | 0.559 | 0.44 | 0.69 | 0.75 | `duty` 20, `toe_clearance_own_swing` 20, `line` 1 |
| F25F9 | `R2/42` | 2 ms | 0.413 (20) | 14.6 (20) | 19.5 | 0.573 | 0.42 | 0.69 | 0.75 | `toe_clearance_own_swing` 20, `duty` 1, `line` 1 |
| F25F9 | `R2/42` | 4 ms | 0.412 (20) | 12.7 (20) | 16.9 | 0.581 | 0.42 | 0.69 | 0.75 | `toe_clearance_own_swing` 20, `line` 1 |
| F25F9 | `R2/42` | 6 ms | 0.410 (15) | 12.6 (15) | 14.5 | 0.575 | 0.48 | 0.69 | 0.76 | `toe_clearance_own_swing` 15, `survived` 5, `line` 1 |
| F25F9 | `R2/43` | 0 ms | 0.412 (20) | 17.8 (20) | 20.9 | 0.551 | 0.45 | 0.69 | 0.76 | `duty` 20, `toe_clearance_own_swing` 20, `line` 1 |
| F25F9 | `R2/43` | 2 ms | 0.415 (20) | 15.0 (20) | 19.7 | 0.567 | 0.43 | 0.69 | 0.75 | `toe_clearance_own_swing` 20, `duty` 18, `line` 1 |
| F25F9 | `R2/43` | 4 ms | 0.415 (20) | 12.0 (20) | 16.9 | 0.579 | 0.43 | 0.69 | 0.75 | `toe_clearance_own_swing` 20, `line` 1 |
| F25F9 | `R2/43` | 6 ms | 0.415 (19) | 11.8 (19) | 14.8 | 0.574 | 0.47 | 0.70 | 0.76 | `toe_clearance_own_swing` 19, `duty` 4, `line` 1, `survived` 1 |
| F25F9cl | `R2/42` | 0 ms | 0.415 (20) | 20.0 (20) | 23.8 | 0.593 | 0.42 | 0.69 | 0.76 | `joint_range` 20, `toe_clearance_own_swing` 11, `line` 1 |
| F25F9cl | `R2/42` | 2 ms | 0.416 (20) | 17.0 (20) | 23.4 | 0.607 | 0.40 | 0.69 | 0.75 | `joint_range` 20, `toe_clearance_own_swing` 20 |
| F25F9cl | `R2/42` | 4 ms | 0.413 (20) | 13.7 (20) | 19.6 | 0.618 | 0.43 | 0.70 | 0.76 | `joint_range` 20, `toe_clearance_own_swing` 20, `line` 1 |
| F25F9cl | `R2/42` | 6 ms | - (0) | - (0) | - | - | - | - | - | `survived` 20 |
| F25F9cl | `R2/43` | 0 ms | 0.414 (20) | 20.6 (20) | 24.0 | 0.587 | 0.42 | 0.70 | 0.77 | `joint_range` 20, `toe_clearance_own_swing` 2 |
| F25F9cl | `R2/43` | 2 ms | 0.417 (20) | 16.8 (20) | 23.5 | 0.605 | 0.41 | 0.69 | 0.76 | `joint_range` 20, `toe_clearance_own_swing` 20, `line` 1 |
| F25F9cl | `R2/43` | 4 ms | 0.416 (20) | 13.0 (20) | 19.5 | 0.620 | 0.44 | 0.70 | 0.76 | `joint_range` 20, `toe_clearance_own_swing` 20, `line` 1 |
| F25F9cl | `R2/43` | 6 ms | - (0) | - (0) | - | - | - | - | - | `survived` 20 |

- **Toe clearance falls with delay on every walk**: on V0 from 24.2 mm with no delay to 20.2 to 20.7 mm at 4 ms and
  18.3 to 18.9 mm at 6 ms; on F25F9 from 17.2 to 17.8 mm to 11.8 to 12.6 mm at 6 ms; on the candidate from 20.0 to
  20.6 mm to 13.0 to 13.7 mm at 4 ms.
- **Today's walk loses its passes at 4 and 6 ms on this study's toe reading.** At 4 ms the toe limit fails in 4 to 9
  of 20 episodes at 0.40 m/s (both floors), at 6 ms in 19 or 20. By Stage 1's reading of the same episodes the toe
  clears 22.8 to 26.9 mm at 4 and 6 ms (both floors).
- **The tail servo is over its rating only on today's walk** (1.02 to 1.08 at 0.40 m/s on the grippy floor, 1.05 to
  1.11 on the slippery floor). On both of the pull request's walks the tail reads 0.40 to 0.48 and the neck, at 0.69
  to 0.70, is the larger of the two. The neck reads 0.70 on today's walk as well.
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
- **At 10 ms the ladder says nothing about speed.** No rung of either set stays up in more than 2 of 20, and the
  falls come before the rung's own walk begins.

## 7. Turns, pushes and the stop

Each episode walks at 0.40 m/s first. Grippy floor, both controller fits; cells as in section 5, with the falls
split into before and after the command (b / a). "n.r." in the middle place means not read: today's walk's quick
turn is built from a reference whose planned tail torque is over the planning limit, so by the rules only its
reading but for A1 can pass. The slippery floor and the `R0` rows are in the digest and in `BAR_TABLES.md`.

| Set | Condition | Row | 0 ms | 4 ms | 6 ms | 10 ms |
|---|---|---|---|---|---|---|
| V0 | `nudge/backward` | `R2/42` | 20 / 13 / 13 (0 b / 0 a) | 14 / 13 / 13 (0 b / 6 a) | 14 / 10 / 10 (0 b / 6 a) | 0 / 0 / 0 (20 b / 0 a) |
| V0 | `nudge/backward` | `R2/43` | 20 / 13 / 13 (0 b / 0 a) | 12 / 10 / 10 (0 b / 8 a) | 9 / 8 / 8 (0 b / 11 a) | 1 / 0 / 0 (19 b / 0 a) |
| V0 | `nudge/forward` | `R2/42` | 19 / 14 / 14 (0 b / 1 a) | 20 / 19 / 19 (0 b / 0 a) | 19 / 19 / 19 (0 b / 1 a) | 0 / 0 / 0 (20 b / 0 a) |
| V0 | `nudge/forward` | `R2/43` | 20 / 14 / 14 (0 b / 0 a) | 14 / 12 / 12 (0 b / 6 a) | 15 / 13 / 13 (0 b / 5 a) | 0 / 0 / 0 (19 b / 1 a) |
| V0 | `nudge/left` | `R2/42` | 20 / 15 / 15 (0 b / 0 a) | 17 / 10 / 10 (0 b / 3 a) | 12 / 5 / 5 (0 b / 8 a) | 0 / 0 / 0 (20 b / 0 a) |
| V0 | `nudge/left` | `R2/43` | 20 / 11 / 11 (0 b / 0 a) | 14 / 6 / 6 (0 b / 6 a) | 15 / 5 / 5 (0 b / 5 a) | 1 / 0 / 0 (19 b / 0 a) |
| V0 | `nudge/right` | `R2/42` | 19 / 13 / 13 (0 b / 1 a) | 15 / 7 / 7 (0 b / 5 a) | 13 / 3 / 3 (0 b / 7 a) | 0 / 0 / 0 (20 b / 0 a) |
| V0 | `nudge/right` | `R2/43` | 20 / 13 / 13 (0 b / 0 a) | 15 / 9 / 9 (0 b / 5 a) | 14 / 4 / 4 (0 b / 6 a) | 0 / 0 / 0 (19 b / 1 a) |
| V0 | `stop/two_step` | `R2/42` | 20 / 20 / 20 (0 b / 0 a) | 20 / 20 / 20 (0 b / 0 a) | 20 / 20 / 20 (0 b / 0 a) | 0 / 0 / 0 (20 b / 0 a) |
| V0 | `stop/two_step` | `R2/43` | 20 / 20 / 20 (0 b / 0 a) | 20 / 20 / 20 (0 b / 0 a) | 20 / 20 / 20 (0 b / 0 a) | 1 / 0 / 0 (19 b / 0 a) |
| V0 | `turn/q90_left` | `R2/42` | 20 / n.r. / 0 (0 b / 0 a) | 20 / n.r. / 10 (0 b / 0 a) | 20 / n.r. / 1 (0 b / 0 a) | 0 / n.r. / 0 (20 b / 0 a) |
| V0 | `turn/q90_left` | `R2/43` | 20 / n.r. / 0 (0 b / 0 a) | 20 / n.r. / 7 (0 b / 0 a) | 20 / n.r. / 2 (0 b / 0 a) | 0 / n.r. / 0 (19 b / 1 a) |
| V0 | `turn/q90_right` | `R2/42` | 20 / n.r. / 0 (0 b / 0 a) | 20 / n.r. / 12 (0 b / 0 a) | 20 / n.r. / 3 (0 b / 0 a) | 0 / n.r. / 0 (20 b / 0 a) |
| V0 | `turn/q90_right` | `R2/43` | 20 / n.r. / 0 (0 b / 0 a) | 20 / n.r. / 9 (0 b / 0 a) | 20 / n.r. / 0 (0 b / 0 a) | 0 / n.r. / 0 (19 b / 1 a) |
| F25F9cl | `nudge/backward` | `R2/42` | 19 / 0 / 0 (0 b / 1 a) | 18 / 0 / 0 (0 b / 2 a) | 0 / 0 / 0 (19 b / 1 a) | 0 / 0 / 0 (20 b / 0 a) |
| F25F9cl | `nudge/backward` | `R2/43` | 17 / 0 / 0 (0 b / 3 a) | 12 / 0 / 0 (0 b / 8 a) | 0 / 0 / 0 (17 b / 3 a) | 0 / 0 / 0 (20 b / 0 a) |
| F25F9cl | `nudge/forward` | `R2/42` | 20 / 0 / 0 (0 b / 0 a) | 20 / 0 / 0 (0 b / 0 a) | 0 / 0 / 0 (19 b / 1 a) | 0 / 0 / 0 (20 b / 0 a) |
| F25F9cl | `nudge/forward` | `R2/43` | 19 / 0 / 0 (0 b / 1 a) | 19 / 0 / 0 (0 b / 1 a) | 0 / 0 / 0 (17 b / 3 a) | 0 / 0 / 0 (20 b / 0 a) |
| F25F9cl | `nudge/left` | `R2/42` | 15 / 0 / 0 (0 b / 5 a) | 12 / 0 / 0 (0 b / 8 a) | 0 / 0 / 0 (19 b / 1 a) | 0 / 0 / 0 (20 b / 0 a) |
| F25F9cl | `nudge/left` | `R2/43` | 15 / 0 / 0 (0 b / 5 a) | 12 / 0 / 0 (0 b / 8 a) | 0 / 0 / 0 (17 b / 3 a) | 0 / 0 / 0 (20 b / 0 a) |
| F25F9cl | `nudge/right` | `R2/42` | 15 / 0 / 0 (0 b / 5 a) | 6 / 0 / 0 (0 b / 14 a) | 0 / 0 / 0 (19 b / 1 a) | 0 / 0 / 0 (20 b / 0 a) |
| F25F9cl | `nudge/right` | `R2/43` | 15 / 0 / 0 (0 b / 5 a) | 11 / 0 / 0 (0 b / 9 a) | 0 / 0 / 0 (17 b / 3 a) | 0 / 0 / 0 (20 b / 0 a) |
| F25F9cl | `stop/two_step` | `R2/42` | 20 / 0 / 0 (0 b / 0 a) | 20 / 0 / 0 (0 b / 0 a) | 1 / 0 / 0 (19 b / 0 a) | 0 / 0 / 0 (20 b / 0 a) |
| F25F9cl | `stop/two_step` | `R2/43` | 20 / 0 / 0 (0 b / 0 a) | 20 / 0 / 0 (0 b / 0 a) | 1 / 0 / 0 (18 b / 1 a) | 0 / 0 / 0 (20 b / 0 a) |
| F25F9cl | `turn/q90_left` | `R2/42` | 20 / 0 / 0 (0 b / 0 a) | 20 / 0 / 0 (0 b / 0 a) | 1 / 0 / 0 (19 b / 0 a) | 0 / 0 / 0 (20 b / 0 a) |
| F25F9cl | `turn/q90_left` | `R2/43` | 20 / 0 / 0 (0 b / 0 a) | 19 / 0 / 0 (0 b / 1 a) | 1 / 0 / 0 (17 b / 2 a) | 0 / 0 / 0 (20 b / 0 a) |
| F25F9cl | `turn/q90_right` | `R2/42` | 20 / 0 / 0 (0 b / 0 a) | 19 / 0 / 0 (0 b / 1 a) | 1 / 0 / 0 (19 b / 0 a) | 0 / 0 / 0 (20 b / 0 a) |
| F25F9cl | `turn/q90_right` | `R2/43` | 20 / 0 / 0 (0 b / 0 a) | 19 / 0 / 0 (0 b / 1 a) | 1 / 0 / 0 (18 b / 1 a) | 0 / 0 / 0 (20 b / 0 a) |

Simulated, today's walk (V0):

- **Stop.** The two-step stop meets every limit in 20 of 20 at 0, 4 and 6 ms on both floors: two touchdowns after
  the command, forward motion over after a median of 1.24 to 1.29 strides, 0.23 to 0.24 m covered. The stop's rule
  does not judge toe clearance.
- **Quick turn.** It stays up 20 of 20 at 0, 4 and 6 ms and has turned a median of 82 to 85° at the judged instant,
  2 s after the command (commanded 90°; the heading limit is 10°). It passes at no delay. With no delay the failing
  limits are the tail servo (20 of 20), duty (20 of 20), in 3 to 10 episodes a drive's peak speed, toe clearance
  and heading, and in 1 to 3 a moment with both feet off the floor. At 4 ms 7 to 12 of 20 episodes meet the other
  limits but for A1, against the 17 the rule asks.
- **Pushes.** With no delay it stays up 19 or 20 of 20 for every direction, and 11 to 15 of 20 meet every limit, so
  no push passes: the failing limits are a drive's peak speed, "back on the commanded speed within 3 s" and, in 1
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

[to be added when the reviewed views are merged: TURN_SECTION]

## 11. Draft decision-log rows (proposed, not adopted)

| Row | Kind | Text |
|---|---|---|
| F28 (draft) | Finding | 75 cm agility bar, first panels (simulation, 15 Nov 2026; `stage2/runs/2026-11-15-75cm/`). Under Stage 1's fitted tracker (fits 42 and 43, made with no delay) none of the 16 conditions is shown at 10 ms on `V0` or `F25F9cl`, and neither of the two walk conditions run on `F25F9`: 0 to 3 of 20 episodes stay up, 0 of 20 at 8 ms, and every 10 ms fall comes before the 0.40 m/s walk begins (candidate and `F25F9`: in the start from standing). At 0.40 m/s all three stay up 20 of 20 to 4 ms; V0 to 6 ms. V0 meets every limit at 0.24 and 0.30 m/s at 0 ms, and every limit but A1 at 0.35 m/s (0 and 4 ms) and 0.40 m/s (0 and 2 ms, both floors), the tail servo at 1.01 to 1.05 and 1.05 to 1.11 of its rating in those records. V0's two-step stop passes at 0, 4 and 6 ms; no quick-turn or push condition passes at any delay. `F25F9cl` passes nothing as registered. Budget used: 4.25 of 16 and 5.51 of 12 core-hours, nothing cut. |
| F29 (draft) | Finding | Two of the candidate's lines in pull request #668's table (section 2.3, lines 95 and 98) agree with these records only under Stage 1's reading with the start-up joint-range limit set aside (a count made after the result, no verdict): 20 of 20 at 0 and 2 ms at 0.40 and 0.50 m/s. Line 99 is close (0 to 3 of 20 at 4 ms; falls at 6 ms). As recorded, Stage 1's reading gives 0 of 20. Under this study's toe-clearance reading the candidate is at the limit at 0 ms (20.0 to 20.6 mm against 20.07 mm) and under it from 2 ms. Its tail servo reads 0.40 to 0.45 of its rating at 0.40 m/s against 1.02 to 1.11 on V0. |
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
