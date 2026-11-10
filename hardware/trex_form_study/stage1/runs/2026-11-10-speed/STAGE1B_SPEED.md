# Stage 1b speed round: report to the owner

**10 Nov 2026, from the records as they stood at 11:05 UTC (final). Simulation only.** Sources: the machine-made tables
`STAGE1B_TABLES.md` and `stage1b_tables.json` (`speedreport`, 10:54 UTC) and the records behind them (Engineering notes,
N1–N11). "Decision n" is `STAGE1B_PLAN.md` section 6; P, F and A rows are in `DECISION_LOG.md`; "Stage 1" is the
previous round, reported in `stage1/runs/2026-11-08/STAGE1_PROBE.md`.

The bodies, in the order of every "a / b / c" figure: `60-N-T-10`, the 60 cm Tapered; `60-N-H-10`, the 60 cm Hybrid;
`60-N-T-10-K`, the Tapered rebuilt with round two's faster knee (the K body).

**Terms.** Each condition runs 20 episodes, simulated tries from a settled stance. An episode **stays up** if it never
falls (N6). It **meets every limit** if it also reaches its target (a walk's speed over the judged 10 s, a turn's turn
rate) and keeps the behaviour spec's limits: duty (each foot down 55–65 % of the stride), no flight (never both feet off
the floor), trunk motion, toe clearance, foot slip, joint range, each drive's sustained-torque planning limit and rated
speed, and, on the nominal walk and the turns, peak foot force and more (N6). A condition is **shown** if all 20 stay up
and at least 17 meet every limit, **survives only** if all stay up but fewer meet every limit, and **fails** otherwise.
A **fitted controller** is the balance tracker R2 (reference walks plus feedback from the robot's own sensors) with its
13 gains set by an automatic search (CMA-ES) from one random seed: one per body here; Stage 1's verdicts needed two, and
called a condition that only one passed **provisional**. **Servo delay** is the time from the controller setting a joint
target to the servo acting on it. **Attribution labels:** each hardware limit active just before the falls is removed in
turn over the same 20 starts; "body-limited" if one removal keeps the robot up significantly longer, otherwise
"controller-limited", the default under 5 falls. The removals are extreme and skip the tail and neck servos, so a label
is only a hint either way (N4). μ is the floor's friction coefficient; Stage 0, round two's design screens.

**A correction to Stage 1's report.** It said the 75 cm body's nominal walk stays up under at least 20 ms of servo
delay. Its own records show that walk falls at 10 ms, as the 60 cm Hybrid's does; it stays up again at 20 ms, which
misled the search. A corrected copy marks each change; no Stage 1 verdict changes (N8, N11).

## Answers

**How fast with today's drives (Q1)?** Re-fitting the tracker for speed made neither body show a faster walk. The
Tapered came closest: at the brisk walk (target 0.315 m/s) 19 of 20 stay up and 17 meet every limit, one fall short of
shown; at the animal-like walk (Froude 0.05, a pace scaled to leg length: 0.446 m/s) 16 of 20 stay up, against 1 before
the refit, but about 3 % short of the target. The Hybrid stays up in 14 of 20 at brisk and 1 at the animal-like walk,
though its Stage 1 controller had held brisk (20 up, 19 meeting every limit; provisional in Stage 1, as its other fit
kept 5 of 20 up). So, by Stage 1's two-controller rule, the nominal walk (0.22 m/s), the one you found too slow, is
still the fastest shown at 60 cm. All this is at zero servo delay and the model's high grip (μ 0.9). One fit per body is
a narrow test: Stage 1's two fits of one body differed by 15–17 of 20 at brisk, and this fit trained mostly with delay,
so a speed fit without delay might do better (§9).

**Does the faster knee change that (Q2)?** No, on the Tapered, the only body tried (decision 8). On the K body the refit
took brisk from 18 to 19 of 20 up (12 to 3 meeting every limit) and the animal-like walk from 2 to 17 up (none meeting
every limit): both still fail, as on the re-fitted pick, at about the same speeds. The refit, not the knee, raised
survival; on the pick's gains the new body stayed up about as often as the pick but met every limit less often on the
turns and at brisk (12 of 20 against 17; N9). It is also heavier and wider, and bends its knee too little for carpet
(§6).

**What does 10 ms of servo delay do (Q3)?** No 60 cm body walks: speed-fitted or not, every walk and walking turn falls
in all 20 episodes on every body (one Hybrid nominal walk excepted), mostly in the start, the first strides or the
change into a faster gait or a turn. Turns in place stay up, missing toe clearance mainly because the feet lift 10 ms
late against the clock it is read on; a few also miss peak foot force or joint range (§4). Fitting with delay did not
help at 10 ms. The faster walks need less delay than the nominal walk: brisk about 4 ms, the animal-like walk about 2 ms
(hints, §4).

**What did the refit cost?** On the plan's list (the nominal walk and four target turns, at no delay), nothing on the
Tapered or the Hybrid, though the Hybrid's nominal walk and left walking turn now sit exactly at 17 of 20; the K body
traded its right 0.35 rad/s walking turn (13 of 20 meet every limit). Outside the list, the Hybrid lost the brisk walk
its Stage 1 controller held (20/19 → 14/13), and with the heading gain near zero the Tapered's animal-like walks wander
sideways, a median 0.62 m in the judged 10 s, which no limit judges (N3).

**Is it today's hardware?** Not the leg drives, as far as this model shows; what could still decide it is unmeasured.
Removing the leg drives' torque limits never helped significantly where enough walks fell to test it (here at 0 ms only
on the Hybrid and on the K body before its refit; in Stage 1's 60 cm speed walks too), and only falls and single
stumbles pass a drive's rated speed (§5). Round two's sizing screens found that lighter materials buy torque headroom
and runtime, not speed headroom (F22). Unmeasured: the servos' delay (at 10 ms nothing walks), the tail servo's
continuous torque (0.97–1.03 of its planning limit at the animal-like walk), the drives' real torque near rated speed
(A2) and the treads' grip. The controller plays a part, and perhaps the foot pads, in shares this model cannot separate
(§5). Learning was not tried here; in Stage 1 the learned add-on passed nothing new at 75 cm and was not judged at 60
cm.

**Go bigger?** It buys speed slowly. At the same gait, speed grows only with the square root of size: 11–12 % faster at
75 cm than at 60 cm, about 40 % at 120 cm (a scaling rule, not a simulation). The 75 cm body's fastest shown walk, the
nominal, has a target of 0.24 m/s. Its brisk walk (target 0.35 m/s) stays up in every Stage 1 episode and, under both fitted
controllers, misses only the tail servo's planning limit, so A1's bench test or a stronger tail servo could change that
verdict (one learned run also misses speed and duty); its animal-like walk misses duty too (§6). It costs about $3.7k
and 4 kg more (round-two estimates), and meets every limit to 6 ms of delay against 4 ms at 60 cm but falls at 10 ms. No
body above 75 cm has been simulated walking with a balance controller.

**Scale back?** For now, plan on a 60 cm demonstration of the nominal walk (0.22 m/s) on hard floors (Stage 1 found no
walk on thin carpet), and only if the servos' end-to-end delay proves small: in the model that walk meets every limit to
about 4 ms, misses one from 6 ms and falls at 8–10 ms. Brisk (0.31 m/s) is a stretch: it needs about 4 ms or less (Q3),
grip on your floors (§9) and a controller that holds it reliably; of the seven fitted at 60 cm so far, one has. Bench
tests and more controller work would decide it (§6, §8).

## 1. What was run

- **Arms.** S: the two 60 cm picks. K: `60-N-T-10-K` alone; decision 8 dropped `60-N-H-10-K`, which fails round two's
  clearance gate.
- **The speed fit (R2s)** is Stage 1's search for the 13 gains with the brisk and animal-like walks added and each
  episode drawing a servo delay of 0–10 ms: one seed (42), 360 evaluations, from the body's Stage 1 controller (the K
  body from its pick's; N1). **Speed G5** then kept the start or one of the fit's later averages, whichever fell least
  in 100 validation episodes over all delays, 14 of them at 0 ms (N2).
- **The rows** (validation runs): the three walks, both 0.35 rad/s walking turns and both 0.5 rad/s turns in place, 20
  episodes each, hard floor at μ 0.9, walks and turns commanded 5 % above target, at 0 and 10 ms, read per row by plan
  section 4's rule, set before the runs. Baselines: Stage 1's controller for Q1 (R2/42 Tapered, R2/43 Hybrid; decision
  6), the K body on its pick's gains for Q2 (R2@0ms), the start's gains at 10 ms for Q3 (R2@10ms). Falls were attributed
  (decisions 2, 3).
- **Deviations.** Decision 10 moved the schedule with the 3.5 h late start. Arm K's `prepare` and gates G1–G4 ran before
  the merge, against decision 7's order; G1's re-run on the merged code (the model-integrity gate: hashes, standing,
  step rate) came after the rows, not before the fits, and failed its step rate on both K bodies, the machine being
  slower than on 8 Nov. Decision 11 set aside G1's stop-on-fail rule, as the step rate changes no episode (N7).

## 2. The speed walks at 0 ms (Q1 and Q2)

Counts of 20. "Default": under 5 falls, so nothing was lifted; every label from this round also names the pad edge as
untestable (§5). Speeds: walks that stayed up, judged window.

| Body, walk (target) | Controller | Up | Met every limit | Class | Limits missed by walks that stayed up | Falls; label | Speed, m/s: min / median / max |
|---|---|---|---|---|---|---|---|
| Tapered, brisk (0.315 m/s) | speed fit | 19 | 17 | fails | toe clearance 1; speed and duty 1 | 1 (change to brisk); default | 0.309 / 0.318 / 0.320 |
| | Stage 1 | 17 | 17 | fails | none | 3 (brisk gait); default | 0.319 / 0.320 / 0.321 |
| Tapered, animal-like (0.446 m/s) | speed fit | 16 | 0 | fails | speed 16, toe clearance 3 | 4 (2 in the gait); default | 0.410 / 0.433 / 0.440 |
| | Stage 1 | 1 | 0 | fails | speed 1 | 19 (18 in the gait); controller-limited | 0.439 |
| Hybrid, brisk (0.314 m/s) | speed fit | 14 | 13 | fails | one stumble, 7 limits | 6 (brisk gait); controller-limited | 0.315 / 0.319 / 0.321 |
| | Stage 1 | 20 | 19 | shown on this row; provisional in Stage 1 | one walk: speed, duty, toe clearance | none | 0.311 / 0.318 / 0.320 |
| Hybrid, animal-like (0.444 m/s) | speed fit | 1 | 0 | fails | speed, no flight | 19 (17 in the gait); controller-limited | 0.438 |
| | Stage 1 | 0 | 0 | fails | — | 20 (in the gait); body-limited: slip | — |
| K body, brisk (0.315 m/s) | speed fit | 19 | 3 | fails | speed 16 | 1 (brisk gait); default | 0.3133 / 0.3146 / 0.3153 |
| K body, animal-like (0.446 m/s) | speed fit | 17 | 0 | fails | speed 17, tail servo 17, slip 15 | 3 (2 in the gait); default | 0.426 / 0.432 / 0.436 |

- **"Speed" means short of the target, not slow**: 0.3133 m/s against 0.3150 fails. The animal-like walks end 1–8 %
  short; on the Tapered bodies the tracker slows its gait clock and steps short, using up the 5 % command margin (N6).
  Commanding the walks further above the target, or F25's gait-tuned animal-like walk (information only, P24), are
  untried levers. Removing slip saves all 6 of the Hybrid's brisk falls but fells another walk, just missing
  significance (p 0.06; N4).

## 3. What the speed fit kept and traded at 0 ms

Up / met every limit, of 20: baseline → speed fit.

| Condition | Tapered (Stage 1 → fit) | Hybrid (Stage 1 → fit) | K body (not re-fitted → fit) |
|---|---|---|---|
| Nominal walk | 20/18 → 20/19 | 20/18 → 20/17 | 20/19 → 20/19 |
| Walking turn 0.35 rad/s, left | 20/19 → 20/20 | 20/19 → 20/17 | 20/16 → 20/17, **gained** |
| Walking turn 0.35 rad/s, right | 20/19 → 20/20 | 20/20 → 20/18 | 20/17 → 20/13, **trade** |
| Turns in place 0.5 rad/s, each side | 20/20 → 20/20 | 20/20 → 20/20 | 20/19 → 20/20 |
| Brisk (outside the list) | 17/17 → 19/17 | 20/19 → 14/13, **lost** | 18/12 → 19/3 |
| Animal-like (outside the list) | 1/0 → 16/0 | 0/0 → 1/0 | 2/0 → 17/0 |

- **The K body's trade** is a turn rate just under target (6 of 20 at 0.347–0.350 rad/s against 0.35, though commanded
  0.3675); its gain, one fewer peak-foot-force miss, a noisy limit. At brisk its median speed fell 0.27 % (0.3154 →
  0.3146 m/s), from just above the 0.3150 m/s target to just below.

## 4. Servo delay (Q3)

At 10 ms, under the speed fit and the start's gains alike, every walk and walking turn but one falls, and the turns in
place keep 20 of 20 up, 0–5 meeting every limit.

- **When they fall.** The brisk and animal-like walks share their start, first stride and change to brisk, and fall at
  the same moments: 116 of 120 before the brisk gait, the other 4 (K body) in its first two brisk strides. So the 10 ms
  rows barely test the brisk gait and never the Froude gait. No removed limit kept more than 2 of 20 up (N4, N5).
- **Turns in place** lift each foot about as high as at 0 ms, only 10 ms later; toe clearance, read on the reference's
  clock, catches the late foot early in its lift: 14.8–16.4 mm against 16.1–16.2 mm (about 20 mm at 0 ms). A few also
  miss peak foot force or joint range (N5).
- **The faster walks need less delay** (hints from 1–7 selection episodes per delay; N1, N2): the selections' brisk
  walks stayed up in every episode to 4 ms, but at 6 ms only on the Hybrid (4 of 4; the Tapered 1, the K body 0), and
  their animal-like walks only at 0–2 ms.
- **A bench test bounds the answer without settling it**: each drive type's command-to-motion delay, end to end, and
  servo stiffness, read as delay added to a loop that already lags (N8). The nominal walk meets every limit to about 4
  ms and falls at 8–10 ms; a slower result rules out this controller, not every controller.

## 5. What stops the faster walks

- **Before the falls.** All 653 falls (54 at 0 ms, 599 at 10 ms) had the pad-edge flag in the 0.5 s before them. The
  controller's own ankle cap raises that flag when it holds a loaded foot's centre of pressure just inside the pad's
  edge, in about a fifth of the updates even in walks that never fall: so the flag does not show that the pads cause
  falls, and it cannot be tested here. In 620 falls a loaded foot slid briefly, even at μ 0.9; in 354 a leg drive was at
  its torque limit (N4).
- **The drives nearest a limit.** The tail servo's RMS torque (the average load that sets its heating), against A1, a
  planning limit of 0.491 N·m (about half the spec sheet's 0.98 N·m, which has no thermal or duty definition), is
  0.97–1.00 at the animal-like walk on the Tapered, 1.00–1.03 on the K body and 0.85–0.94 at brisk: a reason for A1's
  bench test, not a thermal verdict. At the animal-like walk the knee peaks at 0.87 of its rated speed with the battery
  at 24 V or more (Tapered) and the hip pitch at 0.84 (K body); at the pack's nominal 22.2 V, about 0.94 and 0.91, and
  at 21 V, the end of a charge, 1.00 and 0.96 (computed). The model gives full torque up to that speed, unmeasured until
  A2's dyno curves exist. Falls and single stumbles pass it (knees up to 1.26), but a drive at that limit came before
  only 1 of the 54 falls at 0 ms (N6).

## 6. What would change the answer

None of these is adopted (P24); each needs your decision.

| Lever | What it would take | For | Against, or caution |
|---|---|---|---|
| Controller work | More fit seeds; a speed fit without delay or at the bench's delay; another structure (such as stepping to balance); compute only | The refit raised animal-like survival on the Tapered bodies (1 → 16, 2 → 17), not the Hybrid (0 → 1) | One family tried; labels cannot single out the controller (§5); Stage 1's learned add-on passed nothing new at 75 cm and was not judged at 60 cm |
| Servo-delay bench test | Command-to-motion delay and stiffness per drive type | Bounds whether any walk works (§4) | Grip and the controller remain |
| Tail servo | Find why its closed-loop load is 2.7–2.8 times Stage 0's; A1's bench test; then perhaps a stronger servo or lighter tail | The drive nearest a limit | At 75 cm the animal-like walk would still miss duty |
| Ankle and pad | A narrow 60 cm model with bigger pads, prepared and fitted | The pad-edge flag precedes every fall | The controller's cap also raises it; not adopted at 60 cm (P20); bigger toes load the upper ankle drive to 0.88–0.89 of its limit (F23) |
| Faster knee | +0.75 kg; 21 mm wider; too little knee bend for the carpet swing; runtime 40 against 48 min (estimates) | Stage 0 knee speed ratio 0.54 against 0.93 | No gain here; rated speed derived, 2.9 times the vendor's (N6); 17–20 % more grip needed (N9); the hip pitch (0.84) binds next |
| The 75 cm body | +$3.7k (high $25.7k, above P4's $20k), +4 kg, about 11 min less runtime (estimates) | 11–12 % faster at the same gait; in Stage 1 stays up at both speeds | Shows neither: the tail servo (1.04–1.31 × A1); at the animal-like walk duty, and under one fit four more limits; falls at 10 ms |

Tested here: the refit, and the faster knee on the Tapered; the 75 cm body only in Stage 1. The rest are untested.

## 7. Budget and schedule

The round used 8.2 core-hours, 83–87 % of the projection (N10). All three fits reached generation 30, and the rows were
done by 10:54 UTC, inside decision 10's limit of about 14:30.

## 8. Draft decision-log rows for your approval

All are proposals awaiting you (P24); the full texts are in N11.

- **F27 (finding):** this round's results.
- **F26, a correction to Stage 1's draft:** the 75 cm nominal walk falls at 10 ms of delay, not "at least 20 ms" (the
  correction at the top; N11).
- **A1 update:** bench-test the tail servo first, at the speed walks' closed-loop loads (0.42–0.51 N·m at 60 cm,
  0.51–0.64 N·m at 75 cm).
- **A17 (new):** the verdicts assume no servo delay; replace with a bench measurement of each drive type's
  command-to-motion delay and of servo stiffness.

## 9. What this does not show

- **Hardware.** Simulation only: no measured delay, servo stiffness, tail torque limit, drive torque near rated speed or
  tread grip.
- **Your floors.** Hard floor at μ 0.9 only. The treads' grip is round two's unmeasured estimate: 0.8–0.9 on wood,
  0.9–1.0 on tile, 0.6–0.85 on polished concrete, 0.42–0.8 on vinyl and laminate. At Stage 1's slippery test grip, μ
  0.54, the one controller that held brisk kept 17 of 20 up, 14 meeting every limit; the animal-like walks kept 0–2 of
  20 up. No carpet, shove, stand or start-and-stop rows.
- **More than one fit, or a fit without delay.** One fit per body, which drew the animal-like walk at 0–2 ms in only 15
  of 30 generations and was selected over all delays (N1, N2).
- **Another set of starts.** 20 episodes, so one fall fails a condition; the Hybrid's selection kept all 14 validation
  brisk walks at 0–6 ms up, but only 14 of the rows' 20 at 0 ms.
- **Delays between 0 and 10 ms** in the rows; **margins at the thresholds** (a walk 0.6 % under target fails); **other
  controllers**.

## Engineering notes

Paths are relative to `hardware/trex_form_study/stage1/runs/2026-11-10-speed/` unless they begin `STAGE1B_PLAN.md`,
`stage1/`, `round2/` or `designlab/` (relative to `hardware/trex_form_study/`); `<b>` is a body.

**N1. The fits' course and the delay draws.** Objective `stage1-tune-speed-v1` on `stage1-tune-v2`
(`<b>/speedfit/42/cma.json`, `objective.changes`): the mix widened to five episodes, each with its own delay of k
physics steps, k drawn uniformly from 0–5 by `default_rng(tune.derived_seed(seed, slot, 3))`; everything else Stage 1's.
Population 12, sigma0 0.1 of each gain's range, 13 gains searched, capture margin fixed at 0, rate 1.05, training
episodes 1000–1999 (one per generation, all 12 members on it), validation 2000–2019. Each fit is `status: done`, 360
evaluations, one session, no errors, warnings or incidents. All three fits drew the same delays, since the episode order
depends on the seed alone. Falls by kind and drawn delay, falls / episodes, from `history[*].falls_by_kind` and
`latency_steps`; they pool every candidate the search tried, most of them discarded, so they are hints about delay, not
results:

| Kind | Body | 0 ms | 2 ms | 4 ms | 6 ms | 8 ms | 10 ms |
|---|---|---|---|---|---|---|---|
| Nominal walk | T / H / K | 0/60 each | 0/60 each | 0/60 each | 0/60 each | 7, 19, 5 of 36 | 82, 83, 79 of 84 |
| Walking turn | T / H / K | 0/60 | 0/84 | 0/60 | 0/36 | 55, 73, 58 of 96 | 24/24 each |
| Turn in place | all | 0/84 | 0/36 | 0/72 | 0/48 | 0/48 | 0/72 |
| Brisk walk | T / H / K | 24, 22, 10 of 132 | 8, 6, 2 of 72 | 3, 0, 2 of 36 | 22, 21, 28 of 36 | 47, 47, 48 of 48 | 36/36 each |
| Froude 0.05 | T / H / K | 39, 59, 30 of 84 | 57, 57, 41 of 96 | 36, 33, 36 of 36 | 36/36 each | 36/36 each | 72/72 each |

Falls in all: 584, 624 and 543 of 1,800 episodes. The Froude 0.05 walk drew 0 ms in 7 of the 30 generations (0–2 ms in
15), the brisk walk in 11 (17). The step size ranged 0.071–0.095 (T), 0.076–0.107 (H) and fell from 0.095 to 0.039 (K).
A generation's score mostly tracks the delays drawn: generation 7 (8–10 ms on every kind) has a best score of 4.1–5.7,
generations 17–18 0.14–0.23. A rough split of each generation's score spread (`evals.jsonl`: the variance of the fall
penalties from 8–10 ms episodes against that of the other fall penalties and the performance term, covariances ignored)
puts more than half of it on the 8–10 ms episodes in 9, 10 and 15 of the 30 generations. Of the evaluations with no fall
at all (27, 26 and 40), every one came in a generation whose Froude 0.05 episode drew 0–2 ms, except three on H at 4 ms
(generations 2 and 11). A fit for speed without delay in its mix was not run.

**N2. Speed G5** (`<b>/speedg5.json`, `seeds.42.selection`). Falls of 100 validation episodes (5 kinds × 20) for the
start and the means after generations 10 / 20 / 30: T 39, 33, 33, 35 (selected gen 20, over gen 10 on mean survival
penalty 2.405 against 2.432); H 35, 33, 40, 36 (gen 10); K 39, 35, 34, 33 (gen 30). Each validation episode draws its
delay as the fit does, so a cell at one delay holds 1–7 episodes; 14 of the 100 ran at 0 ms (nominal walk 2, walking
turn 5, turn in place 3, brisk 1, Froude 0.05 3). By delay (`conditions.<kind>.by_latency_steps` against
`start_conditions`; hints, not results):
- Brisk: the selections stayed up in all 10 episodes at 0–4 ms on every body (on the K body 8 met every limit); at 6 ms
  in 1, 4 and 0 of 4 against the starts' 3, 4 and 3; at 8–10 ms in none.
- Froude 0.05: the selections stayed up in 5, 0 and 3 of 5 at 0–2 ms and in none of 6 at 4 ms, nor beyond.
- All 14 of the Hybrid selection's brisk episodes at 0–6 ms stayed up, as under its start, so the rule could not see the
  Hybrid's 0 ms brisk loss, which shows only on the rows' 20 starts.
- Walking turns at 8 ms: the selections stayed up in 1, 4 and 5 of 5 against the starts' 0, 1 and 1, none meeting every
  limit.
- Nominal walks at 6 ms: the selections met every limit in 0, 2 and 0 of 4 against the starts' 3, 3 and 1.
- Turns in place met every limit in 5 of 5 at 8 ms on every body, selection and start alike, and in none of 4 at 10 ms.

**N3. The gains that moved** (start → selection; change as a share of the gain's search range; `speedg5.json`
`start.gains`, `seeds.42.selected_gains`; meanings and ranges from `control.GAIN_SPEC`).

| Gain | Tapered | Hybrid | K body |
|---|---|---|---|
| `k_yaw`, heading P gain | 0.273 → 0 (−0.27) | 0.592 → 0.431 (−0.16) | 0.273 → 0.016 (−0.26) |
| `ki_yaw`, heading I gain | 0 → 0.010 (+0.02) | 0.055 → 0 (−0.11) | 0 → 0 |
| `k_phase`, gait-clock P gain, 1/m | 0.108 → 0 (−0.01) | 0.187 → 0 (−0.02) | 0.108 → 0 (−0.01) |
| `ki_phase`, gait-clock I gain, 1/(m·s) | 0.83 → 0.97 (+0.01) | 0.53 → 0.07 (−0.05) | 0.83 → 1.18 (+0.03) |
| `d_trunk`, trunk damping, N·m·s/rad | 0 → 2.39 (+0.24) | 1.50 → 2.25 (+0.07) | 0 → 1.19 (+0.12) |
| `k_trunk`, trunk stiffness, N·m/rad | 80 → 89 (+0.06) | 112 → 129 (+0.11) | 80 → 73 (−0.05) |
| `step_out`, outward step limit, mm | 21.9 → 37.7 (+0.26) | 20.5 → 28.8 (+0.14) | 21.9 → 23.2 (+0.02) |
| `step_x`, forward step limit, mm | 68.2 → 76.0 (+0.08) | 49.2 → 31.5 (−0.18) | 68.2 → 69.5 (+0.01) |
| `cop_x`, ankle shift limit forward, mm | 42.1 → 33.0 (−0.18) | 26.0 → 20.2 (−0.12) | 42.1 → 47.2 (+0.10) |
| `cop_y`, ankle shift limit sideways, mm | 25.0 → 25.8 (+0.03) | 28.7 → 28.5 (−0.01) | 25.0 → 30.0 (+0.17) |
| `decay`, step-offset decay | 0.93 → 0.91 (−0.01) | 2.68 → 1.96 (−0.24) | 0.93 → 1.82 (+0.30) |
| `k_dcm`, ankle balance gain | 4.57 → 3.58 (−0.17) | 4.56 → 5.12 (+0.09) | 4.57 → 4.07 (−0.08) |
| `k_step`, stepping gain | 0 → 0 | 0 → 0.043 (+0.01) | 0 → 0 |

Distance moved in the unit search space: 0.52 / 0.43 / 0.47; one sample at the starting step size (sigma0 0.1 over 13
gains) lands about 0.35 from its mean. Only `k_yaw`, `k_phase`, `d_trunk` and `step_out` moved the same way on every
body. Several selections sit on a bound with the search's mean beyond it (`cma.json`, `history[*].mean`): the gait-clock
P gain at 0 on every body, the heading P gain on the Tapered and the heading I gain on the Hybrid and K body at 0, the
stepping gain at 0 on the Tapered and K body, and the K body's sideways ankle shift at its 30 mm top. The gait-clock I
gain, which slows the clock, rose on the two bodies whose survival rose and fell on the Hybrid. Which change raised
survival is not tested. The Tapered's right turn, whose heading gain went to 0, also turns a little slower than in Stage
1 (median 0.357 against 0.364 rad/s) without missing the target; whether the K body's lower heading gain causes its slow
right turn is not tested. Sideways drift over the judged window (`measures.window.lateral_m`, against the starting
heading): the Tapered's Froude 0.05 survivors 0.14–1.43 m (median 0.62), turning up to 0.33 rad (`heading_change_rad`);
the K body's 0.01–0.73 m (median 0.11); the Hybrid's one survivor 0.03 m; Stage 1's one Tapered survivor 0.07 m. At
brisk the Tapered drifts up to 0.26 m against 0.11 m under Stage 1's controller. The heading change is a term of the
fit's objective but no success term judges it.

**N4. The attribution runs and the fall candidates** (`<b>/eval/<row>/<behavior>/<condition>.json`,
`attribution.lifted.<limit>`: `survived`, `sign_test`; `failure.falls_label`, `failure.label`). A one-sided paired sign
test on survival time over the 20 seeds, p < 0.05, run only with 5 or more falls. The lifts (`control.RELAXATIONS`)
remove the leg drives' torque limits, the envelope, the ankle linkage's limits, slip, the inward step bound and joint
range; the neck and tail servos' torque limits are recorded but never lifted (`control.REPORTED_LIMITS`), and the pad
edge cannot be. At 0 ms three records qualified. The Hybrid's brisk walk (6 falls): removing slip kept 19 of 20 up,
saving all six falls (seeds 3044, 3048, 3051, 3056, 3057, 3058) and making seed 3050 fall, so 6 better and 1 worse, p
0.0625; unlimited leg torque kept 14 (p 0.98). The Hybrid's animal-like walk (19 falls): leg torque 2, ankle linkage 2,
slip 1 of 20 up, p 0.13–0.32. The K body's unrefit animal-like walk (18 falls): slip 4, leg torque 1 of 20 up, p
0.41–0.59. The other five 0 ms records with falls had 1–4 and were not lifted; their labels are controller-limited by
default. At 10 ms every failing walk and walking turn was attributed (300–380 lifted episodes per row); no lift was
significant and none kept more than 2 of 20 up. Servo delay itself cannot be lifted. Fall causes: trunk tilt in 609
falls, the tail touching the floor in 44. Candidates in the 0.5 s before onset (`episodes[].fall_candidates`; the share
of feedback updates in `limits_before_onset.hardware`): the pad edge in all 653 (a median 100 % of the updates, at least
47 %), slip in 620 (a median 10 %, at least one update), a leg drive at its torque limit in 354, the ankle linkage in 98
and a drive at its envelope's limit in 14; at 0 ms, 54, 54 and 39 of the 54 falls. The pad-edge flag is `control.py`'s
`cop_edge = cap_pad or (cop_clip and ankle_sat)`. The cap trigger alone was on in 21–24 % of the updates of Stage 1's G4
nominal walks on the tracker's starting gains, 60 strides, 20 of 20 up on each body
(`stage1/runs/2026-11-08/<b>/gates.json` and `60-N-T-10-K/gates.json`,
`G4.gated.R2_nominal_hard.panel.rows[*].bound_frac.ankle_cap_pad`); the rows do not record the rate in walks that stay
up.

**N5. Where the falls come, and the in-place toe clearance.** Each fallen episode's trace
(`<b>/eval/traces/<row>/<behavior>/<condition>/<seed>.npz`, last `t_ref`) is placed in its plan's segments, timed from
the body's `transitions.json` (`duration_s`) and `refs.json` (`period_s`). At 0 ms: the Tapered's brisk fall in the
change to brisk, its animal-like falls 2 in the Froude gait, 1 in the change to brisk and 1 in the ramp to Froude; the
Hybrid's brisk falls 6 in the brisk gait and animal-like falls 17 in the gait and 2 in the ramp; the K body's brisk fall
in the gait and animal-like falls 2 in the gait and 1 in the ramp; on its pick's gains, its 2 brisk falls in the gait
and 17 of 18 animal-like ones in the gait. At 10 ms the 119 nominal-walk falls come 25 in the start, 85 in the nominal
strides and 9 in the stop. The brisk and Froude 0.05 walks share the start, one nominal stride and the change to brisk,
and their fall times agree seed by seed in all 120 pairs: 25 in the start, 42 in the nominal stride, 49 in the change to
brisk and 4 in the brisk gait (all on the K body; the Froude walk's four come in its one brisk stride and the change to
Froude). The walks fall 4.10–15.10 s in. The 240 walking-turn falls (4.27–21.07 s) come 52 in the start, 169 in the ramp
into the turn, 15 in the turning strides and 4 in the exit. Stage 1's baselines agree with its report (N5 there): the
Tapered's R2/42 animal-like falls 18 in the gait and 1 in the ramp, the Hybrid's R2/43 all 20 in the gait. Toe clearance
(`designlab/stage1/metrics.py`, `clearance.MID_SWING` 0.25–0.75) is the lowest foot point over the middle half of each
swing of the reference, timed on the reference's clock (`ref_u`), so a foot that follows 10 ms late is read early in its
lift. From the 50 Hz traces of the turns in place, each foot's own swing is the same at 0 and 10 ms (airborne 0.31–0.32
s, peaks of 38–40 mm, 24–26 mm a quarter and three quarters of the way through it), and reading the window 10 ms later
gives 18–20 mm at 10 ms (at 0 ms the same shift gives 17–18 mm). At 10 ms the turns in place also miss peak foot force
in 1–3 episodes per turn and joint range in 1–2 on the Hybrid.

**N6. Record keys, limits and what each ratio is measured against.** An episode falls when the trunk tilts over 0.6 rad,
the body drops below 0.7 of its standing height or a shell touches the floor (`plant.Plant.fall_cause`). The success
terms (`tune.SUCCESS`, `speedfit.SUCCESS`, thresholds `tune.py` lines 152–161): duty 0.55–0.65; no flight; trunk roll
and pitch RMS 4° and peak 8°; toe clearance 0.04 H (16.2 mm on the Tapered bodies, 16.1 mm on the Hybrid); slip per
stance, median 5 mm and worst 15 mm; joint range (decision 14); each drive's RMS and peak speed; the nominal walk also
cadence, stride and peak foot force 1.6 body weights; the turns peak foot force, turn rate and no leg interference, and
in place drift 0.1 m. The Tapered's speed-fit misses beside speed: toe clearance 14.6, 14.9 and 16.15 mm at the
animal-like walk and 15.6 mm at brisk, against 16.2 mm, and duty 0.549 against 0.55 in one brisk walk. Speeds:
`episodes[].measures.window.speed_m_s`, the centre of mass's mean forward speed over the judged window (the last 10 s of
the brisk and Froude 0.05 walks; the 10 s before the stop of the nominal walk), along the starting heading; the speed
term passes only if it is at least the target (`designlab/stage1/tune.py` lines 633–634). On the Tapered's Froude 0.05
survivors the sideways drift (N3) takes a median 1 % off that speed. Turn rates: `measures.window.yaw_rate_rad_s`.
Sustained load: `measures.window_sustained_ratio` (= `measures.window.drive_sustained_worst`), each drive's RMS motor
torque over the window over its `sustained` torque in `plant.json` (A1: 0.57 × the vendor rating; tail 0.491 N·m,
X4-36-L 5.985 N·m, RobStride 06 6.27 N·m), the worst drive named. The tail's 0.862 N·m rating is itself derived from its
2.94 N·m stall by the PR #662 servo rule (`designlab/data/actuators.json`, `feetech_sts3215_c018`); its spec sheet gives
10 kg·cm (0.98 N·m) at 12 V with no thermal or duty definition. Stage 0's Froude 0.05 reference loads the 60 cm tail to
0.178 N·m RMS (`round2/refine/narrow_carpet_60/review/screening/60-N-T-10_envelope.json`, `fast_walk.groups.tail`),
against 0.48–0.50 N·m in these closed-loop walks. Peak speed: `episodes[].worst_speed_ratio`, each drive's peak motor
speed over the whole episode over its envelope's `rated_speed` at 24 V (X4-36-L 8.69 rad/s; RobStride 06 15.08 rad/s),
above which the driving torque falls linearly to zero at no-load speed (11.62 and 25.13 rad/s). Below that corner the
envelope allows the full planning peak, 0.70 × the vendor's peak (`designlab/stage1/drive.py`), though vendors quote
rated speed at rated torque; A2's dyno curves are pending, so a ratio under 1 does not certify peak torque at that
speed. Over whole episodes, 8 of the 34 speed-fit falls at 0 ms pass rated speed (knees 1.04–1.26 on the Tapered and
Hybrid, an ankle 1.06 on the K body), as do two single stumbles in walks that stay up (the Hybrid's speed-fit brisk
walk, a knee at 1.06, tilting 38°; the K body's brisk walk on its pick's gains, an ankle at 1.07, 21°); a drive at its
envelope's limit was a fall candidate before onset in 1 of the 54 falls at 0 ms (`fall_candidates`). RobStride rates the
06 at 11 N·m at 100 rpm (±10 %, 48 V); the catalog's 288 rpm is 0.6 × its 480 rpm no-load speed, 2.9 times that, so the
K knee's Stage 0 ratio of 0.54 is optimistic until a dyno curve exists. The 22.2 V and 21 V figures in §5 scale the 24 V
medians (0.873 and 0.838) by 24 / V, the X4-36-L's rated speed falling in proportion below 24 V (E9; the 6S pack swings
21.0–25.2 V, A11); the knee's highest episode, 0.877, becomes 1.00 at 21 V. These are computed, not simulated; Stage 1's
ledger had the 60 cm animal-like walk falling at every voltage. The "7–8 %" of F27 compares those medians with Stage 0's
need without its 1.15 margin (`margins.<drive>.speed_req` / 1.15 / `speed` in the round-two candidate records): knee
0.873 against 0.808 on the Tapered, hip pitch 0.838 against 0.782 on the K body. At the animal-like walk the survivors'
median cadence is 2.81 (Tapered) and 2.79 (K) steps/s against the commanded 2.89, stride 0.774 and 0.769 H against 0.80,
and the gait-clock correction (`phase_rate_range`, a PI on the forward balance error bounded at ±5 %) reaches its −5 %
bound in the median episode; together these use up the 5 % command margin. `python -m designlab.stage1 speedreport`, run
again into a scratch directory, gave `STAGE1B_TABLES.md` identical to the committed one, and `stage1b_tables.json`
identical but for its `written_at`.

**N7. The order of arm K's runs, G1 and the pre-fit test** (decisions 6, 7 and 11; `ARM_K.md` §5; `<K body>/gates.json`;
`logs/run_speed_round.log`). Arm K's `prepare` and G1–G4 ran before the merge, from a dirty tree at 07a73f4 (the G2 and
G4 records name that commit), against decision 7's order; `ARM_K.md` §5 sets out why their products would come out the
same. That first G1 timed its per-core rate on 3 processes, not one per core, and passed. `ARM_K.md` §5 asked for G1
again on the merged code, with the machine idle, before the fits; that was missed at the 07:54 launch and run at 10:58
UTC on 1d42fbd, after the rows. The model hashes, tendon map and repeatability match the earlier records exactly, and
the stands too, but for their timing fields. The step-rate check failed: 18,153 (`60-N-T-10-K`) and 18,051
(`60-N-H-10-K`) steps/s alone, and 15,418 and 15,118 per core, against 20,000, so both gate records now read
`pass: false`. The Tapered pick, timed the same way into scratch, gave 17,544 and 18,322 alone against 26,405 on 8 Nov,
so the machine, not the bodies, was slower. G1's rule is to stop when any check fails (`STAGE1_PLAN.md` section 3);
decision 11 set that aside because the step rate bounds a run's cost and changes no episode, so nothing was re-run.
Decision 6 also asked for `test_stage1_speedfit`, which reruns a recorded G5 episode, to be run again on the merged code
before any fit. The run script's `check` step, which runs it, writes no log; the last recorded pass is in commit bdced9e
(06:07 UTC), before the merges, so the records do not show the re-run. The round's code did repeat Stage 1's ledger
exactly where the two meet (N8).

**N8. Servo delay: Stage 1's ledger, the model and the bench**
(`stage1/runs/2026-11-08/<b>/sweeps/R2/<seed>/latency.json`, `75-M-H-10/sweeps/R3/42/latency.json`,
`evaluations["walk/nominal|latency_steps=k"]`; nominal walk, seeds 3042–3044). Up / meeting every limit:

| Delay | `60-N-T-10` R2/42 | `60-N-H-10` R2/43 | `75-M-H-10` R2/43 | `75-M-H-10` R3/42 |
|---|---|---|---|---|
| 0 ms | 3 / 2 (peak foot force, as at nominal) | 3 / 3 | 3 / 3 | 3 / 3 |
| 4 ms | 3 / 3 | 3 / 3 | 3 / 3 | 3 / 3 |
| 6 ms | 3 / 1 | 3 / 1 | 3 / 3 | 3 / 3 |
| 8 ms | 2 / 0 | 3 / 0 | 3 / 0 | 3 / 0 |
| 10 ms | 0 / 0 | 0 / 0 | **0 / 0** (falls 4.4–6.5 s) | **0 / 0** (4.6–4.8 s) |
| 20 ms | 0 / 0 | 0 / 0 | 3 / 0 | 3 / 0 |

Re-read value by value, the 60 cm readings (stays up to 6 / 8 ms, every limit to 4 / 4 ms) agree with every delay
evaluated (0, 4, 6, 8, 10 and 20 ms); no search on any body ran 2 ms or 12–18 ms. The survival search
(`summary.searches.walk_nominal.survival`) evaluated only 0 and 20 ms on the 75 cm body and read "works across the
interval", since it assumes a single crossing; the success search's own 10 ms evaluation holds the falls. The 75 cm
response is not monotonic and is unexplained; 10 ms is one 100 Hz feedback period, and whether that matters is not
tested. On seeds 3042–3044 this round's R2@10ms nominal walk falls at 5.37, 5.02 and 7.63 s on the Tapered and 4.11,
4.22 and 4.53 s on the Hybrid, the ledger's 10 ms times exactly. The model (`speedfit.LATENCY_RULE`, env.py's latency
model): the whole servo target (position, velocity and feedforward torque) reaches the servos k physics steps late, in
front of an ideal PD servo at the model's stiffness; the targets update at 500 Hz; the controller reads sensors as of
the step's start, one step old, and its feedback, computed every 10 ms, is ramped into the targets over the next 10 ms
(`control.FEEDBACK_EVERY`, `FEEDBACK_HOLD`). A bench figure should therefore be compared with these delays as delay
added to such a loop, end to end (sensing, estimation, the computer and the bus) per drive type, on a bus that can carry
targets at the model's rate. Servo stiffness belongs on the same bench: in Stage 1 the nominal walk met every limit only
down to 0.67–0.68 × the holding gains at 60 cm (0.71 at 75 cm). A delay above these figures rules out this controller
family, not every controller; one below them still leaves grip and the controller to check.

**N9. The K body against its pick** (`round2/refine/narrow_knee_60/` and `narrow_carpet_60/candidates/60-N-T-10.json`;
`ARM_K.md` §7; round-two model estimates, not quotes or measurements, for mass, cost, power and runtime). Mass 11.56
against 10.81 kg; cost $14,834 / $20,642 against $14,809 / $20,544; walking power 81.7 against 69.3 W with the knee
flagged "R estimated" (RobStride publishes no winding resistance); runtime 40.3 against 47.6 min; Stage 0 speed ratios
at Froude 0.05 with the 1.15 margin, knee 0.544 against 0.929 and hip pitch 0.900 against 0.899; hip spacing +10 mm per
side against +6 mm (191.6 against 183.6 mm between the hip-pitch axes); overall width 295 against 274 mm; knee flexion
end 0.856 against 1.051 rad; hip-pitch extension end −0.803 against −1.000 rad; carpet reference's smallest joint margin
0.0065 rad (knee) against 0.180, and the carpet swing fails the walking gates at 0.19 and 0.195 H
(`round2/refine/carpet_lift_narrow_knee_60_0.190_0.195_nominal_only.json`); grip need with the stance foot's yaw counted
0.568 / 0.886 / 0.654 (nominal, brisk, walking turn) against 0.481 / 0.754 / 0.545; scorecard 0.647 against 0.703. The
RobStride 06's sustained limit is A1's rule with no vendor thermal definition. In the rows the K body's median slip per
stance at the nominal walk is 3.1–3.6 mm (median 3.3) against the pick's 1.8–2.1 mm (median 2.0;
`measures.window.slip_median_m`, R2s/42). On the pick's gains (R2@0ms against the pick's Stage 1 R2/42), the K body met
every limit in 19 against 18 nominal walks, 16 and 17 against 19 and 19 walking turns, 19 and 19 against 20 and 20 turns
in place, and 12 against 17 brisk walks; the left walking turn's four misses are peak foot force.

**N10. How the compute was counted.** Fits: the sum of `history[*].cpu_s` in `cma.json`; speed G5: the sum of
`candidates.<c>.runs[*].cpu_s`; rows and attribution: the sums of each record's `episode_wall_s` and
`attribution.wall_s`, one worker per body, so an upper bound on CPU. Arm K's sizing, `prepare` and gates, the G1 re-run
and the pick's timing check are not counted.

| Core-hours | Tapered | Hybrid | K body | Projected (`60-N-T-10/speedfit/cost.json`) |
|---|---|---|---|---|
| Speed fit | 1.60 | 1.57 | 1.65 | 1.90 each, an upper bound |
| Speed G5 | 0.40 | 0.39 | 0.40 | 0.42 each |
| Rows | 0.30 | 0.30 | 0.49 | 0.44 (K body 0.58) |
| Attribution | 0.26 | 0.40 | 0.43 | 0.29–0.43 (K body 0.38–0.57) |
| **Total** | **2.56** | **2.67** | **2.97** | 3.04–3.19 (K body 3.28–3.47) |

The projection was measured on the Tapered alone, on code from before the merge. The fits ran 07:54–09:34 UTC, speed G5
to 09:58 and the rows to 10:29 / 10:36 / 10:54 (`logs/run_speed_round.log`). Each fit took 1.59–1.66 h of wall time,
half of decision 10's four-at-once figure, because three fits ran on their own cores. `cost.json` was measured at commit
dc967c1 (06:02 UTC, before the merge) with speed sources `c43721d3`; the fits ran on `68f5a80b`. Projected totals: the
sum of 1.90, 0.42, 0.44 and 3 × (0.10–0.14) core-hours per arm S body, plus 0.15 and one more attribution on arm K.

**N11. The draft decision-log rows in full** (§8; proposals awaiting you, P24).

**F27 (finding), proposed.** Stage 1b speed round (simulation, 10 Nov 2026; `stage1/runs/2026-11-10-speed/`). R2
re-fitted by CMA-ES (seed 42, 360 evaluations) with the brisk and Froude 0.05 walks and 0–10 ms servo delay in its mix,
and selected over all delays, on `60-N-T-10`, `60-N-H-10` and `60-N-T-10-K` (`60-N-H-10-K` dropped, decision 8). At 0 ms
on hard floor (μ 0.9), one fit seed, no body shows brisk or Froude 0.05: brisk 19/17, 14/13, 19/3 of 20 survive /
succeed; Froude 0.05 16/0, 1/0, 17/0, survivors 1–8 % under target speed. The refit raised Froude 0.05 survival on both
Tapered bodies (1 → 16, 2 → 17); lost the brisk walk of `60-N-H-10`'s Stage 1 R2/43 row (20/19 → 14/13; provisional in
Stage 1, where R2/42 kept 5/20 up); traded `60-N-T-10-K`'s right 0.35 rad/s walking turn (17 → 13 succeed); and, with
the heading gain near 0, let `60-N-T-10`'s Froude 0.05 survivors drift up to 1.43 m sideways in the judged 10 s, which
no success term judges. On its pick's gains `60-N-T-10-K` succeeds less often than `60-N-T-10` on the turns and at brisk
(12 against 17). At 10 ms every walk and walking turn falls (0/20 but one), fitted or not, the speed walks almost all
(116 of 120 pairs) before their faster gait begins; turns in place survive, missing toe clearance as read on the
reference's clock. Fit and selection episodes hint that brisk needs about 4 ms of delay or less and Froude 0.05 about 2
ms. All 653 falls had the untestable pad-edge candidate, which the tracker's ankle cap also raises in walks that do not
fall; no lift was significant. At Froude 0.05 the tail servo runs at 0.97–1.03 of A1, and peak drive speeds exceed Stage
0's reference need by 7–8 % (A2's margin is 1.15). G1 on the merged code failed its step rate on both K bodies (the
machine; no episode affected; decision 11). *Evidence:* `STAGE1B_TABLES.md`; `<b>/eval/`, `<b>/speedg5.json`,
`<b>/speedfit/42/`. *Status:* proposed.

**F26 (Stage 1's draft), proposed correction.** Replace "survives up to 6 / 8 / at least 20 ms of actuation delay (20 ms
ends the search; every limit met to 4 / 4 / 6 ms)" with "survives up to 6 / 8 / 8 ms of actuation delay (on `75-M-H-10`
all 3 episodes fall at 10 ms under R2/43 and R3/42, though all 3 survive at 20 ms, the only larger delay tried, none
succeeding; every limit met to 4 / 4 / 6 ms)". The corrected copy of Stage 1's report (`STAGE1_PROBE.md`) adds a
correction note at its head and marks each change "corrected 10 Nov": the answer on which body does best (line 21: "with
more servo delay" holds only for meeting every limit, 6 against 4 ms); the answer on hard floors (lines 34–36: on every
body the walk misses a limit at 6–8 ms, falls at 8–10 ms and meets every limit only down to 0.67–0.71 × the holding
gains); both 75 cm cells of the ledger's servo-delay row (line 184, R2 and R3/42: "20† / 6" becomes "8 / 6"); the note
on what binds (lines 203–206: the 75 cm falls at 10 ms were never lifted); §6 item 6 (line 349: R3/42 still falls at 10
ms on the 75 cm body, as R2 does) and item 8 (line 357: R3's training delays, to 20 ms, reach past R2's survival edge on
every body, not only "at 60 cm"); this draft (line 390); the caveat on sharp ledger edges in its §9 (line 502: a search
whose far end works tries nothing in between); and a new bullet in N4 on the missed falls. `STAGE1_TABLES.md` is
machine-made and stays as it is (decision 6).

**A1, proposed update.** "Replace with" becomes: "bench thermal tests per joint duty; first the STS3215 tail servo at
the speed walks' closed-loop RMS: at 60 cm (Stage 1b) 0.42–0.46 N·m at brisk and 0.48–0.51 N·m at Froude 0.05; at 75 cm
(Stage 1, R2) 0.51–0.52 and 0.57–0.64 N·m". Each figure is `measures.window_sustained_ratio` × 0.4913 N·m.

**A17 (new), proposed.** "Stage 1's verdicts and Stage 1b's 0 ms rows assume no servo delay. In the model the 60 cm
nominal walk misses a limit from 6 ms and falls from 8 ms (Tapered) or 10 ms (Hybrid); the 75 cm walk misses a limit
from 8 ms and falls at 10 ms, though it stays up at 20 ms. Fit and selection episodes hint that brisk needs about 4 ms
or less and Froude 0.05 about 2 ms. The model's delay is added to a loop that already lags (sensors one physics step
old, feedback at 100 Hz ramped in over 10 ms)." Replace with: "a bench measurement of each drive type's
command-to-motion delay, end to end on the robot's bus, and of servo stiffness".
