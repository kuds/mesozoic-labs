# Stage 1 closed-loop probe: report to the owner

**10 Nov 2026, from the records as they stood at 01:18 UTC (final). Simulation only.** The sources are the machine-made
tables, `STAGE1_TABLES.md`, regenerated at 01:18 UTC from every record after the last evaluation finished at 01:13 UTC,
and the run records behind them. Record paths are relative to `hardware/trex_form_study/stage1/runs/2026-11-08/`, and
`<b>` stands for a body. "Decision n" refers to `STAGE1_PLAN.md`; P, F, E and A rows to `DECISION_LOG.md`. Technical
detail is in the Engineering notes (N1–N11) at the end.

The bodies, in the order of every "a / b / c" figure: `60-N-T-10`, the 60 cm Tapered; `60-N-H-10`, the 60 cm Hybrid;
`75-M-H-10`, the 75 cm Hybrid with the targeted drives and bigger feet.

## Answers

Results are for **R2, the balance tracker** (reference walks plus feedback from on-board sensors) unless said
otherwise. Its gains were fitted twice, giving two **fitted controllers** per body. A condition is **shown** when
both pass: all 20 episodes stay up and at least 17 meet every limit.

**Which body does best on your floors?** In simulation, still the 75 cm Hybrid, though by the plan's pass rule all three
are shown with R2 on the same 14 of 20 gated conditions (the 75 cm body's judged R3 on 12, because of the learned
add-on, not the body). It stays up at both speeds and on most margins goes further: on less grip in turns and brisk,
under larger shoves, with more servo delay and with weaker or slower drives; it walks and turns even without balance
feedback. By meeting every limit, though, it tolerates less servo-stiffness error than either 60 cm body and less
torque scaling than the Hybrid. It costs about $3.7k and 4 kg more. The 60 cm bodies differ little in gated verdicts;
the Hybrid does better at brisk, the Tapered on the fast turn.

**Hard floors, normal and slippery grip.** At the model's default grip (friction coefficient μ 0.9) every body stands,
walks at the nominal pace, turns while walking (0.35 rad/s) and in place (0.5 rad/s), starts and stops, and takes the
test's standing shoves (half the Stage 0 bound) from any side, though forward and backward ones 9–13 % larger topple the
60 cm bodies. At the slippery test grip, μ 0.54, the nominal walk is shown on all three; it stays up down to μ 0.10–0.11
and meets every limit down to 0.21–0.275 (the Tapered 0.71, one noisy peak foot force), against the treads' estimated
0.42–0.8 on vinyl and laminate. Turns were not tested at 0.54; the 60 cm walking turns stay up down to μ 0.30–0.31 but
meet every limit only down to 0.42 (Tapered) and 0.57 (Hybrid, above the test grip), each edge one episode's peak foot
force; the 75 cm ones stay up down to 0.10 and meet every limit down to 0.26. These margins are 3-episode searches on
one fitted controller per body. Every verdict assumes no servo delay and the model's servo stiffness, yet at 60 cm the
nominal walk misses a limit at 6 ms of delay, falls at 8–10 ms and meets every limit only down to 0.67–0.68 × the
holding gains: bench measurements (§6, item 6) decide whether these results carry over.

**Thin carpet.** Standing, yes. Walking, no: every start falls on every body in its first steps (every episode at the
test's μ 0.54, and in a 3-start check at the carpet's own μ 0.95), and training on that carpet (R3: 0 of 40 on the 75 cm
body) did not fix it. The carpet model is unmeasured, so this is a model result, not a verdict on your carpet.

**Speed.** At 60 cm the brisk walk (0.31 m/s) is not shown (the Hybrid is provisional) and the animal pace (about 0.45
m/s) falls in 79 of 80 episodes. The 75 cm body stays up at both speeds but passes neither. With R2, brisk misses only
the tail servo's limit (1.04–1.07×), and that limit is A1, a planning rule at about half the servo's spec-sheet torque,
so a bench test or a stronger servo could change that verdict. The animal pace also misses duty (under one fitted
controller also speed, toe clearance, no-flight and ankle speed), and the judged R3's brisk walk misses speed and duty
on run 43, so a stronger tail servo alone would pass neither.

**Bumps while walking.** A shove at half the Stage 0 bound topples every body most of the time: 0–8 of 20 stay up per
direction, none when shoved backward, and R3, trained with shoves, fares about the same at 75 cm. Walking shoves are
recovered only up to 0.07–0.15 m/s at 60 cm and 0.10–0.17 m/s at 75 cm. The fit never saw a shove: controller work
first.

**What learning (R3) added.** No new pass. On each 60 cm body one of the two learned runs failed its check twice, so by
the plan's rule the body is judged with R2 (the other run had passed). On the 75 cm body both passed (one only under
decision 39's tolerance, set after seeing its 0.003 s miss), so R3 is judged there. It stays up about as often as R2,
but one run turns right slightly too slowly and stops late, making two of R2's 14 shown conditions provisional. Its one
gain is on low pile, the edge of your use: one run stays up in 14 of 20 walks (R2: 0 of 40), none meeting every limit.
Both runs fall in every thin-carpet walk. Learning alone (R4) only learned to stand still, on every body.

**About 80 % at 60 cm for about $15k?** Not shown yet. The 60 cm build does the core hard-floor demonstration and stands
on thin carpet: 14 of 20 gated conditions. The biggest gaps: walking on thin carpet, bumps while walking, brisk and
animal-pace walking, the fast walking turn and the calm look. They look like controller work first; R3 closed none of
them at 75 cm. Open hardware questions: the pads (the pad-edge flag, untestable here), the thigh-twist range, the tail
servo's margin at speed, the treads' grip on vinyl and laminate, and the ankle drive's supply (listed as discontinued;
§6, item 11). The estimate stays $14.8k, but its high side, $20.5–20.6k, is already above P4's $10–20k.

## 1. What was run

Stage 1 asked whether a controller can use the margins the design screens (Stage 0) found. Every controller ran the
same 20 fresh episodes per condition (seeds 3042–3061) from a settled stance, on the nominal plant: no randomisation
and no servo delay.

| Row | Controller |
|---|---|
| R0 | The reference walks played open loop: feedforward and the servos' own position control, no balance feedback. |
| R1 | The tracker fed the true centre of mass: a diagnostic, since a robot cannot measure it. |
| R2 | The tracker on the centre of mass estimated from encoders, the inertial sensor and foot contacts; 13 gains fitted by CMA-ES, seeds 42 and 43. |
| R3 | R2 plus a learned residual (PPO reinforcement learning): small joint offsets and ±25 % pace. The plan's judged controller if it passes G6. |
| R4 | A learned policy alone, the ablation the behaviour spec asks for. |

**Terms.**
- **Stay up (survive):** no fall: trunk tilt under 0.6 rad, root above 0.7 × home height, no clearance proxy below the
  floor.
- **Meet every limit (succeed):** stay up and meet the behaviour spec: speed; duty 0.55–0.65 (the share of the
  stride each foot is down); no flight; roll and pitch; toe clearance; foot slip; each drive within its sustained
  torque and rated speed; joint range; at nominal speed also cadence, stride and peak foot force.
- **Pass:** 20/20 stay up and at least 17 meet every limit. **Provisional:** one fitted controller passes.
- **Gated or reported:** 20 conditions decide R2's verdicts; others, such as the fast walking turn and the carpet
  walk (the tracker was never fitted on carpet, decision 23), are reported only.
- **Body- or controller-limited (decision 20):** each hardware limit active just before the falls is lifted in turn
  over the same 20 starts. If lifting one improves survival significantly, the failure is body-limited by it,
  otherwise controller-limited; failures without falls are labelled by the limits they miss. A controller-limited
  label says nothing about the body.
- **The pad edge:** a flag the tracker raises when its ankle-torque cap binds at what the pad can carry (decision 15):
  more torque would roll the foot. The simulation cannot lift it, so no test could clear it.
- **H:** hip height. **Froude 0.05:** the animal pace, a speed scaled to leg length (0.45 / 0.44 / 0.50 m/s).
  **Δv:** a shove's size, as the velocity change it gives the body. **A1:** the study's own sustained-torque planning
  limit, 0.57 × a drive's rating, until a bench test replaces it.

**Gates.** G1–G4 (plant, references, transitions, tracker) and G7 (metrics reproduce Stage 0) passed. G5 passed on
every body with both fitted controllers (at least 17 of 20 validation episodes of the nominal walk and both target
turns); the fit modelled no servo delay, trunk-mass change or shove, on hard floor only. G6 checks R3 on a short
monitor (six hard-floor tasks and the carpet walk on carpet and low pile, 3 episodes each): at about 1M training steps
it must stay up as long as R2 (G6b; one fix allowed, then G6b again 1M steps later), and at about 2.5M beat R2
somewhere (G6c). Otherwise the body is judged with R2.

**What changed during the run** (N11). The cap became 24 h of wall clock in all, 08:00 UTC 9 Nov to 08:00 UTC 10 Nov, as
the tracker ran 2.4× slower than planned. PPO ran at 74–120 env steps/s per job against the 600 G6a asked, so R3 and R4
were cut from 5M to 4M steps; in the end R3 trained 2.5M on the 75 cm body (1.75–2.02M at 60 cm), and R4, trained to
1.51M at 60 cm, is judged at 1.25M, the last checkpoint all three runs reached. The margin ledger runs one fitted
controller per body, plus R3/42 at 75 cm (no pile damping, decision 37); the stop margin was not run. Decision 39, taken
after seeing a 0.003 s miss, judges G6b within one env step (0.02 s, the resolution of a fall time); it changes no 60 cm
outcome, but without it the 75 cm body would have been judged with R2, shown on two more conditions (§2, N2).

*Evidence:* `STAGE1_PLAN.md` §2–§6 and decisions 13–39; `STAGE1_TABLES.md` (G7; Fit and G5; PPO runs);
`designlab/stage1/panels.py` (`EPISODE_RULE`); `designlab/stage1/control.py` (falls).

## 2. Verdicts

`STAGE1_TABLES.md` holds the full grids, with every row's counts and labels. Below, numbers are episodes that stay up,
of 20, for fitted controllers 42 · 43, and in the last column for R3's learned runs 42 · 43 (both on fitted
controller 43); "(rep.)" marks a condition reported, not gated, for R2. Your floors come first.

| Condition | `60-N-T-10` | `60-N-H-10` | `75-M-H-10` | `75-M-H-10`, R3 (judged) |
|---|---|---|---|---|
| Hard floor: stand; nominal walk at μ 0.9 and 0.54; walking turns 0.35 rad/s; turns in place 0.5 rad/s; start and stop; standing shoves, 4 directions (12 conditions) | shown | shown | shown | shown on 10; provisional on the right walking turn and start and stop (all stay up; 20 · 6 and 20 · 11 meet every limit) |
| Walking shoves, stay up of 40: forward / back / left / right | 2 / 0 / 2 / 2 | 4 / 0 / 4 / 14 | 2 / 0 / 11 / 12 | 3 / 0 / 14 / 13 |
| Brisk walk | 17 · 0 | provisional: 5 · 20 | 20 · 20, tail servo over its limit | 20 · 20, tail servo over its limit; under 43 also speed and duty |
| Animal pace (Froude 0.05) | 1 · 0 | 0 · 0 | 20 · 20, tail servo and duty fail | 20 · 20, tail servo and duty fail; under 43 also speed, no-flight, toe clearance and ankle speed (1.02–1.23× rated) in all 20 |
| Fast walking turn, 0.5 rad/s (rep.) | right shown; left misses limits | misses limits | shown | shown |
| Thin carpet: stand | shown | shown | shown | shown |
| Thin carpet: walk (rep.) | all fall, 3.7–3.9 s | all fall, 3.7–3.8 s | all fall, 4.7–5.2 s | gated: all fall, 5.0–5.7 s |
| Low pile: stand; walk (rep.) | shown; all fall, 3.1–3.4 s | shown; all fall, 3.1–3.2 s | shown; all fall, 4.6–12.4 s | shown; gated: 14 · 0 stay up, none meets every limit |
| Medium and high pile: walk (rep.) | not run: no settled stance | not run | all fall, 3.5–5.4 s | all fall, 3.6–5.4 s |
| Gated verdicts: shown / provisional / failing | 14 / 0 / 6 | 14 / 1 / 5 | 14 / 0 / 6 | 12 / 2 / 8, of 22 |

- **What limits it.** Every failing gated condition at 60 cm reads controller-limited. At 75 cm brisk and the animal
  pace read body-limited by the tail servo's planning limit, and the backward walking shove by slip, though lifting slip
  only delays its falls; its other walking shoves are controller-limited. The pad-edge flag preceded every R1–R3 fall
  and no R0 or R4 one (N1).
- **Carpet.** Every R2 thin-carpet fall, and every 60 cm low-pile fall, comes during the start from standing, before the
  carpet gait's higher swing begins. R3's falls also begin in the start, run 42's near its end (19 of its 20 falls
  complete just after it). It is not grip: in a 3-start check per body at the carpet's own friction (0.95) the nominal
  walk falls at the same moments (decision 37). At 75 cm these falls read slip-limited, but lifting slip saves none.
- **Speed at 60 cm.** The two fitted controllers disagree completely at brisk: the walk sits at its stability edge.
  At the animal pace 30 of the 79 falls come before the Froude reference begins (N5); lifting any drive limit saves
  none.
- **Speed at 75 cm.** The tail servo, a Feetech STS3215, runs at 1.04–1.07 of its A1 limit at brisk and 1.17–1.31 at the
  animal pace; A1, 0.491 N·m, is about half the spec sheet's rated torque (N6). The animal pace also fails duty on both
  fitted controllers, and under 43 ankle speed (up to 1.31×), speed, toe clearance and no-flight in some episodes. Under
  fitted controller 42 the knee, which governed the Stage 0 screens, peaks at 0.86–0.88 of its rating.
- **The fast walking turn at 60 cm** stays up 20/20 but misses limits, on the Hybrid mostly the thigh twist's range
  (decision 14): 8–9 of 20 meet every limit, against 14–18 on the Tapered.
- **R3 on the 75 cm body.** Run 43 makes two hard-floor conditions provisional without a fall, both controller-limited:
  it turns right at 0.347–0.351 rad/s, under the 0.35 target in 14 of 20, and fails to stop within 1.5 strides in 9 of
  20 start-and-stops. R3's other failures carry R2's labels, except that its forward walking shove reads slip-limited
  (lifting slip saves 2 more of 20 on run 42, none on 43).

*Evidence:* `STAGE1_TABLES.md` (verdict grids; verdicts by controller); `<b>/eval/R2/{42,43}/*/*.json` and
`75-M-H-10/eval/R3/{42,43}/*/*.json` (`counts`, `failure`, `attribution.lifted`, `fall_time_s`, `failed_terms`,
`measures.window_sustained_ratio`, `worst_speed_ratio`, `measures.window.yaw_rate_rad_s`); the traces
`<b>/eval/traces/{R2,R3}/*/{walk,carpet}/*/<seed>.npz` (`t_ref`) against `<b>/transitions.json`.

## 3. The margin ledger

Each margin of plan §5 is pushed toward failure by bisection on one fitted controller per body (42, 43, 43: the
body-level fits that R1 and R3 also use), 3 episodes per value (seeds 3042–3044). Each cell reads **stays up to / meets
limits to**: all 3 stay up; none misses a limit it met at the nominal value. † marks the end of the search with nothing
failing. The last column is R3/42, the learned residual on the 75 cm body's fitted controller 43.

| Margin | Stage 0, a / b / c | `60-N-T-10` | `60-N-H-10` | `75-M-H-10` | `75-M-H-10`, R3/42 |
|---|---|---|---|---|---|
| Grip, nominal walk (lowest μ) | 0.48 / 0.47 / 0.36 | 0.10† / 0.71 | 0.11 / 0.275 | 0.10† / 0.21 | 0.10† / 0.28 |
| Grip, walking turn, worse side | 0.54 / 0.53 / 0.41 | 0.31 / 0.42 | 0.30 / 0.57 | 0.10† / 0.26 | 0.10† / 0.325 |
| Grip, brisk walk | 0.75 / 0.73 / 0.56 | falls at 0.9 | 0.53 / 0.575 | 0.11 / 0.29 | 0.11 / 0.34 |
| Standing shove forward, Δv m/s | bound 0.487 / 0.478 / 0.589 | 0.26 / 0.26 | 0.25 / 0.25 | 0.33 / 0.31 | 0.30 / 0.30 |
| Standing shove back; sideways (stays up) | none | 0.27; 0.55–0.56 | 0.26; 0.55 | 0.34; 0.63–0.64 | 0.35; 0.63–0.64 |
| Walking shove, 4 directions | none | 0.07–0.15 | 0.07–0.13 | 0.10–0.17 | 0.08–0.18 |
| Stop: deceleration held | 1.25 / 1.25 / 1.41 m/s² | not run (plan deviation, decision 33) | the same | the same | the same |
| Drive speed, animal pace (lowest scale) | knee needs 0.81 / 0.83 / 0.81 of rated | falls at 1.0 | falls at 1.0 | 0.29 / 0.50 (N4) | 0.26 / 0.87, the knee (N4) |
| Drive torque, nominal walk (lowest scale) | worst drive 0.76 / 0.76 / 0.66 | 0.53 / 0.92 | 0.48 / 0.84 | 0.29 / 0.88 | 0.36 / 0.86 |
| Servo stiffness, controller unaware (lowest scale) | none | 0.58 / 0.68 | 0.58 / 0.67 | 0.55 / 0.71 | 0.57 / 0.72 |
| Servo delay, ms | none | 6 / 4 | 8 / 4 | 20† / 6 | 20† / 6 |
| Trunk mass (highest scale) | none | 1.3† (+13 % robot) | 1.3† (+13 %) | 1.3† (+12 %) | 1.3† (+12 %) |
| Pack voltage, 25.2 / 22.2 / 21 V | 24 V used | walk works at all; animal pace falls at all | the same | walk works at all; animal pace stays up at all, with no new miss at 25.2 and 22.2 V and a new no-flight miss in one episode at 21 V; at no voltage does an episode meet every limit (as at nominal: tail servo, duty, ankle speed, toe clearance) | walk works at all; animal pace stays up at all with no new miss, but no episode meets every limit (tail servo, duty) |
| Pile damping × 0.5–2 | none | low-pile stand holds; carpet walk falls at nominal | the same | the same | not run (decision 37) |

**What binds at each edge** (R2; R3/42's edges are in N4).
- **Grip.** Every fall at a grip edge is slip-limited: lifting slip lets all 3 stay up. The limit edges are one
  episode's peak foot force (noisy, N4: `60-N-T-10`'s walk and right turn, `60-N-H-10`'s right turn), `60-N-H-10`'s
  speed target (walk), thigh-twist range (left turn, 0.44) and a slip fall (brisk), and `75-M-H-10`'s foot slip over
  5 mm, turn rate and brisk speed target.
- **Shoves.** Standing forward and backward falls are controller-limited on every body: nothing liftable was active,
  except slip before one 75 cm backward fall, and lifting it saved none. At 75 cm the first standing miss is recovery
  forward and sideways, but a fall backward (0.35 m/s). Walking shoves are mostly controller-limited; three 60 cm
  directions read slip.
- **Drive speed.** The 60 cm animal pace falls at full drive speed, and lifting drive torque or slip saves none: drive
  speed is not where the 60 cm tracker breaks; walking speed is. The 75 cm one stays up down to 0.29 × drive speed,
  where lifting the torque–speed limit saves all 3: body-limited.
- **Drive torque.** Falls are body-limited by drive torque or slip. The limit edges are a servo's A1 limit, the neck
  at 60 cm and the tail at 75 cm (one peak foot force on `60-N-T-10`).
- **Stiffness and delay.** Controller-limited: toe clearance or a peak foot force goes first, then falls where nothing
  liftable was active (60 cm stiffness) or lifting drive torque and slip saved none. The tracker was fitted with no
  delay, and the stiffness sweep lowers the servo gains without telling the controller: these are provisional bench
  targets, not requirements (N4).
- **Mass and voltage** do not bind. At 21 V the 75 cm animal pace newly loses no-flight in one episode.

**Against Stage 0.** The closed loop stays up on far less grip than Stage 0 required (μ 0.10–0.31 against 0.36–0.56),
but the 60 cm brisk walk gains least: `60-N-H-10`'s stays up only down to μ 0.53 (Stage 0 asked 0.73), just under the
test's 0.54, with single falls at up to 0.57; `60-N-T-10`'s falls even at 0.9. It recovers forward standing shoves to
only about half of Stage 0's ideal bound and walking shoves to 0.14–0.31 of it. It stays up with the drives at
0.29–0.53 of their torque, but servo loads run above Stage 0's: the 60 cm neck at 0.83–0.84 of its A1 limit in the
nominal walk (Stage 0: 0.76), the 75 cm tail at 0.87–0.88 (Stage 0: 0.40).

**R3/42 against R2.** Close and mixed. It stays up on as little grip, and its stiffness, delay and mass margins match
within a search step or two. It needs more grip to meet every limit (μ 0.28 / 0.325 / 0.34 against 0.21 / 0.26 / 0.29:
walk, worse turn, brisk) and more drive torque to stay up (0.36 against 0.29), and survives smaller forward standing
and left walking shoves (0.30 and 0.08 against 0.33 and 0.10 m/s); but it meets every limit on the right turn down to
μ 0.22 (R2: 0.26) and stays up down to 0.26 × drive speed (0.29) and after larger backward standing and forward walking
shoves (0.35 and 0.18 against 0.34 and 0.17 m/s). At the animal pace it keeps the ankles within their rated speed, so
its drive-speed search reads the knee, which binds at 0.87 × drive speed (Stage 0's need: 0.81).

*Evidence:* `<b>/sweeps/R2/{42,43}/*.json` (`summary`, `evaluations`, `rule`; written 22:31–22:44 UTC at 60 cm and
23:25–23:47 UTC at 75 cm) and `75-M-H-10/sweeps/R3/42/*.json` (written 01:07–01:13 UTC); `STAGE1_TABLES.md` (Margin
ledger, with its Stage 0 column); `STAGE1_PLAN.md` §5 and decision 33; the candidates' `margins` in
`round2/refine/{narrow_carpet_60,feet_75}/candidates/*.json`.

## 4. Ablations: R0 to R4

**R0 and R1 against R2.** Episodes that stay up, of 20, as R0 / R1 / R2. R2 is the body-level fitted controller,
whose gains R1 also uses.

| Condition | `60-N-T-10` | `60-N-H-10` | `75-M-H-10` |
|---|---|---|---|
| Nominal walk | 3 / 18 / 20 | 1 / 18 / 20 | 20 / 20 / 20 |
| Nominal walk, μ 0.54 | 5 / 14 / 20 | 7 / 12 / 20 | 20 / 20 / 20 |
| Walking turn 0.35 rad/s, left | 2 / 19 / 20 | 0 / 12 / 20 | 20 / 20 / 20 |
| Walking turn 0.35 rad/s, right | 2 / 18 / 20 | 2 / 17 / 20 | 20 / 20 / 20 |
| Turns in place, left and right | 20 / 20 / 20 | 19–20 / 20 / 20 | 20 / 20 / 20 |
| Start and stop | 4 / 20 / 20 | 6 / 15 / 20 | 20 / 20 / 20 |
| Brisk | 0 / 15 / 17 | 0 / 15 / 20 | 20 / 20 / 20 |
| Animal pace | 0 / 1 / 1 | 0 / 0 / 0 | 16 / 20 / 20 |
| Walking shoves, per direction | 0 / 0–2 / 0–2 | 0 / 0–5 / 0–7 | 0–1 / 0–7 / 0–5 |

- **R0, open loop.** The 60 cm bodies stand and turn in place but fall in most walks, in every brisk and animal-pace
  episode and under every walking shove: balance feedback is what makes the 60 cm walks work. Open loop the 75 cm body
  passes the nominal walk at both grips, all four target turns, the fast turns and start and stop, every limit met, and
  stays up at brisk and in 16 of 20 animal-pace episodes. Without feedback no episode gets back on its command within
  3 s after a forward or backward standing shove; on `60-N-T-10`, 14 of 20 fall after the forward one.
- **R1, perfect state.** At 60 cm it fell *more* often than R2 on the walks and turns, probably because the gains
  were fitted with the estimator in the loop; no record tests that. At 75 cm it stays up as R2 does except on the
  right walking shove (7 against 4), and at the animal pace misses only the tail servo's limit, where R2 with the same
  gains also misses duty, ankle speed, toe clearance, speed and no-flight. Better state estimation is not the first
  place to spend effort, though at 75 cm and speed it costs some margin.

**R3, the learned residual: the G6 history.** Mean survival over the G6 monitor, R3 against R2, in seconds.

| Run | G6b, 1.0M steps | After the fix, 2.0M | G6c, 2.5M | Outcome |
|---|---|---|---|---|
| `60-N-T-10` 42 | 14.14 vs 15.98 | 15.16 vs 15.98 | not reached | failed twice; stopped |
| `60-N-T-10` 43 | 15.41 vs 15.98 | 16.00 vs 15.98, pass | not reached | stopped (decision 38) |
| `60-N-H-10` 42 | 15.03 vs 15.49 | 14.30 vs 15.49 | not reached | failed twice; stopped |
| `60-N-H-10` 43 | 15.92 vs 15.49, pass | not reached | not reached | stopped at 1.75M (decision 38) |
| `75-M-H-10` 42 | 17.99 vs 18.00 (a) | 18.07 vs 18.00, pass | 18.09 vs 18.00, pass | judged at 2,506,752 |
| `75-M-H-10` 43 | 16.94 vs 18.00 | 17.997 vs 18.000 (a) | 18.002 vs 18.000, pass | judged at 2,506,752 |

(a) Failed as recorded; passes under decision 39, set after seeing seed 43's result (N2).

- The 60 cm runs failed mainly by falling on the brisk walk (once also on a walking turn), in 1–3 of 3 monitor episodes
  where R2 fell in at most one. The one allowed fix weighted the brisk and animal-pace walks ×2. Seed 42 failed again on
  both 60 cm bodies, so the plan judges both with R2, although seed 43 had passed.
- At G6c both 75 cm runs beat R2 on reward per step over 3 episodes (+0.011 to +0.025), with no hard-floor falls: on the
  walk, both walking turns, a turn in place and brisk, and seed 43 also on the low-pile carpet walk while falling as
  often as R2 (N3). Both still fell in every carpet and low-pile episode, up to 0.7 s later than R2.

**R3's 20-episode panels on the 75 cm body** (runs 42 · 43, both on fitted controller 43's gains) did not turn those
wins into limits met (N3). Run 42 passes every hard-floor condition R2 passes; run 43 turns right too slowly, stops
late and walks too slowly at brisk and the animal pace (§2), and it sways more at speed (0.145 and 0.143 H mean at
brisk and the animal pace; R2: 0.115–0.127). Both stay up after about as many walking shoves as R2 (30 of 160 against
25) and fall in every thin-carpet walk; on low pile run 42 stays up in 14 of 20 walks (R2: 0 of 40), none meeting every
limit.

**R4, the policy alone,** is judged at 1,253,376 steps, a shorter budget than R3's: its runs were not resumed after the
machine went down (decision 36). On every body it learned only to stand still: in every walk, turn and carpet walk it
keeps both feet down (duty 1.0) and moves at 0.00 m/s, so it stays up 20/20, long pile included at 75 cm, but every
episode misses speed, duty, toe clearance and slip. It stands 20 s on every floor (provisional, as one seed) and stays
up after sideways and backward standing shoves. It falls from every forward shove at 60 cm; at 75 cm it stays up after
10 of 20 standing and 17 of 20 walking ones, never getting back on its command after the walking ones. Its
"body-limited: slip" labels in the tables are an artefact (N1).

**Sensitivity.** One fitted tracker is not a reliable read of a body: the two fits disagree completely on the 60 cm
brisk walk. Of the plant's doubts (§3), servo delay and stiffness matter for the nominal walk; trunk mass and voltage
do not. The 75 cm fits keep a small walking step-placement gain (0.06–0.10) where all four 60 cm fits set it to 0, so
compare shove results across sizes with care.

*Evidence:* `<b>/eval/{R0,R1,R4/42}/*/*.json` and `75-M-H-10/eval/R3/{42,43}/*/*.json` (`counts`, `failed_terms`,
`fall_time_s`, `measures.window`: `speed_m_s`, `duty`, `yaw_rate_rad_s`, `slip_median_m`; `measures.carpet`;
`measures.start_stop`); `STAGE1_TABLES.md` (R0, R1, R3 and R4 columns; Looks; PPO runs); `<b>/ppo/*/run.json`
(`g6.*.monitor`: `mean_survival_s`, `g6c_by_condition`, `policy_beats_zero_on`; `g6.*.amended`; `stopped`, `unstopped`,
`judged_checkpoint`); `<b>/policy_only/42/run.json`; `<b>/eval/R2/*/walk/nominal.json` (`controller.gains`); decisions
24, 27, 32, 38 and 39.

## 5. Budget used

The last evaluation record was written at 01:13:34 UTC on 10 Nov, 17.2 h into decision 21's 24 h cap, which held with
6.8 h to spare. For 1.7 h of it no work was recorded: nothing was written between 19:39:34 and 21:23 UTC, across two
stops of unrecorded cause; a third stop, at about 01:28 UTC, came after the last evaluation and lost nothing (N8). The
60 cm bodies finished at about 23:50 UTC and the 75 cm body at 01:13 UTC, 15.8 h and 17.2 h after their fits began at
08:01.

| Core-hours | `60-N-T-10` | `60-N-H-10` | `75-M-H-10` | Basis |
|---|---|---|---|---|
| Prepare, G1–G4 (before the cap) | 0.21 | 0.21 | 0.21 | wall time |
| Fit, 2 × 720 CMA-ES evaluations | 5.22 | 5.26 | 6.12 | measured |
| G5 | 0.38 | 0.38 | 0.39 | lower bound |
| PPO: R3 × 2 and R4 × 1 | 8.2 | 8.2 | 10.3 | estimate (N7) |
| Panels, R0–R4 | 4.07 | 4.08 | 6.13 | upper bound |
| Ledger: R2, and R3/42 at 75 cm | 0.31 | 0.35 | 0.84 | CPU time |
| **Total** | **18.4** | **18.5** | **24.0** | **60.9 in all** |

- By this upper-bound accounting the total is close to the 62 core-hours four cores give over the 15.5 h that work ran
  inside the cap, and the machine was busy most of that time. `STAGE1_TABLES.md` shows 29.3 / 29.3 / 33.6 (92.2 in all),
  because it counts every training session cut by the fix or the outage as running until 00:12 UTC (N7).
- The plan expected about 10 core-hours per body; the fit alone took 16.6 across the bodies, as decision 21 foresaw.
  The steps R3 and R4 reached are in §1.

*Evidence:* `STAGE1_TABLES.md` (Core-hours per step); `designlab/stage1/report.py` (`_intervals`, `_ppo_times`,
`costs`); `<b>/tune/{42,43}/cma.json` (`sessions`); `<b>/ppo/*/run.json` and `<b>/policy_only/42/run.json` (`sessions`,
`throughput`, `fix.at`, `stopped.at`); the checkpoint files' times; `<b>/eval/summary.json` and
`<b>/sweeps/summary.json` (`sessions`); `75-M-H-10/eval/{R3,R4}/**/*.json` (`episode_wall_s`, `attribution.wall_s`) and
`75-M-H-10/sweeps/R3/42/*.json` (`evaluations`: `cpu_s`).

## 6. What verification training should carry over

1. **Fit on everything that is judged.** The fit held only the nominal walk and the two target turns on hard floor.
   Add carpet (thin carpet first, per P26, covering the start from standing), μ 0.54, brisk, the animal pace and
   walking shoves to the fit and to G5. Each fit evaluation, now about 55 simulated seconds, would grow 2.5–3 times,
   so the refit takes well over decision 23's 6.5 h on four cores unless evaluations are cut.
2. **Walking shoves.** Refit with walking shoves in the mix so that the existing step-placement gain is used (the fit
   drove it from the prototype's 1.0 to 0 on all four 60 cm fits), extend it if needed, and judge walking shoves. R3,
   trained with shoves, fared about the same at 75 cm (§4).
3. **Standing shoves.** Fit the capture step's margin, held at 0 because the fit had no shove; the capture step fired
   in none of the 72 tracker shove records (R1 and R2).
4. **The brisk walk's stability edge.** Judge brisk on more seeds and weight it in the fit.
5. **The tail servo's limit.** Run A1's "replace with", a bench thermal and duty test of the STS3215, before
   re-sizing the head and tail servos; and find why closed-loop loads exceed Stage 0's.
6. **Servo delay and stiffness.** Measure both on the bench; keep randomising delay in training (R3 trains with
   0–20 ms) and add it to the tracker's fit.
7. **A test that changes the pad.** No narrow 60 cm model with bigger feet exists; one needs its own prepare, gates and
   fit (about 5.2 h of wall clock per body) and changes stance width and gains too. The 75 cm body, with the bigger
   feet, still shows the pad-edge flag before every fall.
8. **The learned parts.** The residual barely moved (about 1 % of its authority at 1M steps) and at 2.5M steps passed
   nothing R2 does not; R4 never found a gait in 1.25M steps. Train on a sturdier base controller or with more
   exploration (N3), and let G6c count limits met over more episodes rather than reward per step over 3 (its
   right-walking-turn win missed the turn rate in 14 of 20 panel episodes). Only R3's training shoves (to the full
   Stage 0 bound) and, at 60 cm, its delay (to 20 ms) reach past where R2's nominal walk stays up; its friction, mass
   and stiffness ranges do not (§3).
9. **Memory-bounded evaluation** (decision 36): one behaviour group per process, at most seven processes, a 1.5 GB
   free-memory guard.
10. **Guard against stops** during long runs: none of the three stops' causes is recorded (N8), so log the machine's
    state, and keep the container alive or run where containers are not reclaimed.
11. **Sourcing.** MyActuator lists the 60 cm picks' ankle drive (RMD-X4-P12-10-E, V4) as discontinued, though its US
    store still sells it ($285), and E6 excludes discontinued models. Confirm supply, or re-check the ankle against
    its successor, the X4-10-L (thermal equilibrium at 2.0–2.9 N·m against the study's 2.28 N·m planning limit).

*Evidence:* `<b>/tune/42/cma.json` (`mix`, `simulated_s_per_evaluation`); `<b>/tune.json` (`fixed_gains`);
`designlab/stage1/control.py` (`GAIN_SPEC`, `Tracker._capture`); `<b>/ppo/42/run.json` (`env_configs`: `latency_steps`,
`friction`, `push_prob`, `push_frac`); `designlab/stage1/env.py` (`_draw_push`, `RESIDUAL_*`, `KP_RULE`);
`<b>/eval/R{1,2}/**/push/*.json` (`captures`); `designlab/data/actuators.json` (`myactuator_rmd_x4_p12_10_e`,
`feetech_sts3215_c018`); `ROUND2_REVIEW.md` §7.5 and §7.6.

## 7. Draft decision-log rows for your approval

**F26 (finding), proposed.** Stage 1 closed-loop probe (simulation, 8–10 Nov 2026; `STAGE1_PROBE.md`). The balance
tracker R2, fitted by CMA-ES on the nominal walk and both target turns on hard floor (no actuation delay or push in the
fit), is shown (both fitted controllers: 20/20 survive, at least 17/20 succeed) on the same 14 of 20 gated conditions on
every body: standing on hard floor, robot_floor_carpet and low pile; the nominal walk at μ 0.9 and 0.54; the 0.35 rad/s
walking and 0.5 rad/s in-place turns; start and stop; standing pushes at half the Stage 0 bound. No body is shown on the
brisk or Froude 0.05 walks or on walking pushes (0–8 of 20 survive per direction). `75-M-H-10` survives both speeds
20/20 but none succeeds: the STS3215 tail servo runs at 1.04–1.31 of its A1 planning limit, and at Froude 0.05 duty
fails too, as do, under fitted controller 43 in 4–14 of 20, the ankles' rated speed, the target speed, toe clearance and
no-flight. The 60 cm bodies survive 17 · 0 and 5 · 20 of 20 at brisk (`60-N-H-10` provisional) and 0–1 of 20 at Froude
0.05. None walks on carpet or pile: on robot_floor_carpet every episode falls during the start from standing, also at
the carpet's own μ 0.95 (3 starts per body). Most 60 cm failures are controller-limited (decision 20); the pad-edge
flag, set by the tracker's ankle cap before every tracker fall, could not be tested. Open loop (R0) the 60 cm bodies
fall in most walks while `75-M-H-10` passes the nominal walk, every target turn and start and stop; with the true centre
of mass and the same gains (R1) no body stays up more often except on two walking-push conditions. In the margin ledger
(3 episodes per value, one fitted controller per body; `60-N-T-10` / `60-N-H-10` / `75-M-H-10`) the nominal walk
survives up to 6 / 8 / at least 20 ms of actuation delay (20 ms ends the search; every limit met to 4 / 4 / 6 ms), plant
servo stiffness down to 0.58 / 0.58 / 0.55 × the holding gains with the controller unaware, and μ down to 0.11 on
`60-N-H-10` and to the search's floor, 0.10, on the others. On each 60 cm body R3's seed 42 failed G6b twice, so both
are judged with R2. On `75-M-H-10` both R3 runs passed G6b after the fix (seed 43 only under decision 39's tolerance,
set after seeing its 17.997 vs 18.000 s) and G6c at 2,506,752 steps, where R3 is judged: shown on 12 of R2's 14
conditions and provisional on the other two (seed 43 turns right under 0.35 rad/s and stops late), it fails brisk,
Froude 0.05, walking pushes and both carpet walks it is gated on (seed 42 stays up in 14 of 20 low-pile walks, none
succeeding). R4 (policy only, judged at 1,253,376 steps) learned on every body only to stand still (0.00 m/s, duty 1.0).
*Evidence:* `stage1/runs/2026-11-08/` (`STAGE1_TABLES.md`, `<b>/eval/`, `<b>/ppo/*/run.json`,
`<b>/policy_only/42/run.json`, `<b>/sweeps/`); decisions 21–39. *Status:* proposed.

**A15 (behaviour envelope), proposed update.** Keep the current text, then add:
> **Stage 1 (F26):** in closed loop the push bound is far from reached. At half the bound R2 survives 0–8 of 20 walking
> pushes per direction on every body and recovers walking pushes only up to 0.07–0.15 m/s at 60 cm and 0.10–0.17 m/s at
> 75 cm. It survives and recovers from (is back on the command within 3 s after) forward standing pushes up to 0.25–0.26
> m/s at 60 cm, about half the 0.478–0.487 m/s bound; at 75 cm it survives 0.33 m/s (0.56 of 0.589 m/s) but recovers
> only to 0.31. Sideways ones are survived to 0.55–0.56 and 0.63–0.64 m/s, though the bound is a forward one. All are 3
> episodes per value on one fitted controller per body. R3 on `75-M-H-10`, trained with pushes up to the full bound,
> fares about the same (0–8 of 20 walking pushes per direction).
>
> At 60 cm the Froude 0.05 walk falls at full drive speed (25.2 V is the nominal plant; 22.2 and 21 V only lower speed),
> lifting any drive limit saves none of the falls, and 30 of its 79 falls at μ 0.9 come before the Froude reference
> begins. At 75 cm it survives. Under fitted controller 42 the knee peaks at 0.86–0.88 of its rated speed and only the
> tail servo's limit and duty fail. Under 43 the ankle drives exceed their rated speed (1.13–1.31× in 14 of 20;
> body-limited: drive_sustained, drive_speed), and in the ledger the walk survives every drive's speed scaled to 0.29.
> R3 on 43's gains splits: run 43 exceeds the ankles' rating in all 20 (1.02–1.23×), while run 42 stays within it
> (once 1.02× at μ 0.54) and its ledger finds the knee binding first, at 0.87 × drive speed (knee peak 0.86 of rated;
> Stage 0's need 0.81). So a drive-speed limit does bind in closed loop on one fitted controller's gains: on the ankles,
> not the knee the screens flagged.

"Replace with" becomes: "a controller whose walking step-placement gain is fitted with walking pushes in the mix,
and with the Froude 0.05 walk in its fit (verification training); the Stage 2 bench for the knee's and the 75 cm
ankles' speed". Status: partly tested.

**A16 (carpet screen), proposed update.** Keep the current text, then add:
> **Stage 1 (F26):** the probe used robot_floor_carpet, the repository's robot_floor surrogate (friction 0.95; stiffness
> and damping solref 0.024 / 1.2; all model values, "calibration hypotheses" with no source or specimen fit), and the
> low, medium and high pile models (6.35 / 12.7 / 19.05 mm, on a 4 mm probe grid; PPO trains on an 8 mm grid). The
> carpet walk runs at μ 0.54. Under R2 every body stands 20 s on robot_floor_carpet and on low pile, and none walks on
> any carpet or pile: it falls in every episode, on robot_floor_carpet during the start from standing, also at the
> carpet's own friction (0.95; 3 starts per body). R3, trained a quarter of the time on each of robot_floor_carpet and
> low pile and judged on `75-M-H-10`, falls in all 40 robot_floor_carpet walks and 26 of 40 low-pile walks; its seed-42
> run stays up in 14 of 20 low-pile walks, none meeting every limit. R4 stays up on every carpet and pile only by
> standing still. On every body the low-pile stand holds across pile damping × 0.5–2 (R2; R3's search is cut,
> decision 37).
>
> With the carpet model unsourced, not only its damping, these falls show that the tracker does not transfer to
> this carpet model; they are not carpet results. Per P26, the thin, compact carpet comes first, and the long-pile
> parts of A16 matter less.

"Replace with" becomes: "measured friction, compression and damping of a thin, compact (school) carpet first; a
foot-on-carpet bench test; a carpet-aware controller fit in simulation, covering the start from standing, before
hardware". Status: open.

## 8. The 60 cm gap map (P25)

Your floors first, then the looks, then the edge of the use and beyond. "Limited by" comes from the attributions and
the ledger's lifts where they exist; "Looks like" is a judgement from those labels, not a test.

| # | Gap at 60 cm | What the records show | Limited by | Looks like |
|---|---|---|---|---|
| G1 | Bump while walking | 0–7 of 20 stay up per direction at half the Stage 0 bound; shoves of only 0.07–0.15 m/s recovered; R3 at 75 cm, trained with shoves, about the same | Controller on most; step-placement gain fitted to 0 | **Software first** |
| G2 | Brisk walk (0.31 m/s) | 17 · 0 and 5 · 20 of 20 stay up; tail servo at 0.85–0.91 of its A1 limit (N6) | Controller by decision 20; slip in one of the three failing records | Software first, then grip; then perhaps the tail servo |
| G3 | Animal pace (0.45 m/s) | 0–1 of 20 stay up; 30 of 79 falls come before the Froude reference; tail servo at 0.93–1.00 of A1 in the survivors (N6) | Controller on 3 of 4 records, slip on 1; not drive speed | Software first; then perhaps the tail servo, as at 75 cm; the knee's thin Stage 0 margin stays a bench question |
| G4 | Slippery hard floors (treads estimated at 0.42–0.8 on vinyl and laminate, 0.6–0.85 on polished concrete, 0.8–0.9 on wood) | Nominal walk shown at μ 0.54; turns stay up to 0.30–0.31 but meet every limit only to 0.42–0.57, each end one episode's peak foot force (without it, the Tapered's turns meet every limit to 0.32–0.35 and the Hybrid's left turn to 0.44); brisk falls at up to 0.57, or at full grip | Slip for the falls; peak foot force and joint range for the limits | Mixed: controller for walk and turns; tread for brisk |
| G5 | Fast walking turn, 0.5 rad/s (stretch) | All stay up; 8–9 (Hybrid) and 14–18 (Tapered) of 20 meet every limit | Thigh-twist range, the generator's fixed ±0.30 rad | A range not yet assessed for packaging, or a new turn reference |
| G6 | Walking on thin, compact carpet | 0 of 20 on every fitted controller, falling in the start from standing; at 75 cm R3, trained on this carpet, 0 of 40 too | Controller: nothing liftable helps; never fitted on carpet; not grip; model unsourced | **Software first** |
| G7 | Calm body motion | Sway 0.176–0.180 H against 0.12 H; the reference itself sways 0.168–0.169 H | The reference walk (decision 9) | **Software**: a new reference walk |
| G8 | Livelier head and tail | Head nod 0.051 rad (band 0.05–0.15), tail 0.120 rad (band 0.1–0.2), in band in every episode but at its low end, both set by the reference; neck servo at 0.83–0.84 of its A1 limit | The reference; servo headroom under A1 | Both |
| G9 | Low pile (edge of use) | Stands; every walk falls in the start (at 75 cm one R3 run stays up in 14 of 20, none meeting every limit) | Controller, as G6 | Software |
| G10 | Medium and high pile (outside use) | No settled stance with the round-two feet: not run | Feet, or active ankle balance (F13) | Feet or controller, if ever wanted |
| G11 | Standing bumps a little above the test's | The panels' 0.24 m/s passes 20/20; a forward shove 9 % larger topples both bodies in all 3 ledger episodes | Controller by decision 31: the pad-edge flag was the only candidate; capture margin never fitted | **Software first** |
| G12 | Autonomy: following and obstacle avoidance (P3), the onboard computer (P14), a safety stop | Not tested in Stage 1 | Not assessed | Not assessed |

Software-looking: G1, G6, G7, G9 and G11, and G2 and G3 in the first instance. Mixed or open: G4, G5, G8 (its servo
headroom rests on A1) and G10. Not tested: G12. R3, the learned add-on tried on the 75 cm body, closed none of G1, G6 or
G9 there, so "software" here means the tracker's fit and references (§6). The untested pad-edge flag (N1) sits behind
every fall-based gap (G1–G4, G6, G9, G11); for G11 it was the only candidate, so its "software first" rests on decision
31's labelling rule alone. The 75 cm body, with bigger feet, shows it before every fall too, so bigger feet alone are
not shown to remove it: some of what looks like software there may still be the foot.

Options to close the gaps. Figures come from the round-two record as corrected by two checkers (N9).

| Option | Gaps | Mass | Cost, nominal / high | Main risk | Status |
|---|---|---|---|---|---|
| Controller fit with carpet (start included), μ 0.54, brisk, animal pace and walking shoves | G1–G4, G6, G9 | 0 | compute, well over 6.5 h on four cores | Untested (R3, trained on all of these, closed none of these gaps at 75 cm); carpet model unsourced; pad edge may bind | Not run; first follow-up |
| Walking-shove refit so the step-placement gain is used | G1 | 0 | compute | Stage 0's 0.48–0.49 m/s (0.487 Tapered, 0.478 Hybrid) is an ideal ceiling | Not run |
| A fitted capture margin | G11 | 0 | compute | Acts only while standing | Not run |
| A learned residual on the tracker (R3, as trained here) | G1, G6, G9 | 0 | compute, about 8 core-hours for two runs of 2.5M steps | Tested on the 75 cm body only, where it closed none (walking shoves 0–8 of 20; every thin-carpet walk falls; on low pile one run stays up in 14 of 20, none meeting every limit) and one run lost the turn rate and the stop; at 60 cm it failed G6b twice | Tested at 75 cm; not a fix as trained |
| Floor-dependent speed and yaw limits; tail cancels stance-foot yaw | G4, G2 | 0 | compute | Not designed or simulated (F21) | Not designed |
| Treads measured on your floors (cast PU, 50–60 Shore A) | G4, G2 | not given | about 1 day and $100 | Every μ is an estimate until then | Pending (P21) |
| Bigger feet (toes × 1.15, pads × 1.25) | G4, G10 | +0.09 kg | +$70–71 / +$104–105 | Base-width bodies, not the picks; ankle drive to 0.87 of its limit; not run at 60 cm in Stage 1; 75 cm still shows the pad edge | Not adopted (P20) |
| Knee transmission, a lower ratio | G3 (Stage 0 margin) | not in record | not in record | Hip pitch binds next; Stage 1's 60 cm animal-pace falls are not drive-limited | Not studied |
| Faster knee (RobStride 06) | G3 (Stage 0 margin) | +0.74 kg | +$18–20 / +$88–91 | Base-width bodies only; runtime −6 min; turn grip need rises to 0.71–0.73 | Not adopted (F23) |
| Tail servo: A1's bench test (§6, item 5), then a stronger servo or lighter tail | G2, G3 | not in record | not in record | 60 cm speed walks already at 0.85–1.00 of A1; for a lighter tail see PA11 below | Not studied |
| Supply: the pack's 21–25.2 V swing; an external 24 V input | G3 (Stage 0 knee margin only) | none for the pack | input about $0.2–0.4k, plus a boom $0.3–1.5k | The walk works at 21–25.2 V; the animal pace falls at all | Input adopted (P21) |
| Gait-tuned animal-pace reference (F25) | G3 | 0 | 0 | Knee 0.93–0.95 → 0.69–0.70 of rated; peak acceleration about 3 m/s², twice the spec; cadence about 3.1 steps/s; flat hard floors only; look unassessed | Information only (P24) |
| A new nominal reference walk | G7, G8 | 0 | 0 | Not designed | Finding (decisions 9, 10) |
| A gentler fast turn, or more thigh-twist range | G5 | not in record | not in record | Packaging unassessed; maybe only a generator change | Not studied |
| PA11 covering walls × 0.75 | G8; the tail's load (G2, G3) | −0.31 kg | +$20 / +$40 | Walls keep about 44 % of today's bending stiffness (ribs every 50–80 mm); not re-run on the picks (on `60-M-T-10`: neck 0.76 → 0.70, screened tail peak 0.52 → 0.45; at 75 cm 0.89 → 0.77) | Adopted (P21) |
| The 75 cm body | G5; G4 largely (turns meet every limit to μ 0.26 with R2 and 0.325 with R3/42, brisk to 0.29 and 0.34); G10 standing only, not a Stage 1 stand condition (the round-two screen on medium pile; R4 standing still on medium and high; every walk there falls); not G2–G3 | +3.95 to +4.04 kg | +$3.7k / +$5.0–5.1k | Speed falls become limit failures (tail servo; duty at the animal pace); still fails carpet and walking shoves, with R3 too; runtime −10.6 to −10.8 min | A Stage 1 body |

*Evidence:* §2–§4 records; `gap_options.json` (extraction and both checks); `round2/refine/refine.json`;
`ROUND2_REVIEW.md` §7.3–§10; `round2/refine/MATERIALS.md` §2 and §5–§6; DECISION_LOG.md A1, E6, F13, F16, F21, F23, F25,
P3, P4, P14 and P19–P26.

## 9. What this does not show

- **Hardware.** Everything is simulation: no real servo behaviour beyond the holding gains and the torque–speed
  envelope, no thermal limits beyond A1's fraction, no backlash, no fall damage.
- **Real servo delay and stiffness in the verdicts,** which assume a zero-delay loop and the model's stiffness.
- **The tail servo's real limit:** the 75 cm speed verdicts rest on A1.
- **Sharp ledger edges.** Each is a 3-episode reading on one fitted controller; several searches are not monotonic.
- **Carpet (A16).** The carpet's friction, stiffness and damping and the pile damping are unsourced, and floor friction
  is one number. The treads' grip is an estimate.
- **Real self-collision.** Only the foot pads touch the floor; leg-to-leg clearance is checked geometrically.
- **The body, where a result is controller-limited,** and the untested pad edge behind those labels.
- **Other controllers.** One controller family has been shown; another could do better or worse on the same bodies.
- **Learning with a fuller budget.** R3 trained 2.5M of 5M planned steps and is judged on one body; R4, one seed, is
  judged at 1.25M steps (it trained to 1.51M at 60 cm and 1.25M at 75 cm).
- **A gate rule set after a result:** decision 39 (N2), which made R3, not R2, the 75 cm body's judged controller (two
  fewer shown verdicts).
- **Autonomy and looks as people judge them:** G12, and viewer-rated naturalness.

## Engineering notes

**N1. Labels: how they are made, and where they conflict.**
- Decision 20's paired sign test needs at least 5 falls for p < 0.05. With fewer, nothing is lifted and the condition
  reads controller-limited (`60-N-T-10` brisk, fitted controller 42: 3 falls).
- A verdict reads body-limited only if every failing fitted controller does; the single-controller labels are in
  `STAGE1_TABLES.md`. For example, on `60-N-H-10` fitted controller 42 alone reads slip on the brisk walk (lifting
  slip lets 14 of 20 survive), and 43 alone on the animal pace (5 of 20; sign test 15 better, 5 worse, p 0.021).
  Walking shoves at 60 cm read slip on 2 records and drive torque on 1 of 16.
- "Body-limited" means that lifting helps, not that it fixes: on the 75 cm thin-carpet walk lifting slip delays 16 of
  20 falls by a fraction of a stride and saves none.
- The pad-edge candidate is the tracker's own: its ankle cap, or its centre-of-pressure clip with an ankle motor
  saturated, binding at the pad (`control.py`, `cop_edge`; decision 15). It was a candidate in the 0.5 s before every
  R1–R3 fall (all 984 R2 falls, in 55 records, and all 278 R3 falls, in 17) and before no R0 (665 falls) or R4 (94)
  fall, which run without that cap, so it partly reflects the controller's cap, not only the foot.
- The ledger's 3-seed lifts and the panels' 20-seed attribution disagree twice. On `60-N-T-10`'s brisk walk the
  ledger's lift (drive torque or slip saves its one falling seed) reads body-limited, while the panels had too few
  falls to attribute. On `60-N-H-10`'s animal pace (fitted controller 43) the panels read slip while the ledger's
  3 seeds, which reach the Froude reference, read controller-limited. The verdicts stay the panels'.
- R4's "body-limited: slip_median, slip_worst" labels (10 per body) are an artefact. R4 never lifts a foot, so the slip
  terms have no stance to measure and count as failed, and slip is a hardware term in `panels.SUCCESS_LABEL_RULE`. They
  say nothing about the body.

**N2. Decision 39, a gate rule set after a result.** The monitor holds carpet and pile conditions (decision 27), on
which R2 and R3 fall in every episode, so G6b's mean-survival comparison turns on fall-time differences below the
measurement's resolution: falls are read at the end of 0.02 s env steps. `75-M-H-10` seed 43's gap after the fix (17.997
vs 18.000 s) came from carpet and low-pile falls 0.02–0.08 s earlier than R2's in four of six episodes (decision 39 says
0.01–0.02 s; one fell 0.08 s later), with no hard-floor falls. Decision 39 (23:15 UTC) then judged G6b within one env
step for every run; the plan says it was decided after seeing that result. It changes two outcomes, both on `75-M-H-10`
(seed 42 at 1M steps, 17.988 s, whose fix applied anyway; seed 43 after the fix), and none on a 60 cm body. As recorded,
2 of the 5 checks after the fix passed; under decision 39, 3 did. Seed 43 had been stopped at 22:49 UTC and resumed at
23:15; without decision 39 its body would have been judged with R2 (decisions 32 and 34), which is shown on two more
conditions than R3 (§2). Both readings are kept in run.json (`g6.*.amended`), except on seed 42's run, which was
training when the amendments were written (23:15:41 UTC) and carries no `amended` field; its 1M reading passes under
decision 39 by the plan's text.

**N3. R3's training and G6's rules.**
- G6c (decision 27) needs, on some condition, fewer falls than R2, or as many and a higher mean reward per step on every
  paired episode seed, with a mean gain of at least 0.01 and R2's own spread. So `75-M-H-10` seed 43's low-pile win
  (gains 0.012 / 0.008 / 0.013) is a reward gain while falling in all 3 episodes, at 4.40 / 4.50 / 4.40 s against R2's
  4.40 / 4.54 / 4.42 s.
- Brisk is among R3's wins in 7 of the 13 checks (5 of the 11 G6b checks, and both G6c checks). R3 lost on brisk on both
  60 cm seed-42 runs, on `60-N-T-10` seed 43 at 1M steps and on `75-M-H-10` seed 43 at 1M steps (3 falls where R2 had
  none). Its gains reached about 0.03 reward per step on the hard-floor walks and turns, and +0.013 to +0.030 on brisk.
- Exploration noise is σ 0.02 of the residual's range: the largest σ at which the untuned tuning-start gains and the
  tuned gains together kept at least 18 of 20 `75-M-H-10` nominal-walk episodes up. The start gains set it: at σ 0.03
  they kept 2 of 10, while the tuned gains R3 trains on kept 10 of 10 (9 of 10 at 0.04; at 0.025 the two kept 17 of 20).
  The learning rate followed, 1e-5 to 3e-6 (decision 24).
- At 1M steps on `60-N-T-10` seed 42 the joint residual was 0.012 of its range RMS (about 2.4 mrad) and the clock
  −0.2 %: about 1 % of its authority (±0.2 rad on the legs, ±0.1 rad on the neck and tail, ±25 % pace).
- Over the 20-episode panels (§4) the G6c wins did not become limits met. Seed 43's right walking turn, one of its
  wins, runs at 0.347–0.351 rad/s (R2: 0.364–0.367), under the 0.35 target in 14 of 20; its brisk walk, another, misses
  speed (0.33–0.34 m/s against 0.351) and duty in all 20; its low-pile walk, another, falls in all 20. Its animal pace
  misses speed (0.39–0.44 m/s against 0.496), duty, no-flight, toe clearance (16.5–19.5 mm against 20.1) and the ankles'
  rated speed (1.02–1.23×) in all 20, and falls twice at μ 0.54 (reported). Seed 42's wins match R2 there. On low pile
  it stays up in 14 of 20 walks, though slow (0.239–0.241 m/s against 0.243), slipping (median 5.9–6.9 mm against 5)
  and with its toes 21–25 mm above the pile against 30.1.
- R3's walking shoves: 2 · 1, 0 · 0, 8 · 6 and 7 · 6 of 20 stay up forward, back, left and right (R2: 1 · 1, 0 · 0,
  6 · 5, 8 · 4); against fitted controller 43, whose gains both runs use, 17 and 13 of 80 against 10 (Fisher p 0.21 and
  0.65). Its thin-carpet falls come at 5.0–5.7 s (R2: 4.7–5.2 s). Head nod is 0.053–0.054 rad (R2: 0.051); mean sway
  at nominal / brisk / animal pace is 0.164 / 0.118 / 0.112 H on seed 42 and 0.177 / 0.145 / 0.143 on seed 43 (largest
  0.193), against R2's 0.168–0.170 / 0.123–0.124 / 0.115–0.127.

**N4. Reading the ledger.**
- "Meets limits" is decision 33's "works": no episode falls, and none misses a limit its seed met at the nominal value;
  at the nominal value itself, works means stays up. Read outright (all 3 meet every limit), every `60-N-T-10` search on
  the nominal walk would end "fails at nominal", because seed 3042 misses the peak foot force there (decision 33).
- That peak foot force, a controller term and noisy, also sets `60-N-T-10`'s limit edges for grip (0.71), the right turn
  (0.42), drive torque (0.92) and delay (4 ms), and `60-N-H-10`'s right turn (0.57) and delay (4 ms; 2 of 3 episodes
  miss it at 6 ms). The same seeds meet every limit at μ 0.54 in the panels (seed 3042 at 1.59 body weights), while the
  grip search fails at μ 0.70 and passes at 0.706.
- Not monotonic: on `60-N-H-10`'s brisk walk 1 of 3 falls at μ 0.50, 0.525, 0.5625 and 0.569, but none at 0.531–0.55
  or from 0.575 up; on `60-N-T-10`'s left turn 1 of 3 falls at μ 0.3125, though all stay up from 0.125 to 0.30; on
  its right turn all 3 succeed at μ 0.35 but not at 0.40–0.4125.
- The 75 cm drive-speed search scales both the torque–speed envelope and the rated speeds. Its 3 seeds already miss
  drive speed (ankles 1.13–1.21× rated), the tail limit and duty at full speed, so "meets limits" counts only new
  misses (the speed target and no-flight at 0.49×). Read 0.29, where it stays up, as the margin.
- R3/42's seeds keep the ankles within their rated speed at full speed (the knee peaks at 0.86 of rated), so its
  drive-speed search reads the knee: it first adds a miss at 0.859, where the knee exceeds its scaled rating. It stays
  up to 0.26, body-limited by drive torque, the torque–speed limit and slip.
- R3/42's other edges. Grip: one peak foot force on the walk (the miss at μ 0.275), the turn rate on the left turn, its
  worse side (0.319), foot slip on the right turn (0.2125) and the speed target at brisk (0.3375); its brisk falls at
  μ 0.10 are slip-limited, as R2's. Drive torque: the tail's A1 limit (the miss at 0.854); the falls at 0.355 are
  slip-limited. Shoves: drive torque and the torque–speed limit were active before one of its two forward standing
  falls at 0.313 m/s, and lifting them saved none; the left walking shove is body-limited by drive torque (falls at
  0.092 m/s; lifting it lets all 3 stay up) and the right by slip (0.129), where every R2/43 walking edge is
  controller-limited. Stiffness and delay: toe clearance goes first, as on R2.
- The stiffness sweep scales the plant's servo stiffness while the controller keeps its nominal gains, so the
  feedforward torque delivered falls by the same factor: 0.55–0.58 is the margin for a stiffness error the controller
  does not know about, not the stiffness a servo needs when its gains are measured and given to the controller.
- Trunk mass is read both ways (decision 33): × 1.1 (+4.1–4.4 % of the robot) and the factor that adds 10 % of the
  robot (× 1.227–1.241). All work, up to the search's top, × 1.3.
- Voltage: at 60 cm 25.2 V is the nominal plant (the drives get no speed credit above their 24 V rating), so 22.2 and
  21 V can only lower drive speed. `75-M-H-10`'s 48 V X6-60 hip pitch gains speed at 25.2 V (Stage 0 Fr 0.05 ratio 0.79
  against 0.83 at 24 V), though its seeds' results matched nominal exactly; its animal pace's ankle overspeed grows from
  1.21 to 1.31 and 1.35× as the voltage falls. `60-N-T-10`'s nominal walk "works" at all three with 2 of 3 meeting every
  limit, as at nominal.
- The Stage 0 column (decision 33): grip is walk / turn / brisk per body; drive speed is the knee's need without the
  screens' 15 % margin (0.93 / 0.95 / 0.93 with it); drive torque is the nominal walk's RMS (0.76 / 0.76 / 0.77 with
  the one-foot hold). At 22.2 V the knee's rated speed (8.04 rad/s) is under the animal pace's need with the margin
  (8.08 / 8.25 / 8.12 rad/s; plan §5's 8.08–8.12 omits `60-N-H-10`'s 8.25).
- Stop: not run, a deviation from plan §5 (decision 33); it needs stop references that decelerate at other rates.

**N5. The 60 cm animal-pace walk.** The condition walks the start, a nominal stride, the change to brisk, a brisk
stride and a 4.5 s ramp before the Froude reference begins (reference time 16.7–16.8 s). Of the 79 falls at μ 0.9, 30
come before it, 7 in the brisk stride and 23 in the ramp: 1 of 19 on `60-N-T-10`'s fitted controller 42, all 20 on its
43, 9 of 20 on `60-N-H-10`'s 42 and none on its 43. `60-N-T-10`'s 43 falls at 11.2–13.3 s, the moments of its brisk
falls, at both grips: its animal-pace failure is its brisk failure. The torque–speed limit was a candidate before 1
fall; lifting drive torque or that limit changes no survival.

**N6. The tail servo, A1 and servo loads.** The STS3215's planning rating, 0.862 N·m, is derived from its 2.94 N·m stall
by the PR #662 servo rule, so its A1 limit is 0.491 N·m, about half the spec sheet's 10 kg·cm (0.98 N·m at 12 V, with no
thermal definition). The worst R2 case, 1.31×, is about 0.64 N·m RMS, 0.66 of the spec-sheet rating; under R3 (run 43)
the tail reaches 1.36× at the animal pace, about 0.67 N·m, 0.68 of it. Its RMS torque under R2 over the judged window,
against that limit:

| Walk | Fitted controller 42 | Fitted controller 43 |
|---|---|---|
| Brisk, μ 0.9 / 0.54 | 1.04–1.05 / 1.05–1.06 | 1.05 / 1.06–1.07 |
| Animal pace, μ 0.9 / 0.54 | 1.17–1.19 / 1.22–1.23 | 1.19–1.31 / 1.23–1.27 |

At the animal pace each foot's duty was 0.543–0.554 under fitted controller 42 and 0.512–0.539 under 43, at least one
foot under 0.55 in every episode. Under 43 the ankles exceeded their rated speed in 14 of 20 at μ 0.9 (1.13–1.31×) and
20 of 20 at μ 0.54 (up to 1.32×); the speed target (0.496 m/s) was missed in 11 (down to 0.490) and 20 (0.479–0.491);
toe clearance failed in 14 (down to 13.5 mm) and 20; brief flight came in 4 and 11; and at μ 0.54 the median slip
exceeded 5 mm in 19 of 20, the "slip limit" of the tables. Closed-loop loads run above Stage 0's: the 60 cm neck at
0.83–0.84 of its limit in the nominal walk (Stage 0: 0.76); the 60 cm tail at 0.85–0.91 at brisk and 1.00 (0.998) in the
one 60 cm animal-pace episode at μ 0.9 that stayed up (`60-N-T-10`, 42), 0.93 for the two `60-N-H-10` survivors at
μ 0.54; the 75 cm tail at 0.87–0.88 even at the nominal pace (Stage 0: 0.40 sustained, 0.79 peak). Under R3 the 75 cm
tail runs at 0.85–0.89 of its limit at the nominal pace, 1.03–1.07 at brisk and 1.16–1.36 at the animal pace.

**N7. How the budget was counted.** PPO sessions run from `started_at` to `ended_at` where recorded; otherwise to the
next session's start (the sessions cut by the fix at 17:48–17:50 UTC), to 19:39:34 UTC (the last record written before
the outage), to the run's `stopped.at`, or, on `75-M-H-10`, to seed 42's last checkpoint (23:36 UTC) and seed 43's final
write (00:12 UTC). A session's CPU is its own `throughput` record where there is one (0.66–0.76 CPU-seconds per wall
second after the restart); otherwise its length × the job's rate in its first rollouts (0.39–0.40, measured at load 10
before the panels began, so an upper estimate), or the body's recorded post-restart rate (0.66 at 60 cm, 0.76 at 75 cm)
for the four post-restart sessions without their own record. `STAGE1_TABLES.md`'s PPO rows (10.33 h elapsed, 47.6–48.9
job-hours and 19.0–19.9 core-hours per body) count every session without a recorded end as running until run.json's last
write at 00:12 UTC, R4's single session (cut at the outage) and the sessions cut by the fix included. That is why their
total, 92.17, exceeds what four cores give over the time. Counted as above, PPO's elapsed wall clock was 6.8 / 6.8 /
8.6 h (19.4 / 19.4 / 21.9 job-hours: the runs shared the cores). The panels' figure sums episode wall seconds, one
process each: an upper bound on CPU. The machine ran only one or two jobs after 22:44, before 00:12 and from 00:34
(about 1.4 h in all). The last three items, all on the 75 cm body, took 1.50 core-hours (R3's panels and their
attribution, 00:12–00:40 UTC), 0.71 (R4's panels, 00:40–01:03) and 0.44 (R3/42's ledger, 01:03–01:13).

**N8. The three stops and the run records.**
- Work stopped at about 19:39 UTC: the run directory's last write is at 19:39:34, and the session's minute-by-minute
  memory log (scratchpad `stage1/build/watch/mem.log`, not in the repository) ends at 19:38:54 with 3.1 GB free at load
  15. The memory logs resume at 20:40, stop again from 20:51:48 to 21:23:35 with 8.8 GB free, and training resumed at
  21:23. No record gives either stop's cause. Every running job stopped. R3 resumed from its last checkpoints
  (1,253,376–1,507,328 steps), and the evaluations from their finished records; R4 was not resumed (decision 36).
  Training since each run's last checkpoint was redone: about 50 min on the 75 cm runs, up to 57 min in all (`60-N-H-10`
  seed 43).
- A third stop came after the last evaluation record (01:13:34 UTC): the memory log (`mem3.log`, same folder) ends at
  01:28:25 with 15.4 GB free, and the machine booted again at 01:28:54. No record gives its cause either. Nothing was
  running then, and nothing was lost.
- Decision 36's text records one outage, at about 20:37, and suspects memory; memory was not short before the second or
  the third stop. Its memory guard stays.
- R4's run.json still reads "running" (its single session, from 13:52 UTC, was cut and never closed), as do both 75 cm
  R3 runs; their judged checkpoints, 1,253,376 and 2,506,752, are final. The 60 cm seed-43 runs read
  `stopped_r2_judged` (decision 38) and the seed-42 runs `stopped_g6b`.
- Checkpoints are saved at the first rollout start at or past each 250,000 steps, about every 30 rollouts of 8,192 steps
  (253,952; 507,904; …; 2,506,752), and at the G6b-after-fix check (2,015,232). They and the per-record resume of the
  evaluations limited the loss.

**N9. The gap map's sources and corrections.** The options' figures come from the round-two record as corrected by two
checkers; each correction used was confirmed at its source (`gap_options.json`).
- A one-foot stop is set by how far the pad reaches in front, not behind: 43.5 mm on `60-N-T-10` and 43.2 mm on
  `60-N-H-10`, against the 52.2 and 51.8 mm a 1.5 m/s² stop needs, so both cap it at 1.25 m/s². Stage 1's
  start-and-stop test allows both feet down and passes 20/20, which does not show a 1.5 m/s² stop.
- With bigger feet the base-width 60 cm bodies stand servo-held on medium and high pile in the round-two screen, though
  by the rigid-stance criterion they would tip on high pile (critical compression 13.3–13.7 mm against its 14.1 mm). The
  narrow picks, with round-two feet, tip.
- Bigger feet at 60 cm also: grip need −8 % (Stage 0's quasi-static need; slip binds the 60 cm brisk walk and several
  Stage 1 fall records, and whether bigger feet help there is untested); one-foot stop 1.25 → 1.46 m/s²; the upper ankle
  drive 0.76 → 0.87 of sustained, governing the target turns; the Tapered knee at the animal pace 0.95 → 0.98; runtime
  −1.0 min. Tested on the base-width carpet-ready bodies only; the ankle drive is discontinued (§6, item 11).
- Faster knee: hip pitch speed (0.90) governs next; tested on base-width bodies only; the RobStride 06 winding
  resistance is unpublished. Knee transmission: the X4-36-L knee has torque to spare (0.49–0.51 sustained); hip pitch
  (0.89–0.90 at the animal pace) binds next.
- Supply: a higher bus would need the AK45-36 hip roll and thigh twist replaced (15–28 V only). The input path is an
  LTC4417-class power path, a regeneration clamp at 27 V or below, a Category 1 E-stop and the connector; the 0.678 kg
  dummy puck replaces the pack at equal mass on battery-free runs. 25.2 V is the 60 cm nominal plant: the X4-36-L gets
  no speed credit above 24 V.
- Gait-tuned reference: the hip pitch governs at 0.80 / 0.79; swing 0.08 H suits flat hard floors only; duty sits at
  0.55 and crouch at its bound; cadence rises from 2.75 to about 3.1 steps/s; the peak centre-of-mass acceleration is
  about 3.0 m/s² (3.21–3.22 in the current Fr 0.05 reference), twice the spec's 1.5.
- Fast turn: decision 14 keeps the reference's own range margin less 0.02 rad; the failures sit at hip-yaw margins of
  about 0.03–0.05 rad, and the turn references leave 0.046–0.095 rad. The clearance screens find no interference
  inside ±0.30 rad, and the plant falls back to that limit. (F16 concerns how narrow the stance can be, not joint
  range.)
- The 75 cm body: fall energy 63 J against 36 J; $18.5k / $25.7k in all, so the nominal cost is within P4's $10–20k and
  the high estimate over it. The 60 cm nominal estimate stays $14.8k; its high side, $20.5–20.6k, is over P4's range
  too.
- *Evidence:* `round2/refine/refine.json` (rows `feet_60`, `carpet_60`, `knee_60`, `narrow_carpet_60`, `feet_75`);
  `round2/refine/feet_60/review/screening/*_carpet.json`; `round2/refine/narrow_carpet_60/review/screening/*_envelope.json`
  (`stopping`) and `*_clearance.json`; `designlab/build.py`; `designlab/stage1/plant.py` (`usable_ranges`).

**N10. Provenance of each controller.**

| Body | R2 fits, 42; 43 | Body-level fit (R1, R3, ledger) | R3 | R4 |
|---|---|---|---|---|
| `60-N-T-10` | seed42_gen20; seed43_gen40 | seed42_gen20 | both runs stopped at 2,015,232; judged with R2 | 1,253,376 |
| `60-N-H-10` | seed42_gen60; seed43_gen60 | seed43_gen60 | stopped at 2,015,232 (42) and 1,753,088 (43); judged with R2 | 1,253,376 |
| `75-M-H-10` | seed42_gen60; seed43_gen40 | seed43_gen40 | judged at 2,506,752 (both; final) | 1,253,376 |

G5, fitted controller 42 · 43, of 20 validation episodes meeting every limit (walk / walking turn / turn in place; all
stayed up): `60-N-T-10` 18/17/20 · 20/19/20; `60-N-H-10` 18/20/19 · 19/18/19; `75-M-H-10` 20/20/20 · 20/20/20. The
fits' step-placement gain is 0 on all four 60 cm fits and 0.095 / 0.059 on the 75 cm ones; the capture margin is 0 on
all.

**N11. Budget and judging decisions.**

| Decision | Change |
|---|---|
| 21 | The tracker ran 2.4× slower than planned and a PPO environment at about 200–235 env steps/s, against G6a's 600. The cap became 24 h of wall clock in all, equal per body. The fit ran as planned. |
| 23 | A refit with carpet in the fit mix and G5 (about 6.5 h on four cores) was dropped. R3 and R4 train on hard, hard, robot_floor_carpet and low pile; the carpet walk is reported, not gated, for R0–R2. |
| 26 | G6a's rate failed (74–83 env steps/s per R3 job with nine jobs on four cores), so every R3 and R4 run was cut from 5M to 4M steps. |
| 29 | All nine runs started together at 13:52 UTC, at equal priority (the records' sessions start at 13:52:41–44; the decision's text says 13:51). After the 15:00 review found the work left exceeded the time left, R4 ran at a lower CPU priority (nice 4). |
| 32 | G6b failed on five of six R3 runs; the one fix weighted brisk and the animal pace ×2. |
| 33 | The ledger runs one fitted controller and 3 episodes per value; the stop margin is not run. |
| 35, 36 | Training stops by 01:00 UTC; after the outage R4 is judged at 1,253,376 and R3 resumes to its G6c record at 2,506,752; at most seven evaluation processes, under a memory guard. |
| 37 | Long-pile work runs last and may be cut first; robot_floor_carpet at its own friction shows grip is not why R2 falls. |
| 38 | Both 60 cm bodies' R3 is judged with R2; their seed-43 runs stopped (2,015,232 and 1,753,088) to free the cores. |
| 39 | G6b judged within one env step (0.02 s), decided after `75-M-H-10` seed 43 missed by 0.003 s; no 60 cm outcome changes. |

*Evidence:* `STAGE1_PLAN.md` decisions 21–39; `<b>/tune.json`; `<b>/ppo/*/run.json` and `<b>/policy_only/42/run.json`
(`sessions`, `g6`, `throughput`, `checkpoints`); `designlab/stage1/tune.py` (module docstring: latency and trunk mass
not modelled); `designlab/stage1/train.py` (`g6b_pass`, `R2_JUDGED_RULE`); `ppo_action_noise.json`
(`results.start`, `results.tuned`, `choice`); `ppo_lr_probe.json`.
