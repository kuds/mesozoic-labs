# Buy gate for the 75 cm build: a draft for the owner's approval

16 Nov 2026. **A draft, not a rule.** Asked for by the owner (`DECISION_LOG.md` P39): stronger evidence that the robot
will walk robustly before about $20,000 is spent. Nothing here is adopted until the owner approves it, and no
purchase is decided by it. Every number marked *proposed* is the lead's proposal and is the owner's to change.

Paths are relative to `hardware/trex_form_study/`.

## The idea

Money is spent in stages. Before each purchase, a set of results written down in advance must hold. If they do not
hold, the next purchase is not made, and what was learned cost only the stage before.

| Stage | What is bought | Cost | Source of the cost |
|---|---|---|---|
| 1. Harsher simulation and a trained policy | nothing | no money (the owner's Colab hours, the lead's usage) | |
| 2. Bench: one leg drive of each type, the tail servo, a microcontroller | bench parts | about $1,260 to $2,000 | `stage2/jetson75/BENCH_SHEET.md`: run 1 is $1,998 of catalogue parts, of which $736 is two USB-CAN adapters needed only to test the Jetson-direct wiring |
| 3. Legs-only prototype with ballast for head and tail | the 12 leg drives, leg and pelvis structure | $3,546 for the drives; the structure is not yet priced | `BENCH_SHEET.md` (the drives are needed whatever the compute) |
| 4. The full build | the rest | about $20,000 in all (the owner's figure) | the itemised prices are in `stage2/jetson75/BUDGET_75.md` and on the published price page |

## Gate 1: before any hardware is bought (simulation only)

What changes in the simulation first (the harsher model, to be built and reviewed):

- **A realistic servo model**, drawn at random per run: joint friction and gear play, torque and speed limits, voltage
  sag. Today's model has ideal servos and one unsourced friction value (`DECISION_LOG.md` A18).
- **Delay drawn at random** from 0 to 24 ms in training, as `STAGE2_PLAN.md` already plans.

What must then hold, for a policy trained twice (two training seeds), judged by the bar's registered pass rule (20 of
20 runs stay up and at least 17 meet every limit; `STAGE2_75_PLAN.md` H1) on the 75 cm body with the chosen look's
walk (*all numbers proposed*):

1. **Walking.** The walk at 0.30 and at 0.40 m/s passes on the grippy and on the slippery floor, at every delay from
   0 to 10 ms in 2 ms steps, with the servo model at its nominal values.
2. **Robust to the servo.** The same walks stay up in 20 of 20 at the corners of the servo model's range (friction
   at half and at one and a half times nominal; gear play at 0 and at 1°; trunk mass at 0.9 and 1.1), at 4 ms and
   at 10 ms.
3. **Pushes, turn and stop.** The four pushes, the two-step stop and a 90° turn pass at 4 ms. At 10 ms they are
   reported.
4. **The tail servo.** Its continuous load is under the planning limit at 0.40 m/s.
5. **Both training seeds agree.** A condition counts only if both policies pass it.

If gate 1 fails after two honest attempts at the training setup, the finding goes to the owner with what failed, and
no bench parts are bought on this design.

## Gate 2: before the legs-only prototype (bench)

What the bench measures: the real delay from command to torque (with all 12 drives' traffic on the buses), each
drive type's friction and gear play, and the tail servo's heating at the walk's load.

What must then hold (*proposed*):

1. **Delay.** The 99th percentile of the measured delay is at least 2 ms below the largest delay at which gate 1's
   policy passes the walk.
2. **The model matches.** With the measured friction, gear play and delay put into the simulation, a policy
   retrained on them passes gate 1's conditions 1 to 3 again.
3. **The tail servo** holds the walk's load for 20 minutes without passing its temperature limit.

## Gate 3: before the full build (legs-only prototype)

What must hold (*proposed*): on a flat hard floor, with ballast standing in for the head and tail, it walks at
0.30 m/s for 5 minutes without a fall in 5 of 5 trials, stops within two steps, and takes a 22 N push from each side
in 4 of 5 trials; and the joints follow the simulation's motion closely enough that the same policy file is used in
both.

## When the form factor is reconsidered (added 16 Nov 2026, after the owner asked)

The owner asked whether to continue on this path before considering changes to size, motors or goals. The lead's
recommendation: continue through gate 1, and reconsider the form factor there, on what gate 1 shows. The reason:
today's falls at 10 ms happen under a stand-in controller that was fitted with no delay, and the runner's own
labels for them are hints (`STAGE2_75_REPORT.md` section 5). They cannot yet tell a limit of the body from a limit
of the controller. A policy trained with delay is the first test that can.

If gate 1 fails, how it fails names what to change. *Every row is the lead's proposal, not a finding.*

| What gate 1 shows | What is reconsidered | What is known today (simulation, stand-in controller) |
|---|---|---|
| The walk passes at 0.30 m/s and fails at 0.40 m/s | The goal: accept 0.30 m/s for this body, or change the body | With no delay today's walk meets every limit at 0.24 and 0.30 m/s; at 0.40 m/s only with the tail servo's limit set aside |
| The tail servo's load stays over its planning limit with the counter-phased tail | The tail: a stronger servo, or a shorter or lighter tail | Today's walk: 1.02 to 1.11 of the limit at 0.40 m/s. The pull request's counter-phased tail: 0.40 to 0.48 |
| Pushes fail on a drive's peak speed | The leg drives: a faster drive or a lower gear ratio | With no delay 11 to 15 of 20 runs meet every limit after a push; drive peak speed is one of the failing limits |
| The quick turn fails on drive speed, toe clearance or heading with the tail fixed | The turn that is asked for (a wider or slower turn), then hip range and drive speed | The quick turn stays up 20 of 20 with no delay and does not pass at any delay; with no delay its failing limits are the tail servo and duty in every run and, in some runs, a drive's peak speed, toe clearance and heading. A slower turn needs a wider circle: 2.61 m or 3.13 m as planned, against 2.22 m |
| Falls at the start from standing persist with the foot's edge loaded | The feet and ankles: a longer or wider pad, more ankle torque | The runner labels today's 10 ms falls "controller-limited" and names the pad edge as one possible cause it cannot test |
| Nothing passes at 10 ms and the walk passes at 4 to 6 ms | The compute and wiring (gate 2's delay measurement decides), not the body | Today's walk stays up 20 of 20 at 0 to 6 ms |

Two more things belong to this question:

- **The 60 cm body goes through gate 1 as well** (`60-N-T-10` stays in the plan). If it passes where the 75 cm
  body fails, size is the first change to consider. It costs no hardware.
- **A change of form factor restarts gate 1 for the changed body.** Results do not carry over between bodies.

## What this draft does not do

- It does not make a trained policy pass. Today nothing passes at 10 ms under the stand-in controller
  (`STAGE2_75_REPORT.md`), and two earlier attempts with delay in training or fitting still fell at 10 ms
  (`DECISION_LOG.md` F26, F27).
- It does not price the legs-only structure or the bench's small parts beyond the sheet.
- It does not change the agility bar or any registered rule. The gate's conditions use the bar's own pass rule.

## The lead's confidence, for scale (a judgement, not a result)

About 65 % today that the real robot ends up walking steadily at 0.35 to 0.40 m/s; about 75 % if gate 1 is passed;
about 80 to 85 % if gate 2 is passed; about 90 % if gate 3 is passed.

## What the owner is asked

1. Approve the staged idea, or change the stages.
2. Approve, loosen or tighten each gate's proposed numbers.
3. Say whether gate 1 should also ask for carpet.
4. Say whether 0.30 m/s can be the speed gate 1 must pass, with 0.40 m/s as a stretch that is reported.
