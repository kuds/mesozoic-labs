# Buy gate for the 75 cm build: a draft for the owner's approval

16 Nov 2026. Asked for by the owner (`DECISION_LOG.md` P39): stronger evidence that the robot will walk robustly
before about $20,000 is spent.

**Status, 16 Nov 2026 (`DECISION_LOG.md` P40).** The owner approved the four stages, gate 1's speed, gate 1's
numbers for delay, pushes, turn, stop, the two training runs and the two attempts, and that gate 1 is judged on
hard floors only. Those parts are marked **approved**. On the same day the owner approved gate 1's condition 2 with the servo
model's settings, its pass rule at an extreme, its repeat, and when the maker's answers are waited for
(`DECISION_LOG.md` P41). Everything marked *proposed* is still the lead's proposal and has not been put to the
owner: gate 1's condition 4, and the numbers of gates 2 and 3. No purchase is decided by this file.

Paths are relative to `hardware/trex_form_study/`.

## The idea

Money is spent in stages (**approved**, P40). Before each purchase, a set of results written down in advance must hold. If they do not
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
walk:

1. **Walking (approved, P40).** The walk at 0.30 m/s passes on the grippy and on the slippery floor, at every delay
   from 0 to 10 ms in 2 ms steps, with the servo model at its nominal values. The walk at 0.40 m/s is run the same
   way and reported as a stretch; it does not block a purchase.
2. **Robust to the servo (approved, P41).** With the servo model of
   `stage2/design/servo_model/SERVO_MODEL_DESIGN.md` (its section 4): the 0.30 m/s walk on the grippy floor, at 4
   and at 10 ms, both policies, seeds 3042 to 3061, under seven settings. Six are extremes, one factor at a time:
   friction low and high, gear play 0 and 1.0° on the leg drives, trunk mass 0.9 and 1.1. The seventh is a panel
   of 20 sets of motor values drawn at random from the model's table. That is 28 records and 560 runs.
   - **At an extreme all 20 runs must stay up; the other limits are recorded and shown, not judged.** This is an
     exception to the sentence above that every condition uses the bar's whole rule (P41).
   - The drawn panel must meet the bar's whole rule: 20 of 20 stay up and at least 17 meet every limit.
   - Each drawn-panel record that passes is run once more on the bar's second block of seeds (5042 to 5061, 20
     other drawn sets): 80 more runs. **The repeat must pass too** (P41).
   - Run and shown, stopping nothing yet: the heavy rotor (it rests on a catalogue number the maker must explain)
     and the weak battery (it rests on a wiring resistance nobody has measured). Also shown with no verdict: the
     slippery floor, the drives' gain limits, today's torque-speed line, the other hip-pitch part that may ship,
     and two combined extremes.
   - For scale, by the study's arithmetic with runs taken as independent: a pair of policies that truly falls
     once in 1,000 runs passes the 560 runs about 57 % of the time, and about 53 % with the repeat.
3. **Pushes, turn and stop (approved, P40).** The four pushes, the two-step stop and a 90° turn pass at 4 ms. At
   10 ms they are reported.
4. **The tail servo (*proposed*).** Its continuous load is under the planning limit at the speed gate 1 must pass.
5. **Both training seeds agree (approved, P40).** A condition counts only if both policies pass it.

If gate 1 fails after two honest attempts at the training setup (approved, P40), the finding goes to the owner with
what failed, and no bench parts are bought on this design.

**The maker's answers (approved, P41).** Five facts about the leg drives are open and the maker can give them for
free: which side of the gearbox the listed rotor inertia is on, the gain limits of the motion mode, the hip-pitch
drive at 24 V, which hip-pitch part ships, and which encoder closes the loop. The five questions are in the servo
note's appendix A. The owner sends them. Gate 1 is judged after the answers are in the model, or two weeks after
they were sent without an answer, whichever comes first. Building and training go on meanwhile.

**Floors (approved, P40).** Gate 1 is judged on the hard floors only. Carpet is run as a range and reported with no
verdict (P37). A measured carpet sample joins the bench stage, and carpet is judged on the legs-only prototype
(the numbers for that are *proposed* below).

## Gate 2: before the legs-only prototype (bench)

What the bench measures: the real delay from command to torque (with all 12 drives' traffic on the buses), each
drive type's friction and gear play, the tail servo's heating at the walk's load, and a carpet sample's grip and
sink under a foot pad (P40).

What must then hold (*proposed*):

1. **Delay.** The 99th percentile of the measured delay is at least 2 ms below the largest delay at which gate 1's
   policy passes the walk.
2. **The model matches.** With the measured friction, gear play and delay put into the simulation, a policy
   retrained on them passes gate 1's conditions 1 to 3 again.
3. **The tail servo** holds the walk's load for 20 minutes without passing its temperature limit.

## Gate 3: before the full build (legs-only prototype)

What must hold (*proposed*): on a flat hard floor and on the measured carpet, with ballast standing in for the head
and tail, it walks at 0.30 m/s for 5 minutes without a fall in 5 of 5 trials, stops within two steps, and takes a 22 N push from each side
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
| The walk passes at 0.30 m/s and fails at 0.40 m/s | Nothing blocks the purchase (P40). The owner chooses whether 0.30 m/s is the goal for this body or 0.40 m/s is pursued | With no delay today's walk meets every limit at 0.24 and 0.30 m/s; at 0.40 m/s only with the tail servo's limit set aside |
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

## What the owner decided on 16 Nov 2026 (`DECISION_LOG.md` P40)

1. The four stages: approved as drafted.
2. Gate 1's speed: 0.30 m/s must pass; 0.40 m/s is reported.
3. Gate 1's numbers (the walk at 0 to 10 ms; pushes, stop and turn at 4 ms and reported at 10 ms; two training
   runs agree; two attempts): approved as drafted.
4. Carpet: not in gate 1. Hard floors only.

## What the owner decided later on 16 Nov 2026 (`DECISION_LOG.md` P41)

1. Condition 2's settings: six extremes and the random panel can stop the purchase of the bench parts; the heavy
   rotor and the weak battery are shown and stop nothing yet.
2. At an extreme the robot must only stay up; the other limits are recorded.
3. A random-panel record that passes is repeated on other seeds, and the repeat must pass too.
4. The owner sends the five questions to the maker; gate 1 is judged after the answers or after two weeks.

Still to be put to the owner: gate 1's condition 4 (the tail servo, read with the servo model), and the numbers of
gates 2 and 3, with the six measurements the servo note proposes to add to the bench sheet.
