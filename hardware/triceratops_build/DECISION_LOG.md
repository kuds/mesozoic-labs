# Triceratops build: decision log

Started 18 Nov 2026. Same tags as the T. rex log (`hardware/trex_form_study/DECISION_LOG.md`), so that preferences,
constraints, findings and assumptions never blur:

- **P**: the owner's stated preference, requirement or decision. Each decision row records the options that were put.
- **E**: an engineering constraint (physics, catalog fact, interface).
- **F**: a finding from this build's models or checks, with where it was found. Simulation findings say so.
- **A**: a provisional assumption, to be replaced by a measurement, a quote or a decision.

Status: `open` (awaiting the owner), `adopted` (in use, revisable), `superseded`.

Nothing is adopted without the owner. Recommendations live in the plan files until the owner answers; the answer and
the options put are then recorded here.

## Carried in from the kickoff (P)

From `hardware/triceratops_kickoff/NEW_SESSION_PROMPT.md` and `CONSOLIDATION_PLAN.md` (18 Nov 2026).

| ID | Statement | Status |
|---|---|---|
| P1 | A four-legged Triceratops robot for museum demos, walking at 0.30 m/s (0.7 mph). The kickoff brief's size and mass (75 cm to the frill crown, 1.5 m long, 15 or 20 kg) are the starting point, but size and mass are open decisions D2 and D1. Simulation first (MuJoCo models, then a walking controller, then a one-leg bench plan). Hardware is bought only when the owner approves. | adopted (size and mass open: D1, D2) |
| P2 | Museum dates: first contact mid-December 2026 with simulation renders ("the Field Museum or the Natural History Museum"; Griffin MSI is a further candidate the owner did not name); live Triceratops demos in Q3 2027 (July to September). | adopted |
| P3 | The T. rex is paused and preserved (T. rex log P47-P51) and stays the flagship for end 2027 / early 2028. The outreach shows it as coming. This session does not change T. rex files or branches; shared hardware choices (X4 / X6 drive family, compute and CAN firmware, bench rig, safety stop, battery, neck servo) say when they affect it. | adopted |
| P4 | Legit hardware: catalog drives with datasheets, industrial safety parts, named sensors. Spend in the T. rex's class, $17k-$32k. | adopted |
| P5 | Propose, ask, record: nothing is adopted without the owner; decisions are asked with the question tool, at most four at a time, recommended option first. | adopted |
| P6 | The museum walk is 0.30 m/s. 0.40 m/s is not a walking target. **Superseded 18 Nov 2026 by D2 (60 cm): the museum walk target becomes about 0.25-0.27 m/s** (Froude-similar to 0.30 m/s at 75 cm; the option the owner chose was described as walking 0.25-0.27 m/s), to be fixed in the 60 cm simulation. | superseded |
| P7 | Owner, 18 Nov 2026, in chat: "30k is too much for me to afford"; the build should come in at $10k-$20k. Size and budget are to be settled first, with expressiveness (moving head and mouth) part of the budget. Supersedes the $17k-$32k spend class in P4 (the "legit hardware" part of P4 stands). | adopted |

## Engineering constraints carried in (E)

| ID | Statement | Source | Status |
|---|---|---|---|
| E1 | The X4-36 (24 V drive) is capped at its 83 rpm rated speed when run above 24 V (reseller text quoting MyActuator; to confirm with the maker). The 0.30 m/s walk needs 58-60 rpm at the pitch joint (0.70-0.72 of the cap); 0.40 m/s needs 75-77 rpm (0.91-0.93). | PARTS_PREMIUM.md F1 | adopted (to confirm on the bench) |
| E2 | The X6 and X8 are 48 V drives; at 6S (21 V) their no-load speeds fall to about 77 and 69 rpm. | PARTS_PREMIUM.md section 2.4 | adopted |
| E3 | Two classic CAN buses with six drives each load 67-81 % at 500 Hz and cannot carry 1 kHz; four buses of three load 67-81 % at 1 kHz. | T. rex `stage2/jetson75/LATENCY_AND_BUS.md`; survey arithmetic | adopted |

## Findings (F)

| ID | Statement | Where | Status |
|---|---|---|---|
| F1 | PR #669's `check_models.py` passes under mujoco 3.10.0 (exit 0): robot 15.00 kg, settled foot loads 40.8 / 40.5 / 33.0 / 32.8 N (55.2 % front). Simulation. | `pr669_review/check_models_out.txt` | adopted |
| F2 | In PR #669's model, with the support polygon taken between the ankle pins (point feet), a forefoot lift needs an 18.0-19.1 mm sideways body shift to reach the support triangle's edge (17.8 mm on paper; the rest comes from the model's centre of mass sitting 1.3 mm further forward and 0.5 mm left); a hind foot lifts with 15.9-16.8 mm of margin and no shift. The 80 mm-wide soles are rigid in roll, so counting the sole width the margin is larger; the crawl (task 4) will state which basis it uses. Simulation, static. | `pr669_review/stance_balance_out.txt`; independent check 18 Nov 2026 | adopted |
| F3 | PR #669's 55 % front share depends on its 1.84 kg reserve block 118 mm forward; without it about 53.5 %. | `pr669_review/REVIEW.md`, item 10 | adopted |
| F4 | Standing check, all four build models (T20 and T15, listed and closed) pass every pre-registered rule on both judged seeds, in the first run and again on the corrected models (second run). Second run: T20 listed 55.6 % front (parts list 55.2 %), T20 at 20.00 kg front elbow 7.9 N·m, hind knee 6.2 N·m, pitch 1.3 N·m; T15 listed 54.2 % (parts list 53.6 %), T15 at 15.00 kg elbow 5.7 N·m, knee 4.7 N·m (joint side). One flag: T15 closed 54.6 % against a listed-parts reference. Simulation. | `sim/STAND_REPORT.md`, `sim/results/stand_2026-11-18b/` | adopted |
| F5 | With the ankle a pitch hinge only, each sole is rigid in roll; in the standing model about 95 % of a forefoot's load sits on the sole's inner edge, and the roll drive held 0.5 N·m (first run) and 1.1 N·m (second run, stiffer gains) at 20 kg. Standing roll torque depends on where the centre of pressure sits across the sole (about 0.3-4.6 N·m at 20 kg); the parts list's 4.3 N·m stays the sizing case. Sole roll compliance is a design question for the foot. Simulation; force balance confirmed by the independent check. | `sim/STAND_REPORT.md` | adopted |
| F6 | The neck's gravity moment in the rebuilt T20 head is 0.52 N·m (horizontal lever 34.9 mm; T15 40.7 mm), against the 1.63 N·m the parts list sized the neck servo for (1.51 kg at 110 mm), so that sizing is conservative. Model estimate. | `sim/STAND_REPORT.md`; independent check 18 Nov 2026 | adopted |
| F7 | Scripted crawl (0.02 m/s, one foot at a time, planned margin 65 mm on the ankle-pin basis), judged seeds 1 and 2: with PR #669's foot (passive pitch ankle, sole rigid in roll) both builds FAIL C4 and C5: sideways body shifts tip the roll-rigid soles onto their edges (pins swing median 16, max 25 mm; the soles themselves slide a median 3-5 mm, worst 6-11 mm per stance, no less with the roll hinge), and the third support foot's load dips below 5 % for up to about 0.11 s after lift-off. With a passive roll hinge at each ankle both builds PASS every rule (pins 7-9 mm), but that pass fails under a stiffer friction model (impratio 10: C4 on 4 of 4 runs, C5 on 3 of 4), so it is not robust. Simulation. | `sim/CRAWL_REPORT.md`, `sim/results/crawl_2026-11-18/` (incl. `sensitivity_impratio10.txt`, `DEV_LOG.md`); independent check 18 Nov 2026 | adopted |
| F8 | PR #669's passive pitch ankle range (±22°) limits how far the body can move fore-aft over a planted foot: a development run passed it (+25.5°, ledger row 32), the sole tipped and the pin moved; in the judged runs the ankles came within 1° of the stop. Simulation. | `sim/CRAWL_REPORT.md`, `DEV_LOG.md` | adopted |

## Assumptions (A)

| ID | Statement | To be replaced by | Status |
|---|---|---|---|
| A1 | The parts list's mass reserve (T20 1.17 kg, T15 0.39 kg) first covers, on the chassis, the wireless-stop receiver (mass unpublished, taken as 0.30 kg) and 0.15 kg of small added parts; T15's reserve is all used there. T20's remaining 0.72 kg is pro-rata growth of shells, harness and frame structure (0.30 kg of it on the legs). Revised 18 Nov 2026 after the independent check; first version spread the whole reserve as growth. | Weigh-in | adopted (model) |
| A2 | Leg position-servo gains: the T. rex servo model's nominal for the same drives, kp 598.5 N·m/rad at the drive output, kv 24.48 (roll) and 11.51 N·m·s/rad (pitch, flexion), × ratio² at the joint; an assumption there too, not a maker figure. Revised 18 Nov 2026 after the first crawl runs, from PR #669's softer 150-330 N·m/rad. | Bench identification of the drives' loops; maker's KP/KD ranges (T. rex MAKER_QUESTIONS.md, question 2) | adopted (model) |
| A3 | The OAK-D Pro W camera sits in the chest, as in PR #669 and as the parts list's mass budget counts it; the parts list's section 3 puts it in the head behind the brow. If it moves to the head, the head gains about 0.12 kg with its cable. | Owner or CAD | open |
| A5 | The 1.3:1 pushrod is modelled as a constant ratio; the four-bar with the model's crank geometry stays within 10 % of 1.30 only between -29° and +24° of flexion (1.71 and 2.13 at the range ends; 119° of crank travel against the parts list's 108°). Runs report whether the 1.3:1 joints stayed in that band. | An angle-dependent ratio in the model; CAD | adopted (model) |
| A4 | Positions of the parts added in the rebuild (drives, pushrods, battery pair, power and safety boxes, head and tail servos) are this model's estimates; the parts list's placement script was not committed. | CAD | adopted (model) |

## Decisions for the owner

Round 1 (by 15 Dec 2026): D1 mass, D2 size, D3 compute, D4 bus voltage, D5 chassis. Round 2 (by 15 Jan 2027): D6
what the T. rex milestone shows, D7 a PLEM bench pilot, D8 the pull-request triage. Options and costs are in
`CONSOLIDATION_PLAN.md`, section 7. Each row is filled in when the owner answers.

18 Nov 2026: round 1 was put through the question tool (D1, D2, D4, D5 in one call, with the recommended option first and each option's cost); the call was declined, so no answer is recorded and the decisions stay open. They will be asked again. Until then the models cover both T20 and T15 on a 12S bus, so either answer can be taken without rework.

| ID | Decision | Options put | Owner's answer | Date | Status |
|---|---|---|---|---|---|
| D1 | Mass | 20 kg T20 / 15 kg T15 (first ask, declined); then, 18 Nov 2026, as part of the size options of `budget/SIZE_BUDGET_OPTIONS.md` | **About 12 kg** (follows from D2, option C; an estimate until the 60 cm parts list) | 18 Nov 2026, question tool | adopted |
| D2 | Size and build | 60 cm with the T. rex's drive set (10 x X4 + 2 x X6, 12S), recommended: $23.9-24.7k landed museum-ready / 60 cm budget, all X4 on 6S: $22.3-23.1k / 50 cm: $20.6-21.9k / stay at 75 cm: $28.6k | **60 cm, the T. rex's drive set (option C)** | 18 Nov 2026, question tool | adopted |
| D3 | Compute | (to be asked; the budget assumes a Jetson Orin Nano Super with a Teensy 4.1) | - | - | open |
| D4 | Bus voltage | 48 V class 12S / 24 V 6S | **12S (48 V)**, implied by option C (the 6S alternative was option B, not chosen); to confirm with the 60 cm parts list | 18 Nov 2026 | adopted (implied) |
| D5 | Chassis | (to be asked for the 60 cm robot; now also: a foot that stays flat in roll, F7. A passive roll hinge is the first candidate but not yet robust in simulation) | - | - | open |
| D6 | What the T. rex milestone shows | | - | - | open |
| D7 | PLEM bench pilot | | - | - | open |
| D8 | Pull-request triage | | - | - | open |
| D9 | Delay at which the 0.30 m/s walk's pass/fail verdict is judged until the foreleg bench measures it | 10 ms, as the T. rex (recommended; conservative, harder, replaced by the bench value) / 4 ms (PR #669's turn-stop-push delay; easier, may not survive the bench) / no verdict until the bench (no pass to show before spring 2027) | **10 ms** | 18 Nov 2026, question tool | adopted |
| D10 | How the walk's two controllers start | Model-based first in this container (CPU, about 1-2 days of session work, about 2 CPU-hours of runs), learned policy on Colab later after its own cost estimate (recommended) / both in parallel / model-based only for now / wait for round-1 decisions | **Model-based first, learned later** | 18 Nov 2026, question tool | adopted |
| D11 | Where simulation video clips go (2-6 MB each, over the 500 KB limit) | A never-merged data branch (recommended) / the owner's Google Drive (about 3 MB per file through the connector) / sent to the owner only | **A never-merged data branch** | 18 Nov 2026, question tool | adopted |
| D12 | How the budget ceiling is counted | Landed, all-in: parts + shipping, duty, tax, 7 % contingency (recommended) / parts + about 7 % (the T. rex way) / parts only | **Landed, all-in: $20k** | 18 Nov 2026, question tool | adopted |
| D13 | When to buy the handheld wireless safety stop (about $2.7k of parts with its receiver) | Before any public demo; wired stop and catch for bench and first walks (recommended) / from the start / leave it out | **Before any public demo** | 18 Nov 2026, question tool | adopted |
| D14 | Head-and-tail expression package | Full, about $980: nod, turn to look at visitors, jaw with sound, tail sway, blinking, tail lift, frill lights (recommended) / core, about $750 | **Full** | 18 Nov 2026, question tool | adopted |
