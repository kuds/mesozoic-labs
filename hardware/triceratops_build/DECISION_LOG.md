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
| P6 | The museum walk is 0.30 m/s. 0.40 m/s is not a walking target. | adopted |

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
| F4 | Standing check, all four build models (T20 and T15, listed and closed) pass every pre-registered rule on both judged seeds. T20 at 20.00 kg: 55.2 % front, front elbow 7.8 N·m, hind knee 6.2 N·m, pitch 1.2 N·m (joint side). T15 at 15.00 kg: 54.1 % front, elbow 5.6 N·m, knee 4.7 N·m. Simulation. | `sim/STAND_REPORT.md`, `sim/results/stand_2026-11-18/` | adopted |
| F5 | With the ankle a pitch hinge only, each sole is rigid in roll; in the standing model about 96 % of a forefoot's load sits on the sole's inner edge, and the roll drive holds only 0.5 N·m (20 kg). Standing roll torque depends on where the centre of pressure sits across the sole (about 0.3-4.6 N·m at 20 kg); the parts list's 4.3 N·m stays the sizing case. Sole roll compliance is a design question for the foot. Simulation. | `sim/STAND_REPORT.md` | adopted |
| F6 | The rebuilt T20 head's centre of mass is 49 mm from the neck pitch pivot (T15 48 mm), against the 110 mm the parts list assumed when sizing the neck servo, so that sizing is conservative. Model estimate. | `sim/generated/mass_ledger_T20_closed.json` (`neck`) | adopted |

## Assumptions (A)

| ID | Statement | To be replaced by | Status |
|---|---|---|---|
| A1 | The parts list's mass reserve (T20 1.17 kg, T15 0.39 kg) is modelled as pro-rata growth of shells, harness and frame structure ("closed" variant), not as a block. | Weigh-in | adopted (model) |
| A2 | Leg position-servo gains are PR #669's (roll 150, pitch 200, flexion 330 N·m/rad), scaled with mass for T20 (× 4/3). | Bench identification of the drives' loops (bench plan) | adopted (model) |
| A3 | The OAK-D Pro W camera sits in the chest, as in PR #669 and as the parts list's mass budget counts it; the parts list's section 3 puts it in the head behind the brow. If it moves to the head, the head gains about 0.12 kg with its cable. | Owner or CAD | open |
| A4 | Positions of the parts added in the rebuild (drives, pushrods, battery pair, power and safety boxes, head and tail servos) are this model's estimates; the parts list's placement script was not committed. | CAD | adopted (model) |

## Decisions for the owner

Round 1 (by 15 Dec 2026): D1 mass, D2 size, D3 compute, D4 bus voltage, D5 chassis. Round 2 (by 15 Jan 2027): D6
what the T. rex milestone shows, D7 a PLEM bench pilot, D8 the pull-request triage. Options and costs are in
`CONSOLIDATION_PLAN.md`, section 7. Each row is filled in when the owner answers.

18 Nov 2026: round 1 was put through the question tool (D1, D2, D4, D5 in one call, with the recommended option first and each option's cost); the call was declined, so no answer is recorded and the decisions stay open. They will be asked again. Until then the models cover both T20 and T15 on a 12S bus, so either answer can be taken without rework.

| ID | Decision | Options put | Owner's answer | Date | Status |
|---|---|---|---|---|---|
| D1 | Mass | | - | - | open |
| D2 | Size | | - | - | open |
| D3 | Compute | | - | - | open |
| D4 | Bus voltage | | - | - | open |
| D5 | Chassis | | - | - | open |
| D6 | What the T. rex milestone shows | | - | - | open |
| D7 | PLEM bench pilot | | - | - | open |
| D8 | Pull-request triage | | - | - | open |
