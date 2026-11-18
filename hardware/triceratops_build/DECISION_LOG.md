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
| P1 | A four-legged Triceratops robot for museum demos: 75 cm to the frill crown, 1.5 m long, 15 or 20 kg, walking at 0.30 m/s (0.7 mph). Simulation first (MuJoCo models, then a walking controller, then a one-leg bench plan). Hardware is bought only when the owner approves. | adopted |
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
| F2 | In PR #669's model, a forefoot lift needs an 18.1-19.1 mm sideways body shift to reach the support triangle's edge (17.8 mm on paper); a hind foot lifts with 15.9-16.8 mm of margin and no shift. Simulation, static. | `pr669_review/stance_balance_out.txt` | adopted |
| F3 | PR #669's 55 % front share depends on its 1.84 kg reserve block 118 mm forward; without it about 53.5 %. | `pr669_review/REVIEW.md`, item 10 | adopted |

## Decisions for the owner

Round 1 (by 15 Dec 2026): D1 mass, D2 size, D3 compute, D4 bus voltage, D5 chassis. Round 2 (by 15 Jan 2027): D6
what the T. rex milestone shows, D7 a PLEM bench pilot, D8 the pull-request triage. Options and costs are in
`CONSOLIDATION_PLAN.md`, section 7. Each row is filled in when the owner answers.

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
