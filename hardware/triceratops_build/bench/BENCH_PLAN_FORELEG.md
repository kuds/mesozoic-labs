# One-foreleg bench plan (task 8), with the pass and fail rules written first

18 Nov 2026. **A plan. Nothing is bought, built or measured; no purchase without the owner.** Every number below is a
target, a maker's figure or an estimate from the parts list (`hardware/triceratops_kickoff/PARTS_PREMIUM.md`) or from
this build's simulation (`../sim/`), labelled as such. The bench's readings will be the project's first hardware
measurements; they go in the results tables only, never beside the targets as if they were the same kind of number.

**Why one foreleg first.** The front leg carries the most load (55 % of the weight on the front feet), its elbow is
the riskiest axis, and the bench answers the questions the simulation cannot: real torque and speed at 12S, heat in
the show loop under a shell, the command-to-torque delay, transmission play and stiffness, regen and the stop chain.
The same drives (X4-36-L, X6-60) are the T. rex's, so this bench also serves the T. rex's delay row A17 and its drive
order in Q2 2027 (`CONSOLIDATION_PLAN.md`, section 5.3).

## The leg on the bench

| Axis | T20 (recommended; decision D1 open) | T15 alternative |
|---|---|---|
| Roll | MyActuator RMD-X6-P20-60-E | RMD-X4-P36-36-E-L |
| Upper-leg pitch | RMD-X4-P36-36-E-L | RMD-X4-P36-36-E-L |
| Elbow flexion | RMD-X8-P20-150-E through a 1:1 four-bar pushrod | RMD-X6-P20-60-E through a 1.3:1 four-bar pushrod |

Parts: the parts list's first purchase, the T20 one-foreleg bench set, **about $7.1k of parts, $8.9k landed** (drive
prices from MyActuator's own US store, read 7 Nov 2026; at the third-party US reseller prices found on 18 Nov 2026
(X6 $741.84 and X4 $620 at RobotShop, X8 $645 at aifitlab) the drives' $1,422 becomes about $2.0k). It holds: the foreleg's drives $1,422, its linkage
$120, a torque rig $700, a 60 V bench supply $400, a thermal camera $250, a PCAN-USB FD $379, two MCU carriers $300,
and the full stop chain $3,526 (wireless stop, Pilz relay, e-stops, contactor, fuse, receiver estimate). The torque
rig, supply, camera and PCAN serve the T. rex too.

The leg is mounted on a rigid stand at its shoulder; the foot either hangs free (speed, delay, play), presses on a
load cell (torque, heat), or runs on the overhead catch with the foot on a floor sample (stop chain).

## Rules (fixed before any measurement)

A row **passes** if its measured value meets the rule; **fails** sends the result back to the parts list and the
simulation before any further order. Each row says which simulation value it replaces.

| # | Test | Measure | Pass if | Fail means | Replaces in simulation |
|---|---|---|---|---|---|
| B1 | Torque at 10 rpm and stall, at 42 V and 49 V (pack empty and full) | Output torque on the lever rig, per drive: peak at 10 rpm and stall, 3 repeats | Measured peak at 10 rpm ≥ the design peak ÷ 0.70: X6 roll ≥ 36.3 N·m (25.4 / 0.7), X4 pitch ≥ 26.6 N·m (18.6 / 0.7), X8 elbow ≥ 56.8 N·m at the drive (35.8 / 0.9 / 0.7); and within 10 % of the maker's figure (X6 60, X4 30, X8 120 N·m) | Re-size that axis (parts list section 2.6) | The model's torque limits (0.70 × maker peak) |
| B2 | Which X6 ships for $428 | Label, firmware string, mass (0.82 or 0.78 kg), B1 peak | Recorded. If it is the X6-50-L (45 N·m peak, 50 stall), the T20 roll still passes on paper (0.81 of its peak); re-run the sizing | n/a (an identification) | The X6 data in `sim/data/drives.json` |
| B3 | Heat over a 30-minute show loop, shell on | Winding temperature from the drive's own report and the case by thermal camera, at 25 °C and at 35 °C room temperature; loop: walk-duty torque replay 20 min, held paw pose 2 × 2 min at the continuous design torque (elbow 16.2 N·m at T20), standing the rest | Winding ≤ 90 °C at 35 °C (a 10 °C margin under the maker's 100 °C basis) on every drive; no drive derates or faults | Ducting, a larger drive, or a shorter held pose; the parts list's heat margins (T20 pitch trot 0.98) are re-read | The thermal-equilibrium torques (open-air maker tables) |
| B4 | The X4 above 24 V | No-load and loaded speed of the X4 at 42 V and 49 V; does the drive cap itself at 83 rpm? | At the walk's pitch torque (18.6 N·m) the X4 reaches ≥ 69 rpm (1.15 × the 0.30 m/s walk's 60 rpm), and it runs ≥ 30 minutes at 49 V without a fault | The pitch drive moves to an X6 (+0.46 kg, +$159 each; parts list 2.4), and the T. rex's 12S idea needs the same re-check | The 83 rpm cap (assumption, reseller text) |
| B5 | Command-to-torque delay, p99 | The T. rex bench sheet's method, A17 (`hardware/trex_form_study/stage2/jetson75/BENCH_SHEET.md`, sections 1, 3 and 4.1): tick to torque change, M1 coarse and fine plus the piezo M2, 1,000 commands per drive type, with the other drives' traffic replayed: **4 buses × 3 drives at 500 Hz and at 1 kHz** (this robot) and **2 buses × 6 drives at 500 Hz** (the T. rex layout), computer idle and loaded, Teensy path and Jetson + USB-CAN path | p99 of the delay plus feedback age beyond 2 ms, rounded up to a whole 2 ms step, is **≤ the delay at which the walk verdict is judged** (pre-registered; 10 ms proposed until this bench exists, owner to confirm); and bus load ≤ 80 % at the chosen rate | The walk is re-run at the measured step; if over, the loop rate or the bus layout changes before four legs are ordered | The servo delay (0-10 ms swept; the verdict's delay) |
| B6 | Transmission play and stiffness (elbow) | Lost motion at the elbow under ±2 N·m reversals (drive backlash plus the four-bar's rod ends); torsional stiffness under 0-35 N·m; the four-bar ratio over the full -47° to +36° range (1.3:1 build) and the crank's travel | Play ≤ 1.0° at the joint; stiffness ≥ 500 N·m/rad (an estimate that keeps the lower leg's first mode above about 20 Hz, from a lower-leg inertia of about 0.03 kg·m² about the elbow); the 1.3:1 linkage's transmission angle ≥ 40° everywhere | Rod ends, crank pins or the bearing pair change; the simulation adds play | The model's rigid pushrod and constant ratio (assumption A5) |
| B7 | Regen clamp | Bus voltage during a hard stop from the walk's swing speed and during a controlled sit, with the ideal diodes blocking the pack; clamp resistor temperature | Bus stays ≤ 54 V (under the drives' 55 V maximum) with the 52.5 V clamp; resistors under their rating after 10 stops in a row | Clamp sizing changes before the power board is ordered | None (not modelled) |
| B8 | Stop chain: sit, then cut | From a stop input (wireless stop, mushroom on the robot, pendant) to: the Teensy's damped-sit command, the leg reaching its sit pose on the catch rig, and the contactor opening (Pilz delayed contacts set between 0.5 and 1.5 s); 20 trials per input; wireless stop's latency and range in a hall | In 20 of 20 trials the leg reaches the sit pose ≥ 0.2 s before power drops; no input fails to stop; wireless stop latency recorded | The Pilz delay or the sit routine changes; never ship with a stop that drops the leg | The stop behaviour (not modelled yet) |
| B9 | Foot sensor and sole (if the cells have arrived) | FX29 reading against a reference load 0-445 N; I2C error rate on the leg harness with the drives running; the sole's roll behaviour under load | Error ≤ 2 % of full scale; zero I2C errors in 30 minutes; and see the roll-ankle question below | Per-foot CAN node instead of I2C; foot redesign | The touch sensors' ideal readings |

**The roll-ankle question for the bench (from the crawl, `../sim/CRAWL_REPORT.md`).** In simulation, PR #669's foot (a
passive pitch ankle and a sole rigid in roll) stands on its edges whenever the body shifts sideways, and a crawl failed
two of its rules because of it; a passive roll hinge fixed it. If the owner adopts a roll ankle (decision D5), the
bench foot carries one, and B9 checks that the sole stays flat (centre of pressure within 10 mm of the sole's centre)
through a ±80 mm sideways shift of the shoulder. If not, B9 measures how far the sole tips under that shift.

## Order, and what each step gates

1. **Drives alone, foot free:** B2, B1, B4, B5 (no linkage needed). Gate: the drive family is confirmed for the
   Triceratops and for the T. rex's Q2 2027 order.
2. **Linkage fitted:** B6, then B1 again through the linkage.
3. **Shell fitted, thermal:** B3.
4. **Power and safety:** B7, B8, with the foot on the catch rig.
5. **Feet:** B9.

After step 5, the simulation is re-run at the measured values (delay step, play, stiffness, torque limits), and only
then is the four-leg order (milestone T5) put to the owner. The T. rex's purchase gate uses the same B5 rows.

## If a PLEM bench pilot is approved (decision D7; not before WIM's and MyActuator's answers)

Add, beside the CAN path and off the critical path (`PLEM_FIT.md`, section 5.3): (1) a two-drive pilot on an AGX
Orin with the "-E" drives on EtherCAT: `plem drive check`, csp and csv runs, a stale-command stop, a killed master and
a pulled cable, recording each drive's reaction and PLEM's timing; (2) the same loaded foreleg trajectory over the
Teensy CAN path and the PLEM path, measuring B5's delay (p50, p99), the stop behaviour under load (hold or damped sit,
never a drop) and heat.

## Time (estimate)

About 3-5 bench days for steps 1-2, 1-2 days for step 3 (two 30-minute loops per room temperature plus cool-down),
1-2 days for steps 4-5; plus writing the test programs (the T. rex sheet's section 3.1 lists two for B5; the others
need a torque-replay program and a stop-chain logger). None of this is written yet.

## Safety on the bench

The T. rex sheet's section 9 applies: current-limited supply, the stop chain wired first, the leg guarded, nobody in
the leg's sweep while it is powered, the 12S bus treated as a 50 V circuit.
