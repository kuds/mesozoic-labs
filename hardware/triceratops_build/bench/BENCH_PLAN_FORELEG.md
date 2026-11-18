# One-foreleg bench plan (task 8), with the pass and fail rules written first

18 Nov 2026; revised the same day after an independent check. **A plan. Nothing is bought, built or measured; no
purchase without the owner.** Every number below is a target, a maker's figure or an estimate from the parts list
(`hardware/triceratops_kickoff/PARTS_PREMIUM.md`) or from this build's simulation (`../sim/`), labelled as such. The
bench's readings will be the project's first hardware measurements; they go in the results tables only, never beside
the targets as if they were the same kind of number.

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

**Cost, updated 18 Nov 2026 (decision D15).** The certified wireless stop ($1,900) and its receiver estimate ($800) in
the set below are replaced by a fail-safe radio stop (about $250, estimate), so the set is about $4.6k of parts; the
whole plan is still to be revised for the 60 cm foreleg (decision D2). As first written: the parts list's first purchase, the T20 one-foreleg bench set, is **about $7.1k of parts, $8.9k landed**:
the foreleg's drives $1,422, its linkage $120, a torque rig $700, a 60 V bench supply $400, a thermal camera $250, a
PCAN-USB FD $379, two MCU carriers $300, and the full stop chain $3,526 (wireless stop, Pilz relay, e-stops,
contactor, fuse, receiver estimate). Drive prices are MyActuator's own US store, read 7 Nov 2026. At the US reseller
prices found on 18 Nov 2026 the drives' $1,422 becomes **about $1.5k-$2.0k**: $1.5k at the backordered prices (X6
$490, X4 $400, X8 $645, all backordered), $2.0k with RobotShop's tariff-included X6 ($741.84) and X4 ($620, a listing
for the X4-P36-36-E) and the backordered X8: not an in-stock basket.

**Not in the $7.1k set, but the tests below need them** (parts list section 4 prices; estimates marked there):
- the overhead catch gantry, $800 (B8, step 4);
- the power board with the ideal diodes, precharge and 52.5 V clamp, $300 estimate (B7);
- the watchdog board, $80 estimate, and the 24 V safety rail, $35 estimate (the Pilz relay runs on 24 V; B8);
- one limb shell section for the heat test (B3), not priced;
- a 1.3:1 linkage for the ratio and crank-travel check (B6), $120 estimate, only if T15 or the hind knee is checked here;
- for B5's Jetson path, one single-channel USB-CAN adapter per bus: three more PCAN-USB FD, $1,137 (the T. rex sheet
  used two; this robot has four buses);
- the spare drive the torque rig uses as a dynamometer (B1, B4, B7): the parts list counts it among the spares, not in
  the bench set; an X8, $725 (an X6, $428, covers the roll and pitch tests only);
- for B9: four TE FX29 foot cells, $174 (21-23-week lead; parts list), and a three-cell plate, about $150 (estimate).

That is about **+$2.3k** without the extra adapters, **+$3.4k** with them (with the X8 as the dynamometer), before
freight, duty and tax.

The leg is mounted on a rigid stand at its shoulder; the foot either hangs free (speed, delay, play), presses on a
load cell (torque, heat), or runs on the overhead catch with the foot on a floor sample (stop chain).

## Rules (fixed before any measurement)

A row **passes** if its measured value meets the rule; **fails** sends the result back to the parts list and the
simulation before any further order. Each row says which simulation value it replaces. Rules are one-sided: a drive
that beats its figure passes.

| # | Test | Measure | Pass if | Fail means | Replaces in simulation |
|---|---|---|---|---|---|
| B1 | Torque at 10 rpm and stall, at 42 V and 49 V (pack empty and full) | Output torque on the lever rig, per drive: peak at 10 rpm and stall torque, 3 repeats each | (a) Peak at 10 rpm ≥ the design peak ÷ 0.70: X6 roll ≥ 36.3 N·m (25.4 / 0.7), X4 pitch ≥ 26.6 N·m (18.6 / 0.7), X8 elbow ≥ 56.8 N·m at the drive (35.8 / 0.9 / 0.7); (b) peak at 10 rpm ≥ 90 % of the **shipped variant's** maker figure (X4 30, X8 120, X6-60 60 or X6-50-L 45 N·m); (c) stall ≥ 90 % of the maker's stall figure where one is published (X4 36, X8 150, X6-50-L 50 N·m; none for the X6-60: recorded only) | (a) fails: re-size that axis (parts list 2.6). (b) or (c) fails: the drive is under its data sheet; ask the maker before ordering more | The model's torque limits (0.70 × maker peak) |
| B2 | Which X6 ships for $428 | Label, firmware string, mass (0.82 or 0.78 kg), B1 peak | Recorded. If it is the X6-50-L, the T20 roll's design peak (25.4 N·m) is 0.81 of its 0.70 × 45 N·m allowance, so it still passes on paper; re-run the sizing with 45 N·m | n/a (an identification) | The X6 data in `sim/data/drives.json` |
| B3 | Heat over a 30-minute show loop, shell on | Winding temperature from the drive's own report and the case by thermal camera, at 25 °C and at 35 °C room temperature. Loop: walk-duty torque replay 20 min, held paw pose 2 × 2 min at the continuous design torque (elbow 16.2 N·m at T20), standing the rest. **Plus** a 2-minute trot-duty replay on the X4 pitch and the roll drive (parts list section 8, item 2) | Winding ≤ 90 °C at 35 °C (a 10 °C margin under the maker's 100 °C basis) on every drive in both loops; no drive derates or faults | Ducting, a larger drive, a shorter held pose, or short trots only; the parts list's heat margins (T20 pitch trot 0.98) are re-read | The thermal-equilibrium torques (open-air maker tables) |
| B4 | The X4 above 24 V | No-load speed and speed under load at 42 V and 49 V; does the drive cap itself at 83 rpm? | At the swing's pitch torque (about 7 N·m; parts list Appendix C estimates 6-7 N·m to swing the leg) the X4 reaches ≥ 69 rpm (1.15 × the 0.30 m/s walk's 60 rpm), and it runs ≥ 30 minutes at 49 V without a fault. (Stance torque at low speed is B1's job.) | The pitch drive moves to an X6 (+0.46 kg, +$159 each; parts list 2.4); the T. rex's 12S idea needs the same re-check | The 83 rpm cap (assumption, reseller text) |
| B5 | Command-to-torque delay, p99 | The T. rex bench sheet's method, A17 (`hardware/trex_form_study/stage2/jetson75/BENCH_SHEET.md`, sections 1, 3, 4.1 and 5): tick to torque change by M1 coarse and fine and the piezo M2, **1,000 commands per row** plus the sheet's 300-command check, with the other drives' traffic replayed. Rows: **this robot**, 4 buses × 3 drives at 500 Hz and at 1 kHz, Teensy path and Jetson + USB-CAN path, computer idle and loaded; **the T. rex sheet's rows unchanged** for the X6-60 and X4-36-L: Teensy 2 buses idle and loaded and 3 buses loaded at 500 Hz, Jetson 2 buses idle and loaded at 500 Hz and 1 bus loaded at 250 Hz; and its five 30-minute soak runs (late cycles, replies that missed the next tick, frames lost or resent) | The reading (p99 of the delay plus the feedback age beyond 2 ms, rounded up to a whole 2 ms step) is **≤ 10 ms**, the delay at which the walk verdict is judged (owner's decision D9, 18 Nov 2026), on the path chosen for the robot; bus load ≤ 80 % at the chosen rate; and no frame lost in the soak runs | The walk is re-run at the measured step; if the reading is over 10 ms, the loop rate or the bus layout changes before four legs are ordered | The servo delay (0-10 ms swept; the verdict at 10 ms) |
| B6 | Elbow transmission: play, stiffness, angles | Lost motion at the elbow under ±2 N·m reversals (drive backlash plus the four-bar's rod ends); torsional stiffness under 0-35 N·m; the transmission angle of the **1:1** four-bar over the full -47° to +36° flexion range. If a 1.3:1 linkage is built (T15, or the hind knee): its ratio over the range and its crank travel (simulation estimate 119°, parts list 108°) | Play ≤ 1.0° at the joint; stiffness ≥ 400 N·m/rad, which keeps the first mode of the lower leg on its drive at or above about 20 Hz (from the T20 model's 0.0236 kg·m² about the elbow: 0.0147 for the lower leg and foot plus 0.0089 of the X8's reflected inertia, a catalog estimate); transmission angle ≥ 40° everywhere | Rod ends, crank pins or the bearing pair change; the simulation adds play; for 1.3:1, the angle-dependent ratio goes into the model | The model's rigid pushrod; for 1.3:1, its constant ratio (assumption A5) |
| B7 | Regen clamp, scaled to the robot | Bus voltage during (1) a hard stop of the leg from the walk's swing speed and (2) an injected regen pulse equal to the walk simulation's estimate for **twelve drives** in a hard sit, with the ideal diodes blocking the pack; clamp resistor temperature. The pulse comes from a source that can push energy into the bus: the dynamometer drive back-driving the elbow **from its own, separate supply**, a bidirectional (four-quadrant) supply, or a switched, pre-charged capacitor bank. Its energy and duration are computed in the walk simulation and committed before B7 runs | Bus ≤ 54 V (under the drives' 55 V maximum) in both, with the 52.5 V clamp; resistors under their rating after 10 twelve-drive pulses in a row | Clamp or resistors re-sized before the power board is ordered | None (not modelled; the twelve-drive energy estimate is a task for the walk simulation) |
| B8 | Stop chain: sit, then cut | From a stop input (the handheld radio stop, decision D15: a press, the transmitter switched off, and its signal blocked; the mushroom on the robot; the pendant) to: the Teensy's damped-sit command, the leg reaching its sit pose on the catch rig, and the contactor opening (Pilz delayed contacts set between 0.5 and 1.5 s); 20 trials per input; the radio stop's latency from press and from loss of signal, and its range in a hall | In 20 of 20 trials the leg reaches the sit pose ≥ 0.2 s before power drops; no input fails to stop | The Pilz delay or the sit routine changes; never ship with a stop that drops the leg | The stop behaviour (not modelled yet) |
| B9 | Foot cell and sole | FX29 reading against a reference load 0-445 N; I2C error rate on the leg harness with the drives running; the sole's roll behaviour under a ±80 mm sideways shift of the shoulder, measured with a three-cell plate under the sole (or pressure-mapping film) | Error ≤ 2 % of full scale; zero I2C errors in 30 minutes; the sole's centre of pressure stays within 10 mm of its centre through the shift if a roll-flat foot is fitted, and its tilt is recorded if not | Per-foot CAN node instead of I2C; foot redesign | The touch sensors' ideal readings; the sole's contact model |

**The foot question for the bench (from the crawl, `../sim/CRAWL_REPORT.md`).** In simulation, PR #669's foot (a
passive pitch ankle and a sole rigid in roll) tips onto its edges whenever the body shifts sideways, and a crawl failed
two of its rules because of it. A passive roll hinge kept the soles flat and passed with the default contact model, but
failed under a stiffer friction model, so the foot design is open (decision D5). B9 measures what the chosen foot
does.

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
Teensy CAN path and the PLEM path, with the other drives' bus traffic emulated as PR #669 asks, measuring B5's delay
(p50, p99), the stop behaviour under load (hold or damped sit, never a drop) and heat.

## Time (estimate)

About 6-10 bench days for steps 1-2 (an estimate: B5 alone has 36 rows and five 30-minute soak runs, where the T. rex
sheet estimates 10.5-16.5 hours for its own 12 leg-drive rows and soaks), 1-2 days for step 3
(two 30-minute loops per room temperature plus cool-down), 1-2 days for steps 4-5; plus writing the test programs (the
T. rex sheet's section 3.1 lists two for B5; the others need a torque-replay program, a regen pulse program and a
stop-chain logger). None of this is written yet.

## Safety on the bench

The T. rex sheet's section 9 applies: current-limited supply, the stop chain wired first, the leg guarded, nobody in
the leg's sweep while it is powered, the 12S bus treated as a 50 V circuit.
