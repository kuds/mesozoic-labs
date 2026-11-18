<!-- Working paper, 16 Nov 2026: one of three readers' reports gathered for the design of the servo model of gate 1 (BUY_GATE_DRAFT.md). Not reviewed. Nothing here is a rule or a result. Outside sources are quoted as found and were not checked a second time. -->

# Reader 2: hardware facts in the repository on the real actuators (75-M-H-10; 60-N-T-10 where it differs)

Read only; nothing run but file reads. No file written (the scratchpad folder `/tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/servo_design/` exists and is empty). The repository holds **no hardware measurement** and no third-party measurement of any actuator of the build.

**File keys** (relative to the study directory):
- ACT = `designlab/data/actuators.json`, the catalogue the design lab reads
- RES = `designlab/data/research/actuators_2026-11-07.json`, the research record behind it; it alone holds currents, backlash and the full notes
- BC = `designlab/data/build_catalog.py`; CAT = `designlab/catalog.py`
- REC = `round2/refine/feet_75/candidates/75-M-H-10.json`
- PLANT = `stage1/runs/2026-11-08/75-M-H-10/plant.json`
- LAT = `stage2/jetson75/LATENCY_AND_BUS.md`; BENCH = `stage2/jetson75/BENCH_SHEET.md`; BUD = `stage2/jetson75/BUDGET_75.md`
- DL = `DECISION_LOG.md`; MAT = `round2/refine/MATERIALS.md`

**Labels:**
- [M] maker's value, as the repository records it with the maker's URL on the cited lines. I did not open the pages.
- [M*] maker or distributor document read only through search extracts on 14 Nov (LAT:17-19).
- [A] assumption of the study. [D] derived by the study. [E] my own estimate.

## 1. Actuators of the 75 cm build (a prospective design; nothing is bought)

| Joint | Count | Model (catalogue id) | Source |
|---|---|---|---|
| hip roll, hip yaw (thigh twist), knee, ankle motors a and b | 10 | MyActuator RMD-X4-P36-36-E-L, V4.1 (`myactuator_rmd_x4_p36_36_e_l`) | REC:27-28, 30-32; LAT:99-100; BUD:380 |
| hip pitch | 2 | MyActuator RMD-X6-P20-60-E, V4 (`myactuator_rmd_x6_p20_60_e`) | REC:29; LAT:99; BENCH:178 |
| neck pitch | 1 | ROBOTIS XM540-W270-T (`robotis_xm540_w270_t`) | REC:18; LAT:101 |
| tail yaw | 1 | Feetech STS3215 12 V, ST-3215-C018 (`feetech_sts3215_c018`) | REC:19; LAT:102 |
| expressive head, optional: neck yaw and jaw | 1 + 1 | STS3215-C018 | BUD:318; `stage2/x/75-M-H-10-X/plant.json:279-312` |

- **Jaw:** deferred for the build and modelled in simulation only (DL:44, P29).
- **Ankle:** two motors high on the shank drive pitch and roll through push rods (DL:66, E5). The model uses unit cranks, motor a on q_p + q_r and b on q_p − q_r (`designlab/stage1/drive.py:5-17`).
- **Which X6 ships is open:** the US store page at $428 lists 0.78 kg, 50 N·m and 10 arcmin, which match the V4.1 X6-50-L, not the X6-60 (ACT:208; BUD:578-579). The X6-50-L has peak 45 N·m, stall 50 N·m, Kt 1.8, backdrive 0.2 N·m and no US price (RES:157-186).

## 2. Per model

| Parameter | X4-P36-36-E-L (×10) | X6-P20-60-E (×2) | XM540-W270-T (neck) | STS3215-C018 (tail) |
|---|---|---|---|---|
| Mass | 0.36 kg [M] RES:59 | 0.82 kg [M] RES:127 | 0.165 kg [M] RES:1486 | 0.055 kg [M] RES:1550 |
| Size | Ø55 × 61 mm [M] RES:60-62 | Ø80 × 67.5 mm [M] RES:128-130 | 33.5 × 58.5 × 44 mm [M] RES:1489 | 45.2 × 24.7 × 35 mm [M] RES:1553 |
| Maker rated torque | 10.5 N·m at 83 rpm, 24 V; a duty-cycle rating (90° swings, 5 s holds), not thermal [M] RES:63-64 | 20 N·m at 153 rpm, **48 V**, 320 W; no thermal condition [M] RES:131-132 | none published [M] RES:1490-1491 | 0.981 N·m (10 kg·cm at 0.9 A); no thermal definition [M] RES:1554-1555 |
| Maker peak or stall | peak 30 N·m, defined at 10 rpm; stall 36 N·m [M] RES:64-65, 85 | peak 60 N·m [M] RES:133 | stall 10.6 N·m at 12 V (10.0 at 11.1 V, 12.9 at 14.8 V) [M] RES:1492, 1511 | stall 2.942 N·m at 12 V [M] RES:1556, 1578 |
| Planning sustained torque, the simulation's limit for continuous load | 5.985 N·m = 0.57 × rated [A: A1] CAT:27; PLANT:288 | 11.4 N·m [A: A1] PLANT:300 | 1.770 N·m = 0.167 × stall [A] BC:16-18, 69; PLANT:264 | 0.4913 N·m = 0.167 × stall [A] PLANT:276 |
| Planning peak torque, the simulation's force limit | 21.0 N·m = 0.70 × peak [A: A2] CAT:29; PLANT:287 | 42.0 N·m [A: A2] PLANT:299 | 3.106 N·m = 0.293 × stall [A] BC:70; PLANT:263 | 0.862 N·m [A] PLANT:275 |
| Maker rated and no-load speed | 83 and 111 rpm at 24 V [M] RES:66-67 | 153 and 176 rpm at 48 V [M] RES:134-135 | no rated speed; no-load 30 rpm at 12 V (28 / 37 at 11.1 / 14.8 V) [M] RES:1493-1494, 1511 | no rated speed; no-load 45 rpm [M] RES:1557-1558 |
| Speeds the simulation uses (corner, zero-torque) | 8.692 and 11.624 rad/s [D] PLANT:289-290 | 8.011 and 9.215 rad/s: the 48 V figures × 24/48 [A/D] CAT:77-82; PLANT:301-302 | 1.885 (0.6 × no-load) and 3.142 rad/s [A/D] BC:93; PLANT:265-266 | 2.827 and 4.712 rad/s [A/D] PLANT:277-278 |
| Supply | nominal 24 V, window 20-55 V [M] RES:68-69 | nominal 48 V, window 20-55 V [M] RES:136-137; run on the 24 V bus [A] REC:20 | 12 V recommended, 10.0-14.8 V [M] RES:1495-1496 | 12 V, 4-14 V [M] RES:1559-1560 |
| Current | rated 6.1 A rms (8.6 A peak); 21.2 A rms phase at stall [M] RES:73-74, 85 | rated 9.5 A, peak 29.1 A rms [M] RES:141-142 | 4.4 A at 12 V stall [M] RES:1501 | rated 0.9 A, stall 2.7 A [M] RES:1564-1565 |
| Gear ratio | 36 [M] RES:70 | 19.612 [M] RES:138 | 272.5 [M] RES:1497 | 345 [M] RES:1561 |
| Gear type | no field in the repository | no field | no field | no field |
| Backlash | 15 arcmin (0.25°) [M] RES:75 | ≤15 arcmin [M] RES:143; store page says 10 (RES:154) | 15 arcmin [M] RES:1502 | ≤0.5° [M] RES:1566, 1578 |
| Torque constant, output side | 1.8 N·m/A rms (1.3 per A peak) [M] RES:71, 85 | 2.1 N·m/A rms [M] RES:139 | 2.409 N·m/A [M] RES:1498 | 1.079 N·m/A [M] RES:1562 |
| Winding | line 0.4 Ω, 0.18 mH, 11 pole pairs; catalogue holds 0.2 Ω per phase [M/D] RES:72, 85 | line 0.41 Ω, 0.51 mH, 10 pole pairs [M/D] RES:140, 154 | resistance missing, ACT:1463 | 1.0 Ω terminal, brushed iron-core [M] RES:1563, 1578 |
| Inertia, maker's figure | "inertia 0.000082 kg·m²" [M] RES:85; rotor or output side not stated | "0.66 kg·cm²" [M] RES:154; side not stated | none | none |
| Armature in the simulation | 0.00324 kg·m² [A: A7] BC:14-15, 95; PLANT:291 | 0.004304 kg·m² [A: A7] PLANT:303 | 0.000201 kg·m² [A] BC:95; PLANT:267 | 0.000226 kg·m² [A] PLANT:279 |
| Friction, maker | backdrive 0.2 N·m; efficiency 64.7 % [M] RES:85 | backdrive 1.6 N·m; efficiency 72.7 % [M] RES:154 | none | idle current not given for this model |
| Thermal, maker | equilibrium after 90 min at 8.5 / 8 / 8.5 / 7.5 N·m (0 / 10 / 30 / 83 rpm); 10.5 N·m reaches 100 °C in 5.4-12.4 min; 21 N·m lasts 44 s, 31.5 N·m 7 s [M] RES:64 | stall table only: 30 N·m +17 °C in 15 s, 40 N·m +29 °C in 10 s, 60 N·m +24 °C in 5 s; sibling X6-50-L settles at 9-13 N·m [M] RES:132 | "stall differs from continuous" [M] RES:1491 | none |
| Encoder | dual, 17 and 18 bit; accuracy 0.4°, repeatability 0.3° [M] RES:76, 85 | dual, 17 and 17 bit [M] RES:144 | not in the catalogue; "12 bits" is the author's (d), BENCH:529 | 12-bit magnetic [M] RES:1567 |
| Bus | CAN and EtherCAT [M] RES:76; classic CAN, 1 Mbit/s, 8-byte frames, command 0x140+n, reply 0x240+n, multi-motor 0x280 [M*] LAT:114-115, 441; no CAN FD found, LAT:129 | same | TTL half-duplex, 9.6 kbit/s-4.5 Mbit/s [M] RES:1503; ships at 57,600 bit/s; 5 V logic [M*] LAT:338-345 | TTL half-duplex, 38.4 kbit/s-1 Mbit/s [M] RES:1567; factory 1 Mbit/s [M*] LAT:457 |
| Control modes | "motion mode" on 0x400+n; a **sibling** manual gives position, speed, KP 0-500, KD 0-5 and feedforward torque, unconfirmed for these drives [M*] LAT:104-110, 442; EtherCAT position, velocity and torque modes [M*] LAT:445 | same | not in the repository | "write position, time and speed" (d) LAT:327; modes not in the repository |
| Internal loop rates | not found for V4 or V4.1; the earlier MC-X drivers: current 15 kHz, speed 5 kHz, position 1 kHz or 500 Hz [M*] LAT:241, 443 | same | not in the repository | not in the repository |
| Reply latency | turnaround not found, LAT:216-218; highest accepted command rate not found, LAT:51-52 | same | return delay 500 µs default, settable to 0 [M*] LAT:454 | default disputed, 0 or 500 µs, LAT:460 |
| Command to torque | 0.10-1.50 ms, the author's estimate (2.50 ms if the loop is 500 Hz) [A] LAT:241 | same | not known, LAT:346-347 | not known |
| When commands stop | a time-out setting exists; behaviour not read; planned as torque-off [A] LAT:375-377 | same | not in the repository | not in the repository |
| Price | $269 [M] RES:77 | $428 [M] RES:145 | $482.89 [M] RES:1504 | $22 for the equivalent C047 at Seeed; the C018 is $27.99 [M] RES:1568-1569 |

**The simulation's servo today** (simulation evidence, for comparison):
- Ideal PD position servos. kp = planning sustained torque ÷ 0.01 rad [A] (`designlab/carpet.py:176`; `designlab/stage1/plant.py:858`); kv is critical damping on the link inertia (`plant.py:864`).
- Gains: hip pitch 1140 / 30.8, other X4 joints 598.5 / 7.6-24.5, each ankle motor 299.25 / 2.08, neck 177 / 4.16, tail 49.1 / 2.72 (PLANT:113-150; BENCH:499-506).
- Torque-speed envelope: full planning peak up to the corner speed, then linear to zero at no-load; braking at peak at any speed (`drive.py:21-37`).
- Every joint: damping 0.2 N·m·s/rad and friction loss 0.15 N·m (`designlab/mjcf.py:120`; `round2/refine/feet_75/models/75-M-H-10.xml:11`).
- Delay: k whole 2 ms steps on every target (`designlab/stage1/env.py:513-522`). No backlash (`STAGE2_PLAN.md:306`).
- If the sibling manual's gain caps applied in the model's units, the drive would reach 0.44 of the hip-pitch kp and 0.16-0.66 of the hip and knee kv (BENCH:497-512).

**Three discrepancies, my own reading [E]:**
1. **Envelope.** The simulation allows 21 N·m at 83 rpm on the X4 and 42 N·m at 76.5 rpm on the X6 at 24 V. The maker's rated points are 10.5 N·m at 83 rpm (24 V) and 20 N·m at 153 rpm (48 V). A straight line through each maker rated and no-load point, with speeds halved for the X6 at 24 V, gives about 10.5 and 10 N·m at those corners, and reaches 42 N·m on the X6 only below about 40 rpm (4.2 rad/s). The repository has no torque-speed curve; it flags this itself (`LESSONS_LEARNED.md:113`; A2 "dyno curves").
2. **Inertia.** If the maker's inertia figures are rotor-side, the reflected values are 0.106 kg·m² (X4) and 0.0254 kg·m² (X6): 33 and 5.9 times A7's estimates, and above the knee's whole link inertia in the model (0.055 kg·m², PLANT:143-148). If they are output-side, A7 is 40 to 65 times too high. The catalogue ignores them (BC:95).
3. **Friction.** The maker's backdrive torque on the X6-60 is 1.6 N·m against the model's 0.15 N·m (BENCH:560-561). The maker's own sources disagree between V4 and V4.1 (X4-36: 1.14 against 0.2 N·m; resistance, pole pairs and backlash also differ, RES:52).

## 3. The study's assumptions and findings on servos (DECISION_LOG.md)

| Row | Line | Substance |
|---|---|---|
| A1 | DL:106 | Sustained planning torque = 0.57 × maker rated. MyActuator V4.1 tables settle at 50-80 % of rated. Replace with bench thermal tests, the STS3215 tail first (closed-loop 0.51-0.64 N·m at 75 cm against 0.4913). |
| A2 | DL:107 | Peak planning torque = 0.70 × maker peak; requirement margins 1.25 (peak) and 1.15 (speed). Replace with dyno curves. |
| A6 | DL:111 | Power: gear efficiency 0.75, drive 0.92, idle 1.4 W per axis, DC-DC 0.88, 75 % usable energy, no regeneration credit (`designlab/power.py:21-27`). |
| A7 | DL:112 | Armature = 2.5e-6 kg·m² × (D / 55 mm)⁴ × gear². Replace with vendor data or measurement. |
| A11 | DL:116 | Bus swing 21.0-25.2 V on 6S. On 12S the X-series needs the vendor's bus capacitor and a regeneration clamp below 55 V. |
| A17 | DL:122 | Stage 1 verdicts assume no servo delay. The 75 cm walk misses a limit from 8 ms and falls at 10 ms. Replace with a bench measurement of each drive type's delay and of servo stiffness. |
| A18 | DL:123 | Damping 0.2 N·m·s/rad and friction 0.15 N·m on every joint, unsourced; they dominate the tail servo's load. Replace with a bench measurement. |
| E5 | DL:66 | Ankle driven by push rods from two motors on the shank. |
| E6 | DL:67 | Leg drives must be CAN; RS-485-only and discontinued parts excluded. |
| E8 | DL:69 | X4-36 -E against -E-L: peak 34 against 30 N·m, Kt 1.9 against 1.8; 0.4 Ω is line-to-line. |
| E9 | DL:70 | Supply windows: X-series 20-55 V, AK45 15-28 V only. Below nominal voltage, usable speed falls in proportion. |
| F12 | DL:80 | Stage 0: the neck servo is the hottest drive, 0.76-0.97 of sustained at 75 cm. |
| F15 | DL:83 | Sizing at the animal pace turned the hip pitch into an X6-60 and the thigh twist into an X4-36-L, still on 24 V. |
| F22 | DL:90 | The governing limit is the knee's speed (0.95), which is kinematic; lighter materials buy torque headroom, not speed. |
| F26 | DL:94 | Stage 1: the tail servo at 1.04-1.31 of A1 at the speed walks; the walk meets every limit to 6 ms and stays up to 8 ms; stiffness down to 0.55 × with the controller unaware. |
| F27 | DL:95 | Stage 1b: at 10 ms every 60 cm walk falls, refitted with delay or not; peak drive speeds 7-8 % above Stage 0's need. |
| P18 | DL:33 | The 75 cm body takes the targeted drive upgrade. |
| P29, P30, P32 | DL:44, 45, 47 | The bar is judged at 10 ms until a bench value exists; the microcontroller loop is the baseline; the bench is not expected before a 75 cm build works in simulation. |
| P39 | DL:54 | The buy gate draft; a realistic servo model joins the training path. |

Also relevant:
- **Pass rule:** every success needs, on each drive, RMS torque ≤ sustained and peak speed ≤ the corner speed (`STAGE1_PLAN.md:91`).
- **Randomisation planned:** delay 0-24 ms, stiffness ×0.6-1.2, joint damping and friction ×0.5-1.5, trunk mass ×0.9-1.1 (`STAGE2_PLAN.md:125-126, 360`). The Stage 1 code has delay 0-20 ms and kp ×0.8-1.2 (`designlab/stage1/env.py:234-235`).
- **Draft rows, not adopted:** the leg bus, the delay estimates, the servo form (LAT:469-487).

## 4. What the bench sheet plans to measure (nothing is measured yet, BENCH:11)

| Planned measurement | Where | Servo-model parameter it pins |
|---|---|---|
| H: tick to end of the last command frame; p50, p99, max; two paths; idle and loaded | BENCH:85-87, 381-398 | host and bus share of the command delay |
| D: command frame to torque change, per leg-drive type, in 2 ms and 0.2 ms bins | BENCH:88-89, 119-120, 154-161 | the drive's own command-to-torque delay |
| T = H + D, also by a piezo sensor at 0.1 ms | BENCH:84, 121, 162-171 | end-to-end delay, mapped to k |
| S: feedback age; reply turnaround per drive type | BENCH:93-94, 263-266 | sensor delay |
| 30-minute soak: late cycles, missed replies, lost frames | BENCH:342-345, 405-412 | delay jitter and dropouts |
| Neck and tail servo delay at 100 Hz; their reply-delay and bit-rate settings | BENCH:347-370 | the two servos' delay |
| What a drive does when commands stop | BENCH:741-742 | time-out behaviour |
| Sheet A: largest kp and kv accepted, with units; static stiffness by encoder and at the lever; overshoot of a 0.01 rad step | BENCH:519-544 | reachable gains; static stiffness with gear and mount flex |
| Sheet B: breakaway torque unpowered; torque current at 4 constant speeds, fitted as friction + damping × speed | BENCH:563-590 | Coulomb friction and viscous damping (A18); one-point check of the torque constant |
| Sheet E: tail servo, 20-minute holds at 0.49, 0.52 and 0.64 N·m, static and swinging | BENCH:678-691 | the tail servo's continuous torque (A1), in open air |
| Sheets C and D: ToF stop time; tread grip | BENCH:596-661 | not servo parameters |

**Servo-model parameters no planned measurement covers:**
- **Gear play.** No backlash measurement exists. The delay steps deliberately keep the gear on its driving flank (BENCH:327-331), and sheet A loads one way only. Yet `BUY_GATE_DRAFT.md:48-49` says the bench measures gear play. Push-rod play at the ankle is also absent.
- **The torque-speed envelope:** peak torque against speed, no-load speed at the real bus voltage (A2's dyno curves). Sheet B stops at 4 rad/s on a free shaft, against 8.0-8.7 rad/s used.
- **Voltage sag:** pack internal resistance, harness resistance, bus voltage under load. Only an optional repeat of the delay run on the pack (BENCH:186).
- **Rotor inertia** (A7).
- **Torque rise time**, the drive's loop rate and the accuracy of a feedforward torque command. D times only the first change.
- **Leg drives' and neck servo's heating** and peak-torque duration. Only the tail is planned.
- **Load-dependent and temperature-dependent friction;** friction above 4 rad/s; torque ripple.
- **Encoder noise and velocity filtering.**
- **Braking and regeneration** (bus over-voltage; the envelope assumes full peak when braking).
- **The neck and tail servos' internal control law** and their torque-speed curves.
- **The controller's compute step:** no firmware exists (BENCH:90-92).

## 5. Power system (a prospective design)

| Item | Value | Label and source |
|---|---|---|
| Bus | 24 V nominal | [A] REC:20; PLANT:18 |
| Pack | Gens ace G-Tech 6S 3300 mAh LiPo, 22.2 V, 73.26 Wh, 0.506 kg, $107.19; "60C" = 198 A is the researcher's arithmetic on a marketing rating | [M/D] `designlab/data/parts.json:193-216` |
| Bus swing | 21.0-25.2 V over the usable charge | [A: A11] DL:116 |
| Leg drives on this bus | inside their 20-55 V window; the X6 runs at half its nominal voltage | [M] RES:69, 137; [A] CAT:77-82 |
| Servo rail | 12 V from a step-down regulator. The bench lists one Pololu D36V50F12 (4.5 A) for neck and tail, whose stall currents sum to 7.1 A | [A] BENCH:184, 745-746 |
| Regulators on the robot | the power lot's contents are not in the repository. The 60 cm basis names D36V50F12 (compute), D24V10F5 (logic) and D42V110F12, 12 V 9 A (servos), plus a separately selected regenerative clamp | BUD:598-603; `r0_posture_study_2026-11-07/study/hardware/component_basis.md:12` |
| Walking draw | 89.8 W total, 69.4 W drives; summed drive peaks 193.8 W, 8.9 A at 24 V with auxiliaries. A prescribed nominal walk under A6, not measured; stops, pushes and falls will be higher | simulation, REC:289, 300-301; MAT:205-206, 220-221 |
| External input | 24 V through a priority power path; 5 m of 16/2 cable drops 1.14 V at the 75 cm peak of 8.7 A; clamp at 27 V or below; a 31 A supply | [A, research estimate] MAT:222-238 |
| Voltage in the simulation | Stage 1 ran the envelope at 25.2, 22.2 and 21 V: the 75 cm nominal walk works at all three (3 episodes each). This scales speed only; nothing sags under load | simulation, `STAGE1_PLAN.md:115`; `stage1/runs/2026-11-08/STAGE1_PROBE.md:206` |
| Buses | two CAN buses of 6 drives at 500 Hz, 67-81 % load; neck and tail each on its own serial port at 100 Hz | [D/A] LAT:90, 142, 326-336 |

Not in the repository: pack internal resistance, any sag figure, wire gauges or harness resistance (only harness masses, `designlab/build.py:569, 735`), connector resistance, bus capacitance, fuse ratings. The maker "recommends an external bus capacitor" (RES:35, on the V4 X4-36-E entry).

## 6. The 60 cm body 60-N-T-10, where it differs

Sources: `round2/refine/narrow_carpet_60/candidates/60-N-T-10.json:16-33`; `stage1/runs/2026-11-08/60-N-T-10/plant.json:262-418`.

| Item | 60-N-T-10 | Source |
|---|---|---|
| Hip roll and hip yaw (4) | CubeMars AK45-36 V3: 0.349 kg; rated 8 N·m at 40 rpm, peak 24 N·m, no-load 52 rpm at 24 V; gear 36; backlash 12 arcmin; supply **15-28 V only**; 2 A rated, 6.5 A peak; 0.9 Ω per phase, 1100 µH; backdrive 0.8 N·m; CAN 1 Mbit/s; 16-bit encoders; $185.90. Simulation: sustained 4.56, peak 16.8 N·m, 4.19 and 5.45 rad/s | [M] RES:649-676; plant.json:286-291 |
| Hip pitch and knee (4) | X4-P36-36-E-L, as at 75 cm | plant.json:298, 322 |
| Ankle motors (4) | MyActuator RMD-X4-P12-10-E, V4, flagged **discontinued**: 0.33 kg; rated 4 N·m at 238 rpm, peak 10 N·m, no-load 317 rpm; gear 12.5; Kt 0.8; backlash 15 arcmin; backdrive 0.8 N·m; $285. Simulation: sustained 2.28, peak 7.0 N·m, 24.9 and 33.2 rad/s; armature 0.00039 kg·m² | [M] RES:88-120; plant.json:334-339 |
| Neck | Feetech STS3250: 74.5 g; stall 4.90 N·m at 12 V and 4.2 A; rated 1.57 N·m; no-load 75 rpm; gear 345; backlash ≤0.5°; 6-12 V; $89. Simulation: sustained 0.819, peak 1.437 N·m | [M] RES:1581-1611; plant.json:262-267 |
| Tail, bus, pack | the same STS3215, 24 V, same pack; 69.3 W walking, 115.7 W summed peaks; 10.81 kg | record:300-301; plant.json:26 |
| Bus protocol | CubeMars and MyActuator drives share the leg buses. LAT and BENCH cover the 75 cm body only (LAT:4; BENCH:12); no timing fact on the AK45 or X4-10 exists | absent |
| Supply ceiling | the AK45's 28 V limit is why the clamp is set at 27 V or below | MAT:236-238 |

## Final lists

**Known from the maker** (recorded with URLs; not re-opened by me):
- Mass, size, gear ratio, backlash limit, torque constant, winding resistance, supply window, price, and encoder bits of the leg drives and the tail servo.
- Rated and peak or stall torque, rated and no-load speed, with the rated-torque definition.
- Rated and stall current; X4-36-E-L temperature-rise table; X6-60 short stall table.
- Backdrive torque and efficiency of both leg drives; an inertia figure for each, side not stated.
- Bus type. Through search extracts only: CAN frame format and identifiers, a sibling's gain ranges, the earlier driver generation's loop rates, the XM540's baud rate and return delay.

**Assumed by the study:**
- A1 (0.57 × rated; 0.167 × stall for box servos), A2 (0.70 × peak; 0.293 × stall), A7 (armature rule), A18 (damping and friction), A6 (efficiencies), A11 (bus swing).
- Speed proportional to bus voltage; the usable speed of servos as 0.6 × no-load.
- The envelope's shape and full-peak braking.
- kp = sustained ÷ 0.01 rad with critical damping; the drive's own PD loop as the servo form.
- Delay as whole 2 ms steps, judged at 10 ms; every delay term in LAT.
- Torque-off on lost commands; the neck's 12-bit encoder.

**Not in the repository:**
- Any measurement.
- Gear type; a torque-speed curve for any drive; any X6-60 data at 24 V.
- Whether the maker's inertia is rotor-side or output-side.
- Loop rates and reply turnaround of the V4 and V4.1 drives; their highest command rate; gain ranges and units for these models; time-out behaviour.
- The XM540's control modes, resistance and thermal data; the STS3215's default return delay and control law.
- Pack internal resistance, sag, wiring and connector data; the 75 cm power lot's contents.
- Play in the ankle push rods.
- Which X6 variant ships at $428.

`docs/hardware/HARDWARE_BOM.md` at the repository root names STS3215 and RMD-X6-60 for other robots, not this build.
