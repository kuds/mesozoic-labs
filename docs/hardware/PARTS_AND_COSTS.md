# Parts and costs

Parts and estimated costs of the main robot concepts priced so far. Nothing has been built or bought.
Every figure is a desk estimate from list prices on a stated date and cost-model allowances; none is a quote. Status:
[README.md](README.md). Masses, dimensions and size terms (crown height, frill height, Hybrid and Tapered bodies):
[SIZES.md](SIZES.md). What happened: [LESSONS_LEARNED.md](LESSONS_LEARNED.md).

## How to read the numbers

- **Bases.** *Parts*: list prices and estimates, before freight, import duty and sales tax; *parts + n %* adds a
  contingency. *Turnkey*: the T. rex study's cost model for one prototype, with outsourced fabrication, a subcontractor
  who fits the robot together, bench gear, spares and a 5.3 / 7.2 / 8.4 % contingency that also covers freight, duty and
  tax; *self-assembled* drops the fit-up. *Landed* (Triceratops lists): the cost delivered and paid, that is parts plus
  freight, duty on the MyActuator drives and sales tax, then a contingency. In-house engineering time is never priced.
- **Low / nominal / high.** Nominal is the planning figure, low and high a range.
- **Dates and uncertainty.** July studies: 19 and 24 Jul 2026 prices, ±15–25 %. T. rex catalog: 7 Nov 2026; cost
  estimates 8 and 14 Nov 2026. Triceratops lists: 18 Nov 2026, mixing store, catalog, July and web prices and estimates.
  Most T. rex spread is in the frame, coverings and fit-up, scaled from a 60 cm estimate: $8,054 of the $12,516
  low-to-high gap at 75 cm. Prices rose: Jetson Orin Nano Super kit $249 to $399 in 2026, Pi 5 8 GB $80 (launch) to
  $175.
- **Target.** The project's cost target for the first build is about $20k, counted landed and all-in.

## Cost at a glance

| Concept | Low / nominal / high | Basis | Date |
|---|---|---|---|
| 60 cm Triceratops | $20,105 / $21,240 / $22,988 | landed; duty and frame-cost cases | 18 Nov 2026 |
| 75 cm Triceratops, 20 kg list | $22,908 / $37,280 / $66,386 | landed | 18 Nov 2026 |
| 75 cm Triceratops, 15 kg list | $19,971 / $32,827 / $58,486 | landed | 18 Nov 2026 |
| 75 cm T. rex, Hybrid body (Pi 5) | $14,228 / $20,057 / $28,215 | turnkey | 14 Nov 2026 |
| 60 cm T. rex, Tapered body | $10,801 / $14,809 / $20,544 | turnkey, round-two basis | 8 Nov 2026 |
| One-metre T. rex concept | $17,286 | parts + spares + 25 % | 25 Oct 2026 |
| Compsognathus walker | $480 / $954 / $2,540 (tiers) | parts + 15 % | 19 Jul 2026 |
| Velociraptor walker | $5,615 / $8,665 / $12,340 (tiers) | parts + 20 % | 19 Jul 2026 |
| Raptor runner, full size | $26k / ~$36k / $45k | parts, spares, contingency | 24 Jul 2026 |
| Raptor runner, 0.72 scale | $5.6k / ~$6.9k / $8k | parts + 25 % | 24 Jul 2026 |
| Psittacosaurus on Pupper V3 | ~$4,700 ($4.2k–4.8k) | parts + allowances | 24 Jul 2026 |
| Dibothrosuchus-inspired | ~$10,000 ($9k–11k) | parts + allowances | 24 Jul 2026 |

The two T. rex turnkey figures differ in scope: on the 8 Nov basis of the 60 cm figure the 75 cm body was $18,521
nominal; the 14 Nov figure adds a cable boom, an external DC input, tread tests and thinner covers (+$1,536 nominal).

## 60 cm Triceratops (first planned build)

Chosen on 18 Nov 2026: 60 cm at the frill, 1.2 m long, about 12 kg, with the 75 cm T. rex's leg drive types. Parts
$16,776; landed $20,105 with drives from the maker's US warehouse (no duty), $21,240 at a 25 % planning duty, $22,149
shipped from China (about 41.5 % tariffs, brokerage).

| Subsystem | Part | Qty | Unit price | Price date |
|---|---|---|---|---|
| Leg drives | MyActuator RMD-X4-P36-36-E-L; RMD-X6-P20-60-E (front elbows) | 10; 2 | $269; $428 | 18 Nov 2026 |
| Head and tail | 3 ROBOTIS XC430-W240-T, 2 XL430-W250-T, 2 XL330 eyelids; lights, audio | 1 set | $976 | 18 Nov 2026 |
| Compute and bus | Jetson kit (choice still open), NVMe, two Teensy 4.1 CAN boards | 1 set | $764 | 18 Nov 2026 |
| Sensors, power | IMU, OAK-D Lite, 2 range sensors; four 6S packs, charger | 1 each | $374; $1,899 | 18 Nov 2026 |
| Structure | CNC frame, soles; linkages; PA11 shells (estimates) | 1 each | $2,460; $320; $1,370 | 18 Nov 2026 |

Also in the total: a wired stop chain $941, an uncertified radio stop $250 (estimate), spares $857, cooling $90, wiring
$350, bench gear $2,229 and tools $350. With half the frame cost fixed, parts are $17,476 and landed $20,944–$22,988,
which a later review treats as realistic. Across both frame cases the landed estimate is $20.1k–$23.0k. Detail:
[60 cm parts list][parts60].

## 75 cm Triceratops lists

On 18 Nov 2026 a kickoff parts list priced two builds on the 75 cm body of [PR #669][pr669]: a recommended 20 kg build
(T20 in the list) and a 15 kg build (T15). Neither was adopted: the 60 cm build was chosen the same day to cut cost. Key
parts of the 20 kg build:

| Subsystem | Part | Qty | Unit price | Price date |
|---|---|---|---|---|
| Leg drives | RMD-X4-P36-36-E-L; RMD-X6-P20-60-E; RMD-X8-P20-150-E | 4; 6; 2 | $269; $428; $725 | 7 Nov 2026 |
| Head and tail | ROBOTIS XM540-W270-R and -W150-R; XM430-W350-R | 2; 3 | $494.39; $333.39 | 18 Nov 2026 |
| Compute and bus | Jetson kit; Teensy 4.1 carrier (estimate); PCAN-USB FD | 1; 2; 1 | $399; $150; $379 | 18 Nov 2026 |
| Sensors | Xsens MTi-3 IMU; OAK-D Pro W; TE FX29 foot load cells | 1; 1; 4 | $417.06; $529; $43.57 | 18 Nov 2026 |
| Battery, safety | 6S pack; certified wireless stop; relay | 4; 1; 1 | $107.19; $1,900; $611.99 | 7 Nov; 18 Nov 2026 |
| Structure | CNC 6061-T6 frame; PA11 shells; paint (estimates) | 1 each | $3,400; $1,950; $1,200 | 18 Nov 2026 |

20 kg build: parts $21,182 / $29,653 / $47,200; landed $22,908 / $37,280 / $66,386. 15 kg build (eight RMD-X4, four
RMD-X6, a pack monitor instead of the $440 BMS, a cheaper frame and shells): parts $18,466 / $26,374 / $42,696, landed
$19,971 / $32,827 / $58,486. Landed adds freight 3 / 5 / 6 %, duty 0 / 25 / 73 %, tax 0 / 7 / 10 % and contingency
5 / 7 / 10 %; the 73 % is a reseller's tariff-included price, so the high case is a ceiling.

## 75 cm T. rex

The T. rex study's lead body, the 75 cm Hybrid, was priced at catalog prices on 14 Nov 2026: $20,057 nominal turnkey.

| Subsystem | Part | Qty | Unit price | Price date |
|---|---|---|---|---|
| Leg drives | RMD-X6-P20-60-E (hip pitch); RMD-X4-P36-36-E-L (others) | 2; 10 | $428; $269 | 7 Nov 2026 |
| Head and tail | ROBOTIS XM540-W270-T (neck); Feetech STS3215-C018 (tail) | 1; 1 | $482.89; $22 | 7 Nov 2026 |
| Compute and bus | Pi 5 set or Jetson kit; Teensy 4.1 and CAN lot (estimate) | 1; 1 | $290 or $399; $180 | 7 Nov 2026 |
| Sensors, battery | OAK-D Lite; VL53L5CX; G-Tech 6S 3300 mAh | 1; 2; 1 | $269; $19.95; $107.19 | 7 Nov 2026 |
| Structure | frame, covers, fit-up lots (nominal, scaled estimates) | 1 each | $4,503; $1,891; $2,625 | 14 Nov 2026 |

Turnkey $14,228 / $20,057 / $28,215 with the Pi 5 set, $14,343 / $20,174 / $28,333 with the Jetson kit; self-assembled
(Pi 5) $12,532 / $17,244 / $24,076. PR #668 gave $20,030 at the old $249 Jetson price (superseded; its walking
results are unverified).

## 60 cm T. rex picks

Two narrow 60 cm bodies, Tapered and Hybrid, were picked on 8 Nov 2026. Both use the parts below; the Tapered body was
kept for further simulation on 11 Nov 2026.

| Subsystem | Part | Qty | Unit price | Price date |
|---|---|---|---|---|
| Leg drives | CubeMars AK45-36; RMD-X4-P36-36-E-L; RMD-X4-P12-10-E | 4; 4; 4 | $185.90; $269; $285 | 7 Nov 2026 |
| Head and tail | Feetech STS3250 (neck); STS3215-C018 (tail) | 1; 1 | $89; $22 | 7 Nov 2026 |
| Compute and bus | Pi 5 + AI HAT+ set; Teensy 4.1 and CAN lot (estimate) | 1; 1 | $290; $180 | 7 Nov 2026 |
| Sensors, battery | OAK-D Lite; VL53L5CX; G-Tech 6S 3300 mAh | 1; 2; 1 | $269; $19.95; $107.19 | 7 Nov 2026 |
| Structure | frame, covers, fit-up lots (Tapered; estimates) | 1 each | $3,446; $1,200; $2,258 | 8 Nov 2026 |

Turnkey: Tapered $10,801 / $14,809 / $20,544; Hybrid $10,821 / $14,848 / $20,622. The ankle drive is listed as
discontinued. An earlier 60 cm platform (PRs #662 and #663: twelve RMD-X4-P36-36-E-L, a Jetson) was put at
$16,117–$16,447 nominal, $21,817 high; the cost model scales its lots from it. Detail: [round-two review][round2].

## October 2026 T. rex concepts

[PR #635][pr635] sketched mini, medium and one-metre T. rex concepts in October 2026. Only the one-metre concept carries
a priced list: 18 RobStride drives and $17,286.15 for a first prototype.

| Subsystem | Part | Qty | Unit price | Price date |
|---|---|---|---|---|
| Leg drives | RobStride 03; RobStride 04 (hip pitch, knees) | 8; 4 | $225; $255 | 25 Oct 2026 |
| Head, tail, arms, jaw | RobStride 00 (head, tail); RobStride 05 (arms, jaw) | 3; 3 | $125; $110 | 25 Oct 2026 |
| Compute and bus | Jetson kit (old price); Teensy; CAN channels (allowances) | 1 each | $249; $40; $100 | 25 Oct 2026 |
| Sensors, battery | OAK-D Lite; IMU; load cells; 12S Li-ion pack | 1; 1; 8; 1 | $269; $28.15; $45; $700 | 25 Oct 2026 |
| Structure | aluminium frame and leg links; printed shells (allowances) | 1 each | $2,200; $850 | 25 Oct 2026 |

Robot parts $11,521.15; spares, stand, fixtures and bench tools ($2,265) and about 25 % contingency ($3,500) make
$17,286.15, before tax, freight, duty and labour. The estimate was not re-checked.

## Compsognathus walker

A walking biped with seven powered joints, priced at $954 on 19 Jul 2026, before any model existed. The later revision B
robot model (September 2026) has no cost estimate.

| Subsystem | Part | Qty | Unit price | Price date |
|---|---|---|---|---|
| Servos (legs, tail) | Feetech STS3250 (12 V); STS3215 (7.4 V) | 4; 3 | $55; $15.99 | 19 Jul 2026 |
| Compute and bus | Raspberry Pi 5 8 GB, cooler; servo driver; ESP32-S3 | 1 each | $185.95; $9.99; $15 | 19 Jul 2026 |
| Sensors, battery | BNO085 IMU; foot switches; 4S 3000 mAh LiPo set | 1; 2; 1 | $24.95; $2.95; $77.39 | 19 Jul 2026 |
| Structure | PETG filament; bearings, fasteners and shafts | 1 each | $55; $111.60 | 19 Jul 2026 |

Total $829.74 of parts plus 15 %, $954; tiers about $480, $954 and $2,540. By 7 Nov 2026 the STS3250 listed at $89. A
running version was put at $4k–6k. Detail: [BOM][bom], [revision B record](../reviews/COMPSOGNATHUS_ROBOT_DESIGN.md).

## Velociraptor walker

A walking-only biped sized from the 13.5 kg simulation model on 19 Jul 2026, cut to 10 motors on one CAN bus: about
$8,665.

| Subsystem | Part | Qty | Unit price | Price date |
|---|---|---|---|---|
| Leg, tail drives | RMD-X10-S2 (hips, estimate); RMD-X6-60; AK70-10 | 2; 6; 2 | ~$720; $490; $499 | 19 Jul 2026 |
| Compute and bus | Pi 5 + Teensy 4.1; CANable Pro, power board, ESP32 | 1 each | $220; $108 | 19 Jul 2026 |
| Sensors, power | IMU; encoders; 4 foot sensors; 6S pack | 1; 8; 1; 1 | $24.95; $19.40; $28.42; $197.34 | 19 Jul 2026 |
| Structure | PETG, PA-CF filament; carbon tubes, bearings, fasteners | 1 each | ~$300–500; ~$545 | 19 Jul 2026 |

About $7,220 of parts ($5,378 of it actuators) plus 20 %; tiers about $5,615 / $8,665 / $12,340. It was never redone
after the simulation model was rebuilt at 15.0 kg on 8 Nov 2026. Detail: [BOM][bom].

## Raptor running robots

Two paper cases, priced 24 Jul 2026 (why size decides the actuators: [SIZES.md](SIZES.md)). At full size custom
actuators are needed: about $36,000. At 0.72 scale catalog drives fit, for about $6,900.

| Subsystem | Part | Qty | Unit price | Price date |
|---|---|---|---|---|
| Full: hip pitch, knee | custom dual-motor quasi-direct-drive module | 4 | about $2,900 each to build | 24 Jul 2026 |
| Full: hip roll, ankle; tail | CubeMars AKE90-8; CubeMars AK80-9 | 4; 2 | not itemised | 24 Jul 2026 |
| Scaled: hip pitch; knee | MyActuator RMD-X6-60; RMD-X4-36 | 2; 2 | not itemised | 24 Jul 2026 |
| Scaled: roll; ankle; tail | CubeMars AK70-10; SteadyWin GIM8108-8; AK60-6 | 2; 2; 1 | not itemised | 24 Jul 2026 |

Full: actuators ~$14,500; frame, power, compute, sensing ~$11,200; spares, contingency ~$10,300; plus $15k–40k of
engineering. Scaled: actuators ~$3,180, rest ~$2,380, plus 25 %; not independently verified. Detail: [study][rsa].

## Psittacosaurus on Pupper V3

The sub-$5k tier of 24 Jul 2026: a juvenile Psittacosaurus shell on the open Pupper V3 quadruped, with passive,
removable head, torso and tail: about $4,700, parts only, labour and tools excluded.

| Subsystem | Part | Qty | Unit price | Price date |
|---|---|---|---|---|
| Robot | Pupper V3, official parts list (minimum-price column $1,785.91) | 1 | $2,158.91 | 24 Jul 2026 |
| In the robot | 12 SteadyWin GIM4305; Raspberry Pi 5 8 GB; BNO086 IMU; camera | 1 set | drives $1,320 | 24 Jul 2026 |
| Shell | passive head, torso and tail skins (allowance) | 1 set | $400 | 24 Jul 2026 |
| Other | spares $450, safety $350, freight and tax $550, contingency $800 | 1 each | $2,150 | 24 Jul 2026 |

Range about $4.2k–4.8k. Listings that day: a parts bundle at $2,350 (backordered; no printed parts, batteries or
controller) and a fuller kit at $3,400. Detail: [raptor scaling study][rsa].

## Dibothrosuchus-inspired quadruped

The near-$10k tier of 24 Jul 2026: a 12-joint quadruped on catalog CubeMars AK80-9 drives, with a 2 m/s acceptance
target. Expect $9k–11k; freight, tax, CNC structure or a spare-drive set can push it to $11k–13k.

| Subsystem | Part | Qty | Unit price | Price date |
|---|---|---|---|---|
| Leg drives | CubeMars AK80-9 V3 (price without driver board) | 12 | $479.90 | 24 Jul 2026 |
| Transmission | four remote-knee belt drives, joint encoders, leg fixture | 1 lot | $500 allowance | 24 Jul 2026 |
| Compute, bus, sensors | Jetson kit ($249 then), Teensy 4.1; OAK-D Lite, BNO085 | 1 set | $450 + $350 | 24 Jul 2026 |
| Battery and power | G-Tech 12S 3.5 Ah ($189.99), charger, contactor, fuse | 1 set | $900 allowance | 24 Jul 2026 |
| Structure | frame; shell | 1 each | $600; $250 | 24 Jul 2026 |

$10,000 with e-stop and harness $250, breakage $450 and freight and contingency $491. With the driver board the AK80-9
was $579.90 on 7 Nov 2026. A simulated chassis study (25 Jul 2026) chose twelve AK80-9 over a mixed set with four AK60-6
($4,759), for about $1,000 more. Detail: [raptor scaling study][rsa], [chassis study][chassis].

## Parts considered across the builds

| Part | List price (date) | Used or considered in |
|---|---|---|
| MyActuator RMD-X4-P36-36-E-L | $269 (7 and 18 Nov 2026) | T. rex and Triceratops legs |
| MyActuator RMD-X6-P20-60-E | $428 (7 and 18 Nov 2026); $490 in Jul 2026 | 75 cm T. rex, Triceratops, Velociraptor |
| MyActuator RMD-X8-150 / RMD-X12-320 | $725 / $1,285 (7 Nov 2026) | 75 cm Triceratops elbows; 100–120 cm T. rex hips |
| RobStride 00 / 03 / 04 / 05 / 06 | $125 / $225 / $255 / $110 / $210 (Oct, Nov 2026) | 1 m concept; 60–120 cm T. rex |
| CubeMars AK45-36 | $185.90 (7 Nov 2026); 15–28 V only | 60 cm T. rex hips |
| CubeMars AK70-10 / AK80-9 / AK60-6 | $498.90 / $579.90 / $298.90 with driver (7 Nov 2026) | raptors, Dibothrosuchus |
| Feetech STS3250 / STS3215-C018 | $89 ($55 in Jul 2026) / $22 (7 Nov 2026) | Compsognathus; T. rex neck, tail |
| ROBOTIS XM540-W270-T | $482.89 (7 Nov 2026); RS-485 version $494.39 (18 Nov 2026) | 75 cm T. rex, Triceratops necks |
| ROBOTIS XC430-W240 / XL430-W250 | about $100–197 / $45–70 (18 Nov 2026) | 60 cm Triceratops head, tail |
| Raspberry Pi 5 8 GB + AI HAT+ set | $290 list, about $323 street (7 Nov 2026) | T. rex 60 and 75 cm (adopted) |
| Jetson Orin Nano Super dev kit | $399 (7 Nov 2026); $249 before | 75 cm T. rex option; Triceratops |
| PEAK PCAN-USB FD (USB-CAN) | $368 (7 Nov 2026); $379 (18 Nov 2026) | benches; Jetson-direct layouts |
| Gens ace G-Tech 6S 3300 mAh LiPo | $107.19 (7 Nov 2026) | T. rex and Triceratops packs |

Where sources disagree, the later catalog price comes first. Detail: [actuator catalog][catalog].

## First purchases planned

Every plan buys a bench set before a whole robot. All are planned; nothing has been bought.

| Plan | Bench set | Cost | Date |
|---|---|---|---|
| 75 cm T. rex bench stage | one leg drive of each type, tail servo, microcontroller | ~$1,260–$2,000 | 16 Nov 2026 |
| T. rex delay bench, run 1 | one drive per type, two USB-CAN adapters ($736) | $1,998; ~$2,431 in all | 14 Nov 2026 |
| T. rex delay bench, run 2 | all 12 leg drives ($3,546) and the same tools | $4,867; ~$5,300 in all | 14 Nov 2026 |
| Foreleg bench, 20 kg design | one foreleg's drives, test gear, certified stop | ~$7.1k; $8.9k landed | 18 Nov 2026 |
| Same, radio stop | as above with a cheaper stop; to be redone for 60 cm | ~$4.6k (+$2.3k–3.4k tests) | 18 Nov 2026 |

A legs-only T. rex stage would then add the 12 leg drives ($3,546; structure not priced). The kickoff plan proposed a
first Triceratops order (the 20 kg design's foreleg set and long-lead parts) for 8 Jan 2027, as a planning date; it was
not adopted.

[pr669]: https://github.com/kuds/mesozoic-labs/pull/669
[pr635]: https://github.com/kuds/mesozoic-labs/pull/635
[parts60]: https://github.com/kuds/mesozoic-labs/blob/7bfeef5334d0d7484e5428f1b77d2d1847d35c8c/hardware/triceratops_build/design/PARTS_60CM.md
[round2]: https://github.com/kuds/mesozoic-labs/blob/87bafc864fefb16b040bd7a997fea8273bfc6649/hardware/trex_form_study/ROUND2_REVIEW.md
[catalog]: https://github.com/kuds/mesozoic-labs/blob/87bafc864fefb16b040bd7a997fea8273bfc6649/hardware/trex_form_study/designlab/data/actuators.json
[bom]: https://github.com/kuds/mesozoic-labs/blob/87bafc864fefb16b040bd7a997fea8273bfc6649/docs/hardware/HARDWARE_BOM.md
[rsa]: https://github.com/kuds/mesozoic-labs/blob/87bafc864fefb16b040bd7a997fea8273bfc6649/docs/hardware/RAPTOR_SCALING_AND_ALTERNATIVES.md
[chassis]: https://github.com/kuds/mesozoic-labs/blob/87bafc864fefb16b040bd7a997fea8273bfc6649/environments/shared/scripts/hw_chassis_study.py
