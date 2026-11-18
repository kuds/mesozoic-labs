# Triceratops quadruped: premium parts list (15 kg and 20 kg)

18 Nov 2026. A planning estimate for the owner and the builder. Nothing has been bought or quoted. **Reviewed and corrected the same day by an independent check; each correction is marked [F1]-[F13] and listed under "Review corrections".** It starts from PR #669 (head 84a13e1e) and from the T. rex 75 cm Jetson budget.

**Note on committing (18 Nov 2026).** This file is committed in `hardware/triceratops_kickoff/` beside `CONSOLIDATION_PLAN.md` and `NEW_SESSION_PROMPT.md`. The independent check's verdict was *sound after fixes*: its fixes are applied below and marked [F1]-[F13], so no open issue from that check remains (the questions still to settle on the bench are in section 8). On committing, only editorial changes were made and no figure was changed: this note, which replaces a sentence saying the list was not in the repository; and the script references in sections 2 and 2.8 and the file list at the end, which named paths in the working session's scratchpad. The scripts themselves (`sizing.py`, `headtail.py`, `massbudget.py`, `bom.py`, and the review's `check_sizing.py`, `check_bom.py`, `fix_bom.py` and `apply_fixes.py`) were not committed; their outputs are in Appendices A to C. Section 7's "Suggested timing" for the T. rex (restart its simulation about Q2 2027; buy its hardware after the Q3 2027 demo) is replaced by `CONSOLIDATION_PLAN.md`, section 5.3, which follows [F13]; section 7 is left as the checked document had it. Nothing here is adopted: the owner decides (see `CONSOLIDATION_PLAN.md`, section 7).

## Summary

**Recommended build: T20, a 20 kg robot.** It keeps PR #669's body: 75 cm to the frill crown, 1.50 m long, legs 235 + 230 mm, shoulders and hips 400 mm above the floor. Each leg has three catalog MyActuator drives on a 48 V bus:
- **Roll:** RMD-X6-P20-60.
- **Upper-leg pitch:** RMD-X4-P36-36-L.
- **Elbow or knee flexion:** RMD-X8-P20-150 on the front legs (1:1 linkage); RMD-X6-P20-60 on the hind legs (1.3:1 linkage).

The head carries three ROBOTIS DYNAMIXEL servos (neck pitch, neck yaw, beak) plus eyelids and frill lighting. The tail has two servos. Compute is a Jetson Orin Nano Super with a real-time Teensy 4.1 controller on four CAN buses, one per leg. Sensing is an OAK-D Pro W camera, an Xsens IMU and a TE load cell in each foot. Safety comes from a PL d wireless stop and a Pilz safety relay that sits the robot down before it cuts power. Power is a 12S battery (two 6S packs in series), with a tether, an automatic switchover and an EV-grade battery management system (BMS).

- **Gait it is sized for:** the PR's 0.30 m/s walk, and a 0.60 m/s trot later (short bouts until the bench heat test). **[F1]** The PR's 0.40 m/s stretch walk stays out of reach with X4 pitch drives on either bus: resellers quoting MyActuator say a drive run above its rated voltage must not exceed its rated speed, which is 83 rpm for the 24 V X4, and the 0.40 m/s swing needs 75-77 rpm at the pitch joint.
- **Mass:** 18.8 kg of listed parts plus a 1.2 kg reserve, closing at 20.0 kg. About 55 % of the weight sits on the front feet. The wireless-stop receiver's mass is not published and must come out of the reserve. **[F3]**
- **All-in cost, self-assembled, landed** (freight, import duty and sales tax counted separately): **low $22.9k, nominal $37.3k, high $66.4k.** **[F12]** (The unreviewed list said $21.9k / $34.5k / $60.3k.)
  - Measured the T. rex way (hardware plus 5.3 / 7.2 / 8.4 % to cover freight, duty and tax), it is **$22.3k, $31.8k and $51.2k**. On that basis T20 sits at the top of the brief's $17k-$32k class; on the landed basis its nominal is above it. **[F12]**
  - The $2.3k nominal rise comes from items the unreviewed list lacked or under-priced: a machine-side receiver for the wireless stop if the $1,900 kit lacks one (+$800, estimate), cooling, tools and a transport case (+$1,240, estimates), current ROBOTIS prices (+$99), foot-sensor links and a BMS current sensor (+$180, estimates).
  - The high figure sets every line to its high price at once, including a reseller's tariff-included drive price that is 73 % above the US list (it carries the reseller's margin as well as duty). Treat it as a ceiling, not a forecast.
  - $2.2k of the nominal total is bench gear that the T. rex will reuse.
- **First purchase:** a single-leg bench set, about **$7.1k of parts** (one foreleg's drives $1,422, its linkage $120, torque rig $700, bench supply $400, thermal camera $250, PCAN-USB FD $379, two MCU carriers $300, and the full stop chain $3,526 including the receiver estimate), or about **$8.9k landed** by section 6.1's nominal rules. **[F3] [F12]** Buy it before anything else, following the PR's own gate: one foreleg on the bench before ordering four legs.

**Why it is not a knockoff:**
- **Leg drives:** every leg axis is a catalog planetary servo actuator (19.6:1 or 36:1, so geared, not quasi-direct-drive) **[F12]** with a datasheet, dual encoders, and CAN plus EtherCAT. They come from the same MyActuator family that the T. rex specified. Each axis is sized below from first principles, for torque, heat and speed, and an independent re-derivation reproduces the figures within 8 % (the list's figures are the conservative ones).
- **No hobby servos:** the Feetech STS3215 is gone. The head and tail use ROBOTIS DYNAMIXEL X-series servos in their RS-485 versions; the two small eyelid servos are 5 V TTL XL330s on their own rail. **[F6]**
- **Safety and power:** industrial parts throughout. A PL d wireless stop, a Pilz safety relay, 2-NC mushroom e-stops **[F4]**, a bidirectional DC contactor, and an Orion BMS. The chain as a whole is only as strong as its weakest part, and the custom watchdog board is not a safety-rated part, so do not call the robot PL d until the chain is assessed. **[F3]**
- **Sensors:** named, catalog parts, not "TBD". TE FX29 load cells, an Xsens MTi IMU, and a Luxonis active-IR stereo camera.
- **Build quality:** a CNC-machined 6061-T6 frame, and MJF PA11 shells with a painted museum finish.

**Alternative: T15, a 15 kg robot.** It uses PR #669's own drive set (8 × X4 and 4 × X6) but adds a 1.3:1 linkage on the flexion joint and moves to 48 V.
- **Cost:** $20.0k / $32.8k / $58.5k landed, or $19.4k / $28.3k / $46.3k the T. rex way. **[F12]**
- **Mass reserve:** with the premium head, safety and power parts, it closes at 15.0 kg with only 0.39 kg to spare (the PR had 1.84 kg), before the stop receiver's unpublished mass. **[F3]**
- **No sustained trot:** its X4 roll drives run hot in a trot (101 % of their equilibrium torque at 55 % front load, 111 % at 60 %; the review's own model gives 102 % and 111 %), so trots must be short.

That is why T20 is the recommendation.

## Review corrections (18 Nov 2026)

An independent reviewer re-derived the sizing with a separate script (`parts_check/check_sizing.py`, Appendix C), re-added every BOM line and total (`check_bom.py`: the arithmetic as written was right), and re-checked the actuators, compute, battery and every line over $150 against web-search results read on 18 Nov 2026. The maker pages myactuator.us and openelab.com could not be opened from this session (outside its allowed network), so MyActuator figures still rest on the repo catalog (maker sheets read 7 Nov 2026) and on reseller search summaries.

| Fix | What was wrong | Corrected value |
|---|---|---|
| F1 | The 12S case claimed about 194 rpm no-load for the 24 V X4 at 42 V, so that "the 0.40 m/s walk and a later trot become possible". Resellers quoting MyActuator (search summary, 18 Nov 2026) say that above rated voltage the drive must not exceed its rated speed: 83 rpm for the X4-36. | X4 speed cap 83 rpm on 12S. Pitch uses 0.70 of it at 0.30 m/s (0.83 with the PR's own swing profile), 0.91-0.93 at 0.40 m/s, 0.52 in the 0.60 m/s trot; T15 roll 0.36. **The 0.40 m/s walk is not credible with X4 pitch drives on either bus.** 12S stays required, but for the 48 V X6 and X8, not for the X4. Same caveat for the T. rex if it moves to 12S. |
| F2 | Pushrod pulls were given at a 90° transmission angle; the list itself allows angles down to 40°. | At 40° (from the list's own joint torques): T20 1.59 kN design, 1.82 kN pad-edge abuse; T15 1.16 / 1.33 kN. The review's slightly lower torques give 1.55 / 1.79 and 1.13 / 1.30 kN. Steel rod ends (not polymer) rated at least 2× these; crank pins and the output bearing pair sized to the same. |
| F3 | The $1,900 Safe-D-Stop listing (PL d per Cattron) does not say it includes a DC machine-side receiver. Cattron's machine-side unit for it is the XBMCU (9-36 V DC, Cat 3 / PL d inputs, M12, CAN). The list also called the whole chain PL d. | Receiver line added: $0 / $800 / $1,500 (estimate; quote), its mass to come out of the reserve. The system is not PL d until assessed: the custom watchdog is not a safety-rated part. |
| F4 | Schneider XB4BS8445 is a 1 NO + 1 NC e-stop (Schneider datasheet, per search); the PNOZ input is dual-channel. | **XB4BS8444** (2 NC, screw terminals), price assumed equal (not checked). |
| F5 | FX29K0 foot cells are I2C parts (2.7-5.5 V). I2C is a board-level bus; four identical cells (address not confirmed, likely fixed) on leg-length leads beside 48 V motor cables will not work as one bus. Lead time is 21-23 weeks at distributors. The 2.5× overload figure was found only for the 25 lbf sibling. | Per-leg differential I2C extender pair (PCA9615 class) or a small per-foot CAN node: +$80 (estimate). Order the cells with the phase-1 set. Confirm overload and address in TE's datasheet. |
| F6 | XL330-M288-T eyelid servos are 3.7-6 V (5 V recommended) TTL servos; the list put "the servos" on a 12 V RS-485 bus. | Own 5 V rail (+$20, estimate) and the U2D2's TTL port; price $27.49 (robotis.us, sold out), not $24. |
| F7 | ROBOTIS -R prices were assumed equal to -T. | XM540-W270-R and XM540-W150-R $494.39 (not $482.89); XM430-W350-R $333.39 (not $310.39). robotis.us lists the W270-R and W350-R sold out, the W270-R with an extended lead time. |
| F8 | Jetson low case $249 and Orin NX high case $1,099.99 predate NVIDIA's July 2026 price rise (Orin Nano Super kit $249 -> $399; kits out of stock). | Low $399 (= nominal); high $1,364.29 (RobotShop reComputer J4012, "tariffs included"; listing undated, currency not confirmed; Seeed's Robotics J4012 is $1,439, sold out). |
| F9 | Small price errors. | Pololu D42V110F12 $59.95 (not $69.95; rationed). PCAN-USB FD $379 at DigiKey (+$15 shipping), not $368. |
| F10 | Orion BMS Jr 2: $440 MSRP confirmed, but its current sensor is a separate option that Orion calls strongly recommended, and its charge/discharge outputs are logic signals, not contactor drivers. | Current-sensor line added (+$100, estimate). Wire its discharge-enable into the stop chain's watchdog, not straight to the contactor coil. |
| F11 | No cooling, tools or transport case for a museum robot that travels. | Cooling (fans, ducts, filters) $90, tools $350, padded road case $800 (all estimates). |
| F12 | Totals and claims that the above change; "quasi-direct-drive" wording; the 73 % "duty" case is a reseller price. | T20 $22.9k / $37.3k / $66.4k landed ($22.3k / $31.8k / $51.2k T. rex way); T15 $20.0k / $32.8k / $58.5k ($19.4k / $28.3k / $46.3k). Bench set $7.1k parts / $8.9k landed. Drives described as geared planetary actuators. |
| F13 | Section 7 buys T. rex hardware only after the Q3 2027 demo, which leaves one or two quarters before the end-2027 / early-2028 flagship milestone, with these drives already backordered at resellers. | If that milestone is a walking T. rex, order its long-lead drives in Q2 2027 and resume its simulation in Q1 2027. The lead's call. |

Checked and found sound: the leg-load arithmetic (the review reproduces every peak and heat figure within 8 %, always with the list on the conservative side, mainly because it applies the fore-aft force at floor height rather than at the ankle pin); the duty factors and dynamic factors as stated; the trot half-sine peak f·W·π/(4β); the three-foot lone-forefoot load; the X6/X8 speeds on 12S; the thermal comparisons against the maker's equilibrium tables; the mass-budget sums (14.61 / 18.83 kg listed, 15.00 / 20.00 kg closed); every BOM line total and category sum; run time (117 Wh usable over 200-280 W = 25-35 min); 73 Wh per pack. Checks the list did not make and that do not govern: the pitch torque to swing the leg (about 6-7 N·m) and the stand-up from a belly crouch with four feet down (at most 17.7 N·m at the front elbow at 20 kg, below the 31-32 N·m walk peak).

## 1. What changed from PR #669, and why

| Area | PR #669 | This list | Why |
|---|---|---|---|
| Body geometry, materials, layout | 75 cm crown, 1.50 m, stance 650 / track 350 mm, joints at 400 mm, links 235 / 230 mm, 110 × 80 mm pads; 6061 rails, MJF PA11, urethane soles; dorsal hatch, removable limbs | **Kept** | Consistent across the PDF, OpenSCAD and MuJoCo files; the load path and service access are sound. |
| Motor bus | 6S (21.0–25.2 V) | **12S, 48 V class** (42.0–49.2 V in use; charge ceiling 4.10 V/cell) | The X6 and X8 are 48 V drives: at 6S their no-load speed falls to about 77 and 69 rpm, which rules out the hind 1.3:1 linkage and the front X8 at 1:1. Bus current halves. 50.4 V full charge is inside the drives' 20–55 V window, with a 52.5 V regen clamp. **[F1]** It does not speed up the X4: above its 24 V rating the drive is capped at its 83 rpm rated speed (reseller text quoting MyActuator; confirm), so the 0.40 m/s walk stays out of reach; the 0.30 m/s walk (58–60 rpm) and the 0.60 m/s trot (43–45 rpm) fit. |
| Flexion drive | X6-60 through a 1:1 belt, 12 mm wide, 20 mm pulleys | **T20:** X8-150 1:1 (front), X6-60 1.3:1 (hind). **T15:** X6-60 1.3:1 on all four | At 1:1 the X6 is over its heat limit: 113 % of the equilibrium torque at 15 kg, 150 % at 20 kg. The PR's 24.2 N·m peak holds, but a held paw pose and the trot do not. |
| Flexion transmission | 12 mm belt, "tooth profile TBD" | **Rigid four-bar pushrod** with 35–36 mm cranks and an outboard bearing pair. Fallback: an 8 mm-pitch, high-modulus belt sized from the maker's table | Design pull at a 90° transmission angle is 0.75 kN (T15) and 1.02 kN (T20), and 1.17 kN in the pad-edge abuse case; **at the 40° minimum transmission angle it is 1.16 kN (T15) and 1.59 kN (T20), and 1.82 kN in abuse [F2]**. A rod has no stretch, no tensioner and no belt wear. A belt at a 20 mm radius would see 1.3–1.8 kN. |
| Roll drive | X4 | **T20: X6-60.** T15: X4 | At 20 kg the trot needs 25.4 N·m of roll torque against the X4's 21 N·m design limit, with heat at 135 %. |
| Pitch drive | X4 | **X4, kept** | Passes the walk at both masses. At 20 kg it is thin in a trot (heat 0.91–0.98 at 55 % front, 0.99–1.07 at 60 %), and on 12S its speed is capped at 83 rpm **[F1]**. |
| Bus | 2 classic CAN × 6 drives | **4 classic CAN × 3 drives (one per leg)**, 500 Hz baseline; 1 kHz trial (67–81 % load); EtherCAT trial on the -E drives | The PR's 2 × 6 layout runs 67–81 % at 500 Hz and does not fit 1 kHz. One bus per leg also isolates faults. |
| Compute | Pi 5 + AI HAT+ (Jetson as "studied alternative") | **Jetson Orin Nano Super 8 GB + NVMe** (Orin NX 16 GB in the high case) | Same platform as the T. rex and the PLEM pitch. Fixes the PDF's "Orin / MCU" label. Compute is not the bottleneck. |
| Real-time MCU | Teensy 4.1, 2 CAN | **Teensy 4.1 + MCP2518FD (4th bus), 4 isolated transceivers**, 2 boards (1 spare) | Four buses; isolation against ground bounce from 12 drives. |
| Camera | OAK-D Lite | **OAK-D Pro W** (active IR, wide FOV) | Dim museum halls and low-texture floors; the survey flagged the Lite as the low-tier choice. |
| Chest ToF | 2 × VL53L5CX | **2 × VL53L8CX** | +$5 each; better range in ambient light (repo catalog: 2.8 m vs 1.7 m at 5,000 lux). |
| IMU | unnamed | **Xsens MTi-3 (+ BNO085 backup)** | A calibrated AHRS with a datasheet. |
| Foot load | "4 foot-load sensors", unnamed | **TE FX29K0-100A-0100-L** (I2C, 445 N) in sole cartridges with hard stops, each on a per-leg differential I2C link **[F5]** | Peak trot foot load is 212 N at 20 kg, including the 1.25 dynamic factor. |
| Neck, head | XM540-W270 pitch ±10°, spring-assisted; jaw and yaw fixed | **XM540-W270-R pitch ±20° (spring-assisted), XM540-W150-R yaw, XM430-W350-R beak**; T20 adds 2 × XL330 eyelids (5 V TTL, own rail **[F6]**); frill and eye LEDs | "Look at a person" needs yaw; a beak and blink sell the animal. RS-485 versions for long service loops. |
| Tail | Feetech STS3215 yaw | **XM430-W350-R yaw** (T20: + pitch) | Removes the hobby servo. |
| Stop circuit | "independent stop circuit" | **PL d wireless stop (with a DC machine-side receiver [F3]) + 2 latching 2-NC mushroom e-stops [F4] + watchdog MCU → Pilz PNOZ s5** (instant contacts command a damped sit; delayed contacts open the contactor 0.5–1.5 s later) | A Cat 1 stop: the robot sits down before power drops instead of collapsing. W-RC-style hardware independence. The chain is not PL d as a whole until assessed **[F3]**. |
| Power path | fuse, disconnect | **100 A / 58 V main fuse, bidirectional contactor, precharge, 4 leg fuses, 2 × LTC4359 ideal-diode OR with tether priority, 52.5 V active clamp** | The deck's LTC4417 is a 2.5–36 V part, so it cannot run on 12S. The LTC4359 covers 4–80 V. |
| Battery | 6S 5 Ah (0.70 kg budgeted; the repo's pack is 0.755 kg and 165 mm long) | **2 × Gens ace G-Tech 6S 3300 in series** (12S, 146.5 Wh, 1.012 kg); 2 sets | The T. rex's own pack SKU. Each pack is 73 Wh, under the 100 Wh airline spare-battery line. |
| BMS | none | **T20: Orion BMS Jr 2** (cell V/T, current through a separate sensor **[F10]**, charge/discharge-enable logic outputs into the stop chain **[F10]**, CAN); T15: cell alarm + temperature + CAN power monitor | A museum robot needs automatic cut-off, not only an alarm. |
| Tether | not in the PR (in the deck) | **Mean Well RSP-1000-48**, 5 m lead, pendant e-stop | Most museum hours run on the tether; the battery covers show segments. |
| Mass reserve | 1.84 kg | **T20 1.17 kg; T15 0.39 kg** | The premium items spend most of the PR's reserve. |
| Costing | drives only ($3,864) | **Full BOM** | |

## 2. Sizing the leg drives from first principles

Script: `sizing.py` (in the working session's scratchpad; not committed). Its full output is in Appendix A.

### 2.1 Inputs and assumptions

**Geometry (PR #669 robot.xml):**
- Shoulder and hip pitch axes at 400 mm; ankle pin at 45 mm; links 235 / 230 mm.
- Stance 650 mm; track 350 mm.
- Each leg plane sits 80 mm outboard of its roll axis (roll at y = ±95 mm, foot at ±175 mm).
- The elbow points aft and the stifle points forward; their lever arms mirror each other.

**Load model:**
- The ankle is passive, so the vertical foot force acts under the ankle pin.
- Flexion torque = Fz × (horizontal elbow-to-foot distance) + |Fx| × (elbow height).
- Pitch torque = Fz × |x_foot| + |Fx| × 0.40 m.
- Roll torque = Fz × 0.080 m + |Fy| × 0.40 m.

**Front load:** 55 % of the weight on the front axle as the design point, with 50 % and 60 % as sensitivity cases.
- In static three-foot support, the lone forefoot carries the whole front-axle load.
- The as-built estimate is 55.2 % for T20 and 53.6 % for T15, with the battery 200 mm aft of centre (see section 6).

**Slow walk (the museum gait), from the PR:**
- Lateral-sequence walk, duty factor 0.75, 1.5 Hz, 0.30 m/s, 150 mm stance sweep, 32 mm lift.
- The worst case is the lone forefoot at touchdown, foot 75 mm ahead (the PR's conservative case).
- Fore-aft and lateral forces are 0.10 × Fz.
- **Dynamic factor 1.5** (the PR's allowance, kept).

**Trot (later):**
- 2.0 Hz with duty factors 0.50 (at 0.60 m/s) and 0.55 (at 0.50 m/s).
- Half-sine vertical force. Each diagonal pair carries the weight split by the front share, so the per-foot peak is f × W × π / (4β).
- Fore-aft force is 0.15 × peak × sin(2πs); lateral force is 0.10 × Fz.
- **Dynamic factor 1.25** on top.

**Continuous load:** the largest of these three:
- the walk RMS (front foot loaded at f·W/2 for two thirds of stance and f·W for the middle third);
- the trot RMS;
- a held three-leg pose with the lone forefoot under the shoulder (for example a "paw lift" for visitors).

**Pass rules:**
- Design peak ≤ **70 % of the maker's peak torque** (the 10 rpm figure, not stall). That gives at least 2.1× over the static walking load.
- Continuous ≤ the maker's **thermal-equilibrium torque** from the V4.1 temperature-rise tables: 90 min, motor reaching 100 °C.
  - X4-36-L: 7.5 N·m.
  - X6-50-L: 12 N·m at 0 rpm. This is used as a stand-in for the V4 X6-60, which has no thermal table.
  - X8-150: 34 N·m at 0 rpm.
- Peak speed ≤ **70 % of no-load speed at the 42 V minimum pack voltage**. No-load speed is scaled linearly with voltage (an estimate). This holds for the 48 V X6 and X8.
- **[F1]** For a drive run above its rated voltage (the 24 V X4 on 12S), the cap is its **rated speed, 83 rpm** (resellers quoting MyActuator: above rated voltage, do not exceed rated speed). X4 speed use is given against that cap with no further 0.7 factor, since the cap is already an operating limit.
- Flexion transmission efficiency 0.90.

### 2.2 Loads per axle

| Mass | Weight | Front axle (55 %) | Rear axle | Lone forefoot, walk | Trot peak per forefoot (β 0.50) |
|---|---:|---:|---:|---:|---:|
| 15 kg | 147.2 N | 80.9 N | 66.2 N | 80.9 N | 127.1 N |
| 20 kg | 196.2 N | 107.9 N | 88.3 N | 107.9 N | 169.5 N |

The lever arm about the elbow depends on where the foot is relative to the shoulder:

| Foot position | Elbow lever arm |
|---|---:|
| 75 mm behind (lift-off) | 105.4 mm |
| Under the shoulder | 150.2 mm |
| 25 mm ahead | 161.5 mm |
| 75 mm ahead (touchdown) | 179.1 mm |
| 100 mm ahead | 185.4 mm |

These match the PR and the survey.

### 2.3 Torque per axis (front leg, joint side, N·m)

| Axis | Case | 15 kg | 20 kg |
|---|---|---:|---:|
| Flexion | Walk, lone forefoot at touchdown × 1.5 | 24.0 | 32.1 |
| | Walk, steady window (foot ±25 mm) × 1.5 | 22.1 | 29.5 |
| | Abuse: centre of pressure at the pad edge after the ankle stop × 1.5 | 30.7 | 41.0 |
| | **Trot, β 0.50, 0.60 m/s × 1.25 (design peak)** | **26.9** | **35.8** |
| | Stand on 4 feet (continuous) | 6.1 | 8.1 |
| | Walk RMS | 8.0 | 10.6 |
| | Trot RMS | 11.3 | 15.1 |
| | **Held paw pose (continuous design)** | **12.2** | **16.2** |
| Pitch | **Walk touchdown × 1.5 (design peak)** | **14.0** | **18.6** |
| | Trot × 1.25 | 13.8 | 18.3 |
| | Continuous (trot RMS) | 5.5 | 7.4 |
| Roll | Walk × 1.5 | 14.6 | 19.4 |
| | **Trot × 1.25 (design peak)** | **19.1** | **25.4** |
| | Continuous (trot RMS; held pose 6.5 / 8.6) | 7.6 | 10.1 |

The hind leg carries 45 % of the weight and needs about 18 % less than the front.

**Cross-check against the PR:** its 24.2 N·m at the front motor is 14.5 × 1.5 / 0.9. My walk-touchdown figure is 24.0 N·m at the joint, about 26.7 N·m at the motor, because it also includes the fore-aft force term. That term adds about 2 N·m, as the survey said.

### 2.4 Swing speeds and the bus voltage

Peak joint speeds with a minimum-jerk foot path:

| Gait | Pitch (rpm) | Flexion (rpm) |
|---|---:|---:|
| Walk 0.30 m/s | 59.9 | 50.1 |
| Walk 0.40 m/s | 77.0 | 58.5 |
| Trot 0.60 m/s | 45.3 | 47.0 |

- The PR's 69.0 / 48.1 rpm came from a different swing profile.
- The trot's longer swing time makes it slower than the fast walk.

Drive no-load speeds at the minimum pack voltage (scaled from the maker's 24 V or 48 V figures):

| Drive | 6S at 21 V | 12S at 42 V |
|---|---:|---:|
| X4-36-L | 97 rpm | 194 rpm |
| X6-60 | 77 rpm | 154 rpm |

**[F1] (corrected)** The X4 row above is a no-load estimate only. On 12S the X4 is run above its 24 V rating, and resellers quoting MyActuator say it must then stay at or below its 83 rpm rated speed. Against that cap the pitch drive uses 0.72 at 0.30 m/s (0.83 with the PR's 69 rpm swing profile), 0.93 at 0.40 m/s and 0.55 in the 0.60 m/s trot. The review's independent swing model gives 58.1 / 75.4 / 43.3 rpm (pitch) and 46.4 / 54.5 / 43.3 rpm (flexion), so 0.70 / 0.91 / 0.52.

- **The 0.40 m/s walk is not credible with X4 pitch drives on either bus.** The museum walk is 0.30 m/s; the later trot fits.
- **12S is still required**, for the 48 V drives: at 6S the X6 hind flexion at 1.3:1 needs 76 rpm against about 77 rpm no-load, and the front X8 at 1:1 needs 58.5 rpm against about 69 rpm. On 12S they use 0.71 and 0.60 of their allowance.
- If a 0.40 m/s walk is wanted later, the pitch axis needs a 48 V drive (X6-60, +0.46 kg and +$159 each), which T20's reserve covers for the front pair only.

### 2.5 Drive candidates checked

Usage figures are for the front leg at 55 % front load. A value ≤ 1.00 passes.

| Drive | Rated / peak (stall) N·m | No-load rpm @ 48 V | kg | Interface, encoder | Heat data | Price and source | Verdict |
|---|---|---:|---:|---|---|---|---|
| MyActuator RMD-X4-P36-36-E-L (V4.1) | 10.5 / 30 (36) | 111 @ 24 V; **capped at 83 rpm (rated) when run above 24 V [F1]** | 0.36 | CAN + EtherCAT; dual 17/18-bit | equilibrium 7.5–8.5 N·m | $269 list, myactuator.us (repo catalog, read 7 Nov 2026). US resellers $400 backordered (aifitlab) and $620 tariff-included for the -E sibling (RobotShop), web search 18 Nov 2026 (re-checked by the review the same day) | **T15 roll and pitch; T20 pitch.** Roll at 20 kg fails (1.21 peak, 1.35 heat) |
| MyActuator RMD-X6-P20-60-E (V4) | 20 / 60 | 176 | 0.82 | CAN + EtherCAT; dual 17/17-bit | none for V4 (X6-50-L table used) | $428 list, myactuator.us (repo catalog, 7 Nov 2026); the store page shows X6-50-L figures (45 / 50 N·m, 0.78 kg). RobotShop $741.84 incl. tariffs; $490 backordered at two resellers (web search 18 Nov) | **T15 flexion at 1.3:1; T20 roll and hind flexion at 1.3:1.** Either variant passes those roles |
| MyActuator RMD-X6-P30-80-E (V4.1) | 25 / 80 (may be stall) | 117 | 0.81 | CAN + EtherCAT; dual 17/19-bit | none published | no price found (catalog only; web search 18 Nov found no listing) | **Ask for a quote and its heat table.** At 1:1 it passes T20 front flexion on peak (0.71) and speed (0.82); if it holds ≥ 18 N·m continuous it replaces the X8 and saves 1.0 kg |
| MyActuator RMD-X8-P20-150-E (V4.1) | 43 / 120 (150) | 158 | 1.33 | CAN + EtherCAT; dual 17/19-bit; hollow | equilibrium 34 / 38 / 36 / 20 N·m at 0 / 10 / 30 / 127 rpm | $725 list, myactuator.us, US SKU -03 (repo catalog, 7 Nov 2026); $645 backordered at aifitlab (web search 18 Nov) | **T20 front flexion, 1:1:** 0.47 peak, 0.53 heat, 0.60 speed |
| MyActuator RMD-X8-P20-120-E (V4) | 43 / 120 | 158 | 1.40 | CAN + EtherCAT | (X8-150 table as proxy) | $714 list (repo catalog, 7 Nov 2026) | Fallback for the X8-150; 70 g heavier |
| RobStride 02 | 6 / 17 | 410 | 0.405 | CAN (CAN FD per catalog); dual 14-bit | none | $145, Seeed (repo catalog, 7 Nov 2026) | Too small for roll or pitch (1.17–2.14) |
| RobStride 03 | 20 / 60 | 195 | 0.88 | as above | none | $225, Seeed datasheet updated 21 Oct 2026; EUR 245.95 at OpenELAB (web search 18 Nov) | Passes flexion at 1:1 (0.71 at 15 kg, 0.95 at 20 kg) and roll. Heavier than the X6, no heat data, a different protocol from the T. rex drives. **Budget alternative only** |
| RobStride 04 | 40 / 120 | 200 | 1.42 | as above | none | $255 (repo catalog, 7 Nov 2026) | Passes flexion; heaviest |
| CubeMars AK70-10 | 8.3 / 24.8 | 480 | 0.621 | CAN; single 14-bit | none | $498.90 with driver (repo catalog, 7 Nov 2026) | Roll fails (1.10–1.46) |
| CubeMars AK80-9 V3 | 9 / 22 | 570 | 0.49 | CAN | none | $579.90 with driver (repo catalog, 7 Nov 2026) | Roll fails (1.24–1.65) |
| CubeMars AK80-64 | 48 / 120 | 75 | 0.85 | CAN | none | $989.90 (repo catalog); $1,227.99 with driver at OpenELAB (web search 18 Nov) | Too slow at 1:1 (1.27 speed use); 64:1, back-drive 4.7 N·m, poor for falls |
| Unitree GO-M8010-6 | n/a | n/a | n/a | RS-485 only (repo catalog) | n/a | $369, Unitree shop (web search 18 Nov) | Buyable, but RS-485 only: cannot share the CAN buses or the T. rex drive family. The A1 motor is likewise RS-485 only (repo catalog) |
| MyActuator older "Pro" / V3 lines | n/a | n/a | n/a | n/a | n/a | n/a | Superseded by V4 / V4.1. The repo catalog lists the X8-P9 and X6-P36 lines as discontinued or RS-485 only |

Rated, peak, speed, mass, interface and heat data come from the repo catalog (`hardware/trex_form_study/designlab/data/actuators.json`, built from maker datasheets read 7 Nov 2026). I could not re-open the maker pages today (see "What I did not do"). The review's searches on 18 Nov 2026 found the same reseller picture: X6-60 $741.84 at RobotShop (tariffs included, 2 left) and $490 backordered at aifitlab and Robotics Center; X8-150 $645 backordered at aifitlab; RobStride 03 $225 on Seeed's datasheet (updated 21 Oct 2026), 880 g, 15–60 V; the X4 family's 24 V rating and 20–55 V range on reseller listings.

### 2.6 Chosen drives and their margins

| Build | Axis | Drive, ratio | Peak use (≤ 1) | Heat use (≤ 1) | Speed use (≤ 1) |
|---|---|---|---|---|---|
| **T20** | Roll | X6-60, 1:1 | 0.61 (0.66 at 60 % front) | 0.85 (0.92) | 0.28 |
| | Pitch | X4-36-L, 1:1 | 0.89 (0.97) | 0.98 trot / 0.44 walk (1.07 trot at 60 %) | **0.72 of the 83 rpm cap at 0.30 m/s; 0.93 at 0.40 m/s; 0.55 trot [F1]** |
| | Front flexion | X8-150, 1:1 | 0.47 (0.52) | 0.53 (0.58) | 0.60 |
| | Hind flexion | X6-60, 1.3:1 | 0.60 (0.66 at 50 % front) | 0.94 (1.05 at 50 % front) | 0.71 |
| **T15** | Roll | X4-36-L, 1:1 | 0.91 trot / 0.69 walk (0.99 / 0.76 at 60 %) | 1.01 trot / 0.86 walk (1.11 / 0.94) | **0.36 of the cap [F1]** |
| | Pitch | X4-36-L, 1:1 | 0.66 (0.73) | 0.74 (0.80) | **as T20 pitch [F1]** |
| | Flexion | X6-60, 1.3:1 | 0.55 (0.60); X6-50-L 0.73 | 0.87 (0.94) | 0.71 |

What this means:
- **T20 passes the 0.30 m/s walk on every axis with room to spare, and the trot at the 55 % design point. It does not reach the 0.40 m/s walk (pitch speed, [F1]).** Two axes are thin:
  - The front pitch at 60 % front load. If the robot weighs in above about 57 % front, swap the two front pitch drives for X6 (+0.92 kg, +$318).
  - The hind flexion if the robot ends up rear-heavy (50 % front).
- An all-X8 flexion variant passes everywhere but closes at 19.86 kg with only 0.14 kg of reserve, so it is not recommended.
- **T15 passes the walk.** Its roll drives exceed their heat limit in a sustained trot, so trots must stay short (minutes) until the bench thermal test says otherwise.
- Heat limits are the maker's open-air figures. Inside a painted shell they will be lower, so the shells need ducting (an open question; fans and ducts are now a BOM line [F11]).
- **Independent check (Appendix C):** with the fore-aft force through the ankle pin instead of at floor height, the review gets 31.3 / 34.9 N·m (walk / trot) at the T20 front elbow against the list's 32.1 / 35.8, and a T20 pitch trot heat use of 0.91 against 0.98. The hind-flexion heat figure (0.94, 1.05 at 50 % front) uses a held lone hind foot; in the forefoot paw pose a hind foot carries about 0.39 W rather than 0.45 W, so the realistic figure is about 0.81. Every pass and fail above stands.

### 2.7 Flexion transmission

The flexion motor stays where the PR put it, 90 mm down the upper link, which keeps distal mass low. Mounting an X8 at the elbow instead would add about 7× the drive's own contribution to the leg's swing inertia.

Drive it through a four-bar pushrod with 35–36 mm cranks. Pull at a 90° transmission angle, and **at the 40° minimum the list allows [F2]**:
- **T20** front, 1:1: 1.02 kN design, 1.17 kN pad-edge abuse; **1.59 kN and 1.82 kN at 40°**.
- **T15**, 1.3:1: 0.75 kN design, 0.85 kN abuse; **1.16 kN and 1.33 kN at 40°**.

Use steel M8 rod ends (Aurora / QA1 class, not polymer) and an angular-contact bearing pair at the output, each rated at least 2× the 40° figures; size the crank pins the same way. **[F2]**

The PR's 83° flexion range (−47° to +36°) needs about 108° of crank travel at 1.3:1. Keep the transmission angle ≥ 40° across that range, and check it in CAD.

If packaging forces a belt, use an 8 mm-pitch, high-modulus belt (Gates Poly Chain GT2 8MGT class) on 30–36 mm pulleys. Confirm from the maker's table that its working tension is at least 1.5× the design pull. I did not read the table.

### 2.8 Head, neck and tail servos

Script: `headtail.py` (scratchpad; not committed; output in Appendix B). Rated and peak figures are the repo catalog's values derived from ROBOTIS data.

| Joint | Servo | Need (hold / peak, N·m) | Use of rated / peak |
|---|---|---|---|
| Neck pitch, T20 head 1.51 kg, CoM 110 mm, no spring | XM540-W270-R | 1.63 / 1.86 | 0.52 / 0.42 |
| Same, spring cancelling 70 % of gravity | XM540-W270-R | 0.49 / 0.72 | 0.16 / 0.16 |
| Neck yaw, 35° in 0.6 s | XM540-W150-R | 0 / 0.33 | 0.11 (peak) |
| Tail yaw, ±24° at 1 Hz | XM430-W350-R | 0 / 0.24 | 0.14 (peak) |
| Tail pitch (T20), gravity + 10° at 1 Hz | XM430-W350-R | 0.59 / 0.69 | 0.49 / 0.40 |

All pass with margin, and the spring keeps the neck servo cool through long sessions. Run the X-series servos on their own RS-485 bus at 12 V (a Pololu D42V110F12 rail). **[F6]** The XL330 eyelid servos are 3.7–6 V TTL parts: give them a 5 V rail and the U2D2's TTL port, never the 12 V RS-485 bus.

## 3. The rest of the robot

**Compute and bus:**
- **Main computer:** Jetson Orin Nano Super 8 GB with NVMe, on a vibration tray with locking leads. It runs perception (the OAK-D does stereo on board), the policy and character behaviour.
- **Real-time controller:** a Teensy 4.1 on a custom carrier. It runs its 3 internal CAN controllers plus an MCP2518FD for a fourth bus, with isolated transceivers: **one classic 1 Mbit/s CAN bus per leg (3 drives each)**.
  - The loop runs at 500 Hz as the baseline (about 33–41 % bus load). 1 kHz would load the buses to 67–81 % (the survey's arithmetic).
  - The MyActuator "-E" drives also offer EtherCAT, so a dedicated NIC is listed to test a single-cable 1 kHz path. That matters if WIM's PLEM 1 kHz torque loop is adopted.
- **PLEM and W-RC:** optional. They need a quote from WIM Robotics and are not in the totals.
- **Teensy fallback:** it keeps the PR's damped-crouch behaviour on stale commands.

**Sensors:**
- **IMU:** Xsens MTi-3 at the centre of mass, with a BNO085 on the MCU as a cross-check.
- **Feet:** one FX29 per foot in a sole cartridge, preloaded under a 60A urethane pad, with a hard stop above about 400 N so drop impacts do not overload the cell (confirm the overload rating from TE's datasheet; 2.5× rated was found only for the 25 lbf sibling). Drive current gives a second contact estimate.
  - **[F5]** The FX29K0 is an I2C part (2.7–5.5 V). Do not run four of them on one I2C bus down the legs: give each leg a differential I2C extender pair (PCA9615 class) into its own Teensy I2C port or a mux, or put a small CAN node at each foot. Confirm the I2C address in TE's datasheet. Distributor lead time is 21–23 weeks: order the cells with the phase-1 set.
- **Camera:** OAK-D Pro W in the head behind the brow, on a screw-lock USB-C lead routed through the neck service loop.
- **Chest:** 2 × VL53L8CX for floor and visitor distance.

**Safety stop (W-RC-style, independent of software):**
- **Stop inputs:** a dual-channel chain opened by any of these:
  - the wireless PL d stop (carried by the handler), through its DC machine-side receiver (Cattron's XBMCU: 9–36 V DC, Cat 3 / PL d inputs, M12, CAN). Confirm that the $1,900 Safe-D-Stop kit includes a DC receiver; if not, it is the extra line in the BOM. **[F3]**
  - the latching mushroom e-stop on the robot's back (Schneider XB4BS8444, 2 NC: one contact per channel; the XB4BS8445 is 1 NO + 1 NC) **[F4]**;
  - a second mushroom on the tether pendant;
  - a watchdog MCU, which drops its output on a lost heartbeat from the Teensy or Jetson, a bus over-voltage, or a tilt beyond the limit.
- **Pilz PNOZ s5 C (24 V):**
  - Its instant contacts tell the Teensy to command damping and a controlled sit onto the belly skid.
  - Its delayed contacts (set at 0.5–1.5 s on the bench) drop the 12 V coil of the bidirectional Altran contactor (the sibling AREVS150-BN draws 461 mA at 12 V).
- **Rating:** the wireless stop and the relay are PL d parts; the watchdog MCU is not. The chain's level is set by an assessment of the whole (ISO 13849), not by its best parts. **[F3]**
- **BMS:** the Orion Jr 2's charge- and discharge-enable outputs are logic signals. Feed discharge-enable to the watchdog so a BMS trip runs the same sit-then-cut sequence. **[F10]**
- **Belly:** the skid and bumpers take the sit.

**Power:**
- **Pack:** 12S from two 6S 3300 packs in series. The charge ceiling is 4.10 V/cell (49.2 V), as the repo's 12S plan advises, to leave regen headroom.
- **Power board:**
  - an ideal-diode OR of battery and tether (2 × LTC4359), with the battery path shut down while a valid tether is present, so the tether always takes priority;
  - precharge, 4 leg fuses and bus capacitors;
  - an active clamp at 52.5 V, under the drives' 55 V maximum, dumping into two 50 W resistors. The ideal diodes block regen from reaching the pack, so the clamp must absorb it.
- **Rails:** two 12 V Pololu D42V110F12 rails (12–60 V input; one for Jetson and logic, one for servos), an isolated 24 V safety rail, and a small 5 V rail for the XL330 eyelids **[F6]**. The Jetson kit's DC jack takes 12 V (NVIDIA forum figures 7–20 V or 9–20 V; at least 8 V for Super mode).
- **Tether:** the Mean Well RSP-1000-48 trims to 40–110 % of 48 V (up to about 52.8 V); set it to about 50 V, under the 52.5 V clamp. Its over-voltage trip is 56.6–66.2 V. Distributor lead time is about 12 weeks.
- **Run time** on one 146.5 Wh set at 80 % usable:
  - T15: 33–47 min at 150–210 W.
  - T20: 25–35 min at 200–280 W.
  - Both power figures are the PR's estimate scaled by mass, not measured. Long sessions run on the tether.
- **Battery shipping and travel:**
  - Get the UN38.3 test summary for the pack from the seller.
  - Each 6S pack is 73 Wh. Under IATA passenger rules (general; check with the airline), spares up to 100 Wh can travel in carry-on.
  - Ship by ground or as Class 9 otherwise.
  - Store and charge in a LiPo cabinet.

**Wiring:**
- Trunk 10–12 AWG; drive branches 14–16 AWG.
- CAN twisted pairs with termination at each leg's far end; RS-485 for the servos.
- Each limb module has one M12 A-coded CAN connector and one power quick-disconnect, so a leg swaps in minutes.
- Service loops at the neck and tail.

**Frame:**
- CNC 6061-T6 rails, bulkheads, roll yokes, links and drive mounts, anodized.
- The drive output bearings carry roll and pitch. The elbow and knee get their own angular-contact pair, with outboard shaft support.
- Feet are cast 60A urethane with load-cell cartridges.

**Shells:**
- HP MJF PA11, 1.0–1.2 mm, ribbed, dyed. Then prime, hand-paint, texture and matte-clear for a museum finish.
- Soft TPU horn tips and beak edge, and gaiters over the joints.
- Leave intake and exhaust paths over the drives, with blower fans and dust filters (a BOM line) **[F11]**.

**Transport and tools [F11]:** a padded road case with custom foam for museum trips, a LiPo travel bag, and a tool kit (crimpers for Anderson, ferrules and JST/Molex, torque drivers, hex/Torx, a meter).

## 4. Parts list: T20, 20 kg (recommended)

How to read the source column:
- **RC:** the repo catalog (`hardware/trex_form_study/designlab/data/actuators.json` and `parts.json`). It records a list price from the maker's or distributor's page, read on 7 Nov 2026.
- **MD:** `docs/hardware/RAPTOR_SCALING_AND_ALTERNATIVES.md` on main, with prices last checked 24 Jul 2026.
- **TB:** the T. rex `stage2/jetson_plem/BUILD_BUDGET_75_JETSON.md`.
- **WS:** a web-search result read on 18 Nov 2026. The listing page itself was not opened.
- **EST:** an estimate, needing a quote.
- **[Fn]:** a row changed or added by the review; see "Review corrections". Rows the review re-checked by web search on 18 Nov 2026 cite that date.

All prices are list prices unless the row says "quote". Rows marked "High only" cost nothing in the low and nominal cases.

| Category | Part | Maker and model | Qty | Unit (nominal) | Source, date | Line total (nominal) | Line low - high |
|---|---|---|---:|---:|---|---:|---:|
| Leg drives | Upper-leg pitch | MyActuator RMD-X4-P36-36-E-L (V4.1) | 4 | $269 | RC myactuator.us list, 7 Nov 2026 | $1,076 | $1,076 - $1,076 |
| Leg drives | Roll (4) + hind knee flexion via 1.3:1 linkage (2) | MyActuator RMD-X6-P20-60-E (V4; confirm vs X6-50-L) | 6 | $428 | RC myactuator.us list, 7 Nov 2026 | $2,568 | $2,568 - $2,568 |
| Leg drives | Front elbow flexion via 1:1 linkage | MyActuator RMD-X8-P20-150-E (V4.1, US SKU -03) | 2 | $725 | RC myactuator.us list, 7 Nov 2026 | $1,450 | $1,450 - $1,450 |
| Leg drives | Flexion transmissions (four-bar pushrod or 8 mm-pitch belt), crank arms, output shafts **[F2]** | custom + steel M8 rod ends (Aurora / QA1 class; not polymer) | 4 | $120 | EST | $480 | $240 - $900 |
| Head and tail | Neck pitch (spring-assisted) **[F7]** | ROBOTIS DYNAMIXEL XM540-W270-R | 1 | $494.39 | robotis.us XM list + RobotShop $494.39, WS 18 Nov 2026 (sold out, long lead) | $494.39 | $494.39 - $494.39 |
| Head and tail | Neck yaw **[F7]** | ROBOTIS DYNAMIXEL XM540-W150-R | 1 | $494.39 | robotis.us XM list $494.39, WS 18 Nov 2026 | $494.39 | $494.39 - $494.39 |
| Head and tail | Beak / jaw **[F7]** | ROBOTIS DYNAMIXEL XM430-W350-R | 1 | $333.39 | robotis.us $333.39 (sold out); RobotShop $333.39 incl. tariffs, WS 18 Nov 2026 | $333.39 | $333.39 - $333.39 |
| Head and tail | Tail yaw + tail pitch **[F7]** | ROBOTIS DYNAMIXEL XM430-W350-R | 2 | $333.39 | robotis.us $333.39 (sold out); RobotShop $333.39, WS 18 Nov 2026 | $666.78 | $666.78 - $666.78 |
| Head and tail | Eyelids **[F6]** | ROBOTIS DYNAMIXEL XL330-M288-T (3.7-6 V, TTL: own 5 V rail and the U2D2 TTL port) | 2 | $27.49 | robotis.us $27.49 (sold out), WS 18 Nov 2026; high EST | $54.98 | $54.98 - $70 |
| Head and tail | Servo bus adapter + power hub | ROBOTIS U2D2 + U2D2 PHB | 1 | $55 | EST (not checked) | $55 | $45 - $70 |
| Head and tail | Neck yaw thin-section bearing | thin-section ball bearing | 1 | $75 | TB (X_COSTS -X head) | $75 | $60 - $110 |
| Head and tail | Neck counterbalance spring / gas strut | catalog spring | 1 | $40 | EST | $40 | $25 - $70 |
| Head and tail | Frill blush + eye lighting | addressable RGB LEDs, diffusers, driver | 1 | $100 | EST | $100 | $60 - $180 |
| Head and tail | Character audio (amp, chest transducer, mic) | I2S class-D amp + exciter + MEMS mic | 1 | $65 | TB line 6 | $65 | $35 - $120 |
| Head and tail | 5 V rail for the XL330 eyelids (from the 12 V servo rail) **[F6]** | 5 V >= 2 A step-down (Pololu D36V28F5 class) | 1 | $20 | EST | $20 | $12 - $30 |
| Compute and control | Main computer (high = Orin NX 16 GB reComputer J4012) **[F8]** | NVIDIA Jetson Orin Nano Super Developer Kit 8 GB | 1 | $399 | NVIDIA list $399 since Jul 2026 (was $249; cnx-software 22 Jul 2026), WS 18 Nov 2026; high RobotShop J4012 $1,364.29 incl. tariffs (undated), WS 18 Nov 2026 | $399 | $399 - $1,364.29 |
| Compute and control | NVMe SSD 512 GB-1 TB | M.2 2280 TLC NVMe | 1 | $65 | TB | $65 | $55 - $95 |
| Compute and control | Real-time MCU + 4-bus isolated CAN carrier (1 fitted + 1 spare) | PJRC Teensy 4.1 + MCP2518FD + 4 x isolated CAN transceiver, assembled PCB | 2 | $150 | Teensy WS 18 Nov ($32-35); board EST | $300 | $180 - $600 |
| Compute and control | Safety supervisor (W-RC-style watchdog) | independent MCU board, relay driver, tilt + bus-voltage sense | 1 | $80 | EST (W-RC: quote from WIM) | $80 | $40 - $150 |
| Compute and control | USB-CAN FD bench/debug interface (low = CANable 2.0 isolated) **[F9]** | PEAK-System PCAN-USB FD IPEH-004022 | 1 | $379 | DigiKey $379 + $15 ship; Gridconnect $347, WS 18 Nov 2026; low TB | $379 | $65 - $379 |
| Compute and control | Dedicated EtherCAT trial NIC | USB3 / M.2 gigabit Ethernet adapter | 1 | $40 | EST | $40 | $0 - $60 |
| Sensors | Trunk IMU (low BNO085; high Xsens MTi-630-DK) | Xsens MTi-3 development kit | 1 | $417.06 | nom WS 18 Nov (Arrow); high WS (Cytech ref.); low MD (Adafruit) | $417.06 | $24.95 - $1,618.81 |
| Sensors | Backup IMU on the MCU | Adafruit BNO085 breakout | 1 | $24.95 | MD (Adafruit, 24 Jul 2026) | $24.95 | $0 - $24.95 |
| Sensors | Foot load cells, 100 lbf (445 N), I2C **[F5]** | TE Connectivity FX29K0-100A-0100-L (I2C, 2.7-5.5 V; ~21-23 wk lead) | 4 | $43.57 | WS 18 Nov (DigiKey via Findchips / zenode) | $174.28 | $145.72 - $198.76 |
| Sensors | Head depth camera (low = OAK-D Lite) | Luxonis OAK-D Pro W (A00573) | 1 | $529 | RC luxonis 7 Nov 2026; WS 18 Nov $529 | $529 | $269 - $529 |
| Sensors | Locking USB-C camera cable + service loop | industrial screw-lock USB-C | 1 | $30 | EST | $30 | $20 - $50 |
| Sensors | Chest ToF, 8x8 zones | Pololu VL53L8CX carrier #3419 | 2 | $24.95 | RC pololu.com, 7 Nov 2026 | $49.90 | $49.90 - $49.90 |
| Sensors | Foot-sensor links: per-leg differential I2C extender pair (or a small per-foot CAN node) **[F5]** | NXP PCA9615 class + twisted pair | 4 | $20 | EST | $80 | $40 - $200 |
| Safety stop | Wireless e-stop, PL d (high = FORT Robotics, quote) | Cattron/OTE Safe-D-Stop | 1 | $1,900 | WS 18 Nov (Hoosier Crane listing); high EST | $1,900 | $1,900 - $2,800 |
| Safety stop | Safety relay, 2 instant + 2 delayed contacts (Cat 1 stop) | Pilz PNOZ s5 C 24VDC 2 n/o 2 n/o t (751105) | 1 | $611.99 | low WS Pilz eShop EUR 358 (USD est.); nom WS Zoro | $611.99 | $415 - $611.99 |
| Safety stop | E-stop buttons (on robot + tether pendant) **[F4]** | Schneider Harmony XB4BS8444 (2 NC, screw) - not XB4BS8445 (1 NO + 1 NC) | 2 | $50 | MD street price for XB4BS8445 (24 Jul 2026); 2 NC part assumed equal, not checked | $100 | $90 - $110 |
| Safety stop | Pendant box + robot-side bezel | IP65 1-hole enclosure | 1 | $40 | EST | $40 | $25 - $60 |
| Safety stop | Main contactor, bidirectional, 12 V coil | Altran AREVS150-BAN | 1 | $54.45 | MD (DigiKey, 24 Jul 2026) | $54.45 | $54.45 - $54.45 |
| Safety stop | Main fuse 100 A / 58 V + 58 V holder | Littelfuse BF1 142.5631.6102 + MIDI holder | 1 | $20 | MD fuse $4.70; holder EST | $20 | $20 - $25 |
| Safety stop | Machine-side receiver for the wireless stop, if not in the $1,900 kit (9-36 V DC, PL d Cat 3 inputs, M12, CAN) **[F3]** | Cattron XBMCU | 1 | $800 | EST (no price found; Cattron datasheet via WS 18 Nov 2026); quote | $800 | $0 - $1,500 |
| Power | Battery: 2 sets of 2 x 6S in series (12S, 146.5 Wh/set) | Gens ace G-Tech 3300mAh 6S 60C | 4 | $107.19 | RC genstattu.com, 7 Nov 2026 | $428.76 | $428.76 - $428.76 |
| Power | Battery management system (cell V/T, CAN; charge/discharge-enable logic outputs wired into the stop chain) **[F10]** | Orion BMS Jr 2, 16-cell, CAN (low = T15 monitor set) | 1 | $440 | nom WS 18 Nov (orionbms.com MSRP); high WS reseller | $440 | $124.45 - $656 |
| Power | Power board: 2 x LTC4359 ideal-diode OR, precharge, 4 leg fuses, bus caps, 52.5 V regen clamp + 2 x 50 W resistors | custom PCB, assembled (2) | 1 | $300 | EST (LTC4359 4-80 V: WS 18 Nov) | $300 | $150 - $600 |
| Power | 12 V rails (Jetson/logic; servos) **[F9]** | Pololu D42V110F12 (12-60 V in, 12 V 9 A) | 2 | $59.95 | pololu.com #5677 $59.95 (rationed, backorders), WS 18 Nov 2026 | $119.90 | $119.90 - $119.90 |
| Power | 24 V safety rail (relay, receiver) | isolated DC-DC, 18-75 V in, >= 10 W | 1 | $35 | EST | $35 | $25 - $60 |
| Power | Tether supply | Mean Well RSP-1000-48 | 1 | $267.90 | MD (DigiKey, 24 Jul 2026) | $267.90 | $267.90 - $267.90 |
| Power | Tether lead: 5 m SJOOW, Anderson SB50, strain relief | assorted | 1 | $90 | EST | $90 | $60 - $150 |
| Power | Balance charger + its supply | Junsi iCharger X12 + Mean Well RSP-1000-48 | 1 | $447.89 | MD ($179.99 + $267.90, 24 Jul 2026) | $447.89 | $447.89 - $447.89 |
| Power | LiPo charging/storage containment | LiPo bags or steel cabinet | 1 | $120 | EST | $120 | $60 - $250 |
| Power | Cooling: blower fans, ducts and dust filters for the drive bays **[F11]** | assorted 12 V blowers | 1 | $90 | EST | $90 | $40 - $180 |
| Power | BMS current sensor (dual-range Hall; Orion calls it optional but strongly recommended) **[F10]** | Orion BMS current sensor | 1 | $100 | EST (no price found; orionbms.com spec via WS 18 Nov 2026) | $100 | $60 - $160 |
| Wiring | Harness: 10-12 AWG trunk, 14-16 AWG drive branches, CAN pairs, RS-485, 4 limb quick-disconnects (M12 CAN + power), sleeving | assorted (Amphenol/LTW M12, Anderson, Amass) | 1 | $550 | EST (TB line 9 x ~1.5) | $550 | $350 - $850 |
| Frame | CNC 6061-T6 frame and leg structure, anodized | outsourced CNC (Xometry/Protolabs class) | 1 | $3,400 | EST (from TB line 10) | $3,400 | $2,300 - $5,200 |
| Frame | Joint bearings (elbow/knee pairs, shaft supports, tail pitch) | angular-contact + deep-groove | 1 | $220 | EST | $220 | $120 - $400 |
| Frame | Feet: cast 60A urethane soles, load-cell cartridges, overload stops | custom | 4 | $75 | EST | $300 | $152 - $500 |
| Frame | Fasteners, heat-set inserts, shims, threadlocker | assorted | 1 | $150 | EST | $150 | $80 - $250 |
| Frame | Belly skid + strike bumpers for the controlled sit | EVA / urethane | 1 | $80 | EST | $80 | $40 - $150 |
| Shells | Shells: body, limbs, head, frill, tail | HP MJF PA11 (1.0-1.2 mm, ribbed), dyed | 1 | $1,950 | EST (TB line 11 analog) | $1,950 | $1,100 - $3,400 |
| Shells | Museum paint finish (prime, hand paint, texture, matte clear) | prop / scenic painter | 1 | $1,200 | EST (quote) | $1,200 | $600 - $2,500 |
| Shells | Soft horn tips, beak edge, joint gaiters | TPU / silicone | 1 | $150 | EST | $150 | $80 - $300 |
| Spares | Spare roll/pitch drive | MyActuator RMD-X4-P36-36-E-L | 1 | $269 | RC 7 Nov 2026 | $269 | $269 - $269 |
| Spares | Spare roll / hind flexion drive | MyActuator RMD-X6-P20-60-E | 1 | $428 | RC 7 Nov 2026 | $428 | $428 - $428 |
| Spares | Spare front flexion drive | MyActuator RMD-X8-P20-150-E | 1 | $725 | RC 7 Nov 2026 | $725 | $725 - $725 |
| Spares | Spare foot load cells **[F5]** | TE FX29K0-100A-0100-L | 4 | $43.57 | WS 18 Nov | $174.28 | $145.72 - $198.76 |
| Spares | Rod ends, belts, soles, fuses, connectors | assorted | 1 | $250 | EST | $250 | $150 - $400 |
| Spares | High only: second spare X8 | MyActuator RMD-X8-P20-150-E | 1 | $0 | RC 7 Nov 2026 | $0 | $0 - $725 |
| Spares | High only: spare neck servo **[F7]** | ROBOTIS XM540-W270-R | 1 | $0 | robotis.us WS 18 Nov 2026 | $0 | $0 - $494.39 |
| Spares | High only: spare beak/tail servo **[F7]** | ROBOTIS XM430-W350-R | 1 | $0 | robotis.us WS 18 Nov 2026 | $0 | $0 - $333.39 |
| Spares | High only: second spare X6 | MyActuator RMD-X6-P20-60-E | 1 | $0 | RC 7 Nov 2026 | $0 | $0 - $428 |
| Spares | High only: spare battery set | Gens ace G-Tech 6S 3300 x 2 | 2 | $0 | RC 7 Nov 2026 | $0 | $0 - $214.38 |
| Bench (shared with T. rex) | Single-leg bench + torque rig (lever arm, S-beam load cell, amp; spare drive as dynamometer; high adds a FUTEK TFF400) | custom + FUTEK/Interface class | 1 | $700 | EST (a FUTEK TFF400 adds ~EUR 2,376, WS 18 Nov) | $700 | $400 - $1,800 |
| Bench (shared with T. rex) | Overhead catch gantry, tool balancer, harness | 80/20 extrusion + balancer | 1 | $800 | EST (TB line 12) | $800 | $500 - $1,500 |
| Bench (shared with T. rex) | Current-limited 60 V bench supply | programmable DC supply | 1 | $400 | EST | $400 | $0 - $900 |
| Bench (shared with T. rex) | Thermal camera for drive heat checks | USB / handheld thermal imager | 1 | $250 | EST | $250 | $0 - $500 |
| Tools and transport | Tools: crimpers (Anderson, ferrules, JST/Molex), torque drivers, hex/Torx set, DMM **[F11]** | assorted | 1 | $350 | EST | $350 | $150 - $800 |
| Tools and transport | Transport: padded road case with custom foam for museum travel **[F11]** | road case (1.5 m robot) | 1 | $800 | EST | $800 | $400 - $1,500 |

## 5. Parts list: T15, 15 kg

| Category | Part | Maker and model | Qty | Unit (nominal) | Source, date | Line total (nominal) | Line low - high |
|---|---|---|---:|---:|---|---:|---:|
| Leg drives | Roll + upper-leg pitch | MyActuator RMD-X4-P36-36-E-L (V4.1) | 8 | $269 | RC myactuator.us list, 7 Nov 2026 | $2,152 | $2,152 - $2,152 |
| Leg drives | Elbow/knee flexion via 1.3:1 linkage | MyActuator RMD-X6-P20-60-E (V4; confirm vs X6-50-L) | 4 | $428 | RC myactuator.us list, 7 Nov 2026 | $1,712 | $1,712 - $1,712 |
| Leg drives | Flexion transmissions (four-bar pushrod or 8 mm-pitch belt), crank arms, output shafts **[F2]** | custom + steel M8 rod ends (Aurora / QA1 class; not polymer) | 4 | $120 | EST | $480 | $240 - $900 |
| Head and tail | Neck pitch (spring-assisted) **[F7]** | ROBOTIS DYNAMIXEL XM540-W270-R | 1 | $494.39 | robotis.us XM list + RobotShop $494.39, WS 18 Nov 2026 (sold out, long lead) | $494.39 | $494.39 - $494.39 |
| Head and tail | Neck yaw **[F7]** | ROBOTIS DYNAMIXEL XM540-W150-R | 1 | $494.39 | robotis.us XM list $494.39, WS 18 Nov 2026 | $494.39 | $494.39 - $494.39 |
| Head and tail | Beak / jaw **[F7]** | ROBOTIS DYNAMIXEL XM430-W350-R | 1 | $333.39 | robotis.us $333.39 (sold out); RobotShop $333.39 incl. tariffs, WS 18 Nov 2026 | $333.39 | $333.39 - $333.39 |
| Head and tail | Tail yaw **[F7]** | ROBOTIS DYNAMIXEL XM430-W350-R | 1 | $333.39 | robotis.us $333.39 (sold out); RobotShop $333.39, WS 18 Nov 2026 | $333.39 | $333.39 - $333.39 |
| Head and tail | Servo bus adapter + power hub | ROBOTIS U2D2 + U2D2 PHB | 1 | $55 | EST (not checked) | $55 | $45 - $70 |
| Head and tail | Neck yaw thin-section bearing | thin-section ball bearing | 1 | $75 | TB (X_COSTS -X head) | $75 | $60 - $110 |
| Head and tail | Neck counterbalance spring / gas strut | catalog spring | 1 | $40 | EST | $40 | $25 - $70 |
| Head and tail | Frill blush + eye lighting | addressable RGB LEDs, diffusers, driver | 1 | $100 | EST | $100 | $60 - $180 |
| Head and tail | Character audio (amp, chest transducer, mic) | I2S class-D amp + exciter + MEMS mic | 1 | $65 | TB line 6 | $65 | $35 - $120 |
| Compute and control | Main computer (high = Orin NX 16 GB reComputer J4012) **[F8]** | NVIDIA Jetson Orin Nano Super Developer Kit 8 GB | 1 | $399 | NVIDIA list $399 since Jul 2026 (was $249; cnx-software 22 Jul 2026), WS 18 Nov 2026; high RobotShop J4012 $1,364.29 incl. tariffs (undated), WS 18 Nov 2026 | $399 | $399 - $1,364.29 |
| Compute and control | NVMe SSD 512 GB-1 TB | M.2 2280 TLC NVMe | 1 | $65 | TB | $65 | $55 - $95 |
| Compute and control | Real-time MCU + 4-bus isolated CAN carrier (1 fitted + 1 spare) | PJRC Teensy 4.1 + MCP2518FD + 4 x isolated CAN transceiver, assembled PCB | 2 | $150 | Teensy WS 18 Nov ($32-35); board EST | $300 | $180 - $600 |
| Compute and control | Safety supervisor (W-RC-style watchdog) | independent MCU board, relay driver, tilt + bus-voltage sense | 1 | $80 | EST (W-RC: quote from WIM) | $80 | $40 - $150 |
| Compute and control | USB-CAN FD bench/debug interface (low = CANable 2.0 isolated) **[F9]** | PEAK-System PCAN-USB FD IPEH-004022 | 1 | $379 | DigiKey $379 + $15 ship; Gridconnect $347, WS 18 Nov 2026; low TB | $379 | $65 - $379 |
| Compute and control | Dedicated EtherCAT trial NIC | USB3 / M.2 gigabit Ethernet adapter | 1 | $40 | EST | $40 | $0 - $60 |
| Sensors | Trunk IMU (low BNO085; high Xsens MTi-630-DK) | Xsens MTi-3 development kit | 1 | $417.06 | nom WS 18 Nov (Arrow); high WS (Cytech ref.); low MD (Adafruit) | $417.06 | $24.95 - $1,618.81 |
| Sensors | Backup IMU on the MCU | Adafruit BNO085 breakout | 1 | $24.95 | MD (Adafruit, 24 Jul 2026) | $24.95 | $0 - $24.95 |
| Sensors | Foot load cells, 100 lbf (445 N), I2C **[F5]** | TE Connectivity FX29K0-100A-0100-L (I2C, 2.7-5.5 V; ~21-23 wk lead) | 4 | $43.57 | WS 18 Nov (DigiKey via Findchips / zenode) | $174.28 | $145.72 - $198.76 |
| Sensors | Head depth camera (low = OAK-D Lite) | Luxonis OAK-D Pro W (A00573) | 1 | $529 | RC luxonis 7 Nov 2026; WS 18 Nov $529 | $529 | $269 - $529 |
| Sensors | Locking USB-C camera cable + service loop | industrial screw-lock USB-C | 1 | $30 | EST | $30 | $20 - $50 |
| Sensors | Chest ToF, 8x8 zones | Pololu VL53L8CX carrier #3419 | 2 | $24.95 | RC pololu.com, 7 Nov 2026 | $49.90 | $49.90 - $49.90 |
| Sensors | Foot-sensor links: per-leg differential I2C extender pair (or a small per-foot CAN node) **[F5]** | NXP PCA9615 class + twisted pair | 4 | $20 | EST | $80 | $40 - $200 |
| Safety stop | Wireless e-stop, PL d (high = FORT Robotics, quote) | Cattron/OTE Safe-D-Stop | 1 | $1,900 | WS 18 Nov (Hoosier Crane listing); high EST | $1,900 | $1,900 - $2,800 |
| Safety stop | Safety relay, 2 instant + 2 delayed contacts (Cat 1 stop) | Pilz PNOZ s5 C 24VDC 2 n/o 2 n/o t (751105) | 1 | $611.99 | low WS Pilz eShop EUR 358 (USD est.); nom WS Zoro | $611.99 | $415 - $611.99 |
| Safety stop | E-stop buttons (on robot + tether pendant) **[F4]** | Schneider Harmony XB4BS8444 (2 NC, screw) - not XB4BS8445 (1 NO + 1 NC) | 2 | $50 | MD street price for XB4BS8445 (24 Jul 2026); 2 NC part assumed equal, not checked | $100 | $90 - $110 |
| Safety stop | Pendant box + robot-side bezel | IP65 1-hole enclosure | 1 | $40 | EST | $40 | $25 - $60 |
| Safety stop | Main contactor, bidirectional, 12 V coil | Altran AREVS150-BAN | 1 | $54.45 | MD (DigiKey, 24 Jul 2026) | $54.45 | $54.45 - $54.45 |
| Safety stop | Main fuse 100 A / 58 V + 58 V holder | Littelfuse BF1 142.5631.6102 + MIDI holder | 1 | $20 | MD fuse $4.70; holder EST | $20 | $20 - $25 |
| Safety stop | Machine-side receiver for the wireless stop, if not in the $1,900 kit (9-36 V DC, PL d Cat 3 inputs, M12, CAN) **[F3]** | Cattron XBMCU | 1 | $800 | EST (no price found; Cattron datasheet via WS 18 Nov 2026); quote | $800 | $0 - $1,500 |
| Power | Battery: 2 sets of 2 x 6S in series (12S, 146.5 Wh/set) | Gens ace G-Tech 3300mAh 6S 60C | 4 | $107.19 | RC genstattu.com, 7 Nov 2026 | $428.76 | $428.76 - $428.76 |
| Power | Pack monitoring: cell alarm, pack temperature, bus V/I on CAN | Chargery BS12 V2.3 + DS18B20 + Holybro PM08-CAN | 1 | $124.45 | MD (24 Jul 2026) | $124.45 | $124.45 - $124.45 |
| Power | Power board: 2 x LTC4359 ideal-diode OR, precharge, 4 leg fuses, bus caps, 52.5 V regen clamp + 2 x 50 W resistors | custom PCB, assembled (2) | 1 | $300 | EST (LTC4359 4-80 V: WS 18 Nov) | $300 | $150 - $600 |
| Power | 12 V rails (Jetson/logic; servos) **[F9]** | Pololu D42V110F12 (12-60 V in, 12 V 9 A) | 2 | $59.95 | pololu.com #5677 $59.95 (rationed, backorders), WS 18 Nov 2026 | $119.90 | $119.90 - $119.90 |
| Power | 24 V safety rail (relay, receiver) | isolated DC-DC, 18-75 V in, >= 10 W | 1 | $35 | EST | $35 | $25 - $60 |
| Power | Tether supply | Mean Well RSP-1000-48 | 1 | $267.90 | MD (DigiKey, 24 Jul 2026) | $267.90 | $267.90 - $267.90 |
| Power | Tether lead: 5 m SJOOW, Anderson SB50, strain relief | assorted | 1 | $90 | EST | $90 | $60 - $150 |
| Power | Balance charger + its supply | Junsi iCharger X12 + Mean Well RSP-1000-48 | 1 | $447.89 | MD ($179.99 + $267.90, 24 Jul 2026) | $447.89 | $447.89 - $447.89 |
| Power | LiPo charging/storage containment | LiPo bags or steel cabinet | 1 | $120 | EST | $120 | $60 - $250 |
| Power | Cooling: blower fans, ducts and dust filters for the drive bays **[F11]** | assorted 12 V blowers | 1 | $90 | EST | $90 | $40 - $180 |
| Wiring | Harness: 10-12 AWG trunk, 14-16 AWG drive branches, CAN pairs, RS-485, 4 limb quick-disconnects (M12 CAN + power), sleeving | assorted (Amphenol/LTW M12, Anderson, Amass) | 1 | $550 | EST (TB line 9 x ~1.5) | $550 | $350 - $850 |
| Frame | CNC 6061-T6 frame and leg structure, anodized | outsourced CNC (Xometry/Protolabs class) | 1 | $3,100 | EST (from TB line 10) | $3,100 | $2,100 - $4,800 |
| Frame | Joint bearings (elbow/knee pairs, shaft supports, tail pitch) | angular-contact + deep-groove | 1 | $220 | EST | $220 | $120 - $400 |
| Frame | Feet: cast 60A urethane soles, load-cell cartridges, overload stops | custom | 4 | $75 | EST | $300 | $152 - $500 |
| Frame | Fasteners, heat-set inserts, shims, threadlocker | assorted | 1 | $150 | EST | $150 | $80 - $250 |
| Frame | Belly skid + strike bumpers for the controlled sit | EVA / urethane | 1 | $80 | EST | $80 | $40 - $150 |
| Shells | Shells: body, limbs, head, frill, tail | HP MJF PA11 (1.0-1.2 mm, ribbed), dyed | 1 | $1,750 | EST (TB line 11 analog) | $1,750 | $1,000 - $3,100 |
| Shells | Museum paint finish (prime, hand paint, texture, matte clear) | prop / scenic painter | 1 | $1,200 | EST (quote) | $1,200 | $600 - $2,500 |
| Shells | Soft horn tips, beak edge, joint gaiters | TPU / silicone | 1 | $150 | EST | $150 | $80 - $300 |
| Spares | Spare roll/pitch drive | MyActuator RMD-X4-P36-36-E-L | 1 | $269 | RC 7 Nov 2026 | $269 | $269 - $269 |
| Spares | Spare flexion drive | MyActuator RMD-X6-P20-60-E | 1 | $428 | RC 7 Nov 2026 | $428 | $428 - $428 |
| Spares | Spare foot load cells **[F5]** | TE FX29K0-100A-0100-L | 4 | $43.57 | WS 18 Nov | $174.28 | $145.72 - $198.76 |
| Spares | Rod ends, belts, soles, fuses, connectors | assorted | 1 | $250 | EST | $250 | $150 - $400 |
| Spares | High only: spare neck servo **[F7]** | ROBOTIS XM540-W270-R | 1 | $0 | robotis.us WS 18 Nov 2026 | $0 | $0 - $494.39 |
| Spares | High only: spare beak/tail servo **[F7]** | ROBOTIS XM430-W350-R | 1 | $0 | robotis.us WS 18 Nov 2026 | $0 | $0 - $333.39 |
| Spares | High only: second spare X6 | MyActuator RMD-X6-P20-60-E | 1 | $0 | RC 7 Nov 2026 | $0 | $0 - $428 |
| Spares | High only: spare battery set | Gens ace G-Tech 6S 3300 x 2 | 2 | $0 | RC 7 Nov 2026 | $0 | $0 - $214.38 |
| Bench (shared with T. rex) | Single-leg bench + torque rig (lever arm, S-beam load cell, amp; spare drive as dynamometer; high adds a FUTEK TFF400) | custom + FUTEK/Interface class | 1 | $700 | EST (a FUTEK TFF400 adds ~EUR 2,376, WS 18 Nov) | $700 | $400 - $1,800 |
| Bench (shared with T. rex) | Overhead catch gantry, tool balancer, harness | 80/20 extrusion + balancer | 1 | $800 | EST (TB line 12) | $800 | $500 - $1,500 |
| Bench (shared with T. rex) | Current-limited 60 V bench supply | programmable DC supply | 1 | $400 | EST | $400 | $0 - $900 |
| Bench (shared with T. rex) | Thermal camera for drive heat checks | USB / handheld thermal imager | 1 | $250 | EST | $250 | $0 - $500 |
| Tools and transport | Tools: crimpers (Anderson, ferrules, JST/Molex), torque drivers, hex/Torx set, DMM **[F11]** | assorted | 1 | $350 | EST | $350 | $150 - $800 |
| Tools and transport | Transport: padded road case with custom foam for museum travel **[F11]** | road case (1.5 m robot) | 1 | $800 | EST | $800 | $400 - $1,500 |

## 6. Totals and mass budgets

### 6.1 Cost totals

Corrected by the review **[F12]**; the unreviewed totals were T15 $19,075 / $30,215 / $52,694 and T20 $21,901 / $34,487 / $60,310 landed.

| Line | T15 low | T15 nominal | T15 high | T20 low | T20 nominal | T20 high |
|---|---:|---:|---:|---:|---:|---:|
| Leg drives | $4,104 | $4,344 | $4,764 | $5,334 | $5,574 | $5,994 |
| Head and tail | $1,881 | $1,991 | $2,206 | $2,281 | $2,399 | $2,639 |
| Compute and control | $739 | $1,263 | $2,648 | $739 | $1,263 | $2,648 |
| Sensors | $550 | $1,305 | $2,671 | $550 | $1,305 | $2,671 |
| Safety stop | $2,504 | $3,526 | $5,161 | $2,504 | $3,526 | $5,161 |
| Power | $1,724 | $2,024 | $2,629 | $1,784 | $2,439 | $3,320 |
| Wiring | $350 | $550 | $850 | $350 | $550 | $850 |
| Frame | $2,492 | $3,850 | $6,100 | $2,692 | $4,150 | $6,500 |
| Shells | $1,680 | $3,100 | $5,900 | $1,780 | $3,300 | $6,200 |
| Spares | $993 | $1,121 | $2,766 | $1,718 | $1,846 | $4,216 |
| Bench (shared with T. rex) | $900 | $2,150 | $4,700 | $900 | $2,150 | $4,700 |
| Tools and transport | $550 | $1,150 | $2,300 | $550 | $1,150 | $2,300 |
| **Hardware subtotal** | **$18,466** | **$26,374** | **$42,696** | **$21,182** | **$29,653** | **$47,200** |
| Freight and brokerage (3 / 5 / 6 % of hardware) | $554 | $1,319 | $2,562 | $635 | $1,483 | $2,832 |
| Import duty / tariff on MyActuator drives (0 / 25 / 73 %) | $0 | $1,140 | $3,642 | $0 | $1,629 | $5,598 |
| Sales tax (0 / 7 / 10 % of hardware) | $0 | $1,846 | $4,270 | $0 | $2,076 | $4,720 |
| Contingency (5 / 7 / 10 % of everything above) | $951 | $2,148 | $5,317 | $1,091 | $2,439 | $6,035 |
| **All-in, self-assembled, landed** | **$19,971** | **$32,827** | **$58,486** | **$22,908** | **$37,280** | **$66,386** |
| T. rex method (hardware + 5.3 / 7.2 / 8.4 %) | $19,445 | $28,273 | $46,282 | $22,304 | $31,788 | $51,165 |

How the extra cost lines are set:
- **Duty:** applied to the MyActuator drives only.
  - Low (0 %) assumes the US store's list price is already landed.
  - High (73 %) is RobotShop's tariff-included X6-60 price of $741.84 against the $428 list. That is a reseller price, so it carries the reseller's margin as well as duty; read it as "buy from a US reseller", not as a tariff rate. **[F12]**
  - Nominal (25 %) is an assumption.
- **Sales tax:** depends on where you buy and on exemptions.
- **Not included:**
  - turnkey integration by a contract shop: about +$2.8k / +$4.0k / +$6.0k (estimate, scaled from the T. rex's +$2.8k nominal);
  - PLEM and W-RC from WIM (quote);
  - any labour.

**Ways to bring T20 down by about $5k to $7k nominal without going cheap [F12]:**
- Count the $2.2k of bench gear against the T. rex as well.
- Have the team paint the shells instead of a prop painter (−$1.2k).
- Confirm that the wireless-stop kit includes its DC receiver (−$0.8k of estimate).
- Skip the road case until the first trip off site (−$0.8k).
- Use the OAK-D Lite indoors if the halls are bright (−$260).
- Buy the drives from the US store if its price proves to be landed (−$1.6k of duty).

With all of these the T20 lands near $29.3k nominal landed (about $26.2k the T. rex way), inside the brief's class.

### 6.2 Mass budgets

Masses are in kg. "PR" is PR #669's 15.00 kg budget. The front share is estimated by applying each change at its position to the PR's robot.xml mass model (`massbudget.py`), with the battery 200 mm aft of centre.

| Item | PR #669 | T15 | T20 | What is in it |
|---|---:|---:|---:|---|
| Leg drives | 6.16 | 6.16 | 9.02 | T15: 8 × X4 (0.36) + 4 × X6 (0.82). T20: 4 × X6 roll + 4 × X4 pitch + 2 × X8 (1.33) + 2 × X6 hind flexion |
| Frame, flexion linkages, bearings, feet and load cells | 2.70 | 3.03 | 3.38 | PR frame + 0.15 / 0.50 kg for linkages and larger mounts (est.) + 0.18 kg foot load cells and cartridges (est.) |
| Moving head | 1.00 | 1.33 | 1.51 | PR head + neck yaw XM540 (0.165), yaw bearing, beak XM430 (0.082), linkage, lights; T20 adds 2 × XL330 eyelids and a 0.15 kg heavier frill |
| Tail (shell, core, servos) | 0.38 | 0.41 | 0.49 | PR tail shell 0.325 + 1 or 2 × XM430 (0.082 each) |
| Body and limb shells | 0.57 | 0.85 | 1.25 | MJF PA11 + paint (est.; full limb covers at T20) |
| Compute and sensors | 0.75 | 0.62 | 0.62 | Jetson kit 0.174, NVMe, tray, Teensy carrier, OAK-D Pro W 0.091 + cable, IMU, ToF, I/O carriers |
| Battery | 0.70 | 1.01 | 1.01 | 2 × G-Tech 6S 3300 (0.506 each) |
| Power, safety chain, wiring, cooling | 0.90 | 1.20 | 1.55 | Contactor, PNOZ s5, wireless receiver, power board and clamp resistors, DC-DC rails, harness, fans; T20 adds the Orion BMS (est.) |
| **Listed parts** | 13.16 | 14.61 | 18.83 | |
| **Reserve** | 1.84 | 0.39 | 1.17 | Kept for weigh-in growth (shell, cable and bracket mass always grows) |
| **Total** | **15.00** | **15.00** | **20.00** | |
| Front-axle share (listed parts) | 55.2 % (with the reserve block) | 53.6 % | 55.2 % | Moving the battery 100 mm aft lowers the share by about 1.1 points (T15) or 0.8 points (T20) |

Notes:
- The PR's 55.2 % included its 1.84 kg reserve block at +118 mm. Without it, the PR's own split is about 53.4 %.
- T15 at 53.6 % is below the 55 % design point, which lowers the front loads.
- The all-X8 T20 variant would list 19.86 kg (0.14 kg reserve) at 52.8 % front.
- **[F3] [F11]** The reserves are before the wireless-stop receiver, whose mass is not published (an industrial box of this kind may weigh several hundred grams), and before about 0.1–0.2 kg of small added parts (fans, the 5 V rail, foot-sensor links, the BMS current sensor). T20's 1.17 kg covers them; T15's 0.39 kg may not, which is one more reason to prefer T20.
- The review checked the sums: 14.61 kg and 18.83 kg listed, 15.00 kg and 20.00 kg closed, and the T20 drive mass 9.02 kg (4 × 0.82 + 4 × 0.36 + 2 × 1.33 + 2 × 0.82).

## 7. Shared with the T. rex, which stays on the roadmap

The owner keeps the T. rex as the flagship for end 2027 / early 2028. This list is built so the Triceratops spend also prepares the T. rex.

**Same drive family and SKUs:**
- The X4-36-L and X6-60 are the T. rex's exact drives: 10 × X4 + 2 × X6 in 75-M-H-10.
- The X8-150 adds one larger size from the same maker, with the same protocol, tools and firmware.
- Spare drives serve both robots.

**Same compute and control:**
- Jetson Orin Nano Super, the Teensy 4.1 carrier and its CAN firmware, the PCAN-USB FD, and the PLEM option.

**Same head parts and battery:**
- The XM540-W270 neck servo is the T. rex's neck part.
- The G-Tech 6S 3300 is the T. rex's incumbent pack: the T. rex used one; the Triceratops uses two in series.

**Same safety, power and bench gear:**
- The safety chain design (Pilz Cat 1 stop, bidirectional contactor, watchdog).
- The tether and charger.
- The single-leg bench, torque rig, catch gantry and thermal camera.
- The wireless stop transmitter can serve both robots if a second receiver pairs with it. Confirm with Cattron or FORT.

**One open change for the T. rex:** its X6 hip pitch is a 48 V drive that was speed-limited at 6S. Moving the T. rex to the same 12S bus would unify the power hardware. Its X4s are inside their 20–55 V window, but on 12S they are capped at their 83 rpm rated speed **[F1]**, so the T. rex swing speeds must be re-checked against 83 rpm, not against a voltage-scaled no-load speed. This is a suggestion; it needs a T. rex re-check when that work resumes.

**Suggested timing, for the plan:**
- **Q1 2027:** the single-foreleg bench (this list's phase-1 buy).
- **Q2 2027:** the four-leg walk under the catch gantry.
- **Q3 2027:** the museum demo.
- Restart T. rex simulation work once the Triceratops passes its four-leg walking gate (about Q2 2027). Buy T. rex hardware after the Q3 2027 demo, reusing the bench, power, safety and compute above.

**[F13]** Buying T. rex hardware only after the Q3 2027 demo leaves one or two quarters before the end-2027 / early-2028 flagship milestone, and these drives were backordered at several resellers on 18 Nov 2026. If the milestone is a walking T. rex, resume its simulation in Q1 2027 and order its long-lead drives (10 × X4 + 2 × X6) in Q2 2027, once the Triceratops foreleg bench has confirmed the drive family. If it is a renders-and-roadmap milestone, the timing above is fine.

This is the lead's call. It is listed only because the shared purchases assume it.

## 8. Open questions to settle on the bench

1. **Which X6 ships for $428:** the V4 X6-60 (60 N·m, 0.82 kg) or the V4.1 X6-50-L (45 / 50 N·m, 0.78 kg)? Measure torque at 10 rpm and the stall torque at 42 V and 49 V. Both pass their roles here; the X6-50-L leaves less peak margin (T15 flexion use 0.73).
2. **Heat in the real duty:**
   - Run the X8 front flexion and the X6 hind flexion through the museum show loop (walk, held paw, stand) for 30 min, at 25 °C and at 35 °C, inside a closed shell.
   - Run the X4 roll and pitch in a 2-minute trot.
   - The 90-minute maker equilibria (X4 7.5, X6 12, X8 34 N·m) are open-air figures.
3. **X6-80 (30:1):** get a price and its temperature-rise table. If it holds about 18 N·m continuous, it replaces the front X8 and saves about 1.0 kg.
4. **Bus timing:**
   - The p99 command-to-torque delay with 3 drives per CAN bus at 500 Hz and 1 kHz.
   - Whether the shipped -E drives run EtherCAT from the Jetson.
   - What PLEM needs.
5. **Transmission:** four-bar stiffness, backlash, transmission angle over the full range, and rod-end wear; rod ends, crank pins and the output bearing pair against the 40° pulls (1.16–1.82 kN) **[F2]**. If a belt is used, its working tension against 0.75–1.2 kN, plus stretch and creep.
6. **Regen:** the bus rise in a hard stop and a controlled sit with the ideal diodes blocking the pack. Check that the 52.5 V clamp holds the bus below 55 V and that the resistors stay cool.
7. **Stop chain:**
   - Set the PNOZ delay so the sit completes before power drops.
   - Measure the wireless stop's latency and range in a hall with crowds (RF).
   - Run a dropout test with the robot on the catch gantry.
8. **Foot cells:** the FX29 overload rating against 300–450 N drop loads, the cartridge hard stop, and drift with temperature.
9. **Weigh-in:**
   - Measure the front share at each foot, against a 55 % design point.
   - Hind flexion heat fails at 50 % front.
   - Front pitch is thin above about 57 %.
10. **Power draw and run time at 12S** (150–280 W assumed), and the tether priority switchover under load.
11. **Shell mass and heat:** the actual panel mass against the 0.85 / 1.25 kg estimates, and the drive temperatures with the shell fitted.
12. **Noise:** gear whine in a quiet hall (dB at 1 m), and the soles' grip on terrazzo, wood and carpet.
13. **Landed price and lead time:** check whether myactuator.us prices include duty. Several resellers listed the X4, X6 and X8 as backordered on 18 Nov 2026, so order the phase-1 set early. Other long leads found by the review: FX29 cells 21–23 weeks, the BF1 fuse 16–23 weeks, the RSP-1000-48 about 12 weeks; robotis.us lists the XM540-W270-R and XM430-W350-R sold out; NVIDIA's dev kits were out of stock after the July price rise. **[F5] [F7] [F8]**
14. **X4 above 24 V [F1]:** ask MyActuator whether the rated-speed cap above rated voltage applies to the V4.1 X4-36-L, and measure its speed and heat at 42–50 V on the bench. If the cap holds, the 0.30 m/s walk is the X4 limit.
15. **Wireless stop [F3]:** what the $1,900 Safe-D-Stop kit contains (DC receiver or not), the receiver's mass, mounting and supply, and whether one transmitter can pair with a second receiver for the T. rex.
16. **Foot-sensor bus [F5]:** the FX29's I2C address and overload rating from TE's datasheet, and an I2C error-rate test on the leg harness with the drives running.
17. **Museum acceptance:** what the host museum needs before a live demo (risk assessment, insurance, barrier layout, battery charging rules on site).

## 9. Sources

Repo (read 18 Nov 2026):
- PR #669 files and the PR #669 survey.
- `hardware/trex_form_study/designlab/data/actuators.json` and `parts.json`. These are list prices and specs read from maker or distributor pages on 7 Nov 2026; each entry carries its URL:
  - myactuator.us product pages for the X4-36-L ($269), X6-60 ($428) and X8-150-E-03 ($725);
  - robotis.us;
  - Luxonis docs;
  - pololu.com #3419 and #5677;
  - peak-system.com;
  - genstattu.com 6S 3300;
  - Seeed (RobStride);
  - cubemars.com.
- `docs/hardware/RAPTOR_SCALING_AND_ALTERNATIVES.md` on main, prices last checked 24 Jul 2026: NVIDIA Jetson ($249), Mean Well RSP-1000-48 ($267.90), Junsi iCharger X12 ($179.99), Altran AREVS150-BAN ($54.45), Littelfuse BF1 ($4.70), Schneider XB4BS8445 ($45–55), Adafruit BNO085 ($24.95), Chargery BS12 ($24.50), Adafruit DS18B20 ($14.95), Holybro PM08-CAN ($85).
- `hardware/trex_form_study/stage2/jetson_plem/BUILD_BUDGET_75_JETSON.md` (T. rex budget lines).

Web search, 18 Nov 2026 (result summaries; the listing pages were not opened):
- MyActuator X6-60, RobotShop, $741.84 including tariffs: https://www.robotshop.com/products/myactuator-rmd-x-series-bldc-actuator-201-planetary-reduction-ethercat-communication-60-nm-peak-torque
- MyActuator X4-36-L, aifitlab, $400, backordered: https://aifitlab.com/products/myactuator-rmd-x4-36-l-motor
- MyActuator X4-36-E, RobotShop, $620 / $603.12 including tariffs: https://www.robotshop.com/products/myactuator-rmd-x4-p36-36-e-bldc-actuator-361-planetary-reduction-ethercat
- MyActuator X8-150, aifitlab, $645, backordered: https://aifitlab.com/products/myactuator-rmd-x8-150-motor
- RobStride 03, Seeed datasheet, $225, updated 21 Oct 2026: https://files.seeedstudio.com/Bazaar/product_pdf/100081690.pdf
- Unitree GO-M8010-6, $369: https://shop.unitree.com/products/go1-motor
- CubeMars AK80-64 with driver, OpenELAB, $1,227.99: https://openelab.com/products/cubemars-ak80-64-kv80-actuator
- Seeed reComputer J4012 (Orin NX 16 GB), $1,099.99, price tracker updated 23 Apr 2026: https://specpicks.com/product/B0DSZRGXRF
- Luxonis OAK-D Pro W, $529: https://shop.luxonis.com/products/oak-d-pro-w
- Xsens MTi-3 development kit, Arrow, $417.06: https://www.arrow.com/en/products/mti-3-5a-dk/xsens-technologies-bv
- Xsens MTi-630-DK, Cytech, $1,618.81 reference price: https://www.cytechsystems.com/de/product/mti-630-dk
- TE FX29K0-100A-0100-L, DigiKey $36.43–49.69 via Findchips: https://www.findchips.com/detail/fx29k0-100a-0100-l/TE-Connectivity. The $43.57 single-unit figure is a zenode.ai DigiKey reference from the same search.
- Cattron/OTE Safe-D-Stop, Hoosier Crane, $1,900: https://www.hoosiercrane.com/cattron-safe-d-stop-wireless-emergency-stop
- FORT Robotics wireless e-stop (no price shown): https://store.clearpathrobotics.com/products/wireless-e-stop
- Pilz PNOZ s5 C 24VDC (751105), Pilz eShop, EUR 358: https://www.pilz.com/eshop/p/751105
- Pilz PNOZ s5 C 24VDC (751105), Zoro, $611.99: https://www.zoro.com/pilz-safety-relay-751105/i/G923561920/
- Orion BMS Jr 2, 16 cells with CAN, $440 MSRP: https://www.orionbms.com/?p=2882
- Orion BMS Jr 2, reseller, $656: https://reach.dog/shop/evolve-electrics/products/bms-orion-jr-02
- LTC4417, 2.5–36 V (ADI brochure): https://www.mouser.lt/pdfdocs/IDP-web.pdf
- LTC4359, 4–80 V: https://www.analog.com/en/products/ltc4359.html
- Pololu D42V110F12, 12–60 V input: https://www.pololu.com/product/5677/specs
- Gens ace 6S 4000 (an option, not chosen): $109.99 at https://horizonhobby.com/product/22.8v-4000mah-6s-60c-g-tech-lipo-battery-xt90/GEA406S60X9GT.html; 692 g at https://gensace.de/products/gens-ace-g-tech-4000mah-22-2v-60c-6s1p-lipo-battery-pack-with-ec5-plug
- Teensy 4.1, US retail: $34.95 at https://www.solarbotics.com/?p=128491; $31.99 at Micro Center
- ROBOTIS XM540-W270-T, $482.89: https://www.robotis.us/xm/
- FUTEK TFF400, about EUR 2,376 at Elkome: https://shop.elkome.com/en/sensors/force-and-torgue-sensors/vaantomomenttianturi
- AMC SRST shunt regulators, fixed 50 / 55 / 70 V clamps (neither 50 V nor 55 V suits a 12S bus, hence the custom 52.5 V clamp): https://pdf.directindustry.com/pdf/advanced-motion-controls/srst400/11689-355515.html

Reviewer's web searches, 18 Nov 2026 (result summaries; listing pages not opened; myactuator.us and openelab.com could not be fetched from this session):
- X4 family 24 V rated, 20–55 V range, "above rated voltage do not exceed rated speed" (reseller text): https://openelab.io/products/myactuator-rmd-x4-p36-36-e-l
- X4-36 resellers: €349.95 OpenELAB, $400 aifitlab (backordered), $620 RobotShop incl. tariffs: https://aifitlab.com/products/myactuator-rmd-x4-36-l-motor
- X6-60: $741.84 RobotShop incl. tariffs (2 left); $490 aifitlab and Robotics Center (backorder): https://www.robotshop.com/products/myactuator-rmd-x-series-bldc-actuator-201-planetary-reduction-ethercat-communication-60-nm-peak-torque
- X8-150-E: $645 aifitlab (backordered): https://aifitlab.com/products/myactuator-rmd-x8-150-motor
- RobStride 03: $225 Seeed datasheet (updated 2026-10-21), 880 g, 15–60 V: https://files.seeedstudio.com/Bazaar/product_pdf/100081690.pdf
- NVIDIA Jetson price rise, Orin Nano Super kit $249 -> $399, Orin NX 16 GB module $599 -> $999 (1 ku), kits out of stock: https://www.cnx-software.com/2026/07/22/nvidia-increases-the-price-of-jetson-modules-and-devkits-by-up-to-101/
- reComputer J4012 (Orin NX 16 GB), RobotShop $1,364.29 "incl. tariffs (in USA)" (listing undated; store site and currency not confirmed): https://ca.robotshop.com/products/recomputer-j4012-edge-ai-computer-w-nvidia-jetson-orin-nx-16gb ; Seeed's reComputer Robotics J4012 $1,439, sold out
- Jetson Orin Nano kit DC input (forum figures 7–20 V / 9–20 V; 8 V minimum for Super mode): https://forums.developer.nvidia.com/t/jetson-orin-nano-input-voltage/298625
- ROBOTIS XM540-W270-R and XM540-W150-R $494.39, XM540-W150-T $482.89 (robotis.us XM listing): https://www.robotis.us/xm/ ; W270-R sold out, extended lead: https://www.robotis.us/dynamixel-xm540-w270-r/
- ROBOTIS XM430-W350-R $333.39 (sold out; RobotShop $333.39 incl. tariffs): https://robotis.us/dynamixel-xm430-w350-r
- ROBOTIS XL330-M288-T $27.49 (sold out), 3.7–6 V, TTL: https://robotis.us/dynamixel-xl330-m288-t ; https://emanual.robotis.com/docs/en/dxl/x/xl330-m288/
- Pololu D42V110F12, $59.95, 12–60 V in, 12 V 9 A, rationed: https://www.pololu.com/product/5677
- Pololu VL53L8CX carrier #3419, $24.95: https://www.pololu.com/product/3419
- PCAN-USB FD IPEH-004022, DigiKey $379 (+$15 shipping), Gridconnect $347: https://azcus.digikey.com/en/products/detail/peak-system/IPEH-004022/12326289
- Cattron Safe-D-Stop, PL d (EN ISO 13849), $1,900 at Hoosier Crane (SAFED-V1 10R-ASO-AC F): https://www.hoosiercrane.com/cattron-safe-d-stop-wireless-emergency-stop ; XBMCU machine-side unit, 9–36 V DC, Cat 3 / PL d inputs: https://www.cattron.com/products/xbmcu/
- Pilz PNOZ s5 C 751105, Pilz eShop €358; RS UK £308.40: https://www.pilz.com/eshop/p/751105 (the Zoro $611.99 listing was not re-surfaced)
- Schneider XB4BS8445 = 1 NO + 1 NC; XB4BS8444 = 2 NC: https://www.se.com/us/en/product/XB4BS8445 ; https://www.se.com/mk/en/product/XB4BS8444/emergency-stop-push-button-harmony-xb4-metal-red-mushroom-40mm-22mm-trigger-latching-turn-to-release-2nc
- Altran AREVS150-BAN, 200 A, 12 V coil, aux contact, bidirectional; AREVS150-BN coil 461 mA: https://www.digikey.sg/en/product-highlight/a/altran-magnetics/arevs150-series-high-voltage-dc-contactors
- Littelfuse BF1 142.5631.6102, 100 A / 58 V MIDI-style bolt-down, DigiKey lead 16 weeks: https://www.littelfuse.com/products/fuses-overcurrent-protection/fuses/automotive-fuses/bolt-down-fuses/midi/bf1-58v/142-5631-6102
- Gens ace G-Tech 3300 6S 60C, $107.19 (BuddyRC; Horizon out of stock), 506 g ±20 g, 138 × 42 × 43 mm, EC5: https://www.buddyrc.com/products/gens-ace-3300mah-22-2v-60c-6s1p-lipo-battery-pack-ec5-plug-1
- Orion BMS Jr 2, 16-cell CAN $440 MSRP; 1–16 cells, 60 V max; current sensor optional but strongly recommended; 2 digital outputs: https://www.orionbms.com/?p=2882 ; https://www.orionbms.com/downloads/documents/orionjr2_specifications.pdf
- Holybro PM08-CAN, 2–14S: https://holybro.com/products/dronecan-pm08-power-module-14s-200a
- Mean Well RSP-1000-48, $267.90 (DigiKey via aggregator, Bravo Electro), trim 40–110 %, OVP 56.6–66.2 V, 12-week lead: https://www.bravoelectro.com/rsp-1000-48.html
- Junsi iCharger X12, $179.99, 11–53 V input, 1–12S: https://buddyrc.com/products/icharger-x12-1100w-30a-12s-balance-battery-charger
- Luxonis OAK-D Pro W (A00573), $529 at shop.luxonis.com; resellers $449–499: https://shop.luxonis.com/products/oak-d-pro-w
- Xsens MTi-3-5A-DK, Arrow $417.06; MTi-3-8A7G6-DK about $450 at DigiKey (aggregator): https://www.arrow.com/en/products/mti-3-5a-dk/xsens-technologies-bv
- TE FX29K0-100A-0100-L, DigiKey $43.57 (zenode), 21–23-week lead (findchips, onlinecomponents); digital supply 2.7–5.5 V: https://zenode.ai/parts/fdyd0p46z2 ; https://www.farnell.com/datasheets/3626263.pdf

## 10. What I did not do

- **Vendor pages:** I could not open them. Fetches to myactuator.us and seeedstudio.com failed because those hosts are outside this session's allowed network, and I stopped trying further vendor sites. To re-verify prices live, those hosts have to be allowed in the session's network settings.
  - Current prices therefore come from the repo catalog (7 Nov 2026), the repo's main hardware doc (24 Jul 2026), and web-search summaries (18 Nov 2026).
- **No quotes:** I requested none, and none of the estimates is a quote. That covers the CNC frame, the MJF shells, the paint, the custom PCBs, the rigs, the wireless stop above the one listing, PLEM and W-RC.
- **Mechanical design:** no CAD, no linkage synthesis, and no check of belt-maker tables.
- **Masses:** the safety and power parts (contactor, relay, receiver, BMS) and the shells are estimated.
- **Simulation:** none run.
- **The 60 cm companion** is not costed.
- **Unchecked figures:**
  - ROBOTIS U2D2, power hub and XL330 prices;
  - -R variant pricing (assumed equal to -T);
  - Mean Well RSP-1000-48 output trim range (confirm it covers 48–50 V);
  - FX29 overload rating;
  - Orion Jr 2 mass.
- **Review (18 Nov 2026):** the reviewer could not open myactuator.us or openelab.com (outside the session's allowed network; they have to be allowed in the session's network settings to re-verify live), opened no vendor page, and used search-result summaries only. The X4 rated-speed cap rests on reseller text, not on the maker's sheet. The receiver, cooling, tools, transport-case, foot-link, 5 V-rail and current-sensor prices are estimates. The ROBOTIS -R prices, the U2D2 hub's 3.5–24 V range, the XL330 price and the RSP-1000-48 trim range are now checked (by search); the U2D2 price and the Pilz Zoro listing are not.

## Appendix A: `sizing.py` output

<details><summary>Full output (230 lines)</summary>

```
== Geometry check (fore leg, ankle pin 45 mm, hip 400 mm) ==
 foot x  -100 mm: elbow Fz-arm  87.1 mm, Fx-arm 257.9 mm
 foot x   -75 mm: elbow Fz-arm 105.4 mm, Fx-arm 249.4 mm
 foot x   -25 mm: elbow Fz-arm 137.0 mm, Fx-arm 229.8 mm
 foot x    +0 mm: elbow Fz-arm 150.2 mm, Fx-arm 219.2 mm
 foot x   +25 mm: elbow Fz-arm 161.5 mm, Fx-arm 208.7 mm
 foot x   +75 mm: elbow Fz-arm 179.1 mm, Fx-arm 189.3 mm
 foot x  +100 mm: elbow Fz-arm 185.4 mm, Fx-arm 181.1 mm

== Swing joint speeds (joint side, peak, rpm) ==
 walk 0.30 m/s (PR: 1.5 Hz, beta .75, sweep 150, lift 32)   pitch  59.9  flexion  50.1
 walk 0.40 m/s (sweep 200, same cadence)                    pitch  77.0  flexion  58.5
 trot 0.50 m/s (2.0 Hz, beta .55, lift 45)                  pitch  45.6  flexion  46.9
 trot 0.60 m/s (2.0 Hz, beta .50, lift 50)                  pitch  45.3  flexion  47.0

================ m = 15 kg, front share 0.50 ================
 weight 147.2 N: front axle 73.6 N, rear 73.6 N; CoM 325 mm ahead of hind feet
 WALK lone front foot  73.6 N  x1.5: flexion(joint) touchdown  21.9 / window  20.1 / pad-edge  27.9; pitch  12.7; roll  13.2 N·m
 WALK lone rear foot  73.6 N  x1.5: flexion(joint) touchdown  21.9 / window  20.1 / pad-edge  27.9; pitch  12.7; roll  13.2 N·m
 TROT b=0.55 v=0.5 front: Fpk 105.1 N, sweep 138 mm; peak x1.25: flex  22.1, pitch  11.1, roll  15.8; RMS flex  9.8, pitch  4.7, roll  6.6
 TROT b=0.55 v=0.5 rear: Fpk 105.1 N, sweep 138 mm; peak x1.25: flex  22.1, pitch  11.1, roll  15.8; RMS flex  9.8, pitch  4.7, roll  6.6
 TROT b=0.5 v=0.6 front: Fpk 115.6 N, sweep 150 mm; peak x1.25: flex  24.4, pitch  12.5, roll  17.3; RMS flex 10.3, pitch  5.0, roll  6.9
 TROT b=0.5 v=0.6 rear: Fpk 115.6 N, sweep 150 mm; peak x1.25: flex  24.4, pitch  12.5, roll  17.3; RMS flex 10.3, pitch  5.0, roll  6.9
 CONT walk RMS (front): flex 7.2, pitch 2.2, roll 4.5 N·m (joint)
 CONT stand 4 feet: flex 5.5 joint / 6.1 motor; roll 2.9.  Held paw: flex 11.0 joint / 12.3 motor; roll 5.9

================ m = 15 kg, front share 0.55 ================
 weight 147.2 N: front axle 80.9 N, rear 66.2 N; CoM 358 mm ahead of hind feet
 WALK lone front foot  80.9 N  x1.5: flexion(joint) touchdown  24.0 / window  22.1 / pad-edge  30.7; pitch  14.0; roll  14.6 N·m
 WALK lone rear foot  66.2 N  x1.5: flexion(joint) touchdown  19.7 / window  18.1 / pad-edge  25.1; pitch  11.4; roll  11.9 N·m
 TROT b=0.55 v=0.5 front: Fpk 115.6 N, sweep 138 mm; peak x1.25: flex  24.4, pitch  12.2, roll  17.3; RMS flex 10.8, pitch  5.1, roll  7.3
 TROT b=0.55 v=0.5 rear: Fpk  94.6 N, sweep 138 mm; peak x1.25: flex  19.9, pitch  10.0, roll  14.2; RMS flex  8.8, pitch  4.2, roll  5.9
 TROT b=0.5 v=0.6 front: Fpk 127.1 N, sweep 150 mm; peak x1.25: flex  26.9, pitch  13.8, roll  19.1; RMS flex 11.3, pitch  5.5, roll  7.6
 TROT b=0.5 v=0.6 rear: Fpk 104.0 N, sweep 150 mm; peak x1.25: flex  22.0, pitch  11.3, roll  15.6; RMS flex  9.2, pitch  4.5, roll  6.2
 CONT walk RMS (front): flex 8.0, pitch 2.5, roll 5.0 N·m (joint)
 CONT stand 4 feet: flex 6.1 joint / 6.8 motor; roll 3.2.  Held paw: flex 12.2 joint / 13.5 motor; roll 6.5

================ m = 15 kg, front share 0.60 ================
 weight 147.2 N: front axle 88.3 N, rear 58.9 N; CoM 390 mm ahead of hind feet
 WALK lone front foot  88.3 N  x1.5: flexion(joint) touchdown  26.2 / window  24.2 / pad-edge  33.5; pitch  15.2; roll  15.9 N·m
 WALK lone rear foot  58.9 N  x1.5: flexion(joint) touchdown  17.5 / window  16.1 / pad-edge  22.3; pitch  10.2; roll  10.6 N·m
 TROT b=0.55 v=0.5 front: Fpk 126.1 N, sweep 138 mm; peak x1.25: flex  26.6, pitch  13.3, roll  18.9; RMS flex 11.8, pitch  5.6, roll  7.9
 TROT b=0.55 v=0.5 rear: Fpk  84.1 N, sweep 138 mm; peak x1.25: flex  17.7, pitch   8.9, roll  12.6; RMS flex  7.8, pitch  3.7, roll  5.3
 TROT b=0.5 v=0.6 front: Fpk 138.7 N, sweep 150 mm; peak x1.25: flex  29.3, pitch  15.0, roll  20.8; RMS flex 12.3, pitch  6.0, roll  8.3
 TROT b=0.5 v=0.6 rear: Fpk  92.5 N, sweep 150 mm; peak x1.25: flex  19.6, pitch  10.0, roll  13.9; RMS flex  8.2, pitch  4.0, roll  5.5
 CONT walk RMS (front): flex 8.7, pitch 2.7, roll 5.4 N·m (joint)
 CONT stand 4 feet: flex 6.6 joint / 7.4 motor; roll 3.5.  Held paw: flex 13.3 joint / 14.7 motor; roll 7.1

================ m = 20 kg, front share 0.50 ================
 weight 196.2 N: front axle 98.1 N, rear 98.1 N; CoM 325 mm ahead of hind feet
 WALK lone front foot  98.1 N  x1.5: flexion(joint) touchdown  29.1 / window  26.8 / pad-edge  37.2; pitch  16.9; roll  17.7 N·m
 WALK lone rear foot  98.1 N  x1.5: flexion(joint) touchdown  29.1 / window  26.8 / pad-edge  37.2; pitch  16.9; roll  17.7 N·m
 TROT b=0.55 v=0.5 front: Fpk 140.1 N, sweep 138 mm; peak x1.25: flex  29.5, pitch  14.8, roll  21.0; RMS flex 13.1, pitch  6.2, roll  8.8
 TROT b=0.55 v=0.5 rear: Fpk 140.1 N, sweep 138 mm; peak x1.25: flex  29.5, pitch  14.8, roll  21.0; RMS flex 13.1, pitch  6.2, roll  8.8
 TROT b=0.5 v=0.6 front: Fpk 154.1 N, sweep 150 mm; peak x1.25: flex  32.6, pitch  16.7, roll  23.1; RMS flex 13.7, pitch  6.7, roll  9.2
 TROT b=0.5 v=0.6 rear: Fpk 154.1 N, sweep 150 mm; peak x1.25: flex  32.6, pitch  16.7, roll  23.1; RMS flex 13.7, pitch  6.7, roll  9.2
 CONT walk RMS (front): flex 9.7, pitch 3.0, roll 6.0 N·m (joint)
 CONT stand 4 feet: flex 7.4 joint / 8.2 motor; roll 3.9.  Held paw: flex 14.7 joint / 16.4 motor; roll 7.8

================ m = 20 kg, front share 0.55 ================
 weight 196.2 N: front axle 107.9 N, rear 88.3 N; CoM 358 mm ahead of hind feet
 WALK lone front foot 107.9 N  x1.5: flexion(joint) touchdown  32.1 / window  29.5 / pad-edge  41.0; pitch  18.6; roll  19.4 N·m
 WALK lone rear foot  88.3 N  x1.5: flexion(joint) touchdown  26.2 / window  24.2 / pad-edge  33.5; pitch  15.2; roll  15.9 N·m
 TROT b=0.55 v=0.5 front: Fpk 154.1 N, sweep 138 mm; peak x1.25: flex  32.5, pitch  16.3, roll  23.1; RMS flex 14.4, pitch  6.8, roll  9.7
 TROT b=0.55 v=0.5 rear: Fpk 126.1 N, sweep 138 mm; peak x1.25: flex  26.6, pitch  13.3, roll  18.9; RMS flex 11.8, pitch  5.6, roll  7.9
 TROT b=0.5 v=0.6 front: Fpk 169.5 N, sweep 150 mm; peak x1.25: flex  35.8, pitch  18.3, roll  25.4; RMS flex 15.1, pitch  7.4, roll 10.1
 TROT b=0.5 v=0.6 rear: Fpk 138.7 N, sweep 150 mm; peak x1.25: flex  29.3, pitch  15.0, roll  20.8; RMS flex 12.3, pitch  6.0, roll  8.3
 CONT walk RMS (front): flex 10.6, pitch 3.3, roll 6.6 N·m (joint)
 CONT stand 4 feet: flex 8.1 joint / 9.0 motor; roll 4.3.  Held paw: flex 16.2 joint / 18.0 motor; roll 8.6

================ m = 20 kg, front share 0.60 ================
 weight 196.2 N: front axle 117.7 N, rear 78.5 N; CoM 390 mm ahead of hind feet
 WALK lone front foot 117.7 N  x1.5: flexion(joint) touchdown  35.0 / window  32.2 / pad-edge  44.7; pitch  20.3; roll  21.2 N·m
 WALK lone rear foot  78.5 N  x1.5: flexion(joint) touchdown  23.3 / window  21.5 / pad-edge  29.8; pitch  13.5; roll  14.1 N·m
 TROT b=0.55 v=0.5 front: Fpk 168.1 N, sweep 138 mm; peak x1.25: flex  35.4, pitch  17.7, roll  25.2; RMS flex 15.7, pitch  7.5, roll 10.6
 TROT b=0.55 v=0.5 rear: Fpk 112.1 N, sweep 138 mm; peak x1.25: flex  23.6, pitch  11.8, roll  16.8; RMS flex 10.4, pitch  5.0, roll  7.0
 TROT b=0.5 v=0.6 front: Fpk 184.9 N, sweep 150 mm; peak x1.25: flex  39.1, pitch  20.0, roll  27.7; RMS flex 16.4, pitch  8.0, roll 11.1
 TROT b=0.5 v=0.6 rear: Fpk 123.3 N, sweep 150 mm; peak x1.25: flex  26.1, pitch  13.3, roll  18.5; RMS flex 11.0, pitch  5.4, roll  7.4
 CONT walk RMS (front): flex 11.6, pitch 3.6, roll 7.2 N·m (joint)
 CONT stand 4 feet: flex 8.8 joint / 9.8 motor; roll 4.7.  Held paw: flex 17.7 joint / 19.6 motor; roll 9.4

== Requirement summary per axis (front leg governs; worst of the cases above) ==
 m=15: flexion  design peak  26.9 N·m (joint), cont RMS 11.3, held pose 12.2, peak speed  58.5 rpm
 m=15: pitch    design peak  14.0 N·m (joint), cont RMS  5.5, held pose  0.0, peak speed  77.0 rpm
 m=15: roll     design peak  19.1 N·m (joint), cont RMS  7.6, held pose  6.5, peak speed  30.0 rpm
 m=20: flexion  design peak  35.8 N·m (joint), cont RMS 15.1, held pose 16.2, peak speed  58.5 rpm
 m=20: pitch    design peak  18.6 N·m (joint), cont RMS  7.4, held pose  0.0, peak speed  77.0 rpm
 m=20: roll     design peak  25.4 N·m (joint), cont RMS 10.1, held pose  8.6, peak speed  30.0 rpm

== Candidate check, 12S bus (42 V minimum), front leg, f = 0.55 ==
 usage = need / (0.70 x vendor peak) | cont/thermal-eq | speed / (0.70 x no-load at 42 V); <= 1.00 passes

 m=15 roll (transmission 1.0:1):
   MyActuator X4-36-L       motor peak need  19.1  use_pk 0.91  use_cont 1.01  use_speed 0.22  0.36 kg  fail
   MyActuator X6-60 (V4)    motor peak need  19.1  use_pk 0.45  use_cont 0.63  use_speed 0.28  0.82 kg  PASS
   MyActuator X6-50-L       motor peak need  19.1  use_pk 0.61  use_cont 0.63  use_speed 0.28  0.78 kg  PASS
   RobStride 02             motor peak need  19.1  use_pk 1.60  use_cont  n/a  use_speed 0.12  0.41 kg  fail (no thermal table)
   RobStride 03             motor peak need  19.1  use_pk 0.45  use_cont  n/a  use_speed 0.25  0.88 kg  PASS (no thermal table)
   CubeMars AK70-10         motor peak need  19.1  use_pk 1.10  use_cont  n/a  use_speed 0.10  0.62 kg  fail (no thermal table)
   CubeMars AK80-9 V3       motor peak need  19.1  use_pk 1.24  use_cont  n/a  use_speed 0.09  0.49 kg  fail (no thermal table)

 m=15 pitch (transmission 1.0:1):
   MyActuator X4-36-L       motor peak need  14.0  use_pk 0.66  use_cont 0.74  use_speed 0.57  0.36 kg  PASS
   MyActuator X6-60 (V4)    motor peak need  14.0  use_pk 0.33  use_cont 0.46  use_speed 0.71  0.82 kg  PASS
   MyActuator X6-50-L       motor peak need  14.0  use_pk 0.44  use_cont 0.46  use_speed 0.71  0.78 kg  PASS
   RobStride 02             motor peak need  14.0  use_pk 1.17  use_cont  n/a  use_speed 0.31  0.41 kg  fail (no thermal table)
   RobStride 03             motor peak need  14.0  use_pk 0.33  use_cont  n/a  use_speed 0.64  0.88 kg  PASS (no thermal table)
   CubeMars AK70-10         motor peak need  14.0  use_pk 0.80  use_cont  n/a  use_speed 0.26  0.62 kg  PASS (no thermal table)
   CubeMars AK80-9 V3       motor peak need  14.0  use_pk 0.91  use_cont  n/a  use_speed 0.22  0.49 kg  PASS (no thermal table)

 m=15 flexion (transmission 1.0:1, eta .90):
   MyActuator X6-60 (V4)    motor peak need  29.9  use_pk 0.71  use_cont 1.13  use_speed 0.54  0.82 kg  fail
   MyActuator X6-50-L       motor peak need  29.9  use_pk 0.95  use_cont 1.13  use_speed 0.54  0.78 kg  fail
   MyActuator X6-80         motor peak need  29.9  use_pk 0.53  use_cont  n/a  use_speed 0.82  0.81 kg  PASS (no thermal table)
   MyActuator X8-150        motor peak need  29.9  use_pk 0.36  use_cont 0.40  use_speed 0.60  1.33 kg  PASS
   MyActuator X8-120 (V4)   motor peak need  29.9  use_pk 0.36  use_cont 0.40  use_speed 0.60  1.40 kg  PASS
   RobStride 03             motor peak need  29.9  use_pk 0.71  use_cont  n/a  use_speed 0.49  0.88 kg  PASS (no thermal table)
   RobStride 04             motor peak need  29.9  use_pk 0.36  use_cont  n/a  use_speed 0.48  1.42 kg  PASS (no thermal table)
   CubeMars AK80-64         motor peak need  29.9  use_pk 0.36  use_cont  n/a  use_speed 1.27  0.85 kg  fail (no thermal table)

 m=15 flexion (transmission 1.5:1, eta .90):
   MyActuator X6-60 (V4)    motor peak need  19.9  use_pk 0.47  use_cont 0.75  use_speed 0.81  0.82 kg  PASS
   MyActuator X6-50-L       motor peak need  19.9  use_pk 0.63  use_cont 0.75  use_speed 0.81  0.78 kg  PASS
   MyActuator X6-80         motor peak need  19.9  use_pk 0.36  use_cont  n/a  use_speed 1.22  0.81 kg  fail (no thermal table)
   MyActuator X8-150        motor peak need  19.9  use_pk 0.24  use_cont 0.26  use_speed 0.91  1.33 kg  PASS
   MyActuator X8-120 (V4)   motor peak need  19.9  use_pk 0.24  use_cont 0.26  use_speed 0.91  1.40 kg  PASS
   RobStride 03             motor peak need  19.9  use_pk 0.47  use_cont  n/a  use_speed 0.73  0.88 kg  PASS (no thermal table)
   RobStride 04             motor peak need  19.9  use_pk 0.24  use_cont  n/a  use_speed 0.72  1.42 kg  PASS (no thermal table)
   CubeMars AK80-64         motor peak need  19.9  use_pk 0.24  use_cont  n/a  use_speed 1.91  0.85 kg  fail (no thermal table)

 m=20 roll (transmission 1.0:1):
   MyActuator X4-36-L       motor peak need  25.4  use_pk 1.21  use_cont 1.35  use_speed 0.22  0.36 kg  fail
   MyActuator X6-60 (V4)    motor peak need  25.4  use_pk 0.61  use_cont 0.85  use_speed 0.28  0.82 kg  PASS
   MyActuator X6-50-L       motor peak need  25.4  use_pk 0.81  use_cont 0.85  use_speed 0.28  0.78 kg  PASS
   RobStride 02             motor peak need  25.4  use_pk 2.14  use_cont  n/a  use_speed 0.12  0.41 kg  fail (no thermal table)
   RobStride 03             motor peak need  25.4  use_pk 0.61  use_cont  n/a  use_speed 0.25  0.88 kg  PASS (no thermal table)
   CubeMars AK70-10         motor peak need  25.4  use_pk 1.46  use_cont  n/a  use_speed 0.10  0.62 kg  fail (no thermal table)
   CubeMars AK80-9 V3       motor peak need  25.4  use_pk 1.65  use_cont  n/a  use_speed 0.09  0.49 kg  fail (no thermal table)

 m=20 pitch (transmission 1.0:1):
   MyActuator X4-36-L       motor peak need  18.6  use_pk 0.89  use_cont 0.98  use_speed 0.57  0.36 kg  PASS
   MyActuator X6-60 (V4)    motor peak need  18.6  use_pk 0.44  use_cont 0.61  use_speed 0.71  0.82 kg  PASS
   MyActuator X6-50-L       motor peak need  18.6  use_pk 0.59  use_cont 0.61  use_speed 0.71  0.78 kg  PASS
   RobStride 02             motor peak need  18.6  use_pk 1.56  use_cont  n/a  use_speed 0.31  0.41 kg  fail (no thermal table)
   RobStride 03             motor peak need  18.6  use_pk 0.44  use_cont  n/a  use_speed 0.64  0.88 kg  PASS (no thermal table)
   CubeMars AK70-10         motor peak need  18.6  use_pk 1.07  use_cont  n/a  use_speed 0.26  0.62 kg  fail (no thermal table)
   CubeMars AK80-9 V3       motor peak need  18.6  use_pk 1.21  use_cont  n/a  use_speed 0.22  0.49 kg  fail (no thermal table)

 m=20 flexion (transmission 1.0:1, eta .90):
   MyActuator X6-60 (V4)    motor peak need  39.8  use_pk 0.95  use_cont 1.50  use_speed 0.54  0.82 kg  fail
   MyActuator X6-50-L       motor peak need  39.8  use_pk 1.26  use_cont 1.50  use_speed 0.54  0.78 kg  fail
   MyActuator X6-80         motor peak need  39.8  use_pk 0.71  use_cont  n/a  use_speed 0.82  0.81 kg  PASS (no thermal table)
   MyActuator X8-150        motor peak need  39.8  use_pk 0.47  use_cont 0.53  use_speed 0.60  1.33 kg  PASS
   MyActuator X8-120 (V4)   motor peak need  39.8  use_pk 0.47  use_cont 0.53  use_speed 0.60  1.40 kg  PASS
   RobStride 03             motor peak need  39.8  use_pk 0.95  use_cont  n/a  use_speed 0.49  0.88 kg  PASS (no thermal table)
   RobStride 04             motor peak need  39.8  use_pk 0.47  use_cont  n/a  use_speed 0.48  1.42 kg  PASS (no thermal table)
   CubeMars AK80-64         motor peak need  39.8  use_pk 0.47  use_cont  n/a  use_speed 1.27  0.85 kg  fail (no thermal table)

 m=20 flexion (transmission 1.5:1, eta .90):
   MyActuator X6-60 (V4)    motor peak need  26.6  use_pk 0.63  use_cont 1.00  use_speed 0.81  0.82 kg  fail
   MyActuator X6-50-L       motor peak need  26.6  use_pk 0.84  use_cont 1.00  use_speed 0.81  0.78 kg  fail
   MyActuator X6-80         motor peak need  26.6  use_pk 0.47  use_cont  n/a  use_speed 1.22  0.81 kg  fail (no thermal table)
   MyActuator X8-150        motor peak need  26.6  use_pk 0.32  use_cont 0.35  use_speed 0.91  1.33 kg  PASS
   MyActuator X8-120 (V4)   motor peak need  26.6  use_pk 0.32  use_cont 0.35  use_speed 0.91  1.40 kg  PASS
   RobStride 03             motor peak need  26.6  use_pk 0.63  use_cont  n/a  use_speed 0.73  0.88 kg  PASS (no thermal table)
   RobStride 04             motor peak need  26.6  use_pk 0.32  use_cont  n/a  use_speed 0.72  1.42 kg  PASS (no thermal table)
   CubeMars AK80-64         motor peak need  26.6  use_pk 0.32  use_cont  n/a  use_speed 1.91  0.85 kg  fail (no thermal table)

== 6S vs 12S speed check for the PR's drives (0.40 m/s walk, 0.60 m/s trot) ==
 6S (21.0 V min): MyActuator X4-36-L: no-load  97.1 rpm; trot 0.60 needs  45.3 (  47 %); walk 0.40 needs  77.0
 6S (21.0 V min): MyActuator X6-60 (V4): no-load  77.0 rpm; trot 0.60 needs  47.0 (  61 %); walk 0.40 needs  58.5
 12S (42.0 V min): MyActuator X4-36-L: no-load 194.2 rpm; trot 0.60 needs  45.3 (  23 %); walk 0.40 needs  77.0
 12S (42.0 V min): MyActuator X6-60 (V4): no-load 154.0 rpm; trot 0.60 needs  47.0 (  31 %); walk 0.40 needs  58.5

== Transmission loads (front flexion, design peak at the joint) ==
 m=15: joint  26.9 N·m -> belt/rod force at 20 mm crank 1.34 kN
 m=15: joint  26.9 N·m -> belt/rod force at 30 mm crank 0.90 kN
 m=15: joint  26.9 N·m -> belt/rod force at 35 mm crank 0.77 kN
 m=20: joint  35.8 N·m -> belt/rod force at 20 mm crank 1.79 kN
 m=20: joint  35.8 N·m -> belt/rod force at 30 mm crank 1.19 kN
 m=20: joint  35.8 N·m -> belt/rod force at 35 mm crank 1.02 kN

== CHOSEN BUILDS: usage ratios (<= 1.00 passes); peak vs 70 % of vendor peak, cont vs thermal equilibrium, speed vs 70 % no-load at 42 V ==

 T15 (15 kg)
   f=0.50 front: roll pk 0.83 th 0.92 v 0.22 | pitch pk 0.60 th 0.67 v 0.57 | flexion pk 0.50 th 0.79 v 0.71
   f=0.50 rear : roll pk 0.83 th 0.92 v 0.22 | pitch pk 0.60 th 0.67 v 0.57 | flexion pk 0.50 th 0.79 v 0.71
   f=0.55 front: roll pk 0.91 th 1.01 v 0.22 | pitch pk 0.66 th 0.74 v 0.57 | flexion pk 0.55 th 0.87 v 0.71
   f=0.55 rear : roll pk 0.74 th 0.83 v 0.22 | pitch pk 0.54 th 0.60 v 0.57 | flexion pk 0.45 th 0.71 v 0.71
   f=0.60 front: roll pk 0.99 th 1.11 v 0.22 | pitch pk 0.73 th 0.80 v 0.57 | flexion pk 0.60 th 0.94 v 0.71
   f=0.60 rear : roll pk 0.66 th 0.74 v 0.22 | pitch pk 0.48 th 0.54 v 0.57 | flexion pk 0.40 th 0.63 v 0.71

 T15 if the X6-50-L ships
   f=0.50 front: roll pk 0.83 th 0.92 v 0.22 | pitch pk 0.60 th 0.67 v 0.57 | flexion pk 0.66 th 0.79 v 0.71
   f=0.50 rear : roll pk 0.83 th 0.92 v 0.22 | pitch pk 0.60 th 0.67 v 0.57 | flexion pk 0.66 th 0.79 v 0.71
   f=0.55 front: roll pk 0.91 th 1.01 v 0.22 | pitch pk 0.66 th 0.74 v 0.57 | flexion pk 0.73 th 0.87 v 0.71
   f=0.55 rear : roll pk 0.74 th 0.83 v 0.22 | pitch pk 0.54 th 0.60 v 0.57 | flexion pk 0.60 th 0.71 v 0.71
   f=0.60 front: roll pk 0.99 th 1.11 v 0.22 | pitch pk 0.73 th 0.80 v 0.57 | flexion pk 0.80 th 0.94 v 0.71
   f=0.60 rear : roll pk 0.66 th 0.74 v 0.22 | pitch pk 0.48 th 0.54 v 0.57 | flexion pk 0.53 th 0.63 v 0.71

 T20 (20 kg)
   f=0.50 front: roll pk 0.55 th 0.77 v 0.28 | pitch pk 0.81 th 0.89 v 0.57 | flexion pk 0.43 th 0.48 v 0.60
   f=0.50 rear : roll pk 0.55 th 0.77 v 0.28 | pitch pk 0.81 th 0.89 v 0.57 | flexion pk 0.43 th 0.48 v 0.60
   f=0.55 front: roll pk 0.61 th 0.85 v 0.28 | pitch pk 0.89 th 0.98 v 0.57 | flexion pk 0.47 th 0.53 v 0.60
   f=0.55 rear : roll pk 0.50 th 0.69 v 0.28 | pitch pk 0.73 th 0.80 v 0.57 | flexion pk 0.39 th 0.43 v 0.60
   f=0.60 front: roll pk 0.66 th 0.92 v 0.28 | pitch pk 0.97 th 1.07 v 0.57 | flexion pk 0.52 th 0.58 v 0.60
   f=0.60 rear : roll pk 0.44 th 0.61 v 0.28 | pitch pk 0.64 th 0.71 v 0.57 | flexion pk 0.34 th 0.39 v 0.60

 T20 hind legs with X6 flexion at 1.3:1
   f=0.50 rear : roll pk 0.55 th 0.77 v 0.28 | pitch pk 0.81 th 0.89 v 0.57 | flexion pk 0.66 th 1.05 v 0.71
   f=0.55 rear : roll pk 0.50 th 0.69 v 0.28 | pitch pk 0.73 th 0.80 v 0.57 | flexion pk 0.60 th 0.94 v 0.71
   f=0.60 rear : roll pk 0.44 th 0.61 v 0.28 | pitch pk 0.64 th 0.71 v 0.57 | flexion pk 0.53 th 0.84 v 0.71

== Joint-side design figures used in the BOM (f = 0.55, front leg) ==
 m=15 roll    : design peak  19.1 N·m, continuous   7.6 N·m, peak speed  30.0 rpm
 m=15 pitch   : design peak  14.0 N·m, continuous   5.5 N·m, peak speed  77.0 rpm
 m=15 flexion : design peak  26.9 N·m, continuous  12.2 N·m, peak speed  58.5 rpm
 m=20 roll    : design peak  25.4 N·m, continuous  10.1 N·m, peak speed  30.0 rpm
 m=20 pitch   : design peak  18.6 N·m, continuous   7.4 N·m, peak speed  77.0 rpm
 m=20 flexion : design peak  35.8 N·m, continuous  16.2 N·m, peak speed  58.5 rpm

== Flexion transmission force for the chosen ratio ==
 m=15: output crank/pulley r=36 mm, ratio 1.3: design  26.9 N·m -> 0.75 kN; pad-edge abuse  30.7 N·m -> 0.85 kN
 m=20: output crank/pulley r=35 mm, ratio 1.0: design  35.8 N·m -> 1.02 kN; pad-edge abuse  41.0 N·m -> 1.17 kN

== Walk-only usage (no trot) for the two thin axes ==
 m=15 roll MyActuator X4-36-L f=0.55: walk peak  14.6 N·m -> 0.69; walk RMS  5.0, held  6.5 -> 0.86
 m=15 roll MyActuator X4-36-L f=0.60: walk peak  15.9 N·m -> 0.76; walk RMS  5.4, held  7.1 -> 0.94
 m=20 pitch MyActuator X4-36-L f=0.55: walk peak  18.6 N·m -> 0.89; walk RMS  3.3, held  0.0 -> 0.44
 m=20 pitch MyActuator X4-36-L f=0.60: walk peak  20.3 N·m -> 0.97; walk RMS  3.6, held  0.0 -> 0.48
```
</details>

## Appendix B: `headtail.py` and `massbudget.py` output

```
T15  neck pitch, no spring                        XM540-W270: hold 1.30 N·m (0.42 of rated 3.11), peak 1.48 N·m (0.33 of peak 4.44)
T15  neck pitch, spring cancels 70 % of gravity   XM540-W270: hold 0.39 N·m (0.13 of rated 3.11), peak 0.57 N·m (0.13 of peak 4.44)
T15  neck yaw, 35 deg in 0.6 s                    XM540-W150: hold 0.00 N·m (0.00 of rated 2.14), peak 0.29 N·m (0.10 of peak 3.06)
T20  neck pitch, no spring                        XM540-W270: hold 1.63 N·m (0.52 of rated 3.11), peak 1.86 N·m (0.42 of peak 4.44)
T20  neck pitch, spring cancels 70 % of gravity   XM540-W270: hold 0.49 N·m (0.16 of rated 3.11), peak 0.72 N·m (0.16 of peak 4.44)
T20  neck yaw, 35 deg in 0.6 s                    XM540-W150: hold 0.00 N·m (0.00 of rated 2.14), peak 0.33 N·m (0.11 of peak 3.06)
both tail yaw +/-24 deg at 1 Hz                   XM430-W350: hold 0.00 N·m (0.00 of rated 1.2), peak 0.24 N·m (0.14 of peak 1.72)
T20  tail pitch, gravity + 10 deg at 1 Hz         XM430-W350: hold 0.59 N·m (0.49 of rated 1.2), peak 0.69 N·m (0.40 of peak 1.72)
```

```
T15                      battery x -200 mm: mass before reserve 14.609 kg, CoM x  +23.1 mm, front share  53.6 %
T15                      battery x -250 mm: mass before reserve 14.609 kg, CoM x  +19.6 mm, front share  53.0 %
T15                      battery x -300 mm: mass before reserve 14.609 kg, CoM x  +16.2 mm, front share  52.5 %
   -> battery at -64 mm gives 55.0 % (before reserve ballast)
T20 front X8 / hind X6   battery x -200 mm: mass before reserve 18.837 kg, CoM x  +33.8 mm, front share  55.2 %
T20 front X8 / hind X6   battery x -250 mm: mass before reserve 18.837 kg, CoM x  +31.1 mm, front share  54.8 %
T20 front X8 / hind X6   battery x -300 mm: mass before reserve 18.837 kg, CoM x  +28.4 mm, front share  54.4 %
   -> battery at -224 mm gives 55.0 % (before reserve ballast)
T20 all-X8               battery x -200 mm: mass before reserve 19.857 kg, CoM x  +18.3 mm, front share  52.8 %
T20 all-X8               battery x -250 mm: mass before reserve 19.857 kg, CoM x  +15.8 mm, front share  52.4 %
T20 all-X8               battery x -300 mm: mass before reserve 19.857 kg, CoM x  +13.2 mm, front share  52.0 %
   -> battery at +78 mm gives 55.0 % (before reserve ballast)
```

## Appendix C: the review's independent check (`parts_check/check_sizing.py`)

Written from scratch (law-of-cosines leg, force line through the ankle pin, its own swing law). Usage triplets are peak | heat | speed (≤ 1 passes), worst of walk and trot; the speed rule here is the vendor's 83 rpm cap for the X4 on 12S. The same file also prints the author's voltage-scaled rule for comparison.

```
== Lever arms about the elbow (vertical force through the ankle pin) ==
 ankle x   -75 mm: Fz arm  105.4 mm; Fx arm to ankle pin 204.4 mm (to floor 249.4); knee angle 102.6 deg
 ankle x    +0 mm: Fz arm  150.2 mm; Fx arm to ankle pin 174.2 mm (to floor 219.2); knee angle  99.5 deg
 ankle x   +25 mm: Fz arm  161.5 mm; Fx arm to ankle pin 163.7 mm (to floor 208.7); knee angle  99.9 deg
 ankle x   +75 mm: Fz arm  179.1 mm; Fx arm to ankle pin 144.3 mm (to floor 189.3); knee angle 102.6 deg
 ankle x  +100 mm: Fz arm  185.4 mm; Fx arm to ankle pin 136.1 mm (to floor 181.1); knee angle 105.0 deg

== Swing speeds, joint side (rpm) ==
 walk 0.30 m/s: pitch  58.1, flexion  46.4
 walk 0.40 m/s: pitch  75.4, flexion  54.5
 trot 0.60 m/s: pitch  43.3, flexion  43.3

== NEW: pitch torque to swing the leg (estimate) ==
 X8-150 flexion drive: I ~ 0.058 kg m^2, peak accel   117 rad/s^2 ->  6.8 N m (+ reflected rotor)
 X6-60 flexion drive: I ~ 0.054 kg m^2, peak accel   117 rad/s^2 ->  6.3 N m (+ reflected rotor)

== NEW: stand-up from a crouch, 4 feet down (quasi-static x 1.5) ==
 hip at 300 mm, m 15: front flexion  11.8 N m (joint)
 hip at 300 mm, m 20: front flexion  15.7 N m (joint)
 hip at 250 mm, m 15: front flexion  12.7 N m (joint)
 hip at 250 mm, m 20: front flexion  16.9 N m (joint)
 hip at 200 mm, m 15: front flexion  13.3 N m (joint)
 hip at 200 mm, m 20: front flexion  17.7 N m (joint)

== Requirements, front leg (joint side, N m) ==
 m 15: walk touchdown x1.5 flex  23.5 (Fx at floor  24.0), window  21.6, pitch  13.4, roll  14.6
        trot b.50 0.60 m/s x1.25: Fpk 127.1 N, flex  26.1, pitch  12.7, roll  19.1; RMS flex 10.9, pitch  5.1, roll  7.6
        walk RMS flex  7.8, pitch  2.4, roll  5.0; stand flex  6.1; paw hold flex 12.2, roll  6.5
 m 20: walk touchdown x1.5 flex  31.3 (Fx at floor  32.1), window  28.8, pitch  17.9, roll  19.4
        trot b.50 0.60 m/s x1.25: Fpk 169.5 N, flex  34.9, pitch  16.9, roll  25.4; RMS flex 14.6, pitch  6.8, roll 10.2
        walk RMS flex 10.5, pitch  3.1, roll  6.6; stand flex  8.1; paw hold flex 16.2, roll  8.6

== Chosen builds: usage (<=1 passes) peak | heat | speed; speed rule 'vendor', fastest walk walk 0.30 m/s ==
 T20 f=0.50 front: roll X6-60 0.55|0.77|0.28; pitch X4-36-L 0.77|0.82|0.70; flex X8-150 0.42|0.48|0.48
 T20 f=0.50 rear : roll X6-60 0.55|0.77|0.28; pitch X4-36-L 0.77|0.82|0.70; flex X6-60@1.3 0.64|1.05|0.56!
 T20 f=0.55 front: roll X6-60 0.61|0.85|0.28; pitch X4-36-L 0.85|0.91|0.70; flex X8-150 0.46|0.53|0.48
 T20 f=0.55 rear : roll X6-60 0.50|0.69|0.28; pitch X4-36-L 0.70|0.74|0.70; flex X6-60@1.3 0.58|0.94|0.56
 T20 f=0.60 front: roll X6-60 0.66|0.92|0.28; pitch X4-36-L 0.93|0.99|0.70; flex X8-150 0.50|0.58|0.48
 T20 f=0.60 rear : roll X6-60 0.44|0.62|0.28; pitch X4-36-L 0.62|0.66|0.70; flex X6-60@1.3 0.52|0.84|0.56
 T15 f=0.50 front: roll X4-36-L 0.83|0.92|0.36; pitch X4-36-L 0.58|0.62|0.70; flex X6-60@1.3 0.48|0.79|0.56
 T15 f=0.50 rear : roll X4-36-L 0.83|0.92|0.36; pitch X4-36-L 0.58|0.62|0.70; flex X6-60@1.3 0.48|0.79|0.56
 T15 f=0.55 front: roll X4-36-L 0.91|1.02|0.36!; pitch X4-36-L 0.64|0.68|0.70; flex X6-60@1.3 0.53|0.87|0.56
 T15 f=0.55 rear : roll X4-36-L 0.74|0.83|0.36; pitch X4-36-L 0.52|0.56|0.70; flex X6-60@1.3 0.44|0.71|0.56
 T15 f=0.60 front: roll X4-36-L 0.99|1.11|0.36!; pitch X4-36-L 0.70|0.74|0.70; flex X6-60@1.3 0.58|0.94|0.56
 T15 f=0.60 rear : roll X4-36-L 0.66|0.74|0.36; pitch X4-36-L 0.46|0.49|0.70; flex X6-60@1.3 0.39|0.63|0.56
 (fastest walk 0.40 m/s: only the speed column changes - X4 pitch 0.91, X8 front flexion 0.56, X6 hind flexion 0.66, X4 roll 0.36)

== Flexion pushrod force including transmission angle (NEW) ==
 T15: r_out 36 mm, transmission angle 90 deg: design 0.73 kN, pad-edge abuse 0.84 kN
 T15: r_out 36 mm, transmission angle 60 deg: design 0.84 kN, pad-edge abuse 0.97 kN
 T15: r_out 36 mm, transmission angle 40 deg: design 1.13 kN, pad-edge abuse 1.30 kN
 T20: r_out 35 mm, transmission angle 90 deg: design 1.00 kN, pad-edge abuse 1.15 kN
 T20: r_out 35 mm, transmission angle 60 deg: design 1.15 kN, pad-edge abuse 1.33 kN
 T20: r_out 35 mm, transmission angle 40 deg: design 1.55 kN, pad-edge abuse 1.79 kN

== X4 on 12S: speed against the vendor's 'rated speed above rated voltage' cap (83 rpm) ==
 walk 0.30 m/s: pitch  58.1 rpm = 0.70 of 83 rpm; author's rule would give 0.43
 walk 0.40 m/s: pitch  75.4 rpm = 0.91 of 83 rpm; author's rule would give 0.55
 trot 0.60 m/s: pitch  43.3 rpm = 0.52 of 83 rpm; author's rule would give 0.32
 PR #669 swing profile, 0.30 m/s: pitch 69.0 rpm = 0.83 of 83 rpm
```

Review files, in the working session's scratchpad (`trike/parts_check/`; not committed): `check_sizing.py` and `check_sizing_out.txt`, `check_bom.py` and `check_bom_out.txt` (the arithmetic as written), `fix_bom.py` with `fix_bom_tables.md` and `fix_bom_totals.md` (the corrected tables), `apply_fixes.py` (builds this file).

The author's files, in the same scratchpad (`trike/parts/`; not committed):
- `sizing.py` and `sizing_out.txt`
- `bom.py` and `bom_out.md`
- `massbudget.py`
- `headtail.py`
- `PARTS_PREMIUM.md` (the unreviewed version of this file)
