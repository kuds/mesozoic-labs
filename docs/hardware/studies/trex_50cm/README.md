# 50 cm T-Rex: feasibility and first-prototype budget

**Decision: proceed to detailed layout and a loaded actuator bench test.** A
50 cm expressive biped is credible, but the present miniature cannot simply be
enlarged. The leading costed architecture is approximately **5.9 kg including
500 g reserve**, with **18 actuators** and a **USD 5,600–8,000 first-project
allowance**. This includes a 25% contingency; tax, shipping, import charges and
labor are additional. Prices were checked on **2026-10-24**. These are sizing
estimates, not a fabrication release, supplier quotation or demonstrated walker.

The comparison with the mini and 80 cm concepts is in
[the size decision study](../../TREX_SIZE_TRADE_STUDY.md).

## What supports feasibility

ROBOTIS documents its OP3 at about 51 cm and 3.5 kg without cosmetic skin, with
20 XM430-based axes [1]. This establishes that the size is practical for a
research biped, but its mass distribution and leg geometry differ from a T-Rex.
Our forward head, tail, short legs and crouch need their own load analysis.

The miniature's 20-second stepping diagnostic supplies a useful warning:
loaded leg RMS torque is about 0.13–0.14 Nm and peak knee speed reaches
8.15 rad/s. The longer trial falls, and the motors are ideal torque-limited
actuators. It is neither a successful hardware gait nor a controller to copy.
See [the original evidence](../../../../examples/mini_tyrannosaurus_rex/WALKING_AND_TAIL.md).

Uniform scaling from 25.8 to 50 cm multiplies lengths by 1.938. Constant-density
scaling of the 825 g tail concept would produce approximately 6.0 kg, but actual
motor cases and electronics do not scale that way. The independent component
budget below happens to land close to that number.

## Architecture to investigate

| Assembly | Preliminary choice |
| --- | --- |
| Legs | Six axes each: hip yaw/roll/pitch, knee pitch, ankle pitch/roll |
| Leg actuators | Twelve CubeMars AK45-10 **V3.0** KV75, subject to bench qualification |
| Head | Two XM430-W210-T for yaw and nod; lightweight shell, bearings carry external loads |
| Character | Four XL330-M288-T: independent left/right shoulder pitch, jaw, tail yaw |
| Feet | Initial CAD target about 130–150 × 70–85 mm; replaceable flat compliant soles |
| Proportions | 500 mm tall, approximately 300–330 mm wide and 550–650 mm nose to tail |
| Legs/layout | Start with roughly 100–115 mm thigh/shin links; revise around real motor envelopes |
| Electronics | SBC for policy/perception, separate deterministic motor/IMU bridge and watchdog |
| Sensors | Rigid pelvis IMU, joint feedback, foot contact sensing; one snout camera initially |

All dimensions in this table are design targets. A complete motor/link/cable
packing solution has **not** been established. The 53 mm diameter motor bodies,
hip stack, ankle pair and connector exits must be placed in CAD together.

Keep the T-Rex look through the head silhouette, tiny arms and tail, while
reducing sustained knee crouch and forward mass offset. Put the battery near
the pelvis. Ankle roll is useful for moving the center of mass over the stance
foot while maintaining contact; two extra axes alone do not establish balance.
Do not preserve the mini's narrow roll limits by default, or widen them without
checking collisions. Use removable belly/back panels and partial leg guards.

## Actuator decision: torque, speed and mass together

| Candidate | Mass | Torque basis | Speed basis | Listed USD each |
| --- | ---: | --- | --- | ---: |
| AK45-10 V3.0 | 262 g | 2.5 Nm rated; 7 Nm peak | 120 rpm rated, 24 V | 155.90 |
| XM540-W150-T | 165 g | 1.46 Nm estimated continuous; 7.3 Nm stall | 53 rpm **unloaded**, 12 V | 482.89 |
| XM540-W270-T | 165 g | 2.12 Nm estimated continuous; 10.6 Nm stall | 30 rpm **unloaded**, 12 V | 482.89 |
| XM430-W210-T | 82 g | 0.60 Nm estimated continuous; 3 Nm stall | 77 rpm **unloaded**, 12 V | 310.39 |

Sources [2–5]. ROBOTIS's continuous figures are explicitly estimates of 20% of
stall, not tested thermal guarantees. CubeMars's rated point also needs checking
at low-speed stance loads inside our brackets. Neither peak nor stall is a
continuous working allowance. No-load and rated speeds are different quantities.

The AK45's cost and speed make it the first bench candidate. Twelve weigh
3.144 kg, so it is not an especially lightweight solution. Its current V3
drawing specifies a diameter of 53 mm and axial size of 45.2 mm [7].

For a deliberately limited similarity calculation, slow the mini's cadence by
sqrt(1.938), from 0.361 to 0.503 seconds, and multiply torque by mass ratio times
length ratio. At 5.91 kg this predicts about **2.02 Nm knee RMS**, **1.95 Nm ankle
pitch RMS**, and **56 rpm peak knee speed**. The real replacement masses are
distributed differently, there are new ankle joints, and the old gait is not
robust. These numbers are screening estimates, not a new simulation result.
The reference applied peaks were clipped; scaling them does not establish the
required peak torque.

This screen leaves only about 24% rated-to-RMS ratio margin at the knee. Require
posture/gait improvement toward **at most 1.75 Nm RMS on the most loaded AK45
joints** as an initial design objective, followed by measured thermal duty.
That objective is 70% of the catalog rating, not a manufacturer guarantee.

A useful independent check is gravity moment, mass × 9.81 × horizontal offset.
At 5.91 kg, one loaded support axis would see about **1.74 Nm at 30 mm** but
**2.90 Nm at 50 mm**. Keep load paths near the support axes and avoid a deep
permanent squat. Whole-body dynamics and segment weights require inverse
dynamics after CAD; this simplified moment is not every joint's actual load.

The lighter Dynamixel comparison (ten XM540-W150 leg motors, two XM430 hip-yaw
motors, same head/character drives) budgets 4.58 kg. Its corresponding knee RMS
screen is 1.56 Nm and speed is still 56 rpm: both challenge the W150 at 12 V.
The slower W270 does not solve that speed requirement. A redesigned slower gait
may work, but this route is not presently better demonstrated and costs more.

A stronger brushless fallback is twelve **MYACTUATOR RMD-X4-P12.5-10** units:
the US manufacturer listing gives 4 Nm rated, 238 rpm rated, 330 g and USD 399
each [9]. Keeping the same other allowances gives **6.73 kg** and approximately
**USD 9,223–11,648** including contingency. Its similarity knee RMS is 2.30 Nm,
leaving substantially more catalog torque margin than the AK45 option. This is
a different mass/design budget, not permission to exceed the 6.5 kg stopping
rule on the cheaper architecture. Verify exact revision, low-speed heating and
output bearings before choosing either. If the AK45 bench fails, this is a
more useful fallback to investigate than buying slow high-ratio Dynamixels.

## Reconciled mass and cost

| Mass item | g |
| --- | ---: |
| Twelve leg actuators | 3,144 |
| Two head and four character actuators | 236 |
| Load-bearing structure | 550 |
| Head/body shells and guards | 250 |
| Moving tail / two arms | 80 / 50 |
| Battery and restraint | 450 |
| Computer, camera, sensors, bridge, audio | 180 |
| Power distribution and wiring | 220 |
| Bearings, fasteners and foot pads | 250 |
| Unallocated integration reserve | 500 |
| **Total** | **5,910** |

Motor masses are manufacturer values. Everything else is an unweighed
allowance. Installed weight without using the reserve is 5.41 kg. The budget
includes camera and moving appendages, unlike the miniature's original ledger.

| First-project cost item | USD |
| --- | ---: |
| Twelve leg actuators | 1,870.80 |
| Two head and four character actuators | 730.74 |
| Structure, bearings, hardware, feet | 450–900 |
| Cosmetic parts and reprints | 150–300 |
| Compute, camera, storage, cooling, audio | 200–400 |
| Interfaces, MCU, IMU, contact sensing | 120–250 |
| Power conversion, distribution, wiring, cutoff | 200–400 |
| Battery and charger | 140–250 |
| Spare parts and rework | 250–500 |
| Bench supply, fixture and basic tools | 250–500 |
| Cloud experiments | 100–300 |
| **Subtotal** | **4,462–6,402** |
| **Including 25% contingency** | **5,577–8,002** |

Only the actuator rows are current catalog prices; other rows are planning
allowances. No volume discount or existing owned equipment is assumed. The
equivalent Dynamixel comparison is approximately **10,051–12,476 USD** with the
same contingency and exclusions. Lower purchase price does not eliminate the
brushless route's additional integration work. Supplier lead times and landed
cost must be rechecked when selecting exact revisions.

## Arms, tail, head and power

Independent up/down arms and gentle tail yaw are credible at this scale if kept
light. With a 25 g arm's center of mass 60 mm from its pivot, horizontal static
load is about 0.015 Nm. For an 80 g, 240 mm uniform-rod tail wagging ±15 degrees
at 0.5 Hz, inertia needs about 0.004 Nm. Adding a conservative gravity bound at
20 degrees body lean plus 0.01 Nm friction gives about **0.046 Nm**. An XL330
can be investigated for these light gestures; its estimated continuous figure
is 0.10 Nm [6]. The tail root must separately carry about 0.094 Nm bending.
Use a bearing-supported pivot and a soft/removable tail. This is not a balance
tail, and changing its mass or rhythm requires retesting locomotion.

Budget the moving head near 300 g and its center of mass within 40 mm of the
nod axis: static torque is then about 0.118 Nm. Head motor placement and
transmissions must be included in that moving mass. The same visible head can
be made hollow and lightweight rather than used as a structural enclosure.

Use separate motor and logic power branches. The candidate architecture needs
a roughly 24 V leg bus plus regulated 12 V head and 5 V character/logic supplies;
do not connect the miniature servos to the leg bus. A 6S 3 Ah battery is an
**energy scenario**, not a selected/approved pack: 66.6 Wh nominal and 53.3 Wh
usable under an 80% assumption gives 40/27/20 minutes at 80/120/160 W average
battery draw. Consumption has not been measured. A provisional 20–30 minute
walking-session goal is sensible, not promised runtime. Confirm full-charge
25.2 V, depleted voltage, regenerative transients and the exact driver's allowed
input range before choosing direct battery wiring. The linked AK48-2405-2D-A2
driver manual specifies **15–28 V**, encompassing a charged 6S pack [10]; verify
that the purchased motor contains that driver revision and constrain regenerative
overvoltage. Product nominal voltage alone is insufficient. A bench supply must
also tolerate or clamp regeneration.

Use at least two separately budgeted CAN branches as the starting architecture.
At six motors per bus, 250 Hz, two 160-bit frame allowances per update and
1 Mbit/s, nominal occupancy is 48%; at 500 Hz it is 96%. Actual feedback may
require more packets, so measure bus load rather than promising 500 Hz. The V3
manual has revision-specific torque conversion and feedback formats [8].
Protocol command range is not permissible motor torque. Keep the watchdog and
joint limits below the high-level policy; camera processing must not block
motor updates. A small SBC is a plausible inference host, pending benchmarking;
onboard GPU training is not a requirement for this prototype.

## Practical sequence and stopping rules

1. **Layout and identification:** place real motors, bearings, wires, power
   modules and battery in a 50 cm CAD envelope. Weigh representative printed
   pieces. Reject mass creep past about 6.5 kg unless the load budget is redone.
2. **One/two-motor bench, USD 700–1,200 provisional:** characterize static and
   oscillating loads around 1.5–2.0 Nm, relevant speed, friction, backlash,
   temperature, current, voltage and command delay. Keep duty within the exact
   manufacturer's limits. Validate shutdown behavior with the load supported.
3. **One complete six-axis leg, USD 1,500–2,500 cumulative provisional:** check
   packing, sole contact, cable travel and structural stiffness under a
   representative supported body load. This hardware can feed the full build.
4. **Two legs and weighted pelvis:** validate standing, stepping, turns and
   recovery under a slack fall-arrest fixture. Then add head, shell and gestures
   in measured increments, updating inertias and gait tests each time.

A planning allowance is **300–600 focused hours**, with substantial uncertainty
and parts lead times. That is roughly 4–9 months at 15–20 hours/week, not a
delivery commitment. CAD, embedded actuation and measured sim-to-real work are
the main tasks beyond the existing RL infrastructure.

Add future 50 cm CI gates only after a real model exists: mass/COM/inertia
ledger, motor/cable collision checks, stance and perturbations, torque-speed
and thermal duty, and seeded 60-second walking/turning trials with appendages
in different poses. Keep physical endurance and fault tests separate. A passing
miniature balance test cannot qualify this larger robot.

## Evidence and reproduction

`python3 docs/hardware/studies/trex_50cm/calculate.py` regenerates
[results.json](results.json) from [inputs.json](inputs.json) and the checked-in
miniature report. Input/report/script hashes are recorded. The arithmetic was
executed and checked; no 50 cm CAD, gait simulation or physical test is claimed.

1. [ROBOTIS OP3 specifications](https://docs.robotis.com/docs/systems/op3/introduction/).
2. [CubeMars AK45-10 V3.0](https://www.cubemars.com/product/ak45-10-v3-0-kv75-robotic-actuator.html).
3. [ROBOTIS XM540-W150-T](https://robotis.us/products/dynamixel-xm540-w150-t).
4. [ROBOTIS XM540-W270-T](https://robotis.us/products/dynamixel-xm540-w270-t).
5. [ROBOTIS XM430-W210-T](https://robotis.us/products/dynamixel-xm430-w210-t).
6. [ROBOTIS XL330-M288-T](https://robotis.us/products/dynamixel-xl330-m288-t).
7. [AK45 V3 mechanical drawing](https://www.cubemars.com/data/cms/202607/ak45-10-v3-0-kv75-2d-drawing.pdf).
8. [AK V3 driver software manual](https://www.cubemars.com/data/cms/202610/ak-series-motor-module-v3-driver-software-user-manual-en.pdf).
9. [MYACTUATOR RMD-X4-P12.5-10](https://myactuator.us/products/rmd-x-series/rmd-x4-p12-5-10).
10. [AK48 driver installation manual](https://www.cubemars.com/data/cms/202610/ak48-2405-2d-a2-drive-installation-instructions.pdf).
