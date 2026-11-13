# 80 cm T-Rex: feasibility, architecture and practical costs

**Decision: credible as a later flagship, but not a scaled-up miniature.** Two
catalog-actuator approaches budget approximately **9.8–11.4 kg**, including a
1 kg integration reserve, and **USD 13,500–20,600 for the first development
project**. Use **USD 14,000–21,000** as a rounded planning envelope, before tax,
shipping, import charges and labor. Both remain unbuilt concepts requiring
CAD, measured actuator characterization and fresh locomotion development.

This is an **80 cm T-Rex**, not an Olaf replica. Disney reports Olaf at 88.7 cm
without hair and 14.9 kg, with a different silhouette and mechanism [1]. Its
demonstrated performance does not validate our design or our cost estimates.

## Two approaches worth a bench comparison

| Item | Higher gearing, compact package | Lower gearing, larger joints |
| --- | --- | --- |
| Leg candidate | MYACTUATOR RMD-X4-P36-36-E | CubeMars AK80-9 V3.0 KV100 |
| Quantity | 12 | 12 |
| Published rated / peak torque | 10.5 / 34 Nm | 9 / 22 Nm |
| Published rated speed | 83 rpm at 24 V | 390 rpm at 48 V |
| Gear reduction | 36:1 | 9:1 |
| Manufacturer mass each | 360 g | 490 g |
| Budget each | USD 449 | USD 579.90 |
| Whole-robot budget including reserve | 9.85 kg | 11.41 kg |
| First-project budget incl. 30% contingency | USD 13,522–18,592 | USD 15,564–20,634 |

Sources [2–4]. These ratings are not permission to hold rated torque indefinitely
at zero speed inside a shell. Characterize low-speed duty and cooling in the
actual mounting arrangement. Use the exact **-E**, not the cheaper **-E-L**,
for the MYACTUATOR comparison; their peak ratings and other details differ.
The CubeMars product site lists USD 479.90 beside selectable driver options,
while its official store lists USD 579.90. The budget deliberately uses the
higher figure pending a complete driver-equipped package quote.

The compact 24 V option is attractive for weight, packaging and cost. Its
higher gearing needs closer examination of friction, reflected inertia,
backlash and foot-impact behavior. More peak torque alone does not make it the
better walking joint. The 9:1 option is a useful comparison for controlled
compliance, but has larger 98 mm diameter cases and a higher-voltage supply.
Neither motor is selected for purchase by this study.

Unitree GO-M8010-6 is also relevant to this scale: the manufacturer's current
page lists about 530 g, 23.7 Nm maximum torque and a 12–30 V input range [5].
It does not supply a directly comparable continuous-duty rating there. Do not
use that maximum as a working torque or infer our design is qualified because
Disney used Unitree motors. A separate thermal test and exact supplier quote
would be needed before replacing the costed choices above.

## Mechanical requirements

Use six axes per leg, plus head yaw/nod, jaw, two independent shoulder pitches
and tail yaw: **18 actuators**. Start layout around 800 mm height, approximately
450–520 mm width and **850–1,050 mm nose-to-tail length**. Uniform scaling of
the mini's tail envelope gives about 800 × 498 × 922 mm; actual proportions must
change to fit the hardware and improve load paths. These are layout envelopes,
not finished CAD dimensions. The nearly one-metre length matters for transport
as much as height. Make the tail and selected panels removable.

Use roughly 160–190 mm thigh/shin link candidates and 190–230 × 100–130 mm feet
as starting CAD ranges. Keep knee flexion modest in the neutral stance; retain
the dinosaur silhouette with cosmetic contours rather than a heavy, deep squat.
Keep batteries and computers in the pelvis, avoid ballast in the tail, and use
foam or thin removable cosmetic shells. The chassis, output bearings and feet
must carry impacts independently of decorative parts.

The similarity screen reuses the miniature's recorded 20-second diagnostic and
retimes its period to about 0.636 seconds. It produces **5.38 Nm knee RMS** for
the 9.85 kg geared option and **6.23 Nm** for the 11.41 kg lower-ratio option;
peak knee speed is about **44 rpm**. Actual motor placement, limb inertia and
new ankle-roll axes invalidate literal transfer of these numbers. The reference
gait falls in its longer trial. This is an analytical load screen, not an
80 cm walking demonstration or a certified torque envelope.

At 11.41 kg, a one-leg gravity moment is about **5.60 Nm at 50 mm horizontal
offset** and **8.95 Nm at 80 mm**. A forward-heavy head can consume the working
margin even with these larger motors. The center of mass must shift over the
stance foot; a proportional enlargement does not improve the normalized
support geometry. Investigate initial loaded-joint RMS objectives below 70% of
catalog rated torque, then replace them with measured thermal-duty limits.

## Mass and budget reconciliation

| Mass allocation | Compact 24 V | Larger 48 V |
| --- | ---: | ---: |
| Twelve leg actuators | 4,320 g | 5,880 g |
| Six head/character actuators | 398 g | 398 g |
| Structure | 1,000 g | 1,000 g |
| Head/body shells and guards | 600 g | 600 g |
| Tail / two arms | 180 / 100 g | 180 / 100 g |
| Battery and restraint | 1,000 g | 1,000 g |
| Compute, camera, sensors, bridge, audio | 300 g | 300 g |
| Power distribution and wiring | 450 g | 450 g |
| Bearings, fasteners, feet | 500 g | 500 g |
| Integration reserve | 1,000 g | 1,000 g |
| **Total** | **9,848 g** | **11,408 g** |

Only motor masses are catalog values. Other entries are unweighed engineering
allowances. The common battery allowance compares equal energy, not an identical
pack or a proven installation. If CAD moves either design above roughly 13 kg,
reopen its geometry, duty cycle, actuator and cost budgets.

| First-project cost | Compact 24 V | Larger 48 V |
| --- | ---: | ---: |
| Twelve leg motors | 5,388 | 6,958.80 |
| Head/character motors | 1,413.84 | 1,413.84 |
| Structure, bearings, hardware, feet | 900–1,800 | 900–1,800 |
| Cosmetic shells and reprints | 300–650 | 300–650 |
| Compute, camera, cooling, storage, audio | 300–650 | 300–650 |
| Interfaces, MCU, IMU, contact sensors | 200–450 | 200–450 |
| Power conversion, distribution, wiring, cutoff | 350–700 | 350–700 |
| Battery and charger | 350–650 | 350–650 |
| Spares and rework | 600–1,200 | 600–1,200 |
| Bench supply, fall fixture, basic tools | 400–900 | 400–900 |
| Cloud experiments | 200–500 | 200–500 |
| **Subtotal** | **10,402–14,302** | **11,973–15,873** |
| **Including 30% contingency** | **13,522–18,592** | **15,564–20,634** |

Amounts are USD, checked 2026-10-24. Non-actuator lines are planning allowances,
not quotes. New workshop machines, paid engineering and event staffing are
excluded. Common cost ranges allow different implementation choices but do not
promise the 24 V and 48 V distribution hardware will cost exactly the same.
Do not count a regulator's current number or a motor's phase-current rating as
the measured battery current of the complete robot.

## Character movement and energy

The costed six character axes use one XM540-W150 for head nod, two XM430-W210
for head yaw and tail yaw, and three XC330-M288 for jaw and arms. A 600 g moving
head centered within 60 mm of its nod pivot needs about 0.35 Nm statically;
dynamic load and costume friction still need testing. A 50 g arm with a 100 mm
center-of-mass radius needs about 0.049 Nm horizontally. These small arms can
provide character without grasping or load-carrying ambitions.

A 180 g, 380 mm uniform-rod tail at ±15 degrees and 0.5 Hz needs about 0.022 Nm
inertial torque. Adding a 20-degree body-lean gravity bound and 0.03 Nm friction
gives about **0.167 Nm**; root bending from weight is approximately **0.336 Nm**.
The larger tail drive is intentional: the miniature XL330 has too little
continuous margin for that scenario. Use bearings at the root and a lightweight,
compliant or removable tail. It is an expressive appendage; no balance benefit
is assumed.

An energy-equivalent 12S 3 Ah or 6S 6 Ah scenario contains 133.2 Wh nominal.
Assuming 80% usable energy, 180/240/320 W average battery draw gives
36/27/20 minutes. These are sensitivity cases, not predicted consumption.
Budget swappable batteries and long standing/gesture thermal tests. Full-charge
voltage, low-voltage cutoff, pack discharge, connectors, regeneration and the
exact driver voltage range must be reconciled before buying packs. The 12S
scenario reaches 50.4 V when full; a nominal 48 V product description is not
enough to approve it. Lower voltage can increase bus-current and cable burdens.

The head camera remains in a dark snout insert by default; visible eyes remain
character features. A larger head gives room for optional depth sensing, but
sensor field of view, downward near-field coverage and tracking reliability
still need deliberate placement. Do not assume eye-mounted cameras or face
recognition are necessary for an expressive remote-controlled demonstration.

## Practical development and use

Plan for a braced test fixture, supported load tests, replaceable foot pads,
protected wiring, accessible motor-power cutoff and a known loss-of-command
response. A sudden torque cut can cause a biped to fall, so test that behavior
with support. Tethered laboratory success is distinct from unattended operation
around people. No public-interaction readiness is established by this study.

For rough comparison only, potential energy associated with dropping the
center of mass through 0.40 m is about 45 J at 11.4 kg. A 5.9 kg robot through
0.25 m is about 14.5 J. Actual impact depends on fall geometry and compliance;
these are not injury thresholds. They explain why the larger platform needs a
more substantial development fixture and more careful demonstration setup.

An initial actuator/load bench is approximately **USD 1,500–3,000** depending on
which two candidates and supplies are tested. A complete one-leg fixture is
approximately **USD 4,000–6,000 cumulative**, before a full robot purchase.
These are staged portions of development spending, not additional fixed fees
on top of the project budget. Equipment may be reused; changing actuator family
can strand prototype parts.

Allow approximately **600–1,200 focused hours** as an uncertain solo planning
range: about 7–19 months at 15–20 hours/week, before extra delays. A proven
50 cm control/deployment workflow can reduce some software effort, but neither
the mechanics nor learned weights transfer unchanged. Sustained walking,
turning, low-battery behavior, thermal endurance and repeated demonstration
resets matter more than adding more facial axes early.

## Evidence and reproduction

From the repository root:

```sh
python3 docs/hardware/studies/trex_50cm/calculate.py \
  --input docs/hardware/studies/trex_80cm/inputs.json \
  --output docs/hardware/studies/trex_80cm/results.json
```

[Inputs](inputs.json) and [computed results](results.json) preserve sources,
assumptions and hashes. The arithmetic was run; CAD fit, dynamic simulation and
hardware tests for this scale have not been performed.

1. [Disney Olaf research](https://arxiv.org/html/2512.16705v2).
2. [MYACTUATOR RMD-X4-P36-36-E](https://myactuator.us/products/rmd-x-series/rmd-x4-p36-36).
3. [CubeMars AK80-9 V3.0 specifications](https://www.cubemars.com/product/ak80-9-v3-0-robotic-actuator.html).
4. [CubeMars official store price](https://store.cubemars.com/products/ak80-9-v3-0-kv100).
5. [Unitree GO motor specifications](https://www.unitree.com/mobile/go1/motor/).
6. [Head and small-servo sources](../trex_50cm/inputs.json), plus
   [XC330 manufacturer listing](https://robotis.us/products/dynamixel-xc330-m288-t).
