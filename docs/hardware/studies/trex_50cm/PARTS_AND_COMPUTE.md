# 50 cm T-Rex: compute, parts and character design

An initial [MuJoCo implementation](../../../../examples/tyrannosaurus_rex_50cm/README.md)
now represents the visual concept, 18-axis architecture and complete assigned
mass ledger. Its short standing/gesture checks do not establish physical part
fit, electrical compatibility, thermal duty or walking.

**Latest character direction, 2026-10-24:** the
[head-focused study](../../../../examples/tyrannosaurus_rex_50cm/head_study/README.md)
supersedes the fuller-torso preference. It explores a large skull and jaw over
a compact equipment body, now comparing rounded-box, long-oval and soft-wedge
skulls at roughly 58–59 cm. **Silhouette C is selected** and is now the main
MuJoCo model, at 583.6 mm and 7.957 kg. Compared with this baseline BOM it adds
120 g structure, 10 g bearings/fasteners and 210 g shells/guards while retaining
the 600 g reserve. Earlier alternatives remain available. No motor or computer
change is selected; its unweighed head/structure allocations and
short simulation trials do not yet constitute an updated fabrication quote.

**Earlier proportion follow-up, 2026-10-24:** the
[body-proportion comparison](../../../../examples/tyrannosaurus_rex_50cm/proportion_study/README.md)
recommends developing a fuller **approximately 55 cm** body with unchanged leg
lengths and low-mounted compute/battery. Its assigned mass is 8.067 kg and its
first-build planning range is **$14,554–15,167**, before any motor-ratio change.
The 51.5 and 60 cm alternatives, additive costs and standing-only evidence are
recorded there. Those costs apply to the torso-size scenarios; the line-item
BOM below remains their baseline, not a finalized head-focused build budget.

**Baseline parts configuration, 2026-10-24:** Jetson Orin Nano Super 8GB, a separate
Teensy 4.1 motor bridge, twelve 4 Nm-rated RMD leg actuators and six smaller
head/character motors. Target **48–52 cm standing height**, **7.0 kg allocated
mass plus 0.6 kg reserve**, and approximately **USD 14,000 for the first
development build**. Start with loaded actuator and electronics benches before
ordering the complete motor set.

This detailed plan **supersedes the earlier 50 cm recommendation and its shared
electronics/mass allowances**. The AK45 and Dynamixel studies remain historical
comparisons. The fuller integration ledger makes the cheap AK45 route less
comfortable: stronger legs are now the default to qualify, rather than a fallback.
This is a candidate BOM and visual design brief, **not fabrication CAD, a finished
electrical schematic or a purchase release**. Neither option has demonstrated walking.

## Computer selection

| Option | Cost basis | Recommendation |
| --- | --- | --- |
| **Jetson Orin Nano Super 8GB** | **$399** developer kit, current SparkFun listing; backordered [1] | Preferred for the intended progression to vision, following and navigation |
| Raspberry Pi 5, 4GB | About **$110 bare**, derived from official successive price changes [3]; roughly $180–230 with storage, cooling, power and mounting, before camera/MCU | Credible for teleoperation, small policies and camera-assisted perception; benchmark the workload |
| Orin NX / AGX | Not included | Reconsider only when measured latency, memory or model requirements exceed Nano capacity |

NVIDIA's launch announcement advertised $249 [2]. That historical price is not
the procurement basis here. A supported kit at $249 would reduce installed
parts cost by $150. Reconfirm prices and stock before purchase.

**Walking alone does not inherently require a GPU.** A compact locomotion
policy may run adequately on a Pi. The OAK camera already computes stereo depth
and can run some neural inference, making Pi plus OAK a legitimate alternative.
Jetson provides flexibility for additional CUDA/TensorRT perception and mapping.
Buying it does not implement autonomy. Training stays on a workstation/cloud;
the robot runs inference. The compute choice is a relatively small portion of
this project's cost, so it is reasonable to provide that headroom now.

Keep gait updates independent of blocking camera/network callbacks. Initial
benchmark targets are a **50–100 Hz policy** and **250 Hz motor bridge**; measure
tail latency and missed deadlines with camera, logging and network traffic
active. TOPS is not a control-loop benchmark. Pin a tested JetPack, DepthAI and
policy-runtime image; the repository's development Python environment is not
automatically a deployable Jetson image. No deployment stack is implemented by
this documentation change.

The Teensy timestamps pelvis IMU/foot contacts, schedules two CAN buses and
enforces command freshness/limits. Six leg motors per 1 Mbit/s bus, two frames
per update and a 160-bit frame allowance imply **48% occupancy at 250 Hz**;
500 Hz would consume 96% before retries or extra telemetry. Confirm the exact
protocol. The MCU's CAN controllers still need transceivers, termination and
protection [4]. Motor current loops remain local to the drives.

## Motors and expressive movement

| Assembly | Candidate | Qty | USD each | Subtotal USD | Published basis |
| --- | --- | ---: | ---: | ---: | --- |
| Legs | MYACTUATOR RMD-X4-P12.5-10 | 12 | 399.00 | 4,788.00 | 4 Nm rated, 10 Nm peak; 330 g; 238 rpm rated at nominal 24 V [5] |
| Head yaw/nod | XM430-W210-T | 2 | 310.39 | 620.78 | 82 g; 0.6 Nm estimated continuous, 3 Nm stall at 12 V [7] |
| Arms, jaw, tail | XL330-M288-T | 4 | 27.49 | 109.96 | 18 g; 0.1 Nm estimated continuous, 0.52 Nm stall at 5 V [8] |
| **Total** | **18 axes** | | | **5,518.74** | **4.196 kg** |

Each leg has hip yaw/roll/pitch, knee pitch and ankle pitch/roll. These permit
lateral weight transfer and foot orientation; they do not establish a gait.
Use moderate knee bend and broad flat soles initially. Qualify slow supported
steps before dynamic movement. No MicroDuck-like walking quality is established.

The US RMD designation differs from the global manufacturer's newer
**RMD-X4-P12-10-E** listing [5,6]. Before purchase, obtain matching revision CAD,
firmware/protocol, torque-current conversion, supply limits and output bearing
loads. Do not combine similarly named products' specifications. Published
current/torque fields do not suffice to infer battery draw or calibrated command
torque without confirmation. Low-speed heating must be measured in real mounts.

The **AK45-10 V3.0** comparison is $155.90, 262 g and 2.5 Nm rated [9]. Twelve
save $2,917.20 and 816 g. With the fuller nonmotor ledger, however, that robot
budgets **6.80 kg** and the old diagnostic gait's similarity screen gives about
**2.32 Nm knee RMS**, close to its rating. The stronger proposal budgets **7.62 kg**
and screens near **2.60 Nm**, against 4 Nm rated. This favors RMD qualification.

This is **not inverse dynamics for the new robot**: the miniature's reference
gait eventually falls, real replacement masses are distributed differently,
new ankle-roll loads are missing, and clipped reference peaks do not establish
required peak torque. Initially aim below 70% of catalog rated torque on loaded
joints, then qualify actual thermal duty. ROBOTIS continuous values above are
manufacturer estimates, not a sustained-duty guarantee in this assembly.

The character motors are plausible with strict mass limits:

- **Arms:** independent up/down shoulder motion; approximately 25 g each and
  60 mm center-of-mass offset produce about 0.015 Nm static gravity load.
- **Tail:** hollow/foam shell, 80 g and roughly 240 mm long; an external bearing
  carries bending, and the motor supplies yaw. The earlier ±15°, 0.5 Hz scenario
  estimates about 0.046 Nm including assumed lean/friction. Start with slow
  standing gestures and reduce wagging during walking. This is expression,
  not a validated active balancing tail.
- **Head:** 350 g moving-assembly target and center of mass within 40 mm of
  the nod axis give about 0.137 Nm static moment. That includes the 61 g camera,
  82 g nod motor, 18 g jaw motor and about 189 g shell/brackets. It is a subset
  of the ledger, not additional mass. Use bearings and balanced placement.

## Sensors, packaging and character

The **OAK-D Lite fixed-focus camera belongs in the snout**, not the cosmetic
eyes. It provides stereo depth and RGB and lists at $269, approximately
91 × 28 × 17.5 mm and 61 g [10]. Use three unobstructed optical openings. Check
field of view, glare and lower-field occlusion; calibrate head joints, timestamps
and camera extrinsics. A moving head camera cannot substitute for foot sensing.

Mount the **BNO085 rigidly on the pelvis**, away from motor wiring, with raw
gyro/accelerometer access. Do not depend on magnetic heading near the motors;
measure filter/fusion delay and vibration response [11]. Budget four contact/FSR
elements per foot, mechanically captured in the soles. Treat them as contact
and load-trend sensors until calibrated, not precision force/CoP instruments.

| Item | Initial layout reservation or target |
| --- | --- |
| Overall | 480–520 mm high; about 300–330 mm wide and 550–650 mm nose to tail |
| Feet | About 130–150 × 70–85 mm; check support margin and swing clearance |
| Computer tray | About 120 × 110 × 65 mm plus cable/airflow clearance; verify against actual kit and mounts |
| Battery | Candidate 138 × 42 × 43 mm, 506 g; restrained removable cradle near pelvis [12] |
| Motors | Import exact revision CAD; check hip stack, paired ankles, connectors and bearings throughout travel |
| Access | Rear computer tray, belly battery panel, visible hip service covers; guarded wiring/pinch areas |

These are CAD reservations, not proven clearances. Avoid quietly increasing
height to solve packing problems while retaining the same load claims.

For the initial cartoon concept, use a **sage-green hatchling T-Rex**, cream
belly, large amber cosmetic eyes, rounded brow, short snout, small teeth, tiny
two-finger arms and tapered tail. Rounded three-toe covers sit over practical
wide soles. Keep the head hollow and the knee bend moderate. A dark snout strip
contains camera apertures; dark hip housings, short exposed links and recessed
fasteners make it visibly robotic without exposing loose wiring. Ventilate the
rear tray and retain removable belly/back panels. Show a head tilt, arms-up
greeting and gentle tail wag. The rendering establishes visual direction only,
not part fit, support geometry or manufacturing feasibility.

## Power and runtime

The candidate 6S battery is **22.2 V nominal / 25.2 V fully charged**, not a
fixed 24 V source. The motor's nominal 24 V listing **does not approve direct
6S connection**. Confirm the exact drive's full input range, discharge behavior
and regenerative-voltage limits before selecting the bus. An incompatible
drive requires a pack or conversion redesign and a new mass/cost ledger.

The Jetson carrier's documented input range is **9–20 V** [13], so do not
connect it directly to 6S. The provisional regulated branches are:

| Branch | Candidate supply | Loads |
| --- | --- | --- |
| Compute, retained during motor cutoff | One D36V50F12 | 12 V Jetson, accounting for camera/USB draw |
| Head | Two D36V50F12, one per motor | Separate 12 V injections; published stall current 2.3 A per motor |
| Character | Two D36V50F5, two motors per branch | Separate 5 V injections; two published stall currents sum to 2.94 A per branch |

Five regulators total **$199.75** [14]. Headline currents are typical thermal
figures under specified conditions, not guaranteed enclosed-robot outputs;
qualify at our input voltage and ambient temperature. Separate branches avoid
depending on one marginal rail for all head or character loads. Do not parallel
regulator outputs. Maintain correct data/ground connections while isolating
power injections and keying 5 V/12 V connectors. Sustained stall is not allowed.

Two U2D2 adapters separate head/character communication groups; they **do not
supply actuator power** [15]. Hardware-disable both small-servo and leg power
through the watchdog/cutoff path, not only a Jetson process. Provide DC-rated
switching, branch fuses, cell/current monitoring, strain relief and tested
regenerative-energy handling. Power removal can drop a biped, so use overhead
fall support during testing. The $250 distribution allowance is not a finished
circuit or a selected contactor. A USB power Y adapter/hub is budgeted if the
camera needs it; verify port current and prevent backfeeding.

A 3.3 Ah pack stores 73.26 Wh nominal. Assuming 80% usable energy gives:

| Average whole-robot **battery** power | Energy-only runtime |
| ---: | ---: |
| 120 W | 29.3 minutes |
| 160 W | 22.0 minutes |
| 200 W | 17.6 minutes |

Target a **20–30 minute demonstration session**, then measure it. These are
scenarios, not predicted walking draw. Allow roughly 25–40 W for the complete
compute/camera/interfaces subsystem as an initial scenario; NVIDIA's 7–25 W
module modes do not equal whole-board or robot consumption. Size conductors
and fuses from measured bus transients and component limits, not summed motor
phase currents or the advertised battery C rating.

## Budget

The [line-item BOM](parts_budget.csv) and [editable inputs](parts_budget.json)
distinguish catalog prices from allowances. One battery is installed; a second
is a development spare. The spare RMD can first serve as the qualification unit.
No supplier discounts or owned equipment are deducted.

| Cost group | USD |
| --- | ---: |
| 18 installed actuators | 5,518.74 |
| Compute, camera, storage, USB hardware and audio | 916.00 |
| MCU, CAN/TTL bridges, pelvis IMU, foot sensing | 264.84 |
| Installed battery, five regulators, distribution and wiring | 676.94 |
| Structure, bearings, feet, shells and planned reprints | 1,200.00 |
| **Installed parts allowance** | **8,576.52** |
| Spares, comparison motor, charger, supply, fixture, tools, gamepad and cloud | 1,817.07 |
| **First-project subtotal** | **10,393.59** |
| 25% development contingency | 2,598.40 |
| **Project with contingency, before delivery/tax reserve** | **12,991.99** |
| Freight, taxes and import-cost planning reserve | 900.00 |
| **Funding plan** | **13,891.99 — round to $14,000** |

Use **$13,000–15,000 as a working funding band**, not a guaranteed cap. The $900
is a planning allowance, not a jurisdiction-specific tax calculation; replace
it with checkout/landed quotes. Excludes labor, a new host computer, printer or
machine tools, outsourced engineering, certification and event operations.
The $150 basic-tools allowance assumes access to a reasonable workshop;
substantial tooling or a higher-power programmable supply adds cost.

The earlier $5,600–8,000 proposal used cheap leg motors and broad, optimistic
electronics/integration allowances. Stronger legs alone add about $2,900 to
installed cost. This plan also includes fuller compute/camera, separate power
branches, a real battery, interfaces, spares and development equipment. Jetson
itself accounts for a small part of the increase.

| Mass allocation | g |
| --- | ---: |
| Leg / head / character motors | 3,960 / 164 / 72 |
| Load-bearing structure | 650 |
| Shells and guards | 280 |
| Moving tail / arms | 80 / 50 |
| Battery and restraint | 550 |
| Compute, storage, cooling and mount | 300 |
| Camera | 61 |
| MCU, interfaces, IMU, contacts and audio | 200 |
| Power boards, distribution and wiring | 350 |
| Bearings, feet and fasteners | 300 |
| **Allocated mass** | **7,017** |
| Integration reserve | 600 |
| **Design budget** | **7,617** |

Catalog actuator/camera/battery masses anchor the ledger; other allocations are
unweighed. **Reassess above roughly 7.8 kg or outside 48–52 cm**: this is a project
review threshold, not a mechanical failure limit.

## Spending and qualification sequence

1. **About $1,200–1,600 for the mechanical/control bench:** one RMD, one AK45
   comparison motor, MCU/CAN, current-limited supply, load fixture and measurement.
   These are already represented above, not an additional charge. Confirm exact
   revision, voltage/protocol, then measure sustained stance-like heating,
   torque calibration, backlash, current and shutdown behavior.
2. **About $800–1,000 for a vision/power bench:** Jetson, camera, storage,
   regulator and cabling, also included above. Verify runtime support, transients
   and policy timing before enclosing the computer.
3. **CAD and one leg:** resolve bearings, fasteners, cable bends, full joint
   travel and both ankle axes. Weigh parts and update the model; run inverse
   dynamics and collisions before buying the remaining motor set.
4. **Supported body, then walking:** validate sensor timing, polarity, limits,
   watchdog/cutoff and standing gestures. Train/test with measured actuator
   response, delays and friction. Progress to unsupported walking after supported
   qualification. The existing mini diagnostic is not transferable proof.

Keep the mini as a bounded learning build. This plan does not establish a need
for a separate finished 80 cm robot; a later raptor can use final-scale joint/leg
prototypes when its actuator class is selected.

Reproduce totals, mass/runtime scenarios and the limited load screen with
`python3 docs/hardware/studies/trex_50cm/calculate_parts.py`. It writes the
[results](parts_budget_results.json) and CSV, with input/script/baseline hashes.
This checks arithmetic/provenance, **not physical balance**. No new walking
validation is claimed by this documentation update.

## Sources checked 2026-10-24

1. [SparkFun Jetson kit](https://www.sparkfun.com/nvidia-jetson-orin-nano-developer-kit.html): retail price/availability.
2. [NVIDIA announcement](https://developer.nvidia.com/blog/nvidia-jetson-orin-nano-developer-kit-gets-a-super-boost/): launch price/module modes; [developer-kit guide](https://docs.nvidia.com/jetson/orin-nano-devkit/user-guide/latest/index.html).
3. Official Pi price history: [December 2025](https://www.raspberrypi.com/news/1gb-raspberry-pi-5-now-available-at-45-and-memory-driven-price-rises/), [February 2026](https://www.raspberrypi.com/news/more-memory-driven-price-rises/), [April 2026](https://www.raspberrypi.com/news/a-new-3gb-raspberry-pi-4-for-83-75-and-more-memory-driven-price-increases/), [October 2026](https://www.raspberrypi.com/news/price-increases-for-2gb-raspberry-pi-4-and-raspberry-pi-5/).
4. [PJRC Teensy 4.1](https://www.pjrc.com/store/teensy41.html); [SparkFun price](https://www.sparkfun.com/teensy-4-1.html).
5. [MYACTUATOR US RMD-X4-P12.5-10](https://myactuator.us/products/rmd-x-series/rmd-x4-p12-5-10).
6. [MYACTUATOR global X4-10](https://www.myactuator.com/x4-10details): compare exact revisions.
7. [ROBOTIS XM430-W210-T](https://robotis.us/products/dynamixel-xm430-w210-t).
8. [ROBOTIS XL330-M288-T](https://robotis.us/products/dynamixel-xl330-m288-t).
9. [CubeMars AK45-10 V3.0](https://www.cubemars.com/product/ak45-10-v3-0-kv75-robotic-actuator.html).
10. [Luxonis OAK-D Lite/accessories](https://shop.luxonis.com/products/oak-d-lite-1); [manufacturer dimensions/mass](https://checkout.luxonis.com/products/oak-d-lite-1).
11. [Adafruit BNO085](https://www.adafruit.com/product/4754).
12. [Gens ace 6S 3300mAh](https://genstattu.com/gens-ace-g-tech-3300mah-6s-60c-22-2v-lipo-battery-pack-with-ec5-plug/); [SKYRC S100neo specifications](https://www.skyrc.com/s100neo), charger price is an allowance.
13. [NVIDIA staff input-voltage confirmation](https://forums.developer.nvidia.com/t/power-supply-for-jetson-orin-nano/362326/5); [hardware layout](https://docs.nvidia.com/jetson/orin-nano-devkit/user-guide/latest/hardware_layout.html).
14. Pololu [D36V50F12](https://www.pololu.com/product/4095) and [D36V50F5](https://www.pololu.com/product/4091).
15. [ROBOTIS U2D2](https://robotis.us/products/u2d2): external motor power required.
