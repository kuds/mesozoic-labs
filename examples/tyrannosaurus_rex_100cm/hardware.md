# One-metre T-Rex: hardware and budget

Dated **2026-10-25**, USD. This is an initial engineering bill of materials, not a purchase-ready kit. Listed prices and explicit planning allowances are distinguished in [bom.csv](bom.csv); [budget.json](budget.json) includes the arithmetic and development extras. No orders have been placed.

## Recommended architecture

Use **12 RobStride 03 leg drives, two RobStride 00 neck drives, and four RobStride 05 expression drives**. A common 48 V-class motor family reduces power-rail and interface variety. This is a proposed integration, not a claim that firmware, connectors or CAN protocols are interchangeable without checking the exact revisions.

A **Jetson Orin Nano Super 8 GB developer kit** handles vision, character behavior, logging and eventual policy inference. An independent **Teensy 4.1** handles timestamped proprioception, actuator commands, limits and watchdogs. The Jetson is not the emergency-stop controller. The kit is a practical prototype choice; a production carrier is a later decision.

NVIDIA specifies up to 67 INT8 TOPS and 7–25 W module power modes. Those numbers do not include the whole robot, all peripherals, or guarantee a perception/control deadline. Reserve **60 W for compute and attached peripherals**, retain cooling, and benchmark the intended workload. A larger Jetson is not justified by height alone.

Use three separate CAN channels: left leg, right leg, and expression. Teensy needs external transceivers. A preliminary target is 250 Hz command/feedback per leg, 50–100 Hz expression, and faster local sensing/estimation. Six motors per classic 1 Mbps leg bus need a measured bandwidth and worst-case latency budget; this is not an implemented firmware rate. Motor inner loops remain onboard the actuators.

## Motors and packaging

| Use | Candidate / count | Assigned motor mass | Body envelope | Simulation torque cap |
|---|---|---:|---|---:|
| All six axes in each leg | RobStride 03 × 12 | 0.900 kg each | 106 × 106 × 56 mm | ±20 Nm |
| Head yaw and pitch | RobStride 00 × 2 | 0.310 kg each | 57 × 57 × 51 mm | ±2.0 Nm |
| Two arms, jaw, tail yaw | RobStride 05 × 4 | 0.191 kg each | 46 × 46 × 44 mm | ±0.8 Nm |

The official [RobStride product-information repository](https://github.com/RobStride/Product_Information) reports, in its July 2026 summary, RS03 at 21 Nm rated / 60 Nm peak and 900 g; RS00 at 5 / 14 Nm; RS05 at 1.8 / 5.5 Nm. Its rated torques depend on specified aluminum heat sinks. The [RS03 storefront](https://robstride.com/products/robStride03) instead lists 20 Nm and 880 g. The model carries the heavier 900 g allocation and caps at 20 Nm. **Resolve the actual production revision, drawings, heat-sink conditions and performance curves before buying.** The repository also lists a newer October specification PDF; this concept is not represented as qualified against that unread revision.

Leg cases are cylindrical body proxies. Expression cases are bounding boxes. Neither includes all connectors, output hardware, bolt access, or cooling surfaces. Use metal structural paths and external bearing support where the output moment demands it. Do not assume printed shells carry hip or ankle loads.

The small motor mix gives the arms and tail useful headroom without making them heavy moving assemblies. The tail motor is in the torso at the root; only the 350 g allocated shell/spar moves. Each arm's moving allocation is 120 g. Preserve these allocations when changing the character.

A simple gravity estimate explains why standing is not walking qualification: at 25.1 kg, supporting the whole robot with an 80 mm horizontal moment arm requires approximately **19.7 Nm** before acceleration, friction and impact loads. That is already close to the 20 Nm planning cap. A gait must deliberately transfer weight and limit moments; advertised 60 Nm peaks cannot be treated as continuous operating torque. Do not reduce the ankle/hip motors solely because the double-support stance trace looks easy.

## Sensors and their locations

| Sensor | Location | Role / limitation |
|---|---|---|
| Luxonis OAK-D Lite, fixed focus | Front chest, about 665 mm above floor | RGB + stereo depth for people and obstacles; independent of head turns |
| Seeed Grove BMI088 | Rigid pelvis frame, about 500 mm above floor | Gyro/accelerometer for base estimation; verify I2C timing, vibration and bias |
| Integrated actuator encoders | Every motor | Joint position/velocity; calibrate output zero and backlash |
| Phidgets 3136_0 compression cells, four per foot | Under the load-bearing foot plate | Contact and centre-of-pressure estimate after calibration and mechanical mounting |
| Bus current/voltage + temperature monitoring | Battery, distribution and motor telemetry | Power/thermal limits and fault detection |
| Microphone, speaker, status LEDs | Torso / light head package | Character interaction; budget allowance |

**The cameras are not in the eyes.** Keep the eyes visually expressive and inexpensive. There are two stereo camera poses in the MJCF; they do not simulate a complete OAK camera pipeline or provide photorealistic RGB.

The camera model uses a nominal 75 mm stereo baseline and 60° vertical field of view as illustrative optical geometry, not calibration values. At 665 mm height, a level camera with that field of view first intersects flat ground about **1.15 m ahead**. It cannot see the ground immediately beside the feet. The initial operating envelope is a known, flat indoor floor under supervision. Stairs, drop-offs and cluttered autonomous travel require additional near-ground sensing and validation; no lidar or cliff sensor is silently included in this budget.

The button cells are specified by Phidgets for 50 kg maximum load with 60 kg maximum overload. Four cells do not make a foot safe under arbitrary impacts: a corner can carry most of a landing. Provide overload stops, guided compression, wire relief and a calibrated load path. The selected cells' exact installed mass and the foot stack-up remain unresolved within the model's allocated foot hardware/sensor masses. Do not mistake the ideal MuJoCo touch patches for detailed load-cell CAD.

Two ADS1256-based acquisition boards offer four differential inputs each; the listed Waveshare board is a prototype reference. Provide stable bridge excitation, compatible input range, shielding and timestamped acquisition. Its headline converter sample rate is not the effective synchronized rate of eight filtered load channels.

## Battery, power and runtime

Specify a professionally assembled **12S2P Molicel P45B pack**, including BMS, temperature sensing, enclosure, restraint, main connector and pack-matched charger:

- 24 cells; **43.2 V nominal, 50.4 V fully charged**.
- 9 Ah typical / 8.6 Ah minimum using the cell data sheet.
- **388.8 Wh typical**, before conversion losses and operating reserve.
- Initial installed allocation **2.8 kg in 180 × 120 × 90 mm**. Twenty-four maximum-diameter cells can fit geometrically in a 6 × 4 array, but insulation, BMS, clearances and service access still require the pack supplier's drawing.
- Procurement design target: approximately 40 A continuous discharge and measured short transient capability, with temperature limits. This is a requested pack rating, not one established by the cell rating alone.

The [P45B data sheet](https://www.molicel.com/wp-content/uploads/INR21700P45B_1.4_Product-Data-Sheet-of-INR-21700-P45B-80109.pdf) specifies 3.6 V nominal / 4.2 V charge and 70 g maximum per cell. Its 45 A cell discharge specification includes a temperature cutoff. The complete pack must be derated for its enclosure, BMS, connections and cooling.

The selected motor family accepts the full pack voltage according to the referenced vendor summary. Lower pack voltage than the 48 V nominal motor test condition may reduce available speed/power; the ideal MJCF does not reproduce this.

Provide a DC-rated fuse, precharge, main contactor, physical E-stop/deadman and bus-current measurement. Use a separate protected compute supply so motor faults do not directly brown out logging. The exact compact DC/DC modules remain a procurement/design item: their input and transient ratings must exceed 50.4 V, and their output must meet the chosen Jetson carrier's requirements. There is no high-current 12 V expression rail in this architecture.

A bus brake chopper and pulse-rated resistor need to handle regeneration when the pack is full or the BMS disconnects. A nominal clamp near 55 V with all tolerances and overshoot below the motors' 60 V ceiling is a **design requirement to verify**, not a tested setting. Do not rely on a BMS or a bench supply to absorb arbitrary braking energy. Individual motor phase-current peaks are not quantities to sum directly into battery current.

For budgeting only, using 75% of typical battery energy gives 291.6 Wh:

| Assumed average whole-robot electrical draw | Calculated operating time |
|---:|---:|
| 500 W | 35 minutes |
| 800 W | 22 minutes |
| 1,200 W | 15 minutes |

These are energy scenarios, not measured runtime predictions. The stance simulation produces torque traces, not electrical consumption or motor temperature. A torque-off fault can make a biped fall; early hardware tests need a support/tether arrangement.

## Cost structure

| Expected hardware | Extended USD | Basis |
|---|---:|---|
| 12 × RobStride 03 integrated CAN actuator | $2700.00 | manufacturer list |
| 2 × RobStride 00 integrated CAN actuator | $240.00 | planning allowance |
| 4 × RobStride 05 integrated CAN actuator | $400.00 | planning allowance |
| 1 × NVIDIA Jetson Orin Nano Super 8GB developer kit | $249.00 | reference MSRP allowance |
| 1 × NVMe SSD; fan duct; mounts; cables | $120.00 | planning allowance |
| 1 × Teensy 4.1 | $40.00 | planning allowance |
| 1 × Three 3.3 V compatible CAN transceiver channels and termination | $100.00 | planning allowance |
| 1 × Luxonis OAK-D Lite fixed focus | $269.00 | manufacturer list |
| 1 × Seeed Grove BMI088 SKU 101020584 | $28.15 | manufacturer list |
| 8 × Phidgets 3136_0 50 kg compression button cell | $360.00 | manufacturer list |
| 1 × Two four-channel differential ADS1256-based boards plus excitation and filtering | $180.00 | planning allowance |
| 1 × Speaker amplifier microphone status LEDs and remote deadman | $150.00 | planning allowance |
| 1 × Custom 12S2P Molicel P45B pack with BMS enclosure restraint and temperature sensing | $700.00 | custom pack allowance |
| 1 × Pack-matched 50.4 V CC/CV charger | $150.00 | planning allowance |
| 1 × High-input-voltage DC/DC stages for compute and logic rails | $200.00 | planning allowance |
| 1 × DC-rated main fuse contactor precharge emergency stop and current monitor | $300.00 | planning allowance |
| 1 × Bus overvoltage brake chopper and resistor | $200.00 | planning allowance |
| 1 × Connectors shielded signal wiring power wiring strain relief and service loops | $300.00 | planning allowance |
| 1 × Machined or cut aluminum frame hip brackets and leg links | $1500.00 | fabrication allowance |
| 1 × Output support bearings shafts fasteners feet rubber soles and overload stops | $700.00 | planning allowance |
| 1 × Light printed or formed shells jaw tail spar guards and finish | $450.00 | fabrication allowance |
| **Robot hardware subtotal** | **$9336.15** | Includes fabrication allowances |
| Development spares, support stand, bench equipment | $1900.00 | Allowance |
| Contingency, approximately 25% of the above | $2900.00 | Rounded upward |
| **First prototype planning total** | **$14136.15** | Before tax, freight, duty and engineering labor |

A useful cash target is **about $14,000–$16,000 before engineering labor**, subject to machining and pack quotes. The approximately $9,300 hardware subtotal is not the complete development cost. Multiple failed mechanical iterations, outsourced engineering, a dedicated GPU training workstation, certification, travel and public-demo staffing are outside it.

The RS00/05 USD amounts are allowances, not currency conversions from the vendor's CNY table. The Jetson amount is reference MSRP from NVIDIA's announcement, not a confirmed local offer. Every purchased SKU needs availability and landed-cost confirmation. Existing tools and spares may reduce the additional cash needed.

## What is ready, and what remains

The deliverable contains a runnable MJCF, explicit mass ledger, actual-geometry renders and reproducible basic stance/gesture checks. It deliberately carries a 1.8 kg integration reserve.

The unresolved procurement items are the exact motor revision, custom battery, compact power converters/protection, bearing/bracket details and foot-sensor stack. These must become measured or supplier-CAD inputs before fabrication release. The mass ledger's boxes express allocation; they do not prove all boards, connectors and cooling ducts can be assembled and serviced.

The next useful hardware step is one instrumented leg and the power/control bench, followed by an overhead-supported full chassis. Walking policy development should use measured actuator response, joint limits, foot compliance, communication delay and mass properties. This concept includes no walking policy and no claim that it is ready to walk with purchased hardware.
