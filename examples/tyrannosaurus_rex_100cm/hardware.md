# Long-body one-metre T-Rex: hardware and budget

Revision 2, **2026-10-25**, USD. The design is 1.000 m tall, 2.104 m long and 0.450 m wide, with **33.210 kg of assigned mass**. This is an engineering estimate, not a purchase-ready kit. [bom.csv](bom.csv) separates list prices from allowances; [budget.json](budget.json) supplies quantities and arithmetic. No orders have been placed.

## Actuators and load paths

Use four RS04 drives for hip pitch and knees, eight RS03 drives for the other leg axes, three RS00 drives for the neck and tail, and three RS05 drives for the jaw and arms. Select the **CAN2.0 versions** and confirm exact production revisions, firmware, connectors and command protocol with the supplier.

| Application | Candidate / quantity | Allocated mass each | Case proxy, diameter × depth | Simulation torque cap | Listed unit price |
|---|---|---:|---|---:|---:|
| Hip pitch and knees | RobStride RS04 × 4 | 1.530 kg | 120 × 56 mm | 30 Nm | $255 |
| Hip yaw/roll; ankle pitch/roll | RobStride RS03 × 8 | 0.950 kg | 106 × 56 mm | 20 Nm | $225 |
| Head yaw / head nod / tail yaw | RobStride RS00 × 3 | 0.320 kg | 57 × 52 mm | 3 / 4 / 2 Nm | $125 |
| Arm lift, each / jaw | RobStride RS05 × 3 | 0.250 kg | 46 × 45 mm | 0.8 / 1.2 Nm | $110 |

Prices were checked on the manufacturer's [RS03](https://robstride.com/products/robStride03), [RS04](https://www.robstride.com/products/robStride04), [RS05](https://www.robstride.com/products/robStride05) and [product list](https://robstride.com/) pages. The four lines total **$3,525 for 18 motors**. Storefront specifications still differ from the latest technical PDF; the exact stocked revision and landed price need a written quote.

Dimensions and masses were checked against the manufacturer's [October 2026 specification](https://github.com/RobStride/Product_Information/blob/main/灵足时代RS系列产品规格介绍(2026.010.09).pdf), Git blob `ed6cc9cf1ff9c760cc92d6e04df3578a51e851df`. The dimension drawings are PDF pages 3, 19, 23 and 27 for RS00/03/04/05 respectively. The model uses rounded masses above the stated mass plus tolerance: RS00 310±3 g, RS03 928±20 g, RS04 1510±20 g, RS05 243±3 g. Case proxies include conservative body depth; they omit connectors, shafts, screw access and mounting brackets.

**Rated torque depends on cooling.** The October document gives RS03 21 Nm with a 200 × 200 mm aluminum plate, RS04 35 Nm with 220 × 200 mm (40 Nm with 345 × 345 mm), RS00 5 Nm with 150 × 150 mm, and RS05 1.8 Nm with 150 × 150 mm. Those plates are large compared with the robot's joints. The included **1.2 kg / $300 leg cooling allocation does not demonstrate equivalent heat rejection**, nor qualify the small expression drives' enclosed duty. The MJCF caps are provisional engineering limits, not measured continuous torque. Transient peaks cannot replace a thermal design.

Use metal load paths and separately designed output bearings, especially at hips, ankles, the cantilevered neck and tail root. The 9 mm minimum home case clearance and sampled gesture clearance only concern motor bodies. They do not prove bearing fits, structural strength, joint stops, wiring or every simultaneous pose. Keep long skins light and removable; do not use a solid printed head or tail.

## Basic verification and the walking constraint

The committed [stance report](validation.json) passes all five 10-second tests. In unsupported stance/gesture trials, settled knee RMS is approximately **17.5–18.0 Nm**, with a maximum knee transient around **22.6 Nm**, below the 30 Nm cap. No floor support comes from the tail, torso or a fixture. The ordinary standing controller is joint position control, not a trained balance or walking policy.

The [parts report](parts_verification.json) samples 729 combinations of expression-joint limits/midpoints:

| Assembly | Largest sampled gravity torque about powered axis | Cap | Separate transverse gravity moment |
|---|---:|---:|---:|
| Head nod | 2.10 Nm | 4.0 Nm | 0.14 Nm |
| Jaw | 0.58 Nm | 1.2 Nm | approximately zero |
| Each arm | 0.12 Nm | 0.8 Nm | approximately zero |
| Head yaw | zero with a level torso | 3.0 Nm | **4.97 Nm** |
| Tail yaw | zero with a level torso | 2.0 Nm | **2.04 Nm** |

A vertical yaw axis has little gravity torque about itself; its bearing still supports bending. The long tail has an 850 g moving allocation including root bearings, of which 650 g is shell/spar. Each arm's moving shell allocation is 160 g. Tail wag in the demonstration is ±0.12 rad at 0.5 Hz; angular acceleration and impacts are additional to the static table. Bearing selection needs dynamic loads and an explicit structural margin. These sampled poses are not an exhaustive collision or load certification.

A useful first-order walking screen is **τ ≈ F × d**. At 33.21 kg, one bodyweight is 325.8 N. A single support joint with an 80 mm horizontal ground-force lever arm sees about **26.1 Nm** from that force alone; at 1.5 bodyweights it becomes **39.1 Nm**. The latter exceeds the current 30 Nm pitch/knee cap, and even the static example exceeds the 20 Nm ankle cap. At 1.5 bodyweights the corresponding lever-arm ceilings are approximately **61 mm at 30 Nm** and **41 mm at 20 Nm**, before other moments.

That simplified force-moment calculation is not inverse dynamics: segment gravity, horizontal force, acceleration and joint geometry also matter. It explains why these motors are **candidates for a slow, carefully planned gait**, not verified walking actuators. With the present crouch, shifting both-foot knee loads onto one leg can consume the available margin. Resolve this with a measured single-leg rig and simulated weight-transfer/step trials. Options are a less crouched support phase, lower mass, validated short-duration torque above the planning caps, or stronger drives. Do not commit to fast or robust untethered walking on the standing result alone.

## Compute and control

Retain a **Jetson Orin Nano Super 8 GB developer kit** for perception, character behavior, logging and eventual policy inference. An independent **Teensy 4.1** handles acquisition, actuator commands, joint limits and watchdogs. A hardware E-stop/deadman controls motor power independently of the Jetson. The firmware and fault behavior are still to be implemented.

[NVIDIA](https://developer.nvidia.com/blog/nvidia-jetson-orin-nano-developer-kit-gets-a-super-boost/) specifies up to 67 INT8 TOPS and 7–25 W module modes. Budget 60 W for the complete compute/peripheral allocation, retain fan cooling and benchmark the intended workload. Height alone does not justify a larger Jetson. The 650 g compute allocation occupies 130 × 120 × 50 mm; the separate control/CAN/IMU allocation is 220 g in 70 × 110 × 40 mm. These boxes are planning envelopes, not verified installed board/connector stacks.

Use three separate classic CAN channels: one per leg and one for expression. Teensy needs external, 3.3 V compatible transceivers and appropriate termination. Initial targets of 250 Hz leg command/feedback and 50–100 Hz expression require a measured 1 Mbps bus budget; six drives per leg can make feedback traffic significant. The actuator's inner loop is onboard. No achieved bus rate or latency is claimed.

## Sensors and placement

| Proposed part | Location | Purpose and limitation |
|---|---|---|
| [Luxonis OAK-D Lite, fixed focus](https://shop.luxonis.com/products/oak-d-lite-1) | Snout, optical centres about 890 mm above floor | RGB and stereo depth; 120 g installed allocation carried by head |
| [Seeed Grove BMI088, 101020584](https://www.seeedstudio.com/Grove-6-Axis-Accelerometer-Gyroscope-BMI088.html) | Rigid pelvis, about 615 mm above floor | Base gyro/accelerometer; calibrate bias, vibration and timestamps |
| Motor encoders | All 18 drives | Joint state and moving-camera transform; output zeros/backlash need calibration |
| [Phidgets 3136_0 50 kg compression cells](https://www.phidgets.com/?prodid=227), eight | Four per sole | Contact and centre of pressure after a mechanically guided load path and calibration |
| Two [ADS1256-based boards](https://www.waveshare.com/product/high-precision-ad-da-board.htm) | Protected foot/leg acquisition mounts | Four differential channels each; excitation, filtering and effective sampling rate need bench verification |
| Bus voltage/current, battery and drive temperatures | Power tray, pack and drive telemetry | Electrical and thermal fault detection |
| Microphone, speaker, status LEDs | Torso/head allowance | Character interaction |

**The eyes are cosmetic.** The former chest camera was partly blocked by the long jaw, so revision 2 moves it into the snout. All 70 sampled rays clear the visible robot at home. This checks a limited cone, not the complete calibrated field of view or every head pose. Camera-to-body transforms must follow head joints with synchronized timestamps; perception now moves during gestures.

The MJCF's 75 mm stereo baseline and 60° vertical field of view are illustrative. At the home height, its bottom ray reaches level ground approximately **1.54 m ahead of the camera**. Head nod can change this, but does not establish dependable cliff detection. Near-foot depth/ToF sensors would be an additional design and budget item before autonomous travel around drop-offs. No lidar is included.

Phidgets lists 50 kg maximum load and 60 kg maximum overload per cell. A foot's load can concentrate at a corner, so four cells do not guarantee impact capacity. Provide compression guides, mechanical overload stops and cable relief. Sensor mass and foot hardware are allocated, but detailed installation is unresolved. Ideal MuJoCo touch patches are not load-cell CAD, and the ADC headline sample rate is not the synchronized eight-channel rate.

## Battery and power

Specify a supplier-built **12S2P Molicel P45B pack** with BMS, temperature sensing, enclosure, restraint and a matched charger. The [cell data sheet](https://www.molicel.com/wp-content/uploads/INR21700P45B_1.4_Product-Data-Sheet-of-INR-21700-P45B-80109.pdf) gives the inputs for these calculations:

- 24 cells; **43.2 V nominal / 50.4 V charged**, 9 Ah typical / 8.6 Ah minimum.
- **388.8 Wh typical** before operating reserve and conversion losses.
- 2.8 kg installed allowance in 180 × 120 × 90 mm. A 6 × 4 array of maximum-size bare cells is 129.3 × 86.2 × 70.15 mm, but BMS, insulation, busbars and connectors still need supplier packaging.
- Request about 40 A continuous pack capability with measured short transients and temperature limits. The cell's temperature-limited 45 A rating is not a complete-pack rating.

The October motor document lists 15–60 V operation, so 50.4 V charged is within the stated input range. Speeds/performance characterized at nominal 48 V cannot be assumed throughout discharge. A pack fault or motor-power shutdown can make a biped fall; early mechanical testing needs a support stand/tether.

Include a DC-rated fuse, precharge, contactor, physical E-stop/deadman, current sensing and separately protected compute supply. Exact DC/DC modules remain a **$200 allowance**, not a verified fitted part. Their input/transient ratings must exceed the full pack voltage and their output must suit the Jetson carrier. For reference, the [Orin Nano carrier specification](https://developer.nvidia.com/downloads/assets/embedded/secure/jetson/orin_nano/docs/jetson_orin_nano_devkit_carrier_board_specification_sp.pdf) gives 9–20 V at the DC jack. Do not connect the motor rail directly to it.

A reviewed alternative, [Mean Well RSD-100C-12](https://www.meanwell.com/Upload/PDF/RSD-100/RSD-100-spec.pdf), has a 161 × 68 × 36 mm case and 563 g stated weight. It is not selected: even the bare case needs a larger dimension than the current 50 × 160 × 90 mm power tray permits, before terminals, protection and cooling. A compact converter selection or tray redesign is still necessary. A [Pololu D24V50F5](https://www.pololu.com/product/2851) is a possible downstream 12-to-5 V logic stage; its 38 V maximum input rules out direct connection to this pack. The 750 g power/harness tray allocation must be revisited when exact modules are selected.

Budget a bus brake chopper and pulse-rated resistor for regeneration at full charge or BMS disconnect. A nominal clamp around 55 V, with all tolerances and overshoot below the motors' 60 V ceiling, is an unverified design requirement. Battery charge acceptance and a bench supply are not substitutes for that analysis. Do not add phase-current peaks as if they were battery currents.

For energy planning only, 75% of typical pack energy is 291.6 Wh:

| Assumed whole-robot average electrical draw | Calculated time |
|---:|---:|
| 500 W | 35 minutes |
| 800 W | 22 minutes |
| 1,200 W | 15 minutes |

These are scenarios, not measured runtime. The simulation does not estimate electrical consumption or temperature.

## Cost structure

| Budget group | USD |
|---|---:|
| 18 integrated actuators | $3,525.00 |
| Compute, accessories, Teensy and CAN interfaces | $509.00 |
| Vision, IMU, foot sensing/acquisition and audio/controls | $987.15 |
| Battery, charger, conversion, protection and regeneration | $1,650.00 |
| Harness and joint cooling | $700.00 |
| Frame, bearings/feet/hardware and character shells | $4,150.00 |
| **Robot hardware subtotal** | **$11,521.15** |
| Development spares, stand, leg fixtures and bench tools | $2,265.00 |
| Contingency, about 25% rounded upward | $3,500.00 |
| **First prototype planning total** | **$17,286.15** |

Plan approximately **$17,000–$20,000 before tax, freight, duty and engineering labor**, subject to fabrication and pack quotes. The hardware subtotal is not the total development cost. Engineering hours, extra failed iterations, training computers, certification and public-demo operation are outside this estimate. Existing bench equipment may reduce incremental spend.

The Jetson $249 amount is reference MSRP; current local availability needs confirmation. Fabrication, cooling, power modules and custom pack costs are allowances, not supplier quotes. Compared with the earlier $14,136 concept, this revision adds approximately **$3,150**, mainly in structure, bearings, shells, cooling and development provisions; the motor increase is only $185. Budget detail and CSV consistency are checked automatically.

## Practical build gates

1. Obtain exact motor CAD/revision quotes and complete a metal frame, bearing, stop, cable and cooling design. Replace envelope masses with measured CAD and purchased-part values. Resolve the power tray and pack drawings.
2. Build the power/control bench and one instrumented leg. Measure torque-speed response, sustained temperature, contact sensing, backlash, bus delay, regenerative transients and the actual crouch/step loads.
3. Assemble an overhead-supported chassis, then demonstrate unsupported standing and controlled weight transfer. Add validated foot-clearance, single-support and stepping tests before claiming walking.
4. Add the light removable head/tail shells and restrained gestures while monitoring their effect on balance. Validate sensor visibility through the commanded motion envelope.

The delivered MuJoCo model, renders, ledger and regression reports make this concept reviewable. They do not yet make it fabrication-ready or a verified walking robot.
