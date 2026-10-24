# Experimental Mini T-Rex

The [Mini T-Rex example](../../examples/mini_tyrannosaurus_rex/README.md) is a compact,
cartoon robot model with forward knees and partially covered mechanisms. It is
approximately 25 cm tall and has an **assigned**, unweighed mass of 850 g.
It is an experimental MJCF example, not a registered training plant or a
demonstrated walking robot. The existing anatomical T-Rex plant is unchanged.

## Proposed computer and sensors

| Component | Proposed starting choice | Purpose |
| --- | --- | --- |
| Computer | Radxa ZERO 3W, 4 GB RAM, 32 GB microSD | State estimation, exported policy, behavior and logging |
| Body IMU | Adafruit ISM330DHCX, product 4502, over SPI | Three-axis angular velocity and acceleration, rigidly mounted in the torso |
| Servo interface | ROBOTIS U2D2, USB to three-pin Dynamixel TTL | Communication with the smart servos; it does not supply motor power |
| Joint sensing | XL330 position, velocity, input current, voltage and temperature | Existing motor telemetry; current is not a calibrated joint-torque measurement |
| Optional head IMU | A second ISM330DHCX | Head-motion feedback and gaze experiments |
| Optional sensor bridge | Raspberry Pi Pico 2 | IMU timing, foot-contact ADC inputs and a watchdog protocol |

The Radxa is a 65 × 30 mm board with an RK3566 processor, four Cortex-A55 cores
up to 1.6 GHz and 3.3 V GPIO. Use headless ARM64 Linux and benchmark the actual
sensor driver, estimator, inference and bus latency before choosing a control
rate. A proposed initial target is 200–400 Hz body-IMU sampling and a 50 Hz
policy loop; these rates have not been demonstrated by this example. Train on
an external GPU, then run an exported policy on the robot's CPU. There is no
trained policy or hardware driver in this contribution.

Use 3.3 V host logic. The Adafruit breakout supports SPI and I2C; its STEMMA QT
connectors carry I2C, while SPI requires the appropriate pads, including DO and
CS. Verify the board's pinmux and Linux device configuration rather than assuming
Raspberry Pi GPIO compatibility. A second head sensor can initially use an
enabled I2C bus. A Pico bridge would require new firmware; it does not directly
replace the direction-controlled half-duplex motor interface.

Supply the Radxa from a clean 5 V logic rail with at least the manufacturer's
recommended 2 A supply capability, allowing additional peripheral headroom.
Provide a separately regulated 5 V motor rail, deliberate distribution and a
common ground. Size regulators and wiring from measured transient demand.
Mount the computer and battery in the torso to keep the head light.

The six-axis IMU supports an orientation estimator for local balance; it does
not give drift-free absolute yaw. Exact simulator quaternions, whole-body COM
and world-frame foot coordinates are privileged diagnostics. Thin foot FSRs
could later add loaded/unloaded observations through an ADC, but one per foot
cannot measure center of pressure. A camera is optional and adds head payload.

Primary references, checked 2026-10-24:

- [Radxa ZERO 3 specifications](https://docs.radxa.com/en/zero/zero3),
  [interfaces](https://docs.radxa.com/en/zero/zero3/hardware-design/hardware-interface)
  and [power](https://docs.radxa.com/en/zero/zero3/getting-started/preparation).
- [Adafruit ISM330DHCX](https://www.adafruit.com/product/4502) and
  [breakout pinouts](https://learn.adafruit.com/lsm6dsox-and-ism330dhc-6-dof-imu/pinouts).
- [ROBOTIS U2D2](https://robotis.us/products/u2d2),
  [XL330-M288-T](https://robotis.us/products/dynamixel-xl330-m288-t) and
  [XL330 manual](https://docs.robotis.com/docs/dxl/model_reference/x_series/xl_series/xl330-m288/).
- [Raspberry Pi Pico 2](https://www.raspberrypi.com/products/raspberry-pi-pico-2/).

## Preliminary $2,000 allocation

These are planning allowances, not a verified shopping list. They exclude labor
and assume fifteen active XL330-M288-T motors plus three spares. Component mass,
availability and supplier totals still need reconciliation with mechanical CAD.

| Allocation | USD |
| --- | ---: |
| Motors, including spares | 495 |
| Computer, storage, cooling and USB adapters | 100 |
| U2D2 | 40 |
| Body and optional head IMUs, wiring | 50 |
| Optional Pico 2 bridge and connection materials | 20 |
| Motor/logic regulation, distribution and cutoff | 120 |
| Battery and matched charger | 100 |
| Printed structure and shell revisions | 120 |
| Bearings, fasteners, cables and foot material | 90 |
| Speaker, LEDs and buttons | 40 |
| Bench supply and tools | 180 |
| Suspension/test fixtures | 50 |
| Training compute | 80 |
| Tax and shipping allowance | 150 |
| Engineering reserve | 365 |
| **Total** | **2,000** |

## What remains to verify

The model's 0.30 N m leg cap is a provisional transient command limit. ROBOTIS
lists 0.52 N m **stall** torque at 5 V and approximately 0.10 N m **estimated**
rated torque, with the latter calculated as 20% of stall. It is not a qualified
thermal duty envelope. Preliminary contact-equilibrium calculations give about
0.0879 N m hip-roll and 0.0618 N m knee demand at home; a sustained 1 N backward
load raises the knee example to approximately 0.1031 N m. These leave limited
reserve and are design estimates, not measured motor loads.

The neck/head/jaw allocation is 174 g. A separate kinematic gravity sweep gives
a sampled neck load near 0.0522 N m; additional front-mounted mass increases
it. The coincident head axes are an abstraction, and some combined nod poses
intersect collision proxies. Physical motor, bearing, horn, connector and cable
placement must precede approval of that motion range.

Before selecting the complete actuator set, build a loaded neck and one leg,
measure holding/gesture current and temperature over the intended duty, and
measure external-angle error through small reversals. Encoder increments of
0.0879 degrees do not guarantee that mechanical accuracy. Then reconcile mass,
power and travel in CAD, identify the real actuators, and design a hardware-
observable stance/walking task. Short ideal-actuator holds in MuJoCo do not
establish walking, push recovery, full-range fit or continuous-duty capability.
