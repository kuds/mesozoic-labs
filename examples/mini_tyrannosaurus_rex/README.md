# Mini Tyrannosaurus Rex (experimental robot)

![Forward-knee Mini T-Rex rendered from the MJCF](preview.png)

A standalone **Mini Tyrannosaurus Rex** robot study with custom **forward-bending knees**, partial
removable-looking leg covers and exposed servo/bracket mechanisms. The MJCF is
self-contained: all geometry is primitive or original mesh data in the two
included XML assets. No mesh downloads or external STL files are needed.

This example is **not registered as a Mesozoic Labs plant, Gymnasium environment
or training embodiment**. It has no trained walking policy. Its five-axis leg
architecture was informed by MicroDuck, but these custom joint locations, case
placements, zero positions and limits do not preserve MicroDuck kinematics or
policy compatibility.

Its prototype identifier is `mini_tyrannosaurus_rex`, and its MJCF model name is
`mini_tyrannosaurus_rex_robot`. The existing species ID `tyrannosaurus_rex` and
its anatomical model keep their current names. The `mini_` prefix identifies
this miniature robot prototype, rather than a new biological species.

## Run

From the repository root, using Python 3.11+ and the repository's pinned MuJoCo:

```bash
pip install -e .
python examples/mini_tyrannosaurus_rex/demo.py --check
python examples/mini_tyrannosaurus_rex/demo.py --view
python examples/mini_tyrannosaurus_rex/demo.py --view --support --gesture
python examples/mini_tyrannosaurus_rex/demo.py --render /tmp/mini-trex.png
```

On a headless Linux machine with EGL support, prefix the render command with
`MUJOCO_GL=egl`. On macOS, use MuJoCo's `mjpython` launcher for `--view`.
Rendering needs a working OpenGL backend; numerical checks do not open a viewer.
The PNG writer uses the standard library and adds no imaging dependency.

The native MuJoCo viewer can also open the XML directly. Keep its adjacent
`assets/` directory in place:

```bash
python -m mujoco.viewer --mjcf=examples/mini_tyrannosaurus_rex/model.xml
```

The default torso is free. `--support` enables an external torso weld, and
`--gesture` requires that support. The gesture demonstrates small head/jaw
motions while the torso is held; it is not balancing or walking.

## Model

| Property | Current concept |
| --- | --- |
| Approximate home envelope | 250 mm high × 160.6 mm wide × 284 mm long |
| Assigned total mass | 850 g |
| Actuators | 15: five per leg, four neck/head axes, one jaw |
| Leg home | 44.91 mm links; knee 14.5 mm forward of the hip–ankle line |
| Feet | 75 × 40 mm visual pads; inset 70 × 34 mm contact boxes |
| Servo proxies | Ten leg cases, full 20 × 34 × 26 mm XL330 body dimensions |
| Sensors | 39 sensor definitions / 59 scalar values, including 30 encoder channels |

All joint targets are zero at the home pose. Motor cases attach to their parent
carriers: yaw to torso, roll to yaw carrier, hip pitch to roll carrier, knee to
thigh and ankle to shin. Explicit masses and inertias follow these assigned
placements. Covers remain visual placeholders: their proposed 20–30 g allowance
must fit within the existing structural budget and has not been weighed.

Joint positions/velocities and body/head gyroscope/accelerometer readings model
hardware-observable telemetry. Absolute frame quaternions, subtree COM and world
foot positions are privileged simulator diagnostics, not direct hardware
measurements. Geom group 1 contains the character shell/covers, group 2 the
mechanical platform and group 3 the collision proxies.

## Limits and next work

`--check` compiles the model and checks home penetration plus short free-standing
and supported-gesture rollouts for finite state, warning-free integration,
upright posture and bounded actuator force. These are numerical smoke checks.
There is no contact-aware locomotion controller or tested disturbance recovery.

The 5 V XL330-M288-T reference provides 0.52 Nm **stall** torque. The manufacturer's
0.10 Nm continuous figure is explicitly an estimate from 20% of stall, rather
than a qualified continuous-duty rating. Home static hip-roll/knee loads are
approximately 0.0879/0.0618 Nm, so both exceed a desired 0.05 Nm static target for
2× reserve against that estimate. The 0.30 Nm model cap is a transient command
limit, not a sustained operating rating.

The four-axis neck/head motor stack is not physically packaged: several model
axes coincide and no complete servo/bracket/connector fit is demonstrated.
Sampled head gravity loads reach about 0.0522 Nm at the neck; payload and forward
COM movement worsen this. Encoder resolution is 4096 counts/revolution
(0.0879°/count), which does not establish loaded mechanical accuracy. Gearbox
backlash, compliance, wiring, thermal duty and actuator calibration remain
unvalidated. Some combined head poses already intersect coarse collision proxies;
the small supported demonstration does not establish an unrestricted head range.

See the [hardware proposal and budget](../../docs/hardware/MINI_TYRANNOSAURUS_REX.md) for
component choices, provisional power/compute integration and the staged build
plan. Physical CAD, complete motion clearance, bench measurements and subsequent
plant/training integration are future work.
