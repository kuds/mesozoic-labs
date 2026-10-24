# Mini Tyrannosaurus Rex: serviceable character revision

![Mini Tyrannosaurus Rex revised model](preview.png)

A standalone, original MJCF concept with forward knees, a lighter two-axis head,
an articulated jaw, two linked moving arms, and exposed hip/shoulder mechanisms.
The blunt snout, tiny two-finger hands, cream belly and curved tail remain. The
body has open hip areas, a separate rear-panel mesh and a battery-sled concept.

**The 800 g mass is an unweighed allocation, not a fabricated assembly.** This
example is outside the registered plant/training inventory and has no validated
walking policy. Its custom geometry is not compatible with a Microduck policy. Existing
anatomical species and generated catalogs are unchanged.

An **optional 825 g / 15-motor tail-wag configuration**, a reproducible unsupported
stepping pilot, and build-time balance regressions are described in
[Walking, tail and CI study](WALKING_AND_TAIL.md). The base model stays unchanged.
The eyes are cosmetic; the virtual snout camera has no selected physical module.

## Changes from the original concept

| Feature | Original | Revision 10 |
| --- | --- | --- |
| Assigned mass | 850 g | 800 g, including arm drive |
| Active motors | 15 | 14: ten legs, yaw/nod/jaw, shared arms |
| Head mechanism | Four schematic/coincident neck/head axes plus jaw | Spatially separated yaw/nod motor boxes plus jaw linkage concept |
| Arms | Fixed 10 g bodies | Two shoulder-pitch hinges linked 1:1; one actuator |
| Hip roll-to-pitch lateral offset | 30 mm | 15 mm, full motor cases retained |
| Forward knee offset | 14.5 mm | 10 mm, ankles and home height retained |
| Joint-center link lengths | 44.91 mm | 43.66 mm |
| Nominal forward knee bend | 37.7 degrees | 26.5 degrees, with less extension/crouch travel |
| Service access | Decorative back-panel primitive | Split rear mesh, open hip areas, panel-hidden component study |

The hip carrier flip and straighter knee reduce modeled home loads; they do not
increase motor ratings. The head exterior is 90% of its previous scale, while
all **14 motor boxes retain full 20 x 34 x 26 mm dimensions**. Assigned component
inertias were updated with the changed carriers and upper-body mass placement.

## Views and motion

[Four-view design sheet](design_views.png) shows the character, side, front and
torso-panel-hidden study. Group 4 contains raw/provisional part proxies, not the
larger service envelopes in `components.json`. Their placement is not a verified
packing solution. Battery dimensions are a drawing assumption; no pack is
selected. USB plugs, wire bends, mounts and cooling space still require CAD.

![Supported head, jaw and linked-arm demonstration](supported_gesture.gif)

The animation uses an actual simulated rollout with the torso externally held.
It demonstrates expression motion, not balancing or walking. The default model
has no external support. The jaw and shared arm drives require offset linkages;
their ideal transmissions do not establish real geometry, backlash or losses.

## Reproduce

From the repository root:

```bash
pip install -e '.[viz,test]'
python examples/mini_tyrannosaurus_rex/demo.py --check
python examples/mini_tyrannosaurus_rex/demo.py --view
python examples/mini_tyrannosaurus_rex/demo.py --view --support --gesture
python -m examples.mini_tyrannosaurus_rex.engineering --samples 2000 --points 5 --duration 10 --json examples/mini_tyrannosaurus_rex/validation.json
python examples/mini_tyrannosaurus_rex/render_design.py --output-dir examples/mini_tyrannosaurus_rex --sheet --gesture
pytest environments/tyrannosaurus_rex/tests/test_mini_tyrannosaurus_rex_*.py
```

Use MuJoCo's `mjpython` launcher for interactive viewing on macOS. The original
OpenGL `demo.py --render PNG` remains available. The new CPU renderer uses exact
compiled meshes/transforms and tessellated primitives, with Pillow from the
visualization extra; it requires no OpenGL connection. All MJCF geometry is
inline, with no external STL downloads.

## Interface and limits

| Property | Revision 10 |
| --- | --- |
| Home visual envelope | Approximately 258 mm high x 160.6 mm wide x 269 mm long |
| Generalized coordinates / velocities | 22 / 21, including free root |
| Hinge joints / active motors | 15 / 14; shoulders share one motor |
| Sensors | 39 definitions / 59 scalar values |
| Head yaw | -0.65 to +0.65 rad |
| Head nod | -0.25 to +0.10 rad |
| Jaw | 0 to +0.32 rad |
| Both shoulder pitches | -0.65 to +0.20 rad, linked 1:1 |
| Hip roll | +/-0.20 rad; wider trials caused collisions |
| Feet | 75 x 40 mm visual pads; 70 x 34 mm contact rectangles |

Home joint targets are zero. The unactuated linked shoulder's position/velocity
channels are simulator diagnostics: one physical motor supplies one encoder
unless another sensor is installed. Gyro/accelerometer channels represent IMUs;
exact quaternions, COM and world foot positions remain privileged diagnostics.

## Numerical evidence and limits

[validation.json](validation.json) records MuJoCo version, model/asset SHA-256
hashes, sampling seeds, support assumptions and per-actuator torque data.

| Check | Revision 10 result |
| --- | --- |
| Home static hip-roll load | Approximately 0.0369 Nm versus 0.0879 Nm originally |
| Home static knee load | Approximately 0.0495 Nm versus 0.0618 Nm originally |
| Motor boxes: home plus 2,000 seeded poses | No intersections; minimum separating-axis gap 2.5 mm |
| Head/jaw/arm grid: 625 poses | No non-floor collision-proxy penetration |
| Combined leg/expression sample: 2,000 poses | No non-floor collision-proxy penetration |
| Unsupported 10-second hold, transient caps | Upright, finite, no warnings; approximately 2.02 degrees maximum tilt |
| Same hold with 0.10 Nm hard leg caps | Upright; approximately 2.19 degrees maximum tilt |

Static equilibrium uses vertical forces at the sole lateral centers and sagittal
centers of pressure under the COM. It is one possible load distribution, not a
unique prediction of real servo loads. The hard-cap run is not a thermal model.
Its final-second means reach approximately **0.0337 Nm hip roll, 0.0647 Nm knee
and 0.0728 Nm ankle** (larger side). Load shifts toward the ankles during settling;
the home calculation is not a guaranteed continuous-duty margin. Neither 0.52 Nm
stall torque nor the manufacturer's approximately 0.10 Nm continuous estimate
replaces a loaded bench duty test.

Sampling does not prove every pose clear. Proxies omit horns, cables, panel
thickness and intended mounting interfaces. Motor-box checks are independent of
contact exclusions but still check boxes only. The cosmetic meshes do not
establish hollow-shell manufacturing or structural strength.

**Walking remains unresolved.** The new dynamic pilot alternates real foot lifts,
but its base run falls at 21.304 seconds and its lower-torque cases fail before
qualifying alternations. [Read the results and limits](WALKING_AND_TAIL.md).
The restricted flat-foot lateral probe needs a
center of pressure about 42 mm inward from the support-foot center even at its
best sampled lean, outside the sole's 17 mm half-width. This rejects that static
transfer strategy, not every coordinated/dynamic gait. Wider hip-roll travel
alone caused link/foot collisions. A dynamic weight-transfer controller or a
further lateral-mechanism redesign must precede a walking claim.

## Hardware next steps

The [hardware note](../../docs/hardware/MINI_TYRANNOSAURUS_REX.md) and
[component ledger](components.json) distinguish specifications from allowances,
identify candidate regulators and define power/bench gates. First package real
horns/idlers, linkages, electronics, battery, wiring and service clearances in CAD.
Then measure a loaded leg and head, identify actuator dynamics and add a
hardware-observable task with delay, backlash and voltage variation. No hardware
driver, trained controller or deployment stack is included in this example.
