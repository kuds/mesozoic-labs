# 60 cm T. rex development

The 60 cm robot is the first **candidate physical platform**; the mini remains a
useful simulation reference. Height means floor to the eventual head crown in
the nominal standing pose. A headless chassis does not have to be 60 cm tall.

Start with the [M1 skeleton study](m01_skeleton/README.md). It has a horizontal
spine, forward-folding knees, six independent joints per leg, full-size motor
and electronics envelopes, and fixed future head/tail/arm mass. Its thin green
outline is a proportion guide, not shell CAD. The tail has no actuator.

| Hardware milestone | Purpose | Evidence needed before acceptance |
| --- | --- | --- |
| [M1 skeleton](m01_skeleton/README.md) | Qualify the leg/torso mechanism, mass placement and control | Finished load-bearing CAD, measured actuator response and mass, clearance, standing, crouching, single support and initial steps |
| [M2 integrated](m02_integrated/README.md) | Add the actual light head, fixed tail, body covers and sensors | Repeat M1 with real payloads; qualify walking, turning, sensing, thermal duty and service access |
| [M3 production candidate](m03_production_candidate/README.md) | Make the robot repeatable to assemble and maintain | Selected BOM, toleranced drawings, harness/assembly instructions, endurance and repeat-build evidence |

No milestone is accepted, no hardware is built, and no release tag is created.
Each accepted revision should be frozen by a Git commit/tag with its model,
parts, calibration and evaluation evidence. The current design is `concept_r01`.

## Training and the 100 cm goal

The [MJCF and builders](../../examples/tyrannosaurus_rex_60cm/) are a standalone
engineering example. **The shared SB3 notebook still selects the mini for
`tyrannosaurus_rex / robot`; it does not train this 60 cm model.** This change
does not replace that registered environment or alter its checkpoints.

Once the mechanism is selected, add an explicit size/variant identity to the
training registry. Preserve the 60 cm model's exact source revision, its 12-axis
action mapping, sensor/observation definition and normalization state. The
mini has ten actions, so its policy is not directly interchangeable. Reuse the
curriculum and evaluation tools, then train and validate new policies against
the new dynamics. Introduce measured actuator delay, torque/speed/thermal
limits, contact properties and realistic payload variation before sim-to-real.

Hardware milestones and behavior stages are independent: stance → recovery →
locomotion → character/target behaviors may repeat within each hardware revision.
M1 currently demonstrates short ideal-PD standing and crouching only. It does
not demonstrate learned balance, single-support walking or getting up from a fall.

For the eventual 100 cm+ robot, carry forward bus conventions, calibration,
logging, emergency-stop architecture, policy interfaces and test procedures.
Recalculate the mechanism, actuators, structure, inertia and fall loads at that
size. A 60 cm motor selection or policy is not automatically suitable for 100 cm.

The physical sequence should be a loaded actuator bench, one instrumented leg,
a supported two-leg torso, then untethered stance and controlled steps. This
produces relevant hardware learning without requiring a separate miniature build.
