# Compsognathus: anatomical and Rev B robot prototypes

Two **model-only MuJoCo prototypes**, added September 7, 2026. Both have a
floating base, articulated legs, working ground contact, joint-angle servos,
an IMU, encoders, foot touch sensors and one forward-facing camera. Both hold
a standing pose in forward dynamics. **Neither has a trained walking policy
or a registered Mesozoic Labs training environment yet.**

![MuJoCo renders of the authored home poses](data/model_preview.png)

| Property | Anatomical proxy | Rev B robot |
|---|---|---|
| MJCF | `assets/compsognathus.xml` | `assets/compsognathus_robot.xml` |
| Home standing height | 38.1 cm | **35.3 cm** |
| Nose-to-tail envelope | 86.2 cm | **46.5 cm** |
| Maximum visual width | 13.8 cm | 25.8 cm, including leg hardware |
| Pelvis-frame height | 24.4 cm | **21.8 cm** |
| Model mass, excluding target | 1.000 kg, assumed | **1.5856 kg**, preserved Rev B allocation |
| Actuators | 14 | **12, all in the legs** |
| Tail | Two driven base axes and three passive distal hinges | **Fixed, 50 g allocation; no joint or motor** |
| Head | Neck pitch and jaw pitch | Fixed, 45 g allocation |
| Feet | Three toes and a small plantar pad | **100 × 90 × 8 mm** soles |

Dimensions are measured visual bounds at `home`, not manufacturing
dimensions. The anatomical proxy has a longer tail and slimmer legs; the
robot preserves the previously screened mechanism.

## Run and reproduce

From the repository root with Python 3.11+:

```bash
python -m pip install -e ".[test]"
python -m environments.compsognathus.scripts.view_model --model robot
python -m environments.compsognathus.scripts.view_model --model biological
python -m environments.compsognathus.scripts.build_models
python -m environments.compsognathus.scripts.validate_models --output /tmp/compso-preflight.json
pytest environments/compsognathus/tests environments/shared/tests/test_mjcf_assets.py
```

The repository pins **MuJoCo 3.10.0**, used for these checks. The original
Rev B package used 3.12.0. On macOS use `mjpython` for the interactive viewer.
The viewer starts at `home`, holds fixed joint-angle targets and steps real
physics. `--motors-off` disables actuator gains; `--mass-scale 1.15` increases
body masses and inertias while preserving motor limits.

Pictures need Pillow; videos additionally need `imageio imageio-ffmpeg`:

```bash
python -m environments.compsognathus.scripts.view_model --compare /tmp/models.png
python -m environments.compsognathus.scripts.view_model --model robot --snapshot /tmp/robot.png
python -m environments.compsognathus.scripts.view_model --model robot --video /tmp/standing.mp4 --seconds 5
```

Headless rendering needs a working OSMesa runtime with `MUJOCO_GL=osmesa`,
or EGL with `MUJOCO_GL=egl`. Model loading and physics tests need neither.

## Preserved robot design

`references/compso_rev_b.xml` preserves the original September 6 Rev B model.
The generator verifies its SHA-256 and uses the original first
double-support pose stored in `data/model_parameters.json`. Body transforms,
joint signs/ranges, centres of mass and inertias remain unchanged.

Each leg has hip yaw, hip roll, hip pitch, knee pitch, ankle pitch and ankle
roll. **Both thigh and shin parallel shaft separations are 78.65 mm**. The
ankle adapter adds 40 mm from pitch to roll and 38 mm from roll to sole.
Home foot-centre spacing is 130 mm. Shaft separation is measured
perpendicular to the axis, avoiding the misleading along-shaft offsets in
the source joint origins.

The candidates are **two STS3250-C001 hip-roll servos and ten STS3215-C018
12 V servos**. This supersedes the older ten-leg-plus-two-neck-axis estimate.

| Mass allocation | Mass |
|---|---:|
| Core: battery, computer, yaw servos, power distribution, wiring and structure | 505 g |
| Fixed head and sensor allowance | 45 g |
| Fixed tail and attachment | 50 g |
| Two complete leg assemblies | 985.6 g |
| **Total** | **1,585.6 g** |
| **15% growth scenario** | **1,823.44 g** |

The head and tail are already included. These are allocated/inherited CAD
masses, not weighed hardware. The core remains an aggregate envelope, not
a verified arrangement of the battery, boards and regulator. Decorative
mounts and the camera lens use existing inertial allowances.

## Actuation and contact assumptions

Robot position servos use `kp=25 N·m/rad`, `kv=0.4 N·m·s/rad`, `gear=1`
and absolute-radian command limits matching the joints. These gains are
**simulation assumptions**, not identified Feetech parameters. The derivative
term is inside the torque clamp. Passive joint damping is only 0.005
N·m·s/rad; armature is an assumed 0.0001 kg·m².

The caps are manufacturer **rated** torque multiplied by an assumed 11.5/12
voltage factor and 0.9 derating: approximately **±1.3533 N·m at hip roll**
and **±0.8458 N·m elsewhere**. They are not stall-torque allowances.

These are constant torque limits, not an identified torque-speed,
electrical or thermal model. No-load speeds are retained as reference
parameters but are not enforced by the MJCF. Backlash, delay, voltage sag
and heating remain open. The model assumes a regulated 11.5 V motor rail:
the STS3250 specification stops at 12 V, so a fully charged 12.6 V 3S pack
is not qualified for direct connection. Power hardware must fit the core
mass allocation or trigger a revision.

Robot mechanical geoms collide with the floor. Opposite legs collide with
one another and with the head/tail envelopes. **Within-leg mating parts are
excluded by masks**, and the aggregate core only collides with the floor.
These exclusions prevent artificial forces at intended mechanical
interfaces but preclude a full self-collision clearance claim. Convex mesh
contacts do not establish cable, horn, connector, fastener or bracket fit.

The anatomical proxy uses primitive geometry with normal parent/child
filtering. All three toes and the plantar pad belong to their touch
sensor's rigid foot body, so distal contact is captured. The pad and soft
tissue are simulation approximations, not a fossil-fitted reconstruction.

## Validation scope

`data/preflight_v0.json` records model hashes, dimensions, masses, contact
loads, torque utilisation and the complete trial settings:

- Both stand for **ten seconds** with fixed angle targets, finite state,
  torque limits respected and support exclusively through their feet.
- Each stands after **three seeded ±1° joint-position perturbations**,
  five seconds per trial. This is a narrow smoke test, not certification.
- The robot stands at **1.15× mass/inertia** with unchanged motor limits.
- Foot sensors account for ground support, and the standing centres of mass
  lie inside the contact support polygons.
- Both lose the standing pose when motors are disabled: no hidden weld or
  large passive joint spring supplies the posture.
- Regression tests compare the robot against the preserved Rev B transforms
  and inertias, and pin its dimensions, mass and twelve-actuator layout.

There is no gait, policy training, deliberate push test, uneven terrain,
battery-runtime result or fabrication-ready assembly. The earlier Rev B
341-pose clearance/load screen remains historical prescribed-motion
evidence; it is not a forward-dynamics walking result for this model.

## Repository integration and next steps

The anatomical model follows the T-Rex/raptor `+X forward, +Y left, +Z up`
convention, `pelvis` root, limb names, mocap prey, sensors and `home` keyframe.
The robot retains upstream `left_`/`right_` joint names and mirrored signs
for traceability. Controls are **absolute angles in actuator order**, not
the existing environments' normalized residual actions.

Both assets are in the MJCF inventory tests and a new CI test-suite entry.
They are intentionally absent from the species catalog, training registry,
plant certificates and behavior recipes. Existing species interfaces stay
unchanged. The next milestones are a loaded-servo identification test,
separate biological/robot observation and action contracts, a Gymnasium
wrapper with plant/reset preflight, then weight-transfer and slow-walking
training. The robot policy must use onboard-observable quantities, with
the tail kept fixed.

## Sources

- [T-Rex](../trex/assets/trex.xml) and [raptor](../velociraptor/assets/raptor.xml): repository conventions.
- [Open Duck Mini v2, pinned commit](https://github.com/apirrone/Open_Duck_Mini/tree/b23317a485b3cec7d8417f352478778b3475173c): inherited robot material via Rev B; see [attribution](references/NOTICE.md).
- [Scott Hartman's Compsognathus reconstruction](https://www.skeletaldrawing.com/theropods/compsognathus-longipes/): approximate scale/silhouette reference, including an 85 cm reconstruction. No illustration was copied. The proxy's 95/114/64 mm leg lengths, mass, joint limits and soft tissues are engineering choices, not measurements from that drawing.
- [Bidar, Demay and Thomel (1972), Smithsonian-hosted translation](https://naturalhistory.si.edu/sites/default/files/media/translated_publications/Bidar%26amp%3B%25201972.pdf): anatomical context for longer tibiae, slender metatarsals and three functional foot digits; not used for current taxonomic or palaeoecological claims.
- [Feetech STS3215-C018](https://www.feetechrc.com/525603.html) and [STS3250](https://www.feetechrc.com/562636.html): ratings carried forward from the September 6 Rev B research.
- [MuJoCo 3.10 position actuators](https://mujoco.readthedocs.io/en/3.10.0/XMLreference.html#actuator-position): servo and force-limit semantics.
