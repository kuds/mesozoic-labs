# Definitions used in the round-one study

All quantities are measured on the compiled MuJoCo model at its `home` keyframe (nominal standing
pose), unless stated otherwise. Frame: +X forward, +Y the robot's left, +Z up, floor at z = 0.

## Heights and posture

| Term | Definition |
|---|---|
| Crown height | Floor to the highest point of the skull covering (vertex-exact) in the nominal standing pose. Height classes 60/75/100/120 cm match the crown at the **10° reference posture**. The 0° and 20° versions of the same body report the crown height they reach (controlled comparison, so the body is not rescaled). |
| Hip height `H` | Floor to the hip-pitch axis (acetabulum analogue). |
| Body unit `B` | Scale of every anatomical landmark. `H = leg_scale × B` (leg_scale = 1 except the `compact` proportion variant). |
| Torso inclination | Angle above horizontal of the line from the hip-pitch-axis midpoint (acetabulum analogue) to the shoulder landmark (glenoid analogue). Chest-up is positive. Postures 0°, 10° and 20° are exploratory values, not validated anatomical targets. |
| Pelvis inclination | Angle of the pelvis/sacral reference axis. The pelvis follows half the chest rotation (0°, 5°, 10°). The remainder is a fixed structural lumbar bend, not a joint. |
| Neck and head orientation | The neck joint's nominal angle keeps the skull's long axis at −4° (slightly nose-down) in the world in every posture. A chest-up posture therefore raises the neck base and head and does not tip the skull back. |
| Tail orientation | The tail root carries a built-in pitch so the tail keeps its world orientation (+2°, rising slightly distally) in every posture. |

Postures are made by solving joint angles with the feet flat on the floor. The whole robot, feet
included, is never rotated.

## Widths (kept separate)

| Term | Definition |
|---|---|
| Torso width | Full width of the trunk coverings (arms and exposed rib hoops excluded). |
| Hip-pitch spacing | Distance between the two hip-pitch axes (2 × `y_leg`). |
| Roll-axis offset | Lateral distance from the body centreline to each hip-roll axis. It is 0 for the tandem layout, where both roll motors lie on the centreline (one ahead of, one behind the hip axis). |
| Hardware width | Full width across all hardware primitives (motor envelopes, brackets, structure). |
| Thigh width | Full width across both thigh fairings. |
| Foot spacing | Distance between the two sole-site centres in the nominal stance. |
| Support width / area | Width and area of the convex hull of all floor-contact pads of both feet. |
| Max width | Full width of everything visible. |

## Ratios reported

Crown / torso width, crown / hardware width, hip height / foot spacing, torso length / depth,
skull length / torso length, skull width / torso width, neck length / skull length, tail length /
overall length, overall length / crown, crown / hip height, and the head+neck, tail and leg mass
fractions. The CoM offset ahead of the hip is reported as a fraction of `H`.

## Volume classes in the ledger

| Class | Meaning |
|---|---|
| A | Volume and mass required by the hardware itself: catalog bodies plus installation allowance (motors, battery, compute, boards, sensors). |
| B | Volume and mass caused by the chosen arrangement: structure, brackets, linkages, bearings, cassette, harness and routing, cover supports and fasteners, integration reserve. |
| C | Cosmetic covering volume (hollow shells), which can be redesigned freely. |

Each ledger row also carries a **basis**:
* `sourced`: read from a cited source (see `designlab/data/research/*.json`).
* `derived`: computed from sourced values, with the rule stated.
* `estimate`: an engineering rule, stated, to be replaced by CAD or weighing.
* `measured`: measured on hardware. No row in this study is measured.

## Loads and limits

| Term | Definition |
|---|---|
| Single-support hold | Quasi-static pose with one foot lifted (5 % of H), the CoM over the stance-foot pad centroid, flat stance foot, trunk roll searched over −6°…+6°. The feasible pose with the least stance hip-roll torque is used. "Infeasible" means no balanced, reachable pose exists within joint limits. That is a finding, not a load case. |
| Reference walk | Prescribed periodic walk used only to size loads, speeds and power, not a controller. Nominal: Fr = v²/(gH) = 0.012, stride 0.55 H, duty 0.60. Brisk: Fr = 0.025, stride 0.62 H, duty 0.58. CoM path: linear-inverted-pendulum solution driven by a ZMP reference at the stance-foot centre, corrected iteratively against the multibody ZMP. |
| Sustained planning torque | 0.57 × vendor rated torque (PR #662's 6 N·m on a 10.5 N·m-rated RMD-X4). MyActuator's own V4.1 thermal tables reach equilibrium at 50–80 % of rated, so this fraction is conservative. For box servos it is 0.167 × 12 V stall (PR #662 rule). |
| Peak planning torque | 0.70 × vendor peak (servos: 0.293 × stall). |
| Requirement | Continuous ≥ max(nominal-walk RMS, single-support hold). Peak ≥ 1.25 × brisk-walk peak. Usable speed ≥ 1.15 × brisk-walk peak speed. |
