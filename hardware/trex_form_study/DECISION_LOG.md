# Decision log

Entries are tagged by kind so that preferences, constraints, findings and assumptions never blur:

* **P**: the project owner's stated preference or requirement.
* **E**: an engineering constraint (physics, catalog fact, interface).
* **F**: an experimental finding from this study's models, with where it was measured.
* **A**: a provisional assumption, to be replaced by measurement, a quote or a decision.

Status: `open` (awaiting review), `adopted` (used in round one, revisable), `superseded`.

## Requirements carried in (P)

| ID | Statement | Status |
|---|---|---|
| P1 | Long-term robot 3–4 ft (0.9–1.2 m) tall: walks naturally, navigates, follows voice commands, moves head, tail, jaw and arms. Running is a stretch goal. | adopted |
| P2 | First demonstrator: walk and turn on hard floor and carpet; smooth, coordinated motion, not slow rigid stepping; modest powered head and tail; fixed arms; jaw deferred. | adopted |
| P3 | Onboard Jetson autonomy (obstacle avoidance, person following) with a camera plus ranging. The external page only selects a person and starts/stops. Markerless tracking. | adopted |
| P4 | Removable battery for 15–20 min, external-power operation, $10–20 k prototype including outsourced fabrication, integration and spares. Fabrication outsourced. | adopted |
| P5 | No portability weight ceiling, but weight still matters. Partial coverings and exposed mechanics are acceptable. | adopted |
| P6 | Earlier forms rejected: a rectangular chassis with a head and tail; a deep barrel torso with a high rounded back, hanging belly, small-looking head and no chest/abdomen/hip distinction. More covered silhouette area is not itself success. | adopted |
| P7 | Explore 60, 75, 100 and 120 cm crown heights; narrow, balanced and broad widths; distinct body families; 0°, 10° and 20° chest-up postures. Do not freeze the form factor or start extensive training before two review rounds. | adopted |

## Engineering constraints and findings

| ID | Kind | Statement | Evidence | Status |
|---|---|---|---|---|
| E1 | E | Quasi-static single-support hip-roll torque ≈ g·[m_trunk·y_r + m_swing·(y_r + y_leg)]. It is set by the hip-roll axis offset y_r and the swing-leg plane y_leg, not by foot spacing. Narrowing the feet (R3: 214 → 175 mm soles, hips kept at 214 mm) cannot remove it. | Derivation in `designlab/statics.py` docstring. Reproduced 9.7–9.9 N·m on the #662 model (BASELINE §4). | adopted |
| E2 | E | Putting both hip-roll motors on the body centreline (tandem: one ahead of, one behind the hip axis) makes y_r = 0. The single-support hip-roll load then comes only from the swing leg. | Round-one models | adopted as default layout |
| F1 | F | 60 cm tandem candidate (60-M-S-10): single-support hip roll 2.9 N·m against 9.7 N·m on #662 geometry, with a lighter actuator (CubeMars AK45-36). The 9 N·m vs 6 N·m gap is **resolved geometrically** in the model. It is not yet resolved in hardware. | `round1/stage_a` | open |
| E3 | E | A T. rex mass distribution (long neck, deep chest, light robotic tail) puts the CoM 0.1–0.2 H ahead of the hip. Hip **pitch**, not hip roll, then becomes the governing joint in single support. | Round-one models | adopted |
| F2 | F | Moving the battery from the belly to a fixed tail base behind the hip (the animal's caudofemoral mass region) cut 60 cm single-support hip pitch from 7.5 to 4.3 N·m. That kept every drive on the 24 V X4 class instead of forcing 48 V parts and a mass spiral. | Sizing runs, `designlab/pipeline.py` history | adopted as default (`belly` kept as a variant) |
| E4 | E | Hip yaw is implemented as thigh twist about the femur (roll → pitch → twist). The yaw motor then sits inside the thigh, and the upper thigh can be a drumstick fairing that pivots in place around the hip-pitch motor. Cost: the inclined twist axis carries part of the stance moment (≈ 3 N·m at 60 cm). | Round-one models | adopted |
| E5 | E | Shank = fused tibia + metatarsus with a fixed raised hock. Ankle pitch/roll sit at the MTP joint, driven by push rods from two motors high on the shank (calf), which keeps distal mass low. | Round-one models | adopted |
| E6 | E | Legs must use CAN drives. All Unitree motors and MyActuator -R parts are RS-485 only, and several MyActuator models are discontinued. These are excluded from selection. | `designlab/data/actuators.json` "excluded" | adopted |
| E7 | E | On a 12S (48 V) bus, the incumbent Pololu D36V50F12 (50 V max) and D24V10F5 (36 V max) cannot be used. The D42V110F12 or Mean Well DDR-120C can. | parts research | adopted |
| E8 | E | Incumbent RMD-X4-P36-36-E (V4) vs -E-L (V4.1): peak 34 vs 30 N·m, Kt 1.9 vs 1.8 N·m/A. The 0.4 Ω in PR #662 is line-to-line, so per phase it is ≈ 0.2 Ω and #662's copper-loss estimate was ≈ 2× high. Round one uses the -E-L with per-phase R. | actuator research | adopted |

## Provisional assumptions (A)

| ID | Assumption | Replace with |
|---|---|---|
| A1 | Sustained planning torque = 0.57 × vendor rated (PR #662 convention). MyActuator V4.1 thermal tables reach equilibrium at 50–80 % of rated. CubeMars and RobStride publish no thermal definition. | Bench thermal tests per joint duty |
| A2 | Peak planning torque = 0.70 × vendor peak; requirement margins 1.25 (peak) and 1.15 (speed) | Dyno curves |
| A3 | Structure masses from twin-plate 6061 beam rules (2 × body weight, σ ≤ 70 MPa, tip deflection ≤ 0.5 %) | CAD and weighing |
| A4 | Cover walls 1.6–1.8 mm (60 cm) up to 2.6–2.9 mm (120 cm) MJF PA12, plus 20–36 % internal supports and 2.5–5.5 g per mount | Supplier DFM review |
| A5 | Integration reserve 6 % of installed mass (PR #662 precedent) | Weighing |
| A6 | Power: gear efficiency 0.75, drive 0.92, idle 1.4 W/axis, auxiliaries 35 W + sensors through 0.88 DC-DC, 75 % usable energy, no regeneration credit (PR #662 constants) | Bench current and thermal logs |
| A7 | Reflected rotor inertia (armature) = 2.5e-6 kg·m² × (D/55 mm)⁴ × gear² | Vendor data or measurement |
| A8 | Cost lots scaled from PR #662's 60 cm lines (frame by structure mass^0.6, covers by cover volume^0.85, integration by size^0.7, contingency 5.3/7.2/8.4 %) | Quotes from drawings |
| A9 | Part selection score = mass + 0.8 kg per $1000 of price | Owner preference on cost vs mass |

## Open preferences (asked in round one)

*Filled after the round-one review: size class, body family, posture, width and tail/skull proportion.*
