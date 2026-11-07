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
| E9 | E | Allowable supply windows (sourced): MyActuator X-series 20–55 V (X12 20–70 V); RobStride 03–06 15–60 V, 00/02 24–60 V, 01 24–48 V; CubeMars AK60/70/80/10 18–52 V; CubeMars AK45 15–28 V only. An X4 can share a 12S bus with larger drives; an AK45 cannot. Below its nominal voltage a drive's usable speed falls in proportion. | `designlab/data/actuators.json` (`voltage_min_v`, `voltage_max_v`) | adopted |
| F3 | F | The first full sweep put every 75 cm candidate at 21–23 kg on 48 V with no feasible single-support hold. This was a **sizing artifact**: (1) parts were filtered by exact nominal voltage; (2) the score ignored that the roll and twist motor diameters set the hip spacing; (3) the spiral guard compared against the first iteration, so it depended on the starting parts. With voltage windows, a stated packaging term and start-independent convergence rules, 75-M-S-10 converges at about 16 kg on 24 V with feasible holds. 60 cm results did not change. | Superseded sweep (not kept); `designlab/sizing.py`, `pipeline.py` docstrings; commit message of the fix | adopted |
| F4 | F | Sensor layout v1 failed screening. The throat camera lost 17–24 % of its view to the jaw, and at 60 cm 0 % of a person's torso straight ahead was visible. The camera box intersected the neck motor. The two ToF sensors aimed identically, so 82–90 % of their floor coverage overlapped. The lidar inside the hull blocked 93–100 % of its own scan within 1 m. In v2: the stereo camera sits in the skull roof ahead of the orbits, 5° up (3–4 % self-occlusion; torso visible at every size at home); the ToF pair points 45° down and 12° outward over each foot path; the lidar sits under the base of the throat on a bracket (rear 106° blocked by the chest). Cost of v2: the snout hides the floor straight ahead (nearest visible floor 1.9 m ahead of the toes at 60 cm, 3.0–3.3 m at 100–120 cm), so near-field floor sensing relies on the ToF pair. | `designlab/sensors.py` screen of v1 (sensor agent report) and v2 (`round1/review/screening`) | adopted (v2) |
| F5 | F | A visual check of the generated families found repeats of rejected traits: the Substantial trunk read as a flat-bottomed loaf, the Tapered trunk had an uncovered mid-trunk gap, and the bone-coloured jaw read as a beak. The belly line now rises behind the ribs to a pubic-boot keel, the Tapered back is continuous, and the jaw is skin-toned. The head still reads small against a long trunk in side view; that question goes to the owner (`-bighead` variant shown). | Renders in `round1/review/plates` | open |
| F6 | F | The clearance screen (`designlab/clearance.py`) found five packaging defects in the generator. (1) In the tandem layout the left and right hip brackets overlapped by 6.2–6.5 mm at the centreline in every pose. (2) At 60 and 75 cm the tail motor sat inside the tail-base battery cassette, so no removal path was clear; at 60 cm the cassette also reached into the tail. (3) At 60 cm the power board overlapped the compute tray. (4) Metatarsus plates limited ankle pitch to ±0.14 rad at 60 cm. (5) Pelvis and thigh shells overlapped, by up to 21 mm at 120 cm. Fixes: each bracket now sits on its own roll motor's side; the tail motor sits on the tail pivot and the cassette hangs low under the tail base, with the tail pivot moved back where needed (48 mm at 60 cm); the power board is held behind the compute tray; the plates stop short of the ankle block; the pelvis covers are narrowed inside the thigh fairings. Remaining at 60 cm: the brisk walk still brings the upper-thigh fairing into the upper ankle motor, knee flexion is usable to about 0.57 rad, and the hip roll range is limited by toe contact (inward) and pelvis–thigh shells (outward). | `round1/review/screening/*_clearance.json` | adopted (fixes); open (60 cm remainder) |

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
| A10 | Packaging term in the selection score: 0.33 kg per metre of hip half-spacing per kg of robot. That is first-order roll and twist load growth sized at about 13 N·m/kg, leaving out the loop gain. | Sizing with CAD hip layouts |
| A11 | Bus swing over the usable charge: 6S 21.0–25.2 V, 12S 42.0–50.4 V. X-series drives on 12S need the vendor's external bus capacitor and a regeneration clamp below 55 V. | Electrical design review |
| A12 | The head-mounted camera's pose comes from the neck-pitch encoder. The USB or flex lead runs through the neck with a service loop. | Bend-life test of the lead; calibration procedure |

## Open preferences (asked in round one)

*Filled after the round-one review: size class, body family, posture, width and tail/skull proportion.*
