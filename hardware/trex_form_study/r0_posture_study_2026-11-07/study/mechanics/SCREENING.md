# Mechanical screen

Distances are in the model world frame, measured in millimetres unless stated. +X is forward, +Y is left and +Z is up. The origin in the horizontal plane is the neutral root. The model’s assigned inertials include each installed component and modeled cover. The floor is level. Each table uses the model identified by its filename and SHA-256 in `metrics.json`.

## Standing support

The hull is the convex hull of both rectangular sole undersides. The margin is the signed horizontal distance from the system center of mass to its nearest edge. A positive double-support margin establishes geometric support for this pose. It cannot establish centering over one foot.

| Model | Mass kg | COM x, y, z | Two-sole margin | Corner height spread |
|---|---|---|---|---|
| PR662 | 9.896 | 4.9, -0.9, 318.8 | 58.9 | 0.00 |
| A00 | 9.881 | 13.4, -0.9, 316.7 | 67.4 | 0.00 |
| A10 | 9.877 | 12.9, -0.9, 319.6 | 66.9 | 0.00 |
| A20 | 9.875 | 11.8, -0.9, 322.4 | 65.8 | 0.00 |
| B00 | 10.076 | 14.1, -0.9, 319.0 | 68.1 | 0.00 |
| B10 | 10.075 | 13.6, -0.9, 322.2 | 67.6 | 0.00 |
| B20 | 10.070 | 12.4, -0.9, 325.3 | 66.4 | 0.00 |
| C00 | 9.877 | 14.3, -0.9, 316.6 | 68.3 | 0.00 |
| C10 | 9.874 | 13.7, -0.9, 319.5 | 67.7 | 0.00 |
| C20 | 9.870 | 12.6, -0.9, 322.3 | 66.6 | 0.00 |

## Static joint loads

Gravity and the support reactions balance all six root forces and moments. For two-foot support each vertical reaction acts at COM.x on a sole centerline, with the load split solving lateral moment balance. Single-foot screening counter-rolls hips and ankles in 0.005 rad increments from 0 through 0.45 rad subject to the model limits, with the swing hip/knee/ankle adding −0.25/+0.50/−0.25 rad. The standing sole is translated to the floor. The row selects the largest signed center-of-pressure margin on each side, using the published platform flat-foot search convention. Marks are prescribed gravity loads in N·m, without friction, acceleration or impact. The hip-roll column retains both sides. The single-support hip pitch, knee and ankle trio is for the side with the larger hip-roll load. Ankle is the larger of roll/pitch. Raw files include each signed joint torque and margin for each foot. A negative margin describes an infeasible gravity-only equilibrium.

| Model | 2-foot hip | 2-foot knee | 2-foot ankle | 1-foot hip roll L/R | 1-foot hip pitch/knee/ankle | Worst 1-foot margin | Minimum swing gap |
|---|---|---|---|---|---|---|---|
| PR662 | 1.48 | 3.35 | 1.19 | 9.73 / 9.56 | 2.58 / 6.85 / 2.99 | 9.2 | 56.6 |
| A00 | 1.07 | 2.92 | 0.77 | 9.73 / 9.56 | 1.81 / 6.06 / 3.00 | 9.1 | 56.6 |
| A10 | 1.09 | 2.95 | 0.80 | 9.73 / 9.55 | 1.86 / 6.11 / 3.00 | 9.1 | 56.6 |
| A20 | 1.15 | 3.00 | 0.85 | 9.72 / 9.55 | 1.95 / 6.21 / 3.00 | 9.1 | 56.6 |
| B00 | 1.05 | 2.96 | 0.76 | 9.88 / 9.71 | 1.78 / 6.14 / 3.00 | 9.6 | 56.6 |
| B10 | 1.08 | 2.99 | 0.78 | 9.88 / 9.71 | 1.83 / 6.19 / 3.00 | 9.6 | 56.6 |
| B20 | 1.14 | 3.04 | 0.84 | 9.88 / 9.71 | 1.93 / 6.29 / 3.00 | 9.6 | 56.6 |
| C00 | 1.02 | 2.88 | 0.73 | 9.73 / 9.55 | 1.73 / 5.98 / 3.00 | 9.1 | 56.6 |
| C10 | 1.05 | 2.90 | 0.76 | 9.72 / 9.55 | 1.78 / 6.03 / 3.00 | 9.1 | 56.6 |
| C20 | 1.10 | 2.96 | 0.81 | 9.72 / 9.55 | 1.88 / 6.13 / 3.00 | 9.0 | 56.6 |

An additional stance search permits five independent pelvis roll angles (−10° through +10°), thirteen support counter-roll angles and three levels of swing-leg adduction. It ranks joint-limit-compliant gravity poses with sole margin ≥3 mm, flat support and swing gap ≥8 mm by hip-roll effort. The minima below precede interference screening. None reaches the 6 N·m sustained assumption.

| Model | Best 1-foot hip roll L/R N·m | Pelvis roll L/R | Sole margin L/R mm |
|---|---|---|---|
| PR662 | 8.54 / 8.37 | -10° / +10° | 4.8 / 6.5 |
| A10 | 8.52 / 8.35 | -10° / +10° | 4.9 / 6.6 |
| B10 | 8.64 / 8.47 | -10° / +10° | 5.5 / 7.1 |
| C10 | 8.52 / 8.34 | -10° / +10° | 4.9 / 6.6 |

## Actuated head and tail

The tail yaw shaft is vertical in neutral, while tail weight creates a bending moment on its support. The inertia is the sum of the moving bodies’ central tensors rotated to the pivot frame plus their parallel-axis contributions. Values exclude motor rotor inertia, friction and acceleration commands.

| Model | Neck gravity N·m | Tail yaw gravity N·m | Tail off-axis N·m | Neck / tail inertia kg·m² |
|---|---|---|---|---|
| PR662 | 0.236 | 0.000 | 0.217 | 0.0030 / 0.0035 |
| A00 | 0.189 | 0.000 | 0.136 | 0.0028 / 0.0032 |
| A10 | 0.189 | 0.000 | 0.136 | 0.0028 / 0.0032 |
| A20 | 0.189 | 0.000 | 0.136 | 0.0028 / 0.0032 |
| B00 | 0.269 | 0.000 | 0.177 | 0.0046 / 0.0046 |
| B10 | 0.269 | 0.000 | 0.177 | 0.0046 / 0.0046 |
| B20 | 0.269 | 0.000 | 0.177 | 0.0046 / 0.0046 |
| C00 | 0.217 | 0.000 | 0.175 | 0.0035 / 0.0047 |
| C10 | 0.217 | 0.000 | 0.175 | 0.0035 / 0.0047 |
| C20 | 0.217 | 0.000 | 0.175 | 0.0035 / 0.0047 |

## Free-base motion

Each run uses the installed drive at 22.2 V, retained delay/deadband, torque-speed limit and rolling effort derating. Two seconds of neutral hold and a separate 4.8-second trial with two prescribed lateral transfer cycles are run from home on the model floor. The latter simultaneously commands hip/ankle counter-roll up to 0.24 rad, swing knee up to 0.50 rad, hip and ankle pitch compensation, neck pitch ±0.15 rad and tail yaw ±0.20 rad. A valid loaded single-foot interval needs ≥80% body weight on the stance sole, ≤5% on the other sole, swing gap ≥8 mm and no other floor support over 1 N. Both tests require tilt <12° and sole-only support. Stand requires drift <25 mm. Transfer requires ≥0.10 s loaded on each side. Trajectories are open-loop probes. `motion.json` includes per-axis peak, RMS, time above continuous assumptions and the lowest tail surface sampled at 50 Hz.

| Model | Stand (tilt) | Transfer | Transfer max tilt | Longest L/R s | Peak hip roll N·m | Other floor support |
|---|---|---|---|---|---|---|
| PR662 | pass (3.0°) | fail | 13.8° | 0.00 / 0.00 | 9.47 | none |
| A00 | pass (2.2°) | fail | 13.6° | 0.00 / 0.00 | 10.39 | none |
| A10 | pass (2.3°) | fail | 13.7° | 0.00 / 0.00 | 9.89 | none |
| A20 | pass (2.4°) | fail | 13.7° | 0.00 / 0.00 | 9.87 | none |
| B00 | fail (4.8°) | fail | 13.0° | 0.00 / 0.00 | 11.96 | none |
| B10 | pass (2.3°) | fail | 13.5° | 0.00 / 0.00 | 9.77 | none |
| B20 | pass (2.4°) | fail | 13.7° | 0.00 / 0.00 | 9.75 | none |
| C00 | pass (2.2°) | fail | 13.6° | 0.00 / 0.00 | 10.43 | none |
| C10 | pass (2.2°) | fail | 13.6° | 0.00 / 0.00 | 10.02 | none |
| C20 | pass (2.3°) | fail | 13.7° | 0.00 / 0.00 | 9.87 | none |

## Visibility and tail

Neutral camera ray tests use the 17×11 grid of a 640×400 image at the installed vertical field of view. The ideal floor distance is measured horizontally ahead of the camera at the bottom of its center column on a level floor. Physical self-occlusion uses MuJoCo rays through visible geometry. Each ToF center ray is tested against the model. The raw clearance results repeat this test for nine joint-limit samples of head pitch and tail yaw. Sensor calibration, depth validity, side/rear coverage, detection and tolerance require physical tests.

| Model | Blocked image rays | Center ray | Ideal floor distance m | ToF hits L/R | Lowest tail through articulation mm |
|---|---|---|---|---|---|
| PR662 | 0/187 | clear | 1.84 | floor / floor | — |
| A00 | 8/187 | clear | 1.78 | floor / floor | 403 |
| A10 | 1/187 | clear | 1.78 | floor / floor | 403 |
| A20 | 0/187 | clear | 1.78 | floor / floor | 403 |
| B00 | 10/187 | clear | 1.78 | floor / floor | 390 |
| B10 | 3/187 | clear | 1.78 | floor / floor | 390 |
| B20 | 0/187 | clear | 1.78 | floor / floor | 390 |
| C00 | 8/187 | clear | 1.78 | floor / floor | 402 |
| C10 | 3/187 | clear | 1.78 | floor / floor | 402 |
| C20 | 0/187 | clear | 1.78 | floor / floor | 402 |

## Surface interference

Counts below are poses with confirmed penetration of a modeled hardware primitive or a transverse intersection between two moving panels. The families are the home pose, 16 coupled transfer samples, nine head/tail limit samples and 18 seeded full joint-box samples. The sampled motion paths cover a subset of the full continuous envelope. Direct signed checks use opposing motor cases, bearings, forks, shafts, rails, feet and spars on different welded bodies. Same-body packaging is audited separately. The signed-distance sample is an upper bound on the true minimum. Each triangle is subdivided with maximum edge 5 mm. A positive sampled clearance below 8 mm cannot certify a 3 mm manufacturing gap. Shell-to-shell samples report unsigned separation. For pairs flagged below 3 mm, an explicit edge-versus-triangle test detects transverse crossings. Coplanar contact and unsampled close approaches remain unresolved. The JSON retains each pose, pair and measured signed or unsigned witness under 3 mm. It also includes native pairwise distance between opposing leg cases, links and feet.

| Model | Neutral | Transfer | Head/tail range | Joint-box extremes |
|---|---|---|---|---|
| A00 | 0/1 | 0/16 | 0/9 | 0/18 |
| A10 | 0/1 | 0/16 | 0/9 | 0/18 |
| A20 | 0/1 | 0/16 | 0/9 | 0/18 |
| B00 | 0/1 | 0/16 | 0/9 | 13/18 |
| B10 | 0/1 | 0/16 | 0/9 | 6/18 |
| B20 | 0/1 | 0/16 | 0/9 | 0/18 |
| C00 | 0/1 | 0/16 | 0/9 | 0/18 |
| C10 | 0/1 | 0/16 | 0/9 | 0/18 |
| C20 | 0/1 | 0/16 | 0/9 | 0/18 |
