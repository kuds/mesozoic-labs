# Mechanical evidence for the intermediate T. rex

The nominal 9.896 kg plant briefly supports itself on either foot on the hard-floor surrogate. It still exceeds the assumed continuous actuator rating in single support, tilts too far on carpet, and contains interference between moving assemblies. It is a locomotion development model. The 3 mm clearance release gate is open.

[Machine-readable results](mechanical_report.json) · [Real-time single-support trial, side and front views](nominal_single_support.mp4) · [Independent controller evaluations](../platform/gait/) · [Installed hardware and battery-service checks](../platform/engineering/engineering_report.json)

## Why change the leg travel?

In the initial nominal design, full counter-roll with a level torso left the projected center of mass 15.7 mm outside the supporting sole. The nominal revision allows ±0.36 rad of hip roll and ±0.40 rad of ankle roll. At the same flat-sole pose, the projected center of mass now lies 9.2 mm inside the sole. Compact and wide retain the previous ±0.25 rad limit. Their results combine differences in proportion and travel. A second calculation restricts only hip and ankle roll to ±0.25 rad on the final nominal model: its center of mass lies 15.7 mm outside. That ablation holds all installed masses and dimensions fixed.

| Candidate | Signed best sole margin | Required stance hip-roll torque | Longest L / R support, hard | Longest L / R support, carpet |
| --- | ---: | ---: | ---: | ---: |
| Compact | −18.6 mm | 8.60 N·m | 0 / 0 s | 0 / 0 s |
| Initial nominal | −15.7 mm | 9.41 N·m | — | — |
| Nominal r01 | +9.2 mm | 9.73 N·m | 1.942 / 1.964 s | 1.016 / 1.164 s |
| Wide | −23.2 mm | 11.09 N·m | 0.078 / 0.094 s | 0 / 0 s |

[Initial nominal results](nominal_initial_screen.json) · [Archived initial model, ledger, design, generator and validator](nominal_initial_sources.zip)

The static calculation flexes the opposite knee 0.5 rad and holds the root level. It scans feasible mirrored counter-roll angles at 0.005 rad intervals and selects the greatest signed support margin. A projected point inside the flat sole can carry the vertical force at that center of pressure. The joint gravity compensation is the generalized gravity load minus the vertical ground wrench projected through the stance foot Jacobian. Outside the sole, the computed wrench would tip the robot. These loads omit acceleration and give no passive friction credit. The nominal 9.73 N·m exceeds the modeled 6 N·m continuous allocation by 62%, though it fits the 12 N·m peak allocation. Long stance holds therefore require a duty-cycle and thermal review.

## Free-root motion trials

The 2 ms simulation advances through the shared installed-drive model at each physics tick, including command delay, play, voltage-dependent speed and effort derating. There is no support fixture or root actuation. Stand and crouch pass on both surfaces for all three plants. The alternating transfer and yaw probes use the same command for each variant. Neither probe delivers bilateral 100 ms loaded steps on either surface. The separate learned-policy evaluation determines gait performance.

The one-foot trial shifts first, then lifts. After 0.35 s settling, the controller ramps the hips to 0.36 rad and ankles 0.40 rad in counter-roll over 0.65 s. After 0.8 s from the start of the shift, it bends the unloaded knee 0.55 rad over 0.30 s, with compensating pitch commands. Mirroring targets tests each support foot. Every trial respects the candidate's joint limits. These are scripted bench tests. The hold timing and amplitudes were chosen by an exploratory sweep on the nominal geometry and are development evidence.

A support interval requires ≥80% body weight on the stance sole, ≤5% on the unloaded sole, ≥8 mm swing-bottom height and no other ground contact carrying more than 1 N. The hold gate requires ≥1 s, completion and torso tilt below 12°. In nominal hard-floor trials, the swing sole reaches 51.8 mm left-stance and 51.7 mm right-stance. The torso reaches 5.6° and 5.5°. The left knee exceeds 6 N·m for 2.17 s of the left-side trial and reaches 6.49 N·m RMS. The modeled effort budget allows this short excursion. The reported run does not establish that repeated holds are thermally sustainable.

On carpet, peak tilt reaches 18.3° and 18.2° during the corresponding holds. At 20 V with an additional 150 g distributed with the same head radius of gyration, the left hold also fails (18.3°). Adding head mass and reducing bus voltage together is a coupled stress test. The other trials run at 22.2 V. The camera, battery and drive masses stay installed throughout these trials.

The uniform hard-floor contact uses 0.80 sliding friction, 8 ms normal solver time constant and 0.8 mm impedance transition width. The carpet surrogate uses 0.95, 24 ms and 4 mm with greater damping and torsional/rolling resistance. A high-priority floor material assigns the effective properties to contacts. In the nominal standing trials the deepest sole contacts were 0.16 mm and 2.67 mm respectively. This represents level, dry, bonded low-pile material on a rigid subfloor. Carpet fibers, anisotropy, edge snags, loose rugs and transitions require separate tests. No coefficients have yet been fitted to specimens. [MuJoCo contact parameter rules](https://mujoco.readthedocs.io/en/stable/modeling.html#contact-parameters)

## Clearance and evidence limits

Each plant evaluates 5,315 geometry pairs. The sample set includes 512 seeded independent joint combinations, two smooth command paths discretized into piecewise-linear joint intervals, and three dynamically recorded paths sampled every 20 ms. The bounding-sphere screen, finite-distance query and shared separation-plane fallback include motor and connector reservations, soles, structural rods, cases, sensors, expression shafts and rigid covers. Near-zero unresolved distances are reported. Named witnesses retain the sample and family for replay. Group 4 component solids already represented by installation reserves, wire outlines and motor-face illustrations fall outside this screen. Fixed-body structural members may join; the validator checks their fixed-body reserve and case clearances, plus geometry on distinct bodies. Cables are reservation volumes. Bearing seals, fasteners, tolerances and deformed structures require CAD and physical measurements.

The path certificate subtracts a conservative relative motion bound from neighboring sampled distance lower bounds over each **linear joint interpolation**. It bounds each non-shared hinge's contribution using maximum descendant radius times the hinge travel and the distance to the nearest endpoint. For the nominal turn command, 5,222 of 5,315 pairs certify 3 mm throughout that piecewise-linear path. The other 93 lack this certificate. Recorded dynamic path checks are point samples. This certificate does not cover interpolants between those actual recorded states. Independent random joint samples do not exhaust the whole configuration space.

The broad screen intentionally retains housing-to-mount, shaft-to-bearing and sensor-inside-reserve contacts. Those counts include modeled mating features. A detail drawing must identify the attachment and its moving faces before accepting a local exclusion. Examples requiring an actual geometry change or demonstrated relief include:

- At independent joint sample 456, the two soles penetrate by 39.7 mm in the nominal model. At sample 88, the left sole intersects the opposite ankle-roll motor by 38.8 mm. Independent coordinate limits cannot protect the full command box.
- During the recorded nominal hard-floor hold, the neck rail and cranium intersect by 10.5 mm and the tail-root and shroud by about 8.0 mm. The covers use convex mechanical proxies. Split the moving and fixed cover surfaces at both pivots and retest the motion.
- The rear battery hanger overlaps the disconnect reservation by 5.0 mm. The camera braces overlap the compute reservation by 3.5 mm. Repackage or detail these brackets and check plugged cable access. The separate 240 mm leftward battery extraction test uses the pack plus a 6 mm handling guard and addresses a different clearance.

All actuator force peaks and RMS values use every simulated physics tick through termination; RMS includes startup. Ground reactions use the solver wrench at every tick. Maximum swing clearance uses the lowest sole-box corner, and the reported loaded sole-center speed includes rotation/rocking of the foot. Solver contact states and forces correspond to the start of each physics tick, with velocity measured after integration. Impacts faster than the simulation timestep and real joint temperatures remain unmeasured.

## Reproduce

From the repository root, with the MuJoCo dependencies installed:

```sh
python -m examples.tyrannosaurus_rex_60cm.platform.build --variant all
python -m examples.tyrannosaurus_rex_60cm.platform.validate_platform --verify-sources
python -m examples.tyrannosaurus_rex_60cm.platform.validate_platform --video
python -m pytest environments/shared/tests/test_robot_floor.py environments/shared/tests/test_robot_clearance.py
```

The report records model, component ledger, generator, parameter file, applied floor, drive, validation and clearance code SHA-256 values together with NumPy and MuJoCo versions. The video uses the deterministic measured joint and free-root trajectory at 25 fps, matched side/front views and 1× playback. Video rendering additionally requires Pillow and imageio-ffmpeg. Use `--strict` to return a failing process status for open mechanical issues while preserving the report. The default report-only command finishes even when the design gate fails.
