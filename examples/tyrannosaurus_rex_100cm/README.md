# One-metre T-Rex: long-body revision 2

A one-metre dinosaur robot with a forward-reaching skull, horizontal torso, long tapered tail, small eyes, short arms and raised ankles. The design takes its overall stance and continuous surfaces from the user's WowWee RoboRaptor reference. These are original simulation surfaces, with exposed joint covers and removable flank panels for access.

**Concept only:** the MJCF uses assigned component masses and ideal torque-bounded actuators. It passes the documented basic standing and parts screens. It is not fabrication CAD, a walking controller or hardware qualification.

![Compiled MuJoCo geometry](preview.png)

[Four-view design sheet](design_views.png) · [Standing gesture animation](gesture.gif) · [Hardware and budget](hardware.md) · [Parts CSV](bom.csv)

## Shape and dimensions

| Item | Revision 2, at home pose |
|---|---|
| Height | 1.000 m to skull top |
| Nose-to-tail length | 2.104 m |
| Maximum width | 0.450 m |
| Skull visual length | 525 mm |
| Tail reach from yaw joint | 860 mm |
| Hip pitch / ankle pitch height | 560 / 140 mm |
| Thigh / shin pivot distance | 251 / 236 mm |
| Sole footprint, each | 300 × 150 mm |
| Foot centre spacing | 290 mm |
| Assigned total mass | 33.210 kg; unweighed |
| Integration reserve / leg cooling allocation | 2.200 / 1.200 kg, included above |
| Actuation | 12 leg joints + head yaw/nod, jaw, two arm lifts and tail yaw |

The metre build now has its own long-body silhouette; it no longer uses the compact silhouette-C skull of the mini and medium. The preceding 0.901 m-long, 25.144 kg concept remains in git history at `8a3ad174`. The extra mass comes from larger pitch/knee motors, updated vendor motor masses, longer structure and explicit cooling/hardware allocations. A metre tall does not mean a metre of floor space: transport and demonstrations need room for a roughly two-metre animal and its swept tail.

The head is a hollow shell and light frame. Battery and compute stay near the hips. The 650 g tail shell/spar is supported at its root, with an additional 200 g bearing allocation. Its intended construction is removable for transport. It contributes to balance through its actual mass and lever arm; it is not a tuned dynamic balancing controller.

Each leg has hip yaw/roll/pitch, knee pitch and ankle pitch/roll. A rigid sloping metatarsal connects the raised ankle to the flat sole, without another powered joint. The broad soles remain a practical prototype compromise. Thigh fairings, continuous body lofts and restrained eye size reduce the upright humanoid appearance while keeping motors accessible.

[components.json](components.json) is the mass ledger. Decorative meshes do not determine mass or inertia. Box surrogates are aggregated using the parallel-axis theorem. Cosmetic skins can overlap at seams and articulation covers; collision checks use separate proxies. Panel removal, connector access, brackets, bearings and full travel still require mechanical CAD.

## Cameras and electronics

The depth camera is now in the **snout**. The long jaw obstructed the former chest camera's upper view. Moving the same 120 g installed assembly to the snout cleared the sampled rays, and its mass is included in the head-load calculation. **The eyes remain cosmetic.** Head encoders and synchronized timestamps must provide the changing camera-to-body transform.

A rigid pelvis IMU, motor encoders and four load sensors per foot provide the proposed proprioception. MuJoCo's ideal sensor channels omit bias, noise, calibration, time synchronization and electrical interfaces. The snout camera has a near-ground blind region; no cliff sensor or lidar is included in the quoted budget.

Compute remains a Jetson Orin Nano Super developer kit, with an independent Teensy for low-level control and watchdogs. See [hardware.md](hardware.md) for exact candidates, power architecture, sensor limitations and unresolved purchasing details.

## Reproduce the model and checks

```bash
python -m pip install -e ".[test]"
python -m examples.tyrannosaurus_rex_100cm.build_model
python -m examples.tyrannosaurus_rex_100cm.validate
python -m examples.tyrannosaurus_rex_100cm.verify_parts
python -m examples.tyrannosaurus_rex_100cm.render_design --gif examples/tyrannosaurus_rex_100cm/gesture.gif
python -m examples.tyrannosaurus_rex_100cm.demo --view --gesture
pytest environments/tyrannosaurus_rex/tests/test_tyrannosaurus_rex_100cm.py -v
```

Open `model.xml` with `assets/` beside it, or use the demo. Generated assets are committed, so regeneration is optional. `--service` hides the torso shell and reveals provisional electronics boxes. `--support` explicitly activates a display weld; the default model and animation have a free base.

## What passed

[validation.json](validation.json) records five 10-second trials: free stance, free standing gestures, small roll/pitch angular-velocity disturbances, and a separately labelled supported display trial. Unsupported tests reject fixtures, gravity compensation, applied forces and floor support outside the two soles. They check actual floor loads, drift, tilt, contact, joint limits and effort. The free trials retain over 99% of initial root height, with at most 2.02° torso tilt. Knee settled RMS reaches approximately 18.0 Nm against a 30 Nm cap.

[parts_verification.json](parts_verification.json) records motor/BOM counts and arithmetic, home electronics clearance, 70 optical rays, bare-cell packing and a 729-pose expression gravity sweep. Home motor cases have at least 9 mm separation; the four electronics boxes have 11.5 mm minimum separation from each other and motor cases. The home centre-of-mass projection has approximately 147 mm margin within the two-foot support hull. These clearances do not include connectors or prove complete assembly fit.

Both reports contain source hashes. Regression tests include deliberate fixture, case-interference, electronics-overlap, camera-obstruction and budget-error failures. The dedicated CI workflow reproduces the checks and publishes a review bundle.

Standing is not walking qualification. The current crouch already uses substantial knee torque in double support. A successful gait needs measured thermal and torque-speed limits, weight transfer, single support, contact transitions and recovery testing. The 729 expression poses test sampled gravity moments, not full-range collision clearance. No controller or result here establishes that this robot can walk.

## Repository placement

This stays a checkout-only example alongside the mini and medium concepts. It does not modify a registered training plant or policy interface. A future variant-aware catalog can use `large` while keeping the T-Rex species identity. The recommended physical sequence remains mini → medium → instrumented large leg → supported large chassis.
