# Standing check: report (task 3)

18 Nov 2026. **Simulation results** of the T20 (20 kg) and T15 (15 kg) build models in MuJoCo 3.10.0. The part masses
and positions are planning estimates from the parts list (`hardware/triceratops_kickoff/PARTS_PREMIUM.md` section 6.2);
nothing is weighed or measured. Rules were committed before the run (`PREREG_STAND.md`, commit dc2e405); no rule was
changed after the result. The mass decision (D1) is open, so both builds are modelled.

## Answer

**Both builds stand on four feet in simulation, and all four model variants pass every pre-registered rule on both
judged seeds, with no flag.**

| Build, variant | Mass | Front share | Front elbow | Hind knee | Roll | Pitch | Neck |
|---|---|---|---|---|---|---|---|
| T20, closed (20 kg design point) | 20.00 kg | **55.2 %** (parts list 55.2 %) | 7.8 N·m (parts list 8.1) | 6.2 N·m | 0.5 N·m (parts list 4.3; see below) | 1.2 N·m | 0.15 N·m |
| T20, listed parts only | 18.83 kg | 55.5 % | 7.4 N·m | 5.8 N·m | 0.5 N·m | 1.2 N·m | 0.15 N·m |
| T15, closed (15 kg design point) | 15.00 kg | **54.1 %** (parts list 53.6 %) | 5.6 N·m (parts list 6.1) | 4.7 N·m | 0.4 N·m (parts list 3.2) | 0.7 N·m | 0.16 N·m |
| T15, listed parts only | 14.61 kg | 54.2 % | 5.5 N·m | 4.6 N·m | 0.4 N·m | 0.7 N·m | 0.16 N·m |

Torques are joint-side, the largest of the four legs, mean over the last second of 10 s, seed 1 (seeds 0 and 2 within
0.1 N·m; pitch within 0.15 N·m). Full table: [results/stand_2026-11-18/RESULTS.md](results/stand_2026-11-18/RESULTS.md).

- **Weight share.** The 20 kg build puts 55.2 % of its weight on the front feet with no reserve block: the same as the
  parts list's estimate and the 55 % design point. The 15 kg build sits at 54.1 %, below the design point, which
  lowers its front loads (the parts list says the same: 53.6 %). Left and right split 49.9 / 50.1 %.
- **Standing torques are far inside every drive's continuous limit.** Largest use of the thermal-equilibrium torque
  at the joint: T20 front elbow 7.8 of 30.6 N·m (X8 at 1:1, 0.25), hind knee 6.2 of 14.0 N·m (X6 at 1.3:1, 0.44);
  T15 knee 4.7 of 14.0 N·m (0.34). No actuator came near its torque limit.
- **The elbow figure agrees with the parts list** (7.8 against 8.1 N·m at 20 kg; 5.6 against 6.1 N·m at 15 kg). The
  parts list puts the foot exactly under the shoulder and ignores the small fore-aft ground force and the lower leg's
  own weight, so its figure is the conservative one.

## The roll figure is not a like-for-like comparison

The model's roll drives hold only 0.4-0.5 N·m against the parts list's 3.2-4.3 N·m (vertical foot load × the 80 mm
between the roll axis and the leg plane). The reason, from the joint force balance of the 20 kg model (left foreleg):

- The ankle is a pitch hinge only, so each 110 × 80 mm sole is rigid in roll. In the settled pose about 96 % of the
  foot's load (51.7 of 54.1 N) sits on the sole's **inner edge**, 44 mm from the roll axis instead of 80 mm.
- The ground's moment about the roll axis (2.41 N·m) then nearly balances the leg's own weight hanging outboard
  (1.91 N·m), leaving 0.5 N·m for the drive. There is no sideways foot force (|Fy| < 0.01 N).

So the standing roll torque depends on where the centre of pressure sits across the sole: from the inner edge (41 mm
from the roll axis) to the outer edge (121 mm) it is about 0.3 to 4.6 N·m at 20 kg once the leg's own weight is
counted (54.1 N × 41 or 121 mm, less 1.9 N·m); with the pressure at the sole's centre, about 2.4 N·m. The parts
list's 4.3 N·m takes the centre and ignores the leg's weight, so it is the conservative figure. The rigid sole and stiff contact in the model put it at the inner
edge after a 0.14° roll deflection; a real urethane sole will spread it. **Consequences:**

1. The parts list's roll figures stay the sizing case; this check does not lower them.
2. The R5b flag on roll passed for the wrong reason. It is reported as passed (the rule was written as it was) but it
   carries no evidence about roll sizing.
3. For the walk (task 5), edge loading of a sole rigid in roll matters for balance and for the roll drives. The walk
   models should log the centre of pressure across each sole, and a sole roll compliance (or a passive roll ankle) is a
   design question for the foot. It joins the bench questions on the sole (parts list section 8, item 8).

## What the model is

- Built by [build_model.py](build_model.py) from a copy of PR #669's `robot.xml`
  ([base/pr669_robot.xml](base/pr669_robot.xml)) and drive data in [data/drives.json](data/drives.json).
- Masses by section 6.2 category, checked to ±0.01 kg; each part's mass and position is in
  `generated/mass_ledger_*.json`. The two rounding differences come from the parts list itself: tail 0.489 kg against
  0.49, battery 1.012 kg (two 0.506 kg packs) against 1.01.
- **"listed"** is the sum of listed parts. **"closed"** adds the parts list's reserve (1.17 kg T20, 0.39 kg T15) as
  pro-rata growth of shells, harness and frame structure, because the parts list keeps the reserve for "shell, cable
  and bracket mass" growth (assumption A1 in the decision log).
- **Drives, joint side:** torque limit = 0.70 × maker peak × ratio × 0.90 for flexion (T20: roll 42.0, pitch 21.0,
  front elbow 75.6, hind knee 49.1 N·m; T15: roll and pitch 21.0, flexion 49.1 N·m). Reflected inertia = the catalog's
  estimate (T. rex rule A7) × ratio². The pushrod is a constant ratio (no four-bar nonlinearity, no play, no
  stretch). Position-servo gains are PR #669's, scaled with mass for T20 (assumption A2). Speed caps are recorded in
  `drives.json` but not enforced in the model.
- **Head:** neck yaw (±35°) added between the pitch stage and the skull; neck pitch ±20° with a spring set to cancel
  70 % of the head's gravity moment, as the parts list sizes it. The head's centre of mass is 49 mm from the pitch
  pivot (T20), against 110 mm assumed in the parts list's neck sizing, so that sizing is conservative. Beak and eyelid
  servos are mass only (no joints).
- **Tail:** XM430 yaw servo; T20 adds tail pitch (±15°) with a second XM430.
- **Not modelled:** the wireless-stop machine-side receiver's mass (not published; the parts list takes it from the
  reserve); speed limits, friction, play, delay, sensor noise and voltage sag (task 5 adds them); heat over time.

## Open points this raises

- **A3, camera position.** The parts list's section 3 puts the OAK-D Pro W "in the head behind the brow", but its
  mass budget (6.2) counts it in compute and sensors, not in the moving head, and sizes the neck servo without it. The
  model follows the mass budget (camera in the chest, as PR #669 had it). If the camera goes in the head, the head
  gains about 0.12 kg with its cable.
- The positions of the added parts are this model's estimates. The parts list's own placement script was not
  committed, so the agreement on front share (55.2 % against 55.2 %; 54.1 % against 53.6 %) is a cross-check of two
  independent placements, not a reproduction.
