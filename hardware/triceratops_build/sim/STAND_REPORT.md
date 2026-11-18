# Standing check: report (task 3)

18 Nov 2026. **Simulation results** of the T20 (20 kg) and T15 (15 kg) build models in MuJoCo 3.10.0. The part masses
and positions are planning estimates from the parts list (`hardware/triceratops_kickoff/PARTS_PREMIUM.md` section 6.2);
nothing is weighed or measured. Rules were committed before the first run (`PREREG_STAND.md`, commit dc2e405) and
were not changed. The mass decision (D1) is open, so both builds are modelled.

**Two runs.** The first run (`results/stand_2026-11-18/`) used the models as first built. An independent check then
asked for three model corrections, and the first crawl runs showed the PR's servo gains too soft; the models were
rebuilt (list below) and the same rules re-run (`results/stand_2026-11-18b/`). This report gives the second run; both
passed every judged rule.

## Answer

**Both builds stand on four feet in simulation, and all four model variants pass every pre-registered rule on both
judged seeds.** One flag (R4b) is raised for the T15 closed variant; it compares a closed model with a listed-parts
reference (below).

| Build, variant | Mass | Front share | Front elbow | Hind knee | Roll | Pitch | Neck |
|---|---|---|---|---|---|---|---|
| T20, listed parts | 18.83 kg | **55.6 %** (parts list, listed: 55.2 %) | 7.4 N·m | 5.8 N·m | 1.1 N·m | 1.2 N·m | 0.15 N·m |
| T20, closed (20 kg design point) | 20.00 kg | 55.7 % | 7.9 N·m (parts list 8.1) | 6.2 N·m | 1.1 N·m (parts list 4.3; see below) | 1.3 N·m | 0.15 N·m |
| T15, listed parts | 14.61 kg | **54.2 %** (parts list, listed: 53.6 %) | 5.5 N·m | 4.6 N·m | 0.9 N·m | 0.8 N·m | 0.16 N·m |
| T15, closed (15 kg design point) | 15.00 kg | 54.6 % (flag R4b) | 5.7 N·m (parts list 6.1) | 4.7 N·m | 1.0 N·m (parts list 3.2) | 0.8 N·m | 0.16 N·m |

Torques are joint-side, the largest of the four legs, mean over the last second of 10 s. With the stiffer gains the
three seeds settle to the same values within 0.01 N·m. Full table:
[results/stand_2026-11-18b/RESULTS.md](results/stand_2026-11-18b/RESULTS.md).

- **Weight share.** Listed parts against the parts list's listed-parts estimate: T20 55.6 % against 55.2 % (+0.4
  points), T15 54.2 % against 53.6 % (+0.6). The 55 % design point holds for T20; T15 sits just below it, which
  lowers its front loads (the parts list says the same). Left and right split 49.9-50.0 %.
- **Flag R4b (T15 closed, 54.6 % against 53.6 %, 1.0 point).** The rule compares every variant with the parts list's
  listed-parts figure. In the closed T15 model the whole 0.39 kg reserve sits on the chassis, partly forward (the
  stop receiver at x = +170 mm), which moves the share. Listed against listed the difference is +0.6. Not a fail.
- **Standing torques are far inside every drive's continuous limit.** Largest use of the thermal-equilibrium torque at
  the joint: T20 front elbow 7.9 of 30.6 N·m (X8 at 1:1, 0.26), hind knee 6.2 of 14.0 N·m (X6 at 1.3:1, 0.44); T15
  front elbow 5.7 of 14.0 N·m (X6 at 1.3:1, 0.41). No actuator came near its torque limit.
- **The elbow figure agrees with the parts list** (7.9 against 8.1 N·m at 20 kg; 5.7 against 6.1 N·m at 15 kg). The
  independent check's moment balance for the first run: the ground contact gives 8.14 N·m about the elbow, the lower
  leg's own weight takes off 0.42 N·m. The parts list ignores the leg's weight and puts the foot exactly under the
  shoulder, so its figure is the conservative one.
- **Neck.** The spring carries 71 % of the head's gravity moment (0.52 N·m), leaving 0.15 N·m for the servo.

## The roll figure is not a like-for-like comparison

The roll drives hold only 0.9-1.1 N·m against the parts list's 3.2-4.3 N·m (vertical foot load × the 80 mm between
the roll axis and the leg plane). From the joint force balance of the first run's 20 kg model, left foreleg, which the
independent check reproduced:

| Moment about the roll axis | N·m |
|---|---|
| Ground contact | +2.41 |
| Leg's own weight (2.54 kg at 76.7 mm outboard) | -1.91 |
| Roll actuator | -0.51 |

- The ankle is a pitch hinge only, so each 110 × 80 mm sole is rigid in roll. In the settled pose the sole's inner-edge
  contacts carry about 95 % of the foot's load (51.6 of 54.2 N), at 41 mm from the roll axis; the whole foot's centre
  of pressure sits 44-45 mm from it, not at the 80 mm sole centre. Sideways foot force is negligible (≤ 0.04 N).
- So the standing roll torque depends on where the centre of pressure sits across the sole: from the inner edge (41 mm)
  to the outer edge (121 mm) about 0.3 to 4.6 N·m at 20 kg once the leg's own weight is counted; at the sole's centre
  about 2.4 N·m. With the stiffer gains of the second run the roll drives hold about 1.1 N·m (the centre of pressure was not
  re-measured for that run). The
  parts list's 4.3 N·m takes the sole centre and ignores the leg's weight, so it stays the conservative sizing case.
- **For the walk (task 5):** edge loading of a sole rigid in roll matters for balance and for the roll drives. Log
  the centre of pressure across each sole; a sole roll compliance (or a passive roll ankle) is a design question for
  the foot, beside the bench questions on the sole (parts list section 8, item 8).

## What the model is (second run)

- Built by [build_model.py](build_model.py) from a copy of PR #669's `robot.xml` ([base/pr669_robot.xml](base/pr669_robot.xml))
  and drive data in [data/drives.json](data/drives.json). The independent check regenerated all eight files
  byte-identical and found only the reserve block removed from the base model, nothing double-counted or missing.
- Masses by section 6.2 category, checked to ±0.01 kg; each part's mass and position is in
  `generated/mass_ledger_*.json`. Two rounding differences come from the parts list itself: tail 0.489 kg against 0.49,
  battery 1.012 kg (two 0.506 kg packs) against 1.01.
- **"listed"** is the sum of listed parts. **"closed"** adds the parts list's reserve (T20 1.17 kg, T15 0.39 kg). Its
  first 0.45 kg goes on the chassis as the two items the parts list sets against it: the wireless-stop machine-side
  receiver (mass unpublished, taken as 0.30 kg) and 0.15 kg of small added parts. T15's reserve does not cover both, so
  all 0.39 kg goes there. T20's remaining 0.72 kg is pro-rata growth of shells, harness and frame structure; 0.30 kg of
  it lands on the legs (assumption A1, revised).
- **Drives, joint side:** torque limit = 0.70 × maker peak × ratio × 0.90 for flexion (T20: roll 42.0, pitch 21.0,
  front elbow 75.6, hind knee 49.1 N·m; T15: roll and pitch 21.0, flexion 49.1 N·m). Reflected inertia = the catalog's
  estimate (T. rex rule A7) × ratio².
- **Gains (assumption A2, revised):** the T. rex servo model's nominal for the same drives, kp 598.5 N·m/rad at the
  drive output (kv 24.48 roll, 11.51 pitch and flexion), × ratio² at the joint; also an assumption there, not a maker
  figure. The first run used PR #669's softer 150-330 N·m/rad (× 4/3 for T20).
- **Collisions:** each drive housing has a massless collision twin (added after the independent check, which found
  that drives and pushrods could pass through the torso or another leg). No self-contact occurs at neutral or after
  3 s of standing in any variant.
- **Head:** neck yaw (±35°) between the pitch stage and the skull; neck pitch ±20° with a spring cancelling 70 % of the
  head's gravity moment at neutral. Beak and eyelid servos are mass only. **Tail:** XM430 yaw; T20 adds tail pitch
  (±15°) with a second XM430.
- **Not modelled:** speed limits (recorded in `drives.json`, not enforced), friction, play, delay, sensor noise,
  voltage sag (task 5 adds them); heat over time.

## Limits of this check and open points

- **The 1.3:1 pushrod is modelled as a constant ratio, which it is not.** The independent check solved the four-bar
  with the model's own crank geometry (input crank 27.7 mm, output crank 36 mm, 145 mm between the crank axis and
  the elbow): the ratio is 1.30 at neutral, within 10 % only for flexion between -29° and +24°, and 1.71 and 2.13 at the
  two ends of the -47° to +36° range; the input crank sweeps 119°, against the parts list's "about 108°". Near the
  ends the joint's torque limit, speed and reflected inertia change a lot. The 1:1 front linkage of T20 is an exact
  parallelogram, constant. Until the model carries an angle-dependent ratio, every crawl and walk run must report
  whether the 1.3:1 joints stayed inside -29° to +24°; the crank travel joins the CAD checks.
- **Rule R7 could not fail.** The neck's torque limit (0.70 × 4.44 = 3.11 N·m) equals R7's threshold, and the tail
  pitch limit (1.20 N·m) is the same as its threshold; an overload would have shown as saturation under R6 instead.
- **R4b compares closed variants with a listed-parts reference** (the flag above); it should have named the listed
  variants only.
- **Head centre of mass (F6):** the gravity lever about the neck pitch axis is 34.9 mm horizontal (T15 40.7 mm), giving
  0.52 N·m at T20, against the 1.63 N·m the parts list sized the neck servo for (head 1.51 kg at 110 mm). The sizing is
  conservative.
- **Camera position (A3):** the parts list's section 3 puts the OAK-D Pro W "in the head behind the brow", but its mass
  budget counts it in compute and sensors and sizes the neck servo without it. The model follows the mass budget.
- The positions of the added parts are this model's estimates (A4). The parts list's placement script was not
  committed, so the front-share agreement is a cross-check of two independent placements, not a reproduction.
