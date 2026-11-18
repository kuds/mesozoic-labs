# The 60 cm model: build, packaging, standing check and crawl (simulation)

18 Nov 2026. Everything here is simulation of a model whose part masses and positions are planning estimates;
nothing is built, bought or measured. The model is at **revision 3**, after an independent check of revision 1 and a
recheck of revision 2; earlier results stay on record under "Revision history".

## Answer first

- **The 60 cm model (option C) is built**: PR #669's geometry scaled by 0.8 with the T. rex drive types at catalog
  size; **12.69 kg** with the 10 % reserve (11.54 kg without). Parts list with masses, positions and prices:
  `design/PARTS_60CM.md`.
- **The parts fit.** At catalog or estimated real size, the bought electronics (Jetson, Pilz relay block, Teensy
  carriers, power board, rails, audio) fit in the 60 cm body bay without intersecting, with gaps down to 0.5 mm; each
  flexion drive clears its neighbouring drives by at least 17.5 mm and the body by 26 mm over the leg's roll and pitch
  range (`check_layout60.py`; a geometric check).
- **It stands** (pre-registered, both variants, both judged seeds pass). Standing loads: front elbow 4.0 N·m against
  the X6's 10.8 N·m continuous at the joint; hind knee 3.2 N·m against 6.75; roll 0.8 and pitch 0.7 N·m against 7.5;
  neck 0.09 N·m against the XC430's 0.63 (estimate).
- **The crawl: PR #669's foot fails; the roll-ankle foot passes, but only with the default contact model.** The roll
  ankle passes the judged seeds and all four reported seeds, and fails both runs with the stiffer contact setting; with
  a softer ankle spring (revision 2) it also failed one seed. PR #669's sole, rigid in roll, fails on every seed. The
  foot (D5) is still the first design problem.
- **Two numbers to watch.** The model is **0.5 kg over** the budget's mass range (11.8-12.2 kg); see "Mass" below for
  where it comes from. At 12.69 kg the budget's own rule puts the **hind knees' continuous load at 1.00 of the X4's
  rating** (0.93 at 11.8 kg): at the limit, by estimate. The walk simulation (tasks 5 and 7) will measure it.

## The model (revision 3)

`build_model60.py` writes `generated/trike60_C_{listed,closed}.xml` and `_closed_ankleroll`:

- **Geometry:** every body position, geom position, size and end point, site, joint position, camera and mesh of PR
  #669's `robot.xml` × 0.8. The independent check confirmed PR #669's 60 cm figures: crown 600 mm, 1.200 m long, hips
  320 mm high, links 188 / 184 mm, footprint 520 × 280 mm, sole 88 × 64 mm.
- **Not scaled:** drives and servos (catalog sizes: X4 Ø55 × 61 mm, X6 Ø80 × 67.5 mm, XC430 / XL430 28.6 × 34 ×
  46.6 mm), bought electronics (catalog or estimated sizes), and, as in PR #669's own 60 cm design, the flexion drive's
  90 mm offset along the upper link and the crank lane.
- **Drives:** X4-36-L on all roll and pitch joints and the hind knees; X6 on the front elbows; all 1:1 through a
  parallelogram pushrod. Joint-side limits 0.70 × peak (× 0.90 on flexion): 21 N·m roll and pitch, 37.8 N·m front
  elbow, 18.9 N·m hind knee. The T. rex servo model's nominal gains.
- **Expression (D14, full):** neck pitch with a counterbalance spring cancelling 70 % of the head's moment (0.40
  N·m/rad, an estimate chosen so the spring's torque varies by only ±0.14 N·m over ±20°), neck yaw, tail yaw and tail
  pitch, each a position servo limited to 0.70 × stall. The beak servo and the eyelids are mass only; their joints are
  not modelled yet.
- **Feet:** PR #669's passive pitch ankle, springs scaled for similarity from PR #669's own 15 kg model to the
  design's 12.7 kg at 60 cm: 2.03 N·m/rad, 0.091 N·m·s/rad. The `_ankleroll` variant adds a passive roll hinge (±20°)
  with the same spring.
- **Masses:** catalog masses for bought parts; PR #669's parts scaled (structure × 0.8^2.5, head and tail × 0.8^3);
  the body shell at the parts list's 0.85 kg × 0.8^2, as the budget does; limb covers (T20's, × 0.8^2) on top; the
  pushrod linkages at T15's 0.0825 kg per leg, unscaled, since they carry the same drive torque. The closed variant's
  1.15 kg reserve is 0.15 kg of small parts on the chassis plus × 1.209 on every part whose mass is an estimate.

## Mass against the budget

The closed model, 12.69 kg, is 0.49 kg over the top of the budget's range for option C (12.2 kg). Against that figure
(each item with the same 10 % reserve), from the recheck's breakdown of revision 2 (revision 3 moves less than 0.03 kg
between items): limb covers +0.28 kg (the budget, built from T15, has none); head and tail +0.17 kg (the full
expression package's extra servos +0.10 kg, the neck-yaw bearing and frill lighting +0.07 kg); the safety block
+0.13 kg; compute, audio and the second Teensy +0.11 kg; frame and linkages -0.20 kg. The linkages themselves are
within the budget's allowance.

## Packaging check (`check_layout60.py`; a geometric check, not a judged run; output in `results/layout60_2026-11-18c/`)

| What | Result |
|---|---|
| Bought parts in the chassis bay | No two intersect (31 parts; frame-to-frame joints excluded). The battery and the Pilz block rest on the service tray, the Jetson on its NVMe tray; the tightest gaps are 0.5 mm (the NVMe tray and the Pilz block) and 1.0 mm (between the two packs; the IMU and the blowers; the Teensy carriers and the Pilz block) |
| Front X6 to its own X4 pitch / roll drive | 22.5 / 17.5 mm minimum (the 75 cm T15 model, with PR #669's larger Ø60 mm pitch housing: 20.0 mm) |
| Hind X4 knee to its own X4 pitch / roll drive | 35.0 / 30.0 mm |
| Flexion drives to the body | 26.3 mm (fore), 32.8 mm (hind) minimum |

The leg figures are minima over a 7 × 7 grid of each leg's roll and pitch ranges (the flexion joint does not move
the flexion drive); the recheck's finer sweep agreed within 0.1 mm and found the drive cranks at least 16.6 mm from the
roll drive and 15.5 mm from the body. The body is the model's contact ellipsoid, not the shell's real surface.
**For the shell and bracket design:** the front half of the Pilz block, the power board, the rails and the Teensy
carriers sit forward of the service tray's edge and need a bracket (its mass is in the reserve); the tail-yaw servo,
raised onto the pelvic crossmember, stands about 20 mm above the tail fairing's root and behind the last shell hoop.

## Standing check (`PREREG_STAND60.md` with amendments 1 and 2; results `results/stand60_2026-11-18c/`)

| Variant | Mass | Front | Left | Front elbow | Hind knee | Roll max | Pitch max | Neck | Tail pitch | Verdict | Flag |
|---|---|---|---|---|---|---|---|---|---|---|---|
| listed | 11.54 kg | 55.1 % | 50.0 % | 3.62 N·m | 2.92 N·m | 0.73 N·m | 0.67 N·m | 0.079 N·m | 0.175 N·m | **PASS** | none |
| closed | 12.69 kg | 55.3 % | 49.8 % | 4.00 N·m | 3.20 N·m | 0.83 N·m | 0.71 N·m | 0.090 N·m | 0.209 N·m | **PASS** | R1b: mass above the budget's range |

R1 is a regression check (the model against its own ledger). The two judged seeds settle to the same pose and differ
only in the first seconds, so the settled loads are in effect one result. The front elbow's standing torque is within
3 % of the 75 cm parts list's figure scaled by weight × leg length (4.11 N·m); the roll torque is well under its
scaled figure (2.18 N·m), as at 75 cm.

## Crawl (`PREREG_CRAWL60.md` with amendments 1 and 2; results `results/crawl60_2026-11-18c/`)

The 75 cm crawl's frozen parameters scaled (lengths, including the planner's search grid, × 0.8; times × √0.8), no
development runs, rules C1-C6 with their millimetre thresholds unchanged. Judged seeds 5 and 6 (new for revision 3);
seeds 1-4 reported.

| Model | Seeds | C4 (one foot at a time) | C5 (planted feet hold) | Other rules | Max pin motion | Verdict |
|---|---|---|---|---|---|---|
| PR #669's foot | 5, 6 (judged) | FAIL | FAIL | pass | 19.8 mm | **FAIL** |
| PR #669's foot | 1-4 (reported) | fail | fail | pass | 19.6 mm | |
| roll ankle | 5, 6 (judged) | pass | pass | pass | 7.4 mm | **PASS** |
| roll ankle | 1-4 (reported) | pass | pass | pass | 7.4 mm | |
| roll ankle, impratio 10 | 5, 6 (reported) | **fail on both** | fail on seed 5 | pass | 10.8 mm | |

**Reading it:** with the spring scaled from PR #669's own model, the roll ankle holds its feet and shares the load on
all six friction draws. But its result depends on the contact model (both stiffer-contact runs fail C4) and on the
spring (with the softer revision 2 spring, 1.49 N·m/rad, one seed of four failed C4). So the roll ankle is the right
direction for the foot, not a finished design. PR #669's roll-rigid sole fails at 60 cm as at 75 cm (F7).

**Drive loads in the crawl (reported, not judged; judged runs, both feet).** Peak torques: roll 12.3 N·m and pitch
20.6 N·m (limit 21 N·m), front elbow 11.8 N·m (limit 37.8), hind knee 10.5 N·m (limit 18.9). The pitch peak is one
short spike on a front pitch joint in the roll-ankle run on seed 6: 98 % of the model's limit, 69 % of the drive's
30 N·m peak; no sample saturated. RMS: roll 3.0 N·m, pitch 2.5, front elbow 4.5, hind knee 3.8, under the continuous
ratings. Peak speeds: flexion 46 rpm, pitch 32, roll 7. A 0.02 m/s crawl does not test the walk's continuous load, so
this does not clear the hind knee's estimate above; and the pitch spike is worth finding in the walk work.

## What this means

1. **The foot comes first (D5).** A sole rigid in roll makes the robot stand on two edges at both sizes. A passive roll
   hinge fixes that in the default contact model but not under stiffer contact. Next: a roll hinge with a centring
   spring tuned as a design parameter, a compliant sole, or a smaller, rounder contact, judged under both contact
   settings.
2. **The walk needs feedback.** The crawl replays a plan; task 5's model-based controller (D10) should close the loop
   on the body's tilt and the feet's loads.
3. **Mass and the hind knees.** Update the budget for the full expression package, the limb covers and the safety
   block (about +0.5 kg), then let the walk simulation measure the knees' continuous load. If it is over, the options
   are a 1.2:1 crank ratio on the hind knees (a non-parallelogram linkage), X6 hind knees (+$318 of hardware, +0.92 kg)
   or a lighter build (the reserve is 1.15 kg).

## Revision history

**Revision 1** (commit 6739087d; results `results/stand60_2026-11-18/`, `results/crawl60_2026-11-18/`): 11.31 /
12.44 kg; standing PASS; crawl FAIL for both feet (seeds 1 and 2).

**Revision 2** (commit 4fb39562; results `…-18b/`): after the independent check of revision 1 (21 findings, no
blockers; every result reproduced byte for byte). Standing PASS with the mass flag; crawl: PR #669's foot FAIL, the
roll ankle PASS on judged seeds 3 and 4 but failing C4 on reported seed 2 and on every impratio-10 run.

**Revision 3** (commit ce1dd36c; results `…-18c/`): after the recheck of revision 2 (no blockers; every rerun
reproduced byte for byte; the 75 cm results unchanged).

| Check | Finding | Done |
|---|---|---|
| 1 | Front X6 at 72 mm (scaled) instead of PR #669's 90 mm; 4.5 mm from the X4 pitch drive | rev. 2: 90 mm, 22.5 mm clear |
| 1 | Model mass above the budget's range, unstated | rev. 2: flagged (R1b); attribution corrected in rev. 3's report |
| 1 | Body shell on PR #669's 0.57 kg, unlike the 75 cm model and the budget | rev. 2: 0.85 kg × 0.8^2 |
| 1 | Reserve concentrated (× 1.52) on frame, shells, harness; ledger silent | rev. 2: spread over estimated parts; rev. 3 includes the servo cases (× 1.209); factor in the ledger |
| 1 | Standing R1 compared the model with itself | rev. 2: labelled a regression check; R1b flag added |
| 1 | Crawl planner's grid not scaled | rev. 2: scaled; 75 cm results unchanged byte for byte |
| 1 | Crank lane in the drive; leftover belt parts; buried caps | rev. 2: fixed (the cranks now sit up to 3 mm off their shaft faces; visual only) |
| 1 | Bought electronics scaled × 0.8 and intersecting | rev. 2: real sizes, laid out; rev. 3: the Pilz block, which the check had skipped, cleared |
| 1 | Default-class skip that never fired; disclosure's sole height; docstrings; audio, bus adapter, second Teensy missing; servo joint parameters; neck-spring basis; temp files; timing note | rev. 2: fixed or noted |
| 1 | Head and tail shells × s³ | kept, as the budget; × s² would add 0.15 kg and about 15 % to the neck's 0.09 N·m |
| 1 | Linkage mass without a basis | rev. 2: T15's 0.0825 kg per leg |
| 1 | Ankle springs kept their 75 cm values | rev. 2: scaled, from the wrong reference; rev. 3: from PR #669's own 15 kg model |
| 2 | Packaging check skipped the Pilz block ("contactor" matched "_contact"); it intersected the NVMe tray and the reserve block | rev. 3: proxies matched by suffix; parts moved; rechecked |
| 2 | Report's reserve factor, mass attribution, 75 cm clearance comparison, drive-limit wording, spring wording | rev. 3: corrected here |
| 2 | Tail-yaw servo outside the shells; electronics forward of the tray unsupported; check docstring overstated the sweep | rev. 3: noted above; docstring corrected |

**Recheck of revision 3:** requested; its result will be added here.
