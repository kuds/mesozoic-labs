# The 60 cm model: build, packaging, standing check and crawl (simulation)

18 Nov 2026. Everything here is simulation of a model whose part masses and positions are planning estimates;
nothing is built, bought or measured. The model is at **revision 2**, rebuilt after the independent check of revision
1; the first results are kept below under "Revision history".

## Answer first

- **The 60 cm model (option C) is built**: PR #669's geometry scaled by 0.8 with the T. rex drive types at catalog
  size; **12.69 kg** with the 10 % reserve (11.54 kg without). Parts list with masses, positions and prices:
  `design/PARTS_60CM.md`.
- **The parts fit.** At catalog size, the bought electronics (Jetson, Pilz relay block, Teensy carriers, power board,
  rails, audio) fit in the 60 cm body bay without touching, and each flexion drive clears its neighbours by at least
  17.5 mm and the body by 26 mm over the leg's joint range (`check_layout60.py`; a geometric check).
- **It stands** (pre-registered, both variants, both judged seeds pass). Standing loads: front elbow 4.0 N·m against
  the X6's 10.8 N·m continuous at the joint; hind knee 3.2 N·m against 6.75; roll 0.8 and pitch 0.7 N·m against 7.5;
  neck 0.09 N·m against the XC430's 0.63 (estimate).
- **The crawl: PR #669's foot fails; the roll-ankle foot passes its judged seeds, but not robustly.** It fails one of
  the two reported seeds and every run with the stiffer contact setting. The foot (D5) is still the first design
  problem, at 60 cm as at 75 cm.
- **Two numbers to watch.** The model is **0.5 kg over** the budget's mass range (11.8-12.2 kg), mostly because the
  budget counted the core expression package, not the full one chosen (D14), and because the model adds the safety
  block, linkages and limb covers explicitly. At 12.69 kg, the budget's own rule puts the **hind knees' continuous load
  at 1.00 of the X4's rating** (0.93 at 11.8 kg): at the limit, by estimate. The walk simulation (tasks 5 and 7) will
  measure it.

## The model (revision 2)

`build_model60.py` writes `generated/trike60_C_{listed,closed}.xml` and `_closed_ankleroll`:

- **Geometry:** every body position, geom position, size and end point, site, joint position, camera and mesh of PR
  #669's `robot.xml` × 0.8. The independent check confirmed PR #669's 60 cm figures: crown 600 mm, 1.200 m long, hips
  320 mm high, links 188 / 184 mm, footprint 520 × 280 mm, sole 88 × 64 mm.
- **Not scaled:** drives and servos (catalog sizes: X4 Ø55 × 61 mm, X6 Ø80 × 67.5 mm, XC430 / XL430 28.6 × 34 ×
  46.6 mm), bought electronics (catalog or estimated sizes), and, as in PR #669's own 60 cm design, the flexion drive's
  90 mm offset along the upper link and the crank lane.
- **Drives:** X4-36-L on all roll and pitch joints and the hind knees; X6 on the front elbows; all 1:1 through a
  parallelogram pushrod. Joint-side limits 0.70 × peak (× 0.90 on flexion); the T. rex servo model's nominal gains.
- **Expression (D14, full):** neck pitch with a counterbalance spring cancelling 70 % of the head's moment (0.40
  N·m/rad, an estimate chosen so the spring's torque varies by only 0.14 N·m over ±20°), neck yaw, tail yaw and tail
  pitch, each a position servo limited to 0.70 × stall. The beak servo and the eyelids are mass only; their joints are
  not modelled yet.
- **Feet:** PR #669's passive pitch ankle; springs scaled for similarity (1.49 N·m/rad, the 75 cm 3 N·m/rad × weight ×
  length). The `_ankleroll` variant adds a passive roll hinge (±20°) with the same spring.
- **Masses:** catalog masses for bought parts; PR #669's parts scaled (structure × 0.8^2.5, head and tail × 0.8^3);
  the body shell at the parts list's 0.85 kg × 0.8^2, as the budget does; limb covers (T20's, × 0.8^2) on top; the
  pushrod linkages at T15's 0.0825 kg per leg, unscaled, since they carry the same drive torque. The closed variant's
  1.15 kg reserve is 0.15 kg of small parts on the chassis plus × 1.15 on every part whose mass is an estimate.

## Packaging check (`check_layout60.py`; a geometric check, not a judged run; output in `results/layout60_2026-11-18b/`)

| What | Result |
|---|---|
| Bought parts in the chassis bay, at catalog size | No two intersect (30 parts; frame-to-frame joints excluded). The battery rests on the tray and the Jetson on its NVMe tray; the tightest gaps are 1.0 mm (between the two packs; the IMU and the blowers), 1.1 mm (the tail-yaw servo above the pelvic crossmember), 2.5 mm (the Jetson and the audio board) |
| Front X6 to its own X4 pitch / roll drive | 22.5 / 17.5 mm minimum over the joint grid (75 cm: 22.5 mm) |
| Hind X4 knee to its own X4 pitch / roll drive | 35.0 / 30.0 mm |
| Flexion drives to the body | 26.3 mm (fore), 32.8 mm (hind) minimum over the joint grid |

The grid is 7 × 7 × 7 poses of each leg's roll, pitch and flexion over their full ranges. The body is the model's
contact ellipsoid, not the shell's real surface; the shell design must keep these clearances.

## Standing check (`PREREG_STAND60.md` with amendment 1; results `results/stand60_2026-11-18b/`)

| Variant | Mass | Front | Left | Front elbow | Hind knee | Roll max | Pitch max | Neck | Tail pitch | Verdict | Flag |
|---|---|---|---|---|---|---|---|---|---|---|---|
| listed | 11.54 kg | 55.1 % | 50.0 % | 3.62 N·m | 2.93 N·m | 0.73 N·m | 0.67 N·m | 0.079 N·m | 0.175 N·m | **PASS** | none |
| closed | 12.69 kg | 55.3 % | 49.8 % | 4.00 N·m | 3.20 N·m | 0.83 N·m | 0.72 N·m | 0.090 N·m | 0.210 N·m | **PASS** | R1b: mass above the budget's range |

R1 is a regression check (the model against its own ledger); it was the only mass rule in the first run and could not
fail, which the independent check pointed out. The two judged seeds settle to the same pose and differ only in the
first seconds, so the settled loads are in effect one result. The front elbow's standing torque is within 3 % of the
75 cm parts list's figure scaled by weight × leg length (4.11 N·m); the roll torque is well under its scaled figure
(2.18 N·m), as at 75 cm.

## Crawl (`PREREG_CRAWL60.md` with amendment 1; results `results/crawl60_2026-11-18b/`)

The 75 cm crawl's frozen parameters scaled (lengths, including the planner's search grid, × 0.8; times × √0.8), no
development runs, rules C1-C6 with their millimetre thresholds unchanged. Judged seeds 3 and 4 (new, since the models
changed after a result); seeds 1 and 2 reported.

| Model | Seeds | C4 (one foot at a time) | C5 (planted feet hold) | Other rules | Max pin motion | Verdict |
|---|---|---|---|---|---|---|
| PR #669's foot | 3, 4 (judged) | FAIL | FAIL | pass | 19.2 mm | **FAIL** |
| PR #669's foot | 1, 2 (reported) | fail | fail | pass | 19.8 mm | |
| roll ankle | 3, 4 (judged) | pass | pass | pass | 6.2 mm | **PASS** |
| roll ankle | 1, 2 (reported) | seed 1 pass, **seed 2 fail** | pass | pass | 6.8 mm | |
| roll ankle, impratio 10 | 1-4 (reported) | **fail on all four** | fail on seed 2 | pass | 10.3 mm | |

**Reading it:** the roll ankle fixes the sliding (C5) and usually the load sharing (C4), but its C4 result flips with
the friction draw and with the contact model, so the pass is not robust and is not, on its own, a reason to adopt the
foot. PR #669's roll-rigid sole fails at 60 cm as at 75 cm (F7).

**Drive loads in the crawl (reported, not judged; judged runs).** Peak torques 14.8 N·m (pitch), 11.9 N·m (roll) and
10.9 N·m (flexion), short transients under the 21 N·m (X4) and 37.8 N·m (X6) limits; RMS at most 2.7 N·m (roll),
2.3 N·m (pitch) and 4.5 N·m (flexion), under the continuous ratings; peak speeds 46 rpm (flexion), 32 rpm (pitch) and
7 rpm (roll). No saturation. A 0.02 m/s crawl does not test the walk's continuous load, so this does not clear the hind
knee's estimate above.

## What this means

1. **The foot comes first (D5).** A sole rigid in roll makes the robot stand on two edges at both sizes; a passive roll
   hinge mostly fixes it but depends on friction and contact stiffness. Next: a roll hinge with a centring spring, a
   compliant sole, or a smaller, rounder contact, judged under both contact settings.
2. **The walk needs feedback.** The crawl replays a plan; task 5's model-based controller (D10) should close the loop
   on the body's tilt and the feet's loads.
3. **Mass and the hind knees.** Reconcile the budget with the full expression package and the explicit parts (about
   +0.5 kg), then let the walk simulation measure the knees' continuous load. If it is over, the options are a 1.2:1
   crank ratio on the hind knees (a non-parallelogram linkage), X6 hind knees (about +$320 of hardware and +0.9 kg),
   or a lighter build (the reserve is 1.15 kg).

## Revision history

**Revision 1** (commit 6739087d), judged in `results/stand60_2026-11-18/` and `results/crawl60_2026-11-18/`: 11.31 /
12.44 kg; standing PASS (both variants); crawl FAIL for both feet (PR #669's foot C4 and C5; the roll ankle C4 on 3 of
34 swings, seeds 1 and 2). These verdicts describe revision 1 and stay on record.

**The independent check of revision 1** (18 Nov 2026) reproduced every result byte for byte and found no blockers.
What it found and what was done:

| # | Finding | Done in revision 2 |
|---|---|---|
| 1 | Front X6 at 72 mm (scaled) instead of PR #669's 90 mm; 4.5 mm from the X4 pitch drive | Moved to 90 mm: 22.5 mm |
| 2 | Model mass above the budget's option C range, unstated | Stated here and flagged (R1b); budget used the core expression package |
| 3 | Body shell on PR #669's 0.57 kg basis, unlike the 75 cm model and the budget | 0.85 kg × 0.8^2, as the budget |
| 4 | Closed reserve concentrated (× 1.52) on frame, shells and harness; ledger basis silent | Spread over every estimated part (× 1.15); factor in the ledger |
| 5 | Standing R1 compared the model with itself | Labelled a regression check; R1b flag against the budget added |
| 6 | Crawl planner's grid not scaled (fore-aft -64 to +56 mm, no zero) | Grid steps are parameters, scaled; 75 cm results unchanged byte for byte |
| 7-8 | Crank lane scaled into the drive housing; leftover belt parts and buried caps | Lane unscaled; parts removed |
| 9 | Bought electronics scaled × 0.8 and intersecting | Catalog sizes, laid out without intersections; checked |
| 10 | The scaling skipped default classes by a test that never fired | Scaling limited to the world body |
| 11 | Pre-registration said the soles start 7.2 mm above the floor | Corrected in amendment 1 (that is the sole centres) |
| 12 | Head and tail shells × s³ while the body shell is × s² | Kept, as the budget does; × s² would add 0.15 kg and about 15 % to the neck moment (0.09 N·m stays far under 0.63) |
| 13 | Linkage mass without a basis | T15's four-bar, 0.0825 kg per leg, unscaled |
| 14 | Docstring and comments inexact; audio, bus adapter, second Teensy not modelled | Fixed; added (0.10 + 0.02 kg) |
| 15 | Neck spring stiffness without a basis; stale ledger entry; servo joint parameters inconsistent | Basis stated; entry fixed; one set for all four servo joints |
| 16 | Sensitivity runs left model files in the system temporary folder | Fixed |
| 17 | Swing timing rounded to the 10 ms control step (≤ 4.4 ms) | Noted; negligible |
| 18 | Ankle springs kept their 75 cm values (about 2× stiffer relative to weight) | Scaled for similarity |
| 19 | Standing seeds settle identically | Noted above |
| 20 | Flexion drives can reach the body proxy at extreme poses (also at 75 cm) | At 60 cm, revision 2: at least 26.3 mm over the joint grid |
| 21 | Report wording | Rewritten |

**Recheck of revision 2:** requested from the same reviewer; its result will be added here.
