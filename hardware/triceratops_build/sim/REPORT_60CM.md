# The 60 cm model: build, standing check and crawl (simulation)

18 Nov 2026. Everything here is simulation of a model whose part masses and positions are planning estimates;
nothing is built, bought or measured.

## Answer first

- **The 60 cm model (option C) is built**: PR #669's geometry scaled by 0.8, the T. rex drive types at catalog size,
  12.44 kg with the 10 % reserve (11.31 kg without). Parts list with masses, positions and prices:
  `design/PARTS_60CM.md`.
- **It stands** (pre-registered check, both variants, both judged seeds pass). Standing loads are about half the
  75 cm robot's: front elbow 3.9 N·m against the X6's 10.8 N·m continuous at the joint; hind knee 3.2 N·m against
  6.75; roll 0.8 N·m and pitch 0.6 N·m against 7.5; neck 0.08 N·m against the XC430's 0.63 (estimate).
- **The scripted crawl fails on both feet** (pre-registered, no tuning). PR #669's foot fails as it did at 75 cm: the
  third foot floats and the planted soles tip and slide (C4, C5). The roll-ankle variant holds its feet (C5 passes)
  but fails C4 narrowly, on 3 of 34 swings. Both fail under the stiffer contact setting. **The foot (D5) is still the
  first design problem**, and the walk (task 5) needs feedback, not a replayed script.

## The model

`build_model60.py` writes `generated/trike60_C_{listed,closed}.xml` (and `_closed_ankleroll`):

- **Geometry:** every body position, geom position, size and end point, site, joint position, camera and mesh of PR
  #669's `robot.xml` × 0.8, which gives its own 60 cm figures: hips and shoulders 320 mm high, 520 mm apart, track
  280 mm. Drives and servos keep their catalog sizes (X4 Ø55 × 61 mm, X6 Ø80 × 67.5 mm), so they look larger on the
  smaller frame.
- **Drives:** X4-36-L on all roll and pitch joints and the hind knees; X6 on the front elbows; all 1:1 through a
  parallelogram pushrod (constant ratio). Joint-side limits 0.70 × peak (× 0.90 on flexion). Gains are the T. rex
  servo model's nominal, as at 75 cm.
- **Expression (D14, full):** neck pitch with a counterbalance spring cancelling 70 % of the head's moment, a new neck
  yaw joint, tail yaw and a new tail pitch joint, each a position servo with 0.70 × stall as its limit (XC430-W240,
  XL430-W250). The beak servo and the eyelids are mass only; their joints are not modelled yet.
- **Masses:** catalog masses for bought parts; PR #669's own parts scaled (structure × 0.8^2.5, shells × 0.8^2, head
  and tail × 0.8^3); estimates elsewhere, each with its basis in the ledger (`generated/mass_ledger_*.json`).

## Standing check (`PREREG_STAND60.md`; results `results/stand60_2026-11-18/`)

| Variant | Mass | Front | Left | Front elbow | Hind knee | Roll max | Pitch max | Neck | Tail pitch | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| listed | 11.31 kg | 55.6 % | 49.7 % | 3.59 N·m | 2.84 N·m | 0.73 N·m | 0.58 N·m | 0.079 N·m | 0.175 N·m | **PASS** |
| closed | 12.44 kg | 55.0 % | 49.6 % | 3.91 N·m | 3.16 N·m | 0.80 N·m | 0.64 N·m | 0.079 N·m | 0.175 N·m | **PASS** |

Seeds 1 and 2 differ from the exact neutral pose only during the first seconds (lowest height 0.979-0.980 of the
start, against 0.997; tilt up to 0.16°) and settle to the same pose, as at 75 cm. The front elbow's standing torque
is within 3 % of the 75 cm parts list's figure scaled by weight × leg length (4.03 N·m); the roll torque is well
under its scaled figure (2.14 N·m), as at 75 cm, where the parts list's roll figure was also conservative.

## Crawl (`PREREG_CRAWL60.md`; results `results/crawl60_2026-11-18/`)

The 75 cm crawl's frozen parameters, scaled (lengths × 0.8, times × √0.8), with no development runs; rules C1-C6 with
their millimetre thresholds unchanged.

| Model | Seed | C1 | C2 | C3 | C4 | C5 | C6 | Min margin | Max pin motion | Max tilt |
|---|---|---|---|---|---|---|---|---|---|---|
| PR #669's foot | 1 | pass | pass | pass | **FAIL** | **FAIL** | pass | 50.5 mm | 19.6 mm | 0.99° |
| PR #669's foot | 2 | pass | pass | pass | **FAIL** | **FAIL** | pass | 50.5 mm | 19.7 mm | 0.98° |
| roll ankle | 1 | pass | pass | pass | **FAIL** | pass | pass | 51.0 mm | 8.4 mm | 1.13° |
| roll ankle | 2 | pass | pass | pass | **FAIL** | pass | pass | 51.0 mm | 9.3 mm | 1.14° |

Verdicts: **both FAIL.** Under `impratio="10"` (reported, not judged) both fail C4 and C5; the roll ankle's pin
motion rises to 10.5 mm.

**Why C4 fails (a diagnostic re-run of seed 1, not a judged result).** With PR #669's foot, 16 of the 34 swings fail
because the foot diagonally opposite the swinging one carries only 3-4 % of the weight (the rule asks 5 %): the robot
rests on the two soles' edges and the third foot floats, as at 75 cm (F7). With the roll ankle, 3 of 34 swings fail,
each a right-side swing during which the left foot on the same axle drops to zero load for part of the swing; the
swinging foot itself is unloaded for 87-89 % of its swing (the rule asks 80 %).

**What the crawl says about the drives (reported, not judged).** Peak joint torques during the crawl reach 12.6 N·m
(pitch) and 12.2 N·m (roll), short transients well under the 21 N·m limit; RMS torques are at most 2.5 N·m (roll and
pitch) and 4.2 N·m (flexion), well under the continuous ratings; peak speeds are 45 rpm (flexion), 29 rpm (pitch) and
7 rpm (roll). No actuator saturated. The drives are not what limits the crawl.

## What this means

1. **The foot comes first (D5).** At both sizes, a sole that is rigid in roll makes the robot stand on two edges; a
   passive roll hinge fixes the sliding but not the load sharing, and its benefit depends on the contact model. The
   next design step is a foot that keeps the sole flat and loaded: a roll hinge with a centring spring, a compliant
   sole, or a smaller, rounder contact.
2. **The walk needs feedback.** The crawl is open loop: it plans from the model and replays joint targets. Task 5's
   model-based controller (D10) should close the loop on the body's measured tilt and the feet's loads, which is
   what the C4 failures call for.
3. **The drives have margin.** Standing and crawling loads are a fraction of the X4's and X6's continuous ratings; the
   walk and trot (tasks 5-7) will test the peaks.

## Independent check

To be added when the reviewer's report is in.
