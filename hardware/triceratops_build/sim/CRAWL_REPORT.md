# Scripted crawl: report (task 4)

18 Nov 2026; revised the same day after an independent check. **Simulation results** in MuJoCo 3.10.0; no delay,
noise or sensor models yet; nothing measured. Rules were committed before any crawl ran (`PREREG_CRAWL.md`, commit
1a3284d5); development changes and a design-variant experiment were added in amendment 1 before any judged run
(commit df6e4b40). No rule was changed.

## Answer

**With PR #669's foot as designed, the pre-registered crawl fails. With a passive roll hinge added at each ankle it
passed every rule on both judged seeds at 20 kg and 15 kg, but that pass is thin and depends on the contact model:
with stiffer friction it fails too.** The robust finding is about the foot: a sole that is rigid in roll tips onto its
edges whenever the body shifts sideways. The roll hinge keeps the sole flat in stance; it is a direction, not a proven
fix.

| Model (closed: 20.00 / 15.00 kg) | Judged verdict (seeds 1, 2) | Fails | Min static margin | Max planted pin motion (rule ≤ 10 mm) | Travel (rule ≥ 0.77 m) | Max tilt |
|---|---|---|---|---|---|---|
| T20, PR #669 ankle (passive pitch only) | **FAIL** | C4, C5 | 61.4 mm | 25.1-25.3 mm | 1.05 m | 0.95° |
| T15, PR #669 ankle | **FAIL** | C4, C5 | 64.3-64.5 mm | 24.7-24.8 mm | 1.06 m | 0.83° |
| T20 + passive roll ankle (design variant) | **PASS** | none | 62.5 mm | 7.1-8.3 mm | 1.06 m | 0.63° |
| T15 + passive roll ankle (design variant) | **PASS** | none | 64.8-65.5 mm | 7.7-8.8 mm | 1.05-1.06 m | 0.88° |

Seeds 1 and 2: floor friction 0.71-0.99 and joint start offsets ±0.02 rad. The base and variant runs of one seed do
not share a friction value (the variant's four extra joints shift the random draws; the pre-registration's wording
allows it). Full table: [results/crawl_2026-11-18/RESULTS.md](results/crawl_2026-11-18/RESULTS.md).

**Contact-model sensitivity (not a judged result;
[results/crawl_2026-11-18/sensitivity_impratio10.txt](results/crawl_2026-11-18/sensitivity_impratio10.txt)).** The
same eight runs repeated with a stiffer friction model (MuJoCo `impratio` 10): the roll-ankle variant fails C4 on 4 of
4 runs and C5 on 3 of 4 (pins 10.9-12.4 mm); PR #669's foot still fails (24.1-25.1 mm). The independent check found the
same, and also a failure with `noslip_iterations` 20. Its explanation: the roll spring keeps the sole lined up with the
leaning leg during the swing, so the sole lands up to about 15° tilted and then rolls flat on its edge, which swings
the pin. **Until the variant passes under at least one stiffer contact setting, its pass is not a recommendation.**
Next step for the foot: hold the sole level during the swing (for example a spring that centres the sole to the
ground, not to the leg) and re-test under both contact settings.

- **The crawl:** one foot at a time (left hind, left fore, right hind, right fore), two setup swings, then 8 cycles;
  before each swing, with four feet down, the body is shifted so that the planned centre of mass is inside the
  support triangle on the conservative ankle-pin basis. Planned margin 65 mm where the reach allows; the minimum
  planned was 61.6 mm (T20, a left-fore step with the body at its 100 mm sideways limit); the rule asked ≥ 30 mm. Open
  loop, joint targets from inverse kinematics, the model's position servos. 0.12 m stride, 6 s per cycle:
  **0.02 m/s**, a demonstration of static stability, not the 0.30 m/s museum walk.
- **Measured margin** stayed at least 59 mm in every judged and `impratio` run (58.1-58.5 mm in the independent
  check's `noslip` runs): the body shift works. What fails is the
  contact at the feet.

## What fails on PR #669's foot, and why

PR #669's ankle is a passive pitch hinge under a 110 × 80 mm sole, so each sole is **rigid in roll**.

1. **The soles tip onto their edges (C5).** A sideways body shift over a planted foot leans the leg in roll, and a sole
   rigid in roll leans with it. In the judged runs the base soles were tilted more than 5° in about 75 % of loaded
   stance samples (2-3 % with the roll hinge; the independent check's figures, seed 1). The ankle pins therefore swing,
   almost all sideways: median 16 mm, maximum 25.3 mm per stance (T20 seed 1). **The soles themselves slide less: median
   3-5 mm per stance, worst stance 6-11 mm, and no less with the roll hinge** (the independent check, integrating the
   contact slip velocity, seed 1): C5, which measures the pin, mostly catches the tipping.
2. **The third support foot's load dips at lift-off (C4).** In the judged runs (the independent check's analysis of
   them) the third foot carries a median 9-12 %
   of the weight during a swing, but its minimum is 2.6-5.0 %, reached 14-42 % of the way into the swing (mostly 14-30 %),
   and it stays under 5 % for 2-22 % of the window (up to about 0.11 s). T20 seed 1 fails on 32 of 34 swings; T15
   seed 1 on 17 of 34, mostly by under 0.6 percentage points. At the
   smaller planned margins of the development runs (30-50 mm) the effect was larger: the robot balanced on a diagonal
   pair of soles' edges and the third foot rose off the floor (ledger rows 5-9, 15). The two diagonal soles can carry
   the centre of pressure up to about 61 mm off the diagonal line (the sole's half-extent across it, 60-62 mm).
3. **The pitch ankle's ±22° range limits fore-aft body shifts.** In development a hind ankle reached +25.5°, past its
   stop (ledger row 32); in the judged runs the pitch ankles came within 1° of the stop (-21.1° to -21.7°, the
   independent check).

The development runs ([results/crawl_2026-11-18/DEV_LOG.md](results/crawl_2026-11-18/DEV_LOG.md)) tried the other
ways out within this foot: smaller and larger sideways shifts, fore-aft shifts instead, a faster lift, contact preload
(0, 2 and 5 mm gave the same rule results in the seed-1 runs the independent check made). None passed C4 and C5
together. **The frozen parameters were then tuned on the roll-ankle variant** (ledger rows 17-39); PR #669's foot got
rows 1-16 and the runs at the frozen settings.

A cast urethane sole will not make the sole compliant enough in roll by itself: a 9 mm, 60A sole of this size is about
2,000 N·m/rad (a rough estimate, E ≈ 4 MPa; a bonded pad would be stiffer).

**For the owner (chassis decision D5):** for a statically stable gait like this crawl, which shifts the body sideways,
a foot that stays flat in roll is needed whatever chassis is chosen; whether the 0.30 m/s walk needs it too is for the
walk simulation to show. A passive roll hinge is the first candidate; it is not yet shown to
work robustly in simulation, and it costs a second hinge per foot (mass and price not estimated).

## Report-only quantities (seed 1; `results/crawl_2026-11-18/extra.json`)

Joint side, the worst of the four legs. Limits: 0.70 × maker peak (× ratio × 0.90 for flexion); continuous = the
thermal-equilibrium torque at the joint; speed = 0.70 × no-load at 42 V for the 48 V drives, 83 rpm cap for the X4.

| Joint (T20 + roll ankle) | Peak torque | RMS torque | Peak speed | Use: peak / continuous / speed |
|---|---|---|---|---|
| Roll, X6 | 15.2 N·m of 42.0 | 5.0 of 12.0 | 6.3 rpm | 0.36 / 0.42 / 0.06 |
| Pitch, X4 | 13.7 N·m of 21.0 | 4.3 of 7.5 | 28.9 rpm of 83 | 0.65 / 0.57 / 0.35 |
| Front elbow, X8 at 1:1 | 18.2 N·m of 75.6 | 8.8 of 30.6 | 38.8 rpm of 97 | 0.24 / 0.29 / 0.40 |
| Hind knee, X6 at 1.3:1 | 14.7 N·m of 49.1 | 7.2 of 14.0 | 38.4 rpm (50 at the drive, of 108) | 0.30 / 0.51 / 0.46 |

- T15 + roll ankle: roll on the X4 peaks at 14.7 N·m of 21.0 (0.70), RMS 4.3 of 7.5 N·m (0.57), the tightest axis, as
  the parts list predicts for T15 roll.
- The roll drives carry more with the roll ankle (15.2 against 11.3 N·m peak at 20 kg) because the soles no longer
  take the roll moment.
- A crawl is the gentlest gait; task 7 judges sizing on the 0.30 m/s walk.
- **Margin with the soles' width counted** (report only): at least 94.9 mm on every model.
- **The 1.3:1 pushrod band (A5):** every 1.3:1 joint stayed between -27.1° and +12.5°, inside the -29° to +24° band
  where the constant ratio holds within 10 %.

## Corrections to the descriptions in amendment 1 and `crawl.py` (the code is unchanged)

- **Foot landing.** The amendment says each cycle swing lands "on the body's steady path". In the code, `bx_nom` is the
  previous step's (already shifted) body position plus a quarter stride. The fore-aft adjustment sat at its ±80 mm
  limit on all 34 steps, so the body moved about +110 mm and -50 mm on alternate steps, and the fore feet land about
  80 mm further ahead than the wording suggests.
- **The first second** ramps the targets from the neutral pose to the pressed stance pose (2 mm preload); that changed
  during development and was not listed in the amendment.
- **Travel.** The rule uses 8 strides (0.96 m); the trajectory plans 1.015 m (setup swings and the final centring
  included), and the robot travelled 1.05-1.06 m.
- **Run count.** The amendment says 48 development runs; the ledger counts 49 simulations (one diagnostic trace was
  not counted). Both are inside the budget of 60.

## What this does not show

- No servo delay, gear play, friction model, voltage sag or sensor noise (task 5 adds them, reusing the T. rex
  models); no turning; flat floors only; no feedback from the body's pose (open loop).
- The swing foot's unloaded share (82-86 % of its window against the 80 % rule) and the soft contacts' creep (median
  3-5 mm, worst 6-11 mm per stance) are close to the limits even at 0.02 m/s. The walk work will check contact-model sensitivity as a matter
  of course and measure sole slip and tilt separately from pin motion.
- The roll-ankle variant's hinge is an idealisation (frictionless, ±20°, 3 N·m/rad); a real design needs CAD, mass,
  sealing and a bench test.
