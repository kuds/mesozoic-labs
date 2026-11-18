# Scripted crawl: report (task 4)

18 Nov 2026. **Simulation results** in MuJoCo 3.10.0; no delay, noise or sensor models yet; nothing measured. Rules
were committed before any crawl ran (`PREREG_CRAWL.md`, commit 1a3284d5); development changes and a design-variant
experiment were added in amendment 1 before any judged run (commit df6e4b40). No rule was changed.

## Answer

**With PR #669's foot as designed, the pre-registered crawl fails; with a passive roll hinge added at each ankle, it
passes every rule on both judged seeds, at both 20 kg and 15 kg.**

| Model (closed: 20.00 / 15.00 kg) | Verdict | Fails | Min static margin | Max planted pin motion | Travel (planned 0.96 m) | Max tilt |
|---|---|---|---|---|---|---|
| T20, PR #669 ankle (passive pitch only) | **FAIL** | C4, C5 | 61.4 mm | 25.1-25.3 mm (rule ≤ 10) | 1.05 m | 0.95° |
| T15, PR #669 ankle | **FAIL** | C4, C5 | 64.3-64.5 mm | 24.7-24.8 mm | 1.06 m | 0.83° |
| T20 + passive roll ankle (design variant) | **PASS** | none | 62.5 mm | 7.1-8.3 mm | 1.06 m | 0.63° |
| T15 + passive roll ankle (design variant) | **PASS** | none | 64.8-65.5 mm | 7.7-8.8 mm | 1.05-1.06 m | 0.88° |

Seeds 1 and 2 (floor friction 0.71-0.99, joint start offsets ±0.02 rad). Full table:
[results/crawl_2026-11-18/RESULTS.md](results/crawl_2026-11-18/RESULTS.md); per-run records beside it.

- **The crawl:** one foot at a time (left hind, left fore, right hind, right fore), the body shifted inside the
  support triangle first with a planned static margin of 65 mm (the rule asked ≥ 30 mm) on the conservative ankle-pin
  basis; open loop, joint targets from inverse kinematics, the model's position servos. 0.12 m stride, 6 s per cycle:
  **0.02 m/s**, a slow demonstration gait, not the 0.30 m/s museum walk.
- **Measured margin** stayed at least 61 mm on every model: the body shift works. What fails on PR #669's foot is the
  contact, not the balance.

## Why PR #669's foot fails: a design finding

PR #669's ankle is a passive pitch hinge with a 110 × 80 mm sole, so each sole is **rigid in roll**. Two effects
follow, both found in development (48 runs, `PREREG_CRAWL.md` amendment 1) and confirmed in the judged runs:

1. **The robot stands on two diagonal soles' edges, and the third support foot floats (C4).** With a foot lifted, the
   two diagonal feet's soles can carry the centre of pressure up to about 61 mm off the diagonal line, so the
   position-controlled legs settle with almost no load on the third foot (1-3 % of the weight instead of the ≥ 5 % the
   rule asks). The development runs showed the body then slowly rolling about the diagonal until the swing foot landed.
2. **Shifting the body sideways rocks the soles onto their edges and moves the planted feet (C5).** A sideways body
   shift over a planted foot leans the leg in roll; a sole rigid in roll leans with it and stands on one edge. The
   left feet's lateral centre of pressure sat on the sole's outer edge for most of the stance (median +40 mm), the
   right feet's anywhere from one edge to near the centre (medians -40 to +12 mm), and each ankle pin swung about
   25 mm per stance. Shifting the body fore-aft instead is limited by the pitch ankle's ±22°
   range, which the development runs also hit.

A cast urethane sole will not fix this by itself: a 9 mm, 60A sole of this size is stiff in roll (about
2,000 N·m/rad, a rough estimate from E ≈ 4 MPa). **With a passive roll hinge** (±20°, the same light spring as the
pitch hinge) the soles stay flat (median centre of pressure within 5.5 mm of each sole's centre), the third foot carries at
least 5 % of the weight through every swing (rule C4; 8.5-19 % in the development runs), and the pins move 7-9 mm.

This is a **proposal for the owner** (chassis decision D5), not an adopted change. It costs a second passive hinge per
foot (mass and price not estimated yet) and makes standing depend on the roll drives, as the parts list already
assumes when it sizes them from the full foot load (the standing check's F5).

## Report-only quantities (seed 1; `results/crawl_2026-11-18/extra.json`)

Joint side, the worst of the four legs. Limits: 0.70 × maker peak (× ratio × 0.90 for flexion); continuous = the
thermal-equilibrium torque at the joint; speed = 0.70 × no-load at 42 V for the 48 V drives, 83 rpm cap for the X4.

| Joint (T20 + roll ankle) | Peak torque | RMS torque | Peak speed | Use: peak / continuous / speed |
|---|---|---|---|---|
| Roll, X6 | 15.2 N·m of 42.0 | 5.0 of 12.0 | 6.3 rpm | 0.36 / 0.42 / 0.06 |
| Pitch, X4 | 13.7 N·m of 21.0 | 4.3 of 7.5 | 28.9 rpm of 83 | 0.65 / 0.57 / 0.35 |
| Front elbow, X8 at 1:1 | 18.2 N·m of 75.6 | 8.8 of 30.6 | 38.8 rpm of 97 | 0.24 / 0.29 / 0.40 |
| Hind knee, X6 at 1.3:1 | 14.7 N·m of 49.1 | 7.2 of 14.0 | 38.4 rpm (50 at the drive, of 108) | 0.30 / 0.51 / 0.46 |

- T15 + roll ankle: roll on the X4 peaks at 14.7 N·m of 21.0 (0.70) with RMS 4.3 of 7.5 N·m (0.57), the tightest axis,
  as the parts list predicts for T15 roll.
- The roll drives carry more with the roll ankle (15.2 against 11.3 N·m peak at 20 kg) because the soles no longer
  take the roll moment.
- A crawl is the gentlest gait; task 7 judges sizing on the 0.30 m/s walk.
- **Margin with the soles' width counted** (report only): at least 95 mm on every model.
- **The 1.3:1 pushrod band (A5):** every 1.3:1 joint stayed between -27.1° and +12.5°, inside the -29° to +24° band
  where the constant ratio holds within 10 %.

## What this does not show

- No servo delay, gear play, friction model, voltage sag or sensor noise (task 5 adds them, reusing the T. rex
  models); no turning; flat floors only; no feedback from the body's pose (the crawl is open loop).
- 0.02 m/s is a demonstration of static stability, not a museum gait.
- The roll-ankle variant's hinge is a modelling idealisation (frictionless, ±20°, 3 N·m/rad); a real design needs CAD,
  mass, sealing and a bench test.
- Two judged seeds per model; the development seeds (100-103) also passed for the variant.
