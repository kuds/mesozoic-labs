# Pre-registration: scripted crawl, 60 cm build (option C)

Written 18 Nov 2026, **before** any crawl was run on the 60 cm models. Committed with the models and
`crawl_judged60.py` in one commit. Any rule changed after a result will be labelled "changed after a result".

This repeats the 75 cm crawl (`PREREG_CRAWL.md` with its amendment 1) on the 60 cm models, with no tuning: the same
controller (`crawl.py`), the same rules C1-C6 **with their thresholds unchanged in absolute terms** (10 mm margin,
10 mm pin motion, 0.10 m drift, 10° tilt and heading), the same seeds and the same verdict logic. Keeping the
millimetre thresholds makes the rules stricter for the smaller robot, relative to its size; that is deliberate, since
build tolerances and sensor noise do not shrink with the robot.

## What changes: the gait parameters, scaled (no development runs)

Lengths × 0.8 (the geometric scale); times × √0.8 = 0.894 (Froude scaling, the same dynamic similarity the budget's
speed estimates use); everything dimensionless unchanged. Set in `crawl_judged60.py` as `PARAMS60`:

| Parameter | 75 cm (frozen) | 60 cm |
|---|---|---|
| stride | 0.120 m | 0.096 m |
| lift | 40 mm | 32 mm |
| shift time | 1.00 s | 0.894 s |
| swing time | 0.50 s | 0.447 s |
| planned margin | 65 mm | 52 mm (the 75 cm rule's floor is 30 mm; the scaled floor would be 24 mm) |
| sideways body shift limit | 100 mm | 80 mm |
| fore-aft adjustment limit | 80 mm | 64 mm |
| contact preload | 2 mm | 1.6 mm |
| lift fraction, sideways cost, cycles, standing time, control step | 0.20, 3.0, 8, 1.0 s, 10 ms | unchanged |

## What is run

- **Models:** `generated/trike60_C_closed.xml` (PR #669's foot: passive pitch ankle, sole rigid in roll) and
  `generated/trike60_C_closed_ankleroll.xml` (the same with a passive roll hinge at each ankle, ±20°, as at 75 cm).
- **Judged seeds:** 1 and 2 per model (the same draws as at 75 cm: hinge offsets ±0.02 rad, one friction coefficient in
  [0.6, 1.0]). Budget: 4 judged runs.
- **Sensitivity (reported, not judged):** the same 4 runs with `impratio="10"` added to `<option>` (the stiffer friction
  cone under which the 75 cm roll-ankle pass did not hold). A model whose pass does not hold under it is reported as
  contact-sensitive and is not recommended on the strength of the crawl.
- **Development:** none before the judged runs. If the scaled parameters cannot be planned (the planner raises), that
  is reported as a failure of the scaled plan; any later change of parameters is labelled "changed after a result" and
  judged on new seeds 3 and 4.

## Verdict

Per model, PASS if C1-C6 hold on both judged seeds, as in `PREREG_CRAWL.md`.

## What this does not show

As at 75 cm: a crawl at about 0.02 m/s is not the demo walk; there is no delay, noise or sensor model.

## Amendment 1 (18 Nov 2026, after the judged runs and the independent check)

**Changed after a result.** The judged runs above (`results/crawl60_2026-11-18/`) used revision 1 of the models and
failed. The independent check found that the planner's search grid did not scale with the parameters (its 10 mm and
4 mm steps are fixed in `crawl.py`, so the fore-aft grid ran from -64 to +56 mm without zero, not ±64 mm), and errors in
the models (see `PREREG_STAND60.md`, amendment 1). Because these are changes after a result, the repeat runs use **new
judged seeds, 3 and 4** (the rule set out above for any later change), on revision 2 of the models, in
`results/crawl60_2026-11-18b/`. Seeds 1 and 2 are rerun on revision 2 too and reported, not judged.

**Disclosure.** Before writing this, I had seen the first run's results and my diagnostic of its C4 failures; the
reviewer's sensitivity run with a scaled grid on revision 1 (verdicts unchanged; the base foot's failing swings fell
from 16 to 2 of 34; the roll ankle's rose to 4-5); and revision 2's generator printout and packaging check. I had not
run the crawl on revision 2.

**Changed:**
- `crawl.py` takes the grid steps as parameters (`grid_dx_m`, `grid_dy_m`), defaulting to the old 10 mm and 4 mm, so
  the 75 cm runs are unchanged (rechecked byte for byte); `PARAMS60` sets them to 8 mm and 3.2 mm.
- The models are revision 2. Among its changes, the ankle springs are scaled for similarity (stiffness 3 N·m/rad ×
  (12.44 / 20) × 0.8 = 1.49 N·m/rad; damping also × √0.8); revision 1 kept the 75 cm springs, about twice as stiff
  relative to the robot's weight.
- The impratio-10 runs no longer leave model files in the system temporary folder; the results record the model's
  name instead of a temporary path.
- Rules C1-C6, their thresholds, the gait parameters other than the grid, and the verdict logic are unchanged.

## Amendment 2 (18 Nov 2026, after the recheck of revision 2): revision 3, new judged seeds 5 and 6

**Changed after a result.** The recheck found the ankle springs scaled from the wrong reference: PR #669's 3 N·m/rad
belongs to its own 15 kg model, not to T20's 20 kg, and revision 2 used revision 1's mass. Revision 3 scales from
15 kg to the design's closed mass, 12.7 kg: stiffness 3 × (12.7 / 15) × 0.8 = 2.03 N·m/rad, damping 0.15 × (12.7 / 15)
× 0.8 × √0.8 = 0.091 N·m·s/rad. It also moves the NVMe tray and the reserve block clear of the Pilz block (a 0.15 kg
block moves 60 mm aft). Revision 2's crawl results (`results/crawl60_2026-11-18b/`) stay on record, and serve as the
softer-spring sensitivity (1.49 N·m/rad).

**Seeds.** Judged: **5 and 6** (new). Reported, not judged: 1-4, on revision 3. Sensitivity (reported): impratio 10 on
seeds 5 and 6. Everything else as in amendment 1. Results in `results/crawl60_2026-11-18c/`.

**Disclosure.** Before writing this, I had seen the reviewer's own runs on revision 2 with the spring set to
2.03 N·m/rad (reviewer-only, not committed): the roll ankle passed C1-C6 on seeds 1-4 as built and failed C4 on all
four under impratio 10. I had not run the crawl on revision 3.
