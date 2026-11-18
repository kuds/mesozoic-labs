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

As at 75 cm: a crawl at about 0.02 m/s is not the museum walk; there is no delay, noise or sensor model.
