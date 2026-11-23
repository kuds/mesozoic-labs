# Run 3: terrain training on two seeds, judged by its pre-registration

23 Nov 2026. Simulation only, on the 60 cm model with the capsule sole and the 188 mm upper foreleg. Rules:
`PREREG_RUN3.md` (written before either run). Judged by `judge_run3.py` (committed before it was applied), from the
gait report and the sweep run at the run commit. Nothing here is measured on a robot.

## Answer

Six of the seven rules pass on both seeds; the push rule (T5) fails on both. So the pre-registered statement "the
setup handles small steps, ramps and pushes without losing flat walking" does **not** hold for run 3. Both policies
walk flat ground at target speed, walk 3 cm steps and 8 deg ramps at full speed, stay straight on terrain and on a
slippery floor, and keep their drives under their continuous ratings. Both fall to sideways pushes of 1.0 m/s and
more, where run2b, trained without terrain or pushes on another foot, did not fall at 1.0 m/s.

| Rule | Pass when | Seed 0 | Seed 1 | Verdict | run2b (reference) |
|---|---|---|---|---|---|
| F1 flat speed | >= 0.225 m/s, no falls in 5 | 0.241 m/s, 0 falls | 0.239, 0 | **PASS** | 0.245, 0 |
| F2 flat heat | every joint RMS torque / continuous rating < 1.0 | worst 0.84 (left hind knee) | worst 0.80 (left hind pitch) | **PASS** | 0.72 |
| T1 3 cm step | no fall in 8, >= 80 % of nominal speed | 0, 100 % | 0, 100 % | **PASS** | 0, 46 % |
| T2 8 deg ramp | no fall in 8, >= 80 % of nominal | 0, 100 % | 0, 100 % | **PASS** | 0, 66 % |
| T3 straight on terrain | no fall, drift <= 0.30 m at 2 and 3 cm steps, 5 and 8 deg ramps | 0.08-0.17 m | 0.21-0.26 m | **PASS** | 0.66-1.15 m |
| T4 slippery floor (friction 0.3) | no fall, drift <= 0.20 m, >= 80 % | 0, 0.15 m, 99 % | 0, 0.06 m, 100 % | **PASS** | 0, 0.32 m, 95 % |
| T5 sideways pushes | no fall at -1.0 or +1.0 m/s; <= 2 of 8 at -1.25 and +1.25 | 5, 4; 8, 8 | 0, 2; 8, 8 | **FAIL** | 0, 0; 4, 7 |

Values are as the tools stored them; the percentages and drifts are rounded here. Push columns: falls of 8 at -1.0,
+1.0; -1.25, +1.25 m/s (free-body velocity change, 0.1 s, at 8 s). Nominal speed: 0.243 m/s (seed 0), 0.240 (seed 1).

## Provenance (checked before judging)

- Both runs: one session each, no stop, at commit efd1e1c7 (clean), 10,002,432 steps, seeds 0 and 1,
  `trike60_C_closed_capsule`, `configs/run3_terrain.json` (the run records' sessions).
- The judged checkpoints are each run's final pair. Both tools measured the same pair (sha256 of the zip and of the
  normalisation statistics agree), with the run commit's `trike_env.py` and model (sha256 equal to the commit's
  files). The tools were the run commit's, through a `TRIKE_ENV_DIR` export, because `rl/` has changed since.
- The rule script reproduced the pre-registration's whole run2b reference column from run2b's stored results before
  it was applied. Its first application found a fault in its own checkpoint comparison (the sweep records the pair
  as sha256 values, not a file name); the fix (commit 3c8851e4) changed no rule.
- Seed 0's export first lacked two read-only inputs (the top-level `configs/` and the T. rex study's actuator
  catalogue); the gait report stopped at start-up and was rerun after they were added. Nothing measured changed.

## Reported, not judged (against run2b, as the pre-registration asks)

- **Pushes, the whole family:** both seeds hold every push up to 0.75 m/s either way with no fall. Run2b broke first
  at 1.25 m/s; run 3 at 1.0 m/s on both seeds (seed 0: 5 and 4 of 8; seed 1: 0 and 2 of 8). Training drew pushes
  uniformly from 0 to 1.0 m/s, so the 1.0 m/s point lies at the edge of what was trained.
- **Terrain speed:** run 3 keeps full speed on 3 cm steps and 8 deg ramps, where run2b lost 54 % and 34 %.
- **Delay:** the first break at 0.25 m/s is at 60 ms on both seeds and on run2b (seed 0: 4 of 8 fell; seed 1: 75 %
  below nominal speed; run2b: 31 % below). At 0.15 m/s seed 0 breaks earlier, at 40 ms (2 of 8 fell).
- **Friction, mass, payload, force limit, position gain:** no break in the tested ranges, for either seed or run2b.
- **Flat walking with randomisation (gait report):** worst sideways drift 0.37 m and 0.54 m against run2b's 0.23 m;
  worst heading change 12.5 and 11.9 deg against 4.4. Mechanical power 30.9 and 24.2 W against 22.1 W. Front drives
  at 0.68-0.82 (pitch) and 0.51-0.66 (elbow) of their continuous ratings, against 0.52-0.66 and 0.58-0.61.
- **Balance figures** are not compared with run2b (the capsule stands on a line; `PREREG_RUN3.md`).

## What the result can and cannot be put down to

- **Not to the terrain training alone.** Run 3 differs from run2b in the foot (capsule against the roll-ankle box
  sole) and the slip cost at the contacts, besides the terrain, pushes and friction training. The pre-registration
  says so; the push result cannot be assigned to one of them.
- **On a foreleg that will change.** Both runs use the 188 mm upper foreleg, whose elbow drive overlaps the upper
  strut (F18). The foreleg comparison (D22) trains the 128 and 160 mm forelegs with run 3's setup, so its results
  can be read against these.

## What follows

- Pushes are the weak point: a next run could widen the training push range beyond 1.0 m/s, or add the push
  recovery the sweep measures to the reward. That is a decision for the owner, after D22.
- The steering run (D20) and the foreleg comparison (D22) both start from run 3's setup; this report is their
  reference.

## Files

- Rules: `analysis/PREREG_RUN3.md`; judge: `analysis/judge_run3.py`; verdicts: `analysis/results/judge_run3.json`.
- Gait reports: `analytics/results/run3-s0/`, `run3-s1/`. Sweeps: `robustness/results/run3-s0/`, `run3-s1/`.
