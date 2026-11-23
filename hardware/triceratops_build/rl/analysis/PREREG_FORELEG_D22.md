# Pre-registration: the foreleg comparison, 128 against 160 mm under the RL policy (D22, D23)

23 Nov 2026. Simulation only. Written before any D22 run. Rules, seeds, budgets, stop rules and the choice are fixed
here; a rule changed after a result will be labelled "changed after a result".

**Disclosure.** The thresholds below were set after run 3's results (`REPORT_RUN3.md`) and after measuring run 3's
two seeds with the new foot-load tool (143.5 and 122.9 N), to keep the choice above seed-to-seed scatter. No D22
model has been trained; nothing from a D22 run informed them.

## What is trained

- **Code:** the reviewed commit that the four step sheets name (`runs/trike60-fore128-s0_STEPS.md` and the three
  others); all four runs use the same commit.
- **Models:** `sim/generated/trike60_C_closed_capsule_fore128.xml` (12.751 kg) and `..._fore160.xml` (12.808 kg),
  from `sim/build_foreleg60.py` (PR #697's repackaged forelegs, A-D22a's length-scaled masses; `PLAN_FORELEG_D22.md`).
  The tools check each model's sha256 against the run commit's file.
- **Setup:** run 3's, unchanged apart from the model: `configs/run3_terrain.json` (ENV_PRESET `run3_terrain`),
  10,000,000 steps, 8 envs, the same PPO settings, schedules and evaluations.
- **Seeds and runs:** `trike60-fore128-s0`, `trike60-fore128-s1`, `trike60-fore160-s0`, `trike60-fore160-s1` (SEED 0
  and 1 for each length). No other seeds and no reruns to improve a result.
- **Stop rules (run 3's):** a stopped session resumes under the same RUN_ID; a run that cannot resume is restarted
  once under a new RUN_ID with the same seed, reported, its first files kept; if that also fails, the run is not
  judged. A run is never extended or shortened.
- **What is judged:** each run's final pair (the last train.py writes at 10,0xx,xxx steps), never chosen among
  checkpoints.

## How it is measured

The run commit's tools, on each run's final pair:
- the gait report (`analytics/gait_report.py`) and the sweep (`robustness/sweep.py`), exactly as for run 3
  (`PREREG_RUN3.md`, "How it is measured");
- the foot loads (`analysis/foot_loads.py`): 5 episodes of 20 s at 0.25 m/s (seeds 300000-300004), flat floor,
  randomisation off, observation noise on, 10 ms delay, deterministic; every foot's normal force at every 1 ms
  physics step from 2 s, averaged over a moving 10 ms window.

## Measures per run

- Run 3's rules F1, F2 and T1-T5, with their fields and thresholds (`PREREG_RUN3.md`), applied by `judge_run3.py`'s
  rule function unchanged.
- **L, foot load:** the mean over the 5 episodes of the worst foot's 99th percentile of the 10 ms mean normal force
  (N; `worst_foot_p99_window_n`). A foot-load episode that falls leaves L undefined.
- **H, heat:** F2's figure, the worst joint's RMS torque over its continuous rating in the gait report's target case.
- **N, falls:** falls at run 3's judged sweep points (2 and 3 cm steps, 5 and 8 deg ramps, friction 0.3, pushes of
  -1.25, -1.0, +1.0 and +1.25 m/s; 8 episodes each) plus falls in the gait report's four cases (5 each).

For scale, run 3's two seeds (188 mm): L 143.5 and 122.9 N (16 % apart), H 0.836 and 0.801, N 25 and 18.

## The choice (D23)

`judge_d22.py`, in this order:
1. A length **walks** when F1 and F2 pass on both its seeds. If exactly one length walks, it is chosen. If neither
   does, no length is chosen.
2. Otherwise **128 mm**, unless 160 mm beats it on all three of:
   - **loads:** the larger of 160 mm's two L is at most 0.85 times the smaller of 128 mm's two L (so the gap must
     exceed the seed scatter, and holds for every pairing of seeds);
   - **falls:** 160 mm's N summed over its two seeds is at most 128 mm's (a tie is not "more");
   - **heat:** 160 mm's H averaged over its two seeds is at most 128 mm's average plus 0.05 (run 3's seeds differ
     by 0.035).
3. A run that lacks its tools' output, fails provenance, or has an undefined L makes the choice **not made**; the
   report then gives what was measured without a choice.

The choice is a simulation result about this setup, an input to the owner's decision, not the decision itself.

## Reported, not judged

- Run 3's rules for each length as for run 3 (PASS when both seeds pass, FAIL when neither, MIXED otherwise).
- Everything else the gait report, the sweep and the foot-load tool give, for both lengths, with run 3 (188 mm) as a
  reference: run 3's foreleg cannot be built (F18), and its fore-assembly differs besides the length (the shoulder
  40 mm higher, the X6 placement, about 14 % more forefoot travel per unit action), so it is not a third arm.
- The demo course (`demo_course.py --ckpt-dir`) for each run: winding temperatures, battery, joint and foot loads.
- The raw 1 ms foot-load peaks and the largest 10 ms mean, beside PR #697's figures from its scripted controllers.
- The proportions (upper foreleg over upper hind leg: 0.681 and 0.851; the specimen 0.682) and the posture.
- The gait report's "other foot" case swaps in `trike60_C_closed.xml`, the 188 mm foreleg with PR #669's ankle, so
  for these runs it changes the leg as well as the foot; it is reported, with that said.

## Not covered

- Turning: these policies walk straight, as run 3's do. PR #697's pivot and turning loads came from scripted
  controllers; the RL policy's turning loads wait for the steering run (D20) on the chosen foreleg.
- The moving pushrod, the dressed meshes' clearances and the shoulder crossmember (F19): geometry, not dynamics.
- Measured masses, the real servo loops, the rubber of the foot; the policy observes the simulator's true velocity.

## Checked end to end before any run

On commit d11125e6: a 16,384-step run on each model (`train.py` with `run3_terrain.json`), then the gait report
(`--quick`), the sweep (one episode per point) and `foot_loads.py` on each, and `judge_d22.py` on their output (the
final-pair step check relaxed for the trial only). Every tool ran; provenance passed for both models; the script
returned "no choice: neither length passes F1 and F2 on both seeds", as it should for untrained policies. Not a
result.

## Expected cost

Four Colab runs of about 3 hours each (run 3's took 3.0-3.1 h; the models are 0.06-0.11 kg heavier). Measurement
here: about 15 minutes per run.
