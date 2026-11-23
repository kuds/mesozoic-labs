# Pre-registration: the foreleg comparison, 128 against 160 mm under the RL policy (D22, D23, D24)

23 Nov 2026. Simulation only. Written before any D22 run. Rules, seeds, budgets, stop rules and the choice are fixed
here; a rule changed after a result will be labelled "changed after a result".

**History before any run.** A first draft (commit d11125e6: two seeds per length, a stricter load rule, heat on the
hottest joint only, falls with no tolerance) was reviewed independently. The review showed that two seeds cannot
resolve a 15 % foot-load effect, that its heat measure could not see the front drives, and that its tests treated
noise unevenly. The owner then chose this design (D24). No D22 run used the draft.

**Disclosure.** The thresholds and tolerances below were set after run 3's results (`REPORT_RUN3.md`) and after
measuring run 3's two seeds with the foot-load tool. No D22 model has been trained; nothing from a D22 run informed
them.

## What is trained

- **Code:** the reviewed commit that the step sheet (`runs/trike60-d22_STEPS.md`, one sheet for the eight runs) names.
  All eight runs use the same commit, and its `analysis/judge_d22.py`, `analysis/foot_loads.py`,
  `analysis/judge_run3.py` and `telemetry/loads.py` are the ones that judge them.
- **Models:** `sim/generated/trike60_C_closed_capsule_fore128.xml` (12.751 kg) and `..._fore160.xml` (12.808 kg),
  from `sim/build_foreleg60.py`: PR #697's repackaged forelegs with A-D22a's length-scaled masses
  (`PLAN_FORELEG_D22.md`). The run record and the tools are checked against the run commit's files.
- **Setup:** run 3's, unchanged apart from the model: `configs/run3_terrain.json` (ENV_PRESET `run3_terrain`),
  10,000,000 steps, 8 envs, the same PPO settings, schedules and evaluations. Before this was written, a 2,048-step
  training from the same seed gave identical weights at run 3's commit and at this tree (the review's check).
- **Seeds and runs (D24):** four per length, SEED 0-3: `trike60-fore128-s0` ... `-s3` and `trike60-fore160-s0` ...
  `-s3`. No other seeds and no reruns to improve a result.
- **Stop rules (run 3's):** a stopped session resumes under the same RUN_ID; a run that cannot resume is restarted
  once under a new RUN_ID with the same seed, reported, its first files kept; if that also fails, the run is not
  judged. A run is never extended or shortened.
- **What is judged:** each run's final pair (the last train.py writes at 10,0xx,xxx steps), never chosen among
  checkpoints.

## How it is measured

On each run's final pair, with the run commit's tools: in the tree if `rl/` has not changed since the run commit,
otherwise through a `TRIKE_ENV_DIR` export of the run commit's `rl/` (all three tools honour it, and all three read
the model from the tree's `sim/generated`, whose sha256 is checked against the run commit's file):
- the gait report and the sweep, exactly as for run 3 (`PREREG_RUN3.md`, "How it is measured");
- the foot loads (`analysis/foot_loads.py`): 5 episodes of 20 s at 0.25 m/s (seeds 300000-300004), flat floor,
  randomisation off, observation noise on, 10 ms delay, deterministic; each foot's total normal contact force (the
  sole's contacts summed) at every 1 ms physics step from 2 s, averaged over a moving 10 ms window. Two runs on one
  checkpoint give byte-identical output.

## Measures per run

- Run 3's rules F1, F2 and T1-T5, with their fields and thresholds (`PREREG_RUN3.md`), by `judge_run3.py`'s rule
  function unchanged.
- **L, front-foot load (D24):** the mean over the 5 episodes of the worst front foot's 99th percentile of the 10 ms
  mean normal force (N). A foot-load episode that falls leaves L undefined. The hind feet's figure is reported.
- **H_all and H_front, heat:** the worst joint's RMS torque over its continuous rating in the gait report's target
  case (F2's figure), and the same over the six front drives only (roll, pitch, elbow). The elbows' own figure is
  reported, not judged: the owner chose not to give them a separate test (D24, after the review noted that the
  front pitch drives, not the elbows, set H_front in run 3), since F2 already keeps every drive under its rating.
- **N, falls:** falls at run 3's judged sweep points (2 and 3 cm steps, 5 and 8 deg ramps, friction 0.3, pushes of
  -1.25, -1.0, +1.0 and +1.25 m/s; 8 episodes each) plus falls in the gait report's target, 0.15 m/s and nominal
  cases (5 each). Not its "other foot" case, which for these runs swaps in the 188 mm leg as well as the foot.

For scale, run 3's two seeds (188 mm, measured with this tree's `trike_env.py`, which the run-3 digest test shows is
bit-identical to run 3's for this setup): L 143.5 and 110.6 N (front; hind 135.2 and 122.9 N), H_all 0.836 and
0.801, H_front 0.823 and 0.722 (both the right front pitch drive; the elbows 0.53-0.66), N 25 and 18.

## Checks before judging (`judge_d22.py`; any failure makes the run NOT JUDGED)

- The run record: every session at the run commit with a clean tree, none allowed to change settings, the last one
  completed; the seed and the model; every other setting (PPO, schedules, 8 envs, 10,000,000 steps, the whole
  environment config), and every session's arguments as train.py records them (all but the output folder, seed,
  model and the notebook's process id), equal to run 3's (`analysis/results/run3_reference_settings.json`: run 3
  seed 0's whole record; run 3's two seeds differ in the seed only), with steering off; its last checkpoint is the
  one measured.
- The tools: the model's and `trike_env.py`'s sha256 as each recorded them, against the run commit's files; the run
  commit; the final pair; the same pair in all three tools; the gait report's settings (seeds 100000-100004, 20 s,
  10 ms, deterministic, from 2 s) and 5 episodes in each judged case, so a `--quick` report cannot pass; the
  sweep's 8 episodes per point on seeds 200000-200007; the foot-load tool's settings as above.
- Across runs: each length and seed given once; eight different checkpoints.

## The choice (D23, D24)

`judge_d22.py`, in this order:
1. A length **walks** when F1 and F2 pass on all four of its seeds. If exactly one length walks, it is chosen. If
   neither does, no length is chosen.
2. If any run's L or H_front is undefined, no choice.
3. Otherwise **128 mm**, unless 160 mm meets all three, on the four-seed means:
   - **loads:** mean L(160) at most 0.85 x mean L(128);
   - **falls:** mean N(160) at most mean N(128) + 7;
   - **heat:** mean H_all(160) at most mean H_all(128) + 0.035, and mean H_front(160) at most mean H_front(128) + 0.10.
   The tolerances are run 3's differences between its two seeds with the same leg ("not clearly worse", D24).
4. A run that is NOT JUDGED makes the choice not made; the report then gives what was measured without a choice.

The choice is a simulation result about this setup: an input to the owner's decision, not the decision.

## What this design can and cannot conclude

- **Its power, estimated before any run** (from run 3's seed scatter, about 18 % on the front-foot load, treating
  seeds as independent draws): if 160 mm truly lowers L by 25 %, the loads test passes about 80 % of the time; by
  15 %, about half the time; if the two are truly equal, it wrongly passes about 10 % of the time. The falls and
  heat tests add their own chances of keeping 128 mm. These are rough: the scatter comes from one pair of seeds on
  another leg.
- **What the falls and heat tolerances allow.** They are run 3's single-seed differences, applied to four-seed means,
  so they are generous: an equal 160 mm passes the falls test about 98 % of the time and the heat tests about 95 %;
  a 160 mm truly worse by the full tolerance (7 more falls per seed, about a third more than run 3's 21.5; 0.035
  more of rating overall; 0.10 more on a front drive) still passes about half the time. That is D24's "not clearly
  worse", chosen by the owner; the loads test is where 160 mm has to prove itself.
- "128 mm chosen" means 160 mm did not clearly beat it, not that 128 mm is better.
- Straight walking only: these policies walk straight, as run 3's do. PR #697's load advantage for 160 mm came from
  turns and pivots under scripted controllers; a straight-walking L may not show it. The RL policy's turning loads
  wait for the steering run (D20) on the chosen foreleg.

## Reported, not judged

- Run 3's rules for each length (PASS when all four seeds pass, FAIL when none does, MIXED otherwise).
- Everything else the gait report, the sweep and the foot-load tool give, for both lengths, with run 3 (188 mm) as a
  reference only: its foreleg cannot be built (F18) and its fore-assembly differs besides the length (the shoulder
  40 mm higher, the X6 placement, about 14 % more forefoot travel per unit action).
- The demo course (`demo_course.py --ckpt-dir`) for each run: winding temperatures, battery, joint and foot loads.
- The raw 1 ms foot-load peaks and the largest 10 ms mean, beside PR #697's figures from its scripted controllers.
- The proportions (upper foreleg over upper hind leg: 0.681 and 0.851; the specimen 0.682) and the posture.
- The gait report's "other foot" case, with the caveat above.

## Changes from the plan (`PLAN_FORELEG_D22.md`), made before any run

- Foot loads come from the dedicated flat-walking tool, not from the gait report's target case or the demo course
  (the gait report records foot forces at the 20 ms control step; the 10 ms window needs the 1 ms physics steps);
  the demo course's loads are reported.
- Heat is judged at 0.25 m/s (the gait report's target case); the 0.15 m/s case and the demo course are reported.
- Four seeds per length instead of two (D24).

## Not covered

- The moving pushrod, the dressed meshes' clearances and the shoulder crossmember (F19): geometry, not dynamics.
- Measured masses, the real servo loops, the foot's rubber; the policy observes the simulator's true velocity.

## Expected cost

Eight Colab runs of 3-5 hours each (run 3's two took 3.05 and 5.05 h, at 909 and 550 env steps/s: the speed
depends on the Colab machine), as four rounds of two in parallel. Measurement here: about 15 minutes per run.

## Checked end to end before any run

A 16,384-step run on each model (`train.py` with `run3_terrain.json`, two envs), the gait report (`--quick`), the
sweep (one episode per point) and `foot_loads.py` on each, then `judge_d22.py` (commit d11125e6, then again on the
revised script). Every tool ran on both models; the revised script marked every trial run NOT JUDGED for exactly the
trial's differences (steps, envs, the final pair, episodes per point) and passed every other check. Not a result.
