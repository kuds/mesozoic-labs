# Run 3: terrain training on two seeds, judged by its pre-registration

23 Nov 2026. Simulation only, on the 60 cm model with the capsule sole and the 188 mm upper foreleg. Rules:
`PREREG_RUN3.md` (written before either run). Judged by `judge_run3.py`, from the gait report and the sweep run at the
run commit. What the pre-registration asks to be reported without a verdict (the gait report's four cases and its
command sweep, every sweep point) is in `REPORT_RUN3_APPENDIX.md`, generated from the tools' stored output by
`run3_appendix.py`. Nothing here is measured on a robot.

Revised after its independent review (23 Nov): the appendix added; the push wording corrected for seed 1; the drive
and flat-walking statements narrowed to what was measured; the causes and the limits completed.

## Answer

Six of the seven rules pass on both seeds; the push rule (T5) fails on both. So the pre-registered statement "the
setup handles small steps, ramps and pushes without losing flat walking" does **not** hold for run 3.

- **What passed:** on flat ground both policies walk 0.239-0.241 m/s at a 0.25 m/s command with no falls, and in
  that case every drive stays under its continuous rating (worst 0.84). Both climb a 3 cm step and an 8 deg ramp at
  full speed, drift sideways 0.08-0.26 m on average over 2-3 cm steps and 5-8 deg ramps, and 0.15 m and 0.06 m on
  friction 0.3. These conditions lie inside run 3's training ranges.
- **What failed:** sideways pushes. Seed 0 falls at both +-1.0 m/s (5 and 4 of 8); seed 1 at +1.0 m/s (2 of 8) but
  not at -1.0. Both fall 8 of 8 at +-1.25 m/s, where run2b fell 4 and 7 of 8. Pushes up to 0.75 m/s either way
  caused no falls.
- **Flat walking changed, though F1 and F2 pass** (appendix A): a slower, longer stride (0.70 and 0.86 Hz against
  run2b's 1.22); seed 0 trots, seed 1's pattern is not a locked trot ("other"); about half the strides irregular on
  the worst foot at 0.25 m/s (53 % and 51 % against 0 %) and about four in five at 0.15 m/s (79 % and 80 % against
  51 %); more sideways drift; a steady nose-down lean of 3.6-3.9 deg (run2b 0.6 deg nose up) and more pitch sway
  (SD 2.65 and 1.64 deg against 0.87); more time at the torque limit (the worst episode, averaged over the four pitch
  drives: 2.8 % and 1.5 % against 1.0 %; the worst single drive 5.1 % and 2.3 % against 1.7 %); and about 20-35 %
  more estimated electrical power (49-54 W against 40 W on the shared bus), so 84-89 min of walking per pack against
  101 min.
- **Drives beyond F2's case** (appendix B): at 20 ms of delay or more, the worst joint runs at 1.20-1.50 of its
  continuous rating on both seeds, at 0.25 and at 0.15 m/s (run2b stays under 1.0 up to 30 ms); seed 0 exceeds 1.0 also at mass x1.2-1.6
  and with a 2-3 kg payload (1.01-1.35), seed 1 at mass x1.4-1.6 (1.03-1.11); both at position gain x1.5 (1.16 and
  1.02; run2b 1.08).

| Rule | Pass when | Seed 0 | Seed 1 | Verdict | run2b (reference) |
|---|---|---|---|---|---|
| F1 flat speed | >= 0.225 m/s, no falls in 5 | 0.241 m/s, 0 falls | 0.239, 0 | **PASS** | 0.245, 0 |
| F2 flat heat | every joint RMS torque / continuous rating < 1.0 | worst 0.84 (left hind knee) | worst 0.80 (left hind pitch) | **PASS** | 0.72 |
| T1 3 cm step | no fall in 8, >= 80 % of nominal speed | 0, 100 % | 0, 100 % | **PASS** | 0, 46 % |
| T2 8 deg ramp | no fall in 8, >= 80 % of nominal | 0, 100 % | 0, 100 % | **PASS** | 0, 66 % |
| T3 straight on terrain | no fall, drift <= 0.30 m at 2 and 3 cm steps, 5 and 8 deg ramps | 0 falls; 0.08-0.17 m | 0 falls; 0.21-0.26 m | **PASS** | 0 falls; 0.66-1.15 m |
| T4 slippery floor (friction 0.3) | no fall, drift <= 0.20 m, >= 80 % | 0, 0.15 m, 99 % | 0, 0.06 m, 100 % | **PASS** | 0, 0.32 m, 95 % |
| T5 sideways pushes | no fall at -1.0 or +1.0 m/s; <= 2 of 8 at -1.25 and +1.25 | 5, 4; 8, 8 | 0, 2; 8, 8 | **FAIL** | 0, 0; 4, 7 |

Values are as the tools stored them; the percentages and drifts are rounded here. Push columns: falls of 8 at -1.0,
+1.0; -1.25, +1.25 m/s (free-body velocity change over 0.1 s, at 8 s). Nominal speed: 0.243 m/s (seed 0), 0.240
(seed 1), 0.246 (run2b).

## Provenance

Checked by `judge_run3.py` before judging: the model's and `trike_env.py`'s sha256, as both tools recorded them,
equal the run commit's files; both tools record the run commit; the gait report's checkpoint lies in the final
block (10,000,000-10,008,191 steps); both tools measured the same pair (zip and normalisation sha256).

Checked by hand (the script does not read the run records): each run had one session, started from step 0,
completed at 10,002,432 steps, at commit efd1e1c7 with a clean tree, seeds 0 and 1, `trike60_C_closed_capsule` and
`configs/run3_terrain.json`; the last checkpoint in each record is the judged pair; each sweep point has 8 episodes.
The independent review re-checked these and re-ran three sweep points with the run commit's tools, bit-identical.

- The tools were the run commit's, through a `TRIKE_ENV_DIR` export of efd1e1c7, because `rl/` has changed since.
  Seed 0's export first lacked two read-only inputs (the top-level `configs/` and the T. rex study's actuator
  catalogue, both from the run commit); the gait report stopped at start-up and was rerun after they were added.
- `judge_run3.py` (53f13a20) was committed after seed 0's two tools had finished and before their output was read
  against the rules; it had been checked only on run2b's stored results, whose whole reference column it
  reproduces. Two faults found since, neither changing a rule or a result: its checkpoint comparison read the
  sweep's record wrongly (fixed in 3c8851e4, on first application), and it crashed instead of failing a rule where
  every episode at a point fell or no target episode could be analysed (fixed after the review, with tests; run 3's
  verdicts and values are identical before and after).
- The pre-registration calls the gait report's 3.5 s cut `MIN_ANALYSED_S`; in the code that constant is 1.5 s after
  the 2 s analysis window, which gives the same 3.5 s. No target episode was short here.
- Neither run was stopped, so the notebook's stop and resume path on Colab is still untested: seed 0's step sheet
  asked for the stop and resume tests and its record shows no stop (the owner chose to skip them); seed 1's sheet
  left them out by design.

## What the result can and cannot be put down to

Run 3 differs from run2b in the foot (capsule against the roll-ankle box sole), the slip cost (at the contacts), the
drift cost (`w_lateral_drift` 100 against 0), the training length (10,002,432 against 8,003,584 steps) and the
terrain, push and low-friction training. The design cannot separate them:
- the straighter walking on terrain and low friction may come from the drift cost as much as from the terrain
  training;
- the push falls may come from the training (1.0 m/s was the top of the trained push range) or from the foot: the
  capsule stands on a line, and the centre of mass lies outside the support polygon 65 % and 46 % of the time
  (reported, not compared; appendix A);
- T1-T4 test conditions that run 3 trained on and run2b did not, so run 3's lead there is expected, not a sign of
  generality. At 1.0 m/s pushes, also within run 3's training and not run2b's, run 3 trails: it falls where run2b
  does not. Of the conditions only run 3 trained on, the 1.25 m/s pushes and the friction points below 0.3 lie
  beyond its training: at friction 0.15 and 0.2 both seeds stay upright, drifting 0.32-0.35 m (seed 0) and
  0.04-0.13 m (seed 1), against run2b's 0.56-0.78 m. (Delays of 15-60 ms, the mass, payload, force-limit and gain
  points outside x0.9-1.1 and x0.85-1.0, friction 1.2 and the 1.5 m/s pushes lie beyond both runs' training.)

Also not covered here, as the pre-registration says: the policy observes the simulator's true velocity, the capsule
is a rigid shape, and the terrain is blind to the policy.

Both runs use the 188 mm upper foreleg, whose elbow drive overlaps the upper strut (F18). The foreleg comparison
(D22) trains the 128 and 160 mm forelegs with run 3's setup, so its results can be read against these.

## What follows (options for the owner; nothing started)

- Pushes and the irregular flat gait are the weak points. Which change to try (a wider push range, a push-recovery
  term, the foot, or the drift cost's weight) is open: the causes above cannot be told apart from these runs.
- The steering run (D20) and the foreleg comparison (D22) start from run 3's setup; this report and its appendix are
  their reference.

## Files

- Rules: `analysis/PREREG_RUN3.md`; judge: `analysis/judge_run3.py` (tests: `tests/test_judge_run3.py`); verdicts:
  `analysis/results/judge_run3.json` (both seeds) and `judge_run3_s0_only.json` (seed 0, before seed 1 finished).
- Appendix: `analysis/REPORT_RUN3_APPENDIX.md`, from `analysis/run3_appendix.py`.
- Gait reports: `analytics/results/run3-s0/`, `run3-s1/` (with each run's `gait_report_episodes.json.gz`). Sweeps:
  `robustness/results/run3-s0/`, `run3-s1/`.
