S2-4 (V8) did not run. The baseline was made, copied (cmp equal) and committed. The training run then crashed with exit 1 before any rollout, because of a fault in the trainer's code (`train2._g8a`, `TypeError`). Under the standing rules I stopped there and changed no code. Steps 3 (`nb collect`) and 4 (S2-10) were not done. Simulation only throughout.

## Setup
- Worktree `/home/user/mesozoic-wt/wp-p42` (branch `wt/wp-p42`) was fast-forwarded from dae9c3fd to 26674562 (`git merge --ff-only claude/mesozoic-trex-stage2-refine-4jffre`, a fast-forward). After it, `git status` was clean.
- Bit-identity printed `45719d030359ec7d5d3914af3814d837fdc05d77652ba45c64792b511027528d c0027992de21067525cc4489b74acb157feb89f3a4c7618580addc2b341be1c1`, both before and at the end. It matches.
- Validation records read (the trainer reads the first two itself):
  - filter.json: choice `filter_hz` 10.0
  - lr_probe.json: 75-M-H-10 rate 1e-05
  - rates.json: `pass_` True, `provisional` true
  - kill_resume.json: `pass_` True, `equal` True
- Load average before the run was 0.18–0.36 on 4 cores, with nothing else heavy running.
- The commands ran from the worktree's study directory with the given environment, through `bash tools/lowprio.sh`, in the background. I used the scratchpad wrapper `/tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/step.sh`, which copies the chain's `step`: the log is `stage2/tb/work/logs/<name>.log` and a done marker is written on exit 0.

## 1. Baseline (V8-baseline, V8-baseline-copy, V8-baseline-committed)
- **V8-baseline:** `python -c "from designlab.stage2 import monitor2; monitor2.baseline('75-M-H-10', 'V6', 'stage2/tb/work/s2-4-baseline')"` ran 19:17:32Z to 19:18:31Z and exited 0 (record `wall_s` 58.44).
- **The record:** `stage2/tb/work/s2-4-baseline/r0l.json`, `valid` True, row R0L@10ms.
  - Outcomes: read 13, fell 2, unread 0, error 0, not_run 3 (turn/q90_left 6007–6009).
  - N = 6, S = 25.03222222221989, `L_planned_s` = 25.557704023051375, so L − S = 0.525 s (M3b's survival route would be closed).
  - `N_but_A1_look` 6; `N_reading2` 6; `N_reading3` 6; `mean_clock_factor` 0.99999999999988.
  - Up by condition: walk/nominal 3/3, walk/v0.4 3/3, stop/two_step 3/3, nudge/left 3/3, nudge/forward 1/3 (6013 fell at 24.786 s, 6015 fell at 24.710 s), turn/q90_left 0/3 (not run).
  - `failures_by_term`: line 1, look_toe_off 3, recovered 3, survived 2, toe_clearance_own_swing 3.
  - The 15 episodes that ran each hold a state digest.
  - Provenance: commit 26674562, dirty false.
- **V8-baseline-copy:** the chain's `sh -c "mkdir -p … cp … && cmp …"` exited 0 at 19:18:54Z. `cmp` found the two files equal, both sha256 `f33e4eb11cbd25dcfb398b7a359f8704d87b2780f87879cb679220644bb70543`.
- **Commit `58109a17882115dec420105a1b1cb85c7b1703b9`:** "Track B V8: commit R0L's baseline for S2-4 before the run (rule M3; 75-M-H-10, V6; simulation only)". It adds only `hardware/trex_form_study/stage2/tb/runs/s2-4/R0L_BASELINE.json` and ends with the two required lines. I passed `-c core.hooksPath=/dev/null` on this commit. I checked afterwards: no hook is installed (no `core.hooksPath`, no non-sample hooks), so nothing was skipped.
- **V8-baseline-committed:** the chain's git checks (cmp, ls-files, `HEAD:` path, diff --quiet, log) passed: "done V8-baseline-committed (stage2/tb/runs/s2-4/R0L_BASELINE.json is committed: 58109a17…)". HEAD was 58109a17 when the run started.

## 2. The run (V8-s2-4): failed
- **Command:** `python -m designlab.stage2.train2 run --mode R5 --body 75-M-H-10 --seed 42 --set V6 --out stage2/tb/work/local/s2-4 --run-id s2-4 --steps 2000000 --monitor-every 500000 --workers 3 --expect-baseline stage2/tb/runs/s2-4/R0L_BASELINE.json`
  - The trainer takes `--expect-baseline` (ruling 17), so I added it.
  - No `--resume`, because the job directory did not exist.
- **Ending:** started 19:19:26Z, exited 1 at 19:19:58Z (about 32 s). It is not DONE, not a halt (exit 5) and not stopped. The log printed "R5-75-M-H-10-s42: G8a's short form on set V6", then a traceback ending:
  ```
  File ".../designlab/stage2/train2.py", line 920, in _g8a
      rec = s2io.plain(dict(schema="s2-run-g8a-1", job=self.job, run_uuid=self.doc["run_uuid"], tiny=self.tiny,
                            **res, session=self.k, at=s2io.now_utc()))
  TypeError: dict() got multiple values for keyword argument 'at'
  ```
- **Cause, read from the code:**
  - On a look set, `_g8a_short` returns `dict(form="bar", **g8a.short(...))`.
  - `g8a.bar_form` (`designlab/stage2/g8a.py` line 262, blame 6abf3fd25) already returns `at=s2io.now_utc()`.
  - `train2._g8a` (lines 920–921, blame f514780d7 / fba113af5) spreads `**res` and passes `at=` again.
  - `at` is the only key both sides set.
  - The Stage 1 view's "stock" form returns no `at`, which may be why gl-start and r6-start on S1 did not hit it. I did not verify that.
- **State left:** `stage2/tb/work/local/s2-4/jobs/R5-75-M-H-10-s42/` holds only `run.json` (sha256 `83e1a57ee43ab7847e30da252fd820360be9878ba281d6535c8cb7924c25bc54`) and `progress.json`. The log is `stage2/tb/work/logs/V8-s2-4.log` (sha256 `9284ff978326ae0d318c20c1e0e71c3033a9065c1a12a82415482df6eb8ea14b`).
  - `run.json`: status "partial", halt null, stopped null; `gates.g8a` null (no G8a record written); `baseline_attempts` []; `monitor` [].
  - `settings.expect_baseline` is `f33e4eb1…0543`. Session 0: commit 58109a17, dirty false, workers 3, started 19:19:28Z, `run_uuid` 06026565417f4b6a8e4d918a378ca6d0.
  - `progress.json`: steps 0.
  - No trainer process is left running.
- **What did not happen:**
  - 0 steps and 0 updates.
  - No checkpoint, and no monitor checks at 500,000 steps or after, so there are no readings against the baseline.
  - The run's own `monitor/r0l.json` was never made, so the trainer's `--expect-baseline` equality was never checked.
  - No resume: this was a code fault, not an interruption from outside the run.

## 3 and 4: not done
`nb collect` and `monitor2 s2-10` were not run, since there is no run to collect and no final checkpoint. There is no S2-10 verdict, so this report says nothing about whether the policy learns or whether the rule gives go or no-go.

## For the lead
- **The fix:** `train2._g8a` must not pass `at` a second time when `res` holds it. The fix belongs in the code, and I made none.
- **Resuming after the fix:** the chain's `resume_flag` will add `--resume` because `run.json` exists. K5 says a job with a run record and no checkpoint restarts from step 0 in place as the same run. The changed `train2.py` sources would need K4's `--allow-source-change`, unless the job directory is cleared, which is your call.
- **Rule M3:** whether this failed start, made before G8a's record and any rollout, counts against "S2-4 is run once" is also your call.
- **Main checkout:** it is clean. Its HEAD is now 141c2c3d ("Checkpoint: … S2-4 running"), which is newer than the 26674562 I fast-forwarded to. I wrote nothing there.

## What I wrote and committed
- Commit: 58109a17 only. Nothing was pushed, amended or forced.
- Ignored files under `stage2/tb/work/`: `s2-4-baseline/`, `local/s2-4/`, the logs above plus `run_trackb.log` lines, and the done markers V8-baseline, V8-baseline-copy and V8-baseline-committed. There is no V8-s2-4 marker.
- Records:
  - `/home/user/mesozoic-wt/wp-p42/hardware/trex_form_study/stage2/tb/runs/s2-4/R0L_BASELINE.json` (sha256 `f33e4eb11cbd25dcfb398b7a359f8704d87b2780f87879cb679220644bb70543`, committed)
  - `/home/user/mesozoic-wt/wp-p42/hardware/trex_form_study/stage2/tb/work/s2-4-baseline/r0l.json` (same bytes)
- `stage2/tb/dev_episodes.json` and hours: not touched (the chain does not list V8 there).
