Verdict: sound; go: not decided

VERDICT: sound. The step stopped where the standing rules say it must. The baseline was made, copied, cmp-equal and committed before the run. The trainer then crashed with exit 1 on a fault in the code (train2._g8a), before G8a's record and before any training rollout. No code was changed. Steps 3 (collect) and 4 (S2-10) correctly did not run. Every claim in the report that I could check matches the records and my own tool output. S2-4 has not produced a go/no-go: the records hold no S2-10, so go = "not decided". This is a simulation result only.

FINDINGS (my numbers beside the report's)

1. Baseline: matches, equal, deterministic.
 - sha256: HEAD blob, working file and stage2/tb/work/s2-4-baseline/r0l.json are all f33e4eb11cbd25dcfb398b7a359f8704d87b2780f87879cb679220644bb70543, and `cmp` says equal. This matches the report.
 - I re-derived from the committed record, both by hand and with monitor2.summarize:
   - outcomes: read 13, fell 2, unread 0, error 0, not_run 3 (turn/q90_left 6007-6009).
   - N 6; S 25.03222222221989; L 25.557704023051375; L-S 0.5254818 (< ROUTE_CLOSED_S 1.0, so M3b's route applies).
   - N_but_A1_look 6, N_reading2 6, N_reading3 6; mean_clock_factor 0.9999999999998832.
   - up_by_condition: walk/nominal 3/3, walk/v0.4 3/3, stop/two_step 3/3, nudge/left 3/3, nudge/forward 1/3 (6013 fell at 24.786 s, 6015 at 24.710 s), turn/q90_left 0/3.
   - failures_by_term: line 1, look_toe_off 3, recovered 3, survived 2, toe_clearance_own_swing 3.
   - All 15 episodes that ran hold a state_digest, and the 3 not-run episodes hold none.
   - valid True, row R0L@10ms, provenance commit 26674562 with dirty False, wall_s 58.44.
   - Every value matches the record and the report.
 - By the rule, a go on the final checkpoint would need N >= 8, or S >= 26.032 s, or (M3b) N_but_A1_look >= 8.
 - I re-made the baseline into scratch with monitor2.baseline (exit 0, 53 s). baseline_equal(re-made, committed) gave equal=True, 18 episodes, no differences, digests_compared=True. All 18 digests are identical. Only the wall_s fields differ.

2. Commit 58109a17882115dec420105a1b1cb85c7b1703b9.
 - It adds only stage2/tb/runs/s2-4/R0L_BASELINE.json (1922 lines).
 - Its message ends with exactly the two required lines.
 - It was committed at 19:19:00Z, before the run started at 19:19:26Z. run.json sessions[0].commit is 58109a17 with dirty False, so M3's ancestry holds.
 - The chain's `committed` check line is in run_trackb.log.
 - The operator passed `-c core.hooksPath=/dev/null`. This was not asked for, but it did nothing: core.hooksPath is unset and the hooks directory holds only sample hooks.
 - No .py file changed in this commit or in 141c2c3d. Bit-identity prints the two required hashes now.

3. The run.
 - Log stage2/tb/work/logs/V8-s2-4.log (sha256 9284ff978326ae0d318c20c1e0e71c3033a9065c1a12a82415482df6eb8ea14b). It shows the exact command with --expect-baseline (the trainer takes it: ruling 17, train2 line 2217), start 19:19:26Z, "TypeError: dict() got multiple values for keyword argument 'at'" at train2.py line 920 in _g8a, then "exit 1" at 19:19:58Z.
 - I confirmed the cause in the code:
   - g8a.bar_form returns at=s2io.now_utc() (line 262, blame 6abf3fd25).
   - _g8a_short returns dict(form="bar", **doc) on a look set.
   - _g8a spreads **res and passes at= again (lines 920-921, blame f514780d7/fba113af5).
   - `at` is the only key both sides set. The S1 "stock" branch returns no `at`.
   - kill-resume (gates2 line 149) and r6-start (line 2139) run with gates=False, and there is no g8a.json under stage2/tb/work. So this path was never exercised on a look set before.
 - run.json (sha256 83e1a57ee43ab7847e30da252fd820360be9878ba281d6535c8cb7924c25bc54):
   - status partial; halt null; stopped null; gates.g8a/rate/g8b all null; baseline null.
   - baseline_attempts []; monitor []; checkpoints []; num_timesteps 0; budget_asked 2000000.
   - settings.monitor_every 500000, settings.expect_baseline f33e4eb1...0543, settings.gates True, settings.tiny False.
   - sessions[0]: workers 3, vcpus 4, pythonhashseed "0", and error recording the TypeError.
 - progress.json (sha256 c559f3c4511630d7cbb94197d9a175286e05dcba01b8fbc502f46ec0b088655e) records steps 0.
 - No train2 process is running.
 - Learning rate and filter were never reached in this run, so they are not in its settings. train2._validation('75-M-H-10','V6') gives rate 1e-05 (lr_probe.json) and filter_hz 10.0 (filter.json), as the validation records say. rates.json has pass_ True and provisional True; kill_resume.json has pass_ True and equal True.
 - The run ended in none of DONE, halt or stopped. It was a crash: 0 steps, 0 updates, no monitor check. The trainer's own monitor/r0l.json equality check was never reached.

4. S2-10.
 - I ran monitor2.write_s2_10 into scratch on a copy of the job directory with the committed baseline. It refused with "MonitorError ...: the run has no check".
 - There is no verdict to re-derive, and none is claimed.

5. Merge readiness.
 - wt/wp-p42 (58109a17) and the working branch's tip (now 141c2c3d, a STATUS.md checkpoint made at 19:18:15Z) have diverged from 26674562. A fast-forward is not possible.
 - `git merge-tree` is clean (exit 0, tree 1cacb9b2), because the two commits touch disjoint files.
 - Nothing was pushed.

6. Small additions to the report (no errors in it).
 - The run's own G8a short form (walk/v0.4, seed 3042, delays 0 and 5) ran on V6 before the crash, since res was computed. Its result was lost with no record. This bears on the lead's question of whether the failed start counts against M3's "run once".
 - The report says the resume would need --allow-source-change after a code fix. That follows K4/K5 as written, but the lead should decide it together with the fix.

MY OWN DEVIATION (for the lead)
My check re-ran R0L's 18 zero-action episodes on V6 into scratch at 19:22:40-19:23:33Z, which is after S2-4's session 0 started. Under X1's paragraph "From S2-4's start on ...", that episode set is no record. I could not list it in stage2/tb/dev_episodes.json because I may not write tracked files. The lead may want to list it, labelled. I changed no tracked file and made no commit. git status in the worktree is clean apart from the ignored work/ directory.

Check files in /tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/s24/check/:
- derive.py and derive.out (re-derivation and the S2-10 try)
- rebaseline.sh and rebaseline.log (the re-made baseline)
- rebaseline/r0l.json, sha256 c94dc9fe1f59fb15abd23fe3ac8393d526360ec029145cc038b3ab6228ba1088; its bytes differ from the committed file only by wall times, provenance and at
- jobcopy/

NOT CHECKED
- The load average before the run (0.18-0.36).
- The pre-run bit-identity print, and the fast-forward from dae9c3fd (I only see that HEAD descends from 26674562).
- I did not reproduce the TypeError by running the trainer or G8a again. I confirmed the cause from the code and the log only.
- Background check cadence: there was nothing to poll, since the run crashed in 32 s.
