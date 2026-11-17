WP CT, the trainer's part of ruling 17 (17 Nov 2026), plus finding 3 of roundC__check_FU_17nov.md. Worktree /home/user/mesozoic-wt/wpc-ct, branch wt/wpc-ct. Simulation only. No panel, fit, training run or look-set episode was run; only the unit tests on stubs.

1. MERGES OF THE WORKING BRANCH
- 1c00b280: --no-ff merge of claude/mesozoic-trex-stage2-refine-4jffre at 3d261a79. No conflicts: the old head d1def040 was already an ancestor.
- 1ecd32d1: second --no-ff merge, at 9889291c. The working branch gained three commits while I built (rulings 21-27, the look sets package WP L, ruling 27). They touch none of the package's files and the merge was clean (`git merge-tree` exit 0). After it the record-key pair is still the registered one: 45719d030359ec7d5d3914af3814d837fdc05d77652ba45c64792b511027528d c0027992de21067525cc4489b74acb157feb89f3a4c7618580addc2b341be1c1.

2. PER ITEM
a) survival_credit (monitor2.py: `survival_credit`, new `CREDIT_RTOL` = 1e-9, rule text `RULES["survival"]`).
   - A clock that covered at least 1.05 x the survival time, to within 1e-9 relative, is now credited the survival time itself.
   - The reason for the tolerance: before, `min(1, ratio)` already gave exactly 1 when the clock covered the full time. The ruling only changes anything where a reference time summed step by step rounds a hair below 1.05 x the survival time.
   - Test: `test_r0ls_s_is_exactly_its_mean_survival`. It covers clocks summed per physics step, S == mean_survival_s exactly, a clock one step behind still credited its share, and the 1e-9 boundary.
   - The old product form fails it: "1 failed, 1 passed, 25 deselected in 0.37s".
b) The S1 stand-in is deleted (monitor2.py: `_stand_in` and `_episodes_by` removed; `_episodes` has no ModuleNotFoundError fallback; records use `EPISODES_BY` = "lookbar.episodes", the same value as before with lookbar merged; docstrings updated).
   - The two stand-in tests were replaced by `test_every_set_goes_through_lookbar_and_the_stand_in_is_gone`.
   - Against the earlier monitor2: "1 failed, 25 deselected in 0.28s".
c) `monitor2 baseline` without a job (monitor2.py `main`).
   - `--job` is now optional for `baseline`. Without it, `--out` names the directory, and a refusal's note goes there too.
   - With neither, it stops with an argparse error (exit 2). `run` still needs `--job`.
   - Test: `test_the_baseline_runs_without_a_job`; it also checks the `train2 monitor` entry. Against the earlier monitor2: "1 failed, 26 deselected in 1.11s".
d) `train2 run --expect-baseline PATH`, rule M3 (train2.py: `expected_baseline()`, `EXPECT_BASELINE_RULE`, `_Run.__init__`, `_asked`, `_new_doc`, `_begin`, new `_Run._expect_baseline` called right after `_baseline()` in `go()`, `train()`, CLI, `__all__`, module docstring).
   - After the baseline and before the first rollout, the run's own r0l.json is compared with the file by `monitor2.baseline_equal`.
   - The result is appended to run.json `baseline_expected`; the session records `expect_baseline` {path, sha256}.
   - Not equal, or digests not compared: Halt, exit 5, no step trained.
   - Refused before anything is written (exit 4): a missing, unreadable or non-baseline file, or the option given with `--no-gates`.
   - The file's sha256 is a setting (`settings.expect_baseline`), so rule K4 refuses a resume without it or with another file.
   - Tests: `test_a_baseline_that_is_not_the_committed_one_stops_the_run_before_its_first_rollout` and `test_an_expected_baseline_is_refused_or_kept_as_a_setting`. The committed file is made by `monitor2 baseline` without a job. The stub episodes now carry digests when `STUB_DIGESTS` is set. `test_the_cli_hands_its_arguments_to_train` was extended.
   - Against the earlier train2: "3 failed, 55 deselected in 3.07s". Six mutants (drop the call, drop the setting, drop the digest requirement, drop the no-gates refusal, drop the baseline_problem check, drop the halt): all caught.
e) No timeout on the monitor child. There was none, so there is no code change.
   - `CHILD_RULE` and `_child`'s docstring now say why (ruling 17).
   - Pin test: `test_the_trainer_waits_for_its_monitor_child_without_a_timeout`. It fails on the earlier train2 only through its rule-text assertion ("1 failed, 58 deselected in 0.59s"). The behaviour part passed before as well; a mutant that adds `wait(timeout=3600)` is caught ("1 failed, 58 deselected in 0.49s").
f) Finding 3 (the fallback in `allowed_sources`): a test only, the code unchanged.
   - `test_the_resume_takes_an_allowance_from_the_runs_record` now holds a session with `source_change` true and no `allow_source_change` key.
   - Mutant t6 is caught: "1 failed, 1 passed, 57 deselected in 10.18s". With the earlier test file it survives: "2 passed, 57 deselected in 9.93s".
- Not done: finding 2 of that check is in nb.py, which is the notebook's, not the trainer's.

FOR THE LEAD (choices I made; each with the alternative)
- Digests. Equality requires the state digests to be compared in both records, so a committed file without digests halts the run. Alternative: accept `baseline_equal`'s `equal` without digests. Recommend strict: the rule names the digest, and lookbar's real R0L episodes hold 64-character digests (its own test checks this).
- When the comparison runs. It runs at every session, not only the first; a resume appends another entry. Alternative: compare only before the first rollout.
- A setting of the run. The sha256 sits in settings under K4. As a result every new run's settings gain `expect_baseline` (null when not given). No existing record changes, and gates2's K3 compares named keys only. Alternative: record the file outside settings and allow a resume without it.
- Size of the tolerance. CREDIT_RTOL is 1e-9 relative (rounding only); a clock one 2 ms step behind is not forgiven. Alternative: an absolute tolerance of about one physics step.
- R0L exactness. R0L's S equals its mean survival exactly only if t_ref_reached is read at the instant survival_s is.
- Other modules that need a change (not mine to edit):
  - stage2/tb/run_trackb.sh (gates package): V8-s2-4 should pass `--expect-baseline "$BL"`. Its NOTE "the trainer takes no committed baseline" is now stale. V8-baseline may call `monitor2 baseline --body 75-M-H-10 --set V6 --out "$W/s2-4-baseline"`; the `python -c` form also works.
  - The rules draft's M3 Check could name run.json `baseline_expected` and `settings.expect_baseline`.
- Sources hash. monitor2.py and train2.py are in SOURCE_FILES, so `train2.sources()` changes with this commit (expected for any code change).

3. EXACT TEST SUMMARY LINES (final head 1ecd32d1, lowprio.sh, the standing environment, basetemp in scratch)
- tests/test_stage2_train2.py: "56 passed, 3 skipped, 2 warnings in 258.88s (0:04:18)". The 3 skipped are the STAGE2_SLOW tests.
- tests/test_stage2_monitor2.py: "27 passed in 36.18s".
- Also run at c8132d8b, before the second merge: tests/test_stage2_lookbar.py, test_stage2_s2rows.py, test_stage2_gates2.py and test_stage2_nb.py together gave "150 passed, 22 skipped in 248.85s (0:04:08)". These were not re-run after the second merge.
- ruff check --no-cache on the four files: "All checks passed!".
- Logs: /tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/wave7/ct/logs/ (train2_final.log, monitor2_final.log, others.log).

4. `git diff claude/mesozoic-trex-stage2-refine-4jffre --stat`
 .../trex_form_study/designlab/stage2/monitor2.py   | 167 ++++++-------------
 .../trex_form_study/designlab/stage2/train2.py     | 117 +++++++++++--
 .../trex_form_study/tests/test_stage2_monitor2.py  | 137 +++++++++++----
 .../trex_form_study/tests/test_stage2_train2.py    | 184 ++++++++++++++++++++-
 4 files changed, 435 insertions(+), 170 deletions(-)
- `git status --porcelain`: empty, in the worktree and in the main checkout.
- No new ignored file in the worktree: the __pycache__ and .ruff_cache folders date from 16 Nov. All outputs are under the scratchpad (outside the repo, so `git check-ignore` does not apply). bar.py is untouched; nothing was pushed.

5. COMMITS (oldest first)
- 1c00b280 merge of the working branch
- 3047d418 survival credit
- aadebd08 stand-in deleted
- fdc6bc27 baseline without a job
- b509c475 --expect-baseline
- 983f9671 no timeout (rule text and pin test)
- c8132d8b allowed_sources fallback test
- 1ecd32d1 second merge of the working branch

Every message ends with the two given lines and names no model elsewhere. Branch head: 1ecd32d121f2dbf7d8d4a993dc432238164601ad (wt/wpc-ct).
