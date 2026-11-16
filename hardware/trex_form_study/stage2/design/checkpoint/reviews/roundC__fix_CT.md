# fix:CT (roundC, wave 3): the agent's final report, 16 Nov 2026

Working paper. Nothing here is a rule.

WORK PACKAGE CT (the trainer): FIX REPORT
Worktree /home/user/mesozoic-wt/wpc-ct, branch wt/wpc-ct. Simulation tooling only; nothing here is a result. No panel, fit or training beyond the tiny smoke; no episode on a look set. Nothing pushed, no history rewritten.

One thing outside the worktree directory, for you to know: `git add` refused to run because of an empty `/home/user/mesozoic-labs/.git/worktrees/wpc-ct/index.lock` dated 17:20:05 UTC (this worktree's own git directory, not a tracked file). No process held it. I removed it. Its minute falls in trial t3 of the review's kill experiment, and `s2io.git_state` runs `git status`, which rewrites the index under that lock; the cause is inferred from that, not observed. See "beyond the review" below.

The working branch was an ancestor of HEAD when I started and moved while I worked (46a9c05: the look reading and the 18:21 rulings, none on CT). I merged it last (631e0a3, no conflict); every acceptance run below is on that merged tree.

1. PER FINDING
Each fix has a test that I saw fail with the fix taken out (mutation scripts under the scratch directory).

1. Later probe calls overwrite the validation record (MAJOR): fixed, defect confirmed by reading. `train2.probe_lr` and `filter_choice` add the call's bodies to the record they find.
   - Earlier bodies stay as they were; a body already in the record is refused; `rates_by_body` is taken over every head recorded; the filter's choice is kept.
   - A record of other settings, or one that does not hash to itself, is refused and not replaced.
   - A run that is not tiny is refused without `filter.json`'s choice unless `config` names `filter_hz` (`_need_filter` in `_Run.go`, before anything is written).
   - Tests: `test_a_later_probe_adds_its_heads_to_the_record`, `test_a_later_filter_measurement_keeps_the_choice`, `test_a_run_that_is_not_tiny_needs_the_filter`, and `test_probe_lr_stops_at_the_first_rate_that_passes` (extended).
2. Rule K8's probe builds R5 envs with default configs (MAJOR): fixed. `probe_lr(..., parent=, mode="R5", config=None)` uses the fine-tune's mode and env config fields; `_Run._parent_record` hands them on; the CLI gains `probe-lr --mode --config`. Rule O3's own probe refuses any mode but R5. Test: `test_rule_k8s_probe_runs_on_the_fine_tunes_own_envs`.
3. A monitor that refuses on moved sources counts like a crash: for the lead, code unchanged. The behaviour is as the reviewer says and is rule M1 to the letter.
   - (a) Keep it.
   - (b) monitor2 leaves with its own exit code for a refusal (sources, a checkpoint that is not intact, another run's checkpoint); the trainer halts with exit 5 and the attempt does not count toward M1's two.
   - (c) The trainer hashes its sources again before each check and halts without starting a child.
   - I recommend (b). It needs one sentence in M1 ("a monitor that refuses to run is no attempt").
4. `RESUME_RULE` order: fixed (settings, inputs, sources, versions, checkpoint, as the code and the design). Test: `test_the_resume_rule_names_the_order_the_code_refuses_in`.
5. `run.json` loses the resumed step from `checkpoints`: fixed in `_Run._rollback`. Test: `test_a_kill_between_a_checkpoints_last_file_and_run_json`.
6. A cut `monitor_<n>.json` or `r0l.json` fails every resume: fixed with `_read_record`, used in `check`, `_history` and `_baseline`.
   - The resumed step's cut record is moved aside and the check is run again.
   - A cut `r0l.json` is kept as `r0l.unreadable.s<k>.json` and made again.
   - A cut record of an earlier check still ends the session, now as a `TrainError` with its reason; no rule lets an earlier check run again.
   - Test: `test_a_record_that_was_cut_is_no_record`.
7. A monitor child outlives a trainer killed alone: fixed. `_Run._child` records the child's pid, start time and boot id in the attempt; `_Run._end_children` ends a live one at a resume before anything runs.
   - It acts only on an open attempt of an earlier session, on this host and boot, whose pid still has the recorded start time.
   - The child stays in the trainer's process group, so a group kill still takes it.
   - Tests: `test_a_monitor_child_that_outlived_its_trainer_is_ended_at_the_resume` and `test_a_resume_ends_only_a_live_child_of_its_own_run_on_this_host` (each of the four conditions fails it when removed).
   - Left open: a kill between the child's start and the write of its pid leaves an unrecorded child.
8. `survival_credit` fails open without clock times: fixed in `monitor2._checked(..., policy=)`. A checkpoint row's table (also at the reported delays) is refused when an episode that ended read, fell or unread lacks `clock.t_ref_reached`, `clock.t_ref_planned` or `planned_s`; nothing is written. R0L keeps its survival time. Test: `test_a_policy_rows_episode_without_its_clock_is_no_record`.
9. `progress.json` silent on a check without a record and on a halt: fixed in `_Run._progress`. New keys: `checks_without_record`, `halt`, `last_monitor.counted_without_record`. Tests: `test_progress_names_a_check_counted_without_a_record` and three extended tests.
10. Small points:
   - PYTHONHASHSEED: fixed. `require_hashseed()` is called in `train()` and in the three probes, called or from the command line (exit 4). Test: `test_the_trainer_refuses_without_pythonhashseed_0`.
   - Lost `r0l.json` replaced without comparison: fixed. `run.json` `baseline` now holds each episode's condition, seed, outcome and survival. A baseline made again is compared with them and the summary. Equal: it takes the old one's place and is noted under `baseline.remade`. Not equal: kept as `r0l.differs.s<k>.json`, nothing replaced, halt with exit 5. Test: `test_a_lost_baseline_made_again_must_be_the_runs`.
   - Commit e52b153's title: true (it holds `monitor2.py` only); not fixed, because history is not rewritten.

Beyond the review (commit bc7ca20): `train2`, `monitor2` and `clip2` set `GIT_OPTIONAL_LOCKS=0` at import unless the caller set it, and children inherit it. In a scratch repository I checked that `git status` then leaves the index alone and reads the same status. Test: `test_the_packages_processes_take_no_lock_on_the_checkouts_index`.

2. TEST SUMMARY LINES (merged tree 631e0a3, through `bash tools/lowprio.sh`, the standing env line)
- Hash line: `45719d030359ec7d5d3914af3814d837fdc05d77652ba45c64792b511027528d c0027992de21067525cc4489b74acb157feb89f3a4c7618580addc2b341be1c1` (unchanged).
- `tests/test_stage2_monitor2.py`: exit 0, `22 passed in 18.29s`.
- `tests/test_stage2_train2.py`: exit 0, `46 passed, 3 skipped, 2 warnings in 208.06s (0:03:28)`. Both warnings are SB3's batch-size note in the stub probe test.
- `STAGE2_SLOW=1 ... -k "tiny"`: exit 0, `5 passed, 44 deselected in 251.51s (0:04:11)`. That is the three slow S1 tests plus two fast tests with "tiny" in their names.
- `train2 run --mode R5 --body 75-M-H-10 --seed 42 --set S1 --tiny --workers 2 --out <scratch>/ct_fix/tiny`: exit 0, last line `R5-75-M-H-10-s42: done at 4096 steps`; `run.json` status done, 4096 steps.
- Its rates at load about 2.2, so provisional: 513 env steps/s first rollout, 495 first three; trainer 525 MB, workers 523 and 530 MB.
- The review's random group-kill experiment on the stub, rerun for 300 s with workers limited to 0-2: `SUMMARY: 19 trials, 24 kills that landed, 0 trials with problems`. No `index.lock` and no process was left afterwards.

3. DIFF AND STATUS
`git diff claude/mesozoic-trex-stage2-refine-4jffre --stat`:
 hardware/trex_form_study/designlab/stage2/clip2.py | 191 ++
 .../trex_form_study/designlab/stage2/monitor2.py | 857 ++++++++
 .../trex_form_study/designlab/stage2/train2.py | 2074 ++++++++++++++++++++
 .../trex_form_study/tests/test_stage2_monitor2.py | 720 +++++++
 .../trex_form_study/tests/test_stage2_train2.py | 1940 ++++++++++++++++++
 5 files changed, 5782 insertions(+)
`git status --porcelain`: empty, in the worktree and in the main checkout. The working branch's tip 46a9c05 is an ancestor of HEAD. All my outputs are under /tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/ct_fix/ (outside the repository). The new files a job can write (`r0l.unreadable.s<k>.json`, `r0l.differs.s<k>.json`) fall under `stage2/tb/.gitignore:1:work/` at the designed place.

4. COMMITS OF THIS ROUND, oldest first
0d0ad0c the review's points on a resume, the checks' children and progress (findings 4, 5, 6, 7, 9, 10)
f0c7e48 the probes add to their records; rule K8's probe runs on the fine-tune's envs (findings 1, 2, the probes' part of 10)
83d44b6 a policy row's episode without its clock times is no record (finding 8)
bc7ca20 a killed job leaves no index.lock in the checkout
c1a96a1 test of what a resume may end with SIGKILL
631e0a3 Merge the working branch: the look reading (WP CK) and the rulings after it
Head: 631e0a3. All 17 commits on the branch that are not in the working branch end with the two required lines and name no model elsewhere; none lacks them.

FOR THE LEAD
Readings I built; each could be read otherwise:
- Finding 1, the probe: rule O3's probe is also refused without the filter record on any set but S1, because O3 says it runs with O4's filter, measured first. The reviewer asked this only for runs. The other option is to let it run on the env's 10 Hz and record that.
- Finding 1, merging: a later call with other rates, sigmas, updates, set or `n_steps` is refused, not merged. Each body's entry carries its own stamp (`probed` / `measured`); the top-level versions and time are the last call's.
- Finding 8: I chose "refuse the table" over "flag the episode" or "credit nothing". Under M1, two such closed attempts count the check without a record. CW must therefore return clock times and `planned_s` for every policy episode that ends read, fell or unread.
- Finding 10, baseline: I chose "make again and compare, halt if different". The stricter reading of "once per run" is to halt at once on a lost baseline. On another machine or other library versions a remade baseline may differ in the last digits and halt, so CN should copy `monitor/` with `run.json`.
- Rule K8's probe is still made again when a fine-tune is cut before its first rollout (the record is replaced only if it is that parent's). "Run once" holds for a session that is not cut.
- The lock: `--no-optional-locks` in `s2io._git` would cover every other tool that reads the checkout's state. `s2io` is not this package's file, so I did not touch it.

Unchanged from the builder's list and still yours: `--expect-baseline` for rule M3, the S1 stand-in in monitor2, no timeout on the monitor child, "R0L exactly" to rounding.

For the next packages:
- CN: `progress.json` keeps its schema name with the three new keys; exit codes 4 and 5 as before.
- CG: a run that is not tiny now needs `filter.json`'s choice or `config filter_hz`, as it needs a rate.
- CW: see finding 8 above.

NOT DONE
- The full test suite and the look reading's tests (not this package's; by rule).
- Any probe on a real set; `-X` bodies; more than 2 workers.
- Finding 7 on the real env, and a trainer killed alone during the baseline end to end (the baseline case is held by the unit test on real processes only).
- `_validation` still raises on a validation record that cannot be read as JSON; not a finding, noticed in passing.
