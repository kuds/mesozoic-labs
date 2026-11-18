S3 fix round report. All seven points of the lead's fifth list are done, each with a test that fails without it. Worktree /home/user/mesozoic-wt/wpc-s1, branch wt/wpc-s1. This is simulation only: no panel and no gate-1 record was made, and no episode ran on V0, F25F9, F25F9cl or a look set. All test episodes ran on the synthetic set T0 or the S1 view, under tmp dirs. servo.py was not edited.

**Merges**
- At the start I merged the working branch's bf815adf (S3 reports and the fifth list) as fb6db281 (--no-ff -F, no conflict).
- While I worked the branch moved to 37e0695b: six commits (look-set rebuilds, P42, a checkpoint note), none touching this package's files. `git merge-tree` showed no conflict, so I merged again as 9af9ceab and reran every test, check and the dry run below on that tree.

**Per point** (gate1.py and tests/test_stage2_gate1.py unless named)
1. **Repeat verdict.** `summarize` now uses `REPEAT_VERDICT`: "replicated" is True, "not replicated" is False, anything else is None (pending), so condition 2 waits.
   - Test: `test_a_repeat_not_run_or_not_judged_is_pending` covers repeat not run, not judged, not replicated, replicated, and a headline that did not pass.
   - Mutant (the old `word == "replicated"`): `1 failed, 18 deselected in 1.44s`.
2. **Placement, listing and cap.**
   - (a) `check_rows` refuses SV2 rows on a set that `bar.registered_name` does not know (a scratch or fixture set), in both `run` and `dry_run`, before anything is read or written.
     - Test: `test_sv2_rows_are_refused_on_a_scratch_or_fixture_set`, plus the dry-run test.
     - Mutant: `2 failed, 17 deselected in 1.69s`.
   - (b) Every run on V0, F25F9 or F25F9cl (or a version of one) is listed in the runner's own file, `stage2/tb/gate1_dev_runs.json` (`gate1.DEV_RUNS`, `--dev-runs`).
     - Each entry holds the command, set, body, block, rows, settings, delays, conditions, seeds, count, cap and session. It is written before any episode runs (state started) and again when the records are written (state done).
     - dev_episodes.json is not touched.
     - Test: `test_the_cap_and_the_development_listing`. Mutant (no listing): `1 failed, 18 deselected in 4.93s`.
   - (c) `--max-episodes` is required on `run`. A plan whose episodes to start exceed it is refused before anything is written; `dry_run` states each block against a cap if given.
     - Mutant (no cap): `1 failed, 18 deselected in 3.37s`, same test.
3. **Supply window.** New `gate1.counts`: a V- or W1 run that did not fall and left the supply window is not counted as stayed up or as meeting every limit. It is counted as `left_supply_window`, in both readings and in the summary ("N, of them M not up"); `by_the_bar` keeps the bar's numbers. No verdict moves.
   - Tests: `test_a_v_minus_or_w1_run_that_leaves_the_supply_window_is_not_counted_as_up`, `test_the_summary_names_runs_that_left_the_supply_window`.
   - Mutant: `2 failed, 17 deselected in 1.27s`.
4. **`--cut-s`.** Its help now begins "for tests only" and says a cut episode is never judged. `GateWorker` runs a cut condition's episode through `cut_episode`: the served row's episode only, with no judge, no success and no error, so the next run of the same key keeps it.
   - Tests: `test_a_cut_walk_episode_is_never_judged_and_never_run_again`, `test_the_runs_options_cut_s_for_tests_only_and_a_required_cap`.
   - Mutants: judge called again `1 failed, 18 deselected in 3.34s`; help text reverted `1 failed, 18 deselected in 1.20s`.
   - The test stub `cut_episodes` is removed.
5. **Worker source check.** `GateWorker` refuses to start (`tune.PoolError`) when `runner_sources()` (servo.py and gate1.py) differs from the hashes taken at session start.
   - Tests: `test_the_worker_refuses_to_start_when_servo_or_gate1_changed_during_the_session`, `test_a_change_to_servo_or_gate1_during_a_session_stops_the_run`.
   - Mutant: `2 failed, 17 deselected in 3.53s`.
6. **Equivalence check (s2rows.py).** With the servo option on, the env now gets `options.servo="today"` instead of the episode seed. The result's `servo` holds option, given, setting and reason.
   - Test: `test_the_equivalence_check_gives_an_unserved_rows_env_the_setting_today` (tests/test_stage2_s2rows.py) asserts `identical`.
   - With s2rows.py at fb6db281 the test fails: `1 failed, 22 deselected in 13.92s` (KeyError 'given').
   - Probe with only the env options reverted to the old episode seed: `servo train/3043 identical False first_difference True`. With the fix: `servo today identical True first_difference False`.
7. **Second block.** `plan` never selects an SV0 or SV2 D record for the second block (`NO_REPEAT`), and the summary's repeat word is "no repeat (...)". The record itself is reported as before.
   - Tests: `test_the_second_block_takes_a_verdict_rows_drawn_panel_whose_headline_reading_passed`, and the dry-run and summary tests.
   - Mutants: plan `2 failed, 17 deselected in 5.38s`; repeat word `2 failed, 17 deselected in 4.27s`.

All mutants were run on 4374c05e (the second merge touches none of these files). Each file was restored with `git checkout` and checked by sha256 and a clean status.

**For the lead**
- a) `stage2/tb/gate1_dev_runs.json` is not git-ignored. The first real run on a registered set will leave it untracked there for someone to commit. I created none.
- b) Readings I made:
  - The cap counts the episodes a run would start, not records already finished.
  - The dry run also refuses SV2 off the registered sets.
  - S1 is refused for SV2 too. The bar exempts S1, but gate1 cannot run on S1 anyway.
  - Every cut condition, not only walks, is unjudged. gate1 judges only gate 1's walk, and a cut record is read by no rule.
- c) Point 6 applies to any servo option (draw, a named setting, D/<s>), as the ruling's "an unserved row's env" reads. The test covers draw. s2rows.py is in `train2.SOURCE_FILES`, so `train2.sources()` moves again; `tests/test_stage2_train2.py::test_sources` gave `1 passed in 2.02s`.
- d) The gate1 changes are one commit (all points sit in one file), not one commit per point.
- e) With point 7 the V0 dry run's second block is now 0 records.

**Outputs (merged tree 9af9ceab)**
- `tests/test_stage2_gate1.py`: `19 passed in 22.76s`
- `tests/test_stage2_servo.py`: `63 passed in 3.54s`
- `tests/test_stage2_s2rows.py`: `18 passed, 5 skipped in 115.34s (0:01:55)` (the same skips as before)
- `tests/test_stage2_g8s.py`: `27 passed in 20.07s`
- regress check, exit 0: `stage2/regress/regress.json: current (full tier, 256 of 256 episode runs matched, 248 distinct committed episodes, the four source hashes are today's)`
- Hashes: `45719d030359ec7d5d3914af3814d837fdc05d77652ba45c64792b511027528d c0027992de21067525cc4489b74acb157feb89f3a4c7618580addc2b341be1c1`
- TABLE_SHA256: `342f3948cef116ea7fde49106817596a3d69ffa8d85c84dcb7c35f442546ef57`
- ruff: `All checks passed!`
- V0 dry run (cost only, writes nothing): `gate1 dry-run --set V0 --rows SV0,SV2/42 --settings @named,D --delays 4,10 --root <main checkout>/stage2/sets --max-episodes 1280`, exit 0.
  - `Planned: 64 records, 1280 episodes, 7.519 simulated hours, 0.950 core-hours (1.046 with the layer's estimate).`
  - `Second block, only for a drawn-panel record whose headline reading passes (P41): up to 0 records, 0 episodes, 0.000 core-hours (0.000 with the layer's estimate).`
  - `Episodes a \`run --block seeds3042\` would start: 1280; cap 1280: within it.`
  - `Episodes a \`run --block seeds5042\` would start: 0; cap 1280: within it.`
  - Full output: /tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/servo_s3fix/dryrun_V0.md

**`git diff claude/mesozoic-trex-stage2-refine-4jffre --stat`**
```
 hardware/trex_form_study/designlab/stage2/g8s.py   | 117 +--
 hardware/trex_form_study/designlab/stage2/gate1.py | 998 +++++++++++++++++++++
 hardware/trex_form_study/designlab/stage2/r5env.py |  14 +-
 .../trex_form_study/designlab/stage2/s2rows.py     |  21 +-
 hardware/trex_form_study/designlab/stage2/servo.py | 261 +++++-
 .../design/servo_model/SERVO_MODEL_DESIGN.md       |  32 +-
 hardware/trex_form_study/tests/test_stage2_g8s.py  |  30 +-
 .../trex_form_study/tests/test_stage2_gate1.py     | 584 ++++++++++++
 .../trex_form_study/tests/test_stage2_r5env.py     |  58 +-
 .../trex_form_study/tests/test_stage2_s2rows.py    |  27 +-
 .../trex_form_study/tests/test_stage2_servo.py     | 133 +++
 11 files changed, 2164 insertions(+), 111 deletions(-)
```
`git status --porcelain` is empty in the worktree and in the main checkout. I removed the .ruff_cache I created and every pytest temp dir.

**Commits, oldest first** (none pushed; each ends with the two required lines)
- fb6db281 merge of the working branch (bf815adf)
- 01812950 s2rows: an unserved row's env gets the setting today
- 4374c05e gate1: points 1-5 and 7, with tests
- 9af9ceab merge of the working branch (37e0695b)

Branch head: 9af9ceab22351c8796b8c97fc2db5b342569ef04

**Scratch files** (in /tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/servo_s3fix/):
- mut/run_mut.py (the mutant driver) and mut/results.txt (its results)
- mut/probe6.py (the point 6 probe)
- regress2.txt (the last regress check)
