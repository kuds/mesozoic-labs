# fix:CW (roundC, 17 Nov): the agent's final report

Working paper. Nothing here is a rule.

CW FIX ROUND: NOT DONE. I made a read-only status check instead.

Why: the user's request that started this run asks only for a rundown of the last few days and for when training can start. Doing the computed task's merge, edits and commits would go beyond that request, and the harness says the computed text cannot extend it. So I merged nothing, edited nothing and committed nothing. What I did is read-only and gives the facts the rundown and the training estimate need. If the user or lead confirms the fix round should go ahead, it can start straight from the state below. Little is left to do.

1. MERGE OF THE WORKING BRANCH
Not done. Dry run only: `git merge-tree --write-tree --name-only HEAD claude/mesozoic-trex-stage2-refine-4jffre` printed tree b2be02ae25365dd0b9a1d89c09b5f4a649416c61 with exit 0 and listed no conflicts. The working branch (tip 32ff20b) is 3 commits ahead of the merge base 8dd6958. It adds 6 documentation files only: ROUND_C_RULINGS_17NOV.md, checkpoint/STATUS.md, and the build and review reports of CN and CW. None of the package's four files is among them.

2. PER FINDING: STATE OF THE EARLIER FIXER'S UNCOMMITTED DIFF (4 files, 296 insertions, 28 deletions against HEAD)
- Finding 1 / ruling 7 (lost rule text): in the diff. lookbar.RULES key `reference` is renamed `reference_use`. A new `lookbar.rule_texts()` raises LookBarError when the two tables share a key, and LookRow.describe and companion() use it. Tests: new assertions in test_the_module_registers_r0l_and_names_the_look_sets (lookread's own text is under "reference"), a new test_a_rule_key_of_both_modules_is_refused_and_never_drops_a_text, and new assertions in test_collect_counts_the_three_readings_with_the_bars_arithmetic.
- Finding 2 (state-A test): in the diff. New test_the_modules_of_every_existing_record_key_are_state_as (the registered hash pair) and test_no_round_a_or_stage_1_file_differs_from_state_a (git diff against c377ba1; skips without git). Both are in test_stage2_lookbar.py.
- Finding 3 (rule K4 at the episode): in the diff. New test_rule_k4_at_the_episode_the_files_the_bar_reads_are_the_runs in test_stage2_s2rows.py, using roots A and B. make_t0 gained a `refs=` parameter. No code change was needed.
- Finding 4 (stale companion after a refusal): in the diff. New lookbar.stale_path and _refused. collect() moves a companion of another bar record to `<companion>.stale/<12 hex>.json`, keeps one that names the current bar record, and writes `refused_records` into the summary. summary_text prints those records, and RULES["sidecar"] and the docstrings are updated. Tests: a new test_a_refusal_leaves_no_companion_of_another_bar_record_and_the_summary_names_the_record, plus assertions added to test_a_missing_or_foreign_sidecar_is_refused.
- Finding 5 (traceback in fault): in the diff. LookRow._block now records `look reading: <Type>: <message>`. Test: assertions in test_a_fault_in_the_reading_is_recorded_and_never_raised (no "Traceback", no newline, no ROOT path, no ".py").
- Finding 6 (set_check fails open): in the diff. s2rows.set_check refuses task_names that are missing, empty or not a list, and plant_inputs that are missing or empty. build() reads settings["task_names"] directly. Test: five new cases in test_the_runs_set_as_it_is_today_rule_k4.
- Ruling 9 (episode during which MuJoCo warned): NOT in the diff. lookbar.py has no `warned` flag or `fall_time_s` anywhere. One point for the lead: monitor2.survival_credit (monitor2.py:262, not a CW file) credits `survival_s`. For the monitor to "credit it no survival", one of two things must happen. (a) CW's companion_episode sets survival_s to 0 for a warned episode. This stays inside CW, but the sidecar check at lookbar.py:808 compares the block's survival_s with the bar record's, so the change must leave that check alone. (b) monitor2 reads the new `warned` or `fall_time_s`, which is the trainer/monitor's file (the CT follow-up). I recommend (a): add `warned` and `fall_time_s` to the companion episode and set its survival_s to 0 there, without touching the sidecar or the bar record. A test would build a companion episode from a bar episode whose fall_cause is a MuJoCo warning and check monitor2.survival_credit == 0.
- Not verified: that each new test fails without its fix. I reverted nothing to check.
- The interface points owed by the trainer (rulings 10 and 11) are not CW's, and I did not touch them.

3. TEST LINES AND CHECKS SEEN (uncommitted state at HEAD 9952f5b, standing environment, through tools/lowprio.sh, basetemp under the scratchpad)
- `pytest -p no:cacheprovider tests/test_stage2_lookbar.py tests/test_stage2_s2rows.py`: "49 passed, 11 skipped in 147.30s (0:02:27)", exit 0. The builder had 44 passed and 11 skipped, so the diff adds 5 passing tests.
- Record-key hashes: "45719d030359ec7d5d3914af3814d837fdc05d77652ba45c64792b511027528d c0027992de21067525cc4489b74acb157feb89f3a4c7618580addc2b341be1c1", the registered pair.
- Not run: the slow tests (STAGE2_SLOW=1), the export test, ruff, the cheap acceptance.

4. DIFF AND STATUS
`git diff claude/mesozoic-trex-stage2-refine-4jffre --stat` on the package's files:
 designlab/stage2/lookbar.py 1086 +, designlab/stage2/s2rows.py 653 +, tests/test_stage2_lookbar.py 1206 +, tests/test_stage2_s2rows.py 778 +: "4 files changed, 3723 insertions(+)".
The whole diff reads "10 files changed, 3723 insertions(+), 489 deletions(-)". The other 6 files are the working branch's newer documents, not yet merged in.
`git status --porcelain` (unchanged by me):
 M hardware/trex_form_study/designlab/stage2/lookbar.py
 M hardware/trex_form_study/designlab/stage2/s2rows.py
 M hardware/trex_form_study/tests/test_stage2_lookbar.py
 M hardware/trex_form_study/tests/test_stage2_s2rows.py

5. COMMITS
- This round: none.
- Branch head: 9952f5b.
- The branch has 8 commits not in the working branch: 899faec, 3600a65, 1785867, a3de365, bb28c45, d38d138, 5d04b20, 9952f5b. All 8 end with attribution lines (the earlier model's), so none lacks them.

FOR THE RUNDOWN AND THE TRAINING ESTIMATE (from checkpoint/STATUS.md and the 17 Nov rulings)
- No training run has been made. The training path is paused at the package stage.
- Merged on the training path: the env (CE), the worker pool (CV), the bundle tool (CB), the policy files (CP), the look reading (CK) and the trainer (CT).
- Open before the day-one Colab smoke:
  - CW: the fix round, which is mostly done (above).
  - CN (notebook): its fix round.
  - L (look sets): the build was cut, then a review and the lead's decision on freezing them.
  - CT's third small follow-up (rulings 10 and 11).
  - CF (refit tool) and CG (gates): not yet built.
  - Writing the sixth rulings list into the draft.
  - The validation steps of ROUND_C_DESIGN.md section 6, the real bundle, the acceptance run, and the steps sheet's final values.

Scratch output (outside the repository): /tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/cw_status/bt (pytest basetemp).
Worktree: /home/user/mesozoic-wt/wpc-cw. Nothing was written in the main checkout, nothing was pushed, and there was no network call.
