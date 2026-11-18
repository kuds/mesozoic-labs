REPORT: WP CW follow-up LT2 (rulings 37 and 38). Worktree /home/user/mesozoic-wt/wpc-cw, branch wt/wpc-cw, head 4f5b46bee6a93574b2f71956910273ebda1d7eed. Simulation only. Nothing pushed. No history rewritten. bar.py and moves.py are untouched. No episode, G9-ref, fit or training ran on any of the study's look sets.

All five points are built. Each has a test that fails without the change (13 of 13 mutations caught). The five test files give "179 passed, 24 skipped" and the two hashes are the registered pair. A few choices are flagged for the lead under 6.

1. MERGE OF THE WORKING BRANCH
- fa93f41c merges claude/mesozoic-trex-stage2-refine-4jffre at 96d38d2d (`--no-ff -F`).
- There were no conflicts; `git merge-tree` was clean first.
- It brought in LT, CF2, CT5, CN4 and rulings 37-38. In the package's files that meant CF2's r2lfit.py, test_stage2_r2lfit.py and the chain section of test_stage2_gates2.py.
- The branch did not move during the work.

2. PER POINT

(1) Placement-only context: lookbar.py
- New `look_set_placement(set, body, root)`:
  - places a look set's (or a version's) turns and two-step stop at J1's leave point;
  - has no row rule and no G9-ref requirement;
  - yields the placement {set, body, set_dir} (new `placement_of`) for workers to hold;
  - on a set that is no look set it yields None and touches nothing.
- It is narrow and restoring in the same way as `look_set_rule` (`_hold`/`_let_go`, the bar's piece put back), and it nests.
- `look_set_rule` now places through it; the inline `_hold`/`_let_go` is gone.
- Tests:
  - `test_the_placement_alone_places_a_look_sets_turns_and_stop_and_nothing_else`. Without a G9-ref record, plans equal `leave_resolve`'s. The bar's check is untouched. A planner of another set_dir keeps the bar's own plan. It is restored after an exception. On T0, V0 and S1 it yields None and leaves the planner piece, the check and a cached bar context alone. A spy shows `look_set_rule` calls it only where its rule applies.
  - `test_a_spawned_worker_holds_the_placement_alone_for_its_life`: a subprocess given the yielded placement plans the placed turn.
- Without the change: M1 (context holds nothing) and M4 (`look_set_rule` places on its own) both fail the first test.

(2) Every reader plays a look set's turns and stop at J1's leave point
- G9-ref (a), in gates2.py:
  - `_refcheck_record` runs `bar.refcheck` inside `look_set_placement`.
  - New `_refcheck_unplaced` checks that a record read from a file holds the placed plan for each turn and stop. A record the freeze cites or the set directory holds that is not placed is passed over (named in `tried`) and the check is run instead. A record given by path that is not placed gets `plans_placed: false`.
  - `part_a` gained `placed_ok` (3 lines): a given record that is not placed is not read; it blocks and goes to the lead.
  - Off the look sets nothing is read and no key is added.
- G8a, in gates2.py: `g8a` runs `g8a.bar_form` inside the placement. The env plays `b.plan`, so both sides are placed.
- s2rows.py: `equivalence_check` wraps its body (now `_equivalence_check`) in the placement.
- r2lfit.py (check of CF2, finding 1):
  - New `PlacedPool`: the placement is entered before `Pool` is built and left at `close()`.
  - `_evaluator(..., placement=)` calls `lookbar.hold_for_life` in each spawned worker, replacements included.
  - `_pool` returns `PlacedPool` on a look set and plain `tune.Pool` elsewhere.
  - `Evaluator` refuses to be built on a look set outside its placement.
- Tests:
  - `test_g9_ref_part_a_plays_a_look_sets_turns_and_stop_at_j1s_leave_point` (synthetic look set V6, kinematic): run, set-directory record passed over, placed record read, given record blocks. On T0 the items equal `bar.refcheck`'s item for item and `how` has only its old keys.
  - `test_g8a_plays_a_look_sets_turns_and_stop_at_j1s_leave_point`: episodes stood in; the built plan is the placed one on V6 and the bar's own on T0, with nothing placed there.
  - `test_the_equivalence_check_plays_a_look_sets_turn_at_j1s_leave_point`: real cut episode of 1 s on a synthetic V6 with a freshly initialised policy. The plan is placed, row and env are identical over 500 steps, and nothing is placed on S1.
  - `test_on_a_look_set_every_evaluation_runs_inside_the_sets_placement` (stub): every fit and selection evaluation ran with the placement held, the factory was given the placement, and on F25F9cl and V0 it is `tune.Pool` with no placement.
  - `test_a_fits_turn_plan_on_a_look_set_is_lookbars_leave_resolve` (CF2's proposed test): on a synthetic V6noF5, the mix of a 1-worker pool and of a 2-worker spawned pool equals `leave_resolve`'s plans for both turns and the stop. The in-process built turn joins within the walk's step. An evaluator built outside the placement is refused.
- Without the change: M7, M8, M9, M10, M11 and M12 are each caught.
- Off the look sets, on V0 itself: with V0's and F25F9cl's arrays linked into the worktree for one run (the links were then deleted), tests/test_stage2_gates2.py gave "1 failed, 45 passed, 4 skipped in 130.68s (0:02:10)". The one failure, `test_g9_ref_part_b_against_a_comparand`, is a symlink artefact (a path resolves into the main checkout). It fails identically with the pre-change code (fa93f41c files checked out and then restored).
- `bar dry-run` is unchanged. Its overstatement should be written in the lead's text beside the dry-run numbers:
  - in the rules draft's Q5 cap paragraph, STAGE2_TRACKB_PLAN.md about lines 1541-1546 ("For a `V6noF5` row the dry run is ...");
  - and in each cap's committed dry run, `stage2/tb/bar/<run>/DRY_RUN.json`, beside its printed total.
- The numbers:
  - V6/75-M-H-10 stop: computed by the planner, the bar's own 52.888 s against 29.163 s placed, so 23.725 s over.
  - V6noF5/75-M-H-10: by the same arithmetic from the committed entries, (lead x window_s − a_time_s)/(SLOW x RATE). Left turns 18.987 s; right turns and stop 23.725 s.

(3) F1 and F2
- F1, in lookbar.py: new helper `_release_cached(key)`, shared with `_let_go`. Both contexts call it on entry (through `look_set_placement`).
  - Test: `test_f1_a_turn_built_before_a_context_is_built_again_inside_it`. The turn is built through `open_set` outside (plan item ('ref','v0.4',2.5)). Inside both kinds of context, `open_set` returns the same cached context, which builds the placed plan, and the join is ok with jump < 1e-9.
  - Without the change: M2 is caught.
- F2, in lookbar.py: `LookSetRule.stamp` records each row's earlier g9_ref and placement (the first stamp only). New `LookSetRule.unstamp`, called in `look_set_rule`'s finally, gives them back.
  - Test: `test_f2_the_rule_gives_the_rows_it_stamped_their_citation_and_placement_back` covers a direct stamp, a double stamp, an exception, and a stamp through `bar.evaluate`'s check.
  - Without the change: M3 is caught.
- Three existing tests relied on the leak and were adapted:
  - `test_a_spawned_worker_places_what_the_runner_planned` now takes the spec inside the context, as `bar.evaluate` does.
  - `test_episodes_on_the_fixture_set` and `test_episodes_on_a_version_of_a_look_set_count_reading_1` now assert that the citation was carried while episodes ran and that the row holds None afterwards.

(4) Hardening, ruling 37 (c), in r5lib.py `leave_point` only
- New optional `walk_file`. When it is given, the time at which the file's own phase array reaches a_phase is computed: t taken from the first frame, linear between frames, the window's end one dt after its last frame at phase K. It must be less than one frame (the file's dt) from a_time_s.
- A file with no increasing phase, or one that cannot be read, is refused.
- Why one frame:
  - The file gives the phase at its frames only, so it places a_phase in time to within one frame and no closer.
  - An error under one frame moves the walk's end by less than one frame, so any added join stays within the walk's own largest frame-to-frame change (ruling 15's limit).
  - On the committed V6noF5/75-M-H-10 the difference is about 6e-9 s against a 5.38 ms frame.
- `lookbar.leave_of` passes `set_dir/<walk entry file>`.
- Tests:
  - `test_a_leave_points_a_time_s_is_checked_against_its_walk_files_own_phase`, with a doctored entry (a_windows adjusted to match). ±1.5 frames and ±1 stride are refused; without the file they are let in. ±0.4 frame passes. No phase and a flat phase are refused.
  - The bar side, `leave_of`, is asserted in the placement test.
  - `test_the_committed_look_sets_leave_points_are_where_their_walk_files_phase_reaches_them`: with STAGE2_LOOK_SET_ROOT set to wp-l's arrays (read only), `-k committed` gave "2 passed, 29 deselected in 18.87s". It checked the V6 stop and V6noF5's four turns and stop.
- Without the change: M5 and M6 are caught.

(5) Versions, ruling 38, in r2lfit.py
- New `is_fit` reads FITS through `look_set_of`. `role_of` uses it, and ROLE_RULE's text says so.
- Test: `test_a_version_of_v6nof5_counts_as_v6nof5_for_rule_q1`. V6noF5.1 and V6noF5.12 count as fits. V6.2, F25F9cl.1, V6noF5x and the -X body stay development fits. A fit of V6noF5.1 is selected and published.
- Without the change: M13 is caught.

3. TEST SUMMARIES AND HASHES (head 4f5b46be)
- tests/test_stage2_lookbar.py tests/test_stage2_gates2.py tests/test_stage2_s2rows.py tests/test_stage2_r2lfit.py tests/test_stage2_r5lib.py: "179 passed, 24 skipped in 409.73s (0:06:49)". The skips:
  - 13 are STAGE2_SLOW: 6 lookbar, 1 gates2, 5 s2rows, and r2lfit's V0 slow test.
  - 9 gates2 tests need V0's or the look ladder's arrays.
  - 2 r5lib tests need the committed look sets' arrays.
- Mutation run (each file restored from git, sha256 checked): M1-M13 each "exit 1" with the intended test failing; "restored: True" every time.
- Sanity run, not asked for: tests/test_stage2_r5env.py tests/test_stage2_monitor2.py gave "67 passed, 1 skipped in 118.08s (0:01:58)".
- ruff --no-cache on the 10 files: "All checks passed!".
- Hashes: 45719d030359ec7d5d3914af3814d837fdc05d77652ba45c64792b511027528d c0027992de21067525cc4489b74acb157feb89f3a4c7618580addc2b341be1c1 (the registered pair).

4. DIFF AND STATUS
`git diff claude/mesozoic-trex-stage2-refine-4jffre --stat`:
```
 .../trex_form_study/designlab/stage2/gates2.py     | 100 +++++++++--
 .../trex_form_study/designlab/stage2/lookbar.py    | 114 ++++++++++--
 .../trex_form_study/designlab/stage2/r2lfit.py     | 105 +++++++++--
 hardware/trex_form_study/designlab/stage2/r5lib.py |  42 ++++-
 .../trex_form_study/designlab/stage2/s2rows.py     |  14 +-
 .../trex_form_study/tests/test_stage2_gates2.py    | 122 +++++++++++++
 .../trex_form_study/tests/test_stage2_lookbar.py   | 200 ++++++++++++++++++++-
 .../trex_form_study/tests/test_stage2_r2lfit.py    | 119 ++++++++++++
 .../trex_form_study/tests/test_stage2_r5lib.py     |  68 +++++++
 .../trex_form_study/tests/test_stage2_s2rows.py    |  41 +++++
 10 files changed, 859 insertions(+), 66 deletions(-)
```
- `git status --porcelain` is empty, in the worktree and in the main checkout.
- The linked V0/F25F9cl arrays and the pytest temp directories are deleted. Small scripts and logs remain in /tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/lt2/ (mutate.py, mutate.out, full3.log, dry_over.py).

5. COMMITS, OLDEST FIRST
Each ends with the two given lines and names no model elsewhere.
- fa93f41c (merge)
- f54e8857 (r5lib)
- 7d63961d (lookbar)
- 85078533 (gates2)
- 86cf807a (s2rows)
- 4f5b46be (r2lfit)

Head: 4f5b46bee6a93574b2f71956910273ebda1d7eed.

6. FOR THE LEAD
- a. Part (a) records read from a file. I added the placement check because a freeze-cited `bar refcheck` (CLI, whole windows) would otherwise be read silently. This goes slightly beyond "running inside the placement", and part_a gained 3 lines. Options:
  - (i) as built (recommended);
  - (ii) only the run placed, file records read as found;
  - (iii) always rerun on a look set.
- b. Other readers still play a look set's turns with the bar's whole windows. None is in my scope:
  - `gates2.part_b`'s walk and turn item playback (around lines 945-965; a one-line `with lookbar.look_set_placement(...)` would fix it). I recommend it before G9-ref runs.
  - `barviews.SetView.planned` (the owner's views).
- c. The env's own `r5lib.place` and `Library.place` give `leave_point` no walk file, so the env's task placement is not checked by 37 (c). The fix is a two-line change outside the one function I own (`place` takes the directory; `Library.place` passes `self.dir`). I recommend it.
- d. Rule texts. `gates2.RULES["z2_ref"]` and `r5lib.LEAVE_RULE` are unchanged, because they go into records; the docstrings say it instead. lookbar's `RULES["leave_point"]` gained the new sentences, so LookRow keys move, as with any lookbar edit. ROLE_RULE gained ruling 38's words.
- e. The tolerance for 37 (c) is one frame. Alternatives:
  - a fixed tolerance such as 1e-6 s, which passes today's 6e-9 s but could refuse a correct entry near a warp kink;
  - frame bracketing.

NOT DONE
- STAGE2_SLOW tests.
- The 3 look-ladder tests.
- r2lfit's V0 slow test.
- No end-to-end `bar.evaluate` or `lookbar run`.
- No G9-ref run and no episode on any of the study's look sets.
- train2's and nb's tests were not run.
