REPORT: WP CW, ruling 21 placement + rulings 31 and 32 (D). Worktree /home/user/mesozoic-wt/wpc-cw, branch wt/wpc-cw, head bb1e0a71e13ce8255561fd3be5d50259ef1e6da4.

Everything asked is built and tested. The bar's own runner still places turns in whole windows, and G9-ref (a) and G8a still play that placement on a look set: see "For the lead" under (A). Nothing was pushed. No closed-loop episode ran on any of the study's look sets.

1. MERGES (no conflicts either time)
- adb919a2 merges the working branch at 32b5ecc5 (rulings 21-35, look sets V6 and V6noF5, the gates' G9-ref fixes).
- bb1e0a71 merges it again at e4b94077. The branch had moved during the work (the start's law, ruling 18; ruling 36). That merge touches none of the package's files, and the final test runs are on this head.

2. PER POINT

(A) Ruling 21: where a look set's turns and stop start

Env, in designlab/stage2/r5lib.py (new LEAVE_RULE, leave_point, place, Library.place):
- Library.__init__ places a task's plan before sizing "to the end". Before a causal move whose entry carries look.window_a and an a_phase, TURN_LEAD and STOP_LEAD are read as strides.
- The walk plays m + a_windows windows. m = (lead − a_phase)/K must be a whole number, 0 or more; it is 0 on today's sets.
- The leave point is refused when:
  - window_a names another walk;
  - its K or window_s is not the walk entry's;
  - a_windows × window_s is not a_time_s;
  - a_phase is not at the command's foot (a left command at a whole stride, a right one at a half);
  - the lead does not reach it by whole windows.
- Library.tasks[t]["leave"] and describe()["leave_rule"] appear only on such sets. Every other set keeps its plans item for item.

Tests are in tests/test_stage2_r5lib.py. I took that file as the env's test file; test_stage2_r5env.py is unchanged.
- make_look builds a synthetic look set from Stage 1's arrays: a 3-stride window with a 0.02 rad variation over the window.
- test_a_look_sets_turns_and_stop_leave_the_walk_at_j1s_leave_point covers turn_q90_left/right, stop and start_stop. Each join into the move is within the walk's own frame-to-frame step at every joint (jump < 1e-9 rad). The old whole-window plan jumps 0.0173 rad and is over the limit at 6 or more joints.
- test_on_the_75cm_rounds_sets_and_the_fixture_every_plan_is_as_before: on V0, F25F9 and F25F9cl's committed records, place returns every TASKS plan unchanged. T0's plans and description are unchanged.
- test_a_leave_point_that_does_not_fit_is_refused.
- test_the_committed_look_sets_turns_and_stop_join_within_their_walks_step needs the arrays, which are in neither this worktree nor the main checkout. I ran it with STAGE2_LOOK_SET_ROOT pointing (read only) at /home/user/mesozoic-wt/wp-l/hardware/trex_form_study/stage2/sets, whose arrays match the committed sha256s. Join jumps, after (before) the change:

| Set | Item | After (rad) | Before (rad) |
|---|---|---|---|
| V6noF5/75-M-H-10 | turns 0.35 and 0.5, left | 0.000158 | 0.018 |
| V6noF5/75-M-H-10 | turns 0.35 and 0.5, right | 0.000187 | 0.421 |
| V6noF5/75-M-H-10 | stop | 0.000282 | 0.421 |
| V6/75-M-H-10 | stop | 0.00023 | 0.376 |

All are within the limit after the change.

Without the change: the 4 mutations E1-E4 are each caught (E1 is today's unplaced library: 1 failed).

Bar side, in designlab/stage2/lookbar.py:
- How the bar places a turn: BarPlanner._resolve (bar.py, about lines 807-845) counts TURN_LEAD_STRIDES and STOP_LEAD_STRIDES as whole windows of the file. It runs both in the runner process (the record key's plan, seconds and command) and in every worker (BarContext.built).
- So the context replaces the planner piece BarPlanner._resolve, the same narrow, restoring way it replaces _check_rule:
  - It does so only while the rule applies to a look set.
  - It acts only for planners whose set.json set and body and whose set_dir match the context's.
  - It acts only for bar_turn and bar_stop items whose move carries a leave point. Everything else gets the bar's own function.
  - It puts the piece back at the end only where its own is the one installed; otherwise LookBarError.
  - On exit it releases the process's cached bar._CONTEXT if a placed condition was built there.
- leave_resolve is the bar's own turn and stop arithmetic with the walk item ("ref", W, m + a_windows) and a command.leave note. The phase check, leave_of, calls r5lib.leave_point, so there is one definition.
- Spawned workers (tune.Pool uses spawn) cannot see the parent's patch. So the rows are stamped at the check with the placement, LookRow.spec carries it, and LookRow.from_spec holds it for the worker's life (hold_for_life). In-process (procs 1) the context's own hold stands instead.

Tests are in tests/test_stage2_lookbar.py, on a synthetic set named V6 in tmp. It is kinematic: no episode.
- Placed inside the context; the bar's own placement outside it or where the rule does not apply.
- leave_resolve equals the bar's _resolve field for field when given the bar's own lead.
- Built tracks join within the limit, and are over it again after the context.
- A subprocess rebuilt from the stamped spec plans exactly the runner's placed plan; an unstamped spec plans the bar's own.

Without the change: mutations L1, L10, L11, L12 and L14 are each caught.

For the lead (A):
- a. "The bar's own record keeps its whole-window placement." I read this as: bar.py's runner outside lookbar's context keeps it. A record made by `lookbar run` inside the context holds the placed plan, seconds and command.leave, because it must describe the episodes it ran.
  - The alternative of a whole-window key while the episodes play J1 would misdescribe the record. Not built.
  - The alternative of no placement in the bar evaluation at all would leave the companion's episodes jumping 0.42 rad.
- b. These callers still play the bar's whole-window placement on a look set:
  - bar.refcheck, which is G9-ref part (a) through gates2._refcheck_record. G9-ref (a) on a look set would read its turns and stop on the jumped track.
  - gates2.g8a.
  - s2rows.equivalence_check.

  None of these is in my files. Options: they run inside a placement-only context that lookbar would expose; or a ruling that those readings are the bar's own (C7). r2lfit gets the placement if its evaluations run inside look_set_rule. The monitor (lookbar.episodes) gets it now.
- c. LookRow.sources does not include r5lib.py, although the phase check calls it. The key carries the placed plan. Adding r5lib would move every LookRow record key on any r5lib edit.
- d. An evaluation the rule covers is refused if one of its rows holds no LookRow, because such a row could carry neither the citation nor the placement. All five families' factories make LookRows.

(B) Ruling 31, in lookbar.py and gates2.py
- (1) LOOK_SET_ROWS now includes R5G/<seed>. On a look set or a version of one, any row outside the families is refused (LookSetRefused), R0@<ms> included:
  - inside the context, whoever calls the check (the context now installs itself for any look-set context);
  - in episodes, through LookSetRule.check.
  
  Outside the context the bar's own check is unchanged.
- (3) A version is a look set in episodes and in collect (look_set_of). gates2.look_set_of is new, and gates2.require_g9_ref covers versions. lookbar.require_g9_ref now calls the gates' function and keeps no copy; it turns Refused into LookSetRefused with the same text.
  - The word-for-word test stays: every refusal text is equal in both modules, on V6 and on V6.2. A spy checks that the call goes to the gates' function.
  - Consequence: g8a, rates, kill-resume and r6-start on V6.2 now need G9-ref too.
- (4) The context puts the bar's check back only if its own is installed, else raises LookBarError. Tested for contexts ended out of order and for a check replaced by another hand.
  - A set_freeze that is not an object is refused cleanly in gates2, and so in both modules: "names no freeze of the set".
  - Tests for a non-evaluate caller (a helper holding body and root locals), and for pass_ missing, None, 1 and "true".
- (5) Rows let in are stamped with the G9-ref citation (path, sha256). It travels in the spec into each episode's block; the key is added only when present, so V0's blocks are unchanged.
  - The companion record gets g9_ref (one citation, a list when there are several, None off the look sets), and so does the summary cell.

Without the change: mutations L2-L9, L13, L15, G2 and G3 are each caught.

The companion now always holds the key g9_ref. LookRow records' keys move with lookbar.py's hash, as with any lookbar edit. Plans, checks and judgments on V0, F25F9, F25F9cl and S1 are unchanged (tests above).

(C) Ruling 32 (D), in gates2.py
- New item_reference: a turn or stop item names the last move of its refcheck plan, in plan order, not of its sorted file list. The rule text z2_ref says so.
- Tests:
  - a pure test, including a move whose name sorts before start;
  - the V0 part (a) test, with a later-sorting file added to the item's file list;
  - a g9_ref_lines test on a synthetic record;
  - the check's suggested assertion added in test_g9_ref_passes_only_when_every_part_does.

Without the change: mutations G1 and G4 (the lines loop emptied) are each caught.

3. TEST SUMMARIES AND HASHES (head bb1e0a71, in the worktree)
- tests/test_stage2_lookbar.py tests/test_stage2_s2rows.py tests/test_stage2_gates2.py: "103 passed, 21 skipped in 212.11s (0:03:32)".
  - 12 skips are STAGE2_SLOW.
  - 9 are gates2 tests that need V0's or the look ladder's arrays, which this worktree does not hold.
- Env, tests/test_stage2_r5lib.py tests/test_stage2_r5env.py: "66 passed, 2 skipped in 88.28s (0:01:28)". The skips are the committed look sets' test (no arrays here) and the r5env slow test.
- With STAGE2_LOOK_SET_ROOT set, `-k committed_look_sets`: "1 passed, 28 deselected in 16.06s". Before the second merge the whole env run with the variable gave "67 passed, 1 skipped in 111.51s (0:01:51)".
- Scratch export of commit 0b2d2261 (code and tests copied; V0's and the ladder's arrays linked from the main checkout), tests/test_stage2_gates2.py: "2 failed, 44 passed, 1 skipped in 95.03s (0:01:35)".
  - Every V0 test, my new assertions included, passed there.
  - The 2 failures come from the export itself, in tests I did not change:
    - test_g9_ref_part_b_against_a_comparand: its cited path resolved through the symlink to the main checkout's absolute path.
    - test_every_output_path_of_the_chain_is_ignored_or_a_record: git check-ignore finds no repository in the export.
  - The second merge touched no gates2 file. The export is deleted.
- Mutations: gates2 4 of 4, env 4 of 4, lookbar 15 of 15 caught. Each file was restored and its sha256 checked.
- ruff on the 7 files: "All checks passed!".
- Hashes: 45719d030359ec7d5d3914af3814d837fdc05d77652ba45c64792b511027528d c0027992de21067525cc4489b74acb157feb89f3a4c7618580addc2b341be1c1 (the registered pair). bar.py and moves.py are not in the diff.

4. git diff claude/mesozoic-trex-stage2-refine-4jffre --stat
```
 .../trex_form_study/designlab/stage2/gates2.py     |  67 ++-
 .../trex_form_study/designlab/stage2/lookbar.py    | 503 +++++++++++++++-----
 hardware/trex_form_study/designlab/stage2/r5lib.py | 129 +++++-
 .../trex_form_study/tests/test_stage2_gates2.py    |  91 ++++
 .../trex_form_study/tests/test_stage2_lookbar.py   | 513 ++++++++++++++++++---
 .../trex_form_study/tests/test_stage2_r5lib.py     | 301 +++++++++++-
 .../trex_form_study/tests/test_stage2_s2rows.py    |   1 +
 7 files changed, 1392 insertions(+), 213 deletions(-)
```
`git status --porcelain` is empty in the worktree and in the main checkout. Removed: .ruff_cache (made by my ruff runs), pytest temp directories, the scratch set root and the export. Small scripts and logs remain under /tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/wave8/cw/.

5. COMMITS, oldest first (every message ends with the two given lines)
- adb919a2 (merge)
- 9ebc87b6 (gates2)
- 78dfbb73 (env)
- 37e864e1 (lookbar)
- 0b2d2261 (version test)
- bb1e0a71 (merge)

Head: bb1e0a71e13ce8255561fd3be5d50259ef1e6da4.

NOT DONE
- No STAGE2_SLOW tests.
- No end-to-end bar.evaluate with procs 2. The spawned-worker path is tested through a subprocess rebuilding the spec.
- monitor2, r2lfit, g8a and train2 tests were not run.
- No G9-ref run and no episode on any of the study's look sets.
