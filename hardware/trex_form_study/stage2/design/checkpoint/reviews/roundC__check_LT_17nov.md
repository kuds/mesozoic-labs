VERDICT: merge after two small fixes in lookbar.py (F1, F2). Both are latent: no step of today's chain triggers them, and nothing runs on a look set until G9-ref has passed. If the lead merges now, F1 and F2 must be fixed before the first `lookbar run` or `episodes` on a look set. Everything else asked for is built, tested, and fails without the change. Tip to merge: bb1e0a71e13ce8255561fd3be5d50259ef1e6da4 (or the tip after the fixes).

FINDINGS (most serious first)

F1. Medium. The look-set placement fails open if this process already holds a bar context for the set from earlier, outside the context. File: designlab/stage2/lookbar.py:1012-1040 (look_set_rule, `_hold` at 1035); the release exists only on exit (`_let_go`, 976-993).
- What happens: `bar.context_for` caches one BarContext per process, keyed by (body, run_dir, inputs, crouch). Its BarBuilt conditions are cached per condition. Entering the context does not release that cache.
- When it bites: something earlier in the same process built the look set's turn or stop outside the context. Examples are `s2rows.equivalence_check`, a direct `lookbar.open_set`, or an R0 `bar.evaluate`, which the bar's own check lets run there. Then, inside the context, `bar.evaluate` with procs 1 (local BarWorker, then `context_for`) and `episodes` (`open_set`) replay the cached whole-window track. The record key holds the placed plan.
- Evidence: scratch script check_lt/leak_entry.py on the synthetic look set. Kinematic only, no episode.
  - Outside the context: built item 3 is ('ref','v0.4',2.0).
  - Inside: "same cached BarContext object inside as outside: True".
  - turn/q90_left: "record key's plan item: ('ref', 'v0.4', 0.6666666666666666) | leave in key's command: True | played (built) plan item: ('ref', 'v0.4', 2.0) | leave in played command: False", "played join ok: False jump_rad: 0.01732".
  - q90_right and the stop show the same: key 0.8333 windows, played 2.5.
- Fix: on entry, after `_hold(rule.placement)`, release `bar._CONTEXT["ctx"]` when its run_dir resolves to the placement's set_dir. This is the same code `_let_go` already has; put it in one helper used at entry and exit.
- Test: build the turn through `open_set` outside, enter the context, `open_set` again, and assert the built plan is the placed one and its join is ok.

F2. Low. Rows stamped inside a look set's context keep the stamp after it ends. File: lookbar.py:781-795 (`stamp`), used at 1242 (`episodes`) and from `_decide`.
- `LookRow.g9_ref` and `.placement` stay set. A later use of the same row objects therefore carries them:
  - On any set, the episode's block gets the old citation (`block["g9_ref"]`, line 554), so a V0 or T0 companion could cite V6's G9-ref.
  - The row's spec still carries the placement into later workers.
- Evidence:
  - leak_entry.py part 3 prints, after the context, the row's g9_ref {path, sha256} and placement {set V6, ...}.
  - The builder's own test_episodes_on_a_version_of_a_look_set_count_reading_1 reuses `rows` from V6.2 for T0, and the row still holds V6.2's citation at the end.
  - Today's callers (`lookbar.run`, monitor2) build fresh rows each time, so nothing is affected yet.
- Fix: the context remembers the rows it stamped and resets g9_ref and placement to their previous values on exit; add a test for it.

For the lead (agree with the builder's flags; none of these files is this package's):
- a. Ruling 21's "the bar's own record ... keeps its whole-window placement" is read as: the bar outside lookbar's context. A record made by `lookbar run` on a look set holds the placed plan, seconds and `command.leave`. This fits the job text, which said to replace the planner piece inside the context. The alternatives are:
  - a whole-window key while the episodes play J1, which would misdescribe the record;
  - no placement inside, which gives 0.42 rad jumps.

  The lead should confirm which reading holds.
- b. Some callers still use whole windows on a look set:
  - G9-ref part (a): `bar.refcheck`, called through `gates2._refcheck_record`;
  - `gates2.g8a`;
  - `s2rows.equivalence_check`;
  - `bar dry-run`, whose cost estimate overstates each turn and stop by about 20 s.

  Ruling 21 owes this "before G9-ref is run on any look set", so G9-ref (a) needs an owner before G9-ref runs. Options: a placement-only context exposed by lookbar, or a ruling.
- c. `LookRow.sources` omits r5lib.py, though the phase check calls `r5lib.leave_point`. This is minor, because the key carries the placed plan.
- d. The companion always has `g9_ref` now (None off the look sets). That is new output shape on V0 companions; their keys move with any lookbar edit anyway.
- Optional hardening: `leave_point` trusts the entry's `a_time_s`. It could also check it against the walk file's own `phase` array.

CHECKED AND FOUND RIGHT
- (A) env, r5lib.py 306-372 and 779:
  - m = (lead − a_phase)/K must be a whole number, 0 or more; the walk plays m + a_windows windows. The leave point is refused when it does not fit. Only causal entries with `look.window_a` and an `a_phase` are placed.
  - On the committed look sets the turn's and the stop's first output frame is their command phase (moves.py: window[0] = command for turn and stop), and offset = a_phase − c. So the move starts at window phase a_phase, which is time a_time_s, exactly where the walk ends. The walk enters at the window's frame 0, after the change (J2).
- (A) bar, lookbar.py 853-1009: `leave_resolve` is the bar's `_resolve` arithmetic with only the walk item changed.
  - `_placed_resolve` acts only for planners whose (set, body, set_dir) match, and only for turn and stop items with a leave point.
  - It is installed only when the planner piece is the bar's own, and put back on exit.
  - A spawned worker holds the placement through `from_spec`, before its lazy `built()`.
  - The worker's own check (track event t_ref == command ref_s) passes, per the builder's test.
- Bit-identity:
  - Old (working-branch) and new r5lib, same process, arrays read only from the main checkout. V0: "describe equal: True | tasks equal: True | scripts compared: 15 | segments equal: True | leave_rule in new describe: False". F25F9cl: the same line. S1: the same, with 4 scripts.
  - F25F9 is refused by both with the same text ("refs/nominal.npz is not listed..."). That is a data issue in the main checkout's set, not this package.
- The record-key hashes print the registered pair: 45719d03...528d c0027992...c1c1. bar.py and moves.py are unchanged.
- Committed look sets, using the wp-l worktree's arrays read only (its set JSONs are byte-equal to the worktree's). Joins after the change, with the old jump in brackets, all within the limit:
  - V6 stop: 0.00023 rad (0.376).
  - V6noF5 left turns, 0.35 and 0.5: 0.000158 (0.018).
  - V6noF5 right turns, 0.35 and 0.5: 0.000187 (0.421).
  - V6noF5 stop: 0.000282 (0.421).
  - The 60 cm and -X sets hold no turn or stop with a leave point.
- (B)(1), (3), (4), (5) and (C) work as the rulings say.
- Mutations, run on a scratch export (code and tests copied, data linked). The selected tests passed there unmutated first ("13 passed, 119 deselected in 35.62s"). Each mutation was caught by its intended test with the intended assertion: 12 of 12.
  - M1: env unplaced.
  - M2: bar unplaced.
  - M3: worker does not hold the placement.
  - M4: R0 let in on a look set.
  - M5: `collect` treats a version as no look set.
  - M6: the check is restored unconditionally.
  - M7: companion without the citation.
  - M8: reference read from the file list.
  - M9: `g9_ref_lines` loop emptied.
  - M10: `set_freeze` type not checked.
  - M11: gates2 without versions.
  - M12: `stamp` gives no citation.

  Each file was restored and its sha256 checked; the export is deleted.
- ruff on the 7 files: "All checks passed!" (run with --no-cache).
- Merge readiness:
  - `git merge-tree --write-tree` against the working branch (fe605133) is clean: tree 64e8cdb18c4e4a24c0868acea11cf25c0d5ef891. It adds only the branch's 3 review .md files.
  - The three-dot diff touches only the 7 package files.
  - All 6 commits not in the working branch end with the two attribution lines and name no model elsewhere: adb919a2, 9ebc87b6, 78dfbb73, 37e864e1, 0b2d2261, bb1e0a71. None is missing them.
  - `git status` is empty in the worktree and in the main checkout.

TEST SUMMARY LINES (worktree, head bb1e0a71)
- tests/test_stage2_lookbar.py tests/test_stage2_s2rows.py tests/test_stage2_gates2.py: "103 passed, 21 skipped in 221.35s (0:03:41)". Skips: 12 STAGE2_SLOW, 9 needing V0's or the look ladder's arrays.
- tests/test_stage2_r5lib.py tests/test_stage2_r5env.py: "66 passed, 2 skipped in 86.72s (0:01:26)". The env's tests live in test_stage2_r5lib.py; test_stage2_r5env.py is unchanged.
- With STAGE2_LOOK_SET_ROOT set to wp-l, `-k committed_look_sets`: "1 passed, 28 deselected in 16.56s".

WHAT TO BRING IN
bb1e0a71 by merge, after F1 and F2 are fixed in lookbar.py, with their tests. The commits are, in order: adb919a2 (merge), 9ebc87b6, 78dfbb73, 37e864e1, 0b2d2261, bb1e0a71 (merge).

NOT CHECKED
- STAGE2_SLOW tests.
- The 9 gates2 tests that need V0's or the look ladder's arrays. The builder ran them on an export; I did not.
- An end-to-end `bar.evaluate` with procs 1 or 2. The cache leak was shown through `open_set` / `context_for`, which BarWorker uses, and not through a full evaluation.
- monitor2, r2lfit, g8a and train2 tests.
- No episode was run on any look set, and no G9-ref.

Scripts and logs, under /tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/wave8/check_lt/:
- leak_entry.py
- v0_same.py
- mutate.py
- run2.log, run3.log, run4.log

Temp directories and the export are deleted.
