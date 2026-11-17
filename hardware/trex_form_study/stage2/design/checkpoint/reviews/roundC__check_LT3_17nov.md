VERDICT: merge as it stands. All five points of ruling 39 are built, and in my own mutation run 14 of 15 mutations were caught. The one not caught is a test gap in a guard branch, finding 1. Findings 1 and 2 are low and can go to the lead after the merge. Tip to merge: 8caa90c179082b95ffd3aa4818b4825b2806f271 (wt/wpc-cw), with a merge commit. The working branch is now at 4a9d2aad, which adds only servo_model/LEAD_RULINGS.md, so it cannot fast-forward.

FINDINGS (most serious first)

1. Low: a test gap, and a stale record left behind.
   - The `gates2 refcheck` CLI has a branch that returns exit 2 when a look set's turn or stop item is not placed. No test reaches it: my mutation C3, which replaces `return EXIT_ERROR` with `pass`, was not caught.
   - In that branch, `--out` has already written the record to `<out>/<set>/<body>/refcheck.json`. When `--out` is the sets' root, that is the set directory's file, and `sets freeze` would cite it.
   - G9-ref part (a) would still pass the record over and run its own placed check, so the gate fails closed.
   - The branch can only be reached if the placement places nothing, and the new set/body name check prevents that.
   - Optional fix: unlink the record (or make it some other way) when this check refuses, and add a test with a placement that does nothing.

2. Low, for the lead. I found this by reading the code; I did not run it.
   - `_SetView` refuses a set.json that names another set, but not one that names another body.
   - In that case `look_set_placement` places nothing (a planner is matched by set.json's set and body). Then part (b) would play whole windows without saying so. LT2's `_refcheck_unplaced` would also compute its "placed" plan unplaced and accept a whole-window record, and G8a is affected the same way.
   - The new refcheck command and the views refuse this case.
   - It cannot happen today: all 9 set.json files in this worktree and in wp-l name their own set and body.
   - Fix: add the body to `_SetView`'s check. That is gates2, but outside this round's owned parts.

3. Information.
   - The builder's for-the-lead items b, c and d are correct. The `leave_point` docstring in r5lib.py (about line 324) still says "None: not checked (the env's place gives no file)".
   - The builder's five places in sets.py that name `bar refcheck` (63, 145, 2478, 2662, 2768) are right. There is a sixth: the `--refcheck` help at sets.py 2906 ("the record of bar refcheck to cite").
   - The builder's freeze line is accurate. My probe showed the record lands at `sets.REFCHECK` in the set directory, and `sets.refcheck_state` reads it as current. Where the sets' root is not the default, `--set-root` must be passed as well, the same as for `bar refcheck`.
   - Suggested single line for sets.py: "On a look set or a version of one (V6, V6noF5, V6.2, ...) the record is made with `python -m designlab.stage2.gates2 refcheck --set S --body B --out <the sets' root>` (bar refcheck's record made inside lookbar.look_set_placement, its turns and stop at join rule J1's leave point, the only kind G9-ref part (a) reads); on any other set with `python -m designlab.stage2.bar refcheck --set S --body B --out <the sets' root>`."
   - In barviews, the plan is made by a child process whenever MUJOCO_GL is not a disabled value and lookbar is not imported. That includes an unset MUJOCO_GL. The result is correct but slower.

CHECKED AND FOUND RIGHT

My probe is `probe.py`: kinematic only, synthetic V6, a V6.2 whose set.json names itself, a misnamed V6.3, and T0. V0, F25F9 and F25F9cl arrays were read from the main checkout through a scratch farm. The old code compared against is the working branch's own gates2/barviews/r5lib.
- Synthetic sets: 50 PASS, 0 FAIL. Off the look sets: 25 PASS, 0 FAIL, plus all off-look-set part (b) and views checks PASS.
- Part (b):
  - Every V6 item was played with V6's placement held.
  - The turn plans are `leave_resolve`'s. The walk ends at J1's leave point (0.667 and 0.833 windows, taken from `r5lib.leave_point` with the walk file), the command is at that point and carries `leave`.
  - q90_left's window_ref is [16.631, 19.781]. This matches the "inside the placement" numbers from LT2's check (old: [20.178, 23.328]).
  - The walk items, walks, rungs and comparison are the same as the old code's. V6.2 is placed.
  - Nothing stays placed, including after an exception raised inside the context.
  - On T0, V0 and F25F9cl the whole output equals the old part_b's, and the built plans and commands are equal with nothing placed.
- Views (`SetView.planned`):
  - On V6, all three moves (both turns and the stop) have `leave_resolve`'s plan, seconds and command. t_c sits at the leave point.
  - The child-process path gives a plan identical to the in-process one, including the trace arrays.
  - V6.2 is placed; V6.3 is refused with ViewError.
  - On T0 (4 conditions), V0 (9), F25F9 (9) and F25F9cl (9) the old and new `planned` are identical bit for bit: res, trace, pose, t_c, t_end, freeze and files.
- Placed refcheck:
  - On V6 it gives bar's schema and keys with placed turn and stop plans. Every other item equals bar.refcheck's.
  - Part (a) reads the record both when the freeze cites it (`plans_placed` True, ok == pass_) and as the set directory's refcheck.json.
  - `refcheck_unplaced` returns [] on the placed record and is not empty on the bar's own.
  - V6.2 is placed. V6.3 gives exit 2 and nothing is written. Nothing stays placed after an exception.
  - On T0, V0 and F25F9cl, `gates2 refcheck` equals `bar refcheck` (CLI against CLI) item for item and in every key but at and provenance, with equal exit codes. On V0 and F25F9cl the items also equal the freeze-cited records'.
- r5lib:
  - A doctored a_time_s of +1.5 or -1.2 frames is refused by the new Library ("walk file's own phase"); the old Library built it. +0.5 frame builds in both.
  - Library tasks equal the old code's on S1, V0, F25F9cl and T0. F25F9 is refused identically by both.
  - Library.place reads no walk file on these sets (I put a spy on np.load).
- My mutations ran on a scratch farm (copies, so the worktree was never written), and each file was restored with its sha256 checked:
  - Caught: B1 (part b unplaced), B2 (wrong root), V1–V5 (views never placed, `_resolve_placed` unplaced, child unplaced, lookbar always used here, versions not treated as look sets), C1 (refcheck unplaced), C2 (no name check), C4 (record not written), R1 (walk_file=None), P1 (extra `how` key on V0), P2 (G8a placing V6 while on V0), P3 (G8a with a condition cut).
  - Not caught: C3 (finding 1).
- Record-key hashes: `45719d030359ec7d5d3914af3814d837fdc05d77652ba45c64792b511027528d c0027992de21067525cc4489b74acb157feb89f3a4c7618580addc2b341be1c1`, the registered pair. bar.py and moves.py are unchanged.
- ruff --no-cache on the 6 files: "All checks passed!".
- Only the package's 6 files differ from the working branch.
- No committed Track B run or gate record exists, so the moved r5lib/barviews/gates2 source hashes touch nothing committed.
- `git status --porcelain` is empty in the worktree and in the main checkout.
- Merge readiness:
  - `git merge-tree --write-tree HEAD claude/mesozoic-trex-stage2-refine-4jffre` exits 0 with tree 0fd1ee24b8bd1192d1c2148ec2daa021f20de511. It differs from the branch tip only in the 6 files.
  - The merge db2407f6's tree equals 46bf03f7's.
  - All 7 commits (db2407f6, fe498731, 0d0559a9, d3b52619, ea2c5585, c7188060, 8caa90c1) end with exactly the two attribution lines, and none names a model elsewhere.

TEST SUMMARY LINES (mine)
- Worktree at 8caa90c1, the four files: "150 passed, 27 skipped in 247.45s (0:04:07)".
- Farm controls:
  - part (b): "1 passed, 52 deselected in 13.57s"
  - refcheck: "1 passed, 52 deselected in 23.34s"
  - V0 test: "1 passed, 52 deselected in 35.98s"
  - views: "3 passed, 29 deselected in 11.67s"
  - r5lib: "2 passed, 30 deselected in 2.16s"
- Farm with V0's files copied in, the 12 V0-dependent gates2 and barviews tests: "12 passed, 73 deselected in 81.57s (0:01:21)".
- Farm full gates2 + barviews with V0 linked: "4 failed, 67 passed, 7 skipped, 7 errors in 202.08s (0:03:22)". All of these come from the farm itself:
  - 2 path assertions caused by the symlinks; both pass with copied files.
  - 7 kill-resume errors: the `environments` package is not on the farm's path.
  - gl_start and the chain-paths test: the farm is not a git checkout.
  - All of these passed in the worktree run.

NOT CHECKED
- STAGE2_SLOW tests, the look-ladder tests, and the two committed-look-set r5lib tests (no arrays here; I did not touch wp-l's).
- No real clip was rendered, apart from the osmesa child test inside the suite.
- R5Env on V0 (I compared Library only).
- Part (b) on S1: the changed block sits inside `if v.set != STOCK_SET`.
- End-to-end G9-ref on any look set.
- The builder's timing claim (f) and the builder's own mutation runs (I ran my own instead).

Scripts and logs are in /tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/lt3/check/: probe.py, mutate_check.py, farm.py, farm_f25.py, full.log, control.out, mutate.out, farmfull.log. The farm, the synthetic sets and the basetemps are deleted.
