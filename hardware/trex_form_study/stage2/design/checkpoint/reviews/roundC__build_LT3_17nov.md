REPORT: WP CW, LT3 (ruling 39). Worktree /home/user/mesozoic-wt/wpc-cw, branch wt/wpc-cw, head 8caa90c179082b95ffd3aa4818b4825b2806f271. Simulation only. Nothing pushed, no history rewritten, nothing amended or forced. bar.py and moves.py are untouched. No panel, fit, training, G9-ref or episode ran on any of the study's look sets; everything ran on synthetic sets in the pytest temp dirs. All five points are built, and each has a test that fails without it (11 mutations, all caught, each file restored with its sha256 checked).

1. MERGE OF THE WORKING BRANCH
- db2407f6 merges claude/mesozoic-trex-stage2-refine-4jffre at 46bf03f7 (`--no-ff -F`). There were no conflicts. It brought in the LT2 merge 56b01e11, the build and check reports, ruling 39 and the plan's cap-rule text.
- The working branch has since moved to 4a9d2aad ("Servo model: the lead's third list of rulings"). That commit adds only stage2/design/servo_model/LEAD_RULINGS.md. `git merge-tree --write-tree HEAD <branch>` is clean (exit 0). I did not merge it again.

2. PER POINT
(1) G9-ref part (b), gates2.part_b
- Change: the walk and turn item block (it builds the BarContext, plays each item, then releases) now runs inside `lookbar.look_set_placement(v.set, v.body, v.dir.parent.parent)`. Off the look sets the context does nothing. The docstring says so; RULES are unchanged.
- Test: `test_g9_ref_part_b_plays_a_look_sets_turn_items_at_j1s_leave_point`.
  - On synthetic V6, a spy on `lookread.window_ref` shows every item played with the set's placement held. The turn plans are `leave_resolve`'s; the walk is the bar's own.
  - Each turn's window_ref, steps, lift-offs and four numbers equal a placed play and differ from the whole-window one.
  - On T0 nothing is placed, and every item equals the old loop (window, steps, lift-offs, numbers). Nothing stays placed.
- Without the change: B1 (the `with` removed) fails.

(2) The owner's views, barviews.SetView.planned
- Change: `planned` takes its plan from a new `SetView.resolve`.
  - On a look set or a version of one (new `LOOK_SETS`/`look_set_of`, copies of lookbar's), the planner resolves inside `lookbar.look_set_placement`.
  - A set.json that names another set or body is refused (ViewError).
- Problem found: `lookbar` calls `tbenv.assign_gl("disable")` when it is imported, so a rendering process (clips, MUJOCO_GL=osmesa) cannot import it. I checked: RuntimeError.
  - So in such a process (`_lookbar_here()` False) the plan is made by a child process with `tbenv.child_env("disable")` (`_resolve_in_child`/`_placed_child`) and read back as exact JSON.
  - Off the look sets the plan is the bar's own, as before.
- Tests:
  - `test_the_look_sets_are_lookbars`.
  - `test_a_look_sets_planned_turns_and_stop_leave_the_walk_at_j1s_leave_point`: on synthetic V6, the plan, seconds and command (with `leave`), t_c and t_end of both turns and the stop are `leave_resolve`'s; `planned_turn` carries the placed plan; on T0 the bar's plans, with nothing placed while they are resolved.
  - `test_in_a_process_that_renders_a_look_sets_turn_is_planned_inside_the_placement`: an osmesa child process gets exactly `leave_resolve`'s plan, command and seconds, keeps osmesa and never imports lookbar.
- Without the change: V1 (no placement) fails 2 tests; V2 (lookbar always used in process) fails the osmesa test; V3 (the child resolves unplaced) fails the osmesa test.

(3) Placed refcheck command, gates2
- Command: `python -m designlab.stage2.gates2 refcheck --body B --set S [--out DIR] [--set-root DIR]`.
  - New `gates2.refcheck` runs `bar.refcheck` inside `look_set_placement` and lets bar.refcheck write its own record at `<out>/<set>/<body>/refcheck.json`, in its own format. Without --out nothing is written, as with gates2's --record.
  - On a look set whose set.json names another set or body it is refused before anything runs (exit 2).
  - New `refcheck_unplaced` applies part (a)'s own `_refcheck_unplaced` using the set's records alone (no plant).
  - Output: "refcheck S B: pass|FAIL", the record path and sha256, and a placement line. Exit 0 pass, 1 FAIL, 2 error or an unplaced turn or stop item.
- Test: `test_the_placed_refcheck_gives_a_look_sets_freeze_a_record_g9_ref_part_a_reads`.
  - On V6 `--out <root>` writes the set directory's refcheck.json with bar's schema and keys, placed turn and stop plans, and the walk item identical to the bar's own.
  - `sets.refcheck_state` reads it as current. With a freeze citing it, part (a) reads it as "cited by the set's latest freeze" with plans_placed True. A cited whole-window record is passed over.
  - On T0 the record equals `bar.refcheck`'s, item for item and in every key but `at` and `provenance`.
  - A "V6.2" whose set.json names V6 is refused, and no record is written.
- Without the change: C1 (run outside the placement) and C2 (no name check) both fail.
- One line for the look sets' freeze instructions in sets.py:
  "On a look set or a version of one (V6, V6noF5, V6.2, ...) the record is made with `python -m designlab.stage2.gates2 refcheck --set S --body B --out <the sets' root>` (bar refcheck's record, made inside lookbar.look_set_placement: its turns and stop at join rule J1's leave point, the only kind G9-ref part (a) reads); on every other set with `python -m designlab.stage2.bar refcheck --set S --body B --out <the sets' root>`."
  - Places in sets.py that name bar refcheck today: docstring lines 62-63, the REFCHECK comment at 145, the verify note at about 2478, `refcheck_lines` at about 2662, and the freeze's error text at about 2768.

(4) r5lib.place and Library.place
- Change:
  - `place(plan, refs_doc, tr_doc, set_dir)` now takes the set's directory (required) and gives `leave_point` `walk_file = set_dir / <entry file or refs/<walk>.npz>`.
  - `Library.place` passes `self.dir`.
  - The old test caller (V0, F25F9, F25F9cl plans as before) passes the directory. No file is read where no move has a leave point.
- Test: `test_the_envs_own_placement_checks_a_time_s_against_the_walk_file`.
  - Doctored entry: on make_look's set, a frozen left-turn entry has a_time_s +1.5 frames and a_windows changed to match. `leave_point` without the file lets it in.
  - `r5lib.place`, `r5lib.Library` and `r5env.R5Env` on that set all refuse it ("has a_time_s ... the walk file's own phase array reaches a_phase 2 at ...").
  - The undoctored set builds, and the env builds on it.
- Without the change: R1 (walk_file=None) fails with "DID NOT RAISE".

(5) needs_v0 test: `test_off_the_look_sets_part_a_and_g8a_on_v0_are_as_before`
- Part (a) on V0 reads the freeze-cited record with only the old keys in `how`.
- Part (a) run: a set root of the test holds V0's records with the freeze's citation dropped and its arrays linked. Part (a)'s record equals `bar.refcheck`'s, item for item and in every key but at and provenance; `tried` is empty.
- G8a (episodes stood in) builds all 10 of Z1's conditions with the bar's own plan, seconds and command, and the same track segments with the sha256 of their joint arrays, with nothing placed.
- It skips without V0's arrays. With V0's arrays linked into the worktree for one run, it gave "1 passed, 52 deselected in 43.09s". Two artificial mutations were caught: an extra `how` key off the look sets (V0_p) and a cut G8a (V0_g).
- Same linked run of the whole gates2 and barviews files: "1 failed, 77 passed, 7 skipped in 230.94s (0:03:50)". The one failure is `test_g9_ref_part_b_against_a_comparand`, the symlink artefact LT2's report names. It fails the same way with the gates2.py of the merge commit db2407f6 ("1 failed, 52 deselected in 9.47s"). The links were then deleted.

FOR THE LEAD
- a. `place` now requires `set_dir` instead of taking it as an option, so no caller can skip the check. The only callers are Library.place and tests.
- b. The `leave_point` docstring in r5lib.py still says "None: not checked (the env's place gives no file)". The function is outside my scope, so I left it; it is now stale.
- c. lookbar's `look_set_placement` docstring lists the readers that run inside it. Part (b), the views and gates2 refcheck could be added; that is lookbar.py and not mine.
- d. `gates2.played_files` (G9-ref's input list) still resolves outside the placement. The placed plan loads the same files, so the list is the same. Not my scope, not changed.
- e. The refcheck record carries no "placed" key; placement shows only in its turn and stop plans, which is what part (a) reads. I kept it this way so the record stays exactly bar refcheck's format.
- f. In a rendering process, every condition of a look set costs one child process, a few seconds each.
- g. Source hashes move because the files changed: r5lib.py (in s2rows' SOURCE_FILES and the Track B row sources, as with LT2's edit), barviews.py (clip2's hash) and gates2.py. The record-key hashes do not move.

3. TESTS AND HASHES (head 8caa90c1, no V0 links)
- tests/test_stage2_gates2.py tests/test_stage2_r5lib.py tests/test_stage2_barviews.py tests/test_stage2_lookbar.py: "150 passed, 27 skipped in 242.78s (0:04:02)".
  - The skips: 7 gates2 tests need V0's arrays, 3 need the look ladder's, 1 slow; 2 r5lib tests need the look sets' arrays; 6 barviews tests need V0's arrays and 2 are slow; 6 lookbar tests are STAGE2_SLOW.
- ruff --no-cache on the 6 changed files: "All checks passed!".
- Hashes: 45719d030359ec7d5d3914af3814d837fdc05d77652ba45c64792b511027528d c0027992de21067525cc4489b74acb157feb89f3a4c7618580addc2b341be1c1 (the registered pair).

4. DIFF AND STATUS
`git diff claude/mesozoic-trex-stage2-refine-4jffre --stat` (the package's files; the only other line is servo_model/LEAD_RULINGS.md, -14, from the branch moving on):
 .../trex_form_study/designlab/stage2/barviews.py   |  98 ++++++++-
 .../trex_form_study/designlab/stage2/gates2.py     | 132 +++++++++---
 hardware/trex_form_study/designlab/stage2/r5lib.py |  18 +-
 .../trex_form_study/tests/test_stage2_barviews.py  | 117 +++++++++++
 .../trex_form_study/tests/test_stage2_gates2.py    | 234 +++++++++++++++++++++
 .../trex_form_study/tests/test_stage2_r5lib.py     |  58 ++++-
 6 files changed, 624 insertions(+), 33 deletions(-)
- `git status --porcelain` is empty, in the worktree and in the main checkout.
- The V0 links and the pytest temp directories are deleted. Mutation script and log are in /tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/lt3/ (mutate.py, mut/, full.log, msg*.txt).

5. COMMITS, OLDEST FIRST
Each ends with exactly the two given lines, and none names a model elsewhere.
- db2407f6 (merge)
- fe498731 (part (b))
- 0d0559a9 (r5lib)
- d3b52619 (refcheck command)
- ea2c5585 (barviews)
- c7188060 (V0 test)
- 8caa90c1 (ruff import order)

Head: 8caa90c179082b95ffd3aa4818b4825b2806f271.

NOT DONE
- STAGE2_SLOW tests.
- The look-ladder tests and the committed-look-set r5lib tests: those arrays are not here.
- The barviews V0 tests ran only in the linked run.
- No real clip was rendered.
- No end-to-end G9-ref on a look set.
- sets.py was not edited (its text is given above).
