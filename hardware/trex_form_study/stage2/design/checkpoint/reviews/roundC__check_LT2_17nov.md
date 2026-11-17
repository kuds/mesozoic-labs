VERDICT: merge as it stands. I found no defect in the package's own scope. All five points are built as rulings 37 and 38 say, and each has a test that fails without it: 17 of 17 of my own mutations were caught. Three things outside this package's files go to the lead (findings 1 to 3). Tip to merge: 4f5b46bee6a93574b2f71956910273ebda1d7eed, with a merge commit.

FINDINGS (most serious first)

1. Medium, for the lead (not this package's file part). G9-ref part (b) still plays a look set's turn items with the bar's whole windows.
   - Where: gates2.py about 947-971 (`part_b`, "every walk and turn item: its window's playback"). It builds `bar.BarContext` outside any placement.
   - Evidence (my probe partb_probe.py, kinematic, synthetic V6):
     - As built: q90_left plays ('ref','v0.4',2.0) with window_ref [20.178, 23.328].
     - Inside `look_set_placement`: ('ref','v0.4',0.667) with window_ref [16.631, 19.781].
     - On this synthetic set the four look numbers came out equal. I did not test real sets.
   - Ruling 37 (b) says every reader that plays a look set's turns does so at J1's leave point. Part (b) is such a reader, but the ruling's list does not name it.
   - Fix: wrap that block in `with lookbar.look_set_placement(v.set, v.body, v.dir.parent.parent):`. It needs an owner before G9-ref runs on a look set. I agree with the builder's 6b, which also names `barviews.SetView.planned`.

2. Low, for the lead. A look set's freeze-time refcheck is made outside the placement.
   - sets.py (lines 62-63 and 145) has it made with `python -m designlab.stage2.bar refcheck`, outside any context, so on a look set it is whole-window. The freeze report reads it (sets.py about 2475-2484).
   - G9-ref (a) now passes such a record over and runs its own placed check, which fails closed. But the freeze notes still show whole-window turn and stop results.
   - Options: a placed refcheck command (in lookbar or gates2; bar.py is not edited), or a note in the freeze text.

3. Low, for the lead (builder's 6c, agreed). The env's own task placement is not checked by ruling 37 (c).
   - `r5lib.place` and `Library.place` call `leave_point` without `walk_file`.
   - The bar side is checked. If an entry's a_time_s were wrong, `bar.refcheck` inside the placement would raise (it resolves every usable condition without a try), so G9-ref stops: fail-closed, but as an exception rather than a recorded block.
   - The fix is two lines outside this package.

4. Low, scope. gates2 goes slightly past "running inside the placement": the new `_refcheck_unplaced` and 3 lines in `part_a`.
   - Without them, the freeze-cited whole-window record (finding 2) would be read silently, and ruling 37 (b) would have no effect for part (a).
   - The behaviour fails closed. I agree with the builder's option (i).

5. Low, test gap.
   - The committed off-look-set comparison for part (a) and G8a uses synthetic T0, not V0 or S1, because V0's arrays are not in the worktree.
   - I covered V0 myself (see below). An optional `needs_v0` test could be added.

6. Information only. What moves off the look sets:
   - The LookRow key `row_sha256` moves, because lookbar's RULES text changed. On S1, `equivalence_check` old against new differs only in `at` and `row_sha256`.
   - No lookbar record is committed: `git grep s2-look-1 -- stage2` finds only design texts and make_briefs.py.
   - ROLE_RULE's new text goes into the rules of new R2L records; no fit record exists.

`bar dry-run`: the builder's places for the overstatement match the plan.
- STAGE2_TRACKB_PLAN.md lines 1541-1546, the cap paragraph that begins "For a `V6noF5` row the dry run is".
- Each cap's `stage2/tb/bar/<run>/DRY_RUN.json`, which the plan names at about line 1580.
- I could not reproduce the builder's 18.987 s and 23.725 s:
  - On this branch's committed V6noF5/75-M-H-10 JSON, the planner reports every turn and stop missing (start: "no transition within the gates").
  - The wp-l arrays are being rewritten by the running rebuild.

CHECKED AND FOUND RIGHT
- Point (1), `look_set_placement`:
  - It places with no row rule and no G9-ref requirement.
  - It yields `{set, body, set_dir}`, and a spawned worker holds that for life.
  - Off a look set it yields None and touches nothing: no planner piece, no `_check_rule`, no cached context.
  - It is restored after an exception. `look_set_rule` places through it.
- F1: `_release_cached` is shared by the entry and `_let_go`.
- F2: `stamp` and `unstamp` give rows back their earlier values, including after an exception and a double stamp. The companion takes g9_ref from the episode blocks, so the reset does not lose it.
- Point (4): r5lib changes only `leave_point`. LEAVE_RULE and gates2's `RULES["z2_ref"]` are unchanged. The one-frame tolerance is defensible.
- Point (5): `is_fit`/`role_of` agree with `lookbar.look_set_of` and `gates2.look_set_of`, which are identical.
- My probe on synthetic V6 and V6noF5 (probe_new.py, no physics): 55 PASS, 0 FAIL.
  - G9-ref (a): `_refcheck_record`'s `bar.refcheck` built only placed turn and stop tracks, never whole windows.
  - G8a: the row's built track, and the env's own script from `b.plan` (segments equal to the row's), are placed.
  - `equivalence_check`: placed, including after the turn was first built outside through `open_set`; after an exception nothing stays placed, and `open_set` after the context builds the bar's own plan again.
  - r2lfit: the in-process evaluator, a spawned worker running `r2lfit._evaluator` with the placement, and a 2-worker pool's mix all equal `leave_resolve`. A worker without the placement is refused (R2LError).
  - Every placed join was ok, with jumps of 3.5e-18 to 6.2e-16 rad against a walk step of up to 0.0313.
  - Outside the context the joins fail: whole windows ('ref','v0.4',2.0 or 2.5), jump 0.0173 rad, not ok.
  - Nothing is placed or replaced after any of these.
  - On V0, S1, F25F9cl and F25F9 the context yields None. All 39 of V0's conditions resolve as the bar's own inside it.
- Off the look sets, old against new: the working branch's five files in a symlink farm, V0 arrays read from the main checkout (offsets.py).
  - Part (a): the freeze-cited record is the same, and a forced `bar.refcheck` run gives the same items and the same hashes of all 14 tracks.
  - G8a on V0 (episodes stood in): the same built plans and tracks for all 10 conditions.
  - The R2L evaluator on V0: plain `Pool`, the same mix and the same built tracks.
  - `equivalence_check` on S1 (walk/brisk and turn/walking_0.35_left, 1 s): the same digests and comparison (identical, 500 steps), apart from `at` and `row_sha256` (finding 6).
- Mutations (mutate_check.py): run on a scratch farm with the test files copied, because the tests put their resolved ROOT first on sys.path, so linked tests would import the worktree.
  - Control: lookbar 7 passed, gates2 2, s2rows 1, r2lfit 3, r5lib 1.
  - All 17 were caught by the intended tests, and each file was restored (sha256 checked):
    - no release on entry
    - unstamp a no-op
    - the placement holds nothing
    - the rule placing inline
    - `leave_of` without the walk file
    - refcheck run unplaced
    - `_refcheck_unplaced` returning []
    - `part_a` ignoring the placed plans
    - G8a unplaced
    - equivalence unplaced
    - plain `Pool` on a look set
    - the worker without `hold_for_life`
    - the Evaluator's refusal removed
    - FITS read by name
    - `PlacedPool.close` keeping the placement
    - no phase check
    - a tolerance of two frames
- Record-key hashes: `45719d030359ec7d5d3914af3814d837fdc05d77652ba45c64792b511027528d c0027992de21067525cc4489b74acb157feb89f3a4c7618580addc2b341be1c1`, the registered pair. bar.py and moves.py are unchanged.
- `git status --porcelain` is empty in the worktree and in the main checkout. All my outputs are in the scratchpad (outside the repo). The scratch farms, synthetic sets and basetemps are deleted.
- Merge readiness:
  - The working branch is now 60cf4489. It moved past the builder's merge base 96d38d2d by STATUS.md only, so this is no fast-forward.
  - `git merge-tree --write-tree` is clean (exit 0), tree 55ce09bcb61b9030b97d5ef21252e39bf0050fe9. It differs from the working branch only in the package's 10 files.
  - The merge fa93f41c's tree equals 96d38d2d's.
  - Commits not in the working branch, oldest first: fa93f41c (merge), f54e8857, 7d63961d, 85078533, 86cf807a, 4f5b46be. All six end with exactly the two attribution lines, and none names a model elsewhere. None lacks them.

TEST SUMMARY LINE (my run, worktree at 4f5b46be): "179 passed, 24 skipped in 381.74s (0:06:21)".
- The 24 skips: 6 lookbar STAGE2_SLOW; in gates2, 7 that need V0's arrays, 3 that need the look ladder's, and 1 slow; 5 s2rows slow; 1 r2lfit slow; 2 r5lib that need the look sets' arrays.

WHAT TO BRING IN: merge wt/wpc-cw at 4f5b46bee6a93574b2f71956910273ebda1d7eed with a merge commit. Give findings 1 to 3 owners before G9-ref runs on any look set.

NOT CHECKED
- STAGE2_SLOW tests.
- The gates2 tests that need V0's or the look ladder's arrays (V0 was covered by my own comparison).
- The two committed-look-set r5lib tests: the rebuild was writing wp-l's V6noF5 arrays.
- No end-to-end `bar.evaluate` or `lookbar run`.
- No real equivalence or env episode on a look set: my probe stopped before physics.
- train2, nb, monitor2 and barviews tests.
- ruff.
- The builder's dry-run numbers.

What I ran: only kinematic work on synthetic sets in the scratchpad, plus eight cut 1 s equivalence checks on the S1 view with a freshly initialised policy, for the old-against-new comparison. Nothing on any of the study's look sets; no G9-ref, fit or training.

Scripts and logs, in /tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/lt2/check/:
- probe_new.py, probe_worker.py, probe_new.out
- offsets.py, off_new.out, off_old.out, equiv_dump.py
- partb_probe.py
- mutate_check.py, mutate_check.out
- full.log (my first run: no summary line, because my extra -q on top of the config's -q dropped it), full2.log
