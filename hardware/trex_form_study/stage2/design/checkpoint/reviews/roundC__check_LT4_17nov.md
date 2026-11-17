VERDICT: merge as it stands. All three points of ruling 40 are done. Each new test fails without its fix. Only the four owned files differ. The hashes are the registered pair. The merge is clean, but it needs a merge commit, because the working branch has moved on since the builder's fast-forward.

Tip to merge: 887fad6e215a6c61d18d8a3ef3e29565cb17a77b (wt/wpc-cw).

FINDINGS
1. Merge readiness (information).
   - The working branch is now at 760a6b58. It gained servo S2 after the builder's fast-forward to 23ea3fc1: 6bfecc33 and 760a6b58, which touch r5env.py, train2.py, g8s.py, the servo_model docs, and the tests for r5env, train2 and g8s.
   - So wt/wpc-cw cannot fast-forward any more.
   - `git merge-tree --write-tree HEAD claude/mesozoic-trex-stage2-refine-4jffre` exits 0 with tree 616e2c91089a9ae91caab1f2a5f0e0b45180adc3. That tree differs from the branch tip only in gates2.py, lookbar.py, r5lib.py and tests/test_stage2_gates2.py (87 insertions, 12 deletions).
   - The branch moved only outside the owned files and the three test files.
   - The builder's fast-forward claim holds: the reflog shows "23ea3fc1 ... merge ...: Fast-forward", and 8caa90c1 is an ancestor of 23ea3fc1.
2. Low, information only; it does not block. In the `gates2 refcheck` command, `refcheck()` writes the record before `refcheck_unplaced()` runs.
   - If `refcheck_unplaced` raised an exception instead of returning a list, the record would stay. The command would then exit 2 with a traceback.
   - In practice that cannot happen: the same set load, planner and placement have just run inside `refcheck`, and per-item errors are caught and listed as not placed.
   - The ruling's other option, writing the record only after the check passes, would close this too. Not needed for the ruling as worded.
3. Information. lookbar's module docstring (about lines 88 to 92) still carries the old list of readers: part (a), G8a, equivalence, r2lfit. It lacks part (b), the views and the placed refcheck. The builder also noted this. It is outside this round's parts, so it goes to the next change to lookbar.py.

CHECKED AND FOUND RIGHT
- Point 1 (gates2 `main`, refcheck branch):
  - When `refcheck_unplaced` returns a non-empty list, it unlinks `<out>/<set>/<body>/refcheck.json` (the same path `bar.refcheck` writes) and returns 2.
  - The "record:" line is printed only for a record that stays.
  - Off the look sets nothing changes: `bad` is None there, so the record and "the bar's own" lines are as before.
  - The test monkeypatches the placement inside the real `gates2.refcheck` to one that places nothing. It then runs the real `main` with `--out` set to the sets' root, and again without `--out`.
- Point 2: `_SetView` raises GateError "set.json names the body ..." right after the set check, before any plant or array is read.
  - All 9 set.json files in the main checkout name their own set and body (I read them), so the real sets are unaffected.
  - `_SetView` is built only in `g9_ref` and in tests.
- Point 3 (docstrings):
  - The only production callers of `leave_point` are `r5lib.place` (r5lib.py:410) and `lookbar.leave_of` (lookbar.py:921), and both pass `walk_file`. This holds in the merged tree too.
  - The readers named in `look_set_placement`'s list match the code: `gates2.part_b` at line 1010, `barviews.SetView.resolve` (through `_resolve_placed`, or `_resolve_in_child` when the process renders), and `gates2.refcheck`.
- r5lib.py and lookbar.py are unchanged apart from docstrings. My own AST comparison of 23ea3fc1 against HEAD, with docstrings blanked: r5lib.py True, lookbar.py True.
- bar.py and moves.py are not touched. Hashes: `45719d030359ec7d5d3914af3814d837fdc05d77652ba45c64792b511027528d c0027992de21067525cc4489b74acb157feb89f3a4c7618580addc2b341be1c1`, the registered pair.
- `ruff check --no-cache` on the 4 files: "All checks passed!"
- Commits f73dd025, 37a63f23 and 887fad6e each end with exactly the two attribution lines (raw message checked). None names a model elsewhere. The author and committer name is "Claude", the attribution identity.
- `git status --porcelain` is empty in the worktree and in the main checkout. I wrote no tracked file.

FAILS WITHOUT THE FIX
I put scratch copies of gates2.py first on designlab.stage2's `__path__` through a pytest plugin, so the worktree was never written. Then I ran the two new tests:
- CONTROL (HEAD): "2 passed, 53 deselected in 11.35s"
- M0, the old gates2 from 23ea3fc1: "2 failed, 53 deselected in 6.44s". Errors: `AssertionError: assert (not True)` at line 705, and FileNotFoundError on refs/nominal.npz.
- M1, unlink removed: "1 failed, 1 passed, 53 deselected in 6.45s" (the refcheck test fails).
- M2, `return EXIT_ERROR` replaced by `pass` (the LT3 check's C3): "1 failed, 1 passed, 53 deselected in 6.14s"
- M3, body check removed: "1 failed, 1 passed, 53 deselected in 10.07s" (the body test fails with FileNotFoundError).
- M4, unlink of a wrong path: "1 failed, 1 passed, 53 deselected in 5.86s"
- M5, a copy of HEAD through the plugin: "2 passed, 53 deselected in 9.78s"

TEST LINES (mine, worktree at 887fad6e)
- `tests/test_stage2_gates2.py tests/test_stage2_r5lib.py tests/test_stage2_lookbar.py`: "128 passed, 19 skipped in 235.55s (0:03:55)". The builder's count was the same.
- The skips: the frozen sets' arrays and the look ladder's arrays are not in the worktree, plus the STAGE2_SLOW tests.

NOT CHECKED
- The three test files on the merged tree 616e2c91 with servo S2. I made no checkout of it, per the no-whole-tree-copy rule.
- The skipped tests: the V0 and frozen-set tests, the look-ladder and look-set-array tests, and STAGE2_SLOW.
- Any run on the study's look sets.
- An end-to-end `gates2 refcheck` or G9-ref on a real set.

Files: the scripts and logs are in /tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/lt4/check/: astcmp.py, mkmut.py, mutplug.py, runmut.sh, mut.out and full.log. The basetemps and the mutant copies are deleted.
