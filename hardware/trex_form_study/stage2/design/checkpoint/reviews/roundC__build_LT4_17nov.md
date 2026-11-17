LT4 (ruling 40, check of LT3 findings 1 to 3) is built in worktree /home/user/mesozoic-wt/wpc-cw on branch wt/wpc-cw. All three points are done in three commits, and the three test files pass.

Merge: `git merge --ff-only claude/mesozoic-trex-stage2-refine-4jffre` fast-forwarded 8caa90c1..23ea3fc1. 8caa90c1 was already an ancestor, so no merge commit was made. That brought in LT3, the LT3 reports, ruling 40 and the servo files.

1. Placed refcheck command (gates2.py `main`, `refcheck` branch; commit f73dd025)
   - What changed: when `refcheck_unplaced` finds a look set's turn or stop items not placed, the command now deletes `<out>/<set>/<body>/refcheck.json` before returning exit 2. The refusal line on stderr says "the record written is removed". The "record: ... (sha256 ...)" line is now printed only for a record that stays. The module docstring says so too.
   - Test: `test_the_placed_refcheck_leaves_no_record_when_it_refuses_an_unplaced_item` (needs_s1, synthetic V6). The placement inside `gates2.refcheck` is monkeypatched to one that places nothing.
     - With `--out` set to the sets' root it asserts: exit 2, the record was written and then is gone, the refused items are the three leave keys, and there is no traceback.
     - Without `--out` it asserts exit 2 and the same refusal.
   - Fails without the fix: on the old gates2 it fails with `AssertionError: assert (not True)` on `rec.exists()`. It also fails with the review's mutation C3 (`return EXIT_ERROR` replaced by `pass`) and with the unlink removed.

2. `gates2._SetView` (commit 37a63f23)
   - What changed: after the existing set-name check it raises `GateError("<dir>: set.json names the body '<other>'")`. The set message is unchanged.
   - Test: `test_g9_ref_refuses_a_set_json_that_names_another_body`. It runs on minimal set directories and needs no arrays. It checks the body refusal and, beside it, the existing set refusal.
   - Fails without the fix: on the old gates2 it fails with a FileNotFoundError for refs/nominal.npz instead of the GateError.
   - All 9 set.json files in the main checkout name their own set and body, so these sets are unaffected.

3. Docstrings only (commit 887fad6e)
   - `r5lib.leave_point` no longer says the env's place gives no file. It now says both callers give the walk file (`lookbar.leave_of` and the env's `place`, ruling 39 (c)), and that None, which no caller passes, skips the check.
   - `lookbar.look_set_placement`'s list of readers now includes G9-ref part (b) (`gates2.part_b`), the owner's views (`barviews.SetView.resolve`, in a child process where the process renders) and the placed refcheck command (`gates2.refcheck` / `gates2 refcheck`).
   - An AST comparison with docstrings blanked shows the code of r5lib.py and lookbar.py is unchanged.
   - Not edited: lookbar's module docstring (lines about 90 to 92) has the same old list of readers. It is outside this round's parts.

Tests:
- The three files at HEAD (`tests/test_stage2_gates2.py tests/test_stage2_r5lib.py tests/test_stage2_lookbar.py`): "128 passed, 19 skipped in 232.14s (0:03:52)"
- The two new tests alone: "2 passed, 53 deselected in 12.04s"
- The same two on the old gates2: "2 failed, 53 deselected in 7.94s"
- ruff --no-cache on the 4 files: "All checks passed!"

Hashes: `45719d030359ec7d5d3914af3814d837fdc05d77652ba45c64792b511027528d c0027992de21067525cc4489b74acb157feb89f3a4c7618580addc2b341be1c1`. That is the registered pair; bar.py and moves.py are not touched.

`git diff claude/mesozoic-trex-stage2-refine-4jffre --stat`:
```
 .../trex_form_study/designlab/stage2/gates2.py     | 23 +++++---
 .../trex_form_study/designlab/stage2/lookbar.py    |  8 +--
 hardware/trex_form_study/designlab/stage2/r5lib.py |  5 +-
 .../trex_form_study/tests/test_stage2_gates2.py    | 63 ++++++++++++++++++++++
 4 files changed, 87 insertions(+), 12 deletions(-)
```

`git status --porcelain` in the worktree is empty. The main checkout shows one untracked file I did not write: `?? hardware/trex_form_study/stage2/design/servo_model/S2_check_17nov.md`.

Commits:
- f73dd025: the refcheck command
- 37a63f23: `_SetView`
- 887fad6e: the docstrings

Each message ends with exactly the two attribution lines and names no model elsewhere. Branch head of wt/wpc-cw: 887fad6e215a6c61d18d8a3ef3e29565cb17a77b. Nothing was pushed.

No run touched the study's look sets; only synthetic sets in temp dirs were used. The basetemps are deleted. The test log is kept at /tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/lt4/full.log.
