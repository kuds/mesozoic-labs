CT5 report (WP CT, worktree /home/user/mesozoic-wt/wpc-ct, branch wt/wpc-ct). Both rulings are built, and both test files pass at the head 3e99223a. The working branch moved after my merge (one doc-only commit), so I did not merge it again.

1. MERGE OF THE WORKING BRANCH
- Merge commit ef8f85ac brought in claude/mesozoic-trex-stage2-refine-4jffre up to 32b5ecc5. There were no conflicts: none of its 21 files is one of this package's.
- After that, the working branch gained 3da0c6a8 ("Save the build report of the look start's law", one new review file: roundC__build_LS_17nov.md). It touches nothing of the trainer's. I did not merge it again. `git merge-tree --write-tree claude/mesozoic-trex-stage2-refine-4jffre HEAD` exits 0 (tree 0e9c5bfab4d7...).

2. PER POINT

Ruling 33 (commit db1c01d6):
- monitor2.baseline_equal:
  - It requires state digests only for pairs in which at least one episode ran (outcome != NOT_RUN).
  - Two not-run episodes are still compared on condition, seed, outcome and survival.
  - A digest counts as compared only when both sides hold a non-empty str. This is a new helper, digest_compared, and a new constant DIGEST_KEYS = ("state_digest", "digest"). A digest of None or "" is not compared.
  - digests_compared = bool(ran) and all(ran), so it is false when no episode ran.
- train2:
  - EXPECT_BASELINE_RULE says "with the digests compared in both in every episode that ran (the lead's ruling 33 ...)".
  - The module text and the Halt text name it too.
  - The halt's reason now reads "the state digests were not compared in every episode that ran (a record holds none, or none that is a non-empty string, or no episode ran)".
- Test in test_stage2_train2: test_a_committed_baseline_with_a_condition_not_run_lets_the_run_train.
  - A new STUB_NOT_RUN setting in stub_episodes makes turn/q90_left not run in every row.
  - Those episodes are built by lookbar.episodes itself (helper lookbar_not_run). Only lookbar.open_set is replaced, so the set holds none of the references; the row is a LookRow around a bare panels.Row.
  - The committed baseline holds [read, read, not_run, read, read, read] and the run trains: status done, baseline_expected equal True, 6 episodes, digests_compared True.
  - A committed file whose ran episode lost its digest still halts the run.
- Test in test_stage2_monitor2: test_an_equal_baseline_with_a_condition_not_run_compares_the_digests_of_the_episodes_that_ran.
  - It covers the same case directly on baseline_equal.
  - Two not-run episodes are compared on survival.
  - A not-run episode against one that ran gives a differ and digests not compared.
  - None and "" are not compared under either key, on one side or both (finding 2).
  - Also: a missing digest; the old `digest` key; no episode ran (False); empty records (False).
  - The not-run episodes again come from lookbar.episodes (V6; open_set and require_g9_ref replaced).

Ruling 31 (6) (commit 6f058823):
- monitor2:
  - REFUSALS gains the kind "look_set".
  - New table SET_REFUSALS = (("designlab.stage2.lookbar", "LookSetRefused", "look_set"),), looked up in sys.modules like ROW_REFUSALS. _row_refusal takes the table as an argument.
  - _episodes catches a SET_REFUSALS exception raised by lookbar.episodes. It raises MonitorRefusal("look_set", "the look set <set> refused the row <row> before any episode (LookSetRefused: ...)"), so the CLI leaves with exit 8 and writes its note.
  - Anything else lookbar.episodes raises still propagates as before.
  - RULES["refusal"] and the module text name the case.
- train2: the module text and _refusal's reason for exit 8 without a note name "the look set's G9-ref has not passed (rule X1)". The existing tests' strings are kept.
- Test in test_stage2_monitor2: test_a_look_sets_refusal_is_the_monitors.
  - gates2.GATES is a temporary directory, so the real lookbar.episodes raises the real LookSetRefused ("there is no G9-ref record").
  - Sets V6, V6noF5 and V6.2 are each checked through the CLI, at the baseline and at a check.
  - Result: exit 8, a note of kind look_set, and no record. open_set is never called, and the cause is LookSetRefused.
  - On F25F9 (no look set) nothing is refused: the set is read, and its error stays no refusal.
  - The old REFUSALS assertion is updated.

Commit 3e99223a only rewraps over-long lines. A check confirmed RULES and all constant texts are the same strings.

Failed without the change:
- I built a scratch export: the merge commit ef8f85ac's monitor2.py and train2.py, plus the final tests; every other file symlinked; deleted afterwards. Result: "3 failed, 86 deselected in 7.91s". The three failures:
  - train2: Halt "...R0L_BASELINE.json): the state digests were not compared (a record holds none)";
  - monitor2: {'digests_compared': False} != {'digests_compared': True};
  - monitor2: designlab.stage2.lookbar.LookSetRefused raised out of main.
- The same three failures were also seen in the worktree before I edited the code (there the count line was hidden by -q).
- The intermediate commit db1c01d6, in its own export: monitor2 file "28 passed in 37.91s"; train2 -k "condition_not_run or not_the_committed_one or expected_baseline_is_refused" "3 passed, 57 deselected in 43.01s".

FOR THE LEAD:
- (a) Refusal kind. I added a new kind, "look_set", rather than reusing "row". A look-set refusal is not the row refusing its checkpoint, and train2.moved_sources must not read it as one. Recommend keeping it.
- (b) At a further delay. A look-set refusal that happened only at a further delay (rule M6, after the check's own episodes) is exit 1 (MonitorError), not 8. This is the existing rule for a row's refusal there.
- (c) A pair whose sides hold the digest under different keys (state_digest against digest) counts as not compared.

3. TEST SUMMARY LINES AND HASHES (standing environment, tools/lowprio.sh, basetemp in scratch wave8)
- At 3e99223a:
  - tests/test_stage2_train2.py: "57 passed, 3 skipped, 2 warnings in 279.20s (0:04:39)"
  - tests/test_stage2_monitor2.py: "29 passed in 42.34s"
- At 6f058823, both files together: "86 passed, 3 skipped, 2 warnings in 320.31s (0:05:20)"
- ruff check --no-cache on the four files: "All checks passed!"
- The hashes at the head: 45719d030359ec7d5d3914af3814d837fdc05d77652ba45c64792b511027528d c0027992de21067525cc4489b74acb157feb89f3a4c7618580addc2b341be1c1 (the registered pair). bar.py and moves.py are untouched.

4. DIFF AND STATUS
git diff claude/mesozoic-trex-stage2-refine-4jffre --stat, package files:
 .../trex_form_study/designlab/stage2/monitor2.py   | 257 +++++++++----------
 .../trex_form_study/designlab/stage2/train2.py     | 132 ++++++++--
 .../trex_form_study/tests/test_stage2_monitor2.py  | 274 ++++++++++++++++++---
 .../trex_form_study/tests/test_stage2_train2.py    | 241 +++++++++++++++++-
 4 files changed, 705 insertions(+), 199 deletions(-)
- The plain two-dot diff also lists roundC__build_LS_17nov.md as a deletion (114 lines). That is because the branch has 3da0c6a8 and I did not merge it again.
- The three-dot diff (...HEAD) shows only the four files above.
- This round alone (ef8f85ac..HEAD): 285 insertions, 44 deletions, same four files.
- git status --porcelain: empty, in the worktree and in the main checkout /home/user/mesozoic-labs.
- No new output under the worktree; all outputs went to the scratch directory.

5. COMMITS, OLDEST FIRST
ef8f85ac (merge), db1c01d6, 6f058823, 3e99223a. Each message ends with the two given lines and names no model elsewhere. Branch head: 3e99223abce1f86c9c14c8ebdae39df9da97f8e7. Nothing pushed. No history rewritten.

NOT DONE / NOT CHECKED
- The 3 STAGE2_SLOW trainer tests (skipped).
- Other packages' test files (lookbar, gates2, s2rows, nb).
- Any real look-set or V6 episode (none frozen).
- I did not merge 3da0c6a8.
- Still owed outside this package, from the check: stage2/tb/run_trackb.sh V8-s2-4 should pass --expect-baseline (gates package).

Scratch exports and pytest temp directories are deleted. Logs and scripts are kept in /tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/wave8/ct5/ (logs/, export.sh, split33.py).
