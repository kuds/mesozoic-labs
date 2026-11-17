VERDICT: merge as it stands. Both points (ruling 33 and ruling 31 (6)) are done as the rulings say, and each has a test that fails without it. I found no defect. Tip to merge: wt/wpc-ct at 3e99223abce1f86c9c14c8ebdae39df9da97f8e7, as one merge into claude/mesozoic-trex-stage2-refine-4jffre. The merge is clean against the branch's current tip, fe605133.

FINDINGS (most serious first)
None blocking. There are three notes; none needs a change in this package.

1. For the lead (a text, not the package's). Rule M3 in STAGE2_TRACKB_PLAN.md:1132-1133 still says S2-4's baseline must equal the committed one in "(outcome, survival and state digest)". It does not have ruling 33's words "in every episode that ran". The code (EXPECT_BASELINE_RULE) and the halt reason now carry them. Proposed fix: the lead adds those words to M3 when the old texts are next written.

2. Info: the working branch has moved on since the builder's report. The report puts it at 3da0c6a8; it is now fe605133, six commits later.
   - Those commits change sets.py, moves_look.py and their tests (WP L, rulings 18/23/26) and add review files.
   - None of them touches this package.
   - The merged tree is 5c35ce86 (exit 0). Its designlab/ and tests/ are byte-identical to the tree I tested (036b908f, from the merge with e4b94077). Against the tip, it differs only in monitor2.py, train2.py and their two test files (705 insertions, 199 deletions).

3. Low, for the lead (the builder's point b). A look-set refusal at a further delay (rule M6) becomes MonitorError, exit 1. It is not tested.
   - Practically it cannot happen. In the same call the main delay already ran lookbar's X1 check on the same set, body and G9-ref record, and would have refused first.
   - Exit 1 at a further delay matches the existing rule for a row's refusal there. I agree with the reading as built.

CHECKED AND FOUND RIGHT

Ruling 33 (monitor2.baseline_equal):
- Digests are required only where at least one episode of the pair ran: `x.get("outcome") != NOT_RUN or y.get("outcome") != NOT_RUN`. A missing outcome counts as "ran", which fails closed.
- Every pair, two not-run episodes included, is still compared on condition, seed, outcome and survival_s.
- `digest_compared` counts a digest only when both sides hold a non-empty str under the same key. None or "" on either side is not compared.
- `digests_compared = bool(ran) and all(ran)`, so it is False when no episode ran or the records are empty. The trainer then halts, which fails closed.
- I checked lookbar's real not-run episode shape (lookbar.py:982, no digest key). companion_episode always sets state_digest on read and fell episodes (a str, or None when taking it raised).
- train2: EXPECT_BASELINE_RULE says "in every episode that ran". The halt reason was updated.
- Other users of digests_compared only report it: baseline_same (in a run.json "remade" entry) and write_s2_10. No other module calls baseline_equal (grep).

Ruling 31 (6) (monitor2._episodes):
- Only lookbar.LookSetRefused is caught around lookbar.episodes. It is looked up in sys.modules and becomes MonitorRefusal kind "look_set", which leads to exit 8 and the note. Anything else propagates.
- LookSetRefused can only come from LookSetRule.check(), before open_set and before any episode. Per-episode exceptions are caught inside lookbar.
- The monitor's rows (R0L@10ms, R5/<seed>@10ms, R6/<seed>@10ms; train2.MODES = R5, R6) are all in LOOK_SET_ROWS, so on a look set the only refusal is LookSetRefused.
- The look_set_rule context restores bar._check_rule in a finally block, so no state leaks.
- train2.moved_sources reads only the kinds "sources" and "row", so "look_set" is never read as moved sources.
- The new kind is accepted by MonitorRefusal's REFUSALS check.

Tests that fail without the change:
- The builder's not-run episodes are built by the real lookbar.episodes, with only open_set and, on V6, require_g9_ref replaced.
- The look-set test uses the real LookSetRefused, with gates2.GATES pointing at a temporary directory.
- I ran 14 mutants of the package's code on a scratch export of the merged tree (only the changed files copied, the rest symlinked; export deleted afterwards). Each was caught by the intended new test:
  - a1 (digests required in every pair): monitor "1 failed, 1 passed, 27 deselected in 6.41s"; train2 "1 failed, 59 deselected in 6.74s"
  - a2 (None counts as compared): "1 failed, 1 passed, 27 deselected in 6.01s"
  - a3 ("" counts as compared): "1 failed, 1 passed, 27 deselected in 5.97s"
  - a4 (no bool(ran)): "1 failed, 1 passed, 27 deselected in 6.12s"
  - a5 (or changed to and): "1 failed, 1 passed, 27 deselected in 6.18s"
  - a6 (not-run pairs not compared on survival): "1 failed, 1 passed, 27 deselected in 6.29s"
  - a7 (state_digest key only): "1 failed, 1 passed, 27 deselected in 6.31s"
  - a8 (the old digests_compared code): monitor "1 failed, 1 passed, 27 deselected in 6.09s"; train2 "1 failed, 59 deselected in 6.87s" (Halt "...the state digests were not compared...")
  - b1 (no catch): "1 failed, 1 passed, 27 deselected in 2.25s" (LookSetRefused raised out)
  - b2 (kind "row"): "1 failed, 1 passed, 27 deselected in 2.25s"
  - b3 (every exception a refusal): "1 failed, 1 passed, 27 deselected in 6.05s" (DID NOT RAISE RuntimeError)
  - b4 (looked up in ROW_REFUSALS): "1 failed, 1 passed, 27 deselected in 2.29s"
  - c1 (rule text without "in every episode that ran"): "1 failed, 59 deselected in 13.82s"
  - c2 (digests_compared ignored by the trainer): "1 failed, 59 deselected in 20.64s"
  - Controls with no mutant: "2 passed, 27 deselected in 6.03s" and "1 passed, 59 deselected in 13.98s"

Rewrap commit 3e99223a: an AST comparison of all module-level string constants (including the RULES dict) between 6f058823 and 3e99223a finds no difference, in either file.

Scope and safety:
- Only the four package files differ from the working branch. bar.py and moves.py are identical to the working branch.
- The record-key hashes at the head print the registered pair: 45719d030359ec7d5d3914af3814d837fdc05d77652ba45c64792b511027528d c0027992de21067525cc4489b74acb157feb89f3a4c7618580addc2b341be1c1.
- Behaviour off the look sets is unchanged: look_set_of returns None for V0, F25F9, F25F9cl and S1, so nothing is checked or caught there. baseline_equal is used only by --expect-baseline, the s2-10 report and baseline_same's report.
- ruff check --no-cache on the four files: "All checks passed!"
- git status --porcelain is empty in the worktree and in the main checkout, before and after my runs. No worktree file is newer than my run markers.
- Nothing ran beyond the unit tests (stub env, stub or fixture episodes).

Commits: every commit not in the working branch ends with exactly the two attribution lines, and none names a model elsewhere: 3e99223a, 6f058823, db1c01d6, ef8f85ac (merge), 1ecd32d1, c8132d8b, 983f9671, b509c475, fdc6bc27, aadebd08, 3047d418, 1c00b280. None is missing them.

EXACT TEST SUMMARY LINES
Standing environment, tools/lowprio.sh, -p no:cacheprovider, basetemp under scratch wave8 (deleted afterwards).
- Merged tree (wt/wpc-ct merged with the branch at e4b94077; code and tests identical to the merge with the current tip fe605133), tests/test_stage2_train2.py tests/test_stage2_monitor2.py: "============ 86 passed, 3 skipped, 2 warnings in 325.48s (0:05:25) ============="
- Worktree at HEAD 3e99223a, same two files: "86 passed, 3 skipped, 2 warnings in 322.60s (0:05:22)"
- The 3 skipped are the STAGE2_SLOW tests (test_stage2_train2.py:2433, 2459, 2478).

WHAT TO BRING IN
- Merge wt/wpc-ct at 3e99223abce1f86c9c14c8ebdae39df9da97f8e7 into claude/mesozoic-trex-stage2-refine-4jffre. git merge-tree --write-tree against fe605133 exits 0 with tree 5c35ce86cc35cb84f3632639b8678db317e41c7c. There is no list to cherry-pick.
- Still owed outside this package (from the CT4 check, the gates package): stage2/tb/run_trackb.sh V8-s2-4 should pass --expect-baseline.

NOT CHECKED
- The 3 STAGE2_SLOW trainer tests.
- Other packages' test files (lookbar, gates2, s2rows, nb, sets, moves_look) on the merged tree.
- Any real look-set or V6 episode (none is frozen). The not-run case is tested on episodes built by lookbar.episodes with a fixture open_set, not on real V6 output.
- An end-to-end trainer halt on a monitor refusal of kind look_set. I checked it by reading the code only: the trainer's handling is kind-agnostic except moved_sources.

Kept in /tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/wave8/check_ct5/:
- export.sh
- tools/mut.py, tools/run_mutants.sh, tools/consts.py
- logs/ (merged_both.log, head_both.log, mutants.log, mut_*.log)

Scratch exports and pytest temp directories are deleted.
