VERDICT: merge as it stands. Tip: wt/wpc-cn = 77366c95f7d5e45a9b5c20519d3d3c927cd5cd45. I found no blocker and no major defect. Two points for the lead below; both are wording of the steps sheet and neither stops the merge.

FINDINGS (most serious first)

1. MINOR, for the lead. stage2/tb/colab_steps.txt:227, step 5.3 (e).
   - The text: "(e) Only if you can see what it printed, and there is no such paragraph in it: go on with 5.4."
   - The problem: this is a negative test (no paragraph seen). It does not cover cell 8 while it is still running.
   - Why cell 8 can still be running when the owner gets here: the finish does not write STATUS.json (nb.py `_finish`, 2864-2967; only the SESSION record). So STATUS.json goes past 10 minutes while cell 8 still runs.
   - Cases where that happens: cell 8 waiting on `remount_if_dead` (nb.py:3046-3061), cell 8 in its last copy to a slow Drive, or a hang on a dead mount.
   - In those cases the owner sees partial output under the box, for example "Google Drive's connection was lost ...", with no paragraph yet. (e) then sends the owner to 5.4, whose "Disconnect and delete runtime" deletes the save.
   - That is the case the kept-runtime paragraph exists for. Section 6 says "Tell me" for that same line, so (e) conflicts with it.
   - It is outside the ruling's literal text ("cannot be seen") but against its purpose. It is Colab behaviour, expected and not seen.
   - Proposed fix (sheet only): make (e) positive. "Only if what it printed ends with 'Disconnecting runtime in 5 seconds...' (release_runtime prints this only when it lets the machine go: nb.py:3089), go on with 5.4. If the box is still running, or it printed anything else: do not do 5.4; tell me." The test's (e) assertions would change with it.
   - Alternative: accept as built.

2. MINOR, for the lead. This is the builder's own point (d), and I agree it is real.
   - In a second or later session of the same copy, the box can still show an earlier session's cell-8 printout. That printout has no paragraph if that session let its runtime go, for example after a halt.
   - So (e) can fire on an old printout while the current runtime is being kept.
   - This happens only from session 2 on: the GitHub notebook has no saved outputs (I checked all 9 cells: 0 outputs).
   - The builder's option (i), "Edit > Clear all outputs" in 5.4 before Run all (expected, not yet seen), fixes it.
   - My fix in finding 1 does not cover this case. The two fixes work together.

3. NIT. 5.3 (a) names the copy "as in the table of 4". The smoke's copy is "s2 smoke-1" (3.2), which that table does not name.
   - In practice a smoke ends with DONE.json or HALT.json, so 5.2 catches it first.
   - A smoke resume is also refused by `session` ("a smoke is not resumed"). But 5.4's delete step comes before that refusal.
   - Fix if wanted: "(named as in the table of 4, or "s2 smoke-1" for the smoke)".

4. FOR THE LEAD (an incident, mine, not the builder's).
   - At about 06:56-06:58 UTC my first scratch clone of the merged tree filled the disk to 0 MB free. I deleted it at once.
   - An overlapping V6noF5 build (another agent's, 06:56:07 to 06:58:56, writing to scratchpad/lstart/scratch_sets) logged "built 1 ... exit 0". Its outputs are worth one check for a short or cut file.
   - The disk sits at about 400 MB free (df: 99-100%). That is a hazard for every job.

CHECKED AND FOUND RIGHT
- Ruling 13: done as it says.
  - Before 5.4, the "More than 10 minutes old" branch now says "Do not resume yet". (a) has the owner reopen the tab or the Drive copy, marked expected and not yet seen. (b) has the owner look under "8. FINISH" and press nothing.
  - (c): when the printout cannot be seen (nothing printed, the page does not open, or not sure), the owner does not do 5.4, tells me, and waits.
  - (d) keeps ruling 12's exception.
  - 5.4's head lists where the owner may come from.
  - Section 8 lists the new Colab behaviours.
  - The other ways into 5.4 are 5.2, section 6 and 5.6. They are the lead's word or come after a 5.4 already done.
- Finding 2 (the notebook's test gap):
  - The halt test now holds retry_check in LEAD.json and asserts that the session record, SESSION_1.json and argv carry no retry-check.
  - nb.py is unchanged. n1d (dropping `ld.get("retry_check") is True`) is equivalent: `_halt_to_resume` (nb.py:1315-1321) already requires it.
- Mutants, mine, on a scratch copy of the merged tree. The selections passed unmutated first: "2 passed, 65 deselected in 0.65s" (sheet) and "2 passed, 65 deselected in 1.59s" (retry_check).
  - The old sheet: "2 failed, 65 deselected in 0.81s", CAUGHT.
  - 13 sheet mutants, all CAUGHT, each "1 failed, 1 passed, 65 deselected" or "2 failed": (c) removed; (c) without "do not do 5.4"; (c) only for "nothing printed"; (a) without the reopen; (a) without its mark; (b) pressing play; (e) loosened; the head going on to 5.4; the steps after 5.4; the old 5.4 head; (c) going on to 5.4; the section 8 entry removed; a skip hint after (e).
  - Cell 8 renamed: CAUGHT.
  - n1 (the check's mutant), full file: with the new test "1 failed, 64 passed, 2 skipped in 59.10s", CAUGHT. With the old test file, sheet tests deselected: "63 passed, 2 skipped, 1 deselected in 59.26s", SURVIVED. So the new test is what catches it.
  - n1b (LEAD.json only): CAUGHT, "2 failed".
  - n1c (and changed to or): CAUGHT, "1 failed, 1 passed, 65 deselected".
  - Every mutated file was restored and compared byte for byte.
- Only the package's files differ.
  - `git merge-tree --write-tree HEAD claude/mesozoic-trex-stage2-refine-4jffre`: exit 0, tree 3cdb728fa267fec4fd3c3e94e3d81d9f4c59b4df, against the tip as it is now (5800fa69, which moved during my check; it adds only STATUS.md).
  - That tree differs from the tip only in colab_steps.txt (+/-25) and tests/test_stage2_nb.py (+/-53).
  - The ipynb, nb.py and requirements-colab.txt are unchanged. bar.py is untouched.
- Record-key hashes: the registered pair, 45719d030359ec7d5d3914af3814d837fdc05d77652ba45c64792b511027528d c0027992de21067525cc4489b74acb157feb89f3a4c7618580addc2b341be1c1. I checked them on the worktree head and on the merged tree's designlab (2d898ee7, the merge with 81a395e1).
- The sheet is ASCII, with no CR and no tab. The only line over 115 columns is the existing URL (line 54).
- `ruff check --no-cache tests/test_stage2_nb.py`: "All checks passed!"
- Nothing ran that should not have.
  - `git status --porcelain --ignored` in the worktree shows only the old input directories and .ruff_cache from 16 Nov.
  - The main checkout's status is empty.
  - No new file under stage2/tb.
- Commits not in the working branch: f4e3e278, 1a6d3433 and 77366c95 (a merge; parents 1a6d3433 and 6a6409f8).
  - Each ends with the two given lines. None names a model anywhere else.
  - The builder's fast-forward claim holds: a557af92 is an ancestor of 3d261a79.

TEST SUMMARY LINES (mine, standing environment, tools/lowprio.sh)
- Worktree head 77366c95, tests/test_stage2_nb.py: "66 passed, 1 skipped in 66.61s (0:01:06)". The skip is the STAGE2_SLOW run.
- Merged tree 2d898ee7, as a scratch git copy holding only the files the tests need: "65 passed, 2 skipped in 64.12s (0:01:04)". The second skip, "the Stage 1 arrays are not in this checkout", is there because that copy holds no ignored inputs.
- A first try on that copy without git gave "6 failed, 59 passed, 2 skipped". All six failures were "git cannot tell this checkout's commit". It does not count.

WHAT TO BRING IN
- One merge of 77366c95f7d5e45a9b5c20519d3d3c927cd5cd45 into claude/mesozoic-trex-stage2-refine-4jffre (now at 5800fa69). It is clean.
- Still owed elsewhere: finding 3 of the follow-up check (train2.py:368, CT's test gap). Ruling 13 says both test gaps go in with it.

NOT CHECKED
- The STAGE2_SLOW run.
- Anything only Colab shows: what a reopened tab shows under a box, whether Run all clears queued cells' old output, and drive.mount's wait.
- The sheet on a phone.
- The other test files on the merged tree.

Files are in /tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/wave7/check_cn3/:
- The mutant script: mutate.py
- The mutant results: mutants.log
- The test logs: full_nb3.log, merged_nb2.log
- The old sheet and test, from 6a6409f8: old_sheet.txt, old_test.py.txt
The scratch copy is deleted.
