WP CN, ruling 13 and finding 2 of the follow-up check: done in worktree /home/user/mesozoic-wt/wpc-cn (branch wt/wpc-cn). Head 77366c95f7d5e45a9b5c20519d3d3c927cd5cd45. Nothing pushed. Simulation-only text and test changes; no run that counts.

1. MERGES OF THE WORKING BRANCH
- First merge: my branch's head a557af92 was already the merge base, so `git merge --ff-only` fast-forwarded to 3d261a79. No merge commit and no conflicts.
- Second merge: the working branch then moved on by one commit, 6a6409f8 (rulings 21 to 26 added to ROUND_C_RULINGS_17NOV.md). I merged it again with --no-ff and the message given with -F: merge commit 77366c95. No conflicts; none of the package's files are touched.
- The record-key hash pair after both merges and on the final tree: 45719d030359ec7d5d3914af3814d837fdc05d77652ba45c64792b511027528d c0027992de21067525cc4489b74acb157feb89f3a4c7618580addc2b341be1c1, the registered pair.

2. ITEMS
(a) Ruling 13, the steps sheet before 5.4. File: hardware/trex_form_study/stage2/tb/colab_steps.txt.
- 5.3's "More than 10 minutes old" branch no longer says "Go on with 5.4". It now says "Do not resume yet. First look at the notebook:" and gives five steps:
  - (a) Open the notebook's tab again. If it was closed, open the copy from Drive (My Drive / Colab Notebooks, named as in the table of 4). Marked "(expected, not yet seen)".
  - (b) Scroll down to the last grey box, "8. FINISH". Only look at what is printed under it; press nothing.
  - (c) If you cannot see what it printed (nothing under it, the page does not open, or you are not sure): do not do 5.4. Tell me, and wait for my answer.
  - (d) If it printed the "THIS MACHINE IS KEPT ON PURPOSE" paragraph: do not do 5.4. Run the last box again as in 6 and tell me. This is the earlier exception, kept.
  - (e) Only if you can see what it printed, and there is no such paragraph in it: go on with 5.4.
- 5.4's head is now: Resume (only when 5.3 (e), 5.6 or I send you here; never after "THIS MACHINE IS KEPT ON PURPOSE").
- Section 8's "expected, not yet seen" list now also names two Colab behaviours: where the copy of the notebook is kept, and what a reopened tab shows under the last box (5.3 (a) to (c)).
- The sheet stays ASCII. No new line is over 115 columns.
- Tests, in tests/test_stage2_nb.py:
  - New test: test_the_steps_sheet_looks_at_the_last_box_before_a_resume. It checks that steps (a) to (e) come in order before 5.4, what each step says, that (e) is the only step of 5.3 that leads to 5.4, 5.4's head, that cell 8 is "8. FINISH", that both keep texts begin with the paragraph's words, and the section 8 entry.
  - The ruling-12 assertion in test_the_steps_sheet_quotes_the_programs_own_lines now uses the new wording of (d).
  - Mutants on the sheet, run with -k steps_sheet:
    - The old sheet: "2 failed, 65 deselected".
    - Step (c) removed: "1 failed, 1 passed".
    - (e) loosened to "If there is no such paragraph": "1 failed, 1 passed".
    - The head still sending the owner on to 5.4: "1 failed, 1 passed".
    - The reopen in (a) removed: "1 failed, 1 passed".
    - So all four mutants and the old sheet are caught. Restored afterwards (cmp OK).

(b) Finding 2 of roundC__check_FU_17nov.md: `"retry_check" in halt["remedy"]` had no test of its own.
- In test_a_halt_for_moved_sources_asks_the_lead_for_allow_source_change, LEAD.json now also holds retry_check true:
  - in the refused attempt, which is still refused for want of allow_source_change;
  - in the accepted attempt, which asserts s1["lead"]["doc"]["retry_check"] is True, s1["retry_check"] is False, SESSION_1.json retry_check False, and no --retry-check in the trainer's argv (already asserted).
- Mutant n1, (halt is not None and ld.get("retry_check") is True), applied to nb.py and then restored (cmp OK): "1 failed, 1 passed, 64 deselected in 1.47s". Now caught.
- nb.py is not changed.

(c) Not done, and not mine: the check's finding 3 (train2.py:368, the fallback in `allowed_sources`, test gap in tests/test_stage2_train2.py). Ruling 13 says "two test gaps go in with it"; that one belongs to the trainer package (CT) and is still owed.

(d) For the lead: an old printout under the last box.
- Colab normally keeps a cell's earlier output until that cell runs again (expected, not seen here). So in a second or later session in the same copy, if the browser was lost before cell 8 ran, the owner may see the previous session's cell-8 output. If that session ended by letting the runtime go, that output has no paragraph, so step (e) would send the owner on to 5.4 although the current runtime may be kept.
- Options:
  - (i) In 5.4, before Run all, add "Edit > Clear all outputs" (expected, not yet seen). Then anything under the last box belongs to the current session.
  - (ii) The finish prints the session number, and 5.3 asks the owner to compare it with the "Run directory ... session <k>" line. This needs a change to nb.py, and the comparison is harder for the owner.
  - (iii) Accept as built.
- I recommend (i). I did not build it: it adds an owner step that ruling 13 does not name.

3. TEST OUTPUT SEEN (standing environment, through tools/lowprio.sh, basetemp under /tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/wave7/cn13/)
- Full tests/test_stage2_nb.py on the final tree (77366c95): "66 passed, 1 skipped in 59.62s". The skip: "the real trainer on the S1 view; set STAGE2_SLOW=1" (the slow local run; not run).
- Targeted: "2 passed, 64 deselected in 1.71s" (the two retry_check tests); "2 passed, 65 deselected in 0.67s" (the sheet tests).
- ruff check --no-cache tests/test_stage2_nb.py: "All checks passed!"

4. git diff claude/mesozoic-trex-stage2-refine-4jffre --stat
 hardware/trex_form_study/stage2/tb/colab_steps.txt | 25 ++++++----
 hardware/trex_form_study/tests/test_stage2_nb.py   | 53 +++++++++++++++++++---
 2 files changed, 64 insertions(+), 14 deletions(-)
git status --porcelain: empty, in the worktree and in the main checkout. No __pycache__ was created. All outputs (logs, mutant script, saved copies) are in the scratchpad's wave7/cn13 folder, outside the repository.

5. COMMITS, OLDEST FIRST (each ends with the two given lines)
- f4e3e278: a test that LEAD.json's retry_check answers no halt whose remedy does not name it
- 1a6d3433: before 5.4 the steps sheet has the owner open the notebook's tab and look under the last box
- 77366c95: merge of the working branch (6a6409f8, rulings 21 to 26)
Branch head: wt/wpc-cn = 77366c95f7d5e45a9b5c20519d3d3c927cd5cd45

NOT DONE
- The STAGE2_SLOW local run.
- The sheet on a phone.
- Anything that only Colab shows.
- The trainer's test gap (finding 3).
