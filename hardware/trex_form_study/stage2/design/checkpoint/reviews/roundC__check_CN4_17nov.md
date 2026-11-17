VERDICT: merge as it stands. Tip to merge: wt/wpc-cn at f9019d177c6f11e93cd1a058129ad555ccee710e, merged into claude/mesozoic-trex-stage2-refine-4jffre (now at fe605133). I found no blocker and no major defect. There is one low test gap, optional.

FINDINGS (most serious first)

1. LOW, test gap (optional). tests/test_stage2_nb.py, test_the_steps_sheet_clears_old_outputs_before_a_later_session.
   - Point (2) holds only while every later session of a copy goes through 5.4, or 7.1 for the smoke. The tests guard 5.4 and 7.1 only.
   - My probe mutant S19 changes section 6's "I then tell you to resume (5.4)" (colab_steps.txt:281) to "I then tell you to Run all again". It survives: "4 passed, 65 deselected in 2.77s".
   - The sheet as built is right. This only protects against a later change.
   - My probe S18 (5.6 sends the owner to "Runtime > Run all again" in place of 5.4) is caught by test_the_steps_sheet_quotes_the_programs_own_lines.
   - Proposed fix: one assertion that "Run all" stands only in 3.5, 4.4 and 5.4 (d), and nowhere else in the sheet. Alternative: accept as built.

2. FOR THE LEAD, information only (not this package's file, and it fails closed).
   - The repository's disconnect_runtime prints the line without flush=True (environments/shared/notebook_runtime.py:35). nb.py's own fallback flushes (nb.py:3089).
   - On Colab the kernel's output stream should flush during the 5-second sleep (expected, not seen).
   - If it does not, the last line is "Flushing Google Drive writes...", and (e) sends the owner to the lead. Nothing to change.

3. FOR THE LEAD: the builder's readings a) to d). I find all four defensible.
   - a) 7.1 also clears outputs.
   - b) 3.5 does not clear. After a refused Drive permission, Run all stops at cell 3, so cell 8 has no old output.
   - c) "or you are not sure" and the turning-circle description, marked expected, not yet seen.
   - d) During cell 8's 5-second sleep the box is still running, so (e) sends the owner to the lead.

CHECKED AND FOUND RIGHT
- Point (1) is done as ruled (colab_steps.txt:227-232).
  - (e) is positive: go on to 5.4 only if the box has stopped and its last line is "Disconnecting runtime in 5 seconds...", quoted on a row of its own.
  - Still running, any other last line, or not sure: do not do 5.4, tell me, wait. It carries the mark.
  - The exact text matches the code: nb.py:3089 and notebook_runtime.py:35.
  - Neither path prints anything after that line (release_runtime, nb.py:3065-3091). The two keep paragraphs end "...that is expected.", so they never end with it.
  - Cell 8's source (the ipynb's last cell) calls release_or_keep in `finally`. A crash or stop adds a traceback after the keep paragraph, which (e) stops on.
  - AUTO_DISCONNECT unticked prints "Auto-disconnect skipped...", which (e) also stops on.
  - Clearing is what makes (e) safe from session 2 on. A runtime killed before cell 8 leaves cell 8 empty, and (c) applies.
- Point (2) is done (5.4 (a) to (d), lines 234-242).
  - The order is disconnect, then Edit > Clear all outputs (marked expected, not yet seen), then RESUME, then Run all. 5.4 has exactly one Run all.
  - 5.6 goes back through 5.4. 7.1 points to 5.4 (b). Section 8 lists the new behaviour.
  - The GitHub notebook's 8 code cells hold 0 outputs (checked).
- Point (3) is done (lines 219-220: 'or "s2 smoke-1" for the smoke'), matching 3.2.
- Sheet format: ASCII, no CR, no tab. The only line over 115 columns is the old URL (line 54). The new marks are at column 88, like the others.
- The builder's line references hold: nb.py:3089, notebook_runtime.py:35, RELEASE_CODES = (EXIT_OK, EXIT_HALT).
- My mutants ran on a scratch export: real copies of the sheet, the test, nb.py, tbenv.py, notebook_runtime.py, conftest.py and pyproject.toml; the rest symlinks.
  - Unmutated baseline: "4 passed, 65 deselected in 2.91s".
  - S0, the working branch's old sheet with the new tests: "3 failed, 1 passed, 65 deselected in 0.89s". Caught.
  - Sheet mutants S1 to S18, all caught:
    - S1, (e) negative again: "2 failed, 2 passed".
    - S2, quoted line changed: "2 failed, 2 passed".
    - Each "1 failed, 3 passed": S3 still-running clause dropped; S4 anything-else clause dropped; S5 "box has stopped" dropped; S6 (e) mark dropped; S7 "Tell me" dropped; S8 "go on while running" added; S9 clearing removed; S10 clearing after Run all; S11 (b) mark dropped; S12 Drive sentence dropped; S13 7.1 pointer dropped; S16 section 8 running entry dropped; S18 5.6 bypass.
    - Each "2 failed, 2 passed": S14 smoke copy dropped; S15 section 8 clearing entry dropped; S17 (a) and (b) swapped.
  - Code mutants against the behaviour test, all caught, each "1 failed, 68 deselected": N1 fallback text changed; N2 release on EXIT_UNFINISHED; N3 fallback prints after unassign; N4 the repository's disconnect_runtime text changed; N5 release_or_keep prints after a release; N6 the keep path ends with the release line.
  - Only S19 (finding 1) survived. Every scratch copy was compared byte for byte with the worktree at the end; the export is deleted.
- The behaviour test runs in a subprocess with a stand-in google.colab, so no state leaks into the pytest process. It proves both paths are exercised: the fallback's own note appears only on the "own" path.
- Only the package's files differ.
  - `git merge-tree --write-tree HEAD fe605133`: exit 0, tree 4442831a7076afbe1010df856885185f47278262.
  - That tree differs from the working branch's tip only in colab_steps.txt (+/-34) and tests/test_stage2_nb.py (+90/-?, 110 insertions, 14 deletions in all).
  - HEAD differs from the merged tree only by the three build reports the lead saved in fe605133.
  - nb.py, the ipynb, requirements-colab.txt, bar.py and moves.py are unchanged.
- The builder's merge history holds. The reflog shows a fast-forward from 77366c95 to 3da0c6a8, then commit def58b7f, then the merge f9019d17 (parents def58b7f and e4b94077).
- Record-key hashes on the worktree head are the registered pair: 45719d030359ec7d5d3914af3814d837fdc05d77652ba45c64792b511027528d c0027992de21067525cc4489b74acb157feb89f3a4c7618580addc2b341be1c1.
- Nothing ran that should not have.
  - The worktree's `git status --porcelain` is empty. `--ignored` shows only .ruff_cache (last changed 02:35 UTC) and the old input directories.
  - The main checkout's status is empty.
- `ruff check --no-cache tests/test_stage2_nb.py`: "All checks passed!"
- Commits not in the working branch: def58b7f69f212a8db30055fef0de893b1d78e24 and f9019d177c6f11e93cd1a058129ad555ccee710e (a merge). Both end with the two attribution lines exactly, and no model is named anywhere else in either message. No commit lacks them.

TEST SUMMARY LINES (mine, through tools/lowprio.sh, standing environment)
- Worktree head f9019d17, tests/test_stage2_nb.py: "68 passed, 1 skipped in 67.69s (0:01:07)". The skip: "the real trainer on the S1 view; set STAGE2_SLOW=1".
- Sheet and behaviour selection (`-k "steps_sheet or 5_3_e"`) on the scratch export: "4 passed, 65 deselected in 2.91s".

WHAT TO BRING IN
- One merge of f9019d177c6f11e93cd1a058129ad555ccee710e into claude/mesozoic-trex-stage2-refine-4jffre (tip fe605133). It is clean. No cherry-pick is needed.

NOT CHECKED
- The full test file on the merged tree 4442831a. It differs from the tested head only by three .md build reports, which the test file does not read.
- The STAGE2_SLOW run. Other test files.
- Anything only Colab shows: the Edit > Clear all outputs menu (also on a phone), whether the cleared state is what a reopened Drive copy shows, how a running box looks, and what remains under cell 8 after the runtime is unassigned.
- The sheet on a phone.

Files: /tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/wave8/check_cn4/mutate.py, /tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/wave8/check_cn4/mkexp.py, /tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/wave8/check_cn4/mutants.log, /tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/wave8/check_cn4/full_nb.log, /tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/wave8/check_cn4/old_sheet.txt. The scratch export and the pytest basetemps are deleted.
