WP CN, ruling 34 (check of CN3): steps sheet. Worktree /home/user/mesozoic-wt/wpc-cn, branch wt/wpc-cn. Head: f9019d17f2ab (full id below). All three points are built and tested. nb.py, the notebook, requirements-colab.txt, bar.py and moves.py are not changed.

1. MERGES OF THE WORKING BRANCH
- First merge: a fast-forward from 77366c95 to 3da0c6a8 (`git merge --ff-only`). There was no merge commit and no conflict.
- The working branch moved while I was building (3da0c6a8 to e4b94077: commits c5544698, 3b9ee01c, 55e79155, 7cb824ec, e4b94077; the look sets' rulings 18, 23 and 26, and ruling 36).
- Second merge: f9019d17, `--no-ff`, message written with -F. No conflict. `git merge-tree` gave exit 0 beforehand. None of those commits touch a file of this package.

2. PER POINT
The exact text: `release_runtime` prints "Disconnecting runtime in 5 seconds..." on both of its paths:
- its own fallback: nb.py:3089, `print("Disconnecting runtime in 5 seconds...", flush=True)`;
- the repository's `disconnect_runtime`: environments/shared/notebook_runtime.py:35.
After that line, both only sleep and unassign; nothing else is printed.

(1) 5.3 (e) is now a positive test. Change in stage2/tb/colab_steps.txt:
- It reads: "(e) Read the last line printed under the box. Only if the box has stopped and its last line is / Disconnecting runtime in 5 seconds... / go on with 5.4. (The program prints that line only when it lets the machine go.) If the box is still running, or its last line is anything else, or you are not sure: do not do 5.4. Tell me, and wait for my answer. While a box is running, a circle turns around the button at its left. (expected, not yet seen)".
- Section 8's list now says "(5.3 (a) to (e))" and adds "how a box looks while it is still running (5.3 (e))".

Tests in tests/test_stage2_nb.py:
- `test_the_steps_sheet_looks_at_the_last_box_before_a_resume`: the (e) assertions are replaced. It checks the exact positive wording, the stop branch (still running, anything else, not sure, then "do not do 5.4. Tell me"), the mark, no "go on" in the stop branch, and no "no such paragraph" in 5.3.
- New `test_the_line_that_5_3_e_waits_for_is_the_last_one_a_released_machine_prints`. The line (e) quotes on its own row is the last line `release_or_keep` prints on Colab for exit codes 0 and 5, on both the repository path and the fallback path. For EXIT_UNFINISHED, 1 and None the last line is the "THIS MACHINE IS KEPT ON PURPOSE" paragraph. It runs in a subprocess with a stand-in google.colab.
- Shown to fail without the change:
  - the old sheet with the new tests: "3 failed, 1 passed, 65 deselected in 0.96s";
  - mutant "(e) negative again": "2 failed, 2 passed, 65 deselected in 0.80s";
  - mutant "line text changed": "2 failed, 2 passed, 65 deselected in 0.83s";
  - mutant "still running dropped": "1 failed, 3 passed, 65 deselected in 2.75s";
  - mutant "go on while running": "1 failed, 3 passed, 65 deselected in 2.42s";
  - nb.py mutants against the behaviour test, each "1 failed, 68 deselected" (fallback text changed: 2.73s; release on EXIT_UNFINISHED: 2.72s).
  - Every mutated file was restored byte for byte.

(2) A later session clears old outputs. Change in colab_steps.txt:
- 5.4 is now numbered "in this order":
  - (a) Runtime > Disconnect and delete runtime (if it is offered).
  - (b) Edit > Clear all outputs. This empties what the boxes printed in the session before, so that an old line is not read as a new one. The run's files on your Drive stay as they are. (expected, not yet seen)
  - (c) the first grey box with RESUME ticked; nothing else changes.
  - (d) Runtime > Run all. Allow Drive.
- 7.1 gains: "Either way, in a copy that has run before, clear all outputs first (Edit > Clear all outputs, as in 5.4 (b))."
- Section 8 lists "what Edit > Clear all outputs empties (5.4 (b))".
- New test `test_the_steps_sheet_clears_old_outputs_before_a_later_session`. It checks:
  - the order (a) to (d), with exactly one Run all in 5.4, after the clearing;
  - the mark and the Drive sentence;
  - the RESUME row;
  - the pointer in 7.1;
  - section 8's entry;
  - that the GitHub notebook's code cells hold no outputs, so a first session has nothing to clear.
- Shown to fail without the change: mutants "clearing removed" (1 failed, 3 passed, 65 deselected in 2.51s), "clear after Run all" (2 failed, 2 passed, 65 deselected in 2.69s), "mark removed" (1 failed, 3 passed, 65 deselected in 2.46s) and "7.1 pointer removed" (1 failed, 3 passed, 65 deselected in 2.61s).

(3) 5.3 (a) names the smoke's copy. It now reads: "(My Drive / Colab Notebooks, named as in the table of 4, or "s2 smoke-1" for the smoke)".
- The test asserts that wording in (a), and that 3.2 names the copy "s2 smoke-1".
- Mutant "smoke copy dropped": "2 failed, 2 passed, 65 deselected in 2.83s".

FOR THE LEAD (my readings; each can be undone in one line):
- a) I read "from the second session on" to include 7.1's re-runs, not only 5.4. Alternative: put it in 5.4 only and drop the 7.1 sentence.
- b) 3.5's "Run all again" after a refused Drive permission gets no clearing step. No session has opened then, and cell 8 has not printed. Alternative: add the clearing there too.
- c) (e) adds "or you are not sure", and a description of a running box (a turning circle), marked expected, not yet seen. Alternative: drop the description.
- d) During cell 8's 5-second sleep the box is still running while its last line is already "Disconnecting...". (e) then sends the owner to the lead rather than to 5.4. This is the conservative reading of the ruling.

3. TEST LINES AND HASHES
- The sheet tests after the change (`-k "steps_sheet or 5_3_e"`): "4 passed, 65 deselected in 2.58s".
- tests/test_stage2_nb.py, full file, before the second merge: "68 passed, 1 skipped in 63.87s (0:01:03)".
- The same file on the merged head f9019d17: "68 passed, 1 skipped in 66.50s (0:01:06)". The skip is "the real trainer on the S1 view; set STAGE2_SLOW=1".
- `ruff check --no-cache tests/test_stage2_nb.py`: "All checks passed!"
- Sheet format: plain ASCII, no tab or CR. Every new line is at most 115 columns, and the new marks sit at column 88.
- Hashes on the merged head: 45719d030359ec7d5d3914af3814d837fdc05d77652ba45c64792b511027528d c0027992de21067525cc4489b74acb157feb89f3a4c7618580addc2b341be1c1 (the registered pair).

4. DIFF AND STATUS
`git diff claude/mesozoic-trex-stage2-refine-4jffre --stat`:
 hardware/trex_form_study/stage2/tb/colab_steps.txt | 34 +++++---
 hardware/trex_form_study/tests/test_stage2_nb.py   | 90 +++++++++++++++++++++-
 2 files changed, 110 insertions(+), 14 deletions(-)

`git status --porcelain`: empty.
- `--ignored` shows only the old input directories and .ruff_cache, which was there before.
- The main checkout's status is empty.
- The pytest basetemp has been deleted.
- What is left in /tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/wave8/cn4/: mutate.py, mutate_nb.py, mutants.log, old_sheet.log, new_sheet.log, full_nb.log, full_nb_merged.log, msg1.txt, msg2.txt.

5. COMMITS, OLDEST FIRST
- def58b7f69f212a8db30055fef0de893b1d78e24: the sheet and the tests (ruling 34).
- f9019d17: merge of the working branch at e4b94077 (parents def58b7f, e4b94077).
- Both messages end with the two given lines and name no model anywhere else.
- Branch head: f9019d17 on wt/wpc-cn; its tree is eb875442978adc26641ac55019afa03f7ba46292. Nothing has been pushed.

NOT DONE
- The STAGE2_SLOW run.
- No other test files were run.
- Nothing checked on Colab: the Clear all outputs menu, how a running box looks, and what a reopened tab shows are all marked expected, not yet seen.
- The sheet has not been viewed on a phone.
