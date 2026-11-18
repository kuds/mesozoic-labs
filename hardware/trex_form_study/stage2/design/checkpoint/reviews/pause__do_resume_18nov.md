Item C is done. RESUME.md is written, the two pointers are added, and the result is committed on `wt/pause`. Nothing was pushed, tagged or uploaded, and no pull request was opened.

**Commit:** `cd0c92f8c53da9602b786732337bda851f419f60` on `wt/pause`, with parent `54d8afc1` (item A's commit). It changes 3 files, all in `/home/user/mesozoic-wt/pause/hardware/trex_form_study/`:
- `RESUME.md`: new, 288 lines.
- `README.md`: +3 lines. One bold line under the title says the study is paused (18 Nov 2026, P47) and links to RESUME.md and DECISION_LOG.md. RESUME.md is now the first row of the "Read first" table.
- `NEXT_SESSION_PROMPT.md`: +6 lines. A dated note "Note of 18 Nov 2026: the T. rex study is paused; a restart starts from RESUME.md" sits right under the title. The rest of the file is unchanged.

The commit message ends with exactly the two required lines and names no model anywhere else. Both the worktree and the main checkout show 0 lines of `git status --porcelain`.

**Links:** all 63 relative links in RESUME.md, the 23 in README.md and the 1 in NEXT_SESSION_PROMPT.md resolve in the tree (checked by script). Paths that exist only on other branches are written as plain text with their branch named, not as links: the archive's `stage2/archive/a1/README.md` and s2-b1's `parts/*.tar`.

**RESUME.md's sections:**
1. For the owner: why it stopped, what it reached, what is kept
   - Why it is paused
   - What the study reached (all in a computer model: nothing was built or measured)
   - What is kept, and where
2. How to restart, step by step
   - Step 0: check that the branches are there
   - Step 1: a fresh session on the working branch
   - Step 2: the environment
   - Step 3: the inputs, from the archive branch
   - Step 4 (only if needed): the unfinished side branches
   - Step 5: the bit-identity check
   - Step 6: the tests
   - Step 7: the next step, the owner's day-one Colab smoke
3. What was open at the pause (a table: item, state, pointer)
4. What carries over to other builds (a table: what, paths, what it gives)

**Checks I ran myself, in the `wt/pause` worktree, with the main checkout's venv and no inputs restored:**
- The hash one-liner (`bar.bar_sources()['sha256']`, `panels.panel_sources()['sha256']`) printed `45719d030359ec7d5d3914af3814d837fdc05d77652ba45c64792b511027528d c0027992de21067525cc4489b74acb157feb89f3a4c7618580addc2b341be1c1` and exited 0.
- `python -m designlab.stage2.regress check` printed "stage2/regress/regress.json: current (full tier, 256 of 256 episode runs matched, 248 distinct committed episodes, the four source hashes are today's)" and exited 0.
- RESUME.md gives both commands and their expected output.
- On origin, `claude/mesozoic-trex-stage2-bundle-s2-b1` is at `ecf95a81a6eb…`. Its two V6 parts hold 5 + 12 = 17 set arrays at study paths.
- C1 `cc437b6a` is an ancestor of `b1d5e98e`.
- `stage2/tb/validation/lr_probe.json` has the -X head added (rate 1e-5).
- `stage2/tb/r2l/index.json` lists no fits.

**What I did not do, and points for you:**
- **P51 is mentioned.** It is not in `wt/pause`'s decision log, because the pushed working branch is at `c5274905` (P51), one commit after `b1d5e98e`. RESUME.md says the T. rex stays on the roadmap as the flagship and that P51's row is in the log from `c5274905` on. Step 1 tells a restarter to read P40 to P51, which holds once `wt/pause` is merged onto the working branch.
- **Some statements are true only after you push.** RESUME.md says "It is all in the GitHub repository". That holds only once `wt/pause` is merged and pushed and `claude/mesozoic-trex-stage2-archive-a1` is pushed (it is local only now). Step 0 tells a restarter to check this, with a fallback to `restore_inputs.py restore` if the archive branch is missing.
- **The archive gap is recorded, not fixed.** RESUME.md describes the 52 frozen arrays that are only in worktrees here (V6 look files, V6 on 75-M-H-10-X, V6noF5). I did not add them to the archive.
- **Group j is already done in code.** It has been in `tools/restore_inputs.py` since `f5a0bf5e`, with 30 files, all of which are in the archive. What remains is that its rebuild has never been timed in a fresh worktree. RESUME.md says this instead of listing group j as unbuilt.
- **"Not tried" items.** RESUME.md labels these as not tried: copying the 17 s2-b1 arrays in with `copy-from --group sets`, and `restore --group sets` for the three newest sets.
- **No test suite was run.** No full-suite count after 14 Nov (679 passed, 107 skipped) is recorded in the study, and RESUME.md says so.
- **Additions beyond your list, from STATUS.md:** R2L fits (V10) are parked; the draft rows F28, F29 and E, and F30 (owed, not yet drafted); the compute choice; P25.

Files:
- /home/user/mesozoic-wt/pause/hardware/trex_form_study/RESUME.md
- /home/user/mesozoic-wt/pause/hardware/trex_form_study/README.md
- /home/user/mesozoic-wt/pause/hardware/trex_form_study/NEXT_SESSION_PROMPT.md
