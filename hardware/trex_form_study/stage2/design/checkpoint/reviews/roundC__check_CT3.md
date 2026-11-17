# check:CT3 (roundC, 17 Nov): the agent's final report

Working paper. Nothing here is a rule.

VERDICT: Nothing to merge. The CT third follow-up has not been built. wt/wpc-ct adds nothing to the working branch: `git merge-tree --write-tree` exits 0 and gives tree d2edf58, which is the same as the tree at the working branch's tip. The builder chose not to build because the user's request only asks for a summary of recent work and an update on when training can start; a build with merges and commits would go beyond that. That reading is consistent with the harness rule, and I agree with it. Everything I did was read-only. I changed no tracked file and made no commit. My only output is a log in the scratchpad.

FINDINGS (most serious first)
1. Minor (for the user's answer): the summary's state is one commit out of date. The working branch tip is now c7902f2 (17 Nov 01:19:04 UTC), not 32ff20b (01:13). That commit adds a block to STATUS.md with the owner's screenshot from about 01:17 UTC:
   - 57 % of the week used, resets Saturday 3:00 AM (owner's time)
   - Fable is capped at 50 % of the week and has used 97 % of that cap
   - the new model draws on the remaining 43 % of the week
   - 1 % of the week is about $13.6 on the meter
   - the owner holds one reset
   - the lead pushes at 85 % and at 95 %, and recommends using the reset at about 95 %
   So the open point "how much of the allocation is left" is now partly answered, and the "how soon" estimate should cite this budget. My own arithmetic, not from the file: 43 × $13.6 ≈ $585 on the meter. Commits since 14 Nov are now 396.
   Fix: add this block to the answer to the user.
2. Minor (out of date): wp-l has moved since the summary.
   - Now: head 75d4303, a merge of the working branch made at 01:20:43 UTC, 18 commits ahead.
   - Summary says: b72c61c, 17 ahead.
   - stage2/sets/V6/60-N-T-10/ is still untracked.
   So the look-set build looks restarted, not just cut. I did not check which agent is working there.
3. Info (exactly what is still owed; the summary is right that it is open):
   - (a) Open. train2.py:1131 starts the monitor child with `tbenv.child_env("disable")` and does not set STAGE2_ALLOW_SOURCE_CHANGE. In wpc-cw, s2rows.py:59 reads that variable and s2rows.py:95 defines `RowError(s2io.S2Error)`.
   - (b) Open. monitor2.py does not mention RowError.
   - (c) Partly in place already. monitor2.py:686-707 raises MonitorRefusal("checkpoint") when the record is missing, cannot be read, or is not a JSON object. Still owed:
     - A missing run.json: monitor2.py:667-669 raises a plain MonitorError, which exits with 1. tests/test_stage2_monitor2.py:324-328 asserts exit 1 for that case, so that test must be reversed, not just added to.
     - A wrong-shape `files` value: it raises AttributeError (finding 3 of roundC__review_followups2.md).
     - A fingerprint that does not reproduce when the row loads it.
   - (d) Open. The halt text at train2.py:1282 is unchanged.
   - (e) Open. No test kills the trainer around a refusal.
4. Low (risk): the reviewer's three scratch kill tests that (e) should take in exist only in the session scratchpad, at /tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/review_followups2/ct_copy/hardware/trex_form_study/tests/test_stage2_train2.py, lines 2078, 2106 and 2141. A scratchpad cleanup would lose them. The next CT build should be pointed there.

CHECKED AND FOUND RIGHT
- The worktree is clean (also after my test runs). Head is 2fca27b on wt/wpc-ct. No commit was made this round.
- 57b5bf3 and 2fca27b are not on the working branch. Their trees are identical to 0ad2072 and bf9bc43 on the working branch (empty diff).
- CT's files (train2.py, monitor2.py, clip2.py and their tests) are identical on the branch and on the working branch.
- The record-key hashes are the registered pair: 45719d03...528d c0027992...c1c1.
- I spot-checked the summary's claims and they hold:
  - Rulings 10 and 11 of 17 Nov and the sixth list match the files.
  - STATUS.md: 97.3 % at the stop, $2,700.74 at 21:19, $2,701.94 at the resume, the five maker questions, F28/F29/E/F30 still open.
  - P32 to P41 are in DECISION_LOG.md.
  - CP was merged on 15 Nov at 04:02.
  - wpc-cw is at 9952f5b, 8 ahead, with 4 modified files.
  - wpc-cn is at b54c0aa, 12 ahead, clean.
  - The design's totals (ROUND_C_DESIGN.md:1625-1627) are 6.5 core-hours for validation and S2-4, 5.2 for the 60 cm run and 8.0 for R2L. V12, the acceptance run, takes about 12 min.
  - The "1 to 2 sessions" estimate is the builder's own and is labelled that way.

TESTS (run in the CT worktree through tools/lowprio.sh with `-o addopts=""`; they test the unchanged code)
- tests/test_stage2_monitor2.py: `23 passed in 33.07s`
- tests/test_stage2_train2.py: `48 passed, 3 skipped, 2 warnings in 178.55s (0:02:58)`
- Log: /tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/check_ct3_17nov/train2_tests.log

COMMITS TO BRING IN: none. Both commits not on the working branch end with two attribution lines (the earlier model's, which were right when they were made). Their content is already on the working branch, so merging the tip would only add an empty merge commit.

NOT CHECKED
- No build, and so no mutation tests.
- No full defect read of train2.py and monitor2.py beyond the items above.
- Each item in the 15 to 17 Nov summary was not traced to its commit; I only spot-checked.
- What is running in wp-l.
- Tests on a merged tree (it would be identical to the working branch).
