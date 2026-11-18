# check:CN (roundC, 17 Nov, second attempt): the agent's final report

Working paper. Nothing here is a rule.

VERDICT: merge as it stands. All three majors and the minor points are handled as ruling 1 to 6 settle them. Each fix I reverted on a scratch clone made its named test fail, apart from two cases noted below. I found four minor points; fix them in a small follow-up before the acceptance run. None blocks the merge.

Note: the working branch moved after the fixer's report. It is now at 7df517c1 (the rows package CW merged as f64094e2, plus a note added to the 17 Nov rulings that does not touch CN). I checked against that tip.

FINDINGS (most serious first)

1. MINOR, a rule case the code decides differently (ruling 2, rule M1). nb.py:1345, `retry_check = bool(p["RETRY_CHECK"]) or ld.get("retry_check") is True`, and nb.py:2420.
   - Problem: LEAD.json's retry_check, written to answer one halt, is applied to every later session.
   - Evidence, repro/exp8_retry_persists.py (a) on HEAD:
     - Session 0 halts (remedy ['retry_check']).
     - Session 1 answers it with LEAD.json and is then refused by the trainer.
     - Session 2 is a plain resume and answers no halt: "halt_resumed None | retry_check True". Trainer call 2 gets "--resume True, --retry-check True".
   - With the real trainer (train2.py:1281-1285), a later check whose two attempts were never closed then gets one more attempt each session instead of halting for the lead. This undoes the fixer's own aim (for-the-lead b: "an old LEAD.json cannot reopen a later halt").
   - Fix: take LEAD.json's retry_check only in the session that resumes a halt whose remedy names it (`halt is not None and "retry_check" in halt["remedy"]`). Add a test that a later plain resume carries no --retry-check. Or the lead rules that it persists, and LEAD_RULE says so.

2. MINOR, the steps sheet (ruling 1 against section 5). colab_steps.txt:214-218.
   - Problem: following section 5 deletes a runtime that cell 8 kept on purpose.
   - Evidence, exp8 (b): after an UNFINISHED launch and two finish runs that kept the runtime (exit codes [6, 6]):
     - STATUS.json written_at stayed at the launch's end (03:04:46Z), with state "running".
     - LIVE.json is closed.
     - Only SESSION_0's last_status_at moved.
   - So 5.3 says "More than 10 minutes old ... Go on with 5.4", and 5.4 starts "Runtime > Disconnect and delete runtime". An owner whose browser lost the page during the kept time deletes the only copy of the last save.
   - The exception is named only in section 6's last paragraph (line 300).
   - Fix: in 5.2 or 5.4 add: "If the last box printed THIS MACHINE IS KEPT ON PURPOSE: do not do 5.4; run box 8 again as in 6 and tell me."

3. MINOR, a test gap (ruling 1). nb.py:2152 stores the pack tool's try count in the session state, and no test covers it.
   - Evidence: with that line removed, the whole fast suite still passes (mutant 1i: 60 passed, 2 skipped on the clone).
   - What it protects, repro/exp9_pack_fails.py with a pack tool that always fails:
     - HEAD: finish exit codes [0, 0, 0, 0].
     - Without the line: [6, 6, 6, 6]. A tar that can never be packed would keep the runtime for good.
   - The fix is right. It needs this test.

4. MINOR, the steps sheet. colab_steps.txt:270-272 (from 36347d4).
   - The text "Again and again: your Drive may be full. Empty the Drive's bin first ..., then tell me" sits on the "reading from Drive failed" row. A full Drive makes writes fail, not reads.
   - It also asks the owner to empty the whole Drive bin before telling the lead. That is an irreversible deletion, and it goes against the sheet's own rule (lines 16-17).
   - Fix: one row per warning, and "tell me; I may ask you to empty the bin."

FOR THE LEAD
- (Fixer's c.) Cell 8 releases the runtime on any finish exit code but 6. That includes a crash of the step and a KeyboardInterrupt; the test asserts release on "raise". Ruling 1's reasoning favours keeping it on anything but 0 and 5.
- On Colab, if the Drive mount itself died (errno 107), running cell 8 again does not remount it. The paragraph does say "tell me". Cell 8 could try drive.mount(force_remount=True). This is Colab behaviour, expected and not yet seen.
- (Fixer's g.) Step 6 fakes only QUEUE.json's commit. The real trainer's own source check is never exercised, and `_remedies`' checks do not assert that --allow-source-change reaches the trainer; only the stub test's argv does.
- The fixer's points a, b, d, e, f, h, i, j are defensible readings.

CHECKED AND FOUND RIGHT
- Reverting fixes in HEAD on a scratch clone: 45 reverts, 43 caught by the named test.
  - Ruling 1: the waits, finish's copy, finish's DONE.json, the order 6 before 5, release_or_keep, finish never raises, hours kept_runtime, finishes recorded, cell 8.
  - Ruling 2: exit 4 is no halt; resume_halt, the remedy flags and the remedy detection are required; HALT_<k>.json is kept; a launcher error is resumable; crashed twice is final; LEAD keys; _cmd's retry_check; budget_by into QUEUE.json; both allows from LEAD.json; _say; the first beat; the copy back.
  - Ruling 3: the marker check, the marker at start, the old skip rule, the guard.
  - Ruling 4: seams in session and launch, the dirty Colab checkout, a closed session, the torch cpu build, apt-get update, unassign, another bundle on line 2, the bundle id and content hash checks, collect's logs/, the dead VM's local inputs.
  - Ruling 5: QUEUE parsing and cell 1's text field.
  - Step 6: acceptance's call to `_remedies`.
  - The two that were not caught: 1b (settle without its own fence) is equivalent, because every copy fences itself (nb.py:2082/2084); 1i is finding 3.
- The review's own scripts (byte-identical copies) run on HEAD:
  - exp1: Drive holds [1024, 2048, 3072, 4096], and ckpt_4096.tar is there.
  - exp2: after exit 4, no HALT.json; after exit 5, every resume without LEAD.json is refused.
  - exp3: no HALT.json, DONE, 204800 of 204800 steps.
  - exp4: unfinished, no DONE.json or HALT.json, finish does not raise.
  - exp5: (a) done with no halt; (b) "copied back", resumed_from [None, 3072].
  - exp6: release_runtime calls ['flush', 'unassign']; launch after a failed line 2 is refused.
  - exp7: (a) exit code 6; (b) "released: False"; (c) done, on_drive True, DONE.json True, 4096 on Drive True.
- The notebook diff is as described (13 parameters, QUEUE a string field, cell 8 calls release_or_keep).
- The sheet is ASCII, at most 115 columns except the URL, has no "inferred" and no ALLOW_.
- Only the 5 package files differ from the working branch.
- Hash pair on HEAD and on the merged tree: 45719d030359ec7d5d3914af3814d837fdc05d77652ba45c64792b511027528d c0027992de21067525cc4489b74acb157feb89f3a4c7618580addc2b341be1c1.
- merge-tree --write-tree against 7df517c1: exit 0, tree 8ca8b3ce, the 5 files only (all added).
- The merge 5bdd5aa8 touched only stage2/design documents (12 files).

TEST SUMMARY LINES (lowprio, standing environment, scratch basetemp)
- Worktree HEAD fast suite: "61 passed, 1 skipped in 57.50s", exit 0.
- Worktree HEAD slow suite (STAGE2_SLOW=1 -k local_run): "1 passed, 61 deselected in 268.14s (0:04:28)", exit 0.
  - Record: ok, 23 of 23 checks.
  - Killed after 52.9 s at "checkpoint 1024 of R5-75-M-H-10-s42 is on the stand-in Drive".
  - The 6 step-6 checks are all true, real trainer.
  - Line 4: 506 and 624 env steps/s. Tiny jobs on a shared machine: provisional.
- Merged tree (7df517c1 plus HEAD), fast: "61 passed, 1 skipped in 67.26s (0:01:07)".
- Merged tree, slow: "1 passed, 61 deselected in 343.33s (0:05:43)".
  - 23 of 23 checks and 6 of 6 step-6 checks.
  - The monitor now reads "episodes_by: lookbar.episodes".
  - 4 real clips were made (ok true; ffprobe 3.03 s).
- Scratch clone baseline (no Stage 1 arrays): "60 passed, 2 skipped in 76.12s (0:01:16)".
- ruff: "All checks passed!"
- Episodes run: only tiny R5/R6 jobs on the S1 view of 75-M-H-10 (two slow runs with their step-6 job) and stubs. None on a look set.

COMMITS TO BRING IN
- All 20 commits not in the working branch end with two attribution lines: 12 with the earlier model's line, as briefed, and 8 of this round with the 17 Nov lines.
- None of this round's 8 names a model elsewhere.
- So: one merge of the tip, a003c31e566601d1f0e248aeb8342874b946eb88.

NOT CHECKED
- The acceptance run (as briefed), and anything only Colab shows (mount, remount, disconnect, idle timeouts, red output, the form's quoting).
- The `nb local-run` command itself: I ran the slow test, which calls local_run.
- HEAD's tests against each parent commit (I reverted each fix in HEAD instead).
- The pre-fix reproductions on 73a5112 (I relied on the review's output).
- The sheet on a phone.

HOUSEKEEPING
- My first hash check ran without PYTHONDONTWRITEBYTECODE and created three ignored __pycache__ folders under designlab in the worktree. I removed them. The worktree's `git status --porcelain` is empty, and no tracked file changed anywhere.
- My scripts and logs are in /tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/rerun17/check_cn_17nov/:
  - mutants.py, mutants2.py, mutants.log, mut/, mut2/
  - repro/exp8_retry_persists.py, repro/exp9_pack_fails.py, repro_head.log
  - head_tests.log, slow.log, merged_fast.log, merged_slow.log
  - clone/: the scratch clone
