# check:CN (roundC, 17 Nov): the agent's final report

Working paper. Nothing here is a rule.

REVIEW OF THE CN FIX ROUND (wt/wpc-cn at b54c0aa0c6150db59c7ce9f99b76c31909ddd824, against claude/mesozoic-trex-stage2-refine-4jffre at c7902f2)

VERDICT: DO NOT MERGE. There is no fix to check. The fixer did no fix round. It read the relayed user request as asking only for a status rundown and an update on when training can start, so it did read-only work. I agree with that reading, and I also kept to read-only checks. Every claim in its status report that I checked holds. Three things are slightly off; none of them is serious (findings 4 and 5). The package still has the review's majors. I reproduced major 2 on the branch head.

FINDINGS, most serious first

1. MAJOR, still open (ruling 2): a trainer exit 4 or 5 still writes a final HALT.json, and every resume is refused.
   - Code: designlab/stage2/nb.py 2398-2401 (exit 4 or 5 raises _Halt), 2546-2548 (any launcher error becomes "halted"), 1242-1245 (a run with HALT.json is never resumed).
   - Missing: LEAD.json, a halt marked resumable, and HALT_<k>.json (grep finds none of them).
   - Acceptance step 6 is still listed as not done (nb.py 3480).
   - Evidence: I reran the review's exp2_halt_final.py on b54c0aa, from a scratch copy.
     - After exit 4: "launch state halted; HALT.json: True". Each of the four resumes (plain RESUME, ALLOW_VERSION_CHANGE, ALLOW_SOURCE_CHANGE, TAKEOVER+RETRY_CHECK) gives "REFUSED: the run is finished (HALT.json: job R5-B1-s42 was refused by the trainer (exit code 4) ...".
     - After exit 5: the same result.
   - Fix: as ruling 2 says.

2. MAJOR, partly open (ruling 1).
   - What is built: at a job's end the copy is retried END_COPY_TRIES = 10 more periods (nb.py 118, 2143-2155), and no DONE.json is written without the last copy.
   - What is owed:
     - the fenced retry of at most 20 minutes with a growing wait (grep finds none);
     - no runtime release when the copy fails;
     - the plain paragraph to the owner.
   - The notebook was not touched by any fix commit. Cell 8 still calls `s2_nb.release_runtime(...)` in a `finally` block, so the runtime is always released.

3. MINOR, open (rulings 4 and 5).
   - S2_NB_SEAMS is refused only when IN_COLAB is true (nb.py 1226); a run that is not tiny is not refused.
   - A dirty checkout is only reported as "checkout clean: no" (nb.py 617), not refused.
   - QUEUE is still a raw list in the notebook: `QUEUE = []  # @param {"type":"raw"}`.
   - The steps sheet still uses "(inferred)" (lines 80-314) instead of "expected, not yet seen". At line 151 it says "Copy it from this sheet and paste it".

4. INFO, the fixer's report: the working branch now has 5 commits the CN branch lacks, not 4. The fifth, c7902f2 (01:19 UTC, STATUS.md only, 8 lines), came after the fixer's look. All five are documents only, and the merge stays clean (below).

5. INFO, for the lead: the computed brief says the earlier fixer was cut "after one commit". The reflog shows four fix commits, 21:10 to 21:17 UTC on 16 Nov (ef80550, cc6cef5, 36347d4, b54c0aa). The fixer's report has this right. STATUS.md's 01:13 line, "wt/wpc-cn one more commit than at the stop", does not match its own 21:20 block, which already shows b54c0aa.

FOR THE LEAD: should the fix round run? The relayed request asks only for a rundown and a training-readiness update. A fix round means a merge, edits and commits, which the user has not asked for. Options:
(a) Ask the user to confirm, then run the fix round as briefed (my recommendation).
(b) Treat the session's standing work plan as covering it. That is the lead's or the user's call, not a reviewer's.

CHECKED AND FOUND RIGHT
- Worktree: `git status --porcelain` is empty, and stayed empty after my runs. The ignored-file listing was the same before and after. The main checkout has 0 tracked changes.
- Head is b54c0aa; branch is wt/wpc-cn.
- Record-key hashes print the registered pair: 45719d030359ec7d5d3914af3814d837fdc05d77652ba45c64792b511027528d c0027992de21067525cc4489b74acb157feb89f3a4c7618580addc2b341be1c1
- Diff against the merge base fdb31d5: exactly the five package files (nb.py +3640, stage2_training.ipynb +245, colab_steps.txt +328, requirements-colab.txt +19, test_stage2_nb.py +2197).
- `git merge-tree --write-tree --name-only` against the working tip: exit 0, tree b27357c, no conflicts.
- The 5 working-branch commits (8dd6958, 387b4da, ef2339b, 32ff20b, c7902f2) touch only STATUS.md, ROUND_C_RULINGS_17NOV.md and the four CN/CW review and build reports.
- The fixer's per-ruling status matches the code:
  - ruling 3: met in ef80550;
  - ruling 4: largely met. release_runtime unassigns whatever the flush raises (2704-2727); apt-get update was added (872); torch's cpu label is checked (814-853).
  - ruling 6: nothing to change.
- The CW worktree has the 4 modified, uncommitted files the rollup names (lookbar.py, s2rows.py and their two tests).

TESTS (b54c0aa, through tools/lowprio.sh, basetemp in scratch)
- `pytest -p no:cacheprovider tests/test_stage2_nb.py`: "50 passed, 1 skipped in 47.52s", exit 0.
- The review's exp2 rerun: output under finding 1.
Scratch: /tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/check_cn_17nov/ (exp_common.py, exp2_halt_final.py, exp/, pt/, status_before.txt, status_after.txt).

COMMITS TO BRING IN: none yet, because the branch should not be merged. All 12 commits not in the working branch (ec17bba through b54c0aa, the merges 71912ca and 73a5112 included) end with two attribution lines. These are the earlier pre-17 Nov lines, which were right when the commits were made. After the fix round, bring the package in with a merge of the tip, provided every new commit ends with the 17 Nov lines.

NOT CHECKED
- The slow local_run test (STAGE2_SLOW=1, about 200 s of tiny training). I did not run it as part of a status check, and the fixer recorded no slow-run output for b54c0aa.
- The review's exp1 and exp3 to exp6 on the head.
- A mutation check.
- `acceptance`, and anything only Colab does.
- The rollup's wider claims (592 records, gate 1, merged packages). I saw them only in STATUS.md.

FOR THE USER'S ROLLUP: the fixer's training-readiness answer stands. No Stage 2 training run of any kind has been made, and no honest date can be given. Before the first Colab smoke, these remain:
- the CN fix round (majors 1 and 2 above are still open), then its merge;
- the CW fix round (1 major; 4 files uncommitted);
- the look sets L (build cut; no review, no freeze);
- the trainer follow-up (rulings 10 and 11);
- CF and CG (not started);
- the validation steps;
- the real bundle;
- the acceptance run;
- the sheet's final values.
