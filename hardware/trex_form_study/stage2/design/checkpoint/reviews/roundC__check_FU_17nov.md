# check:FU (17 Nov): the agent's final report

Working paper. Nothing here is a rule.

VERDICT
- wt/wpc-cn (notebook): merge as it stands. Tip a557af92eb67141aaf8e98aba768a722de74f71e.
- wt/wpc-ct (trainer): merge as it stands. Tip d1def04012a9311738877f7fce920bb1aa1a8c74.
- wt/wpc-cw (rows): merge as it stands. Tip 097bfd64c52531e5f5d09c8b0bf0627af4b21c15.
- I found no blocker and no major defect. The items below are minor points, test gaps and choices for the lead. The two test gaps (2 and 3) are a few lines each and can go in before the acceptance run or with the next change.

FINDINGS (most serious first)

1. MINOR, for the lead. CN steps sheet, colab_steps.txt:217-224 (5.3/5.4).
   - Ruling 12 says the sheet "never" tells the owner to delete a runtime that cell 8 kept on purpose. The new exception in 5.3 works only if the owner can still see the last box's output.
   - Section 5 is the "browser shows DISCONNECTED / not sure" path. When cell 8 keeps the runtime, the finish does not move STATUS.json `written_at` (nb.py:3218-3230 records kept_runtime only in the session record). So an owner who lost the page reads "More than 10 minutes old" and goes on to 5.4, "Disconnect and delete runtime".
   - The text is what the CN check itself proposed, and the builder flagged it (point g).
   - Options:
     - (a) In 5.3, before 5.4: reopen the notebook tab; if the last box's output cannot be seen, tell me and do not do 5.4.
     - (b) A best-effort Drive marker when the runtime is kept. Design 3.9 allows STATUS.json no fourth state, so it would be a separate file.
     - (c) Accept as built.

2. NIT, test gap. CN nb.py:1405-1406.
   - The condition `"retry_check" in halt["remedy"]` has no test of its own.
   - My mutant n1, `(halt is not None and ld.get("retry_check") is True)`, survived: "3 passed, 63 deselected".
   - Case it lets through: LEAD.json still holds retry_check from an earlier halt and now answers a halt whose remedy is only allow_source_change; the session would pass `--retry-check`.
   - Fix: in test_a_halt_for_moved_sources_asks_the_lead_for_allow_source_change (tests/test_stage2_nb.py:1145), also write retry_check true into LEAD.json, then assert s1["retry_check"] is False and no `--retry-check` in argv.

3. NIT, test gap. CT train2.py:368, the fallback in `allowed_sources`.
   - The fallback (a session with `source_change` True, for a run.json without `sources_allowed`) is untested.
   - My mutant t6, which drops the `source_change` part, survived: "2 passed, 54 deselected".
   - Fix: give one session in test_the_resume_takes_an_allowance_from_the_runs_record (tests/test_stage2_train2.py:1634) `source_change` True without `allow_source_change`. Or drop the fallback: no real run.json predates the key.

4. FOR THE LEAD, outside ruling 12's text. LEAD.json's `allow_source_change` still holds at every later session (nb.py:1358; ruling 2).
   - So under the notebook, every later session passes `--allow-source-change` for any sources. That is wider than ruling 12's "for the same sources". The trainer's own allowance (`sources_allowed`) does what the ruling asks.
   - In practice it is low risk: on Colab the sources move only with a new REPO_REF, and a modified tracked checkout is refused.
   - Option: take LEAD.json's `allow_source_change` only in the session that answers a halt or refusal, as for retry_check, and let the trainer's `sources_allowed` carry it on from there.

5. NOTE, CW lookbar.py:232 (`_ABS_PATH`). A path with a space is still cut only up to the space:
   - "/content/drive/My Drive/mesozoic/..." becomes ".../drive/My Drive/mesozoic/...".
   - The first part is no longer kept, which is what low point 2 asked for. The space case was noted by the check but not asked for; the study uses MyDrive. No action needed.

6. The builder's points a-h: I agree they are defensible. On d: `moved_sources` recognises the row's refusal by its words ("rule K9" or STAGE2_ALLOW_SOURCE_CHANGE). Both RowError texts in s2rows.py (lines 263 and 331) match. A dedicated marker in s2rows would be the cleaner fix later.

CHECKED AND FOUND RIGHT
- Ruling 12 and the checks' points, read in the code:
  - CN finding 1: LEAD.json's retry_check now applies only in the session that resumes the halt whose remedy names it. A parameter-file RETRY_CHECK still applies.
  - CN halt remedies: `halt_remedy` maps `--retry-check` to retry_check and `--allow-source-change` to allow_source_change.
  - CN release: `release_or_keep` lets the runtime go only on finish codes 0 and 5. On 6 it prints KEEP_RUNTIME_TEXT; on any other code, None or a non-int it prints `keep_other_text`. Both begin "THIS MACHINE IS KEPT ON PURPOSE". `_step_finish` itself returns only 0, 5 or 6.
  - CN remount: cell 8 calls `remount_if_dead(DRIVE, ...)` before the finish. It acts only on ENOTCONN, at most once, and never raises (an Exception is caught; a KeyboardInterrupt goes to the finally block, which keeps the runtime).
    - Wording only: DRIVE is the stage2 folder on Drive, not the run's directory as the builder's report says.
  - CN step 6 (`_remedies`): it fakes the r5core hash and recomputes the sources hash exactly as `train2.sources()` does (sha256_json of files, plant_sources, panel_sources). The source check comes before the version check in `_refuse_sources_versions`. The two new checks can tell the flag reached the trainer.
  - CN sheet findings 2 and 4: done. The sheet is ASCII, and the only line over 115 columns is the existing URL.
  - CT finding 1: `_checkpoint` refuses a record whose layout or fingerprint is not an object. An empty object reaches s2load's own CheckpointError, which is already a refusal. `save_checkpoint` always writes both.
  - CT allowance:
    - `_refuse_sources_versions` runs on every path to `_begin`, so `source_allowance` is set before any monitor child.
    - `_child` (baseline and checks) passes the allowance to the row.
    - `sources_allowed` is written in `_begin` before the first write, so it survives a kill after the session's record. A session refused for versions records nothing.
    - s2rows' K9 compares a subset of what the trainer's hash covers, so an equal trainer hash means an equal row comparison.
  - CT halts: a halt at a check and at the baseline for moved sources ends with SOURCE_REMEDY.
  - CT findings 3 and 4: done.
  - CW: all three low points done; the summary's refusal texts are relative to the output root.
- Mutants, my own, on scratch copies (a copy of each worktree, with its data files hard-linked; the notebook copy got a git index without a checkout for step 6). Each selection passed unmutated first:
  - rows: 8 of 8 caught.
  - trainer: 9 of 10 caught; t6 survived (finding 3).
  - notebook: 8 of 10 caught; n1 survived (finding 2), and n2 (dropping the type check in release_or_keep) survived as equivalent for every reachable code.
  - My first run of n6 and n10 was void (step 6 needs git; the unmutated baseline failed in that copy). Re-run with git: "cn_baseline_git OK | 8 passed, 58 deselected", then n6 and n10 CAUGHT.
- Merged tree: the notebook copy with CT's and CW's files laid over it, which matches the merge into fccebe42 in code (fccebe42 adds only STATUS.md).
  - The slow step-6 run there: LOCAL_RUN.json ok true, 25 of 25 checks, all 8 step-6 checks true.
  - The real trainer refused with "other sources than the run's last session (cd7efafb1208... -> 8cc6265842ee...; files: designlab/stage2/r5core.py): refusing to resume (pass --allow-source-change ...)".
  - The run.json's last session has source_change True, version_change True and source_allowance by --allow-source-change. sources_allowed records that hash for session 1.
  - The resumed checks (monitor_2048, monitor_4096) ran through lookbar.episodes with flags.source_change True.
- Only package files differ:
  - `git merge-tree --write-tree` against fccebe42 exits 0 for each branch: cn tree 04547598 (nb.py, ipynb, colab_steps.txt, test_stage2_nb.py); ct tree 8e779874 (monitor2.py, train2.py, two tests); cw tree 02b27dc1 (lookbar.py, test_stage2_lookbar.py).
  - The packages' files do not overlap.
- Record-key hash pair, in each worktree and on the merged copy: 45719d030359ec7d5d3914af3814d837fdc05d77652ba45c64792b511027528d c0027992de21067525cc4489b74acb157feb89f3a4c7618580addc2b341be1c1.
- Nothing ran that should not have:
  - No new file under stage2/tb/gates or stage2/tb/validation in the main checkout or any worktree.
  - `git status --porcelain` is empty in all three worktrees and in the main checkout. The ignored __pycache__ in ct and cw dates from 16 Nov.
- ruff --no-cache on the changed .py files: "All checks passed!" for each branch.

TEST SUMMARY LINES (mine, standing environment, tools/lowprio.sh, basetemp in scratch)
- wpc-cn tests/test_stage2_nb.py: "65 passed, 1 skipped in 59.61s"
- wpc-ct tests/test_stage2_train2.py: "53 passed, 3 skipped, 2 warnings in 221.60s (0:03:41)"
- wpc-ct tests/test_stage2_monitor2.py: "25 passed, 1 skipped in 31.88s"
- wpc-cw tests/test_stage2_lookbar.py tests/test_stage2_s2rows.py: "53 passed, 11 skipped in 126.83s (0:02:06)"
- Merged copy, tests/test_stage2_nb.py: "65 passed, 1 skipped in 84.39s (0:01:24)"
- Merged copy, tests/test_stage2_monitor2.py: "25 passed, 1 skipped in 33.17s"
- Merged copy, STAGE2_SLOW=1 -k local_run: "1 passed, 65 deselected in 360.26s (0:06:00)". This is the same tiny development run the builder and the earlier CN check ran: tiny jobs on the S1 view of 75-M-H-10, nothing on a look set.

COMMITS / WHAT TO BRING IN
- Every commit not in the working branch ends with the two given lines, and none names a model elsewhere. All have one parent; there is no merge commit.
  - cn: b0b8a798, ba554805, bf48aff7, 7c8ceba4, a557af92.
  - ct: fd1f88ff, a24a9c6a, e09577f3, d1def040.
  - cw: 762befff, a8444f03, 097bfd64.
- Bring in one merge of each tip into claude/mesozoic-trex-stage2-refine-4jffre, in any order: a557af92eb67141aaf8e98aba768a722de74f71e, d1def04012a9311738877f7fce920bb1aa1a8c74, 097bfd64c52531e5f5d09c8b0bf0627af4b21c15.

NOT CHECKED
- The acceptance run, and anything only Colab shows: the remount, ENOTCONN against EIO or ESTALE, a hang on a dead mount, the auth window.
- The trainer's and the rows' STAGE2_SLOW suites.
- train2, lookbar and s2rows tests on the merged copy (I ran them on their own heads only).
- Pre-fix reproductions on the old tips (I used mutants instead).
- The sheet on a phone.

HOUSEKEEPING
- One ad-hoc `python -c` in wpc-cn ran without PYTHONDONTWRITEBYTECODE and created two ignored folders, designlab/__pycache__ and designlab/stage2/__pycache__, at 04:38:41. I removed both; the status is clean.
- The scratch copies are deleted.
- Kept in /tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/wave5/check_fu/:
  - Scripts: run_tests.sh, run_merged.sh, mkclone.sh, mutate.py, specs.py and the spec_*.json files.
  - Logs: mutants.log, cn.log, ct_train2.log, ct_monitor2.log, cw.log, m_nb.log, m_mon.log, m_slow.log, mt_{cn,ct,cw}.txt.
  - The step-6 record: kept/LOCAL_RUN.json, kept/remedies_run.json and kept/remedies-*.log.
