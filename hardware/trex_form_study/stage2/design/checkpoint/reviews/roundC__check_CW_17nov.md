# check:CW (roundC, 17 Nov): the agent's final report

Working paper. Nothing here is a rule.

VERDICT: do not merge as it stands. The fix round was not done. wt/wpc-cw is still at 9952f5b, which is the reviewed code with major finding 1 unfixed. The working branch has not been merged in, nothing is committed, and ruling 9 is not built. The earlier fixer's uncommitted four-file diff is right for findings 1 to 6, and each fix has a test that fails without it: I reverted each fix 18 ways on a scratch copy and every revert was caught. The package can be merged after the fixes listed below. The fixer's report is accurate apart from the working branch tip (finding 5).

FINDINGS, MOST SERIOUS FIRST

1. MAJOR: none of the fix round is committed.
   - Evidence: `git status --porcelain` lists 4 changed files (lookbar.py, s2rows.py and the two test files) and HEAD is 9952f5b. Those files were last changed 16 Nov 21:11 to 21:13. `find -newermt "2026-11-16 21:20"` in the worktree prints nothing, so the diff is the cut fixer's work and nothing since.
   - `git merge-tree --write-tree claude/mesozoic-trex-stage2-refine-4jffre HEAD` exits 0, tree 3fa907b8eb…, and differs from the tip by the four files only. But that is the unfixed code: bringing in the tip now brings in finding 1, a lost rule text in every row description, which is part of the record key.
   - Proposed fix: merge c7902f2 into wt/wpc-cw, keep the diff as it is, add ruling 9, and commit in small commits, each ending with the two current attribution lines.

2. MEDIUM: ruling 9 is not built.
   - Where: lookbar.companion_episode, lookbar.py:554-626; survival_s is taken from the bar episode at :619.
   - Evidence (my script scen_r9_and_paths.py, synthetic records): "R9: outcome fell | survival_s 20.0 | 'warned' in episode: False | 'fall_time_s' in episode: False | monitor2.survival_credit: 20.0". Ruling 9 asks for fall_time_s 0, a `warned` flag and no survival credit.
   - Proposed fix, inside CW's files: carry `warned` into the block from `out["mujoco_warnings"]` (or the "MuJoCo warning" fall_cause that tune.warned_fall writes). In the companion episode set warned=True, fall_time_s=0.0 and survival_s=0.0. Leave the sidecar's survival_s and the sidecar check at :808 alone.
   - Test: a warned bar episode gives warned True, fall_time_s 0 and monitor2.survival_credit 0; an ordinary fall keeps its survival.
   - For the lead, the alternative: monitor2 reads `warned` itself. That is CT's file, so it would go in CT's follow-up.

3. LOW: a fault can still carry this machine's paths (ruling 8 says "without this machine's paths").
   - Where: lookbar.py:446 records `Type: message`. `_reference` (:405-406) puts `Type: message` of whatever playback raised into the LookError that `_read` raises again (:466).
   - So a FileNotFoundError for a missing reference file reads "look reading: FileNotFoundError: [Errno 2] No such file or directory: '<ROOT>/stage2/sets/…'". I built that line with the same f-string; I did not drive it through a real episode.
   - The new test checks two synthetic messages that hold no path.
   - Proposed fix: make paths under the study root relative in the fault text, and add a test with a path in the message. Or the lead reads the ruling as asking only for the `Type: message` form.

4. LOW, for the lead: where a stale companion goes.
   - The companion is moved to `<companion>.stale/`. The bar's own convention is `<record>.superseded/` (bar.py:307, :1976-1978), and both bar.py:2810 and lookbar.collect (:922) skip such files by that name.
   - A later reader of `look/` built the same way would take moved-aside companions as current.
   - Proposed fix: use `.superseded`, or name `.stale` in RULES and in every reader.

5. LOW: one inaccuracy in the report.
   - The report says the working branch tip is 32ff20b, 3 commits ahead of 8dd6958, with merge tree b2be02ae…. That tree is the merge with 32ff20b.
   - The tip is c7902f2 (17 Nov 01:19 UTC, STATUS.md only), 4 commits ahead. Merge-tree against it: exit 0, tree 3fa907b8eb…. The report's "6 documentation files, none of the package's" still holds.

Notes for the lead, not defects:
- The state-A git test leaves barreport.py out of design D18's own list (ROUND_C_DESIGN.md:374). The working branch changed barreport.py after c377ba1 (b7ebe91, 6184f77, 599b837), so D18's command as written is not empty there. barreport.py is in no record key. I would keep the exclusion; the design line needs correcting.
- The new SET_RULE text is in no row description. The renamed key changes the LookRow description, which is what ruling 7 intends. I did not check every directory for an existing Track B record.

WHAT I CHECKED AND FOUND RIGHT
- I read the diff against the review and rulings 7 and 8:
  - Ruling 7: `reference` is renamed `reference_use`; `rule_texts()` refuses a shared key; describe() and companion() use it; there are tests for both.
  - Finding 2: a hash-pair test, and a git-diff test against c377ba1 that skips where there is no git.
  - Finding 3: the K4 check at the episode is tested with roots A and B.
  - Finding 4: a companion of another bar record is moved aside, one that names the current bar record is kept, and the summary gains `refused_records` and lines in summary_text.
  - Finding 5: the fault is recorded as `Type: message`.
  - Finding 6: set_check refuses task_names that are missing, empty or not a list, and plant_inputs that are missing or empty; build() runs set_check (:264) before it reads task_names (:277).
  - For the refusal path: os.replace stays within one directory, so it is atomic, and the summary is written after it. A kill in between leaves the moved file and a "none" cell on the next collect.
- Each fix with a test that fails without it: on a scratch clone (a `git clone --shared` of the worktree, plus the fixer's patch and the ignored inputs), the 10 new or changed tests pass as they stand, and all 18 reverts fail them.
  - f1: full revert; only the rename reverted (the guard fails closed); guard removed; describe and companion bypassing rule_texts.
  - f2: bar.py, s2io.py and stage1/plant.py each changed.
  - f3: _check_files switched off.
  - f4: no refused cells; never moved; always moved; an unreadable file kept; summary_text silent.
  - f5: the traceback put back.
  - f6: task_names open; no type check; plant_inputs open.
- The record-key hashes are the registered pair, at the start and the end: 45719d030359ec7d5d3914af3814d837fdc05d77652ba45c64792b511027528d c0027992de21067525cc4489b74acb157feb89f3a4c7618580addc2b341be1c1.
- ruff on the four files: "All checks passed!".
- Against the working branch, only the package's four files differ ("4 files changed, 3723 insertions(+)"). `git status --porcelain --ignored` in the worktree is unchanged by my runs. The ignored __pycache__ directories date from 16 Nov 21:05, before this round.
- The report's rundown of the project state matches checkpoint/STATUS.md and the 17 Nov rulings.

TEST LINES (in the worktree, standing environment, through tools/lowprio.sh, basetemp under the scratch directory)
- Fast, both files: "49 passed, 11 skipped in 142.60s (0:02:22)", exit 0.
- The brief's acceptance, `STAGE2_SLOW=1 … tests/test_stage2_s2rows.py -k "equivalence or r6"`: "4 passed, 17 deselected in 183.50s (0:03:03)", exit 0.
- `STAGE2_SLOW=1 … tests/test_stage2_lookbar.py`: "39 passed in 94.42s (0:01:34)", exit 0.
- tests/test_stage2_monitor2.py: "22 passed, 1 skipped in 18.84s", exit 0. The skip: "lookbar is merged: the stand-in is no longer reached".
- `git status --porcelain`: the fixer's four modified files only.

COMMITS TO BRING IN
- None now.
- The 8 commits not in the working branch (899faec, 3600a65, 1785867, a3de365, bb28c45, d38d138, 5d04b20, 9952f5b) all end with two attribution lines: the earlier model's, right for when they were made. No other line names a model.
- Once the round is committed: a merge of the tip, provided every new commit ends with the two current lines.

NOT CHECKED
- The whole slow s2rows file (I ran only `-k "equivalence or r6"`), tests/test_stage2_train2.py, and an export without the ignored inputs.
- A kill during collect's move-aside, a Colab runtime or Drive, and any look set or V0 episode.
- The CN and L packages.

Scripts and logs are in /tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/check_cw_17nov/:
- mutate.py
- mutations.log
- mut_logs/
- scen_r9_and_paths.py
- fast_worktree.log
- slow_s2rows_acc.log
- slow_lookbar.log
- monitor2.log
- fixer.patch
- repo/ (the scratch clone)
