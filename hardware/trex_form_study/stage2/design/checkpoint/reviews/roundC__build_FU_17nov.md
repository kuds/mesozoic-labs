# build:FU (17 Nov): the agent's final report

Working paper. Nothing here is a rule.

All three packages are done: every minor point of the three checks and of ruling 12 is built, and each change has a test that fails when the change is undone. All tests pass on the three heads. The record-key hash pair is the registered one in each worktree. Nothing was pushed and the main checkout was not touched.

1. MERGES
- Each branch was an ancestor of the working branch (tip a8f62e12), so all three merges were fast-forwards (`git merge --ff-only`). There were no conflicts and no merge commit.
- The working branch later moved to fccebe42, which only adds 7 lines to stage2/design/checkpoint/STATUS.md. I did not merge it again. `git merge-tree --write-tree` against it is clean for all three branches.

2. PER ITEM

Notebook (wt/wpc-cn)
- **Finding 1 / ruling 12:** in `nb.session`, LEAD.json's `retry_check` now applies only to the session that resumes the halt whose remedy names it. A RETRY_CHECK in a parameter file still applies. LEAD_RULE says so. Test: `test_the_leads_retry_check_answers_only_the_halt_that_names_it` (the check's exp8a: session 2 is a plain resume and carries no `--retry-check`).
- **Halt for moved sources:** new `halt_remedy` / `HALT_REMEDIES` in `_Launcher.run_job`. A trainer halt that names `--allow-source-change` gets the remedy `allow_source_change`; RESUMABLE_RULE says so. Test: `test_a_halt_for_moved_sources_asks_the_lead_for_allow_source_change`.
- **Cell 8 releases only on 0 and 5:** `release_or_keep` uses `RELEASE_CODES = (0, 5)`. On 6 it prints KEEP_RUNTIME_TEXT; on any other code, or none (crash, KeyboardInterrupt), it prints `keep_other_text`'s paragraph and keeps the runtime. Both paragraphs begin "THIS MACHINE IS KEPT ON PURPOSE". Test: `test_the_finish_keeps_the_runtime_while_a_save_is_not_on_drive` (cell 8 run with codes 6, 0, 5, 1, 4, 130, KeyboardInterrupt and OSError).
- **One forced remount:** new `remount_if_dead`, which cell 8 calls before the finish. When the run's Drive directory answers with ENOTCONN (errno 107), it calls `drive.mount("/content/drive", force_remount=True)` once and prints what happened; it never raises. The steps sheet has a new row for its message, marked "expected, not yet seen", and section 8 lists it. Test: `test_cell_8_mounts_a_dead_drive_again_once`.
- **Step 6 exercises the real trainer's source check:** `_remedies` now also fakes the last session's recorded hash of designlab/stage2/r5core.py (`FAKED_SOURCE`, with the sources hash recomputed). The sequence is now five sessions:
  1. Killed run.
  2. Resume refused by the trainer's own source check (exit 4; the text names `--allow-source-change`).
  3. QUEUE.json's commit faked; resume refused by the notebook's K6 check.
  4. LEAD.json `allow_source_change`; the trainer refuses for the versions only, so the flag reached it.
  5. Both remedies; DONE, and the job's run.json last session has `source_change` true and `sources_before` equal to the faked hash.
  - New checks: `source_change_refused_by_the_trainer` and `allow_source_change_reached_the_trainer`. The tests' stub trainer now has train2's source check. Test: `test_design_3_10_step_6_the_leads_remedies_after_a_faked_record`.
- **Finding 2 (steps sheet):** 5.3 says do not do 5.4 after "THIS MACHINE IS KEPT ON PURPOSE" and to run box 8 again and tell me; 5.4 repeats it. The section 6 row also covers the new paragraph ("do not delete it, not even as in 5.4"). Test: the steps-sheet test now requires every "Disconnect and delete runtime" or "let the machine go" step to name the exception.
- **Finding 3:** `test_the_pack_tools_tries_are_kept_so_that_the_finish_knows_them` (the check's exp9). The finish answers [0, 0, 0]; with the state line removed it answers [6, 6, 6].
- **Finding 4 (steps sheet):** one row per warning. A failed copy reads "your Drive may be full. Tell me; I may ask you to empty the Drive's bin. Delete nothing yourself."; a failed read reads "once: nothing to do ... Again and again: tell me." Tested in the steps-sheet test.

Trainer (wt/wpc-ct)
- **Finding 1 / ruling 12:** `monitor2._checkpoint` refuses a record without a `layout` or `fingerprint` object (refusal kind `checkpoint`). The `_job` helper's record gains both objects. RULES["refusal"] text updated. Test: `test_the_other_exits_before_any_episode_are_refusals` (six bad forms plus both missing).
- **Finding 2 / ruling 12, the allowance:** in train2:
  - New `SOURCE_ALLOWANCE_RULE` and `allowed_sources`; `_refuse_sources_versions` sets `self.source_allowance`.
  - A session given `--allow-source-change` adds its sources hash to run.json `sources_allowed` (session index, time; once per hash).
  - A later session on an allowed hash is treated as if given the flag: its resume check accepts the sources, and its monitor child gets `STAGE2_ALLOW_SOURCE_CHANGE=1`.
  - Sessions record `allow_source_change` and `source_allowance`; RESUME_RULE names the allowance.
- **Halts for moved sources:** a check's or the baseline's halt for moved sources (the monitor's own sources check, or the row's rule K9, via `moved_sources`) ends with `SOURCE_REMEDY`, which names `--allow-source-change`.
- Tests for the allowance and the halts:
  - `test_an_allowance_for_moved_sources_holds_for_the_same_sources`: end to end, with a stub row that refuses by rule K9 unless allowed.
  - `test_a_halt_for_moved_sources_names_the_parameter_that_resolves_it`: row K9 at a check, and the monitor's sources check at the baseline.
  - `test_the_resume_takes_an_allowance_from_the_runs_record`.
- **Finding 3:** the second refusal in `test_a_monitor_that_refuses_to_run_is_no_attempt` now resumes with `--retry-check` and no open attempt; the halt must not name that parameter.
- **Finding 4:** the "not merged" texts in monitor2.py and test_stage2_monitor2.py are updated (text only).

Rows (wt/wpc-cw)
- **Low point 1:** `test_a_fault_in_the_reading_is_recorded_and_never_raised` now makes `clock_block` and `on_command_at_stop` raise errors that name paths, and asserts the cut form. The check's two survivors are caught now.
- **Low point 2:** `_no_paths` keeps at most the last three parts and never the first: "/home/user/a" becomes ".../user/a" and "/content" becomes "...". It also takes optional `roots`, under which paths are made relative. Test: `test_a_fault_names_no_path_of_this_machine`.
- **Low point 3:** `collect` passes a refusal's text through `_no_paths` with the output root, so `refused`, `refused_records[].reason` and the raised error name paths relative to the root. Test: `test_a_missing_or_foreign_sidecar_is_refused`.

Revert checks: 24 of 24 caught.
- Notebook: 10 of 10. A first run was void because its basetemp parent was missing (kept as mutants_run1_void_basetemp.log). One mutant first failed only on a state key; I reordered that test so it fails on the exit codes (commit a557af92).
- Trainer: 9 of 9. The first pass found the baseline halt untested; commit d1def040 adds that test.
- Rows: 5 of 5.

For the lead (my reading is built in each case):
- a) The allowance also covers the trainer's own resume check: sources equal to an allowed hash are accepted even when the last session's differ. Alternative: apply it to the row only. I recommend keeping it as built ("holds for the rest of the run for the same sources").
- b) A hash goes into `sources_allowed` whenever a session is given the flag, even if nothing moved. This is harmless. Alternative: record it only when a move was actually allowed.
- c) A refusal of kind `sources` (the checkout moved under the run) also names `--allow-source-change`. The notebook then requires LEAD.json `allow_source_change` to resume it. Alternative: name the flag for the row's K9 refusal only.
- d) `moved_sources` recognises the row's K9 refusal by its text ("rule K9" or STAGE2_ALLOW_SOURCE_CHANGE). That couples train2 to s2rows' wording. A cleaner fix would be a dedicated marker in s2rows (CW's file), which I did not change.
- e) monitor2's `_stand_in` is kept even though `_episodes` cannot reach it while lookbar is merged. Removing it would be a later change.
- f) I did not add the check's optional `rec.steps == steps` check to `_checkpoint`; the ruling names only layout and fingerprint.
- g) Only ENOTCONN counts as a dead mount; whether Colab ever answers EIO or ESTALE instead is not seen. 5.3 relies on the owner seeing the last box's output: if the page is lost the paragraph may not show. A marker file on Drive would be unreliable exactly when Drive writes fail.
- h) Step 6 has one more trainer start than before; the slow run took 357.86 s.

3. TEST SUMMARY LINES (each worktree head, standard environment, through tools/lowprio.sh, basetemp under the scratchpad)
- wpc-cn `tests/test_stage2_nb.py`: "65 passed, 1 skipped in 60.33s (0:01:00)".
- wpc-cn slow, `STAGE2_SLOW=1 -k local_run`, on a557af92: "1 passed, 65 deselected in 357.86s (0:05:57)".
  - This is a tiny development run: two tiny jobs plus the step 6 job on the Stage 1 view of 75-M-H-10, real trainer.
  - Record: LOCAL_RUN.json ok true, 25 of 25 checks, all 8 step-6 checks true.
  - The real trainer printed "train2 run: Refused: .../run.json: other sources than the run's last session (66c49be6c475... -> d6c615e4634f...; files: designlab/stage2/r5core.py): refusing to resume (pass --allow-source-change to resume on them, flagged)", then on the next resume "other library versions than the run's last session".
  - The final run.json's last session has source_change True and version_change True, with sources_before.sha256 66c49be6c4756527... (the faked hash).
  - The resumed checks' monitor records show source_change true (episodes_by lookbar.episodes).
- wpc-ct `tests/test_stage2_train2.py`: "53 passed, 3 skipped, 2 warnings in 220.92s (0:03:40)".
- wpc-ct `tests/test_stage2_monitor2.py`: "25 passed, 1 skipped in 32.61s".
- wpc-cw `tests/test_stage2_lookbar.py tests/test_stage2_s2rows.py`: "53 passed, 11 skipped in 126.47s (0:02:06)".
- ruff `--no-cache` on every changed .py file: "All checks passed!".
- Hash pair at every head: 45719d030359ec7d5d3914af3814d837fdc05d77652ba45c64792b511027528d c0027992de21067525cc4489b74acb157feb89f3a4c7618580addc2b341be1c1.
- Not run: the acceptance run, the trainer's and the rows' STAGE2_SLOW suites, anything on Colab. No episode on a look set; nothing written under stage2/tb/gates/ or stage2/tb/validation/.

4. DIFF AND STATUS
Against claude/mesozoic-trex-stage2-refine-4jffre (its tip is now fccebe42, so each stat also shows "stage2/design/checkpoint/STATUS.md | 7 -": that is the working branch's new commit, not a change of ours). Against a8f62e12, the tip I merged, only package files differ:
- wpc-cn: nb.py | 201, stage2_training.ipynb | 12, colab_steps.txt | 29, test_stage2_nb.py | 256; "4 files changed, 425 insertions(+), 73 deletions(-)".
- wpc-ct: monitor2.py | 26, train2.py | 97, test_stage2_monitor2.py | 23, test_stage2_train2.py | 151; "4 files changed, 259 insertions(+), 38 deletions(-)".
- wpc-cw: lookbar.py | 27, test_stage2_lookbar.py | 42; "2 files changed, 59 insertions(+), 10 deletions(-)".
- `git status --porcelain` is empty in all three worktrees and in the main checkout. No new untracked or ignored files in the worktrees; every output is in the scratchpad, outside the repository.

5. COMMITS (oldest first)
Every message ends with the two given lines and names no model elsewhere (checked by script).
- wt/wpc-cn: b0b8a798, ba554805, bf48aff7, 7c8ceba4, a557af92. Head a557af92eb67141aaf8e98aba768a722de74f71e.
- wt/wpc-ct: fd1f88ff, a24a9c6a, e09577f3, d1def040. Head d1def04012a9311738877f7fce920bb1aa1a8c74.
- wt/wpc-cw: 762befff, a8444f03, 097bfd64. Head 097bfd64c52531e5f5d09c8b0bf0627af4b21c15.

Files are in /tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/wave5/:
- Scripts: tools/hunks.py, tools/mutate.py.
- Notebook: cn/mutants.log, cn/mutants_run1_void_basetemp.log, cn/final_fast.log, cn/slow.log, cn/slow/bt/test_local_run_with_a_kill_and0/LOCAL_RUN.json.
- Trainer: ct/mutants.log, ct/final_train2.log, ct/final_monitor2.log.
- Rows: cw/mutants.log, cw/final_fast.log.
- Commit messages and patches: the msg*.txt and patches/ in cn/, ct/ and cw/.
