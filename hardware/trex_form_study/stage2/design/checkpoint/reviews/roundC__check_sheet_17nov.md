Verdict: sound_after_fixes

## Review of V13 / C2: the stop was right. Every filled value is correct, and one sentence in section 8 is wrong.

C2 does not exist yet. The report says so, and the reason holds: `tests/test_stage2_nb.py` is written for the draft sheet. A correctly filled sheet cannot pass it, and the one change the lead allowed (an expectation that names the Drive-upload step) has nothing to apply to. I found one small factual error in the sheet and a few points the lead must decide. Once those are settled, C2 can be committed.

**What I ran and saw**
- **State:**
  - `wt/c2`, the main checkout's branch `claude/mesozoic-trex-stage2-refine-4jffre` and `origin`'s copy of it all point to C1 `b7527a66528750746edf8f533c7f3b8a066aa1e3`, after a fetch.
  - The worktree shows only ` M .../colab_steps.txt` and `?? .../bundle/s2-b1/ACCEPTANCE.json`.
  - The main checkout's `git status --porcelain` is empty, and no `s2-launch` tag is on the remote.
  - Bit-identity printed the expected pair in both checkouts.
- **Tests:** I ran `tests/test_stage2_nb.py` on the filled sheet. The progress line shows 66 passed, 1 failed (`test_the_steps_sheet_quotes_the_programs_own_lines`, line 548, `AssertionError: <<C1>>`) and 2 skipped. This matches the report.
- **The conflict is real.** I made a scratch copy of the test that changes only the draft expectations the report lists:
  - lines 545-547 (the placeholders), 554 and 556;
  - the stand-in values in lines 559-560, 581 and 583-584;
  - the URL in line 622.

  In that copy, line 1 must equal `nb.line1(C1)`, line 2 `nb.line2("s2-b1", 9227d8a0…f9dc, 19, 19)` and line 3 `nb.line3("75-M-H-10", "V6", 2, 2, 40368)`. All 4 steps-sheet tests passed on the filled sheet. So the filled lines are exactly what the program prints, and the sheet's structure is intact.
- **Baseline:** the same 4 tests pass on C1's draft sheet.
- **The proposed patch** (`/tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/test_stage2_nb.proposed.diff`) applies cleanly to a scratch copy of the test. With it, all 4 tests pass on the filled sheet.
- **No test expectation names the Drive-upload step.** I searched for upload, connector, approve, N_PARTS, TOTAL_MB, DATA_REF and "BEFORE DAY ONE" and found none.

**Filled values checked against the records (all correct)**
- **The commit:** C1's SHA is used in section 1, in the 3.1 address (`.../blob/b7527a66…/hardware/trex_form_study/stage2/stage2_training.ipynb`; the notebook exists at that path at C1), and in 3.4, 4.3, 3.6 (a) and 3.6 (b).
- **The bundle:** "bundle s2-b1, manifest 9227d8a0bf0b, 19 files in 3 parts" matches the manifest. It has 19 files and 3 parts, and C1's manifest is byte-equal to the one beside the source.
- **BUNDLE_SOURCE** is `git:claude/mesozoic-trex-stage2-bundle-s2-b1` in 3.4 and 4.3. On `origin`, the three parts sit at the path `nb._parts_from_git` reads, and each has the manifest's sha256.
- **The rules' sha256 at C1** (from `git show C1:<path> | sha256sum`):
  - `STAGE2_TRACKB_PLAN.md`: ba966378…d4a2
  - `stage2/design/ROUND_C_DESIGN.md`: e10ea47d…14ba
  - `stage2/design/ROUND_C_RULINGS_17NOV.md`: a496521d…5b44
- **The acceptance lines:** lines 1, 2 and 3 equal the record's `attempts.session0` and `session1` lines word for word (40368 physics steps). The "Run directory" forms and the line-4 rates (502, 467, 575; 8 envs, 2 workers, 4 vCPUs) match the record. So do 36 checks with only `resumed_from_drive` false, `kill.on_drive` empty, and the times 21:54 to 22:13 (the scratch marker is at 21:54:38; `at` is 22:13:46).
- **The acceptance record:** its sha256 is 2d1578aa…117f, the same as the source `/home/user/mesozoic-wt/tb-accept-bundle/stage2/tb/bundle/s2-b1/ACCEPTANCE.json`. It is not git-ignored.
- **The queue:** `R5:75-M-H-10:42`, `R6:75-M-H-10:42` and `R5:75-M-H-10:43` each parse with `nb.parse_queue`. The RUN_IDs pass `nb._NAME`.
  - This agrees with STAGE2_PLAN.md section 6 and the Track B plan's brief: R5 at seeds 42 and 43, one R6 per body, and 3 runtimes at once.
  - The plan gives R6 no seed, so seed 42 is a choice, and the report says so.
  - What was left out and why (60 cm parked by P43; -X held back by ruling 44; fine-tunes need a parent checkpoint) is stated correctly.
- **Nothing sends the reader to Drive for the bundle.** Section 2 is the one plain sentence the task asked for. The line-2 failure hint and 7.1 name the branch.
  - 7.2's rebuild fallback says the lead puts the models part on Drive. The code needs it there (`nb.inputs` with `rebuild` reads `models-*` from `<drive>/inputs/s2-b1/`), and the reader is asked to do nothing.
- **Form:** the sheet is ASCII. Every row is at most 115 columns except the URL row, which the test allows. Sections 1-8 and their numbered sub-steps are kept.

**Fixes before C2**
1. **Section 8 is wrong on one point.** It says the "still running (LIVE.json written 41 s ago)" line came "for a second start while the first was alive".
   - In `nb._drill` and the record's `takeover_note`, that refused start came after session 0 had been SIGKILLed. LIVE.json was still fresh, but nothing was running.
   - Suggested wording: "for a second start right after the first had been stopped (its LIVE.json was 41 s old)".
   - It would also be honest to say that the resume which followed ran with TAKEOVER. The owner never sets TAKEOVER and waits ten minutes instead (5.6). The printed resume line is the same either way (`nb.session_line`).
2. **The test patch needs the lead's permission.** It goes beyond the one adjustment allowed. Two comments on it:
   - It checks each rules sha256 only as 64 hex digits, not as the value at C1, so a wrong hash would pass.
   - The kept comment on the `PREREG_TRACKB.md` line ("the rules file's name is a field: the text is a draft") is now out of date.

**For the lead to decide (not faults of the step)**
- **The record says `ok: false`.** C2 would commit a failed acceptance record unless the run is repeated.
- **The sheet still says "DRAFT … DO NOT START FROM THIS DRAFT".** The test requires "DRAFT" in that paragraph, and the smoke's minutes (MIN_SETUP, MIN_Y*, SMOKE_TOTAL_MIN) are still open. As it stands, the sheet cannot be handed to the owner for the day-one smoke without a decision on that header.
- **7.2 depends on an untested upload.** The fallback has the lead upload `models-75-M-H-10.tar`, which is 972,800 bytes. P45 permits this, but P46's route test only shows that 170 kB went through and 5 MB was refused. About 1 MB has not been tried.
- **Two small points for the reader:**
  - 4.3's two-job example (kept because the test requires it) sits beside a table of one-job rows.
  - 4.6's "appears again when the second job of QUEUE starts" never happens with one-job queues.

**Merge readiness:** nothing is committed, so there are no attribution lines to check and nothing to merge. A C2 made of the sheet, the record and the allowed test change, on top of `wt/c2` (C1), would fast-forward the working branch, whose tip is C1 locally and on `origin`.

**Not done by me:** I did not commit anything or change any tracked file. I did not run tests outside `tests/test_stage2_nb.py`, did not open the Colab or GitHub address, and did not run the `git:` source end to end.

My scripts and logs are in `/tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/launch/check_sheet/`: `make_relaxed.py`, `review_filled_tests.log`, and the scratch test copies in `t/`, `p/` and `b/`.
