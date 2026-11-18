## V13 / C2: NOT COMMITTED. The filled sheet fails the steps-sheet test, and none of the failing expectations names the Drive-upload step.

The task asks for two things that can't both hold. Filling the sheet as instructed (C1's SHA, s2-b1, the hash, the rules, 40368, and the SHA in place of the tag in the URL) breaks `tests/test_stage2_nb.py::test_the_steps_sheet_quotes_the_programs_own_lines`. That test still expects the draft placeholders and the tag. No expectation in the test file names the Drive-upload step, so the one adjustment you allowed has nothing to apply to. I changed no test and committed nothing. Branch `wt/c2` is still at C1, and the work is left uncommitted in the worktree for you to decide.

**Where things stand**
- Worktree `/home/user/mesozoic-wt/c2` is on new branch `wt/c2` at C1 (`b7527a66…`). C1 is the tip of the working branch, locally and on `origin`.
- Step (1) is done but untracked: `/home/user/mesozoic-wt/c2/hardware/trex_form_study/stage2/tb/bundle/s2-b1/ACCEPTANCE.json`. Its sha256 is `2d1578aaf1c85b8ac10e79a96621c46cbe2896acf4fe3dd6c049428f66ac117f`, the same as the source. The MANIFEST.json beside the source is byte-equal to C1's.
- Step (2) is done but uncommitted: `/home/user/mesozoic-wt/c2/hardware/trex_form_study/stage2/tb/colab_steps.txt`, sha256 `d186f15e…8485`. It is ASCII and fits 115 columns, except the URL row, which the test allows.
- Bit-identity printed the expected pair in both checkouts.
- The main checkout is at C1 with `git status --porcelain` empty. I wrote nothing there.

**Test line**
- `tests/test_stage2_nb.py` on the filled sheet: 66 passed, 1 failed, 2 skipped, exit 1.
  - The failure is `test_the_steps_sheet_quotes_the_programs_own_lines`, at line 548: `AssertionError: <<C1>>`.
  - The skips: "the Stage 1 arrays are not in this checkout" and "set STAGE2_SLOW=1".
- Baseline at C1 in the same worktree: 67 passed, 2 skipped, exit 0.
- pytest printed no count line in these logs, so I counted from its progress line. The logs are in the scratchpad: `c2_filled_tests.log` and `c2_baseline_tests.log`.

**Expectations that conflict with the task** (all in that one test)
- **Lines 545-547:** the placeholders `<<C1>>`, `<<BUNDLE_ID>>`, `<<HASH12>>`, `<<N_FILES>>`, `<<G8A_STEPS>>`, `<<RULES_FILE` and `<<RULES_SHA256>>` must still be in the sheet.
- **Line 554:** "section 8's first paragraph" must be listed as still open.
- **Line 556:** the literal `<<RULES_FILE: THE COMMITTED RULES TEXT; TODAY STAGE2_TRACKB_PLAN.md>>` must be in the sheet.
- **Lines 559-560, 581, 583-584:** lines 1-3 are compared with the test's stand-in values: `COMMIT="c"*40`, `tb-1`, `0123456789ab`, 137 files, 6000 steps.
- **Line 622:** the URL must contain `s2-launch-1/hardware/trex_form_study/stage2/stage2_training.ipynb`, the tag the task told me to replace.

I saw only the first of these fail; pytest stops at the first. I found the rest by reading the test. A scratch copy with exactly these changes passes, which confirms it.

**A test patch, ready if you allow it** (not applied): `/tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/test_stage2_nb.proposed.diff`
- It keeps the smoke's placeholders as required.
- It requires the acceptance run's fields to be filled, and each of the three rules paths to be followed by a 64-hex sha256.
- It compares the sheet's lines 1-3 with `nb.line1/2/3` built from ACCEPTANCE.json, and with the lines that record holds as printed.
- It checks the URL against the record's `ref`.

I ran it as a temporary untracked copy in the worktree and then removed the copy. All 4 steps-sheet tests passed on the filled sheet. If you allow it, C2 is: apply the patch, `git add` the sheet, the record and the test, and commit.

**The sheet's filled lines** (quoted)
- **Section 1:**
  - `b7527a66528750746edf8f533c7f3b8a066aa1e3` / `This is REPO_REF in every step below. The address in 3.1 names the same 40 characters.`
    - I removed the sentence about the tag `s2-launch-1`, because no tag is pushed. GitHub's and Colab's URLs accept the SHA.
  - `The inputs:  bundle s2-b1, manifest 9227d8a0bf0b, 19 files in 3 parts. They are on a branch of the repository, claude/mesozoic-trex-stage2-bundle-s2-b1, and the notebook reads them from there.`
  - The rules, each with its sha256 at C1:
    - `STAGE2_TRACKB_PLAN.md` `ba966378861e1b11664b338b1dae54e12d7da10e44bfb60260bde7c514e6d4a2`
    - `stage2/design/ROUND_C_DESIGN.md` `e10ea47d9096d3bf93cf5541d00c68e67968ff565ae3613314dfb31c45b414ba`
    - `stage2/design/ROUND_C_RULINGS_17NOV.md` `a496521ddb6dc2874539168f3046074b781448dbd4176a691bdcffc1c2db5b44`
- **Section 2:** `2. BEFORE DAY ONE (nothing for you to do)` / `I put the input files on a branch of the repository, where the notebook reads them; there is nothing to upload.`
- **3.1:** `https://colab.research.google.com/github/kuds/mesozoic-labs/blob/b7527a66528750746edf8f533c7f3b8a066aa1e3/hardware/trex_form_study/stage2/stage2_training.ipynb`
- **3.4 and 4.3:** `REPO_REF b7527a66528750746edf8f533c7f3b8a066aa1e3`, `BUNDLE_ID s2-b1`, `BUNDLE_SOURCE git:claude/mesozoic-trex-stage2-bundle-s2-b1`.
- **3.6:**
  - (a) `Repository ref: b7527a66…aa1e3; commit: b7527a66…aa1e3`. It is still marked expected, not yet seen.
  - (b) `S2 LINE 1/4 commit: b7527a66528750746edf8f533c7f3b8a066aa1e3 (REPO_REF is a commit: yes; checkout clean: yes)`
  - (e) `S2 LINE 2/4 inputs: bundle s2-b1, manifest 9227d8a0bf0b, 19 of 19 files match their sha256: OK`
  - (f) `S2 LINE 3/4 G8a 75-M-H-10 set V6: zero action = R0 at 0 ms and 10 ms, 2 of 2 episodes identical (40368 physics steps): PASS`
  - Lines (a) and (b) are each split over two rows; on screen each is one line.
  - These lines are exactly the ones the record holds for both sessions.
- **Section 8:** written from the record. It covers:
  - what the run did, its record path and sha256;
  - the Run-directory forms it printed (new run; resuming session 1 from 0 steps; still running, 41 s);
  - its line-4 rates (502, 467 and 575 env steps/s at 2 workers on 4 vCPUs), with a note that these are not Colab's rate;
  - that it did not read the inputs from the branch;
  - that one of its 36 checks, `resumed_from_drive`, did not pass, and why.

**The queue**
Source: STAGE2_PLAN.md section 6, "Runs" and "Phases" (R5 seed 42, R6 in spare slots, then seed 43, then the fine-tunes; 3 runtimes at once). Section 4's table:
- Runtime 1: copy `s2 launch-1a`, RUN_ID `launch-1a`, QUEUE `R5:75-M-H-10:42`
- Runtime 2: copy `s2 launch-1b`, RUN_ID `launch-1b`, QUEUE `R6:75-M-H-10:42`
- Runtime 3: copy `s2 launch-1c`, RUN_ID `launch-1c`, QUEUE `R5:75-M-H-10:43`

Choices you should check:
- The RUN_IDs and copy names are mine.
- R6 takes seed 42. The plan gives R6 no seed; the design's example and the acceptance run's queue use 42.
- I put one job per runtime because the owner has 3 runtimes at once and a 40M-step job is about 20 h, against 24 h sessions.
- 4.3's two-job example stays because the test requires it.

**Left out, and why**
- 60-N-T-10 (R5 seeds 42 and 43, and its R6): parked by the owner (P43).
- 75-M-H-10-X (R5 seeds 42 and 43): waits for D5's preconditions and is not in s2-b1 (ruling 44).
- 60-N-T-10-X: both reasons.
- The fine-tunes (up to 8 of 5M steps, K8): each starts from a parent run's checkpoint, so none can be queued at the start of series 1.

**Other edits and open points**
- **Section 7:** 7.1 and 7.2 spoke of re-uploading files "as in 2" and offered the git source as a fallback. They now name the branch, and the fallback is `rebuild` alone.
  - `nb.inputs` with `rebuild` takes the `models-*` part from `<drive>/inputs/s2-b1/`. So 7.2 now says you first put that one small file on the owner's Drive (P45 allows it). Please confirm.
- **Line 2 failure hint:** now reads "could not be read from the branch".
- **The header** still says "DRAFT … DO NOT START FROM THIS DRAFT", because values remain open.
- **Still open, measured only at the smoke:** BACKGROUND, SMOKE_TOTAL_MIN, MIN_SETUP, MIN_Y1/Y3/Y5/Y6/Y8, BUDGET_STEPS, RATE, W, VCPUS, BUDGET_M, HOURS, FLOOR, MIN_LINE4, MIN_LINE4_LATE, HOURS_PER_JOB, MIN_REBUILD, and every "(expected, not yet seen)" mark.
  - I filled no minutes. The acceptance run installed nothing (its `setup_seconds` of 186.6 s and 163.3 s are local) and ran no smoke.
- **The `git:` source** has not been run end to end; the acceptance run took the parts from a stand-in Drive folder. The smoke is its first real read.
- **The record still says `ok: false`.** C2 would commit a failed acceptance record unless the kill-condition fault in `nb.acceptance` is fixed and the run repeated.

**Not done:** no commit, no test edit in the repository, no push, tag or upload, no code change, no rerun of the acceptance run. Commit id: none.
