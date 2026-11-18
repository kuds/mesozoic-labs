Verdict: merge; tip: 52b441d99a0e924892cddb1a083787080adddf4f

## Verdict: merge. C2 (`52b441d99a0e924892cddb1a083787080adddf4f` on `wt/c2`, parent C1 `cc437b6a`) is correct, and I found nothing that needs fixing first.

I changed no tracked file and committed nothing. My sheet-check record, test log and pytest temp files are in `/tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/launch2/check_sheet/`. After my runs, `git status --porcelain` shows 0 lines in the worktree and 0 in the main checkout (which is at C1).

**Sheet check, rerun by me** (after `git fetch origin` of the data branch and the working branch). Run from `/home/user/mesozoic-wt/c2/hardware/trex_form_study` with the run environment and `--before-smoke`. Whole output:
```
S2 SHEET CHECK: left for the smoke: <<LIKE THIS>> <<SMOKE_TOTAL_MIN>> <<MIN_SETUP>> <<MIN_Y1>> <<MIN_Y3>> <<MIN_Y5>> <<MIN_Y6>> <<MIN_Y8>> <<TABLE OF LAUNCHES>> <<BUDGET_STEPS>> <<MIN_LINE4>> <<RATE>> <<W>> <<VCPUS>> <<BUDGET_M>> <<HOURS>> <<FLOOR>> <<MIN_LINE4_LATE>> <<HOURS_PER_JOB>> <<BACKGROUND>> <<MIN_REBUILD>> <<...>>
S2 SHEET CHECK colab_steps.txt at cc437b6a44d5: OK
```
Exit 0.

**Test line:** `tests/test_stage2_nb.py`, run through `tools/lowprio.sh` with the given pytest options: `73 passed, 2 skipped in 68.95s`, exit 0.

**Bit-identity:** the worktree and the main checkout both print `45719d03…528d c0027992…c1c1`.

### Values checked against the records myself
- **REPO_REF:** C1's 40 characters appear in section 1, in 3.1's address, in REPO_REF of 3.4 and 4.3, in 3.6 (a) and in line 1. No other 40-hex value is in the sheet.
  - 3.1's address is `kuds/mesozoic-labs/blob/cc437b6a…4947/hardware/trex_form_study/stage2/stage2_training.ipynb`. The notebook exists at that path at C1, and the remote matches the record's `remote`.
  - Neither `s2-launch-1` nor any other tag text is left.
- **Bundle:** s2-b1, manifest `9227d8a0bf0b`, 19 files in 3 parts, all equal to MANIFEST.json and the record.
- **BUNDLE_SOURCE:** `git:claude/mesozoic-trex-stage2-bundle-s2-b1` in both tables.
  - The local and remote branch are both at `ecf95a81` (ls-remote agrees).
  - Its three parts have the same sha256 as the manifest and as the parts the acceptance run used (I compared sha256 values myself).
- **Rules texts:** I ran `git cat-file blob C1:<path> | sha256sum` on each:
  - `STAGE2_TRACKB_PLAN.md`: ba966378…d4a2
  - `stage2/design/ROUND_C_DESIGN.md`: 93e041b9…540b
  - `stage2/design/ROUND_C_RULINGS_17NOV.md`: 50599143…3764

  All three match the sheet.
- **The acceptance record:** `ACCEPTANCE.json` has sha256 41d67f7f…a2f8, the same as the source file, and the same value is quoted in section 8.
  - Its fields: `ok: true`, 36 of 36 checks, ref, `clone_head` and `ls_remote_sha` all C1, content hash 9227d8a0…f9dc.
- **Lines 1 to 3:** identical to the lines the record holds for session0 and session1 (40368 physics steps, 2 of 2).
- **Section 8:** every value matches the record.
  - LIVE_S 29 (`refused_live`) and RESUME_STEPS 1024 (`session1` resume line).
  - 23:29 is session 0's `started_at` 23:29:57Z; 23:48 is `at` 23:48:08Z (`seconds` 1092).
  - Line 4 rates 463, 484 and 488, with session1 `takeover: true`.
  - Kill during the `model_2048.zip.part` copy with 1024 already on the stand-in Drive.
  - `setup.pip` is `[]`, so nothing was installed.
- **The queue:** no body other than 75-M-H-10 appears anywhere. The table of launches stays the template's field ("FILLED IN AFTER THE SMOKE", in `SHEET_SMOKE_FIELDS`). 4.3's example is the template's own `R5:75-M-H-10:42, R6:75-M-H-10:42`.
- **Diff against C1:** exactly three files: the sheet (M), ACCEPTANCE.json (A) and dev_episodes.json (M). The template and the manifest are unchanged.
- **Commit message:** it ends with exactly the two required lines, with no trailing text after them. No model is named anywhere else ("claude/" appears only inside the branch name).
- **Merge readiness:** `origin/claude/mesozoic-trex-stage2-refine-4jffre` is still C1 after a fetch, and it is an ancestor of C2, so C2 merges as a fast-forward.

### Read as a non-programmer would
- No sentence sends the reader to Drive for the bundle. Section 2 says the files are on a branch and there is nothing to upload or approve. 7.1 and 7.2 apply only on the lead's word.
- 7.2 says "I put the bundle's one small model file on your Drive; there is nothing for you to do." That is consistent with P45, which lets the lead upload the bundle's parts with no further question per file. `nb.inputs` with `rebuild` does read `models-*` from `<drive>/inputs/<id>/`.
- RUN_ID, REPO_REF, TAKEOVER and deleting are only ever mentioned as things never to do or to change. Section 8's TAKEOVER sentence says outright "you never set it, and wait ten minutes instead (5.6)".
- All "(expected, not yet seen)" marks are aligned. Only the URL row is wider than 115 columns, as in the template. The sheet is plain ASCII.

### dev_episodes.json
- The two new entries use the same keys in the same order as the existing six, and each (step, set, body) key is unique.
- **Training steps, 14336:** I checked this against the trainer logs and `updates*.jsonl` under `/home/user/mesozoic-wt/tb-accept/drive/runs/`.
  - In both runs the R5 session 0 log ends at "check at 2048 steps", and the resume is from 1024.
  - So R5 is 2048 + 3072 in each run, and R6 is 4096.
- **Check episodes, 64:** this is 4 + 6 + 18 + 36. Each part is backed by the logs: three G8a short forms and three R0L baselines (once per job: acceptance R5, acceptance R6, remedies R5), six whole monitor checks, and line 3 in two notebook sessions.

### Notes for the lead (none blocks the merge)
1. **The 64 is a planned count.** Each R0L and monitor record lists 6 episodes, but `turn/q90_left` is `not_run` in all nine of them ("no reference within the thigh twist and the leg drives"). So 45 of those 54 actually ran. The two checks at 2048 that the kills cut short did run some episodes and are not counted.
   - The entry's reason states exactly what it counts, and 6 matches each record's `n_episodes`. Whether to list episodes that ran or episodes that were planned is your call.
2. **The first acceptance run's tiny V6 episodes are still unlisted.** That run (b7527a66, 21:54 to 22:13) is not in dev_episodes.json, so D23 and X1 still have a gap. It is outside this step, and the report says so.
3. **The sheet still opens with "*** DRAFT ... DO NOT START FROM THIS DRAFT. ***".** The template has it too, but it contradicts the header's "I send you the sheet again". That is safe, since it leads to no wrong action, but the banner needs your decision before the sheet goes to the owner for the smoke.
4. **The table of launches is still open.** V13 asks for "the queue per runtime", while the template defers the table until after the smoke. The sheet follows the template, as this task asked.
5. **The `git:` source has not yet run end to end through the notebook.** The acceptance run used `drive`; `sheet-check` only checked the parts' sha256 on the branch. The smoke will be its first real use.

**Not done:** I ran no tests beyond `tests/test_stage2_nb.py`. I did not rerun `nb acceptance`. I did not see the acceptance command itself run.
