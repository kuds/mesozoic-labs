## Ruling 45 on `wt/wpc-cn`: all three items are done, each has a test that fails without it, and the branch can be merged

All three items are done as ruling 45 asks. The builder's report matches what I saw myself. I changed no tracked file, committed nothing and pushed nothing. My scripts and logs are in `/tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/nb45/check/`. I deleted the pytest temp dirs (`nb45/bt_*`) afterwards.

**State.** HEAD of `/home/user/mesozoic-wt/wpc-cn` is `6b70887c62204b52e9b293508dbae9086e814de6`, and the worktree is clean. The working branch tip `b45424a9f4458a1e8c96f79e8a6640ddef0b0c25` (local and `origin/`) is an ancestor of HEAD, so the merge is a fast-forward. The main checkout is on the working branch at b45424a9, and its `git status --porcelain` is empty.

**Tests.** I ran `tests/test_stage2_nb.py` from the study directory with the required environment, `tools/lowprio.sh` and `--basetemp` under nb45. Exit 0, summary line:
`74 passed, 1 skipped in 71.92s (0:01:11)`
The skip is `tests/test_stage2_nb.py:3266: the real trainer on the S1 view; set STAGE2_SLOW=1`.

**Bit-identity** printed `45719d030359ec7d5d3914af3814d837fdc05d77652ba45c64792b511027528d c0027992de21067525cc4489b74acb157feb89f3a4c7618580addc2b341be1c1`, the expected pair.

### (1) The kill condition
- **The fix:** `nb.acceptance_kill(jobdir, whole)` returns a reason only when two things hold:
  - a checkpoint above 0 is whole on the stand-in Drive. "Whole" is `_Launcher._whole`: the record, plus each of its three files with the record's size and sha256.
  - a `.part` file exists there.
- `acceptance` now polls it in place of the old `part_exists`. `killed_during_a_copy` also requires a checkpoint above 0 in `kill.on_drive`.
- **Against the other kills:** `local_run`'s and `_remedies`' `copied()` wait for a checkpoint above 0 on Drive. The new condition does that too, adds the whole-file test, and adds the `.part` requirement that the ruling and design 3.10 step 4 ask for.
- **Tests fail without it.** I swapped the function out through a pytest plugin (`nb45mut.py`) and ran the three kill tests against each version:

| Version swapped in | Result |
|---|---|
| The V12 condition (first `.part`) | 3 failed. The live test fails with `AssertionError: (['model_1024.zip.part'], [])`, as V12 did. |
| `local_run`'s condition (no `.part` needed) | 3 failed |
| No whole-file test | the unit test fails at the spoiled `vecnormalize_1024.pkl` |
| Unchanged | 3 passed |

- **The live-launcher test** passed 6 of 6 extra runs.
- **Timing in the real run** (not run here; the STAGE2_SLOW `local_run` was not run either). The V12 record suggests there is room:
  - Session 0 was killed at progress 1536 while copying checkpoint 1024.
  - On the resume, R5 took 82 s for 0 to 4096 steps, so a later copy should come long before the first job ends.
- **Still open:** the race the builder names (a `.part` finishing between the 0.05 s poll and the SIGKILL) remains.

### (2) The template and the sheet check
- `stage2/tb/colab_steps.template.txt` and `colab_steps.txt` are byte-identical, sha256 `a2cdaaa8…6cb3a`.
- The four sheet tests read `TEMPLATE`. No test reads the contents of `colab_steps.txt` any more (checked with grep), so a filled sheet no longer breaks the suite.
- **The V13 sheet (read-only copy).** I ran `sheet-check --before-smoke` with `--ref b7527a66…` and the V12 record. Exit 1, failing only:
  - `lines` (`ok: False (failed: resumed_from_drive)`)
  - `section_8` (`"while the first was alive"`)
- **With section 8 corrected** and a re-stamped passing copy of the record (scratch only): `ok=True`.
- **Each mutation fails its own check and no other:**

| Mutation | Fails |
|---|---|
| Leftover `<<C1>>`, `<<BUNDLE_ID>>` or `<<HASH12>>` | `fields` plus that value's own check |
| A lowercase `<<todo>>` | `fields` |
| An unpaired `<<` | `fields` |
| The default without `--before-smoke` | `fields`, listing the 21 smoke fields |
| REPO_REF in 4.3 another commit | `repo_ref` |
| The Colab address at another commit or a wrong path | `repo_ref` |
| Line 1 "clean: no" | `repo_ref` and `lines` |
| BUNDLE_ID or "bundle X" wrong | `bundle_id` |
| The 12-character hash wrong, or "18 files" | `content_hash` |
| Line 2 "18 of 19" | `content_hash` and `lines` |
| BUNDLE_SOURCE a branch without the parts | `bundle_source` |
| BUNDLE_SOURCE drive in one table | `bundle_source` |
| A rules sha256 wrong or swapped | `sha256_values` |
| Section 8's "from 1024 steps" or "40 s" | `section_8` |
| Line 3 steps | `lines` |
| The whole sheet checked against `--ref b45424a9` | `repo_ref`, `sha256_values` and `lines` |

- The builder's test fails without `sheet_check`. Against the C1 template, the fields test and the command-line test also fail.

### (3) Section 8
The template's new paragraph tells the run-directory lines in `_drill`'s order:
1. the new run;
2. the session stopped during a copy, after an earlier save was whole;
3. the start refused for a fresh LIVE.json although nothing ran;
4. the resume with TAKEOVER, which the owner never sets (5.6).

It adds two fields, `<<LIVE_S>>` and `<<RESUME_STEPS>>`. `test_section_8_of_the_template_says_when_the_refused_start_came` fails on the C1 template (`AssertionError: the acceptance run printed "(new run)" in its first session`). The other four sheet tests still pass on it.

### Attribution and scope
- Five commits:
  - `fa54c61ef0537359360412c59ed4f6832f55d586` (merge of b45424a9)
  - `a6c322c4384fd6e0215e2044f08f1843bf939edf`
  - `f7deb1c7f25ccefdf6edd32bebc1fb8d2f54edea`
  - `95b78a522b7b483b151923c17f7e1ee4911275d1`
  - `6b70887c62204b52e9b293508dbae9086e814de6`
- Each raw message ends with exactly the two required lines, and no other line names a model.
- `bar.py` and `moves.py` are untouched.
- `git diff claude/mesozoic-trex-stage2-refine-4jffre --stat`:

```
 hardware/trex_form_study/designlab/stage2/nb.py    | 363 ++++++++++++++++++-
 .../stage2/tb/colab_steps.template.txt             | 385 +++++++++++++++++++++
 hardware/trex_form_study/stage2/tb/colab_steps.txt |  10 +-
 hardware/trex_form_study/tests/test_stage2_nb.py   | 359 ++++++++++++++++++-
 4 files changed, 1092 insertions(+), 25 deletions(-)
```

- sha256 of the files (they match the builder's):

| File | sha256 |
|---|---|
| `nb.py` | `06629a1be220f2138dddc8df0558a500f7076cf4d00c3531063f506ff2b5b6cc` |
| `test_stage2_nb.py` | `bf2907a9c33f547511380cbdf92a5f82b921da4f4b8245aa41fe7003ff8ddfbb` |
| template and `colab_steps.txt` | `a2cdaaa83c0cbb9a685ad9a7d5b2fd1b3b93584cf4f00915e20050a96da6cb3a` |
| notebook (unchanged) | `b2ca42eee693a76e467354928d6366edef68104e1639d5a48013228bfddf65bb` |
| `requirements-colab.txt` (unchanged) | `8bd4d8dd5d2e8d09ad1544a58eba2ca2ea0d59b7fa6383180f3818be7cc266da` |

### Small points for the lead (none blocks the merge)
- **The Colab address's GitHub owner and repository are not checked.** A sheet whose address named `someone/else` passed with `ok=True`. The ruling's list does not ask for this check.
- **`--before-smoke` lets every field in `SHEET_SMOKE_FIELDS` stay,** including the generic `<<...>>` and TABLE OF LAUNCHES. Use it only at C2.
- **No test fails if the new `on_drive` condition in `killed_during_a_copy` is removed.** That condition is an extra tightening, not something the ruling asked for.
- **The V13 sheet needs the new section 8 wording and the two new fields,** and its "from 0 steps" must match the record of the rerun.
