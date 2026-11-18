## Ruling 45 on `wt/wpc-cn`: all three items are done, and `tests/test_stage2_nb.py` passes

I merged the working branch first. Each of the three items has a test that fails without it. Nothing was pushed, no history was rewritten, and nothing was run on Colab.

**Test line** (`tests/test_stage2_nb.py`, run from the worktree's study directory with the required environment, `tools/lowprio.sh` and `--basetemp` under the scratchpad's `nb45`, which I have since deleted):
`74 passed, 1 skipped in 71.19s (0:01:11)`. The skip is `the real trainer on the S1 view; set STAGE2_SLOW=1`.

**Bit-identity** printed `45719d030359ec7d5d3914af3814d837fdc05d77652ba45c64792b511027528d c0027992de21067525cc4489b74acb157feb89f3a4c7618580addc2b341be1c1`, the expected pair.

### (1) The kill condition
- **The fix:** a new function `nb.acceptance_kill(jobdir, whole=None)` decides the kill. It fires only when both hold:
  - a checkpoint above 0 is whole on the stand-in Drive (the launcher's own `_Launcher._whole` test: the record, plus each of its three files with the record's size and sha256);
  - a `.part` file exists there.
- `acceptance` now polls it in place of `part_exists`.
- I also tightened the record's `killed_during_a_copy` check: it now also needs a checkpoint above 0 on Drive at the kill. The docstring no longer says the function "has not been run".
- **Tests:**
  - `test_the_acceptance_kill_waits_for_a_whole_checkpoint_on_drive` uses hand-made checkpoints. Cases that must not kill: V12's point, step 0, a spoiled or missing file, no `.part`, a record that has gone.
  - The plumbing test checks how `acceptance` itself wires the condition.
  - `test_the_acceptance_kill_on_a_live_launcher_leaves_a_checkpoint_to_resume_from` runs the real launcher with the stub trainer and a 0.5 s copy pause. The resume goes on from the checkpoint that was whole. It passed 6 of 6 runs.
- **Against the old condition** (patched in from scratch): all three tests fail. The live test fails exactly as V12 did: `(['model_1024.zip.part'], [])`.
- **One race is still there, as before:** a `.part` could finish between the 0.05 s poll and the SIGKILL. With the 0.5 s pause I did not see it happen.

### (2) The template and the sheet check
- **Template:** `stage2/tb/colab_steps.template.txt` holds the draft and its placeholders. `colab_steps.txt` is a byte-identical copy (same sha256). The four sheet tests now read the template.
- **The check:** `python -m designlab.stage2.nb sheet-check --sheet stage2/tb/colab_steps.txt --manifest stage2/tb/bundle/s2-b1/MANIFEST.json --ref <C1> [--acceptance PATH] [--before-smoke] [--record PATH]`. It exits 0 when the sheet is OK, 1 when it fails, and 4 for a ref that is not 40 hex. It checks:
  - **fields:** no placeholder is left.
  - **REPO_REF:** in both parameter tables, every 40-hex value, the Colab address (the SHA, or a tag here that points at it) and its notebook path at C1, and line 1.
  - **Bundle id:** the manifest equals the one at C1 byte for byte; the tables and "bundle X, manifest Y" match.
  - **Content hash:** the hash recomputed from the manifest, the 12-character prefix, the file and part counts, and line 2.
  - **Every other sha256:** must be, at C1, the hash of a file named just before it. This is how the rules texts are checked.
  - **BUNDLE_SOURCE:** one value in both tables. Every `git:<branch>` must hold each part with the manifest's sha256; `drive` must name every part.
  - **Lines 1-3:** must be lines the acceptance record printed. The record itself must be whole, `ok: true`, and of this commit and bundle.
  - **Section 8:** its run-directory values must equal the record's, and the wrong sentence must be absent.
- It reads only local refs (`origin/<branch>`, then `<branch>`) and never fetches, so fetch the data branch first.
- **Test:** `test_a_filled_sheet_is_held_to_the_records` fills the template in a scratch repository. It passes, and each wrong value fails its own check and no other. There are also a fields test and a command-line test.
- **Dry run on the V13 sheet:** I ran the check, read-only, on the uncommitted filled sheet in `/home/user/mesozoic-wt/c2`, with `--ref b7527a66… --before-smoke` and that worktree's `ACCEPTANCE.json`. Exit 1, failing only:
  - `lines: ACCEPTANCE.json says ok: False (failed: resumed_from_drive)`
  - `section_8: it says "while the first was alive"…`

  Everything else passed, including the three rules' sha256 at C1 and the parts on the data branch.

### (3) Section 8
- The template (and its copy) now tells the acceptance run's run-directory lines in `_drill`'s order:
  1. the new run;
  2. the session stopped during a copy, after an earlier save was whole;
  3. the start refused while LIVE.json was `<<LIVE_S>>` s old, although nothing was running;
  4. the resume from `<<RESUME_STEPS>>` steps with TAKEOVER, which only the lead sets (the owner waits ten minutes, 5.6), and whose printed line is the same without it.
- `<<LIVE_S>>` and `<<RESUME_STEPS>>` are new fields of the acceptance run, listed under STILL OPEN. The sheet check holds both to the record.
- **Test:** `test_section_8_of_the_template_says_when_the_refused_start_came`. It fails on the old template.

### Decisions for you
- **"No placeholder left" cannot pass at C2.** The fields only the smoke can fill are still open then: 21 in the V13 sheet, including `<<LIKE THIS>>` and `<<...>>`. By default the check requires none left. With `--before-smoke` it lets only those fields stay (`SHEET_SMOKE_FIELDS`) and lists them, so use that flag at C2 and the default after the smoke.
- **TABLE OF LAUNCHES** counts as a smoke field, because the template says "FILLED IN AFTER THE SMOKE". V13 filled it anyway.
- **The V13 sheet was built on the old template.** Its section 8 needs the corrected wording and the two new fields. After the rerun, its "from 0 steps" will no longer match the record.

### Not done
- No acceptance run and no Colab.
- The `STAGE2_SLOW` `local_run` test was not run.
- I did not capture a baseline test summary before my changes: the doubled `-q` (my flag plus the repo's addopts) hid it.
- I did not change the notebook or `requirements-colab.txt`. The notebook's intro still links the sheet at the tag `s2-launch-1`, and no such tag exists on the remote.

### Commits (branch `wt/wpc-cn`, head `6b70887c62204b52e9b293508dbae9086e814de6`)
| Commit | What |
|---|---|
| `fa54c61ef0537359360412c59ed4f6832f55d586` | merge of the working branch (b45424a9) |
| `a6c322c4384fd6e0215e2044f08f1843bf939edf` | (1) the kill condition, with its unit and plumbing tests |
| `f7deb1c7f25ccefdf6edd32bebc1fb8d2f54edea` | (2) the template and `sheet-check` |
| `95b78a522b7b483b151923c17f7e1ee4911275d1` | (3) section 8 |
| `6b70887c62204b52e9b293508dbae9086e814de6` | (1) the live-launcher test |

Every message ends with exactly the two required lines and names no model anywhere else. The worktree is clean. The main checkout's `git status --porcelain` is empty, at b45424a9.

### `git diff claude/mesozoic-trex-stage2-refine-4jffre --stat`
```
 hardware/trex_form_study/designlab/stage2/nb.py    | 363 ++++++++++++++++++-
 .../stage2/tb/colab_steps.template.txt             | 385 +++++++++++++++++++++
 hardware/trex_form_study/stage2/tb/colab_steps.txt |  10 +-
 hardware/trex_form_study/tests/test_stage2_nb.py   | 359 ++++++++++++++++++-
 4 files changed, 1092 insertions(+), 25 deletions(-)
```

### sha256 (files under `/home/user/mesozoic-wt/wpc-cn/hardware/trex_form_study/`)
| File | sha256 |
|---|---|
| `designlab/stage2/nb.py` | `06629a1be220f2138dddc8df0558a500f7076cf4d00c3531063f506ff2b5b6cc` |
| `tests/test_stage2_nb.py` | `bf2907a9c33f547511380cbdf92a5f82b921da4f4b8245aa41fe7003ff8ddfbb` |
| `stage2/tb/colab_steps.template.txt` | `a2cdaaa83c0cbb9a685ad9a7d5b2fd1b3b93584cf4f00915e20050a96da6cb3a` |
| `stage2/tb/colab_steps.txt` | `a2cdaaa83c0cbb9a685ad9a7d5b2fd1b3b93584cf4f00915e20050a96da6cb3a` |
| `stage2/stage2_training.ipynb` (unchanged) | `b2ca42eee693a76e467354928d6366edef68104e1639d5a48013228bfddf65bb` |
| `stage2/tb/requirements-colab.txt` (unchanged) | `8bd4d8dd5d2e8d09ad1544a58eba2ca2ea0d59b7fa6383180f3818be7cc266da` |
