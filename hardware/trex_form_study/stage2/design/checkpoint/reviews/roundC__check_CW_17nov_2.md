# check:CW (roundC, 17 Nov, second attempt): the agent's final report

Working paper. Nothing here is a rule.

CHECK OF THE CW FIX ROUND (wt/wpc-cw at 4d9a4358)

VERDICT: merge as it stands. Every finding of roundC__review_CW.md and rulings 7 to 9 is handled the way the report says. Each fix has a test that fails without it: 33 of 35 reverts on a scratch clone were caught. The two that were not caught are low-severity test gaps; they block nothing and can go in a follow-up. I found no regression. The report checks out on every point I could verify.

FINDINGS, MOST SERIOUS FIRST (all LOW)

1. LOW: ruling 8's "without this machine's paths" is tested at one fault site of three.
   - Where: designlab/stage2/lookbar.py:470 (the clock, state_digest and policy_check faults) and :475 (on_command_at_stop).
   - Both now go through `_fault`. But when I put the raw `f"{key}: {type(exc).__name__}: {exc}"` back at either site, every fix test still passed ("f5_clock_fault_raw: SURVIVED", "f5_stop_fault_raw: SURVIVED").
   - Only the "look reading" fault (:484) is tested end to end. `_fault` and `_no_paths` are also tested directly.
   - Fix: in test_a_fault_in_the_reading_is_recorded_and_never_raised, make `lookbar.clock_block` and `on_command_at_stop` raise an error that names a path, and assert the cut form.

2. LOW: `_no_paths` (lookbar.py:232-246) leaves some absolute paths readable.
   - A path of three parts or fewer is cut to `.../<last three parts>`, which is the whole path.
   - A path containing a space is cut only up to the space.
   - My probe (probe.py):
     - `'/home/user/a'` gives `'.../home/user/a'`;
     - `"/home/user/b c/d"` gives `".../home/user/b c/d"`;
     - `'/content/drive/My Drive/mesozoic/stage2/sets/V6/x.npz'` gives `'.../content/drive/My Drive/mesozoic/stage2/sets/V6/x.npz'`.
   - It matters little in practice: the study uses `MyDrive`, with no space, and real data paths are longer than three parts.
   - Fix: never keep a path's first part, e.g. keep at most `len(parts) - 1` parts. Add a test case.

3. LOW, for the lead (outside ruling 8 as written): a refusal's text in look/summary.json holds absolute paths.
   - Seen in my probe: `refused[0]` and `refused_records[0].reason` both hold the output root's absolute path. They come from `companion()` at :864-865 (`f"{path}: "`, `under {sp.parent}`).
   - The other fields of the cell (`bar_record`, `record`, `moved_to`) are relative.
   - Ruling 8 names faults, and the summary's `provenance` also holds absolute paths, so I do not count this against the fix. Options: pass the text through `_no_paths`, or leave it.

4. Note, a rule: by its own account, the fixer ran `git fetch --dry-run` once. That contacted the remote, which breaks the no-network rule. I cannot see the call itself. FETCH_HEAD is unchanged since 14 Nov 21:09, which is consistent with a dry run that wrote nothing.

WHAT I CHECKED AND FOUND RIGHT

- **Per finding:**
  - F1 / ruling 7:
    - the key is renamed `reference_use`;
    - `rule_texts()` refuses a key both tables hold;
    - `describe()` and `companion()` use it;
    - the tests assert that lookread's text of rule U5 is under "reference".
  - F2: a hash-pair test (it also passes in an export without git), and a git-diff test against c377ba1 (it skips without git). It leaves out barreport.py; that choice is listed for the lead, and D18's speedfit.py is covered under designlab/stage1.
  - F3: the K4 test with roots A and B.
  - F4:
    - a companion of another bar record, or an unreadable one, goes to `<companion>.superseded/<12 hex>.json`;
    - one that names the current bar record is kept;
    - `refused_records` is in the summary and in summary_text;
    - the move uses os.replace within one directory, inside the lock.
  - F5: a fault is recorded as `Type: message`, with paths cut.
  - F6: set_check refuses task_names that are absent, empty or not a list, and plant_inputs that are absent or empty. Both are safe: `panels.key_context` always gives a 4-key plant_inputs, and train2 writes both fields.
  - F7 / ruling 9:
    - a warned episode gets `warned` True, `fall_time_s` 0 and `survival_s` 0, with `simulated_s` kept beside them;
    - the sidecar check and the bar's counts are unchanged, since `_counted` reads no survival;
    - monitor2.survival_credit and summarize credit it 0;
    - an old sidecar without the new keys is still read as warned by its fall cause;
    - a warned episode that the bar recorded as an error stays "error".
  - The trainer's owed points (rulings 10 and 11) are untouched, as the brief says.
- **Reverts on a scratch clone** (`git clone --shared` plus the ignored inputs; it is clean afterwards): 35 reverts, 33 failed a named test with FAILED.
  - Rerun 1 is void: its basetemp parent was missing, so tests that use tmp_path errored for that reason alone. Its log is kept as mutations_run1_invalid_basetemp.log.
  - The 33 caught reverts:
    - f1: 5 ways;
    - f2: bar.py, s2io.py and plant.py each changed;
    - f3: `_check_files` switched off;
    - f4: never moved, always moved, unreadable file kept, no cells, `.stale` name, summary_text silent;
    - f5: traceback back, no cut, `_fault` without the cut, study path not made relative, no lookbehind;
    - f6: task_names open, no type check, plant_inputs open;
    - f7: never warned, survival kept, fall time not 0, block without warnings, fall cause not read, block only, bar only, block without fall time, fall time from the bar only, fall time from the block only.
- **The merge (faee330a):** parents 9952f5b and fc505b0d. It brings in 12 documents only, with no package file. fc505b0d is 5 commits ahead of 8dd6958, and it is the working branch's tip, both here and in the main checkout.
- **Diff against the working branch:** the four files only, "4 files changed, 3918 insertions(+)".
- **Merge readiness:** `git merge-tree --write-tree` against the working branch's tip exits 0 with tree a89a535a, which differs from the tip by the four files only.
- **Record-key hashes**, at the start and at the end: `45719d030359ec7d5d3914af3814d837fdc05d77652ba45c64792b511027528d c0027992de21067525cc4489b74acb157feb89f3a4c7618580addc2b341be1c1`, the registered pair.
- **Nothing ran that should not have:** `git grep look_reading -- '*.json'` and `git grep -l R0L -- '*.json'` print nothing, so no committed record holds a row key that the new rule texts change.
- **Worktree and main checkout:** `git status --porcelain` is empty in both. The worktree's ignored-file list is identical to the one I saved before my runs. Since 01:50 the worktree has new files only from the merge and the commits.
- **ruff** (`--no-cache`) on the four files: "All checks passed!".

TEST LINES (mine; standing environment through tools/lowprio.sh; basetemp under the scratch directory)

- Fast, both files, in the worktree: "53 passed, 11 skipped in 127.18s (0:02:07)", exit 0. All 11 skips need STAGE2_SLOW=1; the state-A git test ran.
- `STAGE2_SLOW=1 ... tests/test_stage2_lookbar.py`: "43 passed in 94.71s (0:01:34)", exit 0.
- Acceptance, `STAGE2_SLOW=1 ... tests/test_stage2_s2rows.py -k "equivalence or r6"`: "4 passed, 17 deselected in 183.97s (0:03:03)", exit 0.
- An export of HEAD by git archive, without the ignored inputs or git: "26 passed, 38 skipped in 6.14s", exit 0. The git test skipped there and the hash test passed.
- Other packages' tests against the new rows, `tests/test_stage2_monitor2.py tests/test_stage2_lookread.py`: "72 passed, 22 skipped in 20.97s", exit 0. The skips: lookread's look-ladder inputs are absent, and monitor2's "lookbar is merged: the stand-in is no longer reached".
- Baseline of the 14 fix tests on the scratch clone: "14 passed, 50 deselected in 17.41s".

COMMITS TO BRING IN

- None lacks the attribution lines. All 13 commits not in the working branch end with two of them:
  - this round's five (faee330a, 5cfdfb67, 8888f6a1, 9c2d4323, 4d9a4358) carry the required pair and name no model anywhere else;
  - the 8 earlier ones carry the earlier model's lines, which were right when they were made.
- Bring them in with one merge of the tip, 4d9a4358fa16b64bb6226613f4322f15012867d4.

WHAT I DID NOT CHECK

- The whole slow s2rows file (I ran only the acceptance's `-k`).
- tests/test_stage2_train2.py: its fast tests use a stub of lookbar.episodes, and its slow tiny runs are training that this brief does not name.
- Any V0 or look-set episode, `lookbar run` through `bar.evaluate`, a 60 cm or -J body, a Colab runtime or Drive.
- A kill during collect's move-aside (read from the code only).

Scripts and logs are in /tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/rerun17/check_cw_17nov/:
- mutate.py
- mutations.log
- mutations_run1_invalid_basetemp.log
- probe.py
- probe_out/
- fast_worktree.log
- slow_lookbar.log
- slow_s2rows_acc.log
- export_fast.log
- other_fast.log
- ignored_before.txt
- test_lookbar.diff
- repo/ (the scratch clone)
- export/
