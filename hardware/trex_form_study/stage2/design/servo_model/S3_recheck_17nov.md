**Verdict: merge as it stands.** It is a fast-forward to `9af9ceab`. All seven points of the fifth list are done, and each one has a test that fails without it. Finding 1 below is a wording point in a column that is reported only; the lead can take it at the next edit.

## Findings
1. **Minor, wording.** For V- and W1, the summary's "Left supply window" cell says `N, of them M not up`.
   - N counts every run that left the window, falls included. M counts only the runs that left it while still on their feet.
   - Probe (`check/probe_supply.py`, hand-made episodes): 1 run fell after leaving the window and 3 stood. The row reads `| SV0/V-@4ms | reported | 16/20 | 16 | ... | 4, of them 3 not up |`. That reads as if 1 of the 4 stayed up, but it fell.
   - The counts are right: survived 16, falls 1, left_supply_window 3, by_the_bar 19/19, and the same in both readings.
   - Fix if wanted: change the cell's wording, e.g. "4; 3 on their feet, counted as left the supply window". The code is in `gate1.summary_text`; the test string is in `test_the_summary_names_runs_that_left_the_supply_window`.
2. **Cosmetic.** A run refused for its cap leaves an empty `--out` directory behind (made for the lock), while its message says "nothing was run or written". It writes no file and no listing entry; I checked this on V0 from the command line with `--max-episodes 0`.
3. **For the lead, not fixes:**
   - `check_rows` refuses only SV2. The bar's rule on scratch sets is "only R0 rows run there", so SV5 (R5 inside) will need the same refusal once it is built.
   - SV2 is refused on S1, but the bar exempts S1. This is the fixer's reading.
   - `stage2/tb/gate1_dev_runs.json`:
     - It is not git-ignored, as the fixer said.
     - `DEV_RUNS` is built from `s2io.STUDY`, so a run started from a worktree lists itself in that worktree.
     - Its entries have no reason field. X1 asks for one in `dev_episodes.json`; the fifth list does not ask for one here.
   - Point 6 gives "today" for every servo option, not only draw. This is the fixer's (c), and `train2.sources()` moves again.

## Each point
Mutants: 19 scratch copies, each with one edit, loaded by an import redirect that keeps the real `__file__`. No tracked file was touched. The unchanged copy loaded the same way gives `19 passed in 20.36s`.

1. **Repeat verdict.** `REPEAT_VERDICT.get(word)`: replicated is True, not replicated is False, anything else is None. Two mutants (`word == "replicated"`, and `word != "not replicated"`) each give `1 failed, 18 deselected`, in test_a_repeat_not_run_or_not_judged_is_pending.
   - The earlier check's finding 1 probe (`probe_summary.py`) now gives:
     - one repeat not run: `repeat='repeat not run' verdict=None holds=None holds_if_registered=None`
     - one repeat not judged: the same, None.
     - all repeats pass: `holds_if_registered=True`.
2. **Placement, listing and cap.**
   - Placement: check_rows made a no-op gives `2 failed, 17 passed`; check_rows dropped from run gives `1 failed, 1 passed, 17 deselected`. From the command line, `dry-run --set T0 --rows SV0,SV2/42` is refused with "amendment 5".
   - Listing: never listed gives `1 failed, 18 passed`; only the "done" entry listed gives `1 failed, 18 deselected`.
   - Cap: no cap gives `1 failed, 18 passed`; `>=` in place of `>` gives `2 failed, 17 passed`.
   - The entry holds the command, set, rows, settings, delays, conditions, seeds, count, cap and session. It is written before any episode runs and again when the records are written. `dev_episodes.json` is not touched.
3. **Supply window.** V-/W1 given the bar's counts: `2 failed, 17 passed`. W1 left out: `1 failed, 18 passed`. Summary column as before: `1 failed, 18 passed`.
4. **`--cut-s`.** A cut episode sent to the bar's judge: `4 failed, 15 passed`. The old help text: `1 failed, 18 passed`. The help now begins "for tests only".
5. **Worker source check.** No check: `2 failed, 17 passed`. Hashes not passed to the pool: `1 failed, 18 passed`.
6. **Equivalence check.** The env given the episode seed again: `1 failed, 22 deselected in 13.03s`. No servo option: `1 failed, 22 deselected in 12.47s`.
   - Probe, servo="draw" on S1 for 0.1 s:
     - with the fix: `{'option': 'draw', 'given': 'today', 'setting': 'today'} identical True`
     - with the old options: `setting 'train/3043' identical False`.
   - In s2rows.py only the equivalence check's servo handling changed.
7. **Second block.** Plan selects stand-in rows: `2 failed, 17 passed`. No stand-in branch in the repeat word: `2 failed, 17 passed`.

## Exact test lines (my runs, tree 9af9ceab)
- `tests/test_stage2_gate1.py`: `19 passed in 21.34s`
- `tests/test_stage2_servo.py`: `63 passed in 3.53s`
- `tests/test_stage2_s2rows.py`: `18 passed, 5 skipped in 108.66s (0:01:48)`. The skips are the slow tests and the -X body tests, as before.
- `tests/test_stage2_g8s.py`: `27 passed in 20.06s`
- `tests/test_stage2_train2.py::test_sources`: `1 passed in 1.96s`
- regress check, exit 0: `stage2/regress/regress.json: current (full tier, 256 of 256 episode runs matched, 248 distinct committed episodes, the four source hashes are today's)`
- Hashes: `45719d030359ec7d5d3914af3814d837fdc05d77652ba45c64792b511027528d c0027992de21067525cc4489b74acb157feb89f3a4c7618580addc2b341be1c1`.
- TABLE_SHA256 is `342f3948cef116ea7fde49106817596a3d69ffa8d85c84dcb7c35f442546ef57`, and `table_sha256()` recomputes the same value.
- ruff on the four changed files: `All checks passed!`
- V0 dry run, cost only, exit 0:
  - `Planned: 64 records, 1280 episodes, 7.519 simulated hours, 0.950 core-hours (1.046 with the layer's estimate).`
  - `Second block, only for a drawn-panel record whose headline reading passes (P41): up to 0 records, 0 episodes, 0.000 core-hours (0.000 with the layer's estimate).`
  - `Episodes a \`run --block seeds3042\` would start: 1280; cap 1280: within it.` With a cap of 1279: `over it, so \`run\` refuses it (RULES['cap']).`

## Merge readiness
- The working branch's tip `37e0695b` is the same locally and on origin. It is an ancestor of `9af9ceab`.
- `git merge-tree --write-tree` gives `70f19013…`, which is HEAD's own tree, with no conflict.
- Against the working branch, only the 11 package files differ. The fix round touched only gate1.py, s2rows.py and their two tests; servo.py was not edited.
- All 11 branch commits, merges included, end with exactly the two required lines and name no model anywhere else.
- `git status` is empty in the worktree and in the main checkout. Nothing changed under the sets.
- **Tip to merge: `9af9ceab22351c8796b8c97fc2db5b342569ef04`.**

## Not checked
- Runs with procs > 1 (the source check inside a spawned worker).
- Whole-length episodes.
- Any run on F25F9 or F25F9cl.
- The listing in a real run on V0. It is tested only with T0 taken as V0; my V0 run from the command line was refused at the cap and ran no episode.
- The full test suite, and the slow skipped tests.
- The SV5 path.

All files are in `/tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/servo_s3fix/check/`:
- `run_mut.py`, `plug/recheck_redirect.py` and the mutant copies in `mut/` (the mutant runs)
- `mut_gate1.txt` and `mut_s2rows.txt` (the mutant results)
- `probe_summary.py`, `probe_supply.py` and `probe6.py` (the probes)
- `tests_summary.txt` (the test lines) and `out_*.txt` (the full test output)
- `regress.txt` and `dryrun_V0.md`

The pytest temp folders are deleted.
