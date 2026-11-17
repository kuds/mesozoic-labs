Verdict: merge

**Verdict: merge `wt/wpc-cw` at c60ff99366340ff538cf73fac498de27b9c616bb.** Both parts of ruling 42 are in the code, and an expected condition that is not done still fails (d). Each new test fails against the old code, and the two test files pass. The fixed `part_d` passes on the committed R0L records of V6/75-M-H-10. Three points below are for the lead; none of them blocks the merge. Simulation only. I changed no tracked file and committed nothing.

**Merge readiness**
- The branch is two commits on 8c8430ed: 1f1de072 (gates2 and its test) and c60ff993 (lookbar and its tests).
- The working branch has since moved to 0f5e8ac5, which adds one file: `stage2/design/checkpoint/reviews/roundC__build_RB60_17nov.md`.
- `git merge-tree --write-tree HEAD 0f5e8ac5` exited 0 with tree e325d6dc..., so the merge has no conflict. The builder had not merged 0f5e8ac5 because it came after their work. I did not merge it either, since I commit nothing.
- Only the four owned files change from 8c8430ed. `bar.py` (21ac6393...) and `moves.py` (ccee9b2c...) are untouched.
- Both commit messages end with exactly the two attribution lines and name no model anywhere else. The worktree and the main checkout are both clean (0 status lines).
- The bit-identity check printed `45719d030359ec7d5d3914af3814d837fdc05d77652ba45c64792b511027528d c0027992de21067525cc4489b74acb157feb89f3a4c7618580addc2b341be1c1`, which matches.
- File hashes: `gates2.py` df3f0729..., was 2537cc2b... `lookbar.py` 6d919fbd..., was 6a7d59e4...

**Ruling 42 (1): `part_d`**
- `part_d` now works out `_expected_conditions` first. A companion whose condition is not in that list goes to `_not_read` and never affects `ok`. The check of expected companions is unchanged, and so is the `ok` formula.
- Probe on the committed records, read-only, with both the new code and the 8c8430ed copy loaded from scratch:
  - New: `ok True`; 13 expected; `records 13 all ok True`; `missing [] problems []`; `bar_records_10ms 7 True`.
  - New, continued: `not_run ['turn/q90_left', 'turn/q90_left_mu0.54', 'turn/q90_right', 'turn/q90_right_mu0.54']`. Each reads "not run", 0 episodes, 0 bar episodes, own sha256 ok, reason "no reference within the thigh twist and the leg drives: the built reference is over a leg drive's Stage 0 limit". `not_expected {}`.
  - Old: `ok False`, `records 17`, the four q90 companions each failing with "its bar record is 'not run', not done".
- Edge probe on a scratch copy of those records:
  - With `turn/q90_left` added to the expected list: `ok False`, failing with "its bar record is 'not run', not done".
  - With the `walk/nominal` companion deleted: `ok False`, `missing ['walk/nominal']`.

**Ruling 42 (2): the G9-ref citation**
- `run` passes `rule.cited` to `collect`, and `collect` passes it to `companion`.
- On a look set, a "not run" record with no episodes now cites `dict(path, sha256)`, the same form the episodes' blocks give the other companions.
- A companion that already cites a record keeps it. One that cites nothing is rewritten once a citation is given.
- I recomputed the current R0L row: `row_sha256 equal: True`, so `RULES` and the row description are unchanged.

**Tests that fail without each change.** I loaded the 8c8430ed module from a scratch file through a pytest plugin, so the worktree was not touched.
- Old `gates2`: `tests/test_stage2_gates2.py:1506 ... assert (False is True)` in `test_g9_run_part_d_lists_the_companions_of_conditions_not_run`.
- Old `lookbar`, result `.FF`:
  - The existing control test `test_the_companion_names_the_g9_ref_record_its_episodes_ran_under` passed.
  - `:2333 TypeError: collect() got an unexpected keyword argument 'g9_ref'`. This one fails only on the new argument, not on the behaviour.
  - `:2373 ... At index 0 diff: ('turn/q90_right', 'not run', None) != ('turn/q90_right', 'not run', {'path': ..., 'sha256': '5ee875b9...'})`. This one fails on the behaviour.
- The parts of the gates2 test that check an expected condition's "not run" or missing companion would also pass on the old code. They guard against a regression rather than test the change, which is what they should do.

**Test run.** `tests/test_stage2_gates2.py tests/test_stage2_lookbar.py`, run in the worktree at c60ff993. The merged tree differs from it only by the one report file.
- Exit 0. pytest printed no "passed" summary line. The progress marks count 101 `.` and 17 `s`.
- The skips:
  - 9 need the frozen sets' arrays or the look ladder, at gates2 lines 746, 869, 892 (×2), 922, 1076, 1165, 1237, 1308 and 1534. Line 1534 is `test_g9_run_expects_every_gated_condition_and_walk_nominal`.
  - 1 needs STAGE2_SLOW (gates2:1871).
  - 6 are slow lookbar tests (lookbar:2382).
- The pytest temp dirs are deleted.

**Points for the lead (none blocks the merge)**
1. **Companions of other conditions are passed over in any state, not only "not run".** I left this as the builder wrote it; the lead can decide whether to tighten it. The ruling says to read only the expected companions; the check says only the unavailable conditions' "not run" ones are passed over. Edge probe:
   - I turned the `turn/q90_right` bar record into "done" with 20 episodes and re-pointed its companion at it. Result: `ok True`, with `not_expected {'turn/q90_right': "its bar record is 'done' with 20 episodes"}`.
   - A not-run companion with a broken own sha256 is still listed under `not_run` with `own_sha256_ok: False`, and `ok` stays True.
   - The builder's own test fixes the bar-record sha256 mismatch case as `ok True`.
   - V3 runs only `--conditions @gated,walk/nominal`, so such a companion would come only from a changed record. `not_expected` is empty in the committed records, so this does not change the V6 verdict.
   - If the lead wants it stricter, a small change would make `not_expected` entries, and a not-run companion whose own sha256 is broken, fail (d).
2. **The look rows' key moves.**
   - The committed `walk/v0.4` record's `row_sources` holds `lookbar.py` 6a7d59e4. The current row holds 6d919fbd, and `row_sources equal: False`.
   - So a new `lookbar run` of R0L into the same out directory would find a different key. Ruling 42 says validation goes on from G9-run, so `V3-r0l` should not be re-run.
   - I did not run `lookbar run` to see the re-run happen.
3. **The committed not-run companions still cite nothing.** `turn/q90_right` has `g9_ref None`, while `walk/v0.4` cites `stage2/tb/gates/V6/75-M-H-10/g9_ref.json` af627ba9.... The command-line `collect` gives no citation, so only a new `lookbar run` would add one. The probe's verdict does not depend on it.

**Not done by me**
- I did not run the `g9-run` command or `gates2.g9_run` on the committed records, and wrote no `g9_run.json` or `g9_run.first.json`.
- I did not run the STAGE2_SLOW tests or restore the set arrays, so the 9 tests that need them were skipped.
- I did not run the tests on the merged tree.

Scripts and logs are in `/tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/g9fix/check_fix/`: `probe.py`, `probe.log`, `edge.py`, `oldmods.py`, `rowsha.py`, `tests_full.log`.
