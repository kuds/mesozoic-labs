CW builder report, ruling 16 (the look sets' own rule). Worktree /home/user/mesozoic-wt/wpc-cw, branch wt/wpc-cw. Simulation only: nothing ran on a look set and no record was made.

## 1. Merging the working branch
- **First merge.** At the start, HEAD 097bfd64 was an ancestor of claude/mesozoic-trex-stage2-refine-4jffre (6c62f30e), so this was a fast-forward (`--ff-only`). There was no merge commit and no conflict.
- **Second merge.** The working branch moved to 3d261a79 while I worked (CF r2lfit, servo S1, the rules draft's sixth list). I merged it with `--no-ff -F` as c4b191dc. No conflict: it touches none of the package's files.

## 2. What changed
**designlab/stage2/lookbar.py** (bar.py is not edited)
- **`look_set_rule(set_name, body, row_names, root=setio.ROOT, g9_ref=None)`** is a contextmanager that yields a `LookSetRule`. It takes `body` because a G9-ref is per set and body; the brief's example signature had no body.
  - It applies only when the set is in `LOOK_SETS` or is a version of one (the name before the first dot, `look_set_of`), and every row is in `LOOK_SET_ROWS`: `R0L`, `R5/<seed>`, `R6/<seed>`, `R2L-<set>/<seed>` (`look_set_row`). A delay `@<ms>ms` is optional and must pass `bar.ms_steps`.
  - Only then does it replace `bar._check_rule` with `LookSetRule.bar_check` (same signature), and it puts it back in a `finally`.
  - `bar_check` lets an evaluation in only if it is a call by `bar.evaluate` itself. It reads that frame's `body` and `root` (`sys._getframe`), and the set, body, set root and rows must all be the context's.
  - Even then the evaluation runs only after `require_g9_ref` passes, read at that moment. Otherwise it raises `LookSetRefused`, which is both a `LookBarError` and a `bar.BarError`.
  - Every other call goes to the check that was in place when the context opened, with the same arguments. Nested contexts hand the evaluation on to the outer one.
- **`require_g9_ref(set, body, root, path)`** is gates2.require_g9_ref's check, built from gates2's own pieces (`gate_path`, `_read`, `SCHEMAS`, `record_sha_ok`, `_latest_freeze`, `cite`). It uses gates2's wording and also checks a version of a look set.
- **`run`** calls `bar.evaluate` inside the context. It also returns a new key, `g9_ref` (the citation, or None).
- **`episodes`** asks `rule.check()` before `open_set`, and only on a look set:
  - all rows in the four families: the G9-ref check;
  - any other rows: the bar's own `_check_rule`;
  - any other set: nothing is checked, as before.
- **`RULES["look_sets"]`** holds the rule text, so it is in every Track B row's description. The module docstring, the `run` and `episodes` docstrings and `__all__` are updated.

**Tests** (tests/test_stage2_lookbar.py, plus one change in tests/test_stage2_s2rows.py). Through `bar.evaluate`, with `bar._check_out` stubbed to stop right after the set check:
- `test_a_look_set_evaluation_of_each_allowed_row_runs_only_once_its_g9_ref_passed`
  - Rows tested: R0L, R0L with no delay, R5, R6, R2L, a mix of them, on V6, V6noF5 and the versions V6.2 and V6noF5.1.
  - Accepted with a passed G9-ref.
  - Refused with no record, a failed one, one of another set or body, a wrong schema, an altered sha256, an unreadable file, a stale freeze, or a set that is not frozen.
  - Also tested: a `g9_ref=` path given directly.
- `test_any_other_row_or_evaluation_on_a_look_set_is_the_bars_own_check_inside_the_context`
  - The bar's own refusals stand for R2/42, R1 (stock row), R9/1 (unknown), R5G, R2L-V6noF5 on V6, R0L@3ms, and mixes.
  - The same holds for another body, another set root, another set, rows outside the context's, and a direct call to the check.
- `test_on_v0_s1_and_a_scratch_set_the_bars_check_is_unchanged_inside_the_context`: 11 calls on V0, F25F9cl.2, S1 and T0 give identical answers with and without the context.
- `test_the_context_puts_the_bars_check_back_whatever_happens_and_the_bars_hash_stays`: after an exception, after a refusal, past the check, and with nested contexts, the bar's module attributes are identical and the registered hashes are unchanged.
- `test_the_g9_ref_check_is_gates2s`: word for word with gates2 on V6 in every case; for V6.2 gates2 returns None and lookbar refuses.
- `test_lookbar_run_and_episodes_go_through_the_rule`: run, the CLI (exit 1 with "rule X1"), and episodes.
- `test_the_rows_and_sets_of_the_look_sets_rule_and_its_text_in_every_row`, and new assertions in `test_what_is_relied_on_from_the_bar` (evaluate's code and locals, `_check_rule`'s parameters).
- Existing tests that ran episodes on a fixture set patched to be a look set now write a passed G9-ref into a temporary `gates2.GATES` and first assert the refusal: `test_episodes_on_the_fixture_set`, `test_episodes_catches_...`, and the s2rows `test_a_slowed_clock_...`.
- 12 deliberate breaks of the code (scratch copy, file restored after) were each caught by the tests: no G9-ref check, no restore, body ignored, root ignored, no check in episodes, run outside the context, R2L of any set, stale freeze ignored, versions ignored, no nested hand-on, rows not a subset, delay unchecked.

## For the lead
1. **Plain R0 rows on a look set.** I built the ruling's words ("every other row goes through the bar's own check unchanged"): `R0@<ms>` on V6 runs with no G9-ref, inside the context as outside. That includes `lookbar run --set V6 --rows R0@10ms` and `episodes` with an R0@ row.
   - Alternative: refuse any row outside the four families on a look set, inside the context and in `episodes`. It is a one-line change in `_decide` and `check`.
   - I recommend the alternative if X1 covers every closed-loop episode there. An R0@ row run through lookbar is R0L's episodes under another name.
2. **R2L set match.** `R2L-<set>` must name the evaluated set exactly; the row itself refuses another set's conditions. Alternatives: the base name, or any look set.
3. **R5G is not admitted.** The ruling names four families; D17 lists R5G as a Track B row.
4. **Reading 1 on versions.** The rule treats a version (V6.2) as a look set, but reading 1 still counts only on the exact names (`set_name in LOOK_SETS` in `episodes` and `collect`). Unchanged; your call.
5. **Gap in gates2 (not my file).** `gates2.require_g9_ref` checks only the exact names and lets a version through.
6. **Frame read.** The context reads `body` and `root` from `bar.evaluate`'s frame. bar.py is hash-frozen and a test pins the code object. The simpler alternative trusts the context's body, but then an evaluation of another body inside the context would pass on the wrong G9-ref.
   - The replacement is process-wide: a concurrent evaluation in another thread with exactly the same set, body, root and rows would also be covered. Nothing does that today.
7. **Monitor (CT, not mine).** `lookbar.episodes` on V6 now raises `LookSetRefused` before any episode when G9-ref has not passed. `monitor2._episodes` lets it propagate rather than treating it as a refusal of the monitor; CT should probably treat it as one.
8. **Stale texts (not mine):** the V3 NOTE in stage2/tb/run_trackb.sh (lines 170-172), and the design's 2.3 sentence. CF needs no change for this, as its own fix report expects: r2lfit's fit never goes through `_check_rule`, and R2L bar records come through `lookbar run`.
9. **Where the citation goes.** The G9-ref citation is only in `run`'s return value, not in any record. The companion schema is unchanged.

Not done: the slow tests (STAGE2_SLOW), any real G9-ref or look-set run, and any change to CF, monitor2 or gates2.

## 3. Test output
On the final head c4b191dc:
- `tests/test_stage2_lookbar.py tests/test_stage2_s2rows.py`: **"60 passed, 11 skipped in 128.95s (0:02:08)"**. All 11 skips are "STAGE2_SLOW=1". The baseline before my change ran 53 passed and 11 skipped, read from the dot line; it printed no summary line.
- Extra: `test_stage2_monitor2.py test_stage2_gates2.py test_stage2_lookread.py test_stage2_r2lfit.py`: **"141 passed, 33 skipped in 156.21s (0:02:36)"**.
- Ruff on the three files: "All checks passed!"
- Hash pair on the final head: `45719d030359ec7d5d3914af3814d837fdc05d77652ba45c64792b511027528d c0027992de21067525cc4489b74acb157feb89f3a4c7618580addc2b341be1c1` (the registered pair).

## 4. Diff and status
`git diff claude/mesozoic-trex-stage2-refine-4jffre --stat`:
```
 .../trex_form_study/designlab/stage2/lookbar.py    | 241 ++++++++++++-
 .../trex_form_study/tests/test_stage2_lookbar.py   | 392 ++++++++++++++++++++-
 .../trex_form_study/tests/test_stage2_s2rows.py    |   9 +-
 3 files changed, 623 insertions(+), 19 deletions(-)
```
`git status --porcelain`: empty. The tests wrote only to pytest's temporary directories; my scratch files are under /tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/wave7/cw16/ (outside the repo, so `git check-ignore` does not apply). The main checkout was not touched and nothing was pushed.

## 5. Commits, oldest first
- 9b60d803 (the rule, plus the existing tests it needs)
- 03043319 (the new tests, and the `__all__` line wrapped)
- c4b191dc (merge of the working branch at 3d261a79)

Branch head: **c4b191dc375efd0fbee9c373dcccf9eeb0306c95**
