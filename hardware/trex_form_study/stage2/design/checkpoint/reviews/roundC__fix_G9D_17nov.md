Ruling 42 is done: G9-run (d) now passes on the committed V6/75-M-H-10 records (read-only probe, `ok True`). The two test files pass, and the three new tests fail against the old code. Two commits on wt/wpc-cw; nothing pushed. Simulation only.

**Merge.** The worktree was clean. `git merge --ff-only claude/mesozoic-trex-stage2-refine-4jffre` fast-forwarded 887fad6e to 8c8430ed, so there was no merge commit.

**What changed**
1. `designlab/stage2/gates2.py`, commit 1f1de072
   - `part_d` works out `_expected_conditions` first and reads only those companions for the verdict.
   - A new helper, `_not_read`, handles every other companion. It goes under `not_run` when the bar record it names is present, has that sha256, says "not run" and has 0 episodes, and the companion also has 0 episodes. Anything else goes under `not_expected` with what was found. Neither list fails (d).
   - An expected condition's "not run" companion still fails with "its bar record is 'not run', not done". A missing one still fails through `missing`.
   - An unreadable or non-companion file still goes to `problems` and fails (unchanged).
   - The module's reading in `RULES["z2_run"]` now says this and cites ruling 42 ("a code error, the rule unchanged"). The g9-run command line now also prints `not run [...]`.
   - Under the ruling, a companion of an unexpected condition that *did* run is no longer checked; it only appears under `not_expected`. There is no such companion in the committed records.
2. `designlab/stage2/lookbar.py`, commit c60ff993
   - `run` passes `rule.cited` to `collect(..., g9_ref=)`, and `collect` passes it to `companion`.
   - On a look set, a "not run" record with no episode now cites that record's path and sha256 (ruling 37 (c)). Companions that ran keep citing what their episode blocks hold. Off the look sets nothing is cited.
   - A kept companion that cites nothing is rewritten when a citation is given; one that already cites a record keeps it.
   - `RULES` and the row description are unchanged: the committed record's `row_sha256` still equals the current row's (`True`).

**Tests that fail without it.** With the 8c8430ed `gates2.py` and `lookbar.py` checked out (then restored), the result was "3 failed, 115 deselected in 2.17s":
- `tests/test_stage2_gates2.py:1506`: `assert (False is True)`, in `test_g9_run_part_d_lists_the_companions_of_conditions_not_run`.
- `tests/test_stage2_lookbar.py:2333`: `TypeError: collect() got an unexpected keyword argument 'g9_ref'`, in `test_the_companion_of_a_condition_not_run_cites_the_g9_ref_record_as_the_others_do`.
- `tests/test_stage2_lookbar.py:2373`: "At index 0 diff: ('turn/q90_right', 'not run', None) != ('turn/q90_right', 'not run', {'path': ..., 'sha256': '5ee875b9...'})", in `test_run_gives_collect_the_g9_ref_record_that_let_its_rows_in`.

The gates2 test also checks that an expected condition's "not run" companion fails and that an absent one fails (`missing`). The first lookbar test asserts the not-run companion holds `g9_ref`.

**Test run.** `tests/test_stage2_gates2.py tests/test_stage2_lookbar.py` gave "101 passed, 17 skipped in 214.30s (0:03:34)", exit 0. The skips were:
- 9 need the frozen sets' arrays or the look ladder (restore groups); line 1534, `test_g9_run_expects_every_gated_condition_and_walk_nominal`, is one of them.
- 1 needs STAGE2_SLOW (gates2:1871).
- 6 are slow lookbar tests (lookbar:2382).

**Probe.** `gates2.part_d('stage2/tb/bar/r0l','75-M-H-10','V6','seeds3042')`, in process, writing nothing:
- `ok True`; `records 13 all ok True`; `missing [] problems []`.
- `not_run ['turn/q90_left', 'turn/q90_left_mu0.54', 'turn/q90_right', 'turn/q90_right_mu0.54']`. Each reads "not run", 0 episodes, 0 bar episodes, own sha256 ok, reason "no reference within the thigh twist and the leg drives: the built reference is over a leg drive's Stage 0 limit".
- `not_expected {}`; `bar_records_10ms 7 True`.
- I did not run the g9-run command, write a new `g9_run.json`, or keep the old one as `g9_run.first.json`.

**Hashes**
- Bit-identity: `45719d030359ec7d5d3914af3814d837fdc05d77652ba45c64792b511027528d c0027992de21067525cc4489b74acb157feb89f3a4c7618580addc2b341be1c1` (matches).
- `bar.py` 21ac6393... and `moves.py` ccee9b2c... are untouched.
- `gates2.py` df3f07299a9bdafc73a2caa310d16ae9d022cd2fbfbe8a4ab6a7b7a66552b0d4 (was 2537cc2b...).
- `lookbar.py` 6d919fbd8febfdb3ce6e82ef220ce121cfbb99530a48a37c8a6c3ee0e7a3bf55 (was 6a7d59e4c116289b3ab0eaec2786a49e46d9d6a0072769e93aa3c0177e89ece4).

**Decisions for the lead**
- **Re-running R0L would redo its episodes.** `lookbar.py`'s hash is part of every look row's bar-record key, and the committed R0L records carry 6a7d59e4. A new `lookbar run` of R0L there would find a different key and run those conditions again. A `lookbar collect` would rewrite all 17 companions, because their `inputs_sha256` includes the sources.
- **The committed not-run companions still cite nothing.** The four q90 companions still hold `g9_ref: null`. The command-line `collect` gives no citation, so only a new `lookbar run` would add it. I wrote none of this; the probe's verdict does not depend on it.

**git diff claude/mesozoic-trex-stage2-refine-4jffre --stat**
```
 .../trex_form_study/designlab/stage2/gates2.py     | 60 +++++++++++++++++-----
 .../trex_form_study/designlab/stage2/lookbar.py    | 47 ++++++++++-------
 .../trex_form_study/tests/test_stage2_gates2.py    | 58 +++++++++++++++++++++
 .../trex_form_study/tests/test_stage2_lookbar.py   | 60 ++++++++++++++++++++++
 4 files changed, 194 insertions(+), 31 deletions(-)
```

**Commits and head.** 1f1de072 (gates2 and its test), then c60ff993 (lookbar and its tests). Branch `wt/wpc-cw` head is c60ff99366340ff538cf73fac498de27b9c616bb; the worktree is clean and the main checkout untouched. Pytest temp dirs are deleted; the logs are in `/tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/g9fix/full2.log`.
