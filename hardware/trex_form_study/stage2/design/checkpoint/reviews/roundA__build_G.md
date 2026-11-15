# build:G (roundA): the agent's final report

WP G is complete on branch wt/wp-g in /home/user/mesozoic-wt/wp-g: the full tier matches 256 of 256 episodes (trace sha256 and episode record) both before and after the `control.py` edit.

**Commits** (after a fast-forward merge of wt/wp-r, 70cd798):
- 2e31543: `designlab/stage2/regress.py`
- 0995bba: `tests/test_stage2_regress.py`
- afc0a3d: record of the unedited tree
- d1a7de7: a failed eval matches nothing, with its test
- fbb2534: the `control.py` edit and `test_window_strides_key`
- e6ca00a: `stage2/regress/regress.json` made at fbb2534

**Acceptance**
- Full tier before the edit (at 0995bba, also given `--record`): exit 0, 256 of 256, 554 s of evals.
- Full tier after the edit (at fbb2534, with `--record`): exit 0, 256 of 256, 550 s of evals.
- `regress check`: exit 0, "current (full tier, every episode matched, the four source hashes are today's)".
- The four test files: exit 0, "168 passed, 16 skipped in 162.66s (0:02:42)".
- `tests/test_stage2_regress.py` with STAGE2_SLOW=1: exit 0, "13 passed in 54.26s" (fast tier 8 of 8).
- `git status --porcelain`: empty, exit 0.

**Key numbers**
- 256 episodes per full run: 8 fast, 88 hard-floor, 160 Stage 1b turns. About 9.2 min wall each; roughly 0.35 core-hours used of the 2.
- The edit moved all four hashes: panels 6a316fc8 to c2863e38, Stage 1 package cc2a5c41 to a3f37228, tuner 1d936548 to c02b9a71, speed fit 93a96812 to 381daa82.
- No test in the three Stage 1 files pins a hash the edit moves.
- Negative control in a scratch copy (`per_ref` doubled): 0 of 4 match, traces equal, records differ at `strides`.

**Deviations**
- Six commits, not two. `run` refuses a dirty checkout, so each record follows the commit it names.
- The 160 Stage 1b episodes are four cases, one per turn, using `--conditions` instead of `--behaviors turn`, so a killed run loses one turn.
- `run` has an extra `scratch=` keyword. Kept cases are reused only on equal source hashes, library versions and `regress.py` sha256.
- The record has extra keys: `rules`, `commands`, `regress_sha256`, `environment`, wall times, and per episode `case`, `record`, `differs`, `note`.
- `check` also requires the record's episode list to equal today's full tier.
- Only the CLI calls `require_one_thread`; `run()` forces one thread on every eval regardless.
- Nine tests beyond the four named.
- `control.py` carries two trailing comments on the new lines.

**For the lead**
- The record must be made again at the freeze commit, after the Jetson body's merge, with `bash tools/lowprio.sh $PY -m designlab.stage2.regress run --tier full --record stage2/regress/regress.json`. Until then `test_regress_record_is_current` fails, by design.
- Commit the merge first, then run, then commit the record.

Nothing left undone.
