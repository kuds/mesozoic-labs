**Verdict: merge after the one listed fix.** The fix is finding 1, one line and a test, in a part of the code that nothing reaches yet. Everything else I found is a minor point or a question for the lead.

## Findings, most serious first

**1. Medium, latent (fix before merge). A drawn-panel repeat that has not run, or could not be judged, counts as a failure.**
- Where: `gate1.summarize`, `designlab/stage2/gate1.py:651-652`: `v = word == "replicated"`.
- Effect: for a row that gives a verdict, "repeat not run" and "repeat not judged" both turn the D record's verdict to False. Condition 2 then reads "a judged record does not pass".
- This breaks C8: a run that cannot be read is run again and never counts. The bar itself keeps a pending state for this (`bar.REPLICATION_PENDING`).
- Probe (`check/probe_summary.py`, hand-written SV5 records, nothing run):
  - all repeats run and pass: `holds_if_registered=True`
  - one repeat not run: `repeat='repeat not run' verdict=False holds_if_registered=False reason_if_registered="a judged record does not pass: {'SV5/R5/42': True, 'SV5/R5/43': False}"`
  - one repeat not judged: the same result, False.
- Why it is latent: SV0 and SV2 give no verdict, and `RULES_REGISTERED` is False, so `holds` is forced to None. But `holds_if_registered` is already printed.
- Fix: map "replicated" to True, "not replicated" to False, and anything else to None. Add a test case for each.

**2. Medium, for the lead. The runner does not apply the bar's run-placement rules (`bar.RULES["checks"]`, `_check_rule`).**
- On a scratch or fixture set it will run SV2 rows (R2 inside). Amendment 5 says never an R2 row there.
- On V0, F25F9 and F25F9cl it runs any number of episodes with no task, no budget and no listing of development episodes.
- The design's S4 row says these runs are "listed as development episodes", and the runner has no way to do that.
- The builder raised the missing budget, as their (e), but not the listing or the R2-on-scratch point.
- Options:
  - a `--dev-episodes` listing, and refusing SV2 on sets that are not registered;
  - or the lead rules that gate1 runs need neither.

**3. Minor, for the lead. `V-` and `W1` runs that leave the supply window are still counted as stayed up.**
- Each run's `left_supply_window` and the record's `runs_left_supply_window` are there.
- But `counts.survived` and the summary's "Up" column include those runs.
- Design section 4, the `V-` row: a run with a step below 20 V "is recorded as 'left the supply window', not as stayed up". Both settings are reported only, so no verdict moves.

**4. Minor. `--cut-s` does not work as its help text says.**
- The help says "a smoke check: not judged". But on a walk the bar's judge raises `Unreadable` for every cut episode.
- So each such episode is recorded as one that failed to run, and is run again on every later run. The builder's own test shows this.
- It needs a sentence in the help text, or the option limited to tests.

**5. Minor, scope.**
- `r5env.describe()` gains a `mid_start_rule` key when the option is on. That is one code line beyond the "rule-text additions only" remit. The builder raised it as (b).
- `train2.sources()` moves, so a training run started before S3 resumes only with `--allow-source-change`. The builder raised it as (h).
- Design line 28 still says the code lines are those of `82ba1ad`. The refresh note sits in 5.1's row instead, because line 28 is outside the remit. The builder raised it as (g).

**6. Note.** `GateWorker` checks during a session that the bar's and the panels' sources have not changed. It does not check `servo.py` or `gate1.py`, while each record's key holds their hashes as taken in the main process. The bar has the same gap for row modules registered from elsewhere.

**I agree with the builder's other points for the lead:**
- (a) Under `servo="draw"` the equivalence check can never come out identical, because the row the bar runs is not served.
- (c) The runner refuses look sets, but gate 1 runs on the chosen look's walk, so someone must rule how served rows join the look-set rule.
- (d) A stand-in row's D record is sent to the second block when its headline reading passes; the bar never replicates rows that give no verdict.
- (f) N and the reported settings are read by no rule.

## Checked and found right

**Point 1: `servo.Served`.**
- It has `bar.Delayed`'s form, with the order delay, then trunk mass, then layer. `describe()` gives the setting, the table's sha256, `verdict` False and `attributed` False.
- Its `sources()` add `servo.py`, and relax is refused.
- `SV0/<setting>` and `SV2/<seed>/<setting>` are registered through `panels.register_rows`; `panels.py` is not edited. SV5 is a documented hook only and is not registered.
- D draws `panel_draw(bar seed)`, which is `SeedSequence([s, 3])`, with the rotor inertia and the bus resistance at nominal.

**Point 2: the readings.** Each run records whether it left the supply window and each leg drive's share of steps inside its play. Each record sums them through `record_readings`, reported only.

**Point 3: the runner, `gate1.py`.**
- It imports the bar's machinery and edits none of it.
- Pass rule:
  - a corner passes on `survived == 20`, with the other limits recorded only;
  - D passes on `bar.counts(...)["pass_"]`, reading 2 (20 stay up, at least 17 meet every limit);
  - a record is judged only on `walk/v0.3`, grippy floor, at 4 or 10 ms, uncut, with a block's 20 seeds in order and no failed episode.
- The second block takes D only, and only a record whose headline reading passed.
- The record has its own schema and key; the bar's and the panels' hashes are held as data.
- It refuses `--out` under `stage2/runs/`, `stage1/` or the sets' root, and refuses look sets.

**Point 4: G8s.** It now compares the env against the registered `SV0` row, and its tests pass.

**Point 5: the S2 follow-ups.**
- `s2rows` passes `episode_seed` only when the config has `servo="draw"`.
- The refusal test now matches `SERVO_RULE`.
- `SERVO_MID_START_RULE` is a new string, and its wording agrees with the fourth list. The 8 old strings are pinned by sha256.

**Design file.**
- A word diff shows only `r5env.py:NNN` references and 5.1's Training env row changed.
- I checked every new reference against `r5env.py` at HEAD, and each points at the right lines.
- The amended row says what the fourth list asks: copies at entry, model gains at the first physics step.

**Tiny run on V0 (development probe, into scratch, not results).**
- `SV0`, one seed each (3042), whole `walk/v0.3` at 4 ms:
  - `D@4ms` (set D/3042) fell at 5.144 s;
  - `M+@4ms` stayed up and met every limit;
  - `P+@4ms` stayed up and failed `duty` and `toe_clearance_own_swing`.
- None was judged (one seed, not a block); verdict None; attribution "not attributed".
- Each record holds the row, set, setting, delay, seeds, the table's sha256 `342f3948cef1…`, the row's served description and the runner's sources.
- After the three rows the worker's plant equals a freshly built plant (`model_differences` == []), with no step hook, its own `Envelope`, and kp and kv the same.
- These three are development episodes on V0 that no file lists. The lead may want to list them.

**Mutants** (scratch copies loaded through an import redirect; no tracked file was touched). A copy's file path alone fails 2 tests in servo and gate1 (the sources are keyed by path), and I discounted those. Every mutant fails at least one other test:

| Mutant | Result |
|---|---|
| servo.py before S3 | 5 failed |
| episode run outside the layer | 26 failed |
| trunk mass not undone | 16 failed |
| one D set for every run | 8 failed |
| stand-in rows given a verdict (`NO_VERDICT_CONTROLLERS=()`) | 5 failed |
| supply-window count changed | `test_record_readings_count_the_runs` fails |
| corner judged by the whole rule | `test_the_pass_rule…` fails |
| repeat need not pass | `test_the_repeat_must_pass_too…` fails |
| second block for every setting | the dry-run and second-block tests fail |
| `gate1.verdict` ignores the stand-in list | `test_the_pass_rule…` fails |
| slippery floor judged | `test_the_pass_rule…` fails |
| g8s.py before S3 | 22 failed |
| s2rows.py before S3 | `1 failed, 22 deselected` |
| r5env.py before S3 | `2 failed, 1 passed, 46 deselected` |
| env's early refusal removed | `1 failed, 48 deselected` |
| an old rule string edited | `1 failed, 48 deselected` |

**Hashes, regress check and lint.**
- The two hashes are `45719d030359ec7d5d3914af3814d837fdc05d77652ba45c64792b511027528d c0027992de21067525cc4489b74acb157feb89f3a4c7618580addc2b341be1c1`. They were the same after my probe run.
- `TABLE_SHA256` is `342f3948cef116ea7fde49106817596a3d69ffa8d85c84dcb7c35f442546ef57`, and recomputing it gives the same value.
- regress check: `stage2/regress/regress.json: current (full tier, 256 of 256 episode runs matched, 248 distinct committed episodes, the four source hashes are today's)`, exit 0.
- ruff: `All checks passed!`

**Files and git state.**
- Only the 11 package files differ from `claude/mesozoic-trex-stage2-refine-4jffre`.
- `git status` is empty in the worktree and in the main checkout, and nothing in the main checkout's `stage1/` or `stage2/` changed.

**Dry run** (my own; V0, rows `SV0,SV2/42`, settings `@named,D`, delays 4 and 10). It wrote nothing.
- `Planned: 64 records, 1280 episodes, 7.519 simulated hours, 0.950 core-hours (1.046 with the layer's estimate).`
- `Second block, only for a drawn-panel record whose headline reading passes (P41): up to 4 records, 80 episodes, 0.059 core-hours (0.065 with the layer's estimate).`

**Merge readiness.**
- `git merge-tree --write-tree claude/mesozoic-trex-stage2-refine-4jffre wt/wpc-s1` gives `470e53499d9bd259629918d668ceef0685e906de`, with no conflict. That is `wt/wpc-s1`'s own tree.
- The working branch's tip `cab73e8b` (local and origin) is an ancestor of `3afc8204`, so the merge is a fast-forward.
- All 7 commits (`9d0a982b` to `3afc8204`, the merge included) end with exactly the two required lines and name no model anywhere else.
- `wt/wpc-s1` is not on the remote.
- Tip to merge: `3afc8204`, or the commit that fixes finding 1 on top of it.

## Exact test summary lines (my runs, merged tree at 3afc8204)

| Tests | Result |
|---|---|
| `tests/test_stage2_servo.py` | `63 passed in 3.50s` |
| `tests/test_stage2_gate1.py` | `10 passed in 17.04s` |
| `tests/test_stage2_g8s.py` | `27 passed in 20.14s` |
| `tests/test_stage2_r5env.py` | `46 passed, 3 skipped in 86.70s (0:01:26)` |
| `tests/test_stage2_s2rows.py` | `18 passed, 5 skipped in 117.04s (0:01:57)` |
| `tests/test_stage2_g8a.py` and r2lfit's import test | `9 passed in 14.87s` |

The skips were already there: the -X body's tests, and the tests that need `STAGE2_SLOW`.

## Not checked
- The full test suite, including the LT4 merge's gates2 and lookbar tests and the slow skipped tests.
- Any whole-length G8s run beyond the tests.
- Runs with `procs > 1` (spawned workers).
- Any run on F25F9, F25F9cl or a look set.
- Any measurement for S4.
- The SV5 path, because it is not built.

## Files
All in `/tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/servo_s3/check/`:
- `probe_v0.py` (the V0 run; its development records are in `probe_out/`)
- `probe_summary.py` (finding 1)
- `run_mut.py` and the `mutplug/` folder (the mutant driver); results in `mut_results1.txt` and `mut_results2.txt`. I deleted the copied source files it reads from `mut/`; it needs them copied again to rerun.
- `dryrun_V0.md` (the dry run's output)
- `design_wd.txt` (the word diff of the design file)

No pytest temp directories remain.
