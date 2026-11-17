S3 report (servo design package S3). Worktree /home/user/mesozoic-wt/wpc-s1, branch wt/wpc-s1. This is simulation only. The servo values are a model and nothing was measured. No panel and no gate-1 record was made.

**1. Merging the working branch**
- At the start, HEAD f94f61cb was an ancestor of claude/mesozoic-trex-stage2-refine-4jffre (d39430bb). `git merge --ff-only` fast-forwarded to d39430bb, which holds S2.
- While I worked, the branch moved to cab73e8b (LT4: gates2.py, lookbar.py, r5lib.py, test_stage2_gates2.py, three reports). None of these is an S3 file.
- `git merge-tree` showed no conflict. I merged with `--no-ff -F` as merge commit 3afc8204, then reran every test and check below on the merged tree.

**2. Per point**

(1) `servo.Served(inner, setting)`, in /home/user/mesozoic-wt/wpc-s1/hardware/trex_form_study/designlab/stage2/servo.py, appended at the end:
- It has the form of `bar.Delayed`. It applies the setting's trunk mass before the inner episode and undoes it after (`servo.trunk_mass`, moved here from g8s). The inner episode runs inside `served(p, params)`.
- The setting can be a NAMED name, `D` (each run uses `panel_draw` of its own bar seed), `D/<s>`, or a ServoParams (gates and tests only).
- `describe()` gives the inner row's description plus the setting, `TABLE_SHA256`, the parameters, `verdict` and `attributed: false`. `sources()` gives the inner row's sources plus servo.py. `spec`/`from_spec` let the row be rebuilt in a worker.
- `relax` is refused: no served row is attributed. `gated()` is False for SV0 and SV2.
- Registration goes through `panels.register_rows` (panels.py is not edited). `SV0/<setting>` serves R0 and `SV2/<seed>/<setting>` serves R2/<seed>. `NAME_RULE` and `parse_name` follow the lead's fourth list.
- SV5 is a documented hook only: `SV5_HOOK` and `served_policy_factory`, never called, so SV5 is not registered.
- `NO_VERDICT_CONTROLLERS = ("SV0","SV2")` and `ATTRIBUTED_CONTROLLERS = ()`.
- `TABLE_SHA256` is unchanged: 342f3948cef116ea7fde49106817596a3d69ffa8d85c84dcb7c35f442546ef57.
- Tests are in tests/test_stage2_servo.py: names, registration and the SV5 hook, describe/sources/spec/no-verdict, one 0.2 s served episode (the D set of its own seed, mass during the run, readings, plant given back bit for bit), and the record readings.
- Mutants (each file restored byte for byte, checked by sha256):
  - servo.py as before S3: `5 failed, 58 deselected in 0.78s`
  - Served running the episode outside the layer: `22 failed, 2 passed, 66 deselected in 9.65s` (servo and g8s tests)

(2) The reported readings:
- Per run, the readings are the layer's own (`Layer.readings`). The runner stores them in every episode record as `servo.setting` plus `servo.readings`.
- Per record, `servo.record_readings` (`READINGS_RULE`, reported only) gives: runs that left the supply window, runs with a collapsed step, steps below 20 V, the lowest bus voltage, each leg drive's share of steps inside its play (mean and max), the mean over the leg drives, and runs without readings.
- Mutant (supply-window count): `1 failed, 62 deselected in 0.63s`.

(3) The runner: /home/user/mesozoic-wt/wpc-s1/hardware/trex_form_study/designlab/stage2/gate1.py (new).
- It imports and does not edit the bar's machinery: `bar.rows`, `choose`, `BLOCKS`, `BarPlanner`, `run_episode` (the episode and the bar's judge), `BarWorker` (subclassed as `GateWorker`, which adds the servo readings), `counts`, `record_key`, `read_partials`, `drop_partials`, and `tune.Pool`.
- Records have their own schema `gate1-servo-1` and their own key: the bar's key plus the setting, `table_sha256` and the runner's own sources. The panels' and the bar's sha256 are held as data. Records go to `<out>/<set>/<body>/<block>/gate1/<row>/...`.
- Runs resume per episode. A finished record with another key is refused, never overwritten.
- Condition 2's rule (`reading`):
  - a corner passes on 20 of 20 staying up; the other limits are recorded, not judged;
  - D is judged by the whole rule (reading 2);
  - other settings are read by no rule;
  - a record is judged only on walk/v0.3 on the grippy floor at 4 or 10 ms, uncut, with a block's 20 seeds and no failed episode.
- Second block: D only, and only when the headline reading passed. The summary prints replicated / not replicated / repeat not judged / repeat not run. The P41 rule (the repeat must pass too) applies to rows that give a verdict.
- Condition 2 "holds" only with served policy rows of two policies. While gate 1's servo rules are not registered, every record says development, `holds` is None, and `holds_if_registered` is shown beside it.
- `--out` is refused under stage2/runs/, stage1/ or the sets' root, and a run on a look set is refused.
- The dry run plans the headline block and the conditional second block. It costs them from the committed `stage2/runs/2026-11-15-75cm/COST.md` rates (R0 for SV0, R2 for SV2), also ×1.10 for the layer, which is the design's section 6 estimate. It runs no episode and writes nothing.
- Tests are in /home/user/mesozoic-wt/wpc-s1/hardware/trex_form_study/tests/test_stage2_gate1.py, on the synthetic set T0 under tmp:
  - The bar's judge cannot read a cut walk (`bar.Unreadable`). So cut-episode tests swap in a judge stub around the real served episode.
  - One whole episode goes through `bar.run_episode`. It is a smoke run on T0: it fell at about 5 s, the test does not assert staying up, and it is not a result.
  - A cut episode without the stub is recorded as a failed episode and run again on the next run (C8).
- Mutants:
  - a corner judged by the whole rule: `1 failed, 9 deselected in 0.80s`
  - the second block taking every setting: `2 failed, 8 deselected in 3.16s`
  - the repeat need not pass: `1 failed, 9 deselected in 1.02s`
  - a stand-in row giving a verdict: `2 failed, 1 passed, 7 deselected in 3.67s`

(4) G8s against Served:
- g8s.py now builds `SV0/<setting>` through servo's registered factory (stock form), and bar_form uses `bar.rows` `SV0/<setting>@<ms>ms`. ServedRow is removed, and the episode check compares `Served.params_for(seed)` with the env's parameters.
- All T13 cases are identical (N, F+, P+, V-, J+, M-, M+, and D at 3042, 3043 and 3044, at 0 and 10 ms, 1 s on S1), and bar_form on T0 passes.
- Mutant (g8s.py as before S3): `22 failed, 5 passed in 18.00s`.

(5) S2 follow-ups:
- s2rows.py: when `cfg.servo == "draw"`, the equivalence check passes `episode_seed=<panel seed>`, so the env draws `training_draw(seed)` instead of refusing. The result gains `servo={option, episode_seed, setting}`. Runs without the option are unchanged.
  - Test: `test_the_equivalence_check_gives_a_servo_draw_run_an_episode_seed`. Mutant (old s2rows.py): `1 failed, 22 deselected in 11.78s`.
- The refusal test now uses `match="SERVO_RULE"`.
  - Mutant removing the env's early refusal (the S2 check's e11, which all S2 tests let through): `1 failed, 48 deselected in 5.47s`.
- New string `r5env.SERVO_MID_START_RULE`. No existing string is edited: eight rule texts are pinned by their sha256 as S2 left them. It is printed in describe()'s servo block when the option is on. A test checks that the copies it names match what `servo.Layer` moves.
  - Mutants: old r5env.py `2 failed, 47 deselected in 3.04s`; an edited existing string `1 failed, 48 deselected in 0.77s`.
- Design (SERVO_MODEL_DESIGN.md):
  - Section 5.1's Training env row is amended to the split: copies at entry, model gains at the first physics step.
  - 25 `r5env.py:NNN` references are refreshed to the file at S3; I checked each target line.
  - Left as they were, being records of an earlier revision: the check table's "431 and 504-505 are 468 and 541-542 there" and section 10's "Minor 9" (`r5env.py:475-476`). The row says so.

**For the lead**
- a) Equivalence check under servo="draw".
  - As ruled: the episode seed is the panel seed. The row the bar runs is not served while the env is, so the check cannot come out identical (the test shows `identical` False).
  - Alternatives: give an unserved row's env `options.servo="today"` (the T3 identity: like with like), or compare against SV5 at the env's setting once it exists.
  - I recommend "today" until SV5 exists. That is a one-line change outside this package's remit.
- b) The new rule sentence is also printed in the env's servo description: one code line beyond adding a string. The alternative is to leave it as a constant only.
- c) The runner refuses look sets. That is rule X1, and the look sets' row rule (`lookbar.LOOK_SET_ROWS`) names no served row. Gate 1 runs on the chosen look's walk, so someone must rule how served rows join the look-set rule and the G9-ref requirement. That needs lookbar.py, which is not an S3 file.
- d) A stand-in row's D record goes to the second block when its headline reading passes. The alternative is the bar's form, where rows that give no verdict are never replicated.
- e) There is no budget ledger. Options: a cap via `bar.Budget` from S4's measured rates, or none until then.
- f) N and the reported settings are read by no rule here (condition 1's N belongs on policy rows at 0 to 10 ms on both floors). D is judged on reading 2; reading 3 is recorded.
- g) The Files paragraph of the design still says all code lines are 82ba1ad's. I put the refresh note in the Training env row instead, which is inside my scope.
- h) servo.py and r5env.py changed again, so `train2.sources()` moves. A run started before S3 resumes only with --allow-source-change, and a G8a result made earlier goes stale (none is stored).

**3. Outputs (merged tree, head 3afc8204)**
- `tests/test_stage2_servo.py`: `63 passed in 3.26s`
- `tests/test_stage2_g8s.py`: `27 passed in 19.84s`
- `tests/test_stage2_gate1.py`: `10 passed in 16.90s`
- `tests/test_stage2_r5env.py`: `46 passed, 3 skipped in 84.31s (0:01:24)`. The skips were already there: two -X tests and one slow test.
- `tests/test_stage2_s2rows.py`: `18 passed, 5 skipped in 106.46s (0:01:46)`. The skips were already there: slow and -X.
- g8a tests + `train2::test_sources` + r2lfit's import test: `10 passed in 15.99s`
- regress check: `stage2/regress/regress.json: current (full tier, 256 of 256 episode runs matched, 248 distinct committed episodes, the four source hashes are today's)`, exit 0.
- Hashes: `45719d030359ec7d5d3914af3814d837fdc05d77652ba45c64792b511027528d c0027992de21067525cc4489b74acb157feb89f3a4c7618580addc2b341be1c1`
- ruff: `All checks passed!`
- Dry run: `gate1 dry-run --set V0 --rows SV0,SV2/42 --settings @named,D --delays 4,10`, with `--root` set to the main checkout's read-only sets. It wrote nothing. Full output is in /tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/servo_s3/dryrun_V0.md. Excerpt:
  - `Rates: ... R2 0.1574, R0 0.0954 wall s per simulated s.  The layer's own cost is not in them: the last column adds the design's estimate (x 1.1, ...)`
  - `| seeds3042 | `SV0/F+@4ms` | corner | `walk/v0.3` | 20 | 21.15 | 0.0112 | 0.0123 | to run |`
  - `| seeds3042 | `SV2/42/D@10ms` | whole | `walk/v0.3` | 20 | 21.15 | 0.0185 | 0.0203 | to run |`
  - `| seeds5042 | `SV0/D@4ms` | whole | `walk/v0.3` | 20 | 21.15 | 0.0112 | 0.0123 | if selected: runs only if the headline record's reading passes (P41) |`
  - `Planned: 64 records, 1280 episodes, 7.519 simulated hours, 0.950 core-hours (1.046 with the layer's estimate).`
  - `Second block, only for a drawn-panel record whose headline reading passes (P41): up to 4 records, 80 episodes, 0.059 core-hours (0.065 with the layer's estimate).`
- Not done: the full test suite, an independent review, any S4 measurement (layer cost, throughput), any whole-length G8s run beyond the tests, and anything closed-loop on a look set.

**4. Diff and status**
`git diff claude/mesozoic-trex-stage2-refine-4jffre --stat`:
```
 hardware/trex_form_study/designlab/stage2/g8s.py   | 117 ++--
 hardware/trex_form_study/designlab/stage2/gate1.py | 760 +++++++++++++++++++++
 hardware/trex_form_study/designlab/stage2/r5env.py |  14 +-
 .../trex_form_study/designlab/stage2/s2rows.py     |  17 +-
 hardware/trex_form_study/designlab/stage2/servo.py | 261 ++++++-
 .../design/servo_model/SERVO_MODEL_DESIGN.md       |  32 +-
 hardware/trex_form_study/tests/test_stage2_g8s.py  |  30 +-
 .../trex_form_study/tests/test_stage2_gate1.py     | 366 ++++++++++
 .../trex_form_study/tests/test_stage2_r5env.py     |  58 +-
 .../trex_form_study/tests/test_stage2_s2rows.py    |  26 +-
 .../trex_form_study/tests/test_stage2_servo.py     | 133 ++++
 11 files changed, 1703 insertions(+), 111 deletions(-)
```
- `git status --porcelain`: empty, in the worktree and in the main checkout.
- I removed the caches I had created (__pycache__ and .ruff_cache), the pytest temp directories and the exported copies of r5env.py.
- `git check-ignore -v` on gate1.py and test_stage2_gate1.py exits 1: not ignored. The runner wrote nothing inside the repository.
- Kept in scratch: the dry-run output and the mutant driver with its results (servo_s3/mut/run_mut.py and results2.txt).

**5. Commits, oldest first** (none pushed)
- 9d0a982b servo.Served, names, factories, readings
- aa407313 G8s against the served SV0 row
- 06956fe9 gate 1's runner (gate1.py)
- 2500304d s2rows episode seed under servo="draw"
- 25f8243e mid-start rule sentence; the refusal test now matches SERVO_RULE
- 7d1a35c7 design: 5.1's Training env row; r5env.py line references
- 3afc8204 merge of the working branch (cab73e8b)

Branch head: 3afc8204 (`git rev-parse` was not run after the merge).
