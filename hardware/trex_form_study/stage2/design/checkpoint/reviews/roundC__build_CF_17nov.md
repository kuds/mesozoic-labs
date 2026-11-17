# build:CF (17 Nov): the agent's final report

Working paper. Nothing here is a rule.

Report: work package CF (R2L refit tool). Worktree /home/user/mesozoic-wt/wpc-cf, branch wt/wpc-cf. All work is simulation only. No record was kept and nothing was pushed.

1. Merge of the working branch
No merge was needed. The worktree was created at the tip of claude/mesozoic-trex-stage2-refine-4jffre (fccebe42), that tip has not moved, and it is an ancestor of HEAD. There were no conflicts.

2. What I built (all three files are new, and no other tracked file changed)
- `stage2/tb/r2l/index.json` holds exactly `{"schema": "s2-r2l-index-1", "fits": []}`. Tested in test_constants_match_the_rules_and_the_modules.
- `designlab/stage2/r2lfit.py` (1470 lines). It follows the design's 2.2 API, with two keyword-only extras listed under "For the lead".
  - Module skeleton: it starts with `tbenv.assign_gl("disable")`, sets the thread variables before numpy, and `main()` runs from the importable copy.
  - `rotation(seed, generations)` (rule Q2): episode seed `tune.training_order(seed)[g]`, delay k = (g+j) mod 6, turn side by B mod 2, nudge direction by B mod 4, start and noise seeds `tune.derived_seed(ep, j, 0|1)`. `fit` writes it to `rotation.json` with a microsecond `written_at`, before the session's `started_at`. A resume refuses a table that differs. `cma.json` records `rotation_met` (the delays each kind and direction did or did not meet). Tests: test_rotation_follows_rule_q2, test_rotation_every_kind_and_direction_meets_every_delay, test_fit_writes_the_rotation_first_and_gives_every_member_the_same_episodes, test_a_resume_refuses_another_rotation_table.
  - `Evaluator` and `_evaluator` (the `tune.Pool` factory): builds the 9 bar conditions with `bar.BarContext` on the frozen set (the freeze must cover every file). `run_slot` mirrors `panels.TrackerRow.episode` line for line in R2 mode, inside `speedfit.servo_delay`; only the start and noise seeds differ. Judged by `bar.judge`. Workers refuse other sources. The spawn path with 2 workers was checked on V0: build only, no episodes, and a wrong sources hash raised `PoolError`.
  - Objective `stage2-r2l-v1` (`episode_terms`, `scored`): `tune.score` over 5 kinds. An evaluation that fails scores the worst case, J = 11. Test: the three objective tests.
  - `fit`: `tune.CMA` with popsize 12, sigma0 0.1, starting from `speedfit.start_record`, seed 42. Each generation is checkpointed to `cma.json` and `evals.jsonl`. `ledger.json` records CPU seconds. Rule Q5's block rule decides each next block (block 0 always runs), with a cap of 30 generations. A fit on a look set needs a passed `stage2/tb/gates/<set>/<body>/g9_ref.json`. Any fit outside rule Q1's (set, body) pairs or settings is marked "development". Tests: test_fit_cut_by_its_budget_ends_at_a_whole_block (ends at 12, 6 and 6 generations; the excess of a started block is recorded), test_a_killed_fit_resumes_at_its_next_generation_with_the_same_state, test_a_look_set_waits_for_g9_ref, test_a_development_fit_is_not_selected_or_published.
  - `select` (rules Q4 and Q5 as settled): the decisive cell runs first and in full, with its overrun recorded against the 1.2 reserve and against the share. Then the reported cells run at k = 0, 2, 4, 1, 3, the selected candidate before the start. A cell starts only if the share still holds its estimate (decisive mean seconds per episode x 25 x 1.25). Otherwise it reads `{"status": "not run: budget"}` with its estimate, the share left and started false. A cell that pushes spending past the share sets `overrun_reported`, and no further cell starts. Every validation evaluation is logged and reused on a rerun. Tests: test_the_decisive_cell_runs_first_and_whole_past_its_reserve, test_reported_cells_run_in_order_while_the_share_holds_their_estimate, test_a_reported_cell_past_the_share_stops_the_others, test_a_rerun_of_the_selection_reuses_every_kept_evaluation.
  - `publish`: copies the records with `nb.collect` (each verified by sha256), then appends `{set, body, seed, r2lg5: {path, sha256}}` to the index. Publishing again changes nothing, and an indexed fit is never replaced.
  - `r2l_row` and `register` (rule Q6): `register` runs at import for every set in the index. `r2l_row` returns `lookbar.LookRow(R2LRow)`, a TrackerRow in R2 mode with the indexed gains. The gains file's sha256 is checked, also again in workers through `from_spec`. `describe()` carries `gains_info` and `gains_file` (path and sha256). Its sources are `r2lfit.py`, `lookbar.py` and `lookread.py`. The row refuses a condition of another set. Tests: test_publish_then_the_row_resolves_and_refuses (it resolves, refuses another set, a stock condition, a missing entry and a changed gains file, and resolves through `bar.rows` at @10ms), test_importing_r2lfit_changes_no_file_or_table (speedfit.py's sha256, VERDICT_ROWS, ATTRIBUTED, ROW_FACTORIES and both key hashes are the same before and after the import).
- `tests/test_stage2_r2lfit.py` (567 lines): 21 fast tests on a stub evaluator, plus 1 slow test (STAGE2_SLOW=1) checking that slot 0 matches the bar's R2 row bit for bit on V0.

For the lead (my readings of the rules, with alternatives; I built the first option in each case and recommend it)
- a. Q3 item terms. I read "X over Y" as min(1, X/Y). This gives line/0.25 m, |heading error|/20°, stop strides/1.5 and recovery/3 s. "Shortfall" (along) and "beyond two" (touchdowns) are hinges, as the text says.
  - Alternative B: hinge at the stated number with that number as scale.
  - Alternative C: add-on design C's hinge at the bar's own limits (heading at 10°).
  - I chose A because the text marks the hinges explicitly, and 20° is twice the bar's 10° tolerance, so it only reads as a scale.
- b. Tracking on turn, stop and nudge: DCM only. `tune.measure`'s speed and heading terms belong to a straight walk and would score a commanded turn or stop as error.
- c. Limits:
  - Stop and nudge: only the episode limits (drive_sustained over the episode, drive_speed, joint_range), as the bar judges them.
  - Turn: no straight-walk speed limit; joint range as the bar reads it (decision 14 margins over the window); no leg-interference term, because Q3 names none.
  - walk/nominal: the "along" target is the bar's own (10 s x its target speed), not 4.0 m.
- d. Seeds:
  - A nudge's onset uses the bar's draw for the episode seed. Q2 gives per-slot seeds for start and noise only.
  - Validation episodes use the same per-slot seeds as the fit.
- e. Budget share = `budget_core_h` + 1.2 + `reported_core_h`, which is 4.0 with the defaults. This replaces the brief's "while 0.4 lasts" with the settled 16 Nov rule.
  - Each evaluation's CPU is counted once.
  - CPU lost to a killed run is not counted.
- f. API extras:
  - `fit(..., *, popsize=12)`: the brief's smoke needs population 4.
  - `select(..., *, log=print)`.
  - A development fit is refused by `select` and `publish`.
- g. G9-ref record format: it is not defined yet (gates2.py does not exist). The gate accepts only `passed: true` or a status or outcome of "passed", and refuses anything else. The gates package (CG) must write one of these, or this check must follow its key.
- h. Not built: Q5's panel-record caps (`stage2/tb/bar/<run>/DRY_RUN.json` and `budget.json`). They belong to whoever runs the R2L rows' bar records. The design's 2.2 gives r2lfit no API for them.
- i. Two things outside my files:
  - As built, `bar._check_rule` refuses non-R0 rows on V6noF5, so R2L-V6noF5 panel records are refused there (lookbar's docstring already notes this).
  - `train2.SOURCE_FILES` lists `r2lfit.py`: it was recorded as absent until now, so the trainer's recorded source hash set changes when this lands.

3. Outputs I saw
- `nice -n 15 $PY -m pytest -p no:cacheprovider tests/test_stage2_r2lfit.py` (through lowprio): `21 passed, 1 skipped in 32.46s`, exit 0.
- With STAGE2_SLOW=1, `-k slot_0`: `1 passed, 21 deselected in 3.62s`. Both episodes fell at 4.6 s (the start's gains at 10 ms on walk/v0.4, seed 2000), so the comparison covers 4.6 s of trace.
- ruff: `All checks passed!`
- `r2lfit rotation --body 75-M-H-10 --set V6noF5 --seed 42 --out <scratch>/wave6/r2l`: exit 0. It printed the 30-generation table; every (kind, direction) met 0,2,4,6,8,10 ms. It wrote nothing.
- Hash pair after the build: `45719d030359ec7d5d3914af3814d837fdc05d77652ba45c64792b511027528d c0027992de21067525cc4489b74acb157feb89f3a4c7618580addc2b341be1c1` (unchanged).

Development episodes (all R2 mode on V0, body 75-M-H-10, none of them a record)
- Smoke fit: `bash tools/lowprio.sh $PY -m designlab.stage2.r2lfit fit --body 75-M-H-10 --set V0 --seed 42 --out /tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/wave6/r2l --max-generations 2 --popsize 4 --workers 1`
  - 8 evaluations, 40 episodes, exit 0, wall time 2m4s.
  - Printed: `done at generation 2 (development); 8 evaluations, 120.5 CPU s (15.06 per evaluation, 3.01 per episode)`. Per generation: 65.9 and 54.6 CPU s.
  - No evaluation errors.
  - 13 of 40 episodes fell: nudge/forward in all 8 evaluations, walk/nominal 4, turn 1.
  - 3.01 CPU s per episode is below Q5's 4.4-11 because the falls end episodes early. V6noF5's planned episodes are longer.
  - Output is in `/tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/wave6/r2l/V0/75-M-H-10/`.
- Slow test: 2 episodes (walk/v0.4, k=5, seed 2000).
- One more check episode (the same condition), to read the fall time.
- Total: 43 episodes.

4. Diff and status
`git diff claude/mesozoic-trex-stage2-refine-4jffre --stat`:
```
 .../trex_form_study/designlab/stage2/r2lfit.py     | 1470 ++++++++++++++++++++
 hardware/trex_form_study/stage2/tb/r2l/index.json  |    1 +
 .../trex_form_study/tests/test_stage2_r2lfit.py    |  567 ++++++++
 3 files changed, 2038 insertions(+)
```
`git status --porcelain`: empty.

`git check-ignore -v`: everything under `stage2/tb/work/r2l/...` (including `.lock`) is ignored by `stage2/tb/.gitignore:1:work/`. The published `stage2/tb/r2l/...` paths and `index.json` are not ignored; they are meant to be committed.

5. Commits, oldest first
- fc1a4b79 Add R2L's empty row index (work package CF)
- f8374e6a Add r2lfit: R2L's fit, selection, records and rows (work package CF)
- 90653044 Test r2lfit on a stub evaluator (work package CF)

Branch head: 90653044ec520ee113c952521b8b57ac1699387d (wt/wpc-cf).
