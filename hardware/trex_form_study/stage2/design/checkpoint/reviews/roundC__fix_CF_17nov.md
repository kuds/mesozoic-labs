# fix:CF (17 Nov): the agent's final report

Working paper. Nothing here is a rule.

Fix report for work package CF (R2L refit tool). Worktree /home/user/mesozoic-wt/wpc-cf, branch wt/wpc-cf. Simulation only. Nothing pushed, no record kept, no history rewritten.

MERGES
The working branch moved twice while I worked, so I merged it twice. Neither merge touched my three files and neither had conflicts.
- 848fa9bc merged 3a42df8a: CG's gates2 and the CN, CT and CW follow-ups.
- 19dc9664 merged 6c62f30e: rulings 13 to 20 and reports, documents only.
The branch tip 6c62f30e is an ancestor of HEAD.

PER FINDING
1. Fixed (gate vs CG's `pass_`). `gate()` now calls `gates2.require_g9_ref(body, set, path=<GATES>/<set>/<body>/g9_ref.json, root=SET_ROOT)` and records its citation, plus `modified_at` and `checked_by`.
   - A gates2 refusal (`GateError`) becomes an R2LError.
   - If gates2 cannot be imported, no fit on a look set starts.
   - A citation that is not a dict with path and sha256 is refused. This includes the None that gates2 returns for a set outside its own LOOK_SETS.
   - GATE_RULE was rewritten to match. The test fixture now writes records in CG's form, through a stub of gates2.
   - Since CG is now merged, `test_the_gate_calls_the_real_gates_package` runs against the real package. A record written by `gates2.finish` on a synthetic frozen set.json passes. Records that did not pass, were made on another freeze, or carry only the old `passed: true` are refused. The LOOK_SETS of gates2 and r2lfit are checked equal.
   - Against CG's real gates2, the reviewed gate (90653044) fails this test. The old fixture form is refused by gates2.
   - The "which way round" question no longer arises: the gate now defers to whichever gates2 is on the branch.
2. Fixed.
   - (a) `role_of(..., seed=42)`: any other seed makes a development fit, which select refuses.
   - (b) `selection_role` writes `role` into r2lg5.json. A `reported_core_h` other than 0.4 makes a development selection: publish refuses it and `gains_of` (rows) refuses it. ROLE_RULE was updated.
3. Fixed. `fit --resume` on a fit whose status is done or ended_by_budget now returns its record and writes nothing. This check comes after the settings and rotation checks. RESUME_RULE was updated.
4. Fixed. `publish` now copies only start.json, r2lg5.json, r2lg5_evals.jsonl and r2lfit/<seed>/ (`publish_files`).
   - These are staged under <set>/<body>/.publish/<seed>/ (ignored by work/) and the staging is removed afterwards.
   - Then `nb.collect` copies them, as design D22 requires.
   - publish holds the set/body lock throughout. A locked directory is refused, which is tested.
   - New rule text PUBLISH_RULE.
5. Fixed: new tests for every listed gap.
   - Per-slot start and noise seeds in run_slot, with control.run's arguments captured and no physics.
   - Resume on other settings.
   - select on a running fit and on other sources.
   - gains_of's set, body and seed check.
   - publish of a development fit and of a changed cma.json.
   - The fewer-falls-first ranking (FallStub).
   - The box repair term in the fitness.

Check that the tests fail without the fixes: script /tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/wave6/cf_fix/revert_check.py, output /tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/wave6/cf_fix/revert_check.out.
- It reverted each fix alone, or applied each of the reviewer's mutants, in r2lfit.py.
- Every one of the 16 variants failed its tests with exit 1.
- The file was restored afterwards ("restored: True").

FOR THE LEAD
- A (new, CG's run_trackb.sh, not my file). Step V10 passes `fit ... --resume` every time. On the first run r2lfit refuses with "nothing to resume", as speedfit and tune also do, so the chain stops there.
  - Options: the chain passes --resume only when r2lfit/42/cma.json exists (recommended; it matches resume_flag), or r2lfit's --resume starts a fresh fit when there is none.
  - Also, V10-collect runs `nb collect --src $W/r2l --dst $TB/r2l`. That copies all of $W/r2l, which repeats publish and undoes finding 4 at chain level. I recommend dropping that step: publish is R2L's collect.
- B (ruling 16). r2lfit's fit and select never go through `bar.evaluate`/`_check_rule`. Only the R2L rows' bar records do. lookbar's look-set context does not exist yet. When it lands, check whether CF needs any change; I expect none beyond R2L being admitted there.
- C. The rule texts are part of a fit's `settings`, so a fit started with the reviewed code cannot be resumed with this code. None exists except scratch smoke runs.
- D. The review's lead items stand unchanged:
  - a. Q3's terms. The builder's reading A is built. The reviewer notes Stage 1 grades the same stop quantity with `hinge(x, 1.5, 1.5)` (tune.py:658), which supports reading B for the stop time. I recommend deciding this before the first fit; A stays until then.
  - b. Q5's panel-record caps: I recommend handling them as an operational step.
  - d. Landing r2lfit changes train2's recorded source set.
  - e. The order of the three fits is operational.

TESTS (exact lines)
- `bash tools/lowprio.sh $PY -m pytest -p no:cacheprovider tests/test_stage2_r2lfit.py -rs` at head 19dc9664: `34 passed, 1 skipped in 66.44s (0:01:06)`, exit 0. The skip is the slow test.
- `STAGE2_SLOW=1 ... -k slot_0` at 8f8ddc3f: `1 passed, 34 deselected in 3.66s`, exit 0. The later merge added only .md files, so the code under test is the same.
- `ruff check --no-cache designlab/stage2/r2lfit.py tests/test_stage2_r2lfit.py`: `All checks passed!`
- Hash pair at the end: `45719d030359ec7d5d3914af3814d837fdc05d77652ba45c64792b511027528d c0027992de21067525cc4489b74acb157feb89f3a4c7618580addc2b341be1c1`. This is the registered pair.
- `git check-ignore -v`: the staging path stage2/tb/work/r2l/<set>/<body>/.publish/42/... and .lock are ignored by stage2/tb/.gitignore:1:work/. The published COLLECTED.json is not ignored; it is meant to be committed.

DEVELOPMENT EPISODES
- 4 in total, from the slow test run twice: `STAGE2_SLOW=1 bash tools/lowprio.sh $PY -m pytest -p no:cacheprovider tests/test_stage2_r2lfit.py -k slot_0`. Each run is 2 episodes in R2 mode on V0, body 75-M-H-10, walk/v0.4, k=5, seed 2000.
- No other physics ran: the revert check and the scratch check against CG's gates2 ran none.
- I did not re-run the 40-episode smoke fit. The evaluator did not change.

DIFF AND STATUS
`git diff claude/mesozoic-trex-stage2-refine-4jffre --stat`:
 .../trex_form_study/designlab/stage2/r2lfit.py     | 1576 ++++++++++++++++++++
 hardware/trex_form_study/stage2/tb/r2l/index.json  |    1 +
 .../trex_form_study/tests/test_stage2_r2lfit.py    |  912 +++++++++++
 3 files changed, 2489 insertions(+)
`git status --porcelain`: empty.

COMMITS, OLDEST FIRST
Each ends with exactly the two attribution lines and names no model anywhere else.
- c399669f r2lfit: read G9-ref through the gates package (finding 1)
- ac190406 r2lfit: rules' seed and share, a finished fit's resume, the fit's own records (findings 2-4)
- 1de545bb Test r2lfit's remaining refusals and the episode's per-slot seeds (finding 5)
- 848fa9bc Merge the working branch (CG, CN, CT, CW)
- 8f8ddc3f Test r2lfit's gate against the merged gates package (finding 1)
- 19dc9664 Merge the working branch (rulings 13 to 20, reports)

Branch head: 19dc9664dfdc425f2736dda5fa54b72637ef57bd (wt/wpc-cf).
