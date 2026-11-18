# check:CF (17 Nov): the agent's final report

Working paper. Nothing here is a rule.

Independent check of work package CF (R2L refit tool). Worktree /home/user/mesozoic-wt/wpc-cf, branch wt/wpc-cf at 90653044. Simulation only. I changed no tracked file and committed nothing. My scripts are under /tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/wave6/check_cf/.

VERDICT: merge as it stands. It is a fast-forward to 90653044ec520ee113c952521b8b57ac1699387d. One fix is owed before the first fit on V6noF5: finding 1, which must be made against the gates package (CG). Findings 2 to 4 can wait for the next change to r2lfit.py. No check gave a wrong result. Every refusal I tried failed closed, except the two role holes in finding 2, and both of those need arguments outside the rules.

FINDINGS (most serious first)

1. Major (it blocks the fits, but fails closed). A defect at the integration with CG. File: designlab/stage2/r2lfit.py:697-716 (the verdict test is at 708-709).
   - The gate accepts g9_ref.json only when it holds `passed: true`, or a `status` or `outcome` of "passed".
   - CG's unmerged gates2 (worktree wpc-cg at b13bcd4c) writes the verdict as `pass_` (gates2.py:925, `pass_=passes`). It has no `passed` key, and its own `require_g9_ref` (gates2.py:374-403) reads `pass_`.
   - So once CG lands, every fit on V6noF5 will be refused even after G9-ref has passed. Rule Q1's two main fits are blocked.
   - r2lfit's gate is also weaker than CG's. It does not check the record's schema, the record's own sha256, the set and body, or that the record was made on the set's latest freeze. CG's `require_g9_ref` checks all four.
   - The test fixture (tests/test_stage2_r2lfit.py:108-109) writes `passed=True`, so the test agrees with r2lfit and not with CG.
   - Fix: when CG is merged, have `gate()` call `gates2.require_g9_ref(body, set_name)` and record its citation. If the order is the other way round, until then accept only `pass_ is True` with matching set and body and the latest freeze, and change the fixture to match. Which way round is for the lead.

2. Minor. A defect: a fit or selection that does not follow the rules can be published as a record.
   - (a) `role_of` (r2lfit.py:682-694) takes no seed. I checked that `role_of('V6noF5','75-M-H-10')` gives "fit" whatever the seed. Rules Q1 and Q2 fix seed 42, so a seed-43 fit on a pair Q1 names can be selected, published and indexed as R2L-V6noF5/43.
   - (b) `select(..., reported_core_h=x)` with x other than 0.4 still gives a judged r2lg5.json, and `publish` (1284-1287) accepts it. The share is then not Q5's 4.0, although the record does show it (`share_core_h`).
   - Fix: role_of requires seed == 42; select marks, or publish refuses, a selection whose `reported_core_h` is not 0.4.

3. Low. A defect: resuming a finished fit changes its record.
   - `fit --resume` on a "done" fit (r2lfit.py:874-916 and 931-933) appends a session and rewrites cma.json.
   - I saw this on my scratch run: the cma.json sha256 went from 8a3660c624e1 to 8bdb651f9998. After that, `publish` refused with "cma.json changed after the selection was made: select again". So it fails closed, but a finished fit's record changes.
   - Fix: refuse, or do nothing, when the status is done or ended_by_budget.

4. Low. A defect: the whole set/body directory is published.
   - `publish` copies everything under <out>/<set>/<body>/ (nb.collect_files, r2lfit.py:1298-1302). Any other seed's r2lfit/<seed>/ in that directory, for example a development fit, goes into the committed place too.
   - Fix: limit the copy to start.json, r2lg5*.json(l) and r2lfit/<seed>/.

5. Low. Test gaps. Each mutant below passed every fast test.
   - The per-slot start and noise seeds in `run_slot` (516-517). Mutant: slot 0's seeds for every slot. The slow test only compares j = 0, where the two coincide, so rule Q2's "five different starts" is untested at the episode. A fast test could capture control.run's seed arguments with a monkeypatch, with no physics.
   - Resume with other settings is not refused-tested (878).
   - select refusing a running fit (1126), and a fit on other sources (1129).
   - gains_of's check of the document's set, body and seed (1336).
   - publish of a development selection (1284), and of a cma.json changed after selection (1286). Both are guarded twice in the code.
   - The fewer-falls-first order of the selection key (1165). The stub never falls.
   - The box repair penalty in the fitness (950).

FOR THE LEAD (choices that are not the builder's to make)

a. Rule Q3's item terms (builder's points a to d). The builder reads "X over Y" as min(1, X/Y). One piece of evidence the builder did not give: Stage 1's own grading of the same stop quantity is a hinge at the number. tune.py:658 has `stop_t["stopped"] = hinge(forward_stopped_strides, 1.5, 1.5)`, and the start term at line 655 has the same form. That supports the builder's alternative B, at least for "the stop time over 1.5 strides".
   - Other open readings, as the builder listed them: the walk/nominal "along" target is the bar's own (10 s x its target speed), not 4.0 m; turn, stop and nudge track the DCM only; a nudge's onset is the bar's draw; validation episodes use the per-slot seeds.

b. Rule Q5's panel-record caps (`stage2/tb/bar/<run>/` with DRY_RUN.json and budget.json) are not in r2lfit.
   - The brief and the design's 2.2 give r2lfit no API for them. Q5 says the bar's dry-run writes no file, "so whoever makes the dry run writes this one".
   - Choose either an operational step for whoever runs the R2L rows' records (my recommendation), or a small writer added beyond the frozen 2.2 listing.

c. Outside this package: R2L rows will be refused on V6noF5.
   - `bar._check_rule` (bar.py:2195-2205) lets only R0 rows run on a set outside PREREGISTERED_SETS = ("V0","F25F9","F25F9cl") (bar.py:96), with or without --task.
   - So no R2L-V6noF5 panel record can run through the bar runner as built. lookbar.py:62 notes the same. bar.py or lookbar needs a change.

d. Outside this package: landing r2lfit.py changes the trainer's recorded source set. `train2.SOURCE_FILES` (train2.py:141-145) lists r2lfit, which until now was recorded as absent. No committed JSON under stage2/ mentions r2lfit (grep), so no existing record changes. A training session already in flight would see moved sources.

e. Rule Q5's order of the three fits (V6noF5 on 75-M-H-10, then on 60-N-T-10, then F25F9cl) is not enforced in code. It is operational.

CHECKED AND FOUND RIGHT

- Rotation (S2-18, Q2):
  - g // 6 blocks; episode seed `training_order(42)[g]`; k = (g+j) mod 6; side by B mod 2; direction by B mod 4; seeds `derived_seed(ep, j, 0|1)`.
  - Each (kind, direction) meets every delay within the stated number of blocks.
  - rotation.json is written before generation 0, with microsecond timestamps.
  - A resume refuses another table; `rotation_met` names what a cut fit did not meet.
  - Mutants M1, M2, M3, M5, M6, M37 and M49 were caught.
- Selection (Q4):
  - The 10 ms cell (k = 5) runs first and whole. The candidates are the start plus `candidate_generations` (2,4,6 or 10,20,30).
  - Validation sides and directions follow seed parity and seed mod 4. Ranking uses `tune.selection_key`.
  - The reported cells run at k = 0,2,4,1,3, selected candidate before the start.
  - Mutants M9, M10, M12, M13, M15, M16, M17, M46 and M50 were caught.
- Budget (Q5 as settled):
  - Block 0 always runs. The block rule is spent + costliest block <= 2.4 core-h, with a cap of 30 generations. A started block finishes and its excess is recorded (`fit_excess_s`).
  - The share is 4.0 core-h less everything spent. The decisive overrun is recorded both against its 1.2 reserve and against the share.
  - The estimate is mean seconds per episode x 25 x 1.25. A cell is "not run: budget" and records its estimate and the share left. `overrun_reported` stops further cells. ledger.json lists every item in the order run.
  - Mutants M7, M8, M33, M34, M39, M40 and M41 were caught.
- Index and rows (Q6):
  - The index holds exactly `{"schema": "s2-r2l-index-1", "fits": []}`.
  - publish verifies each copy by sha256 and never replaces a published file. Publishing again changes nothing.
  - r2l_row returns LookRow(R2LRow) and checks the sha256 again in from_spec. It refuses another set, a stock condition and a missing entry. `register` works and so does bar.rows at @10ms.
  - Mutants M18, M19, M20, M43, M44, M52 and M54 were caught.
- Mutation total: 54 mutants of r2lfit.py loaded through a pytest plugin, with no file changed. 41 were caught. Of the 13 that passed, three are equivalent or covered elsewhere: M11 (the stopped flag), M24 (charging duplicates), M42 (nb.collect also refuses). The other ten are listed in finding 5.
- Real kill and resume, with no physics episode:
  - Setup: the real tune.Pool and the real Evaluator build on V0, Evaluator.evaluate replaced by a formula, and ("V0", body) added to FITS so the run counts as a fit.
  - Fit: SIGKILL at evaluation 100. A rerun without --resume was refused ("exists"). The second session was killed by SIGKILL just before its 4th cma.json write: evals.jsonl had 144 lines against a 132-evaluation checkpoint. The third session resumed to the end.
  - Select: killed by SIGKILL at evaluation 37 (36 kept), killed again at evaluation 50 of the rerun (85 kept), then run whole.
  - Every stale lock was taken over.
  - Against an uninterrupted run, every compared item was equal: CMA state, history without wall times, block decisions, the bytes of evals.jsonl, r2lg5.json's selection, by_delay, decisive, overrun, overrun_reported and spent_s, and the ledger's order and cells.
  - Publish, publish again, register, `bar.rows("R2L-V0/42@10ms")` (giving Delayed(LookRow(R2LRow)), k = 5) and the spec round trip all held.
- The smoke's claims match its output in scratch:
  - The fit is marked development, ran 2 generations of 4 evaluations, and spent fit_cpu_s 120.48.
  - 13 of 40 episodes fell: nudge in all 8 evaluations, walk/nominal 4, turn 1.
- run_slot mirrors panels.TrackerRow.episode statement for statement. The evaluator calls bar.judge as bar.run_episode does (lg only on a turn). Every bar.judge key the terms read exists in bar.py.
- Only the three files differ from the working branch. The two record-key hashes are the registered pair (45719d03...528d c0027992...c1c1), both before and after my runs. git status is empty in the worktree and in the main checkout. All CF outputs are under ignored work/ or scratch; the published stage2/tb/r2l paths are not ignored, which is intended.

TEST SUMMARY LINES (my runs)

- `bash tools/lowprio.sh $PY -m pytest -p no:cacheprovider tests/test_stage2_r2lfit.py`: exit 0, `21 passed, 1 skipped in 33.01s`.
- `STAGE2_SLOW=1 ... -k slot_0`: exit 0, `1 passed, 21 deselected in 3.37s`.
- `ruff check --no-cache designlab/stage2/r2lfit.py tests/test_stage2_r2lfit.py`: `All checks passed!`.

DEVELOPMENT EPISODES I RAN

The slow test only: 2 episodes in R2 mode on V0, body 75-M-H-10, walk/v0.4, k = 5, seed 2000. The mutation runs and the kill-and-resume runs ran no physics episode.

WHAT TO BRING IN

Fast-forward claude/mesozoic-trex-stage2-refine-4jffre (fccebe42) to 90653044. merge-tree gives tree cb15567a, the same as the branch's own tree, with no conflict. In order, the commits are fc1a4b79, f8374e6a and 90653044. Each ends with exactly the two attribution lines, and none names a model anywhere else. There are no merge commits.

WHAT I DID NOT CHECK

- I did not re-run the real 2-generation smoke fit (40 episodes); I read its output files.
- I did not run anything with workers > 1.
- I did not check numerically that the episodes' CPU seconds are in Q5's 4.4-11 core-second planning range on a real look set (no look set is frozen).
- I did not check CG's merged form, only its worktree at b13bcd4c.
- I ran no other package's tests.
