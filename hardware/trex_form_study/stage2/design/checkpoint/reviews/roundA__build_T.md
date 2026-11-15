# build:T (roundA): the agent's final report

WP T is built and committed on `wt/wp-t` (not pushed, not merged); all three acceptance commands exit 0.

**Commits** (after fast-forwarding to `wt/wp-b` 749b3ad): f782916, a575689, ee67e1c, fe78b1c, 069347d, 8197223, bf82370, 79e3e92, d048572, 0c7908d, 0d5abe9, de512f9 (head). Only two files differ from `wt/wp-b`:
- `/home/user/mesozoic-wt/wp-t/hardware/trex_form_study/designlab/stage2/barreport.py`
- `/home/user/mesozoic-wt/wp-t/hardware/trex_form_study/tests/test_stage2_barreport.py`

**Acceptance**
- `$PY -m pytest -p no:cacheprovider tests/test_stage2_barreport.py`: exit 0, `43 passed in 9.54s`.
- `$PY -m designlab.stage2.barreport --out logs/bar-smoke-fix --dest logs/barreport-smoke`: exit 0.
- `git status --porcelain`: exit 0, empty.

**Smoke numbers**
- The lead named no smoke directory. I used WP B's `logs/bar-smoke-fix` (scratch set D0, `R0@10ms`), copied with D0's JSON documents into my worktree's ignored `logs/`.
- Records found: 16 (smoke), one episode each, so all 16 are listed as unjudged.
- All 16 are flagged "other sources (bar)", because `bar.py` changed after that smoke.
- 0 inputs outside the latest freeze; 0 counts that fail to recompute.
- Every table renders, with "not run" for the registered order's cells; output is `/home/user/mesozoic-wt/wp-t/hardware/trex_form_study/logs/barreport-smoke/BAR_TABLES.md`.
- `BAR_TABLES.md` regenerates from `bar_tables.json` byte for byte.

**Other checks**
- A test confirms `planned()` equals the cells of `run_75cm.sh list`: 196 for task 3, 396 for task 4, plus the omega control.
- By hand, not committed: I ran the report on records, a ledger with cut units, a replication and a re-run written by `bar.evaluate` itself with synthetic episodes (nothing simulated). Replication and verdict words matched the runner's.

**Deviations**
- `load` returns `[set][body][block][row][condition]` plus one reserved key `"_run"` for the ledger, hashes, freezes and controls.
- `load` and `write` take an optional `study=`; the CLI adds `--body`.
- "Cut by the pre-registered order" is printed when the record's path is under a cut unit, or when its task cut a unit and the cell is in no started unit. A tier absent from a ledger with no cut reads "not run", so a run in progress is not reported as cut.
- Core-hours are set against each task's cap in `budget.json` (16 and 12).
- Counts and verdicts come from the registered (first finished) record; a re-run stands beside it, labelled.
- Checks beyond the brief:
  - counts recomputed with `bar.counts`;
  - check episodes held to rule R4;
  - `seeds5042` records outside the replication rule;
  - units started after the first cut;
  - conditions each set holds no reference for, read with `bar.BarPlanner.missing`, so an absent cell still prints its reason and numbers.
- JPS table:
  - line 99 also shows `walk/v0.4` at 6, 8 and 10 ms;
  - the stop and nudge rows list the 2 ms cell and show 0 and 4 ms beside it;
  - "JPS's were not causal" is worded as "the causal construction is this round's; JPS ran no 90 degree turn".
- An eighth table lists records that no other table cites.
- Stage 1 arrays were copied into the worktree for the hand check only; they are git-ignored and the tests need none.

**Not done**
- Reading 1 is not tabulated for sets with look bands; a record that holds it is named, not counted.
- No real `V0`, `F25F9` or `F25F9cl` record exists yet, so those tables are exercised only on synthetic records and the D0 smoke.
