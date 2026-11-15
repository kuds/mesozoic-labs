# fix:B (roundA): the agent's final report

WP B after review: every code finding is fixed on wt/wp-b. No V0, F25F9 or F25F9cl set exists yet, so the three V0 acceptance commands were not run; the smoke ran on scratch set D0 instead. All numbers are smoke.

**Findings (J = judge, P = provenance)**
- J1 (blocker): a call's episodes finish before any attribution; the chain runs T3.1/T4.1 with `--attribute none`, then one attribution step per set before T3.2/T4.2; the order test asserts it.
- J2: the turn's joint range and leg gap are read over the window; `drive_speed_span: "episode"`; lead-in excursion test added.
- B-R3 (J5, J15): amended C3 built in `readings.py`, with tests per clause.
- J3, J4, J6–J14: fixed as proposed, except J7's optional joint-jump check (fixture T0 cannot meet it).
- J16: the earlier scratch-set episodes, and my 21 on D0, are in `dev_episodes.json` under `scratch_sets`; uncertain counts say so.
- P1–P13: fixed as written, with tests.
- P14: done in `bar.py` (`check_manifest`), not `setio.py`.

**Acceptance (commit 749b3ad unless noted)**
- `pytest tests/test_stage2_bar.py`: exit 0, "47 passed, 2 skipped in 169.65s".
- `pytest tests/test_stage2_readings.py`: exit 0, "43 passed in 1.95s".
- `bar run --set S1 … --procs 2`: exit 0; four episode records and trace hashes equal Stage 1's.
- `bar reread` (60-N-T-10): exit 0, "40 of 40 stock traces equal"; 20 list the clearance readings, 20 fell.
- Slow tests on D0: exit 0, "2 passed".
- One commit earlier (differs in `__all__` only), on D0: dry-run exit 0; 16 records, every reading present, exit 0; refcheck exit 1 because D0's stand-in walk covers 3.68 m.
- `git status --porcelain`: empty.

**B-R3 evidence** (18 of 18 traces equal; judged, then as first written, in H)

| Row | Walk | Judged | First written |
|---|---|---|---|
| R0 | nominal | 0.0477, 0.0471, 0.0474 | 0.0332, 0.0330, 0.0335 |
| R0 | brisk | 0.0517, 0.0522, 0.0522 | 0.0359, 0.0356, 0.0377 |
| R2/42 | nominal | 0.0438, 0.0452, 0.0413 | 0.0436, 0.0428, 0.0445 |
| R2/42 | brisk | 0.0492, 0.0475, 0.0479 | 0.0433, 0.0436, 0.0455 |
| R2/42 | fr05 | 0.0496, 0.0477, 0.0485 | 0.0407, 0.0418, 0.0417 |

R0 fr05 fails every reading, Stage 1's included. Two R2/42 values differ from R's fixer's (0.0438 against 0.0435, 0.0492 against 0.0483). Full table: `/tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/wpb_fix/C3_reread_table.txt`.

**Commit:** `749b3ad Stage 2 round A (WP B): the orphan-worker test holds without luck; the fixer's scratch-set smoke listed; names exported`

**Public names changed**
- `readings.toe_clearance_own_swing(..., planned_s=None)` with new result keys.
- `bar run` exit 4; an unkept trace has `kept: false` and no `path`.
- New record fields: `set_info`, `change_after_result`, `stand_in_walks`, `supersedes.registered/replaces`, a verdict cell's `replication`.
- The trace gains `planned_s100`; `swings` has 7 columns.

**For the lead**
1. C3's Check paths: `measures.bar.<key>` and `measures.bar.toe.*` (walk, turn); `measures.bar.toe_reported.*` (nudge, stop).
2. My choices in the amended reading: overlap is counted in shared update rows (ties to the earlier swing); a swing whose interval passes the window's end counts as cut, and so does its reference swing.
3. WP S must write: `manifest_sha256 = s2io.sha256_json(manifest)`; `freezes[i].change_after_result`; walk entries' `crouch_frac` and `speed_m_s` matching set.json and the name; `command_phase` matching 2, 2.5, 2.5 strides.
4. Beyond P3's text: the chain writes `done/STOPPED` and exits 4 when steps stay incomplete, and the watchdog line honours it.
5. The runner now refuses R2 rows on scratch or fixture sets; tests set `bar.SYNTHETIC_EPISODES`.
6. I merged the working branch into wt/wp-b first, as B-R3 asks.
