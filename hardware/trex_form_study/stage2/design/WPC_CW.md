# Work package CW: rows for the bar runner: `lookbar.py`, `s2rows.py`

You are building one part of a robotics simulation study (a robotic T. rex biped; everything is simulation). You
need no other context than this brief and the two texts it points into:
- the design: `/tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/design/ROUND_C_DESIGN.md` (read its header, the decisions and sections named under "Read first" below);
- the rules fixed before the runs: `/tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/design/PREREG_TRACKB_draft.md` (rule ids such as "K3" or "W2" refer to it).
Round A's design, where a round-A API is named: `/tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/design/ROUND_A_DESIGN.md`.

**Read first:** design D15, D17, D18, D23; sections 2.2 (`lookbar`, `s2rows`), 2.3 (the last paragraph: what is relied on from `bar`), 3.5 (all of it: where the reading runs, the sidecar, the companion record, the outcomes), 3.11; rules U4-U6, M1, K9, E5. Code: round A's `designlab/stage2/bar.py` (`Delayed`, `rows`, `BarBuilt`, `BarContext`, `run_episode`, `_judge`, `Unreadable`, `counts`, `evaluate`, `BarWorker`, `_Played`), `designlab/stage1/panels.py` (`Row`, `TrackerRow`, `Policy`, `_Acting`, `register_rows`, `row_from_spec`), `policy_rows.py` (how Stage 1 built a policy row; you do not import it).

## Goal

Everything that lets Track B's controllers run through round A's bar runner without editing it: `LookRow` (a wrapper that gives any row the look reading), the `R0L` row, in-process episodes for the monitor, the companion records that hold reading 1, and the policy rows `R5/<seed>`, `R6/<seed>` and `R5G/<seed>` loaded from a checkpoint's files under the cross-version rule (S2-12).

## Files you own (and no others)

- `designlab/stage2/lookbar.py`, `designlab/stage2/s2rows.py`
- `tests/test_stage2_lookbar.py`, `tests/test_stage2_s2rows.py`

Do not edit any other file: nothing under `designlab/stage1/`; no round-A module (`designlab/stage2/bar.py`,
`readings.py`, `setio.py`, `s2io.py`, `moves.py`, `sets.py`, `setscreen.py`, `barreport.py`, `moves_look.py`); no
other package's file; no committed record; nothing at all inside `/home/user/mesozoic-labs` (the main checkout:
you work in your own worktree). If you believe an edit elsewhere is unavoidable, stop and tell the lead: the design
claims none is needed (its section 2.3).

## Setup

Worktree: the absolute path the lead gives you; work only there.
```
cd <your worktree>/hardware/trex_form_study
PY=/home/user/mesozoic-labs/hardware/trex_form_study/.venv/bin/python
export PYTHONPATH=. MUJOCO_GL=disable OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONHASHSEED=0
$PY tools/restore_inputs.py copy-from /home/user/mesozoic-labs/hardware/trex_form_study --group picks-models picks-arrays
```
Versions: Python 3.13.18, torch 2.15.0+cpu, stable-baselines3 2.9.2, gymnasium 1.4.0, mujoco 3.10.0, numpy 2.5.4.
Needs from other packages: round A's `bar.py` (package B) and packages CE (`r5lib`, `r5core`; `r5env` only inside `equivalence_check`), CP (`s2load`), CK (`lookread`) merged into your branch. Wave 3.

## Machine-sharing rules (4 cores, shared with other builders and with running panels)

- One heavy process at a time, always `nice -n 15` (or through `tools/lowprio.sh`); at most 2 worker processes in
  anything you start; nothing longer than about 10 minutes without telling the lead first.
- Never run the full test suite (`pytest tests/` or `pytest` without a file): run your own test files only, slow
  ones (`STAGE2_SLOW=1`) one file at a time.
- Every job you write must be resumable and idempotent (the container restarts without warning) and bounded in
  processes and memory.
- **Episodes (design D23, rule X1):** until the lead tells you that task 3's first unit of the 75 cm round has
  run, no R2-mode, policy or training episode on the sets V0, F25F9 or F25F9cl; no closed-loop episode on a look
  set (V6, V6noF5) before its gate G9-ref has passed. Use Stage 1's own run directory through the read-only S1
  view (`setio.stage1_view`, `stage1/runs/2026-11-08/<body>`) or a synthetic set in a temporary directory.

## Engineering rules

New code only under `designlab/stage2/` (and the paths you own). Records are JSON with their rule texts,
provenance (commit, dirty with your own outputs excluded, versions) and sha256, written atomically (round A's
`s2io.write_json`, `sha256_json`, `git_state` once `s2io` is on your branch). Names are formatted with `%g`.
`git check-ignore` must report every path your code writes that is not a committed record. Tests write only under
`tmp_path`; a test that needs arrays, models or torch skips without them. Fast tests in your test file(s); slow
ones behind `STAGE2_SLOW=1`. Every CLI module follows round A's skeleton (thread variables before numpy; `main()`
of the importable copy) and, where the design's 2.1 says so, starts with `tbenv.assign_gl(...)` before any other
import.

## API (frozen)

Exactly the listing of the design's section 2.2 under `lookbar.py` and `s2rows.py`.

## What to build, with pointers

1. **`LookRow`** (3.5 "Where it runs"): `runs`, `gated`, `hard_only` are the inner row's; `name` the inner's;
   `episode`: `b = ctx.built(cond)`; for a bar walk or turn item (`bar.is_bar(cond)` and `cond.judge in
   ("bar_walk", "bar_turn")`) set the instance attribute `b.recorder = lambda: metrics.Recorder(
   legs_every=tune.LEGS_EVERY)`, call `self.inner.episode(ctx, cond, seed, relax)`, and in a `finally` delete the
   attribute (so the class's method is back even after an exception); then `lookread.window(b, rec)`,
   `lookread.series(b.p, rec.poses, rec.loads, rec.dt, tune.LEGS_EVERY * control.FEEDBACK_EVERY)`, `reading`, the
   reference by `lookread.playback(b.p, b.track, b.track.run_dir, ...)` over the window's reference times (cached
   per condition on the row object in the worker), `compare`; the clock block from the recorder's `t_ref` column
   (mean and min factor: the slope of `t_ref` against `t` over 1.05; `t_ref_reached`; `t_ref_planned = 1.05 x
   planned seconds + the track's start`); `policy_check` from `inner.provenance()` when the inner row has it;
   `self.last = block`; the sidecar written atomically when `sidecar` is set (path and schema in 3.5); return the
   inner's `(b, out, rec)` **unchanged**. A fall, a window not reached, or `LookError("not played")` gives a block
   with `look: None` and its reason, never an exception out of `episode`. The wrapper never touches the plant.
   `describe()` = the inner's with `look_reading: {bands, rule, look_sets}`; `sources()` = the inner's plus
   `lookbar.py` and `lookread.py`; `spec`/`from_spec` so that `panels.row_from_spec` rebuilds it in a worker.
2. **`R0L`**: `panels.register_rows("R0L", factory)` at import; the factory returns
   `LookRow(panels.tracker_rows(body, run_dir, ("R0",))["R0"], name="R0L")`.
3. **`rows(body, names, sidecar)`**: like `bar.rows` (parse `<row>@<ms>ms`, build the inner through
   `panels.tracker_rows`, which finds registered controllers), wrapping a row that is not already a `LookRow`, then
   `bar.Delayed(look_row, k, name)`; the sidecar root is given to each `LookRow`.
4. **`episodes(body, set_name, rows, pairs, traces)`** (the monitor's entry): one process; the set read as
   `bar.BarWorker` reads it (`setio.read_inputs(..., require_frozen=True)`, `bar.context_for`); conditions by key
   from `bar.conditions(set_docs, every=True)`; per pair `bar.run_episode(row, ctx, cond, seed,
   trace_root=traces)`, catching `bar.Unreadable` and any other exception separately; the companion episode (3.5
   "Outcome of an episode": `read`, `fell`, `unread` only when the bar raised `Unreadable`, the episode did not
   fall and the row is a policy row, with reason `clock_behind` when its mean clock factor is below 1 and
   `unreadable` otherwise; else `error`). A policy row is one whose inner row has `provenance()`. On set `S1` (development): stock conditions
   from `bar.stock_conditions()`, `panels.judge`'s record, `look: None`.
5. **`run`** and **`collect`** (3.5 "Companion record"): `run` calls `bar.evaluate(set_name, body, out, rows=rows(
   ..., sidecar=<out>/<set>/<body>/<block>/look_partial), conditions=..., block=..., procs=..., attribute="none")`
   then `collect`; `collect` reads each finished bar record of a `LookRow` row, joins its episodes with their
   sidecars (refusing a missing sidecar or one whose `steps`, `fell`, `survival_s` differ), classifies outcomes
   (an episode the bar recorded as failed to run with an "Unreadable" reason is re-classified by the sidecar's
   clock block), computes `counts_look`, `counts`, `counts_but_A1` with `bar.counts` on copies of the episodes
   whose `succeeded` fields are the reading's, and writes `look/<row>/<condition>.json` (schema `s2-look-1`) and a
   summary. Reading 1 counts only on `LOOK_SETS`.
6. **`s2rows`** (3.11): `S2Policy(panels.Policy)` (`begin`: `r5lib.Library(...).from_plan(built.plan)` on the
   built track, a `r5core.Core` on the tracker's estimator and clock; `act`: `model.predict(s2load.normalize(stats,
   obs), deterministic=True)`, `core.filter`, `tk.residual, tk.rate_factor = core.offsets(a_f)`), loaded lazily
   through `s2load.load_checkpoint(..., expect_run=...)`; `provenance()` with both version sets, `cross_version`,
   the fingerprint difference and tolerance; `R5Row`, `R6Row`, the `R5G` row with `HeadScript.fixed`; the three
   build-time checks of 3.11 (fingerprint and run; the run's recorded hashes of `r5core.py`, `r5lib.py`,
   `lookread.py`, `policy.py`, `env.py` and the panels' sources against today's, refused or flagged under
   `STAGE2_ALLOW_SOURCE_CHANGE=1`; the set and freeze by rule K4); `describe()` exactly 3.11's keys (no measured
   value); `sources()`; `spec`/`from_spec` with paths and hashes; the row refuses a condition whose set is not its
   run's. The factories registered for "R5", "R6", "R5G" return the row inside `LookRow`. `s2rows` imports neither
   `r5env` nor `train2` at module level; `equivalence_check` imports `r5env` inside the function.

## Tests to write

`tests/test_stage2_lookbar.py`: every item of the design's section 4 row "The look reading and its wrapper" that
names it: on the S1 view `Delayed(LookRow(R0), k)` against `Delayed(R0, k)` at k = 0 and 3: digest per physics
step equal (a digest wrapper as `g8a` uses) and `panels.judge`'s episode records equal; `BarBuilt.recorder` is the
class's method again after an episode that raised; the sidecar exists before `run_episode` returns; `collect` on
synthetic bar records and sidecars in `tmp_path` (counts by `bar.counts`; an `unread` episode survived and
not succeeded in all three readings; a missing or inconsistent sidecar refused; a non-look set gives
`counts_look: null`); `spec` round trip through `panels.row_from_spec`; `bar._Played` has `q` and `loads`;
`BarBuilt.recorder` is looked up on the instance. `tests/test_stage2_s2rows.py` (needs torch; a freshly
initialised policy saved through `s2load` by a helper): `describe()` holds both version sets and no float from a
measurement; a checkpoint record naming other versions gives `cross_version: true`; a foreign `run_uuid`, a
perturbed fingerprint and a changed `r5core.py` hash are each refused (the last accepted and flagged under the
environment variable); building and running a row imports neither `r5env` nor `train2` (subprocess,
`sys.modules`); slow (`STAGE2_SLOW=1`, S1 view): `equivalence_check` at k = 5 on `75-M-H-10` and, when its model
is restored, on `75-M-H-10-X`; the R6 row against `R6Env`.

## Acceptance

```
nice -n 15 $PY -m pytest -p no:cacheprovider tests/test_stage2_lookbar.py tests/test_stage2_s2rows.py
STAGE2_SLOW=1 nice -n 15 $PY -m pytest -p no:cacheprovider tests/test_stage2_s2rows.py -k "equivalence or r6"     # S1 view only
git status --porcelain            # only your four files
```
On a real set (`lookbar run --set V0 ...`, `equivalence_check` on V0 or V6): only after the lead says that task
3's first unit has run (V0) or that G9-ref passed (a look set); each such episode goes to the lead for
`stage2/tb/dev_episodes.json`.

## Commit

Commit on your own branch in your worktree, messages ending with the attribution lines your session's system
reminder gives, exactly; name no model anywhere else. Do not merge, do not push unless the lead says so. Final
report: files changed, each acceptance command with its exit code, every place where you read the design or a rule
in one of two possible ways (say which you chose: these are rules fixed before the runs, so say so rather than
decide silently), and anything the design asked for that you could not build.
