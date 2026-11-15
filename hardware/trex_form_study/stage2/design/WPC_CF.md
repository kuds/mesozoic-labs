# Work package CF: R2L's fit: `r2lfit.py`

You are building one part of a robotics simulation study (a robotic T. rex biped; everything is simulation). You
need no other context than this brief and the two texts it points into:
- the design: `/tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/design/ROUND_C_DESIGN.md` (read its header, the decisions and sections named under "Read first" below);
- the rules fixed before the runs: `/tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/design/PREREG_TRACKB_draft.md` (rule ids such as "K3" or "W2" refer to it).
Round A's design, where a round-A API is named: `/tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/design/ROUND_A_DESIGN.md`.

**Read first:** design D21, D22, D23; sections 2.2 (`r2lfit`), 3.8 (all of it); rules Q1-Q6, S2-18 in `STAGE2_PLAN.md` section 9, `STAGE1B_PLAN.md` section 3. Code: `designlab/stage1/speedfit.py` (all of it: you repeat its fit on other episodes without editing it), `tune.py` (`CMA`, `Pool`, `score`, `measure`, `training_order`, `derived_seed`, `selection_key`, `to_unit`, `from_unit`, `repair_penalty`), round A's `bar.py` (`BarContext`, `conditions`, `judge`, `run_episode`).

## Goal

Stage 1b's speed fit (CMA-ES over the tracker's 13 gains) repeated on the bar's tasks with a fixed rotation of delays and directions (S2-18), its selection with the 10 ms cell decisive, its budget ledger, and the rows `R2L-<set>/<seed>` for the bar runner. `speedfit.py` is imported, never edited.

## Files you own (and no others)

- `designlab/stage2/r2lfit.py`
- `tests/test_stage2_r2lfit.py`
- `stage2/tb/r2l/index.json` (created as `{"schema": "s2-r2l-index-1", "fits": []}`)

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
Needs from other packages: round A's `bar.py` (package B) and a built set for the real evaluator (round A's S); package CW's `lookbar.LookRow` for `r2l_row` (import inside the factory). Wave 3.

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

Exactly the listing of the design's section 2.2 under `r2lfit.py`.

## What to build, with pointers (design 3.8 is the specification; rule ids in brackets)

1. **`rotation(seed, generations=30)`** [Q2]: per generation g, block `B = g // 6`: the episode seed
   `tune.training_order(seed)[g]`; per slot j = 0..4 (`walk/v0.4`, `turn/q90_<side>`, `stop/two_step`,
   `nudge/<direction>`, `walk/nominal`) the delay `k = (g + j) % 6`; side `("left", "right")[B % 2]`; direction
   `("forward", "left", "backward", "right")[B % 4]`; start and noise seeds `tune.derived_seed(ep, j, 0 | 1)`.
   Written to `rotation.json` with `written_at` before generation 0; a resume refuses a table that differs.
2. **`Evaluator`** (a `tune.Pool` worker; top-level factory `_evaluator`): builds the five bar conditions of the
   generation through a `bar.BarContext` on the fit's set, runs each as the bar's R2 row runs it
   (`panels.TrackerRow` in mode R2 with the candidate's gains) inside `speedfit.servo_delay(b.p, k)`, judges with
   `bar.judge`, and returns the objective's inputs. Every member of a generation runs the same five episodes.
3. **Objective `stage2-r2l-v1`** [Q3]: `tune.score` with the item terms of 3.8 (each graded on [0, 1]).
4. **`fit`**: `tune.CMA`, population 12, sigma0 0.1, from `speedfit.start_record(body)`'s gains
   (`speedfit.gains_vector`), seed 42; resumable per generation (`cma.json`, `evals.jsonl`, as
   `speedfit.RESUME_RULE`); the ledger `ledger.json` sums evaluation CPU seconds; the fit ends at the last whole
   block of six generations inside `budget_core_h` (2.4), or at 30 [Q5].
5. **`select`** [Q4]: candidates = the start and the mean after each third of the generations judged. First, and
   always whole even if the ledger passes its reserve (record `overrun`): every candidate on validation seeds
   2000-2019 at k = 5, five kinds, turn side by seed parity, nudge direction by seed mod 4; rank by
   `tune.selection_key`. Then reported cells while `reported_core_h` (0.4) lasts: the selected candidate and the
   start at k = 0, 2, 4, 1, 3 on seeds 2000-2004; a cell not reached is `{"status": "not run: budget"}`.
   `r2lg5.json` holds `selection`, `by_delay`, `judged_at_generation`, the gains, the rules.
6. **`publish`**: copies the finished records from the work directory to `stage2/tb/r2l/<set>/<body>/`, verifies
   each by sha256, appends `{set, body, seed, r2lg5: {path, sha256}}` to `index.json`.
7. **Rows** [Q6]: at import, for each set in `index.json`, `panels.register_rows(f"R2L-{set}", r2l_row)`;
   `r2l_row(body, run_dir, name)` finds the entry for (set from the name, body, seed from the name), checks the
   gains file's sha256, and returns `lookbar.LookRow(panels.TrackerRow(name, "R2", gains, gains_info, ...))`;
   `describe()` carries the gains file's path and sha256; the row refuses a condition whose set is not its fit's.
8. Outputs default to `stage2/tb/work/r2l/<set>/<body>/` (ignored); nothing live is written elsewhere.

## Tests to write (`tests/test_stage2_r2lfit.py`; a stub evaluator that scores from a formula, no physics)

Every item of the design's section 4 row "R2L": the rotation **per (kind, direction)**: left and right turns each
meet all six delays within two blocks, each nudge direction within four blocks, every slot meets all six delays
within every block; `rotation.json` precedes generation 0 and a changed table is refused at a resume; every member
of a generation receives the same episode list; the selection's decisive cell runs before any reported cell and
finishes when the ledger is forced past its reserve; a reported cell beyond the share reads "not run: budget"; a
fit cut by its budget ends at a multiple of six generations; `publish` then `r2l_row("...", ..., "R2L-V6noF5/42")`
resolves, and refuses another set, a missing entry and a changed gains file; the sha256 of
`designlab/stage1/speedfit.py` is the same before and after importing `r2lfit`; a killed fit resumes at its next
generation with the same CMA state.

## Acceptance

```
nice -n 15 $PY -m pytest -p no:cacheprovider tests/test_stage2_r2lfit.py
nice -n 15 $PY -m designlab.stage2.r2lfit rotation --body 75-M-H-10 --set V6noF5 --seed 42 --out /tmp/<you>/r2l      # prints the table
git status --porcelain            # only your three files
```
The real evaluator's smoke (two generations of population 4 on set V0, `--out /tmp/...`, no record kept) runs
**only after the lead says that task 3's first unit has run**; report its CPU seconds per evaluation (the budget
of rule Q5 rests on 4.4-11 core-seconds per episode).

## Commit

Commit on your own branch in your worktree, messages ending with the attribution lines your session's system
reminder gives, exactly; name no model anywhere else. Do not merge, do not push unless the lead says so. Final
report: files changed, each acceptance command with its exit code, every place where you read the design or a rule
in one of two possible ways (say which you chose: these are rules fixed before the runs, so say so rather than
decide silently), and anything the design asked for that you could not build.
