# Work package B: `bar.py`, the bar's conditions, rows, judge, runner and records

> **LEAD ADDENDUM, 15 Nov 2026 (read before anything else; it overrides the text below where they differ).**
> The rules are now being committed as `STAGE2_75_PLAN.md` (study top level) and `STAGE2_PLAN.md` section 9. Read the
> committed text, not only the draft: if your worktree lacks `STAGE2_75_PLAN.md`, run
> `git -C <your worktree> merge --no-edit claude/mesozoic-trex-stage2-refine-4jffre` (the working branch), or read
> `/home/user/mesozoic-labs/hardware/trex_form_study/STAGE2_75_PLAN.md`; if it is not there yet, read
> `/home/user/mesozoic-wt/prereg/hardware/trex_form_study/STAGE2_75_PLAN.md`. The lead's rulings on its twelve open
> points are in `/tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/design/PREREG_RULINGS_15NOV.md`.
> What they change for the build:
> - Task 3's budget is 16 core-hours, task 4's 12; cuts are made by unit (one set at one delay within a numbered
>   tier; one record's attribution is a unit); the first unit that does not fit ends the task; `budget.json` holds the
>   units in the order started with estimate, spent-before, start time and used core-hours, and the cut units;
>   `COST.md`'s rates are measured on Stage 1's own committed conditions, never on a reference of the three sets.
> - Before task 3's first unit the ONLY closed-loop episodes on a Stage 2 set are single check episodes on set V0
>   under R0 rows, each listed in `stage2/runs/2026-11-15-75cm/dev_episodes.json`. R2 rows and the delay wrapper are
>   smoke-tested only on Stage 1's own run directory (the read-only S1 view). No episode of F25F9 or F25F9cl and no R2
>   row on V0, F25F9 or F25F9cl in any smoke, test or acceptance command.
> - The omega control (F25F9cl `walk/v0.4` at mu 0.9 under R2/42 at 10 ms and 0 ms with Stage 1's 0.03 H crouch in
>   omega, 40 episodes, seeds 3042-3061, no verdict, not replicated) writes under `controls/omega_s1/` with a
>   `control` field; `omega` and `crouch_frac` are in every record's key.
> - Registered records are the first finished ones; a later record for the same set, row, condition, delay and block
>   carries `supersedes.reason` and prints as "re-run after a result" beside the registered one. A reference added,
>   renamed or re-versioned after any episode is labelled a change after a result.
> - Verdict words: `shown` / `provisional` only at 10 ms and only for reading 2 or 3 as named; at another delay
>   "passes at <ms> ms". A turn frozen with `a1_over` has reading 2 "not read: a1_over" (not a pass). A candidate's
>   verdict always prints with "replicated", "not replicated" or "replication cut by the pre-registered order"; R0
>   rows give no verdict and are not replicated.
> - A turn failing any gate has no reference and goes to the owner; no fallback, no constant changed.
> - `stop/one_step` is read from Stage 1's command and labelled so; `walk/nominal` is `start`, then `nominal` to the
>   end, judged over its last 10 s with peak ground reaction, cadence and stride judged as Stage 1 judges them.


You are building one module of a robotics simulation study (a robotic T. rex biped; everything is simulation).
You need no other context than this brief and the design it points into:
`/tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/design/ROUND_A_DESIGN.md`
("the design"; read its sections 1 (D1, D9-D14, D16, D19, D20, D22), 2.1, 2.3's `bar.py` block, 3.8 in full, 4.4's
last paragraph, 4.6, 6). Maps of the code you build on: `.../scratchpad/maps/04_panels_metrics.md` (the panels,
rows, records, keys; read sections 2, 3.4-3.5, 4, 5, 7, 8, 10, 14) and `designlab/stage1/ledger.py:738-975` (the
precedent: an own runner on the panels' classes).

Background. Stage 1 judged controllers with "panels": 20 seeded closed-loop episodes per condition and row,
recorded with keys and hashes (`designlab/stage1/panels.py`). Stage 2's "bar" is a new set of conditions (a walk
at 0.40 m/s judged on distance and line, a 90 degree turn judged on heading at a fixed time after a command,
nudges, a stop judged on touchdowns after a command), with a servo delay per row, two floor grips, two seed
blocks and three "readings" of success. It runs on "reference sets" (directories in Stage 1's run-directory
layout under `stage2/sets/<set>/<body>/`). You write the runner in `designlab/stage2/bar.py` on the panels'
classes without changing a byte of Stage 1.

Merged before you start: `readings.py`, `s2io.py`, `setio.py` (work package R). Read their docstrings; do not
edit them. The sets come from work package S, which runs beside you: develop against the pseudo-set `S1`
(`setio.stage1_view("stage1/runs/2026-11-08", body)`, the same layout) and against a tiny synthetic set; your
smoke on a real set is your last step.

## Files you own (and no others)

- `designlab/stage2/bar.py`, `tests/test_stage2_bar.py`
- `stage2/runs/2026-11-15-75cm/run_75cm.sh` (the chain of the design's section 6)

## Setup

Worktree: the absolute path the lead gives you; work only there.
```
cd <your worktree>/hardware/trex_form_study
PY=/home/user/mesozoic-labs/hardware/trex_form_study/.venv/bin/python
export PYTHONPATH=. MUJOCO_GL=disable OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
$PY tools/restore_inputs.py copy-from /home/user/mesozoic-labs/hardware/trex_form_study --group picks-models picks-arrays
```
One heavy process at a time while another builder works; compute cap 1 core-hour (an R2 episode costs about 0.45 s
of wall time per simulated second; R0 0.08). Write every run into a scratch `--out` under `logs/` or a temporary
directory, never into `stage1/runs/` and not into `stage2/runs/2026-11-15-75cm/` (only the script goes there).

## API (frozen)

The `bar.py` listing of the design's section 2.3, with the CLI skeleton of section 2 (thread variables before
numpy; the `__main__` guard imports `designlab.stage2.bar` by name and calls its `main`, so that no class is ever
defined in `__main__`: rows are rebuilt in spawned workers from `spec()["cls"]`).

## What to build

1. **Conditions** (`BarCondition`, `conditions(set_docs)`; design D12): the 16 gated ones and the reported ones a
   set has references for; a condition whose reference is `available: false` is kept and gets a "not run" record.
   Also accept stock `panels.Condition`s for the pseudo-set `S1` (the twelve "no loss" conditions and `bar
   reread`): built by `panels.Context.built`, judged by `panels.judge`, record `definition.stock = true`.
2. **Plans** (`BarPlanner.resolve`; design 3.8's plan table, exactly): lengths for the slowest and the fastest
   clock; the walk item by Stage 1's rule; a periodic item's duration from the entry's `window_s`.
3. **Context and built** (`BarContext(panels.Context)`, `BarBuilt(panels.Built)`): `omega = control.omega0(p,
   q_home, crouch_frac=<set.json crouch_frac>)` (D16); `spec = tune.build_spec(p, set_dir, "walk", None, omega,
   refs_doc, tr_doc, RATE, plan=plan, gait=<walk>)`; then `setio.add_events(spec.track, plan, tr_doc)`; the
   turn's `required` margins by decision 14 from `spec.ref_margins`; `legs` true for `bar_turn`; friction as the
   panels do it (`Context.plant(floor, friction)`). Release cached plants and tracks when the set changes.
4. **Rows** (`Delayed`, `rows`; design D11): `Delayed(panels.Row)` wraps any row: `episode()` is `b =
   ctx.built(cond)`, then `with speedfit.servo_delay(b.p, k): return self.inner.episode(ctx, cond, seed, relax)`;
   `describe()` = the inner's plus `latency_steps`, `latency_ms`, the delay rule text; `sources()` = the inner's
   merged with the sha256 of `bar.py` and `speedfit.py`; `spec()` holds the inner's spec; `runs(cond)` true for
   every condition; `gated(cond)` = `cond.gated`. Names `R2/42@<ms>ms`, `R2/43@<ms>ms` (gains by
   `panels.r2_gains(<stage1 run>/<body>, seed)`), `R0@<ms>ms` (`panels.tracker_rows(...)["R0"]`), and
   `<any registered row>@<ms>ms`; even ms, 0-20.
5. **Judge** (`judge`, `run_episode`): the design's 3.8 (a)-(e) through `readings`' functions on the live
   recorder; the two success vectors; every reported value named there; `panels._base(b, out, rec, seed)` for
   the common fields; `stage1` reading for the walk as 3.8 (b) says. An incomplete judged window without a fall
   raises (the worker turns it into `failed_episode`, which is run again on resume). Check the recorder's
   one-step timing convention once on a real episode, as 3.8's first paragraph asks, and report the result.
6. **Traces**: `panels.write_trace`'s arrays plus `contact` (packed bits), `dt`, `weight`, `t100`, `low100`,
   `swings`, `command_s`; sha256 in the episode record.
7. **Records, keys, resume** (design D14, D19, 3.8 "Record"): your own `record_key`, `condition_record`,
   `not_run_record`, `failed_episode`, `counts(episodes, seeds, which)`, `failure`, `summarize`. Per-episode
   partial files written by the worker (atomic, with the trace's sha256), keyed by the first 16 hex of the key's
   sha256; the condition record when its seeds are complete, then the partials deleted; `sessions[]`; **a
   status-done record with another key is refused with the differing key fields printed, unless `--supersede`,
   which moves it to `<record>.superseded/<12 hex>.json` first**; one `s2io.Lock` on the run directory.
8. **Runner** (`evaluate`, `BarWorker`, `_worker`): `tune.Pool(..., factory=functools.partial(_worker, ...))` as
   `panels.evaluate` and `ledger.py` do; the worker calls `s2io.require_one_thread()`, recomputes `bar_sources()`
   and `panels.fingerprint(panel_sources, key_context)` and refuses a difference, reads the set with
   `setio.read_inputs` (frozen and matching, or the `S1` view), and rebuilds rows from their specs. `--block
   seeds5042 --passed-of seeds3042` selects, for the given set, rows and conditions, exactly the records of the
   first block with `pass_` true in `counts` or `counts_but_A1`, and writes `replication_of` into each new
   record.
9. **Attribution** (design D13): `panels.attribution_plan`, `attribution_record`, `gates.attribute` on R2 rows'
   records (`panels.controller_of(row.name) == "R2"`) with 5 or more falls; a turn, nudge or stop record with
   fewer than 5 falls after its command cites the walk record instead; `--attribute judged` (10 ms rows only),
   `all`, `none`. Lifted runs leave no trace.
10. **Budget ledger** (design D22): `--task N` adds every episode's wall seconds to `<out>/budget.json` (atomic);
    before a `run` it compares `spent + dry-run estimate` with 12 core-hours and refuses to start (exit 3, a
    line saying what was cut); attribution is checked per record.
11. **`dry_run`** (writes nothing; prints and returns episodes, planned seconds and cost from rates measured on
    two short episodes per row type), **`reference_recorder(p, track, seconds)`** (any track played
    kinematically into a `metrics.Recorder`; model on `panels.reference_recorder`, `panels.py:2162`, which takes
    straight walks only; contact from the reference's `support`, loads from its `normal`), **`refcheck`** (every
    reference of a set through that recorder and the bar's readings: G9's "each reference passes its own bar"),
    **`reread`** (design 4.4's real-data check: stock rows, imported with `--row-module`, on stock conditions
    read from `--stage1-run`; each stock trace's sha256 is compared with the committed episode record under
    `--records` (default: the `--stage1-run` directory) before the own-swing reading is taken on the live recorder).
12. **`run_75cm.sh`**: the chain of the design's section 6: a `step` function with done-markers under
    `$RUN/done/`, `tools/lowprio.sh`, each command's own exit code appended to `$RUN/logs/<step>.log`, one step
    per set and delay tier in the listed order, `done/ALL` at the end; a comment with the watchdog line. It must
    be safe to run twice and after a kill. Do not run it.

## Tests to write (`tests/test_stage2_bar.py`)

Every item of the design's section 4.6 (first paragraph), fast on the `S1` view or a synthetic set; one slow
episode per judge behind `STAGE2_SLOW=1` on a built set (skip without it).

## Acceptance (report each command's own exit code)

```
$PY -m pytest -p no:cacheprovider tests/test_stage2_bar.py
$PY -m designlab.stage2.bar run --set S1 --stage1-run stage1/runs/2026-11-08 --body 75-M-H-10 --rows R0@0ms,R2/42@0ms \
    --conditions walk/brisk --episodes 2 --procs 2 --out logs/bar-s1        # then compare the two episodes' stock
                                                                            # measures with the committed Stage 1 record: equal
$PY -m designlab.stage2.bar reread --stage1-run stage1/runs/2026-11-08 --records stage1/runs/2026-11-10-speed \
    --row-module designlab.stage1.speedfit --body 60-N-T-10 --rows R2s/42@10ms,R2@10ms \
    --conditions turn/walking_0.35_left,turn/walking_0.35_right,turn/in_place_0.50_left,turn/in_place_0.50_right \
    --episodes 5 --out logs/bar-reread                                      # 40 episodes; every stock trace sha256 equal
# once work package S has a built V0 set (ask the lead for its path; a set frozen in a scratch root is enough):
$PY -m designlab.stage2.bar dry-run --set V0 --body 75-M-H-10 --rows R0@10ms --out logs/bar-smoke
$PY -m designlab.stage2.bar run --set V0 --body 75-M-H-10 --rows R0@10ms --episodes 1 --procs 2 --out logs/bar-smoke
#   (LEAD, 15 Nov, rule R4 of STAGE2_75_PLAN.md: check episodes on a Stage 2 set are R0 rows on V0 only, ONE episode
#   (seed 3042) per condition, each listed in stage2/runs/2026-11-15-75cm/dev_episodes.json with its reason; R2 rows
#   and the delay wrapper are smoke-tested ONLY on the S1 view, as the command further up does; all 16 gated
#   conditions, a smoke, not judged), then kill a second run midway and resume it
$PY -m designlab.stage2.bar refcheck --set V0 --body 75-M-H-10 --out logs/bar-smoke
git status --porcelain            # only your three files
```
Done when the tests pass, the `S1` episodes reproduce Stage 1's measures, the reread asserts 40 of 40 trace
hashes and lists both clearance readings per episode, the smoke writes 16 records with every reading present, a
killed run resumes to records equal but for wall times and sessions, and `refcheck` reports each reference's own
readings (2 touchdowns; 80 degrees at the turn's time; clearance; 4.0 m). Report the measured cost per episode
and any reading whose definition you found ambiguous: the readings are pre-registered rules, so say so rather
than decide silently. Nothing you run here is a result: label every number in your report "smoke".

## Commit

Commit on your own branch in your worktree, messages ending with the attribution lines your session's system
reminder gives, exactly; name no model anywhere else. Do not merge, do not push unless the lead says so.
