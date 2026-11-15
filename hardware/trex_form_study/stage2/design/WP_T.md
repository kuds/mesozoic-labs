# Work package T: `barreport.py`, machine-made tables from the bar's records

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
("the design"; read its sections 1 (D9-D14, D20, D22), 3.8 "Record", 4.6's last sentence, 6, 7 item 10).

Background. `designlab/stage2/bar.py` (merged before you start) writes one JSON record per set, body, seed block,
row, and condition under `<run>/<set>/<body>/<block>/eval/<row>/<behavior>/<condition>.json`: 20 episodes each,
with two success readings per episode (`success`: every limit; `success_but_A1`: every limit but a neck or tail
servo's sustained limit), counts, lower bounds, pass flags, attribution, the set freeze it ran on, sessions and a
key. A row's name carries its servo delay (`R2/42@10ms`, `R2/43@10ms`, `R0@10ms`). The study's rule is that
reports are made from records by code, cite record paths, never mix measured values with targets, and keep
"claims of a pull request" apart from "evidence". You write that code. The owner reads these tables: plain
words, units with every number.

## Files you own (and no others)

- `designlab/stage2/barreport.py`, `tests/test_stage2_barreport.py`

Do not edit any other file. Use `s2io` for JSON and paths and `setio` for names; `bar.counts` and `bar`'s rule
texts where you need them; nothing is simulated here.

## Setup

Worktree: the absolute path the lead gives you; work only there.
```
cd <your worktree>/hardware/trex_form_study
PY=/home/user/mesozoic-labs/hardware/trex_form_study/.venv/bin/python
export PYTHONPATH=. MUJOCO_GL=disable OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
```
No arrays are needed. Develop on synthetic records you write in the tests (the schema is the design's 3.8) and,
if the lead points you at one, on the smoke output of work package B.

## API

```python
def load(out, sets=None, body=None) -> dict          # every record under <out>, indexed [set][body][block][row][condition]
def tables(doc) -> dict                              # the JSON form of every table (bar_tables.json)
def markdown(tabs) -> str                            # BAR_TABLES.md, byte-reproducible from bar_tables.json
def jps_table(doc) -> dict                           # the seed-by-seed comparison with JPS section 2.3
def write(out, dest=None) -> dict                    # writes <dest>/bar_tables.json and BAR_TABLES.md (atomic)
def main(argv=None) -> int                           # CLI: python -m designlab.stage2.barreport --out RUN [--dest DIR] [--set S ...]
```

## Tables to make

1. **The walk at 0.40 m/s (task 3).** Per set (F25F9cl first, then F25F9, V0), grip (mu 0.9, mu 0.54) and delay
   (0-10 ms): for R2/42, R2/43 and R0: up of 20, succeeded of 20 in reading 2 and in reading 3, each with its
   one-sided 95 % lower bound (from the record), `pass_`, the controller verdict (`shown`: both R2 rows pass;
   `provisional`: one), the worst drive, the median `along_commanded_heading_m` and `lateral_from_line_max_m`
   (measured values, labelled), the failed terms' counts, falls before and after the command, the attribution
   label. The judged delay (10 ms) first and marked; every other delay marked "reported". Stage 1's reading
   (`measures.stage1.succeeded`) in its own column, labelled "Stage 1's reading".
2. **The speed ladder** (reported): per set, speed rung and delay; the highest rung shown at 10 ms under both
   rows, in each reading.
3. **The rest of the bar (task 4)**: per set, item (turn left, turn right, four nudges, stop), grip and delay: the
   same columns, plus per item: the turn's median heading error at its time and median time to within 10 deg;
   the nudge's recovered count; the stop's touchdown counts and median stopped time.
4. **The bar at a glance**: per set and body, the 16 gated conditions at 10 ms: verdict in reading 2 and in
   reading 3; "not run" with the reason where a reference is absent.
5. **Replication**: every `seeds5042` record beside the `seeds3042` record it replicates (never pooled, never
   averaged), with the reading that triggered it.
6. **The comparison with JPS 2.3** (`stage2/jetson_plem/JETSON_PLEM_AND_SPEED.md` lines 86-101): one row per
   JPS row, columns "JPS's claim" (its numbers, copied into a constant in your module with the file and line) and
   "this round's record" (the matching condition, row and delay, with record paths), then seed by seed
   (3042-3061) up / succeeded in Stage 1's reading and in the bar's two readings for R2/42 and R2/43. The
   mapping: V0 brisk 0.351 m/s -> V0 `walk/v0.35` at 0 ms; V0 0.496 m/s -> V0 `walk/v0.5`; F25 + F9 0.496 ->
   F25F9 `walk/v0.5`; F25_F9_cl at 0.40 / 0.45 / 0.496, both grips -> F25F9cl `walk/v0.4`, `walk/v0.45`,
   `walk/v0.5` and their `_mu0.54`; 2 ms -> the `@2ms` rows; 4 against 6-10 ms at 0.496 -> F25F9cl `walk/v0.5`
   at 4, 6, 10 ms; the stop and the nudges -> `stop/two_step`, `nudge/forward`, `nudge/left`. State in the table
   that JPS's 0.496 m/s is set beside this round's 0.50 m/s rung, and that the stop and turn here are commanded
   causally (design D5), which JPS's were not. A cell with no record reads "not run" (0.55 m/s, the Jetson-mass
   row, 8 ms at 0.50 m/s, the stop from 0.50 m/s or at 2 ms, nudges at 2 ms), or "cut by the pre-registered
   order" when `budget.json` shows its tier was not started.
7. **Checks printed at the top**: records found, records whose `key.sources` differ from today's
   (`panels.panel_sources`, `bar.bar_sources`), records whose input hashes are no longer in their set's latest
   manifest, records with errors or unjudged counts, core-hours spent per task from `budget.json` against 12.

Rules for every table: readings in the plan's order (reading 1 "with the look": "not defined for this set";
reading 2; reading 3); no measured value in a column that holds a target or a limit, and each limit named with
its unit; every F25F9cl row carries the label "margin set after results on seeds 3042-3061"; record paths
relative to the study directory; numbers formatted with `%g` or a stated fixed precision, never names; no
adjectives. `markdown(tables(doc))` must be a pure function of the records (no time, no host).

## Tests to write (`tests/test_stage2_barreport.py`)

On synthetic records: the tables regenerate byte for byte; the two readings are never pooled and differ where
only the tail is over; `shown` needs both rows, `provisional` one; the judged delay is marked; replication sits
beside its original; a missing record prints "not run" and a tier absent from the ledger "cut by the
pre-registered order"; a record with another source hash is listed at the top and still tabulated, flagged; JPS's
numbers appear only under "JPS's claim"; no cell of a measured column equals a target by construction (a
synthetic record whose measured speed differs from its target shows the measured one).

## Acceptance

```
$PY -m pytest -p no:cacheprovider tests/test_stage2_barreport.py
$PY -m designlab.stage2.barreport --out <a smoke run directory the lead names> --dest logs/barreport-smoke    # if one exists
git status --porcelain            # only your two files
```
Done when the tests pass and, on a smoke directory, the Markdown renders every table with "not run" where
records are absent. Label any number you quote from a smoke as "smoke".

## Commit

Commit on your own branch in your worktree, messages ending with the attribution lines your session's system
reminder gives, exactly; name no model anywhere else. Do not merge, do not push unless the lead says so.
