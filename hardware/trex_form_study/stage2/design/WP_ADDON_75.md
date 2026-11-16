# Brief: the add-on to the 75 cm panels (owner's decision P37, 16 Nov 2026)

Working paper. Nothing here is a rule yet. The rules of each item are written into `STAGE2_75_ADDON_PLAN.md`,
reviewed independently and committed **before** that item's first episode.

## What the owner decided (`DECISION_LOG.md` P37)

1. **More delays, then a refit.** 10 ms stays the judged delay. First run today's walk (set `V0`, body `75-M-H-10`)
   at 14, 16 and 20 ms, to see whether 10 ms is a special point for the stand-in controller (Stage 1's fitted
   tracker, rows `R2/42` and `R2/43`, fitted with no delay). Then refit that controller with delay in its mix on this
   body, twice (two fits), and run the bar with the two new fits.
2. **Carpet as a range, no verdict.** Today's walk on a softer and on a grippier floor at 0 to 4 ms of delay, with
   floor values that have no source, reported as a range.
3. Nothing more is run on the candidate `F25F9cl`. The tail servo's remedy is the look's counter-phased tail, which
   comes with the look work; nothing is run for it here.

## What is already known (read these first)

- `STAGE2_75_REPORT.md` (the results that led here), `STAGE2_75_PLAN.md` (the rules of the panels; sections B, C,
  H, R and the closing amendments), `STAGE2_PLAN.md` sections 4 and 9 (R2L: "the tracker refitted as Stage 1b's
  speed fit (0–10 ms; `STAGE1B_PLAN.md` §3) on the bar's tasks"; S2-9 delays in whole 2 ms steps; S2-18 fixed delay
  draws in a fit), `DECISION_LOG.md` F26, F27, A17, P30, P32, P36, P37, `LESSONS_LEARNED.md` section 3 (lessons 1, 2,
  7, 12, 17, 19).
- The runner: `designlab/stage2/bar.py` (`python -m designlab.stage2.bar run --set V0 --body 75-M-H-10 --rows
  R2/42@10ms,... --conditions ... --out DIR --task N --unit NAME --cost COST.md`), `readings.py`, `setio.py`,
  `speedfit.py`; Stage 1's `designlab/stage1/` (panels, tune, control, plant, env, speedfit of Stage 1b under
  `stage1/runs/2026-11-10-speed/`). The chain that ran the panels: `stage2/runs/2026-11-15-75cm/run_75cm.sh`.
- The frozen set: `stage2/sets/V0/75-M-H-10/` (set.json, refs.json, transitions.json). Its arrays are git-ignored
  and present in the main checkout.
- In Stage 1 this body's slow walk fell at 10 ms and stayed up at 20 ms, meeting no limit there (three episodes;
  F26, A17; `stage2/jetson75/LATENCY_AND_BUS.md` near its end: "10 ms is also one period of the 100 Hz feedback").

## Standing rules that bind every item

- **Simulation only.** Prospective design, simulation evidence and measured results stay apart.
- **Pre-registration.** Pass rules, stop rules, seeds, budgets and the order of runs are written before any run. An
  expectation that can be stated from results already known is stated. Any rule changed after a result is labelled.
  Two fits for any verdict. The bar is judged at 10 ms.
- **Bit-identity.** Existing records, models and references regenerate bit for bit. New behaviour sits behind options
  that default to today's behaviour. No Stage 1 or Stage 1b record is rerun or extended with newer code. The record
  keys hash `bar.py`, `readings.py`, `setio.py` and `speedfit.py`: a change to any of them is a new code state and
  must be argued (what regenerates, how it is checked: `stage2/regress/`, the legacy check).
- **New outputs** go under a new run directory (for example `stage2/runs/2026-11-16-75cm-addon/`); `git check-ignore
  -v` on every new output; records are committed, traces are not.
- **Compute** is a 4-core machine: at most 3 processes, through `tools/lowprio.sh`. Budgets in core-hours, with a
  ledger as in the panels (H6).
- **Independent review** of every rule text and every build before a run that depends on it.

## What a design must deliver, per item

For each of the three items (A: delays 14, 16, 20 ms; B: the carpet range; C: the refit with delay):

1. **The rules, ready to paste** into `STAGE2_75_ADDON_PLAN.md`: what is run (set, rows, conditions, floors, delays,
   seeds, episodes), in what order, under what budget and stop rule; what is judged and what is only reported; the
   expectation, with its source; the labels the records and the report must carry; what is *not* run.
2. **Whether today's code can run it as it stands.** Try the runner's own dry paths and read its argument checks:
   does it accept `@14ms`, `@16ms`, `@20ms` rows; a new task number and budget; a floor other than the two it has?
   Where it cannot: the smallest change, behind a default-off option, with the bit-identity argument and the tests
   that would prove it. Name every file a change touches and whether it is in a record's key.
3. **A cost estimate** in core-hours from the committed rates (`stage2/runs/2026-11-15-75cm/COST.md`,
   `budget.json`), and the wall time at 3 processes.
4. **The risks**: what could make the result unreadable or misleading, and what in the design guards against it.

Item specifics:

- **A (delays).** At least the walk at 0.40 m/s on both floors under `R2/42`, `R2/43` and `R0`, and the slow walk
  (`walk/nominal`), at 14, 16 and 20 ms, seeds 3042 to 3061. Say whether 12 and 18 ms should be added and why, as a
  proposal for the lead (the owner approved 14, 16 and 20). State how the result will be read: what pattern would
  mean "10 ms is a special point" and what would mean "delay simply gets worse". No verdict comes from it.
- **B (carpet range).** Define the floors: how "softer" and "grippier" are set in the model (contact parameters,
  friction), with what values and why those, against the known bounds in `stage2/design/carpet_sources_15nov.md`
  (pile 2.4 to 4.4 mm; nothing sourced for friction or sink) and Stage 1's existing `robot_floor_carpet` (unsourced;
  every walking start fell on it). A grid small enough for about 1 core-hour. Reported as "what would have to be
  true of the carpet for the walk to hold", no verdict.
- **C (refit with delay).** Follow Stage 1b's speed fit as `STAGE2_PLAN.md` describes R2L, adapted to this body and
  to the frozen set `V0`: the episode kinds in the fit's mix (the start from standing and the speed-up are where the
  10 ms falls come, so the mix must hold them), the delays and their fixed rotation (S2-18), the fit's seeds (two
  fits), the number of evaluations, the selection rule and its validation episodes, the budget (the plan's estimate
  is about 3.2 core-hours a fit), and the stop rule. Then the panels the two new fits are judged on (the bar's 16
  conditions at 10 ms; 0, 4 and 6 ms reported), their row names, and how they stand beside the stand-in's records.
  Say what Stage 1b's refit on the 60 cm bodies did and did not show (F27), and what in this design differs.

A design changes no file in the repository. It writes its texts under the scratch directory it is given.
