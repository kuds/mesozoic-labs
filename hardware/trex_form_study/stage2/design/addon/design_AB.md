# Design for items A and B of the add-on to the 75 cm panels (P37)

Working paper, 16 Nov 2026. Nothing here is a rule yet. Simulation only: nothing here was built or measured on
hardware. Paths are relative to `hardware/trex_form_study/`.

**Four kinds of statement are kept apart.** *Simulated*: counts from committed records. *Planned*: figures from a
model at rest or a motion played without physics. *Design estimate*: my own arithmetic or reasoning. *Measured on
hardware*: none exists.

**Limits kept.** I changed no file in the repository (`git status --short` printed 0 lines at the end). I ran no
closed-loop episode, no fit, no training and no standing settle. I ran `bar dry-run` with the committed rates, pure
functions of the runner, and static arithmetic on the compiled model (`settle=False`, no physics step). One command
was refused by the session's permission check before it started: a set of `bar run` calls meant only to print the
runner's refusals. I replaced it with direct calls of the same check functions. So every refusal quoted below comes
from a function call or from reading the code, and I say which.

## 0. The short version

- **Item A needs no code change.** Today's runner takes `@14ms`, `@16ms` and `@20ms` rows. Three commands, 540
  episodes, 0.384 core-hours by the committed rates, about 10 minutes at 3 processes.
- **A's records will sit at the same code state as the registered 0 to 10 ms records.** I recomputed the key of a
  registered record with today's code and it is equal, field for field. That holds only while no module in a key
  changes, so A should run first.
- **The runner refuses a new task number.** `--task` takes 3 or 4 only. A can use `--task 3` in its own run
  directory. The ledger there will show the runner's 16 core-hour cap, so the add-on's own cap needs a guard in the
  chain script.
- **I propose adding 12 and 18 ms, and the slow walk at 8 ms** (+0.28 core-hours). Without 12 ms the report cannot
  say whether 10 ms is the centre of a window or just its first failing point.
- **Item B cannot run on today's command line.** The bar has two floors, both the hard floor, and builds its
  conditions in code.
- **B's smallest change edits no module in any key.** A new file, `designlab/stage2/floors.py`, makes the floor a
  property of the row, the way `bar.Delayed` makes the delay one. `bar.py` stays as it is, so there is no new code
  state and item C's panels stay beside the stand-in's records. I prototyped it in scratch and checked its wiring
  without any physics step.
- **B's grid:** seven floors at 0, 2 and 4 ms, 0.87 core-hours by the committed rates, cap 1.25, about 25 to 30
  minutes at 3 processes. Stage 1's carpet floor lets a pad sink 2.06 mm on one foot (planned). That is 86 % of the
  thinnest pile in the lookup (2.4 mm), so it sits at the soft end of what a thin carpet can be.

## 1. What I checked in today's code (shared by A and B)

Each line gives its source. "My call" means I called the function today and read its output.

| Fact | Source |
|---|---|
| A row is `<row>@<ms>ms`. The delay must be a whole even number of ms from 0 to 20. | `designlab/stage2/bar.py:134-136`, `850-865` |
| `R2/42@12ms`, `@14ms`, `@16ms`, `@18ms`, `@20ms` parse to 6, 7, 8, 9, 10 steps. `@22ms`, `@15ms`, `@24ms` are refused: "a delay of whole 2 ms physics steps, 0-20 ms, not 22". | my call of `bar.parse_row` |
| `bar.rows` builds `R2/42@14ms`, `R2/43@16ms`, `R0@20ms` as `Delayed` rows with k = 7, 8, 10. | my call of `bar.rows` |
| The delay is Stage 1b's queue of the last k + 1 servo targets, with no upper limit of its own. | `designlab/stage1/speedfit.py:324-361` |
| A set offers 41 conditions, 16 gated. Their floors are `('hard', None)` and `('hard', 0.54)` and nothing else. | my call of `bar.conditions` on V0; `bar.py:103`, `457-502` |
| A run names conditions by key, by bare name or by group (`@gated`, `@walk`, `@rest`, `@speed-ladder`, `@ladders`). An unknown name is refused. | `bar.py:506-534`; my call of `bar.choose` with `walk/v0.4_carpet` |
| `--task` takes 3 or 4 only. The caps are 16 and 12 core-hours. | `bar.py:140`, `3468` |
| `Budget(task=5)` is refused ("task 5: choose from 3, 4"). `Budget(task=5, cap_core_h=1.5)` is accepted, and `evaluate()` has a `budget_cap_core_h` argument. The command line has no switch for it. | my calls; `bar.py:1800-1804`, `2328`, `2502` |
| On V0 a run without `--task` must be a one-episode `R0` build check. With R2 rows it is refused. | my call of `bar._check_rule`; `bar.py:2195-2222` |
| With `--task`, a new directory under `stage2/runs/` is accepted as `--out`. Without `--task` it is refused. | my calls of `bar._check_out`; `bar.py:2230-2244` |
| `--cost` defaults to `<out>/COST.md`. A new directory has none. `read_cost` then raises a file-not-found error before any episode. | `bar.py:3494-3495`, `1871-1877`; read, not run |
| The committed rates are 0.1574 (R2) and 0.0954 (R0) wall seconds per simulated second. | my call of `bar.read_cost` on `stage2/runs/2026-11-15-75cm/COST.md` |
| `budget.json` is made at `<out>/budget.json` by the first unit. A new directory starts at 0 core-hours. A unit's estimate is its planned simulated seconds times the rates. A unit that does not fit ends with exit 3 and cuts every later unit. | `bar.py:1792-1851`, `2498-2511`, `3526-3528` |
| `--unit` is free text, the unit's name in the ledger. | `bar.py:2494-2497`, `3469` |
| The runner asks for a frozen set whose files match the freeze. V0 passes: 1 freeze, index 0, manifest `34a820e550b6...`, 20 entries. | my call of `setio.read_inputs(require_frozen=True)`; `bar.py:2389`, `2404` |
| The runner has no check of the commit or of "code state A". A dirty checkout is recorded in the record, not refused. | `bar.py:2409-2415`; `designlab/stage2/s2io.py:400-420` |
| A finished record of another key is never overwritten. A new directory holds none. | `bar.py:2457-2474` |
| Today's key sources equal the registered records': bar `45719d030359ec7d5d3914af3814d837fdc05d77652ba45c64792b511027528d`, panels `c0027992de21067525cc4489b74acb157feb89f3a4c7618580addc2b341be1c1`. | my calls of `bar.bar_sources()` and `panels.panel_sources()`, against `stage2/runs/2026-11-15-75cm/V0/75-M-H-10/seeds3042/eval/R2/42@10ms/walk/v0.4.json` |
| The key of `R2/42@4ms`, `walk/v0.4`, recomputed with today's code, equals the registered record's key with no difference. | my call of `bar.record_key` against `.../eval/R2/42@4ms/walk/v0.4.json` |
| Since code state A (`c377ba1`) only `barreport.py` and `barviews.py` changed under `designlab/`. Neither is in a key. | `git diff --stat c377ba1 HEAD -- designlab/` |
| Default `--attribute judged` attributes rows at 10 ms only. At 14 ms it does nothing; `all` would. | my call of `bar._attribution_needed`; `bar.py:2283-2295` |
| The summary prints "passes at <ms> ms" or "does not pass at <ms> ms" at any delay but 10, never "shown". | `bar.py:2753-2756` |
| `barreport` hard-codes its delays (10, 0, 2, 4, 6, 8). It will not tabulate 12 to 20 ms. | `designlab/stage2/barreport.py:167-180` |
| In a new run directory the records, `budget.json` and `summary.json` are not ignored by git. `traces/`, `partial/`, `logs/`, `done/` and `.lock` are. | my `git check-ignore -v`; `stage2/runs/.gitignore:3-7` |

## 2. Item A: today's walk at 14, 16 and 20 ms

### 2.1 How the result will be read (stated before any result)

**The question.** Is 10 ms a special point for the stand-in controller, or does delay simply get worse?

**What is known (simulated, committed records).** Episodes that stayed up, of 20, set V0, 0.40 m/s walk
(`STAGE2_75_REPORT.md` section 5; I re-read the records):

| Row, floor | 0 ms | 2 ms | 4 ms | 6 ms | 8 ms | 10 ms |
|---|---|---|---|---|---|---|
| `R2/42`, μ 0.9 | 20 | 20 | 20 | 20 | 0 | 0 |
| `R2/43`, μ 0.9 | 20 | 20 | 20 | 20 | 0 | 1 |
| `R2/42`, μ 0.54 | 20 | 20 | 20 | 20 | 0 | 0 |
| `R2/43`, μ 0.54 | 20 | 20 | 20 | 20 | 0 | 3 |
| `R0`, μ 0.9 | 20 | 18 | 19 | 20 | 19 | 20 |
| `R0`, μ 0.54 | 20 | 19 | 20 | 20 | 20 | 20 |

The slow walk (`walk/nominal`, μ 0.9): `R2/42` 20, 20, 20, 2 and `R2/43` 20, 20, 20, 1 at 0, 4, 6 and 10 ms. It was
not run at 2 or 8 ms, and never under `R0`.

Stage 1's margin ledger, slow walk, 3 episodes per delay, `R2/43`
(`stage1/runs/2026-11-08/75-M-H-10/sweeps/R2/43/latency.json`): at 10 ms 0 of 3 stay up (falls at 4.4 to 6.5 s). At
20 ms 3 of 3 stay up and none meets every limit. The 20 ms episodes walk at 0.240 m/s against 0.250 m/s with no
delay, with a peak ground force of 1.94 to 2.17 body weights against 1.21 to 1.26, and an ankle drive at 1.23 of its
rated speed. `R2/42` was never run above 10 ms. No delay from 12 to 18 ms was ever run on any body
(`stage2/jetson75/LATENCY_AND_BUS.md:318`).

**Reading words.** They describe a count. They are not passes and not verdicts.

- A record is **up** when at least 17 of its 20 episodes stay up. It is **down** when at most 3 do. Otherwise it is
  **mixed**. Why 17 and 3: 17 is rule H1's number, and 3 is the most that stayed up in any registered `R2` record of
  this walk at 8 or 10 ms.
- A delay is **up** when all four `R2` records of the 0.40 m/s walk are up (two fits, two floors). It is **down** when
  all four are down. Otherwise it is **mixed**, and the four counts are printed.
- By these words the registered delays read: 0, 2, 4 and 6 ms up; 8 and 10 ms down.

**The patterns.**

- **"Delay simply gets worse."** Every delay run above 10 ms reads down. Then nothing from 8 ms up holds, and 10 ms is
  not special.
- **"10 ms is a special point" (a window).** At least one delay run above 10 ms reads up, while 8 and 10 ms read down.
  The report then names the lowest delay above 10 ms that reads up.
- **"Neither."** No delay above 10 ms reads up and at least one is mixed. The report prints the counts and uses
  neither phrase.

The slow walk is read the same way from its two `R2` records, in its own paragraph. The two paragraphs are not merged.

**What a window would and would not mean.**

- It would not make any delay a verdict. The bar is judged at 10 ms (P30, P32, P37).
- It would not mean the build is fine at that delay. Stage 1's three 20 ms episodes met no limit set (above).
- It would describe the survival curve. It would not explain it. No mechanism is tested by these runs.
- It would bear on item C: a refit with delay in its mix would then be moving a narrow failure, not buying margin
  against a steady trend. That is the lead's reading to make, as an interpretation.

**Where before why.** Every count is printed with where the falls come (the start from standing, the slow stride,
the change of speed, the 0.40 m/s walk itself), from the episodes' `segment_reached` and fall times. At 10 ms 76 of
80 episodes fell and none fell in the 0.40 m/s walk itself (`STAGE2_75_REPORT.md` section 5). So a 0.40 m/s record
at a longer delay may say nothing about walking at 0.40 m/s.

**Of the episodes that stay up** the report prints, per record and not judged: the median speed, the toe clearance,
the failed terms with their counts, the largest drive speed ratio, and for the slow walk the peak ground force.

**Expectations, stated from results already known.**

1. `R0` rows: 18 to 20 of 20 stay up at every delay, as at 0 to 10 ms. `R0` has no feedback, so a delay only shifts
   its commands in time (design estimate). An `R0` record under 17 is a flag on the runner at long delays, not a
   finding about the controller. The `R2` cells of that delay are then printed with the flag and reviewed before
   they are read.
2. The slow walk under `R2/43` at 20 ms: most episodes stay up, from Stage 1's 3 of 3. The plans differ: Stage 1's
   walk is start, walk, stop over 18.4 s; the bar's is start then 8 strides over 14.06 s (my dry run). A record that
   reads down would contradict Stage 1's three episodes and is reported as a contradiction, with both records.
3. `R2/42` at 20 ms, every row at 14 and 16 ms, and the 0.40 m/s walk at 20 ms: no expectation can be stated. No
   record exists.
4. No record at 14 to 20 ms is expected to meet every limit in 17 of 20. From 6 ms up the registered records fail
   toe clearance in 19 or 20 of 20, and Stage 1's 20 ms episodes failed speed, cadence, stride, ground force and
   drive speed.

**A design estimate, labelled as such, for why a window is possible at all.** A pure delay turns the phase of a
loop component at frequency f by 360° × f × delay, so the same phase returns when the delay grows by 1/f. The
tracker's feedback is computed at 100 Hz and ramped in over 10 ms (`designlab/stage1/control.py:20-22`). A component
that alternates from one update to the next has a 20 ms period: 10 ms of delay flips its sign and 20 ms restores it.
If that were what fells the walk, the curve would mirror about 10 ms: 12 like 8, 14 like 6, 16 like 4, 18 like 2,
20 like 0. This is arithmetic, not a result. The baseline loop already lags (sensors one step old, the ramp), so the
centre need not sit at exactly 10 ms. It is why 12 and 18 ms are worth their cost (2.5).

### 2.2 The rules, ready to paste

> **A. More delays (P37, answer 1, first part).** Added on 16 Nov 2026 after the results of the 75 cm panels
> (`STAGE2_75_REPORT.md`) and labelled so. Rules of `STAGE2_75_PLAN.md` hold unless this section says otherwise.
>
> **A-1. What is run.** Set `V0`, body `75-M-H-10`, block `seeds3042` (seeds 3042 to 3061), 20 episodes per record,
> nominal plant, commanded 5 % fast (B1). Conditions: `walk/v0.4`, `walk/v0.4_mu0.54` and `walk/nominal`. Rows:
> `R2/42@<ms>ms`, `R2/43@<ms>ms`, `R0@<ms>ms` with the gains of `stage1/runs/2026-11-08/75-M-H-10/tune.json`
> unchanged (B2). Delays: 14, 16 and 20 ms (k = 7, 8, 10 physics steps), which `STAGE2_PLAN.md` S2-9 names and the
> round listed as not run. That is 9 records and 180 episodes per delay.
>
> **A-2. Order and units.** One unit per delay, in this order: `A1-V0-14ms`, `A1-V0-16ms`, `A1-V0-20ms`.
> [If the lead adopts 2.5: then `A2-V0-12ms`, `A2-V0-18ms`, each as A-1, and `A3-V0-nominal-8ms`: `walk/nominal`
> under `R2/42@8ms` and `R2/43@8ms`.] The run directory is `stage2/runs/2026-11-16-75cm-addon/A/`. The rates are
> those of the committed `stage2/runs/2026-11-15-75cm/COST.md`.
>
> **A-3. Budget and stop rule.** 1.0 core-hour, counted as the sum of episode wall times in
> `stage2/runs/2026-11-16-75cm-addon/A/budget.json`. A unit starts only if the spent core-hours plus its estimate
> fit 1.0. The first unit that does not fit is cut with every unit after it. A started unit finishes. The ledger
> file names its task "3" and prints a cap of 16 core-hours: that is the stock runner's slot and constant, and the
> cap of this section is 1.0, held by the chain script. An episode that failed to run is run again once by the same
> command (C8); a record that still holds one is not read. After the first record of this section exists, no
> delay, seed, row or condition is added or run again except as a new, labelled addition.
>
> **A-4. Code state.** The units run at code state A's key sources: bar `45719d03...528d`, panels
> `c0027992...e1c1` (full values in the chain script). The chain refuses to start on other hashes or on a checkout
> with uncommitted changes. No module that enters a record key is changed before these units have run.
>
> **A-5. What is judged.** Nothing. No verdict comes from this section and none changes. The bar stays judged at
> 10 ms (P30, P32, P37). A cell gives its counts and, by H2, "passes at <ms> ms" or "does not pass at <ms> ms",
> never `shown` or `provisional`.
>
> **A-6. What is reported, and how it is read.** [paste 2.1 from "Reading words" to "Expectations", items 1 to 4.]
> The registered records at 0 to 10 ms under `stage2/runs/2026-11-15-75cm/` are printed beside the new ones, each
> with its path. Nothing is read between delays (B3).
>
> **A-7. Not attributed.** These records take no fall attribution (`--attribute none`). In the registered round the
> attribution of records where 0 of 20 stayed up rested on a change of one physics step (`STAGE2_75_REPORT.md`
> section 5).
>
> **A-8. Labels.** Every table and sentence from these records carries: "simulation; stand-in controller fitted
> with no delay; delays above the judged 10 ms; no verdict; added after the 75 cm results (P37)".
>
> **A-9. Not run.** Turns, pushes and stops at these delays; the speed ladder's other rungs; `walk/nominal_mu0.54`;
> the sets `F25F9` and `F25F9cl` (P37, answer 3); the block `seeds5042`; delays above 20 ms (the runner's limit);
> any delay that is not a whole 2 ms step.

### 2.3 Whether today's code can run it: yes, as it stands

The command lines, in order. I did not run them. Their dry-run forms (`dry-run` in place of `run`, without
`--procs`, `--attribute`, `--task`, `--unit`) ran and listed 9 records of 20 episodes per delay, all "to run".

```bash
cd hardware/trex_form_study
export PYTHONPATH=. MUJOCO_GL=disable PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
RUN=stage2/runs/2026-11-16-75cm-addon/A
COST=stage2/runs/2026-11-15-75cm/COST.md
BAR="bash tools/lowprio.sh .venv/bin/python -m designlab.stage2.bar"

$BAR run --set V0 --body 75-M-H-10 --rows R2/42@14ms,R2/43@14ms,R0@14ms --conditions @walk,walk/nominal \
    --procs 3 --attribute none --task 3 --unit A1-V0-14ms --cost $COST --stage1-run stage1/runs/2026-11-08 --out $RUN
$BAR run --set V0 --body 75-M-H-10 --rows R2/42@16ms,R2/43@16ms,R0@16ms --conditions @walk,walk/nominal \
    --procs 3 --attribute none --task 3 --unit A1-V0-16ms --cost $COST --stage1-run stage1/runs/2026-11-08 --out $RUN
$BAR run --set V0 --body 75-M-H-10 --rows R2/42@20ms,R2/43@20ms,R0@20ms --conditions @walk,walk/nominal \
    --procs 3 --attribute none --task 3 --unit A1-V0-20ms --cost $COST --stage1-run stage1/runs/2026-11-08 --out $RUN

# only if the lead adopts 2.5
$BAR run --set V0 --body 75-M-H-10 --rows R2/42@12ms,R2/43@12ms,R0@12ms --conditions @walk,walk/nominal \
    --procs 3 --attribute none --task 3 --unit A2-V0-12ms --cost $COST --stage1-run stage1/runs/2026-11-08 --out $RUN
$BAR run --set V0 --body 75-M-H-10 --rows R2/42@18ms,R2/43@18ms,R0@18ms --conditions @walk,walk/nominal \
    --procs 3 --attribute none --task 3 --unit A2-V0-18ms --cost $COST --stage1-run stage1/runs/2026-11-08 --out $RUN
$BAR run --set V0 --body 75-M-H-10 --rows R2/42@8ms,R2/43@8ms --conditions walk/nominal \
    --procs 3 --attribute none --task 3 --unit A3-V0-nominal-8ms --cost $COST --stage1-run stage1/runs/2026-11-08 --out $RUN
```

What the runner's own checks say about them:

- **Rows.** Accepted (section 1). The seeds are the default block `seeds3042` and the default 20 episodes.
- **Conditions.** `@walk` is the two gated walk conditions; `walk/nominal` is the reported slow walk on μ 0.9. The
  planner gives 21.078 s per episode for the 0.40 m/s walk (start, one slow stride, the change, 11 strides) and
  14.060 s for the slow walk (start, 8 strides). My call of `BarPlanner.resolve`.
- **`--task`.** Needed, and only 3 or 4 exist. `--task 3` is the walk task's slot. A new directory keeps its own
  ledger. A new number needs code (see below).
- **`--cost`.** Must be given, because the new directory has no `COST.md`.
- **Freeze and code state.** The set is frozen and its files match. No check in the runner would refuse a run
  today. The code-state guard of rule A-4 is the chain script's, not the runner's.
- **Outputs.** Records at `<RUN>/V0/75-M-H-10/seeds3042/eval/R2/42@14ms/walk/v0.4.json` and alike, plus
  `eval/summary.json` and `<RUN>/budget.json`: committed. Traces: ignored. The chain script lives one level up, at
  `stage2/runs/2026-11-16-75cm-addon/run_addon.sh`.

**What is not in the runner and has to be added, in no key:**

1. **A chain script** in the form of `run_75cm.sh` (steps, `done/` markers, exit 3 and 4 handling). Before the
   first unit it checks `git status --short` is empty, compares the two key-source hashes with the values of A-4,
   and prints each unit's dry run.
2. **The cap guard.** Before each unit: read task 3's spent core-hours from `<RUN>/budget.json`, add the unit's
   dry-run total, and if the sum exceeds 1.0 mark this unit and every later one as cut. About ten lines.
3. **A table script** that reads the registered directory for 0 to 10 ms and the add-on's for the rest, and prints
   each cell with its record path. `barreport` cannot do it as it stands.

**If the lead wants a true task number and cap in the ledger.** Two ways, neither needed for A:

- A thin command in a new file that takes the same arguments plus `--task N --budget-core-h X` and calls
  `bar.evaluate(..., task=N, budget_cap_core_h=X)`. The class and the function accept that today; the tests already
  call it (`tests/test_stage2_bar.py:1550`, `1562`). The file is in no existing key. Item B's new file has this
  command anyway (3.3), so A could use it too, at the price of waiting for B's build and review.
- Editing `bar.py` (`BUDGET_CORE_H`, the `--task` choices). `bar.py` is in every bar record's key, so this makes a
  new code state and S2-17 then asks for the baseline rows to be run again. I advise against it.

### 2.4 Cost

By my `bar dry-run` with the committed rates:

| Unit | Records | Episodes | Simulated hours | Core-hours (estimate) |
|---|---|---|---|---|
| one delay (`@walk` and `walk/nominal`, three rows) | 9 | 180 | 0.937 | 0.128 |
| A1: 14, 16, 20 ms | 27 | 540 | 2.81 | **0.384** |
| A2: 12 and 18 ms (proposal) | 18 | 360 | 1.87 | 0.256 |
| A3: slow walk at 8 ms, two `R2` rows (proposal) | 2 | 40 | 0.156 | 0.025 |
| all of it | 47 | 940 | 4.84 | **0.665** |

- In the registered ledger the V0 units whose episodes ran full length used 1.10 to 1.42 times their estimate, and
  units with early falls 0.59 to 0.83 times (`stage2/runs/2026-11-15-75cm/budget.json`, at 3 processes). So A1 may
  use up to about 0.54 core-hours and all of it up to about 0.94. The cap of 1.0 holds both.
- **Wall time at 3 processes** (design estimate): A1 about 8 to 11 minutes, all of it about 14 to 19 minutes. The
  registered units took 4 to 24 s over their compute time each.
- Disk: about 4 MB of traces per record of 20 full episodes (27 MB for one row's walk records at 6 ms); under
  200 MB in all. 15 GB are free.

### 2.5 Proposal for the lead: add 12 and 18 ms, and the slow walk at 8 ms

The owner approved 14, 16 and 20 ms. These three are extra and placed last, so a budget cut removes them first.

- **12 ms.** 8 and 10 ms are down. 12 ms is the pair of 8 ms about 10 ms. If 14 ms reads up, 12 ms says whether the
  window ends just above 10 ms or runs to 13 ms. If 14 ms reads down, 12 ms costs little and closes the grid.
- **18 ms.** If 16 and 20 ms read differently, 18 ms is the only cell that says where the change is. With it the
  grid is every 2 ms step from 0 to 20 ms, and the pairs (8, 12), (6, 14), (4, 16), (2, 18), (0, 20) are complete.
- **The slow walk at 8 ms.** The slow walk has no record at 8 ms, the last delay before it falls. Stage 1 had 3 of
  3 up there, missing toe clearance. Without it the slow walk's curve has a hole next to its only failing point.
- Not proposed: the slow walk at 2 ms (nothing turns on it), and 22 or 24 ms (the runner refuses them).

### 2.6 Risks, and what guards against each

| Risk | Guard |
|---|---|
| A fall in the start or the speed-up is read as "cannot walk at 0.40 m/s". | Every count is printed with where the falls come. The slow walk is its own paragraph. |
| "Stays up at 20 ms" is read as "works at 20 ms". | The reading rule says it is not. The limits columns are printed. Stage 1's 20 ms numbers are quoted. |
| A window is read as an explanation. | The patterns are words for counts. The 20 ms period argument is labelled a design estimate. |
| Three delays leave holes, and the eye fills them. | Nothing is read between delays. 12 and 18 ms are proposed. |
| The two fits differ, and one is quoted. | A delay is up or down only when all four `R2` records agree. `R2/42` has no record above 10 ms, and the text says so. |
| 20 episodes and a chaotic walk: a wide "mixed" band. | The band is named and its counts printed. No significance is claimed. |
| The bar has never run k above 5. The delay queue could misbehave at k = 7 to 10. | The `R0` rows and expectation 1. The queue is generic and Stage 1's ledger ran k = 10 through the same model. |
| A key module changes before A runs, and the curve then spans two code states. | Rule A-4 and the chain's hash guard. A runs before B's and C's builds are merged, or the guard proves they touched no key. |
| The ledger's "task 3, cap 16" is read as the add-on's budget. | Rule A-3 says what it is. The cap guard is in the chain. |
| Tables mix two directories. | The table script cites each cell's record path. |
| The slow walk under `R0` has no baseline at 0 to 10 ms. | It is a flag only (expectation 1), and the text says no baseline exists. |

## 3. Item B: the carpet range, no verdict

### 3.1 How a floor is defined today

- **A bar condition carries two fields from Stage 1's `Condition`:** `floor` (default `"hard"`) and `friction`
  (default none) (`designlab/stage1/panels.py:429-430`). The bar builds its conditions in code, from
  `GRIPS = ((None, ""), (0.54, "_mu0.54"))` (`bar.py:103`, `477-501`). No bar condition sets `floor`.
- **A `_mu0.54` condition** is the hard-floor plant with `control.set_friction(p, 0.54)`: every colliding geom's
  sliding friction becomes 0.54 (`panels.py:979-988`; `designlab/stage1/control.py:1993-1997`). The contact's
  stiffness and damping stay as compiled.
- **The hard floor as compiled** (planned; my read of the compiled model): floor geom `solref` (0.02 s, 1.0),
  `solimp` (0.9, 0.95, 1 mm), friction 0.9, `condim` 3, priority 0. With the pads the pair is `solref` (0.015 s,
  1.0), friction 0.9. A pad sinks 0.11 mm with the weight on two feet and 0.22 mm on one (closed form,
  `carpet.static_depth`).
- **Stage 1's floors** are `hard`, `robot_floor_carpet` and the probe piles `low`, `medium`, `high`
  (`designlab/stage1/plant.py:86`, `901-917`). A pile is a field of several hundred spring probes under each foot.
- **`robot_floor_carpet` sets** the floor geom to `solref` (0.024 s, 1.2), `solimp` (0.65, 0.94, 4 mm), friction
  (0.95 sliding, 0.009 m torsional, 0.0005 m rolling), `condim` 6, with the floor's priority raised so its values
  win (`designlab/carpet.py:935-953`; `designlab/data/research/carpet_2026-11-07.json:132`). The file calls them
  "calibration hypotheses" with "no source or specimen fit". Stage 1's carpet walk then set the sliding friction to
  0.54 (`designlab/stage1/env.py:103`).
- **What that floor is, against the lookup's bound** (planned; my arithmetic on the compiled model, mass
  14.86 kg, 16 pad contacts per foot, pad area 35.4 cm² per foot): a pad sinks 1.59 mm on two feet and **2.06 mm on
  one foot**. The lookup's pile heights are 2.4 to 4.4 mm (`stage2/design/carpet_sources_15nov.md`, secondary
  figures). So the surrogate sinks 86 % of the thinnest pile and 47 % of the thickest. It is softer than the
  study's own low-pile model, whose sourced stiffness (32.9 kPa/mm) gives 1.25 mm on one foot, and P36 says that
  6.35 mm pile is already softer than the floor meant.
- **What Stage 1 found on it** (simulated): standing, 20 of 20 under both fits. The carpet walk, 0 of 20 under both
  fits, every fall at 4.75 to 5.16 s, in the start from standing
  (`stage1/runs/2026-11-08/75-M-H-10/eval/R2/42/carpet/robot_floor_carpet.json`, and `/43/`). The fall label there
  is "body-limited: slip", but with slip lifted 0 of 20 stayed up: the label rests on small shifts of the fall
  time. F26 records that the starts also fell at the carpet's own friction of 0.95 (3 starts). `R0` was never run
  on it.

### 3.2 Can the runner be given another floor without a code change? No

- No switch names a floor. `--conditions` picks from the 41 built-in conditions; my call with another name was
  refused.
- `--row-module` exists and imports a module that can register rows (`bar.py:3479`, `3485-3486`;
  `panels.py:861-871`). That hook is what the smallest change uses.

### 3.3 The smallest change: one new file, and no module in any key is edited

**The idea.** `bar.Delayed` wraps a row and puts a servo delay on the row's plant for the episode
(`bar.py:880-918`). A floor can be wrapped the same way. Stage 1's ledger already swaps plant values in place and
restores them (friction, servo stiffness, mass, the piles' damping: `designlab/stage1/ledger.py:802-850`).
`STAGE2_PLAN.md` S2-17 asks for exactly this: new work "subclass or wrap such modules in `designlab/stage2/`".

**The file: `designlab/stage2/floors.py`** (new). It holds:

1. **A table of floors** by name, each with its recipe and its reason (3.4).
2. **`floor_on(p, name)`**, a context manager. It saves five model arrays (`geom_priority`, `geom_condim`,
   `geom_solref`, `geom_solimp`, `geom_friction`) and the settled pose; applies the floor with Stage 1's own
   `carpet.apply_floor` and `control.set_friction`; settles the zero-command stance on it with Stage 1's own
   `Plant.settle`; and restores everything on exit.
3. **`OnFloor`**, a row: the inner row's episode inside `floor_on`. Its name is `<R0|R2>~<floor>[/<seed>]@<ms>ms`,
   for example `R2~c1.25_mu0.9/42@4ms`. It is built as `OnFloor(Delayed(TrackerRow))`: the floor outside the
   delay. It is never gated, and it runs only the walk conditions of the hard floor as compiled.
4. **A small command**, `python -m designlab.stage2.floors {check|dry-run|run}`. `run` builds the rows and calls
   `bar.evaluate(set, body, out, rows={name: row}, conditions=["walk/v0.4"], attribute="none", task=N,
   budget_cap_core_h=X, unit=..., rates=...)`, with `bar.main`'s exit codes. `check` builds each floor on the
   compiled model, settles the standing pose, and writes `floors_check.json`: the resolved contact values, the
   closed-form sinks, the settle's numbers. `check` runs no episode.

**Why its own command and not the stock one.** Through the stock `--rows` the runner always wraps a registered row
in `Delayed`, and `Delayed` answers "is this gated" and "does this run" itself (`bar.py:893-897`). My prototype
showed the result: the record would be marked `gated: true` and the row would run on `_mu0.54`, turn and stop
conditions too. With the floor outside the delay the record reads `gated: false`, and the row refuses those
conditions. The own command also takes a true task number and cap, which the stock one cannot.

**What I checked on a scratch prototype** (`scratchpad/addon/proto/floors_proto.py`; no physics step taken):

- The stock runner builds `Delayed(OnFloor(TrackerRow))` from `--row-module`, and its dry run lists the rows.
- `OnFloor(Delayed(TrackerRow R2/42, k = 2))` survives the round trip a worker makes (`row_from_spec`).
- It reads `gated: false`; it runs `walk/v0.4` and `walk/nominal`, and refuses `_mu0.54`, turns and stops.
- The gains equal the plain `R2/42` row's.
- The record key holds the floor by value in `key.row.floor` and the new file's sha256 in `key.row_sources`.
- Its controller name (`R2~c1.25_mu0.9`) is in neither the panels' verdict rows (`panels.py:82`) nor the
  attributed controllers (`bar.py:1603`). So `summary.json` gets no verdict cell and no attribution for it.
- The null floor `h_mu0.9` changes none of the five arrays. `rfc_mu0.95` gives the same five arrays as
  `plant.build(body, "robot_floor_carpet")` followed by `set_friction(0.95)`. After every block the arrays equal
  the compiled model's.
- Not checked, because it needs a settle or an episode: that the settle reproduces Stage 1's settled pose, and
  that an episode runs.

**Files touched, and whether each is in a record's key.**

| File | New or edited | In an existing record's key? |
|---|---|---|
| `designlab/stage2/floors.py` | new | No. Its sha256 enters its own records' `row_sources`. |
| `tests/test_stage2_floors.py` | new | No |
| `stage2/runs/2026-11-16-75cm-addon/run_addon.sh` | new | No |
| the add-on's table script | new | No |
| `bar.py`, `readings.py`, `setio.py`, `speedfit.py`, every `designlab/stage1/*.py`, `designlab/carpet.py` | not edited | (they are the keys) |

**The bit-identity argument.** No file that enters a bar key, a panels key, the tuner's or the speed fit's is
edited. So nothing regenerates and nothing needs to. The checks:

1. `bar.bar_sources()` and `panels.panel_sources()` print the two hashes of section 1 after the build.
2. `python -m designlab.stage2.regress check` still passes: its four hashes are Stage 1's and are untouched.
3. The key of a registered record recomputed after the build still equals the stored key (my check of section 1).
4. Unit B0 (rule B-3): the null floor must reproduce a registered record's 20 episodes, trace sha256 included.
   A trace file holds arrays only (`bar.py:1473-1503`); the row's name is in the trace's `path`, not in the file.

**The tests that would prove the new file.**

1. Names and wiring: the row builds, round-trips through its spec, reads `gated: false`, refuses the conditions it
   must refuse, and carries the floor and the file's hash in its key.
2. Model identity: `floor_on(p, "rfc_mu0.95")` gives the same five arrays and, after its settle, the same settled
   pose bit for bit as `plant.build(body, "robot_floor_carpet")` with `set_friction(0.95)`.
3. Exact undo: after `floor_on` the five arrays, the settled pose and the settle record equal the saved ones.
   Then a committed Stage 1 episode, run in the same process, reproduces its committed trace sha256. That is a
   smoke on Stage 1's own run directory, as rule R4 allows.
4. The settle under a delay wrapper equals the settle without one, bit for bit.
5. A condition built on the hard-floor plant equals one built on a plant made on the floor, field for field
   (`omega`, joint limits, mass, track). I expect so: the home rotation comes from the keyframe
   (`plant.py:582`) and the frequency from kinematics (`control.py:944-950`). It must be tested, not assumed.
6. `floors check`: every floor settles; the settle's pad sink is printed beside the closed-form sink.
7. The command: with synthetic episodes, the ledger shows the task number and the cap given, and a unit over the
   cap ends with exit 3.

**The other way, and why not.** Edit `bar.py`: a `--floors FILE` switch, a condition subclass with the floor's
values, a plant path in `BarContext`. The records would be cleaner: the condition itself would name the floor.
But `bar.py` is hashed into every bar key. That makes a new code state. S2-17 then asks for the baseline rows to
be run again into a new directory: the 18 hard-floor walk records at 0, 2 and 4 ms, 0.29 core-hours by the
committed rates, a third of B's budget. And item C's panels would sit at another code state than the stand-in's
records they are set beside. I advise the new file.

### 3.4 The floors: values and reasons

**Two knobs, each with one meaning.**

- **Softer: how far a pad sinks with the whole weight on one foot.** Set by a linear spring per pad contact
  (`carpet.soft_floor`, the study's own law for its piles, decision row A16): impedance 0.5, damping ratio 1.0,
  time constant from the stiffness. The sink is then a plain number in mm that can be set against the pile height.
  `condim` stays 3 as compiled, so a "softer only" floor differs from the hard floor in the contact law alone.
- **Grippier: the sliding friction μ**, set as `_mu0.54` sets it.

**The grid** (planned figures; my arithmetic on the compiled model, no step taken):

| Floor | Contact | Sink on one foot (mm) | Sink on two feet (mm) | Stiffness under the pads (kPa/mm) | `solref` time constant (s) | μ | Why this value |
|---|---|---|---|---|---|---|---|
| (registered) `walk/v0.4` | as compiled | 0.22 | 0.11 | - | 0.015 | 0.9 | the stiff end; records exist, not run again |
| (registered) `walk/v0.4_mu0.54` | as compiled | 0.22 | 0.11 | - | 0.015 | 0.54 | the same, slippery; not run again |
| `h_mu1.25` | as compiled | 0.22 | 0.11 | - | 0.015 | 1.25 | grippier only |
| `c0.6_mu0.9` | linear | 0.60 | 0.30 | 68.6 | 0.0114 | 0.9 | softer only; half of the next |
| `c1.25_mu0.9` | linear | 1.25 | 0.625 | 32.9 | 0.0164 | 0.9 | softer only; the stiffness of the study's low pile, which P36 calls softer than the floor meant |
| `c2.4_mu0.9` | linear | 2.40 | 1.20 | 17.1 | 0.0228 | 0.9 | softer only; the pad sinks the whole height of the thinnest pile in the lookup. A bound, not a guess. |
| `c1.25_mu1.25` | linear | 1.25 | 0.625 | 32.9 | 0.0164 | 1.25 | softer and grippier: the owner's words |
| `c1.25_mu0.54` | linear | 1.25 | 0.625 | 32.9 | 0.0164 | 0.54 | softer and slippery |
| `rfc_mu0.95` | Stage 1's `robot_floor_carpet` | 2.06 | 1.59 | (not linear) | 0.024, ratio 1.2 | 0.95 | the tie to Stage 1; a second contact law at about the same sink as `c2.4` |
| `c4.4_mu0.9` (only by rule B-4) | linear | 4.40 | 2.20 | 9.4 | 0.0308 | 0.9 | the thickest pile's whole height |
| `h_mu0.9` (unit B0 only) | as compiled | 0.22 | 0.11 | - | 0.015 | 0.9 | the null floor, for the identity check |

- **The friction values.** 0.9 is the hard floor as compiled, which has no source. 0.54 and 1.25 are the two ends
  of the band the study's 7 Nov research file gives for rubber-type soles on carpet
  (`designlab/data/research/carpet_2026-11-07.json:258`, Hirai and Gunji 2001). The 15 Nov lookup could not open
  that paper and found no number for thin loop pile. So for this carpet every friction value here is without a
  source, and is labelled so.
- **The sink values.** None is sourced for this carpet. 2.4 and 4.4 mm are geometric bounds from secondary
  figures. 1.25 mm borrows the stiffness of a different, softer carpet.
- **What "softer" also changes.** In this contact model the time constant grows with the softness, so a softer
  floor also answers more slowly. Carpet damping has no source (A16). Only the static sink is set on purpose.

**Delays:** 0, 2 and 4 ms (P37: "at 0 to 4 ms"). 2 to 4 ms is the design estimate for the baseline wiring
(`STAGE2_75_REPORT.md` section 1), so 0 and 4 ms come first and 2 ms last.

**Rows:** per floor `R2~<floor>/42` and `R2~<floor>/43` at each delay, and `R0~<floor>` at 0 ms. `R0` tells whether
the walk itself stays up on that floor without balance feedback. On the hard floor it does, and the feedback is
what fails with delay. Stage 1 never ran `R0` on a carpet.

**Condition:** `walk/v0.4` only: start from standing, one slow stride, the change of speed, then 0.40 m/s. The start
from standing is where Stage 1 fell on its carpet floor, and it is in every episode.

### 3.5 The rules, ready to paste

> **B. The carpet range (P37, answer 5; P36).** Added on 16 Nov 2026 after the results of the 75 cm panels and
> labelled so. It gives no verdict. Every floor value in it is without a source for the carpet meant.
>
> **B-1. What is run.** Set `V0`, body `75-M-H-10`, block `seeds3042`, 20 episodes per record, condition
> `walk/v0.4`. Floors: the seven of the table [paste 3.4's table], each defined in `designlab/stage2/floors.py` and
> written with its resolved contact values into `stage2/runs/2026-11-16-75cm-addon/B/floors_check.json` before the
> first unit. Rows per floor: `R2~<floor>/42@<ms>ms` and `R2~<floor>/43@<ms>ms` at 0, 2 and 4 ms, and
> `R0~<floor>@0ms`. The gains are those of `stage1/runs/2026-11-08/75-M-H-10/tune.json`, unchanged. The hard
> floor's cells at μ 0.9 and 0.54 are the registered records and are not run again.
>
> **B-2. Order and units.** One unit is one floor at one delay. Floors in this order: `c1.25_mu0.9`,
> `c0.6_mu0.9`, `c2.4_mu0.9`, `h_mu1.25`, `c1.25_mu1.25`, `c1.25_mu0.54`, `rfc_mu0.95`. Tiers in this order: B0
> (the identity check); B1: 0 ms on every floor (three rows); B2: 4 ms (two rows); B3: 2 ms (two rows); B4: the
> conditional floor. Names: `B1-<floor>-0ms` and alike. Run directory `stage2/runs/2026-11-16-75cm-addon/B/`,
> task [5], rates from the committed `stage2/runs/2026-11-15-75cm/COST.md`.
>
> **B-3. The identity check, unit B0.** `R2~h_mu0.9/42@4ms` on `walk/v0.4`: the hard floor as compiled, through
> the floor row. Each of its 20 episode records must equal the episode of the same seed in the registered record
> `stage2/runs/2026-11-15-75cm/V0/75-M-H-10/seeds3042/eval/R2/42@4ms/walk/v0.4.json` on every key but `wall_s` and
> the trace's `path` (which holds the row's name); the trace's `sha256` must be equal. If one differs, nothing
> else of this section runs. Its record is a control: it is kept, labelled "identity check", and is no result.
> Its outcome is known beforehand (20 of 20 up, 0 meet every limit, 16 but for A1), so it is no early look.
>
> **B-4. Two rules that depend on a result, fixed here.** (a) A floor on which, at 0 ms, neither fit has more than
> 3 of 20 episodes up is not run at 2 or 4 ms. Its cells read "not run: down at 0 ms". (b) `c4.4_mu0.9` is run, at
> 0 ms (three rows) and then 4 ms (two rows), only if `c2.4_mu0.9` has at least 17 of 20 up under both fits at
> 0 ms and the budget fits. Otherwise it reads "not run".
>
> **B-5. A floor the robot cannot stand on.** If the zero-command stance does not settle on a floor
> (`floors_check.json`), that floor is "not run" with that reason and is not replaced (Stage 1's `NOT_RUN_RULE`).
>
> **B-6. Budget and stop rule.** 1.25 core-hours in `stage2/runs/2026-11-16-75cm-addon/B/budget.json`. A unit
> starts only if it fits; the first that does not is cut with every unit after it; a started unit finishes. An
> episode that failed to run is run again once (C8). After the first record exists, no floor, value, delay or seed
> is added or changed except as a new, labelled addition.
>
> **B-7. What is judged.** Nothing. No record of this section is gated, and none gives a verdict, whatever its
> counts. The words `shown`, `provisional` and "passes" are not used for it.
>
> **B-8. What is reported.** Per floor, delay and row: episodes that stayed up, that met every limit, and that met
> every limit but for A1, each of 20; where the falls come; the failed terms with their counts. Per floor: the
> closed-form sink (planned) and the settle's sink (simulated, zero command), printed apart. Reading words: a
> record is **up** with at least 17 of 20 up, **down** with at most 3, else **mixed**; a floor at a delay is up or
> down only when both fits are. The range sentence is built from those words alone: on which floors the walk is up
> at 0, 2 and 4 ms, on which it is down, and the largest sink and the friction values at which it is still up. It
> is headed "what would have to be true of the carpet for this walk, under this controller, to hold".
>
> **B-9. The limits are the hard floor's.** They are printed and not adapted. The toe clearance is read above the
> floor plane, and the stance foot stands lower by the sink, so the toe clearance is expected to drop by about the
> sink. Stage 1's own carpet toe rule is not applied, and Stage 1's carpet swing reference is not used.
>
> **B-10. Expectations, from results already known.** (1) `rfc_mu0.95`: down at 0 ms under both fits, with the
> falls in the start from standing, as in Stage 1 (F26; 40 episodes at μ 0.54, 3 starts at 0.95). A record that is
> up contradicts Stage 1 and is reported as a contradiction. (2) `h_mu1.25`: up at 0, 2 and 4 ms, since the two
> registered floors (μ 0.9 and 0.54) are both 20 of 20 there. (3) Meeting every limit but for A1 at 4 ms: fewer
> than on the hard floor (11 to 16 of 20 there), because the hard floor's toe clearance at 4 ms is 20.2 to
> 20.7 mm against a 20.07 mm limit and any sink takes from it. (4) The linear floors and every `R0` row: no
> expectation; no record exists.
>
> **B-11. Labels.** Each record carries the floor in `controller.floor` and the label "carpet range (P37): floor
> values without a source; reported, no verdict". Its `definition` names the hard floor, because the condition is
> the stock walk item: every table prints the floor from the row. Every table and sentence carries: "simulation;
> floor values without a source; one contact law and Stage 1's surrogate; stand-in controller never fitted on a
> soft floor; no verdict".
>
> **B-12. Not run.** Turns, pushes, stops, the slow walk and the speed ladder on these floors; delays above 4 ms;
> the probe piles `low`, `medium`, `high` (residential carpet, not the floor meant, P36); the sets `F25F9` and
> `F25F9cl`; the block `seeds5042`; fall attribution; any floor fitted to a specimen (none exists).

### 3.6 Cost

By the committed rates (0.1574 and 0.0954 wall seconds per simulated second), 21.078 s per episode, 20 episodes:
one `R2` record is 0.0184 core-hours and one `R0` record 0.0112.

| Part | Records | Episodes | Core-hours (estimate) |
|---|---|---|---|
| B0, the identity check | 1 | 20 | 0.018 |
| one floor: 0 ms (three rows), 4 ms and 2 ms (two rows each) | 7 | 140 | 0.122 |
| B1 to B3, seven floors | 49 | 980 | 0.851 |
| B0 to B3 | 50 | 1,000 | **0.87** |
| B4, the conditional floor (0 and 4 ms) | 5 | 100 | 0.085 |
| everything | 55 | 1,100 | **0.95** |

- The runner will print larger estimates for the `R0~` rows: it takes the largest measured rate for a controller
  it has no rate for (`bar.py:1892-1897`). By its own sum: 0.92, or 1.01 with B4.
- Full-length units used 1.10 to 1.42 times their estimate in the registered ledger. A floor that fells the robot
  costs less, and rule B-4(a) skips its later cells. The cap of 1.25 holds the plain grid at 1.42 times
  (0.87 × 1.42 = 1.24). If the lead wants 1.0 as the cap, the pre-registered order cuts the last 2 ms units first.
- The soft floors should cost about the same per step as the hard floor. Stage 1's zero-command step rate on
  `robot_floor_carpet` is 0.74 of the hard floor's (`stage1/runs/2026-11-08/75-M-H-10/plant.json`), but physics is
  about an eighth of an `R2` episode's time (0.02 of 0.157 s per simulated second). Design estimate: under 5 %
  more.
- **Wall time at 3 processes** (design estimate): 17 to 25 minutes of compute for B0 to B3, plus about 22 commands
  at 5 to 25 s each: about 25 to 30 minutes.
- **Build and review** are not in these figures: the new file, its tests, the `check` record and an independent
  review come before the first unit.

### 3.7 Risks, and what guards against each

| Risk | Guard |
|---|---|
| Unsourced floors are read as carpet results. | P37's wording in every label; "range", never "carpet result"; no verdict words; the bound and the borrowed stiffness are named as such. |
| One contact law decides the answer. | `rfc_mu0.95` is a second law at about the same sink as `c2.4`. If the two disagree, the report says the range depends on the law. |
| Softer also means slower here, and damping has no source. | Said in the rules' table text (3.4). A16's caveat is quoted: times to a fall are not carpet results. |
| The model floor is uniform. A real loop pile has direction, seams and can catch a claw. Only the pads collide. | Listed under what the range does not show. |
| The toe limit fails by geometry and is read as "the walk fails on carpet". | Rule B-9. The headline is staying up. |
| The record's `definition` says "hard floor". | Rule B-11; the floor is in the row's name, the path and `controller.floor`; the table script prints from the row. |
| The in-place swap leaves the plant changed for the next episode. | One floor per command, so a process sees one floor. Tests 2 and 3. Unit B0. |
| A condition built on the hard plant is stale on the soft floor. | Test 5. |
| A floor does not settle and episodes then error out. | `floors check` before the rules are committed; rule B-5. |
| The stand-in was never fitted on a soft floor, so a fall says little about carpet. | The heading of B-8 names the controller. The `R0` row at 0 ms separates the walk from the feedback. A16 already says this of Stage 1's carpet falls. |
| Every soft floor falls in the start, and the grid then repeats Stage 1. | The sink ladder (0.6, 1.25, 2.4 mm) is there to find where the start begins to fail. If even 0.6 mm is down, that is the finding, and it is reported as about the start. |
| Rules B-4(a) and (b) depend on results. | They are fixed before any run and written into the rules. |
| The new file drifts into a key module during review. | The file list of 3.3; the two hashes are checked after the build. |

## 4. Points for the lead

1. **Order of the three items.** A first, at today's code. Then B's and C's builds, each proving the two key hashes
   unchanged. If C must edit a key module, A must already be done.
2. **12 ms, 18 ms and the slow walk at 8 ms** (2.5): adopt or not. They are last in the order either way.
3. **A's ledger slot.** `--task 3` with the chain's cap guard (no new code), or wait for B's command and use a true
   task number and cap.
4. **B's task number and cap.** I wrote task 5 and 1.25 core-hours. The lead's guess was "about 1 core-hour": the
   estimate is 0.87, and 1.25 is the estimate times the worst ratio seen.
5. **The anchor's friction.** I chose 0.95, the surrogate's own. Stage 1's 40 episodes ran at 0.54. Either keeps
   the expectation; 0.54 would match the larger record.
6. **One record directory or two.** I used `.../2026-11-16-75cm-addon/A/` and `.../B/`, each with its own ledger.
7. **`floors check` settles a standing robot.** It is no episode, but it is physics. I did not run it. The rules
   should say it runs before they are committed, like Stage 1's plant record.

## 5. What I ran (nothing simulated)

- `bar dry-run --set V0 --body 75-M-H-10 --rows R2/42@<d>ms,R2/43@<d>ms,R0@<d>ms --conditions @walk,walk/nominal
  --cost stage2/runs/2026-11-15-75cm/COST.md --out <scratch>` for d = 14, 16, 20: 180 episodes, 0.937 simulated
  hours, 0.128 core-hours each. Nothing was written.
- The same dry run with the scratch prototype as `--row-module` and three floor rows: listed, 60 episodes.
- Calls of `bar.parse_row`, `bar.rows`, `bar._check_rule`, `bar._check_out`, `bar.Budget`, `bar.read_cost`,
  `bar.conditions`, `bar.choose`, `bar.record_key`, `bar.bar_sources`, `panels.panel_sources`, `setio.read_inputs`,
  `BarPlanner.resolve`.
- `plant.build("75-M-H-10", "hard", settle=False)` and the closed-form sink arithmetic of `carpet.static_depth` and
  `carpet.soft_floor`. No `mj_step`.
- Reads of the committed records, `budget.json`, `COST.md`, `plant.json` and the Stage 1 ledger.
- The prototype is at
  `/tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/addon/proto/floors_proto.py`.
  It is a sketch for the builder. It has never run an episode or a settle.
