# Add-on to the 75 cm panels, item C: the refit with delay (design)

16 Nov 2026. Body `75-M-H-10`, set `V0`. **Simulation only. This is a design: nothing in it was run as an
episode, and it changes no file in the repository.** Paths are relative to `hardware/trex_form_study/`.

Four kinds of statement are kept apart below. *Simulated*: counts read from committed records. *Planned*: a
reference's own length or timing, no physics. *Design estimate*: my arithmetic from committed rates. *Measured on
hardware*: there is none.

## 0. Short answer

1. **Today's code cannot run the fit.** Stage 1b's speed fit reads 19 reference files; 17 of them do not exist in
   `V0` (my command, section 2). It also draws its delays from the seed alone, which rule S2-18 forbids for this
   kind of fit. No Stage 2 fit module exists (`designlab/stage2/speedfit.py`, which my task names, does not exist;
   the file in the bar's record key is Stage 1's `designlab/stage1/speedfit.py`).
2. **Today's code can judge a fitted controller, with two small additions outside every record key.** The bar
   runner takes any registered row through `--row-module`. It refuses a new task number on its command line
   (`--task` takes 3 or 4), but its function `bar.evaluate` already takes any task number with a cap.
3. **Smallest addition:** one new module `designlab/stage2/r2lfit.py` (the fit, its selection, the row), its test
   file, and a driver of about 40 lines that calls `bar.evaluate` with the add-on's task number and cap. No
   existing file changes. `bar.py`, `readings.py`, `setio.py`, `speedfit.py` and every Stage 1 module stay byte for
   byte.
4. **The fit:** Stage 1b's CMA-ES fit (13 gains, population 12, step size 0.1, 360 evaluations) on five bar
   episodes per evaluation, each from standing. Delays 0, 2, 4, 6, 8, 10 ms in a fixed rotation. Two fits, seeds 42
   and 43. Selection on 10 ms alone; the other delays reported.
5. **Judging:** rows `R2L-V0/42@<ms>ms` and `R2L-V0/43@<ms>ms` on the bar's 16 conditions at 10 ms (judged), then
   0, 4 and 6 ms (reported). Pass rule: the bar's, unchanged.
6. **Honest expectation:** the bar will not pass at 10 ms whatever the refit does. With no delay at all the
   stand-in passes only the stop (2 of 16 conditions in the every-limit reading, 4 of 16 but for A1; simulated,
   committed records). What the refit can change is whether the robot stays up at 10 ms. That is the pre-registered
   reading of "cured" (rule CJ5).
7. **Cost (design estimate):** 6.8 to 9.7 core-hours in all; caps 6.4 (fits and selection) and 4.5 (the bar);
   2.3 to 3.2 hours of machine time at 3 processes, plus the build and its reviews.

## 1. What I read and what I ran

Read: `stage2/design/WP_ADDON_75.md`; `STAGE1B_PLAN.md` (all); `STAGE2_PLAN.md` sections 4 (B.2), 5, 6 and 9 (S2-9,
S2-16, S2-17, S2-18); `STAGE2_75_PLAN.md` sections B, H, R; `STAGE2_75_REPORT.md` sections 1 and 3 to 7;
`DECISION_LOG.md` P30, P32, P36, P37, F26, F27, A17; `LESSONS_LEARNED.md` sections 2.3 and 3 (lessons 1, 2, 6, 7,
12, 17, 19); `designlab/stage1/speedfit.py`, `tune.py`, `panels.py` (rows); `designlab/stage2/bar.py`;
`stage1/runs/2026-11-10-speed/` (records, `STAGE1B_SPEED.md` notes N1, N2, N10, the run script);
`stage2/runs/2026-11-15-75cm/` (`COST.md`, `budget.json`, V0's records); the paused round C design of the same kind
of fit (`stage2/design/ROUND_C_DESIGN.md` D21 and 3.8, `PREREG_TRACKB_draft.md` section Q, `WPC_CF.md`). No code
for that design exists in this checkout or in `/home/user/mesozoic-wt` (my `find`: no `r2lfit*` file).

Ran (nothing simulated, nothing written in the repository; `git status --porcelain` gave 0 lines afterwards):

| Command | What it showed |
|---|---|
| `python -m designlab.stage2.bar dry-run --set V0 --body 75-M-H-10 --rows R2L-V0/42@10ms,R2L-V0/43@10ms --cost stage2/runs/2026-11-15-75cm/COST.md --out <scratch>` | The dry path takes the new row names. 640 episodes, 3.608 simulated hours, 0.568 core-hours for the two rows on the 16 gated conditions at one delay |
| the same with `--conditions walk/nominal` | 40 episodes of 14.06 s, 0.025 core-hours |
| the same with rows at 14, 16, 20 ms and `--conditions walk/v0.4,walk/v0.4_mu0.54,walk/nominal` | accepted; 360 episodes, 0.295 core-hours |
| the same with `R2L-V0/42@22ms` | refused: "a delay of whole 2 ms physics steps, 0-20 ms, not 22" |
| `bar run ... --task 5` | refused by the argument parser: "invalid choice: '5' (choose from '3', '4')" |
| `bar run --set V0 --rows R2L-V0/42@10ms` with no row module | refused before any episode: "row 'R2L-V0/42': choose from R0, R1, R2/42, R2/43 or a registered controller's rows (R2@0ms, R2@10ms, R2s; register_rows)" |
| `python -m designlab.stage1 speedfit --help`; `speedfit.mix_files()` against both directories | Stage 1b's fit accepts `--body 75-M-H-10`; its 19 input files are all in `stage1/runs/2026-11-08/75-M-H-10/`; 17 are missing in `stage2/sets/V0/75-M-H-10/` |
| `speedfit.start_record("75-M-H-10")` (reads `tune.json` only) | the start is CMA-ES seed 43's selection `seed43_gen40`; `tune.json` sha256 `d451e7c58039…`, the value `R2/43`'s bar records carry |
| `sha256sum` of the four key files | `bar.py 21ac6393a119`, `readings.py e1ac0446692c`, `setio.py f25d8616e5e7`, `speedfit.py f502ec7b2557`: equal to the hashes inside the committed record `stage2/runs/2026-11-15-75cm/V0/75-M-H-10/seeds3042/eval/R2/43@10ms/walk/v0.4.json` |

Every command that imports the study's code ran through `tools/lowprio.sh`, one process at a time, one thread,
with `PYTHONDONTWRITEBYTECODE=1`. The scripts that only read committed JSON used the standard library, also
through `tools/lowprio.sh`. Hard limits: kept, with one small exception named in section 9.

## 2. What exists today, and what does not

### 2.1 How Stage 1 and Stage 1b fitted the tracker

| | Stage 1 (`tune.py`) | Stage 1b (`designlab/stage1/speedfit.py`) |
|---|---|---|
| What is fitted | R2, 13 of 14 gains; `capture_margin` held at 0.0 (`tune.py:130-136`) | the same |
| Search | CMA-ES, population 12, sigma0 0.1 of each gain's range (`tune.py:108-109`) | the same (`STAGE1B_PLAN.md` section 6, decision 1) |
| Evaluations, seeds | 720; seeds 42 and 43 (`tune.py:104-105`) | 360; seed 42 only (`speedfit.py:93-94`) |
| Start | the tuning start (decision 16) | the body's Stage 1 G5 selection (`speedfit.py:597`) |
| Mix per evaluation | nominal walk with start and stop, 0.35 rad/s walking turn, 0.5 rad/s turn in place; no delay | those three, plus the brisk and Froude 0.05 walks; each episode at a delay of 0 to 5 steps of 2 ms **drawn from the episode seed** (`speedfit.py:159-167, 282-285`) |
| Episodes | generation g on training episode `1000 + perm[g]`; every member on it (`tune.py:1173`) | the same |
| Objective | J = S + P. S: per fallen episode 1 + the share not survived. P = 0.5 T + 0.5 L, in [0, 1] (`tune.py:687-702`) | the same over five episodes |
| Selection | start and the mean after each third of the generations, on validation episodes 2000 to 2019; fewest falls, then mean S, then mean P (`tune.py:1384-1391`) | the same, **pooled over all drawn delays** (`speedfit.py:218-229`) |

The stand-in's two rows are Stage 1's selections for `75-M-H-10`: `seed42_gen60` and `seed43_gen40`
(`stage1/runs/2026-11-08/75-M-H-10/tune.json`; my command). The body-level selection, which Stage 1b's rule starts
from, is seed 43's.

### 2.2 What Stage 1b's refit showed, and what it did not (F27)

Simulated, three 60 cm bodies, one fit each (`DECISION_LOG.md` F27):

- **Showed:** at 10 ms every walk and walking turn fell, 0 of 20 but one, fitted or not. Of the speed walks, 116 of
  120 seed-matched pairs fell before their faster gait began. At 0 ms the refit raised Froude 0.05 survival on both
  Tapered bodies (1 to 16, 2 to 17 of 20) and made trades elsewhere.
- **Did not show** whether a fit can find gains for 10 ms, for four reasons that are in its own records:
  1. *The delays each kind met were chance.* In the fit of `60-N-T-10` the walking turn met 10 ms in 2 of 30
     generations, the brisk walk in 3, the nominal walk in 7 (my count from `history[*].latency_steps` in
     `stage1/runs/2026-11-10-speed/60-N-T-10/speedfit/42/cma.json`). All three bodies drew the same delays
     (`STAGE1B_SPEED.md` N1).
  2. *The selection hardly looked at 10 ms.* Of each kind's 20 validation episodes, 2 to 4 ran at 10 ms: 16 of 100
     in all (my count from `speedg5.json`, `conditions.*.by_latency_steps`). The selected gains kept 0 of 3 walking
     turns and 0 of 2 brisk walks up at 10 ms, as the start did.
  3. *One fit per body.* Stage 1's two fits of one body differed by 15 to 17 of 20 at the brisk walk
     (`LESSONS_LEARNED.md` section 2.3).
  4. *The step size never grew.* It stayed between 0.071 and 0.095 on the Tapered body (N1): a local search
     about the start.
- **Did not cover this body.** The 75 cm body was not in Stage 1b (`STAGE1B_PLAN.md` section 2).

### 2.3 What can run on `V0` today

| Need | Exists today? | Evidence |
|---|---|---|
| A fit on V0's references | **No.** Stage 1b's evaluator loads fixed Stage 1 names (`refs/brisk.npz`, `transitions/stop.npz`, the 0.35 and 0.5 rad/s turn files, ...): 17 of its 19 files are not in V0. V0 holds `nominal`, `v0.3` to `v0.5`, `start`, `nominal_to_v0.4`, `stop2_v0.4`, `turn90_pi4_left`, ... | my command; `speedfit.py:207-211, 298-303, 407-449` |
| The bar's tasks in a fit (walk at 0.40 m/s from standing, quick turn, two-step stop, nudge) | **No.** No fit builds `bar.BarCondition` episodes or reads the bar's readings | no `R2L` in any `.py` file (my `grep`) |
| A fixed rotation of delays (S2-18) | **No.** Stage 1b draws k from `derived_seed(episode seed, slot, 3)` | `speedfit.py:159-167` |
| Stage 1b's fit on Stage 1's 75 cm references | Yes, as a command (`python -m designlab.stage1 speedfit --body 75-M-H-10 --inputs stage1/runs/2026-11-08 --out NEW`). **Not proposed:** it is not on V0, has no bar task, and breaks S2-18 | `--help`; section 1 |
| The delay model | Yes: `speedfit.servo_delay(plant, k)` | `speedfit.py:325-361` |
| One bar episode from gains, judged, without a trace | Yes: `bar.run_episode(row, ctx, cond, seed)` with `trace_root=None`, on a `bar.BarContext` | `bar.py:1524-1550, 967-1038` |
| CMA-ES, the worker pool, the score, the selection key, the start record | Yes, importable: `tune.CMA`, `tune.Pool`, `tune.score`, `tune.selection_key`, `tune.candidate_generations`, `tune.training_order`, `speedfit.start_record` | `tune.py:1075, 942, 687, 1389, 1384, 1173`; `speedfit.py:597` |
| A fitted controller as a bar row | Yes, by registration; see 2.4 | `panels.py:865-900`; `bar.py:921-932` |
| A new task number and budget for the bar | **Not on the command line** (`--task` choices are 3 and 4, `bar.py:140, 3468`). **Yes through the function:** `bar.evaluate(..., task=N, budget_cap_core_h=CAP)` builds `Budget(path, N, CAP)`, which accepts any N when a cap is given | `bar.py:2328, 2502, 1800-1804` |
| Delays 14, 16, 20 ms in a row name | Yes, up to 20 ms (`DELAY_MAX_MS`) | `bar.py:135, 853-857`; my dry run |
| A second seed block on V0 | **No**, by rule: "the controls are not run on seeds5042" (75 H4; the rule text in every record). The runner refuses `--passed-of` on a control set. It does not refuse a plain `--block seeds5042` run; I do not propose one | `bar.py:284-290, 2354-2356` |
| Fall attribution for a new controller | **No**: the bar attributes rows of controller `R2` only | `bar.py:1603, 2287` |

### 2.4 How `R2/42` and `R2/43` reach the runner today, and how a new row would

Today: `--rows R2/42@10ms` goes to `bar.rows` (`bar.py:921-932`). It splits the name at `@`, asks
`panels.tracker_rows(body, <stage1 run>/<body>, ("R2/42",))` for the inner row, and wraps it in `bar.Delayed` (the
servo delay). `tracker_rows` reads the gains with `panels.r2_gains`: the file `<stage1 run>/<body>/tune.json`, key
`seeds["42"].selected_gains` (`panels.py:677-687, 887-891`). The Stage 1 run directory is the argument
`--stage1-run`, default `stage1/runs/2026-11-08` (`bar.py:93, 3477`).

Any other controller: `--row-module M` imports module M before the rows are built (`bar.py:3479, 3485-3486`). M
calls `panels.register_rows(controller, factory)`; `tracker_rows` then calls `factory(body, run_dir, name)` for any
row whose name before the `/` is that controller (`panels.py:865-871, 892-893`). Stage 1b's rows `R2s` were added
this way (`speedfit.py:1210-1218`). Worker processes rebuild a row from its `spec()`, whose class path is the
row's own (`bar.py:2141`; `panels.py:855-858`).

Rejected shortcut: writing the new gains into a `tune.json`-shaped file and passing `--stage1-run`. It needs no
code, but the rows would be named `R2/42` and `R2/43`, the stand-in's names, in every record and table.

## 3. The smallest addition

### 3.1 Files

| File | New or changed | In a record's key? |
|---|---|---|
| `designlab/stage2/r2lfit.py` | **new** (about 600 lines; it imports `speedfit`, `tune`, `panels`, `bar` and edits none) | in no existing key. Its sha256 is written into its own records |
| `tests/test_stage2_r2lfit.py` | **new** | no |
| a driver, about 40 lines (for example `designlab/stage2/addonrun.py`, shared with items A and B if they need the same) | **new** | no |
| `stage2/runs/2026-11-16-75cm-addon/` | **new** run directory | - |

Must not change: `designlab/stage2/bar.py`, `readings.py`, `setio.py`; `designlab/stage1/speedfit.py` (these four
are the bar's key, `bar.py:1677-1685`); every other `designlab/stage1/*.py` (the panels', the tuner's and the speed
fit's source hashes follow imports from there, and `train.py` hashes the whole package); `designlab/*.py` at top
level (the panels' sources follow imports into it, `panels.py:1572-1583`). A new file under `designlab/stage2/`
enters none of these hashes: no hash is taken over that directory (my `grep` for `glob("*.py")`: only
`designlab/stage1/` and top-level `designlab/`).

Default off: nothing imports `r2lfit` unless a command names it (`--row-module designlab.stage2.r2lfit`, or its own
command line). Importing it does two things, in that process only:

- `panels.register_rows("R2L-V0", r2l_row)`;
- `panels.VERDICT_ROWS["R2L-V0"] = ("R2L-V0/42", "R2L-V0/43")`, so that `bar.summarize` and the report print the
  pair's verdict (`bar.py:2822`; `barreport.py:1486, 1780`). `VERDICT_ROWS["R2"]` is untouched. Stage 1b did the
  same kind of thing to `panels.ATTRIBUTED` (`speedfit.py:1216`).

### 3.2 What the module holds

```
python -m designlab.stage2.r2lfit {rotation|cost|fit|select} --body 75-M-H-10 --set V0 [--seed 42]
       --out stage2/runs/2026-11-16-75cm-addon/r2l [--workers 3] [--resume]

rotation(seed, generations=30) -> dict     # rule CF3's table; `rotation` prints and writes it; simulates nothing
validation_table() -> dict                 # rule CS2's table
class Evaluator                            # a tune.Pool worker (top-level factory _evaluator)
fit(body, set_name, seed, out, workers=1, resume=False, cap_core_h=2.4, max_generations=30) -> dict
select(body, set_name, seeds=(42, 43), out=..., workers=1, at_generation=None) -> dict
cost(body, set_name, out) -> dict          # one evaluation at the start's gains (smoke; rule CF9)
r2l_row(body, run_dir, name) -> panels.TrackerRow     # "R2L-V0/<seed>"
```

- **Evaluator.** One `bar.BarContext` on the frozen set (`setio.read_inputs(..., require_frozen=True)`), built
  once per worker. An evaluation is `(gains, index)`; the index points into the rotation table (fit) or the
  validation table (selection). For each of the five slots it builds
  `bar.Delayed(panels.TrackerRow(name, "R2", gains, ...), k)` and calls that row's own `episode`, then
  `bar.judge`: **the bar's own episode code path, on other seeds, with no trace written.** It then computes the
  objective's terms (rule CF5).
- **Row.** `r2l_row` reads `<run>/<set>/<body>/r2lg5.json`, where `<run>` is the module's constant
  `stage2/runs/2026-11-16-75cm-addon/r2l` or the environment variable `STAGE2_R2L_RUN` (tests and smoke; Stage
  1b's pattern, `speedfit.py:129, 1124-1128`). It refuses a seed without a selection, a selection that is not
  judged, another body. It returns a plain `panels.TrackerRow` in mode `R2` with the selected gains;
  `gains_info` holds the selection file's path and sha256, the fit seed, the candidate, the generation judged at,
  the set, and the start. `bar.rows` wraps it for `@<ms>ms`. Workers need no import of `r2lfit`: the row's class is
  `panels.TrackerRow`. The row's key therefore differs from `R2/43`'s only in the row's description.
- **Guards.** `fit` and `select` refuse: a set that is not frozen; a checkout that is dirty outside the add-on's
  run directory (`s2io.git_state(own_outputs=...)`); a resume on other sources, inputs, library versions, start or
  rotation table than the first session's; a second start over an existing run.
- **Driver.** Parses `bar run`'s arguments plus `--task N --cap CORE_H`, imports the row modules, calls
  `bar.evaluate(..., task=N, budget_cap_core_h=CAP, rates=bar.read_cost(COST)["rates"])`, returns the bar's exit
  codes (3: cut; 4: an episode failed to run). It also refuses a row `R2L-<set>/...` on another set than `<set>`.

### 3.3 Bit-identity argument

No existing file changes, so every existing record key, model and reference is untouched by construction. The
checks that prove it (all without physics unless marked):

1. The sha256 of `bar.py`, `readings.py`, `setio.py`, `speedfit.py`, and `bar.bar_sources()`,
   `panels.panel_sources()`, `speedfit.speed_sources()`, `tune.tuner_sources()`, `plant.sources_sha256()` are equal
   before and after `import designlab.stage2.r2lfit`.
2. With `r2lfit` imported, `key.sources` of a committed V0 record equals today's
   (`bar 45719d03…`, `panels c0027992…`, from `.../R2/43@10ms/walk/v0.4.json`), and that record's key recomputed
   today hashes to its `key_sha256`.
3. `python -m designlab.stage2.regress check` passes (it compares the four source hashes of
   `stage2/regress/regress.json` with today's; it runs no episode).
4. `bar.rows("75-M-H-10", ["R2/43@10ms"])[...].describe()` is equal with and without the import.
5. *(physics; after the rules are committed; listed as a check episode)* The evaluator, given `R2/43`'s gains,
   `walk/v0.4`, seed 3042, k = 5 and a trace directory, reproduces the committed episode: trace sha256 and episode
   record equal to `.../R2/43@10ms/walk/v0.4.json`, seed 3042, on every field but `wall_s`. This is a reproduction
   of a registered episode and adds no result.

### 3.4 Tests (`tests/test_stage2_r2lfit.py`; a stub evaluator that scores from a formula, no physics)

- Rotation: in every block of six generations every slot meets all six delays; left and right turns each meet all
  six within two blocks; each nudge direction meets all six within four blocks; 150 distinct episode seeds per fit,
  all in 1000 to 1999; none in 2000 to 2099, 3042 to 3061 or 5042 to 5061.
- `rotation.json` is written before generation 0 (`written_at` before `sessions[0].started_at`); a resume refuses
  a table that differs.
- Every member of a generation receives the same five episodes (seed, kind, direction, delay).
- Objective: every term in [0, 1]; a fallen episode scores 1 on every term; P at most 1; an evaluation with one
  more fall always scores worse; a failed evaluation scores 11 (2 per episode + 1).
- A killed fit resumes at its next generation with the same CMA-ES state; evaluations logged after the checkpoint
  are dropped (Stage 1b's rule).
- A fit cut by its cap ends at a multiple of six generations.
- Selection: ranked on the 10 ms cell alone (a stub candidate that is better at 0 ms and worse at 10 ms loses);
  the decisive cell runs before any reported cell and finishes when the ledger is forced past its reserve; a
  reported cell beyond the share reads "not run: budget".
- Row: after a stub selection, `bar.rows(body, ["R2L-V0/42@10ms"])` gives a `bar.Delayed` around a
  `panels.TrackerRow` in mode `R2` with the selected gains and the file's sha256 in `gains_info`; a changed file,
  a missing seed, an unjudged selection and another body are refused.
- The start: `speedfit.start_record("75-M-H-10")["gains"]` equals `panels.r2_gains(<stage1 run>/75-M-H-10, 43)`
  value for value.
- The bit-identity checks 1 to 4 of 3.3. Check 5 behind `STAGE2_SLOW=1`.

## 4. The rules, ready to paste into `STAGE2_75_ADDON_PLAN.md`

Tags: [plan] `STAGE2_PLAN.md`; [S1b] `STAGE1B_PLAN.md`; [75] `STAGE2_75_PLAN.md`; [RC] the reviewed round C text for
the same fit (`PREREG_TRACKB_draft.md` section Q); [NEW] this add-on; [differs: X] where it departs from X.
Task numbers are written `C1` (fits and selection) and `C2` (the bar); the lead gives them integers that do not
collide with items A and B.

### C. The refit with delay (owner's decision P37, item 1, second part)

**CF1. What is fitted.** [plan B.2; S1b; P37] Stage 1's tracker R2 on `75-M-H-10`: the 13 searched gains,
`capture_margin` held at 0.0. On the frozen set `V0` (freeze manifest `34a820e550b6…`), floor as compiled
(μ 0.9), commanded 5 % above (track played at 1.05). Two fits: CMA-ES seeds 42 and 43. [differs: plan B.2's "one
seed per body"; P37 asks for two.] The controller is named `R2L-V0`. It is the plan's R2L kind of fit on today's
walk; it is **not** the plan's R2L, which is fitted on the look set. Check: `r2lfit/<seed>/cma.json` `settings`.

**CF2. Mix.** [RC Q2; the brief] Every evaluation runs five bar episodes, slots j = 0 to 4: `walk/v0.4`,
`turn/q90_<side>`, `stop/two_step`, `nudge/<direction>` (0.25 × the bound), `walk/nominal`. Each is built and run
exactly as the bar builds and runs it (`bar.BarContext`, `panels.TrackerRow` in mode R2, `speedfit.servo_delay`).
So every episode starts from standing. Four of the five then take one slow stride and the speed-up to 0.40 m/s;
the fifth holds the slow walk itself. These are the three track segments in which 76 of the stand-in's 80 walk
episodes fell at 10 ms (`start`, `nominal`, `nominal_to_v0.4`). Planned lengths: 21.08, 19.62 (left) or 20.05
(right), 20.07, 20.38 and 14.06 simulated seconds (the `seconds` of V0's committed records). About 65 % of an
evaluation's planned time lies in those segments or in the slow walk (planned: the three lead-in segments last
12.65 s of reference time, 12.05 s at the 1.05 clock; 4 × 12.05 s + 14.06 s of 95.2 s). Check: `cma.json` `mix`.

**CF3. Delays and their fixed rotation.** [S2-9; S2-18; RC Q2] Delays are k = 0 to 5 physics steps (0 to 10 ms).
In generation g, with block B = g // 6: slot j runs at k = (g + j) mod 6; the turn is to the left for even B and to
the right for odd B; the nudge is forward, left, backward, right for B mod 4 = 0, 1, 2, 3. First block:

| g | `walk/v0.4` | turn | stop | nudge | `walk/nominal` |
|---|---|---|---|---|---|
| 0 | 0 ms | 2 | 4 | 6 | 8 |
| 1 | 2 | 4 | 6 | 8 | 10 |
| 2 | 4 | 6 | 8 | 10 | 0 |
| 3 | 6 | 8 | 10 | 0 | 2 |
| 4 | 8 | 10 | 0 | 2 | 4 |
| 5 | 10 | 0 | 2 | 4 | 6 |

Every generation holds five different delays. Within a block every kind meets every delay once. Over 30
generations each kind meets 10 ms in 5 generations: 60 episodes per kind, 300 of a fit's 1,800 episodes. Every
member of a generation runs the same five episodes. The table for 30 generations is written to
`r2lfit/<seed>/rotation.json` before generation 0. Check: `rotation.json` (`written_at` before
`sessions[0].started_at`); every line of `evals.jsonl` equals it.

**CF4. Episode seeds.** [NEW; differs: S1b, one training episode per generation with per-slot derived seeds]
Slot j of generation g runs bar episode seed `tune.training_order(fit seed)[5 g + j]`: 150 different seeds from
1000 to 1999 per fit. The bar derives the start, sensor-noise and push seeds from it as it does for a panel
episode (slot 0, `panels.episode_seeds`). Reason: the fit's episodes are then bar episodes to the letter, and the
five lead-ins of an evaluation differ. The panels' seeds (3042 to 3061, 5042 to 5061) are never used by the fit or
the selection. Check: `rotation.json`.

**CF5. Objective `stage2-r2l-v1`.** [S1b for the structure; RC Q3 for the item terms; NEW precision] Stage 1's
score (`tune.score`): J = S + P. S: for each fallen episode, 1 + the share of the episode not survived. P = 0.5 T +
0.5 L, with T the mean of all tracking terms and L the mean over the five kinds of (largest + mean) / 2 of the
kind's limit terms. Every term lies in [0, 1]; a fallen episode scores 1 on every term; so a fall always outweighs
every other term. The fitness adds the box repair. Terms, with `hinge(x, limit, scale) = min(1, max(0, x − limit)
/ scale)` (`tune.py:238-257`):

| Kind | Tracking terms | Limit terms |
|---|---|---|
| both walks | DCM error over the episode (scale 0.02 m); speed against the command and heading change over the last 10 s (Stage 1's three) | Stage 1's hinges over the last 10 s: roll and pitch RMS and peak, duty of each foot, flight, slip median and worst, the 10 s speed under target, drives' RMS, drives' peak speed, joint range. Toe clearance on the bar's own-swing reading: `hinge_below(value, 0.04 H, 0.04 H)`, 1 if the bar's toe term fails for another reason. Distance: `hinge_below(along, target, target)`. Line: `hinge(deviation, 0.25 m, 0.25 m)`. `walk/nominal` also has Stage 1's cadence, stride and peak ground reaction |
| turn | DCM error over the episode | the same window hinges over [command, command + 2.0 s + 3.0 s] without speed; own-swing toe; heading error at the judged instant and the largest in the hold, each `hinge(error, 10°, 10°)`; the joint range the bar judges; leg interference as the bar reads it (0 or 1) |
| stop | DCM error over the episode | touchdowns `hinge(n, 2, 2)` (1 if not stopped by the end); 1 Hz deceleration `hinge(a, 1.5, 1.5 m/s²)`; drives' RMS over the episode, peak speed, joint range |
| nudge | DCM error over the episode | recovery time `hinge(t, 3.0 s, 3.0 s)` (1 if not recovered); drives' RMS over the episode, peak speed, joint range |

The objective ranks gain vectors and is not the judge. Two places where it is stricter than the bar, on purpose,
because Stage 1's functions are used unchanged: the duty band is 0.55 to 0.65 for `walk/nominal` too (the bar
allows 0.72 below 0.30 m/s), and the drives' RMS term includes the tail and neck servos (the every-limit reading).
An evaluation that raises, or an episode the bar cannot read, scores as the worst case (J = 11) with its
traceback; an episode during which MuJoCo warned counts as a fall at its start (Stage 1b's `FAILURE_RULE`). Check:
`cma.json` `objective`.

**CF6. Search.** [S1b section 3 and decision 1] CMA-ES (`tune.CMA`), population 12, sigma0 0.1 of each gain's
range, 30 generations (360 evaluations) per fit. Both fits start from the body's Stage 1 G5 selection
(`speedfit.start_record`: `seed43_gen40`, the gains of row `R2/43`), recorded once in `start.json`. The fits
differ only in their seed (sampling and episode order). Check: `start.json`; `cma.json` `settings`.

**CF7. Budget and stop rule of a fit.** [plan section 6: "R2L ... each fit twice Stage 1b's 1.6"; RC Q5] Counted as
the sum of the evaluations' wall seconds in `r2lfit/<seed>/ledger.json`. A fit may use at most 2.4 core-hours. It
starts a block of six generations only if its spent time plus the block's estimate (planned simulated seconds ×
the committed rate; about 0.30 core-hours) fits; it ends at 30 generations or at the last whole block inside the
cap. There is no early stop on a good score. Both fits are judged at the same generation: the last whole block
both reached [S1b decision 7]. A fit that reached fewer than 12 generations (two blocks: both turn sides) is not
judged and is reported "cut". Fits run one after the other, seed 42 then 43, with 3 workers each, and nothing else
heavy on the machine. Check: `ledger.json`; `cma.json` `history`.

**CF8. Records.** [S1b; lesson 7] Under `stage2/runs/2026-11-16-75cm-addon/r2l/V0/75-M-H-10/`: `start.json`;
per fit `r2lfit/<seed>/rotation.json`, `cma.json` (settings, objective, rules, start, per session: commit, dirty
flag, library versions, the sha256 of `r2lfit.py`, of the bar's four files, of the panels' and the tuner's
sources, the set's freeze and every played file's sha256; per generation: its table row, sigma, the mean gains,
J min / median / max, falls by kind and by delay), `evals.jsonl` (one line per evaluation: the gains, J, S, P,
and per episode its kind, seed, delay, fall time, the segment it had reached, its failed bar terms and its
objective terms), `ledger.json`; then `validation.json`, `r2lg5.json`, `r2lg5_evals.jsonl`, `cost.json`. All are
text and are committed. `git check-ignore -v` is run on each new output before the first commit: my check on an
`evals.jsonl` path and on `budget.json` under the new directory printed nothing, and a `traces/` path is ignored
by `stage2/runs/.gitignore:5`. No trace is written by the fit or the selection.

**CF9. What may run before generation 0.** [75 R4's pattern] After these rules are committed and the build is
reviewed: (a) the reproduction of one registered episode (3.3, check 5); (b) the cost measurement: one evaluation
of generation 0's five episodes of seed 42 at the start's gains, of which only the wall seconds per simulated
second are kept. Both are listed in `stage2/runs/2026-11-16-75cm-addon/dev_episodes.json` before they run, are
smoke, and are reported nowhere as results. Any other smoke uses a stub evaluator or episodes cut to 3 s
(standing only), in a temporary directory.

**CS1. Candidates.** [S1b `SELECTION_RULE`] Per fit: the start, and the CMA-ES mean (projected into the bounds)
after each third of the generations judged (10, 20, 30 of a full run). One more candidate is evaluated and never
selected: the 13 searched gains at zero ("R2 with no feedback gain"; a diagnostic, [NEW]; it is not row `R0`,
which also drops the ankle cap and reads the true state).

**CS2. Validation episodes.** [S2-18; differs: S1b's 2000 to 2019] A validation evaluation i (0 to 19) is the five
kinds; kind j runs bar seed 2000 + 20 j + i (seeds 2000 to 2099, a block of 20 per kind). The turn is to the left
for even i, to the right for odd i; the nudge is forward, left, backward, right for i mod 4 = 0 to 3. Reason for
the five blocks: at one delay the bar gives every item of a seed the same lead-in (75 B7), so one block of 20
would hold only 20 different starts in 100 episodes. The table is written to `validation.json` before the first
generation.

**CS3. The decisive cell: 10 ms.** [S2-18] Every candidate runs the 20 validation evaluations at k = 5. Per fit
its four candidates are ranked by `tune.selection_key` on those 100 episodes: fewer falls, then the lower mean S,
then the lower mean P, the lower mean J, then the earlier candidate. The first is that fit's selection. This cell
runs first and always whole, even if the reserve is passed (the overrun is recorded; it is the only one allowed).
Check: `r2lg5.json` `seeds.<s>.selection`.

**CS4. The other delays: reported.** [S2-18; RC Q4] Then, as far as the share reaches: the start and each fit's
selection at k = 0, 1, 2, 3, 4, on validation evaluations 0 to 4 (25 episodes per cell). A cell not reached reads
"not run: budget". These cells select nothing. Per cell and kind the record gives the episodes up, and the episodes
meeting the bar's terms in both readings. Budget: 1.0 core-hour reserved for CS3, 0.6 for CS4. Task `C1` in all:
2 × 2.4 + 1.0 + 0.6 = 6.4 core-hours. Check: `r2lg5.json` `by_delay`.

**CJ1. Rows.** [75 B2; NEW names] `R2L-V0/42@<ms>ms` and `R2L-V0/43@<ms>ms`: the tracker in mode R2 with fit seed
42's or 43's selection, read from the committed `r2lg5.json` (path and sha256 in every record's
`controller.gains_info`), inside the bar's servo delay. The fit and selection records are committed before the
first bar unit. If a fit's selection is its start, its row is run at 10 ms only: each of its episodes must then
reproduce the registered `R2/43@10ms` episode's trace sha256 (a free check of the code state), and at the other
delays the registered `R2/43` records are cited for it.

**CJ2. What is run, in order.** [75 B1, B3, B6, H6] Set `V0`, block `seeds3042` (3042 to 3061), 20 episodes per
record, both rows, `--procs 3`, `--attribute none`. One unit is one delay; a unit starts only if task `C2`'s spent
time plus its estimate fits; the first unit that does not fit is cut with every unit after it.

| Tier | Conditions | Delay | Status of the reading |
|---|---|---|---|
| 1 | the 16 gated conditions and `walk/nominal` | 10 ms | **judged** (the gated 16); `walk/nominal` reported |
| 2 | the same | 0 ms | reported |
| 3 | the same | 4 ms | reported |
| 4 | the same | 6 ms | reported |
| 5 | `walk/v0.4`, `walk/v0.4_mu0.54`, `walk/nominal` | 8 ms, then 2 ms | reported [NEW: the neighbours of the judged delay] |
| 6 | the same three | 14, 16, 20 ms | reported; run only if item A ran these delays for the stand-in |

Task `C2`: 4.5 core-hours. Rates: the committed `COST.md` of the add-on's run directory (a row of a controller
without a measured rate takes the largest measured, `bar.rate_of`).

**CJ3. Pass rule: the bar's, unchanged.** [75 H1, H2, H3] A record passes a reading if 20 of 20 stay up and at
least 17 succeed. `R2L-V0` is `shown` on a condition at 10 ms if both rows pass, `provisional` if one; in the
but-for-A1 reading the words carry "but for A1". A cell at another delay reads "passes at <ms> ms" and is no
verdict. Reading 2 of the quick turn stays "not read: `a1_over`". The μ 0.54 conditions are a floor the fit never
saw.

**CJ4. How the new rows stand beside the stand-in's.** [75 R2; S2-17] The stand-in's registered records
(`stage2/runs/2026-11-15-75cm/V0/75-M-H-10/seeds3042/eval/R2/42@<ms>ms`, `R2/43@<ms>ms`, `R0@<ms>ms`) are cited,
not run again. Every table prints, per condition and delay, five rows: `R2/42`, `R2/43` (the fit's start),
`R2L-V0/42`, `R2L-V0/43`, `R0`. A new record is set beside a registered one only if their keys are equal in every
field but the row's (`row`, `row_sources`): same set freeze, plan, seconds, seeds, omega, bar and panel sources.
The report checks this per pair and prints any pair that fails it as "other code state". The comparison is paired
by seed (same start, noise and push seeds): per condition the table also gives the seeds up under both, under the
new row only, under the stand-in only. Paired counts are reported and give no verdict.

**CJ5. The reading of "cured" (falls at the judged delay).** [NEW; set before the fit] On the three walk records
of a row at 10 ms (`walk/v0.4`, `walk/v0.4_mu0.54`, `walk/nominal`): a row **stays up** if 20 of 20 stay up in each
of the three (the survival half of 75 H1).

- Both rows stay up: "the refit removes the 10 ms falls of the walk, in both fits".
- One row stays up: "in one fit of two" (provisional).
- Neither: **"the refit did not cure it"**. If both rows keep at least 17 of 20 up in each of the three records,
  the sentence adds "it reduced the falls"; the counts and the segment each fall came in are printed either way.
- A fit whose selection is its start counts as a row that does not stay up, with the label "the fit found nothing
  better than its start at 10 ms".

This reading is about falls only. It is not the bar's verdict and is never printed as `shown`.

**CJ6. Trades and the no-feedback reading.** [S1b section 4; NEW] At 0 ms the stand-in passes, under both rows:
the stop on both floors (both readings), the walk at 0.40 m/s on both floors but for A1, and `walk/nominal`. A
refit row that does not pass one of these at 0 ms is reported as a trade, not a success. Beside every 10 ms count
stands the `R0@10ms` count of the same condition. The report prints the selected gains beside the start's, each
also as a share of its bound range, and the zero-gain candidate's decisive cell.

**CJ7. Labels every record, table and sentence of this item carries.** "Simulation only; the 10 ms delay is
assumed, not measured." "Add-on run after the 75 cm panels' results (P37, 16 Nov 2026)." "Refitted on `V0`'s own
references with 0 to 10 ms of delay; fit seeds 1000 to 1999, selection seeds 2000 to 2099; the panels' seeds were
used by neither." "Selected on 10 ms alone." "Not replicated: `V0` is a control and is not run on the second seed
block (75 H4)." "Not attributed: the runner attributes rows of controller R2 only" (a failing pair's cell prints
"not attributed"). "Not the plan's R2L (look set, one seed)."

**CJ8. Not run in this item.** No fit on `F25F9` or `F25F9cl` (P37, item 3). No fit on μ 0.54, on a carpet floor,
or with a delay above 10 ms in its mix. No third fit. No refit of a 60 cm body. No second seed block, no
attribution. No speed ladder, turn-rate ladder, stronger nudges or one-step stop under the new rows. No rerun of
any stand-in row. No Stage 1b speed fit on Stage 1's 75 cm references. No policy (R3, R5, R6).

**CO. Order of work.** [the brief's standing rules; P37] (1) Item A's delay runs are done (P37 puts them first).
(2) These rules are reviewed independently and committed. (3) The module, its tests and the driver are built,
reviewed independently and committed; the bit-identity checks 1 to 4 pass. (4) `rotation.json` of both seeds and
`validation.json` are written and committed. (5) The two listed check episodes of CF9. (6) Fit 42, then fit 43.
(7) The selection: the decisive cell, then the reported cells. (8) The fit and selection records are committed.
(9) The bar's tiers 1 to 6, in order. (10) The tables, made by script from the records. No step starts before the
one before it is committed. A rule changed after step 5 has begun is labelled "changed after a result".

## 5. The expectation, stated before any run

Sources: the committed V0 records (`stage2/runs/2026-11-15-75cm/V0/75-M-H-10/seeds3042/eval/`; my tabulation,
which agrees with `STAGE2_75_REPORT.md` sections 5 to 7). All counts are simulated, of 20 episodes, as
up / every limit / but for A1.

| Condition (μ 0.9) | `R2/42`, `R2/43` at 0 ms | `R2/42`, `R2/43` at 10 ms | `R0` at 10 ms (no verdict) |
|---|---|---|---|
| `walk/v0.4` | 20/0/19, 20/0/19 | 0/0/0, 1/0/0 | 20/0/0 (duty 20, drives' RMS 20) |
| `walk/nominal` | 20/20/20, 20/20/20 | 2/0/0, 1/0/0 | not run |
| `turn/q90_left` | 20/not read/0, twice | 0, 0 up | 20 up, 0 succeed |
| `nudge/*` (four) | 19 to 20 up; 11 to 15 succeed | 0 to 1 up | 2 to 14 up |
| `stop/two_step` | 20/20/20, twice | 0/0/0, 1/0/0 | 20/20/20 |

At 10 ms 76 of the stand-in's 80 walk episodes fall: 29 in the start from standing, 12 in the slow stride, 35 in
the speed-up (`STAGE2_75_REPORT.md` section 5).

**What I expect, and how sure.**

1. *The bar does not pass at 10 ms, whatever the refit does.* With no delay at all the stand-in passes only the
   stop: 2 of 16 conditions in the every-limit reading, 4 of 16 but for A1. The tail servo fails the every-limit
   reading of the 0.40 m/s walk in 20 of 20 at 0 to 6 ms, and under `R0` at 10 ms; no balance gain changes that.
   The quick turn's every-limit reading is not read by rule. No nudge passes at 0 ms. So at 10 ms a `shown` cell is
   plausible only for the stop (2 conditions), and for the walk only but for A1 and only if toe clearance and duty
   hold, which I do not predict: toe clearance fell with delay under the stand-in (24.2 mm at 0 ms to 18.3 to
   18.9 mm at 6 ms against a 20.07 mm limit), and `R0` fails duty in 20 of 20.
2. *Fewer falls at 10 ms than the start: more likely than not, in at least one fit.* This body stays up at 10 ms
   with the balance feedback off (`R0`: 20 of 20 in the walk on both floors and in the stop, 19 to 20 in the
   turns). All 13 searched gains have a lower bound of 0 (`control.py:183-200`). So gains that keep it up at 10 ms
   plausibly exist toward low feedback. On the 60 cm bodies they did not: there `R0` falls in most walks (F26).
3. *Twenty of twenty in both fits: not predicted.* Against it, from this study's own log: F27 (the one refit with
   delay, on three 60 cm bodies, fell at 10 ms in every walk and walking turn but one); F26 (on this body the
   learned add-on trained with 0 to 20 ms of random delay fell in 3 of 3 slow-walk episodes at 10 ms); the start
   lies about 1.4 bound-ranges from the zero-gain corner (my arithmetic from `tune.json`) while sigma0 is 0.1 and
   Stage 1b's step size never grew; `R2` with zero gains is not `R0`; and nobody has explained why this walk falls
   at 10 ms yet stayed up at 20 ms in Stage 1 (three episodes; `LESSONS_LEARNED.md` section 2.3).
4. *A cure by low gains would cost push recovery.* On the grippy floor `R0` keeps 3 to 17 of 20 up after a push
   at 0 ms, where the stand-in keeps 19 or 20. Rule CJ6 exists to show this.

**What in this design differs from Stage 1b's refit.**

| | Stage 1b (F27) | This item |
|---|---|---|
| Body | three 60 cm bodies, where `R0` falls in most walks | `75-M-H-10`, where `R0` stays up at every delay tried (18 to 20 of 20 in the walk) |
| Delay per episode | drawn from the seed; a kind met 10 ms in 2 to 7 of 30 generations | fixed rotation; every kind meets 10 ms in 5 of 30 generations, every delay equally |
| Selection | pooled over the drawn delays; 16 of 100 episodes at 10 ms | on 10 ms alone, 100 episodes per candidate; the other delays reported |
| Fits | one per body | two |
| Episodes | Stage 1's walk with start and stop, two steady turns, two speed walks | the bar's own items; every one starts from standing; 150 different starts per fit |
| Readings in the objective | Stage 1's | Stage 1's, with the bar's own-swing toe clearance and the bar's item terms |
| The same | CMA-ES settings, step size, 360 evaluations, start at the body's Stage 1 selection, the score's structure, delays 0 to 10 ms | |

**What a "did not cure it" would and would not mean.** It would mean: this fit, as the plan defines it (delays
uniform on 0 to 10 ms, 360 evaluations from the Stage 1 gains, twice), did not find gains that keep the walk up at
10 ms. It would not mean that no gains exist: only one sixth of the fit's episodes run at 10 ms, and the search is
local. A fit on 8 to 10 ms alone, or a start at low gains, would be a new item with its own rules.

## 6. Cost (design estimates) and wall time

Rates, wall seconds per simulated second:

| Rate | Value | Source |
|---|---|---|
| committed bar rate, R2 rows, one process | 0.1574 | `stage2/runs/2026-11-15-75cm/COST.md` (table row R2; JSON `rates.R2.wall_per_sim_s`) |
| that rate × the ledger's median overrun | 0.1781 | `budget.json` of the same run: used / estimate of the 10 V0 units at 0 to 6 ms is 1.06 to 1.42, median 1.13 (my computation) |
| Stage 1b's own fits, three at once | 0.194 to 0.197 | `stage1/runs/2026-11-10-speed/<body>/speedfit/42/cma.json`: sums of `history[*].wall_s` over `sim_s` (5,818.5 s / 29,943.2 s; 5,729.6 / 29,436.5; 5,991.1 / 30,360.3; my computation). Stage 1b's fits cost 1.60, 1.57 and 1.65 core-hours and its selections 0.40 each (`STAGE1B_SPEED.md` N10) |
| committed rate × the ledger's largest overrun | 0.2230 | as the second row |

Planned simulated seconds: one evaluation 95.2 s (left turn) or 95.6 s (right); one fit of 30 generations
34,340 s; all figures below assume every episode runs its full length, so they are upper estimates while episodes
still fall.

| Part | Simulated s | Core-hours at 0.1574 / 0.1781 / 0.1973 / 0.2230 |
|---|---|---|
| one fit (360 evaluations) | 34,340 | 1.50 / 1.70 / 1.88 / 2.13 |
| the decisive cell (8 candidates × 100 episodes) | 15,269 | 0.67 / 0.76 / 0.84 / 0.95 |
| the reported cells (3 gain sets × 5 delays × 25 episodes) | 7,158 | 0.31 / 0.35 / 0.39 / 0.44 |
| **task C1: two fits and the selection** | 91,108 | **3.98 / 4.51 / 4.99 / 5.64** (cap 6.4; the plan's "about 3.2 a fit") |
| the bar, one delay: 16 gated conditions, two rows (640 episodes) | 12,989 | 0.568 (the runner's own dry run) / 0.64 / 0.71 / 0.80 |
| `walk/nominal`, one delay, two rows | 562 | 0.025 / 0.028 / 0.031 / 0.035 |
| tiers 1 to 4 (10, 0, 4, 6 ms) | 54,206 | 2.37 / 2.68 / 2.97 / 3.36 |
| tier 5 (three walks at 8 and 2 ms) | 4,497 | 0.20 / 0.22 / 0.25 / 0.28 |
| tier 6 (three walks at 14, 16, 20 ms) | 6,746 | 0.295 (dry run) / 0.33 / 0.37 / 0.42 |
| **task C2: the bar** | 65,449 | **2.86 / 3.24 / 3.59 / 4.05** (cap 4.5) |
| **item C in all** | 156,557 | **6.85 / 7.74 / 8.58 / 9.70** (caps 10.9) |

Wall time at 3 processes (the core-hours divided by 3, with the same four rates): the two fits one after the
other with 3 workers each, 1.0 to 1.4 h; the selection, 0.33 to 0.46 h; the bar, 0.95 to 1.35 h. **About 2.3 to
3.2 hours of machine time**, on a machine that runs nothing else heavy. The build, its tests, the independent
reviews and the commits between the steps are not in this figure.

Not counted: the cost measurement (one evaluation, about 15 to 21 wall seconds) and the reproduction of one
episode.

## 7. Risks, and what guards against each

| Risk | Guard in the design |
|---|---|
| **The "cure" is only the feedback turned down.** The fit may reach survival at 10 ms by moving toward no feedback, which `R0` already shows and which loses push recovery | CJ4 and CJ6: `R0` beside every count; the gains table as shares of their ranges; the zero-gain candidate; the nudges at 0 and 10 ms; the trade rule |
| **A pass at 10 ms that fails at its neighbours.** 10 ms is one period of the 100 Hz feedback, and this walk fell at 10 ms and stood at 20 ms in Stage 1 | The mix holds every delay from 0 to 10 ms equally; tiers 2 to 5 report 0, 2, 4, 6, 8 ms; tier 6 reports 14 to 20 ms if item A ran them; lesson 2.3: a 10 ms verdict is read beside its neighbours |
| **Selecting on 10 ms alone picks gains that are worse without delay** | CS4's cells and tiers 2 to 4 report it; CJ6 names it a trade; no sentence calls the refit "better", only what it does at each delay |
| **Generation scores track the delay draws** (lesson 17) | CF3: five of the six delays in every generation, the same for every member; the table written before generation 0 |
| **A negative read as "impossible"** | Section 5's last paragraph is part of the report's text: what "did not cure it" does and does not mean |
| **One fit is a narrow test** | Two fits; every cell prints both; a verdict needs both (lesson 2) |
| **The rows are tested on one seed block, and the idea came after results on those seeds** | The fit and the selection never see seeds 3042 to 3061; the label "add-on after the results"; "not replicated" printed wherever a pass is printed; μ 0.54 is an unseen floor |
| **The fit's episodes differ silently from the bar's** | The evaluator calls the bar's own row and judge; check 5 of 3.3 reproduces a registered episode bit for bit |
| **The objective's graded terms are new text** | They reuse Stage 1's functions unchanged; the two places where they are stricter than the bar are stated in CF5; the objective is labelled "not the judge"; tests bound every term to [0, 1] and prove a fall outweighs all of them |
| **Code state drifts between the stand-in's records and the new rows** | No file in a key changes; tests 1 to 4 of 3.3; CJ4's key comparison per pair. If item A or B changes a key file after all, the stand-in's V0 rows at 0, 4, 6 and 10 ms must be run again into the new directory first (2.4 to 3.4 core-hours more, design estimate), and both sets of records kept (S2-17) |
| **The budget cuts a fit mid-block, so kinds have met delays unequally** | CF7: a fit ends at a whole block; both fits are judged at the same generation |
| **Time or machine trouble** | Fit and selection resume per generation and per validation evaluation; at most 3 processes through `tools/lowprio.sh`; fits run with nothing else heavy (lesson 12) |
| **Read as the plan's R2L, or as a hardware claim** | CF1 and CJ7's labels |

## 8. Points for the lead to settle before the rules are committed

1. **The start of the two fits.** I follow Stage 1b: both from the body-level selection, which is `R2/43`'s gains.
   The alternative (fit 42 from `R2/42`'s gains, fit 43 from `R2/43`'s) pairs each new row with its own stand-in
   row but needs a start rule that `speedfit.start_record` refuses today.
2. **Validation seeds 2000 to 2099** in five blocks (CS2), where Stage 1b used 2000 to 2019. Labelled as a
   difference.
3. **The objective's item terms** (CF5). I took the reviewed round C wording and made it precise. A simpler form
   exists (the share of the bar's failed terms plus the DCM error); it needs no graded definitions but gives the
   search no slope toward a limit.
4. **Tier 5 (8 and 2 ms) and tier 6 (14 to 20 ms)** are my additions to the brief's 0, 4 and 6 ms; 0.20 to 0.28 and
   0.30 to 0.42 core-hours.
5. **The zero-gain diagnostic candidate** (CS1): 0.08 to 0.12 core-hours.
6. **Fits one after the other with 3 workers each** (1.0 to 1.4 h for both) or side by side with one worker each
   (1.5 to 2.1 h, the third core free).
7. **Task numbers and the driver.** One driver for items A, B and C; integers for `C1` and `C2`.
8. **`COST.md` of the new run directory:** measured again on Stage 1's own condition as on 15 Nov, or the 15 Nov
   file committed there by copy. My estimates use the 15 Nov file.
9. **Order.** P37 puts item A's delays first. These rules do not depend on A's result, except that tier 6 runs
   only if A ran 14, 16 and 20 ms.
10. **Replication.** By 75 H4 a control set is not run on seeds 5042 to 5061, so a pass of a new row stands on one
    block. If tier 1 shows a pass and the lead wants it repeated, that is a labelled amendment after a result.

## 9. Limits

- Repository read only: kept. `git status --porcelain` in `/home/user/mesozoic-labs` printed nothing after my last
  command. I wrote only under the scratch directory (this file, and the dry runs' output under `addon/dry_C/`).
- No closed-loop episode on a Stage 2 set, no fit, no training: kept. I ran the bar's `dry-run` with `--cost`
  (rates given, nothing measured), three commands that are refused before any episode (read in the code first),
  `--help`, and scripts that read committed JSON and `tune.json`.
- Python through `tools/lowprio.sh`, one process at a time, one thread, no network: kept for every command that
  touched the study. One exception: a text-replace script on this design file in my scratch directory ran once
  with the system `python3`, outside `tools/lowprio.sh` (well under a second; it read and wrote this file only).
- One thing my task named that does not exist: `designlab/stage2/speedfit.py`. The speed fit is
  `designlab/stage1/speedfit.py`.
