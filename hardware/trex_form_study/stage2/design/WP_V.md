# Work package V: views of the bar's episodes (the turning path and video clips)

Lead's brief, 15 Nov 2026. Read all of it before you write code. It is self-contained: you do not need
`ROUND_A_DESIGN.md`. Where it is silent, read the code named below, not other design papers.

## Why

The owner (decision log P35) wants to *see* what the 75 cm robot does in simulation: how big a circle it needs to
turn, and video of the behaviours the bar tests (the walk at 0.40 m/s, the quick 90° turn, the stop, the nudges). The
bar's tables give pass counts; this package gives the picture and the room it needs, from the same records.

Everything this package makes is a *view of committed records and their traces*. It judges nothing, changes no record
and no verdict, and adopts nothing.

## The hard limits (break none)

1. **Change no existing file.** New files only: `designlab/stage2/barviews.py`, `tests/test_stage2_barviews.py`,
   and outputs under the run directory named below. `bar.py`, `readings.py`, `setio.py`, `sets.py`, `moves.py`,
   `clips.py`, every Stage 1 module and every record stay byte for byte as they are: the 75 cm panels are running from
   them right now and their bytes are in every record's key. Import what you need.
2. **Run no closed-loop episode on a Stage 2 set** (V0, F25F9, F25F9cl). The panels' chain is the only thing allowed to
   do that. You read its records and traces. If you need a trace before the chain has made one, play a *reference*
   kinematically (no physics), or run a smoke episode on Stage 1's own run directory through the pseudo-set `S1`
   (`bar run --set S1 ... --out <a scratch directory>`; see `bar.py`'s docstring), never on a Stage 2 set.
3. **CPU: one process at a time**, always through `bash tools/lowprio.sh`, one BLAS thread
   (`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`), `LP_NUM_THREADS=1` for the software renderer. The
   panels use three of the machine's four cores and their episode wall times are a budget; do not start a second
   process, a `--procs` above 1, or the full test suite.
4. **Never touch the chain**: do not write, move or delete anything under `stage2/runs/2026-11-15-75cm/` except your
   own output directory `stage2/runs/2026-11-15-75cm/views/`, and do not take the bar's lock.
5. Git: work in your worktree on your branch, small commits, commit as you go, never push. Commit messages end with
   the two attribution lines of the repository's recent commits (copy them exactly from `git log -1`) and name no
   model anywhere else. Run `git check-ignore -v` on every output you mean to commit (the root ignores `logs/`,
   `models/`, `*.zip`, `*.pkl`, `*.npz`; media go where they are ignored or outside git, see below).
6. Words: plain, short sentences; say "planned" for anything read from a reference and "simulated" for anything read
   from an episode, and never mix the two in one number. Every frame of every clip says which it is.

## What exists (read these first)

- `designlab/stage2/bar.py` (docstring; `RULES["trace"]`; `condition_record`; `BarPlanner`; the record layout under
  `--out`): one JSON record per set, body, block, row and condition, each naming its episodes (seed, outcome, readings)
  and each episode's trace file with its sha256 and `kept`.
- `designlab/stage1/panels.py` `TRACE_RULE`, `write_trace`; `designlab/stage1/metrics.py` `COLS`: a trace is an
  `.npz` with `t`, `t_ref`, `rows` (the recorder's columns at 50 Hz: CoM and its velocity, feet, lowest points, roll,
  pitch, yaw, sole centres, loads, `root_x`, `root_y`, ...), `cols` and `q` (the joint angles in plant order), plus the
  bar's extra arrays (contact flags, low points, the per-swing table, the command instant).
- `designlab/stage2/clips.py`: the look ladder's kinematic clips. Reuse by import its scene (`ClipScene`), cameras,
  `overlay`, `encode` (H.264, size cap), `contact_sheet`, `probe`. Its clips are labelled as kinematic playback; yours
  need their own label (below).
- `designlab/stage2/setio.py`: reading a frozen set (`stage2/sets/<set>/75-M-H-10/`): `set.json`, `refs.json`,
  `transitions.json`, the arrays, each checked against its sha256.
- The run directory: `stage2/runs/2026-11-15-75cm/` (`run_75cm.sh` shows every unit; `bar summary --out` lists what
  is done). Task 3 (the walk) runs first and takes several hours; task 4 (turns, nudges, stop) follows. Build and
  test on what exists when you start, and make every command work on a run directory that is still filling.

## What to build

### 1. `barviews.pose_of(trace, model)`: the full pose at each trace row

A trace holds the joint angles, roll, pitch, yaw, `root_x`, `root_y` and the CoM, not the root's height. Recover it
exactly: set the free joint from `root_x`, `root_y`, the attitude and a height of zero, run the kinematics, and shift
the root so that the model's CoM height equals the trace's. State in the docstring which convention the attitude
columns use (read `metrics.py`; do not guess) and test the round trip: a pose written by the plant, recorded by
`metrics.Recorder`, comes back to within 1e-4 m and 1e-4 rad on every body position you draw (use a Stage 1 smoke
episode or the recorder on a kinematic play for the test).

### 2. `barviews turnpath`: the room a turn needs

For one set and each turn condition it has a reference for (`turn/q90_left`, `turn/q90_right`, and the reported rates),
write `views/turnpath/<set>/turnpath.json` and `TURNPATH.md` with two blocks that are never merged:

- **Planned** (from the frozen reference, kinematic): the path speed and turn rate commanded; the path radius
  `v / omega`; the radius fitted to the root's planned path over the steady part of the turn; the *corner box*: how far
  the root advances along its incoming line and how far sideways, from the command instant to the end of the move; and
  the *swept radius*: the largest distance from the turn's centre to any point of the robot over the turn (take the
  snout tip, the tail tip, both feet's outer edges and the body's cover box corners from the model; name the point that
  sets it). From the swept radius give the *turning circle*: the diameter of the circle a full turn at that rate would
  sweep, and the *corridor*: the width a 90° corner needs (state the construction in the file).
- **Simulated** (from each kept trace of each record of that condition: row, delay, floor grip, block): the same
  readings per episode, on the episodes that stayed up, then the median and the smallest and largest over them with the
  count ("17 of 20 episodes stayed up; of those ..."); also the heading reached 2 s after the command (the bar's own
  judged instant; take the instant from the record, do not recompute it) and the heading at the end. Episodes that
  fell are counted and listed, not read.
- Units: metres and degrees, and the same lengths in body lengths (give the body length you used and where it comes
  from). Every row cites its record's path.
- A top-view figure per condition (`.svg` and `.png`): the planned path, each simulated path thin, the corner box, the
  robot's outline at the command and at the end, a scale bar, and a caption that says which lines are planned and which
  simulated. Use the categorical colours and the fonts of `designlab/stage2/look_page.py` or `tools/report_page.py`
  if they define a palette; legible at 400 px width.

### 3. `barviews clips`: video of simulated episodes

For a list of (set, condition, row), render the episode chosen by the rule below from its trace:

- **Which episode (stated here before any record exists, so that no clip is picked by eye):** the block's first seed
  (3042). If that episode's outcome (stayed up or fell) is not the majority outcome of the record's 20 episodes, render
  in addition the lowest seed that has the majority outcome. Every clip's label gives its seed, the record's count
  ("stayed up in 17 of 20") and whether this episode is the first seed or the majority one. A trace that was not kept
  (`kept: false`) gives no clip; the manifest says why.
- **Views:** `visitor` (three-quarter front, as `clips.py`) for every clip; `top` (straight down, the camera high
  enough to hold the whole move, north along the incoming walking line) for turns and nudges; `side` for the walk and
  the stop. Cameras follow the planned path smoothly (not the body's own wobble).
- **On every frame:** top band: "SIMULATION (physics)", the body, the set in plain words (V0 "today's walk", F25F9 "the
  pull request's walk", F25F9cl "the candidate"), the condition in plain words ("quick 90° turn to the left at 0.40
  m/s"), the row in plain words ("controller R2/42, servo delay 10 ms, floor grip 0.9"); bottom band: seed, outcome
  of this episode and the record's count, the view, the clip time. From the command instant, for 1.5 s, a caption
  "command: <turn left 90° | stop | push from the left, N newtons ...>" (read the plan from the record; do not
  hardcode). A fall shows a caption "fell" from the instant the record gives.
- 854 × 480, 25 or 30 fps at real time (1 s of clip is 1 s of simulated time; interpolate between the 50 Hz rows as
  `clips.Playback` interpolates), H.264 through `clips.encode`, each at most 3 MB; a poster PNG and a contact sheet
  beside each. Start each clip 1.0 s before the command (for the walk: the judged window's first 10 s) and end it 1.0 s
  after the move's end or the fall.
- Media are written outside git: `--media <dir>` (default `logs/barviews/`, which the root ignores).
  `views/clips.json` (committed, small) lists every clip: its record path, trace sha256, seed, selection reason,
  outcome, view, file name, bytes and the media file's sha256.
- A `planned` mode renders the same views from the frozen reference alone, labelled "PLANNED MOTION (kinematic, no
  physics)", so that a planned and a simulated clip of the same move can stand side by side.

### 4. Tests (`tests/test_stage2_barviews.py`; fast, no rendering in the default run)

- the pose round trip of item 1;
- the turn readings on a synthetic circular path with a known radius, centre and heading (planned and simulated code
  paths), including a path that stops short and an episode that fell;
- the selection rule on synthetic records (first seed is the majority; first seed is not; no majority episode kept;
  trace not kept);
- labels: every frame's top band contains "SIMULATION (physics)" or "PLANNED MOTION (kinematic, no physics)" and never
  both; plain-word names for the three sets;
- one rendered frame under `MUJOCO_GL=osmesa` behind an environment flag (`STAGE2_SLOW=1`), as the look clips' tests do;
- nothing under `stage2/runs/2026-11-15-75cm/` outside `views/` is written (run the commands on a copy and compare
  the tree).

## Acceptance (report each with the command and its own last lines)

1. `pytest -p no:cacheprovider -q tests/test_stage2_barviews.py` passes.
2. `barviews turnpath --set V0` and `--set F25F9cl` write the planned block from the frozen sets (they exist when you
   start, or the lead will say otherwise), and the simulated block for every turn record that exists at that moment
   (possibly none yet: then the file says "no record yet" and the command still exits 0).
3. `barviews clips` renders, from the records that exist: the walk at 0.40 m/s on the candidate under `R2/42@10ms`
   (first seed), in `visitor` and `side`; and the planned quick turn of V0 and of the candidate in `visitor` and `top`.
   Look at the posters and contact sheets yourself (Read the PNGs) and say what you see: is the robot upright, on the
   floor, feet not sunk, labels legible, the top view oriented as stated.
4. `git status --porcelain` is clean; `git check-ignore -v` output for every new path.

## Report

Return: what you built (files, public functions, commands), the acceptance results, anything in this brief you could
not do or read differently and why, and anything you saw in the records or traces that looks wrong (say what and
where; change nothing).

---

## LEAD ADDENDUM 1, 15 Nov, 19:50 UTC (read before you build; it overrides the text above where they differ)

- **Your worktree** is `/home/user/mesozoic-wt/wp-v` (branch `wt/wp-v`). The three frozen sets' arrays and Stage 1's
  input arrays are copied in (`sets verify` passes there). The interpreter is
  `/home/user/mesozoic-labs/hardware/trex_form_study/.venv/bin/python`.
- **The panels' chain runs in the main checkout**, not in your worktree. Its records and traces are under
  `/home/user/mesozoic-labs/hardware/trex_form_study/stage2/runs/2026-11-15-75cm/` (`<set>/75-M-H-10/...`,
  `budget.json`). That directory is **read-only for you**: read records and traces from it by absolute path (give your
  commands a `--run DIR` option whose default is the worktree-relative run directory), and write every output into
  your own worktree (`stage2/runs/2026-11-15-75cm/views/` there) or your scratch directory. Never write, lock, move
  or delete anything in the main checkout. A record or a trace may be half-written or missing at the moment you look:
  read defensively (a record that does not parse, or a trace whose sha256 is not the record's, is skipped and listed).
- **The chain's order**: task 3 (the walk at every delay, then the speed ladder) first, then task 4 (turns, nudges,
  the stop). Turn traces will appear part-way through your work; make acceptance item 2 and 3 pass on what exists, and
  say in your report which records existed when you ran them.
- Records exist for delays of 0 to 10 ms and for two fitted controllers and the open-loop row. Episodes that fall are
  common in some records. The selection rule stands as written. The lead chooses which records get clips; make
  `barviews clips` take a list (set, condition, row) and also a `--all-gated --row ROW` shorthand.
- Scratch directory: `/tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/wp_v/`.
