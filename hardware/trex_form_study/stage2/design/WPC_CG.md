# Work package CG: gates and validation records: `gates2.py`, `run_trackb.sh`

You are building one part of a robotics simulation study (a robotic T. rex biped; everything is simulation). You
need no other context than this brief and the two texts it points into:
- the design: `/tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/design/ROUND_C_DESIGN.md` (read its header, the decisions and sections named under "Read first" below);
- the rules fixed before the runs: `/tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/design/PREREG_TRACKB_draft.md` (rule ids such as "K3" or "W2" refer to it).
Round A's design, where a round-A API is named: `/tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/design/ROUND_A_DESIGN.md`.

**Read first:** design D17, D18, D22, D23; sections 2.2 (`gates2`), 3.7 (all of it), section 4 (the "record" entries and the reviewer's list), section 6 (the run plan, its order and its layout); rules Z1, Z2, K3, O2-O4, X1.

## Goal

The committed records of the gates and of the validation: G8a per body and head, G9 in its two parts, rates and memory, kill and resume on the real env, the training start with `MUJOCO_GL` set against it, R6's start; and the idempotent chain that runs the plan's steps in order.

## Files you own (and no others)

- `designlab/stage2/gates2.py`
- `tests/test_stage2_gates2.py`
- `stage2/tb/run_trackb.sh`
- the records your commands write under `stage2/tb/gates/` and `stage2/tb/validation/` (committed by the lead as they are made)

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
$PY tools/restore_inputs.py copy-from /home/user/mesozoic-labs/hardware/trex_form_study --group picks-models picks-arrays look x
```
Versions: Python 3.13.18, torch 2.15.0+cpu, stable-baselines3 2.9.2, gymnasium 1.4.0, mujoco 3.10.0, numpy 2.5.4.
Needs from other packages: packages CE, CP, CV, CT, CW, CK merged; round A's `bar.py`; for the committed records the frozen look sets (V6 on each body and head trained). Wave 4.

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

Exactly the listing of the design's section 2.2 under `gates2.py`. Every function writes one JSON record (rule
text, inputs' hashes, sources, versions, commit, dirty, time) to `record` and returns it; exit code 0 when the
gate passes, 1 when it fails, 2 on an error.

## What to build, with pointers

1. **`g8a`** (3.7, rule Z1): `g8a.bar_form(body, set_name, G8A_CONDITIONS, (3042, 3043), (0, 5))` and the record
   schema of 3.7; conditions the set has no reference for go to `not_run` with the set's reason.
2. **`g9_ref`** (3.7, rule Z2, parts (a), (b), (c), (f)): (a) cites round A's `bar refcheck` record by path and
   sha256 (run `bar.refcheck` if absent); (b) per rung of `stage2/look/metrics.json` for the body:
   `lookread.playback` of the set's walk against the ladder's numbers within the four limits, and the entry's
   `look_sha256` against `stage2/look/L1_FREEZE.json`; rungs outside the ladder: four finite numbers, at least 4
   steps; an `-X` body: against the plain body's playback of the same rung; (c) the four degraded playbacks and
   the 3-step one, through `lookread`; (f) for every task of `r5lib.Library(p, body, set_name)`: the largest joint
   jump across each join, 2 s of the tracker's targets under the zero action finite, and the `setscreen` record
   cited for references no bar item plays. Kinematic only: this gate runs **before any closed-loop episode on the
   set**, and `gates2` refuses `g8a`, `rates`, `kill-resume` and `r6-start` on a look set whose `g9_ref.json` is
   missing or failed.
3. **`g9_run`** (parts (d), (e)): reads the companion records `lookbar.collect` wrote for `R0L@10ms` and the
   bar's 10 ms records; cites round A's `bar reread` record for the 60 cm body and head.
4. **`rates`**: `r5env.throughput` per floor in one process, then a `train2`-style rollout loop (no PPO update)
   with 8 envs at 1, 2, 3, 4 workers for 16,384 steps each; env steps per wall second, CPU seconds, resident
   memory of the parent and of each worker; `provisional: true` unless the 1-minute load average before the run
   was under 0.5.
5. **`kill_resume`** (rule K3, exactly its settings): two runs through `train2` under `stage2/tb/work/`, the
   second killed by `os.killpg(..., SIGKILL)` when its `progress.json` reads 81,920 steps and resumed with 3
   workers; equality of every policy tensor, the optimizer's state, the normalisation statistics with their count
   and the fingerprint of the two final checkpoints; the record lists what was compared and the first difference
   if any. A difference is a failed gate that goes to the lead; do not add a tolerance.
6. **`gl_start`** (a `train2 preflight` subprocess started with `MUJOCO_GL=osmesa` in its environment: exit 0 and
   `disable` inside) and **`r6_start`** (an `R6Env` on the body: 200 steps of zero and of random actions, a mid
   start, targets inside the ranges; a 2-rollout tiny R6 training start).
7. **`run_trackb.sh`**: round A's `step` pattern (a done-marker per step under `stage2/tb/work/done/`, each
   command resumable, `tools/lowprio.sh`, the environment line of the design's section 6), the steps V1-V10 of the
   run plan in its order with `nb collect` after each long job, and nothing that needs the lead's hands (V0, V0b,
   V11-V13 are comments naming who does them). Each closed-loop step on a look set appends its entry to
   `stage2/tb/dev_episodes.json` (rule X1) unless it is a record's own step.

## Tests to write (`tests/test_stage2_gates2.py`)

On the S1 view and on a synthetic set in `tmp_path`: `g8a`'s record schema and exit codes (a forced difference
gives 1 and names the condition, seed, delay and first differing step); `g9_ref` parts (c) and (f) on the look
ladder's arrays (skip without); the refusal of a closed-loop gate without a passed `g9_ref.json`; `rates`' record
on a stub env with 2 workers; `kill_resume` on the stub env of package CT's tests (fast) and its difference
report when a tensor is perturbed; `run_trackb.sh` passes `sh -n` and each step's marker makes a second run a
no-op (dry mode: `TB_DRY=1` echoes the commands).

## Acceptance

```
nice -n 15 $PY -m pytest -p no:cacheprovider tests/test_stage2_gates2.py
TB_DRY=1 sh stage2/tb/run_trackb.sh          # prints the chain in order; exit 0
git status --porcelain            # only your files
```
Then, with the lead, as each precondition holds (frozen V6 set of the body; G9-ref first), the run plan's records
in its order: V1 (`g9-ref`), V2 (`g8a`), V3 (R0L through `lookbar run`, `nb collect`, `g9-run`), V5-V7 (probes,
`gl-start`, `r6-start`, `kill-resume`), V4 (`rates`, last, on an idle machine). One heavy process at a time; the
design's section 6 gives each step's estimate. Report every record's path, sha256 and pass or fail.

## Commit

Commit on your own branch in your worktree, messages ending with the attribution lines your session's system
reminder gives, exactly; name no model anywhere else. Do not merge, do not push unless the lead says so. Final
report: files changed, each acceptance command with its exit code, every place where you read the design or a rule
in one of two possible ways (say which you chose: these are rules fixed before the runs, so say so rather than
decide silently), and anything the design asked for that you could not build.
