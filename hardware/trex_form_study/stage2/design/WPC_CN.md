# Work package CN: the notebook: `nb.py`, `stage2_training.ipynb`, the steps sheet

You are building one part of a robotics simulation study (a robotic T. rex biped; everything is simulation). You
need no other context than this brief and the two texts it points into:
- the design: `/tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/design/ROUND_C_DESIGN.md` (read its header, the decisions and sections named under "Read first" below);
- the rules fixed before the runs: `/tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/design/PREREG_TRACKB_draft.md` (rule ids such as "K3" or "W2" refer to it).
Round A's design, where a round-A API is named: `/tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/design/ROUND_A_DESIGN.md`.

**Read first:** design D19, D20, D22, D24, D25; sections 2.2 (`nb`), 3.9 (all of it), 3.10 (the acceptance run), 3.12 (the steps sheet's outline), the section 4 rows "The notebook" and "No checkout dirt"; rules K2, K4-K7, N1-N5, Y1-Y11, X1. Files: `/home/user/mesozoic-labs/notebooks/sb3_training.ipynb` (the scaffold: cell 3 lines 36-58, cell 8 lines 8-21 are what you adapt), `/home/user/mesozoic-labs/docs/investigations/stance_2026_10/stance3/colab_steps.txt` (the sheet's precedent), `tools/requirements-study.txt`.

## Goal

The owner's path: a thin Colab notebook whose cells each run one step of `python -m designlab.stage2.nb` in a subprocess; the queue launcher with its Drive copies, liveness fence and job rules; the collection of finished records; the tally of runtime-hours; the day-one smoke; the acceptance run in a fresh clone; and the steps sheet. Everything also runs here, with a local directory standing in for Drive.

## Files you own (and no others)

- `designlab/stage2/nb.py`
- `stage2/stage2_training.ipynb`
- `stage2/tb/colab_steps.txt`, `stage2/tb/requirements-colab.txt`
- `tests/test_stage2_nb.py`

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
Needs from other packages: package CP's `tbenv` (wave 1). The launcher half (session, launch, copies, LIVE.json, job rules, collect, hours) starts in wave 1 against a stub trainer; the real cells need packages CT (`train2` CLI) and CB (`bundle`); the acceptance run needs the lead (a pushed launch commit and the built bundle). Wave 1 (launcher) and wave 4.

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

Exactly the listing of the design's section 2.2 under `nb.py`; every function is also a CLI subcommand
(`session`, `inputs`, `preflight`, `smoke`, `launch`, `finish`, `collect`, `hours`, `local-run`, `acceptance`).
**`nb.py` imports only the standard library at module level** (it runs before the pins are trusted and is imported
by the notebook's kernel for `release_runtime`); it reaches `train2`, `bundle`, `g8a`, `s2load` only by
subprocess (`[sys.executable, "-m", ...]` with `tbenv.child_env(...)`, cwd the study directory).

## What to build, with pointers

1. **The notebook** (3.9's cell table; 9 cells, no accelerator in the metadata; every code cell parses as
   Python; the kernel imports only the standard library and `google.colab`). Cell 1's parameters exactly as
   listed (with `TAKEOVER`, `ALLOW_VERSION_CHANGE`, `ALLOW_SOURCE_CHANGE`); `S2_PARAMS` (a JSON file named by an
   environment variable) replaces them for local runs and tests. Cell 2: the pins from
   `stage2/tb/requirements-colab.txt` (mujoco 3.10.0, numpy 2.5.4, scipy 1.18.1, gymnasium 1.4.0,
   stable-baselines3 2.9.2, cloudpickle 3.1.2, pillow 13.0.0, PyOpenGL 3.1.10; torch 2.15.0 from the CPU index,
   forced as `torch==2.15.0+cpu` whenever the installed build label is not `cpu`), compared with the build label
   dropped; `apt-get install -y libosmesa6 ffmpeg` (failure reported); the scaffold's Git block; a non-smoke
   launch refuses a `REPO_REF` that is not 40 hex digits; line 1. No unpinned fallback. Cell 8: `nb finish`, then
   `nb.release_runtime(...)` in `try ... finally`.
2. **The four lines** (rule N1): the exact strings of 3.9, their failing forms, ` BELOW FLOOR <f>`, the session
   line's three forms. One function formats each; the tests and the steps sheet quote them from it.
3. **`session`**: SESSION_<k>.json (schema in 3.9), the `.part` sweep, rule K6's refusals (an existing RUN_ID
   without RESUME; RESUME with another bundle, queue or budget; REPO_REF taken from QUEUE.json on a resume, with a
   printed note if the cell's differs; another commit only with `ALLOW_SOURCE_CHANGE`), rule K7's (LIVE.json
   younger than 600 s refuses a RESUME unless `TAKEOVER`; a finished run, DONE.json or HALT.json, is not resumed).
4. **`inputs`**: parts from `<drive>/inputs/<bundle_id>/` (or `rebuilt/`, or `git:<ref>`), `bundle unpack` and
   `verify` against the committed manifest, then `setio.read_inputs(require_frozen=True)` per set and body in a
   subprocess; line 2 with the first 12 hex of `content_sha256`.
5. **`preflight`**: the repository's archive-load preflight, `train2 preflight`, every env of the queue's bodies
   built (both floors, the job's mode), `g8a.short` per body; line 3.
6. **`launch`** (3.9 "The launcher"): the queue, one job at a time (`train2 run ... --out <local> [--resume]` as a
   child in its own process group); every 60 s: LIVE.json re-read and fenced, STATUS.json (schema in 3.9) and
   LIVE.json written, text records copied, checkpoints copied by rule K2 (multiples of 1,000,000 steps and each
   check's; every checkpoint in a tiny run; three files as `.part` then renamed, the JSON last; "copied" after a
   read-back; an older one deleted one cycle after a newer one is copied; `ckpt_<n>.tar` by `s2load pack` for each
   check's), one clip child at a time (`clip2` with `tbenv.child_env("osmesa")`, 300 s limit, failure recorded);
   line 4 from `gates/rate.json`; job rules (exit 0 or 3: next job; a crash: one retry from the newest intact
   checkpoint, then HALT; a `run.json` without a checkpoint: restart from zero); on a resume the job's directory
   is copied back from Drive first; never raises (HALT.json). With `drive == local` nothing is copied.
   `--monitor-every` and `--monitor-also-k` are passed to every `train2 run` and stored in QUEUE.json (S2-4 runs
   with `--monitor-every 500000`, the local 60 cm run with `--monitor-also-k 0`).
7. **`smoke`** (rules Y1-Y8, in that order; SMOKE.json after Y7 and again after Y8; Y8 capped at 40 minutes with a
   "not finished" outcome; Y6 deletes the local job directory before its resume).
8. **`collect(src, dst, kind)`** (D22): copies finished text records (`*.json`, `*.jsonl`, `*.txt`; never under
   `partial/`, `traces/`, `logs/`, `work/`; of `checkpoints/` only `checkpoint_<n>.json`), verifies each copy by
   sha256, writes `COLLECTED.json` (source, files with hashes, the run's `run_uuid`s, time); refuses a job that is
   still running. **`hours`**: per run, each session from `started_at` to `last_status_at`, summed into
   `stage2/tb/hours.json` against the 220 h cap.
9. **`local_run`** (development): the notebook's cells executed as Python in this checkout with `S2_PARAMS`
   (set `S1`, `--tiny`, a stand-in Drive under a temporary directory), a kill after the first copied checkpoint,
   a resume. **`acceptance`** (3.10, its seven steps exactly; the record `ACCEPTANCE.json`).
10. **The steps sheet** (3.12's eight items; plain text, 115 columns, addressed to the owner): every quoted line
    comes from the acceptance run's record; lines only Colab prints are marked inferred; the minutes of each smoke
    item; no instruction asks the owner to change RUN_ID, REPO_REF, TAKEOVER or an ALLOW parameter on an error's
    advice.

## Tests to write (`tests/test_stage2_nb.py`; a stub trainer script that writes `progress.json`, `run.json`, `gates/rate.json` and checkpoint files on a timer)

Fast: the notebook's code cells parse (`ast.parse` of each); the kernel-side cells import only the standard
library and `google.colab` (AST walk); `import designlab.stage2.nb` in a subprocess with numpy blocked; the pins
file equals the study venv's versions (`importlib.metadata`), PyOpenGL and pillow included; the four lines match
their regular expressions and their failing forms; `session`'s refusals (existing RUN_ID; another queue, bundle or
budget; a REPO_REF that is no commit; a live LIVE.json; TAKEOVER; a finished run); the launcher on the stub
trainer: a second launcher on a live stand-in Drive is refused; a superseded launcher halts with "superseded"; a
crashed job is retried once and then HALT; a job stopped by a rule (exit 3) lets the next start; the `.part`
sweep; the copy cadence and the delete-one-cycle-later rule; STATUS.json and SESSION schemas; `collect` (hash
check, COLLECTED.json, refusal of a running job); `hours`; **a local job leaves `git status --porcelain`
unchanged** (a scratch git repository holding `stage2/tb/.gitignore`, the stub job under `stage2/tb/work/`).
Slow (`STAGE2_SLOW=1`): `local_run` on the S1 view with `--tiny` (kill, resume, DONE.json, both SESSION files,
the lines).

## Acceptance

```
nice -n 15 $PY -m pytest -p no:cacheprovider tests/test_stage2_nb.py
STAGE2_SLOW=1 nice -n 15 $PY -m pytest -p no:cacheprovider tests/test_stage2_nb.py -k local_run       # S1 view; 2 workers; about 5 minutes
git status --porcelain            # only your five files
```
Then, with the lead (it needs the pushed launch commit C1, the built bundle and G9-ref of the bundle's sets):
```
nice -n 15 $PY -m designlab.stage2.nb acceptance --ref <C1, 40 hex> --parts stage2/tb/bundle/<id>/parts \
    --scratch /home/user/mesozoic-wt/tb-accept --record stage2/tb/bundle/<id>/ACCEPTANCE.json          # about 12 minutes
```
and the steps sheet written from its record. Report the lines as printed, the kill point and the seconds of each
step.

## Commit

Commit on your own branch in your worktree, messages ending with the attribution lines your session's system
reminder gives, exactly; name no model anywhere else. Do not merge, do not push unless the lead says so. Final
report: files changed, each acceptance command with its exit code, every place where you read the design or a rule
in one of two possible ways (say which you chose: these are rules fixed before the runs, so say so rather than
decide silently), and anything the design asked for that you could not build.
