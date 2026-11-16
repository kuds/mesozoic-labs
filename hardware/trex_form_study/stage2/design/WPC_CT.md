# Work package CT: the trainer: `train2.py`, `monitor2.py`, `clip2.py`

> **Added 16 Nov 2026.** The lead's rulings of 16 Nov on the worker pool, what a resumed run compares, and the servo model are in `stage2/design/ROUND_C_RULINGS_16NOV.md`. Where that file and this brief differ, that file wins.

You are building one part of a robotics simulation study (a robotic T. rex biped; everything is simulation). You
need no other context than this brief and the two texts it points into:
- the design: `/tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/design/ROUND_C_DESIGN.md` (read its header, the decisions and sections named under "Read first" below);
- the rules fixed before the runs: `/tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/design/PREREG_TRACKB_draft.md` (rule ids such as "K3" or "W2" refer to it).
Round A's design, where a round-A API is named: `/tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/design/ROUND_A_DESIGN.md`.

**Read first:** design D2, D12-D16, D22, D23; sections 2.1 (the entry snippet), 2.2 (`train2`, `monitor2`), 3.6 (all of it), 3.9 (the launcher's contract with the trainer: files, exit codes; clips are the launcher's), the section 4 rows on kill and resume, lesson 16, `MUJOCO_GL`, the monitor; rules O1-O4, M1-M6, M3b, K1-K5, K8, K9. Code: `designlab/stage1/train.py` (what you mirror, not edit), `policy.py`.

## Goal

The training entry point for R5 and R6, runnable here and on Colab: 8 envs in W workers, checkpoints as intact sets with the run state, an exact resume, the probes, the 18-episode monitor with its gain, G8b and stop rules, the fine-tune, and a clip tool. `train2` is also the entry of the worker and monitor child processes.

## Files you own (and no others)

- `designlab/stage2/train2.py`, `designlab/stage2/monitor2.py`, `designlab/stage2/clip2.py`
- `tests/test_stage2_train2.py`, `tests/test_stage2_monitor2.py`

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
Needs from other packages: packages CE (`r5env`, `r5core`, `g8a`), CP (`tbenv`, `schedules`, `s2load`), CV (`vecpool`) merged into your branch. `monitor2` calls `lookbar.episodes` (package CW) through a lazy import: build and test against a stub of it (below). Wave 3.

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

Exactly the listing of the design's section 2.2 under `train2.py` and `monitor2.py`; `clip2.py`:
`python -m designlab.stage2.clip2 --trace PATH --body B --set S --out clip.mp4 [--seconds 3]` (starts with
`tbenv.assign_gl("osmesa")`; imports no torch; encodes with the `ffmpeg` on PATH as `designlab/stage2/clips.py`
does; writes `clip_<n>.json` with its inputs' hashes or the failure's reason; exit 0 either way unless `--strict`).

## What to build, with pointers

1. **`train2.train`** (3.6 "Start", "Sources", "Resume", "The callback", `updates.jsonl`, `run.json`): the PPO
   construction exactly as the code block of 3.6 (`kw = policy.model_kwargs(...); kw.update(...)`; the assertion
   that no schedule is a function); `VecNormalize`; `vecpool.PoolVecEnv(specs, workers, worker_cmd=[sys.executable,
   "-m", "designlab.stage2.train2", "worker"])` with specs `("designlab.stage2.r5env:make", {"mode", "cfg"})`;
   `torch.set_num_threads(1)`. The callback: checkpoint (at step 0 too) with the run state, the monitor child
   (`python -m designlab.stage2.train2 monitor ...` with `tbenv.child_env("disable")` and the session's sources
   hash), `monitor2.decide`, then the boundary (D16: reseed SB3, torch, numpy and the vec env with `100000 * seed +
   100 * b`, reset every env, zero `VecNormalize`'s running returns, set `model._last_obs` and
   `_last_episode_starts`). Resume in 3.6's four steps: `load_checkpoint(..., env=venv, lr=...)` (it loads with
   `force_reset=False`), run state from the checkpoint record, a due check replayed, then
   `learn(total_timesteps=budget - n, reset_num_timesteps=False)`; nothing resets or steps the vec env in between.
   No intact checkpoint with a `run.json` present: restart from step 0 in place. Exit codes 0, 3 (G8a), other.
   `--monitor-every N` and `--monitor-also-k K,...` (the monitor's episodes also at those delays, reported: rule
   M6) are settings of the run.
2. **Lesson 16**: wrap `model.policy.optimizer.step` to count the minibatch steps PPO really runs per update;
   `updates.jsonl` as 3.6; the `kl_cut` flag (rule O3).
3. **`--tiny`** (3.6 "Run level"): `n_steps` 64, 4,096 steps, a checkpoint every 1,024, a check every 2,048 on
   each condition's first seed with full-length episodes, G8b scaled to the first check; on set `S1` the monitor
   runs `monitor2.S1_CONDITIONS`. The trainer refuses set `S1` without `--tiny`.
4. **The fine-tune** (`--from-checkpoint`, rule K8) and `probe_lr(parent=...)`.
5. **Probes**: `probe_lr` (rule O3: per body from 1e-4 down, at sigma 0.1 and with `log_std` set to ln 0.03, 6
   updates each, stop at the first rate that passes both; the smallest of the bodies' passing rates is the
   value), `probe_sigma` (O2), `filter_choice` (O4: equal filtered action RMS); each writes one record.
6. **`monitor2`** (D15; rules M1-M6, M3b): `CONDITIONS`, `SEEDS` (6001-6018) as constants; `run` and `baseline`
   call `lookbar.episodes(body, set_name, rows, pairs, traces=...)` (rows built through `lookbar.rows(body,
   ["R5/<seed>@10ms"])` after `s2rows.configure`, or `["R0L@10ms"]`), never re-run an episode, and write
   `monitor_<n>.json` / `r0l.json`: per episode the companion episode as returned (`outcome`, `succeeded`,
   `succeeded_look`, `survival_s`, `clock`), `survival_credit`, then N, S, the counts in readings 2 and 3 and "but
   A1 with the look", `failures_by_term`, `mean_clock_factor`, `distinct_start_seeds`, the checkpoint's three file
   hashes, the sources hash it was given and its own (it refuses to run if they differ). `gain`, `g8b`,
   `fix_weights`, `decide` (pure functions of records and the rules dict: the same state from the same records,
   whether called live or at a resume), `s2_10`.
7. **`train2 preflight`**: build a policy with `MUJOCO_GL` assigned and run one tiny rollout (the notebook's
   check); `train2 worker ADDRESS` -> `vecpool.worker_main`; `train2 monitor ...` -> `monitor2.main`.

## The stub of `lookbar.episodes` for your tests

`episodes(body, set_name, rows, pairs, traces=None) -> list[dict]`, one dict per `(condition key, seed)`:
`{"condition", "seed", "outcome" ("read" | "fell" | "unread" | "error"), "reason" ("clock_behind" | "unreadable" | None), "fell", "survival_s", "planned_s",
"succeeded", "succeeded_but_A1", "succeeded_look", "succeeded_but_A1_look", "failed_terms" [...], "look" {...} |
None, "clock" {"mean_factor", "min_factor", "t_ref_reached", "t_ref_planned"}, "error" | None}`. Tests monkeypatch
`monitor2._episodes` (your one indirection to it) with tables of such dicts.

## Tests to write

`tests/test_stage2_train2.py` (a stub `gymnasium` env importable by name, with randomised episode lengths and a
`set_task_weights` method; a stub monitor): every row of the design's section 4 table on **kill and resume** (bit
for bit after SIGKILL between boundaries, also with another worker count; `obs_rms.count` at the first rollout
after the resume; a kill during the monitor; a kill after a forced G8b fix; `run.json` ahead of the checkpoint;
restart from zero equals a fresh run; every refusal with and without its flag; a later freeze with unchanged
files accepted and with a changed file refused), lesson 16 (a learning rate that trips `target_kl`:
`minibatch_steps_run < possible`; the `kl_cut` flag), `MUJOCO_GL` (a subprocess started with `MUJOCO_GL=osmesa`
runs one rollout, exit 0, `disable` inside it and inside a worker), an archive saved by `train2` has no bytecode
member, the schedule's value follows the step count across a resume. Slow (`STAGE2_SLOW=1`, S1 view): a tiny R5
run and a tiny R6 run, each killed after its first checkpoint and resumed to its budget.
`tests/test_stage2_monitor2.py`: every row of the section 4 table "Monitor, G8b, stop rules".

## Acceptance

```
nice -n 15 $PY -m pytest -p no:cacheprovider tests/test_stage2_monitor2.py
nice -n 15 $PY -m pytest -p no:cacheprovider tests/test_stage2_train2.py
STAGE2_SLOW=1 nice -n 15 $PY -m pytest -p no:cacheprovider tests/test_stage2_train2.py -k "tiny"      # S1 view; 2 workers; a few minutes
nice -n 15 $PY -m designlab.stage2.train2 run --mode R5 --body 75-M-H-10 --seed 42 --set S1 --tiny --workers 2 --out /tmp/<you>/tiny   # exit 0; run.json status "done"
git status --porcelain            # only your five files
```
Report the env steps per second of the tiny run's first rollouts at 2 workers and the trainer's and one worker's
resident memory.

## Commit

Commit on your own branch in your worktree, messages ending with the attribution lines your session's system
reminder gives, exactly; name no model anywhere else. Do not merge, do not push unless the lead says so. Final
report: files changed, each acceptance command with its exit code, every place where you read the design or a rule
in one of two possible ways (say which you chose: these are rules fixed before the runs, so say so rather than
decide silently), and anything the design asked for that you could not build.
