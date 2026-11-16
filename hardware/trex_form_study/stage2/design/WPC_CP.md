# Work package CP: policy files: `tbenv.py`, `schedules.py`, `s2load.py`

You are building one part of a robotics simulation study (a robotic T. rex biped; everything is simulation). You
need no other context than this brief and the two texts it points into:
- the design: `/tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/design/ROUND_C_DESIGN.md` (read its header, the decisions and sections named under "Read first" below);
- the rules fixed before the runs: `/tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/design/PREREG_TRACKB_draft.md` (rule ids such as "K3" or "W2" refer to it).
Round A's design, where a round-A API is named: `/tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/design/ROUND_A_DESIGN.md`.

**Read first:** design D14, D16 (the load), D19 (`repo_on_path`), section 2.1 (the entry snippet), 2.2 (`tbenv`, `schedules`, `s2load`), 3.6 (`checkpoint_<n>.json`, `fingerprint_<n>.npz`), 3.11 (what a row needs from the loader); rules K4, K9.

## Goal

1. `tbenv.py` (standard library only): the environment of every Track B child process, the `MUJOCO_GL` assignment and the repository root on `sys.path`.
2. `schedules.py`: learning-rate and clip schedules as instances of module-level classes, so that a saved archive holds no bytecode (S2-14).
3. `s2load.py`: the one loader of Track B. Checkpoint files, their integrity, the policy fingerprint (S2-12), version sets, the deterministic tar of a checkpoint and its import here.

## Files you own (and no others)

- `designlab/stage2/tbenv.py`, `designlab/stage2/schedules.py`, `designlab/stage2/s2load.py`
- `tests/test_stage2_tbenv.py`, `tests/test_stage2_schedules.py`, `tests/test_stage2_s2load.py`

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
# nothing to restore: no arrays or models are needed
```
Versions: Python 3.13.18, torch 2.15.0+cpu, stable-baselines3 2.9.2, gymnasium 1.4.0, mujoco 3.10.0, numpy 2.5.4.
Needs from other packages: none. Wave 1: start now.

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

## API (frozen; ask the lead before changing a name or a signature)

Exactly the listing of the design's section 2.2 under `tbenv.py`, `schedules.py` and `s2load.py`. Notes:
- `tbenv.assign_gl(backend)`: `os.environ["MUJOCO_GL"] = backend` (assigned, never `setdefault`); for "disable" pop
  `PYOPENGL_PLATFORM`; for "osmesa" set `PYOPENGL_PLATFORM=osmesa` (`designlab/__init__.py:15-16` defaults that
  variable from `MUJOCO_GL`, and mujoco's OSMesa backend raises unless it is unset or `osmesa`); raise
  `RuntimeError` if `mujoco` is already in `sys.modules` and the variable held another backend before.
- `tbenv.child_env(gl, base=None)`: a copy of the environment with `MUJOCO_GL`, `PYOPENGL_PLATFORM` (osmesa only;
  removed for disable), the three thread variables at "1", `PYTHONHASHSEED=0`, `PYTHONPATH="."`.
- `tbenv.repo_on_path()`: appends the repository root (the study directory's parent's parent) to `sys.path` once
  (never at position 0) and returns it. `s2load` calls it before `from environments.shared...` imports, because
  `environments` is not importable from the study directory with `PYTHONPATH=.`.
- `schedules.LinearSchedule(start, end)(progress_remaining) = end + (start - end) * progress_remaining`;
  `ConstantSchedule(value)`; both picklable by reference to this module, with `__eq__` and `__repr__`.
- `s2load` imports torch and stable-baselines3 inside functions only (importing the module must not import them).
- `save_checkpoint(job_dir, steps, model, vecnorm, layout, run, state)` writes, in this order, each through a
  temporary name and `os.replace`: `checkpoints/model_<n>.zip` (`model.save`), `vecnormalize_<n>.pkl`
  (`vecnorm.save`), `fingerprint_<n>.npz` (`obs`, `actions`, `obs_mean`, `obs_var`, `obs_count`, `clip_obs`,
  `epsilon`), `checkpoint_<n>.json` (schema `s2-ckpt-2`, design 3.6; `run` and `state` are stored as given). It
  refuses (raises `CheckpointError`, deletes the three files) if the repository's `inspect_sb3_archive` reports a
  bytecode member in the archive.
- `fingerprint_obs(layout)`: `[256, obs_dim] float32` from `numpy.random.default_rng([20261115, obs_dim,
  actor_dim])`, block by block as the design's 3.6 lists (angles 0.3 z, rates 2 z, gravity (0, 0, -1) + 0.05 z
  renormalised, gyro 0.5 z, loads uniform 0-1, phase a point of the unit circle, support bits, commands uniform on
  their ranges, the heading 0.3 z when the layout has it, previous actions 0.1 z, the head command uniform on
  +-1); the critic's columns zero. `layout` is a plain dict with `blocks` (name -> [i0, i1]), `actor_dim`,
  `obs_dim`, `action_dim`; you do not need `r5core` (use a hand-made layout in tests).
- `normalize(stats, obs)` is `VecNormalize.normalize_obs`'s formula on plain arrays; a test compares it with SB3's.
- `load_model(path, layout, env=None, lr=None)`: through `environments.shared.policy_loading.load_sb3_model(path,
  algorithm="ppo", device="cpu", env=env, force_reset=False, custom_objects=...)`. Training (env given):
  `learning_rate` and `lr_schedule` = `LinearSchedule(*lr)`, `clip_range` = `ConstantSchedule(0.2)`. Inference:
  those with `ConstantSchedule(0.0)`, plus `observation_space` and `action_space` built from `layout`
  (`gymnasium.spaces.Box`, float32, the env's bounds: observation +-inf, action +-1), and `None` for `_last_obs`,
  `_last_original_obs`, `_last_episode_starts`, `ep_info_buffer`, `ep_success_buffer`. If `load_sb3_model` does
  not forward `force_reset`, tell the lead (the design's kill-and-resume equality depends on it).
- `load_checkpoint(...)`: `pair_problem` first (your hashes and bytes, then the repository's
  `environments.shared.curriculum.checkpoints.checkpoint_pair_problem`), then the load, then the fingerprint
  reproduced from the stored observations and plain statistics; `max |difference| > tol` raises `CheckpointError`
  ("not judged"); `expect_run` (a dict with `run_uuid`, `settings_sha256`) must equal the record's.
- `keep(steps_all, monitor_every)`: the multiples of `monitor_every` (as reached: a checkpoint belongs to a check
  when it is the first at or past a multiple) and the newest two.
- `pack` writes `ckpt_<n>.tar` with the bundle's tar rules (USTAR, sorted names, mtime 0, uid and gid 0, empty
  user and group names, mode 0644, no compression); `import_tar(record, tar, into)` unpacks into
  `<into>/checkpoints/`, checks the three sha256 against the record, copies the record, runs `load_checkpoint`, and
  returns the validation record (`{record sha256, files, versions_saved, versions_here, cross_version,
  fingerprint_max_abs_diff, tol, ok}`); the CLI's `--write` stores it.

## Tests to write

`tests/test_stage2_tbenv.py`: `child_env("osmesa")` from a process whose environment holds
`MUJOCO_GL=disable`, `PYOPENGL_PLATFORM=disable` gives both as `osmesa`; `child_env("disable")` removes
`PYOPENGL_PLATFORM`; thread variables and `PYTHONHASHSEED`; `assign_gl` in a subprocess started with
`MUJOCO_GL=osmesa` leaves `disable` and raises when mujoco was imported first; `repo_on_path` appends once and
`import environments.shared.policy_loading` then works in a fresh subprocess; importing `tbenv` in a subprocess
with numpy blocked (`sys.modules["numpy"] = None`) works.
`tests/test_stage2_schedules.py`: values at progress 1, 0.5, 0; pickle round trip; `describe`.
`tests/test_stage2_s2load.py` (needs torch; a tiny `gymnasium` stub env with a 140-wide observation and 17-wide
action, `policy.PrivilegedCriticPolicy` from `designlab.stage1.policy` with `policy.model_kwargs` updated by
`learning_rate=LinearSchedule(...)`): every row of the design's section 4 table "Cross-version loading": no
bytecode member in a saved archive; an archive saved with Stage 1's closure schedule is refused by
`save_checkpoint`; the fingerprint is reproduced exactly in a fresh subprocess; an inference load of an archive
whose space and last-observation members were replaced by objects that raise on unpickling still loads; `normalize`
equals `VecNormalize`'s; a changed byte, a missing file, a truncated zip, a foreign `run_uuid` and a perturbed
fingerprint are each refused with their reason; `newest_intact` skips a broken newest checkpoint and reports why;
`comparable` drops torch's build label; `keep`; `pack` gives identical bytes twice and `import_tar` round-trips;
importing `s2load` imports neither torch nor stable_baselines3.

## Acceptance

```
nice -n 15 $PY -m pytest -p no:cacheprovider tests/test_stage2_tbenv.py tests/test_stage2_schedules.py tests/test_stage2_s2load.py
nice -n 15 $PY -m designlab.stage2.s2load check <tmp job dir> <n>        # on a scratch checkpoint your test helper wrote: "intact; fingerprint 0.0"
nice -n 15 $PY -m designlab.stage2.s2load pack <tmp job dir> <n> --out <tmp>; $PY -m designlab.stage2.s2load import --record ... --tar ... --into <tmp2>
git status --porcelain            # only your six files
```

## Commit

Commit on your own branch in your worktree, messages ending with the attribution lines your session's system
reminder gives, exactly; name no model anywhere else. Do not merge, do not push unless the lead says so. Final
report: files changed, each acceptance command with its exit code, every place where you read the design or a rule
in one of two possible ways (say which you chose: these are rules fixed before the runs, so say so rather than
decide silently), and anything the design asked for that you could not build.
