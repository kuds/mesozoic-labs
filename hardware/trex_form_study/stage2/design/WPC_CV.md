# Work package CV: worker processes: `vecpool.py`

You are building one part of a robotics simulation study (a robotic T. rex biped; everything is simulation). You
need no other context than this brief and the two texts it points into:
- the design: `/tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/design/ROUND_C_DESIGN.md` (read its header, the decisions and sections named under "Read first" below);
- the rules fixed before the runs: `/tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/design/PREREG_TRACKB_draft.md` (rule ids such as "K3" or "W2" refer to it).
Round A's design, where a round-A API is named: `/tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/design/ROUND_A_DESIGN.md`.

**Read first:** design D13, D16 (why nothing may depend on the process), section 2.2 (`vecpool`), the section 4 row "The worker count does not matter".

## Goal

`PoolVecEnv`: a stable-baselines3 `VecEnv` that hosts N envs in W child processes, bit-identical to SB3's `DummyVecEnv` at any W, whose workers import neither torch nor stable-baselines3 and are started as a subprocess with a command line the caller chooses.

## Files you own (and no others)

- `designlab/stage2/vecpool.py`
- `tests/test_stage2_vecpool.py`

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
# nothing to restore
```
Versions: Python 3.13.18, torch 2.15.0+cpu, stable-baselines3 2.9.2, gymnasium 1.4.0, mujoco 3.10.0, numpy 2.5.4.
Needs from other packages: none. Wave 1: start now. (`train2` will pass `worker_cmd=[sys.executable, "-m", "designlab.stage2.train2", "worker"]`.)

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

```python
class PoolVecEnv(VecEnv):
    def __init__(self, specs: list[tuple[str, dict]], workers: int, worker_cmd: list[str] | None = None)
def worker_main(address: str) -> int
# CLI: python -m designlab.stage2.vecpool --worker ADDRESS
```
- `specs[i] = ("pkg.module:factory", kwargs)`: env i is `factory(**kwargs)` (a `gymnasium.Env`); kwargs must be
  plain JSON-able values or dataclasses importable in the child (they travel by pickle over the connection; never
  a closure or lambda).
- `workers == 0`: build the envs here and delegate everything to SB3's own `DummyVecEnv` (the reference
  behaviour). `workers >= 1`: contiguous blocks `numpy.array_split(range(n), W)`.
- Start-up: the parent opens a `multiprocessing.connection.Listener` on a Unix socket in a private temporary
  directory with a random authkey, starts each child as `worker_cmd + [address]` with
  `tbenv.child_env("disable")` plus the authkey in an environment variable (if `tbenv` is not on your branch yet,
  build that environment inline and say so in your report), accepts the connections, sends each child its specs,
  and receives the spaces. The default `worker_cmd` is `[sys.executable, "-m", "designlab.stage2.vecpool",
  "--worker"]`. Do **not** use `multiprocessing.Process`: the command line must be the caller's (a memory guard
  protects processes by a substring of it).
- The worker loop mirrors `DummyVecEnv` statement for statement **without importing stable_baselines3 or
  torch** (read `stable_baselines3/common/vec_env/dummy_vec_env.py` and `base_vec_env.py` in the venv): `reset`:
  per env `reset(seed=self._seeds[i], options=self._options[i])`, then seeds and options cleared; `step`: per
  env `obs, rew, terminated, truncated, info = env.step(action)`, `done = terminated or truncated`,
  `info["TimeLimit.truncated"] = truncated and not terminated`, on done `info["terminal_observation"] = obs` and
  `obs, reset_info = env.reset()`; `seed(seed)`: env i of the whole pool gets `seed + i` (global index, not the
  index inside its worker) at its next reset, and the call returns the list; `env_method`, `get_attr`, `set_attr`,
  `env_is_wrapped` (False), `set_options`, `close`. Observations and rewards return in env order with
  `DummyVecEnv`'s dtypes and shapes (float32 rewards, copies of observations).
- Failure: a child that exits or a broken connection raises `RuntimeError` naming the worker and its exit code
  from the next call; `close()` terminates children, joins with a timeout, kills leftovers, and removes the
  socket directory; an `atexit` hook does the same; a child exits when its parent's connection closes.
- Each worker asserts `MUJOCO_GL == "disable"` and one thread per BLAS variable at start.

## Tests to write (`tests/test_stage2_vecpool.py`; a seeded stub env defined in the test module, importable by name)

The stub: observation 8 floats, action 3, own `numpy` generator seeded at reset, episodes of random length 5-40
with both terminations and truncations, an attribute and a method to call. Tests: 400 steps of fixed random actions
after `seed(123)`, `reset()`: observations, rewards, dones and the infos' `terminal_observation` and
`TimeLimit.truncated` equal `DummyVecEnv`'s bit for bit at W = 1, 2, 4 with 8 envs (W = 8 behind `STAGE2_SLOW`);
with W = 3 and 8 envs (uneven blocks); `seed` returns `DummyVecEnv`'s list; `env_method` and `get_attr` keep env
order; `set_options` reaches the next reset; a worker's `sys.modules` (asked through `env_method`) holds neither
`torch` nor `stable_baselines3`; `/proc/<pid>/cmdline` of a worker holds the `worker_cmd` given; killing a worker
makes the next `step_wait` raise; `close()` leaves no child process and no socket directory; a stable-baselines3
`VecNormalize` wraps the pool and a 2-update PPO run on it equals the same run on `DummyVecEnv` (policy tensors
equal; needs torch; fast: `n_steps` 16).

## Acceptance

```
nice -n 15 $PY -m pytest -p no:cacheprovider tests/test_stage2_vecpool.py          # under 2 minutes; at most 4 workers
git status --porcelain            # only your two files
```

## Commit

Commit on your own branch in your worktree, messages ending with the attribution lines your session's system
reminder gives, exactly; name no model anywhere else. Do not merge, do not push unless the lead says so. Final
report: files changed, each acceptance command with its exit code, every place where you read the design or a rule
in one of two possible ways (say which you chose: these are rules fixed before the runs, so say so rather than
decide silently), and anything the design asked for that you could not build.
