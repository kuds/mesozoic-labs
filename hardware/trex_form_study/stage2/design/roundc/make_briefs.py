"""Writes the ten round C work-package briefs (WPC_<name>.md) from one common frame and a body per package."""
from pathlib import Path

D = Path("/tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/design")
DESIGN = f"{D}/ROUND_C_DESIGN.md"
RULES = f"{D}/PREREG_TRACKB_draft.md"

HEAD = """# Work package {wp}: {title}

You are building one part of a robotics simulation study (a robotic T. rex biped; everything is simulation). You
need no other context than this brief and the two texts it points into:
- the design: `{design}` (read its header, the decisions and sections named under "Read first" below);
- the rules fixed before the runs: `{rules}` (rule ids such as "K3" or "W2" refer to it).
Round A's design, where a round-A API is named: `{d}/ROUND_A_DESIGN.md`.

**Read first:** {read}

## Goal

{goal}

## Files you own (and no others)

{files}

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
{restore}
```
Versions: Python 3.13.18, torch 2.15.0+cpu, stable-baselines3 2.9.2, gymnasium 1.4.0, mujoco 3.10.0, numpy 2.5.4.
Needs from other packages: {needs}

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

{body}

## Commit

Commit on your own branch in your worktree, messages ending with the attribution lines your session's system
reminder gives, exactly; name no model anywhere else. Do not merge, do not push unless the lead says so. Final
report: files changed, each acceptance command with its exit code, every place where you read the design or a rule
in one of two possible ways (say which you chose: these are rules fixed before the runs, so say so rather than
decide silently), and anything the design asked for that you could not build.
"""

WPS = {}

WPS["CP"] = dict(
    title="policy files: `tbenv.py`, `schedules.py`, `s2load.py`",
    read="design D14, D16 (the load), D19 (`repo_on_path`), section 2.1 (the entry snippet), 2.2 (`tbenv`, `schedules`, `s2load`), "
         "3.6 (`checkpoint_<n>.json`, `fingerprint_<n>.npz`), 3.11 (what a row needs from the loader); rules K4, K9.",
    goal="1. `tbenv.py` (standard library only): the environment of every Track B child process, the `MUJOCO_GL` assignment "
         "and the repository root on `sys.path`.\n"
         "2. `schedules.py`: learning-rate and clip schedules as instances of module-level classes, so that a saved archive "
         "holds no bytecode (S2-14).\n"
         "3. `s2load.py`: the one loader of Track B. Checkpoint files, their integrity, the policy fingerprint (S2-12), version "
         "sets, the deterministic tar of a checkpoint and its import here.",
    files="- `designlab/stage2/tbenv.py`, `designlab/stage2/schedules.py`, `designlab/stage2/s2load.py`\n"
          "- `tests/test_stage2_tbenv.py`, `tests/test_stage2_schedules.py`, `tests/test_stage2_s2load.py`",
    restore="# nothing to restore: no arrays or models are needed",
    needs="none. Wave 1: start now.",
    body="""## API (frozen; ask the lead before changing a name or a signature)

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
```""")

WPS["CV"] = dict(
    title="worker processes: `vecpool.py`",
    read="design D13, D16 (why nothing may depend on the process), section 2.2 (`vecpool`), the section 4 row \"The worker count does not matter\".",
    goal="`PoolVecEnv`: a stable-baselines3 `VecEnv` that hosts N envs in W child processes, bit-identical to SB3's "
         "`DummyVecEnv` at any W, whose workers import neither torch nor stable-baselines3 and are started as a "
         "subprocess with a command line the caller chooses.",
    files="- `designlab/stage2/vecpool.py`\n- `tests/test_stage2_vecpool.py`",
    restore="# nothing to restore",
    needs="none. Wave 1: start now. (`train2` will pass `worker_cmd=[sys.executable, \"-m\", \"designlab.stage2.train2\", \"worker\"]`.)",
    body="""## API (frozen)

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
```""")

WPS["CB"] = dict(
    title="the Drive bundle: `bundle.py`",
    read="design D20, D22, section 2.2 (`bundle`), 3.10 (the manifest; the acceptance run is package CN's), the section 4 row \"Bundle\"; rules N3, N4.",
    goal="A tool that packs the git-ignored inputs a Colab clone lacks (model files; a frozen reference set's arrays) "
         "into a few deterministic `.tar` parts with a committed JSON manifest, and verifies and unpacks them. Also "
         "Track B's ignore file.",
    files="- `designlab/stage2/bundle.py`\n- `tests/test_stage2_bundle.py`\n- `stage2/tb/.gitignore`, exactly these lines: `work/`, `*.tar`, `*.zip`, `*.npz`, `*.pkl`, `*.mp4`, `*.tmp`, `*.part`, `.lock`, `logs/`, `partial/`, `traces/`, `checkpoints/`, `parts/`",
    restore="$PY tools/restore_inputs.py copy-from /home/user/mesozoic-labs/hardware/trex_form_study --group picks-models picks-arrays",
    needs="none for the tool (it is tested on Stage 1's files and on synthetic sets in `tmp_path`). Reading a real set's freeze manifest needs round A's `setio` (`setio.frozen`, `setio.load`): import it lazily and skip that test until it is on your branch.",
    body="""## API (frozen)

```python
def build(bundle_id, bodies, sets=("V6",), out=None, part_mb=20, root=None) -> dict   # parts under <out>/parts/, MANIFEST.json in <out>
def content_sha256(manifest: dict) -> str
def verify(manifest, root, parts_dir=None) -> dict      # every file under root against the manifest; with parts_dir also every part
def unpack(manifest, parts_dir, root) -> dict
# CLI: python -m designlab.stage2.bundle build --bundle-id ID --bodies B1,B2 --sets V6 --out stage2/tb/bundle/ID [--part-mb 20]
#      python -m designlab.stage2.bundle verify --manifest PATH --root DIR [--parts DIR]
#      python -m designlab.stage2.bundle unpack --manifest PATH --parts DIR --root DIR
```
- **What is packed** (from committed records only): per body the two model files `plant.model_files(body)` names
  (the case's `.xml` and `.json` under `round2/refine/<case>/models/`; read `designlab/stage1/plant.py:116-133` and
  do not trigger a rebuild: refuse if a file is missing), with the hashes the committed records hold where they
  hold them; per set and body the files of the set's latest freeze manifest (`refs/*.npz`, `transitions/*.npz`;
  never the look-format twins, never a file the freeze does not list). `sets=()` packs models only.
- **Parts**: `models-<body>.tar`; `<set>-<body>-<i>.tar` (files in sorted order, a new part when the next file
  would pass `part_mb`). **Tar rules** (so that the same files give the same bytes in any process and under any
  umask): `tarfile.open(..., "w", format=tarfile.USTAR_FORMAT)`, members in sorted order, each `TarInfo` with the
  study-relative path, `mtime 0`, `uid = gid = 0`, `uname = gname = ""`, `mode 0o644`, no compression, file bytes
  streamed.
- **MANIFEST.json** (schema `s2-bundle-2`, design 3.10): `content` (bundle id, bodies, models, sets, files, parts:
  everything hashed), `content_sha256 = s2io.sha256_json(content)` (canonical JSON, sorted keys; if `s2io` is not
  on your branch use `json.dumps(sort_keys=True, separators=(",", ":"))` and say so), and `provenance` (time,
  commit, dirty, rule text, the tool's own sha256, versions) **outside** the hash. Nothing is ever added to
  `content` after the build; upload receipts are another file (`UPLOADS.json`, the lead's).
- `verify` re-hashes; returns `{files_ok, files, first_problem, parts_ok}` and never raises on a mismatch (the
  notebook prints its result); `unpack` refuses, before writing anything, a part whose sha256 differs, a member
  that is not in `content.files`, a path that leaves `root` (absolute, `..`, a symlink member), and writes each
  file through a temporary name; it is idempotent (a file already there with the right hash is left).
- Paths written by `build` (`parts/*.tar`) must be git-ignored by your `.gitignore`; `MANIFEST.json` must not be.

## Tests to write (`tests/test_stage2_bundle.py`)

Every row of the design's section 4 table "Bundle": two builds of the same files in two subprocesses, one under
`umask 077`, give byte-identical parts and equal `content_sha256`; the content hash does not change when the JSON
file is rewritten with CRLF line endings, another indentation, or a changed `provenance`; a changed file, a missing
file, an extra tar member, a member with `..` and a truncated part are each refused or reported with the path;
`unpack` twice is a no-op; a part over the size splits and every file is in exactly one part; `git check-ignore`
(run in a scratch git repository holding your `.gitignore` at `stage2/tb/`) reports `work/x`, `bundle/b/parts/a.tar`,
`runs/r/jobs/j/checkpoints/model_1.zip`, `x.npz`, `x.pkl`, `clip_1.mp4`, `logs/a.log`, `a.part`, `partial/x.json`,
`traces/t.npz` as ignored and `bundle/b/MANIFEST.json`, `runs/r/jobs/j/run.json`, `gates/V6/b/g8a.json` as not
ignored.

## Acceptance

```
nice -n 15 $PY -m pytest -p no:cacheprovider tests/test_stage2_bundle.py
# a stand-in bundle of the two picks' model files (no set), in a temporary directory:
nice -n 15 $PY -m designlab.stage2.bundle build --bundle-id standin --bodies 75-M-H-10,60-N-T-10 --sets "" --out /tmp/<you>/standin
nice -n 15 $PY -m designlab.stage2.bundle verify --manifest /tmp/<you>/standin/MANIFEST.json --root . --parts /tmp/<you>/standin/parts
nice -n 15 $PY -m designlab.stage2.bundle unpack --manifest /tmp/<you>/standin/MANIFEST.json --parts /tmp/<you>/standin/parts --root /tmp/<you>/unpacked
git status --porcelain            # only your three files
```""")

WPS["CK"] = dict(
    title="the look measures: `lookread.py`",
    read="design D17, section 2.2 (`lookread`), 3.5 (series, window, four numbers, terms, too few steps), 3.7 (G9-ref (b), (c): what the gate will ask of your functions); rules U1-U5. Also `designlab/stage2/look.py` (`_fit_phase`, the hip-midpoint and head-pitch code near lines 546, 600-646) and `designlab/stage2/motion.py` (`BANDS`).",
    goal="The four look measures of the plan's reading 1 (bob, head pitch in the world, tail phase, toe-off) as pure "
         "functions of 50 Hz poses and 500 Hz foot loads, the comparison with a reference's values against the bands, "
         "and the reference's values by kinematic playback of a track. No bar module is edited; the wrapper that calls "
         "you inside an episode is another package's (`lookbar.py`, package CW).",
    files="- `designlab/stage2/lookread.py`\n- `tests/test_stage2_lookread.py`",
    restore="$PY tools/restore_inputs.py copy-from /home/user/mesozoic-labs/hardware/trex_form_study --group picks-models picks-arrays look",
    needs="`playback` needs round A's `bar._Played` (branch `wt/wp-b`): import `bar` inside the function; its test skips until `designlab.stage2.bar` imports on your branch. Everything else: wave 1, start now.",
    body="""## API (frozen)

```python
BANDS = motion.BANDS          # bob_rel 0.30, head_world_pitch_rms_deg 1.5, tail_phase_cycles 0.1, toe_off_rel 0.70 (read the dict's own keys)
MIN_STEPS = 4
def series(p, poses, loads, dt, every) -> dict
def reading(ser: dict, ta: float, tb: float) -> dict
def compare(own: dict, ref: dict) -> dict
def playback(p, track, run_dir, ta_ref: float, tb_ref: float) -> dict
def window(b, rec) -> tuple[float, float] | None
def sources() -> dict
```
- `series(p, poses, loads, dt, every)`: `p` a `plant.Plant`; `poses [n, nq]` full `qpos` every `every` physics
  steps (50 Hz: `every * dt = 0.02`); `loads [steps, 2]` the feet's vertical loads at every physics step. By
  `mujoco.mj_kinematics` on a private `MjData` of `p.m` per pose: `t50`; `hip_mid_z` (the mean height of the two
  hip joints' anchors, as `look.py` computes it); `head_pitch_world` (the head body's pitch in the world about
  home, `look.arrays`' definition); `foot_pitch [n, 2]` (each foot's pitch about its own lateral axis, toe down
  positive: the sign of the look files' `foot_pitch_solved`); `tail` (the tail-yaw joint angle). Contact by
  `readings.contact_flags` when round A's `readings` is on your branch (the 2 % load rule with 20 ms debounce;
  until then implement the same rule locally behind one function and say so), and from it `lift_offs` and
  `touchdowns` per foot (times). Joints and bodies are found by name, never by index (`-X` bodies have 16 hinges).
- `reading(ser, ta, tb)` (design 3.5 "Four numbers"): `bob_pp_m` (mean over the steps wholly inside the window, a
  step from one touchdown of either foot to the next, of max less min of `hip_mid_z`); `head_world_pitch_rms_deg`
  (RMS about the mean); `tail {phase_cycles, amp_rad}` (least squares `A sin(2 pi phi + psi) + c` with `phi` the
  own phase: 0 at each left lift-off, linear in time between; `phase_cycles = psi / 2 pi`); `toe_off_rad` (mean
  over the lift-offs in the window of the lifting foot's pitch at the lift-off instant, interpolated between
  poses); `steps`, `lift_offs` (counts), `window`.
- `compare(own, ref)`: terms `bob` (`|own / ref - 1| <= 0.30`), `head` (`own <= 1.5 deg`), `tail` (wrapped
  difference of the phases `<= 0.1` cycle; not read, passes, when `ref.tail.amp_rad < 0.02`), `toe_off`
  (`own >= 0.70 x ref`; not read when `ref.toe_off_rad < 0.02`); `not_read` (list); **`too_few_steps`**: when
  `ref.steps >= 4 and ref.lift_offs >= 4` and (`own.steps < 4` or `own.lift_offs < 4`): `bob`, `tail`, `toe_off`
  are False, `head` is still read, `too_few_steps True`, no exception. When the reference itself has fewer: raise
  `LookError` (a definition fault). `ok` = all four.
- `playback(p, track, run_dir, ta_ref, tb_ref)`: poses `bar._Played(track, run_dir).q(t_ref)` at reference times
  `ta_ref, ta_ref + 0.02 x rate, ...` (rate 1.05: the nominal commanded clock, so the result is on simulated
  time), loads from `_Played.loads(t_ref, weight)` at every physics step, then `series` and `reading`. `_Played`
  refuses a periodic turning gait: let its `ValueError` pass through as `LookError("not played: ...")`.
- `window(b, rec)` (rule U2): for a bar walk item (`b.cond.judge == "bar_walk"`) the last 10 s of the planned
  episode; for a turn item (`"bar_turn"`) `[t_c + at_s, t_c + at_s + 3 s]` with `t_c = rec.sim_time(b.command["ref_s"])`
  and `at_s` the condition's judged time (2.0 s for the quick turn; read it from the condition as `bar.py` does);
  else `None`. Uses only attributes; do not import `bar` at module level.
- `sources()`: `{path: sha256}` of this file.

## Tests to write (`tests/test_stage2_lookread.py`)

Synthetic series (no MuJoCo) for `reading` and `compare`: each band on both sides of its limit (bob ratio 0.69,
0.71, 1.29, 1.31; head 1.49 and 1.51 deg; tail phase difference 0.09 and 0.11 cycle, also across the wrap at 0.5;
toe-off 0.69 and 0.71 of the reference); segmentation by the robot's own touchdowns ignores a 10 ms shift of the
whole motion (same numbers); "not read" under 0.02 rad for tail amplitude and toe-off, listed and passing; **too
few own steps (3) with a reference of 6: bob, tail and toe-off fail, `too_few_steps` True, no exception**; a
reference with 3 steps raises `LookError`. With MuJoCo and the look ladder's arrays (skip without): `series` on a
kinematic walk built from a look-ladder rung's own poses reproduces `stage2/look/metrics.json`'s numbers for that
rung within G9-ref (b)'s limits (bob 5 %, head 0.1 deg, tail 0.01 cycle, toe-off 0.02 rad): if it does not, stop
and report the differences and the two definitions: rule Z2 sends this back to the lead. Degraded copies of that
playback (pelvis oscillation x 0.5; a 3 degree neck sinusoid; the tail delayed 0.2 cycle; foot pitch zeroed) each
fail exactly their term. A 16-joint (`-X`) plant builds its series when its model is restored (skip otherwise).

## Acceptance

```
nice -n 15 $PY -m pytest -p no:cacheprovider tests/test_stage2_lookread.py
git status --porcelain            # only your two files
```
In your report: the measures of each look-ladder rung you could read against `metrics.json` (a small table).""")

WPS["CE"] = dict(
    title="the env: `r5lib.py`, `r5core.py`, `r5env.py`, `g8a.py`",
    read="design D1, D3-D11, D16 (the three conditions on the env), D23; sections 2.2 (`r5lib`, `r5core`, `r5env`, `g8a`), 2.3, 3.1, 3.2, 3.3, 3.4, 3.7 (G8a's alignment and the \"identical\" rule); rules E1-E5, T1-T3, W1-W5, J1-J5, Z1. Code: `designlab/stage1/env.py` (all of it: you mirror its reset and step), `control.py` (`Tracker.__init__`, `reset`, `step`, `Track.joint`, `Track.info`, `run`), `speedfit.py` (`servo_delay`), `policy_rows.py:492-541` (`PolicyOnlyTracker`), `legs.py`.",
    goal="R5's and R6's training env for 14-joint and 16-joint plants, the reference library it plays (scripts over a "
         "frozen set, or over Stage 1's run directory for development), the core that the env and the policy rows "
         "share (layout, observation, action mapping, reward), and gate G8a's comparison. The env composes Stage 1's "
         "parts and edits none of them.",
    files="- `designlab/stage2/r5lib.py`, `designlab/stage2/r5core.py`, `designlab/stage2/r5env.py`, `designlab/stage2/g8a.py`\n"
          "- `tests/test_stage2_r5lib.py`, `tests/test_stage2_r5core.py`, `tests/test_stage2_r5env.py`, `tests/test_stage2_g8a.py`",
    restore="$PY tools/restore_inputs.py copy-from /home/user/mesozoic-labs/hardware/trex_form_study --group picks-models picks-arrays x",
    needs="round A's `setio.py` and `s2io.py` (package R, branch `wt/wp-r`; the lead merges it into your branch). `g8a.bar_form` and `g8a.short` need round A's `bar.py`: import it inside those functions; their tests skip until `bar` imports. Wave 2.",
    body="""## API (frozen: packages CT and CW build against it)

Exactly the listing of the design's section 2.2 under `r5lib.py`, `r5core.py`, `r5env.py` and `g8a.py`, and
`R5Config` as in 3.2. The interface of `Script`, `Library` and `Core` may not change without the lead.

## What to build, with pointers

1. **`r5lib`** (design 3.3, D3, D5): `Library(p, body, set_name, rate)` reads a frozen set through
   `setio.read_inputs(setio.set_dir(set_name, body), files, require_frozen=True)` and builds each task of
   `TASKS` (rule T1's table) as a plan of round A's track items with `setio.build_track(p, set_dir, plan, omega,
   tr_doc)`; `omega = control.omega0(p, q_home, crouch_frac=the set's)`. `set_name == "S1"`: `setio.stage1_view(
   stage1/runs/2026-11-08, body)` and `S1_TASKS` (`stand`, `walk`, `brisk`, `stop`: the plans of Stage 1's
   `env.TASKS` with those names). A task whose reference is absent or `available: false` is dropped and listed with
   its reason; `tasks` is the **resolved** table and `inputs` holds each used file's sha256 and entry sha256, the
   freeze index and manifest sha256 and `panels.key_context(body)["plant_inputs"]`. `Script`: every table of 3.3,
   each defined for any `t_ref >= 0` and continuing past `t_end` as `Track.joint` / `Track.info` do; `mid_spans()`
   without holds; `required_margin` as 3.3's last bullet. Iterate no `set` anywhere (rule: `PYTHONHASHSEED`
   independence).
2. **`r5core`** (3.1, 3.4, D7-D10): `layout(joints, heading=True)`; `Core` with the observation, the filter
   (`alpha = 1 - exp(-2 pi f x 0.02)`), `offsets` (returns `None` and exactly `1.0` for a zero filtered action with
   the head neutral), `targets` (R6), the per-physics-step accumulators and the reward of 3.4 with
   `REWARD_WEIGHTS`, `REWARD_SCALES` and `REWARD_RULE` equal to rule W2's table (a test compares the dicts with a
   literal copy of that table); `interference` on `legs.LegGap` (moving pairs for the ramp, all pairs for "below
   zero"); `HeadScript.drawn` (rule J4) and `HeadScript.fixed` (rule E5's H1); `R6Tracker` (as
   `policy_rows.PolicyOnlyTracker`, with `reset_at` for a mid-task start and the queue filled with the reference's
   joint targets). `r5core` must not import `r5env`.
3. **`r5env`** (3.2, D6, D11): construction with **every task's tracker built there, on the nominal model**;
   `reset` in 3.2's thirteen steps (nominal arrays restored before the script is chosen; the chosen tracker's
   `damp` and `fric` asserted equal to the nominal copies); `step` in 3.2's seven; `R5Tracker.reset_at` (D6: lend
   the track `q0`, `qd0` for one `super().reset("frame", ...)`, restore, set `tau_ref`); the delay through
   `speedfit.servo_delay(p, k)` with its queue refilled for a mid start; randomisation J1-J5 from the env's own
   generator in the stated order; `set_task_weights`; `describe()` without any path; `env_configs` (floors by D13);
   `throughput`. `R6Env` per D8 (the mid-start action history).
4. **`g8a`** (3.7): `compare` (the pure rule on two digest lists and two end steps); `stock(body, condition,
   seed, k, seconds)`: the env on `Library(p, body, "S1").from_plan(<the stock condition's plan from
   panels.Planner>)` with the zero action against `panels.tracker_rows(body, run_dir, ("R0",))["R0"]`'s episode
   inside `speedfit.servo_delay(b.p, k)`, digests through a wrapper lent to the built condition's `recorder` (an
   instance attribute for the one episode, removed in a `finally`); `bar_form` and `short` the same against
   `bar.rows(body, ["R0@<ms>ms"])` on a set (lazy `bar` import). Alignment exactly as 3.7's table; the env's
   length for a plan is `ceil(round(seconds / dt) / 10)` env steps.

## Tests to write

The rows of the design's section 4 table that name your four test files: G8a (S1 view; a length that is no
multiple of 20 ms; the compare rule on tables); a zero action is not "almost" zero; **reset is history-free**
(an env after 6 randomised episodes against a fresh env, digest per physics step, hard and carpet, stance and mid
starts, 14 and 16 joints) and trackers built nominal; mid-task start and queue (three tests) and never in a hold;
layouts (88/130/15, 102/146/17; 87/129, 101/145 without the heading; joints by name); `-X` env steps where
`env.Stage1Env` still refuses; commands, heading, phase, tables past the end at clock factor 1.25; the resolved
task table and absent tasks on a synthetic set in `tmp_path` (build a tiny frozen set from Stage 1 arrays with
`setio`'s own helpers, kinematic only); the reward's rows (each term in [0, 1]; a kinematic playback scores
imitation 1 and penalties 0; each term moves when its cause is injected; weights and scales equal the rule's
table); R6 on 14 and 16 joints and its mid-start history; factors applied and restored. Slow (`STAGE2_SLOW=1`,
S1 view only): a mid start walks on for 4 s from 3 frames on `75-M-H-10`.

## Acceptance

```
nice -n 15 $PY -m pytest -p no:cacheprovider tests/test_stage2_r5lib.py tests/test_stage2_r5core.py
nice -n 15 $PY -m pytest -p no:cacheprovider tests/test_stage2_r5env.py tests/test_stage2_g8a.py
nice -n 15 $PY -c "from designlab.stage2 import g8a; r = g8a.stock('75-M-H-10', 'walk/brisk', 3042, 5, 6.0); print(r['identical'], r['compared_steps'])"   # True 3000
nice -n 15 $PY -c "from designlab.stage2 import r5env; c = r5env.env_configs('R5', '75-M-H-10', 'S1')[0]; print(r5env.throughput('R5', c, 2000))"            # env steps/s, hard floor; then the carpet env (index 3)
git status --porcelain            # only your eight files
```
Report the two throughput numbers and the resident memory of one env process; they replace the design's
estimates of 350 and 290 env steps/s.""")

WPS["CT"] = dict(
    title="the trainer: `train2.py`, `monitor2.py`, `clip2.py`",
    read="design D2, D12-D16, D22, D23; sections 2.1 (the entry snippet), 2.2 (`train2`, `monitor2`), 3.6 (all of it), 3.9 (the launcher's contract with the trainer: files, exit codes; clips are the launcher's), the section 4 rows on kill and resume, lesson 16, `MUJOCO_GL`, the monitor; rules O1-O4, M1-M6, M3b, K1-K5, K8, K9. Code: `designlab/stage1/train.py` (what you mirror, not edit), `policy.py`.",
    goal="The training entry point for R5 and R6, runnable here and on Colab: 8 envs in W workers, checkpoints as intact "
         "sets with the run state, an exact resume, the probes, the 18-episode monitor with its gain, G8b and stop "
         "rules, the fine-tune, and a clip tool. `train2` is also the entry of the worker and monitor child processes.",
    files="- `designlab/stage2/train2.py`, `designlab/stage2/monitor2.py`, `designlab/stage2/clip2.py`\n"
          "- `tests/test_stage2_train2.py`, `tests/test_stage2_monitor2.py`",
    restore="$PY tools/restore_inputs.py copy-from /home/user/mesozoic-labs/hardware/trex_form_study --group picks-models picks-arrays",
    needs="packages CE (`r5env`, `r5core`, `g8a`), CP (`tbenv`, `schedules`, `s2load`), CV (`vecpool`) merged into your branch. `monitor2` calls `lookbar.episodes` (package CW) through a lazy import: build and test against a stub of it (below). Wave 3.",
    body="""## API (frozen)

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
resident memory.""")

WPS["CW"] = dict(
    title="rows for the bar runner: `lookbar.py`, `s2rows.py`",
    read="design D15, D17, D18, D23; sections 2.2 (`lookbar`, `s2rows`), 2.3 (the last paragraph: what is relied on from `bar`), 3.5 (all of it: where the reading runs, the sidecar, the companion record, the outcomes), 3.11; rules U4-U6, M1, K9, E5. Code: round A's `designlab/stage2/bar.py` (`Delayed`, `rows`, `BarBuilt`, `BarContext`, `run_episode`, `_judge`, `Unreadable`, `counts`, `evaluate`, `BarWorker`, `_Played`), `designlab/stage1/panels.py` (`Row`, `TrackerRow`, `Policy`, `_Acting`, `register_rows`, `row_from_spec`), `policy_rows.py` (how Stage 1 built a policy row; you do not import it).",
    goal="Everything that lets Track B's controllers run through round A's bar runner without editing it: `LookRow` (a "
         "wrapper that gives any row the look reading), the `R0L` row, in-process episodes for the monitor, the "
         "companion records that hold reading 1, and the policy rows `R5/<seed>`, `R6/<seed>` and `R5G/<seed>` "
         "loaded from a checkpoint's files under the cross-version rule (S2-12).",
    files="- `designlab/stage2/lookbar.py`, `designlab/stage2/s2rows.py`\n- `tests/test_stage2_lookbar.py`, `tests/test_stage2_s2rows.py`",
    restore="$PY tools/restore_inputs.py copy-from /home/user/mesozoic-labs/hardware/trex_form_study --group picks-models picks-arrays",
    needs="round A's `bar.py` (package B) and packages CE (`r5lib`, `r5core`; `r5env` only inside `equivalence_check`), CP (`s2load`), CK (`lookread`) merged into your branch. Wave 3.",
    body="""## API (frozen)

Exactly the listing of the design's section 2.2 under `lookbar.py` and `s2rows.py`.

## What to build, with pointers

1. **`LookRow`** (3.5 "Where it runs"): `runs`, `gated`, `hard_only` are the inner row's; `name` the inner's;
   `episode`: `b = ctx.built(cond)`; for a bar walk or turn item (`bar.is_bar(cond)` and `cond.judge in
   ("bar_walk", "bar_turn")`) set the instance attribute `b.recorder = lambda: metrics.Recorder(
   legs_every=tune.LEGS_EVERY)`, call `self.inner.episode(ctx, cond, seed, relax)`, and in a `finally` delete the
   attribute (so the class's method is back even after an exception); then `lookread.window(b, rec)`,
   `lookread.series(b.p, rec.poses, rec.loads, rec.dt, tune.LEGS_EVERY * control.FEEDBACK_EVERY)`, `reading`, the
   reference by `lookread.playback(b.p, b.track, b.track.run_dir, ...)` over the window's reference times (cached
   per condition on the row object in the worker), `compare`; the clock block from the recorder's `t_ref` column
   (mean and min factor: the slope of `t_ref` against `t` over 1.05; `t_ref_reached`; `t_ref_planned = 1.05 x
   planned seconds + the track's start`); `policy_check` from `inner.provenance()` when the inner row has it;
   `self.last = block`; the sidecar written atomically when `sidecar` is set (path and schema in 3.5); return the
   inner's `(b, out, rec)` **unchanged**. A fall, a window not reached, or `LookError("not played")` gives a block
   with `look: None` and its reason, never an exception out of `episode`. The wrapper never touches the plant.
   `describe()` = the inner's with `look_reading: {bands, rule, look_sets}`; `sources()` = the inner's plus
   `lookbar.py` and `lookread.py`; `spec`/`from_spec` so that `panels.row_from_spec` rebuilds it in a worker.
2. **`R0L`**: `panels.register_rows("R0L", factory)` at import; the factory returns
   `LookRow(panels.tracker_rows(body, run_dir, ("R0",))["R0"], name="R0L")`.
3. **`rows(body, names, sidecar)`**: like `bar.rows` (parse `<row>@<ms>ms`, build the inner through
   `panels.tracker_rows`, which finds registered controllers), wrapping a row that is not already a `LookRow`, then
   `bar.Delayed(look_row, k, name)`; the sidecar root is given to each `LookRow`.
4. **`episodes(body, set_name, rows, pairs, traces)`** (the monitor's entry): one process; the set read as
   `bar.BarWorker` reads it (`setio.read_inputs(..., require_frozen=True)`, `bar.context_for`); conditions by key
   from `bar.conditions(set_docs, every=True)`; per pair `bar.run_episode(row, ctx, cond, seed,
   trace_root=traces)`, catching `bar.Unreadable` and any other exception separately; the companion episode (3.5
   "Outcome of an episode": `read`, `fell`, `unread` only when the bar raised `Unreadable`, the episode did not
   fall and the row is a policy row, with reason `clock_behind` when its mean clock factor is below 1 and
   `unreadable` otherwise; else `error`). A policy row is one whose inner row has `provenance()`. On set `S1` (development): stock conditions
   from `bar.stock_conditions()`, `panels.judge`'s record, `look: None`.
5. **`run`** and **`collect`** (3.5 "Companion record"): `run` calls `bar.evaluate(set_name, body, out, rows=rows(
   ..., sidecar=<out>/<set>/<body>/<block>/look_partial), conditions=..., block=..., procs=..., attribute="none")`
   then `collect`; `collect` reads each finished bar record of a `LookRow` row, joins its episodes with their
   sidecars (refusing a missing sidecar or one whose `steps`, `fell`, `survival_s` differ), classifies outcomes
   (an episode the bar recorded as failed to run with an "Unreadable" reason is re-classified by the sidecar's
   clock block), computes `counts_look`, `counts`, `counts_but_A1` with `bar.counts` on copies of the episodes
   whose `succeeded` fields are the reading's, and writes `look/<row>/<condition>.json` (schema `s2-look-1`) and a
   summary. Reading 1 counts only on `LOOK_SETS`.
6. **`s2rows`** (3.11): `S2Policy(panels.Policy)` (`begin`: `r5lib.Library(...).from_plan(built.plan)` on the
   built track, a `r5core.Core` on the tracker's estimator and clock; `act`: `model.predict(s2load.normalize(stats,
   obs), deterministic=True)`, `core.filter`, `tk.residual, tk.rate_factor = core.offsets(a_f)`), loaded lazily
   through `s2load.load_checkpoint(..., expect_run=...)`; `provenance()` with both version sets, `cross_version`,
   the fingerprint difference and tolerance; `R5Row`, `R6Row`, the `R5G` row with `HeadScript.fixed`; the three
   build-time checks of 3.11 (fingerprint and run; the run's recorded hashes of `r5core.py`, `r5lib.py`,
   `lookread.py`, `policy.py`, `env.py` and the panels' sources against today's, refused or flagged under
   `STAGE2_ALLOW_SOURCE_CHANGE=1`; the set and freeze by rule K4); `describe()` exactly 3.11's keys (no measured
   value); `sources()`; `spec`/`from_spec` with paths and hashes; the row refuses a condition whose set is not its
   run's. The factories registered for "R5", "R6", "R5G" return the row inside `LookRow`. `s2rows` imports neither
   `r5env` nor `train2` at module level; `equivalence_check` imports `r5env` inside the function.

## Tests to write

`tests/test_stage2_lookbar.py`: every item of the design's section 4 row "The look reading and its wrapper" that
names it: on the S1 view `Delayed(LookRow(R0), k)` against `Delayed(R0, k)` at k = 0 and 3: digest per physics
step equal (a digest wrapper as `g8a` uses) and `panels.judge`'s episode records equal; `BarBuilt.recorder` is the
class's method again after an episode that raised; the sidecar exists before `run_episode` returns; `collect` on
synthetic bar records and sidecars in `tmp_path` (counts by `bar.counts`; an `unread` episode survived and
not succeeded in all three readings; a missing or inconsistent sidecar refused; a non-look set gives
`counts_look: null`); `spec` round trip through `panels.row_from_spec`; `bar._Played` has `q` and `loads`;
`BarBuilt.recorder` is looked up on the instance. `tests/test_stage2_s2rows.py` (needs torch; a freshly
initialised policy saved through `s2load` by a helper): `describe()` holds both version sets and no float from a
measurement; a checkpoint record naming other versions gives `cross_version: true`; a foreign `run_uuid`, a
perturbed fingerprint and a changed `r5core.py` hash are each refused (the last accepted and flagged under the
environment variable); building and running a row imports neither `r5env` nor `train2` (subprocess,
`sys.modules`); slow (`STAGE2_SLOW=1`, S1 view): `equivalence_check` at k = 5 on `75-M-H-10` and, when its model
is restored, on `75-M-H-10-X`; the R6 row against `R6Env`.

## Acceptance

```
nice -n 15 $PY -m pytest -p no:cacheprovider tests/test_stage2_lookbar.py tests/test_stage2_s2rows.py
STAGE2_SLOW=1 nice -n 15 $PY -m pytest -p no:cacheprovider tests/test_stage2_s2rows.py -k "equivalence or r6"     # S1 view only
git status --porcelain            # only your four files
```
On a real set (`lookbar run --set V0 ...`, `equivalence_check` on V0 or V6): only after the lead says that task
3's first unit has run (V0) or that G9-ref passed (a look set); each such episode goes to the lead for
`stage2/tb/dev_episodes.json`.""")

WPS["CF"] = dict(
    title="R2L's fit: `r2lfit.py`",
    read="design D21, D22, D23; sections 2.2 (`r2lfit`), 3.8 (all of it); rules Q1-Q6, S2-18 in `STAGE2_PLAN.md` section 9, `STAGE1B_PLAN.md` section 3. Code: `designlab/stage1/speedfit.py` (all of it: you repeat its fit on other episodes without editing it), `tune.py` (`CMA`, `Pool`, `score`, `measure`, `training_order`, `derived_seed`, `selection_key`, `to_unit`, `from_unit`, `repair_penalty`), round A's `bar.py` (`BarContext`, `conditions`, `judge`, `run_episode`).",
    goal="Stage 1b's speed fit (CMA-ES over the tracker's 13 gains) repeated on the bar's tasks with a fixed rotation of "
         "delays and directions (S2-18), its selection with the 10 ms cell decisive, its budget ledger, and the rows "
         "`R2L-<set>/<seed>` for the bar runner. `speedfit.py` is imported, never edited.",
    files="- `designlab/stage2/r2lfit.py`\n- `tests/test_stage2_r2lfit.py`\n- `stage2/tb/r2l/index.json` (created as `{\"schema\": \"s2-r2l-index-1\", \"fits\": []}`)",
    restore="$PY tools/restore_inputs.py copy-from /home/user/mesozoic-labs/hardware/trex_form_study --group picks-models picks-arrays",
    needs="round A's `bar.py` (package B) and a built set for the real evaluator (round A's S); package CW's `lookbar.LookRow` for `r2l_row` (import inside the factory). Wave 3.",
    body="""## API (frozen)

Exactly the listing of the design's section 2.2 under `r2lfit.py`.

## What to build, with pointers (design 3.8 is the specification; rule ids in brackets)

1. **`rotation(seed, generations=30)`** [Q2]: per generation g, block `B = g // 6`: the episode seed
   `tune.training_order(seed)[g]`; per slot j = 0..4 (`walk/v0.4`, `turn/q90_<side>`, `stop/two_step`,
   `nudge/<direction>`, `walk/nominal`) the delay `k = (g + j) % 6`; side `("left", "right")[B % 2]`; direction
   `("forward", "left", "backward", "right")[B % 4]`; start and noise seeds `tune.derived_seed(ep, j, 0 | 1)`.
   Written to `rotation.json` with `written_at` before generation 0; a resume refuses a table that differs.
2. **`Evaluator`** (a `tune.Pool` worker; top-level factory `_evaluator`): builds the five bar conditions of the
   generation through a `bar.BarContext` on the fit's set, runs each as the bar's R2 row runs it
   (`panels.TrackerRow` in mode R2 with the candidate's gains) inside `speedfit.servo_delay(b.p, k)`, judges with
   `bar.judge`, and returns the objective's inputs. Every member of a generation runs the same five episodes.
3. **Objective `stage2-r2l-v1`** [Q3]: `tune.score` with the item terms of 3.8 (each graded on [0, 1]).
4. **`fit`**: `tune.CMA`, population 12, sigma0 0.1, from `speedfit.start_record(body)`'s gains
   (`speedfit.gains_vector`), seed 42; resumable per generation (`cma.json`, `evals.jsonl`, as
   `speedfit.RESUME_RULE`); the ledger `ledger.json` sums evaluation CPU seconds; the fit ends at the last whole
   block of six generations inside `budget_core_h` (2.4), or at 30 [Q5].
5. **`select`** [Q4]: candidates = the start and the mean after each third of the generations judged. First, and
   always whole even if the ledger passes its reserve (record `overrun`): every candidate on validation seeds
   2000-2019 at k = 5, five kinds, turn side by seed parity, nudge direction by seed mod 4; rank by
   `tune.selection_key`. Then reported cells while `reported_core_h` (0.4) lasts: the selected candidate and the
   start at k = 0, 2, 4, 1, 3 on seeds 2000-2004; a cell not reached is `{"status": "not run: budget"}`.
   `r2lg5.json` holds `selection`, `by_delay`, `judged_at_generation`, the gains, the rules.
6. **`publish`**: copies the finished records from the work directory to `stage2/tb/r2l/<set>/<body>/`, verifies
   each by sha256, appends `{set, body, seed, r2lg5: {path, sha256}}` to `index.json`.
7. **Rows** [Q6]: at import, for each set in `index.json`, `panels.register_rows(f"R2L-{set}", r2l_row)`;
   `r2l_row(body, run_dir, name)` finds the entry for (set from the name, body, seed from the name), checks the
   gains file's sha256, and returns `lookbar.LookRow(panels.TrackerRow(name, "R2", gains, gains_info, ...))`;
   `describe()` carries the gains file's path and sha256; the row refuses a condition whose set is not its fit's.
8. Outputs default to `stage2/tb/work/r2l/<set>/<body>/` (ignored); nothing live is written elsewhere.

## Tests to write (`tests/test_stage2_r2lfit.py`; a stub evaluator that scores from a formula, no physics)

Every item of the design's section 4 row "R2L": the rotation **per (kind, direction)**: left and right turns each
meet all six delays within two blocks, each nudge direction within four blocks, every slot meets all six delays
within every block; `rotation.json` precedes generation 0 and a changed table is refused at a resume; every member
of a generation receives the same episode list; the selection's decisive cell runs before any reported cell and
finishes when the ledger is forced past its reserve; a reported cell beyond the share reads "not run: budget"; a
fit cut by its budget ends at a multiple of six generations; `publish` then `r2l_row("...", ..., "R2L-V6noF5/42")`
resolves, and refuses another set, a missing entry and a changed gains file; the sha256 of
`designlab/stage1/speedfit.py` is the same before and after importing `r2lfit`; a killed fit resumes at its next
generation with the same CMA state.

## Acceptance

```
nice -n 15 $PY -m pytest -p no:cacheprovider tests/test_stage2_r2lfit.py
nice -n 15 $PY -m designlab.stage2.r2lfit rotation --body 75-M-H-10 --set V6noF5 --seed 42 --out /tmp/<you>/r2l      # prints the table
git status --porcelain            # only your three files
```
The real evaluator's smoke (two generations of population 4 on set V0, `--out /tmp/...`, no record kept) runs
**only after the lead says that task 3's first unit has run**; report its CPU seconds per evaluation (the budget
of rule Q5 rests on 4.4-11 core-seconds per episode).""")

WPS["CG"] = dict(
    title="gates and validation records: `gates2.py`, `run_trackb.sh`",
    read="design D17, D18, D22, D23; sections 2.2 (`gates2`), 3.7 (all of it), section 4 (the \"record\" entries and the reviewer's list), section 6 (the run plan, its order and its layout); rules Z1, Z2, K3, O2-O4, X1.",
    goal="The committed records of the gates and of the validation: G8a per body and head, G9 in its two parts, rates "
         "and memory, kill and resume on the real env, the training start with `MUJOCO_GL` set against it, R6's "
         "start; and the idempotent chain that runs the plan's steps in order.",
    files="- `designlab/stage2/gates2.py`\n- `tests/test_stage2_gates2.py`\n- `stage2/tb/run_trackb.sh`\n- the records your commands write under `stage2/tb/gates/` and `stage2/tb/validation/` (committed by the lead as they are made)",
    restore="$PY tools/restore_inputs.py copy-from /home/user/mesozoic-labs/hardware/trex_form_study --group picks-models picks-arrays look x",
    needs="packages CE, CP, CV, CT, CW, CK merged; round A's `bar.py`; for the committed records the frozen look sets (V6 on each body and head trained). Wave 4.",
    body="""## API (frozen)

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
design's section 6 gives each step's estimate. Report every record's path, sha256 and pass or fail.""")

WPS["CN"] = dict(
    title="the notebook: `nb.py`, `stage2_training.ipynb`, the steps sheet",
    read="design D19, D20, D22, D24, D25; sections 2.2 (`nb`), 3.9 (all of it), 3.10 (the acceptance run), 3.12 (the steps sheet's outline), the section 4 rows \"The notebook\" and \"No checkout dirt\"; rules K2, K4-K7, N1-N5, Y1-Y11, X1. Files: `/home/user/mesozoic-labs/notebooks/sb3_training.ipynb` (the scaffold: cell 3 lines 36-58, cell 8 lines 8-21 are what you adapt), `/home/user/mesozoic-labs/docs/investigations/stance_2026_10/stance3/colab_steps.txt` (the sheet's precedent), `tools/requirements-study.txt`.",
    goal="The owner's path: a thin Colab notebook whose cells each run one step of `python -m designlab.stage2.nb` in a "
         "subprocess; the queue launcher with its Drive copies, liveness fence and job rules; the collection of "
         "finished records; the tally of runtime-hours; the day-one smoke; the acceptance run in a fresh clone; and "
         "the steps sheet. Everything also runs here, with a local directory standing in for Drive.",
    files="- `designlab/stage2/nb.py`\n- `stage2/stage2_training.ipynb`\n- `stage2/tb/colab_steps.txt`, `stage2/tb/requirements-colab.txt`\n- `tests/test_stage2_nb.py`",
    restore="$PY tools/restore_inputs.py copy-from /home/user/mesozoic-labs/hardware/trex_form_study --group picks-models picks-arrays",
    needs="package CP's `tbenv` (wave 1). The launcher half (session, launch, copies, LIVE.json, job rules, collect, hours) starts in wave 1 against a stub trainer; the real cells need packages CT (`train2` CLI) and CB (`bundle`); the acceptance run needs the lead (a pushed launch commit and the built bundle). Wave 1 (launcher) and wave 4.",
    body="""## API (frozen)

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
nice -n 15 $PY -m designlab.stage2.nb acceptance --ref <C1, 40 hex> --parts stage2/tb/bundle/<id>/parts \\
    --scratch /home/user/mesozoic-wt/tb-accept --record stage2/tb/bundle/<id>/ACCEPTANCE.json          # about 12 minutes
```
and the steps sheet written from its record. Report the lines as printed, the kill point and the seconds of each
step.""")

for wp, spec in WPS.items():
    text = HEAD.format(wp=wp, design=DESIGN, rules=RULES, d=D, **spec)
    (D / f"WPC_{wp}.md").write_text(text)
    print(wp, len(text.splitlines()), "lines", len(text), "bytes")
