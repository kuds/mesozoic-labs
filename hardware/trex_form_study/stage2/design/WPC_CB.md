# Work package CB: the Drive bundle: `bundle.py`

You are building one part of a robotics simulation study (a robotic T. rex biped; everything is simulation). You
need no other context than this brief and the two texts it points into:
- the design: `/tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/design/ROUND_C_DESIGN.md` (read its header, the decisions and sections named under "Read first" below);
- the rules fixed before the runs: `/tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/design/PREREG_TRACKB_draft.md` (rule ids such as "K3" or "W2" refer to it).
Round A's design, where a round-A API is named: `/tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/design/ROUND_A_DESIGN.md`.

**Read first:** design D20, D22, section 2.2 (`bundle`), 3.10 (the manifest; the acceptance run is package CN's), the section 4 row "Bundle"; rules N3, N4.

## Goal

A tool that packs the git-ignored inputs a Colab clone lacks (model files; a frozen reference set's arrays) into a few deterministic `.tar` parts with a committed JSON manifest, and verifies and unpacks them. Also Track B's ignore file.

## Files you own (and no others)

- `designlab/stage2/bundle.py`
- `tests/test_stage2_bundle.py`
- `stage2/tb/.gitignore`, exactly these lines: `work/`, `*.tar`, `*.zip`, `*.npz`, `*.pkl`, `*.mp4`, `*.tmp`, `*.part`, `.lock`, `logs/`, `partial/`, `traces/`, `checkpoints/`, `parts/`

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
Needs from other packages: none for the tool (it is tested on Stage 1's files and on synthetic sets in `tmp_path`). Reading a real set's freeze manifest needs round A's `setio` (`setio.frozen`, `setio.load`): import it lazily and skip that test until it is on your branch.

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
```

## Commit

Commit on your own branch in your worktree, messages ending with the attribution lines your session's system
reminder gives, exactly; name no model anywhere else. Do not merge, do not push unless the lead says so. Final
report: files changed, each acceptance command with its exit code, every place where you read the design or a rule
in one of two possible ways (say which you chose: these are rules fixed before the runs, so say so rather than
decide silently), and anything the design asked for that you could not build.
