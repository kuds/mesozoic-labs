"""Shared pins and helpers of the r10 round's measuring tools (lane A: M1-M4).  Importable without torch, numpy or the
repository; it never imports `environments`.

Criteria: S2/r10round/lean/v5/criteria_r10round_lean_v5.txt (sha256 16f28878..., "the text"), which cites THIS12
(S2/r10round/v12/criteria_r10round_v12.txt) by line for unchanged definitions.  Every constant below names the line
it comes from.  PIN = a value the build fills or re-pins before approval (BUILD LIST 6, PINS).

v5 (RUNLOG line 657; tool updater 2026-10-26): no tool holds the branch HEAD (VALIDATIONS: "this text pins the
header's parent (cf9402bc, D19) and the tools' sha256s (PINS); the branch HEAD (the header commit) is recorded in
RUNLOG and <ROUND>/APPROVED, never inside a tool").  approved_branch_head() reads it from the approval record
<ROUND>/APPROVED (JSON, "branch_head", as score_r10round.py reads it) and refuses it unless it is a commit of the run
branch whose only parent is HEADER_PARENT and whose tree differs from that parent's only in comment lines of
configs/tyrannosaurus_rex/anatomical/stance.toml (the header commit: "this text's sha256 in a stance.toml comment").

The measuring copy (the text, HOW IT IS MEASURED): `git archive` of the branch HEAD the runs trained, extracted into a
directory of its own (never the clone), sealed by SEAL_NAME: the archive's commit (git get-tar-commit-id) and a tree
digest over every other file.  Every runner checks the seal before and after it measures, so a copy that changed (a
stray __pycache__, an edit) is refused.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
import tarfile
from datetime import datetime, timezone
from pathlib import Path

COMMON_VERSION = "r10m_common 1.1 (2026-10-26; v5: the branch HEAD from the approval record)"

S2 = "/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad"
SP = "/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad"
BR = S2 + "/r10round/runbranch/clone"                        # the text's BR (read only)
ROUND = SP + "/foot/stance6"                                  # the text's <ROUND> (D16)
PANELS = ROUND + "/panels"                                    # outputs: <ROUND>/panels/<RUN_ID>/<label>/
PY = S2 + "/venv/bin/python"
CRITERIA = S2 + "/r10round/lean/v5/criteria_r10round_lean_v5.txt"
CRITERIA_SHA256 = "16f288782d77e7e35c1b9f807094556050f793b729eaf26c040ce1583bb89794"
THIS12 = S2 + "/r10round/v12/criteria_r10round_v12.txt"
THIS12_SHA256 = "0ab5edeba04b319a4233a4700ebf9abf7d7d7127c30afc7565a7fa3bc34369ac"

# ---------------------------------------------------------------------------------------------------- the branch
BRANCH = "claude/screen-trex-r10-armltsb"
BASE_MAIN = "06b1a8e1fb940eec0d7a8e3af420c5b384ac0e8b"        # the text, WHAT IS TRAINED (Code)
#: The header's parent, D19 cf9402bc (RUNLOG line 646), pinned by the text (VALIDATIONS, PINS).  The branch HEAD the
#: runs train (the header commit) is never held here: approved_branch_head() reads it from the approval record.
HEADER_PARENT = "cf9402bc57dab92c716c7e0ee4bc96f72f237efb"
STOML_REL = "configs/tyrannosaurus_rex/anatomical/stance.toml"   # the text's STOML (Short names)
APPROVED = ROUND + "/APPROVED"                                # <ROUND>/APPROVED, the approval record (JSON)
SEAL_NAME = ".r10m_seal.json"

# ---------------------------------------------------------------------------------------------------- the plant and task
PHYSICS_REVISION = 10                                          # the text, WHAT IS TRAINED; HOW IT IS MEASURED
R10_PHYSICS_SHA256 = "sha256:6a47a0eb78164b9738752e2cf928aa0a6fd3ef98fc953eddc9bcb2eca366da4d"
R10_POLICY_INTERFACE_SHA256 = "sha256:ffbf6631e4de0469cfcaa6771e80f47973b127342bc362f74190451e8555cbd8"
POLICY_INTERFACE_REVISION = 15
TASK_SHA256 = "sha256:1e5964b61a4f0e3a76e9af2b6f76f485cf6e3d2409d48f32f3a7799626a6a456"   # V0 (b) "1e5964b6..."
HP_SHA256 = "sha256:054ab48303167eda1bd66c8894a64ef516bedd041a9e12f57dbd20e75c026b9c"     # V0 (b) "054ab483..."
COLLAPSE_REFERENCE = 4470.4                                    # D20 (a)

# ---------------------------------------------------------------------------------------------------- runs and checkpoints
SEEDS = (60, 42, 80)                                           # D2
RUN_IDS = {60: "stance6_r10LTSB_s60", 42: "stance6_r10LTSB_s42", 80: "stance6_r10LTSB_s80"}   # D16
FINAL_TIMESTEPS = 12001280
CHECKPOINT_CADENCE = 262144                                    # D3 (a)
GRID_1M = 1048576
J_STEPS = (4194304, 5242880, 6291456, 8388608, 10485760, "final")
R1_STEPS = 10485760

# ---------------------------------------------------------------------------------------------------- M1, M2, M3 (pinned files)
#: M1 TRACE: main's #638 trace tool (THIS12 lines 173-176), sha256 on the branch HEAD [measured, BR git show].
M1_SCRIPT_REL = "environments/shared/scripts/episode_trace_report.py"
M1_PINS = {
    "environments/shared/scripts/episode_trace_report.py":
        "b5923e5cfbcafd9f6e108df207a268500313de9bff44572121c8e3b48aad0af2",
    "environments/shared/reporting/episode_trace.py":
        "96ced68e124ddf4fa225ce32d93831a3e3261a2ba5b3ef502a0b900ce7b16c0b",
}
M1_SEED, M1_EPISODES = 3042, 20                                # seeds 3042-3061
#: M2 POSTURE PANEL: DRAFT tools/zs_harness_copy.py (THIS12 lines 181-193), a byte copy beside this file.
M2_HARNESS = "zs_harness_copy.py"
M2_HARNESS_SHA256 = "fbb71d22cfa40266188b1b836a454bb150dec589d724fb74d1cab53c1ef51828"
M2_SEED, M2_EPISODES = 3042, 40                                # seeds 3042-3081
#: M3 CERTIFICATE: main's stance_gate_report.py (THIS12 lines 194-201).
M3_SCRIPT_REL = "environments/shared/scripts/stance_gate_report.py"
M3_PINS = {M3_SCRIPT_REL: "bbfe80168c200332b0997181a1efbb13b01b522eb8b755203a85b5a0eeaf299b"}
M3_SEED, M3_EPISODES = 3042, 40
M3_HEADINGS = (-90.0, -45.0, 0.0, 45.0, 90.0, 180.0)

# ---------------------------------------------------------------------------------------------------- classes' inputs (M1)
HORIZON = 1000                                                 # FULL-HORIZON (1000-step)
SETTLE = 200                                                   # window steps 200..L-1
SAT_ABS = 0.99                                                 # BAR: |command| >= 0.99
NEAR_ABS = 0.90
BAR = 0.10                                                     # ... share exceeds 0.10
DRIVE = 0.60                                                   # DRIVE-BACKED / PARKED line
ACTUATOR_NAMES = (                                             # THIS12 lines 234-238 (MJCF lines 426-491)
    "neck_pitch_act", "neck_yaw_act", "head_pitch_act", "r_hip_pitch_act", "r_hip_roll_act", "r_knee_act",
    "r_ankle_act", "l_hip_pitch_act", "l_hip_roll_act", "l_knee_act", "l_ankle_act", "tail_1_pitch_act",
    "tail_1_yaw_act", "tail_2_pitch_act", "tail_3_pitch_act")
TAIL_ACTUATORS = (11, 12, 13, 14)
TAIL_PITCH_ACTUATORS = (11, 13, 14)
HIP_ROLL_ACTUATORS = (4, 8)
#: THIS12 line 958 (COMPUTE): "the runner refuses a batch below 3 GB free" (the text's BUILD LIST: "disk floor").
DISK_FLOOR_BYTES = 3 * 1024**3

# ---------------------------------------------------------------------------------------------------- K3 sigma (M4, M7)
FROZEN_PARAMS = SP + "/foot/symmetry/design/frozen_params.json"
FROZEN_PARAMS_SHA256 = "58b2c8fbb135ecddf1bf2212d5e243514d90ff1069f5f3e2cb1c3ece9c3e7af8"
SIDECAR = SP + "/foot/stance5/replication/sigma_s70_s80.json"   # the scratch original (REPL's judge reads it)
SIDECAR_SHA256 = "89778b3f81f83f05d4c15423b3658593dfaa767388213044c50a72e2d96b8522"
K3_SIGMA_REPO_REL = "docs/investigations/stance_2026_10/stance5/replication/sigma_s70_s80.json"   # the branch's copy
K3_SIGMA_REPO_SHA256 = "939285e30f377e696fef7bdcc30d6d61786e18c593ae39a1b50c936e3822a0ab"
K3_PANEL_SIGMA_KEY = {60: "source_vectors.60", 42: "source_vectors.42", 80: "control_sigma.80"}   # RUN HANDLING
K3_PANEL_SIGMA_FILE_COLAB = "/content/mesozoic-labs/" + K3_SIGMA_REPO_REL


class Refusal(SystemExit):
    """A refusal: the reason is printed and the process exits 2 (nothing measured, nothing written but the log)."""

    def __init__(self, message: str, code: int = 2):
        super().__init__(code)
        self.message = message


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------------------------------- measuring copy
def tree_digest(root) -> tuple[str, int]:
    """(sha256 over the sorted 'relpath<TAB>sha256' lines of every file under root but the seal, file count)."""
    root = Path(root)
    lines = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for fn in filenames:
            p = Path(dirpath) / fn
            rel = p.relative_to(root).as_posix()
            if rel == SEAL_NAME:
                continue
            if p.is_symlink():
                lines.append(f"{rel}\tsymlink:{os.readlink(p)}")
            else:
                lines.append(f"{rel}\t{sha256_file(p)}")
    lines.sort()
    return hashlib.sha256(("\n".join(lines) + "\n").encode()).hexdigest(), len(lines)


def build_copy(repo: str, commit: str, dest: str, *, tool: str) -> dict:
    """`git archive <commit>` of *repo* into *dest* (refused if it exists), then sealed.  Writes nothing in *repo*."""
    dest_p = Path(dest)
    if dest_p.exists():
        raise Refusal(f"{dest} exists: a measuring copy is built once (delete it first)")
    full = subprocess.run(["git", "--no-optional-locks", "-C", repo, "rev-parse", f"{commit}^{{commit}}"],
                          check=True, capture_output=True, text=True).stdout.strip()
    tar = subprocess.run(["git", "--no-optional-locks", "-C", repo, "archive", "--format=tar", full],
                         check=True, capture_output=True).stdout
    with tarfile.open(fileobj=io.BytesIO(tar)) as tf:
        archived = tf.pax_headers.get("comment")
        if archived != full:
            raise Refusal(f"the archive records commit {archived!r}, not {full}")
        part = Path(str(dest_p) + ".partial")
        if part.exists():
            raise Refusal(f"{part} exists (an interrupted build): delete it first")
        part.mkdir(parents=True)
        tf.extractall(part, filter="tar")
    digest, n = tree_digest(part)
    seal = {"schema": "r10m-measuring-copy/1", "commit": full, "repo": os.path.realpath(repo),
            "archive_sha256": sha256_bytes(tar), "tree_digest": digest, "files": n, "built_utc": utc_now(),
            "built_by": tool}
    (part / SEAL_NAME).write_text(json.dumps(seal, indent=1) + "\n")
    part.rename(dest_p)
    return seal


def check_copy(copy: str, *, expect_commit: str | None) -> dict:
    """The copy's seal after checking the tree still has its digest and (if given) the commit.  Refuses otherwise."""
    seal_p = Path(copy) / SEAL_NAME
    if not seal_p.is_file():
        raise Refusal(f"{copy} is not a sealed measuring copy ({SEAL_NAME} missing)")
    seal = json.loads(seal_p.read_text())
    if expect_commit is not None and seal.get("commit") != expect_commit:
        raise Refusal(f"measuring copy {copy} is commit {seal.get('commit')}, not the pinned {expect_commit}")
    digest, n = tree_digest(copy)
    if digest != seal.get("tree_digest") or n != seal.get("files"):
        raise Refusal(f"measuring copy {copy} changed: tree digest {digest} ({n} files) != sealed "
                      f"{seal.get('tree_digest')} ({seal.get('files')} files)")
    return seal


def check_pins(copy: str, pins: dict) -> dict:
    """{relpath: sha256} of the copy's pinned files; refuses a file whose sha256 is not its pin."""
    got = {}
    for rel, want in pins.items():
        p = Path(copy) / rel
        if not p.is_file():
            raise Refusal(f"{rel} missing from the measuring copy {copy}")
        got[rel] = sha256_file(p)
        if got[rel] != want:
            raise Refusal(f"{rel} sha256 {got[rel]} != the pinned {want}")
    return got


# ---------------------------------------------------------------------------------------------------- checkpoint records
def checkpoint_record(zip_path: str) -> dict:
    """What an SB3 checkpoint ZIP records about itself, read from its `data` member as JSON (nothing unpickled):
    the plant identity, seed, num_timesteps, the task fingerprint's task_sha256 and the symmetry record."""
    import zipfile

    out = {"path": os.path.realpath(zip_path), "sha256": sha256_file(zip_path), "readable": False}
    try:
        with zipfile.ZipFile(zip_path) as z:
            data = json.loads(z.read("data"))
    except Exception as exc:  # noqa: BLE001 - recorded; the caller refuses
        out["error"] = f"{type(exc).__name__}: {exc}"
        return out
    out["readable"] = True
    plant = data.get("_mesozoic_plant_identity")
    out["plant_identity"] = plant if isinstance(plant, dict) else None
    out["physics_revision"] = plant.get("physics_revision") if isinstance(plant, dict) else None
    out["physics_sha256"] = plant.get("physics_sha256") if isinstance(plant, dict) else None
    out["seed"] = data.get("seed")
    out["num_timesteps"] = data.get("num_timesteps")
    task = data.get("mesozoic_task_fingerprint")
    out["task_sha256"] = task.get("task_sha256") if isinstance(task, dict) else None
    sym = data.get("mesozoic_symmetry")
    out["symmetry"] = sym if isinstance(sym, dict) else None
    return out


def refuse_physics(record: dict, want: int = PHYSICS_REVISION) -> None:
    """The runners' physics_revision check (the text, HOW IT IS MEASURED): anything but *want* is refused."""
    if not record.get("readable"):
        raise Refusal(f"{record['path']}: the checkpoint's records cannot be read ({record.get('error')}); "
                      f"its physics_revision is unknown, not {want}")
    if record.get("physics_revision") != want:
        raise Refusal(f"{record['path']}: the checkpoint records physics_revision {record.get('physics_revision')!r}, "
                      f"not {want} (physics_sha256 {record.get('physics_sha256')!r}); refused")


# ---------------------------------------------------------------------------------------------------- outputs
def run_folder_of(path) -> Path | None:
    """The nearest ancestor of *path* (itself included) that holds provenance.json or is a stage folder 01_stance's
    parent: a notebook run folder.  None when there is none."""
    p = Path(os.path.realpath(path))
    for q in (p, *p.parents):
        if (q / "provenance.json").is_file() or (q / "01_stance").is_dir():
            return q
    return None


def refuse_output_in_run(out_dir, inputs) -> None:
    """Outputs never go into a run folder (the text, HOW IT IS MEASURED): refused when *out_dir* lies inside the run
    folder of any input, or inside any run folder at all, or inside the measuring copy."""
    out = Path(os.path.realpath(out_dir))
    for q in (out, *out.parents):
        if (q / "provenance.json").is_file() or q.name == "01_stance" or (q / SEAL_NAME).is_file():
            raise Refusal(f"output {out_dir} lies inside a run folder or a measuring copy ({q}); refused")
    for i in inputs:
        if i is None:
            continue
        rf = run_folder_of(i)
        if rf is not None and (out == rf or rf in out.parents):
            raise Refusal(f"output {out_dir} lies inside the run folder {rf} of {i}; refused")


def free_bytes(path) -> int:
    p = Path(path)
    while not p.exists():
        p = p.parent
    st = os.statvfs(p)
    return st.f_bavail * st.f_frsize


def refuse_below_disk_floor(path) -> int:
    free = free_bytes(path)
    if free < DISK_FLOOR_BYTES:
        raise Refusal(f"below the disk floor: {free / 1024**3:.2f} GB free under {path} (< 3 GB, THIS12 line 958); "
                      "nothing measured", 5)
    return free


def stack_versions() -> dict:
    """The interpreter's versions from package metadata (nothing imported)."""
    import importlib.metadata
    import platform

    out = {"python": platform.python_version()}
    for key in ("torch", "numpy", "gymnasium", "stable_baselines3", "mujoco"):
        try:
            out[key] = importlib.metadata.version(key)
        except importlib.metadata.PackageNotFoundError:
            out[key] = None
    return out


def child_env() -> dict:
    """The environment of a measuring subprocess: one thread, no bytecode, caches outside the copy."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON")}
    env.update({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                "PYTHONDONTWRITEBYTECODE": "1", "MPLCONFIGDIR": S2 + "/caches/mpl", "MPLBACKEND": "Agg"})
    return env


def write_json_x(path, obj) -> None:
    """Write JSON to a new file (refused if it exists)."""
    with open(path, "x") as fh:
        json.dump(obj, fh, indent=1, default=str)
        fh.write("\n")


# ---------------------------------------------------------------------------------------------------- the branch HEAD
def _git(repo: str, *args: str) -> str:
    return subprocess.run(["git", "--no-optional-locks", "-C", repo, *args], check=True, capture_output=True,
                          text=True).stdout


def approved_branch_head(approved: str | None = None, repo: str | None = None) -> dict:
    """The branch HEAD the runs train, read from the approval record (default <ROUND>/APPROVED, JSON key
    "branch_head") and checked in *repo* (read only): a 40-hex commit; exactly one parent, HEADER_PARENT (cf9402bc);
    the two trees differ in exactly one entry, STOML_REL, modified with its mode unchanged; every line the diff adds
    or removes is a TOML comment line (first non-blank character '#').  Refused (Refusal, exit 2) otherwise.
    Returns the head, the record's path and sha256 and the comment lines added and removed.  *repo*: the run-branch
    clone BR unless given (a test's clone)."""
    repo = repo or BR
    path = Path(approved or APPROVED)
    try:
        raw = path.read_bytes()
        rec = json.loads(raw.decode("utf-8"))
    except (OSError, ValueError) as exc:
        raise Refusal(f"no readable approval record at {path} ({type(exc).__name__}: {exc}): the branch HEAD is read "
                      "only from it (criteria lean v5 VALIDATIONS); nothing measured") from exc
    head = rec.get("branch_head") if isinstance(rec, dict) else None
    if not isinstance(head, str) or len(head) != 40 or any(c not in "0123456789abcdef" for c in head):
        raise Refusal(f"{path}: branch_head {head!r} is not a 40-hex commit")
    try:
        kind = _git(repo, "cat-file", "-t", head).strip()
        parents = _git(repo, "rev-list", "--parents", "-n", "1", head).split()[1:]
        raw_diff = _git(repo, "diff", "--raw", "--no-renames", "--no-abbrev", HEADER_PARENT, head).splitlines()
        patch = _git(repo, "diff", "--no-renames", "--no-ext-diff", "--no-textconv", "-U0", HEADER_PARENT, head,
                     "--", STOML_REL).splitlines()
    except (subprocess.CalledProcessError, OSError) as exc:
        raise Refusal(f"{path}: branch_head {head} cannot be read in {repo} ({exc})") from exc
    if kind != "commit":
        raise Refusal(f"{path}: branch_head {head} is a {kind}, not a commit")
    if parents != [HEADER_PARENT]:
        raise Refusal(f"branch_head {head} has parents {parents}, not exactly the header's parent {HEADER_PARENT}")
    entries = [ln.split("\t", 1) for ln in raw_diff if ln.strip()]
    if (len(entries) != 1 or entries[0][1] != STOML_REL or entries[0][0].split()[-1] != "M"
            or entries[0][0].split()[0] != ":" + entries[0][0].split()[1]):
        raise Refusal(f"branch_head {head} differs from {HEADER_PARENT} in {[e[-1] for e in entries]} "
                      f"({[e[0].split()[-1] for e in entries]}), not only in {STOML_REL} (mode unchanged)")
    added, removed, other = [], [], []
    for ln in patch:
        if ln.startswith(("+++", "---", "diff ", "index ", "@@", "\\")):
            continue
        if ln[:1] in "+-":
            body = ln[1:]
            (added if ln[0] == "+" else removed).append(body)
            if not body.lstrip().startswith("#"):
                other.append(ln)
    if other or not (added or removed):
        raise Refusal(f"branch_head {head}: {STOML_REL} changes lines that are not TOML comment lines: {other[:3]} "
                      f"(the header commit differs from {HEADER_PARENT} only in stance.toml comment lines)")
    return {"head": head, "header_parent": HEADER_PARENT, "approved": os.path.realpath(path),
            "approved_sha256": sha256_bytes(raw), "repo": os.path.realpath(repo), "file": STOML_REL,
            "comment_lines_added": added, "comment_lines_removed": removed}
