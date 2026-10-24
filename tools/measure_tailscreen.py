"""Tail_3 bound screen: panels (measure side).  Report only; never trains.  Under SCREEN/runs/ it only reads,
except the crash rule's rename (--retire-crashed).

    PY -I -B SCREEN/tools/measure_tailscreen.py --build-copy            # SCREEN/measure_copy (refused if it exists)
    PY -I -B SCREEN/tools/measure_tailscreen.py --run <name> --label <label>
    PY -I -B SCREEN/tools/measure_tailscreen.py --all-pending [--follow]
    PY -I -B SCREEN/tools/measure_tailscreen.py --validate-v1
    PY -I -B SCREEN/tools/measure_tailscreen.py --retire-crashed <name>  # the crash rule (run_waves.sh only)

The panel is the per-step trace tool as merged on main (#638), run from the measuring copy SCREEN/measure_copy: a git
archive of cf80ecb1 (the env, task and plant the runs trained on; read from the judge clone SP/fix/scratch_copband
with `git archive`, which writes nothing there) with the three trace files of origin/main overlaid (sha256 pinned in
screen_common.OVERLAYS).  Each panel is 20 deterministic episodes, seeds 3042-3061, window steps SETTLE..L-1:

    nice -n 19 PY -I -B measure_copy/environments/shared/scripts/episode_trace_report.py \
        --species tyrannosaurus_rex --embodiment anatomical --model <zip> --vecnorm <pkl> \
        --seed 3042 --episodes 20 --out-dir SCREEN/panels/<name>/<label>

Then: the measuring copy's tree digest is checked (before and after), episode_trace.json and the 20 per-step CSVs are
read, flags.csv is written, one row is appended to SCREEN/panels.csv under an fcntl lock (refused unless the
checkpoint zip still sits at its runs/ path with the sha256 measured, so a replicate retired by the crash rule while
its panel ran never gets a row), the CSVs' sha256 go to csv_sha256.txt, and the CSVs are gzipped (mtime 0, verified
against the recorded sha256 before the original is removed).

Per-episode flags (criteria HOW IT IS MEASURED): TAIL-UP = share of window steps with tail_3 command >= +0.99 above
0.10.  OTHER = the harm statistic (max over the 14 other actuators of the window share at |command| >= 0.99, and
tail_3's LOWER side share at command <= -0.99) above 0.10, or the episode shorter than 1000 steps.  The OTHER flag
does not read the TAIL-UP flag or tail_3's upper side, so a tail-up episode can also be an OTHER episode and adding
tail-up saturation never changes an episode's OTHER flag.

--retire-crashed <name> (the criteria's crash rule; called by run_waves.sh when a replicate's own process failed):
under the panels.csv lock, with one k free for both, it renames runs/<name> -> runs/<name>_crashed<k> (the only
write this tool makes under runs/, a rename) and panels/<name> -> panels/<name>_crashed<k>, and moves that
replicate's panels.csv rows to panels_crashed.csv (with an attempt column), so the rerun's checkpoints are measured
afresh and no row of the crashed attempt is ever read for a verdict.

Exit codes: 0 done; 2 refused (bad arguments, missing inputs, digest mismatch, already measured); 3 the trace tool
failed; 4 a consistency check between the trace JSON and the per-step CSVs failed; 5 paused: below 5 GB free disk;
6 --validate-v1 ran but a value differs from the criteria's V1.
"""
from __future__ import annotations

import os
import sys

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True

import argparse  # noqa: E402
import csv  # noqa: E402
import fcntl  # noqa: E402
import gzip  # noqa: E402
import hashlib  # noqa: E402
import importlib.util  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import tarfile  # noqa: E402
import time  # noqa: E402
import zipfile  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402

TOOLS = Path(__file__).resolve().parent


def _load_common():
    spec = importlib.util.spec_from_file_location("screen_common", TOOLS / "screen_common.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


C = _load_common()
SCREEN = Path(C.SCREEN)
MEASURE_COPY = SCREEN / "measure_copy"
PANELS_DIR = SCREEN / "panels"
PANELS_CSV = SCREEN / "panels.csv"
RUNS = SCREEN / "runs"
VALIDATE = SCREEN / "validate"
TRACE_SCRIPT_REL = "environments/shared/scripts/episode_trace_report.py"
DIGEST_EXCLUDE = {"OVERLAY_SHA256.txt", "TREE_DIGEST.txt"}
HORIZON = 1000
SAT = 0.99
NEAR = 0.90
FLAG_BAR = 0.10
MIN_FREE_BYTES = 5 * 1024**3
TAIL3_NAME = "tail_3_pitch_act"
TAIL1_NAME = "tail_1_pitch_act"
TAIL2_NAME = "tail_2_pitch_act"
JUDGE_CLONE = Path(C.SP) / "fix" / "scratch_copband"
MAIN_CHECKOUT = Path("/home/user/mesozoic-labs")
MAIN_REF = "5f5dbf6b43ff8a284a5a770ee6f941d2eb3e6fe5"  # origin/main at 17:50 UTC (merge of #638 and later)

PANEL_COLUMNS = [
    # identity
    "name", "state", "arm", "rep_seed", "coef", "b", "label", "num_timesteps", "zip_sha256", "pkl_sha256",
    "n_episodes", "seeds",
    # tail_3 command and saturation
    "frac99", "frac90", "cmd", "cmd_episode_sd",
    # tail_3 raw mean
    "raw_window_mean", "share_raw_gt_0p60", "share_raw_gt_0p90", "share_raw_gt_1", "raw_max_window",
    "pinned_step_raw_mean",
    # posture
    "q3_window_mean",
    # flags, falls and returns
    "tailup_flags", "other_flags", "other_reasons", "falls", "short_episodes", "return_mean",
    # other actuators and substitution
    "others_max_frac90", "others_max_frac99", "others_max_abs_raw_mean", "tail1_cmd", "tail2_cmd",
    "tail2_q_window_mean",
    # single-checkpoint class
    "class_single",
    # provenance
    "trace_json_sha256", "measure_tree_digest", "stack", "utc",
]
FLAG_COLUMNS = [
    "seed", "length", "terminated", "truncated", "return", "tail3_cmd_window_mean", "tail3_share_cmd_ge_0p99",
    "tailup_flag", "gate_stat", "gate_actuator", "gate_sign", "other_stat", "other_actuator", "other_sign",
    "other_flag", "other_reason", "fall",
]
CRASHED_CSV_NAME = "panels_crashed.csv"


class Refused(SystemExit):
    pass


def refuse(msg: str, code: int = 2):
    print(f"refused: {msg}", file=sys.stderr)
    raise SystemExit(code)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def seeds_text(seeds) -> str:
    seeds = [int(s) for s in seeds]
    if seeds and seeds == list(range(seeds[0], seeds[0] + len(seeds))):
        return f"{seeds[0]}-{seeds[-1]}"
    return ",".join(str(s) for s in seeds)


def stack_text() -> str:
    return json.dumps(C.stack_versions(), sort_keys=True, separators=(",", ":"))


# ----------------------------------------------------------------------------------------------- measuring copy

def tree_digest(root: Path) -> str:
    """sha256 over the sorted 'relpath<TAB>sha256' lines of every file under root (the two seal files excluded)."""
    lines = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for fn in filenames:
            p = Path(dirpath) / fn
            rel = p.relative_to(root).as_posix()
            if rel in DIGEST_EXCLUDE:
                continue
            if p.is_symlink():
                lines.append(f"{rel}\tsymlink:{os.readlink(p)}")
            else:
                lines.append(f"{rel}\t{sha256_file(p)}")
    lines.sort()
    return hashlib.sha256(("\n".join(lines) + "\n").encode()).hexdigest()


def overlay_check(root: Path) -> list[str]:
    problems = []
    for rel, want in C.OVERLAYS.items():
        p = root / rel
        if not p.is_file():
            problems.append(f"overlay {rel} missing")
        elif sha256_file(p) != want:
            problems.append(f"overlay {rel} sha256 {sha256_file(p)} != pinned {want}")
    return problems


def check_copy() -> str:
    """The sealed tree digest, after checking that the copy still has it.  Refuses otherwise."""
    seal = MEASURE_COPY / "TREE_DIGEST.txt"
    if not seal.is_file():
        refuse(f"{seal} missing: build the measuring copy first (--build-copy)")
    want = seal.read_text().split()[0]
    problems = overlay_check(MEASURE_COPY)
    if problems:
        refuse("; ".join(problems))
    got = tree_digest(MEASURE_COPY)
    if got != want:
        refuse(f"measuring copy tree digest {got} != sealed {want}: the copy changed")
    return want


def build_copy() -> int:
    if MEASURE_COPY.exists():
        refuse(f"{MEASURE_COPY} exists; it is built once")
    tmp = SCREEN / "measure_copy.partial"
    if tmp.exists():
        refuse(f"{tmp} exists (an interrupted build): move it away first")
    tmp.mkdir()
    archive = subprocess.run(
        ["git", "--no-optional-locks", "-C", str(JUDGE_CLONE), "archive", "--format=tar", C.PARENT],
        check=True, capture_output=True).stdout
    with tarfile.open(fileobj=io.BytesIO(archive)) as tf:
        tf.extractall(tmp, filter="tar")
    lines = []
    for rel, want in C.OVERLAYS.items():
        blob = subprocess.run(
            ["git", "--no-optional-locks", "-C", str(MAIN_CHECKOUT), "show", f"{MAIN_REF}:{rel}"],
            check=True, capture_output=True).stdout
        got = hashlib.sha256(blob).hexdigest()
        if got != want:
            refuse(f"origin/main {MAIN_REF[:8]}:{rel} sha256 {got} != pinned {want}")
        (tmp / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp / rel).write_bytes(blob)
        lines.append(f"{want}  {rel}")
    (tmp / "OVERLAY_SHA256.txt").write_text(
        "\n".join(lines) + f"\n# base: git archive {C.PARENT} of {JUDGE_CLONE}\n"
        f"# overlays: git show {MAIN_REF}:<path> in {MAIN_CHECKOUT}\n# built {utc_now()}\n")
    digest = tree_digest(tmp)
    (tmp / "TREE_DIGEST.txt").write_text(f"{digest}\n")
    tmp.rename(MEASURE_COPY)
    print(json.dumps({"measure_copy": str(MEASURE_COPY), "tree_digest": digest,
                      "overlays": dict(C.OVERLAYS), "base": C.PARENT}, indent=1))
    return 0


# ------------------------------------------------------------------------------------------------- reductions

def _f(x) -> str:
    """CSV cell for a float: repr (round-trips exactly); '' for None/NaN."""
    if x is None:
        return ""
    if isinstance(x, float) and math.isnan(x):
        return ""
    return repr(float(x)) if isinstance(x, float) else str(x)


def _mean(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    import numpy as np

    return float(np.mean(xs))


def read_episode_csv(path: Path, names: list[str]):
    import numpy as np

    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    out = {}

    def num(text: str) -> float:  # the trace tool writes a non-finite value as an empty cell
        return float(text) if text != "" else math.nan

    for key in ("command", "raw_mean", "joint_pos"):
        out[key] = np.array([[num(r[f"{n}_{key}"]) for n in names] for r in rows], dtype=np.float64).reshape(
            len(rows), len(names))
    out["step"] = np.array([int(r["step"]) for r in rows])
    out["in_window"] = np.array([int(r["in_window"]) for r in rows])
    return out


def reduce_panel(out_dir: Path) -> tuple[dict, list[dict], list[str]]:
    """(panel aggregates, per-episode flag rows, consistency problems) from a trace output folder."""
    import numpy as np

    trace_path = out_dir / "episode_trace.json"
    with open(trace_path) as fh:
        tr = json.load(fh)
    problems = []
    names = list(tr["actuator_names"])
    if tr.get("traced_actuators") != names:
        problems.append("not every actuator traced")
    if names[C.TAIL3] != TAIL3_NAME:
        problems.append(f"actuator {C.TAIL3} is {names[C.TAIL3]}, not {TAIL3_NAME}")
    if tr.get("deterministic") is not True or tr.get("stochastic_seed") is not None or tr.get("holds"):
        problems.append("trace is not the deterministic judge rollout (stochastic or hold)")
    if tr.get("checkpoint_plant_validated") is not True:
        problems.append("checkpoint_plant_validated is not true")
    if int(tr["settle_steps"]) != C.SETTLE:
        problems.append(f"settle_steps {tr['settle_steps']} != {C.SETTLE}")
    if int(tr["horizon"]) != HORIZON:
        problems.append(f"horizon {tr['horizon']} != {HORIZON}")
    if float(tr["saturation_abs"]) != SAT or float(tr["near_saturation_abs"]) != NEAR:
        problems.append("saturation thresholds differ from 0.99 / 0.9")
    if list(tr["seeds"]) != list(C.PANEL_SEEDS):
        problems.append(f"seeds {seeds_text(tr['seeds'])} != {seeds_text(C.PANEL_SEEDS)}")
    settle = C.SETTLE
    t3, t1, t2 = names.index(TAIL3_NAME), names.index(TAIL1_NAME), names.index(TAIL2_NAME)
    others = [i for i in range(len(names)) if i != t3]
    flags = []
    cm, rm, q3, sh60, sh90, sh1, wmax = [], [], [], [], [], [], []
    pinned_raw_sum, pinned_steps = 0.0, 0
    for e in tr["episodes"]:
        seed, L = int(e["seed"]), int(e["length"])
        ep = read_episode_csv(out_dir / f"episode_trace_seed{seed}.csv", names)
        if ep["command"].shape[0] != L or list(ep["step"]) != list(range(L)):
            problems.append(f"seed {seed}: CSV has {ep['command'].shape[0]} rows, JSON length {L}")
            continue
        if list(ep["in_window"]) != [int(s >= settle) for s in range(L)]:
            problems.append(f"seed {seed}: in_window column differs from steps >= {settle}")
        w = slice(settle, L)
        cmd, raw, jp = ep["command"][w], ep["raw_mean"][w], ep["joint_pos"][w]
        n = cmd.shape[0]
        a3 = e["actuators"][TAIL3_NAME]
        if n:
            f99 = (np.abs(cmd) >= SAT).mean(axis=0)
            pos99 = (cmd >= SAT).mean(axis=0)
            neg99 = (cmd <= -SAT).mean(axis=0)
            gate_stat = float(f99.max())
            gi = int(np.argmax(f99))
            pos, neg = float(pos99[gi]), float(neg99[gi])
            gate_act = names[gi] if gate_stat > 0 else ""
            gate_sign = ("+" if pos >= neg else "-") if gate_stat > 0 else ""
            # Harm statistic: every actuator but tail_3's upper side (tail_3 counts with its lower side only).
            harm = f99.copy()
            harm[t3] = neg99[t3]
            other_stat = float(harm.max())
            oi = int(np.argmax(harm))
            other_act = names[oi] if other_stat > 0 else ""
            if other_stat <= 0:
                other_sign = ""
            elif oi == t3:
                other_sign = "-"
            else:
                other_sign = "+" if float(pos99[oi]) >= float(neg99[oi]) else "-"
            tail_share = float((cmd[:, t3] >= SAT).mean())
            c3 = float(cmd[:, t3].mean())
            r3 = float(raw[:, t3].mean())
            # Consistency with the trace JSON (the CSV floats are repr, so these are equal to the last bit or near).
            jmax = e["max_actuator_saturation_fraction"]
            if jmax is None or abs(gate_stat - float(jmax)) > 1e-12:
                problems.append(f"seed {seed}: gate stat {gate_stat} != JSON {jmax}")
            if abs(float(f99[t3]) - float(a3["saturated_fraction"])) > 1e-12:
                problems.append(f"seed {seed}: tail_3 frac99 differs from JSON")
            if abs(c3 - float(a3["command_window_mean"])) > 1e-9:
                problems.append(f"seed {seed}: tail_3 window command {c3} != JSON {a3['command_window_mean']}")
            if abs(float(raw[:, t3].max()) - float(a3["raw_mean_max"])) > 1e-9:
                problems.append(f"seed {seed}: tail_3 raw max differs from JSON")
            cm.append(c3)
            rm.append(r3)
            q3.append(float(jp[:, t3].mean()))
            sh60.append(float((raw[:, t3] > 0.60).mean()))
            sh90.append(float((raw[:, t3] > 0.90).mean()))
            sh1.append(float((raw[:, t3] > 1.0).mean()))
            wmax.append(float(raw[:, t3].max()))
            pin = np.abs(cmd[:, t3]) >= SAT
            pinned_raw_sum += float(raw[pin, t3].sum())
            pinned_steps += int(pin.sum())
        else:
            gate_stat, gate_act, gate_sign, tail_share, c3 = None, "", "", None, None
            other_stat, other_act, other_sign = None, "", ""
        tailup = tail_share is not None and tail_share > FLAG_BAR
        short = L < HORIZON
        sat_other = other_stat is not None and other_stat > FLAG_BAR
        other = sat_other or short  # independent of the TAIL-UP flag (criteria: OTHER FLAG)
        reasons = []
        if sat_other:
            reasons.append(f"saturation:{other_act}{other_sign}")
        if short:
            reasons.append(f"short:{L}")
        flags.append({
            "seed": seed, "length": L, "terminated": int(bool(e["terminated"])),
            "truncated": int(bool(e["truncated"])), "return": float(e["return"]),
            "tail3_cmd_window_mean": c3, "tail3_share_cmd_ge_0p99": tail_share, "tailup_flag": int(tailup),
            "gate_stat": gate_stat, "gate_actuator": gate_act, "gate_sign": gate_sign, "other_stat": other_stat,
            "other_actuator": other_act, "other_sign": other_sign, "other_flag": int(other),
            "other_reason": ";".join(reasons), "fall": int(bool(e["terminated"]) and short),
        })

    def per_act(key, idx):
        return _mean([ep["actuators"][names[idx]][key] for ep in tr["episodes"]])

    agg = {
        "n_episodes": len(tr["episodes"]),
        "seeds": seeds_text(tr["seeds"]),
        "frac99": per_act("saturated_fraction", t3),
        "frac90": per_act("near_saturated_fraction", t3),
        "cmd": _mean(cm),
        "cmd_episode_sd": float(np.std(cm, ddof=1)) if len(cm) > 1 else None,
        "raw_window_mean": _mean(rm),
        "share_raw_gt_0p60": _mean(sh60),
        "share_raw_gt_0p90": _mean(sh90),
        "share_raw_gt_1": _mean(sh1),
        "raw_max_window": max(wmax) if wmax else None,
        "pinned_step_raw_mean": (pinned_raw_sum / pinned_steps) if pinned_steps else None,
        "q3_window_mean": _mean(q3),
        "tailup_flags": sum(f["tailup_flag"] for f in flags),
        "other_flags": sum(f["other_flag"] for f in flags),
        "other_reasons": ";".join(f"{f['seed']}:{f['other_reason']}" for f in flags if f["other_flag"]),
        "falls": sum(f["fall"] for f in flags),
        "short_episodes": sum(int(f["length"] < HORIZON) for f in flags),
        "return_mean": float(np.mean([float(e["return"]) for e in tr["episodes"]])),
        "others_max_frac90": max((per_act("near_saturated_fraction", i) or 0.0) for i in others),
        "others_max_frac99": max((per_act("saturated_fraction", i) or 0.0) for i in others),
        "others_max_abs_raw_mean": max(abs(per_act("raw_mean_window_mean", i) or 0.0) for i in others),
        "tail1_cmd": per_act("command_window_mean", t1),
        "tail2_cmd": per_act("command_window_mean", t2),
        "tail2_q_window_mean": per_act("joint_pos_window_mean", t2),
        "gate_flagged": sum(int(f["gate_stat"] is not None and f["gate_stat"] > FLAG_BAR) for f in flags),
    }
    agg["class_single"] = class_single(agg["tailup_flags"], agg["cmd"])
    agg["trace_json_sha256"] = sha256_file(trace_path)
    return agg, flags, problems


def class_single(tailup_flags: int, cmd) -> str:
    if tailup_flags >= 1:
        return "RAIL"
    if cmd is not None and cmd >= 0.60:
        return "PARKED"
    return "LOW"


# ------------------------------------------------------------------------------------------------- panels.csv

def read_panels(path: Path = PANELS_CSV) -> list[dict]:
    if not path.is_file():
        return []
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def append_row(path: Path, row: dict, verify=None) -> None:
    """Append one row under an exclusive fcntl lock; refuses a second row for the same (name, label).  `verify`, if
    given, is called under the lock and refuses (raises) when the row no longer describes the current attempt."""
    with open(path, "a+", newline="") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            if verify is not None:
                verify()
            fh.seek(0)
            text = fh.read()
            if text:
                rdr = csv.DictReader(io.StringIO(text))
                if rdr.fieldnames != PANEL_COLUMNS:
                    refuse(f"{path} has other columns than this tool writes")
                for r in rdr:
                    if r["name"] == row["name"] and r["label"] == row["label"]:
                        refuse(f"{path} already has a row for {row['name']} {row['label']}")
            w = csv.DictWriter(fh, fieldnames=PANEL_COLUMNS, lineterminator="\n")
            fh.seek(0, os.SEEK_END)
            if not text:
                w.writeheader()
            w.writerow({k: row[k] for k in PANEL_COLUMNS})
            fh.flush()
            os.fsync(fh.fileno())
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def has_row(name: str, label: str, path: Path = PANELS_CSV) -> bool:
    return any(r["name"] == name and r["label"] == label for r in read_panels(path))


# ------------------------------------------------------------------------------------------------- one panel

def checkpoint_pair(name: str, label: str) -> tuple[Path, Path]:
    d = RUNS / name
    if label == "final":
        return d / "stage1_final.zip", d / "stage1_final_vecnorm.pkl"
    return d / f"stage1_{label}_steps.zip", d / f"stage1_vecnormalize_{label}_steps.pkl"


def parse_name(name: str) -> tuple[str, str, int]:
    try:
        state, arm, r = name.split("_")
        seed = int(r[1:]) if r.startswith("r") else -1
    except ValueError:
        refuse(f"bad run name {name!r}")
    if state not in C.STATES or arm not in C.ARMS or seed not in C.REP_SEEDS:
        refuse(f"{name!r} is not one of the 18 screen names")
    return state, arm, seed


def all_names() -> list[str]:
    return [f"{s}_{a}_r{r}" for s in ("s42", "s80") for a in C.ARMS for r in C.REP_SEEDS]


def zip_num_timesteps(zip_path: Path) -> int:
    with zipfile.ZipFile(zip_path) as z:
        return int(json.loads(z.read("data"))["num_timesteps"])


def free_bytes(path: Path) -> int:
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize


def move_crashed(out_dir: Path) -> None:
    k = 1
    while (out_dir.parent / f"{out_dir.name}_crashed{k}").exists():
        k += 1
    dest = out_dir.parent / f"{out_dir.name}_crashed{k}"
    out_dir.rename(dest)
    print(f"crashed panel folder (no panels.csv row) moved to {dest}", file=sys.stderr)


def run_trace(zip_path: Path, pkl_path: Path, out_dir: Path) -> dict:
    cmd = ["nice", "-n", "19", str(C.PY), "-I", "-B", str(MEASURE_COPY / TRACE_SCRIPT_REL),
           "--species", "tyrannosaurus_rex", "--embodiment", "anatomical",
           "--model", str(zip_path), "--vecnorm", str(pkl_path),
           "--seed", str(C.PANEL_SEEDS[0]), "--episodes", str(len(C.PANEL_SEEDS)), "--out-dir", str(out_dir)]
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON")}
    env.update({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                "PYTHONDONTWRITEBYTECODE": "1"})
    t0 = time.monotonic()
    proc = subprocess.run(cmd, env=env, cwd=str(SCREEN / "logs"), capture_output=True, text=True)
    return {"argv": cmd, "returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr,
            "seconds": round(time.monotonic() - t0, 1)}


def gzip_csvs(out_dir: Path) -> None:
    csvs = sorted(out_dir.glob("episode_trace_seed*.csv"))
    sums = {p.name: sha256_file(p) for p in csvs}
    with open(out_dir / "csv_sha256.txt", "x") as fh:
        for n, s in sums.items():
            fh.write(f"{s}  {n}\n")
        fh.flush()
        os.fsync(fh.fileno())
    for p in csvs:
        gz = p.with_name(p.name + ".gz")
        with open(p, "rb") as src, open(gz, "xb") as raw:
            with gzip.GzipFile(filename=p.name, mode="wb", fileobj=raw, mtime=0, compresslevel=6) as dst:
                shutil.copyfileobj(src, dst)
        with gzip.open(gz, "rb") as fh:
            if hashlib.sha256(fh.read()).hexdigest() != sums[p.name]:
                refuse(f"{gz} does not decompress to the recorded sha256", 4)
        p.unlink()


def write_flags(out_dir: Path, flags: list[dict]) -> None:
    with open(out_dir / "flags.csv", "x", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FLAG_COLUMNS, lineterminator="\n")
        w.writeheader()
        for f in flags:
            w.writerow({k: _f(v) if isinstance(v, float) or v is None else v for k, v in f.items()})


def measure_one(name: str, label: str, *, zip_path: Path, pkl_path: Path, out_dir: Path, panels_csv: Path,
                identity: dict) -> dict:
    """Run, reduce and record one panel.  Returns the panels.csv row."""
    if free_bytes(SCREEN) < MIN_FREE_BYTES:
        print(f"PAUSED: below 5 GB free under {SCREEN} ({free_bytes(SCREEN) / 1e9:.2f} GB); nothing discarded. "
              "Tell the maintainer.", file=sys.stderr)
        raise SystemExit(5)
    for p in (zip_path, pkl_path):
        if not p.is_file():
            refuse(f"{p} missing")
    if has_row(name, label, panels_csv):
        refuse(f"{panels_csv} already has {name} {label}")
    if out_dir.exists():
        move_crashed(out_dir)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    digest = check_copy()
    zip_sha, pkl_sha = sha256_file(zip_path), sha256_file(pkl_path)
    res = run_trace(zip_path, pkl_path, out_dir)
    digest_after = check_copy()
    if res["returncode"] != 0 or not (out_dir / "episode_trace.json").is_file():
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "trace_run.json").write_text(json.dumps(res, indent=1))
        print(res["stderr"][-4000:], file=sys.stderr)
        refuse(f"trace tool failed (exit {res['returncode']}); folder left as a crashed panel", 3)
    if sha256_file(zip_path) != zip_sha or sha256_file(pkl_path) != pkl_sha:
        refuse("the checkpoint pair changed while it was measured", 4)
    with open(out_dir / "trace_run.json", "x") as fh:
        json.dump({**res, "zip": str(zip_path), "pkl": str(pkl_path), "zip_sha256": zip_sha,
                   "pkl_sha256": pkl_sha, "tree_digest_before": digest, "tree_digest_after": digest_after,
                   "stderr_nonempty": bool(res["stderr"].strip())}, fh, indent=1)
    agg, flags, problems = reduce_panel(out_dir)
    if problems:
        (out_dir / "problems.txt").write_text("\n".join(problems) + "\n")
        refuse("trace consistency: " + "; ".join(problems[:5]), 4)
    write_flags(out_dir, flags)
    row = {**identity, "label": label, "num_timesteps": zip_num_timesteps(zip_path), "zip_sha256": zip_sha,
           "pkl_sha256": pkl_sha, **{k: v for k, v in agg.items() if k in PANEL_COLUMNS},
           "measure_tree_digest": digest, "stack": stack_text(), "utc": utc_now()}
    row = {k: (_f(v) if isinstance(v, float) or v is None else v) for k, v in row.items()}
    (out_dir / "panel_extra.json").write_text(json.dumps({"gate_flagged": agg["gate_flagged"]}, indent=1))

    def still_this_attempt() -> None:  # under the panels.csv lock: the crash rule may have retired the replicate
        if not zip_path.is_file() or sha256_file(zip_path) != zip_sha:
            refuse(f"{zip_path} is gone or changed since it was measured (replicate retired by the crash rule?); "
                   "no row written, panel folder left as a crashed panel", 4)

    append_row(panels_csv, row, verify=still_this_attempt)
    gzip_csvs(out_dir)
    print(json.dumps({"panel": f"{name} {label}", "cmd": row["cmd"], "frac99": row["frac99"],
                      "tailup_flags": row["tailup_flags"], "other_flags": row["other_flags"],
                      "class_single": row["class_single"], "seconds": res["seconds"]}))
    return row


def run_identity(name: str) -> dict:
    state, arm, seed = parse_name(name)
    meta_path = RUNS / name / "run_meta.json"
    coef, b = "", ""
    if meta_path.is_file():
        with open(meta_path) as fh:
            meta = json.load(fh)
        coef, b = meta.get("coef", ""), meta.get("b", "")
    return {"name": name, "state": state, "arm": arm, "rep_seed": seed, "coef": coef, "b": b}


def measure_run_label(name: str, label: str) -> dict:
    if label not in C.LABELS:
        refuse(f"label {label!r} not in {C.LABELS}")
    ident = run_identity(name)
    z, p = checkpoint_pair(name, label)
    return measure_one(name, label, zip_path=z, pkl_path=p, out_dir=PANELS_DIR / name / label,
                       panels_csv=PANELS_CSV, identity=ident)


def pending() -> list[tuple[float, str, str]]:
    done = {(r["name"], r["label"]) for r in read_panels()}
    out = []
    for name in all_names():
        for label in C.LABELS:
            if (name, label) in done:
                continue
            z, p = checkpoint_pair(name, label)
            if z.is_file() and p.is_file():  # the pkl is saved after the zip: both present = the zip is complete
                out.append((z.stat().st_mtime, name, label))
    return sorted(out)


def all_pending(follow: bool) -> int:
    total = len(all_names()) * len(C.LABELS)
    while True:
        todo = pending()
        for _, name, label in todo:
            measure_run_label(name, label)
        n_done = len({(r["name"], r["label"]) for r in read_panels()})
        if not follow or n_done >= total:
            print(json.dumps({"measured": n_done, "of": total}))
            return 0
        lost = SCREEN / "LOST.txt"
        if lost.is_file():
            lost_names = {ln.split()[0] for ln in lost.read_text().splitlines() if ln.strip()}
            need = {(n, lb) for n in all_names() if n not in lost_names for lb in C.LABELS}
            if need <= {(r["name"], r["label"]) for r in read_panels()}:
                print(json.dumps({"measured": n_done, "of": total, "lost": sorted(lost_names)}))
                return 0
        time.sleep(60)


# ------------------------------------------------------------------------------------------------- crash rule

def retire_crashed(name: str, *, runs: Path = RUNS, panels_dir: Path = PANELS_DIR, panels_csv: Path = PANELS_CSV,
                   crashed_csv: Path | None = None) -> dict:
    """The criteria's crash rule for replicate `name`, under the panels.csv lock: runs/<name> -> runs/<name>_crashed<k>,
    panels/<name> -> panels/<name>_crashed<k> (same k), and that replicate's panels.csv rows -> panels_crashed.csv
    with attempt '<name>_crashed<k>'.  Never deletes anything.  Returns what was moved."""
    parse_name(name)
    crashed_csv = crashed_csv or panels_csv.with_name(CRASHED_CSV_NAME)
    run_dir = runs / name
    if not run_dir.is_dir():
        refuse(f"{run_dir} does not exist: nothing to retire")
    panels_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(panels_csv, "a+", newline="") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            k = 1
            while (runs / f"{name}_crashed{k}").exists() or (panels_dir / f"{name}_crashed{k}").exists():
                k += 1
            attempt = f"{name}_crashed{k}"
            fh.seek(0)
            text = fh.read()
            keep, moved = [], []
            if text:
                rdr = csv.DictReader(io.StringIO(text))
                if rdr.fieldnames != PANEL_COLUMNS:
                    refuse(f"{panels_csv} has other columns than this tool writes")
                for r in rdr:
                    (moved if r["name"] == name else keep).append(r)
            # 1. the crashed ledger first (append; nothing is lost if a later step fails)
            if moved:
                new_ledger = not crashed_csv.is_file() or crashed_csv.stat().st_size == 0
                with open(crashed_csv, "a", newline="") as led:
                    w = csv.DictWriter(led, fieldnames=["attempt", *PANEL_COLUMNS], lineterminator="\n")
                    if new_ledger:
                        w.writeheader()
                    for r in moved:
                        w.writerow({"attempt": attempt, **r})
                    led.flush()
                    os.fsync(led.fileno())
            # 2. the folders
            run_dir.rename(runs / attempt)
            panel_moved = None
            if (panels_dir / name).exists():
                (panels_dir / name).rename(panels_dir / attempt)
                panel_moved = str(panels_dir / attempt)
            # 3. panels.csv without the crashed attempt's rows (rewritten in place under the lock)
            if moved:
                fh.seek(0)
                fh.truncate()
                w = csv.DictWriter(fh, fieldnames=PANEL_COLUMNS, lineterminator="\n")
                w.writeheader()
                for r in keep:
                    w.writerow(r)
                fh.flush()
                os.fsync(fh.fileno())
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)
    out = {"retired": name, "attempt": attempt, "run_dir": str(runs / attempt), "panel_dir": panel_moved,
           "rows_moved": [r["label"] for r in moved], "ledger": str(crashed_csv), "utc": utc_now()}
    print(json.dumps(out))
    return out


# ------------------------------------------------------------------------------------------------- V1

V1_FULL = ("frac99", "frac90")
V1_ROUND6 = ("cmd", "raw_max_window", "return_mean")
V1_INT = ("tailup_flags", "gate_flagged")


def validate_v1() -> int:
    VALIDATE.mkdir(exist_ok=True)
    v1_csv = VALIDATE / "v1_panels.csv"
    result = {"schema": "tailscreen-v1/1", "utc": utc_now(), "states": {}, "passed": True}
    for state in ("s42", "s80"):
        st = C.STATES[state]
        zip_path = Path(st["zip"])
        pkl_path = zip_path.with_name("stage1_vecnormalize_4194304_steps.pkl")
        if sha256_file(zip_path) != st["zip_sha256"] or sha256_file(pkl_path) != st["pkl_sha256"]:
            refuse(f"{state} start pair sha256 differs from screen_common.STATES")
        name, label = f"start_{state}", str(C.START_STEPS)
        if has_row(name, label, v1_csv):
            refuse(f"{v1_csv} already has {name}: V1 is run once (move validate/ away to rerun)")
        out_dir = VALIDATE / f"V1_{state}"
        row = measure_one(name, label, zip_path=zip_path, pkl_path=pkl_path, out_dir=out_dir, panels_csv=v1_csv,
                          identity={"name": name, "state": state, "arm": "", "rep_seed": "", "coef": "", "b": ""})
        extra = json.loads((out_dir / "panel_extra.json").read_text())
        exp = C.V1_EXPECTED[state]
        got = {}
        cmp = {}
        for k in V1_FULL:
            got[k] = float(row[k])
            cmp[k] = got[k] == float(exp[k])
        for k in V1_ROUND6:
            got[k] = round(float(row[k]), 6)
            cmp[k] = got[k] == round(float(exp[k]), 6)
        got["tailup_flags"] = int(row["tailup_flags"])
        got["gate_flagged"] = int(extra["gate_flagged"])
        for k in V1_INT:
            cmp[k] = got[k] == int(exp["flagged"])
        ok = all(cmp.values())
        result["states"][state] = {"expected": exp, "obtained": got, "equal": cmp, "passed": ok,
                                   "panel_dir": str(out_dir), "trace_json_sha256": row["trace_json_sha256"],
                                   "measure_tree_digest": row["measure_tree_digest"]}
        result["passed"] = result["passed"] and ok
    result["stack"] = C.stack_versions()
    with open(VALIDATE / "v1.json", "x") as fh:
        json.dump(result, fh, indent=1)
    print(json.dumps({s: v["passed"] for s, v in result["states"].items()} | {"passed": result["passed"]}))
    return 0 if result["passed"] else 6


# ------------------------------------------------------------------------------------------------- CLI

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--run", help="replicate name <state>_<arm>_r<seed>")
    g.add_argument("--all-pending", action="store_true")
    g.add_argument("--validate-v1", action="store_true")
    g.add_argument("--build-copy", action="store_true")
    g.add_argument("--retire-crashed", metavar="NAME", help="the crash rule for replicate NAME (run_waves.sh)")
    ap.add_argument("--label")
    ap.add_argument("--follow", action="store_true")
    ns = ap.parse_args(argv)
    (SCREEN / "logs").mkdir(exist_ok=True)
    if ns.build_copy:
        return build_copy()
    if ns.retire_crashed:
        retire_crashed(ns.retire_crashed)
        return 0
    if ns.validate_v1:
        return validate_v1()
    if ns.all_pending:
        return all_pending(ns.follow)
    if not ns.label:
        ap.error("--run needs --label")
    if ns.follow:
        ap.error("--follow goes with --all-pending")
    measure_run_label(ns.run, ns.label)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
