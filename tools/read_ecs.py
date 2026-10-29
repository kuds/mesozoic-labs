"""Entropy continuation screen (ECS): the reader (build list v2 item 6; criteria_ecs_v2.txt THE READINGS).
Report only: reads files, runs the pinned M5 (a ledger reader), never trains and never simulates.

    PY -I -B read_ecs.py --check RUN
        CHECK only (R4 and run_meta).  Reads RUN/run_meta.json and only the update_timesteps, ent_coef and
        learning_rate columns of RUN/01_stance/policy_probe.npz (allow_pickle False), and the same columns of the
        parent's F2 probe.  Writes RUN/check.json.  Exit 0 pass, 4 fail, 2 input error.
    PY -I -B read_ecs.py RUN [--out DIR] [--trace DIR]
        The full read (R1, R2, R3).  Refused (exit 2) for a run without a current passing check.json and, for a
        continuation (RUN under <ECS>/runs/), until every one of the 24 is decided (COUNTED, STOPPED or LOST).
        Writes DIR/m5.json, DIR/m5.txt and DIR/read.json (default DIR: <ECS>/score/read/<name>/ for a continuation,
        RUN/read/ for a validation run under <ECS>/validate/).  A continuation whose C6 fails on a counting check is
        written to <ECS>/lost.json (cause "C6"); one whose ledger R1 cannot run on (the ledger missing or unreadable,
        M5's INPUT ERROR exit 2, or M5 not finishing within M5_TIMEOUT_S twice) is written there with cause "R1"
        (criteria RUN HANDLING LOST; review RR-1).  Any other M5 failure (another exit code, an M5 JSON that read_ecs's
        own checks reject) is a tool defect: exit 2, nothing written to lost.json (criteria: the VOID route).
        Exit 0 read (or LOST written), 2 input error.
    PY -I -B read_ecs.py --ledger CSV --S S --T T [--truncate] --out DIR
        R1 alone on one ledger (V4: the parents' ledgers truncated to timesteps <= T).  Refused for a ledger under
        <ECS>/runs/ until all 24 are decided (outcome-blind; review RR-7).  Exit 0 read, 2 input error.

CHECK (criteria THE READINGS, CHECK and R4): rows of the continuation's probe with update_timesteps in (S, T),
(T - S)/8,192 - 1 of them, each at S + 8,192 k; arm A: ent_coef and learning_rate equal, as float32, to the parent's F2
probe row at the same step; arm B: every ent_coef 0.0 and learning_rate as A.  run_meta: schema, name, mode, status
"done", exit 0, code digest, start pair sha256, parent records sha256, arm config, seed, S, E, torch's intra-op
threads 1, the collapse stop present with patience 10 (D8), and the stop reason: "end" at E (StopAtEnd's flag) or
"collapse" at S < T < E (the collapse callback tripped).  A continuation made updates (its parameter digest changed and
its n_updates grew; review I6).  A no-update validation run (V2, under <ECS>/validate/) stops "end" at its target, and
its parameter and Adam digests are equal at start and end.
R1 THE LEDGER: M5 unchanged ("ledger_r10round.py <ledger> --E <T> --json <out>"); only evaluations with timesteps in
(S, T] are read: POSITIVE (at least one ANY-flagged), the first ANY-flagged step, the flagged count, ANY and TAIL
alarms whose two evaluations both lie in (S, T] (an alarm whose first evaluation is the parent's at S is printed as
straddling S and not counted), TAIL-flagged evaluations; C6's failures split into those that count (grid, rows,
episodes, ANY input missing, anything else) and the TAIL-input-missing ones (printed, report only).
R2 (report only): the M6 arrays and std_per_actuator for all 15 actuators, rows with update_timesteps in (S, T) (the
probe's step key is update_timesteps; M6's own reader looks for rollout_timesteps, HS G7).
R3 (report only): M1's m1_summary.json for the end pair (stage1_<E>_steps; a STOPPED one's stage1_final).
"""
import os
import sys

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True

import argparse  # noqa: E402
import csv  # noqa: E402
import hashlib  # noqa: E402
import importlib.util  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import re  # noqa: E402
import subprocess  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402

TOOLS = Path(__file__).resolve().parent
VERSION = "read_ecs 1.0 (ECS build list v2 item 6; lane R, 2026-10-28)"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


E = sys.modules.get("ecs_common") or _load("ecs_common")

EXIT_OK, EXIT_INPUT, EXIT_CHECK_FAIL = E.EXIT_OK, E.EXIT_REFUSED, E.EXIT_CHECK_FAIL          # 0, 2, 4
CHECK_NAME = E.CHECK_JSON
CHECK_SCHEMA = "ecs-check/1"
READ_SCHEMA = "ecs-read/1"
READ_NAME = "read.json"
# Local (ecs_common has no constant for it): the full read's output folder for a continuation, under <ECS>/score/.
READ_SUBDIR = ("score", "read")
LEDGER_REL = f"{E.STAGE_DIR}/eval_episodes.csv"
PROBE_REL = f"{E.STAGE_DIR}/policy_probe.npz"
MODELS_REL = f"{E.STAGE_DIR}/models"
FINAL_ZIP = "stage1_final.zip"                  # TB l1161 _save_final_and_sync_tb: <stage>_final (a STOPPED end pair)
TRACE_NAME = "m1_summary.json"
GRID = E.EVAL_FREQ                              # 131,072: M5's evaluation grid (r10round_common.py l64)
# Local: M5's ANY-FLAGGED / TAIL-FLAGGED threshold, at least 5 of an evaluation's 10 rows (CRIT6 M5 (a);
# ledger_r10round.py l14; r10round_common.py m5 "any_flagged = any >= 5").  Used only to cross-check M5's own lists.
FLAG_ROWS = 5
N_ACTUATORS = 15
# Local: the 15 actuator names in index order, as M5 prints them (r10round_common.py ACTUATOR_NAMES, l94-96).
ACTUATOR_NAMES = ("neck_pitch_act", "neck_yaw_act", "head_pitch_act", "r_hip_pitch_act", "r_hip_roll_act",
                  "r_knee_act", "r_ankle_act", "l_hip_pitch_act", "l_hip_roll_act", "l_knee_act", "l_ankle_act",
                  "tail_1_pitch_act", "tail_1_yaw_act", "tail_2_pitch_act", "tail_3_pitch_act")
M5_TIMEOUT_S = 900

C6_TAIL_RE = re.compile(r"^evaluation (\d+) episode .*: TAIL input missing \(")
C6_ANY_RE = re.compile(r"^evaluation (\d+) episode .*: ANY input missing \(")
C6_ROWS_RE = re.compile(r"^evaluation (\d+): \d+ rows, episodes ")
C6_GRID_RE = re.compile(r"^(?:evaluation (\d+) (?:absent|beyond E = \d+)|row \d+ timesteps .* is not on the grid)$")


class InputError(Exception):
    """An input the reader cannot read as the text requires (exit 2)."""


class R1Unreadable(InputError):
    """R1 cannot run on a ledger (missing or unreadable, M5's INPUT ERROR, M5 timing out twice): for a continuation,
    LOST with cause "R1" (criteria RUN HANDLING LOST)."""


# ================================================================================================ small helpers
def utcnow():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def tool_sha256():
    return E.sha256_file(Path(__file__))


def require_under(path, root):
    """Every file this tool writes is under the screen's folder (HARD RULES)."""
    p, r = Path(path).resolve(), Path(root).resolve()
    if p != r and r not in p.parents:
        raise InputError(f"{p} is not under {r}")
    return p


def atomic_write_text(path, text):
    path = Path(path)
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def read_json_file(path):
    """(dict, sha256) of a JSON object file; InputError when it is unreadable or not an object."""
    try:
        raw = Path(path).read_bytes()
    except OSError as exc:
        raise InputError(f"cannot read {path}: {exc}") from exc
    try:
        d = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise InputError(f"{path} is not JSON: {exc}") from exc
    if not isinstance(d, dict):
        raise InputError(f"{path} is not a JSON object")
    return d, sha256_bytes(raw)


def read_dir(root, name):
    """<root>/score/read/<name>: the full read's output folder for a continuation."""
    return Path(root).joinpath(*READ_SUBDIR, name)


def run_kind(run_dir, root):
    """"continuation" for <root>/runs/<name>, "validation" for a folder under <root>/validate/, else None."""
    run, root = Path(run_dir).resolve(), Path(root).resolve()
    if run.parent == root / "runs":
        return "continuation"
    if (root / "validate") in run.parents:
        return "validation"
    return None


def finite_or_none(v):
    v = float(v)
    return v if math.isfinite(v) else None


# ================================================================================================ lost.json
def load_lost(path):
    """{name: entry} from <ECS>/lost.json (ecs_common LOST_SCHEMA); {} when the file is absent."""
    path = Path(path)
    if not path.exists():
        return {}
    d, _sha = read_json_file(path)
    if d.get("schema") != E.LOST_SCHEMA or not isinstance(d.get("lost"), dict):
        raise InputError(f"{path}: schema {d.get('schema')!r}, expected {E.LOST_SCHEMA!r} with a 'lost' object")
    for name, entry in d["lost"].items():
        p = E.parse_run_name(name)
        if p is None or p["suffix"] or not isinstance(entry, dict):
            raise InputError(f"{path}: entry {name!r} is not a continuation's own name with an object")
    return dict(d["lost"])


def add_lost(path, name, cause, detail):
    """Add one continuation to lost.json (the reader's C6 case), under a lock shared with the runner's writes."""
    import fcntl

    path = Path(path)
    with open(path.with_name(path.name + ".lock"), "a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        cur = load_lost(path)
        if name in cur:
            return False
        cur[name] = {"cause": cause, "detail": detail, "utc": utcnow(), "by": "read_ecs.py"}
        atomic_write_text(path, json.dumps({"schema": E.LOST_SCHEMA, "lost": cur}, indent=1, sort_keys=True) + "\n")
        return True


# ================================================================================================ the probe
def load_probe(path, keys):
    """Only the named columns of a policy_probe.npz (allow_pickle False; np.load reads a member when it is asked)."""
    import numpy as np

    try:
        with np.load(path, allow_pickle=False) as z:
            missing = [k for k in keys if k not in z.files]
            if missing:
                raise InputError(f"{path}: no {missing}")
            return {k: np.asarray(z[k]) for k in keys}
    except (OSError, ValueError) as exc:
        raise InputError(f"cannot read {path}: {exc}") from exc


def parent_probe(start):
    """The parent's F2 probe columns for R4, after its sha256 is checked against FMAN's (ecs_common)."""
    path = E.f2_record(start, "policy_probe.npz")
    want = E.STARTS[start]["records"]["policy_probe.npz"]
    try:
        got = E.sha256_file(path)
    except OSError as exc:
        raise InputError(f"parent probe {path}: {exc}") from exc
    if got != want:
        raise InputError(f"parent probe {path}: sha256 {got} != FMAN's {want}")
    return path, got, load_probe(path, E.PROBE_CHECK_COLUMNS)


def window_rows(steps, S, T):
    """Indices of rows with update_timesteps in (S, T), in file order."""
    return [i for i, u in enumerate(steps.tolist()) if S < int(u) < T]


def r4(run_cols, parent_cols, arm, S, T):
    """R4 MANIPULATION.  (failures, details).  run_cols / parent_cols: {update_timesteps, ent_coef, learning_rate}."""
    import numpy as np

    fails = []
    n_exp = E.expected_update_rows(S, T)
    steps = np.asarray(run_cols["update_timesteps"]).astype(np.int64)
    idx = window_rows(steps, S, T)
    got_steps = [int(steps[i]) for i in idx]
    exp_steps = [S + E.ROLLOUT * k for k in range(1, n_exp + 1)]
    late = [int(u) for u in steps.tolist() if int(u) >= T]
    prior = sum(1 for u in steps.tolist() if int(u) <= S)
    det = dict(rows_expected=n_exp, rows_in_window=len(idx), first_step=got_steps[0] if got_steps else None,
               last_step=got_steps[-1] if got_steps else None, rows_at_or_after_T=late[:5],
               n_rows_at_or_after_T=len(late),
               seeded_rows_at_or_before_S=prior, mismatches=[])
    if len(idx) != n_exp:
        fails.append(f"R4: {len(idx)} probe rows with update_timesteps in ({S}, {T}), expected {n_exp}")
    if got_steps != exp_steps:
        bad = [s for s in got_steps if s not in set(exp_steps)][:5]
        absent = [s for s in exp_steps if s not in set(got_steps)][:5]
        fails.append(f"R4: probe steps are not S + 8,192 k, k = 1..{n_exp} (off-grid or repeated {bad}; absent "
                     f"{absent})")
    if late:
        fails.append(f"R4: {len(late)} probe rows at or after T = {T} (first {late[0]}): an update on the end rollout")
    if fails:
        return fails, det
    pst = np.asarray(parent_cols["update_timesteps"]).astype(np.int64).tolist()
    where = {}
    for i, u in enumerate(pst):
        where.setdefault(int(u), []).append(i)
    p_idx = []
    for s in exp_steps:
        if len(where.get(s, [])) != 1:
            raise InputError(f"parent probe has {len(where.get(s, []))} rows at update_timesteps {s}")
        p_idx.append(where[s][0])
    ent = np.asarray(run_cols["ent_coef"], dtype=np.float32)[idx]
    lr = np.asarray(run_cols["learning_rate"], dtype=np.float32)[idx]
    p_ent = np.asarray(parent_cols["ent_coef"], dtype=np.float32)[p_idx]
    p_lr = np.asarray(parent_cols["learning_rate"], dtype=np.float32)[p_idx]
    want_ent = p_ent if arm == "A" else np.zeros_like(p_ent)
    bad_ent = [i for i in range(n_exp) if not (ent[i] == want_ent[i])]          # NaN never equals
    bad_lr = [i for i in range(n_exp) if not (lr[i] == p_lr[i])]
    for i in (bad_ent + [j for j in bad_lr if j not in bad_ent])[:10]:
        det["mismatches"].append(dict(step=exp_steps[i], ent_coef=float(ent[i]), ent_coef_expected=float(want_ent[i]),
                                      learning_rate=float(lr[i]), learning_rate_parent=float(p_lr[i])))
    if bad_ent:
        what = "the parent's F2 probe (float32)" if arm == "A" else "0.0"
        fails.append(f"R4: ent_coef differs from {what} at {len(bad_ent)} of {n_exp} rows (first "
                     f"{exp_steps[bad_ent[0]]})")
    if bad_lr:
        fails.append(f"R4: learning_rate differs from the parent's F2 probe (float32) at {len(bad_lr)} of {n_exp} rows "
                     f"(first {exp_steps[bad_lr[0]]})")
    det.update(ent_coef_first=float(ent[0]), ent_coef_last=float(ent[-1]), learning_rate_first=float(lr[0]),
               learning_rate_last=float(lr[-1]))
    return fails, det


# ================================================================================================ CHECK: run_meta
def expected_arm_ppo(arm):
    """run_meta's arm_ppo as run_ecs.py records it: the arm's in-memory [ppo] edit (D1), "unchanged" for arm A."""
    return dict(E.ARM_PPO_EDIT[arm]) or "unchanged"


def check_meta(meta, kind, folder):
    """run_meta.json's part of CHECK.  (failures, identity, info).  identity: start, arm, seed, S, E, T, stop."""
    fails = []
    get = meta.get
    if get("schema") != E.RUN_SCHEMA:
        fails.append(f"run_meta schema {get('schema')!r} != {E.RUN_SCHEMA!r}")
    p = E.parse_run_name(folder)
    if kind == "continuation" and (p is None or p["suffix"]):
        raise InputError(f"{folder}: not a continuation's own folder name (a crashed or void folder is never read)")
    m_start, m_arm, m_seed = get("start"), get("arm"), get("seed")
    if p is not None and (p["start"], p["arm"], p["seed"]) != (m_start, m_arm, m_seed):
        fails.append(f"folder {folder} names {p['start']} {p['arm']} {p['seed']}, run_meta {m_start} {m_arm} {m_seed}")
    # A continuation's identity is its folder's name (BL 3 layout); a validation run's is run_meta's.
    start, arm, seed = (p["start"], p["arm"], p["seed"]) if kind == "continuation" else (m_start, m_arm, m_seed)
    if start not in E.STARTS:
        fails.append(f"run_meta start {start!r} is not one of {tuple(E.STARTS)}")
        return fails, dict(start=None, arm=arm, seed=seed, S=None, E=None, T=None, end=None,
                           stop_reason=get("stop_reason")), dict(criteria_sha256=get("criteria_sha256"))
    if arm not in E.ARMS:
        fails.append(f"run_meta arm {arm!r} is not one of {E.ARMS}")
    if not isinstance(seed, int) or isinstance(seed, bool) or not E.seed_ok(start, arm, seed):
        fails.append(f"run_meta seed {seed!r} is not in the {start} arm {arm} list (T1)")
    S, Eend = E.window(start)
    if get("S") != S or get("E") != Eend:
        fails.append(f"run_meta S, E = {get('S')!r}, {get('E')!r}; expected {S}, {Eend}")
    if get("name") != E.run_name(start, arm, seed if isinstance(seed, int) else 0):
        fails.append(f"run_meta name {get('name')!r} != {E.run_name(start, arm, seed if isinstance(seed, int) else 0)}")
    mode = get("mode")
    want_mode = "continuation" if kind == "continuation" else "no_update"
    if mode != want_mode:
        fails.append(f"run_meta mode {mode!r}: a {kind} run must be {want_mode!r}")
    if get("status") != "done":
        fails.append(f"run_meta status {get('status')!r} != 'done'")
    if get("exit") != 0:
        fails.append(f"run_meta exit {get('exit')!r} != 0")
    code = get("code") or {}
    if code.get("tree_digest") != E.TREE_DIGEST:
        fails.append(f"code tree_digest {code.get('tree_digest')!r} != {E.TREE_DIGEST}")
    if code.get("commit") != E.COMMIT:
        fails.append(f"code commit {code.get('commit')!r} != {E.COMMIT}")
    st = E.STARTS[start]
    sp = get("start_pair") or {}
    if sp.get("zip_sha256") != st["zip_sha256"] or sp.get("pkl_sha256") != st["pkl_sha256"]:
        fails.append(f"start pair sha256 {sp.get('zip_sha256')!r} / {sp.get('pkl_sha256')!r} != "
                     f"{st['zip_sha256']} / {st['pkl_sha256']}")
    pr = get("parent_records") or {}
    for rec in E.PARENT_RECORDS:
        if pr.get(rec) != st["records"][rec]:
            fails.append(f"parent record {rec} sha256 {pr.get(rec)!r} != FMAN's {st['records'][rec]}")
    info = {}
    if arm in E.ARMS:
        want_ppo, got_ppo = expected_arm_ppo(arm), get("arm_ppo")
        if got_ppo != want_ppo or (isinstance(got_ppo, dict) and any(isinstance(v, bool) for v in got_ppo.values())):
            fails.append(f"arm {arm} [ppo] edit {got_ppo!r} != {want_ppo!r}")
        eal = get("ent_coef_after_load")
        if eal != E.ARM_ENT_COEF_AFTER_LOAD[arm] or isinstance(eal, bool):
            fails.append(f"ent_coef after load {eal!r} != {E.ARM_ENT_COEF_AFTER_LOAD[arm]}")
        info["arm_ppo"] = got_ppo
    v5 = get("recipe_v5")
    if not isinstance(v5, dict) or v5.get("problems") != []:
        fails.append(f"run_meta recipe_v5 (V5 before learn()) not passed: {v5!r:.200}")
    thr = get("torch_threads")
    if thr != E.TORCH_THREADS or isinstance(thr, bool):
        fails.append(f"run_meta torch_threads {thr!r} != {E.TORCH_THREADS} (review H6)")
    T, target, stop = get("T"), get("target"), get("stop_reason")
    if not isinstance(T, int) or isinstance(T, bool):
        fails.append(f"run_meta T {T!r} is not an integer")
        T = None
    sae = get("stop_at_end") or {}
    col = get("collapse") or {}
    if col.get("found") != 1 or col.get("patience") != E.COLLAPSE_PATIENCE:
        fails.append(f"the collapse stop (D8) is not in the run with patience {E.COLLAPSE_PATIENCE}: collapse {col}")
    end = Eend if kind == "continuation" else target
    if kind == "continuation" and target != Eend:
        fails.append(f"run_meta target {target!r} != E = {Eend}")
    if kind == "validation" and (not isinstance(target, int) or isinstance(target, bool) or target <= S):
        fails.append(f"run_meta target {target!r} is not a step after S = {S}")
        end = None
    if T is not None and end is not None:
        if stop == "end":
            if T != end:
                fails.append(f"stop reason 'end' at T = {T}, not at {end}")
            if sae.get("flag") is not True:
                fails.append(f"stop reason 'end' without StopAtEnd's flag (stop_at_end {sae})")
        elif stop == "collapse":
            if not S < T < end:
                fails.append(f"stop reason 'collapse' at T = {T}, not in ({S}, {end})")
            if col.get("tripped") is not True:
                fails.append(f"stop reason 'collapse' without the collapse callback's state (collapse {col})")
        else:
            fails.append(f"stop reason {stop!r}: neither 'end' nor 'collapse'")
        if (T - S) % E.ROLLOUT or T <= S:
            fails.append(f"T - S = {T - S} is not a positive multiple of {E.ROLLOUT}")
    if kind == "continuation":                  # the run made updates (review I6)
        d0, d1 = get("digest_start") or {}, get("digest_end") or {}
        if not d0.get("params") or not d1.get("params") or d0.get("params") == d1.get("params"):
            fails.append(f"continuation: parameter digest {d0.get('params')!r} -> {d1.get('params')!r}: no update made")
        n0, n1 = d0.get("n_updates"), d1.get("n_updates")
        if not all(isinstance(n, int) and not isinstance(n, bool) for n in (n0, n1)) or n1 <= n0:
            fails.append(f"continuation: n_updates {n0!r} -> {n1!r} did not grow")
    if kind == "validation":
        d0, d1 = get("digest_start") or {}, get("digest_end") or {}
        for k in ("params", "adam"):
            if not d0.get(k) or d0.get(k) != d1.get(k):
                fails.append(f"no-update run: {k} digest at start {d0.get(k)!r} != at end {d1.get(k)!r}")
        if T is not None and end is not None and not fails:
            n = E.expected_update_rows(S, T)
            if get("train_calls") != n:
                fails.append(f"no-update run: train_calls {get('train_calls')!r} != {n} (the rollouts updated)")
    ident = dict(start=start, arm=arm, seed=seed, S=S, E=Eend, T=T, end=end, stop_reason=stop)
    info.update(torch_threads=thr, torch_interop_threads=get("torch_interop_threads"))
    info.update(criteria_sha256=get("criteria_sha256"), code=code, stack=get("stack"), wall_seconds=get("wall_seconds"),
                collapse=col, stop_at_end=sae, mode=mode, target=target, run_tool_sha256=get("tool_sha256"),
                approved_sha256=get("approved_sha256"))
    return fails, ident, info


def check_run(run_dir, root):
    """CHECK.  Writes <run>/check.json.  Returns (exit code, record)."""
    run = Path(run_dir).resolve()
    root = Path(root).resolve()
    if not run.is_dir():
        raise InputError(f"no run folder {run}")
    kind = run_kind(run, root)
    if kind is None:
        raise InputError(f"{run} is neither <root>/runs/<name> nor under <root>/validate/")
    require_under(run / CHECK_NAME, root)
    fails = []
    sources = {}
    meta_path = run / E.RUN_META
    try:
        meta, msha = read_json_file(meta_path)
        sources["run_meta.json"] = [str(meta_path), msha]
    except InputError as exc:
        meta = None
        fails.append(f"run_meta.json: {exc}")
    ident, info, r4det = {}, {}, None
    if meta is not None:
        mf, ident, info = check_meta(meta, kind, run.name)
        fails += mf
        S, T = ident["S"], ident["T"]
        probe_path = run / PROBE_REL
        if S is not None and T is not None and T > S and (T - S) % E.ROLLOUT == 0 and ident["arm"] in E.ARMS:
            ppath, psha, pcols = parent_probe(ident["start"])
            sources["parent policy_probe.npz"] = [ppath, psha]
            try:
                rcols = load_probe(probe_path, E.PROBE_CHECK_COLUMNS)
                sources["policy_probe.npz"] = [str(probe_path), E.sha256_file(probe_path)]
            except InputError as exc:
                rcols = None
                fails.append(f"policy_probe.npz: {exc}")
            if rcols is not None:
                rf, r4det = r4(rcols, pcols, ident["arm"], S, T)
                fails += rf
        else:
            fails.append("R4 not run: T, S or the arm do not allow it")
    rec = dict(schema=CHECK_SCHEMA, version=VERSION, tool_sha256=tool_sha256(), utc=utcnow(), run=str(run),
               name=(meta or {}).get("name"), mode=kind, passed=not fails, failures=fails, r4=r4det, sources=sources,
               **{k: ident.get(k) for k in ("start", "arm", "seed", "S", "E", "T", "end", "stop_reason")},
               criteria_sha256=info.get("criteria_sha256"), meta=info)
    atomic_write_text(run / CHECK_NAME, json.dumps(rec, indent=1, default=str) + "\n")
    return (EXIT_OK if not fails else EXIT_CHECK_FAIL), rec


def check_is_current(run, check):
    """A check.json is read only while run_meta.json and policy_probe.npz are the files it checked."""
    src = check.get("sources") or {}
    for key, rel in (("run_meta.json", E.RUN_META), ("policy_probe.npz", PROBE_REL)):
        if key not in src:
            return f"check.json names no {key}"
        try:
            now = E.sha256_file(Path(run) / rel)
        except OSError as exc:
            return f"{rel}: {exc}"
        if now != src[key][1]:
            return f"{rel} changed since check.json (sha256 {now} != {src[key][1]})"
    return None


# ================================================================================================ decided
def decided(root):
    """(all decided, [undecided names]): each of the 24 COUNTED or STOPPED (a passing check.json in runs/<name>) or
    LOST (lost.json)."""
    root = Path(root)
    lost = load_lost(root / "lost.json")
    und = []
    for start, arm, seed in E.QUEUE:
        name = E.run_name(start, arm, seed)
        if name in lost:
            continue
        cpath = root / "runs" / name / CHECK_NAME
        try:
            c, _ = read_json_file(cpath)
        except InputError:
            und.append(name)
            continue
        if c.get("passed") is not True or c.get("mode") != "continuation":
            und.append(name)
    return not und, und


# ================================================================================================ R1
def classify_c6(failure):
    """(category, step or None).  'tail' (TAIL input missing: report only) or a counting category."""
    for cat, rx in (("tail", C6_TAIL_RE), ("any", C6_ANY_RE), ("rows", C6_ROWS_RE), ("grid", C6_GRID_RE)):
        m = rx.match(failure)
        if m:
            g = m.group(1) if m.groups() else None
            return cat, (int(g) if g else None)
    return "other", None


def window_reading(m5, S, T):
    """R1 on M5's JSON (pure).  Only evaluations with timesteps in (S, T] are read."""
    per = {}
    for k, v in (m5.get("per_evaluation") or {}).items():
        try:
            per[int(k)] = v
        except (TypeError, ValueError) as exc:
            raise InputError(f"M5 per_evaluation key {k!r} is not a step") from exc
    for key in ("any_flagged", "tail_flagged", "any_alarms", "tail_alarms"):
        if not isinstance(m5.get(key), list):
            raise InputError(f"M5 JSON has no {key} list")
    any_all, tail_all = [int(s) for s in m5["any_flagged"]], [int(s) for s in m5["tail_flagged"]]
    if sorted(s for s, e in per.items() if int(e.get("any", 0)) >= FLAG_ROWS) != sorted(any_all):
        raise InputError("M5's any_flagged list disagrees with its per-evaluation ANY counts")
    if sorted(s for s, e in per.items() if int(e.get("tail", 0)) >= FLAG_ROWS) != sorted(tail_all):
        raise InputError("M5's tail_flagged list disagrees with its per-evaluation TAIL counts")
    c6 = m5.get("c6")
    if not isinstance(c6, dict) or not isinstance(c6.get("failures"), list):
        raise InputError("M5 JSON has no C6 result (it must be run with --E T)")
    if c6.get("E") != T:
        raise InputError(f"M5's C6 was run with E = {c6.get('E')!r}, not T = {T}")

    def inwin(s):
        return S < s <= T
    any_win = sorted(s for s in any_all if inwin(s))
    tail_win = sorted(s for s in tail_all if inwin(s))
    any_alarms = [int(s) for s in m5["any_alarms"]]
    tail_alarms = [int(s) for s in m5["tail_alarms"]]
    counting, tail_missing = [], []
    for f in c6["failures"]:
        cat, step = classify_c6(str(f))
        where = None if step is None else ("window" if inwin(step) else ("seeded" if step <= S else "after T"))
        item = dict(failure=str(f), category=cat, step=step, where=where)
        (tail_missing if cat == "tail" else counting).append(item)
    expected = [S + GRID * k for k in range(1, (T - S) // GRID + 1)] if S % GRID == 0 else []
    window_evals = sorted(s for s in per if inwin(s))
    return dict(
        S=S, T=T, positive=bool(any_win), first_any_flag=any_win[0] if any_win else None, n_any_flagged=len(any_win),
        any_flagged=any_win,
        any_alarms=[s for s in any_alarms if S < s - GRID and s <= T],
        any_alarms_straddling_S=[s for s in any_alarms if s - GRID <= S < s],
        tail_flagged=tail_win,
        tail_alarms=[s for s in tail_alarms if S < s - GRID and s <= T],
        tail_alarms_straddling_S=[s for s in tail_alarms if s - GRID <= S < s],
        flagged_at_or_before_S=[s for s in any_all if s <= S],
        c6_passed_m5=bool(c6.get("passed")), c6_ok=not counting, c6_counting_failures=counting,
        c6_tail_missing=tail_missing, window_evaluations=window_evals,
        window_evaluations_absent=[s for s in expected if s not in per],
        per_evaluation={s: {k: per[s].get(k) for k in ("n", "any", "tail", "unres", "any_by", "tail_by")}
                        for s in window_evals})


M5_INPUT_ERROR_EXIT = 2                         # ledger_r10round.py main: "INPUT ERROR" (the ledger unreadable)
M5_INPUT_ERROR_TEXT = "\nINPUT ERROR: "         # ... printed on stdout after M5's version line (l232-l234)
M5_ATTEMPTS = 2                                 # a timeout is tried once more before R1 is declared unable to run


def run_m5(ledger, T, out_dir):
    """R1: the pinned M5, unchanged, as a subprocess.  (M5 JSON, its sha256, stdout, M5 and common sha256).
    R1Unreadable when M5 reports its INPUT ERROR (exit 2 with its "INPUT ERROR:" line on stdout) or times out
    M5_ATTEMPTS times; InputError (a tool or environment defect) when it cannot start, exits otherwise (an exit 2
    without that line included: Python's own "can't open file" or a usage error also exit 2), or its JSON is
    unreadable."""
    out_dir = Path(out_dir)
    jpath = out_dir / "m5.json"
    m5_sha, common_sha = E.sha256_file(E.M5), E.sha256_file(E.M5_COMMON)
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               PYTHONDONTWRITEBYTECODE="1")
    argv = [E.PY, "-I", "-B", E.M5, str(ledger), "--E", str(int(T)), "--json", str(jpath)]
    p = None
    for attempt in range(1, M5_ATTEMPTS + 1):
        jpath.unlink(missing_ok=True)                # an earlier read's JSON is never read as this one's
        try:
            p = subprocess.run(argv, cwd="/tmp", env=env, capture_output=True, text=True, timeout=M5_TIMEOUT_S,
                               check=False)
            break
        except subprocess.TimeoutExpired as exc:
            if attempt == M5_ATTEMPTS:
                raise R1Unreadable(f"M5 did not finish within {M5_TIMEOUT_S} s ({M5_ATTEMPTS} attempts)") from exc
        except OSError as exc:
            raise InputError(f"M5 did not run: {exc}") from exc
    atomic_write_text(out_dir / "m5.txt", p.stdout + ("\n[stderr]\n" + p.stderr if p.stderr else ""))
    if p.returncode == M5_INPUT_ERROR_EXIT and M5_INPUT_ERROR_TEXT in p.stdout:
        raise R1Unreadable(f"M5 INPUT ERROR (exit 2): {(p.stdout + p.stderr)[-400:]}")
    if p.returncode != 0:
        raise InputError(f"M5 exit {p.returncode}: {(p.stdout + p.stderr)[-400:]}")
    m5, jsha = read_json_file(jpath)
    return m5, jsha, p.stdout, m5_sha, common_sha, argv


def ledger_returns(ledger, S, T):
    """Report only: per evaluation in (S, T] the mean return and length of its rows (the ledger's own columns)."""
    with open(ledger, "rb") as fh:
        text = fh.read().decode("utf-8", errors="replace")
    by = {}
    for r in csv.DictReader(io.StringIO(text, newline="")):
        try:
            t = float(r.get("timesteps"))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(t) or t != int(t) or not S < int(t) <= T:
            continue
        by.setdefault(int(t), []).append(r)

    def mean(rows, key):
        vals = []
        for r in rows:
            try:
                v = float(r.get(key))
            except (TypeError, ValueError):
                continue
            if math.isfinite(v):
                vals.append(v)
        return sum(vals) / len(vals) if vals else None
    per = {s: dict(n=len(rs), mean_return=mean(rs, "return"), mean_length=mean(rs, "length"))
           for s, rs in sorted(by.items())}
    last = per.get(T, {})
    return dict(per_evaluation=per, mean_return_last=last.get("mean_return"), mean_length_last=last.get("mean_length"))


def truncate_ledger(src, dst, T):
    """A copy of a ledger with the rows whose timesteps are <= T (V4); the header and cell text kept."""
    with open(src, "rb") as fh:
        text = fh.read().decode("utf-8")
    rows = list(csv.reader(io.StringIO(text, newline="")))
    if not rows:
        raise InputError(f"{src} is empty")
    head, body = rows[0], rows[1:]
    ti = head.index("timesteps") if "timesteps" in head else None
    if ti is None:
        raise InputError(f"{src} has no timesteps column")
    keep = []
    for r in body:
        try:
            ok = float(r[ti]) <= T
        except (ValueError, IndexError):
            ok = True                                        # kept: M5's C6 then prints it as off the grid
        if ok:
            keep.append(r)
    buf = io.StringIO(newline="")
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(head)
    w.writerows(keep)
    Path(dst).write_text(buf.getvalue(), encoding="utf-8")
    return len(body), len(keep)


# ================================================================================================ R2, R3
def probe_window(path, S, T):
    """R2 (report only): M6's arrays and std_per_actuator, all 15 actuators, rows with update_timesteps in (S, T)."""
    import numpy as np

    cols = load_probe(path, ("update_timesteps", *E.M6_COLUMNS))
    steps = np.asarray(cols["update_timesteps"]).astype(np.int64)
    idx = window_rows(steps, S, T)
    out = dict(file=str(path), sha256=E.sha256_file(path), step_key="update_timesteps",
               steps=[int(steps[i]) for i in idx], arrays={}, summary={})
    for k in E.M6_COLUMNS:
        a = np.asarray(cols[k], dtype=np.float64)
        if a.ndim != 2 or a.shape[1] != N_ACTUATORS:
            out["arrays"][k] = f"shape {a.shape}"
            continue
        sel = a[idx] if idx else np.zeros((0, N_ACTUATORS))
        out["arrays"][k] = {ACTUATOR_NAMES[j]: [finite_or_none(v) for v in sel[:, j]] for j in range(N_ACTUATORS)}
        summ = {}
        for j in range(N_ACTUATORS):
            col = sel[:, j]
            fin = col[np.isfinite(col)]
            summ[ACTUATOR_NAMES[j]] = dict(first=finite_or_none(col[0]) if len(col) else None,
                                           last=finite_or_none(col[-1]) if len(col) else None,
                                           mean=float(fin.mean()) if len(fin) else None,
                                           max=float(fin.max()) if len(fin) else None)
        out["summary"][k] = summ
    std = np.asarray(cols["std_per_actuator"], dtype=np.float64)
    if idx and std.ndim == 2:
        out["std_mean_first_last"] = [float(np.nanmean(std[idx[0]])), float(np.nanmean(std[idx[-1]]))]
    return out


def end_zip(run, stop_reason, T):
    """The end pair's zip: stage1_<T>_steps.zip at E, stage1_final.zip for a STOPPED one (criteria R3)."""
    models = Path(run) / MODELS_REL
    return models / (E.periodic_names(T)[0] if stop_reason == "end" else FINAL_ZIP)


def trace_reading(summary_path, zip_path, T):
    """R3 (report only): M1's m1_summary.json for the end pair; 'missing' does not hold up the reading."""
    summary_path = Path(summary_path)
    out = dict(file=str(summary_path), status="missing")
    if not summary_path.exists():
        return out
    try:
        d, sha = read_json_file(summary_path)
    except InputError as exc:
        out["status"] = f"unreadable ({exc})"
        return out
    out["sha256"] = sha
    ck = d.get("checkpoint") or {}
    try:
        zsha = E.sha256_file(zip_path)
    except OSError:
        zsha = None
    problems = []
    if zsha is None or ck.get("sha256") != zsha:
        problems.append(f"checkpoint sha256 {ck.get('sha256')!r} is not the end pair's {zsha!r} ({zip_path})")
    if ck.get("num_timesteps") != T:
        problems.append(f"checkpoint num_timesteps {ck.get('num_timesteps')!r} != T = {T}")
    if d.get("physics_revision") != 10:
        problems.append(f"physics_revision {d.get('physics_revision')!r} != 10")
    if list(d.get("seeds") or []) != list(E.M1_SEEDS):
        problems.append("seeds are not 3042-3061")
    if problems:
        out.update(status="not of the end pair (printed, not read)", problems=problems)
        return out
    cmd = d.get("cmd_all_actuators") or {}
    out.update(status="read", tool_sha256=d.get("tool_sha256"), zip=str(zip_path), zip_sha256=zsha,
               n_episodes=d.get("n_episodes"), full_horizon_episodes=d.get("full_horizon_episodes"),
               cmd_all_actuators=cmd, cmd_a=d.get("cmd_a"), cmd_tail_3=cmd.get("14"),
               tail_flags=d.get("tail_flags"), n_tail_flags=len(d.get("tail_flags") or []),
               fall_flags=d.get("fall_flags"), n_fall_flags=len(d.get("fall_flags") or []),
               any_actuator_over_bar_episodes=d.get("any_actuator_over_bar_episodes"),
               max_abs_command_any_step=d.get("max_abs_command_any_step"))
    return out


# ================================================================================================ the full read
def render_r1(r1, L):
    L.append(f"R1 window ({r1['S']:,}, {r1['T']:,}]: evaluations read {len(r1['window_evaluations'])}"
             + (f"; absent {r1['window_evaluations_absent']}" if r1["window_evaluations_absent"] else ""))
    L.append("    step         n  ANY  TAIL  UNRES  flags")
    for s, e in sorted(r1["per_evaluation"].items()):
        fl = (("A" if int(e.get("any") or 0) >= FLAG_ROWS else "-")
              + ("T" if int(e.get("tail") or 0) >= FLAG_ROWS else "-"))
        L.append(f"    {s:>10} {e.get('n')!s:>3} {e.get('any')!s:>4} {e.get('tail')!s:>5} {e.get('unres')!s:>6}  {fl}")
    L.append(f"  POSITIVE: {r1['positive']}  first ANY-flagged step: {r1['first_any_flag']}  ANY-flagged evaluations: "
             f"{r1['n_any_flagged']} {r1['any_flagged']}")
    L.append(f"  ANY ALARM steps in the window: {r1['any_alarms']}; straddling S (not counted): "
             f"{r1['any_alarms_straddling_S']}")
    L.append(f"  TAIL-flagged evaluations: {r1['tail_flagged']}; TAIL ALARM steps: {r1['tail_alarms']}; straddling S "
             f"(not counted): {r1['tail_alarms_straddling_S']}")
    if r1["flagged_at_or_before_S"]:
        L.append(f"  ANY-flagged evaluations at or before S (the parent's rows; not read): "
                 f"{r1['flagged_at_or_before_S']}")
    L.append(f"  C6 (M5 --E {r1['T']}): M5 {'PASS' if r1['c6_passed_m5'] else 'FAIL'}; counting failures "
             f"{len(r1['c6_counting_failures'])} -> {'C6 OK' if r1['c6_ok'] else 'C6 FAILS (LOST)'}; TAIL input "
             f"missing (report only) {len(r1['c6_tail_missing'])}")
    for f in r1["c6_counting_failures"][:20]:
        L.append(f"      COUNTS [{f['category']}, {f['where']}] {f['failure']}")
    for f in r1["c6_tail_missing"][:20]:
        L.append(f"      report only [tail, {f['where']}] {f['failure']}")


def _f4(v):
    return "-" if v is None else f"{v:.4f}"


def full_read(run_dir, root, out_dir=None, trace_dir=None):
    """R1, R2, R3 for one checked run.  Returns (record, text lines)."""
    run, root = Path(run_dir).resolve(), Path(root).resolve()
    kind = run_kind(run, root)
    if kind is None:
        raise InputError(f"{run} is neither <root>/runs/<name> nor under <root>/validate/")
    try:
        check, csha = read_json_file(run / CHECK_NAME)
    except InputError as exc:
        raise InputError(f"refused: no readable check.json ({exc})") from exc
    if check.get("passed") is not True:
        raise InputError(f"refused: check.json did not pass ({(check.get('failures') or ['?'])[0]})")
    if check.get("mode") != kind:
        raise InputError(f"refused: check.json mode {check.get('mode')!r} for a {kind} run")
    stale = check_is_current(run, check)
    if stale:
        raise InputError(f"refused: {stale}; run --check again")
    if kind == "continuation":
        ok, und = decided(root)
        if not ok:
            raise InputError(f"refused: {len(und)} of {len(E.QUEUE)} continuations are not decided (outcome-blind "
                             f"until all are): {und[:6]}{' ...' if len(und) > 6 else ''}")
    name = check.get("name") or run.name
    start, arm, seed, S, T, stop = (check.get(k) for k in ("start", "arm", "seed", "S", "T", "stop_reason"))
    out = Path(out_dir).resolve() if out_dir else (read_dir(root, name) if kind == "continuation" else run / "read")
    require_under(out, root)
    out.mkdir(parents=True, exist_ok=True)
    ledger = run / LEDGER_REL
    L = [VERSION, f"tool sha256 {tool_sha256()}  utc {utcnow()}", f"run {run} ({kind}) {name}: start {start} arm {arm} "
         f"seed {seed}; S {S:,}; T {T:,}; stop reason {stop}",
         f"  check.json {run / CHECK_NAME}  sha256 {csha}"]
    try:
        if not ledger.is_file():
            raise R1Unreadable(f"ledger {ledger} missing")
        try:
            lsha = E.sha256_file(ledger)
        except OSError as exc:
            raise R1Unreadable(f"ledger {ledger}: {exc}") from exc
        m5, jsha, _stdout, m5_sha, common_sha, argv = run_m5(ledger, T, out)
    except R1Unreadable as exc:
        if kind != "continuation":
            raise
        return r1_lost(root, run, out, name, check, csha, exc, L)
    r1 = window_reading(m5, S, T)
    pins_ok = m5_sha == E.PIN_SHA256[E.M5] and common_sha == E.PIN_SHA256[E.M5_COMMON]
    r1.update(ledger=str(ledger), ledger_sha256=lsha, m5=E.M5, m5_sha256=m5_sha, m5_common_sha256=common_sha,
              m5_pins_ok=pins_ok, m5_argv=argv, m5_json=str(out / "m5.json"), m5_json_sha256=jsha)
    L.append(f"  ledger {ledger}  sha256 {lsha}")
    L.append(f"  M5 {E.M5}  sha256 {m5_sha} ({'= pin' if m5_sha == E.PIN_SHA256[E.M5] else 'NOT THE PIN'}); common "
             f"sha256 {common_sha} ({'= pin' if common_sha == E.PIN_SHA256[E.M5_COMMON] else 'NOT THE PIN'})")
    L.append(f"  M5 JSON {out / 'm5.json'}  sha256 {jsha}")
    render_r1(r1, L)
    returns = ledger_returns(ledger, S, T)
    L.append(f"  report only, from {ledger}: mean return at T {returns['mean_return_last']}, mean length at T "
             f"{returns['mean_length_last']}")
    probe = run / PROBE_REL
    try:
        r2 = probe_window(probe, S, T)
        L.append(f"R2 (report only) {probe}  sha256 {r2['sha256']}: {len(r2['steps'])} rows "
                 f"{r2['steps'][:1]}..{r2['steps'][-1:]} (step key update_timesteps)")
        L.append(f"    every array's rows in ({S:,}, {T:,}) are in {out / 'r2_arrays.json'}; per actuator "
                 "first / last / mean / max over the rows:")
        for k, summ in r2["summary"].items():
            L.append(f"    {k}:")
            for act in ACTUATOR_NAMES:
                v = summ.get(act) or {}
                L.append(f"      {act:<17} " + " / ".join(_f4(v.get(x)) for x in ("first", "last", "mean", "max")))
        if r2.get("std_mean_first_last"):
            L.append(f"    std_per_actuator mean over 15, first/last row: {r2['std_mean_first_last']}")
    except InputError as exc:
        r2 = dict(file=str(probe), error=str(exc))
        L.append(f"R2 (report only): unreadable ({exc})")
    tdir = Path(trace_dir) if trace_dir else (root / "traces" / name if kind == "continuation" else run / "trace")
    r3 = trace_reading(tdir / TRACE_NAME, end_zip(run, stop, T), T)
    L.append(f"R3 (report only) {r3['file']}  sha256 {r3.get('sha256')}: {r3['status']}")
    for pr in r3.get("problems") or []:
        L.append(f"    {pr}")
    if r3["status"] == "read":
        L.append(f"    end zip {r3['zip']}  sha256 {r3['zip_sha256']}")
        L.append(f"    CMD every actuator: {json.dumps(r3['cmd_all_actuators'])}")
        L.append(f"    tail flags {r3['n_tail_flags']}; falls {r3['n_fall_flags']}; episodes {r3['n_episodes']}, full "
                 f"horizon {r3['full_horizon_episodes']}; max |command| {r3['max_abs_command_any_step']}")
    lost_added = False
    if kind == "continuation" and not r1["c6_ok"]:
        first = r1["c6_counting_failures"][0]["failure"]
        lost_added = add_lost(root / "lost.json", name, "C6",
                              f"C6 failing up to T = {T}: {len(r1['c6_counting_failures'])} counting failures; first: "
                              f"{first}")
        L.append(f"  C6 FAILS: {name} {'written to' if lost_added else 'already in'} {root / 'lost.json'} (LOST)")
    rec = dict(schema=READ_SCHEMA, version=VERSION, tool_sha256=tool_sha256(), utc=utcnow(), run=str(run), kind=kind,
               name=name, start=start, arm=arm, seed=seed, S=S, T=T, stop_reason=stop, check_sha256=csha, r1=r1,
               r2={k: v for k, v in r2.items() if k != "arrays"} | {"arrays_file": str(out / "r2_arrays.json")},
               r3=r3, returns=returns, lost_written=lost_added)
    atomic_write_text(out / "r2_arrays.json", json.dumps(r2.get("arrays"), default=str) + "\n")
    atomic_write_text(out / READ_NAME, json.dumps(rec, indent=1, default=str) + "\n")
    L.append(f"wrote {out / READ_NAME}")
    return rec, L


def r1_lost(root, run, out, name, check, csha, exc, L):
    """R1 cannot run on a continuation's ledger: LOST, cause "R1" (criteria RUN HANDLING LOST; review RR-1).  The
    read.json written beside it carries no outcome."""
    T = check.get("T")
    m5_sha, common_sha = E.sha256_file(E.M5), E.sha256_file(E.M5_COMMON)
    lost_added = add_lost(root / "lost.json", name, "R1", f"R1 cannot run on the ledger up to T = {T}: {exc}")
    L.append(f"  R1 CANNOT RUN: {exc}")
    L.append(f"  {name} {'written to' if lost_added else 'already in'} {root / 'lost.json'} (LOST, cause R1)")
    r1 = dict(S=check.get("S"), T=T, unreadable=str(exc), positive=None, c6_ok=None, m5=E.M5, m5_sha256=m5_sha,
              m5_common_sha256=common_sha, m5_pins_ok=m5_sha == E.PIN_SHA256[E.M5]
              and common_sha == E.PIN_SHA256[E.M5_COMMON])
    rec = dict(schema=READ_SCHEMA, version=VERSION, tool_sha256=tool_sha256(), utc=utcnow(), run=str(run),
               kind="continuation", name=name, start=check.get("start"), arm=check.get("arm"), seed=check.get("seed"),
               S=check.get("S"), T=T, stop_reason=check.get("stop_reason"), check_sha256=csha, r1=r1, r2=None,
               r3=None, returns=None, lost_written=lost_added, lost_cause="R1")
    atomic_write_text(out / READ_NAME, json.dumps(rec, indent=1, default=str) + "\n")
    L.append(f"wrote {out / READ_NAME}")
    return rec, L


def ledger_read(ledger, S, T, out_dir, root, truncate=False):
    """R1 alone on one ledger (V4).  Returns (record, text lines).  Refused for a ledger under <root>/runs/ until all
    are decided (the same outcome-blind gate as the full read; review RR-7)."""
    out = require_under(out_dir, root)
    src = Path(ledger).resolve()
    if (Path(root).resolve() / "runs") in src.parents:
        ok, und = decided(root)
        if not ok:
            raise InputError(f"refused: {src} is a continuation's ledger and {len(und)} of {len(E.QUEUE)} "
                             "continuations are not decided (outcome-blind until all are)")
    out.mkdir(parents=True, exist_ok=True)
    L = [VERSION, f"tool sha256 {tool_sha256()}  utc {utcnow()}", f"ledger {src}  sha256 {E.sha256_file(src)}"]
    target = src
    if truncate:
        target = out / f"eval_episodes_le_{int(T)}.csv"
        n, k = truncate_ledger(src, target, T)
        L.append(f"  truncated to timesteps <= {T:,}: {k} of {n} rows -> {target}  sha256 {E.sha256_file(target)}")
    m5, jsha, _stdout, m5_sha, common_sha, argv = run_m5(target, T, out)
    r1 = window_reading(m5, S, T)
    r1.update(ledger=str(target), ledger_sha256=E.sha256_file(target), m5_sha256=m5_sha, m5_common_sha256=common_sha,
              m5_pins_ok=m5_sha == E.PIN_SHA256[E.M5] and common_sha == E.PIN_SHA256[E.M5_COMMON], m5_argv=argv,
              m5_json=str(out / "m5.json"), m5_json_sha256=jsha)
    L.append(f"  M5 sha256 {m5_sha}; common {common_sha}; JSON {out / 'm5.json'} sha256 {jsha}")
    render_r1(r1, L)
    rec = dict(schema=READ_SCHEMA, version=VERSION, tool_sha256=tool_sha256(), utc=utcnow(), kind="ledger",
               source=str(src), S=S, T=T, r1=r1, returns=ledger_returns(target, S, T))
    atomic_write_text(out / READ_NAME, json.dumps(rec, indent=1, default=str) + "\n")
    L.append(f"wrote {out / READ_NAME}")
    return rec, L


# ================================================================================================ main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", nargs="?")
    ap.add_argument("--check", metavar="RUN")
    ap.add_argument("--out")
    ap.add_argument("--trace", help="the folder holding the end pair's m1_summary.json (default <root>/traces/<name>)")
    ap.add_argument("--ledger")
    ap.add_argument("--S", type=int)
    ap.add_argument("--T", type=int)
    ap.add_argument("--truncate", action="store_true")
    ap.add_argument("--root", default=E.ECS, help="the screen's folder (tests only; default <ECS>)")
    a = ap.parse_args(argv)
    root = Path(a.root).resolve()
    try:
        if a.check:
            code, rec = check_run(a.check, root)
            print(f"{VERSION}\nCHECK {rec['run']}: {'PASS' if rec['passed'] else 'FAIL'} (start {rec.get('start')} arm "
                  f"{rec.get('arm')} seed {rec.get('seed')} T {rec.get('T')} stop {rec.get('stop_reason')})")
            for f in rec["failures"]:
                print(f"  FAIL {f}")
            if rec.get("r4"):
                d = rec["r4"]
                print(f"  R4 rows {d['rows_in_window']} of {d['rows_expected']} ({d['first_step']}..{d['last_step']}); "
                      f"ent_coef first/last {d.get('ent_coef_first')}/{d.get('ent_coef_last')}; seeded rows <= S "
                      f"{d['seeded_rows_at_or_before_S']}")
            for k, (path, sha) in rec["sources"].items():
                print(f"  read {k}: {path}  sha256 {sha}")
            print(f"  wrote {Path(rec['run']) / CHECK_NAME}")
            return code
        if a.ledger:
            if a.S is None or a.T is None or not a.out:
                raise InputError("--ledger needs --S, --T and --out")
            _rec, lines = ledger_read(a.ledger, a.S, a.T, a.out, root, truncate=a.truncate)
        elif a.run:
            _rec, lines = full_read(a.run, root, out_dir=a.out, trace_dir=a.trace)
        else:
            ap.error("give --check RUN, RUN, or --ledger")
    except InputError as exc:
        print(f"{VERSION}\nINPUT ERROR: {exc}")
        return EXIT_INPUT
    print("\n".join(lines))
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
