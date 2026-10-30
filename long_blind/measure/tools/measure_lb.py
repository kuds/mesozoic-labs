"""LB (long blind run) measurement: the end-of-run classification of a finished continuation with the r10 round's own
pinned tools.  Report only: it never trains, never steps an optimizer and never writes outside LB/measure.

    PY -I -B measure_lb.py version
    PY -I -B measure_lb.py pins
    PY -I -B measure_lb.py plan RUN [RUN ...] --out-root DIR [--suffix .copy1] [--start S]      (nothing simulated)
    PY -I -B measure_lb.py classify RUN [RUN ...] --out-root DIR [--suffix .copy1] [--slots 2] [--start S]
                                    [--prune-csv] [--no-simulate] [--parent-ledger CSV]

RUN is a finished continuation folder shaped like BS2's runs/<name>/ (01_stance/eval_episodes.csv, stage_config.json,
progress.json, models/stage1_<steps>_steps.zip + stage1_vecnormalize_<steps>_steps.pkl, models/stage1_final.zip +
stage1_final_vecnorm.pkl; run_meta.json and policy_probe.npz when present).  --suffix names a fetch store's copy suffix
(".copy1" for the round's F2 copies, the validation input); measure_r10round.py accepts the same suffix.  Each RUN's
outputs go to <out-root>/<name>/ (name = the folder name, or the parent folder's for an F2 "copies" folder).

WHAT IS COMPUTED (criteria = S2/r10round/final_lean/criteria_r10round_final.txt, sha256 99a44469..., the round's
approved text; "line N" below is a line of that file).  Every rule is executed by the round's pinned code; this file
only finds the files, launches the pinned runners, and passes their outputs to the pinned class function:
  M5   ledger_r10round.py (lane B, pinned 0d89bcb8...) on RUN/01_stance/eval_episodes.csv with --E E, as a
       subprocess (m5.json, m5.txt), and the same r10round_common.m5 in process for the classes (lines 134-142):
       ANY / TAIL / TAIL-UNRESOLVED counts per evaluation, ANY-FLAGGED and ANY ALARM (lines 169-170), TAIL-FLAGGED
       and TAIL ALARM (lines 171-172), NEEDS RESOLUTION (lines 173-180); C6 on the whole ledger to E (lines 234-245).
  M1   measure_r10round.py m1 (lane A, pinned b293a511...) on every checkpoint C7 names (line 246): the J checkpoints
       up to E (line 84: 4,194,304; 5,242,880; 6,291,456; 8,388,608; 10,485,760; final), R1 (line 181) and every
       RESOLUTION checkpoint (lines 175-177), chosen by complete_r10round.c3_pairs (pinned d322ed0e...); the trace
       tool is the measuring copy's episode_trace_report.py (sha256 b5923e5c..., checked by the runner).
  M3   measure_r10round.py m3 at stage1_final of a FINISHED run (lines 126-128, 246-247): the copy's
       stance_gate_report.py (sha256 bbfe8016...), the six-heading certificate; PASS = its own verdict.
  CLASSES  score_r10round.classes_of (lane B, pinned 3de2330a...), the function the round's scorer applied to its
       three runs, on {M5, the traces read by r10round_common.read_trace, the certificate read by read_certificate or
       the runner's refusal record}: TAIL CLASS (lines 269-279), GATE CLASS (lines 280-283), RUN OUTCOME (lines
       284-286); binding of every trace and certificate to the measured pair by complete_r10round.binding_problems.
  ENDING  C5's facts that apply to a continuation (lines 228-233): progress.json "ended" at E; stage1_final.zip's
       recorded num_timesteps = E (read by r10round_common.zip_num_timesteps, testzip first); FINISHED when E =
       12,001,280; STOPPED when E < 12,001,280 by the trainer's own stop (run_meta.json stop_reason "collapse" or
       "hop_watch", or 01_stance/hop_watch.json present); anything else is not classified (exit 2).
  NOT APPLIED  C1-C4 (RUNLOG launches, the Drive manifest, provenance sessions and "no load key": a continuation is a
       same-stage resume by construction, so it is never a run of the round; LB's own criteria decide whether it is
       counted).  The classes are the round's definitions applied to the continuation's records as measured.
  REPORT ONLY (never a class): the window (S, E] reading (alarms whose two evaluations both lie after S, alarms
       straddling S; BS2 read_bs.m5_window's form), the parent prefix check, the PER-ACTUATOR HABIT SUMMARY (D-1,
       lines 336-339: "the same for every other actuator"; M1's per-actuator outputs, line 119; the probe's
       rollout_mu_mean, BS2's primary form), costs.
Outcome-blind guard (BS2 read_bs.decided's rule, = LB/tools/read_bs.decided as of 2026-10-30 16:56 UTC): a RUN anywhere
under LB is read only when every continuation queued in LB/SCREEN.json has a passing LB/runs/<name>/check.json (mode
"continuation") or is in LB/lost.json; otherwise refused, nothing read.  A folder outside LB is a validation input.
Machine-wide: before each launch, at most 4 python script processes (any agent's; this tool's idle wrappers aside)
may run, else the launch waits.  classify_one(panels_dir=, filed_run_id=) reads M1/M3 outputs from another folder:
a validation reading only (reread_round_panels.py), refused for a run under LB.
Disk: measure_r10round.py refuses below 3 GiB free (r10m_common.DISK_FLOOR_BYTES); this tool checks the floor plus
the job's output before each launch.  --prune-csv deletes, after a trace is read, the per-step CSVs (.csv.gz) of the
episodes in which no actuator is over the BAR (their sha256 stay in csv_sha256.txt and m1_summary.json; lane B's
read_trace needs the CSV only of an episode with a tail actuator over the BAR).
Exit codes: 0 classified (every RUN); 2 refused (pins, guard, inputs, a runner refusal; nothing measured for that RUN);
3 a measurement crashed twice; 4 a consistency check failed.
"""
from __future__ import annotations

import os
import sys

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True

import argparse  # noqa: E402
import csv  # noqa: E402
import hashlib  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import subprocess  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402

VERSION = "measure_lb 1.0 (2026-10-30; LB measurement builder; the r10 round's pinned M1/M3/M5 and classes)"

# ------------------------------------------------------------------------------------------------------------ paths
S1 = "/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad"
S2 = "/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad"
LB = S2 + "/r10round/long_blind"
MEASURE = LB + "/measure"
LB_RUNS = LB + "/runs"                     # the continuations: the outcome-blind guard applies under it
LB_SCREEN = LB + "/SCREEN.json"            # the queue (BS2's bs-screen/1 form)
LB_LOST = LB + "/lost.json"                # BS2's bs-lost/1 form
CHECK_JSON = "check.json"                  # BS2's per-run check (bs-check/1: passed, mode)
WORK = MEASURE + "/work"
PY = S2 + "/venv/bin/python"
CODE = S2 + "/r10round/bias_screen/code"   # BS's sealed copy of 8737cd3d (read only; the continuations' own code)
CODE_COMMIT = "8737cd3d86431de1bdd0fcdb49a0d956a45e084d"
CODE_TREE = "331fe8fc3f6bdf73acc0d3cc324dd753d9d9d695d7ce914e4e011d5c6437ebf1"
CODE_FILES = 1406
ROUND_APPROVED = S1 + "/foot/stance6/APPROVED"   # the round's approval record (branch_head 8737cd3d)
CRITERIA = S2 + "/r10round/final_lean/criteria_r10round_final.txt"
LANE_A = S2 + "/r10round/lean/build/laneA/tools"
LANE_B = S2 + "/r10round/lean/build/laneB/tools"
MEASURE_R10 = LANE_A + "/measure_r10round.py"
LEDGER_R10 = LANE_B + "/ledger_r10round.py"
MPL_DIR_OUTSIDE = S2 + "/caches/mpl"       # r10m_common.child_env's MPLCONFIGDIR (watched, never written by us)
# The parent's ledger (s42, the F2 copy BS2 pinned: FMAN line 71), for the prefix check of a continuation (S > 0)
PARENT_LEDGER = S1 + "/foot/stance6/fetch/F2/stance6_r10LTSB_s42/copies/01_stance/eval_episodes.csv.copy1"
PARENT_LEDGER_SHA256 = "d1bcb78964691cf194e6276f06740d1f24aa69ad36831a046ad6307c1c4db11e"

# ------------------------------------------------------------------------------------------------------------ pins
PINS = {                                   # criteria PINS lines 749-771 and S1/foot/stance6/APPROVED
    CRITERIA: "99a44469a387d01c55b98d76595d3633029681866ed14afcb185521eca0164db",
    ROUND_APPROVED: "072464809f9c0072eb17b3c05821ed0d154ee0378c09bef757e70329240001bf",
    MEASURE_R10: "b293a511d8beb5b6d6f24cec8744e601382748ce9b2f89e6bb653b6c97f5840c",
    LANE_A + "/r10m_common.py": "97431323a84a45026671d681dbd0dc5fe3914aa4068ab20a49e428620b26f5bf",
    LEDGER_R10: "0d89bcb858e19f780c3ce0ac7a5886fc01b33d78164d9ee37d2e50dc1d4b0511",
    LANE_B + "/r10round_common.py": "ea2c5ea6984b8dc00c1c066a4441a59526c515ceb111da17eaf1688a009226c5",
    LANE_B + "/complete_r10round.py": "d322ed0efddec7a4f5f5eb75964fdfdec847c50fc5a4bb07fb7762d1696d533b",
    LANE_B + "/score_r10round.py": "3de2330a4e0451d8c48fa05b53507f6921352e537f15fd5f28558946ba080707",
    CODE + "/environments/shared/scripts/episode_trace_report.py":
        "b5923e5cfbcafd9f6e108df207a268500313de9bff44572121c8e3b48aad0af2",
    CODE + "/environments/shared/reporting/episode_trace.py":
        "96ced68e124ddf4fa225ce32d93831a3e3261a2ba5b3ef502a0b900ce7b16c0b",
    CODE + "/environments/shared/scripts/stance_gate_report.py":
        "bbfe80168c200332b0997181a1efbb13b01b522eb8b755203a85b5a0eeaf299b",
}
STACK = {"python": "3.13.17", "torch": "2.13.0+cpu", "numpy": "2.5.4", "gymnasium": "1.4.0",
         "stable_baselines3": "2.9.0", "mujoco": "3.10.0"}     # BS2 bs_common.STACK (the round's measuring stack)

# ------------------------------------------------------------------------------------------------------------ grid
FINAL = 12_001_280                         # line 55 / line 84 (12,000,000 rounded up to the 8,192 rollout)
LB_S = 2_097_152                           # LB's start: s42's 2,097,152 pair (the study's design)
GRID = 131_072                             # line 82
DISK_FLOOR = 3 * 1024**3                   # r10m_common.DISK_FLOOR_BYTES (THIS12 line 958)
M1_OUT_BYTES = 20 * 1024**2                # one M1 output before pruning (about 17.9 MB measured, see NOTES)
M3_OUT_BYTES = 4 * 1024**2                 # one M3 output (about 2.5 MB measured)
DISK_MARGIN = 64 * 1024**2
MACHINE_SIM_MAX = 4                        # the task's limit: at most 4 such processes at once machine-wide
WRAPPERS = ("measure_lb.py", "measure_r10round.py", "validate_round.py", "mutation_checks.py")  # idle while measuring
D1_WINDOWS = ((0, 4_194_304), (4_194_304, 8_388_608), (8_388_608, FINAL + 1))   # line 338
HABIT_LINE = 0.60                          # the PARKED / DRIVEN line (lines 276-279), applied per actuator: report only
LAST_UPDATES = 8                           # BS2's primary form: the last 8 probe rows before E

CLAUSES = {
    "M5": "criteria lines 134-142 (M5 LEDGER READER); ledger_r10round.py, r10round_common.m5",
    "ANY ALARM": "lines 169-170 (ANY-FLAGGED EVALUATION: >= 5 of 10 over the bar on any actuator; ANY ALARM: the "
                 "second of two consecutive)",
    "TAIL ALARM": "lines 171-172 (TAIL-FLAGGED EVALUATION: >= 5 of 10 over the bar with a tail actuator named; TAIL "
                  "ALARM: the second of two consecutive)",
    "RESOLUTION": "lines 173-180 (TAIL-UNRESOLVED, NEEDS RESOLUTION, RESOLUTION CHECKPOINT, RESOLVED BY TRACE)",
    "R1 R2 CMD MEAN": "lines 181-187 (R1 10,485,760; R2 stage1_final; CMD_a over episodes beyond step 200; MEAN_a)",
    "TRACE RAIL": "lines 188-191 (>= 2 full-horizon TAIL FLAGS on one actuator and sign, or 1 with MEAN_a >= 0.60; "
                  "DRIVE-BACKED)",
    "TAIL FLAG": "lines 165-168 (TAIL ACTUATORS 11-14, PITCH 11/13/14; BAR share at |command| >= 0.99 > 0.10; "
                 "full-horizon 1000 steps; tie counts in each sign)",
    "TAIL CLASS": "lines 269-279 (RAIL kinds 270-272; UNREAD 273-275; PARKED 276-277; CLEAR 278; DRIVEN 279)",
    "GATE CLASS": "lines 280-283 (PASS: FINISHED and the stage1_final certificate passes; FAIL: fails or refused, "
                  "and every STOPPED run)",
    "RUN OUTCOME": "lines 284-286 (POSITIVE = CLEAR and GATE PASS; NEGATIVE = RAIL, UNREAD, PARKED or GATE FAIL)",
    "M1": "lines 119-121 (episode_trace_report.py, seeds 3042-3061, deterministic, every actuator, window 200..L-1)",
    "M3": "lines 126-128 (stance_gate_report.py, six headings, seeds 3042-3081; PASS = its own verdict; a "
          "measurement on a screen task, not certification)",
    "J": "line 84 (J = 4,194,304; 5,242,880; 6,291,456; 8,388,608; 10,485,760; final)",
    "C5": "lines 228-233 (progress.json 'ended' at E; stage1_final.zip num_timesteps E; FINISHED / STOPPED)",
    "C6": "lines 234-245 (every evaluation of the grid up to E, 10 rows each, every ANY and TAIL input measured)",
    "C7": "lines 246-247 (R1, R2, J up to E and every RESOLUTION checkpoint traced; R2's certificate for FINISHED)",
    "RERUN": "lines 428-434 (a crash is rerun on the same files; a measuring tool's refusal is final)",
    "D-1": "lines 336-339 (per evaluation by named actuator and sign; windows [0,4194304), [4194304,8388608), "
           "[8388608,12001280]; the same for every other actuator; report only)",
    "MEASURING COPY": "lines 116-118 (the HEAD the runs trained, python -I -B, OMP 1 thread, nice 10, <= 2 processes)",
}


class Refusal(Exception):
    def __init__(self, message: str, code: int = 2):
        super().__init__(message)
        self.message = message
        self.code = code


# ------------------------------------------------------------------------------------------------------------ helpers
def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def require_under(path, root=MEASURE) -> Path:
    """Every file this tool writes lies under LB/measure (the task's write rule)."""
    p = Path(os.path.realpath(path))
    r = Path(os.path.realpath(root))
    if p != r and r not in p.parents:
        raise Refusal(f"{path} is not under {root}: refused (this tool writes only under LB/measure)")
    return p


def write_json(path, obj) -> str:
    p = require_under(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".partial")
    tmp.write_text(json.dumps(obj, indent=1, default=str) + "\n", encoding="utf-8")
    tmp.replace(p)
    return sha256_file(p)


def write_text(path, text) -> str:
    p = require_under(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".partial")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(p)
    return sha256_file(p)


def free_bytes(path) -> int:
    p = Path(path)
    while not p.exists():
        p = p.parent
    st = os.statvfs(p)
    return st.f_bavail * st.f_frsize


def stack_versions() -> dict:
    import importlib.metadata
    import platform

    out = {"python": platform.python_version()}
    for key in ("torch", "numpy", "gymnasium", "stable_baselines3", "mujoco"):
        try:
            out[key] = importlib.metadata.version(key)
        except importlib.metadata.PackageNotFoundError:
            out[key] = None
    return out


def check_pins() -> list[str]:
    """Every pinned file at its sha256, the measuring copy's seal values, the stack."""
    out = []
    for path, want in PINS.items():
        try:
            got = sha256_file(path)
        except OSError as exc:
            out.append(f"{path}: {exc}")
            continue
        if got != want:
            out.append(f"{path}: sha256 {got} != pinned {want}")
    try:
        seal = json.loads(Path(CODE, ".r10m_seal.json").read_text())
        if seal.get("commit") != CODE_COMMIT or seal.get("tree_digest") != CODE_TREE or seal.get("files") != CODE_FILES:
            out.append(f"{CODE}: seal {seal.get('commit')} / {seal.get('tree_digest')} / {seal.get('files')} is not "
                       f"{CODE_COMMIT} / {CODE_TREE} / {CODE_FILES}")
    except (OSError, ValueError) as exc:
        out.append(f"{CODE}: seal unreadable ({exc})")
    try:
        if json.loads(Path(ROUND_APPROVED).read_text()).get("branch_head") != CODE_COMMIT:
            out.append(f"{ROUND_APPROVED}: branch_head is not {CODE_COMMIT}")
    except (OSError, ValueError) as exc:
        out.append(f"{ROUND_APPROVED}: unreadable ({exc})")
    got = stack_versions()
    out += [f"stack {k}: running {got.get(k)} != {v}" for k, v in STACK.items() if got.get(k) != v]
    return out


_LANE_B = {}


def lane_b():
    """The round's lane B modules, loaded by path after their sha256 check: (r10round_common, ledger_r10round,
    complete_r10round, score_r10round).  score_r10round loads the other three from its own folder (its _load)."""
    if _LANE_B:
        return _LANE_B["C"], _LANE_B["LG"], _LANE_B["CP"], _LANE_B["SC"]
    names = ("r10round_common.py", "ledger_r10round.py", "complete_r10round.py", "score_r10round.py")
    bad = [p for p in check_pins() if any(p.startswith(f"{LANE_B}/{n}") for n in names)]
    if bad:
        raise Refusal("lane B tools not at their pins: " + "; ".join(bad))
    spec = importlib.util.spec_from_file_location("score_r10round", f"{LANE_B}/score_r10round.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["score_r10round"] = mod
    spec.loader.exec_module(mod)
    for n in names:                        # the files read are still the pinned ones
        if sha256_file(f"{LANE_B}/{n}") != PINS[f"{LANE_B}/{n}"]:
            raise Refusal(f"{LANE_B}/{n} changed while it was loaded")
    if Path(mod.C.__file__).resolve() != Path(LANE_B, "r10round_common.py").resolve():
        raise Refusal(f"r10round_common was loaded from {mod.C.__file__}, not {LANE_B}")
    _LANE_B.update(C=mod.C, LG=mod.LG, CP=mod.CP, SC=mod)
    return mod.C, mod.LG, mod.CP, mod


# ------------------------------------------------------------------------------------------------------------ inputs
def run_name(run_dir: Path) -> str:
    """The folder's name; for an F2 store's 'copies' folder, its parent's (the RUN_ID)."""
    return run_dir.parent.name if run_dir.name == "copies" else run_dir.name


def locate(run_dir: Path, rel: str, suffix: str = "") -> Path | None:
    p = run_dir / (rel + suffix)
    return p if p.is_file() else None


def need(run_dir: Path, rel: str, suffix: str) -> Path:
    p = locate(run_dir, rel, suffix)
    if p is None:
        raise Refusal(f"{run_dir}: {rel}{suffix} is missing")
    return p


def pair_rel(C, label) -> tuple[str, str]:
    return C.pair_paths("final" if label == "final" else int(label))


def blind_guard(run_dir: Path) -> dict:
    """BS2 read_bs.decided(): a RUN anywhere under LB is read only when every continuation queued in LB/SCREEN.json
    is decided (its LB/runs/<name>/check.json passed with mode "continuation", or it is in LB/lost.json).  A folder
    outside LB (the round's F2 copies, BS2's runs) is a validation input: no guard."""
    real = Path(os.path.realpath(run_dir))
    runs = Path(os.path.realpath(LB_RUNS))
    if Path(os.path.realpath(LB)) not in real.parents:
        return {"applies": False, "note": "outside LB: a validation input (no outcome-blind guard)"}
    try:
        screen = json.loads(Path(LB_SCREEN).read_text())
    except (OSError, ValueError) as exc:
        raise Refusal(f"outcome-blind guard: {LB_SCREEN} unreadable ({exc}); nothing read") from exc
    queue = screen.get("queue") if isinstance(screen, dict) else None
    if not isinstance(queue, list) or not queue:
        raise Refusal(f"outcome-blind guard: {LB_SCREEN} holds no queue; nothing read")
    lost = {}
    if Path(LB_LOST).exists():
        try:
            d = json.loads(Path(LB_LOST).read_text())
            lost = dict(d.get("lost") or {})
        except (OSError, ValueError, TypeError, AttributeError) as exc:
            raise Refusal(f"outcome-blind guard: {LB_LOST} unreadable ({exc})") from exc
    undecided = []
    for item in queue:
        name = f"{item[0]}_{int(item[1])}"
        if name in lost:
            continue
        try:
            c = json.loads((runs / name / CHECK_JSON).read_text())
        except (OSError, ValueError):
            undecided.append(name)
            continue
        if c.get("passed") is not True or c.get("mode") != "continuation":
            undecided.append(name)
    if undecided:
        raise Refusal(f"outcome-blind guard: {len(undecided)} queued continuation(s) not decided "
                      f"({', '.join(undecided)}); nothing read")
    return {"applies": True, "queue": queue, "lost": sorted(lost), "screen_sha256": sha256_file(LB_SCREEN)}


def read_ending(C, run_dir: Path, suffix: str) -> dict:
    """C5's facts that apply to a continuation (lines 228-233)."""
    prog_p = need(run_dir, "01_stance/progress.json", suffix)
    prog = json.loads(prog_p.read_text())
    zip_p = need(run_dir, "01_stance/models/stage1_final.zip", suffix)
    nts, zprob = C.zip_num_timesteps(zip_p.read_bytes())
    meta_p = locate(run_dir, "run_meta.json", suffix)
    meta = json.loads(meta_p.read_text()) if meta_p else None
    hop = locate(run_dir, "01_stance/hop_watch.json", suffix)
    gv_p = locate(run_dir, "01_stance/gate_verdict.json", suffix)
    E = prog.get("timesteps") if prog.get("status") == "ended" and isinstance(prog.get("timesteps"), int) else None
    out = dict(progress_status=prog.get("status"), progress_timesteps=prog.get("timesteps"),
               progress_written_at=prog.get("written_at"), final_zip_num_timesteps=nts, zip_problem=zprob,
               run_meta_stop_reason=(meta or {}).get("stop_reason"), run_meta_status=(meta or {}).get("status"),
               run_meta_S=(meta or {}).get("S"), hop_watch_present=hop is not None,
               gate_verdict_present=gv_p is not None, E=E, clause=CLAUSES["C5"])
    if E is None:
        raise Refusal(f"{run_dir}: progress.json is not 'ended' with an integer timesteps ({prog.get('status')!r}, "
                      f"{prog.get('timesteps')!r}): not a finished run (C5)")
    if zprob or nts != E:
        raise Refusal(f"{run_dir}: stage1_final.zip records num_timesteps {nts!r} ({zprob}), progress.json ended at "
                      f"{E}: C5 fails")
    if E > FINAL:
        raise Refusal(f"{run_dir}: E = {E} beyond {FINAL}")
    # the trainer's own stops (line 232: only the collapse stop and the hop watch end train() early); LB's run_meta
    # names them "collapse" and "hop_watch" (LB/tools/bs_common.STOP_REASONS)
    own_stop = out["run_meta_stop_reason"] in ("collapse", "hop_watch") or out["hop_watch_present"]
    if E == FINAL:
        if meta is not None and out["run_meta_stop_reason"] not in (None, "end"):
            raise Refusal(f"{run_dir}: E = {FINAL} but run_meta stop_reason {out['run_meta_stop_reason']!r}")
        out["ending"] = "FINISHED"
    elif own_stop:
        out["ending"] = "STOPPED"
    else:
        raise Refusal(f"{run_dir}: ended at E = {E} < {FINAL} without the trainer's own stop (run_meta stop_reason "
                      f"{out['run_meta_stop_reason']!r}, hop_watch.json {'present' if hop else 'absent'}): neither "
                      "FINISHED nor STOPPED (C5); not classified")
    return out


# ------------------------------------------------------------------------------------------------------------ jobs
class Slots:
    """At most `slots` of this tool's measurements at once, and at most MACHINE_SIM_MAX simulation processes
    machine-wide (counted from /proc by SIM_MARKERS) when a job starts."""

    def __init__(self, slots: int, log):
        self.slots = slots
        self.lock = threading.Lock()
        self.log = log

    @staticmethod
    def machine_sims() -> int:
        """Python script processes machine-wide (any agent's), but this tool's own idle wrappers: every simulation
        or measurement script counts (a run_bs.py continuation or validation run, a probe, M1/M3's child tools)."""
        n = 0
        for d in Path("/proc").iterdir():
            if not d.name.isdigit():
                continue
            try:
                args = (d / "cmdline").read_bytes().decode("utf-8", "replace").split("\0")
            except OSError:
                continue
            if not args or not os.path.basename(args[0]).startswith("python"):
                continue
            script = next((a for a in args[1:] if a.endswith(".py")), None)
            if script is None or os.path.basename(script) in WRAPPERS:
                continue
            n += 1
        return n

    def wait_machine(self, name):
        waited = 0
        while self.machine_sims() >= MACHINE_SIM_MAX:
            if waited % 300 == 0:
                self.log(f"{name}: {self.machine_sims()} simulation processes machine-wide (max {MACHINE_SIM_MAX}); "
                         "waiting")
            time.sleep(15)
            waited += 15


def run_job(job: dict, slots: Slots, log) -> dict:
    """One measure_r10round.py invocation; a crash (exit 3) is rerun once on the same files (lines 428-434)."""
    attempts = []
    for attempt in (1, 2):
        need_bytes = DISK_FLOOR + (M3_OUT_BYTES if job["kind"] == "m3" else M1_OUT_BYTES) + DISK_MARGIN
        fb = free_bytes(job["out"])
        if fb < need_bytes:
            res = dict(attempt=attempt, rc=None, refused_here=f"free {fb} bytes < floor + output + margin "
                       f"{need_bytes}: not launched (measure_r10round.py refuses below 3 GiB)")
            attempts.append(res)
            log(f"{job['name']}: {res['refused_here']}")
            return dict(job, attempts=attempts, outcome="NOT LAUNCHED (disk)")
        slots.wait_machine(job["name"])
        tmp = require_under(Path(WORK) / "tmp" / f"{job['run']}_{job['label']}_{job['kind']}")
        tmp.mkdir(parents=True, exist_ok=True)
        env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON")}
        env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1",
                   TMPDIR=str(tmp))
        logp = require_under(Path(job["logs"]) / f"{job['label']}_{job['kind']}.try{attempt}.log")
        logp.parent.mkdir(parents=True, exist_ok=True)
        cmd = ["nice", "-n", "10", PY, "-I", "-B", MEASURE_R10, job["kind"], "--copy", CODE, "--zip", job["zip"],
               "--pkl", job["pkl"], "--stage-config", job["stage_config"], "--out", job["out"],
               "--approved", ROUND_APPROVED]
        t0, u0 = time.monotonic(), utcnow()
        with open(logp, "w") as fh:
            fh.write(f"start {u0}\ncmd: {' '.join(cmd)}\n")
            fh.flush()
            p = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, cwd=str(tmp),
                                 env=env)
            _pid, status, ru = os.wait4(p.pid, 0)
            p.returncode = rc = os.waitstatus_to_exitcode(status)
            wall = time.monotonic() - t0
            fh.write(f"end {utcnow()} rc={rc} wall_s={wall:.1f} cpu_user_s={ru.ru_utime:.1f} "
                     f"cpu_sys_s={ru.ru_stime:.1f} maxrss_kb={ru.ru_maxrss}\n")
        res = dict(attempt=attempt, rc=rc, start=u0, end=utcnow(), wall_s=round(wall, 1),
                   cpu_user_s=round(ru.ru_utime, 1), cpu_sys_s=round(ru.ru_stime, 1),
                   cpu_s=round(ru.ru_utime + ru.ru_stime, 1), maxrss_kb=ru.ru_maxrss, log=str(logp))
        attempts.append(res)
        log(f"{job['name']}: rc {rc}, wall {wall:.0f} s, cpu {res['cpu_s']:.0f} s (attempt {attempt})")
        if rc == 3:
            continue                        # a crash: rerun once on the same files (the runner moves it aside)
        break
    rc = attempts[-1]["rc"]
    refused_file = Path(job["out"]) / f"{job['kind']}_refused.json"
    if rc == 0 and job["kind"] == "m1" and job.get("prune"):
        try:                                # free the disk at once (the floor is close): see prune_csvs
            rec = json.loads((Path(job["out"]) / "m1_summary.json").read_text())
            job = dict(job, pruned=prune_csvs(Path(job["out"]), rec))
        except (OSError, ValueError) as exc:
            job = dict(job, pruned=f"not pruned: {exc}")
    if rc == 0:
        outcome = "MEASURED"
    elif rc == 2 and refused_file.is_file():
        outcome = "REFUSED BY THE MEASURING TOOL (final)"
    elif rc == 3:
        outcome = "CRASHED TWICE"
    elif rc == 5:
        outcome = "REFUSED: DISK FLOOR"
    else:
        outcome = f"REFUSED BY THE RUNNER (rc {rc}; nothing measured)"
    return dict(job, attempts=attempts, outcome=outcome)


# ------------------------------------------------------------------------------------------------------------ readings
def window_reading(C, res: dict, S: int, E: int) -> dict:
    """Report only: M5's flags and alarms read in (S, E] (BS2 read_bs.m5_window's form): an alarm counts in the window
    when both its evaluations lie after S; an alarm whose first evaluation is at or before S straddles S."""
    def inwin(s):
        return S < s <= E
    return dict(S=S, E=E,
                any_flagged=[s for s in res["any_flagged"] if inwin(s)],
                tail_flagged=[s for s in res["tail_flagged"] if inwin(s)],
                any_alarms=[s for s in res["any_alarms"] if S < s - C.GRID and s <= E],
                tail_alarms=[s for s in res["tail_alarms"] if S < s - C.GRID and s <= E],
                needs_resolution=[s for s in res["needs_resolution"] if S < s - C.GRID and s <= E],
                straddling_S=sorted({s for s in res["any_alarms"] + res["tail_alarms"] if s - C.GRID <= S < s}),
                before_S=dict(any_flagged=[s for s in res["any_flagged"] if s <= S],
                              tail_flagged=[s for s in res["tail_flagged"] if s <= S],
                              over_bar_episodes=sum(e["any"] for s, e in res["evals"].items() if s <= S)))


def prefix_check(run_ledger: Path, parent_ledger: Path | None, S: int) -> dict:
    """Report only: the continuation ledger's rows at or before S equal the parent's rows at or before S (text).
    Default parent: s42's F2 ledger copy (sha256 checked)."""
    if not S:
        return {"checked": False, "note": "S = 0: no continuation prefix"}
    if parent_ledger is None:
        parent_ledger = Path(PARENT_LEDGER)
        if sha256_file(parent_ledger) != PARENT_LEDGER_SHA256:
            return {"checked": False, "note": f"{parent_ledger} is not at its sha256 {PARENT_LEDGER_SHA256}"}

    def rows_upto(p):
        with open(p, newline="", encoding="utf-8") as fh:
            r = list(csv.reader(fh))
        ti = r[0].index("timesteps")
        return r[0], [x for x in r[1:] if x and float(x[ti]) <= S]
    h1, a = rows_upto(run_ledger)
    h2, b = rows_upto(parent_ledger)
    return {"checked": True, "parent_ledger": str(parent_ledger), "parent_sha256": sha256_file(parent_ledger),
            "header_equal": h1 == h2, "rows_run": len(a), "rows_parent": len(b), "rows_equal": a == b}


def _num(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def habit_summary(C, header, rows, res, classes, m1_records: dict, probe: Path | None, S: int, E: int,
                  js: list, r1, traced: list | None = None) -> dict:
    """Report only: per actuator (all 15), the ledger's over-bar episodes by named actuator and sign (D-1, lines
    336-339; the sign is M5 (b)'s saturated_actuator_positive_share >= 0.5 rule, applied to every actuator), the ANY
    ALARMs naming it (classes_of's any_alarms), its own alarm series (the ANY ALARM rule of lines 169-170 restricted to
    the episodes naming it), M1's window-mean command (cmd_all_actuators) and full-horizon over-BAR episodes with sign
    at every traced checkpoint (lane A's m1_summary.json), and the probe's on-policy mean command (rollout_mu_mean over
    the last 8 updates before E, BS2's primary form).  HABIT SITES: the round's own thresholds applied to each
    actuator (never a class): L = an own alarm; T = >= 2 full-horizon episodes over the BAR in one sign at one traced
    checkpoint (TRACE RAIL's first clause); D = |CMD| >= 0.60 in one sign at >= 2 J checkpoints, or |mean of CMD(R1),
    CMD(R2)| >= 0.60 (PARKED's and DRIVEN's line)."""
    names = C.ACTUATOR_NAMES
    by = {}
    for r in rows:
        s = C.grid_step(r.get("timesteps"))
        if s is not None:
            by.setdefault(s, []).append(r)
    per = {n: {"over_by_eval": {}, "sign_by_eval": {}} for n in names}
    for s in sorted(by):
        for r in by[s]:
            x = C.decode_row(r, header)
            if not x["any_over"] or x["act"] is None:
                continue
            n = names[x["act"]]
            pos = _num(r.get("saturated_actuator_positive_share"))
            sg = "?" if pos is None else ("+" if pos >= 0.5 else "-")
            per[n]["over_by_eval"][s] = per[n]["over_by_eval"].get(s, 0) + 1
            d = per[n]["sign_by_eval"].setdefault(s, {})
            d[sg] = d.get(sg, 0) + 1
    labels = [str(x) for x in js] + ([str(r1)] if r1 else []) + ["final"] + [str(x) for x in (traced or [])]
    labels = sorted(set(labels), key=lambda x: (x == "final", int(x) if x != "final" else 0))
    jl = [str(x) for x in js] + ["final"]
    probe_d = {}
    if probe is not None:
        try:
            import numpy as np
            with np.load(probe, allow_pickle=False) as z:
                ut = z["update_timesteps"]
                mu = z["rollout_mu_mean"]
            idx = [i for i in range(len(ut)) if ut[i] < E]
            last = idx[-LAST_UPDATES:]
            for a, n in enumerate(names):
                w = {}
                for lo, hi in D1_WINDOWS:
                    sel = [i for i in idx if lo <= ut[i] < hi]
                    w[f"[{lo},{hi})"] = float(np.mean(mu[sel, a])) if sel else None
                cont = [i for i in idx if ut[i] > S] if S else []
                probe_d[n] = dict(last8_before_E=float(np.mean(mu[last, a])) if last else None,
                                  last8_rows=[int(ut[i]) for i in last], windows=w,
                                  after_S=float(np.mean(mu[cont, a])) if cont else None)
        except Exception as exc:  # noqa: BLE001 - report only
            probe_d = {"error": repr(exc)}
    out = {}
    for a, n in enumerate(names):
        d = per[n]
        ev = d["over_by_eval"]
        flagged = sorted(s for s, k in ev.items() if k >= 5)
        own_alarms = [s for s in flagged if s - C.GRID in flagged]
        any_named = [x["step"] for x in classes.get("any_alarms", []) if n in (x.get("actuators") or {})]
        any_sole = [x["step"] for x in classes.get("any_alarms", []) if set((x.get("actuators") or {})) == {n}]
        win = {}
        for lo, hi in D1_WINDOWS:
            win[f"[{lo},{hi})"] = sum(k for s, k in ev.items() if lo <= s < hi)
        signs = {}
        for _s, sd in d["sign_by_eval"].items():
            for k, v in sd.items():
                signs[k] = signs.get(k, 0) + v
        m1 = {}
        for lab in labels:
            rec = m1_records.get(lab)
            if not rec:
                continue
            cm = (rec.get("cmd_all_actuators") or {}).get(str(a))
            pos = neg = tie = 0
            for ep in rec.get("episodes", []):
                ae = (ep.get("actuators") or {}).get(n) or {}
                if ep.get("full_horizon") and ae.get("over_bar"):
                    sg = ae.get("sign")
                    pos += sg == "+"
                    neg += sg == "-"
                    tie += sg == "="
            fall = sum(1 for ep in rec.get("episodes", []) if not ep.get("full_horizon")
                       and ((ep.get("actuators") or {}).get(n) or {}).get("over_bar"))
            m1[lab] = dict(cmd=cm, over_bar_full_plus=pos, over_bar_full_minus=neg, over_bar_full_tie=tie,
                           over_bar_short=fall)
        sites = []
        if own_alarms:
            sites.append("L")
        if any((v["over_bar_full_plus"] + v["over_bar_full_tie"]) >= 2 or (v["over_bar_full_minus"] +
                                                                              v["over_bar_full_tie"]) >= 2
               for v in m1.values()):
            sites.append("T")
        drive = False
        for sg in (1, -1):
            hits = [lab for lab in jl if lab in m1 and m1[lab]["cmd"] is not None and sg * m1[lab]["cmd"] >= HABIT_LINE]
            x = m1.get(str(r1), {}).get("cmd") if r1 else None
            y = m1.get("final", {}).get("cmd")
            mean = (x + y) / 2 if (x is not None and y is not None) else (x if x is not None else y)
            if len(hits) >= 2 or (mean is not None and sg * mean >= HABIT_LINE):
                drive = True
        if drive:
            sites.append("D")
        span = sorted(set(ev) | {int(lab) if lab != "final" else E for lab, v in m1.items()
                                  if v["over_bar_full_plus"] + v["over_bar_full_minus"] + v["over_bar_full_tie"]})
        out[n] = dict(index=a, tail=a in C.TAIL, ledger_over_total=sum(ev.values()), ledger_over_signs=signs,
                      ledger_over_by_window=win,
                      ledger_over_after_S=sum(k for s, k in ev.items() if s > S) if S else None,
                      first_over=min(ev) if ev else None, last_over=max(ev) if ev else None,
                      flagged_evaluations=flagged, own_alarms=own_alarms, any_alarms_naming=any_named,
                      any_alarms_sole=any_sole, m1=m1, probe=probe_d.get(n) if isinstance(probe_d, dict) else None,
                      habit_sites=sites, active_span=[span[0], span[-1]] if span else None)
    sites = {n: v["habit_sites"] for n, v in out.items() if v["habit_sites"]}
    return dict(per_actuator=out, habit_sites=sites, labels=labels,
                probe_error=probe_d.get("error") if isinstance(probe_d, dict) else None,
                note="report only; HABIT SITES apply the round's own lines to each actuator and are never a class")


# ------------------------------------------------------------------------------------------------------------ one run
def prepare(C, LG, CP, run_dir: Path, suffix: str, out_root: Path, start: int | None) -> dict:
    """Inputs, ending, M5 in process and the labels C7 names (nothing simulated)."""
    guard = blind_guard(run_dir)
    name = run_name(run_dir)
    out = require_under(out_root / name)
    ending = read_ending(C, run_dir, suffix)
    E = ending["E"]
    S = start if start is not None else int(ending.get("run_meta_S") or 0)
    S_note = None
    if guard["applies"] and S != LB_S:
        S_note = f"S = {S:,} is not LB's start {LB_S:,} (pass --start {LB_S} unless the pre-registration says otherwise)"
    sc = need(run_dir, "01_stance/stage_config.json", suffix)
    led = need(run_dir, "01_stance/eval_episodes.csv", suffix)
    header, rows = C.read_ledger(led)
    res = C.m5(header, rows)
    c6 = LG.c6_check(header, rows, E)
    _pairs, r1, js, resolution, _n = CP.c3_pairs({}, E, res)
    labels = sorted({*(str(s) for s in js), *([str(r1)] if r1 else []),
                     *(str(v) for v in resolution.values() if v != "final"), "final"},
                    key=lambda x: (x == "final", int(x) if x != "final" else 0))
    files, pairs = {}, {}
    for lab in labels:
        zr, pr = pair_rel(C, lab)
        zp, pp = need(run_dir, zr, suffix), need(run_dir, pr, suffix)
        zb, pb = zp.read_bytes(), pp.read_bytes()
        nts, prob = C.zip_num_timesteps(zb)
        want = E if lab == "final" else int(lab)
        if prob or nts != want:
            raise Refusal(f"{run_dir}: {zr}{suffix} records num_timesteps {nts!r} ({prob}), not {want}")
        files[zr], files[pr] = zb, pb
        pairs[lab] = dict(zip=str(zp), pkl=str(pp), zip_sha256=sha256_bytes(zb), pkl_sha256=sha256_bytes(pb),
                          num_timesteps=nts)
    probe = locate(run_dir, "01_stance/policy_probe.npz", suffix)
    meta = locate(run_dir, "run_meta.json", suffix)
    return dict(name=name, run_dir=str(run_dir), suffix=suffix, out=out, guard=guard, ending=ending, E=E, S=S,
                S_note=S_note,
                stage_config=str(sc), stage_config_sha256=sha256_file(sc), ledger=str(led),
                ledger_sha256=sha256_file(led), header=header, rows=rows, res=res, c6=c6, r1=r1, js=js,
                resolution=resolution, labels=labels, files=files, pairs=pairs, probe=str(probe) if probe else None,
                run_meta=str(meta) if meta else None, finished=ending["ending"] == "FINISHED",
                stopped=ending["ending"] == "STOPPED")


def run_m5_cli(prep: dict) -> dict:
    """The pinned M5 CLI on the ledger (m5.json, m5.txt), as BS2's reader ran it."""
    out = prep["out"]
    out.mkdir(parents=True, exist_ok=True)
    jp = require_under(out / "m5.json")
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON")}
    env.update(OMP_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
    cmd = [PY, "-I", "-B", LEDGER_R10, prep["ledger"], "--E", str(prep["E"]), "--json", str(jp)]
    t0 = time.monotonic()
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL, cwd=WORK,
                         env=env, text=True)
    so, se = p.communicate()
    wall = time.monotonic() - t0
    write_text(out / "m5.txt", so + ("\n[stderr]\n" + se if se else ""))
    if p.returncode != 0:
        raise Refusal(f"M5 exit {p.returncode}: {(so + se)[-400:]}")
    m5 = json.loads(jp.read_text())
    return dict(argv=cmd, rc=p.returncode, wall_s=round(wall, 2), json_sha256=sha256_file(jp), json=m5)


def m5_agrees(cli: dict, res: dict, c6: dict) -> list[str]:
    """The CLI's M5 JSON and the in-process m5 (the same pinned function) give the same flags, alarms and C6."""
    j = cli["json"]
    bad = []
    for k in ("any_alarms", "tail_alarms", "needs_resolution", "any_flagged", "tail_flagged"):
        if [int(x) for x in j.get(k, [])] != [int(x) for x in res[k]]:
            bad.append(f"{k}: CLI {j.get(k)} != in-process {res[k]}")
    if (j.get("c6") or {}).get("passed") != c6["passed"]:
        bad.append(f"C6: CLI {(j.get('c6') or {}).get('passed')} != in-process {c6['passed']}")
    return bad


def jobs_for(prep: dict, simulate: bool, prune: bool = False) -> list[dict]:
    out = prep["out"]
    jobs = []
    if prep["finished"]:
        jobs.append(("m3", "final"))
    jobs += [("m1", lab) for lab in prep["labels"]]
    todo = []
    for kind, lab in jobs:
        d = out / "panels" / lab / kind
        summary = d / f"{kind}_summary.json"
        refused = d / f"{kind}_refused.json"
        j = dict(run=prep["name"], name=f"{prep['name']} {lab} {kind}", kind=kind, label=lab,
                 zip=prep["pairs"][lab]["zip"], pkl=prep["pairs"][lab]["pkl"], stage_config=prep["stage_config"],
                 out=str(require_under(d)), logs=str(out / "logs"), prune=prune)
        if summary.is_file() or refused.is_file():
            j["outcome"] = "ALREADY MEASURED" if summary.is_file() else "ALREADY REFUSED BY THE MEASURING TOOL"
            j["attempts"] = []
        elif not simulate:
            j["outcome"] = "ABSENT (--no-simulate)"
            j["attempts"] = []
        todo.append(j)
    return todo


def prune_csvs(label_dir: Path, rec: dict) -> dict:
    """--prune-csv: delete the .csv.gz of episodes in which no actuator is over the BAR (sha256 kept)."""
    keep = {e["seed"] for e in rec.get("any_actuator_over_bar_episodes", [])}
    removed, kept, freed = [], [], 0
    for ep in rec.get("episodes", []):
        gz = label_dir / f"episode_trace_seed{ep['seed']}.csv.gz"
        if not gz.is_file():
            continue
        if ep["seed"] in keep:
            kept.append(gz.name)
            continue
        require_under(gz)
        freed += gz.stat().st_size
        gz.unlink()
        removed.append(gz.name)
    rec_p = label_dir.parent / f"{label_dir.name}_pruned.json"
    if removed:                             # one record per label, merged when an earlier call removed some
        old = json.loads(rec_p.read_text()) if rec_p.is_file() else {}
        write_json(rec_p, dict(utc=utcnow(), removed=sorted(set(old.get("removed", [])) | set(removed)), kept=kept,
                               freed_bytes=int(old.get("freed_bytes", 0)) + freed,
                               note="their sha256 are in csv_sha256.txt and m1_summary.json csv_sha256; kept: "
                                    "episodes with an actuator over the BAR (lane B read_trace needs the tail ones)"))
    return dict(removed=len(removed), kept=len(kept), freed_bytes=freed, record=str(rec_p) if rec_p.is_file() else None)


def classify_one(prep: dict, job_results: list[dict], m5cli: dict, parent_ledger, prune: bool,
                 panels_dir: Path | None = None, filed_run_id: str | None = None) -> dict:
    """panels_dir / filed_run_id (validation only): read M1/M3 outputs from another folder (read only), e.g. the
    round's own <ROUND>/panels/<RUN_ID>/, whose records were filed under that RUN_ID (binding_problems reads it)."""
    C, LG, CP, SC = lane_b()
    out = prep["out"]
    if panels_dir is not None and (prune or Path(os.path.realpath(LB)) in Path(os.path.realpath(prep["run_dir"])).parents):
        raise Refusal("panels from another folder are a validation reading only (never for a run under LB, never "
                      "with --prune-csv)")
    panels = CP.PanelLocator(Path(panels_dir) if panels_dir else out / "panels", "{label}/{kind}")
    E = prep["E"]
    traces, missing, refused, other = {}, [], [], []
    m1_records = {}
    for lab in prep["labels"]:
        tr, why = panels.trace(lab)
        if tr is not None:
            bad = CP.binding_problems(tr.get("binding") or {}, lab, prep["files"], E, filed_run_id)
            if bad:
                other.append(f"M1 {lab}: {bad}")
                tr = None
        elif why.startswith("REFUSED"):
            refused.append(why)
        else:
            missing.append(why)
        traces[lab] = tr
        if tr is not None:
            m1_records[lab] = tr["record"]
    cert, cert_why = None, None
    if prep["finished"]:
        cert, cert_why = panels.certificate("final")
        if cert is not None:
            bad = CP.binding_problems(cert.get("binding") or {}, "final", prep["files"], E, filed_run_id)
            if bad:
                other.append(f"M3 final: {bad}")
                cert = None
        elif cert_why.startswith("REFUSED"):
            ref = panels.refusal("final", "m3") or {}
            bad = CP.binding_problems(ref.get("binding") or {}, "final", prep["files"], E, filed_run_id)
            if bad:
                other.append(f"M3 final refusal: {bad}")
            else:
                rec = ref.get("record") or {}
                cert = dict(passed=False, refused=True, failures=[f"refused: {ref.get('message')}"], seeds=None,
                            physics_revision=rec.get("physics_revision"), commit=C.record_commit(rec), record=rec,
                            label_dir=str(panels.path("final", "m3")), binding=ref.get("binding"))
    c7 = dict(labels=prep["labels"], missing=missing, refused=refused, measured_on_another_file=other,
              certificate=("measured" if cert is not None and not cert.get("refused") else
                           "refused (GATE FAIL)" if cert is not None else
                           ("not needed (STOPPED)" if not prep["finished"] else f"absent: {cert_why}")),
              clause=CLAUSES["C7"])
    blocked = missing or refused or other or (prep["finished"] and cert is None)
    commits = sorted({t.get("commit") for t in traces.values() if t} | ({cert.get("commit")} if cert else set()))
    phys = sorted({t.get("physics_revision") for t in traces.values() if t} |
                  ({cert.get("physics_revision")} if cert else set()), key=str)
    result = dict(schema="lb-measure/1", version=VERSION, tool_sha256=sha256_file(Path(__file__).resolve()),
                  utc=utcnow(), name=prep["name"], run_dir=prep["run_dir"], S_note=prep.get("S_note"),
                  suffix=prep["suffix"], guard=prep["guard"], ending=prep["ending"], E=E, S=prep["S"],
                  stage_config=prep["stage_config"], stage_config_sha256=prep["stage_config_sha256"],
                  ledger=prep["ledger"], ledger_sha256=prep["ledger_sha256"], pairs=prep["pairs"],
                  labels=prep["labels"], r1=prep["r1"], js=prep["js"],
                  resolution={str(k): v for k, v in prep["resolution"].items()},
                  c6=dict(passed=prep["c6"]["passed"], n_grid=prep["c6"]["n_grid"],
                          failures=prep["c6"]["failures"][:20], n_failures=len(prep["c6"]["failures"]),
                          clause=CLAUSES["C6"]),
                  m5_cli=dict((k, v) for k, v in m5cli.items() if k != "json"),
                  m5_cli_agrees=m5_agrees(m5cli, prep["res"], prep["c6"]) or "equal",
                  c7=c7, measuring_commits=commits, physics_revisions=phys, jobs=job_results,
                  panels_source=str(panels.dir), panels_filed_run_id=filed_run_id,
                  clauses=CLAUSES, pins={p: PINS[p] for p in PINS})
    if commits and commits != [CODE_COMMIT]:
        blocked = True
        result["blocked_commit"] = f"measured on {commits}, not {CODE_COMMIT}"
    if phys and phys != [10]:
        blocked = True
        result["blocked_physics"] = f"physics_revision {phys}"
    if blocked:
        result["classes"] = None
        result["status"] = "NOT CLASSIFIED (C7: a measurement absent, refused or not of this pair)"
        return result
    comp = dict(m5=prep["res"], traces=traces, r1=prep["r1"], resolution=prep["resolution"], js=prep["js"],
                stopped=prep["stopped"], E=E, certificate=cert)
    classes = SC.classes_of(comp)
    result["classes"] = SC._jsonable(classes)
    result["certificate"] = None if cert is None else dict(
        passed=cert.get("passed"), refused=bool(cert.get("refused")), failures=cert.get("failures"),
        physics_revision=cert.get("physics_revision"), commit=cert.get("commit"),
        label=(cert.get("record") or {}).get("label"),
        headings=(cert.get("record") or {}).get("headings"), seconds=((cert.get("record") or {}).get(
            "certificate_run") or {}).get("seconds"))
    by_act = {}
    for x in classes["any_alarms"]:                 # classes_of's attribution: the actuators of the alarm's two evals
        for n in x["actuators"]:
            d = by_act.setdefault(n, dict(count=0, first=None, last=None, sole=0))
            d["count"] += 1
            d["first"] = x["step"] if d["first"] is None else min(d["first"], x["step"])
            d["last"] = x["step"] if d["last"] is None else max(d["last"], x["step"])
            d["sole"] += set(x["actuators"]) == {n}
    result["m5_summary"] = dict(any_flagged=prep["res"]["any_flagged"], tail_flagged=prep["res"]["tail_flagged"],
                                any_alarms=prep["res"]["any_alarms"], tail_alarms=prep["res"]["tail_alarms"],
                                needs_resolution=prep["res"]["needs_resolution"],
                                any_alarms_by_actuator=by_act,
                                over_bar_per_evaluation={s: {k: e[k] for k in ("n", "any", "tail", "unres", "tail_by",
                                                                               "any_by", "reasons")}
                                                         for s, e in sorted(prep["res"]["evals"].items())
                                                         if e["any"] or e["tail"] or e["unres"]},
                                d1=LG.d1_summary(prep["res"]), clause=CLAUSES["M5"],
                                note="every evaluation's line: m5.txt (ledger_r10round.py's own print)")
    result["window"] = window_reading(C, prep["res"], prep["S"], E)
    result["prefix"] = prefix_check(Path(prep["ledger"]), parent_ledger, prep["S"])
    result["habit"] = habit_summary(C, prep["header"], prep["rows"], prep["res"], classes, m1_records,
                                    Path(prep["probe"]) if prep["probe"] else None, prep["S"], E, prep["js"],
                                    prep["r1"], traced=prep["labels"])
    result["m1_cross_check"] = {lab: (t.get("cross_check") if t else None) for lab, t in traces.items()}
    if prune:
        result["pruned"] = {lab: prune_csvs(out / "panels" / lab / "m1", m1_records[lab]) for lab in m1_records}
    result["status"] = "CLASSIFIED"
    return result


# ------------------------------------------------------------------------------------------------------------ text
def _steps(xs, n=6):
    xs = list(xs)
    s = ", ".join(f"{x:,}" if isinstance(x, int) else str(x) for x in xs[:n])
    return s + (f", ... ({len(xs)} in all)" if len(xs) > n else "")


def render(r: dict) -> str:
    L = [VERSION, f"RUN {r['name']}  ({r['run_dir']}{' suffix ' + r['suffix'] if r['suffix'] else ''})",
         f"  ending (C5, lines 228-233): {r['ending']['ending']} at E = {r['E']:,}; S = {r['S']:,}; "
         f"run_meta stop_reason {r['ending']['run_meta_stop_reason']!r}",
         f"  ledger {r['ledger_sha256'][:16]}...; C6 (lines 234-245): {'PASS' if r['c6']['passed'] else 'FAIL'} "
         f"({r['c6']['n_grid']} grid evaluations, {r['c6']['n_failures']} failures)",
         f"  labels traced (C7, line 246): {', '.join(r['labels'])}; measuring commit {r['measuring_commits']}; "
         f"physics_revision {r['physics_revisions']}",
         f"  M5 CLI vs in-process: {r['m5_cli_agrees']}"]
    if r.get("S_note"):
        L.append(f"  NOTE: {r['S_note']}")
    if r.get("classes") is None:
        L.append(f"  STATUS: {r['status']}  {json.dumps(r['c7'])}")
        return "\n".join(L)
    c = r["classes"]
    m = r["m5_summary"]
    L += [f"  TAIL CLASS (lines 269-279): {c['cls']}{' (' + c['kind'] + ')' if c.get('kind') else ''}",
          f"    TAIL ALARMs (lines 171-172): {len(c['tail_alarms'])}; first {c['tail_alarms'][0] if c['tail_alarms'] else None}; "
          f"tail-flagged evaluations {c['flagged_evaluations']}; resolved by trace {c['resolved_by_trace']}",
          f"    alarm actuators: {sorted({k for d in c['tail_alarm_detail'] for k in d['actuators']})}",
          f"    TRACE RAIL (lines 188-191): {c['trace_rail']}; single tail flags {c['single_tail_flags']}",
          f"    PARKED (lines 276-277): {c['parked']}; DRIVEN (line 279): {c['driven']}",
          f"    MEAN_a: {json.dumps({k: (round(v, 4) if v is not None else None) for k, v in c['mean'].items()})}",
          f"  ANY ALARMs (lines 169-170): {len(c['any_alarms'])}; steps {_steps([x['step'] for x in c['any_alarms']])}; "
          f"by actuator {json.dumps(m['any_alarms_by_actuator'])}",
          "    over-bar episodes per evaluation (M5; every line in m5.txt): "
          + "; ".join(f"{int(s):,}: {e['any']} ({', '.join(f'{k} {v}' for k, v in e['any_by'].items())})"
                      for s, e in list(m["over_bar_per_evaluation"].items())[:8])
          + (f"; ... ({len(m['over_bar_per_evaluation'])} evaluations with any)"
             if len(m["over_bar_per_evaluation"]) > 8 else ""),
          f"    any-flagged evaluations {len(m['any_flagged'])}; first over-bar episode "
          f"{m['d1']['first_tail_saturated_episode']} (tail); NEEDS RESOLUTION {m['needs_resolution']}",
          f"  GATE CLASS (lines 280-283): {c['gate']}{' ' + json.dumps(c['gate_why']) if c['gate_why'] else ''}",
          f"    GATE CERTIFICATE (final checkpoint, measured on a screen task; not certification): "
          f"{json.dumps(r.get('certificate'), default=str)[:600]}",
          f"  RUN OUTCOME (lines 284-286): {c['outcome']} {c['reasons']}"]
    w = r["window"]
    L.append(f"  window (S, E] (report only): TAIL ALARMs {len(w['tail_alarms'])} (first "
             f"{w['tail_alarms'][0] if w['tail_alarms'] else None}), ANY ALARMs {len(w['any_alarms'])}, tail-flagged "
             f"{len(w['tail_flagged'])}, straddling S {w['straddling_S']}; before S: {w['before_S']}")
    if r["prefix"].get("checked"):
        L.append(f"  parent prefix (report only): rows equal {r['prefix']['rows_equal']} ({r['prefix']['rows_run']} "
                 f"rows), header equal {r['prefix']['header_equal']}")
    h = r["habit"]
    L.append("  PER-ACTUATOR HABIT SUMMARY (report only; D-1 and M1 per actuator; HABIT SITES L/T/D are the round's "
             "lines applied per actuator, never a class)")
    L.append(f"    habit sites: {json.dumps(h['habit_sites'])}")
    head = "    actuator             over(ledger) [0,4.19M) [4.19,8.39M) [8.39M,E]  own-alarms (first-last)   " \
           "ANY-alarms  M1 CMD at " + " ".join(f"{x:>9}" for x in h["labels"])
    L.append(head)
    for n, v in h["per_actuator"].items():
        wv = list(v["ledger_over_by_window"].values())
        oa = v["own_alarms"]
        cm = " ".join((f"{v['m1'][lab]['cmd']:+9.3f}" if v["m1"][lab]["cmd"] is not None else f"{'n/f':>9}")
                      if lab in v["m1"] else f"{'-':>9}" for lab in h["labels"])
        if v["ledger_over_total"] or oa or v["habit_sites"] or any(abs(v["m1"].get(lab, {}).get("cmd") or 0) >= 0.3
                                                                   for lab in h["labels"]):
            L.append(f"    {n:<20} {v['ledger_over_total']:>5} {wv[0]:>9} {wv[1]:>12} {wv[2]:>10}  "
                     f"{len(oa):>3} ({oa[0] if oa else '-'}-{oa[-1] if oa else '-'})"
                     f"{'':>4}{len(v['any_alarms_naming']):>5}       {cm}  {''.join(v['habit_sites'])}")
    L.append("    (rows shown: an actuator with an over-bar ledger episode, an own alarm, a habit site or |CMD| >= 0.3 "
             "at a traced checkpoint)")
    return "\n".join(L)


# ------------------------------------------------------------------------------------------------------------ main
def cmd_classify(a) -> int:
    out_root = require_under(Path(a.out_root))
    out_root.mkdir(parents=True, exist_ok=True)
    problems = check_pins()
    if problems:
        print("REFUSED: pins:\n  " + "\n  ".join(problems), file=sys.stderr)
        return 2
    C, LG, CP, SC = lane_b()
    lock = threading.Lock()
    biglog = require_under(out_root / "measure_lb.log")

    def log(msg):
        line = f"{utcnow()} {msg}"
        with lock:
            print(line, flush=True)
            with open(biglog, "a") as fh:
                fh.write(line + "\n")

    log(f"{VERSION}; runs {a.runs}; slots {a.slots}; prune {a.prune_csv}; simulate {not a.no_simulate}; "
        f"free {free_bytes(out_root):,} bytes; {MPL_DIR_OUTSIDE} exists {os.path.isdir(MPL_DIR_OUTSIDE)}")
    preps, rc = [], 0
    for rd in a.runs:
        try:
            t0 = time.monotonic()
            prep = prepare(C, LG, CP, Path(rd), a.suffix, out_root, a.start)
            prep["prep_s"] = round(time.monotonic() - t0, 2)
            prep["m5cli"] = run_m5_cli(prep)
            log(f"{prep['name']}: E {prep['E']:,} ({prep['ending']['ending']}), labels {prep['labels']}, "
                f"M5 CLI {prep['m5cli']['wall_s']} s")
            preps.append(prep)
        except Refusal as exc:
            log(f"{rd}: REFUSED: {exc.message}")
            rc = max(rc, exc.code)
    jobs = [j for p in preps for j in jobs_for(p, not a.no_simulate, a.prune_csv)]
    todo = [j for j in jobs if "outcome" not in j]           # per run: the certificate first (longest), then traces
    slots = Slots(a.slots, log)
    done = [j for j in jobs if "outcome" in j]
    t0 = time.monotonic()
    if todo:
        log(f"launching {len(todo)} measurement(s) on {a.slots} slot(s)")
        with ThreadPoolExecutor(max_workers=a.slots) as ex:
            done += list(ex.map(lambda j: run_job(j, slots, log), todo))
    log(f"measurements done in {time.monotonic() - t0:.0f} s wall; {MPL_DIR_OUTSIDE} exists "
        f"{os.path.isdir(MPL_DIR_OUTSIDE)}")
    parent_ledger = Path(a.parent_ledger) if a.parent_ledger else None
    summary = []
    for p in preps:
        jr = [j for j in done if j["run"] == p["name"]]
        for j in jr:
            j.pop("logs", None)
        if any(j["outcome"] in ("CRASHED TWICE",) for j in jr):
            rc = max(rc, 3)
        t1 = time.monotonic()
        res = classify_one(p, jr, p["m5cli"], parent_ledger, a.prune_csv)
        res["costs"] = dict(prepare_s=p["prep_s"], m5_cli_s=p["m5cli"]["wall_s"],
                            classify_s=round(time.monotonic() - t1, 2),
                            jobs={f"{j['label']} {j['kind']}": {k: j["attempts"][-1].get(k) for k in
                                                                 ("wall_s", "cpu_s", "cpu_user_s", "cpu_sys_s",
                                                                  "maxrss_kb")} if j["attempts"] else j["outcome"]
                                  for j in jr},
                            m1_cpu_s=round(sum(j["attempts"][-1].get("cpu_s") or 0 for j in jr
                                               if j["kind"] == "m1" and j["attempts"]), 1),
                            m3_cpu_s=round(sum(j["attempts"][-1].get("cpu_s") or 0 for j in jr
                                               if j["kind"] == "m3" and j["attempts"]), 1))
        res["free_bytes_after"] = free_bytes(out_root)
        res["mpl_dir_outside_exists"] = os.path.isdir(MPL_DIR_OUTSIDE)
        sha = write_json(p["out"] / "measure_lb.json", res)
        text = render(res)
        write_text(p["out"] / "report.txt", text + "\n")
        if res.get("classes") is not None:
            write_json(p["out"] / "classes.json", res["classes"])
            write_json(p["out"] / "habit.json", res["habit"])
        print(text, flush=True)
        log(f"{p['name']}: {res['status']}; measure_lb.json {sha[:16]}")
        if res["status"] != "CLASSIFIED":
            rc = max(rc, 2)
        c = res.get("classes") or {}
        summary.append(dict(run=p["name"], status=res["status"], cls=c.get("cls"), kind=c.get("kind"),
                            gate=c.get("gate"), outcome=c.get("outcome"),
                            tail_alarms=len(c.get("tail_alarms") or []),
                            first_tail_alarm=(c.get("tail_alarms") or [None])[0],
                            any_alarms=len(c.get("any_alarms") or []),
                            habit_sites=(res.get("habit") or {}).get("habit_sites")))
    write_json(out_root / "summary.json", dict(utc=utcnow(), version=VERSION, runs=summary, rc=rc))
    return rc


def cmd_plan(a) -> int:
    """Inputs, ending, M5 (in process only) and the jobs C7 names, with the disk they need; nothing simulated and
    nothing written.  It reads the ledger, so the outcome-blind guard applies."""
    problems = check_pins()
    if problems:
        print("REFUSED: pins:\n  " + "\n  ".join(problems), file=sys.stderr)
        return 2
    C, LG, CP, SC = lane_b()
    rc = 0
    for rd in a.runs:
        try:
            p = prepare(C, LG, CP, Path(rd), a.suffix, require_under(Path(a.out_root)), a.start)
        except Refusal as exc:
            print(f"{rd}: REFUSED: {exc.message}")
            rc = max(rc, exc.code)
            continue
        jobs = jobs_for(p, True)
        new = [j for j in jobs if "outcome" not in j]
        need_b = sum(M3_OUT_BYTES if j["kind"] == "m3" else M1_OUT_BYTES for j in new)
        print(json.dumps(dict(run=p["name"], E=p["E"], ending=p["ending"]["ending"], S=p["S"], labels=p["labels"],
                              resolution={str(k): v for k, v in p["resolution"].items()},
                              c6_passed=p["c6"]["passed"], jobs=[f"{j['label']} {j['kind']}: "
                                                                 f"{j.get('outcome', 'TO MEASURE')}" for j in jobs],
                              output_bytes_before_prune=need_b, free_bytes=free_bytes(p["out"].parent),
                              disk_floor=DISK_FLOOR), indent=1))
    return rc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("version")
    sub.add_parser("pins")
    pl = sub.add_parser("plan")
    pl.add_argument("runs", nargs="+")
    pl.add_argument("--out-root", required=True)
    pl.add_argument("--suffix", default="")
    pl.add_argument("--start", type=int, default=None)
    c = sub.add_parser("classify")
    c.add_argument("runs", nargs="+")
    c.add_argument("--out-root", required=True)
    c.add_argument("--suffix", default="", help="a fetch store's copy suffix, e.g. .copy1 (validation inputs)")
    c.add_argument("--slots", type=int, default=2, help="concurrent measurements (criteria line 117: at most 2)")
    c.add_argument("--start", type=int, default=None, help="S, the continuation's start (default run_meta.json S, "
                                                           "else 0)")
    c.add_argument("--prune-csv", action="store_true")
    c.add_argument("--no-simulate", action="store_true", help="read existing panels only; launch nothing")
    c.add_argument("--parent-ledger", default=None, help="the parent's ledger for the prefix check (report only)")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "version":
            print(VERSION)
            return 0
        if a.cmd == "pins":
            problems = check_pins()
            print("\n".join(problems) if problems else f"all {len(PINS)} pins, the copy's seal and the stack OK")
            return 2 if problems else 0
        if a.cmd == "plan":
            return cmd_plan(a)
        if a.slots < 1 or a.slots > 2:
            raise Refusal("--slots must be 1 or 2 (criteria line 117: at most 2 measuring processes)")
        return cmd_classify(a)
    except Refusal as exc:
        print(f"[measure_lb] REFUSED: {exc.message}", file=sys.stderr, flush=True)
        return exc.code


if __name__ == "__main__":
    raise SystemExit(main())
