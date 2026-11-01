"""The r10 round's M1/M2/M3 runner (lane A).  Report and class inputs only; never trains, never steps an optimizer,
never writes into a run folder or the measuring copy.

Criteria: S2/r10round/lean/v5/criteria_r10round_lean_v5.txt (the text), HOW IT IS MEASURED and CLASSES PER RUN, with
THIS12 lines 173-201 and 297-337 for the definitions it cites.  BUILD LIST 3: "measure_r10round.py (M1/M2/M3,
physics_revision check, flags, DRIVE-BACKED, hip roll, gzip, disk floor 3 GB free)".

    PY -I -B measure_r10round.py version
    PY -I -B measure_r10round.py build-copy --dest DIR [--commit SHA | --header-parent | --approved APPROVED]
    PY -I -B measure_r10round.py check-copy --copy DIR [--header-parent | --approved APPROVED]
    PY -I -B measure_r10round.py m1 --copy DIR --zip Z --pkl P (--stage-config SC | --no-run-records) \
                                     (--out DIR | --run-id RUN_ID --label LABEL) [--header-parent | --approved APPROVED]
    PY -I -B measure_r10round.py m2 --copy DIR (--zip Z --pkl P --statue-json PANEL_JSON | --statue) ...
    PY -I -B measure_r10round.py m3 --copy DIR (--zip Z --pkl P | --zero-action) ...

The commit the measuring copy must be (v5, VALIDATIONS: no tool holds the header commit): the branch HEAD read from
the approval record (default <ROUND>/APPROVED, "branch_head") and checked by r10m_common.approved_branch_head (its only
parent the header's parent cf9402bc, its tree changed only in stance.toml comment lines); or, with --header-parent,
the header's parent cf9402bc itself, the text's pin (validation inputs before the header commit is made; refused with
--run-id: a run's measurement is filed only on the approved HEAD's copy).  The basis is recorded in every output.

Every measurement, before anything is rolled: the version is printed; the disk floor (3 GB free, THIS12 line 958);
the measuring copy's seal (commit = the commit above, tree digest unchanged) and the pinned files' sha256; the
checkpoint's recorded physics_revision (read from the zip's `data` member, nothing unpickled) must be 10, else
REFUSED (V5); with --stage-config, the run's stage_config.json git_commit must be the copy's commit (the text: the
measuring copy is the HEAD the run trained), else REFUSED (a refusal on the run's own records, reported, RUN
HANDLING); the output must lie outside every run folder and the copy and must not hold a finished measurement (an
unfinished one, a crash, is moved aside to <out>.crashed<k> and measured again on the same files).  After: the copy's
seal again.  Every output records the checkpoint's (or, for the statue, the copy's) physics_revision.

M1 TRACE (THIS12 lines 173-180): the copy's episode_trace_report.py exactly as THIS12 line 175 writes it (seeds
3042-3061, deterministic), then, from episode_trace.json and the 20 per-step CSVs (cross-checked), m1_summary.json:
  per episode and actuator: window-mean command, raw mean and joint position; shares at |command| >= 0.99 and >= 0.9;
  |raw| > 1 share; first step at the rail; the window steps at command >= +0.99 and <= -0.99; return; length;
  BAR: share at |command| >= 0.99 > 0.10.  A tail actuator (11-14) over the BAR is a TAIL FLAG in a full-horizon
  (1000-step) episode and a FALL FLAG otherwise (report only), with its sign: "the sign of most of its saturated
  steps" (THIS12 line 308).  An exact tie of the two counts is recorded as sign "=" and (v5, TAIL FLAG: "on an exact
  tie the flag counts once in each sign") counted once in "+" and once in "-" (counts_in_signs);
  CMD_a (v5, CLASSES) = "the mean, over those of M1's 20 episodes at X that last beyond step 200, of each one's
  window-mean command of tail actuator a (the others are printed); with none, CMD_a(X) is not formed and reads as
  below 0.60 in either sign": the episodes averaged are those with a window (length > 200, window 200..L-1 not
  empty), their count recorded (cmd_a_episodes), the others listed (episodes_without_window);
  DRIVE-BACKED at this checkpoint per TAIL FLAG and per sign it counts in: CMD_a >= 0.60 in that sign (a CMD_a not
  formed reads as below 0.60); a TRACE RAIL is DRIVE-BACKED "when, at any checkpoint holding one of its flags, CMD_a
  is >= 0.60 in the flag's sign" (v5): the scorer reads this per checkpoint over R1 and R2;
  TAIL FLAGS per actuator and sign (RESOLVED BY TRACE and TRACE RAIL read them; the scorer forms the classes);
  hip roll (actuators 4 and 8): joint position present at every step, and its window means;
  the CSVs' sha256 go to csv_sha256.txt and the CSVs are gzipped (mtime 0, verified, then the CSV removed).
M2 POSTURE PANEL (THIS12 lines 181-193): zs_harness_copy.py (a byte copy beside this file, sha256 pinned) with
  --code <copy> --plant r9 ("the code copy's own plant", here r10) --mode det (or statue) --seed 3042 --episodes 40;
  a policy panel reads the statue panel.json an M2 statue run made on the same sealed copy.  m2_summary.json.
M3 CERTIFICATE (THIS12 lines 194-201): the copy's stance_gate_report.py (--model/--vecnorm or --zero-action, seeds
  3042-3081, the stage's 40 episodes, --out-dir); PASS = the certificate's own verdict.  m3_summary.json.

A child tool's own outcome (fixer, RA-1 and RA-5): the trace and certificate tools turn their own errors
(EpisodeTraceError, StanceGateReportError, the harness's plant check) into SystemExit(message), exit 1 and no
traceback; stance_gate_report.py also exits 1, after writing its report, whenever the certificate does not pass.
  - M3 exit 0 or 1 with stance_heading_report.json written and no traceback: the certificate's own verdict, PASS or
    FAIL, summarised in m3_summary.json (the exit code must agree with the report's "passed").
  - exit != 0, no output, no traceback, no signal: the tool REFUSED; m<k>_refused.json records its message, the
    checkpoint, the run records and the copy seal; exit 2; the folder is final (never moved aside, never rerun: RUN
    HANDLING allows a rerun only for a crash).  Which rule reads a refusal is the scorer's (lane B), not this runner's.
  - a traceback, a signal, or exit 0 without the tool's output: a CRASH (exit 3; a rerun moves the folder aside).
With --run-id (a run's panel, RA-4), before anything is rolled: the RUN_ID must be a round RUN_ID (or its _b), the
checkpoint's recorded seed and the stage config's run.seed the RUN_ID's seed, and the label the checkpoint's own:
an integer label equal to its recorded num_timesteps with the pair named stage1_<label>_steps.zip /
stage1_vecnormalize_<label>_steps.pkl, or "final" with stage1_final.zip / stage1_final_vecnorm.pkl (a fetch store's
".copy<k>" suffix allowed); --no-run-records is refused with --run-id.
Exit codes: 0 done; 2 refused (by this runner, or the child tool's refusal recorded); 3 the measuring tool crashed
(its folder is left; a rerun moves it aside); 4 an output consistency check failed; 5 below the disk floor.
"""
from __future__ import annotations

import os
import sys

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_k] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True

import argparse  # noqa: E402
import csv  # noqa: E402
import re  # noqa: E402
import gzip  # noqa: E402
import hashlib  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

VERSION = "measure_r10round 1.2 (2026-10-26; fixer: RA-1, RA-4, RA-5; v5: the approved HEAD, O-6, O-7, O-8)"
TOOLS = Path(__file__).resolve().parent


def _load_common():
    spec = importlib.util.spec_from_file_location("r10m_common", TOOLS / "r10m_common.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


C = _load_common()
Refusal = C.Refusal
SPECIES = ("--species", "tyrannosaurus_rex", "--embodiment", "anatomical")


# ------------------------------------------------------------------------------------------------ shared steps
def out_dir_of(ns, kind: str) -> Path:
    if ns.out:
        return Path(ns.out).resolve()
    if not (ns.run_id and ns.label):
        raise Refusal("pass --out, or --run-id and --label (outputs go to <ROUND>/panels/<RUN_ID>/<label>/)")
    return Path(C.PANELS) / ns.run_id / ns.label / kind


def refused_name(summary_name: str) -> str:
    return summary_name.replace("_summary.json", "_refused.json")


def prepare_out(out: Path, inputs, summary_name: str) -> None:
    """Refuse an output inside a run folder or the copy, or one that holds a finished measurement or a recorded
    refusal of the child tool (fixer, RA-5: never rerun); move an unfinished one (a crash) aside so the same files
    are measured again (RUN HANDLING: a measurement crash is rerun)."""
    C.refuse_output_in_run(out, inputs)
    if out.exists():
        if (out / summary_name).is_file():
            raise Refusal(f"{out} already holds a finished measurement ({summary_name}); nothing is overwritten")
        if (out / refused_name(summary_name)).is_file():
            raise Refusal(f"{out} holds the measuring tool's recorded refusal ({refused_name(summary_name)}); a "
                          "refusal is final and is not rerun (RUN HANDLING: reruns only for a crash)")
        k = 1
        while out.with_name(f"{out.name}.crashed{k}").exists():
            k += 1
        out.rename(out.with_name(f"{out.name}.crashed{k}"))
        print(f"an unfinished output was moved aside to {out.name}.crashed{k}", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)


def run_records(ns, seal: dict) -> dict:
    """The run's stage_config.json binding (the measuring copy is the HEAD the run trained), or the validation mark."""
    if ns.no_run_records:
        return {"validation_no_run_records": True,
                "note": "no run records given (a statue or synthetic validation input): commit binding not checked; "
                        "no scorer may read this output as a run's measurement"}
    if not ns.stage_config:
        raise Refusal("pass the run's --stage-config (01_stance/stage_config.json), or --no-run-records for a "
                      "validation input")
    with open(ns.stage_config) as fh:
        sc = json.load(fh)
    rec = {"validation_no_run_records": False, "stage_config": os.path.realpath(ns.stage_config),
           "stage_config_sha256": C.sha256_file(ns.stage_config), "git_commit": sc.get("git_commit"),
           "run_seed": (sc.get("run") or {}).get("seed"),
           "task_sha256": (sc.get("task_fingerprint") or {}).get("task_sha256"),
           "plant_physics_revision": (sc.get("plant_identity") or {}).get("physics_revision")}
    if sc.get("git_commit") != seal["commit"]:
        raise Refusal(f"{ns.stage_config}: the run trained git_commit {sc.get('git_commit')!r}, the measuring copy is "
                      f"{seal['commit']} (a refusal on the run's own records: reported, never COMPLETE)")
    return rec


ROUND_RUN_RE = re.compile(r"^stance6_r10LTSB_s(60|42|80)(_b)?$")
COPY_SUFFIX_RE = re.compile(r"\.copy\d+$")


def pair_names_for(label: str) -> tuple[str, str]:
    if label == "final":
        return "stage1_final.zip", "stage1_final_vecnorm.pkl"
    return f"stage1_{label}_steps.zip", f"stage1_vecnormalize_{label}_steps.pkl"


def check_run_label(ns, ck: dict | None) -> dict:
    """RA-4: a run's panel (--run-id) is filed only under its own RUN_ID and label (see the module docstring)."""
    if not ns.run_id:
        return {}
    m = ROUND_RUN_RE.match(ns.run_id)
    if not m:
        raise Refusal(f"--run-id {ns.run_id!r} is not a RUN_ID of the round (stance6_r10LTSB_s60/42/80 or its _b)")
    seed = int(m.group(1))
    if ns.no_run_records:
        raise Refusal("--run-id files a run's measurement: --no-run-records (a validation input) is refused with it")
    if ck is None:
        return {"run_id": ns.run_id, "seed": seed}
    label = ns.label or ""
    if label != "final" and not label.isdigit():
        raise Refusal(f"--label {label!r} is neither a checkpoint step nor 'final'")
    want_zip, want_pkl = pair_names_for(label)
    got_zip = COPY_SUFFIX_RE.sub("", os.path.basename(ns.zip))
    got_pkl = COPY_SUFFIX_RE.sub("", os.path.basename(ns.pkl))
    if (got_zip, got_pkl) != (want_zip, want_pkl):
        raise Refusal(f"label {label}: the pair is named {got_zip} / {got_pkl}, not {want_zip} / {want_pkl}")
    if label != "final" and ck.get("num_timesteps") != int(label):
        raise Refusal(f"label {label}: the checkpoint records num_timesteps {ck.get('num_timesteps')!r}")
    if ck.get("seed") != seed:
        raise Refusal(f"{ns.run_id}: the checkpoint records seed {ck.get('seed')!r}, not {seed}")
    if ns.stage_config:
        with open(ns.stage_config) as fh:
            rs = (json.load(fh).get("run") or {}).get("seed")
        if rs != seed:
            raise Refusal(f"{ns.run_id}: the stage config's run.seed is {rs!r}, not {seed}")
    return {"run_id": ns.run_id, "seed": seed, "label": label, "pair_names": [got_zip, got_pkl]}


def expected_commit(ns) -> dict:
    """v5: the commit the measuring copy must be (module docstring): the approved branch HEAD, checked; or, with
    --header-parent, the header's parent cf9402bc (validation inputs only; refused with --run-id)."""
    if getattr(ns, "header_parent", False):
        if getattr(ns, "run_id", None):
            raise Refusal("--header-parent measures validation inputs on the header's parent; a run's measurement "
                          "(--run-id) is made only on the copy of the branch HEAD the approval record names")
        if getattr(ns, "approved", None):
            raise Refusal("--header-parent and --approved exclude each other")
        return {"commit": C.HEADER_PARENT, "basis": "header-parent",
                "note": "the header's parent cf9402bc (D19), pinned by the text: a validation before the header commit"}
    h = C.approved_branch_head(getattr(ns, "approved", None))
    return {"commit": h["head"], "basis": "approved", "approval": h}


def preflight(ns, kind: str, *, checkpoint: bool, pins: dict | None, summary_name: str) -> dict:
    print(f"[{kind}] {VERSION}; {C.COMMON_VERSION}", flush=True)
    out = out_dir_of(ns, kind)
    free = C.refuse_below_disk_floor(out.parent if not out.exists() else out)
    expect = expected_commit(ns)
    seal = C.check_copy(ns.copy, expect_commit=expect["commit"])
    pinned = C.check_pins(ns.copy, pins) if pins else {}
    ck = None
    if checkpoint:
        for p in (ns.zip, ns.pkl):
            if not p or not os.path.isfile(p):
                raise Refusal(f"checkpoint file missing: {p}")
        ck = C.checkpoint_record(ns.zip)
        C.refuse_physics(ck)
        ck["vecnorm"] = os.path.realpath(ns.pkl)
        ck["vecnorm_sha256"] = C.sha256_file(ns.pkl)
    filed = check_run_label(ns, ck)
    records = run_records(ns, seal)
    prepare_out(out, [ns.zip if checkpoint else None, ns.pkl if checkpoint else None, ns.stage_config], summary_name)
    return {"out": out, "seal": seal, "pins": pinned, "checkpoint": ck, "run_records": records, "free_bytes": free,
            "filed": filed, "copy_commit": expect}


def run_child(cmd: list[str], log_dir: Path) -> dict:
    t0 = time.monotonic()
    proc = subprocess.run(cmd, env=C.child_env(), cwd=str(log_dir), capture_output=True, text=True)
    return {"argv": cmd, "returncode": proc.returncode, "stdout_tail": proc.stdout[-6000:],
            "stderr_tail": proc.stderr[-6000:], "seconds": round(time.monotonic() - t0, 1),
            "traceback": "Traceback (most recent call last)" in proc.stderr, "signal": proc.returncode < 0}


def child_outcome(res: dict, output_written: bool) -> str:
    """'ok', 'crash' or 'refused' (module docstring).  M3's exit 1 with its report written is 'ok' here: the
    certificate's own FAIL."""
    if res["signal"] or res["traceback"]:
        return "crash"
    if res["returncode"] == 0:
        return "ok" if output_written else "crash"
    return "refused" if not output_written else "ok"


def record_refusal(ns, out: Path, kind: str, summary_name: str, pf: dict, res: dict, after: dict) -> None:
    """The child tool's own refusal (RA-5), recorded once; the runner then exits 2 and the folder is final."""
    out.mkdir(parents=True, exist_ok=True)
    msg = [ln for ln in res["stderr_tail"].strip().splitlines() if ln.strip()]
    rec = {"schema": f"r10m-{kind}-refused/1", "tool": os.path.realpath(__file__), "tool_sha256": C.sha256_file(__file__),
           "version": VERSION, "utc": C.utc_now(), "kind": kind.upper(), "verdict": "REFUSED",
           "message": msg[-1] if msg else "", "physics_revision": (pf["checkpoint"] or {}).get("physics_revision"),
           "checkpoint": pf["checkpoint"], "run_records": pf["run_records"], "filed": pf.get("filed"),
           "copy_seal": pf["seal"], "copy_commit_basis": pf.get("copy_commit"),
           "copy_tree_digest_after": after["tree_digest"], "child_run": res}
    C.write_json_x(out / refused_name(summary_name), rec)
    print(res["stderr_tail"][-3000:], file=sys.stderr)
    raise Refusal(f"the measuring tool refused (exit {res['returncode']}, no traceback): {rec['message'][:300]}; "
                  f"recorded in {out / refused_name(summary_name)} (final, not rerun)", 2)


def recheck_copy(ns, before: dict) -> dict:
    after = C.check_copy(ns.copy, expect_commit=before["commit"])
    if after["tree_digest"] != before["tree_digest"]:
        raise Refusal("the measuring copy changed while it was measured", 4)
    return after


def nice_py() -> list[str]:
    return ["nice", "-n", "10", C.PY, "-I", "-B"]


# ------------------------------------------------------------------------------------------------ M1 reduction
def _num(text: str) -> float:
    return float(text) if text != "" else math.nan


def read_trace_csv(path: Path, names) -> dict:
    """Per-step command, raw mean and joint position of every actuator, step and in_window, from one trace CSV."""
    import numpy as np

    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    out = {key: np.array([[_num(r[f"{n}_{key}"]) for n in names] for r in rows], dtype=np.float64).reshape(len(rows), len(names))
           for key in ("command", "raw_mean", "joint_pos")}
    out["step"] = [int(r["step"]) for r in rows]
    out["in_window"] = [int(r["in_window"]) for r in rows]
    return out


def saturation_sign(pos: int, neg: int) -> str | None:
    """THIS12 line 308: "the sign of most of its saturated steps".  '=' for an exact tie (counted once in each sign,
    v5: counts_in_signs); None for no saturated step."""
    if pos == 0 and neg == 0:
        return None
    if pos > neg:
        return "+"
    if neg > pos:
        return "-"
    return "="


def counts_in_signs(sign: str | None) -> list[str]:
    """v5, TAIL FLAG: "on an exact tie the flag counts once in each sign" ('=' -> '+' and '-')."""
    if sign == "=":
        return ["+", "-"]
    return [sign] if sign in ("+", "-") else []


def drive_backed(sign: str | None, cmd_a) -> bool | None:
    """At this checkpoint, CMD_a >= 0.60 in *sign* ('+' or '-'; v5 TRACE RAIL: DRIVE-BACKED "when, at any checkpoint
    holding one of its flags, CMD_a is >= 0.60 in the flag's sign").  A CMD_a that is not formed "reads as below 0.60
    in either sign" (v5, O-6): False.  None only for a sign that is neither (a tie is read per sign: counts_in_signs)."""
    if sign not in ("+", "-"):
        return None
    if cmd_a is None:
        return False
    return bool(cmd_a >= C.DRIVE) if sign == "+" else bool(-cmd_a >= C.DRIVE)


def reduce_m1(out: Path) -> tuple[dict, list[dict], list[str]]:
    """(summary, flag rows, consistency problems) from a trace output folder (episode_trace.json + the CSVs)."""
    import numpy as np

    with open(out / "episode_trace.json") as fh:
        tr = json.load(fh)
    problems: list[str] = []
    names = list(tr["actuator_names"])
    if tuple(names) != C.ACTUATOR_NAMES:
        problems.append(f"actuator names {names} are not the 15 of THIS12 lines 234-238")
    if tr.get("traced_actuators") != names:
        problems.append("not every actuator traced")
    if tr.get("deterministic") is not True or tr.get("stochastic_seed") is not None or tr.get("holds"):
        problems.append("not the deterministic judge rollout (stochastic or hold)")
    if tr.get("checkpoint_plant_validated") is not True:
        problems.append("checkpoint_plant_validated is not true")
    if int(tr["settle_steps"]) != C.SETTLE or int(tr["horizon"]) != C.HORIZON:
        problems.append(f"settle {tr['settle_steps']} / horizon {tr['horizon']} are not {C.SETTLE} / {C.HORIZON}")
    if float(tr["saturation_abs"]) != C.SAT_ABS or float(tr["near_saturation_abs"]) != C.NEAR_ABS:
        problems.append("saturation thresholds are not 0.99 / 0.9")
    want_seeds = list(range(C.M1_SEED, C.M1_SEED + C.M1_EPISODES))
    if list(tr["seeds"]) != want_seeds:
        problems.append(f"seeds {tr['seeds']} are not 3042-3061")
    episodes, flags, ties = [], [], []
    any_over, no_window = [], []
    max_abs_cmd = 0.0
    hip_present = {i: True for i in C.HIP_ROLL_ACTUATORS}
    for e in tr["episodes"]:
        seed, L = int(e["seed"]), int(e["length"])
        ep = read_trace_csv(out / f"episode_trace_seed{seed}.csv", names)
        if ep["command"].shape[0] != L or ep["step"] != list(range(L)):
            problems.append(f"seed {seed}: the CSV has {ep['command'].shape[0]} rows, the JSON length {L}")
            continue
        if ep["in_window"] != [int(s >= C.SETTLE) for s in range(L)]:
            problems.append(f"seed {seed}: in_window differs from steps >= {C.SETTLE}")
        if L:
            max_abs_cmd = max(max_abs_cmd, float(np.nanmax(np.abs(ep["command"]))))
        for i in C.HIP_ROLL_ACTUATORS:
            if L == 0 or not np.isfinite(ep["joint_pos"][:, i]).all():
                hip_present[i] = False
        w = slice(C.SETTLE, L)
        cmd = ep["command"][w]
        n = cmd.shape[0]
        full = L >= C.HORIZON
        row = {"seed": seed, "length": L, "full_horizon": full, "return": float(e["return"]),
               "terminated": bool(e["terminated"]), "truncated": bool(e["truncated"]),
               "termination_reason": e.get("termination_reason"), "window_steps": n, "actuators": {}}
        if n == 0:
            no_window.append(seed)
        over_here = []
        for i, name in enumerate(names):
            js = e["actuators"][name]
            if n:
                share99 = float((np.abs(cmd[:, i]) >= C.SAT_ABS).mean())
                pos = int((cmd[:, i] >= C.SAT_ABS).sum())
                neg = int((cmd[:, i] <= -C.SAT_ABS).sum())
                if js["saturated_fraction"] is None or share99 != float(js["saturated_fraction"]):
                    problems.append(f"seed {seed} {name}: share {share99} != JSON {js['saturated_fraction']}")
                cw = float(np.mean(cmd[:, i]))
                if js["command_window_mean"] is None or abs(cw - float(js["command_window_mean"])) > 1e-12:
                    problems.append(f"seed {seed} {name}: window command {cw} != JSON {js['command_window_mean']}")
            else:
                share99, pos, neg = None, 0, 0
            over = share99 is not None and share99 > C.BAR
            sign = saturation_sign(pos, neg)
            row["actuators"][name] = {
                "index": i, "cmd_window_mean": js["command_window_mean"], "raw_window_mean": js["raw_mean_window_mean"],
                "joint_pos_window_mean": js["joint_pos_window_mean"], "share_ge_0p99": share99,
                "share_ge_0p9": js["near_saturated_fraction"], "share_raw_gt_1": js["raw_mean_beyond_bound_fraction"],
                "first_saturated_step": js["first_saturated_step"], "steps_ge_p0p99": pos, "steps_le_m0p99": neg,
                "sign": sign, "over_bar": over}
            if over:
                over_here.append(i)
                if sign == "=":
                    ties.append({"seed": seed, "actuator": i, "name": name, "steps_each": pos})
                if i in C.TAIL_ACTUATORS:
                    flags.append({"seed": seed, "actuator": i, "name": name, "sign": sign, "share_ge_0p99": share99,
                                  "length": L, "kind": "TAIL FLAG" if full else "FALL FLAG"})
        if over_here:
            any_over.append({"seed": seed, "actuators": over_here})
        episodes.append(row)
    # CMD_a (v5, O-6): the mean, over those of the 20 episodes that last beyond step 200 (a window 200..L-1 that is not
    # empty), of each one's window-mean command; the others are listed (episodes_without_window); with none, not formed
    cmd_a, cmd_all, cmd_n = {}, {}, {}
    for i, name in enumerate(names):
        vals = [ep["actuators"][name]["cmd_window_mean"] for ep in episodes if ep["window_steps"] > 0]
        formed = len(vals) > 0 and all(v is not None for v in vals)
        cmd_all[str(i)] = float(np.mean(vals)) if formed else None
        if i in C.TAIL_ACTUATORS:
            cmd_a[str(i)] = cmd_all[str(i)]
            cmd_n[str(i)] = len(vals)
    for f in flags:
        f["cmd_a_at_checkpoint"] = cmd_a[str(f["actuator"])]
        f["counts_in_signs"] = counts_in_signs(f["sign"])
        if f["kind"] == "TAIL FLAG":
            f["drive_backed_by_sign"] = {sg: drive_backed(sg, cmd_a[str(f["actuator"])]) for sg in f["counts_in_signs"]}
            f["drive_backed"] = any(f["drive_backed_by_sign"].values())
        else:
            f["drive_backed_by_sign"], f["drive_backed"] = None, None
    by_sign: dict = {}
    for f in flags:
        if f["kind"] == "TAIL FLAG":
            for sg in f["counts_in_signs"]:                  # v5, O-7: a tie counts once in each sign
                key = f"{f['actuator']}{sg}"
                by_sign[key] = by_sign.get(key, 0) + 1
    hip = {}
    for i in C.HIP_ROLL_ACTUATORS:
        vals = [ep["actuators"][names[i]]["joint_pos_window_mean"] for ep in episodes]
        hip[str(i)] = {"name": names[i], "present_every_step": hip_present[i],
                       "window_mean_by_episode": vals,
                       "mean_over_episodes": float(np.mean(vals)) if vals and all(v is not None for v in vals) else None}
    summary = {
        "n_episodes": len(episodes), "seeds": [ep["seed"] for ep in episodes],
        "full_horizon_episodes": sum(1 for ep in episodes if ep["full_horizon"]),
        "episodes_without_window": no_window,
        "cmd_a": cmd_a, "cmd_a_episodes": cmd_n, "cmd_all_actuators": cmd_all,
        "tail_flags": [f for f in flags if f["kind"] == "TAIL FLAG"],
        "fall_flags": [f for f in flags if f["kind"] == "FALL FLAG"],
        "tail_flags_full_horizon_by_actuator_sign": by_sign,
        "sign_ties_over_bar": ties,
        "any_actuator_over_bar_episodes": any_over,
        "max_abs_command_any_step": max_abs_cmd,
        "hip_roll": hip,
        "trace_json_sha256": C.sha256_file(out / "episode_trace.json"),
        "episodes": episodes,
    }
    return summary, flags, problems


def gzip_csvs(out: Path) -> dict:
    csvs = sorted(out.glob("episode_trace_seed*.csv"))
    sums = {p.name: C.sha256_file(p) for p in csvs}
    with open(out / "csv_sha256.txt", "x") as fh:
        for n, s in sums.items():
            fh.write(f"{s}  {n}\n")
    for p in csvs:
        gz = p.with_name(p.name + ".gz")
        with open(p, "rb") as src, open(gz, "xb") as raw:
            with gzip.GzipFile(filename=p.name, mode="wb", fileobj=raw, mtime=0, compresslevel=6) as dst:
                shutil.copyfileobj(src, dst)
        with gzip.open(gz, "rb") as fh:
            if hashlib.sha256(fh.read()).hexdigest() != sums[p.name]:
                raise Refusal(f"{gz} does not decompress to the recorded sha256", 4)
        p.unlink()
    return sums


FLAG_COLUMNS = ("seed", "actuator", "name", "kind", "sign", "counts_in_signs", "share_ge_0p99", "length",
                "cmd_a_at_checkpoint", "drive_backed", "drive_backed_by_sign")


def cmd_m1(ns) -> int:
    pf = preflight(ns, "m1", checkpoint=True, pins=C.M1_PINS, summary_name="m1_summary.json")
    out = pf["out"]
    cmd = [*nice_py(), str(Path(ns.copy) / C.M1_SCRIPT_REL), *SPECIES, "--model", os.path.realpath(ns.zip),
           "--vecnorm", os.path.realpath(ns.pkl), "--seed", str(C.M1_SEED), "--episodes", str(C.M1_EPISODES),
           "--out-dir", str(out)]
    res = run_child(cmd, out.parent)
    after = recheck_copy(ns, pf["seal"])
    outcome = child_outcome(res, (out / "episode_trace.json").is_file())
    if outcome == "refused":
        record_refusal(ns, out, "m1", "m1_summary.json", pf, res, after)
    if outcome != "ok" or res["returncode"] != 0:
        out.mkdir(parents=True, exist_ok=True)
        C.write_json_x(out / "m1_run_failed.json", res)
        print(res["stderr_tail"][-3000:], file=sys.stderr)
        raise Refusal(f"the trace tool crashed (exit {res['returncode']}); {out} is left as a crash", 3)
    if C.sha256_file(ns.zip) != pf["checkpoint"]["sha256"] or C.sha256_file(ns.pkl) != pf["checkpoint"]["vecnorm_sha256"]:
        raise Refusal("the checkpoint pair changed while it was measured", 4)
    summary, flags, problems = reduce_m1(out)
    if problems:
        (out / "problems.txt").write_text("\n".join(problems) + "\n")
        raise Refusal("trace consistency: " + "; ".join(problems[:5]), 4)
    with open(out / "flags.csv", "x", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FLAG_COLUMNS, lineterminator="\n", extrasaction="ignore")
        w.writeheader()
        for f in flags:
            w.writerow({k: ("" if f.get(k) is None else (";".join(f[k]) if isinstance(f[k], list) else
                                                         ";".join(f"{a}:{b}" for a, b in f[k].items())
                                                         if isinstance(f[k], dict) else f[k])) for k in FLAG_COLUMNS})
    csv_sums = gzip_csvs(out)
    record = {"schema": "r10m-m1/1", "tool": os.path.realpath(__file__), "tool_sha256": C.sha256_file(__file__),
              "version": VERSION, "utc": C.utc_now(), "kind": "M1 TRACE",
              "physics_revision": pf["checkpoint"]["physics_revision"], "checkpoint": pf["checkpoint"],
              "run_records": pf["run_records"], "filed": pf["filed"], "copy_seal": pf["seal"],
              "copy_commit_basis": pf.get("copy_commit"),
              "copy_tree_digest_after": after["tree_digest"], "pinned_files": pf["pins"], "trace_run": res,
              "csv_sha256": csv_sums, "stack": C.stack_versions(), **summary}
    C.write_json_x(out / "m1_summary.json", record)
    print(json.dumps({"m1": str(out), "physics_revision": record["physics_revision"],
                      "tail_flags": len(record["tail_flags"]), "fall_flags": len(record["fall_flags"]),
                      "cmd_a": record["cmd_a"], "seconds": res["seconds"]}), flush=True)
    return 0


# ------------------------------------------------------------------------------------------------ M2
def statue_panel_ok(path: str, seal: dict) -> dict:
    """The --statue-json of a policy panel: an M2 statue panel made by this runner on the same sealed copy."""
    p = Path(path)
    summ = p.parent / "m2_summary.json"
    if p.name != "panel.json" or not summ.is_file():
        raise Refusal(f"--statue-json {path} is not an M2 statue panel.json (no m2_summary.json beside it)")
    s = json.loads(summ.read_text())
    if s.get("mode") != "statue":
        raise Refusal(f"{summ}: mode {s.get('mode')!r}, not a statue panel")
    cs = s.get("copy_seal") or {}
    if cs.get("commit") != seal["commit"] or cs.get("tree_digest") != seal["tree_digest"]:
        raise Refusal(f"{summ}: the statue panel was made on another measuring copy ({cs.get('commit')})")
    if C.sha256_file(p) != s.get("panel_json_sha256"):
        raise Refusal(f"{p} changed since its M2 run")
    return {"statue_json": os.path.realpath(p), "sha256": s.get("panel_json_sha256")}


POSTURE_FIELDS = ("tail3_cmd_mean", "tail1_cmd_mean", "tail2_cmd_mean", "q1_deg", "q2_deg", "q3_deg", "chord_base_deg",
                  "chord_dist_deg", "tip_z_m", "com_fa_mm", "com_lat_mm", "pitch_deg", "roll_deg", "pelvis_z_m",
                  "head_tip_z_m", "neck_pitch_deg", "head_pitch_deg", "others_max_sat")


def cmd_m2(ns) -> int:
    if ns.statue == bool(ns.zip):
        raise Refusal("m2: pass --statue, or --zip/--pkl/--statue-json")
    harness = TOOLS / C.M2_HARNESS
    if C.sha256_file(harness) != C.M2_HARNESS_SHA256:
        raise Refusal(f"{harness} sha256 is not the pinned {C.M2_HARNESS_SHA256}")
    pf = preflight(ns, "m2", checkpoint=not ns.statue, pins=None, summary_name="m2_summary.json")
    out = pf["out"]
    statue = None if ns.statue else statue_panel_ok(ns.statue_json, pf["seal"])
    cmd = [*nice_py(), str(harness), "--code", os.path.realpath(ns.copy), "--plant", "r9",
           "--mode", "statue" if ns.statue else "det", "--seed", str(C.M2_SEED), "--episodes", str(C.M2_EPISODES),
           "--out", str(out), "--label", ns.label or ("statue" if ns.statue else Path(ns.zip).stem)]
    if not ns.statue:
        cmd += ["--zip", os.path.realpath(ns.zip), "--pkl", os.path.realpath(ns.pkl), "--statue-json", statue["statue_json"]]
    res = run_child(cmd, out.parent)
    after = recheck_copy(ns, pf["seal"])
    panel_p = out / "panel.json"
    outcome = child_outcome(res, panel_p.is_file())
    if outcome == "refused":
        record_refusal(ns, out, "m2", "m2_summary.json", pf, res, after)
    if outcome != "ok" or res["returncode"] != 0:
        out.mkdir(parents=True, exist_ok=True)
        C.write_json_x(out / "m2_run_failed.json", res)
        print(res["stderr_tail"][-3000:], file=sys.stderr)
        raise Refusal(f"the posture harness crashed (exit {res['returncode']}); {out} is left as a crash", 3)
    panel = json.loads(panel_p.read_text())
    s = panel["summary"]
    ident = s.get("r9_identity") or {}
    problems = []
    if ident.get("physics_revision") != C.PHYSICS_REVISION or ident.get("physics_sha256") != C.R10_PHYSICS_SHA256:
        problems.append(f"the copy's plant is {ident} (the harness's 'r9' flag must mean the copy's r10 plant)")
    if (s.get("plant_check") or {}).get("validate_environment_plant") != "passed":
        problems.append(f"plant check {s.get('plant_check')}")
    if s.get("seeds") != [C.M2_SEED, C.M2_SEED + C.M2_EPISODES - 1] or len(panel["episodes"]) != C.M2_EPISODES:
        problems.append(f"seeds {s.get('seeds')} / {len(panel['episodes'])} episodes")
    if os.path.realpath(s.get("code", "")) != os.path.realpath(ns.copy):
        problems.append(f"code {s.get('code')} is not the copy")
    if problems:
        (out / "problems.txt").write_text("\n".join(problems) + "\n")
        raise Refusal("posture panel consistency: " + "; ".join(problems), 4)
    eps = panel["episodes"]
    full = [e for e in eps if e.get("reached_horizon")]
    means = {}
    for k in POSTURE_FIELDS:
        vals = [e.get(k) for e in full if e.get(k) is not None]
        means[k] = (sum(vals) / len(vals)) if vals else None
    v = s["verdict"]
    record = {"schema": "r10m-m2/1", "tool": os.path.realpath(__file__), "tool_sha256": C.sha256_file(__file__),
              "version": VERSION, "utc": C.utc_now(), "kind": "M2 POSTURE PANEL", "mode": "statue" if ns.statue else "det",
              "physics_revision": (pf["checkpoint"] or {}).get("physics_revision", ident.get("physics_revision")),
              "copy_plant": ident, "checkpoint": pf["checkpoint"], "run_records": pf["run_records"],
              "filed": pf["filed"], "statue_panel": statue, "copy_seal": pf["seal"],
              "copy_commit_basis": pf.get("copy_commit"), "copy_tree_digest_after": after["tree_digest"],
              "harness": {"path": str(harness), "sha256": C.M2_HARNESS_SHA256}, "harness_run": res,
              "panel_json_sha256": C.sha256_file(panel_p), "series_npz_sha256": C.sha256_file(out / "series.npz"),
              "n_episodes": len(eps), "n_full_horizon": len(full), "n_clean": v.get("n_clean"), "passed": v.get("passed"),
              "mean_reward": v.get("mean_reward"), "statue_mean_reward": v.get("statue_mean_reward"),
              "fall_dirs": {d: sum(1 for e in eps if e.get("fall_dir") == d) for d in sorted({e.get("fall_dir") for e in eps})},
              "posture_means_full_horizon": means, "stack": C.stack_versions()}
    C.write_json_x(out / "m2_summary.json", record)
    print(json.dumps({"m2": str(out), "mode": record["mode"], "physics_revision": record["physics_revision"],
                      "n_clean": record["n_clean"], "n_full": record["n_full_horizon"],
                      "mean_reward": record["mean_reward"], "seconds": res["seconds"]}), flush=True)
    return 0


# ------------------------------------------------------------------------------------------------ M3
def cmd_m3(ns) -> int:
    if ns.zero_action == bool(ns.zip):
        raise Refusal("m3: pass --zero-action, or --zip/--pkl")
    pf = preflight(ns, "m3", checkpoint=not ns.zero_action, pins=C.M3_PINS, summary_name="m3_summary.json")
    out = pf["out"]
    cmd = [*nice_py(), str(Path(ns.copy) / C.M3_SCRIPT_REL), *SPECIES, "--seed", str(C.M3_SEED), "--out-dir", str(out)]
    cmd += ["--zero-action"] if ns.zero_action else ["--model", os.path.realpath(ns.zip), "--vecnorm", os.path.realpath(ns.pkl)]
    out.mkdir(parents=True)
    res = run_child(cmd, out.parent)
    after = recheck_copy(ns, pf["seal"])
    rep_p = out / "stance_heading_report.json"
    outcome = child_outcome(res, rep_p.is_file())
    if outcome == "refused":
        record_refusal(ns, out, "m3", "m3_summary.json", pf, res, after)
    if outcome != "ok" or res["returncode"] not in (0, 1):
        C.write_json_x(out / "m3_run_failed.json", res)
        print(res["stderr_tail"][-3000:], file=sys.stderr)
        raise Refusal(f"the certificate tool crashed (exit {res['returncode']}); {out} is left as a crash", 3)
    rep = json.loads(rep_p.read_text())
    plant = rep.get("plant_identity") or {}
    problems = []
    if plant.get("physics_revision") != C.PHYSICS_REVISION or plant.get("physics_sha256") != C.R10_PHYSICS_SHA256:
        problems.append(f"the certificate's plant is {plant.get('physics_revision')} {plant.get('physics_sha256')}")
    if rep.get("scored_gate_kind") != "stance_quality_heading/v1":
        problems.append(f"scored as {rep.get('scored_gate_kind')}")
    if [float(h["yaw_deg"]) for h in rep.get("headings", [])] != list(C.M3_HEADINGS):
        problems.append(f"headings {[h.get('yaw_deg') for h in rep.get('headings', [])]}")
    if list(rep.get("seeds") or []) != list(range(C.M3_SEED, C.M3_SEED + C.M3_EPISODES)):
        problems.append(f"seeds {rep.get('seeds')}")
    if (res["returncode"] == 0) != bool(rep.get("passed")):
        problems.append(f"exit {res['returncode']} disagrees with the report's passed {rep.get('passed')!r}")
    if not ns.zero_action:
        h = rep.get("handoff") or {}
        if h.get("checkpoint_sha256") not in (pf["checkpoint"]["sha256"], "sha256:" + pf["checkpoint"]["sha256"]):
            problems.append(f"certified checkpoint sha256 {h.get('checkpoint_sha256')}")
    if problems:
        (out / "problems.txt").write_text("\n".join(problems) + "\n")
        raise Refusal("certificate consistency: " + "; ".join(problems), 4)
    headings = [{"yaw_deg": h["yaw_deg"], "n_clean": h["metrics"]["n_clean"], "clean_lcb": h["metrics"]["clean_lcb"],
                 "full_horizon_fraction": h["metrics"]["full_horizon_fraction"], "reward_mean": h["metrics"]["reward_mean"],
                 "n_episodes": len(h.get("episode_evidence") or [])} for h in rep["headings"]]
    record = {"schema": "r10m-m3/1", "tool": os.path.realpath(__file__), "tool_sha256": C.sha256_file(__file__),
              "version": VERSION, "utc": C.utc_now(), "kind": "M3 CERTIFICATE",
              "label": ("GATE CERTIFICATE (final checkpoint, measured on a screen task; not certification)"
                        if (pf["checkpoint"] or {}).get("num_timesteps") == C.FINAL_TIMESTEPS else
                        "CERTIFICATE (measured on a screen task; not certification; not the final checkpoint: report only)"),
              "zero_action": bool(ns.zero_action),
              "physics_revision": (pf["checkpoint"] or {}).get("physics_revision", plant.get("physics_revision")),
              "certificate_plant": {k: plant.get(k) for k in ("physics_revision", "physics_sha256")},
              "checkpoint": pf["checkpoint"], "run_records": pf["run_records"], "filed": pf["filed"],
              "copy_seal": pf["seal"], "copy_commit_basis": pf.get("copy_commit"),
              "copy_tree_digest_after": after["tree_digest"], "pinned_files": pf["pins"],
              "certificate_run": res,
              "report_sha256": C.sha256_file(rep_p), "verdict": "PASS" if rep.get("passed") else "FAIL",
              "passed": bool(rep.get("passed")), "failures": rep.get("failures"), "task_sha256": rep.get("task_sha256"),
              "gate_sha256": rep.get("gate_sha256"), "headings": headings, "stack": C.stack_versions()}
    C.write_json_x(out / "m3_summary.json", record)
    print(json.dumps({"m3": str(out), "verdict": record["verdict"], "physics_revision": record["physics_revision"],
                      "n_clean_by_heading": {h["yaw_deg"]: h["n_clean"] for h in headings}, "seconds": res["seconds"]}),
          flush=True)
    return 0


# ------------------------------------------------------------------------------------------------ CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("version")
    b = sub.add_parser("build-copy")
    b.add_argument("--dest", required=True)
    b.add_argument("--commit", default=None, help="a commit named explicitly (default: the approved branch HEAD)")
    c = sub.add_parser("check-copy")
    c.add_argument("--copy", required=True)
    for q in (b, c):
        q.add_argument("--header-parent", action="store_true",
                       help="the header's parent cf9402bc (the text's pin): validation before the header commit")
        q.add_argument("--approved", default=None, help="the approval record (default <ROUND>/APPROVED)")
    for kind in ("m1", "m2", "m3"):
        p = sub.add_parser(kind)
        p.add_argument("--copy", required=True)
        p.add_argument("--header-parent", action="store_true",
                       help="validation inputs on the header's parent cf9402bc (refused with --run-id)")
        p.add_argument("--approved", default=None, help="the approval record (default <ROUND>/APPROVED)")
        p.add_argument("--zip")
        p.add_argument("--pkl")
        p.add_argument("--stage-config", help="the run's 01_stance/stage_config.json (commit binding)")
        p.add_argument("--no-run-records", action="store_true", help="validation input: no run records (recorded)")
        p.add_argument("--out")
        p.add_argument("--run-id")
        p.add_argument("--label")
        if kind == "m2":
            p.add_argument("--statue", action="store_true")
            p.add_argument("--statue-json")
        if kind == "m3":
            p.add_argument("--zero-action", action="store_true")
    ns = ap.parse_args(argv)
    try:
        if ns.cmd == "version":
            print(VERSION)
            return 0
        if ns.cmd == "build-copy":
            if ns.commit and (ns.header_parent or ns.approved):
                raise Refusal("--commit excludes --header-parent and --approved")
            commit = ns.commit or expected_commit(ns)["commit"]
            seal = C.build_copy(C.BR, commit, ns.dest, tool=VERSION)
            print(json.dumps(seal, indent=1))
            return 0
        if ns.cmd == "check-copy":
            print(json.dumps(C.check_copy(ns.copy, expect_commit=expected_commit(ns)["commit"]), indent=1))
            return 0
        if ns.cmd in ("m1", "m2", "m3") and ns.stage_config and ns.no_run_records:
            raise Refusal("--stage-config and --no-run-records exclude each other")
        if ns.cmd == "m1":
            return cmd_m1(ns)
        if ns.cmd == "m2":
            return cmd_m2(ns)
        return cmd_m3(ns)
    except Refusal as exc:
        print(f"[measure_r10round] REFUSED: {exc.message}", file=sys.stderr, flush=True)
        return int(exc.code)


if __name__ == "__main__":
    raise SystemExit(main())
