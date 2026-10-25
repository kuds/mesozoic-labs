"""Tail_3 bound screen: the scorer (criteria_tailscreen_final DECISION, WHEN THE DECISION IS READ).  Report only.

    PY -I -B SCREEN/tools/score_tailscreen.py [--panels SCREEN/panels.csv] [--runs SCREEN/runs] [--out SCREEN/scores]

Reads panels.csv (one row per (name, label), written by measure_tailscreen.py), SCREEN/LOST.txt (maintainer-declared
LOST replicates, one per line: '<name> <RUNLOG line>') and, for the PROVISIONAL checks, the pins in screen_common,
SCREEN/APPROVED, SCREEN/criteria.txt, RUNLOG, screen_common.py's sha256, each runs/<name>/run_meta.json, the
checkpoint files now in runs/<name>/ (each panel row's zip_sha256 must be the sha256 of the checkpoint it names, and
the final's must equal run_meta.final_sha256, so a crashed attempt's row is never read), the scratch clone's diff
file set, tests_out/build_tier.json and gate_pass.json (every GATE_REQUIRED test recorded "passed"),
calib/coef.json, measure_copy's seals and validate/v1.json.
Writes OUT/score.json and OUT/score.txt.

Exit codes: 0 final; 10 incomplete (no line read); 11 provisional (read, printed, not acted on); 2 input error.

Logic (exactly the criteria's):
  replicate (R1 = 5111808, R2 = final): RAIL if tailup_R1 + tailup_R2 >= 1, else PARKED if (cmd_R1 + cmd_R2)/2 >= 0.60,
    else CURED.  A LOST bound replicate is RAIL for its own bound; a LOST CONTROL replicate is CURED at s42 and
    'no rail' at s80 (onset note only); a LOST replicate contributes 0 OTHER flags.
  state reading per bound B (first line that applies; every applying line reported):
    s42: RAIL (>= 1 B RAIL), PARKED (>= 2 B PARKED), OTHER (B's OTHER flags at R1+R2 > CONTROL's),
         NOT SEPARATED (>= 2 CONTROL CURED), PASS;   s80: RAIL, PARKED, OTHER, PASS.
  onset note: INFORMATIVE if >= 2 of 3 s80 CONTROL replicates had >= 1 tail-up flag at any of their 9 labels; the
    note states how many of the 9 labels each s80 CONTROL replicate has, and a missing label of a non-LOST s80
    CONTROL replicate marks the score PROVISIONAL.
  bound verdict: either state RAIL / PARKED / OTHER, or s42 NOT SEPARATED -> NOT ADOPTED (first reason); else PASSED.
  outcome: A both PASSED; B exactly one; C neither (routing per reason, named combinations).
"""
# (no "from __future__ import annotations": the dataclass below must work when loaded by file path)

import os
import sys

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True

import argparse  # noqa: E402
import csv  # noqa: E402
import hashlib  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402
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

EXIT_FINAL, EXIT_INCOMPLETE, EXIT_PROVISIONAL, EXIT_INPUT = 0, 10, 11, 2
STATES = ("s42", "s80")
ARMS = ("control", "b90", "b60")
BOUNDS = {"BOUND90": "b90", "BOUND60": "b60"}
R1, R2 = str(C.R1), str(C.R2)
LINE = 0.60
LINES = {"s42": ("RAIL", "PARKED", "OTHER", "NOT SEPARATED", "PASS"), "s80": ("RAIL", "PARKED", "OTHER", "PASS")}
VERDICT_ORDER = ("RAIL", "PARKED", "OTHER", "NOT SEPARATED")
TOOL_FILES = tuple(C.TOOL_FILES)
REQUIRED_COLUMNS = ("name", "label", "cmd", "tailup_flags", "other_flags", "measure_tree_digest", "seeds", "stack",
                    "n_episodes")
ROUTING = {
    "RAIL": "next candidate chosen by the maintainer: the deterministic-state variant (BRIEF 2A), a stronger dose "
            "(DEC1 D3(c)), or the posture decision (B / C)",
    "PARKED": "the posture decision (ruling (3))",
    "OTHER": "the harm / substitution review of the flagged episodes",
    "NOT SEPARATED": "more s42 replicates, or stop",
}
REASON_TEXT = {
    "RAIL": "it did not keep the rail away in deterministic play",
    "PARKED": "rail cleared but tail parked (ruling (3)); routed to the posture decision",
    "OTHER": "more other-actuator flags or short episodes under the bound; flagged episodes go to the harm / "
             "substitution review",
    "NOT SEPARATED": "the controls cured themselves; the bound is not credited",
}
NAMED_RAIL_PARKED = "the lower bound cleared the rail but not the posture"
NAMED_BOTH_PARKED = "neither bound moved deterministic play below the line"
ALWAYS = ("Nothing trains on the outcome without the maintainer's approval; the outcome changes no screen, verdict, "
          "parent, criterion or stage config.")


class InputError(Exception):
    pass


def names() -> list[str]:
    return [f"{s}_{a}_r{r}" for s in STATES for a in ARMS for r in C.REP_SEEDS]


def parse_name(name: str) -> tuple[str, str, int]:
    s, a, r = name.split("_")
    return s, a, int(r[1:])


# ------------------------------------------------------------------------------------------------- pure logic

def replicate_class(tailup_r1: int, tailup_r2: int, cmd_r1: float, cmd_r2: float) -> str:
    if tailup_r1 + tailup_r2 >= 1:
        return "RAIL"
    return "PARKED" if (cmd_r1 + cmd_r2) / 2.0 >= LINE else "CURED"


def state_lines(state: str, bound_classes, bound_other: int, control_classes, control_other: int) -> list[str]:
    """Every line of the state reading that applies, in the criteria's order (PASS only when none does)."""
    lines = []
    if sum(c == "RAIL" for c in bound_classes) >= 1:
        lines.append("RAIL")
    if sum(c == "PARKED" for c in bound_classes) >= 2:
        lines.append("PARKED")
    if bound_other > control_other:
        lines.append("OTHER")
    if state == "s42" and sum(c == "CURED" for c in control_classes) >= 2:
        lines.append("NOT SEPARATED")
    if not lines:
        lines.append("PASS")
    return lines


def line_index(state: str, reading: str) -> int:
    return LINES[state].index(reading)


def bound_verdict(readings: dict) -> tuple[str, str | None]:
    """readings: {state: reading}.  ('PASSED', None) or ('NOT ADOPTED', reason)."""
    for reason in VERDICT_ORDER:
        if any(r == reason for r in readings.values()):
            return "NOT ADOPTED", reason
    return "PASSED", None


def outcome(verdicts: dict) -> dict:
    v90, v60 = verdicts["BOUND90"], verdicts["BOUND60"]
    passed = [b for b in ("BOUND90", "BOUND60") if verdicts[b]["verdict"] == "PASSED"]
    to = []
    if len(passed) == 2:
        line = "A"
        text = "A. BOTH PASSED."
        to.append("Both bounds as candidates for a from-scratch GPU round proposal (BRIEF section 4), each needing "
                  "its own fixed criteria. The screen does not rank them (K = 3, one shared CONTROL); report-only "
                  "comparisons (CMD, q3, returns, OTHER counts per arm) go beside.")
        to.append("BOUND60's PASS is also posture evidence for the posture decision.")
    elif len(passed) == 1:
        line = "B"
        p = passed[0]
        o = "BOUND60" if p == "BOUND90" else "BOUND90"
        reason = verdicts[o]["reason"]
        text = f"B. ONE PASSED: {p} PASSED, {o} NOT ADOPTED ({reason})."
        to.append(f"{p} as the candidate for a from-scratch GPU round proposal with its own fixed criteria.")
        to.append(f"{o} NOT ADOPTED ({reason}): {ROUTING[reason]}.")
        if p == "BOUND60":
            to.append("BOUND60 passed: the result is also input to the posture decision (B with a tail-chain "
                      "revision, C), since a 0.60 bound is a posture intervention.")
    else:
        line = "C"
        text = (f"C. NO BOUND ADOPTED: BOUND90 NOT ADOPTED ({v90['reason']}), "
                f"BOUND60 NOT ADOPTED ({v60['reason']}).")
        for b in ("BOUND90", "BOUND60"):
            to.append(f"{b} ({verdicts[b]['reason']}): {ROUTING[verdicts[b]['reason']]}.")
        if v90["reason"] == "RAIL" and v60["reason"] == "PARKED":
            to.append(f"Named combination: {NAMED_RAIL_PARKED}.")
        if v90["reason"] == "PARKED" and v60["reason"] == "PARKED":
            to.append(f"Named combination: {NAMED_BOTH_PARKED}.")
    to.append(ALWAYS)
    return {"line": line, "text": text, "to_maintainer": to}


# ------------------------------------------------------------------------------------------------- rows

def _num(row: dict, key: str, cast=float):
    try:
        value = cast(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        hint = " (cmd is empty when no episode reached the window: the maintainer decides)" if row.get(key) == "" else ""
        raise InputError(f"panels.csv {row.get('name')} {row.get('label')}: bad {key}={row.get(key)!r}{hint}") from exc
    if cast is float and value != value:
        raise InputError(f"panels.csv {row.get('name')} {row.get('label')}: {key} is NaN")
    return value


def read_panels(path: Path) -> list[dict] | None:
    if not path.is_file():
        return None
    with open(path, newline="") as fh:
        rdr = csv.DictReader(fh)
        missing = [c for c in REQUIRED_COLUMNS if c not in (rdr.fieldnames or [])]
        if missing:
            raise InputError(f"{path}: missing columns {missing}")
        return list(rdr)


def index_rows(rows: list[dict]) -> dict:
    out = {}
    for r in rows:
        key = (r["name"], r["label"])
        if key in out:
            raise InputError(f"duplicate panels.csv row for {key}")
        if r["name"] in names():  # every screen row is checked, not only the rows a line reads
            if r["label"] not in C.LABELS:
                raise InputError(f"panels.csv {r['name']}: unknown label {r['label']!r}")
            _num(r, "cmd")
            _num(r, "tailup_flags", int)
            _num(r, "other_flags", int)
        out[key] = r
    return out


def read_lost(path: Path) -> dict:
    lost = {}
    if not path.is_file():
        return lost
    for ln in path.read_text().splitlines():
        if not ln.strip() or ln.lstrip().startswith("#"):
            continue
        parts = ln.split(None, 1)
        if parts[0] not in names():
            raise InputError(f"LOST.txt names {parts[0]!r}, not one of the 18 replicates")
        if len(parts) < 2:
            raise InputError(f"LOST.txt line for {parts[0]} carries no RUNLOG line")
        lost[parts[0]] = parts[1]
    return lost


def replicates(idx: dict, lost: dict) -> dict:
    reps = {}
    for name in names():
        state, arm, seed = parse_name(name)
        rec = {"arm": arm, "state": state, "seed": seed, "class": None, "cmd_R1": None, "cmd_R2": None,
               "cmd_mean": None, "tailup_R1R2": None, "other_R1R2": 0, "onset_any_tailup": False,
               "labels_measured": sum((name, lb) in idx for lb in C.LABELS), "missing": None}
        if name in lost:
            rec["missing"] = "LOST"
            if arm != "control":
                rec["class"] = "RAIL"  # never helps its own bound
            else:
                rec["class"] = "CURED" if state == "s42" else "NO RAIL (LOST)"
            reps[name] = rec
            continue
        rec["onset_any_tailup"] = any(
            _num(idx[(name, lb)], "tailup_flags", int) >= 1 for lb in C.LABELS if (name, lb) in idx)
        a, b = idx.get((name, R1)), idx.get((name, R2))
        if a is None or b is None:
            rec["missing"] = "INCOMPLETE: no " + " and ".join(
                lb for lb, r in ((R1, a), (R2, b)) if r is None) + " row"
            reps[name] = rec
            continue
        t1, t2 = _num(a, "tailup_flags", int), _num(b, "tailup_flags", int)
        c1, c2 = _num(a, "cmd"), _num(b, "cmd")
        rec.update({"cmd_R1": c1, "cmd_R2": c2, "cmd_mean": (c1 + c2) / 2.0, "tailup_R1R2": t1 + t2,
                    "other_R1R2": _num(a, "other_flags", int) + _num(b, "other_flags", int),
                    "class": replicate_class(t1, t2, c1, c2)})
        reps[name] = rec
    return reps


def readings(reps: dict) -> tuple[dict, str, dict]:
    states = {}
    for state in STATES:
        ctrl = [reps[f"{state}_control_r{r}"] for r in C.REP_SEEDS]
        cc = [x["class"] for x in ctrl]
        co = sum(x["other_R1R2"] for x in ctrl)
        states[state] = {}
        for bname, arm in BOUNDS.items():
            br = [reps[f"{state}_{arm}_r{r}"] for r in C.REP_SEEDS]
            lines = state_lines(state, [x["class"] for x in br], sum(x["other_R1R2"] for x in br), cc, co)
            states[state][bname] = {"reading": lines[0], "lines_applying": lines,
                                    "other_bound": sum(x["other_R1R2"] for x in br), "other_control": co}
    n_onset = sum(reps[f"s80_control_r{r}"]["onset_any_tailup"] for r in C.REP_SEEDS)
    counts = ", ".join(
        f"r{r} {'LOST' if reps[f's80_control_r{r}']['missing'] == 'LOST' else reps[f's80_control_r{r}']['labels_measured']}"
        for r in C.REP_SEEDS)
    short = [f"s80_control_r{r}" for r in C.REP_SEEDS if reps[f"s80_control_r{r}"]["missing"] != "LOST"
             and reps[f"s80_control_r{r}"]["labels_measured"] < len(C.LABELS)]
    onset = ("INFORMATIVE" if n_onset >= 2 else "onset not tested") + f" ({n_onset} of 3 s80 CONTROL replicates " \
        f"had >= 1 tail-up flag at any measured checkpoint; labels measured of {len(C.LABELS)}: {counts})"
    if short:
        onset += f"; INCOMPLETE: {', '.join(short)} lack labels (PROVISIONAL)"
    verdicts = {}
    for bname in BOUNDS:
        v, reason = bound_verdict({s: states[s][bname]["reading"] for s in STATES})
        verdicts[bname] = {"verdict": v, "reason": reason}
    return states, onset, verdicts


def report_only(idx: dict, reps: dict) -> dict:
    dose = {}
    for state in STATES:
        means = {}
        for arm in ARMS:
            vals = [reps[f"{state}_{arm}_r{r}"]["cmd_mean"] for r in C.REP_SEEDS]
            vals = [v for v in vals if v is not None]
            means[arm] = {"cmd_R1R2_arm_mean": (sum(vals) / len(vals)) if vals else None, "n": len(vals),
                          "classes": [reps[f"{state}_{arm}_r{r}"]["class"] for r in C.REP_SEEDS]}
        m = [means[a]["cmd_R1R2_arm_mean"] for a in ("b60", "b90", "control")]
        ordered = None if None in m else (m[0] <= m[1] <= m[2])
        dose[state] = {"arms": means, "ordered_b60_le_b90_le_control": ordered}
    half = {}
    for name in names():
        half[name] = {lb: ({"cmd": _num(idx[(name, lb)], "cmd"),
                            "tailup_flags": _num(idx[(name, lb)], "tailup_flags", int),
                            "class_single": idx[(name, lb)].get("class_single")} if (name, lb) in idx else None)
                      for lb in ("4587520", "4718592")}
    return {"dose_ordering": dose, "half_point": half}


# ------------------------------------------------------------------------------------------------- PROVISIONAL

@dataclass
class ProvInputs:
    pins: dict                       # SCRATCH_COMMIT, DURABLE_BRANCH, TOOL_SHA256 (from screen_common)
    tool_sha_actual: dict            # file name -> sha256 on disk (None if missing)
    approved: dict | None
    criteria_sha: str | None
    runlog_text: str | None
    run_metas: dict                  # name -> dict | None
    diff_files: list | None
    build_tier: dict | None
    gate_pass: dict | None
    coef: dict | None
    coef_sha: str | None
    coef_mtime_utc: str | None       # ISO Z
    sealed_tree_digest: str | None
    overlay_problems: list
    v1: dict | None
    rows: list = field(default_factory=list)
    lost: dict = field(default_factory=dict)
    screen_common_sha: str | None = None  # sha256 of tools/screen_common.py on disk
    ckpt_sha: dict = field(default_factory=dict)  # (name, label) -> sha256 of that checkpoint zip in runs/<name>


def _utc(s):
    try:
        return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        try:
            return datetime.fromisoformat(str(s).replace("Z", "+00:00"))
        except (TypeError, ValueError):
            return None


def coef_by_rule(coef: dict) -> float | None:
    try:
        return float(format(float(coef["pooled"]["G_pg_rms"]) / (2 * 0.11), ".2g"))
    except (KeyError, TypeError, ValueError):
        return None


def _tests_summary(doc: dict | None) -> tuple[int, int]:
    """(passed, failed-or-errored) from a build_tier.json in any of the shapes the build may write."""
    if not doc:
        return 0, 0
    tests = doc.get("tests", doc.get("results"))
    statuses = []
    if isinstance(tests, dict):
        statuses = [v if isinstance(v, str) else (v or {}).get("status", (v or {}).get("outcome")) for v in
                    tests.values()]
    elif isinstance(tests, list):
        statuses = [t.get("status", t.get("outcome")) if isinstance(t, dict) else None for t in tests]
    if statuses:
        passed = sum(s == "passed" for s in statuses)
        bad = sum(s not in ("passed", "skipped", "deselected", "xfailed") for s in statuses)
        return passed, bad
    return int(doc.get("passed", 0)), int(doc.get("failed", 0)) + int(doc.get("errors", 0))


def provisional_reasons(p: ProvInputs) -> list[str]:
    R = []
    pin = p.pins.get("SCRATCH_COMMIT")
    if not pin:
        R.append("pin unfilled: SCRATCH_COMMIT (screen_common)")
    if not p.pins.get("DURABLE_BRANCH"):
        R.append("pin unfilled: DURABLE_BRANCH (screen_common)")
    tool_pins = p.pins.get("TOOL_SHA256") or {}
    for t in TOOL_FILES:
        if not tool_pins.get(t):
            R.append(f"pin unfilled: TOOL_SHA256[{t}]")
        elif tool_pins[t] != p.tool_sha_actual.get(t):
            R.append(f"tool {t} sha256 {p.tool_sha_actual.get(t)} differs from its pin {tool_pins[t]}")
    a = p.approved
    if a is None:
        R.append("SCREEN/APPROVED missing or unreadable")
        a = {}
    else:
        for k in ("criteria_sha256", "scratch_commit", "COEF_90", "COEF_60", "coef_json_sha256", "durable_branch",
                  "runlog_line", "approved_utc", "screen_common_sha256"):
            if a.get(k) in (None, ""):
                R.append(f"APPROVED pin unfilled: {k}")
        if a.get("screen_common_sha256") and a["screen_common_sha256"] != p.screen_common_sha:
            R.append(f"screen_common.py sha256 {p.screen_common_sha} differs from APPROVED's "
                     f"{a['screen_common_sha256']}")
        if pin and a.get("scratch_commit") and a["scratch_commit"] != pin:
            R.append(f"APPROVED scratch_commit {a['scratch_commit']} != screen_common pin {pin}")
        if p.pins.get("DURABLE_BRANCH") and a.get("durable_branch") and a["durable_branch"] != p.pins["DURABLE_BRANCH"]:
            R.append("APPROVED durable_branch differs from screen_common DURABLE_BRANCH")
    head_pin = pin or a.get("scratch_commit")
    # criteria.txt
    if p.criteria_sha is None:
        R.append("SCREEN/criteria.txt missing")
    else:
        if a.get("criteria_sha256") != p.criteria_sha:
            R.append(f"criteria.txt sha256 {p.criteria_sha} differs from APPROVED's {a.get('criteria_sha256')}")
        if not p.runlog_text or p.criteria_sha not in p.runlog_text:
            R.append("criteria.txt sha256 is not recorded in RUNLOG")
    # run_meta.json
    coef_pin = {"control": 0.0, "b90": a.get("COEF_90"), "b60": a.get("COEF_60")}
    b_pin = {"control": 0.90, "b90": 0.90, "b60": 0.60}
    for name in names():
        if name in p.lost:
            continue
        state, arm, seed = parse_name(name)
        m = p.run_metas.get(name)
        if m is None:
            if any(r.get("name") == name for r in p.rows):
                R.append(f"{name}: panels exist but run_meta.json is missing")
            continue
        clone = m.get("clone") or {}
        if not head_pin or clone.get("head") != head_pin:
            R.append(f"{name}: clone HEAD {clone.get('head')} is not the pinned scratch commit {head_pin}")
        if clone.get("status", None) != "":
            R.append(f"{name}: clone was not clean ({clone.get('status')!r})")
        start = m.get("start") or {}
        st = C.STATES[state]
        if start.get("zip_sha256") != st["zip_sha256"] or start.get("pkl_sha256") != st["pkl_sha256"]:
            R.append(f"{name}: start sha256 differs from {state}'s")
        try:
            coef_ok = coef_pin[arm] is not None and float(m.get("coef")) == float(coef_pin[arm])
            b_ok = float(m.get("b")) == b_pin[arm]
        except (TypeError, ValueError):
            coef_ok = b_ok = False
        if m.get("arm") != arm or not coef_ok or not b_ok:
            R.append(f"{name}: (arm, coef, b) = ({m.get('arm')}, {m.get('coef')}, {m.get('b')}) is not "
                     f"({arm}, {coef_pin[arm]}, {b_pin[arm]})")
        if m.get("rep_seed") != seed:
            R.append(f"{name}: seed {m.get('rep_seed')} is not the name's {seed}")
        if m.get("stack") != C.STACK:
            R.append(f"{name}: stack {m.get('stack')} is not the pinned stack")
        if m.get("status") != "done":
            R.append(f"{name}: run_meta status {m.get('status')!r}, not 'done'")
    # scratch commit file set
    if p.diff_files is None:
        R.append("scratch commit's changed-file set could not be read")
    elif sorted(p.diff_files) != sorted(C.CHANGED_FILES):
        R.append(f"scratch commit changes {sorted(p.diff_files)}, not exactly {list(C.CHANGED_FILES)}")
    # tests, both tiers
    bt_pass, bt_bad = _tests_summary(p.build_tier)
    if p.build_tier is None:
        R.append("tests_out/build_tier.json missing (T1, T7 build tier not shown passed)")
    elif not head_pin or p.build_tier.get("head") != head_pin or bt_bad or not bt_pass:
        R.append(f"build-tier tests not all passed on the pinned commit (head {p.build_tier.get('head')}, "
                 f"passed {bt_pass}, failed {bt_bad})")
    if p.gate_pass is None:
        R.append("tests_out/gate_pass.json missing (T1, T2, T7 gate tier not shown passed)")
    else:
        gp = C.gate_problems(p.gate_pass, head_pin)
        if gp:
            R.append("gate-tier tests not all passed on the pinned commit: " + "; ".join(gp[:4]))
    # coef.json
    if p.coef is None:
        R.append("calib/coef.json missing")
    else:
        rule = coef_by_rule(p.coef)
        for k in ("COEF_90", "COEF_60"):
            if rule is None or p.coef.get(k) != rule:
                R.append(f"coef.json {k} {p.coef.get(k)} does not follow the rule ({rule})")
            if a.get(k) is not None and p.coef.get(k) != a.get(k):
                R.append(f"coef.json {k} differs from APPROVED's pin")
        if a.get("coef_json_sha256") and p.coef_sha != a["coef_json_sha256"]:
            R.append("coef.json sha256 differs from APPROVED's pin")
        if p.coef.get("stack") != C.STACK:
            R.append("coef.json records another stack")
        cutc, cmt = _utc(p.coef.get("utc")), _utc(p.coef_mtime_utc)
        for name, m in p.run_metas.items():
            if m is None or name in p.lost:
                continue
            us = _utc(m.get("utc_start"))
            if us is None or cutc is None or cmt is None or not (cutc < us and cmt < us):
                R.append(f"coef.json was not written before {name}'s first training step")
    # panels
    if p.overlay_problems:
        R.extend(f"measuring copy: {x}" for x in p.overlay_problems)
    want_seeds = f"{C.PANEL_SEEDS[0]}-{C.PANEL_SEEDS[-1]}"
    for r in p.rows:
        tag = f"panel {r.get('name')} {r.get('label')}"
        if not p.sealed_tree_digest or r.get("measure_tree_digest") != p.sealed_tree_digest:
            R.append(f"{tag}: measuring-copy tree digest differs from the sealed one")
        if r.get("seeds") != want_seeds or str(r.get("n_episodes")) != str(len(C.PANEL_SEEDS)):
            R.append(f"{tag}: seeds {r.get('seeds')} / {r.get('n_episodes')} episodes, not {want_seeds} / 20")
        try:
            stack = json.loads(r.get("stack") or "null")
        except ValueError:
            stack = None
        if stack != C.STACK:
            R.append(f"{tag}: stack {r.get('stack')} is not the pinned stack")
    # checkpoint provenance: a row is read only if it describes the checkpoint now in runs/<name> (the crash rule
    # moves a crashed attempt's folder and rows away; a stale row left behind is caught here)
    screen = set(names())
    for r in p.rows:
        name, label = r.get("name"), r.get("label")
        if name not in screen or name in p.lost:
            continue
        got = p.ckpt_sha.get((name, label))
        if got is None or r.get("zip_sha256") != got:
            R.append(f"panel {name} {label}: zip_sha256 {r.get('zip_sha256')} is not the sha256 of the checkpoint now "
                     f"in runs/{name} ({got})")
        if label == R2:
            m = p.run_metas.get(name) or {}
            final = (m.get("final_sha256") or {}).get("stage1_final.zip")
            if final is None or r.get("zip_sha256") != final:
                R.append(f"panel {name} final: zip_sha256 {r.get('zip_sha256')} != run_meta final_sha256 {final}")
    # onset note completeness: every non-LOST s80 CONTROL replicate needs all 9 labels
    for seed in C.REP_SEEDS:
        name = f"s80_control_r{seed}"
        if name in p.lost:
            continue
        have = {r.get("label") for r in p.rows if r.get("name") == name}
        missing = [lb for lb in C.LABELS if lb not in have]
        if missing:
            R.append(f"onset note read on {len(C.LABELS) - len(missing)} of {len(C.LABELS)} labels for {name} "
                     f"(missing {', '.join(missing)})")
    # V1
    if p.v1 is None or p.v1.get("passed") is not True:
        R.append("V1 did not reproduce its values exactly (validate/v1.json missing or not passed)")
    return R


def _sha(path: Path) -> str | None:
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path):
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def load_prov_inputs(root: Path, runs: Path, rows: list[dict], lost: dict) -> ProvInputs:
    approved = _json(root / "APPROVED")
    pin = C.SCRATCH_COMMIT or (approved or {}).get("scratch_commit")
    diff = None
    if pin and Path(C.CLONE).is_dir():
        proc = subprocess.run(["git", "--no-optional-locks", "-C", C.CLONE, "diff", "--name-only",
                               f"{C.PARENT}..{pin}"], capture_output=True, text=True)
        if proc.returncode == 0:
            diff = [x for x in proc.stdout.splitlines() if x.strip()]
    coef_path = root / "calib" / "coef.json"
    mt = None
    if coef_path.is_file():
        mt = datetime.fromtimestamp(coef_path.stat().st_mtime, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    seal = root / "measure_copy" / "TREE_DIGEST.txt"
    overlay_problems = []
    for rel, want in C.OVERLAYS.items():
        if _sha(root / "measure_copy" / rel) != want:
            overlay_problems.append(f"overlay {rel} sha256 differs from the pin")
    runlog = None
    try:
        runlog = Path(C.RUNLOG).read_text()
    except OSError:
        pass
    ckpt_sha = {}
    for r in rows:
        name, label = r.get("name"), r.get("label")
        if name in names() and label in C.LABELS and (name, label) not in ckpt_sha:
            zip_name = "stage1_final.zip" if label == "final" else f"stage1_{label}_steps.zip"
            ckpt_sha[(name, label)] = _sha(runs / name / zip_name)
    return ProvInputs(
        pins={"SCRATCH_COMMIT": C.SCRATCH_COMMIT, "DURABLE_BRANCH": C.DURABLE_BRANCH,
              "TOOL_SHA256": dict(C.TOOL_SHA256)},
        tool_sha_actual={t: _sha(TOOLS / t) for t in TOOL_FILES},
        approved=approved, criteria_sha=_sha(root / "criteria.txt"), runlog_text=runlog,
        run_metas={n: _json(runs / n / "run_meta.json") for n in names()},
        diff_files=diff, build_tier=_json(root / "tests_out" / "build_tier.json"),
        gate_pass=_json(root / "tests_out" / "gate_pass.json"), coef=_json(coef_path), coef_sha=_sha(coef_path),
        coef_mtime_utc=mt, sealed_tree_digest=(seal.read_text().split()[0] if seal.is_file() else None),
        overlay_problems=overlay_problems, v1=_json(root / "validate" / "v1.json"), rows=rows, lost=lost,
        screen_common_sha=_sha(TOOLS / "screen_common.py"), ckpt_sha=ckpt_sha)


# ------------------------------------------------------------------------------------------------- score

def score(rows: list[dict] | None, lost: dict, prov: ProvInputs | None) -> dict:
    """The whole score from rows (None = panels.csv missing), LOST and the PROVISIONAL inputs (None = skip)."""
    idx = index_rows(rows or [])
    reps = replicates(idx, lost)
    incomplete = sorted(n for n, r in reps.items() if r["missing"] and r["missing"] != "LOST")
    out = {"schema": "tailscreen-score/1", "utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "status": None, "provisional_reasons": [], "incomplete": incomplete, "lost": lost, "replicates": reps,
           "states": {}, "onset_note": None, "verdicts": {}, "outcome": None, "report_only": report_only(idx, reps),
           "pins": {"SCRATCH_COMMIT": C.SCRATCH_COMMIT, "DURABLE_BRANCH": C.DURABLE_BRANCH,
                    "TOOL_SHA256": dict(C.TOOL_SHA256)}}
    if prov is not None:
        out["provisional_reasons"] = provisional_reasons(prov)
    if incomplete:
        out["status"] = "incomplete"
        return out
    states, onset, verdicts = readings(reps)
    out.update({"states": states, "onset_note": onset, "verdicts": verdicts, "outcome": outcome(verdicts)})
    out["status"] = "provisional" if out["provisional_reasons"] else "final"
    return out


def render(s: dict) -> str:
    L = [f"TAIL_3 BOUND SCREEN SCORE ({s['utc']})", f"STATUS: {s['status'].upper()}"]
    if s["status"] == "incomplete":
        L.append("Incomplete: no line read. Missing R1/R2 panels (not LOST):")
        L += [f"  {n}: {s['replicates'][n]['missing']}" for n in s["incomplete"]]
    if s["provisional_reasons"]:
        L.append("PROVISIONAL (not acted on) while these stand:" if s["status"] != "incomplete"
                 else "PROVISIONAL items that also stand:")
        L += [f"  - {x}" for x in s["provisional_reasons"]]
    if s["lost"]:
        L.append("LOST (maintainer): " + ", ".join(f"{k} [{v}]" for k, v in s["lost"].items()))
    L.append("")
    L.append("Replicates (R1 = 5111808, R2 = final):")
    for n, r in s["replicates"].items():
        cm = "" if r["cmd_mean"] is None else f"CMD R1 {r['cmd_R1']:+.4f} R2 {r['cmd_R2']:+.4f} mean {r['cmd_mean']:+.4f}"
        L.append(f"  {n:16s} {str(r['class']):16s} tailup {r['tailup_R1R2']} other {r['other_R1R2']} "
                 f"onset {int(r['onset_any_tailup'])} {cm} {r['missing'] or ''}".rstrip())
    if s["status"] != "incomplete":
        L.append("")
        L.append(f"OUTCOME: {s['outcome']['text']}")
        for b, v in s["verdicts"].items():
            L.append(f"  {b}: {v['verdict']}" + (f" ({v['reason']}): {REASON_TEXT[v['reason']]}" if v["reason"] else
                                                  f" (onset note: {s['onset_note']})"))
        for st, d in s["states"].items():
            for b, v in d.items():
                L.append(f"  {st} {b}: {v['reading']} (lines applying: {', '.join(v['lines_applying'])}; OTHER "
                         f"{v['other_bound']} vs CONTROL {v['other_control']})")
        L.append(f"  Onset note: {s['onset_note']}")
        L.append("To the maintainer:")
        L += [f"  - {x}" for x in s["outcome"]["to_maintainer"]]
    L.append("")
    L.append("Report only (never in a verdict):")
    for st, d in s["report_only"]["dose_ordering"].items():
        arms = ", ".join(f"{a} {('%+.4f' % v['cmd_R1R2_arm_mean']) if v['cmd_R1R2_arm_mean'] is not None else 'n/a'}"
                         f" {v['classes']}" for a, v in d["arms"].items())
        L.append(f"  dose ordering {st}: BOUND60 <= BOUND90 <= CONTROL: {d['ordered_b60_le_b90_le_control']} ({arms})")
    L.append("  0.5M readings (4587520 / 4718592): see score.json report_only.half_point")
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--panels", type=Path, default=SCREEN / "panels.csv")
    ap.add_argument("--runs", type=Path, default=SCREEN / "runs")
    ap.add_argument("--out", type=Path, default=SCREEN / "scores")
    ap.add_argument("--root", type=Path, default=SCREEN, help=argparse.SUPPRESS)
    ns = ap.parse_args(argv)
    try:
        rows = read_panels(ns.panels)
        lost = read_lost(ns.root / "LOST.txt")
        prov = load_prov_inputs(ns.root, ns.runs, rows or [], lost)
        s = score(rows, lost, prov)
    except InputError as exc:
        print(f"input error: {exc}", file=sys.stderr)
        return EXIT_INPUT
    if rows is None:
        s["incomplete_note"] = f"{ns.panels} missing"
    ns.out.mkdir(parents=True, exist_ok=True)
    (ns.out / "score.json").write_text(json.dumps(s, indent=1, default=str) + "\n")
    text = render(s)
    (ns.out / "score.txt").write_text(text)
    print(text, end="")
    return {"final": EXIT_FINAL, "incomplete": EXIT_INCOMPLETE, "provisional": EXIT_PROVISIONAL}[s["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
