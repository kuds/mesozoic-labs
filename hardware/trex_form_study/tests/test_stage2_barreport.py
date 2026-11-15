"""Tests for the bar's tables (designlab/stage2/barreport.py): what is read, the words of every cell, the checks at
the top, the comparison with JPS section 2.3, and that the two files regenerate byte for byte.

Run from hardware/trex_form_study (the fit's venv):  python -m pytest -p no:cacheprovider tests/test_stage2_barreport.py -q
All fast.  Every record is synthetic: made with the runner's own functions (``bar.record_key``,
``bar.condition_record``, ``bar.not_run_record``, ``bar.record_path``) from episode dicts written here, in pytest's
temporary directory.  Nothing is simulated, no array is needed, and no number in this file is a result.  The
synthetic F25F9 has the shape of the frozen one (``stage2/sets/F25F9/75-M-H-10``: no nominal walk, the lead-in
``start_v0.4``, two rungs that no change reaches); two tests read the three frozen sets' own committed documents
(JSON only) to hold that shape, the plan's numbers of the changes tried and the references' own toe readings.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from designlab.stage1 import panels  # noqa: E402
from designlab.stage2 import bar, barreport, s2io, setio  # noqa: E402

BODY = "75-M-H-10"
H, R = bar.HEADLINE_BLOCK, bar.REPLICATION_BLOCK
TODAY = barreport._today_sources()
PROV = dict(git=dict(commit="0" * 40, dirty=False, dirty_paths=[]))
SESSION = dict(id="2026-11-15T08:00:00Z/1", started_at="2026-11-15T08:00:00Z", commit="0" * 40, dirty=False, wall_s=5.0)
MARGIN_RULE = "duty and lift set after results on seeds 3042-3061 (JPS 2.3)"
STAGE1_COMMAND = ("Stage 1's (decision 18): where the reference's path speed begins to ramp down; the reference "
                  "anticipates this command")
HEADING_RULE = "no reference within the heading rule"
OWN, LIMITS = barreport.OWN_NUMBERS, barreport.SET_LIMITS


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


# ------------------------------------------------------------------------------------------------
# Synthetic sets, episodes and records
# ------------------------------------------------------------------------------------------------
class Row:
    """What the runner's record functions read of a row."""

    def __init__(self, name: str):
        self.name = name

    def describe(self) -> dict:
        return dict(row=self.name.rsplit("@", 1)[0], delayed_row=self.name, latency_ms=bar.delay_ms(self.name))

    def sources(self) -> dict:
        return {}

    def gated(self, cond) -> bool:
        return bool(cond.gated)


TWIST_RULE = "no reference within the thigh twist and the leg drives"
#: What a synthetic set holds no reference for: {move: (reason, the reference's own numbers)}.
UNAVAILABLE = {
    "V0": {setio.turn_name("1", d): (HEADING_RULE, dict(heading_at_slowest_clock_deg=78.4, limit_deg=80.0))
           for d in setio.DIRECTIONS},
    "F25F9": {setio.turn_name("1", d): (HEADING_RULE, dict(heading_at_slowest_clock_deg=77.7, limit_deg=80.0))
              for d in setio.DIRECTIONS},
    "F25F9cl": dict({setio.turn_name("1", d): (HEADING_RULE, dict(heading_at_slowest_clock_deg=77.7, limit_deg=80.0))
                     for d in setio.DIRECTIONS},
                    **{setio.turn_name(setio.QUICK, "right"): (TWIST_RULE, dict(range_use=0.91, limit=0.87, f=0.6))}),
}
A1_OVER = {"F25F9cl": (setio.turn_name(setio.QUICK, "left"),)}
#: Rule A8 as amended: a set without a frozen nominal walk starts straight into its 0.40 m/s walk.
LEAD_IN = {"F25F9": "start_v0.4"}
START_V04 = "lead-in `start_v0.4`: from standing straight into the 0.40 m/s walk ({}; STAGE2_75_PLAN.md A8, changed after a result)"
#: The synthetic F25F9 in the frozen set's shape: its walks under 0.40 m/s are dropped from the ladder (no file), so
#: it has no `start` and no change from the nominal walk; the changes from the neighbouring rung into 0.45 and
#: 0.50 m/s failed a gate and are recorded under the direct change's name with what was tried.
DROPPED_WALKS = (setio.NOMINAL, "v0.3", "v0.35")
DROPPED = "dropped from the ladder: over clearance swing_clearance (synthetic)"
NOT_A_RUNG = "the walk nominal is not a rung of the set (synthetic)"
NO_CHANGE = NOT_A_RUNG + "; no change from the neighbouring rung within the gates"
#: {rung: (the neighbour's change that was tried, the last value of its ZMP error history in m)}: made-up numbers.
TRIED_ZMP = {"v0.45": ("v0.4_to_v0.45", 0.004321), "v0.5": ("v0.45_to_v0.5", 0.004987)}
#: The synthetic references' own toe readings in m (``refcheck.json``): {set: {condition: reading}}, 0.0222 elsewhere.
H_M = 0.501854
REF_TOE = {"F25F9": {"walk/v0.4": 0.019712}, "F25F9cl": {"turn/q90_left": 0.016208}, "V0": {"walk/v0.4": 0.025111}}
TOE_LIMIT_M = 0.0200741


def walk_numbers(w: str) -> dict:
    return dict(stage0=dict(within_limits=False, over=[dict(limit="clearance swing_clearance", value=0.0371)]),
                speed_m_s=setio.parse_name(w)["speed_m_s"] or 0.2431)


def tried_numbers(rung: str) -> dict:
    """The numbers of a rung no change reaches, as the frozen F25F9 records them: the dropped walk, then what was
    tried in order (the direct change, not built; the neighbour's change with one stride of ramp, built, failed)."""
    change, last = TRIED_ZMP[rung]
    return dict(walks={setio.NOMINAL: walk_numbers(setio.NOMINAL)}, tried=[
        dict(change=setio.change_name(setio.NOMINAL, rung), not_built=NOT_A_RUNG),
        dict(gates=dict(ik_ok=True, join_b=True, join_a=True, zmp_error=False), pass_=False, ik_ok=True,
             join_a=dict(com_max_mm=0.031, zmp_max_mm=1.41, com_ok=True, zmp_ok=True),
             zmp_error_history_m=[0.0471, 0.0212, 0.0061, last], failed_gates=["zmp_error"],
             stage0=dict(within_limits=True, over=[], blocks_freeze=[]), ramp_strides=1.0, change=change)])


def set_docs(name: str) -> dict:
    """A set's documents: every walk and move of the round, the moves of ``UNAVAILABLE`` recorded without a file
    (``available: false`` with their reason and numbers), one freeze that lists every entry with a file and cites
    the set's kinematic check (``refcheck.json``, with the sha256 of the text ``put`` writes).  F25F9 has the frozen
    set's shape (``DROPPED_WALKS``, ``TRIED_ZMP``)."""
    walks = [setio.NOMINAL] + [setio.walk_name(v) for v in setio.SPEEDS]
    moves = ([setio.START] + [setio.change_name(setio.NOMINAL, w) for w in walks[1:]]
             + [setio.stop_name(2, bar.BAR_WALK), setio.stop_name(1, bar.BAR_WALK)]
             + [setio.turn_name(r, d) for r in setio.TURN_RATES for d in setio.DIRECTIONS])
    base = name.split(".")[0]
    margin = MARGIN_RULE if name.startswith("F25F9cl") else None
    gone = UNAVAILABLE.get(base, {})
    refs = {w: dict(file=f"refs/{w}.npz", sha256=sha(name + w), margin_rule=margin, label=None) for w in walks}
    trs = {m: dict(file=f"transitions/{m}.npz", sha256=sha(name + m), a1_over=m in A1_OVER.get(name, ()),
                   move=dict(causal=True)) for m in moves if m not in gone}
    for m, (reason, numbers) in gone.items():
        trs[m] = dict(name=m, kind="turn", available=False, reason=reason, numbers=numbers)
    not_available = {}
    if base == "F25F9":
        for w in DROPPED_WALKS:
            refs[w] = dict(name=w, kind="walk", available=False, reason=DROPPED, numbers=walk_numbers(w), dropped=True)
        trs[setio.START] = dict(name=setio.START, kind="start", available=False, numbers=dict(walks={}),
                                reason=NOT_A_RUNG + "; the set starts from standing straight into v0.4 (start_v0.4)")
        alt = setio.start_name(bar.BAR_WALK)
        trs[alt] = dict(file=f"transitions/{alt}.npz", sha256=sha(name + alt), a1_over=False, move=dict(causal=False))
        for w in walks[1:]:
            m = setio.change_name(setio.NOMINAL, w)
            trs[m] = dict(name=m, kind="change", available=False, reason=NOT_A_RUNG, numbers=dict(walks={}))
            if w in TRIED_ZMP:
                trs[m].update(reason=NO_CHANGE, numbers=tried_numbers(w), fallback=dict(rule="S2S-fallback", step=3))
        not_available = {n: dict(reason=e["reason"], numbers=e.get("numbers"))
                         for n, e in {**refs, **trs}.items() if e.get("available") is False}
    manifest = {n: dict(kind="walk" if n in refs else "move", sha256=e["sha256"], entry_sha256=sha("entry" + name + n))
                for n, e in {**refs, **trs}.items() if e.get("file")}
    sd = dict(set=name, body=BODY, role="synthetic", rules={}, not_available=not_available,
              lead_in=LEAD_IN.get(base, setio.START),
              freezes=[dict(index=0, at="2026-11-15T07:00:00Z", commit="a" * 40, manifest=manifest,
                            manifest_sha256=s2io.sha256_json(manifest))])
    docs = {"set.json": sd, "refs.json": {"references": refs}, "transitions.json": {"transitions": trs},
            "screen.json": None}
    docs["refcheck.json"] = refcheck_doc(name, docs)
    sd["freezes"][0]["refcheck"] = dict(file=f"stage2/sets/{name}/{BODY}/refcheck.json",
                                        sha256=sha(refcheck_text(docs["refcheck.json"])), pass_=None)
    return docs


def refcheck_doc(name: str, docs: dict) -> dict:
    """A set's kinematic check in the runner's form (``bar.refcheck``): one item per walk, turn and stop condition at
    mu 0.9, "not run" where the set has no reference; a walk's and a turn's item holds the reference's own toe
    reading (``REF_TOE``; made-up numbers)."""
    planner = bar.BarPlanner(Path("unused"), docs["refs.json"]["references"], docs["transitions.json"]["transitions"],
                             docs["set.json"])
    items = {}
    for c in bar.conditions(docs):
        if c.friction is not None or c.judge == "bar_nudge":
            continue
        why = planner.missing(c)
        if why is not None:
            items[c.key] = dict(condition=c.key, status="not run", reason=why["reason"], reference=why["reference"],
                                pass_=None)
        elif c.judge == "bar_stop":
            items[c.key] = dict(condition=c.key, status="done", pass_=True)
        else:
            m = REF_TOE.get(name.split(".")[0], {}).get(c.key, 0.0222)
            items[c.key] = dict(condition=c.key, status="done", pass_=m >= TOE_LIMIT_M, toe=dict(
                toe_clearance_own_swing_m=m, toe_clearance_own_swing_H=m / H_M, limit_m=TOE_LIMIT_M,
                ok=m >= TOE_LIMIT_M))
    return dict(schema="bar-refcheck-1", set=name, body=BODY, items=items)


def refcheck_text(doc: dict) -> str:
    return json.dumps(doc, indent=1)


def write_set(study: Path, name: str, docs: dict | None = None) -> dict:
    """A synthetic set's documents and its kinematic check where the tables look for them under ``study``."""
    docs = set_docs(name) if docs is None else docs
    d = study / "stage2" / "sets" / name / BODY
    for n in ("set.json", "refs.json", "transitions.json"):
        s2io.write_json(d / n, docs[n])
    (d / "refcheck.json").write_text(refcheck_text(docs["refcheck.json"]))
    return docs


def episode(cond, seed: int, fell: bool = False, before=None, tail_over: bool = False, fail=(), error=None,
            speed=None, stage1: bool = True, touchdowns: int = 2, convention: str = "causal: the move's first frame") -> dict:
    """A synthetic episode record of a bar condition: it stays up and succeeds unless told otherwise; ``tail_over``
    fails the sustained-drive term in reading 2 only; ``speed`` is a walk's measured speed (default 3 % over the
    rung's target, so that no measured value equals a target)."""
    if error:
        return bar.failed_episode(cond, seed, error)
    terms, t3 = bar.terms_of(cond), bar.terms_but_a1(cond)
    s2 = {t: not fell and t not in fail and not (tail_over and t == "drive_sustained") for t in terms}
    s3 = {t: not fell and t not in fail for t in t3}
    tail = 1.21 if tail_over else 0.71
    span = dict(worst=["tail_yaw_servo", tail] if tail_over else ["left_knee_servo", 0.83],
                worst_leg=["left_knee_servo", 0.83], worst_a1=["tail_yaw_servo", tail],
                drive_sustained=not tail_over, drive_sustained_legs=True)
    j = cond.judge
    windowed = j in ("bar_walk", "bar_turn")
    m_bar: dict = dict(drives=dict(judged_span="window" if windowed else "episode",
                                   window=span if windowed and not fell else None, episode=span))
    measures: dict = dict(bar=m_bar, window=None, stage1=None, legs=None, push=None, recovery=None)
    toe = dict(toe=dict(limit_m=0.0200741), toe_clearance_own_swing_m=0.0236 + 1e-5 * (seed % 7),
               toe_clearance_own_duration_m=0.0171, stage1_toe_clearance_m=0.0228)
    if j == "bar_walk":
        v = setio.parse_name(cond.walk)["speed_m_s"] or 0.2431
        sp = 1.03 * v if speed is None else float(speed)
        measures["stage1"] = dict(succeeded=bool(stage1) and not fell, speed_target_m_s=v)
        if not fell:
            m_bar.update(toe, along_commanded_heading_m=10.0 * sp, lateral_from_line_max_m=0.051 + 0.001 * (seed % 7),
                         along_target_m=10.0 * v, line_max_m=0.25)
            measures["window"] = dict(speed_m_s=sp)
    elif j == "bar_turn" and not fell:
        m_bar.update(toe, heading_error_at_deg=-4.3 - 0.1 * (seed % 5), heading_error_hold_max_deg=6.2, tol_deg=10.0,
                     at_s=2.0, hold_s=3.0, time_to_within_10deg_s=1.71, turned_own_deg=86.4)
    elif j == "bar_nudge":
        measures["push"] = dict(force_N=[0.0, 21.87, 0.0], frac=float(cond.frac))
        measures["recovery"] = dict(after_s=None if fell else 1.23, limit_s=3.0, recovered=not fell)
    elif j == "bar_stop" and not fell:
        m_bar.update(touchdowns_after_command=int(touchdowns), forward_stopped_s=1.11, forward_stopped_strides=1.31,
                     stop_acceleration_1hz_m_s2=0.61, deceleration_raw_peak_m_s2=1.34, command_convention=convention)
    walk = j == "bar_walk"
    return dict(seed=int(seed), fell=bool(fell), fall_time_s=3.0 if fell else None, fall_cause="tilt" if fell else None,
                fall_onset_s=2.5 if fell else None, fall_candidates=["slip"] if fell else None,
                limits_before_onset=None, survival_s=3.0 if fell else 20.0, planned_s=20.0, strides=3.5 if fell else 24.0,
                success=s2, succeeded=all(s2.values()), success_but_A1=s3, succeeded_but_A1=all(s3.values()),
                failed_terms=["survived"] if fell else [t for t, v in s2.items() if not v],
                failed_terms_but_A1=["survived"] if fell else [t for t, v in s3.items() if not v],
                command_s=None if walk else 13.0, command_ref_s=None,
                fell_before_command=None if walk else (bool(before) if fell else False), segment_reached=cond.walk,
                window_reached=not fell, omega=None, crouch_frac=None, measures=measures, relaxed=[], wall_s=0.25)


def put(study: Path, out: Path, set_name: str, row: str, key: str, block: str = H, control=None, not_run=None,
        sources=None, extra=None, per_seed=None, bad_input: bool = False, **kw) -> dict:
    """Write one synthetic record where the runner would (``bar.record_path``) and the set's documents where the
    record's ``inputs.dir`` says; returns the record.  ``per_seed(seed)`` gives an episode's own arguments."""
    docs = write_set(study, set_name)
    cond = next(c for c in bar.conditions(docs, block, every=True) if c.key == key)
    seeds = list(bar.BLOCKS[block])
    row_ = Row(row)
    lead = docs["set.json"]["lead_in"]  # (a set that starts straight into its 0.40 m/s walk has no nominal walk)
    files = sorted(({f"refs/{setio.NOMINAL}.npz"} if lead == setio.START else set())
                   | {f"refs/{cond.walk}.npz"}
                   | ({f"transitions/{cond.move}.npz"} if cond.move else set()))
    fz = docs["set.json"]["freezes"][0]
    listed = {f: {k: fz["manifest"][Path(f).stem][k] for k in ("sha256", "entry_sha256")} for f in files
              if Path(f).stem in fz["manifest"]}
    if bad_input:
        listed[files[-1]] = dict(listed[files[-1]], sha256=sha("another file"))
    inputs = dict(dir=f"stage2/sets/{set_name}/{BODY}", files=listed,
                  freeze=dict(index=0, manifest_sha256=fz["manifest_sha256"]))
    src = dict(TODAY, **(sources or {}))
    more = dict(extra or {})
    if docs["set.json"]["lead_in"] != setio.START:  # every record of such a set says so (A8)
        more.setdefault("lead_in", docs["set.json"]["lead_in"])
    if control:
        more.update(control=bar.CONTROLS[control][1], control_name=control)
    if not_run:
        k = bar.record_key(BODY, set_name, block, row_, cond, None, seeds, inputs, {}, src, None, None)
        rec = bar.not_run_record(BODY, set_name, block, row_, cond, k, not_run, PROV, inputs, docs, more, control)
    else:
        command = None
        if cond.move:
            conv = STAGE1_COMMAND if cond.name == "one_step" else "causal: the move's first frame"
            command = dict(name=cond.move, ref_s=14.5, phase=0.0, convention=conv)
            if cond.judge == "bar_stop":
                kw.setdefault("convention", conv)
        resolved = dict(plan=[["transition", lead], ["ref", cond.walk, 20]], seconds=20.0, files=files,
                        command=command)
        crouch = bar.CONTROLS[control][0] if control else 0.05
        k = bar.record_key(BODY, set_name, block, row_, cond, resolved, seeds, inputs, {}, src,
                           4.8281 if control else 4.8864, crouch)
        eps = [episode(cond, s, **{**kw, **(per_seed(s) if per_seed else {})}) for s in seeds]
        rec = bar.condition_record(BODY, set_name, block, row_, cond, eps, k, seeds, PROV, inputs, [SESSION], docs,
                                   extra=more, control=control)
    s2io.write_json(bar.record_path(out, set_name, BODY, block, row, cond, control), rec)
    return rec


def why_not(set_name: str, key: str) -> dict:
    """Why the runner would not run a condition on a synthetic set (``bar.BarPlanner.missing`` on its documents)."""
    docs = set_docs(set_name)
    planner = bar.BarPlanner(Path("unused"), docs["refs.json"]["references"], docs["transitions.json"]["transitions"],
                             docs["set.json"])
    return planner.missing(next(c for c in bar.conditions(docs, every=True) if c.key == key))


def run_path(set_name: str, row: str, key: str, block: str = H, control=None) -> str:
    """A record's path relative to the run directory, as ``budget.json`` lists it."""
    return (bar.block_dir(Path(), set_name, BODY, block, control) / "eval" / row / f"{key}.json").as_posix()


def build(study: Path, out: Path) -> None:
    """The synthetic run the tests read (comments say what each record is there for)."""
    cl, v0, f25 = "F25F9cl", "V0", "F25F9"
    few = lambda n: (lambda s: dict(fail=("duty",)) if s % 1000 < 42 + n else {})  # noqa: E731  (n episodes fail a term)
    # -- the candidate's walk at 10 ms: one row passes both readings, the other only but for A1 (the tail is over)
    put(study, out, cl, "R2/42@10ms", "walk/v0.4")
    put(study, out, cl, "R2/43@10ms", "walk/v0.4", tail_over=True)
    put(study, out, cl, "R0@10ms", "walk/v0.4", per_seed=lambda s: dict(fell=s < 3045))
    # ... replicated: R2/42 passes again, R2/43 does not (reading 3 triggered it)
    rep = lambda row, key: dict(replication_of=dict(  # noqa: E731
        block=H, record=run_path(cl, row, key), readings=["success", "but_A1"] if row == "R2/42@10ms" else ["but_A1"]))
    put(study, out, cl, "R2/42@10ms", "walk/v0.4", block=R, extra=rep("R2/42@10ms", "walk/v0.4"))
    put(study, out, cl, "R2/43@10ms", "walk/v0.4", block=R, extra=rep("R2/43@10ms", "walk/v0.4"), tail_over=True,
        per_seed=few(10))
    # -- on the low grip both rows pass; the ledger cut their replication
    put(study, out, cl, "R2/42@10ms", "walk/v0.4_mu0.54")
    put(study, out, cl, "R2/43@10ms", "walk/v0.4_mu0.54", per_seed=few(3))
    # -- 4 ms: both pass (no verdict word at a reported delay)
    put(study, out, cl, "R2/42@4ms", "walk/v0.4")
    put(study, out, cl, "R2/43@4ms", "walk/v0.4")
    # -- the ladder and the pull request's rows at 0 ms
    for row in ("R2/42@0ms", "R2/43@0ms"):
        put(study, out, cl, row, "walk/v0.4", speed=0.4151)
        put(study, out, cl, row, "walk/v0.45", speed=0.4662)
        put(study, out, cl, row, "walk/v0.5", speed=0.5139)
        put(study, out, v0, row, "walk/v0.35", speed=0.3661, tail_over=True, stage1=False)
    put(study, out, cl, "R2/42@10ms", "walk/v0.45", per_seed=lambda s: dict(fell=s < 3050))
    put(study, out, cl, "R2/43@10ms", "walk/v0.45", per_seed=lambda s: dict(fell=s < 3050))
    # -- the rest of the bar at 10 ms
    for row in ("R2/42@10ms", "R2/43@10ms"):  # a turn frozen with a1_over: reading 2 is not read
        put(study, out, cl, row, "turn/q90_left", tail_over=True)
        for key in ("turn/q90_right", "turn/t90_1_left"):  # no reference: the runner's own reason and numbers
            put(study, out, cl, row, key, not_run=why_not(cl, key))
        put(study, out, cl, row, "stop/one_step", per_seed=lambda s: dict(touchdowns=1))
        put(study, out, cl, row, "nudge/forward", per_seed=lambda s: dict(fell=s < 3044, before=s < 3043))
    put(study, out, cl, "R2/42@10ms", "nudge/left")  # its other row has no record
    # -- a re-run after a result: the registered record (it fails) is kept beside the later one (it passes)
    put(study, out, cl, "R2/43@10ms", "stop/two_step")
    put(study, out, cl, "R2/42@10ms", "stop/two_step", per_seed=lambda s: dict(touchdowns=3,
                                                                              fail=("touchdowns_after_command",)))
    path = bar.record_path(out, cl, BODY, H, "R2/42@10ms", SimpleNamespace(behavior="stop", name="two_step"))
    field = bar.supersede_record(path, json.loads(path.read_text()), "synthetic: run again to test the label")
    put(study, out, cl, "R2/42@10ms", "stop/two_step", extra=dict(supersedes=field))
    # -- the omega control
    put(study, out, cl, "R2/42@10ms", "walk/v0.4", control="omega_s1")
    # -- the controls of the round
    put(study, out, v0, "R2/42@10ms", "walk/v0.4", per_seed=lambda s: dict(fell=True))
    put(study, out, v0, "R2/43@10ms", "walk/v0.4", per_seed=lambda s: dict(fell=True),
        sources=dict(bar=sha("another bar.py")))  # made by other sources: listed, flagged, tabulated
    put(study, out, v0, "R0@10ms", "walk/v0.4", bad_input=True)  # an input that the latest freeze does not list
    put(study, out, f25, "R0@10ms", "walk/v0.4", per_seed=lambda s: dict(error="synthetic failure") if s == 3050 else {})
    # -- the ledger: task 3 started two units and cut a third; task 4 has none
    started = [run_path(cl, r, k) for r in ("R2/42@10ms", "R2/43@10ms", "R0@10ms") for k in ("walk/v0.4", "walk/v0.4_mu0.54")]
    started += [run_path(f25, r, "walk/v0.4") for r in ("R2/42@10ms", "R2/43@10ms", "R0@10ms")]
    unit = lambda name, paths, used, before=0.0: dict(  # noqa: E731
        unit=name, estimate_core_h=0.4, spent_before_core_h=before, started_at="2026-11-15T08:00:00Z",
        used_core_h=used / 3600.0, items={"episodes": float(used)}, detail=dict(paths=paths))
    cut = dict(unit="T3.2b", estimate_core_h=0.3, spent_before_core_h=15.9, at="2026-11-15T20:00:00Z",
               reason="spent 15.900 + estimate 0.300 core-h exceeds task 3's 16 core-h",
               label="cut by the pre-registered order",
               detail=dict(paths=[run_path(cl, r, "walk/v0.4_mu0.54", R) for r in ("R2/42@10ms", "R2/43@10ms")]))
    s2io.write_json(out / "budget.json", dict(schema="bar-budget-1", rule=bar.RULES["budget"], tasks={"3": dict(
        cap_core_h=16.0, cut=[cut], units=[
            unit("T3.1", started, 1800.0),
            unit("T3.2a", [run_path(cl, r, "walk/v0.4", R) for r in ("R2/42@10ms", "R2/43@10ms")], 5400.0, 0.5)])}))


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    study = tmp_path_factory.mktemp("study")
    out = study / "stage2" / "runs" / "synthetic"
    build(study, out)
    doc = barreport.load(out, study=study)
    tabs = barreport.tables(doc)
    return SimpleNamespace(study=study, out=out, doc=doc, tabs=tabs, md=barreport.markdown(tabs))


def part(tabs: dict, table: str, key: str) -> dict:
    return next(p for p in tabs[table]["parts"] if p["key"] == key)


def lines(tabs: dict, table: str, key: str, **where) -> list[dict]:
    return [x for x in part(tabs, table, key)["lines"] if all(x.get(k) == v for k, v in where.items())]


def one(tabs: dict, table: str, key: str, **where) -> dict:
    found = lines(tabs, table, key, **where)
    assert len(found) == 1, (where, found)
    return found[0]


def every_line(tabs: dict):
    """(table, part, line) of every line of every table."""
    for name in barreport.ORDER:
        for p in tabs[name]["parts"]:
            for x in p["lines"]:
                yield name, p, x


def strings(x):
    if isinstance(x, dict):
        for v in x.values():
            yield from strings(v)
    elif isinstance(x, list):
        for v in x:
            yield from strings(v)
    elif isinstance(x, str):
        yield x


J, REP4 = "10 ms (judged)", "4 ms (reported)"
M = f" [{barreport.MARGIN_LABEL}]"  # how a sentence of the candidate's ends in the JPS table


# ------------------------------------------------------------------------------------------------
# Reading
# ------------------------------------------------------------------------------------------------
def test_load_indexes_every_record_by_set_body_block_row_condition(run):
    doc = run.doc
    rec = doc["F25F9cl"][BODY][H]["R2/43@10ms"]["walk/v0.4"]
    assert rec["counts"]["succeeded"] == 0 and rec["counts_but_A1"]["succeeded"] == 20
    assert rec["_path"] == "stage2/runs/synthetic/F25F9cl/75-M-H-10/seeds3042/eval/R2/43@10ms/walk/v0.4.json"
    assert set(doc) == {"F25F9cl", "F25F9", "V0", barreport.META}
    files = [q for q in run.out.rglob("*.json") if "eval" in q.parts and not any(x.endswith(".superseded") for x in q.parts)]
    main = [q for q in files if "controls" not in q.parts]
    assert len(list(barreport._records(doc))) == len(main) == run.tabs["checks"]["records"]["found"]
    meta = doc[barreport.META]
    assert list(meta["controls"]) == ["omega_s1"] and len(files) - len(main) == run.tabs["checks"]["records"]["controls"] == 1
    assert meta["out"] == "stage2/runs/synthetic" and meta["unreadable"] == [] and meta["misplaced"] == []
    # the record kept beside the re-run is read with it, and is no record of the index
    kept = doc["F25F9cl"][BODY][H]["R2/42@10ms"]["stop/two_step"]["_superseded"]
    assert len(kept) == 1 and kept[0]["record"]["counts"]["succeeded"] == 0 and kept[0]["reason"]["sequence"] == 1


def test_load_keeps_the_sets_and_the_body_asked_for(run):
    doc = barreport.load(run.out, sets=["V0"], study=run.study)
    assert set(doc) == {"V0", barreport.META} and doc[barreport.META]["controls"] == {"omega_s1": {}}
    assert set(barreport.load(run.out, body="60-N-T-10", study=run.study)) == {barreport.META}
    tabs = barreport.tables(doc)
    assert {x["set"] for _, _, x in every_line(tabs) if "set" in x} == {"V0"}
    with pytest.raises(s2io.S2Error, match="no such run directory"):
        barreport.load(run.out / "absent", study=run.study)


def test_files_that_are_no_record_are_listed_and_a_summary_is_skipped(run, tmp_path):
    out = tmp_path / "study" / "runs" / "x"
    put(tmp_path / "study", out, "V0", "R0@10ms", "walk/v0.4")
    edir = bar.block_dir(out, "V0", BODY, H) / "eval"
    bar.summarize(out, "V0", BODY, H)  # the runner's summary.json is no record
    (edir / "R0@10ms" / "walk" / "torn.json").write_text("{")
    s2io.write_json(edir / "R0@10ms" / "walk" / "other.json", dict(schema="panels-1"))
    (edir / "traces" / "R0@10ms").mkdir(parents=True)
    s2io.write_json(edir / "traces" / "R0@10ms" / "x.json", dict(schema=bar.SCHEMA))
    doc = barreport.load(out, study=tmp_path / "study")
    assert (edir / "summary.json").is_file() and len(list(barreport._records(doc))) == 1
    assert sorted(Path(p).name for p in doc[barreport.META]["unreadable"]) == ["other.json", "torn.json"]
    assert "torn.json" in barreport.markdown(barreport.tables(doc))


# ------------------------------------------------------------------------------------------------
# Byte for byte
# ------------------------------------------------------------------------------------------------
def test_the_tables_regenerate_byte_for_byte(run, tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    tabs = barreport.write(run.out, a, study=run.study)
    barreport.write(run.out, b, study=run.study)
    for name in (barreport.TABLES_JSON, barreport.TABLES_MD):
        assert (a / name).read_bytes() == (b / name).read_bytes()
    text = (a / barreport.TABLES_JSON).read_text()
    assert json.loads(text) == tabs == run.tabs
    # BAR_TABLES.md is a function of bar_tables.json alone
    assert barreport.markdown(json.loads(text)) == (a / barreport.TABLES_MD).read_text() == run.md
    assert not list(a.glob("*.tmp"))


def test_the_tables_hold_no_time_and_no_host(run, tmp_path):
    """The same files under another directory give the same bytes; with other times, hosts and commits written into
    the records, nothing changes but the sha256 of the files read (``checks.inputs``)."""
    study = tmp_path / "elsewhere"
    shutil.copytree(run.study, study)
    same = barreport.tables(barreport.load(study / "stage2" / "runs" / "synthetic", study=study))
    assert same == run.tabs and barreport.markdown(same) == run.md
    for q in (study / "stage2" / "runs").rglob("*.json"):
        doc = json.loads(q.read_text())
        if doc.get("schema") == bar.SCHEMA:
            doc["ended_at"] = "2031-01-01T00:00:00Z"
            doc["provenance"] = dict(git=dict(commit="f" * 40, dirty=True), host="another")
            doc["sessions"] = [dict(s, started_at="2031-01-01T00:00:00Z", commit="f" * 40) for s in doc.get("sessions") or []]
            q.write_text(json.dumps(doc))
    tabs = barreport.tables(barreport.load(study / "stage2" / "runs" / "synthetic", study=study))
    a, b = tabs["checks"]["inputs"], run.tabs["checks"]["inputs"]
    assert a["sha256"] != b["sha256"] and list(a["records"]) == list(b["records"])
    assert all(a["records"][k].get("key_sha256") == b["records"][k].get("key_sha256") for k in a["records"])
    without = lambda t: dict(t, checks=dict(t["checks"], inputs=None))  # noqa: E731
    assert without(tabs) == without(run.tabs)
    assert barreport.markdown(tabs).replace(a["sha256"], b["sha256"]) == run.md
    assert str(run.study) not in run.md and str(tmp_path) not in run.md
    assert "2031" not in run.md and "provenance" not in json.dumps(run.tabs)


def test_the_bytes_do_not_change_with_how_the_run_directory_is_typed(run, tmp_path, monkeypatch):
    """A run directory reached through a symbolic link, or by a relative path, is the same directory."""
    link = tmp_path / "study_link"
    link.symlink_to(run.study, target_is_directory=True)
    through = barreport.load(link / "stage2" / "runs" / "synthetic", study=run.study)
    assert through[barreport.META]["out"] == "stage2/runs/synthetic"
    assert barreport.markdown(barreport.tables(through)) == run.md
    assert barreport.tables(barreport.load(run.out, study=link)) == run.tabs  # the study typed through the link
    monkeypatch.chdir(run.out.parent)
    assert barreport.tables(barreport.load("synthetic", study=run.study)) == run.tabs
    # a run directory outside the study is cited from its own name, through a link too
    study = tmp_path / "other_study"
    study.mkdir()
    outside = tmp_path / "outside"
    shutil.copytree(run.out, outside / "run")
    (tmp_path / "ln").symlink_to(outside, target_is_directory=True)
    a = barreport.load(outside / "run", study=study)
    b = barreport.load(tmp_path / "ln" / "run", study=study)
    assert a[barreport.META]["out"] == b[barreport.META]["out"] == "run"
    assert [r["_path"] for r in barreport._records(a)] == [r["_path"] for r in barreport._records(b)]
    assert all(r["_path"].startswith("run/") for r in barreport._records(a))


def test_the_inputs_of_the_tables_are_listed_with_their_sha256(run, tmp_path):
    """What the tables were made from: every record file, the run directory's files, each set's documents and this
    module, each with the sha256 of its bytes, and one hash over the list."""
    inp = run.tabs["checks"]["inputs"]
    recs = {k: v for k, v in inp["records"].items() if v["kind"] == "a record"}
    assert sorted(recs) == sorted(r["_path"] for r in list(barreport._records(run.doc))
                                  + list(barreport._records(run.doc[barreport.META]["controls"]["omega_s1"])))
    for rel, v in inp["records"].items():
        q = run.study / rel
        assert v["sha256"] == hashlib.sha256(q.read_bytes()).hexdigest(), rel
        if v["kind"] != "the reason of a re-run":
            assert v["key_sha256"] == json.loads(q.read_text())["key_sha256"]
    assert Counter(v["kind"] for v in inp["records"].values()) == {
        "a record": 37, "a record kept beside a re-run": 1, "the reason of a re-run": 1}  # 36 and the control's
    assert inp["run_directory"] == {"budget.json": hashlib.sha256((run.out / "budget.json").read_bytes()).hexdigest(),
                                    "COST.md": None, "dev_episodes.json": None}
    assert list(inp["sets"]) == [f"stage2/sets/{s}/{BODY}" for s in ("F25F9", "F25F9cl", "V0")]
    for d, docs in inp["sets"].items():
        assert docs == {n: hashlib.sha256((run.study / d / n).read_bytes()).hexdigest() for n in barreport.SET_FILES}
    module = ROOT / barreport.MODULE_FILE
    assert module.samefile(barreport.__file__)
    assert inp["module"] == dict(file=barreport.MODULE_FILE, sha256=hashlib.sha256(module.read_bytes()).hexdigest())
    assert inp["sources"] == TODAY == run.tabs["checks"]["today_sources"]
    assert inp["sha256"] == s2io.sha256_json({k: v for k, v in inp.items() if k != "sha256"})
    top = run.md.split("## How to read the tables")[0]
    assert f"sha256 over the whole list: `{inp['sha256']}`" in top and f"`budget.json` `{inp['run_directory']['budget.json']}`" in top
    assert "37 record files; 1 record file kept beside a re-run, with 1 reason file; " in top and "`COST.md` absent" in top
    assert f"this module `{barreport.MODULE_FILE}`: `{inp['module']['sha256']}`" in top
    assert "`checks.inputs`" in run.tabs["conventions"][0] and "`budget.json`, `COST.md`, `dev_episodes.json`" in run.tabs["conventions"][0]
    # a set document that changes changes the list's hash, and nothing is hidden: the records are the same files
    study = tmp_path / "study"
    shutil.copytree(run.study, study)
    refs = study / "stage2" / "sets" / "V0" / BODY / "refs.json"
    refs.write_text(refs.read_text() + "\n")
    other = barreport.tables(barreport.load(study / "stage2" / "runs" / "synthetic", study=study))["checks"]["inputs"]
    assert other["records"] == inp["records"] and other["sha256"] != inp["sha256"]
    assert {d: [n for n in docs if docs[n] != inp["sets"][d][n]] for d, docs in other["sets"].items()} == {
        f"stage2/sets/F25F9/{BODY}": [], f"stage2/sets/F25F9cl/{BODY}": [], f"stage2/sets/V0/{BODY}": ["refs.json"]}
    # a set no record names and no document exists for: listed, with nothing to hash
    empty = tmp_path / "empty" / "runs" / "x"
    empty.mkdir(parents=True)
    none = barreport.tables(barreport.load(empty, study=tmp_path / "empty"))["checks"]["inputs"]
    assert none["records"] == {} and set(none["sets"]) == {f"stage2/sets/{s}/{BODY}" for s in barreport.SETS}
    assert all(h is None for docs in none["sets"].values() for h in docs.values())


def test_cli_writes_both_files(run, tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(s2io, "STUDY", run.study)  # the study directory the paths are relative to
    assert barreport.main(["--out", str(run.out), "--dest", str(tmp_path / "d"), "--set", "V0", "--set", "F25F9"]) == 0
    printed = capsys.readouterr().out
    assert "records found" in printed and str(tmp_path / "d" / barreport.TABLES_MD) in printed
    tabs = json.loads((tmp_path / "d" / barreport.TABLES_JSON).read_text())
    assert tabs["checks"]["sets_asked"] == ["V0", "F25F9"] and tabs["schema"] == barreport.SCHEMA
    assert barreport.markdown(tabs) == (tmp_path / "d" / barreport.TABLES_MD).read_text()
    assert barreport.main(["--out", str(tmp_path / "none")]) == 1


# ------------------------------------------------------------------------------------------------
# Readings, verdicts, delays
# ------------------------------------------------------------------------------------------------
def test_the_two_readings_are_never_pooled_and_differ_where_only_the_tail_is_over(run):
    x = one(run.tabs, "walk", "counts", set="F25F9cl", condition="walk/v0.4", delay=J, row="R2/43@10ms")
    assert x["r2"] == dict(k=0, n=20) and x["r3"] == dict(k=20, n=20) and x["up"] == dict(k=20, n=20)
    assert x["r2_pass"] == "does not pass" and x["r3_pass"] == "passes"
    assert x["r2_lcb"] == 0.0 and x["r3_lcb"] == pytest.approx(panels.lcb(20, 20))  # the record's own bounds
    assert x["reading_1"] == barreport.READING_1 and x["stage1"] == dict(k=20, n=20)
    f = one(run.tabs, "walk", "fails", set="F25F9cl", condition="walk/v0.4", delay=J, row="R2/43@10ms")
    assert f["failed_2"] == {"drive_sustained": 20} and f["failed_3"] == {}
    assert f["worst_drive"] == "tail_yaw_servo" and f["worst_ratio"] == dict(largest=1.21, n=20)
    assert f["worst_leg_ratio"] == dict(largest=0.83, n=20)
    # no count of any table pools readings or blocks: every "k of n" has n <= 20
    for _, p, line in every_line(run.tabs):
        for c in p["columns"]:
            v = line.get(c["key"])
            if c["kind"] == "of" and v is not None:
                assert v["n"] <= bar.RUNS and v["k"] <= v["n"], (p["key"], c["key"], v)
    # the readings stand in the plan's order in every table that has them
    for name in barreport.ORDER:
        for p in run.tabs[name]["parts"]:
            titles = [c["title"] for c in p["columns"]]
            order = [i for t in ("Reading 1", "Reading 2", "Reading 3") for i, x_ in enumerate(titles) if x_.startswith(t)][:3]
            assert order == sorted(order), (name, p["key"])


def test_shown_needs_both_rows_and_provisional_one(run):
    v = one(run.tabs, "walk", "verdicts", set="F25F9cl", condition="walk/v0.4", delay=J)
    assert v["reading_2"].startswith("provisional [") and v["reading_3"].startswith("shown but for A1 [")
    low = one(run.tabs, "walk", "verdicts", set="F25F9cl", condition="walk/v0.4_mu0.54", delay=J)
    assert low["reading_2"].startswith("shown [") and low["reading_3"].startswith("shown but for A1 [")
    none = one(run.tabs, "walk", "verdicts", set="V0", condition="walk/v0.4", delay=J)
    assert not none["reading_2"].startswith(("shown", "provisional"))
    assert none["reading_2"] == none["reading_3"] == f"{barreport.NEITHER_PASSES}: not attributed"
    # the words are the runner's own (bar.verdict), cell by cell
    doc = run.doc
    for ckey, ms, want2, want3 in (("walk/v0.4", 10, "provisional", "shown but for A1"),
                                   ("walk/v0.4_mu0.54", 10, "shown", "shown but for A1"),
                                   ("walk/v0.4", 4, "passes at 4 ms", "passes at 4 ms but for A1")):
        got = barreport.verdict_of(doc, "F25F9cl", BODY, ckey, ms)
        cells = [bar._cell(doc["F25F9cl"][BODY][H][f"{r}@{ms}ms"][ckey]) for r in barreport.R2_ROWS]
        assert got["reading_2"]["verdict"] == bar.verdict(cells, ms, "success")["verdict"] == want2
        assert got["reading_3"]["verdict"] == bar.verdict(cells, ms, "but_A1")["verdict"] == want3


def test_reading_1_is_not_defined_for_a_set_without_look_bands(run):
    cells = [x["reading_1"] for _, _, x in every_line(run.tabs) if "reading_1" in x]
    assert len(cells) > 100 and set(cells) == {barreport.READING_1}
    assert all(r["counts_look"] is None and r["counts_look_reason"] == bar.LOOK_NOT_DEFINED
               for r in barreport._records(run.doc))
    # a record that holds the reading is not called "not defined" (these tables do not tabulate it yet)
    rec = dict(run.doc["V0"][BODY][H]["R2/42@10ms"]["walk/v0.4"], counts_look=dict(succeeded=20))
    assert barreport.reading_1_of([None, rec]) == "held by the record (counts_look); not tabulated here"
    assert barreport.digest(rec)["reading_1"] != barreport.READING_1 and barreport.reading_1_of([None]) == barreport.READING_1


def test_an_r0_row_gives_no_verdict_and_one_row_alone_is_incomplete(run):
    assert all(x["controller"].startswith("R2") for x in part(run.tabs, "walk", "verdicts")["lines"])
    r0 = one(run.tabs, "walk", "counts", set="F25F9cl", condition="walk/v0.4", delay=J, row="R0@10ms")
    assert r0["up"] == dict(k=17, n=20) and r0["r2_pass"] == "does not pass"
    g = one(run.tabs, "glance", "glance", set="F25F9cl", condition="walk/v0.4")
    assert g["r0"] == barreport.NO_VERDICT_R0 and g["r0_up"] == dict(k=17, n=20)
    assert not any(x["row"].startswith("R0") for x in part(run.tabs, "replication", "replication")["lines"])
    alone = one(run.tabs, "rest", "verdicts", set="F25F9cl", condition="nudge/left", delay=J)
    assert alone["reading_2"] == "incomplete (R2/43@10ms: not run)"


def test_the_judged_delay_is_marked_and_another_delay_gives_no_verdict_word(run):
    delays = {x["delay"] for _, p, x in every_line(run.tabs) if "delay" in x}
    assert J in delays and REP4 in delays and all(d.endswith(("(judged)", "(reported)")) for d in delays)
    assert all(d.startswith("10 ms") == d.endswith("(judged)") for d in delays)
    for name in ("walk", "ladder", "rest"):  # the judged delay first within each set
        seen: dict = {}
        for x in part(run.tabs, name, "counts")["lines"]:
            seen.setdefault(x["set"], []).append(x["delay"])
        for ds in seen.values():
            if J in ds:
                assert ds[0] == J and J not in ds[ds.count(J):]
    v = one(run.tabs, "walk", "verdicts", set="F25F9cl", condition="walk/v0.4", delay=REP4)
    assert v["reading_2"] == "passes at 4 ms" and v["reading_3"] == "passes at 4 ms but for A1"
    x = one(run.tabs, "walk", "counts", set="F25F9cl", condition="walk/v0.4", delay=REP4, row="R2/42@4ms")
    assert x["r2_pass"] == "passes at 4 ms"
    for _, p, line in every_line(run.tabs):  # "shown" and "provisional" stand at the judged delay only
        if p["key"] == "verdicts" and line["delay"] != J:
            assert not line["reading_2"].startswith(("shown", "provisional")), line
            assert not line["reading_3"].startswith(("shown", "provisional")), line


def test_a_turn_frozen_with_a1_over_is_not_read_in_reading_2(run):
    x = one(run.tabs, "rest", "counts", set="F25F9cl", condition="turn/q90_left", delay=J, row="R2/42@10ms")
    assert x["r2_pass"] == "not read: a1_over" and x["r2"] == dict(k=0, n=20) and x["r3_pass"] == "passes"
    v = one(run.tabs, "rest", "verdicts", set="F25F9cl", condition="turn/q90_left", delay=J)
    assert v["reading_2"] == "not read: a1_over" and v["reading_3"].startswith("shown but for A1 [")
    g = one(run.tabs, "glance", "glance", set="F25F9cl", condition="turn/q90_left")
    assert g["reading_2"] == "not read: a1_over"
    # it is no pass: reading 3 alone triggers a replication line
    r = one(run.tabs, "replication", "replication", set="F25F9cl", condition="turn/q90_left", row="R2/42@10ms")
    assert r["triggered_by"] == ["reading 3 (every limit but for A1)"]


# ------------------------------------------------------------------------------------------------
# Not run, cut, re-run
# ------------------------------------------------------------------------------------------------
def test_a_missing_record_reads_not_run_and_a_unit_the_ledger_never_started_reads_cut(run):
    meta = run.doc[barreport.META]
    word = lambda s, row, key, block=H: barreport.absent_word(meta, s, BODY, block, row, key)  # noqa: E731
    # task 4 has no cut unit: a missing record is "not run"
    assert word("F25F9cl", "R2/43@10ms", "nudge/left") == barreport.NOT_RUN
    assert word("V0", "R2/42@10ms", "stop/two_step") == barreport.NOT_RUN
    # task 3 cut a unit: its record paths read "cut" ...
    assert word("F25F9cl", "R2/42@10ms", "walk/v0.4_mu0.54", R) == barreport.CUT
    # ... and so does every cell of the registered order that no started unit lists (the units after the cut one)
    assert word("V0", "R2/42@2ms", "walk/v0.4") == word("F25F9cl", "R2/42@6ms", "walk/v0.5") == barreport.CUT
    # a cell of a started unit whose record is not there, and a cell outside the registered order, are "not run"
    assert word("F25F9", "R2/42@10ms", "walk/v0.4") == barreport.NOT_RUN
    assert word("F25F9cl", "R2/42@8ms", "walk/v0.5") == barreport.NOT_RUN
    assert word("D0", "R2/42@10ms", "walk/v0.4") == barreport.NOT_RUN
    # in the tables: a whole set and delay without a record is one line with its word
    v2 = one(run.tabs, "walk", "counts", set="V0", delay="2 ms (reported)")
    assert v2["status"] == barreport.CUT and v2["condition"] == "walk/v0.4, walk/v0.4_mu0.54"
    assert v2["row"] == "R2/42@2ms, R2/43@2ms, R0@2ms" and v2.get("up") is None
    assert one(run.tabs, "walk", "verdicts", set="V0", delay="2 ms (reported)")["reading_2"] == barreport.CUT
    r4 = one(run.tabs, "rest", "counts", set="V0", delay=REP4)
    assert r4["status"] == barreport.NOT_RUN and r4["item"] == "14 conditions, 42 records: none found"
    # a single missing record among others has its own line and path
    f = one(run.tabs, "walk", "counts", set="F25F9", condition="walk/v0.4", delay=J, row="R2/42@10ms")
    assert f["status"] == barreport.NOT_RUN and f["path"].endswith("F25F9/75-M-H-10/seeds3042/eval/R2/42@10ms/walk/v0.4.json")
    assert one(run.tabs, "walk", "verdicts", set="F25F9", condition="walk/v0.4", delay=J)["reading_2"] == barreport.NOT_RUN
    assert f"| {barreport.CUT} |" in run.md and f"| {barreport.NOT_RUN} |" in run.md
    # without a ledger nothing is "cut"
    doc = copy.deepcopy(run.doc)
    doc[barreport.META]["budget"] = None
    cut = [s for _, _, x in every_line(barreport.tables(doc)) for s in strings(x)
           if barreport.CUT in s.replace(barreport.NO_REPLICATION_UNIT, "")]
    assert not cut


def test_a_condition_without_a_reference_prints_its_reason_and_the_references_numbers(run):
    x = one(run.tabs, "rest", "counts", set="F25F9cl", condition="turn/t90_1_left", row="R2/42@10ms")
    assert x["status"] == (f"not run: {HEADING_RULE} [transitions/turn90_1_left] ({OWN}: "
                           f"heading_at_slowest_clock_deg=77.7; {LIMITS}: limit_deg=80)")
    assert x["up"] is None and x["r2"] is None and x["r2_pass"] is None  # never a blank, never a fail of the robot
    v = one(run.tabs, "rest", "verdicts", set="F25F9cl", condition="turn/t90_1_left")
    assert v["reading_2"] == v["reading_3"] == x["status"]
    assert not lines(run.tabs, "rest", "fails", condition="turn/t90_1_left")
    g = one(run.tabs, "glance", "glance", set="F25F9cl", condition="turn/q90_right")
    assert g["reading_2"].startswith("not run: no reference within the thigh twist and the leg drives")
    assert g["reading_2"].endswith(f"({OWN}: range_use=0.91, f=0.6; {LIMITS}: limit=0.87)")
    tally = one(run.tabs, "glance", "tally", set="F25F9cl")
    assert sum(tally["reading_2"].values()) == sum(tally["reading_3"].values()) == 16
    assert {k: n for k, n in tally["reading_2"].items() if not k.startswith(("shown", "provisional"))} == {
        "incomplete": 1, f"{barreport.NEITHER_PASSES}: not attributed": 1, "not read: a1_over": 1, "not run": 10}
    # the control for the walk only: its 14 other gated conditions say why they have no record
    f = one(run.tabs, "glance", "glance", set="F25F9", condition="stop/two_step")
    assert f["reading_2"] == barreport.NOT_RUN and f["note"] == barreport.WALK_ONLY_NOTE
    assert one(run.tabs, "glance", "glance", set="V0", condition="stop/two_step")["note"] is None
    assert len(lines(run.tabs, "glance", "glance", set="V0")) == 16


def test_the_tally_counts_a_verdict_as_it_is_printed_with_its_replication_words(run, tmp_path):
    """H4: a candidate's `shown` or `provisional` is never printed without the replication word of each row's record
    that passed, in the count per verdict word too."""
    cut = "replication cut by the pre-registered order"
    tally = one(run.tabs, "glance", "tally", set="F25F9cl")
    assert tally["reading_2"] == {
        "incomplete": 1, f"{barreport.NEITHER_PASSES}: not attributed": 1, "not read: a1_over": 1, "not run": 10,
        "provisional [R2/42@10ms: replicated]": 1, "provisional [R2/43@10ms: replication not run yet]": 1,
        f"shown [R2/42@10ms: {cut}; R2/43@10ms: {cut}]": 1}
    assert tally["reading_3"]["shown but for A1 [R2/42@10ms: replicated; R2/43@10ms: not replicated]"] == 1
    bare = {"shown", "provisional", "shown but for A1", "provisional but for A1"}
    for x in part(run.tabs, "glance", "tally")["lines"]:
        cells = lines(run.tabs, "glance", "glance", set=x["set"], body=x["body"])
        for rkey in ("reading_2", "reading_3"):
            assert sum(x[rkey].values()) == 16 == len(cells) and not set(x[rkey]) & bare
            for word, n in x[rkey].items():  # a verdict word of the tally is a cell of the part above, word for word
                if word.startswith(("shown", "provisional")):
                    assert "[" in word and sum(1 for c in cells if c[rkey] == word) == n, word
    # a control's verdict says that a control is not replicated
    study = tmp_path / "study"
    out = study / "runs" / "x"
    for row in ("R2/42@10ms", "R2/43@10ms"):
        put(study, out, "V0", row, "walk/v0.4")
    tabs = barreport.tables(barreport.load(out, study=study))
    assert one(tabs, "glance", "tally", set="V0")["reading_2"] == {"not run": 15, "shown [a control is not replicated]": 1}
    # printed one word per line as "word = n": a word may hold a colon
    md = barreport.markdown(tabs)
    assert "| not run = 15<br>shown [a control is not replicated] = 1 |" in md
    assert "not read: a1_over = 1" in run.md and "not read: a1_over: 1" not in run.md
    assert barreport._cell_text({"a: b": 2, "c": 1}, dict(kind="tally")) == "a: b = 2<br>c = 1"
    assert barreport._cell_text({}, dict(kind="tally")) == "none"


def test_a_cell_with_no_record_says_when_the_set_holds_no_reference(run):
    """Where the runner wrote no "not run" record (its unit was cut, or has not run), the set's own documents give
    the reason, read by the runner's rule: the cell is never a bare word and never a fail of the robot."""
    gaps = run.doc[barreport.META]["set_gaps"]
    assert set(gaps["F25F9cl"][BODY]) == {"turn/q90_right", "turn/q90_right_mu0.54", "turn/t90_1_left", "turn/t90_1_right"}
    assert set(gaps["V0"][BODY]) == {"turn/t90_1_left", "turn/t90_1_right"}
    # the set in the frozen F25F9's shape: its dropped walks and the two rungs no change reaches, on both grips
    assert set(gaps["F25F9"][BODY]) == set(gaps["V0"][BODY]) | {
        f"walk/{w}{g}" for w in DROPPED_WALKS + tuple(TRIED_ZMP) for g in ("", "_mu0.54")}
    assert run.doc[barreport.META]["set_refused"] == {} and run.tabs["checks"]["sets_refused"] == {}
    note = ("; the set holds no reference for it: no reference within the heading rule [transitions/turn90_1_right] "
            f"({OWN}: heading_at_slowest_clock_deg=77.7; {LIMITS}: limit_deg=80)")
    x = one(run.tabs, "rest", "counts", set="F25F9cl", condition="turn/t90_1_right", row="R2/42@10ms")
    assert x["status"] == barreport.NOT_RUN + note and x.get("up") is None
    # the same reason and numbers as the runner's own "not run" record of the left turn
    left = one(run.tabs, "rest", "counts", set="F25F9cl", condition="turn/t90_1_left", row="R2/42@10ms")
    assert left["status"].split(": ", 1)[1].replace("_left", "_right") == note.split(": ", 1)[1]
    v = one(run.tabs, "rest", "verdicts", set="F25F9cl", condition="turn/t90_1_right")
    assert v["reading_2"] == v["reading_3"] == barreport.NOT_RUN + note and v["which"] == "no record"
    g = one(run.tabs, "glance", "glance", set="F25F9cl", condition="turn/q90_right_mu0.54")
    assert g["reading_2"].startswith(f"not run; the set holds no reference for it: {TWIST_RULE} [") and "f=0.6" in g["reading_2"]
    # a set and delay without any record: the conditions without a reference have their own lines
    v0 = lines(run.tabs, "rest", "counts", set="V0", delay=J)
    assert [x["item"] for x in v0] == ["27 conditions, 68 records: none found", "1 condition, 2 records: none found",
                                       "1 condition, 2 records: none found"]
    assert v0[0]["status"] == barreport.NOT_RUN and "heading_at_slowest_clock_deg=78.4" in v0[1]["status"]
    assert [x["condition"] for x in v0[1:]] == ["turn/t90_1_left", "turn/t90_1_right"]
    # listed at the top
    top = run.tabs["checks"]["no_reference"]
    assert list(top) == ["F25F9/75-M-H-10", "F25F9cl/75-M-H-10", "V0/75-M-H-10"] and len(top["F25F9cl/75-M-H-10"]) == 4
    assert len(top["F25F9/75-M-H-10"]) == 12
    assert "- `F25F9cl/75-M-H-10`, `turn/q90_right`: no reference within the thigh twist and the leg drives" in run.md
    # the absent word itself stays the bare word (a tally counts it as "not run")
    meta = run.doc[barreport.META]
    assert barreport.absent_word(meta, "F25F9cl", BODY, H, "R2/42@10ms", "turn/t90_1_right") == barreport.NOT_RUN
    assert barreport.gap_note(meta, "F25F9cl", BODY, "walk/v0.4") == ""


def test_a_rerun_after_a_result_stands_beside_the_registered_record(run):
    two = lines(run.tabs, "rest", "counts", set="F25F9cl", condition="stop/two_step", row="R2/42@10ms")
    assert len(two) == 2
    later, first = two
    assert any(x.startswith(barreport.RERUN + ": synthetic: run again") for x in later["labels"])
    assert later["r2"] == dict(k=20, n=20) and ".superseded" not in later["path"]
    assert any(x.startswith(barreport.REGISTERED) for x in first["labels"]) and first["r2"] == dict(k=0, n=20)
    assert ".superseded/" in first["path"] and first["labels"][0] == barreport.MARGIN_LABEL
    assert not any(x.startswith(barreport.RERUN) for x in first["labels"])
    # the verdict is the registered records'; the re-run's is printed beside it, labelled
    reg, re_ = lines(run.tabs, "rest", "verdicts", set="F25F9cl", condition="stop/two_step")
    assert reg["which"] == "registered records" and reg["reading_2"].startswith("provisional [")
    assert one(run.tabs, "walk", "verdicts", set="F25F9", condition="walk/v0.4", delay=J)["which"] == "no record"
    assert re_["which"] == barreport.RERUN and re_["reading_2"].startswith("shown [") and barreport.RERUN in re_["labels"]
    g = one(run.tabs, "glance", "glance", set="F25F9cl", condition="stop/two_step")
    assert g["reading_2"].startswith("provisional [")
    assert run.tabs["checks"]["records"]["kept_beside_a_rerun"] == 1
    # where the kept file cannot be read, the counts copied into the later record stand in
    rec = copy.deepcopy(run.doc["F25F9cl"][BODY][H]["R2/42@10ms"]["stop/two_step"])
    kept, rec["_superseded"] = rec["_superseded"], []
    brief = barreport.registered(rec)
    assert brief["_brief"] and brief["counts"]["succeeded"] == 0 and brief["counts"]["pass_"] is False
    # a later record that names no registered one (its reason file was lost): the oldest record kept beside it
    rec["_superseded"], rec["supersedes"] = kept, dict(reason=None, label=barreport.RERUN)
    assert barreport.registered(rec)["counts"]["succeeded"] == 0 and ".superseded/" in barreport.registered(rec)["_path"]
    assert barreport.registered(None) is None and barreport.registered(kept[0]["record"]) is kept[0]["record"]


def test_a_rerun_of_the_second_block_does_not_change_the_replication_word(run, tmp_path):
    study = tmp_path / "study"
    out = study / "runs" / "x"
    put(study, out, "F25F9cl", "R2/42@10ms", "walk/v0.4")
    of = dict(replication_of=dict(block=H, record=run_path("F25F9cl", "R2/42@10ms", "walk/v0.4"),
                                  readings=["success", "but_A1"]))
    put(study, out, "F25F9cl", "R2/42@10ms", "walk/v0.4", block=R, extra=of, per_seed=lambda s: dict(fell=True))
    path = bar.record_path(out, "F25F9cl", BODY, R, "R2/42@10ms", SimpleNamespace(behavior="walk", name="v0.4"))
    field = bar.supersede_record(path, json.loads(path.read_text()), "synthetic: the second block run again")
    put(study, out, "F25F9cl", "R2/42@10ms", "walk/v0.4", block=R, extra=dict(of, supersedes=field))
    doc = barreport.load(out, study=study)
    assert barreport.replication_word(doc, "F25F9cl", BODY, "R2/42@10ms", "walk/v0.4", "success") == "not replicated"
    assert bar.replication_word(out, "F25F9cl", BODY, "R2/42@10ms", "walk/v0.4", "success") == "replicated"  # the later one
    first, later = part(barreport.tables(doc), "replication", "replication")["lines"]
    assert first["b_up"] == dict(k=0, n=20) and first["words"][0].endswith(": not replicated")
    assert any(x.startswith(barreport.REGISTERED) for x in first["labels"]) and ".superseded/" in first["b_path"]
    assert later["b_up"] == dict(k=20, n=20) and later["words"] == [barreport.RERUN + ": the registered record's word stands"]
    assert any(x.startswith(barreport.RERUN + ": synthetic") for x in later["labels"])


def test_the_registered_record_kept_beside_a_rerun_is_checked_like_any_record(run, tmp_path):
    """Rule R2: the registered record gives every count and verdict, so the checks at the top read it too, and what
    it fails is a label of its line and of the verdict made from it."""
    study = tmp_path / "study"
    out = study / "runs" / "x"
    cl = "F25F9cl"
    put(study, out, cl, "R2/42@10ms", "walk/v0.4")
    put(study, out, cl, "R2/43@10ms", "walk/v0.4", per_seed=lambda s: dict(fell=s < 3048),
        sources=dict(bar=sha("another bar.py")), bad_input=True)
    path = bar.record_path(out, cl, BODY, H, "R2/43@10ms", SimpleNamespace(behavior="walk", name="v0.4"))
    first = json.loads(path.read_text())
    first["counts"].update(pass_=True)  # 14 of 20 up, and a stored pass
    field = bar.supersede_record(path, first, "synthetic: run again")
    kept = path.with_name(path.name + ".superseded") / field["record"]
    kept.write_text(json.dumps(first))
    put(study, out, cl, "R2/43@10ms", "walk/v0.4", extra=dict(supersedes=field))
    tabs = barreport.tables(barreport.load(out, study=study))
    c = tabs["checks"]
    assert c["records"]["found"] == 2 and c["records"]["kept_beside_a_rerun"] == 1
    for name in ("other_sources", "not_in_latest_freeze", "counts_not_recomputed"):
        assert [".superseded/" in x["path"] for x in c[name]] == [True], name
    assert c["counts_not_recomputed"][0]["differences"] == [
        "reading 2 (every limit): pass_ is True in the record, False from its episodes"]
    later, reg = lines(tabs, "walk", "counts", set=cl, condition="walk/v0.4", row="R2/43@10ms")
    assert reg["up"] == dict(k=14, n=20) and reg["r2_pass"] == "passes" and ".superseded/" in reg["path"]
    flags = [f"{barreport.OTHER_SOURCES} (bar)", barreport.NOT_IN_MANIFEST, barreport.MISCOUNTED]
    assert all(f in reg["labels"] for f in flags) and not any(f in later["labels"] for f in flags)
    v, w = lines(tabs, "walk", "verdicts", set=cl, condition="walk/v0.4", delay=J)
    assert v["which"] == "registered records" and v["reading_2"].startswith("shown [")
    assert v["labels"] == [barreport.MARGIN_LABEL] + [f"R2/43@10ms: {f}" for f in flags]
    assert w["which"] == barreport.RERUN and w["labels"] == [barreport.MARGIN_LABEL, barreport.RERUN]
    md = barreport.markdown(tabs)
    top = md.split("## How to read the tables")[0]
    assert top.count(".superseded/") == 3 and f"R2/43@10ms: {barreport.MISCOUNTED}" in md
    jps = {x["line"]: x for x in part(tabs, "jps", "jps")["lines"]}[99]["record"]
    assert any(s.startswith("F25F9cl walk/v0.4 at 10 ms, the controller R2") and s.endswith(
        f"[{barreport.MARGIN_LABEL}; " + "; ".join(f"R2/43@10ms: {f}" for f in flags) + "]") for s in jps)


def test_a_record_kept_beside_one_without_a_supersedes_field_is_still_the_registered_one(run, tmp_path):
    """The label and the registered record do not rest on the later record's own word alone: a
    ``<record>.superseded/`` directory beside a record says a record was made there before."""
    study = tmp_path / "study"
    out = study / "runs" / "x"
    put(study, out, "V0", "R2/42@10ms", "walk/v0.4")
    put(study, out, "V0", "R2/43@10ms", "walk/v0.4", per_seed=lambda s: dict(fell=True))  # the first record: all fall
    path = bar.record_path(out, "V0", BODY, H, "R2/43@10ms", SimpleNamespace(behavior="walk", name="v0.4"))
    sdir = path.with_name(path.name + ".superseded")
    sdir.mkdir()
    path.rename(sdir / "ffffffffffff.json")  # kept by hand: no reason file, and the later record names nothing
    put(study, out, "V0", "R2/43@10ms", "walk/v0.4")
    doc = barreport.load(out, study=study)
    rec = doc["V0"][BODY][H]["R2/43@10ms"]["walk/v0.4"]
    assert "supersedes" not in rec and len(rec["_superseded"]) == 1
    reg = barreport.registered(rec)
    assert reg is not rec and reg["counts"]["survived"] == 0 and reg["_path"].endswith(".superseded/ffffffffffff.json")
    tabs = barreport.tables(doc)
    later, first = lines(tabs, "walk", "counts", set="V0", condition="walk/v0.4", row="R2/43@10ms")
    assert later["up"] == dict(k=20, n=20) and first["up"] == dict(k=0, n=20)
    assert [x for x in later["labels"] if x.startswith(barreport.RERUN)] == [
        barreport.RERUN + ": no reason recorded (a record is kept beside it; this one holds no `supersedes` field)"]
    assert any(x.startswith(barreport.REGISTERED) for x in first["labels"])
    assert not any(x.startswith(barreport.RERUN) for x in first["labels"])
    v, w = lines(tabs, "walk", "verdicts", set="V0", condition="walk/v0.4", delay=J)
    assert (v["which"], v["reading_2"]) == ("registered records", "provisional [a control is not replicated]")
    assert (w["which"], w["reading_2"]) == (barreport.RERUN, "shown [a control is not replicated]")
    assert one(tabs, "glance", "glance", set="V0", condition="walk/v0.4")["reading_2"] == v["reading_2"]
    # the kept file cannot be read and no field copies its counts: the registered record has none, and says so
    (sdir / "ffffffffffff.json").write_text("{")
    tabs = barreport.tables(barreport.load(out, study=study))
    later, first = lines(tabs, "walk", "counts", set="V0", condition="walk/v0.4", row="R2/43@10ms")
    assert first["up"] is None and any("the later record holds no supersedes field: no counts" in x for x in first["labels"])
    v = lines(tabs, "walk", "verdicts", set="V0", condition="walk/v0.4", delay=J)[0]
    assert v["which"] == "registered records" and v["reading_2"] == "incomplete"


def test_a_rerun_that_has_not_finished_leaves_the_registered_record(run, tmp_path):
    study = tmp_path / "study"
    out = study / "runs" / "x"
    put(study, out, "F25F9cl", "R2/42@10ms", "walk/v0.4")
    path = bar.record_path(out, "F25F9cl", BODY, H, "R2/42@10ms", SimpleNamespace(behavior="walk", name="v0.4"))
    bar.supersede_record(path, json.loads(path.read_text()), "synthetic: killed before it finished")
    assert not path.exists()
    doc = barreport.load(out, study=study)
    rec = doc["F25F9cl"][BODY][H]["R2/42@10ms"]["walk/v0.4"]
    assert rec["_rerun_pending"] and ".superseded/" in rec["_path"]
    x = one(barreport.tables(doc), "walk", "counts", set="F25F9cl", condition="walk/v0.4", row="R2/42@10ms")
    assert x["r2"] == dict(k=20, n=20) and any("has no record yet" in s for s in x["labels"])


# ------------------------------------------------------------------------------------------------
# Replication
# ------------------------------------------------------------------------------------------------
def test_replication_sits_beside_its_original_and_is_never_pooled(run):
    a = one(run.tabs, "replication", "replication", set="F25F9cl", condition="walk/v0.4", row="R2/42@10ms")
    assert a["a_r2"] == a["b_r2"] == dict(k=20, n=20)
    assert a["a_path"].endswith("seeds3042/eval/R2/42@10ms/walk/v0.4.json")
    assert a["b_path"].endswith("seeds5042/eval/R2/42@10ms/walk/v0.4.json")
    assert a["triggered_by"] == ["reading 2 (every limit)", "reading 3 (every limit but for A1)"]
    assert a["words"] == ["reading 2 (every limit): replicated", "reading 3 (every limit but for A1): replicated"]
    b = one(run.tabs, "replication", "replication", set="F25F9cl", condition="walk/v0.4", row="R2/43@10ms")
    assert b["a_r3"] == dict(k=20, n=20) and b["b_r3"] == dict(k=10, n=20) and b["b_r2"] == dict(k=0, n=20)
    assert b["triggered_by"] == ["reading 3 (every limit but for A1)"]
    assert b["words"] == ["reading 3 (every limit but for A1): not replicated"]
    # a passing record whose replication the ledger cut is listed with that word and no second block
    c = one(run.tabs, "replication", "replication", set="F25F9cl", condition="walk/v0.4_mu0.54", row="R2/42@10ms")
    assert c["b_up"] is None and c["b_r2"] is None and all(w.endswith(bar.REPLICATION_WORDS[2]) for w in c["words"])
    assert c["b_path"].endswith("seeds5042/eval/R2/42@10ms/walk/v0.4_mu0.54.json")
    # the candidate's verdict carries the word of each row's record that passed
    v = one(run.tabs, "walk", "verdicts", set="F25F9cl", condition="walk/v0.4", delay=J)
    assert v["reading_2"] == "provisional [R2/42@10ms: replicated]"
    assert v["reading_3"] == "shown but for A1 [R2/42@10ms: replicated; R2/43@10ms: not replicated]"
    low = one(run.tabs, "walk", "verdicts", set="F25F9cl", condition="walk/v0.4_mu0.54", delay=J)
    assert low["reading_2"] == ("shown [R2/42@10ms: replication cut by the pre-registered order; "
                                "R2/43@10ms: replication cut by the pre-registered order]")
    # only the candidate is replicated; a control's passing record gets no line; a block is never summed
    reps = part(run.tabs, "replication", "replication")["lines"]
    assert {x["set"] for x in reps} == {"F25F9cl"}
    assert not any("40" in str(v) for x in reps for k, v in x.items() if k[:2] in ("a_", "b_") and "path" not in k)
    for name in ("walk", "ladder", "rest"):  # the headline tables read the headline block only
        assert not any("seeds5042" in str(x.get("path")) for x in part(run.tabs, name, "counts")["lines"])


def test_replication_words_are_the_runners(run):
    for row in ("R2/42@10ms", "R2/43@10ms"):
        for key in ("walk/v0.4", "walk/v0.4_mu0.54"):
            for which in ("success", "but_A1"):
                assert (barreport.replication_word(run.doc, "F25F9cl", BODY, row, key, which)
                        == bar.replication_word(run.out, "F25F9cl", BODY, row, key, which)), (row, key, which)
    # a pass that no unit of the ledger covers, in a task without a cut: not run yet
    assert barreport.replication_word(run.doc, "F25F9cl", BODY, "R2/42@10ms", "turn/q90_left", "but_A1") \
        == bar.REPLICATION_PENDING[0]


def test_a_pass_whose_replication_is_in_no_unit_of_the_order_reads_the_same_whatever_the_ledger(run, tmp_path):
    """H6 replicates neither task 4's tier 6 (the reported ladders at 10 ms) nor task 3's tier 9 (the speed ladder at
    4 and 6 ms): the word beside such a pass is one fixed phrase and never moves with an unrelated cut."""
    study = tmp_path / "study"
    out = study / "runs" / "x"
    cl, rows = "F25F9cl", ("R2/42@10ms", "R2/43@10ms")
    for row in rows:
        put(study, out, cl, row, "turn/t90_0.5_left")  # task 4, tier 6: both rows pass at the judged delay
        put(study, out, cl, row, "nudge/forward")  # task 4, tier 1: its replication is tier 2's
        put(study, out, "D0", row, "turn/t90_0.5_left")  # a set outside the round: the order says nothing of it
    for row in ("R2/42@4ms", "R2/43@4ms"):
        put(study, out, cl, row, "walk/v0.5")  # task 3, tier 9
    assert barreport.replication_in_order(cl, BODY, rows[0], "turn/t90_0.5_left") is False
    assert barreport.replication_in_order(cl, BODY, "R2/42@4ms", "walk/v0.5") is False
    assert barreport.replication_in_order(cl, BODY, "R2/42@2ms", "walk/v0.5") is True  # the pull request's row
    assert barreport.replication_in_order(cl, BODY, "R2/42@0ms", "walk/v0.5") is True
    assert barreport.replication_in_order(cl, BODY, rows[0], "nudge/forward") is True
    assert barreport.replication_in_order(cl, BODY, "R0@10ms", "nudge/forward") is False
    assert barreport.replication_in_order("D0", BODY, rows[0], "nudge/forward") is None
    assert barreport.replication_in_order(cl, "60-N-T-10", rows[0], "nudge/forward") is None
    unit = lambda name, paths: dict(unit=name, estimate_core_h=0.2, spent_before_core_h=0.0,  # noqa: E731
                                    started_at="2026-11-15T08:00:00Z", items={"episodes": 600.0}, detail=dict(paths=paths))
    started = unit("T4.1-F25F9cl-10ms", [run_path(cl, r, "nudge/forward") for r in rows])
    ladders = unit("T4.6-F25F9cl-10ms", [run_path(cl, r, "turn/t90_0.5_left") for r in rows])
    cut = dict(unit="T4.7-F25F9cl-0ms-seeds5042", estimate_core_h=0.3, spent_before_core_h=11.9,
               at="2026-11-15T20:00:00Z", reason="does not fit", label=barreport.CUT,
               detail=dict(paths=[run_path(cl, "R2/42@0ms", "stop/two_step", R)]))
    ledgers = dict(none=None, no_cut={"4": dict(cap_core_h=12.0, units=[started, ladders], cut=[])},
                   unrelated_cut={"4": dict(cap_core_h=12.0, units=[started, ladders], cut=[cut]),
                                  "3": dict(cap_core_h=16.0, units=[], cut=[dict(cut, unit="T3.9-F25F9cl-4ms", detail={})])})
    seen: dict = {}
    for name, tasks in ledgers.items():
        if tasks is not None:
            s2io.write_json(out / "budget.json", dict(schema="bar-budget-1", tasks=tasks))
        doc = barreport.load(out, study=study)
        tabs = barreport.tables(doc)
        for which in ("success", "but_A1"):
            for row in rows:
                assert barreport.replication_word(doc, cl, BODY, row, "turn/t90_0.5_left", which) == barreport.NO_REPLICATION_UNIT
            assert barreport.replication_word(doc, cl, BODY, "R2/42@4ms", "walk/v0.5", which) == barreport.NO_REPLICATION_UNIT
            # a set outside the round keeps the runner's word: a cut only where the ledger lists the record as cut
            assert barreport.replication_word(doc, "D0", BODY, rows[0], "turn/t90_0.5_left", which) \
                == bar.replication_word(out, "D0", BODY, rows[0], "turn/t90_0.5_left", which) == bar.REPLICATION_PENDING[0]
        v = one(tabs, "rest", "verdicts", set=cl, condition="turn/t90_0.5_left")
        assert v["reading_2"] == "shown [" + "; ".join(f"{r}: {barreport.NO_REPLICATION_UNIT}" for r in rows) + "]"
        five = one(tabs, "replication", "replication", set=cl, condition="walk/v0.5", row="R2/42@4ms")
        assert five["words"] == [f"{n}: {barreport.NO_REPLICATION_UNIT}" for _, _, n in barreport.READINGS]
        seen[name] = (v["reading_2"], v["reading_3"], five["words"],
                      one(tabs, "rest", "verdicts", set=cl, condition="nudge/forward")["reading_2"])
    assert seen["none"][:3] == seen["no_cut"][:3] == seen["unrelated_cut"][:3]
    # a pass whose replication the order holds still follows the ledger: not run yet, then cut with the task's cut
    pending, was_cut = (f"shown [{'; '.join(f'{r}: {w}' for r in rows)}]"
                        for w in (bar.REPLICATION_PENDING[0], bar.REPLICATION_WORDS[2]))
    assert seen["none"][3] == seen["no_cut"][3] == pending and seen["unrelated_cut"][3] == was_cut
    assert barreport.NO_REPLICATION_UNIT.startswith(bar.REPLICATION_WORDS[2] + " (")  # H4's word, and why
    notes = " ".join(part(run.tabs, "rest", "verdicts")["notes"])
    assert barreport.NO_REPLICATION_UNIT in notes and all(t in notes for t in barreport.NO_REPLICATION_TIERS)


def test_verdict_words_equal_the_runners_summary(run, tmp_path):
    out = tmp_path / "run"
    shutil.copytree(run.out, out)
    summary = bar.summarize(out, "F25F9cl", BODY, H)
    checked = 0
    for ckey, by_ctrl in summary["verdicts"].items():
        if ckey == "stop/two_step":  # the summary reads the later record, the tables the registered one
            continue
        for name, cell in by_ctrl.items():
            ms = bar.delay_ms("R2/42" + name[len("R2"):])
            mine = barreport.verdict_of(run.doc, "F25F9cl", BODY, ckey, ms)
            for rkey in ("reading_2", "reading_3"):
                assert mine[rkey]["verdict"] == cell[rkey]["verdict"], (ckey, name, rkey)
                if cell[rkey].get("replication") and ckey.startswith("walk/v0.4"):
                    assert mine[rkey]["replication"] == cell[rkey]["replication"]
                checked += 1
    assert checked >= 20


# ------------------------------------------------------------------------------------------------
# Labels
# ------------------------------------------------------------------------------------------------
def test_every_line_of_the_candidate_carries_its_margin_label(run):
    n = 0
    for name, p, x in every_line(run.tabs):
        if x.get("set") == "F25F9cl":
            assert barreport.MARGIN_LABEL in x["labels"], (name, p["key"], x)
            n += 1
        elif "set" in x:
            assert barreport.MARGIN_LABEL not in x["labels"]
    assert n > 60
    seeds = [p for p in run.tabs["jps"]["parts"] if p["key"].startswith("seeds F25F9cl")]
    assert seeds and all(p["notes"] == [barreport.MARGIN_LABEL] for p in seeds)
    assert barreport.labels_of("F25F9cl.2") == [barreport.MARGIN_LABEL, "a version of set F25F9cl"]
    assert barreport.labels_of("V0") == []


def test_a_change_after_a_result_a_control_and_stage_1s_command_are_labelled(run):
    rec = copy.deepcopy(run.doc["V0"][BODY][H]["R0@10ms"]["walk/v0.4"])
    rec["change_after_result"] = dict(label="change after a result", freezes=[dict(freeze=1, reason="a new turn", files=[])])
    assert f"{barreport.CHANGE}: freeze 1: a new turn" in barreport.labels_of("V0", rec)
    stop = one(run.tabs, "rest", "counts", set="F25F9cl", condition="stop/one_step", row="R2/42@10ms")
    assert f"command read by {STAGE1_COMMAND}" in stop["labels"] and stop["item"] == "stop, Stage 1's one-step stop"
    m = one(run.tabs, "rest", "bar_stop", set="F25F9cl", condition="stop/one_step", row="R2/42@10ms")
    assert m["convention"] == STAGE1_COMMAND and m["touchdowns"] == {"1": 20} and m["touchdowns_limit"] == 2
    c = one(run.tabs, "controls", "controls", row="R2/42@10ms")
    assert c["control"] == "omega at Stage 1's 0.03 H crouch" and c["verdict"] == "no verdict (a control)"
    assert (c["crouch"], c["ref_crouch"]) == (0.03, 0.05) and c["omega"] != c["ref_omega"]
    assert c["up"] == c["ref_up"] == dict(k=20, n=20) and "controls/omega_s1/" in c["path"] and "controls" not in c["ref_path"]
    assert "control: omega at Stage 1's 0.03 H crouch" in c["labels"]
    absent = one(run.tabs, "controls", "controls", row="R2/42@0ms")  # task 3 cut a unit; the control was not started
    assert absent["status"] == barreport.CUT and absent["up"] is None


def test_a_lead_in_that_is_not_the_rules_is_labelled_from_the_record_and_never_assumed(run, tmp_path):
    """Rule A8 as amended (P34): a set whose ``set.json`` says ``lead_in: "start_v0.4"`` carries that label on every
    line of every table: read from the record where the line has one, from the set's ``set.json`` where it has none."""
    of_rec, of_set = START_V04.format(barreport.LEAD_IN_FROM_RECORD), START_V04.format(barreport.LEAD_IN_FROM_SET)
    assert run.doc[barreport.META]["lead_in"] == {s_: {BODY: LEAD_IN.get(s_, "start")} for s_ in ("F25F9", "F25F9cl", "V0")}
    seen = Counter()
    for name, p, x in every_line(run.tabs):
        lead = [t for t in x.get("labels") or [] if "lead-in" in t]
        if x.get("set") == "F25F9":
            assert lead in ([of_rec], [of_set]), (name, p["key"], x)
            cites = any("R0@10ms/walk/v0.4.json" in t for t in strings(x))  # the set's one record
            assert (lead == [of_rec]) == (cites or p["key"] == "tally"), (name, p["key"], x)  # the tally reads it too
            seen[name] += 1
        else:
            assert not lead, (name, p["key"], x)
    assert set(seen) == {"walk", "ladder", "glance"} and seen["glance"] == 16 + 1  # its 16 conditions and their tally
    jps = {x["line"]: x for x in part(run.tabs, "jps", "jps")["lines"]}
    gap = barreport.gap_note(run.doc[barreport.META], "F25F9", BODY, "walk/v0.5")  # no change reaches the rung
    assert gap.startswith("; the set holds no reference for it: no transition into v0.5 (") and "`zmp_error`" in gap
    assert jps[94]["record"] == [f"F25F9 walk/v0.5 at 0 ms, {r}@0ms: {barreport.CUT}{gap} [{of_set}]"
                                 for r in barreport.R2_ROWS]
    assert not any("lead-in" in t for n, x in jps.items() if n != 94 for t in x["record"])
    lead = run.tabs["checks"]["lead_in"]
    assert lead == dict(rule="start", records_that_differ=[], records_without_the_field=[], sets={
        f"F25F9/{BODY}": "start_v0.4", f"F25F9cl/{BODY}": "start", f"V0/{BODY}": "start"})
    top = run.md.split("## How to read the tables")[0]
    assert f"`F25F9/{BODY}` `start_v0.4`; `F25F9cl/{BODY}` `start`; `V0/{BODY}` `start`." in top
    assert "Records whose own `lead_in` differs from their set's: 0" in top
    assert any("lead-in `<name>`" in c and "never assumed" in c for c in run.tabs["conventions"])
    # the record's own word stands: one that says nothing, and one that says another lead-in than its set's
    study = tmp_path / "study"
    out = study / "runs" / "x"
    for row in ("R2/42@10ms", "R2/43@10ms", "R0@10ms"):  # the rung no change reaches: the runner's "not run" records
        put(study, out, "F25F9", row, "walk/v0.5", not_run=why_not("F25F9", "walk/v0.5"))
    quiet = bar.record_path(out, "F25F9", BODY, H, "R0@10ms", SimpleNamespace(behavior="walk", name="v0.5"))
    rec = json.loads(quiet.read_text())  # the runner writes the lead-in into every record of such a set: taken out
    assert rec.pop("lead_in") == "start_v0.4" and "lead_in" not in (rec.get("set_info") or {})
    quiet.write_text(json.dumps(rec))
    put(study, out, "V0", "R0@10ms", "walk/v0.4", extra=dict(lead_in="start_v0.4"))
    put(study, out, "V0", "R2/42@10ms", "walk/v0.4")
    tabs = barreport.tables(barreport.load(out, study=study))
    assert of_rec in one(tabs, "ladder", "counts", set="F25F9", row="R2/42@10ms")["labels"]
    silent = one(tabs, "ladder", "counts", set="F25F9", row="R0@10ms")
    assert [t for t in silent["labels"] if "lead-in" in t] == [
        START_V04.format(barreport.LEAD_IN_FROM_SET + "; the record holds no `lead_in`")]
    assert one(tabs, "ladder", "verdicts", set="F25F9", delay=J)["labels"] == [of_rec]
    assert one(tabs, "ladder", "highest", set="F25F9", grip="mu 0.9", delay=J)["labels"] == [of_rec]
    other = one(tabs, "walk", "counts", set="V0", row="R0@10ms", condition="walk/v0.4")
    assert other["labels"] == [of_rec, "the record's lead-in is `start_v0.4`; the set's `set.json` says `start`"]
    assert one(tabs, "walk", "counts", set="V0", row="R2/42@10ms", condition="walk/v0.4")["labels"] == []
    assert tabs["checks"]["lead_in"]["records_that_differ"] == [dict(path=other["path"], record="start_v0.4", set="start")]
    assert tabs["checks"]["lead_in"]["records_without_the_field"] == [dict(path=silent["path"], set="start_v0.4")]
    md = barreport.markdown(tabs)
    assert f"- `{other['path']}`: the record says `start_v0.4`, the set's `set.json` `start`" in md
    assert f"- `{silent['path']}` (the set's `set.json` says `start_v0.4`)" in md
    # nothing is assumed of a set by its name: with no `lead_in` in its set.json only the records' own word is printed
    sj = study / "stage2" / "sets" / "F25F9" / BODY / "set.json"
    sd = json.loads(sj.read_text())
    del sd["lead_in"]
    sj.write_text(json.dumps(sd))
    doc = barreport.load(out, study=study)
    tabs = barreport.tables(doc)
    assert "F25F9" not in doc[barreport.META]["lead_in"] and f"F25F9/{BODY}" not in tabs["checks"]["lead_in"]["sets"]
    # (the runner refuses such documents, a set that holds `start_v0.4` and no nominal walk: the checks say so)
    assert tabs["checks"]["sets_refused"][f"F25F9/{BODY}"].startswith(barreport.REFUSED + f": stage2/sets/F25F9/{BODY}: ")
    assert not any("lead-in" in t for t in one(tabs, "ladder", "counts", set="F25F9", row="R0@10ms")["labels"])
    assert not any("lead-in" in t for x in lines(tabs, "walk", "counts", set="F25F9") for t in x["labels"])
    assert of_rec in one(tabs, "ladder", "counts", set="F25F9", row="R2/42@10ms")["labels"]
    assert tabs["checks"]["lead_in"]["records_without_the_field"] == []
    # where a record holds it
    assert barreport.lead_in_of(None) is None and barreport.lead_in_of({}) is None
    assert barreport.lead_in_of(dict(set_info=dict(lead_in="start_v0.4"))) == "start_v0.4"
    assert barreport.lead_in_of(dict(lead_in="start", set_info=dict(lead_in="start_v0.4"))) == "start"
    assert barreport.set_labels("F25F9") == [] and barreport.labels_of("F25F9", dict(lead_in="start")) == []
    assert barreport.lead_in_label("x", "as the record says") == f"lead-in `x` (as the record says; {barreport.LEAD_IN_CITE})"


def test_plain_words_for_items_and_grips(run):
    conds = barreport._conds()
    assert len(conds) == 41 and barreport._group("@gated") == [c.key for c in bar.conditions(set_docs("V0")) if c.gated]
    assert list(conds) == [c.key for c in bar.conditions(set_docs("V0"), every=True)]
    words = {k: (barreport.item_of(d), barreport.grip_of(d)) for k, d in conds.items()}
    assert words["walk/v0.4"] == ("walk at 0.4 m/s", "mu 0.9")
    assert words["walk/nominal_mu0.54"] == ("walk at the nominal speed", "mu 0.54")
    assert words["turn/q90_left"] == ("turn left, 90 deg at pi/4 rad/s (the quick turn)", "mu 0.9")
    assert words["turn/t90_0.35_right"][0] == "turn right, 90 deg at 0.35 rad/s"
    assert words["nudge/backward_x0.375"][0] == "nudge backward, 0.375 x the push bound"
    assert words["stop/two_step_mu0.54"] == ("stop, two touchdowns", "mu 0.54")


# ------------------------------------------------------------------------------------------------
# Checks at the top
# ------------------------------------------------------------------------------------------------
def test_a_record_of_other_sources_is_listed_at_the_top_and_still_tabulated_flagged(run):
    c = run.tabs["checks"]
    assert c["today_sources"] == TODAY and len(c["other_sources"]) == 1
    o = c["other_sources"][0]
    assert o["path"].endswith("V0/75-M-H-10/seeds3042/eval/R2/43@10ms/walk/v0.4.json") and o["differs"] == ["bar"]
    x = one(run.tabs, "walk", "counts", set="V0", condition="walk/v0.4", delay=J, row="R2/43@10ms")
    assert x["up"] == dict(k=0, n=20) and f"{barreport.OTHER_SOURCES} (bar)" in x["labels"]
    clean = one(run.tabs, "walk", "counts", set="V0", condition="walk/v0.4", delay=J, row="R2/42@10ms")
    assert not any(s.startswith(barreport.OTHER_SOURCES) for s in clean["labels"])
    top = run.md.split("## How to read the tables")[0]
    assert "Records whose `key.sources` differ from today's: 1" in top and o["path"] in top


def test_inputs_are_checked_against_the_sets_latest_freeze(run, tmp_path):
    c = run.tabs["checks"]
    assert len(c["not_in_latest_freeze"]) == 1 and c["not_checked"] == []
    bad = c["not_in_latest_freeze"][0]
    assert bad["path"].endswith("V0/75-M-H-10/seeds3042/eval/R0@10ms/walk/v0.4.json")
    assert bad["differences"] == ["refs/v0.4.npz: another sha256"] and bad["set_dir"] == "stage2/sets/V0/75-M-H-10"
    x = one(run.tabs, "walk", "counts", set="V0", condition="walk/v0.4", delay=J, row="R0@10ms")
    assert barreport.NOT_IN_MANIFEST in x["labels"]
    assert c["freezes"]["stage2/sets/V0/75-M-H-10"]["index"] == 0 and "entries" not in c["freezes"]["stage2/sets/V0/75-M-H-10"]
    # a later freeze that lists the file under another hash, an entry it dropped, a set that is gone
    study = tmp_path / "study"
    shutil.copytree(run.study, study)
    sj = study / "stage2" / "sets" / "F25F9cl" / BODY / "set.json"
    sd = json.loads(sj.read_text())
    manifest = copy.deepcopy(sd["freezes"][0]["manifest"])
    manifest["v0.45"]["entry_sha256"] = sha("another entry")
    del manifest["v0.5"]
    sd["freezes"].append(dict(index=1, manifest=manifest, manifest_sha256=s2io.sha256_json(manifest)))
    sj.write_text(json.dumps(sd))
    (study / "stage2" / "sets" / "F25F9" / BODY / "set.json").unlink()
    c2 = barreport.tables(barreport.load(study / "stage2" / "runs" / "synthetic", study=study))["checks"]
    why = {tuple(x["differences"]) for x in c2["not_in_latest_freeze"]}
    assert ("refs/v0.45.npz: another entry_sha256",) in why and ("refs/v0.5.npz: not listed",) in why
    assert [Path(x["path"]).parts[3] for x in c2["not_checked"]] == ["F25F9"]
    assert c2["freezes"]["stage2/sets/F25F9cl/75-M-H-10"]["index"] == 1
    assert c2["freezes"]["stage2/sets/F25F9/75-M-H-10"] == dict(found=False)


def test_errors_unjudged_counts_and_core_hours_are_listed(run):
    c = run.tabs["checks"]
    assert [Path(x["path"]).parts[3] for x in c["errors_or_unjudged"]] == ["F25F9"]
    assert c["errors_or_unjudged"][0]["errors"] == 1 and c["errors_or_unjudged"][0]["judged"] is False
    x = one(run.tabs, "walk", "counts", set="F25F9", condition="walk/v0.4", delay=J, row="R0@10ms")
    assert x["r2_pass"] == "not judged (1 episode failed to run)" and x["episodes"] == 20 and x["up"] == dict(k=19, n=20)
    assert c["records"]["by_set"]["F25F9cl"]["not_run"] == 4 and c["records"]["not_run"] == 4
    b = c["budget"]["3"]
    assert b["cap_core_h"] == 16.0 == b["registered_cap_core_h"] == bar.BUDGET_CORE_H[3]
    assert b["spent_core_h"] == pytest.approx(2.0) and [u["unit"] for u in b["units"]] == ["T3.1", "T3.2a"]
    assert [u["unit"] for u in b["cut"]] == ["T3.2b"] and "4" not in c["budget"]
    top = run.md.split("## How to read the tables")[0]
    assert "Task 3: 2.000 core-hours spent of 16" in top and "| T3.2b | 0.300 | 15.900 |" in top
    assert "2 units started, 1 cut; units started after the first cut (2026-11-15T20:00:00Z): none" in top
    assert "`COST.md`: none" in top and b["started_after_the_first_cut"] == []
    # a unit that started after the first cut is named: the cut units would not be the order's tail
    doc = copy.deepcopy(run.doc)
    doc[barreport.META]["budget"]["tasks"]["3"]["units"].append(dict(
        unit="T3.9", estimate_core_h=0.1, spent_before_core_h=2.0, started_at="2026-11-15T21:00:00Z", items={}, detail={}))
    assert barreport.checks(doc)["budget"]["3"]["started_after_the_first_cut"] == ["T3.9"]
    assert "units started after the first cut (2026-11-15T20:00:00Z): T3.9; sums of the ledger" in barreport.markdown(
        barreport.tables(doc))
    # the budget section agrees with budget.json: each unit's own sum and the sum before it recompute, or are named
    assert b["ledger_differences"] == [] and [u["ledger_used_core_h"] for u in b["units"]] == [0.5, 1.5]
    assert [u["used_core_h"] for u in b["units"]] == [0.5, 1.5] and "from the units before it): 0." in top
    doc = copy.deepcopy(run.doc)
    units = doc[barreport.META]["budget"]["tasks"]["3"]["units"]
    units[0]["used_core_h"], units[1]["spent_before_core_h"] = 0.75, 0.25
    off = barreport.checks(doc)["budget"]["3"]
    assert off["ledger_differences"] == [
        "T3.1: `used_core_h` is 0.750 in the ledger, 0.500 from its items",
        "T3.2a: `spent_before_core_h` is 0.250 in the ledger, 0.500 from the units before it"]
    assert off["spent_core_h"] == pytest.approx(2.0)  # the tables' sum is the episodes' wall seconds, not the ledger's
    assert "from the units before it): 2\n  - T3.1: `used_core_h` is 0.750 in the ledger" in barreport.markdown(barreport.tables(doc))


def test_counts_are_the_records_own_and_recompute_from_their_episodes(run, tmp_path):
    assert run.tabs["checks"]["counts_not_recomputed"] == []
    assert "do not recompute from their own episodes (`bar.counts`): 0" in run.md
    for rec in barreport._records(run.doc):  # every count a table prints is the record's, equal to bar.counts
        assert barreport.counts_differences(rec) == []
        d = barreport.digest(rec)
        if rec["status"] == "done":
            c2, c3 = bar.counts(rec["episodes"], rec["seeds"]), bar.counts(rec["episodes"], rec["seeds"], "but_A1")
            assert (d["up"]["k"], d["r2"]["k"], d["r3"]["k"]) == (c2["survived"], c2["succeeded"], c3["succeeded"])
            assert (d["up_lcb"], d["r2_lcb"], d["r3_lcb"]) == (c2["survival_lcb"], c2["success_lcb"], c3["success_lcb"])
    assert run.tabs["checks"]["second_block_outside_the_rule"] == []
    # a record whose counts were changed by hand is named at the top
    study = tmp_path / "study"
    out = study / "runs" / "x"
    put(study, out, "V0", "R2/42@10ms", "walk/v0.4", per_seed=lambda s: dict(fell=s < 3046))
    path = bar.record_path(out, "V0", BODY, H, "R2/42@10ms", SimpleNamespace(behavior="walk", name="v0.4"))
    rec = json.loads(path.read_text())
    rec["counts"].update(survived=20, succeeded=20, pass_=True)
    path.write_text(json.dumps(rec))
    put(study, out, "V0", "R0@10ms", "walk/v0.4", block=R)  # no second block for a control set or an R0 row
    put(study, out, "V0", "R2/42@10ms", "walk/v0.4", block=R)  # ... nor under an R2 row of a control (it passes)
    tabs = barreport.tables(barreport.load(out, study=study))
    c = tabs["checks"]
    assert [x["why"] for x in c["second_block_outside_the_rule"]] == [
        ["an R0 row", "a control set of the round", "no replication_of field"],
        ["a control set of the round", "no replication_of field"]]
    # table 5 lists them beside the first block and gives them no replication word
    for row, why in (("R0@10ms", "an R0 row, a control set of the round, no replication_of field"),
                     ("R2/42@10ms", "a control set of the round, no replication_of field")):
        x = one(tabs, "replication", "replication", set="V0", row=row)
        assert x["words"] == [f"{barreport.OUTSIDE_RULE}: {why}"] and x["words"][0] in x["labels"]
        assert x["triggered_by"] == ["none: the record is " + barreport.OUTSIDE_RULE] and x["b_up"] == dict(k=20, n=20)
    assert not [w for x in part(tabs, "replication", "replication")["lines"] for w in x["words"]
                if w.endswith(bar.REPLICATION_WORDS[:2])]
    assert barreport.outside_rule(run.doc["F25F9cl"][BODY][R]["R2/42@10ms"]["walk/v0.4"]) == []
    assert barreport.outside_rule(run.doc["V0"][BODY][H]["R0@10ms"]["walk/v0.4"]) == []  # another block
    assert len(c["counts_not_recomputed"]) == 1 and c["counts_not_recomputed"][0]["differences"] == [
        "reading 2 (every limit): survived is 20 in the record, 16 from its episodes",
        "reading 2 (every limit): succeeded is 20 in the record, 16 from its episodes",
        "reading 2 (every limit): pass_ is True in the record, False from its episodes"]
    x = one(tabs, "walk", "counts", set="V0", condition="walk/v0.4", row="R2/42@10ms")
    assert x["r2_pass"] == "passes" and barreport.MISCOUNTED in x["labels"]  # the record's own word, flagged
    v = one(tabs, "walk", "verdicts", set="V0", condition="walk/v0.4", delay=J)
    assert v["labels"] == [f"R2/42@10ms: {barreport.MISCOUNTED}"]
    g = one(tabs, "glance", "glance", set="V0", condition="walk/v0.4")
    assert f"R2/42@10ms: {barreport.MISCOUNTED}" in g["labels"]
    assert not any(barreport.MISCOUNTED in str(x_["labels"]) for _, _, x_ in every_line(run.tabs) if "labels" in x_)
    # reading 2 of a turn frozen with a1_over keeps its pass under pass_if_read: that is no difference
    a1 = run.doc["F25F9cl"][BODY][H]["R2/42@10ms"]["turn/q90_left"]
    assert a1["counts"]["pass_"] is None and a1["counts"]["pass_if_read"] is False and barreport.counts_differences(a1) == []


def test_check_episodes_are_counted_and_held_to_rule_r4(run, tmp_path):
    assert run.tabs["checks"]["dev_episodes"] is None and "no `dev_episodes.json`" in run.md
    out = tmp_path / "study" / "runs" / "x"
    put(tmp_path / "study", out, "V0", "R0@10ms", "walk/v0.4")
    entry = lambda s, row, seed=3042: dict(set=s, body=BODY, block=H, row=row, condition="walk/v0.4", seed=seed)  # noqa: E731
    s2io.write_json(out / "dev_episodes.json", dict(
        episodes=[entry("V0", "R0@10ms"), entry("V0", "R0@0ms"), entry("V0", "R2/42@10ms"), entry("F25F9cl", "R0@10ms"),
                  entry("V0", "R0@10ms", 3043), entry("D0", "R2/42@10ms")],
        scratch_sets=dict(sets=dict(T0="a fixture", D0="a scratch set"))))
    tabs = barreport.tables(barreport.load(out, study=tmp_path / "study"))
    dev = tabs["checks"]["dev_episodes"]
    assert dev["listed"] == 6 and dev["scratch_sets"] == ["D0", "T0"]
    assert [(x["set"], x["row"], x["seed"]) for x in dev["outside_the_rule"]] == [
        ("V0", "R2/42@10ms", 3042), ("F25F9cl", "R0@10ms", 3042), ("V0", "R0@10ms", 3043)]
    md = barreport.markdown(tabs)
    assert "(smoke; none is a record): 6; of these outside rule R4" in md and "set `F25F9cl`, row `R0@10ms`" in md


def test_cost_rates_are_printed_with_how_they_were_measured(run, tmp_path):
    out = tmp_path / "study" / "runs" / "x"
    put(tmp_path / "study", out, "V0", "R0@10ms", "walk/v0.4")
    cost = dict(schema=bar.COST_SCHEMA, rates=dict(R2=dict(
        wall_per_sim_s=0.4567, row="R2/42@0ms", full_length=True, episodes=[dict(seed=3042), dict(seed=3043)],
        warm_up=dict(seed=3042, counted=False))),
        measured_on=dict(condition="walk/brisk", body=BODY, seeds=[3042, 3043]),
        load=dict(before=dict(load_average=[1.5, 2.0, 2.5])))
    (out / "COST.md").write_text("# Cost\n\n```json\n" + json.dumps(cost, indent=1) + "\n```\n")
    md = barreport.markdown(barreport.tables(barreport.load(out, study=tmp_path / "study")))
    assert ("R2 0.457 (row `R2/42@0ms`, 2 episodes, run their full length, after a warm-up episode that is not "
            "counted)") in md
    assert "Stage 1's own condition `walk/brisk`" in md and "load averages before the measurement: 1.5, 2, 2.5" in md
    assert "Core-hours: no `budget.json`" in md


# ------------------------------------------------------------------------------------------------
# Measured values and limits
# ------------------------------------------------------------------------------------------------
def test_a_measured_column_never_holds_a_target(run):
    # a record whose measured speed differs from its target shows the measured one
    x = one(run.tabs, "ladder", "bar_walk", set="V0", condition="walk/v0.35", row="R2/42@0ms")
    assert x["speed"] == dict(median=0.3661, n=20) and x["along"]["median"] == pytest.approx(3.661)
    assert x["along_limit"] == 3.5 and x["lateral_limit"] == 0.25 and x["toe_limit"] == pytest.approx(20.0741)
    assert x["toe"]["median"] == pytest.approx(23.63) and x["toe_first"]["median"] == pytest.approx(17.1)
    assert x["toe_stage1"]["median"] == pytest.approx(22.8)
    assert "| 0.366 (20) | 3.661 (20) | 3.5 |" in run.md
    targets = {10.0 * v for v in setio.SPEEDS} | set(setio.SPEEDS)
    kinds = set()
    for name, p, line in every_line(run.tabs):
        for c in p["columns"]:
            kinds.add(c["kind"])
            v = line.get(c["key"])
            if c["kind"] == "measured":  # by construction of the synthetic records no measurement equals a target
                assert "measured" in c["title"] and "imit" not in c["title"].split(":")[0], c["title"]
                value = v.get("median", v.get("largest")) if isinstance(v, dict) else v
                assert value is None or value not in targets, (name, p["key"], c["key"], v)
                assert v is None or isinstance(v, (float, dict)), (c["key"], v)
            elif c["kind"] in ("limit", "defined"):
                assert "measured" not in c["title"] and not isinstance(v, dict), (c["title"], v)
                assert c["title"].startswith("Limit") or "as defined" in c["title"] or "as run" in c["title"] \
                    or "crouch" in c["title"], c["title"]
    assert {"measured", "limit", "defined", "of", "claim", "reference"} <= kinds
    # the judged toe clearance stands beside its limit, the reference's own reading (neither measured nor a limit),
    # the reading as first written and Stage 1's, both reported
    titles = [c["title"] for c in part(run.tabs, "walk", "bar_walk")["columns"]]
    i = next(i for i, t in enumerate(titles) if t.startswith("Toe clearance, the judged term"))
    assert titles[i + 1].startswith("Limit: at least (mm")
    assert titles[i + 2].startswith("Toe clearance of the reference itself")
    assert "as first written" in titles[i + 3] and "never judged" in titles[i + 3]
    assert "Stage 1's reading" in titles[i + 4] and "never judged" in titles[i + 4]
    assert [c["title"] for c in part(run.tabs, "rest", "bar_turn")["columns"]][-6:-1] == titles[i:i + 5]


def test_toe_clearance_names_the_middle_half_and_a_median_just_under_its_limit_shows(run, tmp_path):
    """C3 as amended: both readings are over a middle half, and the headers say which; a median 0.01 mm under the
    limit is not printed as the limit or above it."""
    cols = {c["key"]: c for c in barreport.TOE_COLUMNS}
    assert "read over the middle half of the planned swing from the own lift-off" in cols["toe"]["title"]
    assert "over the middle half of the swing's own duration" in cols["toe_first"]["title"]
    assert (cols["toe_limit"]["kind"], cols["toe_limit"]["unit"], cols["toe"]["unit"]) == ("limit", "mm", "mm")
    assert barreport.DECIMALS["mm"] == 2 and "mm with 2 decimals" in " ".join(run.tabs["conventions"])
    study = tmp_path / "study"
    out = study / "runs" / "x"
    put(study, out, "V0", "R0@10ms", "walk/v0.4")
    path = bar.record_path(out, "V0", BODY, H, "R0@10ms", SimpleNamespace(behavior="walk", name="v0.4"))
    rec = json.loads(path.read_text())
    for e in rec["episodes"]:
        e["measures"]["bar"]["toe_clearance_own_swing_m"] = 0.02006  # under the limit of 20.0741 mm
    path.write_text(json.dumps(rec))
    tabs = barreport.tables(barreport.load(out, study=study))
    x = one(tabs, "walk", "bar_walk", set="V0", row="R0@10ms")
    assert x["toe"] == dict(median=pytest.approx(20.06), n=20) and x["toe_limit"] == pytest.approx(20.0741)
    assert "| 20.06 (20) | 20.07 | 25.11 (0.0500 H) | 17.10 (20) | 22.80 (20) |" in barreport.markdown(tabs)
    assert "| 23.63 (20) | 20.07 |" in run.md


def test_the_references_own_toe_reading_stands_beside_the_measured_one(run, tmp_path):
    """STAGE2_75_PLAN.md C3's ruling: wherever the toe term of a walk or a turn is printed, the reference's own
    reading stands beside it, from the kinematic check the set's latest freeze cites (sha256 verified), in a column
    of its own kind: never "measured", never a limit.  A check that is missing or is not the cited file is said, and
    no number is printed for it."""
    col = barreport.REF_TOE_COLUMN
    assert col["kind"] == "reference" and col["title"].endswith("; " + barreport.REFERENCE_KIND)
    assert barreport.REFERENCE_KIND == "reference: played kinematically, no physics"
    assert "measured" not in col["title"] and "imit" not in col["title"] and "no result of the robot" in col["title"]
    assert "refcheck.json" in col["title"] and "`_mu0.54` condition takes its reference's reading" in col["title"]
    n = 0
    for name, p, x in every_line(run.tabs):  # every part that prints the toe term has the column, and no other has
        keys = [c["key"] for c in p["columns"]]
        toe_term = "toe" in keys or "failed_2" in keys
        assert ("ref_toe" in keys) == toe_term, (name, p["key"])
        if not toe_term:
            continue
        assert [c for c in p["columns"] if c["key"] == "ref_toe"] == [col]
        judged = barreport._conds()[x["condition"]]["judge"] in ("bar_walk", "bar_turn")
        assert isinstance(x["ref_toe"], dict) == judged and (x["ref_toe"] is None) == (not judged), (name, p["key"], x)
        n += judged
    assert n >= 40
    for p in (part(run.tabs, "walk", "bar_walk"), part(run.tabs, "rest", "bar_turn")):  # beside measured and limit
        keys = [c["key"] for c in p["columns"]]
        assert keys[keys.index("toe"):keys.index("toe") + 3] == ["toe", "toe_limit", "ref_toe"]
        assert [c["kind"] for c in p["columns"] if c["key"] in ("toe", "toe_limit", "ref_toe")] == [
            "measured", "limit", "reference"]
    # the numbers are the check's own (REF_TOE: made up), in mm and as a share of H; the low grip takes its reference's
    f = one(run.tabs, "walk", "bar_walk", set="F25F9", row="R0@10ms")
    assert f["ref_toe"] == dict(mm=pytest.approx(19.712), H=pytest.approx(0.019712 / H_M))
    assert one(run.tabs, "walk", "fails", set="F25F9", row="R0@10ms")["ref_toe"] == f["ref_toe"]
    low = one(run.tabs, "walk", "bar_walk", set="F25F9cl", condition="walk/v0.4_mu0.54", row="R2/42@10ms")
    high = one(run.tabs, "walk", "bar_walk", set="F25F9cl", condition="walk/v0.4", delay=J, row="R2/42@10ms")
    assert low["ref_toe"] == high["ref_toe"] == dict(mm=pytest.approx(22.2), H=pytest.approx(0.0222 / H_M))
    assert barreport.reference_condition("walk/v0.4_mu0.54") == "walk/v0.4" == barreport.reference_condition("walk/v0.4")
    assert barreport.reference_condition("turn/q90_left_mu0.54") == "turn/q90_left"
    t = one(run.tabs, "rest", "bar_turn", set="F25F9cl", condition="turn/q90_left", row="R2/42@10ms")
    assert t["ref_toe"] == dict(mm=pytest.approx(16.208), H=pytest.approx(0.016208 / H_M))
    assert one(run.tabs, "rest", "fails", set="F25F9cl", condition="stop/two_step", row="R2/43@10ms")["ref_toe"] is None
    assert one(run.tabs, "rest", "fails", set="F25F9cl", condition="nudge/forward", row="R2/42@10ms")["ref_toe"] is None
    assert "| 23.63 (19) | 20.07 | 19.71 (0.0393 H) | 17.10 (19) | 22.80 (19) |" in run.md  # (one episode failed to run)
    assert barreport._cell_text(dict(mm=19.712, H=None), col) == "19.71" and barreport._cell_text(None, col) == "-"
    meta = run.doc[barreport.META]
    assert barreport.reference_toe(meta, "V0", BODY, "walk/v0.4") == dict(mm=pytest.approx(25.111), H=pytest.approx(0.025111 / H_M))
    assert barreport.reference_toe(meta, "V0", BODY, "nudge/left") is None  # the term is not judged for a nudge
    assert barreport.reference_toe(meta, "V0", BODY, "stop/two_step") is None
    unplayed = barreport.reference_toe(meta, "F25F9", BODY, "walk/v0.5_mu0.54")  # the check did not play the rung
    assert unplayed.startswith(f"{barreport.NO_REFERENCE_READING}: `stage2/sets/F25F9/{BODY}/refcheck.json` did not "
                               "play `walk/v0.5` (no transition into v0.5 (")
    assert barreport.reference_toe(meta, "D0", BODY, "walk/v0.4").startswith(barreport.NO_REFERENCE_READING + ": no ")
    assert barreport.reference_toe(None, "V0", BODY, "walk/v0.4").startswith(barreport.NO_REFERENCE_READING)
    # at the top: the file each freeze cites, its sha256 as found, the references under the check's limit
    rcs = run.tabs["checks"]["reference_checks"]
    assert list(rcs) == [f"{s_}/{BODY}" for s_ in ("F25F9", "F25F9cl", "V0")] and all(v["read"] for v in rcs.values())
    for sb, v in rcs.items():
        q = run.study / v["file"]
        assert v["file"] == f"stage2/sets/{sb}/refcheck.json" and v["freeze"] == 0 and v["why"] is None
        assert v["sha256"] == v["found_sha256"] == hashlib.sha256(q.read_bytes()).hexdigest()
        assert run.tabs["checks"]["inputs"]["reference_checks"][sb] == dict(file=v["file"], sha256=v["sha256"])
    assert list(rcs[f"F25F9/{BODY}"]["under_its_limit"]) == ["walk/v0.4"] and rcs[f"V0/{BODY}"]["under_its_limit"] == {}
    assert list(rcs[f"F25F9cl/{BODY}"]["under_its_limit"]) == ["turn/q90_left"]
    assert set(rcs[f"F25F9/{BODY}"]["not_played"]) == {"turn/t90_1_left", "turn/t90_1_right", "walk/nominal", "walk/v0.3",
                                                       "walk/v0.35", "walk/v0.45", "walk/v0.5"}
    top = run.md.split("## How to read the tables")[0]
    assert "references whose own reading is under the check's limit: `walk/v0.4` 19.71 mm (0.0393 H) against 20.07 mm" in top
    assert f"the reference check of `V0/{BODY}`: `stage2/sets/V0/{BODY}/refcheck.json` `{rcs[f'V0/{BODY}']['sha256']}`" in top
    assert any(barreport.REFERENCE_KIND in c and "STAGE2_75_PLAN.md C3" in c for c in run.tabs["conventions"])
    assert "`refcheck.json`" in run.tabs["conventions"][0]
    # a check that is not the cited file, is not there, is not cited or is another set's: said, and no number
    cases = {}
    for case in ("changed", "absent", "not cited", "another set's", "no set.json"):
        study = tmp_path / case.replace(" ", "_").replace("'", "")
        shutil.copytree(run.study, study)
        d = study / "stage2" / "sets" / "F25F9" / BODY
        if case == "changed":
            (d / "refcheck.json").write_text((d / "refcheck.json").read_text() + "\n")
        elif case == "absent":
            (d / "refcheck.json").unlink()
        elif case == "not cited":
            sd = json.loads((d / "set.json").read_text())
            del sd["freezes"][-1]["refcheck"]
            (d / "set.json").write_text(json.dumps(sd))
        elif case == "another set's":
            other = (study / "stage2" / "sets" / "V0" / BODY / "refcheck.json").read_text()
            (d / "refcheck.json").write_text(other)
            sd = json.loads((d / "set.json").read_text())
            sd["freezes"][-1]["refcheck"]["sha256"] = sha(other)
            (d / "set.json").write_text(json.dumps(sd))
        else:
            (d / "set.json").unlink()
        tabs = barreport.tables(barreport.load(study / "stage2" / "runs" / "synthetic", study=study))
        cells = [x["ref_toe"] for _, _, x in every_line(tabs) if x.get("set") == "F25F9" and "ref_toe" in x]
        assert len(cells) == 2 and all(isinstance(v, str) for v in cells) and len(set(cells)) == 1, (case, cells)
        assert cells[0].startswith(barreport.NO_REFERENCE_READING + ": ") and not re.search(r"\d\.\d+ H", cells[0])
        v = tabs["checks"]["reference_checks"][f"F25F9/{BODY}"]
        assert v["read"] is False and v["readings"] == 0 and cells[0].endswith(v["why"])
        md = barreport.markdown(tabs)
        assert f"- `F25F9/{BODY}`: not read: {v['why']}; the tables print no reading of the reference for this set" in md
        assert str(tmp_path) not in md
        others = [x["ref_toe"] for _, _, x in every_line(tabs) if x.get("set") == "V0" and x.get("ref_toe") is not None]
        assert others and all(isinstance(o, dict) for o in others)  # the other sets' checks are read as before
        cases[case] = v["why"]
    assert "is not the file the set's latest freeze cites: its sha256 is `" in cases["changed"]
    assert cases["absent"].endswith("which the set's latest freeze cites, was not found")
    assert cases["not cited"] == "the set's latest freeze (freeze 0) cites no reference check"
    assert cases["another set's"].endswith(f"is the check of set 'V0' on body '{BODY}'")
    assert cases["no set.json"] == f"no `set.json` was found under `stage2/sets/F25F9/{BODY}`"


def test_the_frozen_sets_own_documents_read_as_the_plan_states():
    """The three frozen sets' committed documents (JSON only; no array is read): each kinematic check is the file its
    freeze cites and gives the readings the plan's C3 states; F25F9's unreachable rungs print the numbers the plan's
    A8 states; the tables' limit of that gate is the builder's and the sets'; and the synthetic F25F9 of this file
    has the frozen set's shape."""
    from designlab.stage2 import moves  # the builder's constant (nothing is built)

    plan = " ".join((ROOT / barreport.PLAN_FILE).read_text(encoding="utf-8").split())
    stated = {"V0": ("0.0503", "0.0456", "0.0454"), "F25F9cl": ("0.0446", "0.0323", "0.0316"),
              "F25F9": ("0.0393", "0.0320", "0.0313")}
    for name, (walk, left, right) in stated.items():
        d = f"stage2/sets/{name}/{BODY}"
        rc = barreport._refcheck(ROOT, d, name, BODY)
        assert rc["read"] and rc["file"] == f"{d}/refcheck.json" and rc["sha256"] == rc["found_sha256"], name
        got = tuple(f"{rc['toe'][k]['H']:.4f}" for k in ("walk/v0.4", "turn/q90_left", "turn/q90_right"))
        assert got == (walk, left, right), name
        assert all(v["limit_mm"] == pytest.approx(1e3 * 0.04 * H_M, abs=1e-3) for v in rc["toe"].values())
        row = next(x for x in (ROOT / barreport.PLAN_FILE).read_text(encoding="utf-8").split("\n")
                   if x.startswith(f"  | {name} | "))
        assert all(v in row for v in got), (name, row)  # the plan's own table (C3)
        tj = json.loads((ROOT / d / "transitions.json").read_text())
        assert tj["causal_limits"]["zmp_after_first_stride_mm"] == barreport.ZMP_ERROR_LIMIT_MM
        gaps, refused = barreport._set_gaps(ROOT / d, d)
        assert refused is None and gaps, name  # the runner accepts the three sets
    assert barreport.ZMP_ERROR_LIMIT_MM == pytest.approx(1e3 * moves.ZMP_HISTORY_LIMIT_M)
    # A8 as made exact: F25F9's changes into 0.45 and 0.50 m/s end at 4.59 and 5.06 mm against 4.0 mm
    assert "end with a ZMP error of 4.59 and 5.06 mm against A7's 4.0 mm" in plan
    d = f"stage2/sets/F25F9/{BODY}"
    gaps, _ = barreport._set_gaps(ROOT / d, d)
    for rung, change, mm in (("v0.45", "v0.4_to_v0.45", "4.59"), ("v0.5", "v0.45_to_v0.5", "5.06")):
        for grip in ("", "_mu0.54"):
            text = barreport.why_text(gaps[f"walk/{rung}{grip}"])
            assert text.startswith(f"no transition into {rung} (") and f"[transitions/{change}] ({barreport.TRIED_CHANGES}: " in text
            assert (f"(2) `{change}` (ramp of 1 stride): failed `zmp_error`, {mm} mm (the last value of "
                    f"`zmp_error_history_m`) against the limit of 4 mm; {barreport.REST_NUMBERS}: ") in text
            assert f"(1) `nominal_to_{rung}`: not built (the walk nominal is not a rung of the set" in text
    # the synthetic F25F9 has the frozen set's shape: the same entries with or without a file, the same lead-in, the
    # same conditions without a reference, the same form of what was tried
    real = {n: json.loads((ROOT / d / n).read_text()) for n in barreport.SET_FILES}
    mine = set_docs("F25F9")
    for n, key in (("refs.json", "references"), ("transitions.json", "transitions")):
        shape = lambda docs: {k: bool(e.get("file")) and e.get("available") is not False  # noqa: E731
                              for k, e in docs[n][key].items()}
        assert shape(mine) == shape(real), n
    assert mine["set.json"]["lead_in"] == real["set.json"]["lead_in"] == "start_v0.4"
    assert set(mine["set.json"]["not_available"]) == set(real["set.json"]["not_available"])
    assert set(barreport._set_gaps(Path("unused"), "x")[0] or ()) == set()  # (no documents: nothing to say)
    planner = bar.BarPlanner(Path("unused"), mine["refs.json"]["references"], mine["transitions.json"]["transitions"],
                             mine["set.json"])
    assert {c.key for c in bar.conditions(mine, every=True) if planner.missing(c) is not None} == set(gaps)
    for rung in TRIED_ZMP:
        m = mine["transitions.json"]["transitions"][f"nominal_to_{rung}"]["numbers"]
        r = real["transitions.json"]["transitions"][f"nominal_to_{rung}"]["numbers"]
        assert list(m) == list(r) == ["walks", "tried"] and len(m["tried"]) == len(r["tried"]) == 2
        for a, b in zip(m["tried"], r["tried"], strict=True):
            assert ("not_built" in a) == ("not_built" in b) and a["change"] == b["change"] and set(a) <= set(b)
            assert a.get("failed_gates") == b.get("failed_gates") and a.get("ramp_strides") == b.get("ramp_strides")


def test_a_rung_whose_change_failed_prints_what_was_tried_first(run, tmp_path):
    """A8's third step: the cell of a rung no change reaches names each change tried, then "not built" or the gate
    that failed with its reading and its limit, before any other number of the entry."""
    not_built = f"(1) `nominal_to_v0.5`: not built ({NOT_A_RUNG})"
    failed = ("(2) `v0.45_to_v0.5` (ramp of 1 stride): failed `zmp_error`, 4.99 mm (the last value of "
              "`zmp_error_history_m`) against the limit of 4 mm")
    want = (f"no transition into v0.5 ({NO_CHANGE}) [transitions/v0.45_to_v0.5] ({barreport.TRIED_CHANGES}: {not_built}; "
            f"{failed}; {barreport.REST_NUMBERS}: walks.nominal.stage0.within_limits=False, "
            "walks.nominal.stage0.over={limit: clearance swing_clearance, value: 0.0371}, walks.nominal.speed_m_s=0.2431)")
    why = why_not("F25F9", "walk/v0.5")
    assert list(why["numbers"]) == ["walks", "tried"]  # the set's entry: the dropped walk first
    assert barreport.why_text(why) == want and list(why["numbers"]) == ["walks", "tried"]  # (the entry is not reordered)
    assert want.index("`zmp_error`") < want.index("walks.nominal") and barreport.OWN_NUMBERS not in want
    assert "4.32 mm" in barreport.why_text(why_not("F25F9", "walk/v0.45_mu0.54"))
    # a cell with no record (the JPS row, the ladder), and the list at the top
    assert barreport.gap_note(run.doc[barreport.META], "F25F9", BODY, "walk/v0.5") == f"; the set holds no reference for it: {want}"
    x = one(run.tabs, "ladder", "counts", set="F25F9")
    assert x["status"] == f"{barreport.CUT}; the set holds no reference for it: {want}" and x["condition"] == "walk/v0.5"
    assert run.tabs["checks"]["no_reference"][f"F25F9/{BODY}"]["walk/v0.5"] == want
    # the runner's own "not run" record of the rung: its status, the verdict cell, the JPS row and the glance
    study = tmp_path / "study"
    out = study / "runs" / "x"
    for row in ("R2/42@0ms", "R2/43@0ms"):
        rec = put(study, out, "F25F9", row, "walk/v0.5", not_run=why)
        assert rec["status"] == "not run" and list(rec["numbers"]) == ["walks", "tried"]
    tabs = barreport.tables(barreport.load(out, study=study))
    assert one(tabs, "ladder", "counts", set="F25F9", row="R2/42@0ms")["status"] == f"not run: {want}"
    v = one(tabs, "ladder", "verdicts", set="F25F9")
    assert v["reading_2"] == v["reading_3"] == f"not run: {want}"
    jps = {x_["line"]: x_ for x_ in part(tabs, "jps", "jps")["lines"]}[94]["record"]
    assert jps[0].startswith(f"F25F9 walk/v0.5 at 0 ms, R2/42@0ms: not run: {want} [")
    hi = one(tabs, "ladder", "highest", set="F25F9", grip="mu 0.9")
    assert f"walk/v0.5: not run: {want}" in hi["reading_2_rungs"] and hi["reading_2_highest"] == "none"
    assert barreport.markdown(tabs).count(failed) >= 8
    # the other forms an entry can take
    tt = barreport.tried_text
    assert tt([dict(change="a_to_b", ramp_strides=2.0, gates=dict(ik_ok=True, join_a=False),
                    join_a=dict(com_max_mm=1.234, zmp_max_mm=2.5))]) == (
        "(1) `a_to_b` (ramp of 2 strides): failed `join_a`, largest difference from the walk: CoM 1.23 mm, ZMP 2.50 mm "
        "(its limits are the set's `g3_com_mm` and `g3_zmp_mm`)")
    assert tt([dict(change="a_to_b", failed_gates=["ik_ok", "zmp_error", "odd"], ik_fail_frames=3)]) == (
        "(1) `a_to_b`: failed `ik_ok`, 3 frames without a solution; failed `zmp_error` (the entry holds no "
        "`zmp_error_history_m`); failed `odd` (its reading is in the record)")
    assert tt([dict(change="a_to_b", failed_gates=[], pass_=True, stage0=dict(
        blocks_freeze=[dict(limit="clearance swing_clearance", value=0.0371)]))]) == (
        "(1) `a_to_b`: failed blocks the freeze at Stage 0: {limit: clearance swing_clearance, value: 0.0371}")
    assert tt([dict(failed_gates=[], pass_=True), "x"]) == (
        "(1) a change the entry does not name: no failed gate recorded (pass_ True); (2) x")
    # numbers that hold no list `tried` print as before, under the reference's own name
    assert barreport.why_text(dict(reason="r", reference="refs/x", numbers=dict(a=1.5, limit_b=2))) == (
        f"r [refs/x] ({OWN}: a=1.5; {LIMITS}: limit_b=2)")
    # a value that is a list of mappings, an empty list, a long one: one value each, cut where it is long
    assert barreport._leaf([dict(f=1.0, ok=False), dict(f=0.95, ok=True)]) == "{f: 1, ok: False}/{f: 0.95, ok: True}"
    assert barreport._leaf([]) == barreport._leaf({}) == "none (empty in the record)" and barreport._leaf([1, "a", 2.5]) == "1/a/2.5"
    long = barreport._leaf(list(range(200)))
    assert long.endswith(" ... (cut here: the whole value is in the record)") and len(long) < barreport.LEAF_MAX + 60
    assert barreport._flat(dict(builds=[], tried=[dict(f=1.0)]))[0] == "builds=none (empty in the record), tried={f: 1}"


def test_a_set_the_runner_refuses_is_said_with_the_runners_message(run, tmp_path):
    """The tables read each set's conditions without a reference by the runner's own rule; where the runner refuses
    the set's documents its message is printed under the checks and in every cell of that set that has no record.
    Nothing is dropped silently."""
    study = tmp_path / "study"
    out = study / "runs" / "x"
    put(study, out, "V0", "R0@10ms", "walk/v0.4")
    docs = set_docs("V0")  # a set that holds its nominal walk and says it starts straight into the 0.40 m/s walk
    docs["set.json"].update(set="F25F9", lead_in="start_v0.4")
    write_set(study, "F25F9", docs)
    d = f"stage2/sets/F25F9/{BODY}"
    gaps, refused = barreport._set_gaps(study / d, d)
    with pytest.raises(barreport.tune.InputError, match="records lead_in 'start_v0.4' but the set holds its nominal walk"):
        bar.BarPlanner(study / d, docs["refs.json"]["references"], docs["transitions.json"]["transitions"], docs["set.json"])
    assert gaps is None and refused.startswith(
        f"{barreport.REFUSED}: {d}: set.json records lead_in 'start_v0.4' but the set holds its nominal walk: ")
    assert barreport.REFUSED == "the set's documents were refused by the runner"
    doc = barreport.load(out, study=study)
    meta = doc[barreport.META]
    assert meta["set_refused"] == {"F25F9": {BODY: refused}} and "F25F9" not in meta["set_gaps"]
    tabs = barreport.tables(doc)
    assert tabs["checks"]["sets_refused"] == {f"F25F9/{BODY}": refused} and f"F25F9/{BODY}" not in tabs["checks"]["no_reference"]
    md = barreport.markdown(tabs)
    top = md.split("## How to read the tables")[0]
    assert f"cell of theirs with no record says so): 1\n  - `F25F9/{BODY}`: {refused}" in top
    assert str(tmp_path) not in md  # the message names the set's directory as the tables cite it
    for x in lines(tabs, "walk", "counts", set="F25F9") + lines(tabs, "ladder", "counts", set="F25F9"):
        assert x["status"] == f"{barreport.NOT_RUN}; {refused}", x  # never the bare word
    assert one(tabs, "glance", "glance", set="F25F9", condition="walk/v0.4")["reading_2"] == f"{barreport.NOT_RUN}; {refused}"
    assert one(tabs, "glance", "tally", set="F25F9")["reading_2"] == {barreport.NOT_RUN: 16}
    assert barreport.absent_word(meta, "F25F9", BODY, H, "R2/42@10ms", "walk/v0.4") == barreport.NOT_RUN
    assert one(tabs, "walk", "counts", set="V0", row="R2/42@10ms", condition="walk/v0.4")["status"] == barreport.NOT_RUN
    # a lead-in the bar does not know is refused too; documents that are no set's are said in other words
    docs["set.json"]["lead_in"] = "start_v0.45"
    write_set(study, "F25F9", docs)
    assert barreport._set_gaps(study / d, d) == (None, f"{barreport.REFUSED}: set.json of 'F25F9' records lead_in "
                                                       f"'start_v0.45': the bar knows start and start_v0.4 ({bar.RULES['lead_in']})")
    s2io.write_json(study / d / "refs.json", dict(walks={}))
    gaps, said = barreport._set_gaps(study / d, d)
    assert gaps is None and said == f"{barreport.UNREADABLE_SET} (KeyError: 'references')"
    assert barreport.tables(barreport.load(out, study=study))["checks"]["sets_refused"] == {f"F25F9/{BODY}": said}
    (study / d / "refs.json").unlink()  # a document that is not there: nothing to read, nothing said
    assert barreport._set_gaps(study / d, d) == (None, None)
    # the three synthetic sets of the run are accepted, and say so
    assert run.tabs["checks"]["sets_refused"] == {} and "says so): 0\n" in run.md


def test_a_record_with_no_episode_up_reads_as_a_fail_in_every_table(run):
    """A record with 0 of 20 episodes up: "does not pass" in both readings, the controller's cell says that neither
    row passes before the runner's failing word, no measured cell is left blank, and nothing reads as a pass."""
    for row in ("R2/42@10ms", "R2/43@10ms"):
        x = one(run.tabs, "walk", "counts", set="V0", condition="walk/v0.4", delay=J, row=row)
        assert x["up"] == dict(k=0, n=20) and x["r2"] == x["r3"] == dict(k=0, n=20) and x["up_lcb"] == 0.0
        assert x["r2_pass"] == x["r3_pass"] == "does not pass" and x["status"] == "done"
        m = one(run.tabs, "walk", "bar_walk", set="V0", condition="walk/v0.4", row=row)
        assert m["speed"] == m["toe"] == m["along"] == dict(median=None, n=0) and isinstance(m["ref_toe"], dict)
        assert m["along_limit"] == m["lateral_limit"] == m["toe_limit"] == barreport.NO_EPISODE_HOLDS  # no blank
        f = one(run.tabs, "walk", "fails", set="V0", condition="walk/v0.4", delay=J, row=row)
        assert f["worst_ratio"] == f["worst_leg_ratio"] == dict(largest=None, n=0) and f["worst_drive"] is None
        assert f["ratio_limit"] == 1.0 and f["failed_2"] == {"survived": 20}
    fail = f"{barreport.NEITHER_PASSES}: not attributed"
    v = one(run.tabs, "walk", "verdicts", set="V0", condition="walk/v0.4", delay=J)
    g = one(run.tabs, "glance", "glance", set="V0", condition="walk/v0.4")
    assert v["reading_2"] == v["reading_3"] == g["reading_2"] == g["reading_3"] == fail
    assert one(run.tabs, "glance", "tally", set="V0")["reading_2"] == {barreport.CUT: 1, fail: 1, "not run": 14}
    got = barreport.verdict_of(run.doc, "V0", BODY, "walk/v0.4", 10)
    assert got["reading_2"] == dict(verdict="not attributed", text=fail, word=fail, replication=None)  # the runner's word
    hi = one(run.tabs, "ladder", "highest", set="V0", grip="mu 0.9", delay=J)
    assert f"walk/v0.4: {fail}" in hi["reading_2_rungs"] and hi["reading_2_highest"] == "none"
    none = barreport.NO_EPISODE_HOLDS  # no episode gives a measured value or holds a limit: said in each cell
    assert f"| - (0) | - (0) | {none} | - (0) | {none} | - (0) | {none} | 25.11 (0.0500 H) | - (0) | - (0) |" in run.md
    for _, p, x in every_line(run.tabs):  # in the measured parts no limit and no measured cell is left blank
        if p["key"] in barreport.MEASURE_COLUMNS:
            assert not [c["key"] for c in p["columns"] if c["kind"] in ("measured", "limit", "defined")
                        and x.get(c["key"]) is None], x
    notes = " ".join(part(run.tabs, "walk", "verdicts")["notes"])
    assert f"\"{barreport.NEITHER_PASSES}: \"" in notes and "that is a fail, never a pass" in notes
    # an attributed failing word is prefixed the same way; a pass, another delay and a cell without counts are not
    doc = copy.deepcopy(run.doc)
    for row in ("R2/42@10ms", "R2/43@10ms"):
        for k in ("failure", "failure_but_A1"):
            doc["V0"][BODY][H][row]["walk/v0.4"][k]["label"] = "body-limited: drive_torque"
    assert barreport.verdict_of(doc, "V0", BODY, "walk/v0.4", 10)["reading_3"]["text"] == f"{barreport.NEITHER_PASSES}: body-limited"
    # at another delay the runner's failing word of reading 3 ends "but for A1", the reading's name: the tables' cell
    # cannot be read as "would pass but for A1"
    for row in ("R2/42@4ms", "R2/43@4ms"):
        rec = doc["F25F9cl"][BODY][H][row]["walk/v0.4"]
        for ck in ("counts", "counts_but_A1"):
            rec[ck].update(succeeded=0, success_lcb=0.0, pass_=False)
        for e in rec["episodes"]:
            e.update(succeeded=False, succeeded_but_A1=False)
    four = barreport.verdict_of(doc, "F25F9cl", BODY, "walk/v0.4", 4)
    assert four["reading_2"]["text"] == four["reading_2"]["verdict"] == "does not pass at 4 ms"
    assert four["reading_3"]["verdict"] == "does not pass at 4 ms but for A1"  # the runner's word
    assert four["reading_3"]["text"] == "does not pass at 4 ms, also with A1 set aside"
    late = one(barreport.tables(doc), "walk", "verdicts", set="F25F9cl", condition="walk/v0.4", delay=REP4)
    assert late["reading_3"] == "does not pass at 4 ms, also with A1 set aside"
    assert not [s_ for _, _, x in every_line(barreport.tables(doc)) for s_ in strings(x) if "does not pass at 4 ms but for A1" in s_]
    assert one(run.tabs, "walk", "verdicts", set="F25F9cl", condition="walk/v0.4", delay=REP4)["reading_3"] == "passes at 4 ms but for A1"
    for _, p, x in every_line(run.tabs):
        for k in ("reading_2", "reading_3"):
            cell = x.get(k)
            if isinstance(cell, str) and cell.startswith(barreport.NEITHER_PASSES):
                assert x.get("delay", J) == J and not any(w in cell for w in ("shown", "provisional", "passes at")), x
    # the JPS line of such a record says that no speed was measured
    study_doc = copy.deepcopy(run.doc)
    for row in ("R2/42@0ms", "R2/43@0ms"):
        rec = study_doc["V0"][BODY][H][row]["walk/v0.35"]
        for e in rec["episodes"]:
            e["measures"]["window"] = None
    jps = {x["line"]: x for x in part(barreport.tables(study_doc), "jps", "jps")["lines"]}[92]["record"]
    assert any("; no measured speed (no episode reached the judged 10 s)" in s_ for s_ in jps)
    assert not any("measured - " in s_ for x in part(run.tabs, "jps", "jps")["lines"] for s_ in x["record"])


def test_the_items_own_columns(run):
    t = one(run.tabs, "rest", "bar_turn", set="F25F9cl", condition="turn/q90_left", row="R2/42@10ms")
    assert t["heading_at"] == dict(median=-4.5, n=20) and t["within"] == dict(median=1.71, n=20)
    assert (t["heading_limit"], t["at_defined"]) == (10.0, 2.0) and t["heading_hold"]["median"] == 6.2
    n = one(run.tabs, "rest", "bar_nudge", set="F25F9cl", condition="nudge/forward", row="R2/42@10ms")
    assert n["recovered"] == dict(k=18, n=20) and n["back"] == dict(median=1.23, n=18) and n["back_limit"] == 3.0
    assert n["force_defined"] == pytest.approx(21.87)
    f = one(run.tabs, "rest", "fails", set="F25F9cl", condition="nudge/forward", row="R2/42@10ms")
    assert (f["falls_before"], f["falls_after"], f["falls_no_command"]) == (1, 1, 0) and f["failed_2"] == {"survived": 2}
    s = lines(run.tabs, "rest", "bar_stop", set="F25F9cl", condition="stop/two_step", row="R2/42@10ms")
    assert [x["touchdowns"] for x in s] == [{"2": 20}, {"3": 20}] and s[0]["stopped_s"] == dict(median=1.11, n=20)
    assert s[0]["accel_limit"] == 1.5 and s[0]["stopped"] == dict(k=20, n=20)
    w = one(run.tabs, "walk", "fails", set="V0", condition="walk/v0.4", delay=J, row="R2/42@10ms")
    assert w["falls_no_command"] == 20 and w["failed_2"] == {"survived": 20} and w["worst_drive"] is None
    assert w["attribution"] == "not attributed (falls; run with --attribute)" == w["label_2"]


def test_numbers_are_printed_with_a_fixed_precision_or_g():
    assert barreport._num(3) == "3" and barreport._num(4.0) == "4" and barreport._num(0.25) == "0.25"
    assert barreport._num(0.36614, "m/s") == "0.366" and barreport._num(23.6349, "mm") == "23.63"
    assert barreport._num(2.0, "core-h") == "2.000" and barreport._num(1e-9) == "1e-09"
    col = dict(kind="measured", unit="deg")
    assert barreport._cell_text(dict(median=-4.25, n=20), col) == "-4.2 (20)" and barreport._cell_text(None, col) == "-"
    assert barreport._cell_text(dict(median=None, n=0), col) == "- (0)"  # no episode gives it: said, not blank
    assert barreport._cell_text(dict(largest=0.9912, n=2), dict(kind="measured", unit="ratio")) == "0.991 (2)"
    assert barreport._cell_text({}, dict(kind="terms")) == "none"
    assert barreport._cell_text({"a": 2, "b": 1}, dict(kind="terms")) == "a: 2; b: 1"
    assert barreport._cell_text(dict(k=17, n=20), dict(kind="of")) == "17 of 20"
    assert barreport._cell_text("a/b.json", dict(kind="path")) == "`a/b.json`"
    assert barreport._cell_text(True, dict(kind="flag")) == "yes"
    assert barreport._cell_text("not defined", dict(kind="flag")) == "not defined"


# ------------------------------------------------------------------------------------------------
# The speed ladder
# ------------------------------------------------------------------------------------------------
def test_the_highest_rung_that_passes_under_both_rows(run):
    hi = one(run.tabs, "ladder", "highest", set="F25F9cl", grip="mu 0.9", delay=J)
    assert hi["reading_2_highest"] == "none"  # 0.40 m/s is provisional, 0.45 m/s falls
    assert hi["reading_3_highest"] == "walk/v0.4 (walk at 0.4 m/s)"
    assert hi["reading_2_rungs"][0].startswith("walk/nominal: cut by the pre-registered order")
    assert "walk/v0.4: provisional [R2/42@10ms: replicated]" in hi["reading_2_rungs"]
    assert [r.split(":")[0] for r in hi["reading_2_rungs"]] == ["walk/nominal", "walk/v0.3", "walk/v0.35", "walk/v0.4",
                                                                "walk/v0.45", "walk/v0.5"]
    zero = one(run.tabs, "ladder", "highest", set="F25F9cl", grip="mu 0.9", delay="0 ms (reported)")
    assert zero["reading_2_highest"] == zero["reading_3_highest"] == "walk/v0.5 (walk at 0.5 m/s)"
    assert "walk/v0.5: passes at 0 ms" in zero["reading_2_rungs"]
    v0 = one(run.tabs, "ladder", "highest", set="V0", grip="mu 0.9", delay="0 ms (reported)")
    assert v0["reading_2_highest"] == "none" and v0["reading_3_highest"] == "walk/v0.35 (walk at 0.35 m/s)"
    assert not lines(run.tabs, "ladder", "counts", condition="walk/v0.4")  # the 0.40 m/s rung's records are table 1's
    assert one(run.tabs, "ladder", "counts", set="V0", condition="walk/v0.35", row="R2/42@0ms")["stage1"] == dict(k=0, n=20)


def test_a_line_with_counts_or_verdict_words_cites_the_records_they_come_from(run):
    """The glance line cites its R0 record beside the R2 records; the highest rung cites every rung's records."""
    g = one(run.tabs, "glance", "glance", set="F25F9cl", condition="walk/v0.4")
    assert g["r0_up"] == dict(k=17, n=20) and g["r0_path"].endswith("seeds3042/eval/R0@10ms/walk/v0.4.json")
    assert [q.split("/eval/")[1] for q in g["paths"]] == ["R2/42@10ms/walk/v0.4.json", "R2/43@10ms/walk/v0.4.json"]
    assert [c["title"] for c in part(run.tabs, "glance", "glance")["columns"]][-2:] == ["R2 records", "R0 record"]
    n = 0
    for x in part(run.tabs, "glance", "glance")["lines"]:
        assert (x["r0_path"] is None) == (x["r0"] in (barreport.NOT_RUN, barreport.CUT)), x
        assert x["r0_up"] is None or x["r0_path"], x
        n += x["r0_path"] is not None
    assert n == 3  # the R0 records of the synthetic run at 10 ms: one per set
    hi = one(run.tabs, "ladder", "highest", set="F25F9cl", grip="mu 0.9", delay=J)
    assert [q.split("/eval/")[1] for q in hi["paths"]] == [
        f"{r}@10ms/walk/{k}.json" for k in ("v0.4", "v0.45") for r in barreport.R2_ROWS]
    assert part(run.tabs, "ladder", "highest")["columns"][-1]["kind"] == "path"
    absent = (barreport.NOT_RUN, barreport.CUT)
    for x in part(run.tabs, "ladder", "highest")["lines"]:
        said = [r for r in x["reading_2_rungs"] + x["reading_3_rungs"] if not r.split(": ", 1)[1].startswith(absent)]
        assert bool(x["paths"]) == bool(said), x
    # in every table: a line that holds a count cites a record, in its own line or under its part
    for name, p, x in every_line(run.tabs):
        if any(c["kind"] == "of" and x.get(c["key"]) is not None for c in p["columns"]):
            assert any(x.get(c["key"]) for c in p["columns"] if c["kind"] == "path") or p.get("paths"), (name, p["key"], x)


# ------------------------------------------------------------------------------------------------
# The comparison with JPS section 2.3
# ------------------------------------------------------------------------------------------------
def strip(cell: str) -> str:
    return re.sub(r"\*\*|`|\*", "", cell).strip()


def test_jps_claims_are_the_reports_own_lines():
    text = (ROOT / barreport.JPS_FILE).read_text(encoding="utf-8").split("\n")
    assert [c["line"] for c in barreport.JPS_ROWS] == list(range(92, 102)) and set(barreport.JPS_MAP) == set(range(92, 102))
    for claim in (barreport.JPS_HEADER,) + barreport.JPS_ROWS:
        cells = [strip(c) for c in text[claim["line"] - 1].strip().strip("|").split("|")]
        assert cells == [claim[k] for k in barreport.JPS_CLAIM_KEYS], claim["line"]
    assert text[85].startswith("### 2.3 ")
    for cells in barreport.JPS_MAP.values():  # every mapped cell is a condition the bar defines
        for c in cells:
            assert isinstance(c, str) or (c[0] in barreport.SETS and c[1] in barreport._conds() and c[2] % 2 == 0)


def test_jps_numbers_appear_only_under_jps_claim(run):
    claims = {c[k] for c in barreport.JPS_ROWS for k in barreport.JPS_CLAIM_KEYS if k not in ("delay", "floor_plant")}
    assert len(claims) > 30
    jps = part(run.tabs, "jps", "jps")
    assert len(jps["lines"]) == len(barreport.JPS_ROWS) == 10
    for line in jps["lines"]:
        for k, v in line.items():
            for s in strings(v):
                assert k.startswith("jps_claim_") or not any(c in s for c in claims), (k, s)
    for c in jps["columns"]:
        assert c["title"].startswith("JPS's claim: ") == (c["kind"] == "claim") == c["key"].startswith("jps_claim_")
    # nowhere else in the JSON
    rest = {k: v for k, v in run.tabs.items() if k != "jps"}
    rest["jps_other"] = [dict(p, lines=None) if p["key"] == "jps" else p for p in run.tabs["jps"]["parts"]]
    assert not [s for s in strings(rest) if any(c in s for c in claims)]
    # in the Markdown: a claim stands in a cell of the JPS table under a header that says so, and nowhere else
    md = run.md.split("\n")
    head = next(i for i, x in enumerate(md) if x.startswith("| JPS line |"))
    titles = md[head].split(" | ")
    n = 0
    for i, row in enumerate(md):
        if not any(c in row for c in claims):
            continue
        assert head + 2 <= i < head + 2 + 10, row
        cells = row.split(" | ")
        assert len(cells) == len(titles)
        for title, cell in zip(titles, cells, strict=True):
            if any(c in cell for c in claims):
                assert title.startswith("JPS's claim: "), (title, cell)
                n += 1
    assert n >= 40
    assert "(JPS's claim) is set beside this round's 0.50 m/s rung" in run.md
    assert "commanded causally" in run.md and "JPS's were not" in run.md


def test_jps_rows_stand_beside_this_rounds_records_seed_by_seed(run):
    rows = {x["line"]: x for x in part(run.tabs, "jps", "jps")["lines"]}
    # 0.496 m/s beside the 0.50 m/s rung, with the measured speed and the three readings of each row
    r95 = rows[95]
    assert "F25F9cl walk/v0.5 at 0 ms (walk at 0.5 m/s, mu 0.9; rows R2/42@0ms and R2/43@0ms); the rung's target " \
           "speed 0.5 m/s" in r95["matched"]
    assert ("F25F9cl walk/v0.5 at 0 ms, R2/42@0ms: stayed up 20 of 20; Stage 1's reading 20 of 20; reading 2 20 of 20 "
            "(passes at 0 ms); reading 3 20 of 20 (passes at 0 ms); median speed measured 0.514 (20) m/s "
            f"[{barreport.MARGIN_LABEL}]") in r95["record"]
    assert f"F25F9cl walk/v0.45_mu0.54 at 0 ms, R2/42@0ms: cut by the pre-registered order{M}" in r95["record"]
    assert ("F25F9cl walk/v0.5 at 0 ms, the controller R2 from its two rows: reading 2: passes at 0 ms; reading 3: "
            f"passes at 0 ms but for A1{M}") in r95["record"]
    assert ("F25F9cl walk/v0.4 at 10 ms, the controller R2 from its two rows: reading 2: provisional [R2/42@10ms: "
            f"replicated]; reading 3: shown but for A1 [R2/42@10ms: replicated; R2/43@10ms: not replicated]{M}") \
        in rows[99]["record"]
    assert any(p.endswith("seeds3042/eval/R2/43@0ms/walk/v0.5.json") for p in r95["paths"])
    r92 = rows[92]
    assert any("V0 walk/v0.35 at 0 ms, R2/43@0ms: stayed up 20 of 20; Stage 1's reading 0 of 20; reading 2 0 of 20"
               in s and "median speed measured 0.366 (20) m/s" in s for s in r92["record"])
    # cells this round has no condition for, and cells outside the registered order
    assert rows[96]["record"] == ["the walk at 0.55 m/s: this round's speed ladder ends at 0.50 m/s: not run"]
    assert rows[97]["record"][0].endswith(": not run") and rows[96]["paths"] == rows[97]["paths"] == []
    assert f"F25F9cl walk/v0.5 at 8 ms, R2/42@8ms: not run{M}" in rows[99]["record"]
    assert f"F25F9cl walk/v0.5 at 6 ms, R2/42@6ms: cut by the pre-registered order{M}" in rows[99]["record"]
    assert f"F25F9cl stop/two_step at 2 ms, R2/42@2ms: not run{M}" in rows[100]["record"]
    assert rows[100]["record"][-1] == "the stop from 0.50 m/s: this round's stop is from the 0.40 m/s walk: not run"
    assert f"F25F9cl nudge/left at 2 ms, R2/43@2ms: not run{M}" in rows[101]["record"]
    # every line of the candidate ends with its labels; no line of a control holds the margin label
    for x in rows.values():
        for text in x["record"]:
            if text.startswith("F25F9cl "):
                assert text.endswith("]") and barreport.MARGIN_LABEL in text.rsplit(" [", 1)[1], text
            else:
                assert barreport.MARGIN_LABEL not in text, text
    # seed by seed: one part per matched cell with a record, a line per seed, the two rows side by side
    sp = part(run.tabs, "jps", "seeds V0 walk/v0.35 at 0 ms")
    assert [x["seed"] for x in sp["lines"][:-1]] == list(bar.BLOCKS[H])
    assert [c["title"] for c in sp["columns"]] == ["Seed"] + [
        f"{r}@0ms: {t}" for r in ("R2/42", "R2/43") for t in ("stayed up", "Stage 1's reading", "reading 2", "reading 3")]
    assert sp["lines"][0] == dict(seed=3042, up_0=True, stage1_0=False, r2_0=False, r3_0=True,
                                  up_1=True, stage1_1=False, r2_1=False, r3_1=True)
    assert sp["lines"][-1]["r3_0"] == "20 of 20" and sp["lines"][-1]["r2_0"] == "0 of 20"
    assert len(sp["paths"]) == 2 and all("walk/v0.35.json" in p for p in sp["paths"])
    keys = [p["key"] for p in run.tabs["jps"]["parts"][1:]]
    assert keys == ["seeds V0 walk/v0.35 at 0 ms", "seeds F25F9cl walk/v0.4 at 0 ms", "seeds F25F9cl walk/v0.45 at 0 ms",
                    "seeds F25F9cl walk/v0.5 at 0 ms", "seeds F25F9cl walk/v0.4 at 10 ms"]
    judged = part(run.tabs, "jps", "seeds F25F9cl walk/v0.4 at 10 ms")
    assert judged["lines"][3]["r2_1"] is False and judged["lines"][3]["r3_1"] is True  # the tail is over in reading 2
    assert "| 3042 | yes | no | no | yes | yes | no | no | yes |" in run.md


def test_jps_table_is_part_of_the_tables_and_keeps_to_the_sets_asked_for(run):
    assert barreport.jps_table(run.doc)["title"] == run.tabs["jps"]["title"] == "6. The comparison with JPS section 2.3"
    assert json.loads(json.dumps(barreport.jps_table(run.doc))) == run.tabs["jps"]
    only = barreport.jps_table(barreport.load(run.out, sets=["V0"], study=run.study))
    rows = {x["line"]: x for x in only["parts"][0]["lines"]}
    assert rows[95]["record"][0] == "F25F9cl walk/v0.4 at 0 ms: the set is not among the sets asked for"
    assert [p["key"] for p in only["parts"][1:]] == ["seeds V0 walk/v0.35 at 0 ms"]


# ------------------------------------------------------------------------------------------------
# The plan's text and the chain
# ------------------------------------------------------------------------------------------------
def test_every_verdict_table_cites_the_rules_amended_after_a_result(run):
    """Every label "[changed 15 Nov after a result ...]" of the plan is in the tables' list, under its rule, in the
    plan's order, with the plan's own words; a label the plan gains later fails here until the tables name it."""
    plan = " ".join((ROOT / barreport.PLAN_FILE).read_text(encoding="utf-8").split())
    am = barreport.AMENDED_AFTER_RESULT
    assert [a["rule"] for a in am] == ["A7", "A7", "A8", "A11", "A11", "A12", "C3"]
    found = [(re.findall(r"\*\*([A-Z]\d+)\. ", plan[:m.start()])[-1], m.group(0), m.end())
             for m in re.finditer(r"\[changed 15 Nov after a result[^\]]*\]", plan)]
    assert [(rule, label) for rule, label, _ in found] == [(a["rule"], a["label"]) for a in am]
    for a, (_, _, end) in zip(am, found, strict=True):  # the words that follow the label, word for word
        assert plan[end:].startswith(" " + a["text"]), a["rule"]
        assert a["text"].endswith(".") or a["note"], a["rule"]  # a sentence cut short says so
    assert sum(a["label"] == barreport.P34_LABEL for a in am) == 4 and "`DECISION_LOG.md` P34" in barreport.P34_LABEL
    assert {a["rule"] for a in am if a["label"] == barreport.P34_LABEL} == {"A7", "A8", "A11", "A12"}
    note = barreport.amended_note()
    assert note.count("\n  - ") == len(am) and all(f"\n  - {a['rule']} {a['label']} {a['text']}" in note for a in am)
    assert run.tabs["amended_after_result"] == [dict(a) for a in am]
    n = 0
    for name in barreport.ORDER:
        for p in run.tabs[name]["parts"]:
            verdict = any(c["title"].startswith(("Reading 2 (every limit)", "Highest rung", "This round's record"))
                          and c["kind"] in ("word", "list") for c in p["columns"]) or p["key"] in ("replication", "tally")
            if verdict:
                n += 1
                assert note in (p.get("notes") or []), (name, p["key"])
    assert n >= 8
    assert run.md.count(am[-1]["label"]) == n + 1  # and once under "How to read"
    assert run.md.count(barreport.P34_LABEL) == 4 * (n + 1)
    assert barreport.markdown(run.tabs).count("- " + note) == n + 1  # one line per label, under one bullet


def test_the_rules_made_exact_after_a_prescribed_motion_result_are_a_list_of_their_own(run):
    """The lead's ruling: the labels of the plan that begin "[made exact 15 Nov after a prescribed-motion result" are
    printed in a second, separately titled list, read from the plan as the first one is; a label the plan gains
    later fails here until the tables print it."""
    plan = " ".join((ROOT / barreport.PLAN_FILE).read_text(encoding="utf-8").split())
    exact = barreport.MADE_EXACT_AFTER_RESULT
    found = [(re.findall(r"\*\*([A-Z]\d+)\. ", plan[:m.start()])[-1], m.group(0), m.end())
             for m in re.finditer(r"\[made exact 15 Nov after a prescribed-motion result[^\]]*\]", plan)]
    assert [(rule, label) for rule, label, _ in found] == [(a["rule"], a["label"]) for a in exact] and len(exact) == 1
    assert [a["rule"] for a in exact] == ["A8"] and "one stride of ramp" in exact[0]["text"]
    for a, (_, _, end) in zip(exact, found, strict=True):  # the words that follow the label, word for word
        assert plan[end:].startswith(" " + a["text"]) and a["text"].endswith("."), a["rule"]
    # its own title and its own bullet: not a line of the list of rules amended after a result
    note, amended = barreport.made_exact_note(), barreport.amended_note()
    assert note.startswith(barreport.MADE_EXACT_TITLE) and barreport.MADE_EXACT_TITLE != barreport.AMENDED_TITLE
    assert barreport.MADE_EXACT_TITLE.startswith("Rules made exact after a prescribed-motion result")
    assert note.count("\n  - ") == len(exact) and f"\n  - A8 {exact[0]['label']} {exact[0]['text']}" in note
    assert exact[0]["label"] not in amended and "[made exact" not in amended
    assert not [a for a in barreport.AMENDED_AFTER_RESULT if a["label"].startswith("[made exact")]
    assert run.tabs["made_exact_after_result"] == [dict(a) for a in exact]
    n = 0
    for name in barreport.ORDER:  # wherever the first list stands, the second stands after it
        for p in run.tabs[name]["parts"]:
            notes = p.get("notes") or []
            assert (amended in notes) == (note in notes), (name, p["key"])
            if note in notes:
                assert notes.index(note) == notes.index(amended) + 1
                n += 1
    assert n >= 8 and run.md.count(exact[0]["label"]) == n + 1 == run.md.count("- " + barreport.MADE_EXACT_TITLE)
    how = run.md.split("## How to read the tables")[1].split("## 1.")[0]
    assert how.index("- " + barreport.AMENDED_TITLE) < how.index("- " + barreport.MADE_EXACT_TITLE)
    # the plan's other "made exact" labels are not of this kind (they follow no prescribed-motion result)
    assert len(re.findall(r"\[made exact 15 Nov", plan)) > len(found)


def test_the_plans_list_of_what_is_not_run_is_a_table_item_by_item(run, tmp_path):
    """The plan: "Each appears in the report as "not run"".  Table 9 holds one line per item of the plan's sentence,
    in the plan's words, and a finished record of an item is counted and cited instead of the words."""
    plan = " ".join((ROOT / barreport.PLAN_FILE).read_text(encoding="utf-8").split())
    items = [x[0] for x in barreport.NOT_RUN_BY_NAME]
    sentence = plan.split(f"## {barreport.NOT_RUN_SECTION} ", 1)[1].split(" Each appears in the report as", 1)[0]
    assert sentence == ", ".join(items[:4]) + "; " + "; ".join(items[4:]) + "."  # as the plan joins them
    assert len(items) == len(set(items)) == 14 and barreport.ORDER[-1] == "named"
    named = part(run.tabs, "named", "named")["lines"]
    assert [x["item"] for x in named] == items and run.tabs["named"]["title"] == f"9. {barreport.NOT_RUN_SECTION}"
    assert all(x["status"] == barreport.NOT_RUN and x["found"] == 0 and x["paths"] == [] for x in named)
    assert {x["reference"] for x in named} == {f'{barreport.PLAN_FILE}, "{barreport.NOT_RUN_SECTION}"'}
    for item in items:
        assert f"| {item} | {barreport.NOT_RUN} | 0 |" in run.md, item
    assert barreport.named_table(run.doc)["parts"][0]["lines"] == named
    # a finished record of an item: counted and cited; a "not run" record of it is no run
    study = tmp_path / "study"
    out = study / "runs" / "x"
    put(study, out, "F25F9", "R2/42@10ms", "stop/two_step")
    put(study, out, "F25F9", "R2/43@10ms", "stop/two_step", not_run=dict(reason="synthetic", reference="x", numbers=None))
    put(study, out, "F25F9cl", "R0@10ms", "walk/v0.45")
    put(study, out, "F25F9cl", "R0@10ms", "walk/v0.4")  # the bar's own R0 row: none of the list
    put(study, out, "F25F9cl", "R2/42@2ms", "nudge/left")
    put(study, out, "V0", "R2/42@14ms", "walk/v0.4")
    tabs = barreport.tables(barreport.load(out, study=study))
    by = {x["item"]: x for x in part(tabs, "named", "named")["lines"]}
    found = {i: by[i]["paths"] for i in items if by[i]["found"]}
    ends = lambda paths: [q.split("/eval/")[1] for q in paths]  # noqa: E731
    assert {i: ends(q) for i, q in found.items()} == {
        "delays of 14, 16 and 20 ms": ["R2/42@14ms/walk/v0.4.json"],
        items[5]: ["R2/42@2ms/nudge/left.json"],
        items[6]: ["R2/42@10ms/stop/two_step.json"],
        "`R0` rows on the ladders and on the PR's remaining rows": ["R0@10ms/walk/v0.45.json"]}
    assert items[5].startswith("the turn, nudge and stop conditions at 2 and 8 ms") and "F25F9" in items[6]
    for i in found:
        assert by[i]["status"] == "1 finished record found under the run directory" and by[i]["found"] == 1
    assert all(by[i]["status"] == barreport.NOT_RUN for i in items if i not in found)
    assert by["stiffness ×0.7"]["status"] == barreport.NOT_RUN  # no record of the bar could show it
    assert all(m is not None for i, _, m in barreport.NOT_RUN_BY_NAME if i != "stiffness ×0.7")
    # the tests on a record, one by one
    fact = lambda **kw: dict(dict(d={}, base="R2/42", ms=10, behavior="walk", key="walk/v0.4", stock=False,  # noqa: E731
                                  body=BODY, set="V0", look=False), **kw)
    hits = lambda x: [i for i, _, m in barreport.NOT_RUN_BY_NAME if m is not None and m(x)]  # noqa: E731
    assert hits(fact()) == []
    assert hits(fact(behavior="turn", key="turn/x", d=dict(walk=setio.NOMINAL))) == [items[0]]
    assert hits(fact(behavior="carpet")) == ["carpet walks"] and hits(fact(d=dict(floor="robot_floor_carpet"))) == [items[3]]
    assert hits(fact(ms=20)) == ["delays of 14, 16 and 20 ms"] and hits(fact(ms=8)) == []
    assert hits(fact(behavior="stop", key="stop/two_step", ms=8)) == [items[5]]
    assert hits(fact(stock=True)) == ["Stage 1's twelve no-loss conditions (plan section 5)"]
    assert hits(fact(look=True)) == ["the look gates and reading 1"]
    assert hits(fact(key="walk/v0.55")) == ["0.55 m/s"] and hits(fact(body="75-M-H-10-J")) == [items[11]]
    assert hits(fact(behavior="stop", key="stop/x", d=dict(walk=setio.walk_name(0.5)))) == ["a stop from 0.50 m/s"]
    assert hits(fact(body="60-N-T-10")) == ["any 60 cm body"]
    assert hits(fact(set="F25F9cl", base="R0", key="walk/v0.5", ms=2)) == [items[7]]
    assert hits(fact(set="F25F9cl", base="R0", key="turn/t90_0.5_left", behavior="turn")) == [items[7]]


def test_the_registered_order_is_the_chains(run):
    """``planned()`` holds the cells of the chain's units on the headline block (``run_75cm.sh list`` runs nothing)."""
    script = ROOT / "stage2" / "runs" / "2026-11-15-75cm" / "run_75cm.sh"
    listed = subprocess.run(["bash", str(script), "list"], capture_output=True, text=True, check=True).stdout
    cells, controls, units, again = {}, set(), 0, set()
    for line in listed.splitlines():
        if "designlab.stage2.bar run " not in line:
            continue
        args = line.split(": ", 1)[1].split()
        opt = {a: args[i + 1] for i, a in enumerate(args[:-1]) if a.startswith("--")}
        units += 1
        keys = [k for c in opt["--conditions"].split(",") for k in (barreport._group(c) if c.startswith("@") else [c])]
        for row in opt["--rows"].split(","):
            base, ms = barreport._split_row(row)
            for k in keys:
                if opt.get("--block", H) != H:  # a replication unit: the candidate's passes of the first block
                    assert (opt["--block"], opt["--passed-of"], opt["--set"]) == (R, H, bar.CANDIDATE_SET)
                    assert base in barreport.R2_ROWS
                    again.add((k, ms))
                elif "--control" in opt:
                    controls.add((opt["--control"], opt["--set"], k, base, ms))
                else:
                    assert cells.setdefault((opt["--set"], k, ms, base), int(opt["--task"])) == int(opt["--task"])
    assert units > 80 and cells == barreport.planned()
    # the replication units of the order, and what they leave out: the ladder at 4 and 6 ms, the reported ladders
    assert again == barreport.replicated()
    cand = {(k, ms) for (s_, k, ms, base) in cells if s_ == bar.CANDIDATE_SET and base in barreport.R2_ROWS}
    left = cand - again
    assert left == ({(k, ms) for k in barreport._group("@speed-ladder") for ms in (4, 6)}
                    | {(k, 10) for k in barreport._group("@ladders")})
    assert all(barreport.replication_in_order(bar.CANDIDATE_SET, BODY, f"R2/42@{ms}ms", k) is False for k, ms in left)
    c, s, k, base, delays = barreport.OMEGA_CONTROL
    assert controls == {(c, s, k, base, ms) for ms in delays}
    assert sum(1 for t in cells.values() if t == 3) == 3 * 2 * 6 * 3 + 2 * 5 * 4 * 2 + 4 * 2
    assert sum(1 for t in cells.values() if t == 4) == 2 * (14 * 4 * 3 + 15 * 2)
    assert set(barreport.TASK_OF) == set(bar.BEHAVIORS) and set(barreport.TASK_OF.values()) == set(bar.BUDGET_CORE_H)


def test_a_run_directory_without_records_renders_every_table(tmp_path):
    out = tmp_path / "study" / "stage2" / "runs" / "empty"
    out.mkdir(parents=True)
    tabs = barreport.tables(barreport.load(out, study=tmp_path / "study"))
    md = barreport.markdown(tabs)
    assert tabs["checks"]["records"]["found"] == 0 and "Records found: 0 (0 done, 0 \"not run\")" in md
    for name in barreport.ORDER:
        assert f"## {tabs[name]['title']}" in md
    # every cell of the registered order is there and reads "not run"
    w = part(tabs, "walk", "counts")["lines"]
    assert len(w) == 3 * 6 and {x["status"] for x in w} == {barreport.NOT_RUN}
    r = part(tabs, "rest", "counts")["lines"]
    assert len(r) == 2 * 4 and r[0]["item"] == "29 conditions, 72 records: none found"  # 14 x 3 rows + 15 x 2 rows
    g = part(tabs, "glance", "glance")["lines"]
    assert len(g) == 3 * 16 and {x["reading_2"] for x in g} == {barreport.NOT_RUN}
    # table 2 lists the pull request's row of F25F9 though no record of that set exists (task 3, tier 7)
    lad = part(tabs, "ladder", "counts")["lines"]
    assert list(dict.fromkeys(x["set"] for x in lad)) == ["F25F9cl", "F25F9", "V0"]
    f = one(tabs, "ladder", "counts", set="F25F9")
    assert (f["condition"], f["delay"], f["row"], f["status"]) == (
        "walk/v0.5", "0 ms (reported)", "R2/42@0ms, R2/43@0ms", barreport.NOT_RUN)
    assert one(tabs, "ladder", "verdicts", set="F25F9")["reading_2"] == barreport.NOT_RUN
    assert [x["delay"] for x in lines(tabs, "ladder", "highest", set="F25F9")] == ["0 ms (reported)"]
    jps = part(tabs, "jps", "jps")["lines"]
    assert all(s.endswith((": not run", ": not run" + M)) for x in jps for s in x["record"])
    assert len(tabs["jps"]["parts"]) == 1 and part(tabs, "controls", "controls")["lines"][0]["status"] == barreport.NOT_RUN
    assert part(tabs, "other", "other")["lines"] == [] and barreport.CUT not in md.split("## 1.")[1]


def test_a_stock_condition_and_a_row_without_a_delay_are_listed_in_the_last_table(run, tmp_path):
    study = tmp_path / "study"
    out = study / "runs" / "x"
    rec = put(study, out, "V0", "R0@10ms", "walk/v0.4")
    stock = dict(rec, set=bar.STOCK_SET, row="R2/42", behavior="stand", condition="still", key_name="stand/still",
                 counts_but_A1=None, definition=dict(behavior="stand", name="still", judge="stand", stock=True))
    s2io.write_json(bar.block_dir(out, bar.STOCK_SET, BODY, H) / "eval" / "R2/42" / "stand" / "still.json", stock)
    tabs = barreport.tables(barreport.load(out, study=study))
    x = one(tabs, "other", "other", set=bar.STOCK_SET)
    assert x["item"] == "Stage 1's condition stand/still" and x["r3_pass"] == "no such reading (Stage 1's one reading)"
    assert x["r2"] == dict(k=20, n=20) and tabs["checks"]["not_checked"] == [] and tabs["checks"]["records"]["found"] == 2
    assert not lines(tabs, "other", "other", set="V0")  # a record a table cites is not repeated


def test_output_paths_are_not_ignored_by_git():
    """The two files are committed with the run: git must not ignore them (logs/ is where a smoke goes)."""
    run_dir = "stage2/runs/2026-11-15-75cm"
    for name in (barreport.TABLES_JSON, barreport.TABLES_MD):
        r = subprocess.run(["git", "check-ignore", "-q", f"{run_dir}/{name}"], cwd=ROOT)
        assert r.returncode == 1, name
    smoke = "logs/barreport-smoke/" + barreport.TABLES_MD
    assert subprocess.run(["git", "check-ignore", "-q", smoke], cwd=ROOT).returncode == 0
