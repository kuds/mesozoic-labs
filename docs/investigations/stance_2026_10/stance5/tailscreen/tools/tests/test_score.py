"""Synthetic-row tests for score_tailscreen.py (build tier: no torch, no environments, no training).

Covers BUILD_SPEC section 9: every state-reading line per bound and state, every verdict line, outcomes A / B (both
directions) / C with each routing reason and both named combinations, missing-replicate scoping, monotonicity in
CONTROL outcomes, bound independence, each PROVISIONAL trigger, incomplete, and the CLI exit codes.
"""
from __future__ import annotations

import copy
import csv
import importlib.util
import itertools
import json
import random
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


S = _load("score_tailscreen")
C = S.C
SEEDS = C.REP_SEEDS
R1, R2 = S.R1, S.R2
STACK_JSON = json.dumps(C.STACK, sort_keys=True, separators=(",", ":"))
HEAD = "a" * 40
CRIT = "c" * 64
COMMON = "5" * 64
ZSHA = "9" * 64

# ------------------------------------------------------------------------------------------------- builders

CMD = {"CURED": 0.30, "PARKED": 0.75, "RAIL": 0.95}


def row(name, label, cmd=0.3, tailup=0, other=0):
    return {"name": name, "label": label, "cmd": repr(float(cmd)), "tailup_flags": str(tailup),
            "other_flags": str(other), "measure_tree_digest": "D", "seeds": "3042-3061", "stack": STACK_JSON,
            "n_episodes": "20", "class_single": "", "zip_sha256": ZSHA}


def rows_for(classes: dict, others: dict | None = None, skip=(), onset=()) -> list[dict]:
    """classes: name -> RAIL / PARKED / CURED; others: name -> OTHER flags at R1 (R2 gets 0).

    onset: names that get a tail-up flag at 4456448 (an early label, never R1/R2).
    """
    others = others or {}
    out = []
    for name in S.names():
        if name in skip:
            continue
        cls = classes.get(name, "CURED")
        for lb in C.LABELS:
            if lb in (R1, R2):
                t = 1 if (cls == "RAIL" and lb == R1) else 0
                o = others.get(name, 0) if lb == R1 else 0
                out.append(row(name, lb, CMD[cls], t, o))
            else:
                out.append(row(name, lb, 0.3, 1 if (name in onset and lb == "4456448") else 0, 0))
    return out


def base_classes() -> dict:
    """Every bound replicate CURED; CONTROL s42 has 1 CURED (no NOT SEPARATED); CONTROL s80 2 RAIL: all PASS."""
    c = {n: "CURED" for n in S.names()}
    c["s42_control_r101"], c["s42_control_r202"] = "RAIL", "PARKED"
    c["s80_control_r101"], c["s80_control_r202"] = "RAIL", "RAIL"
    return c


def nm(state, arm, i):
    return f"{state}_{arm}_r{SEEDS[i]}"


def good_prov(rows=None, lost=None) -> "S.ProvInputs":
    tools = {t: f"{i:064x}" for i, t in enumerate(S.TOOL_FILES)}
    metas = {}
    for name in S.names():
        st, arm, seed = S.parse_name(name)
        coef = {"control": 0.0, "b90": 2.7, "b60": 2.7}[arm]
        b = {"control": 0.90, "b90": 0.90, "b60": 0.60}[arm]
        metas[name] = {"clone": {"head": HEAD, "status": "", "branch": C.BRANCH}, "arm": arm, "coef": coef, "b": b,
                       "rep_seed": seed, "stack": dict(C.STACK), "status": "done", "utc_start": "2026-10-25T10:00:00Z",
                       "start": {"zip_sha256": C.STATES[st]["zip_sha256"], "pkl_sha256": C.STATES[st]["pkl_sha256"]},
                       "final_sha256": {"stage1_final.zip": ZSHA, "stage1_final_vecnorm.pkl": "8" * 64}}
    return S.ProvInputs(
        pins={"SCRATCH_COMMIT": HEAD, "DURABLE_BRANCH": "claude/handoff-tailscreen", "TOOL_SHA256": dict(tools)},
        tool_sha_actual=dict(tools),
        approved={"criteria_sha256": CRIT, "runlog_line": "x", "approved_utc": "2026-10-25T09:00:00Z",
                  "scratch_commit": HEAD, "COEF_90": 2.7, "COEF_60": 2.7, "coef_json_sha256": "f" * 64,
                  "durable_branch": "claude/handoff-tailscreen", "screen_common_sha256": COMMON},
        criteria_sha=CRIT, runlog_text=f"... criteria sha256 {CRIT} ...", run_metas=metas,
        diff_files=list(C.CHANGED_FILES), build_tier={"head": HEAD, "tests": {"t::a": "passed", "t::b": "skipped"}},
        gate_pass={"head": HEAD, "passed": len(C.GATE_REQUIRED) + 1, "failed": 0, "utc": "2026-10-25T09:30:00Z",
                   "tests": {**{f"f.py::{t}": "passed" for t in C.GATE_REQUIRED}, "f.py::test_other": "skipped"}},
        coef={"COEF_90": 2.7, "COEF_60": 2.7, "pooled": {"G_pg_rms": 0.602}, "stack": dict(C.STACK),
              "utc": "2026-10-24T19:00:00Z"},
        coef_sha="f" * 64, coef_mtime_utc="2026-10-24T19:00:01Z", sealed_tree_digest="D", overlay_problems=[],
        v1={"passed": True}, rows=rows or [], lost=lost or {}, screen_common_sha=COMMON,
        ckpt_sha={(r["name"], r["label"]): ZSHA for r in (rows or [])})


def run(classes=None, others=None, skip=(), lost=None, onset=(), prov=True):
    rows = rows_for(classes if classes is not None else base_classes(), others, skip, onset)
    lost = lost or {}
    return S.score(rows, lost, good_prov(rows, lost) if prov else None)


def reading(s, state, bound):
    return s["states"][state][bound]["reading"]


# ------------------------------------------------------------------------------------------------- baseline

def test_baseline_is_final_and_both_pass():
    s = run()
    assert s["status"] == "final", s["provisional_reasons"]
    assert s["provisional_reasons"] == []
    for st in ("s42", "s80"):
        for b in ("BOUND90", "BOUND60"):
            assert reading(s, st, b) == "PASS"
    assert s["outcome"]["line"] == "A"
    assert s["onset_note"].startswith("INFORMATIVE")


def test_replicate_class_rule():
    assert S.replicate_class(1, 0, 0.1, 0.1) == "RAIL"
    assert S.replicate_class(0, 1, 0.1, 0.1) == "RAIL"
    assert S.replicate_class(0, 0, 0.60, 0.60) == "PARKED"  # >= 0.60, no upper end
    assert S.replicate_class(0, 0, 0.99, 1.0) == "PARKED"
    assert S.replicate_class(0, 0, 0.70, 0.49) == "CURED"  # mean 0.595
    assert S.replicate_class(0, 0, 0.50, 0.70) == "PARKED"  # mean 0.60


# ------------------------------------------------------------------------------------------------- every line

@pytest.mark.parametrize("bound,arm", [("BOUND90", "b90"), ("BOUND60", "b60")])
@pytest.mark.parametrize("state", ["s42", "s80"])
@pytest.mark.parametrize("line", ["RAIL", "PARKED", "OTHER", "NOT SEPARATED", "PASS"])
def test_every_state_reading_line(bound, arm, state, line):
    if state == "s80" and line == "NOT SEPARATED":
        # s80 has no NOT SEPARATED line: CONTROL curing itself at s80 leaves a PASS.
        c = base_classes()
        for i in range(3):
            c[nm("s80", "control", i)] = "CURED"
        s = run(c)
        assert reading(s, "s80", bound) == "PASS"
        assert "NOT SEPARATED" not in s["states"]["s80"][bound]["lines_applying"]
        return
    c, o = base_classes(), {}
    if line == "RAIL":
        c[nm(state, arm, 0)] = "RAIL"
    elif line == "PARKED":
        c[nm(state, arm, 0)] = c[nm(state, arm, 1)] = "PARKED"
    elif line == "OTHER":
        o[nm(state, arm, 2)] = 1
    elif line == "NOT SEPARATED":
        c[nm("s42", "control", 0)] = "CURED"  # now 2 CONTROL CURED
    s = run(c, o)
    assert reading(s, state, bound) == line
    assert s["states"][state][bound]["lines_applying"] == [line]
    other_state = "s80" if state == "s42" else "s42"
    assert reading(s, other_state, bound) == "PASS"


def test_one_parked_is_not_parked_and_all_applying_lines_reported():
    c = base_classes()
    c[nm("s42", "b90", 0)] = "PARKED"
    assert reading(run(c), "s42", "BOUND90") == "PASS"
    c[nm("s42", "b90", 1)] = "RAIL"
    c[nm("s42", "b90", 2)] = "PARKED"
    c[nm("s42", "control", 0)] = "CURED"
    s = run(c, {nm("s42", "b90", 1): 3})
    assert s["states"]["s42"]["BOUND90"]["lines_applying"] == ["RAIL", "PARKED", "OTHER", "NOT SEPARATED"]
    assert reading(s, "s42", "BOUND90") == "RAIL"


def test_other_equal_counts_is_not_other():
    o = {nm("s42", "b60", 0): 2, nm("s42", "control", 1): 2}
    assert reading(run(None, o), "s42", "BOUND60") == "PASS"
    o[nm("s42", "b60", 1)] = 1
    assert reading(run(None, o), "s42", "BOUND60") == "OTHER"


@pytest.mark.parametrize("reason", ["RAIL", "PARKED", "OTHER", "NOT SEPARATED"])
@pytest.mark.parametrize("bound,arm", [("BOUND90", "b90"), ("BOUND60", "b60")])
def test_every_verdict_line(reason, bound, arm):
    c, o = base_classes(), {}
    st = "s80" if reason != "NOT SEPARATED" else "s42"
    if reason == "RAIL":
        c[nm(st, arm, 1)] = "RAIL"
    elif reason == "PARKED":
        c[nm(st, arm, 0)] = c[nm(st, arm, 2)] = "PARKED"
    elif reason == "OTHER":
        o[nm(st, arm, 0)] = 1
    else:
        c[nm("s42", "control", 1)] = "CURED"
    s = run(c, o)
    assert s["verdicts"][bound] == {"verdict": "NOT ADOPTED", "reason": reason}


def test_verdict_order_across_states():
    assert S.bound_verdict({"s42": "PARKED", "s80": "RAIL"}) == ("NOT ADOPTED", "RAIL")
    assert S.bound_verdict({"s42": "OTHER", "s80": "PARKED"}) == ("NOT ADOPTED", "PARKED")
    assert S.bound_verdict({"s42": "NOT SEPARATED", "s80": "OTHER"}) == ("NOT ADOPTED", "OTHER")
    assert S.bound_verdict({"s42": "NOT SEPARATED", "s80": "PASS"}) == ("NOT ADOPTED", "NOT SEPARATED")
    assert S.bound_verdict({"s42": "PASS", "s80": "PASS"}) == ("PASSED", None)


# ------------------------------------------------------------------------------------------------- outcomes

def test_outcome_a():
    s = run()
    assert s["outcome"]["line"] == "A"
    assert "BOUND60's PASS is also posture evidence" in " ".join(s["outcome"]["to_maintainer"])


@pytest.mark.parametrize("failing", ["b90", "b60"])
@pytest.mark.parametrize("reason", ["RAIL", "PARKED", "OTHER"])
def test_outcome_b_both_directions(failing, reason):
    c, o = base_classes(), {}
    if reason == "RAIL":
        c[nm("s42", failing, 0)] = "RAIL"
    elif reason == "PARKED":
        c[nm("s80", failing, 0)] = c[nm("s80", failing, 1)] = "PARKED"
    else:
        o[nm("s80", failing, 0)] = 1
    s = run(c, o)
    out = s["outcome"]
    assert out["line"] == "B"
    passed = "BOUND60" if failing == "b90" else "BOUND90"
    failed = "BOUND90" if failing == "b90" else "BOUND60"
    assert out["text"] == f"B. ONE PASSED: {passed} PASSED, {failed} NOT ADOPTED ({reason})."
    joined = " ".join(out["to_maintainer"])
    assert S.ROUTING[reason] in joined
    assert ("posture decision (B with a tail-chain revision, C)" in joined) == (passed == "BOUND60")


def test_outcome_b_cannot_be_not_separated():
    # NOT SEPARATED comes from the shared CONTROL: it hits both bounds unless an earlier line applied to one.
    c = base_classes()
    c[nm("s42", "control", 0)] = "CURED"
    s = run(c)
    assert s["outcome"]["line"] == "C"
    assert s["verdicts"]["BOUND90"]["reason"] == s["verdicts"]["BOUND60"]["reason"] == "NOT SEPARATED"
    assert S.ROUTING["NOT SEPARATED"] in " ".join(s["outcome"]["to_maintainer"])


@pytest.mark.parametrize("r90,r60", list(itertools.product(["RAIL", "PARKED", "OTHER"], repeat=2)))
def test_outcome_c_routing(r90, r60):
    c, o = base_classes(), {}
    for arm, reason in (("b90", r90), ("b60", r60)):
        if reason == "RAIL":
            c[nm("s42", arm, 2)] = "RAIL"
        elif reason == "PARKED":
            c[nm("s42", arm, 0)] = c[nm("s42", arm, 1)] = "PARKED"
        else:
            o[nm("s80", arm, 1)] = 2
    s = run(c, o)
    out = s["outcome"]
    assert out["line"] == "C"
    joined = " ".join(out["to_maintainer"])
    assert f"BOUND90 ({r90}): {S.ROUTING[r90]}." in joined
    assert f"BOUND60 ({r60}): {S.ROUTING[r60]}." in joined
    assert (S.NAMED_RAIL_PARKED in joined) == (r90 == "RAIL" and r60 == "PARKED")
    assert (S.NAMED_BOTH_PARKED in joined) == (r90 == "PARKED" and r60 == "PARKED")
    assert out["to_maintainer"][-1] == S.ALWAYS


def test_named_combinations_text():
    c = base_classes()
    c[nm("s42", "b90", 0)] = "RAIL"
    c[nm("s80", "b60", 0)] = c[nm("s80", "b60", 1)] = "PARKED"
    s = run(c)
    assert "Named combination: the lower bound cleared the rail but not the posture." in s["outcome"]["to_maintainer"]
    c = base_classes()
    for arm in ("b90", "b60"):
        c[nm("s80", arm, 0)] = c[nm("s80", arm, 2)] = "PARKED"
    s = run(c)
    assert "Named combination: neither bound moved deterministic play below the line." in s["outcome"]["to_maintainer"]


# ------------------------------------------------------------------------------------------------- missing / LOST

@pytest.mark.parametrize("arm,own,other", [("b60", "BOUND60", "BOUND90"), ("b90", "BOUND90", "BOUND60")])
def test_lost_bound_replicate_is_rail_for_own_bound_only(arm, own, other):
    name = nm("s80", arm, 1)
    ref = run()
    s = run(skip=(name,), lost={name: "RUNLOG 2026-10-25 12:00 UTC declared LOST"})
    assert s["status"] == "final"
    assert s["replicates"][name]["class"] == "RAIL"
    assert s["verdicts"][own] == {"verdict": "NOT ADOPTED", "reason": "RAIL"}
    assert s["verdicts"][other] == ref["verdicts"][other]
    for st in ("s42", "s80"):
        assert s["states"][st][other] == ref["states"][st][other]


@pytest.mark.parametrize("arm,other", [("b60", "BOUND90"), ("b90", "BOUND60")])
def test_missing_bound_replicate_never_changes_other_bound_under_any_classes(arm, other):
    rng = random.Random(7)
    for _ in range(40):
        c = {n: rng.choice(["RAIL", "PARKED", "CURED"]) for n in S.names()}
        o = {n: rng.choice([0, 0, 1, 2]) for n in S.names()}
        ref = run(c, o)
        name = nm(rng.choice(["s42", "s80"]), arm, rng.randrange(3))
        s = run(c, o, skip=(name,), lost={name: "RUNLOG line"})
        assert s["verdicts"][other] == ref["verdicts"][other]
        for st in ("s42", "s80"):
            assert s["states"][st][other]["reading"] == ref["states"][st][other]["reading"]


def test_lost_control_s42_counts_cured_and_s80_no_rail():
    name = nm("s42", "control", 0)  # was RAIL in the base: losing it makes 2 CURED -> NOT SEPARATED
    s = run(skip=(name,), lost={name: "RUNLOG line"})
    assert s["replicates"][name]["class"] == "CURED"
    assert reading(s, "s42", "BOUND90") == reading(s, "s42", "BOUND60") == "NOT SEPARATED"
    c = base_classes()
    s80 = nm("s80", "control", 0)
    s = run(c, onset=(nm("s80", "control", 2),), skip=(s80,), lost={s80: "RUNLOG line"})
    assert s["replicates"][s80]["onset_any_tailup"] is False
    assert s["onset_note"].startswith("INFORMATIVE")  # r202 (RAIL) and r303 (early flag) still count
    s = run(c, skip=(s80,), lost={s80: "RUNLOG line"})
    assert s["onset_note"].startswith("onset not tested")
    assert reading(s, "s80", "BOUND90") == "PASS"  # the onset note never changes a reading


def test_missing_not_lost_is_incomplete_and_reads_nothing():
    s = run(skip=(nm("s80", "b60", 2),))
    assert s["status"] == "incomplete"
    assert s["incomplete"] == [nm("s80", "b60", 2)]
    assert s["states"] == {} and s["verdicts"] == {} and s["outcome"] is None
    rows = [r for r in rows_for(base_classes()) if not (r["name"] == nm("s42", "control", 0) and r["label"] == R2)]
    s = S.score(rows, {}, good_prov(rows))
    assert s["status"] == "incomplete"
    assert "final" in s["replicates"][nm("s42", "control", 0)]["missing"]
    s = S.score(None, {}, None)
    assert s["status"] == "incomplete" and len(s["incomplete"]) == 18


def test_bound_verdict_not_read_early_when_its_own_six_are_complete():
    # all b90 rows present, one CONTROL replicate missing: still incomplete (CONTROL is shared)
    s = run(skip=(nm("s80", "control", 0),))
    assert s["status"] == "incomplete" and s["verdicts"] == {}


# ------------------------------------------------------------------------------------------------- monotonicity

CLASSES = ("CURED", "PARKED", "RAIL")


def test_monotonicity_pure_grid():
    """Over all 3^3 CONTROL class vectors x CONTROL OTHER counts (and every bound vector / OTHER count): turning a
    CONTROL replicate from CURED to PARKED or RAIL, or raising CONTROL's OTHER count, never moves a bound's reading
    to an earlier line."""
    checked = 0
    for state in ("s42", "s80"):
        for bvec in itertools.product(CLASSES, repeat=3):
            for bo in range(0, 5):
                for cvec in itertools.product(CLASSES, repeat=3):
                    for co in range(0, 5):
                        base = S.line_index(state, S.state_lines(state, bvec, bo, cvec, co)[0])
                        # raise CONTROL OTHER
                        up = S.line_index(state, S.state_lines(state, bvec, bo, cvec, co + 1)[0])
                        assert up >= base, (state, bvec, bo, cvec, co)
                        for i in range(3):
                            if cvec[i] != "CURED":
                                continue
                            for new in ("PARKED", "RAIL"):
                                cv2 = list(cvec)
                                cv2[i] = new
                                after = S.line_index(state, S.state_lines(state, bvec, bo, cv2, co)[0])
                                assert after >= base, (state, bvec, bo, cvec, co, i, new)
                                checked += 1
    assert checked > 0


def test_monotonicity_end_to_end_verdicts():
    """Through rows and the full scorer: more control onset or rail, or more CONTROL OTHER flags, never worsens a
    bound's reading at either state nor its verdict."""
    order = {"NOT ADOPTED": 0, "PASSED": 1}
    vrank = {r: i for i, r in enumerate(S.VERDICT_ORDER)}
    rng = random.Random(11)
    for _ in range(120):
        c = {n: rng.choice(CLASSES) for n in S.names()}
        o = {n: rng.choice([0, 0, 0, 1, 2]) for n in S.names()}
        ref = run(c, o, prov=False)
        ctrl = [n for n in S.names() if "_control_" in n]
        n = rng.choice(ctrl)
        variants = []
        if c[n] == "CURED":
            for new in ("PARKED", "RAIL"):
                c2 = dict(c)
                c2[n] = new
                variants.append((c2, o, ()))
        o2 = dict(o)
        o2[n] = o.get(n, 0) + 1
        variants.append((c, o2, ()))
        if n.startswith("s80"):
            variants.append((c, o, (n,)))  # more onset: an early tail-up flag
        for c2, o2_, onset in variants:
            s = run(c2, o2_, onset=onset, prov=False)
            for b in ("BOUND90", "BOUND60"):
                for st in ("s42", "s80"):
                    assert S.line_index(st, reading(s, st, b)) >= S.line_index(st, reading(ref, st, b))
                v0, v1 = ref["verdicts"][b], s["verdicts"][b]
                assert order[v1["verdict"]] >= order[v0["verdict"]]
                if v0["verdict"] == v1["verdict"] == "NOT ADOPTED":
                    assert vrank[v1["reason"]] >= vrank[v0["reason"]]


def test_onset_note_states_label_counts_and_flags_missing_labels():
    c = base_classes()
    rows = rows_for(c)
    s = S.score(rows, {}, good_prov(rows))
    assert "labels measured of 9: r101 9, r202 9, r303 9" in s["onset_note"] and "INCOMPLETE" not in s["onset_note"]
    rows = [r for r in rows if not (r["name"] == "s80_control_r303" and r["label"] in ("4325376", "4456448"))]
    s = S.score(rows, {}, good_prov(rows))
    assert s["status"] == "provisional"
    assert "r303 7" in s["onset_note"] and "INCOMPLETE: s80_control_r303" in s["onset_note"]
    assert any("onset note read on 7 of 9 labels for s80_control_r303" in x for x in s["provisional_reasons"])


def test_onset_note_counts_any_label_and_needs_two():
    c = base_classes()
    c[nm("s80", "control", 0)] = c[nm("s80", "control", 1)] = "CURED"
    assert run(c)["onset_note"].startswith("onset not tested")
    s = run(c, onset=(nm("s80", "control", 0),))
    assert s["onset_note"].startswith("onset not tested")
    s = run(c, onset=(nm("s80", "control", 0), nm("s80", "control", 2)))
    assert s["onset_note"].startswith("INFORMATIVE")
    assert reading(s, "s80", "BOUND60") == "PASS"


# ------------------------------------------------------------------------------------------------- independence

@pytest.mark.parametrize("changed,watched", [("b60", "BOUND90"), ("b90", "BOUND60")])
def test_bound_independence(changed, watched):
    rng = random.Random(3 if changed == "b60" else 5)
    for _ in range(60):
        c = {n: rng.choice(CLASSES) for n in S.names()}
        o = {n: rng.choice([0, 0, 1, 2]) for n in S.names()}
        ref = run(c, o, prov=False)
        rows = rows_for(c, o)
        for r in rows:
            if f"_{changed}_" in r["name"]:
                r["cmd"] = repr(rng.uniform(-1, 1))
                r["tailup_flags"] = str(rng.choice([0, 0, 1, 7]))
                r["other_flags"] = str(rng.choice([0, 1, 5]))
        s = S.score(rows, {}, None)
        assert s["verdicts"][watched] == ref["verdicts"][watched]
        for st in ("s42", "s80"):
            assert s["states"][st][watched] == ref["states"][st][watched]


# ------------------------------------------------------------------------------------------------- PROVISIONAL

def _mut(path, value):
    def f(p):
        obj = p
        keys = path.split(".")
        for k in keys[:-1]:
            obj = getattr(obj, k) if not isinstance(obj, dict) else obj[k]
        if isinstance(obj, dict):
            obj[keys[-1]] = value
        else:
            setattr(obj, keys[-1], value)
    return f


def _meta(field_, value, name="s42_b60_r202"):
    def f(p):
        m = p.run_metas[name]
        if "." in field_:
            a, b = field_.split(".")
            m[a][b] = value
        else:
            m[field_] = value
    return f


def _row(field_, value):
    def f(p):
        p.rows[5][field_] = value
    return f


PROV_TRIGGERS = {
    "pin SCRATCH_COMMIT unfilled": _mut("pins.SCRATCH_COMMIT", None),
    "pin DURABLE_BRANCH unfilled": _mut("pins.DURABLE_BRANCH", None),
    "tool pin unfilled": lambda p: p.pins["TOOL_SHA256"].pop("gate.sh"),
    "tool sha differs": lambda p: p.tool_sha_actual.__setitem__("score_tailscreen.py", "0" * 64),
    "APPROVED missing": _mut("approved", None),
    "APPROVED COEF pin unfilled": _mut("approved.COEF_60", None),
    "criteria.txt missing": _mut("criteria_sha", None),
    "criteria sha != APPROVED": _mut("criteria_sha", "d" * 64),
    "criteria sha not in RUNLOG": _mut("runlog_text", "nothing here"),
    "run_meta HEAD other": _meta("clone.head", "b" * 40),
    "run_meta dirty clone": _meta("clone.status", " M environments/shared/mirror_kl_ppo.py"),
    "run_meta start sha": _meta("start.zip_sha256", "0" * 64),
    "run_meta arm": _meta("arm", "b90"),
    "run_meta coef": _meta("coef", 2.8),
    "run_meta b": _meta("b", 0.90),
    "run_meta control coef": _meta("coef", 2.7, "s80_control_r101"),
    "run_meta seed": _meta("rep_seed", 101),
    "run_meta stack": _meta("stack", {**C.STACK, "numpy": "2.1.3"}),
    "run_meta status": _meta("status", "running"),
    "run_meta missing with panels": lambda p: p.run_metas.__setitem__("s42_b60_r202", None),
    "diff file set": _mut("diff_files", list(C.CHANGED_FILES) + ["environments/shared/train_base.py"]),
    "diff unreadable": _mut("diff_files", None),
    "build tier missing": _mut("build_tier", None),
    "build tier failed": _mut("build_tier", {"head": HEAD, "tests": {"t::a": "passed", "t::b": "failed"}}),
    "build tier other head": _mut("build_tier.head", "b" * 40),
    "gate missing": _mut("gate_pass", None),
    "gate failed": _mut("gate_pass.failed", 1),
    "gate other head": _mut("gate_pass.head", "b" * 40),
    "gate no per-test statuses": lambda p: p.gate_pass.pop("tests"),
    "gate T1b skipped": lambda p: p.gate_pass["tests"].__setitem__(
        "f.py::test_tail3_t1b_coef_zero_trains_bit_for_bit_as_no_bound", "skipped"),
    "gate T2 absent": lambda p: p.gate_pass["tests"].pop("f.py::test_tail3_t2_same_seed_reproducibility"),
    "gate T7b deselected": lambda p: p.gate_pass["tests"].__setitem__(
        "f.py::test_tail3_t7b_a_bound_zip_loads_after_one_update", "deselected"),
    "gate another test failed": lambda p: p.gate_pass["tests"].__setitem__("f.py::test_other", "failed"),
    "screen_common sha != APPROVED": _mut("screen_common_sha", "4" * 64),
    "APPROVED screen_common pin unfilled": _mut("approved.screen_common_sha256", None),
    "panel row of another attempt (zip sha)": _row("zip_sha256", "7" * 64),
    "panel checkpoint gone from runs": lambda p: p.ckpt_sha.__setitem__((p.rows[5]["name"], p.rows[5]["label"]), None),
    "final row != run_meta final_sha256": lambda p: p.run_metas["s42_control_r101"]["final_sha256"].__setitem__(
        "stage1_final.zip", "6" * 64),
    "onset label missing (s80 control)": lambda p: p.rows.remove(
        next(r for r in p.rows if r["name"] == "s80_control_r202" and r["label"] == "4456448")),
    "coef missing": _mut("coef", None),
    "coef breaks rule": _mut("coef.COEF_90", 2.6),
    "coef != APPROVED": _mut("approved.COEF_60", 2.8),
    "coef sha != APPROVED": _mut("coef_sha", "e" * 64),
    "coef other stack": _mut("coef.stack", {**C.STACK, "torch": "2.12.0+cpu"}),
    "coef written after start (utc)": _mut("coef.utc", "2026-10-25T11:00:00Z"),
    "coef written after start (mtime)": _mut("coef_mtime_utc", "2026-10-25T11:00:00Z"),
    "panel tree digest": _row("measure_tree_digest", "X"),
    "panel seeds": _row("seeds", "3042-3060"),
    "panel stack": _row("stack", json.dumps({**C.STACK, "mujoco": "3.9.0"})),
    "overlay sha": _mut("overlay_problems", ["overlay x differs"]),
    "V1 missing": _mut("v1", None),
    "V1 not passed": _mut("v1.passed", False),
}


@pytest.mark.parametrize("trigger", sorted(PROV_TRIGGERS))
def test_each_provisional_trigger(trigger):
    rows = rows_for(base_classes())
    p = good_prov(rows)
    assert S.provisional_reasons(p) == []
    PROV_TRIGGERS[trigger](p)
    reasons = S.provisional_reasons(p)
    assert reasons, trigger
    s = S.score(rows, {}, p)
    assert s["status"] == "provisional"
    assert s["outcome"]["line"] == "A"  # still read and printed, just not acted on


def test_provisional_skips_lost_runs_meta():
    name = nm("s42", "b90", 0)
    rows = rows_for(base_classes(), skip=(name,))
    p = good_prov(rows, {name: "RUNLOG line"})
    p.run_metas[name] = None
    assert S.provisional_reasons(p) == []


# ------------------------------------------------------------------------------------------------- inputs / CLI

def _write_panels(path, rows):
    cols = list(rows[0].keys())
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def test_cli_exit_codes(tmp_path, capsys):
    root = tmp_path / "screen"
    root.mkdir()
    out = tmp_path / "out"
    # no panels.csv: incomplete
    assert S.main(["--root", str(root), "--panels", str(root / "panels.csv"), "--runs", str(root / "runs"),
                   "--out", str(out)]) == S.EXIT_INCOMPLETE
    assert json.loads((out / "score.json").read_text())["status"] == "incomplete"
    # complete rows, but nothing pinned or approved here: provisional (read and printed)
    rows = rows_for(base_classes())
    _write_panels(root / "panels.csv", rows)
    assert S.main(["--root", str(root), "--panels", str(root / "panels.csv"), "--runs", str(root / "runs"),
                   "--out", str(out)]) == S.EXIT_PROVISIONAL
    s = json.loads((out / "score.json").read_text())
    assert s["status"] == "provisional" and s["outcome"]["line"] == "A"
    assert "OUTCOME: A. BOTH PASSED." in (out / "score.txt").read_text()
    # duplicate row: input error
    _write_panels(root / "panels.csv", rows + [rows[0]])
    assert S.main(["--root", str(root), "--panels", str(root / "panels.csv"), "--out", str(out)]) == S.EXIT_INPUT
    # malformed number: input error
    bad = copy.deepcopy(rows)
    bad[7]["cmd"] = "n/a"
    _write_panels(root / "panels.csv", bad)
    assert S.main(["--root", str(root), "--panels", str(root / "panels.csv"), "--out", str(out)]) == S.EXIT_INPUT
    # LOST.txt naming a non-screen replicate, or without a RUNLOG line: input error
    _write_panels(root / "panels.csv", rows)
    (root / "LOST.txt").write_text("s42_b60_r404 RUNLOG line 600\n")
    assert S.main(["--root", str(root), "--panels", str(root / "panels.csv"), "--out", str(out)]) == S.EXIT_INPUT
    (root / "LOST.txt").write_text("s42_b60_r101\n")
    assert S.main(["--root", str(root), "--panels", str(root / "panels.csv"), "--out", str(out)]) == S.EXIT_INPUT
    # a missing column: input error
    _write_panels(root / "panels.csv", [{k: v for k, v in r.items() if k != "other_flags"} for r in rows])
    (root / "LOST.txt").unlink()
    assert S.main(["--root", str(root), "--panels", str(root / "panels.csv"), "--out", str(out)]) == S.EXIT_INPUT


def test_report_only_dose_ordering_and_half_point():
    c = base_classes()
    s = run(c)
    d = s["report_only"]["dose_ordering"]["s42"]
    assert d["arms"]["control"]["n"] == 3
    # control s42 has RAIL/PARKED: mean above the CURED bounds; b60 == b90 (all CURED): ordered
    assert d["ordered_b60_le_b90_le_control"] is True
    hp = s["report_only"]["half_point"][nm("s42", "b60", 0)]
    assert set(hp) == {"4587520", "4718592"} and hp["4587520"]["cmd"] == 0.3


def test_scorer_imports_no_torch_or_environments():
    import subprocess
    import sys

    code = ("import importlib.util,sys;"
            f"s=importlib.util.spec_from_file_location('x','{TOOLS / 'score_tailscreen.py'}');"
            "m=importlib.util.module_from_spec(s);s.loader.exec_module(m);"
            "print(any(k=='torch' or k.startswith('environments') for k in sys.modules))")
    out = subprocess.run([sys.executable, "-I", "-B", "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"
