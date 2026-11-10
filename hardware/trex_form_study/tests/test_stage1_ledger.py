"""Tests for the Stage 1 margin ledger (designlab/stage1/ledger.py; STAGE1_PLAN.md section 5).

Run from hardware/trex_form_study (the fit's venv):  python -m pytest tests/test_stage1_ledger.py -q
Every test is fast: the bisection and its endpoint rules on synthetic judges, the margins' intervals and Stage 0
values, every variant applied to a real plant and restored bit for bit, the nominal evaluation against the panels' own
episode, and a tiny real sweep (episodes cut below a second) with its records and resume.  The tests that build a plant
or run episodes need the run directory's references and transitions (git-ignored; ``python -m designlab.stage1
prepare``) and skip without them.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import math
import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import designlab  # noqa: E402,F401  (selects the GL backend before mujoco renders)
from designlab import catalog  # noqa: E402
from designlab.stage1 import PROBE_BODIES, control, drive, env, ledger, panels, tune  # noqa: E402
from designlab.stage1.__main__ import main as cli  # noqa: E402

BODY = "60-N-T-10"
RUN = ROOT / "stage1" / "runs" / "2026-11-08"
needs_run = pytest.mark.skipif(not (RUN / BODY / "refs" / "nominal.npz").exists(),
                               reason="the run directory's references are not generated (designlab.stage1 prepare)")
COND = {c.key: c for c in panels.conditions()}


# ------------------------------------------------------------------------------------------------
# The bisection on synthetic judges
# ------------------------------------------------------------------------------------------------
def run_bisect(search, ok, halvings=ledger.HALVINGS):
    b = ledger.Bisect(search, "success", halvings)
    while not b.done:
        v = b.need()
        b.feed(v, ok(v))
    return b


def test_bisection_brackets_the_threshold_to_its_resolution():
    """7 halvings after both ends: the threshold is the last value that works, the nearest failing value 1/128 of the
    interval away, both sides of the true threshold; downward and upward searches alike."""
    grip = ledger.Search("walk", "walk/nominal", "friction", 0.9, 0.1)
    b = run_bisect(grip, lambda mu: mu >= 0.53)
    r = b.result()
    assert r["status"] == "bisected" and r["halvings"] == 7 and len(r["values"]) == 9
    assert r["threshold"] >= 0.53 > r["fails"] and r["resolution"] == pytest.approx(0.8 / 128)
    assert [x["value"] for x in r["values"][:3]] == [0.9, 0.1, 0.5] and grip.seek == "lowest"
    push = ledger.Search("standing_forward", "push/standing_forward", "push_dv", 0.0, 0.974)
    r = run_bisect(push, lambda dv: dv <= 0.37).result()
    assert r["threshold"] <= 0.37 < r["fails"] and r["fails"] - r["threshold"] == pytest.approx(0.974 / 128)
    assert push.seek == "largest"
    for t in (0.12, 0.5, 0.88):  # the true threshold is always inside the final bracket
        r = run_bisect(grip, lambda mu, t=t: mu >= t).result()
        assert r["fails"] < t <= r["threshold"] + 1e-12


def test_integer_bisection_ends_one_step_wide():
    """Latency in whole physics steps: halved until the bracket is one step (at most 2 + 4 values for 0-10)."""
    lat = ledger.Search("walk", "walk/nominal", "latency_steps", 0, 10, integer=True)
    assert lat.max_values() == 6 and ledger.Search("w", "c", "kp", 1.0, 0.25).max_values() == 9
    for k in range(0, 10):
        r = run_bisect(lat, lambda x, k=k: x <= k).result()
        assert r["status"] == "bisected" and r["threshold"] == k and r["fails"] == k + 1, k
        assert len(r["values"]) <= 6 and all(isinstance(x["value"], int) for x in r["values"])
    assert run_bisect(lat, lambda x: x <= 10).result()["status"] == "works across the interval"


def test_endpoint_rules():
    """Fails at nominal after one value; works across the interval after two; a done search needs nothing; a value
    out of turn is refused."""
    s = ledger.Search("walk", "walk/nominal", "kp", 1.0, 0.25)
    b = run_bisect(s, lambda f: False)
    assert b.result()["status"] == "fails at nominal" and b.fails == 1.0 and b.works is None and b.need() is None
    assert len(b.trail) == 1
    b = run_bisect(s, lambda f: True)
    assert b.result()["status"] == "works across the interval" and b.works == 0.25 and b.fails is None
    assert len(b.trail) == 2
    b = ledger.Bisect(s, "survival")
    with pytest.raises(ValueError):
        b.feed(0.5, True)
    with pytest.raises(ValueError):
        ledger.Bisect(s, "sometimes")


class FakeRow:
    name = "R9/1"

    def __init__(self, hard_only=False):
        self.hard_only = hard_only

    def runs(self, cond):
        return not self.hard_only or cond.floor == "hard"


def episode(seed, outcome):
    """A ledger episode summary: outcome "ok", a list of failed terms, or ("fell", candidates)."""
    if outcome == "ok":
        return dict(seed=seed, fell=False, succeeded=True, failed_terms=[], fall_cause=None, fall_candidates=None)
    if isinstance(outcome, tuple):
        return dict(seed=seed, fell=True, succeeded=False, failed_terms=["survived"], fall_cause="tilt",
                    fall_candidates=list(outcome[1]))
    return dict(seed=seed, fell=False, succeeded=False, failed_terms=list(outcome), fall_cause=None,
                fall_candidates=None)


def entry(cond, var, outcomes, relax=()):
    """An evaluation of the ledger's 3 seeds with these outcomes (``episode``)."""
    return ledger.evaluation_entry(cond, var, [episode(s, o) for s, o in zip(ledger.SEEDS, outcomes, strict=True)],
                                   "valid", relax)


def outcomes(works, survives, cands=()):
    """Per-seed outcomes that work, survive but fail a term, or fall (with these candidates)."""
    if not survives:
        return [("fell", cands), "ok", "ok"]
    return ["ok"] * 3 if works else [["speed"], "ok", "ok"]


def lookup_of(cache):
    return lambda k, v, relax=(): cache.get((k, v.key, tuple(relax)))


def synthetic(margin, outcome, lift=None):
    """A margin run fed by ``outcome(condition, variant) -> (works, survives)`` through its lookup (and a lift by
    ``lift(condition, variant, relaxation) -> survives``; falls name 'slip' as their candidate); returns the run and
    every evaluation made, in order."""
    conds = {k: COND[k] for k in margin.conditions()}
    mr = ledger.MarginRun(FakeRow(), margin, conds, {p.name: ledger.Variant(p.kind, p.value) for p in margin.points})
    cache, made = {}, []
    while True:
        need = mr.advance(lookup_of(cache))
        if not need:
            return mr, cache, made
        for k, var, relax in dict.fromkeys(need):
            if relax:
                cache[(k, var.key, relax)] = entry(k, var, outcomes(True, lift(k, var, relax[0])), relax)
            else:
                w, s = outcome(k, var)
                cache[(k, var.key, ())] = entry(k, var, outcomes(w, s, ("slip",)))
            made.append((k, var.key, relax))


def test_success_and_survival_share_their_evaluations():
    """Both criteria from one set of evaluations: the shared values run once; the survival search goes on past the
    success threshold; a fresh run replays the recorded evaluations to the same results without new ones."""
    m = ledger.Margin("grip", "Grip", "mu", searches=(ledger.Search("walk", "walk/nominal", "friction", 0.9, 0.1),))

    def outcome(k, var):
        mu = 0.9 if var.kind is None else var.value
        return mu >= 0.6, mu >= 0.3

    mr, cache, made = synthetic(m, outcome, lift=lambda k, v, r: True)
    values = [x for x in made if not x[2]]
    assert len(made) == len(set(made)) and 9 < len(values) <= 16
    ok, surv = mr.bisects[("walk", "success")], mr.bisects[("walk", "survival")]
    assert ok.result()["threshold"] >= 0.6 > ok.result()["fails"] and surv.result()["threshold"] >= 0.3 > surv.fails
    assert [v for v, _ in ok.trail[:2]] == [v for v, _ in surv.trail[:2]] == [0.9, 0.1]
    assert made[0] == ("walk/nominal", "nominal", ())  # the nominal value is the panels' condition as it is
    assert [x for x in made if x[2]] == [("walk/nominal", f"friction={surv.fails!r}", ("slip",))]  # LIFT_RULE
    again = ledger.MarginRun(FakeRow(), m, {"walk/nominal": COND["walk/nominal"]}, {})
    assert again.advance(lookup_of(cache)) == [] and again.done
    assert again.bisects[("walk", "survival")].result() == surv.result()


def test_a_value_that_failed_to_run_makes_its_search_wait():
    m = ledger.Margin("kp", "kp", "kp", searches=(ledger.Search("walk", "walk/nominal", "kp", 1.0, 0.25),))
    mr = ledger.MarginRun(FakeRow(), m, {"walk/nominal": COND["walk/nominal"]}, {})
    cache = {("walk/nominal", "nominal", ()): dict(errors=1, works=None, survives=None, episodes=[])}
    assert mr.advance(lookup_of(cache)) == [] and not mr.done
    assert mr.waiting == {("walk", "success"), ("walk", "survival")}
    piles = ledger.margins(BODY)[8]
    hard = ledger.MarginRun(FakeRow(hard_only=True), piles, {k: COND[k] for k in piles.conditions()}, {})
    assert hard.done and hard.bisects == {} and not any(hard.applicable.values())  # R0, R1: no pile search


def test_worse_of_two_directions():
    """The report's turn: the direction with less margin (a higher lowest mu); fails at nominal is the worst; an
    unfinished search leaves it incomplete."""
    a = ledger.Search("turn_left", "turn/walking_0.35_left", "friction", 0.9, 0.1)
    b = ledger.Search("turn_right", "turn/walking_0.35_right", "friction", 0.9, 0.1)
    ra = run_bisect(a, lambda mu: mu >= 0.40).result()
    rb = run_bisect(b, lambda mu: mu >= 0.55).result()
    w = ledger.worse({"turn_left": ra, "turn_right": rb}, [a, b])
    assert w["search"] == "turn_right" and w["threshold"] == rb["threshold"]
    rf = run_bisect(a, lambda mu: False).result()
    assert ledger.worse({"turn_left": rf, "turn_right": rb}, [a, b])["search"] == "turn_left"
    assert ledger.worse({"turn_left": ledger.Bisect(a, "success").result(), "turn_right": rb},
                        [a, b])["status"] == "incomplete"


def test_labels_at_the_fails_end():
    """LABEL_RULE: success failures by their new terms' classes, falls by LIFT_RULE (body-limited by the lifts under
    which all 3 seeds survive, else controller-limited; pending until the lifts are known), falls with no candidate
    controller-limited; the falls' label on a tie (panels.FAILURE_LABEL_RULE); a survival search takes the falls'."""
    v = ledger.Variant("friction", 0.3)
    e = entry("walk/nominal", v, [["slip_worst", "speed"], ["slip_worst"], ("fell", ["slip", "joint_range"])])
    assert ledger.lifts_of(e) == (["slip", "joint_range"], [])
    lab = ledger.binding(e, "success")
    assert lab["label"] == "body-limited: slip_worst" and lab["falls"] == 1 and lab["success_failures"] == 2
    assert ledger.binding(e, "survival")["label"] == "falls: lifts of slip, joint_range pending (LIFT_RULE)"
    lifted = {"slip": entry("walk/nominal", v, ["ok"] * 3, ("slip",)),
              "joint_range": entry("walk/nominal", v, [("fell", ["slip"]), "ok", "ok"], ("joint_range",))}
    lab = ledger.binding(e, "survival", lifted=lifted)
    assert lab["label"] == "body-limited: slip (lifted, all 3 seeds survive)"
    assert lab["lifts"] == {"slip": dict(survives=True, survived=3), "joint_range": dict(survives=False, survived=2)}
    lifted["slip"] = lifted["joint_range"]
    assert ledger.binding(e, "survival", lifted=lifted)["label"].startswith(
        "controller-limited (lifting slip, joint_range saves none")
    e = entry("walk/nominal", v, [("fell", ["cop_at_pad_edge_with_ankle_saturated"]), ("fell", []), ["speed"]])
    lab = ledger.binding(e, "success")
    assert lab["label"].startswith("controller-limited (falls with no relaxable")
    assert "untestable: cop_at" in lab["label"]
    e = entry("walk/nominal", v, [["speed", "duty"], ["cadence"], "ok"])
    assert ledger.binding(e, "success")["label"] == "controller-limited"
    tie = entry("walk/nominal", v, [["slip_worst"], ("fell", []), "ok"])  # one each: the falls' label
    assert ledger.binding(tie, "success")["label"].startswith("controller-limited (falls")
    assert ledger.binding(None, "success") is None


def test_works_is_judged_against_the_nominal_value():
    """WORKS_RULE: a value works if no seed falls and none fails a term it passed at nominal; at nominal itself,
    works is survives (60-N-T-10's R2/42 nominal walk: seed 3042 fails the peak GRF alone); the outright reading is
    kept beside."""
    nom = entry("walk/nominal", ledger.NOMINAL, [["grf"], "ok", "ok"])
    assert nom["succeeds_all"] is False and ledger.works(nom, nom) is True
    same = entry("walk/nominal", ledger.Variant("kp", 0.5), [["grf"], "ok", "ok"])
    assert ledger.works(same, nom) is True and same["succeeds_all"] is False
    assert ledger.works(entry("walk/nominal", ledger.Variant("kp", 0.4), [["grf", "slip_worst"], "ok", "ok"]),
                        nom) is False
    assert ledger.works(entry("walk/nominal", ledger.Variant("kp", 0.4), ["ok", ["grf"], "ok"]), nom) is False
    assert ledger.works(entry("walk/nominal", ledger.Variant("kp", 0.4), ["ok", "ok", ("fell", [])]), nom) is False
    assert ledger.works(None, nom) is None
    assert ledger.binding(entry("walk/nominal", ledger.Variant("kp", 0.4), [["grf", "slip_worst"], "ok", "ok"]),
                          "success", nom)["failed_terms"] == {"slip_worst": dict(count=1, cls="hardware")}
    m = ledger.Margin("kp", "kp", "kp", searches=(ledger.Search("walk", "walk/nominal", "kp", 1.0, 0.25),))

    def outcome(k, var):
        f = 1.0 if var.kind is None else var.value
        return f >= 0.5, True

    cache = {}
    mr = ledger.MarginRun(FakeRow(), m, {"walk/nominal": COND["walk/nominal"]}, {})
    while True:
        need = mr.advance(lookup_of(cache))
        if not need:
            break
        for k, var, _ in need:  # every seed 3042 fails the GRF, as at nominal; below 0.5 seed 3043 slips too
            w, _ = outcome(k, var)
            cache[(k, var.key, ())] = entry(k, var, [["grf"], "ok" if w else ["slip_worst"], "ok"])
    r = mr.bisects[("walk", "success")].result()
    assert r["status"] == "bisected" and r["threshold"] >= 0.5 > r["fails"]


def test_a_point_is_judged_against_its_conditions_nominal_evaluation():
    """A point needs its condition's nominal evaluation (WORKS_RULE) and its falls' lifts (LIFT_RULE)."""
    m = ledger.Margin("mass", "mass", "mass", points=(ledger.Point("x", "walk/nominal", "mass", 1.2),))
    mr = ledger.MarginRun(FakeRow(), m, {"walk/nominal": COND["walk/nominal"]}, {"x": ledger.Variant("mass", 1.2)})
    need = mr.advance(lookup_of({}))
    assert [(k, v.key, r) for k, v, r in need] == [("walk/nominal", "mass=1.2", ()), ("walk/nominal", "nominal", ())]
    cache = {("walk/nominal", "nominal", ()): entry("walk/nominal", ledger.NOMINAL, [["grf"], "ok", "ok"]),
             ("walk/nominal", "mass=1.2", ()): entry("walk/nominal", ledger.Variant("mass", 1.2),
                                                     [["grf"], "ok", ("fell", ["leg_drive_at_torque_limit"])])}
    need = mr.advance(lookup_of(cache))
    assert [(v.key, r) for _, v, r in need] == [("mass=1.2", ("drive_torque",))] and not mr.done
    cache[("walk/nominal", "mass=1.2", ("drive_torque",))] = entry("walk/nominal", ledger.Variant("mass", 1.2),
                                                                   ["ok"] * 3, ("drive_torque",))
    assert mr.advance(lookup_of(cache)) == [] and mr.done
    p = ledger._point_results(mr, lookup_of(cache))["x"]
    assert p["works"] is False and p["survives"] is False and p["works_outright"] is False
    assert p["binding"]["label"] == "body-limited: drive_torque (lifted, all 3 seeds survive)"


# ------------------------------------------------------------------------------------------------
# The margins
# ------------------------------------------------------------------------------------------------
def test_margins_follow_plan_section_5():
    """Every row of plan section 5's ledger with its stated interval, condition and Stage 0 value; the stop is not
    run (no stop references at other decelerations)."""
    for body in PROBE_BODIES:
        ms = {m.name: m for m in ledger.margins(body)}
        assert list(ms) == list(ledger.MARGIN_NAMES)
        g = ms["grip"]
        assert [(s.condition, s.kind, s.nominal, s.far) for s in g.searches] == [
            ("walk/nominal", "friction", 0.9, 0.1), ("turn/walking_0.35_left", "friction", 0.9, 0.1),
            ("turn/walking_0.35_right", "friction", 0.9, 0.1), ("walk/brisk", "friction", 0.9, 0.1)]
        assert g.worse == (("turn", ("turn_left", "turn_right")),)
        p = ms["push"]
        assert len(p.searches) == 8 and {s.condition for s in p.searches} == {
            c.key for c in panels.conditions() if c.behavior == "push"}
        assert all(s.nominal == 0.0 and s.far == pytest.approx(2 * env.PUSH_BOUND_M_S[body]) for s in p.searches)
        assert ms["stop"].not_run and not ms["stop"].searches
        assert [(s.condition, s.kind, s.nominal, s.far) for s in ms["drive_speed"].searches] == [
            ("walk/fr05", "speed_scale", 1.0, 0.25)]
        assert [(s.condition, s.kind, s.far) for s in ms["drive_torque"].searches] == [("walk/nominal", "torque_scale",
                                                                                        0.25)]
        assert [(s.kind, s.nominal, s.far) for s in ms["kp"].searches] == [("kp", 1.0, 0.25)]
        lat = ms["latency"].searches[0]
        assert (lat.kind, lat.nominal, lat.far, lat.integer) == ("latency_steps", 0, 10, True)
        assert [(s.kind, s.far) for s in ms["mass"].searches] == [("mass", 1.3)]
        assert [(x.kind, x.value) for x in ms["mass"].points] == [("mass", 1.1),
                                                                  ("mass", ledger.ROBOT_MASS_POINT[body])]
        assert [(s.condition, s.far) for s in ms["pile_damping"].searches] == [
            ("stand/low", 0.5), ("stand/low", 2.0), ("carpet/low", 0.5), ("carpet/low", 2.0)]
        assert "plan deviation, decision 33" in ms["stop"].not_run
        assert sorted((x.condition, x.value) for x in ms["voltage"].points) == sorted(
            (c, v) for c in ("walk/nominal", "walk/fr05") for v in (25.2, 22.2, 21.0))
        st0 = ledger.STAGE0[body]
        assert st0["push_forward"] == env.PUSH_BOUND_M_S[body]
        assert ledger.stage0(body, "grip")["values"]["turn"] == st0["grip_turn"]
        assert ledger.stage0(body, "kp")["values"] == {}
    # plan section 5's grip cell per body, walk / turn / brisk (ROUND2_REVIEW.md 7.4 and 7.6)
    assert [tuple(ledger.STAGE0[b][f"grip_{k}"] for k in ("walk_nominal", "turn", "walk_brisk"))
            for b in ("60-N-T-10", "60-N-H-10", "75-M-H-10")] == [(0.48, 0.54, 0.75), (0.47, 0.53, 0.73),
                                                                  (0.36, 0.41, 0.56)]
    assert ledger.STAGE0["60-N-H-10"]["grip_walk_brisk"] == 0.73 and ledger.STAGE0["75-M-H-10"]["stop_one_foot"] == 1.41
    names = [m.name for m in ledger.select_margins(ledger.margins(BODY), ["model_doubts", "grip"])]
    assert names == ["grip", "kp", "latency", "mass", "pile_damping"]
    with pytest.raises(ValueError):
        ledger.select_margins(ledger.margins(BODY), ["swim"])
    s = ledger.margins(BODY)[0].searches[0]
    assert s.variant(0.9) is ledger.NOMINAL and s.variant(0.5) == ledger.Variant("friction", 0.5)
    push = ledger.margins(BODY)[1].searches[0]
    assert push.variant(0.0) == ledger.Variant("push_dv", 0.0)  # a zero pulse, not the panels' half bound
    with pytest.raises(ValueError):
        ledger.Variant("gravity", 2.0)


def test_stage0_is_the_screens_values():
    """STAGE0_RULE: every grip, drive speed and drive torque value is the round-two screens' rounded to the plan's
    digits: the turning screen's single-support friction need (the worse foot; the worse turn direction), the knee's
    sizing speed ratio (with the 15 % margin, and without it), the worst sustained-torque ratio (the larger of the
    nominal walk's RMS and the one-foot hold) and the walk's worst RMS ratio."""
    for body in PROBE_BODIES:
        st0, sc = ledger.STAGE0[body], ledger.screens(body)
        for k in ("walk_nominal", "turn", "walk_brisk"):
            assert round(sc["grip"][k], 2) == st0[f"grip_{k}"], (body, k)
        assert round(sc["drive_speed"]["knee_speed_ratio"], 2) == st0["drive_speed_knee"]
        assert round(sc["drive_speed"]["knee_peak_over_rated"], 2) == st0["drive_speed_knee_no_margin"]
        assert round(sc["drive_torque"]["cont_ratio"], 2) == st0["drive_torque_sustained"]
        assert round(sc["drive_torque"]["walk_rms_ratio"], 2) == st0["drive_torque_walk_rms"]
        g = ledger.stage0(body, "grip")
        assert g["values"]["turn_left"] == g["values"]["turn"] == st0["grip_turn"] and g["screens"]["files"]
        assert ledger.stage0(body, "drive_speed")["without_margin"]["walk_fr05"] == st0["drive_speed_knee_no_margin"]
        assert ledger.stage0(body, "drive_torque")["walk_rms"]["walk_nominal"] == st0["drive_torque_walk_rms"]
    assert ledger.screens("75-M-H-10")["drive_torque"]["drive"] == "hip_yaw"  # its 0.77 is the one-foot hold
    assert ledger.screens("75-M-H-10")["drive_torque"]["by"] == "one-foot hold"
    assert ledger.screens("75-M-H-10")["drive_torque"]["walk_rms_drive"] == "neck"


def test_the_trunk_mass_points():
    """MASS_RULE: the trunk x 1.1 and the trunk factor that adds 10 % of the robot's mass, from each body's trunk and
    robot mass (the screened MJCF's trunk body and its subtree, the whole robot)."""
    for body, (trunk, robot) in ledger.MASSES.items():
        xml, _ = ledger.plant.model_files(body)
        m = mujoco.MjModel.from_xml_path(str(xml))
        t = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "trunk")
        assert m.body_parentid[t] == 0  # the root: its subtree is the whole robot
        assert m.body_mass[t] == pytest.approx(trunk, abs=1e-4)
        assert m.body_subtreemass[t] == pytest.approx(robot, abs=1e-4)
        f = ledger.ROBOT_MASS_POINT[body]
        assert (f - 1.0) * trunk / robot == pytest.approx(0.10, abs=5e-4) and 1.2 < f < ledger.MASS_FAR
        assert 0.04 < 0.1 * trunk / robot < 0.045  # the trunk x 1.1: about +4.3 % of the robot


def test_rows_and_the_bus_voltage_rule():
    """'R2' is the body-level G5 selection's CMA-ES seed (env.tuned_gains) and 'R3' is R3/42; a voltage above every
    leg drive's nominal voltage changes nothing (60 cm: 24 V drives), below it every leg drive's speeds."""
    want = {"60-N-T-10": "R2/42", "60-N-H-10": "R2/43", "75-M-H-10": "R2/43"}
    for body, row in want.items():
        if (RUN / body / "tune.json").exists():
            assert ledger.resolve_rows(body, RUN / body, ("R2", "R3", "R2/43")) == [row, "R3/42"] + (
                [] if row == "R2/43" else ["R2/43"])
    assert not ledger.bus_changes("60-N-T-10", 25.2) and ledger.bus_changes("60-N-T-10", 22.2)
    assert ledger.bus_changes("75-M-H-10", 25.2)  # its hip pitch is a 48 V drive: credit up to 48 V
    a = catalog.actuator("myactuator_rmd_x4_p36_36_e_l")
    assert a.speed_at(22.2) == pytest.approx(8.04, abs=0.005)  # plan section 5's knee at 22.2 V


def test_r3_is_left_out_for_a_body_judged_with_r2(tmp_path, monkeypatch):
    """ROWS_RULE and train.R2_JUDGED_RULE: 'R3' leaves R3/42 out, with the reason, when the body's seed 42 failed G6c
    or FIX_RULE stopped one of its R3 runs; R3/42 named explicitly is kept; G6c not reached keeps R3."""
    from designlab.stage1 import train

    monkeypatch.delenv("STAGE1_POLICY_RUNS", raising=False)
    root = tmp_path / "run"
    rd = root / BODY
    if not (RUN / BODY / "tune.json").exists():
        pytest.skip("no tune.json in the run directory")
    rd.mkdir(parents=True)
    (rd / "tune.json").write_text((RUN / BODY / "tune.json").read_text())

    def run(seed, **doc):
        d = train.run_dir_of(root, BODY, "R3", seed)
        d.mkdir(parents=True, exist_ok=True)
        (d / "run.json").write_text(json.dumps(dict(dict(status="running", g6={}), **doc)))

    run(42)
    run(43)
    left = {}
    assert ledger.resolve_rows(BODY, rd, ("R2", "R3"), left)[-1] == "R3/42" and left == {}
    run(42, g6=dict(G6c=dict(at_steps=2506752, pass_=False)))
    assert ledger.resolve_rows(BODY, rd, ("R2", "R3"), left) == ["R2/42"] and "G6c" in left["R3/42"]
    assert ledger.resolve_rows(BODY, rd, ("R3/42",)) == ["R3/42"]
    run(42, g6=dict(G6c=dict(at_steps=2506752, pass_=True)))
    run(43, status="stopped_g6b", stopped=dict(at_steps=2015232, reason="G6b failed again after the fix (FIX_RULE)"))
    left = {}
    assert ledger.resolve_rows(BODY, rd, ("R3",), left) == [] and "R3/43 stopped" in left["R3/42"]
    j = train.judged_checkpoint(root, "R3")
    assert set(j["left_out"]) == {f"{BODY}/42", f"{BODY}/43"} and j["runs"][f"{BODY}/43"]["left_out"]


def test_new_modules_leave_the_existing_keys():
    """The ledger and the report are outside every hashed source set: the panels' record keys (PANEL_SOURCES_RULE)
    and the tuner's sources (the fit's and G5's) do not include them."""
    assert not {"ledger.py", "report.py"} & set(panels.key_modules())
    assert tune.tuner_modules() == ["__init__.py", "control.py", "drive.py", "estimator.py", "legs.py", "metrics.py",
                                    "plant.py", "refs.py", "transitions.py", "tune.py"]
    assert "ledger.py" not in panels.panel_sources()["modules"]


# ------------------------------------------------------------------------------------------------
# Variants on a real plant
# ------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def worker():
    rows = panels.tracker_rows(BODY, RUN / BODY, ("R2/42",))
    return ledger.LedgerWorker(BODY, str(RUN / BODY), rows=[rows["R2/42"].spec()])


ARRAYS = ("geom_friction", "actuator_gainprm", "actuator_biasprm", "body_mass", "body_inertia", "body_subtreemass",
          "body_invweight0", "dof_invweight0", "geom_solref", "actuator_forcelimited")


def arrays(m):
    return {k: np.array(getattr(m, k), copy=True) for k in ARRAYS}


@needs_run
def test_plant_variants_apply_and_restore(worker):
    """Each variant changes what its rule says and the plant is the same model bit for bit after it: friction, kp
    (the controller keeping the nominal gains), latency (a k-step delayed command), trunk mass (the tracker on the
    nominal model), the drives' speed and torque scales and the bus voltage (the envelope and the ratings)."""
    ctx = worker.ctx
    walk = dataclasses.replace(COND["walk/nominal"], seconds=0.4)
    p = ctx.built(walk).p
    m = p.m
    before, env0, drives0, kp0 = arrays(m), p.envelope, p.drives, p.kp.copy()
    cg = (m.geom_contype | m.geom_conaffinity) > 0

    def restored():
        after = arrays(m)
        assert all(np.array_equal(after[k], before[k]) for k in ARRAYS), [k for k in ARRAYS
                                                                          if not np.array_equal(after[k], before[k])]
        assert p.envelope is env0 and p.drives is drives0 and "command" not in p.__dict__ and "reset" not in p.__dict__

    with ctx.variant(walk, ledger.Variant("friction", 0.35)):
        assert np.all(m.geom_friction[cg, 0] == 0.35)
    restored()
    with ctx.variant(walk, ledger.Variant("kp", 0.5)):
        assert np.array_equal(m.actuator_gainprm[:, 0], before["actuator_gainprm"][:, 0] * 0.5)
        assert np.array_equal(m.actuator_biasprm[:, 1], before["actuator_biasprm"][:, 1] * 0.5)
        assert np.array_equal(m.actuator_biasprm[:, 2], before["actuator_biasprm"][:, 2])  # kv unchanged
        assert np.array_equal(p.kp, kp0)  # the controller's gains (env.KP_RULE)
    restored()
    with ctx.variant(walk, ledger.Variant("latency_steps", 2)):
        p.reset()
        home = p.d.ctrl.copy()
        qs = [p.q_settled[p.qadr] + 0.01 * (i + 1) for i in range(4)]
        got = []
        for q in qs:
            p.command(q)
            got.append(p.d.ctrl.copy())
        assert np.array_equal(got[0], home) and np.array_equal(got[1], home)
        assert np.allclose(got[2], p.G @ qs[0]) and np.allclose(got[3], p.G @ qs[1])
    restored()
    nominal_mass = float(m.body_subtreemass[p.trunk])
    init0 = control.Tracker.__dict__["__init__"]
    with ctx.variant(walk, ledger.Variant("mass", 1.2)):
        assert float(m.body_mass[p.trunk]) == pytest.approx(1.2 * before["body_mass"][p.trunk])
        assert float(m.body_subtreemass[p.trunk]) > nominal_mass
        tk = control.Tracker(p, ctx.built(walk).track, "R2")
        assert tk.model is ctx.nominal_model(p) and tk.mass == nominal_mass
        assert control.Tracker(p, ctx.built(walk).track, "R2", model=m).mass > nominal_mass  # a model given is kept
    assert control.Tracker.__dict__["__init__"] is init0  # the tracker as built after the variant

    class Sub(control.Tracker):  # a row's own tracker class (an R4 row may subclass it) takes the nominal model too
        pass

    with ctx.variant(walk, ledger.Variant("mass", 1.2)):
        assert Sub(p, ctx.built(walk).track, "R2").mass == nominal_mass
    restored()
    rated0, peak0 = env0.rated.copy(), env0.peak.copy()
    with ctx.variant(walk, ledger.Variant("speed_scale", 0.6)):
        assert np.array_equal(p.envelope.rated, 0.6 * rated0) and np.array_equal(p.envelope.peak, peak0)
        assert np.array_equal(p.envelope.no_load, 0.6 * env0.no_load)
    restored()
    with ctx.variant(walk, ledger.Variant("torque_scale", 0.7)):
        ref = drive.Envelope(drives0, env0.moment, torque_scale=0.7)  # plant.build(torque_scale=0.7)'s envelope
        assert np.array_equal(p.envelope.peak, ref.peak) and np.array_equal(p.envelope.rated, rated0)
        assert [d.peak for d in p.drives] == [d.peak * 0.7 for d in drives0]
        assert [d.sustained for d in p.drives] == [d.sustained * 0.7 for d in drives0]
        assert np.array_equal(m.actuator_forcerange[:, 1], ref.peak)
    restored()
    with ctx.variant(walk, ledger.Variant("bus_v", 21.0)):
        want = [catalog.actuator(d.part).speed_at(21.0 if d.bus_v is not None else None) for d in drives0]
        assert np.allclose(p.envelope.rated, want, rtol=0, atol=1e-12)
        legs = [d.bus_v is not None for d in drives0]
        assert all((r < r0) == leg for r, r0, leg in zip(p.envelope.rated, rated0, legs, strict=True))
        assert [d.peak for d in p.drives] == [d.peak for d in drives0]
    restored()


@needs_run
def test_a_push_variant_sets_the_pulse(worker):
    """The pulse's force is m dv / 0.1 s along the direction; dv 0 is a zero pulse; the judgement records the dv and
    the Stage 0 bound."""
    ctx = worker.ctx
    cond = COND["push/standing_left"]
    with ctx.variant(cond, ledger.Variant("push_dv", 0.3)):
        b = ctx.built(cond)
        assert isinstance(b, ledger.PushedBuilt) and b.push_bound_m_s == env.PUSH_BOUND_M_S[BODY]
        f = b.push_force()
        c, s = math.cos(b.home_yaw), math.sin(b.home_yaw)  # left of the home heading
        assert f == pytest.approx(b.mass * 0.3 / panels.PUSH_S * np.array([-s, c]), rel=1e-12, abs=1e-12)
        assert b.push(3042)[0] == panels.Context.built(ctx, cond).push(3042)[0]  # the panels' timing
    with ctx.variant(cond, ledger.Variant("push_dv", 0.0)):
        assert not np.any(ctx.built(cond).push_force())
    assert not isinstance(ctx.built(cond), ledger.PushedBuilt)
    with pytest.raises(ValueError):
        with ctx.variant(COND["walk/nominal"], ledger.Variant("push_dv", 0.3)):
            ctx.built(COND["walk/nominal"])


@needs_run
def test_pile_damping_scales_the_damping_alone(worker):
    """On low pile: each probe's solref (tc / f, z f), so MuJoCo's contact damping B x f and stiffness K unchanged
    (efc_KBIP at the settled stance), restored after."""
    ctx = worker.ctx
    cond = COND["carpet/low"]
    p = ctx.built(cond).p
    m, d = p.m, p.d
    probes = ledger._probes(m)
    assert len(probes) > 100
    saved = m.geom_solref[probes].copy()

    def kbip():
        p.reset()
        mujoco.mj_forward(m, d)
        geoms = np.asarray(d.contact.geom)[: d.ncon]
        adr = np.asarray(d.contact.efc_address)[: d.ncon]
        rows = [a for g, a in zip(geoms, adr, strict=True) if (g[0] in probes or g[1] in probes) and a >= 0]
        return d.efc_KBIP[rows].copy()

    k0 = kbip()
    assert len(k0) > 10
    with ctx.variant(cond, ledger.Variant("pile_damping", 2.0)):
        assert np.array_equal(m.geom_solref[probes, 0], saved[:, 0] / 2.0)
        k2 = kbip()
    assert np.allclose(k2[:, 0], k0[:, 0], rtol=1e-9) and np.allclose(k2[:, 1], 2.0 * k0[:, 1], rtol=1e-9)
    assert np.array_equal(m.geom_solref[probes], saved)
    with pytest.raises(ValueError, match="no pile probes"):
        with ctx.variant(dataclasses.replace(COND["walk/nominal"], seconds=0.2), ledger.Variant("pile_damping", 2.0)):
            pass


@needs_run
def test_the_nominal_evaluation_is_the_panels_episode(worker):
    """An unchanged condition's evaluation is the panels' episode bit for bit (a fresh panels context), and so is a
    zero residual standing in for R3; a mass variant records the controller's nominal trunk mass."""
    walk = dataclasses.replace(COND["walk/nominal"], seconds=0.5)
    a = worker.evaluate(("R2/42", walk, None, None), 3043)
    _, docs = tune.read_inputs(RUN / BODY, panels.panel_files())
    rows = panels.tracker_rows(BODY, RUN / BODY, ("R2/42",))
    e = rows["R2/42"].run(panels.Context(BODY, RUN / BODY, docs), walk, 3043)
    for k in ledger.EPISODE_KEYS:
        assert a[k] == e[k], k
    assert a["cpu_s"] > 0 and "error" not in a
    lifted = worker.evaluate(("R2/42", walk, None, None, ("slip",)), 3043)  # LIFT_RULE: the panels' relaxation
    e = rows["R2/42"].run(panels.Context(BODY, RUN / BODY, docs), walk, 3043, ("slip",))
    assert all(lifted[k] == e[k] for k in ledger.EPISODE_KEYS) and lifted["relaxed"] == ["slip"]
    assert worker.ctx.plant("hard", None).m.geom_friction[0, 0] != control.CF_FRICTION  # restored
    r3 = panels.TrackerRow("R3/zero", "R2", rows["R2/42"].gains, rows["R2/42"].gains_info,
                           policy=panels.ZeroResidual())
    worker.rows["R3/zero"] = r3
    try:
        z = worker.evaluate(("R3/zero", walk, None, None), 3043)
        assert all(z[k] == a[k] for k in ledger.EPISODE_KEYS)
        mass = worker.evaluate(("R2/42", walk, "mass", 1.1), 3043)
        cm = mass["controller_model"]
        assert cm["nominal"] and cm["trackers"] == 1 and cm["plant_trunk_mass_kg"] > cm["nominal_trunk_mass_kg"]
    finally:
        worker.rows.pop("R3/zero")


@needs_run
def test_stop_reference_deceleration():
    """The existing stop's reference decelerates well below Stage 0's one-foot bound; the 1 Hz low-pass takes out
    the gait's own oscillation."""
    r = ledger.stop_reference(RUN / BODY)
    assert 0.1 < r["peak_deceleration_m_s2"] < 1.0 < r["peak_deceleration_unfiltered_m_s2"]
    assert r["speed_before_m_s"] == pytest.approx(0.218266 * panels.RATE, rel=0.1)
    assert r["from_to"] == ["nominal", "stand"]


# ------------------------------------------------------------------------------------------------
# A tiny real sweep: records, resume, dry run, the command line
# ------------------------------------------------------------------------------------------------
@pytest.fixture()
def r3_stand_in(monkeypatch):
    """R3 rows registered as a zero residual on R2/42's gains (panels.ZeroResidual stands in for the policy)."""
    monkeypatch.setattr(panels, "ROW_FACTORIES", {})
    g, info = panels.r2_gains(RUN / BODY, 42)

    def factory(body, run_dir, name):
        return panels.TrackerRow(name, "R2", g, info, policy=panels.ZeroResidual(), controller="zero residual")

    panels.register_rows("R3", factory)


@needs_run
def test_a_tiny_sweep_writes_records_and_resumes(tmp_path, r3_stand_in, monkeypatch):
    """The kp margin and the stop on R2 (the body-level selection) and the R3 stand-in, episodes cut to 0.6 s and one
    halving: one record per row and margin with its results, evaluations and key; the stand-in's results are R2/42's;
    a second sweep runs nothing; another ledger source runs it again."""
    log = []
    doc = ledger.sweep(BODY, tmp_path, rows=("R2", "R3/42"), names=("kp", "stop"), cut_s=0.6, halvings=1,
                       inputs=RUN, log=log.append)
    s = doc["sessions"][-1]
    assert s["rows"] == ["R2/42", "R3/42"] and s["episodes_run"] > 0 and s["episodes_run"] % 3 == 0
    recs = {r: json.loads(ledger.record_path(tmp_path, BODY, r, "kp").read_text()) for r in ("R2/42", "R3/42")}
    a, b = recs["R2/42"], recs["R3/42"]
    assert a["status"] == b["status"] == "done" and a["cut_s"] == 0.6 and a["halvings"] == 1
    # cut before the walk's window every seed fails the same terms at every kp: works (WORKS_RULE) is survives
    ws = a["searches"]["walk_nominal"]
    assert ws["success"]["status"] == ws["survival"]["status"] == "works across the interval"
    nominal = a["evaluations"]["walk/nominal|nominal"]
    assert nominal["succeeds_all"] is False and nominal["survives"] and nominal["variant"] == ledger.NOMINAL.describe()
    assert set(a["evaluations"]) == {"walk/nominal|nominal", "walk/nominal|kp=0.25"}
    assert a["summary"] == b["summary"] and a["panel_sources"] == panels.panel_sources()["sha256"]
    from designlab.stage1 import report  # LEDGER_TABLE_RULE: the report recomputes the same panel keys

    assert report.current_panel_keys(RUN, BODY, "R2/42", ["walk/nominal"], 0.6) == a["key"]["panel_keys"]
    assert report.current_panel_keys(RUN, BODY, "R2/42", ["walk/nominal"]) != a["key"]["panel_keys"]  # uncut
    for e in a["evaluations"].values():
        assert e["seeds"] == [3042, 3043, 3044] and len(e["episodes"]) == 3 and e["errors"] == 0
        assert e["valid"] and e["cpu_s"] > 0
    assert set(a["rules"]) >= {"works", "search", "variant", "label", "resume"} and "kp" in a["rule"]
    assert a["key"]["panel_keys"].keys() == {"walk/nominal"} and a["controller"]["row"] == "R2/42"
    stop = json.loads(ledger.record_path(tmp_path, BODY, "R2/42", "stop").read_text())
    assert stop["status"] == "not run" and stop["reference_stop"]["peak_deceleration_m_s2"] > 0
    assert (tmp_path / BODY / "sweeps" / "summary.json").exists()
    again = ledger.sweep(BODY, tmp_path, rows=("R2", "R3/42"), names=("kp", "stop"), cut_s=0.6, halvings=1,
                         inputs=RUN, log=log.append)
    s2 = again["sessions"][-1]
    assert s2["episodes_run"] == 0 and len(s2["skipped"]) == 4 and s2["evaluations_reused"] > 0
    # a record that lost its last evaluations resumes from the ones it holds
    rec = json.loads(ledger.record_path(tmp_path, BODY, "R2/42", "kp").read_text())
    keep = dict(list(rec["evaluations"].items())[:1])
    rec["evaluations"], rec["status"] = keep, "partial"
    ledger.record_path(tmp_path, BODY, "R2/42", "kp").write_text(json.dumps(rec))
    for f in ("R3/42/kp.json", "R3/42/stop.json"):
        (tmp_path / BODY / "sweeps" / f).unlink()
    third = ledger.sweep(BODY, tmp_path, rows=("R2",), names=("kp",), cut_s=0.6, halvings=1, inputs=RUN,
                         log=log.append)
    s3 = third["sessions"][-1]
    assert 0 < s3["episodes_run"] < s["episodes_run"] / 2 + 1
    assert json.loads(ledger.record_path(tmp_path, BODY, "R2/42", "kp").read_text())["summary"] == a["summary"]
    monkeypatch.setattr(ledger, "ledger_sources", lambda: dict(sha256="changed", file="ledger.py", rule=""))
    fourth = ledger.sweep(BODY, tmp_path, rows=("R2",), names=("kp",), cut_s=0.6, halvings=1, inputs=RUN,
                          log=log.append)
    assert fourth["sessions"][-1]["episodes_run"] == s["episodes_run"] / 2


@needs_run
def test_lifts_are_recorded_and_resumed(tmp_path):
    """LIFT_RULE through a Session (synthetic evaluations, no episode): the fails end's falls are lifted once for
    both criteria, the lift is recorded under its own id and key, labels the threshold, and a new session reloads it
    (RESUME_RULE: the key holds the relaxation)."""
    ses = ledger.Session(BODY, tmp_path, rows=("R2",), names=("kp",), inputs=RUN)
    row, m = "R2/42", ses.margins[0]

    def fill(s):
        made = []
        while True:
            need = s.advance()
            if not need:
                return made
            for key, (r, k, var, relax) in need.items():
                f = 1.0 if var.kind is None else var.value
                outs = ["ok"] * 3 if relax or f >= 0.5 else [("fell", ["slip"]), ["grf"], "ok"]
                e = entry(k, var, outs, relax)
                e["valid"] = s.valid(r, k, var, relax)
                s.cache[key] = e
                made.append(key)

    made = fill(ses)
    lifts = [k for k in made if k[3]]
    assert len(lifts) == 1 and lifts[0][3] == ("slip",)
    rec = ses.record(row, m, dict(commit="x"), dict(started_at="now"))
    res = rec["searches"]["walk_nominal"]
    assert rec["status"] == "done" and res["success"]["fails"] == res["survival"]["fails"] < 0.5
    lift_id = f"walk/nominal|kp={res['survival']['fails']!r}|lift=slip"
    assert [i for i in rec["evaluations"] if "|lift=" in i] == [lift_id]
    for crit in ("success", "survival"):
        assert res[crit]["binding"]["label"] == "body-limited: slip (lifted, all 3 seeds survive)", crit
    assert ses.valid(row, "walk/nominal", ledger.NOMINAL, ("slip",)) != ses.valid(row, "walk/nominal", ledger.NOMINAL)
    panels._write_json(ledger.record_path(tmp_path, BODY, row, "kp"), rec)
    again = ledger.Session(BODY, tmp_path, rows=("R2",), names=("kp",), inputs=RUN)
    assert again.advance() == {} and again.runs[(row, "kp")].done and len(again.cache) == len(made)


@needs_run
def test_dry_run_counts(tmp_path):
    """The dry run lists every search with its condition and the episodes it can still need (success alone at least,
    both criteria at most), points once, the stop not run; nothing runs."""
    doc = ledger.dry_run(BODY, tmp_path, rows=("R2",), names=("grip", "latency", "voltage", "stop"), inputs=RUN,
                         measure=False, log=lambda m: None)
    lines = {(x["margin"], x["search"]): x for x in doc["lines"]}
    g = lines[("grip", "walk_nominal")]
    assert (g["episodes_min"], g["episodes_max"]) == (27, 54) and g["seconds"] > 15 and g["core_h_max"] is None
    lat = lines[("latency", "walk_nominal")]
    assert (lat["values_min"], lat["values_max"]) == (6, 12)
    v = [x for x in doc["lines"] if x["margin"] == "voltage"]
    assert len(v) == 6 and all(x["episodes_max"] == 3 for x in v)
    assert {x["variant"]["kind"] for x in v if x["point"] == 25.2} == {None}  # the 24 V drives get no credit
    assert lines[("stop", None)]["status"] == "not run" and doc["totals"]["episodes_min"] == 4 * 27 + 18 + 18
    assert g["lifts_max"] == 2 * len(control.RELAXATIONS) and lines[("stop", None)]["lifts_max"] == 0  # LIFT_RULE
    assert doc["totals"]["lift_episodes_max"] == 3 * (5 * 2 * len(control.RELAXATIONS) + 6 * len(control.RELAXATIONS))
    assert not (tmp_path / BODY).exists()


@needs_run
def test_sweep_command_line(tmp_path, capsys, monkeypatch):
    """``sweep --dry-run`` lists and runs nothing; ``sweep`` of the stop writes its record with no episode; R3/42 with
    no --row-module is the policy rows' row (ROWS_RULE), refused with the hint where its run cannot be read; a row of
    a controller no module registers is refused with the --row-module hint."""
    from designlab.stage1 import policy_rows

    assert cli(["sweep", "--body", BODY, "--rows", "R2", "--margins", "kp,stop", "--dry-run", "--no-measure",
                "--inputs", str(RUN), "--out", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "R2/42" in out and "walk/nominal" in out and "not run" in out and not (tmp_path / BODY).exists()
    from designlab.stage1 import __main__ as main_mod

    monkeypatch.setattr(main_mod, "round_two_models", lambda body: False)  # a checkout without round2's models
    with pytest.raises(SystemExit, match="round2/refine"):
        cli(["sweep", "--body", BODY, "--rows", "R2", "--margins", "stop", "--inputs", str(RUN), "--out",
             str(tmp_path)])
    assert not (tmp_path / BODY).exists()
    assert cli(["sweep", "--body", BODY, "--rows", "R2", "--margins", "stop", "--inputs", str(RUN), "--out",
                str(tmp_path), "--allow-rebuilt-models"]) == 0
    assert ledger.record_path(tmp_path, BODY, "R2/42", "stop").exists()
    monkeypatch.setattr(panels, "ROW_FACTORIES", {})
    monkeypatch.setenv(policy_rows.RUNS_ENV, str(tmp_path / "no_training_run"))
    with pytest.raises(SystemExit, match="PPO venv"):
        cli(["sweep", "--body", BODY, "--rows", "R3/42", "--margins", "stop", "--inputs", str(RUN), "--out",
             str(tmp_path), "--allow-rebuilt-models"])
    assert panels.ROW_FACTORIES["R3"] is policy_rows.r3_row
    with pytest.raises(SystemExit, match="row-module"):
        cli(["sweep", "--body", BODY, "--rows", "R5/1", "--margins", "stop", "--inputs", str(RUN), "--out",
             str(tmp_path), "--allow-rebuilt-models"])


def test_the_ledgers_r3_is_the_policy_rows_row(monkeypatch):
    """ROWS_RULE: with no module registering R3 or R4, the ledger registers the policy rows' (policy_rows.r3_row and
    r4_row) through panels.register_rows for the rows asked for; a controller another module registered keeps its
    rows, and R2 needs nothing."""
    from designlab.stage1 import policy_rows

    monkeypatch.setattr(panels, "ROW_FACTORIES", {})
    assert ledger.register_policy_rows(["R2/42"]) == [] and panels.ROW_FACTORIES == {}
    assert ledger.register_policy_rows(["R2/42", "R3/42"]) == ["R3"]
    assert panels.ROW_FACTORIES == {"R3": policy_rows.r3_row}
    assert ledger.register_policy_rows(["R3/42", "R4/42"]) == ["R4"]
    assert panels.ROW_FACTORIES == {"R3": policy_rows.r3_row, "R4": policy_rows.r4_row}

    def other(body, run_dir, name):
        raise AssertionError("not built here")

    monkeypatch.setattr(panels, "ROW_FACTORIES", {"R3": other})
    assert ledger.register_policy_rows(["R3/42", "R4/42"]) == ["R4"] and panels.ROW_FACTORIES["R3"] is other
    assert ledger.DEFAULT_ROWS == ("R2", "R3") and ledger.R3_ROW == "R3/42"


@needs_run
def test_dry_run_costs_a_policy_row_at_its_own_cost(tmp_path, r3_stand_in, monkeypatch):
    """COST_RULE: a policy row's searches are costed at its own measured cost per simulated second
    (policy_rows.measure_costs on the floors they still need), a tracker row's at R2/42's (panels.measure_costs)."""
    from designlab.stage1 import policy_rows

    calls = []

    def r2_cost(body, run_dir, floors, log=print):
        return dict(floors={f: dict(cpu_per_sim_s=0.1, wall_per_sim_s=0.3) for f in floors})

    def policy_cost(body, run_dir, rows, floors, r2=True, log=print, **kw):
        calls.append((tuple(rows), tuple(floors), r2))
        return dict(rows={r: {f: dict(cpu_per_sim_s=0.2, wall_per_sim_s=0.6) for f in floors} for r in rows},
                    checkpoints={r: dict(steps=1, selection="a test's") for r in rows})

    monkeypatch.setattr(panels, "measure_costs", r2_cost)
    monkeypatch.setattr(policy_rows, "measure_costs", policy_cost)
    doc = ledger.dry_run(BODY, tmp_path, rows=("R2", "R3/42"), names=("kp", "pile_damping"), inputs=RUN,
                         log=lambda m: None)
    assert calls == [(("R3/42",), ("hard", "low"), False)]
    assert doc["costs"]["rows"]["R3/42"]["low"]["cpu_per_sim_s"] == 0.2
    lines = [x for x in doc["lines"] if x["values_max"]]
    assert {x["row"] for x in lines} == {"R2/42", "R3/42"} and {x["floor"] for x in lines} == {"hard", "low"}
    for x in lines:
        cost = 0.2 if x["row"] == "R3/42" else 0.1
        assert x["core_h_max"] == pytest.approx(x["episodes_max"] * x["seconds"] * cost / 3600), x


def test_records_are_json_and_rules_are_text():
    assert all(isinstance(v, str) and len(v) > 40 for v in ledger.RULES.values())
    assert all(isinstance(v, str) for v in ledger.MARGIN_RULES.values()) and set(ledger.MARGIN_RULES) == set(
        ledger.MARGIN_NAMES)
    assert ledger.SEEDS == (3042, 3043, 3044) and ledger.HALVINGS == 7 and math.isclose(ledger.MU_NOMINAL, 0.9)
    v = ledger.Variant("latency_steps", 3)
    assert json.loads(json.dumps(v.describe())) == {"kind": "latency_steps", "value": 3}
    assert ledger.Variant(**json.loads(json.dumps(v.describe()))).key == v.key == "latency_steps=3"
    assert copy.deepcopy(ledger.NOMINAL).key == "nominal"
