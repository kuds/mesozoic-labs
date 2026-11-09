"""Tests for the Stage 1 behaviour panels and gate G7 (designlab/stage1/panels.py; STAGE1_PLAN.md section 3 step 8,
section 4, section 7).

Run from hardware/trex_form_study (the fit's venv):  python -m pytest tests/test_stage1_panels.py -q
Every test is fast: the judgement, labels, verdicts and bounds on synthetic data, and real episodes of a few
simulated seconds (conditions cut short with ``Condition.seconds``).  The tests that run episodes need the run
directory's references and transitions (git-ignored; ``python -m designlab.stage1 prepare``) and skip without them.
"""

from __future__ import annotations

import dataclasses
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import designlab  # noqa: E402,F401  (selects the GL backend before mujoco renders)
from designlab import carpet, statics  # noqa: E402
from designlab.stage1 import control, env, gates, metrics, panels, plant, tune  # noqa: E402
from designlab.stage1.__main__ import main as cli  # noqa: E402
from designlab.stage1.refs import Reference  # noqa: E402

BODY = "60-N-T-10"
RUN = ROOT / "stage1" / "runs" / "2026-11-08"
needs_run = pytest.mark.skipif(not (RUN / BODY / "refs" / "nominal.npz").exists(),
                               reason="the run directory's references are not generated (designlab.stage1 prepare)")
COND = {c.key: c for c in panels.conditions()}


@pytest.fixture(scope="module")
def ctx():
    _, docs = tune.read_inputs(RUN / BODY, panels.panel_files())
    return panels.Context(BODY, RUN / BODY, docs)


@pytest.fixture(scope="module")
def rows():
    return panels.tracker_rows(BODY, RUN / BODY)


# ------------------------------------------------------------------------------------------------
# Conditions and plans
# ------------------------------------------------------------------------------------------------
def test_conditions_and_the_gated_split():
    """Plan section 4's conditions, gated and reported, and which rows run them (R0 and R1 the hard floor only)."""
    cs = panels.conditions()
    assert len(cs) == 28 and len({c.key for c in cs}) == 28
    assert {c.behavior for c in cs} == set(panels.BEHAVIORS)
    reported = {c.key for c in cs if not c.gated}
    assert reported == {"walk/brisk_mu0.54", "walk/fr05_mu0.54", "turn/walking_0.50_left", "turn/walking_0.50_right",
                        "carpet/medium", "carpet/high"}
    by = {b: sorted(c.name for c in cs if c.behavior == b) for b in panels.BEHAVIORS}
    assert by["stand"] == ["hard", "low", "robot_floor_carpet"] and by["start_stop"] == ["twice"]
    assert by["walk"] == ["brisk", "brisk_mu0.54", "fr05", "fr05_mu0.54", "nominal", "nominal_mu0.54"]
    assert len(by["push"]) == 8 and len(by["turn"]) == 6 and by["carpet"] == ["high", "low", "medium",
                                                                                 "robot_floor_carpet"]
    assert all(c.friction == 0.54 for c in cs if c.behavior == "carpet" or c.name.endswith("mu0.54"))
    hard_only = panels.TrackerRow("R0", "R1", hard_only=True)
    every = panels.TrackerRow("R2/42", "R2")
    assert sum(hard_only.runs(c) for c in cs) == 22 and all(hard_only.runs(c) == (c.floor == "hard") for c in cs)
    assert sum(every.runs(c) for c in cs) == 28
    for c in cs:  # every success needs the drives and the joint range (LIMITS_RULE)
        terms = panels.terms_of(c)
        assert terms[0] == "survived" and {"drive_sustained", "drive_speed", "joint_range"} <= set(terms)
    assert "cadence" in panels.terms_of(COND["walk/nominal"]) and "cadence" not in panels.terms_of(COND["walk/brisk"])
    assert {"toe_target", "no_toe_catch", "cadence"} <= set(panels.terms_of(COND["carpet/low"]))
    assert panels.select(cs, ["walk"], ["walk/nominal", "brisk"]) == [COND["walk/nominal"], COND["walk/brisk"]]
    with pytest.raises(ValueError):
        panels.select(cs, ["swim"])


def test_seeds_slots_and_record_paths():
    """Episode seeds: tune's derived seeds in the slot of tune's mix (so G5's episodes are the panels'); push timing
    from the third purpose; a record per row and condition (plan section 7)."""
    s = 3042
    assert panels.episode_seeds(COND["walk/nominal"], s) == dict(slot=0, **{k: tune.derived_seed(s, 0, v) for k, v in
                                                                             panels.PURPOSES.items()})
    assert panels.episode_seeds(COND["turn/walking_0.35_left"], s)["slot"] == 1
    assert panels.episode_seeds(COND["turn/in_place_0.50_right"], s)["slot"] == 2
    assert panels.episode_seeds(COND["carpet/low"], s)["slot"] == 0 and panels.episode_seeds(COND["push/walking_left"],
                                                                                              s)["slot"] == 0
    u = [panels.push_u(COND["push/standing_left"], x) for x in panels.SEEDS]
    assert all(0.0 <= x < 1.0 for x in u) and len(set(u)) == len(u)
    assert panels.push_u(COND["push/standing_left"], 3050) == panels.push_u(COND["push/standing_right"], 3050)
    assert panels.SEEDS == tuple(range(3042, 3062))
    assert panels.record_path(Path("o"), BODY, "R2/42", COND["walk/nominal"]) == Path(
        f"o/{BODY}/eval/R2/42/walk/nominal.json")
    assert panels.record_path(Path("o"), BODY, "R0", COND["push/walking_left"]) == Path(
        f"o/{BODY}/eval/R0/push/walking_left.json")


@needs_run
def test_planned_lengths_are_the_built_tracks():
    """The planner's lengths, read from the files, equal the tracks control.build_track builds (each distinct plan, on
    the hard-floor plant); the brisk and Fr 0.05 walks keep their last 10 s in the steady gait even with a slow clock;
    the nominal walk and the 0.35 and in-place turns are tune's episodes."""
    run = RUN / BODY
    inp, docs = tune.read_inputs(run, panels.panel_files())
    pl = panels.Planner(run, docs["refs.json"]["references"], docs["transitions.json"]["transitions"])
    p = plant.build(BODY, "hard", settle=False)
    om = control.omega0(p, Reference.load(run / "refs" / "nominal.npz").q_home)
    seen = {}
    for c in panels.conditions():
        r = pl.resolve(c)
        key = tuple(tuple(x) for x in r["plan"])
        if key not in seen:
            seen[key] = control.build_track(p, run, r["plan"], om).t_end
        if c.judge in ("stand", "push_stand"):
            assert r["seconds"] < seen[key]  # the hold outlasts the stand
        elif c.judge == "push_walk":
            assert seen[key] >= (1 + panels.CLOCK_SLACK) * panels.RATE * r["seconds"]
        else:
            assert r["seconds"] == seen[key] / panels.RATE, c.key
        if c.judge == "walk" and c.gait != "nominal":
            assert (1 - panels.CLOCK_SLACK) * (seen[key] - tune.WALK_WINDOW_S * panels.RATE) >= r["steady_ref_s"]
    assert panels.plan_is_tunes(COND["walk/nominal"], pl.resolve(COND["walk/nominal"])["plan"])
    assert panels.plan_is_tunes(COND["turn/walking_0.35_left"], pl.resolve(COND["turn/walking_0.35_left"])["plan"])
    assert panels.plan_is_tunes(COND["turn/in_place_0.50_left"], pl.resolve(COND["turn/in_place_0.50_left"])["plan"])
    assert not panels.plan_is_tunes(COND["turn/walking_0.50_left"], pl.resolve(COND["turn/walking_0.50_left"])["plan"])
    assert not panels.plan_is_tunes(COND["carpet/low"], pl.resolve(COND["carpet/low"])["plan"])
    ss = pl.resolve(COND["start_stop/twice"])
    assert len(ss["starts_ref_s"]) == panels.START_STOP_CYCLES and ss["starts_ref_s"][0] > 0
    assert set(panels.files_of(COND["walk/fr05"])) >= {"refs/fr05.npz", "transitions/brisk_to_fr05.npz"}


@pytest.mark.parametrize("body", ["60-N-T-10", "75-M-H-10"])
def test_floors_without_a_settled_stance_are_not_run(body):
    """NOT_RUN_RULE: the 60 cm bodies' zero-command stance does not settle on medium and high pile (plant.json)."""
    if not (RUN / body / "plant.json").exists():
        pytest.skip("no plant.json")
    st = {f: panels.stands(RUN / body, f)[0] for f in plant.FLOORS}
    assert st["hard"] and st["robot_floor_carpet"] and st["low"]
    assert st["medium"] == st["high"] == (body == "75-M-H-10")


# ------------------------------------------------------------------------------------------------
# Bounds, counts, labels, verdicts (synthetic)
# ------------------------------------------------------------------------------------------------
def test_clopper_pearson_lower_bounds():
    """LCB_RULE: 20/20 -> 0.861, 17/20 -> 0.656 (plan section 4), 34/40 -> 0.72526 (recovery_gate's pinned figure)."""
    assert round(panels.lcb(20, 20), 3) == 0.861 and math.isclose(panels.lcb(20, 20), 0.05 ** (1 / 20), rel_tol=1e-12)
    assert round(panels.lcb(17, 20), 3) == 0.656 and round(panels.lcb(34, 40), 5) == 0.72526
    assert panels.lcb(0, 20) == 0.0 and panels.lcb(0, 0) is None
    v = [panels.lcb(k, 20) for k in range(21)]
    assert all(a < b for a, b in zip(v[:-1], v[1:], strict=True))


def _ep(seed, fell=False, ok=True, terms=(), strides=10.0):
    return dict(seed=seed, fell=fell, succeeded=ok and not fell, failed_terms=list(terms), strides=strides,
                fall_time_s=1.0 if fell else None, fall_cause="tilt" if fell else None,
                fall_candidates=["slip"] if fell else None, survival_s=1.0 if fell else 10.0)


def test_counts_and_the_pass_rule():
    """PASS_RULE: 20/20 survive and at least 17/20 succeed; fewer seeds are not judged."""
    eps = [_ep(s, ok=i >= 3) for i, s in enumerate(panels.SEEDS)]
    c = panels.counts(eps)
    assert c["judged"] and c["pass_"] and c["survived"] == 20 and c["succeeded"] == 17
    assert not panels.counts([_ep(s, ok=i >= 4) for i, s in enumerate(panels.SEEDS)])["pass_"]
    c = panels.counts([_ep(s, fell=i == 0) for i, s in enumerate(panels.SEEDS)])
    assert not c["pass_"] and c["falls"] == 1 and round(c["survival_lcb"], 3) == round(panels.lcb(19, 20), 3)
    c = panels.counts([_ep(s) for s in panels.SEEDS[:5]], panels.SEEDS[:5])
    assert not c["judged"] and c["pass_"] is None


def test_success_failures_are_labelled_by_their_terms():
    """SUCCESS_LABEL_RULE: body-limited by the hardware terms failing in more than half of the failed episodes."""
    eps = [_ep(s) for s in panels.SEEDS[:16]]
    eps += [_ep(s, ok=False, terms=t) for s, t in zip(panels.SEEDS[16:], (["drive_speed", "speed"], ["drive_speed"],
                                                                          ["drive_speed"], ["cadence"]), strict=True)]
    lab = panels.success_label(eps)
    assert lab["label"] == "body-limited: drive_speed" and lab["terms"]["drive_speed"] == dict(count=3, cls="hardware")
    assert lab["terms"]["cadence"]["cls"] == "controller"
    eps[-2] = _ep(eps[-2]["seed"], ok=False, terms=["speed"])
    assert panels.success_label(eps)["label"] == "controller-limited"  # drive_speed in 2 of 4 failed episodes
    f = panels.failure("R2/42", eps, None, False)
    assert f["kind"] == "success" and f["label"] == "controller-limited"
    assert panels.failure("R2/42", eps, None, True) is None and panels.failure("R2/42", eps, None, None) is None
    assert panels.term_class("survived") is None and panels.term_class("slip_worst") == "hardware"


def test_falls_are_labelled_by_the_attribution():
    """FAILURE_LABEL_RULE and PANEL_ATTRIBUTION_RULE: decision 20's paired sign test over the same starts, no null."""
    base = [_ep(s, fell=i < 6, strides=3.0 if i < 6 else 10.0) for i, s in enumerate(panels.SEEDS)]
    for e in base:
        e["succeeded"] = not e["fell"]
    rec = dict(episodes=base)
    cands, lift = panels.attribution_plan(rec)
    assert cands == ["slip"] and lift == ["slip"]
    lifted = {"slip": [_ep(e["seed"]) for e in base]}
    a = panels.attribution_record(rec, cands, lifted, 1.0)
    assert a["label"] == "body-limited: slip" and a["null"] is None and a["episodes"] == 20
    assert a["lifted"]["slip"]["sign_test"] == dict(better=6, worse=0, tied=14, p=1 / 64)
    few = [_ep(s, fell=i < 3, strides=3.0 if i < 3 else 10.0) for i, s in enumerate(panels.SEEDS)]
    a3 = panels.attribution_record(dict(episodes=few), ["slip"], {"slip": [_ep(e["seed"]) for e in few]}, 1.0)
    assert a3["label"] == "controller-limited" and a3["lifted"]["slip"]["sign_test"]["p"] == 1 / 8
    assert panels.failure("R2/42", base, a, False)["label"] == "body-limited: slip"
    assert panels.failure("R2/42", base, None, False)["label"].startswith("not attributed (falls; run eval")
    assert panels.failure("R0", base, None, False)["label"].startswith("not attributed (falls; attribution is applied")
    assert gates.attribute([dict(fell=True, strides=1.0)], {}, None, [])["label"] == "controller-limited"


def _record(pass_, label=None, judged=True, status="done", row="R2/42"):
    return dict(row=row, status=status, counts=dict(judged=judged, pass_=pass_ if judged else None),
                failure=None if pass_ else dict(label=label))


def test_verdict_logic():
    """VERDICT_RULE on the two seeds' records."""
    P, B, C, N = (_record(True), _record(False, "body-limited: slip"), _record(False, "controller-limited"),
                  _record(False, "not attributed (falls; run eval --attribute)"))
    assert panels.verdict([P, P])["verdict"] == "shown"
    assert panels.verdict([P, B])["verdict"] == "provisional" == panels.verdict([C, P])["verdict"]
    assert panels.verdict([B, B])["verdict"] == "body-limited"
    assert panels.verdict([B, C])["verdict"] == "controller-limited"
    assert panels.verdict([B, N])["verdict"] == "not attributed" == panels.verdict([C, N])["verdict"]
    assert panels.verdict([P, None])["verdict"] == "incomplete"
    assert panels.verdict([P, _record(True, judged=False)])["verdict"] == "incomplete"
    assert panels.verdict([_record(None, status="not run"), _record(None, status="not run")])["verdict"] == "not run"


# ------------------------------------------------------------------------------------------------
# The success judgement on synthetic metrics
# ------------------------------------------------------------------------------------------------
NJ = 14


def _built(cond, seconds=20.0, **kw):
    res = dict(push=None)
    if cond.judge in ("push_stand", "push_walk"):
        res["push"] = dict(kind="standing", at_s=2.0, spread_s=1.0)
    return panels.Built(cond=cond, p=SimpleNamespace(joints=[f"j{i}" for i in range(NJ)]), plan=[], track=None,
                        spec=None, seconds=seconds, resolved=res, lo=np.full(NJ, -1.0), hi=np.full(NJ, 1.0),
                        required=np.full(NJ, 0.1), H=0.4, mass=10.0, speed_cmd=0.23, stride_s=0.97,
                        push_bound_m_s=0.487, home_yaw=0.0, **kw)


def _out(fell=False, tilt=5.0, sustained=0.5, speed=0.5, seconds=20.0):
    return dict(fell=fell, fall_time_s=1.0 if fell else None, fall_cause="tilt" if fell else None, seconds=seconds,
                strides=seconds, worst_sustained_ratio=["knee", sustained], worst_speed_ratio=["knee", speed],
                tilt_from_home_max_deg=tilt, captures=[], phase_rate_range=[0.0, 0.0])


def _trace(t, com, v=None, yaw=None, q=None):
    rec = metrics.Recorder()
    U = np.zeros((len(t), len(metrics.COLS)))
    U[:, metrics.C["t"]] = t
    U[:, metrics.C["t_ref"]] = t
    U[:, [metrics.C["com_x"], metrics.C["com_y"]]] = com
    if v is not None:
        U[:, [metrics.C["v_x"], metrics.C["v_y"]]] = v
    if yaw is not None:
        U[:, metrics.C["yaw"]] = yaw
    rec.U, rec.rows, rec.dt = U, len(t), 0.002
    rec.Q = np.zeros((len(t), NJ)) if q is None else q
    return rec


def test_stand_success():
    """STAND_RULE with LIMITS_RULE: tilt below 12 deg, the drives and the range; a fall fails every term."""
    b, rec = _built(COND["stand/hard"]), _trace(np.arange(0, 20, 0.01), np.zeros((2000, 2)))
    assert panels.judge(b, _out(tilt=11.9), rec, 1)["succeeded"]
    e = panels.judge(b, _out(tilt=12.1), rec, 1)
    assert not e["succeeded"] and e["failed_terms"] == ["tilt"]
    assert panels.judge(b, _out(sustained=1.01), rec, 1)["failed_terms"] == ["drive_sustained"]
    assert panels.judge(b, _out(speed=1.2), rec, 1)["failed_terms"] == ["drive_speed"]
    q = np.zeros((2000, NJ))
    q[100, 3] = 0.95  # 0.05 rad from the range: short of the 0.10 rad
    e = panels.judge(b, _out(), _trace(np.arange(0, 20, 0.01), np.zeros((2000, 2)), q=q), 1)
    assert e["failed_terms"] == ["joint_range"] and e["joint_worst"] == "j3"
    e = panels.judge(b, _out(fell=True, tilt=40.0), rec, 1)
    assert e["failed_terms"] == list(panels.terms_of(b.cond)) and not e["success"]["survived"]


def test_push_recovery_rule():
    """RECOVERY_RULE: standing, the CoM speed below 5 % of the nominal command from a time within 3 s of the pulse's
    end to the episode's end; walking, the stride-averaged speed along the starting heading at least 90 % of the
    command; a trace a fall cut short has not recovered."""
    t = np.arange(0.0, 8.1, 0.01)
    t_end = 2.6
    for back_after, ok in ((2.0, True), (3.5, False)):
        v = np.where((t > t_end) & (t < t_end + back_after), 0.1, 0.004)
        r = panels.recovery(_trace(t, np.zeros((len(t), 2)), np.column_stack([v, 0 * v])), t_end, True, 0.23, 0.97,
                            until_s=8.1)
        assert r["recovered"] == ok and r["after_s"] == pytest.approx(back_after, abs=0.011)
        assert r["threshold_m_s"] == pytest.approx(0.05 * 0.23)
    cut = t <= 5.0
    v = np.where((t > t_end) & (t < t_end + 1.0), 0.1, 0.004)[cut]
    r = panels.recovery(_trace(t[cut], np.zeros((cut.sum(), 2)), np.column_stack([v, 0 * v])), t_end, True, 0.23,
                        0.97, until_s=8.1)
    assert r["after_s"] is not None and not r["recovered"]
    # walking at the command, a second at half of it after the pulse, then the command again
    vx = np.where((t > t_end) & (t < t_end + 1.0), 0.5 * 0.23, 0.23)
    x = np.concatenate([[0.0], np.cumsum(vx[1:] * np.diff(t))])
    r = panels.recovery(_trace(t, np.column_stack([x, 0 * x]), yaw=np.zeros(len(t))), t_end, False, 0.23, 0.97,
                        until_s=8.1)
    assert r["recovered"] and 1.0 < r["after_s"] < 2.0 and r["threshold_m_s"] == pytest.approx(0.9 * 0.23)
    # the same walk heading off at 30 deg from the starting heading: never 90 % along it
    xy = np.column_stack([x * math.cos(0.52), x * math.sin(0.52)])
    xy[t <= t_end] = np.column_stack([x, 0 * x])[t <= t_end]
    r = panels.recovery(_trace(t, xy, yaw=np.zeros(len(t))), t_end, False, 0.23, 0.97, until_s=8.1)
    assert not r["recovered"]


def test_carpet_toe_terms():
    """CARPET_RULE: the toe at least carpet_target(H) above the pile top over mid-swing, and no toe catch."""
    H = 0.40468919988052376
    target = carpet.carpet_target(H)
    assert target == pytest.approx(0.06 * H) and round(1e3 * target, 1) == 24.3
    ok = panels.carpet_terms(dict(complete=True, toe_clearance_m=target + 1e-3, swings=12), H)["ok"]
    assert ok == dict(toe_target=True, no_toe_catch=True)
    assert panels.carpet_terms(dict(complete=True, toe_clearance_m=target - 1e-3), H)["ok"] == dict(
        toe_target=False, no_toe_catch=True)
    assert panels.carpet_terms(dict(complete=True, toe_clearance_m=-2e-3), H)["ok"] == dict(toe_target=False,
                                                                                            no_toe_catch=False)
    assert panels.carpet_terms(dict(complete=False), H)["ok"] == dict(toe_target=False, no_toe_catch=False)


def test_start_and_stop_terms():
    """START_STOP_RULE: each start at 90 % within 2 strides, each stop within decision 18's 1.5 strides, every
    acceleration peak at most 1.5 m/s^2."""
    st = [dict(reached=True, strides=1.8, acceleration_peak_m_s2=1.2)] * 2
    sp = [dict(within=True, acceleration_peak_m_s2=1.4)] * 2
    assert all(panels.start_stop_terms(st, sp).values()) and len(panels.start_stop_terms(st, sp)) == 8
    ok = panels.start_stop_terms([dict(reached=True, strides=2.3, acceleration_peak_m_s2=1.2), st[1]], sp)
    assert [k for k, v in ok.items() if not v] == ["start1_speed_up"]
    ok = panels.start_stop_terms(st, [sp[0], dict(within=False, acceleration_peak_m_s2=1.6)])
    assert [k for k, v in ok.items() if not v] == ["stop2_stopped", "stop2_acceleration"]
    ok = panels.start_stop_terms(st[:1], sp[:1])  # the second cycle never reached
    assert [k for k, v in ok.items() if not v] == ["start2_speed_up", "start2_acceleration", "stop2_stopped",
                                                   "stop2_acceleration"]
    assert tuple(ok) == panels.START_STOP_TERMS


def test_sway_on_a_synthetic_walk():
    """LOOK_RULE's sway: the lateral peak to peak within each whole stride, lateral to the walk's displacement."""
    t = np.arange(0.0, 12.0, 0.01)
    S, A = 1.0, 0.03
    x, y = 0.22 * t, A * np.sin(2 * np.pi * t / S)
    for ang in (0.0, 0.7):  # a walk in any direction
        c, s = math.cos(ang), math.sin(ang)
        rec = _trace(t, np.column_stack([c * x - s * y, s * x + c * y]))
        sw = panels.sway(rec, 1.0, 11.0, 0.4, S)
        assert sw["strides"] == 10 and sw["mean_H"] == pytest.approx(2 * A / 0.4, rel=2e-3)


# ------------------------------------------------------------------------------------------------
# Real episodes (a few simulated seconds each)
# ------------------------------------------------------------------------------------------------
EPISODE_KEYS = {"seed", "fell", "fall_time_s", "fall_cause", "fall_onset_s", "fall_candidates", "limits_before_onset",
                "survival_s", "planned_s", "strides", "worst_sustained_ratio", "worst_speed_ratio",
                "tilt_from_home_max_deg", "joint_margin_min_rad", "joint_shortfall_rad", "joint_worst", "estimator",
                "ankle_cap_residual", "captures", "phase_rate_range", "success", "succeeded", "failed_terms",
                "measures", "relaxed", "wall_s"}


@needs_run
def test_rows_are_the_g5_selections(rows):
    """ROWS_RULE: R0 no gains and no feedback; R1 the body-level G5 selection; R2/s each seed's own selection."""
    doc = json.loads((RUN / BODY / "tune.json").read_text())
    assert rows["R0"].terms == () and rows["R0"].ankle_cap == () and rows["R0"].hard_only and rows["R0"].mode == "R1"
    g, info = env.tuned_gains(RUN / BODY, BODY)
    assert np.array_equal(rows["R1"].gains, g) and rows["R1"].hard_only and rows["R1"].mode == "R1"
    for s in ("42", "43"):
        r = rows[f"R2/{s}"]
        want = [doc["seeds"][s]["selected_gains"][n] for n in control.GAIN_NAMES]
        assert r.gains.tolist() == want and r.gains_info["candidate"] == doc["seeds"][s]["selected"]
        assert not r.hard_only and r.mode == "R2" and r.terms == control.TERMS
        assert panels.row_from_spec(r.spec()).describe() == r.describe()
    key = panels._portable(rows["R1"].describe())
    assert not any(str(v).startswith("/") for v in key["gains_info"].values())


@needs_run
def test_r0_row_is_r0_on_the_commanded_clock():
    """ROWS_RULE: control.Tracker with no feedback term and no ankle cap commands exactly R0's targets: at rate 1 it is
    control's R0 bit for bit."""
    p = plant.build(BODY, "hard", settle=False)
    ref = Reference.load(RUN / BODY / "refs" / "nominal.npz")
    track = control.Track(p, control.omega0(p, ref.q_home)).add_periodic(ref, 2.0)
    a = control.run(control.Tracker(p, track, "R0"), 0.6)
    qa = p.d.qpos.copy()
    b = control.run(control.Tracker(p, track, "R1", terms=(), ankle_cap=()), 0.6)
    assert np.array_equal(qa, p.d.qpos) and a["worst_speed_ratio"] == b["worst_speed_ratio"]


@needs_run
def test_one_tiny_episode_per_row_type(ctx, rows):
    """R0, R1, R2 and a policy row (a zero residual on R2) each run a 1 s stand; the zero residual is R2 bit for bit
    on a short walk (the row interface a policy row implements)."""
    stand = dataclasses.replace(COND["stand/hard"], seconds=1.0)
    for name in ("R0", "R1", "R2/42"):
        e = rows[name].run(ctx, stand, 3042)
        assert set(e) == EPISODE_KEYS and e["succeeded"] and e["survival_s"] == pytest.approx(1.0)
        assert list(e["success"]) == list(panels.terms_of(stand)) and e["relaxed"] == []
    r3 = panels.TrackerRow("R3/zero", "R2", rows["R2/42"].gains, rows["R2/42"].gains_info, policy=panels.ZeroResidual())
    walk = dataclasses.replace(COND["walk/nominal"], seconds=1.5)
    a, b = rows["R2/42"].run(ctx, walk, 3043), r3.run(ctx, walk, 3043)
    for k in EPISODE_KEYS - {"wall_s"}:
        assert a[k] == b[k], k
    assert r3.describe()["policy"] == dict(policy="ZeroResidual") and r3.spec()["zero_residual"]


@needs_run
def test_new_behaviours_on_real_episodes(ctx, rows):
    """A stand on low pile, a standing push and the carpet walk on low pile (cut short), each with the record format;
    the push's force is the plan's: m x 0.5 x the Stage 0 forward bound / 0.1 s, along the direction."""
    e = rows["R2/42"].run(ctx, dataclasses.replace(COND["stand/low"], seconds=1.0), 3042)
    assert e["succeeded"] and set(e) == EPISODE_KEYS
    push = dataclasses.replace(COND["push/standing_left"], push_at=0.3, push_spread=0.2, recovery_s=0.3, hold_s=0.2)
    e = rows["R1"].run(ctx, push, 3042)
    p = ctx.plant("hard", None)
    m = float(p.m.body_subtreemass[p.trunk])
    pu = e["measures"]["push"]
    assert pu["dv_m_s"] == pytest.approx(0.5 * env.PUSH_BOUND_M_S[BODY]) and pu["bound_m_s"] == 0.487
    assert pu["force_N"] == pytest.approx([0.0, m * 0.5 * 0.487 / 0.1], abs=1e-9)
    assert 0.3 <= pu["onset_s"] < 0.5 and pu["onset_s"] == pytest.approx(0.3 + 0.2 * panels.push_u(push, 3042))
    assert e["survival_s"] == pytest.approx(panels.PUSH_S + 0.3 + 0.2 + 0.3 + 0.2)
    assert set(e["measures"]["recovery"]) >= {"after_s", "recovered", "threshold_m_s"}
    assert "recovered" in e["success"] and "recovered" in e["failed_terms"]  # 0.3 s is not enough to settle
    cp = dataclasses.replace(COND["carpet/low"], seconds=1.5)
    ec = rows["R2/42"].run(ctx, cp, 3042)
    assert set(ec) == EPISODE_KEYS and not ec["fell"] and not ec["succeeded"]  # cut before its window
    assert {"toe_target", "no_toe_catch"} <= set(ec["success"])
    assert ec["measures"]["carpet"]["target_m"] == pytest.approx(carpet.carpet_target(ctx.plant("low", 0.54).meta["H_m"]))
    b = ctx.built(cp)
    floor = statics.gid(b.p.m, "floor")
    assert b.recorder().foot_tables is env.foot_tables and float(b.p.m.geom_friction[floor, 0]) == pytest.approx(0.54)
    e = rows["R2/42"].run(ctx, dataclasses.replace(COND["stand/hard"], seconds=0.4), 3042, relax=("slip",))
    assert e["relaxed"] == ["slip"]
    rec = panels.condition_record(BODY, rows["R2/42"], cp, [ec], dict(plan=[], seconds=1.5, inputs=dict(
        files={}, records={})), panels.SEEDS[:1], {}, dict(dir="x"), 1.0)
    assert json.loads(json.dumps(panels._jsonable(rec)))["counts"]["judged"] is False
    assert set(rec["rules"]) >= {"carpet", "walk", "success", "look", "pass_", "verdict"}


@needs_run
def test_a_recorded_g5_episode_reproduces_through_the_runner():
    """60-N-H-10's G5 validation episode 2010 at the tuning start (its walk fell at 4.896 s) is the panels' walk on
    seed 2010: every recorded value equal."""
    body = "60-N-H-10"
    if not (RUN / body / "refs" / "nominal.npz").exists():
        pytest.skip("no references for 60-N-H-10")
    doc = json.loads((RUN / body / "tune.json").read_text())
    want = next(r for r in doc["candidates"]["start"]["runs"] if r["episode"] == 2010)["episodes"]["walk"]
    _, docs = tune.read_inputs(RUN / body, panels.panel_files())
    row = panels.TrackerRow("R2/start", "R2", control.START_GAINS, dict(source="the tuning start"))
    e = row.run(panels.Context(body, RUN / body, docs), COND["walk/nominal"], 2010)
    assert want["fell"] and e["fall_time_s"] == want["fall_time_s"] == pytest.approx(4.896)
    for k in ("fell", "fall_time_s", "fall_cause", "succeeded", "success", "worst_sustained_ratio",
              "worst_speed_ratio", "estimator", "ankle_cap_residual", "joint_margin_min_rad", "joint_shortfall_rad",
              "joint_worst"):
        assert e[k] == want[k], k
    for k in ("survived_frac", "dcm_error_rms_m", "window_sustained_ratio", "start", "stop_reading"):
        assert json.loads(json.dumps(e["measures"][k])) == want[k], k
    assert {k: e["measures"]["window"][k] for k in want["window"]} == want["window"]


# ------------------------------------------------------------------------------------------------
# Running: records, resume, the command line, G7
# ------------------------------------------------------------------------------------------------
@pytest.fixture()
def tiny(monkeypatch):
    """The panels cut to one 0.5 s stand on hard floor (the condition list is monkeypatched)."""
    cut = [dataclasses.replace(COND["stand/hard"], seconds=0.5)]
    monkeypatch.setattr(panels, "conditions", lambda: cut)
    return cut[0]


@needs_run
def test_records_and_resume(tmp_path, tiny, monkeypatch):
    """A condition's record is written with its key and skipped while the key holds; a changed source re-runs it."""
    log = []
    doc = panels.evaluate(BODY, tmp_path, rows=("R0", "R2/42"), episodes=2, inputs=RUN, log=log.append)
    s = doc["sessions"][-1]
    assert s["episodes_run"] == 4 and not s["skipped"] and len(log) == 2
    path = panels.record_path(tmp_path, BODY, "R2/42", tiny)
    rec = json.loads(path.read_text())
    assert rec["counts"]["episodes"] == 2 and rec["counts"]["pass_"] is None and rec["seeds"] == [3042, 3043]
    assert rec["key"]["sources"] == panels.panel_sources()["sha256"] and rec["key"]["seeds"] == [3042, 3043]
    assert rec["key"]["inputs"]["files"] == {f: plant.sha256(RUN / BODY / f) for f in panels.files_of(tiny)}
    assert doc["grid"]["stand"]["hard"]["rows"]["R2/42"]["survived"] == 2 and doc["G7"]["pass_"]
    doc = panels.evaluate(BODY, tmp_path, rows=("R0", "R2/42"), episodes=2, inputs=RUN, log=log.append)
    assert doc["sessions"][-1]["episodes_run"] == 0 and len(doc["sessions"][-1]["skipped"]) == 2
    monkeypatch.setattr(panels, "panel_sources", lambda files=None: dict(sha256="changed", rule="", modules=[]))
    doc = panels.evaluate(BODY, tmp_path, rows=("R2/42",), episodes=2, inputs=RUN, log=log.append)
    assert doc["sessions"][-1]["episodes_run"] == 2 and len(doc["sessions"]) == 3


@needs_run
def test_eval_command_line(tmp_path, tiny, capsys):
    """``eval`` runs the panels and prints the grid; ``--dry-run`` lists the episodes and runs none."""
    assert cli(["eval", "--body", BODY, "--rows", "R2/43", "--episodes", "1", "--dry-run", "--no-measure", "--inputs",
                str(RUN), "--out", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "R2/43" in out and "stand/hard" in out and "1 episodes" in out and "to run" in out
    assert not (tmp_path / BODY / "eval").exists()
    assert cli(["eval", "--body", BODY, "--rows", "R2/43", "--episodes", "1", "--inputs", str(RUN), "--out",
                str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "stand/hard" in out and "G7 pass" in out
    assert panels.record_path(tmp_path, BODY, "R2/43", tiny).exists()


def test_g7_gate_and_the_panel_sources():
    """G7_RULE on 60-N-T-10's nominal reference: sway and cadence reproduce Stage 0; the panels' sources follow their
    imports."""
    if not (RUN / BODY / "refs" / "nominal.npz").exists():
        pytest.skip("no references")
    g = panels.g7(BODY, RUN / BODY)
    assert g["pass_"] and g["plan"] == dict(sway_H=0.168, cadence_steps_s=1.96)
    assert abs(g["difference"]["sway_H"]) <= 5e-4 and abs(g["difference"]["cadence_steps_s"]) <= 5e-3
    assert g["measured"]["stride_H"] == pytest.approx(0.55, abs=0.005)
    assert g["measured"]["head_amplitude_rad"] == pytest.approx(0.05, abs=1e-3)
    assert g["measured"]["tail_amplitude_rad"] == pytest.approx(0.12, abs=1e-3)
    mods = panels.panel_sources()["modules"]
    assert {"panels.py", "env.py", "gates.py", "tune.py", "control.py", "metrics.py"} <= set(mods)
    assert "train.py" not in mods and "policy.py" not in mods


@needs_run
def test_attribution_runs_on_the_falls_of_r2_rows(tmp_path, monkeypatch):
    """--attribute lifts each candidate limit of an R2 condition's falls over the same seeds (PANEL_ATTRIBUTION_RULE);
    here a standing push eight times the plan's (monkeypatched) makes R2 fall within a tiny episode."""
    cut = dataclasses.replace(COND["push/standing_forward"], push_at=0.3, push_spread=0.0, recovery_s=0.8, hold_s=0.2)
    monkeypatch.setattr(panels, "conditions", lambda: [cut])
    monkeypatch.setattr(panels, "PUSH_FRAC", 4.0)
    doc = panels.evaluate(BODY, tmp_path, rows=("R2/42",), episodes=2, attribute=True, inputs=RUN, log=lambda m: None)
    rec = json.loads(panels.record_path(tmp_path, BODY, "R2/42", cut).read_text())
    assert rec["counts"]["falls"] == 2 and rec["attribution"] is not None
    cands, lift = panels.attribution_plan(rec)
    a = rec["attribution"]
    assert a["candidates"] == cands and set(a["lifted"]) == set(lift) and a["null"] is None
    assert a["episodes"] == 2 * len(lift) == doc["sessions"][-1]["attribution_episodes_run"]
    assert all(len(v) == 2 and all(x["seed"] in (3042, 3043) for x in v) for v in a["runs"].values())
    assert a["label"] == "controller-limited" or a["label"].startswith("body-limited")
    again = panels.evaluate(BODY, tmp_path, rows=("R2/42",), episodes=2, attribute=True, inputs=RUN, log=lambda m: None)
    assert again["sessions"][-1]["episodes_run"] == again["sessions"][-1]["attribution_episodes_run"] == 0


@needs_run
def test_a_policy_acts_every_env_step(ctx, rows):
    """The ``Policy`` hook: ``begin`` after the reset, ``act`` after every ``every``-th physics step."""

    class Counting(panels.Policy):
        def __init__(self):
            self.begun, self.steps = 0, []

        def begin(self, tk, built, seed):
            self.begun += 1
            assert tk.k == 0 and built.cond.key == "stand/hard" and seed == 3042

        def act(self, tk, k):
            self.steps.append(k)
            assert tk.k == k + 1

    pol = Counting()
    row = panels.TrackerRow("R3/count", "R2", rows["R2/42"].gains, policy=pol)
    row.run(ctx, dataclasses.replace(COND["stand/hard"], seconds=0.1), 3042)
    assert pol.begun == 1 and pol.steps == [9, 19, 29, 39, 49] and pol.every == env.ENV_SUBSTEPS
