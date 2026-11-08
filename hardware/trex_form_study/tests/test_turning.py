"""Tests for designlab.turning (prescribed turning trajectories and their loads).

Run from hardware/trex_form_study:  python -m pytest tests/test_turning.py -q
Kept fast with a coarse stride (80 frames) and one multibody-ZMP pass; the CLI uses the full spec.
"""

from __future__ import annotations

import dataclasses
import inspect
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import designlab  # noqa: E402,F401  (selects the GL backend before mujoco)
from designlab import catalog, envelope, gait, sizing, statics, turning  # noqa: E402
from designlab.params import parse  # noqa: E402
from designlab.pipeline import build_candidate  # noqa: E402

CID = "60-M-T-10"
FAST = dataclasses.replace(sizing.NOMINAL, n=80, zmp_iters=1)
FAST_BRISK = dataclasses.replace(sizing.BRISK, n=80, zmp_iters=1)
# gait.generate's settings and rules: CoM tolerance and passes, phase rule, stride-boundary unwrap
REPRO = dict(com_tol=2e-4, com_iters=4, phase_tol=0.0, unwrap="gait")


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """60-M-T-10 as sized in Stage A (the default build if the round-one record is absent)."""
    out = tmp_path_factory.mktemp("turning")
    rec_path = ROOT / "round1" / "stage_a" / "candidates" / f"{CID}.json"
    if rec_path.exists():
        from designlab import round1

        rec = json.loads(rec_path.read_text())
        sel = dict(rec["selection"])
        sel["mass_guess"] = round1._final_mass_guess(rec)
        res = build_candidate(parse(CID), out, selection=sel, B=rec["B"], write=False)
    else:
        res = build_candidate(parse(CID), out, write=False)
    return res, out / f"{CID}.xml"


@pytest.fixture(scope="module")
def model(built):
    res, _ = built
    return res["model"], res["q_home"], res["design"].sk.H


@pytest.fixture(scope="module")
def turns(model):
    M, q_home, H = model
    out = {}
    for rate in (0.35, -0.35):
        tr = turning.generate(M, q_home, H, FAST, yaw_rate=rate)
        idr = turning.inverse_dynamics(M, tr)
        out[rate] = (tr, idr, turning.summarize(M, tr, idr))
    return out


def test_path_frame_is_a_circle_with_the_straight_and_in_place_limits():
    t = np.linspace(0.0, 3.0, 61)
    v, w = 0.25, 0.4
    P, th = turning.path_frame(t, v, w, (0.1, -0.2))
    centre = np.array([0.1, -0.2 + v / w])
    assert np.allclose(np.linalg.norm(P - centre, axis=1), v / w, atol=1e-12)
    assert np.allclose(th, w * t)
    d = np.gradient(P, t, axis=0)  # direction of travel = heading
    assert np.allclose(np.arctan2(d[1:-1, 1], d[1:-1, 0]), th[1:-1], atol=2e-3)
    assert np.allclose(np.linalg.norm(d[1:-1], axis=1), v, rtol=2e-3)
    P0, _ = turning.path_frame(t, v, 1e-12, (0.0, 0.0))  # deviates from a line by v w t^2 / 2 ~ 1e-12 m
    assert np.allclose(P0, np.column_stack([v * t, np.zeros_like(t)]), atol=1e-9)
    assert np.array_equal(turning.path_frame(t, v, 0.0, (0.0, 0.0))[0], np.column_stack([v * t, np.zeros_like(t)]))
    Pin, _ = turning.path_frame(t, 0.0, w, (0.3, 0.4))
    assert np.allclose(Pin, [0.3, 0.4])


def test_footprints_lie_on_concentric_arcs_and_are_yawed_with_the_body():
    feet0 = {"left": np.array([0.01, 0.08, 0.02]), "right": np.array([0.01, -0.08, 0.02])}
    T, duty, v, w = 1.0, 0.6, 0.22, 0.35
    t = np.arange(400) * T / 400
    s = turning.foot_schedule(t, T, duty, 0.04, feet0, np.zeros(2), v, w)
    centre = np.array([0.0, v / w])
    radii = {}
    for side in statics.SIDES:
        st = ~s["swing"][side]
        p = s["pos"][side]
        rel0 = feet0[side][:2] - centre
        r = np.linalg.norm(p[st, :2] - centre, axis=1)
        assert np.allclose(r, np.linalg.norm(rel0), atol=1e-12)
        bearing = np.arctan2(p[st, 0] - centre[0], -(p[st, 1] - centre[1])) - np.arctan2(rel0[0], -rel0[1])
        assert np.allclose(np.angle(np.exp(1j * (s["yaw"][side][st] - bearing))), 0.0, atol=1e-12)
        assert np.allclose(p[st, 2], feet0[side][2])
        assert np.max(np.linalg.norm(np.diff(p, axis=0), axis=1)) < 0.01  # swing joins footprints without jumps
        assert np.max(p[:, 2]) == pytest.approx(feet0[side][2] + 0.04, abs=1e-4)
        radii[side] = np.linalg.norm(rel0)
    assert radii["left"] < v / w < radii["right"]  # left turn: left foot inside, right foot outside
    # turn in place: footprints rotate about the hip midpoint
    s0 = turning.foot_schedule(t, T, duty, 0.04, feet0, np.zeros(2), 0.0, 0.5)
    for side in statics.SIDES:
        st = ~s0["swing"][side]
        assert np.allclose(np.linalg.norm(s0["pos"][side][st, :2], axis=1), np.linalg.norm(feet0[side][:2]), atol=1e-12)


def test_frames_on_a_lift_off_or_touchdown_are_classified_by_the_event():
    """With gait.generate's rule (u < 1 - duty) rounding of t / T decides the frames that fall exactly on a
    touchdown, so some stride periods lose double-support frames and the two feet differ; the event rule does
    not depend on the period."""
    feet0 = {"left": np.array([0.0, 0.08, 0.02]), "right": np.array([0.0, -0.08, 0.02])}
    n, duty = 160, 0.6
    expected = int(round(2 * (duty - 0.5) * n))
    counts_old, counts_new = set(), set()
    for T in np.linspace(0.8, 2.4, 41):
        t = np.arange(n) * (T / n)
        for tol, counts in ((0.0, counts_old), (turning.PHASE_TOL, counts_new)):
            s = turning.foot_schedule(t, T, duty, 0.04, feet0, np.zeros(2), 0.2, 0.3, phase_tol=tol)
            counts.add(sum(1 for p in s["phases"] if len(p) == 2))
            if tol:
                sw = {sd: np.flatnonzero(s["swing"][sd]) for sd in statics.SIDES}
                assert list(sw["left"]) == list(range(0, 64)) and list(sw["right"]) == list(range(80, 144))
    assert counts_new == {expected}
    assert min(counts_old) < expected  # the rounding effect the event rule removes


def test_rotating_frame_pendulum_solution_satisfies_the_pendulum_in_the_world():
    T, n, w0, w, v = 1.0, 400, 5.4, 0.5, 0.22
    dt = T / n
    t = np.arange(n) * dt
    P, th = turning.path_frame(t, v, w, (0.0, 0.0))
    ph = 2 * np.pi * t / T
    q = np.column_stack([0.03 + 0.05 * np.cos(ph) + 0.01 * np.sin(2 * ph), 0.04 * np.sin(ph) - 0.005])
    zref = P + np.einsum("kij,kj->ki", np.array([turning._rot2(a) for a in th]), q)
    com, _ = turning.lipm_com(zref, P, th, dt, w0, w, v)
    PT = turning.path_frame([T], v, w, (0.0, 0.0))[0][0]
    R = turning._rot2(w * T)
    nxt = PT + R @ com[0]  # c(T) = D c(0)
    prv = R.T @ (com[-1] - PT)  # c(-dt) = D^-1 c(T - dt)
    c = np.vstack([prv, com, nxt])
    acc = (c[2:] - 2 * c[1:-1] + c[:-2]) / dt**2
    res = acc - w0**2 * (com - zref)
    assert np.max(np.abs(res)) < 1e-3 * np.max(np.abs(acc))


def test_exact_unwrap_matches_explicit_neighbouring_strides(model, turns):
    M, _, _ = model
    tr = turns[0.35][0]
    phi, P0, PT = tr["transform"]
    qs, dt = tr["q"], tr["dt"]
    prev = np.array([turning.move_pose(q, -phi, PT, P0) for q in qs])
    nxt = np.array([turning.move_pose(q, phi, P0, PT) for q in qs])
    three = np.vstack([prev, qs, nxt])
    n = len(qs)
    v = np.zeros(M.m.nv)
    qd3 = np.zeros((3 * n, M.m.nv))
    for k in range(1, 3 * n - 1):
        mujoco.mj_differentiatePos(M.m, v, 2 * dt, three[k - 1], three[k + 1])
        qd3[k] = v
    qdd3 = (qd3[n + 1 : 2 * n + 1] - qd3[n - 1 : 2 * n - 1]) / (2 * dt)
    qd, qdd = turning.derivatives(M, qs, dt, tr["transform"])
    assert np.allclose(qd, qd3[n : 2 * n], atol=1e-9)
    assert np.allclose(qdd, qdd3, atol=1e-6)


def test_zero_yaw_rate_reproduces_gait_generate(model):
    M, q_home, H = model
    ref = gait.generate(M, q_home, H, FAST)
    s_ref = gait.summarize(M, ref, gait.inverse_dynamics(M, ref))
    # gait.generate's settings, rules and split: the same poses and loads to rounding
    tr = turning.generate(M, q_home, H, FAST, yaw_rate=0.0, **REPRO)
    s = turning.summarize(M, tr, turning.inverse_dynamics(M, tr, split="statics", ds_friction=False))
    assert np.max(np.abs(tr["q"] - ref["q"])) < 1e-12
    for nm in M.names:
        for key in ("rms", "peak", "speed_peak"):
            assert s["joints"][nm][key] == pytest.approx(s_ref["joints"][nm][key], rel=1e-9, abs=1e-12), (nm, key)
    # the screen's tighter CoM tolerance, exact unwrap and split leave hip and knee RMS within a few percent
    tr2 = turning.generate(M, q_home, H, FAST, yaw_rate=0.0)
    s2 = turning.summarize(M, tr2, turning.inverse_dynamics(M, tr2, ds_friction=False))
    for nm in M.names:
        if any(j in nm for j in ("hip", "knee")):
            assert s2["joints"][nm]["rms"] == pytest.approx(s_ref["joints"][nm]["rms"], rel=0.02), nm
    with pytest.raises(ValueError):
        turning.generate(M, q_home, H, FAST, yaw_rate=0.3, unwrap="gait")


def test_left_turn_mirrors_right_turn(model, turns):
    """Every joint, the ankles included: the split's CoP box and friction pyramid are taken in the foot's axes,
    so mirrored turns (which differ in heading by half a stride's yaw) get mirrored splits."""
    M, _, _ = model
    sl, sr = turns[0.35][2], turns[-0.35][2]
    for nm in M.names:
        partner = turning._partner(nm)
        for key in ("rms", "peak", "speed_peak"):
            assert sl["joints"][nm][key] == pytest.approx(sr["joints"][partner][key], rel=0.01, abs=0.01), (nm, key)
    assert sl["min_cop_margin_single"] == pytest.approx(sr["min_cop_margin_single"], abs=1e-3)
    assert sl["double_support"]["min_mu_max"] == pytest.approx(sr["double_support"]["min_mu_max"], rel=0.02)
    assert sl["ik_fail_frames"] == 0 and sr["ik_fail_frames"] == 0


def test_double_support_split_is_heading_invariant_and_exact(model, turns):
    """Rotating a double-support frame about the vertical axis rotates its split (the foot-axis rule), and the
    exact QP solution is feasible and no worse than SLSQP on the same problem."""
    from scipy.optimize import minimize

    M, _, _ = model
    tr, idr, _ = turns[0.35]
    ks = [k for k, p in enumerate(tr["phases"]) if len(p) == 2]
    qd, qdd = idr["qd"], idr["qdd"]
    used_qp = 0
    for k in ks:
        _, w0, how = turning.contact_torques(M, tr["q"][k], qd[k], qdd[k], statics.SIDES)
        phi = 0.7
        q1 = turning.move_pose(tr["q"][k], phi, np.zeros(2), np.zeros(2))
        _, w1, _ = turning.contact_torques(M, q1, turning.move_velocity(qd[k], phi), turning.move_velocity(qdd[k], phi),
                                           statics.SIDES)
        R = turning._rotz(phi)
        for side in statics.SIDES:
            assert np.allclose(R @ w0[side][:3], w1[side][:3], atol=1e-6), (k, side)
            assert np.allclose(R @ w0[side][3:], w1[side][3:], atol=1e-6), (k, side)
        if how != "exact_qp":
            continue
        used_qp += 1
        rhs, J, A, b = turning._base_problem(M, tr["q"][k], qd[k], qdd[k], statics.SIDES)
        Hq, g, G, h = turning._qp_matrices(M, J, rhs, statics.SIDES)
        w = np.concatenate([w0[s] for s in statics.SIDES])
        assert np.allclose(A @ w, b, atol=1e-8) and np.all(G @ w >= h - 1e-7)
        f = lambda x: 0.5 * x @ Hq @ x + g @ x  # noqa: E731
        res = minimize(f, w, jac=lambda x: Hq @ x + g, method="SLSQP", options=dict(maxiter=300, ftol=1e-12),
                       constraints=[dict(type="eq", fun=lambda x: A @ x - b, jac=lambda x: A),
                                    dict(type="ineq", fun=lambda x: G @ x - h, jac=lambda x: G)])
        assert f(w) <= f(res.x) + 1e-7 * max(1.0, abs(f(res.x)))
    assert used_qp > 0  # touchdown and lift-off frames make the inequality constraints active


def test_double_support_friction_requirement_is_a_minimum(model, turns):
    M, _, _ = model
    tr, idr, s = turns[0.35]
    D = idr["ds"]
    ks = [k for k, p in enumerate(tr["phases"]) if len(p) == 2]
    assert np.all(D["feasible"][ks])
    assert np.all(D["min_mu"][ks] >= D["min_mu_lower"][ks] - 1e-9)
    assert np.all(D["min_mu"][ks] <= 1.02 * D["min_mu_lower"][ks] + 1e-6)  # the polyhedral bracket
    # no split needs more than the minimum-effort split where that split keeps both CoPs inside the pads
    F = idr["feet"]
    for k in ks:
        alloc = [F[sd]["mu_req_centroid"][k] for sd in statics.SIDES]
        inside = all(F[sd]["margin"][k] >= 0 for sd in statics.SIDES)
        if inside and all(np.isfinite(alloc)):
            assert D["min_mu"][k] <= max(alloc) + 1e-6
    # and double support needs far less than single support in this turn
    ss = max(s["feet"][sd]["single_support"]["mu_req_max"] for sd in statics.SIDES)
    assert s["double_support"]["min_mu_max"] < ss


def test_turn_balances_the_expected_mean_forces_and_moments(model, turns):
    M, _, _ = model
    tr, _, s = turns[0.35]
    c = s["checks"]
    assert c["vertical_force_mean_over_weight"] == pytest.approx(1.0, abs=1e-6)
    assert np.allclose(c["path_frame_force_mean_n"], c["path_frame_force_expected_n"], atol=5e-3)
    assert c["path_frame_force_mean_n"][1] == pytest.approx(M.mass() * 0.35 * tr["info"]["speed"], rel=0.05)
    assert abs(c["moment_about_turn_axis_mean_nm"]) < 1e-3 * c["moment_about_turn_axis_rms_nm"]
    assert c["trunk_yaw_rate_mean"] == pytest.approx(0.35, abs=1e-9)
    # thigh twist moves in a turn; it does not in a straight walk
    assert s["joints"]["left_hip_yaw"]["speed_peak"] > 0.5


def test_torsion_radius_for_disc_and_off_centre_pressure():
    rng = np.random.default_rng(0)
    R = 0.05
    pts = rng.uniform(-R, R, (200000, 2))
    pts = pts[np.linalg.norm(pts, axis=1) <= R]
    w = np.ones(len(pts))
    assert turning.uniform_radius(pts, w) == pytest.approx(2 * R / 3, rel=5e-3)
    centred = turning.pressure_radius(pts, w, pts.mean(0))
    assert centred == pytest.approx(turning.uniform_radius(pts, w), rel=1e-3)
    edge = turning.pressure_radius(pts, w, np.array([0.8 * R, 0.0]))
    assert 0.0 < edge < 0.6 * centred
    assert turning.pressure_radius(pts, w, np.array([1.5 * R, 0.0])) == 0.0


def test_motor_space_by_side_stacks_to_sizing():
    names = ["neck_pitch", "tail_yaw"] + [f"{s}_{j}" for s in statics.SIDES for j in statics.LEG_JOINTS]
    rng = np.random.default_rng(1)
    tau, om = rng.normal(size=(50, len(names))), rng.normal(size=(50, len(names)))
    for layout in ("tandem", "serial"):
        ref = sizing._motor_space(names, tau, om, layout)
        mine = turning.motor_space_by_side(names, tau, om, layout)
        for g in sizing.GROUPS:
            t = np.concatenate([s[0] for s in mine[g].values()])
            w = np.concatenate([s[1] for s in mine[g].values()])
            assert np.sqrt(np.mean(t**2)) == pytest.approx(np.sqrt(np.mean(ref[g][0] ** 2)))
            assert np.max(np.abs(w)) == pytest.approx(np.max(np.abs(ref[g][1])))


def test_compare_flags_limits_increases_and_near_misses():
    act = catalog.Actuator(id="x", vendor="v", model="m", mass=1.0, diameter=0.05, length=0.05, rated_torque=10.0,
                           peak_torque=20.0, rated_speed_rpm=60.0, no_load_speed_rpm=80.0, nominal_voltage=24.0, kt=None,
                           phase_resistance=None, price=1.0, armature=0.0)
    req = dict(cont=6.0, peak_req=10.0, speed_req=2.0)
    straight = dict(cont=5.0, peak_req=10.0, speed_req=0.0)
    v = turning.compare(req, straight, act, 24.0)
    assert v["over_limit"] == ["cont"]  # 6.0 > 0.57 x 10
    assert v["increase_over_10pct"] == ["cont", "speed_req"]  # 1.2x, and new
    assert v["ratio_to_straight"]["speed_req"] is None
    assert not v["meets"]
    near = turning.compare(dict(cont=5.0, peak_req=10.9, speed_req=6.2), dict(cont=5.0, peak_req=10.0, speed_req=6.0),
                           act, 24.0)
    assert near["near_increase_flag"] == ["peak_req"] and near["increase_over_10pct"] == []
    assert near["near_limit"] == ["speed_req"] and near["over_limit"] == []  # 6.2 of 6.28 rad/s


def test_round_two_hook_feeds_the_envelope_and_the_shortlist(built, monkeypatch):
    """The output format the round-two runner consumes: ``screen_record``'s signature, the per-run ``groups``
    that ``envelope.turning_rms_from`` reads, and the summary keys ``shortlist`` reads."""
    from designlab import round2, shortlist

    assert "turning.screen_record(cid, rec, str(xml), q_home, design, scr)" in inspect.getsource(round2._turning)
    params = list(inspect.signature(turning.screen_record).parameters)
    assert params[:6] == ["cid", "rec", "xml", "q_home", "design", "scr"]
    res, xml = built
    M, q_home, design = res["model"], res["q_home"], res["design"]
    monkeypatch.setattr(sizing, "NOMINAL", FAST)
    monkeypatch.setattr(sizing, "BRISK", FAST_BRISK)
    straight = sizing.requirements(M, q_home, design.sk.H, design.cand.leg_layout)
    cases = {k: turning.CASES[k] for k in ("walk_0.35", "in_place_0.50")}
    r = turning.screen_model(M, q_home, design, FAST, cases=cases, straight=straight, xml=str(xml), validate=False,
                             brisk=FAST_BRISK, probes={})
    doc = turning._clean(dict(id=CID, record="test", H=design.sk.H, mass=M.mass(), bus=design.sel["bus"]) | r)
    rms = envelope.turning_rms_from(doc)
    assert rms is not None and set(rms) == set(sizing.GROUPS)
    assert rms["hip_yaw"] == pytest.approx(doc["requirements"]["walk_0.35"]["hip_yaw"]["rms"], rel=0.02)
    summ = turning.record_summary(doc)
    for k in ("targets_within_limits", "worst_ratio_targets", "hip_yaw_range_used_max", "mu_needed_max",
              "targets_interference_free", "targets_pass"):
        assert k in summ, k
    assert isinstance(summ["targets_within_limits"], bool) and isinstance(summ["targets_interference_free"], bool)
    assert 0.0 < summ["worst_ratio_targets"] and 0.0 < summ["hip_yaw_range_used_max"] < 1.0
    assert 0.0 < summ["mu_needed_max"] < 2.0
    assert shortlist._turn_worst(summ) == pytest.approx(summ["worst_ratio_targets"])
    gates = shortlist.gates(dict(margins={"x": dict(cont_ratio=0.5, peak_ratio=0.5, speed_ratio=0.5)},
                                 runtime=dict(minutes=30.0), walk_feasible=True), {"turning": summ})
    assert gates["turning_targets"] is summ["targets_within_limits"]
    assert summ["targets_pass"] == (summ["targets_within_limits"] and summ["targets_interference_free"]
                                    and summ["targets_ik_ok"])
    for c in cases:
        cc = doc["clearance"]["cases"][c]
        assert cc["interference_free"] == (sum(cc["interference_pairs"]) == 0)
