"""Tests of the candidate checks' shared code (sim/candidate_checks.py) on small cases: no experiment is run (the
longest simulations here are 0.2-0.5 s at zero action or a few trot steps). Needs mujoco 3.10.0, numpy, scipy, torch
and stable-baselines3 (the sweep's loaders); the policy tests also need the checkpoints in the session's scratch
folder (skipped without them). Files are written only under the scratch folder's candidates/pytest/. Run by hand:
    PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider hardware/triceratops_build/sim/tests/test_candidate_checks.py
"""

import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

SIM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIM))

mujoco = pytest.importorskip("mujoco")
pytest.importorskip("scipy")
import candidate_checks as C  # noqa: E402

OUT = C.CAND_SCRATCH / "pytest"
HAVE_POLICIES = all((C.POLICY_DIRS[r]).exists() for r in C.E1_POLICIES)
needs_policies = pytest.mark.skipif(not HAVE_POLICIES, reason="policy checkpoints not in this session's scratch")


@pytest.fixture(scope="module")
def out():
    OUT.mkdir(parents=True, exist_ok=True)
    return OUT


@pytest.fixture(scope="module")
def geom():
    return C.ClashGeometry()


@pytest.fixture(scope="module")
def packaging():
    return json.loads(C.PACKAGING_JSON.read_text())


# ------------------------------------------------------------------------------------------------------ E1 geometry
def _row(packaging, legs, a, b):
    rows = packaging["legs_to_each_other"][legs]["pairs_nearer_than_10mm"]
    hit = [r for r in rows if {r["a"], r["b"]} == {a, b}]
    assert hit, (legs, a, b)
    return hit[0]


def test_hind_soles_at_the_inward_stop_reproduce_the_record(geom, packaging):
    r = _row(packaging, "left_hind / right_hind", "left_hind_sole", "right_hind_sole")
    least, aside, parts = geom.clearance(C.deg_pose(r["at_deg"]))["10"]
    assert abs(1000 * least - r["min_mm"]) < 0.05, (least, r["min_mm"])      # -8.49 mm (record item 10)
    assert set(parts) == {"left_hind_sole", "right_hind_sole"}
    assert 1000 * aside > 6.0                                                # soles aside: 6.55 mm (record)


def test_fore_hind_soles_and_struts_reproduce_the_record(geom, packaging):
    r = _row(packaging, "left_fore / left_hind", "left_fore_sole", "left_hind_sole")
    least = geom.clearance(C.deg_pose(r["at_deg"]))["9L"][0]
    assert abs(1000 * least - r["min_mm"]) < 0.05                            # -48.0 mm (record item 9)
    struts = _row(packaging, "left_fore / left_hind", "left_fore_lower_strut", "left_hind_lower_strut")
    aside = geom.clearance(C.deg_pose(struts["at_deg"]))["9L"][1]
    assert abs(1000 * aside - struts["min_mm"]) < 0.05                       # -19.2 mm, soles aside


def test_right_side_mirrors_the_left(geom, packaging):
    r = _row(packaging, "left_fore / left_hind", "left_fore_sole", "left_hind_sole")
    mirrored = {k.replace("left_", "right_"): v for k, v in r["at_deg"].items()}
    a = geom.clearance(C.deg_pose(r["at_deg"]))["9L"][0]
    b = geom.clearance(C.deg_pose(mirrored))["9R"][0]
    assert abs(a - b) < 1e-6


def test_neutral_pose_is_clear(geom):
    c = geom.clearance(np.zeros(16))
    assert all(v[0] > 0.005 for v in c.values()), c


def test_prefilter_equals_brute_force(geom):
    rng = np.random.default_rng(0)
    m, d = geom.m, geom.d
    lo = np.array([m.jnt_range[m.joint(n).id][0] for n in C.LOG_JOINTS])
    hi = np.array([m.jnt_range[m.joint(n).id][1] for n in C.LOG_JOINTS])
    for _ in range(6):
        q = lo + rng.random(16) * (hi - lo)
        c = geom.clearance(q)
        geom.pose(q)
        for key, (a, b) in C.CLASH_PAIRS.items():
            brute = min(min(float(mujoco.mj_geomDistance(m, d, ga, gb, geom.cap, np.zeros(6))), geom.cap)
                        for ga in geom.parts[a] for gb in geom.parts[b])
            assert abs(c[key][0] - brute) < 1e-5, (key, c[key][0], brute)


def test_batch_matches_single(geom):
    Q = np.zeros((3, 16))
    Q[1, C.LOG_JOINTS.index("left_hind_roll")] = math.radians(-18)
    Q[1, C.LOG_JOINTS.index("right_hind_roll")] = math.radians(-18)
    b = geom.batch(Q)
    for n in range(3):
        s = geom.clearance(Q[n])
        assert b["10"][n] == s["10"][0] and b["9L"][n] == s["9L"][0]


# --------------------------------------------------------------------------------------------------------- E1 stops
def test_candidate_stops_are_the_records(packaging):
    stops = C.candidate_stops()
    assert len(stops) == 17                    # LIMITS: 3 roll, 4 fore pitch, 4 hind pitch, 3 both pitch, 3 flexion
    by = {s["label"]: s for s in stops}
    roll17 = [s for s in stops if s["clash"] == "10" and s["limits_deg"]["left_hind_roll"] == (-17, None)][0]
    assert set(roll17["limits_deg"]) == {"left_hind_roll", "right_hind_roll"}
    assert abs(roll17["record_least_mm_full_range"] - 4.79) < 0.01
    fore5 = [s for s in stops if s["limits_deg"].get("left_fore_pitch") == (None, 5)
             and "left_hind_pitch" not in s["limits_deg"]][0]
    assert set(fore5["limits_deg"]) == {"left_fore_pitch", "right_fore_pitch"}
    assert abs(fore5["record_least_mm_full_range"] - 0.53) < 0.01
    assert len(by) == 17


def test_apply_stop_clips_and_counts():
    Q = np.zeros((4, 16))
    k = C.LOG_JOINTS.index("right_hind_roll")
    Q[:, k] = np.radians([-18.0, -17.5, -16.0, 10.0])
    Qc, ch = C.apply_stop(Q, {"left_hind_roll": (-17, None), "right_hind_roll": (-17, None)})
    assert ch.tolist() == [True, True, False, False]
    assert np.allclose(np.degrees(Qc[:2, k]), -17.0)
    Qc, ch = C.apply_stop(Q, {"right_hind_roll": (None, 0)})
    assert ch.tolist() == [False, False, False, True]


def test_evaluate_stop_and_coupled_limit(geom, packaging):
    r = _row(packaging, "left_hind / right_hind", "left_hind_sole", "right_hind_sole")
    Q = np.vstack([np.zeros((100, 16)), C.deg_pose(r["at_deg"])[None]])
    base = geom.batch(Q)
    stops = {s["limits_deg"]["left_hind_roll"][0]: s for s in C.candidate_stops() if s["clash"] == "10"}
    e17 = C.evaluate_stop(geom, Q, base, stops[-17])
    assert e17["clipped"] == 1 and abs(e17["least_mm_after"] - 4.79) < 0.05 and not e17["acceptable"]
    e16 = C.evaluate_stop(geom, Q, base, stops[-16])
    assert e16["clipped_share"] == pytest.approx(1 / 101) and e16["least_mm_after"] > 18.0 and e16["acceptable"]
    cl = C.coupled_limit(base, "10")
    assert cl["clipped"] == 1 and cl["acceptable"]
    Q2 = np.vstack([np.zeros((10, 16)), np.repeat(C.deg_pose(r["at_deg"])[None], 2, axis=0)])
    assert not C.coupled_limit(geom.batch(Q2), "10")["acceptable"]          # 2 of 12 samples


def test_e1_verdict_needs_two_controllers():
    ok = {"acceptable": True, "clipped_share": 0.0, "least_mm_after": 9.0}
    bad = {"acceptable": False, "clipped_share": 0.02, "least_mm_after": 9.0}
    pc = {f"c{i}": {"least_mm": {"9": 3.0, "10": 3.0}, "stops": {"s": ok, "t": ok}} for i in range(5)}
    pc["c0"]["least_mm"]["9"] = -1.0
    v = C.e1_verdict(pc)
    assert not v["reached"]["9"]["REACHED"] and not v["reached"]["10"]["REACHED"]
    pc["c3"]["least_mm"]["9"] = -0.1
    pc["c2"]["stops"]["t"] = bad
    v = C.e1_verdict(pc)
    assert v["reached"]["9"]["REACHED"] and v["acceptable_stops"]["s"]["ACCEPTABLE"]
    assert not v["acceptable_stops"]["t"]["ACCEPTABLE"]


# -------------------------------------------------------------------------------------------------------- E1 logger
def _ec(**kw) -> dict:
    """A JSON-shaped env config on the adopted model (as a run record stores it)."""
    S = C._sweep()
    cfg = S.G.flat_floor(S.TrikeEnvConfig.from_dict({"model_path": C.MODEL.name, "obs_ankle_angles": True, **kw}))
    return json.loads(json.dumps(cfg.to_dict()))


@pytest.fixture(scope="module")
def adopted_env():
    return C.make_env(C.policy_env_config(_ec()))


def test_logger_samples_every_10ms_and_splits_like_the_demo_course(adopted_env):
    env = adopted_env
    lg = C.StepLogger(env, window_s=0.1)
    idx = np.asarray(env._leg_act)
    eta = np.array([C.ETA_OLD[C.axis_of(a)] for a in C.LEG_ACTUATORS])
    raw = {"tau": [], "om": [], "t": []}

    def demo_like(e):
        raw["t"].append(e.data.time)
        raw["tau"].append(e.data.actuator_force[idx].copy())
        raw["om"].append(e.data.actuator_velocity[idx].copy())
    env.extra_hooks[:] = [lg, demo_like]
    env.reset_hooks[:] = [lg]
    env.reset(seed=3, options={"command_speed": 0.0, "delay_steps": 10})
    for _ in range(25):
        env.step(np.zeros(12))
    res = lg.result()
    assert res["q"].shape == (50, 16)
    assert np.allclose(np.asarray(res["t"]) / 0.01, np.round(np.asarray(res["t"]) / 0.01))
    t, tau, om = np.asarray(raw["t"]), np.asarray(raw["tau"]), np.asarray(raw["om"])
    w = t >= 0.1
    p = tau[w] * om[w]
    td = np.where(p >= 0, tau[w] / eta, tau[w] * eta)              # rl/demo_course.py measure_segment's rule
    assert np.allclose(res["drive_sq"], (td ** 2).mean(0), rtol=1e-12, atol=0)
    assert np.allclose(res["sq_motor"] / eta ** 2 + res["sq_brake"] * eta ** 2, res["drive_sq"], rtol=1e-12)
    assert res["rpm_hist"].sum(axis=1).tolist() == [w.sum()] * 12
    assert np.allclose(res["rpm_peak"], (np.abs(om[w]) * 60 / (2 * math.pi)).max(0))
    env.extra_hooks[:], env.reset_hooks[:] = [], []


def test_policy_env_config_is_the_sweeps(b025):
    S = C._sweep()
    ec = _ec()
    a = C.policy_env_config(ec).to_dict()
    b = S.build_env(ec).cfg.to_dict()
    assert a == b
    assert C.policy_env_config(ec, b025[0], physics_dt=0.00025).frame_skip == 80
    assert C.delay_steps_for(0.00025) == 40 and C.delay_steps_for(0.001) == 10 and C.delay_steps_for(0.000125) == 80


def test_policy_env_config_refuses_a_model_at_another_step(b025):
    ec = _ec()
    with pytest.raises(ValueError):          # the adopted model is at 1 ms: frame_skip 80 would be 80 ms control
        C.policy_env_config(ec, C.MODEL, physics_dt=C.SERIES_DT)
    with pytest.raises(ValueError):          # and the 0.25 ms copy cannot be run as if it were at 1 ms
        C.policy_env_config(ec, b025[0], physics_dt=0.001)
    env = C.make_env(C.policy_env_config(ec, b025[0], physics_dt=C.SERIES_DT))
    assert env.physics_dt == pytest.approx(C.SERIES_DT) and env.dt == pytest.approx(0.02)


def _short(cfg, seconds=0.1):
    return cfg.replace(episode_seconds=seconds)


def test_policy_episode_delay_is_10ms_at_any_step(b025):
    ec = _ec()
    zero = lambda obs: np.zeros(12)  # noqa: E731
    for path, dt, steps in ((C.MODEL, 0.001, 10), (b025[0], C.SERIES_DT, 40)):
        env = C.make_env(_short(C.policy_env_config(ec, path, dt)))
        assert C.policy_point("walk_0.25", dt)["delay_ms"] == steps          # the sweep's key counts physics steps
        lg = C.StepLogger(env)
        rec = C.policy_episode(env, 5, "push_right_0.5", controller=zero, hooks=(lg,))
        assert rec["delay_steps"] == steps and rec["delay_s"] == pytest.approx(0.010)
        assert rec["physics_dt"] == pytest.approx(dt) and rec["control_dt"] == pytest.approx(0.02)
        assert not rec["fell"] and not rec["unstable"]
        assert lg not in env.extra_hooks and lg not in env.reset_hooks           # attached for that episode only
        assert lg.result()["q"].shape == (10, 16)                                # 0.1 s at 10 ms


def test_step_logger_drops_non_finite_poses(adopted_env):
    lg = C.StepLogger(adopted_env)
    lg.t += [0.01, 0.02, 0.03]
    lg.q += [np.zeros(16, np.float32), np.full(16, np.nan, np.float32), np.ones(16, np.float32)]
    res = lg.result()
    assert res["q"].shape == (2, 16) and res["q_nonfinite_dropped"] == 1 and res["t"].tolist() == [0.01, 0.03]


def test_trot_sweep_config_is_fore128_walk60s():
    S, F = C._sweep(), C._f128()
    a = C.trot_sweep_config(C.MODEL).to_dict()
    b = S.build_env(F.env_config(C.MODEL.name), None, False).cfg.to_dict()
    assert a == b


# ---------------------------------------------------------------------------------------------------- E2(a) desk math
def test_split_mapping_and_bounds():
    rng = np.random.default_rng(1)
    sm, sb = rng.random(12) * 30, rng.random(12) * 10
    eo = np.array([C.ETA_OLD[C.axis_of(a)] for a in C.LEG_ACTUATORS])
    stored = sm / eo ** 2 + sb * eo ** 2
    for r in C.REDUCTIONS:
        for eta in (0.90, 0.95, 0.96, 0.98):
            ex = C.drive_sq_from_split(sm, sb, r, eta)
            lo, hi = C.drive_sq_bounds(stored, eo, r, eta)
            assert np.all(lo <= ex * (1 + 1e-12)) and np.all(ex <= hi * (1 + 1e-12))
    assert np.allclose(C.drive_sq_from_split(sm, sb, 1.0, 0.9), sm / 0.81 + sb * 0.81)
    lo, hi = C.drive_sq_bounds(stored, eo, 1.0, 1.0)            # no reduction, no loss: the pitch joints unchanged
    assert np.allclose(lo[eo == 1.0], stored[eo == 1.0]) and np.allclose(hi[eo == 1.0], stored[eo == 1.0])


def test_reduced_profile_uses_the_split_only_when_it_reproduces():
    demo = C.stored_demo("fore128-s3")
    joints = demo["segments"]["stand"]["joints"]
    st = C.stored_segments(demo)
    eo = np.array([C.ETA_OLD[C.axis_of(j)] for j in joints])
    split = {seg: (st[seg] * eo ** 2 * 0.7, st[seg] / eo ** 2 * 0.3) for seg in st}   # reproduces exactly
    p = C.reduced_profile(st, joints, "both", 1.25, 0.95, split)
    assert p["method"] == "exact"
    i = joints.index("left_hind_pitch")
    assert p["walk_0.25"]["sq"][i] == pytest.approx(C.drive_sq_from_split(split["walk_0.25"][0][i],
                                                                          split["walk_0.25"][1][i], 1.25, 0.95))
    assert p["walk_0.25"]["sq"][joints.index("left_hind_roll")] == st["walk_0.25"][joints.index("left_hind_roll")]
    bad = {seg: (a * 1.01, b) for seg, (a, b) in split.items()}
    assert C.reduced_profile(st, joints, "both", 1.25, 0.95, bad)["method"] == "bound"
    rod1 = C.reduced_profile(st, joints, "knees", 1.0, 0.90, None)        # the rod at the knees at r = 1 is today
    assert np.allclose(rod1["walk_0.25"]["sq"], st["walk_0.25"]) and np.allclose(rod1["stand"]["sq"], st["stand"])
    hip = C.reduced_profile(st, joints, "pitch", 1.0, 0.90, None)         # at the hip it adds the linkage's loss
    i = joints.index("left_hind_pitch")
    assert hip["method"] == "bound" and hip["walk_0.25"]["sq"][i] == pytest.approx(st["walk_0.25"][i] / 0.81)
    assert hip["walk_0.25"]["sq_low"][i] == pytest.approx(st["walk_0.25"][i] * 0.81)


def test_e2a_grid_control_row_is_today(monkeypatch):
    monkeypatch.setattr(C, "REDUCTIONS", (1.0,))
    monkeypatch.setattr(C, "PLACEMENTS", {"knees": ("flexion",)})
    monkeypatch.setattr(C, "TRANSMISSIONS", {"rod": C.TRANSMISSIONS["rod"]})
    demo = C.stored_demo("fore128-s3")
    rows = C.e2a_grid(demo, C.stored_segments(demo))
    by = {r["transmission"]: r for r in rows}
    assert set(by) == {"rod", "rod_eta0.9"}                            # the judged rod and the supporting 0.90 rows
    assert by["rod"]["judged"] and by["rod"]["eta"] == 0.95            # the rule's 0.95: no dated source for a rod
    ctl = by["rod_eta0.9"]
    assert not ctl["judged"] and ctl["eta"] == 0.90 and ctl["method"] == "bound"   # no split given: the bound path
    assert ctl["C1_c"] == pytest.approx(98.3, abs=0.05) and ctl["C2"] == pytest.approx(0.914, abs=5e-4)
    assert ctl["C2_low"] == pytest.approx(ctl["C2"])                   # the rod at 0.90, r = 1: both factors are 1
    rod = by["rod"]                                                    # at 0.95 the knees' bounds straddle today's
    assert rod["C2_low"] <= ctl["C2"] + 1e-12 and ctl["C2"] <= rod["C2"] + 1e-12


def test_judged_and_supporting_efficiencies():
    assert C.TRANSMISSIONS["rod"]["eta"] == C.SOURCES["eta_default"]["value"] == 0.95
    assert C.SUPPORTING_ETA == {"rod": 0.90} and C.ETA_OLD["flexion"] == 0.90          # today's linkage unchanged
    specs = C.e2a_specs()
    assert [s[0] for s in specs if s[3]] == ["rod", "belt", "cable"] and [s[0] for s in specs if not s[3]] == ["rod_eta0.9"]
    assert C.variant_r_eta("rod")[1]["left_hind_flexion"] == 0.95 and C.variant_r_eta("R")[1]["left_fore_pitch"] == 0.95


def test_relog_drive_sq_identity_and_supporting_rod():
    rng = np.random.default_rng(4)
    acts = list(C.LEG_ACTUATORS)
    r_, e_ = C.variant_r_eta("rod")
    r = np.array([r_.get(a, 1.0) for a in acts])
    eta = np.array([e_.get(a, C.ETA_OLD[C.axis_of(a)]) for a in acts])
    sm, sb = rng.random(12) * 20, rng.random(12) * 8
    res = {"actuators": acts, "r": r, "eta": eta, "sq_motor": sm, "sq_brake": sb,
           "drive_sq": sm / (r * eta) ** 2 + sb * (eta / r) ** 2}
    assert np.allclose(C.relog_drive_sq(res, {}), res["drive_sq"], rtol=1e-12)
    alt = C.relog_drive_sq(res, {a: 0.90 for a in r_})
    k = acts.index("left_hind_flexion")
    assert alt[k] == pytest.approx(sm[k] / (1.25 * 0.90) ** 2 + sb[k] * (0.90 / 1.25) ** 2)
    k = acts.index("left_hind_roll")
    assert alt[k] == res["drive_sq"][k]


def test_heat_rules_reproduce_the_studys_policy_s3():
    demo = C.stored_demo("fore128-s3")
    st = C.stored_segments(demo)
    h = C.heat_rules(demo, st["walk_0.25"], st["stand"])
    assert h["C1_c"] == pytest.approx(98.3, abs=0.05) and h["C1_joint"] == "left_hind_pitch"
    assert h["C2"] == pytest.approx(0.914, abs=0.0005) and h["C2_joint"] == "left_hind_pitch"
    assert not h["C1_pass"] and not h["C2_pass"]
    f = C.forced_air_c1(demo, 1.0, heat_path="winding_to_ambient")      # ratio 1: the passive figure
    assert f["C1_c"] == pytest.approx(h["C1_c"], abs=1e-9)
    for bad in ("housing_to_air", "heat_sink", ""):    # the one-node model: only a winding-to-ambient or total pair
        with pytest.raises(ValueError):
            C.forced_air_c1(demo, 0.5, heat_path=bad)
    with pytest.raises(TypeError):                     # the heat path must be stated
        C.forced_air_c1(demo, 0.5)


def test_policy_heat_reproduces_the_studys_table():
    s3 = C.policy_heat(C.stored_demo("fore128-s3"))
    assert s3["deciding"]["joint"] == "left_hind_pitch" and s3["deciding"]["mirror"] == "right_hind_pitch"
    assert s3["deciding"]["ratio"] == pytest.approx(0.914, abs=5e-4)
    assert s3["deciding"]["mirror_ratio"] == pytest.approx(0.509, abs=5e-4)
    assert s3["even_split_C2"] == pytest.approx(0.740, abs=5e-4) and s3["even_split_C1_c"] == pytest.approx(73.4, abs=0.05)
    assert not s3["MEETS"]
    s1 = C.policy_heat(C.stored_demo("fore128-s1"))
    assert s1["C1_c"] == pytest.approx(74.4, abs=0.05) and s1["C2"] == pytest.approx(0.746, abs=5e-4)
    assert s1["MEETS"]


def test_speed_percentile_from_histograms():
    rng = np.random.default_rng(2)
    walk, stand = rng.gamma(4, 8, size=(4000, 12)), rng.gamma(2, 2, size=(2000, 12))
    b = 0.1
    nb = int(round(300 / b)) + 1

    def hist(x):
        h = np.zeros((12, nb), dtype=np.int64)
        k = np.minimum((x / b).astype(np.int64), nb - 1)
        for j in range(12):
            h[j] = np.bincount(k[:, j], minlength=nb)
        return h
    p = C.weighted_percentile_from_hist([hist(walk), hist(stand)], [0.75, 0.25], b)
    pooled = np.vstack([walk] * 3 + [stand] * 2)        # 12000 walk : 4000 stand samples = 0.75 : 0.25
    ref = np.percentile(pooled, 99, axis=0)
    assert np.all(np.abs(p - ref) <= 2 * b), (p - ref)
    sc = C.speed_check([hist(walk)], [1.0], [walk.max(0)], b, np.full(12, 1.25))
    assert np.allclose(sc["peak_rpm"], walk.max(0) * 1.25)
    assert sc["pass"] == bool(np.all(np.array(sc["p99_rpm"]) <= 83) and np.all(np.array(sc["peak_rpm"]) <= 91.3))


def test_e2a_speed_check_tests_only_the_reduced_drives():
    acts = list(C.LEG_ACTUATORS)
    b, nb = 0.1, int(round(300 / 0.1)) + 1
    x = np.full((1000, 12), 40.0)
    x[:, acts.index("left_fore_roll")] = 120.0                 # an unreduced roll drive far over the cap
    h = np.zeros((12, nb), dtype=np.int64)
    k = np.minimum((x / b).astype(np.int64), nb - 1)
    for j in range(12):
        h[j] = np.bincount(k[:, j], minlength=nb)
    for pl in ("pitch", "knees", "both"):
        sc = C.e2a_speed_check([h], [1.0], [x.max(0)], b, acts, pl, 1.3)
        assert sc["drives"] == list(C.reduced_joints(pl)) and len(sc["p99_rpm"]) == len(sc["drives"])
        assert sc["pass"] and max(sc["peak_rpm"]) == pytest.approx(52.0)
    assert not C.speed_check([h], [1.0], [x.max(0)], b, np.ones(12))["pass"]   # all 12 would include the roll drive
    x2 = x.copy()
    x2[:, acts.index("right_hind_flexion")] = 70.0                             # 70 x 1.3 = 91 rpm: over 83 at p99
    k = np.minimum((x2 / b).astype(np.int64), nb - 1)
    h2 = np.array([np.bincount(k[:, j], minlength=nb) for j in range(12)])
    assert not C.e2a_speed_check([h2], [1.0], [x2.max(0)], b, acts, "knees", 1.3)["pass"]
    assert C.e2a_speed_check([h2], [1.0], [x2.max(0)], b, acts, "pitch", 1.3)["pass"]


# ------------------------------------------------------------------------------------------------- E2(b) variants
def test_transmission_stiffness_by_hand():
    L = C.SPANS_M["hind_flexion"]
    assert L == pytest.approx(0.1880, abs=1e-4) and C.SPANS_M["fore_flexion"] == pytest.approx(0.1280, abs=1e-4)
    rod = C.transmission_stiffness("rod", "left_hind_flexion")["k_nm_per_rad"]
    assert rod == pytest.approx(200e9 * 36.6e-6 / L * 0.035 ** 2 * math.sin(math.radians(43)) ** 2)
    belt = C.transmission_stiffness("belt", "left_hind_flexion")
    assert belt["R_joint_m"] == pytest.approx(70 * 0.003 / (2 * math.pi))
    assert belt["k_nm_per_rad"] == pytest.approx(2 * (30000 * 4.4482216152605 / 25.4 * 15) / L * belt["R_joint_m"] ** 2)
    cab = C.transmission_stiffness("cable", "left_fore_pitch")["k_nm_per_rad"]
    assert cab == pytest.approx(2 * 97.1e9 * 4.6e-6 / 0.100 * 0.035 ** 2)
    with pytest.raises(ValueError):
        C.transmission_stiffness("rod", "left_fore_roll")


def test_belt_one_span_is_half_and_its_pretension():
    for j in ("left_fore_pitch", "left_fore_flexion", "left_hind_flexion"):
        two = C.transmission_stiffness("belt", j)
        one = C.transmission_stiffness("belt", j, strands=1)
        assert one["k_nm_per_rad"] == pytest.approx(two["k_nm_per_rad"] / 2) and one["strands"] == 1
    with pytest.raises(ValueError):
        C.transmission_stiffness("rod", "left_hind_flexion", strands=2)
    p = C.belt_pretension("left_hind_flexion")                  # 18.9 / 0.90 x 0.98 x 1.25 at 33.42 mm
    assert p["limit_joint_side_nm"] == pytest.approx(18.9 / 0.90 * 0.98 * 1.25, rel=1e-6)
    assert p["pretension_per_span_needed_n"] == pytest.approx(p["limit_joint_side_nm"] / (70 * 0.003 / (2 * math.pi)) / 2)
    assert C.belt_pretension("left_fore_flexion")["drive"] == "X6-60"


@pytest.fixture(scope="module")
def variants(out):
    return {tr: C.write_variant(out / "variants", transmission=tr) for tr in (None, "rod", "belt", "cable")}


@pytest.fixture(scope="module")
def b025(out):
    return C.e2b_model("B025", out / "variants")


def _model_fields(m):
    out = {}
    for obj, pre in ((m, ""), (m.opt, "opt."), (m.stat, "stat."), (m.vis.global_, "vis.global.")):
        for k in dir(obj):
            if k.startswith("_"):
                continue
            try:
                v = getattr(obj, k)
            except Exception:  # noqa: BLE001
                continue
            if isinstance(v, (np.ndarray, int, float)) and not isinstance(v, bool):
                out[pre + k] = np.array(v, copy=True)
    return out


def test_b025_differs_from_the_adopted_model_only_in_its_timestep(b025):
    path, rec = b025
    src, txt = C.MODEL.read_text(), Path(path).read_text()
    assert txt == src.replace('timestep=".001"', 'timestep="0.00025"', 1) and txt != src     # one attribute, text
    a, b = mujoco.MjModel.from_xml_path(str(C.MODEL)), mujoco.MjModel.from_xml_path(str(path))
    fa, fb = _model_fields(a), _model_fields(b)
    assert set(fa) == set(fb) and len(fa) > 200
    differ = [k for k in fa if not np.array_equal(fa[k], fb[k], equal_nan=fa[k].dtype.kind == "f")]
    assert differ == ["opt.timestep"], differ
    assert b.opt.timestep == pytest.approx(C.SERIES_DT)
    for jn in C.reduced_joints("both"):                       # what the first route got wrong: limits and armature kept
        assert b.actuator(jn).forcerange[1] == a.actuator(jn).forcerange[1]
        assert b.joint(jn).armature[0] == a.joint(jn).armature[0]
    assert rec["builder"] == "write_timestep_copy" and rec["source_sha256"] == MODEL_SHA
    b0125, _ = C.e2b_model("B0125", Path(path).parent)
    assert mujoco.MjModel.from_xml_path(str(b0125)).opt.timestep == pytest.approx(C.FALLBACK_DT)
    p1, r1 = C.e2b_model("B1", Path(path).parent)
    assert p1 == C.MODEL and r1["xml_sha256"] == MODEL_SHA


def test_write_variant_refuses_the_rigid_r1_baseline(out):
    with pytest.raises(ValueError):
        C.write_variant(out / "variants", transmission=None, r=1.0, physics_dt=C.SERIES_DT)
    with pytest.raises(ValueError):
        C.write_variant(out / "variants", transmission=None, strands=1)


def test_belt_one_span_variant(out):
    path, rec = C.e2b_model("belt_1span", out / "variants")
    m = mujoco.MjModel.from_xml_path(str(path))
    for jn in C.reduced_joints("both"):
        k = C.transmission_stiffness("belt", jn, strands=1)["k_nm_per_rad"]
        assert m.tendon_stiffness[m.tendon(f"cand_spring_{jn}").id] == pytest.approx(k, rel=1e-8)
    assert rec["strands"] == 1 and rec["variant"] == "belt_1span" and m.opt.timestep == pytest.approx(C.SERIES_DT)
    p2, _ = C.e2b_model("rod", out / "variants", physics_dt=C.FALLBACK_DT)      # item 4's rerun of a variant
    assert mujoco.MjModel.from_xml_path(str(p2)).opt.timestep == pytest.approx(C.FALLBACK_DT)


def test_variants_compile_and_move_the_drive(variants):
    src = mujoco.MjModel.from_xml_path(str(C.MODEL))
    for tr, (path, rec) in variants.items():
        m = mujoco.MjModel.from_xml_path(str(path))
        assert m.opt.timestep == pytest.approx(C.SERIES_DT)
        for jn in C.reduced_joints("both"):
            a = m.actuator(jn)
            j0 = src.joint(jn)
            lim0 = src.actuator(jn).forcerange[1]
            eta = C.TRANSMISSIONS[tr]["eta"] if tr else 0.95
            assert a.forcerange[1] == pytest.approx(lim0 / C.ETA_OLD[C.axis_of(jn)] * eta * 1.25, abs=1e-5)
            assert m.joint(int(a.trnid[0])).name == jn
            if tr is None:
                assert m.joint(jn).armature[0] == pytest.approx(j0.armature[0] * 1.25 ** 2)
                assert m.nq == src.nq
            else:
                out = m.joint(jn + C.OUT_SUFFIX)
                assert out.armature[0] == 0.0 and np.allclose(out.range, j0.range)
                assert m.joint(jn).armature[0] == pytest.approx(j0.armature[0] * 1.25 ** 2)
                t = m.tendon(f"cand_spring_{jn}").id
                assert m.tendon_stiffness[t] == pytest.approx(C.transmission_stiffness(tr, jn)["k_nm_per_rad"],
                                                              rel=1e-8)
                assert np.all(m.tendon_lengthspring[t] == 0.0) and np.all(m.tendon_stiffnesspoly[t] == 0.0)
                assert m.nq == src.nq + 8
        assert abs(m.body_mass.sum() - src.body_mass.sum()) < 0.001
        assert rec["source_sha256"] == MODEL_SHA                          # the adopted model is read, never written
    assert C.sha256_file(C.MODEL) == MODEL_SHA


MODEL_SHA = C.sha256_file(C.MODEL)


def test_series_spring_force(variants):
    path, rec = variants["belt"]
    m = mujoco.MjModel.from_xml_path(str(path))
    d = mujoco.MjData(m)
    jn = "left_hind_flexion"
    rot, outj = m.joint(jn), m.joint(jn + C.OUT_SUFFIX)
    d.qpos[rot.qposadr[0]] += 0.01
    mujoco.mj_forward(m, d)
    k = rec["joints"][jn]["stiffness"]["k_nm_per_rad"]
    assert d.qfrc_passive[rot.dofadr[0]] == pytest.approx(-k * 0.01, rel=1e-9)
    assert d.qfrc_passive[outj.dofadr[0]] == pytest.approx(k * 0.01, rel=1e-6, abs=1e-6)


def test_series_env_syncs_at_reset_and_stands(variants):
    ec = _ec()
    base = C.make_env(C.policy_env_config(ec))
    path, _ = variants["rod"]
    env = C.make_env(C.policy_env_config(ec, path, C.SERIES_DT))
    assert len(env.sync_pairs) == 8 and env.observation_space.shape == base.observation_space.shape
    cap = C.SpeedCap(env, {j: 1.25 for j in C.reduced_joints("both")})
    env.extra_hooks.append(cap)
    env.reset_hooks.append(cap)
    env.reset(seed=11, options={"command_speed": 0.0, "delay_steps": C.delay_steps_for(C.SERIES_DT)})
    d = env.data
    assert all(d.qpos[a] == d.qpos[b] for a, b in env.sync_pairs)
    for _ in range(10):
        _, _, term, _, info = env.step(np.zeros(12))
        assert not term, info["termination"]
    assert np.isfinite(d.qpos).all()


def test_speed_cap_envelope(variants):
    path, _ = variants[None]
    env = C.make_env(C.policy_env_config(_ec(), path, C.SERIES_DT))
    cap = C.SpeedCap(env, {"left_hind_pitch": 1.25})
    cap.on_reset(env)
    base = cap.base.copy()
    w = 83.0 * 2 * math.pi / 60 / 1.25                       # joint speed at the drive's cap
    i, dof = cap.idx[0], cap.dof[0]
    for v, f in ((0.9 * w, 1.0), (0.975 * w, 0.5), (w, 0.0), (1.2 * w, 0.0)):
        env.data.qvel[dof] = v
        cap(env)
        assert env.model.actuator_forcerange[i, 1] == pytest.approx(base[0, 1] * f, abs=1e-9)
        assert env.model.actuator_forcerange[i, 0] == pytest.approx(base[0, 0])
        env.data.qvel[dof] = -v
        cap(env)
        assert env.model.actuator_forcerange[i, 0] == pytest.approx(base[0, 0] * f, abs=1e-9)
        assert env.model.actuator_forcerange[i, 1] == pytest.approx(base[0, 1])
    env.reset(seed=0)                                          # the env restores its nominal limits
    cap.on_reset(env)
    assert np.allclose(cap.base, base)


def test_e2b_compare_rules():
    acts = list(C.LEG_ACTUATORS)
    sq = np.full(12, 4.0)
    base = {s: {"fell": False, "speed": 0.24, "drive_sq": sq, "samples": 18000, "actuators": acts} for s in range(8)}
    same = {s: dict(v) for s, v in base.items()}
    assert C.e2b_compare(same, base)["pass"]
    slow = {s: {**v, "speed": 0.215} for s, v in base.items()}
    assert not C.e2b_compare(slow, base)["pass"]
    hot = {s: {**v, "drive_sq": sq * (1 + 0.1)} for s, v in base.items()}          # 0.267 -> 0.280: +0.013
    assert C.e2b_compare(hot, base)["pass"]
    hotter = {s: {**v, "drive_sq": sq * 1.6} for s, v in base.items()}
    assert not C.e2b_compare(hotter, base)["pass"]
    fell = {s: dict(v) for s, v in base.items()}
    fell[3]["fell"] = True
    assert C.e2b_compare(fell, base)["new_falls"] == [3]
    both = {s: dict(v) for s, v in base.items()}
    both[3]["fell"] = True
    assert C.e2b_compare(fell, both)["pass"]
    v = C.e2b_verdict({p: {"B1": {"c": {"pass": p != "d"}}, "B025": {"c": {"pass": True}}} for p in "abcd"})
    assert v["COMPATIBLE"] and not v["per_policy"]["d"]


def test_e2b_unstable_episodes_are_not_falls():
    acts = list(C.LEG_ACTUATORS)
    base = {s: {"fell": False, "termination": "", "speed": 0.24, "drive_sq": np.full(12, 4.0), "samples": 72000,
                "actuators": acts} for s in range(8)}
    uns = {s: dict(v) for s, v in base.items()}
    uns[2].update(fell=True, termination=C.UNSTABLE)       # sweep.run_episode marks every termination a fall
    r = C.e2b_compare(uns, base)
    assert r["new_falls"] == [] and r["unstable"] == [2] and r["unstable_variant"] == [2] and 2 not in r["paired_up"]
    assert r["harness_fault"] and r["pass"] is None and r["pass_on_stable_seeds"]
    r = C.e2b_compare(base, uns)                           # an unstable baseline episode too
    assert r["unstable_baseline"] == [2] and r["pass"] is None
    tilt = {s: dict(v) for s, v in base.items()}
    tilt[2].update(fell=True, termination="tilt")          # a real fall is still a new fall
    assert C.e2b_compare(tilt, base)["new_falls"] == [2] and C.e2b_compare(tilt, base)["pass"] is False
    T, F, N = {"c": {"pass": True}}, {"c": {"pass": False}}, {"c": {"pass": None}}
    v = C.e2b_verdict({"a": {"B1": T}, "b": {"B1": T}, "c": {"B1": T}, "d": {"B1": N}})
    assert v["COMPATIBLE"] is True and v["not_judged"] == ["d"]
    v = C.e2b_verdict({"a": {"B1": T}, "b": {"B1": T}, "c": {"B1": N}, "d": {"B1": F}})
    assert v["COMPATIBLE"] is None and v["per_policy"] == {"a": True, "b": True, "c": None, "d": False}
    v = C.e2b_verdict({"a": {"B1": T}, "b": {"B1": N}, "c": {"B1": F}, "d": {"B1": F}})
    assert v["COMPATIBLE"] is False
    v = C.e2b_verdict({"a": {"B1": T, "B025": F}, "b": {"B1": T, "B025": N}, "c": {"B1": T}, "d": {"B1": T}})
    assert v["per_policy"]["a"] is False and v["per_policy"]["b"] is None and v["COMPATIBLE"] is None


def test_e2b_judged_runs_item_4():
    j = C.e2b_judged_runs(set())
    assert set(j) - {"_B0125_needed"} == {"R", "rod", "belt", "cable", "belt_1span"} and not j["_B0125_needed"]
    assert all(v["baselines"] == ("B1", "B025") and v["physics_dt"] == C.SERIES_DT for k, v in j.items() if k[0] != "_")
    j = C.e2b_judged_runs({"rod"})
    assert j["rod"]["baselines"] == ("B1", "B0125") and j["rod"]["physics_dt"] == C.FALLBACK_DT and j["_B0125_needed"]
    assert j["belt"]["baselines"] == ("B1", "B025")
    j = C.e2b_judged_runs({"B025"})
    assert all(v["rerun_at_fallback"] for k, v in j.items() if k[0] != "_")
    with pytest.raises(ValueError):
        C.e2b_judged_runs({"B0125"})
    assert C.E2B_VARIANTS["B025"]["builder"] == "write_timestep_copy" and C.E2B_VARIANTS["B1"]["builder"] == "adopted"


# ------------------------------------------------------------------------------------------------------ E5 the trot
def test_pitched_trot_at_zero_is_the_frozen_controller():
    walk60 = C._walk60()
    envs = [C.make_env(walk60.plant_config(model_path=str(C.MODEL))) for _ in range(2)]
    a = walk60.TrotController(envs[0], walk60.WalkParams())
    b = C.PitchedTrot(envs[1], walk60.WalkParams(), 0.0)
    oa, _ = envs[0].reset(seed=101, options={"command_speed": 0.25, "delay_steps": 10})
    ob, _ = envs[1].reset(seed=101, options={"command_speed": 0.25, "delay_steps": 10})
    for _ in range(80):                                          # 0.8 s: through the stand into the first steps
        xa, xb = a.act(oa), b.act(ob)
        assert np.array_equal(xa, xb)
        oa = envs[0].step(xa)[0]
        ob = envs[1].step(xb)[0]


def test_pitched_trot_setpoint_and_feet():
    walk60 = C._walk60()
    env = C.make_env(walk60.plant_config(model_path=str(C.MODEL)))
    pt = C.PitchedTrot(env, walk60.WalkParams(), 2.0)
    for p in np.radians([-3.0, 0.0, 1.0, 4.0]):
        g = np.array([math.sin(p), 0.0, -math.cos(p)])            # nose down by p (walk60.level_rotation's sign)
        assert walk60.level_rotation(g)[2] == pytest.approx(p)
        assert walk60.level_rotation(pt.rotate_gravity(g))[2] == pytest.approx(p - math.radians(2.0))
    plain = walk60.TrotController(env, walk60.WalkParams())
    tgt = np.array([0.26, 0.14, -0.27])
    q1, _ = pt.ctrl.kin.ik("left_fore", tgt, plain.kin.q_home["left_fore"])
    q2, _ = plain.kin.ik("left_fore", tgt + [0, 0, 0.26 * math.tan(math.radians(2.0))], plain.kin.q_home["left_fore"])
    assert np.allclose(q1, q2)


def test_e5_settings_are_fixed_and_in_the_box():
    for size in (0.60, 0.50):
        rows = C.e5_settings(size)
        assert len(rows) == 33 and rows == C.e5_settings(size)
        box, _ = C.e5_box(size)
        d = C.e5_default(size)
        assert all(abs(rows[0][k] - round(d[k], C.E5_ROUND[k])) < 1e-12 for k in C.E5_PARAMS)
        for r in rows[1:]:
            assert all(box[k][0] - 1e-9 <= r[k] <= box[k][1] + 1e-9 for k in C.E5_PARAMS)
        nb = C.e5_neighbours(rows[0], size, 0)
        assert len(nb) <= 10 and all(box[k][0] - 1e-9 <= r[k] <= box[k][1] + 1e-9 for r in nb for k in C.E5_PARAMS)
    assert C.e5_default(0.60)["period_s"] == 0.70
    assert C.e5_default(0.50)["period_s"] == pytest.approx(0.70 * math.sqrt(0.5 / 0.6))
    P = C.walk_params(C.e5_settings(0.50)[5], 0.50)
    assert P.leg_drop_m == pytest.approx(0.278 * 0.5 / 0.6) and P.period_s == C.e5_settings(0.50)[5]["period_s"]


def _ep(fell=False, v=0.24, s1020=0.24, hk=0.8, hk1020=0.8):
    return {"fell": fell, "forward_speed_mps": v, "speed_10_20_mps": s1020, "hind_knee_rms_over_rating_2_20": hk,
            "hind_knee_rms_over_rating_10_20": hk1020}


def test_e5_success_rule():
    conds = ["J1", "J2", "J3", "J4"] + list(C.E5_PUSHES)
    good = {c: [_ep() for _ in range(10)] for c in conds}
    assert C.e5_success(good)["SUCCEEDS"]
    one_slow = {**good, "push_back_0.5": [_ep(s1020=0.20, hk1020=1.05)] + [_ep()] * 9}
    assert C.e5_success(one_slow)["SUCCEEDS"] is False                 # its hind knee is over 1.0
    slow_cool = {**good, "push_back_0.5": [_ep(s1020=0.20, hk1020=0.95)] * 2 + [_ep()] * 8}
    r = C.e5_success(slow_cool)                                        # slow but not hot: not the slow gait
    assert r["slow_gait"]["push_back_0.5"] == 0 and r["SUCCEEDS"]
    fall_push = {**good, "push_right_0.3": [_ep(fell=True)] * 2 + [_ep()] * 8}
    assert C.e5_success(fall_push)["slow_gait"]["push_right_0.3"] == 2
    assert not C.e5_success({**good, "J3": [_ep(v=0.22)] * 10})["SUCCEEDS"]
    assert not C.e5_success({**good, "J2": [_ep(fell=True)] + [_ep()] * 9})["SUCCEEDS"]
    res = {"a": C.e5_success(good), "b": C.e5_success({**good, "J1": [_ep(hk=0.7, hk1020=0.7)] * 10})}
    assert C.e5_choose(res) == "a"                                     # the worst hind knee decides: 0.8 both
    res["c"] = C.e5_success({c: [_ep(hk=0.7, hk1020=0.7)] * 10 for c in conds})
    assert C.e5_choose(res) == "c"


# ------------------------------------------------------------------------------------------------- policies, plan
@needs_policies
def test_checkpoints_verify_and_a_changed_byte_is_refused(out):
    v = C.verify_checkpoint("fore128-s0")
    assert v["match"]
    d = out / "tampered"
    if d.exists():
        shutil.rmtree(d)
    shutil.copytree(C.POLICY_DIRS["fore128-s0"], d)
    pk = next(d.glob("*vecnormalize*.pkl"))
    b = bytearray(pk.read_bytes())
    b[-1] ^= 1
    pk.write_bytes(bytes(b))
    with pytest.raises(ValueError):
        C.verify_checkpoint("fore128-s0", d)
    shutil.rmtree(d)


@needs_policies
def test_policy_loads_through_the_sweeps_loaders():
    ec = C.run_env_config("fore128-s0")
    env = C.make_env(C.policy_env_config(ec))
    act = C.policy_act(*C.load_policy("fore128-s0", env))
    obs, _ = env.reset(seed=700000, options={"command_speed": 0.25, "delay_steps": 10})
    a = act(obs)
    assert a.shape == (12,) and np.isfinite(a).all()


def test_plan_is_complete_and_serialisable():
    p = C.plan(verify=False)
    text = json.dumps(p, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x))
    back = json.loads(text)
    assert len(back["E1"]["stops"]) == 17 and back["E5"]["max_settings"] == 60
    assert len(back["E5"]["sizes"]["0.60"]["stage1"]) == 33 and len(back["E5"]["sizes"]["0.50"]["stage1"]) == 33
    assert set(back["E2"]["stiffness_joint_side"]) == {"rod", "belt", "cable", "belt_1span"}
    assert back["E1"]["seed_blocks"]["A"][0] == 700000
    assert back["E2"]["variants"]["B025"]["builder"] == "write_timestep_copy"
    assert back["E2"]["delay_steps"] == {"0.001": 10, "0.00025": 40, "0.000125": 80}
    assert back["E2"]["transmissions"]["rod"]["eta"] == 0.95 and back["E4"]["heat_paths_accepted"]
    assert back["provenance"]["sha256"]["sim/candidate_checks.py"] == C.sha256_file(SIM / "candidate_checks.py")


def test_provenance_records_the_code_and_the_runner():
    runner = C.CAND_SCRATCH / "pytest" / "runner_probe.py"
    runner.parent.mkdir(parents=True, exist_ok=True)
    runner.write_text("# a runner script\n")
    pv = C.provenance(runner=runner, extra={"variant:x": C.MODEL})
    assert pv["sha256"]["sim/candidate_checks.py"] == C.sha256_file(SIM / "candidate_checks.py")
    assert pv["sha256"]["runner:" + str(runner.resolve())] == C.sha256_file(runner)
    assert pv["sha256"]["variant:x"] == MODEL_SHA and all(pv["sha256"][t] for t in C.BUILD_TOOLS)
    assert pv["commit"] is None or len(pv["commit"]) == 40
    runner.unlink()
