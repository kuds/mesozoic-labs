"""Tests for the scaled-model builder and its comparison (sim/build_scaled_trike.py). Simulation only. Run by hand:
    python -m pytest hardware/triceratops_build/sim/tests/test_build_scaled_trike.py
"""

import dataclasses
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

SIM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIM))

import build_scaled_trike as B  # noqa: E402

FF = B.FF


@pytest.fixture(scope="module")
def models(tmp_path_factory):
    out = tmp_path_factory.mktemp("scaled")
    got = {}
    for S, dk, rule in ((0.60, "D0", "similar"), (0.50, "D0", "similar"), (0.80, "D0", "similar"),
                        (0.70, "D0", "similar"), (0.80, "D8", "similar"), (0.50, "D0", "literal")):
        path, info = B.write_model(S, out, dk, rule)
        got[(S, dk, rule)] = (path, info, mujoco.MjModel.from_xml_path(str(path)))
    return got


@pytest.fixture(scope="module")
def src():
    return mujoco.MjModel.from_xml_path(str(B.SRC))


ARRAYS = ("body_mass", "body_inertia", "body_pos", "body_ipos", "body_quat", "geom_size", "geom_pos", "geom_quat",
          "geom_friction", "geom_solref", "geom_solimp", "mesh_vert", "actuator_gainprm", "actuator_biasprm",
          "actuator_forcerange", "actuator_ctrlrange", "dof_armature", "dof_damping", "jnt_stiffness", "jnt_range",
          "qpos0", "qpos_spring", "site_pos", "site_size")


def test_060_copy_is_the_source_exactly(models, src):
    """At s = 1 with the current drives the builder changes no number: every compiled array is the source's."""
    before = hashlib.sha256(B.SRC.read_bytes()).hexdigest()
    _, info, m = models[(0.60, "D0", "similar")]
    for a in ARRAYS:
        assert np.array_equal(getattr(m, a), getattr(src, a)), a
    assert info["k_kp"] == 1.0 and info["k_kv"] == 1.0 and info["k_ankle_stiffness"] == 1.0
    assert hashlib.sha256(B.SRC.read_bytes()).hexdigest() == before


def test_060_copy_reproduces_the_source_standing_torques(models):
    """The control: the 0.60 m copy and the source XML stand with identical torques and foot loads (same harness,
    same seed)."""
    path, info, _ = models[(0.60, "D0", "similar")]
    a = B.stand_episode(str(B.SRC), None, 101, seconds=2.0)
    b = B.stand_episode(str(path), info, 101, seconds=2.0)
    assert a["tau_mean_5_10s_nm"] == b["tau_mean_5_10s_nm"]
    assert a["load_share_mean_5_10s"] == b["load_share_mean_5_10s"]
    assert not a["fell"] and not b["fell"]


@pytest.mark.parametrize("S,dk", [(0.50, "D0"), (0.55, "D0"), (0.70, "D0"), (0.80, "D0"), (0.80, "D8"), (0.60, "D2")])
def test_part_masses_are_the_study_mass_model(S, dk):
    """Row by row, the builder's masses sum to form_factor_study.mass_model(layout=None, bms=False), class by class."""
    parts = B.part_masses(S, dk)
    mm = FF.mass_model(S, FF.DRIVE_SETS[dk], layout=None, bms=False)
    assert sum(r["kg"] for r in parts.values()) == pytest.approx(mm["total_kg"], abs=1e-12)
    by = B._by_class(parts)
    assert set(by) == set(mm["by_class_kg"])
    for k, v in mm["by_class_kg"].items():
        assert by[k] == pytest.approx(v, abs=1e-12), k


def test_compiled_mass_and_k_match_the_study_predictions(models):
    """Mass and k of the 0.50 / 0.70 / 0.80 m copies are the study's simulate_predictions (its JSON, 5 decimals); the
    compiled mass is the mass model's within the XML's extra digits (the ledger rows carry 5 decimals, the XML 6: the
    source compiles to 12.750593 kg against the rows' 12.7504; 5e-5 relative)."""
    pred = json.loads((B.BUILD / "design" / "form_factor_out.json").read_text())["simulate_predictions"]
    for S in (0.50, 0.70, 0.80):
        _, info, m = models[(S, "D0", "similar")]
        assert info["mass_model_kg"] == pytest.approx(pred[f"{S:.2f}"]["mass_kg"], abs=1e-5)
        assert info["k"] == pytest.approx(pred[f"{S:.2f}"]["k"], abs=1e-5)
        assert m.body_mass.sum() == pytest.approx(info["mass_model_kg"], rel=5e-5)


@pytest.mark.parametrize("S", [0.50, 0.80])
def test_geometry_scales_by_s(models, src, S):
    """Positions, sizes and mesh vertices x s; angles, ranges and contact settings unchanged; for a body whose geoms
    all changed mass by one ratio r, its inertia is the source's x r x s^2."""
    path, info, m = models[(S, "D0", "similar")]
    s = S / 0.60
    np.testing.assert_allclose(m.body_pos, src.body_pos * s, rtol=1e-12, atol=1e-15)
    plane = src.geom_type == mujoco.mjtGeom.mjGEOM_PLANE
    # a mesh geom's compiled size is MuJoCo's equivalent-inertia box of the processed mesh, which moves by up to
    # 5e-4 relative under scaling (mesh processing tolerances); its vertices are checked exactly below instead
    mesh = src.geom_type == mujoco.mjtGeom.mjGEOM_MESH
    keep = ~plane & ~mesh
    np.testing.assert_allclose(m.geom_size[keep], src.geom_size[keep] * s, rtol=1e-9, atol=1e-15)
    np.testing.assert_allclose(m.geom_pos, src.geom_pos * s, rtol=1e-9, atol=1e-12)
    # mesh vertices: the XML's scale attribute is the source's x s with the vertex text untouched; compiled, MuJoCo
    # re-centres each mesh on its own centre of mass and principal axes, which moves vertices by up to ~2e-5 m
    import xml.etree.ElementTree as ET
    a = {e.get("name"): e for e in ET.parse(B.SRC).getroot().iter("mesh")}
    b = {e.get("name"): e for e in ET.parse(path).getroot().iter("mesh")}
    for n, e in a.items():
        assert b[n].get("vertex") == e.get("vertex")
        np.testing.assert_allclose([float(x) for x in b[n].get("scale").split()],
                                   [float(x) * s for x in e.get("scale").split()], rtol=1e-15)
    np.testing.assert_allclose(m.mesh_vert, src.mesh_vert * s, atol=5e-5)
    np.testing.assert_allclose(m.qpos0[:3], src.qpos0[:3] * s, rtol=1e-12)
    for a in ("jnt_range", "actuator_ctrlrange", "geom_solref", "geom_solimp", "geom_friction", "body_quat"):
        assert np.array_equal(getattr(m, a), getattr(src, a)), a
    checked = 0
    ra, rb = ET.parse(B.SRC).getroot(), ET.parse(path).getroot()
    bodies_b = {e.get("name"): e for e in rb.iter("body")}
    for body in ra.iter("body"):
        ga = [g for g in body.findall("geom") if float(g.get("mass") or 0) > 0]
        if not ga:
            continue
        gb = {g.get("name"): g for g in bodies_b[body.get("name")].findall("geom")}
        r = {round(float(gb[g.get("name")].get("mass")) / float(g.get("mass")), 12) for g in ga}
        if len(r) == 1:
            has_mesh = any(g.get("type") == "mesh" for g in body.findall("geom"))
            bid = src.body(body.get("name")).id
            np.testing.assert_allclose(m.body_inertia[bid], src.body_inertia[bid] * r.pop() * s * s,
                                       rtol=2e-3 if has_mesh else 1e-6)
            checked += 1
    assert checked >= 4  # the four feet (sole only); other bodies mix parts with different mass rules


def test_leg_gains_limits_and_springs(models, src):
    """kp x (m/m60) s and kv x (m/m60) s^1.5 (similar rule), kp and kv x (m/m60) s^2 (literal); the drive set's force
    limits and armature; ankle springs by the build's rule; the X8 set's figures."""
    leg = [src.actuator(j).id for j in B.LEG_JOINTS]
    for key in ((0.50, "D0", "similar"), (0.80, "D0", "similar"), (0.50, "D0", "literal"), (0.80, "D8", "similar")):
        _, info, m = models[key]
        S, s = key[0], key[0] / 0.60
        q = info["mass_model_kg"] / info["mass_model_60_kg"]
        kkp = q * s if key[2] == "similar" else q * s * s
        kkv = q * s * math.sqrt(s) if key[2] == "similar" else q * s * s
        np.testing.assert_allclose(m.actuator_gainprm[leg, 0], src.actuator_gainprm[leg, 0] * kkp, rtol=1e-12)
        np.testing.assert_allclose(m.actuator_biasprm[leg, 2], src.actuator_biasprm[leg, 2] * kkv, rtol=1e-12)
        for c in B.CORNERS:
            ja = m.joint(f"{c}_ankle").id
            assert m.jnt_stiffness[ja] == pytest.approx(src.jnt_stiffness[ja] * q * s, rel=1e-12)
            assert m.dof_damping[m.jnt_dofadr[ja]] == pytest.approx(
                src.dof_damping[src.jnt_dofadr[ja]] * q * s * math.sqrt(s), rel=1e-12)
    _, _, m0 = models[(0.80, "D0", "similar")]
    assert np.array_equal(m0.actuator_forcerange[leg], src.actuator_forcerange[leg])
    _, info8, m8 = models[(0.80, "D8", "similar")]
    lim = {j: m8.actuator_forcerange[m8.actuator(j).id, 1] for j in B.LEG_JOINTS}
    for j, v in lim.items():
        assert v == pytest.approx(0.70 * 120.0 * (0.90 if j.endswith("flexion") else 1.0), rel=1e-12)
        assert m8.dof_armature[m8.jnt_dofadr[m8.joint(j).id]] == pytest.approx(0.008925, rel=1e-12)
    assert info8["leg_drives"]["left_hind_flexion"]["continuous_nm"] == pytest.approx(34.0 * 0.90)


@pytest.mark.parametrize("S", [0.50, 0.80])
def test_neck_spring_cancels_the_same_share(models, src, S):
    """The neck spring's torque at neutral over the head's gravity moment is the source's (70 %, the parts list)."""
    def share(model):
        d = mujoco.MjData(model)
        mujoco.mj_forward(model, d)
        j = model.joint("neck_pitch").id
        spring = -model.jnt_stiffness[j] * (d.qpos[model.jnt_qposadr[j]] - model.qpos_spring[model.jnt_qposadr[j]])
        return spring / d.qfrc_bias[model.jnt_dofadr[j]]
    _, _, m = models[(S, "D0", "similar")]
    assert share(m) == pytest.approx(share(src), rel=1e-9)
    assert share(src) == pytest.approx(0.70, abs=0.002)


def test_servo_class_steps_above_67_5_cm(models, src):
    _, i70, m70 = models[(0.70, "D0", "similar")]
    _, i50, m50 = models[(0.50, "D0", "similar")]
    assert i70["big_servos"] and not i50["big_servos"]
    assert m70.actuator_forcerange[m70.actuator("neck_pitch").id, 1] == pytest.approx(0.70 * 4.44)
    assert np.array_equal(m50.actuator_forcerange[m50.actuator("neck_pitch").id], src.actuator_forcerange[
        src.actuator("neck_pitch").id])


def test_walk_params_froude_scaling():
    """Every WalkParams field is classed once; s = 1 gives the frozen defaults; times x sqrt(s), lengths x s, and with
    the command x sqrt(s) the stride (command x period) goes as s."""
    import walk60
    P = walk60.WalkParams()
    classes = (B.LENGTH_FIELDS + B.TIME_FIELDS + B.SECONDS_GAINS + B.RATE_GAINS + B.SHARE_FIELDS + B.FLAG_FIELDS)
    names = [f.name for f in dataclasses.fields(P)]
    assert sorted(classes) == sorted(names) and len(set(classes)) == len(classes)
    assert B.scaled_walk_params(1.0) == P
    s = 0.80 / 0.60
    Q = B.scaled_walk_params(s)
    assert Q.period_s == pytest.approx(P.period_s * math.sqrt(s))
    assert Q.leg_drop_m == pytest.approx(P.leg_drop_m * s)
    assert Q.k_heading == pytest.approx(P.k_heading / math.sqrt(s))
    assert Q.duty == P.duty and Q.k_roll == P.k_roll and Q.frame == P.frame
    stride = lambda p, sc: B.COMMAND60 * math.sqrt(sc) * p.period_s  # noqa: E731
    assert stride(Q, s) == pytest.approx(stride(P, 1.0) * s)


def _fake_case(S, k, rms60, stand60, bump=None, seeds=B.SEEDS):
    """A raw record whose figures are exactly k x the 60 cm ones, with optional (joint index, {seed: factor})."""
    mm = FF.mass_model(S, FF.DRIVE_SETS["D0"], layout=None, bms=False)["total_kg"]
    trot, stand = [], []
    for sd in seeds:
        f = np.ones(12)
        if bump and sd in bump[1]:
            f[bump[0]] = bump[1][sd]
        r = np.asarray(rms60) * k * f
        trot.append({"seed": sd, "joints": list(B.LEG_JOINTS), "rms_nm": r.tolist(), "peak_nm": (3 * r).tolist(),
                     "p999_nm": (2.5 * r).tolist(), "limit_nm": [21.0] * 12, "share_at_limit": [0.0] * 12,
                     "peak_rpm": [50.0] * 12, "rms_rpm": [10.0] * 12, "rms_over_continuous": (r / 7.5).tolist(),
                     "command_mps": 0.25 * math.sqrt(S / 0.6), "speed_mps_2_20s": 0.25 * math.sqrt(S / 0.6),
                     "speed_mps_10_20s": 0.25 * math.sqrt(S / 0.6), "froude_speed_2_20s": 0.47, "fell": False,
                     "power_w": 20.0, "touchdowns_per_s": {c: 1.5 for c in B.CORNERS}})
        stand.append({"seed": sd, "fell": False, "tau_mean_5_10s_nm": (np.asarray(stand60) * k).tolist(),
                      "load_share_min_100ms_1_10s": [0.2] * 4, "load_share_mean_5_10s": [0.25] * 4,
                      "standing_height_m": 0.3})
    info = {"mass_model_kg": mm, "compiled_mass_kg": mm, "kp_rule": "similar", "drives": "D0"}
    return {"model": {"info": info, "role": "test", "delay_steps": 10}, "trot": trot, "stand": stand}


def test_compare_verdict_logic(tmp_path):
    rms60 = np.linspace(1.0, 6.0, 12)
    st60 = np.linspace(0.4, 3.3, 12)
    ref = _fake_case(0.60, 1.0, rms60, st60)
    ref["model"]["info"]["mass_model_kg"] = B.M60
    k50 = FF.torque_factor(FF.mass_model(0.50, FF.DRIVE_SETS["D0"], layout=None, bms=False)["total_kg"], 0.50)
    cases = {"H0.60": ref,
             "H0.50": _fake_case(0.50, k50, rms60, st60),
             "H0.70": _fake_case(0.70, FF.torque_factor(
                 FF.mass_model(0.70, FF.DRIVE_SETS["D0"], layout=None, bms=False)["total_kg"], 0.70), rms60, st60,
                 bump=(4, {101: 1.30, 103: 0.70})),
             "H0.80": _fake_case(0.80, FF.torque_factor(
                 FF.mass_model(0.80, FF.DRIVE_SETS["D0"], layout=None, bms=False)["total_kg"], 0.80), rms60, st60,
                 bump=(7, {102: 1.30, 104: 2.0}))}
    for n, rec in cases.items():
        (tmp_path / f"scaled_raw_{n}.json").write_text(json.dumps(rec))
    res = B.compare(tmp_path)
    assert res["cases"]["H0.60"]["trot"]["S2_holds"] and res["cases"]["H0.60"]["stand"]["S1_pass"]
    assert res["cases"]["H0.50"]["trot"]["S2_holds"]
    assert max(abs(e) for e in res["cases"]["H0.50"]["trot"]["rms_error_3seed_mean"]) < 1e-9
    assert not res["cases"]["H0.70"]["trot"]["S2_holds"]
    assert res["cases"]["H0.70"]["trot"]["joints_failing"] == [B.LEG_JOINTS[4]]
    # one verdict seed out of the band (and a supporting seed far out) still holds: 2 of 3
    assert res["cases"]["H0.80"]["trot"]["S2_holds"]
    assert (tmp_path / "scaled_trike_compare.csv").stat().st_size < 500_000
    # the post-hoc normal-mode reading sees no slow seed in the synthetic data and reproduces the exact case
    ph = res["cases"]["H0.50"]["post_hoc"]
    assert ph["slow_seeds"] == [] and max(abs(e) for e in ph["normal_pair_rms_error_max"]) < 1e-9


def test_result_files_are_small():
    """Everything this tool writes in the build tree is JSON or CSV under 500 KB."""
    for p in B.RESULTS.glob("scaled_*"):
        assert p.suffix in (".json", ".csv")
        assert p.stat().st_size < 500_000, p.name


def test_a_stored_episode_is_reproduced(tmp_path):
    """The stored 0.50 m results come from this builder and harness: the model rebuilt here has the stored sha256, and
    its trot on seed 101 and its stand on seed 102 give the stored figures exactly."""
    raw_path = B.RESULTS / "scaled_raw_H0.50.json"
    if not raw_path.exists():
        pytest.skip("no stored results yet")
    raw = json.loads(raw_path.read_text())
    path, info = B.write_model(0.50, tmp_path, "D0", "similar")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == raw["model"]["xml_sha256"]
    tr = B.trot_episode(str(path), info, 101, 0.50 / 0.60)
    stored = next(r for r in raw["trot"] if r["seed"] == 101)
    assert tr["rms_nm"] == stored["rms_nm"] and tr["peak_nm"] == stored["peak_nm"]
    st = B.stand_episode(str(path), info, 102)
    stored = next(r for r in raw["stand"] if r["seed"] == 102)
    assert st["tau_mean_5_10s_nm"] == stored["tau_mean_5_10s_nm"]
    assert st["load_share_min_100ms_1_10s"] == stored["load_share_min_100ms_1_10s"]


# ---------------------------------------------------------------- added 28 Nov 2026 after review (F3, MECH-2, PH-1)
def test_adopted_kp_rule_keeps_the_source_gains(tmp_path, src):
    path, info = B.write_model(0.50, tmp_path, "D0", "adopted")
    m = mujoco.MjModel.from_xml_path(str(path))
    leg = [src.actuator(j).id for j in B.LEG_JOINTS]
    assert info["k_kp"] == 1.0 and info["k_kv"] == 1.0
    assert np.array_equal(m.actuator_gainprm[leg, 0], src.actuator_gainprm[leg, 0])
    assert np.array_equal(m.actuator_biasprm[leg, 2], src.actuator_biasprm[leg, 2])


def test_layout_copy_moves_the_legs_not_the_masses(tmp_path, src):
    """At 60 cm, L2 moves each link plane 43.7 mm outboard (the study's L2 for D0) and the knee drive onto the pitch
    axis; the total mass is the as-built copy's."""
    path, info = B.write_model(0.60, tmp_path, "D0", "similar", layout="L2")
    m = mujoco.MjModel.from_xml_path(str(path))
    for c, sg in (("left_hind", 1.0), ("right_fore", -1.0)):
        b = m.body(f"{c}_upper_link").id
        assert m.body_pos[b, 1] == pytest.approx(sg * (0.064 + 0.0437), abs=1e-9)
    assert info["layout"]["moves"]["left_hind"]["link_plane_dy_m"] == pytest.approx(0.0437, abs=1e-9)
    assert m.body_mass.sum() == pytest.approx(src.body_mass.sum(), rel=1e-12)
    g = m.geom("left_hind_x4_flexion_stator").id
    assert abs(m.geom_pos[g, 0]) < 1e-9 and abs(m.geom_pos[g, 2]) < 1e-9   # on the pitch axis


def test_post_hoc_cases_were_planned_in_the_addendum():
    """Every case added after review is in the addendum (written before they ran), with the same settings."""
    add = json.loads((B.RESULTS / "scaled_trike_plan_addendum.json").read_text())
    post = [n for n, c in B.CASES.items() if "post hoc" in c[4]]
    assert sorted(post) == sorted(add["cases"])
    for n in post:
        S, dk, rule = B.CASES[n][:3]
        a = add["cases"][n]
        assert (a["size_m"], a["drive_set"], a["kp_rule"], a["layout"]) == (S, dk, rule, B.case_layout(n))
