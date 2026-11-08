"""Tests for the carpet screen (designlab/carpet.py).

Run from hardware/trex_form_study:  python -m pytest tests/test_carpet.py -q
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import designlab  # noqa: E402,F401  (selects the GL backend before mujoco renders)
from designlab import carpet, gait, sizing, statics  # noqa: E402

STAGE_A = ROOT / "round1" / "stage_a"
G = 9.81


# ------------------------------------------------------------------------------------------------
# Data file
# ------------------------------------------------------------------------------------------------
def test_every_source_is_cited_and_every_screening_value_has_a_basis():
    r = carpet.research()
    assert r["accessed"] == "2026-11-07"
    for key, src in r["sources"].items():
        assert src["accessed"] == "2026-11-07", key
        assert src.get("url") or src.get("location"), key
        assert src.get("verification"), key
    sv = r["screening_values"]
    for name in carpet.PILES:
        c = sv["pile_classes"][name]
        assert c["basis_height"] and c["basis_compression"]
        assert "single_source" in c
    for key in ("test_pressure_kPa", "reference_pressure_kPa", "carpet_friction_band", "contact_impedance_d0",
                "contact_damping_ratio", "torsional_rolling_m"):
        assert sv[key]["basis"], key
    # the robot_floor carpet is recorded exactly as it is defined on the platform branches
    rf = r["robot_floor_definition"]
    assert rf["carpet"] == {"friction": [0.95, 0.009, 0.0005], "solref": [0.024, 1.2],
                            "solimp": [0.65, 0.94, 0.004, 0.5, 2.0]}
    assert rf["hard"]["solref"] == [0.008, 1.0]


def test_pile_classes_are_ordered_and_medium_and_high_lie_in_the_spec_depth():
    p = carpet.pile_classes()
    h = [p[n]["height_m"] for n in carpet.PILES]
    d = [p[n]["depth_m"] for n in carpet.PILES]
    assert h == sorted(h) and d == sorted(d)
    assert h == pytest.approx([0.00635, 0.0127, 0.01905])
    for n in ("medium", "high"):  # BEHAVIOR_SPEC section 1: compliant pile depth 8-15 mm
        assert 0.008 <= p[n]["depth_m"] <= 0.015
    for n in carpet.PILES:  # secant stiffness = (98 - 1.96) kPa / compressed depth
        assert p[n]["stiffness_pa_per_m"] * p[n]["depth_m"] == pytest.approx(96.04e3)
        assert p[n]["stiffness_pa_per_m"] * p[n]["depth_m"] == pytest.approx(carpet.pressure_span_pa())


def test_derived_compression_and_friction_band_recompute_from_the_transcribed_tables():
    r = carpet.research()
    hg = r["compression"]["hirai_gunji_2001_98kPa"]
    for k, smp in hg["samples"].items():
        mm = smp["compressibility_pct"] / 100.0 * smp["thick_mm"]
        assert hg["derived_compression_mm"][k] == pytest.approx(mm, abs=0.006)
        assert hg["derived_fraction_of_pile"][k] == pytest.approx(mm / smp["pile_mm"], abs=0.006)
    low = max(hg["derived_fraction_of_pile"][k] for k, s in hg["samples"].items() if s["pile_mm"] <= 6.5)
    assert r["screening_values"]["pile_classes"]["low"]["compression_fraction"] == pytest.approx(low)
    assert r["screening_values"]["pile_classes"]["medium"]["compression_fraction"] == pytest.approx(
        max(hg["derived_fraction_of_pile"].values()))
    dv = r["compression"]["daulta_varshney_2021_200kPa"]
    for k, row in dv["table_2"].items():
        tr, cr = row["TR_pct"] / 100.0, row["CR_pct"] / 100.0
        assert dv["derived_T200_over_T2"][k] == pytest.approx((tr - cr) / (1.0 - cr), abs=0.0015)
    f = r["friction"]["hirai_gunji_2001"]
    rubber = (f["oy_pull_slip_meter"]["pvc_sole_dry"] + f["oy_pull_slip_meter"]["pvc_sole_wet"]
              + f["jsma_slip_meter"]["sbr_sole_dry"] + f["jsma_slip_meter"]["sbr_sole_wet"]
              + sum((f["inclined_plane"][f"load_{n}N"] for n in (0, 10, 20, 30)), []))
    assert r["screening_values"]["carpet_friction_band"]["value"] == [min(rubber), max(rubber)]
    assert min(rubber) <= r["friction"]["brandao_2017"]["table_I_mean_sd"]["carpet"][0] <= max(rubber)


# ------------------------------------------------------------------------------------------------
# Geometry
# ------------------------------------------------------------------------------------------------
GEOMS = """
<mujoco>
  <asset><mesh name="tet" vertex="0 0 0  0.05 0 0  0 0.04 0  0.01 0.01 0.03"/></asset>
  <worldbody>
    <body pos="0 0 0.5" euler="0.3 -0.7 1.1">
      <geom name="box" type="box" size=".03 .02 .01"/>
      <geom name="cap" type="capsule" size=".01 .04" pos=".1 0 0" euler="0.4 0.2 0"/>
      <geom name="cyl" type="cylinder" size=".02 .03" pos="0 .1 0" euler="1.0 0.3 0"/>
      <geom name="sph" type="sphere" size=".015" pos="0 0 .1"/>
      <geom name="ell" type="ellipsoid" size=".03 .02 .01" pos=".1 .1 0" euler="0.5 0.5 0.5"/>
      <geom name="msh" type="mesh" mesh="tet" pos="-.1 0 0"/>
    </body>
  </worldbody>
</mujoco>"""


def _surface_samples(m, d, g, n=400):
    R, c, s, t = d.geom_xmat[g].reshape(3, 3), d.geom_xpos[g], m.geom_size[g], m.geom_type[g]
    u, v = np.meshgrid(np.linspace(0, np.pi, n), np.linspace(0, 2 * np.pi, n))
    sph = np.stack([np.sin(u) * np.cos(v), np.sin(u) * np.sin(v), np.cos(u)], -1).reshape(-1, 3)
    ring = np.stack([np.cos(v[:, 0]), np.sin(v[:, 0]), 0 * v[:, 0]], -1)
    if t == mujoco.mjtGeom.mjGEOM_BOX:
        local = np.array([[a, b, cc] for a in (-1, 1) for b in (-1, 1) for cc in (-1, 1)]) * s[:3]
    elif t == mujoco.mjtGeom.mjGEOM_CAPSULE:
        local = np.vstack([sph * s[0] + [0, 0, h] for h in (-s[1], s[1])])
    elif t == mujoco.mjtGeom.mjGEOM_CYLINDER:
        local = np.vstack([ring * s[0] + [0, 0, h] for h in (-s[1], s[1])])
    elif t == mujoco.mjtGeom.mjGEOM_SPHERE:
        local = sph * s[0]
    elif t == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
        local = sph * s[:3]
    else:
        a, k = m.mesh_vertadr[m.geom_dataid[g]], m.mesh_vertnum[m.geom_dataid[g]]
        local = m.mesh_vert[a : a + k]
    return c + local @ R.T


def test_lowest_point_is_exact_for_every_geom_type():
    m = mujoco.MjModel.from_xml_string(GEOMS)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    for g in range(m.ngeom):
        p = carpet.lowest_point(m, d, g)
        pts = _surface_samples(m, d, g)
        q = pts[np.argmin(pts[:, 2])]
        assert p[2] <= q[2] + 1e-12, carpet._gname(m, g)
        assert p[2] == pytest.approx(q[2], abs=2e-6), carpet._gname(m, g)
        assert np.linalg.norm(p - q) < 2e-3, carpet._gname(m, g)


UNDERSIDE = """
<mujoco>
  <compiler angle="radian"/>
  <worldbody>
    <body name="foot">
      <geom name="pad" type="box" size=".0101 .0083 .004" pos=".0013 -.0007 .004"/>
      <geom name="turned" type="box" size=".010 .006 .004" pos=".05 0 .004" euler="0 0 0.5236"/>
      <geom name="toe" type="capsule" size=".008 .02" pos="0 .05 .009" euler="0 1.5707963267948966 0"/>
      <geom name="plate" type="box" size=".03 .015 .002" pos=".001 0 .012"/>
    </body>
  </worldbody>
</mujoco>"""


def test_underside_map_is_exact_for_level_boxes_and_follows_the_capsule_surface():
    m = mujoco.MjModel.from_xml_string(UNDERSIDE)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    gid = {n: statics.gid(m, n) for n in ("pad", "turned", "toe", "plate")}
    h = 0.001
    c = carpet.underside_cells(m, d, [gid["pad"]], h)  # a box off the grid's lines: exact area and moments
    a, b = 0.0202, 0.0166
    assert c["area"].sum() == pytest.approx(a * b, rel=1e-12)
    assert np.sum(c["area"] * c["x"]) / c["area"].sum() == pytest.approx(0.0013, abs=1e-12)
    xc = 0.0013
    assert np.sum(c["area"] * (c["x"] - xc) ** 2) == pytest.approx(a * b * a**2 / 12, rel=0.01)
    assert np.all(c["z"] == pytest.approx(0.0, abs=1e-12))
    c = carpet.underside_cells(m, d, [gid["turned"]], h)  # turned in plan: 4 x 4 samples per cell
    assert c["area"].sum() == pytest.approx(0.02 * 0.012, rel=0.02)
    c = carpet.underside_cells(m, d, [gid["toe"]], 0.0005)  # under the cylinder part: z = c - sqrt(r^2 - dy^2)
    sel = (np.abs(c["x"]) < 0.019) & (np.abs(c["y"] - 0.05) < 0.0075)
    assert sel.sum() > 100
    assert c["z"][sel] == pytest.approx(0.009 - np.sqrt(0.008**2 - (c["y"][sel] - 0.05) ** 2), abs=2e-6)
    # union: under the pad the pad is lowest, the plate fills the rest of its own footprint
    u = carpet.underside_cells(m, d, [gid["pad"], gid["plate"]], h)
    assert np.sum(u["area"][u["geom"] == gid["pad"]]) == pytest.approx(a * b, rel=1e-12)
    assert u["area"].sum() == pytest.approx(0.06 * 0.03, rel=1e-12)  # the plate covers the pad's footprint
    assert np.all(u["z"][u["geom"] == gid["plate"]] == pytest.approx(0.010, abs=1e-12))


# ------------------------------------------------------------------------------------------------
# Contact model
# ------------------------------------------------------------------------------------------------
BOX = """
<mujoco>
  <option timestep="0.002" integrator="implicitfast" cone="elliptic"/>
  <worldbody>
    <geom name="floor" type="plane" size="1 1 .1"/>
    <body name="foot" pos="0 0 0.0100"><freejoint/>
      <geom name="pad" type="box" size=".03 .02 .01" mass="{mass}"/>
      <geom name="load" type="sphere" size=".005" pos="{cx} 0 0.2" mass="{top}"/>
    </body>
  </worldbody>
</mujoco>"""


def _settle(m, steps=3000, n_con=4):
    d = mujoco.MjData(m)
    for _ in range(steps):
        mujoco.mj_step(m, d)
    mujoco.mj_forward(m, d)
    assert d.ncon == n_con
    return d, float(-np.mean([d.contact[i].dist for i in range(d.ncon)]))


def test_soft_floor_gives_the_derived_stiffness_and_depth_scales_with_weight():
    """The derivation in item 1 of the module docstring, checked against MuJoCo itself."""
    d0 = carpet.screening_values()["contact_impedance_d0"]["value"]
    depths = {}
    for mass in (1.0, 4.0):
        m = mujoco.MjModel.from_xml_string(BOX.format(mass=mass, top=0.0, cx=0.0))
        A_hat = float(m.body_invweight0[statics.bid(m, "foot"), 0])
        fl = carpet.soft_floor(20000.0, 4, A_hat, 0.009, (0.54, 0.009, 0.0005))
        carpet.apply_floor(m, fl)
        d, depths[mass] = _settle(m)
        adr = d.contact[0].efc_address
        assert d.efc_diagA[adr] == pytest.approx(A_hat)  # MuJoCo's A-hat is the body's invweight0
        assert d.efc_R[adr] == pytest.approx((1 - d0) / d0 * A_hat)
        assert depths[mass] == pytest.approx(mass * G / 20000.0, rel=1e-6)
    assert depths[4.0] / depths[1.0] == pytest.approx(4.0, rel=1e-6)  # a physical spring: depth follows load


def test_a_fixed_time_constant_floor_sinks_the_same_under_any_weight():
    """robot_floor's carpet (positive solref, ramped impedance): the static depth matches the closed form and does
    not depend on the body's weight, so a fixed solref is not a fixed physical carpet."""
    rf = carpet.research()["robot_floor_definition"]["carpet"]
    floor = dict(rf, condim=6)
    depths = []
    for mass in (1.0, 4.0):
        m = mujoco.MjModel.from_xml_string(BOX.format(mass=mass, top=0.0, cx=0.0))
        carpet.apply_floor(m, floor)
        A_hat = float(m.body_invweight0[statics.bid(m, "foot"), 0])
        _, depth = _settle(m)
        assert depth == pytest.approx(carpet.static_depth(rf["solref"], rf["solimp"], A_hat, mass * G / 4), rel=1e-5)
        depths.append(depth)
    assert depths[1] == pytest.approx(depths[0], rel=1e-6)


def test_pair_params_follow_mujoco_mixing_and_priority():
    m = mujoco.MjModel.from_xml_string(BOX.format(mass=1.0, top=0.0, cx=0.0))
    fl, pad = statics.gid(m, "floor"), statics.gid(m, "pad")
    m.geom_solref[fl] = (0.02, 1.0)
    m.geom_solref[pad] = (0.01, 1.0)
    m.geom_friction[pad] = (0.9, 0.02, 0.001)
    for prio in (0, 1):
        m.geom_priority[fl] = prio
        d = mujoco.MjData(m)
        d.qpos[2] = 0.0095
        mujoco.mj_forward(m, d)
        pp = carpet.pair_params(m, fl, pad)
        c = d.contact[0]
        assert np.allclose(c.solref, pp["solref"]) and np.allclose(c.solimp, pp["solimp"])
        assert np.allclose([c.friction[0], c.friction[2], c.friction[3]], pp["friction"])
    assert np.allclose(pp["solref"], (0.02, 1.0))  # the higher-priority floor wins


def test_probe_pile_carries_the_body_as_the_linear_pile_predicts():
    """Probes under a free box (the pile floor of item 1): the settled depth and every probe force are the linear
    pile's, with and without an off-centre load that tilts the box."""
    k_A = 1.0e7  # 10 kPa/mm
    for top, cx in ((0.0, 0.0), (1.0, 0.012)):
        xml = BOX.format(mass=1.0, top=top, cx=cx)
        m0 = mujoco.MjModel.from_xml_string(xml)
        d0 = mujoco.MjData(m0)
        mujoco.mj_forward(m0, d0)
        cells = carpet.underside_cells(m0, d0, [statics.gid(m0, "pad")], carpet.PROBE_GRID_M)
        assert cells["area"].sum() == pytest.approx(0.06 * 0.04, rel=1e-12)
        sv = carpet.screening_values()
        floor = dict(A_hat_per_kg=float(m0.body_invweight0[1, 0]), depth_m=0.009, friction=[0.54, 0.009, 0.0005],
                     solimp=[sv["contact_impedance_d0"]["value"]] * 2 + [0.009, 0.5, 2.0],
                     dampratio=sv["contact_damping_ratio"]["value"], timestep=0.002, condim=3)
        spec = mujoco.MjSpec.from_string(xml)
        centres = np.column_stack([cells["x"], cells["y"], cells["z"] + carpet.PROBE_RADIUS_M]) - d0.xpos[1]
        carpet.add_probes(spec, "foot", centres, k_A * cells["area"], floor, 2)
        m = spec.compile()
        assert float(m.body_subtreemass[1]) == pytest.approx(1.0 + top)  # probes add no mass
        fl, pad = statics.gid(m, "floor"), statics.gid(m, "pad")
        m.geom_priority[fl] = 1
        m.geom_conaffinity[fl] |= carpet.PROBE_BIT
        m.geom_contype[pad] = m.geom_conaffinity[pad] = 0
        W = float(m.body_subtreemass[1]) * G
        d, depth = _settle(m, 3000, len(cells["z"]))
        n = d.ncon
        fn = d.efc_force[d.contact.efc_address[:n]]
        dep = -d.contact.dist[:n]
        assert np.all(d.efc_state[d.contact.efc_address[:n]] != 0)  # every probe pushes
        assert fn.sum() == pytest.approx(W, rel=1e-6)
        assert fn == pytest.approx(k_A * carpet.PROBE_GRID_M**2 * dep, rel=1e-4)  # each probe is k_A x its area
        assert depth == pytest.approx(carpet.pile_sinkage(cells, k_A, W), rel=1e-4)


def test_lean_follows_the_linear_solution_and_tips_when_the_front_unloads():
    """Item 5: with every spring loaded the static lean is the linear one; with the CoM further behind the stiffness
    centroid the front springs unload and the stance tips although the linear margin (1.3) is above 1."""
    W, h, margin = 100.0, 0.3, 1.3

    def springs(u):  # equal springs at levers u (CoM at 0), each split over a pair at y = +-50 mm
        u = np.asarray(u, float)
        k = margin * W * h / np.sum((u - u.mean()) ** 2)
        return dict(x=np.repeat(u, 2), y=np.tile([-0.05, 0.05], len(u)), z=np.zeros(2 * len(u)),
                    k=np.full(2 * len(u), k / 2.0), z_sole=0.0)

    sp = springs([-0.04, 0.0, 0.06])
    st = carpet.stance_statics(sp, W, np.array([0.0, 0.0, h]))
    uc = 0.02 / 3.0
    assert st["linear_margin"]["pitch"] == pytest.approx(margin)
    assert st["stiffness_centroid_ahead_of_com_mm"] == pytest.approx(1e3 * uc)
    K = margin * W * h
    assert st["lean_deg"]["pitch"] == pytest.approx(np.degrees(-uc * W / (K - W * h)), abs=0.01)
    assert st["holds"]
    sp = springs([-0.03, 0.01, 0.07])
    st = carpet.stance_statics(sp, W, np.array([0.0, 0.0, h]))
    assert st["linear_margin"]["pitch"] == pytest.approx(margin)
    assert st["lean_deg"]["pitch"] is None and not st["holds"]


# ------------------------------------------------------------------------------------------------
# Friction: exact unwrap across the period boundary
# ------------------------------------------------------------------------------------------------
def test_derivatives_unwrap_one_stride_and_keep_the_gait_mirror_symmetric():
    """Root x = v t + a sin(4 pi t / T + phi): a half-period shift moves it by half a stride, so velocity and
    acceleration at frame k and k + n/2 must agree everywhere, the period boundary included.  (gait.derivatives
    estimates the jump as x_{n-1} + x_1 - x_0, which is wrong on frames 0, 1, n-2 and n-1 unless phi = 0.)"""
    m = mujoco.MjModel.from_xml_string(BOX.format(mass=1.0, top=0.0, cx=0.0))
    n, T, v, a, phi = 160, 1.2, 0.2, 0.004, 0.7
    w, dt = 4 * np.pi / T, T / n
    t = np.arange(n) * dt
    q = np.zeros((n, m.nq))
    q[:, 0] = v * t + a * np.sin(w * t + phi)
    q[:, 2] = 0.3
    q[:, 3] = 1.0
    traj = dict(q=q, dt=dt, t=t, info=dict(stride=v * T, period=T))
    qd, qdd = carpet.derivatives(SimpleNamespace(m=m), traj)
    k = np.arange(n)
    assert qd[k, 0] == pytest.approx(qd[(k + n // 2) % n, 0], abs=1e-9)
    assert qdd[k, 0] == pytest.approx(qdd[(k + n // 2) % n, 0], abs=1e-6)
    # central differences of a sinusoid, exactly, at every frame
    assert qd[:, 0] == pytest.approx(v + a * np.sin(w * dt) / dt * np.cos(w * t + phi), abs=1e-10)
    assert qdd[:, 0] == pytest.approx(-a * (np.sin(w * dt) / dt) ** 2 * np.sin(w * t + phi), abs=1e-7)


# ------------------------------------------------------------------------------------------------
# Toe catch on a synthetic profile
# ------------------------------------------------------------------------------------------------
def test_clearance_metrics_on_a_sine_squared_lift():
    L, stride, delta = 0.040, 0.22, 0.010
    s = np.linspace(0.0, 1.0, 401)
    prof = {side: dict(s=s, z=L * np.sin(np.pi * s) ** 2, x=stride * (10 * s**3 - 15 * s**4 + 6 * s**5),
                       frames=np.arange(len(s)), geom=["pad"] * len(s)) for side in statics.SIDES}
    r = carpet.clearance_metrics(prof, delta, 0.015, 0.4)
    assert r["mid_swing_min_m"] == pytest.approx(0.5 * L - delta, abs=1e-9)  # window edge, sin^2(pi/4) = 0.5
    assert r["phase"] in (pytest.approx(0.25), pytest.approx(0.75))
    s_exit = np.arcsin(np.sqrt(delta / L)) / np.pi
    assert r["exit_phase"] == pytest.approx(s_exit, abs=2e-4)
    assert r["entry_phase"] == pytest.approx(1.0 - s_exit, abs=2e-4)
    mj = 10 * s_exit**3 - 15 * s_exit**4 + 6 * s_exit**5
    assert r["travel_after_liftoff_m"] == pytest.approx(stride * mj, rel=5e-3)
    assert r["travel_before_touchdown_m"] == pytest.approx(stride * mj, rel=5e-3)


# ------------------------------------------------------------------------------------------------
# Round-one model, end to end (short walks and stands to stay fast)
# ------------------------------------------------------------------------------------------------
@pytest.mark.skipif(not (STAGE_A / "candidates" / "60-M-T-10.json").exists(), reason="round-one records not present")
def test_round_one_60cm_toe_catch_friction_standing_and_shortlist_hook(tmp_path):
    from designlab import shortlist

    rec, res = carpet.rebuild_round1("60-M-T-10", STAGE_A, tmp_path)
    assert res["summary"]["xml_sha256"] == rec["xml_sha256"]
    M, q_home, design = res["model"], res["q_home"], res["design"]
    xml = res["summary"]["xml"]
    H = design.sk.H
    spec = replace(sizing.NOMINAL, n=64, zmp_iters=1)
    tr = gait.generate(M, q_home, H, spec)
    # friction: the exact unwrap recomputes the wrap frames; governing <= as reported; the band-limited value is close
    fr = carpet.friction_screen(M, tr, gait.inverse_dynamics(M, tr))
    assert fr["frames_recomputed_for_unwrap"] == [0, 1, 62, 63]
    assert 0.0 < fr["required_max_single"] <= fr["required_governing"] <= fr["required_max"] < 1.0
    assert 0.0 < fr["required_band_limited"] < 1.0
    assert fr["within_carpet_band"]
    # toe catch: central (whole-foot, peak-load) sinkage below the 98 kPa depth on medium and high pile
    under = carpet.foot_underside(M, q_home)
    piles = carpet.pile_classes()
    sink = {p: carpet.pile_sinkage(under["left"], pc["stiffness_pa_per_m"], fr["peak_foot_load_N"])
            for p, pc in piles.items()}
    tc = carpet.toe_catch(M, tr, H, piles, sink)
    hard = tc["hard_floor"]
    assert hard["mid_swing_min_frac_H"] == pytest.approx(0.5 * spec.lift_frac, rel=0.06)  # flat pads lowest
    assert "pad" in hard["lowest_geom"]
    assert tc["touchdown_phase"] >= 1.0
    # toe and claw tips (not the pads under the toes) sit just above the pad plane while the foot is flat
    assert 0.0 < hard["toe_points_mid_swing_min_mm"] - hard["mid_swing_min_mm"] < 0.01 * H * 1e3
    for name, p in tc["piles"].items():
        assert p["mid_swing_min_above_pile_top_mm"] == pytest.approx(hard["mid_swing_min_mm"] - p["stance_sinkage_mm"],
                                                                      abs=1e-6)
        assert p["meets_carpet_target"] == (p["mid_swing_min_above_pile_top_mm"] >= tc["target_carpet_mm"])
        assert 0.0 < p["toe_down_pitch_to_reach_pile_top_deg"] < 20.0 and p["toe_down_first_geom"].startswith("hw_")
        if name != "low":
            assert p["load_based"]["stance_sinkage_mm"] < p["stance_sinkage_mm"]
    # standing: hard floor and low pile hold with the predicted sinkage; on medium pile the linear margin is above 1
    # (a necessary condition only) but the front of the foot unloads and the stance tips, rigid and simulated alike
    floors = carpet.floor_variants(M.m)
    stand = {k: carpet.standing_check(xml, q_home, design, floors[k], H, seconds=4.0, max_seconds=8.0)
             for k in ("hard", "low", "medium")}
    assert stand["hard"]["upright"] and stand["low"]["upright"]
    for side in statics.SIDES:
        f = stand["low"]["feet"][side]
        assert f["sinkage_mm"] == pytest.approx(f["predicted_mm"], rel=0.01)
        assert stand["hard"]["feet"][side]["sinkage_mm"] < 0.5
    assert stand["low"]["contact"]["probes_per_foot"]["left"] > 100
    med = stand["medium"]
    assert med["statics"]["linear_margin"]["pitch"] > 1.0 and not med["statics"]["holds"]
    assert med["status"] == "tipped" and not med["upright"]
    tip = carpet.passive_tip(M, q_home, {"whole_foot": under})
    crit = tip["whole_foot"]["critical_compression_98kPa_mm"]
    assert 1e3 * piles["low"]["depth_m"] < crit < 1e3 * piles["medium"]["depth_m"]
    assert tip["pad_area_centroid_ahead_of_com_mm"] < tip["pad_corner_centroid_ahead_of_com_mm"]
    # the round-two hook, called as round2.screen_one calls it, and its summary through the shortlist
    summ = carpet.screen_record("60-M-T-10", rec, xml, q_home, design, {"nominal": tr}, tmp_path / "scr", seconds=1.0,
                                max_seconds=1.0)
    assert (tmp_path / "scr" / "60-M-T-10_carpet.json").exists() and (tmp_path / "scr" / "60-M-T-10_carpet.png").exists()
    assert summ["pass"] is True
    assert summ["toe_clearance_margin_mm"] == pytest.approx(min(summ["swing"]["nominal"]["above_pile_top_mm"].values()))
    assert summ["friction_margin"] == pytest.approx(summ["friction"]["nominal"]["margin_carpet_band_low"])
    rv = {"carpet": summ}
    assert shortlist.gates(rec, rv)["carpet"] is True
    sc = shortlist.scores(rec, rv)
    assert sc["why"]["carpet"] == dict(toe_clearance_margin_mm=summ["toe_clearance_margin_mm"],
                                       friction_margin=summ["friction_margin"])
    assert 0.0 < sc["scores"]["carpet"] <= 1.0


@pytest.mark.skipif(not (STAGE_A / "candidates" / "60-M-H-10.json").exists(), reason="round-one records not present")
def test_cli_writes_json_and_png_and_refuses_round1(tmp_path):
    carpet.main(["60-M-H-10", "--out", str(tmp_path), "--seconds", "0.5", "--max-seconds", "0.5"])
    doc = json.loads((tmp_path / "60-M-H-10_carpet.json").read_text())
    assert doc["matches_stage_a"] is True
    assert (tmp_path / "60-M-H-10_carpet.png").stat().st_size > 10_000
    assert set(doc["toe_catch"]) == {"nominal", "brisk"}
    assert set(doc["standing"]) == {"hard", "robot_floor_carpet", "low", "medium", "high"}
    assert {"pass", "toe_clearance_margin_mm", "friction_margin"} <= set(doc["summary"])
    with pytest.raises(SystemExit):
        carpet.main(["60-M-H-10", "--out", str(ROOT / "round1" / "carpet_should_not_exist")])
    assert not (ROOT / "round1" / "carpet_should_not_exist").exists()
