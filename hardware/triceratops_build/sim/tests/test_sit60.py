"""Tests for the sit and collapse study (sim/sit60.py). Run by hand:
    python -m pytest hardware/triceratops_build/sim/tests
"""

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")  # before sit60, which imports it

SIM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIM))

import sit60  # noqa: E402


def test_test_copy_adds_only_the_skid(tmp_path):
    """The test copy is the adopted model plus one massless geom; the adopted file is not touched."""
    before = hashlib.sha256(sit60.MODEL.read_bytes()).hexdigest()
    path = sit60.write_test_copy(0.27, 0.02, None, str(tmp_path))
    assert hashlib.sha256(sit60.MODEL.read_bytes()).hexdigest() == before
    a = sit60.MODEL.read_text().splitlines()
    b = path.read_text().splitlines()
    assert len(b) == len(a) + 1
    added = [line for line in b if line not in a]
    assert len(added) == 1 and 'name="belly_skid_TEST"' in added[0] and 'mass="0"' in added[0]
    ma = mujoco.MjModel.from_xml_path(str(sit60.MODEL))
    mb = mujoco.MjModel.from_xml_path(str(path))
    assert mb.ngeom == ma.ngeom + 1
    assert mb.body_mass.sum() == pytest.approx(ma.body_mass.sum(), abs=1e-12)
    g = mb.geom("belly_skid_TEST").id
    assert mb.geom_bodyid[g] == mb.body("chassis").id
    # its bottom 0.27 m below the chassis origin, its top at the belly shell's lowest point, centred on com_x
    assert mb.geom_pos[g][2] - mb.geom_size[g][2] == pytest.approx(-0.27, abs=1e-5)
    assert mb.geom_pos[g][2] + mb.geom_size[g][2] == pytest.approx(sit60.SKID_TOP_Z, abs=1e-5)
    assert mb.geom_pos[g][0] == pytest.approx(0.02, abs=1e-5)
    assert mb.geom_size[g][0] == pytest.approx(sit60.SKID_HALF_X, abs=1e-5)
    wide = sit60.write_test_copy(0.27, 0.02, None, str(tmp_path), sit60.SKID_WIDE_HALF_Y, sit60.SKID_WIDE_FRONT_X)
    mw = mujoco.MjModel.from_xml_path(str(wide))
    gw = mw.geom("belly_skid_TEST").id
    assert mw.geom_size[gw][1] == pytest.approx(sit60.SKID_WIDE_HALF_Y)
    # the same rear end as the narrow skid; the front end moved back to SKID_WIDE_FRONT_X
    assert mw.geom_pos[gw][0] - mw.geom_size[gw][0] == pytest.approx(0.02 - sit60.SKID_HALF_X, abs=1e-5)
    assert mw.geom_pos[gw][0] + mw.geom_size[gw][0] == pytest.approx(sit60.SKID_WIDE_FRONT_X, abs=1e-5)
    assert wide != path
    soft = sit60.write_test_copy(0.27, 0.02, sit60.SKID_SOFT_SOLREF, str(tmp_path), priority=sit60.SKID_SOFT_PRIORITY)
    ms = mujoco.MjModel.from_xml_path(str(soft))
    gs = ms.geom("belly_skid_TEST").id
    assert tuple(ms.geom_solref[gs]) == pytest.approx(sit60.SKID_SOFT_SOLREF)
    assert ms.geom_priority[gs] == sit60.SKID_SOFT_PRIORITY > ms.geom_priority[0]
    with pytest.raises(ValueError):
        sit60.skid_geom_xml(0.27, 0.02, front_x=-0.2)


def _skid_floor_contact(path, chassis_z):
    m = mujoco.MjModel.from_xml_path(str(path))
    d = mujoco.MjData(m)
    d.qpos[2] = chassis_z
    mujoco.mj_forward(m, d)
    g = m.geom("belly_skid_TEST").id
    for i in range(d.ncon):
        c = d.contact[i]
        if {int(c.geom1), int(c.geom2)} == {0, g}:
            # copies: the contact is a view into ``d``, which is freed on return
            return {"solref": c.solref.copy(), "solimp": c.solimp.copy(), "friction": c.friction.copy(),
                    "dim": int(c.dim)}
    raise AssertionError("no skid-floor contact")


def test_soft_skid_contact_runs_at_its_own_solref(tmp_path):
    """The soft skid's floor contact uses its 0.03 s (priority over the floor), not the solmix average of 0.019 s,
    and nothing else about the contact changes against the rigid skid."""
    hard = sit60.write_test_copy(0.27, 0.02, None, str(tmp_path))
    soft = sit60.write_test_copy(0.27, 0.02, sit60.SKID_SOFT_SOLREF, str(tmp_path), priority=sit60.SKID_SOFT_PRIORITY)
    nopri = sit60.write_test_copy(0.27, 0.02, sit60.SKID_SOFT_SOLREF, str(tmp_path), tag="soft_no_priority")
    ch, cs, cn = (_skid_floor_contact(p, 0.268) for p in (hard, soft, nopri))
    assert ch["solref"][0] == pytest.approx(0.008)
    assert cs["solref"][0] == pytest.approx(sit60.SKID_SOFT_SOLREF[0])
    assert cn["solref"][0] == pytest.approx(0.5 * (0.008 + sit60.SKID_SOFT_SOLREF[0]))  # what the first run used
    assert tuple(cs["friction"]) == pytest.approx(tuple(ch["friction"]))
    assert tuple(cs["solimp"]) == pytest.approx(tuple(ch["solimp"]))
    assert cs["dim"] == ch["dim"]


def test_wide_skid_clears_the_elbow_drives(tmp_path):
    """The wide skid's front end is behind the fore X6 flexion stators' rear faces, standing and in the sit pose."""
    import walk60

    geo = sit60.sit_geometry()
    spec = sit60.skid_spec("skid_wide", geo["sit_height_m"])
    assert spec["front_x"] == sit60.SKID_WIDE_FRONT_X and spec["half_y"] == sit60.SKID_WIDE_HALF_Y
    kin = walk60.Kinematics(str(sit60.MODEL))
    m, d = kin.m, kin.d
    q_sit = {c: np.radians(v) for c, v in geo["q_sit_deg"].items()}
    for q in (kin.q_home, q_sit):
        kin.set_q(q)
        ch = m.body("chassis").id
        R = d.xmat[ch].reshape(3, 3)
        for side in ("left", "right"):
            g = m.geom(f"{side}_fore_x6_flexion_stator_collision").id
            x = (R.T @ (d.geom_xpos[g] - d.xpos[ch]))[0]
            assert x - m.geom_size[g][0] > sit60.SKID_WIDE_FRONT_X + 0.01, side
    assert sit60.skid_spec("skid_e20mm", 0.26)["depth_m"] == pytest.approx(0.28)
    assert sit60.skid_spec("skid_soft", 0.26)["priority"] == sit60.SKID_SOFT_PRIORITY
    with pytest.raises(ValueError):
        sit60.skid_spec("adopted", 0.26)


def _skid_posed(path, chassis_z):
    m = mujoco.MjModel.from_xml_path(str(path))
    d = mujoco.MjData(m)  # qpos0: the legs at their home angles
    d.qpos[2] = chassis_z
    mujoco.mj_forward(m, d)
    return m, d, m.geom("belly_skid_TEST").id


def test_skid_contact_checks_tell_the_floor_from_a_leg(tmp_path):
    """``_touches_floor`` and ``_touches_other_than_floor`` (the before-the-stop checks in ``starts``) on test copies
    posed by hand: the narrow skid in the air touches nothing and on the floor only the floor; a copy 0.40 m wide,
    held in the air, reaches into the forelegs and touches a leg but not the floor."""
    narrow = sit60.write_test_copy(0.27, 0.02, None, str(tmp_path))
    m, d, g = _skid_posed(narrow, 0.6)
    assert not sit60._touches_floor(m, d, g) and not sit60._touches_other_than_floor(d, g)
    m, d, g = _skid_posed(narrow, 0.26)  # its bottom 10 mm into the floor
    assert sit60._touches_floor(m, d, g) and not sit60._touches_other_than_floor(d, g)
    into_legs = sit60.write_test_copy(0.27, 0.02, None, str(tmp_path), half_y=0.20, tag="into_legs")
    m, d, g = _skid_posed(into_legs, 0.6)
    assert sit60._touches_other_than_floor(d, g) and not sit60._touches_floor(m, d, g)
    others = {int(c.geom2 if c.geom1 == g else c.geom1) for c in (d.contact[i] for i in range(d.ncon))
              if g in (int(c.geom1), int(c.geom2))}
    assert others and 0 not in others
    assert all(m.geom(o).name.split("_")[1] in ("fore", "hind") for o in others)  # leg geoms only
    m, d, g = _skid_posed(into_legs, 0.26)
    assert sit60._touches_floor(m, d, g) and sit60._touches_other_than_floor(d, g)


def test_case_list_runs_both_contacts_over_the_same_starts():
    """Under impratio 10 the main stops of the adopted model and the skid copy run at the same standing and trot
    starts and seeds as under the model's contact (every trot trigger, not only the headline two); --quick keeps the
    headline starts and one seed."""
    main_b = {"sit_1.0", "sit_0.3", "sit_0.2", "collapse"}

    def starts_of(cases, variant, contact):
        out = set()
        for v, c, kind, seed, triggers, behaviours in cases:
            if v == variant and c == contact and kind in ("stand", "trot"):
                out |= {(kind, seed, t, b) for t in triggers for b in behaviours if b in main_b}
        return out

    full = sit60.case_list()
    for variant in ("adopted", "skid"):
        model, imp = starts_of(full, variant, "model"), starts_of(full, variant, "impratio10")
        assert imp == model
        assert {t for k, _, t, _ in imp if k == "trot"} == set(sit60.TROT_TRIGGERS_S)
        assert len(imp) == 4 * 2 * (1 + len(sit60.TROT_TRIGGERS_S))
    assert len(sit60.TROT_TRIGGERS_S) == 14 and set(sit60.HEADLINE_TROT) <= set(sit60.TROT_TRIGGERS_S)
    assert sum(len(c[4]) * len(c[5]) for c in full) == 888
    quick = sit60.case_list(quick=True)
    assert {c[1] for c in quick} == {"model"} and {c[3] for c in quick} == {1}
    assert all(set(c[4]) <= set(sit60.HEADLINE_TROT) for c in quick if c[2] == "trot")


def test_skid_top_is_the_belly_shell_bottom():
    m = mujoco.MjModel.from_xml_path(str(sit60.MODEL))
    g = m.geom("torso_contact").id
    # within 0.5 mm (-0.0752 m); geoms on one body never collide, so the top only sets the box's height
    assert m.geom_pos[g][2] - m.geom_size[g][2] == pytest.approx(sit60.SKID_TOP_Z, abs=5e-4)


def test_sit_geometry_is_fore_limited_and_reachable():
    geo = sit60.sit_geometry()
    r = geo["reach_depth_m"]
    assert r["left_fore"] == pytest.approx(r["right_fore"]) and r["left_hind"] == pytest.approx(r["right_hind"])
    assert r["left_fore"] > r["left_hind"]  # the forelegs fold less deep: they set the level body's depth
    assert geo["sit_depth_m"] == pytest.approx(r["left_fore"])
    for c, q in geo["q_sit_deg"].items():
        assert q[2] <= sit60.SIT_FLEX_MAX_DEG + 1e-6, c
    assert geo["q_sit_deg"]["left_fore"][2] == pytest.approx(sit60.SIT_FLEX_MAX_DEG, abs=1e-3)
    assert 0.03 < geo["stand_height_m"] - geo["sit_height_m"] < 0.08
    for c, tilt in geo["foot_tilt_at_sit_deg"].items():
        assert abs(tilt) < 22.0, c  # the passive ankle (+-22 deg) can lay the sole flat


def test_sit_targets_end_at_the_sit_depth():
    geo = sit60.sit_geometry()
    q0 = np.zeros(12)
    tg, err, p1 = sit60.sit_targets(sit60.MODEL, q0, geo["sit_depth_m"], 0.3, 0.001)
    assert tg.shape == (300, 12) and err < 1e-4
    for c, p in p1.items():
        assert p[2] == pytest.approx(-geo["sit_depth_m"])
    assert np.abs(np.diff(tg, axis=0)).max() < math.radians(1.0)  # smooth: under 1 deg per ms
    flex = tg[:, [2, 5, 8, 11]]
    assert flex.max() <= math.radians(sit60.SIT_FLEX_MAX_DEG) + 1e-9


def test_min_jerk_ends():
    assert sit60.min_jerk(0.0) == 0.0 and sit60.min_jerk(1.0) == 1.0 and sit60.min_jerk(0.5) == pytest.approx(0.5)


def test_regen_accounting_matches_the_thermal_models_copper_loss():
    """One X4 braking at 5 N m and 2 rad/s for 1 s: 10 W of braking less the copper loss the thermal model gives."""
    import gait_report as gr
    from telemetry.thermal import DriveThermal

    d = gr.drive_table()["pitch"]
    kt, r = d["kt_nm_per_a_rms"], d["phase_resistance_ohm"]
    therm = DriveThermal("x4", kt, r, 7.5)
    cu = therm.copper_loss(5.0, 25.0)
    assert cu == pytest.approx(3 * 0.2 * (5.0 / 1.8) ** 2)
    n = 1000
    tau = np.full((n, 1), 5.0)
    om = np.full((n, 1), -2.0)
    e = sit60.regen(tau, om, [1.0], [kt], [r], 0.001)
    assert e["bus_surplus_j"] == pytest.approx(10.0 - cu, rel=1e-9)
    assert e["gross_regen_j"] == pytest.approx(10.0 - cu, rel=1e-9)
    assert e["copper_j"] == pytest.approx(cu, rel=1e-9)
    assert e["mech_braking_j"] == pytest.approx(10.0, rel=1e-9)
    assert e["beyond_hotel_j"]["30"] == 0.0 and e["beyond_hotel_j"]["25"] == 0.0
    assert e["bus_largest_event_j"] == pytest.approx(10.0 - cu, rel=1e-9)
    assert e["bus_surplus_time_s"] == pytest.approx(1.0)


def test_regen_through_a_linkage_and_on_a_shared_bus():
    """A flexion drive braking through the 0.9 linkage returns 0.9 of the joint's power; a second drive motoring on
    the same bus takes part of it, so the shared surplus is below the gross regen."""
    eta, kt, r = [0.9, 1.0], [2.1, 1.8], [0.205, 0.2]
    tau = np.array([[10.0, 2.0]] * 1000)
    om = np.array([[-3.0, 1.0]] * 1000)  # drive 0 brakes 30 W at the joint; drive 1 motors 2 W
    e = sit60.regen(tau, om, eta, kt, r, 0.001)
    cu0 = 3 * 0.205 * (10.0 * 0.9 / 2.1) ** 2
    cu1 = 3 * 0.2 * (2.0 / 1.8) ** 2
    assert e["gross_regen_j_per_drive"][0] == pytest.approx(27.0 - cu0, abs=1e-3)
    assert e["gross_regen_j_per_drive"][1] == 0.0
    assert e["bus_surplus_j"] == pytest.approx(27.0 - cu0 - 2.0 - cu1, rel=1e-9)
    assert e["beyond_hotel_j"]["25"] == pytest.approx(max(0.0, 27.0 - cu0 - 2.0 - cu1 - 25.0), abs=1e-9)


def test_rest_time_takes_the_last_stretch():
    t = np.arange(1, 1001) * 0.001
    ok = np.ones(1000, dtype=bool)
    ok[:300] = False
    ok[400:450] = False  # moving again after a pause: the rest starts after this
    assert sit60.rest_time(ok, t, 0.001) == pytest.approx(0.450)
    assert sit60.rest_time(ok, t, 0.001, after_s=0.7) == pytest.approx(0.7)
    ok[-1] = False
    assert sit60.rest_time(ok, t, 0.001) is None
    ok[:] = True
    ok[:600] = False  # still at rest at the end, but for under 0.5 s
    assert sit60.rest_time(ok, t, 0.001) is None


def test_settling_measures_the_late_sink_and_the_soles_creep():
    """From 1 s after the reference: the further sink, the most-travelled loaded sole's slide, and the friction use
    in the samples where a loaded sole slides; unloaded or slow samples do not count; None when the window is short."""
    dt, n = 0.001, 4000
    t = np.arange(1, n + 1) * dt
    z = np.where(t < 2.5, 0.26, 0.26 - 0.01 * (t - 2.5))  # sinks 10 mm/s from 2.5 s on
    fn = np.full((n, 4), 30.0)
    ft = np.full((n, 4), 15.0)
    slip = np.zeros((n, 4))
    slip[t > 2.5, 0] = 0.010  # sole 0 creeps at 10 mm/s while loaded, at half its load
    slip[t > 2.5, 1] = 0.001  # sole 1 under the sliding threshold
    fn[:, 3] = 1.0  # sole 3 unloaded: its fast slip does not count
    slip[:, 3] = 0.5
    L = {"t": t, "z": z, "sole_fn": fn, "sole_ft": ft, "sole_slip": slip}
    s = sit60.settling(L, dt, 1.5, 0.85)
    assert s["sink_m"] == pytest.approx(0.26 - z[-1])
    assert s["sole_slide_mm"] == pytest.approx(10.0 * 1.5, rel=1e-2)  # 1.5 s at 10 mm/s
    assert s["sliding_samples"] == int((t > 2.5).sum())
    assert s["sliding_ratio"]["p50"] == pytest.approx(0.5) and s["sliding_ratio"]["max"] == pytest.approx(0.5)
    assert s["z_after_ref_m"]["1"] == pytest.approx(0.26) and s["z_after_ref_m"]["0.1"] == pytest.approx(0.26)
    assert s["z_after_ref_m"]["4"] is None and s["mu"] == 0.85
    short = sit60.settling({k: v[:2000] for k, v in L.items()}, dt, 1.5, 0.85)
    assert short["sink_m"] is None and short["sliding_ratio"] is None and short["z_after_ref_m"]["0.1"] is not None


def test_limit_kinds():
    assert sit60._limit_kind("left_fore_flexion") == "fore flexion"
    assert sit60._limit_kind("right_hind_pitch") == "hind pitch"
    assert sit60._limit_kind("left_hind_roll") == "roll" and sit60._limit_kind("right_fore_ankle") == "ankle"
    assert sit60._limit_kind("neck_pitch") is None


def test_events_and_bus_capacitance_and_back_emf():
    net = np.array([1.0, -2.0, -2.0, 3.0, -1.0, -1.0, -1.0, -1.0])
    assert sit60.events_j(net, 1.0) == pytest.approx(4.0)
    assert sit60.bus_energy_j_per_mf(50.0, 52.5) == pytest.approx(0.5e-3 * (52.5 ** 2 - 50.0 ** 2))
    assert sit60.back_emf_threshold_rpm("X4-36-L", 42.0) == pytest.approx(42.0 / 4.3 * 1000.0 / 36.0)


@pytest.fixture(scope="module")
def stand_snapshot():
    geo = sit60.sit_geometry()
    (snap,) = sit60.starts(sit60.MODEL, None, 1, "stand", (sit60.STAND_TRIGGER_S,))
    return geo, snap


def test_a_hard_sit_from_standing_lowers_and_a_collapse_draws_nothing(stand_snapshot):
    """Integration: the plant stands, the hard sit lowers the chassis by about the sit drop before the power cut,
    and a zero-torque collapse exchanges no electrical energy."""
    geo, snap = stand_snapshot
    r = sit60.run_stop(snap, "sit_0.3", geo["sit_depth_m"], sit60.MODEL, window_s=0.6)
    drop = geo["stand_height_m"] - geo["sit_height_m"]
    assert r["z0_m"] - r["z_at_cut_m"] == pytest.approx(drop, abs=0.008)
    assert r["sit_margin_before_cut_s"] >= 0.2
    assert r["electric"]["mech_braking_j"] > 0.0
    for g in sit60.GROUPS:
        assert r["drives"][g]["tau_peak_over_limit"] <= 1.0 + 1e-9
        assert r["drives"][g]["rpm_peak_over_allow"] == pytest.approx(
            r["drives"][g]["rpm_peak"] / r["drives"][g]["speed_allowance_rpm"])
    # the powered-phase ceiling: what the drives brake at the joints cannot exceed the kinetic energy at the stop
    # input plus the potential energy released before the cut. Standing, that is the whole robot's weight times its
    # centre of mass's fall, which is less than the chassis's: the feet stay on the floor
    assert 0.6 * r["body_weight_n"] * drop < r["pe_released_powered_j"] < r["body_weight_n"] * drop
    assert 0.0 <= r["ke_at_stop_input_j"] < 0.05
    e = r["electric"]
    net_take = e["bus_surplus_j"] + e["copper_j"] - e["bus_drawn_j"]  # the drives' braking less their own work
    assert 0.0 < net_take <= r["powered_ceiling_j"]
    assert e["bus_surplus_j"] <= r["powered_ceiling_j"] and e["bus_surplus_j"] < e["mech_braking_j"]
    assert r["settling"]["sink_m"] is None  # the 0.6 s window ends before 1 s after the cut
    assert r["skid_self_touched_before_trigger"] is False and r["skid_touched_before_trigger"] is False  # no skid
    c = sit60.run_stop(snap, "collapse", geo["sit_depth_m"], sit60.MODEL, window_s=1.3)
    assert c["electric"]["bus_surplus_j"] == 0.0 and c["electric"]["bus_drawn_j"] == 0.0
    sc = c["settling"]  # a collapse's reference is the stop input
    assert sc["t_ref_s"] == 0.0 and sc["mu"] == pytest.approx(0.85) and sc["sink_m"] is not None
    assert sc["z_after_ref_m"]["0.1"] > c["z_end_m"]
    # after 1.3 s of a collapse the folded legs rest on their hard stops: the elbows and knees on the +36 deg stop
    kinds = {sit60._limit_kind(j) for j in c["end_joints_on_limits_nm"]}
    assert {"fore flexion", "hind flexion"} <= kinds
    assert all(v > 0.0 for v in c["end_joints_on_limits_nm"].values())
    assert c["z_end_m"] < r["z0_m"] - 0.03
    assert c["pe_released_powered_j"] is None and c["powered_ceiling_j"] is None


def test_collapse_sensitivity_paths_and_the_summary(stand_snapshot):
    """The back-drive friction and shorted-phase damping slow a collapse; summarise and tables run on a small raw
    set and give the rows, the ceilings and the desk lines."""
    geo, snap = stand_snapshot
    runs = []
    for b in ("collapse", "collapse_friction", "collapse_shorted", "sit_0.3"):
        r = sit60.run_stop(snap, b, geo["sit_depth_m"], sit60.MODEL, window_s=0.6 if b == "sit_0.3" else 0.25)
        r.update({"variant": "adopted", "contact": "model", "start": "stand", "headline": True})
        runs.append(r)
    z = {r["behaviour"]: r["z_end_m"] for r in runs}
    assert z["collapse_shorted"] > z["collapse_friction"] > z["collapse"]  # each slows the fall over 0.25 s
    for r in runs[:3]:
        assert r["electric"]["bus_surplus_j"] == 0.0
    raw = {"meta": {"simulation_only": True}, "runs": runs}
    s = sit60.summarise(raw)
    assert set(s["groups"]) == {f"adopted|model|stand|{b}" for b in
                                ("collapse", "collapse_friction", "collapse_shorted", "sit_0.3")}
    g = s["groups"]["adopted|model|stand|sit_0.3"]
    assert g["n_runs"] == 1 and g["stats"]["powered_ceiling_j"]["max"] == pytest.approx(runs[3]["powered_ceiling_j"])
    v = s["desk"]["by_sit"]["sit_0.3"]
    assert v["max_bus_surplus_j"] == pytest.approx(runs[3]["electric"]["bus_surplus_j"])
    assert v["max_powered_ceiling_j"] >= v["max_drive_net_take_j"] > 0.0
    assert v["n_runs_net_take_over_ceiling"] == 0 and v["n_runs_bus_surplus_over_ceiling"] == 0
    assert v["max_braking_at_joints_j"] > v["max_bus_surplus_j"]
    pb = s["desk"]["power_board"]
    assert pb["resistor_seconds_at_rating_for_one_pulse_s"]["sit_0.3"] == pytest.approx(v["max_bus_surplus_j"] / 100.0)
    # the B7 envelope: the worst-energy run's own peak, and the largest peak over all runs, at least as large
    assert v["its_peak_w"] == pytest.approx(runs[3]["electric"]["bus_surplus_peak_w"]) and v["max_peak_w"] >= v["its_peak_w"]
    assert v["n_runs_braking_over_ceiling"] in (0, 1)
    gc = s["groups"]["adopted|model|stand|collapse"]
    assert set(gc["end_limit_runs"]) == set(sit60.END_LIMIT_KINDS) and gc["end_limit_max_nm"] >= 0.0
    assert "not given in the records" in s["desk"]["pack_charge_acceptance"]
    txt = sit60.tables(s)
    assert "## adopted, contact model" in txt and "| stand | collapse_shorted |" in txt
    assert "powered ceiling J" in txt and "Pack charge acceptance" in txt
    c = json.loads(json.dumps(sit60.compact(s)))  # the committed summary is plain JSON
    st = c["groups"]["adopted|model|stand|sit_0.3"]["stats"]
    assert "n" not in st["z_end_m"] and "n_none" not in st["z_end_m"]  # n equals n_runs, n_none is 0
    assert st["z_end_m"]["max"] == pytest.approx(runs[3]["z_end_m"], rel=1e-3)
    assert c["groups"]["adopted|model|stand|collapse"]["stats"]["powered_ceiling_j"] == {"n": 0, "n_none": 1}
    assert "stats_note" in c["meta"] and "stats_note" not in s["meta"]
