"""Tests for the packaging check of the adopted model (sim/check_packaging60.py). Needs mujoco 3.10.0, numpy and scipy
(the check's local search); skipped without mujoco or scipy. Run by hand:
    python -m pytest hardware/triceratops_build/sim/tests/test_check_packaging60.py
"""

import hashlib
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

SIM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIM))

mujoco = pytest.importorskip("mujoco")
pytest.importorskip("scipy")
import check_packaging60 as cp  # noqa: E402


@pytest.fixture(scope="module")
def coarse():
    before = hashlib.sha256(cp.MODEL.read_bytes()).hexdigest()
    res = cp.run(cp.MODEL, steps=3, trials=False, pair_steps=3)
    assert hashlib.sha256(cp.MODEL.read_bytes()).hexdigest() == before, "the check must not touch the model"
    return res


def pair(rows, a, b):
    hit = [p for p in rows if {p["a"], p["b"]} == {a, b}]
    assert len(hit) == 1, (a, b, len(hit))
    return hit[0]


def test_drive_housings_are_the_recorded_envelopes(coarse):
    assert coarse["envelope_mismatches"] == []


def test_hind_knee_housing_sits_on_its_strut_and_the_front_x6_clears(coarse):
    for side in ("left", "right"):
        p = pair(coarse["legs"][f"{side}_hind"], f"{side}_hind_upper_strut", f"{side}_hind_x4_flexion_stator")
        assert p["neutral_mm"] == pytest.approx(-38.70, abs=0.05) and p["class"] == "part"
        q = pair(coarse["legs"][f"{side}_fore"], f"{side}_fore_upper_strut", f"{side}_fore_x6_flexion_stator")
        assert q["neutral_mm"] == pytest.approx(8.70, abs=0.05)


def test_crossmembers_rails_and_tail_reproduce_the_records(coarse):
    bay = coarse["bay"]
    assert pair(bay, "pelvic_crossmember", "left_hind_x4_roll_stator")["min_mm"] == pytest.approx(-41.10, abs=0.05)
    assert pair(bay, "shoulder_crossmember", "left_fore_x4_roll_stator")["min_mm"] == pytest.approx(-1.10, abs=0.05)
    assert pair(bay, "6061_left_rail", "left_hind_x4_roll_stator")["class"] == "frame-drive"
    assert pair(bay, "nvme_and_vibration_tray", "pilz_relay_contactor_watchdog_radio_rx")["min_mm"] == pytest.approx(
        0.5, abs=0.01)
    t = pair(coarse["tail"], "pelvic_crossmember", "xl430_w250_tail_pitch_case")
    assert t["neutral_mm"] == pytest.approx(-3.90, abs=0.05)
    hp = pair(coarse["legs"]["left_hind"], "pelvic_crossmember", "left_hind_x4_pitch_stator")
    assert hp["neutral_mm"] == pytest.approx(-40.10, abs=0.05)


def test_linkage_is_a_parallelogram_with_the_reviews_angle(coarse):
    for c, pin in (("left_fore", 81.50), ("right_fore", 81.50), ("left_hind", 98.00), ("right_hind", 98.00)):
        lk = coarse["linkage"][c]
        assert lk["crank_radius_mm"] == pytest.approx(28.0, abs=0.01)
        assert lk["crank_ends_differ_mm"] < 0.01
        assert lk["rod_pin_to_pin_mm"] == pytest.approx(pin, abs=0.01)
        assert lk["rod_length_over_range_mm"][1] - lk["rod_length_over_range_mm"][0] < 1e-6
        assert lk["transmission_angle_min_deg"] == pytest.approx(43.0, abs=0.05)
        assert lk["at_flexion_deg"] == pytest.approx(-47.0, abs=0.1)
        assert lk["axial_gaps_mm"]["elbow_axle_to_output_crank"] == pytest.approx(0.75, abs=0.01)


def test_probe_rod_sits_on_the_models_rod_at_neutral():
    m = cp.with_probe_rods(cp.MODEL)
    d = mujoco.MjData(m)
    cp.set_pose(m, d, [], [])
    for c in cp.CORNERS:
        cp.place_rod(m, d, c, cp.linkage(m, c))
        probe, rod = m.geom(f"probe_rod_{c}").id, m.geom(f"{c}_pushrod").id
        assert np.allclose(d.geom_xpos[probe], d.geom_xpos[rod], atol=1e-9)
        assert abs(abs(float(np.dot(d.geom_xmat[probe][2::3], d.geom_xmat[rod][2::3]))) - 1.0) < 1e-9


def test_classify():
    assert cp.classify("left_fore_shoulder_yoke", "left_fore_x4_roll_stator")[0] == "joint"
    assert cp.classify("right_fore_shoulder_yoke", "left_fore_x4_roll_stator")[0] != "joint"
    assert cp.classify("pelvic_crossmember", "left_hind_x4_roll_stator")[0] == "frame-drive"
    assert cp.classify("left_hind_upper_strut", "left_hind_x4_flexion_stator")[0] == "part"
    assert cp.classify("left_hind_upper_limb_cover", "left_hind_x4_flexion_stator")[0] == "cover"
    assert cp.classify("PA11_rib_1_10", "left_hind_drive_crank")[0] == "shell"
    assert cp.classify("left_fore_sole", "left_fore_lower_strut")[0] == "foot shape"
    assert cp.classify("removable_service_tray", "gtech_6s_3300_1", 0.0)[0] == "touch"
    assert cp.classify("left_tof_mount_arm", "left_vl53l8cx")[0] == "joint"
    assert cp.classify("neck_cradle", "xc430_w240_neck_pitch_case")[0] == "joint"


def test_hind_linkage_against_the_rails(coarse):
    for c in ("left_hind", "right_hind"):
        r = coarse["hind_roll_limit_for_rails"][c]
        assert r["least_mm_at_roll_0"] == pytest.approx(0.45, abs=0.02)   # the crank lane 0.45 mm beside the rail
        assert r["least_mm_at_inward_stop"] < 0
        assert r["clear by 3 mm from roll (deg)"] is None


def test_bay_room_around_the_tightest_pairs(coarse):
    room = coarse["bay_room"]
    assert room["NVMe tray"]["+x"]["free_mm"] == pytest.approx(0.5, abs=0.01)
    assert room["NVMe tray"]["+x"]["meets"] == ["pilz_relay_contactor_watchdog_radio_rx"]
    assert max(v["free_mm"] for v in room["Pilz block"].values()) == pytest.approx(2.5, abs=0.01)   # boxed in
    assert room["pack 1 (+y)"]["-y"]["free_mm"] == pytest.approx(1.0, abs=0.01)
    assert room["Teensy carriers"]["+z"]["meets"] is None                                           # free upward
    assert room["IMU"]["+z"]["free_mm"] == pytest.approx(1.0, abs=0.01)


def _combined(prefix, change=None):
    label = next(k for k in cp.TRIALS if k.startswith(prefix))
    kind, edits = cp.TRIALS[label]
    edits = dict(edits, **(change or {}))
    before = hashlib.sha256(cp.MODEL.read_bytes()).hexdigest()
    out = cp.trial(cp.MODEL, edits, 3, kind)
    assert hashlib.sha256(cp.MODEL.read_bytes()).hexdigest() == before, "a trial must not touch the model"
    return out


def test_combined_layout_c_clears_every_solid_part_and_meets_the_rails():
    t = _combined("combined C")
    solid = [p for p in t["near_or_overlapping"] if p["class"] in ("part", "frame-drive")]
    assert solid and min(p["min_mm"] for p in solid) >= 0.5
    # the pairs C exists to clear are two moved parts against each other: measured, and clear
    for a, b in (("shoulder_crossmember", "dc_dc_rails_and_pack_monitor"),
                 ("pelvic_crossmember", "io_and_sensor_carriers")):
        p = pair(t["near_or_overlapping"], a, b)
        assert p["class"] == "part" and p["min_mm"] == pytest.approx(3.40, abs=0.05)
    # the tray stays, re-angled to the inset rails, still meet their rail and the tray, and clear the hind cranks
    for c, x in (("left", "-0.255_1"), ("right", "-0.255_-1"), ("left", "0.095_1"), ("right", "0.095_-1")):
        st = t["mounts_mm"][f"tray_stay_{x}"]
        assert st[f"6061_{c}_rail"] < 0 and st["removable_service_tray"] < 0, (x, st)
        assert pair(t["near_or_overlapping"], f"tray_stay_{x}", "pilz_relay_contactor_watchdog_radio_rx" if
                    x.startswith("0.095") else f"{c}_hind_drive_crank")["min_mm"] > 0.5
    sh, pv = t["mounts_mm"]["shoulder_crossmember"], t["mounts_mm"]["pelvic_crossmember"]
    assert sh["6061_left_rail"] < 0 and sh["6061_right_rail"] < 0 and sh["neck_cradle"] < 0
    assert pv["6061_left_rail"] < 0 and pv["6061_right_rail"] < 0
    assert sh["left_tof_mount_root"] > 0 and pv["tail pivot"] > 0          # left open: re-root, a tail bracket
    for c in cp.CORNERS:                                                     # every roll drive on its bracket
        b = t["mounts_mm"][f"{c}_drop_bracket"]
        assert all(v < 0 for v in b.values()), (c, b)


def test_a_trial_measures_its_moved_parts_against_each_other():
    """With the two bay blocks put back where they are in B but still in C's edits, the trial must find each block
    13.6 mm into its crossmember (both are moved parts)."""
    t = _combined("combined C", {"dc_dc_rails_and_pack_monitor": {"pos": "0.185 0 0"},
                                 "io_and_sensor_carriers": {"pos": "-0.19 0 -0.001"}})
    for a, b in (("shoulder_crossmember", "dc_dc_rails_and_pack_monitor"),
                 ("pelvic_crossmember", "io_and_sensor_carriers")):
        assert pair(t["near_or_overlapping"], a, b)["min_mm"] == pytest.approx(-13.60, abs=0.05)


def test_combined_layout_a_leaves_the_crossmembers_off_the_rails():
    t = _combined("combined A")
    assert t["mounts_mm"]["shoulder_crossmember"]["6061_left_rail"] > 20
    assert t["mounts_mm"]["pelvic_crossmember"]["6061_left_rail"] > 40
    hit = [p for p in t["near_or_overlapping"] if p["class"] == "part" and p["min_mm"] < 0]
    assert {frozenset((p["a"], p["b"])) for p in hit} == {frozenset(("pelvic_crossmember", "xl430_w250_tail_yaw_case"))}


def test_closed_form_capsule_distance_matches_mujoco():
    m = cp.with_probe_rods(cp.MODEL)
    d = mujoco.MjData(m)
    rng = np.random.default_rng(0)
    caps = [g for g in cp.solid_geoms(m) if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_CAPSULE and cp.corner(m.geom(g).name)]
    js = [m.joint(f"{c}_{a}").id for c in cp.CORNERS for a in cp.CHAIN]
    n = 0
    for _ in range(100):
        cp.set_pose(m, d, js, [rng.uniform(*m.jnt_range[j]) for j in js])
        for _ in range(20):
            g, h = rng.choice(caps, 2, replace=False)
            ref = mujoco.mj_geomDistance(m, d, g, h, cp.CAP_M, np.zeros(6))
            if m.geom_bodyid[g] == m.geom_bodyid[h] or ref >= cp.CAP_M:
                continue
            ends = []
            for x in (g, h):
                half, r = cp.bound_capsule(m, x)
                R = d.geom_xmat[x].reshape(3, 3)
                ends.append((d.geom_xpos[x] - R @ half, d.geom_xpos[x] + R @ half, r))
            got = cp.seg_dist(ends[0][0], ends[0][1], ends[1][0], ends[1][1]) - ends[0][2] - ends[1][2]
            assert got == pytest.approx(ref, abs=1e-9)
            n += 1
    assert n > 100


def _planted_clash(bodies, joints, cells, radius=0.004, below=0.0):
    """The model with a small ball on each of two leg bodies, placed to coincide at a pose in the middle of a grid cell
    in every joint they move with (``cells``: the cell's index per joint); there they overlap by 2 x radius."""
    m0 = cp.with_probe_rods(cp.MODEL)
    d0 = mujoco.MjData(m0)
    js = [m0.joint(j).id for j in joints]
    q = [lo + (k + 0.5) * (hi - lo) / (cp.PAIR_STEPS - 1) for (lo, hi), k in zip(m0.jnt_range[js], cells)]
    cp.set_pose(m0, d0, js, q)
    ids = [m0.body(b).id for b in bodies]
    P = 0.5 * (d0.xpos[ids[0]] + d0.xpos[ids[1]]) - np.array([0.0, 0.0, below])
    root = ET.parse(cp.MODEL).getroot()
    for b, f in zip(bodies, ids):
        loc = d0.xmat[f].reshape(3, 3).T @ (P - d0.xpos[f])
        ET.SubElement(root.find(f".//body[@name='{b}']"), "geom", {
            "name": f"{cp.corner(b)}_test_ball", "type": "sphere", "size": f"{radius}",
            "pos": " ".join(f"{v:.9f}" for v in loc), "contype": "0", "conaffinity": "0", "mass": "0", "group": "5"})
    return cp.with_probe_rods(cp.MODEL, root), dict(zip(joints, np.degrees(q)))


def _ball_pair(m):
    res = cp.legs_to_each_other(m, mujoco.MjData(m), cp.PAIR_STEPS, (("left_fore", "left_hind"),), only="test_ball")
    return pair(res["left_fore / left_hind"]["pairs_nearer_than_10mm"], "left_fore_test_ball", "left_hind_test_ball")


def test_legs_against_each_other_find_a_clash_between_grid_points():
    """Balls on the fore roll yoke and the hind upper link (3 joints; they meet at one pose) and on both upper links
    (4 joints; they meet along a curve), placed between grid points: no grid pose sees them overlap, the search finds
    the full 8 mm, the first where it was placed."""
    m, q = _planted_clash(("left_fore_roll_yoke", "left_hind_upper_link"),
                          ("left_fore_roll", "left_hind_roll", "left_hind_pitch"), (3, 4, 2), below=0.15)
    p = _ball_pair(m)
    assert p["grid_min_mm"] > 10 and p["grid_share_overlapping"] == 0.0     # the grid alone steps over it
    assert p["min_mm"] == pytest.approx(-8.0, abs=0.05)
    assert max(abs(p["at_deg"][k] - v) for k, v in q.items()) < 0.05
    m, q = _planted_clash(("left_fore_upper_link", "left_hind_upper_link"),
                          ("left_fore_roll", "left_fore_pitch", "left_hind_roll", "left_hind_pitch"), (3, 6, 4, 2),
                          below=0.15)
    p = _ball_pair(m)
    assert p["grid_min_mm"] > 10 and p["grid_share_overlapping"] == 0.0
    assert p["min_mm"] == pytest.approx(-8.0, abs=0.05)


def test_legs_against_each_other_find_the_deepest_overlap_between_grid_points_over_eight_joints():
    """Balls on the two feet, placed to meet in the middle of a grid cell in all eight joints: the grid sees part of
    the overlap, the search all of it."""
    m, _ = _planted_clash(("left_fore_foot", "left_hind_foot"),
                          [f"{c}_{x}" for c in ("left_fore", "left_hind") for x in cp.CHAIN], (3, 6, 0, 4, 3, 1, 0, 4))
    p = _ball_pair(m)
    assert p["grid_min_mm"] > -7.0
    assert p["min_mm"] == pytest.approx(-8.0, abs=0.05)


def test_legs_against_each_other_at_the_reviewed_poses():
    """The fore and hind soles on one side overlap fully at fore pitch +18, hind pitch -26, both flexed to -47 deg;
    the hind soles overlap at roll -18, pitch +22, knee -47 on both (ankles at 0); the search finds at least as much.
    The joints' grid bounds of the overlap are not a region where the soles always meet: at the bounds' near corner
    (fore pitch +9.5, hind -19, both flexed to -47, roll and ankle 0) they are clear, and only a small share of the
    poses inside the bounds overlap, with the two legs' rolls close."""
    m = cp.with_probe_rods(cp.MODEL)
    d = mujoco.MjData(m)

    def at(a, b, poses):
        cp.set_pose(m, d, [m.joint(k).id for k in poses], [math.radians(v) for v in poses.values()])
        return 1000 * mujoco.mj_geomDistance(m, d, m.geom(a).id, m.geom(b).id, cp.CAP_M, np.zeros(6))

    fh = at("left_fore_sole", "left_hind_sole", {"left_fore_pitch": 18, "left_fore_flexion": -47,
                                                 "left_hind_pitch": -26, "left_hind_flexion": -47})
    hh = at("left_hind_sole", "right_hind_sole", {f"{c}_{x}": v for c in ("left_hind", "right_hind")
                                                  for x, v in (("roll", -18), ("pitch", 22), ("flexion", -47))})
    assert fh == pytest.approx(-48.0, abs=0.01) and hh == pytest.approx(-6.34, abs=0.01)
    res = cp.legs_to_each_other(m, d, cp.PAIR_STEPS, (("left_fore", "left_hind"), ("left_hind", "right_hind")),
                                only="_sole")
    assert pair(res["left_fore / left_hind"]["pairs_nearer_than_10mm"], "left_fore_sole",
                "left_hind_sole")["min_mm"] <= fh + 1e-6
    p = pair(res["left_hind / right_hind"]["pairs_nearer_than_10mm"], "left_hind_sole", "right_hind_sole")
    assert p["min_mm"] <= hh and p["at_deg"]["left_hind_roll"] == pytest.approx(-18.0, abs=0.01)
    r = res["left_fore / left_hind"]["region_overlapping_on_the_grid"]["solid parts (no covers)"]
    b = r["joint_grid_values_taking_part_deg"]
    assert b["left_fore_pitch"][0] == pytest.approx(9.5) and b["left_hind_pitch"][1] == pytest.approx(-19.0)
    assert at("left_fore_sole", "left_hind_sole", {"left_fore_pitch": 9.5, "left_fore_flexion": -47,
                                                    "left_hind_pitch": -19, "left_hind_flexion": -47}) > 5.0
    assert r["share_of_grid_poses"] < r["share_of_the_poses_inside_these_bounds"] < 0.2
    assert r["largest_roll_difference_deg"] < 30.0            # of a 42 deg roll range


def test_legs_against_each_other_in_the_coarse_run(coarse):
    """Even from a 3-value grid the search reaches the struts' full crossing and the soles' full overlap."""
    fh = coarse["legs_to_each_other"]["left_fore / left_hind"]["least"]
    assert fh["solid parts (no covers)"]["min_mm"] == pytest.approx(-48.0, abs=0.05)
    assert fh["solid parts, soles aside"]["min_mm"] == pytest.approx(-19.2, abs=0.05)
    for k in ("left_fore / right_hind", "right_fore / left_hind"):            # diagonal legs never come near
        assert coarse["legs_to_each_other"][k]["least"]["any part"]["pair"] is None
