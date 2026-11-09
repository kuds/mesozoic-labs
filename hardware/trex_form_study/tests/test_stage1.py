"""Tests for the Stage 1 plant, drives and references (designlab/stage1; STAGE1_PLAN.md sections 2 and 3).

Run from hardware/trex_form_study:  python -m pytest tests/test_stage1.py -q
The default run is fast (shortened stand, one reference, short replays) and needs no git-ignored file: the MJCF
models rebuild from their records (``plant.model_files``) and the reference arrays a test replays are generated
into a temporary directory.  The committed run records (``stage1/runs/<run>/<body>/*.json``) are read as records.
The full-length gate checks (20 s stands on hard floor and low pile, every reference against its records, the full
nominal replay and its panel) take minutes and run only with STAGE1_SLOW=1 (the repository registers no slow marker).
"""

from __future__ import annotations

import inspect
import json
import os
import re
import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import designlab  # noqa: E402,F401  (selects the GL backend before mujoco renders)
from designlab import carpet, catalog, mjcf, statics  # noqa: E402
from designlab.stage1 import BODIES, control, drive, estimator, gates, legs, plant, refs, transitions  # noqa: E402
from designlab.stage1.__main__ import main as cli  # noqa: E402

BODY = "60-N-T-10"
RUN = ROOT / "stage1" / "runs" / "2026-11-08"
G4_RUN_STRIDES = gates.G4_STRIDES - 1e-3  # a gated run that walked its full length did not fall
slow = pytest.mark.skipif(not os.environ.get("STAGE1_SLOW"), reason="full-length gate check; set STAGE1_SLOW=1")


class _Selection:
    """``build.Design.act`` from a record's selection, for ``carpet.hold_servos``."""

    def __init__(self, sel: dict):
        self.sel = sel

    def act(self, role: str) -> catalog.Actuator:
        return catalog.actuator(self.sel["joints"][role] if role in self.sel["joints"] else self.sel[role])


@pytest.fixture(scope="module")
def hard():
    return plant.build(BODY, "hard")


@pytest.fixture(scope="module")
def nominal():
    src = refs.source(BODY)
    return src, refs.generate(src, "nominal")


@pytest.fixture(scope="module")
def nominal_npz(nominal, tmp_path_factory):
    """The nominal reference's arrays, written here as ``refs.prepare`` writes them (the run directory's ``refs/``
    is git-ignored, so a fresh checkout has none)."""
    src, g = nominal
    out = tmp_path_factory.mktemp("refs")
    entry = refs.write(src, g, out)
    assert entry["matches_records"]
    return out / Path(entry["file"]).name


@pytest.fixture(scope="module")
def floors():
    return {floor: plant.build(BODY, floor) for floor in plant.FLOORS}


def _screened(body: str = BODY):
    xml, _ = plant.model_files(body)
    return xml, mujoco.MjModel.from_xml_path(str(xml))


# ------------------------------------------------------------------------------------------------
# Hashes and inputs
# ------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("body", sorted(BODIES))
def test_each_model_is_the_one_its_record_sized(body):
    xml, _ = plant.model_files(body)
    v = plant.verify(body, xml)
    assert v["matches_record"] and v["matches_plan"]
    assert v["file_sha256"].startswith({"60-N-T-10": "993c5ac5", "60-N-H-10": "96ef84c5", "75-M-H-10": "0517e4d3"}[body])


def test_a_changed_model_is_refused(tmp_path):
    xml, _ = plant.model_files(BODY)
    bad = tmp_path / xml.name
    bad.write_text(xml.read_text() + "<!-- one byte more -->")
    with pytest.raises(plant.PlantError):
        plant.verify(BODY, bad)


def test_only_the_parallel_ankle_linkage_is_modelled(monkeypatch):
    rec = plant.record(BODY)
    assert rec["candidate"]["leg_layout"] == "tandem" and plant.leg_layout(BODY) == "tandem"
    serial = json.loads(json.dumps(rec))
    serial["candidate"]["leg_layout"] = "serial"
    monkeypatch.setattr(plant, "record", lambda body: serial)
    with pytest.raises(plant.PlantError, match="layout"):
        plant.build(BODY, "hard", settle=False)
    with pytest.raises(ValueError):
        drive.motor_matrix(["left_ankle_pitch", "left_ankle_roll"], "serial")


# ------------------------------------------------------------------------------------------------
# Drives: tendon ankle and torque-speed envelope
# ------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("body", sorted(BODIES))
def test_tendon_map_equals_sizing_motor_space(body):
    r = gates.tendon_map(body, samples=60, seed=3)
    assert r["matches"] and r["leg_layout"] == "tandem"
    assert max(r["max_err_joint_torque_Nm"], r["max_err_motor_speed_rad_s"], r["max_err_plant_matrix_torque_Nm"],
               r["max_err_plant_matrix_speed_rad_s"]) <= 1e-9
    for side in statics.SIDES:  # |tau_p| + |tau_r| reaches 2 peak on each axis and never exceeds it
        a = r["ankle_limit"][side]
        assert a["within"] and a["reaches"]
        assert a["max_abs_pitch_plus_abs_roll_Nm"] == pytest.approx(2 * a["peak_Nm"], abs=1e-9)


def test_envelope_derates_driving_torque_only():
    peak, w1, w0 = 7.0, 4.0, 6.0
    w = np.array([0.0, 2.0, 4.0, 5.0, 6.0, 9.0])
    expect = [7.0, 7.0, 7.0, 3.5, 0.0, 0.0]
    assert drive.torque_limit(w, peak, w1, w0) == pytest.approx(expect)
    lo, hi = drive.torque_limits(w, peak, w1, w0)
    assert hi == pytest.approx(expect) and lo == pytest.approx(np.full(6, -peak))  # forward: braking at peak
    lo, hi = drive.torque_limits(-w, peak, w1, w0)
    assert lo == pytest.approx([-v for v in expect])  # backward: the negative side drives
    assert hi[1:] == pytest.approx(np.full(5, peak)) and hi[0] == pytest.approx(expect[0])
    lo, hi = drive.torque_limits(w, peak, w1, w0, braking="derated")  # the symmetric rule, for sweeps
    assert hi == pytest.approx(expect) and lo == pytest.approx([-v for v in expect])
    with pytest.raises(ValueError):
        drive.torque_limits(w, peak, w1, w0, braking="none")


def test_envelope_follows_the_catalog_on_the_24_v_bus(hard):
    sel = plant.record(BODY)["selection"]
    assert sel["bus"] == 24.0 and hard.meta["options"]["bus_V"] == 24.0
    for d in hard.drives:
        a = catalog.actuator(d.part)
        assert d.peak == a.peak
        assert d.rated_speed == a.speed_at(None if a.box is not None else 24.0)
        nl = a.no_load_speed if a.box is not None else a.no_load_speed * min(1.0, 24.0 / a.nominal_voltage)
        assert d.no_load_speed == pytest.approx(nl, rel=1e-12)
        assert d.no_load_speed > d.rated_speed
    x6 = catalog.actuator("myactuator_rmd_x6_p20_60_e")  # 48 V drive on the 24 V bus: both speeds halve
    assert drive.no_load_speed_at(x6, 24.0) == pytest.approx(0.5 * x6.no_load_speed)


@pytest.mark.parametrize("bus", [25.2, 22.2, 21.0])
def test_pack_voltage_sets_the_leg_drive_envelopes(bus):
    p = plant.build(BODY, "hard", settle=False, bus=bus)
    assert p.meta["options"]["bus_V"] == bus and p.meta["options"]["bus_source"] == "argument"
    for d in p.drives:
        a = catalog.actuator(d.part)
        if a.box is not None:  # the neck and tail servos are not on the leg bus
            assert d.bus_v is None and d.rated_speed == a.rated_speed and d.no_load_speed == a.no_load_speed
            continue
        assert d.bus_v == bus
        assert d.rated_speed == pytest.approx(a.speed_at(bus))  # no credit above the drive's nominal voltage
        assert d.no_load_speed == pytest.approx(a.no_load_speed * min(1.0, bus / a.nominal_voltage))
    if bus == 22.2:  # plan section 5: the knee's rated speed at 22.2 V
        knee = next(d for d in p.drives if d.actuator == "left_knee_servo")
        assert knee.rated_speed == pytest.approx(8.04, abs=0.005)


def test_envelope_is_applied_from_each_drives_own_speed(hard):
    m, d, env = hard.m, hard.d, hard.envelope
    hard.reset()
    i_knee = hard.joints.index("left_knee")
    i_a = hard.joints.index("right_ankle_pitch")  # right ankle motor a turns at w_p + w_r
    i_r = hard.joints.index("right_ankle_roll")
    knee, ank = hard.drives[i_knee], hard.drives[i_a]
    d.qvel[hard.dofs[i_knee]] = 0.5 * (knee.rated_speed + knee.no_load_speed)
    d.qvel[hard.dofs[i_a]] = d.qvel[hard.dofs[i_r]] = -0.5 * (0.25 * ank.rated_speed + 0.75 * ank.no_load_speed)
    env.update()
    assert env.active
    lo, hi = m.actuator_forcerange[:, 0].copy(), m.actuator_forcerange[:, 1].copy()
    assert hi[i_knee] == pytest.approx(0.5 * knee.peak) and lo[i_knee] == pytest.approx(-knee.peak)  # forward knee
    assert lo[i_a] == pytest.approx(-0.25 * ank.peak) and hi[i_a] == pytest.approx(ank.peak)  # motor a backward
    assert hi[i_r] == pytest.approx(ank.peak) and lo[i_r] == pytest.approx(-ank.peak)  # motor b: w_p - w_r = 0
    others = np.setdiff1d(np.arange(m.nu), [i_knee, i_a])
    assert hi[others] == pytest.approx(env.peak[others]) and lo[others] == pytest.approx(-env.peak[others])
    hard.reset()
    env.update()  # back below every corner: the peaks are restored
    assert not env.active
    assert np.array_equal(m.actuator_forcerange[:, 1], env.peak)
    assert np.array_equal(m.actuator_forcerange[:, 0], -env.peak)


def test_envelope_is_rebound_on_reset():
    p = plant.build(BODY, "hard", settle=False)
    assert p.envelope.bound_to(p.m, p.d)
    p.d = mujoco.MjData(p.m)  # a replaced data: the envelope follows it on the next reset
    assert not p.envelope.bound_to(p.m, p.d)
    p.reset()
    assert p.envelope.bound_to(p.m, p.d)


# ------------------------------------------------------------------------------------------------
# Plant
# ------------------------------------------------------------------------------------------------
def _ankle_block(p, side: str, gains) -> np.ndarray:
    """``G^T diag(gains) G`` over one ankle's pitch and roll."""
    idx = [p.joints.index(f"{side}_{j}") for j in ("ankle_pitch", "ankle_roll")]
    Gs = p.G[np.ix_(idx, idx)]
    return Gs.T @ np.diag(np.asarray(gains)[idx]) @ Gs


def test_hold_gains_follow_carpet_hold_servos(hard):
    _, m0 = _screened()
    d0 = mujoco.MjData(m0)
    ref = carpet.hold_servos(m0, d0, m0.key_qpos[0].copy(), _Selection(plant.record(BODY)["selection"]))
    gains = hard.meta["gains"]["drives"]
    for jn, g in ref.items():
        if mjcf.joint_role(jn) in drive.REPLACES:
            continue
        assert gains[f"{jn}_servo"]["kp"] == pytest.approx(g["kp"], rel=1e-12)
        assert gains[f"{jn}_servo"]["kv"] == pytest.approx(g["kv"], rel=1e-12)
    # decision 11, the default: each ankle motor at half its hold servo's kp, one hold servo's stiffness per axis
    assert hard.meta["options"]["ankle_gains"] == "joint" == hard.meta["gains"]["ankle_gains"]
    assert hard.meta["gains"]["ankle_gains_source"] == "STAGE1_PLAN.md decision 11"
    js = hard.meta["gains"]["ankle_joint_space"]
    for side in statics.SIDES:
        ia, ib = (hard.joints.index(f"{side}_{j}") for j in ("ankle_pitch", "ankle_roll"))
        for i in (ia, ib):
            dr = hard.drives[i]
            assert hard.kp[i] == pytest.approx(0.5 * dr.sustained / carpet.SERVO_ERROR_RAD, rel=1e-12)
            assert hard.kv[i] == pytest.approx(2 * carpet.SERVO_ZETA * np.sqrt(hard.kp[i] * gains[dr.actuator]["inertia_kg_m2"]),
                                               rel=1e-12)
        hp, hr = ref[f"{side}_ankle_pitch"], ref[f"{side}_ankle_roll"]
        K, D = _ankle_block(hard, side, hard.kp), _ankle_block(hard, side, hard.kv)
        assert K == pytest.approx(np.diag([hp["kp"], hr["kp"]]), rel=1e-12)  # exactly the screened stiffness
        r = js[side]  # plant.json: the joint-space damping on pitch and roll against hold_servos'
        assert [r["stiffness_Nm_per_rad"][a] for a in ("pitch", "roll", "pitch_roll")] == pytest.approx(
            [K[0, 0], K[1, 1], K[0, 1]], rel=1e-12, abs=1e-12)
        assert [r["damping_Nms_per_rad"][a] for a in ("pitch", "roll", "pitch_roll")] == pytest.approx(
            [D[0, 0], D[1, 1], D[0, 1]], rel=1e-12)
        assert D[0, 0] == pytest.approx(D[1, 1], rel=1e-12)  # kv_a + kv_b on either axis
        assert r["hold_servos"]["pitch"]["kv"] == pytest.approx(hp["kv"], rel=1e-12)
        assert r["hold_servos"]["roll"]["kv"] == pytest.approx(hr["kv"], rel=1e-12)
        assert r["damping_over_hold_servos"]["pitch"] == pytest.approx(D[0, 0] / hp["kv"], rel=1e-12)
        assert r["damping_over_hold_servos"]["roll"] == pytest.approx(D[1, 1] / hr["kv"], rel=1e-12)
        assert r["stiffness_over_hold_servos"] == pytest.approx({"pitch": 1.0, "roll": 1.0}, rel=1e-12)
    # 60-N-T-10: 0.94 N m s/rad on pitch and roll with the linkage's armature (decision 12), against hold_servos'
    # 0.80 / 0.65 (decision 11 quotes 0.73: the value with the screened armature, test_ankle_armature_options)
    assert js["left"]["damping_Nms_per_rad"]["pitch"] == pytest.approx(0.943, abs=0.001)
    assert (js["left"]["hold_servos"]["pitch"]["kv"], js["left"]["hold_servos"]["roll"]["kv"]) == pytest.approx(
        (0.804, 0.649), abs=0.001)


def test_per_motor_ankle_gain_option_doubles_the_joint_space_stiffness():
    p = plant.build(BODY, "hard", ankle_gains="motor", settle=False)
    assert p.meta["gains"]["ankle_gains"] == "motor" and p.meta["options"]["ankle_gains"] == "motor"
    assert p.meta["gains"]["ankle_gains_source"].startswith("argument")
    for side in statics.SIDES:  # each ankle motor: its own sustained torque at 0.01 rad of its own coordinate
        ia, ib = (p.joints.index(f"{side}_{j}") for j in ("ankle_pitch", "ankle_roll"))
        for i in (ia, ib):
            assert p.kp[i] == pytest.approx(p.drives[i].sustained / carpet.SERVO_ERROR_RAD)
        assert _ankle_block(p, side, p.kp) == pytest.approx(2 * p.kp[ia] * np.eye(2))  # two motors share each axis
        assert p.meta["gains"]["ankle_joint_space"][side]["stiffness_over_hold_servos"] == pytest.approx(
            {"pitch": 2.0, "roll": 2.0}, rel=1e-12)
    with pytest.raises(ValueError):
        plant.build(BODY, "hard", ankle_gains="box", settle=False)


def test_ankle_armature_options():
    _, m0 = _screened()
    lk = plant.build(BODY, "hard", settle=False)  # linkage: each motor's armature on its tendon
    sc = plant.build(BODY, "hard", settle=False, ankle_armature="screened")
    A0 = plant.joint_armature(m0, lk.joints)
    A, As = plant.joint_armature(lk.m, lk.joints), plant.joint_armature(sc.m, sc.joints)
    ank = [i for i, jn in enumerate(lk.joints) if "ankle" in jn]
    other = [i for i in range(len(lk.joints)) if i not in ank]
    assert np.diag(A)[ank] == pytest.approx(2 * np.diag(A0)[ank], rel=1e-9)  # both motors turn for either axis
    assert A[np.ix_(other, other)] == pytest.approx(A0[np.ix_(other, other)], abs=1e-15)
    assert As == pytest.approx(A0, abs=1e-15)  # screened: the screened mass matrix
    assert np.diag(lk.armature_extra)[ank] == pytest.approx(np.diag(A0)[ank], rel=1e-9)
    assert not np.any(sc.armature_extra)
    feet = [statics.bid(m0, f"{s}_foot") for s in statics.SIDES]
    assert np.all(lk.m.body_invweight0[feet, 0] < 0.95 * m0.body_invweight0[feet, 0])  # 9 % lower on 60 cm
    assert sc.m.body_invweight0[feet] == pytest.approx(m0.body_invweight0[feet], rel=1e-12)
    eff = lk.meta["ankle"]["armature"]
    assert eff["choice"] == "linkage" and eff["source"] == "STAGE1_PLAN.md decision 12"  # the default
    assert sc.meta["ankle"]["armature"]["source"].startswith("argument")
    assert eff["feet"]["left"]["time_constant_contact_stiffness_ratio"] == pytest.approx(1.10, abs=0.01)
    # the ankles' joint-space damping with joint gains: 0.94 N m s/rad with this armature, 0.73 (the figure in
    # STAGE1_PLAN.md decision 11) with the screened one
    for p, want in ((lk, 0.943), (sc, 0.731)):
        assert p.meta["gains"]["ankle_joint_space"]["left"]["damping_Nms_per_rad"]["pitch"] == pytest.approx(want, abs=0.001)
    with pytest.raises(ValueError):
        plant.build(BODY, "hard", ankle_armature="none", settle=False)


def test_joint_ranges_are_the_usable_ranges_as_the_screens_read_them(hard):
    usable = plant.usable_ranges(BODY)
    for jn in hard.joints:
        j = mujoco.mj_name2id(hard.m, mujoco.mjtObj.mjOBJ_JOINT, jn)
        assert tuple(hard.m.jnt_range[j]) == pytest.approx(usable[jn], abs=1e-12)
    assert usable["left_knee"][1] == pytest.approx(1.0509)  # set by the same leg: kept
    # set by the two toes meeting in a sweep from home: not a limit while walking (the screens fall back to the MJCF)
    assert usable["left_hip_roll"] == pytest.approx((-0.45, 0.2512))
    assert usable["right_hip_roll"] == pytest.approx((-0.2512, 0.45))
    assert plant.usable_ranges(BODY, inter_leg="keep")["left_hip_roll"] == pytest.approx((-0.2261, 0.2512))
    dropped = hard.meta["joint_ranges"]["dropped_inter_leg_limits"]
    assert set(dropped) == {"left_hip_roll", "right_hip_roll"}
    assert dropped["left_hip_roll"][0]["limited_by"] == ["hw_left_toe_2", "hw_right_toe_1"]


def test_command_sets_servo_targets_in_motor_space(hard):
    m, d = hard.m, hard.d
    rng = np.random.default_rng(7)
    q = hard.q_home[hard.qadr] + rng.uniform(-0.05, 0.05, len(hard.joints))
    qd = rng.uniform(-1.0, 1.0, len(hard.joints))
    tau = rng.uniform(-0.3, 0.3, len(hard.joints))
    qpos = hard.q_home.copy()
    qpos[hard.qadr] = q
    qvel = np.zeros(m.nv)
    qvel[hard.dofs] = qd
    hard.set_state(qpos, qvel)
    hard.command(q, qd, tau)
    mujoco.mj_forward(m, d)
    # on the target, the servos produce exactly the feedforward torque in joint space
    assert d.qfrc_actuator[hard.dofs] == pytest.approx(tau, abs=1e-9)
    assert hard.ctrl_home == pytest.approx(hard.G @ hard.q_home[hard.qadr])


def test_sensors_at_rest(hard):
    hard.reset()
    hard.hold()
    hard.step(249)
    m, d = hard.m, hard.d
    q_prev, qd_prev = d.qpos[hard.qadr].copy(), d.qvel[hard.dofs].copy()
    hard.step()
    s = hard.sensors()
    weight = float(m.body_subtreemass[hard.trunk]) * carpet.G
    assert s["foot_load"].sum() == pytest.approx(weight, rel=0.01)
    assert np.all(s["foot_contacts"] > 0)
    R = d.xmat[hard.trunk].reshape(3, 3)
    assert s["accel"] == pytest.approx(R.T @ np.array([0.0, 0.0, carpet.G]), abs=0.05)  # specific force at rest
    assert np.abs(s["gyro"]).max() < 1e-2
    assert s["quat"] == pytest.approx(d.xquat[hard.trunk], abs=1e-6)
    assert np.array_equal(s["q"], q_prev) and np.array_equal(s["qd"], qd_prev)  # read at the last step's start
    imu = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, plant.IMU_SITE)
    assert m.site_pos[imu].tolist() == pytest.approx(hard.meta["imu_site_trunk_m"])


def test_reset_seed_gives_each_episode_its_own_generator():
    p = plant.build(BODY, "hard", seed=5)
    starts = []
    for seed in (11, 12, 11):
        p.reset(joint_noise=0.01, root_velocity_noise=0.02, seed=seed)
        p.rng.normal()  # draws within an episode do not leak into the next
        starts.append(np.r_[p.d.qpos, p.d.qvel])
    assert np.array_equal(starts[0], starts[2]) and not np.array_equal(starts[0], starts[1])


def test_falls_include_a_shell_proxy_below_the_floor(hard):
    hard.reset()
    assert hard.fall_cause() is None and not hard.fallen()
    lowest = min(carpet.lowest_point(hard.m, hard.d, g)[2] for g in hard._cl)
    q = hard.q_settled.copy()
    q[2] -= lowest + 0.002  # a calf fairing proxy 2 mm into the floor, the trunk upright and above 0.7 x home
    assert q[2] > plant.FALL_HEIGHT_FRAC * hard.z_home
    hard.set_state(q, np.zeros(hard.m.nv))
    assert hard.fall_cause().startswith("proxy cl_")
    hard.reset()


def test_toe_lowest_point_is_exact(hard, floors):
    toes = [hard._gname[g] for g in hard._toes]
    assert len(toes) == 12 and all(("_toe_" in t or "_claw_" in t) and "pad" not in t for t in toes)
    assert toes == plant.toe_geoms(_screened()[1])  # the screened model's toes and claws
    for floor, p in floors.items():  # the same geoms on every floor
        assert [p._gname[g] for g in p._toes] == toes, floor
    pads = [hard._gname[g] for side in statics.SIDES for g in carpet.pad_geoms(hard.m)[side]]
    pad_toes = {n for n in pads if "pad_toe" in n}
    assert len(pad_toes) == 6
    low = floors["low"]  # pile_model takes the pads off the floor: their contype no longer tells them from toes
    by_name = {low._gname[g]: g for g in range(low.m.ngeom) if low._gname[g]}
    assert all(low.m.geom_contype[by_name[n]] == 0 for n in pads)
    assert set(plant.toe_geoms(low.m)) - set(toes) == pad_toes  # what the rule alone would pick on the pile model
    rng = np.random.default_rng(2)
    for p in (hard, low):
        for _ in range(20):
            q = p.q_home.copy()
            q[p.qadr] += rng.uniform(-0.4, 0.4, len(p.joints))
            p.set_state(q, np.zeros(p.m.nv))
            z, g = p.toe_lowest()
            assert z == pytest.approx(min(carpet.lowest_point(p.m, p.d, t)[2] for t in p._toes) - p.z_floor, abs=1e-12)
            assert g in toes
        p.reset()


def test_75cm_body_settles_into_contact():
    p = plant.build("75-M-H-10", "hard")
    s = p.meta["settle"]
    assert s["foot_contacts_at_keyframe"] == [0, 0]  # the keyframe pads sit 0.3 um above the floor
    assert s["settled"] and min(s["foot_contacts_settled"]) > 0
    assert sum(s["foot_load_N"]) == pytest.approx(s["weight_N"], rel=0.01)


def test_floors_are_carpet_pys(floors):
    xml, m0 = _screened()
    variants = carpet.floor_variants(m0)
    rf = floors["robot_floor_carpet"]
    want = variants["robot_floor_carpet"]
    assert rf.meta["floor_params"] == want  # robot_floor's carpet exactly
    fl = statics.gid(rf.m, "floor")
    assert rf.m.geom_solref[fl] == pytest.approx(want["solref"]) and rf.m.geom_solimp[fl] == pytest.approx(want["solimp"])
    assert rf.m.geom_friction[fl] == pytest.approx(want["friction"]) and rf.m.geom_condim[fl] == want["condim"]
    assert rf.m.geom_priority[fl] > max(rf.m.geom_priority[g] for g in range(rf.m.ngeom) if g != fl)
    for pile in carpet.PILES:
        got, ref = floors[pile].meta["floor_params"], variants[pile]
        for key in ("k_A_N_per_m3", "depth_m", "height_m", "grid_m", "condim", "dampratio", "timestep",
                    "pad_stiffness_per_foot_N_per_m"):
            assert got[key] == pytest.approx(ref[key], rel=1e-12), (pile, key)
        assert got["solimp"] == pytest.approx(list(ref["solimp"])) and got["friction"] == pytest.approx(list(ref["friction"]))
        assert got["full_cell_probe"]["K_N_per_m"] == pytest.approx(ref["full_cell_probe"]["K_N_per_m"], rel=1e-12)
        m_ref, info = carpet.pile_model(xml, m0.key_qpos[0].copy(), ref)  # the same probes under the same feet
        assert floors[pile].meta["floor_info"]["probes_per_foot"] == info["probes_per_foot"]
        _pile_probe_ratios(floors[pile], m_ref, got, ref)


def _probe_contacts(m, qpos) -> dict:
    """(side, probe position) -> (stiffness k imp / R, damping b / R, regularisation 1 / R) of every probe contact at
    ``qpos``, from MuJoCo's own efc_KBIP and efc_R."""
    d = mujoco.MjData(m)
    d.qpos[:] = qpos
    mujoco.mj_forward(m, d)
    left = statics.bid(m, "left_foot")
    out = {}
    for i in range(d.ncon):
        c = d.contact[i]
        g = int(c.geom[1] if m.geom_contype[c.geom[0]] != carpet.PROBE_BIT else c.geom[0])
        if m.geom_contype[g] != carpet.PROBE_BIT:
            continue
        k, b, imp, _ = d.efc_KBIP[c.efc_address]
        R = d.efc_R[c.efc_address]
        side = "left" if m.geom_bodyid[g] == left else "right"
        out[(side, tuple(np.round(m.geom_pos[g], 12)))] = np.array([k * imp / R, b / R, 1.0 / R])
    return out


def _pile_probe_ratios(p, m_ref, got: dict, ref: dict) -> None:
    """The pile probes' ratios recorded in ``armature_effects`` (plant over screened model) against carpet.py's own
    derivation (the full-cell probe of ``pile_floor``, from the left foot's inverse weight) and against MuJoCo's
    contact parameters of every probe in contact, plant pile model against carpet.py's on the screened model."""
    feet = p.meta["ankle"]["armature"]["feet"]
    left = feet["left"]
    assert left["pile_probe_stiffness_ratio"] == pytest.approx(
        got["full_cell_probe"]["K_N_per_m"] / ref["full_cell_probe"]["K_N_per_m"], rel=1e-12)  # stiffness kept
    assert left["pile_probe_damping_ratio"] == pytest.approx(
        got["full_cell_probe"]["damping_N_s_per_m"] / ref["full_cell_probe"]["damping_N_s_per_m"], rel=1e-12)
    assert left["pile_probe_regularisation_ratio"] == pytest.approx(
        got["regularisation_inertia_kg_per_contact"] / ref["regularisation_inertia_kg_per_contact"], rel=1e-12)
    for f in feet.values():  # damping as 1 / sqrt(invweight0), regularisation as 1 / invweight0 (60 cm: 5 % and 10 %)
        assert f["pile_probe_stiffness_ratio"] == pytest.approx(1.0, abs=1e-4)
        assert f["pile_probe_damping_ratio"] == pytest.approx(np.sqrt(f["time_constant_contact_stiffness_ratio"]), rel=1e-4)
        assert f["pile_probe_regularisation_ratio"] == f["time_constant_contact_stiffness_ratio"]
        assert f["pile_probe_damping_ratio"] == pytest.approx(1.049, abs=0.001)
    q = p.q_home.copy()
    q[2] -= 0.002  # 2 mm into the pile
    ours, theirs = _probe_contacts(p.m, q), _probe_contacts(m_ref, q)
    for side in statics.SIDES:
        keys = [k for k in ours if k[0] == side and k in theirs]
        assert len(keys) > 100
        r = np.array([ours[k] / theirs[k] for k in keys])
        f = feet[side]
        assert r == pytest.approx(np.tile([f["pile_probe_stiffness_ratio"], f["pile_probe_damping_ratio"],
                                           f["pile_probe_regularisation_ratio"]], (len(keys), 1)), rel=1e-9)


def test_settle_tells_which_floors_the_stance_stands_on(floors):
    status = {f: p.meta["settle"]["status"] for f, p in floors.items()}
    assert status == {"hard": "settled", "robot_floor_carpet": "settled", "low": "settled", "medium": "tipped",
                      "high": "tipped"}  # carpet.py's standing check: the 60 cm stances tip on medium and high pile
    for f, p in floors.items():
        s = p.meta["settle"]
        assert p.settled == (f in ("hard", "robot_floor_carpet", "low"))
        if p.settled:
            assert s["tilt_change_last_s_deg"] < carpet.SETTLED_TILT_DEG and s["com_move_last_s_mm"] < carpet.SETTLED_COM_MM
            assert s["seconds"] >= plant.SETTLE_S
    with pytest.raises(plant.PlantError, match="does not stand"):
        floors["medium"].reset()
    floors["medium"].reset(allow_unsettled=True)  # asked for explicitly


def test_seeded_runs_repeat_exactly():
    r = gates.repeatability(BODY, seed=5)
    assert r["repeats"] and r["other_seed_differs"] and r["reseeding_repeats"]


def test_zero_command_stand_on_hard_floor_shortened():
    r = gates.stand(BODY, "hard", seconds=3.0)
    assert r["stands"] and r["fell_at_s"] is None and r["fall_cause"] is None
    assert r["tilt_from_home_max_deg"] < 1.0


# ------------------------------------------------------------------------------------------------
# References
# ------------------------------------------------------------------------------------------------
def test_nominal_reference_matches_the_record(nominal):
    src, g = nominal
    assert round(g["summary"]["gait"]["speed"], 6) == 0.218266
    assert round(g["summary"]["min_cop_margin_single"], 6) == 0.030316
    checks = refs.record_checks(src, g)
    assert checks and all(c["match"] for c in checks)
    assert checks[0]["leaves"] > 80 and checks[0]["n_noise_floor_matches"] == 0  # bit-identical here


def test_record_comparison_catches_a_difference(nominal):
    src, g = nominal
    rec = plant.record(BODY)["gaits"]["nominal"]
    s = dict(g["summary"], min_cop_margin_single=g["summary"]["min_cop_margin_single"] * (1 + 2e-6))
    r = refs.compare(s, rec)
    assert not r["match"] and r["mismatched"] == ["min_cop_margin_single"]
    assert refs.compare(dict(g["summary"], min_cop_margin_single=g["summary"]["min_cop_margin_single"] * (1 + 1e-7)),
                        rec)["match"]


def test_noise_level_leaves_match_within_their_floor(nominal):
    _, g = nominal
    rec = plant.record(BODY)["gaits"]["nominal"]
    assert rec["joints"]["left_hip_yaw"]["speed_peak"] < 1e-6  # IK noise of a straight walk, not a speed
    s = json.loads(json.dumps(refs._plain(g["summary"])))
    s["joints"]["left_hip_yaw"]["speed_peak"] *= 1.5  # another BLAS: different noise
    s["joints"]["left_hip_yaw"]["pos_power_mean"] *= 0.5
    r = refs.compare(s, rec)
    assert r["match"] and r["n_noise_floor_matches"] == 2 and r["n_noise_level_leaves"] >= 6
    s["joints"]["left_knee"]["speed_peak"] *= 1 + 1e-5  # a physical leaf keeps its sixth digit
    assert not refs.compare(s, rec)["match"]
    assert refs.noise_floor("joints.left_knee.speed_peak") == 1e-6 and refs.noise_floor("gait.duty") is None
    assert refs.noise_floor("gait.zmp_error_history_m[3]") == 1e-7


def test_reference_file_round_trip(nominal, tmp_path):
    src, g = nominal
    entry = refs.write(src, g, tmp_path)
    ref = refs.Reference.load(tmp_path / f"{g['name']}.npz")
    assert entry["frames"] == ref.n == 160
    assert ref.period == pytest.approx(g["traj"]["info"]["period"])
    assert np.array_equal(ref.tau, g["idr"]["tau"]) and list(ref.joints) == list(src["M"].names)
    p = plant.build(BODY, "hard", settle=False)
    series = ref.joint_series(p.qadr, p.dofs)
    q0, _, tau0 = ref.at(0.0, series)
    assert q0 == pytest.approx(ref.q[0, p.qadr]) and tau0 == pytest.approx(ref.tau[0])
    assert ref.at(ref.period, series)[0] == pytest.approx(q0)  # periodic in the joints
    mid = ref.at(0.5 * ref.step, series)[0]
    assert mid == pytest.approx(0.5 * (ref.q[0, p.qadr] + ref.q[1, p.qadr]))
    full = series + (ref.qdd_series(p.dofs),)
    stacked = np.hstack(full)
    for t in (0.0, 0.3 * ref.step, 7.6 * ref.step, 1.7 * ref.period):
        assert ref.interp(t, stacked) == pytest.approx(np.concatenate(ref.at(t, full)), abs=1e-12)


# ------------------------------------------------------------------------------------------------
# Replay diagnostics
# ------------------------------------------------------------------------------------------------
def test_replay_feedforward_carries_the_plants_extra_armature(nominal_npz):
    ref = refs.Reference.load(nominal_npz)
    lk = gates.replay(plant.build(BODY, "hard", settle=False), ref, strides=1.0)
    sc = gates.replay(plant.build(BODY, "hard", settle=False, ankle_armature="screened"), ref, strides=1.0)
    off = gates.replay(plant.build(BODY, "hard", settle=False), ref, strides=1.0, armature_ff=False)
    assert lk["armature_feedforward"] and not sc["armature_feedforward"] and not off["armature_feedforward"]
    for r in (lk, sc, off):
        assert not r["fell"] and r["strides"] == pytest.approx(1.0, abs=1e-3)
        assert not r["joint_limit_reached"] and r["range_margin_worst"][2] > 0.1
        assert r["toe_lowest_above_floor_mm"] > -20.0 and r["envelope_clamps"] == {}


def test_leg_gap_on_the_reference_walk(nominal_npz):
    lg = legs.LegGap(BODY)
    ref = refs.Reference.load(nominal_npz)
    r = lg.trace(ref.q[::4])
    assert r["interference_free"] and r["poses"] == 40
    assert r["min_gap_all_pair"] == ["hw_left_hip_bracket", "hw_right_hip_bracket"]  # 4 mm apart at any pose
    assert r["min_gap_all_mm"] == pytest.approx(4.0, abs=0.05)
    assert any(h["pair"] == ["hw_left_hip_bracket", "hw_right_hip_bracket"] for h in r["home_pairs_within_cap"])
    assert r["min_gap_moving_mm"] > 0.0


def test_panel_offsets_are_seeded_and_bounded():
    a, b = gates._offsets(), gates._offsets()
    assert len(a) == gates.PANEL_RUNS and all(np.array_equal(x, y) for x, y in zip(a, b, strict=True))
    assert max(np.abs(x).max() for x in a) <= gates.PANEL_OFFSET_M_S and all(x[2] == 0.0 for x in a)
    s = gates.panel_stats([3.0, 12.0, 60.0])
    assert s["median"] == 12.0 and s["median_at_least_min_strides"] and s["runs_at_least_min_strides"] == 2
    with pytest.raises(ValueError):
        gates.g2(BODY, RUN / BODY, rule="vote")


def _choices(strides: dict) -> dict:
    """``g2``'s plant matrix from (unperturbed run, panel) strides per plant choice."""
    out = {}
    for name, (single, runs) in strides.items():
        pnl = gates.panel_stats(runs)
        out[name] = dict(options=dict(gates.PLANT_CHOICES[name], bus_V=24.0), nominal_strides=single,
                         nominal_fall_cause=None if single >= gates.REPLAY_STRIDES else "tilt", panel=pnl,
                         single_rule_pass=single >= gates.MIN_REPLAY_STRIDES,
                         panel_rule_pass=pnl["median_at_least_min_strides"])
    return out


def test_g2_records_the_build_decisions_with_the_plant_matrix_as_evidence():
    # the defaults are STAGE1_PLAN.md's decisions 11-13
    assert {n: d["taken"] for n, d in gates.DECISIONS.items()} == {11: "joint", 12: "linkage", 13: "panel"}
    sig = inspect.signature(plant.build).parameters
    assert sig["ankle_gains"].default == "joint" and sig["ankle_armature"].default == "linkage"
    assert inspect.signature(gates.g2).parameters["rule"].default == "panel"
    strides = {"motor_gains+linkage_armature": (4.3, [3.0] * 15 + [12.0] * 5),
               "joint_gains+linkage_armature": (60.0, [8.0] * 7 + [33.0] * 13),
               "motor_gains+screened_armature": (8.2, [4.0] * 20),
               "joint_gains+screened_armature": (2.3, [6.0] * 13 + [11.0] * 7)}
    choices = _choices(strides)
    dec = gates.decided(choices, "joint_gains+linkage_armature", "panel")
    assert [d["decision"] for d in dec["decisions"]] == [11, 12, 13] and dec["follows_all"]
    for d in dec["decisions"]:
        assert d["source"].startswith("STAGE1_PLAN.md") and d["source"].endswith(f"decision {d['decision']}")
        assert d["used"] == d["taken"] and d["follows"] and d["reason"]
    assert "decisions 11-13" in dec["source"]
    mat = dec["plant_matrix"]  # the 2 x 2 plant matrix and its panels stay as evidence
    assert set(mat) == set(gates.PLANT_CHOICES) and [n for n, r in mat.items() if r["used_here"]] == [
        "joint_gains+linkage_armature"]
    for name, (single, runs) in strides.items():
        assert mat[name]["unperturbed_strides"] == single and mat[name]["panel_median_strides"] == np.median(runs)
        assert mat[name]["panel_runs_at_least_min_strides"] == sum(r >= 10 for r in runs)
    ev = dec["rule_evidence"]
    assert ev["used"] == "panel" and ev["unperturbed_strides"] == 60.0 and ev["panel_median_strides"] == 33.0
    off = gates.decided(choices, "motor_gains+linkage_armature", "single")  # a comparison run says it deviates
    assert not off["follows_all"] and [d["follows"] for d in off["decisions"]] == [False, True, False]
    assert [d["used"] for d in off["decisions"]] == ["motor", "linkage", "single"]


def test_step_rate_judges_the_median_of_every_sample():
    s = gates.rate_summary([30e3, 10e3, 25e3], [[24e3, 26e3, 5e3], [21e3, 22e3, 23e3]], steps=500)
    assert s["single_process_samples"] == [30e3, 10e3, 25e3]  # every sample recorded
    assert s["per_process_samples"] == [[24e3, 26e3, 5e3], [21e3, 22e3, 23e3]]
    assert s["single_process"] == 25e3 and s["per_process_median"] == [24e3, 22e3]
    assert s["per_process_all_median"] == 22.5e3 and s["per_core"] == 22.5e3 and s["ok"]  # one slow sample is noise
    assert s["lowest_sample"] == 5e3 and s["samples_per_process"] == 3 and s["steps_per_sample"] == 500
    slow_alone = gates.rate_summary([19e3, 19.5e3, 30e3], [[24e3, 26e3, 25e3]])
    assert slow_alone["per_core"] == 19.5e3 and not slow_alone["ok"]  # a slow median fails
    assert gates.RATE_SAMPLES >= 5 and gates.RATE_STEPS >= 50_000
    r = gates._rate(BODY, steps=200, samples=3)  # in this process (no pool): three timed samples
    assert len(r) == 3 and all(v > 0 for v in r)


@pytest.mark.parametrize("body", sorted(BODIES))
def test_gate_records_follow_the_build_decisions(body):
    """The committed run records (``stage1/runs/2026-11-08``) were made with the decided plant and G2 rule."""
    doc = json.loads((RUN / body / "gates.json").read_text())
    g1, g2 = doc["G1"], doc["G2"]
    assert "open_decisions" not in g2 and g2["decided"]["follows_all"] and doc["same_plant_options"]
    assert g2["rule"] == "panel" and len(g2["nominal_panel_with_envelope"]["strides"]) == gates.PANEL_RUNS
    assert (g1["plant_options"]["ankle_gains"], g1["plant_options"]["ankle_armature"]) == ("joint", "linkage")
    assert set(g2["decided"]["plant_matrix"]) == set(gates.PLANT_CHOICES)
    assert g2["pass"] == (g2["all_match"] and g2["panel_pass"])
    sr = g1["step_rate"]
    assert len(sr["single_process_samples"]) == gates.RATE_SAMPLES and sr["steps_per_sample"] == gates.RATE_STEPS
    assert len(sr["per_process_samples"]) == sr["processes"]
    assert all(len(s) == gates.RATE_SAMPLES for s in sr["per_process_samples"])
    assert sr["ok"] == (min(np.median(sr["single_process_samples"]), np.median(sr["per_process_samples"]))
                        >= gates.MIN_STEPS_PER_S)
    pj = json.loads((RUN / body / "plant.json").read_text())
    assert pj["options"] == g1["plant_options"] and pj["gains"]["ankle_gains_source"] == "STAGE1_PLAN.md decision 11"
    assert set(pj["gains"]["ankle_joint_space"]) >= set(statics.SIDES)
    assert all("pile_probe_damping_ratio" in f for f in pj["ankle"]["armature"]["feet"].values())


# ------------------------------------------------------------------------------------------------
# Transitions (plan section 3 step 4, G3)
# ------------------------------------------------------------------------------------------------
def _plan(src: dict, name: str):
    a, b = (transitions.gait_of(BODY, n, src["H"]) for n in transitions.TRANSITIONS[name])
    geo = transitions.Geometry.of(src["M"], src["q_home"], (b if a.stand else a).crouch)
    return transitions.plan(name, a, b, geo), geo


def test_transition_plans_are_continuous(nominal):
    src, _ = nominal
    for name in transitions.TRANSITIONS:  # every plan (no IK: fast)
        pl, _ = _plan(src, name)
        dt = np.diff(pl.t)
        assert np.all(dt > 0) and len(pl.phase) == pl.t.size
        assert np.diff(pl.phase) == pytest.approx(np.full(len(pl.phase) - 1, 1 / transitions.N_FRAMES))
        # the feet, the ZMP target, the heading, the height, the neck and the tail move without jumps: no faster than
        # in the periodic walks' own strides (the first and last output strides; the stand for a start or a stop)
        n = transitions.N_FRAMES
        walk = [sl for sl, g in ((slice(pl.out.start, pl.out.start + n), pl.a), (slice(pl.out.stop - n, pl.out.stop),
                                                                                  pl.b)) if not g.stand]
        for arr in (pl.foot_pos, pl.zmp):
            speed = np.linalg.norm(np.diff(arr, axis=0), axis=-1) / (dt[:, None] if arr.ndim == 3 else dt)
            assert speed.max() <= 1.05 * max(speed[sl].max() for sl in walk), name
        assert np.abs(np.diff(pl.heading) / dt).max() <= max(abs(pl.a.yaw_rate), abs(pl.b.yaw_rate)) + 1e-6, name
        assert np.abs(np.diff(pl.root_z) / dt).max() < 0.05 and np.abs(np.diff(pl.tail) / dt).max() < 2.0, name
        z0 = transitions.Geometry.of(src["M"], src["q_home"], 0.0).feet0
        for side, lo, td, lift, f0, f1 in pl.swings:  # each swing starts on one footprint and lands on the next
            j = statics.SIDES.index(side)
            for p_, f in ((lo, f0), (td, f1)):  # at the frames that fall on the events (a ramp's may fall between)
                k = int(round((p_ - pl.phase[0]) * transitions.N_FRAMES))
                if 0 <= k < len(pl.phase) and abs(pl.phase[k] - p_) < 1e-9:
                    assert pl.foot_pos[k, j, :2] == pytest.approx(f[0], abs=1e-9)
                    assert pl.foot_pos[k, j, 2] == pytest.approx(z0[side][2], abs=1e-12)


def test_start_and_stop_steps(nominal):
    src, g = nominal
    S = g["traj"]["info"]["stride"]
    pl, geo = _plan(src, "start")
    steps = [(s[0], s[1], float(np.linalg.norm(s[5][0] - s[4][0]))) for s in pl.swings
             if pl.first - 0.5 <= s[1] < pl.last]  # the swings of the output window (as transitions.summary lists them)
    assert steps[0][:2] == ("left", 0.0) and steps[0][2] == pytest.approx(0.5 * S, rel=1e-9)  # half a stride first
    assert all(length == pytest.approx(S, rel=1e-9) for _, lo, length in steps[1:])  # then the periodic walk's
    k0 = int(round((-transitions.SHIFT_STRIDES - pl.phase[0]) * transitions.N_FRAMES))
    assert np.all(pl.zmp[:k0] == pl.stand_point["initial"]) and np.all(~pl.swing[: k0 + 1])  # stand, then shift
    assert pl.stand_point["initial"] == pytest.approx(geo.com0[:2])
    st, _ = _plan(src, "stop")
    last = st.swings[-1]
    assert last[:2] == ("left", 0.0) and float(np.linalg.norm(last[5][0] - last[4][0])) == pytest.approx(0.5 * S)
    fl, fr = st.foot_pos[-1, 0, :2], st.foot_pos[-1, 1, :2]  # one step, then the home stance (plan section 4)
    assert fl - fr == pytest.approx(geo.feet0["left"][:2] - geo.feet0["right"][:2], abs=1e-12)
    assert st.zmp[-1] == pytest.approx(st.stand_point["final"]) and not st.swing[-transitions.N_FRAMES:].any()


def test_transition_plan_steady_part_is_the_periodic_walk(nominal):
    src, g = nominal
    arrays = refs.arrays(src, g)
    n = transitions.N_FRAMES
    pl, _ = _plan(src, "start")
    ref = refs.Reference("nominal", arrays)
    mot = transitions._align(pl, float(pl.last - 1), ref)  # the last output stride against the walk's stride 0
    sel = slice(pl.out.stop - n, pl.out.stop)
    assert pl.zmp[sel] == pytest.approx(transitions._move(ref.zmp_ref, mot), abs=1e-9)  # the ZMP target
    assert pl.foot_pos[sel][:, :, :2] == pytest.approx(transitions._move(ref.feet[:, :, :2], mot), abs=2e-4)  # IK tol.
    assert np.array_equal(pl.swing[sel], ~ref.support)


def test_preview_control_reproduces_the_periodic_lipm(nominal):
    _, g = nominal
    tr = g["traj"]
    c, dt, S, om = tr["com_ref"], tr["dt"], tr["info"]["stride"], tr["info"]["omega"]
    n = len(c)
    lam = 2 * np.pi * np.fft.fftfreq(n, d=dt)
    cx = c[:, 0] - S / (n * dt) * np.arange(n) * dt
    cdd = np.real(np.fft.ifft(-(lam**2) * np.fft.fft(cx)))
    cd = np.real(np.fft.ifft(1j * lam * np.fft.fft(cx))) + S / (n * dt)
    zx = c[:, 0] - cdd / om**2  # the ZMP whose periodic LIPM solution is gait.generate's CoM
    gains = transitions.preview_gains(dt, om)
    reps = 8 + gains["N"] // n + 2
    xs, _ = transitions.preview_run(gains, np.concatenate([zx + j * S for j in range(reps)]),
                                    np.array([c[0, 0], cd[0], cdd[0]]))
    err = max(np.abs(xs[j * n : (j + 1) * n, 0] - (c[:, 0] + j * S)).max() for j in range(8))
    assert err < 2e-5  # m: 4 um here


def test_transition_derivatives_on_a_nonuniform_grid(nominal):
    src, _ = nominal
    m = src["M"].m
    t = np.cumsum(np.r_[0.0, np.linspace(0.006, 0.004, 30)])  # a speed change's shrinking frame interval
    qs = np.tile(src["q_home"], (len(t), 1))
    a = np.linspace(0.1, 0.5, m.nv - 6)
    qs[:, 7:] += 0.3 * t[:, None] + a[None, :] * t[:, None] ** 2  # quadratic in time: the scheme is exact
    qd, qdd = transitions.derivatives(m, qs, t)
    assert qd[1:-1, 6:] == pytest.approx(0.3 + 2 * a[None, :] * t[1:-1, None], abs=1e-9)
    assert qdd[2:-2, 6:] == pytest.approx(np.tile(2 * a, (len(t) - 4, 1)), abs=1e-6)
    assert np.isnan(qd[0]).all() and np.isnan(qdd[1]).all()


@pytest.mark.parametrize("body", sorted(BODIES))
def test_transition_records_pass_g3(body):
    """The committed transitions (``transitions.json``) and their G3 record in ``gates.json``."""
    doc = json.loads((RUN / body / "transitions.json").read_text())
    rows = doc["transitions"]
    assert set(rows) == set(transitions.TRANSITIONS)
    for name, e in rows.items():
        assert (e["from_"], e["to"]) == transitions.TRANSITIONS[name] and e["ik_ok"]
        assert e["g3"]["com_max_mm"] <= transitions.G3_COM_MM and e["g3"]["zmp_max_mm"] <= transitions.G3_ZMP_MM
        assert e["g3"]["pass_"] and e["join_a"]["at_join"]["com_mm"] < 0.1 and e["com_tracking_max_m"] < 1e-5
        if e["kind"] == "stop":  # one step, then double support until at rest within about 1.5 strides
            assert e["stop"]["at_rest_strides"] < 1.6 and e["steps"][-1]["lift_off_phase"] == 0.0
            assert e["stop"]["rule"] == transitions.STOP_RULE and e["stop"]["command_phase"] == 0.0
    g3 = json.loads((RUN / body / "gates.json").read_text())["G3"]
    assert g3["pass_"] and not g3["failed"] and not g3["fallback"] and set(g3["transitions"]) == set(rows)
    # decision 18: stopped = forward speed below 5 % of the command within 1.5 strides of the stop command; the stop
    # references stop in about half a stride and come to rest after 1.55-1.57 strides (reported)
    st = g3["stops"]
    assert st["limit_strides"] == gates.STOP_WITHIN_STRIDES == 1.5 and st["speed_frac"] == 0.05
    assert set(st["stops"]) == {n for n, e in rows.items() if e["kind"] == "stop"} and st["every_within"]
    for name, s in st["stops"].items():
        assert s["at_rest_strides"] == rows[name]["stop"]["at_rest_strides"]
        if rows[name]["gaits"]["a"]["speed"] > 0:  # a walk's stop: forward reading
            assert 0.4 < s["forward_stopped_strides"] < 0.6 and s["within"]
            assert 1.5 < s["at_rest_strides"] < 1.6
        else:  # an in-place turn's stop has no forward command
            assert s["forward_stopped_strides"] is None and s["within"] is None


def test_g3_cli_reproduces_the_record(tmp_path):
    out = tmp_path / BODY
    out.mkdir()
    (out / "transitions.json").write_text((RUN / BODY / "transitions.json").read_text())
    assert cli(["gates", "--body", BODY, "--out", str(tmp_path), "--only", "g3"]) == 0
    got = json.loads((out / "gates.json").read_text())["G3"]
    rec = json.loads((RUN / BODY / "gates.json").read_text())["G3"]
    assert set(got.pop("provenance")) == set(rec.pop("provenance")) == {"versions", "git", "stage1_sources"}
    assert got == rec  # the same G3 from the committed transitions.json, whatever code version made the record


# ------------------------------------------------------------------------------------------------
# Tracker (plan section 3 step 5, G4): control law and estimator
# ------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def nominal_ref(nominal_npz):
    return refs.Reference.load(nominal_npz)


def _tracker(p, ref, mode="R1", strides=3.0, **kw):
    track = control.Track(p, control.omega0(p, ref.q_home)).add_periodic(ref, strides)
    return control.Tracker(p, track, mode, **kw)


PROTOTYPE = dict(k_dcm=2.0, cop_x=0.02, cop_y=0.012, k_step=1.0, step_x=0.05, step_out=0.03, decay=1.0, k_trunk=40.0,
                 d_trunk=2.0)  # track_proto.py's ankle, step and trunk terms


def test_tracker_gains_are_one_named_vector_starting_at_the_prototype():
    """Decision 16: the tuning start is the prototype's terms (ankle and step feedback and trunk PD at its values),
    the heading and phase-rate gains zero; the build's earlier defaults stay available for the record."""
    assert len(control.GAIN_NAMES) == 14 == len(control.START_GAINS) == len(set(control.GAIN_NAMES))
    lo, hi = control.GAIN_BOUNDS.T
    for x in control.GAIN_SETS.values():  # both sets lie inside the CMA-ES bounds
        assert np.all(lo <= x) and np.all(x <= hi)
    start, earlier = control.gains_dict(), control.gains_dict("earlier_defaults")
    assert start == control.gains_dict("start") == control.gains_dict(control.START_GAINS)
    assert np.array_equal(control.DEFAULT_GAINS, control.START_GAINS)  # the tracker's default
    assert {k: start[k] for k in PROTOTYPE} == PROTOTYPE == {k: earlier[k] for k in PROTOTYPE}
    assert all(start[k] == 0.0 for k in ("k_yaw", "ki_yaw", "k_phase", "ki_phase"))
    assert (earlier["k_yaw"], earlier["ki_yaw"], earlier["k_phase"], earlier["ki_phase"]) == (0.5, 0.1, 2.0, 1.0)
    assert start["capture_margin"] == earlier["capture_margin"] == 0.0
    assert control.gains_dict(control.START_GAINS * 2)["k_step"] == 2.0
    with pytest.raises(ValueError):
        control.gains_dict(np.ones(13))
    with pytest.raises(ValueError):
        control.gains_dict("tuned")
    assert control.turn_gap(BODY) == pytest.approx(3.43e-3)  # the target turns' smallest gap (turning screen)


@pytest.mark.parametrize("body", sorted(BODIES))
def test_turn_gap_is_a_pelvis_to_thigh_gap(body):
    """The turn gap the inward step clamp uses is between the trunk's pelvis and the same-side thigh, not between the
    legs (the screen's leg-to-leg gap is tens of millimetres)."""
    info = control.turn_gap_info(body)
    assert set(info) == set(control.TURN_CASES)
    case = min(info, key=lambda c: info[c]["min_gap_mm"])
    assert 1e-3 * info[case]["min_gap_mm"] == pytest.approx(control.turn_gap(body))
    trunk, thigh = info[case]["bodies"]
    assert trunk == "trunk" and thigh.endswith("_thigh") and not info[case]["leg_to_leg"]
    assert _side_letter(info[case]["pair"][0]) == thigh[0]  # the pelvis part on the thigh's own side
    assert all(c["lateral_gap_mm"] > 30.0 for c in info.values())


def _side_letter(geom: str) -> str | None:
    m = re.search(r"_([lr])(_|$)", geom)
    return m.group(1) if m else None


def _poly_gap(a: np.ndarray, b: np.ndarray) -> float:
    """The distance between two disjoint convex polygons (the smallest vertex-to-edge distance either way)."""
    best = np.inf
    for P, Q in ((a, b), (b, a)):
        for x in P:
            for i in range(len(Q)):
                u, v = Q[i], Q[(i + 1) % len(Q)]
                t = np.clip((x - u) @ (v - u) / ((v - u) @ (v - u)), 0.0, 1.0)
                best = min(best, float(np.linalg.norm(x - (u + t * (v - u)))))
    return best


def test_inward_room_keeps_the_pad_clearance():
    """Decision 20's inward bound: how far a swing pad may move inward before it is within 5 mm of the stance pad."""
    sq = np.array([[0.0, 0.0], [0.1, 0.0], [0.1, 0.05], [0.0, 0.05]])
    beside = sq + [0.0, 0.07]  # 20 mm to the left of it
    inward = np.array([0.0, -1.0])
    assert control.PAD_CLEARANCE == 0.005
    assert control.inward_room(sq, beside, inward) == pytest.approx(0.015, abs=1e-12)  # parallel edges: exact
    assert _poly_gap(sq, beside + 0.015 * inward) == pytest.approx(0.005, abs=1e-12)
    assert control.inward_room(sq, sq + [0.0, 0.052], inward) == 0.0  # already within 5 mm: no inward step
    assert control.inward_room(sq, beside + [0.2, 0.0], inward) == np.inf  # landing well ahead: never that close
    assert control.inward_room(sq, beside, -inward) == np.inf  # moving away
    tilted = (beside - beside.mean(0)) @ control._rot2(0.3).T + beside.mean(0) + [0.0, 0.02]  # a turned landing
    room = control.inward_room(sq, tilted, inward)
    assert 0.0 < room < np.inf and _poly_gap(sq, tilted + room * inward) >= 0.005 - 1e-12  # kept, with corners spare
    assert _poly_gap(sq, tilted + room * inward) < 0.0075


def test_inward_bound_on_the_reference_touchdowns(nominal_npz, tmp_path):
    """On the nominal walk the swing foot lands half a stride ahead, clear of the stance pad whatever its inward step:
    the bound is unlimited there, and moving the swing foot 50 mm inward near touchdown keeps the leg parts that come
    close in motion at least 25 mm (the cap) apart; the turns' bounds are recorded with G4."""
    (tmp_path / "refs").mkdir()
    (tmp_path / "refs" / "nominal.npz").write_bytes(nominal_npz.read_bytes())
    r = control.inward_step_clearance(BODY, tmp_path, references=("nominal",))
    assert r["pad_clearance_m"] == control.PAD_CLEARANCE and r["deltas_m"][-1] == 0.05
    for side, sw in r["references"]["nominal"]["swings"].items():
        assert sw["bound_m"] == np.inf and len(sw["rows"]) == len(r["deltas_m"])  # no row at an unlimited bound
        assert all(x["moving_leg_pairs_mm"] >= 0.999 * r["cap_mm"] for x in sw["rows"]), side


def test_r0_is_g2s_replay_bit_for_bit(nominal_ref):
    p = plant.build(BODY, "hard", settle=False)
    r = control.run(_tracker(p, nominal_ref, "R0"), nominal_ref.period)
    q_r0 = p.d.qpos.copy()
    p2 = plant.build(BODY, "hard", settle=False)
    g = gates.replay(p2, nominal_ref, strides=1.0)
    assert r["strides"] == g["strides"] and np.array_equal(q_r0, p2.d.qpos)


def test_servo_target_law(hard, nominal_ref):
    """ctrl = q_ref + dq + (kv qd_ref + tau_ff + dtau) / kp in motor coordinates (plan section 2)."""
    tk = _tracker(hard, nominal_ref, ankle_cap=())  # the bare law (the cap: test_tracker_caps_the_ankle_command)
    tk.hold = "hold"
    tk.reset("frame")
    rng = np.random.default_rng(4)
    dq, dtau = rng.uniform(-0.01, 0.01, tk.nj), rng.uniform(-0.5, 0.5, tk.nj)

    def fixed(t_ref, q_ref):
        tk.dq, tk.dtau = dq, dtau

    tk.feedback = fixed
    seen = {}
    step = hard.step
    hard.step = lambda n=1: seen.setdefault("ctrl", hard.d.ctrl.copy())
    try:
        tk.step()
    finally:
        hard.step = step
    x = nominal_ref.interp(0.0, np.hstack(nominal_ref.joint_series(hard.qadr, hard.dofs)
                                          + (nominal_ref.qdd_series(hard.dofs),)))
    nj = tk.nj
    q, qd, tau, qdd = x[:nj], x[nj : 2 * nj], x[2 * nj : 3 * nj], x[3 * nj :]
    ff = tau + tk.damp * qd + tk.fric * np.tanh(qd / gates.FRICTION_SMOOTH) + hard.armature_extra @ qdd
    want = hard.G @ (q + dq) + (hard.kv * (hard.G @ qd) + np.linalg.solve(hard.G.T, ff + dtau)) / hard.kp
    assert seen["ctrl"] == pytest.approx(want, rel=1e-12, abs=1e-12)
    hard.reset()


def test_ankle_feedback_shifts_the_cop(hard, nominal_ref):
    """A DCM error e gives the stance ankle the torque of a ZMP shift clip(k_dcm e) under the body's weight."""
    tk = _tracker(hard, nominal_ref, terms=("ankle",))
    tk.reset("frame")
    k = 30  # left foot in swing: the right foot alone carries the body
    assert not nominal_ref.support[k, 0] and nominal_ref.support[k, 1]
    hard.set_state(nominal_ref.q[k], nominal_ref.qd[k])
    t = k * nominal_ref.step
    view = tk.state.view

    def shifted():
        v = view()
        _, ref = tk.track.info(t)
        pa, ya = tk._anchor_frame(v, int(ref["anchor"]))
        v["com"] = v["com"].copy()  # put the DCM 3 mm ahead and 2 mm to the left of the reference
        v["com"][:2] = pa + control._rot2(ya) @ (ref["dcm_rel"] + [0.003, 0.002]) - v["com_vel"][:2] / tk.track.omega
        return v

    tk.state.view = shifted
    tk.feedback(t, None)
    dp = control._rot2(tk._anchor_frame(view(), 1)[1]) @ np.clip(2.0 * np.array([0.003, 0.002]), -0.012, 0.012)
    Fz = tk.mass * control.G
    M = Fz * np.array([-dp[1], dp[0], 0.0])
    d = hard.d
    for x in ("ankle_pitch", "ankle_roll"):
        i = tk.idx[f"right_{x}"]
        assert tk.dtau[i] == pytest.approx(float(d.xaxis[tk.jid[f"right_{x}"]] @ M), rel=1e-9)
        assert tk.dtau[tk.idx[f"left_{x}"]] == 0.0
    assert np.allclose(np.delete(tk.dtau, [tk.idx[f"{s}_{x}"] for s in statics.SIDES
                                           for x in ("ankle_pitch", "ankle_roll")]), 0.0)
    hard.reset()


def test_ankle_torques_imply_the_contact_cop(hard):
    hard.reset()
    m, d = hard.m, hard.d
    q = hard.q_home[hard.qadr].copy()
    for s in statics.SIDES:
        q[hard.joints.index(f"{s}_ankle_pitch")] += 0.02
        q[hard.joints.index(f"{s}_ankle_roll")] -= 0.01
    hard.command(q)
    hard.step(1000)
    mujoco.mj_forward(m, d)
    tau = d.qfrc_actuator[hard.dofs] + d.qfrc_passive[hard.dofs]
    for i, (side, f) in enumerate(zip(statics.SIDES, hard.feet, strict=True)):
        jids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_ankle_{x}") for x in ("pitch", "roll")]
        t2 = tau[[hard.joints.index(f"{side}_ankle_{x}") for x in ("pitch", "roll")]]
        Fz, cs = 0.0, np.zeros(3)
        for k in range(d.ncon):
            c = d.contact[k]
            if f in (m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]):
                fr = np.zeros(6)
                mujoco.mj_contactForce(m, d, k, fr)
                Fz, cs = Fz + fr[0], cs + fr[0] * c.pos
        cop = control.ankle_cop(d, jids, t2, Fz)
        assert cop == pytest.approx(cs[:2] / Fz, abs=2e-3)  # quasi-static: within 2 mm of MuJoCo's contact CoP
        assert control.ankle_torque(d, jids, cop, Fz) == pytest.approx(t2, rel=1e-9, abs=1e-12)
    hard.reset()


def test_support_polygon_helpers():
    sq = np.array([[0.0, 0.0], [0.1, 0.0], [0.1, 0.1], [0.0, 0.1]])
    assert control.shrink(sq, 0.01) == pytest.approx(np.array([[0.01, 0.01], [0.09, 0.01], [0.09, 0.09], [0.01, 0.09]]))
    assert control.project_into(sq, np.array([0.05, 0.05])) == (pytest.approx([0.05, 0.05]), False)
    x, out = control.project_into(sq, np.array([0.13, 0.05]))
    assert out and x == pytest.approx([0.1, 0.05])
    assert control.signed_distance(sq, np.array([0.05, 0.13])) == pytest.approx(0.03)
    assert control.signed_distance(sq, np.array([0.05, 0.02])) == pytest.approx(-0.02)
    sup = np.array([[True, True]] * 3 + [[False, True]] * 4 + [[True, True]] * 2 + [[True, False]] * 4 + [[True, True]])
    sw = control._swing_of(sup)
    assert list(sw[3:7]) == [0] * 4 and list(sw[9:13]) == [1] * 4
    assert list(control._swing_progress(sw, periodic=False)[3:7]) == pytest.approx([0, 0.25, 0.5, 0.75])
    anc = control._anchor(sw, periodic=False, half=3)  # stance foot; the foot that landed last; the next stance foot
    assert list(anc) == [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1]


def test_r1_tracks_a_few_strides(hard, nominal_ref):
    tk = _tracker(hard, nominal_ref, strides=4.0)
    r = control.run(tk, 3 * nominal_ref.period)
    assert not r["fell"] and r["strides"] == pytest.approx(3.0, abs=1e-3)
    assert max(abs(v) for v in r["phase_rate_range"]) <= control.PHASE_RATE_LIMIT
    assert r["dcm_error_rms_mm"] < 30.0 and r["segment_reached"] == "nominal"
    hard.reset()


@pytest.mark.parametrize("mode", ["R1", "R2"])
def test_saved_episode_state_repeats_bit_for_bit(hard, nominal_ref, mode):
    """``Tracker.load`` of a ``Tracker.save`` taken mid-walk (``run``'s way: MuJoCo's state at the start of the step
    just taken) continues exactly as the episode did: the counterfactuals' unchanged continuation."""
    tk = _tracker(hard, nominal_ref, mode)
    tk.reset("frame", seed=3)
    m, d = hard.m, hard.d
    bufs = [np.empty(mujoco.mj_stateSize(m, control.MJ_STATE)) for _ in range(2)]
    snap = None
    for k in range(352):  # a snapshot off the feedback updates' phase
        bufs.reverse()
        mujoco.mj_getState(m, d, bufs[0], control.MJ_STATE)
        if k == 351:
            snap = tk.save(bufs[1])
            break
        tk.step()
    for _ in range(400):
        tk.step()
    want = (d.qpos.copy(), d.qvel.copy(), d.sensordata.copy(), tk.dq.copy(), tk.e.copy(), tk.delta)
    with control.relaxed(tk, control.RELAXATIONS):  # a relaxed branch in between leaves nothing behind
        tk.load(snap)
        for _ in range(200):
            tk.step()
    tk.load(snap)
    for _ in range(400):
        tk.step()
    got = (d.qpos.copy(), d.qvel.copy(), d.sensordata.copy(), tk.dq.copy(), tk.e.copy(), tk.delta)
    for a, b in zip(want, got, strict=True):
        assert np.array_equal(a, b)
    hard.reset()


def _collide(m, a: int, b: int) -> bool:
    return bool((m.geom_contype[a] & m.geom_conaffinity[b]) or (m.geom_contype[b] & m.geom_conaffinity[a]))


def test_relaxations_lift_the_limits_and_are_undone(hard, nominal_ref):
    tk = _tracker(hard, nominal_ref)
    tk.reset("frame")
    m = hard.m
    before = (m.actuator_forcelimited.copy(), m.geom_friction.copy(), m.jnt_range.copy(), m.geom_contype.copy())
    pads = [np.flatnonzero((tk.foot_of == i) & (m.geom_contype > 0)) for i in (0, 1)]
    floor = statics.gid(m, "floor")
    assert all(_collide(m, a, b) for a in pads[0] for b in pads[1])  # the two feet's pads collide
    with control.relaxed(tk, control.RELAXATIONS):
        assert not m.actuator_forcelimited[tk.leg_act].any() and m.actuator_forcelimited[~tk.leg_act].all()
        assert not hard.envelope.enabled
        cg = (m.geom_contype | m.geom_conaffinity) > 0
        assert np.all(m.geom_friction[cg, 0] == control.CF_FRICTION)
        assert m.jnt_range[tk.hinge_jid] == pytest.approx(before[2][tk.hinge_jid] + [-1.0, 1.0])
        # the leg-to-leg clearance lifted: no inward bound, and the pads pass each other but still meet the floor
        assert tk.inward_bound is None
        assert not any(_collide(m, a, b) for a in pads[0] for b in pads[1])
        assert all(_collide(m, g, floor) for g in np.concatenate(pads))
        assert tk.ankle_cap == control.ANKLE_CAP  # the tracker's own cap limits the controller and stays
    assert hard.envelope.enabled and tk.inward_bound == "pad_clearance"
    for x, y in zip(before, (m.actuator_forcelimited, m.geom_friction, m.jnt_range, m.geom_contype), strict=True):
        assert np.array_equal(x, y)
    with pytest.raises(ValueError):
        with control.relaxed(tk, ("pads",)):
            pass
    assert set(control.RELAXES) == set(control.HW_LIMITS) and set(control.RELAXES.values()) - {None} == set(
        control.RELAXATIONS)
    assert control.RELAXES["cop_at_pad_edge_with_ankle_saturated"] is None  # the pads' size: not relaxable
    # decision 20: no turn-gap clamp anywhere; the inward bound is the leg-to-leg clearance, a hardware limit; the
    # cap's per-axis box is the controller's own, reported
    assert "turn_gap_clamp" not in control.HW_LIMITS + control.REPORTED_LIMITS
    assert "leg_to_leg_clearance" in control.HW_LIMITS and "ankle_cap_box" in control.REPORTED_LIMITS
    assert "neck_tail_at_torque_limit" in control.REPORTED_LIMITS
    assert not tk.ankle_act[~tk.leg_act].any() and tk.ankle_act.sum() == 4
    hard.reset()


def test_slip_is_measured_at_the_loaded_contacts(hard):
    """A foot pivoting on the floor moves its body origin (on the ankle axis) but does not slip; a foot sliding does."""
    hard.reset()
    m, d = hard.m, hard.d
    base = d.qpos.copy()

    def at(v_root, w_world, pivot):
        d.qpos[:] = base
        d.qvel[:] = 0.0
        R = d.xmat[hard.trunk].reshape(3, 3).copy()
        d.qvel[:3] = np.asarray(v_root) + np.cross(w_world, d.qpos[:3] - pivot)  # the whole body turning about pivot
        d.qvel[3:6] = R.T @ np.asarray(w_world)  # (a free joint's angular velocity is in its body frame)
        mujoco.mj_forward(m, d)
        v6 = np.zeros(6)
        mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_XBODY, hard.feet[0], v6, 0)  # the foot's origin
        return control.foot_slip_speed(hard), float(np.linalg.norm(v6[3:5]))

    mujoco.mj_forward(m, d)
    pivot = np.r_[d.xpos[hard.feet[0]][:2], hard.z_floor]  # under the left foot's origin, on the floor
    slip, origin = at([0.1, 0.0, 0.0], np.zeros(3), pivot)
    assert slip == pytest.approx([0.1, 0.1], abs=1e-9) and origin == pytest.approx(0.1)
    slip, origin = at(np.zeros(3), np.array([0.0, 4.0, 0.0]), pivot)  # rocking about a line on the floor
    assert origin == pytest.approx(4.0 * (d.xpos[hard.feet[0]][2] - hard.z_floor), rel=1e-6)  # 80 mm/s
    assert origin > control.SLIP_SPEED and slip[0] < 0.2 * control.SLIP_SPEED
    hard.reset()


def _episode_label(cf: dict) -> str:
    """``control.EPISODE_CF_RULE`` from a counterfactual's continuations."""
    if cf["null"]["avoided"]:
        return "undetermined: chaotic (the null change avoids the fall)"
    if cf["binding"]:
        return "body-limited: " + ", ".join(cf["binding"])
    if (cf["together"] or {}).get("avoided"):
        return "body-limited: " + " + ".join(cf["relaxations"]) + " together"
    if cf["not_relaxable"]:
        return "unconfirmed: " + ", ".join(cf["not_relaxable"]) + " (not relaxable in the plant)"
    return "controller-limited"


def test_episode_counterfactual_diagnostic(nominal_ref):
    """The build's episode-level counterfactual (``run(confirm=True)``, a diagnostic since decision 20), on a plant
    whose drives are too weak (30 % of their torque): the restored episode repeats the fall bit for bit, a 1 mm/s
    nudge at the branch does not avoid it, and the label follows from the continuations.  The drives bind throughout:
    lifting their limits only a second before the onset is too late to save this body, while lifting them from the
    start (how decision 20 tests a condition) lets it walk.  Neck and tail are reported, never candidates."""
    p = plant.build(BODY, "hard", settle=False, torque_scale=0.3)
    tk = _tracker(p, nominal_ref, strides=11.0)
    r = control.run(tk, 10 * nominal_ref.period, confirm=True)
    assert r["fell"] and r["fall_onset_s"] < r["fall_time_s"]
    cf = r["counterfactual"]
    assert cf["reproduces"] and not cf["null"]["avoided"]
    assert cf["branch_s"] <= r["fall_onset_s"] - control.CF_BRANCH_S + 1e-9  # (_confirm's own tolerance)
    assert "leg_drive_at_torque_limit" in cf["candidates"] and "drive_torque" in cf["relaxations"]
    assert cf["binding"] == [x for x in cf["relaxations"] if cf["alone"][x]["avoided"]]
    assert r["fall_label"] == _episode_label(cf)
    act = r["limits_before_onset"]
    assert set(act["hardware"]) == set(control.HW_LIMITS) and set(act["reported"]) == set(control.REPORTED_LIMITS)
    assert act["hardware"]["leg_drive_at_torque_limit"] > 0.5 and set(cf["candidates"]) <= set(control.HW_LIMITS)
    assert all(not a.startswith(("neck", "tail")) for a in act["leg_drives"])
    q_end = p.d.qpos.copy()  # the episode is left at its fall after the counterfactuals
    assert p.fall_cause() == r["fall_cause"] and p.d.time == r["fall_time_s"]
    r2 = control.run(tk, 10 * nominal_ref.period)  # the default: the same episode, its candidates, no label
    assert r2["strides"] == r["strides"] and r2["fall_label"] is None and np.array_equal(p.d.qpos, q_end)
    assert r2["fall_candidates"] == cf["candidates"] == r["fall_candidates"]
    assert not cf["alone"]["drive_torque"]["avoided"]  # a second before the onset: too late
    r3 = control.run(tk, 10 * nominal_ref.period, relax=("drive_torque",))  # from the start: it walks
    assert not r3["fell"] and r3["relaxed"] == ["drive_torque"] and np.all(p.m.actuator_forcelimited)


def _recorded_walk(p, ref, seconds: float) -> list:
    """Sensor readings (noise-free) and the true state of an R1 walk, every physics step."""
    tk = _tracker(p, ref, strides=seconds / ref.period + 1)
    tk.reset("frame")
    clean = estimator.NoisySensors(p, estimator.SensorNoise.none(), 0)
    rec = [dict(qpos=p.d.qpos.copy(), qvel=p.d.qvel.copy(), reading=clean.read())]
    for _ in range(int(round(seconds / p.m.opt.timestep))):
        tk.step()
        mujoco.mj_subtreeVel(p.m, p.d)
        d = p.d
        rec.append(dict(reading=clean.read(), com=d.subtree_com[p.trunk].copy(), com_vel=d.subtree_linvel[p.trunk].copy(),
                        feet=np.array([d.xpos[f].copy() for f in p.feet]), R=d.xmat[p.trunk].reshape(3, 3).copy()))
    return rec


def _replay_estimator(p, rec, noise: estimator.SensorNoise, seed: int = 1) -> np.ndarray:
    """The estimator on a recorded walk; errors every 10 ms: CoM relative to the feet's midpoint (mm), CoM velocity
    (m/s, horizontal), tilt (deg)."""
    est = estimator.Estimator(p)
    rng = np.random.default_rng(seed)
    corrupt = estimator.NoisySensors(p, noise, seed)

    def noisy(r):
        out = {k: np.array(v, float) for k, v in r.items()}
        n, s = noise, noise.scale
        if s:
            out["gyro"] += corrupt.gyro_b + s * n.gyro_sigma * rng.standard_normal(3)
            out["accel"] += corrupt.accel_b + s * n.accel_sigma * rng.standard_normal(3)
            out["qd"] += s * n.joint_rate_sigma * rng.standard_normal(len(out["qd"]))
            out["foot_load"] += s * n.foot_load_sigma_frac * corrupt.weight * rng.standard_normal(2)
        if corrupt.q_step:
            out["q"] = np.round(out["q"] / corrupt.q_step) * corrupt.q_step
        return out

    est.reset(rec[0]["qpos"], rec[0]["qvel"], noisy(rec[0]["reading"]))
    errs = []
    for k, r in enumerate(rec[1:], start=1):
        est.step(noisy(r["reading"]))
        if k % control.FEEDBACK_EVERY == 0:
            est.update(control.FEEDBACK_EVERY * p.m.opt.timestep)
            rel = (est.com[:2] - est.foot[:, :2].mean(0)) - (r["com"][:2] - r["feet"][:, :2].mean(0))
            tilt = np.degrees(np.arccos(np.clip(est.R[:, 2] @ r["R"][:, 2], -1, 1)))
            errs.append((1e3 * np.linalg.norm(rel), np.linalg.norm(est.com_vel[:2] - r["com_vel"][:2]), tilt))
    return np.array(errs)


def test_estimator_on_a_recorded_walk(hard, nominal_ref):
    rec = _recorded_walk(hard, nominal_ref, 2.0)
    rms = lambda E: np.sqrt(np.mean(E**2, axis=0))  # noqa: E731
    clean = rms(_replay_estimator(hard, rec, estimator.SensorNoise.none()))
    noisy = rms(_replay_estimator(hard, rec, estimator.SensorNoise()))
    assert clean[0] < 2.0 and clean[1] < 0.012 and clean[2] < 0.2  # mm, m/s, deg
    assert noisy[0] < 3.0 and noisy[1] < 0.02 and noisy[2] < 0.3
    assert set(estimator.NoisySensors(hard, estimator.SensorNoise(), 0).read()) == {"gyro", "accel", "q", "qd",
                                                                                     "foot_load"}  # no imu_quat
    hard.reset()


def test_estimator_tilt_correction_pulls_toward_the_flat_feet(hard):
    hard.reset()
    hard.hold()
    hard.step(5)
    sens = estimator.NoisySensors(hard, estimator.SensorNoise.none(), 0)
    est = estimator.Estimator(hard)
    q = hard.d.qpos.copy()
    tilt = np.zeros(4)
    mujoco.mju_axisAngle2Quat(tilt, np.array([1.0, 0.0, 0.0]), np.radians(3.0))
    mujoco.mju_mulQuat(q[3:7], q[3:7].copy(), tilt)  # start 3 deg off in roll
    est.reset(q, np.zeros(hard.m.nv), sens.read())
    R = hard.d.xmat[hard.trunk].reshape(3, 3)

    def err():
        return np.degrees(np.arccos(np.clip(est.R[:, 2] @ R[:, 2], -1, 1)))

    e0 = err()
    for k in range(1000):
        hard.step()
        est.step(sens.read())
        if k % 5 == 4:
            est.update(0.01)
    assert e0 == pytest.approx(3.0, abs=0.1) and err() < 0.1  # the 2/s leg-tilt correction: e^-4 of it after 2 s
    rest = estimator.Estimator(hard)
    rest.reset_at_rest(sens.read(), yaw=0.0, base_xy=hard.d.qpos[:2])  # from the sensors alone at rest
    assert np.degrees(np.arccos(np.clip(rest.R[:, 2] @ R[:, 2], -1, 1))) < 0.2
    hard.reset()


# ------------------------------------------------------------------------------------------------
# Decisions 15-20 (STAGE1_PLAN.md section 9, steps 4-5)
# ------------------------------------------------------------------------------------------------
def _cap_feasible(t, peak, Fz, A, b, pad) -> bool:
    if np.any(np.abs(t) > peak + 1e-9):
        return False
    return Fz is None or control.signed_distance(pad, np.linalg.solve(A, -np.asarray(t) / Fz - b)) <= 1e-9


def test_ankle_cap_is_the_nearest_feasible_torque():
    """Decision 15's cap: each axis at most one motor's peak, a loaded foot's CoP inside its (inset) pad, and the
    torque kept the feasible one nearest to the command."""
    A, b = np.array([[-1.0, 0.0], [0.0, 1.0]]), np.zeros(2)  # pitch about y and roll about x through the origin
    pad = np.array([[-0.01, -0.03], [0.07, -0.03], [0.07, 0.03], [-0.01, 0.03]])  # the CoP may lie in this, m
    Fz, peak = 100.0, np.array([5.0, 5.0])  # so pitch = Fz c_x, roll = -Fz c_y
    for tau, want, flags in (([2.0, 1.0], [2.0, 1.0], (False, False)),  # inside both: unchanged
                             ([6.0, 1.0], [5.0, 1.0], (True, False)),  # the box only: per-axis clip, CoP in the pad
                             ([-3.0, 1.0], [-1.0, 1.0], (False, True)),  # behind the pad: the CoP moved onto its edge
                             ([8.0, 4.0], [5.0, 3.0], (True, True))):  # both
        t, box, padb = control.cap_ankle_torque(tau, peak, Fz, A, b, pad)
        assert t == pytest.approx(want, abs=1e-12) and (box, padb) == flags
    t, box, padb = control.cap_ankle_torque([8.0, -9.0], peak)  # a foot in the air: the box alone
    assert t == pytest.approx([5.0, -5.0]) and box and not padb
    rng = np.random.default_rng(1)
    grid = [np.array([x, y]) for x in np.linspace(-5, 5, 41) for y in np.linspace(-5, 5, 41)]
    feasible = [g for g in grid if _cap_feasible(g, peak, Fz, A, b, pad)]
    for _ in range(30):  # feasible, and no feasible grid point nearer
        tau = rng.uniform(-12.0, 12.0, 2)
        t = control.cap_ankle_torque(tau, peak, Fz, A, b, pad)[0]
        assert _cap_feasible(t, peak, Fz, A, b, pad)
        assert np.linalg.norm(t - tau) <= min(np.linalg.norm(g - tau) for g in feasible) + 1e-9


def test_tracker_caps_the_ankle_command(hard, nominal_ref):
    """Decision 15 in the tracker (R1, which predicts the servo's torque exactly): asked for far more than the feet
    can take, each ankle's total torque command stays within one motor's peak on each axis (the plant's linkage would
    allow twice it on one axis) and the stance foot's CoP 5 mm inside its pad."""
    tk = _tracker(hard, nominal_ref)
    tk.hold = "hold"
    tk.reset("frame")
    k = 30  # left foot swinging, right foot in stance
    hard.set_state(nominal_ref.q[k], nominal_ref.qd[k])
    tk.tau_ref, tk.k = k * nominal_ref.step, 1  # off a feedback update: the offsets below stay
    big = np.zeros(tk.nj)
    for s_ in statics.SIDES:
        big[tk.idx[f"{s_}_ankle_pitch"]], big[tk.idx[f"{s_}_ankle_roll"]] = -40.0, 25.0
    tk.dq = tk.dq_prev = np.zeros(tk.nj)
    tk.dtau = tk.dtau_prev = big
    seen = {}
    step = hard.step
    hard.step = lambda n=1: seen.setdefault("ctrl", hard.d.ctrl.copy())
    try:
        tk.step()
    finally:
        hard.step = step
    d = hard.d
    f = hard.kp * (seen["ctrl"] - hard.G @ d.qpos[hard.qadr]) - hard.kv * (hard.G @ d.qvel[hard.dofs])
    tau = hard.G.T @ f  # the servo's joint torque at this state, before the plant's own limits
    W = tk.mass * control.G
    loads = hard.foot_loads()
    assert loads[0] < control.CONTACT_LOAD_FRAC * W < loads[1]
    for i in (0, 1):
        ix = tk.ankle_ix[i]
        assert np.all(np.abs(tau[ix]) <= tk.ankle_peak[i] + 1e-9)
        if i == 1:
            fb = hard.feet[i]
            pad = d.xpos[fb][:2] + tk.pads_in[i] @ control._rot2(control._yaw(d.xmat[fb])).T
            assert control.signed_distance(pad, control.ankle_cop(d, tk.ankle_jid[i], tau[ix], loads[i])) <= 1e-9
    assert tk.cap_bound == [True, True]
    hard.reset()


def test_stop_reading_is_decision_18():
    """Stopped = the forward speed below 5 % of the command, from then on, within 1.5 strides of the stop command; the
    time to rest is reported."""
    t = np.arange(0.0, 4.0, 0.01)
    v = np.where(t < 1.0, 0.2, np.clip(0.2 * (1.5 - t) / 0.5, 0.0, None))  # stop command at 1 s, still at 1.5 s
    r = transitions.stop_reading(t, v, 0.2, 1.0, 1.0, rest=np.abs(v) < transitions.REST_SPEED_M_S)
    assert transitions.STOP_SPEED_FRAC == 0.05 and transitions.STOP_WITHIN_STRIDES == 1.5
    assert r["forward_stopped_strides"] == pytest.approx(0.48) and r["within"]  # below 10 mm/s from 1.48 s
    assert r["at_rest_strides"] == pytest.approx(0.5) and r["at_rest_s"] == pytest.approx(0.5)
    bump = v.copy()
    bump[(t > 2.0) & (t < 2.2)] = 0.05  # moving again: stopped only once it stays below
    assert transitions.stop_reading(t, bump, 0.2, 1.0, 1.0)["forward_stopped_strides"] == pytest.approx(1.2)
    late = transitions.stop_reading(t, v, 0.2, 1.0, 0.3)  # 0.48 s is 1.6 strides of 0.3 s
    assert late["forward_stopped_strides"] == pytest.approx(1.6) and not late["within"]
    assert transitions.stop_reading(t, v, 0.0, 1.0, 1.0)["forward_stopped_strides"] is None  # no forward command
    assert transitions.stop_command_phase(0.6) == pytest.approx(0.0)  # the nominal stop's command: its last lift-off
    s = dict(name="stop", t_ref=1.0, t_sim=1.0, speed_m_s=0.2, period_s=1.0,
             samples=[(x, y, abs(y)) for x, y in zip(t[t >= 1.0], v[t >= 1.0], strict=True)])
    out = control._stop_result(s)  # an episode's stop: the same reading of the samples after the command
    assert out["reached"] and out["forward_stopped_strides"] == pytest.approx(0.48) and out["within"]
    assert control._stop_result(dict(s, t_sim=None, samples=[]))["reached"] is False


def test_estimator_odometry_holds_the_loaded_pads_contact_point(hard):
    """Decision 19: the stance-leg odometry at the loaded pad's contact point.  Flat on its feet, the contact point is
    on the sole at the floor; with the body tilted 5 deg forward about the left foot's front pad edge (heel up) and
    turning about it at 1 rad/s, the contact is that edge and the base velocity estimated from the legs is the true
    one, where the foot body's centre of mass (11 mm above the floor) would be off by about 11 mm/s."""
    hard.reset()
    m, d = hard.m, hard.d
    W = float(m.body_subtreemass[hard.trunk]) * control.G
    est = estimator.Estimator(hard, estimator.EstimatorGains(k_leg_tilt=0.0, k_accel_tilt=0.0))
    est.reset(d.qpos.copy(), np.zeros(m.nv), estimator.NoisySensors(hard, estimator.SensorNoise.none(), 0).read())
    for i in (0, 1):
        assert est.contact[i] and est.contact_world[i][2] - hard.z_floor == pytest.approx(0.0, abs=1e-3)
        assert est.contact_local[i] == pytest.approx(est.sole[i].mean(0))  # every pad corner on the floor
    f = hard.feet[0]
    R_f = d.xmat[f].reshape(3, 3).copy()
    front = est.sole[0][:, 0] >= est.sole[0][:, 0].max() - 1e-9
    C = (d.xpos[f] + est.sole[0][front] @ R_f.T).mean(0)  # the front edge's midpoint, on the floor
    axis = R_f[:, 1]
    rot, Rm = np.zeros(4), np.zeros(9)
    mujoco.mju_axisAngle2Quat(rot, axis, np.radians(5.0))
    mujoco.mju_quat2Mat(Rm, rot)
    q = d.qpos.copy()
    q[:3] = C + Rm.reshape(3, 3) @ (q[:3] - C)
    mujoco.mju_mulQuat(q[3:7], rot, d.qpos[3:7].copy())
    w = 1.0 * axis
    qvel = np.zeros(m.nv)
    qvel[:3] = np.cross(w, q[:3] - C)
    hard.set_state(q, qvel)
    R_t = d.xmat[hard.trunk].reshape(3, 3)
    reading = dict(gyro=R_t.T @ w, accel=R_t.T @ np.array([0.0, 0.0, control.G]), q=d.qpos[hard.qadr].copy(),
                   qd=np.zeros(len(hard.joints)), foot_load=np.array([W, 0.0]))
    qvel[3:6] = R_t.T @ w
    est.reset(q, qvel, reading)
    est.update(0.01)
    assert est.contact_local[0][0] == pytest.approx(est.sole[0][:, 0].max())  # the front edge alone
    assert est.v_base[:2] == pytest.approx(qvel[:2], abs=1e-9)
    old = np.cross(w, d.xipos[f] - C)  # what taking the foot's centre of mass as still would get wrong
    assert np.linalg.norm(old[:2]) > 0.009
    hard.reset()


def test_inward_step_keeps_the_swing_pad_5_mm_from_the_stance_pad(hard, nominal_ref):
    """Decision 20 in the tracker: with the left foot's touchdown beside the right foot (120 mm), a DCM error 0.1 m
    to the right asks the left foot to land 0.1 m inward; the step stops where its pad is 5 mm from the stance pad.
    Lifted (the leg-to-leg clearance relaxation), the step is the full 0.1 m."""
    tk = _tracker(hard, nominal_ref, terms=("step",))
    tk.reset("frame")
    k = 30  # the left foot early in its swing
    assert not nominal_ref.support[k, 0] and nominal_ref.support[k, 1]
    hard.set_state(nominal_ref.q[k], nominal_ref.qd[k])
    t = k * nominal_ref.step
    info, view = tk.track.info, tk.state.view
    _, ref0 = info(t)
    assert np.all(np.isfinite(ref0["land_rel"])) and ref0["land_rel"][0] > 0.1  # it lands half a stride ahead

    def beside(tt):
        seg, ref = info(tt)
        return seg, dict(ref, land_rel=np.array([0.0, 0.12]), land_yaw=0.0)

    def pushed():
        v = view()
        pa, ya = tk._anchor_frame(v, int(ref0["anchor"]))
        v["com"] = v["com"].copy()
        v["com"][:2] = pa + control._rot2(ya) @ (ref0["dcm_rel"] + [0.0, -0.1]) - v["com_vel"][:2] / tk.track.omega
        return v

    tk.track.info, tk.state.view = beside, pushed
    tk.feedback(t, None)
    dfx, dfy = tk.step_off[0]
    assert tk.flags["leg_to_leg_clearance"] and -0.1 < dfy < 0.0
    gap = _poly_gap(tk.pads[1], tk.pads[0] + [0.0, 0.12] + [dfx, dfy])
    assert gap == pytest.approx(control.PAD_CLEARANCE, abs=1e-9)
    tk.inward_bound = None
    tk.feedback(t, None)
    assert tk.step_off[0][1] == pytest.approx(-0.1, abs=1e-9) and not tk.flags["leg_to_leg_clearance"]
    hard.reset()


def test_panel_attribution_rule():
    """Decision 20: a failing condition is body-limited by a limit only if lifting it from the start improves survival
    over the same starts significantly (one-sided exact sign test, p < 0.05) and the 1 mm/s null does not."""

    def fell(x):
        return dict(fell=True, strides=x)

    ok = dict(fell=False, strides=60.0)
    base = [fell(5.0)] * 6 + [ok] * 4
    better = [ok] * 10
    t = gates.sign_test(base, better)
    assert (t["better"], t["worse"], t["tied"]) == (6, 0, 4) and t["p"] == pytest.approx(1 / 64)
    mixed = [fell(6.0)] * 5 + [fell(4.0)] + [ok] * 3 + [fell(50.0)]
    t = gates.sign_test(base, mixed)  # 5 better, 2 worse
    assert (t["better"], t["worse"], t["tied"]) == (5, 2, 3) and t["p"] == pytest.approx(29 / 128)
    assert gates.sign_test(base, base) == dict(better=0, worse=0, tied=10, p=1.0)
    cands = ["slip", "leg_drive_at_envelope_limit", "cop_at_pad_edge_with_ankle_saturated"]
    a = gates.attribute(base, {"slip": better, "envelope": mixed}, base, cands, together=better)
    assert a["label"] == "body-limited: slip" and a["binding"] == ["slip"] and a["lifted"]["slip"]["significant"]
    assert a["not_relaxable"] == ["cop_at_pad_edge_with_ankle_saturated"] and a["together"]["significant"]
    assert a["relaxable"] == {"slip": "slip", "leg_drive_at_envelope_limit": "envelope"}
    # decision 20: a significant null is reported but does not veto a significant lift
    chaotic = gates.attribute(base, {"slip": better}, better, ["slip"])
    assert chaotic["label"] == "body-limited: slip" and chaotic["null"]["significant"]
    assert gates.attribute(base, {"slip": mixed}, base, ["slip"])["label"] == "controller-limited"
    assert gates.attribute(base, {}, base, ["cop_at_pad_edge_with_ankle_saturated"])["label"] == "controller-limited"


def test_run_records_name_their_code():
    prov = plant.provenance()
    assert set(prov) == {"versions", "git", "stage1_sources"} and set(prov["git"]) == {"commit", "dirty"}
    src = prov["stage1_sources"]
    files = {q.name: q.read_bytes() for q in plant.SOURCES.glob("*.py")}
    assert src["sha256"] == plant.sources_sha256() == plant.sources_sha256(files) and len(src["sha256"]) == 64
    assert src["files"] == sorted(files) and "control.py" in files
    files["control.py"] += b"\n"
    assert plant.sources_sha256(files) != src["sha256"]


@pytest.mark.parametrize("body", sorted(BODIES))
def test_g4_records(body):
    """The committed G4 record (decisions 15-20): judged at the tuning start with the panel rule, the unperturbed runs
    reported, each failing condition attributed over its starts, the earlier defaults' record kept as evidence."""
    doc = json.loads((RUN / body / "gates.json").read_text())
    g4 = doc["G4"]
    assert g4["judged_at"] == "start" and g4["rule_name"] == "panel" and g4["decided"]["follows_all"]
    assert [r["decision"] for r in g4["decided"]["decisions"]] == [15, 16, 17, 18, 19, 20]
    prov = g4["provenance"]  # the code that made it: commit, dirty flag, sources hash, unchanged while it ran
    assert prov["git"]["commit"] and prov["git"]["dirty"] is not None and len(prov["stage1_sources"]["sha256"]) == 64
    assert prov["stage1_sources"]["unchanged_during_run"] and doc["G3"]["provenance"]["stage1_sources"][
        "unchanged_during_run"]
    tr = g4["tracker"]
    assert {n: v["start"] for n, v in tr["gains"].items()} == control.gains_dict()
    assert {n: v["earlier_default"] for n, v in tr["gains"].items()} == control.gains_dict("earlier_defaults")
    assert tr["ankle_cap"]["parts"] == list(control.ANKLE_CAP) and tr["inward_bound"]["pad_clearance_m"] == 0.005
    assert tr["turn_gap"]["bounds_the_step"] is False and tr["falls"]["hardware_limits"] == list(control.HW_LIMITS)
    assert set(g4["gated"]) == set(gates.G4_THRESHOLDS)
    for key, g in g4["gated"].items():
        pnl, u = g["panel"], g["unperturbed"]
        assert g["threshold_strides"] == gates.G4_THRESHOLDS[key] and len(pnl["rows"]) == gates.PANEL_RUNS
        assert [r["strides"] for r in pnl["rows"]] == pnl["strides"]
        assert g["panel_pass"] == bool(np.median(pnl["strides"]) >= g["threshold_strides"]) == pnl[
            "median_at_least_threshold"]
        assert g["unperturbed_pass"] == (u["strides"] >= g["threshold_strides"])
        assert u["mode"] == key[:2] and u["ankle_cap"] == list(control.ANKLE_CAP) and u["inward_bound"] == "pad_clearance"
        for r in pnl["rows"] + [u]:
            assert r["fell"] == (r["strides"] < G4_RUN_STRIDES) and (r["fall_candidates"] is None) == (not r["fell"])
        assert (key in g4["attribution"]) == (not g["panel_pass"])  # decision 20: every failing condition, only those
    assert g4["pass"] == g4["panel_rule_pass"] == all(g["panel_pass"] for g in g4["gated"].values())
    assert g4["single_rule_pass"] == all(g["unperturbed_pass"] for g in g4["gated"].values())
    for key, a in g4["attribution"].items():
        rows = g4["gated"][key]["panel"]["rows"]
        cands = [c for c in control.HW_LIMITS if any(c in (r["fall_candidates"] or ()) for r in rows)]
        assert a["candidates"] == cands and a["rule"] == gates.ATTRIBUTION_RULE
        assert set(a["lifted"]) == {control.RELAXES[c] for c in cands if control.RELAXES[c]}

        def runs(strides):
            return [dict(fell=x < G4_RUN_STRIDES, strides=x) for x in strides]

        for t in list(a["lifted"].values()) + [a["null"]] + ([a["together"]] if a["together"] else []):
            assert len(t["strides"]) == gates.PANEL_RUNS
            assert t["sign_test"] == gates.sign_test(runs(a["baseline"]["strides"]), runs(t["strides"]))
            assert t["significant"] == (t["sign_test"]["p"] < gates.ATTRIBUTION_ALPHA)
        sig = [r for r, t in a["lifted"].items() if t["significant"]]
        assert a["label"] == ("body-limited: " + ", ".join(sig) if sig else "controller-limited")
    assert g4["R0"]["nominal"]["strides"] == doc["G2"]["nominal_strides_with_envelope"]  # R0 is G2's replay
    assert set(g4["R0"]) == {"nominal", *gates.G4_REPORTED} and set(g4["reported"]["R2"]) == set(gates.G4_REPORTED)
    for r in g4["stance_start_stop"].values():  # decision 18 on the start-walk-stop runs
        assert len(r["stops"]) == 1 and r["stops"][0]["name"] == "stop"
        if r["stops"][0]["reached"]:
            assert r["stops"][0]["rule"] == transitions.STOP_RULE
    ev = g4["evidence"]["earlier_defaults"]  # the untuned defaults' result, as recorded before decisions 15-20
    assert ev["record_names_code"] is False and ev["committed_in"].startswith("3731e98")
    at = plant.sources_sha256_at(ev["committed_in"])  # None where git does not hold that commit (a shallow clone)
    assert at is None or at == ev["committed_sources_sha256"]
    old = ev["record"]
    assert "judged_at" not in old and set(old["gated"]) == set(gates.G4_THRESHOLDS)
    assert {n: v["default"] for n, v in old["tracker"]["gains"].items()} == control.gains_dict("earlier_defaults")


def test_later_steps_are_not_implemented(tmp_path):
    with pytest.raises(NotImplementedError, match="not implemented yet"):
        cli(["tune", "--body", BODY, "--out", str(tmp_path)])


# ------------------------------------------------------------------------------------------------
# Full-length gate checks (STAGE1_SLOW=1)
# ------------------------------------------------------------------------------------------------
@slow
@pytest.mark.parametrize("body", sorted(BODIES))
@pytest.mark.parametrize("floor", gates.STAND_FLOORS)
def test_g1_stands_20_s_with_zero_command(body, floor):
    r = gates.stand(body, floor)
    assert r["stands"], r


@slow
@pytest.mark.parametrize("body", sorted(BODIES))
def test_g2_references_match_and_the_nominal_replay_reproduces_the_gate_record(body, tmp_path):
    doc = refs.prepare(body, tmp_path)
    assert doc["matches_records"]
    rec = json.loads((RUN / body / "gates.json").read_text())["G2"]
    path = tmp_path / doc["references"]["nominal"]["file"]
    ref = refs.Reference.load(path)
    r = gates.replay(plant.build(body, "hard"), ref)  # deterministic: exactly the recorded unperturbed run
    assert r["strides"] == rec["nominal_strides_with_envelope"]
    assert rec["single_run_pass"] == (r["strides"] >= gates.MIN_REPLAY_STRIDES)
    offsets = gates._offsets()  # and the gated panel (decision 13), run by run
    jobs = [dict(body=body, ref=str(path), plant_kw=dict(envelope=True), offsets=offsets[i : i + gates.PANEL_CHUNK],
                 strides=gates.REPLAY_STRIDES, full=False) for i in range(0, gates.PANEL_RUNS, gates.PANEL_CHUNK)]
    runs = [x["strides"] for chunk in gates.run_jobs(jobs) for x in chunk]
    assert runs == rec["nominal_panel_with_envelope"]["strides"]
    assert rec["panel_pass"] == (np.median(runs) >= gates.MIN_REPLAY_STRIDES) == rec["pass"]
