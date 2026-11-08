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
import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import designlab  # noqa: E402,F401  (selects the GL backend before mujoco renders)
from designlab import carpet, catalog, mjcf, statics  # noqa: E402
from designlab.stage1 import BODIES, drive, gates, legs, plant, refs  # noqa: E402
from designlab.stage1.__main__ import main as cli  # noqa: E402

BODY = "60-N-T-10"
RUN = ROOT / "stage1" / "runs" / "2026-11-08"
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
