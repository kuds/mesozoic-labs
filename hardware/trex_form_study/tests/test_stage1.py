"""Tests for the Stage 1 plant, drives and references (designlab/stage1; STAGE1_PLAN.md sections 2 and 3).

Run from hardware/trex_form_study:  python -m pytest tests/test_stage1.py -q
The default run is fast (shortened stand, one reference, short replays).  The full-length gate checks (20 s stands
on hard floor and low pile, every reference against its records, the full nominal replay) take minutes and run only
with STAGE1_SLOW=1 (the repository registers no slow marker).
"""

from __future__ import annotations

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
    for side in statics.SIDES:  # each ankle motor: its own sustained torque at 0.01 rad of its own coordinate
        ia, ib = (hard.joints.index(f"{side}_{j}") for j in ("ankle_pitch", "ankle_roll"))
        for i in (ia, ib):
            assert hard.kp[i] == pytest.approx(hard.drives[i].sustained / carpet.SERVO_ERROR_RAD)
        K = hard.G[np.ix_([ia, ib], [ia, ib])].T @ np.diag(hard.kp[[ia, ib]]) @ hard.G[np.ix_([ia, ib], [ia, ib])]
        assert K == pytest.approx(2 * hard.kp[ia] * np.eye(2))  # joint space: two motors share each axis


def test_joint_space_ankle_gain_option_gives_one_hold_servo_per_axis():
    p = plant.build(BODY, "hard", ankle_gains="joint", settle=False)
    assert p.meta["gains"]["ankle_gains"] == "joint" and p.meta["options"]["ankle_gains"] == "joint"
    for side in statics.SIDES:
        ia, ib = (p.joints.index(f"{side}_{j}") for j in ("ankle_pitch", "ankle_roll"))
        Gs = p.G[np.ix_([ia, ib], [ia, ib])]
        K = Gs.T @ np.diag(p.kp[[ia, ib]]) @ Gs
        assert K == pytest.approx(p.drives[ia].sustained / carpet.SERVO_ERROR_RAD * np.eye(2))
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
    assert eff["choice"] == "linkage"
    assert eff["feet"]["left"]["time_constant_contact_stiffness_ratio"] == pytest.approx(1.10, abs=0.01)
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


def test_toe_lowest_point_is_exact(hard):
    rng = np.random.default_rng(2)
    for _ in range(20):
        q = hard.q_home.copy()
        q[hard.qadr] += rng.uniform(-0.4, 0.4, len(hard.joints))
        hard.set_state(q, np.zeros(hard.m.nv))
        z, g = hard.toe_lowest()
        assert z == pytest.approx(min(carpet.lowest_point(hard.m, hard.d, t)[2] for t in hard._toes), abs=1e-12)
        assert "toe" in g or "claw" in g
    hard.reset()


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
        _, info = carpet.pile_model(xml, m0.key_qpos[0].copy(), ref)  # the same probes under the same feet
        assert floors[pile].meta["floor_info"]["probes_per_foot"] == info["probes_per_foot"]


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
def test_replay_feedforward_carries_the_plants_extra_armature():
    ref = refs.Reference.load(RUN / BODY / "refs" / "nominal.npz")
    lk = gates.replay(plant.build(BODY, "hard", settle=False), ref, strides=1.0)
    sc = gates.replay(plant.build(BODY, "hard", settle=False, ankle_armature="screened"), ref, strides=1.0)
    off = gates.replay(plant.build(BODY, "hard", settle=False), ref, strides=1.0, armature_ff=False)
    assert lk["armature_feedforward"] and not sc["armature_feedforward"] and not off["armature_feedforward"]
    for r in (lk, sc, off):
        assert not r["fell"] and r["strides"] == pytest.approx(1.0, abs=1e-3)
        assert not r["joint_limit_reached"] and r["range_margin_worst"][2] > 0.1
        assert r["toe_lowest_above_floor_mm"] > -20.0 and r["envelope_clamps"] == {}


def test_leg_gap_on_the_reference_walk():
    lg = legs.LegGap(BODY)
    ref = refs.Reference.load(RUN / BODY / "refs" / "nominal.npz")
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
    ref = refs.Reference.load(tmp_path / doc["references"]["nominal"]["file"])
    r = gates.replay(plant.build(body, "hard"), ref)  # deterministic: exactly the recorded single run
    assert r["strides"] == rec["nominal_strides_with_envelope"]
    assert rec["single_run_pass"] == (r["strides"] >= gates.MIN_REPLAY_STRIDES)
