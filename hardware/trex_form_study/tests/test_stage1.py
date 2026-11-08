"""Tests for the Stage 1 plant, drives and references (designlab/stage1; STAGE1_PLAN.md sections 2 and 3).

Run from hardware/trex_form_study:  python -m pytest tests/test_stage1.py -q
The default run is fast (shortened stand, one reference).  The full-length gate checks (20 s stands on hard floor and
low pile, every reference against its records, the feedforward replays) take minutes and run only with
STAGE1_SLOW=1 (the repository registers no slow marker).
"""

from __future__ import annotations

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
from designlab.stage1 import BODIES, drive, gates, plant, refs  # noqa: E402
from designlab.stage1.__main__ import main as cli  # noqa: E402

BODY = "60-N-T-10"
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


# ------------------------------------------------------------------------------------------------
# Hashes
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


# ------------------------------------------------------------------------------------------------
# Drives: tendon ankle and torque-speed envelope
# ------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("body", sorted(BODIES))
def test_tendon_map_equals_sizing_motor_space(body):
    r = gates.tendon_map(body, samples=60, seed=3)
    assert r["matches"]
    assert max(r["max_err_joint_torque_Nm"], r["max_err_motor_speed_rad_s"], r["max_err_plant_matrix_torque_Nm"],
               r["max_err_plant_matrix_speed_rad_s"]) <= 1e-9
    for side in statics.SIDES:  # |tau_p| + |tau_r| reaches 2 peak on each axis and never exceeds it
        a = r["ankle_limit"][side]
        assert a["within"] and a["reaches"]
        assert a["max_abs_pitch_plus_abs_roll_Nm"] == pytest.approx(2 * a["peak_Nm"], abs=1e-9)


def test_envelope_shape():
    peak, w1, w0 = 7.0, 4.0, 6.0
    w = np.array([0.0, 2.0, 4.0, 5.0, 6.0, 9.0])
    expect = [7.0, 7.0, 7.0, 3.5, 0.0, 0.0]
    assert drive.torque_limit(w, peak, w1, w0) == pytest.approx(expect)
    assert drive.torque_limit(-w, peak, w1, w0) == pytest.approx(expect)  # the same in both directions


def test_envelope_follows_the_catalog_on_the_24_v_bus(hard):
    sel = plant.record(BODY)["selection"]
    assert sel["bus"] == 24.0
    for d in hard.drives:
        a = catalog.actuator(d.part)
        assert d.peak == a.peak
        assert d.rated_speed == a.speed_at(None if a.box is not None else 24.0)
        nl = a.no_load_speed if a.box is not None else a.no_load_speed * min(1.0, 24.0 / a.nominal_voltage)
        assert d.no_load_speed == pytest.approx(nl, rel=1e-12)
        assert d.no_load_speed > d.rated_speed
    x6 = catalog.actuator("myactuator_rmd_x6_p20_60_e")  # 48 V drive on the 24 V bus: both speeds halve
    assert drive.no_load_speed_at(x6, 24.0) == pytest.approx(0.5 * x6.no_load_speed)


def test_envelope_is_applied_from_each_drives_own_speed(hard):
    m, d, env = hard.m, hard.d, hard.envelope
    hard.reset()
    i_knee = hard.joints.index("left_knee")
    i_a = hard.joints.index("right_ankle_pitch")  # right ankle motor a turns at w_p + w_r
    i_r = hard.joints.index("right_ankle_roll")
    knee, ank = hard.drives[i_knee], hard.drives[i_a]
    d.qvel[hard.dofs[i_knee]] = 0.5 * (knee.rated_speed + knee.no_load_speed)
    d.qvel[hard.dofs[i_a]] = d.qvel[hard.dofs[i_r]] = 0.5 * (0.25 * ank.rated_speed + 0.75 * ank.no_load_speed)
    env.update()
    lim = m.actuator_forcerange[:, 1].copy()
    assert lim[i_knee] == pytest.approx(0.5 * knee.peak)
    assert lim[i_a] == pytest.approx(0.25 * ank.peak)  # motor a: 3/4 of the way from rated to no-load speed
    assert lim[i_r] == pytest.approx(ank.peak)  # motor b: w_p - w_r = 0
    others = np.setdiff1d(np.arange(m.nu), [i_knee, i_a])
    assert lim[others] == pytest.approx(env.peak[others])
    assert np.array_equal(m.actuator_forcerange[:, 0], -lim)
    hard.reset()
    env.update()  # back below every corner: the peaks are restored
    assert np.array_equal(m.actuator_forcerange[:, 1], env.peak)


# ------------------------------------------------------------------------------------------------
# Plant
# ------------------------------------------------------------------------------------------------
def test_hold_gains_follow_carpet_hold_servos(hard):
    xml, _ = plant.model_files(BODY)
    m0 = mujoco.MjModel.from_xml_path(str(xml))
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
    assert p.meta["gains"]["ankle_gains"] == "joint"
    for side in statics.SIDES:
        ia, ib = (p.joints.index(f"{side}_{j}") for j in ("ankle_pitch", "ankle_roll"))
        Gs = p.G[np.ix_([ia, ib], [ia, ib])]
        K = Gs.T @ np.diag(p.kp[[ia, ib]]) @ Gs
        assert K == pytest.approx(p.drives[ia].sustained / carpet.SERVO_ERROR_RAD * np.eye(2))
    with pytest.raises(ValueError):
        plant.build(BODY, "hard", ankle_gains="box", settle=False)


def test_joint_ranges_are_the_usable_ranges(hard):
    usable = plant.usable_ranges(BODY)
    for jn in hard.joints:
        j = mujoco.mj_name2id(hard.m, mujoco.mjtObj.mjOBJ_JOINT, jn)
        assert tuple(hard.m.jnt_range[j]) == pytest.approx(usable[jn], abs=1e-12)
    assert usable["left_knee"][1] == pytest.approx(1.0509)
    assert usable["left_hip_roll"] == pytest.approx((-0.2261, 0.2512))


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


def test_75cm_body_settles_into_contact():
    p = plant.build("75-M-H-10", "hard")
    s = p.meta["settle"]
    assert s["foot_contacts_at_keyframe"] == [0, 0]  # the keyframe pads sit 0.3 um above the floor
    assert min(s["foot_contacts_settled"]) > 0
    assert sum(s["foot_load_N"]) == pytest.approx(s["weight_N"], rel=0.01)


def test_seeded_runs_repeat_exactly():
    r = gates.repeatability(BODY, seed=5)
    assert r["repeats"] and r["other_seed_differs"]


def test_zero_command_stand_on_hard_floor_shortened():
    r = gates.stand(BODY, "hard", seconds=3.0)
    assert r["stands"] and r["fell_at_s"] is None
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
    assert checks[0]["leaves"] > 80


def test_record_comparison_catches_a_difference(nominal):
    src, g = nominal
    rec = plant.record(BODY)["gaits"]["nominal"]
    s = dict(g["summary"], min_cop_margin_single=g["summary"]["min_cop_margin_single"] * (1 + 2e-6))
    r = refs.compare(s, rec)
    assert not r["match"] and r["mismatched"] == ["min_cop_margin_single"]
    assert refs.compare(dict(g["summary"], min_cop_margin_single=g["summary"]["min_cop_margin_single"] * (1 + 1e-7)),
                        rec)["match"]


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


#: G2 fails on 60-N-H-10 (stage1/runs/2026-11-08/60-N-H-10/gates.json): the unperturbed open-loop nominal replay on the
#: default plant falls after 3.7 strides; strict, so a change that makes it pass is noticed.
G2_KNOWN_FAIL = pytest.mark.xfail(strict=True, reason="G2 replay below 10 strides on 60-N-H-10 (see gates.json)")


@slow
@pytest.mark.parametrize("body", [pytest.param(b, marks=G2_KNOWN_FAIL) if b == "60-N-H-10" else b
                                  for b in sorted(BODIES)])
def test_g2_every_reference_matches_and_the_nominal_replay_walks(body, tmp_path):
    doc = refs.prepare(body, tmp_path)
    assert doc["matches_records"]
    ref = refs.Reference.load(tmp_path / doc["references"]["nominal"]["file"])
    r = gates.replay(plant.build(body, "hard"), ref)
    assert r["strides"] >= gates.MIN_REPLAY_STRIDES, r
