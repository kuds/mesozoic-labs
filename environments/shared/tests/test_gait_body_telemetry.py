"""The body columns of an evaluation episode (gait.body_telemetry) and the recorder's body captures: report-only.

Synthetic traces pin each reduction (the window, the definitions, the signs, the NaN rules and the invariants the
spec names: abs work >= |net work|, the group works summing to the whole animal's, hypot(fore-aft, lateral) = the
displacement, |signed yaw| = the unsigned, pitch min <= mean <= max); every manifest model pins what its layout reads
and why the rest reads NaN; and real episodes pin each capture to an independent MuJoCo calculation at the same
substep (the actuator force, the contact points' sliding speed, the joint-limit torque, the centroidal momentum,
the interaction wrench, the head's angular velocity), the drift to the screen's own StanceEpisodeMetrics bit for
bit, and the trajectory with every capture on to the one without them, qpos for qpos.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import replace

import numpy as np
import pytest

from environments.shared.episode_ledger import EpisodeLedger
from environments.shared.gait.balance_telemetry import heading_frames
from environments.shared.gait.body_telemetry import (
    ANGULAR_MOMENTUM_COLUMNS,
    COMMAND_SUMMARY_COLUMNS,
    DRIFT_COLUMNS,
    ENERGY_FIELDS,
    HEAD_RATE_COLUMNS,
    MARGIN_COLUMNS,
    PARKED_COMMAND_ABS,
    RAIL_STEP_SHARE,
    TAIL_CHORD_COLUMNS,
    BodyLayout,
    LimitJoint,
    episode_body_telemetry,
)
from environments.shared.gait.constants import CONTACT_THRESHOLD_N
from environments.shared.gait.posture_telemetry import PostureJoint
from environments.shared.gait.stance_metrics import episode_stance_metrics
from environments.shared.model_id import ModelId
from environments.shared.tests.test_gait_pad_telemetry import _trace

TYRANNOSAURUS_REX = ModelId("tyrannosaurus_rex", "anatomical")
STEPS = 400
SETTLE = 100
SUBSTEPS = 5
DT_SUB = 0.01 / SUBSTEPS


def _layout(**changes) -> BodyLayout:
    """Four actuators (two legs, a tail, a neck), two tail hinges and one that cannot be signed, two stops."""
    layout = BodyLayout(
        labels=("r", "l"),
        actuator_names=("r_leg_act", "l_leg_act", "tail_act", "neck_act"),
        force_limits=np.array([[-100.0, 200.0], [-100.0, 200.0], [-10.0, 10.0], [np.nan, np.nan]]),
        groups=(("legs", (0, 1)), ("tail", (2,)), ("neck_head", (3,))),
        foot_actuators=((0,), (1,)),
        tail_hinges=(
            ("tail_1_pitch", PostureJoint("tail_1_pitch", 0, 1.0, 0.1)),
            ("tail_1_yaw", PostureJoint("tail_1_yaw", 1, -1.0, 0.0)),
            ("tail_2_pitch", None),
        ),
        tail_tip_site=3,
        tail_tip_site_name="tail_tip",
        tail_base_local=np.array([-0.2, 0.0, 0.0]),
        elliptic=True,
        limit_joints=(LimitJoint("tail_1_pitch", 4, 0), LimitJoint("r_knee", 7, 2)),
        spanned=(True, True),
        mass_kg=10.0,
        gravity=9.81,
        wrench_joints=(("tail_1_pitch", 4), ("neck_pitch", 1)),
        head_body=("skull", 2),
        reasons={"tail_2_pitch_mean_deg": "tail_2_pitch cannot be signed"},
    )
    return replace(layout, **changes)


def _body_trace(steps: int = STEPS):
    """A synthetic statue with every body capture: set commands, angles, forces, slides, stops and loads."""
    rng = np.random.default_rng(7)
    trace = _trace(steps)
    action = np.zeros((steps, 4))
    action[:, 0] = 0.7  # parked +
    action[:, 1] = np.where(np.arange(steps) % 4 == 0, -1.0, -0.2)  # a quarter of the steps at the -1 rail
    action[:, 2] = rng.uniform(-0.3, 0.3, steps)
    action[:, 3] = -0.65  # parked -
    joints = np.zeros((steps, 3))
    joints[:, 0] = 0.3  # tail_1_pitch: home 0.1, so +0.2 rad (tip up)
    joints[:, 1] = 0.05  # tail_1_yaw: sign -1, so -0.05 rad (tip right)
    joints[:, 2] = -0.4
    # The root moves forward-left along a wiggle and turns left.
    t = np.arange(steps) / steps
    root = np.stack([0.3 * t, 0.1 * t + 0.01 * np.sin(40 * t), 0.9 + 0.01 * np.cos(10 * t)], axis=1)
    yaw = np.radians(20.0) * t
    pitch = np.radians(3.0) * np.sin(7 * t)
    quat = np.stack([np.cos(yaw / 2) * np.cos(pitch / 2), -np.sin(yaw / 2) * np.sin(pitch / 2),
                     np.cos(yaw / 2) * np.sin(pitch / 2), np.sin(yaw / 2) * np.cos(pitch / 2)], axis=1)  # fmt: skip
    tip = root + np.array([-1.5, 0.0, 0.1])
    force = rng.normal(0.0, 50.0, (steps, SUBSTEPS, 4))
    velocity = rng.normal(0.0, 0.5, (steps, SUBSTEPS, 4))
    slip = np.full((steps, SUBSTEPS, 2), 0.01)
    slip[::3, :, 1] = np.nan  # the left foot unloaded on every third step
    stops = np.zeros((steps, SUBSTEPS, 2))
    stops[: steps // 2, :, 0] = 30.0  # the tail on its lower stop through the first half
    momentum = np.zeros((steps, 6))
    momentum[:, 1] = rng.normal(0.0, 2.0, steps)  # angular momentum about world y
    momentum[:, 3] = 0.5  # linear momentum along x: 0.05 m/s at 10 kg
    com = root - np.array([0.0, 0.0, 0.1])
    lines = np.zeros((steps, 2, 2, 3))
    for foot, offset in enumerate((-0.15, 0.15)):
        lines[:, foot, 0] = root + np.array([-0.1, offset, -0.9])  # heel lines 0.1 m behind the root
        lines[:, foot, 1] = root + np.array([0.2, offset, -0.9])  # toe lines 0.2 m ahead
    wrench = np.zeros((steps, 2, 6))
    wrench[:, 0] = [0.0, 70.0, 0.5, 0.0, 0.0, 150.0]  # tail: holds 150 N up, +70 N m about world y
    wrench[:, 1] = [0.0, -40.0, 0.0, 0.0, 0.0, 130.0]  # neck: -40 N m about y
    angvel = np.zeros((steps, 1, 3))
    angvel[:, 0, 1] = np.radians(10.0) * np.sin(np.arange(steps))
    trace = replace(
        trace,
        action=action,
        joint_qpos=joints,
        root_pos=root,
        root_quat=quat,
        site_pos=tip[:, None, :],
        site_names=("tail_tip",),
        actuator_force_substep=force,
        actuator_velocity_substep=velocity,
        leg_slip_speed=slip,
        joint_limit_torque=stops,
        joint_limit_names=("tail_1_pitch", "r_knee"),
        centroidal_momentum=momentum,
        com_pos=com,
        spawn_com_pos=com[0],
        support_line_pos=lines,
        joint_wrench=wrench,
        wrench_joint_names=("tail_1_pitch", "neck_pitch"),
        body_angvel=angvel,
        angvel_body_names=("skull",),
    )
    return trace


def _out(trace=None, layout=None, settle=SETTLE):
    return episode_body_telemetry(_body_trace() if trace is None else trace, layout or _layout(), settle_steps=settle)


class TestTheColumns:
    def test_the_columns_come_in_the_specs_order(self):
        columns = _layout().columns()
        assert columns[:8] == (
            "r_leg_act_command_mean",
            "l_leg_act_command_mean",
            "tail_act_command_mean",
            "neck_act_command_mean",
            *COMMAND_SUMMARY_COLUMNS,
        )
        posture = columns[8:21]
        assert posture[:3] == ("tail_1_pitch_mean_deg", "tail_1_yaw_mean_deg", "tail_2_pitch_mean_deg")
        assert posture[3:6] == TAIL_CHORD_COLUMNS and "root_pitch_mean_deg" not in posture
        energy = columns[21:42]
        assert energy[:6] == tuple(f"legs_{name}" for name in ENERGY_FIELDS)
        assert energy[-3:] == ("action_delta_mean", "r_leg_work_abs_j", "l_leg_work_abs_j")
        drift = columns[42:54]
        assert drift[:6] == DRIFT_COLUMNS and drift[6:9] == ("r_slip_m", "r_slip_speed_max_mps", "r_site_travel_m")
        assert columns[54:58] == (
            "tail_1_pitch_stop_share",
            "tail_1_pitch_stop_torque_mean_nm",
            "r_knee_stop_share",
            "r_knee_stop_torque_mean_nm",
        )
        assert columns[58:66] == MARGIN_COLUMNS and columns[66:72] == ANGULAR_MOMENTUM_COLUMNS
        assert columns[72] == "tail_base_force_down_mean_n" and columns[-4:] == HEAD_RATE_COLUMNS
        assert len(set(columns)) == len(columns) == 86
        # Without the balance group, the root's pitch mean joins the posture columns, before its range.
        pitched = _layout(root_pitch_mean=True).posture_columns()
        assert pitched[pitched.index("root_pitch_mean_deg") + 1] == "root_pitch_min_deg"
        assert list(_out()) == list(columns)

    def test_an_empty_window_reads_nan_and_empty_text(self):
        trace = _body_trace(SETTLE)
        out = _out(trace)
        assert out["rail_actuators"] == out["parked_actuators"] == ""
        assert all(math.isnan(value) for key, value in out.items() if not isinstance(value, str))


class TestCommands:
    def test_means_rails_and_parked_actuators(self):
        trace = _body_trace()
        out = _out(trace)
        window = trace.action[SETTLE:]
        for index, name in enumerate(_layout().actuator_names):
            assert out[f"{name}_command_mean"] == pytest.approx(float(window[:, index].mean()), rel=1e-12, abs=1e-15)
        # The left leg sits at -1 on a quarter of the window: more than RAIL_STEP_SHARE, so at rail (-), not parked.
        assert 0.25 > RAIL_STEP_SHARE
        assert out["actuators_at_rail"] == 1.0 and out["rail_actuators"] == "l_leg_act-"
        assert abs(out["r_leg_act_command_mean"]) >= PARKED_COMMAND_ABS
        assert out["actuators_parked"] == 2.0 and out["parked_actuators"] == "r_leg_act+;neck_act-"

    def test_a_rail_tie_reads_plus_and_a_missing_action_reads_nan(self):
        trace = _body_trace()
        action = trace.action.copy()
        action[:, 2] = np.where(np.arange(STEPS) % 2 == 0, 1.0, -1.0)  # half +1, half -1: a tie
        out = _out(replace(trace, action=action))
        assert out["rail_actuators"] == "l_leg_act-;tail_act+"
        action[SETTLE + 5] = np.nan  # end_step(None) on a window step
        missing = _out(replace(trace, action=action))
        assert math.isnan(missing["tail_act_command_mean"]) and missing["rail_actuators"] == ""
        assert math.isnan(missing["action_delta_mean"]) and math.isnan(missing["legs_command_delta_rms"])
        action = trace.action.copy()
        action[SETTLE - 3] = np.nan  # before the window and its first difference: not read
        assert _out(replace(trace, action=action))["action_delta_mean"] == _out(trace)["action_delta_mean"]


class TestPosture:
    def test_tail_hinges_chord_and_root(self):
        trace = _body_trace()
        out = _out(trace)
        window = slice(SETTLE, STEPS)
        assert out["tail_1_pitch_mean_deg"] == pytest.approx(math.degrees(0.2), rel=1e-12)
        assert out["tail_1_yaw_mean_deg"] == pytest.approx(math.degrees(-0.05), rel=1e-12)
        assert math.isnan(out["tail_2_pitch_mean_deg"])  # its reason: it cannot be signed
        # The chord from the base (0.2 m behind the root, carried by the root pose) to the tip.
        _forward, pitch = heading_frames(trace.root_quat)
        w, x, y, z = trace.root_quat.T
        back = -0.2 * np.stack([1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)], axis=1)
        chord = trace.site_pos[:, 0] - (trace.root_pos + back)
        angles = np.degrees(np.arctan2(chord[:, 2], np.hypot(chord[:, 0], chord[:, 1])))
        assert out["tail_tip_chord_mean_deg"] == pytest.approx(float(angles[window].mean()), rel=1e-12)
        assert out["tail_tip_chord_min_deg"] == pytest.approx(float(angles[window].min()), rel=1e-12)
        assert out["root_height_mean_m"] == float(trace.root_pos[window, 2].mean())
        assert out["root_height_min_m"] == float(trace.root_pos[window, 2].min())
        pitched = _out(trace, _layout(root_pitch_mean=True))
        assert pitched["root_pitch_min_deg"] <= pitched["root_pitch_mean_deg"] <= pitched["root_pitch_max_deg"]
        assert pitched["root_pitch_mean_deg"] == float(pitch[window].mean())  # balance_telemetry's own pitch
        roll = np.degrees(np.arcsin(2 * (y * z + w * x)))
        assert out["root_roll_mean_deg"] == pytest.approx(float(roll[window].mean()), abs=1e-12)

    def test_the_chord_reads_nan_without_the_tail_tip_capture(self):
        out = _out(replace(_body_trace(), site_pos=None, site_names=()))
        assert all(math.isnan(out[column]) for column in TAIL_CHORD_COLUMNS)


class TestEnergy:
    def test_work_power_torque_and_jitter_per_group(self):
        trace = _body_trace()
        out = _out(trace)
        force = trace.actuator_force_substep[SETTLE:]
        power = force * trace.actuator_velocity_substep[SETTLE:]
        for group, indices in _layout().groups:
            part = power[..., list(indices)]
            assert out[f"{group}_work_abs_j"] == pytest.approx(float(np.abs(part).sum() * DT_SUB), rel=1e-12)
            assert out[f"{group}_work_net_j"] == pytest.approx(float(part.sum() * DT_SUB), rel=1e-12)
            assert out[f"{group}_work_abs_j"] >= abs(out[f"{group}_work_net_j"])
            assert out[f"{group}_power_peak_w"] == pytest.approx(float(np.abs(part).sum(axis=-1).max()), rel=1e-12)
            rms = np.sqrt(np.mean(force[..., list(indices)] ** 2))
            assert out[f"{group}_torque_rms_nm"] == pytest.approx(float(rms), rel=1e-12)
        # The group works sum to the whole animal's; the legs' to the two legs'.
        total = sum(out[f"{group}_work_abs_j"] for group, _indices in _layout().groups)
        assert total == pytest.approx(float(np.abs(power).sum() * DT_SUB), rel=1e-12)
        assert out["legs_work_abs_j"] == pytest.approx(out["r_leg_work_abs_j"] + out["l_leg_work_abs_j"], rel=1e-12)
        # The torque fraction: side-aware limits (+200 / -100 on the legs); the neck has none.
        legs = force[..., :2]
        fraction = np.where(legs >= 0.0, legs / 200.0, -legs / 100.0)
        assert out["legs_torque_rms_frac"] == pytest.approx(float(np.sqrt(np.mean(fraction**2))), rel=1e-12)
        assert math.isnan(out["neck_head_torque_rms_frac"])
        # The command jitter: each window step against the one before it, the first against step settle - 1.
        deltas = np.diff(trace.action[SETTLE - 1 :], axis=0)
        assert out["action_delta_mean"] == pytest.approx(float(np.mean(np.sum(deltas**2, axis=1))), rel=1e-12)
        assert out["tail_command_delta_rms"] == pytest.approx(float(np.sqrt(np.mean(deltas[:, 2] ** 2))), rel=1e-12)

    def test_without_the_actuator_capture_only_the_command_columns_are_read(self):
        out = _out(replace(_body_trace(), actuator_force_substep=None, actuator_velocity_substep=None))
        assert math.isnan(out["legs_work_abs_j"]) and math.isnan(out["r_leg_work_abs_j"])
        assert math.isfinite(out["legs_command_delta_rms"]) and math.isfinite(out["action_delta_mean"])


class TestDriftAndSlip:
    def test_the_drift_is_the_screens_and_its_parts_add_up(self):
        trace = _body_trace()
        out = _out(trace)
        stance = episode_stance_metrics(trace, settle_steps=SETTLE)
        assert out["window_displacement_m"] == stance.window_displacement_m  # copied, bit for bit
        assert out["window_yaw_change_deg"] == stance.window_yaw_change_deg
        assert abs(out["window_yaw_change_signed_deg"]) == out["window_yaw_change_deg"]
        assert out["window_yaw_change_signed_deg"] > 0.0  # it turns left
        assert math.hypot(out["window_displacement_fore_aft_m"], out["window_displacement_lateral_m"]) == (
            pytest.approx(out["window_displacement_m"], abs=1e-12)
        )
        root = trace.root_pos[SETTLE - 1 :, :2]
        assert out["window_path_length_m"] == pytest.approx(
            float(np.linalg.norm(np.diff(root, axis=0), axis=1).sum()), rel=1e-12
        )
        assert out["window_path_length_m"] > out["window_displacement_m"]  # the wiggle
        # The heading at the window's start: forward is turned 20 deg * (settle - 1) / steps to the left.
        heading = math.radians(20.0) * (SETTLE - 1) / STEPS
        move = root[-1] - root[0]
        forward = move @ [math.cos(heading), math.sin(heading)]
        assert out["window_displacement_fore_aft_m"] == pytest.approx(forward, abs=1e-12)
        # The given screen metrics are copied rather than recomputed.
        given = replace(stance, window_displacement_m=1.25, window_yaw_change_deg=3.5)
        copied = episode_body_telemetry(trace, _layout(), settle_steps=SETTLE, stance=given)
        assert (copied["window_displacement_m"], copied["window_yaw_change_deg"]) == (1.25, 3.5)

    def test_slip_and_site_travel(self):
        trace = _body_trace()
        out = _out(trace)
        assert out["r_slip_m"] == pytest.approx(0.01 * (STEPS - SETTLE) * SUBSTEPS * DT_SUB, rel=1e-12)
        loaded = np.isfinite(trace.leg_slip_speed[SETTLE:, :, 1]).sum()
        assert out["l_slip_m"] == pytest.approx(0.01 * loaded * DT_SUB, rel=1e-12)
        assert out["r_slip_speed_max_mps"] == 0.01
        stance = episode_stance_metrics(trace, settle_steps=SETTLE)
        assert max(out["r_site_travel_m"], out["l_site_travel_m"]) == stance.max_foot_slide_m
        unloaded = trace.leg_slip_speed.copy()
        unloaded[:, :, 0] = np.nan  # never loaded: it slid 0 while loaded, its top speed is unmeasured
        idle = _out(replace(trace, leg_slip_speed=unloaded))
        assert idle["r_slip_m"] == 0.0 and math.isnan(idle["r_slip_speed_max_mps"])
        pyramidal = _out(replace(trace, leg_slip_speed=None))
        assert math.isnan(pyramidal["r_slip_m"]) and math.isfinite(pyramidal["r_site_travel_m"])


class TestTheExtras:
    def test_joint_stops(self):
        out = _out()
        assert out["tail_1_pitch_stop_share"] == pytest.approx((STEPS / 2 - SETTLE) / (STEPS - SETTLE), rel=1e-12)
        assert out["tail_1_pitch_stop_torque_mean_nm"] == pytest.approx(30.0 * out["tail_1_pitch_stop_share"])
        assert out["r_knee_stop_share"] == 0.0 == out["r_knee_stop_torque_mean_nm"]

    def test_com_and_capture_point_margins_and_the_angular_momentum(self):
        trace = _body_trace()
        out = _out(trace)
        window = slice(SETTLE, STEPS)
        forward, _pitch = heading_frames(trace.root_quat)
        along = np.einsum("tj,tj->t", forward, trace.com_pos[:, :2])
        toe = np.einsum("tj,tsj->ts", forward, trace.support_line_pos[:, :, 1, :2]).mean(axis=1)
        heel = np.einsum("tj,tsj->ts", forward, trace.support_line_pos[:, :, 0, :2]).mean(axis=1)
        assert out["com_toe_margin_min_m"] == pytest.approx(float((toe - along)[window].min()), abs=1e-12)
        assert out["com_heel_margin_mean_m"] == pytest.approx(float((along - heel)[window].mean()), abs=1e-12)
        # The capture point: 0.05 m/s along x, over omega0 = sqrt(g / h).
        omega = np.sqrt(9.81 / trace.com_pos[:, 2])
        capture = trace.com_pos[:, :2] + np.stack([0.05 / omega, np.zeros(STEPS)], axis=1)
        captured = np.einsum("tj,tj->t", forward, capture)
        assert out["capture_toe_margin_mean_m"] == pytest.approx(float((toe - captured)[window].mean()), abs=1e-12)
        assert out["capture_toe_margin_min_m"] < out["com_toe_margin_min_m"]  # moving toward the toes
        left = np.stack([-forward[:, 1], forward[:, 0]], axis=1)
        pitch = np.einsum("tj,tj->t", trace.centroidal_momentum[:, :2], left)[window]
        assert out["angular_momentum_pitch_rms_nms"] == pytest.approx(float(np.sqrt(np.mean(pitch**2))), rel=1e-12)
        assert out["angular_momentum_yaw_rms_nms"] == 0.0
        unspanned = _out(trace, _layout(spanned=(False, False)))
        assert all(math.isnan(unspanned[column]) for column in MARGIN_COLUMNS)
        assert math.isfinite(unspanned["angular_momentum_pitch_rms_nms"])

    def test_the_base_wrenches_signs(self):
        trace = _body_trace()
        out = _out(trace)
        forward, _pitch = heading_frames(trace.root_quat)
        left = np.stack([-forward[:, 1], forward[:, 0]], axis=1)[SETTLE:]
        # The parent holds the tail up (150 N) and against its weight's moment: the tail pitches it nose up.
        assert out["tail_base_force_down_mean_n"] == 150.0
        assert out["tail_base_pitch_moment_mean_nm"] == pytest.approx(float(np.mean(left @ [0.0, 70.0])), rel=1e-12)
        assert out["tail_base_pitch_moment_mean_nm"] > 0.0 > out["neck_base_pitch_moment_mean_nm"]
        assert out["tail_base_yaw_moment_mean_nm"] == -0.5  # minus the parent's +0.5 N m about z: turns it right
        assert out["tail_base_pitch_moment_std_nm"] == pytest.approx(float(np.std(left @ [0.0, 70.0])), abs=1e-12)

    def test_the_heads_angular_rate(self):
        trace = _body_trace()
        out = _out(trace)
        rate = np.degrees(trace.body_angvel[SETTLE:, 0])
        assert out["head_angular_rate_rms_dps"] == pytest.approx(float(np.sqrt(np.mean(rate[:, 1] ** 2))), rel=1e-12)
        assert out["head_yaw_rate_rms_dps"] == 0.0
        components = (out[f"head_{axis}_rate_rms_dps"] ** 2 for axis in ("pitch", "roll", "yaw"))
        assert sum(components) == pytest.approx(out["head_angular_rate_rms_dps"] ** 2, rel=1e-9)

    def test_each_capture_missing_costs_its_columns_only(self):
        trace = _body_trace()
        full = _out(trace)
        for field, columns in (
            ("joint_limit_torque", list(_layout().stop_columns())),
            ("centroidal_momentum", [*MARGIN_COLUMNS, *ANGULAR_MOMENTUM_COLUMNS]),
            ("joint_wrench", [column for column in _layout().dynamics_columns() if "_base_" in column]),
            ("body_angvel", list(HEAD_RATE_COLUMNS)),
        ):
            out = _out(replace(trace, **{field: None}))
            assert all(math.isnan(out[column]) for column in columns), field
            kept = [key for key in full if key not in columns and not isinstance(full[key], str)]
            assert all(out[key] == full[key] or math.isnan(full[key]) for key in kept), field


#: Per recorded model (and the two that run no recorder): its groups' sizes, tail hinges with their signs, stop
#: joints, wrench joints, head body and NaN reasons' columns.  Measured from the models' MJCF on 2026-10-25
#: (T. rex physics r10).
EXPECTED = {
    ModelId("tyrannosaurus_rex", "anatomical"): {
        "groups": {"legs": 8, "tail": 4, "neck_head": 3},
        "tail": {"tail_1_pitch": 1.0, "tail_1_yaw": -1.0, "tail_2_pitch": 1.0, "tail_3_pitch": 1.0},
        "stops": 12,
        "wrench": ("tail_1_pitch", "neck_pitch"),
        "head": "skull",
        "reasons": set(),
    },
    ModelId("compsognathus_longipes", "anatomical"): {
        "groups": {"legs": 10, "tail": 2, "neck_head": 2},
        "tail": {
            "tail_1_pitch": 1.0,
            "tail_1_yaw": -1.0,
            "tail_2_pitch": 1.0,
            "tail_3_pitch": 1.0,
            "tail_4_pitch": 1.0,
        },
        "stops": 15,
        "wrench": ("tail_1_pitch", "neck_pitch"),
        "head": "skull",
        "reasons": set(MARGIN_COLUMNS),
    },
    ModelId("compsognathus_longipes", "robot"): {
        "groups": {"legs": 12, "tail": 0, "neck_head": 0},
        "tail": {},
        "stops": 12,
        "wrench": ("", ""),
        "head": "fixed_head",
        "reasons": {
            *(f"{group}_{name}" for group in ("tail", "neck_head") for name in ENERGY_FIELDS),
            *TAIL_CHORD_COLUMNS,
            *MARGIN_COLUMNS,
            *(
                f"{base}_{name}"
                for base in ("tail_base", "neck_base")
                for name in (
                    "force_down_mean_n",
                    "pitch_moment_mean_nm",
                    "pitch_moment_std_nm",
                    "roll_moment_mean_nm",
                    "yaw_moment_mean_nm",
                )
            ),
        },  # fmt: skip
    },
    ModelId("dibothrosuchus_elaphros", "anatomical"): {
        "groups": {"legs": 20, "tail": 4, "neck_head": 3},
        "tail": {
            "tail_1_pitch": 1.0,
            "tail_1_yaw": -1.0,
            "tail_2_pitch": 1.0,
            "tail_3_pitch": 1.0,
            "tail_4_pitch": 1.0,
        },
        "stops": 25,
        "wrench": ("tail_1_pitch", "neck_pitch"),
        "head": "skull",
        "reasons": set(MARGIN_COLUMNS),
    },
}


@pytest.mark.parametrize("model_id", list(EXPECTED), ids=str)
def test_each_recorded_model_reads_what_it_has_and_says_why_not_the_rest(model_id):
    pytest.importorskip("mujoco")
    import mujoco

    from environments.shared.config import load_stage_config
    from environments.shared.gait.morphology import Morphology
    from environments.shared.species_registry import env_class

    env = env_class(model_id)(**load_stage_config(model_id, "stance")["env_kwargs"])
    try:
        morphology = Morphology.from_env(env.unwrapped, model_id)
        layout = BodyLayout.from_env(env, morphology)
        expected = EXPECTED[model_id]
        sizes = {group: len(indices) for group, indices in layout.groups}
        assert sizes == expected["groups"] and sum(sizes.values()) == env.unwrapped.model.nu  # no "other" group
        assert {name: joint.sign for name, joint in layout.tail_hinges if joint is not None} == expected["tail"]
        assert len(layout.limit_joints) == expected["stops"]
        assert tuple(name for name, _joint in layout.wrench_joints) == expected["wrench"]
        assert layout.head_body[0] == expected["head"]
        assert set(layout.reasons) == expected["reasons"]
        assert set(layout.reasons) <= set(layout.columns()) and len(set(layout.columns())) == len(layout.columns())
        # Each leg's actuators lie on its own chain; the legs are the union.
        legs = dict(layout.groups)["legs"]
        assert sorted(index for indices in layout.foot_actuators for index in indices) == sorted(legs)
        # The tail signs by moving the hinge: + raises the tip (pitch) or moves it left (yaw), at the keyframe.
        model = env.unwrapped.model
        data = mujoco.MjData(model)
        if model.nkey:
            mujoco.mj_resetDataKeyframe(model, data, int(getattr(env.unwrapped, "_reset_keyframe_id", 0)))
        mujoco.mj_kinematics(model, data)
        home = data.qpos.copy()
        left = data.xmat[morphology.root_body].reshape(3, 3)[:, 1].copy()
        tip = layout.tail_tip_site
        for name, joint in layout.tail_hinges:
            data.qpos[:] = home
            data.qpos[model.jnt_qposadr[model.joint(name).id]] += 0.05
            mujoco.mj_kinematics(model, data)
            moved = data.site_xpos[tip].copy()
            data.qpos[:] = home
            mujoco.mj_kinematics(model, data)
            shift = moved - data.site_xpos[tip]
            along = shift[2] if name.endswith("pitch") else float(np.dot(shift, left))
            assert math.copysign(1.0, along) == joint.sign, name
    finally:
        env.close()


def test_a_ledger_written_before_the_body_columns_keeps_its_rows_and_gains_them_last(tmp_path):
    """A same-stage resume from a ledger without the body columns (an r9 or a pre-change run): rows kept, columns
    appended after the ones it has, its old rows blank there."""
    prior = EpisodeLedger(tmp_path / "eval_episodes.csv", fixed_columns=("evaluation", "timesteps"))
    prior.extend([{"evaluation": 0, "timesteps": 1000, "head_tip_height_mean_m": 1.1}])
    assert prior.write()
    ledger = EpisodeLedger(tmp_path / "eval_episodes.csv", fixed_columns=("evaluation", "timesteps"))
    assert ledger.seed_prior(tmp_path / "eval_episodes.csv", max_timesteps=1000) == 1
    body = {column: 0.5 for column in _layout().columns()}
    body.update(rail_actuators="", parked_actuators="tail_act+")
    ledger.extend([{"evaluation": 1, "timesteps": 2000, "head_tip_height_mean_m": 1.0, **body}])
    assert ledger.write()
    rows = ledger.rows()
    assert ledger.columns == ("evaluation", "timesteps", "head_tip_height_mean_m", *_layout().columns())
    assert rows[0]["tail_act_command_mean"] == "" and rows[1]["parked_actuators"] == "tail_act+"


# ---------------------------------------------------------------------- real episodes: the captures
def _env(model_id):
    from environments.shared.config import load_stage_config
    from environments.shared.species_registry import env_class

    return env_class(model_id)(**load_stage_config(model_id, "stance")["env_kwargs"])


class SubstepReference:
    """A substep hook chained ON TOP of the recorder's: it reads, after each mj_step, what the recorder read, and
    computes each capture independently with MuJoCo's own functions on a scratch copy (never on env.data)."""

    def __init__(self, env, morphology, layout) -> None:
        import mujoco

        self.env, self.morphology, self.layout = env, morphology, layout
        self.model = env.model
        self.copy = mujoco.MjData(self.model)
        self.rows: list[dict] = []
        self.step_substeps = 0
        self.previous: Callable[[], None] = lambda: None  # the recorder's hook, set once it is attached

    def __call__(self) -> None:
        import mujoco

        self.previous()
        model, data, morphology = self.model, self.env.data, self.morphology
        row: dict = {"force": data.actuator_force.copy()}
        speed, load, buffer = np.zeros(morphology.n_feet), np.zeros(morphology.n_feet), np.zeros(6)
        for index in range(data.ncon):
            contact = data.contact[index]
            first, second = int(contact.geom[0]), int(contact.geom[1])
            if morphology.is_floor[first] == morphology.is_floor[second]:
                continue
            other = second if morphology.is_floor[first] else first
            if not morphology.is_animal[other] or morphology.geom_foot[other] < 0:
                continue
            mujoco.mj_contactForce(model, data, index, buffer)
            normal = float(buffer[0])
            # The animal geom's material point at the contact (the floor is static): w x (p - geom_xpos) + v.
            mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_GEOM, other, buffer, 0)
            point = buffer[3:] + np.cross(buffer[:3], contact.pos - data.geom_xpos[other])
            tangent = point - np.dot(point, contact.frame[:3]) * contact.frame[:3]
            foot = int(morphology.geom_foot[other])
            speed[foot] += normal * float(np.linalg.norm(tangent))
            load[foot] += normal
        loaded = load > CONTACT_THRESHOLD_N
        row["slip"] = np.where(loaded, speed / np.where(loaded, load, 1.0), np.nan)
        dense = np.asarray(data.efc_J).reshape(data.nefc, model.nv) if data.nefc else np.zeros((0, model.nv))
        torque = np.zeros(len(self.layout.limit_joints))
        for column, joint in enumerate(self.layout.limit_joints):
            rows = np.flatnonzero(
                (np.asarray(data.efc_type) == int(mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT))
                & (np.asarray(data.efc_id) == joint.joint)
            )
            dof = int(model.jnt_dofadr[joint.joint])
            torque[column] = float(np.sum(dense[rows, dof] * np.asarray(data.efc_force)[rows]))
        row["limit"] = torque
        self.step_substeps += 1
        if self.step_substeps == self.env.frame_skip:
            self.step_substeps = 0
            mujoco.mj_copyData(self.copy, model, data)  # the copy keeps every derived field of this substep
            mujoco.mj_subtreeVel(model, self.copy)
            root = int(morphology.root_body)
            row["momentum"] = np.concatenate(
                [self.copy.subtree_angmom[root], self.copy.subtree_linvel[root] * model.body_subtreemass[root]]
            )
            wrench = []
            for _name, joint in self.layout.wrench_joints:
                if joint < 0:
                    continue
                body = int(model.jnt_bodyid[joint])
                torque_c, force = data.cfrc_int[body, :3], data.cfrc_int[body, 3:]
                arm = data.subtree_com[root] - data.xanchor[joint]
                wrench.append(np.concatenate([torque_c + np.cross(arm, force), force]))
            row["wrench"] = np.array(wrench)
            mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, self.layout.head_body[1], buffer, 0)
            row["head"] = buffer[:3].copy()
        self.rows.append(row)


def _roll(model_id, *, captures: bool, steps: int = 80, seed: int = 3042, scale: float = 0.4, cone=None, range_=None):
    """One recorded episode of random commands; with *captures*, every body capture and a SubstepReference on top."""
    pytest.importorskip("mujoco")
    from environments.shared.gait.morphology import Morphology
    from environments.shared.gait.posture_telemetry import PostureLayout
    from environments.shared.gait.recorder import SubstepContactRecorder

    env = _env(model_id)
    try:
        unwrapped = env.unwrapped
        if cone is not None:
            unwrapped.model.opt.cone = cone
        if range_ is not None:
            name, low, high = range_
            unwrapped.model.jnt_range[unwrapped.model.joint(name).id] = (low, high)
        morphology = Morphology.from_env(unwrapped, model_id)
        layout = BodyLayout.from_env(env, morphology)
        posture = PostureLayout.from_env(env, morphology)
        recorder = SubstepContactRecorder(unwrapped, morphology, balance=True)
        reference = None
        if captures:
            recorder.record_sites(sorted(set(posture.sites()) | set(layout.sites())))
            layout.capture(recorder)
        qpos = []
        rng = np.random.default_rng(seed)
        recorder.attach()
        if captures:
            reference = SubstepReference(unwrapped, morphology, layout)
            reference.previous = unwrapped._substep_probe_hook
            unwrapped._substep_probe_hook = reference
        try:
            env.reset(seed=seed)
            for _ in range(steps):
                action = np.clip(scale * rng.standard_normal(unwrapped.model.nu), -1.0, 1.0)
                _obs, reward, terminated, truncated, _info = env.step(action)
                recorder.end_step(action, reward)
                qpos.append(unwrapped.data.qpos.copy())
                if terminated or truncated:
                    break
            trace = recorder.end_episode(terminated=terminated, truncated=truncated)
        finally:
            if reference is not None:
                unwrapped._substep_probe_hook = reference.previous
            recorder.detach()
        return trace, layout, np.asarray(qpos), reference
    finally:
        env.close()


RECORDED = list(EXPECTED)


@pytest.mark.parametrize("model_id", RECORDED, ids=str)
def test_the_captures_are_mujocos_own_numbers_at_the_same_substep_and_change_no_step(model_id):
    # Commands of SD 0.1 slide every model's feet and keep each standing past the 20-step window start.
    trace, layout, qpos, reference = _roll(model_id, captures=True, scale=0.1)
    plain, _layout_, plain_qpos, _reference = _roll(model_id, captures=False, scale=0.1)
    np.testing.assert_array_equal(qpos, plain_qpos)  # every capture on: the same trajectory, qpos for qpos
    for name in ("joint_qpos", "root_pos", "root_quat", "foot_pos", "leg_floor_mean", "reward", "com_pos"):
        np.testing.assert_array_equal(getattr(trace, name), getattr(plain, name), err_msg=name)
    assert plain.actuator_force_substep is None and plain.joint_wrench is None and plain.leg_slip_speed is None
    steps, substeps = trace.length, int(trace.frame_skip)
    rows = reference.rows
    assert len(rows) == steps * substeps
    force = np.array([row["force"] for row in rows]).reshape(steps, substeps, -1)
    np.testing.assert_array_equal(trace.actuator_force_substep, force)  # data.actuator_force, every substep
    # The sliding speed from efc_vel's tangential rows is the contact points' own, to 1e-12 (elliptic cone).
    slip = np.array([row["slip"] for row in rows]).reshape(steps, substeps, -1)
    assert np.isfinite(slip).any()
    np.testing.assert_array_equal(np.isnan(trace.leg_slip_speed), np.isnan(slip))
    np.testing.assert_allclose(trace.leg_slip_speed, slip, rtol=0, atol=1e-12, equal_nan=True)
    limit = np.array([row["limit"] for row in rows]).reshape(steps, substeps, -1)
    np.testing.assert_allclose(trace.joint_limit_torque, limit, rtol=0, atol=1e-9)
    boundary = [row for row in rows if "momentum" in row]
    np.testing.assert_allclose(
        trace.centroidal_momentum, np.array([row["momentum"] for row in boundary]), rtol=0, atol=1e-10
    )
    if trace.joint_wrench is not None:
        np.testing.assert_allclose(trace.joint_wrench, np.array([row["wrench"] for row in boundary]), atol=1e-9)
    np.testing.assert_allclose(trace.body_angvel[:, 0], np.array([row["head"] for row in boundary]), atol=1e-12)
    out = episode_body_telemetry(trace, layout, settle_steps=min(20, steps - 1))
    assert set(out) == set(layout.columns())
    for column, value in out.items():
        if column in layout.reasons:
            assert value == "" or math.isnan(value), column
        elif not isinstance(value, str) and not column.endswith("_slip_speed_max_mps"):
            assert math.isfinite(value), column  # (a foot never loaded in the window has no top slip speed)
    assert any(math.isfinite(out[f"{label}_slip_speed_max_mps"]) for label in layout.labels)


@pytest.mark.parametrize("side", ["lower", "upper"])
def test_a_joint_on_its_stop_reads_its_limit_torque_signed_by_the_side(side):
    """The T. rex tail_1 pitch range moved in memory to one side of where the springs hold it (about 0 on physics
    r10): above it, the tail rests on its lower stop and is pushed up (+); below it, on its upper stop (-)."""
    pytest.importorskip("mujoco")
    low, high = (0.05, 0.1) if side == "lower" else (-0.1, -0.05)
    trace, layout, qpos, reference = _roll(
        TYRANNOSAURUS_REX, captures=True, steps=160, scale=0.0, range_=("tail_1_pitch", low, high)
    )
    column = trace.joint_limit_names.index("tail_1_pitch")
    torque = trace.joint_limit_torque[60:, :, column]
    sign = 1.0 if side == "lower" else -1.0
    assert (sign * torque > 0.0).all()  # on its stop at every substep after the spawn transient, pushed back
    np.testing.assert_allclose(  # MuJoCo's own: efc_J * efc_force at the joint's dof (SubstepReference)
        trace.joint_limit_torque,
        np.array([row["limit"] for row in reference.rows]).reshape(trace.joint_limit_torque.shape),
        atol=1e-9,
    )
    out = episode_body_telemetry(trace, layout, settle_steps=60)
    assert out["tail_1_pitch_stop_share"] == 1.0 and sign * out["tail_1_pitch_stop_torque_mean_nm"] > 5.0
    assert out["r_knee_stop_share"] == 0.0


def test_under_a_pyramidal_cone_slip_is_unmeasured_and_said_so():
    pytest.importorskip("mujoco")
    import mujoco

    trace, layout, _qpos, _reference = _roll(
        ModelId("dibothrosuchus_elaphros", "anatomical"), captures=True, steps=30, cone=mujoco.mjtCone.mjCONE_PYRAMIDAL
    )
    assert trace.leg_slip_speed is None and not layout.elliptic
    out = episode_body_telemetry(trace, layout, settle_steps=10)
    for label in layout.labels:
        assert math.isnan(out[f"{label}_slip_m"]) and "pyramidal" in layout.reasons[f"{label}_slip_m"]
        assert math.isfinite(out[f"{label}_site_travel_m"])


def test_the_tex_statues_base_wrenches_carry_the_tail_and_the_neck():
    """Standing still, each base carries its subtree's weight, and the neck's moment is TyrannosaurusRexEnv's own
    figure (neck_pitch gravity torque 42.85 N m, ``_cache_ids``)."""
    trace, layout, _qpos, _reference = _roll(TYRANNOSAURUS_REX, captures=True, steps=260, scale=0.0)
    out = episode_body_telemetry(trace, layout, settle_steps=200)
    assert out["tail_base_force_down_mean_n"] == pytest.approx(15.6 * 9.81, rel=5e-3)  # the tail's 15.6 kg
    assert out["neck_base_force_down_mean_n"] == pytest.approx(13.7 * 9.81, rel=5e-3)  # neck and skull, 13.7 kg
    assert out["neck_base_pitch_moment_mean_nm"] == pytest.approx(-42.85, abs=0.5)  # pitches the pelvis nose down
    assert out["tail_base_pitch_moment_mean_nm"] > 0.0  # the tail counterweight pitches it nose up
    assert out["tail_1_pitch_stop_share"] == 0.0  # physics r10: the springs hold the tail off its stop


def test_the_body_captures_are_refused_once_attached_and_for_joints_they_cannot_read():
    pytest.importorskip("mujoco")
    from environments.shared.gait.morphology import Morphology
    from environments.shared.gait.recorder import RecorderError, SubstepContactRecorder

    env = _env(TYRANNOSAURUS_REX)
    try:
        unwrapped = env.unwrapped
        recorder = SubstepContactRecorder(unwrapped, Morphology.from_env(unwrapped, TYRANNOSAURUS_REX))
        root_joint = 0  # the free joint: not a limited hinge
        with pytest.raises(RecorderError):
            recorder.record_joint_limits([root_joint])
        with pytest.raises(RecorderError):
            recorder.record_joint_limits([unwrapped.model.njnt])
        with pytest.raises(RecorderError):
            recorder.record_body_velocity([1, 1])  # not distinct
        assert recorder.record_joint_limits([]) is recorder
        with recorder:
            for method, argument in (
                ("record_actuators", ()),
                ("record_slip", ()),
                ("record_momentum", ()),
                ("record_joint_limits", ([4],)),
                ("record_wrenches", ([4],)),
                ("record_body_velocity", ([4],)),
            ):
                with pytest.raises(RecorderError):
                    getattr(recorder, method)(*argument)
    finally:
        env.close()


def test_a_failing_body_capture_costs_only_the_body_fields(monkeypatch, caplog):
    pytest.importorskip("mujoco")
    import logging

    from environments.shared.gait import recorder as recorder_module

    def raising(self, data, contact):
        raise FloatingPointError("a body capture went wrong")

    monkeypatch.setattr(recorder_module.SubstepContactRecorder, "_body_capture", raising)
    with caplog.at_level(logging.WARNING, logger=recorder_module.__name__):
        trace, _layout_, qpos, _reference = _roll(TYRANNOSAURUS_REX, captures=True, steps=40, scale=0.0)
    plain, _layout_, plain_qpos, _reference = _roll(TYRANNOSAURUS_REX, captures=False, steps=40, scale=0.0)
    np.testing.assert_array_equal(qpos, plain_qpos)
    assert trace.actuator_force_substep is None and trace.centroidal_momentum is None
    assert trace.com_pos is not None and trace.site_names == ("head_tip", "tail_tip")  # balance and sites kept
    assert sum("Body telemetry" in record.getMessage() for record in caplog.records) == 1
