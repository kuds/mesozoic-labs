"""The head posture and hip roll of an evaluation episode (gait.posture_telemetry): report-only.

Synthetic traces pin the reduction (the window, the signs, the home offsets,
the sum over a long neck, the NaN rules); every manifest model pins what its
layout reads and why the rest reads NaN; the model geometry pins the sign
conventions (+ nose down, + abduction on either side); and a real T. rex
episode pins the head tip to the env's own ``info["head_tip_z"]`` and the
pitches to the joints' own angles, with the recorder's site capture leaving
the trajectory bit for bit what it was.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from environments.shared.gait.posture_telemetry import (
    HEAD_POSTURE_COLUMNS,
    HIP_ROLL_FIELDS,
    PostureJoint,
    PostureLayout,
    episode_posture_telemetry,
)
from environments.shared.model_id import ModelId
from environments.shared.tests.test_gait_pad_telemetry import _trace

TYRANNOSAURUS_REX = ModelId("tyrannosaurus_rex", "anatomical")
STEPS = 400
SETTLE = 100

#: Every manifest model: its head-tip site, neck pitch hinges, head pitch hinge and hip roll hinge per foot with the
#: abduction sign (None: not read).  Measured from the models' MJCF on 2026-10-25 (physics r10 for the T. rex).
EXPECTED = {
    ModelId("tyrannosaurus_rex", "anatomical"): (
        "head_tip",
        ("neck_pitch",),
        "head_pitch",
        {"r": ("r_hip_roll", -1.0), "l": ("l_hip_roll", 1.0)},
    ),
    ModelId("compsognathus_longipes", "anatomical"): (
        "head_tip",
        ("neck_pitch",),
        None,
        {"r": ("r_hip_roll", -1.0), "l": ("l_hip_roll", 1.0)},
    ),
    ModelId("compsognathus_longipes", "robot"): (
        "head_tip",
        (),
        None,
        {"r": ("right_hip_roll", 1.0), "l": ("left_hip_roll", -1.0)},
    ),
    ModelId("brachiosaurus_altithorax", "anatomical"): (
        "head_tip",
        ("neck_1_pitch", "neck_2_pitch", "neck_3_pitch", "neck_4_pitch"),
        "head_pitch",
        {
            "fr": ("fr_hip_roll", -1.0),
            "fl": ("fl_hip_roll", 1.0),
            "rr": ("rr_hip_roll", -1.0),
            "rl": ("rl_hip_roll", 1.0),
        },
    ),
    ModelId("dibothrosuchus_elaphros", "anatomical"): (
        "snout_tip",
        ("neck_pitch",),
        "head_pitch",
        {
            "fr": ("fr_hip_roll", -1.0),
            "fl": ("fl_hip_roll", 1.0),
            "rr": ("rr_hip_roll", -1.0),
            "rl": ("rl_hip_roll", 1.0),
        },
    ),
    ModelId("velociraptor_mongoliensis", "anatomical"): (
        None,
        (),
        None,
        {"r": ("r_hip_roll", -1.0), "l": ("l_hip_roll", 1.0)},
    ),
}


def _layout(**changes) -> PostureLayout:
    """A two-foot layout over a 6-column joint_qpos: two neck hinges (one signed -1), a head hinge, two hip rolls."""
    layout = PostureLayout(
        labels=("r", "l"),
        head_site=7,
        head_site_name="head_tip",
        neck_joints=(PostureJoint("neck_1_pitch", 0, 1.0, 0.0), PostureJoint("neck_2_pitch", 1, -1.0, 0.1)),
        head_joint=PostureJoint("head_pitch", 2, 1.0, 0.0),
        hip_roll=(PostureJoint("r_hip_roll", 3, -1.0, 0.0), PostureJoint("l_hip_roll", 4, 1.0, 0.05)),
    )
    return replace(layout, **changes)


def _posture_trace(steps: int = STEPS):
    """A synthetic statue with the head tip recorded: its height ramps, its joints hold set angles."""
    joints = np.zeros((steps, 6))
    joints[:, 0] = 0.2  # neck 1: +0.2 rad
    joints[:, 1] = 0.0  # neck 2: home 0.1, sign -1 -> +0.1 rad
    joints[:, 2] = -0.05  # head: -0.05 rad (nose up)
    joints[:, 3] = np.linspace(-0.1, 0.1, steps)  # right hip roll swings through home
    joints[:, 4] = 0.15  # left hip roll: 0.1 rad from its home 0.05, abducted
    heights = np.linspace(1.0, 0.5, steps)
    sites = np.zeros((steps, 1, 3))
    sites[:, 0, 2] = heights
    return replace(_trace(steps), joint_qpos=joints, site_pos=sites, site_names=("head_tip",)), heights


class TestTheReduction:
    def test_the_window_means_signs_and_home_offsets(self):
        trace, heights = _posture_trace()
        out = episode_posture_telemetry(trace, _layout(), settle_steps=SETTLE)
        window = slice(SETTLE, STEPS)
        assert list(out) == [*HEAD_POSTURE_COLUMNS, *(f"{s}_{f}" for s in "rl" for f in HIP_ROLL_FIELDS)]
        assert out["head_tip_height_mean_m"] == float(heights[window].mean())
        assert out["head_tip_height_min_m"] == float(heights[window].min()) == 0.5
        # The neck is the sum of its hinges, each signed and offset from its home: 0.2 + (-(0.0 - 0.1)) = 0.3 rad.
        assert out["neck_pitch_mean_deg"] == pytest.approx(math.degrees(0.3), rel=1e-12)
        assert out["head_pitch_mean_deg"] == pytest.approx(math.degrees(-0.05), rel=1e-12)
        roll = -trace.joint_qpos[window, 3]  # the right leg's sign is -1, its home 0
        assert out["r_hip_roll_mean_deg"] == pytest.approx(math.degrees(roll.mean()), rel=1e-12)
        assert out["r_hip_roll_max_abs_deg"] == pytest.approx(math.degrees(np.abs(roll).max()), rel=1e-12)
        assert out["l_hip_roll_mean_deg"] == pytest.approx(math.degrees(0.1), rel=1e-12)
        assert out["l_hip_roll_max_abs_deg"] == pytest.approx(math.degrees(0.1), rel=1e-12)

    def test_the_window_starts_at_the_settle_step(self):
        trace, heights = _posture_trace()
        early = trace.joint_qpos.copy()
        early[:SETTLE, :] = 9.0  # the settle transient is outside the window
        sites = trace.site_pos.copy()
        sites[:SETTLE, 0, 2] = -5.0
        out = episode_posture_telemetry(
            replace(trace, joint_qpos=early, site_pos=sites), _layout(), settle_steps=SETTLE
        )
        assert out == episode_posture_telemetry(trace, _layout(), settle_steps=SETTLE)

    def test_an_empty_window_reads_nan_everywhere(self):
        trace, _ = _posture_trace(SETTLE)
        out = episode_posture_telemetry(trace, _layout(), settle_steps=SETTLE)
        assert all(math.isnan(value) for value in out.values())
        with pytest.raises(ValueError):
            episode_posture_telemetry(trace, _layout(), settle_steps=1.5)

    def test_what_the_model_or_the_trace_lacks_reads_nan_and_the_rest_is_measured(self):
        trace, _ = _posture_trace()
        full = episode_posture_telemetry(trace, _layout(), settle_steps=SETTLE)
        # No site capture: the head tip is unmeasured, nothing else moves.
        bare = episode_posture_telemetry(replace(trace, site_pos=None, site_names=()), _layout(), settle_steps=SETTLE)
        assert math.isnan(bare["head_tip_height_mean_m"]) and math.isnan(bare["head_tip_height_min_m"])
        assert {k: v for k, v in bare.items() if "head_tip" not in k} == {
            k: v for k, v in full.items() if "head_tip" not in k
        }
        # A model with no head site, neck, head hinge or right hip roll.
        lacking = _layout(
            head_site=-1,
            head_site_name=None,
            neck_joints=(),
            head_joint=None,
            hip_roll=(None, _layout().hip_roll[1]),
        )
        out = episode_posture_telemetry(trace, lacking, settle_steps=SETTLE)
        assert lacking.sites() == () and _layout().sites() == (7,)
        assert all(math.isnan(out[key]) for key in (*HEAD_POSTURE_COLUMNS, "r_hip_roll_mean_deg"))
        assert out["l_hip_roll_mean_deg"] == full["l_hip_roll_mean_deg"]
        # A NaN step (an unread joint) is skipped, as the balance telemetry skips one.
        holes = trace.joint_qpos.copy()
        holes[SETTLE + 3, 2] = np.nan
        out = episode_posture_telemetry(replace(trace, joint_qpos=holes), _layout(), settle_steps=SETTLE)
        assert out["head_pitch_mean_deg"] == pytest.approx(full["head_pitch_mean_deg"], rel=1e-12)


def _model_layout(model_id):
    pytest.importorskip("mujoco")
    from environments.shared.gait.morphology import Morphology
    from environments.shared.species_registry import env_class

    env = env_class(model_id)()
    morphology = Morphology.from_env(env.unwrapped, model_id)
    return env, morphology, PostureLayout.from_env(env, morphology)


@pytest.mark.parametrize("model_id", sorted(EXPECTED), ids=str)
def test_every_model_reads_what_it_has_and_says_why_the_rest_reads_nan(model_id):
    env, morphology, layout = _model_layout(model_id)
    try:
        model = env.unwrapped.model
        site, neck, head, hips = EXPECTED[model_id]
        assert layout.head_site_name == site and (layout.head_site >= 0) == (site is not None)
        assert tuple(joint.name for joint in layout.neck_joints) == neck
        assert (layout.head_joint.name if layout.head_joint else None) == head
        assert layout.labels == morphology.labels
        assert {
            label: (joint.name, joint.sign) for label, joint in zip(layout.labels, layout.hip_roll) if joint
        } == hips
        # Every pitch hinge reads + nose down: its axis points left (sign +1) on every model.
        assert all(joint.sign == 1.0 for joint in (*layout.neck_joints, *([layout.head_joint] if head else [])))
        # Each hinge's column is its qpos in EpisodeTrace.joint_qpos (every qpos entry but the root's seven).
        from environments.shared.gait.recorder import joint_qpos_addresses

        addresses = joint_qpos_addresses(model, morphology.root_body)
        for joint in (*layout.neck_joints, *(j for j in layout.hip_roll if j), *([layout.head_joint] if head else [])):
            assert addresses[joint.column] == model.jnt_qposadr[model.joint(joint.name).id], joint.name
        # The columns are the same set on every model with the same feet; what a model lacks says why.
        assert layout.columns() == HEAD_POSTURE_COLUMNS + tuple(
            f"{label}_{name}" for label in morphology.labels for name in HIP_ROLL_FIELDS
        )
        lacking = set()
        if site is None:
            lacking |= set(HEAD_POSTURE_COLUMNS[:2])
        if not neck:
            lacking.add("neck_pitch_mean_deg")
        if head is None:
            lacking.add("head_pitch_mean_deg")
        assert set(layout.reasons) == lacking
        assert all(reason for reason in layout.reasons.values())
    finally:
        env.close()


@pytest.mark.parametrize("model_id", sorted(EXPECTED), ids=str)
def test_the_signs_are_the_bodys_own_nose_down_and_abduction(model_id):
    """From the home keyframe: + on a pitch hinge lowers the head tip, + on a hip roll moves the foot outward."""
    import mujoco

    env, morphology, layout = _model_layout(model_id)
    try:
        model = env.unwrapped.model
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, 0)
        mujoco.mj_kinematics(model, data)
        home = data.qpos.copy()
        left = data.xmat[morphology.root_body].reshape(3, 3)[:, 1].copy()
        left[2] = 0.0
        left /= np.linalg.norm(left)

        def moved(joint: PostureJoint, site: int) -> np.ndarray:
            data.qpos[:] = home
            data.qpos[model.jnt_qposadr[model.joint(joint.name).id]] += joint.sign * 0.05
            mujoco.mj_kinematics(model, data)
            position = np.array(data.site_xpos[site], dtype=np.float64)
            data.qpos[:] = home
            mujoco.mj_kinematics(model, data)
            shift: np.ndarray = position - np.asarray(data.site_xpos[site], dtype=np.float64)
            return shift

        for joint in (*layout.neck_joints, *([layout.head_joint] if layout.head_joint else [])):
            assert moved(joint, layout.head_site)[2] < 0.0, joint.name  # the head tip goes down
        for foot, joint in zip(morphology.feet, layout.hip_roll):
            if joint is None:
                continue
            outward = left if foot.side != "R" else -left
            assert float(np.dot(moved(joint, foot.primary_site), outward)) > 0.0, joint.name  # the foot goes out
    finally:
        env.close()


class TestARealEpisode:
    def _roll(self, *, sites: bool, steps: int = 260):
        pytest.importorskip("mujoco")
        from environments.shared.config import load_stage_config
        from environments.shared.gait.morphology import Morphology
        from environments.shared.gait.recorder import SubstepContactRecorder
        from environments.shared.species_registry import env_class

        config = load_stage_config(TYRANNOSAURUS_REX, "stance")
        env = env_class(TYRANNOSAURUS_REX)(**config["env_kwargs"])
        morphology = Morphology.from_env(env.unwrapped, TYRANNOSAURUS_REX)
        layout = PostureLayout.from_env(env, morphology)
        recorder = SubstepContactRecorder(env.unwrapped, morphology)
        if sites:
            recorder.record_sites(layout.sites())
        heads, qpos = [], []
        try:
            with recorder:
                env.reset(seed=3042)
                for _ in range(steps):
                    action = np.zeros(env.action_space.shape[0])
                    _obs, reward, terminated, truncated, info = env.step(action)
                    recorder.end_step(action, reward)
                    heads.append(info["head_tip_z"])
                    qpos.append(env.unwrapped.data.qpos.copy())
                    if terminated or truncated:
                        break
                trace = recorder.end_episode(terminated=terminated, truncated=truncated)
        finally:
            env.close()
        return trace, layout, np.asarray(heads), np.asarray(qpos), env.unwrapped.model

    def test_the_head_tip_is_the_envs_own_reading_and_the_pitches_the_joints_own_angles(self):
        trace, layout, heads, qpos, model = self._roll(sites=True)
        assert trace.site_names == ("head_tip",) and trace.site_pos.shape == (trace.length, 1, 3)
        # Bit for bit the env's info["head_tip_z"] (the head_clearance term's reading; z on the flat floor).
        np.testing.assert_array_equal(trace.site_pos[:, 0, 2], heads)
        settle = 200
        out = episode_posture_telemetry(trace, layout, settle_steps=settle)
        assert out["head_tip_height_mean_m"] == float(heads[settle:].mean())
        assert out["head_tip_height_min_m"] == float(heads[settle:].min())
        for column, name in (("neck_pitch_mean_deg", "neck_pitch"), ("head_pitch_mean_deg", "head_pitch")):
            angles = qpos[settle:, model.jnt_qposadr[model.joint(name).id]]
            assert out[column] == pytest.approx(math.degrees(float(angles.mean())), rel=1e-12, abs=1e-12), column
        for label, sign in (("r", -1.0), ("l", 1.0)):
            angles = sign * qpos[settle:, model.jnt_qposadr[model.joint(f"{label}_hip_roll").id]]
            assert out[f"{label}_hip_roll_mean_deg"] == pytest.approx(math.degrees(float(angles.mean())), rel=1e-12)
        assert all(math.isfinite(value) for value in out.values())

    def test_the_site_capture_changes_no_step_of_the_episode(self):
        captured, _layout_, _heads, qpos, _model = self._roll(sites=True, steps=120)
        plain, _layout_, _heads, plain_qpos, _model = self._roll(sites=False, steps=120)
        np.testing.assert_array_equal(qpos, plain_qpos)
        assert plain.site_pos is None and plain.site_names == ()
        for name in ("joint_qpos", "root_pos", "root_quat", "foot_pos", "leg_floor_mean", "sole_cop", "reward"):
            np.testing.assert_array_equal(getattr(captured, name), getattr(plain, name), err_msg=name)


def test_record_sites_is_refused_once_attached_and_for_a_site_the_model_lacks():
    pytest.importorskip("mujoco")
    from environments.shared.gait.morphology import Morphology
    from environments.shared.gait.recorder import RecorderError, SubstepContactRecorder
    from environments.shared.species_registry import env_class

    env = env_class(TYRANNOSAURUS_REX)()
    try:
        recorder = SubstepContactRecorder(env.unwrapped, Morphology.from_env(env.unwrapped, TYRANNOSAURUS_REX))
        with pytest.raises(RecorderError):
            recorder.record_sites([env.unwrapped.model.nsite])
        with pytest.raises(RecorderError):
            recorder.record_sites([-1])
        assert recorder.record_sites([]) is recorder
        with recorder:
            with pytest.raises(RecorderError):
                recorder.record_sites([0])
    finally:
        env.close()
