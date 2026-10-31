"""Engineering smoke checks for the standalone Mini T-Rex example.

These bound a home hold and supported gesture, not walking, fabrication fit,
continuous motor torque or a registered training embodiment.
"""

import mujoco
import numpy as np
import pytest

from examples.mini_tyrannosaurus_rex import demo


def _assert_finite_state(data):
    for values in (data.qpos, data.qvel, data.sensordata, data.actuator_force):
        assert np.isfinite(values).all()


def _assert_no_warnings(data):
    warnings = {
        mujoco.mjtWarning(i).name: int(warning.number) for i, warning in enumerate(data.warning) if warning.number
    }
    assert not warnings, warnings


def _assert_bounded_actuator_forces(model, data):
    assert model.actuator_forcelimited.all()
    assert np.all(data.actuator_force >= model.actuator_forcerange[:, 0] - 1e-10)
    assert np.all(data.actuator_force <= model.actuator_forcerange[:, 1] + 1e-10)


def test_compiled_mini_tyrannosaurus_rex_has_documented_interface_and_mass():
    model, data = demo.load_model()

    assert (model.nq, model.nv, model.nu, model.nsensor) == (22, 21, 14, 39)
    dynamic_mass = float(model.body_mass[model.body_mocapid == -1].sum())
    assert dynamic_mass == pytest.approx(0.800)
    assert np.all(model.body_inertia[1:] > 0)
    assert np.isfinite(model.body_inertia).all()
    assert np.all(2 * model.body_inertia[1:].max(axis=1) <= model.body_inertia[1:].sum(axis=1) + 1e-12)
    _assert_finite_state(data)


def test_home_has_forward_knees_legal_joints_and_full_size_motor_envelopes():
    model, data = demo.load_model()

    for joint in range(model.njnt):
        if model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        angle = data.qpos[model.jnt_qposadr[joint]]
        assert model.jnt_limited[joint]
        assert model.jnt_range[joint, 0] <= angle <= model.jnt_range[joint, 1]

    cases = []
    for side in ("left", "right"):
        hip, knee, ankle = [
            data.xanchor[model.joint(f"{side}_{part}").id].copy() for part in ("hip_pitch", "knee", "ankle")
        ]
        fraction = (knee[2] - hip[2]) / (ankle[2] - hip[2])
        hip_ankle_line_x = hip[0] + fraction * (ankle[0] - hip[0])
        assert knee[0] - hip_ankle_line_x > 0.001

        for part in ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle"):
            geom = model.geom(f"{side}_{part}_servo_case")
            cases.append(geom.id)
            assert geom.type[0] == mujoco.mjtGeom.mjGEOM_BOX
            assert np.sort(2 * geom.size) == pytest.approx([0.020, 0.026, 0.034])

    assert len(set(cases)) == 10


def test_display_support_is_opt_in_and_can_be_disabled():
    model, data = demo.load_model()
    support = model.equality("display_support").id

    assert model.eq_type[support] == mujoco.mjtEq.mjEQ_WELD
    assert not model.eq_active0[support]
    assert not data.eq_active[support]
    demo.set_support(model, data, True)
    assert data.eq_active[support]
    demo.set_support(model, data, False)
    assert not data.eq_active[support]
    assert not model.eq_active0[support]


def test_free_home_hold_stays_upright_for_five_seconds():
    model, data = demo.load_model()
    support = model.equality("display_support").id
    start_height = float(data.qpos[2])
    minimum_height = start_height
    maximum_tilt = 0.0

    for _ in range(round(5.0 / model.opt.timestep)):
        mujoco.mj_step(model, data)
        _assert_finite_state(data)
        _assert_bounded_actuator_forces(model, data)
        assert not data.eq_active[support]
        minimum_height = min(minimum_height, float(data.qpos[2]))
        up = data.xmat[model.body("torso").id].reshape(3, 3)[:, 2]
        maximum_tilt = max(maximum_tilt, float(np.arccos(np.clip(up[2], -1.0, 1.0))))

    assert data.time == pytest.approx(5.0, abs=model.opt.timestep)
    assert minimum_height > 0.90 * start_height
    assert data.qpos[2] > 0.95 * start_height
    assert maximum_tilt < np.deg2rad(10.0)
    _assert_no_warnings(data)


def test_supported_gesture_moves_head_with_finite_bounded_forces():
    model, data = demo.load_model(supported=True)
    support = model.equality("display_support").id
    torso = model.body("torso").id
    yaw_qpos = model.jnt_qposadr[model.joint("head_yaw").id]
    start_position = data.xpos[torso].copy()
    maximum_yaw_excursion = 0.0

    for _ in range(round(3.0 / model.opt.timestep)):
        controls = demo.gesture_controls(model, float(data.time))
        assert controls.shape == (model.nu,)
        assert np.isfinite(controls).all()
        assert np.all(controls >= model.actuator_ctrlrange[:, 0])
        assert np.all(controls <= model.actuator_ctrlrange[:, 1])
        data.ctrl[:] = controls
        mujoco.mj_step(model, data)
        assert data.eq_active[support]
        _assert_finite_state(data)
        _assert_bounded_actuator_forces(model, data)
        maximum_yaw_excursion = max(maximum_yaw_excursion, abs(float(data.qpos[yaw_qpos])))

    assert maximum_yaw_excursion > 0.02
    assert np.linalg.norm(data.xpos[torso] - start_position) < 0.001
    _assert_no_warnings(data)
