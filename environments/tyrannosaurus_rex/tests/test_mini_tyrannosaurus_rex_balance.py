"""Unsupported stance regressions for both current mini-robot configurations.

Passing this suite is not a walking gate or a qualified motor duty rating.
The deliberately weak-motor case proves that the gate can reject a bad plant.
"""

import mujoco
import numpy as np
import pytest

from examples.mini_tyrannosaurus_rex import balance_checks, demo, tail_option


def _model(with_tail):
    return tail_option.load_model() if with_tail else demo.load_model()[0]


@pytest.mark.parametrize("with_tail", [False, True], ids=["fixed-tail", "wag-tail"])
@pytest.mark.parametrize("kick", [(0, 0, 0), (0.15, 0, 0), (0, -0.15, 0)], ids=["hold", "roll-kick", "pitch-kick"])
def test_unsupported_stance_with_conservative_leg_caps_and_small_disturbances(with_tail, kick):
    model = _model(with_tail)
    saved = model.actuator_forcerange.copy()
    report = balance_checks.stance_trial(model, angular_kick=kick)
    assert report["duration_s"] == pytest.approx(30.0, abs=model.opt.timestep)
    assert not balance_checks.stance_failures(report), report
    assert model.actuator_forcerange == pytest.approx(saved)


@pytest.mark.parametrize("with_tail", [False, True], ids=["fixed-tail", "wag-tail"])
def test_balance_gate_rejects_insufficient_leg_torque_without_raising_motor_limits(with_tail):
    model = _model(with_tail)
    legs = [i for i in range(model.nu) if any(p in model.actuator(i).name for p in ("hip_", "knee", "ankle"))]
    model.actuator_forcerange[legs] = [-0.025, 0.025]
    report = balance_checks.stance_trial(model, duration=6)
    failures = balance_checks.stance_failures(report)
    assert "height loss" in failures or "excessive tilt" in failures, report
    assert all(row["applied_cap_Nm"] <= 0.025 for row in report["settled_leg_effort"].values())
    assert model.actuator_forcerange[legs] == pytest.approx(np.tile([-0.025, 0.025], (len(legs), 1)))


def test_balance_gate_refuses_hidden_support_or_zero_gravity():
    model, _ = demo.load_model()
    model.eq_active0[model.equality("display_support").id] = True
    with pytest.raises(ValueError, match="world weld"):
        balance_checks.stance_trial(model)
    model.eq_active0[model.equality("display_support").id] = False
    model.opt.gravity[:] = 0
    with pytest.raises(ValueError, match="Earth gravity"):
        balance_checks.stance_trial(model)


def test_balance_gate_requires_a_free_root_and_bounded_actuators():
    model, _ = demo.load_model()
    assert model.jnt_type[model.joint("root").id] == mujoco.mjtJoint.mjJNT_FREE
    model.actuator_forcelimited[model.actuator("left_knee_servo").id] = False
    with pytest.raises(ValueError, match="force limit"):
        balance_checks.stance_trial(model)


def test_balance_gate_refuses_gravity_compensation():
    model, _ = demo.load_model()
    model.body_gravcomp[model.body("torso").id] = 0.5
    with pytest.raises(ValueError, match="gravity compensation"):
        balance_checks.stance_trial(model)


def test_long_stance_is_not_an_artifact_of_the_default_two_millisecond_step():
    model = tail_option.load_model()
    model.opt.timestep = 0.001
    report = balance_checks.stance_trial(model)
    assert not balance_checks.stance_failures(report), report
    assert report["maximum_root_drift_m"] < 0.005
