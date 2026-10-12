"""Scale-appropriate balance gates, using the existing species' knee checks."""

from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from environments.compsognathus.model import load_model, set_motors_enabled
from environments.compsognathus.scripts.validate_models import RESET_SEEDS, hold_trial, support_sample
from environments.shared.tests.static_balance_helpers import JointLimitsAtHomeBase


@pytest.mark.parametrize("variant", ["biological", "robot"])
def test_home_has_loaded_support_and_knee_travel(variant):
    model, data = load_model(variant)
    sample = support_sample(model, data)
    assert sample["support_margin_m"] is not None, sample
    assert sample["support_margin_m"] > 0.005
    assert min(sample["foot_loads_N"]) > 0
    checks = JointLimitsAtHomeBase()
    checks.knee_names = [f"{s}_knee" for s in (("r", "l") if variant == "biological" else ("right", "left"))]
    env = SimpleNamespace(model=model, data=data)
    checks.test_no_joint_limit_violations(env)
    checks.test_knees_flexed_at_home(env)


@pytest.mark.parametrize("variant", ["biological", "robot"])
@pytest.mark.parametrize("seed", RESET_SEEDS)
def test_small_reset_errors_recover_double_support(variant, seed):
    trial = hold_trial(variant, seed=seed, seconds=5)
    assert trial["standing_passed"], trial


@pytest.mark.parametrize("variant", ["biological", "robot"])
def test_mass_growth_keeps_balance_with_original_motor_limits(variant):
    normal, _ = load_model(variant)
    heavy, _ = load_model(variant, 1.15)
    np.testing.assert_array_equal(normal.actuator_forcerange, heavy.actuator_forcerange)
    trial = hold_trial(variant, mass_scale=1.15, seconds=10)
    assert trial["standing_passed"], trial


@pytest.mark.parametrize("variant", ["biological", "robot"])
def test_disabling_actuation_is_real_and_reversible(variant):
    model, data = load_model(variant)
    original_flags = int(model.opt.disableflags)
    gains = model.actuator_gainprm.copy()
    data.ctrl[:] = model.actuator_ctrlrange[:, 1]
    set_motors_enabled(model, False)
    mujoco.mj_forward(model, data)
    np.testing.assert_array_equal(data.actuator_force, 0)
    np.testing.assert_array_equal(data.qfrc_actuator, 0)
    set_motors_enabled(model, True)
    mujoco.mj_forward(model, data)
    assert int(model.opt.disableflags) == original_flags
    np.testing.assert_array_equal(model.actuator_gainprm, gains)
    assert np.max(np.abs(data.actuator_force)) > 0.1


def test_robot_feet_have_six_independent_control_axes():
    model, data = load_model("robot")
    for side in ("left", "right"):
        jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jp, jr, model.site(f"{side}_sole").id)
        dofs = [model.jnt_dofadr[j] for j in model.actuator_trnid[:, 0] if model.joint(int(j)).name.startswith(side)]
        # Angular rows scaled to a representative leg length, in metres.
        singular = np.linalg.svd(np.vstack([jp[:, dofs], 0.2 * jr[:, dofs]]), compute_uv=False)
        assert singular[-1] > 0.005, singular
