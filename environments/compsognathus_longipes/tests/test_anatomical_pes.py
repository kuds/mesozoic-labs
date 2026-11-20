"""Exercise the fossil-scaled hindfoot and the lumped ankle/digital flexor."""

import mujoco
import numpy as np
import pytest

from environments.compsognathus_longipes.model import PARAMETERS, load_model, set_motors_enabled


def test_hindlimb_proportions_and_raised_medial_hallux():
    model, data = load_model("anatomical")
    for side, medial in (("r", 1), ("l", -1)):
        hip, knee, ankle, mtp = (data.joint(f"{side}_{name}").xanchor for name in ("hip_pitch", "knee", "ankle", "toe"))
        femur = np.linalg.norm(hip - knee)
        # CNJ79 (Peyer 2006, table 1), allowing for rounded millimetres.
        assert np.linalg.norm(knee - ankle) / femur == pytest.approx(131.0 / 108.8, abs=0.01)
        assert np.linalg.norm(ankle - mtp) / femur == pytest.approx(80.9 / 108.8, abs=0.01)
        hallux = model.geom(f"{side}_hallux").id
        assert model.geom_bodyid[hallux] == model.body(f"{side}_metatarsus").id
        assert (data.geom_xpos[hallux, 1] - mtp[1]) * medial > 0
        assert mujoco.mj_geomDistance(model, data, hallux, model.geom("floor").id, 0.1, None) > 0.020
        tips = [data.geom(f"{side}_claw_d{digit}").xpos[0] for digit in (2, 3, 4)]
        assert tips[1] > max(tips[0], tips[2]) + 0.008
        assert abs(tips[0] - tips[2]) < 0.005


@pytest.mark.parametrize("side", ["l", "r"])
def test_passive_digital_flexor_goes_slack_and_returns_energy_in_extension(side):
    model, data = load_model("anatomical")
    set_motors_enabled(model, False)
    joints = [model.joint(f"{side}_{joint}").id for joint in ("ankle", "toe")]
    qpos, dofs = model.jnt_qposadr[joints], model.jnt_dofadr[joints]
    p = PARAMETERS["anatomical"]
    arms = np.array(p["digital_flexor_moment_arms_m"])
    for angle in (0.2, 0.0, -0.2):
        data.qpos[qpos] = angle
        mujoco.mj_forward(model, data)
        tension = p["digital_flexor_stiffness_n_m"] * max(0.0, -angle * arms.sum())
        np.testing.assert_allclose(data.qfrc_passive[dofs], arms * tension, atol=1e-12)
        np.testing.assert_array_equal(data.qfrc_actuator[dofs], 0)
