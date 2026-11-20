"""Regression checks for the static force audit and signed support distances."""

import mujoco
import numpy as np
import pytest

from environments.compsognathus_longipes.model import load_model
from environments.compsognathus_longipes.scripts.static_balance import (
    at_rest,
    fixed_pose_tail_mass_sensitivity,
    support_polygon,
)


def test_directional_clearance_and_exterior_distance():
    triangle = [[0, 0], [2, 0], [0, 2]]
    supported = support_polygon(triangle, [0.5, 0.25])
    assert supported["signed_margin_m"] == pytest.approx(0.25)
    assert supported["directional_clearance_m"] == pytest.approx(
        {"forward": 1.25, "rearward": 0.5, "left": 1.25, "right": 0.25}
    )
    # Outside a corner the nearest point is a vertex, not a supporting line.
    assert support_polygon(triangle, [-1, -1])["signed_margin_m"] == pytest.approx(-np.sqrt(2))
    assert support_polygon([[0, 0], [2, 0]], [1, 1])["signed_margin_m"] == pytest.approx(-1)
    assert support_polygon([], [0, 0])["signed_margin_m"] is None


@pytest.mark.parametrize("embodiment", ["anatomical", "robot"])
def test_force_audit_includes_root_passive_axes_and_constraint_reactions(embodiment):
    model, data = load_model(embodiment)
    audit = at_rest(model, data, embodiment)
    actual = np.array([row["residual"] for row in audit["dofs"].values()])
    expected = np.zeros(model.nv)
    mujoco.mj_mulM(model, data, expected, data.qacc)
    np.testing.assert_allclose(actual, expected, atol=1e-9)
    for i, values in enumerate(audit["dofs"].values()):
        assert values["contacts"] + values["equality"] + values["other_constraints"] == pytest.approx(
            data.qfrc_constraint[i], abs=1e-10
        )
    assert audit["support"]["both_feet"]["signed_margin_m"] > 0
    for side in ("left", "right"):
        assert audit["support"][f"{side}_foot_alone_at_same_com"]["signed_margin_m"] < 0
    assert len(audit["actuators"]) == model.nu


def test_tail_mass_transfer_preserves_mass_and_recovers_the_authored_com():
    model, data = load_model("anatomical")
    snapshot = at_rest(model, data, "anatomical")
    hull = snapshot["support"]["both_feet"]
    light, nominal, heavy = fixed_pose_tail_mass_sensitivity(model, data, hull["hull_xy_m"])["cases"]
    for case in (light, nominal, heavy):
        assert case["total_mass_kg"] == pytest.approx(snapshot["mass_kg"])
        assert case["tail_mass_kg"] + case["trunk_own_mass_kg"] == pytest.approx(
            nominal["tail_mass_kg"] + nominal["trunk_own_mass_kg"]
        )
    assert heavy["tail_mass_kg"] == pytest.approx(1.25 * nominal["tail_mass_kg"])
    assert light["tail_mass_kg"] == pytest.approx(0.75 * nominal["tail_mass_kg"])
    np.testing.assert_allclose(nominal["com_m"], snapshot["com_m"], atol=1e-12)
    assert nominal["signed_margin_m"] == pytest.approx(hull["signed_margin_m"])
    assert heavy["com_m"][0] < nominal["com_m"][0] < light["com_m"][0]
    np.testing.assert_allclose(np.add(light["com_m"], heavy["com_m"]) / 2, nominal["com_m"], atol=1e-12)
