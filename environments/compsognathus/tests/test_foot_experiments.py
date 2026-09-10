"""Guard the controlled comparisons, contact accounting and replay probes."""

import csv
import json

import mujoco
import numpy as np
import pytest

from environments.compsognathus import MODEL_PATHS
from environments.compsognathus.experiments.feet import (
    distal_clearance_xml,
    foot_sensor_groups,
    foot_xml,
    obstacle_xml,
    spectral_metrics,
)
from environments.compsognathus.experiments.foot_probe import FootProbe
from environments.compsognathus.scripts.probe_feet import (
    ActionIntervention,
    _load_model,
    export_panel,
    mechanical_episode,
)
from environments.shared.plant_contract import (
    PlantCompatibilityError,
    current_plant_identity,
    validate_compiled_plant,
)


@pytest.mark.parametrize(
    "mechanism,dimensions",
    [
        ("baseline", (24, 23, 14)),
        ("fixed", (22, 21, 12)),
        ("coupled", (24, 23, 12)),
        ("independent", (28, 27, 12)),
    ],
)
def test_ablation_preserves_neutral_geometry_mass_and_common_controls(mechanism, dimensions):
    before = MODEL_PATHS["biological"].read_bytes()
    original, home = _load_model(foot_xml("baseline"))
    model, data = _load_model(foot_xml(mechanism))
    assert (model.nq, model.nv, model.nu) == dimensions
    assert model.body_subtreemass[model.body("pelvis").id] == pytest.approx(1.0)
    np.testing.assert_allclose(
        data.subtree_com[model.body("pelvis").id], home.subtree_com[original.body("pelvis").id], atol=1e-12
    )
    for geom in range(original.ngeom):
        other = model.geom(original.geom(geom).name).id
        np.testing.assert_allclose(data.geom_xpos[other], home.geom_xpos[geom], atol=1e-12)
        np.testing.assert_allclose(data.geom_xmat[other], home.geom_xmat[geom], atol=1e-12)
        np.testing.assert_array_equal(model.geom_size[other], original.geom_size[geom])
    for actuator in range(model.nu):
        old = original.actuator(model.actuator(actuator).name).id
        np.testing.assert_array_equal(model.actuator_forcerange[actuator], original.actuator_forcerange[old])
        np.testing.assert_allclose(data.ctrl[actuator], home.ctrl[old], atol=1e-12)
    assert MODEL_PATHS["biological"].read_bytes() == before


@pytest.mark.parametrize("mechanism", ["fixed", "coupled", "independent"])
def test_experimental_models_cannot_masquerade_as_current_training_plant(mechanism):
    identity = current_plant_identity("compsognathus")
    baseline, _ = _load_model(foot_xml("baseline"))
    validate_compiled_plant(baseline, identity)
    model, _ = _load_model(foot_xml(mechanism))
    with pytest.raises(PlantCompatibilityError, match="not the current"):
        validate_compiled_plant(model, identity)


def test_coupled_and_independent_have_equal_nominal_spring_damping_and_armature():
    coupled, _ = _load_model(foot_xml("coupled"))
    independent, _ = _load_model(foot_xml("independent"))
    for side in ("r", "l"):
        cj = coupled.joint(side + "_digits234_pitch").id
        ij = [independent.joint(side + "_digits" + str(d) + "_pitch").id for d in (2, 3, 4)]
        assert coupled.jnt_stiffness[cj] == pytest.approx(independent.jnt_stiffness[ij].sum())
        for field in ("dof_damping", "dof_armature"):
            assert getattr(coupled, field)[coupled.jnt_dofadr[cj]] == pytest.approx(
                getattr(independent, field)[independent.jnt_dofadr[ij]].sum()
            )
    assert independent.nexclude == coupled.nexclude + 6


@pytest.mark.parametrize("side", ["r", "l"])
@pytest.mark.parametrize("mechanism", ["baseline", "fixed", "coupled", "independent"])
def test_moving_body_touch_sensors_account_for_ground_force(mechanism, side):
    xml = obstacle_xml(foot_xml(mechanism), side=side, location="middle", height=0.003)
    model, data = _load_model(xml)
    probe = FootProbe(model, data)
    # This window includes the obstacle landing but ends before a rigid-foot
    # controller can fall: exercise the sensors without treating it as a gate.
    for _ in range(200):
        mujoco.mj_step(model, data)
        probe.sample()
    result = probe.results()
    assert result["both_feet_sensed"]
    assert result["obstacle_contact_samples"] > 0
    assert result["max_sensor_error_N"] < 1e-8
    assert result["max_self_contact_N"] < 1e-8
    if mechanism in ("coupled", "independent"):
        assert any(name.startswith(side + "_digits") for name in result["digit_sensors_with_load"])
    assert len(foot_sensor_groups(model)[0]) == {"baseline": 1, "fixed": 1, "coupled": 2, "independent": 4}[mechanism]


def test_short_failed_trial_does_not_report_missing_post_settle_metrics_as_zero():
    xml = obstacle_xml(foot_xml("baseline"), side="r", location="middle", height=0.003)
    result = mechanical_episode(xml, 4010, duration=5)
    assert not result["full_horizon"]
    assert result["duration_s"] < 4
    assert result["unsupported_windows"] is None
    assert result["pad_load_fraction"] is None
    assert result["sole_pitch_spectrum"]["peak_hz"] is None


def test_independent_digits_hold_small_middle_obstacle_with_unchanged_motor_caps():
    result = mechanical_episode(
        obstacle_xml(foot_xml("independent"), side="r", location="middle", height=0.003), 4010, duration=5
    )
    assert result["full_horizon"]
    assert result["unsupported_windows"] == 0
    assert not result["body_ground_contact"]
    assert result["max_sensor_error_N"] < 1e-8


def test_selected_joint_filter_has_no_reset_transient_or_cross_joint_effect():
    model, _ = _load_model(foot_xml("baseline"))
    probe = ActionIntervention(model, "feet_lp5", 0.02)
    first, second = np.ones(model.nu), -np.ones(model.nu)
    np.testing.assert_array_equal(probe.apply(first, 0), first)
    filtered = probe.apply(second, 1)
    selected = [model.actuator(s + "_toe_act").id for s in ("r", "l")]
    other = np.setdiff1d(np.arange(model.nu), selected)
    assert np.all(filtered[selected] > -1)
    np.testing.assert_array_equal(filtered[other], second[other])
    np.testing.assert_array_equal(second, -np.ones(model.nu))
    fresh = ActionIntervention(model, "feet_lp5", 0.02)
    np.testing.assert_array_equal(fresh.apply(second, 0), second)


@pytest.mark.parametrize("kind,expected", [("mean", 0.25), ("home", 0.0)])
def test_hold_is_causal_and_ramps_over_one_second(kind, expected):
    model, _ = _load_model(foot_xml("baseline"))
    probe = ActionIntervention(model, "feet_" + kind + "_after4s", 0.02)
    for step in range(200):
        probe.apply(np.full(model.nu, 0.25), step)
    raw = np.ones(model.nu)
    np.testing.assert_array_equal(probe.apply(raw, 200), raw)
    np.testing.assert_allclose(probe.apply(raw, 225)[probe.indices], (1 + expected) / 2)
    np.testing.assert_allclose(probe.apply(raw, 250)[probe.indices], expected)
    # Later raw actions cannot change the causal reference.
    np.testing.assert_allclose(probe.apply(-raw, 300)[probe.indices], expected)


def test_spectrum_resolves_chatter_without_inventing_frequency_for_static_signal():
    dt = 0.02
    x = np.sin(2 * np.pi * 15 * np.arange(800) * dt)
    result = spectral_metrics(x, dt)
    assert result["peak_hz"] == pytest.approx(15, abs=0.07)
    assert result["power_above_10hz"] > 0.99
    assert spectral_metrics(np.ones((800, 2)), dt)["peak_hz"] is None
    assert spectral_metrics(np.zeros((2, 2)), dt)["rms"] is None


@pytest.mark.parametrize("bad", [0, -1, float("nan"), float("inf")])
@pytest.mark.parametrize("parameter", ["stiffness", "damping", "armature"])
def test_invalid_passive_parameters_rejected(parameter, bad):
    with pytest.raises(ValueError, match="finite and positive"):
        foot_xml("independent", **{parameter: bad})


@pytest.mark.parametrize("changes", [{"side": "x"}, {"location": "heel"}, {"height": -0.1}, {"height": 0.02}])
def test_invalid_obstacle_rejected(changes):
    args = {"side": "r", "location": "middle", "height": 0.003, **changes}
    with pytest.raises(ValueError):
        obstacle_xml(foot_xml("baseline"), **args)


def test_distal_envelope_clearance_preserves_linkage_and_foot_contact_geometry():
    xml = foot_xml("independent")
    assert distal_clearance_xml(xml, 0) == xml
    original, home = _load_model(xml)
    model, data = _load_model(distal_clearance_xml(xml, 0.002))
    np.testing.assert_array_equal(model.body_mass, original.body_mass)
    np.testing.assert_array_equal(model.body_pos, original.body_pos)
    np.testing.assert_array_equal(model.jnt_pos, original.jnt_pos)
    np.testing.assert_array_equal(model.actuator_forcerange, original.actuator_forcerange)
    for side in ("r", "l"):
        for name in (side + "_plantar_pad", *(side + "_toe_d" + str(d) + "_geom" for d in (2, 3, 4))):
            gid = model.geom(name).id
            np.testing.assert_array_equal(data.geom_xpos[gid], home.geom_xpos[gid])
        gid = model.geom(side + "_metatarsus_geom").id
        assert model.geom_size[gid, 1] < original.geom_size[gid, 1]


@pytest.mark.parametrize("clearance", [-0.001, 0.004, float("nan"), float("inf")])
def test_invalid_distal_clearance_rejected(clearance):
    with pytest.raises(ValueError):
        distal_clearance_xml(foot_xml("independent"), clearance)


def test_trial_export_preserves_missing_metrics_and_git_stable_line_endings(tmp_path):
    export_panel(
        tmp_path / "panel",
        {
            "metadata": {"research_only": True},
            "episodes": [{"reward": 1.5, "spectrum": {"peak_hz": None}, "digits": ["r", "l"]}],
        },
    )
    path = tmp_path / "panel.csv"
    assert b"\r" not in path.read_bytes()
    with path.open() as stream:
        row = next(csv.DictReader(stream))
    assert row["spectrum.peak_hz"] == ""
    assert json.loads(row["digits"]) == ["r", "l"]
    assert json.loads((tmp_path / "panel.metadata.json").read_text()) == {"research_only": True}
