"""Protect the medium concept's mechanics and honest free-base stance checks."""

from __future__ import annotations

import json

import mujoco
import numpy as np
import pytest

from examples.tyrannosaurus_rex_50cm import PARTS_BUDGET_PATH, build_model, demo, head_design, proportions, validate


@pytest.mark.parametrize("selected", [False, True])
def test_components_reconcile_with_budget_and_explicit_design_additions(selected):
    path = demo.MODEL_PATH if selected else demo.BASELINE_PATH
    mass = 7.957 if selected else 7.617
    model, _ = demo.load_model(path=path)
    ledger = json.loads(path.with_name("components.json").read_text())
    budget = json.loads(PARTS_BUDGET_PATH.read_text())
    expected_categories = {k: v / 1000 for k, v in budget["mass_allowances_g"].items()}
    if selected:
        for category, addition in (
            ("load_bearing_structure", 0.120),
            ("bearings_feet_fasteners", 0.010),
            ("shells_and_guards", 0.210),
        ):
            expected_categories[category] += addition
    assert ledger["categories_kg"] == pytest.approx(expected_categories)
    assert sum(c["mass_kg"] for c in ledger["components"]) == pytest.approx(mass)
    assert model.body_mass.sum() == pytest.approx(mass)
    for b in range(1, model.nbody):
        expected = sum(c["mass_kg"] for c in ledger["components"] if c["body"] == model.body(b).name)
        assert model.body_mass[b] == pytest.approx(expected)
        inertia = model.body_inertia[b]
        assert np.all(inertia > 0)
        assert 2 * max(inertia) <= sum(inertia) + 1e-12


def test_interface_and_physics_have_no_hidden_support():
    model, data = demo.load_model()
    assert (model.nq, model.nv, model.nu, model.nsensor) == (25, 24, 18, 48)
    assert model.joint("root").type == mujoco.mjtJoint.mjJNT_FREE
    assert model.opt.gravity == pytest.approx([0, 0, -9.81])
    assert model.neq == 1
    assert not model.eq_active0.any() and not data.eq_active.any()
    assert model.nmocap == 0
    assert not model.body_gravcomp.any()
    assert not data.xfrc_applied.any()
    assert np.count_nonzero(model.geom_bodyid == 0) == 1
    assert model.actuator_forcelimited.all()
    assert model.actuator_forcerange[:12] == pytest.approx(np.tile([-4, 4], (12, 1)))
    for j in range(1, model.njnt):
        assert model.jnt_limited[j]
        assert model.jnt_range[j, 0] <= data.qpos[model.jnt_qposadr[j]] <= model.jnt_range[j, 1]
    assert not validate.internal_contacts(model, data)
    assert data.ncon == 8
    for c in data.contact[: data.ncon]:
        assert model.geom(c.geom1).name == "floor"
        assert model.geom(c.geom2).name in ("left_sole_collision", "right_sole_collision")
        assert c.dist > -0.0001


def test_size_knee_direction_and_twelve_provisional_motor_envelopes():
    model, data = demo.load_model()
    low, high = validate.visual_bounds(model, data)
    assert 0.58 <= high[2] - low[2] <= 0.59
    assert high[1] - low[1] < 0.35
    cases = [model.geom(i) for i in range(model.ngeom) if model.geom(i).name.endswith("_case_collision")]
    assert len(cases) == 12
    for case in cases:
        assert case.size[:2] == pytest.approx([0.030, 0.0275])
        assert case.contype[0] and case.conaffinity[0]
    for side in ("left", "right"):
        hip, knee, ankle = [data.xanchor[model.joint(f"{side}_{j}").id] for j in ("hip_pitch", "knee", "ankle_pitch")]
        fraction = (knee[2] - hip[2]) / (ankle[2] - hip[2])
        assert knee[0] - (hip[0] + fraction * (ankle[0] - hip[0])) > 0.02


def test_cameras_are_in_the_head_snout_and_foot_sensors_are_on_feet():
    model, data = demo.load_model()
    for name in ("snout_left", "snout_rgb", "snout_right"):
        camera = model.camera(name).id
        assert model.cam_bodyid[camera] == model.body("head").id
        assert model.cam_pos[camera, 0] > 0.13
        # MuJoCo cameras look down their local negative Z axis.
        assert -data.cam_xmat[camera].reshape(3, 3)[:, 2] == pytest.approx([1, 0, 0])
    for side in ("left", "right"):
        for i in range(4):
            site = model.site(f"{side}_contact_{i}").id
            assert model.site_bodyid[site] == model.body(side + "_foot").id
            assert model.sensor(f"{side}_contact_force_{i}").type == mujoco.mjtSensor.mjSENS_TOUCH


@pytest.mark.parametrize("gesture", [False, True])
def test_ten_second_free_stance_and_character_regressions(gesture):
    model, _ = demo.load_model()
    result = validate.run_trial(model, duration=10, gesture=gesture)
    assert result["passed"], result
    assert not result["external_support_active"]
    assert max(result["rms_torque_nm"][:12]) < 2.8
    assert sum(result["final_foot_contact_force_n"]) == pytest.approx(model.body_mass.sum() * 9.81, rel=0.05)
    if gesture:
        for name, minimum in (
            ("head_yaw", 0.3),
            ("head_nod", 0.06),
            ("jaw_pitch", 0.10),
            ("tail_yaw", 0.30),
            ("left_shoulder_pitch", 0.45),
            ("right_shoulder_pitch", 0.35),
        ):
            idx = model.joint(name).id - 1
            assert result["joint_excursion_rad"][idx] > minimum, name


def test_display_weld_is_explicit_and_optional():
    model, data = demo.load_model()
    demo.set_support(model, data, True)
    assert data.eq_active[0]
    demo.set_support(model, data, False)
    assert not data.eq_active[0]
    result = validate.run_trial(model, duration=3, supported=True, gesture=True)
    assert result["passed"] and result["external_support_active"]


def test_stance_check_rejects_insufficient_leg_torque():
    model, _ = demo.load_model()
    model.actuator_forcerange[:12, :] = [-0.1, 0.1]
    result = validate.run_trial(model, duration=3)
    assert not result["passed"]
    assert result["min_height_ratio"] < 0.9 or result["max_torso_tilt_deg"] > 10


def test_source_generator_reproduces_committed_model_and_assets(tmp_path):
    build_model.build(tmp_path)
    for name in ("model.xml", "components.json", "assets/rounded.obj", "assets/soft.obj", "assets/tail.obj"):
        assert (tmp_path / name).read_bytes() == (demo.MODEL_PATH.parent / name).read_bytes(), name
    build_model.build_baseline(tmp_path / "baseline_51cm", assets_directory=tmp_path / "assets")
    for name in ("model.xml", "components.json"):
        assert (tmp_path / "baseline_51cm" / name).read_bytes() == (demo.BASELINE_PATH.parent / name).read_bytes()
    # The selected default and the retained C alternative must have identical
    # compiled physics; only their relative mesh paths differ.
    selected_model, _ = demo.load_model()
    option_model, _ = demo.load_model(path=head_design.OUTPUT / "soft_wedge/model.xml")
    for field in ("body_mass", "body_inertia", "body_pos", "jnt_pos", "actuator_forcerange"):
        assert getattr(selected_model, field) == pytest.approx(getattr(option_model, field))


def test_generator_canonicalizes_cross_version_float_sum_residue():
    # This difference was observed between Python 3.11 and 3.12 in Linux CI.
    assert build_model.values(3.095999999999999) == build_model.values(3.096) == "3.096"
    assert build_model.values((0.0, -3.5e-20, 1.2)) == "0 0 1.2"


@pytest.mark.parametrize(
    "variant,height,mass", [("compact", 0.515, 7.857), ("balanced", 0.550, 8.067), ("tall", 0.600, 8.347)]
)
def test_fuller_body_studies_retain_real_legs_reserve_and_free_standing(variant, height, mass, tmp_path):
    path = proportions.build_variant(variant, tmp_path / variant)
    model, data = demo.load_model(path=path)
    original, home = demo.load_model(path=demo.BASELINE_PATH)
    ledger = json.loads(path.with_name("components.json").read_text())
    assert ledger["categories_kg"]["unallocated_integration_reserve"] == pytest.approx(0.600)
    assert model.body_mass.sum() == pytest.approx(mass)
    low, high = validate.visual_bounds(model, data)
    assert high[2] - low[2] == pytest.approx(height)
    assert not data.eq_active.any() and not model.body_gravcomp.any()
    for side in ("left", "right"):
        for name in ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle_pitch", "ankle_roll"):
            joint = side + "_" + name
            assert data.xanchor[model.joint(joint).id] == pytest.approx(home.xanchor[original.joint(joint).id])
    assert model.actuator_forcerange == pytest.approx(original.actuator_forcerange)
    assert not validate.internal_contacts(model, data)
    result = validate.run_trial(model, duration=10, gesture=True)
    assert result["passed"], result
    assert max(result["rms_torque_nm"][:12]) < 2.4


def test_proportion_budget_preserves_base_and_applies_contingency_after_additions():
    scenarios = proportions.budget_scenarios()["scenarios"]
    assert scenarios["baseline"]["installed_usd"] == ["8576.52", "8576.52"]
    assert scenarios["baseline"]["first_build_funding_usd"] == ["13891.99", "13891.99"]
    assert scenarios["balanced"]["installed_usd"] == ["8926.52", "9276.52"]
    assert scenarios["balanced"]["first_build_funding_usd"] == ["14554.49", "15166.99"]


def test_proportion_payload_adds_mass_above_existing_reserve(tmp_path):
    nominal, _ = demo.load_model(path=proportions.build_variant("balanced", tmp_path / "nominal"))
    model, _ = demo.load_model(path=proportions.build_variant("balanced", tmp_path / "head", extra_mass_body="head"))
    assert model.body_mass.sum() - nominal.body_mass.sum() == pytest.approx(0.5)
    assert model.body_subtreemass[model.body("head_yaw_body").id] == pytest.approx(0.85)


@pytest.mark.parametrize(
    "variant,head_mass",
    [
        ("large_head", 0.530),
        ("oversized_head", 0.635),
        ("rounded_box", 0.530),
        ("long_oval", 0.530),
        ("soft_wedge", 0.530),
    ],
)
def test_head_led_design_retains_hardware_and_moves_without_support(variant, head_mass, tmp_path):
    path = head_design.build(variant, tmp_path / variant)
    model, data = demo.load_model(path=path)
    original, home = demo.load_model(path=demo.BASELINE_PATH)
    low, high = head_design.head_bounds(model, data)
    robot_low, robot_high = validate.visual_bounds(model, data)
    assert (high[2] - low[2]) / (robot_high[2] - robot_low[2]) > 0.36
    assert high[1] - low[1] > 0.27
    assert model.body_subtreemass[model.body("head_yaw_body").id] == pytest.approx(head_mass)
    ledger = json.loads(path.with_name("components.json").read_text())
    assert ledger["categories_kg"]["unallocated_integration_reserve"] == pytest.approx(0.6)
    for side in ("left", "right"):
        for joint in ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle_pitch", "ankle_roll"):
            name = side + "_" + joint
            assert data.xanchor[model.joint(name).id] == pytest.approx(home.xanchor[original.joint(name).id])
    assert model.geom("camera_package_envelope").size == pytest.approx(original.geom("camera_package_envelope").size)
    assert model.cam_pos[model.camera("snout_left").id, 1] - model.cam_pos[
        model.camera("snout_right").id, 1
    ] == pytest.approx(0.07)
    assert model.actuator_forcerange == pytest.approx(original.actuator_forcerange)
    assert not model.eq_active0.any() and not model.body_gravcomp.any()
    assert not validate.internal_contacts(model, data)
    trial = validate.run_trial(model, duration=10, gesture=True)
    assert trial["passed"], trial
    assert trial["joint_excursion_rad"][15] > 0.4
    assert trial["joint_excursion_rad"][16] > 0.07
    assert trial["joint_excursion_rad"][17] > 0.1
    assert max(trial["peak_torque_nm"][15:17]) < 0.5
    assert trial["peak_torque_nm"][17] < 0.09


@pytest.mark.parametrize("duration", [0, -1, float("nan"), float("inf")])
def test_invalid_trial_duration_is_rejected(duration):
    model, _ = demo.load_model()
    with pytest.raises(ValueError):
        validate.run_trial(model, duration=duration)
