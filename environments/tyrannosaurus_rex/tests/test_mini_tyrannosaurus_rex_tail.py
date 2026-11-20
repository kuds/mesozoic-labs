"""Optional tail load/space gates; no fabrication or walking claim."""

import json

import numpy as np
import pytest

from examples.mini_tyrannosaurus_rex import demo, engineering, tail_option, tail_study


def test_optional_tail_adds_budgeted_mass_and_full_motor_without_changing_base():
    base, _ = demo.load_model()
    model = tail_option.load_model()
    ledger = json.loads(demo.MODEL_PATH.with_name("components.json").read_text())
    option = ledger["optional_tail_yaw"]
    assert model.body_mass.sum() == pytest.approx(base.body_mass.sum() + option["added_mass_g"] / 1000)
    assert (model.nq, model.nv, model.nu, model.nsensor) == (23, 22, 15, 41)
    assert model.body_mass.sum() * 1000 == pytest.approx(option["target_total_mass_g"])
    assert sum(row["mass_budget_g"] for row in option["added_components"]) == option["added_mass_g"]
    assert len(engineering.motor_cases(model)) == model.nu
    assert np.all(2 * model.body_inertia[1:].max(axis=1) <= model.body_inertia[1:].sum(axis=1) + 1e-12)
    assert base.body_mass.sum() == pytest.approx(0.800)
    assert base.nu == 14


def test_optional_tail_motor_and_legal_commands_keep_sampled_clearance():
    model = tail_option.load_model()
    motors = engineering.motor_clearance_sweep(model, samples=64)
    assert motors["overlap_poses"] == 0, motors["witness"]
    assert motors["minimum_separating_axis_gap_m"] > 0.001
    contacts = engineering.combined_collision_sweep(model, samples=128)
    assert contacts["overlap_poses"] == 0, contacts["witness"]


def test_rotating_tail_mesh_clears_component_service_boxes_even_with_contact_exclusions():
    model = tail_option.load_model()
    report = tail_study.packaging_clearance(model)
    for name, row in report["minimum_axis_gaps"].items():
        assert row["moving_tail_mesh_gap_m"] > 0.002, (name, row)
        assert row["motor_box_gap_m"] > 0.002, (name, row)


def test_gentle_tail_wag_moves_under_gravity_without_external_support():
    report = tail_study.wag_trial(tail_option.load_model(), duration=6)
    assert not report["externally_supported"]
    assert np.ptp(report["actual_tail_range_rad"]) > 0.40
    assert report["tail_peak_torque_Nm"] < 0.01
    assert report["tracking_rms_error_rad"] < 0.08
    assert report["minimum_height_ratio"] > 0.90
    assert report["maximum_torso_tilt_deg"] < 10


def test_tail_option_keeps_home_load_in_documented_concept_bounds():
    report = engineering.static_equilibrium(tail_option.load_model())
    for side in ("left", "right"):
        assert abs(report["joint_loads_Nm"][f"{side}_hip_roll"]) < 0.05
        # Tail option is deliberately separate: its knee demand is higher than
        # the base design's 0.05 Nm home target, despite battery repositioning.
        assert abs(report["joint_loads_Nm"][f"{side}_knee"]) < 0.06
    assert abs(report["joint_loads_Nm"]["tail_yaw"]) < 1e-8
