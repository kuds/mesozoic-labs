"""Catch unsafe geometry regressions in the separate 60 cm engineering study."""

import json

import numpy as np

from examples.tyrannosaurus_rex_60cm.build import HARDWARE
from examples.tyrannosaurus_rex_60cm.validate import hardware_contract, load, packing, source_hashes


def test_full_size_motor_and_free_base_contract():
    model, data = load()
    assert hardware_contract(model, data)["status"] == "pass"
    assert np.isclose(data.site("future_head_crown").xpos[2], 0.6)
    assert all(model.joint(model.actuator_trnid[i, 0]).name.startswith(("left_", "right_")) for i in range(model.nu))


def test_shrunken_motor_is_rejected():
    model, data = load()
    model.geom_size[model.geom("left_knee_case").id, 0] = 0.02
    assert hardware_contract(model, data)["status"] == "fail"


def test_electronics_overlap_is_rejected():
    model, data = load()
    model.geom_pos[model.geom("battery_reserve").id] = model.geom_pos[model.geom("compute_reserve").id]
    report = packing(model, data, [("overlap", model.key_qpos[0].copy())])
    assert report["status"] != "pass"
    assert any(
        set(p["pair"]) == {"battery_reserve", "compute_reserve"} and p["status"] != "pass"
        for p in report["worst_pairs"]
    )


def test_evidence_is_current_and_scope_is_not_overstated():
    report = json.loads((HARDWARE / "validation.json").read_text())
    assert report["source_sha256"] == source_hashes()
    for key in ("clearance", "hardware_contract", "installed_parts", "standing", "crouch"):
        assert report[key]["status"] == "pass"
    assert report["insufficient_torque_negative_control"]["observed"] == "fail"
    # Retain the diagnostic: independent joint limits do not prevent crossing feet.
    assert report["combined_range_probe"]["status"] == "fail"
    assert report["combined_range_probe"]["failed_by_pose_family"].keys() == {"combined_range"}
