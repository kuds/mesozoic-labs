"""Assembly and unsupported stance gates for the checkout-only metre concept."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import pytest

from environments.shared.robot_validation import require_unsupported
from examples.tyrannosaurus_rex_100cm.build_model import build
from examples.tyrannosaurus_rex_100cm.demo import MODEL_PATH, load_model
from examples.tyrannosaurus_rex_100cm.validate import check_model, motor_case_clearance
from examples.tyrannosaurus_rex_100cm.verify_parts import (
    budget_summary,
    camera_visibility,
    check_parts,
    electronics_clearance,
)


@pytest.fixture(scope="module")
def report() -> dict[str, Any]:
    return check_model(duration=10)


def test_free_stance_gestures_and_small_disturbances(report: dict[str, Any]) -> None:
    assert report["passed"], report
    for trial in report["trials"].values():
        assert trial["passed"], trial
    assert not report["home"]["internal_contacts"]


def test_mass_ledger_and_finite_actuator_limits() -> None:
    model, data = load_model()
    ledger = json.loads(MODEL_PATH.with_name("components.json").read_text())
    assert model.nu == 18
    assert (model.nq, model.nv) == (25, 24)
    assert sum(c["mass_kg"] for c in ledger["components"]) == pytest.approx(model.body_mass.sum())
    for body in range(1, model.nbody):
        assigned = sum(c["mass_kg"] for c in ledger["components"] if c["body"] == model.body(body).name)
        assert assigned == pytest.approx(model.body_mass[body])
        assert (model.body_inertia[body] > 0).all()
    require_unsupported(model, data)
    assert np.all(model.actuator_forcelimited)
    assert np.isfinite(model.actuator_forcerange).all()


def test_fixture_cannot_pass_as_unsupported_balance() -> None:
    model, data = load_model()
    data.eq_active[model.equality("display_support").id] = True
    with pytest.raises(ValueError):
        require_unsupported(model, data)


def test_generated_model_ledger_and_meshes_are_current(tmp_path: Path) -> None:
    build(tmp_path)
    for generated in sorted(tmp_path.rglob("*")):
        if generated.is_file():
            committed = MODEL_PATH.parent / generated.relative_to(tmp_path)
            assert committed.read_bytes() == generated.read_bytes(), str(committed)
    model = mujoco.MjModel.from_xml_path(str(tmp_path / "model.xml"))
    assert model.nu == 18


@pytest.mark.parametrize("filename", ["validation.json", "parts_verification.json"])
def test_recorded_evidence_tracks_source_bytes(filename: str) -> None:
    record = json.loads(MODEL_PATH.with_name(filename).read_text())
    root = MODEL_PATH.parent.parent.parent
    for relative, expected in record["source_sha256"].items():
        path = (
            MODEL_PATH.parent / relative
            if filename == "parts_verification.json" or relative.startswith("../")
            else root / relative
        )
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, relative


def test_parent_child_motor_interference_is_not_hidden_by_contact_filtering() -> None:
    model, data = load_model()
    assert motor_case_clearance(model, data)["cases"] == 18
    assert motor_case_clearance(model, data)["distance_m"] > 0.002
    # Place the hip-roll case on the yaw case. Their body relationship normally
    # suppresses contact generation, but a direct geometry query must catch it.
    model.geom_pos[model.geom("left_hip_roll_case_collision").id] = [0, 0, 0]
    mujoco.mj_forward(model, data)
    assert motor_case_clearance(model, data)["distance_m"] < 0.002


def test_packaging_expression_loads_camera_and_budget() -> None:
    result = check_parts()
    assert result["passed"], result


def test_overlapping_electronics_fail_the_packaging_screen() -> None:
    model, data = load_model()
    model.geom_pos[model.geom("compute_tray_envelope").id] = model.geom_pos[model.geom("battery_envelope").id]
    mujoco.mj_forward(model, data)
    # Deeply nested boxes may return zero rather than a signed penetration;
    # either result must fail the same positive-clearance gate as a near miss.
    assert electronics_clearance(model, data)["distance_m"] < 0.002


def test_camera_screen_detects_a_panel_in_front_of_the_lenses() -> None:
    model, data = load_model()
    model.geom_pos[model.geom("snout_camera_bar").id, 0] = 0.48
    mujoco.mj_forward(model, data)
    assert not camera_visibility(model, data)["passed"]


def test_budget_rejects_missing_motors_and_stale_csv(tmp_path: Path) -> None:
    ledger = json.loads(MODEL_PATH.with_name("components.json").read_text())
    budget = json.loads(MODEL_PATH.with_name("budget.json").read_text())
    bom = MODEL_PATH.with_name("bom.csv")
    row = next(r for r in budget["rows"] if r.get("motor_model") == "RS04")
    row["quantity"] -= 1
    assert not budget_summary(ledger, budget, bom)["passed"]
    row["quantity"] += 1
    stale = tmp_path / "stale.csv"
    stale.write_text(bom.read_text().replace("RS04,4,", "RS04,3,"))
    assert not budget_summary(ledger, budget, stale)["passed"]
