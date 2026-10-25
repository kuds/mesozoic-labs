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
from examples.tyrannosaurus_rex_100cm.validate import check_model

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

def test_recorded_evidence_tracks_source_bytes() -> None:
    record = json.loads(MODEL_PATH.with_name("validation.json").read_text())
    root = MODEL_PATH.parent.parent.parent
    for relative, expected in record["source_sha256"].items():
        path = MODEL_PATH.parent / relative if relative.startswith("../") else root / relative
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, relative
