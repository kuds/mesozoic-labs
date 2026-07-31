"""Tests for environments.shared.plant_contract.source_layer."""

from __future__ import annotations

import environments.shared.plant_contract as plant_contract
from environments.shared.plant_contract.digests import (
    _semantic_digest,
)
from environments.shared.plant_contract.source_layer import _source_payload


def test_source_closure_remains_byte_exact(monkeypatch, tmp_path):
    model_path = tmp_path / "model.xml"
    model_path.write_text('<mujoco model="test"><worldbody/></mujoco>\n', encoding="utf-8")
    monkeypatch.setattr(plant_contract.constants, "REPOSITORY_ROOT", tmp_path)

    original = _source_payload(model_path)
    original_digest = _semantic_digest(plant_contract.SOURCE_SCHEMA, original)
    model_path.write_text('<mujoco model="test"><!-- byte change --><worldbody/></mujoco>\n', encoding="utf-8")
    changed = _source_payload(model_path)
    changed_digest = _semantic_digest(plant_contract.SOURCE_SCHEMA, changed)

    assert original["dependencies"][0]["content_sha256"] != changed["dependencies"][0]["content_sha256"]
    assert original_digest != changed_digest
