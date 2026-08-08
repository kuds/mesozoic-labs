"""Tests for environments.shared.plant_contract.manifest."""

from __future__ import annotations

import copy
import json

import mujoco
import pytest

import environments.shared.plant_contract as plant_contract
from environments.shared.plant_contract import (
    BUNDLED_MANIFEST_PATH,
    GENERATED_MANIFEST_PATH,
    PlantContractError,
    check_plant_manifest,
    current_plant_identity,
    load_plant_versions,
)
from environments.shared.plant_contract.manifest import _enforce_revision_changes

CANONICAL_MUJOCO_VERSION = load_plant_versions()[0]


@pytest.mark.skipif(
    mujoco.__version__ != CANONICAL_MUJOCO_VERSION,
    reason="exact manifest checks run only with the canonical MuJoCo version",
)
def test_committed_manifest_is_current_and_covers_all_species():
    manifest = check_plant_manifest()

    assert manifest["fingerprint_tool_version"] == plant_contract.FINGERPRINT_TOOL_VERSION == 2
    assert manifest["generated_with"]["float_significant_digits"] == 12
    assert set(manifest["plants"]) == {"velociraptor", "trex", "brachiosaurus", "dibothrosuchus"}
    for entry in manifest["plants"].values():
        assert entry["policy_interface"]["revision"] >= 1
        assert entry["physics"]["revision"] >= 1
        assert entry["visual"]["revision"] >= 1


def test_bundled_runtime_manifest_matches_repository_manifest():
    assert BUNDLED_MANIFEST_PATH.read_bytes() == GENERATED_MANIFEST_PATH.read_bytes()


def test_runtime_identity_falls_back_to_bundled_manifest(monkeypatch, tmp_path):
    monkeypatch.setattr(plant_contract.constants, "SPECIES_MANIFEST_PATH", tmp_path / "missing-species.toml")
    monkeypatch.setattr(plant_contract.constants, "PLANT_VERSIONS_PATH", tmp_path / "missing-versions.toml")
    monkeypatch.setattr(plant_contract.constants, "GENERATED_MANIFEST_PATH", tmp_path / "missing-manifest.json")

    identity = plant_contract.current_plant_identity("velociraptor")

    assert identity.species == "velociraptor"
    assert identity.nu == identity.action_dim == 22


@pytest.mark.parametrize(
    ("species", "observation_dim", "action_dim"),
    [("velociraptor", 67, 22), ("trex", 61, 15), ("brachiosaurus", 83, 30)],
)
def test_current_identity_matches_each_executable_environment(species, observation_dim, action_dim):
    # Identity generation requires exact SB3/MJX observation parity.
    identity = current_plant_identity(species, verify_generated=False)

    assert identity.observation_dim == observation_dim
    assert identity.action_dim == identity.nu == action_dim


def test_changed_layer_requires_revision_increment():
    old = {
        "plants": {
            "velociraptor": {
                "policy_interface": {"sha256": "interface-a", "revision": 1},
                "physics": {"sha256": "physics-a", "revision": 1},
                "visual": {"sha256": "visual-a", "revision": 1},
            }
        }
    }
    new = copy.deepcopy(old)
    new["plants"]["velociraptor"]["physics"]["sha256"] = "physics-b"

    with pytest.raises(PlantContractError, match="physics fingerprint changed"):
        _enforce_revision_changes(old, new)

    new["plants"]["velociraptor"]["physics"]["revision"] = 2
    _enforce_revision_changes(old, new)


def test_layer_revision_cannot_roll_back_when_fingerprint_is_unchanged():
    old = {
        "plants": {
            "velociraptor": {
                "policy_interface": {"sha256": "interface-a", "revision": 2},
                "physics": {"sha256": "physics-a", "revision": 2},
                "visual": {"sha256": "visual-a", "revision": 2},
            }
        }
    }
    new = copy.deepcopy(old)
    new["plants"]["velociraptor"]["visual"]["revision"] = 1

    with pytest.raises(PlantContractError, match="visual_revision cannot decrease"):
        _enforce_revision_changes(old, new)


@pytest.mark.skipif(
    mujoco.__version__ != CANONICAL_MUJOCO_VERSION,
    reason="exact manifest checks run only with the canonical MuJoCo version",
)
def test_manifest_check_enforces_revisions_against_external_baseline(tmp_path):
    baseline = json.loads(GENERATED_MANIFEST_PATH.read_text(encoding="utf-8"))
    baseline["plants"]["velociraptor"]["physics"]["sha256"] = "sha256:" + "0" * 64
    baseline_path = tmp_path / "base-plant-manifest.json"
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")

    with pytest.raises(PlantContractError, match="physics fingerprint changed"):
        check_plant_manifest(baseline_path=baseline_path)
