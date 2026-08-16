"""Tests for environments.shared.stage_manifest.

The manifest's one promise is no silent renumbering: an integer stage
reference means the legacy number forever, a new stage is reachable only
by its semantic ID, and a manifest that tries to rewrite history is
rejected at load.
"""

from __future__ import annotations

import pytest

from environments.shared.stage_manifest import (
    STAGE_MANIFEST_SCHEMA,
    StageManifestError,
    load_stage_manifest,
)


class TestTrexManifest:
    def test_four_stages_in_curriculum_order(self):
        manifest = load_stage_manifest("trex")
        assert not manifest.synthesized
        assert [entry.id for entry in manifest.stages] == ["stance", "recovery", "locomotion", "behavior"]
        assert [entry.position for entry in manifest.stages] == [1, 2, 3, 4]

    def test_integer_references_mean_legacy_numbers_not_positions(self):
        """THE no-silent-renumbering guarantee: 2 is locomotion, forever."""
        manifest = load_stage_manifest("trex")
        assert manifest.resolve(1).id == "stance"
        assert manifest.resolve(2).id == "locomotion"
        assert manifest.resolve(2).position == 3
        assert manifest.resolve(3).id == "behavior"
        assert manifest.resolve(3).position == 4

    def test_recovery_is_reachable_only_by_id(self):
        manifest = load_stage_manifest("trex")
        recovery = manifest.resolve("recovery")
        assert recovery.position == 2
        assert recovery.legacy_number is None
        assert recovery.config_file == "recovery.toml"
        with pytest.raises(StageManifestError, match="named by ID"):
            manifest.by_legacy_number(4)

    def test_unknown_references_are_fatal(self):
        manifest = load_stage_manifest("trex")
        with pytest.raises(StageManifestError, match="no stage 'sprint'"):
            manifest.resolve("sprint")
        with pytest.raises(StageManifestError, match="positions 1..4"):
            manifest.by_position(5)
        with pytest.raises(StageManifestError, match="invalid stage reference"):
            manifest.resolve(True)


class TestSynthesizedManifests:
    @pytest.mark.parametrize("species", ["velociraptor", "brachiosaurus", "dibothrosuchus"])
    def test_manifestless_species_get_the_legacy_three_stages(self, species):
        manifest = load_stage_manifest(species)
        assert manifest.synthesized
        assert [entry.id for entry in manifest.stages] == ["stance", "locomotion", "behavior"]
        assert [entry.legacy_number for entry in manifest.stages] == [1, 2, 3]
        # For a species without new stages, positions and legacy agree.
        assert all(entry.position == entry.legacy_number for entry in manifest.stages)


def _write_manifest(tmp_path, species, body):
    species_dir = tmp_path / species
    species_dir.mkdir(parents=True, exist_ok=True)
    for name in ("stage1_balance.toml", "recovery.toml"):
        (species_dir / name).write_text("[stage]\nname = 'x'\n")
    (species_dir / "stages.toml").write_text(body)
    return tmp_path


class TestFailClosedValidation:
    def test_wrong_schema_is_fatal(self, tmp_path):
        root = _write_manifest(tmp_path, "trex", 'schema = "mesozoic.stage-manifest/v2"\n')
        with pytest.raises(StageManifestError, match="requires 'mesozoic.stage-manifest/v1'"):
            load_stage_manifest("trex", configs_dir=root)

    def test_unknown_stage_id_is_fatal(self, tmp_path):
        body = f'schema = "{STAGE_MANIFEST_SCHEMA}"\n[[stages]]\nid = "sprint"\nconfig = "stage1_balance.toml"\n'
        root = _write_manifest(tmp_path, "trex", body)
        with pytest.raises(StageManifestError, match="unknown id 'sprint'"):
            load_stage_manifest("trex", configs_dir=root)

    def test_missing_config_file_is_fatal(self, tmp_path):
        body = f'schema = "{STAGE_MANIFEST_SCHEMA}"\n[[stages]]\nid = "stance"\nconfig = "missing.toml"\n'
        root = _write_manifest(tmp_path, "trex", body)
        with pytest.raises(StageManifestError, match="not found"):
            load_stage_manifest("trex", configs_dir=root)

    def test_rewriting_history_is_fatal(self, tmp_path):
        """legacy_number 2 IS locomotion; a manifest cannot reassign it."""
        body = (
            f'schema = "{STAGE_MANIFEST_SCHEMA}"\n'
            '[[stages]]\nid = "behavior"\nconfig = "stage1_balance.toml"\nlegacy_number = 2\n'
        )
        root = _write_manifest(tmp_path, "trex", body)
        with pytest.raises(StageManifestError, match="must not rewrite history"):
            load_stage_manifest("trex", configs_dir=root)

    def test_reordering_legacy_stages_is_fatal(self, tmp_path):
        body = (
            f'schema = "{STAGE_MANIFEST_SCHEMA}"\n'
            '[[stages]]\nid = "locomotion"\nconfig = "stage1_balance.toml"\nlegacy_number = 2\n'
            '[[stages]]\nid = "stance"\nconfig = "recovery.toml"\nlegacy_number = 1\n'
        )
        root = _write_manifest(tmp_path, "trex", body)
        with pytest.raises(StageManifestError, match="historical order"):
            load_stage_manifest("trex", configs_dir=root)

    def test_duplicate_ids_are_fatal(self, tmp_path):
        body = (
            f'schema = "{STAGE_MANIFEST_SCHEMA}"\n'
            '[[stages]]\nid = "stance"\nconfig = "stage1_balance.toml"\nlegacy_number = 1\n'
            '[[stages]]\nid = "stance"\nconfig = "recovery.toml"\n'
        )
        root = _write_manifest(tmp_path, "trex", body)
        with pytest.raises(StageManifestError, match="duplicate stage ids"):
            load_stage_manifest("trex", configs_dir=root)

    def test_unknown_entry_keys_are_fatal(self, tmp_path):
        body = (
            f'schema = "{STAGE_MANIFEST_SCHEMA}"\n'
            '[[stages]]\nid = "stance"\nconfig = "stage1_balance.toml"\nnumber = 1\n'
        )
        root = _write_manifest(tmp_path, "trex", body)
        with pytest.raises(StageManifestError, match="unknown keys"):
            load_stage_manifest("trex", configs_dir=root)


class TestRecoveryStageConfig:
    """The recovery stage config is stage 1's task plus exactly one delta."""

    def test_loads_by_semantic_id_and_only_by_it(self):
        from environments.shared.config import load_stage_config

        config = load_stage_config("trex", "recovery")
        assert config["name"] == "recovery"
        env = config["env_kwargs"]
        assert env["perturbation_capture_velocity_multiple"] == 1.5
        # Integer loading is untouched: 2 still means locomotion.
        assert load_stage_config("trex", 2)["name"] == load_stage_config("trex", "locomotion")["name"]

    def test_env_mirrors_stance_plus_exactly_the_perturbation_block(self):
        import tomllib

        stance = tomllib.load(open("configs/trex/stage1_balance.toml", "rb"))
        recovery = tomllib.load(open("configs/trex/recovery.toml", "rb"))
        perturbation_keys = {key for key in recovery["env"] if key.startswith("perturbation_")}
        assert perturbation_keys == {
            "perturbation_capture_velocity_multiple",
            "perturbation_interval",
            "perturbation_jitter",
            "perturbation_duration",
            "perturbation_direction",
        }
        mirrored = {key: value for key, value in recovery["env"].items() if key not in perturbation_keys}
        # Freshness pin: a stage-1 shaping change that forgets this file
        # fails here, with the fix being to re-mirror (and re-derive the
        # recovery rails once the W4 gate exists).
        assert mirrored == stance["env"]
        assert recovery["ppo"] == stance["ppo"]

    def test_gate_is_fail_closed_until_w4(self):
        import tomllib

        from environments.shared.curriculum.gate_schema import GateSchemaError, validate_gate_config

        recovery = tomllib.load(open("configs/trex/recovery.toml", "rb"))
        curriculum = recovery["curriculum"]
        assert curriculum["gate_kind"] == "none/v1"
        # A standalone pilot run of the recovery stage is legal...
        assert validate_gate_config("recovery", curriculum, advancement_enabled=False) == "none/v1"
        # ...but an advancing curriculum through it is refused until the
        # recovery_quality/v1 gate (W4) replaces the placeholder — the
        # schema's own non-advancing-pilot rule doing exactly its job.
        with pytest.raises(GateSchemaError, match="non-advancing"):
            validate_gate_config("recovery", curriculum)
