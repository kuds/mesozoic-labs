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

        from environments.shared.stage_manifest import load_stage_manifest

        stance_file = load_stage_manifest("trex").resolve("stance").config_file
        stance = tomllib.load(open(f"configs/trex/{stance_file}", "rb"))
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
        # [ppo] mirrors stance with exactly one measured delta:
        # ent_coef_decay_timesteps is anchored to recovery's own 3M budget
        # (2M, ~2/3 — the same ratio as stance's 7M-of-11M) rather than
        # mirrored, because the 20260821_142144 pilot trained its whole
        # budget at ent_coef >= 0.0014 when the mirrored 7M horizon never
        # completed (2026-08 review §3.4).
        assert (
            recovery["ppo"] | {"ent_coef_decay_timesteps": stance["ppo"]["ent_coef_decay_timesteps"]}
            == stance["ppo"]
        )
        assert recovery["ppo"]["ent_coef_decay_timesteps"] == 2_000_000

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


class TestStageLabels:
    def test_legacy_integers_keep_their_historical_form(self):
        from environments.shared.stage_manifest import stage_label

        assert stage_label(1) == "stage1"
        assert stage_label(3) == "stage3"

    def test_semantic_ids_are_their_own_label(self):
        from environments.shared.stage_manifest import stage_label

        assert stage_label("recovery") == "recovery"

    def test_unknown_references_are_fatal(self):
        from environments.shared.stage_manifest import stage_label

        with pytest.raises(StageManifestError, match="unknown stage id"):
            stage_label("sprint")
        with pytest.raises(StageManifestError, match="invalid stage reference"):
            stage_label(True)


class TestLoadAllStages:
    def test_trex_gains_recovery_without_moving_the_integers(self):
        from environments.shared.config import load_all_stages

        configs = load_all_stages("trex")
        assert list(configs) == [1, "recovery", 2, 3]  # manifest order
        assert configs[2]["name"] == "locomotion"
        assert configs["recovery"]["env_kwargs"]["perturbation_capture_velocity_multiple"] == 1.5

    def test_manifestless_species_are_untouched(self):
        from environments.shared.config import load_all_stages

        assert list(load_all_stages("velociraptor")) == [1, 2, 3]

    def test_the_legacy_curriculum_ignores_semantic_only_stages(self):
        """thresholds_from_configs must not validate recovery's none/v1
        placeholder under advancement — that would refuse the whole
        legacy 1→2→3 curriculum the moment a species gains a manifest."""
        from environments.shared.config import load_all_stages
        from environments.shared.curriculum.manager import thresholds_from_configs

        thresholds = thresholds_from_configs(load_all_stages("trex"))
        assert sorted(thresholds) == [1, 2, 3]


class TestCliStageParsing:
    def test_digits_are_legacy_numbers_and_words_are_ids(self):
        from environments.shared.cli import _parse_stage_ref

        assert _parse_stage_ref("2") == 2
        assert _parse_stage_ref("recovery") == "recovery"


class TestStageDirnames:
    """Run-directory naming (2026-08-20): 01_stance style, readers accept all."""

    def test_dirnames_sort_in_curriculum_order_and_carry_the_id(self):
        from environments.shared.stage_manifest import stage_dirname

        names = [stage_dirname("trex", ref) for ref in ("stance", "recovery", "locomotion", "behavior")]
        assert names == ["01_stance", "02_recovery", "03_locomotion", "04_behavior"]
        assert names == sorted(names)

    def test_integer_refs_keep_their_legacy_meaning(self):
        from environments.shared.stage_manifest import stage_dirname

        # stage 2 is locomotion FOREVER — at position 3 in the manifest.
        assert stage_dirname("trex", 2) == "03_locomotion"
        assert stage_dirname("velociraptor", 2) == "02_locomotion"

    def test_candidates_cover_every_generation_newest_first(self):
        from environments.shared.stage_manifest import stage_dir_candidates

        assert stage_dir_candidates("trex", 1) == ("01_stance", "stage1", "stance")
        assert stage_dir_candidates("trex", "recovery") == ("02_recovery", "recovery")

    def test_find_stage_dir_prefers_whatever_exists(self, tmp_path):
        from environments.shared.stage_manifest import find_stage_dir

        legacy = tmp_path / "legacy_run"
        (legacy / "stage1").mkdir(parents=True)
        assert find_stage_dir(legacy, 1).name == "stage1"

        modern = tmp_path / "modern_run"
        (modern / "01_stance").mkdir(parents=True)
        (modern / "02_recovery").mkdir()
        assert find_stage_dir(modern, 1).name == "01_stance"
        assert find_stage_dir(modern, "recovery").name == "02_recovery"

        # Missing stays legible: the historical name comes back for errors.
        assert find_stage_dir(modern, 3).name == "stage3"


class TestSerializedStageKeys:
    """resolve_stage_key and the canonical reference/key spellings.

    Added with the bundle/catalog migration (2026-08-23): JSON object keys
    and CSV cells force every stage reference through a string, and this is
    the one place that decides what those strings mean.
    """

    def test_decimal_strings_are_legacy_numbers_never_positions(self):
        from environments.shared.stage_manifest import resolve_stage_key

        # "2" means locomotion (legacy) even though recovery holds
        # manifest position 2 — the no-silent-renumbering guarantee.
        assert resolve_stage_key("trex", "2").id == "locomotion"
        assert resolve_stage_key("trex", 2).id == "locomotion"
        assert resolve_stage_key("trex", "recovery").id == "recovery"

    def test_unknown_and_malformed_keys_fail_closed(self):
        from environments.shared.stage_manifest import StageManifestError, resolve_stage_key

        with pytest.raises(StageManifestError):
            resolve_stage_key("trex", "warp")
        with pytest.raises(StageManifestError):
            resolve_stage_key("trex", "4")
        with pytest.raises(StageManifestError):
            resolve_stage_key("velociraptor", "recovery")
        with pytest.raises(StageManifestError):
            resolve_stage_key("trex", True)

    def test_canonical_reference_and_key_spellings(self):
        from environments.shared.stage_manifest import load_stage_manifest

        manifest = load_stage_manifest("trex")
        assert [entry.reference for entry in manifest.stages] == [1, "recovery", 2, 3]
        assert [entry.key for entry in manifest.stages] == ["1", "recovery", "2", "3"]

    def test_advancing_stages_are_the_numbered_trio(self):
        from environments.shared.stage_manifest import load_stage_manifest

        assert [entry.id for entry in load_stage_manifest("trex").advancing_stages] == [
            "stance",
            "locomotion",
            "behavior",
        ]
        assert [entry.id for entry in load_stage_manifest("velociraptor").advancing_stages] == [
            "stance",
            "locomotion",
            "behavior",
        ]
