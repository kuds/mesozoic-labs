"""Unit tests for the species registry module."""

import pytest

from environments.shared.species_registry import (
    SPECIES_FACTORIES,
    get_species_config,
)
from environments.shared.train_base import SpeciesConfig


class TestSpeciesFactories:
    """Verify the SPECIES_FACTORIES registry is well-formed."""

    def test_registry_contains_all_canonical_names(self):
        for name in ("velociraptor", "trex", "brachiosaurus"):
            assert name in SPECIES_FACTORIES

    def test_registry_contains_aliases(self):
        for alias in ("raptor", "t-rex", "brachio"):
            assert alias in SPECIES_FACTORIES

    def test_alias_resolves_to_same_species(self):
        assert SPECIES_FACTORIES["raptor"] is SPECIES_FACTORIES["velociraptor"]
        assert SPECIES_FACTORIES["t-rex"] is SPECIES_FACTORIES["trex"]
        assert SPECIES_FACTORIES["brachio"] is SPECIES_FACTORIES["brachiosaurus"]


class TestGetSpeciesConfig:
    """Verify get_species_config returns valid SpeciesConfig objects."""

    @pytest.mark.parametrize("name", ["velociraptor", "trex", "brachiosaurus"])
    def test_canonical_names_return_config(self, name):
        config = get_species_config(name)
        assert isinstance(config, SpeciesConfig)
        assert config.species == name

    @pytest.mark.parametrize(
        "alias, expected_species",
        [
            ("raptor", "velociraptor"),
            ("t-rex", "trex"),
            ("brachio", "brachiosaurus"),
        ],
    )
    def test_aliases_return_correct_species(self, alias, expected_species):
        config = get_species_config(alias)
        assert config.species == expected_species

    def test_case_insensitive_lookup(self):
        config = get_species_config("Velociraptor")
        assert config.species == "velociraptor"

    def test_unknown_species_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown species"):
            get_species_config("stegosaurus")

    @pytest.mark.parametrize("name", ["velociraptor", "trex", "brachiosaurus"])
    def test_config_has_required_fields(self, name):
        config = get_species_config(name)
        assert config.env_class is not None
        assert isinstance(config.stage_descriptions, str)
        assert isinstance(config.height_label, str)
        assert isinstance(config.stage3_section_label, str)
        assert isinstance(config.success_keys, list)
        assert len(config.success_keys) > 0

    @pytest.mark.parametrize("name", ["velociraptor", "trex", "brachiosaurus"])
    def test_env_class_is_instantiable(self, name):
        """The env_class in the config should be importable and callable."""
        config = get_species_config(name)
        env = config.env_class()
        obs, _ = env.reset(seed=0)
        assert obs is not None
        env.close()
