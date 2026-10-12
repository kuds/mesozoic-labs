"""Unit tests for the species registry module.

The registry holds no table (decision D-D33): every model's config is derived
from the manifest, its stage manifest and its env class, so each manifest
model resolves and nothing else does.  Retired IDs and aliases are refused in
``test_retired_species_identifiers.py``.
"""

import pytest

from environments.shared.model_id import ModelId
from environments.shared.species_names import model_ids
from environments.shared.species_registry import env_class, get_species_config
from environments.shared.train_base import SpeciesConfig

MODEL_IDS = pytest.mark.parametrize("model_id", model_ids(), ids=lambda m: f"{m.species}-{m.embodiment}")


class TestEnvClass:
    """Every manifest model resolves to the embodiment class its derived name says."""

    @MODEL_IDS
    def test_every_manifest_model_resolves_to_its_embodiment_class(self, model_id):
        cls = env_class(model_id)
        assert cls.__name__ == model_id.env_class_name
        assert cls.__module__ == f"{model_id.package}.env"
        assert cls.embodiment == model_id.embodiment


class TestGetSpeciesConfig:
    """Verify get_species_config returns valid SpeciesConfig objects."""

    @MODEL_IDS
    def test_every_model_returns_a_config(self, model_id):
        config = get_species_config(model_id)
        assert isinstance(config, SpeciesConfig)
        assert config.model_id == model_id

    def test_unknown_species_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown species"):
            get_species_config(ModelId("stegosaurus_stenops", "anatomical"))

    def test_unknown_embodiment_raises_value_error(self, missing_model_id):
        with pytest.raises(ValueError, match=f"has no {missing_model_id.embodiment!r} embodiment"):
            get_species_config(missing_model_id)

    @MODEL_IDS
    def test_config_has_required_fields(self, model_id):
        config = get_species_config(model_id)
        assert config.env_class is not None
        assert isinstance(config.stage_descriptions, str)
        assert isinstance(config.height_label, str)
        assert isinstance(config.behavior_section_label, str)
        assert isinstance(config.success_keys, list)
        assert len(config.success_keys) > 0

    @MODEL_IDS
    def test_env_class_is_instantiable(self, model_id):
        """The env_class in the config should be importable and callable."""
        config = get_species_config(model_id)
        env = config.env_class()
        obs, _ = env.reset(seed=0)
        assert obs is not None
        env.close()
