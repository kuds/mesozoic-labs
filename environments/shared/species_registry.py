"""The training registry: each manifest model's env class and :class:`SpeciesConfig`.

Nothing here is keyed by hand (decision D-D33).  :func:`env_class` imports
the model's ``env_entrypoint`` from ``configs/species_manifest.toml``, and
:func:`get_species_config` builds the :class:`SpeciesConfig` from that
class, the model's stage manifest and three class attributes every species
base class carries: ``HEIGHT_LABEL``, ``BEHAVIOR_SECTION_LABEL`` and
``SUCCESS_KEYS``.  A new embodiment therefore needs no edit here.  These
three are presentation and success bookkeeping only; the policy layer reads
other class attributes (``supported_training_backends``, ``action_mapping``,
``action_filter_cutoff_hz``, ``residual_linear_slope``,
``_shaped_residual_mask`` and ``_foot_sensor_groups``), not these.

Usage:
    python -m environments.shared.train --species velociraptor_mongoliensis --embodiment anatomical train --stage 1
    python -m environments.shared.train --species tyrannosaurus_rex --embodiment anatomical curriculum
    python -m environments.shared.train --species brachiosaurus_altithorax --embodiment anatomical train --stage 2
    python -m environments.shared.train --species dibothrosuchus_elaphros --embodiment anatomical train --stage 1
"""

from __future__ import annotations

import importlib

from environments.shared.config import load_stage_config
from environments.shared.model_id import ModelId
from environments.shared.species_names import model_entry
from environments.shared.stage_manifest import load_stage_manifest
from environments.shared.train_base import SpeciesConfig


def env_class(model_id: ModelId) -> type:
    """The model's environment class, imported from its manifest ``env_entrypoint`` (``module:Class``)."""
    module_name, _, class_name = str(model_entry(model_id)["env_entrypoint"]).partition(":")
    cls = getattr(importlib.import_module(module_name), class_name)
    if not isinstance(cls, type):
        raise TypeError(f"{model_id} env_entrypoint {module_name}:{class_name} is not a class")
    return cls


def get_species_config(model_id: ModelId) -> SpeciesConfig:
    """The model's training config, derived from the manifest, its stage manifest and its env class.

    ``stage_descriptions`` lists each legacy-numbered stage as
    ``<legacy number>=<[stage] name>`` (``"1=stance, 2=locomotion, 3=bite"``
    on the T-Rex), the integers ``--stage`` accepts.
    """
    cls = env_class(model_id)
    descriptions = ", ".join(
        f"{entry.reference}={load_stage_config(model_id, entry.reference)['name']}"
        for entry in load_stage_manifest(model_id).advancing_stages
    )
    return SpeciesConfig(
        model_id=model_id,
        env_class=cls,
        stage_descriptions=descriptions,
        height_label=str(getattr(cls, "HEIGHT_LABEL")),
        behavior_section_label=str(getattr(cls, "BEHAVIOR_SECTION_LABEL")),
        success_keys=list(getattr(cls, "SUCCESS_KEYS")),
    )
