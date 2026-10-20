"""The feet, legs, support geoms, soles and body weight of an env, and the measurement definition they enter.

:class:`Morphology`, :class:`FootSpec`, :class:`FootRegistryEntry`,
:class:`MorphologyError` and :func:`support_entry` live in
:mod:`environments.shared.floor_contact.morphology` (scratch, the stance4
copband design, never merged): the T. rex's opt-in CoP band builds a
morphology to read each step's contacts with the recorder's own reduction,
and no species env module imports this package.  They are re-exported here,
where every measurement reader imports them; that module's docstring holds
the generic rule and the ``FOOT_SUPPORT`` registry.
"""

from __future__ import annotations

from typing import Any

from ..floor_contact.morphology import FootRegistryEntry, FootSpec, Morphology, MorphologyError, support_entry
from ..model_id import ModelId

__all__ = ["FootRegistryEntry", "FootSpec", "Morphology", "MorphologyError", "measurement_definition", "support_entry"]


def measurement_definition(model_id: ModelId) -> dict[str, Any]:
    """This checkout's measurement DEFINITION for *model_id*, keyed as a manifest records it.

    ``{schema, constants, species, embodiment, registry}`` --
    :data:`.constants.MEASUREMENT_DEFINITION_KEYS`, the part of a
    :func:`.constants.measurement_manifest` that code alone fixes, so a
    reader with no model (the judge, publication) can compare a recorded
    manifest's definition against the one it would measure with:
    :func:`.constants.measurement_definition_sha256` of a recorded manifest
    equals that of this dict exactly when the two were measured under the
    same constants and the same registry entry (:func:`support_entry`); a
    model with no entry records ``registry: None``.
    """
    from .constants import MEASUREMENT_MANIFEST_SCHEMA, measurement_constants

    entry = support_entry(model_id)
    return {
        "schema": MEASUREMENT_MANIFEST_SCHEMA,
        "constants": measurement_constants(),
        **model_id.record(),
        "registry": None if entry is None else entry.as_manifest(),
    }
