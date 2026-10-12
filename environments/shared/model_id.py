"""The identity of a model: the pair (species, embodiment) (decision D-D33).

A species is named by its binomial in snake_case (``tyrannosaurus_rex``), and
an embodiment is one word of :data:`EMBODIMENTS`: ``anatomical`` for the
anatomical model, ``robot`` for a robot of the species.  Neither is
normalised and neither has aliases; a model has no default embodiment.

Every machine identifier of a model (its config, log and result directories,
its package and environment class, its Gymnasium ID, its MJCF asset and its
catalog anchor) is *built* from the pair by one rule below and never parsed
back, so there is no ``ModelId.parse``.  Code that reads an identity takes the
two fields of a record (:meth:`ModelId.from_record`) or two path components,
never a joined string.

Records carry the two fields (:meth:`ModelId.record`).  ``str(model_id)``
(``tyrannosaurus_rex/anatomical``) is for messages and in-process keys only
and is never written into a record.  An identity is compared through
:class:`ModelId` equality or :meth:`ModelId.matches`, never on the species
alone, which does not separate two embodiments of one species;
``[tool.mypy] strict_equality`` makes a ``str == ModelId`` comparison a type
error.

Standard library only, so any module may import it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

#: The closed vocabulary of embodiments, in model order within a species.
EMBODIMENTS = ("anatomical", "robot")
#: Display labels, for model display names such as "Tyrannosaurus Rex (Anatomical)".
EMBODIMENT_LABELS = {"anatomical": "Anatomical", "robot": "Robot"}

# Binomial snake_case: genus, then epithet (and any further lower-case words).
_SPECIES_ID = re.compile(r"[a-z]+(?:_[a-z]+)+")
_IDENTITY_FIELDS = ("species", "embodiment")


def _pascal(identifier: str) -> str:
    return "".join(part.capitalize() for part in identifier.split("_"))


def _record_fields(record: Mapping[str, Any]) -> tuple[str, str]:
    """The record's species and embodiment, refusing a missing or non-string field."""
    values = []
    for field in _IDENTITY_FIELDS:
        if field not in record:
            raise ValueError(f"record has no {field!r} field; a model is identified by species and embodiment")
        value = record[field]
        if not isinstance(value, str):
            raise ValueError(f"record field {field!r} must be a string, got {value!r}")
        values.append(value)
    return values[0], values[1]


@dataclass(frozen=True, order=True)
class ModelId:
    """One model: a species and one of its embodiments."""

    species: str
    embodiment: str

    def __post_init__(self) -> None:
        if not isinstance(self.species, str) or _SPECIES_ID.fullmatch(self.species) is None:
            raise ValueError(
                f"species must be a binomial ID in snake_case such as 'tyrannosaurus_rex', got {self.species!r}"
            )
        if self.embodiment not in EMBODIMENTS:
            raise ValueError(f"embodiment must be one of {', '.join(EMBODIMENTS)}, got {self.embodiment!r}")

    def __str__(self) -> str:
        """``<species>/<embodiment>``: messages and in-process keys only, never a record."""
        return f"{self.species}/{self.embodiment}"

    def path(self, root: Path) -> Path:
        """``root/<species>/<embodiment>``: the model's directory under configs/, a log base or results/."""
        return root / self.species / self.embodiment

    @property
    def package(self) -> str:
        """``environments.<species>.<embodiment>``."""
        return f"environments.{self.species}.{self.embodiment}"

    @property
    def species_pascal(self) -> str:
        """``TyrannosaurusRex``: the species base class is ``species_pascal + "Env"``."""
        return _pascal(self.species)

    @property
    def pascal(self) -> str:
        """``TyrannosaurusRexAnatomical``."""
        return self.species_pascal + _pascal(self.embodiment)

    @property
    def env_class_name(self) -> str:
        """``TyrannosaurusRexAnatomicalEnv``, defined in ``<package>.env``."""
        return self.pascal + "Env"

    @property
    def gym_id(self) -> str:
        """``MesozoicLabs/TyrannosaurusRex-Anatomical-v0``."""
        return f"MesozoicLabs/{self.species_pascal}-{_pascal(self.embodiment)}-v0"

    @property
    def asset_name(self) -> str:
        """``<species>_<embodiment>.xml``, also the MJCF's ``<mujoco model=...>`` stem."""
        return f"{self.species}_{self.embodiment}.xml"

    def asset_path(self, environments_root: Path) -> Path:
        """The MJCF under the ``environments`` package root: ``<species>/<embodiment>/assets/<asset_name>``.

        The one rule the environment classes, the species' model paths and the
        species manifest's ``model_path`` check share.
        """
        return self.path(environments_root) / "assets" / self.asset_name

    @property
    def anchor(self) -> str:
        """``<species>-<embodiment>``: the model's catalog anchor."""
        return f"{self.species}-{self.embodiment}"

    def record(self) -> dict[str, str]:
        """The two fields a record carries."""
        return {"species": self.species, "embodiment": self.embodiment}

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> ModelId:
        """The model a record names; both fields are required, as exact strings."""
        species, embodiment = _record_fields(record)
        return cls(species, embodiment)

    def matches(self, record: Mapping[str, Any]) -> bool:
        """Whether a record's species and embodiment name this model.

        A record missing either field raises ``ValueError`` naming it, so a
        record never matches on its species alone.
        """
        return _record_fields(record) == (self.species, self.embodiment)
