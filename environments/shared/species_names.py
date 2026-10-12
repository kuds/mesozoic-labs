"""The models of ``configs/species_manifest.toml`` (schema 3): the one reader.

Every model is the pair (species, embodiment) of decision D-D33: a species
table ``[species.<species>]`` holds the species-level fields and one table
``[species.<species>.embodiments.<embodiment>]`` per model.  Species IDs are
binomial snake_case (``tyrannosaurus_rex``) and embodiments come from
:data:`~environments.shared.model_id.EMBODIMENTS`; both are matched as exact
strings, with no alias, display name, case or separator variant.  The model
display name is derived (``"Tyrannosaurus Rex (Anatomical)"``), never stored.
"""

from __future__ import annotations

import argparse
import copy
import tomllib
from collections.abc import Sequence
from functools import cache
from pathlib import Path
from typing import Any

from .model_id import EMBODIMENT_LABELS, EMBODIMENTS, ModelId

_MANIFEST_PATH = Path(__file__).resolve().parents[2] / "configs" / "species_manifest.toml"

#: The manifest schema this reader accepts; 3 since decision D-D33.
SPECIES_MANIFEST_SCHEMA_VERSION = 3

_TRAINING_BACKENDS = ("stable-baselines3", "jax-mjx")

_SPECIES_KEYS = frozenset({"display_order", "display_name", "embodiments"})
#: What the runtime readers need from every embodiment table: registration
#: and the training registry (``env_entrypoint``), the plant contract
#: (``model_path``) and :func:`model_ids` (``training_backends``, required on
#: every embodiment, with no default; review L5).
_REQUIRED_EMBODIMENT_KEYS = ("env_entrypoint", "model_path", "training_backends")


def _read_manifest(path: Path) -> dict[str, Any]:
    """Parse *path* and refuse what the readers below would misread.

    Checked here: ``schema_version``; every species ID and embodiment name
    (through :class:`ModelId`); the species-level keys; a unique positive integer
    ``display_order`` and a non-empty ``display_name`` per species; at least
    one embodiment per species; and the keys in
    :data:`_REQUIRED_EMBODIMENT_KEYS`.  Every other field is validated by
    the catalog generator (``species_catalog``).
    """
    with path.open("rb") as source:
        raw = tomllib.load(source)
    version = raw.get("schema_version")
    if version != SPECIES_MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            f"{path} declares schema_version {version!r}; this reader accepts {SPECIES_MANIFEST_SCHEMA_VERSION}"
        )
    species_tables = raw.get("species")
    if not isinstance(species_tables, dict) or not species_tables:
        raise ValueError(f"{path} declares no [species.<species>] table")
    orders: dict[int, str] = {}
    for species, table in species_tables.items():
        if not isinstance(table, dict):
            raise ValueError(f"{path}: species.{species} must be a table")
        unknown = sorted(set(table) - _SPECIES_KEYS)
        if unknown:
            raise ValueError(
                f"{path}: species.{species} has unknown keys {unknown}; a species table holds only "
                f"{sorted(_SPECIES_KEYS)}, and the per-model fields go in its embodiment tables"
            )
        order = table.get("display_order")
        if isinstance(order, bool) or not isinstance(order, int) or order < 1 or order in orders:
            raise ValueError(f"{path}: species.{species} needs a unique positive integer display_order, got {order!r}")
        orders[order] = species
        if not isinstance(table.get("display_name"), str) or not table["display_name"]:
            raise ValueError(f"{path}: species.{species} needs a non-empty display_name")
        embodiments = table.get("embodiments")
        if not isinstance(embodiments, dict) or not embodiments:
            raise ValueError(f"{path}: species.{species} declares no [species.{species}.embodiments.<e>] table")
        for embodiment, entry in embodiments.items():
            model_id = ModelId(species, embodiment)
            if not isinstance(entry, dict):
                raise ValueError(f"{path}: {model_id} must be a table")
            missing = [key for key in _REQUIRED_EMBODIMENT_KEYS if key not in entry]
            if missing:
                raise ValueError(f"{path}: {model_id} lacks {missing}")
            backends = entry["training_backends"]
            if not isinstance(backends, list) or not backends or set(backends) - set(_TRAINING_BACKENDS):
                raise ValueError(
                    f"{path}: {model_id} training_backends must be a non-empty subset of {_TRAINING_BACKENDS}"
                )
    return raw


@cache
def _manifest() -> dict[str, Any]:
    return _read_manifest(_MANIFEST_PATH)


def _species_table(species: str) -> dict[str, Any]:
    tables: dict[str, Any] = _manifest()["species"]
    if not isinstance(species, str) or species not in tables:
        raise ValueError(f"Unknown species {species!r}; valid species: {', '.join(species_ids())}")
    table: dict[str, Any] = tables[species]
    return table


def species_ids() -> tuple[str, ...]:
    """Every species ID, in manifest display order."""
    tables = _manifest()["species"]
    return tuple(sorted(tables, key=lambda species: tables[species]["display_order"]))


def embodiments_of(species: str) -> tuple[str, ...]:
    """The embodiments *species* has, in :data:`EMBODIMENTS` order; an unknown species raises."""
    declared = _species_table(species)["embodiments"]
    return tuple(embodiment for embodiment in EMBODIMENTS if embodiment in declared)


def model_ids(*, backend: str | None = None) -> tuple[ModelId, ...]:
    """Every model, in manifest order: species ``display_order``, then :data:`EMBODIMENTS` order.

    With *backend*, only the models whose ``training_backends`` list it.
    """
    if backend is not None and backend not in _TRAINING_BACKENDS:
        raise ValueError(f"Unknown training backend: {backend!r}")
    models = [ModelId(species, embodiment) for species in species_ids() for embodiment in embodiments_of(species)]
    if backend is not None:
        models = [model_id for model_id in models if backend in _embodiment_table(model_id)["training_backends"]]
    return tuple(models)


def require_model_id(species: str, embodiment: str) -> ModelId:
    """The manifest model named exactly by *species* and *embodiment*.

    Exact strings only (decision D-D33): an alias, a display name or a case
    or separator variant names nothing.  The error lists the valid species,
    or the embodiments that species has.
    """
    embodiments = embodiments_of(species)
    if embodiment not in embodiments:
        raise ValueError(f"{species} has no {embodiment!r} embodiment; available: {', '.join(embodiments)}")
    return ModelId(species, embodiment)


def _embodiment_table(model_id: ModelId) -> dict[str, Any]:
    declared = _species_table(model_id.species)["embodiments"]
    if model_id.embodiment not in declared:
        raise ValueError(
            f"{model_id.species} has no {model_id.embodiment!r} embodiment; "
            f"available: {', '.join(embodiments_of(model_id.species))}"
        )
    table: dict[str, Any] = declared[model_id.embodiment]
    return table


def model_entry(model_id: ModelId) -> dict[str, Any]:
    """A copy of the model's ``[species.<s>.embodiments.<e>]`` table; a model the manifest lacks raises."""
    return copy.deepcopy(_embodiment_table(model_id))


def species_display_name(species: str) -> str:
    """The species' display name (``"Tyrannosaurus Rex"``); an unknown species raises."""
    return str(_species_table(species)["display_name"])


def model_display_name(model_id: ModelId) -> str:
    """``"<species display name> (<Embodiment>)"``, e.g. ``"Compsognathus Longipes (Robot)"``.

    A model the manifest lacks raises.
    """
    _embodiment_table(model_id)
    return f"{species_display_name(model_id.species)} ({EMBODIMENT_LABELS[model_id.embodiment]})"


# The two command-line forms of a model (decision D-D33).  A single-model tool
# takes ``--species`` and ``--embodiment``, both required, with no default
# embodiment; a multi-model tool takes ``--model-id SPECIES EMBODIMENT``, as
# often as needed, and defaults to its panel as an explicit tuple of models.


def add_model_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the required ``--species`` and ``--embodiment`` of a single-model tool.

    The choices are the manifest's IDs; :func:`parsed_model_id` then checks the pair.
    """
    models = model_ids()
    parser.add_argument("--species", required=True, choices=species_ids(), help="species ID")
    parser.add_argument(
        "--embodiment",
        required=True,
        choices=[embodiment for embodiment in EMBODIMENTS if any(m.embodiment == embodiment for m in models)],
        help="embodiment of the species",
    )


def parsed_model_id(parser: argparse.ArgumentParser, args: argparse.Namespace) -> ModelId:
    """The model ``--species`` and ``--embodiment`` name; a pair the manifest lacks is a usage error (exit 2)."""
    try:
        return require_model_id(args.species, args.embodiment)
    except ValueError as exc:
        parser.error(str(exc))


def add_model_id_argument(parser: argparse.ArgumentParser, *, default: Sequence[ModelId], what: str) -> None:
    """Add the repeatable ``--model-id SPECIES EMBODIMENT`` of a multi-model tool.

    *default* is the tool's panel, written out as models; *what* says what the
    tool does with each model, for the help text.
    """
    parser.add_argument(
        "--model-id",
        nargs=2,
        action="append",
        metavar=("SPECIES", "EMBODIMENT"),
        dest="model_ids",
        help=f"a model to {what}; repeatable (default: {'; '.join(f'{m.species} {m.embodiment}' for m in default)})",
    )
    parser.set_defaults(model_ids_default=tuple(default))


def parsed_model_ids(parser: argparse.ArgumentParser, args: argparse.Namespace) -> tuple[ModelId, ...]:
    """The models ``--model-id`` named, in order, or the tool's default panel; an unknown pair is a usage error."""
    if not args.model_ids:
        return tuple(args.model_ids_default)
    models: list[ModelId] = []
    for species, embodiment in args.model_ids:
        try:
            model_id = require_model_id(species, embodiment)
        except ValueError as exc:
            parser.error(str(exc))
        if model_id in models:
            parser.error(f"--model-id {species} {embodiment} is given twice")
        models.append(model_id)
    return tuple(models)
