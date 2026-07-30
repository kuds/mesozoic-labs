"""Canonical algorithm, backend, and path spellings.

One place decides how ``PPO``/``jax-mjx``/plant identities are spelled in
published artifacts, so a summary, a CSV row, and a directory name cannot
disagree about what produced a run."""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Any, Mapping

from .errors import ResultBundleError


def canonical_algorithm(algorithm: str) -> str:
    """Return the public algorithm label independent of training backend."""
    normalized = algorithm.strip().lower().replace("-", "_").replace("/", "_").replace(" ", "_")
    if normalized in {"ppo", "jax_ppo", "jax_mjx_ppo"}:
        return "PPO"
    if normalized == "sac":
        return "SAC"
    raise ResultBundleError(f"unsupported training algorithm: {algorithm!r}")


def canonical_backend(algorithm: str, backend: str | None = None) -> str:
    """Return the canonical training-backend identifier."""
    if backend is not None:
        normalized = backend.strip().lower().replace("_", "-")
        aliases = {
            "jax": "jax-mjx",
            "jax-mjx": "jax-mjx",
            "stable-baselines3": "stable-baselines3",
            "sb3": "stable-baselines3",
        }
        try:
            return aliases[normalized]
        except KeyError as exc:
            raise ResultBundleError(f"unsupported training backend: {backend!r}") from exc
    return "jax-mjx" if "jax" in algorithm.lower() else "stable-baselines3"


def _normalize_plant_identity(
    value: Mapping[str, Any] | None,
    *,
    species: str | None = None,
) -> dict[str, Any] | None:
    """Return a canonical plant-identity mapping or raise a bundle error."""
    if value is None:
        return None
    try:
        from ..plant_contract import PlantContractError, PlantIdentity

        identity = PlantIdentity.from_mapping(value)
    except (PlantContractError, KeyError, TypeError, ValueError) as exc:
        raise ResultBundleError(f"invalid plant identity: {exc}") from exc
    if species is not None and identity.species != species:
        raise ResultBundleError(f"plant identity species mismatch: expected {species!r}, found {identity.species!r}")
    normalized = identity.to_dict()
    if dict(value) != normalized:
        raise ResultBundleError("plant identity must use the canonical v1 field types and values")
    return normalized


def _portable_relative_path(value: Any, *, field: str) -> PurePosixPath:
    if not isinstance(value, str) or not value.strip():
        raise ResultBundleError(f"{field} must be a non-empty relative POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ResultBundleError(f"{field} must be a normalized relative POSIX path: {value!r}")
    return path
