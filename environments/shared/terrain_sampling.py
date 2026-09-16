"""Balanced terrain families for a single command-tracking policy.

Every shuffled block contains the configured number of episodes from each
family. The course changes only at reset; pose, command and terrain-layout
randomness remain owned by the existing species behavior environment.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np

from environments.shared.behavior_env import get_behavior_env_class
from environments.shared.plant_contract import REPOSITORY_ROOT
from environments.shared.species_names import resolve_species_id
from environments.shared.terrain import TerrainConfig

TERRAIN_FAMILIES = ("flat", "sloped", "bumps", "depressions", "mixed")
TerrainTemplate = Literal["sloped", "bumps", "depressions", "mixed"]
_MAX_BLOCK_SIZE = 1000


@dataclass(frozen=True)
class TerrainSamplerConfig:
    """Integer episode counts per shuffled block; zero disables a family."""

    flat: int = 1
    sloped: int = 1
    bumps: int = 1
    depressions: int = 1
    mixed: int = 1

    def __post_init__(self) -> None:
        for family, weight in asdict(self).items():
            if isinstance(weight, bool) or not isinstance(weight, int) or weight < 0:
                raise ValueError(f"terrain sampler weight {family} must be a nonnegative integer")
        if not 1 <= self.block_size <= _MAX_BLOCK_SIZE:
            raise ValueError(f"terrain sampler total weight must be in [1, {_MAX_BLOCK_SIZE}]")

    @property
    def families(self) -> tuple[str, ...]:
        return tuple(family for family in TERRAIN_FAMILIES if getattr(self, family) > 0)

    @property
    def block_size(self) -> int:
        return int(sum(getattr(self, family) for family in TERRAIN_FAMILIES))


@dataclass(frozen=True)
class TerrainSelection:
    family: str
    block_index: int
    block_position: int
    block_size: int
    selection_seed: int


def _nonnegative_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(value)


def select_terrain_family(
    config: TerrainSamplerConfig, *, run_seed: int, episode_seed: int, episode_index: int
) -> TerrainSelection:
    """Select an episode without mutable RNG state or dependence on commands."""
    run_seed = _nonnegative_integer(run_seed, "run_seed")
    episode_seed = _nonnegative_integer(episode_seed, "episode_seed")
    episode_index = _nonnegative_integer(episode_index, "episode_index")
    block_index, block_position = divmod(episode_index, config.block_size)
    selection_seed = int(np.random.SeedSequence([run_seed, episode_seed, block_index, 0x7E25]).generate_state(1)[0])
    block = [family for family in TERRAIN_FAMILIES for _ in range(getattr(config, family))]
    np.random.default_rng(selection_seed).shuffle(block)
    return TerrainSelection(block[block_position], block_index, block_position, config.block_size, selection_seed)


def sampler_source_identity() -> dict[str, str]:
    """Exact implementation proof used by behavior checkpoint transitions."""
    path = Path(__file__)
    return {str(path.relative_to(REPOSITORY_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()}


class TerrainSamplingMixin:
    """Add family selection without changing existing saved-task identities."""

    terrain_config: TerrainConfig
    flat_probability: float
    run_seed: int
    _episode_seed: int
    _episode_index: int

    def __init__(
        self,
        *,
        terrain_sampler: TerrainSamplerConfig,
        terrain: TerrainConfig,
        flat_probability: float = 0.0,
        **env_kwargs: Any,
    ):
        if not isinstance(terrain_sampler, TerrainSamplerConfig):
            raise ValueError("terrain_sampler must be a TerrainSamplerConfig")
        if not isinstance(terrain, TerrainConfig) or terrain.mode != "gentle":
            raise ValueError("terrain sampler requires an enabled gentle terrain configuration")
        if flat_probability != 0.0:
            raise ValueError("terrain sampler controls flat episodes; flat_probability must be zero")
        # Validate every enabled template up front. A valid slope profile may
        # otherwise have a grid too coarse for the requested bump radius.
        for family in terrain_sampler.families:
            if family != "flat":
                replace(terrain, template=cast(TerrainTemplate, family))
        self.terrain_sampler = terrain_sampler
        cast(Any, super()).__init__(terrain=terrain, flat_probability=0.0, **env_kwargs)

    @property
    def terrain_families(self) -> tuple[str, ...]:
        return self.terrain_sampler.families

    @property
    def behavior_identity(self) -> dict[str, Any]:
        identity = cast(dict[str, Any], cast(Any, super()).behavior_identity)
        identity.update(
            terrain_sampler=asdict(self.terrain_sampler),
            sampler_sources=sampler_source_identity(),
        )
        return identity

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        if seed is not None:
            seed = _nonnegative_integer(seed, "seed")
        episode_seed = self._episode_seed if seed is None else seed
        episode_index = self._episode_index if seed is None else 0
        selection = select_terrain_family(
            self.terrain_sampler,
            run_seed=self.run_seed,
            episode_seed=episode_seed,
            episode_index=episode_index,
        )
        parent_options = dict(options) if options is not None else None
        forced_family = None if parent_options is None else parent_options.pop("terrain_family", None)
        if forced_family is not None and forced_family not in self.terrain_families:
            raise ValueError(f"terrain_family must be an enabled family: {', '.join(self.terrain_families)}")
        family = selection.family if forced_family is None else forced_family
        base_terrain, base_flat_probability = self.terrain_config, self.flat_probability
        try:
            # Reuse the species' original plane/heightfield pools, including
            # T. rex's matching neck-contact probe. Only samples change.
            self.terrain_config = (
                base_terrain if family == "flat" else replace(base_terrain, template=cast(TerrainTemplate, family))
            )
            self.flat_probability = 1.0 if family == "flat" else 0.0
            observation, info = cast(Any, super()).reset(seed=seed, options=parent_options)
        finally:
            # Identity describes the whole training distribution and must not
            # depend on whichever terrain happened to be sampled most recently.
            self.terrain_config, self.flat_probability = base_terrain, base_flat_probability
        info["terrain_sampling"] = {
            **asdict(selection),
            "family": family,
            "mode": "balanced_shuffle" if forced_family is None else "evaluation_override",
            "weights": asdict(self.terrain_sampler),
        }
        return observation, info


@lru_cache(maxsize=None)
def _sampled_behavior_env_class(species: str) -> type[Any]:
    base = get_behavior_env_class(species)
    return type(
        f"{base.__name__.removesuffix('Env')}SampledTerrainEnv",
        (TerrainSamplingMixin, base),
        {"species": species, "__module__": __name__},
    )


def get_sampled_behavior_env_class(species: str) -> type[Any]:
    """Return a sampled-terrain subclass for any supported SB3 species."""
    return _sampled_behavior_env_class(resolve_species_id(species))
