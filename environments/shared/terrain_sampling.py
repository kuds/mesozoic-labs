"""Balanced terrain families for a single command-tracking policy.

Every shuffled block contains the configured number of episodes from each
family. The course changes only at reset; pose, command and terrain-layout
randomness remain owned by the species behavior environment, whose reset
selects each episode's family with :func:`select_terrain_family`.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from environments.shared.plant_contract import REPOSITORY_ROOT

# "flat" is the original plane and "terrain_contact" a zero-height heightfield on the terrain's map; the rest
# are gentle templates. A block lists its episodes in this order before the shuffle, and a zero weight adds
# nothing, so appending "terrain_contact" leaves every block without it unchanged; "flat" stays first.
TERRAIN_FAMILIES = ("flat", "sloped", "bumps", "depressions", "mixed", "terrain_contact")
TerrainTemplate = Literal["sloped", "bumps", "depressions", "mixed"]
_MAX_BLOCK_SIZE = 1000


@dataclass(frozen=True)
class TerrainSamplerConfig:
    """Integer episode counts per shuffled block; zero disables a family.

    ``terrain_contact`` defaults to zero, so ``TerrainSamplerConfig()`` is the five original families.
    """

    flat: int = 1
    sloped: int = 1
    bumps: int = 1
    depressions: int = 1
    mixed: int = 1
    terrain_contact: int = 0

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
