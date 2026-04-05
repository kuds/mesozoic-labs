"""mjlab adapter scaffold for Mesozoic Labs.

This module provides a thin bridge between the project's species registry
and `mjlab <https://github.com/mujocolab/mjlab>`_ — a lightweight framework
that pairs Isaac-Lab's manager-based API with MuJoCo Warp (GPU-accelerated
MuJoCo).

Why an adapter and not a drop-in?
---------------------------------
mjlab composes environments via **managers** (observation / reward /
termination / event / curriculum managers) rather than a monolithic
``gym.Env`` subclass. Our existing ``MJXDinoEnv`` and ``BaseDinoEnv`` use the
classic single-class pattern. This module lets us reuse our MJCF models,
TOML stage configs, and pure reward/obs functions while expressing the
*composition* in mjlab's idiom.

Status
------
**Proof of concept.** The adapter is intentionally small. It wires a species
registered with ``register_species_mjlab`` into mjlab's ``ManagerBasedRLEnv``
config objects. A working pilot should target velociraptor Stage 1 (balance)
and compare throughput + wall-clock-to-reward against the existing MJX path.

Requires the optional ``mjlab`` extra (NVIDIA GPU for training; macOS eval
only). See ``pyproject.toml``::

    pip install -e ".[mjlab]"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

# mjlab is an optional dependency: import lazily so the module is importable
# (and testable) even when mjlab/MuJoCo-Warp aren't installed.
try:
    import mjlab  # type: ignore  # noqa: F401

    _MJLAB_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dep
    _MJLAB_AVAILABLE = False


def require_mjlab() -> None:
    """Raise a helpful error if mjlab is not installed."""
    if not _MJLAB_AVAILABLE:
        raise ImportError(
            "mjlab is not installed. Install with: pip install -e '.[mjlab]'\n"
            "mjlab requires an NVIDIA GPU for training (macOS is evaluation-only).\n"
            "See https://github.com/mujocolab/mjlab"
        )


@dataclass(frozen=True)
class MJLabSpeciesConfig:
    """Species-level mjlab configuration.

    Maps 1:1 onto the fields mjlab's manager-based env expects. Each species
    registers one of these; stage-specific reward weights continue to live in
    the TOML stage configs under ``configs/<species>/<stage>.toml``.
    """

    species: str
    mjcf_path: Path
    frame_skip: int
    episode_length_s: float  # mjlab uses seconds, not steps
    num_envs_default: int = 4096  # MuJoCo Warp scales to thousands on one GPU

    # Observation / action specification (for sanity checks only — mjlab
    # infers these from the model + observation managers).
    obs_dim: int = 0
    action_dim: int = 0

    # Manager factories — populated at registration time. Each returns a
    # mjlab ManagerTermCfg / ObsGroupCfg / etc. Kept as opaque callables so
    # this module remains importable without mjlab installed.
    observation_manager_factory: Callable[[], Any] | None = None
    reward_manager_factory: Callable[[dict[str, float]], Any] | None = None
    termination_manager_factory: Callable[[], Any] | None = None
    event_manager_factory: Callable[[], Any] | None = None  # domain randomization

    # Sensible defaults for domain randomization ranges. Phase 2 roadmap item.
    randomization: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "friction": (0.7, 1.3),
            "damping_scale": (0.8, 1.2),
            "gravity_scale": (0.95, 1.05),
            "actuator_gain_scale": (0.9, 1.1),
        }
    )


_MJLAB_REGISTRY: dict[str, MJLabSpeciesConfig] = {}


def register_species_mjlab(config: MJLabSpeciesConfig) -> None:
    """Register a species for mjlab. Mirrors ``register_species_mjx``."""
    _MJLAB_REGISTRY[config.species] = config


def get_species_mjlab(species: str) -> MJLabSpeciesConfig:
    if species not in _MJLAB_REGISTRY:
        raise KeyError(
            f"Species '{species}' not registered for mjlab. "
            f"Import environments.{species}.mjlab_config to register it. "
            f"Available: {sorted(_MJLAB_REGISTRY)}"
        )
    return _MJLAB_REGISTRY[species]


def make_mjlab_env(
    species: str,
    stage: int,
    num_envs: int | None = None,
    device: str = "cuda",
) -> Any:
    """Construct a mjlab ``ManagerBasedRLEnv`` for the given species + stage.

    This is the single entry point scripts and notebooks should call. It:

    1. Loads the species config from the mjlab registry.
    2. Loads the stage reward weights from the TOML config.
    3. Composes mjlab managers (obs/reward/term/events) from the species
       factories.
    4. Returns a mjlab env configured for ``num_envs`` parallel workers.

    Parameters
    ----------
    species : str
        One of the registered species (``"velociraptor"``, ``"trex"``,
        ``"brachiosaurus"``).
    stage : int
        1 (balance), 2 (locomotion), or 3 (strike/bite/food_reach).
    num_envs : int, optional
        Number of parallel envs. Defaults to species config.
    device : str
        ``"cuda"`` for training, ``"cpu"`` for debugging, ``"mps"`` for
        macOS eval (limited).
    """
    require_mjlab()

    from environments.shared.config import load_stage_config  # lazy

    cfg = get_species_mjlab(species)
    stage_cfg = load_stage_config(species, stage)
    reward_weights = dict(stage_cfg.get("reward_weights", {}))

    # The following block is the integration point with mjlab's API. Kept
    # as pseudo-code with explicit TODOs so the spike PR can land without
    # pulling in the full mjlab dep tree in CI.
    #
    # from mjlab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
    # from mjlab.scene import InteractiveSceneCfg
    #
    # env_cfg = ManagerBasedRLEnvCfg(
    #     scene=InteractiveSceneCfg(
    #         num_envs=num_envs or cfg.num_envs_default,
    #         env_spacing=4.0,
    #         robot=_load_mjcf_asset(cfg.mjcf_path),
    #     ),
    #     observations=cfg.observation_manager_factory(),
    #     rewards=cfg.reward_manager_factory(reward_weights),
    #     terminations=cfg.termination_manager_factory(),
    #     events=cfg.event_manager_factory(),
    #     episode_length_s=cfg.episode_length_s,
    #     decimation=cfg.frame_skip,
    #     sim=_sim_cfg(device=device),
    # )
    # return ManagerBasedRLEnv(cfg=env_cfg)
    raise NotImplementedError(
        "mjlab integration is a scaffold. Wire up ManagerBasedRLEnv once the "
        "mjlab dep is installed locally. See module docstring for the pilot plan. "
        f"(species={cfg.species}, stage={stage}, reward_keys={sorted(reward_weights)}, "
        f"num_envs={num_envs or cfg.num_envs_default}, device={device})"
    )


__all__ = [
    "MJLabSpeciesConfig",
    "register_species_mjlab",
    "get_species_mjlab",
    "make_mjlab_env",
    "require_mjlab",
]
