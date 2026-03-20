"""Per-species, per-stage search spaces for Ray Tune sweeps.

Search spaces are loaded from JSON config files (``configs/sweep_ppo.json``,
``configs/sweep_sac.json``) rather than being hardcoded.  The JSON format uses
the same per-stage layout as the Vertex AI sweep configs::

    {
      "stage1": {
        "trials": 40,
        "timesteps": 6000000,
        "ppo_learning_rate": {"type": "double", "min": 1e-5, "max": 3e-4, "scale": "log"},
        ...
      },
      ...
    }

Job-settings keys (``trials``, ``timesteps``, ``parallel``, ``n_envs``) are
automatically stripped — only parameter specs (dicts with a ``"type"`` key) are
returned as search-space entries.

The ``to_ray_tune()`` converter translates these into ``ray.tune`` distribution
objects so the same definitions can be reused by both Vertex AI and Ray Tune.

Call ``build_search_space(species, stage, algorithm)`` to get a ready-to-use
``dict[str, dict[str, Any]]``, or pass ``config_path`` to override the default
config file location.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from .search_space import _load_search_space_file, _search_space_for_stage

logger = logging.getLogger(__name__)


def detect_gpu_model() -> str:
    """Return a short GPU model name (e.g. ``"A100"``) or ``""`` if unavailable."""
    from environments.shared.config import _detect_gpu_info

    info = _detect_gpu_info()
    return str(info.get("gpu_model", ""))


# Type alias for a single parameter spec dict.
_ParamSpec = dict[str, Any]
_SearchSpace = dict[str, _ParamSpec]

# Default config directory relative to the repo root.
_CONFIGS_DIR = Path(__file__).resolve().parents[4] / "configs"


def _default_config_path(algorithm: str) -> Path:
    """Return the default JSON config path for an algorithm."""
    return _CONFIGS_DIR / f"sweep_{algorithm}.json"


def build_search_space(
    species: str,
    stage: int,
    algorithm: str,
    *,
    config_path: str | Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Build the search space from a JSON config file.

    Loads the per-stage config from ``configs/sweep_{algorithm}.json`` (or the
    path given by *config_path*), extracts the search-space parameters for the
    requested *stage*, and returns them in generic dict format::

        {"param_name": {"type": "double", "min": ..., "max": ..., "scale": ...}, ...}

    Use ``to_ray_tune()`` to convert to Ray Tune distribution objects.

    Parameters
    ----------
    species:
        Species name (currently informational — logged for record-keeping).
    stage:
        Curriculum stage number (1, 2, or 3).
    algorithm:
        Algorithm name (``"ppo"`` or ``"sac"``).
    config_path:
        Optional path to a sweep config JSON file.  Defaults to
        ``configs/sweep_{algorithm}.json``.
    """
    path = Path(config_path) if config_path else _default_config_path(algorithm)
    logger.info(
        "Loading search space from %s (species=%s, stage=%d, algorithm=%s)",
        path,
        species,
        stage,
        algorithm,
    )
    config = _load_search_space_file(str(path))
    search_space = _search_space_for_stage(config, stage)
    logger.info("Search space has %d parameters for stage %d", len(search_space), stage)
    return search_space


def to_ray_tune(search_space: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Convert a generic search-space dict to Ray Tune distribution objects.

    Supports the same parameter types as ``trial._build_parameter_spec``:

    - ``double`` with ``scale="log"`` → ``tune.loguniform(min, max)``
    - ``double`` with ``scale="linear"`` → ``tune.uniform(min, max)``
    - ``discrete`` → ``tune.choice(values)``
    - ``categorical`` → ``tune.choice(values)``
    """
    from ray import tune

    result: dict[str, Any] = {}
    for name, spec in search_space.items():
        kind = spec.get("type", "double")
        if kind == "double":
            if spec.get("scale") == "log":
                result[name] = tune.loguniform(spec["min"], spec["max"])
            else:
                result[name] = tune.uniform(spec["min"], spec["max"])
        elif kind == "discrete":
            result[name] = tune.choice(spec["values"])
        elif kind == "categorical":
            result[name] = tune.choice(spec["values"])
        else:
            raise ValueError(f"Unknown parameter type {kind!r} for {name}")
    return result


def save_search_space(
    search_space: dict[str, dict[str, Any]],
    dest_dir: str | Path,
    *,
    species: str = "",
    stage: int = 0,
    algorithm: str = "",
    gpu_model: str = "",
    max_concurrent: int = 0,
    n_envs: int = 0,
    timesteps_per_trial: int = 0,
    num_trials: int = 0,
    eval_freq: int = 0,
    seed: int = 0,
    grace_period: int = 0,
    reduction_factor: int = 0,
    collapse_min_evals: int = 0,
    collapse_patience: int = 0,
    use_asha: bool = True,
) -> Path:
    """Write the resolved search space to *dest_dir* as JSON for record keeping.

    The file is named ``search_space_stage{stage}_{algorithm}.json`` and
    includes metadata (species, stage, algorithm, parameter count) alongside the
    full parameter definitions so each sweep run is fully reproducible.

    When any runtime keyword arguments are provided, a ``"runtime"`` section is
    added to the JSON with GPU, concurrency, environment, and training settings.

    Returns the path to the written file.
    """
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)

    filename = f"search_space_stage{stage}_{algorithm}.json" if stage and algorithm else "search_space.json"
    filepath = dest / filename

    from environments.shared.config import get_library_version

    payload: dict[str, Any] = {
        "species": species,
        "stage": stage,
        "algorithm": algorithm,
        "library_version": get_library_version(),
        "num_parameters": len(search_space),
    }

    # Build runtime section from provided keyword arguments.
    runtime: dict[str, Any] = {}
    if gpu_model:
        runtime["gpu_model"] = gpu_model
    if max_concurrent:
        runtime["max_concurrent"] = max_concurrent
    if n_envs:
        runtime["n_envs"] = n_envs
    if timesteps_per_trial:
        runtime["timesteps_per_trial"] = timesteps_per_trial
    if num_trials:
        runtime["num_trials"] = num_trials
    if eval_freq:
        runtime["eval_freq"] = eval_freq
    if seed:
        runtime["seed"] = seed
    runtime["use_asha"] = use_asha
    if grace_period:
        runtime["grace_period"] = grace_period
    if reduction_factor:
        runtime["reduction_factor"] = reduction_factor
    if collapse_min_evals:
        runtime["collapse_min_evals"] = collapse_min_evals
    if collapse_patience:
        runtime["collapse_patience"] = collapse_patience
    if runtime:
        payload["runtime"] = runtime

    payload["parameters"] = search_space

    filepath.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    logger.info("Search space saved to %s (%d params)", filepath, len(search_space))
    return filepath
