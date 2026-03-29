"""Per-species, per-stage search spaces for Ray Tune sweeps.

Search spaces are loaded from species-specific JSON config files
(``configs/{species}/sweep_{algorithm}.json``).  The JSON format uses the same per-stage
layout as the Vertex AI sweep configs::

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


def detect_gpu_info() -> dict[str, Any]:
    """Return full GPU details dict, matching the format used by ``stage_config.json``.

    Keys (when a GPU is available): ``gpu_model``, ``gpu_full_name``,
    ``gpu_memory_gb``, ``cuda_version`` (or ``driver_version``).
    Returns an empty dict when no GPU is detected.
    """
    from environments.shared.config import _detect_gpu_info

    return _detect_gpu_info()


# Type alias for a single parameter spec dict.
_ParamSpec = dict[str, Any]
_SearchSpace = dict[str, _ParamSpec]

# Default config directory relative to the repo root.
_CONFIGS_DIR = Path(__file__).resolve().parents[4] / "configs"


def _default_config_path(algorithm: str, species: str = "") -> Path:
    """Return the species-specific sweep config path.

    Returns ``configs/{species}/sweep_{algorithm}.json``.  Raises
    :class:`FileNotFoundError` when *species* is empty or the file does not
    exist.
    """
    if not species:
        raise FileNotFoundError(
            f"No species specified and no generic sweep config fallback. "
            f"Provide a species name or an explicit config_path."
        )
    path = _CONFIGS_DIR / species / f"sweep_{algorithm}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No sweep config found for species={species!r}, algorithm={algorithm!r}. Expected: {path}"
        )
    return path


def resolve_config_path(
    algorithm: str,
    species: str = "",
    config_path: str | Path | None = None,
) -> Path:
    """Resolve the sweep config file path.

    When *config_path* is provided, relative paths are resolved against the
    repo root.  Otherwise the species-specific default is returned.  Raises
    :class:`FileNotFoundError` when no config exists for the species/algorithm.
    """
    if config_path:
        path = Path(config_path)
        if not path.is_absolute():
            repo_root = Path(__file__).resolve().parents[4]
            path = repo_root / path
        return path
    return _default_config_path(algorithm, species)


def build_search_space(
    species: str,
    stage: int,
    algorithm: str,
    *,
    config_path: str | Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Build the search space from a JSON config file.

    Loads the per-stage config from the species-specific or generic sweep
    config (or the path given by *config_path*), extracts the search-space
    parameters for the requested *stage*, and returns them in generic dict
    format::

        {"param_name": {"type": "double", "min": ..., "max": ..., "scale": ...}, ...}

    Use ``to_ray_tune()`` to convert to Ray Tune distribution objects.

    Parameters
    ----------
    species:
        Species name — used to locate a species-specific config file.
    stage:
        Curriculum stage number (1, 2, or 3).
    algorithm:
        Algorithm name (``"ppo"`` or ``"sac"``).
    config_path:
        Optional path to a sweep config JSON file.  Defaults to
        ``configs/{species}/sweep_{algorithm}.json``.
    """
    path = resolve_config_path(algorithm, species, config_path)
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
    gpu_info: dict[str, Any] | None = None,
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
    load_path: str = "",
    config_path: str = "",
) -> Path:
    """Write the resolved search space to *dest_dir* as JSON for record keeping.

    The file is named ``search_space_stage{stage}_{algorithm}.json`` and
    includes metadata (species, stage, algorithm, parameter count) alongside the
    full parameter definitions so each sweep run is fully reproducible.

    When any runtime keyword arguments are provided, a ``"runtime"`` section is
    added to the JSON with GPU, concurrency, environment, and training settings.

    GPU details are supplied as a full dict via *gpu_info* (matching the
    ``stage_config.json`` format with keys like ``gpu_model``,
    ``gpu_full_name``, ``gpu_memory_gb``, ``cuda_version``).  When provided it
    is stored under a ``"gpu"`` key at the top level, consistent with
    ``stage_config.json``.

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

    # Full GPU details — same format as stage_config.json.
    if gpu_info:
        payload["gpu"] = gpu_info

    # Build runtime section from provided keyword arguments.
    runtime: dict[str, Any] = {}
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
    if load_path:
        runtime["load_path"] = load_path
    if runtime:
        payload["runtime"] = runtime

    # Resume settings — stored so the notebook can reload them when resuming
    # a sweep without needing to manually re-enter paths.
    resume: dict[str, Any] = {}
    resume["sweep_dir"] = str(dest_dir)
    resume["config_path"] = config_path or ""
    payload["resume"] = resume

    payload["parameters"] = search_space

    filepath.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    logger.info("Search space saved to %s (%d params)", filepath, len(search_space))
    return filepath


def load_resume_settings(search_space_path: str | Path) -> dict[str, Any]:
    """Load resume and runtime settings from a saved search space JSON file.

    Returns a dict with keys that map directly to notebook configuration
    variables::

        {
            "species": "brachiosaurus",
            "stage": 2,
            "algorithm": "ppo",
            "resume": True,
            "resume_sweep_dir": "/content/drive/.../stage2_ppo_20260326_160015",
            "sweep_config_path": "/content/drive/.../search_space_stage2_ppo.json",
            # Runtime settings (from the "runtime" section):
            "max_concurrent": 3,
            "n_envs": 8,
            "timesteps_per_trial": 16000000,
            "num_trials": 50,
            "eval_freq": 50000,
            "seed": 42,
            "use_asha": True,
            "grace_period": 40,
            "reduction_factor": 2,
            "collapse_min_evals": 8,
            "collapse_patience": 5,
            "load_path": "...",
        }

    Parameters
    ----------
    search_space_path:
        Path to a ``search_space_stage{N}_{algo}.json`` file previously
        written by :func:`save_search_space`.
    """
    path = Path(search_space_path)
    if not path.exists():
        raise FileNotFoundError(f"Search space file not found: {path}")

    data: dict[str, Any] = json.loads(path.read_text())

    settings: dict[str, Any] = {
        "species": data.get("species", ""),
        "stage": data.get("stage", 0),
        "algorithm": data.get("algorithm", ""),
        "resume": True,
    }

    # Resume section — sweep_dir and config_path.
    resume_section = data.get("resume", {})
    settings["resume_sweep_dir"] = resume_section.get("sweep_dir", "")
    # Point config_path back to this file itself so the same parameters are
    # used on resume.
    settings["sweep_config_path"] = resume_section.get("config_path", "") or str(path)

    # Flatten runtime settings into top-level keys.
    runtime = data.get("runtime", {})
    for key in (
        "max_concurrent",
        "n_envs",
        "timesteps_per_trial",
        "num_trials",
        "eval_freq",
        "seed",
        "use_asha",
        "grace_period",
        "reduction_factor",
        "collapse_min_evals",
        "collapse_patience",
        "load_path",
        "gpu_model",
    ):
        if key in runtime:
            settings[key] = runtime[key]

    return settings
