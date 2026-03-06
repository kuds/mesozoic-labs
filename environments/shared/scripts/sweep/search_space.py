"""Search space loading, resolution, and per-stage extraction."""

import json
import logging
import sys
from pathlib import Path

from .constants import _DEFAULT_SEARCH_SPACES

logger = logging.getLogger(__name__)


def _load_search_space_file(path: str) -> dict:
    """Load a search space definition from a JSON file.

    The file can be either:

    * **Flat** — a single dict of parameter specs applied to all stages::

        {"ppo_learning_rate": {"type": "double", "min": 1e-5, "max": 3e-4, "scale": "log"}, ...}

    * **Per-stage** — top-level keys ``"stage1"``, ``"stage2"``, ``"stage3"``
      each mapping to a stage-specific search space dict::

        {
          "stage1": {"ppo_learning_rate": ..., "env_alive_bonus": ...},
          "stage2": {"ppo_learning_rate": ...},
          "stage3": {"ppo_learning_rate": ...}
        }

    Returns the parsed dict as-is; the caller detects the format by checking
    for ``"stage1"`` / ``"stage2"`` / ``"stage3"`` keys.
    """
    file_path = Path(path)
    if not file_path.exists():
        logger.error("Search space file not found: %s", file_path)
        sys.exit(1)
    try:
        result: dict = json.loads(file_path.read_text())
        return result
    except json.JSONDecodeError as exc:
        logger.error("Invalid JSON in search space file %s: %s", file_path, exc)
        sys.exit(1)


def _resolve_search_space(
    search_space_json: str | None,
    search_space_file: str | None,
    algorithm: str,
) -> dict:
    """Resolve the search space from CLI args.

    Priority: ``--search-space`` (inline JSON) > ``--search-space-file`` >
    algorithm default.

    Returns either a flat search-space dict (all stages share the same space)
    or a dict with ``"stage1"``/``"stage2"``/``"stage3"`` keys for per-stage
    spaces.
    """
    if search_space_json:
        try:
            result: dict = json.loads(search_space_json)
            logger.info("Search space resolved from inline --search-space JSON")
            return result
        except json.JSONDecodeError as exc:
            logger.error("Invalid --search-space JSON: %s", exc)
            sys.exit(1)

    if search_space_file:
        logger.info("Search space resolved from file: %s", search_space_file)
        return _load_search_space_file(search_space_file)

    if algorithm not in _DEFAULT_SEARCH_SPACES:
        logger.error(
            "No default search space for algorithm %r. Available: %s",
            algorithm,
            list(_DEFAULT_SEARCH_SPACES.keys()),
        )
        sys.exit(1)
    logger.info("Using default %s search space", algorithm)
    return _DEFAULT_SEARCH_SPACES[algorithm]


def _is_per_stage(config: dict) -> bool:
    """Return True if *config* uses per-stage keys (stage1/stage2/stage3)."""
    return "stage1" in config or "stage2" in config or "stage3" in config


def _split_stage_block(block: dict) -> tuple[dict, dict]:
    """Split a stage block into (search_space, settings).

    Within a stage block, any key whose value is a dict containing a
    ``"type"`` field is a search-space parameter.  All other keys
    (``trials``, ``timesteps``, ``parallel``, ``n_envs``) are job settings.
    """
    search_space: dict = {}
    settings: dict = {}
    for key, value in block.items():
        if isinstance(value, dict) and "type" in value:
            search_space[key] = value
        else:
            settings[key] = value
    return search_space, settings


def _search_space_for_stage(config: dict, stage: int) -> dict:
    """Extract the search space for a specific stage.

    If *config* has ``"stage1"``/``"stage2"``/``"stage3"`` keys, return
    the search-space parameters for the requested stage (job settings like
    ``trials`` and ``timesteps`` are filtered out).  Otherwise return
    *config* as-is (flat format — same space for all stages).
    """
    if _is_per_stage(config):
        key = f"stage{stage}"
        if key not in config:
            logger.error("Per-stage search space file is missing '%s' key", key)
            sys.exit(1)
        search_space, _ = _split_stage_block(config[key])
        return search_space
    return config


def _settings_for_stage(config: dict, stage: int) -> dict:
    """Extract job settings (trials, timesteps, parallel, n_envs) for a stage.

    Returns an empty dict for flat configs or stages without settings.
    """
    if _is_per_stage(config):
        key = f"stage{stage}"
        if key in config:
            _, settings = _split_stage_block(config[key])
            return settings
    return {}
