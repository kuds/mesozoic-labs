"""
Load curriculum stage configurations from TOML files.

Each species has a configs/<species>/ directory with one TOML file per stage:
    stage1_balance.toml
    stage2_locomotion.toml
    stage3_<behavior>.toml

Each TOML file has four tables: [stage], [env], [ppo]/[sac], and [curriculum].
The [curriculum] table contains per-stage training and advancement settings:
    timesteps           - number of timesteps to train this stage
    min_avg_reward      - minimum average reward to advance (optional)
    min_avg_episode_length - minimum average episode length to advance (optional)
    required_consecutive   - number of consecutive passes required (optional)
"""

from pathlib import Path
from typing import Any, Dict, Optional

import tomllib

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIGS_DIR = _REPO_ROOT / "configs"

# Map stage number -> filename pattern per species (discovered automatically)
_STAGE_FILE_PREFIX = {1: "stage1_", 2: "stage2_", 3: "stage3_"}


def _find_stage_file(species: str, stage: int) -> Path:
    """Find the TOML config file for a given species and stage."""
    species_dir = _CONFIGS_DIR / species
    if not species_dir.is_dir():
        raise FileNotFoundError(f"Config directory not found: {species_dir}")

    prefix = _STAGE_FILE_PREFIX[stage]
    matches = list(species_dir.glob(f"{prefix}*.toml"))
    if not matches:
        raise FileNotFoundError(f"No config file matching '{prefix}*.toml' in {species_dir}")
    if len(matches) > 1:
        raise ValueError(f"Multiple config files matching '{prefix}*.toml' in {species_dir}: {matches}")
    return matches[0]


def load_stage_config(
    species: str,
    stage: int,
    config_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Load a curriculum stage configuration from TOML.

    Args:
        species: Species name (e.g. "velociraptor", "brachiosaurus", "trex").
        stage: Curriculum stage number (1, 2, or 3).
        config_path: Optional explicit path to a TOML file. Overrides
            automatic discovery when provided.

    Returns:
        Dictionary with keys "name", "description", "env_kwargs",
        "ppo_kwargs", and "sac_kwargs".  Values in [env] that are lists
        are converted to tuples so they can be passed directly to the
        environment constructors.
    """
    if config_path is not None:
        path = Path(config_path)
    else:
        path = _find_stage_file(species, stage)

    with open(path, "rb") as f:
        raw = tomllib.load(f)

    stage_meta = raw.get("stage", {})
    env_raw = raw.get("env", {})
    ppo_raw = raw.get("ppo", {})
    sac_raw = raw.get("sac", {})
    curriculum_raw = raw.get("curriculum", {})

    # Convert lists to tuples for range parameters (e.g. prey_distance_range)
    env_kwargs = {}
    for key, value in env_raw.items():
        if isinstance(value, list):
            env_kwargs[key] = tuple(value)
        else:
            env_kwargs[key] = value

    return {
        "name": stage_meta.get("name", f"stage{stage}"),
        "description": stage_meta.get("description", ""),
        "env_kwargs": env_kwargs,
        "ppo_kwargs": dict(ppo_raw),
        "sac_kwargs": dict(sac_raw),
        "curriculum_kwargs": dict(curriculum_raw),
    }


def load_all_stages(species: str) -> Dict[int, Dict[str, Any]]:
    """Load all curriculum stage configs for a species.

    Returns:
        Dictionary mapping stage number (1, 2, 3) to stage config dicts.
    """
    return {stage: load_stage_config(species, stage) for stage in (1, 2, 3)}
