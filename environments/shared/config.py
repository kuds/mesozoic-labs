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

import json
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import tomllib
except ImportError:
    import tomli as tomllib

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


def save_stage_config(
    stage_dir: str | Path,
    stage: int,
    stage_config: Dict[str, Any],
    algorithm: str,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """Save the reward weights and model hyperparameters for a stage to JSON.

    Writes ``stage_config.json`` into *stage_dir* with the full reward signal
    (env_kwargs), the algorithm hyperparameters, curriculum thresholds, and any
    extra run-level metadata (seed, n_envs, etc.).

    Args:
        stage_dir: Directory for this stage (e.g. ``run_dir/stage1``).
        stage: Stage number (1, 2, or 3).
        stage_config: The config dict returned by :func:`load_stage_config`.
        algorithm: Algorithm name (``"PPO"`` or ``"SAC"``).
        extra: Optional dict of additional metadata to include at the top level
            (e.g. ``{"seed": 42, "n_envs": 4}``).

    Returns:
        Path to the written JSON file.
    """
    stage_dir = Path(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)

    algo_key = "ppo_kwargs" if algorithm.upper() == "PPO" else "sac_kwargs"

    # Convert tuples back to lists for JSON serialisation
    env_kwargs = {}
    for key, value in stage_config.get("env_kwargs", {}).items():
        env_kwargs[key] = list(value) if isinstance(value, tuple) else value

    data: Dict[str, Any] = {
        "stage": stage,
        "name": stage_config.get("name", ""),
        "description": stage_config.get("description", ""),
        "algorithm": algorithm.upper(),
        "reward_weights": env_kwargs,
        "hyperparameters": stage_config.get(algo_key, {}),
        "curriculum": stage_config.get("curriculum_kwargs", {}),
    }
    if extra:
        data["run"] = extra

    out_path = stage_dir / "stage_config.json"
    out_path.write_text(json.dumps(data, indent=2) + "\n")
    return out_path


def append_stage_result_csv(csv_path: str | Path, data: dict) -> Path:
    """Append one stage training result row to a CSV file.

    Creates the file with a header row on the first call; subsequent calls
    append rows without re-writing the header.  If a later call introduces
    new keys that were not in the original header, the file is rewritten with
    the expanded column set so no data is silently dropped.

    All values in *data* should be scalars (strings, numbers, booleans).
    Non-scalar values are converted to their string representation.

    Args:
        csv_path: Path to the CSV file (created if it does not exist).
        data: Ordered dict of column name → value for this row.

    Returns:
        Path to the CSV file.
    """
    import csv as _csv

    csv_path = Path(csv_path)

    if not csv_path.exists():
        with open(csv_path, "w", newline="") as f:
            writer = _csv.DictWriter(f, fieldnames=list(data.keys()), extrasaction="ignore")
            writer.writeheader()
            writer.writerow(data)
    else:
        with open(csv_path, "r", newline="") as f:
            reader = _csv.DictReader(f)
            existing_fieldnames: list[str] = list(reader.fieldnames or [])
            existing_rows = list(reader)

        new_keys = [k for k in data if k not in existing_fieldnames]
        if new_keys:
            # Rewrite with expanded header so no column is silently dropped
            all_fieldnames = existing_fieldnames + new_keys
            with open(csv_path, "w", newline="") as f:
                writer = _csv.DictWriter(f, fieldnames=all_fieldnames, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(existing_rows)
                writer.writerow(data)
        else:
            with open(csv_path, "a", newline="") as f:
                writer = _csv.DictWriter(f, fieldnames=existing_fieldnames, extrasaction="ignore")
                writer.writerow(data)

    return csv_path
