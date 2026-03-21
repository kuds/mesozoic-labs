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

from __future__ import annotations

import inspect
import json
import logging
from pathlib import Path
from typing import Any

_logger = logging.getLogger(__name__)

try:
    import tomllib
except ImportError:
    import tomli as tomllib

_REPO_ROOT = Path(__file__).resolve().parents[2]


def get_library_version() -> str:
    """Return the mesozoic-labs package version string.

    Tries ``importlib.metadata`` first (works when the package is installed),
    then falls back to parsing ``pyproject.toml`` at the repository root.
    """
    try:
        from importlib.metadata import version

        return version("mesozoic-labs")
    except Exception:
        pass

    pyproject = _REPO_ROOT / "pyproject.toml"
    if pyproject.exists():
        with open(pyproject, "rb") as f:
            data = tomllib.load(f)
        return str(data.get("project", {}).get("version", "unknown"))

    return "unknown"


# Known GPU short-names extracted from full device strings.
_GPU_SHORT_NAMES = ("A100", "H100", "L4", "L40", "T4", "V100", "A10G", "A10", "RTX")


def _detect_gpu_info() -> dict[str, Any]:
    """Return a dict with GPU details, or an empty dict if no GPU is available."""
    # Try torch first (most accurate when available).
    try:
        import torch

        if torch.cuda.is_available():
            full_name = torch.cuda.get_device_name(0)
            short_name = full_name
            for short in _GPU_SHORT_NAMES:
                if short in full_name.upper():
                    short_name = short
                    break
            props = torch.cuda.get_device_properties(0)
            return {
                "gpu_model": short_name,
                "gpu_full_name": full_name,
                "gpu_memory_gb": round(props.total_mem / 1e9, 1),
                "cuda_version": torch.version.cuda or "",
            }
    except Exception:
        pass

    # Fallback: query nvidia-smi directly (works without torch).
    return _detect_gpu_info_nvidia_smi()


def _detect_gpu_info_nvidia_smi() -> dict[str, Any]:
    """Detect GPU info via nvidia-smi. Returns empty dict on failure."""
    import subprocess

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return {}
        line = result.stdout.strip().split("\n")[0]
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            return {}
        full_name, memory_mb, driver_version = parts[0], parts[1], parts[2]
        short_name = full_name
        for short in _GPU_SHORT_NAMES:
            if short in full_name.upper():
                short_name = short
                break
        return {
            "gpu_model": short_name,
            "gpu_full_name": full_name,
            "gpu_memory_gb": round(float(memory_mb) / 1024, 1),
            "driver_version": driver_version,
        }
    except Exception:
        return {}


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
    config_path: str | None = None,
) -> dict[str, Any]:
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


def load_all_stages(species: str) -> dict[int, dict[str, Any]]:
    """Load all curriculum stage configs for a species.

    Returns:
        Dictionary mapping stage number (1, 2, 3) to stage config dicts.
    """
    return {stage: load_stage_config(species, stage) for stage in (1, 2, 3)}


def save_stage_config(
    stage_dir: str | Path,
    stage: int,
    stage_config: dict[str, Any],
    algorithm: str,
    extra: dict[str, Any] | None = None,
    env_class: type | None = None,
    species: str | None = None,
) -> Path:
    """Save the reward weights and model hyperparameters for a stage to JSON.

    Writes ``stage_config.json`` into *stage_dir* with the full reward signal
    (env_kwargs), the algorithm hyperparameters, curriculum thresholds, and any
    extra run-level metadata (seed, n_envs, etc.).

    When *env_class* is provided, constructor defaults for parameters not
    already present in the TOML-derived ``env_kwargs`` are merged in so
    that the saved JSON captures the effective configuration (including
    values like ``healthy_z_range`` that may rely on class defaults).

    Args:
        stage_dir: Directory for this stage (e.g. ``run_dir/stage1``).
        stage: Stage number (1, 2, or 3).
        stage_config: The config dict returned by :func:`load_stage_config`.
        algorithm: Algorithm name (``"PPO"`` or ``"SAC"``).
        extra: Optional dict of additional metadata to include at the top level
            (e.g. ``{"seed": 42, "n_envs": 4}``).
        env_class: Optional environment class whose ``__init__`` defaults are
            merged into ``env_kwargs`` for completeness.
        species: Optional species name (e.g. ``"velociraptor"``, ``"trex"``).

    Returns:
        Path to the written JSON file.
    """
    stage_dir = Path(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)

    algo_key = "ppo_kwargs" if algorithm.upper() == "PPO" else "sac_kwargs"

    # Start with env class constructor defaults so that the saved JSON
    # captures the full effective configuration, then overlay with
    # explicit TOML values (which take precedence).
    env_kwargs: dict[str, Any] = {}
    if env_class is not None:
        try:
            sig = inspect.signature(env_class)
            skip = {"self", "render_mode"}
            for name, param in sig.parameters.items():
                if name in skip or param.default is inspect.Parameter.empty:
                    continue
                env_kwargs[name] = param.default
        except (ValueError, TypeError):
            pass

    # Overlay TOML-derived values and convert tuples to lists for JSON
    for key, value in stage_config.get("env_kwargs", {}).items():
        env_kwargs[key] = list(value) if isinstance(value, tuple) else value

    # Also convert any defaults that were tuples
    for key, value in env_kwargs.items():
        if isinstance(value, tuple):
            env_kwargs[key] = list(value)

    data: dict[str, Any] = {
        "species": species or "",
        "stage": stage,
        "name": stage_config.get("name", ""),
        "description": stage_config.get("description", ""),
        "algorithm": algorithm.upper(),
        "library_version": get_library_version(),
        "reward_weights": env_kwargs,
        "hyperparameters": stage_config.get(algo_key, {}),
        "curriculum": stage_config.get("curriculum_kwargs", {}),
    }
    if extra:
        data["run"] = extra

    gpu_info = _detect_gpu_info()
    if gpu_info:
        data["gpu"] = gpu_info

    out_path = stage_dir / "stage_config.json"
    out_path.write_text(json.dumps(data, indent=2) + "\n")
    return out_path


def append_stage_result_csv(csv_path: str | Path, data: dict) -> Path:
    """Append one stage training result row to a CSV file.

    Delegates to :func:`environments.shared.reporting.write_results_csv`
    in append mode, which creates the file with a header on the first call
    and expands the column set if later calls introduce new keys.

    Args:
        csv_path: Path to the CSV file (created if it does not exist).
        data: Ordered dict of column name → value for this row.

    Returns:
        Path to the CSV file.
    """
    from .reporting import write_results_csv

    return write_results_csv([data], csv_path, append=True)


def _upload_to_gcs(
    local_path: str | Path,
    bucket_name: str,
    gcs_path: str,
    project: str | None = None,
) -> bool:
    """Upload a local file to Google Cloud Storage.

    Args:
        local_path: Path to the local file to upload.
        bucket_name: GCS bucket name (without ``gs://`` prefix).
        gcs_path: Destination blob path inside the bucket.
        project: GCP project ID (optional, uses default if *None*).

    Returns:
        *True* if the upload succeeded, *False* otherwise.
    """
    local_path = Path(local_path)
    if not local_path.exists():
        _logger.warning("Cannot upload to GCS: local file not found: %s", local_path)
        return False

    try:
        from google.cloud import storage as _gcs

        client = _gcs.Client(project=project)
        bucket = client.bucket(bucket_name)
        bucket.blob(gcs_path).upload_from_filename(str(local_path))
        _logger.info("Uploaded to GCS: gs://%s/%s", bucket_name, gcs_path)
        return True
    except Exception as exc:
        _logger.warning(
            "Failed to upload %s to GCS: %s. Local copy remains at: %s",
            gcs_path,
            exc,
            local_path,
        )
        return False


def upload_curriculum_artifacts(
    base_dir: str | Path,
    species: str,
    algorithm: str,
    bucket: str | None = None,
    project: str | None = None,
) -> None:
    """Upload curriculum training artifacts to GCS.

    Uploads:
    * ``curriculum_results.csv`` → ``training/<species>/<run>/curriculum_results.csv``
    * ``training_summary.txt`` → ``training/<species>/<run>/training_summary.txt``
    * Each stage's ``best_model.zip`` and ``stage<N>_final.zip`` →
      ``training/<species>/<run>/stage<N>/models/``
    * Each stage's ``stage_summary.txt`` →
      ``training/<species>/<run>/stage<N>/stage_summary.txt``
    * Each stage's replay videos (``*.mp4``) →
      ``training/<species>/<run>/stage<N>/``

    When *bucket* is ``None`` (no GCP info provided), this function is a
    no-op and all artifacts remain local only.

    Args:
        base_dir: The curriculum run's base directory
            (e.g. ``logs/velociraptor/curriculum_20240228_150000``).
        species: Species name (``"velociraptor"``, ``"brachiosaurus"``, ``"trex"``).
        algorithm: Algorithm name (``"ppo"`` or ``"sac"``).
        bucket: GCS bucket name (without ``gs://`` prefix).  Pass *None* to
            skip cloud upload and keep artifacts local only.
        project: GCP project ID (optional, uses default if *None*).
    """
    base_dir = Path(base_dir)

    if bucket is None:
        _logger.info(
            "No GCS bucket specified — curriculum artifacts saved locally only: %s",
            base_dir,
        )
        return

    run_name = base_dir.name  # e.g. curriculum_20240228_150000
    gcs_run_prefix = f"training/{species}/{run_name}"

    # 1. Upload curriculum_results.csv
    csv_path = base_dir / "curriculum_results.csv"
    if csv_path.exists():
        _upload_to_gcs(csv_path, bucket, f"{gcs_run_prefix}/curriculum_results.csv", project=project)

    # 2. Upload training_summary.txt
    training_summary = base_dir / "training_summary.txt"
    if training_summary.exists():
        _upload_to_gcs(training_summary, bucket, f"{gcs_run_prefix}/training_summary.txt", project=project)

    # 3. Upload per-stage artifacts
    for stage in range(1, 4):
        stage_dir = base_dir / f"stage{stage}"
        if not stage_dir.is_dir():
            continue

        gcs_stage_prefix = f"{gcs_run_prefix}/stage{stage}"

        # stage_summary.txt
        stage_summary = stage_dir / "stage_summary.txt"
        if stage_summary.exists():
            _upload_to_gcs(stage_summary, bucket, f"{gcs_stage_prefix}/stage_summary.txt", project=project)

        # Replay videos (*.mp4)
        for video in stage_dir.glob("*.mp4"):
            _upload_to_gcs(video, bucket, f"{gcs_stage_prefix}/{video.name}", project=project)

        # Models
        stage_model_dir = stage_dir / "models"
        if not stage_model_dir.is_dir():
            continue

        gcs_model_prefix = f"{gcs_stage_prefix}/models"

        # best_model.zip + matched vecnorm (from EvalCallback + SaveVecNormalizeCallback)
        best_model = stage_model_dir / "best_model.zip"
        if best_model.exists():
            _upload_to_gcs(best_model, bucket, f"{gcs_model_prefix}/best_model.zip", project=project)
        best_vecnorm = stage_model_dir / "best_model_vecnorm.pkl"
        if best_vecnorm.exists():
            _upload_to_gcs(best_vecnorm, bucket, f"{gcs_model_prefix}/best_model_vecnorm.pkl", project=project)

        # stage<N>_final.zip + vecnorm
        final_model = stage_model_dir / f"stage{stage}_final.zip"
        if final_model.exists():
            _upload_to_gcs(final_model, bucket, f"{gcs_model_prefix}/stage{stage}_final.zip", project=project)

        vecnorm = stage_model_dir / f"stage{stage}_final_vecnorm.pkl"
        if vecnorm.exists():
            _upload_to_gcs(vecnorm, bucket, f"{gcs_model_prefix}/stage{stage}_final_vecnorm.pkl", project=project)
