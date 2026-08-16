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
import tomllib
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .plant_contract import PlantIdentity

_logger = logging.getLogger(__name__)

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


def get_git_commit() -> str:
    """Return the repository's current git commit hash, or ``"unknown"``.

    Runs ``git rev-parse HEAD`` from the repository root so it works even when
    the process working directory is elsewhere (e.g. a Colab notebook whose CWD
    is ``/content``). Falls back to the ``GITHUB_SHA`` environment variable (set
    in CI) before giving up, so a saved stage config records the exact code
    revision that produced the run for reproducibility.
    """
    import os
    import subprocess

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except Exception:
        pass
    return os.environ.get("GITHUB_SHA", "unknown")


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
                "gpu_memory_gb": round(props.total_memory / 1e9, 1),
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
    stage: "int | str",
    config_path: str | None = None,
) -> dict[str, Any]:
    """Load a curriculum stage configuration from TOML.

    Args:
        species: Species name (e.g. "velociraptor", "brachiosaurus", "trex").
        stage: Either a legacy stage number (1, 2, or 3 — resolved through
            the historical ``stage{N}_*`` file prefix, so existing callers
            and artifacts keep their meaning) or a semantic stage ID
            (``"stance"``/``"recovery"``/``"locomotion"``/``"behavior"``,
            resolved through the species' stage manifest).  Stages without
            a legacy number — recovery — are reachable only by ID.
        config_path: Optional explicit path to a TOML file. Overrides
            automatic discovery when provided.

    Returns:
        Dictionary with keys "name", "description", "env_kwargs",
        "ppo_kwargs", "sac_kwargs", "jax_kwargs", and
        "curriculum_kwargs".  Values in [env] that are lists are
        converted to tuples so they can be passed directly to the
        environment constructors.
    """
    if config_path is not None:
        path = Path(config_path)
    elif isinstance(stage, str):
        from .stage_manifest import load_stage_manifest

        entry = load_stage_manifest(species).by_id(stage)
        path = _CONFIGS_DIR / species / entry.config_file
    else:
        path = _find_stage_file(species, stage)

    with open(path, "rb") as f:
        raw = tomllib.load(f)

    stage_meta = raw.get("stage", {})
    env_raw = raw.get("env", {})
    ppo_raw = raw.get("ppo", {})
    sac_raw = raw.get("sac", {})
    jax_raw = raw.get("jax", {})
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
        "jax_kwargs": dict(jax_raw),
        "curriculum_kwargs": dict(curriculum_raw),
    }


def load_all_stages(species: str) -> "dict[int | str, dict[str, Any]]":
    """Load every stage config the species' manifest declares.

    Returns:
        Dictionary keyed the way each stage is referenced: legacy stages by
        their historical number (1, 2, 3 — unchanged for every existing
        consumer), stages without a numeric history (recovery) by their
        semantic ID.  Iteration order is the manifest's curriculum order.
    """
    from .stage_manifest import load_stage_manifest

    manifest = load_stage_manifest(species)
    configs: "dict[int | str, dict[str, Any]]" = {}
    for entry in manifest.stages:
        key: "int | str" = entry.legacy_number if entry.legacy_number is not None else entry.id
        configs[key] = load_stage_config(species, key)
    return configs


def save_stage_config(
    stage_dir: str | Path,
    stage: "int | str",
    stage_config: dict[str, Any],
    algorithm: str,
    extra: dict[str, Any] | None = None,
    env_class: type | None = None,
    species: str | None = None,
    plant_identity: PlantIdentity | None = None,
    task_fingerprint: dict[str, Any] | None = None,
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
        plant_identity: Optional current plant identity.  When supplied it is
            embedded in the config and written as ``plant_identity.json``.

    Returns:
        Path to the written JSON file.
    """
    stage_dir = Path(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)

    _ALGO_KEY_MAP = {"PPO": "ppo_kwargs", "SAC": "sac_kwargs", "JAX_PPO": "jax_kwargs"}
    algo_key = _ALGO_KEY_MAP.get(algorithm.upper(), f"{algorithm.lower()}_kwargs")

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
        "git_commit": get_git_commit(),
        "reward_weights": env_kwargs,
        "hyperparameters": stage_config.get(algo_key, {}),
        "curriculum": stage_config.get("curriculum_kwargs", {}),
    }
    if extra:
        data["run"] = extra
    if plant_identity is not None:
        data["plant_identity"] = plant_identity.to_dict()
    if task_fingerprint is not None:
        data["task_fingerprint"] = dict(task_fingerprint)

    gpu_info = _detect_gpu_info()
    if gpu_info:
        data["gpu"] = gpu_info

    out_path = stage_dir / "stage_config.json"
    out_path.write_text(json.dumps(data, indent=2) + "\n")
    if plant_identity is not None:
        from .plant_contract import write_plant_identity

        write_plant_identity(stage_dir / "plant_identity.json", plant_identity)
    if task_fingerprint is not None:
        from .task_fingerprint import write_task_fingerprint

        write_task_fingerprint(stage_dir / "task_fingerprint.json", task_fingerprint)
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
    client=None,
) -> bool:
    """Upload a local file to Google Cloud Storage.

    Args:
        local_path: Path to the local file to upload.
        bucket_name: GCS bucket name (without ``gs://`` prefix).
        gcs_path: Destination blob path inside the bucket.
        project: GCP project ID (optional, uses default if *None*).
        client: Optional pre-built ``google.cloud.storage.Client`` to reuse
            across uploads (avoids one auth handshake per file).

    Returns:
        *True* if the upload succeeded, *False* otherwise.
    """
    local_path = Path(local_path)
    if not local_path.exists():
        _logger.warning("Cannot upload to GCS: local file not found: %s", local_path)
        return False

    try:
        if client is None:
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
    * Each stage's replay videos (``replays/*.mp4``, or ``*.mp4`` in a
      legacy flat stage directory) →
      ``training/<species>/<run>/stage<N>/``, at the same relative path

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

    # One client for the whole batch (each _upload_to_gcs call would
    # otherwise perform its own auth handshake).  Best-effort: on failure,
    # fall back to per-file client creation inside _upload_to_gcs.
    client = None
    try:
        from google.cloud import storage as _gcs

        client = _gcs.Client(project=project)
    except Exception as exc:
        _logger.warning("Could not create shared GCS client (%s).", exc)

    # 1. Upload run-level artifacts
    for name in ("curriculum_results.csv", "training_summary.txt", "plant_identity.json"):
        run_file = base_dir / name
        if run_file.exists():
            _upload_to_gcs(run_file, bucket, f"{gcs_run_prefix}/{name}", project=project, client=client)

    # 2. Upload per-stage artifacts
    for stage in range(1, 4):
        stage_dir = base_dir / f"stage{stage}"
        if not stage_dir.is_dir():
            continue

        gcs_stage_prefix = f"{gcs_run_prefix}/stage{stage}"

        # Summaries and analysis sidecars.  metrics.json / stage_config.json
        # are what `sweep collect-results` consumes, so uploading them makes
        # the run collectable from GCS alone.
        for name in (
            "stage_summary.txt",
            "stage_config.json",
            "plant_identity.json",
            "task_fingerprint.json",
            "metrics.json",
            "evaluations.npz",
            "diagnostics.npz",
        ):
            sidecar = stage_dir / name
            if sidecar.exists():
                _upload_to_gcs(sidecar, bucket, f"{gcs_stage_prefix}/{name}", project=project, client=client)

        # Replay videos.  Resolved through stage_layout rather than a local
        # glob so both the nested `replays/` directory and the legacy flat
        # layout upload, and so a future move cannot silently stop uploading
        # them the way a `glob("*.mp4")` here would.  The relative path is
        # preserved, so GCS mirrors the run directory.
        #
        # Deliberately videos only, matching what this function has always
        # uploaded.  `iter_generated_artifacts` would also sweep in the
        # figures and the per-frame stance CSVs — the latter are ~1.7 MB
        # each — and quietly enlarging what lands in someone's bucket is not
        # a layout change.  Widening the scope is a separate decision.
        from .reporting import stage_layout

        for artifact in stage_layout.iter_replay_files(stage_dir):
            if artifact.suffix != ".mp4":
                continue
            relative = artifact.relative_to(stage_dir).as_posix()
            _upload_to_gcs(artifact, bucket, f"{gcs_stage_prefix}/{relative}", project=project, client=client)

        # Models
        stage_model_dir = stage_dir / "models"
        if not stage_model_dir.is_dir():
            continue

        gcs_model_prefix = f"{gcs_stage_prefix}/models"

        # best_model.zip + matched vecnorm (from EvalCallback +
        # SaveVecNormalizeCallback), stage<N>_final.zip + vecnorm.
        for name in (
            "best_model.zip",
            "best_model_vecnorm.pkl",
            f"stage{stage}_final.zip",
            f"stage{stage}_final_vecnorm.pkl",
        ):
            model_file = stage_model_dir / name
            if model_file.exists():
                _upload_to_gcs(model_file, bucket, f"{gcs_model_prefix}/{name}", project=project, client=client)
