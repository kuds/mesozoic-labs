"""
Load curriculum stage configurations from TOML files.

Each species has a configs/<species>/ directory with one TOML file per stage,
named as the species' stage manifest declares (trex: stance.toml,
recovery.toml, locomotion.toml, behavior.toml; manifest-less species keep
their historical stage{N}_* names, which their synthesized manifest records).

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
import math
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .plant_contract import PlantIdentity

_logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]

#: The canonical species names, sorted, as the report scripts expose them
#: for argparse ``choices``.  Deliberately not the registry's key set: that
#: also carries aliases (raptor, t-rex, brachio, dibo) that are not part of
#: the CLI surface.
SPECIES_NAMES: tuple[str, ...] = (
    "brachiosaurus",
    "compsognathus",
    "compsognathus_robot",
    "dibothrosuchus",
    "trex",
    "velociraptor",
)


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

# Integer stage refs resolve through the stage manifest (declared or
# synthesized) — the old stage{N}_* glob lives on only inside the manifest
# synthesizer for manifest-less species.

#: Every top-level table a stage TOML may declare.  This is the complete set
#: the loader below reads — the scripts that open a stage TOML themselves
#: (``stance_quality_baseline``, ``zero_action_baseline``, the report
#: scripts) read only ``[env]``, the JAX and sweep paths go through
#: :func:`load_stage_config`, and ``stages.toml`` has its own reader in
#: ``stage_manifest``.  Anything else is rejected rather than ignored: a
#: misspelled ``[environment]`` used to load as an empty ``[env]`` and the
#: stage silently trained on constructor defaults (review CF4).
_STAGE_CONFIG_TABLES = frozenset({"stage", "env", "ppo", "sac", "jax", "curriculum"})
_ALGORITHM_TABLES = ("ppo", "sac", "jax")


def load_stage_config(
    species: str,
    stage: "int | str",
    config_path: str | None = None,
) -> dict[str, Any]:
    """Load a curriculum stage configuration from TOML.

    Args:
        species: Species name (e.g. "velociraptor", "brachiosaurus", "trex").
        stage: Either a legacy stage number (1, 2, or 3 — resolved through
            the stage manifest's legacy_number mapping, so existing callers
            and artifacts keep their meaning) or a semantic stage ID — any
            id the species' manifest declares, resolved through it.  Stages
            without a legacy number — recovery, every open id — are
            reachable only by ID.
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
    else:
        # Every stage reference resolves through the manifest: declared
        # manifests name their config files explicitly (trex's are id-named
        # as of 2026-08-20 — stance.toml, not stage1_balance.toml), and
        # synthesized manifests record the historical stage{N}_* filename
        # they were built from, so manifest-less species are unchanged.
        from .stage_manifest import load_stage_manifest

        entry = load_stage_manifest(species).resolve(stage)
        path = _CONFIGS_DIR / species / entry.config_file

    with open(path, "rb") as f:
        raw = tomllib.load(f)

    unknown_tables = sorted(set(raw) - _STAGE_CONFIG_TABLES)
    if unknown_tables:
        raise ValueError(
            f"{path}: unknown top-level table(s) {unknown_tables}; a stage config may declare only "
            f"{sorted(_STAGE_CONFIG_TABLES)}. A misspelled table would otherwise load as empty and the "
            "stage would train on class / library defaults — a different experiment, with no warning."
        )
    if not raw.get("env"):
        _logger.warning(
            "%s declares no [env] table (or an empty one): the stage will train on the environment "
            "constructor's defaults for every reward, termination and horizon parameter.",
            path,
        )
    if not any(raw.get(table) for table in _ALGORITHM_TABLES):
        _logger.warning(
            "%s declares no algorithm table (%s): every backend will train on library-default hyperparameters.",
            path,
            ", ".join(f"[{table}]" for table in _ALGORITHM_TABLES),
        )

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


def build_env(species: str, stage: "int | str", **env_overrides: Any) -> Any:
    """Construct the stage's environment from its committed config.

    The one construction path the report scripts and the frozen recovery
    gate share: every ``[env]`` key the stage TOML declares reaches the
    constructor, exactly as training builds the environment.  Keyword
    *env_overrides* replace individual ``[env]`` values for the diagnostics
    that need a deliberately different plant state -- the foot-sensor
    cross-check zeroes ``reset_noise_scale`` so the plant settles
    deterministically.  The registry import is deferred because
    ``species_registry`` imports ``train_base``, which imports this module.
    """
    from .species_registry import get_species_config

    config = load_stage_config(species, stage)
    env_class = get_species_config(species).env_class
    return env_class(**{**config["env_kwargs"], **env_overrides})


def _recorded_checkpoint_task_sha256(checkpoint: Path) -> str | None:
    """The task-fingerprint digest an SB3 checkpoint ZIP recorded, or ``None``.

    A digest-only view over
    :func:`~environments.shared.task_fingerprint.read_checkpoint_task_fingerprint`,
    which reads the archive's ``data`` member so the lineage can be written
    before — and independently of — the model load.  ``None`` for a
    checkpoint minted before fingerprints existed and for a non-SB3 file (a
    JAX ``.pkl``); both are honest "unknown parent task", never a guess.
    """
    from .task_fingerprint import read_checkpoint_task_fingerprint

    recorded = read_checkpoint_task_fingerprint(checkpoint)
    digest = recorded.get("task_sha256") if recorded is not None else None
    return digest if isinstance(digest, str) and digest else None


#: Run-block key recording how long a stage trained, in seconds, summed over
#: the sessions that reached a final save in this stage directory (decision
#: D-A15; a session stopped before its final save records nothing, and a CLI
#: resume into a fresh directory records only its own session).  Written by
#: :func:`record_stage_duration` when ``train_base.train`` saves the final
#: model rather than by :func:`save_stage_config`, which runs BEFORE
#: training: a node resumed by the notebook's resume cell and judged later
#: would otherwise report the judge session's 0.0 seconds.  Not a lineage
#: key — the audit ignores it.
STAGE_DURATION_KEY = "duration_seconds"


def read_stage_duration(stage_dir: str | Path) -> float | None:
    """The ``run.duration_seconds`` a stage directory records, or ``None``.

    ``None`` when ``stage_config.json`` is absent, unreadable, has no run
    block, or the key is missing or not a finite non-negative number — every
    case an honest "unknown", so a caller accumulating a resumed session's
    duration starts from nothing rather than from a guess.
    """
    path = Path(stage_dir) / "stage_config.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    run_block = data.get("run") if isinstance(data, dict) else None
    value = run_block.get(STAGE_DURATION_KEY) if isinstance(run_block, dict) else None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    seconds = float(value)
    return seconds if math.isfinite(seconds) and seconds >= 0.0 else None


def record_stage_duration(stage_dir: str | Path, duration_seconds: float) -> Path:
    """Record *duration_seconds* into ``stage_config.json``'s run block, atomically.

    Sets the value; it does not accumulate.  A caller resuming an
    interrupted stage reads the prior value with :func:`read_stage_duration`
    BEFORE it re-saves the stage config (which writes a fresh run block) and
    records the sum on exit.  A stage directory without ``stage_config.json``
    is an error: the duration is a property of a recorded stage, not a
    record on its own.
    """
    from .file_io import atomic_write_text

    seconds = float(duration_seconds)
    if isinstance(duration_seconds, bool) or not math.isfinite(seconds) or seconds < 0.0:
        raise ValueError(f"duration_seconds must be a finite non-negative number, not {duration_seconds!r}")
    path = Path(stage_dir) / "stage_config.json"
    if not path.is_file():
        raise FileNotFoundError(f"cannot record the stage duration: no stage_config.json in {stage_dir}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must hold a JSON object")
    run_block = data.get("run")
    if not isinstance(run_block, dict):
        run_block = {}
    run_block[STAGE_DURATION_KEY] = seconds
    data["run"] = run_block
    atomic_write_text(path, json.dumps(data, indent=2) + "\n")
    return path


#: Files whose presence means a stage directory is already a RECORDED stage
#: (decision D-A20): its config was written by an earlier session, or a gate
#: has judged it.  Writing a fresh stage into such a directory silently
#: overwrites that record, so :func:`refuse_occupied_stage_dir` refuses it
#: unless the load is an explicit same-stage resume.
STAGE_DIR_OCCUPANCY_FILES = ("stage_config.json", "gate_verdict.json")


class StageDirectoryOccupiedError(RuntimeError):
    """A stage directory already holds a recorded stage and the load is not a same-stage resume."""


def refuse_occupied_stage_dir(stage_dir: str | Path, *, task_load_mode: str | None) -> None:
    """Refuse to write a fresh stage into a directory that already records one (D-A20).

    Raises :class:`StageDirectoryOccupiedError` when *stage_dir* holds any
    of :data:`STAGE_DIR_OCCUPANCY_FILES` and *task_load_mode* is not
    ``"resume_same_stage"``.  A same-stage resume continues the stage the
    directory records, so it is the one load that may re-save its config
    there; every other write — from scratch, or entering from a parent under
    ``initialize_next_stage`` — is a new variant, and a variant is a new run.
    Callers pass the EFFECTIVE load mode: ``None`` when nothing is loaded,
    exactly as ``save_stage_config`` records ``load_mode``, so a
    ``resume_same_stage`` default without a checkpoint does not slip past
    the guard.  Never raises for a missing or empty directory.
    """
    directory = Path(stage_dir)
    found = [name for name in STAGE_DIR_OCCUPANCY_FILES if (directory / name).is_file()]
    if not found or task_load_mode == "resume_same_stage":
        return
    raise StageDirectoryOccupiedError(
        f"{directory} already records a stage ({', '.join(found)} present) and this is not a "
        f"same-stage resume (load mode {task_load_mode!r}); refusing to overwrite it. Give a new "
        "variant a fresh run directory (--output-dir / RUN_ID), or resume this stage explicitly with "
        "--load <checkpoint> --load-mode resume_same_stage (the notebook's RESUME cell)."
    )


#: Run-block keys recording where a stage's initial weights came from.  Read
#: back by the result-bundle audit under exactly these names (review RP4),
#: which audits each key when present and skips it when absent — so a
#: from-scratch stage writes none of them, and an unfingerprinted parent
#: leaves ``parent_task_sha256`` out rather than recording a null.
#: ``load_path`` names the file whose sha256 is recorded — ``<stem>.zip`` when
#: the trainer was handed an SB3 stem — so a parent inside the run directory
#: resolves to the manifest entry the audit cross-checks the hash against.
#: ``parent_run_id`` is present iff the parent was a certified ancestor
#: reused from ANOTHER run (BEHAVIOR_RECIPES_PLAN §4.2): its value is that
#: run's provenance ``run_id`` when it has one, else the run directory name.
#: A ``--load`` and a same-run curriculum handoff never write it.
LOAD_LINEAGE_KEYS = ("load_path", "load_mode", "parent_checkpoint_sha256", "parent_task_sha256", "parent_run_id")

#: A same-stage resume of a stage that ENTERED from its parent keeps that
#: edge in the :data:`LOAD_LINEAGE_KEYS` (the reuse rule's chain-by-digest
#: check, ``ancestors._check_chain``, and the audit read those) and records
#: the periodic checkpoint it continued from under these two keys instead:
#: the path as given to the trainer and that file's sha256.  Neither is a
#: lineage key — the resume continues the recorded stage, it does not
#: re-parent it — so a resumed-then-judged node stays reusable on top of
#: the parent it was trained from (BEHAVIOR_RECIPES_PLAN §4.7 branch 3).
RESUME_LINEAGE_KEYS = ("resume_load_path", "resume_checkpoint_sha256")

#: Run-block keys a WIDENED root records its parent under
#: (``environments/shared/scripts/widen_checkpoint.py``; BEHAVIOR_RECIPES_PLAN
#: §4.6 "Widening instead of retraining", decision D-C8): the parent archive
#: as given, that file's sha256, its VecNormalize sidecar's sha256, the task
#: digest the parent recorded (``None`` when it carried none), the parent's
#: ``policy_interface_sha256`` and ``policy_interface_revision`` (the
#: interface it was trained under, one revision behind), the parent run's
#: id (its provenance ``run_id``, else its directory name — ``ancestors
#: .run_id_for`` semantics) and ``widened_by`` (tool version and the commit
#: the widening ran at).  A widened root records its parent HERE, NOT under
#: :data:`LOAD_LINEAGE_KEYS`: ``ancestors._check_chain`` refuses a root that
#: entered under ``initialize_next_stage``, and the result-bundle audit binds
#: ``parent_run_id`` to an ``ancestors/`` record — a widened checkpoint is
#: the same policy under a wider interface, not a warm-start across an edge.
#: These keys are provenance the audit ignores; no reader consumes them
#: (``_recorded_load_lineage``, ``_recorded_edge_lineage`` and the audit read
#: only :data:`LOAD_LINEAGE_KEYS`).  The archive carries the same parent
#: hashes under its ``mesozoic_widen_lineage`` attribute.
WIDEN_LINEAGE_KEYS = (
    "widened_from_path",
    "widened_from_checkpoint_sha256",
    "widened_from_normalization_sha256",
    "widened_from_task_sha256",
    "widened_from_policy_interface_sha256",
    "widened_from_policy_interface_revision",
    "widened_from_run_id",
    "widened_by",
)

#: Besides *extra* and the lineage keys, the ``run`` block always records
#: ``hyperparameters_sha256`` — :func:`hyperparameters_sha256` over the
#: stage's algorithm block and stage-entry shaping keys — and, when the run
#: was given one, a free-text ``label`` (decision D-A21).  Neither is a
#: load-lineage key: they describe THIS stage's training recipe, not where
#: its weights came from.  ``reporting.bundles`` copies both into the
#: stage's ``provenance.deliverables`` record and the audit cross-checks
#: the digest against the run block it came from.

#: The stage-config key holding each algorithm's hyperparameter table —
#: the block ``save_stage_config`` records as ``"hyperparameters"`` and
#: :func:`hyperparameters_sha256` digests.
_ALGORITHM_KWARGS_KEYS = {"PPO": "ppo_kwargs", "SAC": "sac_kwargs", "JAX_PPO": "jax_kwargs"}

#: Prefixes of the ``curriculum_kwargs`` keys that shape a stage's ENTRY
#: (the warm-up and ramp callbacks): they change what a node is trained
#: under without touching the task fingerprint, so the digest covers them.
SHAPING_KEY_PREFIXES = ("warmup_", "ramp_")


def _algorithm_kwargs_key(algorithm: str) -> str:
    return _ALGORITHM_KWARGS_KEYS.get(algorithm.upper(), f"{algorithm.lower()}_kwargs")


def _json_ready(value: Any) -> Any:
    """Canonicalise tuples to lists (recursively), as the stage config on disk records them."""
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def _hyperparameter_view(algorithm: str, hyperparameters: Any, curriculum: Any) -> dict[str, Any]:
    """The digested view: the algorithm, its block, and the shaping keys of the curriculum block."""
    if not isinstance(hyperparameters, dict):
        hyperparameters = {}
    if not isinstance(curriculum, dict):
        curriculum = {}
    return {
        "algorithm": algorithm.upper(),
        "hyperparameters": _json_ready(hyperparameters),
        "shaping": {str(k): _json_ready(v) for k, v in curriculum.items() if str(k).startswith(SHAPING_KEY_PREFIXES)},
    }


def hyperparameters_sha256(stage_config: dict[str, Any], algorithm: str) -> str:
    """The ``sha256:<hex>`` digest of a stage's training recipe (decision D-A21).

    Hashes, as canonical JSON, ``{"algorithm": ALGORITHM, "hyperparameters":
    <the algorithm's kwargs table>, "shaping": <the curriculum_kwargs whose
    keys start with one of SHAPING_KEY_PREFIXES>}`` — exactly the block
    ``save_stage_config`` records as ``"hyperparameters"`` plus the
    stage-entry shaping knobs, with tuples canonicalised to lists as the
    JSON on disk has them.  Independent of key order.  Env kwargs, gate
    thresholds and everything else in the config leave it unchanged: the
    task fingerprint owns the task, this owns how it was trained.
    """
    from .result_bundle.hashing import canonical_json_sha256

    view = _hyperparameter_view(
        algorithm,
        stage_config.get(_algorithm_kwargs_key(algorithm), {}),
        stage_config.get("curriculum_kwargs", {}),
    )
    return canonical_json_sha256(view)


def recorded_hyperparameters_sha256(recorded_stage_config: Mapping[str, Any]) -> str | None:
    """The recipe digest a recorded ``stage_config.json`` states, else the one its blocks imply.

    The run block's ``hyperparameters_sha256`` when the stage recorded one
    (every stage saved since decision D-A21); otherwise the digest is
    DERIVED from the file's recorded ``"algorithm"``, top-level
    ``"hyperparameters"`` and the shaping keys of its ``"curriculum"`` —
    exactly what :func:`hyperparameters_sha256` hashes at save time, so a
    pre-D-A21 stage digests to what its run block would have recorded
    (decision D-B16: a replicate is never skipped for predating the field
    alone).  ``None`` when the file records no algorithm to digest under.
    """
    run_block = recorded_stage_config.get("run")
    if isinstance(run_block, Mapping):
        recorded = run_block.get("hyperparameters_sha256")
        if isinstance(recorded, str) and recorded.strip():
            return recorded
    algorithm = recorded_stage_config.get("algorithm")
    if not isinstance(algorithm, str) or not algorithm.strip():
        return None
    hyperparameters = recorded_stage_config.get("hyperparameters", {})
    curriculum = recorded_stage_config.get("curriculum", {})
    return hyperparameters_sha256(
        {
            _algorithm_kwargs_key(algorithm): dict(hyperparameters) if isinstance(hyperparameters, Mapping) else {},
            "curriculum_kwargs": dict(curriculum) if isinstance(curriculum, Mapping) else {},
        },
        algorithm,
    )


def hyperparameter_diff(
    stage_config: dict[str, Any],
    algorithm: str,
    recorded_stage_config: dict[str, Any],
) -> list[str]:
    """Sorted dotted keys on which a stage's recipe differs from a recorded ``stage_config.json``.

    Compares the current node's algorithm block and shaping keys (as
    :func:`hyperparameters_sha256` sees them) with the recorded file's
    top-level ``"hyperparameters"`` and the shaping keys of its
    ``"curriculum"``.  Keys are ``<algorithm>.<key>`` (``ppo.learning_rate``)
    and ``shaping.<key>`` (``shaping.warmup_timesteps``); a key present on
    one side only differs; a recorded ``"algorithm"`` other than the
    current one differs as ``algorithm``.  ``[]`` when nothing differs — in
    which case the two digests agree.
    """
    current = _hyperparameter_view(
        algorithm,
        stage_config.get(_algorithm_kwargs_key(algorithm), {}),
        stage_config.get("curriculum_kwargs", {}),
    )
    recorded_algorithm = recorded_stage_config.get("algorithm")
    recorded = _hyperparameter_view(
        str(recorded_algorithm) if isinstance(recorded_algorithm, str) and recorded_algorithm else algorithm,
        recorded_stage_config.get("hyperparameters", {}),
        recorded_stage_config.get("curriculum", {}),
    )
    differing: list[str] = []
    if current["algorithm"] != recorded["algorithm"]:
        differing.append("algorithm")
    for prefix, section in ((algorithm.lower(), "hyperparameters"), ("shaping", "shaping")):
        ours, theirs = current[section], recorded[section]
        for key in set(ours) | set(theirs):
            if key not in ours or key not in theirs or ours[key] != theirs[key]:
                differing.append(f"{prefix}.{key}")
    return sorted(differing)


def _checkpoint_lineage(
    load_path: str | None,
    load_mode: str | None,
    *,
    parent_run_id: str | None = None,
) -> dict[str, Any]:
    """The load-lineage keys for a stage's ``run`` block; empty from scratch.

    A loaded checkpoint records the path of the file hashed, the load mode,
    the ZIP's own sha256 and the task digest it carries — the load was
    validated against the current task at ``--load`` time but persisted
    nowhere readable.  A stem is recorded as the ``.zip`` SB3 appends to it
    (``train_curriculum`` and the notebook both hand SB3 a stem): the manifest
    hashes the ``.zip``, and a bare stem never matches a manifest key, so the
    audit's parent-hash cross-check would never fire.  A relative path stays
    relative; nothing else about the path is rewritten.  ``parent_run_id``
    is recorded only when it is a non-empty string — the audit reads
    absence as "the parent came from this run".
    """
    if not load_path:
        return {}
    from .result_bundle import sha256_file

    recorded_path = load_path
    checkpoint = Path(load_path)
    if not checkpoint.is_file() and not load_path.endswith(".zip"):
        # SB3 accepts the stem and appends .zip itself; hash the file it loads.
        recorded_path = load_path + ".zip"
        checkpoint = Path(recorded_path)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"cannot record load lineage: checkpoint not found at {load_path!r}")
    lineage: dict[str, Any] = {
        "load_path": recorded_path,
        "load_mode": load_mode,
        "parent_checkpoint_sha256": sha256_file(checkpoint),
    }
    parent_task_sha256 = _recorded_checkpoint_task_sha256(checkpoint)
    if parent_task_sha256 is not None:
        lineage["parent_task_sha256"] = parent_task_sha256
    if isinstance(parent_run_id, str) and parent_run_id:
        lineage["parent_run_id"] = parent_run_id
    return lineage


def _recorded_edge_lineage(stage_dir: Path) -> dict[str, Any]:
    """The ``initialize_next_stage`` lineage *stage_dir*'s ``stage_config.json`` records, else ``{}``.

    Read by :func:`save_stage_config` before it rewrites the run block on a
    same-stage resume.  A missing or unreadable config, a run block without
    a load, or one that records a ``resume_same_stage`` load (a root, or a
    plain ``--load``) all read as "no edge to keep".
    """
    path = stage_dir / "stage_config.json"
    if not path.is_file():
        return {}
    try:
        record: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    run_block = record.get("run") if isinstance(record, dict) else None
    if not isinstance(run_block, dict) or run_block.get("load_mode") != "initialize_next_stage":
        return {}
    return {key: run_block[key] for key in LOAD_LINEAGE_KEYS if key in run_block}


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
    *,
    load_path: str | None = None,
    load_mode: str | None = None,
    parent_run_id: str | None = None,
    label: str | None = None,
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
        load_path: The checkpoint this stage's weights were loaded from, as
            given to the trainer, or ``None`` for a from-scratch stage.
        load_mode: The task load mode the checkpoint was loaded under
            (``resume_same_stage`` / ``initialize_next_stage``).
        parent_run_id: The run the loaded checkpoint was reused from when it
            is a certified ancestor of ANOTHER run; ``None`` (or empty) for a
            parent trained in this run or a plain ``--load``.
        label: Free text naming this run (``--label`` / ``RUN_LABEL``),
            recorded as ``run["label"]`` only when it is a non-empty string
            (whitespace is stripped); ``None`` records nothing.

    The ``run`` block always carries ``hyperparameters_sha256``
    (:func:`hyperparameters_sha256` over the algorithm block and the
    shaping keys; decision D-A21) alongside *extra*, so it is always
    written; when a checkpoint was loaded it also carries the
    :data:`LOAD_LINEAGE_KEYS`, and a from-scratch stage carries none of
    them (the audit reads absence as "no parent").

    A ``resume_same_stage`` load into a directory whose existing
    ``stage_config.json`` records an ``initialize_next_stage`` entry keeps
    that edge's lineage keys verbatim (the resume continues the recorded
    stage on top of the same parent; the reuse rule chains on
    ``parent_checkpoint_sha256`` and the audit cross-checks it) and records
    the checkpoint it continued from under the :data:`RESUME_LINEAGE_KEYS`.
    Any other resume — a root, a plain ``--load`` — records the load under
    the lineage keys as before.

    Returns:
        Path to the written JSON file.
    """
    stage_dir = Path(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)

    algo_key = _algorithm_kwargs_key(algorithm)

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
    # D-A21: the recipe digest is always recorded (so the run block always
    # exists); the label only when one was given.
    run_block: dict[str, Any] = {
        **(extra or {}),
        "hyperparameters_sha256": hyperparameters_sha256(stage_config, algorithm),
    }
    if isinstance(label, str) and label.strip():
        run_block["label"] = label.strip()
    lineage = _checkpoint_lineage(load_path, load_mode, parent_run_id=parent_run_id)
    edge = _recorded_edge_lineage(stage_dir) if load_mode == "resume_same_stage" else {}
    if edge:
        # A same-stage resume continues the stage that entered from its
        # parent: keep the edge, record the continued-from checkpoint apart.
        run_block.update(edge)
        run_block["resume_load_path"] = lineage["load_path"]
        run_block["resume_checkpoint_sha256"] = lineage["parent_checkpoint_sha256"]
    else:
        run_block.update(lineage)
    data["run"] = run_block
    if plant_identity is not None:
        data["plant_identity"] = plant_identity.to_dict()
    if task_fingerprint is not None:
        data["task_fingerprint"] = dict(task_fingerprint)

    gpu_info = _detect_gpu_info()
    if gpu_info:
        data["gpu"] = gpu_info

    from .file_io import atomic_write_json

    # Atomic, like record_stage_duration's rewrite: a same-stage resume reads
    # this file back (its duration and parent edge), and a reclaim mid-write
    # would otherwise leave it truncated (CU-3; the bytes are unchanged).
    out_path = atomic_write_json(stage_dir / "stage_config.json", data)
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
    * Every ancestor record of a node reused from another run
      (``ancestors/<stage_id>/*`` — ``ancestor.json``, the copied
      ``gate_verdict.json`` and stage config; never a checkpoint) →
      ``training/<species>/<run>/ancestors/<stage_id>/``, because the
      bundle audit requires them and a mirror without them cannot audit

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

    # 2. Upload per-stage artifacts — every stage directory the run wrote,
    # in either naming generation (stage{N}, bare ids like "recovery", or
    # the NN_id form new runs use), recognised by the one species-aware
    # helper so any id this species' manifest declares uploads as a stage
    # and nothing else (``models`` at run level, the ``ancestors`` records
    # mirrored separately below) ever does.  Iterating the disk instead of
    # a fixed 1..3 range keeps semantic-only stages (recovery) from silently
    # never syncing.
    from .stage_manifest import stage_ref_from_dirname

    stage_dir_list = [
        child
        for child in sorted(base_dir.iterdir())
        if child.is_dir() and stage_ref_from_dirname(child.name, species=species) is not None
    ]
    for stage_dir in stage_dir_list:
        # Mirror whatever the run actually named the directory.
        gcs_stage_prefix = f"{gcs_run_prefix}/{stage_dir.name}"

        # Summaries and analysis sidecars.  metrics.json / stage_config.json
        # are what `sweep collect-results` consumes, so uploading them makes
        # the run collectable from GCS alone; gate_verdict.json lives at the
        # stage root and is what makes the stage reusable as an ancestor.
        for name in (
            "stage_summary.txt",
            "stage_config.json",
            "plant_identity.json",
            "task_fingerprint.json",
            "gate_verdict.json",
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
        # SaveVecNormalizeCallback), plus the stage's final checkpoint pair.
        # The final files are stage_label-prefixed (stage1_final.zip,
        # recovery_final.zip, ...), so glob rather than reconstruct the
        # prefix — it depends on how the stage was invoked.
        model_files = [stage_model_dir / "best_model.zip", stage_model_dir / "best_model_vecnorm.pkl"]
        model_files.extend(sorted(stage_model_dir.glob("*_final.zip")))
        model_files.extend(sorted(stage_model_dir.glob("*_final_vecnorm.pkl")))
        for model_file in model_files:
            if model_file.exists():
                _upload_to_gcs(
                    model_file, bucket, f"{gcs_model_prefix}/{model_file.name}", project=project, client=client
                )

    # 3. Ancestor records (BEHAVIOR_RECIPES_PLAN §4.2).  A node reused from
    # another run through --trunk-from leaves ancestors/<stage_id>/ holding
    # ancestor.json and verbatim copies of the ancestor's gate_verdict.json,
    # stage_config.json, task_fingerprint.json and plant_identity.json —
    # small records, never the checkpoint.  The bundle audit requires every
    # record the provenance claims, so a mirror without them audits as a
    # conflict; every file in every record directory uploads at its own
    # relative path.
    from .result_bundle.constants import ANCESTORS_DIRNAME

    ancestors_dir = base_dir / ANCESTORS_DIRNAME
    if ancestors_dir.is_dir():
        for record_dir in sorted(ancestors_dir.iterdir()):
            if not record_dir.is_dir():
                continue
            for record_file in sorted(record_dir.iterdir()):
                if record_file.is_file():
                    _upload_to_gcs(
                        record_file,
                        bucket,
                        f"{gcs_run_prefix}/{ANCESTORS_DIRNAME}/{record_dir.name}/{record_file.name}",
                        project=project,
                        client=client,
                    )
