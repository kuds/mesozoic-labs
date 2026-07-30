"""Schema-v2 ``summary.json`` construction.

Turns in-memory stage results into the backend-independent public result
summary, including its provenance block."""

from __future__ import annotations

import json as _json
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from .formatting import _optional_metric, format_duration_hms, parse_optional_bool


def _backend_version(algorithm: str) -> "str | None":
    """Installed version of the training backend, or ``None`` if undetectable."""
    package = "jax" if algorithm.upper().startswith("JAX") else "stable_baselines3"
    try:
        from importlib.metadata import version

        return version(package)
    except Exception:
        return None


def _run_provenance(overrides: "Mapping[str, Any] | None" = None) -> dict[str, Any]:
    """Build a summary ``provenance`` block, recording the repository commit.

    Defaults are conservative: a freshly generated summary is ``historical`` /
    ``unverified``. Per the results contract, ``current`` or ``verified`` may
    only be claimed once the model hash, config hash, backend version, and
    evaluation-episode count are all recorded. The repository commit is filled
    automatically so a generated summary ties back to the exact code revision;
    callers with full identity can pass ``overrides`` to complete and certify it.
    """
    from ..config import get_git_commit

    provenance: dict[str, Any] = {
        "model_revision_status": "historical",
        "verification_status": "unverified",
        "evaluation_episodes": None,
        "repository_commit": get_git_commit(),
        "model_hash": None,
        "config_hash": None,
    }
    if overrides:
        provenance.update(overrides)
    return provenance


def _canonical_stage_summary(result: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize one in-memory stage result to the public schema-v2 shape."""
    stage = int(result["stage"])
    duration = float(result.get("duration_seconds", 0.0))
    passed = parse_optional_bool(result.get("publication_gate_passed"))
    if passed is None:
        raise ValueError(f"stage {stage} is missing an explicit boolean publication_gate_passed value")

    stage_summary: dict[str, Any] = {
        "name": str(result.get("name") or f"Stage {stage}"),
        "description": str(result.get("description") or f"Curriculum stage {stage}"),
        "timesteps": int(result["timesteps"]),
        "best_eval_reward": _optional_metric(result.get("best_eval_reward"), digits=2),
        "best_eval_std": _optional_metric(result.get("best_eval_std"), digits=2),
        "best_eval_step": _optional_metric(
            result.get("best_eval_timestep", result.get("best_eval_step")),
        ),
        "final_eval_reward": _optional_metric(result.get("mean_reward"), digits=2),
        "final_eval_std": _optional_metric(result.get("std_reward"), digits=2),
        "avg_episode_length": _optional_metric(result.get("mean_episode_length"), digits=1),
        "avg_episode_length_std": _optional_metric(result.get("std_episode_length"), digits=1),
        "avg_forward_vel": _optional_metric(result.get("mean_forward_vel"), digits=3),
        "avg_forward_vel_std": _optional_metric(result.get("std_forward_vel"), digits=3),
        "mean_distance_traveled": _optional_metric(result.get("mean_distance_traveled"), digits=3),
        "mean_success_rate": _optional_metric(result.get("mean_success_rate"), digits=4),
        "training_time_seconds": round(duration, 1),
        "training_time": format_duration_hms(duration),
        "stage_passed": passed,
        "publication_gate_passed": passed,
    }
    selected_metrics = {
        "selected_model_reward": ("best_model_reward", 2),
        "selected_model_reward_std": ("best_model_std_reward", 2),
        "selected_model_episode_length": ("best_model_length", 1),
        "selected_model_episode_length_std": ("best_model_std_length", 1),
        "selected_model_forward_vel": ("best_model_fwd_vel", 3),
        "selected_model_forward_vel_std": ("best_model_std_fwd_vel", 3),
        "selected_model_distance": ("best_model_distance", 3),
        "selected_model_success_rate": ("best_model_success_rate", 4),
    }
    for output_key, (result_key, digits) in selected_metrics.items():
        if result_key in result:
            stage_summary[output_key] = _optional_metric(result.get(result_key), digits=digits)
    return stage_summary


def build_result_summary(
    stage_results_list: Sequence[Mapping[str, Any]],
    species: str,
    algorithm: str,
    seed: int,
    *,
    hardware: str = "Google Colab",
    provenance: Mapping[str, Any] | None = None,
    backend: str | None = None,
    backend_version: str | None = None,
    parallel_envs: int | None = None,
    run_id: str | None = None,
    result_date: str | None = None,
    plant_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one backend-independent result summary from normalized stage data."""
    from ..result_bundle import canonical_algorithm, canonical_backend

    if not stage_results_list:
        raise ValueError("at least one stage result is required")
    stages: dict[str, dict[str, Any]] = {}
    for result in stage_results_list:
        stage_key = str(int(result["stage"]))
        if stage_key in stages:
            raise ValueError(f"duplicate stage result: {stage_key}")
        stages[stage_key] = _canonical_stage_summary(result)

    total_duration = sum(float(stage["training_time_seconds"] or 0.0) for stage in stages.values())
    total_timesteps = sum(int(result["timesteps"]) for result in stage_results_list)
    final_stage_key = max(stages, key=int)
    public_algorithm = canonical_algorithm(algorithm)
    public_backend = canonical_backend(algorithm, backend)
    summary: dict[str, Any] = {
        "schema_version": 2,
        "bundle_status": "complete" if set(stages) == {"1", "2", "3"} else "partial",
        "species": species,
        "algorithm": public_algorithm,
        "backend": public_backend,
        "backend_version": backend_version
        if backend_version is not None
        else _backend_version("JAX_PPO" if public_backend == "jax-mjx" else public_algorithm),
        "hardware": hardware,
        "seed": seed,
        "date": result_date or datetime.now().strftime("%Y-%m-%d"),
        "stages": stages,
        "total_timesteps": total_timesteps,
        "total_training_time_seconds": round(total_duration, 1),
        "total_training_time": format_duration_hms(total_duration),
        "final_avg_reward": stages[final_stage_key]["final_eval_reward"],
        "provenance": _run_provenance(provenance),
    }
    if parallel_envs is not None:
        summary["parallel_envs"] = parallel_envs
    if run_id is not None:
        summary["run_id"] = run_id
    effective_plant_identity = plant_identity
    if effective_plant_identity is None:
        candidate = stage_results_list[-1].get("plant_identity")
        if isinstance(candidate, Mapping):
            effective_plant_identity = candidate
    if effective_plant_identity is not None:
        summary["plant_identity"] = dict(effective_plant_identity)
    return summary


def save_results_json(
    stage_results_list: list[dict[str, Any]],
    species: str,
    algorithm: str,
    seed: int,
    results_dir: "str | Path",
    hardware: str = "Google Colab",
    provenance: "Mapping[str, Any] | None" = None,
    *,
    backend: str | None = None,
    backend_version: str | None = None,
    parallel_envs: int | None = None,
    run_id: str | None = None,
    result_date: str | None = None,
    plant_identity: Mapping[str, Any] | None = None,
) -> Path:
    """Save a schema-v2 ``summary.json`` to *results_dir*.

    This compatibility wrapper can write partial summaries.  The canonical
    :func:`save_result_bundle` workflow only publishes ``summary.json`` after
    all three stages exist.
    """
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)
    summary = build_result_summary(
        stage_results_list,
        species,
        algorithm,
        seed,
        hardware=hardware,
        provenance=provenance,
        backend=backend,
        backend_version=backend_version,
        parallel_envs=parallel_envs,
        run_id=run_id,
        result_date=result_date,
        plant_identity=plant_identity,
    )
    summary_path = results_path / "summary.json"
    summary_path.write_text(_json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary_path
