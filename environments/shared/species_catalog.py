"""Build the public species catalog from executable and versioned sources.

The hand-authored species manifest contains presentation metadata and pointers
only. Interface dimensions come from the environments/MJCF models, layered plant
identity comes from the committed generated plant manifest, curriculum facts come
from the stage TOML files, and published metrics come from immutable result
summaries. The generated JSON is committed so the Docusaurus site can build
without requiring Python or MuJoCo in its Node.js build environment.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import tomllib
from datetime import date
from pathlib import Path
from typing import Any, cast

import mujoco

from environments.shared.config import load_all_stages
from environments.shared.curriculum import StageThreshold
from environments.shared.plant_contract import (
    GENERATED_MANIFEST_PATH,
    PHYSICS_SCHEMA,
    PLANT_MANIFEST_SCHEMA,
    POLICY_INTERFACE_SCHEMA,
    SOURCE_SCHEMA,
    VISUAL_SCHEMA,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST_PATH = REPOSITORY_ROOT / "configs" / "species_manifest.toml"
DEFAULT_PLANT_MANIFEST_PATH = GENERATED_MANIFEST_PATH
DEFAULT_OUTPUT_PATH = REPOSITORY_ROOT / "website" / "src" / "data" / "species.generated.json"
DEFAULT_README_PATH = REPOSITORY_ROOT / "README.md"

README_BLOCKS = {
    "species": ("<!-- BEGIN GENERATED: SPECIES -->", "<!-- END GENERATED: SPECIES -->"),
    "results": ("<!-- BEGIN GENERATED: RESULTS -->", "<!-- END GENERATED: RESULTS -->"),
    "notebooks": ("<!-- BEGIN GENERATED: NOTEBOOKS -->", "<!-- END GENERATED: NOTEBOOKS -->"),
}

ALLOWED_CAPABILITY_STATUSES = {"implemented", "experimental", "in_progress", "planned", "not_started"}
ALLOWED_MODEL_REVISION_STATUSES = {"current", "historical"}
ALLOWED_VERIFICATION_STATUSES = {"verified", "unverified"}
ALLOWED_TRAINING_BACKENDS = {"stable-baselines3", "jax-mjx"}
PROVENANCE_IDENTIFIERS = ("repository_commit", "model_hash", "config_hash")
DEFAULT_STAGE_THRESHOLD = StageThreshold()


class CatalogError(ValueError):
    """Raised when catalog inputs are incomplete or contradictory."""


def _repo_path(relative_path: str, *, field: str) -> Path:
    """Resolve and validate a repository-relative manifest path."""
    path = (REPOSITORY_ROOT / relative_path).resolve()
    try:
        path.relative_to(REPOSITORY_ROOT)
    except ValueError as exc:
        raise CatalogError(f"{field} must stay within the repository: {relative_path}") from exc
    if not path.exists():
        raise CatalogError(f"{field} does not exist: {relative_path}")
    return path


def _load_manifest(path: Path) -> dict[str, Any]:
    with path.open("rb") as manifest_file:
        manifest = tomllib.load(manifest_file)
    if manifest.get("schema_version") != 1:
        raise CatalogError("species manifest schema_version must be 1")
    return manifest


def _require_mapping(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CatalogError(f"{field} must be an object")
    return cast(dict[str, Any], value)


def _require_nonempty_string(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CatalogError(f"{field} must be a non-empty string")
    return value.strip()


def _optional_nonempty_string(value: Any, *, field: str) -> str | None:
    if value is None:
        return None
    return _require_nonempty_string(value, field=field)


def _require_positive_int(value: Any, *, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise CatalogError(f"{field} must be a positive integer")
    return value


def _require_sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.startswith("sha256:"):
        raise CatalogError(f"{field} must be sha256:<64 lowercase hex>")
    digest = value.removeprefix("sha256:")
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise CatalogError(f"{field} must be sha256:<64 lowercase hex>")
    return value


def _load_plant_manifest(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as manifest_file:
            manifest = json.load(manifest_file)
    except (OSError, json.JSONDecodeError) as exc:
        raise CatalogError(f"cannot read generated plant manifest {path}: {exc}") from exc
    manifest = _require_mapping(manifest, field="generated plant manifest")
    if manifest.get("schema") != PLANT_MANIFEST_SCHEMA:
        raise CatalogError(f"generated plant manifest must use schema {PLANT_MANIFEST_SCHEMA}")
    _require_positive_int(manifest.get("fingerprint_tool_version"), field="plant fingerprint_tool_version")
    generated_with = _require_mapping(manifest.get("generated_with"), field="plant generated_with")
    _require_nonempty_string(generated_with.get("mujoco"), field="plant generated_with.mujoco")
    _require_mapping(manifest.get("plants"), field="plant manifest plants")
    return manifest


def _optional_number(value: Any, *, field: str) -> int | float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise CatalogError(f"{field} must be null or a finite number")
    return value


def _load_environment(entrypoint: str) -> type[Any]:
    module_name, separator, class_name = entrypoint.partition(":")
    if not separator or not module_name or not class_name:
        raise CatalogError(f"invalid environment entrypoint: {entrypoint}")
    try:
        module = importlib.import_module(module_name)
        environment_class = getattr(module, class_name)
    except (ImportError, AttributeError) as exc:
        raise CatalogError(f"cannot import environment entrypoint: {entrypoint}") from exc
    return cast(type[Any], environment_class)


def _space_dimension(space: Any, *, label: str) -> int:
    shape = getattr(space, "shape", None)
    if not shape:
        raise CatalogError(f"{label} space must have a fixed shape")
    return int(math.prod(shape))


def _model_facts(model_path: Path) -> dict[str, int | float]:
    model = mujoco.MjModel.from_xml_path(str(model_path))
    dynamic_mass = sum(
        float(mass) for mass, mocap_id in zip(model.body_mass, model.body_mocapid, strict=True) if mocap_id == -1
    )
    return {
        "nq": int(model.nq),
        "nv": int(model.nv),
        "nu": int(model.nu),
        "dynamic_mass_kg": round(dynamic_mass, 6),
    }


def _environment_facts(entrypoint: str, expected_nu: int) -> dict[str, Any]:
    environment_class = _load_environment(entrypoint)
    environment = environment_class()
    try:
        observation_dim = _space_dimension(environment.observation_space, label="observation")
        action_dim = _space_dimension(environment.action_space, label="action")
    finally:
        environment.close()
    if action_dim != expected_nu:
        raise CatalogError(f"{entrypoint} action dimension {action_dim} does not match model.nu {expected_nu}")
    return {
        "entrypoint": entrypoint,
        "observation_dim": observation_dim,
        "action_dim": action_dim,
    }


def _public_plant_contract(
    species_id: str,
    raw_entry: Any,
    *,
    model_relative_path: str,
    model: dict[str, Any],
    environment: dict[str, Any],
) -> dict[str, Any]:
    """Validate one generated plant entry and select its public contract."""
    entry = _require_mapping(raw_entry, field=f"plant contract for {species_id}")
    if entry.get("species") != species_id:
        raise CatalogError(f"plant contract species mismatch for {species_id}: {entry.get('species')}")
    if entry.get("model_path") != model_relative_path:
        raise CatalogError(
            f"plant contract model_path mismatch for {species_id}: expected {model_relative_path}, "
            f"got {entry.get('model_path')}"
        )

    source = _require_mapping(entry.get("source"), field=f"plant source for {species_id}")
    if source.get("schema") != SOURCE_SCHEMA:
        raise CatalogError(f"plant source for {species_id} must use schema {SOURCE_SCHEMA}")
    if source.get("root") != model_relative_path:
        raise CatalogError(f"plant source root mismatch for {species_id}: {source.get('root')}")

    policy = _require_mapping(entry.get("policy_interface"), field=f"plant policy interface for {species_id}")
    if policy.get("schema") != POLICY_INTERFACE_SCHEMA:
        raise CatalogError(f"plant policy interface for {species_id} must use schema {POLICY_INTERFACE_SCHEMA}")
    policy_revision = _require_positive_int(policy.get("revision"), field=f"{species_id} policy revision")
    policy_observation_dim = _require_positive_int(
        policy.get("observation_dim"), field=f"{species_id} plant observation_dim"
    )
    policy_action_dim = _require_positive_int(policy.get("action_dim"), field=f"{species_id} plant action_dim")
    if policy_observation_dim != environment["observation_dim"]:
        raise CatalogError(
            f"plant observation_dim mismatch for {species_id}: contract={policy_observation_dim}, "
            f"environment={environment['observation_dim']}"
        )
    if policy_action_dim != environment["action_dim"]:
        raise CatalogError(
            f"plant action_dim mismatch for {species_id}: contract={policy_action_dim}, "
            f"environment={environment['action_dim']}"
        )

    physics = _require_mapping(entry.get("physics"), field=f"plant physics for {species_id}")
    if physics.get("schema") != PHYSICS_SCHEMA:
        raise CatalogError(f"plant physics for {species_id} must use schema {PHYSICS_SCHEMA}")
    physics_revision = _require_positive_int(physics.get("revision"), field=f"{species_id} physics revision")
    for dimension in ("nq", "nv", "nu"):
        contract_value = _require_positive_int(physics.get(dimension), field=f"{species_id} plant {dimension}")
        if contract_value != model[dimension]:
            raise CatalogError(
                f"plant {dimension} mismatch for {species_id}: contract={contract_value}, model={model[dimension]}"
            )

    visual = _require_mapping(entry.get("visual"), field=f"plant visual for {species_id}")
    if visual.get("schema") != VISUAL_SCHEMA:
        raise CatalogError(f"plant visual for {species_id} must use schema {VISUAL_SCHEMA}")
    visual_revision = _require_positive_int(visual.get("revision"), field=f"{species_id} visual revision")

    return {
        "bundle_sha256": _require_sha256(entry.get("bundle_sha256"), field=f"{species_id} plant bundle"),
        "source_closure_sha256": _require_sha256(source.get("closure_sha256"), field=f"{species_id} source closure"),
        "policy_interface": {
            "schema": POLICY_INTERFACE_SCHEMA,
            "revision": policy_revision,
            "observation_schema": _require_nonempty_string(
                policy.get("observation_schema"), field=f"{species_id} observation_schema"
            ),
            "sha256": _require_sha256(policy.get("sha256"), field=f"{species_id} policy interface"),
        },
        "physics": {
            "schema": PHYSICS_SCHEMA,
            "revision": physics_revision,
            "sha256": _require_sha256(physics.get("sha256"), field=f"{species_id} physics"),
        },
        "visual": {
            "schema": VISUAL_SCHEMA,
            "revision": visual_revision,
            "sha256": _require_sha256(visual.get("sha256"), field=f"{species_id} visual"),
        },
    }


def _stage_config_path(species_id: str, stage_number: int) -> Path:
    matches = sorted((REPOSITORY_ROOT / "configs" / species_id).glob(f"stage{stage_number}_*.toml"))
    if len(matches) != 1:
        raise CatalogError(
            f"expected one stage {stage_number} config for {species_id}, found {[path.name for path in matches]}"
        )
    return matches[0]


def _public_video_path(relative_path: str) -> str:
    prefix = "website/static"
    if not relative_path.startswith(f"{prefix}/"):
        raise CatalogError(f"stage video must be under {prefix}: {relative_path}")
    return relative_path[len(prefix) :]


def _build_stages(species_id: str, raw_videos: list[dict[str, Any]]) -> list[dict[str, Any]]:
    video_by_stage: dict[int, dict[str, Any]] = {}
    for raw_video in raw_videos:
        stage_number = int(raw_video["stage"])
        if stage_number not in {1, 2, 3}:
            raise CatalogError(f"stage video for {species_id} must use stage 1, 2, or 3: {stage_number}")
        if stage_number in video_by_stage:
            raise CatalogError(f"duplicate stage video for {species_id} stage {stage_number}")
        relative_path = str(raw_video["path"])
        video_path = _repo_path(relative_path, field=f"{species_id} stage {stage_number} video")
        poster_path = REPOSITORY_ROOT / "website" / "static" / "img" / "posters" / f"{video_path.stem}.jpg"
        if not poster_path.exists():
            raise CatalogError(f"poster does not exist for {relative_path}: {poster_path.relative_to(REPOSITORY_ROOT)}")
        algorithm = _require_nonempty_string(
            raw_video.get("algorithm"), field=f"{species_id} stage {stage_number} video algorithm"
        )
        backend = raw_video.get("backend")
        backend_version = _optional_nonempty_string(
            raw_video.get("backend_version"), field=f"{species_id} stage {stage_number} video backend_version"
        )
        model_revision_status = raw_video.get("model_revision_status")
        verification_status = raw_video.get("verification_status")
        if backend not in ALLOWED_TRAINING_BACKENDS:
            raise CatalogError(f"invalid stage-video backend for {species_id} stage {stage_number}: {backend}")
        if model_revision_status not in ALLOWED_MODEL_REVISION_STATUSES:
            raise CatalogError(
                f"invalid stage-video model_revision_status for {species_id} stage {stage_number}: "
                f"{model_revision_status}"
            )
        if verification_status not in ALLOWED_VERIFICATION_STATUSES:
            raise CatalogError(
                f"invalid stage-video verification_status for {species_id} stage {stage_number}: {verification_status}"
            )
        identifiers = {key: raw_video.get(key) for key in PROVENANCE_IDENTIFIERS}
        for key, value in identifiers.items():
            identifiers[key] = _optional_nonempty_string(value, field=f"{species_id} stage {stage_number} video {key}")
        if model_revision_status == "current" or verification_status == "verified":
            missing = [key for key, value in identifiers.items() if not value]
            if backend_version is None:
                missing.append("backend_version")
            if missing:
                raise CatalogError(
                    f"current or verified stage video for {species_id} stage {stage_number} "
                    f"is missing identifiers: {missing}"
                )
        video_by_stage[stage_number] = {
            "path": _public_video_path(relative_path),
            "algorithm": algorithm,
            "backend": str(backend),
            "backend_version": backend_version,
            "model_revision_status": str(model_revision_status),
            "verification_status": str(verification_status),
            **identifiers,
        }

    configs = load_all_stages(species_id)
    stages: list[dict[str, Any]] = []
    for stage_number in (1, 2, 3):
        stage_config = configs[stage_number]
        curriculum = stage_config["curriculum_kwargs"]
        config_path = _stage_config_path(species_id, stage_number)
        name = str(stage_config["name"])
        stages.append(
            {
                "number": stage_number,
                "name": name,
                "title": name.replace("_", " ").title(),
                "description": str(stage_config["description"]),
                "config_path": config_path.relative_to(REPOSITORY_ROOT).as_posix(),
                "timesteps": int(curriculum["timesteps"]),
                "advancement_gate": {
                    "min_avg_reward": curriculum.get("min_avg_reward"),
                    "min_avg_episode_length": curriculum.get("min_avg_episode_length"),
                    "min_avg_forward_velocity": curriculum.get("min_avg_forward_vel"),
                    "min_success_rate": curriculum.get("min_success_rate"),
                    "min_eval_episodes": int(
                        curriculum.get("min_eval_episodes", DEFAULT_STAGE_THRESHOLD.min_eval_episodes)
                    ),
                    "required_consecutive": int(
                        curriculum.get("required_consecutive", DEFAULT_STAGE_THRESHOLD.required_consecutive)
                    ),
                },
                "video": video_by_stage.get(stage_number),
            }
        )
    return stages


def _validate_provenance(provenance: Any, *, result_path: str) -> dict[str, Any]:
    provenance = _require_mapping(provenance, field=f"provenance in {result_path}")
    model_revision_status = provenance.get("model_revision_status")
    verification_status = provenance.get("verification_status")
    if model_revision_status not in ALLOWED_MODEL_REVISION_STATUSES:
        raise CatalogError(f"invalid model_revision_status in {result_path}: {model_revision_status}")
    if verification_status not in ALLOWED_VERIFICATION_STATUSES:
        raise CatalogError(f"invalid verification_status in {result_path}: {verification_status}")
    evaluation_episodes = provenance.get("evaluation_episodes")
    if evaluation_episodes is not None and (
        not isinstance(evaluation_episodes, int) or isinstance(evaluation_episodes, bool) or evaluation_episodes <= 0
    ):
        raise CatalogError(f"evaluation_episodes must be null or a positive integer in {result_path}")

    identifiers = {
        key: _optional_nonempty_string(provenance.get(key), field=f"provenance.{key} in {result_path}")
        for key in PROVENANCE_IDENTIFIERS
    }
    if model_revision_status == "current" or verification_status == "verified":
        missing = [key for key, value in identifiers.items() if not value]
        if evaluation_episodes is None:
            missing.append("evaluation_episodes")
        if missing:
            raise CatalogError(f"current or verified result {result_path} is missing identifiers: {missing}")
    return {
        "model_revision_status": model_revision_status,
        "verification_status": verification_status,
        "evaluation_episodes": evaluation_episodes,
        **identifiers,
    }


def _validate_result_summary(summary: Any, *, species_id: str, relative_path: str) -> dict[str, Any]:
    """Validate the public fields consumed by the catalog and website."""
    summary = _require_mapping(summary, field=f"result summary {relative_path}")
    if summary.get("schema_version") != 2:
        raise CatalogError(f"result summary schema_version must be 2: {relative_path}")
    if summary.get("species") != species_id:
        raise CatalogError(f"result species mismatch in {relative_path}: {summary.get('species')}")

    algorithm = _require_nonempty_string(summary.get("algorithm"), field=f"algorithm in {relative_path}")
    expected_algorithm = Path(relative_path).parent.name
    if algorithm.lower() != expected_algorithm:
        raise CatalogError(
            f"result algorithm mismatch in {relative_path}: expected {expected_algorithm}, got {algorithm}"
        )
    backend = summary.get("backend")
    if backend not in ALLOWED_TRAINING_BACKENDS:
        raise CatalogError(f"invalid backend in {relative_path}: {backend}")
    backend_version = _optional_nonempty_string(
        summary.get("backend_version"), field=f"backend_version in {relative_path}"
    )

    result_date = _require_nonempty_string(summary.get("date"), field=f"date in {relative_path}")
    try:
        date.fromisoformat(result_date)
    except ValueError as exc:
        raise CatalogError(f"date in {relative_path} must use ISO YYYY-MM-DD format") from exc

    _optional_nonempty_string(summary.get("hardware"), field=f"hardware in {relative_path}")
    seed = summary.get("seed")
    if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool) or seed < 0):
        raise CatalogError(f"seed in {relative_path} must be null or a non-negative integer")
    parallel_envs = summary.get("parallel_envs")
    if parallel_envs is not None:
        _require_positive_int(parallel_envs, field=f"parallel_envs in {relative_path}")

    raw_stages = _require_mapping(summary.get("stages"), field=f"stages in {relative_path}")
    if set(raw_stages) != {"1", "2", "3"}:
        raise CatalogError(f"stages in {relative_path} must contain exactly 1, 2, and 3")
    stage_timesteps = 0
    for stage_key in ("1", "2", "3"):
        raw_stage = _require_mapping(raw_stages[stage_key], field=f"stage {stage_key} in {relative_path}")
        prefix = f"stage {stage_key} in {relative_path}"
        _require_nonempty_string(raw_stage.get("name"), field=f"name for {prefix}")
        _require_nonempty_string(raw_stage.get("description"), field=f"description for {prefix}")
        stage_timesteps += _require_positive_int(raw_stage.get("timesteps"), field=f"timesteps for {prefix}")
        for metric_key in (
            "best_eval_reward",
            "final_eval_reward",
            "avg_forward_vel",
            "avg_episode_length",
        ):
            if metric_key not in raw_stage:
                raise CatalogError(f"{metric_key} is required for {prefix}")
            _optional_number(raw_stage[metric_key], field=f"{metric_key} for {prefix}")
        success_rate = _optional_number(raw_stage.get("mean_success_rate"), field=f"mean_success_rate for {prefix}")
        if success_rate is not None and not 0.0 <= success_rate <= 1.0:
            raise CatalogError(f"mean_success_rate for {prefix} must be between 0 and 1")
        _optional_nonempty_string(raw_stage.get("training_time"), field=f"training_time for {prefix}")
        if not isinstance(raw_stage.get("stage_passed"), bool):
            raise CatalogError(f"stage_passed for {prefix} must be a boolean")

    total_timesteps = _require_positive_int(summary.get("total_timesteps"), field=f"total_timesteps in {relative_path}")
    if total_timesteps != stage_timesteps:
        raise CatalogError(
            f"total_timesteps in {relative_path} is {total_timesteps}, but stage totals sum to {stage_timesteps}"
        )
    _optional_nonempty_string(summary.get("total_training_time"), field=f"total_training_time in {relative_path}")
    _optional_number(summary.get("final_avg_reward"), field=f"final_avg_reward in {relative_path}")

    provenance = _validate_provenance(summary.get("provenance"), result_path=relative_path)
    if provenance["model_revision_status"] == "current" or provenance["verification_status"] == "verified":
        if backend_version is None:
            raise CatalogError(f"current or verified result {relative_path} is missing backend_version")
    return cast(dict[str, Any], summary)


def _max_reported_velocity(stage_summaries: list[dict[str, Any]]) -> int | float | None:
    reported_velocities = [
        stage["avg_forward_vel"] for stage in stage_summaries if stage["avg_forward_vel"] is not None
    ]
    return max(reported_velocities) if reported_velocities else None


def _build_result(species_id: str, relative_path: str) -> dict[str, Any]:
    result_path = _repo_path(relative_path, field=f"{species_id} result summary")
    with result_path.open(encoding="utf-8") as result_file:
        summary = _validate_result_summary(json.load(result_file), species_id=species_id, relative_path=relative_path)

    stage_summaries: list[dict[str, Any]] = []
    for stage_key in sorted(summary["stages"], key=int):
        raw_stage = summary["stages"][stage_key]
        stage_summaries.append(
            {
                "number": int(stage_key),
                "name": raw_stage.get("name", f"stage{stage_key}"),
                "description": raw_stage.get("description", ""),
                "timesteps": raw_stage.get("timesteps"),
                "best_eval_reward": raw_stage.get("best_eval_reward"),
                "final_eval_reward": raw_stage.get("final_eval_reward"),
                "avg_forward_vel": raw_stage.get("avg_forward_vel"),
                "avg_episode_length": raw_stage.get("avg_episode_length"),
                "mean_success_rate": raw_stage.get("mean_success_rate"),
                "training_time": raw_stage.get("training_time"),
                "stage_passed": raw_stage.get("stage_passed"),
            }
        )

    max_average_forward_velocity = _max_reported_velocity(stage_summaries)
    stage_three = next((stage for stage in stage_summaries if stage["number"] == 3), None)
    return {
        "summary_path": relative_path,
        "algorithm": summary["algorithm"],
        "backend": summary["backend"],
        "backend_version": summary.get("backend_version"),
        "date": summary["date"],
        "hardware": summary.get("hardware"),
        "seed": summary.get("seed"),
        "parallel_envs": summary.get("parallel_envs"),
        "total_timesteps": summary.get(
            "total_timesteps", sum(stage.get("timesteps") or 0 for stage in stage_summaries)
        ),
        "total_training_time": summary.get("total_training_time"),
        "final_avg_reward": summary.get("final_avg_reward"),
        "max_average_forward_velocity": max_average_forward_velocity,
        "stage3_success_rate": stage_three.get("mean_success_rate") if stage_three else None,
        "provenance": _validate_provenance(summary["provenance"], result_path=relative_path),
        "stages": stage_summaries,
    }


def _build_success_metrics(species_id: str, raw_metrics: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_metrics, list) or not raw_metrics:
        raise CatalogError(f"{species_id} must define at least one backend-scoped success metric")

    covered_backends: set[str] = set()
    metrics: list[dict[str, Any]] = []
    for index, raw_metric_value in enumerate(raw_metrics, start=1):
        raw_metric = _require_mapping(raw_metric_value, field=f"{species_id} success metric {index}")
        backends = raw_metric.get("backends")
        if not isinstance(backends, list) or not backends:
            raise CatalogError(f"{species_id} success metric {index} must name at least one backend")
        if any(backend not in ALLOWED_TRAINING_BACKENDS for backend in backends):
            raise CatalogError(f"{species_id} success metric {index} has invalid backends: {backends}")
        if len(set(backends)) != len(backends):
            raise CatalogError(f"{species_id} success metric {index} repeats a backend")
        duplicate_backends = covered_backends.intersection(backends)
        if duplicate_backends:
            raise CatalogError(f"{species_id} defines more than one success metric for {sorted(duplicate_backends)}")
        covered_backends.update(backends)
        metrics.append(
            {
                "backends": list(backends),
                "key": _require_nonempty_string(
                    raw_metric.get("key"), field=f"key for {species_id} success metric {index}"
                ),
                "label": _require_nonempty_string(
                    raw_metric.get("label"), field=f"label for {species_id} success metric {index}"
                ),
                "definition": _require_nonempty_string(
                    raw_metric.get("definition"), field=f"definition for {species_id} success metric {index}"
                ),
            }
        )

    if covered_backends != ALLOWED_TRAINING_BACKENDS:
        missing = sorted(ALLOWED_TRAINING_BACKENDS - covered_backends)
        raise CatalogError(f"{species_id} has no success metric for backends: {missing}")
    return metrics


def build_catalog(
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    plant_manifest_path: Path = DEFAULT_PLANT_MANIFEST_PATH,
) -> dict[str, Any]:
    """Build and validate the deterministic public species catalog."""
    manifest_path = manifest_path.resolve()
    manifest = _load_manifest(manifest_path)
    plant_manifest_path = plant_manifest_path.resolve()
    plant_manifest = _load_plant_manifest(plant_manifest_path)
    raw_plant_entries = _require_mapping(plant_manifest["plants"], field="plant manifest plants")

    raw_capabilities = manifest.get("project_capabilities", {})
    capabilities: dict[str, dict[str, str]] = {}
    for capability_id, capability in raw_capabilities.items():
        status = capability.get("status")
        if status not in ALLOWED_CAPABILITY_STATUSES:
            raise CatalogError(f"invalid capability status for {capability_id}: {status}")
        capabilities[capability_id] = {"label": str(capability["label"]), "status": str(status)}

    raw_notebooks = sorted(manifest.get("notebooks", []), key=lambda item: int(item["display_order"]))
    notebook_ids: set[str] = set()
    notebooks: list[dict[str, Any]] = []
    for raw_notebook in raw_notebooks:
        notebook_id = str(raw_notebook["id"])
        if notebook_id in notebook_ids:
            raise CatalogError(f"duplicate notebook id: {notebook_id}")
        notebook_ids.add(notebook_id)
        notebook_path = str(raw_notebook["path"])
        _repo_path(notebook_path, field=f"notebook {notebook_id}")
        notebooks.append(
            {
                "id": notebook_id,
                "label": str(raw_notebook["label"]),
                "description": str(raw_notebook["description"]),
                "path": notebook_path,
            }
        )

    species_entries = sorted(manifest.get("species", []), key=lambda item: int(item["display_order"]))
    species_ids: set[str] = set()
    aliases: set[str] = set()
    manifested_result_paths: set[str] = set()
    species_catalog: list[dict[str, Any]] = []
    for raw_species in species_entries:
        species_id = str(raw_species["id"])
        if species_id in species_ids:
            raise CatalogError(f"duplicate species id: {species_id}")
        species_ids.add(species_id)

        species_aliases = [str(alias) for alias in raw_species.get("aliases", [])]
        duplicate_aliases = aliases.intersection(species_aliases)
        if duplicate_aliases:
            raise CatalogError(f"duplicate species aliases: {sorted(duplicate_aliases)}")
        aliases.update(species_aliases)

        model_relative_path = str(raw_species["model_path"])
        model_path = _repo_path(model_relative_path, field=f"{species_id} model")
        model: dict[str, Any] = {"path": model_relative_path, **_model_facts(model_path)}
        environment = _environment_facts(str(raw_species["env_entrypoint"]), int(model["nu"]))
        model["plant_contract"] = _public_plant_contract(
            species_id,
            raw_plant_entries.get(species_id),
            model_relative_path=model_relative_path,
            model=model,
            environment=environment,
        )

        training_notebook_ids = [str(notebook_id) for notebook_id in raw_species.get("training_notebooks", [])]
        unknown_notebooks = sorted(set(training_notebook_ids) - notebook_ids)
        if unknown_notebooks:
            raise CatalogError(f"{species_id} references unknown notebooks: {unknown_notebooks}")

        result_summary_paths = [str(path) for path in raw_species.get("result_summary_paths", [])]
        duplicate_result_paths = manifested_result_paths.intersection(result_summary_paths)
        if duplicate_result_paths:
            raise CatalogError(f"result summaries are listed more than once: {sorted(duplicate_result_paths)}")
        manifested_result_paths.update(result_summary_paths)

        species_catalog.append(
            {
                "id": species_id,
                "display_name": str(raw_species["display_name"]),
                "tagline": str(raw_species["tagline"]),
                "gait": str(raw_species["gait"]),
                "specialty": str(raw_species["specialty"]),
                "aliases": species_aliases,
                "environment": environment,
                "model": model,
                "training_notebooks": training_notebook_ids,
                "success_metrics": _build_success_metrics(species_id, raw_species.get("success_metrics")),
                "stages": _build_stages(species_id, raw_species.get("stage_videos", [])),
                "historical_results": [_build_result(species_id, result_path) for result_path in result_summary_paths],
            }
        )

    discovered_result_paths = {
        path.relative_to(REPOSITORY_ROOT).as_posix() for path in (REPOSITORY_ROOT / "results").glob("*/*/summary.json")
    }
    if manifested_result_paths != discovered_result_paths:
        missing = sorted(discovered_result_paths - manifested_result_paths)
        unknown = sorted(manifested_result_paths - discovered_result_paths)
        raise CatalogError(f"manifest/result summary coverage mismatch; unlisted={missing}, unknown={unknown}")

    implemented_species = {
        path.name
        for path in (REPOSITORY_ROOT / "environments").iterdir()
        if path.is_dir() and (path / "envs").is_dir() and (path / "assets").is_dir()
    }
    if species_ids != implemented_species:
        missing = sorted(implemented_species - species_ids)
        unknown = sorted(species_ids - implemented_species)
        raise CatalogError(f"manifest/implemented species coverage mismatch; unlisted={missing}, unknown={unknown}")

    plant_species = set(raw_plant_entries)
    if species_ids != plant_species:
        missing = sorted(species_ids - plant_species)
        unknown = sorted(plant_species - species_ids)
        raise CatalogError(f"plant/species manifest coverage mismatch; missing={missing}, unknown={unknown}")

    return {
        "schema_version": 2,
        "manifest_path": manifest_path.relative_to(REPOSITORY_ROOT).as_posix(),
        "plant_manifest": {
            "path": plant_manifest_path.relative_to(REPOSITORY_ROOT).as_posix(),
            "schema": PLANT_MANIFEST_SCHEMA,
            "fingerprint_tool_version": int(plant_manifest["fingerprint_tool_version"]),
            "generated_with": dict(plant_manifest["generated_with"]),
        },
        "project_capabilities": capabilities,
        "notebooks": notebooks,
        "species": species_catalog,
    }


def render_catalog_json(catalog: dict[str, Any]) -> str:
    """Serialize a catalog deterministically for source control."""
    return json.dumps(catalog, indent=2, ensure_ascii=False) + "\n"


def _format_millions(value: int | float | None) -> str:
    if value is None:
        return "—"
    millions = value / 1_000_000
    return f"{millions:g}M"


def _format_number(value: int | float | None, *, suffix: str = "") -> str:
    if value is None:
        return "—"
    return f"{value:.2f}{suffix}"


def _format_percent(value: int | float | None) -> str:
    if value is None:
        return "—"
    return f"{value * 100:.1f}%"


def _format_passed(value: bool | None) -> str:
    if value is None:
        return "—"
    return "Yes" if value else "No"


def _format_evaluation_episodes(value: int | None) -> str:
    if value is None:
        return "evaluation episode count not recorded"
    return f"{value} evaluation episodes"


def _format_backend(backend: str, version: str | None = None) -> str:
    label = {"stable-baselines3": "Stable-Baselines3", "jax-mjx": "JAX/MJX"}.get(backend, backend)
    return f"{label} {version}" if version else f"{label} (version not recorded)"


def _success_metric_for_backend(species: dict[str, Any], backend: str) -> dict[str, Any]:
    for metric in species["success_metrics"]:
        if backend in metric["backends"]:
            return cast(dict[str, Any], metric)
    raise CatalogError(f"{species['id']} has no success metric for backend {backend}")


def _format_advancement_gate(gate: dict[str, Any]) -> str:
    criteria: list[str] = []
    if gate["min_avg_reward"] is not None:
        criteria.append(f"reward ≥ {gate['min_avg_reward']:g}")
    if gate["min_avg_episode_length"] is not None:
        criteria.append(f"episode length ≥ {gate['min_avg_episode_length']:g}")
    if gate["min_avg_forward_velocity"] is not None:
        criteria.append(f"avg. velocity ≥ {gate['min_avg_forward_velocity']:g} m/s")
    if gate["min_success_rate"] is not None:
        criteria.append(f"task success ≥ {_format_percent(gate['min_success_rate'])}")
    criteria.append(f"≥ {gate['min_eval_episodes']} episodes/evaluation")
    criteria.append(f"{gate['required_consecutive']} consecutive passes")
    return "; ".join(criteria)


def render_readme_species(catalog: dict[str, Any]) -> str:
    """Render active species facts and current stages for the root README."""
    lines = [
        "The active-species tables are generated from `configs/species_manifest.toml`, the layered plant contract,",
        "the executable Gymnasium environments, compiled MJCF models, and current stage TOML files. The budgets and",
        "gates shown are for the Stable-Baselines3 curriculum path. Do not edit the generated block by hand.",
    ]
    for species in catalog["species"]:
        environment = species["environment"]
        model = species["model"]
        plant = model["plant_contract"]
        lines.extend(
            [
                "",
                f"### {species['display_name']}",
                "",
                f"{species['tagline']}. **Specialty:** {species['specialty']}.",
                "",
                "| Generated specification | Value |",
                "|---|---|",
                f"| Observation dimension | {environment['observation_dim']} |",
                f"| Action dimension / actuators | {environment['action_dim']} |",
                f"| Generalized coordinates / velocities | nq={model['nq']}, nv={model['nv']} |",
                f"| Compiled dynamic model mass | {model['dynamic_mass_kg']:.1f} kg |",
                "| Plant contract revisions | "
                f"policy r{plant['policy_interface']['revision']}; "
                f"physics r{plant['physics']['revision']}; visual r{plant['visual']['revision']} "
                "([details](docs/PLANT_CONTRACT.md)) |",
                f"| Model | `{model['path']}` |",
                "",
                "| Current stage | Objective | SB3 configured budget | SB3 early-advancement gate |",
                "|---|---|---:|---:|",
            ]
        )
        for stage in species["stages"]:
            lines.append(
                f"| {stage['number']} — {stage['title']} | {stage['description']} | "
                f"{_format_millions(stage['timesteps'])} | "
                f"{_format_advancement_gate(stage['advancement_gate'])} |"
            )
        lines.extend(["", "**Backend-specific success semantics:**"])
        for metric in species["success_metrics"]:
            scopes = " / ".join(
                _format_backend(backend).replace(" (version not recorded)", "") for backend in metric["backends"]
            )
            lines.append(f"- **{scopes} — {metric['label']}:** {metric['definition']}")
        lines.extend(["", f"[Full documentation →](environments/{species['id']}/README.md)"])
        if species["id"] == "velociraptor":
            lines.extend(
                [
                    "",
                    "[Hugging Face models →](https://huggingface.co/kuds/mesozoic-labs-velocipastor)",
                ]
            )
    return "\n".join(lines)


def render_readme_results(catalog: dict[str, Any]) -> str:
    """Render provenance-labelled historical experiment summaries."""
    lines = [
        "The summaries below are historical experiment records generated from the versioned JSON files under",
        "`results/`. They are not evidence for the current model revision unless provenance is marked both current",
        "and verified. Current stage budgets may therefore differ from the steps reported here.",
    ]
    for species in catalog["species"]:
        for result in species["historical_results"]:
            provenance = result["provenance"]
            result_metric = _success_metric_for_backend(species, result["backend"])
            lines.extend(
                [
                    "",
                    f"### {species['display_name']} ({result['algorithm'].upper()} · "
                    f"{_format_backend(result['backend']).replace(' (version not recorded)', '')}) — {result['date']}",
                    "",
                    f"**Provenance:** {provenance['model_revision_status'].title()} model; "
                    f"{provenance['verification_status']}; "
                    f"{_format_evaluation_episodes(provenance['evaluation_episodes'])}; "
                    f"{_format_backend(result['backend'], result['backend_version'])}. "
                    f"**Run total:** {_format_millions(result['total_timesteps'])} steps; "
                    f"{result['total_training_time'] or 'time not recorded'}. "
                    f"[Source summary]({result['summary_path']}).",
                    "",
                    "| Stage | Best eval reward | Avg. forward velocity | Task success | Trained steps | Passed |",
                    "|---|---:|---:|---:|---:|---:|",
                ]
            )
            for stage in result["stages"]:
                lines.append(
                    f"| {stage['number']} — {str(stage['name']).replace('_', ' ').title()} | "
                    f"{_format_number(stage['best_eval_reward'])} | "
                    f"{_format_number(stage['avg_forward_vel'], suffix=' m/s')} | "
                    f"{_format_percent(stage['mean_success_rate'])} | "
                    f"{_format_millions(stage['timesteps'])} | {_format_passed(stage['stage_passed'])} |"
                )
            lines.extend(
                [
                    "",
                    f"**Current {_format_backend(result['backend']).replace(' (version not recorded)', '')} "
                    "catalog definition for this task label:** "
                    f"{result_metric['definition']}",
                ]
            )
    return "\n".join(lines)


def render_readme_notebooks(catalog: dict[str, Any]) -> str:
    """Render the public notebook inventory."""
    lines = ["| Notebook | Description |", "|---|---|"]
    for notebook in catalog["notebooks"]:
        lines.append(f"| [`{notebook['path']}`]({notebook['path']}) | {notebook['description']} |")
    return "\n".join(lines)


def _replace_generated_block(source: str, block_name: str, content: str) -> str:
    begin, end = README_BLOCKS[block_name]
    if source.count(begin) != 1 or source.count(end) != 1:
        raise CatalogError(f"README must contain exactly one {block_name} generated block")
    before, remainder = source.split(begin, 1)
    _, after = remainder.split(end, 1)
    return f"{before}{begin}\n{content.rstrip()}\n{end}{after}"


def render_readme(source: str, catalog: dict[str, Any]) -> str:
    """Replace all generator-managed README sections deterministically."""
    rendered = source
    rendered = _replace_generated_block(rendered, "species", render_readme_species(catalog))
    rendered = _replace_generated_block(rendered, "results", render_readme_results(catalog))
    rendered = _replace_generated_block(rendered, "notebooks", render_readme_notebooks(catalog))
    return rendered


def write_catalog(
    output_path: Path = DEFAULT_OUTPUT_PATH,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    readme_path: Path = DEFAULT_README_PATH,
    plant_manifest_path: Path = DEFAULT_PLANT_MANIFEST_PATH,
) -> None:
    """Build and write the generated catalog JSON and README blocks."""
    catalog = build_catalog(manifest_path, plant_manifest_path)
    readme_source = readme_path.read_text(encoding="utf-8")
    rendered_readme = render_readme(readme_source, catalog)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_catalog_json(catalog), encoding="utf-8")
    readme_path.write_text(rendered_readme, encoding="utf-8")


def check_catalog(
    output_path: Path = DEFAULT_OUTPUT_PATH,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    readme_path: Path = DEFAULT_README_PATH,
    plant_manifest_path: Path = DEFAULT_PLANT_MANIFEST_PATH,
) -> None:
    """Raise when committed catalog outputs are missing or stale."""
    catalog = build_catalog(manifest_path, plant_manifest_path)
    expected = render_catalog_json(catalog)
    if not output_path.exists():
        raise CatalogError(f"generated catalog is missing: {output_path.relative_to(REPOSITORY_ROOT)}")
    actual = output_path.read_text(encoding="utf-8")
    if actual != expected:
        raise CatalogError(
            "generated species catalog is stale; run "
            "`python -m environments.shared.species_catalog` and commit the result"
        )
    readme_source = readme_path.read_text(encoding="utf-8")
    if render_readme(readme_source, catalog) != readme_source:
        raise CatalogError(
            "generated README species data is stale; run "
            "`python -m environments.shared.species_catalog` and commit the result"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--plant-manifest", type=Path, default=DEFAULT_PLANT_MANIFEST_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--readme", type=Path, default=DEFAULT_README_PATH)
    parser.add_argument("--check", action="store_true", help="Fail instead of writing when generated data is stale")
    args = parser.parse_args(argv)

    try:
        if args.check:
            check_catalog(args.output, args.manifest, args.readme, args.plant_manifest)
            print(f"Species catalog is current: {args.output} and {args.readme}")
        else:
            write_catalog(args.output, args.manifest, args.readme, args.plant_manifest)
            print(f"Wrote species catalog: {args.output} and {args.readme}")
    except CatalogError as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
