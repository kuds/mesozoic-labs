"""Cross-check the published summary against the per-episode evidence.

The summary's aggregates have to be reproducible from the recorded evaluation
episodes and agree with the CSV; anything else means the numbers were edited
or the artifacts came from different runs."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..curriculum.stance_gate import STANCE_GATE_KIND
from .errors import ResultBundleError


def _optional_csv_number(value: str | None) -> float | None:
    if value is None or not value.strip():
        return None
    try:
        numeric = float(value)
    except ValueError as exc:
        raise ResultBundleError(f"invalid numeric CSV value: {value!r}") from exc
    if not math.isfinite(numeric):
        raise ResultBundleError(f"non-finite numeric CSV value: {value!r}")
    return numeric


def _optional_csv_bool(value: str | None) -> bool | None:
    if value is None or not value.strip():
        return None
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ResultBundleError(f"invalid boolean CSV value: {value!r}")


def compare_summary_to_csv(
    summary: Mapping[str, Any],
    csv_path: str | Path,
    *,
    tolerance: float = 1e-6,
) -> list[str]:
    """Return contradictions between canonical summary and derived CSV."""
    path = Path(csv_path)
    with path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    rows_by_stage: dict[str, dict[str, str]] = {}
    problems: list[str] = []
    for row in rows:
        stage = (row.get("stage") or "").strip()
        if stage in rows_by_stage:
            problems.append(f"CSV contains duplicate stage {stage}")
        rows_by_stage[stage] = row

    summary_stages = summary.get("stages")
    if not isinstance(summary_stages, Mapping):
        return ["summary stages must be an object"]
    if set(rows_by_stage) != set(summary_stages):
        problems.append(f"stage sets differ: summary={sorted(summary_stages)} csv={sorted(rows_by_stage)}")

    numeric_fields = {
        "timesteps": "timesteps",
        "best_eval_reward": "best_mean_reward",
        "best_eval_std": "best_mean_reward_std",
        "best_eval_step": "best_eval_step",
        "final_eval_reward": "last_mean_reward",
        "final_eval_std": "last_mean_reward_std",
        "avg_episode_length": "last_mean_episode_length",
        "avg_episode_length_std": "last_mean_episode_length_std",
        "avg_forward_vel": "mean_forward_vel",
        "avg_forward_vel_std": "std_forward_vel",
        "mean_distance_traveled": "mean_distance_traveled",
        "mean_success_rate": "mean_success_rate",
        "training_time_seconds": "training_duration_seconds",
        "selected_model_reward": "selected_model_mean_reward",
        "selected_model_reward_std": "selected_model_reward_std",
        "selected_model_episode_length": "selected_model_mean_episode_length",
        "selected_model_episode_length_std": "selected_model_episode_length_std",
        "selected_model_forward_vel": "selected_model_mean_forward_vel",
        "selected_model_forward_vel_std": "selected_model_mean_forward_vel_std",
        "selected_model_distance": "selected_model_mean_distance",
        "selected_model_success_rate": "selected_model_success_rate",
    }
    for stage, stage_summary_value in summary_stages.items():
        if stage not in rows_by_stage or not isinstance(stage_summary_value, Mapping):
            continue
        row = rows_by_stage[stage]
        for summary_key, csv_key in numeric_fields.items():
            expected = stage_summary_value.get(summary_key)
            actual = _optional_csv_number(row.get(csv_key))
            if expected is None and actual is None:
                continue
            if expected is None or actual is None or abs(float(expected) - actual) > tolerance:
                problems.append(f"stage {stage} {summary_key}/{csv_key} differs: summary={expected!r} csv={actual!r}")
        expected_passed = stage_summary_value.get("stage_passed")
        actual_passed = _optional_csv_bool(row.get("stage_passed"))
        if expected_passed != actual_passed:
            problems.append(f"stage {stage} stage_passed differs: summary={expected_passed!r} csv={actual_passed!r}")
        expected_publication_gate = stage_summary_value.get("publication_gate_passed")
        actual_publication_gate = _optional_csv_bool(row.get("publication_gate_passed"))
        if expected_publication_gate != actual_publication_gate:
            problems.append(
                f"stage {stage} publication_gate_passed differs: "
                f"summary={expected_publication_gate!r} csv={actual_publication_gate!r}"
            )

        provenance = summary.get("provenance")
        provenance = provenance if isinstance(provenance, Mapping) else {}
        identities: dict[str, Any] = {
            "schema_version": summary.get("schema_version"),
            "species": summary.get("species"),
            "algorithm": summary.get("algorithm"),
            "backend": summary.get("backend"),
            "seed": summary.get("seed"),
            "run_id": summary.get("run_id"),
            "repository_commit": provenance.get("repository_commit"),
            "config_hash": provenance.get("config_hash"),
            "model_hash": provenance.get("model_hash"),
        }
        plant_identity = summary.get("plant_identity")
        if isinstance(plant_identity, Mapping):
            identities.update({f"plant_{key}": value for key, value in plant_identity.items()})
        for key, expected in identities.items():
            if expected is None:
                continue
            if key not in row:
                problems.append(f"stage {stage} CSV is missing required column {key}")
                continue
            actual_value: Any = row[key]
            if key in {"schema_version", "seed"} and actual_value != "":
                try:
                    actual_value = int(actual_value)
                except ValueError as exc:
                    raise ResultBundleError(f"invalid integer CSV value for {key}: {actual_value!r}") from exc
            if str(actual_value).lower() != str(expected).lower():
                problems.append(f"stage {stage} {key} differs: summary={expected!r} csv={actual_value!r}")
    return problems


def _evaluation_evidence_aggregates(
    evidence_path: Path,
    *,
    checkpoint_label: str,
    expected_episodes: int,
    evaluation_seeds: Sequence[int],
    stage: int,
) -> dict[str, float]:
    """Parse one fixed-seed episode file and return publication aggregates."""
    required_columns = {
        "episode",
        "evaluation_seed",
        "checkpoint",
        "reward",
        "length",
        "mean_forward_velocity",
        "distance_traveled",
        "success",
    }
    if not evidence_path.is_file():
        raise ResultBundleError(f"missing {checkpoint_label} evaluation evidence: {evidence_path}")
    with evidence_path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        fieldnames = set(reader.fieldnames or [])
        rows = list(reader)
    missing_columns = sorted(required_columns - fieldnames)
    if missing_columns:
        raise ResultBundleError(
            f"{checkpoint_label} evaluation evidence for stage {stage} is missing columns: {missing_columns}"
        )
    if len(rows) != expected_episodes:
        raise ResultBundleError(
            f"{checkpoint_label} evaluation evidence for stage {stage} has "
            f"{len(rows)} rows; expected {expected_episodes}"
        )

    rewards: list[float] = []
    lengths: list[float] = []
    forward_velocities: list[float] = []
    distances: list[float] = []
    successes: list[float] = []
    recorded_seed: int | None = None
    for expected_episode, row in enumerate(rows, start=1):
        try:
            episode = int(row.get("episode", ""))
            seed = int(row.get("evaluation_seed", ""))
            length = int(row.get("length", ""))
        except (TypeError, ValueError) as exc:
            raise ResultBundleError(
                f"invalid integer in {checkpoint_label} evaluation evidence for stage {stage}"
            ) from exc
        if episode != expected_episode:
            raise ResultBundleError(
                f"{checkpoint_label} evaluation evidence for stage {stage} must number episodes consecutively"
            )
        if seed not in evaluation_seeds:
            raise ResultBundleError(
                f"{checkpoint_label} evaluation seed {seed} for stage {stage} is not recorded in provenance"
            )
        if recorded_seed is None:
            recorded_seed = seed
        elif seed != recorded_seed:
            raise ResultBundleError(f"{checkpoint_label} evaluation evidence for stage {stage} mixes evaluation seeds")
        if row.get("checkpoint") != checkpoint_label:
            raise ResultBundleError(
                f"{checkpoint_label} evaluation evidence for stage {stage} has the wrong checkpoint label"
            )
        if length <= 0:
            raise ResultBundleError(f"{checkpoint_label} evaluation episode length for stage {stage} must be positive")
        reward = _optional_csv_number(row.get("reward"))
        forward_velocity = _optional_csv_number(row.get("mean_forward_velocity"))
        distance = _optional_csv_number(row.get("distance_traveled"))
        success = _optional_csv_bool(row.get("success"))
        if reward is None or forward_velocity is None or distance is None or success is None:
            raise ResultBundleError(f"{checkpoint_label} evaluation evidence for stage {stage} contains blank values")
        rewards.append(reward)
        lengths.append(float(length))
        forward_velocities.append(forward_velocity)
        distances.append(distance)
        successes.append(float(success))

    def _mean(values: Sequence[float]) -> float:
        return math.fsum(values) / len(values)

    def _population_std(values: Sequence[float]) -> float:
        mean_value = _mean(values)
        return math.sqrt(math.fsum((value - mean_value) ** 2 for value in values) / len(values))

    assert recorded_seed is not None
    return {
        "evaluation_seed": float(recorded_seed),
        "reward": _mean(rewards),
        "reward_std": _population_std(rewards),
        "episode_length": _mean(lengths),
        "episode_length_std": _population_std(lengths),
        "forward_vel": _mean(forward_velocities),
        "forward_vel_std": _population_std(forward_velocities),
        "distance": _mean(distances),
        "success_rate": _mean(successes),
    }


def _compare_evaluation_aggregates(
    stage_summary: Mapping[str, Any],
    aggregates: Mapping[str, float],
    *,
    stage: int,
    label: str,
    metric_map: Mapping[str, tuple[str, float]],
) -> None:
    for summary_metric, (aggregate_metric, tolerance) in metric_map.items():
        expected = stage_summary.get(summary_metric)
        if expected is None:
            raise ResultBundleError(
                f"summary is missing {label} evaluation aggregate {summary_metric} for stage {stage}"
            )
        if not isinstance(expected, (int, float)) or isinstance(expected, bool):
            raise ResultBundleError(f"summary {summary_metric} for stage {stage} must be numeric")
        actual = aggregates[aggregate_metric]
        if not math.isfinite(float(expected)) or abs(float(expected) - actual) > tolerance:
            raise ResultBundleError(
                f"{label} evaluation aggregate for stage {stage} {summary_metric} differs: "
                f"summary={expected!r} evidence={actual!r}"
            )


def validate_evaluation_evidence(
    run_dir: str | Path,
    summary: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> None:
    """Bind selected and terminal-policy episode rows to canonical claims."""
    run_path = Path(run_dir).resolve()
    expected_episodes = provenance.get("evaluation_episodes")
    evaluation_seeds = provenance.get("evaluation_seeds")
    if not isinstance(expected_episodes, int) or isinstance(expected_episodes, bool) or expected_episodes <= 0:
        raise ResultBundleError("provenance evaluation_episodes must be a positive integer")
    if not isinstance(evaluation_seeds, list) or not evaluation_seeds:
        raise ResultBundleError("provenance evaluation_seeds must be a non-empty list")
    seed_roles = provenance.get("seed_roles")
    if not isinstance(seed_roles, Mapping):
        raise ResultBundleError("provenance seed_roles must be an object")
    publication_evaluation_seed = seed_roles.get("publication_evaluation")
    if (
        not isinstance(publication_evaluation_seed, int)
        or isinstance(publication_evaluation_seed, bool)
        or publication_evaluation_seed < 0
    ):
        raise ResultBundleError("provenance publication_evaluation seed role must be a non-negative integer")
    stages = summary.get("stages")
    if not isinstance(stages, Mapping):
        raise ResultBundleError("summary stages must be an object")

    selected_metric_map = {
        "selected_model_reward": ("reward", 0.0051),
        "selected_model_reward_std": ("reward_std", 0.0051),
        "selected_model_episode_length": ("episode_length", 0.0501),
        "selected_model_episode_length_std": ("episode_length_std", 0.0501),
        "selected_model_forward_vel": ("forward_vel", 0.0051),
        "selected_model_forward_vel_std": ("forward_vel_std", 0.0051),
        "selected_model_distance": ("distance", 0.0051),
        "selected_model_success_rate": ("success_rate", 0.0051),
    }
    final_metric_map = {
        "final_eval_reward": ("reward", 0.0051),
        "final_eval_std": ("reward_std", 0.0051),
        "avg_episode_length": ("episode_length", 0.0501),
        "avg_episode_length_std": ("episode_length_std", 0.0501),
        "avg_forward_vel": ("forward_vel", 0.0051),
        "avg_forward_vel_std": ("forward_vel_std", 0.0051),
        "mean_distance_traveled": ("distance", 0.0051),
        "mean_success_rate": ("success_rate", 0.0051),
    }
    for stage in (1, 2, 3):
        stage_summary = stages.get(str(stage))
        if not isinstance(stage_summary, Mapping):
            raise ResultBundleError(f"summary is missing stage {stage}")
        selected_aggregates = _evaluation_evidence_aggregates(
            run_path / f"stage{stage}" / "evaluation_selected.csv",
            checkpoint_label="selected",
            expected_episodes=expected_episodes,
            evaluation_seeds=evaluation_seeds,
            stage=stage,
        )
        if int(selected_aggregates["evaluation_seed"]) != publication_evaluation_seed:
            raise ResultBundleError(f"stage {stage} selected evidence does not use the publication_evaluation seed")
        _compare_evaluation_aggregates(
            stage_summary,
            selected_aggregates,
            stage=stage,
            label="selected",
            metric_map=selected_metric_map,
        )
        config_path = run_path / f"stage{stage}" / "stage_config.json"
        try:
            config_value = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ResultBundleError(f"cannot read publication gate config for stage {stage}: {exc}") from exc
        if not isinstance(config_value, Mapping):
            raise ResultBundleError(f"publication gate config for stage {stage} must be an object")
        curriculum = config_value.get("curriculum", config_value.get("curriculum_kwargs", {}))
        if not isinstance(curriculum, Mapping):
            raise ResultBundleError(f"publication gate config for stage {stage} must contain a curriculum object")
        # A stage whose gate this evidence file cannot express must NOT be
        # certified on whichever thresholds happen to be checkable.
        #
        # ``stance_quality/v1`` carries ``min_avg_reward`` only as a rail --
        # deliberately set below the zero-action statue, which clears it by
        # 68% -- and states its real criteria as a full-horizon fraction and
        # two ceilings on unsupported duty. The per-episode evidence CSV has
        # ``reward`` and ``length`` but no duty column, so evaluating the four
        # legacy thresholds here would certify the stage on the rail alone:
        # exactly the "a statue clears this gate" failure the stance gate was
        # introduced to remove, reappearing in the publication path.
        gate_kind = curriculum.get("gate_kind")
        if gate_kind == STANCE_GATE_KIND:
            raise ResultBundleError(
                f"stage {stage} declares gate_kind {gate_kind!r}, whose criteria "
                "(min_full_horizon_fraction, max_unsupported_duty, "
                "max_unsupported_duty_ucb) cannot be checked from the publication "
                "evidence file: it records per-episode reward and length but no "
                "unsupported duty. Certifying on min_avg_reward alone would pass a "
                "policy this gate exists to reject, so the bundle refuses instead. "
                "Add a per-episode duty column to evaluation_selected.csv and teach "
                "this function to evaluate the stance criteria."
            )

        publication_thresholds = {
            "min_avg_reward": "reward",
            "min_avg_episode_length": "episode_length",
            "min_avg_forward_vel": "forward_vel",
            "min_success_rate": "success_rate",
        }
        for threshold_name, aggregate_name in publication_thresholds.items():
            threshold_value = curriculum.get(threshold_name)
            if threshold_value is None:
                continue
            if not isinstance(threshold_value, (int, float)) or isinstance(threshold_value, bool):
                raise ResultBundleError(
                    f"publication gate threshold {threshold_name} for stage {stage} must be numeric"
                )
            threshold = float(threshold_value)
            if not math.isfinite(threshold):
                raise ResultBundleError(f"publication gate threshold {threshold_name} for stage {stage} must be finite")
            actual = selected_aggregates[aggregate_name]
            if actual < threshold:
                raise ResultBundleError(
                    f"stage {stage} publication gate fails {threshold_name}: "
                    f"evidence={actual:.6g} threshold={threshold:.6g}"
                )
        final_aggregates = _evaluation_evidence_aggregates(
            run_path / f"stage{stage}" / "evaluation_final.csv",
            checkpoint_label="final",
            expected_episodes=expected_episodes,
            evaluation_seeds=evaluation_seeds,
            stage=stage,
        )
        if int(final_aggregates["evaluation_seed"]) != publication_evaluation_seed:
            raise ResultBundleError(f"stage {stage} final evidence does not use the publication_evaluation seed")
        _compare_evaluation_aggregates(
            stage_summary,
            final_aggregates,
            stage=stage,
            label="final",
            metric_map=final_metric_map,
        )
