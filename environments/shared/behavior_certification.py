"""Skill gates and comparable seeded panels for command/terrain behaviors.

The behavior certificate is distinct from canonical curriculum certification.
Its panels use fixed environment seeds as well as fixed episode seeds: merely
reusing episode seeds with different training run seeds changes the course.
"""

from __future__ import annotations

import json
import math
import tomllib
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

import numpy as np

from environments.shared.curriculum.recovery_gate import binomial_lcb
from environments.shared.direction_commands import wrap_angle
from environments.shared.result_bundle import sha256_file

RULES_PATH = Path(__file__).resolve().parents[2] / "configs" / "behavior_certification.toml"


def load_certification_rules() -> dict[str, Any]:
    return tomllib.loads(RULES_PATH.read_text())


class _PanelTerrainMixin:
    """Stratify existing fixed-template tasks without changing their task identity."""

    @property
    def terrain_families(self) -> tuple[str, ...]:
        sampler = getattr(self, "terrain_sampler", None)
        if sampler is not None:
            return tuple(sampler.families)
        terrain = getattr(self, "terrain_config", None)
        if terrain is None:
            return ("flat",)
        family = "terrain_contact" if terrain.mode == "flat" else terrain.template
        probability = float(getattr(self, "flat_probability", 0.0))
        return tuple((["flat"] if probability > 0 else []) + ([family] if probability < 1 else []))

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        parent = cast(Any, super())
        if getattr(self, "terrain_sampler", None) is not None:
            return cast(tuple[np.ndarray, dict], parent.reset(seed=seed, options=options))
        options = dict(options or {})
        family = options.pop("terrain_family", None)
        if family is not None and family not in self.terrain_families:
            raise ValueError("Certification requested a terrain outside this task")
        previous = getattr(self, "flat_probability", 0.0)
        try:
            if family is not None:
                self.flat_probability = 1.0 if family == "flat" else 0.0
            observation, info = parent.reset(seed=seed, options=options)
        finally:
            self.flat_probability = previous
        if family is not None:
            info["terrain_sampling"] = {"family": family, "mode": "certification_override"}
        return observation, info


def _panel_env(species: str, recipe_path: Path, run_seed: int) -> Any:
    from environments.shared.behavior_env import get_behavior_env_class
    from environments.shared.terrain_sampling import get_sampled_behavior_env_class
    from environments.shared.train_behaviors import read_recipe

    _, commands, terrain, kwargs = read_recipe(recipe_path, species)
    factory = get_sampled_behavior_env_class if "terrain_sampler" in kwargs else get_behavior_env_class
    cls = type("BehaviorCertificationPanelEnv", (_PanelTerrainMixin, factory(species)), {})
    return cls(commands=commands, terrain=terrain, run_seed=run_seed, **kwargs)


def evaluate_saved_panel(
    *,
    model_path: Path,
    normalization_path: Path,
    recipe_path: Path,
    species: str,
    identity: Mapping[str, Any],
    output_dir: Path,
    run_seed: int,
    seed_start: int,
    episodes: int,
) -> dict[str, Any]:
    """Reload the exact saved pair; score deterministic actions with frozen stats."""
    from environments.shared.behavior_checkpoint import load_behavior_checkpoint
    from environments.shared.behavior_evaluation import evaluate_behavior

    original_hashes = sha256_file(model_path), sha256_file(normalization_path)
    env = _panel_env(species, recipe_path, run_seed)
    try:
        if env.behavior_identity != dict(identity):
            raise ValueError("Certification environment differs from the saved behavior identity")
        model, normalizer, _ = load_behavior_checkpoint(
            model_path, normalization_path, env, behavior_identity=identity, species=species
        )
    except BaseException:
        env.close()
        raise
    try:
        report = evaluate_behavior(
            model, normalizer, episode_seeds=list(range(seed_start, seed_start + episodes)), output_dir=output_dir
        )
        for episode in report["episodes"]:
            stem = f"episode_{episode['episode']:03d}_seed_{episode['episode_seed']}"
            reset = json.loads((output_dir / "episodes" / f"{stem}_reset.json").read_text())
            episode["terrain"] = reset["terrain"]
        report["protocol"]["environment_run_seed"] = run_seed
        if original_hashes != (sha256_file(model_path), sha256_file(normalization_path)):
            raise ValueError("Model or normalization changed while its certification panel was running")
        report["model_sha256"], report["normalization_sha256"] = original_hashes
        report["training_lineage"] = {
            "seed": getattr(model, "mesozoic_behavior_training_seed", None),
            "parent_model_sha256": getattr(model, "mesozoic_behavior_training_parent_sha256", None),
            "parent_normalization_sha256": getattr(
                model, "mesozoic_behavior_training_parent_normalization_sha256", None
            ),
            "stage_start_timesteps": getattr(model, "mesozoic_behavior_stage_start", None),
            "num_timesteps": model.num_timesteps,
            "stage_start_updates": getattr(model, "mesozoic_behavior_stage_start_updates", None),
            "optimizer_updates": model._n_updates,
        }
        (output_dir / "evaluation_summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        return report
    finally:
        normalizer.close()


def _ratio(value: Any, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (float, int))
        or not math.isfinite(value)
        or not 0 <= value <= 1
    ):
        raise ValueError(f"Missing or invalid {name}")
    return float(value)


def _event_fraction(events: Sequence[Mapping[str, Any]]) -> float:
    return sum(event.get("settled") is True for event in events) / len(events) if events else 0.0


def episode_measurements(episode: Mapping[str, Any], course_distance: float) -> dict[str, float]:
    """Episode-level samples avoid counting correlated commands as independent trials."""
    if not math.isfinite(course_distance) or course_distance <= 0:
        raise ValueError("Course distance must be finite and positive")
    events = episode.get("events", [])
    if not isinstance(events, list):
        raise ValueError("Episode command evidence is missing")
    eligible = [event for event in events if event.get("eligible") is True]
    stops = [event for event in eligible if event.get("heading_active") is False]
    turns = []
    previous_heading = None
    for event in events:
        heading = event.get("desired_heading")
        if not isinstance(heading, (int, float)) or not math.isfinite(heading):
            raise ValueError("Command heading evidence is missing")
        if (
            previous_heading is not None
            and event.get("eligible") is True
            and event.get("heading_active") is True
            and abs(wrap_angle(heading - previous_heading)) > 0.05
        ):
            turns.append(event)
        previous_heading = heading
    progress = episode.get("max_course_progress_m")
    if (
        not isinstance(progress, (float, int))
        or isinstance(progress, bool)
        or not math.isfinite(progress)
        or progress < 0
    ):
        raise ValueError("Course progress is missing or invalid")
    if not isinstance(episode.get("full_horizon"), bool) or not isinstance(episode.get("fall"), bool):
        raise ValueError("Survival evidence is missing")
    return {
        "survival": float(episode["full_horizon"] and not episode["fall"]),
        "tracking": _ratio(episode.get("tracking_fraction"), "tracking fraction"),
        "settling": _event_fraction(eligible),
        "stopping": _event_fraction(stops),
        "turning": _event_fraction(turns),
        "stop_exposed": float(bool(stops)),
        "turn_exposed": float(bool(turns)),
        "progress": min(1.0, progress / course_distance),
        "progress_m": float(progress),
    }


def judge_behavior_panel(
    report: Mapping[str, Any], identity: Mapping[str, Any], rules: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Fail closed on coverage, short horizons, weak survival or command performance."""
    rules = dict(rules or load_certification_rules())
    failures: list[str] = []
    coverage = report.get("terrain_coverage", {})
    families = coverage.get("enabled_families", [])
    if not families or coverage.get("complete") is not True:
        failures.append("Every enabled terrain family must be evaluated")
    episodes = report.get("episodes", [])
    seen_seeds = [episode.get("episode_seed") for episode in episodes]
    if len(seen_seeds) != len(set(seen_seeds)) or any(
        not isinstance(seed, int) or isinstance(seed, bool) for seed in seen_seeds
    ):
        failures.append("Certification episodes require distinct integer seeds")
    require_stop = identity["commands"]["stop_probability"] > 0
    require_turn = identity["commands"]["turn_increment_max"] > 0
    per_family: dict[str, Any] = {}
    for family in families:
        departure_radius = 0.0
        needs_terrain_departure = family not in ("flat", "terrain_contact")
        if needs_terrain_departure:
            terrain = identity.get("terrain")
            if not isinstance(terrain, Mapping) or any(
                isinstance(terrain.get(name), bool)
                or not isinstance(terrain.get(name), (int, float))
                or not math.isfinite(terrain[name])
                or terrain[name] <= 0
                for name in ("apron_radius", "blend_width")
            ):
                failures.append(f"{family}: valid terrain apron and blend dimensions are required")
                departure_radius = math.inf
            else:
                # The initial contact region is deliberately flat. Progress
                # within it does not demonstrate handling the named terrain.
                # Require the root to leave the entire apron/blend region;
                # short species may need longer courses before certification.
                departure_radius = float(terrain["apron_radius"] + terrain["blend_width"])
        rows = [episode for episode in episodes if episode.get("terrain_family") == family]
        samples = []
        for episode in rows:
            try:
                horizon = episode.get("horizon_s")
                if (
                    not isinstance(horizon, (int, float))
                    or not math.isfinite(horizon)
                    or horizon < rules["minimum_horizon_s"]
                ):
                    raise ValueError("Episode horizon is too short for certification")
                samples.append(episode_measurements(episode, float(identity["course_distance"])))
            except ValueError as exc:
                failures.append(f"{family}: {exc}")
        n = len(samples)
        if n < rules["episodes_per_family"]:
            failures.append(f"{family}: {n} valid episodes; requires {rules['episodes_per_family']}")
        survival = sum(row["survival"] == 1 for row in samples)
        successes = sum(
            row["survival"] == 1
            and row["tracking"] >= rules["minimum_tracking_fraction"]
            and row["settling"] >= rules["minimum_event_settle_fraction"]
            and row["progress"] >= rules["minimum_course_fraction"]
            and (not needs_terrain_departure or row["progress_m"] > departure_radius)
            and (
                not require_stop or not row["stop_exposed"] or row["stopping"] >= rules["minimum_stop_settle_fraction"]
            )
            and (not require_turn or not row["turn_exposed"] or row["turning"] >= rules["minimum_turn_settle_fraction"])
            for row in samples
        )
        survival_lcb = float(binomial_lcb(survival, n)) if n else 0.0
        success_lcb = float(binomial_lcb(successes, n)) if n else 0.0
        if survival_lcb < rules["minimum_survival_lcb"]:
            failures.append(f"{family}: survival lower bound {survival_lcb:.3f} below {rules['minimum_survival_lcb']}")
        if success_lcb < rules["minimum_success_lcb"]:
            failures.append(
                f"{family}: skill-success lower bound {success_lcb:.3f} below {rules['minimum_success_lcb']}"
            )
        command_results = {}
        for label, required in (("stop", require_stop), ("turn", require_turn)):
            exposed = [row for row in samples if row[f"{label}_exposed"]]
            minimum_fraction = rules[f"minimum_{label}_settle_fraction"]
            measurement = "stopping" if label == "stop" else "turning"
            settled = sum(row["survival"] == 1 and row[measurement] >= minimum_fraction for row in exposed)
            fraction = settled / len(exposed) if exposed else 0.0
            command_results[label] = {
                "exposed_episodes": len(exposed),
                "successful_episodes": settled,
                "success_fraction": fraction,
            }
            if required and len(exposed) < rules["minimum_command_episodes"]:
                failures.append(f"{family}: too few episodes with eligible {label} commands")
            if required and fraction < minimum_fraction:
                failures.append(f"{family}: {label} success fraction {fraction:.3f} below {minimum_fraction}")
        per_family[family] = {
            "episodes": n,
            "survived": survival,
            "successful": successes,
            "survival_lcb": survival_lcb,
            "success_lcb": success_lcb,
            "terrain_departure_radius_m": departure_radius if math.isfinite(departure_radius) else None,
            "terrain_departure_episodes": sum(row["progress_m"] > departure_radius for row in samples)
            if needs_terrain_departure
            else None,
            "commands": command_results,
        }
    return {
        "schema": "mesozoic.behavior-certificate/v1",
        "passed": not failures,
        "failures": failures,
        "rules": rules,
        "families": per_family,
    }
