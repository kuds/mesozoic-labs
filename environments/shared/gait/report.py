"""Fresh, full-panel physical gait evidence for the exact selected policy pair.

Report-only evaluation never changes task settings or historical verdicts.
The enforced gate uses only explicitly declared criteria and checks the
written evidence independently. Every episode retains its substep trace.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, cast

import numpy as np

from ..constants import PUBLICATION_SEED_START
from ..curriculum.gait_gate import GAIT_GATE_KIND, GaitGateThresholds, evaluate_gait_gate
from ..curriculum.gate_schema import validate_gate_config
from ..file_io import atomic_savez, atomic_write_csv, atomic_write_json
from ..policy_loading import load_sb3_checkpoint
from ..result_bundle.hashing import canonical_json_sha256, sha256_file
from ..result_bundle.reentry import refuse_write_into_complete_run
from ..task_fingerprint import read_checkpoint_task_fingerprint, stage_task_fingerprint, validate_recorded_task
from .identity import measurement_protocol, protocol_sha256
from .metrics import episode_gait_metrics
from .morphology import GaitMorphology
from .recorder import SubstepContactRecorder
from .seeds import checkpoint_seed_provenance
from .types import GaitProtocol

REPORT_SCHEMA = "mesozoic.gait-report/v1"
PANEL_FIELDS = (
    "episode",
    "seed",
    "length",
    "completed_horizon",
    "duration_s",
    "reward",
    "mean_speed_mps",
    "phase_match_fraction",
    "simultaneous_fraction",
    "flight_fraction",
    "max_flight_s",
    "max_slip_distance_over_leg",
    "body_support_fraction",
    "telemetry_valid",
    "checkpoint_sha256",
    "normalization_sha256",
    "task_sha256",
    "measurement_protocol_sha256",
    "metrics_json",
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _provisional_thresholds(
    profile: str, digest: str, curriculum: dict[str, Any], episodes: int, duration_s: float
) -> GaitGateThresholds:
    # These are interpretable development diagnostics, never a certificate.
    # Flight is permitted for the alternating-biped profile: a real running
    # gait must not be rejected merely because it has an aerial phase.
    return GaitGateThresholds.from_curriculum(
        {
            "gait_profile": profile,
            "measurement_protocol_sha256": digest,
            "min_eval_episodes": episodes,
            "gait_panel_seed_start": 0,
            "min_gait_success_lcb": 0.8,
            "min_episode_forward_vel": max(0.0, float(curriculum.get("min_avg_forward_vel", 0.0))),
            "min_episode_duration_s": duration_s,
            "min_complete_cycles_per_foot": 3,
            "min_phase_match_fraction": 0.8,
            "max_simultaneous_fraction": 0.2,
            "max_flight_fraction": 1.0 if profile == "biped_alternating" else 0.05,
            "max_flight_s": 0.5 if profile == "biped_alternating" else 0.08,
            "max_slip_distance_over_leg": 0.1,
            "max_body_support_fraction": 0.01,
        }
    )


def roll_gait_panel(
    env: Any,
    species: str,
    predict: Callable[[Any], np.ndarray],
    save_dir: str | Path,
    *,
    episodes: int,
    seed: int,
    horizon: int,
    protocol: GaitProtocol,
    settle_s: float,
    direction_xy: tuple[float, float],
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, str]]]:
    """Roll explicitly seeded episodes, restoring the hook on every exit."""
    output = Path(save_dir)
    rows: list[dict[str, Any]] = []
    traces: list[dict[str, str]] = []
    description: dict[str, Any] = {}
    for index in range(episodes):
        observation, _ = env.reset(seed=seed + index)
        morph = GaitMorphology.from_env(env, species)
        if not description:
            description = morph.describe()
        elif description != morph.describe():
            raise ValueError("gait morphology/normalization changed within the fixed panel")
        reset_digest = canonical_json_sha256(
            {
                "qpos": env.data.qpos.tolist(),
                "qvel": env.data.qvel.tolist(),
                "mocap_pos": env.data.mocap_pos.tolist(),
            }
        )
        reward_sum = 0.0
        terminated = truncated = False
        info: dict[str, Any] = {}
        length = 0
        with SubstepContactRecorder(env, morph) as recorder:
            for length in range(1, horizon + 1):
                action = predict(observation)
                observation, reward, terminated, truncated, info = env.step(action)
                reward_sum += float(reward)
                if terminated or truncated:
                    break
        trace = recorder.trace()
        row = episode_gait_metrics(
            trace,
            body_weight_n=morph.body_weight_n,
            leg_length_m=morph.leg_length_m,
            foot_names=morph.foot_names,
            protocol=protocol,
            settle_s=settle_s,
            direction_xy=direction_xy,
        )
        row.update(
            {
                "episode": index,
                "seed": seed + index,
                "length": length,
                "reward": reward_sum,
                "completed_horizon": bool(length == horizon and not terminated),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "termination_reason": str(info.get("termination_reason", "")),
                "reset_state_sha256": reset_digest,
            }
        )
        relative = f"gait_traces/episode_{index:04d}.npz"
        atomic_savez(output / relative, **trace)
        traces.append({"path": relative, "sha256": sha256_file(output / relative)})
        rows.append(_json_safe(row))
    return rows, description, traces


def write_gait_report(
    species_cfg: Any,
    stage_config: dict[str, Any],
    model_path: str | Path | None,
    vecnorm_path: str | Path | None,
    save_dir: str | Path,
    *,
    episodes: int = 40,
    seed: int = PUBLICATION_SEED_START,
    algorithm: str = "PPO",
    allow_legacy_plant: bool = False,
    protocol: GaitProtocol | None = None,
    settle_s: float = 1.0,
    direction_xy: tuple[float, float] = (1.0, 0.0),
) -> dict[str, Any]:
    """Write fresh evidence or leave an explicit incomplete refusal record.

    ``model_path=None`` evaluates zero action as a negative control and can
    never produce a certificate. Missing normalization is fatal for a saved
    policy. Complete published result bundles are never edited in place.
    """
    from ..plant_contract import current_plant_identity, validate_environment_plant

    output = Path(save_dir)
    for ancestor in (output.resolve(), *output.resolve().parents):
        refuse_write_into_complete_run(ancestor, what="Gait evaluation")
    output.mkdir(parents=True, exist_ok=True)
    placeholder = {
        "schema": REPORT_SCHEMA,
        "status": "incomplete",
        "report_only": stage_config.get("curriculum_kwargs", {}).get("gate_kind") != GAIT_GATE_KIND,
        "certification_eligible": False,
        "species": species_cfg.species,
    }
    # Invalidate an earlier result before validation or loading can fail.
    atomic_write_json(output / "gait_report.json", placeholder, allow_nan=False)
    if isinstance(episodes, bool) or not isinstance(episodes, int) or episodes < 1:
        raise ValueError("gait panel episodes must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("gait panel seed must be a nonnegative integer")
    if not math.isfinite(settle_s) or settle_s < 0:
        raise ValueError("gait settle_s must be finite and nonnegative")
    direction = np.asarray(direction_xy, dtype=float)
    if direction.shape != (2,) or not np.all(np.isfinite(direction)) or np.linalg.norm(direction) <= 0:
        raise ValueError("gait direction must be a finite nonzero 2-vector")
    direction /= np.linalg.norm(direction)
    direction_xy = (float(direction[0]), float(direction[1]))
    protocol = protocol or GaitProtocol()
    species = species_cfg.species
    curriculum = stage_config.get("curriculum_kwargs", {})
    report_only = curriculum.get("gate_kind") != GAIT_GATE_KIND
    if not report_only:
        validate_gate_config(stage_config.get("_gait_stage", "locomotion"), curriculum)
    if not report_only and (
        episodes != curriculum.get("min_eval_episodes") or seed != curriculum.get("gait_panel_seed_start")
    ):
        raise ValueError("certification panel must match its declared fixed episode count and seed start")
    stage = stage_config.get("_gait_stage", "locomotion")
    env_kwargs = dict(stage_config.get("env_kwargs", {}))
    horizon = int(env_kwargs.get("max_episode_steps", 1000))
    if horizon <= 0:
        raise ValueError("gait horizon must be positive")
    env = normalizer = None
    try:
        plant = current_plant_identity(species)
        task = stage_task_fingerprint(species, stage, env_kwargs=env_kwargs, plant_identity=plant)
        env = species_cfg.env_class(**env_kwargs)
        validate_environment_plant(env, plant, artifact=f"{species} gait report environment")
        digest_checkpoint = digest_normalization = None
        seed_provenance: dict[str, Any] | None = None
        seed_provenance_error: str | None = None
        if model_path is None:
            if not report_only:
                raise ValueError("a zero-action reference cannot be certified as a saved checkpoint")

            def predict(obs: Any) -> np.ndarray:
                return np.zeros(env.action_space.shape, dtype=np.float64)

            controller = "zero_action_reference"
            task_validated = False
        else:
            model_file = Path(model_path)
            if not model_file.is_file() and not model_file.name.endswith(".zip"):
                model_file = model_file.with_name(model_file.name + ".zip")
            if vecnorm_path is None:
                raise ValueError("gait evaluation requires the checkpoint's explicit matched VecNormalize file")
            normalization_file = Path(vecnorm_path)
            digest_checkpoint, digest_normalization = sha256_file(model_file), sha256_file(normalization_file)
            try:
                seed_provenance = checkpoint_seed_provenance(model_file, seed_start=seed, episodes=episodes)
            except ValueError as error:
                if not report_only:
                    raise
                seed_provenance_error = str(error)
            recorded_task = read_checkpoint_task_fingerprint(model_file)
            validate_recorded_task(recorded_task, task, mode="resume_same_stage", artifact=str(model_file))
            # A v1 migration allowance is insufficient to mint a new exact
            # task-bound certificate, even if the general loader accepts it.
            task_validated = bool(recorded_task and recorded_task.get("task_sha256") == task["task_sha256"])
            model, normalizer, _ = load_sb3_checkpoint(
                str(model_file),
                str(normalization_file),
                lambda: species_cfg.env_class(**env_kwargs),
                guess_sidecar=False,
                allow_unnormalized=False,
                plant_identity=plant,
                allow_legacy_plant=allow_legacy_plant,
            )
            if normalizer is None:
                raise ValueError("saved gait policy has no normalization")

            def predict(obs: Any) -> np.ndarray:
                return np.asarray(model.predict(normalizer.normalize_obs(obs), deterministic=True)[0])

            controller = "selected_checkpoint"
        env.reset(seed=seed)
        morph = GaitMorphology.from_env(env, species)
        measured_duration = horizon * float(env.dt) - settle_s
        if measured_duration <= 0:
            raise ValueError("gait settling exclusion must leave a positive analysis window")
        payload = measurement_protocol(
            species,
            protocol,
            settle_s=settle_s,
            direction_xy=direction_xy,
            horizon=horizon,
            physics_dt_s=float(env.model.opt.timestep),
            control_dt_s=float(env.dt),
            episodes=episodes,
            seed_start=seed,
        )
        digest_protocol = protocol_sha256(payload)
        profile = curriculum.get(
            "gait_profile", "biped_alternating" if len(morph.foot_names) == 2 else "quadruped_walk"
        )
        if report_only:
            thresholds = _provisional_thresholds(profile, digest_protocol, curriculum, episodes, measured_duration)
        else:
            thresholds = GaitGateThresholds.from_curriculum(curriculum)
            if thresholds.measurement_protocol_sha256 != digest_protocol:
                raise ValueError(
                    "declared gait protocol hash differs from the detector, implementation, or fixed panel"
                )
        rows, morphology, traces = roll_gait_panel(
            env,
            species,
            predict,
            output,
            episodes=episodes,
            seed=seed,
            horizon=horizon,
            protocol=protocol,
            settle_s=settle_s,
            direction_xy=direction_xy,
        )
        # If files were replaced during an evaluation, the measured policy
        # cannot be attributed to whichever bytes happen to exist afterwards.
        if model_path is not None and (
            sha256_file(model_file) != digest_checkpoint or sha256_file(normalization_file) != digest_normalization
        ):
            raise ValueError("checkpoint or normalization changed during gait evaluation")
        result = evaluate_gait_gate(rows, thresholds, foot_names=morph.foot_names)
        unique_resets = len({row["reset_state_sha256"] for row in rows})
        eligible = bool(
            model_path is not None
            and task_validated
            and seed_provenance is not None
            and not allow_legacy_plant
            and unique_resets == episodes
        )
        reasons = Counter(reason for failures in result.episode_failures for reason in failures)
        bindings = {
            "checkpoint_sha256": digest_checkpoint,
            "normalization_sha256": digest_normalization,
            "task_sha256": task["task_sha256"],
            "measurement_protocol_sha256": digest_protocol,
        }
        csv_rows = []
        for row in rows:
            csv_rows.append(
                {
                    **{key: row.get(key) for key in PANEL_FIELDS if key not in {*bindings, "metrics_json"}},
                    **bindings,
                    "metrics_json": json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False),
                }
            )
        panel_path = atomic_write_csv(output / "gait_panel.csv", PANEL_FIELDS, csv_rows)
        report = {
            "schema": REPORT_SCHEMA,
            "status": "complete",
            "species": species,
            "stage": stage,
            "algorithm": algorithm,
            "seed_start": seed,
            "controller": controller,
            "plant_validated": not allow_legacy_plant,
            "task_validated": task_validated,
            "certification_eligible": eligible,
            "report_only": report_only,
            "certified": bool(not report_only and eligible and result.passed),
            "development_panel_passed": bool(result.passed),
            "threshold_status": "provisional development criteria"
            if report_only
            else "explicit declared gate criteria",
            **bindings,
            "plant_identity": plant.to_dict(),
            "task_fingerprint": task,
            "seed_provenance": seed_provenance,
            "seed_provenance_error": seed_provenance_error,
            "measurement_protocol": payload,
            "gait_profile": profile,
            "foot_names": list(morph.foot_names),
            "morphology": morphology,
            "thresholds": asdict(thresholds),
            "episodes": rows,
            "panel_csv": {"path": panel_path.name, "sha256": sha256_file(panel_path)},
            "traces": traces,
            "panel": {
                "episodes": episodes,
                "seed_start": seed,
                "unique_reset_states": unique_resets,
                "analysis_duration_s": measured_duration,
            },
            "statistics": {
                "success_count": result.success_count,
                "n_episodes": result.n_episodes,
                "success_fraction": result.success_fraction,
                "success_lcb": result.success_lcb,
                "confidence": 0.95,
                "method": "one-sided exact binomial lower confidence bound",
            },
            "failures": list(result.failures),
            "episode_failure_counts": dict(reasons),
        }
        report = cast(dict[str, Any], _json_safe(report))
        atomic_write_json(output / "gait_report.json", report, allow_nan=False)
        return report
    except Exception as error:
        atomic_write_json(
            output / "gait_report.json", {**placeholder, "error": f"{type(error).__name__}: {error}"}, allow_nan=False
        )
        raise
    finally:
        if normalizer is not None:
            normalizer.close()
        if env is not None:
            env.close()
