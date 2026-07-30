"""Post-training stage artifact generation.

The shared entry-points that the training notebook, the sweep trial worker,
and the JAX/MJX trainer all call so that stage artifacts stay consistent
across backends."""

from __future__ import annotations

import json as _json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

from .bundles import save_result_bundle
from .csv_output import save_evaluation_episodes
from .text_summaries import write_stage_summary, write_training_summary

if TYPE_CHECKING:
    from ..plant_contract import PlantIdentity

logger = logging.getLogger(__name__)


def build_stage_results_from_eval_data(
    stage_dir: "str | Path",
    stage: int,
    stage_config: dict[str, Any],
    timesteps: int,
    duration_seconds: float = 0.0,
) -> dict[str, Any]:
    """Build a ``stage_results`` dict from on-disk evaluation artifacts.

    Reads ``evaluations.npz`` (written by SB3's ``EvalCallback``) and
    ``metrics.json`` to reconstruct the same results dict that the
    training notebook's ``train_stage`` produces.  This allows sweep
    trials and any other post-hoc consumers to build a consistent
    results dict without re-running evaluation.

    If *duration_seconds* is 0 and a ``metrics.json`` exists, the duration
    is read from ``training_duration_seconds`` in that file.

    Fields that require a live policy evaluation (``mean_forward_vel``,
    ``mean_success_rate``, ``best_model_*``) default to ``0.0`` / ``""``
    and can be updated by the caller after running ``eval_policy``.
    """
    import numpy as _np

    stage_dir = Path(stage_dir)
    model_dir = stage_dir / "models"

    # ── Parse evaluations.npz ───────────────────────────────────────────
    eval_npz = stage_dir / "evaluations.npz"
    mean_reward = 0.0
    std_reward = 0.0
    mean_length = 0.0
    std_length = 0.0
    best_eval_reward: float | str = ""
    best_eval_std: float | str = ""
    best_eval_length: float | str = ""
    best_eval_std_length: float | str = ""
    best_eval_timestep: int | str = ""

    if eval_npz.exists():
        eval_data = _np.load(str(eval_npz))
        eval_rewards = eval_data["results"]
        eval_lengths = eval_data["ep_lengths"]
        eval_timesteps = eval_data["timesteps"]

        mean_per_eval = eval_rewards.mean(axis=1)
        best_idx = int(mean_per_eval.argmax())

        best_eval_reward = round(float(mean_per_eval[best_idx]), 2)
        best_eval_std = round(float(eval_rewards[best_idx].std()), 2)
        best_eval_length = round(float(eval_lengths[best_idx].mean()), 1)
        best_eval_std_length = round(float(eval_lengths[best_idx].std()), 1)
        best_eval_timestep = int(eval_timesteps[best_idx])

        # Use last eval as "final" metrics
        mean_reward = float(mean_per_eval[-1])
        std_reward = float(eval_rewards[-1].std())
        mean_length = float(eval_lengths[-1].mean())
        std_length = float(eval_lengths[-1].std())

    # ── Duration and provenance from sidecars ───────────────────────────
    metrics: dict[str, Any] = {}
    metrics_path = stage_dir / "metrics.json"
    if metrics_path.exists():
        metrics = _json.loads(metrics_path.read_text())
        if duration_seconds == 0.0:
            duration_seconds = metrics.get("training_duration_seconds", 0.0)
    plant_identity = metrics.get("plant_identity")
    if not isinstance(plant_identity, Mapping):
        saved_config_path = stage_dir / "stage_config.json"
        if saved_config_path.exists():
            saved_config = _json.loads(saved_config_path.read_text())
            plant_identity = saved_config.get("plant_identity")

    best_model_path = model_dir / "best_model"
    vecnorm_path = str(model_dir / "best_model_vecnorm.pkl")
    sim_dt = stage_config.get("env_kwargs", {}).get("sim_dt", 0.01)

    result = {
        "stage": stage,
        "name": stage_config["name"],
        "description": stage_config["description"],
        "timesteps": timesteps,
        "duration_seconds": duration_seconds,
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "mean_episode_length": mean_length,
        "std_episode_length": std_length,
        "mean_forward_vel": 0.0,
        "std_forward_vel": 0.0,
        "mean_success_rate": 0.0,
        "best_eval_reward": best_eval_reward,
        "best_eval_std": best_eval_std,
        "best_eval_length": best_eval_length,
        "best_eval_std_length": best_eval_std_length,
        "best_eval_timestep": best_eval_timestep,
        "sim_dt": sim_dt,
        "model_path": str(best_model_path),
        "vecnorm_path": vecnorm_path,
    }
    if isinstance(plant_identity, Mapping):
        result["plant_identity"] = dict(plant_identity)
    return result

def generate_stage_artifacts(
    species_cfg,
    stage_config: dict[str, Any],
    stage: int,
    algorithm: str,
    stage_dir: "str | Path",
    seed: int,
    stage_results: dict[str, Any] | None = None,
    timesteps: int = 0,
    record_videos: bool = True,
    generate_graphs: bool = True,
    allow_legacy_plant: bool = False,
) -> dict[str, Any]:
    """Write stage summary, record replay videos, and generate training graphs.

    This is the single shared entry-point for generating post-training
    artifacts.  Both the training notebook and the sweep trial worker
    call this function so that the artifacts are always consistent.

    When *stage_results* is ``None``, a results dict is built from on-disk
    eval data via :func:`build_stage_results_from_eval_data`.  Callers
    that already have richer metrics (e.g. the notebook, which runs a
    full 30-episode eval) should pass their own *stage_results*.

    When *generate_graphs* is ``True`` (the default), training curves and
    diagnostic graphs are saved to the stage directory.  Requires
    ``matplotlib``.

    Returns the (possibly enriched) *stage_results* dict.
    """
    stage_dir = Path(stage_dir)
    model_dir = stage_dir / "models"
    species = species_cfg.species

    if stage_results is None:
        stage_results = build_stage_results_from_eval_data(
            stage_dir,
            stage,
            stage_config,
            timesteps=timesteps,
        )

    write_stage_summary(stage_dir, stage_results, species, algorithm)
    logger.info("Stage summary written to: %s", stage_dir / "stage_summary.txt")

    # ── Generate training graphs ────────────────────────────────────────
    if generate_graphs:
        try:
            from environments.shared.visualization import (
                plot_diagnostics_graphs,
                plot_foot_contacts,
                plot_stance_diagnostics,
                plot_training_curves,
            )

            stage_dirs = [(stage, stage_dir)]
            stage_configs = {stage: stage_config}

            plot_training_curves(
                stage_dirs,
                stage_configs,
                species,
                algorithm,
                save_path=stage_dir / "training_curves.png",
                show=False,
            )
            plot_diagnostics_graphs(
                stage_dirs,
                stage_configs,
                species,
                algorithm,
                save_dir=stage_dir,
                show=False,
            )
            plot_foot_contacts(
                stage_dirs,
                stage_configs,
                species,
                algorithm,
                save_path=stage_dir / "foot_contacts.png",
                show=False,
            )
            plot_stance_diagnostics(
                stage_dirs,
                stage_configs,
                species,
                algorithm,
                save_path=stage_dir / "stance_diagnostics.png",
                show=False,
            )
        except ImportError:
            logger.warning("Skipping graph generation (matplotlib not installed).")
        except Exception:
            logger.warning("Graph generation failed.", exc_info=True)

    if not record_videos:
        return stage_results

    # ── Record replay videos for best and final models ──────────────────
    from ..plant_contract import PlantCompatibilityError, current_plant_identity, validate_model_plant

    try:
        from environments.shared.evaluation import TREX_STAGE1_CAMERA_VIEWS, record_stage_video
        from environments.shared.train_base import _ensure_sb3

        sb3 = _ensure_sb3()
        env_kwargs = stage_config["env_kwargs"].copy()
        alg_cls = sb3["SAC"] if algorithm == "sac" else sb3["PPO"]
        plant_identity = current_plant_identity(species)

        best_model_path = model_dir / "best_model"
        vecnorm_path = str(model_dir / "best_model_vecnorm.pkl")
        final_path = model_dir / f"stage{stage}_final"
        final_vecnorm_path = str(final_path) + "_vecnorm.pkl"
        replay_diagnostics = species.lower() == "trex" and stage == 1
        replay_camera_views = TREX_STAGE1_CAMERA_VIEWS if replay_diagnostics else None

        if (model_dir / "best_model.zip").exists():
            best_model = alg_cls.load(str(best_model_path))
            validate_model_plant(
                best_model,
                plant_identity,
                artifact=str(best_model_path) + ".zip",
                allow_legacy=allow_legacy_plant,
            )
            record_stage_video(
                best_model,
                env_class=species_cfg.env_class,
                env_kwargs=env_kwargs,
                stage=stage,
                stage_dir=stage_dir,
                species=species,
                algorithm=algorithm,
                seed=seed,
                vecnorm_path=vecnorm_path,
                label="best",
                plant_identity=plant_identity,
                allow_legacy_plant=allow_legacy_plant,
                camera_views=replay_camera_views,
                collect_stance_diagnostics=replay_diagnostics,
            )

        if (Path(str(final_path) + ".zip")).exists():
            final_model = alg_cls.load(str(final_path))
            validate_model_plant(
                final_model,
                plant_identity,
                artifact=str(final_path) + ".zip",
                allow_legacy=allow_legacy_plant,
            )
            record_stage_video(
                final_model,
                env_class=species_cfg.env_class,
                env_kwargs=env_kwargs,
                stage=stage,
                stage_dir=stage_dir,
                species=species,
                algorithm=algorithm,
                seed=seed,
                vecnorm_path=final_vecnorm_path,
                label="final",
                plant_identity=plant_identity,
                allow_legacy_plant=allow_legacy_plant,
                camera_views=replay_camera_views,
                collect_stance_diagnostics=replay_diagnostics,
            )
    except PlantCompatibilityError:
        raise
    except Exception:
        logger.warning("Video recording failed.", exc_info=True)

    return stage_results

def save_jax_stage_artifacts(
    species: str,
    stage: int,
    stage_config: dict[str, Any],
    stage_results: dict[str, Any],
    stage_dir: "str | Path",
    run_dir: "str | Path",
    eval_results: Any,
    params: Any,
    obs_rms: Any,
    *,
    final_eval_results: Any | None = None,
    seed: int = 42,
    num_envs: int = 2048,
    reward_cfg: dict[str, float] | None = None,
    best_params: Any | None = None,
    best_reward: float = 0.0,
    best_update: int = 0,
    evaluation_seed: int = 42,
    backend_version: str | None = None,
    plant_identity: PlantIdentity | None = None,
) -> dict[str, Path]:
    """Save all post-training artifacts for a JAX/MJX training stage.

    Orchestrates the same artifact generation that the SB3 path performs
    via :func:`generate_stage_artifacts`, but using JAX-native checkpoint
    formats and without requiring an SB3 ``SpeciesConfig``.

    Artifacts saved:

    * ``stage_summary.txt`` — human-readable stage summary
    * ``stage_config.json`` — frozen config snapshot
    * ``collected_results.csv`` — one row per stage (append-safe)
    * ``diagnostics.npz`` — per-step evaluation diagnostics
    * ``best_model.pkl`` — best checkpoint (params + obs stats)
    * ``stage{N}_final.pkl`` — final checkpoint
    * ``training_summary.txt`` — run-level summary

    Args:
        species: Species identifier (e.g. ``"velociraptor"``).
        stage: Curriculum stage number (1, 2, or 3).
        stage_config: Config dict from :func:`config.load_stage_config`.
        stage_results: Results dict with keys like ``mean_reward``,
            ``timesteps``, ``duration_seconds``, ``model_path``, etc.
        stage_dir: Directory for this stage's output files.
        run_dir: Parent run directory (for CSV and training summary).
        eval_results: :class:`jax_eval.EvalResults` instance with
            selected-checkpoint episode and per-step diagnostic data.
        final_eval_results: Episode evidence for the terminal parameters.
            Defaults to *eval_results* only for backward-compatible callers
            whose selected and terminal parameters are identical.
        params: Final JAX network parameters.
        obs_rms: Observation normalisation statistics.
        seed: Random seed used for training.
        num_envs: Number of parallel environments.
        reward_cfg: Reward weight dict (included in config snapshot).
        best_params: Best-performing parameters (falls back to *params*).
        best_reward: Best evaluation reward achieved during training.
        best_update: Update number at which *best_params* was recorded.
        evaluation_seed: Seed used for the fixed publication evaluation.
        backend_version: Optional explicit JAX version for portable tests or
            environments where package metadata cannot be detected.
        plant_identity: Optional precomputed plant identity.  The current
            identity is resolved and verified when omitted.

    Returns:
        Dict mapping artifact name to its file path.
    """
    import numpy as _np

    from ..config import save_stage_config
    from ..jax_checkpoint import save_checkpoint
    from ..plant_contract import current_plant_identity, write_plant_identity
    from ..result_bundle import (
        ResultBundleError,
        initialize_result_bundle,
        load_provenance,
        validate_result_bundle,
    )

    stage_dir = Path(stage_dir)
    run_dir = Path(run_dir)
    if stage not in {1, 2, 3}:
        raise ValueError("stage must be 1, 2, or 3")
    if stage_dir.parent.resolve() != run_dir.resolve():
        raise ValueError("stage_dir must be run_dir/stage<N> for a portable result bundle")
    if stage_dir.name != f"stage{stage}":
        raise ValueError(f"stage_dir must be named stage{stage}")
    if plant_identity is None:
        plant_identity = current_plant_identity(species)

    manifest_path = run_dir / "artifact_manifest.json"
    if manifest_path.exists():
        try:
            existing_manifest = _json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, _json.JSONDecodeError) as exc:
            raise ResultBundleError(f"cannot read existing artifact manifest: {exc}") from exc
        if isinstance(existing_manifest, Mapping) and existing_manifest.get("status") == "complete":
            validate_result_bundle(run_dir, require_complete=True)
            raise ResultBundleError("completed result bundle is immutable; use a new run_id to rewrite a stage")

    existing_stages: set[int] = set()
    for existing_result_path in sorted(run_dir.glob("stage*/stage_result.json")):
        try:
            saved_result = _json.loads(existing_result_path.read_text(encoding="utf-8"))
            saved_stage = int(saved_result["stage"])
        except (OSError, _json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise ResultBundleError(f"cannot read prior JAX stage record {existing_result_path}") from exc
        if existing_result_path.parent.name != f"stage{saved_stage}" or saved_stage not in {1, 2, 3}:
            raise ResultBundleError(f"mislabeled prior JAX stage record: {existing_result_path}")
        if saved_stage in existing_stages:
            raise ResultBundleError(f"duplicate prior JAX stage record for stage {saved_stage}")
        existing_stages.add(saved_stage)
    combined_stages = existing_stages | {stage}
    expected_stages = set(range(1, max(combined_stages) + 1))
    if combined_stages != expected_stages or max(combined_stages) > stage:
        raise ResultBundleError(f"JAX stages must be saved as a contiguous prefix; found {sorted(combined_stages)}")

    episode_fields = ("rewards", "lengths", "forward_vels", "distances", "successes")
    has_episode_evidence = all(hasattr(eval_results, field) for field in episode_fields)
    terminal_eval_results = final_eval_results if final_eval_results is not None else eval_results
    has_final_evidence = all(hasattr(terminal_eval_results, field) for field in episode_fields)
    evaluation_episode_count = len(getattr(eval_results, "rewards", [])) if has_episode_evidence else 30
    if not has_episode_evidence:
        raise ResultBundleError("JAX selected evaluation evidence is missing episode arrays")
    if evaluation_episode_count <= 0:
        raise ResultBundleError("JAX selected evaluation evidence must contain at least one episode")
    if not has_final_evidence:
        raise ResultBundleError("JAX terminal evaluation evidence is missing episode arrays")
    for label, evidence in (("selected", eval_results), ("final", terminal_eval_results)):
        evidence_lengths = {len(getattr(evidence, field)) for field in episode_fields}
        if evidence_lengths != {evaluation_episode_count}:
            raise ResultBundleError(f"JAX {label} evaluation evidence sequences must have equal lengths")
    selected_rewards = _np.asarray(eval_results.rewards, dtype=float)
    selected_lengths = _np.asarray(eval_results.lengths, dtype=float)
    selected_forward_velocities = _np.asarray(eval_results.forward_vels, dtype=float)
    selected_distances = _np.asarray(eval_results.distances, dtype=float)
    selected_successes = _np.asarray(eval_results.successes, dtype=float)
    final_rewards = _np.asarray(terminal_eval_results.rewards, dtype=float)
    final_lengths = _np.asarray(terminal_eval_results.lengths, dtype=float)
    final_forward_velocities = _np.asarray(terminal_eval_results.forward_vels, dtype=float)
    final_distances = _np.asarray(terminal_eval_results.distances, dtype=float)
    final_successes = _np.asarray(terminal_eval_results.successes, dtype=float)
    stage_results.update(
        {
            "mean_reward": round(float(final_rewards.mean()), 2),
            "std_reward": round(float(final_rewards.std()), 2),
            "mean_episode_length": round(float(final_lengths.mean()), 1),
            "std_episode_length": round(float(final_lengths.std()), 1),
            "mean_forward_vel": round(float(final_forward_velocities.mean()), 3),
            "std_forward_vel": round(float(final_forward_velocities.std()), 3),
            "mean_distance_traveled": round(float(final_distances.mean()), 3),
            "mean_success_rate": round(float(final_successes.mean()), 4),
            "best_model_reward": round(float(selected_rewards.mean()), 2),
            "best_model_std_reward": round(float(selected_rewards.std()), 2),
            "best_model_length": round(float(selected_lengths.mean()), 1),
            "best_model_std_length": round(float(selected_lengths.std()), 1),
            "best_model_fwd_vel": round(float(selected_forward_velocities.mean()), 3),
            "best_model_std_fwd_vel": round(float(selected_forward_velocities.std()), 3),
            "best_model_distance": round(float(selected_distances.mean()), 3),
            "best_model_success_rate": round(float(selected_successes.mean()), 4),
        }
    )

    try:
        captured_provenance = load_provenance(run_dir)
    except ResultBundleError:
        captured_provenance = {}
    seed_roles = captured_provenance.get("seed_roles") or {
        "training": seed,
        "publication_evaluation": evaluation_seed,
    }
    initialize_result_bundle(
        run_dir,
        species=species,
        algorithm="JAX_PPO",
        backend="jax-mjx",
        seed=seed,
        evaluation_seeds=[evaluation_seed],
        evaluation_episodes=evaluation_episode_count,
        seed_roles=seed_roles,
        parallel_envs=num_envs,
        hardware=str(captured_provenance.get("hardware") or "Google Colab"),
        plant_identity=plant_identity.to_dict(),
        run_id=captured_provenance.get("run_id"),
    )
    captured_provenance = load_provenance(run_dir)
    if manifest_path.exists():
        manifest_path.unlink()

    model_dir = stage_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    write_plant_identity(run_dir / "plant_identity.json", plant_identity)
    stage_results["plant_identity"] = plant_identity.to_dict()

    # 1. Stage summary text file
    summary_path = write_stage_summary(stage_dir, stage_results, species, "JAX/MJX PPO")
    paths["stage_summary"] = summary_path
    logger.info("Stage summary saved: %s", summary_path)

    # 2. Stage config snapshot
    config_path = save_stage_config(
        stage_dir,
        stage=stage,
        stage_config=stage_config,
        algorithm="jax_ppo",
        species=species,
        extra={
            "seed": seed,
            "num_envs": num_envs,
            "reward_cfg": reward_cfg or {},
        },
        plant_identity=plant_identity,
    )
    paths["stage_config"] = config_path
    logger.info("Stage config saved: %s", config_path)

    # 3. Per-episode evidence for both the selected and terminal parameters.
    evaluation_path = save_evaluation_episodes(
        stage_dir,
        rewards=eval_results.rewards,
        lengths=eval_results.lengths,
        forward_velocities=eval_results.forward_vels,
        distances=eval_results.distances,
        successes=eval_results.successes,
        evaluation_seed=evaluation_seed,
        checkpoint_label="selected",
    )
    paths["evaluation_episodes"] = evaluation_path
    final_evaluation_path = save_evaluation_episodes(
        stage_dir,
        rewards=terminal_eval_results.rewards,
        lengths=terminal_eval_results.lengths,
        forward_velocities=terminal_eval_results.forward_vels,
        distances=terminal_eval_results.distances,
        successes=terminal_eval_results.successes,
        evaluation_seed=evaluation_seed,
        checkpoint_label="final",
    )
    paths["final_evaluation_episodes"] = final_evaluation_path

    # 4. Diagnostics NPZ from eval results
    diag_data: dict[str, Any] = {
        # JAX diagnostics are a contiguous evaluation trace rather than
        # rollout snapshots. Expose a compatible step axis for dashboards.
        "timesteps": _np.arange(len(eval_results.diag_tilt), dtype=int),
        "tilt_angle": _np.array(eval_results.diag_tilt),
        "forward_vel": _np.array(eval_results.diag_fwd_vel),
        "pelvis_height": _np.array(eval_results.diag_pelvis_h),
        "energy": _np.array(eval_results.diag_energy),
    }
    if eval_results.diag_l_foot:
        diag_data["l_foot_contact"] = _np.array(eval_results.diag_l_foot)
        diag_data["r_foot_contact"] = _np.array(eval_results.diag_r_foot)
    for comp_name, comp_vals in eval_results.diag_reward_components.items():
        diag_data[f"reward_{comp_name}"] = _np.array(comp_vals)
    for diagnostic_name, diagnostic_vals in getattr(eval_results, "diag_reward_diagnostics", {}).items():
        diag_data[diagnostic_name] = _np.array(diagnostic_vals)
    if species.lower() == "trex" and stage == 1 and eval_results.diag_l_foot and eval_results.diag_r_foot:
        from ..stance_diagnostics import derive_stance_info

        derived_rows = [
            derive_stance_info(
                {
                    "r_foot_contact": right_force,
                    "l_foot_contact": left_force,
                }
            )
            for right_force, left_force in zip(
                eval_results.diag_r_foot,
                eval_results.diag_l_foot,
                strict=True,
            )
        ]
        if derived_rows:
            for diagnostic_name in derived_rows[0]:
                diag_data[diagnostic_name] = _np.array(
                    [row[diagnostic_name] for row in derived_rows],
                    dtype=float,
                )

    diag_path = stage_dir / "diagnostics.npz"
    _np.savez(diag_path, **diag_data)
    paths["diagnostics"] = diag_path
    logger.info("Diagnostics saved: %s", diag_path)

    # 5. Model checkpoints (best + final)
    effective_best = best_params if best_params is not None else params
    best_model_path = model_dir / "best_model.pkl"
    save_checkpoint(
        best_model_path,
        effective_best,
        obs_rms=obs_rms,
        extra={"best_reward": best_reward, "best_update": best_update},
        plant_identity=plant_identity,
    )
    paths["best_model"] = best_model_path

    final_model_path = model_dir / f"stage{stage}_final.pkl"
    save_checkpoint(final_model_path, params, obs_rms=obs_rms, plant_identity=plant_identity)
    paths["final_model"] = final_model_path
    logger.info("Models saved: %s, %s", best_model_path, final_model_path)

    # 6. Persist one idempotent stage record for cross-session curricula.
    stage_results["model_path"] = best_model_path.resolve().relative_to(run_dir.resolve()).as_posix()
    persisted_keys = (
        "stage",
        "name",
        "description",
        "timesteps",
        "duration_seconds",
        "mean_reward",
        "std_reward",
        "mean_episode_length",
        "std_episode_length",
        "mean_forward_vel",
        "std_forward_vel",
        "mean_distance_traveled",
        "mean_success_rate",
        "best_eval_reward",
        "best_eval_std",
        "best_eval_length",
        "best_eval_std_length",
        "best_eval_timestep",
        "selection_training_return",
        "selection_training_update",
        "gate_passed",
        "publication_gate_passed",
        "best_model_reward",
        "best_model_std_reward",
        "best_model_length",
        "best_model_std_length",
        "best_model_fwd_vel",
        "best_model_std_fwd_vel",
        "best_model_distance",
        "best_model_success_rate",
        "model_path",
        "plant_identity",
    )
    persisted_result = {key: stage_results[key] for key in persisted_keys if key in stage_results}
    stage_result_path = stage_dir / "stage_result.json"
    stage_result_path.write_text(_json.dumps(persisted_result, indent=2, sort_keys=True) + "\n")
    paths["stage_result"] = stage_result_path

    accumulated_results: list[dict[str, Any]] = []
    accumulated_configs: dict[int, dict[str, Any]] = {}
    for existing_result_path in sorted(run_dir.glob("stage*/stage_result.json")):
        saved_result = _json.loads(existing_result_path.read_text())
        saved_stage = int(saved_result["stage"])
        accumulated_results.append(saved_result)
        saved_config_path = existing_result_path.parent / "stage_config.json"
        saved_config = _json.loads(saved_config_path.read_text())
        accumulated_configs[saved_stage] = {
            "name": saved_config.get("name", f"Stage {saved_stage}"),
            "description": saved_config.get("description", f"Curriculum stage {saved_stage}"),
            "env_kwargs": saved_config.get("reward_weights", {}),
            "jax_kwargs": saved_config.get("hyperparameters", {}),
            "curriculum_kwargs": saved_config.get("curriculum", {}),
        }
    accumulated_results.sort(key=lambda result: int(result["stage"]))

    # 7. Training summary (run-level, regenerated from every saved stage)
    training_summary_path = write_training_summary(
        run_dir,
        accumulated_results,
        species,
        algorithm="JAX/MJX PPO",
        seed=seed,
        n_envs=num_envs,
    )
    paths["training_summary"] = training_summary_path
    logger.info("Training summary saved: %s", training_summary_path)

    # 8. Canonical CSV/provenance/manifest; summary.json appears at Stage 3.
    bundle_paths = save_result_bundle(
        accumulated_results,
        accumulated_configs,
        species,
        "JAX_PPO",
        seed,
        run_dir,
        backend="jax-mjx",
        backend_version=backend_version,
        parallel_envs=num_envs,
        hardware=str(captured_provenance.get("hardware") or "Google Colab"),
        evaluation_episodes=evaluation_episode_count,
        evaluation_seeds=[evaluation_seed],
        seed_roles=captured_provenance.get("seed_roles"),
        plant_identity=plant_identity.to_dict(),
        run_id=captured_provenance.get("run_id"),
    )
    paths.update(bundle_paths)
    logger.info("Canonical JAX result bundle saved: %s", run_dir)

    return paths
