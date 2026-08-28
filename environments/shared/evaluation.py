"""Policy evaluation and video recording utilities.

Extracted from ``train_base.py`` for maintainability.  Contains the
evaluation loop, video recorder, and human-readable result logger.
"""

from __future__ import annotations

import logging
import re
import zlib
from pathlib import Path
from typing import Any, Mapping

from .plant_contract import PlantIdentity
from .stage_manifest import stage_label


def replay_seed(seed: int, stage: "int | str") -> int:
    """Deterministic replay-env seed, decorrelated from training/eval seeds.

    Integer stages keep the historical ``seed + 2000 + stage`` arithmetic so
    every existing replay stays reproducible. A semantic stage id maps to a
    stable small offset via crc32 — NOT ``hash()``, which varies per process
    — so the same stage replays the same episode on every run. Regression:
    the ``int + str`` form raised ``TypeError`` inside the best-effort
    replay recorder, which silently cost the recovery stage both its
    replays in the field.
    """
    if isinstance(stage, int):
        return seed + 2000 + stage
    return seed + 2000 + (zlib.crc32(stage.encode("utf-8")) % 1000)


logger = logging.getLogger(__name__)

TREX_STAGE1_CAMERA_VIEWS: dict[str, dict[str, float]] = {
    # Fixed view angles make stance comparisons repeatable while preserving
    # the existing pelvis-tracking camera.
    "side": {"azimuth": 90.0, "elevation": -8.0, "distance": 3.4},
    "front": {"azimuth": 180.0, "elevation": -8.0, "distance": 3.4},
}

try:
    import numpy as _np
except ImportError:
    _np = None  # type: ignore[assignment]


def eval_policy(
    model,
    eval_env,
    success_keys: list[str],
    n_episodes: int = 30,
):
    """Evaluate a trained policy collecting per-episode metrics.

    Runs *n_episodes* deterministic rollouts and returns five parallel lists:

    * **rewards** -- total reward per episode
    * **lengths** -- step count per episode
    * **fwd_vels** -- mean forward velocity per episode
    * **successes** -- 1.0 if any *success_keys* triggered, else 0.0
    * **distances** -- total distance traveled per episode (cumulative XY path)
    """
    import numpy as _np

    rewards: list[float] = []
    lengths: list[float] = []
    fwd_vels: list[float] = []
    successes: list[float] = []
    distances: list[float] = []

    for _ in range(n_episodes):
        obs = eval_env.reset()
        ep_reward, ep_len = 0.0, 0
        ep_fwd: list[float] = []
        ep_success = 0.0
        ep_distance = 0.0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, infos = eval_env.step(action)
            ep_reward += float(reward[0])
            ep_len += 1
            if "forward_vel" in infos[0]:
                ep_fwd.append(float(infos[0]["forward_vel"]))
            if "distance_traveled" in infos[0]:
                ep_distance = float(infos[0]["distance_traveled"])
            if any(infos[0].get(k) for k in success_keys):
                ep_success = 1.0
            done = bool(dones[0])
        rewards.append(ep_reward)
        lengths.append(float(ep_len))
        fwd_vels.append(float(_np.mean(ep_fwd)) if ep_fwd else 0.0)
        successes.append(ep_success)
        distances.append(ep_distance)

    return rewards, lengths, fwd_vels, successes, distances


def eval_policy_quality(
    model,
    eval_env,
    success_keys: list[str],
    n_episodes: int = 50,
) -> dict:
    """Evaluate a trained policy collecting quality metrics for post-training selection.

    Runs *n_episodes* deterministic rollouts using :class:`LocomotionMetrics`
    and returns aggregated quality metrics including spinning detection
    signals, reward component breakdowns, and heading stability.

    This is designed to be called after training to populate the
    ``metrics.json`` sidecar with columns that enable model selection
    beyond raw reward.

    Returns:
        Dict of aggregated eval metrics (prefixed with ``eval_``).
    """
    from .metrics import LocomotionMetrics, env_dt
    from .stance_diagnostics import STANCE_INFO_KEYS

    episode_reports = []
    dt = env_dt(eval_env)

    for _ in range(n_episodes):
        obs = eval_env.reset()
        metrics = LocomotionMetrics(dt=dt)
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, infos = eval_env.step(action)
            metrics.record_step(infos[0], float(reward[0]))
            done = bool(dones[0])
        episode_reports.append(metrics.compute())

    agg = LocomotionMetrics.aggregate_episodes(episode_reports)

    # Extract the key quality metrics with eval_ prefix
    result: dict = {}

    # Core performance
    for key in (
        "mean_episode_length",
        "mean_total_reward",
    ):
        if key in agg:
            result[f"eval_{key}"] = round(agg[key], 4)

    # Spinning detection
    for key in (
        "mean_mean_pelvis_angular_velocity",
        "mean_max_pelvis_angular_velocity",
        "mean_mean_pelvis_yaw_velocity",
        "mean_max_pelvis_yaw_velocity",
    ):
        if key in agg:
            # Simplify key: mean_mean_pelvis_angular_velocity → eval_mean_pelvis_angular_velocity
            clean_key = key.replace("mean_mean_", "mean_").replace("mean_max_", "max_")
            result[f"eval_{clean_key}"] = round(agg[key], 4)

    # Heading stability
    for key in (
        "mean_mean_heading_alignment",
        "std_mean_heading_alignment",
        "mean_std_heading_alignment",
    ):
        if key in agg:
            clean_key = key.replace("mean_mean_", "mean_").replace("mean_std_", "std_")
            result[f"eval_{clean_key}"] = round(agg[key], 4)

    # Posture quality
    for key in (
        "mean_mean_tilt_angle",
        "mean_mean_pelvis_height",
    ):
        if key in agg:
            clean_key = key.replace("mean_mean_", "mean_")
            result[f"eval_{clean_key}"] = round(agg[key], 4)

    # Distance traveled
    if "mean_distance_traveled" in agg:
        result["eval_distance_traveled"] = round(agg["mean_distance_traveled"], 4)

    # Reporting-only stance diagnostics. ``mean_mean_*`` is the mean of each
    # episode's timestep mean; ``std_mean_*`` captures episode variation.
    for metric in STANCE_INFO_KEYS:
        mean_key = f"mean_mean_{metric}"
        std_key = f"std_mean_{metric}"
        if mean_key in agg:
            result[f"eval_mean_{metric}"] = round(agg[mean_key], 4)
        if std_key in agg:
            result[f"eval_std_{metric}"] = round(agg[std_key], 4)

    # Reward component breakdown (cumulative per episode, averaged across episodes)
    for key, value in agg.items():
        if key.startswith("mean_reward_component_"):
            component = key.replace("mean_reward_component_", "")
            result[f"eval_reward_{component}"] = round(value, 4)

    # Termination reason distribution
    term_counts = agg.get("termination_counts")
    if term_counts:
        total = sum(term_counts.values())
        for reason, count in term_counts.items():
            result[f"eval_term_{reason}_pct"] = round(count / total, 4)

    return result


def record_stage_video(
    model,
    env_class: type,
    env_kwargs: dict,
    stage: int,
    stage_dir,
    species: str = "dino",
    algorithm: str = "ppo",
    seed: int = 42,
    vecnorm_path: str | None = None,
    max_steps: int = 1000,
    label: str | None = None,
    output_dir: "str | Path | None" = None,
    plant_identity: PlantIdentity | None = None,
    allow_legacy_plant: bool = False,
    camera_views: Mapping[str, Mapping[str, float]] | None = None,
    collect_stance_diagnostics: bool = False,
):
    """Record and save a video of the trained policy for a given stage.

    When *vecnorm_path* is provided, observations are normalized using the
    saved ``VecNormalize`` running statistics so the policy sees the same
    input distribution it was trained on.

    When *camera_views* is supplied, one additional synchronized video is
    written per named camera preset. When *collect_stance_diagnostics* is
    true, a per-frame ``*_stance.csv`` is written beside the replay.

    *output_dir* is where the video files land; it defaults to *stage_dir*
    for callers that predate the ``replays/`` subdirectory. *stage_dir* is
    still what identifies the stage in log messages, so the two are separate
    parameters rather than one overloaded path.

    Requires the ``mediapy`` package (``pip install mediapy``).
    """
    try:
        import mediapy
    except ImportError:
        logger.warning("Skipping video for stage %s (mediapy not installed).", stage)
        return

    from .plant_contract import current_plant_identity, validate_environment_plant, validate_model_plant
    from .train_base import _ensure_sb3

    sb3 = _ensure_sb3()
    if plant_identity is None and species != "dino":
        plant_identity = current_plant_identity(species)
    if plant_identity is not None:
        validate_model_plant(
            model,
            plant_identity,
            artifact=f"{species} stage {stage} video model",
            allow_legacy=allow_legacy_plant,
        )
    render_env = env_class(render_mode="rgb_array", **env_kwargs)
    if plant_identity is not None:
        try:
            validate_environment_plant(
                render_env,
                plant_identity,
                artifact=f"{species} stage {stage} video environment",
            )
        except Exception:
            render_env.close()
            raise

    vec_normalize = None
    if vecnorm_path and Path(vecnorm_path).exists():
        dummy_env = sb3["DummyVecEnv"]([lambda: env_class(**env_kwargs)])
        vec_normalize = sb3["VecNormalize"].load(vecnorm_path, dummy_env)
        if plant_identity is not None:
            try:
                validate_model_plant(
                    vec_normalize,
                    plant_identity,
                    artifact=vecnorm_path,
                    allow_legacy=allow_legacy_plant,
                )
            except Exception:
                vec_normalize.close()
                render_env.close()
                raise
        vec_normalize.training = False
        vec_normalize.norm_reward = False

    obs, _ = render_env.reset(seed=replay_seed(seed, stage))
    frames = []
    named_frames: dict[str, list[Any]] = {name: [] for name in (camera_views or {})}
    stance_rows: list[dict[str, float]] = []
    episode_reward = 0.0

    for step_index in range(max_steps):
        if vec_normalize is not None:
            obs_input = vec_normalize.normalize_obs(obs)
        else:
            obs_input = obs
        action, _ = model.predict(obs_input, deterministic=True)
        obs, reward, terminated, truncated, info = render_env.step(action)
        frames.append(render_env.render())
        if collect_stance_diagnostics:
            from .stance_diagnostics import capture_trex_stance_snapshot

            stance_rows.append(capture_trex_stance_snapshot(render_env, info, step_index + 1))
        if camera_views:
            camera = getattr(render_env, "_camera", None)
            if camera is None:
                logger.warning("Skipping named camera views: environment did not expose an RGB camera.")
                camera_views = None
            else:
                original_camera = {name: getattr(camera, name) for name in ("azimuth", "elevation", "distance")}
                try:
                    for view_name, preset in camera_views.items():
                        for attr, value in original_camera.items():
                            setattr(camera, attr, value)
                        for attr in ("azimuth", "elevation", "distance"):
                            if attr in preset:
                                setattr(camera, attr, float(preset[attr]))
                        named_frames[view_name].append(render_env.render())
                finally:
                    for attr, value in original_camera.items():
                        setattr(camera, attr, value)
        episode_reward += reward
        if terminated or truncated:
            break

    render_env.close()
    if vec_normalize is not None:
        vec_normalize.close()

    suffix = f"_{label}" if label else ""
    destination = Path(output_dir) if output_dir is not None else Path(stage_dir)
    destination.mkdir(parents=True, exist_ok=True)
    video_path = str(destination / f"{species}_{algorithm.lower()}_{stage_label(stage)}{suffix}.mp4")
    mediapy.write_video(video_path, frames, fps=50)
    video_stem = Path(video_path).with_suffix("")
    for view_name, view_frames in named_frames.items():
        if not view_frames:
            continue
        view_path = str(video_stem.with_name(f"{video_stem.name}_{view_name}").with_suffix(".mp4"))
        mediapy.write_video(view_path, view_frames, fps=50)
        logger.info("  Saved %s camera replay to: %s", view_name, view_path)
    if collect_stance_diagnostics:
        from .stance_diagnostics import write_stance_diagnostics_csv

        stance_path = video_stem.with_name(f"{video_stem.name}_stance").with_suffix(".csv")
        if write_stance_diagnostics_csv(stance_path, stance_rows) is not None:
            logger.info("  Saved stance diagnostics to: %s", stance_path)
    logger.info("Stage %s video: reward=%.2f | %d frames", stage, episode_reward, len(frames))
    logger.info("  Saved to: %s", video_path)
    return video_path, frames


#: train()'s default single-stage directory is ``{stage_dirname}_{timestamp}``
#: ("02_recovery_20260821_142144") — the timestamp must come off before the
#: component can be recognized as an id-based stage directory.
_RUN_TIMESTAMP_SUFFIX = re.compile(r"_\d{8}_\d{6}$")
#: ``stage{N}`` must match as a whole token: a raw substring scan reads
#: "stage1" inside "stage1b"/"stage10" and misclassifies the checkpoint.
_PATH_TOKEN_SPLIT = re.compile(r"[^A-Za-z0-9]+")


def detect_stage_from_path(model_path: str) -> "int | str":
    """Infer the stage a checkpoint belongs to from its path, any layout.

    Walks path components deepest-first, so the stage directory next to the
    checkpoint always beats an outer folder that merely mentions a stage.
    Each component is checked for the id-based layouts — ``NN_{id}``
    (2026-08-20), the same with train()'s run timestamp appended, and the
    bare id the 20260819 runs used — with the id mapped to its legacy
    number when it has one and passed through as the semantic id
    (``"recovery"``) when it does not.  Historical ``stage{N}`` references
    match as whole tokens only.  Falls back to stage 1, matching the
    historical default.
    """
    from .stage_manifest import KNOWN_STAGE_IDS, LEGACY_STAGE_IDS

    id_to_legacy = {stage_id: number for number, stage_id in LEGACY_STAGE_IDS.items()}
    legacy_tokens = {f"stage{number}": number for number in LEGACY_STAGE_IDS}
    for part in reversed(Path(model_path).parts):
        candidate = _RUN_TIMESTAMP_SUFFIX.sub("", part)
        if len(candidate) > 3 and candidate[:2].isdigit() and candidate[2] == "_":
            candidate = candidate[3:]
        if candidate in KNOWN_STAGE_IDS:
            return id_to_legacy.get(candidate, candidate)
        for token in _PATH_TOKEN_SPLIT.split(part):
            if token in legacy_tokens:
                return legacy_tokens[token]
    return 1


def evaluate(
    species_cfg,
    stage_configs: "dict[int | str, dict[str, Any]]",
    model_path: str,
    n_episodes: int = 30,
    render: bool = True,
    stage: "int | str | None" = None,
    algorithm: str = "ppo",
    allow_legacy_plant: bool = False,
):
    """Evaluate a trained model with full locomotion metrics."""
    from .metrics import LocomotionMetrics
    from .metrics import env_dt as _env_dt
    from .plant_contract import current_plant_identity, validate_environment_plant, validate_model_plant
    from .train_base import _ensure_sb3

    sb3 = _ensure_sb3()
    plant_identity = current_plant_identity(species_cfg.species)

    logger.info("Loading model from: %s", model_path)

    if stage is None:
        stage = detect_stage_from_path(model_path)
        logger.info("Auto-detected stage %s from filename", stage)

    env_kwargs = stage_configs[stage]["env_kwargs"].copy()

    vecnorm_path = model_path.replace(".zip", "_vecnorm.pkl")
    if not vecnorm_path.endswith("_vecnorm.pkl"):
        vecnorm_path = model_path + "_vecnorm.pkl"

    render_mode = "human" if render else None

    def _make_eval_env():
        env = species_cfg.env_class(render_mode=render_mode, **env_kwargs)
        try:
            validate_environment_plant(
                env,
                plant_identity,
                artifact=f"{species_cfg.species} stage {stage} evaluation environment",
            )
        except Exception:
            env.close()
            raise
        return sb3["Monitor"](env)

    vec_env = sb3["DummyVecEnv"]([_make_eval_env])

    if Path(vecnorm_path).exists():
        logger.info("Loading normalization stats from: %s", vecnorm_path)
        vec_env = sb3["VecNormalize"].load(vecnorm_path, vec_env)
        try:
            validate_model_plant(
                vec_env,
                plant_identity,
                artifact=vecnorm_path,
                allow_legacy=allow_legacy_plant,
            )
        except Exception:
            vec_env.close()
            raise
        vec_env.training = False
        vec_env.norm_reward = False
    else:
        logger.warning("No VecNormalize stats found. Results may differ from training.")

    alg_cls = sb3["SAC"] if algorithm == "sac" else sb3["PPO"]
    try:
        model = alg_cls.load(model_path, env=vec_env)
        validate_model_plant(
            model,
            plant_identity,
            artifact=model_path,
            allow_legacy=allow_legacy_plant,
        )
    except Exception:
        vec_env.close()
        raise

    logger.info(
        "Evaluating for %d episodes (stage %s: %s)...",
        n_episodes,
        stage,
        stage_configs[stage]["name"],
    )

    episode_reports = []
    eval_dt = _env_dt(vec_env)

    for ep in range(n_episodes):
        obs = vec_env.reset()
        metrics = LocomotionMetrics(dt=eval_dt)
        total_reward = 0.0
        step = 0

        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = vec_env.step(action)
            step_reward = float(rewards[0])
            total_reward += step_reward
            step += 1
            metrics.record_step(infos[0], step_reward)

            if dones[0]:
                break

        report = metrics.compute()
        episode_reports.append(report)

        term_reason = report.get("termination_reason", "truncated")
        logger.info(
            "  Episode %d: reward=%.2f, length=%d, fwd_vel=%.3f m/s, tilt=%.2f rad, ended=%s",
            ep + 1,
            total_reward,
            step,
            report.get("mean_forward_velocity", 0.0),
            report.get("mean_tilt_angle", 0.0),
            term_reason,
        )

    vec_env.close()

    agg = LocomotionMetrics.aggregate_episodes(episode_reports)
    _log_eval_results(species_cfg, agg, n_episodes)


def _log_eval_results(
    species_cfg,
    agg: dict,
    n_episodes: int,
):
    """Log formatted evaluation results."""
    logger.info("=" * 60)
    logger.info("Evaluation Results (%d episodes)", n_episodes)
    logger.info("=" * 60)

    logger.info("--- Core Performance ---")
    logger.info(
        "  Reward:       %.2f +/- %.2f",
        agg.get("mean_total_reward", 0),
        agg.get("std_total_reward", 0),
    )
    logger.info(
        "  Ep Length:    %.1f +/- %.1f",
        agg.get("mean_episode_length", 0),
        agg.get("std_episode_length", 0),
    )

    logger.info("--- Velocity ---")
    logger.info(
        "  Forward vel:  %.3f +/- %.3f m/s",
        agg.get("mean_mean_forward_velocity", 0),
        agg.get("std_mean_forward_velocity", 0),
    )
    logger.info("  Max fwd vel:  %.3f m/s", agg.get("mean_max_forward_velocity", 0))
    logger.info("  Consistency:  %.3f", agg.get("mean_velocity_consistency", 0))
    logger.info(
        "  Distance:     %.3f +/- %.3f m",
        agg.get("mean_total_distance", 0),
        agg.get("std_total_distance", 0),
    )
    if "mean_distance_traveled" in agg:
        logger.info(
            "  Path length:  %.3f +/- %.3f m",
            agg.get("mean_distance_traveled", 0),
            agg.get("std_distance_traveled", 0),
        )

    logger.info("--- Gait Quality ---")
    logger.info("  Symmetry:     %.3f", agg.get("mean_gait_symmetry", 0))
    logger.info("  Stride freq:  %.3f Hz", agg.get("mean_stride_frequency", 0))
    logger.info("  Cost of transport: %.4f", agg.get("mean_cost_of_transport", 0))

    logger.info("--- Balance ---")
    logger.info(
        "  %s: %.3f +/- %.3f m",
        species_cfg.height_label,
        agg.get("mean_mean_pelvis_height", 0),
        agg.get("std_mean_pelvis_height", 0),
    )
    logger.info(
        "  Mean tilt:     %.3f +/- %.3f rad",
        agg.get("mean_mean_tilt_angle", 0),
        agg.get("std_mean_tilt_angle", 0),
    )
    logger.info("  Max tilt:      %.3f rad", agg.get("mean_max_tilt_angle", 0))

    if "mean_initial_prey_distance" in agg:
        logger.info("--- %s ---", species_cfg.stage3_section_label)
        logger.info(
            "  Initial dist:   %.3f m",
            agg.get("mean_initial_prey_distance", 0),
        )
        logger.info("  Final dist:     %.3f m", agg.get("mean_final_prey_distance", 0))
        logger.info("  Min dist:       %.3f m", agg.get("mean_min_prey_distance", 0))
        if "mean_time_to_target" in agg:
            # Mean over episodes that reached the target (never-reached
            # episodes report NaN and are excluded by aggregate_episodes).
            logger.info(
                "  Time to target: %.3f s (reached in %.0f%% of episodes)",
                agg["mean_time_to_target"],
                100.0 * agg.get("mean_target_reached", 0.0),
            )
        else:
            logger.info("  Time to target: never reached")
    if "mean_heading_alignment" in agg:
        logger.info("  Heading align:  %.3f", agg.get("mean_heading_alignment", 0))
    if "mean_success_rate" in agg:
        logger.info(
            "  Success rate:   %.1f%%",
            100.0 * agg.get("mean_success_rate", 0),
        )

    term_counts = agg.get("termination_counts")
    if term_counts:
        logger.info("--- Termination Reasons ---")
        for reason, count in sorted(term_counts.items(), key=lambda x: -x[1]):
            pct = 100.0 * count / n_episodes
            logger.info("  %-20s %d (%.0f%%)", reason, count, pct)

    logger.info("=" * 60)
