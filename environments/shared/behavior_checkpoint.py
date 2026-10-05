"""Prepare a canonical PPO walker for direction and terrain behavior training.

This module requires the SB3 training extra. Preparation preserves the walker's
function on zero commands, clears only the reserved command connections and their
optimizer moments, and carries the observation statistics with the command slice
reseeded to mean 0 / variance 1 (invariant 8, decision D-D3). Behavior artifacts carry
the canonical stamps: the plant identity, the recipe env's task fingerprint and, as its
task lineage, the preparation report. That fingerprint's stage is never a stage id
(preparation refuses one), so no canonical node resumes or warm-starts from a behavior
artifact (decision D-D9).
"""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.running_mean_std import RunningMeanStd
from stable_baselines3.common.utils import ConstantSchedule
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecNormalize, unwrap_vec_normalize

from environments.shared.command_frame import COMMAND_WIDTH, reseed_command_slice
from environments.shared.plant_contract import (
    MODEL_IDENTITY_ATTRIBUTE,
    current_plant_identity,
    validate_model_plant,
    validate_recorded_identity,
)
from environments.shared.policy_loading import (
    ACTION_DELTA_ATOL,
    PolicyLoadError,
    _checkpoint_algorithm,
    assert_command_blind,
    load_sb3_model,
    neutralize_command_columns,
)
from environments.shared.result_bundle import sha256_file
from environments.shared.species_names import resolve_species_id, species_display_names
from environments.shared.stage_manifest import STAGE_ID_PATTERN
from environments.shared.task_fingerprint import (
    MODEL_TASK_ATTRIBUTE,
    MODEL_TASK_LINEAGE_ATTRIBUTE,
    read_checkpoint_attribute,
)

PREPARATION_SCHEMA = "mesozoic.behavior-preparation/v2"
#: The two first layers ``policy_loading.neutralize_command_columns`` zeroes in a PPO walker.
COMMAND_LAYERS = ("mlp_extractor.policy_net.0", "mlp_extractor.value_net.0")
ACTION_EQUIVALENCE_ATOL = ACTION_DELTA_ATOL


class BehaviorCheckpointError(ValueError):
    """A checkpoint cannot safely initialize or resume this behavior experiment."""


def _species(species: str | None, task: Mapping[str, Any]) -> str:
    """Resolve the requested plant, retaining old T-Rex API calls without metadata."""
    recorded = task.get("species")
    resolved = resolve_species_id(species or recorded or "trex")
    if resolved not in species_display_names(backend="stable-baselines3"):
        raise BehaviorCheckpointError(f"Species {resolved!r} has no supported SB3 behavior interface")
    if recorded is not None and recorded != resolved:
        raise BehaviorCheckpointError("Behavior task fingerprint and requested species disagree")
    return resolved


def _paths(model_path: str | Path, vecnorm_path: str | Path) -> tuple[Path, Path]:
    model_path, vecnorm_path = Path(model_path), Path(vecnorm_path)
    if not model_path.is_file() or not vecnorm_path.is_file():
        raise BehaviorCheckpointError("Both the checkpoint and its matching normalization file must exist")
    return model_path, vecnorm_path


def _identity(value: Mapping[str, Any] | None) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise BehaviorCheckpointError("task_fingerprint must be a JSON object")
    try:
        return dict(json.loads(json.dumps(dict(value), allow_nan=False, sort_keys=True)))
    except (TypeError, ValueError) as exc:
        raise BehaviorCheckpointError("task_fingerprint must contain finite JSON values") from exc


def _recorded_task(model_path: Path) -> dict[str, Any]:
    """The task fingerprint a behavior checkpoint records; an older marker is refused by name (D-D9)."""
    recorded = read_checkpoint_attribute(model_path, MODEL_TASK_ATTRIBUTE)
    if not isinstance(recorded, Mapping) or "task_sha256" not in recorded:
        raise BehaviorCheckpointError(
            "The checkpoint records no task fingerprint: a behavior artifact from before the task-fingerprint "
            "identity is evaluation-only (decision D-D9); evaluate it at the commit that trained it "
            "(run.json's git_commit)"
        )
    return dict(recorded)


def _venv(env: Any, observation_dim: int, action_dim: int) -> VecEnv:
    if isinstance(env, VecEnv) and unwrap_vec_normalize(env) is not None:
        raise BehaviorCheckpointError("Pass an unnormalized environment; the saved sidecar supplies normalization")
    if env.observation_space.shape != (observation_dim,) or env.action_space.shape != (action_dim,):
        raise BehaviorCheckpointError("Behavior environment must preserve the canonical observation/action dimensions")
    return env if isinstance(env, VecEnv) else DummyVecEnv([lambda: env])


def _load_ppo(model_path: Path, env: VecNormalize, learning_rate: float) -> PPO:
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise BehaviorCheckpointError("learning_rate must be finite and positive")
    if _checkpoint_algorithm(str(model_path)) is not PPO:
        raise BehaviorCheckpointError("Behavior preparation supports PPO checkpoints only")
    # Avoid executing cloudpickled Python 3.13 schedule bytecode on Python 3.12.
    # SB3 custom_objects replaces these values BEFORE deserialization. Only
    # training settings are replaced; network and optimizer tensors are loaded.
    schedule = ConstantSchedule(float(learning_rate))
    model = cast(
        PPO,
        load_sb3_model(
            str(model_path),
            algorithm=PPO,
            env=env,
            device="cpu",
            custom_objects={
                "learning_rate": float(learning_rate),
                "lr_schedule": schedule,
                "clip_range": ConstantSchedule(0.2),
                "clip_range_vf": None,
            },
        ),
    )
    for group in model.policy.optimizer.param_groups:
        group["lr"] = float(learning_rate)
    return model


def _normalizer(path: Path, venv: VecEnv) -> VecNormalize:
    try:
        normalizer = VecNormalize.load(str(path), venv)
    except AttributeError as exc:
        if "BehaviorVecNormalize" not in str(exc):
            raise
        raise BehaviorCheckpointError(
            "This normalization file pickles the deleted command-passthrough class BehaviorVecNormalize: its policy "
            "saw other command inputs and is not a continuation (decision D-D3); evaluate it at the commit that "
            "trained it (run.json's git_commit)"
        ) from exc
    if not isinstance(normalizer.obs_rms, RunningMeanStd):
        raise BehaviorCheckpointError("Only flat observation normalization is supported")
    if not normalizer.norm_obs:
        raise BehaviorCheckpointError("The walker must carry active observation normalization")
    if (
        not np.all(np.isfinite(normalizer.obs_rms.mean))
        or not np.all(np.isfinite(normalizer.obs_rms.var))
        or np.any(normalizer.obs_rms.var < 0)
        or not np.isfinite(normalizer.obs_rms.count)
        or normalizer.obs_rms.count <= 0
    ):
        raise BehaviorCheckpointError("The normalization statistics are invalid")
    return normalizer


def prepare_behavior_checkpoint(
    model_path: str | Path,
    vecnorm_path: str | Path,
    env: Any,
    *,
    learning_rate: float = 5e-5,
    task_fingerprint: Mapping[str, Any],
    species: str | None = None,
) -> tuple[PPO, VecNormalize, dict[str, Any]]:
    """Initialize a separate behavior task from a canonical current-interface PPO walker.

    ``env`` is a Gym environment or unnormalized SB3 VecEnv. Nothing is saved or
    stepped here. ``task_fingerprint`` is the recipe env's, required with its
    ``task_sha256`` (an artifact stamped without one could never be resumed or
    adapted), for exact task checks on :func:`load_behavior_checkpoint`. The returned plain
    ``VecNormalize`` carries the walker's statistics with the three command inputs
    reseeded to mean 0 / variance 1 (count kept), so commands enter the policy at
    O(1); every statistic, the commands' included, updates during training.
    """
    model_path, vecnorm_path = _paths(model_path, vecnorm_path)
    behavior = _identity(task_fingerprint)
    species = _species(species, behavior)
    if "task_sha256" not in behavior:
        raise BehaviorCheckpointError(
            "Preparation needs the recipe env's task fingerprint (env.task_fingerprint) with its task_sha256: "
            "an artifact stamped without one could never be resumed or adapted"
        )
    stage = behavior.get("stage")
    if (isinstance(stage, int) and not isinstance(stage, bool)) or (
        isinstance(stage, str) and STAGE_ID_PATTERN.match(stage) is not None
    ):
        raise BehaviorCheckpointError(
            "Preparation stamps the recipe env's task fingerprint, whose stage is never a stage id: stage "
            f"{stage!r} would let a manifest node resume or warm-start from the artifact (decision D-D9)"
        )
    current = current_plant_identity(species)
    raw_plant = read_checkpoint_attribute(model_path, MODEL_IDENTITY_ATTRIBUTE)
    validate_recorded_identity(raw_plant, current, artifact="parent walker checkpoint")
    task = read_checkpoint_attribute(model_path, MODEL_TASK_ATTRIBUTE)
    if not isinstance(task, Mapping) or task.get("species") != species or task.get("stage") not in (2, "locomotion"):
        raise BehaviorCheckpointError(f"The source must record the {species} locomotion task")
    venv = _venv(env, current.observation_dim, current.action_dim)
    normalizer = _normalizer(vecnorm_path, venv)
    assert isinstance(normalizer.obs_rms, RunningMeanStd)
    validate_model_plant(normalizer, current, artifact="parent walker normalization")
    if not np.array_equal(normalizer.obs_rms.mean[-COMMAND_WIDTH:], np.zeros(COMMAND_WIDTH)):
        raise BehaviorCheckpointError("The source command channels were not constant zero")
    model = _load_ppo(model_path, normalizer, learning_rate)
    validate_model_plant(model, current, artifact="loaded parent walker")

    # The command-column primitive (policy_loading): seeded synthetic raw
    # observations exercise the complete normalization -> network path without
    # consuming reset draws from the training environment; the parent's actions
    # and values on zero commands are taken before the command columns are
    # zeroed and the slice reseeded, and must be unchanged on live commands
    # afterwards.
    def prepare() -> list[str]:
        changed = neutralize_command_columns(model, observation_dim=current.observation_dim)
        reseed_command_slice(normalizer.obs_rms)
        return changed

    try:
        probe = assert_command_blind(model, normalizer, atol=ACTION_EQUIVALENCE_ATOL, prepare=prepare)
    except PolicyLoadError as exc:
        raise BehaviorCheckpointError(str(exc)) from exc

    report = {
        "schema": PREPARATION_SCHEMA,
        "mode": "initialize_next_stage",
        "parent_task_sha256": task.get("task_sha256"),
        "child_task_sha256": behavior.get("task_sha256"),
        "species": species,
        "task_fingerprint": behavior,
        "parent_plant_identity": current.to_dict(),
        "parent_checkpoint_sha256": sha256_file(model_path),
        "parent_normalization_sha256": sha256_file(vecnorm_path),
        "canonical_certification": False,
        "parent_task_fingerprint": dict(task),
        "parent_training_timesteps": model.num_timesteps,
        "learning_rate": float(learning_rate),
        "clip_range": 0.2,
        "clip_range_vf": None,
        "zeroed_command_parameters": probe["prepared"],
        "optimizer_command_columns_zeroed": True,
        "command_normalization": "reseeded to mean 0 / variance 1, count kept; statistics keep updating",
        "command_stats_reseeded": True,
        "noncommand_stats": "preserved at preparation; adapt during training",
        "equivalence_probe_seed": probe["equivalence_probe_seed"],
        "equivalence_probe_observations": probe["equivalence_probe_observations"],
        "max_action_delta": probe["max_action_delta"],
        "max_value_delta": probe["max_value_delta"],
    }
    for artifact in (model, normalizer):
        setattr(artifact, MODEL_IDENTITY_ATTRIBUTE, current.to_dict())
        setattr(artifact, MODEL_TASK_ATTRIBUTE, copy.deepcopy(behavior))
        setattr(artifact, MODEL_TASK_LINEAGE_ATTRIBUTE, copy.deepcopy(report))
    normalizer.training = True
    normalizer.norm_reward = False
    model._last_obs = None
    model._last_original_obs = None
    return model, normalizer, report


def load_behavior_checkpoint(
    model_path: str | Path,
    vecnorm_path: str | Path,
    env: Any,
    *,
    task_fingerprint: Mapping[str, Any],
    learning_rate: float = 5e-5,
    species: str | None = None,
) -> tuple[PPO, VecNormalize, dict[str, Any]]:
    """Resume an exact behavior task without resetting learned command connections.

    This restores weights, optimizer tensors and normalization statistics.
    The caller controls remaining-budget arithmetic and schedule continuation;
    the explicit constant learning rate is installed safely before deserialization.
    """
    model_path, vecnorm_path = _paths(model_path, vecnorm_path)
    recorded, requested = _recorded_task(model_path), _identity(task_fingerprint)
    if recorded != requested:
        differing = sorted(key for key in recorded.keys() | requested.keys() if recorded.get(key) != requested.get(key))
        raise BehaviorCheckpointError(f"Behavior task fingerprint differs from the saved task in {differing}")
    species = _species(species, recorded)
    current = current_plant_identity(species)
    plant = read_checkpoint_attribute(model_path, MODEL_IDENTITY_ATTRIBUTE)
    validate_recorded_identity(plant, current, artifact="behavior checkpoint plant")
    venv = _venv(env, current.observation_dim, current.action_dim)
    normalizer = _normalizer(vecnorm_path, venv)
    if (
        getattr(normalizer, MODEL_IDENTITY_ATTRIBUTE, None) != plant
        or getattr(normalizer, MODEL_TASK_ATTRIBUTE, None) != recorded
    ):
        raise BehaviorCheckpointError("Behavior model and normalization identities disagree")
    model = _load_ppo(model_path, normalizer, learning_rate)
    if getattr(model, MODEL_TASK_ATTRIBUTE, None) != recorded:
        raise BehaviorCheckpointError("Behavior task fingerprint differs from its artifact stamp")
    normalizer.training = True
    normalizer.norm_reward = False
    report = copy.deepcopy(getattr(model, MODEL_TASK_LINEAGE_ATTRIBUTE, {}))
    report["resume_checkpoint_sha256"] = sha256_file(model_path)
    report["resume_normalization_sha256"] = sha256_file(vecnorm_path)
    report["resume_learning_rate"] = float(learning_rate)
    return model, normalizer, report


# A transition changes a task's requested behavior, not the meaning/order of
# policy inputs, its actuator dynamics, or the implementation interpreting them
# (the fingerprint's stage, command schema and adapter name that implementation).
_TRANSITION_COMMAND_SETTINGS = frozenset(
    {
        "cruise_speed",
        "speed_range",
        "stop_probability",
        "turn_increment_max",
        "straight_probability",
        "switch_interval_s",
        "switch_jitter_s",
    }
)
_TRANSITION_REWARD_SETTINGS = frozenset(
    {
        "fall_penalty",
        "forward_vel_max",
        "speed_penalty_threshold",
        "head_clearance_target",
        "head_clearance_tolerance",
        "height_target_tolerance",
        "neck_posture_tolerance",
        "foot_contact_gate",
        "foot_contact_saturation_force",
        "foot_load_balance_min_support_force",
        "foot_load_balance_airborne_penalty",
        "support_conditioned_alive_fraction",
        "leg_home_pose_tolerance",
        "leg_home_pose_broad_fraction",
        "leg_home_pose_broad_scale",
        "tail_home_pose_tolerance",
        "action_saturation_threshold",
        "idle_velocity_threshold",
    }
)
# command_config is the command section's config again, checked there.
_TRANSITION_ENV_SETTINGS = frozenset(
    {"max_episode_steps", "command_config", "terrain", "terrain_sampler", "tracking_weight", "course_distance"}
)


def _validate_sampler_config(config: Any) -> None:
    """A transition may add, remove or reweight the sampling layer, never carry an invalid one."""
    if config is None:
        return
    from environments.shared.terrain_sampling import TerrainSamplerConfig

    try:
        TerrainSamplerConfig(**config)
    except (TypeError, ValueError) as exc:
        raise BehaviorCheckpointError("Invalid terrain sampler configuration") from exc


def _validate_behavior_transition(previous: Mapping[str, Any], requested: Mapping[str, Any]) -> None:
    for fingerprint in (previous, requested):
        command = fingerprint.get("command")
        if not isinstance(fingerprint.get("env"), Mapping) or not isinstance(command, Mapping):
            raise BehaviorCheckpointError("Only supported command/terrain tasks allow behavior transitions")
        if not isinstance(command.get("config"), Mapping):
            raise BehaviorCheckpointError("Behavior task fingerprint command config must be an object")
        _validate_sampler_config(fingerprint["env"].get("terrain_sampler"))
    differences = [
        name
        for name in ("schema", "species", "stage", "backend", "plant", "perturbation")
        if previous.get(name) != requested.get(name)
    ]
    old_command, new_command = previous["command"], requested["command"]
    for name in old_command.keys() | new_command.keys():
        if name != "config" and old_command.get(name) != new_command.get(name):
            differences.append(f"command.{name}")
    old_config, new_config = old_command["config"], new_command["config"]
    for name in old_config.keys() | new_config.keys():
        if name not in _TRANSITION_COMMAND_SETTINGS and old_config.get(name) != new_config.get(name):
            differences.append(f"command.config.{name}")
    old_env, new_env = previous["env"], requested["env"]
    for name in old_env.keys() | new_env.keys():
        reward_setting = name.endswith(("_weight", "_bonus")) or name in _TRANSITION_REWARD_SETTINGS
        if not reward_setting and name not in _TRANSITION_ENV_SETTINGS and old_env.get(name) != new_env.get(name):
            differences.append(f"env.{name}")
    if differences:
        raise BehaviorCheckpointError("Incompatible behavior transition fields: " + ", ".join(sorted(differences)))


def adapt_behavior_checkpoint(
    model_path: str | Path,
    vecnorm_path: str | Path,
    env: Any,
    *,
    task_fingerprint: Mapping[str, Any],
    learning_rate: float = 5e-5,
    species: str | None = None,
) -> tuple[PPO, VecNormalize, dict[str, Any]]:
    """Warm-start another compatible behavior stage without erasing command learning.

    Terrain, balanced family sampling, command sampling/switch schedules, reward
    settings and horizon/course length may change. Input scales and adapters,
    the implementation versions, parent plant and all other environment mechanics
    must match. Every transition records immediate parent hashes and the previous
    task fingerprint; it never promotes the result into canonical certification.
    ``task_fingerprint`` is the recipe env's, required with its ``task_sha256``
    as for preparation: the adapted artifacts are stamped with it.
    """
    requested = _identity(task_fingerprint)
    if "task_sha256" not in requested:
        raise BehaviorCheckpointError(
            "Adaptation needs the recipe env's task fingerprint (env.task_fingerprint) with its task_sha256: "
            "an artifact stamped without one could never be resumed or adapted"
        )
    model_path, vecnorm_path = _paths(model_path, vecnorm_path)
    previous = _recorded_task(model_path)
    species = _species(species, requested)
    _validate_behavior_transition(previous, requested)
    model, normalizer, report = load_behavior_checkpoint(
        model_path,
        vecnorm_path,
        env,
        task_fingerprint=previous,
        learning_rate=learning_rate,
        species=species,
    )
    transition = {
        "schema": "mesozoic.behavior-transition/v2",
        "parent_checkpoint_sha256": sha256_file(model_path),
        "parent_normalization_sha256": sha256_file(vecnorm_path),
        "parent_training_timesteps": model.num_timesteps,
        "previous_task_fingerprint": copy.deepcopy(previous),
        "task_fingerprint": copy.deepcopy(requested),
        "learning_rate": float(learning_rate),
        "command_weights_preserved": True,
        "optimizer_tensors_preserved": True,
        "normalization_statistics_preserved": True,
    }
    # The lineage names the task the artifacts now record; the walker parent stays
    # its parent and each step from it is a transition.
    report["child_task_sha256"] = requested.get("task_sha256")
    report["task_fingerprint"] = copy.deepcopy(requested)
    report.setdefault("transitions", []).append(transition)
    for artifact in (model, normalizer):
        setattr(artifact, MODEL_TASK_ATTRIBUTE, copy.deepcopy(requested))
        setattr(artifact, MODEL_TASK_LINEAGE_ATTRIBUTE, copy.deepcopy(report))
    return model, normalizer, report
