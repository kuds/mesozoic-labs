"""Prepare a canonical T-Rex PPO walker for an explicitly separate behavior experiment.

This module requires the SB3 training extra. Preparation preserves the walker's
function on zero commands, clears only the reserved command connections and their
optimizer moments, and carries the observation statistics. Behavior artifacts are
marked with a schema the canonical certification pipeline deliberately refuses.
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
from environments.shared.policy_loading import _checkpoint_algorithm
from environments.shared.result_bundle import sha256_file
from environments.shared.task_fingerprint import MODEL_TASK_ATTRIBUTE, read_checkpoint_attribute

BEHAVIOR_IDENTITY_SCHEMA = "mesozoic.behavior-artifact/v1"
PREPARATION_ATTRIBUTE = "mesozoic_behavior_preparation"
PREPARATION_SCHEMA = "mesozoic.behavior-preparation/v1"
COMMAND_LAYERS = ("mlp_extractor.policy_net.0", "mlp_extractor.value_net.0")
ACTION_EQUIVALENCE_ATOL = 1e-6


class BehaviorCheckpointError(ValueError):
    """A checkpoint cannot safely initialize or resume this behavior experiment."""


class BehaviorVecNormalize(VecNormalize):
    """Normalize proprioception while passing the three pre-scaled commands unchanged.

    Running statistics still update for all inputs, but command statistics and
    ``clip_obs`` never affect command values. This survives ``save``/``load``;
    the wrapper class is part of the saved normalization artifact.
    """

    def _normalize_obs(self, obs: np.ndarray, obs_rms: RunningMeanStd) -> np.ndarray:
        normalized = super()._normalize_obs(obs, obs_rms)
        normalized[..., -COMMAND_WIDTH:] = obs[..., -COMMAND_WIDTH:]
        return np.asarray(normalized)

    def _unnormalize_obs(self, obs: np.ndarray, obs_rms: RunningMeanStd) -> np.ndarray:
        restored = super()._unnormalize_obs(obs, obs_rms)
        restored[..., -COMMAND_WIDTH:] = obs[..., -COMMAND_WIDTH:]
        return np.asarray(restored)


def _paths(model_path: str | Path, vecnorm_path: str | Path) -> tuple[Path, Path]:
    model_path, vecnorm_path = Path(model_path), Path(vecnorm_path)
    if not model_path.is_file() or not vecnorm_path.is_file():
        raise BehaviorCheckpointError("Both the checkpoint and its matching normalization file must exist")
    return model_path, vecnorm_path


def _identity(value: Mapping[str, Any] | None) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise BehaviorCheckpointError("behavior_identity must be a JSON object")
    try:
        return dict(json.loads(json.dumps(dict(value), allow_nan=False, sort_keys=True)))
    except (TypeError, ValueError) as exc:
        raise BehaviorCheckpointError("behavior_identity must contain finite JSON values") from exc


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
    model = PPO.load(
        str(model_path),
        env=env,
        device="cpu",
        custom_objects={
            "learning_rate": float(learning_rate),
            "lr_schedule": schedule,
            "clip_range": ConstantSchedule(0.2),
            "clip_range_vf": None,
        },
    )
    for group in model.policy.optimizer.param_groups:
        group["lr"] = float(learning_rate)
    return model


def _normalizer(path: Path, venv: VecEnv) -> VecNormalize:
    normalizer = VecNormalize.load(str(path), venv)
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


def _zero_command_connections(model: PPO, observation_dim: int) -> list[str]:
    import torch

    changed = []
    for name in COMMAND_LAYERS:
        layer = model.policy.get_submodule(name)
        if not isinstance(layer, torch.nn.Linear) or layer.in_features != observation_dim:
            raise BehaviorCheckpointError(f"Unsupported policy input layer: {name}")
        with torch.no_grad():
            layer.weight[:, -COMMAND_WIDTH:].zero_()
            for key, value in model.policy.optimizer.state.get(layer.weight, {}).items():
                if not torch.is_tensor(value) or value.ndim == 0:
                    continue
                if value.shape != layer.weight.shape:
                    raise BehaviorCheckpointError(f"Unsupported optimizer tensor {name}.{key}: {tuple(value.shape)}")
                value[:, -COMMAND_WIDTH:].zero_()
        changed.append(f"{name}.weight")
    return changed


def prepare_behavior_checkpoint(
    model_path: str | Path,
    vecnorm_path: str | Path,
    env: Any,
    *,
    learning_rate: float = 5e-5,
    behavior_identity: Mapping[str, Any] | None = None,
) -> tuple[PPO, BehaviorVecNormalize, dict[str, Any]]:
    """Initialize a separate behavior task from a canonical current-interface PPO walker.

    ``env`` is a Gym environment or unnormalized SB3 VecEnv. Nothing is saved or
    stepped here. Supply the full behavior configuration/source identity for
    exact task checks on :func:`load_behavior_checkpoint`. The returned wrapper
    updates proprioceptive statistics during training; the three command inputs
    retain their pre-scaled values regardless of statistics or clipping.
    """
    import torch

    model_path, vecnorm_path = _paths(model_path, vecnorm_path)
    current = current_plant_identity("trex")
    if current.policy_interface_revision != 13 or current.observation_dim != 64:
        raise BehaviorCheckpointError("This preparation supports only the reviewed T-Rex r13, 64-input interface")
    raw_plant = read_checkpoint_attribute(model_path, MODEL_IDENTITY_ATTRIBUTE)
    validate_recorded_identity(raw_plant, current, artifact="parent walker checkpoint")
    task = read_checkpoint_attribute(model_path, MODEL_TASK_ATTRIBUTE)
    if not isinstance(task, Mapping) or task.get("species") != "trex" or task.get("stage") not in (2, "locomotion"):
        raise BehaviorCheckpointError("The source must record the T-Rex locomotion task")
    behavior = _identity(behavior_identity)
    venv = _venv(env, current.observation_dim, current.action_dim)
    normalizer = _normalizer(vecnorm_path, venv)
    assert isinstance(normalizer.obs_rms, RunningMeanStd)
    validate_model_plant(normalizer, current, artifact="parent walker normalization")
    if not np.array_equal(normalizer.obs_rms.mean[-COMMAND_WIDTH:], np.zeros(COMMAND_WIDTH)):
        raise BehaviorCheckpointError("The source command channels were not constant zero")
    model = _load_ppo(model_path, normalizer, learning_rate)
    validate_model_plant(model, current, artifact="loaded parent walker")

    # Seeded synthetic raw observations exercise the complete normalization ->
    # network path without consuming reset draws from the training environment.
    rng = np.random.default_rng(3042)
    raw = rng.normal(size=(64, current.observation_dim))
    raw = normalizer.obs_rms.mean + raw * np.sqrt(normalizer.obs_rms.var + normalizer.epsilon)
    raw[:, -COMMAND_WIDTH:] = 0.0
    parent_inputs = normalizer.normalize_obs(raw)
    parent_actions = model.predict(parent_inputs, deterministic=True)[0]
    with torch.no_grad():
        parent_values = model.policy.predict_values(torch.as_tensor(parent_inputs, device=model.device)).cpu().numpy()
    changed = _zero_command_connections(model, current.observation_dim)
    # Exact same saved state, now with an explicit command passthrough class.
    normalizer.__class__ = BehaviorVecNormalize
    normalizer = cast(BehaviorVecNormalize, normalizer)
    reseed_command_slice(normalizer.obs_rms)
    commands = rng.uniform(-1.0, 1.0, size=(len(raw), COMMAND_WIDTH))
    raw[:, -COMMAND_WIDTH:] = commands
    prepared_inputs = normalizer.normalize_obs(raw)
    prepared_actions = model.predict(prepared_inputs, deterministic=True)[0]
    with torch.no_grad():
        prepared_values = (
            model.policy.predict_values(torch.as_tensor(prepared_inputs, device=model.device)).cpu().numpy()
        )
    action_delta = float(np.max(np.abs(prepared_actions - parent_actions)))
    value_delta = float(np.max(np.abs(prepared_values - parent_values)))
    if not np.isfinite(action_delta) or action_delta > ACTION_EQUIVALENCE_ATOL:
        raise BehaviorCheckpointError(f"Command preparation changed parent actions: max delta {action_delta}")
    if not np.isfinite(value_delta) or value_delta > ACTION_EQUIVALENCE_ATOL:
        raise BehaviorCheckpointError(f"Command preparation changed parent values: max delta {value_delta}")

    marker = {
        "schema": BEHAVIOR_IDENTITY_SCHEMA,
        "species": "trex",
        "behavior_identity": behavior,
        "parent_plant_identity": current.to_dict(),
        "parent_checkpoint_sha256": sha256_file(model_path),
        "parent_normalization_sha256": sha256_file(vecnorm_path),
        "canonical_certification": False,
    }
    report = {
        **marker,
        "schema": PREPARATION_SCHEMA,
        "parent_task_fingerprint": dict(task),
        "parent_training_timesteps": model.num_timesteps,
        "learning_rate": float(learning_rate),
        "clip_range": 0.2,
        "clip_range_vf": None,
        "zeroed_command_parameters": changed,
        "optimizer_command_columns_zeroed": True,
        "command_normalization": "passthrough; pre-scaled inputs ignore running statistics and clip_obs",
        "command_stats_reseeded": True,
        "noncommand_stats": "preserved at preparation; adapt during training",
        "equivalence_probe_seed": 3042,
        "equivalence_probe_observations": len(raw),
        "max_action_delta": action_delta,
        "max_value_delta": value_delta,
    }
    for artifact in (model, normalizer):
        setattr(artifact, MODEL_IDENTITY_ATTRIBUTE, copy.deepcopy(marker))
        setattr(artifact, MODEL_TASK_ATTRIBUTE, copy.deepcopy(marker))
        setattr(artifact, PREPARATION_ATTRIBUTE, copy.deepcopy(report))
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
    behavior_identity: Mapping[str, Any],
    learning_rate: float = 5e-5,
) -> tuple[PPO, BehaviorVecNormalize, dict[str, Any]]:
    """Resume an exact behavior task without resetting learned command connections.

    This restores weights, optimizer tensors and normalization statistics.
    The caller controls remaining-budget arithmetic and schedule continuation;
    the explicit constant learning rate is installed safely before deserialization.
    """
    model_path, vecnorm_path = _paths(model_path, vecnorm_path)
    marker = read_checkpoint_attribute(model_path, MODEL_IDENTITY_ATTRIBUTE)
    expected_behavior = _identity(behavior_identity)
    if not isinstance(marker, Mapping) or marker.get("schema") != BEHAVIOR_IDENTITY_SCHEMA:
        raise BehaviorCheckpointError("The checkpoint is not an explicitly marked behavior artifact")
    if marker.get("behavior_identity") != expected_behavior or marker.get("canonical_certification") is not False:
        raise BehaviorCheckpointError("Behavior configuration/source identity differs from the saved task")
    current = current_plant_identity("trex")
    validate_recorded_identity(marker.get("parent_plant_identity"), current, artifact="behavior parent plant")
    venv = _venv(env, current.observation_dim, current.action_dim)
    normalizer = _normalizer(vecnorm_path, venv)
    if not isinstance(normalizer, BehaviorVecNormalize):
        raise BehaviorCheckpointError("Behavior resume requires the saved command-passthrough normalizer")
    if getattr(normalizer, MODEL_IDENTITY_ATTRIBUTE, None) != marker:
        raise BehaviorCheckpointError("Behavior model and normalization identities disagree")
    model = _load_ppo(model_path, normalizer, learning_rate)
    if getattr(model, MODEL_TASK_ATTRIBUTE, None) != marker:
        raise BehaviorCheckpointError("Behavior task marker differs from its artifact identity")
    normalizer.training = True
    normalizer.norm_reward = False
    report = copy.deepcopy(getattr(model, PREPARATION_ATTRIBUTE, {}))
    report["resume_checkpoint_sha256"] = sha256_file(model_path)
    report["resume_normalization_sha256"] = sha256_file(vecnorm_path)
    report["resume_learning_rate"] = float(learning_rate)
    return model, normalizer, report


# A transition changes a task's requested behavior, not the meaning/order of
# policy inputs, its actuator dynamics, or the implementation interpreting them.
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
_TRANSITION_TOP_SETTINGS = frozenset({"terrain", "flat_probability", "tracking_weight", "course_distance"})


def _validate_behavior_transition(previous: Mapping[str, Any], requested: Mapping[str, Any]) -> None:
    required = {"schema", "backend", "parent_plant", "sources", "commands", "env"}
    if not required <= previous.keys() or not required <= requested.keys():
        raise BehaviorCheckpointError("Behavior transitions require complete source and requested task identities")
    if previous["schema"] != "mesozoic.trex-command-terrain/v1":
        raise BehaviorCheckpointError("Only the reviewed T-Rex command/terrain task supports behavior transitions")
    for name in ("commands", "env", "sources", "parent_plant"):
        if not isinstance(previous[name], Mapping) or not isinstance(requested[name], Mapping):
            raise BehaviorCheckpointError(f"Behavior identity {name} must be an object")
    differences = []
    for name in previous.keys() | requested.keys():
        if name not in _TRANSITION_TOP_SETTINGS | {"commands", "env"} and previous.get(name) != requested.get(name):
            differences.append(name)
    old_commands, new_commands = previous["commands"], requested["commands"]
    for name in old_commands.keys() | new_commands.keys():
        if name not in _TRANSITION_COMMAND_SETTINGS and old_commands.get(name) != new_commands.get(name):
            differences.append(f"commands.{name}")
    old_env, new_env = previous["env"], requested["env"]
    for name in old_env.keys() | new_env.keys():
        reward_setting = name.endswith(("_weight", "_bonus")) or name in _TRANSITION_REWARD_SETTINGS
        if not reward_setting and name != "max_episode_steps" and old_env.get(name) != new_env.get(name):
            differences.append(f"env.{name}")
    if differences:
        raise BehaviorCheckpointError("Incompatible behavior transition fields: " + ", ".join(sorted(differences)))


def adapt_behavior_checkpoint(
    model_path: str | Path,
    vecnorm_path: str | Path,
    env: Any,
    *,
    behavior_identity: Mapping[str, Any],
    learning_rate: float = 5e-5,
) -> tuple[PPO, BehaviorVecNormalize, dict[str, Any]]:
    """Warm-start another compatible behavior stage without erasing command learning.

    Terrain, flat-terrain sampling, command sampling/switch schedules, reward
    settings and horizon/course length may change. Input scales and adapters,
    source code, parent plant and all other environment mechanics must match.
    Every transition records immediate parent hashes and the previous full task
    identity; it never promotes the result into canonical certification.
    """
    model_path, vecnorm_path = _paths(model_path, vecnorm_path)
    previous_marker = read_checkpoint_attribute(model_path, MODEL_IDENTITY_ATTRIBUTE)
    if not isinstance(previous_marker, Mapping) or previous_marker.get("schema") != BEHAVIOR_IDENTITY_SCHEMA:
        raise BehaviorCheckpointError("Behavior adaptation requires a marked behavior checkpoint")
    previous = previous_marker.get("behavior_identity")
    requested = _identity(behavior_identity)
    if not isinstance(previous, Mapping):
        raise BehaviorCheckpointError("Source behavior identity is missing")
    _validate_behavior_transition(previous, requested)
    model, normalizer, report = load_behavior_checkpoint(
        model_path,
        vecnorm_path,
        env,
        behavior_identity=previous,
        learning_rate=learning_rate,
    )
    transition = {
        "schema": "mesozoic.behavior-transition/v1",
        "parent_checkpoint_sha256": sha256_file(model_path),
        "parent_normalization_sha256": sha256_file(vecnorm_path),
        "parent_training_timesteps": model.num_timesteps,
        "previous_behavior_identity": copy.deepcopy(dict(previous)),
        "behavior_identity": copy.deepcopy(requested),
        "learning_rate": float(learning_rate),
        "command_weights_preserved": True,
        "optimizer_tensors_preserved": True,
        "normalization_statistics_preserved": True,
    }
    marker = copy.deepcopy(dict(previous_marker))
    marker["behavior_identity"] = requested
    marker["transition_parent_checkpoint_sha256"] = transition["parent_checkpoint_sha256"]
    marker["transition_parent_normalization_sha256"] = transition["parent_normalization_sha256"]
    report["behavior_identity"] = copy.deepcopy(requested)
    report.setdefault("transitions", []).append(transition)
    for artifact in (model, normalizer):
        setattr(artifact, MODEL_IDENTITY_ATTRIBUTE, copy.deepcopy(marker))
        setattr(artifact, MODEL_TASK_ATTRIBUTE, copy.deepcopy(marker))
        setattr(artifact, PREPARATION_ATTRIBUTE, copy.deepcopy(report))
    return model, normalizer, report
