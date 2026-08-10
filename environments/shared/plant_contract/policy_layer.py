"""Policy-interface layer: ordered observations, actions, sensors, controls.

This is the layer a trained policy is actually coupled to, so it also captures
the observation-builder and action-mapping *semantics* for both the SB3 and
JAX/MJX backends and asserts that the two agree."""

from __future__ import annotations

import importlib
from typing import Any, Mapping

import mujoco
import numpy as np

from .constants import (
    _ACTION_MAPPING_HOME_KEYFRAME_RESIDUAL,
    _ACTION_MAPPING_MIDPOINT,
    _MIDPOINT_ACTION_MAPPING_DESCRIPTION,
)
from .digests import _callable_semantics, _canonical_float, _module_function_semantics
from .errors import PlantContractError
from .identity import PlantVersion
from .introspection import _fields, _names


def _action_mapping_contract(
    model: mujoco.MjModel,
    env: Any,
) -> tuple[str | dict[str, Any], tuple[str, ...]]:
    """Describe the species' normalized-action mapping and JAX implementation."""
    mode = str(getattr(env, "action_mapping", _ACTION_MAPPING_MIDPOINT))
    if mode == _ACTION_MAPPING_MIDPOINT:
        return _MIDPOINT_ACTION_MAPPING_DESCRIPTION, ("scale_action_jax",)
    if mode != _ACTION_MAPPING_HOME_KEYFRAME_RESIDUAL:
        raise PlantContractError(f"unsupported action mapping mode: {mode!r}")

    home_keyframe_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if home_keyframe_id < 0:
        raise PlantContractError("home-keyframe residual mapping requires a named 'home' keyframe")
    nominal_ctrl = np.asarray(model.key_ctrl[home_keyframe_id], dtype=np.float64)
    ctrl_range = np.asarray(model.actuator_ctrlrange, dtype=np.float64)
    if nominal_ctrl.shape != (model.nu,) or not np.all(np.isfinite(nominal_ctrl)):
        raise PlantContractError("home keyframe controls must be finite and match the actuator dimension")
    if np.any(nominal_ctrl < ctrl_range[:, 0]) or np.any(nominal_ctrl > ctrl_range[:, 1]):
        raise PlantContractError("home keyframe controls must remain inside every actuator control range")

    return (
        {
            "schema": "mesozoic.action-mapping/v1",
            "mode": mode,
            "input_clip": [-1.0, 1.0],
            "negative_endpoint": "ordered-ctrlrange-minimum",
            "origin": {
                "keyframe": "home",
                "keyframe_id": home_keyframe_id,
                "ctrl": nominal_ctrl,
            },
            "positive_endpoint": "ordered-ctrlrange-maximum",
            "interpolation": "piecewise-affine/v1",
        },
        ("scale_action_around_nominal_jax",),
    )


def _joint_component_names(model: mujoco.MjModel, *, velocity: bool) -> list[str]:
    widths = {
        int(mujoco.mjtJoint.mjJNT_FREE): 6 if velocity else 7,
        int(mujoco.mjtJoint.mjJNT_BALL): 3 if velocity else 4,
        int(mujoco.mjtJoint.mjJNT_SLIDE): 1,
        int(mujoco.mjtJoint.mjJNT_HINGE): 1,
    }
    components: list[str] = []
    for index, joint_type in enumerate(model.jnt_type):
        if int(joint_type) == int(mujoco.mjtJoint.mjJNT_FREE):
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index) or f"joint_{index}"
        width = widths[int(joint_type)]
        components.extend(f"{name}[{component}]" for component in range(width))
    return components


def _space_payload(space: Any) -> dict[str, Any]:
    return {
        "shape": list(space.shape),
        "dtype": str(space.dtype),
        "low": np.asarray(space.low),
        "high": np.asarray(space.high),
    }


def _deterministic_probe_data(model: mujoco.MjModel) -> mujoco.MjData:
    """Create synthetic observation state without platform-sensitive physics."""
    probe_data = mujoco.MjData(model)
    mujoco.mj_resetData(model, probe_data)
    if model.nq:
        probe_data.qpos[:] = np.linspace(-0.2, 0.3, model.nq)
    if model.nv:
        probe_data.qvel[:] = np.linspace(-0.25, 0.35, model.nv)
    if model.nsensordata:
        probe_data.sensordata[:] = np.linspace(0.05, 0.05 * model.nsensordata, model.nsensordata)
    for index in range(model.nbody):
        probe_data.xpos[index] = (0.1 + 0.03 * index, -0.4 + 0.02 * index, 0.6 + 0.01 * index)
    for index in range(model.nmocap):
        probe_data.mocap_pos[index] = (1.25 + 0.1 * index, -0.45 + 0.05 * index, 0.8 + 0.02 * index)
    return probe_data


def _portable_probe_values(observation: np.ndarray) -> np.ndarray:
    """Quantize executable probe results across supported CPU architectures."""
    return np.round(np.asarray(observation, dtype=np.float64), decimals=6)


def _observation_probe(model: mujoco.MjModel, env: Any) -> dict[str, Any]:
    """Execute the observation ABI against deterministic, non-degenerate state.

    Source fingerprints catch formula edits; this probe also catches changes to
    cached body IDs, sensor offsets, and other runtime mappings used by the
    observation method.
    """
    probe_data = _deterministic_probe_data(model)

    original_model = env.model
    original_data = env.data
    try:
        env.model = model
        env.data = probe_data
        observation = np.asarray(env._get_obs())
    finally:
        env.model = original_model
        env.data = original_data

    expected_shape = tuple(env.observation_space.shape)
    if observation.shape != expected_shape or not np.all(np.isfinite(observation)):
        raise PlantContractError(
            f"observation probe produced invalid output: shape={observation.shape}, expected={expected_shape}"
        )
    return {
        "schema": "mesozoic.observation-probe/v1",
        "dtype": str(observation.dtype),
        "shape": list(observation.shape),
        "quantization_decimals": 6,
        "values": _portable_probe_values(observation),
    }


def _jax_policy_interface_payload(
    model: mujoco.MjModel,
    env: Any,
    version: PlantVersion,
) -> dict[str, Any]:
    """Resolve the MJX observation/control ABI without requiring JAX.

    Species registration modules only populate ordinary Python data.  Running
    their observation builder with NumPy catches stale numeric body IDs and
    sensor offsets while keeping the canonical contract usable in CPU-only CI.
    """
    module_name = f"environments.{version.species}.mjx_config"
    try:
        registration_module = importlib.import_module(module_name)
        mjx_env_module = importlib.import_module("environments.shared.mjx_env")
    except ImportError as exc:
        raise PlantContractError(f"cannot import MJX plant registration {module_name}: {exc}") from exc

    registry = getattr(mjx_env_module, "_SPECIES_CONFIGS", {})
    if version.species not in registry:
        # A test or embedding process may have cleared the registry after the
        # module's first import.  Reloading deterministically re-registers it.
        importlib.reload(registration_module)
    raw_config = getattr(mjx_env_module, "_SPECIES_CONFIGS", {}).get(version.species)
    if not isinstance(raw_config, Mapping):
        raise PlantContractError(f"MJX plant registration is missing for {version.species}")

    from ..obs_functions import SensorLayout

    body_ids = {str(name): int(body_id) for name, body_id in dict(raw_config.get("body_ids", {})).items()}
    if not body_ids:
        raise PlantContractError(f"MJX plant registration has no root body mapping for {version.species}")
    resolved_bodies: dict[str, dict[str, Any]] = {}
    for body_name, registered_id in body_ids.items():
        resolved_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if resolved_id < 0:
            raise PlantContractError(f"MJX root body {body_name!r} does not exist for {version.species}")
        if registered_id != resolved_id:
            raise PlantContractError(
                f"MJX root body ID for {body_name!r} is stale: registered={registered_id}, resolved={resolved_id}"
            )
        resolved_bodies[body_name] = {"registered_id": registered_id, "resolved_id": resolved_id}

    root_name = "torso" if version.observation_schema == "quadrupedal-target/v1" else "pelvis"
    if root_name not in body_ids:
        raise PlantContractError(f"MJX plant registration lacks required {root_name!r} body mapping")

    sensor_layout = SensorLayout(
        gyro_start=int(raw_config.get("sensor_gyro_start", 0)),
        accel_start=int(raw_config.get("sensor_accel_start", 3)),
        quat_start=int(raw_config.get("sensor_quat_start", 6)),
        foot_indices=tuple(int(index) for index in raw_config.get("sensor_foot_indices", ())),
        foot_aux_indices=tuple(
            tuple(int(index) for index in group) for group in raw_config.get("sensor_foot_aux_indices", ())
        ),
    )
    if len(sensor_layout.foot_aux_indices) > len(sensor_layout.foot_indices):
        raise PlantContractError(
            f"MJX sensor layout for {version.species} declares more aux foot sensor groups "
            f"({len(sensor_layout.foot_aux_indices)}) than feet ({len(sensor_layout.foot_indices)})"
        )
    used_sensor_indices = (
        *range(sensor_layout.gyro_start, sensor_layout.gyro_start + 3),
        *range(sensor_layout.accel_start, sensor_layout.accel_start + 3),
        *range(sensor_layout.quat_start, sensor_layout.quat_start + 4),
        *sensor_layout.foot_indices,
        *(index for group in sensor_layout.foot_aux_indices for index in group),
    )
    if any(index < 0 or index >= model.nsensordata for index in used_sensor_indices):
        raise PlantContractError(
            f"MJX sensor layout for {version.species} indexes outside nsensordata={model.nsensordata}"
        )

    registered_frame_skip = int(raw_config.get("frame_skip", -1))
    if registered_frame_skip <= 0:
        raise PlantContractError(f"MJX frame_skip must be positive for {version.species}")
    registered_action_mapping = str(raw_config.get("action_mapping", _ACTION_MAPPING_MIDPOINT))
    sb3_action_mapping = str(getattr(env, "action_mapping", _ACTION_MAPPING_MIDPOINT))
    if registered_action_mapping != sb3_action_mapping:
        raise PlantContractError(
            f"SB3 and MJX action mappings diverge for {version.species}: "
            f"sb3={sb3_action_mapping!r}, mjx={registered_action_mapping!r}"
        )
    registered_filter_cutoff = float(raw_config.get("action_filter_cutoff_hz", 0.0))
    sb3_filter_cutoff = float(getattr(env, "action_filter_cutoff_hz", 0.0))
    if registered_filter_cutoff != sb3_filter_cutoff:
        raise PlantContractError(
            f"SB3 and MJX action filter cutoffs diverge for {version.species}: "
            f"sb3={sb3_filter_cutoff!r}, mjx={registered_filter_cutoff!r}"
        )

    probe_data = _deterministic_probe_data(model)
    target_pos = probe_data.mocap_pos[0] if model.nmocap else np.array([1.25, -0.45, 0.8])
    observation = np.asarray(
        mjx_env_module.build_mjx_observation(
            probe_data,
            target_pos,
            {
                "species": version.species,
                "body_ids": body_ids,
                "sensor_gyro_start": sensor_layout.gyro_start,
                "sensor_accel_start": sensor_layout.accel_start,
                "sensor_quat_start": sensor_layout.quat_start,
                "sensor_foot_indices": sensor_layout.foot_indices,
                "sensor_foot_aux_indices": sensor_layout.foot_aux_indices,
            },
        )
    )
    expected_shape = tuple(env.observation_space.shape)
    if observation.shape != expected_shape or not np.all(np.isfinite(observation)):
        raise PlantContractError(
            f"MJX observation probe produced invalid output: shape={observation.shape}, expected={expected_shape}"
        )

    payload = {
        "registration_module": module_name,
        "frame_skip": registered_frame_skip,
        "control_timestep": model.opt.timestep * registered_frame_skip,
        "body_ids": resolved_bodies,
        "sensor_layout": {
            "gyro_start": sensor_layout.gyro_start,
            "accel_start": sensor_layout.accel_start,
            "quat_start": sensor_layout.quat_start,
            "foot_indices": sensor_layout.foot_indices,
            "foot_aux_indices": sensor_layout.foot_aux_indices,
        },
        "observation_builder": (
            "build_quadruped_obs" if version.observation_schema == "quadrupedal-target/v1" else "build_bipedal_obs"
        ),
        "observation_probe": {
            "schema": "mesozoic.mjx-observation-probe/v1",
            "dtype": str(observation.dtype),
            "shape": list(observation.shape),
            "quantization_decimals": 6,
            "values": _portable_probe_values(observation),
        },
    }
    if registered_action_mapping != _ACTION_MAPPING_MIDPOINT:
        payload["action_mapping"] = registered_action_mapping
    if registered_filter_cutoff > 0.0:
        # Conditional so filter-free species keep their payload (and thus
        # their policy fingerprint) byte-identical.
        payload["action_filter_cutoff_hz"] = _canonical_float(registered_filter_cutoff)
    return payload


def _mocap_target_name(model: mujoco.MjModel) -> str:
    """Return the name of the plant's single mocap target body."""
    mocap_ids = np.flatnonzero(np.asarray(model.body_mocapid) >= 0)
    if mocap_ids.size != 1:
        raise PlantContractError(f"plant must define exactly one mocap target body, found {mocap_ids.size}")
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(mocap_ids[0]))
    if not name:
        raise PlantContractError("the mocap target body must be named")
    return str(name)


def _policy_interface_payload(
    model: mujoco.MjModel,
    env: Any,
    version: PlantVersion,
    *,
    require_backend_parity: bool = False,
) -> dict[str, Any]:
    touch_sensor_names = [
        name
        for index, name in enumerate(_names(model, mujoco.mjtObj.mjOBJ_SENSOR, model.nsensor))
        if int(model.sensor_type[index]) == int(mujoco.mjtSensor.mjSENS_TOUCH)
    ]
    # Segment labels come from the plant itself rather than a species
    # allow-list: the root follows the declared observation schema (as the MJX
    # payload already does) and the target follows the MJCF mocap body name.
    root_label = "torso" if version.observation_schema == "quadrupedal-target/v1" else "pelvis"
    target_label = _mocap_target_name(model)
    observation_functions = (
        ("_array_mod", "build_bipedal_obs", "build_quadruped_obs")
        if version.observation_schema == "quadrupedal-target/v1"
        else ("_array_mod", "build_bipedal_obs")
    )
    observation_segments = [
        {"name": "joint_position", "components": _joint_component_names(model, velocity=False)},
        {"name": "joint_velocity", "components": _joint_component_names(model, velocity=True)},
        {"name": f"{root_label}_orientation_quaternion", "width": 4},
        {"name": f"{root_label}_angular_velocity", "width": 3},
        {"name": f"{root_label}_linear_velocity", "width": 3},
        {"name": f"{root_label}_linear_acceleration", "width": 3},
        {"name": "foot_contact", "components": touch_sensor_names},
        {"name": f"{target_label}_direction", "width": 3, "normalization_epsilon": _canonical_float(1e-8)},
        {"name": f"{target_label}_distance", "width": 1},
    ]
    collision_ids = np.flatnonzero((model.geom_contype != 0) | (model.geom_conaffinity != 0))
    sb3_observation_probe = _observation_probe(model, env)
    jax_interface = _jax_policy_interface_payload(model, env, version)
    action_mapping, jax_action_functions = _action_mapping_contract(model, env)
    mjx_env_module = importlib.import_module("environments.shared.mjx_env")
    jax_setup_module = importlib.import_module("environments.shared.jax_setup")
    backend_observation_equal = np.array_equal(
        sb3_observation_probe["values"],
        jax_interface["observation_probe"]["values"],
    )
    if require_backend_parity and not backend_observation_equal:
        raise PlantContractError(
            f"SB3 and MJX observation probes diverge for {version.species}; "
            "check cached body IDs, sensor offsets, and observation builders"
        )
    interface_implementations = {
        "sb3_observation": _callable_semantics(env._get_obs),
        "sb3_action_mapping": _callable_semantics(env._scale_action),
        "backend_neutral_observation": _module_function_semantics(
            "environments.shared.obs_functions",
            observation_functions,
        ),
        "jax_action_mapping": _module_function_semantics(
            "environments.shared.mjx_utils",
            jax_action_functions,
        ),
        # These are intentionally production observation callables, not a
        # parallel test implementation. Reward/termination code stays out
        # of the interface fingerprint.
        "jax_observation_callers": {
            "training_reset_and_step": _callable_semantics(mjx_env_module.build_mjx_observation),
            "cpu_evaluation": _callable_semantics(jax_setup_module.make_obs_fn),
        },
    }
    if str(getattr(env, "action_mapping", _ACTION_MAPPING_MIDPOINT)) == _ACTION_MAPPING_HOME_KEYFRAME_RESIDUAL:
        interface_implementations["home_reset"] = {
            "sb3": _callable_semantics(env.reset),
            "jax": _module_function_semantics(
                "environments.shared.mjx_utils",
                ("reset_mujoco_data_to_home",),
            ),
        }
    action_filter_cutoff = float(getattr(env, "action_filter_cutoff_hz", 0.0))
    if action_filter_cutoff > 0.0:
        # Conditional: species without the filter keep their fingerprint.
        # For species with it, the cutoff and the filter arithmetic are part
        # of what a checkpoint's action means — edits to action_filter.py
        # move this fingerprint and force a policy revision bump.
        interface_implementations["action_low_pass_filter"] = {
            "cutoff_hz": _canonical_float(action_filter_cutoff),
            "implementation": _module_function_semantics(
                "environments.shared.action_filter",
                ("low_pass_alpha", "apply_low_pass"),
            ),
        }
    return {
        "observation_schema": version.observation_schema,
        "observation": _space_payload(env.observation_space),
        "observation_probe": sb3_observation_probe,
        "observation_segments": observation_segments,
        "action": _space_payload(env.action_space),
        "action_mapping": action_mapping,
        "interface_implementations": interface_implementations,
        "jax_interface": jax_interface,
        "backend_observation_equal": backend_observation_equal,
        "frame_skip": int(env.frame_skip),
        "physics_timestep": model.opt.timestep,
        "control_timestep": model.opt.timestep * int(env.frame_skip),
        "dimensions": {
            "nq": int(model.nq),
            "nv": int(model.nv),
            "nu": int(model.nu),
            "na": int(model.na),
            "nsensor": int(model.nsensor),
            "nsensordata": int(model.nsensordata),
        },
        "bodies": _names(model, mujoco.mjtObj.mjOBJ_BODY, model.nbody),
        "joints": {
            "names": _names(model, mujoco.mjtObj.mjOBJ_JOINT, model.njnt),
            **_fields(model, ("jnt_type", "jnt_qposadr", "jnt_dofadr")),
        },
        "collision_geoms": [_names(model, mujoco.mjtObj.mjOBJ_GEOM, model.ngeom)[index] for index in collision_ids],
        "sites": {
            "names": _names(model, mujoco.mjtObj.mjOBJ_SITE, model.nsite),
            **_fields(model, ("site_bodyid", "site_type", "site_pos", "site_quat", "site_size")),
        },
        "actuators": {
            "names": _names(model, mujoco.mjtObj.mjOBJ_ACTUATOR, model.nu),
            **_fields(
                model,
                (
                    "actuator_trntype",
                    "actuator_trnid",
                    "actuator_ctrllimited",
                    "actuator_ctrlrange",
                ),
            ),
        },
        "sensors": {
            "names": _names(model, mujoco.mjtObj.mjOBJ_SENSOR, model.nsensor),
            **_fields(
                model,
                (
                    "sensor_type",
                    "sensor_datatype",
                    "sensor_needstage",
                    "sensor_objtype",
                    "sensor_objid",
                    "sensor_reftype",
                    "sensor_refid",
                    "sensor_dim",
                    "sensor_adr",
                    "sensor_cutoff",
                    "sensor_noise",
                    "sensor_delay",
                    "sensor_interval",
                    "sensor_intprm",
                    "sensor_user",
                ),
            ),
        },
    }
