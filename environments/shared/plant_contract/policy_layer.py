"""Policy-interface layer: ordered observations, actions, sensors, controls.

This is the layer a trained policy is actually coupled to, so it also captures
the observation-builder and action-mapping *semantics* of the one training
backend, Stable-Baselines3.  Every model is on a heading-free schema
(``bipedal-heading-free/v1`` or ``quadrupedal-heading-free/v1``) and also
hashes the frame helpers its observation calls
(``environments/shared/heading_observation.py``).  The ``jax-mjx`` backend and
the frozen MJX interface core that served it were removed (configs/plant_versions.toml
note 23); a model that still declares it is refused."""

from __future__ import annotations

from typing import Any, Mapping

import mujoco
import numpy as np

from ..command_frame import COMMAND_COMPONENTS, COMMAND_PROBE_VECTOR, COMMAND_SEGMENT_NAME, COMMAND_WIDTH
from ..heading_observation import HEADING_FREE_FUNCTIONS, HEADING_FREE_ROOT_LABELS
from .constants import (
    _ACTION_MAPPING_HOME_KEYFRAME_RESIDUAL,
    _ACTION_MAPPING_MIDPOINT,
    _HOME_KEYFRAME_ACTION_MAPPINGS,
    _MIDPOINT_ACTION_MAPPING_DESCRIPTION,
)
from .digests import _callable_semantics, _canonical_float, _module_function_semantics
from .errors import PlantContractError
from .identity import PlantVersion
from .introspection import _fields, _names

#: The training backends a model may declare (``supported_training_backends``
#: on its environment, ``training_backends`` in configs/species_manifest.toml).
SUPPORTED_TRAINING_BACKENDS = ("stable-baselines3",)

#: The payload's ``jax_interface`` value.  It has been this literal on every
#: SB3-only model since the velociraptor's policy interface r11 and is kept so
#: their policy-interface digests stay byte-identical; it names the removed
#: backend as unsupported, never a supported one.
_REMOVED_JAX_INTERFACE = {"supported": False, "backend": "jax-mjx"}


def declared_training_backends(env: Any, model_id: Any) -> tuple[str, ...]:
    """The env's ``supported_training_backends``, refused unless it is exactly Stable-Baselines3.

    An env that declares nothing trains on Stable-Baselines3.  ``jax-mjx`` is
    refused by name: the frozen MJX interface core that served it was removed
    (configs/plant_versions.toml note 23).
    """
    backends = tuple(getattr(env, "supported_training_backends", SUPPORTED_TRAINING_BACKENDS))
    if "jax-mjx" in backends:
        raise PlantContractError(
            f"{model_id} declares the jax-mjx training backend, which was removed with the frozen MJX interface "
            "core (configs/plant_versions.toml note 23): declare supported_training_backends = ('stable-baselines3',)"
        )
    if backends != SUPPORTED_TRAINING_BACKENDS:
        raise PlantContractError(
            f"invalid training backends for {model_id}: {backends}; the only backend is {SUPPORTED_TRAINING_BACKENDS}"
        )
    return backends


def _action_mapping_contract(model: mujoco.MjModel, env: Any) -> str | dict[str, Any]:
    """Describe the species' normalized-action mapping."""
    mode = str(getattr(env, "action_mapping", _ACTION_MAPPING_MIDPOINT))
    if mode == _ACTION_MAPPING_MIDPOINT:
        return _MIDPOINT_ACTION_MAPPING_DESCRIPTION
    if mode not in _HOME_KEYFRAME_ACTION_MAPPINGS:
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

    description: dict[str, Any] = {
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
    }
    if mode == _ACTION_MAPPING_HOME_KEYFRAME_RESIDUAL:
        return description
    description["residual_shaping"] = _soft_cubic_residual_shaping(model, env, mode)
    return description


def _soft_cubic_residual_shaping(model: mujoco.MjModel, env: Any, mode: str) -> dict[str, Any]:
    """Describe the per-actuator shaping of home-keyframe-residual-softcubic/v1.

    The clipped action ``a`` becomes ``b = s*a + (1 - s)*a**3`` on the shaped
    actuators and stays ``b = a`` on the rest, and ``b`` then takes the
    piecewise-affine span above, so action zero still commands the home ctrl
    and +-1 still reach the ordered ctrlrange endpoints.  The slope at home and
    the mask are what change an action's meaning, so both enter the digest:
    ``linear_slope`` per actuator (``s`` where shaped, 1 elsewhere) and the
    ``shaped`` mask in actuator order.  ``0 < s <= 1`` keeps the map monotone
    (``b' = s + 3 (1 - s) a**2 > 0``) with its endpoints fixed.
    """
    slope = float(getattr(env, "residual_linear_slope", float("nan")))
    if not 0.0 < slope <= 1.0:
        raise PlantContractError(f"{mode} needs a residual_linear_slope in (0, 1], got {slope!r}")
    shaped = np.asarray(getattr(env, "_shaped_residual_mask", np.zeros(0, dtype=bool)))
    if shaped.dtype != np.bool_ or shaped.shape != (model.nu,) or not shaped.any():
        raise PlantContractError(
            f"{mode} needs a boolean _shaped_residual_mask over the {model.nu} actuators selecting at least one, "
            f"got dtype={shaped.dtype}, shape={shaped.shape}"
        )
    return {
        "form": "soft-cubic/v1",
        "linear_slope": np.where(shaped, slope, 1.0),
        "shaped": shaped,
    }


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
    original_command = env._command
    try:
        env.model = model
        env.data = probe_data
        # A NON-ZERO command so the probe exercises the appended slot
        # (BEHAVIOR_RECIPES_PLAN §4.6).
        env._command = np.asarray(COMMAND_PROBE_VECTOR, dtype=np.float32)
        observation = np.asarray(env._get_obs())
    finally:
        env.model = original_model
        env.data = original_data
        env._command = original_command

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


def _require_no_mocap_body(model: mujoco.MjModel, model_id: Any) -> None:
    """Refuse a plant with a mocap body: no scene carries a target since the prey, food and course target went."""
    mocap_ids = np.flatnonzero(np.asarray(model.body_mocapid) >= 0)
    if model.nmocap or mocap_ids.size:
        names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(index)) for index in mocap_ids]
        raise PlantContractError(
            f"{model_id} defines {int(model.nmocap)} mocap bodies {names}; no plant carries a mocap body "
            "(the observation reads no target)"
        )


def _command_segment() -> dict[str, Any]:
    """The trailing command segment, the same on every schema."""
    # Appended LAST on every species (BEHAVIOR_RECIPES_PLAN §4.6, D-C1):
    # body-relative (v_x, v_y, yaw_rate), pre-scaled; zeros under "none".
    return {
        "name": COMMAND_SEGMENT_NAME,
        "width": COMMAND_WIDTH,
        "components": list(COMMAND_COMPONENTS),
        "frame": "body-relative",
        "range": [_canonical_float(-1.0), _canonical_float(1.0)],
    }


def _segment_width(segment: Mapping[str, Any]) -> int:
    return len(segment["components"]) if "components" in segment else int(segment["width"])


def _touch_sensor_groups(source: str, declared: Any, touch_by_address: Mapping[int, str]) -> list[list[int]]:
    """The env's touch-sensor groups named *source*, as sensordata addresses; refused empty or off a touch sensor."""
    groups = [[int(address) for address in group] for group in declared]
    unknown = sorted({address for group in groups for address in group} - set(touch_by_address))
    if not groups or not all(groups) or unknown:
        raise PlantContractError(
            f"the foot_contact segment needs touch-sensor groups, none empty; {source}={groups}, "
            f"addresses that start no touch sensor: {unknown}"
        )
    return groups


def _foot_contact_groups(model: mujoco.MjModel, env: Any) -> list[list[str]]:
    """Each foot channel's touch sensors, by name, from the env's sensordata addresses.

    The channels are the env's ``_foot_observation_groups`` when it declares them (the velociraptor's per-site
    channels since its policy interface r14), which must read exactly the touch sensors of its per-foot
    ``_foot_sensor_groups`` (the rewards' feet), and the per-foot groups otherwise, so every other model's segment
    is unchanged.
    """
    sensor_names = _names(model, mujoco.mjtObj.mjOBJ_SENSOR, model.nsensor)
    touch_by_address = {
        int(model.sensor_adr[index]): sensor_names[index]
        for index in range(model.nsensor)
        if int(model.sensor_type[index]) == int(mujoco.mjtSensor.mjSENS_TOUCH)
    }
    per_foot = _touch_sensor_groups("_foot_sensor_groups", getattr(env, "_foot_sensor_groups", ()), touch_by_address)
    declared = getattr(env, "_foot_observation_groups", None)
    groups = per_foot
    if declared is not None:
        groups = _touch_sensor_groups("_foot_observation_groups", declared, touch_by_address)
        if sorted(address for group in groups for address in group) != sorted(
            address for group in per_foot for address in group
        ):
            raise PlantContractError(
                f"_foot_observation_groups={groups} must read exactly the touch sensors of "
                f"_foot_sensor_groups={per_foot}"
            )
    return [[touch_by_address[address] for address in group] for group in groups]


def _foot_floor_contact_groups(model: mujoco.MjModel, env: Any) -> list[list[str]]:
    """Each floor-contact foot channel's geoms, by name, from the env's ``_foot_observation_floor_geoms``.

    Refused empty, with an empty channel, off the model's geoms, on the floor
    geom itself, or with a geom in two channels.
    """
    geom_names = _names(model, mujoco.mjtObj.mjOBJ_GEOM, model.ngeom)
    groups = [[int(geom) for geom in group] for group in env._foot_observation_floor_geoms]
    flat = [geom for group in groups for geom in group]
    floor = int(env.floor_geom_id)
    if (
        not groups
        or not all(groups)
        or any(not 0 <= geom < model.ngeom for geom in flat)
        or floor in flat
        or len(set(flat)) != len(flat)
    ):
        raise PlantContractError(
            f"the floor-contact foot_contact segment needs disjoint, non-empty groups of the model's geoms, none the "
            f"floor geom {floor}; _foot_observation_floor_geoms={groups}"
        )
    return [[geom_names[geom] for geom in group] for group in groups]


def _foot_contact_segment(model: mujoco.MjModel, env: Any) -> dict[str, Any]:
    """The observation's foot channels: touch-sensor groups, or floor-contact geom groups where the env declares them.

    A model that declares ``_foot_observation_floor_geoms`` (the
    Compsognathus robot, whose soles collide with each other) reads each
    channel as its geoms' summed normal force against the floor geom, so
    force between the animal's own geoms is not counted; every other model
    reads touch sensors (:func:`_foot_contact_groups`).
    """
    if getattr(env, "_foot_observation_floor_geoms", None) is None:
        foot_groups = _foot_contact_groups(model, env)
        return {"name": "foot_contact", "width": len(foot_groups), "groups": foot_groups}
    geom_groups = _foot_floor_contact_groups(model, env)
    return {
        "name": "foot_contact",
        "width": len(geom_groups),
        "source": "floor-contact-normal-force",
        "floor_geom": _names(model, mujoco.mjtObj.mjOBJ_GEOM, model.ngeom)[int(env.floor_geom_id)],
        "geom_groups": geom_groups,
    }


def _heading_free_observation_segments(
    model: mujoco.MjModel,
    env: Any,
    root_label: str,
) -> list[dict[str, Any]]:
    """The heading-free layout: ``bipedal-heading-free/v1`` (root ``pelvis``) and ``quadrupedal-heading-free/v1``
    (root ``torso``), the ``bipedal-target-heading-free/v1`` layout of configs/plant_versions.toml note 18 (D-D31)
    without its target direction and distance.

    The two root blocks are named for their frame and source, with the
    fallback and normalization thresholds the helpers apply, and the foot
    channels are described by their touch sensors: one per foot (a sum over
    its listed sensors), or one per group of the env's
    ``_foot_observation_groups`` where it declares them (the velociraptor's
    per-site channels since its policy interface r14), or by their geoms
    where the env reads floor-contact force instead
    (:func:`_foot_contact_segment`), so the widths sum to the observation
    width, which the caller checks.
    """
    return [
        {"name": "joint_position", "components": _joint_component_names(model, velocity=False)},
        {"name": "joint_velocity", "components": _joint_component_names(model, velocity=True)},
        {
            "name": f"{root_label}_projected_gravity",
            "width": 3,
            "frame": root_label,
            "source": "qpos[3:7]",
            "convention": "unit-down",
            "normalization_floor": _canonical_float(1e-12),
        },
        {"name": f"{root_label}_angular_velocity", "width": 3, "frame": "gyro-site"},
        {
            "name": f"{root_label}_heading_linear_velocity",
            "width": 3,
            "components": ["forward", "left", "up"],
            "frame": f"gravity-aligned-{root_label}-heading",
            "source": "qvel[0:3]",
            "fallback": f"horizontal-{root_label}-y-rotated-clockwise-90",
            "fallback_threshold": _canonical_float(1e-8),
        },
        {"name": f"{root_label}_linear_acceleration", "width": 3, "frame": "accelerometer-site"},
        _foot_contact_segment(model, env),
        _command_segment(),
    ]


def _policy_interface_payload(
    model: mujoco.MjModel,
    env: Any,
    version: PlantVersion,
) -> dict[str, Any]:
    # The root label follows the declared observation schema; the legacy
    # world-frame schemas (bipedal-target/v1, quadrupedal-target/v1) went
    # with the target, so every model declares a heading-free one.
    schema = version.observation_schema
    if schema not in HEADING_FREE_ROOT_LABELS:
        raise PlantContractError(
            f"{version.model_id} declares observation schema {schema!r}; the observation schemas are "
            f"{sorted(HEADING_FREE_ROOT_LABELS)}"
        )
    _require_no_mocap_body(model, version.model_id)
    observation_segments = _heading_free_observation_segments(model, env, HEADING_FREE_ROOT_LABELS[schema])
    segment_widths = [_segment_width(segment) for segment in observation_segments]
    observation_width = int(np.prod(env.observation_space.shape))
    if sum(segment_widths) != observation_width or observation_segments[-1]["name"] != COMMAND_SEGMENT_NAME:
        raise PlantContractError(
            f"{version.model_id}'s {schema} segments must sum to the observation width {observation_width} "
            f"and end with the command; widths {segment_widths} sum to {sum(segment_widths)}, last segment "
            f"{observation_segments[-1]['name']!r}"
        )
    collision_ids = np.flatnonzero((model.geom_contype != 0) | (model.geom_conaffinity != 0))
    sb3_observation_probe = _observation_probe(model, env)
    action_mapping = _action_mapping_contract(model, env)
    backends = declared_training_backends(env, version.model_id)
    interface_implementations: dict[str, Any] = {
        "sb3_observation": _callable_semantics(env._get_obs),
        "sb3_action_mapping": _callable_semantics(env._scale_action),
        # _get_obs calls these helpers: hashing only its call site would leave
        # the frame arithmetic outside the digest.
        "heading_free_observation": _module_function_semantics(
            "environments.shared.heading_observation",
            HEADING_FREE_FUNCTIONS,
        ),
    }
    if getattr(env, "_foot_observation_floor_geoms", None) is not None:
        # The observation probe has no contacts to read, so the floor-contact
        # reading's arithmetic enters the digest through its tokens.
        interface_implementations["foot_floor_contact_observation"] = _callable_semantics(
            env._foot_floor_contact_forces
        )
    # Explicit capability declaration, never an import-error fallback.
    interface_implementations["supported_training_backends"] = list(backends)
    interface_implementations["sb3_sensor_cache"] = _callable_semantics(env._cache_ids)
    if str(getattr(env, "action_mapping", _ACTION_MAPPING_MIDPOINT)) in _HOME_KEYFRAME_ACTION_MAPPINGS:
        interface_implementations["home_reset"] = {"sb3": _callable_semantics(env.reset)}
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
        # Both kept for digest stability (see _REMOVED_JAX_INTERFACE).
        "jax_interface": dict(_REMOVED_JAX_INTERFACE),
        "backend_observation_equal": None,
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
