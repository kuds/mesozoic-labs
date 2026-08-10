"""JAX-native batched dinosaur environment using MuJoCo MJX.

Unlike the Gymnasium environments in ``base_env.py``, this module uses a
**functional style** suitable for ``jax.jit`` and ``jax.vmap``:

- State is explicit (returned / passed as arguments, never mutated).
- All operations are pure functions (no ``self.data`` mutation).
- No Python-level loops over environments — parallelism via ``vmap``.

The MJX env shares the **same MJCF models**, **same TOML stage configs**,
and **same pure reward / observation functions** as the Gymnasium path.

Usage::

    from environments.shared.mjx_env import MJXDinoEnv

    env = MJXDinoEnv("trex", stage=1, num_envs=2048)
    rng = jax.random.PRNGKey(42)
    states = env.reset(rng)
    actions = jax.random.uniform(rng, (2048, env.action_dim))
    states, rewards, terminated, truncated, info = env.step(states, actions, rng)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .mjx_utils import check_jax

_logger = logging.getLogger(__name__)

ACTION_MAPPING_MIDPOINT = "midpoint/v1"
ACTION_MAPPING_HOME_KEYFRAME_RESIDUAL = "home-keyframe-residual/v1"

# ---------------------------------------------------------------------------
# TOML [env] key translation
# ---------------------------------------------------------------------------
# The Gymnasium envs use species-flavoured kwarg names (strike_*, bite_*,
# food_*, prey_*); the MJX env uses species-neutral names.  Stage TOMLs are
# shared between both backends, so translate here.  Without this, keys like
# velociraptor's ``strike_approach_weight`` were silently ignored while the
# species registry's ``approach_weight`` default leaked into every stage.
_ENV_KEY_ALIASES: dict[str, str] = {
    "strike_approach_weight": "approach_weight",
    "bite_approach_weight": "approach_weight",
    "food_approach_weight": "approach_weight",
    "snap_approach_weight": "approach_weight",
    "prey_distance_range": "target_distance_range",
    "food_distance_range": "target_distance_range",
    "prey_lateral_range": "target_lateral_range",
    "food_lateral_range": "target_lateral_range",
    "food_reach_threshold": "success_threshold",
}

# Reward-weight keys actually read by the MJX step / jax_reward_termination.
_KNOWN_REWARD_KEYS: frozenset = frozenset(
    {
        "forward_vel_weight",
        "forward_vel_max",
        "alive_bonus",
        "energy_penalty_weight",
        "posture_weight",
        "approach_weight",
        "foot_contact_gate",
        "foot_contact_weight",
        "drift_penalty_weight",
        "speed_penalty_weight",
        "speed_penalty_threshold",
        "nosedive_weight",
        "nosedive_termination_threshold",
        "spin_penalty_weight",
        "smoothness_weight",
        "height_weight",
        "tail_stability_weight",
        "heading_weight",
        "lateral_penalty_weight",
        "bite_head_proximity_weight",
        "strike_claw_proximity_weight",
        "food_head_proximity_weight",
        "snap_snout_proximity_weight",
        "strike_bonus",
        "bite_bonus",
        "food_reach_bonus",
        "snap_bonus",
        "bilateral_support_weight",
        "foot_contact_saturation_force",
        "foot_load_balance_weight",
        "foot_load_balance_min_support_force",
        "foot_load_balance_airborne_penalty",
        "action_jerk_weight",
        "support_conditioned_alive_fraction",
        "leg_home_pose_weight",
        "leg_home_pose_tolerance",
        "head_clearance_weight",
        "head_clearance_target",
        "head_clearance_tolerance",
        "neck_posture_weight",
        "neck_posture_tolerance",
        "height_target_tolerance",
    }
)

# TOML [env] keys implemented only by the SB3 (Gymnasium) reward path.
# Accepted without warning (the TOMLs are shared) but have no effect in MJX.
_SB3_ONLY_ENV_KEYS: frozenset = frozenset(
    {
        "backward_vel_penalty_weight",
        "gait_symmetry_weight",
        "gait_stability_weight",
        "idle_penalty_weight",
        "idle_velocity_threshold",
        "strike_proximity_weight",
        "food_height_range",
        # Inert everywhere since ground settling: both resets overwrite the
        # root height as a function of the sampled joint pose.  SB3 still
        # draws it for RNG-stream compatibility; MJX never drew it at all.
        "reset_height_noise_scale",
    }
)


def canonicalize_env_kwargs(env_kwargs: dict[str, Any]) -> dict[str, Any]:
    """Translate species-flavoured TOML ``[env]`` keys to MJX names.

    Applies :data:`_ENV_KEY_ALIASES`, converts ``natural_pitch`` (radians)
    to ``natural_forward_z``, and warns about keys that neither the MJX
    config nor its reward functions recognise (likely typos).  Returns a
    new dict; the input is not modified.
    """
    config_fields = {f.name for f in MJXEnvConfig.__dataclass_fields__.values()}
    out: dict[str, Any] = {}
    for key, value in env_kwargs.items():
        canonical = _ENV_KEY_ALIASES.get(key, key)
        if canonical == "natural_pitch":
            out["natural_forward_z"] = -math.sin(float(value))
            continue
        if (
            canonical not in config_fields
            and canonical not in _KNOWN_REWARD_KEYS
            and canonical not in _SB3_ONLY_ENV_KEYS
            and canonical != "reward_weights"
        ):
            _logger.warning(
                "Unknown [env] key %r for MJX env — it will be ignored. Check for typos or add it to the MJX reward path.",
                key,
            )
        out[canonical] = value
    return out


@dataclass(frozen=True)
class MJXEnvConfig:
    """Immutable configuration for a species + stage in the MJX env."""

    species: str
    stage: int
    frame_skip: int
    max_episode_steps: int
    healthy_z_range: tuple[float, float]
    max_tilt_angle: float
    reward_weights: dict[str, float] = field(default_factory=dict)
    body_ids: dict[str, int] = field(default_factory=dict)
    sensor_foot_indices: tuple[int, ...] = ()
    # Extra touch sensors summed into each foot's reading, aligned with
    # ``sensor_foot_indices``.  Needed where the load-bearing geoms sit on
    # child bodies that a single touch sensor cannot see.  Empty leaves both
    # the observation width and the per-foot value unchanged.
    sensor_foot_aux_indices: tuple[tuple[int, ...], ...] = ()
    sensor_gyro_start: int = 0
    sensor_accel_start: int = 3
    sensor_quat_start: int = 6
    action_mapping: str = ACTION_MAPPING_MIDPOINT
    # First-order low-pass on the commanded action, in Hz; 0.0 disables it.
    # Plant-interface field (not curriculum-tunable): an enabled cutoff
    # changes what a checkpoint's action means.  Must equal the SB3 env's
    # class attribute — the plant contract asserts backend agreement.
    # See environments/shared/action_filter.py.
    action_filter_cutoff_hz: float = 0.0
    natural_forward_z: float = 0.0
    # Reward-only target.  ``None`` preserves the vertical posture reward;
    # Velociraptor supplies its natural forward lean.  Absolute-tilt and
    # nosedive termination remain independent and unchanged.
    posture_target_forward_z: float | None = None
    forward_vel_max: float = 8.0
    fall_penalty: float = -100.0
    # Target spawning ranges
    target_distance_range: tuple[float, float] = (3.0, 8.0)
    target_lateral_range: tuple[float, float] = (-2.0, 2.0)
    target_z: float = 0.5
    # Name of the mocap target body in the MJCF ("prey" for the bipeds,
    # "food" for brachiosaurus).  Used by CPU evaluation to resolve the
    # target position.
    target_body_name: str = "prey"
    # Species standing height for the height-maintenance reward gradient
    # (the SB3 envs use e.g. 0.90m for T-Rex, 1.2m for Brachiosaurus).
    # None falls back to healthy_z_range[1] — a much flatter gradient —
    # for species that don't use height_weight.
    target_standing_z: float | None = None
    # Body-height termination: maps body name -> z threshold.
    # If a monitored body's xpos z drops below its threshold, the episode
    # terminates (JAX-compatible alternative to contact-pair iteration).
    termination_body_heights: dict[str, float] = field(default_factory=dict)
    # Stage 3 success: proximity-based contact detection.
    # success_sites: MuJoCo site names checked against target_pos.
    #   Success triggers if ANY site is within success_threshold of target.
    #   E.g. ("head_tip",) for T-Rex, ("r_claw_tip", "l_claw_tip") for raptor.
    # success_threshold: distance below which success is triggered.
    # success_bonus_key: reward_weights key for success bonus (e.g. "bite_bonus",
    #   "strike_bonus", "food_reach_bonus").  Success detection is gated on
    #   this value being > 0, so stages 1-2 automatically disable it via
    #   their TOML configs which set the bonus to 0.
    success_sites: tuple[str, ...] = ()
    success_threshold: float = 0.3
    success_bonus_key: str = ""
    # Site-height termination: maps site name -> z threshold.
    # Like body-height termination but uses site_xpos (more precise for
    # extremities like snout tips that are far from their parent body origin).
    termination_site_heights: dict[str, float] = field(default_factory=dict)
    # Tail stability sensor index (gyro sensor for tail tip angular velocity)
    sensor_tail_gyro_start: int | None = None
    # Optional species-specific reward geometry.  These names are resolved
    # against the loaded model and its XML ``home`` keyframe without entering
    # the observation or action mapping, so enabling their zero-default reward
    # weights does not change the checkpoint-facing plant interface.
    leg_home_pose_joint_names: tuple[str, ...] = ()
    neck_posture_joint_names: tuple[str, ...] = ()
    head_clearance_site: str | None = None
    # Reset noise parameters
    reset_noise_scale: float = 0.0  # Joint angle perturbation range
    init_qpos_noise: float = 0.0  # XY position jitter range
    init_yaw_noise: float = 0.0  # Yaw rotation perturbation range (radians)


@dataclass
class EnvState:
    """Mutable per-environment state carried between steps.

    In JAX this is a pytree (all fields are JAX arrays or scalars).
    """

    data: Any  # mjx.Data  (kept as Any to avoid import at module level)
    obs: Any  # jnp.ndarray
    step_count: Any  # jnp.int32
    prev_action: Any  # jnp.ndarray
    # Second lag, for the frequency-aware jerk term (see reward_action_jerk).
    prev_prev_action: Any  # jnp.ndarray
    # Command low-pass carry (action_filter_cutoff_hz > 0); seeded with the
    # first post-reset action via a step_count==0 select, mirroring the SB3
    # env's _filter_action.  Carried untouched when the filter is off.
    filtered_action: Any  # jnp.ndarray
    prev_target_distance: Any  # jnp.float32
    target_pos: Any  # jnp.ndarray (3,)
    initial_pos_2d: Any  # jnp.ndarray (2,) — spawn XY for drift penalty


# Register EnvState as a JAX pytree so it can be returned from jit/vmap.
# Guarded: JAX is optional (not installed in CPU-only test environments).
try:
    import jax  # noqa: E402

    jax.tree_util.register_dataclass(
        EnvState,
        data_fields=[
            "data",
            "obs",
            "step_count",
            "prev_action",
            "prev_prev_action",
            "filtered_action",
            "prev_target_distance",
            "target_pos",
            "initial_pos_2d",
        ],
        meta_fields=[],
    )
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Species registration
# ---------------------------------------------------------------------------

_SPECIES_CONFIGS: dict[str, dict[str, Any]] = {}

# These values define the checkpoint-facing plant ABI.  Curriculum files may
# tune rewards and termination thresholds, but may not silently replace the
# control cadence or observation mapping recorded by the plant contract.
_PLANT_INTERFACE_CONFIG_FIELDS = frozenset(
    {
        "frame_skip",
        "body_ids",
        "sensor_foot_indices",
        "sensor_foot_aux_indices",
        "sensor_gyro_start",
        "sensor_accel_start",
        "sensor_quat_start",
        "action_mapping",
        "action_filter_cutoff_hz",
    }
)


def register_species_mjx(species: str, **kwargs: Any) -> None:
    """Register a species configuration for MJX environments.

    Called from per-species ``mjx_config.py`` modules.
    """
    _SPECIES_CONFIGS[species] = kwargs


def build_mjx_observation(data: Any, target_pos: Any, config: MJXEnvConfig | Mapping[str, Any]) -> Any:
    """Build the production MJX policy observation from registered ABI data.

    The plant contract executes and fingerprints this small function directly,
    while both reset and step call the same implementation.  Keep reward and
    termination logic outside it so those changes do not invalidate a policy's
    observation/action interface.
    """
    from .obs_functions import SensorLayout, build_bipedal_obs, build_quadruped_obs

    def value(name: str, default: Any = None) -> Any:
        if isinstance(config, Mapping):
            return config[name] if name in config else default
        return getattr(config, name, default)

    body_ids = value("body_ids")
    # Dispatch on the registered root body, not on a species allow-list: every
    # quadruped registers a "torso" root and every biped a "pelvis" root, so a
    # new species is admitted by its own mjx_config rather than by editing this
    # function.  Editing it would re-fingerprint the policy interface of every
    # species that already shipped.
    quadrupedal = "torso" in body_ids
    root_name = "torso" if quadrupedal else "pelvis"
    root_body_id = int(body_ids[root_name])
    sensor_layout = SensorLayout(
        gyro_start=int(value("sensor_gyro_start")),
        accel_start=int(value("sensor_accel_start")),
        quat_start=int(value("sensor_quat_start")),
        foot_indices=tuple(int(index) for index in value("sensor_foot_indices")),
        foot_aux_indices=tuple(
            tuple(int(index) for index in group) for group in (value("sensor_foot_aux_indices", ()) or ())
        ),
    )
    observation_builder = build_quadruped_obs if quadrupedal else build_bipedal_obs
    return observation_builder(
        data.qpos,
        data.qvel,
        data.sensordata,
        data.xpos[root_body_id],
        target_pos,
        sensor_layout,
    )


def _get_model_path(species: str) -> str:
    """Resolve the MJCF model path for a species."""
    species_dir_map = {
        "velociraptor": "velociraptor",
        "trex": "trex",
        "brachiosaurus": "brachiosaurus",
        "dibothrosuchus": "dibothrosuchus",
    }
    asset_name_map = {
        "velociraptor": "raptor.xml",
        "trex": "trex.xml",
        "brachiosaurus": "brachiosaurus.xml",
        "dibothrosuchus": "dibothrosuchus.xml",
    }
    dir_name = species_dir_map[species]
    asset_name = asset_name_map[species]
    return str(Path(__file__).parent.parent / dir_name / "assets" / asset_name)


def _resolve_home_pose_joints(
    mj_model: Any,
    joint_names: tuple[str, ...],
) -> tuple[tuple[int, ...], tuple[float, ...]]:
    """Resolve scalar joint qpos addresses and targets from ``key home``."""
    if not joint_names:
        return (), ()

    import mujoco

    home_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if home_id < 0:
        raise ValueError("home-pose rewards require a keyframe named 'home'")

    qpos_indices: list[int] = []
    home_positions: list[float] = []
    for joint_name in joint_names:
        joint_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise ValueError(f"home-pose reward joint {joint_name!r} does not exist")
        qpos_index = int(mj_model.jnt_qposadr[joint_id])
        qpos_indices.append(qpos_index)
        home_positions.append(float(mj_model.key_qpos[home_id, qpos_index]))

    return tuple(qpos_indices), tuple(home_positions)


# ---------------------------------------------------------------------------
# Ground settling
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _GroundSettle:
    """Per-model constants for the analytic reset ground settle.

    The reset jitters joint angles and yaw with the root height fixed at the
    keyframe value, but the two are not independent in the world: bending the
    legs changes how far the lowest geom sits below the root, so a fixed root
    height is wrong for most sampled poses — the same plant defect the
    Gymnasium reset settled in ``BaseDinoEnv._settle_root_on_ground``, where
    the contact solver answered spawn penetration with up to 19x body weight.

    MJX has no ``mj_geomDistance``, so the probe is analytic instead: for each
    animal geom the lowest point under gravity is ``xpos_z`` minus a support
    extent that decomposes over the world-z row ``r2`` of the geom rotation as

        extent = const + sum_i lin[i] * |r2[i]| + sqrt(sum_i (ell[i] * r2[i])^2)

    which is EXACT for every primitive the species use (sphere, capsule,
    cylinder, box, ellipsoid) against a horizontal plane floor.  Meshes have
    no closed form here and raise at construction rather than mis-settling.

    Floor assumptions, kept in step with ``BaseDinoEnv._static_floor_geoms``:
    exactly one horizontal floor, treated as an INFINITE plane at the highest
    floor geom's z.  The Gymnasium probe instead takes the minimum
    ``mj_geomDistance`` over every floor geom, which respects finite plane
    extents.  Against one 100x100 plane at z=0 with a spawn near the origin the
    two agree to ~1e-9 (pinned by the cross-backend settle-target test), but
    multiple floors at different heights, or a spawn past the plane's extent,
    would diverge.  A non-+z plane normal and a heightfield both raise here
    rather than settle wrongly.
    """

    geom_ids: Any  # (n,) int32 ndarray of animal (free-root subtree) geom ids
    const_extent: Any  # (n,) float32 ndarray, constant term per geom
    lin_extent: Any  # (n, 3) float32 ndarray, coefficients on |r2|
    ell_extent: Any  # (n, 3) float32 ndarray, coefficients inside the sqrt
    floor_z: float  # surface height of the highest horizontal floor plane
    root_z_index: int  # qpos index of the free root's height


def _ground_settle_constants(mj_model: Any) -> _GroundSettle | None:
    """Precompute :class:`_GroundSettle` for a model, or ``None`` if the model
    has no free-jointed animal or no plane floor to settle it against."""
    import mujoco
    import numpy as np

    free = [j for j in range(mj_model.njnt) if mj_model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
    if not free:
        return None

    floor_zs: list[float] = []
    for g in range(mj_model.ngeom):
        if int(mj_model.geom_bodyid[g]) != 0:
            continue
        if mj_model.geom_type[g] == mujoco.mjtGeom.mjGEOM_HFIELD:
            raise ValueError(
                "MJX ground settling supports horizontal plane floors only; a "
                "heightfield floor needs an iterative settle (see "
                "BaseDinoEnv._static_floor_geoms for the same caveat on SB3)."
            )
        if mj_model.geom_type[g] != mujoco.mjtGeom.mjGEOM_PLANE:
            continue
        normal = np.empty(3)
        mujoco.mju_rotVecQuat(normal, np.array([0.0, 0.0, 1.0]), np.asarray(mj_model.geom_quat[g], dtype=float))
        if abs(normal[2] - 1.0) > 1e-9:
            raise ValueError("MJX ground settling requires the floor plane normal to be world +z")
        floor_zs.append(float(mj_model.geom_pos[g][2]))
    if not floor_zs:
        return None

    root_body = int(mj_model.jnt_bodyid[free[0]])
    bodies = {root_body}
    for b in range(root_body + 1, mj_model.nbody):
        if int(mj_model.body_parentid[b]) in bodies:
            bodies.add(b)

    geom_ids: list[int] = []
    const_extent: list[float] = []
    lin_extent: list[tuple[float, float, float]] = []
    ell_extent: list[tuple[float, float, float]] = []
    for g in range(mj_model.ngeom):
        if int(mj_model.geom_bodyid[g]) not in bodies:
            continue
        # Skip non-colliding geoms, matching BaseDinoEnv._root_subtree_geoms:
        # a geom that generates no contacts can never rest on the floor, so
        # settling to one would hold the real feet above the ground.
        if int(mj_model.geom_contype[g]) == 0 and int(mj_model.geom_conaffinity[g]) == 0:
            continue
        geom_type = mj_model.geom_type[g]
        size = mj_model.geom_size[g]
        if geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
            parts = (float(size[0]), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        elif geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE:
            parts = (float(size[0]), (0.0, 0.0, float(size[1])), (0.0, 0.0, 0.0))
        elif geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
            parts = (0.0, (0.0, 0.0, float(size[1])), (float(size[0]), float(size[0]), 0.0))
        elif geom_type == mujoco.mjtGeom.mjGEOM_BOX:
            parts = (0.0, (float(size[0]), float(size[1]), float(size[2])), (0.0, 0.0, 0.0))
        elif geom_type == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
            parts = (0.0, (0.0, 0.0, 0.0), (float(size[0]), float(size[1]), float(size[2])))
        else:
            name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_GEOM, g) or f"geom{g}"
            raise ValueError(f"MJX ground settling has no support extent for geom {name!r} (type {int(geom_type)})")
        geom_ids.append(g)
        const_extent.append(parts[0])
        lin_extent.append(parts[1])
        ell_extent.append(parts[2])
    if not geom_ids:
        return None

    return _GroundSettle(
        geom_ids=np.array(geom_ids, dtype=np.int32),
        const_extent=np.array(const_extent, dtype=np.float32),
        lin_extent=np.array(lin_extent, dtype=np.float32),
        ell_extent=np.array(ell_extent, dtype=np.float32),
        floor_z=max(floor_zs),
        root_z_index=int(mj_model.jnt_qposadr[free[0]]) + 2,
    )


def _mjx_lowest_ground_clearance(settle: _GroundSettle, geom_xpos: Any, geom_xmat: Any) -> Any:
    """Signed distance from the animal's lowest geom to the floor, in metres.

    JAX counterpart of ``BaseDinoEnv.lowest_ground_clearance``; negative means
    the pose interpenetrates the floor.  Requires kinematics to have been run
    on the state the arrays come from.
    """
    import jax.numpy as jnp

    xpos = geom_xpos[settle.geom_ids]
    r2 = geom_xmat[settle.geom_ids].reshape(-1, 3, 3)[:, 2, :]
    extent = (
        settle.const_extent
        + jnp.sum(settle.lin_extent * jnp.abs(r2), axis=1)
        + jnp.sqrt(jnp.sum((settle.ell_extent * r2) ** 2, axis=1))
    )
    return jnp.min(xpos[:, 2] - extent) - settle.floor_z


# ---------------------------------------------------------------------------
# MJX Environment
# ---------------------------------------------------------------------------


class MJXDinoEnv:
    """JAX-native batched dinosaur environment.

    This environment wraps MuJoCo MJX for GPU-accelerated physics
    simulation of dinosaur locomotion.  It mirrors the Gymnasium
    ``BaseDinoEnv`` lifecycle (reset → step → obs/reward/done) but
    uses functional pure functions compatible with ``jax.jit``/``vmap``.

    Args:
        species: One of ``"trex"``, ``"velociraptor"``, ``"brachiosaurus"``.
        stage: Curriculum stage (1, 2, or 3).
        num_envs: Number of parallel environments (default 2048).
    """

    def __init__(self, species: str, stage: int = 1, num_envs: int = 2048, env_kwargs: dict[str, Any] | None = None):
        check_jax()

        import jax
        import jax.numpy as jnp
        import mujoco
        import mujoco.mjx as mjx

        self.num_envs = num_envs

        # Load MJCF model (one-time, on CPU)
        model_path = _get_model_path(species)
        self.mj_model = mujoco.MjModel.from_xml_path(model_path)
        self.mjx_model = mjx.put_model(self.mj_model)

        # Action / observation dimensions
        self.action_dim = self.mj_model.nu
        self.ctrl_range = jnp.array(self.mj_model.actuator_ctrlrange)

        # Build config from species registration, then overlay TOML stage
        # config.  Deep-copy the registry entry: the merge below writes into
        # this dict, and mutating the live registry would bake one stage's
        # TOML into every env constructed later in the same process (e.g.
        # stage-1 penalties leaking into a stage-2 curriculum env).
        species_kwargs = dict(_SPECIES_CONFIGS.get(species, {}))
        if "reward_weights" in species_kwargs:
            species_kwargs["reward_weights"] = dict(species_kwargs["reward_weights"])

        # Merge TOML env_kwargs over species defaults.  Reward weights are
        # merged at the dict level so that stage-specific weights override
        # the species defaults while keeping any keys not mentioned in TOML.
        # TOML keys are canonicalized first (strike_approach_weight ->
        # approach_weight, prey_distance_range -> target_distance_range, …)
        # so the shared SB3/JAX TOMLs actually take effect here.
        if env_kwargs:
            env_kwargs = canonicalize_env_kwargs(env_kwargs)
            # Velociraptor uses natural pitch for both nosedive semantics and
            # its reward-only posture target.  Keep a caller-provided
            # natural_pitch override in parity with RaptorEnv while leaving
            # upright-target species unchanged.  An explicit posture target
            # still takes precedence.
            if (
                "natural_forward_z" in env_kwargs
                and species_kwargs.get("posture_target_forward_z") is not None
                and "posture_target_forward_z" not in env_kwargs
            ):
                env_kwargs["posture_target_forward_z"] = env_kwargs["natural_forward_z"]
            for key in sorted(_PLANT_INTERFACE_CONFIG_FIELDS & env_kwargs.keys()):
                if env_kwargs[key] != species_kwargs.get(key):
                    raise ValueError(
                        f"{key!r} is part of the versioned plant interface and cannot be overridden "
                        "by a stage configuration"
                    )
            # Separate reward-weight keys from top-level MJXEnvConfig fields
            config_fields = {f.name for f in MJXEnvConfig.__dataclass_fields__.values()}
            merged_reward_weights = dict(species_kwargs.get("reward_weights", {}))

            for key, value in env_kwargs.items():
                if key == "reward_weights":
                    merged_reward_weights.update(value)
                elif key in config_fields:
                    species_kwargs[key] = value
                else:
                    # Keys not in MJXEnvConfig but present in TOML [env]
                    # are reward weight overrides (e.g. alive_bonus,
                    # posture_weight) — merge into reward_weights dict.
                    merged_reward_weights[key] = value

            species_kwargs["reward_weights"] = merged_reward_weights

        self.config = MJXEnvConfig(
            species=species,
            stage=stage,
            **species_kwargs,
        )

        support_alive_fraction = float(self.config.reward_weights.get("support_conditioned_alive_fraction", 0.0))
        if not 0.0 <= support_alive_fraction <= 1.0:
            raise ValueError("support_conditioned_alive_fraction must be in [0, 1]")
        if float(self.config.reward_weights.get("height_target_tolerance", 0.0)) < 0.0:
            raise ValueError("height_target_tolerance must be non-negative")

        for body_name, registered_id in self.config.body_ids.items():
            resolved_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if resolved_id < 0:
                raise ValueError(f"registered MJX body {body_name!r} does not exist in the {species} model")
            if int(registered_id) != resolved_id:
                raise ValueError(
                    f"registered MJX body ID for {body_name!r} is stale: "
                    f"registered={registered_id}, resolved={resolved_id}"
                )

        # Resolve termination body names to (body_id, z_threshold) pairs
        self._termination_body_checks: list[tuple[int, float]] = []
        for body_name, z_thresh in self.config.termination_body_heights.items():
            bid = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if bid >= 0:
                self._termination_body_checks.append((bid, z_thresh))

        # Resolve termination site names to (site_id, z_threshold) pairs
        self._termination_site_checks: list[tuple[int, float]] = []
        for site_name, z_thresh in self.config.termination_site_heights.items():
            sid = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SITE, site_name)
            if sid >= 0:
                self._termination_site_checks.append((sid, z_thresh))

        # Resolve stage 3 success site IDs
        self._success_site_ids: tuple[int, ...] = tuple(
            mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SITE, name)
            for name in self.config.success_sites
            if mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SITE, name) >= 0
        )

        (
            self._leg_home_pose_qpos_indices,
            leg_home_pose_targets,
        ) = _resolve_home_pose_joints(self.mj_model, self.config.leg_home_pose_joint_names)
        (
            self._neck_posture_qpos_indices,
            neck_posture_targets,
        ) = _resolve_home_pose_joints(self.mj_model, self.config.neck_posture_joint_names)
        self._leg_home_pose_targets = jnp.asarray(leg_home_pose_targets)
        self._neck_posture_targets = jnp.asarray(neck_posture_targets)
        self._head_clearance_site_id: int | None = None
        if self.config.head_clearance_site is not None:
            head_clearance_site_id = mujoco.mj_name2id(
                self.mj_model,
                mujoco.mjtObj.mjOBJ_SITE,
                self.config.head_clearance_site,
            )
            if head_clearance_site_id < 0:
                raise ValueError(f"head-clearance site {self.config.head_clearance_site!r} does not exist")
            self._head_clearance_site_id = head_clearance_site_id

        reward_weights = self.config.reward_weights
        if reward_weights.get("leg_home_pose_weight", 0.0) > 0 and not self._leg_home_pose_qpos_indices:
            raise ValueError("leg_home_pose_weight requires configured leg_home_pose_joint_names")
        if reward_weights.get("neck_posture_weight", 0.0) > 0 and not self._neck_posture_qpos_indices:
            raise ValueError("neck_posture_weight requires configured neck_posture_joint_names")
        if reward_weights.get("head_clearance_weight", 0.0) > 0 and self._head_clearance_site_id is None:
            raise ValueError("head_clearance_weight requires a configured head_clearance_site")
        if (
            reward_weights.get("bilateral_support_weight", 0.0) > 0
            or reward_weights.get("foot_load_balance_weight", 0.0) > 0
            or support_alive_fraction > 0
        ) and len(self.config.sensor_foot_indices) < 2:
            raise ValueError("bilateral support rewards require at least two configured foot sensors")
        if (
            reward_weights.get("bilateral_support_weight", 0.0) > 0 or support_alive_fraction > 0
        ) and reward_weights.get("foot_contact_saturation_force", 100.0) <= 0:
            raise ValueError("foot_contact_saturation_force must be positive when support shaping is enabled")
        for weight_key, tolerance_key in (
            ("leg_home_pose_weight", "leg_home_pose_tolerance"),
            ("head_clearance_weight", "head_clearance_tolerance"),
            ("neck_posture_weight", "neck_posture_tolerance"),
        ):
            if reward_weights.get(weight_key, 0.0) > 0 and reward_weights.get(tolerance_key, 0.0) <= 0:
                raise ValueError(f"{tolerance_key} must be positive when {weight_key} is enabled")

        # Cache reset data once (avoids constructing data per env per reset).
        # The home-residual action interface must start from the same complete
        # keyframed state that defines action zero: qpos, qvel, act, ctrl, and
        # mocap state.  Other species intentionally retain the historical MJX
        # default reset and midpoint action mapping.
        self.nominal_ctrl = None
        if self.config.action_mapping == ACTION_MAPPING_HOME_KEYFRAME_RESIDUAL:
            from .mjx_utils import reset_mujoco_data_to_home

            home_data = mujoco.MjData(self.mj_model)
            reset_mujoco_data_to_home(self.mj_model, home_data)
            nominal_ctrl = home_data.ctrl.copy()
            ctrl_range = self.mj_model.actuator_ctrlrange
            if ((nominal_ctrl < ctrl_range[:, 0]) | (nominal_ctrl > ctrl_range[:, 1])).any():
                raise ValueError("home keyframe controls must lie inside every actuator control range")
            self.nominal_ctrl = jnp.array(nominal_ctrl)
            mujoco.mj_forward(self.mj_model, home_data)
            self._default_data = mjx.put_data(self.mj_model, home_data)
        elif self.config.action_mapping == ACTION_MAPPING_MIDPOINT:
            self._default_data = mjx.make_data(self.mjx_model)
        else:
            raise ValueError(f"unknown MJX action mapping {self.config.action_mapping!r}")

        # Ground-settle constants and target.  The target is the clearance the
        # model's home keyframe was authored with — the same semantics as
        # BaseDinoEnv.home_ground_clearance — probed once here through the
        # exact kinematics path the reset uses.  For the home-residual mapping
        # the reset base pose IS that keyframe, so a noise-free reset computes
        # shift == 0.0 bit-for-bit.  For the midpoint mapping the base pose is
        # qpos0, which can hover far above the floor (brachiosaurus: 610 mm);
        # targeting the keyframe clearance settles it onto the ground instead
        # of preserving the drop.
        self._ground_settle = _ground_settle_constants(self.mj_model)
        self._home_ground_clearance = 0.0
        if self._ground_settle is not None:
            keyframe = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_KEY, "home")
            if keyframe < 0 and self.mj_model.nkey > 0:
                keyframe = 0
            if keyframe >= 0:
                target_qpos = jnp.array(self.mj_model.key_qpos[keyframe])
            else:
                target_qpos = self._default_data.qpos
            probed = mjx.kinematics(self.mjx_model, self._default_data.replace(qpos=target_qpos))
            self._home_ground_clearance = float(
                _mjx_lowest_ground_clearance(self._ground_settle, probed.geom_xpos, probed.geom_xmat)
            )

        # Pre-compile step and reset functions.  ``forward_vel_scale`` is
        # passed as a JAX scalar (broadcast via in_axes=None) so the reward
        # ramp can adjust the forward-velocity weight at runtime without
        # forcing a re-trace.
        self._step_single = jax.jit(self._make_step_fn())
        self._reset_single = jax.jit(self._make_reset_fn())
        self._batched_step = jax.jit(jax.vmap(self._step_single, in_axes=(0, 0, 0, None)))
        self._batched_reset = jax.jit(jax.vmap(self._reset_single, in_axes=(0,)))

        # Fused step + auto-reset: runs step and reset in a single vmapped
        # kernel, selecting the reset state only where episodes ended.
        self._batched_step_autoreset = jax.jit(jax.vmap(self._make_fused_step_fn(), in_axes=(0, 0, 0, None)))

    def _make_step_fn(self):
        """Build the single-env step function as a closure over model/config."""
        import jax
        import jax.numpy as jnp
        import mujoco.mjx as mjx

        from .action_filter import apply_low_pass, low_pass_alpha
        from .mjx_utils import scale_action_around_nominal_jax, scale_action_jax
        from .reward_functions import (
            check_nosedive_termination,
            quat_to_forward_2d,
            quat_to_forward_z,
            reward_action_jerk,
            reward_action_smoothness,
            reward_alive,
            reward_angular_velocity_penalty,
            reward_approach_shaping,
            reward_bilateral_support,
            reward_drift_penalty,
            reward_energy,
            reward_foot_load_balance,
            reward_forward_velocity,
            reward_head_clearance,
            reward_heading_alignment,
            reward_height_maintenance,
            reward_lateral_velocity_penalty,
            reward_lean_aware_posture,
            reward_nosedive,
            reward_posture,
            reward_proximity,
            reward_soft_home_pose,
            reward_speed_penalty,
            reward_target_centered_height,
        )

        model = self.mjx_model
        config = self.config
        ctrl_range = self.ctrl_range
        nominal_ctrl = self.nominal_ctrl
        frame_skip = config.frame_skip
        dt = float(self.mj_model.opt.timestep) * frame_skip
        # Trace-time gate: None bakes the legacy unfiltered kernel.
        action_filter_alpha = (
            low_pass_alpha(config.action_filter_cutoff_hz, dt) if config.action_filter_cutoff_hz > 0.0 else None
        )
        # Pre-resolved body-height termination pairs (body_id, z_threshold)
        body_height_checks = tuple(self._termination_body_checks)
        # Pre-resolved site-height termination pairs (site_id, z_threshold)
        site_height_checks = tuple(self._termination_site_checks)
        # Stage 3 success detection
        success_site_ids = self._success_site_ids
        success_threshold = config.success_threshold
        success_bonus = config.reward_weights.get(config.success_bonus_key, 0.0)
        foot_sensor_groups = tuple(
            (
                foot_idx,
                *(config.sensor_foot_aux_indices[position] if position < len(config.sensor_foot_aux_indices) else ()),
            )
            for position, foot_idx in enumerate(config.sensor_foot_indices)
        )
        leg_home_pose_qpos_indices = self._leg_home_pose_qpos_indices
        leg_home_pose_targets = self._leg_home_pose_targets
        neck_posture_qpos_indices = self._neck_posture_qpos_indices
        neck_posture_targets = self._neck_posture_targets
        head_clearance_site_id = self._head_clearance_site_id

        def step_fn(state: EnvState, action, rng, forward_vel_scale):
            """Pure single-environment step function.

            ``forward_vel_scale`` (a scalar JAX array) multiplies the
            ``forward_vel_weight`` reward at runtime so the trainer can
            ramp it without retracing this kernel.
            """
            if action_filter_alpha is not None:
                # Command low-pass, in lockstep with BaseDinoEnv._filter_action:
                # clip into the scaler's box, seed with the first post-reset
                # action, then blend.  Dynamics AND the action-derived reward
                # terms below consume the filtered command.
                clipped = jnp.clip(action, -1.0, 1.0)
                action = jnp.where(
                    state.step_count == 0,
                    clipped,
                    apply_low_pass(state.filtered_action, clipped, action_filter_alpha),
                )
            # Scale action
            if nominal_ctrl is None:
                ctrl = scale_action_jax(action, ctrl_range)
            else:
                ctrl = scale_action_around_nominal_jax(action, ctrl_range, nominal_ctrl)
            data = state.data.replace(ctrl=ctrl)

            # Physics step with frame skip, aggregating contact across the
            # substeps in lockstep with BaseDinoEnv.step: physics runs
            # frame_skip times per control step, and reading only the final
            # substep let a control-clock-locked hop unload between samples
            # (the seed-43 stage-1 bounce measured exactly one unloaded
            # sample in 5).  The carry accumulates the per-foot MIN touch
            # force and an any-substep OR of the height-emulated floor-strike
            # checks; both backends read sensors at the same per-substep
            # phase, so the five snapshots match the SB3 loop's.
            def _foot_force_totals(d):
                totals = []
                for sensor_group in foot_sensor_groups:
                    total = jnp.float32(0.0)
                    for sensor_idx in sensor_group:
                        total = total + d.sensordata[sensor_idx]
                    totals.append(total)
                return jnp.stack(totals)

            def _height_strike(d):
                struck = jnp.bool_(False)
                for body_id, z_thresh in body_height_checks:
                    struck = struck | (d.xpos[body_id, 2] < z_thresh)
                for site_id, z_thresh in site_height_checks:
                    struck = struck | (d.site_xpos[site_id, 2] < z_thresh)
                return struck

            def body_fn(_, carry):
                d, carry_min, carry_struck = carry
                d = mjx.step(model, d)
                return (d, jnp.minimum(carry_min, _foot_force_totals(d)), carry_struck | _height_strike(d))

            data, min_foot_forces, substep_struck = jax.lax.fori_loop(
                0,
                frame_skip,
                body_fn,
                (data, jnp.full((len(foot_sensor_groups),), jnp.inf, dtype=jnp.float32), jnp.bool_(False)),
            )

            # Build observation
            pelvis_id = config.body_ids.get("pelvis", config.body_ids.get("torso", 0))
            pelvis_xpos = data.xpos[pelvis_id]
            target_pos = state.target_pos
            initial_pos_2d = state.initial_pos_2d

            obs = build_mjx_observation(data, target_pos, config)

            # Compute core rewards
            vel_2d = data.qvel[:2]
            # Use direction from agent toward target as forward reference
            target_rel_2d = target_pos[:2] - pelvis_xpos[:2]
            forward_ref = target_rel_2d / (jnp.linalg.norm(target_rel_2d) + 1e-8)

            weights = config.reward_weights
            # Apply runtime ramp scale (default 1.0) to forward_vel_weight.
            forward_vel_weight = weights.get("forward_vel_weight", 1.0) * forward_vel_scale
            r_forward, fwd_vel = reward_forward_velocity(
                vel_2d, forward_ref, config.forward_vel_max, forward_vel_weight
            )
            # Foot contact: the substep-MIN aggregate from the physics loop's
            # carry.  Each foot's per-substep reading is the pad sensor plus
            # any per-toe sensors declared for it, because a touch sensor
            # cannot see geoms on child bodies -- without the aux terms a
            # T-Rex standing on its digits reads as airborne.  MIN over
            # per-substep per-foot SUMS, matching
            # BaseDinoEnv._aggregated_foot_contact_forces exactly.
            foot_contact_threshold = 0.1  # Newtons
            foot_forces = min_foot_forces
            has_foot_contact = jnp.any(foot_forces > foot_contact_threshold)
            r_bilateral_support, bilateral_support_quality = reward_bilateral_support(
                foot_forces,
                weights.get("foot_contact_saturation_force", 100.0),
                weights.get("bilateral_support_weight", 0.0),
            )
            r_foot_load_balance, _ = reward_foot_load_balance(
                foot_forces[:2],
                weights.get("foot_load_balance_weight", 0.0),
                weights.get("foot_load_balance_min_support_force", 0.0),
                weights.get("foot_load_balance_airborne_penalty", 0.0),
            )

            # The opt-in support-conditioned path mirrors the Gymnasium
            # formula exactly.  At its zero default, retain the historical
            # MJX height/contact gate byte-for-byte so stages 2/3 and old
            # configs do not change.
            raw_alive = reward_alive(weights.get("alive_bonus", 0.1))
            support_alive_fraction = weights.get("support_conditioned_alive_fraction", 0.0)
            if support_alive_fraction > 0:
                alive_gate = (1.0 - support_alive_fraction) + support_alive_fraction * bilateral_support_quality
            else:
                height_frac = jnp.clip(
                    (pelvis_xpos[2] - config.healthy_z_range[0])
                    / (config.healthy_z_range[1] - config.healthy_z_range[0]),
                    0.0,
                    1.0,
                )
                foot_contact_gate = weights.get("foot_contact_gate", 0.0)
                alive_gate = jnp.where(
                    foot_contact_gate > 0,
                    height_frac * has_foot_contact.astype(jnp.float32),
                    height_frac,
                )
            r_alive = raw_alive * alive_gate
            r_energy = reward_energy(action, ctrl_range.shape[0], weights.get("energy_penalty_weight", 0.001))

            pelvis_quat = data.sensordata[config.sensor_quat_start : config.sensor_quat_start + 4]
            if config.posture_target_forward_z is not None:
                r_posture, tilt = reward_lean_aware_posture(
                    pelvis_quat,
                    config.max_tilt_angle,
                    weights.get("posture_weight", 0.2),
                    config.posture_target_forward_z,
                )
            else:
                r_posture, tilt = reward_posture(
                    pelvis_quat,
                    config.max_tilt_angle,
                    weights.get("posture_weight", 0.2),
                )

            target_dist = jnp.linalg.norm(target_pos - pelvis_xpos)
            r_approach, _ = reward_approach_shaping(
                target_dist,
                state.prev_target_distance,
                # Canonical key only: TOML keys are canonicalized on the way
                # in, and a legacy first-choice key would shadow it.
                weights.get("approach_weight", 0.0),
                config.forward_vel_max,
                dt,
            )

            total_reward = (
                r_forward + r_alive + r_energy + r_posture + r_approach + r_bilateral_support + r_foot_load_balance
            )

            # Foot-contact bonus: reward having at least one foot on the ground
            foot_contact_w = weights.get("foot_contact_weight", 0.0)
            if foot_contact_w > 0:
                r_foot = foot_contact_w * has_foot_contact.astype(jnp.float32)
                total_reward = total_reward + r_foot

            # Stage-specific reward components (active when weight > 0 in TOML config)
            drift_w = weights.get("drift_penalty_weight", 0.0)
            if drift_w > 0:
                r_drift, _ = reward_drift_penalty(pelvis_xpos[:2], initial_pos_2d, drift_w)
                total_reward = total_reward + r_drift

            speed_w = weights.get("speed_penalty_weight", 0.0)
            if speed_w > 0:
                r_speed, _ = reward_speed_penalty(vel_2d, speed_w, weights.get("speed_penalty_threshold", 0.10))
                total_reward = total_reward + r_speed

            nosedive_w = weights.get("nosedive_weight", 0.0)
            if nosedive_w > 0:
                r_nosedive, _ = reward_nosedive(pelvis_quat, nosedive_w, config.natural_forward_z)
                total_reward = total_reward + r_nosedive

            spin_w = weights.get("spin_penalty_weight", 0.0)
            if spin_w > 0:
                pelvis_angvel = data.sensordata[config.sensor_gyro_start : config.sensor_gyro_start + 3]
                r_spin, _ = reward_angular_velocity_penalty(pelvis_angvel, spin_w)
                total_reward = total_reward + r_spin

            smoothness_w = weights.get("smoothness_weight", 0.0)
            jerk_w = weights.get("action_jerk_weight", 0.0)
            if smoothness_w > 0:
                r_smooth, _ = reward_action_smoothness(action, state.prev_action, ctrl_range.shape[0], smoothness_w)
                total_reward = total_reward + r_smooth
            # Gated independently of smoothness_w: the jerk term exists
            # precisely because the first-difference term cannot see frequency,
            # so a stage may enable either without the other.
            if jerk_w > 0:
                r_jerk, _ = reward_action_jerk(
                    action, state.prev_action, state.prev_prev_action, ctrl_range.shape[0], jerk_w
                )
                total_reward = total_reward + r_jerk

            height_w = weights.get("height_weight", 0.0)
            if height_w > 0:
                target_z = (
                    config.target_standing_z if config.target_standing_z is not None else config.healthy_z_range[1]
                )
                height_target_tolerance = weights.get("height_target_tolerance", 0.0)
                if height_target_tolerance > 0:
                    r_height, _, _ = reward_target_centered_height(
                        pelvis_xpos[2],
                        target_z,
                        height_target_tolerance,
                        height_w,
                    )
                else:
                    # Legacy one-sided gradient saturates at the species
                    # standing height, not the termination ceiling.
                    r_height = reward_height_maintenance(
                        pelvis_xpos[2],
                        config.healthy_z_range[0],
                        target_z,
                        height_w,
                    )
                total_reward = total_reward + r_height

            leg_home_pose_w = weights.get("leg_home_pose_weight", 0.0)
            if leg_home_pose_w > 0:
                leg_joint_positions = jnp.take(data.qpos, jnp.asarray(leg_home_pose_qpos_indices))
                r_leg_home_pose, _, _ = reward_soft_home_pose(
                    leg_joint_positions,
                    leg_home_pose_targets,
                    weights.get("leg_home_pose_tolerance", 0.35),
                    leg_home_pose_w,
                )
                total_reward = total_reward + r_leg_home_pose

            head_clearance_w = weights.get("head_clearance_weight", 0.0)
            if head_clearance_w > 0 and head_clearance_site_id is not None:
                r_head_clearance, _ = reward_head_clearance(
                    data.site_xpos[head_clearance_site_id, 2],
                    weights.get("head_clearance_target", 0.60),
                    weights.get("head_clearance_tolerance", 0.48),
                    head_clearance_w,
                )
                total_reward = total_reward + r_head_clearance

            neck_posture_w = weights.get("neck_posture_weight", 0.0)
            if neck_posture_w > 0:
                neck_joint_positions = jnp.take(data.qpos, jnp.asarray(neck_posture_qpos_indices))
                r_neck_posture, _, _ = reward_soft_home_pose(
                    neck_joint_positions,
                    neck_posture_targets,
                    weights.get("neck_posture_tolerance", 0.35),
                    neck_posture_w,
                )
                total_reward = total_reward + r_neck_posture

            # Tail stability: penalise tail tip angular velocity
            tail_w = weights.get("tail_stability_weight", 0.0)
            if tail_w > 0 and config.sensor_tail_gyro_start is not None:
                tail_angvel = data.sensordata[config.sensor_tail_gyro_start : config.sensor_tail_gyro_start + 3]
                r_tail, _ = reward_angular_velocity_penalty(tail_angvel, tail_w)
                total_reward = total_reward + r_tail

            # Heading alignment: reward facing toward target
            heading_w = weights.get("heading_weight", 0.0)
            if heading_w > 0:
                body_fwd_2d = quat_to_forward_2d(pelvis_quat)
                r_heading, _ = reward_heading_alignment(body_fwd_2d, forward_ref, heading_w)
                total_reward = total_reward + r_heading

            # Lateral velocity penalty: penalise crab-walking
            lateral_w = weights.get("lateral_penalty_weight", 0.0)
            if lateral_w > 0:
                body_fwd_2d_lat = quat_to_forward_2d(pelvis_quat)
                r_lateral, _ = reward_lateral_velocity_penalty(vel_2d, body_fwd_2d_lat, lateral_w)
                total_reward = total_reward + r_lateral

            # Head/claw proximity reward: continuous gradient for final positioning
            # Supports species-specific keys: bite_head_proximity_weight (T-Rex),
            # strike_claw_proximity_weight (raptor), food_head_proximity_weight (brachio).
            proximity_w = weights.get(
                "bite_head_proximity_weight",
                weights.get("strike_claw_proximity_weight", weights.get("food_head_proximity_weight", 0.0)),
            )
            if proximity_w > 0 and success_site_ids and target_pos is not None:
                for sid in success_site_ids:
                    site_dist = jnp.linalg.norm(data.site_xpos[sid] - target_pos)
                    r_prox, _ = reward_proximity(site_dist, config.forward_vel_max, proximity_w)
                    total_reward = total_reward + r_prox

            # Termination
            body_z = pelvis_xpos[2]
            terminated = (body_z < config.healthy_z_range[0]) | (body_z > config.healthy_z_range[1])
            terminated = terminated | (tilt > config.max_tilt_angle)

            # Nosedive termination (forward pitch beyond natural lean)
            forward_z = quat_to_forward_z(pelvis_quat)
            nosedive_threshold = weights.get("nosedive_termination_threshold", 0.5)
            nosedive_terminated, _ = check_nosedive_termination(
                forward_z, config.natural_forward_z, threshold=nosedive_threshold
            )
            terminated = terminated | nosedive_terminated

            # Body/site-height floor-strike termination, evaluated at EVERY
            # physics substep via the fori_loop carry (substep_struck already
            # includes the final substep's state).  A strike that resolves
            # within one control step -- a tail slap that bounces back up --
            # used to be invisible to the boundary sample; the SB3 backend
            # latches the analogous any-substep contact in its step loop.
            terminated = terminated | substep_struck

            # Stage 3 success: proximity-based contact detection
            success = jnp.bool_(False)
            if success_bonus > 0 and success_site_ids:
                for sid in success_site_ids:
                    dist = jnp.linalg.norm(data.site_xpos[sid] - target_pos)
                    success = success | (dist < success_threshold)
                total_reward = jnp.where(success, total_reward + success_bonus, total_reward)

            # Add fall penalty if terminated (but not on success)
            total_reward = jnp.where(terminated & ~success, total_reward + config.fall_penalty, total_reward)
            terminated = terminated | success

            step_count = state.step_count + 1
            truncated = step_count >= config.max_episode_steps

            new_state = EnvState(
                data=data,
                obs=obs,
                step_count=step_count,
                prev_action=action,
                prev_prev_action=state.prev_action,
                filtered_action=(action if action_filter_alpha is not None else state.filtered_action),
                prev_target_distance=target_dist,
                target_pos=target_pos,
                initial_pos_2d=initial_pos_2d,
            )

            return new_state, total_reward, terminated, truncated

        return step_fn

    def _make_reset_fn(self):
        """Build the single-env reset function."""
        import jax
        import jax.numpy as jnp
        import mujoco.mjx as mjx

        model = self.mjx_model
        config = self.config
        default_data = self._default_data
        settle = self._ground_settle
        home_clearance = self._home_ground_clearance

        def reset_fn(rng):
            """Pure single-environment reset function."""
            # Reuse cached default data instead of calling mjx.make_data each time
            data = jax.tree.map(lambda x: x, default_data)

            # Spawn target at random position
            rng, rng_dist, rng_lat = jax.random.split(rng, 3)
            distance = jax.random.uniform(
                rng_dist,
                minval=config.target_distance_range[0],
                maxval=config.target_distance_range[1],
            )
            lateral = jax.random.uniform(
                rng_lat,
                minval=config.target_lateral_range[0],
                maxval=config.target_lateral_range[1],
            )
            target_pos = jnp.array([distance, lateral, config.target_z])

            # Apply reset noise (joint, position, yaw perturbations)
            qpos = data.qpos
            if config.reset_noise_scale > 0:
                rng, rng_joint = jax.random.split(rng)
                joint_noise = jax.random.uniform(
                    rng_joint,
                    (qpos.shape[0] - 7,),
                    minval=-config.reset_noise_scale,
                    maxval=config.reset_noise_scale,
                )
                qpos = qpos.at[7:].add(joint_noise)

            if config.init_qpos_noise > 0:
                rng, rng_xy = jax.random.split(rng)
                xy_noise = jax.random.uniform(
                    rng_xy,
                    (2,),
                    minval=-config.init_qpos_noise,
                    maxval=config.init_qpos_noise,
                )
                qpos = qpos.at[0:2].add(xy_noise)

            if config.init_yaw_noise > 0:
                rng, rng_yaw = jax.random.split(rng)
                yaw_angle = jax.random.uniform(
                    rng_yaw,
                    (),
                    minval=-config.init_yaw_noise,
                    maxval=config.init_yaw_noise,
                )
                half_yaw = yaw_angle / 2.0
                yaw_quat = jnp.array([jnp.cos(half_yaw), 0.0, 0.0, jnp.sin(half_yaw)])
                base_quat = qpos[3:7]
                w1, x1, y1, z1 = yaw_quat
                w2, x2, y2, z2 = base_quat
                new_quat = jnp.array(
                    [
                        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
                    ]
                )
                qpos = qpos.at[3:7].set(new_quat)

            data = data.replace(qpos=qpos)

            if settle is not None:
                # Settle the animal ON the ground.  Joint and yaw jitter change
                # how far the lowest geom sits below the root, so the fixed
                # keyframe root height is wrong for most sampled poses — the
                # defect the Gymnasium reset settled in ca56f6c, which the
                # contact solver otherwise answers with a catapult (up to 19x
                # body weight measured on SB3).  This must come after ALL pose
                # noise.  A kinematics pass is enough to place the geoms; the
                # full forward below then runs once on the settled pose.  The
                # shift is a pure function of the sampled pose: no RNG drawn,
                # and a noise-free reset yields shift == 0.0 exactly because
                # the target was probed through this same code path at init.
                probed = mjx.kinematics(model, data)
                clearance = _mjx_lowest_ground_clearance(settle, probed.geom_xpos, probed.geom_xmat)
                qpos = qpos.at[settle.root_z_index].add(home_clearance - clearance)
                data = data.replace(qpos=qpos)

            # Forward kinematics
            data = mjx.forward(model, data)

            # Build initial observation
            pelvis_id = config.body_ids.get("pelvis", config.body_ids.get("torso", 0))
            pelvis_xpos = data.xpos[pelvis_id]

            obs = build_mjx_observation(data, target_pos, config)

            target_dist = jnp.linalg.norm(target_pos - pelvis_xpos)
            action_dim = model.nu
            state = EnvState(
                data=data,
                obs=obs,
                step_count=jnp.int32(0),
                prev_action=jnp.zeros(action_dim),
                prev_prev_action=jnp.zeros(action_dim),
                # Placeholder until the first step seeds it (step_count==0).
                filtered_action=jnp.zeros(action_dim),
                prev_target_distance=target_dist,
                target_pos=target_pos,
                initial_pos_2d=pelvis_xpos[:2],
            )
            return state

        return reset_fn

    def _make_fused_step_fn(self):
        """Build a single-env function that steps, then auto-resets if done.

        Fusing step + reset into one function means a single ``vmap`` call
        handles both, avoiding a second batched reset kernel and the
        Python-level ``tree_map`` / ``where`` selection.

        The fused step also returns ``final_obs`` — the post-step obs
        BEFORE auto-reset.  Callers that need to bootstrap value at a
        truncation boundary (where the episode ends without termination)
        must use ``final_obs``, not ``new_state.obs``, since the latter
        is the reset-episode obs once auto-reset fires.
        """
        step_fn = self._make_step_fn()
        reset_fn = self._make_reset_fn()

        import jax
        import jax.numpy as jnp

        def fused_step(state: EnvState, action, rng, forward_vel_scale):
            rng_step, rng_reset = jax.random.split(rng)
            new_state, reward, terminated, truncated = step_fn(state, action, rng_step, forward_vel_scale)
            done = terminated | truncated

            # Capture the pre-reset obs so callers can bootstrap at truncation.
            final_obs = new_state.obs

            reset_state = reset_fn(rng_reset)

            # Select reset state where episode ended
            out_state = jax.tree.map(
                lambda s, r: jnp.where(done, r, s) if hasattr(s, "shape") else s,
                new_state,
                reset_state,
            )
            return out_state, reward, terminated, truncated, final_obs

        return fused_step

    def reset(self, rng):
        """Reset all environments.

        Args:
            rng: JAX PRNGKey.

        Returns:
            Batched ``EnvState`` for all environments.
        """
        import jax

        rngs = jax.random.split(rng, self.num_envs)
        return self._batched_reset(rngs)

    def step(
        self,
        states: EnvState,
        actions,
        rng,
        *,
        return_final_obs: bool = False,
        forward_vel_scale: Any = None,
    ):
        """Step all environments with fused auto-reset.

        Uses a single vmapped kernel that computes both the physics step
        and the reset state, selecting the reset only where episodes ended.
        This avoids a separate batched reset call and host-level tree_map.

        Args:
            states: Batched ``EnvState``.
            actions: Actions array of shape ``(num_envs, action_dim)``.
            rng: JAX PRNGKey (used for step and auto-reset).
            return_final_obs: When ``True``, also return the pre-reset
                observation so callers can bootstrap at truncation.
            forward_vel_scale: Optional scalar (``float`` or 0-d JAX array)
                that multiplies the ``forward_vel_weight`` reward
                component at runtime.  ``None`` (default) is treated as
                ``1.0`` — no scaling.  Use this from a curriculum / ramp
                so the policy sees a smoothly-changing reward without
                forcing JIT recompilation.

        Returns:
            ``(new_states, rewards, terminated, truncated)`` by default,
            or ``(new_states, rewards, terminated, truncated, final_obs)``
            when ``return_final_obs=True``.
        """
        import jax
        import jax.numpy as jnp

        scale = jnp.float32(1.0) if forward_vel_scale is None else jnp.asarray(forward_vel_scale, dtype=jnp.float32)
        rngs = jax.random.split(rng, self.num_envs)
        new_states, rewards, terminated, truncated, final_obs = self._batched_step_autoreset(
            states, actions, rngs, scale
        )
        if return_final_obs:
            return new_states, rewards, terminated, truncated, final_obs
        return new_states, rewards, terminated, truncated
