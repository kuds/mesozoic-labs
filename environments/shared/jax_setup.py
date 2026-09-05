"""High-level setup helpers for JAX/MJX training.

Provides high-level setup functions that replace hundreds of lines of
boilerplate.  These helpers wire together the existing library modules
(``config``, ``mjx_env``, ``jax_ppo``, ``jax_eval``, etc.) so that a
notebook, Docker script, or any other caller can set up and run
training in a handful of calls.

Usage::

    from environments.shared.jax_setup import (
        setup_species,
        setup_output_dirs,
        create_env,
        run_stage_evaluation,
    )

    ctx = setup_species("trex", stage=1)
    dirs = setup_output_dirs("trex", stage=1, storage_root="logs")
    env = create_env(ctx, num_envs=2048)
    # ... training via JaxTrainer ...
    selected_eval, final_eval, stage_results, gate_passed, gate_failures = run_stage_evaluation(
        ctx, env, params, network, obs_rms
    )

Usage (Docker)::

    python -m environments.shared.jax_training --species trex --stage 1
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

_logger = logging.getLogger(__name__)

# Camera / display defaults not stored in MJXEnvConfig
_CAMERA_DEFAULTS: dict[str, dict[str, Any]] = {
    "trex": {"camera_track_body": "pelvis", "camera_distance": 3.0},
    "velociraptor": {"camera_track_body": "pelvis", "camera_distance": 2.0},
    "brachiosaurus": {"camera_track_body": "torso", "camera_distance": 5.0},
    "dibothrosuchus": {"camera_track_body": "torso", "camera_distance": 1.6},
}

#: Display names keyed by LEGACY stage number (stage_manifest: an integer
#: reference always means the legacy number).  Stages without one (recovery)
#: take their id, title-cased — see :func:`_stage_display_name`.
_STAGE_NAMES: dict[str, dict[int, str]] = {
    "trex": {1: "Balance", 2: "Locomotion", 3: "Bite"},
    "velociraptor": {1: "Balance", 2: "Locomotion", 3: "Strike"},
    "brachiosaurus": {1: "Balance", 2: "Locomotion", 3: "Food Reach"},
    "dibothrosuchus": {1: "Balance", 2: "Locomotion", 3: "Snap"},
}

#: The manifest id of the stage whose task engages the success machinery
#: (success sites, bite/strike/food-reach bonus) — legacy "stage 3".
_BEHAVIOR_STAGE_ID = "behavior"


def _stage_display_name(species: str, stage: "int | str") -> str:
    from .stage_manifest import load_stage_manifest

    entry = load_stage_manifest(species).resolve(stage)
    if entry.legacy_number is not None:
        return _STAGE_NAMES.get(species, {}).get(entry.legacy_number, f"Stage {entry.legacy_number}")
    return entry.id.title()


def _is_behavior_stage_ref(stage: "int | str") -> bool:
    """Whether a bare stage reference names the behavior stage.

    For callers with no species in hand (:func:`print_eval_summary`): a legacy
    number resolves through the manifest module's universal legacy mapping, a
    semantic id is compared directly.  Replaces the ``stage >= 3`` comparison
    that raised ``TypeError`` for ``"recovery"``.
    """
    from .stage_manifest import LEGACY_STAGE_IDS

    if isinstance(stage, bool):
        return False
    if isinstance(stage, int):
        return LEGACY_STAGE_IDS.get(stage) == _BEHAVIOR_STAGE_ID
    return stage == _BEHAVIOR_STAGE_ID


# ---------------------------------------------------------------------------
# Species context — everything downstream cells need
# ---------------------------------------------------------------------------


@dataclass
class SpeciesContext:
    """All resolved configuration for a species + stage.

    Replaces the notebook's scattered globals (``_cfg``, ``_stage_cfg``,
    ``_jax_kw``, ``reward_cfg``, ``SENSOR_LAYOUT``, ``ROOT_BODY_ID``,
    ``TERMINATION_BODY_CHECKS``, etc.) with a single object.
    """

    species: str
    #: Stage reference as given: a legacy number or a semantic id (see
    #: :attr:`stage_entry` for the resolved manifest entry).
    stage: "int | str"

    # MuJoCo model (CPU, for eval / video)
    mj_model: Any = None

    # MJXDinoEnv config (from registration + TOML merge)
    # Provides: healthy_z_range, max_tilt_angle, natural_forward_z,
    # posture_target_forward_z,
    # sensor_*, termination_body_heights, success_sites, etc.
    env_config: Any = None  # MJXEnvConfig

    # TOML config sections
    stage_config: dict[str, Any] = field(default_factory=dict)
    env_kwargs: dict[str, Any] = field(default_factory=dict)
    jax_kwargs: dict[str, Any] = field(default_factory=dict)
    reward_cfg: dict[str, Any] = field(default_factory=dict)

    # Resolved MuJoCo IDs
    root_body_id: int = 0
    floor_geom_id: int = 0
    termination_body_checks: tuple[tuple[int, float], ...] = ()
    termination_site_checks: tuple[tuple[int, float], ...] = ()
    sensor_tail_gyro_start: int | None = None

    # Sensor layout
    sensor_layout: Any = None  # SensorLayout

    # Action / obs dimensions
    obs_dim: int = 0
    act_dim: int = 0
    ctrl_range: Any = None  # jnp.ndarray
    action_mapping: str = "midpoint/v1"

    # Display / video
    camera_track_body: str = "pelvis"
    camera_distance: float = 3.0
    stage_name: str = ""

    @property
    def stage_entry(self) -> Any:
        """The species-manifest entry this context's stage reference resolves to."""
        from .stage_manifest import load_stage_manifest

        return load_stage_manifest(self.species).resolve(self.stage)

    @property
    def is_behavior_stage(self) -> bool:
        """Whether this is the species' behavior stage (legacy stage 3).

        The stage whose task engages the success machinery — success sites
        and the species' bite/strike/food-reach bonus.  Resolved through the
        manifest so ``"recovery"`` (no legacy number) is simply "not it",
        where the old ``self.stage >= 3`` raised ``TypeError``.
        """
        return bool(self.stage_entry.id == _BEHAVIOR_STAGE_ID)

    @property
    def healthy_z_range(self) -> tuple[float, float]:
        return self.env_config.healthy_z_range if self.env_config else (0.0, 2.0)

    @property
    def max_tilt_angle(self) -> float:
        return self.env_config.max_tilt_angle if self.env_config else 1.047

    @property
    def natural_forward_z(self) -> float:
        if self.env_config:
            return float(self.env_config.natural_forward_z)
        if "natural_forward_z" in self.reward_cfg:
            return float(self.reward_cfg["natural_forward_z"])
        from .mjx_env import _SPECIES_CONFIGS

        return float(_SPECIES_CONFIGS.get(self.species, {}).get("natural_forward_z", 0.0))

    @property
    def posture_target_forward_z(self) -> float | None:
        """Natural posture target, resolved even before ``create_env``.

        Setup and validation callers may inspect the context before creating
        the MJX environment, so fall back to the species registry rather than
        silently selecting the vertical default.
        """
        if self.env_config:
            target = self.env_config.posture_target_forward_z
            return float(target) if target is not None else None
        from .mjx_env import _SPECIES_CONFIGS

        if "posture_target_forward_z" in self.reward_cfg:
            target = self.reward_cfg["posture_target_forward_z"]
            return float(target) if target is not None else None

        target = _SPECIES_CONFIGS.get(self.species, {}).get("posture_target_forward_z")
        if target is not None and "natural_forward_z" in self.reward_cfg:
            target = self.reward_cfg["natural_forward_z"]
        return float(target) if target is not None else None

    @property
    def frame_skip(self) -> int:
        return self.env_config.frame_skip if self.env_config else 5

    @property
    def max_episode_steps(self) -> int:
        return self.env_config.max_episode_steps if self.env_config else 1000

    @property
    def termination_body_heights(self) -> dict[str, float]:
        return dict(self.env_config.termination_body_heights) if self.env_config else {}

    @property
    def termination_site_heights(self) -> dict[str, float]:
        return dict(self.env_config.termination_site_heights) if self.env_config else {}

    @property
    def success_sites(self) -> tuple[str, ...]:
        if self.env_config and self.is_behavior_stage:
            return tuple(self.env_config.success_sites)
        return ()

    @property
    def success_threshold(self) -> float:
        return self.env_config.success_threshold if self.env_config else 0.3

    @property
    def target_body_name(self) -> str:
        return self.env_config.target_body_name if self.env_config else "prey"

    @property
    def forward_vel_max(self) -> float:
        # TOML [env] forward_vel_max overrides the species registration
        if "forward_vel_max" in self.reward_cfg:
            return float(self.reward_cfg["forward_vel_max"])
        return self.env_config.forward_vel_max if self.env_config else 8.0

    @property
    def target_standing_z(self) -> float | None:
        return self.env_config.target_standing_z if self.env_config else None


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------


def setup_species(species: str, stage: "int | str" = 1) -> SpeciesContext:
    """Load all configuration for a species + stage and resolve model IDs.

    This single call replaces notebook cells 6, 8, and 10 (~170 lines).
    It loads the TOML config, imports the species MJX registration,
    loads the MuJoCo model, and resolves all body/site IDs.

    Args:
        species: One of ``"trex"``, ``"velociraptor"``, ``"brachiosaurus"``.
        stage: Curriculum stage reference — a legacy number (1, 2, 3) or a
            semantic stage id (``"recovery"``), resolved through the
            species' stage manifest.

    Returns:
        A :class:`SpeciesContext` with everything resolved and ready.
    """
    import jax.numpy as jnp
    import mujoco

    from .config import load_stage_config
    from .jax_training import _import_species_config
    from .mjx_env import _get_model_path
    from .obs_functions import SensorLayout

    # Import species config to trigger MJX registration
    _import_species_config(species)

    # Load TOML stage config
    stage_config = load_stage_config(species, stage)
    env_kw = stage_config.get("env_kwargs", {})
    jax_kw = stage_config.get("jax_kwargs", {})
    # Canonicalize species-flavoured TOML keys (strike_approach_weight ->
    # approach_weight, etc.) so the JAX reward functions actually see them,
    # and merge them OVER the species-registry default weights — the same
    # registry+TOML merge MJXDinoEnv performs.  Without the registry layer,
    # eval rewards silently dropped species defaults (e.g. trex
    # tail_stability_weight) that training applied.
    from .mjx_env import _SPECIES_CONFIGS as _MJX_SPECIES_CONFIGS
    from .mjx_env import canonicalize_env_kwargs

    _registry_weights = dict(_MJX_SPECIES_CONFIGS.get(species, {}).get("reward_weights", {}))
    reward_cfg = {**_registry_weights, **canonicalize_env_kwargs(env_kw)}

    # Load MuJoCo model (CPU)
    model_path = _get_model_path(species)
    mj_model = mujoco.MjModel.from_xml_path(model_path)

    # Resolve body/geom IDs
    root_body_name = next(
        iter(
            # MJX config registers body_ids as e.g. {"pelvis": 1}
            # but we want the name to look up from the model directly
            k
            for k in _get_registered_body_ids(species)
        ),
        "pelvis",
    )
    root_body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, root_body_name)
    floor_geom_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "floor")

    # Resolve termination checks
    from .mjx_env import _SPECIES_CONFIGS, ACTION_MAPPING_MIDPOINT

    species_kw = _SPECIES_CONFIGS.get(species, {})

    term_body_heights = species_kw.get("termination_body_heights", {})
    termination_body_checks = tuple(
        (mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, name), thresh)
        for name, thresh in term_body_heights.items()
        if mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, name) >= 0
    )
    term_site_heights = species_kw.get("termination_site_heights", {})
    termination_site_checks = tuple(
        (mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, name), thresh)
        for name, thresh in term_site_heights.items()
        if mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, name) >= 0
    )

    # Sensor layout
    sensor_layout = SensorLayout(
        gyro_start=species_kw.get("sensor_gyro_start", 0),
        accel_start=species_kw.get("sensor_accel_start", 3),
        quat_start=species_kw.get("sensor_quat_start", 6),
        foot_indices=species_kw.get("sensor_foot_indices", (10, 11)),
        foot_aux_indices=species_kw.get("sensor_foot_aux_indices", ()),
    )
    sensor_tail_gyro_start = species_kw.get("sensor_tail_gyro_start")

    # Obs/action dimensions (compute from a test forward pass)
    from .obs_functions import build_bipedal_obs

    mj_data = mujoco.MjData(mj_model)
    mujoco.mj_forward(mj_model, mj_data)

    from mujoco import mjx

    test_data = mjx.put_data(mj_model, mj_data)
    test_obs = build_bipedal_obs(
        qpos=test_data.qpos,
        qvel=test_data.qvel,
        sensordata=test_data.sensordata,
        pelvis_xpos=test_data.xpos[root_body_id],
        target_pos=jnp.zeros(3),
        sensor_layout=sensor_layout,
    )

    # Camera / display
    cam = _CAMERA_DEFAULTS.get(species, {})
    stage_name = _stage_display_name(species, stage)

    ctx = SpeciesContext(
        species=species,
        stage=stage,
        mj_model=mj_model,
        stage_config=stage_config,
        env_kwargs=env_kw,
        jax_kwargs=jax_kw,
        reward_cfg=reward_cfg,
        root_body_id=root_body_id,
        floor_geom_id=floor_geom_id,
        termination_body_checks=termination_body_checks,
        termination_site_checks=termination_site_checks,
        sensor_tail_gyro_start=sensor_tail_gyro_start,
        sensor_layout=sensor_layout,
        obs_dim=int(test_obs.shape[0]),
        act_dim=mj_model.nu,
        ctrl_range=jnp.array(mj_model.actuator_ctrlrange),
        action_mapping=species_kw.get("action_mapping", ACTION_MAPPING_MIDPOINT),
        camera_track_body=cam.get("camera_track_body", root_body_name),
        camera_distance=cam.get("camera_distance", 3.0),
        stage_name=stage_name,
    )

    _logger.info(
        "%s stage %s (%s): obs=%d act=%d root=%s(%d)",
        species,
        stage,
        stage_name,
        ctx.obs_dim,
        ctx.act_dim,
        root_body_name,
        root_body_id,
    )

    return ctx


def _get_registered_body_ids(species: str) -> dict[str, int]:
    """Get registered body_ids for a species from MJX config."""
    from .mjx_env import _SPECIES_CONFIGS

    result: dict[str, int] = _SPECIES_CONFIGS.get(species, {}).get("body_ids", {"pelvis": 1})
    return result


def setup_output_dirs(
    species: str,
    stage: int | str,
    storage_root: str | Path = "logs",
    timestamp: str | None = None,
) -> dict[str, Path]:
    """Create the output directory structure for a training run.

    Args:
        species: Species name.
        stage: Curriculum stage reference — legacy integer or semantic id.
        storage_root: Root directory for all logs.
        timestamp: Optional timestamp string (defaults to now).

    Returns:
        Dict with keys ``run_dir``, ``stage_dir``, ``model_dir``.
    """
    from .stage_manifest import stage_dirname

    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    root = Path(storage_root)
    run_dir = root / species / "jax" / timestamp
    # NN_id generation (stage_manifest.stage_dirname, adopted 2026-08-20).
    # A literal f"stage{stage}" here would reintroduce the renumbering
    # hazard for legacy integers and mint "stagerecovery" for semantic
    # ids; readers accept both generations (stage_dir_candidates), so
    # only the writer decides which one new runs get.
    stage_dir = run_dir / stage_dirname(species, stage)
    model_dir = stage_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    return {"run_dir": run_dir, "stage_dir": stage_dir, "model_dir": model_dir}


def build_env_kwargs(
    reward_cfg: dict[str, Any],
    jax_kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Merge TOML [env] reward config with [jax] overrides.

    Applies fall_penalty and reset noise overrides from the ``[jax]``
    section, mirroring the logic in ``jax_training.py``.  Uses direct
    assignment — setdefault is a no-op when [env] already defines the key.

    Args:
        reward_cfg: Full TOML ``[env]`` section (reward weights + env params).
        jax_kwargs: TOML ``[jax]`` section.

    Returns:
        Merged env_kwargs dict suitable for ``MJXDinoEnv(..., env_kwargs=)``.
    """
    env_kwargs = dict(reward_cfg)
    if "fall_penalty" in jax_kwargs:
        env_kwargs["fall_penalty"] = jax_kwargs["fall_penalty"]
    for key in ("reset_noise_scale", "init_qpos_noise", "init_yaw_noise"):
        if key in jax_kwargs:
            env_kwargs[key] = jax_kwargs[key]
    return env_kwargs


def make_network(ctx: SpeciesContext) -> Any:
    """Build the actor-critic sized by the stage's ``[jax.policy_kwargs] net_arch``.

    The ONE network factory for a species context: the training path
    (:func:`~environments.shared.jax_training.train_jax`) and every
    load/eval path that rebuilds the module to apply saved params must
    agree on the hidden widths, or the checkpoint's params will not fit.
    Both read the same ``jax_kwargs`` through
    :func:`~environments.shared.jax_curriculum.network_hidden_dims`.
    """
    from .jax_curriculum import network_hidden_dims
    from .jax_ppo import make_actor_critic

    return make_actor_critic(ctx.act_dim, hidden_dims=network_hidden_dims(ctx.jax_kwargs))


def create_env(
    ctx: SpeciesContext,
    num_envs: int = 2048,
) -> Any:
    """Create and return a ``MJXDinoEnv`` from a species context.

    Merges environment kwargs automatically. The context's CPU model is kept
    at its XML-authored options because checkpoint evaluation and video use
    that instance, while ``MJXDinoEnv`` loads a separate training model.
    Mutating only the CPU copy would make evaluation physics diverge from
    training.

    Args:
        ctx: Species context from :func:`setup_species`.
        num_envs: Number of parallel environments.

    Returns:
        ``MJXDinoEnv`` instance.
    """
    from .mjx_env import MJXDinoEnv

    env_kwargs = build_env_kwargs(ctx.reward_cfg, ctx.jax_kwargs)

    env = MJXDinoEnv(
        ctx.species,
        # mjx_env still annotates stage as int; it only records the value.
        stage=ctx.stage,  # type: ignore[arg-type]
        num_envs=num_envs,
        env_kwargs=env_kwargs,
    )
    ctx.env_config = env.config
    return env


# ---------------------------------------------------------------------------
# Observation / action helpers
# ---------------------------------------------------------------------------


def make_obs_fn(ctx: SpeciesContext):
    """Create an observation function bound to the species context.

    Returns a function ``get_obs(data) -> obs_array`` suitable for
    ``evaluate_policy_cpu`` and ``record_training_video``.

    The observation embeds the target direction/distance, so it must point
    at the same target the evaluation's success detection uses — the
    model's target body (prey/food).  A hardcoded origin target used to
    flip the perceived target direction backwards as the agent walked
    away, so stage-3 gates and videos evaluated the wrong task.
    """
    import jax.numpy as jnp
    import mujoco

    from .obs_functions import build_bipedal_obs

    root_body_id = ctx.root_body_id
    sensor_layout = ctx.sensor_layout
    target_body_id = mujoco.mj_name2id(ctx.mj_model, mujoco.mjtObj.mjOBJ_BODY, ctx.target_body_name)

    def get_obs(data):
        if target_body_id >= 0:
            target_pos = data.xpos[target_body_id]
        else:
            target_pos = jnp.zeros(3)
        return build_bipedal_obs(
            qpos=data.qpos,
            qvel=data.qvel,
            sensordata=data.sensordata,
            pelvis_xpos=data.xpos[root_body_id],
            target_pos=target_pos,
            sensor_layout=sensor_layout,
        )

    return get_obs


def make_scale_action_fn(ctx: SpeciesContext):
    """Create an action scaling function bound to the species context."""
    import jax.numpy as jnp
    import mujoco

    from .mjx_env import ACTION_MAPPING_HOME_KEYFRAME_RESIDUAL, ACTION_MAPPING_MIDPOINT
    from .mjx_utils import scale_action_around_nominal_jax, scale_action_jax

    ctrl_range = ctx.ctrl_range
    action_mapping = ctx.action_mapping

    if action_mapping == ACTION_MAPPING_HOME_KEYFRAME_RESIDUAL:
        home_keyframe_id = mujoco.mj_name2id(ctx.mj_model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if home_keyframe_id < 0:
            raise ValueError("home-keyframe residual action mapping requires a keyframe named 'home'")
        nominal_ctrl = jnp.array(ctx.mj_model.key_ctrl[home_keyframe_id])
        if bool(jnp.any((nominal_ctrl < ctrl_range[:, 0]) | (nominal_ctrl > ctrl_range[:, 1]))):
            raise ValueError("home keyframe controls must lie inside every actuator control range")

        def scale_home_residual_action(action):
            return scale_action_around_nominal_jax(action, ctrl_range, nominal_ctrl)

        return scale_home_residual_action

    if action_mapping != ACTION_MAPPING_MIDPOINT:
        raise ValueError(f"unknown JAX action mapping {action_mapping!r}")

    def scale_midpoint_action(action):
        return scale_action_jax(action, ctrl_range)

    return scale_midpoint_action


def make_reward_fns(ctx: SpeciesContext):
    """Create reward and termination functions bound to the species context.

    Returns:
        ``(compute_reward, compute_reward_detailed, is_terminated)`` tuple.

    ``compute_reward`` accepts optional per-step keyword arguments
    (``target_pos``, ``prev_target_distance``, ``prev_action``,
    ``forward_ref_2d``, ``success_site_positions``) which
    ``evaluate_policy_cpu`` supplies so eval episode rewards include the
    same approach/proximity/success/fall components as training.
    """
    import jax.numpy as jnp
    import mujoco

    from .jax_reward_termination import compute_reward_components, compute_total_reward
    from .jax_reward_termination import is_terminated as is_terminated_fn
    from .mjx_env import _SPECIES_CONFIGS as _MJX_SPECIES_CONFIGS
    from .mjx_env import _resolve_home_pose_joints

    # Success bonus resolves through the species' bonus key (bite_bonus /
    # strike_bonus / food_reach_bonus), matching MJXDinoEnv.step; the
    # machinery only engages when success sites exist (stage 3).
    success_bonus_key = _MJX_SPECIES_CONFIGS.get(ctx.species, {}).get("success_bonus_key", "")
    success_bonus = float(ctx.reward_cfg.get(success_bonus_key, 0.0)) if success_bonus_key else 0.0
    species_reward_geometry = ctx.env_config or _MJX_SPECIES_CONFIGS.get(ctx.species, {})

    def geometry_value(name: str, default: Any = None) -> Any:
        if isinstance(species_reward_geometry, dict):
            return species_reward_geometry.get(name, default)
        return getattr(species_reward_geometry, name, default)

    leg_home_pose_qpos_indices, leg_home_pose_targets = _resolve_home_pose_joints(
        ctx.mj_model,
        tuple(geometry_value("leg_home_pose_joint_names", ())),
    )
    neck_posture_qpos_indices, neck_posture_targets = _resolve_home_pose_joints(
        ctx.mj_model,
        tuple(geometry_value("neck_posture_joint_names", ())),
    )
    tail_home_pose_qpos_indices, tail_home_pose_targets = _resolve_home_pose_joints(
        ctx.mj_model,
        tuple(geometry_value("tail_home_pose_joint_names", ())),
    )
    explicit_tail_targets = tuple(geometry_value("tail_home_pose_targets", ()))
    if explicit_tail_targets:
        # Statue-derived settled pose overrides the keyframe (see MJXEnvConfig).
        tail_home_pose_targets = tuple(float(v) for v in explicit_tail_targets)
    head_clearance_site_id = None
    head_clearance_site = geometry_value("head_clearance_site")
    if head_clearance_site is not None:
        resolved_head_site_id = mujoco.mj_name2id(ctx.mj_model, mujoco.mjtObj.mjOBJ_SITE, head_clearance_site)
        if resolved_head_site_id < 0:
            raise ValueError(f"head-clearance site {head_clearance_site!r} does not exist")
        head_clearance_site_id = resolved_head_site_id

    foot_sensor_groups = tuple(
        (
            foot_index,
            *(
                ctx.sensor_layout.foot_aux_indices[position]
                if position < len(ctx.sensor_layout.foot_aux_indices)
                else ()
            ),
        )
        for position, foot_index in enumerate(ctx.sensor_layout.foot_indices)
    )

    reward_kw = dict(
        root_body_id=ctx.root_body_id,
        healthy_z_min=ctx.healthy_z_range[0],
        healthy_z_max=ctx.healthy_z_range[1],
        target_standing_z=ctx.target_standing_z,
        max_tilt_angle=ctx.max_tilt_angle,
        natural_forward_z=ctx.natural_forward_z,
        posture_target_forward_z=ctx.posture_target_forward_z,
        n_actuators=ctx.mj_model.nu,
        sensor_quat_start=ctx.sensor_layout.quat_start,
        sensor_gyro_start=ctx.sensor_layout.gyro_start,
        # Flattened: this feeds "is any foot in contact", so pad and digit
        # sensors are interchangeable here.  Per-foot totals (which must stay
        # grouped) are built separately for the eval diagnostics below.
        foot_indices=(
            tuple(ctx.sensor_layout.foot_indices)
            + tuple(index for group in ctx.sensor_layout.foot_aux_indices for index in group)
        ),
        foot_sensor_groups=foot_sensor_groups,
        leg_home_pose_qpos_indices=leg_home_pose_qpos_indices,
        leg_home_pose_targets=jnp.asarray(leg_home_pose_targets) if leg_home_pose_targets else None,
        neck_posture_qpos_indices=neck_posture_qpos_indices,
        neck_posture_targets=jnp.asarray(neck_posture_targets) if neck_posture_targets else None,
        tail_home_pose_qpos_indices=tail_home_pose_qpos_indices,
        tail_home_pose_targets=jnp.asarray(tail_home_pose_targets) if tail_home_pose_targets else None,
        head_clearance_site_id=head_clearance_site_id,
        sensor_tail_gyro_start=ctx.sensor_tail_gyro_start,
        forward_vel_max=ctx.forward_vel_max,
        dt=float(ctx.mj_model.opt.timestep) * ctx.frame_skip,
        fall_penalty=float(ctx.jax_kwargs.get("fall_penalty", ctx.reward_cfg.get("fall_penalty", 0.0))),
        success_threshold=ctx.success_threshold,
        success_bonus=success_bonus if ctx.is_behavior_stage else 0.0,
    )

    nosedive_threshold = ctx.reward_cfg.get("nosedive_termination_threshold", 0.5)

    termination_kw = dict(
        root_body_id=ctx.root_body_id,
        healthy_z_range=ctx.healthy_z_range,
        max_tilt_angle=ctx.max_tilt_angle,
        natural_forward_z=ctx.natural_forward_z,
        nosedive_threshold=nosedive_threshold,
        body_height_checks=ctx.termination_body_checks,
        site_height_checks=ctx.termination_site_checks,
        sensor_quat_start=ctx.sensor_layout.quat_start,
    )

    def compute_reward(data, action, reward_cfg, **step_kwargs):
        return compute_total_reward(data, action, reward_cfg, **reward_kw, **step_kwargs)

    def compute_reward_detailed(data, action, reward_cfg, **step_kwargs):
        # compute_reward_components takes a subset of the per-step kwargs.
        # Both action lags are forwarded so the eval component panel's
        # action_jerk uses the real second lag rather than a zero stand-in.
        allowed = {
            k: v
            for k, v in step_kwargs.items()
            if k in ("prev_action", "prev_prev_action", "forward_ref_2d", "initial_pos_2d", "aggregated_foot_forces")
        }
        detail_kw = {
            k: v
            for k, v in reward_kw.items()
            if k not in ("forward_vel_max", "dt", "fall_penalty", "success_threshold", "success_bonus")
        }
        return compute_reward_components(data, action, reward_cfg, **detail_kw, **allowed)

    def is_terminated(data):
        return is_terminated_fn(data, **termination_kw)

    return compute_reward, compute_reward_detailed, is_terminated


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def run_stage_evaluation(
    ctx: SpeciesContext,
    env: Any,
    params: Any,
    network: Any,
    obs_rms: Any,
    *,
    n_episodes: int = 30,
    best_params: Any = None,
    best_reward: float = -float("inf"),
    best_update: int = -1,
    total_steps: int = 0,
    elapsed: float = 0.0,
    eval_seed: int = 42,
) -> tuple[Any, Any, dict[str, Any], bool, list[str]]:
    """Run stage gate evaluation and build stage results dict.

    Replaces notebook cell 25 (~127 lines) with a single call.

    Args:
        ctx: Species context.
        env: ``MJXDinoEnv`` instance.
        params: Current network parameters.
        network: Flax ActorCritic module.
        obs_rms: Running mean/std for observation normalization.
        n_episodes: Number of evaluation episodes.
        best_params: Best parameters from training (used if available).
        best_reward: Best reward achieved during training.
        best_update: Update index of best reward.
        total_steps: Total environment steps completed.
        elapsed: Total training time in seconds.
        eval_seed: Seed for evaluation reset noise (reproducible gates).

    Returns:
        ``(selected_eval_results, final_eval_results, stage_results,
        gate_passed, gate_failures)`` tuple.
    """
    import jax
    import numpy as np

    from .config import load_stage_config
    from .curriculum.gate_schema import validate_gate_config
    from .jax_curriculum import jax_gate_thresholds
    from .jax_eval import EvalConfig, check_stage_gate_for_config, evaluate_policy_cpu
    from .jax_normalization import normalize_obs

    # Bind every evaluation helper to the actual instantiated training
    # environment.  Besides reset noise, this carries registered geometry and
    # target_standing_z used by the bounded height/stance rewards.
    ctx.env_config = env.config
    get_obs = make_obs_fn(ctx)
    scale_action = make_scale_action_fn(ctx)
    compute_reward, compute_reward_detailed, _ = make_reward_fns(ctx)

    final_params = jax.device_get(params)
    selected_params = best_params if best_params is not None else final_params
    target_distance_range = tuple(env.config.target_distance_range)
    target_lateral_range = tuple(env.config.target_lateral_range)

    eval_config = EvalConfig(
        n_episodes=n_episodes,
        max_episode_steps=ctx.max_episode_steps,
        frame_skip=ctx.frame_skip,
        healthy_z_range=ctx.healthy_z_range,
        max_tilt_angle=ctx.max_tilt_angle,
        natural_forward_z=ctx.natural_forward_z,
        posture_target_forward_z=ctx.posture_target_forward_z,
        # TOML can tighten the nosedive threshold (e.g. trex stage 1);
        # without this the eval terminated later than training.
        nosedive_threshold=ctx.reward_cfg.get("nosedive_termination_threshold", 0.5),
        termination_body_heights=ctx.termination_body_heights,
        termination_site_heights=ctx.termination_site_heights,
        success_sites=ctx.success_sites,
        success_threshold=ctx.success_threshold,
        target_body=ctx.target_body_name,
        root_body_id=ctx.root_body_id,
        sensor_quat_start=ctx.sensor_layout.quat_start,
        sensor_gyro_start=ctx.sensor_layout.gyro_start,
        action_mapping=ctx.action_mapping,
        # Drive the same filtered plant the training kernel does.
        action_filter_cutoff_hz=float(getattr(env.config, "action_filter_cutoff_hz", 0.0)),
        # Evaluate on the same joint-angle reset distribution used by the
        # instantiated training environment.  This includes the effective
        # [jax] override after registry/[env]/[jax] merging.
        reset_noise_scale=float(env.config.reset_noise_scale),
        init_qpos_noise=float(env.config.init_qpos_noise),
        init_yaw_noise=float(env.config.init_yaw_noise),
        target_distance_range=target_distance_range,
        target_lateral_range=target_lateral_range,
        target_z=float(env.config.target_z),
        forward_vel_max=ctx.forward_vel_max,
        target_standing_z=(ctx.target_standing_z if ctx.target_standing_z is not None else 0.90),
        # Scheduled pushes (recovery stage): thread the instantiated env's
        # perturbation_* fields one-to-one so the eval applies the same
        # pushes training did.  A multiple of 0 (every non-recovery stage,
        # and any config that does not declare pushes) keeps the eval
        # push-free; the defaults are MJXEnvConfig's.
        perturbation_capture_velocity_multiple=float(
            getattr(env.config, "perturbation_capture_velocity_multiple", 0.0)
        ),
        perturbation_interval=float(getattr(env.config, "perturbation_interval", 2.0)),
        perturbation_jitter=float(getattr(env.config, "perturbation_jitter", 0.5)),
        perturbation_duration=float(getattr(env.config, "perturbation_duration", 0.20)),
        perturbation_direction=str(getattr(env.config, "perturbation_direction", "uniform_horizontal")),
        seed=eval_seed,
    )

    foot_indices = tuple(ctx.sensor_layout.foot_indices)
    foot_aux_indices = tuple(ctx.sensor_layout.foot_aux_indices)

    selected_eval_results = evaluate_policy_cpu(
        ctx.mj_model,
        selected_params,
        network,
        obs_rms,
        get_obs_fn=get_obs,
        normalize_obs_fn=normalize_obs,
        scale_action_fn=scale_action,
        reward_fn=compute_reward,
        reward_cfg=ctx.reward_cfg,
        config=eval_config,
        foot_sensor_indices=foot_indices,
        foot_aux_indices=foot_aux_indices,
        reward_components_fn=compute_reward_detailed,
    )
    final_eval_results = (
        selected_eval_results
        if best_params is None
        else evaluate_policy_cpu(
            ctx.mj_model,
            final_params,
            network,
            obs_rms,
            get_obs_fn=get_obs,
            normalize_obs_fn=normalize_obs,
            scale_action_fn=scale_action,
            reward_fn=compute_reward,
            reward_cfg=ctx.reward_cfg,
            config=eval_config,
            foot_sensor_indices=foot_indices,
            foot_aux_indices=foot_aux_indices,
            reward_components_fn=compute_reward_detailed,
        )
    )

    # Gate check — dispatched on the stage's declared gate_kind, so the
    # criteria checked here are the ones the config actually names.
    #
    # This used to read four fixed thresholds and call check_stage_gate
    # directly, which knows nothing about gate_kind. For a stance_quality/v1
    # stage that certified on whichever of the four happened to be set: on
    # trex stage 1 that is min_avg_reward = 1950 alone, since
    # min_avg_episode_length was retired when the stance gate replaced it.
    # Both the zero-action statue (3271.8) and the chattering policy the gate
    # exists to reject (2133.4) cleared it — and the verdict is written into
    # publication_gate_passed below.
    stage_config = dict(load_stage_config(ctx.species, ctx.stage))
    # Apply the stage's [curriculum.jax] threshold overrides (additive: absent,
    # the shared thresholds are unchanged) so the JAX eval judges the same bar
    # the JAX curriculum does.  Validate the declared table first so a
    # malformed override table fails with the schema's message.
    validate_gate_config(ctx.stage, stage_config.get("curriculum_kwargs", {}), advancement_enabled=True)
    stage_config["curriculum_kwargs"] = jax_gate_thresholds(ctx.stage, stage_config, species=ctx.species)
    gate_passed, gate_failures = check_stage_gate_for_config(selected_eval_results, stage_config)

    num_envs = env.num_envs
    rollout_len = ctx.jax_kwargs.get("rollout_len", 64)

    stage_results = {
        "stage": ctx.stage,
        "name": ctx.stage_name,
        "description": f"JAX/MJX PPO stage {ctx.stage}",
        "timesteps": total_steps,
        "duration_seconds": elapsed,
        "mean_reward": round(final_eval_results.mean_reward, 2),
        "std_reward": round(final_eval_results.std_reward, 2),
        "mean_episode_length": round(final_eval_results.mean_length, 1),
        "std_episode_length": round(final_eval_results.std_length, 1),
        "mean_forward_vel": round(final_eval_results.mean_forward_vel, 3),
        "std_forward_vel": round(
            float(np.std(final_eval_results.forward_vels)) if final_eval_results.forward_vels else 0.0,
            3,
        ),
        "mean_distance_traveled": round(final_eval_results.mean_distance, 2),
        "mean_success_rate": round(final_eval_results.mean_success_rate, 3),
        "best_eval_reward": round(selected_eval_results.mean_reward, 2),
        "best_eval_std": round(selected_eval_results.std_reward, 2),
        "best_eval_length": round(selected_eval_results.mean_length, 1),
        "best_eval_timestep": best_update * rollout_len * num_envs if best_update >= 0 else None,
        "selection_training_return": round(best_reward, 4) if np.isfinite(best_reward) else None,
        "selection_training_update": best_update if best_update >= 0 else None,
        "sim_dt": ctx.mj_model.opt.timestep * ctx.frame_skip,
        "evaluation_reset_noise_scale": eval_config.reset_noise_scale,
        "evaluation_init_qpos_noise": eval_config.init_qpos_noise,
        "evaluation_init_yaw_noise": eval_config.init_yaw_noise,
        "evaluation_target_distance_range": list(target_distance_range),
        "evaluation_target_lateral_range": list(target_lateral_range),
        "evaluation_target_z": eval_config.target_z,
        "evaluation_seed": eval_config.seed,
        "best_model_reward": round(selected_eval_results.mean_reward, 2),
        "best_model_std_reward": round(selected_eval_results.std_reward, 2),
        "best_model_length": round(selected_eval_results.mean_length, 1),
        "best_model_std_length": round(selected_eval_results.std_length, 1),
        "best_model_fwd_vel": round(selected_eval_results.mean_forward_vel, 3),
        "best_model_std_fwd_vel": round(
            float(np.std(selected_eval_results.forward_vels)) if selected_eval_results.forward_vels else 0.0,
            3,
        ),
        "best_model_distance": round(selected_eval_results.mean_distance, 2),
        "best_model_success_rate": round(selected_eval_results.mean_success_rate, 3),
        "best_model_n_episodes": n_episodes,
        "gate_passed": gate_passed,
        "publication_gate_passed": gate_passed,
        # Carried on the results dict, not only returned alongside it, so the
        # REASONS reach the stage summary and stage_result.json rather than
        # living only in this function's return value and the console.
        "gate_failures": gate_failures,
    }

    return selected_eval_results, final_eval_results, stage_results, gate_passed, gate_failures


def print_species_summary(ctx: SpeciesContext) -> None:
    """Print a human-readable summary of the species context."""
    print(f"Species:     {ctx.species}")
    print(f"Stage:       {ctx.stage} ({ctx.stage_name})")
    print(f"Obs dim:     {ctx.obs_dim}")
    print(f"Act dim:     {ctx.act_dim}")
    print(f"Root body:   id={ctx.root_body_id}")
    print(f"Frame skip:  {ctx.frame_skip}")
    print(f"Max steps:   {ctx.max_episode_steps}")
    print(f"Healthy z:   {ctx.healthy_z_range}")
    print(f"Body checks: {len(ctx.termination_body_checks)}")
    print(f"Site checks: {len(ctx.termination_site_checks)}")


def print_eval_summary(
    eval_results: Any,
    gate_passed: bool,
    gate_failures: list[str],
    stage: "int | str",
) -> None:
    """Print evaluation results and gate status."""
    import numpy as np

    print(f"\nEvaluation results ({len(eval_results.rewards)} episodes):")
    print(f"  Mean reward:   {eval_results.mean_reward:.2f} +/- {eval_results.std_reward:.2f}")
    print(f"  Mean length:   {eval_results.mean_length:.1f} +/- {eval_results.std_length:.1f}")
    print(f"  Mean fwd vel:  {eval_results.mean_forward_vel:.3f} m/s")
    print(f"  Mean distance: {eval_results.mean_distance:.2f} m")
    print(f"  Mean tilt:     {np.degrees(eval_results.mean_tilt):.1f} deg")
    print(f"  Mean pelvis H: {eval_results.mean_height:.3f} m")
    if _is_behavior_stage_ref(stage):
        print(f"  Success rate:  {100.0 * eval_results.mean_success_rate:.0f}%")

    if gate_failures:
        print(f"\n*** STAGE {stage} GATE NOT PASSED ***")
        for f in gate_failures:
            print(f"  - {f}")
    else:
        print(f"\n*** STAGE {stage} GATE PASSED ***")
