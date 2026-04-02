"""Notebook and Docker helpers for JAX/MJX training.

Provides high-level setup functions that replace hundreds of lines of
inline notebook code.  These helpers wire together the existing library
modules (``config``, ``mjx_env``, ``jax_ppo``, ``jax_eval``, etc.) so
that a Colab notebook or Docker script can set up and run training in
a handful of calls.

Usage (notebook)::

    from environments.shared.jax_notebook import (
        setup_species,
        setup_training,
        setup_output_dirs,
        run_stage_evaluation,
    )

    ctx = setup_species("trex", stage=1)
    dirs = setup_output_dirs("trex", stage=1, storage_root="logs")
    env, network, params, obs_rms, optimizer, ppo_cfg, train_cfg = setup_training(ctx)
    # ... training via JaxTrainer ...
    eval_results, stage_results = run_stage_evaluation(ctx, env, params, network, obs_rms)

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
}

_STAGE_NAMES: dict[str, dict[int, str]] = {
    "trex": {1: "Balance", 2: "Locomotion", 3: "Bite"},
    "velociraptor": {1: "Balance", 2: "Locomotion", 3: "Strike"},
    "brachiosaurus": {1: "Balance", 2: "Locomotion", 3: "Food Reach"},
}


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
    stage: int

    # MuJoCo model (CPU, for eval / video)
    mj_model: Any = None

    # MJXDinoEnv config (from registration + TOML merge)
    # Provides: healthy_z_range, max_tilt_angle, natural_forward_z,
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

    # Display / video
    camera_track_body: str = "pelvis"
    camera_distance: float = 3.0
    stage_name: str = ""

    @property
    def healthy_z_range(self) -> tuple[float, float]:
        return self.env_config.healthy_z_range if self.env_config else (0.0, 2.0)

    @property
    def max_tilt_angle(self) -> float:
        return self.env_config.max_tilt_angle if self.env_config else 1.047

    @property
    def natural_forward_z(self) -> float:
        return self.env_config.natural_forward_z if self.env_config else 0.0

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
        if self.env_config and self.stage >= 3:
            return tuple(self.env_config.success_sites)
        return ()

    @property
    def success_threshold(self) -> float:
        return self.env_config.success_threshold if self.env_config else 0.3


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------


def setup_species(species: str, stage: int = 1) -> SpeciesContext:
    """Load all configuration for a species + stage and resolve model IDs.

    This single call replaces notebook cells 6, 8, and 10 (~170 lines).
    It loads the TOML config, imports the species MJX registration,
    loads the MuJoCo model, and resolves all body/site IDs.

    Args:
        species: One of ``"trex"``, ``"velociraptor"``, ``"brachiosaurus"``.
        stage: Curriculum stage (1, 2, or 3).

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
    reward_cfg = dict(env_kw)

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
    from .mjx_env import _SPECIES_CONFIGS

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
    stage_name = _STAGE_NAMES.get(species, {}).get(stage, f"Stage {stage}")

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
        camera_track_body=cam.get("camera_track_body", root_body_name),
        camera_distance=cam.get("camera_distance", 3.0),
        stage_name=stage_name,
    )

    _logger.info(
        "%s stage %d (%s): obs=%d act=%d root=%s(%d)",
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
    stage: int,
    storage_root: str | Path = "logs",
    timestamp: str | None = None,
) -> dict[str, Path]:
    """Create the output directory structure for a training run.

    Args:
        species: Species name.
        stage: Curriculum stage.
        storage_root: Root directory for all logs.
        timestamp: Optional timestamp string (defaults to now).

    Returns:
        Dict with keys ``run_dir``, ``stage_dir``, ``model_dir``.
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    root = Path(storage_root)
    run_dir = root / species / "jax" / timestamp
    stage_dir = run_dir / f"stage{stage}"
    model_dir = stage_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    return {"run_dir": run_dir, "stage_dir": stage_dir, "model_dir": model_dir}


def build_env_kwargs(
    reward_cfg: dict[str, Any],
    jax_kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Merge TOML [env] reward config with [jax] overrides.

    Applies fall_penalty and reset noise overrides from the ``[jax]``
    section, mirroring the logic in ``jax_training.py``.

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


def apply_mjx_solver_tuning(mj_model: Any) -> None:
    """Apply standard MJX solver tuning to a MuJoCo model.

    Should be called *before* creating ``MJXDinoEnv`` (which internalises
    the model).
    """
    import mujoco

    mj_model.opt.solver = mujoco.mjtSolver.mjSOL_CG
    mj_model.opt.iterations = 4
    mj_model.opt.ls_iterations = 4
    mj_model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_FILTERPARENT


def create_env(
    ctx: SpeciesContext,
    num_envs: int = 2048,
) -> Any:
    """Create and return a ``MJXDinoEnv`` from a species context.

    Applies solver tuning and merges env kwargs automatically.

    Args:
        ctx: Species context from :func:`setup_species`.
        num_envs: Number of parallel environments.

    Returns:
        ``MJXDinoEnv`` instance.
    """
    from .mjx_env import MJXDinoEnv

    apply_mjx_solver_tuning(ctx.mj_model)
    env_kwargs = build_env_kwargs(ctx.reward_cfg, ctx.jax_kwargs)

    env = MJXDinoEnv(
        ctx.species,
        stage=ctx.stage,
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
    """
    import jax.numpy as jnp

    from .obs_functions import build_bipedal_obs

    root_body_id = ctx.root_body_id
    sensor_layout = ctx.sensor_layout

    def get_obs(data):
        return build_bipedal_obs(
            qpos=data.qpos,
            qvel=data.qvel,
            sensordata=data.sensordata,
            pelvis_xpos=data.xpos[root_body_id],
            target_pos=jnp.zeros(3),
            sensor_layout=sensor_layout,
        )

    return get_obs


def make_scale_action_fn(ctx: SpeciesContext):
    """Create an action scaling function bound to the species context."""
    from .mjx_utils import scale_action_jax

    ctrl_range = ctx.ctrl_range

    def scale_action(action):
        return scale_action_jax(action, ctrl_range)

    return scale_action


def make_reward_fns(ctx: SpeciesContext):
    """Create reward and termination functions bound to the species context.

    Returns:
        ``(compute_reward, compute_reward_detailed, is_terminated)`` tuple.
    """
    from .jax_reward_termination import compute_reward_components, compute_total_reward
    from .jax_reward_termination import is_terminated as is_terminated_fn

    reward_kw = dict(
        root_body_id=ctx.root_body_id,
        healthy_z_min=ctx.healthy_z_range[0],
        max_tilt_angle=ctx.max_tilt_angle,
        natural_forward_z=ctx.natural_forward_z,
        n_actuators=ctx.mj_model.nu,
        sensor_quat_start=ctx.sensor_layout.quat_start,
        sensor_gyro_start=ctx.sensor_layout.gyro_start,
        foot_indices=ctx.sensor_layout.foot_indices,
        sensor_tail_gyro_start=ctx.sensor_tail_gyro_start,
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

    def compute_reward(data, action, reward_cfg):
        return compute_total_reward(data, action, reward_cfg, **reward_kw)

    def compute_reward_detailed(data, action, reward_cfg):
        return compute_reward_components(data, action, reward_cfg, **reward_kw)

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
) -> tuple[Any, dict[str, Any], bool, list[str]]:
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

    Returns:
        ``(eval_results, stage_results)`` tuple.
    """
    import jax
    import numpy as np

    from .config import load_stage_config
    from .jax_eval import EvalConfig, check_stage_gate, evaluate_policy_cpu
    from .jax_normalization import normalize_obs

    get_obs = make_obs_fn(ctx)
    scale_action = make_scale_action_fn(ctx)
    compute_reward, _, _ = make_reward_fns(ctx)

    eval_params = best_params if best_params is not None else jax.device_get(params)

    eval_config = EvalConfig(
        n_episodes=n_episodes,
        max_episode_steps=ctx.max_episode_steps,
        frame_skip=ctx.frame_skip,
        healthy_z_range=ctx.healthy_z_range,
        max_tilt_angle=ctx.max_tilt_angle,
        natural_forward_z=ctx.natural_forward_z,
        termination_body_heights=ctx.termination_body_heights,
        termination_site_heights=ctx.termination_site_heights,
        success_sites=ctx.success_sites,
        success_threshold=ctx.success_threshold,
        target_body="prey",
        root_body_id=ctx.root_body_id,
        sensor_quat_start=ctx.sensor_layout.quat_start,
        reset_noise_scale=0.01,
        forward_vel_max=8.0,
    )

    foot_indices = tuple(ctx.sensor_layout.foot_indices)

    eval_results = evaluate_policy_cpu(
        ctx.mj_model,
        eval_params,
        network,
        obs_rms,
        get_obs_fn=get_obs,
        normalize_obs_fn=normalize_obs,
        scale_action_fn=scale_action,
        reward_fn=compute_reward,
        reward_cfg=ctx.reward_cfg,
        config=eval_config,
        foot_sensor_indices=foot_indices,
    )

    # Gate check
    stage_config = load_stage_config(ctx.species, ctx.stage)
    curriculum = stage_config.get("curriculum_kwargs", {})
    gate_min_reward = curriculum.get("min_avg_reward", -float("inf"))
    gate_min_length = curriculum.get("min_avg_episode_length", 0)
    gate_passed, gate_failures = check_stage_gate(
        eval_results,
        gate_min_reward,
        gate_min_length,
    )

    num_envs = env.num_envs
    rollout_len = ctx.jax_kwargs.get("rollout_len", 64)

    stage_results = {
        "stage": ctx.stage,
        "name": ctx.stage_name,
        "description": f"JAX/MJX PPO stage {ctx.stage}",
        "timesteps": total_steps,
        "duration_seconds": elapsed,
        "mean_reward": round(eval_results.mean_reward, 2),
        "std_reward": round(eval_results.std_reward, 2),
        "mean_episode_length": round(eval_results.mean_length, 1),
        "std_episode_length": round(eval_results.std_length, 1),
        "mean_forward_vel": round(eval_results.mean_forward_vel, 3),
        "std_forward_vel": round(float(np.std(eval_results.forward_vels)) if eval_results.forward_vels else 0.0, 3),
        "mean_distance_traveled": round(eval_results.mean_distance, 2),
        "best_eval_reward": round(best_reward, 2),
        "best_eval_timestep": best_update * rollout_len * num_envs,
        "sim_dt": ctx.mj_model.opt.timestep * ctx.frame_skip,
        "best_model_reward": round(eval_results.mean_reward, 2),
        "best_model_std_reward": round(eval_results.std_reward, 2),
        "best_model_length": round(eval_results.mean_length, 1),
        "best_model_std_length": round(eval_results.std_length, 1),
        "best_model_fwd_vel": round(eval_results.mean_forward_vel, 3),
        "best_model_std_fwd_vel": round(
            float(np.std(eval_results.forward_vels)) if eval_results.forward_vels else 0.0, 3
        ),
        "best_model_distance": round(eval_results.mean_distance, 2),
        "best_model_n_episodes": n_episodes,
        "gate_passed": gate_passed,
    }

    return eval_results, stage_results, gate_passed, gate_failures


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
    stage: int,
) -> None:
    """Print evaluation results and gate status."""
    import numpy as np

    print(f"\nEvaluation results ({eval_results.n_episodes} episodes):")
    print(f"  Mean reward:   {eval_results.mean_reward:.2f} +/- {eval_results.std_reward:.2f}")
    print(f"  Mean length:   {eval_results.mean_length:.1f} +/- {eval_results.std_length:.1f}")
    print(f"  Mean fwd vel:  {eval_results.mean_forward_vel:.3f} m/s")
    print(f"  Mean distance: {eval_results.mean_distance:.2f} m")
    print(f"  Mean tilt:     {np.degrees(eval_results.mean_tilt):.1f} deg")
    print(f"  Mean pelvis H: {eval_results.mean_height:.3f} m")

    if gate_failures:
        print(f"\n*** STAGE {stage} GATE NOT PASSED ***")
        for f in gate_failures:
            print(f"  - {f}")
    else:
        print(f"\n*** STAGE {stage} GATE PASSED ***")
