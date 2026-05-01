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

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .mjx_utils import check_jax


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
    sensor_gyro_start: int = 0
    sensor_accel_start: int = 3
    sensor_quat_start: int = 6
    natural_forward_z: float = 0.0
    forward_vel_max: float = 8.0
    fall_penalty: float = -100.0
    # Target spawning ranges
    target_distance_range: tuple[float, float] = (3.0, 8.0)
    target_lateral_range: tuple[float, float] = (-2.0, 2.0)
    target_z: float = 0.5
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


def register_species_mjx(species: str, **kwargs: Any) -> None:
    """Register a species configuration for MJX environments.

    Called from per-species ``mjx_config.py`` modules.
    """
    _SPECIES_CONFIGS[species] = kwargs


def _get_model_path(species: str) -> str:
    """Resolve the MJCF model path for a species."""
    species_dir_map = {
        "velociraptor": "velociraptor",
        "trex": "trex",
        "brachiosaurus": "brachiosaurus",
    }
    asset_name_map = {
        "velociraptor": "raptor.xml",
        "trex": "trex.xml",
        "brachiosaurus": "brachiosaurus.xml",
    }
    dir_name = species_dir_map[species]
    asset_name = asset_name_map[species]
    return str(Path(__file__).parent.parent / dir_name / "assets" / asset_name)


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

        # Build config from species registration, then overlay TOML stage config
        species_kwargs = _SPECIES_CONFIGS.get(species, {})

        # Merge TOML env_kwargs over species defaults.  Reward weights are
        # merged at the dict level so that stage-specific weights override
        # the species defaults while keeping any keys not mentioned in TOML.
        if env_kwargs:
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

        # Cache default MJX data once (avoids mjx.make_data per env per step)
        self._default_data = mjx.make_data(self.mjx_model)

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

        from .mjx_utils import scale_action_jax
        from .obs_functions import SensorLayout, build_bipedal_obs
        from .reward_functions import (
            check_nosedive_termination,
            quat_to_forward_2d,
            quat_to_forward_z,
            reward_action_smoothness,
            reward_alive,
            reward_angular_velocity_penalty,
            reward_approach_shaping,
            reward_drift_penalty,
            reward_energy,
            reward_forward_velocity,
            reward_heading_alignment,
            reward_height_maintenance,
            reward_lateral_velocity_penalty,
            reward_nosedive,
            reward_posture,
            reward_proximity,
            reward_speed_penalty,
        )

        model = self.mjx_model
        config = self.config
        ctrl_range = self.ctrl_range
        frame_skip = config.frame_skip
        dt = float(self.mj_model.opt.timestep) * frame_skip
        # Pre-resolved body-height termination pairs (body_id, z_threshold)
        body_height_checks = tuple(self._termination_body_checks)
        # Pre-resolved site-height termination pairs (site_id, z_threshold)
        site_height_checks = tuple(self._termination_site_checks)
        # Stage 3 success detection
        success_site_ids = self._success_site_ids
        success_threshold = config.success_threshold
        success_bonus = config.reward_weights.get(config.success_bonus_key, 0.0)

        sensor_layout = SensorLayout(
            gyro_start=config.sensor_gyro_start,
            accel_start=config.sensor_accel_start,
            quat_start=config.sensor_quat_start,
            foot_indices=config.sensor_foot_indices,
        )

        def step_fn(state: EnvState, action, rng, forward_vel_scale):
            """Pure single-environment step function.

            ``forward_vel_scale`` (a scalar JAX array) multiplies the
            ``forward_vel_weight`` reward at runtime so the trainer can
            ramp it without retracing this kernel.
            """
            # Scale action
            ctrl = scale_action_jax(action, ctrl_range)
            data = state.data.replace(ctrl=ctrl)

            # Physics step with frame skip
            def body_fn(_, d):
                return mjx.step(model, d)

            data = jax.lax.fori_loop(0, frame_skip, body_fn, data)

            # Build observation
            pelvis_id = config.body_ids.get("pelvis", config.body_ids.get("torso", 0))
            pelvis_xpos = data.xpos[pelvis_id]
            target_pos = state.target_pos
            initial_pos_2d = state.initial_pos_2d

            obs = build_bipedal_obs(
                qpos=data.qpos,
                qvel=data.qvel,
                sensordata=data.sensordata,
                pelvis_xpos=pelvis_xpos,
                target_pos=target_pos,
                sensor_layout=sensor_layout,
            )

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
            # Foot contact: read touch sensors
            foot_contact_threshold = 0.1  # Newtons
            has_foot_contact = jnp.bool_(False)
            for foot_idx in config.sensor_foot_indices:
                has_foot_contact = has_foot_contact | (data.sensordata[foot_idx] > foot_contact_threshold)

            # Condition alive bonus on height AND foot contact:
            # no reward for lying flat or balancing on head/nose
            raw_alive = reward_alive(weights.get("alive_bonus", 0.1))
            height_frac = jnp.clip(
                (pelvis_xpos[2] - config.healthy_z_range[0]) / (config.healthy_z_range[1] - config.healthy_z_range[0]),
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
            r_posture, tilt = reward_posture(pelvis_quat, config.max_tilt_angle, weights.get("posture_weight", 0.2))

            target_dist = jnp.linalg.norm(target_pos - pelvis_xpos)
            r_approach, _ = reward_approach_shaping(
                target_dist,
                state.prev_target_distance,
                weights.get("bite_approach_weight", weights.get("approach_weight", 0.0)),
                config.forward_vel_max,
                dt,
            )

            total_reward = r_forward + r_alive + r_energy + r_posture + r_approach

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
            if smoothness_w > 0:
                r_smooth, _ = reward_action_smoothness(action, state.prev_action, ctrl_range.shape[0], smoothness_w)
                total_reward = total_reward + r_smooth

            height_w = weights.get("height_weight", 0.0)
            if height_w > 0:
                r_height = reward_height_maintenance(
                    pelvis_xpos[2], config.healthy_z_range[0], config.healthy_z_range[1], height_w
                )
                total_reward = total_reward + r_height

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

            # Body-height floor contact termination
            for body_id, z_thresh in body_height_checks:
                terminated = terminated | (data.xpos[body_id, 2] < z_thresh)

            # Site-height floor contact termination (more precise for extremities)
            for site_id, z_thresh in site_height_checks:
                terminated = terminated | (data.site_xpos[site_id, 2] < z_thresh)

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

        from .obs_functions import SensorLayout, build_bipedal_obs

        model = self.mjx_model
        config = self.config
        default_data = self._default_data

        sensor_layout = SensorLayout(
            gyro_start=config.sensor_gyro_start,
            accel_start=config.sensor_accel_start,
            quat_start=config.sensor_quat_start,
            foot_indices=config.sensor_foot_indices,
        )

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

            # Forward kinematics
            data = mjx.forward(model, data)

            # Build initial observation
            pelvis_id = config.body_ids.get("pelvis", config.body_ids.get("torso", 0))
            pelvis_xpos = data.xpos[pelvis_id]

            obs = build_bipedal_obs(
                qpos=data.qpos,
                qvel=data.qvel,
                sensordata=data.sensordata,
                pelvis_xpos=pelvis_xpos,
                target_pos=target_pos,
                sensor_layout=sensor_layout,
            )

            target_dist = jnp.linalg.norm(target_pos - pelvis_xpos)
            action_dim = model.nu
            state = EnvState(
                data=data,
                obs=obs,
                step_count=jnp.int32(0),
                prev_action=jnp.zeros(action_dim),
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
