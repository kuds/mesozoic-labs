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

    def __init__(self, species: str, stage: int = 1, num_envs: int = 2048):
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

        # Build config from species registration + TOML stage config
        species_kwargs = _SPECIES_CONFIGS.get(species, {})
        self.config = MJXEnvConfig(
            species=species,
            stage=stage,
            **species_kwargs,
        )

        # Pre-compile step and reset functions
        self._step_single = jax.jit(self._make_step_fn())
        self._reset_single = jax.jit(self._make_reset_fn())
        self._batched_step = jax.jit(jax.vmap(self._step_single, in_axes=(0, 0, 0)))
        self._batched_reset = jax.jit(jax.vmap(self._reset_single, in_axes=(0,)))

    def _make_step_fn(self):
        """Build the single-env step function as a closure over model/config."""
        import jax
        import jax.numpy as jnp
        import mujoco.mjx as mjx

        from .mjx_utils import scale_action_jax
        from .obs_functions import SensorLayout, build_bipedal_obs
        from .reward_functions import (
            reward_alive,
            reward_approach_shaping,
            reward_energy,
            reward_forward_velocity,
            reward_posture,
        )

        model = self.mjx_model
        config = self.config
        ctrl_range = self.ctrl_range
        frame_skip = config.frame_skip
        dt = float(self.mj_model.opt.timestep) * frame_skip

        sensor_layout = SensorLayout(
            gyro_start=config.sensor_gyro_start,
            accel_start=config.sensor_accel_start,
            quat_start=config.sensor_quat_start,
            foot_indices=config.sensor_foot_indices,
        )

        def step_fn(state: EnvState, action, rng):
            """Pure single-environment step function."""
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
            # Use initial direction toward target as forward reference
            target_rel_2d = target_pos[:2] - jnp.zeros(2)
            forward_ref = target_rel_2d / (jnp.linalg.norm(target_rel_2d) + 1e-8)

            weights = config.reward_weights
            r_forward, fwd_vel = reward_forward_velocity(
                vel_2d, forward_ref, config.forward_vel_max, weights.get("forward_vel_weight", 1.0)
            )
            r_alive = reward_alive(weights.get("alive_bonus", 0.1))
            r_energy = reward_energy(action, ctrl_range.shape[0], weights.get("energy_penalty_weight", 0.001))

            pelvis_quat = data.sensordata[config.sensor_quat_start : config.sensor_quat_start + 4]
            r_posture, tilt = reward_posture(pelvis_quat, config.max_tilt_angle, weights.get("posture_weight", 0.2))

            target_dist = jnp.linalg.norm(target_pos - pelvis_xpos)
            r_approach, _ = reward_approach_shaping(
                float(target_dist),
                float(state.prev_target_distance),
                weights.get("approach_weight", 1.0),
                config.forward_vel_max,
                dt,
            )

            total_reward = r_forward + r_alive + r_energy + r_posture + r_approach

            # Termination
            body_z = pelvis_xpos[2]
            terminated = (body_z < config.healthy_z_range[0]) | (body_z > config.healthy_z_range[1])
            terminated = terminated | (tilt > config.max_tilt_angle)

            # Add fall penalty if terminated
            total_reward = jnp.where(terminated, total_reward + config.fall_penalty, total_reward)

            step_count = state.step_count + 1
            truncated = step_count >= config.max_episode_steps

            new_state = EnvState(
                data=data,
                obs=obs,
                step_count=step_count,
                prev_action=action,
                prev_target_distance=target_dist,
                target_pos=target_pos,
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

        sensor_layout = SensorLayout(
            gyro_start=config.sensor_gyro_start,
            accel_start=config.sensor_accel_start,
            quat_start=config.sensor_quat_start,
            foot_indices=config.sensor_foot_indices,
        )

        def reset_fn(rng):
            """Pure single-environment reset function."""
            data = mjx.make_data(model)

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
            )
            return state

        return reset_fn

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

    def step(self, states: EnvState, actions, rng):
        """Step all environments.

        Args:
            states: Batched ``EnvState``.
            actions: Actions array of shape ``(num_envs, action_dim)``.
            rng: JAX PRNGKey (used for auto-reset).

        Returns:
            (new_states, rewards, terminated, truncated) tuple.
        """
        import jax

        rngs = jax.random.split(rng, self.num_envs)
        new_states, rewards, terminated, truncated = self._batched_step(states, actions, rngs)

        # Auto-reset terminated or truncated environments
        dones = terminated | truncated
        reset_rngs = jax.random.split(rng, self.num_envs)
        reset_states = self._batched_reset(reset_rngs)

        # Where done, use reset state; otherwise keep new state
        new_states = jax.tree.map(
            lambda new, rst, d: jax.numpy.where(d, rst, new) if hasattr(new, "shape") else new,
            new_states,
            reset_states,
            dones,
        )

        return new_states, rewards, terminated, truncated
