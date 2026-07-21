"""CPU-based evaluation for JAX-trained policies.

Runs deterministic evaluation episodes on CPU MuJoCo and collects
per-step biomechanical diagnostics. Used for curriculum stage-gate
checks and generating diagnostic plots.

Usage::

    from environments.shared.jax_eval import evaluate_policy_cpu

    results = evaluate_policy_cpu(
        mj_model, params, network, obs_stats,
        get_obs_fn=get_obs, scale_action_fn=scale_action,
        reward_fn=compute_reward, reward_cfg=reward_cfg,
        config=eval_config,
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .reward_functions import reward_lean_aware_posture, reward_posture


@dataclass
class EvalConfig:
    """Configuration for CPU evaluation episodes."""

    n_episodes: int = 25
    max_episode_steps: int = 1000
    frame_skip: int = 5
    healthy_z_range: tuple[float, float] = (0.3, 2.0)
    max_tilt_angle: float = 1.047
    natural_forward_z: float = 0.0
    posture_target_forward_z: float | None = None
    nosedive_threshold: float = 0.5
    termination_body_heights: dict[str, float] | None = None
    termination_site_heights: dict[str, float] | None = None
    success_sites: tuple[str, ...] = ()
    success_threshold: float = 0.3
    target_body: str | None = None
    root_body_id: int = 1
    sensor_gyro_start: int = 0
    sensor_quat_start: int = 6
    action_mapping: str = "midpoint/v1"
    reset_noise_scale: float = 0.01
    forward_vel_max: float = 8.0
    # Success bonus added to the reward when a success site reaches the
    # target (stage 3), and penalty applied on fall termination — both
    # mirror the training env so eval episode rewards are comparable.
    success_bonus: float = 0.0
    fall_penalty: float = 0.0
    # Species standing height for the height-reward decomposition (matches
    # the SB3 envs' target_z: 0.90 trex, 1.2 brachio).
    target_standing_z: float = 0.90
    # Seed for reset noise — evaluation is otherwise non-reproducible and
    # curriculum gate decisions would vary run-to-run for borderline policies.
    seed: int = 42


def _posture_reward_for_eval(
    quat: np.ndarray,
    config: EvalConfig,
    weight: float,
) -> float:
    """Mirror the posture primitive selected by JAX training."""
    if config.posture_target_forward_z is None:
        reward, _ = reward_posture(quat, config.max_tilt_angle, weight)
    else:
        reward, _ = reward_lean_aware_posture(
            quat,
            config.max_tilt_angle,
            weight,
            config.posture_target_forward_z,
        )
    return float(reward)


@dataclass
class EvalResults:
    """Collected evaluation results and per-step diagnostics."""

    # Per-episode metrics
    rewards: list[float] = field(default_factory=list)
    lengths: list[int] = field(default_factory=list)
    forward_vels: list[float] = field(default_factory=list)
    distances: list[float] = field(default_factory=list)
    tilt_angles: list[float] = field(default_factory=list)
    pelvis_heights: list[float] = field(default_factory=list)
    # True for episodes that ended by reaching the success target (stage 3).
    successes: list[bool] = field(default_factory=list)

    # Per-step diagnostics (across all episodes)
    diag_tilt: list[float] = field(default_factory=list)
    diag_fwd_vel: list[float] = field(default_factory=list)
    diag_pelvis_h: list[float] = field(default_factory=list)
    diag_l_foot: list[float] = field(default_factory=list)
    diag_r_foot: list[float] = field(default_factory=list)
    diag_energy: list[float] = field(default_factory=list)
    diag_reward_components: dict[str, list[float]] = field(
        default_factory=lambda: {
            "forward": [],
            "alive": [],
            "energy": [],
            "posture": [],
            "height": [],
            "nosedive": [],
            "drift": [],
            "speed": [],
            "spin": [],
            "smoothness": [],
        }
    )

    @property
    def mean_reward(self) -> float:
        return float(np.mean(self.rewards)) if self.rewards else 0.0

    @property
    def std_reward(self) -> float:
        return float(np.std(self.rewards)) if self.rewards else 0.0

    @property
    def mean_length(self) -> float:
        return float(np.mean(self.lengths)) if self.lengths else 0.0

    @property
    def std_length(self) -> float:
        return float(np.std(self.lengths)) if self.lengths else 0.0

    @property
    def mean_forward_vel(self) -> float:
        return float(np.mean(self.forward_vels)) if self.forward_vels else 0.0

    @property
    def mean_distance(self) -> float:
        return float(np.mean(self.distances)) if self.distances else 0.0

    @property
    def mean_tilt(self) -> float:
        return float(np.mean(self.tilt_angles)) if self.tilt_angles else 0.0

    @property
    def mean_height(self) -> float:
        return float(np.mean(self.pelvis_heights)) if self.pelvis_heights else 0.0

    @property
    def mean_success_rate(self) -> float:
        return float(np.mean(self.successes)) if self.successes else 0.0


def evaluate_policy_cpu(
    mj_model: Any,
    params: Any,
    network: Any,
    obs_stats: Any,
    *,
    get_obs_fn: Any,
    normalize_obs_fn: Any,
    scale_action_fn: Any,
    reward_fn: Any,
    reward_cfg: dict[str, float],
    config: EvalConfig,
    foot_sensor_indices: tuple[int, ...] = (),
) -> EvalResults:
    """Run deterministic evaluation episodes on CPU MuJoCo.

    Args:
        mj_model: MuJoCo model.
        params: JAX network parameters.
        network: Flax ActorCritic network module.
        obs_stats: RunningMeanStd observation statistics.
        get_obs_fn: Function(mjx_data) -> obs array.
        normalize_obs_fn: Function(obs, obs_stats) -> normalized obs.
        scale_action_fn: Function(action) -> scaled ctrl.
        reward_fn: Function(mjx_data, action, reward_cfg) -> scalar reward.
        reward_cfg: Reward weight dict.
        config: Evaluation configuration.
        foot_sensor_indices: Sensor indices for foot contacts (for diagnostics).

    Returns:
        ``EvalResults`` with per-episode and per-step metrics.
    """
    import jax.numpy as jnp
    import mujoco
    import mujoco.mjx as mjx

    from .mjx_env import ACTION_MAPPING_HOME_KEYFRAME_RESIDUAL, ACTION_MAPPING_MIDPOINT
    from .mjx_utils import reset_mujoco_data_to_home

    mj_data = mujoco.MjData(mj_model)
    act_dim = mj_model.nu
    results = EvalResults()
    reset_rng = np.random.default_rng(config.seed)

    # Resolve body-height termination checks
    _body_checks: list[tuple[int, float]] = []
    if config.termination_body_heights:
        for bname, z_thresh in config.termination_body_heights.items():
            bid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, bname)
            if bid >= 0:
                _body_checks.append((bid, z_thresh))

    # Resolve site-height termination checks
    _site_checks: list[tuple[int, float]] = []
    if config.termination_site_heights:
        for sname, z_thresh in config.termination_site_heights.items():
            sid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, sname)
            if sid >= 0:
                _site_checks.append((sid, z_thresh))

    # Resolve success site IDs and target body
    _success_site_ids: list[int] = []
    for sname in config.success_sites:
        sid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, sname)
        if sid >= 0:
            _success_site_ids.append(sid)
    _target_body_id = -1
    if config.target_body is not None:
        _target_body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, config.target_body)

    for ep in range(config.n_episodes):
        if config.action_mapping == ACTION_MAPPING_HOME_KEYFRAME_RESIDUAL:
            reset_mujoco_data_to_home(mj_model, mj_data)
        elif config.action_mapping == ACTION_MAPPING_MIDPOINT:
            mujoco.mj_resetData(mj_model, mj_data)
        else:
            raise ValueError(f"unknown JAX action mapping {config.action_mapping!r}")
        mj_data.qpos[7:] += reset_rng.uniform(
            -config.reset_noise_scale,
            config.reset_noise_scale,
            size=mj_data.qpos[7:].shape,
        )
        mujoco.mj_forward(mj_model, mj_data)

        ep_reward = 0.0
        ep_fwd_vels = []
        ep_tilts = []
        ep_heights = []
        ep_success = False
        start_pos = mj_data.qpos[:2].copy()
        prev_action = None
        # Track the target for reward parity with the training env (approach
        # shaping compares against the previous step's distance).
        prev_target_dist: float | None = None
        if _target_body_id >= 0:
            prev_target_dist = float(np.linalg.norm(mj_data.xpos[_target_body_id] - mj_data.xpos[config.root_body_id]))

        for step in range(config.max_episode_steps):
            cpu_data = mjx.put_data(mj_model, mj_data)
            obs_raw = get_obs_fn(cpu_data)
            obs = normalize_obs_fn(obs_raw, obs_stats)

            mean, _log_std, _value = network.apply(params, obs)
            action = jnp.clip(mean, -1.0, 1.0)

            ctrl = np.array(scale_action_fn(action))
            mj_data.ctrl[:] = ctrl
            for _ in range(config.frame_skip):
                mujoco.mj_step(mj_model, mj_data)

            # Per-step diagnostics
            fwd_vel = float(mj_data.qvel[0])
            body_z = float(mj_data.xpos[config.root_body_id, 2])
            root_quat = mj_data.sensordata[config.sensor_quat_start : config.sensor_quat_start + 4]
            tilt = float(np.arccos(np.clip(1.0 - 2.0 * (root_quat[1] ** 2 + root_quat[2] ** 2), -1, 1)))
            energy = float(np.sum(np.square(np.array(action)))) / act_dim

            ep_fwd_vels.append(fwd_vel)
            ep_tilts.append(tilt)
            ep_heights.append(body_z)

            results.diag_fwd_vel.append(fwd_vel)
            results.diag_tilt.append(tilt)
            results.diag_pelvis_h.append(body_z)
            results.diag_energy.append(energy)

            # Foot contacts — sensor order alternates right/left (bipeds:
            # R, L; quadrupeds: R, L, RR, RL), so split by parity rather
            # than lumping every non-first foot into the left series.
            for i, idx in enumerate(foot_sensor_indices):
                if len(mj_data.sensordata) > idx:
                    target_list = results.diag_r_foot if i % 2 == 0 else results.diag_l_foot
                    target_list.append(float(mj_data.sensordata[idx]))

            # Reward decomposition — all active components
            fwd_norm = float(np.clip(fwd_vel / config.forward_vel_max, -1.0, 1.0))
            results.diag_reward_components["forward"].append(reward_cfg.get("forward_vel_weight", 0.0) * fwd_norm)
            results.diag_reward_components["alive"].append(reward_cfg.get("alive_bonus", 0.0))
            results.diag_reward_components["energy"].append(-reward_cfg.get("energy_penalty_weight", 0.0) * energy)
            posture_reward = _posture_reward_for_eval(
                root_quat,
                config,
                reward_cfg.get("posture_weight", 0.2),
            )
            results.diag_reward_components["posture"].append(posture_reward)

            # Height maintenance
            height_w = reward_cfg.get("height_weight", 0.0)
            if height_w > 0:
                healthy_z_min = config.healthy_z_range[0]
                target_z = config.target_standing_z
                height_frac = float(np.clip((body_z - healthy_z_min) / (target_z - healthy_z_min), 0.0, 1.0))
                results.diag_reward_components["height"].append(height_w * height_frac)
            else:
                results.diag_reward_components["height"].append(0.0)

            # Nosedive penalty
            nosedive_w = reward_cfg.get("nosedive_weight", 0.0)
            root_quat_nd = mj_data.sensordata[config.sensor_quat_start : config.sensor_quat_start + 4]
            w_q, x_q, y_q, z_q = root_quat_nd[0], root_quat_nd[1], root_quat_nd[2], root_quat_nd[3]
            forward_z_nd = float(2.0 * (x_q * z_q - w_q * y_q))
            if nosedive_w > 0:
                nosedive_excess = max(0.0, -(forward_z_nd - config.natural_forward_z))
                results.diag_reward_components["nosedive"].append(-nosedive_w * nosedive_excess)
            else:
                results.diag_reward_components["nosedive"].append(0.0)

            # Drift penalty
            drift_w = reward_cfg.get("drift_penalty_weight", 0.0)
            if drift_w > 0:
                drift_dist = float(np.linalg.norm(mj_data.qpos[:2] - start_pos))
                drift_norm = drift_dist / 2.0
                results.diag_reward_components["drift"].append(-drift_w * drift_norm**2)
            else:
                results.diag_reward_components["drift"].append(0.0)

            # Speed penalty
            speed_w = reward_cfg.get("speed_penalty_weight", 0.0)
            if speed_w > 0:
                speed_2d = float(np.linalg.norm(mj_data.qvel[:2]))
                speed_thresh = reward_cfg.get("speed_penalty_threshold", 0.10)
                excess = max(0.0, speed_2d - speed_thresh)
                excess_norm = min(excess / 1.0, 1.0)
                results.diag_reward_components["speed"].append(-speed_w * excess_norm)
            else:
                results.diag_reward_components["speed"].append(0.0)

            # Spin penalty
            spin_w = reward_cfg.get("spin_penalty_weight", 0.0)
            if spin_w > 0:
                gyro = mj_data.sensordata[config.sensor_gyro_start : config.sensor_gyro_start + 3]
                angvel_mag = float(np.linalg.norm(gyro))
                angvel_norm = min(angvel_mag / 10.0, 1.0)
                results.diag_reward_components["spin"].append(-spin_w * angvel_norm)
            else:
                results.diag_reward_components["spin"].append(0.0)

            # Action smoothness
            smoothness_w = reward_cfg.get("smoothness_weight", 0.0)
            if smoothness_w > 0 and prev_action is not None:
                action_delta = float(np.sum(np.square(np.array(action) - np.array(prev_action))))
                max_delta = act_dim * 4.0
                delta_norm = min(action_delta / max_delta, 1.0)
                results.diag_reward_components["smoothness"].append(-smoothness_w * delta_norm)
            else:
                results.diag_reward_components["smoothness"].append(0.0)

            # Per-step state for reward parity with the training env:
            # target position/direction, previous-step target distance
            # (approach shaping), previous action (smoothness), and success
            # site positions (proximity reward + success bonus).  Passed as
            # kwargs so simple 3-arg reward_fns keep working when absent.
            step_kwargs: dict[str, Any] = {}
            if _target_body_id >= 0:
                target_pos_np = np.array(mj_data.xpos[_target_body_id])
                pelvis_np = np.array(mj_data.xpos[config.root_body_id])
                step_kwargs["target_pos"] = jnp.asarray(target_pos_np)
                step_kwargs["prev_target_distance"] = jnp.asarray(prev_target_dist)
                rel_2d = target_pos_np[:2] - pelvis_np[:2]
                rel_norm = float(np.linalg.norm(rel_2d))
                if rel_norm > 1e-8:
                    # Training projects velocity onto the current
                    # agent-to-target direction; mirror it here.
                    step_kwargs["forward_ref_2d"] = jnp.asarray(rel_2d / rel_norm)
            if prev_action is not None:
                step_kwargs["prev_action"] = prev_action
            if _success_site_ids:
                step_kwargs["success_site_positions"] = jnp.asarray(
                    np.stack([np.array(mj_data.site_xpos[sid]) for sid in _success_site_ids])
                )

            # Reuse a single mjx.put_data call for the post-step data;
            # mj_data has already been advanced by frame_skip mj_step calls.
            post_step_data = mjx.put_data(mj_model, mj_data)
            r = float(reward_fn(post_step_data, action, reward_cfg, **step_kwargs))
            ep_reward += r

            # Update AFTER the reward call: the training env compares the
            # current action/distance against the previous step's values.
            prev_action = action
            if _target_body_id >= 0:
                prev_target_dist = float(
                    np.linalg.norm(mj_data.xpos[_target_body_id] - mj_data.xpos[config.root_body_id])
                )

            if body_z < config.healthy_z_range[0] or body_z > config.healthy_z_range[1]:
                break
            if tilt > config.max_tilt_angle:
                break

            # Nosedive termination (excessive forward pitch)
            root_quat = mj_data.sensordata[config.sensor_quat_start : config.sensor_quat_start + 4]
            w, x, y, z = root_quat[0], root_quat[1], root_quat[2], root_quat[3]
            forward_z = float(2.0 * (x * z - w * y))
            if forward_z < config.natural_forward_z - config.nosedive_threshold:
                break

            # Body-height floor contact termination
            if any(mj_data.xpos[bid, 2] < zt for bid, zt in _body_checks):
                break

            # Site-height termination (extremities like snout tip)
            if any(mj_data.site_xpos[sid, 2] < zt for sid, zt in _site_checks):
                break

            # Stage 3 success: proximity-based contact detection
            if _success_site_ids and _target_body_id >= 0:
                target_pos = mj_data.xpos[_target_body_id]
                if any(
                    float(np.linalg.norm(mj_data.site_xpos[sid] - target_pos)) < config.success_threshold
                    for sid in _success_site_ids
                ):
                    ep_success = True
                    break

        ep_length = step + 1
        distance = float(np.linalg.norm(mj_data.qpos[:2] - start_pos))

        results.rewards.append(ep_reward)
        results.lengths.append(ep_length)
        results.forward_vels.append(float(np.mean(ep_fwd_vels)))
        results.distances.append(distance)
        results.tilt_angles.append(float(np.mean(ep_tilts)))
        results.pelvis_heights.append(float(np.mean(ep_heights)))
        results.successes.append(ep_success)

    return results


def check_stage_gate(
    results: EvalResults,
    gate_min_reward: float = -float("inf"),
    gate_min_length: float = 0,
    gate_min_forward_vel: float = 0.0,
    gate_min_success_rate: float = 0.0,
) -> tuple[bool, list[str]]:
    """Check curriculum stage gate conditions.

    Args:
        results: Evaluation results.
        gate_min_reward: Minimum average reward to pass.
        gate_min_length: Minimum average episode length to pass.
        gate_min_forward_vel: Minimum average forward velocity to pass
            (stage-2 TOML ``min_avg_forward_vel``; 0 disables).
        gate_min_success_rate: Minimum success rate to pass (stage-3 TOML
            ``min_success_rate``; 0 disables).

    Returns:
        (passed, failures) tuple where failures is a list of failure descriptions.
    """
    failures = []
    if results.mean_reward < gate_min_reward:
        failures.append(f"mean reward {results.mean_reward:.2f} < {gate_min_reward}")
    if results.mean_length < gate_min_length:
        failures.append(f"mean episode length {results.mean_length:.1f} < {gate_min_length}")
    if gate_min_forward_vel > 0 and results.mean_forward_vel < gate_min_forward_vel:
        failures.append(f"mean forward vel {results.mean_forward_vel:.2f} < {gate_min_forward_vel}")
    if gate_min_success_rate > 0 and results.mean_success_rate < gate_min_success_rate:
        failures.append(f"success rate {results.mean_success_rate:.2f} < {gate_min_success_rate}")
    return len(failures) == 0, failures
