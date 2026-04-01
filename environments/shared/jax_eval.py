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


@dataclass
class EvalConfig:
    """Configuration for CPU evaluation episodes."""

    n_episodes: int = 25
    max_episode_steps: int = 1000
    frame_skip: int = 5
    healthy_z_range: tuple[float, float] = (0.3, 2.0)
    max_tilt_angle: float = 1.047
    natural_forward_z: float = 0.0
    nosedive_threshold: float = 0.5
    termination_body_heights: dict[str, float] | None = None
    success_sites: tuple[str, ...] = ()
    success_threshold: float = 0.3
    target_body: str | None = None
    root_body_id: int = 1
    sensor_quat_start: int = 6
    reset_noise_scale: float = 0.01
    forward_vel_max: float = 8.0


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

    # Per-step diagnostics (across all episodes)
    diag_tilt: list[float] = field(default_factory=list)
    diag_fwd_vel: list[float] = field(default_factory=list)
    diag_pelvis_h: list[float] = field(default_factory=list)
    diag_l_foot: list[float] = field(default_factory=list)
    diag_r_foot: list[float] = field(default_factory=list)
    diag_energy: list[float] = field(default_factory=list)
    diag_reward_components: dict[str, list[float]] = field(
        default_factory=lambda: {"forward": [], "alive": [], "energy": [], "posture": []}
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

    mj_data = mujoco.MjData(mj_model)
    act_dim = mj_model.nu
    results = EvalResults()

    # Resolve body-height termination checks
    _body_checks: list[tuple[int, float]] = []
    if config.termination_body_heights:
        for bname, z_thresh in config.termination_body_heights.items():
            bid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, bname)
            if bid >= 0:
                _body_checks.append((bid, z_thresh))

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
        mujoco.mj_resetData(mj_model, mj_data)
        mj_data.qpos[7:] += np.random.uniform(
            -config.reset_noise_scale,
            config.reset_noise_scale,
            size=mj_data.qpos[7:].shape,
        )
        mujoco.mj_forward(mj_model, mj_data)

        ep_reward = 0.0
        ep_fwd_vels = []
        ep_tilts = []
        ep_heights = []
        start_pos = mj_data.qpos[:2].copy()

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

            # Foot contacts
            for i, idx in enumerate(foot_sensor_indices):
                if len(mj_data.sensordata) > idx:
                    target_list = results.diag_l_foot if i == 0 else results.diag_r_foot
                    target_list.append(float(mj_data.sensordata[idx]))

            # Reward decomposition
            fwd_norm = float(np.clip(fwd_vel / config.forward_vel_max, -1.0, 1.0))
            results.diag_reward_components["forward"].append(reward_cfg.get("forward_vel_weight", 0.0) * fwd_norm)
            results.diag_reward_components["alive"].append(reward_cfg.get("alive_bonus", 0.0))
            results.diag_reward_components["energy"].append(-reward_cfg.get("energy_penalty_weight", 0.0) * energy)
            tilt_norm = min(tilt / config.max_tilt_angle, 1.0)
            results.diag_reward_components["posture"].append(-reward_cfg.get("posture_weight", 0.2) * tilt_norm**2)

            cpu_data = mjx.put_data(mj_model, mj_data)
            r = float(reward_fn(cpu_data, action, reward_cfg))
            ep_reward += r

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

            # Stage 3 success: proximity-based contact detection
            if _success_site_ids and _target_body_id >= 0:
                target_pos = mj_data.xpos[_target_body_id]
                if any(
                    float(np.linalg.norm(mj_data.site_xpos[sid] - target_pos)) < config.success_threshold
                    for sid in _success_site_ids
                ):
                    break

        ep_length = step + 1
        distance = float(np.linalg.norm(mj_data.qpos[:2] - start_pos))

        results.rewards.append(ep_reward)
        results.lengths.append(ep_length)
        results.forward_vels.append(float(np.mean(ep_fwd_vels)))
        results.distances.append(distance)
        results.tilt_angles.append(float(np.mean(ep_tilts)))
        results.pelvis_heights.append(float(np.mean(ep_heights)))

    return results


def check_stage_gate(
    results: EvalResults,
    gate_min_reward: float = -float("inf"),
    gate_min_length: float = 0,
) -> tuple[bool, list[str]]:
    """Check curriculum stage gate conditions.

    Args:
        results: Evaluation results.
        gate_min_reward: Minimum average reward to pass.
        gate_min_length: Minimum average episode length to pass.

    Returns:
        (passed, failures) tuple where failures is a list of failure descriptions.
    """
    failures = []
    if results.mean_reward < gate_min_reward:
        failures.append(f"mean reward {results.mean_reward:.2f} < {gate_min_reward}")
    if results.mean_length < gate_min_length:
        failures.append(f"mean episode length {results.mean_length:.1f} < {gate_min_length}")
    return len(failures) == 0, failures
