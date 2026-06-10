"""Composable JAX reward & termination functions for training notebooks.

These functions compose the primitive reward/termination functions from
``reward_functions.py`` into the same logic used by ``mjx_env.py``'s step
function.  The notebook imports these instead of re-implementing the logic,
ensuring a single source of truth.

All functions are JAX-trace-safe and vmap-compatible.  ``reward_cfg`` is
expected to be a plain Python dict (passed via ``in_axes=None`` in vmap),
so Python-level ``if`` on its values resolves at JIT trace time.
"""

from __future__ import annotations

from typing import Any

from .reward_functions import (
    check_height_tilt_termination,
    check_nosedive_termination,
    quat_to_forward_2d,
    quat_to_forward_z,
    quat_to_tilt,
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

Array = Any


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------


def compute_total_reward(
    data: Any,
    action: Array,
    reward_cfg: dict[str, float],
    *,
    root_body_id: int,
    healthy_z_min: float,
    healthy_z_max: float = 2.0,
    target_standing_z: float | None = None,
    max_tilt_angle: float,
    natural_forward_z: float,
    n_actuators: int,
    sensor_quat_start: int = 6,
    sensor_gyro_start: int = 0,
    foot_indices: tuple[int, ...] = (10, 11),
    prev_action: Array | None = None,
    sensor_tail_gyro_start: int | None = None,
    forward_ref_2d: Array | None = None,
    target_pos: Array | None = None,
    prev_target_distance: Array | None = None,
    forward_vel_max: float = 8.0,
    dt: float = 0.01,
    fall_penalty: float = 0.0,
    success_site_positions: Array | None = None,
    success_threshold: float = 0.3,
    success_bonus: float = 0.0,
) -> Array:
    """Compute total scalar reward matching ``mjx_env.py`` step logic.

    Parameters are split into ``reward_cfg`` (per-stage, read at trace time)
    and keyword arguments (per-species constants).

    ``forward_ref_2d`` is the 2D direction used for forward-velocity and
    heading-alignment rewards.  When tracking a target, pass the normalised
    agent-to-target direction so that both rewards are consistent with
    ``MJXDinoEnv.step()``.  Defaults to world +X ``[1, 0]``.
    """
    import jax.numpy as jnp

    vel_2d = data.qvel[:2]
    forward_dir = forward_ref_2d if forward_ref_2d is not None else jnp.array([1.0, 0.0])
    pelvis_z = data.xpos[root_body_id, 2]
    pelvis_xpos = data.xpos[root_body_id]
    root_quat = data.sensordata[sensor_quat_start : sensor_quat_start + 4]

    # Forward velocity
    r_forward, _ = reward_forward_velocity(
        vel_2d,
        forward_dir,
        reward_cfg.get("forward_vel_max", forward_vel_max),
        reward_cfg.get("forward_vel_weight", 0.0),
    )

    # Alive bonus (height-gated, optionally foot-contact-gated)
    raw_alive = reward_alive(reward_cfg.get("alive_bonus", 0.1))
    height_frac = jnp.clip(
        (pelvis_z - healthy_z_min) / (healthy_z_max - healthy_z_min),
        0.0,
        1.0,
    )
    foot_contact_gate = reward_cfg.get("foot_contact_gate", 0.0)
    if foot_contact_gate > 0:
        has_foot_contact = _check_foot_contact(data, foot_indices)
        alive_gate = height_frac * has_foot_contact.astype(jnp.float32)
    else:
        alive_gate = height_frac
    r_alive = raw_alive * alive_gate

    r_energy = reward_energy(action, n_actuators, reward_cfg.get("energy_penalty_weight", 0.001))
    r_posture, _ = reward_posture(root_quat, max_tilt_angle, reward_cfg.get("posture_weight", 0.2))

    total = r_forward + r_alive + r_energy + r_posture

    # Approach shaping (reward moving toward target)
    approach_w = reward_cfg.get("bite_approach_weight", reward_cfg.get("approach_weight", 0.0))
    if approach_w > 0 and target_pos is not None and prev_target_distance is not None:
        target_dist = jnp.linalg.norm(target_pos - pelvis_xpos)
        r_approach, _ = reward_approach_shaping(
            target_dist,
            prev_target_distance,
            approach_w,
            forward_vel_max,
            dt,
        )
        total = total + r_approach

    # Conditional components (weight > 0 resolved at trace time)
    foot_contact_w = reward_cfg.get("foot_contact_weight", 0.0)
    if foot_contact_w > 0:
        has_foot_contact = _check_foot_contact(data, foot_indices)
        total = total + foot_contact_w * has_foot_contact.astype(jnp.float32)

    height_w = reward_cfg.get("height_weight", 0.0)
    if height_w > 0:
        # Saturate at the species standing height (SB3 parity), not the
        # termination ceiling — the latter flattens the gradient 5-10x.
        _target_z = target_standing_z if target_standing_z is not None else healthy_z_max
        total = total + reward_height_maintenance(pelvis_z, healthy_z_min, _target_z, height_w)

    nosedive_w = reward_cfg.get("nosedive_weight", 0.0)
    if nosedive_w > 0:
        r_nd, _ = reward_nosedive(root_quat, nosedive_w, natural_forward_z)
        total = total + r_nd

    drift_w = reward_cfg.get("drift_penalty_weight", 0.0)
    if drift_w > 0:
        r_dr, _ = reward_drift_penalty(data.xpos[root_body_id, :2], jnp.zeros(2), drift_w)
        total = total + r_dr

    speed_w = reward_cfg.get("speed_penalty_weight", 0.0)
    if speed_w > 0:
        r_sp, _ = reward_speed_penalty(vel_2d, speed_w, reward_cfg.get("speed_penalty_threshold", 0.10))
        total = total + r_sp

    spin_w = reward_cfg.get("spin_penalty_weight", 0.0)
    if spin_w > 0:
        angvel = data.sensordata[sensor_gyro_start : sensor_gyro_start + 3]
        r_sn, _ = reward_angular_velocity_penalty(angvel, spin_w)
        total = total + r_sn

    smoothness_w = reward_cfg.get("smoothness_weight", 0.0)
    if smoothness_w > 0:
        smooth_ref = prev_action if prev_action is not None else jnp.zeros_like(action)
        r_sm, _ = reward_action_smoothness(action, smooth_ref, n_actuators, smoothness_w)
        total = total + r_sm

    # Tail stability: penalise tail tip angular velocity
    tail_w = reward_cfg.get("tail_stability_weight", 0.0)
    if tail_w > 0 and sensor_tail_gyro_start is not None:
        tail_angvel = data.sensordata[sensor_tail_gyro_start : sensor_tail_gyro_start + 3]
        r_tail, _ = reward_angular_velocity_penalty(tail_angvel, tail_w)
        total = total + r_tail

    # Heading alignment: reward facing the forward direction
    heading_w = reward_cfg.get("heading_weight", 0.0)
    if heading_w > 0:
        body_fwd_2d = quat_to_forward_2d(root_quat)
        r_heading, _ = reward_heading_alignment(body_fwd_2d, forward_dir, heading_w)
        total = total + r_heading

    # Lateral velocity penalty: penalise crab-walking
    lateral_w = reward_cfg.get("lateral_penalty_weight", 0.0)
    if lateral_w > 0:
        body_fwd_2d_lat = quat_to_forward_2d(root_quat)
        r_lateral, _ = reward_lateral_velocity_penalty(vel_2d, body_fwd_2d_lat, lateral_w)
        total = total + r_lateral

    # Head/claw proximity reward: continuous gradient for final positioning
    proximity_w = reward_cfg.get(
        "bite_head_proximity_weight",
        reward_cfg.get("strike_claw_proximity_weight", reward_cfg.get("food_head_proximity_weight", 0.0)),
    )
    if proximity_w > 0 and success_site_positions is not None and target_pos is not None:
        for i in range(success_site_positions.shape[0]):
            site_dist = jnp.linalg.norm(success_site_positions[i] - target_pos)
            r_prox, _ = reward_proximity(site_dist, forward_vel_max, proximity_w)
            total = total + r_prox

    # Success bonus (stage 3 proximity-based contact detection)
    success = jnp.bool_(False)
    if success_bonus > 0 and success_site_positions is not None:
        for i in range(success_site_positions.shape[0]):
            dist = jnp.linalg.norm(success_site_positions[i] - target_pos)
            success = success | (dist < success_threshold)
        total = jnp.where(success, total + success_bonus, total)

    # Fall penalty on termination (but not on success)
    if fall_penalty != 0.0:
        tilt = quat_to_tilt(root_quat)
        terminated = (pelvis_z < healthy_z_min) | (pelvis_z > healthy_z_max)
        terminated = terminated | (tilt > max_tilt_angle)
        forward_z = quat_to_forward_z(root_quat)
        nosedive_threshold = reward_cfg.get("nosedive_termination_threshold", 0.5)
        nosedive_terminated, _ = check_nosedive_termination(forward_z, natural_forward_z, threshold=nosedive_threshold)
        terminated = terminated | nosedive_terminated
        total = jnp.where(terminated & ~success, total + fall_penalty, total)

    return total


def compute_reward_components(
    data: Any,
    action: Array,
    reward_cfg: dict[str, float],
    *,
    root_body_id: int,
    healthy_z_min: float,
    healthy_z_max: float = 2.0,
    target_standing_z: float | None = None,
    max_tilt_angle: float,
    natural_forward_z: float,
    n_actuators: int,
    sensor_quat_start: int = 6,
    sensor_gyro_start: int = 0,
    foot_indices: tuple[int, ...] = (10, 11),
    prev_action: Array | None = None,
    sensor_tail_gyro_start: int | None = None,
    forward_ref_2d: Array | None = None,
) -> dict[str, Array]:
    """Compute per-component reward breakdown for diagnostics.

    Returns a dict of scalar rewards.  Keys prefixed with ``_`` are
    diagnostic state variables (not reward components).
    """
    import jax.numpy as jnp

    vel_2d = data.qvel[:2]
    forward_dir = forward_ref_2d if forward_ref_2d is not None else jnp.array([1.0, 0.0])
    pelvis_z = data.xpos[root_body_id, 2]
    root_quat = data.sensordata[sensor_quat_start : sensor_quat_start + 4]

    r_forward, _ = reward_forward_velocity(
        vel_2d,
        forward_dir,
        reward_cfg.get("forward_vel_max", 8.0),
        reward_cfg.get("forward_vel_weight", 0.0),
    )

    raw_alive = reward_alive(reward_cfg.get("alive_bonus", 0.1))
    height_frac = jnp.clip(
        (pelvis_z - healthy_z_min) / (healthy_z_max - healthy_z_min),
        0.0,
        1.0,
    )
    has_foot_contact = _check_foot_contact(data, foot_indices)
    foot_contact_gate = reward_cfg.get("foot_contact_gate", 0.0)
    if foot_contact_gate > 0:
        alive_gate = height_frac * has_foot_contact.astype(jnp.float32)
    else:
        alive_gate = height_frac
    r_alive = raw_alive * alive_gate

    r_energy = reward_energy(action, n_actuators, reward_cfg.get("energy_penalty_weight", 0.001))
    r_posture, _ = reward_posture(root_quat, max_tilt_angle, reward_cfg.get("posture_weight", 0.2))

    components: dict[str, Array] = {
        "forward": r_forward,
        "alive": r_alive,
        "energy": r_energy,
        "posture": r_posture,
        "foot_contact": reward_cfg.get("foot_contact_weight", 0.0) * has_foot_contact.astype(jnp.float32),
    }

    height_w = reward_cfg.get("height_weight", 0.0)
    if height_w > 0:
        _target_z = target_standing_z if target_standing_z is not None else healthy_z_max
        components["height"] = reward_height_maintenance(pelvis_z, healthy_z_min, _target_z, height_w)

    nosedive_w = reward_cfg.get("nosedive_weight", 0.0)
    if nosedive_w > 0:
        r_nd, _ = reward_nosedive(root_quat, nosedive_w, natural_forward_z)
        components["nosedive"] = r_nd

    drift_w = reward_cfg.get("drift_penalty_weight", 0.0)
    if drift_w > 0:
        r_dr, _ = reward_drift_penalty(data.xpos[root_body_id, :2], jnp.zeros(2), drift_w)
        components["drift"] = r_dr

    speed_w = reward_cfg.get("speed_penalty_weight", 0.0)
    if speed_w > 0:
        r_sp, _ = reward_speed_penalty(vel_2d, speed_w, reward_cfg.get("speed_penalty_threshold", 0.10))
        components["speed"] = r_sp

    spin_w = reward_cfg.get("spin_penalty_weight", 0.0)
    if spin_w > 0:
        angvel = data.sensordata[sensor_gyro_start : sensor_gyro_start + 3]
        r_sn, _ = reward_angular_velocity_penalty(angvel, spin_w)
        components["spin"] = r_sn

    smoothness_w = reward_cfg.get("smoothness_weight", 0.0)
    if smoothness_w > 0:
        smooth_ref = prev_action if prev_action is not None else jnp.zeros_like(action)
        r_sm, _ = reward_action_smoothness(action, smooth_ref, n_actuators, smoothness_w)
        components["smoothness"] = r_sm

    tail_w = reward_cfg.get("tail_stability_weight", 0.0)
    if tail_w > 0 and sensor_tail_gyro_start is not None:
        tail_angvel = data.sensordata[sensor_tail_gyro_start : sensor_tail_gyro_start + 3]
        r_tail, _ = reward_angular_velocity_penalty(tail_angvel, tail_w)
        components["tail_stability"] = r_tail

    heading_w = reward_cfg.get("heading_weight", 0.0)
    if heading_w > 0:
        body_fwd_2d = quat_to_forward_2d(root_quat)
        r_heading, _ = reward_heading_alignment(body_fwd_2d, forward_dir, heading_w)
        components["heading"] = r_heading

    lateral_w = reward_cfg.get("lateral_penalty_weight", 0.0)
    if lateral_w > 0:
        body_fwd_2d_lat = quat_to_forward_2d(root_quat)
        r_lateral, _ = reward_lateral_velocity_penalty(vel_2d, body_fwd_2d_lat, lateral_w)
        components["lateral"] = r_lateral

    # Diagnostic state variables
    components["_pelvis_z"] = pelvis_z
    components["_forward_z"] = quat_to_forward_z(root_quat)
    components["_has_foot_contact"] = has_foot_contact.astype(jnp.float32)

    return components


# ---------------------------------------------------------------------------
# Termination
# ---------------------------------------------------------------------------


def is_terminated(
    data: Any,
    *,
    root_body_id: int,
    healthy_z_range: tuple[float, float],
    max_tilt_angle: float,
    natural_forward_z: float,
    nosedive_threshold: float = 0.5,
    body_height_checks: tuple[tuple[int, float], ...] = (),
    site_height_checks: tuple[tuple[int, float], ...] = (),
    sensor_quat_start: int = 6,
) -> Array:
    """Composite termination check matching ``mjx_env.py`` step logic."""
    root_quat = data.sensordata[sensor_quat_start : sensor_quat_start + 4]
    body_z = data.xpos[root_body_id, 2]
    tilt = quat_to_tilt(root_quat)

    terminated, _ = check_height_tilt_termination(
        body_z,
        tilt,
        healthy_z_range,
        max_tilt_angle,
    )

    forward_z = quat_to_forward_z(root_quat)
    nosedive_terminated, _ = check_nosedive_termination(
        forward_z,
        natural_forward_z,
        threshold=nosedive_threshold,
    )
    terminated = terminated | nosedive_terminated

    for bid, z_thresh in body_height_checks:
        terminated = terminated | (data.xpos[bid, 2] < z_thresh)

    for sid, z_thresh in site_height_checks:
        terminated = terminated | (data.site_xpos[sid, 2] < z_thresh)

    return terminated


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _check_foot_contact(data: Any, foot_indices: tuple[int, ...]) -> Array:
    """Check if any foot sensor reads above threshold."""
    import jax.numpy as jnp

    threshold = 0.1  # Newtons
    has_contact = jnp.bool_(False)
    for idx in foot_indices:
        has_contact = has_contact | (data.sensordata[idx] > threshold)
    return has_contact
