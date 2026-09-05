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
    reward_action_jerk,
    reward_action_saturation,
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

Array = Any


def _reward_posture_for_target(
    quat: Array,
    max_tilt_angle: float,
    weight: float,
    posture_target_forward_z: float | None,
) -> tuple[Array, Array]:
    """Select vertical or natural-lean posture shaping at trace time."""
    if posture_target_forward_z is None:
        return reward_posture(quat, max_tilt_angle, weight)
    return reward_lean_aware_posture(
        quat,
        max_tilt_angle,
        weight,
        posture_target_forward_z,
    )


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
    posture_target_forward_z: float | None = None,
    sensor_quat_start: int = 6,
    sensor_gyro_start: int = 0,
    foot_indices: tuple[int, ...] = (10, 11),
    foot_sensor_groups: tuple[tuple[int, ...], ...] | None = None,
    leg_home_pose_qpos_indices: tuple[int, ...] = (),
    leg_home_pose_targets: Array | None = None,
    neck_posture_qpos_indices: tuple[int, ...] = (),
    neck_posture_targets: Array | None = None,
    tail_home_pose_qpos_indices: tuple[int, ...] = (),
    tail_home_pose_targets: Array | None = None,
    head_clearance_site_id: int | None = None,
    prev_action: Array | None = None,
    prev_prev_action: Array | None = None,
    initial_pos_2d: Array | None = None,
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
    additional_terminated: Array | None = None,
    aggregated_foot_forces: Array | None = None,
) -> Array:
    """Compute total scalar reward matching ``mjx_env.py`` step logic.

    Parameters are split into ``reward_cfg`` (per-stage, read at trace time)
    and keyword arguments (per-species constants).

    ``forward_ref_2d`` is the 2D direction used for forward-velocity and
    heading-alignment rewards.  When tracking a target, pass the normalised
    agent-to-target direction so that both rewards are consistent with
    ``MJXDinoEnv.step()``.  Defaults to world +X ``[1, 0]``.

    ``prev_action`` / ``prev_prev_action`` are the two action lags the
    smoothness and jerk terms difference against.  ``None`` means zeros —
    the MJX kernel's reset carries — not "no charge".
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

    # Substep-MIN override: the training backends aggregate touch force
    # across the frame_skip physics substeps (see mjx_env step_fn and
    # BaseDinoEnv.step); a caller scoring a stepped trajectory passes the
    # aggregate so eval prices the same quantity training optimizes.
    # Default None keeps single-state semantics for the parity tests.
    if aggregated_foot_forces is not None:
        foot_forces = aggregated_foot_forces
    else:
        foot_forces = _per_foot_forces(data, foot_indices, foot_sensor_groups)
    has_foot_contact = jnp.any(foot_forces > 0.1)
    r_bilateral_support, bilateral_support_quality = reward_bilateral_support(
        foot_forces,
        reward_cfg.get("foot_contact_saturation_force", 100.0),
        reward_cfg.get("bilateral_support_weight", 0.0),
    )
    r_foot_load_balance, _ = reward_foot_load_balance(
        foot_forces[:2],
        reward_cfg.get("foot_load_balance_weight", 0.0),
        reward_cfg.get("foot_load_balance_min_support_force", 0.0),
        reward_cfg.get("foot_load_balance_airborne_penalty", 0.0),
    )

    # The opt-in path is formula-identical to Gymnasium.  Fraction zero
    # preserves the historical MJX height/contact-gated alive reward.
    raw_alive = reward_alive(reward_cfg.get("alive_bonus", 0.1))
    support_alive_fraction = reward_cfg.get("support_conditioned_alive_fraction", 0.0)
    if support_alive_fraction > 0:
        alive_gate = (1.0 - support_alive_fraction) + support_alive_fraction * bilateral_support_quality
    else:
        height_frac = jnp.clip(
            (pelvis_z - healthy_z_min) / (healthy_z_max - healthy_z_min),
            0.0,
            1.0,
        )
        foot_contact_gate = reward_cfg.get("foot_contact_gate", 0.0)
        if foot_contact_gate > 0:
            alive_gate = height_frac * has_foot_contact.astype(jnp.float32)
        else:
            alive_gate = height_frac
    r_alive = raw_alive * alive_gate

    r_energy = reward_energy(action, n_actuators, reward_cfg.get("energy_penalty_weight", 0.001))
    r_posture, _ = _reward_posture_for_target(
        root_quat,
        max_tilt_angle,
        reward_cfg.get("posture_weight", 0.2),
        posture_target_forward_z,
    )

    total = r_forward + r_alive + r_energy + r_posture + r_bilateral_support + r_foot_load_balance

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
        total = total + foot_contact_w * has_foot_contact.astype(jnp.float32)

    height_w = reward_cfg.get("height_weight", 0.0)
    if height_w > 0:
        _target_z = target_standing_z if target_standing_z is not None else healthy_z_max
        height_target_tolerance = reward_cfg.get("height_target_tolerance", 0.0)
        if height_target_tolerance > 0:
            r_height, _, _ = reward_target_centered_height(
                pelvis_z,
                _target_z,
                height_target_tolerance,
                height_w,
            )
        else:
            # Legacy one-sided gradient saturates at the standing height.
            r_height = reward_height_maintenance(pelvis_z, healthy_z_min, _target_z, height_w)
        total = total + r_height

    leg_home_pose_w = reward_cfg.get("leg_home_pose_weight", 0.0)
    if leg_home_pose_w > 0 and leg_home_pose_targets is not None:
        leg_joint_positions = jnp.take(data.qpos, jnp.asarray(leg_home_pose_qpos_indices))
        r_leg_home_pose, _, _ = reward_soft_home_pose(
            leg_joint_positions,
            leg_home_pose_targets,
            reward_cfg.get("leg_home_pose_tolerance", 0.35),
            leg_home_pose_w,
            reward_cfg.get("leg_home_pose_broad_fraction", 0.0),
            reward_cfg.get("leg_home_pose_broad_scale", 6.0),
        )
        total = total + r_leg_home_pose

    head_clearance_w = reward_cfg.get("head_clearance_weight", 0.0)
    if head_clearance_w > 0 and head_clearance_site_id is not None:
        r_head_clearance, _ = reward_head_clearance(
            data.site_xpos[head_clearance_site_id, 2],
            reward_cfg.get("head_clearance_target", 0.60),
            reward_cfg.get("head_clearance_tolerance", 0.48),
            head_clearance_w,
        )
        total = total + r_head_clearance

    neck_posture_w = reward_cfg.get("neck_posture_weight", 0.0)
    if neck_posture_w > 0 and neck_posture_targets is not None:
        neck_joint_positions = jnp.take(data.qpos, jnp.asarray(neck_posture_qpos_indices))
        r_neck_posture, _, _ = reward_soft_home_pose(
            neck_joint_positions,
            neck_posture_targets,
            reward_cfg.get("neck_posture_tolerance", 0.35),
            neck_posture_w,
        )
        total = total + r_neck_posture

    tail_home_pose_w = reward_cfg.get("tail_home_pose_weight", 0.0)
    if tail_home_pose_w > 0 and tail_home_pose_targets is not None:
        tail_joint_positions = jnp.take(data.qpos, jnp.asarray(tail_home_pose_qpos_indices))
        r_tail_home_pose, _, _ = reward_soft_home_pose(
            tail_joint_positions,
            tail_home_pose_targets,
            reward_cfg.get("tail_home_pose_tolerance", 0.10),
            tail_home_pose_w,
        )
        total = total + r_tail_home_pose

    action_saturation_w = reward_cfg.get("action_saturation_weight", 0.0)
    if action_saturation_w > 0:
        r_action_saturation, _ = reward_action_saturation(
            action,
            action_saturation_w,
            reward_cfg.get("action_saturation_threshold", 0.9),
        )
        total = total + r_action_saturation

    nosedive_w = reward_cfg.get("nosedive_weight", 0.0)
    if nosedive_w > 0:
        r_nd, _ = reward_nosedive(root_quat, nosedive_w, natural_forward_z)
        total = total + r_nd

    drift_w = reward_cfg.get("drift_penalty_weight", 0.0)
    if drift_w > 0:
        drift_origin = initial_pos_2d if initial_pos_2d is not None else jnp.zeros(2)
        r_dr, _ = reward_drift_penalty(data.xpos[root_body_id, :2], drift_origin, drift_w)
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

    # Frequency-aware jerk (second difference), gated independently of the
    # smoothness term exactly as the MJX step is.  Missing lags are zeros,
    # not "no charge": the training kernel's prev/prev_prev carries reset
    # to zeros, so its first two steps ARE charged and this composer must
    # price them identically.  (SB3 leaves both lags None and skips those
    # steps -- a pre-existing backend difference this composer does not
    # arbitrate; it scores the MJX plant.)
    jerk_w = reward_cfg.get("action_jerk_weight", 0.0)
    if jerk_w > 0:
        jerk_ref = prev_action if prev_action is not None else jnp.zeros_like(action)
        jerk_ref_2 = prev_prev_action if prev_prev_action is not None else jnp.zeros_like(action)
        r_jk, _ = reward_action_jerk(action, jerk_ref, jerk_ref_2, n_actuators, jerk_w)
        total = total + r_jk

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

    # Head/claw/snout proximity reward: continuous gradient for final positioning.
    # The weight is species-flavoured in the shared stage TOMLs, so accept every
    # spelling; a new species adds its key here rather than nesting another get.
    proximity_w = 0.0
    for proximity_key in (
        "bite_head_proximity_weight",
        "strike_claw_proximity_weight",
        "food_head_proximity_weight",
        "snap_snout_proximity_weight",
    ):
        if proximity_key in reward_cfg:
            proximity_w = reward_cfg[proximity_key]
            break
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
        if additional_terminated is not None:
            terminated = terminated | additional_terminated
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
    posture_target_forward_z: float | None = None,
    sensor_quat_start: int = 6,
    sensor_gyro_start: int = 0,
    foot_indices: tuple[int, ...] = (10, 11),
    foot_sensor_groups: tuple[tuple[int, ...], ...] | None = None,
    leg_home_pose_qpos_indices: tuple[int, ...] = (),
    leg_home_pose_targets: Array | None = None,
    neck_posture_qpos_indices: tuple[int, ...] = (),
    neck_posture_targets: Array | None = None,
    tail_home_pose_qpos_indices: tuple[int, ...] = (),
    tail_home_pose_targets: Array | None = None,
    head_clearance_site_id: int | None = None,
    prev_action: Array | None = None,
    prev_prev_action: Array | None = None,
    initial_pos_2d: Array | None = None,
    sensor_tail_gyro_start: int | None = None,
    forward_ref_2d: Array | None = None,
    aggregated_foot_forces: Array | None = None,
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

    # Substep-MIN override: the training backends aggregate touch force
    # across the frame_skip physics substeps (see mjx_env step_fn and
    # BaseDinoEnv.step); a caller scoring a stepped trajectory passes the
    # aggregate so eval prices the same quantity training optimizes.
    # Default None keeps single-state semantics for the parity tests.
    if aggregated_foot_forces is not None:
        foot_forces = aggregated_foot_forces
    else:
        foot_forces = _per_foot_forces(data, foot_indices, foot_sensor_groups)
    has_foot_contact = jnp.any(foot_forces > 0.1)
    r_bilateral_support, bilateral_support_quality = reward_bilateral_support(
        foot_forces,
        reward_cfg.get("foot_contact_saturation_force", 100.0),
        reward_cfg.get("bilateral_support_weight", 0.0),
    )
    r_foot_load_balance, foot_load_imbalance = reward_foot_load_balance(
        foot_forces[:2],
        reward_cfg.get("foot_load_balance_weight", 0.0),
        reward_cfg.get("foot_load_balance_min_support_force", 0.0),
        reward_cfg.get("foot_load_balance_airborne_penalty", 0.0),
    )

    raw_alive = reward_alive(reward_cfg.get("alive_bonus", 0.1))
    support_alive_fraction = reward_cfg.get("support_conditioned_alive_fraction", 0.0)
    if support_alive_fraction > 0:
        alive_gate = (1.0 - support_alive_fraction) + support_alive_fraction * bilateral_support_quality
    else:
        height_frac = jnp.clip(
            (pelvis_z - healthy_z_min) / (healthy_z_max - healthy_z_min),
            0.0,
            1.0,
        )
        foot_contact_gate = reward_cfg.get("foot_contact_gate", 0.0)
        if foot_contact_gate > 0:
            alive_gate = height_frac * has_foot_contact.astype(jnp.float32)
        else:
            alive_gate = height_frac
    r_alive = raw_alive * alive_gate

    r_energy = reward_energy(action, n_actuators, reward_cfg.get("energy_penalty_weight", 0.001))
    r_posture, _ = _reward_posture_for_target(
        root_quat,
        max_tilt_angle,
        reward_cfg.get("posture_weight", 0.2),
        posture_target_forward_z,
    )

    components: dict[str, Array] = {
        "forward": r_forward,
        "alive": r_alive,
        "energy": r_energy,
        "posture": r_posture,
        "foot_contact": reward_cfg.get("foot_contact_weight", 0.0) * has_foot_contact.astype(jnp.float32),
        "bilateral_support": r_bilateral_support,
        "foot_load_balance": r_foot_load_balance,
    }

    _target_z = target_standing_z if target_standing_z is not None else healthy_z_max
    height_w = reward_cfg.get("height_weight", 0.0)
    height_target_tolerance = reward_cfg.get("height_target_tolerance", 0.0)
    if height_target_tolerance > 0:
        r_height, height_error, height_quality = reward_target_centered_height(
            pelvis_z,
            _target_z,
            height_target_tolerance,
            height_w,
        )
    else:
        height_error = jnp.abs(pelvis_z - _target_z)
        height_quality = jnp.clip(
            (pelvis_z - healthy_z_min) / (_target_z - healthy_z_min),
            0.0,
            1.0,
        )
        r_height = height_w * height_quality
    components["height"] = r_height

    if leg_home_pose_targets is not None:
        leg_joint_positions = jnp.take(data.qpos, jnp.asarray(leg_home_pose_qpos_indices))
        r_leg_home_pose, leg_home_pose_error, leg_home_pose_quality = reward_soft_home_pose(
            leg_joint_positions,
            leg_home_pose_targets,
            reward_cfg.get("leg_home_pose_tolerance", 0.35),
            reward_cfg.get("leg_home_pose_weight", 0.0),
            reward_cfg.get("leg_home_pose_broad_fraction", 0.0),
            reward_cfg.get("leg_home_pose_broad_scale", 6.0),
        )
        components["leg_home_pose"] = r_leg_home_pose
        components["_leg_home_pose_error"] = leg_home_pose_error
        components["_leg_home_pose_quality"] = leg_home_pose_quality

    if head_clearance_site_id is not None:
        head_tip_z = data.site_xpos[head_clearance_site_id, 2]
        r_head_clearance, head_clearance_quality = reward_head_clearance(
            head_tip_z,
            reward_cfg.get("head_clearance_target", 0.60),
            reward_cfg.get("head_clearance_tolerance", 0.48),
            reward_cfg.get("head_clearance_weight", 0.0),
        )
        components["head_clearance"] = r_head_clearance
        components["_head_tip_z"] = head_tip_z
        components["_head_pelvis_rel_z"] = head_tip_z - pelvis_z
        components["_head_clearance_quality"] = head_clearance_quality

    if neck_posture_targets is not None:
        neck_joint_positions = jnp.take(data.qpos, jnp.asarray(neck_posture_qpos_indices))
        r_neck_posture, neck_posture_error, neck_posture_quality = reward_soft_home_pose(
            neck_joint_positions,
            neck_posture_targets,
            reward_cfg.get("neck_posture_tolerance", 0.35),
            reward_cfg.get("neck_posture_weight", 0.0),
        )
        components["neck_posture"] = r_neck_posture
        components["_neck_posture_error"] = neck_posture_error
        components["_neck_posture_quality"] = neck_posture_quality

    if tail_home_pose_targets is not None:
        tail_joint_positions = jnp.take(data.qpos, jnp.asarray(tail_home_pose_qpos_indices))
        r_tail_home_pose, tail_home_pose_error, tail_home_pose_quality = reward_soft_home_pose(
            tail_joint_positions,
            tail_home_pose_targets,
            reward_cfg.get("tail_home_pose_tolerance", 0.10),
            reward_cfg.get("tail_home_pose_weight", 0.0),
        )
        components["tail_home_pose"] = r_tail_home_pose
        components["_tail_home_pose_error"] = tail_home_pose_error
        components["_tail_home_pose_quality"] = tail_home_pose_quality

    action_saturation_w = reward_cfg.get("action_saturation_weight", 0.0)
    if action_saturation_w > 0:
        r_action_saturation, action_saturation = reward_action_saturation(
            action,
            action_saturation_w,
            reward_cfg.get("action_saturation_threshold", 0.9),
        )
        components["action_saturation"] = r_action_saturation
        components["_action_saturation_fraction"] = action_saturation

    nosedive_w = reward_cfg.get("nosedive_weight", 0.0)
    if nosedive_w > 0:
        r_nd, _ = reward_nosedive(root_quat, nosedive_w, natural_forward_z)
        components["nosedive"] = r_nd

    drift_w = reward_cfg.get("drift_penalty_weight", 0.0)
    if drift_w > 0:
        drift_origin = initial_pos_2d if initial_pos_2d is not None else jnp.zeros(2)
        r_dr, _ = reward_drift_penalty(data.xpos[root_body_id, :2], drift_origin, drift_w)
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

    # Same zero-lag convention as compute_total_reward (see the note there).
    jerk_w = reward_cfg.get("action_jerk_weight", 0.0)
    if jerk_w > 0:
        jerk_ref = prev_action if prev_action is not None else jnp.zeros_like(action)
        jerk_ref_2 = prev_prev_action if prev_prev_action is not None else jnp.zeros_like(action)
        r_jk, action_jerk = reward_action_jerk(action, jerk_ref, jerk_ref_2, n_actuators, jerk_w)
        components["action_jerk"] = r_jk
        components["_action_jerk"] = action_jerk

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
    components["_bilateral_support_quality"] = bilateral_support_quality
    components["_foot_load_imbalance"] = foot_load_imbalance
    components["_raw_alive"] = jnp.asarray(raw_alive)
    components["_alive_gate"] = alive_gate
    components["_height_error"] = height_error
    components["_height_quality"] = height_quality

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
    """Composite termination check over ONE data snapshot.

    Scores the state it is handed -- a boundary sample.  The training step in
    ``mjx_env.py`` additionally evaluates the body/site height checks at
    EVERY physics substep through its ``fori_loop`` carry, so a strike that
    resolves between control boundaries terminates training but is invisible
    to this single-snapshot checker; auditing a training termination from
    post-step data alone can therefore disagree with the trainer.
    """
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


def _per_foot_forces(
    data: Any,
    foot_indices: tuple[int, ...],
    foot_sensor_groups: tuple[tuple[int, ...], ...] | None,
) -> Array:
    """Return one summed touch force per foot.

    ``foot_sensor_groups`` preserves pad+digit grouping for species such as
    T. rex.  Legacy callers that only provide ``foot_indices`` retain one
    sensor per foot.
    """
    import jax.numpy as jnp

    groups = foot_sensor_groups
    if groups is None:
        groups = tuple((index,) for index in foot_indices)

    forces = []
    for group in groups:
        force = jnp.float32(0.0)
        for index in group:
            force = force + data.sensordata[index]
        forces.append(force)
    return jnp.stack(forces)


def _check_foot_contact(data: Any, foot_indices: tuple[int, ...]) -> Array:
    """Check if any foot sensor reads above threshold."""
    import jax.numpy as jnp

    threshold = 0.1  # Newtons
    has_contact = jnp.bool_(False)
    for idx in foot_indices:
        has_contact = has_contact | (data.sensordata[idx] > threshold)
    return has_contact
