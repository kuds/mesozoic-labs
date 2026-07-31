"""Backend-agnostic pure reward functions for Mesozoic Labs environments.

Every function in this module accepts plain array inputs (positions,
velocities, sensor data) and returns scalar rewards.  No ``self``,
no MuJoCo data objects.  All operations use the intersection of
NumPy and JAX (``+``, ``-``, ``*``, ``/``, ``clip``, ``sum``, ``dot``,
``norm``) so that the same code works with both ``np.ndarray`` and
``jnp.ndarray``.

**JAX-trace-safe**: all functions avoid ``float()``, Python ``min``/
``max``/``abs``/``if`` on array values so they can be used inside
``jax.jit``, ``jax.vmap``, and ``jax.lax.scan`` without breaking
tracing.

The existing Gymnasium environments (``BaseDinoEnv`` subclasses)
delegate to these functions via thin wrappers that extract the
relevant arrays from ``self.data``.  The MJX (JAX) path calls them
directly on JAX arrays.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Type alias: works with both numpy and jax arrays
# ---------------------------------------------------------------------------
Array = Any  # np.ndarray | jnp.ndarray


def _array_mod(arr: Array) -> Any:
    """Return the array module (numpy or jax.numpy) for *arr*."""
    cls_module = type(arr).__module__
    if cls_module.startswith("jax"):
        import jax.numpy as jnp

        return jnp
    return np


# ---------------------------------------------------------------------------
# Core reward components
# ---------------------------------------------------------------------------


def reward_forward_velocity(
    vel_2d: Array,
    forward_dir_2d: Array,
    vel_max: float,
    weight: float,
) -> tuple[Array, Array]:
    """Forward velocity reward, normalised to ``[-weight, +weight]``.

    Returns:
        (reward, raw_forward_vel) tuple.
    """
    xp = _array_mod(vel_2d)
    forward_vel = xp.dot(vel_2d, forward_dir_2d)
    forward_vel_norm = xp.clip(forward_vel / vel_max, -1.0, 1.0)
    reward = weight * forward_vel_norm
    return reward, forward_vel


def reward_backward_penalty(
    forward_vel: Array,
    vel_max: float,
    weight: float,
) -> tuple[Array, Array]:
    """Backward velocity penalty.

    Returns:
        (reward, backward_vel) tuple.
    """
    xp = _array_mod(forward_vel)
    backward_vel = xp.maximum(0.0, -forward_vel)
    backward_vel_norm = xp.minimum(backward_vel / vel_max, 1.0)
    reward = -weight * backward_vel_norm
    return reward, backward_vel


def reward_drift_penalty(
    current_pos_2d: Array,
    initial_pos_2d: Array,
    weight: float,
) -> tuple[Array, Array]:
    """Quadratic drift penalty (horizontal displacement from spawn).

    Returns:
        (reward, drift_distance) tuple.
    """
    xp = _array_mod(current_pos_2d)
    drift_2d = current_pos_2d - initial_pos_2d
    drift_dist = xp.linalg.norm(drift_2d)
    drift_norm = drift_dist / 2.0
    reward = -weight * (drift_norm**2)
    return reward, drift_dist


def reward_alive(alive_bonus: float) -> float:
    """Constant alive bonus."""
    return alive_bonus


def reward_energy(action: Array, n_actuators: int, weight: float) -> Array:
    """Normalised energy penalty: ``-weight * mean(action**2)``."""
    xp = _array_mod(action)
    energy = xp.sum(xp.square(action))
    energy_norm = energy / n_actuators
    return -weight * energy_norm


def reward_action_smoothness(
    action: Array,
    prev_action: Array | None,
    n_actuators: int,
    weight: float,
) -> tuple[Array, Array]:
    """Action-smoothness penalty.

    Returns:
        (reward, action_delta) tuple.
    """
    if prev_action is None:
        return 0.0, 0.0
    xp = _array_mod(action)
    action_delta = xp.sum(xp.square(action - prev_action))
    max_action_delta = n_actuators * 4.0
    action_delta_norm = action_delta / max_action_delta
    reward = -weight * action_delta_norm
    return reward, action_delta


# ---------------------------------------------------------------------------
# Posture / orientation
# ---------------------------------------------------------------------------


def quat_to_tilt(quat: Array) -> Array:
    """Tilt angle (radians) between body up-axis and world up."""
    xp = _array_mod(quat)
    _w, x, y, _z = quat[0], quat[1], quat[2], quat[3]
    body_up_z = 1.0 - 2.0 * (x * x + y * y)
    return xp.arccos(xp.clip(body_up_z, -1.0, 1.0))


def quat_to_forward_2d(quat: Array) -> Array:
    """Body forward direction (+X local) projected into XY plane (normalised)."""
    xp = _array_mod(quat)
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]
    body_forward_x = 1.0 - 2.0 * (y * y + z * z)
    body_forward_y = 2.0 * (x * y + w * z)
    body_forward_2d = xp.array([body_forward_x, body_forward_y])
    length = xp.linalg.norm(body_forward_2d)
    body_forward_2d = xp.where(length > 1e-6, body_forward_2d / length, body_forward_2d)
    return body_forward_2d


def quat_to_forward_z(quat: Array) -> Array:
    """Z-component of the body's local X-axis in world frame (nosedive indicator)."""
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]
    return 2.0 * (x * z - w * y)


def reward_posture(
    quat: Array,
    max_tilt_angle: float,
    weight: float,
) -> tuple[Array, Array]:
    """Quadratic tilt penalty.

    Returns:
        (reward, tilt_angle) tuple.
    """
    xp = _array_mod(quat)
    tilt_angle = quat_to_tilt(quat)
    tilt_angle_norm = xp.minimum(tilt_angle / max_tilt_angle, 1.0)
    reward = -weight * (tilt_angle_norm**2)
    return reward, tilt_angle


def reward_lean_aware_posture(
    quat: Array,
    max_tilt_angle: float,
    weight: float,
    natural_forward_z: float,
) -> tuple[Array, Array]:
    """Penalise deviation from a species' natural forward-leaning posture.

    The world-up vector expressed in body coordinates is compared with a
    target that has the requested forward-axis Z component, zero lateral
    lean, and a positive up component.  This makes the reward invariant to
    yaw while distinguishing correct forward pitch from backward pitch and
    roll.

    The reward uses a normalised squared chord distance instead of
    ``acos(alignment) ** 2``.  The chord form has the same local quadratic
    behaviour but keeps JAX gradients finite at the exact target pose.

    Returns:
        ``(reward, absolute_tilt_angle)``.  The absolute tilt remains relative
        to world-up so termination and historical diagnostics keep their
        existing semantics.
    """
    xp = _array_mod(quat)
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]

    body_forward_z = 2.0 * (x * z - w * y)
    body_up_z = 1.0 - 2.0 * (x * x + y * y)

    target_forward_z = xp.clip(natural_forward_z, -1.0, 1.0)
    target_up_z = xp.sqrt(xp.maximum(0.0, 1.0 - target_forward_z**2))
    # The target's lateral-axis Z component is zero, so its dot-product
    # contribution vanishes.  Roll is still penalised through body_up_z.
    alignment = body_forward_z * target_forward_z + body_up_z * target_up_z

    max_chord_error = 1.0 - xp.cos(max_tilt_angle)
    chord_error_norm = xp.clip((1.0 - alignment) / max_chord_error, 0.0, 1.0)
    reward = -weight * chord_error_norm

    tilt_angle = xp.arccos(xp.clip(body_up_z, -1.0, 1.0))
    return reward, tilt_angle


def reward_nosedive(
    quat: Array,
    weight: float,
    natural_forward_z: float,
) -> tuple[Array, Array]:
    """Nosedive penalty (excessive forward pitch beyond natural lean).

    Returns:
        (reward, forward_z) tuple.
    """
    xp = _array_mod(quat)
    forward_z = quat_to_forward_z(quat)
    nosedive_excess = xp.maximum(0.0, -(forward_z - natural_forward_z))
    reward = -weight * nosedive_excess
    return reward, forward_z


def reward_height_maintenance(
    body_height: Array,
    healthy_z_min: float,
    target_z: float,
    weight: float,
) -> Array:
    """Smooth gradient toward staying at target height."""
    xp = _array_mod(body_height)
    height_frac = xp.clip((body_height - healthy_z_min) / (target_z - healthy_z_min), 0.0, 1.0)
    return weight * height_frac


def reward_target_centered_height(
    body_height: Array,
    target_z: float,
    tolerance: float,
    weight: float,
) -> tuple[Array, Array, Array]:
    """Bounded height reward centred on the desired standing height.

    ``tolerance`` is the Gaussian scale: an absolute error of one tolerance
    receives ``exp(-1)`` quality.  Callers retain the historical one-sided
    :func:`reward_height_maintenance` path when tolerance is zero.

    Returns:
        ``(reward, absolute_height_error, height_quality)``.
    """
    xp = _array_mod(body_height)
    safe_tolerance = xp.maximum(tolerance, 1e-8)
    height_error = xp.abs(body_height - target_z)
    height_quality = xp.exp(-xp.square(height_error / safe_tolerance))
    return weight * height_quality, height_error, height_quality


# ---------------------------------------------------------------------------
# Support / home-pose shaping
# ---------------------------------------------------------------------------


def reward_bilateral_support(
    foot_forces: Array,
    saturation_force: float,
    weight: float,
) -> tuple[Array, Array]:
    """Reward simultaneous load-bearing support on every supplied foot.

    Each foot saturates independently at ``saturation_force`` and the minimum
    quality is used, so unloading either side collapses the bilateral score.

    Returns:
        ``(reward, bilateral_support_quality)`` with quality in ``[0, 1]``.
    """
    xp = _array_mod(foot_forces)
    safe_saturation = xp.maximum(saturation_force, 1e-8)
    per_foot_quality = xp.clip(foot_forces / safe_saturation, 0.0, 1.0)
    support_quality = xp.min(per_foot_quality)
    return weight * support_quality, support_quality


def reward_foot_load_balance(
    foot_forces: Array,
    weight: float,
    min_support_force: float = 0.0,
) -> tuple[Array, Array]:
    """Penalise unequal loading of a bilateral support pair.

    An unsupported pair is treated as maximally imbalanced rather than
    perfectly balanced.  The ratio ``|R - L| / (R + L + 1e-8)`` evaluates to
    **zero** when both feet read zero, so being airborne used to score the same
    as standing evenly on both feet -- and strictly better than honest single
    support, which pays the full penalty:

    ======================  =========  ===================  =======
    state                   bilateral  foot_load_balance    sum
    ======================  =========  ===================  =======
    both feet down, even      +0.600              -0.000     +0.600
    airborne                   0.000              -0.000      0.000
    one foot carries load      0.000              -0.300     -0.300
    ======================  =========  ===================  =======

    That ordering makes flight cheaper than single support, which is the one
    thing a balance stage must never reward -- a policy off the ground cannot
    reject a disturbance at all.

    Args:
        foot_forces: ``(right, left)`` normal forces.
        weight: Penalty weight.
        min_support_force: Total force below which the pair counts as
            unsupported.  The default ``0.0`` closes only the exact ``[0, 0]``
            case and leaves every loaded state numerically unchanged; raise it
            to also deny credit for a token grazing contact.

    Returns:
        ``(reward, normalized_load_imbalance)``.  The diagnostic is zero for
        equal loads, approaches one when one foot carries all the load, and is
        exactly one when the pair is unsupported.
    """
    xp = _array_mod(foot_forces)
    right_force, left_force = foot_forces[0], foot_forces[1]
    total = right_force + left_force
    imbalance = xp.where(
        total > min_support_force,
        xp.abs(right_force - left_force) / (total + 1e-8),
        1.0,
    )
    return -weight * imbalance, imbalance


def reward_soft_home_pose(
    joint_positions: Array,
    home_positions: Array,
    tolerance: float,
    weight: float,
) -> tuple[Array, Array, Array]:
    """Reward a soft joint-space neighbourhood around an XML home pose.

    The quality is the mean per-joint Gaussian rather than a single Gaussian
    of the RMS error.  That keeps one displaced joint from erasing useful
    shaping on every other joint while the RMS remains an intuitive raw
    diagnostic.

    Returns:
        ``(reward, rms_error, mean_pose_quality)``.
    """
    xp = _array_mod(joint_positions)
    safe_tolerance = xp.maximum(tolerance, 1e-8)
    error = joint_positions - home_positions
    rms_error = xp.sqrt(xp.mean(xp.square(error)))
    pose_quality = xp.mean(xp.exp(-xp.square(error / safe_tolerance)))
    return weight * pose_quality, rms_error, pose_quality


def reward_head_clearance(
    head_height: Array,
    target_height: float,
    tolerance: float,
    weight: float,
) -> tuple[Array, Array]:
    """Smoothly reward head clearance between a floor and full-credit target.

    The zero-quality floor is ``target_height - tolerance``.  The cubic
    smoothstep has zero slope at both endpoints, avoiding a sharp gradient at
    the head-contact boundary while remaining exactly bounded in ``[0, 1]``.

    Returns:
        ``(reward, head_clearance_quality)``.
    """
    xp = _array_mod(head_height)
    safe_tolerance = xp.maximum(tolerance, 1e-8)
    clearance_floor = target_height - tolerance
    normalized = xp.clip((head_height - clearance_floor) / safe_tolerance, 0.0, 1.0)
    clearance_quality = normalized * normalized * (3.0 - 2.0 * normalized)
    return weight * clearance_quality, clearance_quality


# ---------------------------------------------------------------------------
# Angular velocity / stability
# ---------------------------------------------------------------------------


def reward_angular_velocity_penalty(
    angvel: Array,
    weight: float,
    max_angvel: float = 10.0,
) -> tuple[Array, Array]:
    """Angular velocity (spin) penalty.

    Returns:
        (reward, instability_magnitude) tuple.
    """
    xp = _array_mod(angvel)
    instability = xp.linalg.norm(angvel)
    instability_norm = xp.minimum(instability / max_angvel, 1.0)
    reward = -weight * instability_norm
    return reward, instability


# ---------------------------------------------------------------------------
# Heading / lateral / speed
# ---------------------------------------------------------------------------


def reward_heading_alignment(
    body_forward_2d: Array,
    forward_ref_2d: Array,
    weight: float,
) -> tuple[Array, Array]:
    """Heading alignment reward (reward facing toward target).

    Returns:
        (reward, heading_alignment_cos) tuple.
    """
    xp = _array_mod(body_forward_2d)
    heading_alignment = xp.dot(body_forward_2d, forward_ref_2d)
    reward = weight * heading_alignment
    return reward, heading_alignment


def reward_lateral_velocity_penalty(
    vel_2d: Array,
    body_forward_2d: Array,
    weight: float,
) -> tuple[Array, Array]:
    """Lateral (crab-walk) velocity penalty.

    Returns:
        (reward, lateral_vel) tuple.
    """
    xp = _array_mod(vel_2d)
    lateral_vel = xp.abs(vel_2d[0] * body_forward_2d[1] - vel_2d[1] * body_forward_2d[0])
    lateral_vel_norm = xp.clip(lateral_vel / 5.0, 0.0, 1.0)
    reward = -weight * lateral_vel_norm
    return reward, lateral_vel


def reward_speed_penalty(
    vel_2d: Array,
    weight: float,
    threshold: float = 0.10,
    max_excess: float = 1.0,
) -> tuple[Array, Array]:
    """Penalise absolute 2D speed exceeding *threshold*.

    Returns:
        (reward, absolute_speed) tuple.
    """
    xp = _array_mod(vel_2d)
    speed = xp.linalg.norm(vel_2d)
    excess = xp.maximum(0.0, speed - threshold)
    excess_norm = xp.minimum(excess / max_excess, 1.0)
    reward = -weight * excess_norm
    return reward, speed


def reward_idle_penalty(
    vel_2d: Array,
    weight: float,
    threshold: float = 0.05,
) -> tuple[Array, Array]:
    """Penalise low 2D speed (standing still).

    Returns:
        (reward, absolute_speed) tuple.
    """
    xp = _array_mod(vel_2d)
    speed = xp.linalg.norm(vel_2d)
    idle_frac = xp.where(speed >= threshold, 0.0, 1.0 - speed / threshold)
    reward = -weight * idle_frac
    return reward, speed


# ---------------------------------------------------------------------------
# Approach / proximity
# ---------------------------------------------------------------------------


def reward_approach_shaping(
    current_distance: Array,
    prev_distance: Array | None,
    weight: float,
    max_speed: float,
    dt: float,
) -> tuple[Array, Array]:
    """Approach shaping reward (reward closing distance).

    Returns:
        (reward, approach_delta) tuple.
    """
    if prev_distance is None:
        return 0.0, 0.0
    xp = _array_mod(current_distance)
    approach_delta = prev_distance - current_distance
    max_delta = max_speed * dt
    approach_delta_norm = xp.clip(approach_delta / max_delta, -1.0, 1.0)
    reward = weight * approach_delta_norm
    return reward, approach_delta


def reward_proximity(
    distance: Array,
    max_distance: float,
    weight: float,
) -> tuple[Array, Array]:
    """Continuous proximity reward (1.0 at target, 0.0 at max distance).

    Returns:
        (reward, proximity) tuple.
    """
    xp = _array_mod(distance)
    safe_max = max(max_distance, 1e-6)  # max_distance is always a plain float
    proximity = xp.maximum(0.0, 1.0 - distance / safe_max)
    reward = weight * proximity
    return reward, proximity


# ---------------------------------------------------------------------------
# Termination checks
# ---------------------------------------------------------------------------


def check_height_tilt_termination(
    body_z: Array,
    tilt_angle: Array,
    healthy_z_range: tuple[float, float],
    max_tilt_angle: float,
) -> tuple[Array, str | None]:
    """Check common height and tilt termination conditions.

    JAX-trace-safe: uses ``|`` instead of Python ``if`` on traced
    values.  Reason strings are returned for plain NumPy inputs
    (Gymnasium path) but ``None`` under JAX tracing.

    Returns:
        (terminated, reason) where reason is ``None`` if not terminated
        or when running under JAX tracing.
    """
    xp = _array_mod(body_z)
    fallen = body_z < healthy_z_range[0]
    too_high = body_z > healthy_z_range[1]
    tilted = tilt_angle > max_tilt_angle
    terminated = fallen | too_high | tilted
    # Reason strings only for NumPy path (not trace-safe)
    if xp is np:
        if fallen:
            return True, "fallen"
        if too_high:
            return True, "too_high"
        if tilted:
            return True, "excessive_tilt"
        return False, None
    return terminated, None


def check_nosedive_termination(
    forward_z: Array,
    natural_forward_z: float,
    threshold: float = 0.5,
) -> tuple[Array, str | None]:
    """Check nosedive termination.

    Returns:
        (terminated, reason).
    """
    xp = _array_mod(forward_z)
    terminated = forward_z < natural_forward_z - threshold
    if xp is np:
        if terminated:
            return True, "nosedive"
        return False, None
    return terminated, None


def check_distance_contact(
    pos_a: Array,
    pos_b: Array,
    threshold: float,
) -> Array:
    """Check proximity-based contact (JAX-compatible alternative to contact pairs)."""
    xp = _array_mod(pos_a)
    dist = xp.linalg.norm(pos_a - pos_b)
    return dist < threshold
