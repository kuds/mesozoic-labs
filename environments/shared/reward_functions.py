"""Backend-agnostic pure reward functions for Mesozoic Labs environments.

Every function in this module accepts plain array inputs (positions,
velocities, sensor data) and returns scalar rewards.  No ``self``,
no MuJoCo data objects.  All operations use the intersection of
NumPy and JAX (``+``, ``-``, ``*``, ``/``, ``clip``, ``sum``, ``dot``,
``norm``) so that the same code works with both ``np.ndarray`` and
``jnp.ndarray``.

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
) -> tuple[float, float]:
    """Forward velocity reward, normalised to ``[-weight, +weight]``.

    Returns:
        (reward, raw_forward_vel) tuple.
    """
    xp = _array_mod(vel_2d)
    forward_vel = float(xp.dot(vel_2d, forward_dir_2d))
    forward_vel_norm = float(xp.clip(forward_vel / vel_max, -1.0, 1.0))
    reward = weight * forward_vel_norm
    return reward, forward_vel


def reward_backward_penalty(
    forward_vel: float,
    vel_max: float,
    weight: float,
) -> tuple[float, float]:
    """Backward velocity penalty.

    Returns:
        (reward, backward_vel) tuple.
    """
    backward_vel = max(0.0, -forward_vel)
    backward_vel_norm = min(backward_vel / vel_max, 1.0)
    reward = -weight * backward_vel_norm
    return reward, backward_vel


def reward_drift_penalty(
    current_pos_2d: Array,
    initial_pos_2d: Array,
    weight: float,
) -> tuple[float, float]:
    """Quadratic drift penalty (horizontal displacement from spawn).

    Returns:
        (reward, drift_distance) tuple.
    """
    xp = _array_mod(current_pos_2d)
    drift_2d = current_pos_2d - initial_pos_2d
    drift_dist = float(xp.linalg.norm(drift_2d))
    drift_norm = drift_dist / 2.0
    reward = -weight * (drift_norm**2)
    return reward, drift_dist


def reward_alive(alive_bonus: float) -> float:
    """Constant alive bonus."""
    return alive_bonus


def reward_energy(action: Array, n_actuators: int, weight: float) -> float:
    """Normalised energy penalty: ``-weight * mean(action**2)``."""
    xp = _array_mod(action)
    energy = float(xp.sum(xp.square(action)))
    energy_norm = energy / n_actuators
    return -weight * energy_norm


def reward_action_smoothness(
    action: Array,
    prev_action: Array | None,
    n_actuators: int,
    weight: float,
) -> tuple[float, float]:
    """Action-smoothness penalty.

    Returns:
        (reward, action_delta) tuple.
    """
    if prev_action is None:
        return 0.0, 0.0
    xp = _array_mod(action)
    action_delta = float(xp.sum(xp.square(action - prev_action)))
    max_action_delta = n_actuators * 4.0
    action_delta_norm = action_delta / max_action_delta
    reward = -weight * action_delta_norm
    return reward, action_delta


# ---------------------------------------------------------------------------
# Posture / orientation
# ---------------------------------------------------------------------------


def quat_to_tilt(quat: Array) -> float:
    """Tilt angle (radians) between body up-axis and world up."""
    xp = _array_mod(quat)
    _w, x, y, _z = quat[0], quat[1], quat[2], quat[3]
    body_up_z = 1.0 - 2.0 * (x * x + y * y)
    return float(xp.arccos(xp.clip(body_up_z, -1.0, 1.0)))


def quat_to_forward_2d(quat: Array) -> Array:
    """Body forward direction (+X local) projected into XY plane (normalised)."""
    xp = _array_mod(quat)
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]
    body_forward_x = 1.0 - 2.0 * (y * y + z * z)
    body_forward_y = 2.0 * (x * y + w * z)
    body_forward_2d = xp.array([float(body_forward_x), float(body_forward_y)])
    length = float(xp.linalg.norm(body_forward_2d))
    if length > 1e-6:
        body_forward_2d = body_forward_2d / length
    return body_forward_2d


def quat_to_forward_z(quat: Array) -> float:
    """Z-component of the body's local X-axis in world frame (nosedive indicator)."""
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]
    return float(2.0 * (x * z - w * y))


def reward_posture(
    quat: Array,
    max_tilt_angle: float,
    weight: float,
) -> tuple[float, float]:
    """Quadratic tilt penalty.

    Returns:
        (reward, tilt_angle) tuple.
    """
    tilt_angle = quat_to_tilt(quat)
    tilt_angle_norm = min(tilt_angle / max_tilt_angle, 1.0)
    reward = -weight * (tilt_angle_norm**2)
    return reward, tilt_angle


def reward_nosedive(
    quat: Array,
    weight: float,
    natural_forward_z: float,
) -> tuple[float, float]:
    """Nosedive penalty (excessive forward pitch beyond natural lean).

    Returns:
        (reward, forward_z) tuple.
    """
    forward_z = quat_to_forward_z(quat)
    nosedive_excess = max(0.0, -(forward_z - natural_forward_z))
    reward = -weight * nosedive_excess
    return reward, forward_z


def reward_height_maintenance(
    body_height: float,
    healthy_z_min: float,
    target_z: float,
    weight: float,
) -> float:
    """Smooth gradient toward staying at target height."""
    height_frac = max(0.0, min((body_height - healthy_z_min) / (target_z - healthy_z_min), 1.0))
    return weight * height_frac


# ---------------------------------------------------------------------------
# Angular velocity / stability
# ---------------------------------------------------------------------------


def reward_angular_velocity_penalty(
    angvel: Array,
    weight: float,
    max_angvel: float = 10.0,
) -> tuple[float, float]:
    """Angular velocity (spin) penalty.

    Returns:
        (reward, instability_magnitude) tuple.
    """
    xp = _array_mod(angvel)
    instability = float(xp.linalg.norm(angvel))
    instability_norm = min(instability / max_angvel, 1.0)
    reward = -weight * instability_norm
    return reward, instability


# ---------------------------------------------------------------------------
# Heading / lateral / speed
# ---------------------------------------------------------------------------


def reward_heading_alignment(
    body_forward_2d: Array,
    forward_ref_2d: Array,
    weight: float,
) -> tuple[float, float]:
    """Heading alignment reward (reward facing toward target).

    Returns:
        (reward, heading_alignment_cos) tuple.
    """
    xp = _array_mod(body_forward_2d)
    heading_alignment = float(xp.dot(body_forward_2d, forward_ref_2d))
    reward = weight * heading_alignment
    return reward, heading_alignment


def reward_lateral_velocity_penalty(
    vel_2d: Array,
    body_forward_2d: Array,
    weight: float,
) -> tuple[float, float]:
    """Lateral (crab-walk) velocity penalty.

    Returns:
        (reward, lateral_vel) tuple.
    """
    xp = _array_mod(vel_2d)
    lateral_vel = abs(float(vel_2d[0]) * float(body_forward_2d[1]) - float(vel_2d[1]) * float(body_forward_2d[0]))
    lateral_vel_norm = float(xp.clip(lateral_vel / 5.0, 0.0, 1.0))
    reward = -weight * lateral_vel_norm
    return reward, lateral_vel


def reward_speed_penalty(
    vel_2d: Array,
    weight: float,
    threshold: float = 0.10,
    max_excess: float = 1.0,
) -> tuple[float, float]:
    """Penalise absolute 2D speed exceeding *threshold*.

    Returns:
        (reward, absolute_speed) tuple.
    """
    xp = _array_mod(vel_2d)
    speed = float(xp.linalg.norm(vel_2d))
    excess = max(0.0, speed - threshold)
    excess_norm = min(excess / max_excess, 1.0)
    reward = -weight * excess_norm
    return reward, speed


def reward_idle_penalty(
    vel_2d: Array,
    weight: float,
    threshold: float = 0.05,
) -> tuple[float, float]:
    """Penalise low 2D speed (standing still).

    Returns:
        (reward, absolute_speed) tuple.
    """
    xp = _array_mod(vel_2d)
    speed = float(xp.linalg.norm(vel_2d))
    if speed >= threshold:
        return 0.0, speed
    idle_frac = 1.0 - speed / threshold
    reward = -weight * idle_frac
    return reward, speed


# ---------------------------------------------------------------------------
# Approach / proximity
# ---------------------------------------------------------------------------


def reward_approach_shaping(
    current_distance: float,
    prev_distance: float | None,
    weight: float,
    max_speed: float,
    dt: float,
) -> tuple[float, float]:
    """Approach shaping reward (reward closing distance).

    Returns:
        (reward, approach_delta) tuple.
    """
    if prev_distance is None:
        return 0.0, 0.0
    approach_delta = prev_distance - current_distance
    max_delta = max_speed * dt
    approach_delta_norm = max(-1.0, min(approach_delta / max_delta, 1.0))
    reward = weight * approach_delta_norm
    return reward, approach_delta


def reward_proximity(
    distance: float,
    max_distance: float,
    weight: float,
) -> tuple[float, float]:
    """Continuous proximity reward (1.0 at target, 0.0 at max distance).

    Returns:
        (reward, proximity) tuple.
    """
    proximity = max(0.0, 1.0 - distance / max(max_distance, 1e-6))
    reward = weight * proximity
    return reward, proximity


# ---------------------------------------------------------------------------
# Termination checks
# ---------------------------------------------------------------------------


def check_height_tilt_termination(
    body_z: float,
    tilt_angle: float,
    healthy_z_range: tuple[float, float],
    max_tilt_angle: float,
) -> tuple[bool, str | None]:
    """Check common height and tilt termination conditions.

    Returns:
        (terminated, reason) where reason is ``None`` if not terminated.
    """
    if body_z < healthy_z_range[0]:
        return True, "fallen"
    if body_z > healthy_z_range[1]:
        return True, "too_high"
    if tilt_angle > max_tilt_angle:
        return True, "excessive_tilt"
    return False, None


def check_nosedive_termination(
    forward_z: float,
    natural_forward_z: float,
    threshold: float = 0.5,
) -> tuple[bool, str | None]:
    """Check nosedive termination.

    Returns:
        (terminated, reason).
    """
    if forward_z < natural_forward_z - threshold:
        return True, "nosedive"
    return False, None


def check_distance_contact(
    pos_a: Array,
    pos_b: Array,
    threshold: float,
) -> bool:
    """Check proximity-based contact (JAX-compatible alternative to contact pairs)."""
    xp = _array_mod(pos_a)
    dist = float(xp.linalg.norm(pos_a - pos_b))
    return dist < threshold
