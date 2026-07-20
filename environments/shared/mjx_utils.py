"""JAX/MJX utility helpers for Mesozoic Labs environments.

All JAX imports are **lazy** so that users who only install the ``train``
(SB3) dependencies don't get import errors.  Call :func:`check_jax` at
the top of any module that needs JAX to produce a clear error message.
"""

from __future__ import annotations


def check_jax() -> None:
    """Raise a clear error if JAX/MJX is not installed."""
    try:
        import jax  # noqa: F401
        import mujoco.mjx  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "JAX/MJX training requires the [jax] extras.  Install with:\n  pip install mesozoic-labs[jax]"
        ) from exc


def scale_action_jax(action, ctrl_range):
    """Scale normalised action [-1, 1] to actuator control range.

    Args:
        action: JAX array of shape ``(n_actuators,)`` in ``[-1, 1]``.
        ctrl_range: JAX array of shape ``(n_actuators, 2)`` with
            ``[min, max]`` per actuator.

    Returns:
        Scaled control array.
    """

    ctrl_min = ctrl_range[:, 0]
    ctrl_max = ctrl_range[:, 1]
    return ctrl_min + (action + 1.0) * 0.5 * (ctrl_max - ctrl_min)


def scale_action_around_nominal_jax(action, ctrl_range, nominal_ctrl):
    """Scale normalized actions around a non-midpoint nominal control.

    ``action == 0`` maps exactly to ``nominal_ctrl`` while ``-1`` and ``+1``
    retain access to the actuator's minimum and maximum controls.  The two
    halves are scaled independently because a biomechanically useful nominal
    pose is not generally the midpoint of an actuator's control range.

    Args:
        action: JAX array of shape ``(n_actuators,)`` in ``[-1, 1]``.
        ctrl_range: JAX array of shape ``(n_actuators, 2)`` with
            ``[min, max]`` per actuator.
        nominal_ctrl: JAX array of shape ``(n_actuators,)`` containing the
            control targets for the nominal pose.

    Returns:
        Scaled control array.
    """
    import jax.numpy as jnp

    action = jnp.clip(action, -1.0, 1.0)
    ctrl_min = ctrl_range[:, 0]
    ctrl_max = ctrl_range[:, 1]
    below_nominal = nominal_ctrl + action * (nominal_ctrl - ctrl_min)
    above_nominal = nominal_ctrl + action * (ctrl_max - nominal_ctrl)
    return jnp.where(action < 0.0, below_nominal, above_nominal)


def unscale_action_jax(ctrl, ctrl_range):
    """Inverse of :func:`scale_action_jax`: control range → [-1, 1]."""

    ctrl_min = ctrl_range[:, 0]
    ctrl_max = ctrl_range[:, 1]
    return 2.0 * (ctrl - ctrl_min) / (ctrl_max - ctrl_min + 1e-8) - 1.0


def reset_mujoco_data_to_home(mj_model, mj_data) -> None:
    """Reset CPU MuJoCo data to the named ``home`` keyframe.

    ``mj_resetDataKeyframe`` restores the complete keyframed state: qpos,
    qvel, actuator state, controls, mocap bodies, time, and user data.  This
    helper keeps home-residual MJX training, CPU evaluation, and visualization
    aligned with the Gymnasium environment's reset semantics.

    Raises:
        ValueError: If the model does not define a keyframe named ``home``.
    """
    import mujoco

    home_keyframe_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if home_keyframe_id < 0:
        raise ValueError("home-keyframe residual action mapping requires a keyframe named 'home'")
    mujoco.mj_resetDataKeyframe(mj_model, mj_data, home_keyframe_id)
