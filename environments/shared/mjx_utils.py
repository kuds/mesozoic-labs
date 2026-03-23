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


def unscale_action_jax(ctrl, ctrl_range):
    """Inverse of :func:`scale_action_jax`: control range → [-1, 1]."""

    ctrl_min = ctrl_range[:, 0]
    ctrl_max = ctrl_range[:, 1]
    return 2.0 * (ctrl - ctrl_min) / (ctrl_max - ctrl_min + 1e-8) - 1.0
