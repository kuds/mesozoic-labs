"""FROZEN (D-D17): the MJX action-mapping and home-reset helpers the plant contract hashes.

D-D17 retired the JAX/MJX runtime that called these (cleanup PR-B; it is
recoverable from the ``0.3.8`` tag or PR-B's first parent).
``plant_contract/policy_layer.py`` hashes the tokens of
``scale_action_around_nominal_jax`` and ``reset_mujoco_data_to_home`` into the
policy-interface digests of trex, velociraptor, brachiosaurus and
dibothrosuchus, and names ``scale_action_jax`` for a species on the midpoint
mapping (none is). None of them is executed, and nothing trains on them. Do not
edit, reformat or move them: a token change moves four species' plant
identities. This docstring is not hashed; ``test_plant_contract_frozen_mjx.py``
pins the digests. See ``environments/shared/mjx_env.py``.
"""

from __future__ import annotations


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
