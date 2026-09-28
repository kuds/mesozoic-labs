"""FROZEN (D-D17): ``make_obs_fn``, kept only because the plant contract hashes it.

D-D17 retired the JAX/MJX runtime this module set up (cleanup PR-B; it is
recoverable from the ``0.3.8`` tag or PR-B's first parent).
``plant_contract/policy_layer.py`` still hashes the tokens of ``make_obs_fn``
into the policy-interface digests of trex, velociraptor, brachiosaurus and
dibothrosuchus. It is never executed, and nothing trains on it. Do not edit,
reformat or move it, signature included: a token change moves four species'
plant identities. This docstring is not hashed; ``test_plant_contract_frozen_mjx.py``
pins the digest. See ``environments/shared/mjx_env.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # The retired setup context. make_obs_fn's hashed signature still names it.
    SpeciesContext = Any


def make_obs_fn(ctx: SpeciesContext):
    """Create an observation function bound to the species context.

    Returns a function ``get_obs(data) -> obs_array`` suitable for
    ``evaluate_policy_cpu`` and ``record_training_video``.

    The observation embeds the target direction/distance, so it must point
    at the same target the evaluation's success detection uses — the
    model's target body (prey/food).  A hardcoded origin target used to
    flip the perceived target direction backwards as the agent walked
    away, so stage-3 gates and videos evaluated the wrong task.
    """
    import jax.numpy as jnp
    import mujoco

    from .obs_functions import build_bipedal_obs

    root_body_id = ctx.root_body_id
    sensor_layout = ctx.sensor_layout
    target_body_id = mujoco.mj_name2id(ctx.mj_model, mujoco.mjtObj.mjOBJ_BODY, ctx.target_body_name)

    def get_obs(data):
        if target_body_id >= 0:
            target_pos = data.xpos[target_body_id]
        else:
            target_pos = jnp.zeros(3)
        return build_bipedal_obs(
            qpos=data.qpos,
            qvel=data.qvel,
            sensordata=data.sensordata,
            pelvis_xpos=data.xpos[root_body_id],
            target_pos=target_pos,
            sensor_layout=sensor_layout,
            # CPU evaluation of the walker trunk: the command segment is
            # explicit zeros (BEHAVIOR_RECIPES_PLAN §4.6, command_mode "none").
            command=jnp.zeros(3, dtype=jnp.float32),
        )

    return get_obs
