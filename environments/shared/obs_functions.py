"""Backend-agnostic pure observation-building functions.

These functions construct observation vectors from raw simulation state
arrays.  They work with both ``np.ndarray`` (Gymnasium / SB3 path) and
``jnp.ndarray`` (MJX / JAX path) because they only use operations in
the intersection of NumPy and JAX.

Each species has a thin wrapper that extracts the relevant slices from
the MuJoCo ``data`` object and passes them to these functions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

Array = Any  # np.ndarray | jnp.ndarray


def _array_mod(arr: Array) -> Any:
    """Return the array module (numpy or jax.numpy) for *arr*."""
    cls_module = type(arr).__module__
    if cls_module.startswith("jax"):
        import jax.numpy as jnp

        return jnp
    return np


# ---------------------------------------------------------------------------
# Sensor layout configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SensorLayout:
    """Immutable description of where sensor readings live in ``sensordata``.

    Each field is an ``int`` index into the flat ``sensordata`` array.
    Slices are built by the caller using these start indices and known
    widths (gyro=3, accel=3, quat=4, touch=1).
    """

    gyro_start: int = 0
    accel_start: int = 3
    quat_start: int = 6
    # Species-specific foot sensor indices (filled per species).  One entry
    # per foot; each contributes exactly one observation channel.
    foot_indices: tuple[int, ...] = ()
    # Optional extra touch sensors to add into each foot's channel, aligned
    # with ``foot_indices``.  A MuJoCo touch sensor only sums contacts on
    # geoms of its site's own body, so a species whose toes are separate
    # bodies needs one sensor per toe and sums them here.  Empty (the
    # default, and the case for every species with single-body feet) leaves
    # the reading and the observation width exactly as they were.
    foot_aux_indices: tuple[tuple[int, ...], ...] = ()

    def foot_contact_values(self, sensordata: Array) -> list[Array]:
        """Return one total contact force per foot, pad plus any aux sensors."""
        totals = []
        for position, index in enumerate(self.foot_indices):
            total = sensordata[index]
            if position < len(self.foot_aux_indices):
                for aux_index in self.foot_aux_indices[position]:
                    total = total + sensordata[aux_index]
            totals.append(total)
        return totals


# ---------------------------------------------------------------------------
# Bipedal observation builder (Raptor, T-Rex)
# ---------------------------------------------------------------------------


def build_bipedal_obs(
    qpos: Array,
    qvel: Array,
    sensordata: Array,
    pelvis_xpos: Array,
    target_pos: Array,
    sensor_layout: SensorLayout,
    root_qpos_dim: int = 7,
    root_qvel_dim: int = 6,
) -> Array:
    """Construct observation vector for bipedal species.

    Works with both ``np.ndarray`` and ``jnp.ndarray``.

    Args:
        qpos: Full generalized positions.
        qvel: Full generalized velocities.
        sensordata: Full sensor data array.
        pelvis_xpos: Pelvis Cartesian position (3,).
        target_pos: Target (prey) Cartesian position (3,).
        sensor_layout: Sensor index layout.
        root_qpos_dim: Number of root freejoint position elements to skip.
        root_qvel_dim: Number of root freejoint velocity elements to skip.

    Returns:
        Flat observation array (float32).
    """
    xp = _array_mod(qpos)

    joint_pos = qpos[root_qpos_dim:]
    joint_vel = qvel[root_qvel_dim:]

    sl = sensor_layout
    pelvis_quat = sensordata[sl.quat_start : sl.quat_start + 4]
    pelvis_gyro = sensordata[sl.gyro_start : sl.gyro_start + 3]
    pelvis_accel = sensordata[sl.accel_start : sl.accel_start + 3]
    pelvis_linvel = qvel[:3]

    foot_contacts = xp.array(sl.foot_contact_values(sensordata))

    target_rel = target_pos - pelvis_xpos
    target_dist = xp.linalg.norm(target_rel)
    target_dir = target_rel / (target_dist + 1e-8)

    obs = xp.concatenate(
        [
            joint_pos,
            joint_vel,
            pelvis_quat,
            pelvis_gyro,
            pelvis_linvel,
            pelvis_accel,
            foot_contacts,
            target_dir,
            xp.array([target_dist]),
        ]
    )
    return obs.astype(xp.float32)


# ---------------------------------------------------------------------------
# Quadrupedal observation builder (Brachiosaurus)
# ---------------------------------------------------------------------------


def build_quadruped_obs(
    qpos: Array,
    qvel: Array,
    sensordata: Array,
    torso_xpos: Array,
    target_pos: Array,
    sensor_layout: SensorLayout,
    root_qpos_dim: int = 7,
    root_qvel_dim: int = 6,
) -> Array:
    """Construct observation vector for quadrupedal species.

    Same structure as :func:`build_bipedal_obs` but with 4 foot contacts
    and using "torso" terminology.  Kept separate for clarity; the
    underlying implementation is identical.
    """
    return build_bipedal_obs(
        qpos=qpos,
        qvel=qvel,
        sensordata=sensordata,
        pelvis_xpos=torso_xpos,
        target_pos=target_pos,
        sensor_layout=sensor_layout,
        root_qpos_dim=root_qpos_dim,
        root_qvel_dim=root_qvel_dim,
    )
