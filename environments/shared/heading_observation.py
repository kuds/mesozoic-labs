"""Heading-free root observation frames: the ``bipedal-target-heading-free/v1`` schema.

A policy that reads the root's world quaternion, world linear velocity and
world target direction can learn to depend on its world heading, which no
stance criterion sees (docs/HEADING_INVARIANCE_DESIGN_2026_10.md).  This
schema replaces those three blocks with quantities a rotation about world
vertical leaves unchanged:

* projected gravity ``g = R^T (0, 0, -1)``, the unit down direction in pelvis
  coordinates (tilt is kept, world yaw is dropped);
* ``H v``, the root's world linear velocity in the gravity-aligned
  pelvis-heading frame (forward, left, up);
* ``R^T d``, the normalized world target direction in pelvis coordinates.

``R`` maps pelvis coordinates to world coordinates.  It is built from the
free root's own quaternion ``qpos[3:7]``, validated and normalized in
float64, never from ``xmat`` or an orientation sensor: ``_get_obs`` runs at
construction before any forward pass, where both are still zero.  A global
yaw ``Q`` gives ``R' = Q R``, ``v' = Q v`` and ``d' = Q d``, so all three
outputs are invariant; the gyro and accelerometer already report in their
own site frame and are not rotated again.

Pure NumPy; nothing here reads or writes the simulator.  The plant contract
hashes the source tokens of every function named in
:data:`HEADING_FREE_FUNCTIONS` into a heading-free species' policy-interface
digest (``interface_implementations.heading_free_observation``), so an edit
to any of them is a policy-interface revision of every species on the
schema.  Only function bodies are hashed, so the thresholds live in them as
literals rather than as module constants, and the bodies use ``%``
formatting (the token digest refuses f-strings).
"""

from __future__ import annotations

import numpy as np

#: The observation schema these helpers implement (configs/plant_versions.toml note 18).
HEADING_FREE_OBSERVATION_SCHEMA = "bipedal-target-heading-free/v1"

#: Every function of this module, in call order: the plant contract fingerprints all of them.
HEADING_FREE_FUNCTIONS = (
    "root_quaternion",
    "body_to_world",
    "projected_gravity",
    "world_to_heading",
    "heading_free_root_segments",
)


class ObservationStateError(ValueError):
    """The simulator state defines no heading-free observation: the free root's quaternion is invalid."""


def root_quaternion(qpos: np.ndarray) -> np.ndarray:
    """The free root's wxyz quaternion ``qpos[3:7]``, validated and normalized in float64.

    Raises :class:`ObservationStateError` for a quaternion that is not four
    finite numbers or whose norm is below 1e-12, rather than silently
    substituting a world-axis default.  The sign is kept: every consumer
    below is invariant to ``q -> -q``.
    """
    quaternion = np.array(qpos[3:7], dtype=np.float64)
    if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
        raise ObservationStateError(
            "the free root's quaternion qpos[3:7] must be four finite numbers, got %r" % (quaternion,)
        )
    norm = float(np.linalg.norm(quaternion))
    if norm < 1e-12:
        raise ObservationStateError("the free root's quaternion qpos[3:7] has norm %r and defines no rotation" % norm)
    return quaternion / norm


def body_to_world(quaternion: np.ndarray) -> np.ndarray:
    """``R``, pelvis coordinates to world coordinates, from a unit wxyz quaternion.

    Quadratic in the quaternion, so ``q`` and ``-q`` give the same matrix.
    Agrees with ``mujoco.mju_quat2Mat`` to a few ULPs.
    """
    w, x, y, z = (float(component) for component in quaternion)
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
            [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
            [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def projected_gravity(rotation: np.ndarray) -> np.ndarray:
    """``g = R^T (0, 0, -1)``: the unit down direction in pelvis coordinates (minus ``R``'s third row)."""
    return -np.array(rotation[2, :], dtype=np.float64)


def world_to_heading(rotation: np.ndarray) -> np.ndarray:
    """``H``: world coordinates to the gravity-aligned pelvis-heading frame (forward, left, up).

    Forward is the normalized horizontal projection of pelvis X.  When that
    projection is shorter than 1e-8 (pelvis X vertical, a terminal pose),
    the horizontal projection of pelvis Y rotated clockwise 90 degrees
    defines forward instead: it rotates with the scene and is finite there,
    though no yaw chart can be continuous at that singularity.
    """
    forward = np.array([rotation[0, 0], rotation[1, 0]], dtype=np.float64)
    if float(np.linalg.norm(forward)) < 1e-8:
        forward = np.array([rotation[1, 1], -rotation[0, 1]], dtype=np.float64)
    cosine, sine = forward / np.linalg.norm(forward)
    return np.array([[cosine, sine, 0.0], [-sine, cosine, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def heading_free_root_segments(
    qpos: np.ndarray,
    root_linear_velocity: np.ndarray,
    target_direction: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(g, H v, R^T d)`` in float64: the schema's three root segments.

    *qpos* is the simulator's full ``qpos`` (the free root first);
    *root_linear_velocity* is the free joint's world-frame ``qvel[0:3]``;
    *target_direction* is the normalized world-frame direction from the
    pelvis to the target.  The caller casts the assembled observation to
    float32.
    """
    rotation = body_to_world(root_quaternion(qpos))
    gravity = projected_gravity(rotation)
    heading_velocity = world_to_heading(rotation) @ np.asarray(root_linear_velocity, dtype=np.float64)
    target_direction_body = rotation.T @ np.asarray(target_direction, dtype=np.float64)
    return gravity, heading_velocity, target_direction_body
