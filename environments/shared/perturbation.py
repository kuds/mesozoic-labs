"""Deterministic external-push scheduling for the recovery stage (1b).

Species-generic by construction (STAGE1_SPLIT_PLAN §3.3, plan §W1): nothing
in this module names a species.  The push magnitude derives from each
plant's own mass and stance geometry (``derive_push_parameters``), the
schedule derives from an integer seed through a portable integer hash, and
every entry point is either a pure NumPy function or an init-time helper
that touches only ``mujoco`` on the host.

Design contract:

* **Off is byte-identical.**  A ``capture_velocity_multiple`` of ``0.0``
  means no schedule, no derivation, no RNG draws, no ``xfrc`` writes —
  integrating environments must not change any behaviour until the multiple
  is positive.
* **Schedule identity is a hash, not an RNG stream.**  Pairing (the same
  episode seed must produce the same pushes for the policy and for every
  null controller) comes from ``lowbias32`` over ``uint32``, so a schedule
  is a pure function of its seed.  The hash was chosen when a JAX backend
  had to reproduce the same schedules (retired by D-D17); it stays because
  it defines every existing schedule (``SCHEDULE_IMPLEMENTATION`` in
  ``task_fingerprint.py``).
* **Static shapes.**  ``push_schedule`` returns fixed-size arrays
  (``max_pushes`` rows); pushes scheduled at or beyond the horizon simply
  never activate.
* **The clock starts at episode step 0.**  The first draw lands at
  ``interval ± jitter``.  Whether a stage's gate scores pushes inside the
  settle window is a gate decision (plan §6 decision 4), not a scheduler
  one — the engine keeps a single unambiguous clock.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

Array = Any


def lowbias32(x: Array) -> Array:
    """The lowbias32 integer mixer over ``uint32``.

    Chosen over an RNG stream so a schedule is a pure function of its seed
    (the module docstring's schedule-identity contract).
    """
    x = np.asarray(x).astype(np.uint32)
    # uint32 wraparound is the mixer's defined behaviour; NumPy >= 2 warns
    # on scalar overflow even for unsigned types, so silence it explicitly.
    with np.errstate(over="ignore"):
        x = x ^ (x >> np.uint32(16))
        x = x * np.uint32(0x7FEB352D)
        x = x ^ (x >> np.uint32(15))
        x = x * np.uint32(0x846CA68B)
        x = x ^ (x >> np.uint32(16))
    return x


def hash_uniform01(seed: Array, index: Array) -> Array:
    """Deterministic uniform [0, 1) from ``(seed, index)``."""
    seed_u = np.asarray(seed).astype(np.uint32)
    index_u = np.asarray(index).astype(np.uint32)
    mixed = lowbias32(seed_u ^ lowbias32(index_u ^ np.uint32(0x9E3779B9)))
    return mixed.astype(np.float32) / np.float32(4294967296.0)


def max_pushes_for(horizon_steps: int, interval_steps: int, jitter_steps: int) -> int:
    """Static schedule size: enough rows that no in-horizon push is dropped."""
    earliest_spacing = max(interval_steps - jitter_steps, 1)
    return int(horizon_steps // earliest_spacing) + 1


def push_schedule(
    schedule_seed: Array,
    *,
    max_pushes: int,
    interval_steps: int,
    jitter_steps: int,
) -> tuple[Array, Array]:
    """Deterministic push start-steps and unit directions for one episode.

    The k-th push starts at ``round((k+1) * interval + U(-jitter, jitter))``
    with an independent uniform heading on the horizontal circle
    (``uniform_horizontal`` — the only direction mode currently defined).
    Returns ``(start_steps[max_pushes] int32, directions[max_pushes, 2]
    float32)``.  Rows are strictly ordered when ``interval > 2 * jitter``,
    which ``validate_push_config`` enforces at configuration time.
    """
    k = np.arange(max_pushes, dtype=np.uint32)
    jitter_u = hash_uniform01(schedule_seed, k * np.uint32(2))
    theta_u = hash_uniform01(schedule_seed, k * np.uint32(2) + np.uint32(1))
    nominal = (k.astype(np.float32) + np.float32(1.0)) * np.float32(interval_steps)
    jitter = (jitter_u * np.float32(2.0) - np.float32(1.0)) * np.float32(jitter_steps)
    start_steps = np.round(nominal + jitter).astype(np.int32)
    theta = theta_u * np.float32(2.0 * math.pi)
    directions = np.stack([np.cos(theta), np.sin(theta)], axis=-1).astype(np.float32)
    return start_steps, directions


def external_push_force(
    step_index: Array,
    start_steps: Array,
    directions: Array,
    *,
    duration_steps: int,
    force_newtons: float,
) -> Array:
    """World-frame ``(fx, fy, fz)`` push force active at *step_index*.

    Pure and static-shaped: sums the (at most one, given validated spacing)
    active push window.  Zero vector outside every window, so integrators
    can write the result to ``xfrc_applied`` unconditionally each control
    step — the force is self-clearing and never persists past its window.
    """
    step = np.asarray(step_index).astype(np.int32)
    active = (step >= start_steps) & (step < start_steps + np.int32(duration_steps))
    horizontal = np.sum(active.astype(np.float32)[:, None] * directions, axis=0) * np.float32(force_newtons)
    return np.concatenate([horizontal, np.zeros((1,), dtype=np.float32)])


def validate_push_config(
    *,
    capture_velocity_multiple: float,
    interval_s: float,
    jitter_s: float,
    duration_s: float,
    direction: str,
    control_dt: float,
) -> dict[str, int]:
    """Fail-closed validation; returns the schedule constants in steps.

    Raises ``ValueError`` on any configuration the scheduler's invariants do
    not cover, rather than producing a schedule whose windows can overlap
    (two simultaneous pushes would sum to a force the derivation never
    calibrated).
    """
    if capture_velocity_multiple < 0.0:
        raise ValueError("perturbation_capture_velocity_multiple must be >= 0")
    if min(interval_s, duration_s) <= 0.0 or jitter_s < 0.0:
        raise ValueError("perturbation interval/duration must be > 0 and jitter >= 0")
    if direction != "uniform_horizontal":
        raise ValueError(f"unknown perturbation_direction {direction!r}; known: 'uniform_horizontal'")
    interval_steps = int(round(interval_s / control_dt))
    jitter_steps = int(round(jitter_s / control_dt))
    duration_steps = int(round(duration_s / control_dt))
    if duration_steps < 1 or interval_steps < 1:
        raise ValueError("perturbation interval/duration are below one control step")
    if interval_steps - 2 * jitter_steps <= duration_steps:
        raise ValueError(
            "perturbation windows can overlap: require interval - 2*jitter > duration "
            f"(got {interval_steps} - 2*{jitter_steps} <= {duration_steps} steps)"
        )
    return {
        "interval_steps": interval_steps,
        "jitter_steps": jitter_steps,
        "duration_steps": duration_steps,
    }


def derive_push_parameters(
    model: Any,
    *,
    capture_velocity_multiple: float,
    duration_s: float,
    keyframe_id: int = 0,
) -> dict[str, float]:
    """Derive the per-species push force from the plant itself.

    Host-side (mujoco), run once at environment construction.
    The capture-point argument: a stance can absorb a CoM velocity of about
    ``v_cap = r_support * sqrt(g / z_com)`` without stepping (linear
    inverted pendulum), so a push delivering ``multiple * v_cap`` of CoM
    velocity change is sub-critical below 1.0 and super-critical above it —
    the same dimensionless difficulty on every plant, which is what makes
    one config value species-generic.  All intermediates are returned so
    the run's provenance can persist them per species (the same multiple
    yields very different newtons on different plants).

    Masses use the root body's kinematic subtree — never ``mj_getTotalmass``,
    which also counts scene bodies such as the prey (plan §W1).

    ``xfrc_applied`` acts at the selected body's inertial centre.  A massless
    root is a coordinate frame, whose compiler-assigned inertial position
    need not lie on the physical trunk.  In that case only, target the
    heaviest massive body rigidly welded to that root; never an articulated
    limb.  ``root_body_id`` still identifies the mass/CoM subtree, while the
    extra ``push_body_id`` records the actual force target.  Massive roots
    retain their existing force point and manifest without alteration.
    """
    import mujoco

    if capture_velocity_multiple <= 0.0:
        raise ValueError("derive_push_parameters requires a positive capture_velocity_multiple")

    free_joints = [j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
    if len(free_joints) < 1:
        raise ValueError("plant has no free root joint; cannot locate the push target body")
    root_body = int(model.jnt_bodyid[free_joints[0]])
    push_body = root_body
    if model.body_mass[root_body] == 0.0:
        rigid_candidates = np.flatnonzero((model.body_weldid == root_body) & (model.body_mass > 0.0))
        if rigid_candidates.size == 0:
            raise ValueError("massless root has no massive rigidly attached body for the push force")
        push_body = int(rigid_candidates[np.argmax(model.body_mass[rigid_candidates])])

    data = mujoco.MjData(model)
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, keyframe_id)
    mujoco.mj_forward(model, data)

    subtree_mass = float(model.body_subtreemass[root_body])
    com = np.asarray(data.subtree_com[root_body])
    com_height = float(com[2])
    if com_height <= 0.0:
        raise ValueError("home-keyframe CoM height is not above the floor")

    # Support geometry from the plant's own home-keyframe floor contacts:
    # the margin from the CoM's ground projection to the axis-aligned hull
    # of the contact points, taken conservatively as the smallest of the
    # four side margins.
    floor_geoms = {g for g in range(model.ngeom) if model.geom_bodyid[g] == 0}
    contact_points = [
        np.asarray(c.pos[:2]) for c in data.contact if (int(c.geom1) in floor_geoms) != (int(c.geom2) in floor_geoms)
    ]
    if len(contact_points) < 2:
        raise ValueError("home keyframe has fewer than two floor contacts; support polygon is undefined")
    points = np.stack(contact_points)
    margins = np.concatenate([points.max(axis=0) - com[:2], com[:2] - points.min(axis=0)])
    support_radius = float(margins.min())
    if support_radius <= 0.0:
        raise ValueError("home-keyframe CoM projects outside the support hull; plant cannot be push-calibrated")

    gravity = abs(float(model.opt.gravity[2]))
    capture_velocity = support_radius * math.sqrt(gravity / com_height)
    delta_v = capture_velocity_multiple * capture_velocity
    impulse = subtree_mass * delta_v
    force = impulse / duration_s
    params = {
        "root_body_id": float(root_body),
        "subtree_mass_kg": subtree_mass,
        "com_height_m": com_height,
        "support_radius_m": support_radius,
        "capture_velocity_mps": capture_velocity,
        "capture_velocity_multiple": float(capture_velocity_multiple),
        "delta_v_mps": delta_v,
        "impulse_ns": impulse,
        "force_n": force,
        "duration_s": float(duration_s),
    }
    if push_body != root_body:
        params["push_body_id"] = float(push_body)
    return params
