"""Whole-body balance and floor-force telemetry on a floor-truth :class:`.recorder.EpisodeTrace`: report-only.

The pad and whole-foot centres of pressure (:mod:`.pad_telemetry`) say where
each foot carries its load.  They cannot say how close the animal is to
falling over its toes, nor how it holds a lean: the T. rex stance runs that
nosedived reached the toe line with every foot rail still inside its bar.
This module reduces the balance fields a recorder built with
``balance=True`` adds to the trace (the centre of mass, the applied
actuator and ankle torques, the floor forces) into one number per key and
episode, over the gate's window ``[settle_steps, T)``:

* where the centre of mass stands and how the root pitches:
  ``com_fore_aft_mean_m`` (the CoM ahead of the mean foot reference site
  along the heading, m), ``root_pitch_mean_deg`` (the root's forward axis
  above horizontal, + nose up) and ``xcom_fore_aft_max_m`` (the extrapolated
  CoM, ``xi = c + v / omega0`` with ``omega0 = sqrt(g / h)``, Hof 2008);
* Hof's split of the CoM's forward acceleration (Hof 2007), from the floor
  forces: ``a = f . F / m`` is ``cop_term = F_z f . (c - CoP) / (m h)``, the
  part the centre of pressure explains, plus ``counter_rotation``, the rest
  (``hof_cop_term_mean_mps2``, ``hof_counter_rotation_mean_mps2`` and
  ``hof_counter_rotation_rms_mps2``);
* the achieved left/right joint offsets of each sagittal leg pair from the
  home pose, in degrees about the leftward axis, as the commanded leg DC is
  signed (``leg_q_<l>_minus_<r>_<joint>_deg`` and the mirror-even
  ``leg_q_<l>_<r>_mean_<joint>_deg``);
* on feet that declare a support span (the T. rex), the XcoM's margins to
  the declared toe and heel lines (``xcom_toe_margin_min_m``,
  ``xcom_toe_margin_mean_m``, ``xcom_heel_margin_min_m``, + inside), the
  window fraction past the toe line (``xcom_past_toe_fraction``), the
  shortest time for the CoM to reach the toe line at its current speed
  (``time_to_toe_min_s``) and, on an episode that terminated, how long
  before its end the XcoM first passed the toe line (``xcom_toe_lead_s``,
  :data:`FALL_ONLY_KEYS`); on hinged digits (the T. rex), the XcoM's margin
  to the digit tips (``xcom_tip_margin_min_m``);

and the floor and actuator forces (:func:`episode_force_telemetry`): how
close the actuators run to their force limits, the shear the floor carries
per foot over its normal load, how much of the friction cone each foot
uses, and, on a declared ankle, the ankle torque over the foot's normal
load: the whole-foot CoP's lever ahead of the ankle, read from the joint.

The heading frame of every sample is computed at once
(:func:`heading_frames`, the ``heading_observation`` helpers' rule
vectorised).  One rule says which fields exist, the pad telemetry's: a field
is kept only where the model has what it measures (:meth:`BalanceLayout.episode_keys`),
and NaN in a field that exists means "not measured this time".  Nothing
here is a gate criterion, a reward or an observation, and no digest reads it.
Module level is pure numpy; :meth:`BalanceLayout.from_env` imports mujoco.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import combinations
from typing import TYPE_CHECKING, Any

import numpy as np

from .constants import CONTACT_THRESHOLD_N

if TYPE_CHECKING:
    from .morphology import Morphology
    from .recorder import EpisodeTrace

NAN = float("nan")

#: The balance keys every box-soled model records.
GENERIC_BALANCE_KEYS: tuple[str, ...] = (
    "com_fore_aft_mean_m",
    "root_pitch_mean_deg",
    "xcom_fore_aft_max_m",
    "hof_cop_term_mean_mps2",
    "hof_counter_rotation_mean_mps2",
    "hof_counter_rotation_rms_mps2",
)
#: The keys of a model whose feet declare a support span (the T. rex).
SPAN_BALANCE_KEYS: tuple[str, ...] = (
    "xcom_toe_margin_min_m",
    "xcom_toe_margin_mean_m",
    "xcom_heel_margin_min_m",
    "xcom_past_toe_fraction",
    "time_to_toe_min_s",
    "xcom_toe_lead_s",
)
#: The key of a model with hinged digits (the T. rex).
TIP_BALANCE_KEYS: tuple[str, ...] = ("xcom_tip_margin_min_m",)
#: Per-episode keys that are defined only on an episode that fell: kept out of the full-horizon means.
FALL_ONLY_KEYS: tuple[str, ...] = ("xcom_toe_lead_s",)
#: Every scalar balance key a ``gate_progress.npz`` row may hold (the leg-offset keys aside), with the
#: ``fall_`` mean of each :data:`FALL_ONLY_KEYS` key in place of the key: ``progress.json``'s ``balance`` block.
BALANCE_SCALAR_KEYS: tuple[str, ...] = (
    *GENERIC_BALANCE_KEYS,
    *(key for key in SPAN_BALANCE_KEYS if key not in FALL_ONLY_KEYS),
    *(f"fall_{key}" for key in FALL_ONLY_KEYS),
    *TIP_BALANCE_KEYS,
)
#: The whole-animal force keys.
FORCE_KEYS: tuple[str, ...] = ("actuator_force_frac_max", "actuator_force_limited_fraction", "shear_over_normal")
#: The per-foot force fields, as suffixes of ``<label>_`` (the moment arm on a declared ankle only).
FOOT_FORCE_FIELDS: tuple[str, ...] = ("shear_over_normal", "cone_use_mean", "cone_use_max")
ANKLE_FORCE_FIELD = "ankle_moment_arm_m"
#: The prefix of the achieved leg-offset keys (``leg_q_<l>_minus_<r>_<joint>_deg``, ``leg_q_<l>_<r>_mean_<joint>_deg``).
LEG_Q_PREFIX = "leg_q_"
#: An actuator at this fraction of its force limit or more counts as at the limit.
FORCE_LIMIT_FRACTION = 0.99
#: A speed or a distance at or below this is not moving toward the toe line (``time_to_toe_min_s``).
_TIME_TO_TOE_FLOOR = 1e-6


def _leg_dc_parts(key: str) -> tuple[str, str, str]:
    """``leg_dc_<l>_minus_<r>_<joint>_deg`` -> ``(l, r, joint)``; foot labels hold no underscore."""
    stem = key.removeprefix("leg_dc_").removesuffix("_deg")
    left, rest = stem.split("_minus_", 1)
    right, joint = rest.split("_", 1)
    return left, right, joint


def mirror_even_key(pair: Any) -> str:
    """The mirror-even counterpart of a leg DC key: ``leg_dc_l_minus_r_ankle_deg`` -> ``leg_dc_l_r_mean_ankle_deg``.

    *pair* is a ``SagittalLegPair`` (its ``key``) or the key itself.
    """
    left, right, joint = _leg_dc_parts(str(getattr(pair, "key", pair)))
    return f"leg_dc_{left}_{right}_mean_{joint}_deg"


def actuator_force_limits(model: Any) -> np.ndarray:
    """(nu, 2) each actuator's ``forcerange`` where it is force-limited with a non-degenerate range, else NaN."""
    limits = np.full((int(model.nu), 2), np.nan)
    ranges = np.asarray(model.actuator_forcerange, dtype=np.float64).reshape(-1, 2)
    limited = np.asarray(model.actuator_forcelimited).reshape(-1).astype(bool) & (ranges[:, 0] < ranges[:, 1])
    limits[limited] = ranges[limited]
    return limits


def actuator_force_fraction(force: np.ndarray, limits: np.ndarray) -> np.ndarray:
    """``|F| / limit`` per sample and actuator: the upper limit for ``F >= 0``, the negated lower one otherwise.

    NaN for an actuator without a limit, and where the limit on that side is
    not positive.
    """
    force = np.asarray(force, dtype=np.float64)
    limit = np.where(force >= 0.0, limits[:, 1], -limits[:, 0])
    usable = np.isfinite(limit) & (limit > 0.0)
    fraction: np.ndarray = np.where(usable, np.abs(force) / np.where(usable, limit, 1.0), np.nan)
    return fraction


def actuator_limited_fraction(force: np.ndarray, limits: np.ndarray) -> float:
    """The fraction of samples (rows of the (T, nu) *force*) on which some force-limited actuator is at its limit.

    "At its limit" is :data:`FORCE_LIMIT_FRACTION` of it or more
    (:func:`actuator_force_fraction`); the limited actuators are those with a
    finite (nu, 2) *limits* row.  NaN without a limited actuator or a sample.
    The evaluation key ``actuator_force_limited_fraction`` and the training
    ledger's column of that name are this one definition.
    """
    force = np.asarray(force, dtype=np.float64).reshape(-1, limits.shape[0])
    limited = np.isfinite(limits).all(axis=1)
    if not limited.any() or not force.shape[0]:
        return NAN
    fraction = actuator_force_fraction(force, limits)
    at_limit = np.nan_to_num(fraction[:, limited], nan=0.0) >= FORCE_LIMIT_FRACTION
    return float(np.mean(at_limit.any(axis=1)))


@dataclass(frozen=True)
class LegJointPair:
    """The achieved-angle counterpart of ``eval_diagnostics.SagittalLegPair``: joint qpos columns, not actuators."""

    #: ``l_minus_r_ankle``: the keys are ``leg_q_<stem>_deg`` and the mirror-even ``leg_q_<l>_<r>_mean_<joint>_deg``.
    stem: str
    #: Columns of ``EpisodeTrace.joint_qpos``.
    left_column: int
    right_column: int
    #: The ``SagittalLegPair`` signs: each joint axis's sign along the leftward axis at the home pose.
    left_sign: float
    right_sign: float
    #: Each joint's qpos at the env's reset keyframe (``qpos0`` without one), rad.
    left_home: float
    right_home: float

    @property
    def difference_key(self) -> str:
        return f"{LEG_Q_PREFIX}{self.stem}_deg"

    @property
    def mean_key(self) -> str:
        left, rest = self.stem.split("_minus_", 1)
        right, joint = rest.split("_", 1)
        return f"{LEG_Q_PREFIX}{left}_{right}_mean_{joint}_deg"


@dataclass(frozen=True, eq=False)
class BalanceLayout:
    """What one model's balance telemetry measures: its feet, mass, declarations and home pose (read once)."""

    labels: tuple[str, ...]
    #: ``Morphology.body_weight_n / g``.
    mass_kg: float
    #: ``|model.opt.gravity[2]|``.
    gravity: float
    #: Per foot: whether its registered sole is a box (the pad feet: the recorder runs on such a model).
    box_soled: tuple[bool, ...]
    #: Per foot: whether it declares an ankle and a support span (``FOOT_SUPPORT``).
    spanned: tuple[bool, ...]
    #: The hinged digits' count (``Morphology.support_geom_mtp``).
    hinged_digits: int
    #: Per foot: the sign of its declared ankle axis along the leftward axis at home; NaN without an ankle.
    ankle_left_sign: tuple[float, ...]
    leg_pairs: tuple[LegJointPair, ...]
    #: (nu, 2) :func:`actuator_force_limits`.
    force_limits: np.ndarray = field(repr=False)
    #: The horizontal reference-site distance of every foot pair at home (``stance_metrics``'s pair order).
    home_pair_distances: np.ndarray = field(repr=False)

    @classmethod
    def from_env(cls, env: Any, morphology: "Morphology", pairs: Any) -> "BalanceLayout":
        """Describe *env*'s model (wrappers looked through) for *morphology* and its ``SagittalLegPair`` *pairs*.

        The home pose is the env's reset keyframe (``_reset_keyframe_id``,
        ``qpos0`` without a keyframe), as ``Morphology.from_env`` takes it,
        on a scratch ``MjData``: ``env.data`` is never touched.
        """
        import mujoco

        from .recorder import joint_qpos_addresses

        unwrapped = getattr(env, "unwrapped", env)
        model = unwrapped.model
        scratch = mujoco.MjData(model)
        if model.nkey:
            keyframe = int(getattr(unwrapped, "_reset_keyframe_id", 0))
            if not 0 <= keyframe < model.nkey:
                raise ValueError(f"_reset_keyframe_id {keyframe} is not one of the model's {model.nkey} keyframes")
            mujoco.mj_resetDataKeyframe(model, scratch, keyframe)
        mujoco.mj_kinematics(model, scratch)
        box = int(mujoco.mjtGeom.mjGEOM_BOX)
        feet = morphology.feet
        root_left = np.array(scratch.xmat[morphology.root_body].reshape(3, 3)[:, 1], dtype=np.float64)
        root_left[2] = 0.0
        norm = float(np.linalg.norm(root_left))
        left = root_left / norm if norm > 0.0 else np.array([0.0, 1.0, 0.0])
        ankle_signs = tuple(
            math.copysign(1.0, float(np.dot(scratch.xaxis[foot.ankle_joint], left))) if foot.ankle_joint >= 0 else NAN
            for foot in feet
        )
        columns = {
            int(address): column for column, address in enumerate(joint_qpos_addresses(model, morphology.root_body))
        }
        leg_pairs = []
        for pair in pairs:
            joints = [int(model.actuator_trnid[int(actuator), 0]) for actuator in (pair.left, pair.right)]
            addresses = [int(model.jnt_qposadr[joint]) for joint in joints]
            if any(address not in columns for address in addresses):
                continue
            left_d, right_d, joint = _leg_dc_parts(pair.key)
            leg_pairs.append(
                LegJointPair(
                    stem=f"{left_d}_minus_{right_d}_{joint}",
                    left_column=columns[addresses[0]],
                    right_column=columns[addresses[1]],
                    left_sign=float(pair.left_sign),
                    right_sign=float(pair.right_sign),
                    left_home=float(scratch.qpos[addresses[0]]),
                    right_home=float(scratch.qpos[addresses[1]]),
                )
            )
        sites = np.array([scratch.site_xpos[foot.primary_site] for foot in feet], dtype=np.float64)
        return cls(
            labels=tuple(foot.label for foot in feet),
            mass_kg=float(morphology.body_weight_n) / abs(float(model.opt.gravity[2])),
            gravity=abs(float(model.opt.gravity[2])),
            box_soled=tuple(foot.sole_geom >= 0 and foot.sole_type == box for foot in feet),
            spanned=tuple(
                foot.ankle_joint >= 0 and math.isfinite(foot.heel_line_m) and math.isfinite(foot.toe_line_m)
                for foot in feet
            ),
            hinged_digits=sum(1 for joint in morphology.support_geom_mtp if joint >= 0),
            ankle_left_sign=ankle_signs,
            leg_pairs=tuple(leg_pairs),
            force_limits=actuator_force_limits(model),
            home_pair_distances=pair_distances(sites[None])[0],
        )

    def is_empty(self) -> bool:
        """No foot has a box sole: the model attaches no recorder, so nothing is recorded."""
        return not any(self.box_soled)

    def balance_keys(self) -> tuple[str, ...]:
        """The balance keys this model has (:data:`GENERIC_BALANCE_KEYS`, the leg offsets, span and tip keys)."""
        if self.is_empty():
            return ()
        keys = list(GENERIC_BALANCE_KEYS)
        for pair in self.leg_pairs:
            keys += [pair.difference_key, pair.mean_key]
        if any(self.spanned):
            keys += SPAN_BALANCE_KEYS
        if self.hinged_digits:
            keys += TIP_BALANCE_KEYS
        return tuple(keys)

    def force_keys(self) -> tuple[str, ...]:
        """The force keys this model has: :data:`FORCE_KEYS`, then per foot, foot-major, the moment arm last."""
        if self.is_empty():
            return ()
        keys = list(FORCE_KEYS)
        for label, sign in zip(self.labels, self.ankle_left_sign):
            keys += [f"{label}_{name}" for name in FOOT_FORCE_FIELDS]
            if math.isfinite(sign):
                keys.append(f"{label}_{ANKLE_FORCE_FIELD}")
        return tuple(keys)

    def episode_keys(self) -> tuple[str, ...]:
        """Every per-episode key: :meth:`balance_keys`, then :meth:`force_keys` (empty for an empty layout)."""
        return self.balance_keys() + self.force_keys()


def pair_distances(positions: np.ndarray) -> np.ndarray:
    """(..., F, 3) -> (..., P): the horizontal distance of every foot pair, in ``stance_metrics``'s order."""
    pairs = list(combinations(range(positions.shape[-2]), 2))
    if not pairs:
        return np.zeros((*positions.shape[:-2], 0))
    return np.stack([np.linalg.norm(positions[..., i, :2] - positions[..., j, :2], axis=-1) for i, j in pairs], axis=-1)


def heading_frames(root_quat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(T,4) wxyz root quaternions -> the heading's forward unit vector (T,2) and the root pitch (T,) in degrees.

    The forward vector is ``heading_observation.world_to_heading(body_to_world(q))[0, :2]`` for every sample at
    once: the normalised horizontal projection of the root's x axis, or, where that projection is shorter than
    1e-8 (the x axis vertical), the horizontal projection of its y axis turned clockwise.  The pitch is
    ``asin(R[2, 0])``, the root's forward axis above horizontal (+ nose up).
    """
    quat = np.asarray(root_quat, dtype=np.float64).reshape(-1, 4)
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    r00 = 1.0 - 2.0 * (y * y + z * z)
    r10 = 2.0 * (x * y + w * z)
    r20 = 2.0 * (x * z - w * y)
    r01 = 2.0 * (x * y - w * z)
    r11 = 1.0 - 2.0 * (x * x + z * z)
    forward = np.stack([r00, r10], axis=1)
    short = np.sqrt(r00 * r00 + r10 * r10) < 1e-8
    if short.any():
        forward[short] = np.stack([r11[short], -r01[short]], axis=1)
    forward /= np.linalg.norm(forward, axis=1, keepdims=True)
    pitch = np.degrees(np.arcsin(np.clip(r20, -1.0, 1.0)))
    return forward, pitch


def _check_settle(settle_steps: Any) -> int:
    if isinstance(settle_steps, bool) or int(settle_steps) != settle_steps or settle_steps < 0:
        raise ValueError(f"settle_steps must be a non-negative whole number of steps, got {settle_steps!r}")
    return int(settle_steps)


def _finite(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    return values[np.isfinite(values)]


def _mean(values: np.ndarray) -> float:
    finite = _finite(values)
    return float(finite.mean()) if finite.size else NAN


def _min(values: np.ndarray) -> float:
    finite = _finite(values)
    return float(finite.min()) if finite.size else NAN


def _max(values: np.ndarray) -> float:
    finite = _finite(values)
    return float(finite.max()) if finite.size else NAN


def _has_balance_fields(trace: "EpisodeTrace") -> bool:
    return getattr(trace, "com_pos", None) is not None and getattr(trace, "floor_force", None) is not None


@dataclass(frozen=True)
class _Kinematics:
    """The per-sample quantities both reductions share."""

    forward: np.ndarray  # (T,2)
    pitch: np.ndarray  # (T,)
    com: np.ndarray  # (T,3)
    velocity: np.ndarray  # (T,3)
    height: np.ndarray  # (T,)
    xcom: np.ndarray  # (T,2)


def _kinematics(trace: "EpisodeTrace", layout: BalanceLayout) -> _Kinematics:
    """The heading frames, the CoM, its velocity by differencing the boundary samples, and the XcoM."""
    assert trace.com_pos is not None and trace.spawn_com_pos is not None
    forward, pitch = heading_frames(trace.root_quat)
    com = np.asarray(trace.com_pos, dtype=np.float64)
    dt = float(trace.dt)
    frame_skip = int(trace.frame_skip)
    velocity = np.full_like(com, np.nan)
    if com.shape[0] > 1:
        velocity[1:] = np.diff(com, axis=0) / dt
    if com.shape[0] and frame_skip > 1:
        # The spawn sample is the episode's first substep and the first boundary sample its last: frame_skip - 1
        # substeps apart, not one control step.
        velocity[0] = (com[0] - np.asarray(trace.spawn_com_pos, dtype=np.float64)) / (
            (frame_skip - 1) * dt / frame_skip
        )
    height = com[:, 2]
    with np.errstate(invalid="ignore", divide="ignore"):
        omega0 = np.where(height > 0.0, np.sqrt(layout.gravity / np.where(height > 0.0, height, 1.0)), np.nan)
    xcom = com[:, :2] + velocity[:, :2] / omega0[:, None]
    return _Kinematics(forward, pitch, com, velocity, height, xcom)


def _along(forward: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """Per sample, ``forward . vector`` over the horizontal components (vectors (T,2+) or (T,K,2+))."""
    if vectors.ndim == 2:
        result: np.ndarray = np.einsum("tj,tj->t", forward, vectors[:, :2])
        return result
    along: np.ndarray = np.einsum("tj,tkj->tk", forward, vectors[:, :, :2])
    return along


def _hof_split(trace: "EpisodeTrace", layout: BalanceLayout, kin: _Kinematics) -> tuple[np.ndarray, ...]:
    """Per sample ``(a, cop_term, counter_rotation)`` of the CoM's forward acceleration (m/s^2): Hof's split."""
    assert trace.floor_force is not None and trace.floor_cop is not None
    # (T,3): the floor is the only horizontal external force on a stage without pushes; a recovery stage's push
    # (xfrc_applied on the root) is not in it, so there the split is the floor's share only (the docs say so).
    total = np.asarray(trace.floor_force, dtype=np.float64).sum(axis=1)
    mass = layout.mass_kg
    acceleration = _along(kin.forward, total) / mass
    with np.errstate(invalid="ignore", divide="ignore"):
        lever = _along(kin.forward, kin.com - np.asarray(trace.floor_cop, dtype=np.float64))
        cop_term = total[:, 2] * lever / (mass * kin.height)
    return acceleration, cop_term, acceleration - cop_term


def hof_split(trace: "EpisodeTrace", layout: BalanceLayout) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per boundary sample, the CoM's forward acceleration and Hof's split of it (Hof 2007), in m/s^2.

    ``a = f . F / m`` (``F`` the net floor force on the animal, ``f`` the
    heading's forward vector), ``cop_term = F_z f . (c - CoP) / (m h)`` (the
    part the floor centre of pressure's offset from the CoM explains) and
    ``counter_rotation = a - cop_term`` (the rest: the rate of change of the
    angular momentum about the CoM over ``m h``).  NaN where the floor
    forces are unmeasured.  Raises ``ValueError`` for a trace without
    balance fields.

    ``F`` is the floor's force alone: while a recovery stage's scheduled
    push acts (``xfrc_applied`` on the root), ``a`` is the floor's share of
    the CoM's acceleration, not all of it.  Each sample is the control
    step's last substep, so where contacts chatter between substeps (the
    ``frame_skip``-10 Compsognathus) one sample can stand far from the
    step's mean.
    """
    if not _has_balance_fields(trace):
        raise ValueError("the trace has no balance fields: record it with balance=True")
    acceleration, cop_term, counter_rotation = _hof_split(trace, layout, _kinematics(trace, layout))
    return acceleration, cop_term, counter_rotation


def episode_balance_telemetry(trace: "EpisodeTrace", layout: BalanceLayout, *, settle_steps: int) -> dict[str, float]:
    """The balance fields of one episode, keyed as :meth:`BalanceLayout.balance_keys` (NaN when unmeasured).

    Every field is NaN for an empty window (``T <= settle_steps``) and for a
    trace without balance fields (a recorder built without ``balance``, or
    whose balance capture failed).
    """
    settle = _check_settle(settle_steps)
    out = {key: NAN for key in layout.balance_keys()}
    steps = trace.length
    if not out or steps <= settle or not _has_balance_fields(trace):
        return out
    window = slice(settle, steps)
    kin = _kinematics(trace, layout)
    forward = kin.forward
    feet_point = np.asarray(trace.foot_pos, dtype=np.float64).mean(axis=1)  # (T,3)
    com_ahead = _along(forward, kin.com - feet_point)
    xcom_ahead = _along(forward, kin.xcom - feet_point[:, :2])
    out["com_fore_aft_mean_m"] = _mean(com_ahead[window])
    out["root_pitch_mean_deg"] = _mean(kin.pitch[window])
    out["xcom_fore_aft_max_m"] = _max(xcom_ahead[window])

    _acceleration, cop_term, counter_rotation = _hof_split(trace, layout, kin)
    out["hof_cop_term_mean_mps2"] = _mean(cop_term[window])
    out["hof_counter_rotation_mean_mps2"] = _mean(counter_rotation[window])
    rotation = _finite(counter_rotation[window])
    out["hof_counter_rotation_rms_mps2"] = float(np.sqrt(np.mean(rotation * rotation))) if rotation.size else NAN

    # The achieved joint offsets from home, signed about the leftward axis as the commanded leg DC is.
    joints = np.asarray(trace.joint_qpos, dtype=np.float64)
    for pair in layout.leg_pairs:
        left = pair.left_sign * (joints[window, pair.left_column] - pair.left_home)
        right = pair.right_sign * (joints[window, pair.right_column] - pair.right_home)
        out[pair.difference_key] = math.degrees(_mean(left - right))
        out[pair.mean_key] = math.degrees(_mean((left + right) / 2.0))

    xcom_along = _along(forward, kin.xcom)
    if any(layout.spanned) and trace.support_line_pos is not None:
        spanned = np.flatnonzero(np.asarray(layout.spanned, dtype=bool))
        lines = np.asarray(trace.support_line_pos, dtype=np.float64)[:, spanned]  # (T,S,2,3)
        heel_line = _along(forward, lines[:, :, 0]).mean(axis=1)
        toe_line = _along(forward, lines[:, :, 1]).mean(axis=1)
        toe_margin = toe_line - xcom_along
        heel_margin = xcom_along - heel_line
        out["xcom_toe_margin_min_m"] = _min(toe_margin[window])
        out["xcom_toe_margin_mean_m"] = _mean(toe_margin[window])
        out["xcom_heel_margin_min_m"] = _min(heel_margin[window])
        measured = toe_margin[window][np.isfinite(toe_margin[window])]
        out["xcom_past_toe_fraction"] = float(np.mean(measured < 0.0)) if measured.size else NAN
        distance = toe_line - _along(forward, kin.com)
        speed = _along(forward, kin.velocity)
        moving = (distance > _TIME_TO_TOE_FLOOR) & (speed > _TIME_TO_TOE_FLOOR)
        with np.errstate(invalid="ignore", divide="ignore"):
            time_to_toe = np.where(moving, distance / np.where(moving, speed, 1.0), np.inf)
        out["time_to_toe_min_s"] = _min(time_to_toe[window])
        if trace.terminated and not trace.truncated:
            past = np.flatnonzero(toe_margin[window] < 0.0)
            if past.size:
                out["xcom_toe_lead_s"] = (steps - (settle + int(past[0]))) * float(trace.dt)
    if layout.hinged_digits and trace.digit_tip_pos is not None and trace.digit_tip_foot is not None:
        tips = _along(forward, np.asarray(trace.digit_tip_pos, dtype=np.float64))  # (T,D)
        owners = np.asarray(trace.digit_tip_foot, dtype=np.int64)
        per_foot = [tips[:, owners == foot].max(axis=1) for foot in np.unique(owners).tolist()]
        if per_foot:
            tip_line = np.mean(per_foot, axis=0)
            out["xcom_tip_margin_min_m"] = _min((tip_line - xcom_along)[window])
    return out


def episode_actuator_force_fraction(trace: "EpisodeTrace", layout: BalanceLayout, *, settle_steps: int) -> np.ndarray:
    """(nu,) the window mean of each actuator's ``|F| / limit`` (:func:`actuator_force_fraction`); NaN unmeasured."""
    settle = _check_settle(settle_steps)
    nu = layout.force_limits.shape[0]
    force = getattr(trace, "actuator_force", None)
    if force is None or trace.length <= settle:
        return np.full(nu, np.nan)
    fraction = actuator_force_fraction(np.asarray(force, dtype=np.float64)[settle:], layout.force_limits)
    finite = np.isfinite(fraction)
    count = finite.sum(axis=0)
    total = np.where(finite, fraction, 0.0).sum(axis=0)
    means: np.ndarray = np.where(count > 0, total / np.maximum(count, 1), np.nan)
    return means


def episode_force_telemetry(trace: "EpisodeTrace", layout: BalanceLayout, *, settle_steps: int) -> dict[str, float]:
    """The force fields of one episode, keyed as :meth:`BalanceLayout.force_keys` (NaN when unmeasured).

    * ``actuator_force_frac_max`` -- the max over the force-limited
      actuators of the window mean of ``|F| / limit``;
      ``actuator_force_limited_fraction`` -- the window fraction of steps on
      which some limited actuator is at :data:`FORCE_LIMIT_FRACTION` of its
      limit or more;
    * per foot, over the window steps whose floor force on the limb is above
      ``CONTACT_THRESHOLD_N``: ``<label>_shear_over_normal``, the summed
      forward floor force over the summed normal force (+ the floor pushes
      the foot forward), and ``<label>_cone_use_mean`` / ``_max``, the
      horizontal floor force over ``mu F_z`` (``mu`` the limb's
      load-weighted sliding friction); ``shear_over_normal`` is the same for
      the whole animal;
    * on a declared ankle, ``<label>_ankle_moment_arm_m``: the summed ankle
      torque about the leftward axis over the summed normal force on the
      limb, + ahead of the ankle.  The floor's moment on the foot about the
      ankle is ``-d F_z`` for a load ``d`` ahead of it, and the ankle torque
      balances it, so this estimates the whole-foot CoP's lever; it also
      carries the floor's shear times the ankle's height and the foot's own
      weight.

    Under a pyramidal cone the floor forces are NaN, and so is every key
    but the actuator ones.
    """
    settle = _check_settle(settle_steps)
    out = {key: NAN for key in layout.force_keys()}
    steps = trace.length
    if not out or steps <= settle or getattr(trace, "actuator_force", None) is None:
        return out
    window = slice(settle, steps)
    assert trace.actuator_force is not None
    out["actuator_force_frac_max"] = _max(episode_actuator_force_fraction(trace, layout, settle_steps=settle))
    out["actuator_force_limited_fraction"] = actuator_limited_fraction(
        np.asarray(trace.actuator_force, dtype=np.float64)[window], layout.force_limits
    )
    if not _has_balance_fields(trace):
        return out
    forward, _pitch = heading_frames(trace.root_quat)
    forces = np.asarray(trace.floor_force, dtype=np.float64)[window]  # (W,F+1,3)
    heading = forward[window]
    mu = np.asarray(trace.floor_mu, dtype=np.float64)[window] if trace.floor_mu is not None else None

    def shear(force: np.ndarray) -> float:
        vertical = force[:, 2]
        loaded = vertical > CONTACT_THRESHOLD_N
        if not loaded.any():
            return NAN
        return float(_along(heading[loaded], force[loaded]).sum() / vertical[loaded].sum())

    out["shear_over_normal"] = shear(forces.sum(axis=1))
    torque = np.asarray(trace.ankle_torque, dtype=np.float64)[window] if trace.ankle_torque is not None else None
    for index, label in enumerate(layout.labels):
        limb = forces[:, index]
        out[f"{label}_shear_over_normal"] = shear(limb)
        vertical = limb[:, 2]
        loaded = vertical > CONTACT_THRESHOLD_N
        if mu is not None and loaded.any():
            with np.errstate(invalid="ignore", divide="ignore"):
                use = np.linalg.norm(limb[loaded, :2], axis=1) / (mu[loaded, index] * vertical[loaded])
            out[f"{label}_cone_use_mean"] = _mean(use)
            out[f"{label}_cone_use_max"] = _max(use)
        sign = layout.ankle_left_sign[index]
        key = f"{label}_{ANKLE_FORCE_FIELD}"
        if key in out and torque is not None and loaded.any():
            moment = sign * torque[loaded, index]
            if np.isfinite(moment).all():
                out[key] = float(moment.sum() / vertical[loaded].sum())
    return out
