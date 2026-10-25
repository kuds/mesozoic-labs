"""Head posture and hip roll per evaluation episode, on a floor-truth :class:`.recorder.EpisodeTrace`: report-only.

The stance gate reads the feet, the floor forces, the tilt of the soles and
the saturation of the commands.  It does not read where the head is: on T.
rex physics r10 the walking parent of stance3 passed 37 of 40 of the v2 gate
zero-shot with its head tip down from 1.24 to 0.61 m and its neck pitched
about +33 degrees (the CPU checks of 2026-10-25).  Nor does the eval-episode
ledger record the hip roll, the one leg hinge the left-right offsets
(``leg_q_<l>_minus_<r>_<joint>_deg``) leave out.  This module reduces both,
per episode, over the gate's window ``[settle_steps, T)`` (the window every
stance metric uses), into the ledger-only columns of
:data:`HEAD_POSTURE_COLUMNS` and, per foot, :data:`HIP_ROLL_FIELDS`:

* ``head_tip_height_mean_m`` and ``head_tip_height_min_m``: the head tip's
  height (the world ``z`` of the model's head-tip site: ``head_tip``, else
  ``snout_tip``, :data:`HEAD_TIP_SITES`), the window's mean and minimum, in
  metres above ``z = 0`` (the floor plane; a heightfield's local height is
  not subtracted).  Read at each control step's boundary substep, as the
  recorder reads the feet and the root (the state one physics step before
  the control boundary): the head-tip height the env's own
  ``info["head_tip_z"]`` and its ``head_clearance`` term read on flat
  ground.  The recorder captures it only when asked
  (:meth:`.recorder.SubstepContactRecorder.record_sites`).
* ``neck_pitch_mean_deg``: the window mean of the neck's pitch, the sum over
  the model's neck pitch hinges (:data:`NECK_PITCH_JOINT`: ``neck_pitch``
  on the T. rex, ``neck_1_pitch`` to ``neck_4_pitch`` on the
  brachiosaurus) of each hinge's angle from the home keyframe, signed about
  the animal's leftward axis, so that + pitches the head down (nose down)
  on every model.  ``head_pitch_mean_deg``: the same for the ``head_pitch``
  hinge (the head on the neck).  Both from ``EpisodeTrace.joint_qpos``, the
  control-boundary state.  On every model today each hinge's home angle is
  0 and its axis points left, so these are the joints' own angles.
* ``<foot>_hip_roll_mean_deg``: per foot, the window mean of its leg's hip
  roll hinge (``<prefix>_hip_roll``, the prefix of the foot's reference
  site, as :func:`~environments.shared.eval_diagnostics.sagittal_leg_dc_pairs`
  spells a side's joints: ``r_hip_roll``, ``right_hip_roll``,
  ``fl_hip_roll``) from its home angle, signed so that + is abduction (the
  foot moves away from the body's midline) on either side: the mirror
  convention, so a mirror-symmetric posture reads the same on both legs.
  ``<foot>_hip_roll_max_abs_deg``: the window maximum of the hinge's
  absolute angle from home (unsigned).

The sign of each hinge is fixed once, at the home keyframe on a scratch
``MjData`` (``env.data`` is never touched): a pitch hinge's axis along the
leftward axis (the root's ``y`` axis, made horizontal), and a hip roll
hinge's ``axis x (foot site - anchor)`` along the outward direction of its
side.  A hinge more than about 45 degrees off the axis it should turn about
(:data:`AXIS_MIN_ALIGNMENT`) is not read.

What a model does not have reads NaN, with the reason kept in
:attr:`PostureLayout.reasons`: no head-tip site (the velociraptor), no neck
pitch hinge (the compsognathus robot, the velociraptor), no ``head_pitch``
hinge (the anatomical compsognathus, whose head turns on the neck alone),
no hip roll hinge on a leg.  The columns themselves are the same on every
model with the same feet.  An empty window (``T <= settle_steps``) and a
trace without the head-tip capture read NaN too.

Nothing here is a gate criterion, a reward, an observation, a
``StanceEpisodeMetrics`` field or a ``gate_progress.npz`` key, and no digest
reads it.  Module level is pure numpy; :meth:`PostureLayout.from_env`
imports mujoco.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .morphology import Morphology
    from .recorder import EpisodeTrace

NAN = float("nan")

#: The head-tip site, the first of these names the model has.
HEAD_TIP_SITES: tuple[str, ...] = ("head_tip", "snout_tip")
#: The neck's pitch hinges: ``neck_pitch``, or the numbered ``neck_<n>_pitch`` of a long neck.
NECK_PITCH_JOINT = re.compile(r"^neck(?:_\d+)?_pitch$")
#: The head's pitch hinge on the neck.
HEAD_PITCH_JOINT = "head_pitch"
#: A leg's hip roll hinge: the foot's reference-site prefix, then this.
HIP_ROLL_SUFFIX = "_hip_roll"
#: A hinge is read only when its axis lies within about 45 degrees of the direction it should turn about: the
#: leftward axis for a pitch hinge, ``|axis . left| >= 0.7`` (``eval_diagnostics.SAGITTAL_AXIS_MIN_LATERAL``'s
#: rule); for a hip roll hinge, the foot's swing ``axis x (site - anchor)`` along the outward direction, at least
#: this fraction of the swing's length.
AXIS_MIN_ALIGNMENT = 0.7

#: The whole-animal columns, in ledger order.
HEAD_POSTURE_COLUMNS: tuple[str, ...] = (
    "head_tip_height_mean_m",
    "head_tip_height_min_m",
    "neck_pitch_mean_deg",
    "head_pitch_mean_deg",
)
#: The per-foot fields, as suffixes of ``<label>_`` (``r_hip_roll_mean_deg``), in ledger order.
HIP_ROLL_FIELDS: tuple[str, ...] = ("hip_roll_mean_deg", "hip_roll_max_abs_deg")


@dataclass(frozen=True)
class PostureJoint:
    """One hinge the posture reads: its name, its ``EpisodeTrace.joint_qpos`` column, its sign and home angle."""

    name: str
    column: int
    #: +1 or -1: ``sign * (q - home)`` is + nose down (a pitch hinge) or + abduction (a hip roll hinge).
    sign: float
    #: The hinge's angle at the env's reset keyframe (``qpos0`` without one), rad.
    home: float


@dataclass(frozen=True, eq=False)
class PostureLayout:
    """What one model's posture telemetry reads: its head-tip site, neck and head hinges and hip rolls (read once)."""

    #: The feet, in the morphology's order (one pair of hip roll columns each).
    labels: tuple[str, ...]
    #: The head-tip site's id and name; -1 and ``None`` when the model has none.
    head_site: int
    head_site_name: str | None
    neck_joints: tuple[PostureJoint, ...]
    head_joint: PostureJoint | None
    #: Per foot (as :attr:`labels`), its hip roll hinge, ``None`` when its leg has none that can be read.
    hip_roll: tuple[PostureJoint | None, ...]
    #: Why a column reads NaN on every episode of this model: ``{column: reason}``.
    reasons: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_env(cls, env: Any, morphology: "Morphology") -> "PostureLayout":
        """Describe *env*'s model (wrappers looked through) for *morphology*, at its home keyframe.

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
        root_left = np.array(scratch.xmat[morphology.root_body].reshape(3, 3)[:, 1], dtype=np.float64)
        root_left[2] = 0.0
        norm = float(np.linalg.norm(root_left))
        left = root_left / norm if norm > 0.0 else np.array([0.0, 1.0, 0.0])
        columns = {
            int(address): column for column, address in enumerate(joint_qpos_addresses(model, morphology.root_body))
        }
        hinge = int(mujoco.mjtJoint.mjJNT_HINGE)
        reasons: dict[str, str] = {}

        def joint_named(name: str) -> int:
            joint = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name))
            if joint < 0 or int(model.jnt_type[joint]) != hinge or int(model.jnt_qposadr[joint]) not in columns:
                return -1
            return joint

        def read(joint: int, sign: float) -> PostureJoint:
            address = int(model.jnt_qposadr[joint])
            return PostureJoint(str(model.joint(joint).name), columns[address], sign, float(scratch.qpos[address]))

        def pitch(joint: int) -> PostureJoint | None:
            along = float(np.dot(scratch.xaxis[joint], left))
            return read(joint, math.copysign(1.0, along)) if abs(along) >= AXIS_MIN_ALIGNMENT else None

        head_site, head_name = -1, None
        for name in HEAD_TIP_SITES:
            site = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name))
            if site >= 0:
                head_site, head_name = site, name
                break
        if head_site < 0:
            reason = f"the model has no head-tip site ({' or '.join(HEAD_TIP_SITES)})"
            reasons.update(dict.fromkeys(HEAD_POSTURE_COLUMNS[:2], reason))

        names = [str(model.joint(joint).name) for joint in range(model.njnt)]
        neck_ids = [joint_named(name) for name in names if NECK_PITCH_JOINT.match(name)]
        neck = tuple(found for joint in neck_ids if joint >= 0 and (found := pitch(joint)) is not None)
        if not neck:
            reasons["neck_pitch_mean_deg"] = (
                "the model has no neck pitch hinge (neck_pitch or neck_<n>_pitch) about the leftward axis"
            )
        head_id = joint_named(HEAD_PITCH_JOINT)
        head = pitch(head_id) if head_id >= 0 else None
        if head is None:
            reasons["head_pitch_mean_deg"] = f"the model has no {HEAD_PITCH_JOINT} hinge about the leftward axis"

        hip_roll: list[PostureJoint | None] = []
        for foot in morphology.feet:
            prefix = str(model.site(foot.primary_site).name).split("_", 1)[0]
            joint = joint_named(f"{prefix}{HIP_ROLL_SUFFIX}")
            found = None
            if joint >= 0:
                outward = left if foot.side != "R" else -left  # a label with no side letter reads as a left foot
                swing = np.cross(scratch.xaxis[joint], scratch.site_xpos[foot.primary_site] - scratch.xanchor[joint])
                length = float(np.linalg.norm(swing))
                along = float(np.dot(swing, outward))
                if length > 0.0 and abs(along) >= AXIS_MIN_ALIGNMENT * length:
                    found = read(joint, math.copysign(1.0, along))
            if found is None:
                reason = f"the {foot.label} leg has no {prefix}{HIP_ROLL_SUFFIX} hinge that swings the foot sideways"
                reasons.update({f"{foot.label}_{name}": reason for name in HIP_ROLL_FIELDS})
            hip_roll.append(found)
        return cls(
            labels=tuple(foot.label for foot in morphology.feet),
            head_site=head_site,
            head_site_name=head_name,
            neck_joints=neck,
            head_joint=head,
            hip_roll=tuple(hip_roll),
            reasons=reasons,
        )

    def columns(self) -> tuple[str, ...]:
        """The ledger columns: :data:`HEAD_POSTURE_COLUMNS`, then per foot :data:`HIP_ROLL_FIELDS`."""
        return HEAD_POSTURE_COLUMNS + tuple(f"{label}_{name}" for label in self.labels for name in HIP_ROLL_FIELDS)

    def sites(self) -> tuple[int, ...]:
        """The sites the recorder must capture for these columns (the head tip's), empty when there is none."""
        return (self.head_site,) if self.head_site >= 0 else ()


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


def head_tip_heights(trace: "EpisodeTrace", layout: PostureLayout) -> np.ndarray | None:
    """(T,) the head tip's height at each boundary sample, m; ``None`` when the trace did not capture the site."""
    names = tuple(getattr(trace, "site_names", ()) or ())
    positions = getattr(trace, "site_pos", None)
    if layout.head_site_name is None or positions is None or layout.head_site_name not in names:
        return None
    heights: np.ndarray = np.asarray(positions, dtype=np.float64)[:, names.index(layout.head_site_name), 2]
    return heights


def episode_posture_telemetry(trace: "EpisodeTrace", layout: PostureLayout, *, settle_steps: int) -> dict[str, float]:
    """The posture columns of one episode, keyed as :meth:`PostureLayout.columns` (NaN where unmeasured).

    Every column is NaN for an empty window (``T <= settle_steps``); the
    head-tip columns also for a trace recorded without the head-tip capture.
    """
    settle = _check_settle(settle_steps)
    out = dict.fromkeys(layout.columns(), NAN)
    steps = trace.length
    if steps <= settle:
        return out
    window = slice(settle, steps)
    heights = head_tip_heights(trace, layout)
    if heights is not None:
        out["head_tip_height_mean_m"] = _mean(heights[window])
        out["head_tip_height_min_m"] = _min(heights[window])
    joints = np.asarray(trace.joint_qpos, dtype=np.float64)

    def offset(joint: PostureJoint) -> np.ndarray:
        signed: np.ndarray = joint.sign * (joints[window, joint.column] - joint.home)
        return signed

    if layout.neck_joints:
        out["neck_pitch_mean_deg"] = math.degrees(
            _mean(np.sum([offset(joint) for joint in layout.neck_joints], axis=0))
        )
    if layout.head_joint is not None:
        out["head_pitch_mean_deg"] = math.degrees(_mean(offset(layout.head_joint)))
    for label, joint in zip(layout.labels, layout.hip_roll):
        if joint is None:
            continue
        roll = offset(joint)
        out[f"{label}_hip_roll_mean_deg"] = math.degrees(_mean(roll))
        out[f"{label}_hip_roll_max_abs_deg"] = math.degrees(_max(np.abs(roll)))
    return out
