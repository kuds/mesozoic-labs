"""Commands, tail and body posture, energy, drift and slip, and the body's dynamics per evaluation episode: report-only.

The eval-episode ledger records each episode's outcome beside its feet, its
balance, its forces and its head posture (:mod:`.posture_telemetry`).  It
did not record what the policy commands each actuator, where the tail and
the pelvis sit, how much work the servos do, how far the animal drifts and
how much its loaded feet slide; nor what MuJoCo already computes about the
body's dynamics (the joint stops' load, the centroidal momentum, the
interaction wrench at the tail's and the neck's base, the head's rotation).
This module reduces them, per episode, over the gate's window
``[settle_steps, T)`` (a per-substep quantity over the ``frame_skip``
substeps of those steps), into ledger-only columns, in this order
(:meth:`BodyLayout.columns`; the T. rex names):

1a. **Commands** (:data:`COMMAND_SUMMARY_COLUMNS`): ``<actuator>_command_mean``
    per actuator in model order, the window mean of the deterministic
    command ``EpisodeTrace.action`` (clipped to [-1, 1], action units, +
    toward the actuator's ``ctrlrange`` upper end); ``actuators_at_rail``,
    the actuators whose share of window steps at ``|command| >=
    SATURATION_ABS`` exceeds :data:`RAIL_STEP_SHARE`, and ``rail_actuators``
    naming them (";"-joined, each suffixed "+" or "-" by the sign of the
    majority of its saturated steps, ties "+"); ``actuators_parked``, the
    others whose ``|mean| >=`` :data:`PARKED_COMMAND_ABS`, and
    ``parked_actuators`` (suffixed by the mean's sign).  Descriptive
    classes (the tail screen's lines), never a verdict.
1b. **Tail and body posture**: ``<tail hinge>_mean_deg`` for each hinge
    named ``tail_<n>_pitch`` / ``tail_<n>_yaw``, the window mean of its
    angle from home, signed so that + raises the tail tip (pitch, mirror
    even) or moves it to the animal's left (yaw, mirror odd);
    ``tail_tip_chord_mean_deg`` / ``_min_deg`` / ``_max_deg``, the angle
    above horizontal of the chord from the tail's base (the first tail
    hinge's anchor, carried by the recorded root pose) to the ``tail_tip``
    site; ``root_height_mean_m`` / ``_min_m`` (the root body's height above
    ``z = 0``); ``root_pitch_min_deg`` / ``_max_deg`` (``asin(R[2, 0])``, +
    nose up: :func:`.balance_telemetry.heading_frames`' pitch, whose mean is
    the balance group's ``root_pitch_mean_deg``; a model without the
    balance group gets the mean here); ``root_roll_mean_deg`` / ``_min_deg``
    / ``_max_deg`` (``asin(R[2, 1])``, + left side up).
2.  **Energy and smoothness**, per actuator group (:data:`ENERGY_GROUPS`:
    ``legs``, the actuators on the chains from the root to each foot's
    reference site; ``tail``; ``neck_head``; ``other`` only on a model with
    actuators outside them), from the substep force ``F`` and velocity
    ``v`` (``actuator_force``, ``actuator_velocity``; ``p = F v`` is the
    mechanical power of any transmission): ``<g>_work_abs_j`` (``sum |p|
    dt``), ``<g>_work_net_j`` (``sum p dt``, + the actuators deliver
    energy), ``<g>_power_peak_w`` (the largest substep sum of ``|p|``),
    ``<g>_torque_rms_nm`` (``sqrt(mean F^2)``), ``<g>_torque_rms_frac`` (the
    same of ``F / limit`` with the side-aware force limit, the force-limited
    actuators only) and ``<g>_command_delta_rms`` (the RMS step-to-step
    command change); ``action_delta_mean``, the episode's term of the
    evaluation's ``action_delta``; ``<foot>_leg_work_abs_j`` per foot.
3.  **Drift and slip**: ``window_displacement_m`` and
    ``window_yaw_change_deg`` copied from the episode's
    ``StanceEpisodeMetrics``; the displacement's ``_fore_aft_m`` (+ forward)
    and ``_lateral_m`` (+ left) parts in the heading frame at the window's
    start; ``window_path_length_m`` (the root's horizontal path);
    ``window_yaw_change_signed_deg`` (+ turning left); per foot,
    ``<foot>_slip_m`` (the distance the loaded foot's contact points slide:
    the floor-force-weighted tangential speed of the limb's floor contacts,
    ``efc_vel``'s tangential rows, integrated over the substeps the limb
    carries more than ``CONTACT_THRESHOLD_N``), ``<foot>_slip_speed_max_mps``
    and ``<foot>_site_travel_m`` (the reference site's net horizontal
    travel; its maximum over the feet is the screen's ``max_foot_slide_m``).

Then the four ``MjData`` extras the sensor assessment surfaced (no MuJoCo
sensor is added: each is read from ``MjData`` by the recorder):

g.  **Joint stops**, for each limited tail hinge and leg joint:
    ``<joint>_stop_share`` (the share of window substeps on which its limit
    constraint pushes) and ``<joint>_stop_torque_mean_nm`` (the window mean
    of that constraint's torque about the joint's own axis, 0 off the stop;
    + toward the upper end of the range, so a joint resting on its LOWER
    stop reads +).  Commanded to the rail is the commands' column; carrying
    load on the mechanical stop is this one.
h.  **Centre of mass, capture point and centroidal angular momentum**: on a
    model whose feet declare a support span (the T. rex), the CoM's and
    the capture point's margins to the toe and heel lines along the heading
    (``com_toe_margin_min_m`` ... ``capture_heel_margin_mean_m``, + inside;
    the capture point ``c + v / sqrt(g / h)`` takes the CoM velocity from
    the centroidal momentum at the boundary substep, where the balance
    group's ``xcom_*`` differences the boundary CoM samples); and the
    angular momentum about the CoM in the heading frame,
    ``angular_momentum_<pitch|roll|yaw>_rms_nms`` and ``_max_abs_nms``.
c, d. **Tail-base and neck-base wrench**: the wrench the tail (the neck
    with the head) exerts on its parent body, about the base joint's anchor,
    from ``cfrc_int``: ``<base>_force_down_mean_n`` (the load it hangs on
    the parent), ``<base>_pitch_moment_mean_nm`` (+ pitches the parent nose
    up), ``<base>_pitch_moment_std_nm``, ``<base>_roll_moment_mean_nm`` (+
    rolls it left side up) and ``<base>_yaw_moment_mean_nm`` (+ turns it
    left), for ``tail_base`` and ``neck_base``.
a.  **Head angular rate**: the RMS of the head body's angular velocity
    (``head_angular_rate_rms_dps``) and of its pitch, roll and yaw
    components in the heading frame, deg/s: head stabilisation.

The boundary-sampled columns read each control step's last substep, as the
recorder's root and site samples do.  What a model lacks reads NaN on every
episode, with the reason in :attr:`BodyLayout.reasons`; the column set is
fixed per model.  An empty window (``T <= settle_steps``) reads NaN (text
columns ``""``), and so does a trace recorded without the capture a column
needs.  Nothing here is a gate criterion, a reward, an observation, a
``StanceEpisodeMetrics`` field or a ``gate_progress.npz`` key, and no digest
reads it.  Module level is pure numpy; :meth:`BodyLayout.from_env` imports
mujoco.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from .balance_telemetry import actuator_force_fraction, actuator_force_limits, heading_frames
from .constants import SATURATION_ABS
from .posture_telemetry import AXIS_MIN_ALIGNMENT, HEAD_TIP_SITES, PostureJoint

if TYPE_CHECKING:
    from .morphology import Morphology
    from .recorder import EpisodeTrace, SubstepContactRecorder
    from .stance_metrics import StanceEpisodeMetrics

NAN = float("nan")

#: A command whose window mean is at least this far from 0 is PARKED (the tail screen's "CMD >= 0.60").
PARKED_COMMAND_ABS = 0.60
#: An actuator is AT RAIL when more than this share of its window steps are saturated (the screen's tail-up flag).
RAIL_STEP_SHARE = 0.10
#: The tail hinges whose angle is read (actuated or passive): ``tail_<n>_pitch`` and ``tail_<n>_yaw``.
TAIL_HINGE = re.compile(r"^tail_\d+_(?P<kind>pitch|yaw)$")
#: The joints of the energy groups other than the legs (the legs are found from the feet).
TAIL_JOINT = re.compile(r"^tail_")
NECK_HEAD_JOINT = re.compile(r"^(?:neck|head|jaw)")
NECK_JOINT = re.compile(r"^neck")
#: The tail-tip site of the chord.
TAIL_TIP_SITE = "tail_tip"
#: The energy groups every recorded model has columns for (NaN where empty); ``other`` joins them only on a model
#: with an actuator outside them.
ENERGY_GROUPS: tuple[str, ...] = ("legs", "tail", "neck_head")
OTHER_GROUP = "other"
#: Per energy group, ``<group>_<field>``.
ENERGY_FIELDS: tuple[str, ...] = (
    "work_abs_j",
    "work_net_j",
    "power_peak_w",
    "torque_rms_nm",
    "torque_rms_frac",
    "command_delta_rms",
)
COMMAND_SUFFIX = "_command_mean"
COMMAND_SUMMARY_COLUMNS: tuple[str, ...] = (
    "actuators_at_rail",
    "rail_actuators",
    "actuators_parked",
    "parked_actuators",
)
#: The text columns (``""`` when unmeasured).
TEXT_COLUMNS: tuple[str, ...] = ("rail_actuators", "parked_actuators")
TAIL_CHORD_COLUMNS: tuple[str, ...] = ("tail_tip_chord_mean_deg", "tail_tip_chord_min_deg", "tail_tip_chord_max_deg")
ROOT_HEIGHT_COLUMNS: tuple[str, ...] = ("root_height_mean_m", "root_height_min_m")
ROOT_PITCH_MEAN = "root_pitch_mean_deg"
ROOT_PITCH_COLUMNS: tuple[str, ...] = ("root_pitch_min_deg", "root_pitch_max_deg")
ROOT_ROLL_COLUMNS: tuple[str, ...] = ("root_roll_mean_deg", "root_roll_min_deg", "root_roll_max_deg")
ACTION_DELTA_COLUMN = "action_delta_mean"
LEG_WORK_FIELD = "leg_work_abs_j"
DRIFT_COLUMNS: tuple[str, ...] = (
    "window_displacement_m",
    "window_displacement_fore_aft_m",
    "window_displacement_lateral_m",
    "window_path_length_m",
    "window_yaw_change_deg",
    "window_yaw_change_signed_deg",
)
#: Per foot, ``<foot>_<field>``.
SLIP_FIELDS: tuple[str, ...] = ("slip_m", "slip_speed_max_mps", "site_travel_m")
#: Per limited tail hinge and leg joint, ``<joint>_<field>``.
STOP_FIELDS: tuple[str, ...] = ("stop_share", "stop_torque_mean_nm")
MARGIN_COLUMNS: tuple[str, ...] = (
    "com_toe_margin_min_m",
    "com_toe_margin_mean_m",
    "com_heel_margin_min_m",
    "com_heel_margin_mean_m",
    "capture_toe_margin_min_m",
    "capture_toe_margin_mean_m",
    "capture_heel_margin_min_m",
    "capture_heel_margin_mean_m",
)
ANGULAR_MOMENTUM_COLUMNS: tuple[str, ...] = tuple(
    f"angular_momentum_{axis}_{stat}" for axis in ("pitch", "roll", "yaw") for stat in ("rms_nms", "max_abs_nms")
)
#: The two bases, then per base ``<base>_<field>``.
WRENCH_BASES: tuple[str, ...] = ("tail_base", "neck_base")
WRENCH_FIELDS: tuple[str, ...] = (
    "force_down_mean_n",
    "pitch_moment_mean_nm",
    "pitch_moment_std_nm",
    "roll_moment_mean_nm",
    "yaw_moment_mean_nm",
)
HEAD_RATE_COLUMNS: tuple[str, ...] = (
    "head_angular_rate_rms_dps",
    "head_pitch_rate_rms_dps",
    "head_roll_rate_rms_dps",
    "head_yaw_rate_rms_dps",
)


@dataclass(frozen=True)
class LimitJoint:
    """A joint whose stop is read: its name, model id and ``EpisodeTrace.joint_qpos`` column."""

    name: str
    joint: int
    column: int


@dataclass(frozen=True, eq=False)
class BodyLayout:
    """What one model's body telemetry reads (read once, at the home keyframe on a scratch ``MjData``)."""

    #: The feet, in the morphology's order.
    labels: tuple[str, ...]
    #: The actuators' names, in the action's order.
    actuator_names: tuple[str, ...]
    #: (nu, 2) :func:`.balance_telemetry.actuator_force_limits`.
    force_limits: np.ndarray = field(repr=False)
    #: ``(group, actuator indices)`` for :data:`ENERGY_GROUPS`, then ``other`` when it has any.
    groups: tuple[tuple[str, tuple[int, ...]], ...] = ()
    #: Per foot, the actuators on its chain.
    foot_actuators: tuple[tuple[int, ...], ...] = ()
    #: ``(name, joint)`` for each tail hinge the model has, in model order; ``None`` where it cannot be signed.
    tail_hinges: tuple[tuple[str, PostureJoint | None], ...] = ()
    #: The tail-tip site's id and name (-1 and ``None``: the chord is not read) and the tail's base in the root
    #: frame.
    tail_tip_site: int = -1
    tail_tip_site_name: str | None = None
    tail_base_local: np.ndarray | None = field(default=None, repr=False)
    #: Whether :data:`ROOT_PITCH_MEAN` is a column here (a model without the balance group).
    root_pitch_mean: bool = False
    #: The model's friction cone is elliptic (slip is read only then).
    elliptic: bool = True
    #: The joints whose stop is read, and their model ids.
    limit_joints: tuple[LimitJoint, ...] = ()
    #: Per foot, whether it declares an ankle and a support span (``FOOT_SUPPORT``): the margins' lines.
    spanned: tuple[bool, ...] = ()
    #: The animal's mass (the root's subtree) and ``|gravity|``.
    mass_kg: float = NAN
    gravity: float = NAN
    #: The joint at each of :data:`WRENCH_BASES` (its body's interaction with its parent, about its anchor);
    #: ``(name, id)``, ``("", -1)`` when the model has none.
    wrench_joints: tuple[tuple[str, int], ...] = ()
    #: The head body (the head-tip site's); ``("", -1)`` without one.
    head_body: tuple[str, int] = ("", -1)
    #: Why a column reads NaN on every episode of this model: ``{column: reason}``.
    reasons: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_env(cls, env: Any, morphology: "Morphology", *, root_pitch_mean: bool = False) -> "BodyLayout":
        """Describe *env*'s model (wrappers looked through) for *morphology*, at its home keyframe.

        *root_pitch_mean*: also a :data:`ROOT_PITCH_MEAN` column (the caller
        passes ``True`` when the ledger has no balance group to hold it).
        ``env.data`` is never touched: the home pose is the env's reset
        keyframe (``_reset_keyframe_id``, ``qpos0`` without a keyframe) on a
        scratch ``MjData``, as ``PostureLayout.from_env`` takes it.
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
        root = int(morphology.root_body)
        root_left = np.array(scratch.xmat[root].reshape(3, 3)[:, 1], dtype=np.float64)
        root_left[2] = 0.0
        norm = float(np.linalg.norm(root_left))
        left = root_left / norm if norm > 0.0 else np.array([0.0, 1.0, 0.0])
        up = np.array([0.0, 0.0, 1.0])
        columns = {int(address): column for column, address in enumerate(joint_qpos_addresses(model, root))}
        hinge = int(mujoco.mjtJoint.mjJNT_HINGE)
        joint_names = [str(model.joint(joint).name) for joint in range(model.njnt)]
        reasons: dict[str, str] = {}

        def depth(body: int) -> int:
            count = 0
            while body > 0:
                body = int(model.body_parentid[body])
                count += 1
            return count

        # Actuators and their groups: the legs from the feet's chains, then the tail, the neck and head, the rest.
        names = []
        for index in range(model.nu):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, index)
            names.append(str(name) if name else f"actuator[{index}]")
        joint_transmissions = (int(mujoco.mjtTrn.mjTRN_JOINT), int(mujoco.mjtTrn.mjTRN_JOINTINPARENT))
        actuator_joint = [
            int(model.actuator_trnid[index, 0]) if int(model.actuator_trntype[index]) in joint_transmissions else -1
            for index in range(model.nu)
        ]
        chains: list[set[int]] = []
        for foot in morphology.feet:
            body, chain = int(model.site_bodyid[foot.primary_site]), set()
            while body > 0 and body != root:
                chain.add(body)
                body = int(model.body_parentid[body])
            chains.append(chain)
        chain_joints = [
            {joint for joint in range(model.njnt) if int(model.jnt_bodyid[joint]) in chain} for chain in chains
        ]
        foot_actuators = tuple(
            tuple(index for index, joint in enumerate(actuator_joint) if joint >= 0 and joint in joints)
            for joints in chain_joints
        )
        legs = sorted({index for indices in foot_actuators for index in indices})
        taken = set(legs)

        def by_name(pattern: re.Pattern[str]) -> list[int]:
            found = [
                index
                for index, joint in enumerate(actuator_joint)
                if index not in taken and joint >= 0 and pattern.match(joint_names[joint])
            ]
            taken.update(found)
            return found

        groups: list[tuple[str, tuple[int, ...]]] = [("legs", tuple(legs))]
        groups.append(("tail", tuple(by_name(TAIL_JOINT))))
        groups.append(("neck_head", tuple(by_name(NECK_HEAD_JOINT))))
        other = tuple(index for index in range(model.nu) if index not in taken)
        if other:
            groups.append((OTHER_GROUP, other))
        force_limits = actuator_force_limits(model)
        for group, indices in groups:
            if not indices:
                reason = f"the model has no {group} actuator"
                reasons.update({f"{group}_{name}": reason for name in ENERGY_FIELDS})
            elif not np.isfinite(force_limits[list(indices)]).all(axis=1).any():
                reasons[f"{group}_torque_rms_frac"] = f"no {group} actuator has a force limit"

        # The tail hinges, each signed by where it swings the tail tip at home.
        tip = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, TAIL_TIP_SITE))
        tail_hinges: list[tuple[str, PostureJoint | None]] = []
        for joint, name in enumerate(joint_names):
            match = TAIL_HINGE.match(name)
            address = int(model.jnt_qposadr[joint])
            if match is None or int(model.jnt_type[joint]) != hinge or address not in columns:
                continue
            found = None
            if tip >= 0:
                swing = np.cross(scratch.xaxis[joint], scratch.site_xpos[tip] - scratch.xanchor[joint])
                along = float(np.dot(swing, up if match.group("kind") == "pitch" else left))
                length = float(np.linalg.norm(swing))
                if length > 0.0 and abs(along) >= AXIS_MIN_ALIGNMENT * length:
                    found = PostureJoint(
                        name, columns[address], math.copysign(1.0, along), float(scratch.qpos[address])
                    )
            if found is None:
                direction = "up" if match.group("kind") == "pitch" else "to the left"
                reasons[f"{name}_mean_deg"] = (
                    f"{name} cannot be signed: no {TAIL_TIP_SITE} site, or it does not swing the tail tip {direction}"
                )
            tail_hinges.append((name, found))

        # The chord's base: the anchor of the first joint of the tail's first body, a child of the root.
        tail_bodies = sorted(
            {int(model.jnt_bodyid[joint]) for joint, name in enumerate(joint_names) if TAIL_HINGE.match(name)},
            key=lambda body: (depth(body), body),
        )
        base_local = None
        if tip < 0 or not tail_hinges:
            chord_reason = f"the model has no tail hinge (tail_<n>_pitch|yaw) or no {TAIL_TIP_SITE} site"
        elif int(model.body_parentid[tail_bodies[0]]) != root:
            chord_reason = (
                "a joint lies between the root and the tail's first hinge, so its base is not fixed on the root"
            )
        else:
            first = int(model.body_jntadr[tail_bodies[0]])
            rotation = scratch.xmat[root].reshape(3, 3)
            base_local = rotation.T @ (scratch.xanchor[first] - scratch.xpos[root])
            chord_reason = ""
        if chord_reason:
            reasons.update(dict.fromkeys(TAIL_CHORD_COLUMNS, chord_reason))

        # The stops: every limited tail hinge and leg joint (hinges), in model order.
        leg_joints = set().union(*chain_joints) if chain_joints else set()
        limit_joints = tuple(
            LimitJoint(name, joint, columns[int(model.jnt_qposadr[joint])])
            for joint, name in enumerate(joint_names)
            if (TAIL_HINGE.match(name) or joint in leg_joints)
            and int(model.jnt_type[joint]) == hinge
            and bool(model.jnt_limited[joint])
            and int(model.jnt_qposadr[joint]) in columns
        )

        spanned = tuple(
            foot.ankle_joint >= 0 and math.isfinite(foot.heel_line_m) and math.isfinite(foot.toe_line_m)
            for foot in morphology.feet
        )
        if not any(spanned):
            reasons.update(
                dict.fromkeys(MARGIN_COLUMNS, "no foot declares a support span (heel and toe lines, FOOT_SUPPORT)")
            )
        elliptic = int(model.opt.cone) == int(mujoco.mjtCone.mjCONE_ELLIPTIC)
        if not elliptic:
            reason = "the model's friction cone is pyramidal: its efc rows are pyramid edges, not the sliding velocity"
            reasons.update(
                {
                    f"{foot.label}_{name}": reason
                    for foot in morphology.feet
                    for name in ("slip_m", "slip_speed_max_mps")
                }
            )

        # The bases of the wrenches: the shallowest body with a tail (a neck) joint, and its first joint.
        wrench_joints = []
        for base, pattern in zip(WRENCH_BASES, (TAIL_JOINT, NECK_JOINT)):
            bodies = sorted(
                {int(model.jnt_bodyid[joint]) for joint, name in enumerate(joint_names) if pattern.match(name)},
                key=lambda body: (depth(body), body),
            )
            if bodies and bodies[0] > 0:
                first = int(model.body_jntadr[bodies[0]])
                wrench_joints.append((joint_names[first], first))
            else:
                wrench_joints.append(("", -1))
                part = "tail" if base == "tail_base" else "neck"
                reasons.update(
                    {f"{base}_{name}": f"the model has no {part} joint ({pattern.pattern})" for name in WRENCH_FIELDS}
                )

        head_body: tuple[str, int] = ("", -1)
        for name in HEAD_TIP_SITES:
            site = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name))
            if site >= 0:
                body = int(model.site_bodyid[site])
                head_body = (str(model.body(body).name), body)
                break
        if head_body[1] < 0:
            reasons.update(
                dict.fromkeys(HEAD_RATE_COLUMNS, f"the model has no head-tip site ({' or '.join(HEAD_TIP_SITES)})")
            )

        return cls(
            labels=tuple(foot.label for foot in morphology.feet),
            actuator_names=tuple(names),
            force_limits=force_limits,
            groups=tuple(groups),
            foot_actuators=foot_actuators,
            tail_hinges=tuple(tail_hinges),
            tail_tip_site=tip if base_local is not None else -1,
            tail_tip_site_name=TAIL_TIP_SITE if base_local is not None else None,
            tail_base_local=None if base_local is None else np.asarray(base_local, dtype=np.float64),
            root_pitch_mean=bool(root_pitch_mean),
            elliptic=elliptic,
            limit_joints=limit_joints,
            spanned=spanned,
            mass_kg=float(model.body_subtreemass[root]),
            gravity=abs(float(model.opt.gravity[2])),
            wrench_joints=tuple(wrench_joints),
            head_body=head_body,
            reasons=reasons,
        )

    # ------------------------------------------------------------------ the columns and the captures
    def command_columns(self) -> tuple[str, ...]:
        """(1a) ``<actuator>_command_mean`` per actuator, then :data:`COMMAND_SUMMARY_COLUMNS`."""
        return tuple(f"{name}{COMMAND_SUFFIX}" for name in self.actuator_names) + COMMAND_SUMMARY_COLUMNS

    def posture_columns(self) -> tuple[str, ...]:
        """(1b) the tail hinges, the chord, the root's height, pitch range (and mean, see ``root_pitch_mean``), roll."""
        pitch = (ROOT_PITCH_MEAN,) if self.root_pitch_mean else ()
        hinges = tuple(f"{name}_mean_deg" for name, _joint in self.tail_hinges)
        return hinges + TAIL_CHORD_COLUMNS + ROOT_HEIGHT_COLUMNS + pitch + ROOT_PITCH_COLUMNS + ROOT_ROLL_COLUMNS

    def energy_columns(self) -> tuple[str, ...]:
        """(2) per group :data:`ENERGY_FIELDS`, then ``action_delta_mean``, then ``<foot>_leg_work_abs_j``."""
        groups = tuple(f"{group}_{name}" for group, _indices in self.groups for name in ENERGY_FIELDS)
        return groups + (ACTION_DELTA_COLUMN,) + tuple(f"{label}_{LEG_WORK_FIELD}" for label in self.labels)

    def drift_columns(self) -> tuple[str, ...]:
        """(3) :data:`DRIFT_COLUMNS`, then per foot :data:`SLIP_FIELDS`."""
        return DRIFT_COLUMNS + tuple(f"{label}_{name}" for label in self.labels for name in SLIP_FIELDS)

    def stop_columns(self) -> tuple[str, ...]:
        """(g) per limited tail hinge and leg joint, :data:`STOP_FIELDS`."""
        return tuple(f"{joint.name}_{name}" for joint in self.limit_joints for name in STOP_FIELDS)

    def dynamics_columns(self) -> tuple[str, ...]:
        """(h), (c, d) and (a): the margins and angular momentum, the base wrenches, the head's rate."""
        wrench = tuple(f"{base}_{name}" for base in WRENCH_BASES for name in WRENCH_FIELDS)
        return MARGIN_COLUMNS + ANGULAR_MOMENTUM_COLUMNS + wrench + HEAD_RATE_COLUMNS

    def columns(self) -> tuple[str, ...]:
        """Every ledger column, in ledger order: (1a), (1b), (2), (3), then (g), (h), (c, d), (a)."""
        return (
            self.command_columns()
            + self.posture_columns()
            + self.energy_columns()
            + self.drift_columns()
            + self.stop_columns()
            + self.dynamics_columns()
        )

    def sites(self) -> tuple[int, ...]:
        """The sites the recorder must capture for the chord (the tail tip's), empty when the chord is not read."""
        return (self.tail_tip_site,) if self.tail_tip_site >= 0 else ()

    def capture(self, recorder: "SubstepContactRecorder") -> None:
        """Ask *recorder* (before it attaches) for every capture but the sites: the caller unites those."""
        recorder.record_actuators()
        recorder.record_slip()
        recorder.record_momentum()
        if self.limit_joints:
            recorder.record_joint_limits([joint.joint for joint in self.limit_joints])
        wrenches = [joint for _name, joint in self.wrench_joints if joint >= 0]
        if wrenches:
            recorder.record_wrenches(wrenches)
        if self.head_body[1] >= 0:
            recorder.record_body_velocity([self.head_body[1]])


# ---------------------------------------------------------------------- the reduction
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


def _rms(values: np.ndarray) -> float:
    finite = _finite(values)
    return float(np.sqrt(np.mean(finite * finite))) if finite.size else NAN


def root_rotations(root_quat: np.ndarray) -> np.ndarray:
    """(T,4) wxyz quaternions -> (T,3,3) rotation matrices (body to world)."""
    quat = np.asarray(root_quat, dtype=np.float64).reshape(-1, 4)
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    rotation: np.ndarray = np.stack(
        [
            np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], axis=-1),
            np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], axis=-1),
            np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], axis=-1),
        ],
        axis=1,
    )
    return rotation


def heading_axes(root_quat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per sample, the heading frame's forward and leftward unit vectors in 3-D (T,3) each (horizontal)."""
    forward2, _pitch = heading_frames(root_quat)
    zeros = np.zeros(forward2.shape[0])
    forward = np.stack([forward2[:, 0], forward2[:, 1], zeros], axis=1)
    left = np.stack([-forward2[:, 1], forward2[:, 0], zeros], axis=1)
    return forward, left


def tail_chord_angles(trace: "EpisodeTrace", layout: BodyLayout) -> np.ndarray | None:
    """(T,) the tail chord's angle above horizontal at each boundary sample, deg; ``None`` when it is not read."""
    names = tuple(getattr(trace, "site_names", ()) or ())
    positions = getattr(trace, "site_pos", None)
    name = layout.tail_tip_site_name
    if name is None or layout.tail_base_local is None or positions is None or name not in names:
        return None
    tip = np.asarray(positions, dtype=np.float64)[:, names.index(name)]
    base = np.asarray(trace.root_pos, dtype=np.float64) + np.einsum(
        "tij,j->ti", root_rotations(trace.root_quat), layout.tail_base_local
    )
    chord = tip - base
    angles: np.ndarray = np.degrees(np.arctan2(chord[:, 2], np.hypot(chord[:, 0], chord[:, 1])))
    return angles


def _commands(out: dict[str, Any], trace: "EpisodeTrace", layout: BodyLayout, settle: int) -> None:
    actions = np.asarray(trace.action, dtype=np.float64)[settle:]
    names = layout.actuator_names
    if actions.ndim != 2 or actions.shape[1] != len(names) or not actions.size or not np.isfinite(actions).all():
        return
    means = actions.mean(axis=0)
    saturated = np.abs(actions) >= SATURATION_ABS
    rail = saturated.mean(axis=0) > RAIL_STEP_SHARE
    for index, name in enumerate(names):
        out[f"{name}{COMMAND_SUFFIX}"] = float(means[index])
    railed = []
    for index in np.flatnonzero(rail).tolist():
        positive = int(np.count_nonzero(actions[saturated[:, index], index] > 0.0))
        negative = int(np.count_nonzero(saturated[:, index])) - positive
        railed.append(f"{names[index]}{'+' if positive >= negative else '-'}")
    parked = [
        f"{names[index]}{'+' if means[index] > 0.0 else '-'}"
        for index in range(len(names))
        if not rail[index] and abs(float(means[index])) >= PARKED_COMMAND_ABS
    ]
    out["actuators_at_rail"] = float(len(railed))
    out["rail_actuators"] = ";".join(railed)
    out["actuators_parked"] = float(len(parked))
    out["parked_actuators"] = ";".join(parked)


def _posture(out: dict[str, Any], trace: "EpisodeTrace", layout: BodyLayout, window: slice) -> None:
    joints = np.asarray(trace.joint_qpos, dtype=np.float64)
    for name, joint in layout.tail_hinges:
        if joint is not None:
            out[f"{name}_mean_deg"] = math.degrees(_mean(joint.sign * (joints[window, joint.column] - joint.home)))
    chord = tail_chord_angles(trace, layout)
    if chord is not None:
        out["tail_tip_chord_mean_deg"] = _mean(chord[window])
        out["tail_tip_chord_min_deg"] = _min(chord[window])
        out["tail_tip_chord_max_deg"] = _max(chord[window])
    height = np.asarray(trace.root_pos, dtype=np.float64)[window, 2]
    out["root_height_mean_m"] = _mean(height)
    out["root_height_min_m"] = _min(height)
    _forward, pitch = heading_frames(trace.root_quat)
    if layout.root_pitch_mean:
        out[ROOT_PITCH_MEAN] = _mean(pitch[window])
    out["root_pitch_min_deg"] = _min(pitch[window])
    out["root_pitch_max_deg"] = _max(pitch[window])
    roll = np.degrees(np.arcsin(np.clip(root_rotations(trace.root_quat)[:, 2, 1], -1.0, 1.0)))
    out["root_roll_mean_deg"] = _mean(roll[window])
    out["root_roll_min_deg"] = _min(roll[window])
    out["root_roll_max_deg"] = _max(roll[window])


def command_deltas(trace: "EpisodeTrace", settle: int) -> np.ndarray | None:
    """(D, nu) the window's step-to-step command changes: each window step from ``max(settle, 1)`` against the one
    before it (the evaluation's ``action_delta`` rule); ``None`` when a step it reads has no action."""
    actions = np.asarray(trace.action, dtype=np.float64)
    first = max(settle, 1)
    if actions.shape[0] <= first:
        return np.zeros((0, actions.shape[1] if actions.ndim == 2 else 0))
    used = actions[first - 1 :]
    if not np.isfinite(used).all():
        return None
    deltas: np.ndarray = np.diff(used, axis=0)
    return deltas


def _energy(out: dict[str, Any], trace: "EpisodeTrace", layout: BodyLayout, settle: int) -> None:
    deltas = command_deltas(trace, settle)
    nu = len(layout.actuator_names)
    if deltas is not None and deltas.shape[1:] == (nu,) and deltas.shape[0]:
        out[ACTION_DELTA_COLUMN] = float(np.mean(np.sum(deltas * deltas, axis=1)))
        for group, indices in layout.groups:
            if indices:
                part = deltas[:, list(indices)]
                out[f"{group}_command_delta_rms"] = float(np.sqrt(np.mean(part * part)))
    force_substep = getattr(trace, "actuator_force_substep", None)
    velocity_substep = getattr(trace, "actuator_velocity_substep", None)
    if force_substep is None or velocity_substep is None:
        return
    force = np.asarray(force_substep, dtype=np.float64)[settle:]
    velocity = np.asarray(velocity_substep, dtype=np.float64)[settle:]
    if force.shape[-1] != nu or not force.size:
        return
    dt_sub = float(trace.dt) / int(trace.frame_skip)
    power = force * velocity  # (W, S, nu)
    fraction = actuator_force_fraction(force.reshape(-1, nu), layout.force_limits)
    limited = np.isfinite(layout.force_limits).all(axis=1)
    for group, indices in layout.groups:
        if not indices:
            continue
        chosen = list(indices)
        part = power[..., chosen]
        out[f"{group}_work_abs_j"] = float(np.abs(part).sum() * dt_sub)
        out[f"{group}_work_net_j"] = float(part.sum() * dt_sub)
        out[f"{group}_power_peak_w"] = float(np.abs(part).sum(axis=-1).max())
        out[f"{group}_torque_rms_nm"] = float(np.sqrt(np.mean(force[..., chosen] ** 2)))
        bounded = [index for index in chosen if limited[index]]
        if bounded:
            out[f"{group}_torque_rms_frac"] = _rms(fraction[:, bounded])
    for label, indices in zip(layout.labels, layout.foot_actuators):
        if indices:
            out[f"{label}_{LEG_WORK_FIELD}"] = float(np.abs(power[..., list(indices)]).sum() * dt_sub)


def _drift(
    out: dict[str, Any], trace: "EpisodeTrace", layout: BodyLayout, settle: int, stance: "StanceEpisodeMetrics"
) -> None:
    from .stance_metrics import _yaw

    steps = trace.length
    out["window_displacement_m"] = float(stance.window_displacement_m)
    out["window_yaw_change_deg"] = float(stance.window_yaw_change_deg)
    root = np.asarray(trace.root_pos, dtype=np.float64)
    quat = np.asarray(trace.root_quat, dtype=np.float64)
    # The window's reference is the last settle step's sample, the spawn's without a settle window (stance_metrics).
    before = root[settle - 1, :2] if settle >= 1 else np.asarray(trace.spawn_root_pos, dtype=np.float64)[:2]
    heading = quat[settle - 1] if settle >= 1 else np.asarray(trace.spawn_root_quat, dtype=np.float64)
    forward, _pitch = heading_frames(heading[None])
    displacement = root[steps - 1, :2] - before
    out["window_displacement_fore_aft_m"] = float(np.dot(displacement, forward[0]))
    out["window_displacement_lateral_m"] = float(np.dot(displacement, [-forward[0, 1], forward[0, 0]]))
    path = np.concatenate([before[None], root[settle:steps, :2]])
    out["window_path_length_m"] = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
    yaw = np.unwrap(np.concatenate([[_yaw(np.asarray(trace.spawn_root_quat))], _yaw(quat)]))  # [spawn, step 0, ...]
    out["window_yaw_change_signed_deg"] = float(np.degrees(yaw[steps] - yaw[settle]))
    feet = np.asarray(trace.foot_pos, dtype=np.float64)
    feet_before = feet[settle - 1, :, :2] if settle >= 1 else np.asarray(trace.spawn_foot_pos)[:, :2]
    travel = np.linalg.norm(feet[steps - 1, :, :2] - feet_before, axis=-1)  # stance_metrics' max_foot_slide_m terms
    slip = getattr(trace, "leg_slip_speed", None)
    speed = None if slip is None or not layout.elliptic else np.asarray(slip, dtype=np.float64)[settle:]
    dt_sub = float(trace.dt) / int(trace.frame_skip)
    for index, label in enumerate(layout.labels):
        if index < travel.shape[0]:
            out[f"{label}_site_travel_m"] = float(travel[index])
        if speed is not None and index < speed.shape[-1]:
            loaded = _finite(speed[..., index])
            out[f"{label}_slip_m"] = float(loaded.sum() * dt_sub)
            out[f"{label}_slip_speed_max_mps"] = float(loaded.max()) if loaded.size else NAN


def _stops(out: dict[str, Any], trace: "EpisodeTrace", layout: BodyLayout, settle: int) -> None:
    names = tuple(getattr(trace, "joint_limit_names", ()) or ())
    torque = getattr(trace, "joint_limit_torque", None)
    if torque is None:
        return
    torque = np.asarray(torque, dtype=np.float64)[settle:]
    for joint in layout.limit_joints:
        if joint.name in names:
            series = torque[..., names.index(joint.name)]
            out[f"{joint.name}_stop_share"] = float(np.mean(series != 0.0))
            out[f"{joint.name}_stop_torque_mean_nm"] = float(np.mean(series))


def _margins(out: dict[str, Any], trace: "EpisodeTrace", layout: BodyLayout, window: slice) -> None:
    momentum = getattr(trace, "centroidal_momentum", None)
    if momentum is None:
        return
    momentum = np.asarray(momentum, dtype=np.float64)
    forward, left = heading_axes(trace.root_quat)
    angular = momentum[:, :3]
    for axis, components in (
        ("pitch", np.einsum("tj,tj->t", angular, left)),
        ("roll", np.einsum("tj,tj->t", angular, forward)),
        ("yaw", angular[:, 2]),
    ):
        out[f"angular_momentum_{axis}_rms_nms"] = _rms(components[window])
        out[f"angular_momentum_{axis}_max_abs_nms"] = _max(np.abs(components[window]))
    com = getattr(trace, "com_pos", None)
    lines = getattr(trace, "support_line_pos", None)
    spanned = np.flatnonzero(np.asarray(layout.spanned, dtype=bool))
    if not spanned.size or com is None or lines is None:
        return
    com = np.asarray(com, dtype=np.float64)
    # The heel and toe lines as balance_telemetry takes them: each spanned foot's, along the heading, averaged.
    lines = np.asarray(lines, dtype=np.float64)[:, spanned]  # (T, S, 2, 3)
    heel = np.einsum("tj,tsj->ts", forward, lines[:, :, 0]).mean(axis=1)
    toe = np.einsum("tj,tsj->ts", forward, lines[:, :, 1]).mean(axis=1)
    height = com[:, 2]
    velocity = momentum[:, 3:5] / layout.mass_kg
    with np.errstate(invalid="ignore", divide="ignore"):
        omega0 = np.where(height > 0.0, np.sqrt(layout.gravity / np.where(height > 0.0, height, 1.0)), np.nan)
    capture = com[:, :2] + velocity / omega0[:, None]
    for name, point in (("com", com[:, :2]), ("capture", capture)):
        along = np.einsum("tj,tj->t", forward[:, :2], point)
        toe_margin, heel_margin = (toe - along)[window], (along - heel)[window]
        out[f"{name}_toe_margin_min_m"] = _min(toe_margin)
        out[f"{name}_toe_margin_mean_m"] = _mean(toe_margin)
        out[f"{name}_heel_margin_min_m"] = _min(heel_margin)
        out[f"{name}_heel_margin_mean_m"] = _mean(heel_margin)


def _wrenches(out: dict[str, Any], trace: "EpisodeTrace", layout: BodyLayout, window: slice) -> None:
    names = tuple(getattr(trace, "wrench_joint_names", ()) or ())
    wrench = getattr(trace, "joint_wrench", None)
    if wrench is None:
        return
    wrench = np.asarray(wrench, dtype=np.float64)
    forward, left = heading_axes(trace.root_quat)
    for base, (name, _joint) in zip(WRENCH_BASES, layout.wrench_joints):
        if not name or name not in names:
            continue
        # (M_p, F): the parent's wrench on the subtree about the anchor; the subtree's on the parent is minus it.
        moment = wrench[:, names.index(name), :3]
        force = wrench[:, names.index(name), 3:]
        pitch = np.einsum("tj,tj->t", moment, left)[window]  # -M . (-left): + pitches the parent nose up
        out[f"{base}_force_down_mean_n"] = _mean(force[window, 2])
        out[f"{base}_pitch_moment_mean_nm"] = _mean(pitch)
        finite = _finite(pitch)
        out[f"{base}_pitch_moment_std_nm"] = float(np.std(finite)) if finite.size else NAN
        out[f"{base}_roll_moment_mean_nm"] = _mean(-np.einsum("tj,tj->t", moment, forward)[window])
        out[f"{base}_yaw_moment_mean_nm"] = _mean(-moment[window, 2])


def _head_rate(out: dict[str, Any], trace: "EpisodeTrace", layout: BodyLayout, window: slice) -> None:
    names = tuple(getattr(trace, "angvel_body_names", ()) or ())
    angvel = getattr(trace, "body_angvel", None)
    name = layout.head_body[0]
    if angvel is None or not name or name not in names:
        return
    rate = np.degrees(np.asarray(angvel, dtype=np.float64)[:, names.index(name)])  # (T, 3), deg/s
    forward, left = heading_axes(trace.root_quat)
    out["head_angular_rate_rms_dps"] = _rms(np.linalg.norm(rate[window], axis=1))
    out["head_pitch_rate_rms_dps"] = _rms(np.einsum("tj,tj->t", rate, left)[window])
    out["head_roll_rate_rms_dps"] = _rms(np.einsum("tj,tj->t", rate, forward)[window])
    out["head_yaw_rate_rms_dps"] = _rms(rate[window, 2])


def empty_body_row(layout: BodyLayout) -> dict[str, Any]:
    """Every column unmeasured: NaN, and ``""`` for the text columns."""
    return {column: "" if column in TEXT_COLUMNS else NAN for column in layout.columns()}


def episode_body_telemetry(
    trace: "EpisodeTrace",
    layout: BodyLayout,
    *,
    settle_steps: int,
    stance: "StanceEpisodeMetrics | None" = None,
) -> dict[str, Any]:
    """The body columns of one episode, keyed as :meth:`BodyLayout.columns` (NaN, or ``""``, where unmeasured).

    *stance*: the episode's ``StanceEpisodeMetrics`` (the in-training
    screen's, when it computed them), whose ``window_displacement_m`` and
    ``window_yaw_change_deg`` are copied; computed here
    (``stance_metrics.episode_stance_metrics``) when not given.
    """
    settle = _check_settle(settle_steps)
    out = empty_body_row(layout)
    steps = trace.length
    if steps <= settle:
        return out
    window = slice(settle, steps)
    if stance is None:
        from .stance_metrics import episode_stance_metrics

        stance = episode_stance_metrics(trace, settle_steps=settle)
    _commands(out, trace, layout, settle)
    _posture(out, trace, layout, window)
    _energy(out, trace, layout, settle)
    _drift(out, trace, layout, settle, stance)
    _stops(out, trace, layout, settle)
    _margins(out, trace, layout, window)
    _wrenches(out, trace, layout, window)
    _head_rate(out, trace, layout, window)
    # A model-level reason wins: a column the model cannot measure reads NaN on every episode.
    for column in layout.reasons:
        if column in out:
            out[column] = "" if column in TEXT_COLUMNS else NAN
    return {column: out[column] for column in layout.columns()}
