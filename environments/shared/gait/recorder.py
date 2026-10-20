"""SubstepContactRecorder: floor-truth contact per foot, recorded through the env's own substep hook.

The recorder attaches to ``BaseDinoEnv._substep_probe_hook``, the slot the
step loop calls after EVERY ``mj_step`` (``base_env.py``), and reads
``model`` / ``data`` without ever writing them, so a recorded trajectory is
bit-identical to an unrecorded one (``tests/test_gait_recorder.py`` pins it,
qpos for qpos).  It edits no env file and moves no digest: ``step()`` is not
a fingerprinted interface, and no species env module imports this package.

The hot path (the hook, once per physics substep) only COPIES raw rows:

1. the previous hook runs first -- the trex behavior env's terrain probe
   owns the slot there, and is chained, not displaced;
2. at the first substep of an episode (``env._step_count == 0`` and no
   substep of the step seen yet) the spawn geometry is sampled;
3. the contact geom pairs, their normal forces and their points;
4. the env's own per-substep touch reading, ``env._foot_contact_forces()``;
5. the sole geoms' poses (for the CoP), and the hinged digits' poses (for
   where along each digit its load sits);
6. at the last substep of a control step, the boundary geometry (foot
   sites, sole frames, root pose), the joint qpos and the applied ``ctrl``.

``end_episode`` then reduces the whole episode at once in numpy
(``np.bincount`` over (step, substep, foot) and (step, substep, support
geom)): a first per-contact Python version cost +0.9 to +1.9 ms per control
step, and batching the classification per episode is what brought it down to
+0.19 to +0.41 ms (+21 to +33%) plus 21-105 ms of reduction per 1000-step
zero-action episode across the six species (trex: 1.15 s -> 1.53 s + 44 ms;
sequential, ``OMP_NUM_THREADS=1``, best of 2, on a quiet machine).  Those are
the PROTOTYPE's figures; this recorder also copies the leg-against-body
contacts and the applied ``ctrl``, and has been timed only on a loaded
machine since, so read them as the order of the cost, not its measure.  A
40-episode trex panel takes about 17 s longer on them.  Attach it to
EVALUATION envs only.

The normal force is decoded from ``efc_force`` exactly as ``mj_contactForce``
returns ``force[0]``: ``efc_force[efc_address]`` for an elliptic cone or a
condim-1 contact, the sum of the contact's ``2 * (condim - 1)`` pyramid edge
forces for a pyramidal one, and 0 for ``efc_address == -1``.  When every geom
shares one condim and the model declares no explicit ``<pair>`` (true of all
six species: condim 3) one gather serves every contact.  Measured against a
per-contact ``mj_contactForce`` loop on all six species, under the pyramidal
cone (the trex until its physics r9, velociraptor, dibothrosuchus and
brachiosaurus until D-D30) and the elliptic one (every species since)
alike, the per-foot and per-geom forces agree EXACTLY (max |diff| 0.0).  No
plant declares the pyramidal cone now; test_gait_recorder.py keeps its
decode covered on dibothrosuchus with the cone switched in memory.

The whole-foot centre of pressure
---------------------------------
``EpisodeTrace.foot_cop`` measures every floor contact of the limb in the
SOLE frame of its own substep, as ``sole_cop`` does, but along the forward
axis from the foot's declared ankle axis (a hinge of the sole's body whose
axis is the sole's lateral axis, so a fixed line of that frame;
``morphology.FootSpec.ankle_forward_m``) and in metres, so that
``foot_cop * leg_floor_mean`` is the step's floor-load moment
about the ankle axis with each lever arm measured along the sole.  That is
the vertical load's moment about the ankle axis exactly while the sole is
level, and differs from it by at most ``|d . n| sin(tilt) + |d . f| (1 -
cos(tilt))`` per contact otherwise (``d`` from the ankle, ``n`` and ``f`` the
sole's normal and forward axes): about the ankle's height above the contacts
times the sine of the sole's tilt (``tests/test_gait_recorder.py`` checks
both).  It is not the ankle actuator's torque, which also carries the
floor's shear times that height.

Timing
------
The hook runs after each ``mj_step``, so ``data.contact``, ``efc_force`` and
the kinematics (``xpos``, ``xquat``, ``geom_xmat``, ``site_xpos``) all
belong to that substep's forward pass -- the state at the START of the
substep -- while ``qpos`` is already integrated.  Hence:

* the geometry sampled at an episode's first substep is exactly the
  post-reset spawn pose (bitwise, checked on trex and compsognathus);
* the geometry sampled at a step's last substep is consistent with the
  contacts read at that substep and sits ONE physics timestep (2 ms) before
  the control boundary -- immaterial for stance -- while ``joint_qpos`` is
  the boundary state;
* both are taken inside the hook, so the recorder stays correct when the
  caller learns of a step only after the env has already auto-reset (SB3
  ``DummyVecEnv``): a new episode is detected lazily, at its first substep,
  and ``end_step`` / ``end_episode`` only close what the hook recorded.

``data.contact.<field>`` returns VIEWS into MjData's contact buffer, which
the next ``mj_step`` overwrites, so every per-substep row is copied.

Whole-body balance and floor forces (opt-in)
--------------------------------------------
``balance=True`` also copies, once per control step at the boundary
substep, the animal's centre of mass (``subtree_com`` of the free root),
the applied actuator forces, the declared ankles' applied torques
(``qfrc_actuator + qfrc_passive`` at the ankle dof), and each contact's
frame, constraint rows (``efc_address``, ``dim`` and the step's
``efc_force``) and sliding friction, plus the centre of mass at the
episode's first substep.  ``_reduce`` turns them into the
:class:`EpisodeTrace` balance fields (``com_pos`` ... ``digit_tip_foot``),
which :mod:`.balance_telemetry` reduces, decoding the tangential forces
of the whole episode in one vectorised pass (decoded at each step, they
cost about 12 us a T. rex step).  The tangential forces are the
elliptic cone's two tangential ``efc_force`` rows, so the force on the
animal of a floor contact is ``s * frame^T (normal, t1, t2)`` with ``s = +1``
when geom1 is the floor (exactly ``mj_contactForce``'s first three
components, rotated to the world); under a pyramidal cone they are NaN, and
so is every floor force.  Every addition is behind ``balance``: with it off
(the certificate, the replay CSV, the probes), the hot path and the trace
are what they were.  With it on, the copies run last in the hook, in their
own guard, and the reduction builds the balance fields in its own guard
after every other field, so a failure in either costs the balance fields
(``None``) and logs once, never the step, the episode or the other fields.

Hook ownership and failing closed
---------------------------------
The hook is a single slot.  :meth:`SubstepContactRecorder.attach` chains
whatever sat there and :meth:`~SubstepContactRecorder.detach` restores it
exactly -- deleting the instance attribute it created when the env had
none, so the class-level ``None`` shows through again -- and refuses while a
foreign hook sits on top.  A tool that overwrites the slot after the
recorder attached (``stance_duty_validation.py`` does) silently disables
it, which ``end_step`` then reports loudly: it raises :class:`RecorderError`
unless exactly ``frame_skip`` substeps were seen.  So does an
``end_episode`` with an unclosed step or no steps, and a new episode that
starts before the previous one was closed.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Callable

import mujoco
import numpy as np

from ..floor_contact.reduction import (
    classify_floor_contacts,
    decode_normal_forces,
    decode_settings,
    per_substep_force,
    sole_cop_steps,
    whole_foot_cop_steps,
)
from ..model_id import ModelId
from .constants import CONTACT_THRESHOLD_N
from .morphology import Morphology

_HOOK = "_substep_probe_hook"

logger = logging.getLogger(__name__)


class RecorderError(RuntimeError):
    """The recorder saw a step sequence it cannot reduce faithfully, so it refuses rather than guess."""


@dataclass(frozen=True, eq=False)
class EpisodeTrace:
    """One recorded episode, per control step: T steps, F feet, G support geoms.

    Forces are floor NORMAL forces in N, reduced over the step's substeps;
    lengths are in m and angles in degrees.  Geometry is the boundary sample
    (the last substep's forward pass; see the module docstring).
    """

    model_id: ModelId
    dt: float
    frame_skip: int
    body_weight_n: float
    labels: tuple[str, ...]
    #: Per support geom: its foot, and its name.
    support_geom_foot: tuple[int, ...]
    support_geom_names: tuple[str, ...]
    #: (T,F) the whole limb's floor force: mean / min / max over the substeps,
    #: and the fraction of substeps above CONTACT_THRESHOLD_N.
    leg_floor_mean: np.ndarray
    leg_floor_min: np.ndarray
    leg_floor_max: np.ndarray
    leg_down_frac: np.ndarray
    #: (T,G) per registered support geom: mean floor force, fraction of substeps above the threshold.
    support_floor_mean: np.ndarray
    support_loaded_frac: np.ndarray
    #: (T,F) leg geoms outside the support set on the floor (shin, heel capsule, the robot's cheeks).
    offsupport_floor_mean: np.ndarray
    #: (T,) trunk, tail, head and arms on the floor: mean and max over the substeps.
    nonleg_floor_mean: np.ndarray
    nonleg_floor_max: np.ndarray
    #: (T,) every animal geom on the floor; the max is the impact peak (the reset pop, a landing).
    total_floor_mean: np.ndarray
    total_floor_max: np.ndarray
    #: (T,) substeps with every leg at or below the threshold.
    feet_airborne_substeps: np.ndarray
    #: (T,F) normal force from contacts with ANOTHER leg (foot-on-foot, limb-on-limb); (T,) the
    #: fraction of substeps with any such contact.
    interleg_force_mean: np.ndarray
    interleg_substep_frac: np.ndarray
    #: (T,F) normal force from contacts of a leg with the rest of the animal (trunk, tail, arms);
    #: (T,) the fraction of substeps with any such contact.
    leg_body_force_mean: np.ndarray
    leg_body_substep_frac: np.ndarray
    #: (T,F) loaded (> threshold) floor contact points on the sole geom, mean over the substeps:
    #: about 4 for a flat box, 2 on an edge, 1 on a corner.  0 for a foot without a sole.
    sole_contacts_mean: np.ndarray
    #: (T,F,2) force-weighted centre of pressure on the sole, (fore +, outer +) / half-size;
    #: NaN on a step with no sole contact.
    sole_cop: np.ndarray
    #: (T,F) the WHOLE-FOOT fore-aft centre of pressure, in m ahead of the foot's declared ankle axis
    #: (``FOOT_SUPPORT``'s ``ankle``): the force-weighted position of every floor contact of the limb
    #: (sole, digits and any other leg geom) along the sole's forward axis of the contact's own substep,
    #: pooled over the step's substeps as ``sole_cop`` is.  Times the limb's floor force
    #: (``leg_floor_mean``) it is the step's vertical-load moment about the ankle axis, up to the sole's
    #: tilt (module docstring).  NaN on a step with no limb floor force, and on every step of a foot that
    #: declares no ankle.
    foot_cop: np.ndarray
    #: (F,2) each foot's declared support span (heel line, toe line), m ahead of its ankle axis along the
    #: sole's forward axis: what normalises ``foot_cop``; NaN for a foot that declares none.
    foot_support_span: np.ndarray
    #: (T,G) per support geom that is a hinged digit (``Morphology.support_geom_mtp``): the force-weighted
    #: position of its floor contacts along the digit, 0 at the MTP joint's axis and 1 at the capsule's
    #: tip, pooled over the substeps; NaN for every other support geom and on a step the digit carries nothing.
    support_load_pos: np.ndarray
    #: (T,F) the MIN over substeps of ``env._foot_contact_forces()``: bitwise the env's own
    #: ``_aggregated_foot_contact_forces()``, and still valid after an auto-reset.
    touch: np.ndarray
    #: (T,F,3) reference sites; (T,3) / (T,4) root body position and quaternion (w, x, y, z).
    foot_pos: np.ndarray
    root_pos: np.ndarray
    root_quat: np.ndarray
    #: (T, nq - 7) every qpos entry but the root free joint's, at the control boundary.
    joint_qpos: np.ndarray
    #: (T,F) the sole's tilt from flat, roll (outer edge up +) and pitch (toe up +); NaN without a sole.
    sole_tilt_deg: np.ndarray
    sole_roll_deg: np.ndarray
    sole_pitch_deg: np.ndarray
    #: (T,F) box soles: highest minus lowest bottom corner, and the lowest corner's height above
    #: z = 0 (the plane; a heightfield's local height is not subtracted).  NaN for other soles.
    sole_corner_lift: np.ndarray
    sole_min_height: np.ndarray
    #: (T,nu) the policy's command, clipped to [-1, 1], before any action filter (NaN when
    #: ``end_step`` was given none); the ``ctrl`` the plant applied (after the filter and scaling);
    #: (T,) the reward ``end_step`` was given (NaN when none).
    action: np.ndarray
    ctrl: np.ndarray
    reward: np.ndarray
    #: The post-reset pose: (F,3) sites, (3,) root position, (4,) root quaternion, (F,) sole tilt.
    spawn_foot_pos: np.ndarray
    spawn_root_pos: np.ndarray
    spawn_root_quat: np.ndarray
    spawn_sole_tilt_deg: np.ndarray
    #: (F,) the keyframe's authored sole tilt (the morphology's), NaN without a sole.
    home_sole_tilt_deg: np.ndarray
    terminated: bool = False
    truncated: bool = False
    termination_reason: str = ""
    #: ``per_geom=True`` only: (T,A) each colliding animal geom's mean floor force, and the names.
    geom_floor_mean: np.ndarray | None = None
    geom_floor_names: tuple[str, ...] = ()
    # --- ``balance=True`` only (``None`` otherwise, and when the balance capture failed) ---
    #: (T,3) the whole animal's centre of mass (``subtree_com`` of the free root; a prey is a separate
    #: mocap body), boundary sample; (3,) the same at the episode's first substep, ``frame_skip - 1``
    #: substeps before the first boundary sample.
    com_pos: np.ndarray | None = None
    spawn_com_pos: np.ndarray | None = None
    #: (T,nu) the applied actuator forces, boundary substep.
    actuator_force: np.ndarray | None = None
    #: (T,F) ``qfrc_actuator + qfrc_passive`` at each foot's declared ankle dof (N m about the joint
    #: axis); NaN for a foot that declares no ankle.
    ankle_torque: np.ndarray | None = None
    #: (T,F+1,3) the net floor force ON the animal, world frame, boundary substep: per limb, then the
    #: rest of the animal (trunk, tail, arms).  NaN wherever a contact's tangential force is unknown
    #: (a pyramidal cone).
    floor_force: np.ndarray | None = None
    #: (T,3) the whole animal's floor centre of pressure, ``sum F_z p / sum F_z`` over its floor
    #: contacts with ``F_z > 0``; NaN when that sum is at most ``CONTACT_THRESHOLD_N``.
    floor_cop: np.ndarray | None = None
    #: (T,F) the ``F_z``-weighted mean sliding friction of the limb's floor contacts; NaN unloaded.
    floor_mu: np.ndarray | None = None
    #: (T,F,2,3) a foot with a declared support span: its heel line and toe line points on the
    #: sole's centre line, from the boundary sole pose; NaN for every other foot.
    support_line_pos: np.ndarray | None = None
    #: (T,D,3) each hinged digit's tip (the capsule's far end), boundary substep; (D,) its foot.
    digit_tip_pos: np.ndarray | None = None
    digit_tip_foot: np.ndarray | None = None

    @property
    def length(self) -> int:
        return int(self.leg_floor_mean.shape[0])

    @property
    def n_feet(self) -> int:
        return int(self.leg_floor_mean.shape[1])


class SubstepContactRecorder:
    """Records floor truth on one env: ``attach()``, ``end_step`` after every step, ``end_episode`` at the end.

    Usage (always detach in a ``finally``, or use the recorder as a context
    manager)::

        morphology = Morphology.from_env(env, model_id)
        with SubstepContactRecorder(env, morphology) as recorder:
            obs, _ = env.reset(seed=seed)
            while True:
                action = predict(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                recorder.end_step(action, reward)
                if terminated or truncated:
                    break
            trace = recorder.end_episode(terminated=terminated, truncated=truncated)

    Under SB3, attach to ``vec_env.venv.envs[0].unwrapped`` and call
    ``end_step(actions[0], rewards[0])`` after each vec step: the auto-reset
    needs no wrapper.  ``per_geom=True`` also records every colliding animal
    geom's floor force (``EpisodeTrace.geom_floor_mean``), a debugging aid
    that roughly doubles the trace.  ``balance=True`` also records the
    whole-body balance and floor-force fields (the module docstring); a
    model the recorder cannot set them up for raises :class:`RecorderError`
    here, so a caller can fall back to ``balance=False``.
    """

    def __init__(self, env: Any, morphology: Morphology, *, per_geom: bool = False, balance: bool = False) -> None:
        unwrapped = getattr(env, "unwrapped", env)
        if not hasattr(type(unwrapped), _HOOK) or not hasattr(unwrapped, "_foot_contact_forces"):
            raise RecorderError(f"{type(unwrapped).__name__} has no {_HOOK} slot; it is not a BaseDinoEnv")
        model = unwrapped.model
        if int(unwrapped.frame_skip) != morphology.frame_skip:
            raise RecorderError(
                f"the morphology was built for frame_skip={morphology.frame_skip}, the env runs {unwrapped.frame_skip}"
            )
        if morphology.geom_foot.shape != (model.ngeom,):
            raise RecorderError("the morphology was built for another model (its geom tables do not fit)")
        self._env = unwrapped
        self._morphology = morphology
        self._frame_skip = morphology.frame_skip
        self._nu = int(model.nu)
        self._sites = np.array([foot.primary_site for foot in morphology.feet], dtype=np.int64)
        # A foot without a sole samples geom 0 in its slot; the reduction masks it out.
        self._soles = np.array([max(foot.sole_geom, 0) for foot in morphology.feet], dtype=np.int64)
        self._root = int(morphology.root_body)
        self._joint_qpos_index = joint_qpos_addresses(model, self._root)
        self._corners_local = [_sole_corners(model, foot) for foot in morphology.feet]
        self._half_size = [
            np.asarray(model.geom_size[foot.sole_geom], dtype=np.float64) if foot.sole_geom >= 0 else None
            for foot in morphology.feet
        ]
        self._support_names = tuple(str(model.geom(int(geom)).name) for geom in morphology.support_geom_ids)
        self._ankle_forward = np.array([foot.ankle_forward_m for foot in morphology.feet], dtype=np.float64)
        self._support_span = np.array(
            [[foot.heel_line_m, foot.toe_line_m] for foot in morphology.feet], dtype=np.float64
        ).reshape(-1, 2)
        # The hinged digits: their support index, geom, and the MTP axis and the tip along the capsule's
        # local z axis (constants in the geom frame: the MTP joint is a hinge of the digit's own body).
        mtp = tuple(morphology.support_geom_mtp) or (-1,) * len(morphology.support_geom_ids)
        self._digit_support = np.array([index for index, joint in enumerate(mtp) if joint >= 0], dtype=np.int64)
        self._digit_geoms = np.array(
            [morphology.support_geom_ids[index] for index in self._digit_support], dtype=np.int64
        )
        self._digit_base_z, self._digit_tip_z = _digit_axis_ends(
            model, self._digit_geoms, [mtp[i] for i in self._digit_support]
        )
        self._per_geom = bool(per_geom)
        self._animal_index = np.full(model.ngeom, -1, dtype=np.int64)
        self._animal_index[list(morphology.animal_geoms)] = np.arange(len(morphology.animal_geoms))
        self._animal_names = tuple(str(model.geom(int(geom)).name) for geom in morphology.animal_geoms)
        self._decoded_model: Any = None
        self._edge_offsets: np.ndarray | None = None
        self._elliptic = False
        self._attached = False
        self._had_instance_hook = False
        self._previous_hook: Callable[[], None] | None = None
        self._bound_hook = self._hook
        self._balance = bool(balance)
        self._balance_failed = False
        self._warned_balance = False
        if self._balance:
            self._setup_balance(model, morphology)
        self._reset_episode_buffers()

    def _setup_balance(self, model: Any, morphology: Morphology) -> None:
        """Resolve the indices the balance capture reads (read-only); :class:`RecorderError` when it cannot."""
        try:
            feet = morphology.feet
            ankle_dofs = np.array(
                [int(model.jnt_dofadr[foot.ankle_joint]) if foot.ankle_joint >= 0 else -1 for foot in feet],
                dtype=np.int64,
            )
            # The -1 is never an index: the hot path gathers at `_ankle_take` and writes NaN off the ankles
            # (a bare [-1] would silently read the last dof).
            self._ankle_dofs = ankle_dofs
            self._has_ankle = ankle_dofs >= 0
            self._ankle_take = np.maximum(ankle_dofs, 0)
            support_foot = tuple(morphology.support_geom_foot)
            self._digit_tip_foot = np.array(
                [support_foot[int(index)] for index in self._digit_support.tolist()], dtype=np.int64
            )
            self._digit_tip_local = np.asarray(self._digit_tip_z, dtype=np.float64).copy()
            self._line_offsets = np.array(
                [[foot.ankle_forward_m + foot.heel_line_m, foot.ankle_forward_m + foot.toe_line_m] for foot in feet],
                dtype=np.float64,
            ).reshape(-1, 2)
            self._forward_axis = np.array([foot.sole_forward_axis for foot in feet], dtype=np.int64)
            self._forward_sign = np.array([foot.sole_forward_sign for foot in feet], dtype=np.float64)
            self._nonleg_foot = len(feet)
        except Exception as exc:  # noqa: BLE001 - reported as the recorder's own refusal
            raise RecorderError(f"the balance capture cannot be set up for this model: {exc!r}") from exc

    # ------------------------------------------------------------------ attachment
    @property
    def attached(self) -> bool:
        return self._attached

    @property
    def episode_open(self) -> bool:
        """An episode has started (its first substep was seen) and not yet been closed."""
        return self._in_episode

    def attach(self) -> "SubstepContactRecorder":
        """Install the recorder's hook, chaining the one already there; refuses a second attach."""
        if self._attached:
            raise RecorderError("the recorder is already attached")
        env = self._env
        self._had_instance_hook = _HOOK in vars(env)
        self._previous_hook = getattr(env, _HOOK)
        setattr(env, _HOOK, self._bound_hook)
        self._attached = True
        self._reset_episode_buffers()
        return self

    def detach(self) -> None:
        """Restore the hook slot exactly as :meth:`attach` found it; a no-op when not attached.

        Raises :class:`RecorderError`, leaving everything in place, when the
        slot no longer holds this recorder's hook: something chained on top
        must detach first, or a tool overwrote the slot.
        """
        if not self._attached:
            return
        env = self._env
        if getattr(env, _HOOK, None) is not self._bound_hook:
            raise RecorderError(f"another hook sits on top of the recorder in {_HOOK}; detach it first")
        if self._had_instance_hook:
            setattr(env, _HOOK, self._previous_hook)
        else:
            delattr(env, _HOOK)
        self._previous_hook = None
        self._attached = False

    def __enter__(self) -> "SubstepContactRecorder":
        return self.attach()

    def __exit__(self, *exc_info: object) -> None:
        self.detach()

    # ------------------------------------------------------------------ the hot path
    def _reset_episode_buffers(self) -> None:
        self._in_episode = False
        self._n_sub = 0  # substeps of the current, not yet closed, step
        self._contact_geoms: list[np.ndarray] = []
        self._contact_normal: list[np.ndarray] = []
        self._contact_pos: list[np.ndarray] = []
        self._contact_counts: list[int] = []  # per substep
        self._touch: list[tuple[float, ...]] = []  # per substep
        self._sole_pose: list[np.ndarray] = []  # per substep, (F, 12): xpos then xmat
        self._digit_pose: list[np.ndarray] = []  # per substep, (D, 12): the hinged digits' xpos then xmat
        self._boundary: list[np.ndarray] = []  # per step
        self._joint_qpos: list[np.ndarray] = []  # per step
        self._ctrl: list[np.ndarray] = []  # per step
        self._actions: list[np.ndarray] = []  # per closed step
        self._rewards: list[float] = []
        self._spawn: np.ndarray | None = None
        # balance=True only: per step at the boundary substep, and the spawn sample.
        self._bal_spawn_com: np.ndarray | None = None
        self._bal_com: list[np.ndarray] = []
        self._bal_force: list[np.ndarray] = []
        self._bal_ankle: list[np.ndarray] = []
        self._bal_frame: list[np.ndarray] = []  # per boundary substep with contacts, (ncon, 9)
        self._bal_rows: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []  # efc_address, dim, efc_force
        self._bal_mu: list[np.ndarray] = []  # (ncon,)

    def _configure_decode(self, model: Any) -> None:
        """Choose the normal-force decode for ``model`` (re-run when an env swaps its model)."""
        self._decoded_model = model
        self._elliptic, self._edge_offsets = decode_settings(model)

    def _normal_forces(self, contact: Any, efc_force: np.ndarray) -> np.ndarray:
        """Each contact's normal force, exactly as ``mj_contactForce`` returns ``force[0]``."""
        return decode_normal_forces(contact, efc_force, elliptic=self._elliptic, edge_offsets=self._edge_offsets)

    def _geometry_row(self, data: Any) -> np.ndarray:
        return np.concatenate(
            [
                data.site_xpos[self._sites].ravel(),
                data.geom_xpos[self._soles].ravel(),
                data.geom_xmat[self._soles].ravel(),
                data.xpos[self._root],
                data.xquat[self._root],
            ]
        )

    def _hook(self) -> None:
        previous = self._previous_hook
        if previous is not None:
            previous()
        env = self._env
        data = env.data
        if self._n_sub == 0 and int(env._step_count) == 0:
            if self._in_episode and self._actions:
                raise RecorderError("a new episode started before end_episode() closed the previous one")
            self._reset_episode_buffers()
            self._in_episode = True
            self._spawn = self._geometry_row(data)
        if env.model is not self._decoded_model:
            self._configure_decode(env.model)
        ncon = int(data.ncon)
        if ncon:
            contact = data.contact
            self._contact_normal.append(self._normal_forces(contact, data.efc_force))
            self._contact_geoms.append(contact.geom.copy())
            self._contact_pos.append(contact.pos.copy())
        self._contact_counts.append(ncon)
        self._touch.append(env._foot_contact_forces())
        self._sole_pose.append(np.concatenate([data.geom_xpos[self._soles], data.geom_xmat[self._soles]], axis=1))
        if self._digit_geoms.size:
            self._digit_pose.append(
                np.concatenate([data.geom_xpos[self._digit_geoms], data.geom_xmat[self._digit_geoms]], axis=1)
            )
        self._n_sub += 1
        if self._n_sub == self._frame_skip:
            self._boundary.append(self._geometry_row(data))
            self._joint_qpos.append(data.qpos[self._joint_qpos_index])
            self._ctrl.append(data.ctrl.copy())
        # Called only at the substeps it copies: the spawn substep and the boundary one.
        if (
            self._balance
            and not self._balance_failed
            and (self._n_sub == self._frame_skip or (self._n_sub == 1 and int(env._step_count) == 0))
        ):
            try:
                self._balance_capture(env, data)
            except Exception:  # noqa: BLE001 - the hook runs inside mj_step's loop: a failure costs the balance fields
                self._balance_failed = True
                self._warn_balance("the balance capture failed in the substep hook")

    def _warn_balance(self, message: str) -> None:
        """Warn once per recorder: a failed capture costs its remaining episodes' balance fields, a failed
        reduction that episode's (``None``)."""
        if not self._warned_balance:
            self._warned_balance = True
            logger.warning(
                "Balance telemetry: %s; the balance fields read None (logged once per recorder).",
                message,
                exc_info=True,
            )

    def _balance_capture(self, env: Any, data: Any) -> None:
        """Copy the balance rows: the spawn centre of mass at an episode's first substep, the rest at the boundary."""
        if self._n_sub == 1 and int(env._step_count) == 0:
            self._bal_spawn_com = data.subtree_com[self._root].copy()
        if self._n_sub != self._frame_skip:
            return
        self._bal_com.append(data.subtree_com[self._root].copy())
        self._bal_force.append(data.actuator_force.copy())
        take = self._ankle_take  # an undeclared ankle reads dof 0 here, NaN after the reduction
        self._bal_ankle.append(data.qfrc_actuator[take] + data.qfrc_passive[take])
        ncon = int(data.ncon)
        if ncon:
            contact = data.contact
            self._bal_frame.append(contact.frame.copy())
            self._bal_rows.append((contact.efc_address.copy(), contact.dim.copy(), data.efc_force.copy()))
            self._bal_mu.append(contact.friction[:, 0].copy())

    def _tangential_forces(self, rows: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray]]) -> np.ndarray:
        """Each contact's two tangential forces (contact frame), as ``mj_contactForce`` returns ``force[1:3]``.

        *rows* holds one ``(efc_address, dim, efc_force)`` per boundary
        substep with contacts, as the capture copied them; the result has
        one row per contact, in that order.  The elliptic cone's rows after
        the normal one, each contact's address read in its own substep's
        ``efc_force``; 0 for a contact with no rows (``efc_address`` -1) or
        of too low a dimension (condim 1), and for every contact of a
        substep whose ``efc_force`` is empty (the guard
        :meth:`_normal_forces` applies).  NaN under a pyramidal cone, whose
        rows are pyramid edges.
        """
        counts = np.array([len(address) for address, _dim, _efc in rows], dtype=np.int64)
        total = int(counts.sum())
        if not self._elliptic:
            return np.full((total, 2), np.nan)
        if not total:
            return np.zeros((0, 2), dtype=np.float64)
        sizes = np.array([efc.size for _address, _dim, efc in rows], dtype=np.int64)
        efc_force = np.concatenate([efc for _address, _dim, efc in rows])
        address = np.concatenate([address for address, _dim, _efc in rows]).astype(np.int64)
        dim = np.concatenate([dim for _address, dim, _efc in rows])
        size = np.repeat(sizes, counts)  # each contact's own substep's row count
        start = np.repeat(np.cumsum(sizes) - sizes, counts)
        offsets = np.array([1, 2], dtype=np.int64)[None, :]
        valid = (address >= 0) & (size > 0)
        if not efc_force.size:
            return np.zeros((total, 2), dtype=np.float64)
        local = np.minimum(np.where(valid, address, 0)[:, None] + offsets, np.maximum(size, 1)[:, None] - 1)
        gathered = efc_force[np.minimum(start[:, None] + local, efc_force.size - 1)]
        keep = valid[:, None] & (dim[:, None] > offsets)
        tangential: np.ndarray = np.where(keep, gathered, 0.0)
        return tangential

    # ------------------------------------------------------------------ closing steps and episodes
    def end_step(self, action: Any = None, reward: float = float("nan")) -> None:
        """Close the control step just taken, with the policy's ``action`` and the env's ``reward``.

        Call it after EVERY ``env.step`` (after the vec step under SB3).
        Raises :class:`RecorderError` unless the hook fired exactly
        ``frame_skip`` times since the last call -- 0 means the hook is not
        installed (overwritten, or never attached), ``2 * frame_skip`` a
        skipped ``end_step`` -- when the recorder never saw the episode
        start (attached mid-episode), or when ``action`` does not have the
        env's ``nu`` entries.
        """
        if self._n_sub != self._frame_skip:
            raise RecorderError(
                f"the substep hook fired {self._n_sub} times for a frame_skip={self._frame_skip} step "
                "(0: the hook is not installed; a multiple: an end_step() was skipped)"
            )
        if self._spawn is None:
            self._n_sub = 0
            raise RecorderError(
                "the recorder saw no episode start: it was attached mid-episode; reset the env after attaching"
            )
        if action is None:
            row = np.full(self._nu, np.nan)
        else:
            row = np.clip(np.asarray(action, dtype=np.float64).ravel(), -1.0, 1.0)
            if row.size != self._nu:
                raise RecorderError(f"the action has {row.size} entries; the env has {self._nu} actuators")
        self._n_sub = 0
        self._actions.append(row)
        self._rewards.append(float(reward))

    def end_episode(
        self, *, terminated: bool = False, truncated: bool = False, termination_reason: str = ""
    ) -> EpisodeTrace:
        """Reduce and return the episode recorded since its first substep, and clear the buffers.

        Raises :class:`RecorderError` with a step whose ``end_step`` has not
        been called, or when no step was recorded.
        """
        if self._n_sub:
            raise RecorderError("end_episode() with a step that end_step() has not closed")
        if self._spawn is None or not self._actions:
            raise RecorderError("end_episode() with no recorded steps")
        trace = self._reduce(terminated=bool(terminated), truncated=bool(truncated), reason=str(termination_reason))
        self._reset_episode_buffers()
        return trace

    # ------------------------------------------------------------------ the reduction
    def _reduce(self, *, terminated: bool, truncated: bool, reason: str) -> EpisodeTrace:
        morphology = self._morphology
        feet = morphology.feet
        n_feet = morphology.n_feet
        n_support = len(morphology.support_geom_ids)
        substeps = self._frame_skip
        steps = len(self._actions)
        total_substeps = steps * substeps
        if len(self._contact_counts) != total_substeps or len(self._boundary) != steps:
            raise RecorderError(
                f"{len(self._contact_counts)} substeps and {len(self._boundary)} boundary samples "
                f"recorded for {steps} steps of frame_skip {substeps}"
            )
        counts = np.asarray(self._contact_counts, dtype=np.int64)
        substep_of = np.repeat(np.arange(total_substeps), counts)
        if counts.sum():
            geoms = np.concatenate(self._contact_geoms).astype(np.int64)
            normal = np.concatenate(self._contact_normal)
            points = np.concatenate(self._contact_pos)
        else:
            geoms = np.zeros((0, 2), dtype=np.int64)
            normal = np.zeros(0)
            points = np.zeros((0, 3))
        geom1, geom2 = geoms[:, 0], geoms[:, 1]
        floor1, floor2 = morphology.is_floor[geom1], morphology.is_floor[geom2]
        other, on_floor, foot, leg_on_floor = classify_floor_contacts(morphology, geoms)

        def per_substep(mask: np.ndarray, index: np.ndarray, width: int) -> np.ndarray:
            return per_substep_force(substep_of, normal, mask, index, width, steps, substeps)

        zero = np.zeros_like(substep_of)
        leg = per_substep(leg_on_floor, foot, n_feet)
        support_index = np.where(leg_on_floor, morphology.geom_support[other], -1)
        on_support = support_index >= 0
        support = per_substep(on_support, support_index, n_support) if n_support else np.zeros((steps, substeps, 0))
        offsupport = per_substep(leg_on_floor & ~on_support, foot, n_feet)
        nonleg = per_substep(on_floor & (foot < 0), zero, 1)[..., 0]
        total = per_substep(on_floor, zero, 1)[..., 0]

        # Animal-on-animal contacts: leg against another leg, and leg against the rest of the body.
        # Contacts inside one limb are neither.
        foot1, foot2 = morphology.geom_foot[geom1], morphology.geom_foot[geom2]
        off_floor = ~floor1 & ~floor2 & morphology.is_animal[geom1] & morphology.is_animal[geom2]
        interleg = off_floor & (foot1 >= 0) & (foot2 >= 0) & (foot1 != foot2)
        interleg_force = per_substep(interleg, foot1, n_feet) + per_substep(interleg, foot2, n_feet)
        interleg_any = (np.bincount(substep_of[interleg], minlength=total_substeps) > 0).reshape(steps, substeps)
        leg_body = off_floor & ((foot1 >= 0) != (foot2 >= 0))
        body_leg = np.where(foot1 >= 0, foot1, foot2)
        leg_body_force = per_substep(leg_body, body_leg, n_feet)
        leg_body_any = (np.bincount(substep_of[leg_body], minlength=total_substeps) > 0).reshape(steps, substeps)

        # Sole contacts and the centre of pressure, in the sole frame of the contact's own substep.
        sole_of_foot = np.array([spec.sole_geom for spec in feet] + [-2])  # foot -1 -> -2, never a geom
        on_sole = leg_on_floor & (other == sole_of_foot[foot])
        loaded = on_sole & (normal > CONTACT_THRESHOLD_N)
        sole_contacts = np.bincount(
            substep_of[loaded] * n_feet + foot[loaded], minlength=total_substeps * n_feet
        ).reshape(steps, substeps, n_feet)
        pose = np.stack(self._sole_pose) if on_sole.any() or leg_on_floor.any() else None  # (total_substeps, F, 12)
        cop = sole_cop_steps(feet, self._half_size, pose, substep_of, foot, on_sole, points, normal, steps, substeps)
        # The whole-foot centre of pressure: every floor contact of the limb, in the sole frame of its own
        # substep, m ahead of the declared ankle axis along the sole's (signed) forward axis.
        foot_cop = whole_foot_cop_steps(
            feet, self._ankle_forward, pose, substep_of, foot, leg_on_floor, points, normal, steps, substeps
        )
        # Where along each hinged digit its floor load sits: 0 at the MTP axis, 1 at the tip.
        load_pos = np.full((steps, n_support), np.nan)
        if self._digit_geoms.size and on_support.any():
            digit_pose = np.stack(self._digit_pose)  # (total_substeps, D, 12)
            for column, support_column in enumerate(self._digit_support.tolist()):
                chosen = np.flatnonzero(on_support & (support_index == support_column))
                if not chosen.size:
                    continue
                at = substep_of[chosen]
                axis = digit_pose[at, column, 3:].reshape(-1, 3, 3)[:, :, 2]  # the capsule's local z, in world
                along = np.einsum("nj,nj->n", axis, points[chosen] - digit_pose[at, column, :3])
                base, tip = self._digit_base_z[column], self._digit_tip_z[column]
                position = (along - base) / (tip - base)
                weight = normal[chosen]
                step_of = at // substeps
                weight_sum = np.bincount(step_of, weights=weight, minlength=steps)
                has_load = weight_sum > 1e-9
                weighted = np.bincount(step_of, weights=weight * position, minlength=steps)
                load_pos[has_load, support_column] = weighted[has_load] / weight_sum[has_load]
        touch = np.asarray(self._touch, dtype=np.float64).reshape(steps, substeps, n_feet).min(axis=1)

        site, sole_pos, sole_frame, root_pos, root_quat = self._unpack(np.asarray(self._boundary))
        assert self._spawn is not None
        spawn_site, _spawn_sole_pos, spawn_frame, spawn_root_pos, spawn_root_quat = self._unpack(self._spawn[None])
        tilt, roll, pitch = _sole_angles(morphology, sole_frame)
        spawn_tilt, _roll, _pitch = _sole_angles(morphology, spawn_frame)
        lift = np.full((steps, n_feet), np.nan)
        lowest = np.full((steps, n_feet), np.nan)
        for index, corners in enumerate(self._corners_local):
            if corners is None:
                continue
            heights = _corner_heights(sole_pos[:, index], sole_frame[:, index], corners)
            lift[:, index] = heights.max(axis=1) - heights.min(axis=1)
            lowest[:, index] = heights.min(axis=1)

        geom_floor_mean = None
        geom_floor_names: tuple[str, ...] = ()
        if self._per_geom:
            animal = np.where(on_floor, self._animal_index[other], -1)
            geom_floor_mean = per_substep(animal >= 0, animal, len(self._animal_names)).mean(axis=1)
            geom_floor_names = self._animal_names

        # The balance fields last, in their own guard: a failure costs them alone (None), never the fields above.
        balance: dict[str, Any] = {}
        if self._balance and not self._balance_failed:  # a failed capture has already warned
            try:
                balance = self._reduce_balance(
                    steps=steps,
                    substep_of=substep_of,
                    geoms=geoms,
                    normal=normal,
                    points=points,
                    on_floor=on_floor,
                    foot=foot,
                    sole_pos=sole_pos,
                    sole_frame=sole_frame,
                )
            except Exception:  # noqa: BLE001 - the balance reduction must not cost the episode's other fields
                balance = {}
                self._warn_balance("an episode's balance fields could not be reduced")

        return EpisodeTrace(
            model_id=morphology.model_id,
            dt=morphology.dt,
            frame_skip=substeps,
            body_weight_n=morphology.body_weight_n,
            labels=morphology.labels,
            support_geom_foot=morphology.support_geom_foot,
            support_geom_names=self._support_names,
            leg_floor_mean=leg.mean(axis=1),
            leg_floor_min=leg.min(axis=1),
            leg_floor_max=leg.max(axis=1),
            leg_down_frac=(leg > CONTACT_THRESHOLD_N).mean(axis=1),
            support_floor_mean=support.mean(axis=1),
            support_loaded_frac=(support > CONTACT_THRESHOLD_N).mean(axis=1),
            offsupport_floor_mean=offsupport.mean(axis=1),
            nonleg_floor_mean=nonleg.mean(axis=1),
            nonleg_floor_max=nonleg.max(axis=1),
            total_floor_mean=total.mean(axis=1),
            total_floor_max=total.max(axis=1),
            feet_airborne_substeps=np.all(leg <= CONTACT_THRESHOLD_N, axis=2).sum(axis=1),
            interleg_force_mean=interleg_force.mean(axis=1),
            interleg_substep_frac=interleg_any.mean(axis=1),
            leg_body_force_mean=leg_body_force.mean(axis=1),
            leg_body_substep_frac=leg_body_any.mean(axis=1),
            sole_contacts_mean=sole_contacts.mean(axis=1),
            sole_cop=cop,
            foot_cop=foot_cop,
            foot_support_span=self._support_span.copy(),
            support_load_pos=load_pos,
            touch=touch,
            foot_pos=site,
            root_pos=root_pos,
            root_quat=root_quat,
            joint_qpos=np.asarray(self._joint_qpos),
            sole_tilt_deg=tilt,
            sole_roll_deg=roll,
            sole_pitch_deg=pitch,
            sole_corner_lift=lift,
            sole_min_height=lowest,
            action=np.asarray(self._actions),
            ctrl=np.asarray(self._ctrl),
            reward=np.asarray(self._rewards, dtype=np.float64),
            spawn_foot_pos=spawn_site[0],
            spawn_root_pos=spawn_root_pos[0],
            spawn_root_quat=spawn_root_quat[0],
            spawn_sole_tilt_deg=spawn_tilt[0],
            home_sole_tilt_deg=np.array([spec.home_sole_tilt_deg for spec in feet]),
            terminated=terminated,
            truncated=truncated,
            termination_reason=reason,
            geom_floor_mean=geom_floor_mean,
            geom_floor_names=geom_floor_names,
            **balance,
        )

    def _reduce_balance(
        self,
        *,
        steps: int,
        substep_of: np.ndarray,
        geoms: np.ndarray,
        normal: np.ndarray,
        points: np.ndarray,
        on_floor: np.ndarray,
        foot: np.ndarray,
        sole_pos: np.ndarray,
        sole_frame: np.ndarray,
    ) -> dict[str, Any]:
        """The :class:`EpisodeTrace` balance fields of the episode just recorded (``balance=True``).

        The boundary contacts are the last per-substep entries the hook
        appended for each step (substep ``s * frame_skip + frame_skip - 1``),
        in the order the balance capture copied their frames, constraint rows
        forces and friction.
        """
        morphology = self._morphology
        n_feet = morphology.n_feet
        substeps = self._frame_skip
        if (
            self._bal_spawn_com is None
            or len(self._bal_com) != steps
            or len(self._bal_force) != steps
            or len(self._bal_ankle) != steps
        ):
            raise RecorderError(f"{len(self._bal_com)} balance samples recorded for {steps} steps (or no spawn sample)")
        boundary = (substep_of % substeps) == substeps - 1
        if self._bal_frame:
            frames = np.concatenate(self._bal_frame).reshape(-1, 3, 3)
            tangent = self._tangential_forces(self._bal_rows)
            friction = np.concatenate(self._bal_mu)
        else:
            frames, tangent, friction = np.zeros((0, 3, 3)), np.zeros((0, 2)), np.zeros(0)
        if frames.shape[0] != int(boundary.sum()):
            raise RecorderError(
                f"{frames.shape[0]} boundary contact frames recorded for {int(boundary.sum())} boundary contacts"
            )
        step_of = substep_of[boundary] // substeps
        floor1 = morphology.is_floor[geoms[boundary, 0]]
        animal_floor = on_floor[boundary]
        limb = np.where(foot[boundary] >= 0, foot[boundary], self._nonleg_foot)
        # The force ON the animal: + the contact force when the floor is geom1 (the normal points from geom1 to
        # geom2), - when it is geom2.  frame^T (normal, t1, t2) is the contact force in the world frame.
        local = np.concatenate([normal[boundary][:, None], tangent], axis=1)
        world = np.einsum("nij,ni->nj", frames, local) * np.where(floor1, 1.0, -1.0)[:, None]
        chosen = np.flatnonzero(animal_floor)
        width = n_feet + 1
        floor_force = np.zeros((steps, width, 3))
        for axis in range(3):
            floor_force[:, :, axis] = np.bincount(
                step_of[chosen] * width + limb[chosen], weights=world[chosen, axis], minlength=steps * width
            ).reshape(steps, width)
        # The whole animal's floor CoP, from the contacts that press on it (F_z > 0).
        vertical = world[:, 2]
        pressing = animal_floor & (vertical > 0)
        press = np.flatnonzero(pressing)
        weight = np.bincount(step_of[press], weights=vertical[press], minlength=steps)
        cop = np.full((steps, 3), np.nan)
        loaded = weight > CONTACT_THRESHOLD_N
        boundary_points = points[boundary]
        for axis in range(3):
            moment = np.bincount(
                step_of[press], weights=vertical[press] * boundary_points[press, axis], minlength=steps
            )
            cop[loaded, axis] = moment[loaded] / weight[loaded]
        # Each limb's F_z-weighted sliding friction.
        on_limb = pressing & (limb < n_feet)
        mine = np.flatnonzero(on_limb)
        index = step_of[mine] * n_feet + limb[mine]
        limb_weight = np.bincount(index, weights=vertical[mine], minlength=steps * n_feet).reshape(steps, n_feet)
        limb_mu = np.bincount(index, weights=vertical[mine] * friction[mine], minlength=steps * n_feet).reshape(
            steps, n_feet
        )
        floor_mu = np.full((steps, n_feet), np.nan)
        limb_loaded = limb_weight > CONTACT_THRESHOLD_N
        floor_mu[limb_loaded] = limb_mu[limb_loaded] / limb_weight[limb_loaded]
        # The declared heel and toe lines on each spanned sole's centre line, from the boundary sole pose.
        lines = np.full((steps, n_feet, 2, 3), np.nan)
        rotation = sole_frame.reshape(steps, n_feet, 3, 3)
        for index_foot in range(n_feet):
            offsets = self._line_offsets[index_foot]
            if not np.isfinite(offsets).all():
                continue
            forward = rotation[:, index_foot, :, self._forward_axis[index_foot]] * self._forward_sign[index_foot]
            for column in range(2):
                lines[:, index_foot, column] = sole_pos[:, index_foot] + forward * offsets[column]
        # The hinged digits' tips at the boundary substep.
        digits = self._digit_geoms.size
        tips = np.zeros((steps, digits, 3))
        if digits:
            pose = np.stack(self._digit_pose[substeps - 1 :: substeps])  # (T, D, 12): the boundary substeps
            axis_z = pose[:, :, 3:].reshape(steps, digits, 3, 3)[:, :, :, 2]
            tips = pose[:, :, :3] + axis_z * self._digit_tip_local[None, :, None]
        return {
            "com_pos": np.asarray(self._bal_com, dtype=np.float64),
            "spawn_com_pos": np.asarray(self._bal_spawn_com, dtype=np.float64),
            "actuator_force": np.asarray(self._bal_force, dtype=np.float64),
            "ankle_torque": np.where(self._has_ankle, np.asarray(self._bal_ankle, dtype=np.float64), np.nan),
            "floor_force": floor_force,
            "floor_cop": cop,
            "floor_mu": floor_mu,
            "support_line_pos": lines,
            "digit_tip_pos": tips,
            "digit_tip_foot": self._digit_tip_foot.copy(),
        }

    def _unpack(self, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split :meth:`_geometry_row` rows into sites (N,F,3), sole xpos (N,F,3), sole xmat (N,F,9), root pos, quat."""
        n_feet = self._morphology.n_feet
        cut = np.cumsum([3 * n_feet, 3 * n_feet, 9 * n_feet, 3])
        site, sole_pos, sole_frame, root_pos, root_quat = np.split(rows, cut, axis=1)
        return (
            site.reshape(-1, n_feet, 3),
            sole_pos.reshape(-1, n_feet, 3),
            sole_frame.reshape(-1, n_feet, 9),
            root_pos,
            root_quat,
        )


def joint_qpos_addresses(model: Any, root_body: int) -> np.ndarray:
    """Every qpos address but the root free joint's seven, in order: the columns of ``EpisodeTrace.joint_qpos``."""
    free = [
        joint
        for joint in range(model.njnt)
        if int(model.jnt_type[joint]) == int(mujoco.mjtJoint.mjJNT_FREE) and int(model.jnt_bodyid[joint]) == root_body
    ]
    root_address = int(model.jnt_qposadr[free[0]]) if free else -7
    return np.array(
        [address for address in range(model.nq) if not root_address <= address < root_address + 7], dtype=np.int64
    )


def _digit_axis_ends(model: Any, geoms: np.ndarray, joints: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Per hinged digit capsule: its MTP axis and its tip along the capsule's local z axis (m, geom frame).

    The MTP joint's anchor (fixed in the digit's body) projected on the
    capsule axis is the base; the tip is the end of the capsule farther from
    it, the segment's half-length plus the radius out.
    """
    base = np.zeros(len(geoms))
    tip = np.zeros(len(geoms))
    for index, (geom, joint) in enumerate(zip(geoms.tolist(), joints)):
        rotation = np.zeros(9)
        mujoco.mju_quat2Mat(rotation, np.asarray(model.geom_quat[geom], dtype=np.float64))
        anchor = rotation.reshape(3, 3).T @ (
            np.asarray(model.jnt_pos[joint], dtype=np.float64) - np.asarray(model.geom_pos[geom], dtype=np.float64)
        )
        reach = float(model.geom_size[geom][1] + model.geom_size[geom][0])
        base[index] = float(anchor[2])
        tip[index] = reach if base[index] <= 0.0 else -reach
    return base, tip


def _sole_corners(model: Any, foot: Any) -> np.ndarray | None:
    """A box sole's four floor-facing corners in its local frame (4,3); ``None`` for any other sole."""
    if foot.sole_geom < 0 or foot.sole_type != int(mujoco.mjtGeom.mjGEOM_BOX):
        return None
    half = np.asarray(model.geom_size[foot.sole_geom], dtype=np.float64)
    first, second = [axis for axis in range(3) if axis != foot.sole_normal_axis]
    corners = []
    for sign_first in (-1.0, 1.0):
        for sign_second in (-1.0, 1.0):
            corner = np.zeros(3)
            corner[foot.sole_normal_axis] = foot.sole_normal_sign * half[foot.sole_normal_axis]
            corner[first] = sign_first * half[first]
            corner[second] = sign_second * half[second]
            corners.append(corner)
    return np.array(corners)


def _corner_heights(position: np.ndarray, frame: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """World heights (N,4) of a box sole's local ``corners`` (4,3), given its centre (N,3) and frame (N,9).

    A box rolled by an angle ``a`` about its forward axis lifts its outer
    corners by ``2 * half_width * sin(a)`` above the inner ones: the corner
    lift the trex audit read as "back-outer corner up".
    """
    rotation = frame.reshape(-1, 3, 3)
    heights: np.ndarray = position[:, 2][:, None] + np.einsum("nij,cj->nci", rotation, corners)[:, :, 2]
    return heights


def _sole_angles(morphology: Morphology, frames: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sole frames (N,F,9) -> tilt, roll (outer edge up +) and pitch (toe up +) in degrees; NaN without a sole."""
    count, n_feet = frames.shape[:2]
    rotation = frames.reshape(count, n_feet, 3, 3)
    tilt = np.full((count, n_feet), np.nan)
    roll = np.full((count, n_feet), np.nan)
    pitch = np.full((count, n_feet), np.nan)
    for index, foot in enumerate(morphology.feet):
        if foot.sole_geom < 0:
            continue
        normal = foot.sole_normal_sign * rotation[:, index, :, foot.sole_normal_axis]  # into the floor when flat
        forward = foot.sole_forward_sign * rotation[:, index, :, foot.sole_forward_axis]
        lateral = foot.sole_lateral_sign * rotation[:, index, :, foot.sole_lateral_axis]  # the animal's left
        tilt[:, index] = np.degrees(np.arccos(np.clip(-normal[:, 2], -1.0, 1.0)))
        pitch[:, index] = np.degrees(np.arcsin(np.clip(forward[:, 2], -1.0, 1.0)))
        left_edge_up = np.degrees(np.arcsin(np.clip(lateral[:, 2], -1.0, 1.0)))
        roll[:, index] = -left_edge_up if foot.side == "R" else left_edge_up
    return tilt, roll, pitch
