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
5. the sole geoms' poses (for the CoP);
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
per-contact ``mj_contactForce`` loop on all six species, pyramidal (trex,
velociraptor, dibothrosuchus, brachiosaurus) and elliptic (compsognathus and
its robot) alike, the per-foot and per-geom forces agree EXACTLY (max |diff|
0.0).

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

from dataclasses import dataclass
from typing import Any, Callable

import mujoco
import numpy as np

from .constants import CONTACT_THRESHOLD_N
from .morphology import Morphology

_HOOK = "_substep_probe_hook"


class RecorderError(RuntimeError):
    """The recorder saw a step sequence it cannot reduce faithfully, so it refuses rather than guess."""


@dataclass(frozen=True, eq=False)
class EpisodeTrace:
    """One recorded episode, per control step: T steps, F feet, G support geoms.

    Forces are floor NORMAL forces in N, reduced over the step's substeps;
    lengths are in m and angles in degrees.  Geometry is the boundary sample
    (the last substep's forward pass; see the module docstring).
    """

    species: str
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

        morphology = Morphology.from_env(env, species)
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
    that roughly doubles the trace.
    """

    def __init__(self, env: Any, morphology: Morphology, *, per_geom: bool = False) -> None:
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
        free = [
            joint
            for joint in range(model.njnt)
            if int(model.jnt_type[joint]) == int(mujoco.mjtJoint.mjJNT_FREE)
            and int(model.jnt_bodyid[joint]) == self._root
        ]
        root_address = int(model.jnt_qposadr[free[0]]) if free else -7
        self._joint_qpos_index = np.array(
            [address for address in range(model.nq) if not root_address <= address < root_address + 7], dtype=np.int64
        )
        self._corners_local = [_sole_corners(model, foot) for foot in morphology.feet]
        self._half_size = [
            np.asarray(model.geom_size[foot.sole_geom], dtype=np.float64) if foot.sole_geom >= 0 else None
            for foot in morphology.feet
        ]
        self._support_names = tuple(str(model.geom(int(geom)).name) for geom in morphology.support_geom_ids)
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
        self._reset_episode_buffers()

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
        self._boundary: list[np.ndarray] = []  # per step
        self._joint_qpos: list[np.ndarray] = []  # per step
        self._ctrl: list[np.ndarray] = []  # per step
        self._actions: list[np.ndarray] = []  # per closed step
        self._rewards: list[float] = []
        self._spawn: np.ndarray | None = None

    def _configure_decode(self, model: Any) -> None:
        """Choose the normal-force decode for ``model`` (re-run when an env swaps its model)."""
        self._decoded_model = model
        self._elliptic = int(model.opt.cone) == int(mujoco.mjtCone.mjCONE_ELLIPTIC)
        condims = {int(value) for value in model.geom_condim}
        self._edge_offsets = None
        if len(condims) == 1 and int(model.npair) == 0:
            condim = condims.pop()
            edges = 1 if (self._elliptic or condim == 1) else 2 * (condim - 1)
            self._edge_offsets = np.arange(edges, dtype=np.int64)[None, :]

    def _normal_forces(self, contact: Any, efc_force: np.ndarray) -> np.ndarray:
        """Each contact's normal force, exactly as ``mj_contactForce`` returns ``force[0]``."""
        address = np.asarray(contact.efc_address)
        valid = address >= 0
        if efc_force.size == 0:
            return np.zeros(address.shape[0], dtype=np.float64)
        base = np.where(valid, address, 0)[:, None]
        last = efc_force.size - 1
        if self._edge_offsets is not None:
            gathered = efc_force[np.minimum(base + self._edge_offsets, last)]
            return np.where(valid, gathered.sum(axis=1), 0.0)
        dim = np.asarray(contact.dim)
        edges = np.ones_like(dim) if self._elliptic else np.where(dim > 1, 2 * (dim - 1), 1)
        offsets = np.arange(int(edges.max()), dtype=np.int64)[None, :]
        gathered = efc_force[np.minimum(base + offsets, last)]
        keep = (offsets < edges[:, None]) & valid[:, None]
        normal: np.ndarray = np.where(keep, gathered, 0.0).sum(axis=1)
        return normal

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
        self._n_sub += 1
        if self._n_sub == self._frame_skip:
            self._boundary.append(self._geometry_row(data))
            self._joint_qpos.append(data.qpos[self._joint_qpos_index])
            self._ctrl.append(data.ctrl.copy())

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
        other = np.where(floor1, geom2, geom1)  # the non-floor side of a floor contact
        on_floor = (floor1 ^ floor2) & morphology.is_animal[other]
        foot = np.where(on_floor, morphology.geom_foot[other], -1)
        leg_on_floor = on_floor & (foot >= 0)

        def per_substep(mask: np.ndarray, index: np.ndarray, width: int) -> np.ndarray:
            flat = np.bincount(
                substep_of[mask] * width + index[mask], weights=normal[mask], minlength=total_substeps * width
            )
            return flat.reshape(steps, substeps, width)

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
        cop = np.full((steps, n_feet, 2), np.nan)
        if on_sole.any():
            pose = np.stack(self._sole_pose)  # (total_substeps, F, 12)
            for index in np.unique(foot[on_sole]).tolist():
                spec = feet[index]
                half = self._half_size[index]
                assert half is not None  # a foot with sole contacts has a sole
                chosen = np.flatnonzero(on_sole & (foot == index))
                at = substep_of[chosen]
                rotation = pose[at, index, 3:].reshape(-1, 3, 3)
                local = np.einsum("nji,nj->ni", rotation, points[chosen] - pose[at, index, :3])
                fore = spec.sole_forward_sign * local[:, spec.sole_forward_axis] / half[spec.sole_forward_axis]
                left = spec.sole_lateral_sign * local[:, spec.sole_lateral_axis] / half[spec.sole_lateral_axis]
                outer = -left if spec.side == "R" else left
                weight = normal[chosen]
                step_of = at // substeps
                weight_sum = np.bincount(step_of, weights=weight, minlength=steps)
                has_load = weight_sum > 1e-9
                for column, coordinate in enumerate((fore, outer)):
                    weighted = np.bincount(step_of, weights=weight * coordinate, minlength=steps)
                    cop[has_load, index, column] = weighted[has_load] / weight_sum[has_load]
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

        return EpisodeTrace(
            species=morphology.species,
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
        )

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
