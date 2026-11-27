"""Opt-in per-episode physics randomisation (domain randomisation) of the shared env.

Off by default for every model (design record
``docs/TRICERATOPS_ROBOT_PORT_2026_11.md`` §6).  Six ``[env]`` keys, each
``None`` (off, the default) or a range ``(low, high)``, finite with
``low <= high``; ``low > 0`` for every key but the payload (friction
included), whose ``low >= 0``.  A species takes them only by listing them in
its constructor (a species that does not list them raises ``TypeError`` if a
TOML sets them):

* ``randomize_friction`` -- an absolute sliding coefficient, one draw per
  episode, written to ``geom_friction[:, 0]`` of every static floor geom and
  of the geoms ``BaseDinoEnv._randomized_friction_geoms()`` returns (every
  colliding animal geom unless the species narrows it); the torsional and
  rolling columns keep their authored values.
* ``randomize_body_mass_scale`` -- a factor per animal body (the free root's
  subtree): its mass and inertia by one factor.  The world, a mocap target and
  every other prop keep theirs.
* ``randomize_actuator_kp_scale`` -- a factor per position servo (FIXED gain,
  AFFINE bias, ``biasprm[1] == -gainprm[0]``): ``gainprm[0]`` and
  ``biasprm[1]`` together, the damping ``biasprm[2]`` kept.
* ``randomize_actuator_force_limit_scale`` -- a factor per force-limited
  actuator, on both ends of its ``forcerange``.
* ``randomize_total_mass_scale`` -- one factor on every animal body's mass and
  inertia, after the per-body factors; in a trainer's rollout env only
  (``training_rollouts``), 1 elsewhere.
* ``randomize_payload_kg`` -- kilograms added to the payload body (the root
  if it has mass, else the heaviest body welded to it, as the push target is
  chosen), its inertia scaled by ``(m + p) / m``; in a trainer's rollout env
  only, 0 kg elsewhere.  The rule spreads the payload like the body's own
  mass, so it assumes a payload body heavy against the payload (the 60 cm
  Triceratops robot's 6 kg chassis takes up to 1 kg, x1.17); on a light body
  the factor is unbounded (a 1 kg payload on a 0.5 kg body triples its
  inertia), and only a massless one is refused.

Refused at construction, against the constructor's model, and again when a
model's record is taken (its first episode, so after any edit the species
makes following ``super().__init__()``), each with a ``ValueError`` naming
the key: friction on a model with explicit contact pairs (``npair > 0``,
whose friction is the pair's own), with no static floor geom, or with a
``_randomized_friction_geoms()`` that returns no geom, anything but a 1-D
array of integer geom ids (a boolean mask is refused), or a geom that is not
a colliding animal geom; the kp key on a model with no position servo; the
force key on a model with no force-limited actuator; a payload on a body with
no mass.  A behaviour pool model of another ``nbody`` or ``nu`` is refused
at its first episode (the draw block's layout is the constructor model's).
The construction check leaves the env's geom caches as it found them, so a
collision-mask edit made after ``super().__init__()`` reaches the settle.

**Each episode** (:meth:`EpisodePhysicsRandomizer.begin_episode`, from
``BaseDinoEnv._begin_episode``, before the keyframe is loaded): every model
the env changed is restored to its nominal values; the stream
(``episode_start.feature_stream``, :data:`~environments.shared.episode_start.EPISODE_PHYSICS_STREAM_KEY`)
is rebuilt on a seeded reset and on the first, and continued across unseeded
ones; one block of ``3 + nbody + 2 x nu`` uniforms is drawn whatever is
enabled (``u[0]`` friction, then one per body, one per actuator for kp, one per
actuator for force, ``u[-2]`` the total factor, ``u[-1]`` the payload, each
``low + (high - low) x u``), so enabling one key never moves another's draw,
and a seed's per-body factors are the same in training and evaluation; the
enabled quantities are written from the nominal copies in the order friction,
per-body, total, payload, kp, force.  When a mass was written,
``mj_setConst`` recomputes the derived constants on the record's own scratch
``MjData``, never the live one (on live data it moves ``qpos`` to
``qpos_spring``), and ``stat.center``, ``stat.extent`` and ``stat.meansize``
go back to nominal: setConst recomputes them from geometry unlike the
compiler, and nothing randomised moves geometry.  (The scratch data is
defensive: the keyframe load that follows in ``reset`` overwrites the live
data anyway.)  A raise while the episode's values are written restores the
model's nominal values before it propagates, so no model is left randomised
but flagged nominal.

**Exact restore.** Each model's record keeps exact copies of
:data:`RANDOMIZED_FIELDS`, :data:`SETCONST_FIELDS` and :data:`STAT_FIELDS`, and
restoring copies them back (``np.copyto``), never another setConst (which is
not a no-op on every unmodified model).  A model's record is taken the first
time an episode starts on it: the constructor's model and a behaviour env's
pool models alike, after any edit the species makes following
``super().__init__()``.  No managed field may be written after that while
randomisation is on: the next restore would undo it.

The plant identity is the nominal plant: the physics digest covers every
randomised field, so after a randomised reset ``validate_environment_plant``
and ``validate_compiled_plant`` refuse the live model (fail-closed), and pass
inside ``with env.nominal_physics():``, in which no episode may begin and the
randomisation may not be reconfigured: the block's exit writes back the
values it found, over the new episode's or the new configuration's nominal
plant.  Each raises ``RuntimeError``
(:meth:`EpisodePhysicsRandomizer.refuse_inside_nominal`) before anything
moves: a reset is refused by ``BaseDinoEnv._check_episode_may_begin``, first
in ``EpisodeStartHook.reset`` (before gymnasium reseeds ``np_random``) and in
the behaviour mixin's ``reset`` (before it moves its episode counters or
selects a pool model), so a refused reset leaves the env, its generators and
the episode's factors as they were.  Not fingerprinted:
the task fingerprint records the keys in its ``env`` section and, while any
is on, :data:`EPISODE_PHYSICS_IMPLEMENTATION` in its ``episode_physics``
section.  The symmetry module's state mirror refuses an env holding a
randomised plant: it mirrors the state, not the plant.
Numpy at top level; MuJoCo is imported where it is called.
"""

from __future__ import annotations

import contextlib
import math
from collections.abc import Iterator
from dataclasses import dataclass, fields
from numbers import Real
from typing import TYPE_CHECKING, Any

import numpy as np

from .episode_start import EPISODE_PHYSICS_STREAM_KEY, feature_stream

if TYPE_CHECKING:
    import mujoco

#: The ``[env]`` keys of the randomisation, in signature order.
EPISODE_PHYSICS_ENV_KEYS = (
    "randomize_friction",
    "randomize_body_mass_scale",
    "randomize_actuator_kp_scale",
    "randomize_actuator_force_limit_scale",
    "randomize_total_mass_scale",
    "randomize_payload_kg",
)
#: The keys applied only in a trainer's rollout env (``training_rollouts``).
TRAINING_ONLY_ENV_KEYS = ("randomize_total_mass_scale", "randomize_payload_kg")
#: Names what the keys mean; the task fingerprint's ``episode_physics`` section
#: records it.  Bump it on any change of meaning, not merely of shape, as
#: ``task_fingerprint.SCHEDULE_IMPLEMENTATION`` is bumped.
EPISODE_PHYSICS_IMPLEMENTATION = "mesozoic.episode-physics-randomization/v1"

#: The model fields a randomised quantity writes.
RANDOMIZED_FIELDS = (
    "geom_friction",
    "body_mass",
    "body_inertia",
    "actuator_gainprm",
    "actuator_biasprm",
    "actuator_forcerange",
)
#: The model fields ``mj_setConst`` writes (MuJoCo 3.10.0; the restore test
#: compares every model array with a randomisation-off env's, so a field
#: missing here fails by name).
SETCONST_FIELDS = (
    "body_subtreemass",
    "body_invweight0",
    "dof_invweight0",
    "dof_M0",
    "tendon_invweight0",
    "tendon_length0",
    "actuator_acc0",
    "actuator_length0",
    "cam_pos0",
    "cam_poscom0",
    "cam_mat0",
    "light_pos0",
    "light_poscom0",
    "light_dir0",
)
#: ``model.stat``'s fields, all of which ``mj_setConst`` rewrites.
STAT_FIELDS = ("meaninertia", "meanmass", "meansize", "extent", "center")
#: The statistics setConst recomputes from geometry differently from the
#: compiler; put back to nominal after it, since nothing randomised moves geometry.
GEOMETRIC_STAT_FIELDS = ("meansize", "extent", "center")


@dataclass(frozen=True)
class EpisodePhysicsConfig:
    """A validated randomisation: each field ``None`` (off) or a range ``(low, high)``, named by its key."""

    randomize_friction: tuple[float, float] | None = None
    randomize_body_mass_scale: tuple[float, float] | None = None
    randomize_actuator_kp_scale: tuple[float, float] | None = None
    randomize_actuator_force_limit_scale: tuple[float, float] | None = None
    randomize_total_mass_scale: tuple[float, float] | None = None
    randomize_payload_kg: tuple[float, float] | None = None


def _validated_range(name: str, value: Any) -> tuple[float, float] | None:
    if value is None:
        return None
    if isinstance(value, (str, bytes)) or not hasattr(value, "__len__") or len(value) != 2:
        raise ValueError(f"{name} must be None or a pair (low, high), got {value!r}")
    if any(isinstance(end, bool) or not isinstance(end, Real) for end in value):
        raise ValueError(f"{name} must be a pair of numbers (low, high), got {value!r}")
    low, high = float(value[0]), float(value[1])
    payload = name == "randomize_payload_kg"
    bound = "0 <= low <= high" if payload else "0 < low <= high"
    if not (math.isfinite(low) and math.isfinite(high)) or high < low or (low < 0.0 if payload else low <= 0.0):
        raise ValueError(f"{name} must be finite with {bound}, got {value!r}")
    return (low, high)


def validate_episode_physics(**ranges: Any) -> EpisodePhysicsConfig | None:
    """The validated randomisation, or ``None`` while every key is off.

    Takes the six keys of :data:`EPISODE_PHYSICS_ENV_KEYS` by name (an unknown
    one raises ``TypeError``).  Refuses a range that is not a pair of numbers
    (a string, a bool, three values), a non-finite end, ``low > high``, and
    ``low <= 0`` for every key but the payload, whose ``low`` may be 0.
    """
    if unknown := sorted(set(ranges) - set(EPISODE_PHYSICS_ENV_KEYS)):
        raise TypeError(f"unknown episode physics keys: {', '.join(unknown)}")
    config = EpisodePhysicsConfig(
        **{name: _validated_range(name, ranges.get(name)) for name in EPISODE_PHYSICS_ENV_KEYS}
    )
    if all(getattr(config, field.name) is None for field in fields(config)):
        return None
    return config


def animal_bodies(model: "mujoco.MjModel") -> np.ndarray:
    """The free-joint root body and its descendants (the animal): never the world, a mocap target or a prop."""
    import mujoco

    free = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
    if not free.size:
        raise ValueError("physics randomisation needs a model with a free-joint root body")
    root = int(model.jnt_bodyid[free[0]])
    bodies = {root}
    for body in range(root + 1, model.nbody):
        if int(model.body_parentid[body]) in bodies:
            bodies.add(body)
    return np.asarray(sorted(bodies), dtype=np.int64)


def payload_body(model: "mujoco.MjModel") -> int:
    """The body a payload is added to: the root if it has mass, else the heaviest body welded to it.

    The rule ``perturbation.derive_push_parameters`` targets its push with; a
    massless root with no massive welded body is returned as it is (and a
    payload on it is refused).
    """
    root = int(animal_bodies(model)[0])
    if float(model.body_mass[root]) > 0.0:
        return root
    rigid = np.flatnonzero((model.body_weldid == root) & (model.body_mass > 0.0))
    return int(rigid[np.argmax(model.body_mass[rigid])]) if rigid.size else root


def position_servos(model: "mujoco.MjModel") -> np.ndarray:
    """The actuators whose kp the kp key scales: FIXED gain, AFFINE bias and ``biasprm[1] == -gainprm[0]``."""
    import mujoco

    return np.flatnonzero(
        (model.actuator_gaintype == mujoco.mjtGain.mjGAIN_FIXED)
        & (model.actuator_biastype == mujoco.mjtBias.mjBIAS_AFFINE)
        & (model.actuator_biasprm[:, 1] == -model.actuator_gainprm[:, 0])
    )


def force_limited_actuators(model: "mujoco.MjModel") -> np.ndarray:
    """The actuators whose ``forcerange`` the force-limit key scales."""
    return np.flatnonzero(model.actuator_forcelimited != 0)


def colliding_animal_geoms(model: "mujoco.MjModel") -> np.ndarray:
    """The animal's colliding geoms: ``BaseDinoEnv._root_subtree_geoms()``'s definition, read without its cache."""
    colliding = (model.geom_contype != 0) | (model.geom_conaffinity != 0)
    return np.flatnonzero(np.isin(model.geom_bodyid, animal_bodies(model)) & colliding)


def _name(model: "mujoco.MjModel", kind: Any, index: int) -> str:
    import mujoco

    return mujoco.mj_id2name(model, kind, int(index)) or f"#{int(index)}"


@dataclass(frozen=True)
class _Targets:
    """What a randomisation writes on one model: its animal bodies, payload body, servos, force limits and friction geoms."""

    bodies: np.ndarray
    payload_body: int
    servos: np.ndarray
    force_limited: np.ndarray
    #: The static floor geoms and the hook's, sorted; empty while friction is off.
    friction_geoms: np.ndarray


def _hook_geoms(env: Any, model: "mujoco.MjModel") -> np.ndarray:
    """``env._randomized_friction_geoms()`` as geom ids, refused unless each is a colliding animal geom of *model*."""
    import mujoco

    hook = "randomize_friction: _randomized_friction_geoms()"
    geoms = np.asarray(env._randomized_friction_geoms())
    if not geoms.size:
        raise ValueError(f"{hook} returned no geom")
    if geoms.ndim != 1 or geoms.dtype.kind not in "iu":
        # A boolean mask would index as geoms 0 and 1.
        raise ValueError(
            f"{hook} must return a 1-D array of integer geom ids, got dtype {geoms.dtype} and shape {geoms.shape}"
        )
    geoms = geoms.astype(np.int64)
    if int(geoms.min()) < 0 or int(geoms.max()) >= int(model.ngeom):
        raise ValueError(f"{hook} returned geom ids outside [0, ngeom = {int(model.ngeom)}): {geoms.tolist()}")
    if (stray := np.setdiff1d(geoms, colliding_animal_geoms(model))).size:
        names = [_name(model, mujoco.mjtObj.mjOBJ_GEOM, geom) for geom in stray]
        raise ValueError(f"{hook} returned geoms that are not colliding animal geoms: {names}")
    return geoms


def _targets(env: Any, model: "mujoco.MjModel", config: EpisodePhysicsConfig) -> _Targets:
    """What *config* writes on *model* (``env``'s live model), refusing what it cannot apply there (``ValueError``)."""
    import mujoco

    bodies, body = animal_bodies(model), payload_body(model)
    servos, limited = position_servos(model), force_limited_actuators(model)
    friction_geoms = np.empty(0, dtype=np.int64)
    if config.randomize_friction is not None:
        if int(model.npair) > 0:
            raise ValueError(
                f"randomize_friction: the model has {int(model.npair)} explicit contact pair(s), whose friction "
                "is the pair's own and not the geoms'; not supported"
            )
        floor = np.asarray(env._static_floor_geoms(), dtype=np.int64)
        if not floor.size:
            raise ValueError("randomize_friction needs a static floor geom (a world-body plane or heightfield)")
        friction_geoms = np.union1d(floor, _hook_geoms(env, model))
    if config.randomize_actuator_kp_scale is not None and not servos.size:
        raise ValueError(
            "randomize_actuator_kp_scale: the model has no position servo "
            "(FIXED gain, AFFINE bias, biasprm[1] == -gainprm[0])"
        )
    if config.randomize_actuator_force_limit_scale is not None and not limited.size:
        raise ValueError("randomize_actuator_force_limit_scale: the model has no force-limited actuator")
    if config.randomize_payload_kg is not None and not float(model.body_mass[body]) > 0.0:
        raise ValueError(
            f"randomize_payload_kg: the payload body {_name(model, mujoco.mjtObj.mjOBJ_BODY, body)!r} "
            "has no mass (a massless root with no massive welded body), so (m + p) / m is undefined"
        )
    return _Targets(bodies, body, servos, limited, friction_geoms)


class _ModelRecord:
    """One model's exact nominal copies, its targets, a scratch ``MjData``, and whether it holds an episode's values."""

    def __init__(self, model: "mujoco.MjModel", targets: _Targets):
        import mujoco

        self.model = model
        self.targets = targets
        self.nominal = {name: getattr(model, name).copy() for name in RANDOMIZED_FIELDS + SETCONST_FIELDS}
        self.nominal_stat = {name: np.array(getattr(model.stat, name), dtype=np.float64) for name in STAT_FIELDS}
        self.scratch = mujoco.MjData(model)
        self.randomized = False

    def capture(self) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """Exact copies of the live values of every managed field and statistic."""
        model = self.model
        return (
            {name: getattr(model, name).copy() for name in RANDOMIZED_FIELDS + SETCONST_FIELDS},
            {name: np.array(getattr(model.stat, name), dtype=np.float64) for name in STAT_FIELDS},
        )

    def write_stat(self, stat: dict[str, np.ndarray], names: tuple[str, ...]) -> None:
        for name in names:
            value = stat[name]
            if value.shape:
                np.copyto(getattr(self.model.stat, name), value)
            else:
                setattr(self.model.stat, name, float(value))

    def write(self, arrays: dict[str, np.ndarray], stat: dict[str, np.ndarray]) -> None:
        for name, value in arrays.items():
            np.copyto(getattr(self.model, name), value)
        self.write_stat(stat, STAT_FIELDS)

    def restore(self) -> None:
        self.write(self.nominal, self.nominal_stat)
        self.randomized = False


class EpisodePhysicsRandomizer:
    """The per-episode physics randomiser of one env, built only while a key is on (``BaseDinoEnv._episode_physics``).

    Construction checks *config* against ``env.model`` and refuses what it
    cannot apply; it changes nothing, and keeps nothing of the model but its
    ``nbody`` and ``nu`` (the draw block's layout).  What a model's episodes
    write is derived again, with the same refusals, when its record is taken
    at its first episode.  The model is first changed by :meth:`begin_episode`.
    """

    def __init__(self, env: Any, config: EpisodePhysicsConfig):
        model = env.model
        self.config = config
        self.nbody, self.nu = int(model.nbody), int(model.nu)
        # The friction check reads the env's geom helpers, which cache; put the caches back as they were, so a
        # collision-mask edit made after super().__init__() still reaches the settle at the first reset.
        caches = (env._root_subtree_geom_ids, env._static_floor_geom_ids)
        try:
            _targets(env, model, config)
        finally:
            env._root_subtree_geom_ids, env._static_floor_geom_ids = caches
        self.rng: np.random.Generator | None = None
        self.factors: dict[str, Any] | None = None
        self._records: list[_ModelRecord] = []
        self._nominal_depth = 0

    @property
    def draw_count(self) -> int:
        """The uniforms one episode draws, whatever is enabled: ``3 + nbody + 2 x nu``."""
        return 3 + self.nbody + 2 * self.nu

    def _find(self, model: Any) -> _ModelRecord | None:
        for record in self._records:
            if record.model is model:
                return record
        return None

    def _record(self, env: Any) -> _ModelRecord:
        """The live model's record, taken at the first episode on it (it has never been changed then).

        Its targets are derived, and the construction refusals run again, on
        the model as it is then, so a species edit made after
        ``super().__init__()`` decides them.
        """
        model = env.model
        record = self._find(model)
        if record is None:
            if (int(model.nbody), int(model.nu)) != (self.nbody, self.nu):
                raise ValueError(
                    f"physics randomisation: a pool model changed nbody or nu ({int(model.nbody)}, {int(model.nu)}"
                    f" against the constructor model's {self.nbody}, {self.nu})"
                )
            record = _ModelRecord(model, _targets(env, model, self.config))
            self._records.append(record)
        return record

    def holds_episode_values(self, model: Any) -> bool:
        """Whether *model* holds an episode's randomised values now (inside a ``nominal()`` block it does not)."""
        record = self._find(model)
        return record is not None and record.randomized

    def restore_all(self) -> None:
        """Every model this env randomised back to its nominal values."""
        for record in self._records:
            if record.randomized:
                record.restore()

    def refuse_inside_nominal(self, what: str) -> None:
        """Raise ``RuntimeError`` naming *what* while a :meth:`nominal` block is open; nothing otherwise.

        The block's exit writes back the values it found, so a reset inside it
        would end with the previous episode's values over the new episode's,
        and a reconfiguration with them over the plant the new randomiser
        records as nominal.  The env calls it before anything of the reset or
        the reconfiguration moves (``BaseDinoEnv._check_episode_may_begin``,
        ``BaseDinoEnv._configure_episode_physics``); :meth:`begin_episode`
        calls it again.
        """
        if self._nominal_depth:
            raise RuntimeError(
                f"physics randomisation: {what} inside a nominal_physics() block is refused, since the block's exit "
                "would write the previous episode's values back over it; leave the block first"
            )

    def begin_episode(self, env: Any, seed: int | None) -> None:
        """Restore, draw this episode's block of uniforms, and write the enabled quantities into ``env.model``.

        Raises ``RuntimeError`` inside a :meth:`nominal` block (:meth:`refuse_inside_nominal`),
        whose exit would write the previous episode's values back over this
        one's.  A reset is refused before it reaches here, so this check only
        guards a direct call.
        """
        import mujoco

        self.refuse_inside_nominal("a reset")
        self.restore_all()
        self.factors = None
        if seed is not None or self.rng is None:
            self.rng = feature_stream(env, seed, EPISODE_PHYSICS_STREAM_KEY)
        record = self._record(env)
        model, nominal, config, targets = record.model, record.nominal, self.config, record.targets
        training = bool(env.training_rollouts)
        u = self.rng.random(self.draw_count)
        nbody, nu = self.nbody, self.nu
        u_friction, u_mass = u[0], u[1 : 1 + nbody]
        u_kp, u_force = u[1 + nbody : 1 + nbody + nu], u[1 + nbody + nu : 1 + nbody + 2 * nu]
        u_total, u_payload = u[-2], u[-1]

        def span(bounds: tuple[float, float], x: Any) -> Any:
            return bounds[0] + (bounds[1] - bounds[0]) * x

        factors: dict[str, Any] = {}
        written = mass_written = False
        # A raise mid-write restores the nominal values before it propagates, so no model is ever left holding
        # randomised values while flagged nominal (which every restore and nominal() block would skip).
        try:
            if config.randomize_friction is not None:
                mu = float(span(config.randomize_friction, u_friction))
                model.geom_friction[targets.friction_geoms, 0] = mu
                factors["friction"] = mu
                written = True
            bodies = targets.bodies
            if config.randomize_body_mass_scale is not None:
                scale = span(config.randomize_body_mass_scale, u_mass[bodies])
                model.body_mass[bodies] = nominal["body_mass"][bodies] * scale
                model.body_inertia[bodies] = nominal["body_inertia"][bodies] * scale[:, None]
                factors["body_mass_scale"] = scale.copy()
                written = mass_written = True
            if config.randomize_total_mass_scale is not None:
                whole = float(span(config.randomize_total_mass_scale, u_total)) if training else 1.0
                if training:
                    model.body_mass[bodies] *= whole
                    model.body_inertia[bodies] *= whole
                    written = mass_written = True
                factors["total_mass_scale"] = whole
            if config.randomize_payload_kg is not None:
                payload = float(span(config.randomize_payload_kg, u_payload)) if training else 0.0
                if training:
                    body = targets.payload_body
                    mass = float(model.body_mass[body])
                    model.body_mass[body] = mass + payload
                    model.body_inertia[body] *= (mass + payload) / mass
                    written = mass_written = True
                factors["payload_kg"] = payload
            servos, limited = targets.servos, targets.force_limited
            if config.randomize_actuator_kp_scale is not None:
                kp = span(config.randomize_actuator_kp_scale, u_kp[servos])
                model.actuator_gainprm[servos, 0] = nominal["actuator_gainprm"][servos, 0] * kp
                model.actuator_biasprm[servos, 1] = nominal["actuator_biasprm"][servos, 1] * kp
                factors["actuator_kp_scale"] = kp.copy()
                written = True
            if config.randomize_actuator_force_limit_scale is not None:
                limit = span(config.randomize_actuator_force_limit_scale, u_force[limited])
                model.actuator_forcerange[limited] = nominal["actuator_forcerange"][limited] * limit[:, None]
                factors["actuator_force_limit_scale"] = limit.copy()
                written = True
            if mass_written:
                # Never the live data: setConst would move its qpos to qpos_spring.
                mujoco.mj_setConst(model, record.scratch)
                record.write_stat(record.nominal_stat, GEOMETRIC_STAT_FIELDS)
        except BaseException:
            record.restore()
            raise
        record.randomized = written
        self.factors = factors

    @contextlib.contextmanager
    def nominal(self, env: Any) -> Iterator[None]:
        """A block in which ``env.model`` holds its nominal values; the episode's values are written back after it.

        The live values are captured on entry and nothing else is kept, so the
        exit writes back exactly what the block found, whether the block
        returns or raises; blocks nest.  No episode may begin and the
        randomisation may not be reconfigured inside one: each raises
        ``RuntimeError`` (:meth:`refuse_inside_nominal`) while a block is open.
        """
        record = self._find(env.model)
        self._nominal_depth += 1
        try:
            if record is None or not record.randomized:
                yield
                return
            live = record.capture()
            record.restore()
            try:
                yield
            finally:
                record.write(*live)
                record.randomized = True
        finally:
            self._nominal_depth -= 1
