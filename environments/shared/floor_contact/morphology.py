"""Which geoms are a foot's, which one is its sole, and how heavy the animal is -- read off a live env.

:meth:`Morphology.from_env` turns an env into the lookup tables the
:class:`environments.shared.gait.recorder.SubstepContactRecorder` classifies
contacts with.  It reads the env's own declarations and never writes the env:
the foot sensors (``_foot_sensor_groups``), the floor
(``_static_floor_geoms``), the animal (``_root_subtree_geoms``) and the home
keyframe, the last on a scratch ``MjData`` so ``env.data`` is untouched.

It lives outside the gait package (scratch, the stance4 copband design,
never merged) so that the T. rex's opt-in CoP band, which reduces each
step's contacts with the recorder's own code (:mod:`.step_cop`), can build
it without a species env importing ``gait``; ``gait/morphology.py``
re-exports it, beside the measurement definition.

The generic rule (no species knowledge)
---------------------------------------
* **Feet:** one per ``_foot_sensor_groups`` entry, in that order -- 2 on the
  bipeds, 4 on the quadrupeds.  The label is the first sensor's name prefix
  (``r``/``l``; ``fr``/``fl``/``rr``/``rl``; the robot's ``r_foot_touch``
  gives ``r``).
* **Leg (the whole limb):** the subtree of the highest ancestor of the
  foot's touch-site bodies that holds no other foot's site and is not the
  free-joint root -- ``gait_probe.RobotDescription``'s rule, which gives the
  ``*_thigh`` subtrees on five species and ``hip_roll_assembly(_2)`` on the
  robot.  "Down" is judged on the whole limb's floor force, as the GAIT plan
  calibrated it, so a heel or a shin on the floor still bears load.
* **Support geoms:** the colliding geoms on the touch-site bodies and their
  descendants.
* **Sole:** the support geom of type box (else ellipsoid) with the largest
  footprint.  Its frame is chosen at the home keyframe: the normal is the
  local axis (with sign) closest to world -z, forward the remaining axis
  closest to the root body's +x, and the third axis is signed to point to the
  animal's left.  All five species with a sole give normal ``-z``, forward
  ``+x``.
* **Body weight:** ``body_subtreemass[root] * |g_z|`` -- the load the floor
  carries at rest -- never ``mj_getTotalmass``, which counts props and prey
  (STAGE1_SPLIT_PLAN's body-weight rule).

The registry (explicit for all six models)
------------------------------------------
The generic rule is wrong for four of the six, so every embodiment class
names its support set, sole and reference site explicitly in its
``FOOT_SUPPORT`` class attribute, where the reason for each difference from
the generic rule sits beside it, and :func:`support_entry` reads it (decision
D-D33: a new embodiment needs no edit here).  ``tests/test_gait_morphology.py``
pins each difference from the generic answer.  The registry is MEASUREMENT
data, not plant data: it is recorded in the gait package's
:func:`~environments.shared.gait.constants.measurement_manifest`, not in
``plant_versions.toml``, and editing an entry after a stage adopts
``stance_quality/v2`` needs a new
:data:`~environments.shared.gait.constants.MEASUREMENT_VERSION`.  A
model with no entry falls back to the generic rule, and the manifest records
``registry: None``.

The whole-foot centre of pressure
---------------------------------
An entry may also declare the foot's ``ankle`` (a joint name template) and
its support span, ``heel_line`` and ``toe_line``: metres from the ankle's
hinge axis along the sole's forward axis.  The recorder then measures the
WHOLE-FOOT fore-aft centre of pressure (every floor contact of the limb,
sole and digits alike) from that axis, and the span normalises it.  The
ankle must be a hinge of the sole's own body, so its axis is fixed in the
sole frame and the span is a property of the foot, not of a pose; its axis
must be the sole's lateral axis (to :data:`_ANKLE_AXIS_OFF_LATERAL` of its
length), because only about a lateral axis is a contact's distance along
the forward axis its lever arm: for a unit axis ``a`` in the sole frame the
vertical load at ``d`` on a level sole has the lever ``a_f d_l - a_l d_f``,
so a forward component would weigh in lateral offsets and a normal one
shrink the arm.  The three keys are declared together or not at all.  A
foot whose entry declares none has no whole-foot centre of pressure
(``EpisodeTrace.foot_cop`` is NaN for it).
Only the T. rex declares one: its pad sits on the body its ankle hinges.
Every other sole sits on a hinge below the ankle (the anatomical
compsognathus's, the dibothrosuchus's and the brachiosaurus's on
``{s}_toe``, the compsognathus robot's on its ankle roll; the velociraptor
has none), so which axis counts as their ankle is left to a later decision.

Digits with an MTP joint (generic rule): a registered support geom other
than the sole is a hinged digit when it is a capsule on a CHILD body of the
sole's body that is joined to it by exactly one hinge, its MTP joint.  The
recorder reads where along such a digit its floor load sits (0 at the MTP
axis, 1 at the capsule's tip); ``support_geom_mtp`` names the joint, -1 for
every other support geom.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

import mujoco
import numpy as np

from environments.shared.model_id import ModelId


@dataclass(frozen=True)
class FootRegistryEntry:
    """One model's foot geoms, as name templates.

    ``{s}`` is the foot label (``r``, ``l``, ``fr``, ``fl``, ``rr``, ``rl``)
    and ``{side}`` is ``right`` / ``left`` (the robot's geom names spell the
    side out).

    * ``support`` -- the geoms a flat, loaded foot stands on.  Each must be a
      colliding geom of that foot's leg.
    * ``sole`` -- the geom whose frame is the sole (a box or an ellipsoid),
      one of ``support``; ``None`` for a foot with no pad to be flat on.
    * ``foot_site`` -- the site that is the foot's position (stance width,
      foot shift).  ``None`` means the first touch sensor's site; every entry
      pins it, so a re-ordered sensor group cannot move the reference point.
    """

    support: tuple[str, ...]
    sole: str | None
    foot_site: str | None = None
    #: The hinge joint the whole-foot centre of pressure is measured from, a
    #: template like the geom names; ``None`` for a foot that declares no
    #: support span.  A hinge of the sole's own body (module docstring).
    ankle: str | None = None
    #: The declared support span along the sole's forward axis, in m from the
    #: ankle's axis: the heel line and the toe line the whole-foot centre of
    #: pressure is normalised on (-1 on the heel line, +1 on the toe line).
    heel_line: float | None = None
    toe_line: float | None = None

    def __post_init__(self) -> None:
        declared = [value is not None for value in (self.ankle, self.heel_line, self.toe_line)]
        if any(declared) and not all(declared):
            raise MorphologyError(
                "a FOOT_SUPPORT entry declares its ankle, heel_line and toe_line together or not at all, not "
                f"ankle={self.ankle!r}, heel_line={self.heel_line!r}, toe_line={self.toe_line!r}"
            )
        if self.heel_line is not None and self.toe_line is not None:
            for name, value in (("heel_line", self.heel_line), ("toe_line", self.toe_line)):
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise MorphologyError(f"FOOT_SUPPORT {name} must be a finite number of metres, not {value!r}")
            if not self.heel_line < self.toe_line:
                raise MorphologyError(
                    f"FOOT_SUPPORT heel_line {self.heel_line!r} must lie behind toe_line {self.toe_line!r}"
                )

    @property
    def declares_span(self) -> bool:
        """Whether the entry declares an ankle and a support span (the whole-foot centre of pressure is measured)."""
        return self.ankle is not None

    def as_manifest(self) -> dict[str, Any]:
        """The entry as a measurement manifest records it (``registry``): the templates and the span, JSON-ready."""
        return {
            "support": list(self.support),
            "sole": self.sole,
            "foot_site": self.foot_site,
            "ankle": self.ankle,
            "heel_line": None if self.heel_line is None else float(self.heel_line),
            "toe_line": None if self.toe_line is None else float(self.toe_line),
        }


def support_entry(model_id: ModelId) -> FootRegistryEntry | None:
    """The model's registry entry: its env class's ``FOOT_SUPPORT``, or ``None`` for the generic rule.

    Read off the class the manifest names (``species_registry.env_class``),
    so no model instance is built; a class without ``FOOT_SUPPORT`` gets the
    generic rule.  The import loads ``species_registry`` and, through it,
    ``train_base`` and the curriculum (which builds on ``gait``); an env that
    knows its own class passes :func:`entry_from_support` of its
    ``FOOT_SUPPORT`` to :meth:`Morphology.from_env` instead, as the T. rex's
    CoP band does.
    """
    from ..species_registry import env_class

    return entry_from_support(getattr(env_class(model_id), "FOOT_SUPPORT", None))


def entry_from_support(declared: Mapping[str, Any] | None) -> FootRegistryEntry | None:
    """The registry entry a ``FOOT_SUPPORT`` class attribute declares, or ``None`` (the generic rule) for none."""
    if declared is None:
        return None
    return FootRegistryEntry(
        support=tuple(declared["support"]),
        sole=declared["sole"],
        foot_site=declared.get("foot_site"),
        ankle=declared.get("ankle"),
        heel_line=declared.get("heel_line"),
        toe_line=declared.get("toe_line"),
    )


# The foot labels the first sensor's name prefix may give; any other prefix
# becomes ``foot<i>``.  The last letter is the side.
_FOOT_LABELS = ("r", "l", "fr", "fl", "rr", "rl")
#: How far a declared ankle's axis may lie off the sole's lateral axis, as the norm of its forward and
#: normal components over its length (about 6e-5 degrees): round-off, not a cant.
_ANKLE_AXIS_OFF_LATERAL = 1e-6
_PREFIX_ALIASES = {"right": "r", "left": "l"}
_SIDE_WORDS = {"R": "right", "L": "left"}


class MorphologyError(RuntimeError):
    """The env cannot be described: no foot sensors, a registry name the model lacks, a geom on two legs..."""


@dataclass(frozen=True, eq=False)
class FootSpec:
    """One foot, as model ids.  Axis indices are local axes of the sole geom (0, 1, 2 = x, y, z)."""

    label: str
    #: ``"R"`` / ``"L"``, or ``"?"`` for a label with no side letter (the
    #: roll and CoP "outer" sign then treats the foot as a left one).
    side: str
    sensor_names: tuple[str, ...]
    #: The reference site: the registry's ``foot_site``, else the first sensor's site.
    primary_site: int
    leg_bodies: frozenset[int]
    #: Colliding geoms of the whole limb.
    leg_geoms: tuple[int, ...]
    #: Registered sole / digit geoms, a subset of ``leg_geoms``.
    support_geoms: tuple[int, ...]
    #: The sole geom and its ``mjtGeom`` type; -1 for both when the foot has no sole.
    sole_geom: int
    sole_type: int
    #: The local axis (and sign) that points into the floor when the sole is flat.
    sole_normal_axis: int
    sole_normal_sign: float
    #: The remaining local axis closest to the root body's +x at the home keyframe.
    sole_forward_axis: int
    sole_forward_sign: float
    #: The third axis; ``sign * R[:, axis]`` points to the animal's left at home.
    sole_lateral_axis: int
    sole_lateral_sign: float
    #: The keyframe's own sole tilt (degrees from flat), NaN without a sole.
    home_sole_tilt_deg: float
    #: The registry's ankle hinge (-1: none declared) and its axis's coordinate along the sole's
    #: signed forward axis, in the sole frame (m; constant, the ankle being a hinge of the sole's body).
    ankle_joint: int = -1
    ankle_forward_m: float = float("nan")
    #: The declared support span, m from the ankle axis along the sole's forward axis; NaN without one.
    heel_line_m: float = float("nan")
    toe_line_m: float = float("nan")


@dataclass(frozen=True, eq=False)
class Morphology:
    """The feet, floor and animal of one env, and the per-geom lookups the recorder reduces with.

    ``geom_foot`` / ``geom_support`` / ``is_floor`` / ``is_animal`` are
    read-only arrays over every model geom: the leg index (-1: not a leg
    geom), the index into ``support_geom_ids`` (-1: not support), and the two
    masks.  ``support_geom_ids`` is flattened foot-major, and
    ``support_geom_foot`` gives each one's foot.  ``registry_entry`` is the
    :class:`FootRegistryEntry` the geoms came from, ``None`` under the
    generic rule.  Build one per env INSTANCE: ``frame_skip`` and ``dt`` are
    per instance (both Compsognathus models run ``frame_skip = 10``).
    """

    model_id: ModelId
    feet: tuple[FootSpec, ...]
    floor_geoms: tuple[int, ...]
    animal_geoms: tuple[int, ...]
    root_body: int
    body_weight_n: float
    frame_skip: int
    dt: float
    geom_foot: np.ndarray = field(repr=False)
    geom_support: np.ndarray = field(repr=False)
    is_floor: np.ndarray = field(repr=False)
    is_animal: np.ndarray = field(repr=False)
    support_geom_ids: tuple[int, ...] = ()
    support_geom_foot: tuple[int, ...] = ()
    registry_entry: FootRegistryEntry | None = None
    #: Per support geom (as ``support_geom_ids``): its name, and the MTP joint of a hinged digit (-1: not one).
    support_geom_names: tuple[str, ...] = ()
    support_geom_mtp: tuple[int, ...] = ()

    @property
    def n_feet(self) -> int:
        return len(self.feet)

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(foot.label for foot in self.feet)

    @property
    def is_quadruped(self) -> bool:
        return self.n_feet == 4

    def describe(self, model: Any) -> dict[str, Any]:
        """The morphology by NAME, for reports and the registry test (``model``: the one it was built from)."""

        def name(geom_id: int) -> str | None:
            return None if geom_id < 0 else str(model.geom(int(geom_id)).name)

        def axis(index: int, sign: float) -> str:
            return f"{'-' if sign < 0 else '+'}{'xyz'[index]}"

        return {
            **self.model_id.record(),
            "body_weight_n": self.body_weight_n,
            "frame_skip": self.frame_skip,
            "dt": self.dt,
            "root_body": str(model.body(self.root_body).name),
            "floor_geoms": [name(geom) for geom in self.floor_geoms],
            "uses_registry": self.registry_entry is not None,
            "feet": [
                {
                    "label": foot.label,
                    "side": foot.side,
                    "sensors": list(foot.sensor_names),
                    "reference_site": str(model.site(foot.primary_site).name),
                    "leg_geoms": [name(geom) for geom in foot.leg_geoms],
                    "support_geoms": [name(geom) for geom in foot.support_geoms],
                    "sole_geom": name(foot.sole_geom),
                    "sole_normal": axis(foot.sole_normal_axis, foot.sole_normal_sign) if foot.sole_geom >= 0 else None,
                    "sole_forward": axis(foot.sole_forward_axis, foot.sole_forward_sign)
                    if foot.sole_geom >= 0
                    else None,
                    "home_sole_tilt_deg": foot.home_sole_tilt_deg,
                    "ankle_joint": None if foot.ankle_joint < 0 else str(model.joint(foot.ankle_joint).name),
                    "support_span_m": None if foot.ankle_joint < 0 else [foot.heel_line_m, foot.toe_line_m],
                }
                for foot in self.feet
            ],
            "mtp_joints": {
                name: str(model.joint(joint).name)
                for name, joint in zip(self.support_geom_names, self.support_geom_mtp)
                if joint >= 0
            },
        }

    @classmethod
    def from_env(
        cls,
        env: Any,
        model_id: ModelId,
        *,
        registry: Mapping[ModelId, FootRegistryEntry | None] | None = None,
    ) -> "Morphology":
        """Describe ``env`` (wrappers are looked through) for ``model_id``.

        The registry entry is :func:`support_entry` of ``model_id``, unless
        ``registry`` (keyed by model) is given: then the model's entry there,
        and the generic rule when it has none -- ``registry={}`` forces the
        generic rule, what the registry test compares against.

        Raises :class:`MorphologyError` for: no ``_foot_sensor_groups`` (or
        an empty group), a group index that is not a sensor's, a foot sensor
        not on a site, two feet with one label, no free joint, a geom on two
        legs, a registry name missing from the model or not a colliding geom
        of that leg, a sole that is not one of the foot's support geoms, a
        reference site off the leg, a foot with no support geom, an invalid
        home keyframe, and no floor geom.
        """
        unwrapped = getattr(env, "unwrapped", env)
        model = unwrapped.model
        groups = tuple(tuple(int(index) for index in group) for group in getattr(unwrapped, "_foot_sensor_groups", ()))
        if not groups or any(not group for group in groups):
            raise MorphologyError(f"{type(unwrapped).__name__} declares no (or an empty) _foot_sensor_groups entry")
        free = [joint for joint in range(model.njnt) if int(model.jnt_type[joint]) == int(mujoco.mjtJoint.mjJNT_FREE)]
        if not free:
            raise MorphologyError("the model has no free joint, so it has no animal root")
        root = int(model.jnt_bodyid[free[0]])
        sensor_at_address = {int(model.sensor_adr[sensor]): sensor for sensor in range(model.nsensor)}

        def subtree(body: int) -> set[int]:
            # MuJoCo numbers bodies parent-first, so one forward sweep closes the subtree.
            bodies = {body}
            for child in range(body + 1, model.nbody):
                if int(model.body_parentid[child]) in bodies:
                    bodies.add(child)
            return bodies

        sensor_names: list[tuple[str, ...]] = []
        site_ids: list[list[int]] = []
        labels: list[str] = []
        for index, group in enumerate(groups):
            sensors = []
            for address in group:
                if address not in sensor_at_address:
                    raise MorphologyError(
                        f"foot sensor group {index} names sensordata[{address}], which starts no sensor"
                    )
                sensors.append(sensor_at_address[address])
            for sensor in sensors:
                if int(model.sensor_objtype[sensor]) != int(mujoco.mjtObj.mjOBJ_SITE):
                    raise MorphologyError(f"foot sensor {model.sensor(sensor).name} is not attached to a site")
            names = tuple(str(model.sensor(sensor).name) for sensor in sensors)
            prefix = names[0].split("_")[0].lower()
            prefix = _PREFIX_ALIASES.get(prefix, prefix)
            labels.append(prefix if prefix in _FOOT_LABELS else f"foot{index}")
            sensor_names.append(names)
            site_ids.append([int(model.sensor_objid[sensor]) for sensor in sensors])
        if len(set(labels)) != len(labels):
            raise MorphologyError(f"two feet share a label: {labels}")
        site_bodies = [{int(model.site_bodyid[site]) for site in sites} for sites in site_ids]
        entry = support_entry(model_id) if registry is None else registry.get(model_id)

        # The home keyframe, on scratch data: the sole axes are chosen there.
        scratch = mujoco.MjData(model)
        if model.nkey:
            keyframe = int(getattr(unwrapped, "_reset_keyframe_id", 0))
            if not 0 <= keyframe < model.nkey:
                raise MorphologyError(f"_reset_keyframe_id {keyframe} is not one of the model's {model.nkey} keyframes")
            mujoco.mj_resetDataKeyframe(model, scratch, keyframe)
        mujoco.mj_forward(model, scratch)
        root_frame = scratch.xmat[root].reshape(3, 3)
        root_forward, root_left = root_frame[:, 0], root_frame[:, 1]

        colliding = (model.geom_contype != 0) | (model.geom_conaffinity != 0)
        animal = tuple(int(geom) for geom in unwrapped._root_subtree_geoms())
        animal_set = set(animal)
        geom_foot = np.full(model.ngeom, -1, dtype=np.int64)
        feet = []
        for index, bodies in enumerate(site_bodies):
            label = labels[index]
            others = set().union(*(other for j, other in enumerate(site_bodies) if j != index))
            tops = set()
            for body in bodies:
                while True:
                    parent = int(model.body_parentid[body])
                    if parent in (0, root) or subtree(parent) & others:
                        break
                    body = parent
                tops.add(body)
            leg: set[int] = set()
            for body in tops:
                leg |= subtree(body)
            leg_geoms = tuple(
                geom for geom in range(model.ngeom) if int(model.geom_bodyid[geom]) in leg and colliding[geom]
            )
            for geom in leg_geoms:
                if geom_foot[geom] >= 0:
                    raise MorphologyError(f"geom {model.geom(geom).name} belongs to two legs")
                if geom not in animal_set:
                    raise MorphologyError(f"leg geom {model.geom(geom).name} is not in the free-joint subtree")
                geom_foot[geom] = index
            side = label[-1].upper() if label[-1] in "rl" else "?"
            ankle = -1
            if entry is not None:
                support, sole, primary_site = _registered_geoms(model, entry, label, side, model_id, leg_geoms, leg)
                ankle = _registered_ankle(model, entry, label, side, model_id, sole)
            else:
                support, sole = _generic_geoms(model, leg_geoms, bodies, subtree)
                primary_site = site_ids[index][0]
            if not support:
                raise MorphologyError(f"foot {label} has no support geom")
            feet.append(
                _foot_spec(
                    model,
                    scratch,
                    root_forward,
                    root_left,
                    label,
                    side,
                    sensor_names[index],
                    primary_site,
                    leg,
                    leg_geoms,
                    support,
                    sole,
                    ankle,
                    entry,
                )
            )

        floor = tuple(int(geom) for geom in unwrapped._static_floor_geoms())
        if not floor:
            raise MorphologyError("the model has no static floor geom (world-body plane or heightfield)")
        is_floor = np.zeros(model.ngeom, dtype=bool)
        is_floor[list(floor)] = True
        is_animal = np.zeros(model.ngeom, dtype=bool)
        is_animal[list(animal)] = True
        support_ids: list[int] = []
        support_foot: list[int] = []
        support_mtp: list[int] = []
        geom_support = np.full(model.ngeom, -1, dtype=np.int64)
        for index, foot in enumerate(feet):
            for geom in foot.support_geoms:
                geom_support[geom] = len(support_ids)
                support_ids.append(geom)
                support_foot.append(index)
                support_mtp.append(_digit_mtp_joint(model, geom, foot.sole_geom))
        for array in (geom_foot, geom_support, is_floor, is_animal):
            array.setflags(write=False)
        return cls(
            model_id=model_id,
            feet=tuple(feet),
            floor_geoms=floor,
            animal_geoms=animal,
            root_body=root,
            body_weight_n=float(model.body_subtreemass[root]) * abs(float(model.opt.gravity[2])),
            frame_skip=int(unwrapped.frame_skip),
            dt=float(unwrapped.dt),
            geom_foot=geom_foot,
            geom_support=geom_support,
            is_floor=is_floor,
            is_animal=is_animal,
            support_geom_ids=tuple(support_ids),
            support_geom_foot=tuple(support_foot),
            registry_entry=entry,
            support_geom_names=tuple(str(model.geom(int(geom)).name) for geom in support_ids),
            support_geom_mtp=tuple(support_mtp),
        )


def _registered_geoms(
    model: Any,
    entry: FootRegistryEntry,
    label: str,
    side: str,
    model_id: ModelId,
    leg_geoms: tuple[int, ...],
    leg: set[int],
) -> tuple[list[int], int, int]:
    """``(support geoms, sole geom or -1, reference site)`` from a registry entry, each checked against the leg."""
    names = {"s": label, "side": _SIDE_WORDS.get(side, label)}
    support = []
    for template in entry.support:
        name = template.format(**names)
        geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if geom < 0:
            raise MorphologyError(f"registry geom {name!r} ({model_id}) is not in the model")
        if geom not in leg_geoms:
            raise MorphologyError(f"registry geom {name!r} ({model_id}) is not a colliding geom of leg {label}")
        if geom in support:
            raise MorphologyError(f"registry geom {name!r} ({model_id}) is listed twice for foot {label}")
        support.append(geom)
    sole = -1
    if entry.sole is not None:
        name = entry.sole.format(**names)
        sole = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if sole < 0 or sole not in support:
            raise MorphologyError(f"registry sole {name!r} ({model_id}) is not one of foot {label}'s support geoms")
    if entry.foot_site is None:
        raise MorphologyError(f"registry entry for {model_id} pins no foot_site")
    name = entry.foot_site.format(**names)
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    if site < 0:
        raise MorphologyError(f"registry foot site {name!r} ({model_id}) is not in the model")
    if int(model.site_bodyid[site]) not in leg:
        raise MorphologyError(f"registry foot site {name!r} ({model_id}) is not on leg {label}")
    return support, int(sole), int(site)


def _registered_ankle(model: Any, entry: FootRegistryEntry, label: str, side: str, model_id: ModelId, sole: int) -> int:
    """The registry's ankle hinge for one foot (-1 when the entry declares none): a hinge of the sole's own body.

    On the sole's body its axis is fixed in the sole frame; :func:`_foot_spec`
    then checks that the axis pitches the sole.
    """
    if not entry.declares_span:
        return -1
    assert entry.ankle is not None
    name = entry.ankle.format(s=label, side=_SIDE_WORDS.get(side, label))
    joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if joint < 0:
        raise MorphologyError(f"registry ankle {name!r} ({model_id}) is not a joint of the model")
    if sole < 0:
        raise MorphologyError(f"registry ankle {name!r} ({model_id}) is declared for foot {label}, which has no sole")
    if int(model.jnt_type[joint]) != int(mujoco.mjtJoint.mjJNT_HINGE):
        raise MorphologyError(f"registry ankle {name!r} ({model_id}) is not a hinge")
    if int(model.jnt_bodyid[joint]) != int(model.geom_bodyid[sole]):
        raise MorphologyError(
            f"registry ankle {name!r} ({model_id}) is not a joint of the body of foot {label}'s sole, so the "
            "sole's support span would move with another joint"
        )
    return int(joint)


def _body_to_geom(model: Any, geom: int) -> np.ndarray:
    """The rotation of *geom*'s frame in its body's frame (3x3; columns are the geom's axes)."""
    rotation = np.zeros(9)
    mujoco.mju_quat2Mat(rotation, np.asarray(model.geom_quat[geom], dtype=np.float64))
    return rotation.reshape(3, 3)


def _in_geom_frame(model: Any, geom: int, point_in_body: np.ndarray) -> np.ndarray:
    """A point fixed in *geom*'s body, expressed in the geom's own frame."""
    offset = np.asarray(point_in_body, dtype=np.float64) - np.asarray(model.geom_pos[geom], dtype=np.float64)
    local: np.ndarray = _body_to_geom(model, geom).T @ offset
    return local


def _digit_mtp_joint(model: Any, geom: int, sole: int) -> int:
    """The MTP joint of a hinged digit (module docstring's generic rule), else -1."""
    if sole < 0 or geom == sole or int(model.geom_type[geom]) != int(mujoco.mjtGeom.mjGEOM_CAPSULE):
        return -1
    body = int(model.geom_bodyid[geom])
    if int(model.body_parentid[body]) != int(model.geom_bodyid[sole]):
        return -1
    joints = [joint for joint in range(model.njnt) if int(model.jnt_bodyid[joint]) == body]
    if len(joints) != 1 or int(model.jnt_type[joints[0]]) != int(mujoco.mjtJoint.mjJNT_HINGE):
        return -1
    return int(joints[0])


def _generic_geoms(
    model: Any, leg_geoms: tuple[int, ...], site_bodies: set[int], subtree: Callable[[int], set[int]]
) -> tuple[list[int], int]:
    """The generic rule's ``(support geoms, sole geom or -1)``: the touch-site bodies' geoms and the largest box."""
    covered: set[int] = set()
    for body in site_bodies:
        covered |= subtree(body)
    support = [geom for geom in leg_geoms if int(model.geom_bodyid[geom]) in covered]
    sole, best = -1, 0.0
    for wanted in (int(mujoco.mjtGeom.mjGEOM_BOX), int(mujoco.mjtGeom.mjGEOM_ELLIPSOID)):
        for geom in support:
            if int(model.geom_type[geom]) == wanted:
                footprint = float(np.prod(np.sort(model.geom_size[geom])[-2:]))
                if footprint > best:
                    best, sole = footprint, geom
        if sole >= 0:
            break
    return support, sole


def _foot_spec(
    model: Any,
    home: Any,
    root_forward: np.ndarray,
    root_left: np.ndarray,
    label: str,
    side: str,
    sensor_names: tuple[str, ...],
    primary_site: int,
    leg: set[int],
    leg_geoms: tuple[int, ...],
    support: list[int],
    sole: int,
    ankle: int = -1,
    entry: FootRegistryEntry | None = None,
) -> FootSpec:
    """A :class:`FootSpec`, with the sole frame chosen at the home pose ``home`` (forward-complete scratch data)."""
    normal_axis, forward_axis, lateral_axis = 2, 0, 1
    normal_sign = forward_sign = lateral_sign = 1.0
    home_tilt = float("nan")
    if sole >= 0:
        frame = home.geom_xmat[sole].reshape(3, 3)
        down = frame.T @ np.array([0.0, 0.0, -1.0])
        normal_axis = int(np.argmax(np.abs(down)))
        normal_sign = float(np.sign(down[normal_axis]))
        rest = [axis for axis in range(3) if axis != normal_axis]
        forward = frame.T @ root_forward
        forward_axis = rest[int(np.argmax(np.abs(forward[rest])))]
        forward_sign = float(np.sign(forward[forward_axis]) or 1.0)
        lateral_axis = 3 - normal_axis - forward_axis
        lateral_sign = float(np.sign(np.dot(frame[:, lateral_axis], root_left)) or 1.0)
        home_tilt = float(np.degrees(np.arccos(np.clip(abs(down[normal_axis]), -1.0, 1.0))))
    ankle_forward = heel_line = toe_line = float("nan")
    if ankle >= 0 and entry is not None and entry.heel_line is not None and entry.toe_line is not None:
        # The distance along the forward axis is a lever arm about the ankle only when its axis IS the sole's
        # lateral axis: a forward component would weigh lateral offsets into the moment, a normal one shrink it.
        axis = _body_to_geom(model, sole).T @ np.asarray(model.jnt_axis[ankle], dtype=np.float64)
        off_lateral = float(np.linalg.norm(np.delete(axis, lateral_axis)))
        if off_lateral > _ANKLE_AXIS_OFF_LATERAL * float(np.linalg.norm(axis)):
            off_deg = math.degrees(math.atan2(off_lateral, abs(float(axis[lateral_axis]))))
            raise MorphologyError(
                f"registry ankle {model.joint(ankle).name!r} of foot {label} does not pitch the sole about its "
                f"lateral axis: its axis is {off_deg:.3g} degrees off it, and a contact's distance along the "
                "sole's forward axis is its lever arm about the ankle only for a lateral axis"
            )
        anchor = _in_geom_frame(model, sole, model.jnt_pos[ankle])
        ankle_forward = forward_sign * float(anchor[forward_axis])
        heel_line, toe_line = float(entry.heel_line), float(entry.toe_line)
    return FootSpec(
        label=label,
        side=side,
        sensor_names=sensor_names,
        primary_site=int(primary_site),
        leg_bodies=frozenset(leg),
        leg_geoms=leg_geoms,
        support_geoms=tuple(support),
        sole_geom=int(sole),
        sole_type=int(model.geom_type[sole]) if sole >= 0 else -1,
        sole_normal_axis=normal_axis,
        sole_normal_sign=normal_sign,
        sole_forward_axis=forward_axis,
        sole_forward_sign=forward_sign,
        sole_lateral_axis=lateral_axis,
        sole_lateral_sign=lateral_sign,
        home_sole_tilt_deg=home_tilt,
        ankle_joint=int(ankle),
        ankle_forward_m=ankle_forward,
        heel_line_m=heel_line,
        toe_line_m=toe_line,
    )
