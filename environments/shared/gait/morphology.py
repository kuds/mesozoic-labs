"""Which geoms are a foot's, which one is its sole, and how heavy the animal is -- read off a live env.

:meth:`Morphology.from_env` turns an env into the lookup tables the
:class:`.recorder.SubstepContactRecorder` classifies contacts with.  It reads
the env's own declarations and never writes the env: the foot sensors
(``_foot_sensor_groups``), the floor (``_static_floor_geoms``), the animal
(``_root_subtree_geoms``) and the home keyframe, the last on a scratch
``MjData`` so ``env.data`` is untouched.

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

The registry (explicit for all six species)
-------------------------------------------
The generic rule is wrong for four of the six, so :data:`SUPPORT_REGISTRY`
names every species' support set, sole and reference site explicitly, and
``tests/test_gait_morphology.py`` pins each difference from the generic
answer.  The registry is MEASUREMENT data, not plant data: it is recorded in
:func:`.constants.measurement_manifest`, not in ``plant_versions.toml``, and
editing an entry after a stage adopts ``stance_quality/v2`` needs a new
:data:`.constants.MEASUREMENT_VERSION`.  A species with no entry falls back
to the generic rule, and the manifest records ``registry: None``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Callable, Mapping

import mujoco
import numpy as np


@dataclass(frozen=True)
class FootRegistryEntry:
    """One species' foot geoms, as name templates.

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

    def as_manifest(self) -> dict[str, Any]:
        """The entry as a measurement manifest records it (``registry``): the templates, JSON-ready."""
        return {"support": list(self.support), "sole": self.sole, "foot_site": self.foot_site}


#: Every species' support set, sole and reference site, and why each
#: differs from the generic rule (statue numbers are 40-episode means on the
#: current plants, seeds 3042-3081):
#:
#: * ``trex`` drops ``{s}_metatarsus_geom``, the capsule above the plantar
#:   box: it carries 0 load in the statue, and as a REQUIRED support geom it
#:   would fail every statue's coverage.
#: * ``velociraptor`` adds ``{s}_toe_d4_geom`` and ``{s}_metatarsus_geom`` to
#:   the ``{s}_toe_d3_geom`` the generic rule finds (the only touch sensor is
#:   on d3): the statue stands on d3 / d4 / metatarsus at 0.557 / 0.182 /
#:   0.261 of each foot's load.  It has no sole (capsules only).  The d2
#:   sickle claw ``{s}_claw_geom`` is contype/conaffinity 2/2 against the
#:   floor's 1/1, so it never touches the floor and is not support.  The
#:   reference site is pinned to ``{s}_foot`` so the planned sensor revision,
#:   which adds metatarsus and d4 touch sensors to the groups, cannot move it.
#: * ``compsognathus`` equals the generic answer: plantar pad plus d2-d4.  Its
#:   metatarsus is a TERMINATING geom (``_body_ground_geoms``), not support.
#: * ``compsognathus_robot`` drops the two ``{side}_roll_cheek_*`` geoms that
#:   share the rolling-foot body: a cheek on the floor is a rolled foot, not a
#:   supported one.
#: * ``dibothrosuchus`` equals the generic answer: one box per foot.
#: * ``brachiosaurus`` drops ``{s}_meta_geom``, the metapodial capsule in the
#:   touch group, which carries 0 load in the statue (at most 3.2e-4 of the
#:   floor load, grazing).  Its ellipsoid sole is authored tilted 19.2 degrees
#:   (fore) and 15.8 degrees (hind), so its flatness keys must be the
#:   ``*_excess_*`` forms measured against that keyframe tilt.
SUPPORT_REGISTRY: Mapping[str, FootRegistryEntry] = MappingProxyType(
    {
        "trex": FootRegistryEntry(
            support=("{s}_plantar_geom", "{s}_toe_d2_geom", "{s}_toe_d3_geom", "{s}_toe_d4_geom"),
            sole="{s}_plantar_geom",
            foot_site="{s}_foot",
        ),
        "velociraptor": FootRegistryEntry(
            support=("{s}_toe_d3_geom", "{s}_toe_d4_geom", "{s}_metatarsus_geom"),
            sole=None,
            foot_site="{s}_foot",
        ),
        "compsognathus": FootRegistryEntry(
            support=("{s}_plantar_pad", "{s}_toe_d2_geom", "{s}_toe_d3_geom", "{s}_toe_d4_geom"),
            sole="{s}_plantar_pad",
            foot_site="{s}_foot_touch_volume",
        ),
        "compsognathus_robot": FootRegistryEntry(
            support=("{side}_sole",),
            sole="{side}_sole",
            foot_site="{side}_foot_touch_volume",
        ),
        "dibothrosuchus": FootRegistryEntry(
            support=("{s}_foot_geom",),
            sole="{s}_foot_geom",
            foot_site="{s}_foot_contact",
        ),
        "brachiosaurus": FootRegistryEntry(
            support=("{s}_foot_geom",),
            sole="{s}_foot_geom",
            foot_site="{s}_foot_contact",
        ),
    }
)

# The foot labels the first sensor's name prefix may give; any other prefix
# becomes ``foot<i>``.  The last letter is the side.
_FOOT_LABELS = ("r", "l", "fr", "fl", "rr", "rl")
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
    per instance (compsognathus and its robot run ``frame_skip = 10``).
    """

    species: str
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
            "species": self.species,
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
                }
                for foot in self.feet
            ],
        }

    @classmethod
    def from_env(
        cls,
        env: Any,
        species: str,
        *,
        registry: Mapping[str, FootRegistryEntry] = SUPPORT_REGISTRY,
    ) -> "Morphology":
        """Describe ``env`` (wrappers are looked through) for ``species``.

        ``species`` takes any name ``species_names.resolve_species_id``
        accepts (``raptor`` is ``velociraptor``); a name it does not know is
        kept as given and, having no registry entry, gets the generic rule.
        ``registry={}`` forces the generic rule for every species -- what the
        registry test compares against.

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
        species = _species_id(species)
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
        entry = registry.get(species)

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
            if entry is not None:
                support, sole, primary_site = _registered_geoms(model, entry, label, side, species, leg_geoms, leg)
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
        geom_support = np.full(model.ngeom, -1, dtype=np.int64)
        for index, foot in enumerate(feet):
            for geom in foot.support_geoms:
                geom_support[geom] = len(support_ids)
                support_ids.append(geom)
                support_foot.append(index)
        for array in (geom_foot, geom_support, is_floor, is_animal):
            array.setflags(write=False)
        return cls(
            species=species,
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
        )


def _species_id(name: str) -> str:
    """The manifest id of ``name`` (``raptor`` -> ``velociraptor``), else ``name`` itself."""
    from ..species_names import resolve_species_id

    try:
        return resolve_species_id(name)
    except ValueError:
        return name


def measurement_definition(species: str) -> dict[str, Any]:
    """This checkout's measurement DEFINITION for *species*, keyed as a manifest records it.

    ``{schema, constants, species, registry}`` --
    :data:`.constants.MEASUREMENT_DEFINITION_KEYS`, the part of a
    :func:`.constants.measurement_manifest` that code alone fixes, so a
    reader with no model (the judge, publication) can compare a recorded
    manifest's definition against the one it would measure with:
    :func:`.constants.measurement_definition_sha256` of a recorded manifest
    equals that of this dict exactly when the two were measured under the
    same constants and the same registry entry.  *species* resolves as
    :meth:`Morphology.from_env` resolves it (``raptor`` is
    ``velociraptor``); a species with no entry records ``registry: None``.
    """
    from .constants import MEASUREMENT_MANIFEST_SCHEMA, measurement_constants

    species = _species_id(species)
    entry = SUPPORT_REGISTRY.get(species)
    return {
        "schema": MEASUREMENT_MANIFEST_SCHEMA,
        "constants": measurement_constants(),
        "species": species,
        "registry": None if entry is None else entry.as_manifest(),
    }


def _registered_geoms(
    model: Any,
    entry: FootRegistryEntry,
    label: str,
    side: str,
    species: str,
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
            raise MorphologyError(f"registry geom {name!r} ({species}) is not in the model")
        if geom not in leg_geoms:
            raise MorphologyError(f"registry geom {name!r} ({species}) is not a colliding geom of leg {label}")
        if geom in support:
            raise MorphologyError(f"registry geom {name!r} ({species}) is listed twice for foot {label}")
        support.append(geom)
    sole = -1
    if entry.sole is not None:
        name = entry.sole.format(**names)
        sole = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if sole < 0 or sole not in support:
            raise MorphologyError(f"registry sole {name!r} ({species}) is not one of foot {label}'s support geoms")
    if entry.foot_site is None:
        raise MorphologyError(f"registry entry for {species} pins no foot_site")
    name = entry.foot_site.format(**names)
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    if site < 0:
        raise MorphologyError(f"registry foot site {name!r} ({species}) is not in the model")
    if int(model.site_bodyid[site]) not in leg:
        raise MorphologyError(f"registry foot site {name!r} ({species}) is not on leg {label}")
    return support, int(sole), int(site)


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
    )
