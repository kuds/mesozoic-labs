"""Explicit distal-foot geometry registries, separate from whole-leg support.

The registry includes the metatarsal/metacarpal segments that genuinely bear
load on the authored plants. It never ascends to the thigh or shin. Unknown
species or missing registered geometries are refused, rather than silently
counting an entire leg as a foot, and so is a registered geometry whose
collision masks keep it off the declared terrain (it could never bear load).

Foot clearance is the signed distance to the terrain: ``mj_geomDistance`` to
a plane, and, on a heightfield, the vertical gap between sampled foot
surfaces and the env's own terrain height map (``TerrainRealization``),
because ``mj_geomDistance`` does not return surface distance for a
heightfield. A heightfield without a height map that matches the compiled
samples is refused rather than measured wrongly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

FOOT_REGISTRY_VERSION = "mesozoic.foot-geometries/v2"


def _paired(suffixes: tuple[str, ...]) -> dict[str, tuple[str, ...]]:
    return {side: tuple(f"{side}_{suffix}" for suffix in suffixes) for side in ("r", "l")}


def _quadruped() -> dict[str, tuple[str, ...]]:
    return {side: (f"{side}_meta_geom", f"{side}_foot_geom") for side in ("fr", "fl", "rr", "rl")}


FOOT_GEOMETRIES: dict[str, dict[str, tuple[str, ...]]] = {
    "trex": _paired(("metatarsus_geom", "plantar_geom", "toe_d2_geom", "toe_d3_geom", "toe_d4_geom")),
    # The sickle claw (digit II, ``claw_geom``) has collision masks 2/2: it
    # can touch only the other claw and the prey, never terrain, so it is no
    # load-bearing foot segment (v1 listed it).
    "velociraptor": _paired(("metatarsus_geom", "toe_d3_geom", "toe_d4_geom")),
    "brachiosaurus": _quadruped(),
    "dibothrosuchus": _quadruped(),
    "compsognathus": _paired(("metatarsus_geom", "plantar_pad", "toe_d2_geom", "toe_d3_geom", "toe_d4_geom")),
    "compsognathus_robot": {
        "r": ("right_sole", "right_roll_cheek_-1", "right_roll_cheek_1"),
        "l": ("left_sole", "left_roll_cheek_-1", "left_roll_cheek_1"),
    },
}


#: Primitive foot shapes whose surface the heightfield clearance can sample.
_SAMPLED_SURFACE_TYPES = ("mjGEOM_SPHERE", "mjGEOM_CAPSULE", "mjGEOM_ELLIPSOID", "mjGEOM_CYLINDER", "mjGEOM_BOX")
#: Sample density of a foot surface (points per unit sphere, per ring, per box face edge).
_SPHERE_SAMPLES = 96
_RING_SAMPLES = 16
_FACE_SAMPLES = 5


def _can_collide(model: Any, first: int, second: int) -> bool:
    """MuJoCo's collision-mask rule for one geometry pair."""
    return bool(
        int(model.geom_contype[first]) & int(model.geom_conaffinity[second])
        or int(model.geom_contype[second]) & int(model.geom_conaffinity[first])
    )


def _fibonacci_sphere(count: int) -> np.ndarray:
    """Deterministic, nearly uniform unit vectors."""
    index = np.arange(count, dtype=np.float64) + 0.5
    z = 1.0 - 2.0 * index / count
    radius = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    angle = math.pi * (3.0 - math.sqrt(5.0)) * index
    points: np.ndarray = np.stack((radius * np.cos(angle), radius * np.sin(angle), z), axis=1)
    return points


def _surface_samples(model: Any, geom: int) -> np.ndarray:
    """Points on one primitive foot surface, in the geometry's own frame."""
    import mujoco

    kind = int(model.geom_type[geom])
    size = np.asarray(model.geom_size[geom], dtype=np.float64)
    if kind in (int(mujoco.mjtGeom.mjGEOM_SPHERE), int(mujoco.mjtGeom.mjGEOM_ELLIPSOID)):
        radii = size[:3] if kind == int(mujoco.mjtGeom.mjGEOM_ELLIPSOID) else np.full(3, size[0])
        sphere: np.ndarray = _fibonacci_sphere(_SPHERE_SAMPLES) * radii
        return sphere
    angles = 2.0 * math.pi * np.arange(_RING_SAMPLES) / _RING_SAMPLES
    circle = np.stack((np.cos(angles), np.sin(angles), np.zeros(_RING_SAMPLES)), axis=1)
    if kind == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
        radius, half = size[0], size[1]
        cap = _fibonacci_sphere(_SPHERE_SAMPLES) * radius
        cap[:, 2] += np.where(cap[:, 2] >= 0.0, half, -half)
        rings = [circle * radius + [0.0, 0.0, z] for z in np.linspace(-half, half, 5)]
        capsule: np.ndarray = np.concatenate([cap, *rings])
        return capsule
    if kind == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
        radius, half = size[0], size[1]
        rings = [circle * radius + [0.0, 0.0, z] for z in np.linspace(-half, half, 5)]
        disks = [circle * radius * f + [0.0, 0.0, z] for f in (0.0, 0.5) for z in (-half, half)]
        cylinder: np.ndarray = np.concatenate([*rings, *disks])
        return cylinder
    if kind == int(mujoco.mjtGeom.mjGEOM_BOX):
        grid = np.linspace(-1.0, 1.0, _FACE_SAMPLES)
        u, v = (array.ravel() for array in np.meshgrid(grid, grid))
        faces = []
        for axis in range(3):
            for sign in (-1.0, 1.0):
                face = np.empty((u.size, 3))
                face[:, axis] = sign
                face[:, (axis + 1) % 3] = u
                face[:, (axis + 2) % 3] = v
                faces.append(face)
        box: np.ndarray = np.concatenate(faces) * size[:3]
        return box
    raise ValueError(f"gait heightfield clearance cannot sample foot geometry type {kind}")


def _lowest_point(model: Any, data: Any, geom: int) -> np.ndarray:
    """Exact lowest world point of a primitive foot (its support point along -z)."""
    import mujoco

    kind = int(model.geom_type[geom])
    size = np.asarray(model.geom_size[geom], dtype=np.float64)
    centre = np.asarray(data.geom_xpos[geom], dtype=np.float64)
    rotation = np.asarray(data.geom_xmat[geom], dtype=np.float64).reshape(3, 3)
    down = np.array([0.0, 0.0, -1.0])
    local = rotation[2] * -1.0  # rotation.T @ down
    point: np.ndarray
    if kind == int(mujoco.mjtGeom.mjGEOM_SPHERE):
        point = centre + size[0] * down
    elif kind == int(mujoco.mjtGeom.mjGEOM_ELLIPSOID):
        scaled = size[:3] * local
        norm = math.sqrt(float(scaled @ scaled))
        point = centre + rotation @ (size[:3] * scaled / norm) if norm > 0.0 else centre
    elif kind in (int(mujoco.mjtGeom.mjGEOM_CAPSULE), int(mujoco.mjtGeom.mjGEOM_CYLINDER)):
        axis = rotation[:, 2]
        end = centre + math.copysign(size[1], float(axis @ down)) * axis
        if kind == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
            point = end + size[0] * down
        else:
            across = down - float(down @ axis) * axis
            norm = math.sqrt(float(across @ across))
            point = end + size[0] * across / norm if norm > 1e-12 else end
    else:
        point = centre + rotation @ (np.sign(local) * size[:3])
    return point


def _terrain_height_map(env: Any, model: Any, heightfields: list[int]) -> Callable[[Any, Any], Any]:
    """The env's height map for its one heightfield, checked against the compiled samples."""
    import mujoco

    if len(heightfields) != 1:
        raise ValueError("gait clearance supports exactly one heightfield terrain geometry")
    realization = getattr(env, "terrain", None)
    height_at = getattr(realization, "height_at", None)
    if realization is None or not callable(height_at):
        raise ValueError(
            "gait clearance on HFIELD terrain needs the env's terrain height map (TerrainRealization.height_at): "
            "mj_geomDistance does not return surface distance for a heightfield"
        )
    geom = heightfields[0]
    hfield = int(model.geom_dataid[geom])
    config = realization.config
    start = int(model.hfield_adr[hfield])
    samples = np.asarray(model.hfield_data[start : start + config.nrow * config.ncol])
    if (
        hfield < 0
        or int(model.hfield_nrow[hfield]) != config.nrow
        or int(model.hfield_ncol[hfield]) != config.ncol
        or not np.array_equal(samples, np.asarray(realization.normalized_heights, dtype=np.float32).ravel())
        or not np.allclose(model.geom_pos[geom], [0.0, 0.0, realization.geom_z], rtol=0.0, atol=1e-12)
        or not np.allclose(model.geom_quat[geom], [1.0, 0.0, 0.0, 0.0], rtol=0.0, atol=1e-12)
        or int(model.geom_type[geom]) != int(mujoco.mjtGeom.mjGEOM_HFIELD)
    ):
        raise ValueError("the env's terrain height map does not describe its compiled heightfield")
    return height_at  # type: ignore[no-any-return]


@dataclass(frozen=True)
class GaitMorphology:
    """Resolved IDs tied to one live model, plus pinned scale references."""

    species: str
    model: Any
    foot_names: tuple[str, ...]
    foot_geom_ids: tuple[tuple[int, ...], ...]
    foot_site_ids: tuple[int, ...]
    touch_addresses: tuple[tuple[int, ...], ...]
    geom_foot: np.ndarray
    terrain_geom_ids: tuple[int, ...]
    animal_geom_ids: frozenset[int]
    root_body_id: int
    root_qpos_address: int
    body_weight_n: float
    leg_length_m: float
    #: Height map of a heightfield terrain (``TerrainRealization.height_at``),
    #: with each foot geometry's surface samples in its own frame; ``None`` on planes.
    terrain_height: Callable[[Any, Any], Any] | None = field(default=None, compare=False)
    surface_samples: tuple[tuple[np.ndarray, ...], ...] = field(default=(), compare=False)

    @classmethod
    def from_env(cls, env: Any, species: str) -> GaitMorphology:
        import mujoco

        from ..species_names import resolve_species_id

        species = resolve_species_id(species)
        if species not in FOOT_GEOMETRIES:
            raise ValueError(f"no explicit gait foot registry for {species}")
        env = env.unwrapped
        model = env.model
        groups = tuple(tuple(int(a) for a in group) for group in env._foot_sensor_groups)
        address_to_sensor = {int(model.sensor_adr[s]): s for s in range(model.nsensor)}
        foot_names: list[str] = []
        sites: list[int] = []
        for group in groups:
            sensor = address_to_sensor[group[0]]
            if int(model.sensor_objtype[sensor]) != int(mujoco.mjtObj.mjOBJ_SITE):
                raise ValueError("gait touch reference must be a site sensor")
            prefix = str(model.sensor(sensor).name).split("_")[0]
            prefix = {"right": "r", "left": "l"}.get(prefix, prefix)
            foot_names.append(prefix)
            sites.append(int(model.sensor_objid[sensor]))
        registry = FOOT_GEOMETRIES[species]
        if len(foot_names) != len(set(foot_names)) or set(foot_names) != set(registry):
            raise ValueError(f"{species} foot sensor labels do not match its explicit registry: {foot_names}")
        geom_foot = np.full(model.ngeom, -1, dtype=np.int32)
        geom_ids: list[tuple[int, ...]] = []
        for index, foot in enumerate(foot_names):
            ids = tuple(int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)) for name in registry[foot])
            if any(g < 0 for g in ids):
                raise ValueError(f"{species} is missing registered {foot} foot geometry")
            if any(geom_foot[g] >= 0 for g in ids):
                raise ValueError("a geometry belongs to more than one registered foot")
            geom_foot[list(ids)] = index
            geom_ids.append(ids)
        free = [j for j in range(model.njnt) if int(model.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE)]
        if len(free) != 1:
            raise ValueError("gait measurement requires exactly one animal free joint")
        root_body = int(model.jnt_bodyid[free[0]])
        animal_bodies = {root_body}
        for body in range(root_body + 1, model.nbody):
            if int(model.body_parentid[body]) in animal_bodies:
                animal_bodies.add(body)
        animal_geoms = frozenset(g for g in range(model.ngeom) if int(model.geom_bodyid[g]) in animal_bodies)
        if any(g not in animal_geoms for ids in geom_ids for g in ids):
            raise ValueError("registered foot geometry lies outside the animal subtree")
        terrain = tuple(int(g) for g in env._static_floor_geoms())
        if not terrain:
            raise ValueError("gait measurement requires declared terrain geometries")
        for foot, ids in zip(foot_names, geom_ids, strict=True):
            for g in ids:
                if not any(_can_collide(model, g, t) for t in terrain):
                    raise ValueError(
                        f"{species} registered {foot} foot geometry {model.geom(g).name} cannot collide with "
                        "the declared terrain"
                    )
        heightfields = [g for g in terrain if int(model.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_HFIELD)]
        terrain_height = None
        surface: tuple[tuple[np.ndarray, ...], ...] = ()
        if heightfields:
            terrain_height = _terrain_height_map(env, model, heightfields)
            supported = {int(getattr(mujoco.mjtGeom, name)) for name in _SAMPLED_SURFACE_TYPES}
            if any(int(model.geom_type[g]) not in supported for ids in geom_ids for g in ids):
                raise ValueError("gait heightfield clearance needs primitive registered foot geometries")
            surface = tuple(tuple(_surface_samples(model, g) for g in ids) for ids in geom_ids)

        # Resolve the reference on separate data: this cannot disturb a live
        # rollout or consume its reset randomness. Hip anchor height above the
        # authored flat reference matches the historical audit's convention.
        home = mujoco.MjData(model)
        home_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home"))
        if home_id < 0:
            raise ValueError("gait scale requires the declared home keyframe")
        mujoco.mj_resetDataKeyframe(model, home, home_id)
        mujoco.mj_forward(model, home)
        hip_names = ("rr_hip_pitch", "rl_hip_pitch") if len(foot_names) == 4 else ("r_hip_pitch", "l_hip_pitch")
        hip_ids = [int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)) for name in hip_names]
        # The hardware model uses semantic left/right names.
        if any(j < 0 for j in hip_ids) and species == "compsognathus_robot":
            hip_ids = [
                int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name))
                for name in ("right_hip_pitch", "left_hip_pitch")
            ]
        if any(j < 0 for j in hip_ids):
            raise ValueError(f"{species} is missing a declared hip reference joint")
        leg_length = float(np.mean(home.xanchor[hip_ids, 2]))
        gravity = float(np.linalg.norm(model.opt.gravity))
        body_weight = float(sum(model.body_mass[b] for b in animal_bodies)) * gravity
        if not np.isfinite(leg_length) or leg_length <= 0 or not np.isfinite(body_weight) or body_weight <= 0:
            raise ValueError("gait normalization scales must be finite and positive")
        return cls(
            species=species,
            model=model,
            foot_names=tuple(foot_names),
            foot_geom_ids=tuple(geom_ids),
            foot_site_ids=tuple(sites),
            touch_addresses=groups,
            geom_foot=geom_foot,
            terrain_geom_ids=terrain,
            animal_geom_ids=animal_geoms,
            root_body_id=root_body,
            root_qpos_address=int(model.jnt_qposadr[free[0]]),
            body_weight_n=body_weight,
            leg_length_m=leg_length,
            terrain_height=terrain_height,
            surface_samples=surface,
        )

    def foot_clearance(self, data: Any, foot: int) -> float:
        """Signed clearance of one registered foot above the declared terrain, in metres.

        A plane uses ``mj_geomDistance`` (exact). A heightfield uses the lowest
        sampled point of every registered geometry above the height map under
        it, with each geometry's exact lowest point included, so the value is
        exact on locally flat ground; off the map it is NaN (invalid telemetry).
        """
        import mujoco

        model = self.model
        best = math.inf
        for index, geom in enumerate(self.foot_geom_ids[foot]):
            for terrain in self.terrain_geom_ids:
                if int(model.geom_type[terrain]) != int(mujoco.mjtGeom.mjGEOM_HFIELD):
                    best = min(
                        best, float(mujoco.mj_geomDistance(model, data, geom, terrain, 2 * self.leg_length_m, None))
                    )
                    continue
                if self.terrain_height is None:
                    raise ValueError("heightfield clearance requires the terrain height map")
                rotation = np.asarray(data.geom_xmat[geom]).reshape(3, 3)
                points = np.vstack(
                    (
                        np.asarray(data.geom_xpos[geom]) + self.surface_samples[foot][index] @ rotation.T,
                        _lowest_point(model, data, geom),
                    )
                )
                gap = points[:, 2] - np.asarray(self.terrain_height(points[:, 0], points[:, 1]), dtype=np.float64)
                if not np.all(np.isfinite(gap)):
                    return math.nan
                best = min(best, float(np.min(gap)))
        return best

    def describe(self) -> dict[str, Any]:
        return {
            "registry_version": FOOT_REGISTRY_VERSION,
            "species": self.species,
            "foot_names": list(self.foot_names),
            "foot_geometries": {foot: list(FOOT_GEOMETRIES[self.species][foot]) for foot in self.foot_names},
            "terrain_geometries": [str(self.model.geom(g).name) for g in self.terrain_geom_ids],
            "body_weight_n": self.body_weight_n,
            "leg_length_m": self.leg_length_m,
            "leg_length_reference": "mean home-keyframe hind hip anchor height above authored z=0 plane",
            "root_body": str(self.model.body(self.root_body_id).name),
            **(
                {"terrain_clearance": "vertical gap of sampled foot surfaces above the env's terrain height map"}
                if self.terrain_height is not None
                else {}
            ),
        }
