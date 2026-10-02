"""Explicit distal-foot geometry registries, separate from whole-leg support.

The registry includes the metatarsal/metacarpal segments that genuinely bear
load on the authored plants. It never ascends to the thigh or shin. Unknown
species or missing registered geometries are refused, rather than silently
counting an entire leg as a foot.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

FOOT_REGISTRY_VERSION = "mesozoic.foot-geometries/v1"


def _paired(suffixes: tuple[str, ...]) -> dict[str, tuple[str, ...]]:
    return {side: tuple(f"{side}_{suffix}" for suffix in suffixes) for side in ("r", "l")}


def _quadruped() -> dict[str, tuple[str, ...]]:
    return {side: (f"{side}_meta_geom", f"{side}_foot_geom") for side in ("fr", "fl", "rr", "rl")}


FOOT_GEOMETRIES: dict[str, dict[str, tuple[str, ...]]] = {
    "trex": _paired(("metatarsus_geom", "plantar_geom", "toe_d2_geom", "toe_d3_geom", "toe_d4_geom")),
    "velociraptor": _paired(("metatarsus_geom", "toe_d3_geom", "toe_d4_geom", "claw_geom")),
    "brachiosaurus": _quadruped(),
    "dibothrosuchus": _quadruped(),
    "compsognathus": _paired(("metatarsus_geom", "plantar_pad", "toe_d2_geom", "toe_d3_geom", "toe_d4_geom")),
    "compsognathus_robot": {
        "r": ("right_sole", "right_roll_cheek_-1", "right_roll_cheek_1"),
        "l": ("left_sole", "left_roll_cheek_-1", "left_roll_cheek_1"),
    },
}


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
        )

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
        }
