"""Ground reaction and foot clearance measurements for fixed MuJoCo course surfaces."""

from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass
class GroundSample:
    loads: np.ndarray  # summed normal force on each foot, in the caller's mapping order
    clearance: np.ndarray  # minimum signed separation from any ground surface, metres
    contact_xy: list  # horizontal projection of contacts above the reporting force threshold
    slip: float  # normal-force weighted mean speed in the local tangent plane (m/s)
    other_load: float  # normal force between the ground and unlisted body geoms (N)
    surface_loads: np.ndarray  # pad normal force by ground geom


class GroundContacts:
    """Probe named pad geoms against fixed terrain, independent of species and controller.

    Pads and terrain must be pairs supported by MuJoCo's `mj_geomDistance`. Distances
    larger than `distance_limit` are reported at the limit. Velocities are evaluated
    at the point on the moving foot, including its angular motion.
    """

    def __init__(self, model, feet_to_pad_names, surface_names, *, distance_limit=0.5, point_force_n=1.0):
        self.surfaces = [model.geom(name).id for name in surface_names]
        if not self.surfaces or any(model.geom_bodyid[id_] != 0 for id_ in self.surfaces):
            raise ValueError("Ground geoms must be fixed in the world body")
        self.surface_index = {g: i for i, g in enumerate(self.surfaces)}
        if len(self.surface_index) != len(self.surfaces):
            raise ValueError("Ground geom names must be distinct")
        if (
            not np.isfinite(distance_limit)
            or distance_limit <= 0
            or not np.isfinite(point_force_n)
            or point_force_n < 0
        ):
            raise ValueError(
                "The distance limit must be positive and finite, and the force threshold finite and nonnegative"
            )
        self.pads = [[model.geom(name).id for name in names] for names in feet_to_pad_names.values()]
        self.owner = {g: i for i, foot in enumerate(self.pads) for g in foot}
        if not self.pads or not all(self.pads) or len(self.owner) != sum(map(len, self.pads)):
            raise ValueError("Each foot needs a nonempty, distinct set of contact geoms")
        if set(self.owner).intersection(self.surfaces):
            raise ValueError("Ground geoms cannot also be foot pads")
        self.limit = float(distance_limit)
        self.point_force_n = float(point_force_n)
        self.wrench = np.empty(6)
        self.jac = np.empty((3, model.nv))

    def measure(self, model, data):
        loads = np.zeros(len(self.pads))
        surface_loads = np.zeros(len(self.surfaces))
        points = []
        slip_sum = other = 0.0
        for i, contact in enumerate(data.contact):
            surface, geom = contact.geom
            if surface not in self.surface_index:
                surface, geom = geom, surface
                if surface not in self.surface_index:
                    continue
            mujoco.mj_contactForce(model, data, i, self.wrench)
            force = self.wrench[0]
            if geom not in self.owner:
                other += force
                continue
            loads[self.owner[geom]] += force
            surface_loads[self.surface_index[surface]] += force
            if force > self.point_force_n:
                points.append(contact.pos[:2].copy())
            mujoco.mj_jac(model, data, self.jac, None, contact.pos, model.geom_bodyid[geom])
            tangents = contact.frame.reshape(3, 3)[1:]
            slip_sum += force * np.linalg.norm(tangents @ (self.jac @ data.qvel))
        clearance = np.array(
            [
                min(
                    mujoco.mj_geomDistance(model, data, pad, ground, self.limit, None)
                    for pad in foot
                    for ground in self.surfaces
                )
                for foot in self.pads
            ]
        )
        total = loads.sum()
        return GroundSample(loads, clearance, points, slip_sum / total if total else 0.0, other, surface_loads)


def support_margin(points, com):
    """Signed distance (m) to closest edge of the planar convex support hull."""
    points = sorted(set(map(tuple, points)))
    if len(points) == 1:
        return -float(np.linalg.norm(com - points[0]))
    if not points:
        raise AssertionError("No ground support")

    def lower_chain(pts):
        chain = []
        for c in pts:
            while len(chain) >= 2:
                a, b = chain[-2:]
                if (b[0] - a[0]) * (c[1] - a[1]) > (b[1] - a[1]) * (c[0] - a[0]):
                    break
                chain.pop()
            chain.append(c)
        return chain

    hull = np.array(lower_chain(points)[:-1] + lower_chain(reversed(points))[:-1])
    if len(hull) == 2:
        delta = hull[1] - hull[0]
        t = np.clip(np.dot(com - hull[0], delta) / np.dot(delta, delta), 0, 1)
        return -float(np.linalg.norm(com - hull[0] - t * delta))
    edges = np.roll(hull, -1, axis=0) - hull
    delta = com - hull
    return float(np.min((edges[:, 0] * delta[:, 1] - edges[:, 1] * delta[:, 0]) / np.linalg.norm(edges, axis=1)))
