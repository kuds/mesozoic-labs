"""Reusable MuJoCo walking courses in SI units. World +Z is up and the initial heading is +X."""

from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass(frozen=True)
class Mound:
    x: float
    y: float
    x_radius: float
    y_radius: float
    height: float
    z_radius: float = 0.10

    def __post_init__(self):
        if not np.isfinite([self.x, self.y, self.height, *self.radii]).all() or min(self.radii) <= 0:
            raise ValueError("Mound positions and dimensions must be finite, and radii must be positive")
        if not 0 < self.height <= self.z_radius:
            raise ValueError("A mound cap must rise above the floor and no higher than its vertical radius")

    @property
    def center(self):
        return (self.x, self.y, self.height - self.z_radius)

    @property
    def radii(self):
        return (self.x_radius, self.y_radius, self.z_radius)


@dataclass(frozen=True)
class Course:
    """A tilted plane or an overlapping union of a flat plane and ellipsoid caps."""

    pitch_deg: float = 0.0  # positive rises toward world +X, the animal's starting heading
    bank_deg: float = 0.0  # positive rises to the animal's left (+Y)
    mounds: tuple[Mound, ...] = ()

    def __post_init__(self):
        angles = np.abs([self.pitch_deg, self.bank_deg])
        if not np.isfinite(angles).all() or np.any(angles >= 90):
            raise ValueError("Course slopes must be finite and less than 90 degrees in magnitude")
        if self.mounds and np.any(angles):
            raise ValueError("Mound heights are defined relative to a flat base plane")

    def height(self, x, y):
        """World Z of the highest surface at each XY used by the ground-aware controller."""
        x, y = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        z = x * np.tan(np.radians(self.pitch_deg)) + y * np.tan(np.radians(self.bank_deg))
        for mound in self.mounds:
            r2 = ((x - mound.x) / mound.x_radius) ** 2 + ((y - mound.y) / mound.y_radius) ** 2
            cap = mound.center[2] + mound.z_radius * np.sqrt(np.maximum(0.0, 1.0 - r2))
            z = np.where(r2 <= 1.0, np.maximum(z, cap), z)
        return z

    @property
    def rotation(self):
        """Shortest rotation from world up to the plane normal."""
        normal = np.array([-np.tan(np.radians(self.pitch_deg)), -np.tan(np.radians(self.bank_deg)), 1.0])
        normal /= np.linalg.norm(normal)
        quat = np.array([1.0 + normal[2], -normal[1], normal[0], 0.0])
        return quat / np.linalg.norm(quat)

    def surface_names(self, floor_name="ground"):
        return (floor_name, *(f"mound_{i + 1}" for i in range(len(self.mounds))))

    def model(self, path, floor_name="ground"):
        """Compile an MJCF with its named static plane and any additional mounds."""
        spec = mujoco.MjSpec.from_file(str(path))
        floor = spec.geom(floor_name)
        if floor is None or floor.type != mujoco.mjtGeom.mjGEOM_PLANE or floor.parent.name != "world":
            raise ValueError("The course needs a static ground plane centered at the world origin")
        if floor.frame is not None:
            # A world-body geom may still inherit an MJCF frame translation/rotation.
            world_position = spec.compile().geom(floor_name).pos.copy()
            floor.set_frame(spec.worldbody.add_frame())
            floor.pos = world_position
        if np.any(floor.pos):
            raise ValueError("The course needs a static ground plane centered at the world origin")
        floor.alt.type = mujoco.mjtOrientation.mjORIENTATION_QUAT
        floor.quat = self.rotation
        for i, mound in enumerate(self.mounds):
            spec.worldbody.add_geom(
                name=f"mound_{i + 1}",
                type=mujoco.mjtGeom.mjGEOM_ELLIPSOID,
                pos=mound.center,
                size=mound.radii,
                contype=floor.contype,
                conaffinity=floor.conaffinity,
                friction=floor.friction,
                condim=floor.condim,
                solref=floor.solref,
                solimp=floor.solimp,
                solmix=floor.solmix,
                margin=floor.margin,
                gap=floor.gap,
                priority=floor.priority,
                rgba=[0.36, 0.40, 0.32, 1.0],
            )
        return spec.compile()

    def reset(self, model, data, root_joint="root"):
        """Place the neutral skeleton on the plane once. The root remains free thereafter."""
        joint = model.joint(root_joint)
        if joint.type[0] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError(f"Expected a free root joint, got {root_joint!r}")
        mujoco.mj_resetData(model, data)
        adr = joint.qposadr[0]
        rotation = self.rotation
        origin = data.qpos[adr : adr + 3].copy()
        mujoco.mju_rotVecQuat(data.qpos[adr : adr + 3], origin, rotation)
        original = data.qpos[adr + 3 : adr + 7].copy()
        mujoco.mju_mulQuat(data.qpos[adr + 3 : adr + 7], rotation, original)
        mujoco.mj_forward(model, data)
