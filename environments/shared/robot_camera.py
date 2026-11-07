"""CPU pinhole RGB-D rendering from the installed MuJoCo camera and range sites.

The rendering boundary returns only RGB bytes, noisy axial depths and ranges.
The diagnostic hit indices and surface normals never leave the renderer.
"""

import mujoco
import numpy as np


class RobotCamera:
    def __init__(self, model, name, width=320, height=200, seed=0):
        self.model = model
        self.id = model.camera(name).id
        self.width, self.height = width, height
        self.rng = np.random.default_rng(seed)
        self.fovy_deg = float(model.cam_fovy[self.id])
        self.focal = 0.5 * height / np.tan(np.deg2rad(self.fovy_deg) / 2)
        y, x = np.indices((height, width))
        vectors = np.stack(
            ((x + 0.5 - width / 2) / self.focal, (height / 2 - y - 0.5) / self.focal, -np.ones_like(x)), axis=-1
        ).reshape(-1, 3)
        self.rays = vectors / np.linalg.norm(vectors, axis=1)[:, None]
        # Render the visible structure, motor cases and covers. Engineering
        # reservations (group 5) and hidden collision duplicates (3) are omitted.
        self.groups = np.array([1, 1, 1, 0, 1, 0], dtype=np.uint8)
        self.colors = model.geom_rgba[:, :3].copy()
        materials = model.geom_matid >= 0
        self.colors[materials] = model.mat_rgba[model.geom_matid[materials], :3]
        self.light = np.array([-0.45, -0.32, 0.83])
        self.tof_ids = []

    def set_range_sites(self, names):
        self.tof_ids = [self.model.site(name).id for name in names]

    def _render_view(self, data, origin, rotation, exclude=-1):
        world = np.ascontiguousarray(self.rays @ rotation.T)
        n = world.shape[0]
        hits = np.empty(n, np.int32)
        distances = np.empty(n)
        normals = np.empty((n, 3))
        mujoco.mj_multiRay(
            self.model, data, origin, world.ravel(), self.groups, True, exclude,
            hits, distances, normals.ravel(), n, 12.0
        )
        valid = hits >= 0
        rgb = np.full((n, 3), [218, 224, 228], dtype=np.uint8)
        brightness = (0.69 + 0.31 * np.clip(normals[valid] @ self.light, -0.8, 1))[:, None]
        rgb[valid] = np.clip(255 * self.colors[hits[valid]] * brightness, 0, 255).astype(np.uint8)
        axial_depth = -self.rays[:, 2] * distances
        axial_depth[~valid] = np.nan
        return rgb.reshape(self.height, self.width, 3), axial_depth.reshape(self.height, self.width)

    def capture(self, data):
        origin = data.cam_xpos[self.id]
        rotation = data.cam_xmat[self.id].reshape(3, 3)
        rgb, depth = self._render_view(data, origin, rotation)
        # Development assumptions for a synchronized, downsized stereo pair.
        # A NaN represents no return or depth outside the operating window.
        valid = np.isfinite(depth) & (depth >= 0.30) & (depth <= 8.0)
        depth[~valid] = np.nan
        depth += self.rng.normal(0, 1, depth.shape) * (0.003 + 0.001 * np.nan_to_num(depth) ** 2)
        depth[self.rng.random(depth.shape) < 0.012] = np.nan
        readings = []
        for site in self.tof_ids:
            axis = data.site_xmat[site].reshape(3, 3)[:, 2]
            geom = np.empty(1, dtype=np.int32)
            distance = mujoco.mj_ray(self.model, data, data.site_xpos[site], axis, self.groups, True, -1, geom)
            measured = float(distance + self.rng.normal(0, 0.008))
            readings.append(measured if 0.04 <= measured <= 2 else float("nan"))
        return rgb, depth.astype(np.float32), tuple(readings)
