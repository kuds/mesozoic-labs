"""Renders of compiled candidate models (MuJoCo, offscreen; set MUJOCO_GL=osmesa or egl).

Views are true orthographic projections of the compiled MJCF at the ``home`` keyframe, except
the three-quarter view, which is a perspective view.  Modes:

* ``exterior``  intended exterior (covers opaque, hardware visible through openings);
* ``cutaway``   covers rendered translucent so installed hardware is visible;
* ``hardware``  covers hidden;
* ``silhouette`` black silhouette (segmentation of every robot geom).

All geometry comes from the model file; nothing is drawn by hand except labels and dimension
lines, which are computed from the model.
"""

from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

BG = (246, 245, 240)
VIEWS = {
    # name: (azimuth, elevation, orthographic)
    "side": (90.0, 0.0, True),
    "front": (180.0, 0.0, True),
    "top": (90.0, -89.99, True),
    "threequarter": (138.0, -16.0, False),
    "rear_threequarter": (35.0, -14.0, False),
}


def font(size: int, bold: bool = False):
    names = ["DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"]
    for n in names:
        for d in ("/usr/share/fonts/truetype/dejavu", "/usr/share/fonts/dejavu", "/usr/share/fonts"):
            p = os.path.join(d, n)
            if os.path.exists(p):
                return ImageFont.truetype(p, size)
    return ImageFont.load_default()


class Scene:
    def __init__(self, xml_path: str, width: int = 1200, height: int = 800, key: str = "home"):
        self.m = mujoco.MjModel.from_xml_path(xml_path)
        self.m.vis.global_.offwidth = max(self.m.vis.global_.offwidth, width)
        self.m.vis.global_.offheight = max(self.m.vis.global_.offheight, height)
        self.d = mujoco.MjData(self.m)
        kid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_KEY, key)
        if kid >= 0:
            mujoco.mj_resetDataKeyframe(self.m, self.d, kid)
        mujoco.mj_forward(self.m, self.d)
        self.W, self.H = width, height
        self.r = mujoco.Renderer(self.m, height, width)
        self.rgba0 = self.m.geom_rgba.copy()
        self.floor = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, "floor")

    def close(self):
        self.r.close()

    # ------------------------------------------------------------------ geometry
    def bounds(self, groups=(0, 1, 2)) -> tuple[np.ndarray, np.ndarray]:
        """World AABB of visible robot geometry (vertex-exact for meshes)."""
        m, d = self.m, self.d
        lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
        for g in range(m.ngeom):
            if g == self.floor or m.geom_group[g] not in groups:
                continue
            if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                mid = m.geom_dataid[g]
                a, n = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
                v = m.mesh_vert[a : a + n]
                R = d.geom_xmat[g].reshape(3, 3)
                w = d.geom_xpos[g] + v @ R.T
                lo, hi = np.minimum(lo, w.min(0)), np.maximum(hi, w.max(0))
            else:
                p, rb = d.geom_xpos[g], m.geom_rbound[g]
                if rb <= 0:
                    continue
                aabb = m.geom_aabb[g]  # center(3), half-size(3) in geom frame
                R = d.geom_xmat[g].reshape(3, 3)
                half = np.abs(R) @ aabb[3:]
                c = p + R @ aabb[:3]
                lo, hi = np.minimum(lo, c - half), np.maximum(hi, c + half)
        return lo, hi

    # ------------------------------------------------------------------ rendering
    def _style(self, mode: str):
        m = self.m
        m.geom_rgba[:] = self.rgba0
        if mode == "cutaway":
            for g in range(m.ngeom):
                if m.geom_group[g] == 1:
                    m.geom_rgba[g, 3] = 0.16
        opt = mujoco.MjvOption()
        for g in range(6):
            opt.geomgroup[g] = 0
        groups = {"exterior": (0, 1, 2), "cutaway": (0, 1, 2), "hardware": (0, 2), "silhouette": (0, 1, 2),
                  "covers": (1,)}[mode]
        for g in groups:
            opt.geomgroup[g] = 1
        opt.sitegroup[:] = 0
        return opt

    def render(self, view: str, mode: str = "exterior", center=None, extent: float | None = None,
               floor: bool = True) -> np.ndarray:
        m, d = self.m, self.d
        az, el, ortho = VIEWS[view]
        lo, hi = self.bounds()
        ctr = 0.5 * (lo + hi) if center is None else np.asarray(center)
        size = hi - lo
        aspect = self.W / self.H
        if extent is None:
            if view == "side":
                extent = 1.12 * max(size[2], size[0] / aspect)
            elif view == "front":
                extent = 1.12 * max(size[2], size[1] / aspect)
            elif view == "top":
                extent = 1.12 * max(size[1], size[0] / aspect)
            else:
                extent = 1.0 * max(size)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = ctr
        cam.azimuth, cam.elevation = az, el
        if ortho:
            m.vis.global_.orthographic = 1
            m.vis.global_.fovy = extent
            cam.distance = 6.0 * max(size)
        else:
            m.vis.global_.orthographic = 0
            m.vis.global_.fovy = 30.0
            cam.distance = 1.05 * extent / np.tan(np.radians(15.0))
        opt = self._style(mode)
        if not floor or mode == "silhouette" or view in ("top", "front", "side"):
            opt.geomgroup[0] = 1  # keep feet pads
            self.m.geom_rgba[self.floor, 3] = 0.0
        if mode == "silhouette":
            self.r.enable_depth_rendering()
            self.r.update_scene(d, cam, opt)
            depth = self.r.render()
            self.r.disable_depth_rendering()
            mask = depth < depth.max() - 1e-6
            img = np.empty((self.H, self.W, 3), np.uint8)
            img[:] = (255, 255, 255)
            img[mask] = (12, 12, 12)
            self.m.geom_rgba[:] = self.rgba0
            self._last = dict(ctr=ctr, extent=extent, view=view)
            return img
        self.r.update_scene(d, cam, opt)
        img = self.r.render().copy()
        self.r.enable_depth_rendering()
        self.r.update_scene(d, cam, opt)
        depth = self.r.render()
        self.r.disable_depth_rendering()
        bgmask = depth >= depth.max() - 1e-6
        img[bgmask] = BG
        self.m.geom_rgba[:] = self.rgba0
        self._last = dict(ctr=ctr, extent=extent, view=view)
        return img

    def project(self, p_world, view: str, ctr, extent) -> tuple[float, float]:
        """Pixel coordinates of a world point in an orthographic view."""
        s = self.H / extent
        p = np.asarray(p_world) - ctr
        if view == "side":
            u, v = p[0], p[2]
        elif view == "front":
            u, v = -p[1], p[2]
        elif view == "top":
            u, v = p[0], p[1]
        else:
            raise ValueError(view)
        return self.W / 2 + u * s, self.H / 2 - v * s


def label(img: np.ndarray, lines: list[str], xy=(18, 14), size=20, color=(30, 34, 32)) -> Image.Image:
    im = Image.fromarray(img) if isinstance(img, np.ndarray) else img
    dr = ImageDraw.Draw(im)
    y = xy[1]
    for k, t in enumerate(lines):
        f = font(size if k else size + 4, bold=(k == 0))
        dr.text((xy[0], y), t, fill=color, font=f)
        y += (size + 10) if k == 0 else (size + 6)
    return im


def scale_bar(im: Image.Image, px_per_m: float, length_m: float = 0.1, xy=None, color=(40, 40, 40)):
    dr = ImageDraw.Draw(im)
    W, H = im.size
    x0, y0 = xy if xy else (24, H - 34)
    L = px_per_m * length_m
    dr.line([(x0, y0), (x0 + L, y0)], fill=color, width=3)
    dr.line([(x0, y0 - 6), (x0, y0 + 6)], fill=color, width=2)
    dr.line([(x0 + L, y0 - 6), (x0 + L, y0 + 6)], fill=color, width=2)
    dr.text((x0, y0 + 8), f"{int(round(length_m * 1000))} mm", fill=color, font=font(15))
    return im
