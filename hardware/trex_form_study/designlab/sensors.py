"""Sensor coverage screening on the compiled model: chest camera, ToF pair and 2D lidar.

Screening, not certification.  This is a geometric ray cast of what the robot's own body hides
from its sensors, run on the compiled MJCF so that what is reported is what was built.  It is for
comparing candidates (size, family, posture); it does not predict perception performance.

Usage::

    python -m designlab.sensors 60-M-S-10 --out <dir> [--gait nominal|brisk|none] [--gait-stride 2]

builds the candidate into ``<dir>`` (``<id>.xml``, ledger ``<id>.json``) and writes
``<id>_sensors.json`` and ``<id>_sensors.png``.  Library entry point:
``screen(xml_path, q_home, gait_traj, fov)``.

Fields of view (``SPECS``)
--------------------------
Angles are the ``fov`` strings of ``data/parts.json``; usable ranges are the screening brief's:

* OAK-D Lite stereo depth: HFOV 73 x VFOV 58 deg (DFOV 86), range 0.4-8 m;
* OAK-D Pro W stereo depth: HFOV 127 x VFOV 79.5 deg (DFOV 150), range 0.2-12 m;
* VL53L5CX ToF: 45 x 45 deg (65 deg diagonal), 8 x 8 zones, range 0.02-4 m;
* 2D lidar: 360 deg in the site xy plane, range 0.05-12 m.

parts.json also quotes an *ideal* stereo depth range (~0.8-12 m Lite, ~0.7-12 m Pro W); floor
results are repeated for that minimum (keys ``*_ideal``).  The camera part comes from the ledger
next to the model (``selection['camera']``) unless ``fov`` is given (``fov_for_parts``).

Projection: normalised image coordinates (a, b) in [-1, 1]^2 (a to the image right = site -y,
b up = site +z) map to rays by a rectilinear (pinhole) or an equidistant (f-theta) model,
whichever predicts the sourced diagonal FOV better: Lite rectilinear (85.5 vs 86 deg), Pro W
equidistant (149.8 vs 150 deg), VL53L5CX equidistant (63.6 vs 65 deg).  Ray grids are uniform
in (a, b) at cell centres, i.e. uniform over image pixels.

Ray model
---------
* ``mujoco.mj_multiRay`` / ``mj_ray`` from the sensor site (site x-axis = optical or sensing axis,
  z-axis = image up), all six geom groups enabled, static geoms (the floor) included,
  ``bodyexclude=-1``.  mj_ray ignores geoms whose rgba alpha is 0, so the invisible convex
  collision proxies ``cl_*`` never count; occluders are the visible cover shells (group 1),
  the hardware primitives (group 2) and the foot pads (group 0).  Physics contacts are not used.
* Hits on the sensor boxes ``hw_camera``, ``hw_tof_l``, ``hw_tof_r``, ``hw_lidar`` (name prefixes)
  are filtered: the ray is re-cast from 10 um beyond the hit.  Rays that pass through *another*
  sensor's box are counted (``other_sensor_box_rays``) so such an obstruction is visible.
* Rays start at the site, which is the sensor box centre (not the lens, not either stereo imager).

Sensor windows (assumption)
---------------------------
The camera and ToF sites lie inside cover shells and the MJCF has no cut-outs (the throat camera
sits inside ``cover_neck``, the S-family ToF pair inside ``cover_chest_ventral``).  A sensor is
*enclosed* if at least ``ENCLOSURE_MIN`` of ``MOUNT_CHECK_RAYS`` directions around its site (whole
sphere) first hit robot geometry within ``WINDOW_DEPTH_FRAC`` x H (H = hip height of the model at
home); measured: 0.83-0.98 for the enclosed throat camera and S-family ToF, 0.27-0.44 for the exposed
Hybrid/Tapered ToF.  Only for an enclosed sensor a window is assumed, in its *mounting panels*:
cover panels that are the first hit (after the sensor boxes), within that depth, for at least
``WINDOW_MIN_SHARE`` of its FOV rays at home.  Along every ray the leading run of
mounting-panel hits within that depth is passed (the window); anything after it, including the
same panel further out, occludes.  The window is re-evaluated at every pose, i.e. assumed cut
large enough for all evaluated poses; its required extent (crossing points in the panel's own
body frame, union over poses) is reported, and a panel on another body than the sensor is
flagged (``moves_with``), e.g. the neck skin (head body, neck_pitch) in front of a trunk camera.
The lidar gets no window: it must protrude below the hull, so every hit is an obstruction.

Metrics
-------
1. Camera self-occlusion: ``CAMERA_GRID`` (41 x 31) rays over the FOV; ``blocked`` = fraction whose
   first hit is robot geometry within ``CAMERA_OCCLUSION_RANGE`` (3 m), split by group: head (head
   body: neck, skull, jaw, teeth, eyes), arms (``cover_left_arm``, ``cover_right_arm``), legs (all
   leg bodies), trunk_covers, trunk_hw (other trunk geoms), tail.
2. Floor blind zone: along the image centre column (``CENTRE_COLUMN_RAYS`` rays), the nearest
   visible floor point and the nearest one also inside the depth range, as distance ahead of the
   toe tips (largest x of any foot-body geom at home) and from the camera; over the floor
   rectangle ``FLOOR_RECT`` (2 m long from the toe tips, 1.2 m wide centred on the camera y),
   sampled at ``FLOOR_STEP`` cell centres: area not visible (outside the FOV or occluded) and area
   without valid depth (additionally closer than the minimum range).
3. Person visibility: vertical segments 0.5-1.7 m (knee to head) at horizontal distances 1, 2, 3 m
   from the camera, straight ahead and at +-30 deg bearing, test points every 50 mm; a point counts
   if it is in the FOV, in the depth range and the ray from the camera reaches it unobstructed.
   ``torso_follow``: the torso band 0.9-1.5 m at 1.5, 2.0, 2.5 and 3.0 m straight ahead must have
   at least ``TORSO_MIN_FRACTION`` (0.75, a screening assumption) of its points visible; the
   +-30 deg result is reported alongside.
4. ToF: the 8 x 8 zone-centre rays of each sensor (row 0 = top, column 0 = image left): hit kind
   (floor, robot group or none), slant range and floor point ahead of the toes; and floor coverage
   of the near strip (``NEAR_STRIP`` = 0.8 m ahead of the toes, 1.2 m wide, and the foot path =
   feet outer edges + ``FOOT_PATH_MARGIN``), per ToF and combined with the camera's depth-valid
   floor.  A floor point is covered by a ToF if it is in its FOV, inside 0.02-4 m and unobstructed.
5. Lidar: ``LIDAR_RAYS`` (720) rays in the site xy plane; fraction of the scan blocked by robot
   geometry within ``LIDAR_OCCLUSION_RANGE`` (1 m) and within the 12 m range, by group and as
   angular sectors (0 deg = site x, positive toward site +y), at home and over gait frames (worst,
   mean).  Because the as-built scan plane may lie inside the hull, the home scan is repeated with
   the plane lowered along site -z (``LIDAR_DROPS_MM``; only the plane moves, not the housing);
   ``cleared`` is the first drop at which the trunk-blocked fraction is within ``LIDAR_TRUNK_TOL``
   of its minimum over the sweep (lowering further no longer helps); the gait is also run there.
6. Metrics 1-3 are repeated at the neck_pitch joint-range minimum and maximum, and metric 1 and the
   lidar are evaluated on every ``gait_stride``-th frame of a gait trajectory if one is given.

Also ``mount_check`` (cover or hardware cutting through a sensor box: ``MOUNT_CHECK_RAYS`` rays from
the site, compared with the analytic box exit) and a top-view PNG for visual review.

Limitations
-----------
Geometric visibility only: no depth noise, texture, lighting, IR projector, ToF multipath or lidar
beam divergence and pitch error; the projection is a best fit to quoted H/V/D angles, not a
calibration; windows are assumed (no frame, glass or refraction); one viewpoint per sensor (stereo
baseline ignored); flat floor; the person is a line, not a body; the gait is the kinematic
reference walk of ``gait.generate``, not a controller.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path

import mujoco
import numpy as np

from . import catalog

# ------------------------------------------------------------------------------------------------
# Sensor records and screening thresholds
# ------------------------------------------------------------------------------------------------
SPECS = {
    "luxonis_oak_d_lite": dict(kind="camera", hfov_deg=73.0, vfov_deg=58.0, dfov_deg=86.0, range_m=(0.4, 8.0)),
    "luxonis_oak_d_pro_w": dict(kind="camera", hfov_deg=127.0, vfov_deg=79.5, dfov_deg=150.0, range_m=(0.2, 12.0)),
    "st_vl53l5cx_carrier": dict(kind="tof", hfov_deg=45.0, vfov_deg=45.0, dfov_deg=65.0, zones=(8, 8),
                                range_m=(0.02, 4.0)),
    "pololu_vl53l8cx": dict(kind="tof", hfov_deg=45.0, vfov_deg=45.0, dfov_deg=65.0, zones=(8, 8),
                            range_m=(0.02, 4.0)),
    "ldrobot_stl19p": dict(kind="lidar", hfov_deg=360.0, range_m=(0.05, 12.0)),
    "slamtec_rplidar_c1": dict(kind="lidar", hfov_deg=360.0, range_m=(0.05, 12.0)),
}

OWN_SENSOR_GEOMS = ("hw_camera", "hw_tof_l", "hw_tof_r", "hw_lidar")
ARM_GEOMS = ("cover_left_arm", "cover_right_arm")
#: Robot geometry groups (class codes 1..6; 0 = floor/environment, -1 = no hit).
GROUPS = ("head", "arms", "legs", "trunk_covers", "trunk_hw", "tail")
TRUNK_CODES = (4, 5)

CAMERA_GRID = (41, 31)  # rays across x (columns) and up (rows)
CAMERA_OCCLUSION_RANGE = 3.0  # m
CENTRE_COLUMN_RAYS = 241
FLOOR_STEP = 0.025  # m, floor sample cell
FLOOR_RECT = (2.0, 1.2)  # m, length ahead of the toe tips, width centred on the camera y
NEAR_STRIP = 0.8  # m ahead of the toe tips (ToF + camera combined coverage)
FOOT_PATH_MARGIN = 0.05  # m beyond the feet's outer edges
FLOOR_VIEW = (-1.0, 3.6, 1.6)  # PNG raster: x from toe-1.0 to toe+3.6 m, |y| <= 1.6 m
PERSON_Z = (0.50, 1.70)  # m, knee to head
PERSON_DZ = 0.05
PERSON_DISTANCES = (1.0, 2.0, 3.0)
PERSON_BEARINGS_DEG = (-30.0, 0.0, 30.0)  # positive = to the robot's left
TORSO_Z = (0.90, 1.50)
TORSO_DISTANCES = (1.5, 2.0, 2.5, 3.0)
TORSO_MIN_FRACTION = 0.75  # screening assumption for "sees the torso"
LIDAR_RAYS = 720
LIDAR_OCCLUSION_RANGE = 1.0  # m
LIDAR_DROPS_MM = (0, 10, 20, 30, 40, 50, 60, 80, 100, 125, 150)
LIDAR_TRUNK_TOL = 0.01
WINDOW_DEPTH_FRAC = 0.35  # x hip height
WINDOW_MIN_SHARE = 0.10
ENCLOSURE_MIN = 0.6
MOUNT_CHECK_RAYS = 400

GEOMGROUP_ALL = np.ones(6, np.uint8)
RAY_CUTOFF = 100.0  # m (mj_multiRay cutoff for non-plane geoms)
RAY_EPS = 1e-5  # m, re-cast offset past a filtered hit
MAX_RECAST = 32

#: Floor raster codes: 0 outside FOV, 1 visible with valid depth, 2 visible without valid depth,
#: 10 + group code occluded by that group, -1 unresolved.
F_OUT, F_VALID, F_NODEPTH, F_OCC, F_UNRES = 0, 1, 2, 10, -1


# ------------------------------------------------------------------------------------------------
# Fields of view
# ------------------------------------------------------------------------------------------------
def projection_for(hfov_deg: float, vfov_deg: float, dfov_deg: float | None) -> str:
    """'rectilinear' or 'equidistant', whichever predicts the quoted diagonal FOV better."""
    if not dfov_deg:
        return "rectilinear"
    h, v = np.radians(hfov_deg) / 2, np.radians(vfov_deg) / 2
    rect = 2 * np.degrees(np.arctan(np.hypot(np.tan(h), np.tan(v)))) if max(h, v) < np.pi / 2 else np.inf
    eqd = 2 * np.degrees(np.hypot(h, v))
    return "rectilinear" if abs(rect - dfov_deg) <= abs(eqd - dfov_deg) else "equidistant"


def _ideal_range(text: str):
    mt = re.search(r"[Ii]deal depth range\s*~?\s*([\d.]+)\s*-\s*([\d.]+)\s*m", text or "")
    return (float(mt.group(1)), float(mt.group(2))) if mt else None


def fov_for_parts(camera: str, tof: str, lidar: str | None = None) -> dict:
    """FOV dict for ``screen`` from catalog part ids (``SPECS`` numbers plus the parts.json text)."""
    out = {}
    for key, pid in (("camera", camera), ("tof", tof), ("lidar", lidar)):
        if not pid:
            out[key] = None
            continue
        if pid not in SPECS:
            raise ValueError(f"no field-of-view record for part {pid!r}; add it to sensors.SPECS")
        spec = dict(SPECS[pid])
        try:
            src = catalog.part(pid).fov
        except KeyError:
            src = ""
        spec.update(part=pid, parts_json_fov=src)
        if spec["kind"] != "lidar":
            spec["projection"] = projection_for(spec["hfov_deg"], spec["vfov_deg"], spec.get("dfov_deg"))
        if spec["kind"] == "camera":
            spec["ideal_range_m"] = _ideal_range(src)
        out[key] = spec
    return out


def fov_from_ledger(xml_path) -> dict:
    """FOV dict from the ledger ``<id>.json`` written next to ``<id>.xml`` by ``pipeline``."""
    sel = json.loads(Path(xml_path).with_suffix(".json").read_text())["selection"]
    return fov_for_parts(sel["camera"], sel["tof"], sel.get("lidar"))


class Lens:
    """Normalised image coordinates (a right, b up, in [-1, 1]) <-> ray directions."""

    def __init__(self, spec: dict):
        self.hh = np.radians(spec["hfov_deg"]) / 2
        self.vh = np.radians(spec["vfov_deg"]) / 2
        self.projection = spec.get("projection") or projection_for(spec["hfov_deg"], spec["vfov_deg"],
                                                                   spec.get("dfov_deg"))

    def local(self, a, b) -> np.ndarray:
        a, b = np.broadcast_arrays(np.asarray(a, float), np.asarray(b, float))
        if self.projection == "rectilinear":
            v = np.stack([np.ones_like(a), -a * np.tan(self.hh), b * np.tan(self.vh)], -1)
        else:
            u, w = a * self.hh, b * self.vh
            th, ph = np.hypot(u, w), np.arctan2(w, u)
            v = np.stack([np.cos(th), -np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph)], -1)
        return v / np.linalg.norm(v, axis=-1, keepdims=True)

    def world(self, R, a, b) -> np.ndarray:
        return self.local(a, b) @ np.asarray(R).T

    def coords(self, R, v):
        loc = np.atleast_2d(v) @ np.asarray(R)
        x, y, z = loc[:, 0], loc[:, 1], loc[:, 2]
        if self.projection == "rectilinear":
            with np.errstate(divide="ignore", invalid="ignore"):
                a = np.where(x > 1e-9, (-y / x) / np.tan(self.hh), np.inf)
                b = np.where(x > 1e-9, (z / x) / np.tan(self.vh), np.inf)
        else:
            th, ph = np.arctan2(np.hypot(y, z), x), np.arctan2(z, -y)
            a, b = th * np.cos(ph) / self.hh, th * np.sin(ph) / self.vh
        return a, b

    def inside(self, R, v) -> np.ndarray:
        a, b = self.coords(R, v)
        return (np.abs(a) <= 1.0 + 1e-9) & (np.abs(b) <= 1.0 + 1e-9)

    @staticmethod
    def grid(nx: int, ny: int):
        """Cell-centre samples of an nx x ny image grid (each ray stands for an equal image area)."""
        a, b = np.meshgrid(-1 + (2 * np.arange(nx) + 1) / nx, -1 + (2 * np.arange(ny) + 1) / ny)
        return a.ravel(), b.ravel()

    def aspect(self) -> float:
        """Image height / width."""
        if self.projection == "rectilinear":
            return np.tan(self.vh) / np.tan(self.hh)
        return self.vh / self.hh


# ------------------------------------------------------------------------------------------------
# Model, classification and ray casting
# ------------------------------------------------------------------------------------------------
class _Scene:
    def __init__(self, xml_path, q_home=None):
        m = self.m = mujoco.MjModel.from_xml_path(str(xml_path))
        self.d = mujoco.MjData(m)
        if q_home is None:
            k = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home")
            if k < 0:
                raise ValueError("model has no 'home' keyframe and no q_home was given")
            q_home = m.key_qpos[k]
        self.q_home = np.array(q_home, float).copy()
        self.gname = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or f"geom{g}" for g in range(m.ngeom)]
        self.bname = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or f"body{b}" for b in range(m.nbody)]
        self.floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        code = np.zeros(m.ngeom, int)
        for g in range(m.ngeom):
            b = int(m.geom_bodyid[g])
            bn, gn = self.bname[b], self.gname[g]
            if b == 0:
                code[g] = 0
            elif bn == "head":
                code[g] = 1
            elif gn in ARM_GEOMS:
                code[g] = 2
            elif bn.startswith(("left_", "right_")):
                code[g] = 3
            elif bn == "tail":
                code[g] = 6
            elif gn.startswith("cover_"):
                code[g] = 4
            else:
                code[g] = 5
        self.code = code
        self.own = np.array([n.startswith(OWN_SENSOR_GEOMS) for n in self.gname])
        self.cover = np.array([n.startswith("cover_") for n in self.gname])
        self.visible = (m.geom_rgba[:, 3] > 0) & (m.geom_group <= 2)
        self._nowin = np.zeros(m.ngeom, bool)
        self.set_pose(self.q_home)
        thighs = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{s}_thigh") for s in ("left", "right")]
        self.H = float(np.mean([self.d.xpos[b][2] for b in thighs]))

    def set_pose(self, q):
        self.d.qpos[:] = q
        self.d.qvel[:] = 0
        mujoco.mj_forward(self.m, self.d)

    def site(self, name: str) -> int:
        return mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, name)

    def geom(self, name: str) -> int:
        return mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, name)

    def site_pose(self, s: int):
        return self.d.site_xpos[s].copy(), self.d.site_xmat[s].reshape(3, 3).copy()

    def classes(self, g) -> np.ndarray:
        g = np.asarray(g)
        return np.where(g >= 0, self.code[np.maximum(g, 0)], -1)

    def cast(self, origin, dirs, self_box: int = -1, window=None, depth: float = 0.0):
        """First hit along each ray after filtering sensor boxes and window crossings.

        Returns (geom id or -1, distance or -1, first window geom or -1, its distance, flag: ray
        passed through another sensor's box)."""
        m, d = self.m, self.d
        o = np.ascontiguousarray(origin, float)
        v = np.ascontiguousarray(dirs, float).reshape(-1, 3)
        n = len(v)
        g = np.full(n, -1, np.int32)
        t = np.full(n, -1.0)
        wg = np.full(n, -1, np.int32)
        wt = np.full(n, np.nan)
        other = np.zeros(n, bool)
        if n == 0:
            return g, t, wg, wt, other
        mujoco.mj_multiRay(m, d, o, v.ravel(), GEOMGROUP_ALL, 1, -1, g, t, None, n, RAY_CUTOFF)
        win = self._nowin if window is None else window
        gg = np.maximum(g, 0)
        todo = np.flatnonzero((g >= 0) & (self.own[gg] | (win[gg] & (t <= depth))))
        one = np.zeros(1, np.int32)
        for i in todo:
            vi = v[i]
            gi, ti = int(g[i]), float(t[i])
            for _ in range(MAX_RECAST):
                if self.own[gi]:
                    if gi != self_box:
                        other[i] = True
                elif win[gi] and ti <= depth:
                    if wg[i] < 0:
                        wg[i], wt[i] = gi, ti
                else:
                    break
                dd = mujoco.mj_ray(m, d, o + vi * (ti + RAY_EPS), vi, GEOMGROUP_ALL, 1, -1, one)
                if dd < 0:
                    gi, ti = -1, -1.0
                    break
                gi, ti = int(one[0]), ti + RAY_EPS + dd
            g[i], t[i] = gi, ti
        return g, t, wg, wt, other

    def points_visible(self, sensor: "_Sensor", o, R, P, use_range=True):
        """Visibility of world points from a sensor: (visible, in_fov, in_range, blocked, blocking class)."""
        P = np.atleast_2d(P)
        dv = P - o
        dist = np.linalg.norm(dv, axis=1)
        u = dv / dist[:, None]
        infov = sensor.lens.inside(R, u)
        lo, hi = sensor.range
        inrange = (dist >= lo) & (dist <= hi) if use_range else np.ones(len(P), bool)
        blocked = np.zeros(len(P), bool)
        bcls = np.full(len(P), -1)
        idx = np.flatnonzero(infov)
        if len(idx):
            g, t, *_ = self.cast(o, u[idx], sensor.box, sensor.window, sensor.depth)
            tol = 2e-3 + 1e-3 * dist[idx]
            hit = (g >= 0) & (t < dist[idx] - tol)
            blocked[idx] = hit
            bcls[idx] = np.where(hit, self.classes(g), -1)
        return infov & inrange & ~blocked, infov, inrange, blocked, bcls

    def geom_points(self, g: int) -> np.ndarray:
        """World points bounding a geom: mesh vertices, box corners, capsule/sphere/cylinder extremes."""
        m, d = self.m, self.d
        R = d.geom_xmat[g].reshape(3, 3)
        c = d.geom_xpos[g]
        s = m.geom_size[g]
        typ = m.geom_type[g]
        E = np.vstack([np.eye(3), -np.eye(3)])
        if typ == mujoco.mjtGeom.mjGEOM_MESH:
            mid = m.geom_dataid[g]
            a, n = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
            return c + m.mesh_vert[a : a + n] @ R.T
        if typ == mujoco.mjtGeom.mjGEOM_BOX:
            loc = np.array([[sx * s[0], sy * s[1], sz * s[2]] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
            return c + loc @ R.T
        if typ == mujoco.mjtGeom.mjGEOM_CAPSULE:
            ends = np.array([c + R[:, 2] * s[1], c - R[:, 2] * s[1]])
            return (ends[:, None, :] + s[0] * E[None]).reshape(-1, 3)
        if typ == mujoco.mjtGeom.mjGEOM_CYLINDER:
            ang = np.linspace(0, 2 * np.pi, 13)[:-1]
            ring = np.cos(ang)[:, None] * R[:, 0] + np.sin(ang)[:, None] * R[:, 1]
            return np.vstack([c + R[:, 2] * h + s[0] * ring for h in (s[1], -s[1])])
        if typ == mujoco.mjtGeom.mjGEOM_SPHERE:
            return c + s[0] * E
        if typ == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
            return c + (E * s[None, :3]) @ R.T
        return np.zeros((0, 3))


class _Sensor:
    def __init__(self, sc: _Scene, name: str, spec: dict):
        self.name = name
        self.site = sc.site(name)
        self.box = sc.geom(f"hw_{name}")
        self.body = int(sc.m.site_bodyid[self.site])
        self.spec = spec
        self.lens = Lens(spec) if spec["kind"] != "lidar" else None
        self.range = tuple(float(x) for x in spec["range_m"])
        self.window = np.zeros(sc.m.ngeom, bool)
        self.window_share: dict = {}
        self.enclosure = 0.0
        self.depth = WINDOW_DEPTH_FRAC * sc.H
        self.window_pts: dict = {}  # panel name -> list of (pose label, local pts, sensor-frame pts, distances)


def _fibonacci_sphere(n: int) -> np.ndarray:
    k = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * k / n)
    th = np.pi * (1 + np.sqrt(5.0)) * k
    return np.column_stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)])


def _runs(mask: np.ndarray, circular: bool = False) -> list[tuple[int, int]]:
    """[start, end] index runs of True (inclusive); a circular run may wrap (start > end)."""
    mask = np.asarray(mask, bool)
    n = len(mask)
    if not mask.any():
        return []
    if mask.all():
        return [(0, n - 1)]
    idx = np.flatnonzero(mask)
    runs, s = [], idx[0]
    for a, b in zip(idx[:-1], idx[1:], strict=True):
        if b != a + 1:
            runs.append((s, a))
            s = b
    runs.append((s, idx[-1]))
    if circular and len(runs) > 1 and runs[0][0] == 0 and runs[-1][1] == n - 1:
        runs = [(runs[-1][0], runs[0][1])] + runs[1:-1]
    return runs


def _r(x, nd=4):
    """Round floats (recursively) for the JSON report."""
    if isinstance(x, dict):
        return {str(k): _r(v, nd) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_r(v, nd) for v in x]
    if isinstance(x, np.ndarray):
        return _r(x.tolist(), nd)
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return None if not np.isfinite(x) else round(float(x), nd)
    return x


# ------------------------------------------------------------------------------------------------
# The screen
# ------------------------------------------------------------------------------------------------
class _Screen:
    def __init__(self, xml_path, q_home, fov: dict):
        self.sc = sc = _Scene(xml_path, q_home)
        self.fov = fov
        if sc.site("camera") < 0:
            raise ValueError("model has no 'camera' site")
        self.cam = _Sensor(sc, "camera", fov["camera"])
        self.tofs = [_Sensor(sc, n, fov["tof"]) for n in ("tof_l", "tof_r") if sc.site(n) >= 0 and fov.get("tof")]
        self.lidar = _Sensor(sc, "lidar", fov["lidar"]) if sc.site("lidar") >= 0 and fov.get("lidar") else None
        j = mujoco.mj_name2id(sc.m, mujoco.mjtObj.mjOBJ_JOINT, "neck_pitch")
        self.neck_adr = int(sc.m.jnt_qposadr[j]) if j >= 0 else None
        self.neck_range = tuple(float(x) for x in sc.m.jnt_range[j]) if j >= 0 else None
        sc.set_pose(sc.q_home)
        feet = [mujoco.mj_name2id(sc.m, mujoco.mjtObj.mjOBJ_BODY, f"{s}_foot") for s in ("left", "right")]
        fpts = np.vstack([sc.geom_points(g) for g in range(sc.m.ngeom)
                          if sc.m.geom_bodyid[g] in feet and sc.visible[g]])
        self.x_toe = float(fpts[:, 0].max())
        self.y_c = float(sc.d.site_xpos[self.cam.site][1])
        self.foot_half = float(np.abs(fpts[:, 1] - self.y_c).max())
        for s in [self.cam] + self.tofs:
            self._identify_window(s)

    # ------------------------------------------------------------------ helpers
    def pose(self, label: str) -> np.ndarray:
        q = self.sc.q_home.copy()
        if label in ("neck_min", "neck_max") and self.neck_adr is not None:
            q[self.neck_adr] = self.neck_range[0 if label == "neck_min" else 1]
        return q

    def _identify_window(self, s: _Sensor):
        sc = self.sc
        sc.set_pose(sc.q_home)
        o, R = sc.site_pose(s.site)
        g, t, *_ = sc.cast(o, _fibonacci_sphere(MOUNT_CHECK_RAYS), s.box)
        s.enclosure = float(((sc.classes(g) >= 1) & (t <= s.depth)).mean())  # robot geometry only, not the floor
        if s.enclosure < ENCLOSURE_MIN:
            return
        v = s.lens.world(R, *Lens.grid(*CAMERA_GRID))
        g, t, *_ = sc.cast(o, v, s.box)
        first = g[(g >= 0) & (t <= s.depth)]
        first = first[sc.cover[first]]
        for gid, c in Counter(first.tolist()).items():
            share = c / len(v)
            s.window_share[sc.gname[gid]] = share
            if share >= WINDOW_MIN_SHARE:
                s.window[gid] = True

    def _collect_window(self, s: _Sensor, label: str, o, R, v, wg, wt, keep):
        """Store window crossings of rays that go on to see the environment (``keep``)."""
        sc = self.sc
        for gid in np.unique(wg[(wg >= 0) & keep]):
            k = (wg == gid) & keep
            p = o + v[k] * wt[k][:, None]
            b = int(sc.m.geom_bodyid[gid])
            loc = (p - sc.d.xpos[b]) @ sc.d.xmat[b].reshape(3, 3)
            s.window_pts.setdefault(sc.gname[gid], []).append((label, loc, (p - o) @ R, wt[k]))

    def _floor_lattice(self, view: bool):
        st = FLOOR_STEP
        if view:
            kx = np.arange(int(round(FLOOR_VIEW[0] / st)), int(round(FLOOR_VIEW[1] / st)))
            jy = np.arange(-int(round(FLOOR_VIEW[2] / st)), int(round(FLOOR_VIEW[2] / st)))
        else:
            kx = np.arange(0, int(round(FLOOR_RECT[0] / st)))
            jy = np.arange(-int(round(FLOOR_RECT[1] / 2 / st)), int(round(FLOOR_RECT[1] / 2 / st)))
        XX, YY = np.meshgrid(self.x_toe + (kx + 0.5) * st, self.y_c + (jy + 0.5) * st)
        KX, JY = np.meshgrid(kx, jy)
        return XX, YY, KX, JY

    def _floor_raster(self, s: _Sensor, o, R, XX, YY, rng=None):
        """Floor visibility codes (F_*) and slant distance for lattice points."""
        sc = self.sc
        P = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(XX.size)])
        dv = P - o
        dist = np.linalg.norm(dv, axis=1)
        u = dv / dist[:, None]
        infov = s.lens.inside(R, u)
        code = np.full(len(P), F_OUT)
        idx = np.flatnonzero(infov)
        g, t, wg, wt, _ = sc.cast(o, u[idx], s.box, s.window, s.depth)
        cls = sc.classes(g)
        tol = 2e-3 + 1e-3 * dist[idx]
        seen = (cls == 0) & (np.abs(t - dist[idx]) <= tol)
        occ = (cls >= 1) & (t < dist[idx] - tol)
        lo, hi = rng or s.range
        sub = np.full(len(idx), F_UNRES)
        sub[seen] = np.where((dist[idx][seen] >= lo) & (dist[idx][seen] <= hi), F_VALID, F_NODEPTH)
        sub[occ] = F_OCC + cls[occ]
        code[idx] = sub
        return code.reshape(XX.shape), dist.reshape(XX.shape)

    @staticmethod
    def _area_stats(code, dist, rng, ideal=None) -> dict:
        n = code.size
        a = FLOOR_STEP**2
        seen = (code == F_VALID) | (code == F_NODEPTH)
        valid = seen & (dist >= rng[0]) & (dist <= rng[1])
        out = dict(area_m2=n * a, not_visible_m2=float((~seen).sum() * a), no_valid_depth_m2=float((~valid).sum() * a),
                   visible_fraction=float(seen.mean()), depth_valid_fraction=float(valid.mean()),
                   outside_fov_m2=float((code == F_OUT).sum() * a),
                   occluded_m2={GROUPS[k - 1]: float((code == F_OCC + k).sum() * a) for k in range(1, 7)
                                if (code == F_OCC + k).any()},
                   too_close_m2=float((seen & (dist < rng[0])).sum() * a),
                   unresolved_cells=int((code == F_UNRES).sum()))
        if ideal:
            iv = seen & (dist >= ideal[0]) & (dist <= ideal[1])
            out["no_valid_depth_ideal_m2"] = float((~iv).sum() * a)
        return out

    # ------------------------------------------------------------------ 1. camera occlusion
    def camera_grid(self, label: str, collect=True):
        sc, cam = self.sc, self.cam
        o, R = sc.site_pose(cam.site)
        nx, ny = CAMERA_GRID
        v = cam.lens.world(R, *Lens.grid(nx, ny))
        g, t, wg, wt, other = sc.cast(o, v, cam.box, cam.window, cam.depth)
        cls = sc.classes(g)
        blocked = (cls >= 1) & (t <= CAMERA_OCCLUSION_RANGE)
        if collect:
            self._collect_window(cam, label, o, R, v, wg, wt, ~blocked)
        n = len(v)
        names = Counter(sc.gname[k] for k in g[blocked])
        res = dict(blocked=float(blocked.mean()),
                   by_group={GROUPS[k - 1]: float((blocked & (cls == k)).sum() / n) for k in range(1, 7)},
                   top_geoms={nm: c / n for nm, c in names.most_common(6)},
                   floor=float((cls == 0).mean()), sky=float((cls < 0).mean()),
                   robot_beyond_range=float(((cls >= 1) & ~blocked).mean()),
                   window_rays=float((wg >= 0).mean()), other_sensor_box_rays=int(other.sum()))
        imap = np.where(blocked, cls, np.where(cls >= 1, 7, cls)).reshape(ny, nx)
        return res, imap

    # ------------------------------------------------------------------ 2. floor blind zone
    def centre_column(self) -> dict:
        sc, cam = self.sc, self.cam
        o, R = sc.site_pose(cam.site)
        b = np.linspace(-1, 1, CENTRE_COLUMN_RAYS)
        v = cam.lens.world(R, np.zeros_like(b), b)
        g, t, *_ = sc.cast(o, v, cam.box, cam.window, cam.depth)
        cls = sc.classes(g)
        fl = cls == 0
        p = o + v * t[:, None]
        ahead = p[:, 0] - self.x_toe
        out = {}
        v0 = v[0]
        out["fov_edge_floor_ahead_of_toes_m"] = float(o[0] + v0[0] * (-o[2] / v0[2]) - self.x_toe) if v0[2] < -1e-9 \
            else None
        ideal = cam.spec.get("ideal_range_m")
        masks = [("visible", fl), ("depth_valid", fl & (t >= cam.range[0]) & (t <= cam.range[1]))]
        if ideal:
            masks.append(("depth_valid_ideal", fl & (t >= ideal[0]) & (t <= ideal[1])))
        for key, mk in masks:
            if not mk.any():
                out[key] = None
                continue
            k = int(np.flatnonzero(mk)[np.argmin(ahead[mk])])
            if k == 0:
                lim = "fov_edge"
            elif cls[k - 1] >= 1:
                lim = "occlusion:" + GROUPS[cls[k - 1] - 1]
            elif fl[k - 1]:
                lim = "min_range"
            else:
                lim = "none"
            out[key] = dict(ahead_of_toes_m=float(ahead[k]), horizontal_from_camera_m=float(np.hypot(*(p[k] - o)[:2])),
                            slant_m=float(t[k]), image_b=float(b[k]), limited_by=lim)
        out["visible_intervals_ahead_of_toes_m"] = [[float(ahead[i]), float(min(ahead[j], 99.0))]
                                                    for i, j in _runs(fl)][:6]
        return out

    def floor_rect(self) -> dict:
        cam = self.cam
        o, R = self.sc.site_pose(cam.site)
        XX, YY, _, _ = self._floor_lattice(view=False)
        code, dist = self._floor_raster(cam, o, R, XX, YY)
        return self._area_stats(code, dist, cam.range, cam.spec.get("ideal_range_m"))

    # ------------------------------------------------------------------ 3. person
    def person(self) -> dict:
        sc, cam = self.sc, self.cam
        o, R = sc.site_pose(cam.site)
        hd = np.array([R[0, 0], R[1, 0]])
        hd = hd / np.linalg.norm(hd) if np.linalg.norm(hd) > 1e-6 else np.array([1.0, 0.0])
        zs = np.round(np.arange(PERSON_Z[0], PERSON_Z[1] + 1e-9, PERSON_DZ), 4)
        zt = np.round(np.arange(TORSO_Z[0], TORSO_Z[1] + 1e-9, PERSON_DZ), 4)

        def seg(dist_h, bearing, z):
            c, s_ = np.cos(np.radians(bearing)), np.sin(np.radians(bearing))
            dxy = np.array([c * hd[0] - s_ * hd[1], s_ * hd[0] + c * hd[1]])
            P = np.column_stack([np.full(len(z), o[0] + dist_h * dxy[0]), np.full(len(z), o[1] + dist_h * dxy[1]), z])
            vis, infov, inr, blk, bcls = sc.points_visible(cam, o, R, P)
            return vis, infov, blk, bcls, P

        segs = []
        for bearing in PERSON_BEARINGS_DEG:
            for dh in PERSON_DISTANCES:
                vis, infov, blk, bcls, P = seg(dh, bearing, zs)
                occ = Counter(GROUPS[c - 1] for c in bcls[blk] if c >= 1)
                segs.append(dict(distance_m=dh, bearing_deg=bearing, visible_fraction=float(vis.mean()),
                                 in_fov_fraction=float(infov.mean()), occluded_fraction=float(blk.mean()),
                                 occluded_by={k: c / len(zs) for k, c in occ.items()},
                                 visible_z_m=[float(zs[vis].min()), float(zs[vis].max())] if vis.any() else None,
                                 xy=[float(P[0, 0]), float(P[0, 1])]))
        torso = []
        for bearing in PERSON_BEARINGS_DEG:
            for dh in TORSO_DISTANCES:
                vis, *_ = seg(dh, bearing, zt)
                torso.append(dict(distance_m=dh, bearing_deg=bearing, visible_fraction=float(vis.mean())))
        ahead = [t_["visible_fraction"] for t_ in torso if t_["bearing_deg"] == 0.0]
        side = [t_["visible_fraction"] for t_ in torso if t_["bearing_deg"] != 0.0]
        return dict(segments=segs, torso=torso,
                    torso_follow=dict(min_fraction_ahead=float(min(ahead)), ok=bool(min(ahead) >= TORSO_MIN_FRACTION),
                                      min_fraction_30deg=float(min(side)) if side else None,
                                      ok_30deg=bool(min(side) >= TORSO_MIN_FRACTION) if side else None,
                                      threshold=TORSO_MIN_FRACTION))

    # ------------------------------------------------------------------ 4. ToF
    def tof(self, s: _Sensor, near_XX, near_YY) -> tuple[dict, dict]:
        sc = self.sc
        sc.set_pose(sc.q_home)
        o, R = sc.site_pose(s.site)
        nzx, nzy = s.spec.get("zones", (8, 8))
        ac = -1 + (2 * np.arange(nzx) + 1) / nzx
        bc = 1 - (2 * np.arange(nzy) + 1) / nzy  # row 0 at the top
        A, Bm = np.meshgrid(ac, bc)
        v = s.lens.world(R, A.ravel(), Bm.ravel())
        g, t, wg, wt, other = sc.cast(o, v, s.box, s.window, s.depth)
        cls = sc.classes(g)
        self._collect_window(s, "home", o, R, v, wg, wt, cls <= 0)
        p = o + v * t[:, None]
        zones = []
        for k in range(len(v)):
            kind = "none" if cls[k] < 0 else ("floor" if cls[k] == 0 else GROUPS[cls[k] - 1])
            valid = kind == "floor" and s.range[0] <= t[k] <= s.range[1]
            zones.append(dict(row=k // nzx, col=k % nzx, kind=kind, range_m=float(t[k]) if t[k] >= 0 else None,
                              valid=bool(valid),
                              floor_xy=[float(p[k, 0]), float(p[k, 1])] if kind == "floor" else None,
                              ahead_of_toes_m=float(p[k, 0] - self.x_toe) if kind == "floor" else None))
        fz = [z for z in zones if z["valid"]]
        selfz = Counter(z["kind"] for z in zones if z["kind"] not in ("floor", "none"))
        summ = dict(zones_floor_valid=len(fz), zones_floor_beyond_range=sum(z["kind"] == "floor" and not z["valid"]
                                                                            for z in zones),
                    zones_self=dict(selfz), zones_none=sum(z["kind"] == "none" for z in zones),
                    floor_ahead_of_toes_m=[min(z["ahead_of_toes_m"] for z in fz), max(z["ahead_of_toes_m"] for z in fz)]
                    if fz else None,
                    floor_range_m=[min(z["range_m"] for z in fz), max(z["range_m"] for z in fz)] if fz else None,
                    window_rays=float((wg >= 0).mean()), other_sensor_box_rays=int(other.sum()),
                    site_pos_m=o.tolist(), axis=R[:, 0].tolist(),
                    axis_depression_deg=float(np.degrees(np.arcsin(-R[2, 0]))))
        code, dist = self._floor_raster(s, o, R, near_XX, near_YY)
        cover = code == F_VALID
        # zone footprints on the floor (geometry only, for the plot)
        ak = -1 + 2 * np.arange(nzx + 1) / nzx
        bk = 1 - 2 * np.arange(nzy + 1) / nzy
        Ak, Bk = np.meshgrid(ak, bk)
        vc = s.lens.world(R, Ak.ravel(), Bk.ravel())
        with np.errstate(divide="ignore", invalid="ignore"):
            tt = np.where(vc[:, 2] < -1e-9, -o[2] / vc[:, 2], np.nan)
        tt[tt > s.range[1]] = np.nan
        corners = (o + vc * tt[:, None]).reshape(nzy + 1, nzx + 1, 3)[..., :2]
        plot = dict(o=o, corners=corners, zones=[(z["floor_xy"], z["valid"], z["kind"]) for z in zones])
        return dict(summary=summ, zones=zones, _cover=cover), plot

    # ------------------------------------------------------------------ 5. lidar
    def lidar_scan(self, drop: float = 0.0):
        sc, li = self.sc, self.lidar
        o, R = sc.site_pose(li.site)
        o = o - R[:, 2] * drop
        al = np.arange(LIDAR_RAYS) * (2 * np.pi / LIDAR_RAYS)
        v = np.cos(al)[:, None] * R[:, 0] + np.sin(al)[:, None] * R[:, 1]
        g, t, _, _, other = sc.cast(o, v, li.box)
        return sc.classes(g), t, g, o, R, int(other.sum())

    def lidar_summary(self, cls, t, g, with_sectors=True) -> dict:
        sc = self.sc
        n = len(cls)
        near = (cls >= 1) & (t <= LIDAR_OCCLUSION_RANGE)
        anyr = (cls >= 1) & (t <= self.lidar.range[1])
        out = dict(blocked_1m=float(near.mean()), blocked_in_range=float(anyr.mean()),
                   by_group_1m={GROUPS[k - 1]: float((near & (cls == k)).sum() / n) for k in range(1, 7)},
                   trunk_1m=float((near & np.isin(cls, TRUNK_CODES)).sum() / n),
                   floor_in_range=float(((cls == 0) & (t <= self.lidar.range[1])).mean()),
                   below_min_range=float(((cls >= 0) & (t >= 0) & (t < self.lidar.range[0])).mean()))
        if with_sectors:
            step = 360.0 / n
            secs = []
            for i, j in _runs(near, circular=True):
                idx = np.arange(i, j + 1) if i <= j else np.r_[np.arange(i, n), np.arange(0, j + 1)]
                grp = Counter(GROUPS[c - 1] for c in cls[idx])
                geo = Counter(sc.gname[k] for k in g[idx])
                secs.append(dict(start_deg=i * step - step / 2, end_deg=(j + 1) * step - step / 2,
                                 width_deg=len(idx) * step, groups=dict(grp), main_geom=geo.most_common(1)[0][0],
                                 min_range_m=float(t[idx].min())))
            out["sectors"] = secs
        return out

    # ------------------------------------------------------------------ mount check
    def mount_check(self, s: _Sensor) -> dict:
        sc = self.sc
        if s.box < 0:
            return {}
        o, _ = sc.site_pose(s.site)
        dirs = _fibonacci_sphere(MOUNT_CHECK_RAYS)
        c = sc.d.geom_xpos[s.box]
        Rb = sc.d.geom_xmat[s.box].reshape(3, 3)
        h = sc.m.geom_size[s.box]
        pl = (o - c) @ Rb
        vl = dirs @ Rb
        with np.errstate(divide="ignore", invalid="ignore"):
            tx = np.where(vl > 0, (h - pl) / vl, np.where(vl < 0, (-h - pl) / vl, np.inf))
        t_exit = tx.min(axis=1)
        g, t, *_ = sc.cast(o, dirs, s.box)
        inside = (g >= 0) & (t < t_exit - 1e-6)
        out = {}
        for gid in np.unique(g[inside]):
            k = inside & (g == gid)
            out[sc.gname[gid]] = dict(direction_fraction=float(k.mean()),
                                      max_depth_mm=float(1e3 * (t_exit[k] - t[k]).max()))
        return out

    # ------------------------------------------------------------------ window report
    def window_report(self, s: _Sensor) -> dict:
        sc = self.sc
        panels = {}
        for name, recs in s.window_pts.items():
            gid = sc.geom(name)
            loc = np.vstack([r[1] for r in recs])
            home = [r for r in recs if r[0] == "home"]
            body = int(sc.m.geom_bodyid[gid])
            ent = dict(body=sc.bname[body], share_home=s.window_share.get(name, 0.0),
                       moves_with=None if body == s.body else sc.bname[body],
                       poses=sorted({r[0] for r in recs if not r[0].startswith("gait")}),
                       gait_frames=len({r[0] for r in recs if r[0].startswith("gait")}),
                       extent_panel_frame_mm=(1e3 * (loc.max(0) - loc.min(0))).tolist())
            if home:
                sp = np.vstack([r[2] for r in home])
                dd = np.concatenate([r[3] for r in home])
                ent["home_extent_sensor_frame_mm"] = dict(
                    width_y=float(1e3 * np.ptp(sp[:, 1])), height_z=float(1e3 * np.ptp(sp[:, 2])),
                    depth_x=[float(1e3 * sp[:, 0].min()), float(1e3 * sp[:, 0].max())],
                    distance_mm=[float(1e3 * dd.min()), float(1e3 * dd.max())])
            panels[name] = ent
        return dict(depth_limit_m=s.depth, enclosure=s.enclosure, enclosed=bool(s.enclosure >= ENCLOSURE_MIN),
                    first_hit_shares={k: v for k, v in s.window_share.items()},
                    mounting_panels=[sc.gname[g] for g in np.flatnonzero(s.window)], panels=panels)

    # ------------------------------------------------------------------ outline for the plot
    def outline(self):
        sc = self.sc
        from scipy.spatial import ConvexHull

        sc.set_pose(sc.q_home)
        hulls = []
        for g in range(sc.m.ngeom):
            if g == sc.floor or not sc.visible[g]:
                continue
            pts = sc.geom_points(g)
            if len(pts) < 3:
                continue
            xy = pts[:, :2]
            try:
                xy = xy[ConvexHull(xy).vertices]
            except Exception:  # noqa: BLE001  (degenerate hull: keep the bounding box)
                lo, hi = xy.min(0), xy.max(0)
                xy = np.array([[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]])
            body = sc.bname[int(sc.m.geom_bodyid[g])]
            kind = "foot" if body.endswith("_foot") else ("cover" if sc.cover[g] else "hw")
            hulls.append((kind, xy, float(pts[:, 2].max())))
        return hulls


def screen(xml_path, q_home=None, gait_traj: dict | None = None, fov: dict | None = None, *,
           png_path=None, gait_stride: int = 1, title: str | None = None) -> dict:
    """Run every sensor metric on a compiled model (see the module docstring for definitions).

    ``q_home``: qpos of the nominal stance (default: the model's 'home' keyframe).
    ``gait_traj``: ``gait.generate`` output (uses ``q``); every ``gait_stride``-th frame is screened.
    ``fov``: ``fov_for_parts`` output (default: from the ledger JSON next to the xml).
    Returns a JSON-serialisable dict; writes the top-view PNG if ``png_path`` is given."""
    t0 = time.perf_counter()
    fov = fov or fov_from_ledger(xml_path)
    S = _Screen(xml_path, q_home, fov)
    sc, cam = S.sc, S.cam
    timing = {}
    poses = ["home"] + (["neck_min", "neck_max"] if S.neck_adr is not None else [])
    cam_out, imaps, plot = {}, {}, {}
    for label in poses:
        sc.set_pose(S.pose(label))
        res, imap = S.camera_grid(label)
        res["neck_pitch_rad"] = float(sc.d.qpos[S.neck_adr]) if S.neck_adr is not None else None
        res["centre_column"] = S.centre_column()
        res["floor_rect"] = S.floor_rect()
        res["person"] = S.person()
        cam_out[label] = res
        imaps[label] = (imap, res["blocked"], res["neck_pitch_rad"])
    timing["camera_poses"] = time.perf_counter() - t0

    # home floor raster over the plot window (also gives the camera's near-strip coverage)
    t1 = time.perf_counter()
    sc.set_pose(sc.q_home)
    o, R = sc.site_pose(cam.site)
    VX, VY, KX, JY = S._floor_lattice(view=True)
    vcode, vdist = S._floor_raster(cam, o, R, VX, VY)
    plot.update(VX=VX, VY=VY, vcode=vcode, cam_o=o, cam_R=R)
    nsx, nsy = int(round(NEAR_STRIP / FLOOR_STEP)), int(round(FLOOR_RECT[1] / 2 / FLOOR_STEP))
    near = (KX >= 0) & (KX < nsx) & (JY >= -nsy) & (JY < nsy)  # lattice cells of the near strip
    strip_X = VX[near].reshape(2 * nsy, nsx)
    strip_Y = VY[near].reshape(2 * nsy, nsx)
    cam_cover = (vcode == F_VALID)[near].reshape(2 * nsy, nsx)
    tof_out, tof_plot = {}, []
    covers = {}
    for s in S.tofs:
        r_, p_ = S.tof(s, strip_X, strip_Y)
        covers[s.name] = r_.pop("_cover")
        tof_out[s.name] = r_
        tof_plot.append(p_)
    foot = np.abs(strip_Y - S.y_c) <= S.foot_half + FOOT_PATH_MARGIN
    tof_any = np.zeros_like(cam_cover)
    for c in covers.values():
        tof_any |= c
    cov = {}
    for key, msk in (("strip_1p2m", np.ones_like(cam_cover)), ("foot_path", foot)):
        ent = dict(area_m2=float(msk.sum() * FLOOR_STEP**2), camera=float(cam_cover[msk].mean()),
                   tof_union=float(tof_any[msk].mean()), combined=float((cam_cover | tof_any)[msk].mean()))
        for nm, c in covers.items():
            ent[nm] = float(c[msk].mean())
        if len(covers) == 2:
            a_, b_ = covers.values()
            ent["tof_overlap_of_union"] = float((a_ & b_)[msk].sum() / max((a_ | b_)[msk].sum(), 1))
        cov[key] = ent
    cov["foot_path_half_width_m"] = S.foot_half + FOOT_PATH_MARGIN
    tof_out["near_coverage"] = cov
    plot["tof"] = tof_plot
    timing["floor_view_and_tof"] = time.perf_counter() - t1

    # lidar: as built, lowered-plane sweep
    t2 = time.perf_counter()
    lid_out = None
    if S.lidar is not None:
        sc.set_pose(sc.q_home)
        cls, t, g, lo_, lR, oth = S.lidar_scan(0.0)
        as_built = S.lidar_summary(cls, t, g)
        as_built["other_sensor_box_rays"] = oth
        sweep = []
        for mm in LIDAR_DROPS_MM:
            c_, t_, g_, *_ = S.lidar_scan(mm / 1000.0)
            sm = S.lidar_summary(c_, t_, g_, with_sectors=False)
            sweep.append(dict(drop_mm=mm, blocked_1m=sm["blocked_1m"], trunk_1m=sm["trunk_1m"],
                              by_group_1m=sm["by_group_1m"], blocked_in_range=sm["blocked_in_range"],
                              plane_height_m=float(lo_[2] - mm / 1000.0 * lR[2, 2])))
        tmin = min(e["trunk_1m"] for e in sweep)
        clear_mm = next(e["drop_mm"] for e in sweep if e["trunk_1m"] <= tmin + LIDAR_TRUNK_TOL)
        c_, t_, g_, *_ = S.lidar_scan(clear_mm / 1000.0)
        cleared = S.lidar_summary(c_, t_, g_)
        cleared["drop_mm"] = clear_mm
        lid_out = dict(part=S.lidar.spec["part"], site_pos_m=lo_.tolist(), plane_normal=lR[:, 2].tolist(),
                       as_built=as_built, drop_sweep=sweep, cleared=cleared,
                       mount_check=S.mount_check(S.lidar))
        plot["lidar"] = dict(o=lo_, R=lR, as_built=as_built["sectors"], cleared=cleared["sectors"], drop_mm=clear_mm)
    timing["lidar_home"] = time.perf_counter() - t2

    # gait frames
    t3 = time.perf_counter()
    gait_out = None
    if gait_traj is not None:
        qs = np.asarray(gait_traj["q"])
        frames = list(range(0, len(qs), max(1, int(gait_stride))))
        occ, groups, lid0, lidc, lid_union = [], [], [], [], None
        for k in frames:
            sc.set_pose(qs[k])
            res, _ = S.camera_grid(f"gait{k}")
            occ.append(res["blocked"])
            groups.append(res["by_group"])
            if S.lidar is not None:
                c0, t0_, g0, *_ = S.lidar_scan(0.0)
                lid0.append(S.lidar_summary(c0, t0_, g0, with_sectors=False)["blocked_1m"])
                cc, tc, gc, *_ = S.lidar_scan(lid_out["cleared"]["drop_mm"] / 1000.0)
                lidc.append(S.lidar_summary(cc, tc, gc, with_sectors=False)["blocked_1m"])
                blk = (cc >= 1) & (tc <= LIDAR_OCCLUSION_RANGE)
                lid_union = blk if lid_union is None else (lid_union | blk)
        w = int(np.argmax(occ))
        gait_out = dict(frames=len(frames), stride=int(gait_stride),
                        camera=dict(worst=float(occ[w]), worst_frame=frames[w], mean=float(np.mean(occ)),
                                    best=float(np.min(occ)), worst_by_group=groups[w], per_frame=occ))
        if S.lidar is not None:
            wl = int(np.argmax(lid0))
            gait_out["lidar_as_built"] = dict(worst_1m=float(lid0[wl]), worst_frame=frames[wl],
                                              mean_1m=float(np.mean(lid0)))
            wc = int(np.argmax(lidc))
            gait_out["lidar_cleared"] = dict(drop_mm=lid_out["cleared"]["drop_mm"], worst_1m=float(lidc[wc]),
                                             worst_frame=frames[wc], mean_1m=float(np.mean(lidc)),
                                             union_any_frame_1m=float(lid_union.mean()))
            step = 360.0 / LIDAR_RAYS
            plot["lidar"]["gait_union"] = [(i * step - step / 2, (j + 1) * step - step / 2)
                                           for i, j in _runs(lid_union, circular=True)]
        phases = gait_traj.get("phases")
        if phases is not None:
            gait_out["camera"]["worst_support"] = list(phases[frames[w]])
        info = gait_traj.get("info", {})
        gait_out["spec"] = {k: info[k] for k in ("speed", "stride", "period", "froude") if k in info}
    timing["gait"] = time.perf_counter() - t3

    # mount checks (camera over the neck range) and windows
    mounts = {}
    for label in poses:
        sc.set_pose(S.pose(label))
        mounts[label] = S.mount_check(cam)
    sc.set_pose(sc.q_home)
    o, R = sc.site_pose(cam.site)
    camera = dict(part=cam.spec["part"], projection=cam.lens.projection, hfov_deg=cam.spec["hfov_deg"],
                  vfov_deg=cam.spec["vfov_deg"], range_m=list(cam.range), ideal_range_m=cam.spec.get("ideal_range_m"),
                  site_pos_m=o.tolist(), height_m=float(o[2]), optical_axis=R[:, 0].tolist(),
                  pitch_deg=float(np.degrees(np.arcsin(R[2, 0]))), poses=cam_out, gait=gait_out and gait_out["camera"],
                  window=S.window_report(cam), mount_check=mounts)
    for s in S.tofs:
        tof_out[s.name]["window"] = S.window_report(s)
        tof_out[s.name]["mount_check"] = S.mount_check(s)

    out = dict(model=dict(xml=str(xml_path), hip_height_m=sc.H, toe_tip_x_m=S.x_toe, centre_y_m=S.y_c,
                          feet_half_width_m=S.foot_half,
                          neck_range_rad=list(S.neck_range) if S.neck_range else None),
               fov=fov, camera=camera, tof=tof_out, lidar=lid_out, gait=gait_out,
               method=dict(camera_grid=list(CAMERA_GRID), camera_occlusion_range_m=CAMERA_OCCLUSION_RANGE,
                           floor_step_m=FLOOR_STEP, floor_rect_m=list(FLOOR_RECT), near_strip_m=NEAR_STRIP,
                           person_z_m=list(PERSON_Z), person_dz_m=PERSON_DZ, torso_z_m=list(TORSO_Z),
                           torso_min_fraction=TORSO_MIN_FRACTION, lidar_rays=LIDAR_RAYS,
                           lidar_occlusion_range_m=LIDAR_OCCLUSION_RANGE, window_depth_frac_H=WINDOW_DEPTH_FRAC,
                           window_min_share=WINDOW_MIN_SHARE, enclosure_min=ENCLOSURE_MIN, groups=list(GROUPS)))
    out["summary"], out["flags"] = _summarize(out)
    timing["total"] = time.perf_counter() - t0
    out["timing_s"] = timing
    if png_path is not None:
        t4 = time.perf_counter()
        plot.update(hulls=S.outline(), x_toe=S.x_toe, y_c=S.y_c, imaps=imaps, lens=cam.lens, cam=camera,
                    person=cam_out["home"]["person"]["segments"], title=title or Path(xml_path).stem,
                    summary=out["summary"], foot_half=S.foot_half)
        _plot(Path(png_path), plot)
        out["timing_s"]["png"] = time.perf_counter() - t4
    return _r(out)


def _summarize(out: dict):
    cam = out["camera"]
    home = cam["poses"]["home"]
    s = dict(camera_part=cam["part"], camera_height_m=cam["height_m"],
             camera_blocked={k: v["blocked"] for k, v in cam["poses"].items()},
             camera_blocked_by_group_home={k: v for k, v in home["by_group"].items() if v > 0})
    if cam["gait"]:
        s["camera_blocked_gait"] = dict(worst=cam["gait"]["worst"], mean=cam["gait"]["mean"])
    cc = home["centre_column"]
    s["centre_column_nearest_visible_floor_ahead_of_toes_m"] = cc["visible"]["ahead_of_toes_m"] \
        if cc["visible"] else None
    s["centre_column_nearest_depth_floor_ahead_of_toes_m"] = cc["depth_valid"]["ahead_of_toes_m"] \
        if cc["depth_valid"] else None
    s["floor_rect_not_visible_m2"] = {k: v["floor_rect"]["not_visible_m2"] for k, v in cam["poses"].items()}
    s["floor_rect_no_valid_depth_m2"] = home["floor_rect"]["no_valid_depth_m2"]
    s["person_visible_fraction_home"] = {f"{e['distance_m']:.0f}m@{e['bearing_deg']:+.0f}": e["visible_fraction"]
                                         for e in home["person"]["segments"]}
    s["torso_follow"] = {k: dict(ok=v["person"]["torso_follow"]["ok"],
                                 min_ahead=v["person"]["torso_follow"]["min_fraction_ahead"],
                                 min_30deg=v["person"]["torso_follow"]["min_fraction_30deg"])
                         for k, v in cam["poses"].items()}
    tof = out["tof"]
    s["tof"] = {k: dict(floor_zones=v["summary"]["zones_floor_valid"], self_zones=v["summary"]["zones_self"],
                        floor_ahead_of_toes_m=v["summary"]["floor_ahead_of_toes_m"])
                for k, v in tof.items() if k.startswith("tof_")}
    s["near_strip_coverage"] = {k: tof["near_coverage"][k] for k in ("strip_1p2m", "foot_path")}
    if out["lidar"]:
        li = out["lidar"]
        s["lidar_blocked_1m"] = dict(as_built=li["as_built"]["blocked_1m"], cleared=li["cleared"]["blocked_1m"],
                                     cleared_drop_mm=li["cleared"]["drop_mm"],
                                     cleared_by_group={k: v for k, v in li["cleared"]["by_group_1m"].items() if v > 0})
        if out["gait"]:
            s["lidar_blocked_1m_gait"] = dict(as_built=out["gait"]["lidar_as_built"],
                                              cleared=out["gait"]["lidar_cleared"])
    flags = []
    win = cam["window"]
    for name, p in win["panels"].items():
        ext = p.get("home_extent_sensor_frame_mm", {})
        flags.append(f"camera looks through {name} ({p['share_home']:.0%} of FOV rays): assumed window about "
                     f"{ext.get('width_y', float('nan')):.0f} x {ext.get('height_z', float('nan')):.0f} mm, "
                     f"{ext.get('distance_mm', [0, 0])[0]:.0f}-{ext.get('distance_mm', [0, 0])[1]:.0f} mm from the site"
                     + (f"; the panel moves with the {p['moves_with']} body, union extent in its frame "
                        f"{' x '.join(f'{x:.0f}' for x in p['extent_panel_frame_mm'])} mm" if p["moves_with"] else ""))
    for pose, mc in cam["mount_check"].items():
        for gname, e in mc.items():
            flags.append(f"{pose}: {gname} cuts through the camera box ({e['max_depth_mm']:.1f} mm deep)")
    for k, v in s["torso_follow"].items():
        if not v["ok"]:
            flags.append(f"{k}: torso (0.9-1.5 m) at 1.5-3 m straight ahead only {v['min_ahead']:.0%} visible "
                         f"(< {TORSO_MIN_FRACTION:.0%})")
    if out["lidar"]:
        li = out["lidar"]
        if li["as_built"]["trunk_1m"] > 0.05:
            flags.append(f"lidar scan plane intersects the trunk: {li['as_built']['trunk_1m']:.0%} of the scan blocked "
                         f"by trunk geometry within 1 m as built; trunk blockage stops improving at a "
                         f"{li['cleared']['drop_mm']} mm lower plane")
    cc = home["centre_column"]["visible"]
    if cc:
        flags.append(f"camera centre column first sees the floor {cc['ahead_of_toes_m']:.2f} m ahead of the toe tips "
                     f"(limited by {cc['limited_by']}); {home['floor_rect']['not_visible_m2']:.2f} of "
                     f"{home['floor_rect']['area_m2']:.2f} m2 of the {FLOOR_RECT[0]:.0f} x {FLOOR_RECT[1]:.1f} m "
                     f"rectangle is not visible")
    nc = out["tof"]["near_coverage"]
    flags.append(f"near strip ({NEAR_STRIP:.1f} m ahead of the toes): combined camera + ToF coverage "
                 f"{nc['strip_1p2m']['combined']:.0%} of 1.2 m width, {nc['foot_path']['combined']:.0%} of the foot "
                 f"path (camera {nc['strip_1p2m']['camera']:.0%}, ToF {nc['strip_1p2m']['tof_union']:.0%})")
    for k, v in out["tof"].items():
        if k.startswith("tof_") and v["summary"]["zones_self"]:
            flags.append(f"{k}: {sum(v['summary']['zones_self'].values())} of {len(v['zones'])} zones hit the robot "
                         f"({v['summary']['zones_self']})")
    return s, flags


# ------------------------------------------------------------------------------------------------
# Top-view PNG
# ------------------------------------------------------------------------------------------------
INK, INK2, MUTED, GRIDC, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
GROUP_COLORS = {1: "#eb6834", 2: "#6250d6", 3: "#e87ba4", 4: "#eda100", 5: "#008300", 6: "#1baf7a"}
FLOOR_VALID, FLOOR_NODEPTH, FLOOR_IMG, LIDAR_C, FOV_C = "#6da7ec", "#cde2fb", "#b7d3f6", "#e34948", "#2a78d6"


def _hex(c, alpha=1.0):
    c = c.lstrip("#")
    return tuple(int(c[k : k + 2], 16) / 255 for k in (0, 2, 4)) + (alpha,)


def _plot(path: Path, P: dict):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch, Polygon, Rectangle, Wedge

    fig = plt.figure(figsize=(16.5, 9.6), dpi=110, facecolor=SURF)
    gs = fig.add_gridspec(3, 2, width_ratios=(2.55, 1.0), hspace=0.55, wspace=0.10, left=0.045, right=0.985,
                          top=0.90, bottom=0.15)
    ax = fig.add_subplot(gs[:, 0])
    ax.set_facecolor(SURF)
    VX, VY, code = P["VX"], P["VY"], P["vcode"]
    st = FLOOR_STEP
    rgba = np.zeros(code.shape + (4,))
    rgba[code == F_VALID] = _hex(FLOOR_VALID, 0.85)
    rgba[code == F_NODEPTH] = _hex(FLOOR_NODEPTH, 0.95)
    for k, c in GROUP_COLORS.items():
        rgba[code == F_OCC + k] = _hex(c, 0.55)
    ext = (VX.min() - st / 2, VX.max() + st / 2, VY.min() - st / 2, VY.max() + st / 2)
    ax.imshow(rgba, extent=ext, origin="lower", interpolation="nearest", zorder=0)
    # robot outline (top view, drawn low to high)
    xs = [VX.min(), VX.max()]
    for kind, xy, _ in sorted(P["hulls"], key=lambda h: h[2]):
        fc = {"cover": "#d9d8d2", "hw": "#b8b7b0", "foot": INK2}[kind]
        ax.add_patch(Polygon(xy, closed=True, fc=fc, ec=MUTED, lw=0.35, alpha=0.95 if kind != "cover" else 0.85,
                             zorder=2 if kind != "foot" else 2.5))
        xs += [xy[:, 0].min(), xy[:, 0].max()]
    x_toe, y_c = P["x_toe"], P["y_c"]
    L, W = FLOOR_RECT
    ax.add_patch(Rectangle((x_toe, y_c - W / 2), L, W, fill=False, ec=INK2, lw=1.0, zorder=5))
    ax.add_patch(Rectangle((x_toe, y_c - W / 2), NEAR_STRIP, W, fill=False, ec=INK, lw=1.6, zorder=5))
    ax.text(x_toe + L, y_c + W / 2 + 0.03, f"{L:.0f} x {W:.1f} m from toe tips", ha="right", va="bottom", fontsize=8.5,
            color=INK2)
    ax.text(x_toe + 0.02, y_c - W / 2 - 0.03, f"{NEAR_STRIP:.1f} m near strip", ha="left", va="top", fontsize=8.5,
            color=INK)
    ax.axvline(x_toe, color=INK2, lw=0.6, zorder=1)
    # camera FOV footprint (geometry only)
    o, R, lens = P["cam_o"], P["cam_R"], P["lens"]
    s_ = np.linspace(-1, 1, 81)
    one = np.ones_like(s_)
    border = np.concatenate([np.column_stack([s_, -one]), np.column_stack([one, s_]),
                             np.column_stack([s_[::-1], one]), np.column_stack([-one, s_[::-1]])])
    v = lens.world(R, border[:, 0], border[:, 1])
    with np.errstate(divide="ignore", invalid="ignore"):
        tt = np.where(v[:, 2] < -1e-9, -o[2] / v[:, 2], np.nan)
    fp = o[:2] + v[:, :2] * tt[:, None]
    fp[(fp[:, 0] > ext[1] + 2) | (np.abs(fp[:, 1]) > ext[3] + 2)] = np.nan
    ax.plot(fp[:, 0], fp[:, 1], color=FOV_C, lw=1.3, zorder=6)
    ax.plot([o[0]], [o[1]], marker="o", ms=6, mfc=FOV_C, mec=SURF, zorder=9)
    # ToF zones
    for tp in P.get("tof", []):
        C = tp["corners"]
        for r in range(C.shape[0] - 1):
            for c in range(C.shape[1] - 1):
                quad = np.array([C[r, c], C[r, c + 1], C[r + 1, c + 1], C[r + 1, c]])
                if np.isfinite(quad).all():
                    ax.add_patch(Polygon(quad, closed=True, fill=False, ec=INK, lw=0.25, alpha=0.45, zorder=6))
        fl = np.array([z[0] for z in tp["zones"] if z[1]])
        if len(fl):
            ax.plot(fl[:, 0], fl[:, 1], ls="none", marker=".", ms=3.5, color=INK, zorder=7)
        ax.plot([tp["o"][0]], [tp["o"][1]], marker="s", ms=4.5, mfc=INK, mec=SURF, zorder=9)
    # lidar sectors
    li = P.get("lidar")
    if li:
        lo, lR = li["o"], li["R"]
        yaw = np.degrees(np.arctan2(lR[1, 0], lR[0, 0]))
        for sec in li.get("gait_union", []):
            ax.add_patch(Wedge(lo[:2], LIDAR_OCCLUSION_RANGE, yaw + sec[0], yaw + sec[1], fc=_hex(LIDAR_C, 0.10),
                               ec="none", zorder=3))
        for sec in li["cleared"]:
            ax.add_patch(Wedge(lo[:2], LIDAR_OCCLUSION_RANGE, yaw + sec["start_deg"], yaw + sec["end_deg"],
                               fc=_hex(LIDAR_C, 0.28), ec=LIDAR_C, lw=0.5, zorder=3))
        for sec in li["as_built"]:
            ax.add_patch(Wedge(lo[:2], LIDAR_OCCLUSION_RANGE + 0.08, yaw + sec["start_deg"], yaw + sec["end_deg"],
                               width=0.06, fc=LIDAR_C, ec="none", zorder=4))
        ax.add_patch(Wedge(lo[:2], LIDAR_OCCLUSION_RANGE, 0, 360, fill=False, ec=LIDAR_C, lw=0.5, zorder=4))
        ax.plot([lo[0]], [lo[1]], marker="D", ms=5, mfc=LIDAR_C, mec=SURF, zorder=9)
    # person test positions
    for e in P["person"]:
        x, y = e["xy"]
        ax.plot([x], [y], marker="o", ms=7, mfc=SURF, mec=INK, mew=1.0, zorder=8)
        ax.text(x + 0.05, y + 0.05, f"{e['visible_fraction']:.0%}", fontsize=8, color=INK, zorder=8, clip_on=True)
    xlo = min(xs) - 0.1
    ax.set_xlim(xlo, max(ext[1], max(e["xy"][0] for e in P["person"]) + 0.35))
    ax.set_ylim(ext[2], ext[3])
    ax.set_aspect("equal")
    ax.set_xlabel("x forward (m)", color=INK2)
    ax.set_ylabel("y left (m)", color=INK2)
    ax.grid(color=GRIDC, lw=0.5, zorder=-1)
    ax.tick_params(colors=INK2, labelsize=8.5)
    for sp in ax.spines.values():
        sp.set_color("#c3c2b7")
    sm, cam = P["summary"], P["cam"]
    ax.set_title(f"Top view at home: camera floor visibility, ToF zones, lidar blocked sectors  "
                 f"(camera {cam['part']}, {cam['projection']}, height {cam['height_m']:.2f} m, "
                 f"pitch {cam['pitch_deg']:+.0f} deg)", fontsize=10, color=INK, loc="left")
    handles = [Patch(fc=FLOOR_VALID, label="floor visible, depth in range"),
               Patch(fc=FLOOR_NODEPTH, label="floor visible, outside depth range")]
    handles += [Patch(fc=_hex(GROUP_COLORS[k], 0.55), label=f"floor hidden by {GROUPS[k - 1].replace('_', ' ')}")
                for k in GROUP_COLORS if (code == F_OCC + k).any()]
    handles += [Line2D([], [], color=FOV_C, lw=1.3, label="camera FOV footprint (no occlusion)"),
                Line2D([], [], color=INK, lw=0.5, alpha=0.6, marker=".", label="ToF 8x8 zones (dot: floor hit)"),
                Line2D([], [], ls="none", marker="o", mfc=SURF, mec=INK,
                       label="person test point (% visible, 0.5-1.7 m)")]
    if li:
        handles += [Patch(fc=LIDAR_C, label="lidar blocked <1 m, as built (outer ring)"),
                    Patch(fc=_hex(LIDAR_C, 0.28), ec=LIDAR_C,
                          label=f"lidar blocked <1 m, plane {li['drop_mm']} mm lower")]
        if li.get("gait_union"):
            handles.append(Patch(fc=_hex(LIDAR_C, 0.10), label="lidar blocked in any gait frame (lowered plane)"))
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.045, 0.005), ncol=4, fontsize=8.5, frameon=False,
               labelcolor=INK2)
    # image-plane occlusion maps
    cmap = ListedColormap([SURF, FLOOR_IMG] + [GROUP_COLORS[k] for k in range(1, 7)] + [MUTED])
    for k, (label, (imap, frac, neck)) in enumerate(P["imaps"].items()):
        a2 = fig.add_subplot(gs[k, 1])
        a2.imshow(imap + 1, cmap=cmap, vmin=-0.5, vmax=8.5, origin="lower", extent=(-1, 1, -1, 1),
                  interpolation="nearest")
        a2.set_aspect(lens.aspect())
        nk = f", neck {np.degrees(neck):+.0f} deg" if neck is not None else ""
        a2.set_title(f"camera view, {label}{nk}: {frac:.1%} blocked", fontsize=9, color=INK, loc="left")
        a2.set_xticks([-1, 0, 1], [f"-{cam['hfov_deg'] / 2:.0f}", "0", f"+{cam['hfov_deg'] / 2:.0f} deg"])
        a2.set_yticks([-1, 0, 1], [f"-{cam['vfov_deg'] / 2:.0f}", "0", f"+{cam['vfov_deg'] / 2:.0f}"])
        a2.tick_params(colors=INK2, labelsize=7.5)
        for sp in a2.spines.values():
            sp.set_color("#c3c2b7")
    leg2 = [Patch(fc=FLOOR_IMG, label="floor"), Patch(fc=SURF, ec="#c3c2b7", label="open (no hit)")] + \
        [Patch(fc=GROUP_COLORS[k], label=GROUPS[k - 1].replace("_", " ")) for k in range(1, 7)]
    fig.legend(handles=leg2, loc="lower right", bbox_to_anchor=(0.985, 0.005), ncol=4, fontsize=8, frameon=False,
               labelcolor=INK2)
    cb = sm["camera_blocked"]
    line2 = "camera blocked: " + ", ".join(f"{k} {v:.1%}" for k, v in cb.items())
    if "camera_blocked_gait" in sm:
        line2 += f", gait worst {sm['camera_blocked_gait']['worst']:.1%}"
    if "lidar_blocked_1m" in sm:
        lb = sm["lidar_blocked_1m"]
        line2 += (f"   |   lidar blocked <1 m: as built {lb['as_built']:.0%}, "
                  f"{lb['cleared_drop_mm']} mm lower {lb['cleared']:.0%}")
    fig.suptitle(f"{P['title']}: sensor coverage screen (geometric, screening only)", x=0.045, y=0.975, ha="left",
                 fontsize=13, color=INK, fontweight="bold")
    fig.text(0.045, 0.935, line2, fontsize=9.5, color=INK2, ha="left")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURF)
    plt.close(fig)


# ------------------------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m designlab.sensors", description=__doc__.split("\n\n")[0])
    ap.add_argument("candidate", help="candidate id, e.g. 60-M-S-10")
    ap.add_argument("--out", required=True, help="output directory (build, <id>_sensors.json, <id>_sensors.png)")
    ap.add_argument("--gait", choices=("nominal", "brisk", "none"), default="nominal",
                    help="reference walk to screen (sizing.NOMINAL / sizing.BRISK); default nominal")
    ap.add_argument("--gait-stride", type=int, default=2, help="screen every n-th gait frame (default 2)")
    ap.add_argument("--no-png", action="store_true")
    a = ap.parse_args(argv)
    from . import gait, params, pipeline, sizing

    t0 = time.perf_counter()
    cand = params.parse(a.candidate)
    out = Path(a.out)
    res = pipeline.build_candidate(cand, out)
    t_build = time.perf_counter() - t0
    traj, t_gait = None, 0.0
    if a.gait != "none":
        t1 = time.perf_counter()
        spec = sizing.NOMINAL if a.gait == "nominal" else sizing.BRISK
        traj = gait.generate(res["model"], res["q_home"], res["design"].sk.H, spec)
        t_gait = time.perf_counter() - t1
    xml = res["summary"]["xml"]
    png = None if a.no_png else out / f"{cand.id}_sensors.png"
    r = screen(xml, res["q_home"], traj, fov_from_ledger(xml), png_path=png, gait_stride=a.gait_stride, title=cand.id)
    r["candidate"] = cand.id
    r["gait_name"] = a.gait
    r["timing_s"].update(build=round(t_build, 3), gait_generate=round(t_gait, 3),
                         wall=round(time.perf_counter() - t0, 3))
    path = out / f"{cand.id}_sensors.json"
    path.write_text(json.dumps(r, indent=1))
    s = r["summary"]
    print(f"{cand.id}: camera {s['camera_part']} at {s['camera_height_m']:.2f} m, blocked "
          + ", ".join(f"{k} {v:.1%}" for k, v in s["camera_blocked"].items())
          + (f", gait worst {s['camera_blocked_gait']['worst']:.1%}" if "camera_blocked_gait" in s else ""))
    for f in r["flags"]:
        print("  -", f)
    print(f"  wrote {path} ({r['timing_s']['wall']:.1f} s)")
    return r


if __name__ == "__main__":
    main()
