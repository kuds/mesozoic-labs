"""Review plates: publication-quality figures rendered from compiled candidate models.

The plates are for a design review with a non-specialist decision maker.  They are *screening*
figures: they show what the generator built and measure it, and they are not certified engineering
drawings.  Every picture is a MuJoCo render of the compiled MJCF (``render.Scene``) at its ``home``
keyframe (nominal standing pose).  Nothing is drawn by hand except text, dimension lines, callout
leaders, the grid and the human scale figure, and all of those are placed from model geometry.

Plates (each writes one PNG and returns its path; a sidecar ``<png stem>.json`` next to the PNG
records every number printed on the plate, the scales and the model hashes):

* ``candidate_plate(xml, out_png, title, subtitle)``: side, front and top orthographic views at one
  common physical scale (scale bar in the footer), plus a perspective three-quarter view (not to
  scale).  Dimensions: crown height and hip-pitch-axis height (side), overall nose-to-tail length
  (side), torso width, width across thighs and foot-centre spacing (front), max width (top).
* ``cutaway_plate(xml, out_png, title)``: side and three-quarter views with the covers translucent
  (``cutaway``) and a side view with the covers hidden (``hardware``), with callouts for the
  battery cassette, compute, camera, power distribution, control stack, lidar and ToF sensors,
  neck and tail actuators and the leg actuators (``hw_*_motor``), with masses and parts from the
  ledger.
* ``lineup(xmls_with_labels, out_png, mode, normalize, title)``: side views on one ground line,
  either at one physical scale with a 0.5 m grid and a 1.75 m person drawn from simple shapes
  (``normalize=False``), or each scaled to the same crown height (``normalize=True``, labelled
  height-normalized; sizes are then not comparable).
* ``matrix(xml_grid, row_labels, col_labels, out_png, view, mode, title)``: a grid of side, front or
  top views at one physical scale with per-cell captions (crown height, torso inclination, mass and,
  for front and top views, widths).
* ``silhouette_sheet(xmls_with_labels, out_png)``: compact black side, front and top silhouettes at
  one physical scale.

``subtitle`` / ``title`` arguments may be empty; the plates then describe the candidate from its
parameters (``params``).

Definitions
-----------
As ``DEFINITIONS.md``; everything is measured on the compiled model at ``home``, floor at z = 0.

* Crown height: the highest vertex of the ``cover_skull`` mesh (as ``pipeline.crown_height``).
* Hip-pitch-axis height: z of the ``left_hip_pitch`` / ``right_hip_pitch`` joint anchors (mean) at
  ``home``.  This is 1-4 mm below the ledger's design hip height ``H``, because the stance solve
  rolls each hip by about 1° to set the foot spacing; the plates print the measured value and say
  so in the small print (the check against ``H`` is repeated with the hip-roll joints at zero).
* Overall length (nose to tail) and max width: x and y extent of every visible geom (groups 0-2,
  alpha > 0, floor excluded).  Mesh extents are vertex-exact; boxes use their corners and
  capsules, cylinders, spheres and ellipsoids their exact axis-aligned extreme points (this is
  tighter than ``Scene.bounds``, which takes rotated bounding boxes).
* Torso width: y extent of the trunk body's ``cover_*`` shells, arms and rib hoops excluded.
* Width across thighs: y extent of the ``cover_*thigh*`` fairings.  A model without thigh fairings
  falls back to everything visible on the thigh and femur bodies and the label says so.
* Foot-centre spacing: distance between the ``left_sole`` and ``right_sole`` sites.
* Mass: total model mass (``body_subtreemass`` of the free-joint body), which includes the 6 %
  integration reserve, as the ledger's ``mass_kg``.
* Torso inclination (matrix captions): ledger ``metrics.torso_inclination_deg`` when the ledger has
  it (round-one review ledgers), otherwise the posture's ``chest_deg`` from ``params.POSTURES``,
  which the skeleton solve meets exactly by construction.

Crown height, hip height, mass, foot spacing and the XML SHA-256 are cross-checked against the
ledger ``<id>.json`` next to the XML (tolerances 0.5 mm, 0.5 mm, 1 g, 1 mm).  A mismatch is printed
in red on the plate and stored in the sidecar; it is not silently corrected.  Without a ledger the
plate says so and uses the model alone.

Projection and placement
------------------------
Pixel positions come from the renderer's own OpenGL camera (``scene.camera[0]`` and ``[1]``
averaged, which is the mono camera MuJoCo renders with), so annotations are exact for the
orthographic views and for the perspective three-quarter view.  Checked against segmentation
centroids of individual geoms: agreement within 0.5 px in all four views.  (``render.Scene.project``
is not used: its ``front`` branch maps +y to the image left, whereas MuJoCo's front camera, azimuth
180°, shows the robot's left (+y) on the image right.)  Orthographic views are placed
by choosing the camera centre so that a reference world point lands on a chosen pixel; the ground
row of each view is the projection of z = 0, which is how views share one ground line.  Renders are
made at ``SS`` (2) times the output size and box-downsampled; the robot's coverage (alpha) comes
from a segmentation render, so silhouettes and composited tiles have anti-aliased edges and the
floor never counts as robot.

Callout anchors are the visible pixel nearest the centroid of the named geoms' visible pixels in a
``hardware``-mode segmentation rendered with the same camera (so they point at what is visible
through the translucent covers).  A part with fewer than ``MIN_VISIBLE_PX`` visible pixels is
anchored at its projected centre and its dot is drawn hollow (hidden behind other hardware).

The three-quarter view keeps the renderer's floor and the key light's shadow but fades the floor
towards the page background with floor-plane distance from the model's footprint and shadow, and at
the tile border (cosmetic only; ``FLOOR_KEEP``, ``FLOOR_FADE``).  Callout labels are placed in bands
above and below each drawing, in the anchors' x order, by an isotonic least-squares spread; adjacent
labels are swapped while that removes leader crossings, and a band that is still too full gets a
second tier (rows grow to fit).

Limitations
-----------
* Screening figures, not certified drawings: dimensions are measured on the generated model, which
  is itself an estimate (see the ledger ``basis`` column); they are not manufacturing tolerances.
* The three-quarter view is perspective and is never to scale.
* The 1.75 m person in common-scale lineups is a schematic made of simple shapes, not a model.
* OSMesa allows one live ``mujoco.Renderer`` per process: every function opens one ``Scene`` at a
  time and closes it (``with _scene(...)``).  Do not call these functions while another renderer
  is open in the same process.
"""

from __future__ import annotations

import copy
import hashlib
import json
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from . import catalog, render
from .params import FAMILIES, POSTURES, PROPORTIONS, WIDTHS, parse

# ------------------------------------------------------------------------------------------------
# Style
# ------------------------------------------------------------------------------------------------
PLATE_W = 2400  #: full-plate width, px
MARGIN = 80  #: outer margin, px
SS = 2  #: supersampling factor of every render
MIN_VISIBLE_PX = 12  #: render pixels a callout part needs to count as visible
FLOOR_KEEP = 0.6  #: three-quarter view: floor contrast kept under the robot and its shadow (0 = page background)
FLOOR_FADE = 0.08  #: three-quarter view: e-folding floor-plane distance of the fade, fraction of the model size

BG = render.BG
INK = (30, 34, 32)
INK2 = (84, 90, 88)
INK3 = (128, 134, 131)
DIM = (28, 80, 138)
ACCENT = (186, 82, 30)
RULE = (212, 209, 200)
GRID = (226, 224, 216)
GRID_MAJOR = (204, 202, 193)
SIL = (22, 24, 23)
PERSON_FILL = (212, 213, 208)
PERSON_LINE = (146, 149, 145)
WARN = (170, 40, 30)

F_TITLE, F_SUB, F_PANEL, F_PANEL_D = 50, 28, 23, 20
F_DIM_N, F_DIM_V = 20, 24
F_CALL_T, F_CALL_D = 21, 18
CALLOUT_WRAP = 250  #: max width of a callout detail line, px
F_BODY, F_CAP, F_SMALL = 26, 21, 18
PANEL_H = 42  #: height reserved for a panel label

_GEOM = int(mujoco.mjtObj.mjOBJ_GEOM)
_AXES = {  # orthographic views: image-right and image-up world directions (checked after each render)
    "side": (np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])),
    "front": (np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])),
    "top": (np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])),
}
_VIEW_TEXT = {"side": "right side · orthographic", "front": "orthographic", "top": "orthographic",
              "threequarter": "perspective · not to scale"}


@lru_cache(maxsize=None)
def _font(size: int, bold: bool = False):
    return render.font(int(size), bold)


_MEAS = ImageDraw.Draw(Image.new("RGB", (8, 8)))


def _tw(text: str, font) -> float:
    return _MEAS.textlength(text, font=font)


def _lh(font) -> int:
    a, d = font.getmetrics()
    return a + d


def _mm(v: float) -> str:
    return f"{1000.0 * v:.0f}"


def _kg(v: float) -> str:
    return f"{v:.1f}"


def _wrap(text: str, font, max_w: float) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        t = f"{cur} {w}".strip()
        if cur and _tw(t, font) > max_w:
            lines.append(cur)
            cur = w
        else:
            cur = t
    if cur:
        lines.append(cur)
    return lines or [""]


def _fit_font(text: str, max_w: float, size: int, bold: bool, min_size: int):
    while size > min_size and _tw(text, _font(size, bold)) > max_w:
        size -= 2
    return _font(size, bold)


# ------------------------------------------------------------------------------------------------
# Model geometry (exact extents, measurements, ledger cross-checks)
# ------------------------------------------------------------------------------------------------
def _unit_dirs() -> np.ndarray:
    v = [np.array(p, float) for p in
         [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]]
    for a in (1, -1):
        for b in (1, -1):
            v += [np.array([a, b, 0.0]), np.array([a, 0.0, b]), np.array([0.0, a, b])]
            for c in (1, -1):
                v.append(np.array([a, b, c], float))
    return np.array([x / np.linalg.norm(x) for x in v])


_DIRS = _unit_dirs()


def _geom_points(m, d, g: int) -> np.ndarray:
    """World points of geom ``g`` whose axis-aligned extremes are those of the geom itself."""
    t = m.geom_type[g]
    c = d.geom_xpos[g]
    R = d.geom_xmat[g].reshape(3, 3)
    s = m.geom_size[g]
    if t == mujoco.mjtGeom.mjGEOM_MESH:
        mid = m.geom_dataid[g]
        a, n = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
        return c + m.mesh_vert[a:a + n] @ R.T
    if t == mujoco.mjtGeom.mjGEOM_BOX:
        sg = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float)
        return c + (sg * s[:3]) @ R.T
    if t == mujoco.mjtGeom.mjGEOM_SPHERE:
        return c + s[0] * _DIRS
    if t == mujoco.mjtGeom.mjGEOM_CAPSULE:
        a = R[:, 2]
        return np.vstack([e + s[0] * _DIRS for e in (c + a * s[1], c - a * s[1])])
    if t == mujoco.mjtGeom.mjGEOM_CYLINDER:
        a, r, h = R[:, 2], s[0], s[1]
        ang = np.linspace(0, 2 * np.pi, 33)[:-1]
        ring = np.outer(np.cos(ang), R[:, 0]) + np.outer(np.sin(ang), R[:, 1])
        pts = [c + a * h + r * ring, c - a * h + r * ring]
        for e in np.eye(3):
            for sgn in (1, -1):
                perp = sgn * e - np.dot(sgn * e, a) * a
                nrm = np.linalg.norm(perp)
                if nrm > 1e-9:
                    pts.append(np.array([c + a * h + r * perp / nrm, c - a * h + r * perp / nrm]))
        return np.vstack(pts)
    if t == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
        return c + (_DIRS * s[:3]) @ R.T
    return c[None, :].copy()


def _name(m, objtype, i: int) -> str:
    return mujoco.mj_id2name(m, objtype, i) or ""


def _load(xml) -> tuple:
    m = mujoco.MjModel.from_xml_path(str(xml))
    d = mujoco.MjData(m)
    k = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home")
    if k >= 0:
        mujoco.mj_resetDataKeyframe(m, d, k)
    mujoco.mj_forward(m, d)
    return m, d


def _visible_geoms(m) -> list[int]:
    return [g for g in range(m.ngeom) if m.geom_group[g] <= 2 and m.geom_rgba[g, 3] > 0
            and m.geom_type[g] != mujoco.mjtGeom.mjGEOM_PLANE]


def _sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read_ledger(xml) -> dict:
    p = Path(xml).with_suffix(".json")
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return {}


@dataclass
class _Meas:
    """Measurements of one compiled model at its home keyframe (see the module docstring)."""

    xml: str
    cid: str
    sha256: str
    ledger: dict
    lo: np.ndarray
    hi: np.ndarray
    nose: np.ndarray
    tail: np.ndarray
    wlo: np.ndarray  # visible point with the smallest y (robot's right)
    whi: np.ndarray  # visible point with the largest y (robot's left)
    crown: float
    crown_pt: np.ndarray
    hip: np.ndarray  # midpoint of the two hip-pitch joint anchors
    torso_lo: np.ndarray
    torso_hi: np.ndarray
    thigh_lo: np.ndarray
    thigh_hi: np.ndarray
    thigh_fairing: bool
    soles: np.ndarray  # (left, right) sole-site positions
    mass: float
    hip_roll_deg: float = 0.0  # largest |hip roll| at home (the stance solve rolls the hips)
    hip_zero_roll: float | None = None  # hip-pitch anchor height with the hip-roll joints at zero
    checks: dict = field(default_factory=dict)

    @property
    def length(self) -> float:
        return float(self.hi[0] - self.lo[0])

    @property
    def max_width(self) -> float:
        return float(self.hi[1] - self.lo[1])

    @property
    def top(self) -> float:
        return float(self.hi[2])

    @property
    def hip_height(self) -> float:
        return float(self.hip[2])

    @property
    def torso_width(self) -> float:
        return float(self.torso_hi[1] - self.torso_lo[1])

    @property
    def thigh_width(self) -> float:
        return float(self.thigh_hi[1] - self.thigh_lo[1])

    @property
    def foot_spacing(self) -> float:
        return float(abs(self.soles[0][1] - self.soles[1][1]))

    @property
    def center(self) -> np.ndarray:
        return 0.5 * (self.lo + self.hi)

    @property
    def name(self) -> str:
        return Path(self.xml).name

    def torso_inclination(self) -> tuple[float, str]:
        met = self.ledger.get("metrics") or {}
        if met.get("torso_inclination_deg") is not None:
            return float(met["torso_inclination_deg"]), "ledger metrics"
        cand = self.ledger.get("candidate") or {}
        post = cand.get("posture")
        if post is None:
            try:
                post = parse(self.cid).posture
            except (ValueError, IndexError):
                return float("nan"), "unknown"
        return float(POSTURES[int(post)]["chest_deg"]), "posture definition (params.POSTURES)"

    def facts(self) -> dict:
        incl, src = self.torso_inclination()
        return dict(id=self.cid, xml=self.xml, xml_sha256=self.sha256, mass_kg=self.mass, crown_m=self.crown,
                    hip_pitch_axis_m=self.hip_height, length_m=self.length, max_width_m=self.max_width,
                    torso_width_m=self.torso_width, thigh_width_m=self.thigh_width,
                    thigh_width_source="thigh fairings" if self.thigh_fairing else "thigh/femur geometry (no fairing)",
                    foot_centre_spacing_m=self.foot_spacing, torso_inclination_deg=incl,
                    torso_inclination_source=src, ledger_checks=self.checks)


def _measure(xml) -> _Meas:
    xml = str(xml)
    m, d = _load(xml)
    vis = _visible_geoms(m)
    pts = {g: _geom_points(m, d, g) for g in vis}
    allp = np.vstack(list(pts.values()))
    lo, hi = allp.min(0), allp.max(0)
    gname = {g: _name(m, mujoco.mjtObj.mjOBJ_GEOM, g) for g in vis}
    bname = {g: _name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g])) for g in vis}
    sk = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "cover_skull")
    if sk >= 0:
        sp = _geom_points(m, d, sk)
        crown_pt = sp[np.argmax(sp[:, 2])]
    else:  # no skull cover: highest visible point
        crown_pt = allp[np.argmax(allp[:, 2])]
    hips = []
    for side in ("left", "right"):
        j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_hip_pitch")
        if j >= 0:
            hips.append(d.xanchor[j].copy())
    hip = np.mean(hips, axis=0) if hips else np.array([0.0, 0.0, np.nan])

    def ext(gs):
        if not gs:
            return None
        p = np.vstack([pts[g] for g in gs])
        return p[np.argmin(p[:, 1])], p[np.argmax(p[:, 1])]

    torso = ext([g for g in vis if bname[g] == "trunk" and gname[g].startswith("cover_")
                 and "arm" not in gname[g] and "rib_hoop" not in gname[g]])
    thigh = ext([g for g in vis if gname[g].startswith("cover_") and "thigh" in gname[g]])
    fairing = thigh is not None
    if thigh is None:
        thigh = ext([g for g in vis if bname[g].endswith(("_thigh", "_femur"))])
    if torso is None:
        torso = (allp[np.argmin(allp[:, 1])], allp[np.argmax(allp[:, 1])])
    soles = []
    for side in ("left", "right"):
        s = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f"{side}_sole")
        soles.append(d.site_xpos[s].copy() if s >= 0 else np.full(3, np.nan))
    root = int(m.jnt_bodyid[0]) if m.njnt else 1
    mass = float(m.body_subtreemass[root])
    # The stance solve rolls each hip slightly to put the feet at the nominal spacing, which moves the
    # hip-pitch axis a few millimetres below the design hip height H; measure it with zero roll too.
    rolls = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{s}_hip_roll") for s in ("left", "right")]
    rolls = [int(m.jnt_qposadr[j]) for j in rolls if j >= 0]
    pitch = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{s}_hip_pitch") for s in ("left", "right")]
    pitch = [j for j in pitch if j >= 0]
    roll_deg, zero_roll = 0.0, None
    if rolls and pitch:
        roll_deg = float(max(abs(np.degrees(d.qpos[a])) for a in rolls))
        q = d.qpos.copy()
        d.qpos[rolls] = 0.0
        mujoco.mj_kinematics(m, d)
        zero_roll = float(np.mean([d.xanchor[j][2] for j in pitch]))
        d.qpos[:] = q
        mujoco.mj_forward(m, d)
    led = _read_ledger(xml)
    sha = _sha256(xml)
    cid = led.get("id") or Path(xml).stem
    g = _Meas(xml=xml, cid=cid, sha256=sha, ledger=led, lo=lo, hi=hi, nose=allp[np.argmax(allp[:, 0])],
              tail=allp[np.argmin(allp[:, 0])], wlo=allp[np.argmin(allp[:, 1])], whi=allp[np.argmax(allp[:, 1])],
              crown=float(crown_pt[2]), crown_pt=crown_pt, hip=hip, torso_lo=torso[0], torso_hi=torso[1],
              thigh_lo=thigh[0], thigh_hi=thigh[1], thigh_fairing=fairing, soles=np.array(soles), mass=mass,
              hip_roll_deg=roll_deg, hip_zero_roll=zero_roll)
    g.checks = _ledger_checks(g)
    return g


def _ledger_checks(g: _Meas) -> dict:
    """Compare model measurements with the ledger written next to the XML."""
    led = g.ledger
    if not led:
        return {"ledger": "missing"}
    summ = led.get("summary") or {}
    out = {}

    def chk(key, model_val, led_val, tol):
        if led_val is None:
            return
        ok = bool(abs(float(model_val) - float(led_val)) <= tol)
        out[key] = dict(model=float(model_val), ledger=float(led_val), ok=ok)

    chk("crown_m", g.crown, summ.get("crown"), 5e-4)
    chk("hip_height_m", g.hip_height, led.get("H"), 5e-4)
    hh = out.get("hip_height_m")
    if hh and not hh["ok"] and g.hip_zero_roll is not None and abs(g.hip_zero_roll - hh["ledger"]) <= 5e-4:
        # design H is the same axis before the stance hip roll: explained, not an inconsistency
        hh.update(ok=True, model_zero_roll=g.hip_zero_roll, stance_hip_roll_deg=g.hip_roll_deg,
                  explained="ledger H is the design hip height; the stance solve rolls the hips")
    chk("mass_kg", g.mass, led.get("mass_kg"), 1e-3)
    chk("foot_centre_spacing_m", g.foot_spacing, (led.get("hip") or {}).get("foot_spacing"), 1e-3)
    if summ.get("xml_sha256"):
        out["xml_sha256"] = dict(model=g.sha256, ledger=summ["xml_sha256"], ok=g.sha256 == summ["xml_sha256"])
    return out


def _check_failures(g: _Meas) -> list[str]:
    if g.checks.get("ledger") == "missing":
        return [f"{g.cid}: no ledger next to the XML (values from the model only)"]
    return [f"{g.cid}: ledger {k} {v['ledger']} vs model {v['model']}" for k, v in g.checks.items()
            if isinstance(v, dict) and not v.get("ok", True)]


def _hip_note(g: _Meas) -> str:
    """Small-print note when the measured hip-pitch axis differs from the ledger's design H."""
    hh = g.checks.get("hip_height_m") or {}
    if "explained" not in hh:
        return ""
    return (f"Hip-pitch axis measured at the home pose: {_mm(g.hip_height)} mm. The ledger's design hip height H "
            f"({_mm(hh['ledger'])} mm) is the same axis before the stance solve rolls each hip by "
            f"{g.hip_roll_deg:.2f}° to set the foot spacing.")


# ------------------------------------------------------------------------------------------------
# Rendering
# ------------------------------------------------------------------------------------------------
@contextmanager
def _scene(xml, w: int, h: int):
    """One ``render.Scene`` of ``w`` x ``h`` output pixels (rendered at ``SS`` x that size)."""
    sc = render.Scene(str(xml), int(w) * SS, int(h) * SS)
    try:
        yield sc
    finally:
        sc.close()


class _Cam:
    """Mono OpenGL camera of the renderer's last ``update_scene`` (average of the two eyes)."""

    def __init__(self, sc):
        c0, c1 = sc.r.scene.camera[0], sc.r.scene.camera[1]

        def av(a, b):
            return 0.5 * (np.asarray(a, float) + np.asarray(b, float))

        self.pos = av(c0.pos, c1.pos)
        f, u = av(c0.forward, c1.forward), av(c0.up, c1.up)
        self.fwd, self.up = f / np.linalg.norm(f), u / np.linalg.norm(u)
        self.right = np.cross(self.fwd, self.up)
        self.center = 0.5 * (float(c0.frustum_center) + float(c1.frustum_center))
        self.bottom, self.top, self.near = float(c0.frustum_bottom), float(c0.frustum_top), float(c0.frustum_near)
        self.ortho = bool(c0.orthographic)
        self.W, self.H = sc.W, sc.H

    def project(self, P) -> np.ndarray:
        """Continuous render-pixel coordinates (origin at the top-left image corner) of world points."""
        P = np.atleast_2d(np.asarray(P, float))
        rel = P - self.pos
        x, y, z = rel @ self.right, rel @ self.up, rel @ self.fwd
        if not self.ortho:
            x, y = self.near * x / z, self.near * y / z
        hh = 0.5 * (self.top - self.bottom)
        hw = hh * self.W / self.H
        return np.c_[self.W / 2 * (1 + (x - self.center) / hw),
                     self.H / 2 * (1 - (y - 0.5 * (self.top + self.bottom)) / hh)]


def _segment(sc) -> np.ndarray:
    """Segmentation (objid, objtype) of the scene of the last ``Scene.render`` call."""
    sc.r.enable_segmentation_rendering()
    try:
        return sc.r.render().copy()
    finally:
        sc.r.disable_segmentation_rendering()


def _robot_mask(seg: np.ndarray, floor: int) -> np.ndarray:
    return (seg[..., 1] == _GEOM) & (seg[..., 0] >= 0) & (seg[..., 0] != floor)


def _reduce_mask(mask: np.ndarray, w: int, h: int) -> np.ndarray:
    a = mask[:h * SS, :w * SS].astype(np.float32)
    return a.reshape(h, SS, w, SS).mean(axis=(1, 3))


def _ink_image(alpha: np.ndarray, ink=SIL) -> Image.Image:
    bg = np.array(BG, np.float32)
    out = bg + alpha[..., None] * (np.array(ink, np.float32) - bg)
    return Image.fromarray(np.clip(out + 0.5, 0, 255).astype(np.uint8))


@dataclass
class _Tile:
    img: Image.Image  # RGB, page background where there is no robot
    alpha: np.ndarray  # robot coverage 0..1 (h, w)
    cam: _Cam
    view: str
    ctr: np.ndarray
    extent: float
    w: int
    h: int
    ox: int = 0
    oy: int = 0

    def px(self, P) -> np.ndarray:
        """Canvas coordinates (continuous) of world points."""
        return self.cam.project(P) / SS + np.array([self.ox, self.oy], float)


def _ortho(sc, view: str, mode: str, s: float, ref, ref_px, w: int, h: int, depth_pt) -> _Tile:
    """Orthographic ``view`` at ``s`` px/m with world point ``ref`` on tile pixel ``ref_px``.

    The tile is the top-left ``w`` x ``h`` output pixels of the scene's render."""
    if w * SS > sc.W or h * SS > sc.H:
        raise ValueError("tile larger than the scene's renderer")
    right, up = _AXES[view]
    fwd = np.cross(up, right)
    sr = s * SS
    a = (ref_px[0] * SS - sc.W / 2.0) / sr
    b = (sc.H / 2.0 - ref_px[1] * SS) / sr
    ctr = np.asarray(ref, float) - a * right - b * up
    ctr = ctr + fwd * float(np.dot(np.asarray(depth_pt, float) - ctr, fwd))
    ext = sc.H / sr
    rgb = sc.render(view, "silhouette" if mode == "silhouette" else mode, center=ctr, extent=ext, floor=False)
    cam = _Cam(sc)
    err = float(np.abs(cam.project(ref)[0] / SS - np.asarray(ref_px, float)).max())
    if err > 0.25:
        raise RuntimeError(f"{view} view placement off by {err:.2f} px (render.Scene camera convention changed?)")
    seg = _segment(sc)
    alpha = _reduce_mask(_robot_mask(seg, sc.floor), w, h)
    if mode == "silhouette":
        img = _ink_image(alpha)
    else:
        img = Image.fromarray(np.ascontiguousarray(rgb[:h * SS, :w * SS])).reduce(SS)
    return _Tile(img, alpha, cam, view, ctr, ext, w, h)


def _model_points(sc) -> np.ndarray:
    m, d = sc.m, sc.d
    return np.vstack([_geom_points(m, d, g) for g in _visible_geoms(m)])


def _perspective(sc, view: str, mode: str, box, floor: bool = True) -> _Tile:
    """Perspective ``view`` framed so the model's projected extent fills ``box`` (x0, y0, x1, y1, tile
    px; the tile is the whole scene) and is centred on it."""
    w, h = sc.W // SS, sc.H // SS
    P = _model_points(sc)
    lo, hi = P.min(0), P.max(0)
    ctr = 0.5 * (lo + hi)
    ext = float(np.max(hi - lo))
    sc.render(view, "silhouette", center=ctr, extent=ext)  # probe: camera axes and distance law
    cam = _Cam(sc)
    kd = float(np.linalg.norm(cam.pos - ctr)) / ext
    tan_h = cam.top / cam.near
    tx, ty = 0.5 * (box[0] + box[2]) * SS, 0.5 * (box[1] + box[3]) * SS
    bw, bh = (box[2] - box[0]) * SS, (box[3] - box[1]) * SS
    for _ in range(80):
        cam.pos = ctr - kd * ext * cam.fwd
        uv = cam.project(P)
        (u0, v0), (u1, v1) = uv.min(0), uv.max(0)
        k = max((u1 - u0) / bw, (v1 - v0) / bh)
        du, dv = 0.5 * (u0 + u1) - tx, 0.5 * (v0 + v1) - ty
        if abs(k - 1) < 2e-4 and abs(du) < 0.1 and abs(dv) < 0.1:
            break
        mpp = 2 * kd * ext * tan_h / sc.H
        ctr = ctr + cam.right * du * mpp - cam.up * dv * mpp
        ext *= k ** 0.85
    rgb = sc.render(view, mode, center=ctr, extent=ext, floor=floor)
    cam2 = _Cam(sc)
    if np.linalg.norm(cam2.pos - (ctr - kd * ext * cam.fwd)) > 1e-5 * (1.0 + kd * ext):
        warnings.warn("perspective framing model disagrees with render.Scene's camera; framing may be loose")
    seg = _segment(sc)
    robot = _robot_mask(seg, sc.floor)
    if floor and mode != "silhouette":
        rgb = _fade_floor(sc, cam2, seg, rgb, P, w, h)
    alpha = _reduce_mask(robot, w, h)
    img = _ink_image(alpha) if mode == "silhouette" else Image.fromarray(rgb).reduce(SS)
    return _Tile(img, alpha, cam2, view, ctr, ext, w, h)


def _shadow_footprint(sc, P: np.ndarray) -> np.ndarray:
    """Floor-plane (xy) points under the model: its vertical projection and its shadow from every
    shadow-casting light (projected from the light position, or along the direction for a
    directional light)."""
    m, d = sc.m, sc.d
    out = [P[:, :2]]
    for i in range(m.nlight):
        if not m.light_castshadow[i]:
            continue
        lp, ld = d.light_xpos[i], d.light_xdir[i]
        if hasattr(m, "light_type"):
            directional = int(m.light_type[i]) == int(mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)
        else:
            directional = bool(m.light_directional[i])
        if directional and ld[2] < -1e-6:
            out.append(P[:, :2] + P[:, 2:3] * (ld[:2] / -ld[2]))
        elif not directional and lp[2] > P[:, 2].max() + 1e-6:
            k = lp[2] / (lp[2] - P[:, 2])
            out.append(lp[:2] + (P[:, :2] - lp[:2]) * k[:, None])
    return np.vstack(out)


def _fade_floor(sc, cam: _Cam, seg: np.ndarray, rgb: np.ndarray, P: np.ndarray, w: int, h: int) -> np.ndarray:
    """Blend floor pixels towards the page background: full fade beyond the model's footprint and
    shadow (floor-plane distance, e-folding ``FLOOR_FADE`` x model size) and at the tile border."""
    from scipy import ndimage

    floorm = (seg[..., 1] == _GEOM) & (seg[..., 0] == sc.floor)
    vv, uu = np.nonzero(floorm)
    if not len(vv) or cam.ortho:
        return rgb
    hh = 0.5 * (cam.top - cam.bottom)
    hw = hh * cam.W / cam.H
    xn = (2.0 * (uu + 0.5) / cam.W - 1.0) * hw + cam.center
    yn = (1.0 - 2.0 * (vv + 0.5) / cam.H) * hh + 0.5 * (cam.top + cam.bottom)
    rays = cam.near * cam.fwd + xn[:, None] * cam.right + yn[:, None] * cam.up
    with np.errstate(divide="ignore", invalid="ignore"):
        t = -cam.pos[2] / rays[:, 2]
    fp = cam.pos[:2] + t[:, None] * rays[:, :2]
    F = _shadow_footprint(sc, P)
    size = float(np.max(P.max(0) - P.min(0)))
    res = size / 250.0
    lo2 = F.min(0) - size
    n2 = np.ceil((F.max(0) + size - lo2) / res).astype(int) + 1
    free = np.ones((n2[1], n2[0]), bool)
    ij = np.floor((F - lo2) / res).astype(int)
    free[ij[:, 1], ij[:, 0]] = False
    dist = ndimage.distance_transform_edt(free) * res
    q = np.floor((fp - lo2) / res).astype(int)
    ok = (t > 0) & (q[:, 0] >= 0) & (q[:, 0] < n2[0]) & (q[:, 1] >= 0) & (q[:, 1] < n2[1])
    dd = np.full(len(fp), np.inf)
    dd[ok] = dist[q[ok, 1], q[ok, 0]]
    wgt = FLOOR_KEEP * np.exp(-dd / (FLOOR_FADE * size))
    # vignette: the floor reaches the page background before the tile border (no visible frame)
    edge = np.minimum(np.minimum(uu, cam.W - 1 - uu), np.minimum(vv, cam.H - 1 - vv)) / (SS * 0.12 * min(w, h))
    edge = np.clip(edge, 0.0, 1.0)
    wgt = wgt * edge * edge * (3.0 - 2.0 * edge)
    bgc = np.array(BG, np.float32)
    f = rgb.astype(np.float32)
    f[vv, uu] = bgc + wgt[:, None] * (f[vv, uu] - bgc)
    return np.clip(f + 0.5, 0, 255).astype(np.uint8)


def _anchors(sc, tile: _Tile, groups: dict) -> dict:
    """Callout anchors: for each key, the visible pixel nearest the centroid of the visible pixels of
    its geoms (``groups[key]`` = list of geom names) in a hardware-mode segmentation with the tile's
    camera.  Returns key -> (canvas xy, visible render pixels, hidden flag)."""
    sc.render(tile.view, "hardware", center=tile.ctr, extent=tile.extent, floor=False)
    cam = _Cam(sc)
    if np.linalg.norm(cam.pos - tile.cam.pos) > 1e-6 * (1 + np.linalg.norm(tile.cam.pos)):
        raise RuntimeError("hardware render used a different camera than the displayed tile")
    seg = _segment(sc)[:tile.h * SS, :tile.w * SS]
    ids, typ = seg[..., 0], seg[..., 1]
    out = {}
    for key, names in groups.items():
        gids = [mujoco.mj_name2id(sc.m, mujoco.mjtObj.mjOBJ_GEOM, n) for n in names]
        gids = [g for g in gids if g >= 0]
        if not gids:
            continue
        mask = (typ == _GEOM) & np.isin(ids, gids)
        n = int(mask.sum())
        c = np.mean([sc.d.geom_xpos[g] for g in gids], axis=0)
        pc = tile.px(c)[0]
        if n >= MIN_VISIBLE_PX:
            ys, xs = np.nonzero(mask)
            cx, cy = xs.mean(), ys.mean()
            k = int(np.argmin((xs - cx) ** 2 + (ys - cy) ** 2))
            p = np.array([(xs[k] + 0.5) / SS + tile.ox, (ys[k] + 0.5) / SS + tile.oy])
            out[key] = (p, n, False, c, pc)
        else:
            out[key] = (pc, n, True, c, pc)
    return out


# ------------------------------------------------------------------------------------------------
# Canvas and drawing primitives
# ------------------------------------------------------------------------------------------------
class _Canvas:
    def __init__(self, w: int, h: int):
        self.w, self.h = int(w), int(h)
        self.im = Image.new("RGB", (self.w, self.h), BG)
        self.cover = np.zeros((self.h, self.w), np.float32)
        self.dr = ImageDraw.Draw(self.im)

    def place(self, tile: _Tile, ox: int, oy: int):
        """Composite a tile (rendered over the page background) at integer offset (ox, oy)."""
        tile.ox, tile.oy = int(ox), int(oy)
        x0, y0 = max(0, tile.ox), max(0, tile.oy)
        x1, y1 = min(self.w, tile.ox + tile.w), min(self.h, tile.oy + tile.h)
        if x1 <= x0 or y1 <= y0:
            return
        sx, sy = x0 - tile.ox, y0 - tile.oy
        t = np.asarray(tile.img, np.float32)[sy:sy + y1 - y0, sx:sx + x1 - x0]
        a = tile.alpha[sy:sy + y1 - y0, sx:sx + x1 - x0]
        base = np.asarray(self.im, np.float32)[y0:y1, x0:x1]
        out = t + (base - np.array(BG, np.float32)) * (1.0 - a[..., None])
        self.im.paste(Image.fromarray(np.clip(out + 0.5, 0, 255).astype(np.uint8)), (x0, y0))
        self.cover[y0:y1, x0:x1] = np.maximum(self.cover[y0:y1, x0:x1], a)
        self.dr = ImageDraw.Draw(self.im)

    def ghosted(self, draw_fn, strength: float = 0.75):
        """Draw with ``draw_fn(ImageDraw)`` on a layer that is faded where the robot is (extension
        lines that cross the model stay readable without hiding it)."""
        layer = Image.new("RGBA", (self.w, self.h), (0, 0, 0, 0))
        draw_fn(ImageDraw.Draw(layer))
        arr = np.asarray(layer).copy()
        arr[..., 3] = (arr[..., 3].astype(np.float32) * (1.0 - strength * self.cover)).astype(np.uint8)
        self.im = Image.alpha_composite(self.im.convert("RGBA"), Image.fromarray(arr)).convert("RGB")
        self.dr = ImageDraw.Draw(self.im)

    def save(self, out_png) -> str:
        out_png = Path(out_png)
        out_png.parent.mkdir(parents=True, exist_ok=True)
        self.im.save(out_png, optimize=True)
        return str(out_png)


def _xy(p) -> tuple:
    """Continuous pixel coordinate -> PIL drawing coordinate (pixel centres at integers)."""
    return float(p[0]) - 0.5, float(p[1]) - 0.5


def _arrowhead(dr, tip, direction, color, length: float = 13.0, half: float = 4.6):
    u = np.asarray(direction, float)
    u = u / np.linalg.norm(u)
    n = np.array([-u[1], u[0]])
    t = np.asarray(tip, float)
    b = t - u * length
    dr.polygon([_xy(t), _xy(b + n * half), _xy(b - n * half)], fill=color)


def _dimline(dr, p0, p1, color=DIM, width: int = 2):
    """Dimension line with arrowheads whose tips are at p0 and p1 (outside arrows on short spans)."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    v = p1 - p0
    L = float(np.linalg.norm(v))
    if L < 1e-6:
        return
    u = v / L
    if L >= 3.0 * 13.0:
        dr.line([_xy(p0), _xy(p1)], fill=BG, width=width + 4)
        dr.line([_xy(p0 + u * 4), _xy(p1 - u * 4)], fill=color, width=width)
        _arrowhead(dr, p0, -u, color)
        _arrowhead(dr, p1, u, color)
    else:
        a, b = p0 - u * 24, p1 + u * 24
        dr.line([_xy(a), _xy(b)], fill=BG, width=width + 4)
        dr.line([_xy(a), _xy(b)], fill=color, width=width)
        _arrowhead(dr, p0, u, color)
        _arrowhead(dr, p1, -u, color)


def _ext_line(dr, p0, p1, color=DIM, width: int = 2):
    dr.line([_xy(p0), _xy(p1)], fill=color + (215,), width=width)


def _text_block(dr, xy, lines, align: str = "center", valign: str = "middle", pad=(8, 4), box: bool = True,
                spacing: int = 3, outline=None) -> tuple:
    """Draw ``lines`` [(text, font, colour)] as a block anchored at ``xy``; returns the box."""
    ws = [_tw(t, f) for t, f, _ in lines]
    hs = [_lh(f) for _, f, _ in lines]
    w = max(ws) + 2 * pad[0]
    h = sum(hs) + spacing * (len(lines) - 1) + 2 * pad[1]
    x, y = float(xy[0]), float(xy[1])
    x0 = {"center": x - w / 2, "left": x, "right": x - w}[align]
    y0 = {"middle": y - h / 2, "top": y, "bottom": y - h}[valign]
    if box and outline is not None:
        dr.rounded_rectangle([x0, y0, x0 + w, y0 + h], radius=6, fill=BG, outline=outline, width=1)
    elif box:
        dr.rectangle([x0, y0, x0 + w, y0 + h], fill=BG)
    yy = y0 + pad[1]
    for (t, f, c), lw, lh in zip(lines, ws, hs):
        if align == "center":
            tx = x0 + (w - lw) / 2
        elif align == "left":
            tx = x0 + pad[0]
        else:
            tx = x0 + w - pad[0] - lw
        dr.text((tx, yy), t, font=f, fill=c)
        yy += lh + spacing
    return (x0, y0, x0 + w, y0 + h)


def _dim_label(name: str, value: str) -> list:
    return [(name, _font(F_DIM_N), INK2), (value, _font(F_DIM_V, True), DIM)]


def _inline_label(dr, x, y, name: str, value: str):
    """'name  value' on one line, vertically centred on y, starting at x."""
    fn, fv = _font(F_DIM_N), _font(F_DIM_V, True)
    an, dn = fn.getmetrics()
    av, dv = fv.getmetrics()
    dr.text((x, y), name, font=fn, fill=INK2, anchor="lm")
    dr.text((x + _tw(name, fn) + 10, y), value, font=fv, fill=DIM, anchor="lm")
    return x + _tw(name, fn) + 10 + _tw(value, fv)


def _inline_width(name: str, value: str) -> float:
    return _tw(name, _font(F_DIM_N)) + 10 + _tw(value, _font(F_DIM_V, True))


def _hip_marker(dr, p, r: float = 9.0):
    x, y = _xy(p)
    dr.ellipse([x - r - 2, y - r - 2, x + r + 2, y + r + 2], fill=BG)
    dr.ellipse([x - r, y - r, x + r, y + r], outline=DIM, width=2)
    dr.line([(x - r - 5, y), (x + r + 5, y)], fill=DIM, width=2)
    dr.line([(x, y - r - 5), (x, y + r + 5)], fill=DIM, width=2)


def _panel_label(dr, x, y, name: str, desc: str = ""):
    f = _font(F_PANEL, True)
    dr.text((x, y), name, font=f, fill=INK)
    if desc:
        dr.text((x + _tw(name, f) + 14, y + 2), desc, font=_font(F_PANEL_D), fill=INK3)


def _nice_bar(s: float, max_px: float) -> tuple[float, int]:
    best = (0.01, 2)
    for L, n in ((0.01, 2), (0.02, 4), (0.05, 5), (0.1, 5), (0.2, 4), (0.25, 5), (0.5, 5), (1.0, 5), (2.0, 4),
                 (5.0, 5), (10.0, 5)):
        if L * s <= max_px:
            best = (L, n)
    return best


def _fmt_m(v: float) -> str:
    return f"{v:.2f}".rstrip("0").rstrip(".") if v < 1 else f"{v:g}"


def _scale_bar(dr, x, y, s: float, max_px: float = 330, caption: str = "", height: int = 10,
               small: bool = False) -> tuple:
    """Segmented scale bar with its left end at (x, y) (bar top); returns its bounding box."""
    L, n = _nice_bar(s, max_px)
    seg = L * s / n
    f = _font(F_SMALL if small else F_CAP)
    for k in range(n):
        x0, x1 = x + k * seg, x + (k + 1) * seg
        dr.rectangle([x0, y, x1, y + height], fill=INK if k % 2 == 0 else BG, outline=INK, width=1)
    marks = [0, n] if n > 2 and small else list(range(0, n + 1, 1 if n <= 2 else (n // 2 if n % 2 == 0 else n)))
    if 0 not in marks:
        marks.insert(0, 0)
    for k in marks:
        v = L * k / n
        t = "0" if k == 0 else (_fmt_m(v) + " m" if k == n else _fmt_m(v))
        dr.text((x + k * seg, y + height + 5), t, font=f, fill=INK2, anchor="ma")
    x1 = x + n * seg
    if caption:
        dr.text((x1 + 22, y + height / 2), caption, font=f, fill=INK2, anchor="lm")
        x1 = x1 + 22 + _tw(caption, f)
    return (x, y, x1, y + height + 5 + _lh(f))


def _header_height(title: str, subtitle: str) -> int:
    ft = _fit_font(title, PLATE_W - 2 * MARGIN, F_TITLE, True, 34)
    h = 56 + _lh(ft) + 14
    if subtitle:
        h += len(_wrap(subtitle, _font(F_SUB), PLATE_W - 2 * MARGIN)) * (_lh(_font(F_SUB)) + 6)
    return h + 18 + 34


def _header(cv: _Canvas, title: str, subtitle: str) -> int:
    dr = cv.dr
    ft = _fit_font(title, PLATE_W - 2 * MARGIN, F_TITLE, True, 34)
    y = 56
    dr.text((MARGIN, y), title, font=ft, fill=INK)
    y += _lh(ft) + 14
    if subtitle:
        fs = _font(F_SUB)
        for ln in _wrap(subtitle, fs, PLATE_W - 2 * MARGIN):
            dr.text((MARGIN, y), ln, font=fs, fill=INK2)
            y += _lh(fs) + 6
    y += 18
    dr.line([(MARGIN, y), (PLATE_W - MARGIN, y)], fill=RULE, width=2)
    return y + 34


def _footer_height(n_small: int) -> int:
    return 30 + _lh(_font(F_BODY)) + 18 + n_small * (_lh(_font(F_SMALL)) + 5) + 56


def _footer(cv: _Canvas, y: int, facts: str, small: list, scale: float | None = None,
            scale_caption: str = "", warn: list | None = None):
    dr = cv.dr
    dr.line([(MARGIN, y), (PLATE_W - MARGIN, y)], fill=RULE, width=2)
    y += 30
    fb = _font(F_BODY)
    if facts:
        dr.text((MARGIN, y), facts, font=fb, fill=INK)
    if scale is not None:
        L, n = _nice_bar(scale, 300)
        fcap = _font(F_CAP)
        wbar = L * scale + 22 + _tw(scale_caption, fcap)
        _scale_bar(dr, PLATE_W - MARGIN - wbar, y + 4, scale, 300, scale_caption)
    y += _lh(fb) + 18
    fs = _font(F_SMALL)
    for ln in (warn or []):
        dr.text((MARGIN, y), ln, font=fs, fill=WARN)
        y += _lh(fs) + 5
    for ln in small:
        dr.text((MARGIN, y), ln, font=fs, fill=INK3)
        y += _lh(fs) + 5


def _small_print_lines(lines: list[str]) -> list[str]:
    fs = _font(F_SMALL)
    out = []
    for ln in lines:
        out += _wrap(ln, fs, PLATE_W - 2 * MARGIN)
    return out


def _provenance() -> str:
    return (f"Rendered with MuJoCo {mujoco.__version__} from the compiled MJCF at its 'home' keyframe (nominal "
            "standing pose); every image is a model render, and dimensions and labels are computed from the "
            "model geometry and the ledger next to it. Screening figure, not a certified drawing.")


def _write_sidecar(out_png, plate: str, data: dict):
    p = Path(out_png).with_suffix(".json")
    doc = dict(plate=plate, png=str(out_png), mujoco=mujoco.__version__, **data)
    p.write_text(json.dumps(doc, indent=1, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x)))


def _describe(g: _Meas) -> str:
    """Plain-language description of a candidate from its parameters."""
    try:
        c = parse(g.cid)
    except (ValueError, IndexError, KeyError):
        return g.cid
    bits = [FAMILIES.get(c.family, c.family), f"{WIDTHS[c.width]['label'].lower()} width",
            POSTURES[c.posture]["label"].replace("Level torso", "level torso"), f"{c.height} cm height class"]
    if c.proportion != "base":
        bits.append(PROPORTIONS[c.proportion]["label"].lower())
    if c.leg_layout != "tandem":
        bits.append(f"{c.leg_layout} leg layout")
    if c.battery_pos != "tailbase":
        bits.append(f"battery in the {c.battery_pos}")
    return " · ".join(bits)


# ------------------------------------------------------------------------------------------------
# 1. Candidate plate
# ------------------------------------------------------------------------------------------------
def candidate_plate(xml, out_png, title: str, subtitle: str = "") -> str:
    """Side / front / top orthographic views at one common scale plus a three-quarter view.

    Dimension lines: crown height and hip-pitch-axis height (side), overall nose-to-tail length
    (side), torso width, width across thighs and foot-centre spacing (front), max width (top).  The
    footer gives mass, crown, hip height, length and max width, the scale bar of the orthographic
    views and the SHA-256 prefix of the XML file.  ``subtitle`` defaults to a description of the
    candidate's parameters."""
    g = _measure(xml)
    subtitle = subtitle or _describe(g)
    L, Wd, Hz = g.length, g.max_width, g.top
    avail = PLATE_W - 2 * MARGIN
    gap_c, tail_pad, r_min = 56, 44, 840
    # --- side-view dimension band (two vertical dimensions right of the nose)
    lab_hip = _dim_label("Hip-pitch axis", f"{_mm(g.hip_height)} mm")
    lab_crown = _dim_label("Crown height", f"{_mm(g.crown)} mm")
    box_w = max(max(_tw(t, f) for t, f, _ in lab) for lab in (lab_hip, lab_crown)) + 16
    off_in = 64
    spacing = box_w / 2 + 22
    band = off_in + spacing + box_w / 2 + 14
    # --- front-view labels (right of the drawing)
    thigh_name = "Width across thighs" if g.thigh_fairing else "Width across thighs (no fairing)"
    flabs = [("Torso width", f"{_mm(g.torso_width)} mm"), (thigh_name, f"{_mm(g.thigh_width)} mm"),
             ("Foot-centre spacing", f"{_mm(g.foot_spacing)} mm")]
    flab_w = max(_inline_width(n, v) for n, v in flabs)
    front_fixed = 60 + 22 + flab_w + 20
    # --- top-view label
    lab_w = _dim_label("Max width", f"{_mm(g.max_width)} mm")
    top_band = 64 + 18 + max(_tw(t, f) for t, f, _ in lab_w) + 24
    # --- common scale
    s = (avail - gap_c - r_min - tail_pad - max(band, top_band)) / L
    s = min(s, 620.0 / Hz, (avail - gap_c - tail_pad - max(band, top_band) - front_fixed) / (L + Wd))
    left_w = int(np.ceil(tail_pad + L * s + max(band, top_band)))
    right_w = avail - gap_c - left_w
    # --- vertical layout
    y0 = _header_height(title, subtitle)
    above = 132
    y_top = y0 + PANEL_H + above
    y_g = int(round(y_top + Hz * s))
    y_len = y_g + 66
    row1_h = (y_len + 40) - y0
    y1 = y0 + row1_h + 30
    top_draw_top = y1 + PANEL_H + 26
    h_top = int(np.ceil(PANEL_H + 26 + Wd * s + 34))
    h2 = max(h_top, int(0.58 * right_w))
    tq_w, tq_h = right_w, h2 - PANEL_H
    small = _small_print_lines([
        f"Model {g.name} · SHA-256 {g.sha256[:16]} · {_provenance()}",
        "Crown: highest vertex of the skull cover. Hip-pitch axis: hip-pitch joint anchor height. Length and max "
        "width: extent of all visible geometry. Torso width: trunk covers without arms. Width across thighs: thigh "
        "fairings. Foot-centre spacing: distance between the two sole sites (DEFINITIONS.md). " + _hip_note(g),
    ])
    warn = _check_failures(g)
    y_f = y1 + h2 + 26
    H = y_f + _footer_height(len(small) + len(warn))
    cv = _Canvas(PLATE_W, H)
    x_left, x_right = MARGIN, MARGIN + left_w + gap_c
    # --- orthographic renders (one scene, three crops)
    front_block = Wd * s + 22 + flab_w
    front_x0 = x_right + max(60.0, (right_w - front_block) / 2 - 10)  # canvas x of the drawing's left edge
    with _scene(xml, max(left_w, right_w), max(row1_h, h_top)) as sc:
        side = _ortho(sc, "side", "exterior", s, [g.lo[0], 0.0, 0.0], (tail_pad, y_g - y0), left_w, row1_h, g.center)
        front = _ortho(sc, "front", "exterior", s, [g.center[0], g.lo[1], 0.0], (front_x0 - x_right, y_g - y0),
                       right_w, row1_h, g.center)
        top = _ortho(sc, "top", "exterior", s, [g.lo[0], g.hi[1], g.center[2]], (tail_pad, top_draw_top - y1),
                     left_w, h_top, g.center)
    with _scene(xml, tq_w, tq_h) as sc:
        tq = _perspective(sc, "threequarter", "exterior", (34, 24, tq_w - 34, tq_h - 30))
    cv.place(side, x_left, y0)
    cv.place(front, x_right, y0)
    cv.place(top, x_left, y1)
    cv.place(tq, x_right, y1 + PANEL_H)
    dr = cv.dr
    _header(cv, title, subtitle)
    _panel_label(dr, x_left, y0, "SIDE", _VIEW_TEXT["side"])
    _panel_label(dr, x_right, y0, "FRONT", _VIEW_TEXT["front"])
    _panel_label(dr, x_left, y1, "TOP", _VIEW_TEXT["top"])
    _panel_label(dr, x_right, y1, "THREE-QUARTER", _VIEW_TEXT["threequarter"])
    # --- side view geometry (canvas px)
    nose, tail = side.px(g.nose)[0], side.px(g.tail)[0]
    crown, hip = side.px(g.crown_pt)[0], side.px(g.hip)[0]
    gy = float(side.px([0.0, 0.0, 0.0])[0][1])
    x_in = nose[0] + off_in
    x_out = x_in + spacing
    fr = {k: front.px(v)[0] for k, v in dict(tl=g.torso_lo, th=g.torso_hi, ql=g.thigh_lo, qh=g.thigh_hi,
                                                 sl=g.soles[0], sr=g.soles[1]).items()}
    f_top = float(front.px([0.0, 0.0, g.top])[0][1])
    y_in, y_out = f_top - 46, f_top - 96
    tw_lo, tw_hi = top.px(g.wlo)[0], top.px(g.whi)[0]
    t_nose = top.px(g.nose)[0]
    x_w = t_nose[0] + 64

    def ext_lines(d):
        _ext_line(d, (crown[0] + 10, crown[1]), (x_out + 12, crown[1]))
        _ext_line(d, (hip[0] + 16, hip[1]), (x_in + 12, hip[1]))
        _ext_line(d, (nose[0] + 28, gy), (x_out + 12, gy))
        _ext_line(d, (nose[0], nose[1] + 8), (nose[0], y_len + 12))
        _ext_line(d, (tail[0], tail[1] + 8), (tail[0], y_len + 12))
        for k in ("tl", "th"):
            _ext_line(d, (fr[k][0], fr[k][1] - 6), (fr[k][0], y_in - 12))
        for k in ("ql", "qh"):
            _ext_line(d, (fr[k][0], fr[k][1] - 6), (fr[k][0], y_out - 12))
        for k in ("sl", "sr"):
            _ext_line(d, (fr[k][0], gy + 6), (fr[k][0], y_len + 12))
        for p in (tw_lo, tw_hi):
            _ext_line(d, (p[0] + 8, p[1]), (x_w + 12, p[1]))

    cv.ghosted(ext_lines)
    dr = cv.dr
    # ground lines
    dr.line([(tail[0] - 26, gy - 0.5), (nose[0] + 26, gy - 0.5)], fill=INK2, width=2)
    fx0 = min(fr["ql"][0], fr["tl"][0], fr["sl"][0], fr["sr"][0]) - 34
    fx1 = max(fr["qh"][0], fr["th"][0], fr["sl"][0], fr["sr"][0]) + 34
    dr.line([(fx0, gy - 0.5), (fx1, gy - 0.5)], fill=INK2, width=2)
    # side dimensions
    _dimline(dr, (x_in, gy), (x_in, hip[1]))
    _dimline(dr, (x_out, gy), (x_out, crown[1]))
    _dimline(dr, (tail[0], y_len), (nose[0], y_len))
    _hip_marker(dr, hip)
    _text_block(dr, (x_in, 0.5 * (gy + hip[1])), lab_hip)
    _text_block(dr, (x_out, 0.5 * (hip[1] + crown[1])), lab_crown)
    _text_block(dr, (0.5 * (tail[0] + nose[0]), y_len), _dim_label("Overall length, nose to tail",
                                                                     f"{_mm(g.length)} mm"))
    # front dimensions
    _dimline(dr, (fr["tl"][0], y_in), (fr["th"][0], y_in))
    _dimline(dr, (fr["ql"][0], y_out), (fr["qh"][0], y_out))
    _dimline(dr, (fr["sl"][0], y_len), (fr["sr"][0], y_len))
    x_lab = max(fr[k][0] for k in fr) + 24
    for (n, v), yy in zip(flabs, (y_in, y_out, y_len)):
        _inline_label(dr, x_lab, yy, n, v)
    # top dimension
    _dimline(dr, (x_w, tw_hi[1]), (x_w, tw_lo[1]))
    _text_block(dr, (x_w + 18, 0.5 * (tw_lo[1] + tw_hi[1])), lab_w, align="left")
    # footer
    facts = (f"Mass {_kg(g.mass)} kg   ·   Crown {_mm(g.crown)} mm   ·   Hip-pitch axis {_mm(g.hip_height)} mm   ·   "
             f"Length {_mm(g.length)} mm   ·   Max width {_mm(g.max_width)} mm")
    _footer(cv, y_f, facts, small, scale=s, scale_caption="side, front and top views", warn=warn)
    path = cv.save(out_png)
    _write_sidecar(out_png, "candidate_plate", dict(
        title=title, subtitle=subtitle, scale_px_per_m=s, measurements=g.facts(),
        notes="side/front/top orthographic at scale_px_per_m; three-quarter perspective, not to scale"))
    return path


# ------------------------------------------------------------------------------------------------
# 2. Cutaway plate
# ------------------------------------------------------------------------------------------------
def _part_name(pid: str, vendor: bool = True) -> str:
    """Short vendor + model name of a catalog part or actuator id (the id itself if unknown)."""
    rec = catalog.parts().get(pid) or catalog.actuators().get(pid)
    if rec is None:
        return pid
    ven = rec.vendor.split(" (")[0].strip()
    model = rec.model.split(" (")[0].split(",")[0].strip()
    if not vendor or not ven or ven.split()[0].lower() in model.lower() or len(ven) > 14:
        return model
    return f"{ven} {model}"


@dataclass
class _Callout:
    key: str
    title: str
    detail: list  # list of str
    anchors: list = field(default_factory=list)  # canvas xy
    hidden: list = field(default_factory=list)
    band: str = ""
    cx: float = 0.0
    y0: float = 0.0
    w: float = 0.0
    h: float = 0.0
    pref_dx: float = 0.0  # preferred label offset from the anchors' mean x, in label widths

    def lines(self) -> list:
        out = [(self.title, _font(F_CALL_T, True), INK)]
        for d in self.detail:
            out.append((d, _font(F_CALL_D), INK2))
        return out


def _hardware_callouts(g: _Meas, m, leg_group: bool = True) -> tuple[list, dict]:
    """Callout definitions for the main installed hardware, from the ledger rows; returns the callouts
    and their geom groups (key -> geom names)."""
    rows = {r["name"]: r for r in g.ledger.get("rows", [])}
    sel = g.ledger.get("selection") or {}

    def mass(*names):
        return sum(rows[n]["mass_kg"] for n in names if n in rows)

    def has(n):
        return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n) >= 0

    def part(n):
        return _part_name(rows[n]["source"]) if n in rows else ""

    fd = _font(F_CALL_D)

    def wrap(t):
        return _wrap(t, fd, CALLOUT_WRAP) if t else []

    items, groups = [], {}

    def add(key, title, geoms, detail):
        geoms = [n for n in geoms if has(n)]
        if not geoms:
            return
        items.append(_Callout(key, title, detail))
        groups[key] = geoms

    add("battery", "Battery cassette", ["hw_battery_cassette", "hw_battery"],
        wrap(part("battery")) + [f"{mass('battery', 'battery_cassette'):.2f} kg with cassette"])
    add("compute", "Compute", ["hw_compute_tray", "hw_compute"],
        wrap(part("compute")) + [f"{mass('compute', 'compute_tray'):.2f} kg with tray"])
    add("camera", "Camera", ["hw_camera"], wrap(part("camera")) + [f"{mass('camera'):.2f} kg"])
    bus = sel.get("bus")
    add("power", "Power distribution", ["hw_power_distribution"],
        ([f"{bus:.0f} V bus"] if bus else []) + [f"{mass('power_distribution'):.2f} kg"])
    add("control", "Control stack", ["hw_control_stack"], ["MCU, CAN, IMU", f"{mass('control_stack'):.2f} kg"])
    add("lidar", "2D lidar", ["hw_lidar"], wrap(part("lidar")))
    add("tof", "ToF sensors", ["hw_tof_l", "hw_tof_r"], wrap(f"2 x {part('tof_l')}") if part("tof_l") else [])
    add("neck", "Neck actuator", ["hw_neck_motor"], wrap(part("neck_motor")))
    add("tail", "Tail actuator", ["hw_tail_motor"], wrap(part("tail_motor")))
    if leg_group:
        legs = [r for r in g.ledger.get("rows", []) if r.get("category") == "leg_actuator"]
        counts = {}
        for r in legs:
            counts[r["source"]] = counts.get(r["source"], 0) + 1
        detail = sum((wrap(f"{n} x {_part_name(a)}") for a, n in sorted(counts.items(), key=lambda kv: -kv[1])), [])
        detail.append(f"{sum(r['mass_kg'] for r in legs):.2f} kg in total")
        names = [f"hw_{r['name']}" for r in legs if has(f"hw_{r['name']}")]
        if not names:
            names = [_name(m, mujoco.mjtObj.mjOBJ_GEOM, k) for k in range(m.ngeom)
                     if _name(m, mujoco.mjtObj.mjOBJ_GEOM, k).startswith(("hw_left_", "hw_right_"))
                     and _name(m, mujoco.mjtObj.mjOBJ_GEOM, k).endswith(("_motor", "_motor_a", "_motor_b"))]
        if names:
            items.append(_Callout("legs", "Leg actuators", detail))
            groups["legs"] = names
    return items, groups


_LEG_JOINTS = [("hip_roll_motor", "Hip roll"), ("hip_pitch_motor", "Hip pitch"), ("hip_yaw_motor", "Thigh twist"),
               ("knee_motor", "Knee"), ("ankle_motor_a", "Ankle A"), ("ankle_motor_b", "Ankle B")]


def _leg_callouts(g: _Meas, m) -> tuple[list, dict]:
    """One callout per leg-actuator geom of both sides (the visible side is chosen later)."""
    rows = {r["name"]: r for r in g.ledger.get("rows", [])}
    items, groups = [], {}
    for suffix, title in _LEG_JOINTS:
        for side in ("left", "right"):
            n = f"{side}_{suffix}"
            if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"hw_{n}") < 0:
                continue
            det = [_part_name(rows[n]["source"], vendor=False)] if n in rows else []
            items.append(_Callout(n, title, det))
            groups[n] = [f"hw_{n}"]
    return items, groups


def _resolve_leg_sides(items: list, anchors: dict, min_sep: float = 30.0) -> list:
    """Per joint keep the better-visible of the left and right actuator, or both when their projected
    centres are more than ``min_sep`` px apart in the image (e.g. the tandem hip-roll motors ahead of
    and behind the hip); both then get front/rear (from world x) in their titles."""
    out = []
    by_title = {}
    for it in items:
        if it.key in anchors:
            by_title.setdefault(it.title, []).append(it)
    for title, its in by_title.items():
        if len(its) == 2 and np.linalg.norm(anchors[its[0].key][4] - anchors[its[1].key][4]) > min_sep:
            rear, front = sorted(its, key=lambda it: anchors[it.key][3][0])
            if abs(anchors[rear.key][3][0] - anchors[front.key][3][0]) > 1e-3:
                rear.title, front.title = f"{title}, rear", f"{title}, front"
            out += [rear, front]
        else:
            out.append(max(its, key=lambda it: anchors[it.key][1]))
    return out


def _layout_callouts(items: list, x0: float, x1: float, top_y: float, bot_y: float, mid_y: float,
                     gap: float = 18.0):
    """Place label cards in a band above (bottom edge at ``top_y``) or below (top edge at ``bot_y``)
    the drawing.  Each label goes to the band on its anchors' side of ``mid_y``; while a band is
    over-full, its label nearest ``mid_y`` moves to the other band if that one has room.  Within a
    band labels keep the anchors' x order, are spread so they do not overlap, and adjacent labels
    are swapped while that removes leader crossings (then shortens leaders).  A band still over-full
    gets a second tier."""
    width = x1 - x0

    def ymean(it):
        return float(np.mean([a[1] for a in it.anchors]))

    def need(band):
        its = [it for it in items if it.band == band]
        return sum(it.w for it in its) + gap * max(len(its) - 1, 0)

    for it in items:
        ln = it.lines()
        it.w = max(_tw(t, f) for t, f, _ in ln) + 16
        it.h = sum(_lh(f) for _, f, _ in ln) + 3 * (len(ln) - 1) + 8
        it.band = "top" if ymean(it) < mid_y else "bottom"
    for _ in range(len(items)):
        over = [b for b in ("top", "bottom") if need(b) > width]
        if not over:
            break
        other = "bottom" if over[0] == "top" else "top"
        mv = min((it for it in items if it.band == over[0]), key=lambda it: abs(ymean(it) - mid_y))
        if need(other) + gap + mv.w > width:
            break
        mv.band = other
    for band in ("top", "bottom"):
        its = sorted([it for it in items if it.band == band], key=lambda it: _desired_x(it))
        if not its:
            continue
        if sum(it.w for it in its) + gap * (len(its) - 1) <= width:
            tiers = [_untangle(its, x0, x1, gap, band, top_y, bot_y)]
        else:
            tiers = [its[0::2], its[1::2]]
        hmax0 = max(it.h for it in tiers[0])
        for t, tier in enumerate(tiers):
            _spread(tier, x0, x1, gap)
            for it in tier:
                it.y0 = top_y - it.h - t * (hmax0 + 14) if band == "top" else bot_y + t * (hmax0 + 14)


def _desired_x(it) -> float:
    return float(np.mean([a[0] for a in it.anchors])) + it.pref_dx * it.w


def _leaders(it) -> list:
    """Leader segments of a placed label (anchor -> attachment point on the card edge)."""
    bx0, bx1 = it.cx - it.w / 2, it.cx + it.w / 2
    ay = it.y0 + it.h if it.band == "top" else it.y0
    if len(it.anchors) > 1:
        ax = min(max(float(np.mean([a[0] for a in it.anchors])), bx0 + 12), bx1 - 12)
        return [(np.asarray(a, float), np.array([ax, ay])) for a in it.anchors]
    return [(np.asarray(a, float), np.array([min(max(a[0], bx0 + 12), bx1 - 12), ay])) for a in it.anchors]


def _cross(p1, p2, q1, q2) -> bool:
    def orient(a, b, c):
        return np.sign((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))

    return (orient(p1, p2, q1) * orient(p1, p2, q2) < 0) and (orient(q1, q2, p1) * orient(q1, q2, p2) < 0)


def _untangle(its: list, x0, x1, gap, band, top_y, bot_y) -> list:
    """Adjacent swaps of the label order while they reduce (crossings, total leader length)."""
    def score(order):
        _spread(order, x0, x1, gap)
        for it in order:
            it.y0 = top_y - it.h if band == "top" else bot_y
        segs = [(k, s) for k, it in enumerate(order) for s in _leaders(it)]
        n = sum(_cross(a[1][0], a[1][1], b[1][0], b[1][1]) for i, a in enumerate(segs) for b in segs[i + 1:]
                if a[0] != b[0])
        return n, sum(float(np.linalg.norm(s[1] - s[0])) for _, s in segs)

    order = list(its)
    best = score(order)
    for _ in range(4 * len(order) ** 2):
        improved = False
        for k in range(len(order) - 1):
            trial = order[:k] + [order[k + 1], order[k]] + order[k + 2:]
            sc_ = score(trial)
            if sc_[0] < best[0] or (sc_[0] == best[0] and sc_[1] < best[1] - 1.0):
                order, best, improved = trial, sc_, True
        if not improved:
            break
    return order


def _spread(its: list, x0: float, x1: float, gap: float):
    """Label centres in the given order, as close as possible (least squares) to their desired x,
    without overlap and inside [x0, x1]: an isotonic regression (pool adjacent violators) on the
    desired centres minus the cumulative minimum separations, clipped to the band."""
    n = len(its)
    if not n:
        return
    c = np.array([_desired_x(it) for it in its])
    w = np.array([it.w for it in its])
    o = np.zeros(n)
    for k in range(1, n):
        o[k] = o[k - 1] + 0.5 * (w[k - 1] + w[k]) + gap
    blocks = []  # pool adjacent violators, non-decreasing fit
    for v in c - o:
        blocks.append([v, 1])
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]:
            s, m = blocks.pop()
            blocks[-1][0] += s
            blocks[-1][1] += m
    y = np.concatenate([np.full(m, s / m) for s, m in blocks])
    lb, ub = x0 + 0.5 * w[0] - o[0], x1 - 0.5 * w[-1] - o[-1]
    y = np.full(n, lb) if lb > ub else np.clip(y, lb, ub)
    for k, it in enumerate(its):
        it.cx = float(y[k] + o[k])


def _draw_callouts(cv: _Canvas, items: list):
    """Leaders (halo + line; thinner and lighter for a label with several anchors), anchor dots and
    label cards."""
    dr = cv.dr
    for it in items:
        group = len(it.anchors) > 1
        for a, b in _leaders(it):
            dr.line([_xy(a), _xy(b)], fill=BG, width=5 if group else 6)
            dr.line([_xy(a), _xy(b)], fill=INK2 if group else INK, width=1 if group else 2)
    for it in items:
        for a, hid in zip(it.anchors, it.hidden):
            x, y = _xy(a)
            dr.ellipse([x - 7, y - 7, x + 7, y + 7], fill=BG)
            if hid:
                dr.ellipse([x - 5, y - 5, x + 5, y + 5], outline=ACCENT, width=2)
            else:
                dr.ellipse([x - 5, y - 5, x + 5, y + 5], fill=ACCENT)
        _text_block(dr, (it.cx, it.y0), it.lines(), valign="top", pad=(8, 4), outline=RULE)


def _attach(items: list, anchors: dict):
    keep = []
    for it in items:
        if it.key in anchors:
            p, n, hid = anchors[it.key][:3]
            it.anchors, it.hidden = [p], [hid]
            keep.append(it)
    return keep


def _attach_leg_group(item: _Callout, anchors: dict, legs: list):
    """Anchors of the group callout: one per visible leg actuator (sides resolved as in
    ``_resolve_leg_sides``)."""
    chosen = _resolve_leg_sides(legs, anchors)
    item.anchors = [anchors[it.key][0] for it in chosen]
    item.hidden = [anchors[it.key][2] for it in chosen]


def _shift(items: list, dx: float, dy: float):
    for it in items:
        it.cx += dx
        it.y0 += dy
        it.anchors = [np.asarray(a, float) + np.array([dx, dy]) for a in it.anchors]


def _label_extent(items: list, dy: float = 0.0) -> tuple[float, float]:
    if not items:
        return np.inf, -np.inf
    return min(it.y0 for it in items) + dy, max(it.y0 + it.h for it in items) + dy


def cutaway_plate(xml, out_png, title: str, subtitle: str = "") -> str:
    """Installed hardware: side and three-quarter views with translucent covers and a side view with
    the covers hidden, with callouts drawn from geom positions in the compiled model.

    Callouts: battery cassette, compute, camera, power distribution, control stack, lidar and ToF
    sensors (when fitted), neck and tail actuators, and the leg actuators (``hw_*_motor``; one
    group label with a leader to each visible motor in the large side view, one label per joint in
    the hardware-only view).  Rows grow to fit their callout bands."""
    g = _measure(xml)
    m, _ = _load(xml)
    subtitle = subtitle or (f"{g.cid}: covers drawn translucent (top and right) and hidden (bottom left). "
                            "Each callout points at the named part in the compiled model.")
    L, Hz = g.length, g.top
    avail = PLATE_W - 2 * MARGIN
    band1, band2, gap_c = 150, 140, 56
    s1 = min((avail - 80) / L, 640.0 / Hz)
    w1 = avail
    h1 = int(np.ceil(PANEL_H + band1 + Hz * s1 + 30 + band1))
    w2a = int(round(0.56 * (avail - gap_c)))
    w2b = avail - gap_c - w2a
    s2 = min((w2a - 60) / L, 420.0 / Hz)
    h2 = int(np.ceil(PANEL_H + band2 + Hz * s2 + 24 + band2))
    tq_h = h2 - PANEL_H
    main, main_groups = _hardware_callouts(g, m)
    legs, leg_groups = _leg_callouts(g, m)
    # --- renders and anchors, tile-local coordinates (tiles at ox = oy = 0)
    gy1 = PANEL_H + band1 + Hz * s1
    with _scene(xml, w1, h1) as sc:
        t1 = _ortho(sc, "side", "cutaway", s1, [g.lo[0], 0.0, 0.0], (40 + 0.5 * (w1 - 80 - L * s1), gy1), w1, h1,
                    g.center)
        a1 = _anchors(sc, t1, {**main_groups, **leg_groups})
    gy2 = PANEL_H + band2 + Hz * s2
    with _scene(xml, w2a, h2) as sc:
        t2 = _ortho(sc, "side", "hardware", s2, [g.lo[0], 0.0, 0.0], (30 + 0.5 * (w2a - 60 - L * s2), gy2), w2a, h2,
                    g.center)
        a2 = _anchors(sc, t2, leg_groups)
    with _scene(xml, w2b, tq_h) as sc:
        t3 = _perspective(sc, "threequarter", "cutaway", (40, band2 - 8, w2b - 40, tq_h - band2 + 8))
        a3 = _anchors(sc, t3, {k: v for k, v in main_groups.items() if k in ("battery", "compute", "camera",
                                                                              "power", "control")})
    # --- callout layout, tile-local
    c1 = _attach([copy.deepcopy(it) for it in main if it.key != "legs"], a1)
    lg = [it for it in main if it.key == "legs"]
    if lg:
        grp = copy.deepcopy(lg[0])
        _attach_leg_group(grp, a1, [copy.deepcopy(it) for it in legs])
        grp.pref_dx = -1.0  # label behind the leg (under the tail), so the leaders fan out over open space
        if grp.anchors:
            c1.append(grp)
    _layout_callouts(c1, 0, w1, PANEL_H + band1 - 22, gy1 + 30, gy1 - 0.55 * Hz * s1)
    c2 = _attach(_resolve_leg_sides([copy.deepcopy(it) for it in legs], a2), a2)
    _layout_callouts(c2, 0, w2a, PANEL_H + band2 - 18, gy2 + 26, gy2 - 0.62 * Hz * s2, gap=14)
    c3 = _attach([copy.deepcopy(it) for it in main if it.key in a3], a3)
    _layout_callouts(c3, 0, w2b, band2 - 14, tq_h - band2 + 22, 0.5 * tq_h, gap=14)
    # --- rows sized to tiles and labels (row-local y; labels stay below the panel label)
    lo1, hi1 = _label_extent(c1)
    top1 = max(0.0, PANEL_H + 6 - lo1)
    row1_h = int(np.ceil(top1 + max(h1, hi1 + 10)))
    lo2 = min(_label_extent(c2)[0], _label_extent(c3, PANEL_H)[0])
    hi2 = max(_label_extent(c2)[1], _label_extent(c3, PANEL_H)[1])
    top2 = max(0.0, PANEL_H + 6 - lo2)
    row2_h = int(np.ceil(top2 + max(h2, hi2 + 10)))
    y0 = _header_height(title, subtitle)
    y1 = y0 + row1_h + 24
    small = _small_print_lines([
        f"Model {g.name} · SHA-256 {g.sha256[:16]} · {_provenance()}",
        "Cutaway: shell covers rendered at 16 % opacity; hardware view: shells hidden. Callout dots sit on the "
        "visible pixels of the named geoms (hollow dot: part hidden behind other hardware in that view). Masses and "
        "parts from the ledger next to the model; part names from the sourced catalog. The cosmetic eyes, teeth and "
        "claws are hardware-group primitives in the model, so they remain in the hardware view. " + _hip_note(g),
    ])
    warn = _check_failures(g)
    y_f = y1 + row2_h + 26
    cv = _Canvas(PLATE_W, y_f + _footer_height(len(small) + len(warn)))
    xa, xb = MARGIN, MARGIN + w2a + gap_c
    pos = [(t1, c1, MARGIN, int(round(y0 + top1))), (t2, c2, xa, int(round(y1 + top2))),
           (t3, c3, xb, int(round(y1 + top2 + PANEL_H)))]
    for t, items, ox, oy in pos:
        cv.place(t, ox, oy)
        _shift(items, ox, oy)
    _header(cv, title, subtitle)
    dr = cv.dr
    _panel_label(dr, MARGIN, y0, "SIDE, CUTAWAY", "covers translucent · orthographic")
    _panel_label(dr, xa, y1, "SIDE, HARDWARE ONLY", "covers hidden · orthographic · leg actuators")
    _panel_label(dr, xb, y1, "THREE-QUARTER, CUTAWAY", "perspective · not to scale")
    for t in (t1, t2):
        gyc = float(t.px([0.0, 0.0, 0.0])[0][1])
        xs = t.px([g.tail, g.nose])[:, 0]
        dr.line([(xs[0] - 24, gyc - 0.5), (xs[1] + 24, gyc - 0.5)], fill=INK2, width=2)
    for items in (c1, c2, c3):
        _draw_callouts(cv, items)
    facts = (f"Mass {_kg(g.mass)} kg   ·   Crown {_mm(g.crown)} mm   ·   Hip-pitch axis {_mm(g.hip_height)} mm   ·   "
             f"Length {_mm(g.length)} mm")
    _footer(cv, y_f, facts, small, warn=warn)
    path = cv.save(out_png)
    _write_sidecar(out_png, "cutaway_plate", dict(
        title=title, subtitle=subtitle, measurements=g.facts(),
        callouts={name: [dict(title=it.title, detail=it.detail, anchors=it.anchors, hidden=it.hidden) for it in items]
                  for name, items in (("side_cutaway", c1), ("side_hardware", c2), ("threequarter_cutaway", c3))}))
    return path


# ------------------------------------------------------------------------------------------------
# 3. Lineup
# ------------------------------------------------------------------------------------------------
def _person(dr, x_c: float, y_g: float, s: float):
    """Neutral 1.75 m person (front view) built from simple shapes, centred at x_c, feet on y_g."""
    def rr(x0, z0, x1, z1, r):  # metres (x lateral, z height) -> rounded rectangle in px
        return [x_c + x0 * s, y_g - z1 * s, x_c + x1 * s, y_g - z0 * s], r * s

    parts = [rr(-0.165, 0.0, -0.035, 0.93, 0.05), rr(0.035, 0.0, 0.165, 0.93, 0.05),  # legs
             rr(-0.19, 0.86, 0.19, 1.47, 0.07),  # torso
             rr(-0.265, 0.80, -0.175, 1.44, 0.045), rr(0.175, 0.80, 0.265, 1.44, 0.045),  # arms
             rr(-0.05, 1.42, 0.05, 1.54, 0.02)]  # neck
    head = [x_c - 0.105 * s, y_g - 1.75 * s, x_c + 0.105 * s, y_g - 1.535 * s]
    lw = max(2.0, 0.012 * s)
    for (box, r) in parts:
        dr.rounded_rectangle([box[0] - lw, box[1] - lw, box[2] + lw, box[3] + lw], radius=r + lw, fill=PERSON_LINE)
    dr.ellipse([head[0] - lw, head[1] - lw, head[2] + lw, head[3] + lw], fill=PERSON_LINE)
    for (box, r) in parts:
        dr.rounded_rectangle(box, radius=r, fill=PERSON_FILL)
    dr.ellipse(head, fill=PERSON_FILL)


def lineup(xmls_with_labels, out_png, mode: str = "silhouette", normalize: bool = False, title: str = "") -> str:
    """Side views of several candidates on one shared ground line.

    ``normalize=False``: one physical scale (px per metre identical for all), 0.5 m grid and a 1.75 m
    person as a neutral scale reference.  ``normalize=True``: each candidate is scaled so its crown
    height is the same (labelled height-normalized; each figure gets its own scale bar).  ``mode`` is
    a ``render.Scene`` mode (``silhouette``, ``exterior``, ``cutaway`` or ``hardware``)."""
    items = [(str(x), lab) for x, lab in xmls_with_labels]
    gs = [_measure(x) for x, _ in items]
    n = len(gs)
    title = title or ("Height-normalized side views" if normalize else "Side views at one physical scale")
    if normalize:
        subtitle = ("HEIGHT-NORMALIZED: each candidate is drawn so its crown height is the same; sizes are not "
                    "comparable between figures (each has its own scale bar). Shared ground line.")
    else:
        subtitle = ("One physical scale for all figures, shared ground line, 0.5 m grid. The grey figure is a "
                    "1.75 m person drawn from simple shapes as a scale reference.")
    avail = PLATE_W - 2 * MARGIN
    axis_w = 0 if normalize else 92
    person_m = 0.0 if normalize else 0.66
    gap = 64
    Ls = np.array([g.length for g in gs])
    tops = np.array([g.top for g in gs])
    crowns = np.array([g.crown for g in gs])
    if normalize:
        h_t = min(470.0, (avail - gap * (n - 1)) / float(np.sum(Ls / crowns)))
        scales = h_t / crowns
    else:
        s = min((avail - axis_w - gap * n) / (float(Ls.sum()) + person_m), 820.0 / max(float(tops.max()), 1.80))
        scales = np.full(n, s)
    widths = Ls * scales
    used = float(widths.sum()) + (person_m * scales[0] if not normalize else 0.0)
    free = avail - axis_w - used
    gap_eff = free / (n + (0 if normalize else 0)) if not normalize else free / max(n - 1, 1)
    if normalize:
        hmax = float(np.max(tops * scales))
        z_grid = 0.0
    else:  # grid up to the next 0.5 m line above the tallest figure (person included)
        z_grid = 0.5 * np.ceil((max(float(tops.max()), 1.75) + 0.05) / 0.5)
        hmax = z_grid * float(scales[0])
    y0 = _header_height(title, subtitle)
    y_g = int(round(y0 + (64 if normalize else 30) + hmax))
    lab_lines = 4
    y_lab = y_g + 24
    lab_h = (lab_lines + (1 if normalize else 0)) * (_lh(_font(F_CAP)) + 4) + (34 if normalize else 0)
    small = _small_print_lines([
        "Models: " + "   ".join(f"{g.cid} {g.sha256[:12]}" for g in gs),
        _provenance() + " Side views are orthographic; the ground row of each is its projection of z = 0.",
    ])
    warn = sum((_check_failures(g) for g in gs), [])
    y_f = y_lab + lab_h + 30
    cv = _Canvas(PLATE_W, y_f + _footer_height(len(small) + len(warn)))
    dr = cv.dr
    x_area0, x_area1 = MARGIN + axis_w, PLATE_W - MARGIN
    if not normalize:
        s = float(scales[0])
        z = 0.5
        while z <= z_grid + 1e-9:
            yy = y_g - z * s
            dr.line([(x_area0, yy), (x_area1, yy)], fill=GRID_MAJOR if abs(z % 1.0) < 1e-9 else GRID, width=1)
            dr.text((x_area0 - 14, yy), f"{z:.1f} m", font=_font(F_CAP), fill=INK3, anchor="rm")
            z += 0.5
        xg = x_area0
        top_px = y_g - z_grid * s
        while xg <= x_area1 + 1e-6:
            dr.line([(xg, top_px), (xg, y_g)], fill=GRID, width=1)
            xg += 0.5 * s
    # figures
    x = x_area0 + (0 if normalize else person_m * scales[0] + gap_eff * 0.5)
    placed = []
    for k, ((xml, lab), g) in enumerate(zip(items, gs)):
        sk = float(scales[k])
        pad = 12
        tw_, th_ = int(np.ceil(widths[k] + 2 * pad)), int(np.ceil(tops[k] * sk + 2 * pad))
        with _scene(xml, tw_, th_) as sc:
            t = _ortho(sc, "side", mode, sk, [g.lo[0], 0.0, 0.0], (pad, th_ - pad), tw_, th_, g.center)
        ox, oy = int(round(x - pad)), int(round(y_g - (th_ - pad)))
        cv.place(t, ox, oy)
        gy = float(t.px([0.0, 0.0, 0.0])[0][1])
        if abs(gy - y_g) > 0.51:
            raise RuntimeError(f"ground row of {g.cid} at {gy:.2f}, expected {y_g}")
        placed.append((t, g, lab, sk, x))
        x += widths[k] + gap_eff
    dr = cv.dr
    if not normalize:
        _person(dr, x_area0 + 0.5 * person_m * scales[0], y_g, float(scales[0]))
        dr.text((x_area0 + 0.5 * person_m * scales[0], y_lab), "Person 1.75 m", font=_font(F_CAP, True), fill=INK2,
                anchor="ma")
        dr.text((x_area0 + 0.5 * person_m * scales[0], y_lab + _lh(_font(F_CAP)) + 4), "(scale reference)",
                font=_font(F_CAP), fill=INK3, anchor="ma")
    else:
        yc = y_g - h_t
        for xx in np.arange(MARGIN, PLATE_W - MARGIN, 18):
            dr.line([(xx, yc), (xx + 9, yc)], fill=GRID_MAJOR, width=2)
        dr.text((PLATE_W - MARGIN, yc - 8), "equal crown height", font=_font(F_CAP), fill=INK3, anchor="rb")
        dr.text((MARGIN, yc - 14), "HEIGHT-NORMALIZED", font=_font(F_PANEL, True), fill=ACCENT, anchor="lb")
    dr.line([(MARGIN if normalize else x_area0, y_g - 0.5), (PLATE_W - MARGIN, y_g - 0.5)], fill=INK2, width=2)
    smin = float(scales.min())
    for t, g, lab, sk, xl in placed:
        xc = xl + 0.5 * g.length * sk
        fl, fc = _font(F_CAP, True), _font(F_CAP)
        yy = y_lab
        dr.text((xc, yy), str(lab), font=fl, fill=INK, anchor="ma")
        yy += _lh(fl) + 4
        lines = [f"Crown {_mm(g.crown)} mm", f"Length {_mm(g.length)} mm", f"Mass {_kg(g.mass)} kg"]
        if normalize:
            lines.append("Reference scale (largest)" if sk / smin < 1.005 else f"Enlarged ×{sk / smin:.2f} vs. largest")
        for ln in lines:
            dr.text((xc, yy), ln, font=fc, fill=INK2, anchor="ma")
            yy += _lh(fc) + 4
        if normalize:
            Lb, nb = _nice_bar(sk, min(120.0, 0.6 * g.length * sk))
            _scale_bar(dr, xc - 0.5 * Lb * sk, yy + 6, sk, min(120.0, 0.6 * g.length * sk), small=True)
    _header(cv, title, subtitle)
    _footer(cv, y_f, "", small, scale=None if normalize else float(scales[0]),
            scale_caption="" if normalize else "all figures", warn=warn)
    path = cv.save(out_png)
    _write_sidecar(out_png, "lineup", dict(
        title=title, mode=mode, normalize=normalize,
        scales_px_per_m={g.cid: float(sk) for g, sk in zip(gs, scales)},
        enlargement_vs_largest={g.cid: float(sk / smin) for g, sk in zip(gs, scales)},
        person_height_m=None if normalize else 1.75, grid_m=None if normalize else 0.5,
        ground_row_px=y_g, candidates=[dict(label=str(lab), **g.facts()) for (_, lab), g in zip(items, gs)]))
    return path


# ------------------------------------------------------------------------------------------------
# 4. Matrix
# ------------------------------------------------------------------------------------------------
def matrix(xml_grid, row_labels, col_labels, out_png, view: str = "side", mode: str = "exterior",
           title: str = "") -> str:
    """Grid of ``view`` renders (``side``, ``front`` or ``top``) at one physical scale.

    Side and top cells are aligned on the hip-pitch axis (same x in every cell), front cells on the
    centreline; side and front cells of a row share a ground line.  Each cell is captioned from the
    model and its ledger: crown height, torso inclination and mass, and for front and top views also
    max width and torso width.  ``None`` entries in ``xml_grid`` leave a cell empty."""
    if view not in _AXES:
        raise ValueError(f"matrix view must be one of {sorted(_AXES)}")
    grid = [[(str(x) if x is not None else None) for x in row] for row in xml_grid]
    nr, nc = len(grid), max(len(r) for r in grid)
    meas = {x: _measure(x) for row in grid for x in row if x}
    gs = list(meas.values())
    title = title or f"{view.capitalize()} views at one physical scale"
    subtitle = (f"{view.capitalize()} views ({mode}), all cells at one physical scale"
                + ("; cells aligned on the hip-pitch axis" if view in ("side", "top") else "; cells centred on the centreline")
                + ". Captions from the model and its ledger.")
    avail = PLATE_W - 2 * MARGIN
    fr = _font(F_SUB, True)
    rl_w = min(320.0, max(230.0, max([_tw(w_, fr) for lab in row_labels for w_ in str(lab).split()] + [0.0]) + 40))
    gap = 36
    cw = (avail - rl_w - gap * (nc - 1)) / nc
    pad = 22
    if view == "side":
        back = max(g.hip[0] - g.lo[0] for g in gs)
        fwd = max(g.hi[0] - g.hip[0] for g in gs)
        wmax, hmax = back + fwd, max(g.top for g in gs)
    elif view == "front":
        wmax = 2 * max(max(abs(g.lo[1]), abs(g.hi[1])) for g in gs)
        hmax = max(g.top for g in gs)
    else:
        back = max(g.hip[0] - g.lo[0] for g in gs)
        fwd = max(g.hi[0] - g.hip[0] for g in gs)
        wmax = back + fwd
        hmax = 2 * max(max(abs(g.lo[1]), abs(g.hi[1])) for g in gs)
    h_cap = {"side": 520.0, "front": 640.0, "top": 300.0}[view] * (1.0 if nr <= 2 else 0.8)
    s = min((cw - 2 * pad) / wmax, h_cap / hmax)
    draw_h = int(np.ceil(hmax * s + 2 * pad))
    cap_lines = 3 if view == "side" else 4
    cap_h = cap_lines * (_lh(_font(F_CAP)) + 4) + 8
    row_h = draw_h + cap_h + 26
    y0 = _header_height(title, subtitle)
    y_cols = y0
    y_rows = y_cols + _lh(_font(F_SUB, True)) + 18
    small = _small_print_lines([
        "Models: " + "   ".join(f"{g.cid} {g.sha256[:12]}" for g in gs),
        _provenance() + (" Torso inclination: " + ", ".join(sorted({g.torso_inclination()[1] for g in gs})) + "."),
    ])
    warn = sum((_check_failures(g) for g in gs), [])
    y_f = y_rows + nr * row_h + 10
    cv = _Canvas(PLATE_W, y_f + _footer_height(len(small) + len(warn)))
    x_cells = MARGIN + rl_w
    cells = []
    for i, row in enumerate(grid):
        for j, xml in enumerate(row):
            if not xml:
                continue
            g = meas[xml]
            cx0, cy0 = x_cells + j * (cw + gap), y_rows + i * row_h
            w_, h_ = int(np.floor(cw)), draw_h
            if view == "side":
                ref, rpx = [g.hip[0], 0.0, 0.0], (pad + back * s, draw_h - pad)
            elif view == "front":
                ref, rpx = [g.center[0], 0.0, 0.0], (0.5 * cw, draw_h - pad)
            else:
                ref, rpx = [g.hip[0], 0.0, g.center[2]], (pad + back * s, 0.5 * draw_h)
            with _scene(xml, w_, h_) as sc:
                t = _ortho(sc, view, mode, s, ref, rpx, w_, h_, g.center)
            ox, oy = int(round(cx0)), int(round(cy0))
            cv.place(t, ox, oy)
            cells.append((i, j, t, g))
    _header(cv, title, subtitle)
    dr = cv.dr
    fcol = _font(F_SUB, True)
    for j, lab in enumerate(col_labels[:nc]):
        dr.text((x_cells + j * (cw + gap) + cw / 2, y_cols), str(lab), font=fcol, fill=INK, anchor="ma")
    for i, lab in enumerate(row_labels[:nr]):
        lines = _wrap(str(lab), fr, rl_w - 24)
        yc = y_rows + i * row_h + draw_h / 2 - len(lines) * (_lh(fr) + 4) / 2
        for ln in lines:
            dr.text((MARGIN, yc), ln, font=fr, fill=INK)
            yc += _lh(fr) + 4
        if i > 0:
            yl = y_rows + i * row_h - 14
            dr.line([(MARGIN, yl), (PLATE_W - MARGIN, yl)], fill=RULE, width=1)
    fc, fb = _font(F_CAP), _font(F_CAP, True)
    for i, j, t, g in cells:
        cx0 = x_cells + j * (cw + gap)
        if view in ("side", "front"):
            gy = float(t.px([0.0, 0.0, 0.0])[0][1])
            xs = (t.px([g.tail, g.nose])[:, 0] if view == "side" else t.px([g.wlo, g.whi])[:, 0])
            dr.line([(xs.min() - 18, gy - 0.5), (xs.max() + 18, gy - 0.5)], fill=INK2, width=2)
        incl, _ = g.torso_inclination()
        yy = y_rows + i * row_h + draw_h + 6
        dr.text((cx0 + cw / 2, yy), g.cid, font=fb, fill=INK, anchor="ma")
        yy += _lh(fb) + 4
        lines = [f"Crown {_mm(g.crown)} mm  ·  Torso {incl:.0f}°", f"Mass {_kg(g.mass)} kg"]
        if view != "side":
            lines.append(f"Max width {_mm(g.max_width)} mm  ·  Torso width {_mm(g.torso_width)} mm")
        for ln in lines:
            dr.text((cx0 + cw / 2, yy), ln, font=fc, fill=INK2, anchor="ma")
            yy += _lh(fc) + 4
    _footer(cv, y_f, "", small, scale=s, scale_caption="every cell", warn=warn)
    path = cv.save(out_png)
    _write_sidecar(out_png, "matrix", dict(
        title=title, view=view, mode=mode, scale_px_per_m=s, row_labels=list(map(str, row_labels)),
        col_labels=list(map(str, col_labels)),
        cells=[dict(row=i, col=j, **g.facts()) for i, j, _, g in cells]))
    return path


# ------------------------------------------------------------------------------------------------
# 5. Silhouette sheet
# ------------------------------------------------------------------------------------------------
def silhouette_sheet(xmls_with_labels, out_png, title: str = "", subtitle: str = "") -> str:
    """Compact black silhouettes at one physical scale.  For each candidate: the side view, the front
    view to its right on the same ground line, and the top view below the side view (aligned in x).
    Candidates are laid out in a grid (the column count that gives the largest scale within about
    1700 px of drawing height); each grid row is as tall as its tallest candidate."""
    items = [(str(x), lab) for x, lab in xmls_with_labels]
    gs = [_measure(x) for x, _ in items]
    n = len(gs)
    title = title or "Silhouettes: side, front and top at one physical scale"
    subtitle = subtitle or ("For each candidate: side view, front view to its right (same ground line) and top view "
                            "below (aligned with the side view). One physical scale for the whole sheet.")
    avail = PLATE_W - 2 * MARGIN
    Lm = max(g.length for g in gs)
    Wm = max(g.max_width for g in gs)
    gap_f, gap_t, gap_b, gap_r = 26, 18, 44, 34
    lab_h = _lh(_font(F_CAP, True)) + 10
    best = None
    for nc in range(1, n + 1):
        rows = [list(range(r * nc, min(n, (r + 1) * nc))) for r in range(int(np.ceil(n / nc)))]
        hw = sum(max(gs[k].top for k in row) + max(gs[k].max_width for k in row) for row in rows)
        s_w = (avail - gap_b * (nc - 1) - nc * gap_f) / (nc * (Lm + Wm))
        s_h = (1700.0 - len(rows) * (lab_h + gap_t + gap_r)) / hw
        if best is None or min(s_w, s_h) > best[0] + 1e-9:
            best = (min(s_w, s_h), nc, rows)
    s, nc, rows = best
    bw = (avail - gap_b * (nc - 1)) / nc
    y0 = _header_height(title, subtitle)
    row_top, y = [], float(y0)
    for row in rows:
        row_top.append(y)
        y += lab_h + (max(gs[k].top for k in row) + max(gs[k].max_width for k in row)) * s + gap_t + gap_r
    small = _small_print_lines([
        "Models: " + "   ".join(f"{g.cid} {g.sha256[:12]}" for g in gs),
        _provenance(),
    ])
    warn = sum((_check_failures(g) for g in gs), [])
    y_f = int(y + 6)
    cv = _Canvas(PLATE_W, y_f + _footer_height(len(small) + len(warn)))
    pad = 6
    blocks = []
    for r, row in enumerate(rows):
        rH = max(gs[k].top for k in row)
        rW = max(gs[k].max_width for k in row)
        y_g = row_top[r] + lab_h + rH * s
        y_c = y_g + gap_t + 0.5 * rW * s  # centreline of the row's top views
        for c, k in enumerate(row):
            (xml, lab), g = items[k], gs[k]
            gw = (g.length + g.max_width) * s + gap_f
            x_side = MARGIN + c * (bw + gap_b) + 0.5 * (bw - gw)
            x_front = x_side + g.length * s + gap_f
            sw, sh = int(np.ceil(g.length * s + 2 * pad)), int(np.ceil(g.top * s + 2 * pad))
            fw = int(np.ceil(g.max_width * s + 2 * pad))
            th = int(np.ceil(g.max_width * s + 2 * pad))
            y_mid = 0.5 * (g.lo[1] + g.hi[1])
            with _scene(xml, max(sw, fw), max(sh, th)) as sc:
                ts = _ortho(sc, "side", "silhouette", s, [g.lo[0], 0.0, 0.0], (pad, sh - pad), sw, sh, g.center)
                tf = _ortho(sc, "front", "silhouette", s, [g.center[0], g.lo[1], 0.0], (pad, sh - pad), fw, sh,
                            g.center)
                tt = _ortho(sc, "top", "silhouette", s, [g.lo[0], y_mid, g.center[2]], (pad, 0.5 * th), sw, th,
                            g.center)
            cv.place(ts, int(round(x_side - pad)), int(round(y_g - (sh - pad))))
            cv.place(tf, int(round(x_front - pad)), int(round(y_g - (sh - pad))))
            cv.place(tt, int(round(x_side - pad)), int(round(y_c - 0.5 * th)))
            blocks.append((x_side, x_front, y_g, g, lab))
    _header(cv, title, subtitle)
    dr = cv.dr
    for x_side, x_front, y_g, g, lab in blocks:
        y_lab = y_g - g.top * s - lab_h
        dr.text((x_side, y_lab), str(lab), font=_font(F_CAP, True), fill=INK)
        dr.text((x_front + g.max_width * s, y_lab + 3), f"crown {_mm(g.crown)} mm · {_kg(g.mass)} kg",
                font=_font(F_SMALL), fill=INK3, anchor="ra")
        dr.line([(x_side - 10, y_g + 0.5), (x_side + g.length * s + 10, y_g + 0.5)], fill=GRID_MAJOR, width=1)
        dr.line([(x_front - 10, y_g + 0.5), (x_front + g.max_width * s + 10, y_g + 0.5)], fill=GRID_MAJOR, width=1)
    _footer(cv, y_f, "", small, scale=s, scale_caption="every silhouette", warn=warn)
    path = cv.save(out_png)
    _write_sidecar(out_png, "silhouette_sheet", dict(
        title=title, scale_px_per_m=s, columns=nc,
        candidates=[dict(label=str(lab), **g.facts()) for (_, lab), g in zip(items, gs)]))
    return path
