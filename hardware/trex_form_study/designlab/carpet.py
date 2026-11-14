"""Carpet screen (round two): contact-model parameters, toe catch, friction and a compliant-floor standing check.

This is a screen on the compiled model, not a controller.  Nothing here shows that the robot can walk, turn or
stand unaided on carpet.  Items 2 and 3 come from the prescribed reference walks (``gait.generate`` with
``sizing.NOMINAL`` and ``sizing.BRISK``).  Item 4 holds every joint at its home angle with stiff position servos:
a servo approximation with no balance control.  Where a result would need a controller, the output says so.

Data
----
``data/research/carpet_2026-11-07.json`` holds every carpet number with its source (URL, access date 2026-11-07),
a verification note and a single-source flag, and records the repository's existing ``robot_floor`` carpet
verbatim.  The values used here (its ``screening_values``):

* pile classes at the top of each industry band: low 6.35 mm (1/4 in), medium 12.7 mm (1/2 in), high 19.05 mm
  (3/4 in);
* compressed depth ``delta = f x pile height``, f being the compression from 1.96 to 98 kPa as a fraction of pile
  length measured by Hirai and Gunji (2001): 0.46 for low pile, 0.74 for medium pile (a 7 mm cut-pile rug; the
  figure includes its non-woven backing); high pile carries 0.74 over (assumption).  delta = 2.9, 9.4 and
  14.1 mm, so medium and high fall in BEHAVIOR_SPEC's 8-15 mm compliant pile depth.  All three rest on one paper;
* carpet friction band 0.54-1.25 for rubber-type soles (Hirai and Gunji 2001; Brandao et al. 2017 measured
  0.82 +- 0.02 on a robot foot);
* swing targets 0.06 H and 15 mm above carpet, 0.04 H on hard floor (BEHAVIOR_SPEC section 1).

1. Pile model
-------------
The pile is linear and local (a Winkler foundation): every point of the foot underside that is below the
uncompressed pile top is pushed up with pressure ``k_A`` times its depth; nothing else is pushed.

* ``k_A = (98 - 1.96) kPa / delta``, the secant stiffness per unit area of the sourced compression;
* the surface is the whole underside of each foot: the lower envelope of its pads, toes, claws, hallux and foot
  plate on a 0.5 mm grid (``foot_underside``; level boxes are exact, each cell taking the covered fraction of a
  pad).  At home the toe capsules sit 0.8-1.0 mm above the pad soles, the claws about 1.8 mm, the hallux 6-8 mm
  and the plate 10-13 mm, so they carry load once the pads sink past them.  The pads alone are kept for
  comparison (``geometry="pads"``).

MuJoCo (item 4).  A soft contact settles where its normal force is ``f = k d r / R``, with
``k = 1 / (d_w^2 tc^2 z^2)`` for ``solref = (tc, z)``, ``R = (1 - d) / d * A`` and A the contact's diagonal
inverse inertia, which for a contact between a foot and the world is ``body_invweight0[foot, 0]``
(``efc_diagA``; checked in MuJoCo 3.10).  Each contact is therefore a spring of stiffness
``K = d^2 / (d_w^2 tc^2 z^2 (1 - d) A)``.  A belongs to the foot, so a time-constant solref is a different
stiffness on every robot, and a free body's static depth does not depend on its weight.  The pile floor
therefore fixes the stiffness of every contact (``pile_floor``, ``pile_model``):

* the underside is sampled by point springs ("probes": 0.5 mm spheres on the foot body, one per 4 mm cell or
  covered part of a cell, the lowest point on the underside), each with ``K = k_A x its area``, the pile under
  it.  In this copy of the model the pad boxes do not touch the floor: the probes carry the robot, each part of
  the foot in proportion to its area;
* constant impedance ``d = d_w = 0.5`` (assumption), so each probe is a linear spring with
  ``tc = 1 / (z sqrt(K (1 - d) A))`` and ``solimp = (0.5, 0.5, delta, 0.5, 2)`` (the width records the compliant
  depth and has no effect while d_0 = d_w).  The static stiffness does not depend on d.  MuJoCo's default 0.9 is
  not used because the contact regularisation adds an inertia of ``d / ((1 - d) A)`` per contact (5-8 kg at 0.9,
  0.6-0.9 kg at 0.5 here); with several hundred probes, 0.9 left the servo-held stance rocking heel to toe at
  about 2.5 Hz without settling;
* ``z = 1`` is MuJoCo's constraint-space damping ratio, an assumption standing in for carpet damping, which was
  not found.  It is not a physical damping ratio;
* friction 0.54 (the band's lower edge), condim 3, priority above the floor.
Only the static stiffness is matched.  How fast a stance moves is set by the assumed damping and by that
regularisation inertia, so the time a stance takes to tip is not a result, and these floors are not suited to
touchdown or other dynamic studies, or to controller training, until carpet damping is sourced.

2. Toe catch (prescribed reference walks)
-----------------------------------------
At every frame of each walk the exact lowest point of every geom on each foot body is computed: pad boxes, toe
and claw capsules, hallux and foot plate, whatever their collision flags.  So the toes and claws count, not
just the sole site.  A swing runs from lift-off (phase 0) to touchdown, the first stance frame, kept at its own
phase (up to one frame past 1).  Heights are measured from the stance sole plane (the walk's floor, z = 0).

* Hard floor: as in round one, the screened minimum is over mid-swing, 25-75 % of the swing
  (``clearance.MID_SWING``), because the prescribed lift starts and ends on the floor.
* Carpet: the stance sole sinks into the pile, so the uncompressed pile top lies that sinkage above the walk's
  floor and the clearance above it is ``z - sinkage``.  Three sinkages per pile class: the 98 kPa compressed
  depth delta (the screened value and a deep bound: at it the hallux and, on high pile, the plate would be in the
  pile and carry load); the whole-foot sinkage of the linear pile under the walk's peak foot load (the central
  estimate; a real pile stiffens as it packs, so a linear pile can understate sinkage below 98 kPa); and the full
  pile height (a bound).  Per pile: the mid-swing minimum and its phase; pass or fail against max(0.06 H, 15 mm)
  and against the pile top (> 0); the share of the swing that meets the target; and the toe-catch exposure
  (where the foot leaves the pile after lift-off and re-enters it before touchdown, its forward travel inside the
  pile at each end, and its forward speed at re-entry).
* The lift that would meet the target in mid-swing is ``lift_frac x (target + sinkage) / hard-floor mid-swing
  minimum``.  This assumes the lowest point scales with the lift, which holds while the flat pads are lowest.
* The prescribed walk keeps the swing foot flat.  The toe-down pitch that would bring the foot to the pile top at
  the worst mid-swing frame is found by turning the swing ankle-pitch joint (bisection on the lowest point of
  every foot geom; the geom that gets there first is named).  It is reported, not simulated.

3. Friction (prescribed reference walks)
----------------------------------------
``gait.inverse_dynamics`` gives the required friction ratio |f_t| / f_n per frame, the largest over the stance
feet (``required_max_gait_inverse_dynamics``).  At round one's walk accuracy (``gait.set_accuracy``) its finite
differences unwrap root x across the period boundary as ``x_{n-1} + x_1 - x_0`` instead of ``x_0 + stride``,
which puts an acceleration error of 0.5-1.2 m/s^2 on frames 0, 1, n-2 and n-1.  The screen recomputes those
frames, and every double-support frame, with the exact unwrap (``derivatives``).  At round two's accuracy
``gait.inverse_dynamics`` already uses the exact unwrap, so no frame is recomputed for it
(``frames_recomputed_for_unwrap`` is empty) and only the double-support frames are.  Single support is unique.  Double support uses the minimum-effort split with a 0.8
friction pyramid per axis per foot (``statics._double_support_qp``), which ignores friction use and can load a
nearly unloaded foot sideways just before it lifts (``required_max_as_reported``).  So the screen also computes
the total double-support ratio ``|sum f_t| / sum f_n``, the least any split needs.  The governing requirement
is the larger of the single-support maximum and that total.  The realised CoM path carries frame-to-frame noise
(``gait.generate`` holds the CoM to 0.2 mm at round one's accuracy and to 1e-6 m at round two's), so the
requirement is also given from the CoM path low-passed to its
first 12 stride harmonics (``|a_xy| / (g + a_z)``, the least any split needs at every frame) and as the 95th
percentile of the governing series.  Margins (available minus governing) are given against the carpet band's
lower edge, its foot-pressure subset, the robot-foot carpet value, and hard floor: the compiled model (0.9),
robot_floor (0.80) and the lower edge of the sourced robot-foot band on ceramic, marble and wood.  Double-support
frames where a foot sits on the QP's pyramid (its x or y ratio at least 0.79) are counted.  The peak mean pad
pressure is reported beside the sources' test pressures (98 kPa compression, about 140 kPa friction).

4. Standing check (contact-model sanity check, servo approximation)
-------------------------------------------------------------------
A fresh copy of the compiled model per floor: as compiled (hard control), robot_floor's carpet (on the pads, the
compiled model's only floor contacts) and the low, medium and high pile of item 1 under the whole foot.  Every
actuated joint is held at q_home by its position servo with ``kp = sustained planning torque / 0.01 rad`` and
``kv = 2 sqrt(kp M_jj)`` (critical damping of the joint alone, M_jj from the mass matrix at q_home); force limits
stay at the planning peak.  The robot starts at q_home lowered by the predicted sinkage and runs at least 10 s,
then on in 1 s steps until it has settled or tipped, up to 30 s.  Settled: over the last second the trunk tilt
spanned less than 0.05 deg and the CoM less than 0.5 mm, and neither moved more than in the second before (near
the boundary a stance can creep for seconds and then fall; ``_settled``).  Upright means settled, a trunk tilt
that never exceeded 12 deg (PR #662's hold gate) and a CoM drop under 0.1 H; a stance still creeping at 30 s is
reported as unsettled and is not counted upright.  Reported: per-foot sinkage (depth of the sole plane at the
pads' area centroid, and the deepest contact) against the prediction for the measured foot load, CoM drift and
drop, trunk tilt and its span over the last second, and servo saturation.  If the robot tips, sinkage is averaged
before the tilt reaches 2 deg.

5. Passive tip (rigid robot on the unilateral pile)
---------------------------------------------------
A servo-held robot is nearly a rigid body on the pile.  The linear tip margin ``sum_i k_i (x_i - x_c)^2 / (W h)``
over the loaded cells (x_c their stiffness centroid, W the weight, h the CoM height) is a necessary condition only:
the CoM sits behind the stiffness centroid, so the robot leans back until the front of the foot unloads, the
restoring stiffness falls, and the stance can tip at margins well above 1.  ``stance_statics`` therefore solves the
static equilibrium with contacts that can only push: for each lean angle, the sinkage that carries the weight and
the centre of pressure.  The stance holds if the centre of pressure passes under the displaced CoM within 12 deg
(pitch and roll separately).  The softest pile that holds is found by bisection on k_A and given as its
compression at 98 kPa, for the whole foot and for the pads alone.  This rigid result ignores the servos; on the
round-one models the servo-held simulation holds on piles 3-5 % softer than it allows.  ``--critical``
(``critical_pile_sim``) also finds the boundary by bisection in the servo-held simulation.

Limits
------
The pile is linear and local: only the 98 kPa end point of its compression curve is sourced, measured over 5 min
on a presser foot of about 1 cm2 (a stance lasts 0.5-0.7 s on pads of 3-8 cm2), and the medium figure includes a
rug backing.  Pile direction, seams, loose rugs, thresholds and wear are not represented.  There is no toe-drag
force and no kinetic friction.  The standing check has no balance control, and only its static stiffness is
calibrated.  The reference walks are prescribed, so there is no slip, impact or touchdown dynamics.

Run from hardware/trex_form_study:
    python -m designlab.carpet 60-M-T-10 75-M-H-10 --out <scratch dir> [--critical]
writes ``<out>/<id>_carpet.json`` and ``<out>/<id>_carpet.png``, with the rebuilt round-one models under
``<out>/mjcf``.  Round two calls ``screen_record`` from ``round2.screen_one``; its summary carries the keys
``shortlist`` reads (``pass``, ``toe_clearance_margin_mm``, ``friction_margin``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from functools import lru_cache
from pathlib import Path

import mujoco
import numpy as np

from . import gait, mjcf, sizing, statics
from .clearance import MID_SWING

DATA = Path(__file__).resolve().parent / "data" / "research" / "carpet_2026-11-07.json"
STUDY = Path(__file__).resolve().parents[1]
PILES = ("low", "medium", "high")
G = 9.81
STAND_SECONDS = 10.0  # the standing check runs at least this long ...
STAND_MAX_SECONDS = 30.0  # ... then on in 1 s steps until it settles or tips, up to this
SETTLE_WINDOW_S = 1.0  # settled values and changes are taken over the last second
SETTLED_TILT_DEG = 0.05  # settled: the trunk tilt changed by less than this over the last second ...
SETTLED_COM_MM = 0.5  # ... and the CoM moved by less than this
LOG_DT = 0.02
SERVO_ERROR_RAD = 0.01  # the hold servo reaches the drive's sustained planning torque at this error (assumption)
SERVO_ZETA = 1.0
UPRIGHT_TILT_DEG = 12.0  # PR #662's hold gate ('torso tilt below 12 deg')
ONSET_TILT_DEG = 2.0
COLLAPSE_FRAC_H = 0.10
QP_BOUND = 0.79  # a double-support foot whose x or y ratio reaches this sits on statics' 0.8 friction pyramid
BAND_HARMONICS = 12  # stride harmonics kept for the band-limited friction requirement
MAP_GRID_M = 0.0005  # height map of the foot underside for the pile statics
PROBE_GRID_M = 0.004  # probe spacing of the simulated pile
PROBE_RADIUS_M = 0.0005
PROBE_BIT = 2  # contype bit of the probes; the floor's conaffinity gains it
LEAN_STEP_DEG = 0.25
TOE_TOKENS = ("toe", "claw")
_GT = mujoco.mjtGeom
_EZ = np.array([0.0, 0.0, 1.0])


# ------------------------------------------------------------------------------------------------
# Data
# ------------------------------------------------------------------------------------------------
@lru_cache(maxsize=1)
def research() -> dict:
    return json.loads(DATA.read_text())


def screening_values() -> dict:
    return research()["screening_values"]


def pile_classes() -> dict:
    """Height, compression fraction, compressed depth and secant stiffness per unit area of each pile class."""
    sv = screening_values()
    p_hi = 1e3 * sv["test_pressure_kPa"]["value"]
    p_lo = 1e3 * sv["reference_pressure_kPa"]["value"]
    out = {}
    for name in PILES:
        c = sv["pile_classes"][name]
        h = 1e-3 * c["pile_height_mm"]
        delta = c["compression_fraction"] * h
        out[name] = dict(height_m=h, compression_fraction=c["compression_fraction"], depth_m=delta,
                         stiffness_pa_per_m=(p_hi - p_lo) / delta)
    return out


def pressure_span_pa() -> float:
    """Pressure span of the sourced compression (98 - 1.96 kPa), in Pa."""
    sv = screening_values()
    return 1e3 * (sv["test_pressure_kPa"]["value"] - sv["reference_pressure_kPa"]["value"])


def carpet_target(H: float) -> float:
    t = screening_values()["swing_targets"]
    return max(t["carpet_frac_H"] * H, 1e-3 * t["carpet_min_mm"])


def friction_references() -> dict:
    """Available friction coefficients the required ratios are compared with."""
    sv = screening_values()
    hard = sv["hard_floor_friction"]
    return {"carpet_band_low": sv["carpet_friction_band"]["value"][0],
            "carpet_foot_pressure_low": sv["carpet_friction_foot_pressure"]["value"][0],
            "carpet_robot_foot": research()["friction"]["brandao_2017"]["table_I_mean_sd"]["carpet"][0],
            "hard_model": hard["model"], "hard_robot_floor": hard["robot_floor_hard"],
            "hard_robot_foot_low": hard["robot_foot_band"][0]}


# ------------------------------------------------------------------------------------------------
# Geometry
# ------------------------------------------------------------------------------------------------
def lowest_point(m, d, g) -> np.ndarray:
    """World coordinates of the lowest point of geom g (exact for box, capsule, cylinder, sphere, ellipsoid, mesh)."""
    R = d.geom_xmat[g].reshape(3, 3)
    c = d.geom_xpos[g].copy()
    s = m.geom_size[g]
    t = m.geom_type[g]
    if t == _GT.mjGEOM_BOX:
        return c + R @ (-np.sign(R[2]) * s[:3])
    if t in (_GT.mjGEOM_CAPSULE, _GT.mjGEOM_CYLINDER):
        a = R[:, 2]
        end = c - (1.0 if a[2] >= 0 else -1.0) * s[1] * a
        if t == _GT.mjGEOM_CAPSULE:
            return end - s[0] * _EZ
        u = _EZ - a[2] * a
        nu = np.linalg.norm(u)
        return end - s[0] * u / nu if nu > 1e-12 else end
    if t == _GT.mjGEOM_SPHERE:
        return c - s[0] * _EZ
    if t == _GT.mjGEOM_ELLIPSOID:
        s2 = s[:3] ** 2
        return c + R @ (-s2 * R[2] / np.sqrt(np.sum(s2 * R[2] ** 2)))
    if t == _GT.mjGEOM_MESH:
        mid = m.geom_dataid[g]
        a, n = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
        w = c + m.mesh_vert[a : a + n] @ R.T
        return w[np.argmin(w[:, 2])].copy()
    return c - m.geom_rbound[g] * _EZ


def foot_geoms(m) -> dict:
    """Every geom with extent on each foot body (pads, toes, claws, hallux, plate)."""
    out = {}
    for side in statics.SIDES:
        b = statics.bid(m, f"{side}_foot")
        out[side] = [g for g in range(m.ngeom) if m.geom_bodyid[g] == b and m.geom_rbound[g] > 0
                     and np.any(m.geom_size[g] > 0)]
    return out


def pad_geoms(m) -> dict:
    out = {}
    for side in statics.SIDES:
        b = statics.bid(m, f"{side}_foot")
        out[side] = [g for g in range(m.ngeom) if m.geom_bodyid[g] == b and m.geom_contype[g] == 1
                     and m.geom_type[g] == _GT.mjGEOM_BOX]
    return out


def pad_area(m, side: str) -> float:
    """Bottom-face area of one foot's floor-contact pad boxes (m^2; the pads are level at home)."""
    return float(sum(4.0 * m.geom_size[g][0] * m.geom_size[g][1] for g in pad_geoms(m)[side]))


def _gname(m, g) -> str:
    return mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g)


def _is_toe(m, g) -> bool:
    """Toe and claw geoms.  The floor-contact pads under the toes (``pad_toe*``) are pads, not toe tips."""
    return m.geom_contype[g] != 1 and any(tok in _gname(m, g) for tok in TOE_TOKENS)


def _xy_extent(m, d, g):
    R = d.geom_xmat[g].reshape(3, 3)
    c = d.geom_xpos[g]
    s = m.geom_size[g]
    t = m.geom_type[g]
    if t == _GT.mjGEOM_BOX:
        pts = np.array([c + R @ (np.array([a, b, e]) * s[:3]) for a in (-1, 1) for b in (-1, 1) for e in (-1, 1)])
        return pts[:, :2].min(0), pts[:, :2].max(0)
    if t == _GT.mjGEOM_CAPSULE:
        ends = np.array([c - s[1] * R[:, 2], c + s[1] * R[:, 2]])
        return ends[:, :2].min(0) - s[0], ends[:, :2].max(0) + s[0]
    if t == _GT.mjGEOM_SPHERE:
        return c[:2] - s[0], c[:2] + s[0]
    raise ValueError(f"geom {_gname(m, g)}: type {int(t)} is not supported by the underside map")


def underside_cells(m, d, geoms, h: float) -> dict:
    """Lower envelope of the union of ``geoms`` on a square grid of spacing h (world xy), as area elements: per element
    its position, the height z of the surface, its area and the geom that owns it.

    Level boxes (the pads and plate; the feet are flat at home and in stance) are exact: each cell gets the box's
    covered fraction, placed at the covered part's centroid (sampled 4 x 4 when the box is turned in plan).  Capsules
    (sampled along their axis every 0.25 mm or closer) and spheres take the cell's centre.  Where surfaces overlap
    in a cell the lowest takes its fraction first and the next lowest the rest."""
    ext = [_xy_extent(m, d, g) for g in geoms]
    lo = np.min([e[0] for e in ext], axis=0)
    hi = np.max([e[1] for e in ext], axis=0)
    nx, ny = (np.ceil((hi - lo) / h).astype(int) + 1).tolist()
    xs = lo[0] + (np.arange(nx) + 0.5) * h
    ys = lo[1] + (np.arange(ny) + 0.5) * h
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    ng = len(geoms)
    Zl = np.full((ng, nx, ny), np.inf)
    Fl = np.zeros((ng, nx, ny))
    Px, Py = np.broadcast_to(X, (ng, nx, ny)).copy(), np.broadcast_to(Y, (ng, nx, ny)).copy()
    sub = (np.arange(4) - 1.5) / 4.0 * h
    for gi, (g, (glo, ghi)) in enumerate(zip(geoms, ext, strict=True)):
        i0, i1 = max(0, int((glo[0] - lo[0]) / h) - 1), min(nx, int((ghi[0] - lo[0]) / h) + 2)
        j0, j1 = max(0, int((glo[1] - lo[1]) / h) - 1), min(ny, int((ghi[1] - lo[1]) / h) + 2)
        Xw, Yw = X[i0:i1, j0:j1], Y[i0:i1, j0:j1]
        R = d.geom_xmat[g].reshape(3, 3)
        c = d.geom_xpos[g]
        s = m.geom_size[g]
        t = m.geom_type[g]
        if t == _GT.mjGEOM_BOX:
            if abs(R[2, 2]) < 1.0 - 1e-6:
                raise ValueError(f"geom {_gname(m, g)}: the underside map needs level boxes")
            if max(abs(R[0, 0]), abs(R[0, 1])) > 1.0 - 1e-9:  # box axes along world x and y: exact overlap
                hx, hy = (s[0], s[1]) if abs(R[0, 0]) > 0.5 else (s[1], s[0])
                x0, x1 = np.maximum(c[0] - hx, Xw - h / 2), np.minimum(c[0] + hx, Xw + h / 2)
                y0, y1 = np.maximum(c[1] - hy, Yw - h / 2), np.minimum(c[1] + hy, Yw + h / 2)
                frac = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None) / h**2
                px, py = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
            else:  # turned in plan: 4 x 4 samples per cell
                SX = Xw[..., None, None] + sub[:, None]
                SY = Yw[..., None, None] + sub[None, :]
                dx, dy = SX - c[0], SY - c[1]
                inside = (np.abs(R[0, 0] * dx + R[1, 0] * dy) <= s[0]) & (np.abs(R[0, 1] * dx + R[1, 1] * dy) <= s[1])
                n_in = inside.sum(axis=(-1, -2))
                frac = n_in / 16.0
                px = np.where(n_in > 0, (SX * inside).sum(axis=(-1, -2)) / np.maximum(n_in, 1), Xw)
                py = np.where(n_in > 0, (SY * inside).sum(axis=(-1, -2)) / np.maximum(n_in, 1), Yw)
            zb = np.where(frac > 0, c[2] - s[2], np.inf)
            Px[gi, i0:i1, j0:j1], Py[gi, i0:i1, j0:j1] = px, py
        elif t == _GT.mjGEOM_CAPSULE:
            a, r = R[:, 2], s[0]
            zb = np.full(Xw.shape, np.inf)
            for u in np.linspace(-s[1], s[1], max(9, int(np.ceil(2.0 * s[1] / 0.00025)) + 1)):
                p = c + u * a
                rho2 = (Xw - p[0]) ** 2 + (Yw - p[1]) ** 2
                ok = rho2 < r * r
                if ok.any():
                    zb = np.where(ok, np.minimum(zb, p[2] - np.sqrt(np.where(ok, r * r - rho2, 0.0))), zb)
            frac = np.isfinite(zb).astype(float)
        else:  # sphere
            rho2 = (Xw - c[0]) ** 2 + (Yw - c[1]) ** 2
            ok = rho2 < s[0] ** 2
            zb = np.where(ok, c[2] - np.sqrt(np.where(ok, s[0] ** 2 - rho2, 0.0)), np.inf)
            frac = ok.astype(float)
        Zl[gi, i0:i1, j0:j1], Fl[gi, i0:i1, j0:j1] = zb, frac
    order = np.argsort(Zl, axis=0)
    Zs = np.take_along_axis(Zl, order, 0)
    Fs = np.where(np.isfinite(Zs), np.take_along_axis(Fl, order, 0), 0.0)
    cum = np.minimum(1.0, np.cumsum(Fs, axis=0))
    share = np.diff(np.concatenate([np.zeros((1, nx, ny)), cum]), axis=0)
    keep = share > 1e-9
    gid = np.asarray(geoms)[order]
    return dict(x=np.take_along_axis(Px, order, 0)[keep], y=np.take_along_axis(Py, order, 0)[keep], z=Zs[keep],
                area=share[keep] * h**2, geom=gid[keep], cell_m=float(h))


def foot_underside(M: statics.Model, q, geometry: str = "whole_foot", h: float = MAP_GRID_M) -> dict:
    """Underside cells of each foot at pose q (feet flat), z measured up from the sole plane (the pad bottoms).
    ``geometry``: "whole_foot" (every foot geom) or "pads"."""
    m, d = M.m, M.d
    M.forward(q)
    sel = foot_geoms(m) if geometry == "whole_foot" else pad_geoms(m)
    z_sole = min(lowest_point(m, d, g)[2] for side in statics.SIDES for g in pad_geoms(m)[side])
    out = dict(geometry=geometry, cell_m=float(h), z_sole=float(z_sole))
    for side in statics.SIDES:
        c = underside_cells(m, d, sel[side], h)
        c["z"] = c["z"] - z_sole
        out[side] = c
    return out


# ------------------------------------------------------------------------------------------------
# Pile statics (item 5): rigid robot on unilateral linear springs
# ------------------------------------------------------------------------------------------------
def _sink(c: np.ndarray, k: np.ndarray, force: float) -> float:
    """Sinkage s at which unilateral springs of stiffness k with contact thresholds c carry ``force``:
    ``sum_i k_i max(0, s - c_i) = force`` (exact: the sum is piecewise linear in s)."""
    o = np.argsort(c)
    cs, ks = c[o], k[o]
    K = np.cumsum(ks)
    KC = np.cumsum(ks * cs)
    j = int(np.searchsorted(cs[1:] * K[:-1] - KC[:-1], force))
    return float((force + KC[j]) / K[j])


def pile_sinkage(cells: dict, k_A: float, force: float) -> float:
    """Level sinkage (m) of one foot's underside elements in a linear pile of stiffness k_A (N/m^3) under ``force``."""
    return _sink(cells["z"], k_A * cells["area"], force)


def pile_springs(under: dict, k_A: float) -> dict:
    """Both feet's underside elements as springs of a linear pile (``foot_underside`` output)."""
    cat = {key: np.concatenate([under[s][key] for s in statics.SIDES]) for key in ("x", "y", "z", "area")}
    return dict(x=cat["x"], y=cat["y"], z=cat["z"], k=k_A * cat["area"], z_sole=under["z_sole"])


def _pad_corners(m, d, pads: dict) -> np.ndarray:
    pts = []
    for side in statics.SIDES:
        for g in pads[side]:
            R = d.geom_xmat[g].reshape(3, 3)
            h = m.geom_size[g]
            for sx in (-1, 1):
                for sy in (-1, 1):
                    pts.append(d.geom_xpos[g] + R @ np.array([sx * h[0], sy * h[1], -h[2]]))
    return np.array(pts)


def _lean(u, z, k, W: float, h: float, max_deg: float = UPRIGHT_TILT_DEG, step_deg: float = LEAN_STEP_DEG):
    """Static lean (deg, signed) of a rigid body on unilateral springs at levers u from the CoM's projection, with the
    CoM h above the sole plane: the first angle at which the springs' resultant passes under the displaced CoM.
    None if that does not happen within ``max_deg`` (the stance tips)."""

    def r(th):  # centre of pressure minus the displaced CoM, along u
        c = z - th * u
        s = _sink(c, k, W)
        return float(np.sum(k * np.maximum(s - c, 0.0) * u) / W - h * th)

    r0 = r(0.0)
    if r0 == 0.0:
        e = np.radians(0.01)
        return 0.0 if (r(e) > 0.0 > r(-e)) else None
    sgn = -np.sign(r0)
    th0 = 0.0
    for deg in np.arange(step_deg, max_deg + 1e-9, step_deg):
        th = sgn * np.radians(deg)
        if np.sign(r(th)) != np.sign(r0):
            a, b = th0, th
            for _ in range(30):
                mid = 0.5 * (a + b)
                if np.sign(r(mid)) == np.sign(r0):
                    a = mid
                else:
                    b = mid
            return float(np.degrees(0.5 * (a + b)))
        th0 = th
    return None


def stance_statics(sp: dict, W: float, com, pitch_only: bool = False) -> dict:
    """Rigid robot on the springs ``sp`` (``pile_springs`` or corner springs): level sinkage, stiffness centroid of
    the loaded springs against the CoM, linear tip margins (a necessary condition only) and the unilateral
    equilibrium lean in pitch and roll (None: the stance tips within 12 deg)."""
    u, v, z, k = sp["x"] - com[0], sp["y"] - com[1], sp["z"], sp["k"]
    h = float(com[2] - sp["z_sole"])
    s = _sink(z, k, W)
    on = z < s
    kk = k[on]
    uc, vc = float(np.sum(kk * u[on]) / kk.sum()), float(np.sum(kk * v[on]) / kk.sum())
    lin = dict(pitch=float(np.sum(kk * (u[on] - uc) ** 2) / (W * h)), roll=float(np.sum(kk * (v[on] - vc) ** 2) / (W * h)))
    pitch = _lean(u, z, k, W, h)
    roll = None if pitch_only or pitch is None else _lean(v, z, k, W, h)
    return dict(level_sinkage_mm=1e3 * s, loaded_springs=int(on.sum()),
                stiffness_centroid_ahead_of_com_mm=1e3 * uc, linear_margin=lin,
                lean_deg=dict(pitch=pitch, roll=roll), holds=bool(pitch is not None and (pitch_only or roll is not None)))


def critical_pile(under: dict, W: float, com, lo: float = 1e6, hi: float = 1e9, iters: int = 12) -> float | None:
    """Softest linear pile (k_A, N/m^3) on which the rigid robot holds (``stance_statics``), by bisection on log k_A.
    None if even ``hi`` tips; ``lo`` if ``lo`` already holds."""

    def holds(kA):  # roll is only solved when pitch holds
        return stance_statics(pile_springs(under, kA), W, com)["holds"]

    if not holds(hi):
        return None
    if holds(lo):
        return lo
    for _ in range(iters):
        mid = float(np.sqrt(lo * hi))
        if holds(mid):
            hi = mid
        else:
            lo = mid
    return hi


def passive_tip(M: statics.Model, q_home, unders: dict) -> dict:
    """Floor-independent part of item 5: the critical pile of the rigid robot for each contact geometry, and the
    centroids of the pads against the CoM."""
    m, d = M.m, M.d
    M.forward(q_home)
    W, com = M.mass() * G, M.com()
    dp = pressure_span_pa()
    out = {}
    for geo, under in unders.items():
        kc = critical_pile(under, W, com)
        rec = dict(critical_k_A_kPa_per_mm=None if kc is None else 1e-6 * kc,
                   critical_compression_98kPa_mm=None if kc is None else 1e3 * dp / kc,
                   critical_pad_stiffness_per_foot_kN_per_m=None if kc is None else 1e-3 * kc * pad_area(m, "left"))
        if kc is not None:
            st = stance_statics(pile_springs(under, kc), W, com)
            rec.update(linear_margin_at_critical=st["linear_margin"], lean_at_critical_deg=st["lean_deg"],
                       stiffness_centroid_ahead_of_com_mm=st["stiffness_centroid_ahead_of_com_mm"])
        out[geo] = rec
    M.forward(q_home)
    pads = pad_geoms(m)
    gs = [g for side in statics.SIDES for g in pads[side]]
    a = np.array([4.0 * m.geom_size[g][0] * m.geom_size[g][1] for g in gs])
    xa = float(np.sum(a * d.geom_xpos[gs, 0]) / a.sum())
    xk = float(_pad_corners(m, d, pads)[:, 0].mean())
    out["pad_area_centroid_ahead_of_com_mm"] = 1e3 * (xa - float(com[0]))
    out["pad_corner_centroid_ahead_of_com_mm"] = 1e3 * (xk - float(com[0]))
    out["weight_N"], out["com_height_m"] = W, float(com[2])
    return out


# ------------------------------------------------------------------------------------------------
# 2. Toe catch
# ------------------------------------------------------------------------------------------------
def swing_profiles(M: statics.Model, traj: dict) -> dict:
    """Per foot, from lift-off to touchdown: swing phase s, lowest point of the foot (z, and which geom), lowest pad
    and lowest toe/claw point, and the foot body's x (unwrapped across the periodic stride).  The touchdown sample
    is the first stance frame at its own phase u / sw (up to one frame past 1)."""
    m, d = M.m, M.d
    info = traj["info"]
    T, sw, stride = info["period"], 1.0 - info["duty"], info["stride"]
    t = np.asarray(traj["t"])
    n = len(t)
    geoms = foot_geoms(m)
    names = {g: _gname(m, g) for s in geoms for g in geoms[s]}
    rec = {s: dict(z=np.zeros(n), z_pad=np.zeros(n), z_toe=np.zeros(n), x=np.zeros(n), geom=[""] * n)
           for s in statics.SIDES}
    for k in range(n):
        M.forward(traj["q"][k])
        for side in statics.SIDES:
            pts = {g: lowest_point(m, d, g) for g in geoms[side]}
            g0 = min(pts, key=lambda g: pts[g][2])
            r = rec[side]
            r["z"][k], r["geom"][k] = pts[g0][2], names[g0]
            r["z_pad"][k] = min(pts[g][2] for g in pts if m.geom_contype[g] == 1)
            r["z_toe"][k] = min((pts[g][2] for g in pts if _is_toe(m, g)), default=np.nan)
            r["x"][k] = d.xpos[M.foot[side]][0]
    ph = (t / T) % 1.0
    out = {}
    for side, off in (("left", 0.0), ("right", 0.5)):
        u = (ph - off) % 1.0
        swing = np.where(u < sw - 1e-12)[0]
        swing = swing[np.argsort(u[swing])]
        stance = np.where(u >= sw - 1e-12)[0]
        td = stance[np.argmin(u[stance])]
        idx = np.concatenate([swing, [td]])
        x = rec[side]["x"][idx].copy()
        for i in range(1, len(x)):  # a swing that wraps the stride end jumps back by one stride length
            while x[i] < x[i - 1] - 0.5 * stride:
                x[i] += stride
        out[side] = dict(s=np.concatenate([u[swing] / sw, [u[td] / sw]]), z=rec[side]["z"][idx], x=x,
                         z_pad=rec[side]["z_pad"][idx], z_toe=rec[side]["z_toe"][idx], frames=idx,
                         geom=[rec[side]["geom"][i] for i in idx])
    return out


def _cross(s, c, i0: int, i1: int) -> float:
    """Phase where c crosses zero between samples i0 and i1 (linear)."""
    c0, c1 = c[i0], c[i1]
    return float(s[i1]) if c1 == c0 else float(s[i0] + (s[i1] - s[i0]) * (0.0 - c0) / (c1 - c0))


def clearance_metrics(prof: dict, delta: float, target: float, swing_time: float) -> dict:
    """Clearance of each swing foot above a surface ``delta`` above the walk's floor; worst foot first."""
    per = {}
    for side, p in prof.items():
        s, x = p["s"], p["x"]
        c = p["z"] - delta
        mid = np.where((s >= MID_SWING[0]) & (s <= MID_SWING[1]))[0]
        k = mid[np.argmin(c[mid])]
        above = np.where(c >= 0.0)[0]
        r = dict(mid_swing_min_m=float(c[k]), phase=float(s[k]), frame=int(p["frames"][k]), lowest_geom=p["geom"][k],
                 apex_m=float(c.max()), apex_phase=float(s[np.argmax(c)]),
                 swing_share_meeting_target=float(np.mean(np.interp(np.linspace(0, 1, 2001), s, c) >= target)))
        if len(above) == 0:
            r.update(leaves_surface=False, exit_phase=None, entry_phase=None, travel_after_liftoff_m=float(x[-1] - x[0]),
                     travel_before_touchdown_m=float(x[-1] - x[0]), speed_at_entry_m_s=None)
        else:
            i, j = int(above[0]), int(above[-1])
            s_exit = _cross(s, c, i - 1, i) if i > 0 else float(s[0])
            s_entry = _cross(s, c, j, j + 1) if j < len(s) - 1 else float(s[-1])
            speed = (x[j + 1] - x[j]) / ((s[j + 1] - s[j]) * swing_time) if j < len(s) - 1 else 0.0
            r.update(leaves_surface=True, exit_phase=s_exit, entry_phase=s_entry,
                     travel_after_liftoff_m=float(np.interp(s_exit, s, x) - x[0]),
                     travel_before_touchdown_m=float(x[-1] - np.interp(s_entry, s, x)), speed_at_entry_m_s=float(speed))
        per[side] = r
    worst = min(per, key=lambda sd: per[sd]["mid_swing_min_m"])
    return dict(per[worst], foot=worst, other_foot=per["right" if worst == "left" else "left"])


def toe_down_pitch(M: statics.Model, q, side: str, delta: float, max_deg: float = 45.0) -> dict:
    """Toe-down turn of the ``side`` ankle-pitch joint (deg) at which the lowest point of that foot (every foot geom)
    reaches ``delta`` above the walk's floor, and the geom that gets there first (geometry only)."""
    m, d = M.m, M.d
    qa = m.jnt_qposadr[statics.jid(m, f"{side}_ankle_pitch")]
    geoms = foot_geoms(m)[side]

    def pose(deg):
        qq = np.array(q, float).copy()
        qq[qa] += np.radians(deg)
        M.forward(qq)
        return {g: lowest_point(m, d, g) for g in geoms}

    p0 = pose(0.0)
    front = max(p0, key=lambda g: p0[g][0])
    sgn = 1.0 if pose(1.0)[front][2] < pose(-1.0)[front][2] else -1.0

    def low(deg):
        pts = pose(sgn * deg)
        g = min(pts, key=lambda gg: pts[gg][2])
        return float(pts[g][2]), g

    z0, g0 = low(0.0)
    if z0 <= delta:
        return dict(deg=0.0, first_geom=_gname(m, g0))
    if low(max_deg)[0] > delta:
        return dict(deg=None, first_geom=None)
    a, b = 0.0, max_deg
    for _ in range(40):
        mid = 0.5 * (a + b)
        if low(mid)[0] > delta:
            a = mid
        else:
            b = mid
    return dict(deg=float(b), first_geom=_gname(m, low(b)[1]))


def _exposure(r: dict) -> dict:
    return dict(leaves_pile=r["leaves_surface"], liftoff_exit_phase=r["exit_phase"],
                travel_in_pile_after_liftoff_mm=1e3 * r["travel_after_liftoff_m"], touchdown_entry_phase=r["entry_phase"],
                travel_in_pile_before_touchdown_mm=1e3 * r["travel_before_touchdown_m"],
                forward_speed_at_entry_mm_s=None if r["speed_at_entry_m_s"] is None else 1e3 * r["speed_at_entry_m_s"])


def toe_catch(M: statics.Model, traj: dict, H: float, piles: dict | None = None,
              sinkage_load: dict | None = None) -> dict:
    """Swing-foot clearance of one reference walk over hard floor and each pile class (module docstring, item 2).
    ``sinkage_load`` (pile -> m) adds the central estimate: the stance foot's whole-underside sinkage under the
    walk's peak foot load."""
    piles = piles or pile_classes()
    prof = swing_profiles(M, traj)
    info = traj["info"]
    sw_time = (1.0 - info["duty"]) * info["period"]
    target = carpet_target(H)
    hard_frac = screening_values()["swing_targets"]["hard_frac_H"]
    hard = clearance_metrics(prof, 0.0, hard_frac * H, sw_time)
    lift_frac = info["lift"] / H
    toe_mid = min(float(np.min(p["z_toe"][(p["s"] >= MID_SWING[0]) & (p["s"] <= MID_SWING[1])])) for p in prof.values())
    q_worst = traj["q"][hard["frame"]]

    def need(sink):
        return lift_frac * (target + sink) / hard["mid_swing_min_m"] if hard["mid_swing_min_m"] > 0 else None

    out = dict(lift_frac_H=lift_frac, lift_mm=1e3 * info["lift"], swing_time_s=sw_time,
               frames_per_swing=len(prof["left"]["s"]) - 1, touchdown_phase=float(prof["left"]["s"][-1]),
               target_carpet_mm=1e3 * target, target_hard_mm=1e3 * hard_frac * H,
               hard_floor=dict(mid_swing_min_mm=1e3 * hard["mid_swing_min_m"],
                               mid_swing_min_frac_H=hard["mid_swing_min_m"] / H, phase=hard["phase"], foot=hard["foot"],
                               frame=hard["frame"], lowest_geom=hard["lowest_geom"], apex_mm=1e3 * hard["apex_m"],
                               apex_phase=hard["apex_phase"], meets_hard_target=hard["mid_swing_min_m"] >= hard_frac * H,
                               meets_carpet_target_without_pile=hard["mid_swing_min_m"] >= target,
                               toe_points_mid_swing_min_mm=1e3 * toe_mid),
               piles={},
               profile={side: dict(s=np.round(p["s"], 4).tolist(), z_mm=np.round(1e3 * p["z"], 2).tolist(),
                                   z_toe_mm=np.round(1e3 * p["z_toe"], 2).tolist(),
                                   x_mm=np.round(1e3 * (p["x"] - p["x"][0]), 2).tolist())
                        for side, p in prof.items()})
    for name, pc in piles.items():
        r = clearance_metrics(prof, pc["depth_m"], target, sw_time)
        b = clearance_metrics(prof, pc["height_m"], target, sw_time)
        pitch = toe_down_pitch(M, q_worst, hard["foot"], pc["depth_m"])
        rec = dict(pile_height_mm=1e3 * pc["height_m"], stance_sinkage_mm=1e3 * pc["depth_m"],
                   mid_swing_min_above_pile_top_mm=1e3 * r["mid_swing_min_m"], phase=r["phase"], foot=r["foot"],
                   lowest_geom=r["lowest_geom"], clears_pile_top=r["mid_swing_min_m"] > 0.0,
                   meets_carpet_target=r["mid_swing_min_m"] >= target, apex_above_pile_top_mm=1e3 * r["apex_m"],
                   swing_share_meeting_target=r["swing_share_meeting_target"], **_exposure(r),
                   lift_frac_needed_for_target=need(pc["depth_m"]),
                   toe_down_pitch_to_reach_pile_top_deg=pitch["deg"], toe_down_first_geom=pitch["first_geom"],
                   fully_compressed_bound=dict(stance_sinkage_mm=1e3 * pc["height_m"],
                                               mid_swing_min_above_pile_top_mm=1e3 * b["mid_swing_min_m"],
                                               meets_carpet_target=b["mid_swing_min_m"] >= target,
                                               clears_pile_top=b["mid_swing_min_m"] > 0.0))
        if sinkage_load is not None:
            s_l = float(sinkage_load[name])
            c = clearance_metrics(prof, s_l, target, sw_time)
            pl = toe_down_pitch(M, q_worst, hard["foot"], s_l)
            rec["load_based"] = dict(stance_sinkage_mm=1e3 * s_l, mid_swing_min_above_pile_top_mm=1e3 * c["mid_swing_min_m"],
                                     clears_pile_top=c["mid_swing_min_m"] > 0.0,
                                     meets_carpet_target=c["mid_swing_min_m"] >= target,
                                     swing_share_meeting_target=c["swing_share_meeting_target"], **_exposure(c),
                                     lift_frac_needed_for_target=need(s_l), toe_down_pitch_to_reach_pile_top_deg=pl["deg"],
                                     toe_down_first_geom=pl["first_geom"])
        out["piles"][name] = rec
    return out


# ------------------------------------------------------------------------------------------------
# 3. Friction
# ------------------------------------------------------------------------------------------------
def derivatives(M: statics.Model, traj: dict):
    """Velocities and accelerations of a periodic reference walk by central differences, with root x unwrapped by
    exactly one stride across the period boundary (``gait.derivatives`` estimates that jump as
    ``x_{n-1} + x_1 - x_0``, an error on frames 0, 1, n-2 and n-1)."""
    m = M.m
    qs, dt, stride = np.asarray(traj["q"]), float(traj["dt"]), float(traj["info"]["stride"])
    n = len(qs)
    qd = np.zeros((n, m.nv))
    v = np.zeros(m.nv)
    for k in range(n):
        a, b = qs[(k - 1) % n].copy(), qs[(k + 1) % n].copy()
        if k == 0:
            a[0] -= stride
        if k == n - 1:
            b[0] += stride
        mujoco.mj_differentiatePos(m, v, 2 * dt, a, b)
        qd[k] = v
    qdd = (np.roll(qd, -1, axis=0) - np.roll(qd, 1, axis=0)) / (2 * dt)
    return qd, qdd


def band_limited_requirement(M: statics.Model, traj: dict, harmonics: int = BAND_HARMONICS) -> np.ndarray:
    """Least friction any force split needs at each frame, ``|a_xy| / (g + a_z)`` of the CoM, from the CoM path
    low-passed to its first ``harmonics`` stride harmonics (the total contact force is ``m (a_com - g)``)."""
    info = traj["info"]
    T, n = float(info["period"]), len(traj["q"])
    com = np.zeros((n, 3))
    for k, q in enumerate(traj["q"]):
        M.forward(q)
        com[k] = M.com()
    com[:, 0] -= info["stride"] / T * np.asarray(traj["t"])
    w = 2.0 * np.pi * np.fft.rfftfreq(n, d=T / n)
    C = np.fft.rfft(com, axis=0)
    C[harmonics + 1 :] = 0.0
    acc = np.fft.irfft(-(w**2)[:, None] * C, n, axis=0)
    return np.hypot(acc[:, 0], acc[:, 1]) / (G + acc[:, 2])


def friction_screen(M: statics.Model, traj: dict, idr: dict) -> dict:
    """Required friction of a reference walk (module docstring, item 3).  ``idr`` is ``gait.inverse_dynamics``'s
    result; the frames whose derivatives the exact unwrap changes, and every double-support frame, are recomputed
    with ``statics.contact_torques`` and the exact derivatives."""
    qd, qdd = derivatives(M, traj)
    fr_id = np.asarray(idr["friction"], float)
    normal = np.asarray(idr["normal"], float).copy()
    n = len(fr_id)
    single = np.array([len(p) == 1 for p in traj["phases"]])
    changed = (np.abs(qd - idr["qd"]).max(axis=1) > 1e-12) | (np.abs(qdd - idr["qdd"]).max(axis=1) > 1e-9)
    per_foot, total, axis = fr_id.copy(), fr_id.copy(), np.zeros(n)
    for k in np.where(~single | changed)[0]:
        sup = traj["phases"][k] or ("left", "right")
        _, w = statics.contact_torques(M, traj["q"][k], qd[k], qdd[k], support=sup)
        f = {s: w[s][:3] for s in sup}
        per_foot[k] = max(float(np.linalg.norm(f[s][:2]) / max(f[s][2], 1e-6)) for s in sup)
        tot = sum(f.values())
        total[k] = float(np.linalg.norm(tot[:2]) / max(tot[2], 1e-6))
        axis[k] = max(float(max(abs(f[s][0]), abs(f[s][1])) / max(f[s][2], 1e-6)) for s in sup)
        for j, s in enumerate(statics.SIDES):
            normal[k, j] = f[s][2] if s in f else 0.0
    gov_series = np.where(single, per_foot, total)
    k = int(np.argmax(per_foot))
    refs = friction_references()
    req = float(per_foot.max())
    req_single = float(per_foot[single].max()) if single.any() else 0.0
    req_total = float(total[~single].max()) if (~single).any() else 0.0
    gov = max(req_single, req_total)
    bl = band_limited_requirement(M, traj)
    area = {side: pad_area(M.m, side) for side in statics.SIDES}
    p_peak = max(normal[:, j].max() / area[side] for j, side in enumerate(statics.SIDES))
    p_static = M.mass() * G / area["left"]
    return dict(required_max_gait_inverse_dynamics=float(fr_id.max()),
                at_frame_gait_inverse_dynamics=int(np.argmax(fr_id)),
                frames_recomputed_for_unwrap=[int(i) for i in np.where(changed)[0]],
                required_max=req, at_frame=k, at_phase=k / n, in_single_support=bool(single[k]),
                normal_force_at_max_N={side: float(normal[k, j]) for j, side in enumerate(statics.SIDES)},
                required_max_single=req_single, required_max_double_split=float(per_foot[~single].max()) if (~single).any() else 0.0,
                required_max_double_total=req_total, required_governing=gov,
                governing_at_frame=int(np.argmax(gov_series)), required_p95=float(np.percentile(gov_series, 95)),
                required_band_limited=float(bl.max()), band_limited_at_frame=int(np.argmax(bl)),
                band_limited_harmonics=BAND_HARMONICS,
                double_support_frames=int((~single).sum()),
                double_support_frames_at_qp_bound=int(np.sum((~single) & (axis >= QP_BOUND))),
                available=refs, margin={key: v - gov for key, v in refs.items()},
                margin_band_limited={key: v - float(bl.max()) for key, v in refs.items()},
                margin_as_reported={key: v - req for key, v in refs.items()},
                within_carpet_band=gov < refs["carpet_band_low"], within_hard_floor=gov < refs["hard_robot_foot_low"],
                as_reported_within_carpet_band=req < refs["carpet_band_low"],
                peak_foot_load_N=float(normal.max()),
                peak_mean_pad_pressure_kPa=1e-3 * p_peak, single_support_static_pad_pressure_kPa=1e-3 * p_static)


# ------------------------------------------------------------------------------------------------
# 1. Contact model
# ------------------------------------------------------------------------------------------------
def impedance(solimp, x: float) -> float:
    """MuJoCo's impedance d(r) at penetration depth x >= 0 (``getimpedance`` with the clamp to [1e-4, 0.9999])."""
    d0, dw, width, mid, p = (float(v) for v in solimp)
    if d0 == dw or width <= 1e-15:
        imp = 0.5 * (d0 + dw)
    else:
        u = abs(x) / width
        if u >= 1.0:
            y = 1.0
        elif u <= 0.0:
            y = 0.0
        elif p == 1.0:
            y = u
        elif u <= mid:
            y = u**p / mid ** (p - 1.0)
        else:
            y = 1.0 - (1.0 - u) ** p / (1.0 - mid) ** (p - 1.0)
        imp = d0 + y * (dw - d0)
    return float(np.clip(imp, 1e-4, 0.9999))


def contact_force(solref, solimp, A_hat: float, x: float, timestep: float = 0.002) -> float:
    """Static normal force (N) of one soft contact at depth x >= 0: ``k d x / R`` (positive, time-constant solref)."""
    tc, z = float(solref[0]), float(solref[1])
    if tc <= 0:
        raise ValueError("direct (negative) solref is not used by this screen")
    tc = max(tc, 2.0 * timestep)  # MuJoCo's refsafe lower bound
    dw = float(np.clip(solimp[1], 1e-4, 0.9999))
    imp = impedance(solimp, x)
    k = 1.0 / (dw**2 * tc**2 * z**2)
    return k * imp * x / max(1e-15, (1.0 - imp) / imp * A_hat)


def static_depth(solref, solimp, A_hat: float, force: float, timestep: float = 0.002) -> float:
    """Depth at which one contact carries ``force`` (bisection; the force-depth curve is monotone here)."""
    lo, hi = 0.0, 0.2
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if contact_force(solref, solimp, A_hat, mid, timestep) < force:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def tangent_stiffness(solref, solimp, A_hat: float, x: float, timestep: float = 0.002) -> float:
    """dF/dx of one contact at depth x (N/m), by central difference."""
    h = max(1e-7, 1e-4 * x)
    return (contact_force(solref, solimp, A_hat, x + h, timestep)
            - contact_force(solref, solimp, A_hat, max(x - h, 0.0), timestep)) / (x + h - max(x - h, 0.0))


def soft_floor(K_foot: float, n_contacts: int, A_hat: float, depth: float, friction, d0: float | None = None,
               zeta: float | None = None, timestep: float = 0.002) -> dict:
    """Contact parameters for a linear spring of stiffness K_foot (N/m) shared by ``n_contacts`` contacts of a body with
    contact inverse inertia ``A_hat`` (module docstring, item 1).  ``zeta`` is MuJoCo's constraint-space damping
    ratio, not a physical one."""
    sv = screening_values()
    d0 = sv["contact_impedance_d0"]["value"] if d0 is None else d0
    zeta = sv["contact_damping_ratio"]["value"] if zeta is None else zeta
    K_i = K_foot / n_contacts
    tc = 1.0 / (zeta * np.sqrt(K_i * (1.0 - d0) * A_hat))
    if tc < 2.0 * timestep:  # MuJoCo would raise the time constant and the floor would be softer than derived
        raise ValueError(f"time constant {tc:.4f} s is below 2 x timestep; the stiffness cannot be represented")
    return dict(solref=[float(tc), float(zeta)], solimp=[d0, d0, float(depth), 0.5, 2.0], friction=[float(f) for f in friction],
                condim=6, K_foot_N_per_m=float(K_foot), K_contact_N_per_m=float(K_i), n_contacts_per_foot=int(n_contacts),
                A_hat_per_kg=float(A_hat), damping_per_foot_N_s_per_m=float(n_contacts * 2.0 / (tc * (1.0 - d0) * A_hat)))


def pair_params(m, g1: int, g2: int) -> dict:
    """Contact parameters MuJoCo uses for a geom pair: the higher priority's, else solmix-weighted solref and solimp
    (element-wise minimum for direct solref), element-wise maximum friction and condim."""
    p1, p2 = int(m.geom_priority[g1]), int(m.geom_priority[g2])
    if p1 != p2:
        w = g1 if p1 > p2 else g2
        return dict(solref=m.geom_solref[w].copy(), solimp=m.geom_solimp[w].copy(), friction=m.geom_friction[w].copy(),
                    condim=int(m.geom_condim[w]))
    s1, s2 = float(m.geom_solmix[g1]), float(m.geom_solmix[g2])
    mix = 0.5 if s1 + s2 < 1e-15 else s1 / (s1 + s2)
    r1, r2 = m.geom_solref[g1], m.geom_solref[g2]
    solref = mix * r1 + (1 - mix) * r2 if (r1[0] > 0 and r2[0] > 0) else np.minimum(r1, r2)
    return dict(solref=solref, solimp=mix * m.geom_solimp[g1] + (1 - mix) * m.geom_solimp[g2],
                friction=np.maximum(m.geom_friction[g1], m.geom_friction[g2]),
                condim=int(max(m.geom_condim[g1], m.geom_condim[g2])))


def pile_floor(m, k_A: float, depth: float, height: float, grid: float = PROBE_GRID_M) -> dict:
    """A linear pile of stiffness k_A (N/m^3) for the feet of model m (item 1).  Each probe gets
    ``soft_floor(k_A x its area)``; the record shows a full-cell probe."""
    sv = screening_values()
    fric = [sv["carpet_friction_band"]["value"][0], *sv["torsional_rolling_m"]["value"]]
    A_hat = float(m.body_invweight0[statics.bid(m, "left_foot"), 0])
    K = k_A * grid**2
    p = soft_floor(K, 1, A_hat, depth, fric, timestep=float(m.opt.timestep))
    d0 = float(p["solimp"][0])
    return dict(kind="pile", k_A_N_per_m3=float(k_A), stiffness_kPa_per_mm=1e-6 * float(k_A), depth_m=float(depth),
                height_m=float(height), grid_m=float(grid), solimp=p["solimp"], friction=p["friction"], condim=3,
                dampratio=float(p["solref"][1]), A_hat_per_kg=A_hat, timestep=float(m.opt.timestep),
                full_cell_probe=dict(K_N_per_m=float(K), solref=p["solref"],
                                     damping_N_s_per_m=float(2.0 / (p["solref"][0] * (1.0 - d0) * A_hat))),
                pad_stiffness_per_foot_N_per_m=float(k_A * pad_area(m, "left")),
                regularisation_inertia_kg_per_contact=float(d0 / ((1.0 - d0) * A_hat)))


def floor_variants(m, piles: dict | None = None) -> dict:
    """name -> floor parameters (None: the floor as compiled; kind "pile": a probe pile, ``pile_model``)."""
    piles = piles or pile_classes()
    rf = research()["robot_floor_definition"]["carpet"]
    variants = {"hard": None, "robot_floor_carpet": dict(solref=list(rf["solref"]), solimp=list(rf["solimp"]),
                                                         friction=list(rf["friction"]), condim=6)}
    for name, pc in piles.items():
        variants[name] = pile_floor(m, pc["stiffness_pa_per_m"], pc["depth_m"], pc["height_m"])
    return variants


def apply_floor(m, floor: dict) -> None:
    """Give the floor geom the parameters of ``floor`` and a priority above every other geom (robot_floor style)."""
    fl = statics.gid(m, "floor")
    m.geom_priority[fl] = int(max((m.geom_priority[g] for g in range(m.ngeom) if g != fl), default=0)) + 1
    m.geom_condim[fl] = int(floor["condim"])
    m.geom_solref[fl] = floor["solref"]
    m.geom_solimp[fl] = floor["solimp"]
    m.geom_friction[fl] = floor["friction"]


def add_probes(spec, body: str, centres_local: np.ndarray, stiffness: np.ndarray, floor: dict, priority: int) -> None:
    """Weld probe spheres (radius ``PROBE_RADIUS_M``) to ``body`` at ``centres_local`` (body frame), each a linear
    spring of the given stiffness (N/m) through its own solref (``soft_floor``); they collide with the floor only
    (contype ``PROBE_BIT``) and add no mass."""
    b = spec.body(body)
    for p, K in zip(np.asarray(centres_local, float), np.asarray(stiffness, float), strict=True):
        prm = soft_floor(K, 1, floor["A_hat_per_kg"], floor["depth_m"], floor["friction"], d0=floor["solimp"][0],
                         zeta=floor["dampratio"], timestep=floor["timestep"])
        g = b.add_geom()
        g.type = _GT.mjGEOM_SPHERE
        g.size = [PROBE_RADIUS_M, 0.0, 0.0]
        g.pos = p
        g.density = 0.0
        g.contype, g.conaffinity = PROBE_BIT, 0
        g.priority = priority
        g.solref, g.solimp, g.friction = prm["solref"], floor["solimp"], floor["friction"]
        g.condim = int(floor["condim"])
        g.group = 5
        g.rgba = [0.85, 0.3, 0.2, 0.6]


def pile_model(xml, q_home, floor: dict, geometry: str = "whole_foot"):
    """A copy of the compiled model on the linear pile ``floor`` (``pile_floor``): probes under the feet (whole foot
    or pads only), the pad boxes off the floor.  Returns (model, info); ``info["fine"]`` is the 1 mm underside map
    of the same geometry for the rigid statics."""
    M0 = statics.Model.load(str(xml))
    fine = foot_underside(M0, q_home, geometry, MAP_GRID_M)
    under = foot_underside(M0, q_home, geometry, floor["grid_m"])
    spec = mujoco.MjSpec.from_file(str(xml))
    prio = 1 + int(M0.m.geom_priority.max())
    cells = {}
    for side in statics.SIDES:
        c = under[side]
        keep = c["z"] <= floor["height_m"]  # the pile cannot compress beyond its height
        cells[side] = {key: c[key][keep] for key in ("x", "y", "z", "area")}
        cells[side]["cell_m"] = c["cell_m"]
        b = M0.foot[side]
        xb, Rb = M0.d.xpos[b].copy(), M0.d.xmat[b].reshape(3, 3).copy()
        centres = np.column_stack([cells[side]["x"], cells[side]["y"],
                                   cells[side]["z"] + under["z_sole"] + PROBE_RADIUS_M])  # lowest point on the element
        add_probes(spec, f"{side}_foot", (centres - xb) @ Rb, floor["k_A_N_per_m3"] * cells[side]["area"], floor,
                   prio + 1)
    m = spec.compile()
    fl = statics.gid(m, "floor")
    m.geom_priority[fl] = prio
    m.geom_conaffinity[fl] |= PROBE_BIT
    pads = pad_geoms(m)
    for side in statics.SIDES:
        for g in pads[side]:
            m.geom_contype[g] = 0
            m.geom_conaffinity[g] = 0
    return m, dict(pads=pads, cells=cells, geometry=geometry, z_sole=under["z_sole"], fine=fine,
                   probes_per_foot={side: int(len(cells[side]["z"])) for side in statics.SIDES})


# ------------------------------------------------------------------------------------------------
# 4. Standing check
# ------------------------------------------------------------------------------------------------
def hold_servos(m, d, q_home, design) -> dict:
    """Stiff PD position servos at q_home (module docstring, item 4); returns the gains per joint."""
    d.qpos[:] = q_home
    d.qvel[:] = 0
    mujoco.mj_forward(m, d)
    Mfull = np.zeros((m.nv, m.nv))
    mujoco.mj_fullM(m, d, Mfull)
    gains = {}
    for a in range(m.nu):
        j = int(m.actuator_trnid[a, 0])
        jn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
        kp = design.act(mjcf.joint_role(jn)).sustained / SERVO_ERROR_RAD
        inertia = float(Mfull[m.jnt_dofadr[j], m.jnt_dofadr[j]])
        kv = 2.0 * SERVO_ZETA * np.sqrt(kp * inertia)
        m.actuator_gainprm[a, 0] = kp
        m.actuator_biasprm[a, 1] = -kp
        m.actuator_biasprm[a, 2] = -kv
        d.ctrl[a] = q_home[m.jnt_qposadr[j]]
        gains[jn] = dict(kp=float(kp), kv=float(kv), joint_inertia=inertia, force_limit=float(m.actuator_forcerange[a, 1]))
    return gains


def _spans(tilt, com) -> tuple[float, float]:
    """Peak-to-peak trunk tilt (deg) and CoM movement (mm, diagonal of its bounding box) over a window of samples."""
    c = np.asarray(com)
    return float(np.ptp(tilt)), 1e3 * float(np.linalg.norm(np.ptp(c, axis=0)))


def _settled(tilt: list, com: list, lag: int) -> bool:
    """Settled: over the last second the tilt spans less than ``SETTLED_TILT_DEG`` and the CoM less than
    ``SETTLED_COM_MM``, and neither moved more than in the second before (a creep that is speeding up is a slow fall),
    unless both are below a tenth of their limits."""
    t1, c1 = _spans(tilt[-1 - lag :], com[-1 - lag :])
    t0, c0 = _spans(tilt[-1 - 2 * lag : -lag], com[-1 - 2 * lag : -lag])
    if t1 >= SETTLED_TILT_DEG or c1 >= SETTLED_COM_MM:
        return False
    return (t1 <= t0 and c1 <= c0) or (t1 < 0.1 * SETTLED_TILT_DEG and c1 < 0.1 * SETTLED_COM_MM)


def _foot_contacts(m, d, side_of: np.ndarray):
    """Per contact: foot index (0 left, 1 right, -1 neither), depth and normal force."""
    n = d.ncon
    if n == 0:
        return np.zeros(0, int), np.zeros(0), np.zeros(0)
    geom = d.contact.geom[:n]
    side = np.maximum(side_of[geom[:, 0]], side_of[geom[:, 1]])
    depth = -np.asarray(d.contact.dist[:n], float)
    if m.opt.cone == mujoco.mjtCone.mjCONE_ELLIPTIC:
        fn = np.asarray(d.efc_force[d.contact.efc_address[:n]], float)
    else:
        f6 = np.zeros(6)
        fn = np.zeros(n)
        for i in range(n):
            mujoco.mj_contactForce(m, d, i, f6)
            fn[i] = f6[0]
    return side, depth, fn


def standing_check(xml, q_home, design, floor: dict | None, H: float, seconds: float = STAND_SECONDS,
                   max_seconds: float = STAND_MAX_SECONDS, geometry: str = "whole_foot") -> dict:
    """Servo-held stance on one floor (module docstring, items 4 and 5)."""
    pile = isinstance(floor, dict) and floor.get("kind") == "pile"
    if pile:
        m, info = pile_model(xml, q_home, floor, geometry)
        pads = info["pads"]
    else:
        m = mujoco.MjModel.from_xml_path(str(xml))
        if floor is not None:
            apply_floor(m, floor)
        info, pads = None, pad_geoms(m)
    d = mujoco.MjData(m)
    gains = hold_servos(m, d, q_home, design)
    feet = {side: statics.bid(m, f"{side}_foot") for side in statics.SIDES}
    root = int(m.jnt_bodyid[0])
    W = float(m.body_subtreemass[root]) * G
    A_hat = float(m.body_invweight0[feet["left"], 0])
    dt = float(m.opt.timestep)
    com_home = d.subtree_com[root].copy()
    z_sole = min(lowest_point(m, d, g)[2] for side in statics.SIDES for g in pads[side])
    # the pads' area centroid on the sole plane, in each foot's frame (where foot sinkage is read)
    centre = {}
    for side in statics.SIDES:
        b, gs = feet[side], pads[side]
        a = np.array([4.0 * m.geom_size[g][0] * m.geom_size[g][1] for g in gs])
        cw = np.sum(a[:, None] * d.geom_xpos[gs], axis=0) / a.sum()
        cw[2] = z_sole
        centre[side] = d.xmat[b].reshape(3, 3).T @ (cw - d.xpos[b])
    if pile:

        def predict(F):
            return pile_sinkage(info["cells"]["left"], floor["k_A_N_per_m3"], F)

        contact = dict(model="probes", geometry=geometry, probes_per_foot=info["probes_per_foot"],
                       probe_grid_m=floor["grid_m"], full_cell_probe=floor["full_cell_probe"], solimp=floor["solimp"],
                       friction=floor["friction"], condim=floor["condim"], A_hat_per_kg=A_hat,
                       probe_area_per_foot_cm2=1e4 * float(info["cells"]["left"]["area"].sum()))
    else:
        pp = pair_params(m, statics.gid(m, "floor"), pads["left"][0])
        n_c = 4 * len(pads["left"])

        def predict(F):
            return static_depth(pp["solref"], pp["solimp"], A_hat, F / n_c, dt)

        contact = dict(model="pad corners", solref=[float(v) for v in pp["solref"]],
                       solimp=[float(v) for v in pp["solimp"]], friction=[float(v) for v in pp["friction"]],
                       condim=int(pp["condim"]), A_hat_per_kg=A_hat, contacts_per_foot=n_c)
    x0 = predict(0.5 * W)
    contact["predicted_standing_depth_mm"] = 1e3 * x0
    q = np.array(q_home, float).copy()
    q[2] -= x0
    d.qpos[:] = q
    d.qvel[:] = 0
    mujoco.mj_forward(m, d)
    trunk = statics.bid(m, "trunk")
    R0 = d.xmat[trunk].reshape(3, 3).copy()
    com0 = d.subtree_com[root].copy()
    side_of = np.full(m.ngeom, -1)
    for j, side in enumerate(statics.SIDES):
        side_of[m.geom_bodyid == feet[side]] = j
    every = max(1, int(round(LOG_DT / dt)))
    per_s = int(round(1.0 / dt))
    lag = per_s // every
    n_min, n_max = int(round(seconds / dt)), int(round(max(seconds, max_seconds) / dt))
    L = {key: [] for key in ("t", "tilt", "pitch", "roll", "com", "sink", "deep", "force", "ncon")}
    sat = np.zeros(m.nu, bool)
    status, fell_at = "unsettled", None
    k = 0
    while True:
        sat |= np.abs(d.actuator_force) >= m.actuator_forcerange[:, 1] - 1e-9
        if k % every == 0:
            Rw = d.xmat[trunk].reshape(3, 3) @ R0.T
            sd, dep, fn = _foot_contacts(m, d, side_of)
            L["t"].append(k * dt)
            L["tilt"].append(float(np.degrees(np.arccos(np.clip(Rw[2, 2], -1.0, 1.0)))))
            L["pitch"].append(float(np.degrees(np.arctan2(Rw[0, 2], Rw[2, 2]))))
            L["roll"].append(float(np.degrees(np.arctan2(-Rw[1, 2], Rw[2, 2]))))
            L["com"].append(d.subtree_com[root].copy())
            L["sink"].append([float(-(d.xpos[feet[s]] + d.xmat[feet[s]].reshape(3, 3) @ centre[s])[2])
                              for s in statics.SIDES])
            L["deep"].append([float(dep[sd == j].max()) if np.any(sd == j) else np.nan for j in range(2)])
            L["force"].append([float(fn[sd == j].sum()) for j in range(2)])
            L["ncon"].append([int(np.sum(sd == j)) for j in range(2)])
            if L["tilt"][-1] > UPRIGHT_TILT_DEG or com0[2] - L["com"][-1][2] > COLLAPSE_FRAC_H * H:
                status, fell_at = "tipped", k * dt
                break
            if k >= n_min and k % per_s == 0 and len(L["t"]) > 2 * lag:
                if _settled(L["tilt"], L["com"], lag):
                    status = "settled"
                    break
                if k >= n_max:
                    break
        mujoco.mj_step(m, d)
        k += 1
    t_log, tilt = np.array(L["t"]), np.array(L["tilt"])
    com, sink, deep = np.array(L["com"]), np.array(L["sink"]), np.array(L["deep"])
    force, ncon = np.array(L["force"]), np.array(L["ncon"])
    drop = float(com0[2] - com[-1, 2])
    fell = fell_at is not None
    upright = bool(status == "settled" and drop < COLLAPSE_FRAC_H * H)
    if not fell:
        win = t_log >= t_log[-1] - SETTLE_WINDOW_S
        window = "last 1 s"
    else:
        onset = np.where(tilt > ONSET_TILT_DEG)[0]
        t_on = t_log[onset[0]] if len(onset) else t_log[-1]
        win = (t_log >= 0.1) & (t_log <= t_on)
        if not win.any():
            win = t_log <= t_on
        window = f"0.1 s to {t_on:.2f} s (before the tilt reached {ONSET_TILT_DEG:g} deg)"
    feet_out = {}
    for j, side in enumerate(statics.SIDES):
        F = float(np.mean(force[win, j]))
        dj = deep[win, j][np.isfinite(deep[win, j])]
        feet_out[side] = dict(sinkage_mm=1e3 * float(np.mean(sink[win, j])),
                              deepest_contact_mm=1e3 * float(dj.mean()) if len(dj) else float("nan"),
                              predicted_mm=1e3 * predict(F) if F > 0 else float("nan"), normal_force_N=F,
                              contacts=int(round(float(np.mean(ncon[win, j])))))
    # rigid-robot statics on the same floor (item 5)
    if pile:
        st = stance_statics(pile_springs(info["fine"], floor["k_A_N_per_m3"]), W, com_home)
    else:
        md = mujoco.MjData(m)
        md.qpos[:] = q_home
        mujoco.mj_kinematics(m, md)
        pts = _pad_corners(m, md, pads)
        k_t = tangent_stiffness(pp["solref"], pp["solimp"], A_hat, x0, dt)
        contact["tangent_stiffness_per_foot_N_per_m"] = n_c * k_t
        st = stance_statics(dict(x=pts[:, 0], y=pts[:, 1], z=pts[:, 2] - z_sole, k=np.full(len(pts), k_t), z_sole=z_sole),
                            W, com_home)
    span_tilt, span_com = _spans(tilt[-1 - min(lag, len(t_log) - 1) :], com[-1 - min(lag, len(t_log) - 1) :])
    sub = slice(None, None, max(1, int(round(0.1 / LOG_DT))))  # 0.1 s samples for the record
    return dict(
        floor="as compiled" if floor is None else {k2: v for k2, v in floor.items()},
        contact=contact, upright=upright, status=status, seconds=float(t_log[-1]),
        fell_at_s=fell_at, tilt_max_deg=float(tilt.max()), tilt_end_deg=float(tilt[-1]),
        pitch_end_deg=L["pitch"][-1], roll_end_deg=L["roll"][-1],
        tilt_change_last_s_deg=span_tilt, com_move_last_s_mm=span_com,
        sinkage_window=window, feet=feet_out,
        com_drift_mm=1e3 * float(np.linalg.norm(com[-1, :2] - com0[:2])), com_drop_mm=1e3 * drop,
        servo_saturated=[mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a) for a in np.where(sat)[0]],
        statics=st, servo_gains=gains,
        trace=dict(t_s=np.round(t_log[sub], 3).tolist(), tilt_deg=np.round(tilt[sub], 3).tolist(),
                   pitch_deg=np.round(np.array(L["pitch"])[sub], 3).tolist(),
                   sinkage_mm=np.round(1e3 * np.nanmean(sink[sub], axis=1), 3).tolist()))


def critical_pile_sim(xml, q_home, design, H: float, m, k_start: float, steps: int = 4,
                      seconds: float = STAND_SECONDS, max_seconds: float = STAND_MAX_SECONDS) -> dict:
    """Softest linear pile on which the servo-held stance (whole foot) settles upright in the simulation, by
    bisection on log k_A from ``k_start`` (normally the rigid critical value).  Returns the bracket and every run."""
    dp = pressure_span_pa()
    height = max(pc["height_m"] for pc in pile_classes().values())
    runs = []

    def holds(kA):
        r = standing_check(xml, q_home, design, pile_floor(m, kA, dp / kA, height), H, seconds, max_seconds)
        runs.append(dict(k_A_kPa_per_mm=1e-6 * kA, compression_98kPa_mm=1e3 * dp / kA, upright=r["upright"],
                         status=r["status"], seconds=r["seconds"], tilt_end_deg=r["tilt_end_deg"],
                         rigid_linear_margin_pitch=r["statics"]["linear_margin"]["pitch"]))
        return r["upright"]

    lo = hi = float(k_start)
    if holds(hi):
        lo = hi / 1.5
        for _ in range(3):
            if not holds(lo):
                break
            hi, lo = lo, lo / 1.5
        else:
            return dict(found=False, note="still upright at the softest pile tried", runs=runs)
    else:
        hi = lo * 1.5
        for _ in range(6):
            if holds(hi):
                break
            lo, hi = hi, hi * 1.5
        else:
            return dict(found=False, note="not upright at the stiffest pile tried", runs=runs)
    for _ in range(steps):
        mid = float(np.sqrt(lo * hi))
        if holds(mid):
            hi = mid
        else:
            lo = mid
    return dict(found=True, k_A_kPa_per_mm=1e-6 * hi, compression_98kPa_mm=1e3 * dp / hi,
                bracket_compression_98kPa_mm=[1e3 * dp / hi, 1e3 * dp / lo],
                pad_stiffness_per_foot_kN_per_m=1e-3 * hi * pad_area(m, "left"), runs=runs)


# ------------------------------------------------------------------------------------------------
# Screen, record, plot, CLI
# ------------------------------------------------------------------------------------------------
def screen(M: statics.Model, q_home, design, xml, trajs: dict | None = None, seconds: float = STAND_SECONDS,
           max_seconds: float = STAND_MAX_SECONDS, critical_sim: bool = False) -> dict:
    t0 = time.perf_counter()
    H = design.sk.H
    piles = pile_classes()
    if trajs is None:
        trajs = {name: gait.generate(M, q_home, H, spec) for name, spec in (("nominal", sizing.NOMINAL),
                                                                              ("brisk", sizing.BRISK))}
    unders = {geo: foot_underside(M, q_home, geo) for geo in ("whole_foot", "pads")}
    fric, toe = {}, {}
    for name, tr in trajs.items():
        fric[name] = friction_screen(M, tr, gait.inverse_dynamics(M, tr))
        F = fric[name]["peak_foot_load_N"]
        sink = {p: pile_sinkage(unders["whole_foot"]["left"], pc["stiffness_pa_per_m"], F) for p, pc in piles.items()}
        toe[name] = toe_catch(M, tr, H, piles, sink)
    t1 = time.perf_counter()
    floors = floor_variants(M.m, piles)
    stand = {name: standing_check(xml, q_home, design, fl, H, seconds, max_seconds) for name, fl in floors.items()}
    t2 = time.perf_counter()
    tip = passive_tip(M, q_home, unders)
    if critical_sim and tip["whole_foot"]["critical_k_A_kPa_per_mm"]:
        tip["simulated_whole_foot"] = critical_pile_sim(xml, q_home, design, H, M.m,
                                                        1e6 * tip["whole_foot"]["critical_k_A_kPa_per_mm"],
                                                        seconds=seconds, max_seconds=max_seconds)
    doc = dict(module="designlab.carpet", data_file=DATA.name, H_m=H, mass_kg=M.mass(),
               pile_classes={k: dict(pile_height_mm=1e3 * v["height_m"], compression_fraction=v["compression_fraction"],
                                     compressed_depth_mm=1e3 * v["depth_m"],
                                     secant_stiffness_kPa_per_mm=1e-6 * v["stiffness_pa_per_m"]) for k, v in piles.items()},
               targets=dict(carpet_mm=1e3 * carpet_target(H), hard_mm=1e3 * screening_values()["swing_targets"]["hard_frac_H"] * H),
               floors={k: ("as compiled" if v is None else v) for k, v in floors.items()},
               toe_catch=toe, friction=fric, standing=stand, passive_tip=tip,
               provenance=dict(
                   reference_trajectory="toe_catch and friction: prescribed reference walks (gait.generate, sizing.NOMINAL "
                                        "and sizing.BRISK), kinematics and inverse dynamics only",
                   servo_held="standing: every joint held at q_home by stiff position servos, no balance control",
                   static_only="the pile floors match the sourced static stiffness only; damping is assumed and MuJoCo's "
                               "contact regularisation adds inertia, so times to tip (fell_at_s) are not results and the "
                               "floors are not suited to touchdown or dynamic studies or to controller training",
                   needs_controller="walking or standing on carpet, foot pitch during swing, slip, toe-drag forces, "
                                    "touchdown dynamics and any stance that tips in the standing check"),
               timing_s=dict(walks_and_friction=round(t1 - t0, 2), standing=round(t2 - t1, 2),
                             passive_tip=round(time.perf_counter() - t2, 2)))
    doc["summary"] = summarize(doc)
    return doc


def summarize(doc: dict) -> dict:
    toe, fric, stand, tip = doc["toe_catch"], doc["friction"], doc["standing"], doc["passive_tip"]
    swing = {}
    for g, r in toe.items():
        P = r["piles"]
        swing[g] = dict(hard_mid_swing_min_mm=r["hard_floor"]["mid_swing_min_mm"],
                        hard_mid_swing_min_frac_H=r["hard_floor"]["mid_swing_min_frac_H"],
                        hard_meets_0p04H=r["hard_floor"]["meets_hard_target"],
                        above_pile_top_mm={p: v["mid_swing_min_above_pile_top_mm"] for p, v in P.items()},
                        phase={p: v["phase"] for p, v in P.items()},
                        meets_carpet_target={p: v["meets_carpet_target"] for p, v in P.items()},
                        clears_pile_top={p: v["clears_pile_top"] for p, v in P.items()},
                        travel_in_pile_before_touchdown_mm={p: v["travel_in_pile_before_touchdown_mm"] for p, v in P.items()},
                        reentry_speed_mm_s={p: v["forward_speed_at_entry_mm_s"] for p, v in P.items()},
                        lift_frac_needed={p: v["lift_frac_needed_for_target"] for p, v in P.items()},
                        toe_down_pitch_deg={p: v["toe_down_pitch_to_reach_pile_top_deg"] for p, v in P.items()},
                        toe_down_first_geom={p: v["toe_down_first_geom"] for p, v in P.items()})
        if all("load_based" in v for v in P.values()):
            swing[g]["load_based"] = dict(
                stance_sinkage_mm={p: v["load_based"]["stance_sinkage_mm"] for p, v in P.items()},
                above_pile_top_mm={p: v["load_based"]["mid_swing_min_above_pile_top_mm"] for p, v in P.items()},
                meets_carpet_target={p: v["load_based"]["meets_carpet_target"] for p, v in P.items()},
                lift_frac_needed={p: v["load_based"]["lift_frac_needed_for_target"] for p, v in P.items()},
                toe_down_pitch_deg={p: v["load_based"]["toe_down_pitch_to_reach_pile_top_deg"] for p, v in P.items()})
    friction = {g: dict(required_governing=r["required_governing"], required_band_limited=r["required_band_limited"],
                        required_p95=r["required_p95"], required_max_single=r["required_max_single"],
                        required_max_double_total=r["required_max_double_total"],
                        required_max_as_reported=r["required_max"],
                        required_max_gait_inverse_dynamics=r["required_max_gait_inverse_dynamics"],
                        margin_carpet_band_low=r["margin"]["carpet_band_low"],
                        margin_hard_robot_foot_low=r["margin"]["hard_robot_foot_low"],
                        double_support_frames_at_qp_bound=r["double_support_frames_at_qp_bound"],
                        peak_foot_load_N=r["peak_foot_load_N"], peak_mean_pad_pressure_kPa=r["peak_mean_pad_pressure_kPa"])
                for g, r in fric.items()}
    standing = {f: dict(upright=r["upright"], status=r["status"], seconds=r["seconds"],
                        sinkage_mm=[r["feet"][s]["sinkage_mm"] for s in statics.SIDES],
                        predicted_mm=[r["feet"][s]["predicted_mm"] for s in statics.SIDES],
                        tilt_end_deg=r["tilt_end_deg"], tilt_change_last_s_deg=r["tilt_change_last_s_deg"],
                        com_drift_mm=r["com_drift_mm"], rigid_holds=r["statics"]["holds"],
                        rigid_lean_pitch_deg=r["statics"]["lean_deg"]["pitch"],
                        linear_margin_pitch=r["statics"]["linear_margin"]["pitch"],
                        linear_margin_roll=r["statics"]["linear_margin"]["roll"])
                for f, r in stand.items()}
    passive = {geo: dict(critical_compression_98kPa_mm=tip[geo]["critical_compression_98kPa_mm"],
                         critical_k_A_kPa_per_mm=tip[geo]["critical_k_A_kPa_per_mm"],
                         critical_pad_stiffness_per_foot_kN_per_m=tip[geo]["critical_pad_stiffness_per_foot_kN_per_m"])
               for geo in ("whole_foot", "pads")}
    passive["pad_area_centroid_ahead_of_com_mm"] = tip["pad_area_centroid_ahead_of_com_mm"]
    if "simulated_whole_foot" in tip:
        sim = tip["simulated_whole_foot"]
        passive["simulated_whole_foot"] = {k: sim.get(k) for k in ("found", "compression_98kPa_mm",
                                                                    "bracket_compression_98kPa_mm", "k_A_kPa_per_mm")}
    checks = dict(
        carpet_swing_target_met={p: all(s["meets_carpet_target"][p] for s in swing.values()) for p in PILES},
        pile_top_cleared_mid_swing={p: all(s["clears_pile_top"][p] for s in swing.values()) for p in PILES},
        hard_swing_target_met=all(s["hard_meets_0p04H"] for s in swing.values()),
        friction_within_carpet_band=all(r["within_carpet_band"] for r in fric.values()),
        friction_within_hard_floor_band=all(r["within_hard_floor"] for r in fric.values()),
        friction_as_reported_within_carpet_band=all(r["as_reported_within_carpet_band"] for r in fric.values()),
        servo_held_upright={f: r["upright"] for f, r in stand.items()})
    passed = bool(checks["friction_within_carpet_band"] and all(checks["pile_top_cleared_mid_swing"].values()))
    return dict(
        H_m=doc["H_m"], mass_kg=doc["mass_kg"], carpet_target_mm=doc["targets"]["carpet_mm"],
        # the keys shortlist.gates and shortlist.scores read
        **{"pass": passed},
        toe_clearance_margin_mm=min(v for s in swing.values() for v in s["above_pile_top_mm"].values()),
        friction_margin=min(f["margin_carpet_band_low"] for f in friction.values()),
        gate=("pass: the governing friction need is below the carpet band's lower edge on both walks and the swing foot "
              "clears every pile top in mid-swing with the stance sole at the 98 kPa depth.  toe_clearance_margin_mm: "
              "the least of those clearances (mm above the pile top); friction_margin: band lower edge minus the "
              "governing need, least over the walks.  The 0.06 H carpet target depends on the reference lift, which "
              "every candidate shares, and is reported in checks, not gated; the standing check is a contact-model "
              "sanity check and is not gated."),
        swing=swing, friction=friction, standing=standing, passive_tip=passive, checks=checks)


def _clean(x):
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return _clean(x.tolist())
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return float(x) if np.isfinite(x) else None
    return x


def write(doc: dict, out_dir: Path, cid: str, png: bool = True) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{cid}_carpet.json"
    path.write_text(json.dumps(_clean(doc), indent=1))
    if png:
        from .round1 import compact_png

        plot(doc, out_dir / f"{cid}_carpet.png", cid)
        compact_png(out_dir / f"{cid}_carpet.png")
    return path


def screen_record(cid: str, rec: dict, xml, q_home, design, trajs: dict, scr, seconds: float = STAND_SECONDS,
                  max_seconds: float = STAND_MAX_SECONDS) -> dict:
    """Round-two hook (``round2.screen_one``): screen a sized, rebuilt candidate and write <scr>/<id>_carpet.*.
    The summary it returns carries ``pass``, ``toe_clearance_margin_mm`` and ``friction_margin`` for the shortlist."""
    M = statics.Model.load(str(xml))
    doc = screen(M, q_home, design, xml, trajs, seconds, max_seconds)
    doc.update(id=cid, xml_sha256=hashlib.sha256(Path(xml).read_bytes()).hexdigest(),
               sized_sha256=rec.get("xml_sha256"))
    write(doc, Path(scr), cid)
    return _clean(doc["summary"])


def _spread(ys: list, gap: float) -> list:
    """Label positions at least ``gap`` apart, in the input order of increasing y, moved as little as possible."""
    order = np.argsort(ys)
    out = np.array(ys, float)[order]
    for _ in range(50):
        moved = False
        for i in range(1, len(out)):
            if out[i] - out[i - 1] < gap:
                push = 0.5 * (gap - (out[i] - out[i - 1]))
                out[i - 1] -= push
                out[i] += push
                moved = True
        if not moved:
            break
    res = np.empty(len(out))
    res[order] = out
    return res.tolist()


def plot(doc: dict, path, cid: str) -> None:
    """Lowest swing-foot point against swing phase for both walks, with each pile class's band and target."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    surf, ink, ink2, muted, gridc, axis_c, band = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#f0efec"
    pile_c = {"low": "#86b6ef", "medium": "#2a78d6", "high": "#104281"}  # one-hue ordinal ramp (blue 250/450/650)
    target = doc["targets"]["carpet_mm"]
    hard_t = doc["targets"]["hard_mm"]
    fig, axs = plt.subplots(1, 2, figsize=(8.8, 3.9), dpi=110, sharey=True)
    fig.patch.set_facecolor(surf)
    s_grid = np.linspace(0.0, 1.0, 401)
    y_hi, y_lo = 0.0, 0.0
    for ax, gname in zip(axs, ("nominal", "brisk"), strict=False):
        tc = doc["toe_catch"].get(gname)
        ax.set_facecolor(surf)
        if tc is None:
            continue
        env = np.min([np.interp(s_grid, p["s"], p["z_mm"]) for p in tc["profile"].values()], axis=0)
        ax.axvspan(MID_SWING[0], MID_SWING[1], color=band, zorder=0, lw=0)
        ax.axhline(hard_t, color=muted, lw=0.9, ls=(0, (1, 2)), zorder=1)
        for name in reversed(PILES):
            p = tc["piles"][name]
            top, backing = p["stance_sinkage_mm"], p["stance_sinkage_mm"] - p["pile_height_mm"]
            ax.axhspan(backing, top, color=pile_c[name], alpha=0.09, lw=0, zorder=1)
            ax.axhline(top, color=pile_c[name], lw=1.3, zorder=2)
            ax.axhline(top + target, color=pile_c[name], lw=1.3, ls=(0, (4, 3)), zorder=2)
            if p.get("load_based"):
                ax.axhline(p["load_based"]["stance_sinkage_mm"], color=pile_c[name], lw=1.0, ls=(0, (1, 1.5)), zorder=2)
            y_hi, y_lo = max(y_hi, top + target), min(y_lo, backing)
        ax.plot(s_grid, env, color=ink, lw=2.0, solid_capstyle="round", zorder=3)
        y_hi = max(y_hi, float(env.max()))
        ax.set_title(f"{gname.capitalize()} walk: lift {tc['lift_frac_H']:.2f} H, swing {tc['swing_time_s']:.2f} s",
                     loc="left", fontsize=8.8, color=ink)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Swing phase (0 lift-off, 1 touchdown)", fontsize=8, color=ink2)
        ax.grid(True, axis="y", color=gridc, lw=0.6)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(axis_c)
        ax.tick_params(colors=axis_c, labelcolor=ink2, labelsize=7.5)
    axs[0].set_ylabel("Height above the stance sole, mm", fontsize=8, color=ink2)
    lo, hi = y_lo - 1.0, 1.08 * y_hi + 2.0
    axs[0].set_ylim(lo, hi)
    handles = [Line2D([], [], color=ink, lw=2.0), Patch(color=pile_c["medium"], alpha=0.25, lw=0),
               Line2D([], [], color=pile_c["medium"], lw=1.3),
               Line2D([], [], color=pile_c["medium"], lw=1.0, ls=(0, (1, 1.5))),
               Line2D([], [], color=pile_c["medium"], lw=1.3, ls=(0, (4, 3))),
               Line2D([], [], color=muted, lw=0.9, ls=(0, (1, 2))), Patch(color=band, lw=0)]
    labels = ["lowest point of the swing foot (lower of the two feet)", "uncompressed pile along the swing path",
              "pile top: stance sole sunk to its 98 kPa depth (screened)",
              "pile top: whole-foot sinkage at the walk's peak load", f"pile top + max(0.06 H, 15 mm) = {target:.1f} mm",
              "hard floor: 0.04 H", "mid-swing window (screened)"]
    fig.legend(handles, labels, loc="lower left", ncol=3, frameon=False, fontsize=7.0, labelcolor=ink2,
               bbox_to_anchor=(0.01, 0.0))
    fig.suptitle(f"{cid}: swing-foot clearance against carpet pile (prescribed reference walks, not a controller)",
                 x=0.012, y=0.975, ha="left", fontsize=9.6, color=ink, fontweight="bold")
    fig.subplots_adjust(left=0.07, right=0.865, top=0.86, bottom=0.31, wspace=0.07)
    # direct labels at the right edge, in text ink, spread so they never overlap (after the layout is fixed)
    tc = doc["toe_catch"].get("brisk") or doc["toe_catch"]["nominal"]
    items = [(tc["piles"][n]["stance_sinkage_mm"], f"{n} pile top", ink2) for n in PILES]
    items += [(tc["piles"][n]["stance_sinkage_mm"] + target, f"{n} target", ink2) for n in PILES]
    items += [(hard_t, "hard-floor target", muted)]
    h_pt = axs[1].get_position().height * fig.get_figheight() * 72.0
    ys = _spread([y for y, _, _ in items], 9.0 * (hi - lo) / h_pt)
    for (y, text, col), yl in zip(items, ys, strict=True):
        axs[1].annotate(text, xy=(1.0, y), xycoords=("axes fraction", "data"), xytext=(1.01, yl),
                        textcoords=("axes fraction", "data"), ha="left", va="center", fontsize=7, color=col,
                        annotation_clip=False)
    fig.savefig(path, facecolor=surf)
    plt.close(fig)


def rebuild_round1(cid: str, stage_a: Path, models: Path):
    """Rebuild a sized round-one model exactly (round1.py recipe); returns (Stage A record, build result)."""
    from .params import parse
    from .pipeline import build_candidate
    from .round1 import _final_mass_guess

    rec = json.loads((Path(stage_a) / "candidates" / f"{cid}.json").read_text())
    sel = dict(rec["selection"])
    sel["mass_guess"] = _final_mass_guess(rec)
    return rec, build_candidate(parse(cid), Path(models), selection=sel, B=rec["B"], write=False)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Carpet screen (see the module docstring).")
    ap.add_argument("ids", nargs="+", help="round-one candidate ids, e.g. 60-M-T-10")
    ap.add_argument("--out", required=True, help="scratch output directory (never inside round1/)")
    ap.add_argument("--stage-a", default=str(STUDY / "round1" / "stage_a"), help="Stage A record directory")
    ap.add_argument("--seconds", type=float, default=STAND_SECONDS, help="least standing-check duration (s)")
    ap.add_argument("--max-seconds", type=float, default=STAND_MAX_SECONDS,
                    help="longest standing-check duration while the stance has not settled (s)")
    ap.add_argument("--critical", action="store_true",
                    help="also find the softest pile that holds by bisection in the servo-held simulation")
    ap.add_argument("--no-png", action="store_true")
    a = ap.parse_args(argv)
    import designlab  # noqa: F401  (GL backend before any rendering)

    out = Path(a.out).resolve()
    if out == (STUDY / "round1").resolve() or (STUDY / "round1").resolve() in out.parents:
        raise SystemExit("refusing to write inside round1/: pass a scratch --out directory")
    out.mkdir(parents=True, exist_ok=True)
    for cid in a.ids:
        t0 = time.perf_counter()
        rec, res = rebuild_round1(cid, Path(a.stage_a), out / "mjcf")
        xml = out / "mjcf" / f"{cid}.xml"
        sha = hashlib.sha256(xml.read_bytes()).hexdigest()
        doc = screen(res["model"], res["q_home"], res["design"], xml, seconds=a.seconds, max_seconds=a.max_seconds,
                     critical_sim=a.critical)
        doc.update(id=cid, xml_sha256=sha, stage_a_sha256=rec.get("xml_sha256"),
                   matches_stage_a=sha == rec.get("xml_sha256"))
        path = write(doc, out, cid, png=not a.no_png)
        s = doc["summary"]
        sw = s["swing"]["nominal"]
        crit = s["passive_tip"]["whole_foot"]["critical_compression_98kPa_mm"]
        print(f"{cid}: H {s['H_m']:.3f} m; nominal mid-swing {sw['hard_mid_swing_min_mm']:.1f} mm on hard floor, above "
              f"pile top " + ", ".join(f"{p} {v:.1f}" for p, v in sw["above_pile_top_mm"].items())
              + f" mm (target {s['carpet_target_mm']:.1f}); friction need "
              f"{max(f['required_governing'] for f in s['friction'].values()):.2f} (band-limited "
              f"{max(f['required_band_limited'] for f in s['friction'].values()):.2f}); servo-held upright "
              + ", ".join(f"{f}={'yes' if r['upright'] else r['status']}" for f, r in s["standing"].items())
              + (f"; rigid critical pile {crit:.1f} mm at 98 kPa" if crit else "")
              + f"; pass {s['pass']}; SHA match {doc['matches_stage_a']}; {time.perf_counter() - t0:.1f} s -> {path}",
              flush=True)


if __name__ == "__main__":
    main()
