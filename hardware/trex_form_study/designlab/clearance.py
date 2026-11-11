"""Clearance and interference screening of a compiled candidate (screening, not certification).

Reads the MJCF written by ``pipeline.build_candidate`` (with its ``home`` pose) and the reference
walks from ``gait.generate``; nothing is taken from the parameter tables, so what is reported is
what was built.  Metres in code; millimetres or fractions of the hip height ``H`` in the report.

Run from hardware/trex_form_study:
    python -m designlab.clearance 60-M-S-10 --out <dir>      # writes <dir>/<id>_clearance.json (+ .png)

Geometry screened
-----------------
* Hardware: ``hw_*`` primitives in group 0 (floor-contact foot pads) and group 2, except cosmetic
  details whose names contain ``tooth``, ``eye``, ``claw`` or ``hallux`` and zero-size geoms.
  Ledger items without MJCF geometry (bearings, harness, sensing allowances) are not screened.
* Shells: the ``cl_*`` convex collision proxies (group 3).  Each is the convex hull of a
  three-station strip of a panel's *mid-surface*: the wall thickness is not included (the outer
  skin lies t/2, about 0.8-1.5 mm, further out), concave regions are bridged and the hollow
  interior is filled.  A proxy overlap therefore means "the shells meet or nearly meet"; hardware
  sitting inside another body's hollow cover would also register.  The visual ``cover_*`` meshes
  and panels without proxies (the fixed arms) take no part in interference checks.

1. Body-pair interference
-------------------------
Pairs: every screened geom pair on two different rigid bodies that are not parent and child
(MuJoCo's own parent filter; bodies welded without a joint count as their parent).  Trunk-head and
trunk-tail seams are parent/child and are therefore not checked.  Pairs across a two-axis joint
built as two hinges at one point (hip pitch + thigh twist: roll body vs femur; ankle pitch + roll:
shank vs foot) are kept and tagged ``link = "2-dof-joint"``: their parts meet at the joint by
design, so a finding there is a joint-packaging conflict.
Distance: signed distance from ``mujoco.mj_geomDistance`` (negative = penetration depth along the
minimum-translation direction), evaluated only for pairs whose world AABBs come within
``REPORT_CAP`` = 25 mm (broad phase); anything farther is reported as "> 25 mm".
Classes: interference d < 0; tight 0 <= d < ``TIGHT`` = 3 mm.  Likely cause from the geom kinds:
'shell-shell', 'shell-hardware', 'hardware-hardware'.
Verification: in MuJoCo 3.10 the native GJK/EPA occasionally returns 0.0, or the negated
distance, for separated convex pairs (mostly parallel plates and proxy hulls; confirmed with an
exact polytope computation and an LP intersection test), and the analytic capsule-box routine
misreports deep penetration, even as a small positive distance.  Therefore: capsule-box pairs
always use an exact segment-box signed distance; every other result below 3 mm is re-evaluated
with the second geom shifted by 10 um in two to five directions and kept if the values agree
within 0.05 mm (none exactly 0), replaced by the median of an agreeing majority otherwise, and
failing that recomputed exactly (Minkowski difference hull; cylinders as circumscribed
128-gons, conservative by at most 0.03 % of the radius).  Counts are in ``verification``.
Values of 3 mm or more are MuJoCo's.  Checked against the exact computation on 33 669 pair
evaluations within 25 mm (home, 24 gait frames and 16 random in-range poses of each of 60-M-S-10,
60-M-H-20, 75-W-T-00 and 120-M-S-10): raw MuJoCo was off by more than 0.05 mm in 3.6-3.7 % of
them; after verification none was.
Poses ("families"):
  home              the q_home pose;
  gait_<name>       every frame of each gait (the stride is periodic, so the path is closed);
                    ``max_step_mm`` bounds how far any screened surface moves relative to the
                    trunk between consecutive evaluated poses, i.e. the sampling resolution
                    (``substeps`` adds interpolated poses between frames to refine it);
  joint_extremes    sweeps from home to each limit in steps of at most 0.05 rad, all other joints
                    at home, root fixed: every leg joint alone (min and max), per side the four
                    corners of hip_pitch x knee and of hip_roll x hip_yaw (straight line in joint
                    space), neck_pitch and tail_yaw (and Stage 2's expressive-head neck_yaw and jaw on a
                    model that has them).  The floor is ignored (feet may go below it).
                    Each sweep reports the first joint value at which a new pair (parts not
                    interfering at home) interferes, and the deepest new pairs;
  tail_limits_in_gait  tail_yaw at each limit at phases 0, 0.25, 0.5, 0.75 of each gait; only
                    pairs with a geom on the tail body (the rest is the gait family's).
Counts per family are unique geom pairs whose minimum over the family falls in a class;
``interference_pairs_not_at_home`` excludes pairs whose parts already interfere at home (proxy
pieces of one panel count as one part, so a home overlap that spreads to neighbouring pieces is
not counted as new; the same rule defines "new" in the joint sweeps).  Each family lists
its worst 40 pairs below 3 mm (25 for joint_extremes) and every body pair within 25 mm.
Same-body component overlaps (informational, outside the pass/fail): components (motors, battery,
boards, sensors) on one rigid body that intersect at home without one box enclosing the other
(e.g. a battery inside its cassette).  Structure, linkage and foot parts (names containing plate,
beam, bracket, spar, rod, toe, pad, ujoint) meet at joints and attachments in this model and are
only counted.  Parent/child hardware overlaps (e.g. trunk vs tail) are not screened anywhere
except by the battery path check.

2. Swing-foot clearance
-----------------------
Swing frames of a foot are those whose support tuple does not contain it.  Pad height = the lowest
point of that foot's floor-contact pad boxes (exact corners).  The reference lift starts and ends
on the floor, so the full-swing minimum is ~0 by construction (reported as ``full_swing_min``);
the screened value is the minimum over mid-swing, the frames between 25 % and 75 % of each swing's
duration (``MID_SWING``), when the foot is carried past the stance leg.  ``apex`` is the smallest
per-swing maximum.  Lateral gap = minimum signed distance between the swing foot's pads and every
screened geom of the stance leg (all six bodies of that side) over single-support frames, with the
lateral (y) component of the closest-point segment.
Pass: mid-swing clearance >= ``SWING_MIN_FRAC`` x H = 0.04 H in every gait.

3. Tail clearance
-----------------
Lowest point of every geom on the tail body (covers, proxies and hardware; meshes vertex-exact)
above the floor (z = 0) at home, over the tail_yaw range (sweep from home to both limits) and over
every gait frame; reported in mm and as a fraction of H.  Pass: >= ``TAIL_MIN_FRAC`` x H = 0.25 H.
The gait frames with tail_yaw forced to its limits are reported separately (not in the pass/fail).

4. Battery service path
-----------------------
The ``hw_battery_cassette`` box is translated (no rotation) along its own +z (dorsal hatch), -z
(ventral), +y and -y (flanks) for its full extent along that axis plus ``BATTERY_EXTRA`` = 20 mm,
in ``BATTERY_STEP`` = 2 mm steps, at the home pose.  Obstacles: every other screened hardware geom
on the trunk (``hw_battery`` and the cassette itself excluded) and every screened geom (hardware
and shell proxies) of the moving bodies.  Trunk shells are assumed to carry an opening or hatch on
the path; the cover panel each path crosses first (``mj_ray`` from the cassette face) is reported.
A direction is clear if no obstacle is overlapped (d < 0) anywhere on the travel; an obstacle that
already overlaps the cassette at home is flagged and also blocks.  Pass: at least one clear
direction.  Latches, connectors, cable slack and hand or tool access are not modelled.

Summary block (``summary``): compact counts per family; ``worst_pairs`` (top 10 over home, the
gaits and the tail-limit poses; proxy pieces collapsed to their panel, left/right mirror images
merged, ``sides`` naming the sides on which a same-named pair occurs);
``worst_pairs_joint_extremes`` (top 5); ``joint_range_usable`` (per joint and direction, the first
joint value at which a new pair interferes, with that pair; null = the full range adds no new
interference); hardware interference at home; swing, tail and battery results; ``pass``.
Pass/fail (``summary.pass``): no interference at home; no interference along the nominal gait;
swing clearance; tail clearance; battery path.  These are screening gates on the model as built.
Not represented: part compliance and deflection under load, backlash, manufacturing tolerances,
fasteners, cable routing, cover attachment features, dynamic overshoot beyond the reference
trajectories, and contact with the floor or the environment other than the swing and tail heights.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial import ConvexHull, QhullError

from . import STUDY_REVISION

TIGHT = 0.003  # m; 0 <= d < TIGHT is "tight", d < 0 is "interference"
REPORT_CAP = 0.025  # m; distances are evaluated up to this value, larger ones reported as "> cap"
LATERAL_CAP = 0.40  # m; distance cap for the swing-foot lateral gap
SWING_MIN_FRAC = 0.04  # mid-swing pad clearance threshold, fraction of H
TAIL_MIN_FRAC = 0.25  # tail lowest-point threshold, fraction of H
#: Round-two gate set (``gates="r2"``): also no overlapping components on one rigid body, no interference
#: in the brisk walk, and every swept joint usable at least ``RANGE_MARGIN_RAD`` beyond the walks' excursion.
RANGE_MARGIN_RAD = 0.10
#: Joints that rest on one end of their range at home by design, with that end ("min" or "max"): Stage 2's
#: expressive-head jaw rests closed on its lower stop (``build.JAW_RANGE``).  Their margin at that end is reported
#: (``joint_range_margins[joint]["rests_on_stop"]``) but is no walking range limit, so the ``joint_range_margin`` gate
#: leaves it out.  No earlier model has such a joint.
REST_ON_STOP = {"jaw": "min"}
#: The hinges whose own seam (the moving body against its parent) the body-pair screen leaves out (section 1:
#: parent/child pairs), so that its joint sweeps cannot see it and cannot cut the joint's range there: today's neck
#: pitch and tail yaw (the head and the tail on the trunk) and Stage 2's expressive-head neck yaw (the head on the
#: trunk) and jaw (the jaw on the head).  ``seam_sweep`` sweeps those seams.
SEAM_JOINTS = ("neck_pitch", "tail_yaw", "neck_yaw", "jaw")
SEAM_DEEPEN_M = 0.002  # a home overlap that deepens by more than this along a seam sweep is listed
SEAM_RULE = (
    "the body-pair screen pairs no body with its parent (section 1), so a joint sweep cannot see the joint's own seam "
    "(the moving body against its parent: the neck against the chest for the neck pitch and the neck yaw, the tail "
    "against the trunk for the tail yaw, the jaw against the skull and the neck for the jaw) and cannot cut its range "
    "there: joint_range_usable stays at the MJCF limit unless another body interferes.  seam_sweep sweeps each such "
    "joint alone from home to each limit (steps of at most SWEEP_STEP, the other joints at home, root fixed) over every "
    "screened geom of the moving body against every screened geom of its parent (the cl_* proxies and the hw_* "
    "primitives, the screen's own set and distance oracle) and reports the pairs apart at home that interfere along "
    "the sweep (new contacts, panel-level as the screen counts them) and the pairs overlapping at home that overlap "
    f"more deeply (by more than {1e3 * SEAM_DEEPEN_M:g} mm).  The proxies are coarse hulls of the covers' mid-surfaces "
    "and meet at a seam by design (the neck proxy reaches into the chest's), so a deepening overlap is where to look, "
    "not a measured collision")
MID_SWING = (0.25, 0.75)  # fraction of each swing's duration
SWEEP_STEP = 0.05  # rad, maximum joint increment along a sweep
TAIL_PHASES = (0.0, 0.25, 0.5, 0.75)
BATTERY_STEP = 0.002  # m
BATTERY_EXTRA = 0.020  # m beyond the cassette's own extent
BATTERY_DIRECTIONS = (("+z", 2, 1.0, "dorsal hatch"), ("-z", 2, -1.0, "ventral"), ("+y", 1, 1.0, "left flank"),
                      ("-y", 1, -1.0, "right flank"))
SKIP_TOKENS = ("tooth", "eye", "claw", "hallux")
STRUCTURE_TOKENS = ("plate", "beam", "bracket", "spar", "rod", "toe", "pad", "ujoint")
SIDES = ("left", "right")
LEG_JOINTS = ("hip_roll", "hip_pitch", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")
CORNERS = (("hip_pitch", "knee"), ("hip_roll", "hip_yaw"))
CLASSES = ("hardware-hardware", "shell-hardware", "shell-shell")
VERIFY_SHIFT = 1e-5  # m, probe shift used to re-evaluate a suspicious distance
VERIFY_TOL = 5e-5  # m, agreement required between probes

_GT = mujoco.mjtGeom
_CAPSULE_BOX = {int(_GT.mjGEOM_CAPSULE), int(_GT.mjGEOM_BOX)}
_SHIFTS = np.array([[1.0, 0, 0], [0, 0.6, 0.8], [0, 1.0, 0], [-0.8, 0, 0.6], [0, 0, -1.0]]) * VERIFY_SHIFT
_BOX = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)
_NCYL = 128
_CYL = np.array([[np.cos(a) / np.cos(np.pi / _NCYL), np.sin(a) / np.cos(np.pi / _NCYL), z]
                 for a in np.linspace(0, 2 * np.pi, _NCYL, endpoint=False) for z in (-1.0, 1.0)])


# ================================================================================================
# Exact convex distance (verification fallback)
# ================================================================================================
def _div(x, y):
    return x / np.where(np.abs(y) > 1e-300, y, 1.0)


def _origin_triangle_distance(a, b, c) -> np.ndarray:
    """Distance from the origin to each triangle (a[i], b[i], c[i]) (Ericson, closest point)."""
    ab, ac = b - a, c - a
    d1, d2 = -np.einsum("ij,ij->i", ab, a), -np.einsum("ij,ij->i", ac, a)
    d3, d4 = -np.einsum("ij,ij->i", ab, b), -np.einsum("ij,ij->i", ac, b)
    d5, d6 = -np.einsum("ij,ij->i", ab, c), -np.einsum("ij,ij->i", ac, c)
    va, vb, vc = d3 * d6 - d5 * d4, d5 * d2 - d1 * d6, d1 * d4 - d3 * d2
    den = va + vb + vc
    cases = [
        ((d1 <= 0) & (d2 <= 0), a),
        ((d3 >= 0) & (d4 <= d3), b),
        ((vc <= 0) & (d1 >= 0) & (d3 <= 0), a + _div(d1, d1 - d3)[:, None] * ab),
        ((d6 >= 0) & (d5 <= d6), c),
        ((vb <= 0) & (d2 >= 0) & (d6 <= 0), a + _div(d2, d2 - d6)[:, None] * ac),
        ((va <= 0) & (d4 - d3 >= 0) & (d5 - d6 >= 0), b + _div(d4 - d3, (d4 - d3) + (d5 - d6))[:, None] * (c - b)),
        (np.ones(len(a), bool), a + _div(vb, den)[:, None] * ab + _div(vc, den)[:, None] * ac),
    ]
    out = np.empty_like(a)
    done = np.zeros(len(a), bool)
    for mask, pts in cases:
        sel = mask & ~done
        out[sel] = pts[sel]
        done |= sel
    return np.linalg.norm(out, axis=1)


def _segment_distance(P, Q) -> float:
    """Distance between segments P[0]-P[-1] and Q[0]-Q[-1] (points allowed)."""
    p0, q0 = P[0], Q[0]
    d1, d2, r = P[-1] - p0, Q[-1] - q0, p0 - q0
    a, e, f = d1 @ d1, d2 @ d2, d2 @ r
    eps = 1e-18
    if a <= eps and e <= eps:
        return float(np.linalg.norm(r))
    if a <= eps:
        s, t = 0.0, float(np.clip(f / e, 0, 1))
    else:
        c = d1 @ r
        if e <= eps:
            s, t = float(np.clip(-c / a, 0, 1)), 0.0
        else:
            b = d1 @ d2
            den = a * e - b * b
            s = float(np.clip((b * f - c * e) / den, 0, 1)) if den > eps else 0.0
            t = (b * s + f) / e
            if t < 0:
                s, t = float(np.clip(-c / a, 0, 1)), 0.0
            elif t > 1:
                s, t = float(np.clip((b - c) / a, 0, 1)), 1.0
    return float(np.linalg.norm(p0 + d1 * s - q0 - d2 * t))


def segment_box_signed_distance(p0, p1, h) -> float:
    """Exact signed distance between segment p0-p1 and the box |x_i| <= h_i (box frame).

    Separated: the squared point-box distance along the segment is piecewise quadratic between
    the face-plane crossings; each piece is minimised in closed form.  Intersecting: minus the
    minimum translation distance, the smallest overlap over the Minkowski-difference facet normals
    (the three box axes and the segment direction crossed with each box axis).
    """
    p0, p1, h = [float(v) for v in p0], [float(v) for v in p1], [float(v) for v in h]
    dv = [p1[i] - p0[i] for i in range(3)]
    ts = [0.0, 1.0]
    for i in range(3):
        if abs(dv[i]) > 1e-15:
            for s in (-h[i], h[i]):
                t = (s - p0[i]) / dv[i]
                if 0.0 < t < 1.0:
                    ts.append(t)
    ts.sort()
    best = np.inf
    for a, b in zip(ts[:-1], ts[1:]):
        tm = 0.5 * (a + b)
        qa = qb = qc = 0.0
        for i in range(3):
            pm = p0[i] + tm * dv[i]
            if pm > h[i]:
                e = p0[i] - h[i]
            elif pm < -h[i]:
                e = p0[i] + h[i]
            else:
                continue
            qa += dv[i] * dv[i]
            qb += 2.0 * e * dv[i]
            qc += e * e
        t = min(max(-qb / (2.0 * qa), a), b) if qa > 0.0 else a
        best = min(best, qa * t * t + qb * t + qc)
    if best > 1e-22:
        return float(np.sqrt(best))
    ln = float(np.sqrt(sum(x * x for x in dv)))
    axes = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
    if ln > 1e-12:
        u = [x / ln for x in dv]
        for ex in axes[:3]:
            c = (u[1] * ex[2] - u[2] * ex[1], u[2] * ex[0] - u[0] * ex[2], u[0] * ex[1] - u[1] * ex[0])
            cn = float(np.sqrt(sum(x * x for x in c)))
            if cn > 1e-9:
                axes.append(tuple(x / cn for x in c))
    mtd = np.inf
    for n in axes:
        a0 = n[0] * p0[0] + n[1] * p0[1] + n[2] * p0[2]
        a1 = n[0] * p1[0] + n[1] * p1[1] + n[2] * p1[2]
        hb = abs(n[0]) * h[0] + abs(n[1]) * h[1] + abs(n[2]) * h[2]
        mtd = min(mtd, max(a0, a1) + hb, hb - min(a0, a1))
    return -float(mtd)


def convex_signed_distance(A: np.ndarray, ra: float, B: np.ndarray, rb: float) -> float:
    """Exact signed distance between conv(A) inflated by ra and conv(B) inflated by rb.

    Uses the Minkowski difference hull D = A - B: outside, the distance from the origin to D's
    boundary triangles; inside, minus the distance to the nearest facet plane (the penetration
    depth).  Exact for polytopes; capsules and spheres enter as their core plus radius.
    """
    if len(A) <= 2 and len(B) <= 2:
        return _segment_distance(A, B) - ra - rb
    D = (A[:, None, :] - B[None, :, :]).reshape(-1, 3)
    try:
        hull = ConvexHull(D)
    except QhullError:
        hull = ConvexHull(D, qhull_options="QJ")
    off = hull.equations[:, 3]  # n.x + off <= 0 inside; at the origin the value is off
    if np.all(off <= 0.0):
        sd = float(off.max())
    else:
        tri = D[hull.simplices]
        sd = float(_origin_triangle_distance(tri[:, 0], tri[:, 1], tri[:, 2]).min())
    return sd - ra - rb


# ================================================================================================
# Screening context
# ================================================================================================
def _mm(x, nd=2):
    if x is None or not np.isfinite(x):
        return None
    return round(float(x) * 1000.0, nd) + 0.0


def _panel_key(name: str) -> str:
    """cl_<panel>_<k> -> cl_<panel> (proxy pieces of one panel collapse to the panel)."""
    return re.sub(r"_\d+$", "", name) if name.startswith("cl_") else name


class _Screen:
    """Compiled model, the screened geom set, the candidate pair table and the distance oracle."""

    def __init__(self, xml_path):
        m = mujoco.MjModel.from_xml_path(str(xml_path))
        self.m, self.d = m, mujoco.MjData(m)
        self.gname = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or f"geom_{g}" for g in range(m.ngeom)]
        self.bname = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or f"body_{b}" for b in range(m.nbody)]
        # rigid groups: a body without joints moves with its parent
        grp = np.arange(m.nbody)
        for b in range(1, m.nbody):
            r = b
            while r != 0 and m.body_jntnum[r] == 0:
                r = int(m.body_parentid[r])
            grp[b] = r
        gpar = np.full(m.nbody, -1)
        for b in range(1, m.nbody):
            if grp[b] == b:
                gpar[b] = grp[m.body_parentid[b]]
        self.grp, self.gpar = grp, gpar
        self.geom_grp = grp[m.geom_bodyid]
        self.screened = np.array([g for g in range(m.ngeom) if self._is_screened(g)], int)
        self.kind = {int(g): ("shell" if self.gname[g].startswith("cl_") else "hardware") for g in self.screened}
        # candidate pairs: different rigid groups, not parent/child
        G = self.geom_grp[self.screened]
        excl = np.zeros((m.nbody, m.nbody), bool)
        for b in range(m.nbody):
            excl[b, b] = True
            if gpar[b] >= 0:
                excl[b, gpar[b]] = excl[gpar[b], b] = True
        ok = np.triu(~excl[G[:, None], G[None, :]], 1)
        ia, ib = np.nonzero(ok)
        self.I, self.J = self.screened[ia], self.screened[ib]
        bp_index, self.bp_list = {}, []
        bp = np.empty(len(self.I), int)
        for k, (a, b) in enumerate(zip(self.geom_grp[self.I], self.geom_grp[self.J])):
            key = (min(a, b), max(a, b))
            if key not in bp_index:
                bp_index[key] = len(self.bp_list)
                self.bp_list.append(key)
            bp[k] = bp_index[key]
        self.pair_bp = bp
        self.bp_link = [self._link(a, b) for a, b in self.bp_list]
        cls = []
        for a, b in zip(self.I, self.J):
            kinds = (self.kind[int(a)], self.kind[int(b)])
            cls.append("shell-shell" if kinds == ("shell", "shell") else
                       "hardware-hardware" if kinds == ("hardware", "hardware") else "shell-hardware")
        self.pair_cls = np.array(cls)
        # panel-level pair identity (proxy pieces of one panel count as one part)
        pk, pp = {}, np.empty(len(self.I), int)
        for k, (a, b) in enumerate(zip(self.I, self.J)):
            pp[k] = pk.setdefault((_panel_key(self.gname[a]), _panel_key(self.gname[b])), len(pk))
        self.pair_panel = pp
        self.excluded_body_pairs = sorted({(self.bname[int(gpar[b])], self.bname[b]) for b in range(1, m.nbody)
                                           if grp[b] == b and gpar[b] > 0})
        self._verts = {}
        self._ft = np.zeros(6)
        self._cache = {}
        self._type = [int(t) for t in m.geom_type]
        self.stats = dict(distance_calls=0, capsule_box_exact=0, verified=0, consensus=0, exact=0, corrected=0,
                          cache_hits=0)

    # ------------------------------------------------------------------ setup helpers
    def _is_screened(self, g) -> bool:
        m, nm = self.m, self.gname[g]
        if nm.startswith("cl_"):
            return True
        if not nm.startswith("hw_") or m.geom_group[g] not in (0, 2) or any(t in nm for t in SKIP_TOKENS):
            return False
        return bool(m.geom_rbound[g] > 0 and np.any(m.geom_size[g] > 0))

    def _link(self, a, b) -> str:
        for top, low in ((a, b), (b, a)):
            mid = self.gpar[low]
            if mid > 0 and self.gpar[mid] == top and np.linalg.norm(self.m.body_pos[low]) < 1e-9:
                return "2-dof-joint"
        return "non-adjacent"

    def gid(self, name) -> int:
        return mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, name)

    def jnt(self, name):
        """(name, qpos address, lower, upper) of a limited hinge, or None."""
        m = self.m
        j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
        if j < 0 or m.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE or not m.jnt_limited[j]:
            return None
        return name, int(m.jnt_qposadr[j]), float(m.jnt_range[j][0]), float(m.jnt_range[j][1])

    def body_of(self, g) -> str:
        return self.bname[self.geom_grp[g]]

    def verts(self, g) -> np.ndarray:
        if g not in self._verts:
            mid = self.m.geom_dataid[g]
            a, n = self.m.mesh_vertadr[mid], self.m.mesh_vertnum[mid]
            self._verts[g] = np.array(self.m.mesh_vert[a : a + n], float)
        return self._verts[g]

    # ------------------------------------------------------------------ pose and geometry
    def set_q(self, q):
        self.d.qpos[:] = q
        mujoco.mj_kinematics(self.m, self.d)

    def aabb(self):
        """Conservative world AABBs of all geoms (local geom AABB rotated into the world)."""
        d, m = self.d, self.m
        R = d.geom_xmat.reshape(-1, 3, 3)
        c = d.geom_xpos + np.einsum("nij,nj->ni", R, m.geom_aabb[:, :3])
        h = np.einsum("nij,nj->ni", np.abs(R), m.geom_aabb[:, 3:])
        return c - h, c + h

    def core(self, g):
        """World points whose hull, inflated by the returned radius, is geom g."""
        m, d = self.m, self.d
        R, p, s, t = d.geom_xmat[g].reshape(3, 3), d.geom_xpos[g], m.geom_size[g], m.geom_type[g]
        if t == _GT.mjGEOM_MESH:
            return p + self.verts(g) @ R.T, 0.0
        if t == _GT.mjGEOM_BOX:
            return p + (_BOX * s[:3]) @ R.T, 0.0
        if t == _GT.mjGEOM_CYLINDER:
            return p + (_CYL * np.array([s[0], s[0], s[1]])) @ R.T, 0.0
        if t == _GT.mjGEOM_CAPSULE:
            return p + np.array([[0.0, 0.0, -s[1]], [0.0, 0.0, s[1]]]) @ R.T, float(s[0])
        if t == _GT.mjGEOM_SPHERE:
            return p[None, :].copy(), float(s[0])
        raise ValueError(f"geom type {t} of {self.gname[g]} not supported by the exact check")

    def min_z(self, g) -> float:
        """Lowest world z of geom g (exact for mesh vertices and primitives)."""
        m, d = self.m, self.d
        R, z, s, t = d.geom_xmat[g].reshape(3, 3), d.geom_xpos[g][2], m.geom_size[g], m.geom_type[g]
        if t == _GT.mjGEOM_MESH:
            return float(z + (self.verts(g) @ R[2]).min())
        if t == _GT.mjGEOM_BOX:
            return float(z - np.abs(R[2]) @ s[:3])
        if t == _GT.mjGEOM_CAPSULE:
            return float(z - abs(R[2, 2]) * s[1] - s[0])
        if t == _GT.mjGEOM_CYLINDER:
            return float(z - abs(R[2, 2]) * s[1] - s[0] * np.sqrt(max(0.0, 1.0 - R[2, 2] ** 2)))
        if t == _GT.mjGEOM_SPHERE:
            return float(z - s[0])
        if t == _GT.mjGEOM_ELLIPSOID:
            return float(z - np.sqrt(np.sum((R[2] * s[:3]) ** 2)))
        return float(z - m.geom_rbound[g])

    def lowest(self, geoms):
        vals = [self.min_z(g) for g in geoms]
        k = int(np.argmin(vals))
        return vals[k], int(geoms[k])

    # ------------------------------------------------------------------ distance oracle
    def exact(self, g1, g2) -> float:
        A, ra = self.core(g1)
        B, rb = self.core(g2)
        return convex_signed_distance(A, ra, B, rb)

    def _key(self, g1, g2):
        d = self.d
        R1 = d.geom_xmat[g1].reshape(3, 3)
        rel = np.concatenate([R1.T @ (d.geom_xpos[g2] - d.geom_xpos[g1]),
                              (R1.T @ d.geom_xmat[g2].reshape(3, 3)).ravel()])
        return g1, g2, np.round(rel, 9).tobytes()

    def capsule_box(self, g1, g2) -> float:
        """Exact signed distance of a capsule-box pair (either order)."""
        d, m = self.d, self.m
        gc, gb = (g1, g2) if self._type[g1] == _GT.mjGEOM_CAPSULE else (g2, g1)
        P, r = self.core(gc)
        Rb = d.geom_xmat[gb].reshape(3, 3)
        loc = (P - d.geom_xpos[gb]) @ Rb
        return segment_box_signed_distance(loc[0], loc[1], m.geom_size[gb][:3]) - r

    def dist(self, g1, g2, distmax) -> float:
        """Signed distance (m), capped at distmax; results below TIGHT are verified (see module doc)."""
        m, d, ft = self.m, self.d, self._ft
        self.stats["distance_calls"] += 1
        if {self._type[g1], self._type[g2]} == _CAPSULE_BOX:
            # MuJoCo's analytic capsule-box routine misreports deep penetration (even as a small
            # positive distance), so these pairs always use the exact segment-box computation.
            self.stats["capsule_box_exact"] += 1
            return min(self.capsule_box(g1, g2), distmax)
        r = mujoco.mj_geomDistance(m, d, g1, g2, distmax, ft)
        if r >= TIGHT:
            return r
        key = self._key(g1, g2)
        hit = self._cache.get(key)
        if hit is not None:
            self.stats["cache_hits"] += 1
            return min(hit, distmax)
        val = self._verify(g1, g2, distmax, r)
        if val < 0.0:
            # A penetration that the shifted probes agree on can still be a sign flip of the convex routine
            # (round two: one frame read -9.65 mm between frames at +9.7 mm).  Cross-check with the exact
            # convex distance and trust it when it says the pair is separated.
            ex = self.exact(g1, g2)
            if ex > VERIFY_TOL:
                self.stats["sign_flip_fixed"] = self.stats.get("sign_flip_fixed", 0) + 1
                val = ex
        self._cache[key] = val
        if abs(val - r) > VERIFY_TOL:
            self.stats["corrected"] += 1
        return min(val, distmax)

    def _verify(self, g1, g2, distmax, r) -> float:
        m, d, ft = self.m, self.d, self._ft
        self.stats["verified"] += 1
        p0 = d.geom_xpos[g2].copy()
        vals = [r]
        try:
            for k, dv in enumerate(_SHIFTS):
                d.geom_xpos[g2] = p0 + dv
                vals.append(mujoco.mj_geomDistance(m, d, g1, g2, distmax, ft))
                if k == 1 and min(abs(v) for v in vals) > 0.0 and max(vals) - min(vals) <= VERIFY_TOL:
                    return r
        finally:
            d.geom_xpos[g2] = p0
        nz = sorted(v for v in vals if v != 0.0)
        best = (0, -1)
        for i in range(len(nz)):
            j = i
            while j + 1 < len(nz) and nz[j + 1] - nz[i] <= VERIFY_TOL:
                j += 1
            if j - i > best[1] - best[0]:
                best = (i, j)
        if best[1] - best[0] + 1 >= 3:
            self.stats["consensus"] += 1
            return float(np.median(nz[best[0] : best[1] + 1]))
        self.stats["exact"] += 1
        return self.exact(g1, g2)

    def near_pairs(self, cap=REPORT_CAP, mask=None):
        """(pair indices, signed distances) of candidate pairs closer than cap at the current pose."""
        lo, hi = self.aabb()
        ga, gb = self.I, self.J
        near = np.all((lo[ga] - cap <= hi[gb]) & (lo[gb] - cap <= hi[ga]), axis=1)
        if mask is not None:
            near &= mask
        idx = np.flatnonzero(near)
        ks, ds = [], []
        for k, a, b in zip(idx.tolist(), ga[idx].tolist(), gb[idx].tolist()):
            r = self.dist(a, b, cap)
            if r < cap:
                ks.append(k)
                ds.append(r)
        return np.array(ks, int), np.array(ds, float)

    def pair_record(self, k, val, at) -> dict:
        a, b = int(self.I[k]), int(self.J[k])
        return {"geoms": [self.gname[a], self.gname[b]], "bodies": [self.body_of(a), self.body_of(b)],
                "class": str(self.pair_cls[k]), "link": self.bp_link[self.pair_bp[k]], "min_mm": _mm(val), "at": at}


class _Family:
    """Per-pair, per-body-pair and per-pose minima over one pose family."""

    def __init__(self, S: _Screen, name: str):
        self.S, self.name = S, name
        n, nb = len(S.I), len(S.bp_list)
        self.poses = []
        self.min = np.full(n, np.inf)
        self.arg = np.full(n, -1)
        self.n_int = np.zeros(n, int)
        self.n_tight = np.zeros(n, int)
        self.bp_min = np.full(nb, np.inf)
        self.bp_arg = np.full(nb, -1)
        self.bp_pair = np.full(nb, -1)
        self.pose_min, self.pose_pair = [], []

    def add(self, desc: dict, ks: np.ndarray, ds: np.ndarray) -> int:
        p = len(self.poses)
        self.poses.append(desc)
        if not len(ks):
            self.pose_min.append(np.inf)
            self.pose_pair.append(-1)
            return p
        upd = ds < self.min[ks]
        self.min[ks[upd]] = ds[upd]
        self.arg[ks[upd]] = p
        self.n_int[ks[ds < 0]] += 1
        self.n_tight[ks[(ds >= 0) & (ds < TIGHT)]] += 1
        o = np.argsort(ds, kind="stable")
        bps = self.S.pair_bp[ks[o]]
        ub, first = np.unique(bps, return_index=True)
        vals, prs = ds[o][first], ks[o][first]
        better = vals < self.bp_min[ub]
        self.bp_min[ub[better]] = vals[better]
        self.bp_arg[ub[better]] = p
        self.bp_pair[ub[better]] = prs[better]
        j = int(np.argmin(ds))
        self.pose_min.append(float(ds[j]))
        self.pose_pair.append(int(ks[j]))
        return p

    def counts(self, home_min=None) -> dict:
        S = self.S
        neg = self.min < 0
        tight = (self.min >= 0) & (self.min < TIGHT)
        out = dict(interference_pairs=int(neg.sum()), tight_pairs=int(tight.sum()),
                   interference_body_pairs=len(set(S.pair_bp[neg].tolist())),
                   tight_body_pairs=len(set(S.pair_bp[neg | tight].tolist()) - set(S.pair_bp[neg].tolist())),
                   by_class={c: dict(interference=int((neg & (S.pair_cls == c)).sum()),
                                     tight=int((tight & (S.pair_cls == c)).sum())) for c in CLASSES})
        if home_min is not None:
            out["interference_pairs_not_at_home"] = int((neg & self.new_vs_home(home_min)).sum())
        return out

    def new_vs_home(self, home_min) -> np.ndarray:
        """Pairs whose panels/parts do not interfere anywhere at home."""
        return ~np.isin(self.S.pair_panel, self.S.pair_panel[home_min < 0])

    def worst(self) -> dict | None:
        if not self.poses:
            return None
        p = int(np.argmin(self.pose_min))
        if not np.isfinite(self.pose_min[p]):
            return dict(min_mm=None, note=f"no screened pair within {REPORT_CAP * 1000:.0f} mm")
        return self.S.pair_record(self.pose_pair[p], self.pose_min[p], self.poses[p]) | dict(pose_index=p)

    def report(self, home_min=None, max_pairs=40) -> dict:
        S = self.S
        ks = np.flatnonzero(self.min < TIGHT)
        ks = ks[np.argsort(self.min[ks], kind="stable")]
        n_listed = len(ks)
        pairs = []
        for k in ks[:max_pairs]:
            rec = S.pair_record(k, self.min[k], self.poses[self.arg[k]])
            if len(self.poses) > 1:
                rec["n_poses_interference"] = int(self.n_int[k])
                rec["n_poses_tight"] = int(self.n_tight[k])
            if home_min is not None:
                rec["home_mm"] = _mm(home_min[k]) if np.isfinite(home_min[k]) else f"> {REPORT_CAP * 1000:.0f}"
            pairs.append(rec)
        bps = np.flatnonzero(np.isfinite(self.bp_min))
        bps = bps[np.argsort(self.bp_min[bps], kind="stable")]
        body_pairs = []
        for b in bps:
            k = int(self.bp_pair[b])
            body_pairs.append(dict(bodies=[S.bname[x] for x in S.bp_list[b]], min_mm=_mm(self.bp_min[b]),
                                   geoms=[S.gname[int(S.I[k])], S.gname[int(S.J[k])]], **{"class": str(S.pair_cls[k])},
                                   link=S.bp_link[b], at=self.poses[self.bp_arg[b]]))
        return dict(n_poses=len(self.poses), counts=self.counts(home_min), worst=self.worst(),
                    pairs_below_tight=n_listed, pairs_listed=len(pairs), pairs=pairs, body_pairs=body_pairs,
                    body_pairs_beyond_cap=len(S.bp_list) - len(body_pairs))


# ================================================================================================
# Checks
# ================================================================================================
def _hip_height(S: _Screen, H):
    if H is not None:
        return float(H), "argument"
    zs = []
    for side in SIDES:
        j = mujoco.mj_name2id(S.m, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_hip_pitch")
        if j >= 0:
            zs.append(float(S.d.xanchor[j][2]))
    if zs:
        return float(np.mean(zs)), "model: mean hip-pitch axis height at home"
    return float(S.d.xpos[S.m.jnt_bodyid[0]][2]), "model: root body height at home"


def _sweep_list(S: _Screen) -> list:
    sweeps = []

    def add_single(name):
        j = S.jnt(name)
        if j:
            for end, val in (("min", j[2]), ("max", j[3])):
                sweeps.append((f"{name}:{end}", [(j[0], j[1], val)]))

    for side in SIDES:
        for jn in LEG_JOINTS:
            add_single(f"{side}_{jn}")
        for a, b in CORNERS:
            ja, jb = S.jnt(f"{side}_{a}"), S.jnt(f"{side}_{b}")
            if not (ja and jb):
                continue
            for ea, va in (("min", ja[2]), ("max", ja[3])):
                for eb, vb in (("min", jb[2]), ("max", jb[3])):
                    sweeps.append((f"{ja[0]}:{ea}+{jb[0]}:{eb}", [(ja[0], ja[1], va), (jb[0], jb[1], vb)]))
    add_single("neck_pitch")
    add_single("tail_yaw")
    add_single("neck_yaw")  # the expressive head's joints (Stage 2), on a model that has them
    add_single("jaw")
    return sweeps


def _foot_geoms(S: _Screen):
    m = S.m
    pads, leg = {}, {}
    for side in SIDES:
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{side}_foot")
        pads[side] = [g for g in range(m.ngeom) if m.geom_bodyid[g] == b and m.geom_contype[g] == 1]
        leg[side] = [int(g) for g in S.screened if S.body_of(g).startswith(f"{side}_")]
    return pads, leg


def _swing_runs(phases, side) -> list[list[int]]:
    """Contiguous (cyclic) runs of frames in which ``side`` is not in the support tuple."""
    sw = np.array([side not in tuple(p) for p in phases])
    n = len(sw)
    if not sw.any():
        return []
    if sw.all():
        return [list(range(n))]
    start = int(np.argmin(sw))
    runs, cur = [], []
    for j in range(n):
        k = (start + j) % n
        if sw[k]:
            cur.append(k)
        elif cur:
            runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)
    return runs


def _swing_report(S, gait_name, phases, pad_h, lat, H, n) -> dict:
    out = {}
    worst_mid = (np.inf, None, None)
    for side in SIDES:
        runs = _swing_runs(phases, side)
        h = pad_h[side]
        full, mid, apex = (np.inf, None), (np.inf, None), np.inf
        for run in runs:
            L = len(run)
            vals = [(h[k], k, j / L) for j, k in enumerate(run) if np.isfinite(h[k])]
            if not vals:
                continue
            for v, k, s in vals:
                if v < full[0]:
                    full = (v, k)
                if MID_SWING[0] <= s <= MID_SWING[1] and v < mid[0]:
                    mid = (v, k)
            apex = min(apex, max(v for v, _, _ in vals))
        out[side] = dict(swings=len(runs), full_swing_min_mm=_mm(full[0]), full_swing_min_frame=full[1],
                         mid_swing_min_mm=_mm(mid[0]), mid_swing_min_frame=mid[1],
                         mid_swing_min_frac_H=round(mid[0] / H, 4) if np.isfinite(mid[0]) else None,
                         apex_mm=_mm(apex), apex_frac_H=round(apex / H, 4) if np.isfinite(apex) else None)
        if mid[0] < worst_mid[0]:
            worst_mid = (mid[0], side, mid[1])
    out["mid_swing_min_mm"] = _mm(worst_mid[0])
    out["mid_swing_min_frac_H"] = round(worst_mid[0] / H, 4) if np.isfinite(worst_mid[0]) else None
    out["mid_swing_min_at"] = (dict(gait=gait_name, foot=worst_mid[1], frame=worst_mid[2],
                                    phase=round(worst_mid[2] / n, 3)) if worst_mid[1] else None)
    out["lateral_gap"] = lat
    return out


def _lateral_gap(S: _Screen, pads, leg, swing, stance):
    """Min signed distance between the swing foot pads and the stance leg's screened geoms."""
    best = (np.inf, None, None, None)
    ft = np.zeros(6)
    for p in pads[swing]:
        for g in leg[stance]:
            r = S.dist(p, g, LATERAL_CAP)
            if r < best[0]:
                mujoco.mj_geomDistance(S.m, S.d, p, g, LATERAL_CAP, ft)
                best = (r, p, g, abs(ft[4] - ft[1]))
    return best


def _same_body_overlaps(S: _Screen) -> dict:
    """Intersecting components (not structure) on one rigid body at the current pose (informational)."""
    hw = [int(g) for g in S.screened if S.kind[int(g)] == "hardware"]
    comp = [g for g in hw if not any(t in S.gname[g] for t in STRUCTURE_TOKENS)]
    lo, hi = S.aabb()
    over, nested, structural = [], 0, 0
    for i, a in enumerate(hw):
        for b in hw[i + 1 :]:
            if S.geom_grp[a] != S.geom_grp[b] or np.any(lo[a] > hi[b]) or np.any(lo[b] > hi[a]):
                continue
            r = S.dist(a, b, REPORT_CAP)
            if r >= 0:
                continue
            if _encloses(S, a, b) or _encloses(S, b, a):
                nested += 1
            elif a in comp and b in comp:
                over.append(dict(geoms=[S.gname[a], S.gname[b]], body=S.body_of(a), min_mm=_mm(r)))
            else:
                structural += 1
    over.sort(key=lambda r: r["min_mm"])
    return dict(overlaps=over, nested_envelopes=nested, structure_contacts=structural,
                note="components (motors, battery, boards, sensors) on one rigid body that intersect without one "
                     "box enclosing the other; overlaps involving structure/linkage/foot parts (names with "
                     f"{', '.join(STRUCTURE_TOKENS)}) are joints or attachments in this model and only counted")


def _encloses(S: _Screen, outer, inner) -> bool:
    m, d = S.m, S.d
    if m.geom_type[outer] != _GT.mjGEOM_BOX:
        return False
    P, r = S.core(inner)
    R = d.geom_xmat[outer].reshape(3, 3)
    loc = (P - d.geom_xpos[outer]) @ R
    return bool(np.all(np.abs(loc) <= m.geom_size[outer][:3] - r + 1e-9))


def _battery(S: _Screen, q_home) -> dict:
    m, d = S.m, S.d
    g = S.gid("hw_battery_cassette")
    if g < 0 or m.geom_type[g] != _GT.mjGEOM_BOX:
        return dict(found=False, note="no box geom named hw_battery_cassette", clear_directions=[])
    S.set_q(q_home)
    trunk = S.geom_grp[g]
    excl = {g, S.gid("hw_battery")}
    obstacles = [int(x) for x in S.screened if int(x) not in excl and
                 (S.geom_grp[x] != trunk or S.kind[int(x)] == "hardware")]
    R = d.geom_xmat[g].reshape(3, 3).copy()
    p0 = d.geom_xpos[g].copy()
    half = m.geom_size[g][:3].copy()
    lo, hi = S.aabb()
    hb = np.abs(R) @ half
    out = dict(found=True, body=S.body_of(g), center_m=[round(float(v), 4) for v in p0],
               size_mm=[_mm(2 * v, 1) for v in half], directions={})
    group1 = np.array([0, 1, 0, 0, 0, 0], np.uint8)
    hitid = np.zeros(1, np.int32)
    try:
        for label, ax, sgn, what in BATTERY_DIRECTIONS:
            a = sgn * R[:, ax]
            travel = 2 * half[ax] + BATTERY_EXTRA
            n = int(np.ceil(travel / BATTERY_STEP - 1e-9))
            steps = np.minimum(BATTERY_STEP * np.arange(1, n + 1), travel)
            c1 = p0 + travel * a
            slo, shi = np.minimum(p0, c1) - hb - TIGHT, np.maximum(p0, c1) + hb + TIGHT
            cand = [x for x in obstacles if np.all(lo[x] <= shi) and np.all(hi[x] >= slo)]
            home, per = {}, {}
            for s in [0.0] + steps.tolist():
                d.geom_xpos[g] = p0 + s * a
                for x in cand:
                    r = S.dist(g, x, REPORT_CAP)
                    if s == 0.0:
                        home[x] = r
                        continue
                    rec = per.setdefault(x, [np.inf, None])
                    rec[0] = min(rec[0], r)
                    if r < 0 and rec[1] is None:
                        rec[1] = s
            d.geom_xpos[g] = p0
            blockers = sorted((v[0], x, v[1]) for x, v in per.items() if v[0] < 0)
            blk = [dict(geom=S.gname[x], body=S.body_of(x), kind=S.kind[x], min_mm=_mm(v),
                        first_blocked_at_travel_mm=_mm(s0, 1), overlaps_at_home=bool(home.get(x, 1.0) < 0))
                   for v, x, s0 in blockers]
            clr = min([v[0] for v in per.values()], default=np.inf)
            # first cover panel crossed: rays from the face centre and four inset corners
            u, v = [i for i in range(3) if i != ax]
            face = p0 + sgn * half[ax] * R[:, ax]
            hits = {}
            for cu, cv in ((0, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)):
                o = face + cu * max(half[u] - 0.001, 0) * R[:, u] + cv * max(half[v] - 0.001, 0) * R[:, v]
                dist = mujoco.mj_ray(m, d, o, a, group1, 1, -1, hitid)
                if dist >= 0 and hitid[0] >= 0:
                    nm = S.gname[int(hitid[0])]
                    hits[nm] = min(hits.get(nm, np.inf), dist)
            out["directions"][label] = dict(
                meaning=what, axis_world=[round(float(x), 4) for x in a], travel_mm=_mm(travel, 1), steps=n,
                clear=not blk, clear_ignoring_home_overlaps=not [b for b in blk if not b["overlaps_at_home"]],
                min_clearance_mm=_mm(clr) if np.isfinite(clr) else f"> {REPORT_CAP * 1000:.0f}",
                blocked_by=blk,
                cover_crossed={k: _mm(v, 1) for k, v in sorted(hits.items(), key=lambda kv: kv[1])})
    finally:
        d.geom_xpos[g] = p0
    out["clear_directions"] = [k for k, v in out["directions"].items() if v["clear"]]
    out["clear_ignoring_home_overlaps"] = [k for k, v in out["directions"].items() if v["clear_ignoring_home_overlaps"]]
    home_ov = {}
    for v in out["directions"].values():
        for b in v["blocked_by"]:
            if b["overlaps_at_home"]:
                home_ov[b["geom"]] = b["body"]
    out["overlapping_at_home"] = sorted(home_ov)
    return out


def _max_step(S: _Screen, frames_q, trunk_body) -> float:
    """Largest relative-to-trunk displacement of any screened surface between consecutive frames."""
    m, d = S.m, S.d
    g = S.screened
    rb = m.geom_rbound[g]
    P, Rs = [], []
    for q in frames_q:
        S.set_q(q)
        Rt = d.xmat[trunk_body].reshape(3, 3)
        P.append((d.geom_xpos[g] - d.xpos[trunk_body]) @ Rt)
        Rs.append(np.einsum("ji,njk->nik", Rt, d.geom_xmat[g].reshape(-1, 3, 3)))
    worst = 0.0
    n = len(frames_q)
    for k in range(n):
        j = (k + 1) % n
        dp = np.linalg.norm(P[j] - P[k], axis=1)
        tr = np.einsum("nij,nij->n", Rs[j], Rs[k])
        ang = np.arccos(np.clip((tr - 1.0) / 2.0, -1.0, 1.0))
        worst = max(worst, float(np.max(dp + rb * ang)))
    return worst


# ================================================================================================
# Main entry
# ================================================================================================
def screen(xml_path, q_home, gait_trajs: dict[str, dict], out_png: str | None = None, *, H: float | None = None,
           frame_stride: int = 1, substeps: int = 0, gates: str = "r1") -> dict:
    """Run the four clearance checks on a compiled candidate; returns a JSON-serialisable dict.

    xml_path    MJCF written by ``pipeline.build_candidate``.
    q_home      nominal standing pose (qpos).
    gait_trajs  {name: trajectory} as returned by ``gait.generate`` (uses ``q`` and ``phases``);
                the "nominal" entry carries the gait pass/fail gate.
    out_png     optional diagnostic plot (distance, swing-pad height and tail height per frame).
    H           hip height (m) for the fractional thresholds; measured from the model if omitted.
    frame_stride  evaluate every n-th gait frame (1 = every frame).
    substeps    extra poses interpolated between consecutive evaluated frames for the interference
                check (0 = the trajectory's frames only); ``max_step_mm`` reports the resulting
                resolution.
    gates       "r1" (round one: home, nominal walk, swing, tail, battery) or "r2" (round two: also
                component overlaps on one body, the brisk walk and joint-range margins).
    """
    t0 = time.perf_counter()
    timing = {}
    S = _Screen(xml_path)
    m = S.m
    q_home = np.asarray(q_home, float).copy()
    S.set_q(q_home)
    H, H_src = _hip_height(S, H)
    pads, leg = _foot_geoms(S)
    tail_b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "tail")
    tail_geoms = [g for g in range(m.ngeom) if tail_b >= 0 and S.grp[m.geom_bodyid[g]] == tail_b]
    trunk_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "trunk")
    trunk_body = trunk_body if trunk_body >= 0 else int(m.jnt_bodyid[0])
    timing["setup"] = time.perf_counter() - t0

    tail_rec = {}

    def tail_update(key, desc):
        if not tail_geoms:
            return np.nan
        z, gz = S.lowest(tail_geoms)
        if key not in tail_rec or z < tail_rec[key][0]:
            tail_rec[key] = (z, gz, desc)
        return z

    # ---------------------------------------------------------------- (a) home
    t = time.perf_counter()
    home = _Family(S, "home")
    ks, ds = S.near_pairs()
    home.add(dict(pose="home"), ks, ds)
    home_min = home.min.copy()
    home_neg_panels = set(S.pair_panel[ks[ds < 0]].tolist())
    same_body = _same_body_overlaps(S)
    S.set_q(q_home)
    tail_update("home", dict(pose="home"))
    home_pad = {s: S.lowest(pads[s])[0] for s in SIDES if pads[s]}
    timing["home"] = time.perf_counter() - t

    # ---------------------------------------------------------------- (b) gait frames
    t = time.perf_counter()
    gait_out, series = {}, {}
    nsub = max(0, int(substeps))
    for gname, tr in gait_trajs.items():
        Q = np.asarray(tr["q"], float)
        n = len(Q)
        phases = [tuple(p) for p in tr.get("phases", [("left", "right")] * n)]
        frames = list(range(0, n, max(1, int(frame_stride))))
        F = _Family(S, f"gait_{gname}")
        pad_h = {s: np.full(n, np.nan) for s in SIDES}
        swing_h = np.full(n, np.nan)
        tail_z = np.full(n, np.nan)
        frame_min = np.full(n, np.inf)
        lat = (np.inf, None, None, None, None)
        evaluated = []
        for i_f, k in enumerate(frames):
            nxt = frames[(i_f + 1) % len(frames)]
            for sub in range(nsub + 1):
                if sub == 0:
                    q = Q[k]
                    desc = dict(gait=gname, frame=k, phase=round(k / n, 3), support=list(phases[k]))
                else:
                    u = sub / (nsub + 1)
                    q = _interpolate(m, Q, k, nxt, u)
                    desc = dict(gait=gname, frame=k, substep=f"{sub}/{nsub + 1}",
                                phase=round(((k + u * ((nxt - k) % n or n)) / n) % 1.0, 4), support=list(phases[k]))
                S.set_q(q)
                evaluated.append(q.copy())
                ks, ds = S.near_pairs()
                F.add(desc, ks, ds)
                if len(ds):
                    frame_min[k] = min(frame_min[k], float(ds.min()))
                if sub:
                    continue
                for s in SIDES:
                    if pads[s]:
                        pad_h[s][k] = S.lowest(pads[s])[0]
                sw = [s for s in SIDES if s not in phases[k]]
                if len(sw) == 1:
                    swing_h[k] = pad_h[sw[0]][k]
                    stance = "right" if sw[0] == "left" else "left"
                    r, gp, gs, dy = _lateral_gap(S, pads, leg, sw[0], stance)
                    if r < lat[0]:
                        lat = (r, gp, gs, dy, desc)
                tail_z[k] = tail_update(f"gait_{gname}", desc)
        lat_out = None
        if lat[1] is not None:
            S.set_q(Q[lat[4]["frame"]])
            ex = S.exact(lat[1], lat[2])
            lat_out = dict(min_mm=_mm(ex), lateral_component_mm=_mm(lat[3]), geoms=[S.gname[lat[1]], S.gname[lat[2]]],
                           bodies=[S.body_of(lat[1]), S.body_of(lat[2])], at=lat[4], verified="exact")
        rep = F.report(home_min)
        rep["frames"] = n
        rep["frames_evaluated"] = len(frames)
        rep["frame_stride"] = max(1, int(frame_stride))
        rep["substeps"] = nsub
        rep["ik_ok"] = bool(tr["ik_ok"]) if "ik_ok" in tr else None
        rep["max_step_mm"] = _mm(_max_step(S, evaluated, trunk_body), 1)
        fm = np.minimum(frame_min[frames], REPORT_CAP)
        rep["min_mm_by_frame"] = [_mm(v) for v in fm]
        gait_out[gname] = dict(interference=rep, swing=_swing_report(S, gname, phases, pad_h, lat_out, H, n))
        series[gname] = dict(phase=np.array(frames) / n, min_mm=fm * 1000.0,
                             swing_frac=swing_h[frames] / H, tail_frac=tail_z[frames] / H)
    timing["gaits"] = time.perf_counter() - t

    # ---------------------------------------------------------------- (c) joint sweeps
    t = time.perf_counter()
    ext = _Family(S, "joint_extremes")
    sweeps_out = []
    for label, joints in _sweep_list(S):
        dq = [target - q_home[adr] for _, adr, target in joints]
        n = max(1, int(np.ceil(max(abs(x) for x in dq) / SWEEP_STEP - 1e-9)))
        best = (np.inf, -1, None)
        new, first_new = {}, None
        for i in range(1, n + 1):
            q = q_home.copy()
            vals = {}
            for (jn, adr, _), dqi in zip(joints, dq):
                q[adr] = q_home[adr] + dqi * i / n
                vals[jn] = round(float(q[adr]), 4)
            S.set_q(q)
            ks, ds = S.near_pairs()
            desc = dict(sweep=label, q=vals)
            ext.add(desc, ks, ds)
            if len(ds):
                j = int(np.argmin(ds))
                if ds[j] < best[0]:
                    best = (float(ds[j]), int(ks[j]), vals)
                fresh = [(v, k) for k, v in zip(ks.tolist(), ds.tolist())
                         if v < 0 and S.pair_panel[k] not in home_neg_panels]
                if fresh and first_new is None:
                    v, k = min(fresh)
                    first_new = dict(at=vals, geoms=_pair_names(S, k), min_mm=_mm(v))
                for v, k in fresh:
                    if k not in new:
                        new[k] = [v, vals]
                    else:
                        new[k][0] = min(new[k][0], v)
            if joints[0][0] == "tail_yaw":
                tail_update("tail_yaw_range", desc)
        new_l = sorted(new.items(), key=lambda kv: kv[1][0])
        sweeps_out.append(dict(
            sweep=label, joints={jn: [round(float(q_home[adr]), 4), round(float(tg), 4)] for jn, adr, tg in joints},
            steps=n, min_mm=_mm(best[0]) if np.isfinite(best[0]) else f"> {REPORT_CAP * 1000:.0f}",
            worst=(dict(geoms=_pair_names(S, best[1]), **{"class": str(S.pair_cls[best[1]])}, min_mm=_mm(best[0]),
                        at=best[2]) if best[1] >= 0 else None),
            first_new_interference=first_new,
            n_new_interference_pairs=len(new_l),
            deepest_new_interference=[dict(geoms=_pair_names(S, k), **{"class": str(S.pair_cls[k])}, min_mm=_mm(v),
                                           first_at=first) for k, (v, first) in new_l[:3]]))
    timing["joint_sweeps"] = time.perf_counter() - t

    # ---------------------------------------------------------------- (d) tail limits within the gaits
    t = time.perf_counter()
    combo = _Family(S, "tail_limits_in_gait")
    jt = S.jnt("tail_yaw")
    tail_mask = (S.geom_grp[S.I] == tail_b) | (S.geom_grp[S.J] == tail_b)
    if jt is not None and tail_b >= 0:
        for gname, tr in gait_trajs.items():
            Q = np.asarray(tr["q"], float)
            n = len(Q)
            for ph in TAIL_PHASES:
                k = int(round(ph * n)) % n
                for end, val in (("min", jt[2]), ("max", jt[3])):
                    q = Q[k].copy()
                    q[jt[1]] = val
                    S.set_q(q)
                    desc = dict(gait=gname, frame=k, phase=round(k / n, 3), tail_yaw=round(val, 4))
                    ks, ds = S.near_pairs(mask=tail_mask)
                    combo.add(desc, ks, ds)
                    tail_update("tail_limits_in_gait", desc)
    timing["tail_combos"] = time.perf_counter() - t

    # ---------------------------------------------------------------- battery
    t = time.perf_counter()
    battery = _battery(S, q_home)
    timing["battery"] = time.perf_counter() - t

    # ---------------------------------------------------------------- assemble
    interference = {"home": home.report()}
    for gname in gait_trajs:
        interference[f"gait_{gname}"] = gait_out[gname]["interference"]
    interference["joint_extremes"] = ext.report(home_min, max_pairs=25) | dict(sweeps=sweeps_out)
    interference["tail_limits_in_gait"] = combo.report(home_min)

    tail_out = {}
    for key, (z, gz, desc) in tail_rec.items():
        tail_out[key] = dict(min_mm=_mm(z, 1), frac_H=round(z / H, 4), geom=S.gname[gz], at=desc)
    gate_keys = [k for k in tail_out if k == "home" or k == "tail_yaw_range" or k.startswith("gait_")]
    tail_gate = min((tail_rec[k][0] for k in gate_keys), default=np.nan)
    tail_gate_key = min(gate_keys, key=lambda k: tail_rec[k][0]) if gate_keys else None

    swing = {g: v["swing"] for g, v in gait_out.items()}
    report = dict(
        module="designlab.clearance", study_revision=STUDY_REVISION, xml=str(xml_path), model=_model_name(xml_path),
        H_m=round(H, 5), H_source=H_src,
        thresholds=dict(tight_mm=TIGHT * 1000, report_cap_mm=REPORT_CAP * 1000, swing_min_frac_H=SWING_MIN_FRAC,
                        swing_min_mm=_mm(SWING_MIN_FRAC * H, 1), mid_swing_window=list(MID_SWING),
                        tail_min_frac_H=TAIL_MIN_FRAC, tail_min_mm=_mm(TAIL_MIN_FRAC * H, 1),
                        sweep_step_rad=SWEEP_STEP, battery_step_mm=BATTERY_STEP * 1000,
                        battery_extra_travel_mm=BATTERY_EXTRA * 1000, verify_shift_mm=VERIFY_SHIFT * 1000,
                        verify_tol_mm=VERIFY_TOL * 1000),
        geometry=dict(screened_hardware=sum(1 for v in S.kind.values() if v == "hardware"),
                      screened_shell_proxies=sum(1 for v in S.kind.values() if v == "shell"),
                      skipped_cosmetic=[S.gname[g] for g in range(m.ngeom) if S.gname[g].startswith("hw_")
                                        and any(tk in S.gname[g] for tk in SKIP_TOKENS)],
                      covers_without_proxies=_covers_without_proxies(S),
                      candidate_pairs=int(len(S.I)), body_pairs=len(S.bp_list),
                      excluded_parent_child=[list(p) for p in S.excluded_body_pairs],
                      two_dof_joint_body_pairs=[[S.bname[a], S.bname[b]] for (a, b), lk in zip(S.bp_list, S.bp_link)
                                                if lk == "2-dof-joint"]),
        interference=interference,
        same_body_hardware_at_home=same_body,
        swing=swing,
        tail=dict(records=tail_out, home_pad_bottom_mm={s: _mm(v, 2) for s, v in home_pad.items()}),
        battery=battery,
        verification=dict(S.stats),
    )
    report["joint_range_margins"] = _range_margins(S, interference["joint_extremes"]["sweeps"], gait_trajs)
    report["summary"] = _summary(report, H, tail_gate, tail_gate_key, swing, battery, gates=gates)
    timing["total"] = time.perf_counter() - t0
    report["timing_s"] = {k: round(v, 3) for k, v in timing.items()}
    if out_png:
        _plot(series, report["model"] or Path(str(xml_path)).stem, out_png)
        report["png"] = str(out_png)
    return _clean(report)


def _pair_names(S: _Screen, k: int) -> list:
    return [S.gname[int(S.I[k])], S.gname[int(S.J[k])]]


def _interpolate(m, Q, k, nxt, u) -> np.ndarray:
    """Pose a fraction u of the way from frame k to frame nxt (joints linear, free-joint quaternions
    normalised-linear).  Across the stride wrap the root position is extrapolated from the previous
    step instead (only a rigid translation, which does not change body-to-body distances)."""
    qa, qb = Q[k], Q[nxt].copy()
    gap = (nxt - k) % len(Q) or len(Q)
    for j in range(m.njnt):
        if m.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE:
            continue
        a = m.jnt_qposadr[j]
        if nxt <= k:
            qb[a : a + 3] = qa[a : a + 3] + gap * (qa[a : a + 3] - Q[k - 1][a : a + 3])
        if np.dot(qa[a + 3 : a + 7], qb[a + 3 : a + 7]) < 0:
            qb[a + 3 : a + 7] *= -1
    q = qa + u * (qb - qa)
    for j in range(m.njnt):
        if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            a = m.jnt_qposadr[j]
            q[a + 3 : a + 7] /= np.linalg.norm(q[a + 3 : a + 7])
    return q


def _covers_without_proxies(S: _Screen) -> list:
    proxies = {_panel_key(S.gname[g])[3:] for g in S.screened if S.kind[int(g)] == "shell"}
    return [S.gname[g] for g in range(S.m.ngeom) if S.gname[g].startswith("cover_") and S.gname[g][6:] not in proxies]


def _model_name(xml_path) -> str | None:
    try:
        with open(xml_path, encoding="utf-8") as fh:
            head = fh.read(400)
    except OSError:
        return None
    mt = re.search(r'<mujoco[^>]*model="([^"]+)"', head)
    return mt.group(1) if mt else None


def _mirror(name: str) -> str:
    """Side-neutral name (left/right and _l/_r tokens replaced) for merging mirrored findings."""
    name = re.sub(r"(^|_)(left|right)(?=_|$)", r"\1X", name)
    return re.sub(r"_(l|r)(?=_|$)", "_X", name)


def _worst(inter: dict, families, top: int) -> list:
    """Worst pairs over the given families: proxy pieces collapse to their panel, mirror images merge."""
    best = {}
    for fname in families:
        for r in inter[fname]["pairs"]:
            names = tuple(sorted((_mirror(r["bodies"][0]) + "/" + _mirror(_panel_key(r["geoms"][0])),
                                  _mirror(r["bodies"][1]) + "/" + _mirror(_panel_key(r["geoms"][1])))))
            sides = {s for b in r["bodies"] for s in SIDES if b.startswith(s)}
            rec = best.get(names)
            if rec is None or r["min_mm"] < rec["min_mm"]:
                new = dict(geoms=r["geoms"], bodies=r["bodies"], min_mm=r["min_mm"], family=fname, at=r["at"],
                           link=r["link"], **{"class": r["class"]})
                if "n_poses_interference" in r:
                    new["n_poses_interference"] = r["n_poses_interference"]
                new["sides"] = sorted(sides | set(rec["sides"])) if rec else sorted(sides)
                best[names] = new
            else:
                rec["sides"] = sorted(set(rec["sides"]) | sides)
    return sorted(best.values(), key=lambda r: r["min_mm"])[:top]


def _compact_counts(c: dict) -> dict:
    out = dict(interference_pairs=c["interference_pairs"], tight_pairs=c["tight_pairs"],
               interference_body_pairs=c["interference_body_pairs"],
               interference_by_class={k: v["interference"] for k, v in c["by_class"].items()})
    if "interference_pairs_not_at_home" in c:
        out["interference_pairs_not_at_home"] = c["interference_pairs_not_at_home"]
    return out


def _joint_ranges(sweeps: list) -> dict:
    """Single-joint sweeps -> first joint value at which a pair not interfering at home interferes."""
    out = {}
    for s in sweeps:
        if "+" in s["sweep"]:
            continue
        jn, end = s["sweep"].split(":")
        rec = out.setdefault(jn, dict(home=s["joints"][jn][0], limits=[None, None], first_new_interference=[None, None],
                                      limited_by=[None, None]))
        i = 0 if end == "min" else 1
        rec["limits"][i] = s["joints"][jn][1]
        if s["first_new_interference"]:
            rec["first_new_interference"][i] = s["first_new_interference"]["at"][jn]
            rec["limited_by"][i] = s["first_new_interference"]["geoms"]
    return out


def _seam_distances(S: _Screen, A: list, B: list) -> dict:
    """(a, b) -> signed distance (m, capped at ``REPORT_CAP``) of every pair of ``A`` x ``B`` at the current pose, by
    the screen's own oracle (``_Screen.dist``) on the pairs whose world AABBs come within the cap."""
    lo, hi = S.aabb()
    out = {}
    for a in A:
        for b in B:
            near = np.all((lo[a] - REPORT_CAP <= hi[b]) & (lo[b] - REPORT_CAP <= hi[a]))
            out[(a, b)] = S.dist(a, b, REPORT_CAP) if near else REPORT_CAP
    return out


def _cover_behind_hinge(S: _Screen, q0: np.ndarray, joint: str, cover: str) -> dict | None:
    """How far a hinged cover (the visual mesh ``cover`` on ``joint``'s body) reaches behind its hinge at home, along
    the parent body's forward axis, and how its part behind the hinge moves at the joint's open limit."""
    m, d = S.m, S.d
    j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, joint)
    g = S.gid(cover)
    if j < 0 or g < 0 or m.geom_type[g] != _GT.mjGEOM_MESH:
        return None
    adr, hi = int(m.jnt_qposadr[j]), float(m.jnt_range[j][1])

    def pts():
        R = d.geom_xmat[g].reshape(3, 3)
        return d.geom_xpos[g] + S.verts(g) @ R.T

    S.set_q(q0)
    anchor = d.xanchor[j].copy()
    fwd = d.xmat[int(m.body_parentid[m.jnt_bodyid[j]])].reshape(3, 3)[:, 0].copy()
    P0 = pts()
    behind = (anchor - P0) @ fwd
    sel = behind > 0.0
    q = q0.copy()
    q[adr] = hi
    S.set_q(q)
    P1 = pts()
    S.set_q(q0)
    if not sel.any():
        return dict(cover=cover, joint=joint, extent_behind_hinge_m=0.0)
    disp = P1[sel] - P0[sel]
    return dict(cover=cover, joint=joint, hinge_world_m=anchor.tolist(), extent_behind_hinge_m=float(behind.max()),
                cover_extent_m=float(np.ptp(P0 @ fwd)), open_limit_rad=hi,
                rise_at_open_limit_m=float(disp[:, 2].max()),
                displacement_at_open_limit_m=float(np.linalg.norm(disp, axis=1).max()),
                note="the part of the cover behind the hinge swings up and back as the joint opens")


def seam_sweep(xml_path, joints=SEAM_JOINTS, q_home: np.ndarray | None = None, step: float = SWEEP_STEP,
               top: int = 5) -> dict:
    """``SEAM_RULE``: each of ``joints`` the model has (a limited hinge), swept alone from home (``q_home``, default the
    model's ``home`` key) to each limit, every screened geom of its moving body against every one of its parent's.
    Per joint and direction: the first new contact (panel-level pairs apart at home that interfere) with its pairs, and
    the home overlaps that deepen by more than ``SEAM_DEEPEN_M`` (the ``top`` deepest, in mm, with where along the
    sweep).  On a model with a jaw, ``jaw.cover_behind_hinge``: how far today's jaw cover reaches behind the hinge."""
    S = _Screen(xml_path)
    m = S.m
    q0 = (m.key_qpos[0] if q_home is None else np.asarray(q_home, float)).copy()
    out = dict(rule=SEAM_RULE, deepen_threshold_mm=1e3 * SEAM_DEEPEN_M, step_rad=step, joints={})
    for jn in joints:
        jj = S.jnt(jn)
        if jj is None:
            continue
        _, adr, lo, hi = jj
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jn)
        grp = int(S.grp[m.jnt_bodyid[jid]])
        par = int(S.gpar[grp])
        A = [int(g) for g in S.screened if S.geom_grp[g] == grp]
        B = [int(g) for g in S.screened if S.geom_grp[g] == par]
        S.set_q(q0)
        home = _seam_distances(S, A, B)
        pkey = {k: (_panel_key(S.gname[k[0]]), _panel_key(S.gname[k[1]])) for k in home}
        home_panels = {pkey[k] for k, v in home.items() if v < 0.0}
        rec = dict(moving=S.bname[grp], parent=S.bname[par], geoms=[len(A), len(B)], range=[lo, hi],
                   home=float(q0[adr]), pairs_overlapping_at_home=int(sum(v < 0.0 for v in home.values())),
                   panel_pairs_overlapping_at_home=len(home_panels), directions={})
        for end, lim in (("min", lo), ("max", hi)):
            span = lim - float(q0[adr])
            if abs(span) < 1e-12:
                rec["directions"][end] = dict(limit=lim, swept=False)
                continue
            n = max(1, int(np.ceil(abs(span) / step - 1e-9)))
            first, deepen = {}, {}
            for v in np.linspace(float(q0[adr]), lim, n + 1)[1:]:
                q = q0.copy()
                q[adr] = v
                S.set_q(q)
                for k, dv in _seam_distances(S, A, B).items():
                    pk = pkey[k]
                    if dv < 0.0 and pk not in home_panels and pk not in first:
                        first[pk] = (float(v), float(dv))
                    grow = home[k] - dv
                    if home[k] < 0.0 and grow > deepen.get(pk, (0.0, None, None))[0]:
                        deepen[pk] = (float(grow), float(v), float(dv))
            S.set_q(q0)
            new = sorted(first.items(), key=lambda kv: abs(kv[1][0] - float(q0[adr])))
            deep = sorted(((pk, g) for pk, g in deepen.items() if g[0] > SEAM_DEEPEN_M), key=lambda kv: -kv[1][0])
            rec["directions"][end] = dict(
                limit=lim, swept=True, poses=n,
                first_new_contact=None if not new else dict(at=new[0][1][0], pair=list(new[0][0])),
                new_contacts=[dict(pair=list(pk), first_at=v, depth_mm=_mm(-dv) if dv < 0 else 0.0)
                              for pk, (v, dv) in new[:top]], new_contact_pairs=len(new),
                deepening=[dict(pair=list(pk), deepens_mm=_mm(g), at=v, depth_mm=_mm(-dv)) for pk, (g, v, dv)
                           in deep[:top]], deepening_pairs=len(deep),
                deepens_mm_max=_mm(deep[0][1][0]) if deep else 0.0)
        swept = [r for r in rec["directions"].values() if r.get("swept")]
        rec["new_contact"] = any(r["first_new_contact"] for r in swept)
        rec["deepens_mm_max"] = max((r["deepens_mm_max"] for r in swept), default=0.0)
        if jn == "jaw":
            rec["cover_behind_hinge"] = _cover_behind_hinge(S, q0, jn, "cover_jaw")
        out["joints"][jn] = rec
    out["model"] = _model_name(xml_path)
    return out


def gated_margins(margins: dict) -> list:
    """(joint, "min" or "max", margin) of every margin in ``joint_range_margins`` that the ``joint_range_margin`` gate
    reads: all but the ends that a joint rests on by design (``REST_ON_STOP``)."""
    return [(jn, end, v) for jn, r in margins.items() for end, v in (("min", r.get("margin_min")),
                                                                     ("max", r.get("margin_max")))
            if v is not None and r.get("rests_on_stop") != end]


def _range_margins(S: _Screen, sweeps: list, gait_trajs: dict) -> dict:
    """Per swept joint: the walks' excursion against the usable range (first new interference with the
    trunk or the same leg, else the joint limit), in rad.  Positive margins mean the walks stay inside."""
    usable = _joint_ranges(sweeps)
    out = {}
    for jn, rec in usable.items():
        j = S.jnt(jn)
        if j is None or not gait_trajs:
            continue
        adr = j[1]
        vals = np.concatenate([np.asarray(tr["q"], float)[:, adr] for tr in gait_trajs.values()])
        # a limit set by the other leg (e.g. the two feet's toes meeting in a single-joint sweep from home) is
        # not a range limit while walking: the other leg moves, and leg-to-leg contact is checked on every
        # walk frame (and by the swing lateral gap); such limits fall back to the joint limit here
        lim = []
        for i in (0, 1):
            fni, by = rec["first_new_interference"][i], rec["limited_by"][i]
            inter_leg = bool(by) and any("left_" in g for g in by) and any("right_" in g for g in by)
            lim.append(rec["limits"][i] if (fni is None or inter_leg) else fni)
        lo, hi = lim
        m_lo = float(vals.min() - lo) if lo is not None else None
        m_hi = float(hi - vals.max()) if hi is not None else None
        out[jn] = dict(walk_min=round(float(vals.min()), 4), walk_max=round(float(vals.max()), 4),
                       usable=[lo, hi], margin_min=None if m_lo is None else round(m_lo, 4),
                       margin_max=None if m_hi is None else round(m_hi, 4))
        if jn in REST_ON_STOP:  # its margin at that end is not gated (REST_ON_STOP)
            out[jn]["rests_on_stop"] = REST_ON_STOP[jn]
    return out


def _summary(report, H, tail_gate, tail_gate_key, swing, battery, gates: str = "r1") -> dict:
    inter = report["interference"]
    counts = {k: _compact_counts(v["counts"]) for k, v in inter.items()}
    operating = [k for k in inter if k == "home" or k.startswith("gait_") or k == "tail_limits_in_gait"]
    sw = {}
    for g, v in swing.items():
        sw[g] = dict(mid_swing_min_mm=v["mid_swing_min_mm"], mid_swing_min_frac_H=v["mid_swing_min_frac_H"],
                     apex_min_mm=min((v[s]["apex_mm"] for s in SIDES if v[s]["apex_mm"] is not None), default=None),
                     full_swing_min_mm=min((v[s]["full_swing_min_mm"] for s in SIDES
                                            if v[s]["full_swing_min_mm"] is not None), default=None),
                     lateral_gap_mm=v["lateral_gap"]["min_mm"] if v["lateral_gap"] else None,
                     lateral_gap_geoms=v["lateral_gap"]["geoms"] if v["lateral_gap"] else None)
    swing_vals = [v["mid_swing_min_frac_H"] for v in sw.values() if v["mid_swing_min_frac_H"] is not None]
    nominal = "gait_nominal" in counts
    passes = dict(
        interference_free_home=counts["home"]["interference_pairs"] == 0,
        interference_free_nominal_gait=(counts["gait_nominal"]["interference_pairs"] == 0) if nominal else None,
        swing_clearance=bool(min(swing_vals) >= SWING_MIN_FRAC) if swing_vals else None,
        tail_clearance=bool(tail_gate / H >= TAIL_MIN_FRAC) if np.isfinite(tail_gate) else None,
        battery_path=bool(battery.get("clear_directions")) if battery.get("found", True) else None,
    )
    if gates == "r2":
        brisk = "gait_brisk" in counts
        margins = [v for _, _, v in gated_margins(report.get("joint_range_margins", {}))]
        passes.update(
            interference_free_brisk_gait=(counts["gait_brisk"]["interference_pairs"] == 0) if brisk else None,
            component_overlap_free=not report["same_body_hardware_at_home"]["overlaps"],
            joint_range_margin=bool(min(margins) >= RANGE_MARGIN_RAD) if margins else None,
        )
    passes["all"] = all(v is True for k, v in passes.items() if not (k == "battery_path" and v is None))
    hh_home = [r for r in inter["home"]["pairs"] if r["class"] == "hardware-hardware" and r["min_mm"] < 0]
    return dict(
        model=report["model"], H_m=report["H_m"],
        thresholds=dict(interference="d < 0", tight_mm=TIGHT * 1000, swing=f">= {SWING_MIN_FRAC} H mid-swing",
                        tail=f">= {TAIL_MIN_FRAC} H", battery=">= 1 clear direction"),
        counts=counts,
        worst_pairs=_worst(inter, operating, 10),
        worst_pairs_joint_extremes=_worst(inter, ["joint_extremes"], 5),
        joint_range_usable=_joint_ranges(inter["joint_extremes"]["sweeps"]),
        hardware_interference_at_home=[dict(geoms=r["geoms"], min_mm=r["min_mm"]) for r in hh_home],
        swing_clearance=sw,
        tail_min_height=dict(min_mm=_mm(tail_gate, 1),
                             frac_H=round(tail_gate / H, 4) if np.isfinite(tail_gate) else None,
                             where=tail_gate_key,
                             with_tail_limits_in_gait_frac_H=report["tail"]["records"].get(
                                 "tail_limits_in_gait", {}).get("frac_H")),
        battery_clear_directions=battery.get("clear_directions", []),
        battery_clear_ignoring_home_overlaps=battery.get("clear_ignoring_home_overlaps", []),
        battery_overlapping_at_home=battery.get("overlapping_at_home", []),
        same_body_component_overlaps=[dict(geoms=r["geoms"], min_mm=r["min_mm"])
                                      for r in report["same_body_hardware_at_home"]["overlaps"]],
        joint_range_margins=report.get("joint_range_margins", {}),
        gate_set=gates,
        **{"pass": passes},
    )


def _clean(x):
    """Recursively convert numpy types and non-finite floats for JSON."""
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return [_clean(v) for v in x.tolist()]
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return float(x) if np.isfinite(x) else None
    return x


# ================================================================================================
# Diagnostic plot
# ================================================================================================
def _plot(series: dict, title: str, out_png):
    """Small multiples (one y-axis each) of the per-frame screening quantities for each gait."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    surf, ink, ink2, muted, grid, axis_c = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
    slots = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # fixed categorical order (reference palette)
    panels = [("min_mm", f"Closest approach between bodies, mm (capped at {REPORT_CAP * 1000:.0f})",
               [(0.0, "interference"), (TIGHT * 1000, f"tight {TIGHT * 1000:.0f} mm")]),
              ("swing_frac", "Swing-foot pad bottom above floor, fraction of H",
               [(SWING_MIN_FRAC, f"{SWING_MIN_FRAC} H")]),
              ("tail_frac", "Tail lowest point above floor, fraction of H", [(TAIL_MIN_FRAC, f"{TAIL_MIN_FRAC} H")])]
    fig, axs = plt.subplots(3, 1, figsize=(7.2, 6.6), dpi=110, sharex=True)
    fig.patch.set_facecolor(surf)
    for ax, (key, ttl, refs) in zip(axs, panels):
        ax.set_facecolor(surf)
        for c, (name, s) in zip(slots, series.items()):
            ax.plot(s["phase"], s[key], color=c, lw=1.6, solid_capstyle="round", solid_joinstyle="round",
                    label=f"{name} gait")
        for y, lab in refs:
            ax.axhline(y, color=muted, lw=0.8, zorder=1)
            ax.annotate(lab, xy=(1.0, y), xycoords=("axes fraction", "data"), xytext=(-4, 3),
                        textcoords="offset points",
                        ha="right", va="bottom", fontsize=7.5, color=ink2)
        ax.set_title(ttl, loc="left", fontsize=8.8, color=ink)
        ax.grid(True, axis="y", color=grid, lw=0.6)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(axis_c)
        ax.tick_params(colors=axis_c, labelcolor=ink2, labelsize=7.5)
    axs[-1].set_xlabel("Gait phase (fraction of one stride)", fontsize=8, color=ink2)
    axs[-1].set_xlim(0, 1)
    h, lab = axs[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper right", ncol=len(lab), frameon=False, fontsize=8, labelcolor=ink2)
    fig.suptitle(f"{title.removeprefix('trex_')}: clearance along the reference gaits", x=0.015, ha="left",
                 fontsize=10, color=ink,
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(out_png, facecolor=surf)
    plt.close(fig)


# ================================================================================================
# CLI
# ================================================================================================
def main(argv=None):
    ap = argparse.ArgumentParser(description="Clearance and interference screening (see module docstring).")
    ap.add_argument("ids", nargs="+", help="candidate ids, e.g. 60-M-S-10")
    ap.add_argument("--out", required=True, help="output directory (model, ledger, <id>_clearance.json/.png)")
    ap.add_argument("--stride", type=int, default=1, help="evaluate every n-th gait frame (default 1: all)")
    ap.add_argument("--substeps", type=int, default=0,
                    help="poses interpolated between gait frames for the interference check (default 0)")
    ap.add_argument("--no-png", action="store_true", help="skip the diagnostic plot")
    a = ap.parse_args(argv)
    from . import gait, pipeline, sizing
    from .params import parse

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for cid in a.ids:
        t0 = time.perf_counter()
        cand = parse(cid)
        res = pipeline.build_candidate(cand, out)
        t1 = time.perf_counter()
        M, q_home, H = res["model"], res["q_home"], res["design"].sk.H
        trajs = {"nominal": gait.generate(M, q_home, H, sizing.NOMINAL),
                 "brisk": gait.generate(M, q_home, H, sizing.BRISK)}
        t2 = time.perf_counter()
        png = None if a.no_png else out / f"{cand.id}_clearance.png"
        rep = screen(res["summary"]["xml"], q_home, trajs, out_png=png, H=H, frame_stride=a.stride,
                     substeps=a.substeps)
        t3 = time.perf_counter()
        rep["candidate"] = cand.id
        rep["gait_info"] = {k: _clean(dict(frames=len(tr["q"]), ik_ok=bool(tr.get("ik_ok", True)),
                                           ik_fail_frames=len(tr.get("ik_fail_frames", [])),
                                           **{kk: tr["info"][kk] for kk in ("speed", "stride", "period", "lift", "duty")
                                              if kk in tr.get("info", {})}))
                            for k, tr in trajs.items()}
        rep["timing_s"] = dict(rep["timing_s"], build=round(t1 - t0, 3), gait_generation=round(t2 - t1, 3),
                               screen=round(t3 - t2, 3), cli_total=round(t3 - t0, 3))
        path = out / f"{cand.id}_clearance.json"
        path.write_text(json.dumps(rep, indent=1))
        s = rep["summary"]
        c = s["counts"]
        print(f"{cand.id}: home {c['home']['interference_pairs']} interf/{c['home']['tight_pairs']} tight; "
              f"nominal {c.get('gait_nominal', {}).get('interference_pairs')} interf; "
              f"swing {s['swing_clearance'].get('nominal', {}).get('mid_swing_min_frac_H')} H; "
              f"tail {s['tail_min_height']['frac_H']} H; battery {s['battery_clear_directions']}; "
              f"pass {s['pass']['all']}; {t3 - t0:.1f} s -> {path}")


if __name__ == "__main__":
    main()
