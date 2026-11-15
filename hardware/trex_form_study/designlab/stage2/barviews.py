"""Views of the bar's episodes: the room a turn needs and video clips (decision log P35).

Everything here is a *view* of committed records and their traces, or of a frozen reference.  It judges nothing,
changes no record and no verdict, adopts nothing and runs no episode.  Two words are kept apart everywhere:
**planned** is read from a frozen reference played kinematically (no physics), **simulated** is read from an
episode's trace.  No number mixes the two, and every frame of every clip says which it is.

The pose of a trace row (``pose_of``)
-------------------------------------
A trace (``panels.TRACE_RULE``, ``bar.RULES["trace"]``) holds, per 50 Hz row, the joint angles, the trunk's ``roll``,
``pitch`` and ``yaw``, ``root_x``, ``root_y`` and both feet's lowest points; it holds neither the root's height nor
the CoM's.  The attitude columns are ``metrics.Recorder.update``'s: ``roll, pitch = control._roll_pitch(R R_home^T)``,
the roll and pitch of ``R R_home^T = Rz(a) Ry(pitch) Rx(roll)`` (the trunk's rotation ``R`` relative to its home
attitude ``R_home``; ``a`` is not recorded), and ``yaw = control._yaw(R) = atan2(R[1, 0], R[0, 0])``, the heading of
the trunk's own x axis.  So ``R = Rz(a) Ry(pitch) Rx(roll) R_home`` with ``a = yaw - atan2(w[1], w[0])``, ``w =
Ry(pitch) Rx(roll) R_home[:, 0]``: exact.  The root's height: the free joint is set from ``root_x``, ``root_y``, that
attitude and a height of zero, the kinematics run, and the root is lifted until each foot's lowest point
(``metrics.foot_lowest``) is the trace's ``low0`` / ``low1`` (the mean of the two feet's answers; their difference is
returned as a check).  The brief asked for the CoM's height; a trace has none, the feet's low points give the same
shift exactly.

One thing is not exact on a simulated trace.  ``metrics.Recorder.update`` runs after the physics step: its kinematic
columns describe the state at the step's start (``bar.RULES["convention"]``), its joint angles are ``qpos`` after
the step, one physics step (2 ms) later.  ``pose_of`` moves the joint angles back by that step (``joint_lag_steps=1``,
the default): a cubic spline through the 50 Hz joint angles at the instants they were taken, read at the rows' own
instants (``POSE_BOUND``; a straight line between the rows left up to 0.27 mm).  It returns how far the rebuilt
feet and CoM lie from the trace's own ``foot*_x/y`` and ``com_x/y`` columns (``residual_m``).  This is a measured
bound, not an exact rebuild: where a foot lands or the robot hits the floor the joint speeds jump between two rows
and no curve through the 50 Hz angles follows them.  A kinematic play's recorder (``bar.reference_recorder``) writes
both at one instant: ``joint_lag_steps=0``, and the round trip is exact to the arrays' precision.

``barviews turnpath``: the room a turn needs
--------------------------------------------
Per set and turn reference, ``views/turnpath/<set>/turnpath.json``, ``TURNPATH.md`` and a top-view figure per
condition, with a **planned** block (the frozen reference played at the bar's commanded clock) and a **simulated**
block (every kept trace of every record of that turn that stayed up), never merged.  ``TURN_RULES`` states each
reading; it is copied into the file.

``barviews clips``: video
-------------------------
Simulated episodes rendered from their traces (``SIM_LABEL`` on every frame), or the frozen reference alone
(``--planned``; ``PLAN_LABEL``).  Which episode: ``SELECTION_RULE``.  An episode that fell before its condition's
own move began (the turn, the stop or the push was never commanded; the walk at its speed never began) says so on
every frame (``CLIP_RULES["before"]``).  Media go outside git (``--media``, default ``logs/barviews/``);
``views/clips.json`` lists them.

Run from hardware/trex_form_study, one process, through ``bash tools/lowprio.sh``::

    python -m designlab.stage2.barviews turnpath --set V0 [--run DIR] [--out DIR] [--figure-row R0@10ms]
    python -m designlab.stage2.barviews clips F25F9cl:walk/v0.4:R2/42@10ms ... [--run DIR] [--media DIR] [--out DIR]
    python -m designlab.stage2.barviews clips --planned V0:turn/q90_left F25F9cl:turn/q90_left
    python -m designlab.stage2.barviews clips --all-gated --row R2/42@10ms [--set F25F9cl] [--list]
    python -m designlab.stage2.barviews posecheck F25F9cl:walk/v0.4:R2/42@10ms [--run DIR]

``--run`` is the run directory the records are read from (default: this checkout's
``stage2/runs/2026-11-15-75cm``); nothing is ever written into it except under its own ``views/``.  ``clips`` needs
``MUJOCO_GL=osmesa`` (and ``LP_NUM_THREADS=1``); the other commands need no renderer.
"""

from __future__ import annotations

import os

if __name__ == "__main__":  # before numpy: one core; one llvmpipe thread for the software renderer
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "LP_NUM_THREADS"):
        os.environ.setdefault(_v, "1")
    for _v in ("MESA_SHADER_CACHE_DISABLE", "MESA_GLSL_CACHE_DISABLE"):  # (``renderer_env``, before any GL library)
        os.environ.setdefault(_v, "true")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import textwrap  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from ..stage1 import control, metrics, plant  # noqa: E402
from ..stage1.refs import Reference  # noqa: E402
from . import bar, s2io, setio  # noqa: E402

SCHEMA_TURNPATH = "barviews-turnpath-1"
SCHEMA_CLIPS = "barviews-clips-1"
BODY = "75-M-H-10"
#: The run directory the records are read from by default, and where this module writes (its ``views/`` only).
RUN = s2io.STUDY / "stage2" / "runs" / "2026-11-15-75cm"
VIEWS = "views"
MEDIA = Path("logs") / "barviews"
SETS = bar.PREREGISTERED_SETS
#: The labels on every frame's first line: exactly one of them.
SIM_LABEL = "SIMULATION (physics)"
PLAN_LABEL = "PLANNED MOTION (kinematic, no physics)"
#: The sets, the bodies and the floor grips in plain words.
SET_WORDS = {"V0": "today's walk", "F25F9": "the pull request's walk", "F25F9cl": "the candidate"}
BODY_WORDS = {"60-N-T-10": "60 cm Tapered", "75-M-H-10": "75 cm Hybrid"}
GRIP_HARD = 0.9  # the hard floor as compiled (panels.CONDITIONS_RULE): a condition without a friction of its own
#: Geom groups of the robot's surface (``render.Scene.bounds``: pads, covers, hardware); group 3 holds proxies.
SURFACE_GROUPS = (0, 1, 2)
#: The steady part of a turn: the rows whose heading has turned between these shares of the commanded angle.
STEADY = (1.0 / 6.0, 5.0 / 6.0)
STEADY_MIN_ROWS = 8
#: The turn itself is done at the first row at which the trunk has turned its commanded angle to within this (deg).
TURNED_WITHIN_DEG = 0.5
#: The figure's simulated paths come from one record on the hard floor, headline block: the first of these rows (the
#: judged delay's, in the chain's order: the two fitted controllers, then the open-loop row, which gives no
#: verdict; then the same three at each shorter servo delay the chain runs, the nearest to the judged delay first,
#: down to none) whose record has an episode that stayed up and can be read; the caption names it, its delay, and
#: the rows before it.  ``--figure-row`` names one row instead.
FIGURE_DELAYS_MS = (10, 8, 6, 4, 2, 0)
FIGURE_ROWS = tuple(f"{ctl}@{ms}ms" for ms in FIGURE_DELAYS_MS for ctl in ("R2/42", "R2/43", "R0"))
#: How close ``pose_of`` comes to the plant's own body positions on an episode's trace (module docstring): measured,
#: not exact.  ``tests/test_stage2_barviews.py`` checks the first on a Stage 1 smoke episode run with the plant's
#: positions kept (STAGE2_SLOW=1); ``barviews posecheck`` gives the second for any record.
POSE_BOUND = (
    "the pose of an episode's trace row is a measured bound, not exact: the attitude and the root's place are exact, "
    "the joint angles are a cubic spline through the 50 Hz angles read one physics step earlier.  On a Stage 1 "
    "smoke episode run with the plant's own body positions kept (75-M-H-10, turn/walking_0.35_left, R2/42@0ms, "
    "seed 3042, 1461 rows) every body origin came back within 0.08 mm on every row but the first and within 0.11 mm "
    "on the first (the start's transient lies before the first joint sample).  On the chain's traces there is no "
    "such truth, only the trace's own foot and CoM columns: over the 1120 episodes that stayed up in the turn "
    "records of V0 and F25F9cl (67 records, 15 Nov) the rebuilt feet lay up to 0.73 mm from them in plan (the "
    "median of the records' largest: 0.19 mm) and the CoM up to 0.04 mm, and the two feet's answers for the "
    "root's height differed by up to 1.02 mm (1.35 mm in an episode that fell): the largest on the rows next to "
    "a touchdown, a slip or a fall, where the joint speeds jump between two rows.  The brief's 0.1 mm is "
    "therefore met on a kinematic play and, but for the first row, on the smoke episode, not on every row of the "
    "chain's traces (each clip's manifest entry and each turn record's block give their own largest distances: "
    "pose_residual_m, pose_check_worst; barviews posecheck prints them for any record)")
#: Clips: size, rate, size cap, lead and tail around the move, the command caption's length, the walk's length.
FPS = 30
MAX_BYTES = 3_000_000
LEAD_S = 1.0
TAIL_S = 1.0
CAPTION_S = 1.5
WALK_CLIP_S = 10.0
#: A top view's eye is never lower than this (m): the picture then holds about 1.5 m along the walking line.
TOP_MIN_HEIGHT_M = 4.0
#: The headlight's diffuse part in a top view (the model's own 0.55 washes the floor out when it shines straight down).
TOP_HEADLIGHT_DIFFUSE = 0.2

TURN_RULES = dict(
    words="planned: read from the frozen reference played kinematically at the bar's commanded clock (no physics). "
          "simulated: read from an episode's trace. No number mixes the two.",
    clock=f"the bar plays a reference {bar.RATE:g} times faster than the reference's own clock "
          "(bar.RULES['clocks']); every time here is on the bar's clock (the seconds of an episode), so a planned "
          f"speed or rate on the bar's clock is the reference's own times {bar.RATE:g}; lengths and radii do not "
          "change with the clock",
    frame="the corner frame of a turn: origin at the root (the trunk's origin) at the command instant; 'along' is "
          "the commanded heading (the home heading, the bar's psi_cmd), the incoming walking line; 'sideways' is "
          "square to it, positive toward the side of the turn",
    command="the command instant t_c: planned, the reference clock's time of the move's first frame over the "
            "commanded clock rate; simulated, the episode record's command_s (never recomputed)",
    move_end="the end of the move: the time on the bar's clock at which the reference clock reaches the end of the "
             "turn's own file (a trace's t_ref column); a trace that ends before it is read to its end and says so. "
             "A turn's file goes on after the turn: it holds the walk out of the turn too, and the files of "
             "different turns and sets differ in length",
    ends="every largest and smallest value over a span of time is taken over the 50 Hz rows in the span and the "
         "span's two ends themselves (read between the rows on either side, linearly): a value 'at the end' is "
         "then never beyond the largest",
    turn_itself=f"the turn itself: from the command to the first row at which the trunk has turned the commanded "
                f"angle to within {TURNED_WITHIN_DEG:g} degrees.  Given: the seconds it took, where the root is then "
                "(along its incoming line and sideways), the box of the whole robot (every surface point) over "
                "those rows, and how far the root is then from its place at the end of the move (the walk after "
                "the turn that the corner box to the move's end includes).  A path whose trunk never turns that "
                "far has none of these",
    heading="the trunk's heading: the trace's yaw column (control._yaw of the trunk's rotation), unwrapped, as "
            "bar's turn reading takes it; 'turned' is its change from the commanded heading, positive toward the "
            "side of the turn",
    steady=f"the steady part of the turn: from the first row at which the trunk has turned {STEADY[0]:.4g} of the "
           f"commanded angle to the first row at which it has turned {STEADY[1]:.4g} of it (15 to 75 degrees of a "
           "90 degree turn); a path that never gets there 'stops short' and has no fitted radius",
    radius_commanded="the path radius commanded: the walk's path speed over the turn's rate (v / omega), both from "
                     "the frozen turn's entry",
    radius_fitted="the radius fitted: the circle of the root's path over the steady part, with the trunk's heading "
                  "as the place on the arc: least squares of root = centre + R x n(heading) over the steady rows, "
                  "n the unit vector square to the heading that points away from the side of the turn; its centre "
                  "is the turn's centre, and the RMS distance of the rows from that model is given beside it.  A "
                  "circle fitted to the positions alone (the algebraic fit, recorded as radius_free_fit_m) reads "
                  "the root's sideways sway of about 3 cm a step as curvature: it gave 0.29 m on V0's planned "
                  "quick turn, whose commanded radius is 0.51 m, where this fit gives 0.50 m",
    corner_box="the corner box to the end of the move: from the command to the end of the move, how far the root "
               "gets along its incoming line (advance) and sideways, at the largest and at the move's end; the same "
               "box for the whole robot (every surface point).  Its sideways side is mostly the walk after the "
               "turn and grows with the length of the move's file: read the room of the turn from 'the turn "
               "itself', not from this box",
    surface="the robot's surface points: per body of the model, the corners of the convex hull of its pads, covers "
            "and hardware (geom groups 0, 1 and 2: mesh vertices, and the corners of a primitive's bounding box), "
            "carried on each row's pose; the snout tip is the head's point farthest forward at home, the tail tip "
            "the tail's farthest back",
    swept_radius="the swept radius: the largest distance in plan from the turn's centre to any surface point over "
                 "the steady part of the turn (where the robot is on its circle); the point that sets it is named.  "
                 "A circle of that radius does not hold the robot through this 90 degree turn: at the command the "
                 "robot still stands along its incoming line and its tail reaches further from the centre.  That "
                 "is the second reading, 'the largest distance from the turn's centre from the command until the "
                 "trunk has turned the commanded angle' (the turn itself), with the point and the instant that "
                 "set it",
    turning_circle="the turning circle: twice the swept radius of the steady part, the diameter of the circle a "
                   "full turn at that rate would sweep about the same centre (a robot that keeps turning; not the "
                   "larger distance at the command)",
    corridor="the corridor: the narrowest L-shaped corridor, two legs of one width meeting square, that holds "
             "every surface point from the command to the end of the move (both ends included).  Its outer wall along the incoming leg "
             "touches the sweep's point farthest to the outside (sideways = Y0), its far wall across the incoming "
             "line touches the sweep's point farthest along (along = X); the inner corner is at (X - w, Y0 + w) "
             "and the width w is the smallest for which no point lies inside that corner's block: w = the largest, "
             "over all points, of min(X - along, sideways - Y0).  It holds the robot up to the end of the move "
             "only: a robot that leaves the move off its heading needs more room further on",
    body_length="lengths are given in metres and in body lengths; a body length is the model's overall length at "
                "its home keyframe along the home heading, from the surface's point farthest back to its point "
                "farthest forward",
    simulated="simulated: every record of the turn under the run directory (each row, delay, floor grip and "
              "block) is listed; an episode that stayed up and whose trace is kept and has the record's sha256 is "
              "read; an episode that fell is counted and listed, not read; the heading at the bar's judged "
              "instant is the record's own (command_s + at_s; the commanded angle + the sign times "
              "heading_error_at_deg)",
    pose=POSE_BOUND,
)
SELECTION_RULE = (
    "which episode a clip shows (stated before any record existed): the block's first seed; if that episode's "
    "outcome (stayed up or fell) is not the majority outcome of the record's episodes, in addition the lowest seed "
    "with the majority outcome whose trace is kept.  An episode that failed to run has no outcome and no clip.  On "
    "a tie the first seed's outcome counts as the majority.  A trace that was not kept, is missing or does not "
    "have the record's sha256 gives no clip; the manifest says why")


class ViewError(s2io.S2Error):
    """A view that cannot be made (a refusal; nothing is written)."""


# ------------------------------------------------------------------------------------------------
# The body's model: surface points, parts, the body length
# ------------------------------------------------------------------------------------------------
def _part(body_name: str) -> str:
    for side in ("left", "right"):
        if body_name.startswith(side + "_"):
            return f"{side} foot" if body_name == f"{side}_foot" else f"{side} leg"
    return {"head": "head", "tail": "tail", "trunk": "trunk"}.get(body_name, body_name)


class BodyModel:
    """What the views need of a body: its Stage 1 plant (``plant.build(body, "hard", settle=False)``; never
    stepped), a private ``MjData`` for kinematics, the surface points per body (``TURN_RULES["surface"]``) and the
    body length."""

    def __init__(self, body: str = BODY):
        from scipy.spatial import ConvexHull, QhullError

        self.name = body
        self.p = p = plant.build(body, "hard", settle=False)
        self.m = m = p.m
        self.d = mujoco.MjData(m)
        self.psi_cmd = float(control._yaw(p._R_home))
        self.dt = float(m.opt.timestep)
        self._boxes, self._caps = metrics._lowest_point_tables(m, p.feet)
        floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        pts: dict[int, list] = {}
        for g in range(m.ngeom):
            if g == floor or m.geom_group[g] not in SURFACE_GROUPS or m.geom_bodyid[g] == 0:
                continue
            if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                mid = m.geom_dataid[g]
                v = np.array(m.mesh_vert[m.mesh_vertadr[mid] : m.mesh_vertadr[mid] + m.mesh_vertnum[mid]], float)
            else:
                if m.geom_rbound[g] <= 0:
                    continue
                c, h = m.geom_aabb[g, :3], m.geom_aabb[g, 3:]
                v = c + h * np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)
            R = np.zeros(9)
            mujoco.mju_quat2Mat(R, m.geom_quat[g])
            pts.setdefault(int(m.geom_bodyid[g]), []).append((g, m.geom_pos[g] + v @ R.reshape(3, 3).T))
        self.bodies, self.local, self.geom_of = [], [], []
        for b, items in sorted(pts.items()):
            allp = np.concatenate([v for _, v in items])
            gid = np.concatenate([np.full(len(v), g) for g, v in items])
            try:
                keep = ConvexHull(allp).vertices
            except QhullError:  # (a flat or tiny set: keep every point)
                keep = np.arange(len(allp))
            self.bodies.append(b)
            self.local.append(allp[keep])
            self.geom_of.append(gid[keep])
        self.count = np.array([len(x) for x in self.local])
        self.body_of = np.repeat(self.bodies, self.count)
        names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in self.bodies]
        self.parts = [_part(n) for n in names]
        geoms = np.concatenate(self.geom_of)
        self.labels = [f"{_part(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(b)))}: "
                       f"{mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, int(g))}"
                       for b, g in zip(self.body_of, geoms, strict=True)]
        # the body length, the snout tip and the tail tip at the home keyframe
        self.q_home = np.array(m.key_qpos[0], float)
        home = self.surface(self.q_home)
        c, s = math.cos(self.psi_cmd), math.sin(self.psi_cmd)
        fwd = home[:, 0] * c + home[:, 1] * s
        part = np.array([lab.split(":")[0] for lab in self.labels])
        self.i_snout = int(np.flatnonzero(part == "head")[np.argmax(fwd[part == "head"])])
        self.i_tail = int(np.flatnonzero(part == "tail")[np.argmin(fwd[part == "tail"])])
        self.labels[self.i_snout] += " (the snout tip)"
        self.labels[self.i_tail] += " (the tail tip)"
        self.length_m = float(fwd.max() - fwd.min())
        self.length_info = dict(
            body_length_m=self.length_m, rule=TURN_RULES["body_length"],
            front=self.labels[int(np.argmax(fwd))], back=self.labels[int(np.argmin(fwd))],
            model=s2io.portable(str(plant.model_files(body)[0])), H_m=float(p.meta["H_m"]),
            surface_points=int(len(self.labels)))

    def kin(self, qpos) -> mujoco.MjData:
        """Kinematics (and the CoM) of the pose ``qpos`` in this model's private data."""
        self.d.qpos[:] = qpos
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_comPos(self.m, self.d)
        return self.d

    def surface(self, qpos=None) -> np.ndarray:
        """The surface points [N, 3] (world) on ``qpos`` (None: on the kinematics last run)."""
        d = self.d if qpos is None else self.kin(qpos)
        return np.concatenate([d.xpos[b] + loc @ d.xmat[b].reshape(3, 3).T
                               for b, loc in zip(self.bodies, self.local, strict=True)])

    def outline(self, qpos) -> list[np.ndarray]:
        """Per body, the plan outline of its surface points on ``qpos`` (a closed polygon [k, 2], world xy)."""
        from scipy.spatial import ConvexHull

        pts = self.surface(qpos)[:, :2]
        out, i = [], 0
        for n in self.count:
            xy = pts[i : i + n]
            i += n
            out.append(xy[ConvexHull(xy).vertices] if n >= 3 else xy)
        return out


_BODIES: dict = {}


def body_model(body: str = BODY) -> BodyModel:
    if body not in _BODIES:
        _BODIES[body] = BodyModel(body)
    return _BODIES[body]


# ------------------------------------------------------------------------------------------------
# The pose of a trace row
# ------------------------------------------------------------------------------------------------
def attitude(roll: float, pitch: float, yaw: float, R_home) -> np.ndarray:
    """The trunk's rotation from a trace row's ``roll``, ``pitch`` and ``yaw`` (module docstring): ``Rz(a) Ry(pitch)
    Rx(roll) R_home`` with ``a`` set so that ``atan2(R[1, 0], R[0, 0]) = yaw``."""
    cr, sr, cp, sp = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch)
    Ryx = np.array([[cp, sp * sr, sp * cr], [0.0, cr, -sr], [-sp, cp * sr, cp * cr]])
    M = Ryx @ np.asarray(R_home, float).reshape(3, 3)
    a = yaw - math.atan2(M[1, 0], M[0, 0])
    ca, sa = math.cos(a), math.sin(a)
    return np.array([[ca, -sa, 0.0], [sa, ca, 0.0], [0.0, 0.0, 1.0]]) @ M


def _col(trace, name: str) -> np.ndarray:
    cols = [str(c) for c in trace["cols"]]
    return np.asarray(trace["rows"], float)[:, cols.index(name)]


def joints_at_rows(t, Q, dt: float, lag_steps: int) -> np.ndarray:
    """The joint angles at the rows' own times ``t``, from angles ``Q`` recorded ``lag_steps`` physics steps after
    them: the cubic spline (not-a-knot) through ``Q`` at the instants ``t + lag_steps dt``, read at ``t`` (the
    first row lies ``lag_steps`` steps before the first sample: the spline's first piece carried back).  Fewer than
    four rows: a straight line between a row and the one before it."""
    Q = np.asarray(Q, float)
    if not lag_steps or len(Q) < 2:
        return Q.copy()
    t = np.asarray(t, float)
    if len(Q) >= 4:
        from scipy.interpolate import CubicSpline

        return np.asarray(CubicSpline(t + lag_steps * dt, Q, axis=0)(t), float)
    step = np.diff(t)
    frac = (lag_steps * dt) / np.r_[step[0], step]
    dQ = np.vstack([Q[1] - Q[0], np.diff(Q, axis=0)])
    return Q - frac[:, None] * dQ


def pose_of(trace, model: BodyModel, joint_lag_steps: int = 1) -> dict:
    """The full pose at each row of ``trace`` (a bar or Stage 1 trace: ``t``, ``rows``, ``cols``, ``q``, and ``dt``
    where it has one) on ``model``; the convention of the attitude columns and the root's height are in the module
    docstring.  ``joint_lag_steps``: 1 for an episode's trace (the default), 0 for a kinematic play's recorder.

    Returns ``dict(t, t_ref, qpos [rows, nq], yaw (unwrapped), root_xy [rows, 2], height_spread_m [rows], residual_m)``:
    ``height_spread_m`` is the difference between the two feet's answers for the root's height, ``residual_m`` the
    largest distance of the rebuilt feet (body origins), sole heights and CoM from the trace's own columns."""
    p, m = model.p, model.m
    t = np.asarray(trace["t"], float)
    n = len(t)
    dt = float(trace["dt"]) if "dt" in trace else model.dt
    Q = joints_at_rows(t, trace["q"], dt, joint_lag_steps)
    if Q.shape != (n, len(p.qadr)):
        raise ViewError(f"the trace holds {Q.shape} joint angles, the model has {len(p.qadr)} joints on {n} rows")
    roll, pitch, yaw = _col(trace, "roll"), _col(trace, "pitch"), _col(trace, "yaw")
    rx, ry = _col(trace, "root_x"), _col(trace, "root_y")
    low = np.column_stack([_col(trace, "low0"), _col(trace, "low1")])
    feet = np.column_stack([_col(trace, c) for c in ("foot0_x", "foot0_y", "foot1_x", "foot1_y")])
    com = np.column_stack([_col(trace, "com_x"), _col(trace, "com_y")])
    qpos = np.zeros((n, m.nq))
    spread = np.zeros(n)
    res = dict(feet_xy=0.0, com_xy=0.0)
    quat = np.zeros(4)
    d = model.d
    for i in range(n):
        R = attitude(roll[i], pitch[i], yaw[i], p._R_home)
        mujoco.mju_mat2Quat(quat, R.ravel())
        q = qpos[i]
        q[0], q[1], q[2] = rx[i], ry[i], 0.0
        q[3:7] = quat
        q[p.qadr] = Q[i]
        model.kin(q)
        z = low[i] + p.z_floor - metrics.foot_lowest(m, d, model._boxes, model._caps)
        q[2] = float(np.mean(z))
        spread[i] = abs(float(z[0] - z[1]))
        f = np.r_[d.xpos[p.feet[0]][:2], d.xpos[p.feet[1]][:2]]
        res["feet_xy"] = max(res["feet_xy"], float(np.max(np.abs(f - feet[i]))))
        res["com_xy"] = max(res["com_xy"], float(np.max(np.abs(d.subtree_com[p.trunk][:2] - com[i]))))
    return dict(t=t, t_ref=np.asarray(trace["t_ref"], float), qpos=qpos, yaw=np.unwrap(yaw),
                root_xy=np.column_stack([rx, ry]), height_spread_m=spread,
                residual_m=dict(res, feet_height=float(spread.max()) if n else 0.0), joint_lag_steps=int(joint_lag_steps))


def q_at(pose: dict, t: float) -> np.ndarray:
    """The pose at time ``t`` between the 50 Hz rows, as ``clips.Playback`` interpolates: root position and joint
    angles linear, the root's quaternion by normalised linear interpolation; held at the first and the last row."""
    T, Q = pose["t"], pose["qpos"]
    if t <= T[0]:
        return Q[0].copy()
    if t >= T[-1]:
        return Q[-1].copy()
    k = int(np.searchsorted(T, t, side="right")) - 1
    a = (t - T[k]) / (T[k + 1] - T[k])
    if a <= 0.0:  # (a row's own time: that row's pose)
        return Q[k].copy()
    q0, q1 = Q[k], Q[k + 1]
    q = (1.0 - a) * q0 + a * q1
    r0, r1 = q0[3:7], q1[3:7]
    r = (1.0 - a) * r0 + a * (r1 if np.dot(r0, r1) >= 0 else -r1)
    q[3:7] = r / np.linalg.norm(r)
    return q


# ------------------------------------------------------------------------------------------------
# A turn's readings (one function for the planned and the simulated path)
# ------------------------------------------------------------------------------------------------
def fit_circle(xy) -> tuple[np.ndarray, float, float]:
    """(centre, radius, RMS distance of the points from the circle) of the algebraic least-squares circle through
    ``xy`` [n, 2]: the fit to the positions alone, recorded beside ``fit_turn_circle``'s."""
    xy = np.asarray(xy, float)
    A = np.column_stack([2.0 * xy, np.ones(len(xy))])
    sol, *_ = np.linalg.lstsq(A, np.sum(xy * xy, axis=1), rcond=None)
    c = sol[:2]
    r = math.sqrt(max(float(sol[2] + c @ c), 0.0))
    return c, r, float(np.sqrt(np.mean((np.linalg.norm(xy - c, axis=1) - r) ** 2)))


def fit_turn_circle(xy, heading, sign: float) -> tuple[np.ndarray, float, float]:
    """(centre, radius, RMS residual) of ``TURN_RULES["radius_fitted"]``: least squares of ``xy = centre + R
    n(heading)``, ``n = sign (sin, -cos)`` (for a left turn the centre lies to the left of the heading)."""
    xy, psi = np.asarray(xy, float), np.asarray(heading, float)
    n = len(xy)
    A = np.zeros((2 * n, 3))
    A[0::2, 0] = A[1::2, 1] = 1.0
    A[0::2, 2], A[1::2, 2] = sign * np.sin(psi), -sign * np.cos(psi)
    b = xy.reshape(-1)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    res = (A @ sol - b).reshape(-1, 2)
    return sol[:2], float(sol[2]), float(np.sqrt(np.mean(np.sum(res * res, axis=1))))


def corridor_width(along, side) -> dict:
    """``TURN_RULES["corridor"]`` on the sweep's points (corner frame; any shape, flattened)."""
    x, y = np.ravel(np.asarray(along, float)), np.ravel(np.asarray(side, float))
    X, Y0 = float(x.max()), float(y.min())
    k = np.minimum(X - x, y - Y0)
    i = int(np.argmax(k))
    w = float(k[i])
    return dict(width_m=w, far_wall_along_m=X, outer_wall_sideways_m=Y0, inner_corner_m=[X - w, Y0 + w], index=i)


def _between(tq: float, t, x):
    """``x`` [n, ...] at the time ``tq``, linear between the rows on either side (a row's own time: that row; held
    at the ends); None if a row it needs is not carried (NaN)."""
    if len(t) == 1:
        v = x[0]
    else:
        k = int(np.clip(np.searchsorted(t, tq, side="right") - 1, 0, len(t) - 2))
        a = min(max((tq - t[k]) / (t[k + 1] - t[k]), 0.0), 1.0)
        v = x[k] if a == 0.0 else x[k + 1] if a == 1.0 else (1.0 - a) * x[k] + a * x[k + 1]
    return None if np.isnan(v).any() else np.asarray(v, float)


def turn_readings(t, xy, heading, t_c: float, t_end: float | None, psi_cmd: float, sign: float, angle: float,
                  pts=None, labels=None) -> dict:
    """The room a turn takes, from one path (``TURN_RULES``): ``t`` [n] seconds on the bar's clock, ``xy`` [n, 2]
    the root, ``heading`` [n] the trunk's unwrapped heading (rad), ``t_c`` the command, ``t_end`` the end of the
    move (None: not reached), ``psi_cmd`` the commanded heading, ``sign`` +1 left / -1 right, ``angle`` the
    commanded angle (rad); ``pts`` [n, N, 2] the surface points on each row (or None; NaN on rows that are not
    carried) with their ``labels`` [N].  Lengths in metres, angles in degrees; what cannot be read is None with its
    reason in ``notes``."""
    t, xy, heading = np.asarray(t, float), np.asarray(xy, float), np.asarray(heading, float)
    out: dict = dict(command_s=float(t_c), move_end_s=None if t_end is None else float(t_end), notes=[])
    if not len(t) or t[0] > t_c + bar.EPS_S or t[-1] < t_c - bar.EPS_S:
        out["notes"].append("the trace does not hold the command instant")
        return dict(out, reaches_command=False)
    psi = heading - 2.0 * math.pi * round((heading[0] - psi_cmd) / (2.0 * math.pi))
    turned = sign * (psi - psi_cmd)  # positive toward the side of the turn
    ex = np.array([math.cos(psi_cmd), math.sin(psi_cmd)])
    ey = sign * np.array([-ex[1], ex[0]])
    o = np.array([np.interp(t_c, t, xy[:, 0]), np.interp(t_c, t, xy[:, 1])])
    along, side = (xy - o) @ ex, (xy - o) @ ey
    reaches_end = t_end is not None and t[-1] >= t_end - bar.EPS_S
    t_b = float(t_end) if reaches_end else float(t[-1])
    win = np.flatnonzero((t >= t_c - bar.EPS_S) & (t <= t_b + bar.EPS_S))
    out.update(reaches_command=True, reaches_move_end=bool(reaches_end), read_to_s=t_b, rows=int(len(win)),
               turned_at_command_deg=math.degrees(float(np.interp(t_c, t, turned))),
               turned_at_read_end_deg=math.degrees(float(np.interp(t_b, t, turned))),
               turned_at_trace_end_deg=math.degrees(float(turned[-1])))
    if not reaches_end:
        out["notes"].append("the trace ends before the end of the move: read to the trace's end")
    if not len(win):
        return out
    # the root from the command to the end of the move: the rows and the two ends themselves (TURN_RULES["ends"];
    # at the command the root is the frame's origin)
    a_b, s_b = float(np.interp(t_b, t, along)), float(np.interp(t_b, t, side))
    A, S = np.r_[0.0, along[win], a_b], np.r_[0.0, side[win], s_b]
    out["corner_box_root"] = dict(
        advance_max_m=float(A.max()), advance_at_end_m=a_b, sideways_max_m=float(S.max()), sideways_at_end_m=s_b,
        back_max_m=float(max(-A.min(), 0.0)), outside_max_m=float(max(-S.min(), 0.0)),
        seconds_after_command=t_b - float(t_c))
    # the turn itself: to the first row at which the trunk has turned the commanded angle
    done = win[turned[win] >= angle - math.radians(TURNED_WITHIN_DEG)]
    rows1 = np.zeros(0, int)
    out["turn_itself"] = None
    if len(done):
        i1 = int(done[0])
        rows1 = win[win <= i1]
        A1, S1 = np.r_[0.0, along[rows1]], np.r_[0.0, side[rows1]]
        out["turn_itself"] = dict(
            within_deg=TURNED_WITHIN_DEG, at_s=float(t[i1]), seconds_after_command=float(t[i1] - t_c),
            turned_deg=math.degrees(float(turned[i1])), rows=int(len(rows1)),
            root_advance_m=float(along[i1]), root_sideways_m=float(side[i1]),
            root_advance_max_m=float(A1.max()), root_sideways_max_m=float(S1.max()),
            root_back_max_m=float(max(-A1.min(), 0.0)), root_outside_max_m=float(max(-S1.min(), 0.0)),
            root_to_move_end_m=float(math.hypot(a_b - along[i1], s_b - side[i1])) if reaches_end else None,
            robot=None)
    else:
        out["notes"].append(
            f"the turn itself is not read: the trunk turned {math.degrees(float(turned[win].max())):.1f} deg at most "
            f"between the command and {t_b:.2f} s, not the commanded {math.degrees(angle):.0f} deg to within "
            f"{TURNED_WITHIN_DEG:g} deg")
    # the steady part
    lo, hi = STEADY[0] * angle, STEADY[1] * angle
    after = win[turned[win] >= lo]
    steady = np.zeros(0, int)
    out["steady"] = dict(from_deg=math.degrees(lo), to_deg=math.degrees(hi), complete=False, rows=0)
    if len(after):
        i0 = int(after[0])
        over = win[(win >= i0) & (turned[win] >= hi)]
        if len(over):
            steady = np.arange(i0, int(over[0]) + 1)
            out["steady"].update(complete=True, rows=int(len(steady)), from_s=float(t[i0]), to_s=float(t[steady[-1]]))
    fit = None
    if len(steady) >= STEADY_MIN_ROWS:
        c, r, rms = fit_turn_circle(xy[steady], psi[steady], sign)
        T = float(t[steady[-1]] - t[steady[0]])
        rate = float(psi[steady[-1]] - psi[steady[0]]) / T
        fit = dict(radius_fitted_m=r, fit_rms_m=rms, centre_along_m=float((c - o) @ ex), centre_sideways_m=float((c - o) @ ey),
                   turn_rate_deg_s=math.degrees(abs(rate)), path_speed_m_s=float(r * abs(rate)),
                   radius_free_fit_m=fit_circle(xy[steady])[1], seconds=T, rows=int(len(steady)))
        out["centre_world"] = [float(c[0]), float(c[1])]
    else:
        out["notes"].append(
            f"stops short: the trunk turned {math.degrees(float(turned[win].max())):.1f} deg at most between the "
            f"command and {t_b:.2f} s; the steady part is {math.degrees(lo):.0f} to {math.degrees(hi):.0f} deg"
            if not out["steady"]["complete"] else f"the steady part holds {len(steady)} rows, fewer than {STEADY_MIN_ROWS}")
    out["fit"] = fit
    out["swept"] = out["swept_turn"] = out["corner_box_robot"] = out["corridor"] = None
    if pts is not None:
        pts = np.asarray(pts, float)
        p_c, p_b = _between(float(t_c), t, pts), _between(t_b, t, pts)  # the surface at the span's two ends
        head = [] if p_c is None else [p_c]
        whole = np.array(head + list(pts[win]) + ([] if p_b is None else [p_b]))
        pa, ps = (whole - o) @ ex, (whole - o) @ ey
        out["corner_box_robot"] = dict(along_min_m=float(pa.min()), along_max_m=float(pa.max()),
                                       sideways_min_m=float(ps.min()), sideways_max_m=float(ps.max()))
        turn = np.array(head + list(pts[rows1])) if len(rows1) else None
        if turn is not None:
            ta, ts = (turn - o) @ ex, (turn - o) @ ey
            out["turn_itself"]["robot"] = dict(along_min_m=float(ta.min()), along_max_m=float(ta.max()),
                                               sideways_min_m=float(ts.min()), sideways_max_m=float(ts.max()))
        if fit is not None:
            centre = np.asarray(out["centre_world"])
            dist = np.linalg.norm(pts[steady] - centre, axis=2)
            r_i, k_i = np.unravel_index(int(np.argmax(dist)), dist.shape)
            parts = sorted({lab.split(":")[0] for lab in labels}) if labels is not None else []
            per = {}
            for part in parts:
                sel = np.array([lab.split(":")[0] == part for lab in labels])
                per[part] = float(dist[:, sel].max())
            out["swept"] = dict(swept_radius_m=float(dist[r_i, k_i]), turning_circle_m=2.0 * float(dist[r_i, k_i]),
                                set_by=None if labels is None else labels[int(k_i)], at_s=float(t[steady[r_i]]),
                                inner_radius_m=float(dist.min()), farthest_by_part_m=per, over="the steady part")
            if turn is not None:
                d1 = np.linalg.norm(turn - centre, axis=2)
                r_1, k_1 = np.unravel_index(int(np.argmax(d1)), d1.shape)
                at = float(np.r_[[float(t_c)] * len(head), t[rows1]][r_1])
                out["swept_turn"] = dict(
                    largest_distance_m=float(d1[r_1, k_1]), set_by=None if labels is None else labels[int(k_1)],
                    at_s=at, seconds_after_command=at - float(t_c),
                    over="from the command until the trunk has turned the commanded angle")
        if reaches_end:
            cw = corridor_width(pa, ps)
            k = cw.pop("index") % pts.shape[1]
            out["corridor"] = dict(cw, inner_corner_set_by=None if labels is None else labels[int(k)])
        else:
            out["notes"].append("no corridor: the trace ends before the end of the move")
    return out


#: The readings a table and a summary take from ``turn_readings``: name -> (path in the dict, unit).
READINGS = {
    "radius_fitted": (("fit", "radius_fitted_m"), "m"),
    "turn_rate": (("fit", "turn_rate_deg_s"), "deg/s"),
    "path_speed": (("fit", "path_speed_m_s"), "m/s"),  # the fitted radius times the mean turn rate
    "turn_seconds": (("turn_itself", "seconds_after_command"), "s"),  # the turn itself (TURN_RULES["turn_itself"])
    "turn_root_advance": (("turn_itself", "root_advance_m"), "m"),
    "turn_root_sideways": (("turn_itself", "root_sideways_m"), "m"),
    "turn_robot_along_max": (("turn_itself", "robot", "along_max_m"), "m"),
    "turn_robot_sideways_max": (("turn_itself", "robot", "sideways_max_m"), "m"),
    "walk_after_turn": (("turn_itself", "root_to_move_end_m"), "m"),
    "advance_max": (("corner_box_root", "advance_max_m"), "m"),
    "advance_at_end": (("corner_box_root", "advance_at_end_m"), "m"),
    "sideways_max": (("corner_box_root", "sideways_max_m"), "m"),
    "sideways_at_end": (("corner_box_root", "sideways_at_end_m"), "m"),
    "swept_radius": (("swept", "swept_radius_m"), "m"),
    "largest_from_centre_in_turn": (("swept_turn", "largest_distance_m"), "m"),
    "turning_circle": (("swept", "turning_circle_m"), "m"),
    "corridor": (("corridor", "width_m"), "m"),
    "turned_at_judged_instant": (("turned_at_judged_instant_deg",), "deg"),
    "turned_at_move_end": (("turned_at_read_end_deg",), "deg"),
    "turned_at_trace_end": (("turned_at_trace_end_deg",), "deg"),
}


def reading(r: dict, name: str):
    node = r
    for key in READINGS[name][0]:
        if not isinstance(node, dict) or node.get(key) is None:
            return None
        node = node[key]
    return float(node)


def spread(values) -> dict | None:
    """``dict(n, median, min, max)`` of the values that are not None (None if there is none)."""
    v = [float(x) for x in values if x is not None]
    if not v:
        return None
    return dict(n=len(v), median=float(np.median(v)), min=float(min(v)), max=float(max(v)))


def summarize(readings: list[dict]) -> dict:
    """Per reading of ``READINGS``, ``spread`` over the episodes' ``turn_readings`` (the count is in each ``n``)."""
    return {name: spread([reading(r, name) for r in readings]) for name in READINGS}


# ------------------------------------------------------------------------------------------------
# A set: its frozen files, plans and planned (kinematic) traces
# ------------------------------------------------------------------------------------------------
class SetView:
    """One frozen set read for the views: its documents, the bar's planner and the body's model.  ``planned``
    plays a condition's reference kinematically at the bar's commanded clock (``bar.reference_recorder``: no
    physics) and returns it in a trace's shape."""

    def __init__(self, set_name: str, body: str = BODY, root=setio.ROOT):
        self.name, self.body = set_name, body
        self.dir = setio.set_dir(set_name, body, root)
        self.docs = setio.load(self.dir)
        self.set_doc = self.docs["set.json"]
        self.refs, self.trs = self.docs["refs.json"]["references"], self.docs["transitions.json"]["transitions"]
        self.planner = bar.BarPlanner(self.dir, self.refs, self.trs, self.set_doc)
        self.model = body_model(body)
        self._omega: dict = {}
        self._planned: dict = {}

    def conditions(self) -> list:
        return bar.conditions(self.docs)

    def condition(self, key: str):
        for c in bar.conditions(self.docs, every=True):
            if c.key == key:
                return c
        raise ViewError(f"{self.name}: no condition {key!r}")

    def omega(self, files) -> float:
        crouch = bar.set_crouch(self.set_doc, self.refs, bar.walks_of(files))
        if crouch not in self._omega:
            q_home = Reference.load(self.dir / "refs" / f"{self.planner.home_walk}.npz").q_home
            self._omega[crouch] = float(control.omega0(self.model.p, q_home, crouch_frac=crouch))
        return self._omega[crouch]

    def move_end_ref_s(self, plan, move: str | None) -> float | None:
        """The reference clock's time at the end of the commanded move's own file on ``plan`` (None: no move)."""
        plan = [tuple(x) for x in plan]
        if move is None or ("transition", move) not in plan:
            return None
        t0, dur = self.planner.segment_times(plan)[plan.index(("transition", move))]
        return float(t0 + dur)

    def inputs_differ(self, record: dict) -> list[str]:
        """The input files of ``record`` whose sha256 is not this checkout's set's (empty: the same files)."""
        listed = {e["file"]: e.get("sha256") for doc in (self.refs, self.trs) for e in doc.values()
                  if isinstance(e, dict) and e.get("file")}
        return sorted(f for f, x in ((record.get("inputs") or {}).get("files") or {}).items()
                      if listed.get(f) != x.get("sha256"))

    def planned(self, cond) -> dict:
        """The condition's reference played kinematically: ``dict(trace, pose, res, t_c, t_end, freeze, files)``;
        the files are checked against their entries and the set's latest freeze first (``setio.read_inputs``)."""
        if cond.key in self._planned:
            return self._planned[cond.key]
        res = self.planner.resolve(cond)
        inputs, _ = setio.read_inputs(self.dir, res["files"], require_frozen=True)
        p = self.model.p
        track = setio.build_track(p, self.dir, res["plan"], self.omega(res["files"]), self.trs)
        rec = bar.reference_recorder(p, track, res["seconds"], self.dir, rate=bar.RATE)
        every = bar.panels.TRACE_EVERY
        U = rec.U[::every]
        trace = dict(t=U[:, metrics.C["t"]].copy(), t_ref=U[:, metrics.C["t_ref"]].copy(), rows=U,
                     cols=np.array(metrics.COLS), q=rec.Q[::every], dt=np.array(rec.dt))
        t_c = None if res.get("command_ref_s") is None else float(res["command_ref_s"]) / bar.RATE
        end_ref = self.move_end_ref_s(res["plan"], cond.move)
        out = dict(trace=trace, pose=pose_of(trace, self.model, joint_lag_steps=0), res=res, t_c=t_c,
                   t_end=None if end_ref is None else end_ref / bar.RATE, freeze=inputs.get("freeze"),
                   files={f: x["sha256"] for f, x in inputs["files"].items()}, seconds=float(res["seconds"]))
        self._planned[cond.key] = out
        return out


_SETS: dict = {}


def set_view(set_name: str, body: str = BODY, root=setio.ROOT) -> SetView:
    key = (set_name, body, str(root))
    if key not in _SETS:
        _SETS[key] = SetView(set_name, body, root)
    return _SETS[key]


def surface_rows(model: BodyModel, pose: dict, rows) -> np.ndarray:
    """The surface points in plan [len(rows), N, 2] on the poses of ``rows``."""
    return np.array([model.surface(pose["qpos"][i])[:, :2] for i in rows])


def path_readings(model: BodyModel, pose: dict, t_c: float, t_end: float | None, sign: float, angle: float) -> dict:
    """``turn_readings`` of a pose series: the surface points are carried on the rows from the command to the end
    of the move only (the rows the readings take, and one row on either side)."""
    t = pose["t"]
    n = len(t)
    stop = t[-1] if t_end is None else t_end
    rows = np.flatnonzero((t >= t_c - bar.EPS_S) & (t <= stop + bar.EPS_S))
    if len(rows):  # (and the row on either side: the span's two ends are read between rows)
        rows = np.arange(max(int(rows[0]) - 1, 0), min(int(rows[-1]) + 2, n))
    pts = np.full((n, len(model.labels), 2), np.nan)
    if len(rows):
        pts[rows] = surface_rows(model, pose, rows)
    return turn_readings(t, pose["root_xy"], pose["yaw"], t_c, t_end, model.psi_cmd, sign, angle, pts, model.labels)


# ------------------------------------------------------------------------------------------------
# Records and traces, read defensively (the run directory may still be filling)
# ------------------------------------------------------------------------------------------------
def run_label(run) -> str:
    """The run directory as a record cites it: its path from ``stage2/`` on (the same in every checkout)."""
    parts = Path(run).resolve().parts
    return "/".join(parts[parts.index("stage2"):]) if "stage2" in parts else Path(run).name


def read_record(path) -> tuple[dict | None, str | None]:
    """(the record, None), or (None, why it cannot be read)."""
    try:
        doc = json.loads(Path(path).read_text())
    except OSError as exc:
        return None, f"cannot be opened ({type(exc).__name__})"
    except ValueError:
        return None, "does not parse (half-written?)"
    if not isinstance(doc, dict) or doc.get("schema") != bar.SCHEMA:
        return None, f"is not a {bar.SCHEMA} record"
    return doc, None


def record_files(run, set_name: str, body: str = BODY, behavior: str | None = None) -> list[Path]:
    """Every record file of ``set_name`` under ``run`` (each block, row and condition; superseded copies, traces,
    partial episodes and summaries left out), sorted; ``behavior`` keeps one behaviour's."""
    base = Path(run) / set_name / body
    out = []
    for blk in sorted(base.glob("seeds*")):
        ev = blk / "eval"
        if not ev.is_dir():
            continue
        for f in sorted(ev.rglob("*.json")):
            rel = f.relative_to(ev).parts
            if rel[0].startswith("traces") or any(x.endswith(".superseded") for x in rel) or len(rel) < 3:
                continue
            if behavior is None or rel[-2] == behavior:
                out.append(f)
    return out


def record_place(run, path) -> dict:
    """``dict(set, block, row, condition)`` of a record file from its path under ``run`` (``bar.record_path``)."""
    parts = Path(path).relative_to(run).parts
    i = parts.index("eval")
    return dict(set=parts[0], block=parts[i - 1], row="/".join(parts[i + 1 : -2]),
                condition=f"{parts[-2]}/{Path(parts[-1]).stem}")


def trace_file(run, record: dict, episode: dict) -> Path | None:
    tr = episode.get("trace") or {}
    if not tr.get("path") or tr.get("kept") is False:
        return None
    return Path(run) / record["set"] / record["body"] / record["block"] / "eval" / "traces" / tr["path"]


def read_trace(run, record: dict, episode: dict) -> tuple[dict | None, str | None]:
    """(the episode's trace arrays, None), or (None, why there is none): not written, not kept, missing, or not
    the bytes whose sha256 the record holds."""
    tr = episode.get("trace")
    if not tr:
        return None, "the record names no trace for this episode"
    if tr.get("kept") is False or not tr.get("path"):
        return None, f"the trace was not kept ({tr.get('reason') or 'no path in the record'})"
    path = trace_file(run, record, episode)
    try:
        data = path.read_bytes()
    except OSError as exc:
        return None, f"the trace file is missing ({type(exc).__name__})"
    if hashlib.sha256(data).hexdigest() != tr.get("sha256"):
        return None, "the trace file's sha256 is not the record's"
    try:
        with np.load(io.BytesIO(data), allow_pickle=False) as z:
            return {k: z[k] for k in z.files}, None
    except (OSError, ValueError) as exc:
        return None, f"the trace file cannot be read ({type(exc).__name__})"


def outcome(episode: dict) -> str:
    """"stayed up", "fell" or "failed to run" (an episode ``bar.RULES["error"]`` records with its reason)."""
    if episode.get("error"):
        return "failed to run"
    return "fell" if episode.get("fell") else "stayed up"


def count_words(record: dict) -> str:
    eps = record.get("episodes") or []
    up = sum(outcome(e) == "stayed up" for e in eps)
    bad = sum(outcome(e) == "failed to run" for e in eps)
    return f"stayed up in {up} of {len(eps)}" + (f" ({bad} failed to run)" if bad else "")


def grip_of(record: dict) -> float:
    mu = (record.get("definition") or {}).get("friction")
    return GRIP_HARD if mu is None else float(mu)


def row_words(record: dict) -> str:
    """"controller R2/42, servo delay 10 ms, floor grip 0.9" from a record."""
    row = record["row"]
    ctl = bar.controller_of(row)
    ms = bar.delay_ms(row)
    name = "open loop (R0)" if ctl == "R0" else f"controller {row.split('@')[0]}"
    return f"{name}, servo delay {ms} ms, floor grip {grip_of(record):g}"


def condition_words(definition: dict, transitions: dict | None = None) -> str:
    """A condition in plain words from a record's ``definition`` (or a condition's ``describe()``)."""
    walk = definition.get("walk") or definition.get("gait") or ""
    v = setio.parse_name(walk).get("speed_m_s") if walk else None
    at = "the nominal walk" if v is None else f"{v:.2f} m/s"
    beh = definition["behavior"]
    if beh == "walk":
        return f"walk at {at}"
    if beh == "turn":
        tn = setio.parse_name(definition["move"])
        quick = "quick " if tn["rate"] == setio.QUICK else ""
        rate = "" if quick else f" ({tn['rate_rad_s']:g} rad/s)"
        return f"{quick}{tn['angle_deg']}° turn to the {tn['direction']}{rate} at {at}"
    if beh == "stop":
        steps = setio.parse_name(definition["move"])["steps"]
        return f"stop from {at} within {'two steps' if steps == 2 else 'one step'}"
    if beh == "nudge":
        return f"nudge {definition['direction']} ({definition['frac']:g} of the push bound) while walking at {at}"
    return f"{beh} {definition.get('name')}"


# ------------------------------------------------------------------------------------------------
# turnpath
# ------------------------------------------------------------------------------------------------
def turn_conditions(sv: SetView) -> tuple[list, list]:
    """(the set's turn conditions on the hard floor that have a reference, [{condition, reason}] for the others):
    one per turn reference; a low-grip condition plays the same reference and is listed with its records."""
    have, missing = [], []
    for c in sv.conditions():
        if c.judge != "bar_turn" or c.friction is not None:
            continue
        why = sv.planner.missing(c)
        if why is None:
            have.append(c)
        else:
            missing.append(dict(condition=c.key, move=c.move, reason=why["reason"], reference=why["reference"]))
    return have, missing


def planned_turn(sv: SetView, cond) -> dict:
    """The planned block of one turn (``TURN_RULES``)."""
    pl = sv.planned(cond)
    res = pl["res"]
    entry = sv.trs[cond.move]
    v, rate = float(entry["gaits"]["a"]["speed"]), abs(float(entry["turn"]["rate_rad_s"]))  # (a right turn's is negative)
    r = path_readings(sv.model, pl["pose"], pl["t_c"], pl["t_end"], float(res["sign"]), float(res["angle_rad"]))
    t_at = pl["t_c"] + float(res["at_s"])
    psi = pl["pose"]["yaw"]
    r["turned_at_judged_instant_deg"] = math.degrees(
        float(res["sign"]) * (float(np.interp(t_at, pl["pose"]["t"], psi)) - sv.model.psi_cmd))
    return dict(
        what="planned", source="the frozen reference played kinematically at the bar's commanded clock",
        move=cond.move, file=entry["file"], file_sha256=entry["sha256"], freeze=pl["freeze"],
        plan=[list(x) for x in res["plan"]], clock_rate=bar.RATE,
        commanded=dict(path_speed_m_s=v, turn_rate_rad_s=rate, turn_rate_deg_s=math.degrees(rate),
                       radius_m=v / rate, angle_deg=math.degrees(float(res["angle_rad"])),
                       direction=entry["turn"]["direction"],
                       on_the_bars_clock=dict(path_speed_m_s=v * bar.RATE, turn_rate_deg_s=math.degrees(rate) * bar.RATE)),
        judged_instant_s_after_command=float(res["at_s"]), readings=r,
        pose_check=pl["pose"]["residual_m"])


def simulated_turn(sv: SetView, run, path: Path, with_paths: bool = False) -> dict:
    """The simulated block of one record (``TURN_RULES["simulated"]``): per episode that stayed up its readings,
    then their spread; the episodes that fell, and those not read with the reason."""
    rel = Path(path).relative_to(run).as_posix()
    where = record_place(run, path)
    rec, why = read_record(path)
    if rec is None:  # (listed under its turn all the same: the path says which record it is)
        try:
            move = sv.condition(where["condition"]).move
        except ViewError:
            move = None
        return dict(where, record=rel, read=False, move=move, reason=f"the record {why}")
    out = dict(record=rel, read=True, status=rec.get("status"), row=rec.get("row"), delay_ms=bar.delay_ms(rec["row"]),
               floor_grip=grip_of(rec), block=rec.get("block"), condition=rec.get("key_name"),
               move=(rec.get("definition") or {}).get("move"), key_sha256=rec.get("key_sha256"))
    if rec.get("status") != "done":
        return dict(out, reason=f"status {rec.get('status')!r}: {rec.get('reason')}", episodes=[], fell=[], not_read=[])
    differ = sv.inputs_differ(rec)
    if differ:
        return dict(out, read=False, reason="the record was made from other files than this checkout's set: "
                                            + ", ".join(differ))
    tn = setio.parse_name(out["move"])
    end_ref = sv.move_end_ref_s(rec["plan"], out["move"])
    eps, fell, not_read, paths = [], [], [], []
    for e in rec.get("episodes") or []:
        oc = outcome(e)
        if oc == "fell":
            fell.append(dict(seed=e["seed"], fall_time_s=e.get("fall_time_s"), fall_cause=e.get("fall_cause"),
                             command_s=e.get("command_s"),
                             when="before the command" if e.get("fell_before_command") else "after the command"))
            continue
        if oc != "stayed up":
            not_read.append(dict(seed=e["seed"], reason=f"failed to run: {e.get('error')}"))
            continue
        trace, why = read_trace(run, rec, e)
        if trace is None:
            not_read.append(dict(seed=e["seed"], reason=why))
            continue
        pose = pose_of(trace, sv.model)
        t_c = float(e["command_s"])
        reached = end_ref is not None and len(pose["t_ref"]) and end_ref <= pose["t_ref"][-1] + 1e-12
        t_end = float(np.interp(end_ref, pose["t_ref"], pose["t"])) if reached else None
        r = path_readings(sv.model, pose, t_c, t_end, tn["sign"], tn["angle_rad"])
        b = (e.get("measures") or {}).get("bar") or {}
        if b.get("heading_error_at_deg") is not None:  # the bar's own judged instant and reading
            r["judged_instant_s"] = float(e["command_s"]) + float(b["at_s"])
            r["turned_at_judged_instant_deg"] = float(b["angle_deg"]) + float(b["sign"]) * float(b["heading_error_at_deg"])
            mine = math.degrees(tn["sign"] * (float(np.interp(r["judged_instant_s"], pose["t"], pose["yaw"]
                                                              - 2 * math.pi * round((pose["yaw"][0] - sv.model.psi_cmd)
                                                                                    / (2 * math.pi)))) - sv.model.psi_cmd))
            r["turned_at_judged_instant_from_trace_deg"] = mine
        eps.append(dict(seed=e["seed"], trace_sha256=e["trace"]["sha256"], readings=r, pose_check=pose["residual_m"]))
        if with_paths and r.get("reaches_command"):
            paths.append(_anchored_path(sv.model, pose, t_c, t_end, tn["sign"]))
    n = len(rec.get("episodes") or [])
    up = len(eps) + sum("failed to run" not in x["reason"] for x in not_read)
    text = f"{up} of {n} episodes stayed up"
    if len(eps) != up:
        text += f"; {len(eps)} of those could be read (the others' traces: see not_read)"
    out.update(episodes=eps, fell=fell, not_read=not_read, count=dict(episodes=n, stayed_up=up, read=len(eps), fell=len(fell)),
               count_text=text, summary=summarize([x["readings"] for x in eps]) if eps else None,
               counts=rec.get("counts"), gated=rec.get("gated"))
    if with_paths:
        out["_paths"] = paths
    return out


def compact_simulated(s: dict) -> dict:
    """A simulated block as the committed file holds it: each episode's readings as the values of ``READINGS``
    (metres, degrees; ``reading_units``) with what set them and the notes, in place of ``turn_readings``' whole
    dict (it can be made again from the trace), and without the paths the figure draws."""
    out = {k: v for k, v in s.items() if k not in ("_paths", "episodes")}
    if "episodes" in s:
        out["episodes"] = [
            dict(seed=e["seed"], trace_sha256=e["trace_sha256"],
                 values={name: reading(e["readings"], name) for name in READINGS},
                 command_s=e["readings"].get("command_s"), move_end_s=e["readings"].get("move_end_s"),
                 judged_instant_s=e["readings"].get("judged_instant_s"),
                 swept_radius_set_by=(e["readings"].get("swept") or {}).get("set_by"),
                 largest_from_centre_in_turn_set_by=(e["readings"].get("swept_turn") or {}).get("set_by"),
                 corridor_inner_corner_set_by=(e["readings"].get("corridor") or {}).get("inner_corner_set_by"),
                 notes=e["readings"].get("notes") or [])
            for e in s["episodes"]]
        if s["episodes"]:  # pose_of's largest residual over the record's episodes (metres; module docstring)
            out["pose_check_worst"] = {k: max(e["pose_check"][k] for e in s["episodes"]) for k in s["episodes"][0]["pose_check"]}
    return out


def _anchored_path(model: BodyModel, pose: dict, t_c: float, t_end: float | None, sign: float) -> np.ndarray:
    """The root's path in its own corner frame [k, 2] (along, sideways) from ``LEAD_S`` before the command to the
    end of the move (or the trace's end)."""
    t, xy = pose["t"], pose["root_xy"]
    ex = np.array([math.cos(model.psi_cmd), math.sin(model.psi_cmd)])
    ey = sign * np.array([-ex[1], ex[0]])
    o = np.array([np.interp(t_c, t, xy[:, 0]), np.interp(t_c, t, xy[:, 1])])
    sel = (t >= t_c - LEAD_S) & (t <= (t[-1] if t_end is None else t_end) + bar.EPS_S)
    return np.column_stack([(xy[sel] - o) @ ex, (xy[sel] - o) @ ey])


def figure_record(sims: list, rows=FIGURE_ROWS) -> tuple[dict | None, list]:
    """(the record whose paths a turn's figure draws, the rows before it that had none to draw): ``FIGURE_ROWS``'
    rule on the turn's simulated blocks; (the first row's record, []) if no row has a path; (None, []) if none of
    the rows has a record."""
    have = []
    for row in rows:
        s = next((x for x in sims if x.get("row") == row and x.get("floor_grip") == GRIP_HARD
                  and x.get("block") == bar.HEADLINE_BLOCK and x.get("read")), None)
        if s is None:
            continue
        if s.get("_paths"):
            return s, [f"{x['row']} ({x.get('count_text') or x.get('reason')})" for x in have]
        have.append(s)
    return (have[0] if have else None), []


def turnpath(set_name: str, run=RUN, out=None, body: str = BODY, figure_row: str | None = None, log=print) -> dict:
    """``barviews turnpath``: write ``<out>/turnpath/<set>/turnpath.json``, ``TURNPATH.md`` and a figure per turn
    (module docstring); returns the document."""
    run = Path(run)
    out = check_out(run, out)
    sv = set_view(set_name, body)
    conds, missing = turn_conditions(sv)
    d = out / "turnpath" / set_name
    L = sv.model.length_m
    files = record_files(run, set_name, body, "turn")
    sims = [simulated_turn(sv, run, f, with_paths=True) for f in files]
    turns, figs = [], {}
    for c in conds:
        pl = planned_turn(sv, c)
        mine = [s for s in sims if s.get("move") == c.move]
        pick, before = figure_record(mine, FIGURE_ROWS if figure_row is None else (figure_row,))
        stem = c.key.replace("/", "_")
        figs[c.key] = figure(sv, c, pl, pick, d / stem, before)
        turns.append(dict(condition=c.key, in_words=condition_words(c.describe()), gated=bool(c.gated), move=c.move,
                          planned=pl, simulated=[compact_simulated(s) for s in mine],
                          simulated_note=None if mine else "no record yet",
                          figure=dict(svg=f"{stem}.svg", png=f"{stem}.png", caption=figs[c.key],
                                      simulated_from=None if pick is None else pick["record"],
                                      rows_before_without_a_path=before,
                                      rule="the simulated paths are those of the first of the rows "
                                           + ", ".join(FIGURE_ROWS if figure_row is None else (figure_row,))
                                           + " (hard floor, headline block) whose record has an episode that "
                                             "stayed up and can be read")))
        log(f"{set_name} {c.key}: planned radius {pl['readings']['fit']['radius_fitted_m']:.3f} m, "
            f"{len(mine)} simulated record(s)")
    unread = [compact_simulated(s) for s in sims if not s.get("read")]
    doc = dict(schema=SCHEMA_TURNPATH, set=set_name, set_in_words=SET_WORDS.get(set_name, set_name), body=body,
               what="views of committed records and a frozen reference: no verdict, nothing adopted",
               run=run_label(run), rules=TURN_RULES, body_length=sv.model.length_info,
               reading_units={name: unit for name, (_, unit) in READINGS.items()},
               set_freeze=(setio.frozen(sv.set_doc) or {}).get("index"),
               set_manifest_sha256=(setio.frozen(sv.set_doc) or {}).get("manifest_sha256"),
               turns=turns, no_reference=missing, records_not_read=unread,
               records_seen=[Path(f).relative_to(run).as_posix() for f in files],
               provenance=dict(git=s2io.git_state(own_outputs=[str(out)]), at=s2io.now_utc(),
                               module="designlab/stage2/barviews.py",
                               module_sha256=s2io.sha256_file(Path(__file__))))
    doc = in_body_lengths(s2io.plain(doc), L)
    s2io.write_json(d / "turnpath.json", doc)
    (d / "TURNPATH.md").write_text(turnpath_md(doc))
    return doc


def in_body_lengths(x, L: float):
    """``x`` with, beside every number under a key that ends in ``_m``, the same length in body lengths under the
    key with ``_bl`` in place of ``_m`` (lists of numbers too)."""
    if isinstance(x, list):
        return [in_body_lengths(v, L) for v in x]
    if not isinstance(x, dict):
        return x
    out = {}
    for k, v in x.items():
        out[k] = in_body_lengths(v, L)
        if isinstance(k, str) and k.endswith("_m") and k != "body_length_m":
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                out[k[:-2] + "_bl"] = v / L
            elif isinstance(v, list) and v and all(isinstance(a, (int, float)) for a in v):
                out[k[:-2] + "_bl"] = [a / L for a in v]
            elif isinstance(v, dict) and all(isinstance(a, (int, float)) for a in v.values()):
                out[k[:-2] + "_bl"] = {kk: a / L for kk, a in v.items()}
    return out


def check_out(run, out) -> Path:
    """The output directory: ``<this checkout's run directory>/views`` by default.  Refused: a directory inside the
    run directory the records are read from, unless it is that run directory's own ``views/``."""
    out = Path(out) if out is not None else RUN / VIEWS
    o, r = Path(os.path.realpath(out)), Path(os.path.realpath(run))
    if (o == r or r in o.parents) and not (o == r / VIEWS or (r / VIEWS) in o.parents):
        raise ViewError(f"{out}: inside the run directory {run} and not under its {VIEWS}/ (nothing else there is "
                        "this module's to write)")
    return out


# ------------------------------------------------------------------------------------------------
# turnpath: the text and the figure
# ------------------------------------------------------------------------------------------------
def _len(x, L: float, nd: int = 2) -> str:
    return "-" if x is None else f"{x:.{nd}f} m ({x / L:.2f} BL)"


def _num(x, nd: int = 1, unit: str = "") -> str:
    return "-" if x is None else f"{x:.{nd}f}{unit}"


def _range(s: dict | None, L: float | None = None, nd: int = 2, unit: str = "") -> str:
    """"median (smallest to largest)" of a ``spread``; lengths with their body lengths."""
    if s is None:
        return "-"
    if L is not None:
        return (f"{s['median']:.{nd}f} m ({s['min']:.{nd}f} to {s['max']:.{nd}f}); "
                f"{s['median'] / L:.2f} BL ({s['min'] / L:.2f} to {s['max'] / L:.2f})")
    return f"{s['median']:.{nd}f}{unit} ({s['min']:.{nd}f} to {s['max']:.{nd}f})"


def seed_runs(seeds) -> str:
    """"3042-3047, 3050": the seeds as runs of consecutive numbers."""
    seeds = sorted(int(x) for x in seeds)
    out, i = [], 0
    while i < len(seeds):
        j = i
        while j + 1 < len(seeds) and seeds[j + 1] == seeds[j] + 1:
            j += 1
        out.append(str(seeds[i]) if j == i else f"{seeds[i]}-{seeds[j]}")
        i = j + 1
    return ", ".join(out)


def fell_words(fell: list) -> str:
    """The episodes that fell in a sentence's clause: how many, and per 'before / after the command' the seeds and
    the fall instants' span."""
    if not fell:
        return "0 fell"
    parts = []
    for when in ("before the command", "after the command"):
        g = [x for x in fell if x["when"] == when]
        if g:
            ts = [x["fall_time_s"] for x in g]
            span = f"at {min(ts):.2f} s" if len(g) == 1 else f"between {min(ts):.2f} and {max(ts):.2f} s"
            parts.append(f"{len(g)} {when} (seed{'s' if len(g) > 1 else ''} {seed_runs(x['seed'] for x in g)}, {span})")
    return f"{len(fell)} fell: " + "; ".join(parts)


def turnpath_md(doc: dict) -> str:
    L = doc["body_length"]["body_length_m"]
    words = doc["set_in_words"]
    A = [f"# The room a turn needs: {doc['set']} ({words}) on {doc['body']}", "",
         "A view of the frozen reference and of the committed records. It gives no verdict and adopts nothing.", "",
         "Two words are kept apart. **Planned** is read from the frozen reference, played kinematically at the bar's "
         "commanded clock, with no physics. **Simulated** is read from an episode's trace. No number mixes the two.",
         "",
         f"Lengths are in metres and in body lengths (BL). One body length is {L:.3f} m: the model's overall length "
         f"at its home keyframe, from {doc['body_length']['back']} to {doc['body_length']['front']}.", "",
         f"Records are read from `{doc['run']}` and cited by their path under it; it held "
         f"{len(doc['records_seen'])} turn record{'' if len(doc['records_seen']) == 1 else 's'} of this set when this "
         f"file was written ({doc['provenance']['at']}). Set freeze {doc['set_freeze']}, manifest "
         f"`{str(doc['set_manifest_sha256'])[:12]}`.", ""]
    for tn in doc["turns"]:
        pl = tn["planned"]
        r, cmd = pl["readings"], pl["commanded"]
        fit, box, rob, sw, co = r["fit"], r["corner_box_root"], r["corner_box_robot"], r["swept"], r["corridor"]
        ti, st = r.get("turn_itself"), r.get("swept_turn")
        tr_ = (ti or {}).get("robot")
        to_end = (f"to the end of the move's file, {box['seconds_after_command']:.2f} s after the command"
                  + (f"; includes {ti['root_to_move_end_m']:.2f} m of walk after the turn" if ti and
                     ti.get("root_to_move_end_m") is not None else ""))
        ang = f"{cmd['angle_deg']:.0f}°"
        A += [f"## {tn['condition']}: {tn['in_words']}", "",
              f"![top view]({tn['figure']['png']})", "", tn["figure"]["caption"], "",
              "### Planned", "",
              f"From `{pl['file']}` (sha256 `{pl['file_sha256'][:12]}`), played at {pl['clock_rate']:g} times its own "
              "clock.", "",
              "| reading (planned) | value |", "|---|---|",
              f"| path speed commanded | {cmd['path_speed_m_s']:.2f} m/s on the reference's clock; "
              f"{cmd['on_the_bars_clock']['path_speed_m_s']:.2f} m/s on the bar's clock (the reference is played "
              f"{pl['clock_rate']:g} times faster) |",
              f"| turn rate commanded | {cmd['turn_rate_deg_s']:.1f} deg/s on the reference's clock; "
              f"{cmd['on_the_bars_clock']['turn_rate_deg_s']:.1f} deg/s on the bar's clock |",
              f"| path radius commanded (speed over rate) | {_len(cmd['radius_m'], L)} |",
              f"| radius fitted to the root's path, steady part | {_len(fit and fit['radius_fitted_m'], L)}, RMS "
              f"{_num(fit and 1e3 * fit['fit_rms_m'], 0)} mm off the circle (the root sways) |",
              f"| **the turn itself**: until the trunk has turned {ang} (to within {TURNED_WITHIN_DEG:g}°) | "
              + (f"{ti['seconds_after_command']:.2f} s after the command |" if ti else "- (the trunk never turns that far) |"),
              f"| the turn itself, root: where it is then | {_len(ti and ti['root_advance_m'], L)} along the incoming "
              f"line, {_len(ti and ti['root_sideways_m'], L)} sideways |",
              f"| the turn itself, whole robot: along | {_len(tr_ and tr_['along_min_m'], L)} to "
              f"{_len(tr_ and tr_['along_max_m'], L)} |",
              f"| the turn itself, whole robot: sideways | {_len(tr_ and tr_['sideways_min_m'], L)} to "
              f"{_len(tr_ and tr_['sideways_max_m'], L)} |",
              f"| corner box {to_end}; root: advance along the incoming line | largest "
              f"{_len(box['advance_max_m'], L)}, at the move's end {_len(box['advance_at_end_m'], L)} |",
              f"| corner box to the end of the move's file; root: sideways | largest {_len(box['sideways_max_m'], L)}, "
              f"at the move's end {_len(box['sideways_at_end_m'], L)} |",
              f"| corner box to the end of the move's file; whole robot: along | {_len(rob['along_min_m'], L)} to "
              f"{_len(rob['along_max_m'], L)} |",
              f"| corner box to the end of the move's file; whole robot: sideways | {_len(rob['sideways_min_m'], L)} to "
              f"{_len(rob['sideways_max_m'], L)} |",
              f"| swept radius over the steady part of the turn (the turning circle uses this one) | "
              f"{_len(sw and sw['swept_radius_m'], L)}, set by {sw and sw['set_by']} |",
              f"| largest distance from the turn's centre, from the command until the trunk has turned {ang} | "
              f"{_len(st and st['largest_distance_m'], L)}, set by {st and st['set_by']}"
              + ("" if not st else ", at the command |" if st["seconds_after_command"] < 0.005
                 else f", {st['seconds_after_command']:.2f} s after the command |") + ("" if st else " |"),
              f"| turning circle (twice the swept radius of the steady part) | {_len(sw and sw['turning_circle_m'], L)} |",
              f"| corridor for the 90° corner | {_len(co and co['width_m'], L)}, inner corner set by "
              f"{co and co['inner_corner_set_by']} |",
              f"| turned {pl['judged_instant_s_after_command']:g} s after the command | "
              f"{_num(r['turned_at_judged_instant_deg'], 1, '°')} |",
              f"| turned at the end of the move ({r['read_to_s'] - r['command_s']:.2f} s after the command) | "
              f"{_num(r['turned_at_read_end_deg'], 1, '°')} |", ""]
        if sw:
            A += ["Farthest point from the turn's centre, by part (planned): "
                  + "; ".join(f"{k} {v:.2f} m" for k, v in sorted(sw["farthest_by_part_m"].items(), key=lambda kv: -kv[1]))
                  + ".", ""]
        A += ["### Simulated", ""]
        if not tn["simulated"]:
            A += ["No record yet: the chain had made no record of this turn when this file was written.", ""]
            continue
        for s in tn["simulated"]:
            if not s.get("read") or s.get("status") != "done":
                A += [f"- `{s['record']}` (row {s.get('row')}, block {s.get('block')}). Not read: {s.get('reason')}.", ""]
                continue
            head = (f"`{s['record']}`: controller row {s['row']}, servo delay {s['delay_ms']} ms, floor grip "
                    f"{s['floor_grip']:g}, block {s['block']}")
            A += [f"**{head}.**", ""]
            A += [f"{s['count_text']}; {fell_words(s['fell'])}.", ""]
            for x in s["not_read"]:
                A += [f"- seed {x['seed']} not read: {x['reason']}."]
            if s["not_read"]:
                A += [""]
            sm = s["summary"]
            if not sm:
                A += ["No episode of this record stayed up with a trace that can be read: nothing is read.", ""]
                continue
            n = s["count"]["read"]
            A += [f"Of the {n} read (median, then smallest to largest; the last column counts the episodes that have "
                  "the reading: the turn itself is read only where the trunk turned the commanded angle to within "
                  f"{TURNED_WITHIN_DEG:g}° before the end of the move):", "",
                  "| reading (simulated) | median (smallest to largest) | episodes |", "|---|---|---|"]
            for name, title in (("radius_fitted", "radius fitted to the root's path, steady part"),
                                ("turn_root_advance", "the turn itself, root: along the incoming line when the trunk "
                                                      f"has turned {ang}"),
                                ("turn_root_sideways", f"the turn itself, root: sideways when the trunk has turned {ang}"),
                                ("turn_robot_along_max", "the turn itself, whole robot: farthest along"),
                                ("turn_robot_sideways_max", "the turn itself, whole robot: farthest sideways"),
                                ("walk_after_turn", "walk after the turn, to the end of the move's file (root)"),
                                ("advance_max", "corner box to the end of the move's file, root: largest advance"),
                                ("advance_at_end", "corner box to the end of the move's file, root: advance at the end"),
                                ("sideways_max", "corner box to the end of the move's file, root: largest sideways"),
                                ("sideways_at_end", "corner box to the end of the move's file, root: sideways at the end"),
                                ("swept_radius", "swept radius over the steady part"),
                                ("largest_from_centre_in_turn", "largest distance from the turn's centre, from the "
                                                                f"command until the trunk has turned {ang}"),
                                ("turning_circle", "turning circle (twice the swept radius of the steady part)"),
                                ("corridor", "corridor for the 90° corner")):
                A += [f"| {title} | {_range(sm[name], L)} | {sm[name]['n'] if sm[name] else 0} |"]
            for name, title, unit in (("turn_seconds", "the turn itself: seconds from the command until the trunk "
                                                       f"has turned {ang} (to within {TURNED_WITHIN_DEG:g}°)", " s"),
                                      ("turn_rate", "turn rate over the steady part", " deg/s"),
                                      ("path_speed", "path speed over the steady part", " m/s"),
                                      ("turned_at_judged_instant", "turned at the bar's judged instant", "°"),
                                      ("turned_at_move_end", "turned at the end of the move", "°"),
                                      ("turned_at_trace_end", "turned at the end of the trace", "°")):
                A += [f"| {title} | {_range(sm[name], None, 1 if unit not in (' m/s', ' s') else 2, unit)} | "
                      f"{sm[name]['n'] if sm[name] else 0} |"]
            A += [""]
    if doc["no_reference"]:
        A += ["## Turns without a reference in this set", ""]
        A += [f"- {x['condition']} (`{x['reference']}`): {x['reason']}" for x in doc["no_reference"]] + [""]
    if doc["records_not_read"]:
        A += ["## Records that could not be read", ""]
        A += [f"- `{x['record']}`: {x.get('reason')}" for x in doc["records_not_read"]] + [""]
    A += ["## How each reading is made", ""]
    A += [f"- **{k.replace('_', ' ')}**: {v}." for k, v in doc["rules"].items()] + [""]
    return "\n".join(A)


#: The figure's colours: ``designlab/page_kit.py``'s tokens (series 1 and 2, inks, rule, band, surface).
COLOURS = dict(planned="#2a78d6", simulated="#eb6834", ink="#1c211d", ink2="#4a514a", ink3="#6b7068", rule="#d8d1c0",
               band="#ddd6c5", surface="#fbf9f3")


def figure(sv: SetView, cond, planned: dict, sim: dict | None, stem: Path, before=()) -> str:
    """The top view of one turn to ``<stem>.svg`` and ``<stem>.png`` (800 px wide: legible at 400): north is the
    incoming walking line, every path is drawn from its own position at the command.  Returns the caption."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon, Rectangle

    C, model = COLOURS, sv.model
    pl = sv.planned(cond)
    pose, t_c, t_end = pl["pose"], pl["t_c"], pl["t_end"]
    sign = float(pl["res"]["sign"])
    r = planned["readings"]
    ex = np.array([math.cos(model.psi_cmd), math.sin(model.psi_cmd)])
    ey = sign * np.array([-ex[1], ex[0]])
    o = np.array([np.interp(t_c, pose["t"], pose["root_xy"][:, 0]), np.interp(t_c, pose["t"], pose["root_xy"][:, 1])])

    def plot_xy(world_xy):  # picture x: to the robot's right at the command; picture y: along the incoming line
        a, s = (np.asarray(world_xy, float) - o) @ ex, (np.asarray(world_xy, float) - o) @ ey
        return np.column_stack([-sign * s, a])

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "svg.fonttype": "none",
                         "svg.hashsalt": "barviews"})  # (the same bytes on every run)
    words = condition_words(cond.describe())
    drawn = [] if sim is None else sim.get("_paths", [])
    if drawn:
        ctl = "the open-loop row " if bar.controller_of(sim["row"]) in bar.NO_VERDICT_CONTROLLERS else ""
        ms = sim["delay_ms"]
        delay = f"servo delay {ms} ms" + ("" if ms == bar.JUDGED_MS else f" (the judged delay is {bar.JUDGED_MS} ms)")
        skipped = ("  Rows before it with no path to draw: " + "; ".join(before) + ".") if before else ""
        note = (f"Simulated paths: {ctl}{sim['row']}, {delay}, floor grip {sim['floor_grip']:g}, {sim['block']}; "
                f"{sim['count_text']}; {len(drawn)} drawn." + skipped)
        simtext = (f"Orange, thin: SIMULATED root path{'s' if len(drawn) != 1 else ''} of the {len(drawn)} "
                   f"episode{'s' if len(drawn) != 1 else ''} that stayed up in "
                   f"`{sim['record']}` ({ctl}{sim['row']}, {delay}; {sim['count_text']})." + skipped)
    elif sim is not None:
        note = f"Simulated: no episode to draw from {sim['row']} ({sim.get('count_text') or sim.get('reason')})."
        simtext = f"SIMULATED: no episode of `{sim['record']}` stayed up with a trace that can be read; none is drawn."
    else:
        note = "Simulated: no record yet, so no simulated path is drawn."
        simtext = "SIMULATED: no record yet; no simulated path is drawn."
    foot = textwrap.wrap(f"{sv.name} ({SET_WORDS.get(sv.name, sv.name)}): {words}. Top view, up is the incoming "
                         "walking line. Every path is drawn from its own place at the command. " + note, 84)
    rob, box, ti = r["corner_box_robot"], r["corner_box_root"], r.get("turn_itself")
    ang = f"{math.degrees(float(pl['res']['angle_rad'])):.0f}°"
    after = "" if not ti or ti.get("root_to_move_end_m") is None else f" ({ti['root_to_move_end_m']:.2f} m of it after the turn)"
    handles = [plt.Line2D([], [], color=C["planned"], linewidth=2.0, label="planned: root path of the frozen reference"),
               plt.Line2D([], [], color=C["ink3"], linewidth=0.9, linestyle=(0, (1, 2)),
                          label=f"to the end of the move's file (planned root): {box['sideways_max_m']:.2f} m sideways"
                                + (",\n" + after.strip(" ()").replace("of it", "of it walk") if after else "")),
               Polygon([[0, 0]], facecolor=C["band"], edgecolor=C["ink3"], linewidth=0.5,
                       label="planned outline at the command and at the move's end")]
    if ti:
        handles.insert(1, plt.Line2D(
            [], [], color=C["ink"], linewidth=1.0, linestyle=(0, (4, 3)), marker="o", markersize=3.5,
            markerfacecolor=C["surface"], markeredgecolor=C["ink"],
            label=f"the turn itself (planned root): {ti['root_advance_m']:.2f} m along, {ti['root_sideways_m']:.2f} m "
                  "sideways"))
    if drawn:
        handles.insert(1, plt.Line2D([], [], color=C["simulated"], linewidth=0.9,
                                     label=f"simulated: root path{'s' if len(drawn) != 1 else ''} of {len(drawn)} "
                                           f"episode{'s' if len(drawn) != 1 else ''} that stayed up"))
    # the picture's extent: the planned robot's corner box (and the simulated paths), a margin, room for the bar
    xs = sorted([-sign * rob["sideways_min_m"], -sign * rob["sideways_max_m"]])
    ys = [rob["along_min_m"], rob["along_max_m"]]
    for path in drawn:
        xs = [min(xs[0], float((-sign * path[:, 1]).min())), max(xs[1], float((-sign * path[:, 1]).max()))]
        ys = [min(ys[0], float(path[:, 0].min())), max(ys[1], float(path[:, 0].max()))]
    pad = 0.10
    room = 0.42  # on the outside of the turn, for the labels
    xlim = (xs[0] - pad - (room if sign < 0 else 0.0), xs[1] + pad + (room if sign > 0 else 0.0))
    ylim = (ys[0] - pad - 0.26, ys[1] + pad)
    W_in, left = 4.0, 0.12
    ax_w = W_in - 2 * left
    ax_h = ax_w * (ylim[1] - ylim[0]) / (xlim[1] - xlim[0])
    legend_lines = sum(h.get_label().count("\n") + 1 for h in handles)
    below = 0.10 + 0.125 * len(foot) + 0.06 + 0.15 * legend_lines + 0.08
    H_in = ax_h + below + 0.06
    fig = plt.figure(figsize=(W_in, H_in), dpi=200, facecolor=C["surface"])
    ax = fig.add_axes([left / W_in, below / H_in, ax_w / W_in, ax_h / H_in], facecolor=C["surface"])
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    for when in (t_c, t_end):
        for poly in model.outline(q_at(pose, when)):
            ax.add_patch(Polygon(plot_xy(poly), closed=True, facecolor=C["band"], edgecolor=C["ink3"], linewidth=0.5,
                                 alpha=0.55, zorder=2))
    x0, x1 = sorted([sign * box["outside_max_m"], -sign * box["sideways_max_m"]])
    ax.add_patch(Rectangle((x0, -box["back_max_m"]), x1 - x0, box["advance_max_m"] + box["back_max_m"], fill=False,
                           edgecolor=C["ink3"], linewidth=0.9, linestyle=(0, (1, 2)), zorder=3))
    if ti:  # the turn itself: from the root's place at the command to its place when the trunk has turned the angle
        tx, ty = -sign * ti["root_sideways_m"], ti["root_advance_m"]
        ax.add_patch(Rectangle((min(0.0, tx), 0.0), abs(tx), ty, fill=False, edgecolor=C["ink"], linewidth=1.0,
                               linestyle=(0, (4, 3)), zorder=6))
    for path in drawn:
        ax.plot(-sign * path[:, 1], path[:, 0], color=C["simulated"], linewidth=0.7, alpha=0.75, zorder=4)
    sel = (pose["t"] >= t_c - LEAD_S) & (pose["t"] <= t_end + bar.EPS_S)
    pxy = plot_xy(pose["root_xy"][sel])
    ax.plot(pxy[:, 0], pxy[:, 1], color=C["planned"], linewidth=2.0, zorder=5, solid_capstyle="round")
    ax.plot([0], [0], marker="o", markersize=5, color=C["planned"], markeredgecolor=C["surface"], markeredgewidth=1.0,
            zorder=6)
    if ti:
        ax.plot([tx], [ty], marker="o", markersize=4.5, color=C["surface"], markeredgecolor=C["ink"],
                markeredgewidth=1.0, zorder=7)
    side_out = "left" if sign > 0 else "right"  # labels go on the outside of the turn, where the picture is free
    ax.annotate("command", (0, 0), xytext=(sign * 22, 0), textcoords="offset points", color=C["ink"], fontsize=7.5,
                ha=side_out, va="center", zorder=7)
    ax.annotate("planned path", pxy[0], xytext=(sign * 22, 0), textcoords="offset points", color=C["ink"],
                fontsize=7.5, ha=side_out, va="center", zorder=7)
    bx, by = (xlim[0] + 0.05 if sign > 0 else xlim[1] - 0.55), ylim[0] + 0.05
    ax.plot([bx, bx + 0.5], [by, by], color=C["ink"], linewidth=1.5, solid_capstyle="butt", zorder=8)
    ax.text(bx + 0.25, by + 0.03, "0.5 m", color=C["ink"], fontsize=7.5, ha="center", va="bottom", zorder=8)
    nx = bx + 0.80 if sign > 0 else bx - 0.30
    ax.annotate("", (nx, by + 0.24), xytext=(nx, by), arrowprops=dict(arrowstyle="-|>", color=C["ink2"], linewidth=0.8),
                zorder=8)
    ax.text(nx + sign * 0.04, by + 0.02, "incoming\nwalking line", color=C["ink2"], fontsize=6.5,
            ha="left" if sign > 0 else "right", va="bottom", linespacing=1.2, zorder=8)
    y_leg = (0.10 + 0.125 * len(foot) + 0.06) / H_in
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(left / W_in, y_leg), frameon=False, fontsize=6.8,
               labelcolor=C["ink"], handlelength=2.2, borderaxespad=0.0, labelspacing=0.35)
    fig.text(left / W_in, 0.08 / H_in, "\n".join(foot), color=C["ink2"], fontsize=6.0, ha="left", va="bottom",
             linespacing=1.3)
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(stem) + ".svg", facecolor=C["surface"], metadata={"Date": None})
    fig.savefig(str(stem) + ".png", facecolor=C["surface"], dpi=200, metadata={"Software": None})
    plt.close(fig)
    return (f"Top view of {words} on {sv.name} ({SET_WORDS.get(sv.name, sv.name)}); up is the incoming walking line. "
            "Blue, thick: the PLANNED path of the root (the frozen reference, kinematic), from 1 s before the command "
            "to the end of the move's file; the grey shapes are the PLANNED outline of the robot at the command and at "
            "the end of the move's file. "
            + (f"The dashed box is the turn itself: the PLANNED root is {ti['root_advance_m']:.2f} m along its "
               f"incoming line and {ti['root_sideways_m']:.2f} m sideways (the ring) when the trunk has turned {ang}, "
               f"{ti['seconds_after_command']:.2f} s after the command. " if ti else "")
            + f"The dotted box is the PLANNED root's corner box to the end of the move's file, "
              f"{box['seconds_after_command']:.2f} s after the command: {box['advance_max_m']:.2f} m along and "
              f"{box['sideways_max_m']:.2f} m sideways{after}; its sideways side is mostly the walk after the turn and "
              "depends on the file's length. " + simtext
            + " Every path is drawn from its own position at the command. The scale bar is 0.5 m.")


# ------------------------------------------------------------------------------------------------
# Clips: which episode, which seconds, which words
# ------------------------------------------------------------------------------------------------
#: The views of a behaviour's clips, and each view in words.
VIEWS_OF = {"walk": ("visitor", "side"), "turn": ("visitor", "top"), "nudge": ("visitor", "top"),
            "stop": ("visitor", "side")}
#: ``clips.CAMERAS``' eye height and distance along the floor (m) of the two following views (checked against
#: ``clips.CAMERAS`` when a clip is rendered).
VIEW_EYES = {"visitor": (1.5, 2.5), "side": (0.6, 2.5)}
#: Holding the robot in the free part of the picture (``CLIP_RULES["cameras"]``): the factors by which the eye may
#: be pulled back along its line of sight, and the heights (m) by which the look-at point may be raised (the label
#: bands and the captions take more of the picture's top than of its bottom).
PULL_BACK = tuple(round(1.0 + 0.1 * k, 1) for k in range(16))
LOOK_LIFT_M = tuple(round(0.05 * k, 2) for k in range(13))


def view_words(view: str, eye=None, pulled_back: bool = False) -> str:
    """A view in words; ``eye`` (height, distance along the floor, m) where it is not ``VIEW_EYES``'."""
    if view == "top":
        return "top view: straight down; up is the incoming walking line"
    h, d = eye or VIEW_EYES[view]
    name, where = {"visitor": ("visitor's view", "three-quarter front"), "side": ("side view", "on the robot's right")}[view]
    return (f"{name}: eye {h:.1f} m high, {d:.1f} m away" + (" (pulled back to hold the robot)" if pulled_back else "")
            + f", {where}")


VIEW_WORDS = {v: view_words(v) for v in ("visitor", "side", "top")}
#: The label's short word for how the episode was chosen (``SELECTION_RULE``; a pick's ``selection``).
SELECTION_SHORT = {"first seed": "first seed", "majority outcome": "majority outcome"}
#: An episode that fell before its command (or a walk that fell before its judged window): the clip is the last
#: seconds before the fall (a walk: up to ``WALK_CLIP_S``).
BEFORE_FALL_S = 4.0
CLIP_RULES = dict(
    selection=SELECTION_RULE,
    span=f"a clip starts {LEAD_S:g} s before the command and ends {TAIL_S:g} s after the end of the move (a turn: "
         "the end of its own file on the reference clock, read from the trace; a stop: the planned rest; a nudge: "
         "the pulse's end plus the bar's recovery limit) or at the trace's end if that comes first; a walk: the "
         f"first {WALK_CLIP_S:g} s of its judged window.  An episode that fell: the clip runs to the fall instant "
         f"the record gives and the last recorded pose is held for {TAIL_S:g} s under the caption 'fell' (a trace "
         "ends at the fall); if it fell before its command the clip is the last "
         f"{BEFORE_FALL_S:g} s before the fall; a walk that fell (it has no judged window): the last "
         f"{WALK_CLIP_S:g} s before the fall, from the episode's start if it is shorter",
    before="an episode that fell before its condition's own move began shows something else than its title: the "
           "start from standing or the change of speed, seconds before the turn, the stop or the push was due.  "
           "'Before' is read from the record: a turn or a stop, the episode's fell_before_command; a nudge, its "
           "push's fell_before_pulse; a walk, a segment_reached that is not the walk's own reference.  Such a clip "
           "says so on every frame, in two lines under the title ('THE TURN WAS NEVER COMMANDED: ...', with the "
           "fall's second, the part of its plan the episode had reached, and when the command was due: the "
           "command's reference time over the commanded clock rate, 'about', or a push's own onset), in the bottom "
           "line and in the fall's caption; its manifest entry has fell_before_its_move true and no command "
           "caption.  Episodes of different conditions with one seed are the same motion up to their commands: "
           "one fall in the start can be the clip of several records",
    time=f"{FPS} frames a second at real time: frame k shows the simulated time t0 + k / {FPS} s; between the 50 Hz "
         "rows the root position and the joint angles are interpolated linearly and the root's quaternion by "
         "normalised linear interpolation (clips.Playback's rule)",
    pose="simulated: pose_of on the episode's trace (the attitude and the root's place exact, the height from the "
         "feet's low points, the joint angles moved back one physics step); " + POSE_BOUND + ".  planned: the "
         "frozen reference played kinematically at the bar's commanded clock (exact)",
    cameras="visitor and side: clips.CAMERAS' eye height, distance and bearing, the bearing counted from the "
            "incoming walking line and kept for the whole clip; the look-at point follows the planned root path "
            "(the reference at the reference time the episode's own clock has reached, so that the clock's play "
            "does not move the picture), averaged over one stride, moved to the body's bounding box centre as it "
            "lies from the root at home and at that centre's height.  A simulated clip moves that path by one "
            "constant for the whole clip: the mean, over the clip's seconds, of the simulated root's place from "
            "its planned place (a robot that drifted off its plan before the clip is then still in the picture). "
            "Within a clip the cameras do not follow the body: not its wobble, and not a drift from the plan "
            "during the clip, which shows as the robot leaving the picture's centre.  The robot is held in the "
            "free part of the picture by two constants per clip: the look-at point is raised by the smallest of "
            f"{LOOK_LIFT_M[0]:g} to {LOOK_LIFT_M[-1]:g} m (steps of 0.05 m) and the eye pulled back along its line "
            f"of sight by the smallest factor of {PULL_BACK[0]:g} to {PULL_BACK[-1]:g} (steps of 0.1; the factor "
            "first, then the raise) that keep every surface point of the robot, on every third frame, inside "
            "the part of the picture the label bands leave free; a clip of an episode that fell: free of the "
            "fall's caption too.  The label gives the eye's height and distance when the eye was pulled back; if "
            "no pair holds the robot, the largest factor and its best raise are used and the command prints a "
            "warning.  top: a fixed eye straight above the middle of the clip's sweep, high "
            "enough to hold it, the picture's up along the incoming walking line, the headlight's diffuse part "
            f"lowered to {TOP_HEADLIGHT_DIFFUSE:g} (straight down the model's own washes the floor out)",
    labels=f"first line of every frame: '{SIM_LABEL}' or '{PLAN_LABEL}', never both, the body and the set; second "
           "line: the condition and the row in plain words; bottom: seed, how it was chosen ('first seed' or "
           "'majority outcome': the lowest seed with the majority outcome), this episode's outcome (a fall with "
           "its second of the episode), the record's count; then the view and the clock.  The clock of a simulated "
           "clip gives two times, 'clip 4.00 s = episode 8.22 s': the seconds since the clip's first frame and "
           "the episode's own time, the one a fall's second and the records use.  From the command, for "
           f"{CAPTION_S:g} s: 'command: ...' (read from the record); from the fall instant: 'fell'",
    watch="the clips are 854 x 480 and made for a landscape or a full screen.  The title's second line and the "
          "bottom's first line are 17 px tall, the other lines 13 to 15 px: at a phone's portrait width (390 px) "
          "they come out under 8 px, hard to read",
    media=f"854 x 480, H.264 through clips.encode, at most {MAX_BYTES} bytes each; a poster and a contact sheet "
          "(the frames at 1/8, 3/8, 5/8 and 7/8 of the clip) beside each; the media are written outside git and "
          "listed here with their sha256",
)


def _kept(e: dict) -> bool:
    tr = e.get("trace") or {}
    return bool(tr.get("path")) and tr.get("kept") is not False


def select_episodes(record: dict) -> dict:
    """``SELECTION_RULE`` on a record: ``dict(picks, skipped, majority, counts)``.  A pick is ``dict(seed, selection
    ("first seed" | "majority outcome"), words, episode)``; a skip ``dict(seed, selection, reason)``."""
    eps = sorted(record.get("episodes") or [], key=lambda e: e["seed"])
    seeds = [int(s) for s in (record.get("seeds") or [e["seed"] for e in eps])]
    by = {int(e["seed"]): e for e in eps}
    n_up = sum(outcome(e) == "stayed up" for e in eps)
    n_fell = sum(outcome(e) == "fell" for e in eps)
    first = by.get(seeds[0]) if seeds else None
    o1 = None if first is None else outcome(first)
    if n_up != n_fell:
        majority = "stayed up" if n_up > n_fell else "fell"
    else:
        majority = o1 if o1 in ("stayed up", "fell") else "stayed up"
    out = dict(picks=[], skipped=[], majority=majority,
               counts={"episodes": len(eps), "stayed up": n_up, "fell": n_fell, "failed to run": len(eps) - n_up - n_fell})

    def why_not(e):
        tr = e.get("trace") or {}
        return f"the trace was not kept ({tr.get('reason') or 'the record holds no path'})"

    if first is None:
        out["skipped"].append(dict(seed=seeds[0] if seeds else None, selection="first seed",
                                   reason="the record holds no episode of its first seed"))
    elif o1 == "failed to run":
        out["skipped"].append(dict(seed=seeds[0], selection="first seed",
                                   reason=f"the episode failed to run ({first.get('error')})"))
    elif not _kept(first):
        out["skipped"].append(dict(seed=seeds[0], selection="first seed", reason=why_not(first)))
    else:
        out["picks"].append(dict(seed=seeds[0], selection="first seed", words="the first seed", episode=first))
    if o1 != majority:
        same = [e for e in eps if outcome(e) == majority]
        kept = [e for e in same if _kept(e)]
        if kept:
            passed = [int(e["seed"]) for e in same if e["seed"] < kept[0]["seed"]]
            out["picks"].append(dict(seed=int(kept[0]["seed"]), selection="majority outcome", episode=kept[0],
                                     words="the lowest seed with the majority outcome", passed_over=passed))
        else:
            out["skipped"].append(dict(seed=int(same[0]["seed"]) if same else None, selection="majority outcome",
                                       reason="no episode with the majority outcome has a kept trace"
                                       if same else "no episode has the majority outcome"))
    return out


def push_words(force_xy, psi_cmd: float) -> str:
    """A nudge's direction in words from its force (N, world xy): 'to its left (from its right)'."""
    c, s = math.cos(psi_cmd), math.sin(psi_cmd)
    fwd, left = c * force_xy[0] + s * force_xy[1], -s * force_xy[0] + c * force_xy[1]
    if abs(fwd) >= abs(left):
        return "forward (from behind)" if fwd > 0 else "backward (from the front)"
    return "to its left (from its right)" if left > 0 else "to its right (from its left)"


def command_words(definition: dict, episode: dict | None = None, psi_cmd: float = 0.0) -> str | None:
    """The caption of a condition's command, from the record's definition (and, a nudge, the episode's pulse)."""
    beh = definition["behavior"]
    if beh == "turn":
        tn = setio.parse_name(definition["move"])
        return f"command: turn {tn['direction']} {tn['angle_deg']}°"
    if beh == "stop":
        return "command: stop"
    if beh == "nudge":
        push = ((episode or {}).get("measures") or {}).get("push") or {}
        if push.get("force_N") is None:
            return f"command: push {definition['direction']}"
        f = push["force_N"]
        return (f"command: push {push_words(f, psi_cmd)}, {math.hypot(f[0], f[1]):.0f} newtons for "
                f"{push['end_s'] - push['onset_s']:.2f} s")
    return None


def clip_span(definition: dict, t_last: float, command_s: float | None, move_end_s: float | None,
              window_start_s: float | None, fall_s: float | None) -> dict:
    """``CLIP_RULES["span"]``: ``dict(t0, t1, hold_s, command_s, fall_s, notes)`` in simulated seconds."""
    walk = definition["behavior"] == "walk"
    notes = []
    if fall_s is not None:
        t1, hold = float(fall_s), TAIL_S
        if walk or command_s is None or command_s > fall_s:
            t0 = max(0.0, t1 - (WALK_CLIP_S if walk else BEFORE_FALL_S))
            notes.append("fell: it has no judged window" if walk else "fell before the command")
            command_s = None if walk or command_s is None or command_s > fall_s else command_s
        else:
            t0 = max(0.0, command_s - LEAD_S)
    elif walk:
        if window_start_s is None:
            raise ViewError("a walk that did not fall has no judged window in its record")
        t0, hold = float(window_start_s), 0.0
        t1 = min(t0 + WALK_CLIP_S, t_last)
    else:
        if command_s is None:
            raise ViewError("no fall and no command instant in the record")
        t0, hold = max(0.0, command_s - LEAD_S), 0.0
        want = t_last if move_end_s is None else move_end_s + TAIL_S
        if move_end_s is None:
            notes.append("the trace ends before the end of the move: the clip ends with the trace")
        elif want > t_last + bar.EPS_S:
            notes.append(f"the trace ends {t_last - move_end_s:.2f} s after the end of the move: the clip ends there")
        t1 = min(want, t_last)
    return dict(t0=float(t0), t1=float(t1), hold_s=float(hold), command_s=command_s,
                fall_s=None if fall_s is None else float(fall_s), notes=notes)


def segment_words(name: str | None) -> str:
    """A part of an episode's plan (the record's ``segment_reached``) in plain words."""
    def walk(w):
        v = setio.parse_name(w).get("speed_m_s")
        return "the nominal walk" if v is None else f"{v:.2f} m/s"

    if not name:
        return "a part of its plan the record does not name"
    if name == "stand":
        return "the stand after the stop"
    try:
        p = setio.parse_name(name)
        if p["kind"] == "start":
            return "the start from standing" + (f" into the walk at {walk(p['walk'])}" if p.get("walk") else "")
        if p["kind"] == "change":
            return f"the change from {walk(p['a'])} to {walk(p['b'])}"
        if p["kind"] == "stop":
            return "the stop"
        if p["kind"] == "turn":
            return f"the turn to the {p['direction']}"
        return "the nominal walk" if p["speed_m_s"] is None else f"the walk at {p['speed_m_s']:.2f} m/s"
    except (ValueError, KeyError):
        return f"the part of its plan named {name}"


def fell_before(definition: dict, episode: dict) -> bool:
    """Whether the episode fell before its condition's own move began (``CLIP_RULES["before"]``)."""
    if outcome(episode) != "fell":
        return False
    beh = definition["behavior"]
    if beh == "walk":
        reached = episode.get("segment_reached")
        return reached is not None and reached != (definition.get("walk") or definition.get("gait"))
    if beh == "nudge":
        pulse = ((episode.get("measures") or {}).get("push") or {}).get("fell_before_pulse")
        if pulse is not None:
            return bool(pulse)
    if episode.get("fell_before_command") is not None:
        return bool(episode["fell_before_command"])
    return episode.get("command_s") is None or float(episode["command_s"]) > float(episode["fall_time_s"])


def before_words(definition: dict) -> str:
    """"before the command" (a turn, a stop), "before the push", "before the walk at 0.40 m/s began"."""
    beh = definition["behavior"]
    if beh == "walk":
        return f"before the {condition_words(definition)} began"
    return "before the push" if beh == "nudge" else "before the command"


def before_notice(definition: dict, episode: dict) -> list[str] | None:
    """The two lines every frame of a clip carries under its title when the episode fell before its condition's
    own move began (``CLIP_RULES["before"]``); None otherwise."""
    if not fell_before(definition, episode):
        return None
    beh = definition["behavior"]
    head = {"turn": "THE TURN WAS NEVER COMMANDED", "stop": "THE STOP WAS NEVER COMMANDED",
            "nudge": "THE PUSH NEVER CAME"}.get(beh) or f"THE {condition_words(definition).upper().replace('M/S', 'm/s')} NEVER BEGAN"
    line = (f"it fell {float(episode['fall_time_s']):.2f} s into the episode, in "
            f"{segment_words(episode.get('segment_reached'))}")
    push = (episode.get("measures") or {}).get("push") or {}
    if beh == "nudge" and push.get("onset_s") is not None:
        line += f"; the push was due at {float(push['onset_s']):.1f} s"
    elif beh in ("turn", "stop") and episode.get("command_ref_s") is not None:
        line += f"; the command was due at about {float(episode['command_ref_s']) / bar.RATE:.1f} s"
    return [f"{head}: this episode fell before it", line]


def fall_caption(definition: dict, episode: dict) -> str:
    """The caption from the fall instant on: 'fell (tilt); ...', and 'fell before the command (tilt); ...'."""
    when = f" {before_words(definition)}" if fell_before(definition, episode) else ""
    return f"fell{when} ({episode.get('fall_cause')}); the trace ends here, the last pose is held"


def clip_labels(kind: str, body: str, set_name: str, definition: dict, view: str, record: dict | None = None,
                pick: dict | None = None, eye=None, pulled_back: bool = False) -> tuple[list, list]:
    """A clip's label lines ``(top, bottom)`` (``CLIP_RULES["labels"]``); ``kind`` "simulated" or "planned";
    ``eye`` (height, distance, m) where the view's eye is not ``VIEW_EYES``'."""
    if kind not in ("simulated", "planned"):
        raise ValueError(kind)
    who = f"{body} ({BODY_WORDS[body]})" if body in BODY_WORDS else body
    setw = f"{SET_WORDS[set_name]} ({set_name})" if set_name in SET_WORDS else set_name
    cond = condition_words(definition)
    vw = view_words(view, eye, pulled_back)
    if kind == "planned":
        return ([f"{PLAN_LABEL}  |  {who}  |  {setw}",
                 f"{cond}  |  the frozen reference played at the bar's commanded clock ({bar.RATE:g} x)"],
                [vw])
    e = pick["episode"]
    oc = outcome(e)
    how = oc
    if oc == "fell":  # (the fall's second is the episode's own time: the clock gives both times)
        how = f"fell {e['fall_time_s']:.2f} s into the episode" + (
            f", {before_words(definition)}" if fell_before(definition, e) else "")
    return ([f"{SIM_LABEL}  |  {who}  |  {setw}", f"{cond}  |  {row_words(record)}"],
            [f"seed {pick['seed']} ({SELECTION_SHORT.get(pick['selection'], pick['words'])}): {how}  |  {count_words(record)}", vw])


# ------------------------------------------------------------------------------------------------
# Clips: cameras and frames
# ------------------------------------------------------------------------------------------------
def smooth_rows(x, rows: int) -> np.ndarray:
    """``x`` [n, k] averaged over a centred window of ``rows`` rows (shorter at the ends)."""
    x = np.asarray(x, float)
    w = max(1, min(int(rows), len(x)))
    if w <= 1:
        return x.copy()
    k = np.ones(w)
    cnt = np.convolve(np.ones(len(x)), k, mode="same")
    return np.column_stack([np.convolve(x[:, j], k, mode="same") / cnt for j in range(x.shape[1])])


def lookat_path(planned: dict, t, t_ref, stride_s: float, offset=(0.0, 0.0)) -> np.ndarray:
    """The look-at point's plan path [n, 2] at the rows ``t`` of a clip's pose series (``CLIP_RULES["cameras"]``):
    the planned root at the reference time ``t_ref`` of each row (carried on at its last second's mean velocity
    past the reference's end), averaged over one stride ``stride_s`` of simulated time, plus ``offset`` (forward,
    left: the body's bounding box centre from its root at home) turned with the planned heading, averaged
    likewise."""
    P = planned["pose"]
    tp, xy = P["t"] * bar.RATE, P["root_xy"]
    t_ref = np.asarray(t_ref, float)
    out = np.column_stack([np.interp(t_ref, tp, xy[:, 0]), np.interp(t_ref, tp, xy[:, 1])])
    late = t_ref > tp[-1]
    if late.any():
        k = int(np.searchsorted(tp, tp[-1] - bar.RATE))
        v = (xy[-1] - xy[k]) / (tp[-1] - tp[k])
        out[late] = xy[-1] + np.outer(t_ref[late] - tp[-1], v)
    t = np.asarray(t, float)
    step = float(np.median(np.diff(t))) if len(t) > 1 else 0.02
    rows = int(round(stride_s / step))
    psi = smooth_rows(np.interp(t_ref, tp, P["yaw"])[:, None], rows)[:, 0]
    c, s = np.cos(psi), np.sin(psi)
    return smooth_rows(out, rows) + np.column_stack([c * offset[0] - s * offset[1], s * offset[0] + c * offset[1]])


def camera_vectors(cam: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(eye, forward, right, up) of MuJoCo's free camera ``cam`` (lookat, distance, azimuth, elevation in deg)."""
    az, el = math.radians(cam["azimuth"]), math.radians(cam["elevation"])
    f = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
    u = np.array([-math.sin(el) * math.cos(az), -math.sin(el) * math.sin(az), math.cos(el)])
    return np.asarray(cam["lookat"], float) - cam["distance"] * f, f, np.cross(f, u), u


def project(points, cam: dict, size=(854, 480), fovy_deg: float = 26.0) -> np.ndarray:
    """Pixel coordinates (u right, v down) of world points seen by ``cam`` (also straight down, where
    ``clips.project`` has no horizon to stand on)."""
    eye, f, r, u = camera_vectors(cam)
    d = np.asarray(points, float) - eye
    k = 0.5 * size[1] / math.tan(math.radians(fovy_deg) / 2)
    zc = d @ f
    return np.column_stack([0.5 * size[0] + k * (d @ r) / zc, 0.5 * size[1] - k * (d @ u) / zc])


def follow_camera(view: str, lookat_xy, z: float, psi: float, spec: dict, back: float = 1.0) -> dict:
    """The visitor's or the side camera on a look-at point: ``spec`` is ``clips.CAMERAS[view]``, its bearing
    counted from the incoming walking line ``psi``; ``back`` moves the eye along its line of sight to that many
    times its distance from the look-at point."""
    b = math.radians(spec["bearing_deg"]) + psi
    look = np.array([lookat_xy[0], lookat_xy[1], z])
    eye = np.array([look[0] + spec["distance_m"] * math.cos(b), look[1] + spec["distance_m"] * math.sin(b),
                    spec["height_m"]])
    eye = look + back * (eye - look)
    d = look - eye
    return dict(lookat=look, distance=float(np.linalg.norm(d)), azimuth=math.degrees(math.atan2(d[1], d[0])),
                elevation=math.degrees(math.atan2(d[2], math.hypot(d[0], d[1]))), eye=eye)


def top_camera(points_xy, z: float, psi: float, size=(854, 480), bands=(48, 44), fovy_deg: float = 26.0,
               margin_m: float = 0.15) -> dict:
    """The fixed top camera of a clip: straight down, the picture's up along ``psi``, centred on the sweep
    ``points_xy`` [n, 2] in the part of the picture the label bands leave free, and high enough to hold it with
    ``margin_m`` to spare (never lower than ``TOP_MIN_HEIGHT_M``)."""
    ex = np.array([math.cos(psi), math.sin(psi)])
    ey = np.array([-ex[1], ex[0]])
    p = np.asarray(points_xy, float)
    a, s = p @ ex, p @ ey
    ha, hs = 0.5 * (a.max() - a.min()) + margin_m, 0.5 * (s.max() - s.min()) + margin_m
    k = 0.5 * size[1] / math.tan(math.radians(fovy_deg) / 2)
    free = 0.5 * (size[1] - bands[0] - bands[1])
    dist = max(ha * k / free, hs * k / (0.5 * size[0]), TOP_MIN_HEIGHT_M - z)
    shift = 0.5 * (bands[0] - bands[1]) * dist / k  # the free part's middle lies below the picture's middle
    c = 0.5 * (a.max() + a.min()) * ex + 0.5 * (s.max() + s.min()) * ey + shift * ex
    return dict(lookat=np.array([c[0], c[1], z]), distance=float(dist), azimuth=math.degrees(psi), elevation=-90.0,
                eye=np.array([c[0], c[1], z + dist]))


def renderer_env(env=None) -> dict:
    """Keep the software renderer from writing outside the output directories: Mesa keeps a shader cache under the
    home directory (``~/.cache/mesa_shader_cache_db``) unless it is told not to.  Sets ``MESA_SHADER_CACHE_DISABLE``
    (and the older ``MESA_GLSL_CACHE_DISABLE``) to "true" in ``env`` (default: this process's) where they are not
    set; called before the first GL context is made."""
    env = os.environ if env is None else env
    for name in ("MESA_SHADER_CACHE_DISABLE", "MESA_GLSL_CACHE_DISABLE"):
        env.setdefault(name, "true")
    return env


def _clips():
    """``designlab.stage2.clips`` (imported late: it selects a GL backend), after a check that one is there."""
    if os.environ.get("MUJOCO_GL") not in ("osmesa", "egl"):
        raise ViewError("rendering needs MUJOCO_GL=osmesa (and LP_NUM_THREADS=1); it is "
                        f"{os.environ.get('MUJOCO_GL')!r}")
    renderer_env()
    from . import clips

    return clips


_SCENES: dict = {}


def scene(body: str):
    """``clips.ClipScene`` of the body (one per process), checked against the plant's joint layout."""
    if body not in _SCENES:
        cs = _clips().ClipScene(body)
        m, pm = cs.m, body_model(body).m
        names = [[mujoco.mj_id2name(x, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(x.njnt)] for x in (m, pm)]
        if m.nq != pm.nq or names[0] != names[1]:
            raise ViewError(f"{body}: the rendered model's joints are not the plant's")
        _SCENES[body] = cs
    return _SCENES[body]


def draw_frame(cs, q, cam: dict, travel_xy, headlight_diffuse: float | None = None) -> np.ndarray:
    """One covered frame of ``clips.ClipScene`` at pose ``q`` from ``cam``; the key light travels with the cameras
    (``clips.ClipScene.frame``'s rule, in the plane) and the sky is ``clips.SKY`` where the view passes the floor.
    ``headlight_diffuse`` replaces the headlight's diffuse part for this frame (``TOP_HEADLIGHT_DIFFUSE``)."""
    clips = _clips()
    m = cs.m
    if headlight_diffuse is not None:
        keep = m.vis.headlight.diffuse.copy()
        m.vis.headlight.diffuse[:] = headlight_diffuse
        try:
            return draw_frame(cs, q, cam, travel_xy)
        finally:
            m.vis.headlight.diffuse[:] = keep
    m.light_pos[:] = cs.light_pos0
    m.light_pos[:, 0] += travel_xy[0]
    m.light_pos[:, 1] += travel_xy[1]
    cs.pose(q)
    m.vis.global_.orthographic = 0
    m.vis.global_.fovy = clips.FOVY_DEG
    c = mujoco.MjvCamera()
    c.type = mujoco.mjtCamera.mjCAMERA_FREE
    c.lookat[:] = cam["lookat"]
    c.distance, c.azimuth, c.elevation = cam["distance"], cam["azimuth"], cam["elevation"]
    r = cs.sc.r
    r.update_scene(cs.d, c, cs.opt_covered)
    img = r.render().copy()
    if cam["elevation"] + 0.5 * clips.FOVY_DEG > 0.0:
        r.enable_depth_rendering()
        r.update_scene(cs.d, c, cs.opt_covered)
        depth = r.render()
        r.disable_depth_rendering()
        img[depth >= 0.999 * cs.far] = clips.SKY
    return img


#: The captions' colours (``page_kit``'s series 1 and "bad"); a caption box is ``CAPTION_BOX_PX`` tall, the first
#: ``CAPTION_GAP_PX`` under the top band, the next ``CAPTION_STEP_PX`` lower.
CAPTION_COLOURS = {"command": (42, 120, 214), "fell": (162, 57, 42)}
CAPTION_BOX_PX, CAPTION_GAP_PX, CAPTION_STEP_PX = 30, 8, 36
#: The label lines' sizes (px) and row heights (px): the top band's first line (bold) and its others, the notice's
#: first line (bold) and its others, the bottom band's first line and its others (the last shares its row with
#: the clock).  ``clips.overlay`` writes every line but the first at 13 px; here the lines a viewer needs most
#: (the condition and the row; the seed, the outcome and the count; the notice) are larger.
LABEL_PX = dict(top=((15, 19), (17, 21)), notice=((16, 21), (14, 18)), bottom=((17, 21), (13, 17)), clock=14)
BAND_PAD_PX = (8, 6)  # (the top band's, the bottom band's: 48 and 44 px for two lines each)


def band_heights(top: list, bottom: list, notice: list | None = None) -> tuple[int, int]:
    """(the top band's height, the bottom band's) in px for these label lines (``draw_labels``)."""
    T, N, B = LABEL_PX["top"], LABEL_PX["notice"], LABEL_PX["bottom"]
    th = BAND_PAD_PX[0] + T[0][1] + T[1][1] * (len(top) - 1)
    if notice:
        th += 4 + N[0][1] + N[1][1] * (len(notice) - 1)
    return th, BAND_PAD_PX[1] + B[0][1] + B[1][1] * (len(bottom) - 1)


def free_top(band_px: int, captions: int) -> int:
    """The first picture row (px from the top) that neither the top band nor ``captions`` caption boxes cover."""
    return band_px if not captions else band_px + CAPTION_GAP_PX + CAPTION_STEP_PX * (captions - 1) + CAPTION_BOX_PX


def clock_words(t_clip: float, t_episode: float | None = None) -> str:
    """The clock of a frame: the clip's own time, and for a simulated clip the episode's time beside it."""
    if t_episode is None:
        return f"clip {t_clip:5.2f} s   1x"
    return f"clip {t_clip:5.2f} s = episode {t_episode:5.2f} s   1x"


def _fit_font(dr, text: str, size: int, bold: bool, width: float, smallest: int = 10):
    from .. import render

    font = render.font(size, bold=bold)
    while dr.textlength(text, font=font) > width and size > smallest:
        size -= 1
        font = render.font(size, bold=bold)
    return font


def draw_labels(img: np.ndarray, top: list, bottom: list, clock: str, notice: list | None = None) -> np.ndarray:
    """``img`` with the label bands (``CLIP_RULES["labels"]``, ``LABEL_PX``): ``top`` lines top left (the first
    bold), under them the ``notice`` lines beside a red bar, ``bottom`` lines bottom left and ``clock`` bottom
    right; dark ink on light bands."""
    from PIL import Image, ImageDraw

    im = Image.fromarray(img).convert("RGBA")
    W, H = im.size
    th, bh = band_heights(top, bottom, notice)
    im.alpha_composite(Image.new("RGBA", (W, th), (252, 252, 251, 205)))
    im.alpha_composite(Image.new("RGBA", (W, bh), (252, 252, 251, 205)), (0, H - bh))
    layer = Image.new("RGBA", im.size, (0, 0, 0, 0))
    dr = ImageDraw.Draw(layer)
    ink, ink2 = (11, 11, 11, 255), (58, 58, 55, 255)
    y = 5
    for k, text in enumerate(top):
        size, row = LABEL_PX["top"][min(k, 1)]
        dr.text((12, y), text, fill=ink if k == 0 else ink2, font=_fit_font(dr, text, size, k == 0, W - 24))
        y += row
    if notice:
        y += 3
        rows = [LABEL_PX["notice"][min(k, 1)] for k in range(len(notice))]
        dr.rectangle([12, y + 1, 17, y + sum(r for _, r in rows) - 4], fill=CAPTION_COLOURS["fell"] + (255,))
        for text, (size, row), bold in zip(notice, rows, [True] + [False] * len(notice), strict=False):
            dr.text((26, y), text, fill=ink, font=_fit_font(dr, text, size, bold, W - 38))
            y += row
    fc = _fit_font(dr, clock, LABEL_PX["clock"], False, W)
    cw = dr.textlength(clock, font=fc)
    y = H - bh + 4
    for k, text in enumerate(bottom):
        size, row = LABEL_PX["bottom"][min(k, 1)]
        last = k == len(bottom) - 1  # the last line shares its row with the clock
        dr.text((12, y), text, fill=ink2, font=_fit_font(dr, text, size, False, W - 24 - (cw + 16 if last else 0)))
        if last:
            dr.text((W - cw - 10, y - (1 if len(bottom) > 1 else -2)), clock, fill=ink2, font=fc)
        y += row
    im.alpha_composite(layer)
    return np.asarray(im.convert("RGB"))


def draw_caption(img: np.ndarray, text: str, kind: str, y: int = 54) -> np.ndarray:
    """``img`` with a caption box whose top is at ``y``: a coloured bar, then ``text`` in bold ink on a light box."""
    from PIL import Image, ImageDraw

    im = Image.fromarray(img).convert("RGBA")
    layer = Image.new("RGBA", im.size, (0, 0, 0, 0))
    dr = ImageDraw.Draw(layer)
    f = _fit_font(dr, text, 17, True, im.size[0] - 60, smallest=11)
    w = int(dr.textlength(text, font=f))
    dr.rectangle([12, y, 12 + w + 26, y + CAPTION_BOX_PX], fill=(252, 252, 251, 235))
    dr.rectangle([12, y, 18, y + CAPTION_BOX_PX], fill=CAPTION_COLOURS[kind] + (255,))
    dr.text((27, y + 5), text, fill=(11, 11, 11, 255), font=f)
    im.alpha_composite(layer)
    return np.asarray(im.convert("RGB"))


def frame_captions(t: float, span: dict, command: str | None, fall_words: str | None) -> list[tuple[str, str]]:
    """The captions on the frame at simulated time ``t``: ``[(kind, text)]`` (``CLIP_RULES["labels"]``)."""
    out = []
    tc, tf = span.get("command_s"), span.get("fall_s")
    if command and tc is not None and tc - 1e-9 <= t < tc + CAPTION_S and (tf is None or t < tf):
        out.append(("command", command))
    if tf is not None and t >= tf - 1e-9:
        out.append(("fell", fall_words or "fell"))
    return out


def clip_framing(surfaces: list, cam_of, bands: tuple, captions: list, size=(854, 480), fovy_deg: float = 26.0) -> float:
    """How far (px; negative: outside) the robot's surface stays inside the free part of the picture over a clip:
    ``surfaces`` [k][N, 3] on the sampled frames, ``cam_of(k)`` each one's camera, ``bands`` the two label bands'
    heights and ``captions`` [k] the number of caption boxes on each sampled frame."""
    m = math.inf
    for k, surf in enumerate(surfaces):
        uv = project(surf, cam_of(k), size, fovy_deg)
        m = min(m, float(uv[:, 0].min()), float(size[0] - uv[:, 0].max()),
                float(uv[:, 1].min() - free_top(bands[0], captions[k])), float(size[1] - bands[1] - uv[:, 1].max()))
    return m


def hold_in_picture(margin_of, backs=PULL_BACK, lifts=LOOK_LIFT_M) -> tuple[float, float, float]:
    """(pull-back factor, raise of the look-at point in m, the margin there): the first of ``backs``, and for it
    the first of ``lifts``, at which ``margin_of(back, lift)`` is not negative; none: the last factor and its best
    raise (``CLIP_RULES["cameras"]``)."""
    best = None
    for back in backs:
        best = None
        for lift in lifts:
            m = margin_of(back, lift)
            if m >= 0.0:
                return float(back), float(lift), float(m)
            if best is None or m > best[2]:
                best = (float(back), float(lift), float(m))
    return best


def render_clip(kind: str, sv: SetView, cond, pose: dict, span: dict, view: str, labels, command: str | None,
                fall_words: str | None, stem: Path, meta: dict, notice: list | None = None) -> dict:
    """Render one clip to ``<stem>.mp4`` with ``<stem>.png`` (poster) and ``<stem>_sheet.png``; returns what the
    manifest holds of the picture (frames, camera, framing margin, files, bytes, sha256).  ``labels(eye,
    pulled_back)`` gives the label lines ``(top, bottom)`` (``clip_labels``; ``eye`` None: the view's own);
    ``notice`` the lines under the title on every frame (``before_notice``)."""
    clips = _clips()
    from PIL import Image

    for name, (h, d) in VIEW_EYES.items():
        if (clips.CAMERAS[name]["height_m"], clips.CAMERAS[name]["distance_m"]) != (h, d):
            raise ViewError(f"clips.CAMERAS[{name!r}] is not the eye VIEW_EYES states")
    model, cs = sv.model, scene(sv.body)
    pl = sv.planned(cond)
    stride = float(pl["res"]["period_s"]) / bar.RATE
    box, _ = cs.home_box(model.q_home)
    z, psi = float(box[2]), model.psi_cmd
    d_home = box[:2] - model.q_home[:2]  # the bounding box centre from the root at home, in the heading's frame
    off = (math.cos(psi) * d_home[0] + math.sin(psi) * d_home[1], -math.sin(psi) * d_home[0] + math.cos(psi) * d_home[1])
    look = lookat_path(pl, pose["t"], pose["t_ref"], stride, off)
    shift = np.zeros(2)
    if kind == "simulated":  # one constant per clip: where the robot is from its planned place, on average
        P = pl["pose"]
        sel = (pose["t"] >= span["t0"] - 1e-9) & (pose["t"] <= span["t1"] + 1e-9)
        tp = P["t"] * bar.RATE
        plan_xy = np.column_stack([np.interp(pose["t_ref"][sel], tp, P["root_xy"][:, j]) for j in (0, 1)])
        shift = np.mean(pose["root_xy"][sel] - plan_xy, axis=0) if sel.any() else shift
        look = look + shift
    n = int(round((span["t1"] - span["t0"] + span["hold_s"]) * FPS)) + 1
    taus = np.arange(n) / FPS
    t_show = np.minimum(span["t0"] + taus, span["t1"])  # (the pose is held after a fall)
    fell = span.get("fall_s") is not None
    top, bottom = labels(None, False)
    bands = band_heights(top, bottom, notice)
    caps = [frame_captions(span["t0"] + float(tau), span, command, fall_words) for tau in taus]
    cam_fixed = None
    if view == "top":  # (a clip with a fall: the sweep is held under the fall's caption too)
        rows = np.flatnonzero((pose["t"] >= span["t0"] - 0.03) & (pose["t"] <= span["t1"] + 0.03))[::3]
        cam_fixed = top_camera(np.concatenate([model.surface(pose["qpos"][i])[:, :2] for i in rows]), z, psi,
                               bands=(free_top(bands[0], 1 if fell else 0), bands[1]))
    elif view not in clips.CAMERAS:
        raise ViewError(f"view {view!r}: choose from visitor, side, top")
    lxy = np.column_stack([np.interp(t_show, pose["t"], look[:, 0]), np.interp(t_show, pose["t"], look[:, 1])])

    def cam_of(k: int, back: float = 1.0, lift: float = 0.0) -> dict:
        if cam_fixed is not None:
            return cam_fixed
        return follow_camera(view, lxy[k], z + lift, psi, clips.CAMERAS[view], back)

    # the framing, on every third frame: how far the robot's surface stays inside the free part of the picture
    ks = list(range(0, n, 3))
    surfaces = [model.surface(q_at(pose, float(t_show[k]))) for k in ks]

    def margin_of(back: float, lift: float, with_captions: bool) -> float:
        return clip_framing(surfaces, lambda j: cam_of(ks[j], back, lift), bands,
                            [len(caps[k]) if with_captions else 0 for k in ks], clips.SIZE, clips.FOVY_DEG)

    back, lift = 1.0, 0.0
    if cam_fixed is None:  # hold the robot in the free picture (CLIP_RULES["cameras"]); a fall: under its caption
        back, lift, _ = hold_in_picture(lambda b, h: margin_of(b, h, fell))
    margin, margin_captions = margin_of(back, lift, False), margin_of(back, lift, True)
    eye = None
    if cam_fixed is None and back != 1.0:
        e0 = cam_of(0, back, lift)
        eye = (float(e0["eye"][2]), float(np.hypot(*(e0["eye"][:2] - e0["lookat"][:2]))))
        top, bottom = labels(eye, True)
    away = max(float(np.linalg.norm(0.5 * (sf[:, :2].min(axis=0) + sf[:, :2].max(axis=0)) - lxy[k]))
               for k, sf in zip(ks, surfaces, strict=True))
    frames = np.empty((n, clips.SIZE[1], clips.SIZE[0], 3), np.uint8)
    t0w = time.perf_counter()
    try:
        for k, tau in enumerate(taus):
            t = float(t_show[k])
            cam = cam_of(k, back, lift)
            travel = (lxy[k] if cam_fixed is None else cam_fixed["lookat"][:2]) - box[:2]
            raw = draw_frame(cs, q_at(pose, t), cam, travel, TOP_HEADLIGHT_DIFFUSE if cam_fixed is not None else None)
            clock = clock_words(float(tau), t if kind == "simulated" else None)
            img = draw_labels(raw, top, bottom, clock, notice)
            for j, (ckind, text) in enumerate(caps[k]):
                img = draw_caption(img, text, ckind, y=bands[0] + CAPTION_GAP_PX + CAPTION_STEP_PX * j)
            frames[k] = img
    finally:
        cs.m.light_pos[:] = cs.light_pos0
    render_s = time.perf_counter() - t0w
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    mp4 = Path(str(stem) + ".mp4")
    t1w = time.perf_counter()
    enc = clips.encode(frames, mp4, FPS, MAX_BYTES, meta=meta)
    encode_s = time.perf_counter() - t1w
    tc = span.get("command_s")
    if tc is not None and span["t0"] <= tc <= span["t1"]:
        poster_tau = min(tc + 0.5 * CAPTION_S - span["t0"], taus[-1])
    elif fell:
        poster_tau = max(0.0, taus[-1] - span["hold_s"])
    else:
        poster_tau = 0.5 * taus[-1]
    Image.fromarray(frames[int(round(poster_tau * FPS))]).save(str(stem) + ".png")
    sheet_tau = [taus[-1] * x for x in (0.125, 0.375, 0.625, 0.875)]
    clips.contact_sheet({key: frames[int(round(tt * FPS))] for key, tt in zip(clips.SHEET_T, sheet_tau, strict=True)},
                        Path(str(stem) + "_sheet.png"))
    cam_doc = (dict(kind="fixed", lookat=cam_fixed["lookat"].tolist(), height_above_lookat_m=cam_fixed["distance"],
                    up_deg=math.degrees(psi)) if cam_fixed is not None
               else dict(clips.CAMERAS[view], kind="follows the planned root path, averaged over one stride",
                         lookat_height_m=z, lookat_offset_from_root_m=[float(off[0]), float(off[1])],
                         shift_from_plan_m=[float(shift[0]), float(shift[1])], pulled_back=float(back),
                         lookat_raised_m=float(lift),
                         eye_height_m=VIEW_EYES[view][0] if eye is None else eye[0],
                         eye_distance_m=VIEW_EYES[view][1] if eye is None else eye[1],
                         largest_distance_box_centre_to_lookat_m=away))
    return dict(file=mp4.name, poster=mp4.stem + ".png", sheet=mp4.stem + "_sheet.png", view=view, kind=kind,
                bytes=enc["bytes"], sha256=hashlib.sha256(mp4.read_bytes()).hexdigest(), frames=int(n), fps=FPS,
                seconds=float(taus[-1]), size=list(clips.SIZE), labels=dict(top=top, bottom=bottom, notice=notice),
                bands_px=list(bands), captions=dict(command=command, fell=fall_words), span=span,
                poster_at_s=float(poster_tau), sheet_at_s=[float(x) for x in sheet_tau],
                camera=dict(cam_doc, fovy_deg=clips.FOVY_DEG), framing_margin_px=float(margin),
                framing_margin_under_captions_px=float(margin_captions),
                encode=dict(enc["settings"], tries=enc["tries"]), probe=clips.probe(mp4),
                render_s=render_s, encode_s=encode_s)


def clip_log(ent: dict) -> str:
    """One line for the person rendering: the file, its size and times, the framing; a warning if the robot leaves
    the free part of the picture."""
    m = ent["framing_margin_px"]
    line = (f"  {ent['file']}: {ent['frames']} frames, {ent['bytes'] / 1e6:.2f} MB, render {ent['render_s']:.0f} s, "
            f"encode {ent['encode_s']:.0f} s, margin {m:.0f} px"
            + (f", eye pulled back {ent['camera']['pulled_back']:g} x" if ent["camera"].get("pulled_back", 1.0) != 1.0 else "")
            + (f", look-at raised {ent['camera']['lookat_raised_m']:g} m" if ent["camera"].get("lookat_raised_m") else ""))
    if m < 0:
        line += f"\n  WARNING: the robot leaves the free picture by {-m:.0f} px in {ent['file']}"
    return line


def _slug(text: str) -> str:
    return text.replace("/", "-").replace("@", "_")


def simulated_clips(run, set_name: str, ckey: str, row: str, media: Path, block: str = bar.HEADLINE_BLOCK,
                    views=None, body: str = BODY, log=print) -> tuple[list, list]:
    """The clips of one record (``SELECTION_RULE``, ``CLIP_RULES``): ``(entries, skipped)`` for the manifest."""
    path = record_path(run, set_name, body, block, row, ckey)
    rel = path.relative_to(run).as_posix()
    item = dict(set=set_name, condition=ckey, row=row, block=block, record=rel)
    rec, why = read_record(path)
    if rec is None:
        return [], [dict(item, reason=f"the record {why}" if path.exists() else "no record yet")]
    if rec.get("status") != "done":
        return [], [dict(item, reason=f"the record's status is {rec.get('status')!r}: {rec.get('reason')}")]
    sv = set_view(set_name, body)
    differ = sv.inputs_differ(rec)
    if differ:
        return [], [dict(item, reason="the record was made from other files than this checkout's set: " + ", ".join(differ))]
    definition = rec["definition"]
    cond = sv.condition(ckey)
    sel = select_episodes(rec)
    entries = []
    skipped = [dict(item, seed=x["seed"], selection=x["selection"], reason=x["reason"]) for x in sel["skipped"]]
    for pick in sel["picks"]:
        e = pick["episode"]
        trace, why = read_trace(run, rec, e)
        if trace is None:
            skipped.append(dict(item, seed=pick["seed"], selection=pick["selection"], reason=why))
            continue
        pose = pose_of(trace, sv.model)
        b = (e.get("measures") or {}).get("bar") or {}
        push = (e.get("measures") or {}).get("push") or {}
        beh = definition["behavior"]
        command_s = push.get("onset_s") if beh == "nudge" else e.get("command_s")
        move_end = None
        if beh == "turn":
            end_ref = sv.move_end_ref_s(rec["plan"], definition["move"])
            if end_ref is not None and len(pose["t_ref"]) and end_ref <= pose["t_ref"][-1] + 1e-12:
                move_end = float(np.interp(end_ref, pose["t_ref"], pose["t"]))
        elif beh == "stop":
            move_end = b.get("rest_s")
        elif beh == "nudge" and push.get("end_s") is not None:
            move_end = float(push["end_s"]) + float(definition.get("recovery_s") or bar.RECOVERY_S)
        win = (e.get("measures") or {}).get("window") or {}
        fell = outcome(e) == "fell"
        span = clip_span(definition, float(pose["t"][-1]), command_s, move_end, win.get("t_start_s"),
                         float(e["fall_time_s"]) if fell else None)
        # the command's caption only where the clip holds the command (an episode that fell before it has none)
        command = None if span["command_s"] is None else command_words(definition, e, sv.model.psi_cmd)
        before = fell_before(definition, e)
        notice = before_notice(definition, e)
        fall_words = fall_caption(definition, e) if fell else None
        for view in (views or VIEWS_OF[beh]):
            def labels(eye, pulled_back, view=view):
                return clip_labels("simulated", body, set_name, definition, view, rec, pick, eye, pulled_back)

            stem = Path(media) / set_name / f"{body}_{set_name}_{_slug(ckey)}_{_slug(row)}_{block}_seed{pick['seed']}_{view}"
            ent = render_clip("simulated", sv, cond, pose, span, view, labels, command, fall_words, stem,
                              dict(title=f"{body} {set_name} {ckey} {row} seed {pick['seed']} {view}",
                                   comment=f"{SIM_LABEL}; designlab/stage2/barviews.py; trace sha256 {e['trace']['sha256']}"),
                              notice)
            ent.update(item, file=f"{set_name}/{ent['file']}", poster=f"{set_name}/{ent['poster']}",
                       sheet=f"{set_name}/{ent['sheet']}", seed=pick["seed"], selection=pick["selection"],
                       selection_words=pick["words"], passed_over=pick.get("passed_over", []),
                       fell_before_its_move=bool(before), segment_reached=e.get("segment_reached"),
                       outcome=outcome(e), record_count=count_words(rec), majority_outcome=sel["majority"],
                       trace=e["trace"].get("path"), trace_sha256=e["trace"]["sha256"],
                       record_key_sha256=rec.get("key_sha256"), pose_residual_m=pose["residual_m"],
                       in_words=condition_words(definition), row_in_words=row_words(rec))
            entries.append(ent)
            log(clip_log(ent))
    return entries, skipped


def planned_clips(set_name: str, ckey: str, media: Path, views=None, body: str = BODY, log=print) -> tuple[list, list]:
    """The planned clips of one condition: the frozen reference alone, the same views and seconds."""
    sv = set_view(set_name, body)
    item = dict(set=set_name, condition=ckey, row=None, block=None, record=None)
    cond = sv.condition(ckey)
    why = sv.planner.missing(cond)
    if cond.judge == "bar_nudge":
        return [], [dict(item, reason="a nudge has no reference of its own: its planned motion is the walk's")]
    if why is not None:
        return [], [dict(item, reason=f"no reference: {why['reason']}")]
    pl = sv.planned(cond)
    definition = cond.describe()
    pose = pl["pose"]
    span = clip_span(definition, float(pose["t"][-1]), pl["t_c"], pl["t_end"],
                     max(0.0, pl["seconds"] - WALK_CLIP_S) if cond.behavior == "walk" else None, None)
    entries = []
    for view in (views or VIEWS_OF[cond.behavior]):
        def labels(eye, pulled_back, view=view):
            return clip_labels("planned", body, set_name, definition, view, None, None, eye, pulled_back)

        stem = Path(media) / set_name / f"{body}_{set_name}_{_slug(ckey)}_planned_{view}"
        ent = render_clip("planned", sv, cond, pose, span, view, labels, command_words(definition), None, stem,
                          dict(title=f"{body} {set_name} {ckey} planned {view}",
                               comment=f"{PLAN_LABEL}; designlab/stage2/barviews.py"))
        ent.update(item, file=f"{set_name}/{ent['file']}", poster=f"{set_name}/{ent['poster']}",
                   sheet=f"{set_name}/{ent['sheet']}", seed=None, selection=None, outcome=None,
                   reference_files=pl["files"], set_freeze=pl["freeze"], in_words=condition_words(definition))
        entries.append(ent)
        log(clip_log(ent))
    return entries, []


def gated_items(row: str, sets=SETS, body: str = BODY) -> list[tuple]:
    """``--all-gated --row ROW``: (set, condition, row) of every gated condition of each set."""
    return [(s, c.key, row) for s in sets for c in set_view(s, body).conditions() if c.gated]


def list_clips(items, run=RUN, block: str = bar.HEADLINE_BLOCK, views=None, body: str = BODY, log=print) -> list[dict]:
    """What ``barviews clips`` would render of ``items``, without rendering or writing: per record its count, the
    episodes ``SELECTION_RULE`` picks and the views, or why there would be no clip."""
    out = []
    for set_name, ckey, row in items:
        path = record_path(run, set_name, body, block, row, ckey)
        row_out = dict(set=set_name, condition=ckey, row=row, block=block, record=path.relative_to(run).as_posix(), clips=[])
        out.append(row_out)
        rec, why = read_record(path)
        if rec is None or rec.get("status") != "done":
            row_out["reason"] = ("no record yet" if not path.exists() else f"the record {why}" if rec is None
                                 else f"the record's status is {rec.get('status')!r}")
            log(f"{set_name} {ckey} {row}: no clip ({row_out['reason']})")
            continue
        sel = select_episodes(rec)
        vs = tuple(views or VIEWS_OF[rec["definition"]["behavior"]])
        row_out.update(count=count_words(rec), majority=sel["majority"], views=list(vs), skipped=sel["skipped"])
        for pick in sel["picks"]:
            trace, why = read_trace(run, rec, pick["episode"])
            oc = outcome(pick["episode"])
            if fell_before(rec["definition"], pick["episode"]):
                oc += f" {before_words(rec['definition'])}"
            row_out["clips"].append(dict(seed=pick["seed"], selection=pick["selection"], outcome=oc,
                                         fell_before_its_move=fell_before(rec["definition"], pick["episode"]),
                                         trace=None if trace is not None else why))
        log(f"{set_name} {ckey} {row}: {row_out['count']}; "
            + ("; ".join(f"seed {c['seed']} ({c['selection']}, {c['outcome']})" + (f" NO CLIP: {c['trace']}" if c["trace"] else "")
                         for c in row_out["clips"]) or "no episode to render")
            + f"; views {', '.join(vs)}" + "".join(f"; skipped seed {x['seed']}: {x['reason']}" for x in sel["skipped"]))
    return out


def make_clips(items, planned: bool = False, run=RUN, media=None, out=None, block: str = bar.HEADLINE_BLOCK,
               views=None, body: str = BODY, log=print) -> dict:
    """``barviews clips``: render ``items`` and write ``<out>/clips.json`` (the manifest; the clips of earlier runs
    that this run did not make again are kept, and so are its skips of other items)."""
    run = Path(run)
    out = check_out(run, out)
    media = Path(media) if media is not None else s2io.STUDY / MEDIA
    if os.path.realpath(media) == os.path.realpath(run) or Path(os.path.realpath(run)) in Path(os.path.realpath(media)).parents:
        raise ViewError(f"{media}: the media directory may not lie inside the run directory")
    _clips()
    entries, skipped = [], []
    t0 = time.perf_counter()
    git = s2io.git_state(own_outputs=[str(out)])
    made = dict(at=s2io.now_utc(), commit=git["commit"], dirty=git["dirty"], module_sha256=s2io.sha256_file(Path(__file__)))
    for it in items:
        log(" ".join(str(x) for x in it) + (" (planned)" if planned else ""))
        if planned:
            e, s = planned_clips(it[0], it[1], media, views, body, log)
        else:
            e, s = simulated_clips(run, it[0], it[1], it[2], media, block, views, body, log)
        for x in s:
            log(f"  no clip: {x['reason']}")
        entries += [dict(x, made=made) for x in e]
        skipped += [dict(x, kind="planned" if planned else "simulated", at=made["at"]) for x in s]
    path = out / "clips.json"
    old = s2io.read_json(path) if path.exists() else {}
    mine = {e["file"] for e in entries}

    def same(a, b):
        return all(a.get(k) == b.get(k) for k in ("kind", "set", "condition", "row", "block"))

    done = [dict(kind="planned" if planned else "simulated", set=it[0], condition=it[1],
                 row=None if planned else it[2], block=None if planned else block) for it in items]
    doc = dict(schema=SCHEMA_CLIPS, what="video clips of the bar's episodes and of the frozen references: views, no "
               "verdict, nothing adopted", labels=dict(simulated=SIM_LABEL, planned=PLAN_LABEL), rules=CLIP_RULES,
               run=run_label(run), media_dir="outside git (the --media directory of the run; files relative to it)",
               fps=FPS, max_bytes=MAX_BYTES,
               clips=sorted([e for e in old.get("clips", []) if e["file"] not in mine] + entries, key=lambda e: e["file"]),
               skipped=[x for x in old.get("skipped", []) if not any(same(x, d) for d in done)] + skipped,
               provenance=dict(note="of the last run of the command; each clip holds its own under 'made'", git=git,
                               at=made["at"], module="designlab/stage2/barviews.py",
                               module_sha256=made["module_sha256"], wall_s=time.perf_counter() - t0))
    s2io.write_json(path, s2io.plain(doc))
    return doc


# ------------------------------------------------------------------------------------------------
# Command line
# ------------------------------------------------------------------------------------------------
def parse_item(text: str, planned: bool = False) -> tuple:
    """``SET:CONDITION:ROW`` (a simulated clip's record) or, planned, ``SET:CONDITION``."""
    parts = text.split(":")
    if len(parts) != (2 if planned else 3) or not all(parts):
        raise ViewError(f"{text!r}: expected " + ("SET:CONDITION" if planned else "SET:CONDITION:ROW")
                        + " (for example F25F9cl:walk/v0.4" + ("" if planned else ":R2/42@10ms") + ")")
    return tuple(parts)


def record_path(run, set_name: str, body: str, block: str, row: str, ckey: str) -> Path:
    behavior, _, name = ckey.partition("/")
    return Path(run) / set_name / body / block / "eval" / row / behavior / f"{name}.json"


def posecheck(items, run=RUN, block: str = bar.HEADLINE_BLOCK, body: str = BODY, log=print) -> list[dict]:
    """How far the poses ``pose_of`` rebuilds lie from the traces' own columns, per kept trace of the named
    records: the largest distance of the feet and the CoM (mm), with and without the one-step shift of the joint
    angles.  Reads only."""
    out = []
    model = body_model(body)
    for set_name, ckey, row in items:
        rec, why = read_record(record_path(run, set_name, body, block, row, ckey))
        if rec is None:
            log(f"{set_name} {ckey} {row}: the record {why}")
            continue
        mine = []
        for e in rec.get("episodes") or []:
            trace, why = read_trace(run, rec, e)
            if trace is None:
                continue
            a, b = pose_of(trace, model, 1)["residual_m"], pose_of(trace, model, 0)["residual_m"]
            mine.append(dict(set=set_name, condition=ckey, row=row, seed=e["seed"], rows=int(len(trace["t"])),
                             with_shift_mm={k: 1e3 * v for k, v in a.items()},
                             without_shift_mm={k: 1e3 * v for k, v in b.items()}))
        out += mine
        if mine:
            w, wo = [x["with_shift_mm"] for x in mine], [x["without_shift_mm"] for x in mine]
            log(f"{set_name} {ckey} {row}: {len(mine)} traces; feet within {max(x['feet_xy'] for x in w):.2f} mm in "
                f"plan and {max(x['feet_height'] for x in w):.2f} mm in height, CoM within "
                f"{max(x['com_xy'] for x in w):.2f} mm (joint angles moved back one step); without the shift "
                f"{max(x['feet_xy'] for x in wo):.2f} mm, {max(x['feet_height'] for x in wo):.2f} mm, "
                f"{max(x['com_xy'] for x in wo):.2f} mm")
        else:
            log(f"{set_name} {ckey} {row}: no kept trace to read")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m designlab.stage2.barviews", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--run", default=str(RUN), help="the run directory the records are read from (read only)")
        p.add_argument("--body", default=BODY)

    t = sub.add_parser("turnpath", help="the room a turn needs: planned and simulated, never merged")
    common(t)
    t.add_argument("--set", required=True, dest="set_name")
    t.add_argument("--out", default=None, help="the views directory (default: this checkout's run directory's views/)")
    t.add_argument("--figure-row", default=None,
                   help="the row whose record's paths the figures draw (default: the first of "
                        + ", ".join(FIGURE_ROWS) + " with an episode that stayed up)")
    c = sub.add_parser("posecheck", help="pose_of's rebuilt feet and CoM against the traces' own columns")
    common(c)
    c.add_argument("items", nargs="+", help="SET:CONDITION:ROW")
    c.add_argument("--block", default=bar.HEADLINE_BLOCK)
    v = sub.add_parser("clips", help="video of simulated episodes from their traces, or of the frozen references")
    common(v)
    v.add_argument("items", nargs="*", help="SET:CONDITION:ROW (with --planned: SET:CONDITION)")
    v.add_argument("--planned", action="store_true", help="the frozen reference alone, labelled as planned motion")
    v.add_argument("--all-gated", action="store_true", help="every gated condition of each set under --row")
    v.add_argument("--row", default=None)
    v.add_argument("--set", action="append", dest="sets", choices=SETS, help="with --all-gated: these sets only")
    v.add_argument("--block", default=bar.HEADLINE_BLOCK, choices=tuple(bar.BLOCKS))
    v.add_argument("--views", default=None, help="comma-separated: visitor, side, top (default: by behaviour)")
    v.add_argument("--list", action="store_true", dest="only_list",
                   help="print which episodes would be rendered and why; renders and writes nothing")
    v.add_argument("--media", default=None, help=f"where the media go (default {MEDIA}/, git-ignored)")
    v.add_argument("--out", default=None, help="the views directory (default: this checkout's run directory's views/)")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "turnpath":
            doc = turnpath(a.set_name, a.run, a.out, a.body, a.figure_row)
            n = sum(len(x["simulated"]) for x in doc["turns"])
            print(f"{a.set_name}: {len(doc['turns'])} planned turns, {n} simulated records"
                  + ("" if n else " (no record yet)") + f", {len(doc['records_not_read'])} not read")
        elif a.cmd == "posecheck":
            posecheck([parse_item(x) for x in a.items], a.run, a.block, a.body)
        elif a.cmd == "clips":
            items = [parse_item(x, a.planned) for x in a.items]
            if a.all_gated:
                if a.planned or not a.row:
                    raise ViewError("--all-gated needs --row ROW and is for simulated clips")
                items += gated_items(a.row, tuple(a.sets or SETS), a.body)
            if not items:
                raise ViewError("no item: name SET:CONDITION:ROW (or --all-gated --row ROW)")
            views = None if a.views is None else tuple(x for x in a.views.split(",") if x)
            if a.only_list:
                if a.planned:
                    raise ViewError("--list is for simulated clips (a planned clip has no episode to choose)")
                list_clips(items, a.run, a.block, views, a.body)
            else:
                doc = make_clips(items, a.planned, a.run, a.media, a.out, a.block, views, a.body)
                print(f"{len(doc['clips'])} clips and {len(doc['skipped'])} skips listed in the manifest")
    except (ViewError, bar.tune.InputError) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
