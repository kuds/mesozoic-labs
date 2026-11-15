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
the step, one physics step (2 ms) later.  ``pose_of`` moves the joint angles back by that step (linear between the
50 Hz rows; ``joint_lag_steps=1``, the default) and returns how far the rebuilt feet and CoM lie from the trace's
own ``foot*_x/y`` and ``com_x/y`` columns (``residual_m``).  A kinematic play's recorder (``bar.reference_recorder``)
writes both at one instant: ``joint_lag_steps=0``, and the round trip is exact to the arrays' precision.

``barviews turnpath``: the room a turn needs
--------------------------------------------
Per set and turn reference, ``views/turnpath/<set>/turnpath.json``, ``TURNPATH.md`` and a top-view figure per
condition, with a **planned** block (the frozen reference played at the bar's commanded clock) and a **simulated**
block (every kept trace of every record of that turn that stayed up), never merged.  ``TURN_RULES`` states each
reading; it is copied into the file.

``barviews clips``: video
-------------------------
Simulated episodes rendered from their traces (``SIM_LABEL`` on every frame), or the frozen reference alone
(``--planned``; ``PLAN_LABEL``).  Which episode: ``SELECTION_RULE``.  Media go outside git (``--media``, default
``logs/barviews/``); ``views/clips.json`` lists them.

Run from hardware/trex_form_study, one process, through ``bash tools/lowprio.sh``::

    python -m designlab.stage2.barviews turnpath --set V0 [--run DIR] [--out DIR] [--figure-row R2/42@10ms]
    python -m designlab.stage2.barviews clips F25F9cl:walk/v0.4:R2/42@10ms ... [--run DIR] [--media DIR] [--out DIR]
    python -m designlab.stage2.barviews clips --planned V0:turn/q90_left F25F9cl:turn/q90_left
    python -m designlab.stage2.barviews clips --all-gated --row R2/42@10ms [--set F25F9cl]
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
#: The figure's simulated paths come from this row's record on the hard floor, headline block.
FIGURE_ROW = "R2/42@10ms"
#: Clips: size, rate, size cap, lead and tail around the move, the command caption's length, the walk's length.
FPS = 30
MAX_BYTES = 3_000_000
LEAD_S = 1.0
TAIL_S = 1.0
CAPTION_S = 1.5
WALK_CLIP_S = 10.0
#: A top view's eye is never lower than this (m): the picture then holds about 1.5 m along the walking line.
TOP_MIN_HEIGHT_M = 4.0

TURN_RULES = dict(
    words="planned: read from the frozen reference played kinematically at the bar's commanded clock (no physics). "
          "simulated: read from an episode's trace. No number mixes the two.",
    clock=f"the bar plays a reference at {bar.RATE:g} times its own clock (bar.RULES['clocks']); every time here is "
          "in simulated seconds, so a planned speed or rate in simulated time is the reference's own times "
          f"{bar.RATE:g}; lengths and radii do not change with the clock",
    frame="the corner frame of a turn: origin at the root (the trunk's origin) at the command instant; 'along' is "
          "the commanded heading (the home heading, the bar's psi_cmd), the incoming walking line; 'sideways' is "
          "square to it, positive toward the side of the turn",
    command="the command instant t_c: planned, the reference clock's time of the move's first frame over the "
            "commanded clock rate; simulated, the episode record's command_s (never recomputed)",
    move_end="the end of the move: the simulated time at which the reference clock reaches the end of the turn's "
             "own file (the trace's t_ref column); a trace that ends before it is read to its end and says so",
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
    corner_box="the corner box: over the rows from the command to the end of the move, how far the root gets "
               "along its incoming line (advance) and sideways, at the largest and at the move's end; the same "
               "box for the whole robot (every surface point)",
    surface="the robot's surface points: per body of the model, the corners of the convex hull of its pads, covers "
            "and hardware (geom groups 0, 1 and 2: mesh vertices, and the corners of a primitive's bounding box), "
            "carried on each row's pose; the snout tip is the head's point farthest forward at home, the tail tip "
            "the tail's farthest back",
    swept_radius="the swept radius: the largest distance in plan from the turn's centre to any surface point over "
                 "the steady part of the turn (where the robot is on its circle; the straight entry and exit are "
                 "in the corner box and the corridor, not here); the point that sets it is named",
    turning_circle="the turning circle: twice the swept radius, the diameter of the circle a full turn at that "
                   "rate would sweep about the same centre",
    corridor="the corridor: the narrowest L-shaped corridor, two legs of one width meeting square, that holds "
             "every surface point from the command to the end of the move.  Its outer wall along the incoming leg "
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
    """The joint angles at the rows' own times, from angles recorded ``lag_steps`` physics steps after them: linear
    between a row and the one before it (the first row: the same slope taken from the next one)."""
    Q = np.asarray(Q, float)
    if not lag_steps or len(Q) < 2:
        return Q.copy()
    t = np.asarray(t, float)
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


def turn_readings(t, xy, heading, t_c: float, t_end: float | None, psi_cmd: float, sign: float, angle: float,
                  pts=None, labels=None) -> dict:
    """The room a turn takes, from one path (``TURN_RULES``): ``t`` [n] simulated seconds, ``xy`` [n, 2] the root,
    ``heading`` [n] the trunk's unwrapped heading (rad), ``t_c`` the command, ``t_end`` the end of the move (None:
    not reached), ``psi_cmd`` the commanded heading, ``sign`` +1 left / -1 right, ``angle`` the commanded angle
    (rad); ``pts`` [n, N, 2] the surface points on each row (or None) with their ``labels`` [N].  Lengths in
    metres, angles in degrees; what cannot be read is None with its reason in ``notes``."""
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
    out["corner_box_root"] = dict(
        advance_max_m=float(along[win].max()), advance_at_end_m=float(np.interp(t_b, t, along)),
        sideways_max_m=float(side[win].max()), sideways_at_end_m=float(np.interp(t_b, t, side)),
        back_max_m=float(max(-along[win].min(), 0.0)), outside_max_m=float(max(-side[win].min(), 0.0)))
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
    out["swept"] = out["corner_box_robot"] = out["corridor"] = None
    if pts is not None:
        pts = np.asarray(pts, float)
        pa, ps = (pts - o) @ ex, (pts - o) @ ey
        out["corner_box_robot"] = dict(along_min_m=float(pa[win].min()), along_max_m=float(pa[win].max()),
                                       sideways_min_m=float(ps[win].min()), sideways_max_m=float(ps[win].max()))
        if fit is not None:
            dist = np.linalg.norm(pts[steady] - np.asarray(out["centre_world"]), axis=2)
            r_i, k_i = np.unravel_index(int(np.argmax(dist)), dist.shape)
            parts = sorted({lab.split(":")[0] for lab in labels}) if labels is not None else []
            per = {}
            for part in parts:
                sel = np.array([lab.split(":")[0] == part for lab in labels])
                per[part] = float(dist[:, sel].max())
            out["swept"] = dict(swept_radius_m=float(dist[r_i, k_i]), turning_circle_m=2.0 * float(dist[r_i, k_i]),
                                set_by=None if labels is None else labels[int(k_i)], at_s=float(t[steady[r_i]]),
                                inner_radius_m=float(dist.min()), farthest_by_part_m=per)
        if reaches_end:
            cw = corridor_width(pa[win], ps[win])
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
    "advance_max": (("corner_box_root", "advance_max_m"), "m"),
    "advance_at_end": (("corner_box_root", "advance_at_end_m"), "m"),
    "sideways_max": (("corner_box_root", "sideways_max_m"), "m"),
    "sideways_at_end": (("corner_box_root", "sideways_at_end_m"), "m"),
    "swept_radius": (("swept", "swept_radius_m"), "m"),
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
    of the move only (the rows the readings take)."""
    t = pose["t"]
    n = len(t)
    stop = t[-1] if t_end is None else t_end
    rows = np.flatnonzero((t >= t_c - bar.EPS_S) & (t <= stop + bar.EPS_S))
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
    v, rate = float(entry["gaits"]["a"]["speed"]), float(entry["turn"]["rate_rad_s"])
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
                       in_simulated_time=dict(path_speed_m_s=v * bar.RATE, turn_rate_deg_s=math.degrees(rate) * bar.RATE)),
        judged_instant_s_after_command=float(res["at_s"]), readings=r,
        pose_check=pl["pose"]["residual_m"])


def simulated_turn(sv: SetView, run, path: Path, with_paths: bool = False) -> dict:
    """The simulated block of one record (``TURN_RULES["simulated"]``): per episode that stayed up its readings,
    then their spread; the episodes that fell, and those not read with the reason."""
    rel = Path(path).relative_to(run).as_posix()
    rec, why = read_record(path)
    if rec is None:
        return dict(record=rel, read=False, reason=f"the record {why}")
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


def _anchored_path(model: BodyModel, pose: dict, t_c: float, t_end: float | None, sign: float) -> np.ndarray:
    """The root's path in its own corner frame [k, 2] (along, sideways) from ``LEAD_S`` before the command to the
    end of the move (or the trace's end)."""
    t, xy = pose["t"], pose["root_xy"]
    ex = np.array([math.cos(model.psi_cmd), math.sin(model.psi_cmd)])
    ey = sign * np.array([-ex[1], ex[0]])
    o = np.array([np.interp(t_c, t, xy[:, 0]), np.interp(t_c, t, xy[:, 1])])
    sel = (t >= t_c - LEAD_S) & (t <= (t[-1] if t_end is None else t_end) + bar.EPS_S)
    return np.column_stack([(xy[sel] - o) @ ex, (xy[sel] - o) @ ey])


def turnpath(set_name: str, run=RUN, out=None, body: str = BODY, figure_row: str = FIGURE_ROW, log=print) -> dict:
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
        pick = next((s for s in mine if s.get("row") == figure_row and s.get("floor_grip") == GRIP_HARD
                     and s.get("block") == bar.HEADLINE_BLOCK and s.get("read")), None)
        stem = c.key.replace("/", "_")
        figs[c.key] = figure(sv, c, pl, pick, d / stem)
        turns.append(dict(condition=c.key, in_words=condition_words(c.describe()), gated=bool(c.gated), move=c.move,
                          planned=pl, simulated=[{k: v for k, v in s.items() if k != "_paths"} for s in mine],
                          simulated_note=None if mine else "no record yet",
                          figure=dict(svg=f"{stem}.svg", png=f"{stem}.png", caption=figs[c.key],
                                      simulated_from=None if pick is None else pick["record"])))
        log(f"{set_name} {c.key}: planned radius {pl['readings']['fit']['radius_fitted_m']:.3f} m, "
            f"{len(mine)} simulated record(s)")
    unread = [s for s in sims if not s.get("read")]
    doc = dict(schema=SCHEMA_TURNPATH, set=set_name, set_in_words=SET_WORDS.get(set_name, set_name), body=body,
               what="views of committed records and a frozen reference: no verdict, nothing adopted",
               run=run_label(run), rules=TURN_RULES, body_length=sv.model.length_info,
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
         f"Records are read from `{doc['run']}` and cited by their path under it. Set freeze "
         f"{doc['set_freeze']}, manifest `{str(doc['set_manifest_sha256'])[:12]}`.", ""]
    for tn in doc["turns"]:
        pl = tn["planned"]
        r, cmd = pl["readings"], pl["commanded"]
        fit, box, rob, sw, co = r["fit"], r["corner_box_root"], r["corner_box_robot"], r["swept"], r["corridor"]
        A += [f"## {tn['condition']}: {tn['in_words']}", "",
              f"![top view]({tn['figure']['png']})", "", tn["figure"]["caption"], "",
              "### Planned", "",
              f"From `{pl['file']}` (sha256 `{pl['file_sha256'][:12]}`), played at {pl['clock_rate']:g} times its own "
              "clock.", "",
              "| reading (planned) | value |", "|---|---|",
              f"| path speed commanded | {cmd['path_speed_m_s']:.2f} m/s on the reference's clock "
              f"({cmd['in_simulated_time']['path_speed_m_s']:.2f} m/s in simulated time) |",
              f"| turn rate commanded | {cmd['turn_rate_deg_s']:.1f} deg/s on the reference's clock "
              f"({cmd['in_simulated_time']['turn_rate_deg_s']:.1f} deg/s in simulated time) |",
              f"| path radius commanded (speed over rate) | {_len(cmd['radius_m'], L)} |",
              f"| radius fitted to the root's path, steady part | {_len(fit and fit['radius_fitted_m'], L)}, RMS "
              f"{_num(fit and 1e3 * fit['fit_rms_m'], 0)} mm off the circle (the root sways) |",
              f"| corner box, root: advance along the incoming line | largest {_len(box['advance_max_m'], L)}, at the "
              f"move's end {_len(box['advance_at_end_m'], L)} |",
              f"| corner box, root: sideways | largest {_len(box['sideways_max_m'], L)}, at the move's end "
              f"{_len(box['sideways_at_end_m'], L)} |",
              f"| corner box, whole robot: along | {_len(rob['along_min_m'], L)} to {_len(rob['along_max_m'], L)} |",
              f"| corner box, whole robot: sideways | {_len(rob['sideways_min_m'], L)} to "
              f"{_len(rob['sideways_max_m'], L)} |",
              f"| swept radius | {_len(sw and sw['swept_radius_m'], L)}, set by {sw and sw['set_by']} |",
              f"| turning circle (twice the swept radius) | {_len(sw and sw['turning_circle_m'], L)} |",
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
            head = (f"`{s['record']}`: " if s.get("row") is None else
                    f"`{s['record']}`: controller row {s['row']}, servo delay {s['delay_ms']} ms, floor grip "
                    f"{s['floor_grip']:g}, block {s['block']}")
            if not s.get("read") or s.get("status") != "done":
                A += [f"- {head}. Not read: {s.get('reason')}.", ""]
                continue
            A += [f"**{head}.**", ""]
            fell = s["fell"]
            A += [f"{s['count_text']}; {len(fell)} fell"
                  + (" (" + ", ".join(f"seed {x['seed']} at {x['fall_time_s']:.2f} s, {x['when']}" for x in fell) + ")"
                     if fell else "") + ".", ""]
            for x in s["not_read"]:
                A += [f"- seed {x['seed']} not read: {x['reason']}."]
            sm = s["summary"]
            if not sm:
                A += ["No episode of this record stayed up with a trace that can be read: nothing is read.", ""]
                continue
            n = s["count"]["read"]
            A += [f"Of the {n} read (median, then smallest to largest):", "",
                  "| reading (simulated) | median (smallest to largest) | episodes |", "|---|---|---|"]
            for name, title in (("radius_fitted", "radius fitted to the root's path, steady part"),
                                ("advance_max", "corner box, root: largest advance"),
                                ("advance_at_end", "corner box, root: advance at the move's end"),
                                ("sideways_max", "corner box, root: largest sideways"),
                                ("sideways_at_end", "corner box, root: sideways at the move's end"),
                                ("swept_radius", "swept radius"), ("turning_circle", "turning circle"),
                                ("corridor", "corridor for the 90° corner")):
                A += [f"| {title} | {_range(sm[name], L)} | {sm[name]['n'] if sm[name] else 0} |"]
            for name, title, unit in (("turn_rate", "turn rate over the steady part", " deg/s"),
                                      ("path_speed", "path speed over the steady part", " m/s"),
                                      ("turned_at_judged_instant", "turned at the bar's judged instant", "°"),
                                      ("turned_at_move_end", "turned at the end of the move", "°"),
                                      ("turned_at_trace_end", "turned at the end of the trace", "°")):
                A += [f"| {title} | {_range(sm[name], None, 1 if unit != ' m/s' else 2, unit)} | "
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


def figure(sv: SetView, cond, planned: dict, sim: dict | None, stem: Path) -> str:
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

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "svg.fonttype": "none"})
    words = condition_words(cond.describe())
    drawn = [] if sim is None else sim.get("_paths", [])
    if drawn:
        note = (f"Simulated paths: {sim['row']}, floor grip {sim['floor_grip']:g}, {sim['block']}; "
                f"{sim['count_text']}; {len(drawn)} drawn.")
        simtext = (f"Orange, thin: SIMULATED root paths of the {len(drawn)} episodes that stayed up in "
                   f"`{sim['record']}` ({sim['count_text']}).")
    elif sim is not None:
        note = f"Simulated: no episode to draw from {sim['row']} ({sim.get('count_text') or sim.get('reason')})."
        simtext = f"SIMULATED: no episode of `{sim['record']}` stayed up with a trace that can be read; none is drawn."
    else:
        note = "Simulated: no record yet, so no simulated path is drawn."
        simtext = "SIMULATED: no record yet; no simulated path is drawn."
    foot = textwrap.wrap(f"{sv.name} ({SET_WORDS.get(sv.name, sv.name)}): {words}. Top view, up is the incoming "
                         "walking line. Every path is drawn from its own place at the command. " + note, 84)
    handles = [plt.Line2D([], [], color=C["planned"], linewidth=2.0, label="planned: root path of the frozen reference"),
               plt.Line2D([], [], color=C["ink2"], linewidth=0.8, linestyle=(0, (4, 3)), label="corner box of the planned root"),
               Polygon([[0, 0]], facecolor=C["band"], edgecolor=C["ink3"], linewidth=0.5,
                       label="planned outline at the command and at the move's end")]
    if drawn:
        handles.insert(1, plt.Line2D([], [], color=C["simulated"], linewidth=0.9,
                                     label=f"simulated: root paths of {len(drawn)} episodes that stayed up"))
    # the picture's extent: the planned robot's corner box (and the simulated paths), a margin, room for the bar
    rob, box = r["corner_box_robot"], r["corner_box_root"]
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
    below = 0.10 + 0.125 * len(foot) + 0.06 + 0.15 * len(handles) + 0.08
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
                           edgecolor=C["ink2"], linewidth=0.8, linestyle=(0, (4, 3)), zorder=3))
    ax.text(0.5 * (x0 + x1) - sign * 0.12 * (x1 - x0), 0.42 * box["advance_max_m"],
            f"corner box of the planned root:\n{box['advance_max_m']:.2f} m along,\n{box['sideways_max_m']:.2f} m sideways",
            color=C["ink2"], fontsize=6.5, ha="center", va="center", linespacing=1.25, zorder=7)
    for path in drawn:
        ax.plot(-sign * path[:, 1], path[:, 0], color=C["simulated"], linewidth=0.7, alpha=0.75, zorder=4)
    sel = (pose["t"] >= t_c - LEAD_S) & (pose["t"] <= t_end + bar.EPS_S)
    pxy = plot_xy(pose["root_xy"][sel])
    ax.plot(pxy[:, 0], pxy[:, 1], color=C["planned"], linewidth=2.0, zorder=5, solid_capstyle="round")
    ax.plot([0], [0], marker="o", markersize=5, color=C["planned"], markeredgecolor=C["surface"], markeredgewidth=1.0,
            zorder=6)
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
            "to the end of the move; the grey shapes are the PLANNED outline of the robot at the command and at the "
            "end of the move; the dashed box is the corner box of the planned root. " + simtext
            + " Every path is drawn from its own position at the command. The scale bar is 0.5 m.")


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
    t.add_argument("--figure-row", default=FIGURE_ROW)
    c = sub.add_parser("posecheck", help="pose_of's rebuilt feet and CoM against the traces' own columns")
    common(c)
    c.add_argument("items", nargs="+", help="SET:CONDITION:ROW")
    c.add_argument("--block", default=bar.HEADLINE_BLOCK)
    a = ap.parse_args(argv)
    try:
        if a.cmd == "turnpath":
            doc = turnpath(a.set_name, a.run, a.out, a.body, a.figure_row)
            n = sum(len(x["simulated"]) for x in doc["turns"])
            print(f"{a.set_name}: {len(doc['turns'])} planned turns, {n} simulated records"
                  + ("" if n else " (no record yet)") + f", {len(doc['records_not_read'])} not read")
        elif a.cmd == "posecheck":
            posecheck([parse_item(x) for x in a.items], a.run, a.block, a.body)
    except (ViewError, bar.tune.InputError) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
