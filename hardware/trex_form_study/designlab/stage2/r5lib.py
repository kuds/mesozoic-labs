"""R5's reference library: scripts over a frozen set (round C design D3, D5; section 3.3; rules E4, T1-T3).

An episode of R5 or R6 plays a *script*: a plan of round A's track items (``("ref", name, windows)``,
``("transition", name)``, ``("hold", seconds)``) over one frozen set directory, built with ``setio.build_track``,
so its track and command events are the bar's own.  A ``Script`` adds the side tables the env, the policy rows and
the reward read; each is defined for any reference time ``t_ref >= 0`` and continues past the plan's end exactly as
``control.Track.joint`` and ``Track.info`` do (a periodic last segment loops, a hold holds, a finite last segment
holds its last frame), because a clock up to 25 % fast passes the end of a plan sized for a clock 5 % fast.

* **Command** (``COMMAND_RULE``, rule E4): piecewise constant on the reference clock, in the references' own units,
  and it changes at a segment's first frame only.  Nothing before a segment's first frame depends on that segment.
* **Commanded heading**: the integral of the commanded yaw rate over reference time from the track's start.
* **Phase** in [0, 1), 0 at a left lift-off: a periodic file's own ``phase`` where it holds one (a K-stride
  window), else the local time over the entry's stride period; a move's own ``phase`` over its frame times; 0 in a
  hold.  ``stride_s``, ``walking``, ``steady``, ``support`` and ``mid_swing`` come from the same frames (the
  track's own ``swing``, ``u``, ``anchor`` and ``stand``).  ``swing_s`` is the planned length of the reference's
  swing in progress, else of its next one (``setio.planned_swings``: joined across segments), so that inside a
  change of gait each swing has its own length.
* **Start states**: each file's full ``q`` and ``qd`` per frame; ``start_state(t)`` is the frame at or before
  ``t``.  ``mid_spans`` are where a mid-task start may land (rule T3): never in a hold.
* **Reward references** by forward kinematics of each file frame on the plant's model: the feet relative to the
  root in the root's heading frame, the head body's roll and pitch about home and, for a periodic walk, the CoM's
  progress along the walk's own stride direction (``walk_progress``).
* **Required joint-range margin**: decision 14's, as ``bar.BarContext.built`` computes it; it never exceeds what
  the reference itself keeps.

``Library`` reads a set (``setio.read_inputs`` on the files its tasks need, frozen), resolves the task table
(``TASKS``: rule T1; a task whose reference is absent or ``available: false`` is dropped and listed with its
reason, rule T2) and builds scripts for tasks and for any plan (``from_plan``: a bar condition's).  The pseudo-set
``S1`` is Stage 1's own run directory through ``setio.stage1_view`` with the small table ``S1_TASKS``: for
development and tests only.

Nothing here iterates a ``set`` (an episode may not depend on ``PYTHONHASHSEED``).  Imports ``designlab.stage1``
and round A's ``setio`` and ``s2io``; never ``bar``, ``r5core`` or ``r5env``.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass

import mujoco
import numpy as np

from .. import statics
from ..stage1 import HEAD_X_BODIES, JETSON_BODIES, control, env, metrics, plant, tune
from ..stage1.refs import Reference
from ..stage1.transitions import Transition
from . import s2io, setio

#: The pseudo-set of Stage 1's own run directories (development and tests only).
STOCK_SET = "S1"
STAGE1_RUN = s2io.STUDY / "stage1" / "runs" / "2026-11-08"
#: Where the bodies that are not Stage 1's own keep their Stage 1 arrays (``stage2/x/run_x.sh``, ``stage2/j/run_j.sh``).
S1_CROUCH = 0.03  # Stage 1's crouch of the track's LIPM frequency (``control.omega0``'s default)

#: Rule T1: an episode lasts at most this long (s, simulated); a task's last walk is planned to reach it.
MAX_SECONDS = 30.0
#: The stand task: a 12 s hold of the home stance, 10 s played (Stage 1's ``env.TASKS["stand"]``, ``STAND_SECONDS``).
STAND_HOLD_S = 12.0
STAND_SECONDS = env.STAND_SECONDS
#: The bar's lead strides before a turn and before the two-step stop (``bar.TURN_LEAD_STRIDES``,
#: ``bar.STOP_LEAD_STRIDES[2]``: a test compares them), in windows of the walk's file.
TURN_LEAD = {"left": 2.0, "right": 2.5}
STOP_LEAD = 2.5
#: The stop task's final hold (reference s).  Rule T1 says "a hold" and gives no length: this covers the bar's
#: stop item, whose hold after the rest is 2.0 s plus its end margin at the slowest clock.
STOP_HOLD_S = 4.0
#: The start-stop task's hold between its stop and its second start (rule T1).
START_STOP_HOLD_S = 2.0
START_STOP_WINDOWS = 2
#: The turn in place of design D5 (names proposed there; a set that names them otherwise has no such task).
INPLACE_RATE = 0.5
INPLACE_STRIDES = 3
#: Rule T3: a mid-task start lands between this long after the first non-hold frame and this long (simulated)
#: before the plan's end.
MID_AFTER_S = 0.2
MID_BEFORE_END_S = 2.0
#: A periodic item played "to the end" of its task (resolved to whole windows by ``Library``).
END = "end"
BAR_WALK = setio.walk_name(setio.BAR_SPEED)

COMMAND_RULE = (
    "the command (v, w) is piecewise constant on the reference clock, in the references' own units (the "
    "observation multiplies by the commanded clock rate): a hold (0, 0); a periodic walk or turn in place its "
    "entry's speed_m_s and yaw_rate_rad_s; a start, a change, a start into the turn in place and a stop the "
    "gait they lead to (gaits.b; a stand is (0, 0)) from the segment's first frame; a finite turn of angle A at "
    "rate w0 from walk a keeps a's speed and commands sign x |w0| for A / |w0| seconds of reference time from "
    "its first frame, then 0.  The commanded heading is the integral of w over reference time from the track's "
    "start.  Nothing before a segment's first frame depends on that segment; past the plan's end the last "
    "segment's command continues")


class Absent(Exception):
    """A task's reference is not in the set (rule T2): the message is the reason."""


@dataclass(frozen=True)
class TaskDef:
    """One row of the task table (rule T1): its name, the plan it builds (round A's track items; ``END`` in a
    periodic item's count stands for "to the end"), its relative weight, the monitor conditions whose failure
    doubles its weight (rule M4), and the simulated seconds played from a stance start (None: the plan's own
    length, at most ``MAX_SECONDS``)."""

    name: str
    plan: tuple
    weight: float
    monitor: tuple = ()
    seconds: float | None = None


def _lead(walk: str) -> list:
    """The items before the steady walk ``walk``: ``start``; for a rung, also one window of ``nominal`` and the
    direct change (rule T1's L for 0.40 m/s)."""
    if walk == setio.NOMINAL:
        return [("transition", setio.START)]
    return [("transition", setio.START), ("ref", setio.NOMINAL, 1),
            ("transition", setio.change_name(setio.NOMINAL, walk))]


def _walk_plan(walk: str) -> tuple:
    return tuple(_lead(walk) + [("ref", walk, END)])


def _turn_plan(rate: str, direction: str) -> tuple:
    return tuple(_lead(BAR_WALK) + [("ref", BAR_WALK, TURN_LEAD[direction]),
                                    ("transition", setio.turn_name(rate, direction)), ("ref", BAR_WALK, END)])


def _stop_items() -> list:
    return _lead(BAR_WALK) + [("ref", BAR_WALK, STOP_LEAD), ("transition", setio.stop_name(2, BAR_WALK))]


def inplace_names(direction: str) -> dict:
    """Design D5's names of the turn in place and of its start and stop, for one direction."""
    tag = f"inplace_{INPLACE_RATE:g}_{direction}"
    return dict(ref=tag, start=f"start_{tag}", stop=f"stop_{tag}")


def _inplace_plan(direction: str) -> tuple:
    n = inplace_names(direction)
    return (("transition", n["start"]), ("ref", n["ref"], INPLACE_STRIDES), ("transition", n["stop"]))


def _tasks() -> tuple:
    out = [TaskDef("stand", (("hold", STAND_HOLD_S), ("transition", setio.START)), 1.0, seconds=STAND_SECONDS),
           TaskDef("walk_nominal", _walk_plan(setio.NOMINAL), 2.0, monitor=("walk/nominal",)),
           TaskDef(f"walk_{BAR_WALK}", _walk_plan(BAR_WALK), 3.0,
                   monitor=(f"walk/{BAR_WALK}", "nudge/forward", "nudge/left"))]
    for v in setio.SPEEDS:
        if v != setio.BAR_SPEED:
            out.append(TaskDef(f"walk_{setio.walk_name(v)}", _walk_plan(setio.walk_name(v)), 0.5))
    for d in setio.DIRECTIONS:
        out.append(TaskDef(f"turn_q90_{d}", _turn_plan(setio.QUICK, d), 1.5, monitor=("turn/q90_left",)))
    for rate in setio.TURN_RATES:
        if rate != setio.QUICK:
            for d in setio.DIRECTIONS:
                out.append(TaskDef(f"turn_{rate}_{d}", _turn_plan(rate, d), 0.25))
    out.append(TaskDef("stop", tuple(_stop_items() + [("hold", STOP_HOLD_S)]), 2.0, monitor=("stop/two_step",)))
    out.append(TaskDef("start_stop", tuple(_stop_items() + [("hold", START_STOP_HOLD_S), ("transition", setio.START),
                                                            ("ref", setio.NOMINAL, START_STOP_WINDOWS)]), 1.0,
                       monitor=("stop/two_step",)))
    for d in setio.DIRECTIONS:
        out.append(TaskDef(f"inplace_{d}", _inplace_plan(d), 0.5))
    return tuple(out)


#: Rule T1's table, in its order.
TASKS = _tasks()
#: The S1 view's table: Stage 1's own plans of ``env.TASKS`` for the stand, the walk and the brisk walk, and a stop
#: that ends in a hold (``env.TASKS`` has no task of that name: Stage 1's start, two strides and its stop, then
#: ``STOP_HOLD_S``).
S1_TASKS = (
    TaskDef("stand", tuple(tuple(x) for x in env.TASKS["stand"][0]), 1.0, seconds=STAND_SECONDS),
    TaskDef("walk", tuple(tuple(x) for x in env.TASKS["walk"][0]), 2.0, monitor=("walk/nominal",)),
    TaskDef("brisk", tuple(tuple(x) for x in env.TASKS["brisk"][0]), 2.0, monitor=("walk/brisk",)),
    TaskDef("stop", (("transition", "start"), ("ref", "nominal", 2), ("transition", "stop"), ("hold", STOP_HOLD_S)),
            2.0),
)


def s1_root(body: str):
    """The directory that holds ``body``'s Stage 1 run directory: Stage 1's run for its own bodies, ``stage2/x``
    for the expressive heads and ``stage2/j`` for the Jetson body (their ``prepare`` runs)."""
    if body in HEAD_X_BODIES:
        return s2io.STUDY / "stage2" / "x"
    if body in JETSON_BODIES:
        return s2io.STUDY / "stage2" / "j"
    return STAGE1_RUN


def plan_files(plan) -> list[str]:
    """The files a plan loads, with ``refs/nominal.npz`` (the home pose of ``control.omega0``), sorted."""
    out = [f"refs/{setio.NOMINAL}.npz"]
    for item in plan:
        f = f"refs/{item[1]}.npz" if item[0] == "ref" else f"transitions/{item[1]}.npz" if item[0] == "transition" \
            else None
        if f is not None and f not in out:
            out.append(f)
    return sorted(out)


# ------------------------------------------------------------------------------------------------
# Kinematic readings shared with the reward (r5core reads the true state with the same functions)
# ------------------------------------------------------------------------------------------------
def head_body(p: plant.Plant) -> int:
    return statics.bid(p.m, "head")


def head_home(p: plant.Plant) -> np.ndarray:
    """The head body's world orientation at the model's home pose (3 x 3): what "about home" means for the head."""
    kd = mujoco.MjData(p.m)
    kd.qpos[:] = p.q_home
    mujoco.mj_kinematics(p.m, kd)
    return kd.xmat[head_body(p)].reshape(3, 3).copy()


def foot_rel_of(xpos, xmat, trunk: int, feet) -> np.ndarray:
    """[2, 3]: each foot body's position less the root's, rotated into the root's heading."""
    yaw = control._yaw(xmat[trunk])
    c, s = math.cos(yaw), math.sin(yaw)
    r = xpos[trunk]
    out = np.empty((2, 3))
    for j, f in enumerate(feet):
        dx, dy, dz = (xpos[f] - r).tolist()
        out[j, 0], out[j, 1], out[j, 2] = c * dx + s * dy, -s * dx + c * dy, dz
    return out


def head_rp_of(xmat, head: int, R_head_home: np.ndarray) -> np.ndarray:
    """[2]: the head body's roll and pitch in the world about its home orientation (``control._roll_pitch`` reads
    the third row only, so a yaw of the whole robot changes neither)."""
    return np.array(control._roll_pitch(np.asarray(xmat[head]).reshape(3, 3) @ R_head_home.T))


def fk_tables(p: plant.Plant, q: np.ndarray, R_head_home: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``foot_rel`` [n, 2, 3] and ``head_rp`` [n, 2] of the poses ``q`` [n, nq] by forward kinematics on ``p.m``
    (a scratch MjData: the plant's own data is not touched)."""
    m = p.m
    kd = mujoco.MjData(m)
    head = head_body(p)
    q = np.atleast_2d(q)
    foot, rp = np.empty((len(q), 2, 3)), np.empty((len(q), 2))
    for k in range(len(q)):
        kd.qpos[:] = q[k]
        mujoco.mj_kinematics(m, kd)
        foot[k] = foot_rel_of(kd.xpos, kd.xmat, p.trunk, p.feet)
        rp[k] = head_rp_of(kd.xmat, head, R_head_home)
    return foot, rp


# ------------------------------------------------------------------------------------------------
# A script
# ------------------------------------------------------------------------------------------------
PERIODIC, FINITE, HOLD = 0, 1, 2
_KIND = {"periodic": PERIODIC, "finite": FINITE, "hold": HOLD}


class _Seg:
    """One track segment's side tables (``Script``).  Tables that are interpolated (``phase``, ``foot_rel``,
    ``head_rp``) carry one row more than the segment has frames, as ``control.Segment.S`` does for a finite
    segment (the next segment's first frame) and as the periodic wrap needs."""

    __slots__ = ("kind", "name", "t0", "duration", "step", "n", "t", "nf", "v", "w", "pulse_s", "turn", "heading0",
                 "stride_s", "period", "swing_s", "phase", "swing", "u", "anchor", "stand", "foot_rel", "head_rp",
                 "q", "qd", "walk", "along", "swing_len")


def _unwrapped(ph: np.ndarray) -> np.ndarray:
    """A phase series made monotone: 1 is added at every frame where it falls (a wrap)."""
    ph = np.asarray(ph, float)
    return ph + np.concatenate([[0.0], np.cumsum(np.diff(ph) < -0.5)])


def _swing_lengths(swing: np.ndarray, step: float) -> np.ndarray | None:
    """Per frame of a periodic file: the length (s) of the swing the frame belongs to, or, in double support, of
    the next swing (wrapping); None for a file without a swing."""
    n = len(swing)
    out = np.full(n, np.nan)
    for j in (0, 1):
        for run in control._swing_runs(swing == j, True):
            out[run] = len(run) * step
    if not np.any(np.isfinite(out)):
        return None
    for k in range(2 * n - 1, -1, -1):  # backward twice round: each double-support frame takes the next swing's
        if not np.isfinite(out[k % n]):
            out[k % n] = out[(k + 1) % n]
    return out


class Script:
    """One plan on one plant: the track and its side tables (module docstring).  ``seconds`` is the simulated
    length played from a stance start when it is shorter than the plan (the stand task); it bounds ``mid_spans``."""

    def __init__(self, lib: "Library", plan, seconds: float | None = None, track: control.Track | None = None):
        self.plan = [tuple(x) for x in plan]
        self.rate = float(lib.rate)
        self.seconds = None if seconds is None else float(seconds)
        p = lib.p
        if track is None:
            track = setio.build_track(p, lib.dir, self.plan, lib.omega, lib.tr_doc)
        elif len(track.segments) != len(self.plan):
            raise ValueError(f"the track has {len(track.segments)} segments, the plan {len(self.plan)} items")
        self.track = track
        self.t_end = float(track.t_end)
        self.events = list(track.events)
        self._segs = lib._tables(self.plan, track)
        self._n = len(self._segs)
        self._i = 0
        self._lt, self._lv = None, None
        self.has_turn = any(s.turn for s in self._segs)
        #: The track's planned swings on the reference clock (start, end), in order.
        sw = sorted((t0, t1) for _, t0, t1 in setio.planned_swings(track))
        self._sw0, self._sw1 = [a for a, _ in sw], [b for _, b in sw]
        nj = len(p.joints)
        m = p.m
        jid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jn) for jn in p.joints]
        lo, hi = m.jnt_range[jid, 0].copy(), m.jnt_range[jid, 1].copy()
        q = np.vstack([seg.S[:, :nj] for seg in track.segments])
        #: The track's own smallest margin per joint (an end a joint rests on by design is no limit:
        #: ``plant.REST_ON_STOP_RULE``), and decision 14's required margin.
        self.ref_margin = metrics.joint_margins(q, lo, hi, plant.rest_ends(p.joints))
        if self.has_turn:
            self.required_margin = np.maximum(
                np.minimum(tune.WALK_RANGE_RAD, self.ref_margin - tune.TURN_RANGE_SLACK_RAD), tune.AT_LIMIT_RAD)
        else:
            self.required_margin = np.minimum(tune.WALK_RANGE_RAD, self.ref_margin)
        self._ref_yaw0 = float(track.info(0.0)[1]["yaw"])

    # ---------------------------------------------------------------- lookup
    def _loc(self, t: float):
        """(segment tables, local time, frame at or before, share toward the next row) at reference time ``t``;
        the segment by ``control.Track.segment``'s rule, the frame by ``Track.joint``'s."""
        if t == self._lt:
            return self._lv
        if t < 0.0:
            t = 0.0
        segs, i = self._segs, self._i
        if t < segs[i].t0:
            i = 0
        while i + 1 < self._n and t >= segs[i + 1].t0:
            i += 1
        self._i = i
        s = segs[i]
        tl = t - s.t0
        if s.kind == PERIODIC:
            x = tl / s.step
            f = math.floor(x)
            k0, a = int(f) % s.n, x - f
        elif s.kind == HOLD:
            k0, a = 0, 0.0
        else:
            tt = s.t
            k0 = min(max(int(tt.searchsorted(tl, side="right")) - 1, 0), s.nf - 1)
            a = min(max((tl - tt[k0]) / (tt[k0 + 1] - tt[k0]), 0.0), 1.0)
        self._lt, self._lv = t, (s, tl, k0, a)
        return self._lv

    def segment_index(self, t_ref: float) -> int:
        self._loc(t_ref)
        return self._i

    def kind(self, t_ref: float) -> str:
        """"periodic", "finite" or "hold": the segment in force."""
        return ("periodic", "finite", "hold")[self._loc(t_ref)[0].kind]

    # ---------------------------------------------------------------- command, heading
    def command(self, t_ref: float) -> tuple[float, float]:
        s, tl, _, _ = self._loc(t_ref)
        if s.turn:
            return s.v, (s.w if tl < s.pulse_s else 0.0)
        return s.v, s.w

    def heading(self, t_ref: float) -> float:
        s, tl, _, _ = self._loc(t_ref)
        return s.heading0 + s.w * (min(tl, s.pulse_s) if s.turn else tl)

    def ref_yaw(self, t_ref: float) -> float:
        """The reference's own (unwrapped) yaw less its yaw at the track's start (``Track.info``'s)."""
        return float(self.track.info(max(t_ref, 0.0))[1]["yaw"]) - self._ref_yaw0

    # ---------------------------------------------------------------- gait
    def phase(self, t_ref: float) -> float:
        s, tl, k0, a = self._loc(t_ref)
        if s.kind == HOLD:
            return 0.0
        if s.phase is None:
            return (tl / s.stride_s) % 1.0
        return ((1.0 - a) * s.phase[k0] + a * s.phase[k0 + 1]) % 1.0

    def stride_s(self, t_ref: float) -> float:
        s, _, k0, _ = self._loc(t_ref)
        return s.stride_s if s.kind == PERIODIC else float(s.period[k0])

    def swing_s(self, t_ref: float) -> float:
        """The planned length (reference s) of the reference's swing in progress at ``t_ref``, else of its next
        swing, else of its last: inside a periodic segment from its file's own swings, elsewhere from the track's
        planned swings, joined across segments; a track that plans no complete swing gives its segment's
        gait's."""
        s, _, k0, _ = self._loc(t_ref)
        if s.swing_len is not None:  # a periodic segment: its file's own swings, which loop as the track does
            return float(s.swing_len[k0])
        t0s = self._sw0
        if not t0s:
            return s.swing_s
        i = bisect.bisect_right(t0s, t_ref) - 1
        if i < 0 or (t_ref >= self._sw1[i] and i + 1 < len(t0s)):
            i += 1
        return self._sw1[i] - t0s[i]

    def walking(self, t_ref: float) -> bool:
        s, _, k0, _ = self._loc(t_ref)
        return not bool(s.stand[k0])

    def walk_start(self, t_ref: float) -> float | None:
        """The first frame's reference time of the periodic walking segment ``t_ref`` lies in (None outside one)."""
        s = self._loc(t_ref)[0]
        return s.t0 if s.walk else None

    def steady(self, t_ref: float) -> bool:
        s, tl, _, _ = self._loc(t_ref)
        return bool(s.walk and tl >= s.stride_s)

    def walk_progress(self, t_a: float, t_b: float) -> float:
        """How far the reference's CoM moves along its own stride direction from reference time ``t_a`` to
        ``t_b``, both taken inside the periodic walking segment ``t_b`` lies in (``t_a`` no earlier than its first
        frame); 0.0 outside such a segment."""
        s, tl, k0, a = self._loc(t_b)
        if not s.walk:
            return 0.0
        window = s.n * s.step

        def at(x: float) -> float:
            y = x / s.step
            f = math.floor(y)
            k = int(f) % s.n
            a = y - f
            return math.floor(x / window + 1e-12) * s.along[-1] + (1.0 - a) * s.along[k] + a * s.along[k + 1]

        return at(tl) - at(max(t_a - s.t0, 0.0))

    def contact_ref(self, t_ref: float) -> tuple[bool, bool, bool, bool]:
        """(left support, right support, left in mid-swing, right in mid-swing) of the reference: the frame at or
        before ``t_ref``, the swing progress interpolated as ``Track.info`` does."""
        s, _, k0, a = self._loc(t_ref)
        sw = int(s.swing[k0])
        if sw < 0:
            return True, True, False, False
        u = float(s.u[k0])
        if a > 0.0:
            k1 = (k0 + 1) % s.n if s.kind == PERIODIC else min(k0 + 1, s.nf - 1)
            if k1 != k0 and s.anchor[k1] == s.anchor[k0] and s.swing[k1] == sw:
                u = (1.0 - a) * u + a * float(s.u[k1])
        mid = MID_SWING[0] <= u <= MID_SWING[1]
        return sw != 0, sw != 1, mid and sw == 0, mid and sw == 1

    def support(self, t_ref: float) -> np.ndarray:
        return np.array(self.contact_ref(t_ref)[:2])

    def mid_swing(self, t_ref: float) -> np.ndarray:
        return np.array(self.contact_ref(t_ref)[2:])

    def swing_progress(self, t_ref: float) -> tuple[int, float]:
        """(the reference's swinging foot, -1 none; its progress in [0, 1)) at the frame at or before ``t_ref``."""
        s, _, k0, _ = self._loc(t_ref)
        return int(s.swing[k0]), float(s.u[k0])

    # ---------------------------------------------------------------- reward references
    def foot_rel(self, t_ref: float) -> np.ndarray:
        s, _, k0, a = self._loc(t_ref)
        if s.kind == HOLD:
            return s.foot_rel[0]
        return (1.0 - a) * s.foot_rel[k0] + a * s.foot_rel[k0 + 1]

    def head_rp(self, t_ref: float) -> np.ndarray:
        s, _, k0, a = self._loc(t_ref)
        if s.kind == HOLD:
            return s.head_rp[0]
        return (1.0 - a) * s.head_rp[k0] + a * s.head_rp[k0 + 1]

    # ---------------------------------------------------------------- starts
    def start_state(self, t_ref: float) -> tuple[float, np.ndarray, np.ndarray]:
        """(the time of the file frame at or before ``t_ref``, that frame's qpos, its qvel at the file's own
        rate).  A hold has no file frame: ``ValueError``."""
        s, tl, k0, _ = self._loc(max(float(t_ref), 0.0))
        if s.kind == HOLD:
            raise ValueError(f"reference time {t_ref:g} s lies in a hold: a mid-task start never lands there")
        if s.kind == PERIODIC:
            kk = int(math.floor(tl / s.step + 1e-9))
            return s.t0 + kk * s.step, s.q[kk % s.n].copy(), s.qd[kk % s.n].copy()
        return s.t0 + float(s.t[k0]), s.q[k0].copy(), s.qd[k0].copy()

    @property
    def t_stop(self) -> float:
        """The reference time a stance start plays to at the commanded clock: the plan's end, or ``seconds``."""
        return self.t_end if self.seconds is None else min(self.t_end, self.seconds * self.rate)

    def mid_spans(self) -> list[tuple[float, float]]:
        moving = [s for s in self._segs if s.kind != HOLD]
        if not moving:
            return []
        lo, hi = moving[0].t0 + MID_AFTER_S, self.t_stop - MID_BEFORE_END_S * self.rate
        out = []
        for s in moving:
            a, b = max(s.t0, lo), min(s.t0 + s.duration, hi)
            if b > a:
                out.append((float(a), float(b)))
        return out

    def describe(self) -> dict:
        return dict(plan=[list(x) for x in self.plan], t_end_s=self.t_end, seconds=self.seconds,
                    events=s2io.plain(self.events), finite_turn=self.has_turn,
                    required_margin_rad=[float(x) for x in self.required_margin])


#: The middle half of a swing (``readings.MID_SWING``).
MID_SWING = (0.25, 0.75)


# ------------------------------------------------------------------------------------------------
# The library
# ------------------------------------------------------------------------------------------------
def _num(x, default: float = 0.0) -> float:
    """A JSON number that may be missing, null or NaN (a stand's period and duty in ``transitions.json``)."""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return default
    return x if math.isfinite(x) else default


class Library:
    """The scripts of one set on one plant (module docstring).  ``set_name`` ``"S1"`` is the Stage 1 view (no
    freeze); any other set is read from ``root`` and must be frozen unless ``require_frozen`` is False (tests).
    Build it, and every script, while the plant's model is nominal: a track's reference quantities and the LIPM
    frequency read its masses."""

    def __init__(self, p: plant.Plant, body: str, set_name: str, rate: float, root=setio.ROOT,
                 require_frozen: bool = True):
        self.p, self.body, self.set, self.rate = p, body, set_name, float(rate)
        self.require_frozen = bool(require_frozen)
        if set_name == STOCK_SET:
            self.dir = setio._anchored(s1_root(body)) / body
            view, docs = setio.stage1_view(self.dir.parent, body)
            self.set_doc, self._view = {}, view["files"]
            self.crouch_frac = S1_CROUCH
            table = S1_TASKS
        else:
            self.dir = setio.set_dir(set_name, body, root)
            docs = setio.load(self.dir)
            self.set_doc, self._view = docs["set.json"], None
            if self.set_doc.get("crouch_frac") is None:
                raise tune.InputError(f"{set_name}/{body}: set.json records no crouch_frac (the crouch of the track's "
                                      "LIPM frequency, round A D16)")
            self.crouch_frac = float(self.set_doc["crouch_frac"])
            table = TASKS
        self.refs_doc = docs["refs.json"]["references"]
        self.tr_doc = docs["transitions.json"]["transitions"]
        self._objs, self._fk, self._scripts = {}, {}, {}
        self._files, self._freeze = {}, None
        templates, self.tasks = {}, {}
        for td in table:
            try:
                for item in td.plan:
                    self._need(item)
                templates[td.name] = td
            except Absent as why:
                self.tasks[td.name] = dict(plan=None, weight=float(td.weight), available=False, reason=str(why),
                                           seconds=None, monitor=list(td.monitor))
        need = [f"refs/{setio.NOMINAL}.npz"]
        for td in templates.values():
            for f in plan_files(td.plan):
                if f not in need:
                    need.append(f)
        self._read(need)
        self.R_head_home = head_home(p)
        self.omega = float(control.omega0(p, self._load(("ref", setio.NOMINAL)).q_home, crouch_frac=self.crouch_frac))
        self._defs = {}
        for td in table:  # (the table's order: the resolved table lists the absent tasks in place)
            if td.name not in templates:
                continue
            plan = self._resolve(td.plan)
            t_end = sum(d for _, d in self.segment_times(plan))
            seconds = float(td.seconds) if td.seconds is not None else min(t_end / self.rate, MAX_SECONDS)
            self.tasks[td.name] = dict(plan=[list(x) for x in plan], weight=float(td.weight), available=True,
                                       reason=None, seconds=seconds, monitor=list(td.monitor))
            self._defs[td.name] = td
        self.tasks = {td.name: self.tasks[td.name] for td in table}

    # ---------------------------------------------------------------- what the set has
    def why_not(self, sub: str, name: str) -> str | None:
        """None if the set has a usable entry ``name`` (``sub``: "refs" or "transitions"), else the reason
        (the entry's, ``set.json``'s ``not_available`` note, or that there is no entry)."""
        e = (self.refs_doc if sub == "refs" else self.tr_doc).get(name)
        if isinstance(e, dict) and e.get("available") is not False and e.get("file"):
            return None
        noted = (self.set_doc.get("not_available") or {}).get(name) or {}
        if e is None:
            return str(noted.get("reason") or f"the set has no entry {sub}/{name}")
        return str(e.get("reason") or noted.get("reason") or "recorded as not available")

    def _need(self, item) -> None:
        if item[0] == "hold":
            return
        sub = "refs" if item[0] == "ref" else "transitions"
        why = self.why_not(sub, item[1])
        if why is not None:
            raise Absent(f"{sub}/{item[1]}: {why}")

    def _read(self, files) -> None:
        """Check ``files`` against the set (``setio.read_inputs``; the S1 view was checked whole) and note their
        hashes; every file read in one library's life must lie under one freeze."""
        new = [f for f in files if f not in self._files]
        if not new:
            return
        if self._view is not None:
            missing = [f for f in new if f not in self._view]
            if missing:
                raise tune.InputError(f"{self.dir}: {', '.join(missing)} not listed in refs.json or transitions.json")
            for f in new:
                self._files[f] = dict(self._view[f])
            return
        rec, _ = setio.read_inputs(self.dir, new, require_frozen=self.require_frozen)
        if self._files and rec["freeze"] != self._freeze:
            raise tune.InputError(f"{self.dir}: the set's freeze changed while it was being read")
        self._freeze = rec["freeze"]
        for f in new:
            self._files[f] = dict(rec["files"][f])

    def _load(self, item):
        """The file of a plan item (``Reference`` or ``Transition``; None for a hold), loaded once and only after
        its hash was checked."""
        if item[0] == "hold":
            return None
        key = (item[0], item[1])
        if key not in self._objs:
            sub = "refs" if item[0] == "ref" else "transitions"
            f = f"{sub}/{item[1]}.npz"
            self._read([f])
            self._objs[key] = (Reference if item[0] == "ref" else Transition).load(self.dir / f)
        return self._objs[key]

    def _fk_of(self, item) -> tuple[np.ndarray, np.ndarray]:
        key = (item[0], item[1])
        if key not in self._fk:
            self._fk[key] = fk_tables(self.p, self._load(item).q, self.R_head_home)
        return self._fk[key]

    def _along_of(self, item) -> np.ndarray:
        """[n + 1]: a periodic walk's CoM progress along its own stride direction per frame, from its first frame;
        the last entry is one window's advance (``stride_motion``)."""
        key = ("along", item[1])
        if key not in self._fk:
            ref, p = self._load(item), self.p
            adv = np.asarray(ref.stride_motion, float)[3:5]
            length = float(np.hypot(adv[0], adv[1]))
            e = adv / length if length > 0.0 else np.array([1.0, 0.0])
            kd = mujoco.MjData(p.m)
            com = np.empty((len(ref.q), 2))
            for k in range(len(ref.q)):
                kd.qpos[:] = ref.q[k]
                mujoco.mj_kinematics(p.m, kd)
                mujoco.mj_comPos(p.m, kd)
                com[k] = kd.subtree_com[p.trunk][:2]
            self._fk[key] = np.concatenate([(com - com[0]) @ e, [length]])
        return self._fk[key]

    @property
    def inputs(self) -> dict:
        """What the scripts built so far read: per file its sha256 and its entry's hashed subset, the freeze that
        covers them (None: the S1 view, or an unfrozen test set) and the plant's round-two inputs.  No path."""
        from ..stage1 import panels

        return dict(set=self.set, body=self.body, files={f: dict(self._files[f]) for f in sorted(self._files)},
                    freeze=None if self._freeze is None else dict(self._freeze),
                    plant_inputs=panels.key_context(self.body)["plant_inputs"])

    # ---------------------------------------------------------------- plans
    def segment_times(self, plan) -> list[tuple[float, float]]:
        """(t0, duration) of each item on the reference clock, as ``control.build_track`` lays them out."""
        objs = [self._load(item) for item in plan]
        out, t = [], 0.0
        for i, (item, obj) in enumerate(zip(plan, objs, strict=True)):
            if item[0] == "ref":
                dur = float(item[2]) * obj.period
            elif item[0] == "transition":
                nxt = objs[i + 1] if i + 1 < len(objs) else None
                if isinstance(nxt, Reference):
                    dt_last = nxt.step
                elif isinstance(nxt, Transition):
                    dt_last = float(nxt.t[1] - nxt.t[0])
                else:
                    dt_last = float(obj.t[-1] - obj.t[-2])
                dur = float(obj.t[-1] + dt_last)
            else:
                dur = float(item[1])
            out.append((t, dur))
            t = t + dur
        return out

    def _resolve(self, template) -> list:
        """A table row's plan with every "to the end" item sized: whole windows of its file until the plan reaches
        ``MAX_SECONDS`` at the commanded clock (at least one)."""
        plan = [tuple(x) for x in template]
        for i, item in enumerate(plan):
            if item[0] == "ref" and item[2] == END:
                probe = plan[:i] + [("ref", item[1], 1)]
                t0, window = self.segment_times(probe)[i]
                k = max(1, math.ceil((MAX_SECONDS * self.rate - t0) / window - 1e-9))
                plan[i] = ("ref", item[1], k)
        return plan

    # ---------------------------------------------------------------- scripts
    def script(self, task: str) -> Script:
        t = self.tasks.get(task)
        if t is None:
            raise KeyError(f"task {task!r}: choose from {', '.join(self.tasks)}")
        if not t["available"]:
            raise Absent(f"task {task!r}: {t['reason']}")
        key = ("task", task)
        if key not in self._scripts:
            self._scripts[key] = Script(self, [tuple(x) for x in t["plan"]], self._defs[task].seconds)
        return self._scripts[key]

    def from_plan(self, plan: list, track: control.Track | None = None) -> Script:
        """The script of any plan over the set (a bar condition's): the same side tables on the same files.
        ``track`` gives a track already built from this plan on this plant (a row's built condition), so that it
        is not built twice; scripts are kept by plan."""
        plan = [tuple(x) for x in plan]
        for item in plan:
            try:
                self._need(item)
            except Absent as why:
                raise tune.InputError(f"{self.set}/{self.body}: {why}") from None
        self._read(plan_files(plan))
        key = ("plan", tuple(plan), None if track is None else id(track))
        if key not in self._scripts:
            self._scripts[key] = Script(self, plan, track=track)
        return self._scripts[key]

    def _tables(self, plan, track: control.Track) -> list[_Seg]:
        """The side tables of ``track``'s segments (``Script``)."""
        objs = [self._load(item) for item in plan]
        segs = track.segments
        out = []
        heading = 0.0
        for i, (item, obj, seg) in enumerate(zip(plan, objs, segs, strict=True)):
            if _KIND[seg.kind] != {"ref": PERIODIC, "transition": FINITE, "hold": HOLD}[item[0]]:
                raise ValueError(f"segment {i} of the track is {seg.kind}, the plan has {item!r} there")
            s = _Seg()
            s.kind, s.name, s.t0, s.duration = _KIND[seg.kind], seg.name, float(seg.t0), float(seg.duration)
            s.step, s.n, s.t, s.nf = float(seg.step), int(seg.n), seg.t, 0
            s.swing, s.u, s.anchor, s.stand = (seg.info[k] for k in ("swing", "u", "anchor", "stand"))
            s.period = np.asarray(seg.info["period"], float)
            s.turn, s.pulse_s, s.walk, s.phase, s.q, s.qd, s.along = False, 0.0, False, None, None, None, None
            s.swing_len = None
            if s.kind == PERIODIC:
                e = self.refs_doc[item[1]]
                s.v, s.w = _num(e.get("speed_m_s")), _num(e.get("yaw_rate_rad_s"))
                s.stride_s = float(e["period_s"])
                s.swing_s = (1.0 - float(seg.duty)) * s.stride_s
                s.walk = s.v > 0.0
                if "phase" in obj.arrays:
                    ph = _unwrapped(obj.arrays["phase"])
                    turns = round(float(ph[-1] - ph[0]) + float(ph[-1] - ph[-2]))
                    s.phase = np.concatenate([ph, [ph[0] + turns]])
                foot, rp = self._fk_of(item)
                s.foot_rel, s.head_rp = np.concatenate([foot, foot[:1]]), np.concatenate([rp, rp[:1]])
                s.q, s.qd = obj.q, obj.qd
                s.along = self._along_of(item) if s.walk else None
                s.swing_len = _swing_lengths(np.asarray(s.swing), s.step)
            elif s.kind == FINITE:
                e = self.tr_doc[item[1]]
                a, b = (e.get("gaits") or {}).get("a") or {}, (e.get("gaits") or {}).get("b") or {}
                turn = e.get("turn") if isinstance(e.get("turn"), dict) else None
                s.nf = len(s.anchor)
                if turn is not None and turn.get("angle_rad") is not None and turn.get("rate_rad_s"):
                    sign = -1.0 if turn.get("direction") == "right" else 1.0
                    w0 = abs(float(turn["rate_rad_s"]))
                    s.turn, s.v, s.w = True, _num(a.get("speed")), sign * w0
                    s.pulse_s = abs(float(turn["angle_rad"])) / w0
                else:
                    s.v, s.w = _num(b.get("speed")), _num(b.get("yaw_rate"))
                g = b if _num(b.get("period")) > 0.0 else a
                s.stride_s = _num(g.get("period"), float(s.period[-1]))
                s.swing_s = (1.0 - _num(g.get("duty"), 0.6)) * s.stride_s
                dt_last = float(s.t[-1] - s.t[-2])
                own = float(obj.t[-1] - obj.t[-2])
                ph = np.asarray(obj.arrays["phase"], float)
                s.phase = np.concatenate([ph, [ph[-1] + (ph[-1] - ph[-2]) * dt_last / own]])
                foot, rp = self._fk_of(item)
                joined = i + 1 < len(plan) and plan[i + 1][0] != "hold"
                nxt = self._fk_of(plan[i + 1]) if joined else (foot[-1:], rp[-1:])
                s.foot_rel, s.head_rp = np.concatenate([foot, nxt[0][:1]]), np.concatenate([rp, nxt[1][:1]])
                s.q, s.qd = obj.q, obj.qd
            else:
                s.v = s.w = 0.0
                s.stride_s = float(seg.period)
                src, k = (i + 1, 0) if i == 0 else (i - 1, -1)  # (the pose ``control.build_track`` holds)
                foot, rp = self._fk_of(plan[src])
                s.foot_rel, s.head_rp = foot[k:][:1], rp[k:][:1]
                s.swing_s = 0.0
            s.heading0 = heading
            heading = heading + s.w * (min(s.duration, s.pulse_s) if s.turn else s.duration)
            out.append(s)
        for i, s in enumerate(out):  # a hold takes the planned swing of the gait beside it
            if s.kind == HOLD:
                near = [x for x in (out[i - 1] if i else None, out[i + 1] if i + 1 < len(out) else None)
                        if x is not None and x.kind != HOLD]
                s.swing_s = near[0].swing_s if near else 0.4 * s.stride_s
        return out

    def describe(self) -> dict:
        """The resolved task table, the absent tasks with their reasons and the files read with their hashes: what
        a run's settings hold of its references (no path)."""
        return dict(set=self.set, body=self.body, rate=self.rate, omega=self.omega, crouch_frac=self.crouch_frac,
                    max_seconds=MAX_SECONDS, command_rule=COMMAND_RULE,
                    tasks={n: dict(t) for n, t in self.tasks.items() if t["available"]},
                    absent={n: t["reason"] for n, t in self.tasks.items() if not t["available"]},
                    inputs=self.inputs)


__all__ = ["Absent", "COMMAND_RULE", "END", "Library", "MAX_SECONDS", "S1_TASKS", "STOCK_SET", "Script", "TASKS",
           "TaskDef", "fk_tables", "foot_rel_of", "head_body", "head_home", "head_rp_of", "plan_files", "s1_root"]
