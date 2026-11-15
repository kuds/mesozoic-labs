"""Stage 2 moves: one builder of starts, speed changes, stops and finite turns for any ``gait.GaitSpec`` on any body.

A *move* joins two periodic gaits, or a gait and the home stance.  It is Stage 1's transition
(``designlab/stage1/transitions.py``) with everything that module hard-codes passed in as data (``Schedule``): the
phase functions of the plan's parameters, the output window, which swings exist, and, for a commanded move, the
command phase.  Two solve paths:

* **Stage 1's construction** (``Move.causal`` false: a start, a speed change, Stage 1's one-step stop and every one of
  its 18 transitions).  ``plan`` is ``transitions.plan`` line for line with the schedule as data, and ``generate`` runs
  ``transitions.generate``'s loop on Stage 1's own ``initial_state``, ``com_path``, ``solve_frames``, ``derivatives``,
  ``multibody_zmp`` and ``idyn``, so the floating-point operations are Stage 1's and the arrays are its arrays bit for
  bit (``legacy_check``; ``stage2/sets/legacy_check.json``).
* **Causal commanded moves** (``Move.causal``: the two-touchdown stop and the finite turn).  The reference is the
  periodic walk, frame for frame, up to the command frame; the preview controller first sees the commanded plan at
  that frame; the move's file begins there (``RULES["S2A-causal"]``).  A command may only be given at a lift-off.

This module writes no set file: ``generate``, ``arrays`` and ``summary`` return values and the set builder places them.
A ``Move`` holds function objects (its schedule's phase functions, its laws): it is neither picklable nor hashable, so
a pool worker gets ``(body, name)`` and builds the move itself, and records hold ``Move.definition()``.

The look (later): every law of the plan comes from ``Move.laws`` (``Laws``; ``TODAY`` is Stage 1's lines verbatim).
With ``laws.solver == "look"`` the frame solver, the inverse dynamics, the start state and the joins come from
``designlab.stage2.moves_look``, loaded by name; ``TODAY`` never reaches it.

Run from hardware/trex_form_study (one BLAS thread; the three thread variables must be 1 or unset):
    python -m designlab.stage2.moves legacy --body 75-M-H-10 [--name start ...] [--procs 1]
        [--stage1-run stage1/runs/2026-11-08] [--record stage2/sets/legacy_check.json]
regenerates Stage 1's transitions of a body through this module into a temporary directory, compares each npz's sha256
with the run's ``transitions.json``, writes or merges the record when ``--record`` is given, and exits 1 on a
difference.
"""

from __future__ import annotations

import os
import sys

if __name__ == "__main__":  # before any numpy import, as look.py: one core per process (bit-reproducible arrays)
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import importlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import subprocess  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from dataclasses import asdict, dataclass, replace  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Callable  # noqa: E402

import numpy as np  # noqa: E402

from .. import gait, sizing, statics, turning  # noqa: E402
from ..stage1 import plant, refs, transitions  # noqa: E402
from ..stage1.transitions import PHASE_TOL, SETTLE_STRIDES, SHIFT_STRIDES, SIDES, SUB, _ramp, _rot2  # noqa: E402

STUDY = Path(__file__).resolve().parents[2]
N_FRAMES = transitions.N_FRAMES
THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
STAGE1_RUN = STUDY / "stage1" / "runs" / "2026-11-08"

#: A turn's ramps last this long in reference time, for every set, body and rate (``RULES["S2A-ramp"]``).  The shared
#: constant is work package R's ``setio.RAMP_S``; when ``setio`` is merged this line becomes
#: ``from .setio import RAMP_S`` (same value) and nothing else changes.
RAMP_S = 0.375
#: The stride rule (``RULES["S2A-stride"]``): the stride factors tried, largest first; the thigh-twist limits of the
#: kinematic pass and of the built reference (fractions of the usable range, worse leg); the step-downs allowed.
STRIDE_GRID = (1.00, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60)
TWIST_KINEMATIC = 0.85
TWIST_FINAL = 0.87
STEPS_DOWN_MAX = 2
#: The excused causal transient fades to zero over these phases after the command (``RULES["S2A-fade"]``).
CAUSAL_FADE = (0.5, 1.0)
#: Gates of a move (``RULES["S2A-gates"]``).
CONSISTENCY_LIMIT_M = 2.5e-3
ZMP_HISTORY_LIMIT_M = 4.0e-3
CAUSAL_COP_MARGIN_M = 0.010
JOINT_AT_COMMAND_RAD = 1e-3
#: Gates read on the reference itself: a turn's heading at the bar's instant (deg) and its closure (rad); a stop's
#: touchdowns after the command and its 1 Hz low-passed forward acceleration (m/s^2).
TURN_HEADING_MIN_DEG = 80.0
TURN_CLOSURE_RAD = 1e-9
KAPPA_TOL = 1e-6
STOP_TOUCHDOWNS = 2
STOP_ACCEL_LP_LIMIT_M_S2 = 1.5
#: Sampling rate of the reference's own acceleration reading (Hz); the 1 Hz filter is ``metrics.lowpass``.
ACCEL_FS = 500.0
#: integral over [0, 1] of the minimum-jerk step squared (``transitions._ramp``): 100/7 - 300/8 + 345/9 - 18 + 36/11.
RAMP_SQ_INTEGRAL = 181.0 / 462.0

_PROTO = "constants set on the 14 Nov prototypes of V0 on 75-M-H-10, before any closed-loop run"
#: The pre-registered rules a move is made by (rule id -> text); a move records the ids it used (``Move.rules``).
RULES = {
    "S2A-causal": (
        "a commanded move (a stop, a turn) is causal: its reference is the periodic walk, frame for frame, up to the "
        "command frame; the preview controller first sees the commanded plan at that frame (the control law in its "
        "incremental form, so that new preview information enters only through its own increments); the move's file "
        "begins at the command frame, which is the command instant on a track. A command may only be given at a "
        "lift-off (a multiple of half a stride). Named and not used: Stage 1's convention (decision 18: the command "
        "is the start of the reference's own ramp), under which the reference's CoM leaves the walk before the "
        "command; decision 18's reading is kept in every stop record (reference.stage1_stop). Starts and speed "
        "changes stay Stage 1's construction"),
    "S2A-rule-F": (
        "committed footprints: a stance that began before the command keeps the footprint of the walk's own path "
        "(gait a's constant period, speed and yaw rate, from the same origin); a swing that lifts at or after the "
        "command lands on the commanded path"),
    "S2A-fade": (
        "the causal transient (the cart-table ZMP of the preview controller's own output less its ZMP reference: what "
        "a controller without foreknowledge cannot track) is excused from the multibody-ZMP correction with the "
        f"weight 1 - s((p - c - {CAUSAL_FADE[0]:g}) / {CAUSAL_FADE[1] - CAUSAL_FADE[0]:g}), s the minimum-jerk step: "
        f"in full up to {CAUSAL_FADE[0]:g} stride after the command phase c, not at all from {CAUSAL_FADE[1]:g} "
        "stride after it; from there the loop corrects the multibody ZMP against the plan's target, as Stage 1's"),
    "S2A-ramp": (
        f"a turn's yaw rate and stride period ramp together over {RAMP_S:g} s of reference time (minimum-jerk), for "
        f"every set, body and rate: r = 2 x {RAMP_S:g} / (T_walk + T_turn) strides"),
    "S2A-appendage": (
        "inside a move whose stride period T(p) is shorter than the walk's, the tail's and the neck's swing amplitudes "
        "are multiplied by (T(p) / T_walk)^2 (constant angular acceleration amplitude), for every set, look sets "
        "included; a look choice (lead's ruling 1)"),
    "S2A-stride": (
        "inside the turn the stride period is f x T_walk at the same path speed; f is the largest value of "
        + ", ".join(f"{f:g}" for f in STRIDE_GRID) + " for which the finite turn's own kinematic pass (the causal plan "
        "solved once, no ZMP correction), in both directions, has ik_ok and uses at most "
        f"{TWIST_KINEMATIC:g} of the usable thigh-twist range on the worse leg; the reference built with all passes "
        f"must show ik_ok and at most {TWIST_FINAL:g}, else f steps down once (at most {STEPS_DOWN_MAX:g} times in "
        f"all); {_PROTO}"),
    "S2A-gates": (
        "a move passes when: ik_ok; its last output stride is within "
        f"{transitions.G3_COM_MM:g} mm of gait b's CoM and {transitions.G3_ZMP_MM:g} mm of its multibody ZMP (the "
        "home stance for a stop); a move of Stage 1's construction: its first stride the same against gait a, and the "
        f"final ZMP error at most {1e3 * ZMP_HISTORY_LIMIT_M:g} mm; a causal move: the warm-up stride and the command "
        f"frame within {transitions.G3_COM_MM:g} mm of the walk's CoM, joints within {JOINT_AT_COMMAND_RAD:g} rad at "
        f"the command frame, multibody ZMP within {transitions.G3_ZMP_MM:g} mm over the warm-up frames up to two "
        "frames before the command (the ZMP at the command frame is recorded, not gated: its central difference "
        f"straddles the command), final consistency error at most {1e3 * CONSISTENCY_LIMIT_M:g} mm, ZMP error from "
        f"one stride after the command at most {1e3 * ZMP_HISTORY_LIMIT_M:g} mm, single-support CoP at least "
        f"{1e3 * CAUSAL_COP_MARGIN_M:g} mm inside the pad; a turn: total heading within {TURN_CLOSURE_RAD:g} rad of "
        f"its angle, heading at the bar's instant at least {TURN_HEADING_MIN_DEG:g} deg on the nominal commanded "
        f"clock and on the slowest, thigh twist at most {TWIST_FINAL:g} of the usable range; the two-touchdown stop: "
        f"exactly {STOP_TOUCHDOWNS:g} touchdowns after the command, forward CoM speed below "
        f"{100 * transitions.STOP_SPEED_FRAC:g} % of the walk's within {transitions.STOP_WITHIN_STRIDES:g} strides of "
        f"the command, 1 Hz low-passed forward CoM acceleration within {STOP_ACCEL_LP_LIMIT_M_S2:g} m/s^2 (a zero-phase "
        "order-2 Butterworth at 1 Hz turns a speed change dv into at most 2.22 x dv m/s^2, so this term detects "
        f"rocking, not abruptness; the unfiltered peak is recorded beside it); {_PROTO}"),
}


class MoveError(RuntimeError):
    pass


# ------------------------------------------------------------------------------------------------
# Laws (the look's entry point; today's are Stage 1's lines)
# ------------------------------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class PlanCtx:
    """What a law may read; built by ``plan`` before the laws are called.

    ``period_factor`` is called first, with a context whose ``t``, ``heading``, ``swing`` and ``swing_u`` are None
    and whose ``swings`` and ``stances`` are empty (they follow from the periods).  ``lift`` is called frame by frame
    while ``swing`` and ``swing_u`` are being filled.  Every other law sees the finished context."""

    phase: np.ndarray  # [N] computed frames, strides
    t: np.ndarray  # [N] s
    fn: dict  # the schedule's phase functions (T, duty, lift, v, w, amp, head_amp, tail_amp, crouch)
    amp: np.ndarray  # [N] the amplitude ramp
    weight: np.ndarray  # [N] share of gait b in [0, 1] (0: gait a's law, 1: gait b's; the schedule's ramp)
    heading: np.ndarray  # [N] rad
    swing: np.ndarray  # [N, 2] bool
    swing_u: np.ndarray  # [N, 2]
    swings: list  # (side, lift_off, touchdown, lift, footprint from, footprint to)
    stances: list  # (side, from, to, footprint)
    geo: transitions.Geometry
    a: "GaitX"
    b: "GaitX"
    command: float | None


@dataclass(frozen=True)
class Laws:
    """The laws of a plan.  ``TODAY`` reproduces ``transitions.plan``; the look's come from ``moves_look``."""

    name: str  # "today" | "look"
    lift: Callable[[np.ndarray, PlanCtx, tuple], np.ndarray]  # (u in [0, 1], ctx, swing) -> lift fraction
    height: Callable[[PlanCtx], np.ndarray]  # -> pelvis z [N]
    neck: Callable[[PlanCtx], np.ndarray]  # -> rad [N]
    tail: Callable[[PlanCtx], np.ndarray]  # -> rad [N]
    trunk_roll: Callable[[PlanCtx], np.ndarray] | None = None  # -> rad [N] about the heading axis; None: 0
    foot: Callable[[PlanCtx], dict] | None = None  # -> {pitch [N,2], contact_offset [N,2,3], rolling [N,2]}; None: flat
    zmp_point: Callable[[PlanCtx, tuple], np.ndarray] | None = None  # (ctx, stance) -> world xy; None: pad centroid
    period_factor: Callable[[PlanCtx], Callable] | None = None  # -> phase -> factor on fn["T"]; None: 1
    solver: str = "stage1"  # "stage1": transitions.solve_frames, idyn, initial_state, compare
    #                         "look":   moves_look.solve_frames, idyn, initial_state, compare, arrays_extra


#: Stage 1's laws, verbatim (``transitions.py:378``, ``:420``, ``:421``, ``:422``).
TODAY = Laws(
    name="today",
    lift=lambda u, ctx, swing: math.sin(math.pi * u) ** 2,
    height=lambda ctx: ctx.geo.root0[2] - ctx.amp * ctx.fn["crouch"](ctx.phase),
    neck=lambda ctx: ctx.amp * ctx.fn["head_amp"](ctx.phase) * np.sin(4 * np.pi * ctx.phase),
    tail=lambda ctx: ctx.amp * ctx.fn["tail_amp"](ctx.phase) * np.sin(2 * np.pi * ctx.phase + np.pi / 2),
)


def _solver(laws: Laws):
    """The module that holds ``solve_frames``, ``idyn``, ``initial_state`` and ``compare`` for these laws."""
    if laws.solver == "look":
        return importlib.import_module("designlab.stage2.moves_look")
    return transitions


# ------------------------------------------------------------------------------------------------
# Gaits, schedules, moves
# ------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class GaitX:
    """A periodic gait as a move sees it: its GaitSpec on a body, or the stand."""

    name: str  # the reference's name in the set ("v0.4"), or "stand"
    spec: gait.GaitSpec | None  # None: the stand
    H: float = math.nan
    yaw_rate: float = 0.0  # rad/s, for a periodic turn gait (Stage 1's six; look turns later)
    in_place: bool = False  # a turn in place: speed 0.0 (Stage 1's in_place_0.50)
    split: str = "statics"  # "exact" for a turn gait

    @classmethod
    def of(cls, name, spec, H, yaw_rate=0.0, in_place=False) -> "GaitX":
        if name == "stand" or spec is None:
            raise ValueError("a gait needs a spec and a name other than 'stand' (GaitX.stand() is the stand)")
        return cls(name, spec, H, yaw_rate, bool(in_place), "exact" if (yaw_rate != 0.0 or in_place) else "statics")

    @classmethod
    def stand(cls) -> "GaitX":
        return cls("stand", None)

    def legacy(self) -> transitions.Gait:
        """``transitions.gait_of``'s constructor call, argument for argument; the duty is ``spec.duty`` when
        ``spec.swing_time`` is None (Stage 1's value and type), else ``describe(H)["duty"]`` (F3)."""
        if self.spec is None:
            return transitions.Gait("stand")
        spec, H = self.spec, self.H
        info = spec.describe(H)
        duty = spec.duty if spec.swing_time is None else info["duty"]
        return transitions.Gait(self.name, info["period"], duty, spec.lift_frac * H,
                                0.0 if self.in_place else info["speed"], self.yaw_rate, spec.head_amp, spec.tail_amp,
                                spec.crouch * H, spec.zmp_iters, spec.zmp_gain, self.split)

    def plain(self) -> dict:
        """JSON-plain description (a move's identity holds it)."""
        return refs._plain(dict(name=self.name, spec=None if self.spec is None else asdict(self.spec),
                                H=None if self.spec is None else float(self.H), yaw_rate=float(self.yaw_rate),
                                in_place=self.in_place, split=self.split))


@dataclass(frozen=True)
class Schedule:
    """Everything ``transitions.plan`` hard-codes, as data."""

    kind: str  # "start" | "stop" | "walk" (Stage 1's three) | "turn"
    window: tuple  # output [first, last) in gait phase; last is an integer
    fn: dict  # phase -> T, duty, lift, v, w, amp, head_amp, tail_amp, crouch (as transitions._param_fns)
    ramps: dict  # recorded as given
    lift_min: float | None = None  # swings exist only for lift-offs >= this (a start: 0.0)
    lift_max: float | None = None  # ... and <= this (a stop: 0.0)
    command: float | None = None  # command phase, a multiple of 0.5; None: Stage 1's non-causal construction
    rule_f: bool = False  # committed footprints (RULES["S2A-rule-F"]); needs command
    pre: float = 1.0  # computed strides before ``first`` (discarded warm-up)
    post: float = 0.25


@dataclass(frozen=True)
class Move:
    name: str  # file stem
    a: GaitX
    b: GaitX
    schedule: Schedule
    builder: str  # "start" | "change" | "stop_one_step" | "stop_two_step" | "turn" | "stage1"
    args: dict  # the builder's arguments as JSON-plain values: with a, b this IS the move's identity
    causal: bool = False
    laws: Laws = TODAY
    rules: tuple = ()  # ids of the pre-registered rules it was made by (recorded)

    def definition(self) -> dict:
        """The move's identity as JSON-plain values; no function object."""
        return refs._plain(dict(builder=self.builder, args=dict(self.args), specs=dict(a=self.a.plain(),
                                                                                         b=self.b.plain()),
                                causal=self.causal, laws=self.laws.name, rules=list(self.rules)))


def _const(x):
    return lambda p: x + 0 * np.asarray(p, float)


def legacy_schedule(a: transitions.Gait, b: transitions.Gait) -> Schedule:
    """Stage 1's schedule: ``fn, ramps = transitions._param_fns(kind, a, b)`` (the same function objects), the window
    ``transitions.WINDOWS[kind]``, lift_min 0.0 for a start, lift_max 0.0 for a stop, no command."""
    kind = transitions.kind_of(a.name, b.name)
    fn, ramps = transitions._param_fns(kind, a, b)
    return Schedule(kind=kind, window=transitions.WINDOWS[kind], fn=fn, ramps=ramps,
                    lift_min=0.0 if kind == "start" else None, lift_max=0.0 if kind == "stop" else None,
                    pre=transitions.PRE_STRIDES, post=transitions.POST_STRIDES)


def _need_walk(g: GaitX, what: str) -> None:
    if g.spec is None:
        raise ValueError(f"{what} needs a walk, not the stand")


def start(name, b: GaitX) -> Move:
    """Stand to gait ``b``: Stage 1's start."""
    _need_walk(b, "a start")
    return Move(name, GaitX.stand(), b, legacy_schedule(transitions.Gait("stand"), b.legacy()), "start", {},
                rules=("S2A-gates",))


def change(name, a: GaitX, b: GaitX, ramp_strides: float = 1.0) -> Move:
    """Gait ``a`` to gait ``b``, every parameter ramping over ``ramp_strides`` strides from phase 0 (Stage 1's
    gait-to-gait transition; with one stride of ramp its schedule is Stage 1's own).  Refuses two crouches: a move
    has one LIPM frequency."""
    _need_walk(a, "a change")
    _need_walk(b, "a change")
    ga, gb = a.legacy(), b.legacy()
    if ga.crouch != gb.crouch:
        raise ValueError(f"{name}: {a.name} crouches {ga.crouch:g} m and {b.name} {gb.crouch:g} m; a move has one "
                         "crouch (one LIPM frequency)")
    L = float(ramp_strides)
    if L <= 0:
        raise ValueError("ramp_strides must be positive")
    if L == transitions.WALK_RAMP_STRIDES:
        sch = legacy_schedule(ga, gb)
    else:
        def mix(xa, xb):
            return lambda p: xa + (xb - xa) * _ramp(np.asarray(p, float) / L)

        fn = dict(T=mix(ga.period, gb.period), duty=mix(ga.duty, gb.duty), lift=mix(ga.lift, gb.lift),
                  v=mix(ga.speed, gb.speed), w=mix(ga.yaw_rate, gb.yaw_rate), amp=lambda p: 1.0 + 0 * np.asarray(p, float),
                  head_amp=mix(ga.head_amp, gb.head_amp), tail_amp=mix(ga.tail_amp, gb.tail_amp),
                  crouch=mix(ga.crouch, gb.crouch))
        first = transitions.WINDOWS["walk"][0]
        sch = Schedule(kind="walk", window=(first, int(math.ceil(L)) + 3), fn=fn, ramps=dict(all=[0.0, L]),
                       pre=transitions.PRE_STRIDES, post=transitions.POST_STRIDES)
    return Move(name, a, b, sch, "change", dict(ramp_strides=L), rules=("S2A-gates",))


def stop_one_step(name, a: GaitX) -> Move:
    """Gait ``a`` to the stand: Stage 1's one-step stop (not causal; a labelled comparison in a set)."""
    _need_walk(a, "a stop")
    return Move(name, a, GaitX.stand(), legacy_schedule(a.legacy(), transitions.Gait("stand")), "stop_one_step", {},
                rules=("S2A-gates",))


def stop_two_step(name, a: GaitX) -> Move:
    """The two-touchdown stop: Stage 1's stop plan (the left foot lifts last, at phase 0, and lands beside the right)
    commanded causally at the right foot's lift-off, phase -0.5.  After the command the right foot lands on its
    periodic footprint, then the left beside it."""
    _need_walk(a, "a stop")
    sch = replace(legacy_schedule(a.legacy(), transitions.Gait("stand")), window=(-0.5, 3), command=-0.5,
                  rule_f=False)
    return Move(name, a, GaitX.stand(), sch, "stop_two_step", {}, causal=True,
                rules=("S2A-causal", "S2A-fade", "S2A-gates"))


def _fine_grid(first, last, pre, post, n):
    """``plan``'s fine phase grid and the index of ``first`` on it."""
    lo, hi = first - pre, last + post
    pf = np.arange(int(round((lo - 3) * n * SUB)), int(round((hi + 3) * n * SUB)) + 1) / (n * SUB)
    return pf, int(np.argmin(np.abs(pf - first)))


def _planned_heading(fn: dict, window, pre: float, post: float, n: int = N_FRAMES) -> float:
    """The heading a plan of these phase functions integrates from its first to its last output frame (``plan``'s
    own trapezoid on its own grid)."""
    first, last = window
    lo, hi = first - pre, last + post
    pf, i0 = _fine_grid(first, last, pre, post, n)
    y = fn["w"](pf) * fn["T"](pf)
    c = np.concatenate([[0.0], np.cumsum(0.5 * (y[1:] + y[:-1]) * (1.0 / (n * SUB)))])
    thf = c - c[i0]
    phase = np.arange(int(round(lo * n)), int(round(hi * n))) / n
    out = slice(int(round((first - lo) * n)), int(round((last - lo) * n)))
    heading = np.interp(phase, pf, thf)[out]
    return float(heading[-1] - heading[0])


def _turn_fns(g: transitions.Gait, w0: float, T_t: float, p0: float, p1: float, r: float, kappa: float) -> dict:
    T_w = g.period

    def sig(p):
        p = np.asarray(p, float)
        return _ramp((p - p0) / r) - _ramp((p - p1) / r)

    def T(p):
        return T_w + (T_t - T_w) * sig(p)

    return dict(T=T, duty=_const(g.duty), lift=_const(g.lift), v=_const(g.speed), w=lambda p: kappa * w0 * sig(p),
                amp=_const(1.0), head_amp=lambda p: g.head_amp * (T(p) / T_w) ** 2,
                tail_amp=lambda p: g.tail_amp * (T(p) / T_w) ** 2, crouch=_const(g.crouch))


def turn_design(g: transitions.Gait, angle: float, rate: float, stride_factor: float, ramp_s: float = RAMP_S,
                p0: float | None = None, pre: float = 1.0, post: float = 0.25) -> dict:
    """The numbers of a finite turn from walk ``g``: command phase, ramp and hold lengths, window, and ``kappa``, the
    factor on the yaw rate that closes the plan's own integrated heading on ``angle`` (1 first, then one re-plan).
    ``p0``: the command phase; None (every reference): the inside foot's lift-off in the stride that starts at 0."""
    A, w0, f = float(angle), float(rate), float(stride_factor)
    if not (A > 0 and w0 != 0 and f > 0 and ramp_s > 0):
        raise ValueError("a turn needs a positive angle, a signed non-zero rate, a positive stride factor and ramp")
    T_w = float(g.period)
    T_t = f * T_w
    r = 2.0 * ramp_s / (T_w + T_t)  # a ramp lasts integral T dp = r (T_w + T_t) / 2
    inside = 0.0 if w0 > 0 else 0.5  # the inside foot's lift-off: left for a left turn
    p0 = inside if p0 is None else float(p0)
    if abs((p0 - inside) - round(p0 - inside)) > PHASE_TOL:
        raise ValueError(f"a turn is commanded at a lift-off of its inside foot, not at phase {p0:g}")
    L = (A / abs(w0) + (T_t - T_w) * r * (1.0 - 2.0 * RAMP_SQ_INTEGRAL)) / T_t  # integral w T dp = A, for L >= r
    if L < r:
        raise ValueError(f"turn of {A:g} rad at {w0:g} rad/s, stride factor {f:g}: the hold would be negative "
                         f"(L {L:g} < ramp {r:g} strides); the closed form needs L >= r")
    p1 = p0 + L
    window = (p0, int(math.ceil(p1 + r)) + 2)
    h1 = _planned_heading(_turn_fns(g, w0, T_t, p0, p1, r, 1.0), window, pre, post)
    kappa = A / abs(h1)
    if abs(kappa - 1.0) >= KAPPA_TOL:
        raise MoveError(f"turn: kappa {kappa!r} is not within {KAPPA_TOL:g} of 1 (planned heading {h1!r} rad)")
    total = _planned_heading(_turn_fns(g, w0, T_t, p0, p1, r, kappa), window, pre, post)
    return dict(angle_rad=A, rate_rad_s=w0, direction="left" if w0 > 0 else "right",
                first_swing="left" if w0 > 0 else "right", stride_factor=f, ramp_s=float(ramp_s), ramp_strides=r,
                hold_strides=L - r, turn_strides=L + r, period_walk_s=T_w, period_in_turn_s=T_t, p0=p0, p1=p1,
                window=window, kappa=kappa, heading_total_rad=total)


def turn(name, a: GaitX, angle: float, rate: float, stride_factor: float, ramp_s: float = RAMP_S) -> Move:
    """A finite turn of ``angle`` rad from walk ``a`` to the same walk on the new heading, at the signed yaw rate
    ``rate`` (positive: left), causal, commanded at the inside foot's lift-off.  Yaw rate and stride period ramp
    together over ``ramp_s`` of reference time, hold, and ramp back; inside the turn the stride period is
    ``stride_factor`` x the walk's at the same path speed, duty and lift; the tail's and the neck's amplitudes carry
    (T(p) / T_walk)^2."""
    return _turn(name, a, angle, rate, stride_factor, ramp_s)


def _turn(name, a: GaitX, angle, rate, stride_factor, ramp_s=RAMP_S, p0: float | None = None) -> Move:
    """``turn`` with the command at another lift-off of the inside foot (``p0``; the tests' use: the same turn one
    stride earlier).  The argument is part of the move's identity only when it is given."""
    _need_walk(a, "a turn")
    g = a.legacy()
    if g.yaw_rate != 0.0 or a.in_place:
        raise ValueError("a finite turn starts from a straight walk")
    args = dict(angle=float(angle), rate=float(rate), stride_factor=float(stride_factor), ramp_s=float(ramp_s))
    if p0 is not None:
        args["p0"] = float(p0)
    d = turn_design(g, **args)
    p0, p1, r = d["p0"], d["p1"], d["ramp_strides"]
    fn = _turn_fns(g, d["rate_rad_s"], d["period_in_turn_s"], p0, p1, r, d["kappa"])
    sch = Schedule(kind="turn", window=d["window"], fn=fn, ramps=dict(turn_in=[p0, p0 + r], turn_out=[p1, p1 + r]),
                   command=p0, rule_f=True, pre=1.0, post=0.25)
    return Move(name, a, a, sch, "turn", args, causal=True,
                rules=("S2A-causal", "S2A-rule-F", "S2A-fade", "S2A-ramp", "S2A-appendage", "S2A-stride", "S2A-gates"))


def stage1_gait(body: str, name: str, H: float) -> GaitX:
    """The gait of a Stage 1 reference name (``refs.reference_names``) or "stand", as ``transitions.gait_of``."""
    if name == "stand":
        return GaitX.stand()
    walks = refs.walk_specs(body)
    if name in walks:
        return GaitX.of(name, walks[name], H)
    c = refs.turn_cases()[name]
    return GaitX.of(name, sizing.NOMINAL, H, yaw_rate=c["yaw_rate"], in_place=c["in_place"])


def stage1(body: str, name: str, H: float) -> Move:
    """One of ``transitions.TRANSITIONS`` on a Stage 1 body, with Stage 1's schedule."""
    a_name, b_name = transitions.TRANSITIONS[name]
    a, b = stage1_gait(body, a_name, H), stage1_gait(body, b_name, H)
    return Move(name, a, b, legacy_schedule(a.legacy(), b.legacy()), "stage1", dict(body=body, name=name))


# ------------------------------------------------------------------------------------------------
# Plan: transitions.plan with the schedule as data
# ------------------------------------------------------------------------------------------------
def _weight(sch: Schedule, from_stand: bool, to_stand: bool, phase: np.ndarray) -> np.ndarray:
    """Share of gait b's laws per frame: the schedule's own ``fn["weight"]`` if it has one; 1 from a stand, 0 into
    one; the parameter ramp of a gait-to-gait schedule (``ramps["all"]``); else 0."""
    if "weight" in sch.fn:
        return np.asarray(sch.fn["weight"](phase), float)
    if from_stand:
        return np.ones(len(phase))
    if not to_stand and "all" in sch.ramps:
        p_lo, p_hi = sch.ramps["all"]
        return _ramp((phase - p_lo) / (p_hi - p_lo))
    return np.zeros(len(phase))


def plan(move: Move, geo: transitions.Geometry, n: int = N_FRAMES) -> transitions.Plan:
    """The gait plan of a move: ``transitions.plan``'s body with the move's schedule and laws.  The returned plan
    also carries ``ctx`` (the ``PlanCtx`` the laws read) and ``extras`` (what the optional laws returned)."""
    sch, laws = move.schedule, move.laws
    a, b = move.a.legacy(), move.b.legacy()
    from_stand, to_stand = move.a.spec is None, move.b.spec is None
    cmd = sch.command
    if cmd is not None and abs(2.0 * cmd - round(2.0 * cmd)) > PHASE_TOL:
        raise ValueError(f"{move.name}: a command may only be given at a lift-off (a multiple of half a stride), not "
                         f"at phase {cmd:g}: at any other phase a swing is in the air")
    if sch.rule_f and cmd is None:
        raise ValueError(f"{move.name}: committed footprints (rule F) need a command phase")
    fn, ramps = sch.fn, sch.ramps
    first, last = sch.window
    lo, hi = first - sch.pre, last + sch.post
    # --- fine phase grid: time, heading and path origin
    pf = np.arange(int(round((lo - 3) * n * SUB)), int(round((hi + 3) * n * SUB)) + 1) / (n * SUB)
    i0 = int(np.argmin(np.abs(pf - first)))
    Tf, wf, vf = fn["T"](pf), fn["w"](pf), fn["v"](pf)
    phase = np.arange(int(round(lo * n)), int(round(hi * n))) / n
    K = len(phase)
    amp = fn["amp"](phase)
    weight = _weight(sch, from_stand, to_stand, phase)
    if laws.period_factor is not None:
        pre_ctx = PlanCtx(phase=phase, t=None, fn=fn, amp=amp, weight=weight, heading=None, swing=None, swing_u=None,
                          swings=[], stances=[], geo=geo, a=move.a, b=move.b, command=cmd)
        Tf = Tf * laws.period_factor(pre_ctx)(pf)
    dp = 1.0 / (n * SUB)

    def cumint(y):  # cumulative trapezoid, zero at phase ``first``
        c = np.concatenate([[0.0], np.cumsum(0.5 * (y[1:] + y[:-1]) * dp)])
        return c - c[i0]

    tf = cumint(Tf)
    thf = cumint(wf * Tf)
    Pf = np.column_stack([cumint(vf * Tf * np.cos(thf)), cumint(vf * Tf * np.sin(thf))]) + geo.origin

    def pose(p):
        p = float(np.clip(p, pf[0], pf[-1]))
        return np.array([np.interp(p, pf, Pf[:, 0]), np.interp(p, pf, Pf[:, 1])]), float(np.interp(p, pf, thf))

    def period(p):
        return float(np.interp(p, pf, Tf))

    pose_pre = None
    if sch.rule_f:  # the walk's own path: gait a's constant period, speed and yaw rate from the same origin
        Tp, wp, vp = _const(a.period)(pf), _const(a.yaw_rate)(pf), _const(a.speed)(pf)
        thp = cumint(wp * Tp)
        Pp = np.column_stack([cumint(vp * Tp * np.cos(thp)), cumint(vp * Tp * np.sin(thp))]) + geo.origin

        def pose_pre(p):
            p = float(np.clip(p, pf[0], pf[-1]))
            return np.array([np.interp(p, pf, Pp[:, 0]), np.interp(p, pf, Pp[:, 1])]), float(np.interp(p, pf, thp))

    # --- swings: left lifts at integer phases, right at half-integers
    swings = {}
    for side, off in (("left", 0.0), ("right", 0.5)):
        events = [k + off for k in range(int(math.floor(lo)) - 3, int(math.ceil(hi)) + 4)]
        if sch.lift_min is not None:
            events = [e for e in events if e >= sch.lift_min]
        if sch.lift_max is not None:
            events = [e for e in events if e <= sch.lift_max]
        swings[side] = [(e, e + 1.0 - float(fn["duty"](e)), float(fn["lift"](e))) for e in events]

    def footprint(side, stance_from, stance_to):
        """Footprint of a stance [stance_from, stance_to] (None: before the first / after the last swing)."""
        if stance_from is None:  # the initial stand: the path frame at rest at its start
            P, th = pose(pf[0])
        elif stance_to is None:  # the final stand: the path frame's final pose
            P, th = pose(pf[-1])
        elif pose_pre is not None and stance_from < cmd - PHASE_TOL:  # rule F: planted before the command
            P, th = pose_pre(0.5 * (stance_from + stance_to))
        else:
            P, th = pose(0.5 * (stance_from + stance_to))
        rel = geo.feet0[side][:2] - geo.origin
        return P + _rot2(th) @ rel, th

    sw_list = []
    stances = {}  # side -> list of (from, to, xy, yaw)
    for side in SIDES:
        ev = swings[side]
        st = []
        for i in range(len(ev) + 1):
            s_from = ev[i - 1][1] if i > 0 else (None if from_stand else ev[0][0] - 1.0 + (1.0 - float(fn["duty"](ev[0][0] - 1.0))))
            s_to = ev[i][0] if i < len(ev) else (None if to_stand else ev[-1][0] + 1.0)
            xy, th = footprint(side, s_from, s_to)
            st.append((s_from, s_to, xy, th))
        stances[side] = st
        for i, (e_lo, e_td, lift) in enumerate(ev):
            f0, f1 = st[i], st[i + 1]
            sw_list.append((side, e_lo, e_td, lift, (f0[2], f0[3]), (f1[2], f1[3])))
    # --- frames
    t = np.interp(phase, pf, tf)
    out = slice(int(round((first - lo) * n)), int(round((last - lo) * n)))
    foot_pos, foot_yaw = np.zeros((K, 2, 3)), np.zeros((K, 2))
    swing = np.zeros((K, 2), bool)
    swing_u = np.full((K, 2), np.nan)
    heading = np.interp(phase, pf, thf)
    path = np.column_stack([np.interp(phase, pf, Pf[:, 0]), np.interp(phase, pf, Pf[:, 1])])
    swings_sorted = sorted(sw_list, key=lambda s: s[1])
    ctx = PlanCtx(phase=phase, t=t, fn=fn, amp=amp, weight=weight, heading=heading, swing=swing, swing_u=swing_u,
                  swings=swings_sorted, stances=[(side, s[0], s[1], (s[2], s[3])) for side in SIDES for s in stances[side]],
                  geo=geo, a=move.a, b=move.b, command=cmd)
    sw_of = {side: [s for s in sw_list if s[0] == side] for side in SIDES}  # per side, in lift-off order
    stance_i = np.zeros((K, 2), int)
    for j, side in enumerate(SIDES):
        ev, st = swings[side], stances[side]
        z0 = geo.feet0[side][2]
        for k, p in enumerate(phase):
            i = int(np.searchsorted([e[0] - PHASE_TOL for e in ev], p, side="right"))  # swings lifted by p
            # i swings have lifted off by phase p; is the latest still in the air?
            if i > 0 and p < ev[i - 1][1] - PHASE_TOL:
                e_lo, e_td, lift = ev[i - 1]
                u = (p - e_lo) / (e_td - e_lo)
                a_ = float(_ramp(u))
                (xy0, th0), (xy1, th1) = (st[i - 1][2], st[i - 1][3]), (st[i][2], st[i][3])
                foot_pos[k, j, :2] = (1 - a_) * xy0 + a_ * xy1
                foot_pos[k, j, 2] = z0 + lift * laws.lift(u, ctx, sw_of[side][i - 1])
                foot_yaw[k, j] = (1 - a_) * th0 + a_ * th1
                swing[k, j], swing_u[k, j] = True, u
            else:
                foot_pos[k, j, :2], foot_pos[k, j, 2], foot_yaw[k, j] = st[i][2], z0, st[i][3]
            stance_i[k, j] = i
    # --- stand points and the ZMP target

    def centroid(j, k):
        side = SIDES[j]
        if laws.zmp_point is not None:
            s = stances[side][stance_i[k, j]]
            return np.asarray(laws.zmp_point(ctx, (side, s[0], s[1], (s[2], s[3]))), float)
        return foot_pos[k, j, :2] + _rot2(foot_yaw[k, j]) @ geo.pad_c[side]

    P_i, th_i = pose(pf[0])
    P_f, th_f = pose(pf[-1])
    stand_i = P_i + _rot2(th_i) @ (geo.com0[:2] - geo.origin)
    stand_f = P_f + _rot2(th_f) @ (geo.com0[:2] - geo.origin)
    lifts = sorted((e[0], j) for j, side in enumerate(SIDES) for e in swings[side])
    downs = sorted((e[1], j) for j, side in enumerate(SIDES) for e in swings[side])
    zmp = np.zeros((K, 2))
    for k, p in enumerate(phase):
        sup = [j for j in (0, 1) if not swing[k, j]]
        if len(sup) == 1:
            zmp[k] = centroid(sup[0], k)
            continue
        prev = [d for d in downs if d[0] <= p + PHASE_TOL]
        nxt = [u for u in lifts if u[0] > p + PHASE_TOL]
        if not prev:  # the initial stand and the weight shift (start)
            j_first = nxt[0][1]
            j_stance = 1 - j_first
            a_ = float(np.clip((p + SHIFT_STRIDES) / SHIFT_STRIDES, 0.0, 1.0))
            zmp[k] = (1 - a_) * stand_i + a_ * centroid(j_stance, k)
        elif not nxt:  # after the last touchdown (stop): to the final stand
            p_td, j_land = prev[-1]
            a_ = float(np.clip((p - p_td) / SETTLE_STRIDES, 0.0, 1.0))
            zmp[k] = (1 - a_) * centroid(1 - j_land, k) + a_ * stand_f
        else:
            p_td, j_land = prev[-1]
            p_lo, j_lift = nxt[0]
            a_ = float(np.clip((p - p_td) / (p_lo - p_td), 0.0, 1.0))
            zmp[k] = (1 - a_) * centroid(j_lift, k) + a_ * centroid(j_land, k)
    root_z = laws.height(ctx)
    neck = laws.neck(ctx)
    tail = laws.tail(ctx)
    kind = sch.kind if sch.kind in transitions.WINDOWS else "walk"
    pl = transitions.Plan(name=move.name, kind=kind, a=a, b=b, first=first, last=last, phase=phase, t=t, out=out,
                          foot_pos=foot_pos, foot_yaw=foot_yaw, swing=swing, swing_u=swing_u, zmp=zmp, heading=heading,
                          path=path, root_z=root_z, neck=neck, tail=tail, stand_point=dict(initial=stand_i, final=stand_f),
                          swings=swings_sorted, pose_fn=pose, period_fn=period, ramps=ramps)
    pl.ctx = ctx
    pl.extras = dict(trunk_roll=None if laws.trunk_roll is None else laws.trunk_roll(ctx),
                     foot=None if laws.foot is None else laws.foot(ctx))
    return pl


# ------------------------------------------------------------------------------------------------
# The causal builder's pieces
# ------------------------------------------------------------------------------------------------
def walk_correction(ref: refs.Reference, omega: float) -> np.ndarray:
    """What ``gait.generate`` added to the ZMP target of a straight periodic walk, per frame of the stride (n, 2),
    recovered from its saved LIPM CoM path: ``com_ref - com_ref'' / omega^2 - zmp_ref``, the second derivative taken
    spectrally on the periodic part (``com_ref`` less k / n of the stride advance)."""
    c = np.asarray(ref.com_ref, float)
    n = len(c)
    adv = np.array([float(ref.stride_motion[3]), 0.0])
    per = c - np.outer(np.arange(n) / n, adv)  # the periodic part
    k = np.fft.fftfreq(n, d=1.0 / n)
    w = 2 * np.pi * k / (n * ref.step)
    acc = np.real(np.fft.ifft(-(w[:, None] ** 2) * np.fft.fft(per, axis=0), axis=0))
    return c - acc / omega**2 - np.asarray(ref.zmp_ref, float)


def _walk_shift(ref: refs.Reference, first: float) -> np.ndarray:
    """The translation that puts a straight walk's stride 0 into a plan whose path origin is at phase ``first``."""
    return np.array([-first * float(ref.stride_motion[3]), 0.0])


def _straight(ref: refs.Reference) -> None:
    sm = np.asarray(ref.stride_motion, float)
    if sm[0] != 0.0 or sm[4] != 0.0 or np.any(sm[1:3] != 0.0):
        raise ValueError("a causal move starts from a straight periodic walk (its stride motion a pure advance in x)")


def walk_state(ref: refs.Reference, phase0: float, first: float) -> np.ndarray:
    """(c, c', c'') per axis (3, 2) of the straight periodic walk's LIPM path (``com_ref``) at gait phase ``phase0``
    in the frame of a plan whose path origin is at phase ``first``: ``transitions.initial_state``'s fourth-order
    differences with the frame offset generalised to any multiple of 1 / n."""
    _straight(ref)
    mot = transitions._stride_motion(ref)
    inv = transitions._inverse(mot)
    c, n, dt = ref.com_ref, ref.n, ref.step
    k0 = int(round(phase0 * n))
    if abs(phase0 * n - k0) > 1e-6:
        raise ValueError("the computed window must start on a frame of the walk")
    shift = _walk_shift(ref, first)

    def at(k):
        s, kk = divmod(k0 + k, n)
        x = c[kk]
        for _ in range(abs(s)):
            x = transitions._move(x, inv if s < 0 else mot)
        return x + shift

    p = np.array([at(k) for k in (-2, -1, 0, 1, 2)])
    x0 = np.zeros((3, 2))
    x0[0] = p[2]
    x0[1] = (-p[4] + 8 * p[3] - 8 * p[1] + p[0]) / (12 * dt)
    x0[2] = (-p[4] + 16 * p[3] - 30 * p[2] + 16 * p[1] - p[0]) / (12 * dt * dt)
    return x0


def walk_frames(ref: refs.Reference, phases, first: float, key: str) -> np.ndarray:
    """The straight periodic walk's array ``key`` ((n, 2) or (n, 3); x and y are carried, z is kept) at gait phases
    ``phases`` (multiples of 1 / n) in the frame of a plan whose path origin is at phase ``first``."""
    _straight(ref)
    mot = transitions._stride_motion(ref)
    inv = transitions._inverse(mot)
    arr, n = np.asarray(getattr(ref, key), float), ref.n
    shift = _walk_shift(ref, first)
    out = []
    for p in phases:
        s, kk = divmod(int(round(p * n)), n)
        x = arr[kk][:2]
        for _ in range(abs(s)):
            x = transitions._move(x, inv if s < 0 else mot)
        out.append(np.concatenate([x + shift, arr[kk][2:]]))
    return np.array(out)


def preview_run_switch(gains: dict, z_before: np.ndarray, z_after: np.ndarray, x0: np.ndarray, k_switch: int):
    """``transitions.preview_run`` with new information: grid steps k < ``k_switch`` preview ``z_before``, the steps
    from ``k_switch`` on preview ``z_after`` (two sequences on the same grid).  The first step is ``preview_run``'s
    position form; every later step is the same control law in its incremental form,
    u_k = u_{k-1} - Gi e_k - Gx (x_k - x_{k-1}) - Gp . (z[k+1 : k+1+N] - z[k : k+N]), z the sequence in force at step
    k, so that a change of previewed sequence enters only through its own increments (the position form across the
    change gives a jerk impulse).  On one sequence it is the same controller.  Returns states (M, 3), jerks (M,)."""
    A, B, C, Gi, Gx, Gp, N = (gains[k] for k in ("A", "B", "C", "Gi", "Gx", "Gp", "N"))
    Mn = len(z_after) - N
    xs, us = np.zeros((Mn, 3)), np.zeros(Mn)
    x = np.asarray(x0, float).copy()
    u = x_prev = None
    for k in range(Mn):
        zs = z_before if k < k_switch else z_after
        xs[k] = x
        e = C @ x - zs[k]
        if u is None:  # the first step: the position form (a cold start, as transitions.preview_run)
            u = -Gi * e - Gx @ x - Gp @ zs[k + 1 : k + 1 + N]
        else:
            u = u - Gi * e - Gx @ (x - x_prev) - Gp @ (zs[k + 1 : k + 1 + N] - zs[k : k + N])
        us[k] = u
        x_prev = x
        x = A @ x + B * u
    return xs, us


def com_path_causal(pl: transitions.Plan, zcmd: np.ndarray, zpre_t: np.ndarray, zpre: np.ndarray, x0: np.ndarray,
                    omega: float, k_cmd: int, dt: float | None = None):
    """As ``transitions.com_path`` (same gains, same uniform grid at the smallest frame interval, same cubic
    read-out), but grid steps before the command frame ``k_cmd`` preview ``zpre`` (the continuing walk's corrected
    ZMP reference, on its own longer time axis ``zpre_t``) and the steps from it on preview ``zcmd`` (the commanded
    plan's, at the plan's frames).  Returns the CoM xy at the plan's frames (K, 2) and the cart-table ZMP of the same
    cubic, c - c'' / omega^2 (K, 2).  ``dt``: the grid interval (default: the plan's smallest frame interval)."""
    t, K, n = pl.t, len(pl.t), N_FRAMES
    dt_min = float(np.min(np.diff(t))) if dt is None else float(dt)
    g = transitions.preview_gains(dt_min, omega)
    Np = g["N"]
    # the preview past the last frame: as transitions.com_path
    reps = int(math.ceil(transitions.PREVIEW_S / (n * dt_min))) + 2
    if pl.b.stand:
        ext_t = t[-1] + dt_min * np.arange(1, reps * n + 1)
        ext_z = np.tile(zcmd[-1], (len(ext_t), 1))
    else:
        mot = transitions._stride_motion_at(pl, pl.phase[-n])
        T_b = pl.b.period
        z_last, t_last, ext_z, ext_t = zcmd[-n:], t[-n:], [], []
        for r_ in range(1, reps + 1):
            z_last = transitions._move(z_last, mot)
            ext_z.append(z_last)
            ext_t.append(t_last + r_ * T_b)
        ext_z, ext_t = np.vstack(ext_z), np.concatenate(ext_t)
    tt = np.concatenate([t, ext_t])
    zz = np.vstack([zcmd, ext_z])
    grid = t[0] + dt_min * np.arange(int(math.floor((t[-1] - t[0]) / dt_min)) + 2 + Np)
    k_switch = int(np.sum(grid[: len(grid) - Np] < t[k_cmd] - 1e-12))  # grid steps before the command frame
    if zpre_t[-1] < min(grid[-1], t[k_cmd] + transitions.PREVIEW_S + 2 * dt_min):
        raise ValueError("the continuing walk's reference ends inside the preview of the last step before the command")
    out, zct = np.zeros((K, 2)), np.zeros((K, 2))
    for ax in (0, 1):
        zc = np.interp(grid, tt, zz[:, ax])
        zp = np.interp(grid, zpre_t, zpre[:, ax])
        xs, us = preview_run_switch(g, zp, zc, x0[:, ax], k_switch)
        idx = np.clip(((t - t[0]) / dt_min + 1e-9).astype(int), 0, len(xs) - 1)
        s = t - grid[idx]
        out[:, ax] = xs[idx, 0] + xs[idx, 1] * s + xs[idx, 2] * s * s / 2 + us[idx] * s**3 / 6
        zct[:, ax] = out[:, ax] - (xs[idx, 2] + us[idx] * s) / omega**2  # the cart-table ZMP of the same cubic
    return out, zct


def causal_fade(phase: np.ndarray, command: float) -> np.ndarray:
    """The weight of the excused transient per frame (``RULES["S2A-fade"]``): 1 up to ``CAUSAL_FADE[0]`` of a stride
    after the command, 0 from ``CAUSAL_FADE[1]`` after it."""
    f0, f1 = CAUSAL_FADE
    return 1.0 - _ramp((np.asarray(phase, float) - (command + f0)) / (f1 - f0))


def causal_errors(pl: transitions.Plan, command: float, zmb: np.ndarray, zct: np.ndarray, zref: np.ndarray) -> dict:
    """One correction pass's errors of a causal move, at the plan's computed frames:

    * ``tau_c``: the excused transient, fade(p) x (cart-table ZMP of the controller's output - its reference) from
      the command frame on, 0 before it;
    * ``e``: what the loop corrects, multibody ZMP - plan target - tau_c from the command frame on, 0 before it;
    * ``consistency``: max |e| over the output frames; ``target``: max |multibody ZMP - plan target| over the output
      frames (Stage 1's reading); ``transient``: max |tau_c|; ``post_window``: max |multibody ZMP - plan target| over
      the output frames one stride or more after the command."""
    k_cmd, o = pl.out.start, pl.out
    tau_c = causal_fade(pl.phase, command)[:, None] * (zct - zref)
    tau_c[:k_cmd] = 0.0
    e_t = np.nan_to_num(zmb - pl.zmp)
    e = np.nan_to_num(zmb - pl.zmp - tau_c)
    e[:k_cmd] = 0.0
    post = pl.phase[o] >= command + 1.0 - PHASE_TOL
    en_t = np.linalg.norm(e_t[o], axis=1)
    return dict(tau_c=tau_c, e=e, consistency=float(np.max(np.linalg.norm(e[o], axis=1))), target=float(np.max(en_t)),
                transient=float(np.max(np.linalg.norm(tau_c[o], axis=1))),
                post_window=float(np.max(en_t[post])) if post.any() else None)


def walk_schedule(g: transitions.Gait, window, command: float | None = None, rule_f: bool = False, pre: float = 1.0,
                  post: float = 0.25) -> Schedule:
    """The schedule of a periodic gait walking on: every parameter its own constant."""
    fn = dict(T=_const(g.period), duty=_const(g.duty), lift=_const(g.lift), v=_const(g.speed), w=_const(g.yaw_rate),
              amp=_const(1.0), head_amp=_const(g.head_amp), tail_amp=_const(g.tail_amp), crouch=_const(g.crouch))
    return Schedule(kind="walk", window=tuple(window), fn=fn, ramps={}, command=command, rule_f=rule_f, pre=pre,
                    post=post)


def continuing_walk(move: Move) -> Move:
    """The walk that would have continued had no command come: gait a's own constants (``walk_schedule``) on a window
    from the command to past the preview of the last grid step before it."""
    g, sch = move.a.legacy(), move.schedule
    c = sch.command
    last = int(math.ceil(c)) + int(math.ceil(transitions.PREVIEW_S / g.period)) + 2
    return Move(f"{move.name}/continuing_walk", move.a, move.a,
                walk_schedule(g, (c, last), pre=sch.pre, post=sch.post), "walk", {}, laws=move.laws)


def _rotate(theta: np.ndarray, v: np.ndarray) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.column_stack([c * v[:, 0] - s * v[:, 1], s * v[:, 0] + c * v[:, 1]])


def _causal_setup(src: dict, move: Move, ref_a: refs.Reference | None) -> dict:
    """Plan, continuing walk, start state and first ZMP reference of a causal move."""
    sch = move.schedule
    if not move.causal or sch.command is None:
        raise ValueError(f"{move.name}: a causal move has a command phase")
    if move.a.spec is None or ref_a is None:
        raise ValueError(f"{move.name}: a causal move starts from a periodic walk and needs its reference")
    if move.a.spec.sway_gain != 1:
        raise ValueError(f"{move.name}: the walk's ZMP correction is recovered from its LIPM path, which needs "
                         f"sway_gain 1 (got {move.a.spec.sway_gain:g})")
    M, q_home = src["M"], src["q_home"]
    a, b = move.a.legacy(), move.b.legacy()
    if a.yaw_rate != 0.0 or move.a.in_place:
        raise ValueError(f"{move.name}: a causal move starts from a straight walk")
    if ref_a.n != N_FRAMES:
        raise ValueError(f"{move.name}: the walk's reference has {ref_a.n} frames, not {N_FRAMES}")
    c = float(sch.command)
    geo = transitions.Geometry.of(M, q_home, a.crouch)
    pl = plan(move, geo)
    wl = plan(continuing_walk(move), geo)
    k_cmd = pl.out.start
    if abs(pl.phase[k_cmd] - c) > PHASE_TOL or abs(pl.t[k_cmd]) > 1e-12:
        raise MoveError(f"{move.name}: the output window must begin at the command phase")
    n = N_FRAMES
    delta = walk_correction(ref_a, geo.omega)
    kk = np.round(pl.phase * n).astype(int) % n
    zref0 = pl.zmp + _rotate(pl.heading, delta[kk]) * np.asarray(sch.fn["amp"](pl.phase), float)[:, None]
    kw = np.round(wl.phase * n).astype(int) % n
    zpre = wl.zmp + delta[kw]
    x0 = walk_state(ref_a, float(pl.phase[0]), float(pl.first))
    return dict(pl=pl, geo=geo, a=a, b=b, c=c, k_cmd=k_cmd, zref0=zref0, zpre_t=wl.t, zpre=zpre, x0=x0)


def range_use(src: dict, q: np.ndarray) -> dict:
    """joint -> largest fraction of the usable range reached over the frames ``q`` (``turning.summarize``'s formula on
    ``plant.usable_ranges``: 1.0 = at the limit)."""
    m = src["M"].m
    out = {}
    for jn, (lo, hi) in plant.usable_ranges(src["body"]).items():
        qv = q[:, m.jnt_qposadr[statics.jid(m, jn)]]
        out[jn] = float(max(np.max(qv) / hi, np.min(qv) / lo))
    return out


def hip_yaw_use(use: dict) -> float:
    """The worse leg's thigh-twist use of a ``range_use`` dict."""
    return float(max(v for k, v in use.items() if k.endswith("_hip_yaw")))


def _ik(foot_err: np.ndarray) -> tuple[bool, int]:
    return bool(np.all(foot_err <= turning.IK_TOL)), int(np.sum(np.any(foot_err > turning.IK_TOL, axis=1)))


# ------------------------------------------------------------------------------------------------
# Generation
# ------------------------------------------------------------------------------------------------
def generate(src: dict, move: Move, ref_a: refs.Reference | None, verbose: bool = False) -> dict:
    """One move of ``src``'s body (``refs.source``): ``transitions.generate``'s dict (same keys) plus ``move``,
    ``command_phase``, ``causal``, ``zmp_consistency_history_m``, ``causal_transient_m``, ``post_window_zmp_error_m``,
    ``kappa`` (turns), ``ik_fail_frames``, ``range_use`` and, for a causal move, ``warmup`` (the computed frames up
    to the command frame, for ``join_command``) and ``tau_c``.  ``ref_a`` is gait a's periodic reference (required
    unless it is the stand)."""
    S = _solver(move.laws)
    if move.causal:
        return _generate_causal(S, src, move, ref_a, verbose)
    return _generate_stage1(S, src, move, ref_a, verbose)


def _generate_stage1(S, src: dict, move: Move, ref_a, verbose: bool) -> dict:
    """``transitions.generate``'s body (``transitions.py:638-678``) on ``plan(move, geo)``."""
    gait.set_accuracy(refs.ACCURACY)
    t0 = time.perf_counter()
    M, q_home = src["M"], src["q_home"]
    name = move.name
    a, b = move.a.legacy(), move.b.legacy()
    walk = b if a.stand else a
    geo = transitions.Geometry.of(M, q_home, walk.crouch)
    pl = plan(move, geo)
    x0 = S.initial_state(pl, ref_a, geo)
    zref = pl.zmp.copy()
    iters = walk.zmp_iters if b.stand else b.zmp_iters
    gain = walk.zmp_gain if b.stand else b.zmp_gain
    history = []
    for it in range(iters + 1):
        com_t = transitions.com_path(pl, zref, x0, geo.omega)
        qs, foot_err, com_err = S.solve_frames(M, q_home, pl, com_t, geo)
        qd, qdd = transitions.derivatives(M.m, qs, pl.t)
        zmb = transitions.multibody_zmp(M, qs, qd, qdd)
        e = np.nan_to_num(zmb - pl.zmp)
        history.append(float(np.max(np.linalg.norm(e[pl.out], axis=1))))
        if verbose:
            print(f"  {name} pass {it}: ZMP error max {1e3 * history[-1]:.3f} mm ({time.perf_counter() - t0:.0f} s)",
                  flush=True)
        if it == iters:
            break
        zref = zref - gain * e
    o = pl.out
    split = "exact" if "exact" in (a.split, b.split) else "statics"
    tau, cop_margin, normal, friction = S.idyn(M, qs[o], qd[o], qdd[o], pl.swing[o], split)
    feet = np.zeros((pl.n_out, 2, 3))
    com = np.zeros((pl.n_out, 3))
    for i, k in enumerate(range(o.start, o.stop)):
        M.forward(qs[k])
        feet[i] = [M.foot_pose(s)[0] for s in SIDES]
        com[i] = M.com()
    ik_ok, ik_fail = _ik(foot_err[o])
    return dict(name=name, plan=pl, geo=geo, a=a, b=b, q=qs[o], qd=qd[o], qdd=qdd[o], tau=tau, com_ref=com_t[o],
                zmp_ref=pl.zmp[o], zmp_mb=zmb[o], zref=zref[o], cop_margin=cop_margin, normal=normal, friction=friction,
                feet=feet, com=com, foot_error=foot_err[o], com_error=com_err[o], zmp_error_history_m=history,
                zmp_iters=iters, zmp_gain=gain, split=split, x0=x0, seconds=time.perf_counter() - t0,
                ik_ok=ik_ok, q_home=np.asarray(q_home, float).copy(), joints=list(M.names),
                move=move, command_phase=None, causal=False, zmp_consistency_history_m=None, causal_transient_m=None,
                post_window_zmp_error_m=None, kappa=None, ik_fail_frames=ik_fail, range_use=range_use(src, qs[o]))


def _generate_causal(S, src: dict, move: Move, ref_a, verbose: bool) -> dict:
    """The causal solve: the preview controller runs on the continuing walk up to the command frame and previews the
    commanded plan from it on; the correction loop excuses the causal transient for one stride (module docstring)."""
    gait.set_accuracy(refs.ACCURACY)
    t0 = time.perf_counter()
    M, q_home, name = src["M"], src["q_home"], move.name
    cs = _causal_setup(src, move, ref_a)
    pl, geo, a, b, c, k_cmd, x0 = (cs[k] for k in ("pl", "geo", "a", "b", "c", "k_cmd", "x0"))
    zref = cs["zref0"].copy()
    iters, gain = a.zmp_iters, a.zmp_gain
    target, consistency, transient, post = [], [], [], []
    for it in range(iters + 1):
        com_t, zct = com_path_causal(pl, zref, cs["zpre_t"], cs["zpre"], x0, geo.omega, k_cmd)
        qs, foot_err, com_err = S.solve_frames(M, q_home, pl, com_t, geo)
        qd, qdd = transitions.derivatives(M.m, qs, pl.t)
        zmb = transitions.multibody_zmp(M, qs, qd, qdd)
        err = causal_errors(pl, c, zmb, zct, zref)
        target.append(err["target"])
        consistency.append(err["consistency"])
        transient.append(err["transient"])
        post.append(err["post_window"])
        if verbose:
            print(f"  {name} pass {it}: consistency {1e3 * consistency[-1]:.3f} mm, against the target "
                  f"{1e3 * target[-1]:.3f} mm, causal transient {1e3 * transient[-1]:.3f} mm "
                  f"({time.perf_counter() - t0:.0f} s)", flush=True)
        if it == iters:
            break
        zref = zref - gain * err["e"]
    o = pl.out
    split = "exact" if move.schedule.kind == "turn" or "exact" in (a.split, b.split) else "statics"
    tau, cop_margin, normal, friction = S.idyn(M, qs[o], qd[o], qdd[o], pl.swing[o], split)
    feet = np.zeros((pl.n_out, 2, 3))
    com_all = np.zeros((o.stop, 3))
    for k in range(o.stop):
        M.forward(qs[k])
        com_all[k] = M.com()
        if k >= o.start:
            feet[k - o.start] = [M.foot_pose(s)[0] for s in SIDES]
    ik_ok, ik_fail = _ik(foot_err[o])
    w = slice(0, k_cmd + 1)  # the warm-up stride and the command frame
    kappa = turn_design(a, **move.args)["kappa"] if move.builder == "turn" else None
    return dict(name=name, plan=pl, geo=geo, a=a, b=b, q=qs[o], qd=qd[o], qdd=qdd[o], tau=tau, com_ref=com_t[o],
                zmp_ref=pl.zmp[o], zmp_mb=zmb[o], zref=zref[o], cop_margin=cop_margin, normal=normal, friction=friction,
                feet=feet, com=com_all[o], foot_error=foot_err[o], com_error=com_err[o], zmp_error_history_m=target,
                zmp_iters=iters, zmp_gain=gain, split=split, x0=x0, seconds=time.perf_counter() - t0,
                ik_ok=ik_ok, q_home=np.asarray(q_home, float).copy(), joints=list(M.names),
                move=move, command_phase=c, causal=True, zmp_consistency_history_m=consistency,
                causal_transient_m=transient, post_window_zmp_error_m=post, kappa=kappa, ik_fail_frames=ik_fail,
                range_use=range_use(src, qs[o]), tau_c=err["tau_c"][o],
                warmup=dict(phase=pl.phase[w], q=qs[w], com=com_all[w], zmp_mb=zmb[w],
                            ik_ok=bool(np.all(foot_err[w] <= turning.IK_TOL))))


def kinematic(src: dict, move: Move, ref_a) -> dict:
    """One solve of the move's frames on its first CoM path, no ZMP correction and no dynamics: the output frames'
    ``q``, ``ik_ok``, ``ik_fail_frames``, ``range_use`` (and ``hip_yaw_range_use``, the worse leg's)."""
    S = _solver(move.laws)
    gait.set_accuracy(refs.ACCURACY)
    t0 = time.perf_counter()
    M, q_home = src["M"], src["q_home"]
    if move.causal:
        cs = _causal_setup(src, move, ref_a)
        pl, geo = cs["pl"], cs["geo"]
        com_t, _ = com_path_causal(pl, cs["zref0"], cs["zpre_t"], cs["zpre"], cs["x0"], geo.omega, cs["k_cmd"])
    else:
        a, b = move.a.legacy(), move.b.legacy()
        geo = transitions.Geometry.of(M, q_home, (b if a.stand else a).crouch)
        pl = plan(move, geo)
        com_t = transitions.com_path(pl, pl.zmp.copy(), S.initial_state(pl, ref_a, geo), geo.omega)
    qs, foot_err, _ = S.solve_frames(M, q_home, pl, com_t, geo)
    o = pl.out
    ik_ok, ik_fail = _ik(foot_err[o])
    use = range_use(src, qs[o])
    return dict(name=move.name, plan=pl, geo=geo, q=qs[o], foot_error=foot_err[o], ik_ok=ik_ok, ik_fail_frames=ik_fail,
                range_use=use, hip_yaw_range_use=hip_yaw_use(use), seconds=time.perf_counter() - t0)


def arrays(g: dict) -> dict:
    """The arrays of one generated move for its npz file: ``transitions.arrays(g)`` exactly for today's laws; the
    look's solver appends its keys after them."""
    out = transitions.arrays(g)
    if g["move"].laws.solver == "look":
        out.update(_solver(g["move"].laws).arrays_extra(g))
    return out


# ------------------------------------------------------------------------------------------------
# Readings of the reference itself
# ------------------------------------------------------------------------------------------------
def forward_acceleration(t, com_xy, heading, fs: float = ACCEL_FS):
    """The model CoM's acceleration along the per-frame heading on a uniform grid at ``fs``: times, the 1 Hz
    low-passed series (decision 10's filter, ``metrics.lowpass``) and the unfiltered one."""
    from ..stage1 import metrics

    t, c = np.asarray(t, float), np.asarray(com_xy, float)
    tu = np.arange(t[0], t[-1], 1.0 / fs)
    x, y, h = np.interp(tu, t, c[:, 0]), np.interp(tu, t, c[:, 1]), np.interp(tu, t, np.asarray(heading, float))
    ax, ay = np.gradient(np.gradient(x, tu), tu), np.gradient(np.gradient(y, tu), tu)
    af = ax * np.cos(h) + ay * np.sin(h)
    return tu, metrics.lowpass(af, fs), af


def _accel_reading(g: dict) -> dict:
    pl = g["plan"]
    o = pl.out
    from ..stage1 import metrics

    _, lp, raw = forward_acceleration(pl.t[o] - pl.t[o][0], g["com"][:, :2], pl.heading[o])
    return dict(accel_forward_lp1hz_peak_m_s2=float(np.max(np.abs(lp))),
                accel_forward_raw_peak_m_s2=float(np.max(np.abs(raw[5:-5]))),
                accel_filter=dict(hz=metrics.ACCEL_LP_HZ, order=metrics.ACCEL_LP_ORDER, zero_phase=True, fs_hz=ACCEL_FS,
                                  of="model CoM, along the planned heading, second differences"))


def turn_reference(t, heading, angle: float, rate: float) -> dict:
    """A turn reference's own heading readings (design D9).  ``t``: reference seconds from the command, ``heading``:
    the planned heading from the command frame's.  The bar reads a turn at bar time at_s = angle / |rate| simulated
    seconds after the command; the track is played at ``tune.COMMAND_RATE`` times the tracker's phase-rate factor
    (within ``control.PHASE_RATE_LIMIT``), so the reference itself is read at reference time RATE x at_s (the nominal
    commanded clock) and at (1 - limit) x RATE x at_s (the slowest clock)."""
    from ..stage1 import control, tune

    t, h = np.asarray(t, float), np.degrees(np.abs(np.asarray(heading, float) - float(heading[0])))
    rate_cmd, slow = float(tune.COMMAND_RATE), 1.0 - float(control.PHASE_RATE_LIMIT)
    at_s = float(angle) / abs(float(rate))
    k = np.flatnonzero(h >= TURN_HEADING_MIN_DEG)
    t80 = None
    if len(k):
        k = int(k[0])
        t80 = float(t[k]) if k == 0 else float(np.interp(TURN_HEADING_MIN_DEG, h[k - 1 : k + 1], t[k - 1 : k + 1]))
    return dict(at_s=at_s, command_rate=rate_cmd, slowest_clock=slow,
                heading_at_bar_deg=float(np.interp(rate_cmd * at_s, t, h)),
                heading_at_slowest_clock_deg=float(np.interp(slow * rate_cmd * at_s, t, h)),
                heading_at_ref_clock_deg=float(np.interp(at_s, t, h)),
                time_to_80deg_bar_s=None if t80 is None else t80 / rate_cmd, heading_min_deg=TURN_HEADING_MIN_DEG)


def stop_reference(g: dict) -> dict:
    """A causal stop reference's own readings: touchdowns after the command, the stopped time from the command
    (decision 18's quantity: forward model-CoM speed below 5 % of the walk's from then on), the margin between the
    last touchdown before the command and the command, and Stage 1's stop reading with its own command phase."""
    pl = g["plan"]
    o, c = pl.out, float(g["command_phase"])
    ph, t = pl.phase[o], pl.t[o] - pl.t[o][0]
    downs = sorted(s[2] for s in pl.swings)
    after = [p for p in downs if p > c + PHASE_TOL]
    before = [p for p in downs if p <= c + PHASE_TOL]
    th = pl.pose_fn(float(pl.last))[1]
    v = np.gradient(g["com"][:, :2], t, axis=0) @ np.array([math.cos(th), math.sin(th)])
    rd = transitions.stop_reading(t, v, pl.a.speed, 0.0, pl.a.period)
    fwd = rd["forward_stopped_strides"]
    s1 = transitions.stop_time_arrays(ph, t, g["com_ref"], th, pl.stand_point["final"], pl.a.speed, pl.a.period,
                                      pl.a.duty)
    s1["command_phase_is"] = ("Stage 1's convention (decision 18): the start of the reference's own speed ramp, "
                              f"{s1['command_phase'] - c:g} stride after this move's command")
    return dict(touchdowns_after_command=len(after), touchdown_phase=[float(p) for p in after],
                touchdown_s=[float(np.interp(p, ph, t)) for p in after],
                forward_stopped_s=None if fwd is None else float(fwd * pl.a.period), forward_stopped_strides=fwd,
                stopped_rule=f"the forward model-CoM speed (along the final heading) below "
                             f"{100 * transitions.STOP_SPEED_FRAC:g} % of the walk's speed from then on, counted in "
                             "seconds and in the walk's strides from this move's command (its first frame); gate: "
                             f"within {transitions.STOP_WITHIN_STRIDES:g} strides",
                margin_before_command_s=float((c - before[-1]) * pl.a.period) if before else None, stage1_stop=s1)


def join_command(g: dict, ref_a: refs.Reference) -> dict:
    """A causal move's warm-up stride and command frame against the walk's own frames (design 3.3)."""
    w, pl = g["warmup"], g["plan"]
    n, first = N_FRAMES, float(pl.first)
    com_w = walk_frames(ref_a, w["phase"], first, "com")
    zmp_w = walk_frames(ref_a, w["phase"], first, "zmp_mb")
    kq = np.round(w["phase"] * n).astype(int) % n
    dc = 1e3 * np.linalg.norm(w["com"] - com_w, axis=1)
    dz = 1e3 * np.linalg.norm(w["zmp_mb"] - zmp_w, axis=1)  # NaN on the first two computed frames (no acceleration)
    dq = np.abs(w["q"][:, 7:] - ref_a.q[kq, 7:]).max(1)
    out = dict(reference=ref_a.name, frames=len(dc), warmup_com_max_mm=float(np.max(dc)),
               warmup_zmp_max_mm=float(np.nanmax(dz[:-2])), warmup_joint_max_rad=float(np.max(dq)),
               warmup_ik_ok=bool(w["ik_ok"]),
               at_command=dict(com_mm=float(dc[-1]), joint_rad=float(dq[-1]), zmp_mm=float(dz[-1]),
                               zmp_note="recorded, not gated: the multibody ZMP's central difference at the command "
                                        "frame straddles the command"),
               limits=dict(com_mm=transitions.G3_COM_MM, zmp_mm=transitions.G3_ZMP_MM, joint_rad=JOINT_AT_COMMAND_RAD,
                           zmp_frames="the warm-up frames up to two frames before the command"))
    out["ok"] = bool(out["warmup_com_max_mm"] <= transitions.G3_COM_MM and dc[-1] <= transitions.G3_COM_MM
                     and dq[-1] <= JOINT_AT_COMMAND_RAD and out["warmup_zmp_max_mm"] <= transitions.G3_ZMP_MM)
    return out


def _cop_margin_single(g: dict) -> dict | None:
    pl = g["plan"]
    single = np.flatnonzero(np.sum(pl.swing[pl.out], axis=1) == 1)
    if not len(single):
        return None
    m = g["cop_margin"][single]
    i, j = np.unravel_index(int(np.nanargmin(m)), m.shape)
    return dict(m=float(m[i, j]), frame=int(single[i]), foot=SIDES[j])


def summary(g: dict, ref_a, ref_b, q_home) -> dict:
    """The entry of one generated move: ``transitions.summary``'s keys (no wall time), the move's definition and
    rules, its joins, histories, range use, the reference's own readings and the gates (``RULES["S2A-gates"]``).  For
    a causal move ``join_a`` and ``stop`` are None: its output starts at the command and leaves the walk at once, so
    Stage 1's first-stride comparison and its stop reading have no meaning there (the join is ``join_command``;
    Stage 1's stop reading is ``reference.stage1_stop``, with its own command phase labelled)."""
    move, pl = g["move"], g["plan"]
    S = _solver(move.laws)
    causal = bool(g["causal"])
    sa = None if causal else S.compare(g, ref_a, "a", q_home)
    sb = S.compare(g, ref_b, "b", q_home)
    o = pl.out
    steps = [dict(side=s, lift_off_phase=float(lo), touchdown_phase=float(td), lift_m=float(h),
                  from_xy=[float(v) for v in f0[0]], from_yaw=float(f0[1]), to_xy=[float(v) for v in f1[0]],
                  to_yaw=float(f1[1]), length_m=float(np.linalg.norm(f1[0] - f0[0])))
             for s, lo, td, h, f0, f1 in pl.swings if pl.first - 0.5 <= lo < pl.last]
    margins = g["cop_margin"]
    single = np.sum(pl.swing[o], axis=1) == 1
    out = dict(name=g["name"], kind=pl.kind, from_=pl.a.name, to=pl.b.name, gaits=dict(a=asdict(pl.a), b=asdict(pl.b)),
               output_phase=[pl.first, pl.last], frames=pl.n_out, duration_s=float(pl.t[o][-1] - pl.t[o][0]),
               ramps=pl.ramps, steps=steps, zmp_iters=g["zmp_iters"], zmp_gain=g["zmp_gain"],
               zmp_error_history_m=g["zmp_error_history_m"], split=g["split"], ik_ok=g["ik_ok"],
               com_tracking_max_m=float(np.max(g["com_error"])),
               min_cop_margin_single_m=float(np.nanmin(margins[single])) if single.any() else None,
               min_cop_margin_m=float(np.nanmin(margins)), max_friction_ratio=float(np.max(g["friction"])),
               stop=None if causal else transitions.stop_time(g), join_a=sa, join_b=sb,
               g3=dict(rule=f"last output stride within {transitions.G3_COM_MM:g} mm of the periodic walk's CoM and "
                            f"{transitions.G3_ZMP_MM:g} mm of its multibody ZMP "
                            f"({'the home stance' if pl.b.stand else pl.b.name})",
                       com_max_mm=sb["com_max_mm"], zmp_max_mm=sb["zmp_max_mm"],
                       pass_=bool(sb["com_ok"] and sb["zmp_ok"] and g["ik_ok"])))
    # --- Stage 2's keys
    use = g["range_use"]
    gates = dict(ik_ok=bool(g["ik_ok"]), join_b=bool(sb["com_ok"] and sb["zmp_ok"]))
    out.update(move=dict(kind=move.schedule.kind, builder=move.builder, causal=causal,
                         command_phase=g["command_phase"], command_t_s=0.0 if causal else None,
                         rules=list(move.rules), laws=move.laws.name),
               definition=move.definition(), ik_fail_frames=int(g["ik_fail_frames"]), range_use=use,
               hip_yaw_range_use=hip_yaw_use(use), join_command=None,
               zmp_consistency_history_m=g["zmp_consistency_history_m"], causal_transient_m=g["causal_transient_m"],
               post_window_zmp_error_m=g["post_window_zmp_error_m"], cop_margin_single_min=_cop_margin_single(g))
    reference = _accel_reading(g)
    if causal:
        jc = join_command(g, ref_a)
        out["join_command"] = jc
        cop = out["cop_margin_single_min"]
        gates.update(join_command=bool(jc["ok"]),
                     zmp_consistency=bool(g["zmp_consistency_history_m"][-1] <= CONSISTENCY_LIMIT_M),
                     post_window_zmp=bool(g["post_window_zmp_error_m"][-1] is not None
                                          and g["post_window_zmp_error_m"][-1] <= ZMP_HISTORY_LIMIT_M),
                     cop_margin_single=bool(cop is not None and cop["m"] >= CAUSAL_COP_MARGIN_M))
    else:
        gates.update(join_a=bool(sa["com_ok"] and sa["zmp_ok"]),
                     zmp_error=bool(g["zmp_error_history_m"][-1] <= ZMP_HISTORY_LIMIT_M))
    if move.builder == "turn":
        d = turn_design(pl.a, **move.args)
        total = float(pl.heading[o][-1] - pl.heading[o][0])
        tr = turn_reference(pl.t[o] - pl.t[o][0], pl.heading[o], d["angle_rad"], d["rate_rad_s"])
        out["turn"] = dict(angle_rad=d["angle_rad"], rate_rad_s=d["rate_rad_s"], direction=d["direction"],
                           first_swing=d["first_swing"], ramp_s=d["ramp_s"], ramp_strides=d["ramp_strides"],
                           hold_strides=d["hold_strides"], stride_factor=d["stride_factor"],
                           period_in_turn_s=d["period_in_turn_s"], kappa=d["kappa"], heading_total_rad=total,
                           appendage_rule=RULES["S2A-appendage"], stride_rule=None)
        reference.update(tr)
        gates.update(heading_total=bool(abs(abs(total) - d["angle_rad"]) <= TURN_CLOSURE_RAD),
                     heading_at_bar=bool(tr["heading_at_bar_deg"] >= TURN_HEADING_MIN_DEG),
                     heading_at_slowest_clock=bool(tr["heading_at_slowest_clock_deg"] >= TURN_HEADING_MIN_DEG),
                     thigh_twist=bool(out["hip_yaw_range_use"] <= TWIST_FINAL))
    if move.builder == "stop_two_step":
        sr = stop_reference(g)
        reference.update(sr)
        fwd = sr["forward_stopped_strides"]
        gates.update(touchdowns_after_command=bool(sr["touchdowns_after_command"] == STOP_TOUCHDOWNS),
                     stopped_within=bool(fwd is not None and fwd <= transitions.STOP_WITHIN_STRIDES),
                     accel_lp1hz=bool(reference["accel_forward_lp1hz_peak_m_s2"] <= STOP_ACCEL_LP_LIMIT_M_S2))
    out.update(reference=reference, gates=gates, pass_=bool(all(gates.values())))
    return refs._plain(out)


# ------------------------------------------------------------------------------------------------
# The stride rule
# ------------------------------------------------------------------------------------------------
def steady_turn_kinematic(src: dict, a: GaitX, rate: float, stride_factor: float) -> dict:
    """The steady turn's kinematic reading beside the finite turn's: ``turning.generate`` of walk ``a`` with its
    stride (and so its period) times ``stride_factor`` at the same speed, no ZMP correction.  ``turning.generate``
    builds today's walk only; for a look spec the reading is not available and says so."""
    spec = replace(a.spec, stride_rel=a.spec.stride_rel * stride_factor, zmp_iters=0)
    if gait.look_active(spec):
        return dict(available=False, reason="turning.generate builds today's walk only (a look spec is refused)")
    with _accuracy():
        tr = turning.generate(src["M"], src["q_home"], src["H"], spec, yaw_rate=abs(float(rate)))
    use = range_use(src, tr["q"])
    return dict(available=True, generator="turning.generate, zmp_iters 0, left turn", period_s=float(tr["info"]["period"]),
                ik_ok=bool(np.all(tr["ik_ok"])), hip_yaw_range_use=hip_yaw_use(use))


class _accuracy:
    """``gait.set_accuracy("r2")`` for a block, restoring the process's level after."""

    def __enter__(self):
        self.prev = gait.ACCURACY
        gait.set_accuracy(refs.ACCURACY)

    def __exit__(self, *exc):
        gait.set_accuracy(self.prev)
        return False


def pick_stride_factor(src: dict, a: GaitX, ref_a, angle: float, rate: float, grid=STRIDE_GRID,
                       limit: float = TWIST_KINEMATIC) -> dict:
    """The stride rule's kinematic pick (``RULES["S2A-stride"]``): for f in ``grid``, largest first, the kinematic
    pass of the left and of the right turn at that f; the first f whose two passes both have ``ik_ok`` and a
    thigh-twist use of at most ``limit`` is chosen.  Returns every f tried with its two readings, the chosen f or
    None, the steady turn's kinematic reading at the chosen f, the rule's text and constants, ``steps_down`` 0."""
    tried, chosen = [], None
    for f in grid:
        row = dict(f=float(f), left=None, right=None, ik_fail_frames={}, ik_ok=True, pass_=True)
        for direction, sign in (("left", 1.0), ("right", -1.0)):
            try:
                mv = turn(f"turn_{direction}", a, angle, sign * abs(float(rate)), f)
            except ValueError as err:
                row.update(ik_ok=False, pass_=False, error=str(err))
                continue
            k = kinematic(src, mv, ref_a)
            use = hip_yaw_use(k["range_use"])
            row[direction] = use
            row["ik_fail_frames"][direction] = int(k["ik_fail_frames"])
            row["ik_ok"] = bool(row["ik_ok"] and k["ik_ok"])
            row["pass_"] = bool(row["pass_"] and k["ik_ok"] and use <= limit)
        tried.append(row)
        if row["pass_"]:
            chosen = float(f)
            break
    steady = None if chosen is None else steady_turn_kinematic(src, a, rate, chosen)
    return dict(rule="S2A-stride", text=RULES["S2A-stride"], grid=[float(f) for f in grid],
                limit_kinematic=float(limit), limit_final=TWIST_FINAL, steps_down_max=STEPS_DOWN_MAX,
                angle_rad=float(angle), rate_rad_s=abs(float(rate)), tried=tried, chosen=chosen,
                steady_turn_kinematic=steady, steps_down=0, builds=[], outcome="picked" if chosen is not None else "none",
                reason=None if chosen is not None else
                f"no stride factor of the grid passes the kinematic thigh-twist rule ({limit:g})")


def step_down(record: dict, built: dict) -> dict:
    """The stride rule after a full build at ``record["chosen"]``.  ``built``: the built reference's ``ik_ok`` and
    ``hip_yaw_range_use`` and the caller's Stage 0 flags ``over_leg_drive`` (a leg drive over a limit) and
    ``a1_over`` (a neck or tail planning limit over, after the appendage rule).  Returns a new record with the build
    appended to ``builds`` and ``outcome``:

    * ``"frozen"``: the reference stands (``a1_over`` says whether it is read under the third reading only);
    * ``"rebuild"``: ``chosen`` is now the next factor down the grid (``steps_down`` + 1): build again;
    * ``"none"``: no reference for this set, body and rate (``chosen`` None, ``reason``): over a leg drive's limit,
      or the twist rule failed with no step-down left."""
    f = record["chosen"]
    if f is None:
        raise ValueError("no stride factor is chosen: nothing was built")
    rec = dict(record, builds=[*record.get("builds", []), dict(f=float(f), **built)])
    grid = [float(x) for x in record["grid"]]
    if built.get("over_leg_drive"):
        return dict(rec, chosen=None, outcome="none", a1_over=None,
                    reason="the built reference is over a leg drive's Stage 0 limit")
    if built["ik_ok"] and built["hip_yaw_range_use"] <= record["limit_final"]:
        return dict(rec, outcome="frozen", a1_over=bool(built.get("a1_over", False)), reason=None)
    lower = [x for x in grid if x < f - 1e-12]
    if record["steps_down"] >= record["steps_down_max"] or not lower:
        return dict(rec, chosen=None, outcome="none", a1_over=None,
                    reason=f"the built reference is not within the thigh-twist rule ({record['limit_final']:g}) and "
                           "no step-down is left")
    return dict(rec, chosen=max(lower), steps_down=record["steps_down"] + 1, outcome="rebuild", a1_over=None,
                reason=None)


# ------------------------------------------------------------------------------------------------
# The legacy check: Stage 1's 18 transitions, bit for bit
# ------------------------------------------------------------------------------------------------
LEGACY_RULE = ("every one of Stage 1's transitions of a body (transitions.TRANSITIONS), regenerated by "
               "designlab.stage2.moves (the plan with Stage 1's schedule passed as data; Stage 1's own initial_state, "
               "com_path, solve_frames, derivatives, multibody_zmp and idyn) at the full correction passes, accuracy "
               "r2, one BLAS thread, from the run's own periodic references, and written through a file object, has "
               "the sha256 its entry in the Stage 1 run's transitions.json records")


def require_one_thread() -> int:
    """Each of the three BLAS thread variables must be exactly "1" (the arrays are bit-reproducible only then).
    Work package R's ``s2io.require_one_thread`` is the shared form of this check."""
    bad = {v: os.environ.get(v) for v in THREAD_VARS if os.environ.get(v) != "1"}
    if bad:
        raise MoveError(f"one BLAS thread is required: set {', '.join(THREAD_VARS)} to 1 (got {bad})")
    return 1


def sha256_file(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _study_path(path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else STUDY / path


def _legacy_job(job: tuple) -> dict:
    """One transition of one body regenerated through this module (a pool worker builds everything itself)."""
    body, name, run = job
    require_one_thread()
    run = Path(run)
    t0 = time.perf_counter()
    a_name, _ = transitions.TRANSITIONS[name]
    recorded = json.loads((run / body / "transitions.json").read_text())["transitions"][name]["sha256"]
    if a_name != "stand":
        ref_sha = json.loads((run / body / "refs.json").read_text())["references"][a_name]["sha256"]
        got = sha256_file(run / body / "refs" / f"{a_name}.npz")
        if got != ref_sha:
            raise MoveError(f"{body} refs/{a_name}.npz has sha256 {got[:12]}..., refs.json records {ref_sha[:12]}...")
    src = refs.source(body)  # a fresh model per transition, as transitions._job
    g = generate(src, stage1(body, name, src["H"]), transitions._load_ref(run / body, a_name))
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"{name}.npz.tmp"
        with open(path, "wb") as fh:
            np.savez(fh, **arrays(g))
        sha = sha256_file(path)
    return dict(body=body, name=name, sha256=sha, recorded=recorded, equal=sha == recorded, passes=int(g["zmp_iters"]),
                accuracy=gait.ACCURACY, seconds=time.perf_counter() - t0)


def _git_state(leave_out: Path | None) -> dict:
    """The commit and whether the study directory differs from it (``leave_out``: the record being written)."""
    out = dict(commit=None, dirty=None, dirty_paths=None)
    try:
        out["commit"] = subprocess.run(["git", "-C", str(STUDY), "rev-parse", "HEAD"], capture_output=True, text=True,
                                       check=True).stdout.strip()
        top = Path(subprocess.run(["git", "-C", str(STUDY), "rev-parse", "--show-toplevel"], capture_output=True,
                                  text=True, check=True).stdout.strip())
        st = subprocess.run(["git", "-C", str(STUDY), "status", "--porcelain", "--untracked-files=all", "--", "."],
                            capture_output=True, text=True, check=True).stdout
        skip = None if leave_out is None else Path(leave_out).resolve()
        paths = [ln for ln in st.splitlines() if ln.strip() and (top / ln[3:].strip()).resolve() != skip]
        out.update(dirty=bool(paths), dirty_paths=paths)
    except (OSError, subprocess.CalledProcessError):
        pass
    return out


def legacy_check(bodies, names=None, procs: int = 1, record=None, run=STAGE1_RUN) -> dict:
    """Regenerate Stage 1's transitions ``names`` (default: all 18) of ``bodies`` through this module and compare each
    npz's sha256 with the Stage 1 run's ``transitions.json``.  With ``record`` the result is written to that JSON
    file, merged per body and per name with what the file already holds from this same ``moves.py`` and the same
    panels' sources (a record made by other code is not merged into); a body is ``complete`` when all 18 names are
    there.  Returns the record."""
    from ..stage1 import panels

    require_one_thread()
    run = _study_path(run)
    names = list(names or transitions.TRANSITIONS)
    jobs = [(body, name, str(run)) for body in bodies for name in names]
    procs = max(1, min(int(procs), len(jobs)))
    results = []
    if procs == 1:
        for job in jobs:
            results.append(_legacy_job(job))
            _legacy_line(results[-1])
    else:
        import multiprocessing as mp

        with mp.get_context("spawn").Pool(procs) as pool:
            for r in pool.imap(_legacy_job, jobs, chunksize=1):
                results.append(r)
                _legacy_line(r)
    record = None if record is None else _study_path(record)
    head = dict(rule=LEGACY_RULE, moves_sha256=sha256_file(Path(__file__)),
                panel_sources_sha256=panels.panel_sources()["sha256"], versions=panels.library_versions(),
                blas_threads=require_one_thread(), accuracy=refs.ACCURACY,
                stage1_run=run.relative_to(STUDY).as_posix() if run.is_relative_to(STUDY) else str(run))
    doc = dict(head, bodies={})
    if record is not None and record.exists():
        old = json.loads(record.read_text())
        if all(old.get(k) == v for k, v in head.items()):
            doc["bodies"] = old.get("bodies", {})
    for r in results:
        b = doc["bodies"].setdefault(r["body"], dict(complete=False, names={}, passes={}))
        b["names"][r["name"]] = dict(sha256=r["sha256"], recorded=r["recorded"], equal=bool(r["equal"]))
        b["passes"][r["name"]] = r["passes"]
    for b in doc["bodies"].values():
        b["names"] = {nm: b["names"][nm] for nm in transitions.TRANSITIONS if nm in b["names"]}
        b["passes"] = {nm: b["passes"][nm] for nm in b["names"]}
        b["complete"] = set(b["names"]) == set(transitions.TRANSITIONS)
        b["equal"] = sum(1 for e in b["names"].values() if e["equal"])
        b["of"] = len(b["names"])
    doc.update(_git_state(record), at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    if record is not None:
        record.parent.mkdir(parents=True, exist_ok=True)
        tmp = record.with_name(record.name + ".tmp")
        with open(tmp, "w") as fh:
            json.dump(refs._plain(doc), fh, indent=1)
        os.replace(tmp, record)
    doc["checked"] = [dict(body=r["body"], name=r["name"], equal=bool(r["equal"])) for r in results]
    return doc


def _legacy_line(r: dict) -> None:
    print(f"  {r['body']} {r['name']}: {r['seconds']:.0f} s, {r['passes']} passes, sha256 {r['sha256'][:16]}... "
          f"{'equal to' if r['equal'] else 'DIFFERS from'} Stage 1's {r['recorded'][:16]}...", flush=True)


def main(argv=None) -> int:
    require_one_thread()
    ap = argparse.ArgumentParser(prog="python -m designlab.stage2.moves", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("legacy", help="regenerate Stage 1's transitions through moves and compare their sha256")
    p.add_argument("--body", action="append", required=True)
    p.add_argument("--name", action="append", choices=list(transitions.TRANSITIONS))
    p.add_argument("--procs", type=int, default=1)
    p.add_argument("--stage1-run", default=str(STAGE1_RUN))
    p.add_argument("--record", default=None, help="write or merge the record (stage2/sets/legacy_check.json)")
    a = ap.parse_args(argv)
    doc = legacy_check(a.body, a.name, a.procs, a.record, a.stage1_run)
    bad = [c for c in doc["checked"] if not c["equal"]]
    print(f"legacy check: {len(doc['checked']) - len(bad)} of {len(doc['checked'])} equal"
          + (f"; record {a.record}" if a.record else "")
          + "".join(f"\n  DIFFERS: {c['body']} {c['name']}" for c in bad), flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    from designlab.stage2 import moves as _named  # the importable copy owns every class and all module state

    sys.exit(_named.main())
