"""Stage 2 moves with the look: the look's laws, frame solver, inverse dynamics and joins for ``designlab.stage2.moves``,
and the periodic look walk in Stage 1's reference format.

``moves.generate`` loads this module by name when a move's laws have ``solver == "look"``; today's laws
(``moves.TODAY``) never reach it.  Nothing here changes a default: a move between two of today's walks keeps today's
laws, and with every look field at its default the laws below give ``moves.TODAY``'s arrays bit for bit
(``laws_for(..., force=True)``; tested).

**The periodic look walk** (``walk_arrays``).  The frames are ``look.generate``'s, unchanged: the look-format file
(``look.arrays``) is the look ladder's own and hashes to it (``stage2/look/L1_FREEZE.json``).  The Stage 1-format file
is derived from the same generation: Stage 1's reference keys (``refs.arrays``' list, in its order) with ``tau``,
``omega``, ``cop_margin``, ``normal`` and ``friction`` from ``look.look_inverse_dynamics`` (F5's rolling split), then
``window_strides`` (K, read by ``control.Track.add_periodic``) and every look key.  A window of K strides is periodic
as a whole: ``stride_motion`` is the window's.

**The look inside a move** (``RULES``; each rule is checked on arrays by ``tests/test_stage2_moves_look.py``):
L1 duty from ``describe``; L2 the tail's amplitude and phase mixed with the ramp's weight; L3 the stabilised head's
settle dip after each touchdown of the plan; L4 the lift shape sampled at lift-off; L5 the pelvis bob between the
plan's steps; L6 trunk roll; L7 step times through the period; L8 the variation envelope; L10 flat feet while the gait is changing, per stance, as
the lead settled it on 15 and 16 Nov 2026 (``STAGE2_TRACKB_PLAN.md`` L10, L11; F5's toe-off lives in the periodic
walks; L9, rolling contact inside a move, is not built); L13 the turn in place, whose flat feet are its own law; and,
in a start into a walk, L1 and L4 with Stage 1's start's duty and lift peak mixed in by the start's amplitude ramp at
each lift-off (S2L-L1L4-start, ruling 18 of 17 Nov 2026).

**Joins to a window of K strides** (J1-J3).  F14 samples a window uniformly in time and warps its phase, so no
lift-off inside a window falls on a frame; a move's frames are uniform in phase.  ``Window`` reads a window at any
phase (its arrays are C2 in time: periodic cubic splines; its LIPM path spectrally), and ``View`` is a window as one
move sees it: the move's phase p is the window's phase p + offset.  A causal move leaves window a at the warp's own
time of its command phase (``View.a_time_s``); every move enters window b at its boundary.

Run nothing from here: ``designlab.stage2.sets`` builds the look sets (``sets.LOOK_SETS``) and calls this module
for what a look set needs beside a set of today's laws (``walk_arrays``, ``bind_move``, ``pick_stride_factor``,
``entry_extra`` and the turn in place's ``hold_move``, ``periodicity`` and ``periodic_cut``).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace

import numpy as np
from scipy.interpolate import CubicSpline

from .. import gait, statics, turning
from ..stage1 import refs, transitions
from ..stage1.transitions import PHASE_TOL, SIDES, _ramp, _rot2
from . import look, moves

N = moves.N_FRAMES
#: Gates of a periodic reference cut from a long plan (design 3.9): consecutive hold strides mapped onto each other
#: by the stride motion.
PERIODIC_COM_MM = 1.0
PERIODIC_ZMP_MM = 2.0
PERIODIC_JOINT_RAD = 1e-3
#: Where a causal move leaves a window of K strides (J1), in strides from the window's first frame.  The values are
#: the bar's lead strides before a command (bar.TURN_LEAD_STRIDES, bar.STOP_LEAD_STRIDES): a left turn's command is
#: a left lift-off (an integer phase), a right turn's and the two-touchdown stop's a right lift-off.
LEAVE_PHASE = {"turn_left": 2.0, "turn_right": 2.5, "stop_two_step": 2.5}

RULES = {
    "S2L-L1": ("F3: a gait's duty is describe(H)['duty'] (moves.GaitX.legacy); inside a ramp it mixes with the ramp's "
               "weight like every other parameter, sampled at each lift-off"),
    "S2L-L2": ("F9: tail = amp x tail_amp(p) x sin(2 pi p + tail_phase(p)), amplitude and phase mixed with the ramp's "
               "weight; a stand has amplitude 0; in a turn's quicker steps tail_amp(p) carries (T(p) / T_walk)^2 "
               "(S2A-appendage)"),
    "S2L-L3": ("F10: a stabilising head holds the neck at minus head_gain x the trunk's pitch about home (zero in these "
               "plans) and dips by head_settle after each touchdown of the plan, deepest settle_lag after it "
               "(64 u^3 (1 - u)^3 over 2 settle_lag); depth and lag are mixed with the ramp's weight at the touchdown, "
               "a nodding gait's share nods; the whole is scaled by the plan's amplitude ramp and, in a turn's quicker "
               "steps, the depth by (T(p) / T_walk)^2 (S2A-appendage)"),
    "S2L-L4": "F4: the lift shape is gait.lift_profile(u, lift_peak), lift_peak mixed with the ramp's weight at the "
              "swing's lift-off",
    "S2L-L5": ("F1/F2: pelvis z = root0_z - amp x (crouch + bob(p) x H x gait.bob_profile(x)); a step runs from one "
               "swing's middle to the next swing's middle (the crests), x its phase fraction; the trough comes "
               "bob_skew x duty after the step's touchdown and the dip lasts bob_drop seconds, as in the periodic "
               "walk; bob amplitude mixed with the ramp's weight per frame, trough and drop at the touchdown; before "
               "the first swing and after the last there is no dip; the correction passes are the gait's own "
               "(look.LOOK_ZMP_ITERS)"),
    "S2L-L6": ("F6: trunk quaternion = yaw(heading) x home x roll(-amp x roll_amp(p) x cos(2 pi (p - (1 - duty(p) / 2) "
               "- roll_phase(p)))), the walk's roll law in phase, amplitude and phase mixed with the ramp's weight"),
    "S2L-L7": ("F14: step-time deviations enter the plan's period step by step: T(p) x (1 + env(p) x 2 e_j "
               "sin^2(2 pi u)), e_j the window's deviation of the step at the window's phase, u the phase inside the "
               "step (gait._Warp's rate).  The variation is a pure warp of time: path speed and yaw rate are divided "
               "by the same factor, so the path, the heading and every footprint are the unvaried plan's in phase "
               "(the window's own footprints are uniform in phase)"),
    "S2L-L8": ("F14's variation envelope env(p) is 1 on the frames a move shares with a periodic window (its warm-up "
               "and join strides) and fades with the minimum-jerk step to 0 across the next stride inward: for "
               "window a, 1 up to the command (a causal move) or up to the end of the first output stride (Stage 1's "
               "construction), 0 from one stride later; for window b, 0 up to two strides before the end, 1 over "
               "the last output stride.  Between the fades a move has no F14 variation (step times, gaze, tail).  "
               "Past the last output frame (computed, not written) the step deviation is the one that makes the last "
               "computed stride last exactly T_b, which the preview's periodic continuation assumes; the ZMP target "
               "is constant there (single support)"),
    "S2L-L10": ("flat feet while the gait is changing, stated per stance (STAGE2_TRACKB_PLAN.md L10 and L11 as the "
                "lead settled them on 15 and 16 Nov 2026).  The instant of a move is its command (a causal move: the "
                "stop, a turn) or the start of its first ramp (a start, a change: move.command_phase, Stage 1's "
                "convention).  A stance runs from a foot's touchdown to its lift-off.  A stance that ends at or before "
                "the instant follows the law of the walk the move leaves (gait a: the roll before its lift-off, "
                "amplitude gait.roll_amplitude at that lift-off, start by gait.rolling_entry of that gait, on the "
                "move's own step times).  A stance that ends after the instant and no later than the end of the move's "
                "last ramp is flat: no roll, no rolling entry, foot_pitch 0 and rolling false through it.  A stance "
                "that ends after the last ramp has ended follows the law of the walk the move enters (gait b), so that "
                "the move's last frame is the walk's first, foot pitch included.  A start has no stance before its "
                "instant but the stand's, which are flat; a stop has no last ramp's end, so every stance after its "
                "command is flat.  The swing that lifts at the instant carries the walk's toe-off pitch to zero by "
                "mid-swing, as in the walk.  The toes-up reach before landing (toe_reach) is part of the same law: a "
                "swing that lands at or before the instant has gait a's; a later swing has the reach only when the "
                "stance it leaves and the stance it lands into both follow a walk's law (the landing stance's gait's "
                "reach), so the swing that lifts at the instant of a move with a ramp, and every swing that leaves or "
                "lands into a flat stance, has none.  Before the "
                "instant a causal move's frames are the look walk's, foot pitch included; a start's and a change's "
                "are the move's own, built with foreknowledge (STAGE2_75_PLAN.md A5), with the foot pitch of the "
                "walk it leaves.  (L9, rolling contact inside a move while the gait is changing, is not built: cuts 1 "
                "and 2 of design 3.9, taken from the start by the lead's ruling of 15 Nov 2026.)"),
    "S2L-L13": ("the turn in place has flat feet as its own law (the lead's ruling of 16 Nov 2026; "
                "STAGE2_TRACKB_PLAN.md L13): no toe-off and no rolling entry in any stride, its start and its stop "
                "included.  It is built from the set's nominal look gait with F5's fields at their defaults "
                "(toe_off 0, toe_reach 0) and without F14 (one stride: cut 4 of design 3.9); rule S2L-L10 is about a "
                "walk's moves and is not named by this gait's entries"),
    "S2L-J1": ("a causal move leaves a window of K strides at the warp's own time of its command phase: a_phase "
               "strides after the window's first frame, a_time_s = _Warp.time(a_phase); the track plays the walk for "
               "whole windows plus a_time_s.  The move's warm-up, its start state and its join_command frames are the "
               "window resampled on the move's phase grid (q, com, zmp_mb by periodic cubic splines in time; the LIPM "
               "path and its derivatives spectrally).  The walk's ZMP correction is the window's own at every computed "
               "frame (it varies from stride to stride), for the continuing walk that the controller previews before "
               "the command and as the commanded plan's first reference"),
    "S2L-J2": ("a move enters a window of K strides at its boundary: its last output stride is compared (join_b) with "
               "the window's last stride resampled on the move's phase grid, and the walk then starts at frame 0; a "
               "move of Stage 1's construction leaves window a at its boundary too (its first output stride is the "
               "window's first stride)"),
    "S2L-J3": "by S2L-L8 a move's own frames carry a window's variation on the shared strides only",
    "S2L-periodic": ("a periodic turn reference is one stride cut from the steady part of a long plan and gated for "
                     "periodicity: the strides before and after it, mapped onto it by the plan's own stride motion, "
                     f"agree within {PERIODIC_COM_MM:g} mm (model CoM), {PERIODIC_ZMP_MM:g} mm (multibody ZMP) and "
                     f"{PERIODIC_JOINT_RAD:g} rad (joints); it carries no F14 variation (cut 4 of design 3.9) and "
                     "flat feet (its own law: S2L-L13)"),
    "S2L-L1L4-start": (
        "L1 and L4 in a start (STAGE2_TRACKB_PLAN.md, the paragraph of that name; ruling 18 of "
        "stage2/design/ROUND_C_RULINGS_17NOV.md; settled 17 Nov 2026 after the look sets' starts were built over the "
        "knee's Stage 0 speed limit, before any look set was frozen and before any closed-loop episode on one).  In a "
        "start, whose gait a is the stand, into a walk, each swing's lift law (L4: gait.lift_profile and its "
        "lift_peak) and its duty (L1) mix the law of Stage 1's start for that body into the look walk's by the "
        "start's amplitude ramp amp (Stage 1's, transitions._param_fns('start')), sampled at the swing's lift-off: "
        "x = x_stage1 + (x_look - x_stage1) amp, exactly x_stage1 at amp 0 and exactly x_look at amp 1.  Stage 1's "
        "start for a body is moves.stage1(body, 'start', H): the stand to Stage 1's nominal walk (refs.walk_specs: "
        "sizing.NOMINAL, which has no swing_time, so that its duty is spec.duty, the constant duty of "
        "transitions._param_fns('start')) under Stage 1's laws (moves.TODAY: the lift sin^2(pi u), which is "
        "gait.lift_profile(u, 0.5), the GaitSpec's lift_peak).  The duty is mixed in the start's schedule (its duty "
        "function), so that every law that reads the duty reads the mixed one, as in a change (L1); the lift height "
        "(lift_frac H) stays the walk's.  Nothing else of the start changes and no other move is touched: a change, a "
        "stop, a turn and the turn in place (its own law, S2L-L13) keep their laws.  The entry's look.start_mix holds "
        "Stage 1's values, the look walk's, and each mixed swing's lift-off, amp, duty and lift peak"),
    "S2L-causal-grid": (
        "the causal preview grid on the look sets (ruling 28 of stage2/design/ROUND_C_RULINGS_17NOV.md, after the "
        "independent review of the join code, stage2/design/checkpoint/reviews/roundC__review_JR_17nov.md, gate 3; "
        "STAGE2_75_PLAN.md A6 and its dated note of 17 Nov 2026; a code error against A6, not a rule change).  A "
        "causal move's preview controller run (moves.com_path_causal: the warm-up and everything after the command, "
        "in every correction pass and in the stride rule's kinematic pass) takes its grid, and so its gains, its "
        "preview length and its switch frame, from the frames up to the command frame alone: dt = min(diff(t[:k_cmd "
        "+ 1])), k_cmd the command frame.  moves.com_path_causal's default, the smallest frame interval of the whole "
        "plan, can lie after the command (a turn's quicker steps, window b's fade-in), so that the warm-up would "
        "depend on what follows the command, which A6 rules out (the preview controller first sees the command at "
        "the command frame).  moves.py is not edited (D18): a look set's move is generated by moves_look.generate "
        "and moves_look.kinematic, which put this grid in front of moves.com_path_causal while that one move is "
        "built and restore the function after it; every other move, the round's three sets' among them, is built by "
        "moves.py as it stands.  Everything else is as coded, and every gate of the move, after the command "
        "included, is read on the arrays built so.  The entry's look.preview_grid holds the grid, the whole plan's "
        "smallest interval and the command frame"),
}
LOOK_RULE_IDS = ("S2L-L1", "S2L-L2", "S2L-L3", "S2L-L4", "S2L-L5", "S2L-L6", "S2L-L7", "S2L-L8", "S2L-L10")
JOIN_RULE_IDS = ("S2L-J1", "S2L-J2", "S2L-J3")
#: The rule of a start's lift law and duty (ruling 18: ``RULES["S2L-L1L4-start"]``), named by a start's entry only.
START_RULE_ID = "S2L-L1L4-start"
#: The rule of a causal move's preview grid on the look sets (ruling 28: ``RULES["S2L-causal-grid"]``), named by the
#: entries of the causal moves (the stop, the turns) only.
CAUSAL_GRID_RULE_ID = "S2L-causal-grid"
#: The rules of the turn in place's three references: L10 is not among them (its flat feet are its own law, L13).
INPLACE_RULE_IDS = tuple(r for r in LOOK_RULE_IDS if r != "S2L-L10") + ("S2L-L13", "S2L-periodic")


def mixes_stage1_start(move: "moves.Move") -> bool:
    """Whether ``RULES["S2L-L1L4-start"]`` is a law of ``move``: a start (``moves.start``: gait a the stand) into a
    walk; not a start into a turn gait or into the turn in place."""
    b = move.b
    return bool(move.builder == "start" and move.a.spec is None and b.spec is not None and not b.in_place
                and b.yaw_rate == 0.0)


def causal_rule_ids() -> tuple:
    """The look rule ids of a causal move's entry (the stop, a turn): ``LOOK_RULE_IDS``, the join rules and
    ``CAUSAL_GRID_RULE_ID``."""
    return LOOK_RULE_IDS + JOIN_RULE_IDS + (CAUSAL_GRID_RULE_ID,)


def rule_ids(move: "moves.Move") -> tuple:
    """The look rule ids of a move's entry: ``LOOK_RULE_IDS`` and the join rules (and ``START_RULE_ID`` for a start
    into a walk, ``CAUSAL_GRID_RULE_ID`` for a causal move: ``causal_rule_ids``), or ``INPLACE_RULE_IDS`` for a move
    of the turn in place (its start and its stop)."""
    if move.a.in_place or move.b.in_place:
        return INPLACE_RULE_IDS
    if move.causal:
        return causal_rule_ids()
    return LOOK_RULE_IDS + JOIN_RULE_IDS + ((START_RULE_ID,) if mixes_stage1_start(move) else ())


def stage1_start_law(body: str, H: float) -> dict:
    """The law of Stage 1's start on ``body`` that ``RULES["S2L-L1L4-start"]`` mixes into a look start, read from
    Stage 1's own start (``moves.stage1(body, "start", H)``: the stand to Stage 1's nominal walk, Stage 1's schedule
    and laws): its duty (the schedule's duty function, one constant) and its lift peak.  ``LookError`` where that
    start's duty is not one constant or its laws are not ``moves.TODAY`` (whose lift, sin^2(pi u), is
    ``gait.lift_profile(u, 0.5)``)."""
    s1 = moves.stage1(body, "start", float(H))
    g = s1.b
    if s1.laws is not moves.TODAY or g.spec is None or s1.a.spec is not None:
        raise LookError(f"{body}: Stage 1's start is not the stand to a walk under Stage 1's laws")
    probe = np.asarray(s1.schedule.fn["duty"](np.linspace(-3.0, 4.0, 57)), float)
    duty = float(probe[0])
    if not np.all(probe == duty) or duty != g.legacy().duty:
        raise LookError(f"{body}: Stage 1's start has no single duty ({probe.min()!r} to {probe.max()!r})")
    lift_peak = float(g.spec.lift_peak)
    if lift_peak != 0.5:  # moves.TODAY's lift is sin^2(pi u): gait.lift_profile's at a peak of 0.5 only
        raise LookError(f"{body}: Stage 1's nominal walk has lift_peak {lift_peak!r}; Stage 1's lift law has 0.5")
    return dict(start=s1.name, walk=g.name, duty=duty, lift_peak=lift_peak,
                source=(f"moves.stage1({body!r}, 'start', H): transitions.TRANSITIONS['start'], the stand to "
                        f"{g.name!r} (refs.walk_specs: sizing.NOMINAL; swing_time None, so its duty is spec.duty, "
                        "transitions._param_fns('start')'s constant duty); moves.TODAY's lift sin^2(pi u), "
                        "gait.lift_profile(u, 0.5)"))


def start_mix(x_stage1, x_look, amp):
    """``RULES["S2L-L1L4-start"]``'s mix: ``x_stage1 + (x_look - x_stage1) amp``, exactly ``x_stage1`` where amp
    <= 0 and exactly ``x_look`` where amp >= 1 (outside the start's amplitude ramp nothing moves)."""
    amp = np.asarray(amp, float)
    x_look = np.asarray(x_look, float)
    return np.where(amp >= 1.0, x_look, np.where(amp <= 0.0, x_stage1 + 0.0 * x_look,
                                                 x_stage1 + (x_look - x_stage1) * amp))


def _start_duty(fn: dict, duty_stage1: float) -> dict:
    """A start's schedule functions with the duty mixed by ``start_mix`` at the start's own amplitude ramp
    (``fn["amp"]``): a swing that lifts off at phase e lands at e + 1 - duty(e) (``moves.plan``)."""
    duty_look, amp = fn["duty"], fn["amp"]

    def duty(p):
        return start_mix(duty_stage1, duty_look(p), amp(p))

    return dict(fn, duty=duty)

#: Keys of a window's arrays whose first column advances with the walk (x); every other key is periodic.
_ADVANCING = ("com", "com_ref", "zmp_mb", "zmp_ref", "q")


class LookError(RuntimeError):
    pass


# ------------------------------------------------------------------------------------------------
# The periodic look walk in Stage 1's reference format
# ------------------------------------------------------------------------------------------------
#: Stage 1's reference keys, in ``refs.arrays``' order.
STAGE1_KEYS = ("t", "q", "qd", "qdd", "tau", "omega", "support", "com_ref", "zmp_ref", "zmp_mb", "cop_margin", "normal",
               "friction", "feet", "com", "q_home", "joints", "dt", "stride_motion")


def walk_arrays(src: dict, spec: gait.GaitSpec) -> tuple[dict, dict, dict]:
    """One periodic walk of ``src``'s body (``look.source``) from one generation: (Stage 1-format arrays, look-format
    arrays, parts).  Call it inside ``s2io.r2()``.  The look-format arrays are ``look.arrays(src, look.generate(src,
    spec))``, nothing else: for a look-ladder spec the file hashes to the ladder's rung.  The Stage 1-format arrays
    are ``STAGE1_KEYS`` from the same frames (the feedforward from ``look.look_inverse_dynamics``, F5's rolling
    split), then ``window_strides`` and every look key not among them.  ``parts``: the generator's info, the dynamics'
    summary (``gait.summarize``), the rolls inside the window and the frames' count."""
    if gait.ACCURACY != look.ACCURACY:
        raise LookError(f"walk_arrays runs at accuracy {look.ACCURACY} (inside s2io.r2()), not {gait.ACCURACY}")
    M = src["M"]
    g = look.generate(src, spec)
    la = look.arrays(src, g)
    info = g["info"]
    tr = look.trajectory(la, info)
    idr = look.look_inverse_dynamics(M, tr, la["rolling"], la["toe_edge"])
    for key in ("qd", "qdd"):
        if not np.array_equal(idr[key], la[key]):
            raise LookError(f"walk: {key} of the inverse dynamics differs from the look-format arrays' by "
                            f"{float(np.max(np.abs(idr[key] - la[key]))):.3e} (accuracy {gait.ACCURACY})")
    K = int(info.get("strides", 1))
    s1 = dict(t=la["t"], q=la["q"], qd=la["qd"], qdd=la["qdd"], tau=idr["tau"], omega=idr["omega"],
              support=la["support"], com_ref=la["com_ref"], zmp_ref=la["zmp_ref"], zmp_mb=la["zmp_mb"],
              cop_margin=idr["cop_margin"], normal=idr["normal"], friction=idr["friction"], feet=la["feet"],
              com=la["com"], q_home=la["q_home"], joints=la["joints"], dt=la["dt"], stride_motion=la["stride_motion"])
    if tuple(s1) != STAGE1_KEYS:
        raise LookError("the Stage 1 keys are out of order")
    s1["window_strides"] = np.array(K)
    for key, val in la.items():
        if key not in s1:
            s1[key] = val
    summary = gait.summarize(M, tr, idr)
    rolls = {s: [dict(r) for r in g["look"]["rolls"][s] if 0 <= r["t_lift_off"] < info["window_s"]] for s in SIDES}
    parts = dict(info={k: v for k, v in info.items() if k != "look"}, look_fields=info.get("look"), summary=summary,
                 rolls=rolls, frames=len(la["t"]), strides=K, split_failed_frames=len(idr["split_failed_frames"]),
                 rolling_frames=int(np.count_nonzero(np.asarray(la["rolling"]).any(axis=1))),
                 dynamics="look.look_inverse_dynamics (gait.inverse_dynamics with F5's rolling contact)",
                 generator="look.generate (gait.generate at accuracy r2)")
    return s1, la, parts


# ------------------------------------------------------------------------------------------------
# A window of K strides read at any phase
# ------------------------------------------------------------------------------------------------
class Window:
    """A periodic look walk's window (``walk_arrays``' Stage 1-format arrays) read at any gait phase.

    ``phi`` is the window's phase in strides (0 at its first frame, the left foot's lift-off; any real: the window
    repeats, advancing ``advance`` in x).  ``at(key, phi)`` resamples an array by a periodic cubic spline in time
    (the window is C2 in time); ``com_state`` and ``correction`` read the LIPM path spectrally (it is band-limited by
    construction)."""

    def __init__(self, name: str, arrays: dict, spec: gait.GaitSpec, H: float):
        self.name, self.arrays, self.spec, self.H = name, arrays, spec, float(H)
        info = spec.describe(H)
        self.T, self.stride, self.duty = float(info["period"]), float(info["stride"]), float(info["duty"])
        self.K = int(arrays["window_strides"]) if "window_strides" in arrays else 1
        if self.K != int(spec.strides):
            raise LookError(f"{name}: the file holds {self.K} strides, the spec {spec.strides}")
        self.frames = len(arrays["t"])
        self.dt = float(arrays["dt"])
        self.W = self.frames * self.dt
        self.advance = float(arrays["stride_motion"][3])
        if abs(self.W - self.K * self.T) > 1e-9 * self.W or abs(self.advance - self.K * self.stride) > 1e-9:
            raise LookError(f"{name}: the window is not {self.K} strides of the spec's walk")
        rng_steps = np.random.default_rng(np.random.SeedSequence(spec.noise_seed).spawn(3)[0])
        self.warp = gait._Warp(self.T, self.K, gait.step_deviations(spec, rng_steps))
        self.e = self.warp.e
        self.varies = bool(np.any(self.e) or spec.gaze_var or spec.tail_var)
        if "phase" in arrays:
            d = float(np.max(np.abs(self.warp.phase(np.asarray(arrays["t"], float)) - arrays["phase"])))
            if d > 1e-9:
                raise LookError(f"{name}: the file's phase is not this spec's warp (largest difference {d:.3e})")
        self._splines: dict = {}
        self._fft = None
        self._delta: dict = {}

    @classmethod
    def load(cls, path, spec: gait.GaitSpec, H: float, name: str | None = None) -> "Window":
        r = refs.Reference.load(path)
        return cls(name or r.name, r.arrays, spec, H)

    # --- time and splines
    def time(self, phi):
        """The window's own time of phase ``phi`` (s; unwrapped: it grows by W per window)."""
        return self.warp.time(np.asarray(phi, float))

    def _spline(self, key: str):
        if key not in self._splines:
            y = np.asarray(self.arrays[key], float)
            shape = y.shape[1:]
            y = y.reshape(self.frames, -1).copy()
            t = np.asarray(self.arrays["t"], float)
            adv = key in _ADVANCING
            if adv:
                y[:, 0] -= self.advance * t / self.W
            if not np.all(np.isfinite(y)):
                raise LookError(f"{self.name}: {key} holds a value that is not finite")
            cs = CubicSpline(np.append(t, self.W), np.vstack([y, y[:1]]), bc_type="periodic")
            self._splines[key] = (cs, adv, shape)
        return self._splines[key]

    def at(self, key: str, phi) -> np.ndarray:
        """``arrays[key]`` at window phases ``phi`` (the first column of an advancing key unwrapped)."""
        cs, adv, shape = self._spline(key)
        t = np.atleast_1d(self.time(phi))
        y = cs(np.mod(t, self.W))
        if adv:
            y[:, 0] += self.advance * t / self.W
        return y.reshape((len(t),) + shape)

    # --- the LIPM path, spectrally
    def _spectrum(self):
        if self._fft is None:
            c = np.asarray(self.arrays["com_ref"], float)
            n = self.frames
            per = c - np.outer(np.arange(n) / n, [self.advance, 0.0])
            C = np.fft.rfft(per, axis=0) / n
            w = 2 * np.pi * np.arange(C.shape[0]) / self.W
            mult = np.full(C.shape[0], 2.0)
            mult[0] = 1.0
            if n % 2 == 0:
                mult[-1] = 1.0
            self._fft = (C, w, mult)
        return self._fft

    def com_state(self, phi: float) -> np.ndarray:
        """(c, c', c'') per axis (3, 2) of the window's LIPM path (``com_ref``) at phase ``phi``."""
        C, w, mult = self._spectrum()
        t = float(self.time(phi))
        ex = np.exp(1j * w * t)[:, None] * mult[:, None]
        x0 = np.zeros((3, 2))
        x0[0] = np.real(np.sum(C * ex, axis=0)) + np.array([self.advance * t / self.W, 0.0])
        x0[1] = np.real(np.sum(C * ex * (1j * w)[:, None], axis=0)) + np.array([self.advance / self.W, 0.0])
        x0[2] = np.real(np.sum(C * ex * (-(w**2))[:, None], axis=0))
        return x0

    def correction(self, omega: float) -> np.ndarray:
        """What the generator added to the ZMP target, per frame of the window (frames, 2): ``com_ref - com_ref'' /
        omega^2 - zmp_ref`` (``moves.walk_correction``'s reading, over the whole window)."""
        key = float(omega)
        if key not in self._delta:
            c = np.asarray(self.arrays["com_ref"], float)
            n = self.frames
            per = c - np.outer(np.arange(n) / n, [self.advance, 0.0])
            k = np.fft.fftfreq(n, d=1.0 / n)
            w = 2 * np.pi * k / self.W
            acc = np.real(np.fft.ifft(-(w[:, None] ** 2) * np.fft.fft(per, axis=0), axis=0))
            delta = c - acc / omega**2 - np.asarray(self.arrays["zmp_ref"], float)
            t = np.asarray(self.arrays["t"], float)
            self._delta[key] = (delta, CubicSpline(np.append(t, self.W), np.vstack([delta, delta[:1]]),
                                                   bc_type="periodic"))
        return self._delta[key][0]

    def correction_at(self, omega: float, phi) -> np.ndarray:
        self.correction(omega)
        return self._delta[float(omega)][1](np.mod(np.atleast_1d(self.time(phi)), self.W))


class View:
    """A window as one move sees it: the move's phase p is the window's phase p + ``offset``; ``first`` is the
    phase at which the move's path origin lies (its plan's ``first``); ``command`` a causal move's command phase.  It stands where a move takes a
    ``refs.Reference`` of one stride: ``n`` is the move's frames per stride."""

    n = N

    def __init__(self, window: Window, offset: float, first: float, command: float | None = None,
                 a_phase: float | None = None):
        if abs(offset - round(offset)) > PHASE_TOL:
            raise LookError("a view's offset is a whole number of strides (left lift-offs are integer phases)")
        self.window, self.offset, self.first = window, float(round(offset)), float(first)
        self.command = None if command is None else float(command)
        self.a_phase = None if a_phase is None else float(a_phase)
        self.name = window.name

    @property
    def a_time_s(self) -> float | None:
        """The window's own time at which a causal move leaves it (J1); None for a boundary join."""
        return None if self.a_phase is None else float(self.window.time(self.a_phase))

    @property
    def step(self) -> float:
        return self.window.T / N

    def phi(self, p):
        return np.asarray(p, float) + self.offset

    def shift(self) -> np.ndarray:
        """The translation that puts the window's frames into the move's plan (path origin at phase ``first``)."""
        return np.array([-self.window.stride * (self.offset + self.first), 0.0])

    def frames(self, key: str, phases) -> np.ndarray:
        y = self.window.at(key, self.phi(phases))
        if key in _ADVANCING:
            y = y.copy()
            y[:, :2] += self.shift() if y.shape[1] >= 2 else 0.0
        return y

    def describe(self) -> dict:
        w = self.window
        return dict(reference=w.name, strides=w.K, window_s=w.W, period_s=w.T, offset_strides=self.offset,
                    a_phase=self.a_phase, a_time_s=self.a_time_s,
                    a_windows=None if self.a_phase is None else self.a_time_s / w.W,
                    varies=w.varies, step_deviations=[float(x) for x in w.e])


# ------------------------------------------------------------------------------------------------
# What a move's laws are bound to
# ------------------------------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class Bind:
    """The frame of one move's look laws: where the move begins and ends (L10) and the windows it shares (L8)."""

    builder: str
    causal: bool
    command: float | None
    first: float
    last: float
    post: float
    p_in: float  # the move's instant as the foot law reads it: the command (causal) or the first ramp's start;
    #              -inf from a stand (a start has no stance before its instant but the stand's)
    p_out: float  # the move's last ramp ends; +inf into a stand
    a_end: float  # window a's variation is 1 up to here and 0 from one stride later
    instant: float | None = None  # the instant as the entry records it (move.command_phase): the command, or the
    #                               start of the move's first ramp (a start's too); None: unbound laws
    va: View | None = None
    vb: View | None = None
    geometry: dict | None = None  # gait.foot_geometry at home
    entry: dict | None = None  # {"a" | "b": gait.rolling_entry's dict of that gait}
    start_law: dict | None = None  # a start into a walk: Stage 1's start's duty and lift peak (S2L-L1L4-start)

    def env_a(self, p, continuing: bool = False) -> np.ndarray:
        p = np.asarray(p, float)
        if self.va is None or not self.va.window.varies:
            return np.zeros_like(p)
        if continuing:
            return np.ones_like(p)
        return 1.0 - _ramp(p - self.a_end)

    def env_b(self, p, continuing: bool = False) -> np.ndarray:
        p = np.asarray(p, float)
        if self.vb is None or not self.vb.window.varies or continuing:
            return np.zeros_like(p)
        return _ramp(p - (self.last - 2.0))


def _step_e(view: View, p: np.ndarray) -> np.ndarray:
    """The window's step deviation at move phases ``p``."""
    w = view.window
    j = np.floor(2.0 * (p + view.offset) + 1e-9).astype(int) % (2 * w.K)
    return w.e[j]


def period_factor_of(bind: Bind, continuing: bool = False):
    """phase -> the factor on the plan's period (L7, L8)."""

    def f(p):
        p = np.asarray(p, float)
        u = p - np.floor(2.0 * p + 1e-9) / 2.0
        s2 = 2.0 * np.sin(2 * np.pi * u) ** 2
        out = np.ones_like(p)
        if bind.va is not None and bind.va.window.varies:
            out = out + bind.env_a(p, continuing) * _step_e(bind.va, p) * s2
        if bind.vb is not None and bind.vb.window.varies and not continuing:
            e = _step_e(bind.vb, p)
            wb = bind.vb.window
            e_post = -(wb.e[2 * wb.K - 2] + 2.0 * wb.e[2 * wb.K - 1])  # the last computed stride lasts T_b (L8)
            e = np.where(p >= bind.last - PHASE_TOL, e_post, e)
            out = out + bind.env_b(p) * e * s2
        return out

    return f


def _mix(xa, xb, w):
    return xa + (xb - xa) * w


def _bump3(u):
    return gait._bump3(u)


def _time_of(ctx: moves.PlanCtx, p: float, factor=None) -> float:
    """The plan's time of phase ``p``: the nearest computed frame at or before it (the first frame for an earlier
    phase) plus the period (the schedule's, times ``factor``: L7) integrated from that frame to ``p``.  Local, so
    that a time before a command depends on nothing after it; a straight line between two frames would be off by a
    microsecond under F14's warp."""
    ph, t = ctx.phase, ctx.t
    p = float(p)
    k = int(np.clip(np.searchsorted(ph, p, side="right") - 1, 0, len(ph) - 1))
    if p == ph[k]:
        return float(t[k])
    x = np.linspace(ph[k], p, 17 if ph[0] <= p <= ph[-1] else 257)
    T = np.asarray(ctx.fn["T"](x), float) * (1.0 if factor is None else factor(x))
    return float(t[k] + np.sum(0.5 * (T[1:] + T[:-1]) * np.diff(x)))


def _weight_at(ctx: moves.PlanCtx, p: float) -> float:
    return float(np.interp(p, ctx.phase, ctx.weight))


#: Plans whose foot law ran, by their swing array (``idyn`` gets a slice of it and finds the plan's rolling contact).
_PLANS: list = []


def _register(swing: np.ndarray, foot: dict) -> None:
    _PLANS.append((swing, foot))
    del _PLANS[:-16]


def _registered(swing_view: np.ndarray):
    base = swing_view if swing_view.base is None else swing_view.base
    for sw, foot in reversed(_PLANS):
        if sw is base:
            k0 = (swing_view.__array_interface__["data"][0] - sw.__array_interface__["data"][0]) // sw.strides[0]
            return foot, int(k0)
    raise LookError("idyn: these frames' plan is not one the look's foot law made (no rolling contact is known)")


def laws_for(a: moves.GaitX, b: moves.GaitX, bind: Bind | None = None, force: bool = False) -> moves.Laws:
    """The laws of a move from gait ``a`` to gait ``b`` (``RULES``).  With no look field active on either gait and no
    window bound, today's laws (``moves.TODAY``), unless ``force``: then the look's laws and solver, which at the
    default fields compute today's values by today's expressions."""
    specs = [g.spec for g in (a, b) if g.spec is not None]
    if not specs:
        raise ValueError("a move joins at least one gait")
    active = any(gait.look_active(s) for s in specs)
    bound = bind is not None and (bind.va is not None or bind.vb is not None)
    if not (active or bound or force):
        return moves.TODAY
    sa = a.spec if a.spec is not None else b.spec
    sb = b.spec if b.spec is not None else a.spec
    H = a.H if a.spec is not None else b.H
    same = a.spec is not None and b.spec is not None and a.spec == b.spec
    if bind is None:
        bind = Bind(builder="unbound", causal=False, command=None, first=-math.inf, last=math.inf, post=0.25,
                    p_in=-math.inf, p_out=math.inf, a_end=-math.inf)
    for key in ("bob_skew", "bob_drop"):
        if (getattr(sa, key) is None) != (getattr(sb, key) is None):
            raise ValueError(f"a move between a gait with {key} and one without is not defined (S2L-L5)")
    for s in (sa, sb):
        if s.head_mode not in ("nod", "stabilise"):
            raise ValueError(f"head_mode {s.head_mode!r}: 'nod' or 'stabilise'")
        if s.zmp_inset_y:
            raise ValueError("F7 (zmp_inset_y) is not built in moves")

    def continuing(ctx) -> bool:  # the walk that would have continued, planned by moves for a causal move's preview
        return bool(bind.causal and ctx.command is None)

    def time_of(ctx, p) -> float:
        return _time_of(ctx, p, period_factor_of(bind, continuing(ctx)))

    def appendage(ctx, p):
        """(T(p) / T_walk)^2 where a turn's steps are quicker than the walk's, else 1 (S2A-appendage)."""
        if not same:
            return np.ones_like(np.asarray(p, float))
        return np.minimum(1.0, (np.asarray(ctx.fn["T"](p), float) / a.legacy().period) ** 2)

    def variation(ctx, key_spec: str, key_arr: str):
        """env_a x amount_a x g_a + env_b x amount_b x g_b of one slow variation, per frame (None: none)."""
        out = None
        cont = continuing(ctx)
        for view, s, env in ((bind.va, sa, bind.env_a(ctx.phase, cont)), (bind.vb, sb, bind.env_b(ctx.phase, cont))):
            amount = getattr(s, key_spec)
            if view is None or not amount or not np.any(env):
                continue
            if key_arr not in view.window.arrays:
                raise LookError(f"{view.name}: the window's file has no {key_arr!r}")
            term = env * amount * view.window.at(key_arr, view.phi(ctx.phase))
            out = term if out is None else out + term
        return out

    # --- L4
    def lift(u, ctx, swing):
        lp = _mix(sa.lift_peak, sb.lift_peak, _weight_at(ctx, swing[1]))
        if bind.start_law is not None:  # S2L-L1L4-start: Stage 1's start's lift peak mixed in by amp at lift-off
            lp = float(start_mix(bind.start_law["lift_peak"], lp, ctx.fn["amp"](swing[1])))
        if lp == 0.5:
            return math.sin(math.pi * u) ** 2
        return float(gait.lift_profile(u, lp))

    # --- L5
    def bob(ctx) -> np.ndarray | None:
        if not (sa.bob or sb.bob):
            return None
        ph = ctx.phase
        prof = np.zeros(len(ph))
        sw = sorted(ctx.swings, key=lambda s: s[1])
        mids = [0.5 * (s[1] + s[2]) + _mix(sa.bob_phase, sb.bob_phase, _weight_at(ctx, s[2])) for s in sw]
        for i in range(len(sw) - 1):
            m0, m1, td = mids[i], mids[i + 1], sw[i][2]
            sel = (ph >= m0) & (ph < m1)
            if not sel.any():
                continue
            w = _weight_at(ctx, td)
            d = float(ctx.fn["duty"](td))
            x_tr = [0.5 if s.bob_skew is None else (td + s.bob_skew * d - m0) / (m1 - m0) for s in (sa, sb)]
            trough = _mix(x_tr[0], x_tr[1], w)
            drop = None
            if sa.bob_drop is not None:
                drop = _mix(sa.bob_drop, sb.bob_drop, w) / ((m1 - m0) * float(ctx.fn["T"](td)))
            prof[sel] = gait.bob_profile((ph[sel] - m0) / (m1 - m0), trough, drop)
        return _mix(sa.bob, sb.bob, ctx.weight) * H * prof

    def height(ctx):
        z = ctx.geo.root0[2] - ctx.amp * ctx.fn["crouch"](ctx.phase)
        dz = bob(ctx)
        return z if dz is None else z - ctx.amp * dz

    # --- L3
    def neck(ctx):
        ph = ctx.phase
        stab = [1.0 if s.head_mode == "stabilise" else 0.0 for s in (sa, sb)]
        share = _mix(stab[0], stab[1], ctx.weight)
        out = ctx.amp * ctx.fn["head_amp"](ph) * np.sin(4 * np.pi * ph)
        if any(stab):
            out = out * (1.0 - share)
            settle = np.zeros(len(ph))
            for s in ctx.swings:
                td = s[2]
                w = _weight_at(ctx, td)
                depth = _mix(stab[0] * sa.head_settle, stab[1] * sb.head_settle, w) * float(appendage(ctx, td))
                lag = _mix(sa.settle_lag, sb.settle_lag, w)
                if depth:
                    settle += depth * _bump3((ctx.t - time_of(ctx, td)) / (2.0 * lag))
            out = out - ctx.amp * settle
        gaze = variation(ctx, "gaze_var", "gaze_var")
        return out if gaze is None else out + ctx.amp * gaze

    # --- L2
    def tail(ctx):
        ph = ctx.phase
        tp = _mix(sa.tail_phase, sb.tail_phase, ctx.weight)
        amp = ctx.amp * ctx.fn["tail_amp"](ph)
        var = variation(ctx, "tail_var", "tail_var")
        if var is not None:
            amp = amp * (1.0 + var)
        return amp * np.sin(2 * np.pi * ph + tp)

    # --- L6
    def trunk_roll(ctx):
        ph = ctx.phase
        ra = _mix(sa.roll_amp, sb.roll_amp, ctx.weight)
        rp = _mix(sa.roll_phase, sb.roll_phase, ctx.weight)
        return -ctx.amp * ra * np.cos(2 * np.pi * (ph - (1.0 - ctx.fn["duty"](ph) / 2.0) - rp))

    # --- L10
    def foot(ctx):
        K = len(ctx.phase)
        pitch = np.zeros((K, 2))
        offset = np.zeros((K, 2, 3))
        contact = np.full((K, 2, 2, 3), np.nan)
        rolling = np.zeros((K, 2), bool)
        cont = continuing(ctx)
        tol = 1e-6
        toes = bool(sa.toe_off or sb.toe_off or sa.toe_reach or sb.toe_reach)
        if toes:
            if bind.geometry is None:
                raise LookError("the toe-off of a look gait needs the body's foot geometry (bind_move)")
            by = {"a": sa, "b": sb}
            for j, side in enumerate(SIDES):
                sw = sorted((s for s in ctx.swings if s[0] == side), key=lambda s: s[1])
                for i, (_, lo, td, _lift, _f0, _f1) in enumerate(sw):
                    began = sw[i - 1][2] if i > 0 else None  # this stance began at the previous touchdown
                    # the stance that ends at this lift-off (L10 as settled): gait a's law if it ends at or before
                    # the instant, gait b's if it ends after the last ramp has ended, flat between the two
                    which = None
                    if cont or lo <= bind.p_in + tol:
                        which = "a"
                    elif lo > bind.p_out + tol:
                        which = "b"
                    if a.spec is None and which == "a":
                        which = None
                    if b.spec is None and which == "b":
                        which = None
                    if began is None and i == 0 and a.spec is None:
                        which = None
                    s = by.get(which)
                    if s is not None and s.toe_off:
                        ent = bind.entry[which][side]
                        if ent is None:
                            raise ValueError(f"F5: the ZMP reference never lies {s.toe_off_margin} m inside the "
                                             f"rolling support when the {side} foot trails")
                        d = float(ctx.fn["duty"](lo))
                        f_r = max(lo - (1.0 - s.toe_off_start) * d, lo - (1.0 - ent["a"]) * (d - 0.5))
                        t_r, t_lo, t_e = (time_of(ctx, f) for f in (f_r, lo, lo + 0.5 * (td - lo)))
                        amp = gait.roll_amplitude(s, t_r, t_lo)
                        pitch[:, j] += gait.toe_off_bump(ctx.t, t_r, t_lo, t_e, amp)[0]
                    # the swing that begins at this lift-off: the toes-up reach before landing.  A swing that lands
                    # at or before the instant has gait a's (the frames before the instant follow the walk the move
                    # leaves); a later swing has it only when the stance it leaves and the stance it lands into
                    # both follow a walk's law, and then the landing stance's
                    leave = which
                    lo_next = sw[i + 1][1] if i + 1 < len(sw) else None  # the landing stance ends there
                    land = None if lo_next is None else ("a" if lo_next <= bind.p_in + tol else
                                                         ("b" if lo_next > bind.p_out + tol else None))
                    if cont or td <= bind.p_in + tol:
                        which = "a"
                    else:
                        which = land if (leave is not None and land is not None) else None
                    if (which == "a" and a.spec is None) or (which == "b" and b.spec is None):
                        which = None
                    s = by.get(which)
                    if s is not None and s.toe_reach:
                        u = ctx.swing_u[:, j]
                        sel = ctx.swing[:, j] & (ctx.phase >= lo - PHASE_TOL) & (ctx.phase < td) & (u > 0.5)
                        pitch[sel, j] -= s.toe_reach * _bump3(2.0 * u[sel] - 1.0)
            rolling = ~ctx.swing & (pitch > 1e-9)
            if np.any(rolling & (np.sum(ctx.swing, axis=1) == 1)[:, None]):
                raise ValueError("F5: a foot rolls in single support (the roll must start in double support)")
            geo = bind.geometry
            for j, side in enumerate(SIDES):
                for k in np.flatnonzero(pitch[:, j]):
                    off, sup, _part = gait.roll_offset(geo[side], float(pitch[k, j]))
                    offset[k, j] = off
                    contact[k, j] = sup
        out = dict(pitch=pitch, contact_offset=offset, rolling=rolling, contact_foot=contact,
                   var_env=np.column_stack([bind.env_a(ctx.phase, cont), bind.env_b(ctx.phase, cont)]),
                   bob=bob(ctx), flat=not np.any(pitch))
        _register(ctx.swing, out)
        return out

    # --- J1: the continuing walk's ZMP points on the window's own (uniform) footprints
    zmp_point = None
    if bind.causal and bind.va is not None and bind.va.window.varies:
        stride_a = bind.va.window.stride

        def zmp_point(ctx, stance):
            side, s_from, s_to, (xy, yaw) = stance
            if not continuing(ctx):
                return xy + _rot2(yaw) @ ctx.geo.pad_c[side]
            pm = 0.5 * (s_from + s_to)
            return (np.array([stride_a * (pm - bind.command), 0.0]) + ctx.geo.feet0[side][:2] + ctx.geo.pad_c[side])

    varies = any(v is not None and v.window.varies for v in (bind.va, bind.vb))
    factor = (lambda pre: period_factor_of(bind, continuing(pre))) if varies else None
    roll = trunk_roll if (sa.roll_amp or sb.roll_amp) else None
    return moves.Laws(name="look", lift=lift, height=height, neck=neck, tail=tail, trunk_roll=roll, foot=foot,
                      zmp_point=zmp_point, period_factor=factor, solver="look")


# ------------------------------------------------------------------------------------------------
# Moves bound to their windows
# ------------------------------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class Bound:
    """A move with the look's laws and what ``moves.generate`` and ``moves.summary`` take beside it."""

    move: moves.Move
    ref_a: object  # View | refs.Reference | None
    ref_b: object
    bind: Bind

    def look(self) -> dict:
        """The entry's ``look`` block: the laws' frame and the joins (JSON-plain)."""
        b = self.bind
        fin = lambda x: None if x is None or not math.isfinite(x) else float(x)  # noqa: E731
        va = self.ref_a if isinstance(self.ref_a, View) else None
        inplace = self.move.a.in_place or self.move.b.in_place
        return refs._plain(dict(
            laws="look", rules=list(rule_ids(self.move)), instant_phase=fin(b.instant),
            flat_from_phase=fin(b.p_in), flat_until_phase=fin(b.p_out), variation_a_until_phase=fin(b.a_end),
            window_a=None if va is None else va.describe(),
            window_b=None if not isinstance(self.ref_b, View) else self.ref_b.describe(),
            a_phase=None if va is None else va.a_phase, a_time_s=None if va is None else va.a_time_s,
            a_windows=None if va is None or va.a_phase is None else va.a_time_s / va.window.W,
            feet=("flat in every stride: this gait's own law (S2L-L13)" if inplace else
                  "flat in every stance that ends after the instant (instant_phase) and no later than the end of the "
                  "move's last ramp (flat_until_phase; null: none ends later); the walk's own law before and after "
                  "(S2L-L10)")))


def _entries(src: dict, a: moves.GaitX, b: moves.GaitX) -> tuple[dict | None, dict | None]:
    """The body's foot geometry and each gait's rolling entry (``gait.rolling_entry``), where a gait has F5."""
    specs = {k: g.spec for k, g in (("a", a), ("b", b)) if g.spec is not None}
    if not any(s.toe_off or s.toe_reach for s in specs.values()):
        return None, None
    M, q0 = src["M"], src["q_home"]
    M.forward(q0)
    geo = src.get("geometry") or gait.foot_geometry(M)
    feet0 = {s: M.foot_pose(s)[0].copy() for s in SIDES}
    pad_c = {s: statics._polygon_centroid(M.pad_polygon(s)) - feet0[s][:2] for s in SIDES}
    inset = {s: np.zeros(2) for s in SIDES}
    entry = {}
    for k, g in (("a", a), ("b", b)):
        s = specs.get(k)
        entry[k] = None
        if s is not None and s.toe_off:
            entry[k] = gait.rolling_entry(geo, pad_c, inset, feet0, float(s.describe(g.H)["stride"]), s.toe_off_margin)
    return geo, entry


def _warped(fn: dict, factor) -> dict:
    """The schedule's phase functions with path speed and yaw rate divided by the period's factor (L7)."""
    v, w = fn["v"], fn["w"]
    return dict(fn, v=lambda p: v(p) / factor(p), w=lambda p: w(p) / factor(p))


def bind_move(src: dict, move: moves.Move, wa: Window | None, wb: Window | None,
              a_phase: float | None = None) -> Bound:
    """``move`` (one of ``moves``' builders' moves) with the look's laws, bound to gait a's window ``wa`` and gait
    b's ``wb`` (None: a stand, or a reference of one stride given later as a plain ``refs.Reference``).  ``a_phase``:
    where a causal move leaves window a (J1; default ``LEAVE_PHASE``)."""
    sch = move.schedule
    first, last = float(sch.window[0]), float(sch.window[1])
    if sch.post != 0.25:
        raise LookError("the look's laws assume a quarter stride computed past the last output frame")
    ramp_ends = [float(v[1]) for v in sch.ramps.values()]
    ramp_starts = [float(v[0]) for v in sch.ramps.values()]
    if move.causal:
        c = float(sch.command)
        if a_phase is None:
            a_phase = leave_phase(move)
        offset = a_phase - c
        va = None if wa is None else View(wa, offset, first, command=c, a_phase=a_phase)
        p_in, a_end, instant = c, c, c
    else:
        va = None if wa is None else View(wa, -first, first)
        instant = min(ramp_starts) if ramp_starts else first  # the start of the move's first ramp (L10)
        p_in = instant if move.a.spec is not None else -math.inf
        a_end = first + 1.0
        if va is not None and wa.varies and a_end + 1.0 > p_in + PHASE_TOL:
            raise LookError(f"{move.name}: window a's variation would fade inside the move's first ramp")
    p_out = (max(ramp_ends) if ramp_ends else p_in) if move.b.spec is not None else math.inf
    vb = None if wb is None else View(wb, wb.K - last, last - 1.0)
    if vb is not None and wb.varies and ramp_ends and last - 2.0 < p_out - PHASE_TOL:
        raise LookError(f"{move.name}: window b's variation would return inside the move's last ramp")
    geo, entry = _entries(src, move.a, move.b)
    bind = Bind(builder=move.builder, causal=move.causal, command=sch.command, first=first, last=last, post=sch.post,
                p_in=p_in, p_out=p_out, a_end=a_end, instant=float(instant), va=va, vb=vb, geometry=geo, entry=entry)
    laws = laws_for(move.a, move.b, bind)
    if laws is moves.TODAY:
        return Bound(move, va, vb, bind)
    if mixes_stage1_start(move):  # S2L-L1L4-start: the lift law (laws_for's lift) and the duty mixed by amp
        bind = replace(bind, start_law=stage1_start_law(src["body"], move.b.H))
        laws = laws_for(move.a, move.b, bind)
    fn = sch.fn if laws.period_factor is None else _warped(sch.fn, period_factor_of(bind))
    if bind.start_law is not None:
        fn = _start_duty(fn, bind.start_law["duty"])
    new = replace(move, schedule=replace(sch, fn=fn), laws=laws, rules=tuple(move.rules) + rule_ids(move))
    return Bound(new, va, vb, bind)


def definition_of(move: moves.Move, a_phase: float | None = None) -> dict:
    """The definition a move of ``moves``' builders has once it is bound to the look's laws (``bind_move``), made
    without a body or a window: ``Move.definition()`` with ``laws`` "look", the look's rule ids appended and, for a
    causal move, where it leaves its walk's window (``leave_phase``: J1).  ``sets.items`` holds it as the item's
    identity."""
    d = move.definition()
    d.update(laws="look", rules=list(move.rules) + list(rule_ids(move)))
    if move.causal:
        d["leave_phase"] = float(leave_phase(move) if a_phase is None else a_phase)
    return refs._plain(d)


def leave_phase(move: moves.Move) -> float:
    """Where a causal move of the bar leaves its walk's window (``LEAVE_PHASE``; J1)."""
    if move.builder == "stop_two_step":
        return LEAVE_PHASE["stop_two_step"]
    if move.builder == "turn":
        return LEAVE_PHASE[f"turn_{'left' if move.args['rate'] > 0 else 'right'}"]
    raise LookError(f"{move.name}: say where this causal move leaves its walk's window (a_phase)")


def start(src, name, b: moves.GaitX, wb: Window | None) -> Bound:
    return bind_move(src, moves.start(name, b), None, wb)


def change(src, name, a: moves.GaitX, b: moves.GaitX, wa: Window | None, wb: Window | None,
           ramp_strides: float = 1.0) -> Bound:
    return bind_move(src, moves.change(name, a, b, ramp_strides), wa, wb)


def stop_one_step(src, name, a: moves.GaitX, wa: Window | None) -> Bound:
    return bind_move(src, moves.stop_one_step(name, a), wa, None)


def stop_two_step(src, name, a: moves.GaitX, wa: Window | None, a_phase: float | None = None) -> Bound:
    return bind_move(src, moves.stop_two_step(name, a), wa, None, a_phase)


def turn(src, name, a: moves.GaitX, wa: Window | None, angle: float, rate: float, stride_factor: float,
         ramp_s: float = moves.RAMP_S, a_phase: float | None = None) -> Bound:
    return bind_move(src, moves.turn(name, a, angle, rate, stride_factor, ramp_s), wa, wa, a_phase)


# ------------------------------------------------------------------------------------------------
# The causal preview grid of a look set's move (ruling 28: RULES["S2L-causal-grid"])
# ------------------------------------------------------------------------------------------------
def causal_grid_s(pl: transitions.Plan, k_cmd: int) -> float:
    """The preview grid of a look set's causal move: the smallest frame interval up to the command frame ``k_cmd``,
    ``min(diff(t[:k_cmd + 1]))`` (``RULES["S2L-causal-grid"]``)."""
    t = np.asarray(pl.t, float)[: int(k_cmd) + 1]
    if len(t) < 2:
        raise LookError("a causal move's command frame has no frame before it: no preview grid up to the command")
    return float(np.min(np.diff(t)))


class causal_grid:
    """While open, ``moves.com_path_causal`` takes a causal move's grid from the frames up to its command frame
    (``causal_grid_s``) where its caller passes none (``moves._generate_causal`` and ``moves._kinematic`` pass none);
    a grid a caller passes is used as given.  ``moves.py`` is not edited (D18): the function is replaced in
    ``moves``' namespace for the block and put back after it, also when the block raises.  The context refuses to
    open while another replacement is installed, and on leaving puts the function back only if its own is the one
    installed (otherwise it raises and leaves what it finds).  Opened only by ``generate`` and ``kinematic`` for a
    look set's causal move; every other caller of ``moves`` sees the function as it stands."""

    MARK = "_moves_look_causal_grid"

    def __enter__(self):
        own = moves.com_path_causal
        if getattr(own, self.MARK, False):
            raise LookError("moves.com_path_causal is already replaced by a causal_grid context (it is not re-entered)")

        def com_path_causal(pl, zcmd, zpre_t, zpre, x0, omega, k_cmd, dt=None):
            return own(pl, zcmd, zpre_t, zpre, x0, omega, k_cmd, causal_grid_s(pl, k_cmd) if dt is None else dt)

        setattr(com_path_causal, self.MARK, True)
        com_path_causal.__doc__ = "moves.com_path_causal on the causal grid (moves_look.causal_grid)"
        self.own, self.mine = own, com_path_causal
        moves.com_path_causal = com_path_causal
        return self

    def __exit__(self, *exc):
        if moves.com_path_causal is not self.mine:
            raise LookError("moves.com_path_causal was replaced inside a causal_grid context: it is not put back")
        moves.com_path_causal = self.own
        return False


def generate(src: dict, move: moves.Move, ref_a, verbose: bool = False) -> dict:
    """``moves.generate`` for a look set's move: a causal move (the stop, a turn) is generated inside ``causal_grid``
    (its controller run on the grid of the frames up to its command, ``RULES["S2L-causal-grid"]``); any other move
    is ``moves.generate``'s, unchanged."""
    if not move.causal:
        return moves.generate(src, move, ref_a, verbose)
    with causal_grid():
        return moves.generate(src, move, ref_a, verbose)


def kinematic(src: dict, move: moves.Move, ref_a) -> dict:
    """``moves.kinematic`` for a look set's move (the stride rule's kinematic pass): a causal move's first CoM path
    on the causal grid (``causal_grid``), as ``generate`` builds the move."""
    if not move.causal:
        return moves.kinematic(src, move, ref_a)
    with causal_grid():
        return moves.kinematic(src, move, ref_a)


def preview_grid_record(g: dict) -> dict | None:
    """What a causal move's entry records of its preview grid (``look.preview_grid``): the rule, the grid (the
    smallest frame interval up to the command frame), the whole plan's smallest interval (``moves.com_path_causal``'s
    default) and the command frame.  None for a move that is not causal."""
    if not g.get("causal"):
        return None
    pl = g["plan"]
    k = int(pl.out.start)
    return dict(rule=CAUSAL_GRID_RULE_ID, grid_s=causal_grid_s(pl, k),
                whole_plan_min_s=float(np.min(np.diff(np.asarray(pl.t, float)))), command_frame=k)


def pick_stride_factor(src: dict, a: moves.GaitX, wa: Window | None, angle: float, rate: float,
                       grid=moves.STRIDE_GRID, limit: float = moves.TWIST_KINEMATIC,
                       other_margin: float = moves.OTHER_JOINT_MARGIN_RAD) -> dict:
    """``moves.pick_stride_factor`` (the stride rule's kinematic pick: the same rule, readings and record) on the
    look's turns: each kinematic pass is the turn bound to the look's laws and to the walk's window (``turn``), on
    the causal grid (``kinematic``)."""
    tried, chosen = [], None
    for f in grid:
        row = dict(f=float(f), left=None, right=None, tightest_other={}, ik_fail_frames={}, ik_ok=True, pass_=True)
        for direction, sign in (("left", 1.0), ("right", -1.0)):
            try:
                bd = turn(src, f"turn_{direction}", a, wa, angle, sign * abs(float(rate)), f)
            except ValueError as err:
                row.update(ik_ok=False, pass_=False, error=str(err))
                continue
            k = kinematic(src, bd.move, bd.ref_a)  # (a causal move: on the causal grid)
            use = moves.hip_yaw_use(k["range_use"])
            other = k.get("tightest_other")
            row[direction] = use
            row["tightest_other"][direction] = None if other is None else dict(
                joint=other["joint"], end=other["end"], margin_rad=float(other["margin_rad"]))
            row["ik_fail_frames"][direction] = int(k["ik_fail_frames"])
            row["ik_ok"] = bool(row["ik_ok"] and k["ik_ok"])
            row["pass_"] = bool(row["pass_"] and k["ik_ok"] and use <= limit
                                and other is not None and other["margin_rad"] >= other_margin)
        tried.append(row)
        if row["pass_"]:
            chosen = float(f)
            break
    steady = None if chosen is None else moves.steady_turn_kinematic(src, a, rate, chosen)
    return dict(rule="S2A-stride", text=moves.RULES["S2A-stride"], grid=[float(f) for f in grid],
                limit_kinematic=float(limit), limit_final=moves.TWIST_FINAL, limit_other_joints_rad=float(other_margin),
                steps_down_max=moves.STEPS_DOWN_MAX,
                angle_rad=float(angle), rate_rad_s=abs(float(rate)), tried=tried, chosen=chosen,
                steady_turn_kinematic=steady, steps_down=0, builds=[], outcome="picked" if chosen is not None else "none",
                reason=None if chosen is not None else
                f"no stride factor of the grid passes the kinematic rule (thigh twist at most {limit:g} of its range, "
                f"every other gated joint at least {other_margin:g} rad of range, ik_ok, in both directions)")


def entry_extra(bd: Bound, g: dict, entry: dict) -> dict:
    """What a look move's entry holds beside ``moves.summary``'s keys, written into ``entry`` (and returned):

    * ``move.command_phase``: the move's instant.  ``moves.summary`` gives a causal move's command; for a move of
      Stage 1's construction (a start, a change, the one-step stop) it gives None, and rule L10 names the instant
      there too: the start of the move's first ramp (``Bind.instant``; Stage 1's convention).  ``move.causal`` and
      ``move.command_t_s`` stay as they are: the file of such a move does not begin at its instant;
    * ``look``: the laws' frame and the joins (``Bound.look``), and in a start into a walk ``look.start_mix``
      (``start_mix_record``: ``RULES["S2L-L1L4-start"]``);
    * ``join_events``: a causal move's warm-up ZMP read apart from the plan's events (a reading, not a gate);
    * ``look.preview_grid``: a causal move's preview grid (``preview_grid_record``; ``RULES["S2L-causal-grid"]``);
    * ``rolling_split_failed_frames`` and the gate ``rolling_split``: on a move that has rolling frames, the frames
      whose rolling-contact split did not solve (``idyn``): any such frame fails the entry (its feedforward torque
      there is not the rolling contact's).  A move without a rolling frame has no such gate."""
    move = entry["move"]
    if move.get("command_phase") is None and bd.bind.instant is not None:
        move["command_phase"] = float(bd.bind.instant)
    foot = g["plan"].extras["foot"]
    failed = len(foot.get("split_failed_frames", []))
    rolls = int(np.count_nonzero(foot["rolling"][g["plan"].out].any(axis=1)))
    entry.update(look=bd.look(), join_events=join_events(g, bd.ref_a), rolling_frames=rolls,
                 rolling_split_failed_frames=failed)
    grid = preview_grid_record(g)
    if grid is not None:  # (RULES["S2L-causal-grid"])
        entry["look"]["preview_grid"] = grid
    if bd.bind.start_law is not None:
        entry["look"]["start_mix"] = start_mix_record(bd, g["plan"])
    if rolls:
        entry["gates"]["rolling_split"] = bool(failed == 0)
        entry["pass_"] = bool(all(entry["gates"].values()))
    return entry


def start_mix_record(bd: Bound, pl: transitions.Plan) -> dict:
    """What a start's entry records of ``RULES["S2L-L1L4-start"]`` (``look.start_mix``): Stage 1's start's duty and
    lift peak and where they come from, the look walk's, and every swing whose lift-off lies inside the start's
    amplitude ramp (amp < 1; the later swings are the look walk's exactly): its side, lift-off phase, amp there, the
    duty and lift peak the mix gives (``duty`` is the schedule's at the lift-off, ``duty_plan`` 1 minus the swing's
    length in the plan)."""
    law, b, fn = bd.bind.start_law, bd.move.b, pl.ctx.fn
    swings = []
    for side, lo, td, *_ in pl.swings:
        amp = float(fn["amp"](lo))
        if amp >= 1.0:
            continue
        lp_walk = _mix(b.spec.lift_peak, b.spec.lift_peak, _weight_at(pl.ctx, lo))  # (laws_for's lift: weight 1)
        swings.append(dict(side=side, lift_off_phase=float(lo), amp=amp, duty=float(fn["duty"](lo)),
                           duty_plan=float(1.0 - (td - lo)),
                           lift_peak=float(start_mix(law["lift_peak"], lp_walk, amp))))
    return refs._plain(dict(rule=START_RULE_ID, stage1=dict(law), look=dict(
        walk=b.name, duty=float(b.legacy().duty), lift_peak=float(b.spec.lift_peak)), swings=swings))


# ------------------------------------------------------------------------------------------------
# The solver side (what moves.generate calls)
# ------------------------------------------------------------------------------------------------
def solve_frames(M: statics.Model, q_home, pl: transitions.Plan, com_t: np.ndarray, geo: transitions.Geometry):
    """``transitions.solve_frames`` plus the plan's trunk roll (L6) and foot pitch with its contact offset (a foot
    that rolls keeps its floor contact: ``gait.roll_offset``, about the yawed foot's own lateral axis).  With neither,
    it is ``transitions.solve_frames`` itself."""
    _world_contacts(pl)
    ex = getattr(pl, "extras", None) or {}
    roll, foot = ex.get("trunk_roll"), ex.get("foot")
    if roll is not None and not np.any(roll):
        roll = None
    if roll is None and (foot is None or foot["flat"]):
        return transitions.solve_frames(M, q_home, pl, com_t, geo)
    m = M.m
    K = len(pl.phase)
    neck = m.jnt_qposadr[statics.jid(m, "neck_pitch")]
    tail = m.jnt_qposadr[statics.jid(m, "tail_yaw")]
    off = geo.com0[:2] - geo.root0[:2]
    qs = np.zeros((K, m.nq))
    q = np.asarray(q_home, float).copy()
    foot_err = np.zeros((K, 2))
    com_err = np.zeros(K)
    pitch = None if foot is None else foot["pitch"]
    for k in range(K):
        targets = {}
        for j, s in enumerate(SIDES):
            Rz = turning._rotz(pl.foot_yaw[k, j])
            if pitch is not None and pitch[k, j]:
                targets[s] = (pl.foot_pos[k, j] + Rz @ foot["contact_offset"][k, j], Rz @ gait._rot_y(float(pitch[k, j])))
            else:
                targets[s] = (pl.foot_pos[k, j], Rz)
        q[2] = pl.root_z[k]
        quat = turning._quat_yaw(pl.heading[k], geo.quat0)
        if roll is not None:
            quat = gait._quat_mul(quat, [math.cos(roll[k] / 2), math.sin(roll[k] / 2), 0.0, 0.0])
        q[3:7] = quat
        q[neck], q[tail] = pl.neck[k], pl.tail[k]
        q[:2] = com_t[k] - _rot2(pl.heading[k]) @ off
        for _ in range(transitions.COM_ITERS):
            q, _ok = statics.leg_ik(M, q, targets, iters=transitions.IK_ITERS)
            err = com_t[k] - M.com()[:2]
            if np.linalg.norm(err) < transitions.COM_TOL:
                break
            q[0] += err[0]
            q[1] += err[1]
        M.forward(q)
        foot_err[k] = turning._foot_errors(M, targets)
        com_err[k] = np.linalg.norm(com_t[k] - M.com()[:2])
        qs[k] = q
    return qs, foot_err, com_err


def idyn(M: statics.Model, qs, qd, qdd, swing, split: str):
    """``transitions.idyn`` with F5's rolling contact where a stance foot of the plan rolls (double support only:
    ``look._rolling_contact_torques``, as ``look.look_inverse_dynamics``; a rolling foot's CoP margin is NaN).  A plan
    without a rolling frame is ``transitions.idyn`` itself."""
    foot, k0 = _registered(swing)
    K = len(qs)
    rolling = foot["rolling"][k0 : k0 + K]
    if not rolling.any():
        return transitions.idyn(M, qs, qd, qdd, swing, split)
    world = foot["contact_world"]
    tau = np.zeros((K, len(M.act_dofs)))
    cop_margin = np.full((K, 2), np.nan)
    normal = np.zeros((K, 2))
    friction = np.zeros(K)
    failed = []
    for k in range(K):
        sup = tuple(s for j, s in enumerate(SIDES) if not swing[k, j]) or SIDES
        rol = {s: world[k0 + k, j] for j, s in enumerate(SIDES) if s in sup and rolling[k, j]}
        if rol and len(sup) == 2:
            tk, wr, ok = look._rolling_contact_torques(M, qs[k], qd[k], qdd[k], sup, rol)
            if not ok:
                failed.append(k)
        elif split == "exact":
            tk, wr, _ = turning.contact_torques(M, qs[k], qd[k], qdd[k], sup, split="exact")
        else:
            tk, wr = statics.contact_torques(M, qs[k], qd[k], qdd[k], support=sup)
        tau[k] = tk
        for j, side in enumerate(SIDES):
            if side in wr:
                f = wr[side][:3]
                normal[k, j] = f[2]
                friction[k] = max(friction[k], float(np.linalg.norm(f[:2]) / max(f[2], 1e-6)))
                if side in rol:
                    continue
                c = statics.cop(M, side, wr[side])
                M.forward(qs[k])
                cop_margin[k, j] = statics.support_margin(c, M.pad_polygon(side))
    foot["split_failed_frames"] = failed
    return tau, cop_margin, normal, friction


def _is_view(ref) -> bool:
    return isinstance(ref, View)


def initial_state(pl: transitions.Plan, ref_a, geo: transitions.Geometry) -> np.ndarray:
    """(c, c', c'') per axis (3, 2) at the first computed frame: Stage 1's for a stand or a reference of one stride;
    for a window, its LIPM path at that phase, read spectrally (J1, J2)."""
    if not _is_view(ref_a):
        return transitions.initial_state(pl, ref_a, geo)
    x0 = ref_a.window.com_state(float(ref_a.phi(pl.phase[0])))
    x0[0] += ref_a.shift()
    return x0


def _world_contacts(pl: transitions.Plan) -> None:
    """Fill the plan's rolling contacts in the world (once per plan): the planned flat foot's origin and yaw."""
    foot = (getattr(pl, "extras", None) or {}).get("foot")
    if foot is None or "contact_world" in foot:
        return
    world = np.full(foot["contact_foot"].shape, np.nan)
    for k, j in zip(*np.nonzero(foot["rolling"]), strict=True):
        world[k, j] = pl.foot_pos[k, j][None, :] + foot["contact_foot"][k, j] @ turning._rotz(pl.foot_yaw[k, j]).T
    foot["contact_world"] = world


def walk_state(ref, phase0: float, first: float) -> np.ndarray:
    """``moves.walk_state`` for a window: its LIPM path's state at move phase ``phase0``."""
    if not _is_view(ref):
        return moves.walk_state(ref, phase0, first)
    x0 = ref.window.com_state(float(ref.phi(phase0)))
    x0[0] += ref.shift()
    return x0


def walk_frames(ref, phases, first: float, key: str) -> np.ndarray:
    """``moves.walk_frames`` for a window: ``key`` resampled on the move's phases, in the move's plan."""
    if not _is_view(ref):
        return moves.walk_frames(ref, phases, first, key)
    return ref.frames(key, np.asarray(phases, float))


def walk_forward_acceleration(ref, phases) -> np.ndarray:
    """``moves.walk_forward_acceleration`` for a window (what a track plays before a move that leaves it and after
    one that joins it: the history of the move's 1 Hz acceleration reading): the model CoM's acceleration along the
    walk's heading (a straight walk: x) at the move's phases, the second derivative of the window's periodic cubic
    spline of ``com`` in time."""
    if not _is_view(ref):
        return moves.walk_forward_acceleration(ref, phases)
    w = ref.window
    cs, _adv, _shape = w._spline("com")
    t = np.atleast_1d(w.time(ref.phi(np.asarray(phases, float))))
    return np.asarray(cs(np.mod(t, w.W), 2))[:, 0]


class _Correction:
    """A window's ZMP correction as ``moves._causal_setup`` reads it.  ``moves`` indexes the walk's correction by the
    frame-of-stride of each computed frame (``delta[kk]``), which repeats one stride's correction; a window's varies
    from stride to stride (by centimetres with F14's tail and step-time variation).  This object answers the same
    indexing by position: the i-th index is the computed frame at move phase ``p_first + i / n`` (both the commanded
    plan and the continuing walk are computed from one stride before the command), and the value is the window's own
    correction at that phase.  An index that is not those frames' is refused."""

    shape = (N, 2)

    def __init__(self, view: "View", omega: float, p_first: float):
        self.view, self.omega, self.p_first = view, float(omega), float(p_first)

    def __getitem__(self, kk):
        kk = np.asarray(kk)
        p = self.p_first + np.arange(len(kk)) / N
        if kk.ndim != 1 or not np.array_equal(np.round(p * N).astype(int) % N, kk):
            raise LookError("the window's ZMP correction is read by the computed frames of a causal move, from one "
                            "stride before its command (moves._causal_setup)")
        return self.view.window.correction_at(self.omega, self.view.phi(p))


def walk_correction(ref, omega: float):
    """``moves.walk_correction`` for a window: its own correction at every computed frame of the causal move
    (``_Correction``; J1)."""
    if not _is_view(ref):
        return moves.walk_correction(ref, omega)
    if ref.command is None:
        raise LookError("walk_correction: this view is not a causal move's (it has no command phase)")
    return _Correction(ref, omega, ref.command - 1.0)


def compare(g: dict, ref, side: str, q_home=None) -> dict:
    """``transitions.compare`` (same keys); for a window, the move's first ("a") or last ("b") output stride against
    the window's stride at the same phases, resampled on the move's phase grid (J1, J2)."""
    if not _is_view(ref):
        return transitions.compare(g, ref, side, q_home)
    pl = g["plan"]
    k_out = np.arange(pl.n_out)
    if side == "a":
        sel, stride_phase, join = k_out[:N], float(pl.first), 0
    else:
        sel, stride_phase, join = k_out[-N:], float(pl.last - 1), N - 1
    ph = pl.phase[pl.out][sel]
    w = ref.window
    # the window's frames in its own coordinates, carried onto the plan's stride that starts at stride_phase
    P, th = pl.pose_fn(stride_phase)
    origin = g["geo"].origin + np.array([w.stride * float(ref.phi(stride_phase)), 0.0])
    mot = (th, origin, P)
    com_w = w.at("com", ref.phi(ph))
    com_ref = np.column_stack([transitions._move(com_w[:, :2], mot), com_w[:, 2]])
    zmp_ref = transitions._move(w.at("zmp_mb", ref.phi(ph)), mot)
    q_ref, qd_ref, tau_ref = w.at("q", ref.phi(ph))[:, 7:], w.at("qd", ref.phi(ph))[:, 6:], w.at("tau", ref.phi(ph))
    com, zmb = g["com"][sel], g["zmp_mb"][sel]
    qj, qdj, tau = g["q"][sel][:, 7:], g["qd"][sel][:, 6:], g["tau"][sel]
    dc = 1e3 * np.linalg.norm(com - com_ref, axis=1)
    dz = 1e3 * np.linalg.norm(zmb - zmp_ref, axis=1)
    dq, dqd, dt_ = np.abs(qj - q_ref).max(1), np.abs(qdj - qd_ref).max(1), np.abs(tau - tau_ref).max(1)
    out = dict(stride_phase=stride_phase, reference=ref.name,
               com_max_mm=float(np.max(dc)), com_mean_mm=float(np.mean(dc)), zmp_max_mm=float(np.nanmax(dz)),
               zmp_mean_mm=float(np.nanmean(dz)), joint_max_rad=float(np.max(dq)), joint_rate_max_rad_s=float(np.max(dqd)),
               at_join=dict(frame="first output frame" if side == "a" else "last output frame", com_mm=float(dc[join]),
                            zmp_mm=float(dz[join]), joint_rad=float(dq[join]), joint_rate_rad_s=float(dqd[join]),
                            tau_Nm=float(dt_[join])),
               tau_max_Nm=float(np.max(dt_)), window=ref.describe(),
               window_phase=[float(ref.phi(ph[0])), float(ref.phi(ph[-1]))])
    out["com_ok"], out["zmp_ok"] = out["com_max_mm"] <= transitions.G3_COM_MM, out["zmp_max_mm"] <= transitions.G3_ZMP_MM
    return out


def arrays_extra(g: dict) -> dict:
    """The look keys of a move's file, appended after ``transitions.arrays``' keys: the planned foot pitch (rad,
    heel-up positive), which stance feet roll and their floor contact (two end points, world; NaN elsewhere), the
    trunk roll, the pelvis dip (m), the variation envelopes of windows a and b, and the plan's period factor."""
    pl = g["plan"]
    o = pl.out
    ex = pl.extras
    foot = ex["foot"]
    _world_contacts(pl)
    n = pl.n_out
    roll = ex["trunk_roll"]
    t = pl.t
    factor = np.gradient(t, pl.phase) / np.asarray(pl.ctx.fn["T"](pl.phase), float)
    return dict(foot_pitch=foot["pitch"][o], rolling=foot["rolling"][o], rolling_contact=foot["contact_world"][o],
                trunk_roll=np.zeros(n) if roll is None else np.asarray(roll, float)[o],
                bob=np.zeros(n) if foot["bob"] is None else (pl.ctx.amp * foot["bob"])[o],
                var_env=foot["var_env"][o], period_factor=factor[o])


# ------------------------------------------------------------------------------------------------
# A periodic turn cut from a long plan (the turn in place)
# ------------------------------------------------------------------------------------------------
def hold_move(src: dict, name: str, b: moves.GaitX, hold_strides: int = 4) -> Bound:
    """Stand to gait ``b`` (a turn in place or a walking turn of the look's laws) with ``hold_strides`` more strides
    of steady gait than Stage 1's start writes: the long plan a periodic stride is cut from."""
    mv = moves.start(name, b)
    sch = mv.schedule
    mv = replace(mv, schedule=replace(sch, window=(sch.window[0], int(sch.window[1]) + int(hold_strides))),
                 args=dict(mv.args, hold_strides=int(hold_strides)))
    return bind_move(src, mv, None, None)


def _map_stride(pl: transitions.Plan, p_from: float, p_to: float):
    """The rigid motion that carries the plan's path pose at phase ``p_from`` onto its pose at ``p_to``."""
    P0, th0 = pl.pose_fn(p_from)
    P1, th1 = pl.pose_fn(p_to)
    return (th1 - th0, P0, P1)


def periodicity(g: dict, stride_from: float) -> dict:
    """The output stride that starts at phase ``stride_from`` against the strides before and after it, each mapped
    onto it by the plan's own stride motion: model CoM (mm), multibody ZMP (mm), joints (rad)."""
    pl = g["plan"]
    ph = pl.phase[pl.out]
    k0 = int(round((stride_from - ph[0]) * N))
    out = {}
    worst = dict(com_mm=0.0, zmp_mm=0.0, joint_rad=0.0)
    for label, d in (("before", -1), ("after", 1)):
        ka = k0 + d * N
        if ka < 0 or ka + N > len(ph):
            raise LookError(f"periodicity: no output stride {label} phase {stride_from:g}")
        mot = _map_stride(pl, stride_from + d, stride_from)
        com = np.column_stack([transitions._move(g["com"][ka : ka + N, :2], mot), g["com"][ka : ka + N, 2]])
        zmp = transitions._move(g["zmp_mb"][ka : ka + N], mot)
        r = dict(com_mm=float(1e3 * np.max(np.linalg.norm(com - g["com"][k0 : k0 + N], axis=1))),
                 zmp_mm=float(1e3 * np.nanmax(np.linalg.norm(zmp - g["zmp_mb"][k0 : k0 + N], axis=1))),
                 joint_rad=float(np.max(np.abs(g["q"][ka : ka + N, 7:] - g["q"][k0 : k0 + N, 7:]))))
        out[label] = r
        for key in worst:
            worst[key] = max(worst[key], r[key])
    ok = bool(worst["com_mm"] <= PERIODIC_COM_MM and worst["zmp_mm"] <= PERIODIC_ZMP_MM
              and worst["joint_rad"] <= PERIODIC_JOINT_RAD)
    return dict(out, stride_phase=[float(stride_from), float(stride_from + 1)], **worst,
                limits=dict(com_mm=PERIODIC_COM_MM, zmp_mm=PERIODIC_ZMP_MM, joint_rad=PERIODIC_JOINT_RAD),
                rule=RULES["S2L-periodic"], ok=ok)


def periodic_cut(src: dict, g: dict, stride_from: float) -> dict:
    """One output stride of a long plan as a periodic reference in Stage 1's turn-reference format (``refs.arrays``'
    keys with ``path``, ``heading``, ``foot_target``, ``foot_yaw``, ``swing``), then ``phase`` and the look keys of
    ``arrays_extra``.  The stride is carried into a reference's own frame, as Stage 1's turn references are: its path
    frame at its first frame is the home origin (the hip-pitch midpoint) with heading 0; ``stride_motion`` is the
    plan's own motion from the stride's first frame to the next stride's, in that frame.  Time and phase restart at
    the stride's first frame."""
    pl = g["plan"]
    ph = pl.phase[pl.out]
    k0 = int(round((stride_from - ph[0]) * N))
    s = slice(k0, k0 + N)
    full = moves.arrays(g)
    t = full["t"][s] - full["t"][k0]
    period = float(full["t"][k0 + N] - full["t"][k0])
    if not np.allclose(np.diff(t), period / N, rtol=1e-9, atol=1e-12):
        raise LookError("periodic_cut: the stride's frames are not uniform in time")
    P0, th0 = pl.pose_fn(stride_from)
    P1, th1 = pl.pose_fn(stride_from + 1.0)
    home = np.asarray(g["geo"].origin, float)
    mot = (-th0, P0, home)  # the plan's frame at the stride's first frame onto the reference's own

    def xy(a):
        a = np.array(a, float)
        a[..., :2] = transitions._move(a[..., :2], mot)
        return a

    stride = (float(th1 - th0), home, transitions._move(P1, mot))
    out = dict(t=t, q=np.array([turning.move_pose(q, *mot) for q in full["q"][s]]),
               qd=np.array([turning.move_velocity(v, mot[0]) for v in full["qd"][s]]),
               qdd=np.array([turning.move_velocity(v, mot[0]) for v in full["qdd"][s]]), tau=full["tau"][s],
               omega=full["omega"][s], support=full["support"][s], com_ref=xy(full["com_ref"][s]),
               zmp_ref=xy(full["zmp_ref"][s]), zmp_mb=xy(full["zmp_mb"][s]), cop_margin=full["cop_margin"][s],
               normal=full["normal"][s], friction=full["friction"][s], feet=xy(full["feet"][s]), com=xy(full["com"][s]),
               q_home=full["q_home"], joints=full["joints"], dt=np.array(period / N),
               stride_motion=np.array([stride[0], *map(float, stride[1]), *map(float, stride[2])]),
               path=xy(full["path"][s]), heading=full["heading"][s] - th0, foot_target=xy(full["foot_target"][s]),
               foot_yaw=full["foot_yaw"][s] - th0, swing=full["swing"][s], phase=full["phase"][s] - stride_from)
    for key in ("swing_u", "foot_pitch", "rolling", "trunk_roll", "bob"):
        out[key] = full[key][s]
    return out


def spec_plain(spec: gait.GaitSpec) -> dict:
    return refs._plain(asdict(spec))


# ------------------------------------------------------------------------------------------------
# Readings beside the join gates (reported, never gated)
# ------------------------------------------------------------------------------------------------
#: Frames this near a planned lift-off or touchdown, or a rolling frame, count as "at an event" in ``join_events``.
EVENT_FRAMES = 3


def _near_events(ph, events, rolling=None) -> np.ndarray:
    """``join_events``' frames "at an event": within ``EVENT_FRAMES`` frames (in phase) of a planned lift-off or
    touchdown (``events``: phases), or of a frame of the same warm-up on which a stance foot rolls (``rolling``: one
    flag per frame of ``ph``).  A frame beyond either end of the warm-up is no frame: nothing wraps round."""
    ph = np.asarray(ph, float)
    ev = np.asarray(events, float)
    near = (np.min(np.abs(ph[:, None] - ev[None, :]), axis=1) <= (EVENT_FRAMES + 0.5) / N if ev.size
            else np.zeros(len(ph), bool))
    if rolling is not None:
        pad = np.zeros(len(ph) + 2 * EVENT_FRAMES, bool)
        pad[EVENT_FRAMES : EVENT_FRAMES + len(ph)] = np.asarray(rolling, bool)
        for d in range(2 * EVENT_FRAMES + 1):  # frame i is near a rolling frame i - EVENT_FRAMES ... i + EVENT_FRAMES
            near |= pad[d : d + len(ph)]
    return near


def join_events(g: dict, ref_a) -> dict | None:
    """A causal move's warm-up multibody ZMP against the window's (``moves.join_command``'s own difference), read
    apart from the plan's events.  A window's frames are uniform in time and the move's in phase, so near a
    lift-off, a touchdown or a toe-off roll (features a few frames long) the two are different samplings of the same
    motion, and their second differences differ by millimetres whatever the join.  A reading for the record, on the
    gate's frames and against the gate's limit (``moves.WARMUP_ZMP_LIMIT_MM``); the gate is ``join_command``'s."""
    if not g.get("causal") or not _is_view(ref_a):
        return None
    w, pl = g["warmup"], g["plan"]
    ph = np.asarray(w["phase"], float)
    dz = 1e3 * np.linalg.norm(w["zmp_mb"] - walk_frames(ref_a, ph, float(pl.first), "zmp_mb"), axis=1)
    foot = pl.extras.get("foot")
    near = _near_events(ph, sorted({float(x) for s in pl.swings for x in (s[1], s[2])}),
                        None if foot is None else foot["rolling"][: len(ph)].any(axis=1))
    gated = np.arange(len(ph)) < len(ph) - 2  # the gate's frames: up to two frames before the command
    ok = gated & np.isfinite(dz)
    over = ok & (dz > moves.WARMUP_ZMP_LIMIT_MM)
    away = ok & ~near
    return dict(note="a reading, not a gate: the gate is join_command's warmup_zmp_max_mm",
                frames=int(ok.sum()), zmp_max_mm=float(np.max(dz[ok])), zmp_p50_mm=float(np.percentile(dz[ok], 50)),
                zmp_p90_mm=float(np.percentile(dz[ok], 90)), limit_mm=moves.WARMUP_ZMP_LIMIT_MM,
                frames_over_limit=int(over.sum()), phases_over_limit=[float(x) for x in ph[over]],
                frames_over_limit_away_from_events=int((over & ~near).sum()),
                event_frames=int((ok & near).sum()), zmp_max_mm_away_from_events=float(np.max(dz[away])) if away.any() else None,
                events_rule=f"within {EVENT_FRAMES} frames of a planned lift-off or touchdown, or of a frame of the "
                            "warm-up on which a stance foot rolls (no frame beyond either end of the warm-up)")
