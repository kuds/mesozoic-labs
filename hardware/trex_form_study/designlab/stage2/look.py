"""Stage 2 Track A: the look ladder (``STAGE2_PLAN.md`` section 4 A; ``stage2/research/research_look.md`` 4.1).

Each body's reference walk is rebuilt from its record exactly as Stage 1's references are (``stage1.refs.source``),
then generated at round two's accuracy (``gait.set_accuracy("r2")``) with ``gait.GaitSpec``'s look fields added one
feature group at a time, so the owner sees what each feature buys (look gate L1).  Simulation only: a prescribed
reference with perfect tracking, priced by the Stage 0 screen before any render; nothing is adopted (P24).

Bodies and speeds (``BODIES``, ``SPEEDS``)
-----------------------------------------
* ``60-N-T-10`` (60 cm Tapered) and ``75-M-H-10`` (75 cm Hybrid), current heads (neck pitch, tail yaw).
* ``nominal``: ``sizing.NOMINAL`` (Froude 0.012).  ``fr05``: ``envelope.FAST`` (Froude 0.05, the animal pace).
* ``v040``: 0.40 m/s, the agility bar's speed (``speed_spec``).  Its stride, duty and lift are the gait that Stage 1's
  ``brisk_to_fr05`` speed change passes through at 0.40 m/s: ``transitions.py`` mixes the period, duty and lift of the
  two walks with the same weight as their speeds, so w = (0.40 - v_brisk) / (v_fr05 - v_brisk) and T, duty and lift
  are mixed with w (60 cm: w 0.651, T 0.751 s, stride 0.742 H, duty 0.567; 75 cm: w 0.338, T 0.861 s, stride
  0.686 H, duty 0.573; lift 0.11 H, both walks' lift).  It is a reference the study already passes through, between
  two screened walks.

The ladder (``VARIANTS``; research 4.1, one feature group per rung)
------------------------------------------------------------------
V0 today's walk; V1 + F1/F2 (bob, weight acceptance); V2 + F3 (swing time); V3 + F4/F5 (lift shape, toe-off);
V4 + F6 (trunk roll); V5 + F9/F10 (tail phase, stabilised head); V6 + F14 (variation), and at 75 cm F13 (a longer
stride at the animal pace).  The look rungs run 12 multibody-ZMP correction passes (``LOOK_ZMP_ITERS``; today's 6):
the bob and the toe-off slow the correction's convergence, and 12 passes bring it back to about today's error.  Extras (``EXTRAS``): V4L, V4 with the roll's other direction (the swing side drops, as
human pelvic list; the direction is the owner's call, research F6), and V6S, V6 with F7 (the single-support ZMP
moved inward; sway reduction, optional).  F8 (no trunk pitch) is today's; F11 and F12 are deferred.

Values (``feature_values``; research 3.1's starting values, the middle of each range, recorded in ``VALUES``)
--------------------------------------------------------------------------------------------------------
* F1 ``bob``: 75 % (the middle of 50-100 %) of the stance-leg shortening the flat walk enforces,
  (duty stride / 2)^2 / (2 (1 - crouch) H), with V0's duty and stride at that speed (0.0140 H nominal and 0.0259 H at
  the animal pace, the research's 0.014 and 0.026 H), so the bob stays the same from V1 to V6 at a speed.
* F2 ``bob_skew`` 0.125 of stance (10-15 %).  F3 ``swing_time``: the animal-pace walk's swing (1 - 0.56) T_fr05, 0.320 s
  at 60 cm and 0.356 s at 75 cm (the research's 0.32 / 0.36 s).  F4 ``lift_peak`` 0.375 (35-40 %).
* F5 ``toe_off`` 0.40 rad (0.3-0.5), ``toe_off_start`` 0.825 (80-85 %), ``toe_reach`` 0.075 rad (0.05-0.10); the roll
  also waits until the ZMP reference is 5 mm inside the rolling support (``toe_off_margin``, not in the research: a
  rigid foot on its toe edge carries load only there), and its peak moves into early swing where the stance part is
  short (``gait.toe_off_bump``).  Two consequences, measured per variant and kept as the research's values: (1) the
  roll can start only in double support once the ZMP reference is inside the toe-edge support, so with F3's short
  double support at 0.40 m/s and the animal pace most of the 0.40 rad comes after the lift-off (``measure``'s
  ``toe_off_at_lift_off_rad``, ``toe_off_ground_roll_s``, ``toe_off_peak_after_lift_off_s``; a ground toe-off there
  would need the single-support ZMP moved toward the toes, which the research does not propose); (2) the middle toe's
  capsule and claw reach 10-27 mm ahead of its pad's front edge and carry no contact in the plant, so a roll about
  that edge takes them below the floor (``toe_clearance``: clear to 0.082 rad at 60 cm and 0.068 rad at 75 cm; at
  0.40 rad the claw reaches 7.3 / 9.4 mm below; ``measure``'s ``toe_geoms_below_floor_max_mm``).
* F6 ``roll_amp`` 0.0262 rad (1.5 deg, 1-2 deg), banking toward the stance leg (V4L: the swing side drops).
* F9 ``tail_amp`` 0.125 rad (0.10-0.15) at the phase that minimises the stance pad's peak yaw moment on V4
  (``tail_check``); F10 ``head_mode`` "stabilise", ``head_gain`` 0.9 (0.8-1.0), ``head_settle`` 0.015 rad (0.01-0.02)
  at ``settle_lag`` 0.075 s (0.05-0.1).
* F13 ``stride_rel`` 0.875 H (0.85-0.90) at the animal pace, 75 cm only.  F7 ``zmp_inset_y`` 10 mm (8-12).
* F14: ``phase_jitter`` 1.5 % (1-2 %), ``asym`` 2.5 % (2-3 %), ``var_band`` 0.05-0.2 Hz, ``gaze_var`` 0.015 rad (the
  research gives no size for walking gaze drift: the idle micro-motion's 0.01-0.02 rad), ``tail_var`` 0.2 (the tail
  amplitude drifts over F9's 0.10-0.15 rad), ``noise_seed`` 2026, and a window of whole strides at least ``CLIP_S``
  (the renders' 10 s clip) long, periodic as a whole.

Outputs (``prepare``): ``<out>/refs/<body>/<speed>/<variant>.npz`` (``arrays``: frames, joint trajectories, timing and
the per-frame quantities the screen and the renders read; git-ignored like Stage 1's arrays), ``<out>/look_refs.json``
(specs, timing, convergence, margins, measured metrics, each file's sha256; ``body_info``: foot geometry, markers and
``toe_clearance``) and ``<out>/tail_phase.json`` (``tail_check``: research Bottom line 4's sign check on V0, against
the turning screen's recorded straight run at the nominal walk, and F9's phase on V4).  Both are updated as each body
and speed finishes; ``<out>/partial/`` holds a job's progress until then (``--resume`` keeps it after a restart).

Run from hardware/trex_form_study:
    python -m designlab.stage2.look prepare --body 60-N-T-10 --body 75-M-H-10 --out stage2/look --procs 3 [--resume]
    python -m designlab.stage2.look g2 --body 60-N-T-10     # Stage 1's nominal, brisk and fr05 arrays' sha256
"""

from __future__ import annotations

import os

if __name__ == "__main__":  # one core per process, as the Stage 1 runner (the arrays are then bit-reproducible)
    for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from contextlib import contextmanager  # noqa: E402
from dataclasses import asdict, replace  # noqa: E402
from pathlib import Path  # noqa: E402

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from .. import carpet, envelope, gait, sizing, statics, turning  # noqa: E402

G = gait.G
SIDES = statics.SIDES
STUDY = Path(__file__).resolve().parents[2]
OUT = STUDY / "stage2" / "look"
ACCURACY = "r2"
BODIES = ("60-N-T-10", "75-M-H-10")
SPEEDS = ("nominal", "v040", "fr05")
#: The agility bar's walking speed (m/s; STAGE2_PLAN section 1).
AGILITY_SPEED = 0.40
VARIANTS = ("V0", "V1", "V2", "V3", "V4", "V5", "V6")
EXTRAS = ("V4L", "V6S")
#: F14's window: whole strides lasting at least the renders' clip (research 4.1: 10 s clips at 30 fps).
CLIP_S = 10.0
#: The F1 bob's share of the stance-leg shortening the flat walk enforces (research F1: screen 50 % and 100 %).
BOB_SHARE = 0.75
#: The multibody-ZMP correction passes of the look rungs (V1 on; today's walk, V0, keeps sizing's 6).  The bob's vertical
#: acceleration and the toe-off's foot pitch slow the correction's convergence: on 60-N-T-10's animal-pace V3 the error
#: after 6 passes is 3.2 mm (V0's 1.95 mm), after 12 about 1.9 mm, falling geometrically by about 0.93 a pass.
LOOK_ZMP_ITERS = 12
#: Research 3.1's ranges and the values taken (the middle where a range is given), with each rule.
VALUES = {
    "F1 bob": dict(range="50-100 % of the enforced stance-leg shortening (0.007-0.014 H nominal, 0.013-0.026 H at the "
                         "animal pace)", value=BOB_SHARE,
                   rule="bob = 0.75 (duty stride / 2)^2 / (2 (1 - crouch) H) with V0's duty and stride at that speed"),
    "F1 bob_phase": dict(range="hips at today's height at mid single support", value=0.0),
    "F2 bob_skew": dict(range="trough at 10-15 % of stance", value=0.125),
    "F3 swing_time": dict(range="the animal-pace walk's swing time, 0.32 s at 60 cm and 0.36 s at 75 cm", value=None,
                          rule="(1 - envelope.FAST.duty) x the Froude 0.05 walk's period on the body"),
    "F4 lift_peak": dict(range="35-40 % of the swing", value=0.375),
    "F5 toe_off": dict(range="0.3-0.5 rad by lift-off", value=0.40),
    "F5 toe_off_start": dict(range="80-85 % of stance", value=0.825),
    "F5 toe_reach": dict(range="0.05-0.10 rad", value=0.075),
    "F5 toe_off_margin": dict(range="not in the research", value=0.005,
                              rule="the roll waits until the ZMP reference is 5 mm inside the rolling support (the "
                                   "trailing toe pad's front edge and the leading foot's pads)"),
    "F6 roll_amp": dict(range="1-2 deg", value=math.radians(1.5),
                        rule="toward the stance leg at its mid-stance (V4); V4L the other way"),
    "F7 zmp_inset_y": dict(range="screen 8-12 mm", value=0.010, rule="optional extra V6S"),
    "F9 tail_amp": dict(range="0.10-0.15 rad", value=0.125),
    "F9 tail_phase": dict(range="the phase minimising the stance foot's yaw moment", value=None,
                          rule="tail_check on V4 per body and speed: the peak |M_z| of the single-support stance pad"),
    "F10 head_gain": dict(range="0.8-1.0", value=0.9),
    "F10 head_settle": dict(range="0.01-0.02 rad", value=0.015),
    "F10 settle_lag": dict(range="0.05-0.1 s", value=0.075),
    "F13 stride_rel": dict(range="0.85-0.90 H at the animal pace, 75 cm only", value=0.875),
    "F14 phase_jitter": dict(range="step time +-1-2 %", value=0.015),
    "F14 asym": dict(range="left-right asymmetry <= 2-3 %", value=0.025),
    "F14 var_band": dict(range="0.05-0.2 Hz", value=[0.05, 0.2]),
    "F14 gaze_var": dict(range="not sized in the research", value=0.015,
                         rule="the idle micro-motion's 0.01-0.02 rad (research 5.1), middle"),
    "F14 tail_var": dict(range="not sized in the research", value=0.2,
                         rule="the tail amplitude drifts over F9's 0.10-0.15 rad (0.125 +- 20 %)"),
    "F14 noise_seed": dict(range="any", value=2026),
    "zmp_iters": dict(range="sizing's 6 passes", value=LOOK_ZMP_ITERS,
                      rule="look rungs (V1 on) run 12 multibody-ZMP correction passes: the bob and the toe-off slow the "
                           "convergence (60-N-T-10 animal-pace V3: 3.2 mm after 6, 1.9 mm after 12; V0 1.95 mm after 6)"),
    "v040": dict(range="0.40 m/s", value=AGILITY_SPEED,
                 rule="the gait the brisk_to_fr05 speed change passes through at 0.40 m/s: period, duty and lift mixed "
                      "with the speed's weight (transitions.py)"),
}
#: The tail-phase check: the phases run (rad) and the grid the linear fit is minimised on.
TAIL_PHASES = (0.0, 0.5 * math.pi, math.pi, 1.5 * math.pi)
TAIL_GRID = 720
#: Point-light markers (research 4.1 step 3) the renders draw: name -> (body, how the point is found).
POINTS = ("snout", "eye", "neck_base", "hip_l", "hip_r", "knee_l", "knee_r", "mtp_l", "mtp_r", "toe_l", "toe_r",
          "tail_mid", "tail_tip")


@contextmanager
def accuracy(level: str = ACCURACY):
    """``gait.set_accuracy(level)`` for the block, restoring the process's level after (it is module state)."""
    prev = gait.ACCURACY
    gait.set_accuracy(level)
    try:
        yield
    finally:
        gait.set_accuracy(prev)


# ------------------------------------------------------------------------------------------------
# Specs
# ------------------------------------------------------------------------------------------------
def speed_spec(H: float, v: float = AGILITY_SPEED, lo: gait.GaitSpec = sizing.BRISK,
               hi: gait.GaitSpec = envelope.FAST) -> gait.GaitSpec:
    """The gait ``transitions.py``'s ``lo``-to-``hi`` speed change passes through at speed v (module docstring)."""
    a, b = lo.describe(H), hi.describe(H)
    w = (v - a["speed"]) / (b["speed"] - a["speed"])
    if not 0.0 <= w <= 1.0:
        raise ValueError(f"{v} m/s is outside {a['speed']:.3f}-{b['speed']:.3f} m/s")
    T = a["period"] + w * (b["period"] - a["period"])
    return replace(lo, froude=v * v / (G * H), stride_rel=v * T / H, duty=lo.duty + w * (hi.duty - lo.duty),
                   lift_frac=lo.lift_frac + w * (hi.lift_frac - lo.lift_frac))


def base_spec(H: float, speed: str) -> gait.GaitSpec:
    """V0 at a speed (``SPEEDS``)."""
    if speed == "nominal":
        return sizing.NOMINAL
    if speed == "fr05":
        return envelope.FAST
    if speed == "v040":
        return speed_spec(H)
    raise KeyError(speed)


def enforced_shortening(spec: gait.GaitSpec) -> float:
    """The stance leg's shortening (fraction of H) that a constant hip height enforces: (duty stride / 2)^2 /
    (2 (1 - crouch)), the hip passing over the foot at the walk's height (research section 1)."""
    return (spec.duty * spec.stride_rel / 2) ** 2 / (2 * (1 - spec.crouch))


def swing_time(H: float) -> float:
    """F3: the animal-pace walk's swing time on a body of hip height H (s)."""
    return float((1.0 - envelope.FAST.duty) * envelope.FAST.describe(H)["period"])


def feature_values(body: str, H: float, speed: str, tail_phase: float | None = None) -> dict:
    """The look fields each feature group sets at this body and speed (``VALUES``)."""
    v0 = base_spec(H, speed)
    out = {
        "F1F2": dict(bob=BOB_SHARE * enforced_shortening(v0), bob_skew=0.125),
        "F3": dict(swing_time=swing_time(H)),
        "F4F5": dict(lift_peak=0.375, toe_off=0.40, toe_off_start=0.825, toe_reach=0.075, toe_off_margin=0.005),
        "F6": dict(roll_amp=math.radians(1.5)),
        "F9F10": dict(tail_amp=0.125, head_mode="stabilise", head_gain=0.9, head_settle=0.015, settle_lag=0.075),
        "F14": dict(phase_jitter=0.015, asym=0.025, gaze_var=0.015, tail_var=0.2, var_band=(0.05, 0.2),
                    noise_seed=2026),
        "F13": dict(stride_rel=0.875) if (speed == "fr05" and body.startswith("75")) else {},
        "F7": dict(zmp_inset_y=0.010),
    }
    if tail_phase is not None:
        out["F9F10"]["tail_phase"] = float(tail_phase)
    return out


def window_strides(spec: gait.GaitSpec, H: float, clip_s: float = CLIP_S) -> int:
    """F14: the fewest whole strides lasting at least ``clip_s``."""
    return int(math.ceil(clip_s / spec.describe(H)["period"] - 1e-9))


def ladder(body: str, H: float, speed: str, tail_phase: float | None = None, extras: bool = True) -> dict:
    """{variant: GaitSpec} at one body and speed.  V5, V6 and V6S need F9's ``tail_phase`` (``tail_check``); without
    it they are left out."""
    f = feature_values(body, H, speed, tail_phase)
    s = {"V0": base_spec(H, speed)}
    s["V1"] = replace(s["V0"], **f["F1F2"], zmp_iters=LOOK_ZMP_ITERS)
    s["V2"] = replace(s["V1"], **f["F3"])
    s["V3"] = replace(s["V2"], **f["F4F5"])
    s["V4"] = replace(s["V3"], **f["F6"])
    if extras:
        s["V4L"] = replace(s["V3"], roll_amp=-f["F6"]["roll_amp"])
    if tail_phase is not None:
        s["V5"] = replace(s["V4"], **f["F9F10"])
        v6 = replace(s["V5"], **f["F14"], **f["F13"])
        s["V6"] = replace(v6, strides=window_strides(v6, H))
        if extras:
            s["V6S"] = replace(s["V6"], **f["F7"])
    return s


def features_of(variant: str) -> list[str]:
    """The research's feature numbers a ladder rung carries."""
    rungs = {"V0": [], "V1": ["F1", "F2"], "V2": ["F3"], "V3": ["F4", "F5"], "V4": ["F6"], "V5": ["F9", "F10"],
             "V6": ["F14", "F13 (75 cm, animal pace)"]}
    if variant == "V4L":
        return features_of("V3") + ["F6 (swing side drops)"]
    if variant == "V6S":
        return features_of("V6") + ["F7"]
    if variant not in VARIANTS:
        return []
    out = []
    for v in VARIANTS[: VARIANTS.index(variant) + 1]:
        out += rungs[v]
    return out


# ------------------------------------------------------------------------------------------------
# Generation
# ------------------------------------------------------------------------------------------------
def source(body: str) -> dict:
    """The body rebuilt from its record (``stage1.refs.source``: model checked against the record's xml_sha256,
    full-precision home pose) plus its foot geometry and point-light markers."""
    from ..stage1 import refs

    src = refs.source(body)
    src["M"].forward(src["q_home"])
    src["geometry"] = gait.foot_geometry(src["M"])
    src["markers"] = markers(src["M"])
    src["M"].forward(src["q_home"])
    return src


def toe_clearance(src: dict, side: str = "left", pitch: float = 0.40, iters: int = 30) -> dict:
    """F5 on the rigid foot at home: the foot pitched heel-up about its toe pivot (``gait.foot_geometry``; the other foot
    and the trunk at home, IK as the generator's) and the lowest point of each of its geoms that is not a floor pad
    (the toes, claws, hallux and plate: no contact in the plant, ``contype`` 0).  Returns the largest pitch at which
    every such geom stays at or above the floor (bisection) and, at ``pitch`` (F5's ``toe_off``), how far below the
    floor each one reaches (mm; negative: below).  In the plant these geoms pass through the floor; on the hardware
    they would touch it first and take the pivot from the pads."""
    M, q0, geo = src["M"], src["q_home"], src["geometry"]
    m, d = M.m, M.d
    M.forward(q0)
    base = {s: M.foot_pose(s)[0].copy() for s in SIDES}
    others = [g for g in carpet.foot_geoms(m)[side] if m.geom_contype[g] != 1]

    def lows(th: float) -> dict:
        R = gait._rot_y(th)
        tg = {s: (base[s].copy(), np.eye(3)) for s in SIDES}
        tg[side] = (base[side] + (np.eye(3) - R) @ geo[side]["toe_pivot"], R)
        q, ok = statics.leg_ik(M, q0.copy(), tg, iters=200)
        if not ok:
            raise RuntimeError(f"toe_clearance: IK failed at a pitch of {th} rad")
        M.forward(q)
        out = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g): float(carpet.lowest_point(m, d, g)[2]) for g in others}
        M.forward(q0)
        return out

    lo, hi = 0.0, float(pitch)
    if min(lows(hi).values()) >= 0.0:
        lo = hi
    else:
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if min(lows(mid).values()) >= 0.0 else (lo, mid)
    at = lows(pitch)
    worst = min(at, key=at.get)
    return dict(side=side, pivot=geo[side]["toe_pad"] + " front bottom edge", clear_pitch_rad=float(lo),
                at_pitch_rad=float(pitch), lowest_mm_at_pitch={k: 1e3 * v for k, v in sorted(at.items(), key=lambda kv: kv[1])},
                deepest=worst, deepest_mm=1e3 * at[worst])


def generate(src: dict, spec: gait.GaitSpec) -> dict:
    """One variant's trajectory at round two's accuracy, with the look per-frame quantities completed for today's
    walk too (its frames carry no ``look`` block)."""
    M, q0, H = src["M"], src["q_home"], src["H"]
    t0 = time.perf_counter()
    with accuracy(ACCURACY):
        tr = gait.generate(M, q0, H, spec)
        qd, qdd = gait.derivatives(M, tr["q"], tr["dt"], stride=tr["info"]["stride"] * tr["info"].get("strides", 1))
    if "look" not in tr:
        tr = _complete_v0(src, spec, tr)
    tr["qd"], tr["qdd"] = qd, qdd
    tr["seconds"] = time.perf_counter() - t0
    return tr


def _complete_v0(src: dict, spec: gait.GaitSpec, tr: dict) -> dict:
    """A default-field walk (``gait.generate``'s own path) with the look block its frames imply: phase t / T, flat
    feet, no bob or roll; its margins against the pads (``gait.support_margins``)."""
    M, q0 = src["M"], src["q_home"]
    info = tr["info"]
    T, duty, n = info["period"], info["duty"], len(tr["t"])
    phi = tr["t"] / T
    sw = 1.0 - duty
    swing = np.zeros((n, 2), bool)
    swing_u = np.full((n, 2), np.nan)
    for j, off in enumerate((0.0, 0.5)):
        u = np.mod(np.mod(phi, 1.0) - off, 1.0)
        swing[:, j] = u < sw
        swing_u[swing[:, j], j] = u[swing[:, j]] / sw
    neck_adr = M.m.jnt_qposadr[statics.jid(M.m, "neck_pitch")]
    tail_adr = M.m.jnt_qposadr[statics.jid(M.m, "tail_yaw")]
    margin, margin_ref, hulls = gait.support_margins(M, tr["q"], tr["phases"], tr["zmp_mb"], tr["zmp_ref"])
    feet = np.zeros((n, 2, 3))
    for k in range(n):
        M.forward(tr["q"][k])
        feet[k] = [M.foot_pose(s)[0] for s in SIDES]
    touchdowns = sorted(float((i + off + sw) * T) for off in (0.0, 0.5) for i in range(-1, 2))
    info = info | dict(strides=1, window_s=float(T), swing_time=float(sw * T), look=dict(gait.LOOK_DEFAULTS),
                       bob_m=0.0, bob_trough_step=0.5, step_deviations=[0.0, 0.0], gaze_freqs_hz=[], tail_freqs_hz=[],
                       rolling_entry=None, zmp_margin_min_m=float(np.min(margin)),
                       zmp_margin_ref_min_m=float(np.min(margin_ref)))
    look = dict(phase=phi, swing=swing, swing_u=swing_u, foot_pitch=np.zeros((n, 2)), rolling=np.zeros((n, 2), bool),
                foot_base=feet, foot_target=feet, bob=np.zeros(n), trunk_roll=np.zeros(n),
                trunk_quat=np.tile(q0[3:7], (n, 1)), neck=tr["q"][:, neck_adr].copy(), tail=tr["q"][:, tail_adr].copy(),
                head_settle=np.zeros(n), gaze_var=np.zeros(n), tail_var=np.zeros(n), zmp_margin=margin,
                zmp_ref_margin=margin_ref, support_hull=hulls, touchdowns=touchdowns, rolls={s: [] for s in SIDES},
                geometry=src["geometry"], warp=gait._Warp(T, 1))
    return dict(tr, info=info, look=look)


# ------------------------------------------------------------------------------------------------
# Per-frame quantities for the screen and the renders
# ------------------------------------------------------------------------------------------------
def markers(M: statics.Model) -> list[tuple[str, int, np.ndarray]]:
    """The point-light markers (``POINTS``) as (name, body id, point in the body's frame), found on the model at its
    current pose (call at home): the skull cover's foremost vertex (snout), an eye, the neck-pitch axis (neck base),
    the hip-pitch and knee axes, the foot bodies' origins (the ankle, the animal's MTP), the middle claw's tip, the
    tail cover's rearmost vertex (tail tip: over every ``cover_tail*`` mesh on the tail body, one cover on the Tapered
    tail, a root and segments on the Hybrid's) and the point midway from the tail-yaw axis to it (mid-tail)."""
    m, d = M.m, M.d

    def local(b, p):
        return d.xmat[b].reshape(3, 3).T @ (np.asarray(p) - d.xpos[b])

    def geom(name):
        g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name)
        if g < 0:
            raise KeyError(f"no geom {name} on the model")
        return g

    def vertices(g):
        mid = m.geom_dataid[g]
        a, k = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
        return d.geom_xpos[g] + m.mesh_vert[a : a + k] @ d.geom_xmat[g].reshape(3, 3).T

    head, tail = statics.bid(m, "head"), statics.bid(m, "tail")
    out = []
    skull = vertices(geom("cover_skull"))
    out.append(("snout", head, local(head, skull[np.argmax(skull[:, 0])])))
    out.append(("eye", head, local(head, d.geom_xpos[geom("hw_eye_0")])))
    jn = statics.jid(m, "neck_pitch")
    out.append(("neck_base", int(m.jnt_bodyid[jn]), m.jnt_pos[jn].copy()))
    for jname, tag in (("hip_pitch", "hip"), ("knee", "knee")):
        for side in SIDES:
            j = statics.jid(m, f"{side}_{jname}")
            out.append((f"{tag}_{side[0]}", int(m.jnt_bodyid[j]), m.jnt_pos[j].copy()))
    for side in SIDES:
        out.append((f"mtp_{side[0]}", M.foot[side], np.zeros(3)))
    for side in SIDES:
        g = geom(f"hw_{side}_claw_0")
        c, a = d.geom_xpos[g], d.geom_xmat[g].reshape(3, 3)[:, 2]
        ends = [c + m.geom_size[g][1] * a, c - m.geom_size[g][1] * a]
        out.append((f"toe_{side[0]}", M.foot[side], local(M.foot[side], max(ends, key=lambda e: e[0]))))
    covers = [g for g in range(m.ngeom) if m.geom_bodyid[g] == tail and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
              and (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith("cover_tail")]
    if not covers:
        raise KeyError("no cover_tail* mesh on the tail body")
    tip = np.vstack([vertices(g) for g in covers])
    tip = tip[np.argmin(tip[:, 0])]
    out.append(("tail_mid", tail, local(tail, 0.5 * (tip + d.xanchor[statics.jid(m, "tail_yaw")]))))
    out.append(("tail_tip", tail, local(tail, tip)))
    assert [o[0] for o in out] == list(POINTS)
    return out


def _foot_pitch(R: np.ndarray) -> float:
    """Heel-up pitch of a foot about its lateral axis (rad): atan2(-R[2, 0], R[0, 0])."""
    return float(math.atan2(-R[2, 0], R[0, 0]))


def arrays(src: dict, g: dict) -> dict:
    """The arrays of one variant for its npz file.  Beside Stage 1's reference keys (t, dt, q, qd, qdd, support,
    com_ref, zmp_ref, zmp_mb, feet, com, q_home, joints, stride_motion): the gait phase (strides, warped by F14), swing
    flags and swing fraction, the target and solved foot pitch, which feet roll on the toe edge, the toe pivot's
    world position and its edge's two end points (``toe_edge``: a rolling foot's whole support), foot orientations,
    the support hull (NaN-padded) and the ZMP's margin inside it, the hip midpoint (the hip-roll axes', fixed on the
    trunk), bob, trunk roll and orientation, neck and tail angles (and F10's settle and F14's variations), the head's
    pitch in the world, the lowest pad and toe/claw points of each foot, the point-light markers and the events."""
    M, tr, lk = src["M"], g, g["look"]
    m, d = M.m, M.d
    n = len(tr["q"])
    feet = np.zeros((n, 2, 3))
    foot_quat = np.zeros((n, 2, 4))
    pitch_solved = np.zeros((n, 2))
    pivot = np.zeros((n, 2, 3))
    edge = np.zeros((n, 2, 2, 3))
    com = np.zeros((n, 3))
    hip_mid = np.zeros((n, 3))
    head_pitch = np.zeros(n)
    pad_z = np.zeros((n, 2))
    toe_z = np.zeros((n, 2))
    pts = np.zeros((n, len(POINTS), 3))
    geoms = carpet.foot_geoms(m)
    hips = [statics.jid(m, f"{s}_hip_roll") for s in SIDES]  # on the trunk: the hip-pitch axes move with hip roll
    head = statics.bid(m, "head")
    geo = src["geometry"]
    for k in range(n):
        M.forward(tr["q"][k])
        for j, s in enumerate(SIDES):
            p, R = M.foot_pose(s)
            feet[k, j] = p
            mujoco.mju_mat2Quat(foot_quat[k, j], R.ravel())
            pitch_solved[k, j] = _foot_pitch(R)
            pivot[k, j] = p + R @ geo[s]["toe_pivot"]
            edge[k, j] = p + geo[s]["toe_edge"] @ R.T
            low = {gg: carpet.lowest_point(m, d, gg)[2] for gg in geoms[s]}
            pad_z[k, j] = min(z for gg, z in low.items() if m.geom_contype[gg] == 1)
            toe_z[k, j] = min((z for gg, z in low.items() if carpet._is_toe(m, gg)), default=np.nan)
        com[k] = M.com()
        hip_mid[k] = 0.5 * (d.xanchor[hips[0]] + d.xanchor[hips[1]])
        head_pitch[k] = gait._nose_up(d.xquat[head])
        for i, (_, b, loc) in enumerate(src["markers"]):
            pts[k, i] = d.xpos[b] + d.xmat[b].reshape(3, 3) @ loc
    hmax = max(len(h) for h in lk["support_hull"])
    hull = np.full((n, hmax, 2), np.nan)
    for k, h in enumerate(lk["support_hull"]):
        hull[k, : len(h)] = h
    info = tr["info"]
    K = int(info.get("strides", 1))
    ev = events(lk["phase"], tr["t"], info["duty"], K)
    return dict(t=tr["t"], dt=np.array(float(tr["dt"])), q=tr["q"], qd=tr["qd"], qdd=tr["qdd"],
                support=np.array([[s in ph for s in SIDES] for ph in tr["phases"]]), com_ref=tr["com_ref"],
                zmp_ref=tr["zmp_ref"], zmp_mb=tr["zmp_mb"], feet=feet, com=com, q_home=src["q_home"],
                joints=np.array(M.names), stride_motion=np.array([0.0, 0.0, 0.0, K * float(info["stride"]), 0.0]),
                phase=lk["phase"], swing=lk["swing"], swing_u=lk["swing_u"], foot_pitch=lk["foot_pitch"],
                foot_pitch_solved=pitch_solved, rolling=lk["rolling"], toe_pivot=pivot, toe_edge=edge, foot_quat=foot_quat,
                foot_target=lk["foot_target"], support_hull=hull, zmp_margin=lk["zmp_margin"],
                zmp_ref_margin=lk["zmp_ref_margin"], hip_mid=hip_mid, bob=lk["bob"], trunk_roll=lk["trunk_roll"],
                trunk_quat=lk["trunk_quat"], neck=lk["neck"], tail=lk["tail"], head_settle=lk["head_settle"],
                gaze_var=lk["gaze_var"], tail_var=lk["tail_var"], head_pitch_world=head_pitch, pad_floor_z=pad_z,
                toe_floor_z=toe_z, points=pts, point_names=np.array(POINTS),
                lift_off_t=ev["lift_off"], touchdown_t=ev["touchdown"])


def events(phase, t, duty: float, K: int) -> dict:
    """Lift-off and touchdown times (s) of each foot over the window (left then right; the left foot lifts at integer
    phases, the right at half-integers; the left lands at phase 1 - duty), from the frames' phase by interpolation."""
    sw = 1.0 - duty
    out = {}
    for key, shift in (("lift_off", 0.0), ("touchdown", sw)):
        rows = []
        for off in (0.0, 0.5):
            f = np.arange(K) + off + shift
            rows.append(np.interp(f, phase, t, left=np.nan, right=np.nan))
        out[key] = np.array(rows)
    return out


def trajectory(arrs: dict, info: dict) -> dict:
    """A ``gait.generate``-style trajectory dict from a variant's arrays and its info (for ``gait.inverse_dynamics``,
    ``clearance.screen`` and ``turning_traj``): t, q, phases (support sets), dt, the ZMP and CoM references, info."""
    sup = np.asarray(arrs["support"], bool)
    phases = [tuple(s for j, s in enumerate(SIDES) if sup[k, j]) for k in range(len(sup))]
    return dict(t=np.asarray(arrs["t"]), q=np.asarray(arrs["q"]), phases=phases, dt=float(arrs["dt"]),
                zmp_ref=np.asarray(arrs["zmp_ref"]), com_ref=np.asarray(arrs["com_ref"]), zmp_mb=np.asarray(arrs["zmp_mb"]),
                info=dict(info), ik_ok=True, ik_fail_frames=[])


def turning_traj(traj: dict, M: statics.Model, q_home) -> dict:
    """The trajectory as ``turning.inverse_dynamics`` reads a straight run (yaw rate 0): the path frame from the
    hip-pitch midpoint at home, the stride motion of the window and the exact unwrap."""
    info = traj["info"]
    M.forward(q_home)
    origin = 0.5 * sum(M.d.xanchor[statics.jid(M.m, f"{s}_hip_pitch")][:2] for s in SIDES)
    adv = float(info["stride"]) * int(info.get("strides", 1))
    path = origin[None, :] + np.column_stack([info["speed"] * traj["t"], np.zeros(len(traj["t"]))])
    return dict(traj, transform=(0.0, origin.copy(), origin + np.array([adv, 0.0])), path=path,
                heading=np.zeros(len(traj["t"])),
                info=dict(info, yaw_rate=0.0, unwrap="exact", origin=origin.tolist()))


# ------------------------------------------------------------------------------------------------
# Measured look metrics (research 4.1 step 5), from a variant's arrays
# ------------------------------------------------------------------------------------------------
def _sub_frame(y, k: int, dt: float) -> float:
    """Time offset (s) of an extreme at frame k refined by the parabola through k-1, k, k+1 (periodic)."""
    n = len(y)
    a, b, c = y[(k - 1) % n], y[k], y[(k + 1) % n]
    den = a - 2 * b + c
    return 0.0 if den == 0 else float(0.5 * (a - c) / den * dt)


def _fit_phase(phase, y, harmonic: float = 1.0) -> tuple[float, float]:
    """Least-squares y = A sin(2 pi h phase + psi) + c: (A, psi)."""
    X = np.column_stack([np.sin(2 * np.pi * harmonic * phase), np.cos(2 * np.pi * harmonic * phase), np.ones(len(y))])
    a, b, _ = np.linalg.lstsq(X, y, rcond=None)[0]
    return float(math.hypot(a, b)), float(math.atan2(b, a))


def measure(arrs: dict, info: dict, H: float) -> dict:
    """Each feature measured on the frames (not read from the spec): bob amplitude and trough time, duty and swing
    time, lift peak, toe-off pitch at lift-off and its peak, trunk roll amplitude and phase, sway, the tail's phase and
    whether its tip moves toward the stance side in each swing, the head's nod and settle, the stride, the step
    times (F14), cadence, the ZMP's margin and the extremes' phases (research 4.1's phase chart)."""
    t, dt = np.asarray(arrs["t"]), float(arrs["dt"])
    n = len(t)
    T, duty = float(info["period"]), float(info["duty"])
    K = int(info.get("strides", 1))
    phase = np.asarray(arrs["phase"])
    sup = np.asarray(arrs["support"], bool)
    lo, td = np.asarray(arrs["lift_off_t"]), np.asarray(arrs["touchdown_t"])
    W = n * dt
    out = {}
    # F1/F2 bob: the hip midpoint's height; the trough after each touchdown as a fraction of stance
    z = np.asarray(arrs["hip_mid"])[:, 2]
    out["bob_pp_H"] = float((z.max() - z.min()) / H)
    tr_frac = []
    for tt in td.ravel():
        if not np.isfinite(tt):
            continue
        a0, a1 = tt - 0.5 * (1 - duty) * T, tt + 0.5 * duty * T
        ks = np.flatnonzero(((t - a0) % W) <= (a1 - a0))
        if len(ks) == 0 or out["bob_pp_H"] < 1e-9:
            continue
        k = int(ks[np.argmin(z[ks])])
        tk = t[k] + _sub_frame(z, k, dt)
        tr_frac.append(((tk - tt + 0.5 * W) % W - 0.5 * W) / (duty * T))
    out["bob_trough_frac_stance"] = float(np.mean(tr_frac)) if tr_frac else None
    # research F2's "settling over 0.1-0.15 s": the trough's time after the touchdown
    out["bob_trough_after_touchdown_s"] = float(np.mean(tr_frac) * duty * T) if tr_frac else None
    # F3 duty and swing time
    out["duty"] = [float(sup[:, j].mean()) for j in range(2)]
    sw_t = []
    for j in range(2):
        for a, b in zip(lo[j], td[j], strict=True):
            if np.isfinite(a) and np.isfinite(b):
                sw_t.append((b - a) % W)
    out["swing_time_s"] = float(np.mean(sw_t))
    out["swing_time_spread_s"] = float(np.ptp(sw_t))
    # F4 lift peak (the lowest pad point's height over each swing), F5 toe-off
    pad_z, pitch = np.asarray(arrs["pad_floor_z"]), np.asarray(arrs["foot_pitch_solved"])
    rol = np.asarray(arrs["rolling"], bool)
    peaks, at_lo, pmax, pk_after, ground = [], [], [], [], []
    for j in range(2):
        for a, b in zip(lo[j], td[j], strict=True):
            if not (np.isfinite(a) and np.isfinite(b)):
                continue
            dur = (b - a) % W
            ks = np.flatnonzero(((t - a) % W) < dur)
            k = int(ks[np.argmax(pad_z[ks, j])])
            peaks.append((((t[k] + _sub_frame(pad_z[:, j], k, dt)) - a) % W) / dur)
            at_lo.append(float(np.interp(a, t, pitch[:, j], period=W)))
            near = np.flatnonzero(((t - a + 0.5 * dur) % W) < dur)  # half a swing either side of the lift-off
            pmax.append(float(pitch[near, j].max()))
            if pmax[-1] > 1e-9:  # the roll's peak relative to the lift-off, and how long the foot rolled on the floor
                kp = int(near[np.argmax(pitch[near, j])])
                pk_after.append(((t[kp] + _sub_frame(pitch[:, j], kp, dt)) - a + 0.5 * W) % W - 0.5 * W)
                before = np.flatnonzero(((a - t) % W) <= duty * T)  # the stance that ends at this lift-off
                ground.append(float(np.count_nonzero(rol[before, j]) * dt))
    out["lift_peak_frac_swing"] = float(np.mean(peaks))
    out["toe_off_at_lift_off_rad"] = float(np.mean(at_lo))
    out["toe_off_peak_rad"] = float(np.max(pmax))
    # where the stance part of the roll is short (little double support), the pitch peaks after the lift-off
    out["toe_off_peak_after_lift_off_s"] = float(np.mean(pk_after)) if pk_after else None
    out["toe_off_ground_roll_s"] = float(np.mean(ground)) if ground else None
    out["foot_pitch_min_rad"] = float(pitch.min())
    stance_flat = sup & ~rol
    out["stance_pad_height_max_mm"] = float(1e3 * np.max(np.abs(pad_z[stance_flat]))) if stance_flat.any() else None
    piv = np.asarray(arrs["toe_pivot"])
    out["rolling_pivot_height_max_mm"] = float(1e3 * np.max(np.abs(piv[rol][:, 2]))) if rol.any() else None
    # the toes and claws carry no contact in the plant (contype 0): a roll about the pad's front edge takes the middle
    # toe's capsule and claw, which reach further forward, below the floor (``toe_clearance``)
    toe_z = np.asarray(arrs["toe_floor_z"])
    out["toe_geoms_below_floor_max_mm"] = float(max(0.0, -1e3 * np.nanmin(toe_z)))
    out["toe_geoms_below_floor_frames"] = int(np.count_nonzero(np.nanmin(toe_z, axis=1) < -1e-4))
    # F6 trunk roll (about the trunk's long axis, relative to home) and sway
    qh = np.asarray(arrs["q_home"])[3:7]
    qc = np.array([qh[0], -qh[1], -qh[2], -qh[3]])
    roll = []
    for qt in np.asarray(arrs["q"])[:, 3:7]:
        r = gait._quat_mul(qc, qt)
        roll.append(2 * math.atan2(r[1], r[0]))
    roll = np.array(roll)
    out["trunk_roll_amp_rad"] = float(0.5 * (roll.max() - roll.min()))
    out["trunk_roll_rms_deg"] = float(math.degrees(np.sqrt(np.mean(roll**2))))
    if out["trunk_roll_amp_rad"] > 1e-9:
        amp, psi = _fit_phase(phase, roll)
        # roll = -A cos(2 pi (phase - (1 - duty/2))) = A sin(2 pi phase - 2 pi (1 - duty/2) - pi/2)
        out["trunk_roll_toward_stance"] = bool(np.cos(psi - (-2 * np.pi * (1 - duty / 2) - np.pi / 2)) > 0)
    com = np.asarray(arrs["com"])
    out["sway_amp_H"] = float(0.5 * (com[:, 1].max() - com[:, 1].min()) / H)
    # F9 tail: phase and the tip's motion in each swing; F10 head: nod, settle, world pitch
    tail = np.asarray(arrs["tail"])
    amp, psi = _fit_phase(phase, tail)
    out["tail_amp_rad"], out["tail_phase_rad"] = amp, psi
    tip_y = np.asarray(arrs["points"])[:, list(POINTS).index("tail_tip"), 1]
    vy = (np.roll(tip_y, -1) - np.roll(tip_y, 1)) / (2 * dt)
    toward = []
    for j, s in enumerate(SIDES):
        stance_sign = -1.0 if s == "left" else 1.0  # the left swings over the right (-y) stance foot
        for a, b in zip(lo[j], td[j], strict=True):
            if np.isfinite(a) and np.isfinite(b):
                tm = a + 0.5 * ((b - a) % W)
                toward.append(bool(np.sign(np.interp(tm % W, t, vy, period=W)) == stance_sign))
    out["tail_tip_toward_stance_in_swing"] = float(np.mean(toward))
    neck = np.asarray(arrs["neck"])
    out["nod_amp_rad"] = _fit_phase(phase, neck, 2.0)[0]
    settle = np.asarray(arrs["head_settle"])
    if settle.max() > 0:
        lags = []
        for tt in td.ravel():
            if np.isfinite(tt):
                ks = np.flatnonzero(((t - tt) % W) < 0.25 * T)
                k = int(ks[np.argmin(neck[ks])])
                lags.append(((t[k] + _sub_frame(neck, k, dt)) - tt) % W)
        out["settle_lag_s"] = float(np.mean(lags))
        out["settle_depth_rad"] = float(np.max(settle * float(info["look"]["head_settle"])))
    hp = np.asarray(arrs["head_pitch_world"])
    out["head_world_pitch_rms_deg"] = float(math.degrees(np.sqrt(np.mean((hp - hp.mean()) ** 2))))
    # F13 stride: each foot's footprint spacing (the foot origin at each touchdown)
    feet = np.asarray(arrs["feet"])
    strides = []
    for j in range(2):
        xs = [float(np.interp(tt, t, feet[:, j, 0])) for tt in td[j] if np.isfinite(tt) and t[0] <= tt <= t[-1]]
        strides += list(np.diff(xs))
    out["stride_H"] = float(np.mean(strides) / H) if strides else float(info["stride"] / H)
    # F14 step times: the merged lift-offs, left and right alternating
    ev = np.sort(lo[np.isfinite(lo)])
    steps = np.diff(np.concatenate([ev, [ev[0] + W]]))
    dev = steps / (0.5 * T) - 1.0
    first_left = bool(np.isclose(ev[0], np.nanmin(lo[0])))
    left = dev[0::2] if first_left else dev[1::2]
    right = dev[1::2] if first_left else dev[0::2]
    out["step_time_dev_max"] = float(np.max(np.abs(dev)))
    out["step_time_dev_std"] = float(np.std(dev))
    out["step_time_left_over_right"] = float((1 + left.mean()) / (1 + right.mean()))
    out["cadence_steps_s"] = float(2 * K / W)
    out["zmp_margin_min_mm"] = float(1e3 * np.min(arrs["zmp_margin"]))
    # phase chart: when each part peaks (stride fractions of its own maximum), research 4.1 step 3
    chart = {"hip_height_max": z, "trunk_roll_max": roll, "neck_max": neck, "tail_max": tail,
             "left_foot_height_max": pad_z[:, 0]}
    out["extremes_phase"] = {k: (float(np.mod(phase[int(np.argmax(v))], 1.0)) if np.ptp(v) > 1e-9 else None)
                             for k, v in chart.items()}
    out["mechanical_tells"] = dict(constant_hip_height=out["bob_pp_H"] < 1e-3, flat_feet_at_lift_off=
                                   out["toe_off_at_lift_off_rad"] < 0.01, identical_steps=out["step_time_dev_std"] < 1e-4,
                                   metronome_nod=out["nod_amp_rad"] > 0.01)
    return out


# ------------------------------------------------------------------------------------------------
# The tail-phase check (research Bottom line 4; F9's phase)
# ------------------------------------------------------------------------------------------------
def stance_yaw_moment(src: dict, traj: dict) -> dict:
    """``turning.inverse_dynamics`` on the walk as a straight run (``turning_traj``; exact split, mu 0.8): the
    single-support stance pad's free yaw moment M_z about its CoP per frame (NaN in double support), its peak and RMS,
    and the single-support friction need mu_req."""
    M, q0 = src["M"], src["q_home"]
    idr = turning.inverse_dynamics(M, turning_traj(traj, M, q0), turning.MU, split="exact", ds_friction=False)
    n = len(traj["q"])
    mz, mu = np.full(n, np.nan), np.full(n, np.nan)
    for k, ph in enumerate(traj["phases"]):
        if len(ph) == 1:
            mz[k] = idr["feet"][ph[0]]["mz"][k]
            mu[k] = idr["feet"][ph[0]]["mu_req"][k]
    return dict(mz=mz, peak_nm=float(np.nanmax(np.abs(mz))), rms_nm=float(np.sqrt(np.nanmean(mz**2))),
                mu_req_max=float(np.nanmax(mu)))


def tail_check(src: dict, spec: gait.GaitSpec, amp: float, phases=TAIL_PHASES, grid: int = TAIL_GRID) -> dict:
    """The stance pad's single-support yaw moment against the tail's phase on one walk.  The walk runs with no tail
    and with ``amp`` at each of ``phases``; per frame M_z = A + B cos(p) + C sin(p) is fitted (the tail's effect is
    linear at these amplitudes; the fit's residual and the no-tail run against A are reported), its peak |M_z| is
    minimised over ``grid`` phases, and the walk is run again at the minimiser to confirm it.  Today's phase is pi/2
    (``gait.GaitSpec.tail_phase``); research Bottom line 4 reads it as adding to the legs' yaw momentum."""
    runs = {}
    none = stance_yaw_moment(src, generate(src, replace(spec, tail_amp=0.0)))
    Z = []
    for p in phases:
        r = stance_yaw_moment(src, generate(src, replace(spec, tail_amp=amp, tail_phase=float(p))))
        runs[f"{p:.6f}"] = dict(phase_rad=float(p), peak_nm=r["peak_nm"], rms_nm=r["rms_nm"], mu_req_max=r["mu_req_max"])
        Z.append(r["mz"])
    Z = np.array(Z)
    ss = np.isfinite(Z).all(axis=0)
    X = np.column_stack([np.ones(len(phases)), np.cos(phases), np.sin(phases)])
    coef, *_ = np.linalg.lstsq(X, Z[:, ss], rcond=None)
    resid = float(np.max(np.abs(X @ coef - Z[:, ss])))
    grid_p = np.linspace(-np.pi, np.pi, grid, endpoint=False)
    pred = np.array([np.max(np.abs(coef[0] + coef[1] * np.cos(p) + coef[2] * np.sin(p))) for p in grid_p])
    best = float(grid_p[int(np.argmin(pred))])
    conf = stance_yaw_moment(src, generate(src, replace(spec, tail_amp=amp, tail_phase=best)))

    def predicted(p):
        return float(np.max(np.abs(coef[0] + coef[1] * np.cos(p) + coef[2] * np.sin(p))))

    today, flipped = 0.5 * math.pi, -0.5 * math.pi
    pred = dict(today=predicted(today), flipped=predicted(flipped), best=predicted(best))
    return dict(amp_rad=amp, no_tail=dict(peak_nm=none["peak_nm"], rms_nm=none["rms_nm"], mu_req_max=none["mu_req_max"]),
                runs=runs, fit_max_residual_nm=resid, no_tail_vs_fit_mean_nm=float(np.max(np.abs(coef[0] - none["mz"][ss]))),
                predicted_peak_nm=pred,
                peak_vs_no_tail_rel={k: v / none["peak_nm"] - 1.0 for k, v in pred.items()},
                best_phase_rad=best, best_phase_cycles_from_today=float(((best - today) / (2 * np.pi) + 0.5) % 1.0 - 0.5),
                confirmed=dict(peak_nm=conf["peak_nm"], rms_nm=conf["rms_nm"], mu_req_max=conf["mu_req_max"]),
                today_adds_to_leg_yaw_momentum=bool(predicted(today) > none["peak_nm"]),
                flipped_lowers_peak=bool(predicted(flipped) < predicted(today)))


def turning_record(body: str) -> dict:
    """The turning screen's recorded straight run of the nominal walk (``turning.screen_model``'s ``runs.straight``:
    ``turning.generate`` at yaw rate 0, exact split, mu 0.8; round two's record of the body): the single-support stance
    pads' largest |M_z| and friction need.  The tail check's run at today's phase on V0 at the nominal walk is the same
    walk through ``gait.generate`` (``tail_check``; research Bottom line 4's "turning screen's straight runs")."""
    from ..stage1 import candidate_id, case_dir

    path = case_dir(body) / "review" / "screening" / f"{candidate_id(body)}_turning.json"
    doc = json.loads(path.read_text())
    ss = {s: doc["runs"]["straight"]["feet"][s]["single_support"] for s in SIDES}
    return dict(record=path.relative_to(STUDY).as_posix(), spec=doc["spec"],
                peak_nm=max(ss[s]["mz_abs_max"] for s in SIDES), mu_req_max=max(ss[s]["mu_req_max"] for s in SIDES))


# ------------------------------------------------------------------------------------------------
# Files
# ------------------------------------------------------------------------------------------------
def _plain(x):
    if isinstance(x, dict):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if isinstance(x, np.ndarray):
        return _plain(x.tolist())
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    return x


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(src: dict, speed: str, variant: str, spec: gait.GaitSpec, g: dict, out_dir: Path) -> dict:
    """Save one variant's arrays (``refs/<body>/<speed>/<variant>.npz``) and return its metadata entry."""
    path = Path(out_dir) / "refs" / src["body"] / speed / f"{variant}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    arrs = arrays(src, g)
    np.savez(path, **arrs)
    info = g["info"]
    keep = {k: v for k, v in info.items() if k not in ("look",)}
    rolls = {s: [{k: v for k, v in r.items()} for r in g["look"]["rolls"][s] if 0 <= r["t_lift_off"] < info["window_s"]]
             for s in SIDES}
    return dict(body=src["body"], speed=speed, variant=variant, features=features_of(variant),
                file=str(path.relative_to(out_dir)), sha256=_sha(path), frames=len(g["q"]), spec=_plain(asdict(spec)),
                info=_plain(keep), ik_ok=bool(g["ik_ok"]), ik_fail_frames=len(g["ik_fail_frames"]),
                zmp_error_final_mm=1e3 * info["zmp_error_history_m"][-1],
                zmp_margin_min_mm=1e3 * info["zmp_margin_min_m"], rolls=_plain(rolls),
                measured=_plain(measure(arrs, info, src["H"])), seconds=g["seconds"])


def _partial_path(out_dir: Path, body: str, speed: str) -> Path:
    return Path(out_dir) / "partial" / f"{body}__{speed}.json"


def _save_partial(path: Path, res: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(_plain(res), indent=1))
    tmp.replace(path)


def _entry_current(e: dict, spec: gait.GaitSpec, out_dir: Path) -> bool:
    """A saved entry is this spec's and its file is the one it records."""
    f = Path(out_dir) / e["file"]
    return e["spec"] == _plain(asdict(spec)) and f.exists() and _sha(f) == e["sha256"]


def body_info(src: dict) -> dict:
    """Per body: H, the foot geometry in the foot frame (pads, toe and heel pivot edges), the point-light markers and
    F5's toe clearance on the rigid foot (``toe_clearance``)."""
    geo = src["geometry"]
    feet = {s: dict(toe_pad=geo[s]["toe_pad"], toe_edge=geo[s]["toe_edge"], heel_pad=geo[s]["heel_pad"],
                    heel_edge=geo[s]["heel_edge"], pads={k: v for k, v in geo[s]["pads"].items()}) for s in SIDES}
    return _plain(dict(H_m=src["H"], foot_frame="the foot body's frame at home (world-aligned)", feet=feet,
                       markers={nm: dict(body=mujoco.mj_id2name(src["M"].m, mujoco.mjtObj.mjOBJ_BODY, b), point=p)
                                for nm, b, p in src["markers"]},
                       toe_clearance=toe_clearance(src, pitch=feature_values(src["body"], src["H"], "nominal")["F4F5"]["toe_off"])))


def _job(args) -> dict:
    """One body and speed: the tail checks (V0: today's tail; V4: F9's phase), then every rung.  Progress is saved to
    ``<out>/partial/<body>__<speed>.json`` after the tail checks and after each rung, and a run with ``resume`` keeps
    what that file holds (a rung whose spec and file still match), so a restart loses at most the rung in hand."""
    body, speed, out_dir, variants, resume = args
    out_dir = Path(out_dir)
    src = _source_cached(body)
    H = src["H"]
    t0 = time.perf_counter()
    part = _partial_path(out_dir, body, speed)
    res = dict(body=body, speed=speed, entries={}, tail={}, tail_phase=None)
    if resume and part.exists():
        res.update(json.loads(part.read_text()))
    lad0 = ladder(body, H, speed)
    if not res["tail"]:
        res["tail"]["V0"] = tail_check(src, lad0["V0"], amp=lad0["V0"].tail_amp)
        if speed == "nominal":  # the same walk as the turning screen's recorded straight run
            rec = turning_record(body)
            ours = res["tail"]["V0"]["runs"][f"{0.5 * math.pi:.6f}"]
            res["tail"]["V0"]["turning_screen"] = dict(rec, today_peak_rel=ours["peak_nm"] / rec["peak_nm"] - 1.0,
                                                       today_mu_req_rel=ours["mu_req_max"] / rec["mu_req_max"] - 1.0)
        res["tail"]["V4"] = tail_check(src, lad0["V4"], amp=feature_values(body, H, speed)["F9F10"]["tail_amp"])
        res["tail_phase"] = res["tail"]["V4"]["best_phase_rad"]
        _save_partial(part, res)
        print(f"  {body} {speed}: tail checks done, F9 phase {res['tail_phase']:.4f} rad", flush=True)
    lad = ladder(body, H, speed, res["tail_phase"])
    for v in variants:
        if v not in lad:
            continue
        if v in res["entries"] and _entry_current(res["entries"][v], lad[v], out_dir):
            continue
        g = generate(src, lad[v])
        res["entries"][v] = e = write(src, speed, v, lad[v], g, out_dir)
        _save_partial(part, res)
        print(f"  {body} {speed} {v}: {e['seconds']:.0f} s, ik {e['ik_ok']}, zmp err {e['zmp_error_final_mm']:.2f} mm, "
              f"margin {e['zmp_margin_min_mm']:.1f} mm", flush=True)
    res["seconds"] = time.perf_counter() - t0
    return res


_SOURCES: dict = {}


def _source_cached(body: str) -> dict:
    if body not in _SOURCES:
        _SOURCES[body] = source(body)
    return _SOURCES[body]


def _merge(out_dir: Path, results: list, infos: dict, t0: float, prov: dict) -> dict:
    """Merge finished jobs into ``look_refs.json`` and ``tail_phase.json`` (rung by rung; other bodies, speeds and rungs
    are kept), with the run's provenance ``prov`` (taken before it wrote anything)."""
    refs_path, tail_path = out_dir / "look_refs.json", out_dir / "tail_phase.json"
    doc = json.loads(refs_path.read_text()) if refs_path.exists() else {}
    tdoc = json.loads(tail_path.read_text()) if tail_path.exists() else {}
    doc.setdefault("bodies", {})
    doc.setdefault("body_info", {})
    tdoc.setdefault("bodies", {})
    for r in results:
        doc["bodies"].setdefault(r["body"], {}).setdefault(r["speed"], {}).update(r["entries"])
        tdoc["bodies"].setdefault(r["body"], {})[r["speed"]] = dict(tail_phase_F9_rad=r["tail_phase"], **r["tail"])
    doc["body_info"].update(infos)
    common = dict(accuracy=ACCURACY, values=VALUES, variants=list(VARIANTS), extras=list(EXTRAS),
                  speeds={s: ("0.40 m/s" if s == "v040" else s) for s in SPEEDS}, provenance=prov,
                  versions=_versions())
    doc.update(common, generator="gait.generate (look fields; designlab/stage2/look.py)",
               note="kinematic references with perfect tracking; priced by the Stage 0 screen before rendering; "
                    "nothing adopted (P24)", seconds=time.perf_counter() - t0)
    tdoc.update(common, method=(tail_check.__doc__ or "").split("\n\n")[0].replace("\n", " "),
                convention="tail = tail_amp sin(2 pi phase + tail_phase), phase 0 at the left foot's lift-off; a "
                           "positive tail_yaw swings the tip to the robot's right (axis about +z); today's tail_phase "
                           "is pi/2")
    out_dir.mkdir(parents=True, exist_ok=True)
    for path, d in ((refs_path, doc), (tail_path, tdoc)):
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(_plain(d), indent=1))
        tmp.replace(path)
    return doc


def _info_job(body: str) -> tuple[str, dict]:
    return body, body_info(_source_cached(body))


def prepare(bodies=BODIES, out_dir: Path = OUT, speeds=SPEEDS, variants=VARIANTS + EXTRAS, procs: int = 1,
            resume: bool = False) -> dict:
    """Generate, measure and save the ladder; ``<out>/look_refs.json`` and ``<out>/tail_phase.json`` are updated in
    place as each body and speed finishes (``_merge``), and its partial file is then removed.  With ``resume``, each
    job keeps the rungs its partial file holds (``_job``)."""
    import multiprocessing as mp

    from ..stage1 import plant

    out_dir = Path(out_dir)
    prov = plant.provenance()  # before this run writes anything
    jobs = [(b, s, str(out_dir), tuple(variants), resume) for b in bodies for s in speeds]
    t0 = time.perf_counter()
    infos = dict(_info_job(b) for b in bodies)
    doc = _merge(out_dir, [], infos, t0, prov)
    if procs > 1:
        with mp.get_context("fork").Pool(procs) as pool:
            for r in pool.imap_unordered(_job, jobs, chunksize=1):
                doc = _merge(out_dir, [r], {}, t0, prov)
                _partial_path(out_dir, r["body"], r["speed"]).unlink(missing_ok=True)
                print(f"  {r['body']} {r['speed']}: merged ({r['seconds']:.0f} s)", flush=True)
    else:
        for j in jobs:
            r = _job(j)
            doc = _merge(out_dir, [r], {}, t0, prov)
            _partial_path(out_dir, r["body"], r["speed"]).unlink(missing_ok=True)
    part = out_dir / "partial"
    if part.is_dir() and not any(part.iterdir()):
        part.rmdir()
    return doc


def remeasure(out_dir: Path = OUT) -> dict:
    """``measure`` again on every saved variant (each file checked against its recorded sha256 first); rewrites the
    measured blocks of ``look_refs.json``."""
    path = Path(out_dir) / "look_refs.json"
    doc = json.loads(path.read_text())
    for d in doc["bodies"].values():
        for entries in d.values():
            for e in entries.values():
                f = Path(out_dir) / e["file"]
                if _sha(f) != e["sha256"]:
                    raise ValueError(f"{f} is not the file look_refs.json records")
                with np.load(f, allow_pickle=False) as z:
                    arrs = {k: z[k] for k in z.files}
                H = e["info"]["stride"] / e["spec"]["stride_rel"]
                e["measured"] = _plain(measure(arrs, dict(e["info"], look=e["spec"]), H))
    path.write_text(json.dumps(_plain(doc), indent=1))
    return doc


def _versions() -> dict:
    import platform

    import scipy

    return dict(python=platform.python_version(), numpy=np.__version__, mujoco=mujoco.__version__,
                scipy=scipy.__version__)


def g2_hashes(body: str, names=("nominal", "brisk", "fr05")) -> dict:
    """Stage 1's references of ``body`` regenerated as ``stage1.refs.prepare`` builds them (``refs.generate`` at r2,
    ``refs.write``'s npz), each file's sha256 (gate G2 compares them with ``stage1/runs/2026-11-08/<body>/refs.json``;
    run with one BLAS thread, as the Stage 1 runner)."""
    from ..stage1 import refs

    src = refs.source(body)
    out = {}
    with tempfile.TemporaryDirectory() as tmp:
        for name in names:
            g = refs.generate(src, name)
            e = refs.write(src, g, Path(tmp))
            out[name] = dict(sha256=e["sha256"], matches_records=e["matches_records"])
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m designlab.stage2.look", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare", help="generate, measure and save the look ladder")
    p.add_argument("--body", action="append", choices=BODIES)
    p.add_argument("--speed", action="append", choices=SPEEDS)
    p.add_argument("--variant", action="append", choices=VARIANTS + EXTRAS)
    p.add_argument("--out", default=str(OUT))
    p.add_argument("--procs", type=int, default=1)
    p.add_argument("--resume", action="store_true", help="keep the rungs each job's partial file holds")
    q = sub.add_parser("g2", help="Stage 1's nominal, brisk and fr05 arrays' sha256 (JSON on stdout)")
    q.add_argument("--body", required=True)
    r = sub.add_parser("remeasure", help="recompute look_refs.json's measured blocks from the saved arrays")
    r.add_argument("--out", default=str(OUT))
    a = ap.parse_args(argv)
    if a.cmd == "prepare":
        prepare(tuple(a.body or BODIES), Path(a.out), tuple(a.speed or SPEEDS), tuple(a.variant or VARIANTS + EXTRAS),
                a.procs, a.resume)
    elif a.cmd == "remeasure":
        remeasure(Path(a.out))
    else:
        print(json.dumps(g2_hashes(a.body)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
