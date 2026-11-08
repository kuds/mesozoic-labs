"""Behavior envelope (round two, "Stage 0"): can a sized body support the target behaviors at all?

These are necessary-condition checks on the compiled model with prescribed motions or simple balance
models.  None of them is a controller, and passing them does not show that a controller can produce
the behavior; failing one shows that no controller can with this body and these drives.

1. Animal-like walking speed.  Sizing used Froude 0.012 (nominal) and 0.025 (brisk).  The behavior
   spec wants natural, not slow and rigid, walking and proposes moving toward Fr 0.02-0.05; the
   animal's preferred walk is about Fr 0.05.  A prescribed walk at ``FAST_FROUDE`` with a longer stride
   (``FAST``) is generated and inverse-solved exactly as the sizing walks; each joint group's RMS,
   peak and speed are compared with its selected drive using the sizing margins (sustained planning
   torque; 1.25 x peak against the planning peak; 1.15 x speed against the speed at the bus voltage).
2. Stopping on one foot.  Decelerating the centre of mass at ``STOP_DECEL`` (the spec's start/stop
   limit, 1.5 m/s^2) moves the zero-moment point ahead of the CoM by a z_com / g (linear inverted
   pendulum).  Reported against the stance pad's reach ahead of its centre: the largest deceleration
   one foot can support with the ankle alone is g r_front / z_com.  More than that needs a step.
3. Push recovery by one step (capturability of the linear inverted pendulum, Koolen et al. 2012):
   after a push the instantaneous capture point must be stepped to.  With the CoP held at the front of
   the stance pad (reach r), a step of at most l_max landing after the swing time dt captures an initial
   capture-point offset up to d1 = r + l_max exp(-omega dt), omega = sqrt(g / z_com); the largest
   recoverable push from standing is v1 = omega d1 (a CoM velocity change; impulse m v1).  l_max is the
   furthest flat-footed forward foot placement the leg reaches by IK at the walking hip height; dt is
   the time to make that hip-pitch and knee excursion with a triangular speed profile peaking at each
   drive's speed at the bus voltage, plus ``SWING_OVERHEAD_S`` for lift-off and touchdown.  An upper
   bound under ideal balance control (instant weight transfer, no torque limits in the stance leg);
   use it to compare bodies, not as a prediction.
4. Heat over a session.  RMS torque per drive over ``SESSION`` (a mix of standing, nominal and brisk
   walking, and turning when the turning screen has run), against the sustained planning torque
   (0.57 x rated, assumption A1).  Standing uses the double-support hold.

Run from hardware/trex_form_study via ``designlab.round2`` (``screen_record``) or
``python -m designlab.envelope <round2 dir> <ids...>``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import gait, sizing, statics

G = 9.81
FAST_FROUDE = 0.05
FAST = gait.GaitSpec(froude=FAST_FROUDE, stride_rel=0.80, duty=0.56, lift_frac=0.11)
STOP_DECEL = 1.5  # m/s^2 (BEHAVIOR_SPEC.md section 1, start/stop)
SWING_OVERHEAD_S = 0.08
#: share of a following session; turning is taken from the turning screen when it exists, otherwise its
#: share is walked at the nominal speed
SESSION = dict(standing=0.25, nominal=0.50, brisk=0.10, turning=0.15)
GROUPS = sizing.GROUPS


def _ratios(design, bus, ms: dict) -> dict:
    out = {}
    for g in GROUPS:
        a = design.act(g)
        t, w = ms[g]
        rms, peak, speed = float(np.sqrt(np.mean(t**2))), float(np.max(np.abs(t))), float(np.max(np.abs(w)))
        spd = a.speed_at(bus if a.box is None else None)
        out[g] = dict(actuator=a.id, rms=rms, peak=peak, speed=speed, cont_ratio=rms / a.sustained,
                      peak_ratio=sizing.PEAK_MARGIN * peak / a.peak,
                      speed_ratio=sizing.SPEED_MARGIN * speed / max(spd, 1e-6))
        out[g]["worst"] = max(out[g]["cont_ratio"], out[g]["peak_ratio"], out[g]["speed_ratio"])
    return out


def fast_walk(M: statics.Model, q_home, design, bus: float, layout: str) -> dict:
    # The turning module's generator at zero yaw rate: the same prescribed walk as gait.generate, but the CoM is
    # tracked to 1e-6 m and the stride boundary is unwrapped exactly, so its multibody-ZMP correction converges
    # (about 1 mm instead of centimetres; turning.py docstring).  A walk whose reference CoP still leaves the pad
    # does not balance as prescribed, so its drive loads are not meaningful: the result is then inconclusive.
    from . import turning

    tr = turning.generate(M, q_home, design.sk.H, FAST, yaw_rate=0.0)
    idr = turning.inverse_dynamics(M, tr)
    single = [k for k, ph in enumerate(tr["phases"]) if len(ph) == 1]
    cm = np.asarray(idr["cop_margin"], float)
    cop_min = float(np.nanmin(cm[single])) if single and np.isfinite(cm[single]).any() else float("nan")
    ms = sizing._motor_space(M.names, idr["tau"], idr["omega"], layout)
    r = _ratios(design, bus, ms)
    worst = max(r.items(), key=lambda kv: kv[1]["worst"])
    on_pad = bool(np.isfinite(cop_min) and cop_min > -0.010)
    ik = bool(tr.get("ik_ok", True))
    return dict(spec=FAST.describe(design.sk.H), generator="turning.generate (yaw 0, CoM to 1e-6 m)", ik_ok=ik,
                ik_fail_frames=len(tr.get("ik_fail_frames", [])), min_cop_margin_single_mm=1000 * cop_min,
                reference_cop_on_pad=on_pad, groups=r, worst_group=worst[0], worst_ratio=worst[1]["worst"],
                drives_ok=(bool(ik and worst[1]["worst"] <= 1.0) if on_pad else None))


def _pad_reach(M: statics.Model, q_home, side: str) -> tuple[float, float]:
    """Pad extent ahead of and behind the pad centroid (m), at home."""
    M.forward(q_home)
    poly = np.asarray(M.pad_polygon(side))
    c = statics._polygon_centroid(poly)
    return float(poly[:, 0].max() - c[0]), float(c[0] - poly[:, 0].min())


def stopping(M: statics.Model, q_home) -> dict:
    M.forward(q_home)
    z = float(M.com()[2])
    front, back = _pad_reach(M, q_home, "left")
    need = STOP_DECEL * z / G
    return dict(com_height_m=z, pad_front_reach_m=front, zmp_shift_needed_m=need, ratio=need / max(front, 1e-6),
                max_one_foot_decel_ms2=G * front / z, one_foot=bool(need <= front))


def _max_step(M: statics.Model, q_home, H: float, side: str = "left") -> tuple[float, np.ndarray]:
    """Furthest flat-footed forward placement of one foot (m) at the walking hip height, by IK."""
    m = M.m
    q0 = q_home.copy()
    q0[2] -= sizing.NOMINAL.crouch * H
    M.forward(q_home)
    feet = {s: M.foot_pose(s)[0].copy() for s in statics.SIDES}
    best, best_q = 0.0, q0.copy()
    lo, hi = 0.0, 1.2 * H
    for _ in range(14):
        dx = 0.5 * (lo + hi)
        tg = {s: (feet[s].copy(), np.eye(3)) for s in statics.SIDES}
        tg[side][0][0] += dx
        tg[side][0][2] += 0.0
        # the body is held over the stance foot (the other foot); only the stepping leg reaches
        q, ok = statics.leg_ik(M, q0.copy(), tg, iters=200)
        if ok and _within_limits(m, q):
            lo, best, best_q = dx, dx, q
        else:
            hi = dx
    return best, best_q


def _within_limits(m, q) -> bool:
    for j in range(m.njnt):
        if m.jnt_limited[j] and m.jnt_type[j] == 3:  # hinge
            v = q[m.jnt_qposadr[j]]
            if v < m.jnt_range[j][0] - 1e-6 or v > m.jnt_range[j][1] + 1e-6:
                return False
    return True


def push_recovery(M: statics.Model, q_home, design, bus: float) -> dict:
    H = design.sk.H
    M.forward(q_home)
    z = float(M.com()[2])
    omega = np.sqrt(G / z)
    front, _ = _pad_reach(M, q_home, "right")
    l_max, q_step = _max_step(M, q_home, H, "left")
    m = M.m
    times = {}
    for g, jn in (("hip_pitch", "left_hip_pitch"), ("knee", "left_knee")):
        adr = m.jnt_qposadr[statics.jid(m, jn)]
        dq = abs(float(q_step[adr] - q_home[adr]))
        a = design.act(g)
        w = a.speed_at(bus if a.box is None else None)
        times[g] = dict(excursion_rad=dq, drive_speed_rad_s=w, time_s=2.0 * dq / max(w, 1e-6))
    dt = max(t["time_s"] for t in times.values()) + SWING_OVERHEAD_S
    d1 = front + l_max * np.exp(-omega * dt)
    v1 = omega * d1
    mass = float(M.mass())
    return dict(com_height_m=z, omega=float(omega), pad_front_reach_m=front, max_step_m=l_max,
                max_step_over_H=l_max / H, swing=times, swing_time_s=dt, capture_offset_m=float(d1),
                max_push_dv_ms=float(v1), max_push_impulse_ns=float(mass * v1),
                max_push_froude=float(v1 / np.sqrt(G * H)))


def session_heat(M: statics.Model, q_home, design, layout: str, runs: dict, turning: dict | None) -> dict:
    """Session RMS torque per drive against the sustained planning torque."""
    names = M.names
    tau_ds, _ = statics.contact_torques(M, q_home, support=("left", "right"))
    ms_stand = sizing._motor_space(names, tau_ds[None, :], np.zeros((1, len(names))), layout)
    ms = {k: sizing._motor_space(names, v["idr"]["tau"], v["idr"]["omega"], layout) for k, v in runs.items()}
    out = {}
    for g in GROUPS:
        a = design.act(g)
        sq = SESSION["standing"] * float(np.mean(ms_stand[g][0] ** 2))
        sq += SESSION["nominal"] * float(np.mean(ms["nominal"][g][0] ** 2))
        sq += SESSION["brisk"] * float(np.mean(ms["brisk"][g][0] ** 2))
        t_rms = None
        if turning and g in turning:
            t_rms = float(turning[g])
            sq += SESSION["turning"] * t_rms**2
        else:
            sq += SESSION["turning"] * float(np.mean(ms["nominal"][g][0] ** 2))
        rms = float(np.sqrt(sq))
        out[g] = dict(actuator=a.id, session_rms=rms, sustained=a.sustained, ratio=rms / a.sustained,
                      turning_rms_used=t_rms)
    worst = max(out.items(), key=lambda kv: kv[1]["ratio"])
    return dict(session=SESSION, groups=out, worst_group=worst[0], worst_ratio=worst[1]["ratio"],
                turning_included=bool(turning))


def screen(M: statics.Model, q_home, design, rec: dict, trajs: dict | None = None, turning: dict | None = None) -> dict:
    bus = rec["selection"]["bus"]
    layout = design.cand.leg_layout
    runs = {}
    for name, spec in (("nominal", sizing.NOMINAL), ("brisk", sizing.BRISK)):
        tr = trajs[name] if trajs and name in trajs else gait.generate(M, q_home, design.sk.H, spec)
        runs[name] = dict(traj=tr, idr=gait.inverse_dynamics(M, tr))
    fw = fast_walk(M, q_home, design, bus, layout)
    st = stopping(M, q_home)
    pr = push_recovery(M, q_home, design, bus)
    sh = session_heat(M, q_home, design, layout, runs, turning)
    return dict(module="designlab.envelope", fast_walk=fw, stopping=st, push_recovery=pr, session_heat=sh,
                summary=dict(fast_walk_drives_ok=fw["drives_ok"], fast_walk_reference_cop_on_pad=fw["reference_cop_on_pad"],
                             fast_walk_worst=[fw["worst_group"], fw["worst_ratio"]],
                             fast_walk_ik_ok=fw["ik_ok"], fast_walk_speed_ms=fw["spec"]["speed"],
                             stop_one_foot=st["one_foot"], stop_max_one_foot_decel_ms2=st["max_one_foot_decel_ms2"],
                             push_max_dv_ms=pr["max_push_dv_ms"], push_max_impulse_ns=pr["max_push_impulse_ns"],
                             push_max_step_over_H=pr["max_step_over_H"], push_swing_time_s=pr["swing_time_s"],
                             session_worst=[sh["worst_group"], sh["worst_ratio"]],
                             session_turning_included=sh["turning_included"]))


def turning_rms_from(turning_doc: dict) -> dict | None:
    """Per-group RMS torque of the turning screen's walking turn, if its output provides one
    (``groups`` -> {group: {"rms": ...}} under the first walking-turn run); otherwise None."""
    try:
        runs = turning_doc.get("runs", {})
        key = next(k for k in runs if "walk" in k and "0.35" in k) if runs else None
        g = runs[key]["groups"] if key else turning_doc["groups"]
        return {k: float(v["rms"]) for k, v in g.items() if isinstance(v, dict) and "rms" in v}
    except (StopIteration, KeyError, TypeError):
        return None


def screen_record(cid: str, rec: dict, M, q_home, design, trajs: dict, scr: Path, turning: dict | None = None) -> dict:
    out = screen(M, q_home, design, rec, trajs, turning)
    (scr / f"{cid}_envelope.json").write_text(json.dumps(out, indent=1, default=lambda x: x.tolist()
                                                          if hasattr(x, "tolist") else str(x)))
    return out["summary"]


def main():
    import argparse

    import designlab  # noqa: F401

    from . import round2
    from .pipeline import build_candidate

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("round2")
    ap.add_argument("ids", nargs="+")
    a = ap.parse_args()
    out = Path(a.round2)
    for cid in a.ids:
        rec = json.loads((out / "candidates" / f"{cid}.json").read_text())
        cand = round2.candidate(cid, rec.get("overrides", ()))
        res = build_candidate(cand, out / "review" / "mjcf", selection=dict(rec["selection"]), B=rec["B"], write=False)
        s = screen(res["model"], res["q_home"], res["design"], rec)
        print(cid, json.dumps(s["summary"]))


if __name__ == "__main__":
    main()
