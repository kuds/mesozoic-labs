"""Scratch prototype of the generalised move planner (round A design).  Not a record; nothing here is written into
the checkout.  ``gplan`` is ``transitions.plan`` with the schedule passed in (phase functions, window, swing filter,
command phase and the "committed footprint" rule F); with ``transitions._param_fns``' own functions it must return
``transitions.plan``'s arrays exactly."""
from __future__ import annotations

import math
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from designlab import gait, statics, turning
from designlab.stage1 import refs, transitions as tr
from designlab.stage1.transitions import PHASE_TOL, SETTLE_STRIDES, SHIFT_STRIDES, SIDES, SUB, Gait, Plan, _ramp, _rot2

SCR = Path(__file__).resolve().parent


def const(x):
    return lambda p: x + 0 * np.asarray(p, float)


def gplan(name, kind, a, b, geo, fn, ramps, window, lift_min=None, lift_max=None, cmd=None, fn_pre=None, n=160,
          from_stand=None, to_stand=None, pre=1.0, post=0.25):
    """kind: the label only.  lift_min / lift_max: swings exist only for lift-offs >= / <= these phases.
    cmd: command phase; with fn_pre (the walk's own constant T, v, w) a stance that began before cmd keeps the
    footprint of the un-commanded path (rule F)."""
    from_stand = a.stand if from_stand is None else from_stand
    to_stand = b.stand if to_stand is None else to_stand
    first, last = window
    lo, hi = first - pre, last + post
    pf = np.arange(int(round((lo - 3) * n * SUB)), int(round((hi + 3) * n * SUB)) + 1) / (n * SUB)
    i0 = int(np.argmin(np.abs(pf - first)))
    Tf, wf, vf = fn["T"](pf), fn["w"](pf), fn["v"](pf)
    dp = 1.0 / (n * SUB)

    def cumint(y):
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
    if cmd is not None and fn_pre is not None:
        Tp, wp, vp = fn_pre["T"](pf), fn_pre["w"](pf), fn_pre["v"](pf)
        thp = cumint(wp * Tp)
        Pp = np.column_stack([cumint(vp * Tp * np.cos(thp)), cumint(vp * Tp * np.sin(thp))]) + geo.origin

        def pose_pre(p):
            p = float(np.clip(p, pf[0], pf[-1]))
            return np.array([np.interp(p, pf, Pp[:, 0]), np.interp(p, pf, Pp[:, 1])]), float(np.interp(p, pf, thp))

    swings = {}
    for side, off in (("left", 0.0), ("right", 0.5)):
        events = [k + off for k in range(int(math.floor(lo)) - 3, int(math.ceil(hi)) + 4)]
        if lift_min is not None:
            events = [e for e in events if e >= lift_min]
        if lift_max is not None:
            events = [e for e in events if e <= lift_max]
        swings[side] = [(e, e + 1.0 - float(fn["duty"](e)), float(fn["lift"](e))) for e in events]

    def footprint(side, stance_from, stance_to):
        if stance_from is None:
            P, th = pose(pf[0])
        elif stance_to is None:
            P, th = pose(pf[-1])
        elif pose_pre is not None and stance_from < cmd - PHASE_TOL:
            P, th = pose_pre(0.5 * (stance_from + stance_to))
        else:
            P, th = pose(0.5 * (stance_from + stance_to))
        rel = geo.feet0[side][:2] - geo.origin
        return P + _rot2(th) @ rel, th

    sw_list, stances = [], {}
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
    phase = np.arange(int(round(lo * n)), int(round(hi * n))) / n
    K = len(phase)
    t = np.interp(phase, pf, tf)
    out = slice(int(round((first - lo) * n)), int(round((last - lo) * n)))
    foot_pos, foot_yaw = np.zeros((K, 2, 3)), np.zeros((K, 2))
    swing = np.zeros((K, 2), bool)
    swing_u = np.full((K, 2), np.nan)
    for j, side in enumerate(SIDES):
        ev, st = swings[side], stances[side]
        z0 = geo.feet0[side][2]
        for k, p in enumerate(phase):
            i = int(np.searchsorted([e[0] - PHASE_TOL for e in ev], p, side="right"))
            if i > 0 and p < ev[i - 1][1] - PHASE_TOL:
                e_lo, e_td, lift = ev[i - 1]
                u = (p - e_lo) / (e_td - e_lo)
                a_ = float(_ramp(u))
                (xy0, th0), (xy1, th1) = (st[i - 1][2], st[i - 1][3]), (st[i][2], st[i][3])
                foot_pos[k, j, :2] = (1 - a_) * xy0 + a_ * xy1
                foot_pos[k, j, 2] = z0 + lift * math.sin(math.pi * u) ** 2
                foot_yaw[k, j] = (1 - a_) * th0 + a_ * th1
                swing[k, j], swing_u[k, j] = True, u
            else:
                foot_pos[k, j, :2], foot_pos[k, j, 2], foot_yaw[k, j] = st[i][2], z0, st[i][3]

    def centroid(j, k):
        side = SIDES[j]
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
        if not prev:
            j_first = nxt[0][1]
            j_stance = 1 - j_first
            a_ = float(np.clip((p + SHIFT_STRIDES) / SHIFT_STRIDES, 0.0, 1.0))
            zmp[k] = (1 - a_) * stand_i + a_ * centroid(j_stance, k)
        elif not nxt:
            p_td, j_land = prev[-1]
            a_ = float(np.clip((p - p_td) / SETTLE_STRIDES, 0.0, 1.0))
            zmp[k] = (1 - a_) * centroid(1 - j_land, k) + a_ * stand_f
        else:
            p_td, j_land = prev[-1]
            p_lo, j_lift = nxt[0]
            a_ = float(np.clip((p - p_td) / (p_lo - p_td), 0.0, 1.0))
            zmp[k] = (1 - a_) * centroid(j_lift, k) + a_ * centroid(j_land, k)
    heading = np.interp(phase, pf, thf)
    path = np.column_stack([np.interp(phase, pf, Pf[:, 0]), np.interp(phase, pf, Pf[:, 1])])
    amp = fn["amp"](phase)
    root_z = geo.root0[2] - amp * fn["crouch"](phase)
    neck = amp * fn["head_amp"](phase) * np.sin(4 * np.pi * phase)
    tail = amp * fn["tail_amp"](phase) * np.sin(2 * np.pi * phase + np.pi / 2)
    return Plan(name=name, kind=kind, a=a, b=b, first=first, last=last, phase=phase, t=t, out=out, foot_pos=foot_pos,
                foot_yaw=foot_yaw, swing=swing, swing_u=swing_u, zmp=zmp, heading=heading, path=path, root_z=root_z,
                neck=neck, tail=tail, stand_point=dict(initial=stand_i, final=stand_f),
                swings=sorted(sw_list, key=lambda s: s[1]), pose_fn=pose, period_fn=period, ramps=ramps)


def legacy(name, a, b, geo):
    kind = tr.kind_of(a.name, b.name)
    fn, ramps = tr._param_fns(kind, a, b)
    return gplan(name, kind, a, b, geo, fn, ramps, tr.WINDOWS[kind], lift_min=0.0 if kind == "start" else None,
                 lift_max=0.0 if kind == "stop" else None)


def gait_from_spec(name, spec, H, yaw_rate=0.0):
    info = spec.describe(H)
    return Gait(name, float(info["period"]), float(info["duty"]), spec.lift_frac * H, float(info["speed"]), yaw_rate,
                spec.head_amp, spec.tail_amp, spec.crouch * H, spec.zmp_iters, spec.zmp_gain, "statics")


def periodic_ref(src, name, spec, cache=True):
    p = SCR / "cache" / f"{src['body']}_{name}.npz"
    if cache and p.exists():
        return refs.Reference.load(p)
    gait.set_accuracy("r2")
    M, q0, H = src["M"], src["q_home"], src["H"]
    t0 = time.perf_counter()
    g = gait.generate(M, q0, H, spec)
    idr = gait.inverse_dynamics(M, g)
    arrs = refs.arrays(src, dict(traj=g, idr=idr))
    p.parent.mkdir(exist_ok=True)
    np.savez(p, **arrs)
    print(f"  periodic {name}: {time.perf_counter() - t0:.0f} s, ik_ok {g['ik_ok']}, zmp hist mm "
          f"{[round(1e3 * x, 3) for x in g['info']['zmp_error_history_m']]}", flush=True)
    return refs.Reference.load(p)


def solve(src, pl, geo, ref_a, iters, gain, verbose=True, x0=None, zref0=None):
    """transitions.generate's loop on a given plan (x0, zref0: a given start state and first ZMP reference)."""
    gait.set_accuracy("r2")
    M, q_home = src["M"], src["q_home"]
    t0 = time.perf_counter()
    x0 = tr.initial_state(pl, ref_a, geo) if x0 is None else x0
    zref = pl.zmp.copy() if zref0 is None else zref0.copy()
    hist = []
    for it in range(iters + 1):
        com_t = tr.com_path(pl, zref, x0, geo.omega)
        qs, foot_err, com_err = tr.solve_frames(M, q_home, pl, com_t, geo)
        qd, qdd = tr.derivatives(M.m, qs, pl.t)
        zmb = tr.multibody_zmp(M, qs, qd, qdd)
        e = np.nan_to_num(zmb - pl.zmp)
        hist.append(float(np.max(np.linalg.norm(e[pl.out], axis=1))))
        if verbose:
            print(f"   pass {it}: ZMP err {1e3 * hist[-1]:.3f} mm ({time.perf_counter() - t0:.0f} s)", flush=True)
        if it == iters:
            break
        zref = zref - gain * e
    o = pl.out
    com = np.zeros((pl.n_out, 3))
    for i, k in enumerate(range(o.start, o.stop)):
        M.forward(qs[k])
        com[i] = M.com()
    return dict(name=pl.name, plan=pl, geo=geo, q=qs[o], qd=qd[o], qdd=qdd[o], com_ref=com_t[o], zmp_mb=zmb[o],
                com=com, foot_error=foot_err[o], zmp_error_history_m=hist,
                ik_ok=bool(np.all(foot_err[o] <= turning.IK_TOL)), ik_fail=int(np.sum(np.any(foot_err[o] > turning.IK_TOL, axis=1))),
                seconds=time.perf_counter() - t0, qs_all=qs, com_all=com_t)


def hip_yaw_use(M, q):
    out = {}
    for side in statics.SIDES:
        j = statics.jid(M.m, f"{side}_hip_yaw")
        lo, hi = M.m.jnt_range[j]
        qi = M.leg_q[side][list(statics.LEG_JOINTS).index("hip_yaw")]
        out[side] = float(max(np.max(q[:, qi]) / hi, np.min(q[:, qi]) / lo))
    return out


def lp_forward_acc(t, com_xy, heading, fs=500.0):
    """1 Hz low-passed CoM acceleration along the (per-frame) heading, on a uniform grid (decision 10's filter)."""
    from designlab.stage1 import metrics
    tu = np.arange(t[0], t[-1], 1.0 / fs)
    x = np.interp(tu, t, com_xy[:, 0]); y = np.interp(tu, t, com_xy[:, 1]); h = np.interp(tu, t, heading)
    ax = np.gradient(np.gradient(x, tu), tu); ay = np.gradient(np.gradient(y, tu), tu)
    af = ax * np.cos(h) + ay * np.sin(h)
    return tu, metrics.lowpass(af, fs), af


def walk_correction(ref, omega):
    """The periodic walk's corrected ZMP reference less its ZMP target, per frame of the stride (path frame = world for
    a straight walk): the cart-table ZMP of its saved LIPM CoM path (com_ref - com_ref'' / omega^2, spectral second
    derivative of the periodic part) minus zmp_ref."""
    c = np.asarray(ref.com_ref, float)
    n = len(c)
    adv = np.array([float(ref.stride_motion[3]), 0.0])
    per = c - np.outer(np.arange(n) / n, adv)          # periodic part
    k = np.fft.fftfreq(n, d=1.0 / n)
    w = 2 * np.pi * k / (n * ref.step)
    acc = np.real(np.fft.ifft(-(w[:, None] ** 2) * np.fft.fft(per, axis=0), axis=0))
    return c - acc / omega**2 - np.asarray(ref.zmp_ref, float)
