"""Scratch prototype of the causal builder: the preview controller runs on the continuing walk (its corrected ZMP
reference) up to the command frame and only from that frame on previews the commanded plan.  Frames before the
command are a warm-up (they reproduce the walk and are not part of the move's file)."""
from __future__ import annotations

import math
import time

import numpy as np

from designlab import gait, turning
from designlab.stage1 import transitions as tr
import proto_moves as pm

N = 160


def state_at(ref, phase0, first):
    """(c, c', c'') of the straight periodic walk at gait phase ``phase0`` in the frame of a plan whose path origin is
    at phase ``first``."""
    mot = tr._stride_motion(ref); inv = tr._inverse(mot)
    c, dt = ref.com_ref, ref.step
    k0 = int(round(phase0 * N))
    shift = np.array([-first * float(ref.stride_motion[3]), 0.0])

    def at(k):
        s, kk = divmod(k0 + k, N)
        x = c[kk]
        for _ in range(abs(s)):
            x = tr._move(x, inv if s < 0 else mot)
        return x + shift
    p = np.array([at(k) for k in (-2, -1, 0, 1, 2)])
    x0 = np.zeros((3, 2))
    x0[0] = p[2]
    x0[1] = (-p[4] + 8 * p[3] - 8 * p[1] + p[0]) / (12 * dt)
    x0[2] = (-p[4] + 16 * p[3] - 30 * p[2] + 16 * p[1] - p[0]) / (12 * dt * dt)
    return x0


def walk_frames(ref, phases, first, key="com_ref"):
    mot = tr._stride_motion(ref); inv = tr._inverse(mot)
    shift = np.array([-first * float(ref.stride_motion[3]), 0.0])
    out = []
    for p in phases:
        s, kk = divmod(int(round(p * N)), N)
        x = getattr(ref, key)[kk][:2]
        for _ in range(abs(s)):
            x = tr._move(x, inv if s < 0 else mot)
        out.append(x + shift)
    return np.array(out)


def _extend(pl, t, z, to_stand, reps):
    dt_min = float(np.min(np.diff(t)))
    if to_stand:
        ext_t = t[-1] + dt_min * np.arange(1, reps * N + 1)
        ext_z = np.tile(z[-1], (len(ext_t), 1))
    else:
        mot = tr._stride_motion_at(pl, pl.phase[-N])
        T_b = pl.b.period
        z_last, t_last, ez, et = z[-N:], t[-N:], [], []
        for r_ in range(1, reps + 1):
            z_last = tr._move(z_last, mot)
            ez.append(z_last); et.append(t_last + r_ * T_b)
        ext_z, ext_t = np.vstack(ez), np.concatenate(et)
    return np.concatenate([t, ext_t]), np.vstack([z, ext_z])


def com_path_causal(pl, zcmd, zpre_t, zpre, x0, omega, k_cmd):
    """As transitions.com_path, but grid steps before the command frame preview ``zpre`` (the continuing walk, given
    on its own long time axis ``zpre_t``) and the steps from it on preview ``zcmd``."""
    t, K = pl.t, len(pl.t)
    dt_min = float(np.min(np.diff(t)))
    g = tr.preview_gains(dt_min, omega)
    A, B, C, Gi, Gx, Gp, Np = (g[k] for k in ("A", "B", "C", "Gi", "Gx", "Gp", "N"))
    reps = int(math.ceil(tr.PREVIEW_S / (N * dt_min))) + 2
    tt, zz = _extend(pl, t, zcmd, pl.b.stand, reps)
    grid = t[0] + dt_min * np.arange(int(math.floor((t[-1] - t[0]) / dt_min)) + 2 + Np)
    Mn = len(grid) - Np
    out = np.zeros((K, 2))
    zct = np.zeros((K, 2))
    t_cmd = t[k_cmd]
    for ax in (0, 1):
        zc = np.interp(grid, tt, zz[:, ax])
        zp = np.interp(grid, zpre_t, zpre[:, ax])
        xs, us = np.zeros((Mn, 3)), np.zeros(Mn)
        x = np.asarray(x0[:, ax], float).copy()
        esum = 0.0
        u = x_prev = None
        for k in range(Mn):
            zs = zp if grid[k] < t_cmd - 1e-12 else zc
            xs[k] = x
            e = C @ x - zs[k]
            if u is None:  # the first step: the position form (a cold start, as transitions.preview_run)
                esum += e
                u = -Gi * esum - Gx @ x - Gp @ zs[k + 1 : k + 1 + Np]
            else:  # the incremental form: new preview information enters only through its own increments
                u = u - Gi * e - Gx @ (x - x_prev) - Gp @ (zs[k + 1 : k + 1 + Np] - zs[k : k + Np])
            us[k] = u
            x_prev = x
            x = A @ x + B * u
        idx = np.clip(((t - t[0]) / dt_min + 1e-9).astype(int), 0, len(xs) - 1)
        s = t - grid[idx]
        out[:, ax] = xs[idx, 0] + xs[idx, 1] * s + xs[idx, 2] * s * s / 2 + us[idx] * s**3 / 6
        zct[:, ax] = out[:, ax] - (xs[idx, 2] + us[idx] * s) / omega**2   # the cart-table ZMP of the same cubic, exactly
    com_path_causal.zct = zct
    return out


def solve_causal(src, pl, geo, ref_a, walk_long, delta, amp, iters, gain, p_cmd, heading_rot=True, verbose=True):
    """pl: computed from p_cmd - 1 (one warm-up stride), output from p_cmd.  walk_long: the continuing walk's plan on
    a longer window with the same first frame.  delta: the walk's ZMP correction per stride frame (path frame)."""
    gait.set_accuracy("r2")
    M, q_home = src["M"], src["q_home"]
    t0 = time.perf_counter()
    k_cmd = pl.out.start
    kk = np.round(pl.phase * N).astype(int) % N
    rot = lambda th, v: np.column_stack([np.cos(th) * v[:, 0] - np.sin(th) * v[:, 1], np.sin(th) * v[:, 0] + np.cos(th) * v[:, 1]])
    d_cmd = rot(pl.heading, delta[kk]) * amp(pl.phase)[:, None]
    kw = np.round(walk_long.phase * N).astype(int) % N
    zpre = walk_long.zmp + delta[kw]
    x0 = state_at(ref_a, float(pl.phase[0]), float(pl.first))
    zref = pl.zmp + d_cmd
    hist, hist_cmd = [], []
    for it in range(iters + 1):
        com_t = com_path_causal(pl, zref, walk_long.t, zpre, x0, geo.omega, k_cmd)
        qs, foot_err, com_err = tr.solve_frames(M, q_home, pl, com_t, geo)
        qd, qdd = tr.derivatives(M.m, qs, pl.t)
        zmb = tr.multibody_zmp(M, qs, qd, qdd)
        tau_c = com_path_causal.zct - zref                    # what the causal controller could not track
        tau_c[:k_cmd] = 0.0
        e_t = np.nan_to_num(zmb - pl.zmp)                     # against the plan's target (the legacy reading)
        e = np.nan_to_num(zmb - pl.zmp - tau_c)               # against the target plus the causal transient
        e[:k_cmd] = 0.0
        en = np.linalg.norm(e[pl.out], axis=1)
        hist.append(float(en.max()))
        hist_cmd.append((float(np.linalg.norm(e_t[pl.out], axis=1).max()), float(np.linalg.norm(tau_c[pl.out], axis=1)[2:-2].max())))
        if verbose:
            print(f"   pass {it}: consistency err {1e3 * hist[-1]:.3f} mm at out frame {int(np.argmax(en))}; against the target "
                  f"{1e3 * hist_cmd[-1][0]:.2f} mm; causal transient {1e3 * hist_cmd[-1][1]:.2f} mm ({time.perf_counter() - t0:.0f} s)", flush=True)
        if it == iters:
            break
        zref = zref - gain * e
    o = pl.out
    com = np.zeros((len(pl.phase), 3))
    for k in range(len(pl.phase)):
        M.forward(qs[k])
        com[k] = M.com()
    return dict(plan=pl, q=qs[o], qd=qd[o], qdd=qdd[o], com_ref=com_t[o], zmp_mb=zmb[o], com=com[o], q_all=qs,
                com_all=com, com_ref_all=com_t, zmb_all=zmb, foot_error=foot_err[o], zmp_error_history_m=hist,
                ik_ok=bool(np.all(foot_err[o] <= turning.IK_TOL)),
                ik_fail=int(np.sum(np.any(foot_err[o] > turning.IK_TOL, axis=1))), seconds=time.perf_counter() - t0,
                zref=zref[o])
