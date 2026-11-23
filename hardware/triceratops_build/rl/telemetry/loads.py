"""Joint bearing loads and foot contact forces from a MuJoCo simulation (robot-agnostic; decision D21). Simulation.

For each named joint, every sampled physics step:
- ``force``: the reaction force the joint's body takes from its parent (N; the bearing's combined radial and axial
  load);
- ``bending``: the moment about the joint's anchor perpendicular to its axis (N m; the bending the bearing pair
  carries; the moment along the axis is the drive's or the spring's torque).
From MuJoCo's ``cfrc_int`` after ``mj_rnePostConstraint``: a spatial force [moment; force] at the subtree centre of
mass of the body's tree root, moved to the joint anchor by ``m_p = m_ref + (ref - p) x f``.

Bodies with several joints (a hip with pitch and roll hinges on one body, as in the T. rex model): MuJoCo chains a
body's joints in order, so they behave as a SERIAL gimbal with a massless yoke between them. ``cfrc_int`` is per body,
so each joint on it gets the body's force, and each bearing's bending is the moment perpendicular to its own axis,
which includes the other drive's torque: in a serial gimbal each bearing carries it (tested against the same hip
built with an explicit yoke body). For a joint actuated in parallel, with both drives on the parent acting through
linkages, that is too high; such a mechanism needs its own model. The summary names the joints that share a body.
A ball joint transmits no moment (except limits and damping): its bending is not reported. A free joint (a
floating base) is refused.

For each named foot geom, its total normal contact force (N) at each sample. Only the summary's percentiles and
maxima are kept; rates of rise are not computed (save the series with ``foot`` if they are wanted).

Collect with ``LoadRecorder`` (call it after each physics step, as an env's per-step hook; it only reads the state,
after filling ``cfrc_int``); summarise with ``summary``: per joint and foot, the median, 95th and 99th percentiles
and the maximum, in N, N m and body weights. These are the loads of a rigid model with estimated masses: a sizing
input for the bench plan, not a strength check.
"""

from __future__ import annotations

import mujoco
import numpy as np


class LoadRecorder:
    def __init__(self, model, joint_names, foot_geom_names=(), every: int = 1):
        self.m = model
        self.jids = [model.joint(n).id for n in joint_names]
        self.names = list(joint_names)
        for n, j in zip(self.names, self.jids):
            if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                raise ValueError(f"joint {n!r} is a free joint: a floating base has no parent to take loads from")
        self.bodies = [int(model.jnt_bodyid[j]) for j in self.jids]
        self.roots = [int(model.body_rootid[b]) for b in self.bodies]
        self.ball = [model.jnt_type[j] == mujoco.mjtJoint.mjJNT_BALL for j in self.jids]
        self.shared = {n: [model.joint(j).name for j in range(model.njnt) if model.jnt_bodyid[j] == b and j != jid]
                       for n, jid, b in zip(self.names, self.jids, self.bodies)}
        self.feet = [model.geom(n).id for n in foot_geom_names]
        self.foot_names = list(foot_geom_names)
        self.every, self.k = max(1, int(every)), 0
        self.force, self.moment, self.foot = [], [], []
        self._f6 = np.zeros(6)

    def __call__(self, m, d) -> None:
        self.k += 1
        if self.k % self.every:
            return
        mujoco.mj_rnePostConstraint(m, d)  # fills cfrc_int (reading only: no state changes)
        fr, mo = [], []
        for j, b, r, ball in zip(self.jids, self.bodies, self.roots, self.ball):
            mom, f = d.cfrc_int[b, :3], d.cfrc_int[b, 3:]
            m_p = mom + np.cross(d.subtree_com[r] - d.xanchor[j], f)
            fr.append(float(np.linalg.norm(f)))
            if ball:
                mo.append(np.nan)
                continue
            axis = d.xaxis[j]
            mo.append(float(np.linalg.norm(m_p - np.dot(m_p, axis) * axis)))
        self.force.append(fr)
        self.moment.append(mo)
        if self.feet:
            fn = np.zeros(len(self.feet))
            for i in range(d.ncon):
                c = d.contact[i]
                for k, g in enumerate(self.feet):
                    if c.geom1 == g or c.geom2 == g:
                        mujoco.mj_contactForce(m, d, i, self._f6)
                        fn[k] += abs(self._f6[0])
            self.foot.append(fn.tolist())

    def summary(self, weight_n: float, physics_dt: float | None = None, window_s: float = 0.010) -> dict:
        """Percentiles and maxima. With ``physics_dt`` (s) and every-step sampling, each foot also gets its normal
        force averaged over a moving ``window_s`` window (``normal_window_n``; default 10 ms, the window PR #697's
        fixture figures use), which smooths single-sample contact spikes."""
        def stats(x):
            x = np.asarray(x, dtype=np.float64)
            if not x.size or not np.isfinite(x).any():
                return None
            q = np.percentile(x, [50, 95, 99])
            return {"median": float(q[0]), "p95": float(q[1]), "p99": float(q[2]), "max": float(x.max())}

        F, M = np.asarray(self.force), np.asarray(self.moment)
        out = {"samples": int(F.shape[0]) if F.size else 0, "foot_samples": len(self.foot), "weight_n": weight_n,
               "joints": {}, "feet": {}}
        for i, n in enumerate(self.names):
            out["joints"][n] = {"force_n": stats(F[:, i]) if F.size else None,
                                "force_bw": stats(F[:, i] / weight_n) if F.size else None,
                                "bending_nm": None if self.ball[i] or not M.size else stats(M[:, i]),
                                "shares_body_with": self.shared[n]}
        if self.foot:
            P = np.asarray(self.foot)
            k = None
            if physics_dt:
                k = max(1, int(round(window_s / (physics_dt * self.every))))
                out["window_s"] = k * physics_dt * self.every
            for i, n in enumerate(self.foot_names):
                out["feet"][n] = {"normal_n": stats(P[:, i]), "normal_bw": stats(P[:, i] / weight_n)}
                if k and P.shape[0] >= k:
                    w = np.convolve(P[:, i], np.ones(k) / k, mode="valid")
                    out["feet"][n]["normal_window_n"] = stats(w)
        return out
