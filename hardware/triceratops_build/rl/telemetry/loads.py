"""Joint bearing loads and foot impact forces from a MuJoCo simulation (robot-agnostic; decision D21). Simulation.

For each named joint, every physics step: the reaction force the child body takes from its parent through the joint
(N; the bearing's combined radial and axial load), and the moment about the joint's anchor PERPENDICULAR to its
axis (N m; the bending the bearing pair carries; the moment along the axis is the drive's or the spring's torque).
From MuJoCo's ``cfrc_int`` after ``mj_rnePostConstraint``: a spatial force [moment; force] at the subtree centre of
mass of the body's tree root, moved to the joint anchor by ``m_p = m_ref + (ref - p) x f``.

For each named foot geom, its total normal contact force (N), so impact peaks and their rates can be read.

Collect with ``LoadRecorder`` (call it after each physics step, as an env's per-step hook); summarise with
``summary``: per joint and foot, the median, 95th and 99th percentiles and the maximum, in N, N m and body weights.
These are the loads of a rigid model with estimated masses: a sizing input for the bench plan, not a strength check.
"""

from __future__ import annotations

import mujoco
import numpy as np


class LoadRecorder:
    def __init__(self, model, joint_names, foot_geom_names=(), every: int = 1):
        self.m = model
        self.jids = [model.joint(n).id for n in joint_names]
        self.names = list(joint_names)
        self.bodies = [int(model.jnt_bodyid[j]) for j in self.jids]
        self.roots = [int(model.body_rootid[b]) for b in self.bodies]
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
        for j, b, r in zip(self.jids, self.bodies, self.roots):
            mom, f = d.cfrc_int[b, :3], d.cfrc_int[b, 3:]
            p = d.xanchor[j]
            m_p = mom + np.cross(d.subtree_com[r] - p, f)
            axis = d.xaxis[j]
            m_perp = m_p - np.dot(m_p, axis) * axis
            fr.append(float(np.linalg.norm(f)))
            mo.append(float(np.linalg.norm(m_perp)))
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

    def summary(self, weight_n: float) -> dict:
        def stats(x):
            x = np.asarray(x, dtype=np.float64)
            if not x.size:
                return None
            q = np.percentile(x, [50, 95, 99])
            return {"median": float(q[0]), "p95": float(q[1]), "p99": float(q[2]), "max": float(x.max())}

        F, M = np.asarray(self.force), np.asarray(self.moment)
        out = {"samples": int(F.shape[0]) if F.size else 0, "weight_n": weight_n, "joints": {}, "feet": {}}
        for i, n in enumerate(self.names):
            out["joints"][n] = {"force_n": stats(F[:, i]) if F.size else None,
                                "force_bw": stats(F[:, i] / weight_n) if F.size else None,
                                "bending_nm": stats(M[:, i]) if M.size else None}
        if self.foot:
            P = np.asarray(self.foot)
            for i, n in enumerate(self.foot_names):
                out["feet"][n] = {"normal_n": stats(P[:, i]), "normal_bw": stats(P[:, i] / weight_n)}
        return out
