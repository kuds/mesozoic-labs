"""Joint bearing loads and foot contact forces from a MuJoCo simulation (robot-agnostic; decision D21). Simulation.

For each named joint, every sampled physics step:
- ``force``: the reaction force the joint's body takes from its parent (N; the bearing's combined radial and axial
  load);
- ``bending``: the moment about the joint's anchor that the structure carries, i.e. with the components along the
  body's hinge axes removed (N m; the drives or springs carry those). For a body with one hinge, that is the moment
  perpendicular to its axis, which the bearing pair carries.
From MuJoCo's ``cfrc_int`` after ``mj_rnePostConstraint``: a spatial force [moment; force] at the subtree centre of
mass of the body's tree root, moved to the joint anchor by ``m_p = m_ref + (ref - p) x f``.

Bodies with several joints (a hip with pitch and roll hinges on one body, as in the T. rex model): ``cfrc_int`` is
per body, so each joint on it gets the body's force, and the bending has ALL of the body's hinge axes removed
(Gram-Schmidt on non-parallel axes), so one hinge's drive torque is not reported as the other's bending. The
summary names the joints that share a body. A ball joint transmits no moment (except limits and damping): its
bending is not reported. Slide joints remove nothing from the moment. A free joint (a floating base) is refused.

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


def _orthonormal(axes) -> np.ndarray:
    """An orthonormal basis (rows) of the span of ``axes`` (world vectors), dropping parallel ones."""
    basis = []
    for a in axes:
        v = np.asarray(a, dtype=np.float64).copy()
        for b in basis:
            v -= np.dot(v, b) * b
        n = np.linalg.norm(v)
        if n > 1e-9:
            basis.append(v / n)
    return np.array(basis).reshape(-1, 3)


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
        # every hinge on each joint's body, named or not: their axes are drive directions, not bending
        self.hinges = [[j for j in range(model.njnt) if model.jnt_bodyid[j] == b
                        and model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE] for b in self.bodies]
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
        for j, b, r, ball, hinges in zip(self.jids, self.bodies, self.roots, self.ball, self.hinges):
            mom, f = d.cfrc_int[b, :3], d.cfrc_int[b, 3:]
            m_p = mom + np.cross(d.subtree_com[r] - d.xanchor[j], f)
            fr.append(float(np.linalg.norm(f)))
            if ball:
                mo.append(np.nan)
                continue
            for e in _orthonormal([d.xaxis[h] for h in hinges]):
                m_p = m_p - np.dot(m_p, e) * e
            mo.append(float(np.linalg.norm(m_p)))
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
            if not x.size or not np.isfinite(x).any():
                return None
            q = np.percentile(x, [50, 95, 99])
            return {"median": float(q[0]), "p95": float(q[1]), "p99": float(q[2]), "max": float(x.max())}

        F, M = np.asarray(self.force), np.asarray(self.moment)
        out = {"samples": int(F.shape[0]) if F.size else 0, "weight_n": weight_n, "joints": {}, "feet": {}}
        for i, n in enumerate(self.names):
            out["joints"][n] = {"force_n": stats(F[:, i]) if F.size else None,
                                "force_bw": stats(F[:, i] / weight_n) if F.size else None,
                                "bending_nm": None if self.ball[i] or not M.size else stats(M[:, i]),
                                "shares_body_with": self.shared[n]}
        if self.foot:
            P = np.asarray(self.foot)
            for i, n in enumerate(self.foot_names):
                out["feet"][n] = {"normal_n": stats(P[:, i]), "normal_bw": stats(P[:, i] / weight_n)}
        return out
