"""Stance solving and contact-consistent joint loads.

Joint torques come from the full free-base equations of motion of the compiled model:

    M(q) qdd + h(q, qd) = S^T tau + sum_f J_f^T w_f

The six unactuated (base) rows determine the foot wrenches ``w_f``: uniquely in single
support, and as the minimum-joint-effort distribution in double support.  The centre of
pressure of each foot is then checked against its pad polygon.  Nothing assumes a fixed base.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
from scipy.spatial import ConvexHull

SIDES = ("left", "right")
LEG_JOINTS = ("hip_roll", "hip_yaw", "hip_pitch", "knee", "ankle_pitch", "ankle_roll")


def jid(m, name):
    return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)


def bid(m, name):
    return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)


def gid(m, name):
    return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name)


@dataclass
class Model:
    m: mujoco.MjModel
    d: mujoco.MjData
    leg_q: dict  # side -> qpos indices (6)
    leg_v: dict  # side -> dof indices (6)
    foot: dict  # side -> body id
    pads: dict  # side -> list of geom ids touching the floor
    act_dofs: np.ndarray  # all actuated dof indices (non-root)
    names: list  # joint names for act_dofs

    @classmethod
    def load(cls, path: str) -> "Model":
        m = mujoco.MjModel.from_xml_path(path)
        d = mujoco.MjData(m)
        leg_q, leg_v, foot, pads = {}, {}, {}, {}
        for s in SIDES:
            ids = [jid(m, f"{s}_{j}") for j in LEG_JOINTS]
            leg_q[s] = np.array([m.jnt_qposadr[i] for i in ids])
            leg_v[s] = np.array([m.jnt_dofadr[i] for i in ids])
            foot[s] = bid(m, f"{s}_foot")
            pads[s] = [g for g in range(m.ngeom) if m.geom_bodyid[g] == foot[s] and m.geom_contype[g] == 1]
        act, names = [], []
        for j in range(m.njnt):
            if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE:
                act.append(m.jnt_dofadr[j])
                names.append(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j))
        return cls(m, d, leg_q, leg_v, foot, pads, np.array(act), names)

    # ------------------------------------------------------------------ kinematics helpers
    def forward(self, q):
        self.d.qpos[:] = q
        self.d.qvel[:] = 0
        mujoco.mj_forward(self.m, self.d)

    def foot_pose(self, side):
        b = self.foot[side]
        return self.d.xpos[b].copy(), self.d.xmat[b].reshape(3, 3).copy()

    def pad_polygon(self, side) -> np.ndarray:
        """World xy polygon of the pads of one foot (bottom faces)."""
        pts = []
        for g in self.pads[side]:
            R = self.d.geom_xmat[g].reshape(3, 3)
            h = self.m.geom_size[g]
            c = self.d.geom_xpos[g]
            for sx in (-1, 1):
                for sy in (-1, 1):
                    p = c + R @ np.array([sx * h[0], sy * h[1], -h[2]])
                    pts.append(p[:2])
        pts = np.array(pts)
        hull = ConvexHull(pts)
        return pts[hull.vertices]

    @property
    def root(self) -> int:
        return int(self.m.jnt_bodyid[0])

    def com(self):
        return self.d.subtree_com[self.root].copy()

    def mass(self):
        return float(self.m.body_subtreemass[self.root])


def _rot_err(R_cur, R_tgt):
    E = R_tgt @ R_cur.T
    return 0.5 * np.array([E[2, 1] - E[1, 2], E[0, 2] - E[2, 0], E[1, 0] - E[0, 1]])


def leg_ik(M: Model, q0: np.ndarray, targets: dict, iters: int = 80, tol: float = 1e-7, damping: float = 1e-4):
    """Solve both legs for foot body poses ``targets[side] = (pos, R)`` with the base fixed."""
    q = q0.copy()
    m, d = M.m, M.d
    jacp = np.zeros((3, m.nv))
    jacr = np.zeros((3, m.nv))
    ok = True
    for side, (p_t, R_t) in targets.items():
        for _ in range(iters):
            M.forward(q)
            p, R = M.foot_pose(side)
            e = np.concatenate([p_t - p, _rot_err(R, R_t)])
            if np.linalg.norm(e) < tol:
                break
            mujoco.mj_jacBody(m, d, jacp, jacr, M.foot[side])
            J = np.vstack([jacp, jacr])[:, M.leg_v[side]]
            dq = J.T @ np.linalg.solve(J @ J.T + damping * np.eye(6), e)
            q[M.leg_q[side]] += dq
            for qi in M.leg_q[side]:
                j = [jj for jj in range(m.njnt) if m.jnt_qposadr[jj] == qi][0]
                lo, hi = m.jnt_range[j]
                q[qi] = np.clip(q[qi], lo, hi)
        else:
            ok = False
        M.forward(q)
        p, R = M.foot_pose(side)
        if np.linalg.norm(np.concatenate([p_t - p, _rot_err(R, R_t)])) > 1e-4:
            ok = False
    return q, ok


def solve_stance(M: Model, q_design: np.ndarray, foot_half_spacing: float, iters: int = 30):
    """Nominal stance: feet flat at +-foot_half_spacing, shifted fore/aft so the CoM is centred
    (x) over the combined support polygon; trunk pose fixed at the design pose."""
    q = q_design.copy()
    M.forward(q)
    base = {}
    for side, s in (("left", 1), ("right", -1)):
        p, R = M.foot_pose(side)
        base[side] = (p.copy(), R.copy(), s)
    dx = 0.0
    ok = True
    for _ in range(iters):
        targets = {side: (np.array([p[0] + dx, s * foot_half_spacing, p[2]]), np.eye(3)) for side, (p, R, s) in base.items()}
        q, ok = leg_ik(M, q, targets)
        poly = np.vstack([M.pad_polygon("left"), M.pad_polygon("right")])
        hull = poly[ConvexHull(poly).vertices]
        cx = _polygon_centroid(hull)
        com = M.com()
        err = com[0] - cx[0]
        if abs(err) < 2e-4:
            break
        dx += err
    return q, ok, dx


def _polygon_centroid(P):
    x, y = P[:, 0], P[:, 1]
    a = x * np.roll(y, -1) - np.roll(x, -1) * y
    A = 0.5 * a.sum()
    cx = ((x + np.roll(x, -1)) * a).sum() / (6 * A)
    cy = ((y + np.roll(y, -1)) * a).sum() / (6 * A)
    return np.array([cx, cy])


def support_margin(point_xy, polygon) -> float:
    """Signed distance from point to polygon edges (positive inside)."""
    P = polygon
    n = len(P)
    dmin = np.inf
    inside = True
    for k in range(n):
        a, b = P[k], P[(k + 1) % n]
        e = b - a
        nrm = np.array([e[1], -e[0]]) / np.linalg.norm(e)
        # orientation: polygon from ConvexHull is counter-clockwise -> inward normal is (-e_y, e_x)
        nin = -nrm
        dist = np.dot(point_xy - a, nin)
        dmin = min(dmin, dist)
        if dist < 0:
            inside = False
    return float(dmin) if inside else float(dmin)


def contact_torques(M: Model, q, qd=None, qdd=None, support=("left", "right"), weights=None):
    """Joint torques and foot wrenches for given motion and support set (see module doc)."""
    m, d = M.m, M.d
    d.qpos[:] = q
    d.qvel[:] = 0 if qd is None else qd
    mujoco.mj_forward(m, d)
    Mfull = np.zeros((m.nv, m.nv))
    mujoco.mj_fullM(m, d, Mfull)
    acc = np.zeros(m.nv) if qdd is None else qdd
    rhs = Mfull @ acc + d.qfrc_bias  # = S^T tau + J^T w
    jacp = np.zeros((3, m.nv))
    jacr = np.zeros((3, m.nv))
    Js = []
    for side in support:
        mujoco.mj_jacBody(m, d, jacp, jacr, M.foot[side])
        Js.append(np.vstack([jacp, jacr]).copy())
    J = np.vstack(Js)  # (6k, nv)
    base = np.arange(6)
    act = M.act_dofs
    A = J[:, base].T  # (6, 6k)
    b = rhs[base]
    if len(support) == 1:
        w = np.linalg.solve(A, b)
    else:
        w = _double_support_qp(M, J, A, b, rhs, act, support, weights)
    tau = rhs[act] - J[:, act].T @ w
    wrenches = {side: w[6 * k : 6 * k + 6] for k, side in enumerate(support)}
    return tau, wrenches


def _double_support_qp(M, J, A, b, rhs, act, support, weights, mu=0.8):
    """Minimum-effort split of the (unique) total contact wrench between two flat feet, with
    non-negative normal force, friction pyramid and centre of pressure inside each pad box."""
    from scipy.optimize import minimize

    Ja = J[:, act].T
    ra = rhs[act]
    W = np.eye(len(act)) if weights is None else np.diag(weights)
    nw = J.shape[0]
    H = Ja.T @ W @ Ja + 1e-6 * np.eye(nw)
    g = -Ja.T @ W @ ra
    # unconstrained-in-inequalities KKT solution as the warm start
    K = np.block([[H, A.T], [A, np.zeros((6, 6))]])
    w0 = np.linalg.solve(K, np.concatenate([-g, b]))[:nw]
    G_rows, h_rows = [], []  # G w >= h
    for k, side in enumerate(support):
        o = 6 * k
        p, _ = M.foot_pose(side)
        hgt = p[2]
        poly = M.pad_polygon(side) - p[:2]
        xmin, ymin = poly.min(0)
        xmax, ymax = poly.max(0)
        e = np.zeros((10, nw))
        # fz >= 0
        e[0, o + 2] = 1
        # friction: mu fz -+ fx >= 0, mu fz -+ fy >= 0
        e[1, o + 2], e[1, o + 0] = mu, -1
        e[2, o + 2], e[2, o + 0] = mu, 1
        e[3, o + 2], e[3, o + 1] = mu, -1
        e[4, o + 2], e[4, o + 1] = mu, 1
        # CoP x = -(m_y + h f_x)/fz in [xmin, xmax];  CoP y = (m_x - h f_y)/fz in [ymin, ymax]
        e[5, o + 4], e[5, o + 0], e[5, o + 2] = -1, -hgt, -xmin  # -(my + h fx) - xmin fz >= 0
        e[6, o + 4], e[6, o + 0], e[6, o + 2] = 1, hgt, xmax  # (my + h fx) + xmax fz >= 0
        e[7, o + 3], e[7, o + 1], e[7, o + 2] = 1, -hgt, -ymin  # (mx - h fy) - ymin fz >= 0
        e[8, o + 3], e[8, o + 1], e[8, o + 2] = -1, hgt, ymax  # -(mx - h fy) + ymax fz >= 0
        e[9, o + 5] = 0  # placeholder (yaw moment unconstrained)
        G_rows.append(e[:9])
        h_rows.append(np.zeros(9))
    Gm = np.vstack(G_rows)
    hm = np.concatenate(h_rows)
    if np.all(Gm @ w0 >= hm - 1e-9):
        return w0
    res = minimize(lambda x: 0.5 * x @ H @ x + g @ x, w0, jac=lambda x: H @ x + g, method="SLSQP",
                   constraints=[dict(type="eq", fun=lambda x: A @ x - b, jac=lambda x: A),
                                dict(type="ineq", fun=lambda x: Gm @ x - hm, jac=lambda x: Gm)],
                   options=dict(maxiter=200, ftol=1e-10))
    return res.x if res.success else w0


def total_zmp(M: Model, q, qd, qdd) -> np.ndarray:
    """ZMP (world xy) of the total contact wrench required by the motion (independent of the
    split between feet), from the base rows of the equations of motion."""
    m, d = M.m, M.d
    d.qpos[:] = q
    d.qvel[:] = qd
    mujoco.mj_forward(m, d)
    Mfull = np.zeros((m.nv, m.nv))
    mujoco.mj_fullM(m, d, Mfull)
    rhs = Mfull @ qdd + d.qfrc_bias
    # A wrench (f, tau) applied at a world point p on the root: generalized force = J_p^T f + J_r^T tau.
    jacp = np.zeros((3, m.nv))
    jacr = np.zeros((3, m.nv))
    p0 = np.array([d.xpos[M.root][0], d.xpos[M.root][1], 0.0])
    mujoco.mj_jac(m, d, jacp, jacr, p0, M.root)
    A = np.vstack([jacp, jacr])[:, :6].T
    w = np.linalg.solve(A, rhs[:6])
    f, tau = w[:3], w[3:]
    if f[2] <= 1e-6:
        return np.array([np.nan, np.nan])
    return p0[:2] + np.array([-tau[1] / f[2], tau[0] / f[2]])


def cop(M: Model, side, wrench) -> np.ndarray:
    """Centre of pressure (world xy) of a foot wrench expressed at the foot body origin."""
    f, mom = wrench[:3], wrench[3:]
    p, _ = M.foot_pose(side)
    # moment about the floor point below the origin: m_floor = m + (p - p_floor) x f
    pf = np.array([p[0], p[1], 0.0])
    m_floor = mom + np.cross(p - pf, f)
    if f[2] <= 1e-6:
        return np.array([np.nan, np.nan])
    return pf[:2] + np.array([-m_floor[1] / f[2], m_floor[0] / f[2]])


def single_support_hold(M: Model, q_stance: np.ndarray, stance: str, lift: float, trunk_rolls_deg=(-6, -3, 0, 3, 6),
                        swing_in=(0.0, 0.02), margin_req: float = 0.003):
    """Quasi-static single-support holds: CoM over the stance pad centroid, swing foot lifted.

    Searches trunk roll (about the stance side) and swing-foot lateral tuck; returns the
    feasible pose with the least stance hip-roll torque, plus every evaluated case.
    """
    m = M.m
    swing = "right" if stance == "left" else "left"
    cases = []
    M.forward(q_stance)
    p_st, _ = M.foot_pose(stance)
    p_sw, _ = M.foot_pose(swing)
    poly = M.pad_polygon(stance)
    centre = _polygon_centroid(poly)
    root_q = slice(0, 7)
    names = M.names
    hip_roll_idx = names.index(f"{stance}_hip_roll")
    for roll_deg in trunk_rolls_deg:
        for tuck in swing_in:
            q = q_stance.copy()
            ok = True
            for _ in range(25):
                # shift trunk laterally so the CoM sits over the stance pad centroid
                M.forward(q)
                com = M.com()
                err = centre - com[:2]
                q[0] += err[0]
                q[1] += err[1]
                sgn = 1 if stance == "left" else -1
                # trunk roll toward the stance side (positive roll_deg leans over the stance foot)
                quat0 = q_stance[3:7]
                rq = np.zeros(4)
                mujoco.mju_axisAngle2Quat(rq, np.array([1.0, 0, 0]), np.radians(roll_deg) * sgn)
                qq = np.zeros(4)
                mujoco.mju_mulQuat(qq, rq, quat0)
                q[3:7] = qq
                tgt = {stance: (p_st.copy(), np.eye(3)),
                       swing: (p_sw + np.array([0, (1 if swing == "left" else -1) * -tuck, lift]), np.eye(3))}
                q, ok = leg_ik(M, q, tgt, iters=60)
                if np.linalg.norm(err) < 2e-4:
                    break
            tau, w = contact_torques(M, q, support=(stance,))
            c = cop(M, stance, w[stance])
            margin = support_margin(c, poly)
            cases.append(dict(roll_deg=roll_deg, tuck=tuck, ik_ok=bool(ok), margin=margin,
                              hip_roll=float(tau[hip_roll_idx]), tau=tau.tolist(), q=q.tolist()))
    feas = [c for c in cases if c["ik_ok"] and c["margin"] >= margin_req]
    best = min(feas, key=lambda c: abs(c["hip_roll"])) if feas else None
    del root_q, m
    return best, cases
