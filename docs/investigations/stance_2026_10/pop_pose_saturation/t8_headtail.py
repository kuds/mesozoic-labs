"""Head vs tail on r10: static pitch moments about the hip axis and kinematic CoM authority at the settled statue."""
import sys, json, numpy as np, mujoco
D = "scratchpad"
sys.path.insert(0, D)
from common import make_env, custom_reset

G = 9.81
env, u = make_env()
m, d = u.model, u.data
custom_reset(env, u, 0)
for _ in range(400):
    env.step(np.zeros(m.nu))
mujoco.mj_forward(m, d)

pel = m.body("pelvis").id
M = float(m.body_subtreemass[pel])
com = d.subtree_com[pel].copy()
xh = float(np.mean([d.xanchor[m.joint(n).id][0] for n in ("r_hip_pitch", "l_hip_pitch")]))

fl, f6, P, F = u.floor_geom_id, np.zeros(6), [], []
for i in range(d.ncon):
    c = d.contact[i]
    if fl in (c.geom1, c.geom2):
        mujoco.mj_contactForce(m, d, i, f6)
        P.append(c.pos.copy()); F.append(f6[0])
P, F = np.array(P), np.array(F)
cop = (F[:, None] * P).sum(0) / F.sum()

out = dict(total_mass=M, hip_x=xh, com_x=float(com[0]), cop_x=float(cop[0]),
           support_x=[float(P[:, 0].min()), float(P[:, 0].max())], groups=[])
print(f"total mass {M:.2f} kg | CoM {1e3*(com[0]-xh):+.1f} mm from hip axis, {1e3*(com[0]-cop[0]):+.1f} mm from CoP | "
      f"contact x span {1e3*(P[:,0].min()-cop[0]):+.1f}..{1e3*(P[:,0].max()-cop[0]):+.1f} mm about CoP")
rows = [("pelvis own geoms", float(m.body_mass[pel]), d.xipos[pel].copy())]
for i in range(m.nbody):
    if m.body_parentid[i] == pel and i != pel:
        rows.append((m.body(i).name, float(m.body_subtreemass[i]), d.subtree_com[i].copy()))
print(f"{'group':18s} {'kg':>6s} {'%':>5s} {'x-hip mm':>9s} {'moment N.m':>11s}  (+ = ahead of hip, nose-down)")
for n, mm, x in rows:
    mom = mm * G * (x[0] - xh)
    out["groups"].append(dict(name=n, mass=mm, dx_hip=float(x[0] - xh), moment=float(mom)))
    print(f"{n:18s} {mm:6.2f} {100*mm/M:5.1f} {1e3*(x[0]-xh):+9.1f} {mom:+11.1f}")

d2 = mujoco.MjData(m)
def com_at(changes):
    d2.qpos[:] = d.qpos
    for jn, val in changes.items():
        d2.qpos[m.jnt_qposadr[m.joint(jn).id]] = val
    mujoco.mj_kinematics(m, d2); mujoco.mj_comPos(m, d2)
    return d2.subtree_com[pel].copy()

base = com_at({})
jac = np.zeros((3, m.nv))
d2.qpos[:] = d.qpos; mujoco.mj_kinematics(m, d2); mujoco.mj_comPos(m, d2)
mujoco.mj_jacSubtreeCom(m, d2, jac, pel)
q = lambda jn: float(d.qpos[m.jnt_qposadr[m.joint(jn).id]])
print("\nkinematic authority (root and all other joints frozen):")
out["authority"] = {}
for jn, lo, hi in (("neck_pitch", -30, 40), ("head_pitch", -25, 35),
                   ("tail_1_pitch", -12, 12), ("tail_2_pitch", -12, 12), ("tail_3_pitch", -12, 12)):
    dof = m.jnt_dofadr[m.joint(jn).id]
    a = 1e3 * (com_at({jn: np.radians(lo)}) - base); b = 1e3 * (com_at({jn: np.radians(hi)}) - base)
    out["authority"][jn] = dict(q_deg=np.degrees(q(jn)), dxdq_mm_per_deg=float(1e3 * jac[0, dof] * np.pi / 180),
                                dx_lo=float(a[0]), dx_hi=float(b[0]), dz_lo=float(a[2]), dz_hi=float(b[2]))
    print(f"  {jn:13s} q={np.degrees(q(jn)):+6.2f} deg  dCoMx/dq {1e3*jac[0,dof]*np.pi/180:+6.3f} mm/deg  "
          f"at {lo:+d}/{hi:+d} deg: dx {a[0]:+6.2f}/{b[0]:+6.2f} mm, dz {a[2]:+6.2f}/{b[2]:+6.2f} mm")

print("\nwhole tail swung rigidly by tail_1_pitch (hypothetical beyond +-12 deg; kinematics only):")
out["tail_swing"] = {}
for th in (-60, -30, -12, 12, 30, 60):
    dx = 1e3 * (com_at({"tail_1_pitch": q("tail_1_pitch") + np.radians(th)}) - base)
    out["tail_swing"][th] = float(dx[0])
    print(f"  {th:+4d} deg: dCoMx {dx[0]:+6.2f} mm, dCoMz {dx[2]:+6.2f} mm")
print("\nlateral authority (yaw offsets from the statue; kinematics only):")
out["lateral"] = {}
for jn, degs in (("tail_1_yaw", (4, 8)), ("neck_yaw", (10, 20))):
    for dg in degs:
        dd = 1e3 * (com_at({jn: q(jn) + np.radians(dg)}) - base)
        out["lateral"][f"{jn}{dg:+d}"] = float(dd[1])
        print(f"  {jn:10s} {dg:+3d} deg: dCoMy {dd[1]:+6.2f} mm, dCoMx {dd[0]:+6.2f} mm")
json.dump(out, open(f"{D}/t8_headtail.json", "w"), indent=1, default=float)
