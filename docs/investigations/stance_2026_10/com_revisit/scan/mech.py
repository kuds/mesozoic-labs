"""The robot creep's mechanism, measured on any model's spawn (generalised from the verifier's v_trace.py and the lens's
mismatch.py; no file of the tree is written).

Usage: python mech.py <model> <seed> <noise|none> <t_start_s> <n_substeps> [spec-json]
  The episode is driven by env.step (neutral action) to t_start, then physics is stepped with mj_step directly; at each
  substep a scratch MjData runs mj_forward on the identical state (the constraint solver's forces and acceleration).
  Printed for the stretch:
    - the implicit-damping mismatch: realised v against v + h (M + h D)^-1 (qfrc_smooth + qfrc_constraint) (D by
      central finite differences of qfrc_actuator + qfrc_passive in qvel) and against the solver's own v + h qacc;
      the DOF where the solver's prediction is most wrong;
    - the discrete work of qfrc_constraint along v1, v0 and their mean, of the damping (D v1 . v1 h), and the change
      of E = kinetic + gravity + servo spring (kp/2 (ctrl - actuator_length)^2);
    - the pre-stress: per foot the floor normal and tangential load (world frame), the friction ratio, the
      force-weighted contact point in the foot site's frame (and the extent of the foot's floor contacts), the
      contact count, and the hip-roll and ankle servo errors (actuator_length - ctrl);
    - the largest eigenvalue of h M^-1 D and its dominant DOF at this state;
    - the root vz RMS and its spectral peak.
"""

import json
import sys

import mujoco
import numpy as np

import scanlib as S
from environments.shared.gait.morphology import Morphology

model_name, seed = sys.argv[1], int(sys.argv[2])
noise = None if sys.argv[3] == "none" else float(sys.argv[3])
t_start, n = float(sys.argv[4]), int(sys.argv[5])
spec = json.loads(sys.argv[6]) if len(sys.argv) > 6 else {}

CLS = S.CLASSES[model_name]
inst = CLS()
env = inst.make_env() if noise is None else inst.make_env(reset_noise_scale=noise)
S.perturb(env, spec)
u = env.unwrapped
m, d = u.model, u.data
morph = Morphology.from_env(env, CLS.model_id)
env.reset(seed=seed)
neutral = np.zeros(env.action_space.shape, dtype=np.float32)
for _ in range(int(round(t_start / u.dt))):
    env.step(neutral)
h = m.opt.timestep
print(f"{model_name} seed {seed} noise {noise} spec {spec}: from t = {d.time:.3f} s, {n} substeps of {h * 1e3:.2f} ms "
      f"(integrator {m.opt.integrator})")


def damping_fd(eps=1e-6):
    D = np.zeros((m.nv, m.nv))
    q0 = d.qvel.copy()
    for j in range(m.nv):
        f = []
        for s in (+1, -1):
            d.qvel[:] = q0
            d.qvel[j] += s * eps
            mujoco.mj_forward(m, d)
            f.append(d.qfrc_actuator + d.qfrc_passive)
        D[:, j] = -(f[0] - f[1]) / (2 * eps)
    d.qvel[:] = q0
    mujoco.mj_forward(m, d)
    return D


D = damping_fd()
pos_act = [a for a in range(m.nu) if m.actuator_biastype[a] == mujoco.mjtBias.mjBIAS_AFFINE]
kp = m.actuator_gainprm[:, 0]
names = [m.actuator(a).name.replace("_act", "") for a in range(m.nu)]
watch = [a for a in range(m.nu) if any(s in names[a] for s in ("hip_roll", "ankle", "claw", "toe"))]
floor = morph.is_floor
geom_foot = np.asarray(morph.geom_foot)
is_animal = np.asarray(morph.is_animal)
site = [int(f.primary_site) for f in morph.feet]
labels = [f.label for f in morph.feet]


def energy(dd):
    M = np.zeros((m.nv, m.nv))
    mujoco.mj_fullM(m, dd, M)
    ke = 0.5 * dd.qvel @ M @ dd.qvel
    pe = -float(np.dot(m.body_mass, dd.xipos[:, 2])) * m.opt.gravity[2]
    sp = 0.5 * float(sum(kp[a] * (dd.ctrl[a] - dd.actuator_length[a]) ** 2 for a in pos_act))
    return ke + pe + sp


M0 = np.zeros((m.nv, m.nv))
mujoco.mj_fullM(m, d, M0)
ev, V = np.linalg.eig(np.linalg.solve(M0, D))
k = int(np.argmax(ev.real))
dom = int(np.argmax(np.abs(V[:, k])))
print(f"  largest eigenvalue of h M^-1 D at this state {h * ev.real[k]:.3f} (dominant DOF {dom}, joint "
      f"{m.joint(int(m.dof_jntid[dom])).name})")

scratch = mujoco.MjData(m)
f6 = np.zeros(6)
errF, errP, errP_dof = [], [], np.zeros(m.nv)
W = {"constraint_v1": 0.0, "constraint_v0": 0.0, "constraint_mid": 0.0, "damping_v1": 0.0}
E0 = None
vz = []
nf = len(labels)
fn_s, ft_s, cop_s, ext_s, nc_s = [], [], [], [], []
serr = []
for step in range(n):
    scratch.qpos[:] = d.qpos
    scratch.qvel[:] = d.qvel
    scratch.act[:] = d.act
    scratch.ctrl[:] = d.ctrl
    scratch.time = d.time
    scratch.qacc_warmstart[:] = d.qacc_warmstart
    mujoco.mj_forward(m, scratch)
    if E0 is None:
        E0 = energy(scratch)
    M = np.zeros((m.nv, m.nv))
    mujoco.mj_fullM(m, scratch, M)
    v0 = d.qvel.copy()
    fc = scratch.qfrc_constraint.copy()
    vF = v0 + h * np.linalg.solve(M + h * D, scratch.qfrc_smooth + fc)
    vP = v0 + h * scratch.qacc
    fn = np.zeros(nf)
    ft = np.zeros((nf, 2))
    copw = np.zeros((nf, 3))
    ext = [[] for _ in range(nf)]
    nc = np.zeros(nf)
    for i in range(scratch.ncon):
        c = scratch.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        if floor[g1] == floor[g2]:
            continue
        g = g2 if floor[g1] else g1
        if not is_animal[g] or geom_foot[g] < 0:
            continue
        s = int(geom_foot[g])
        mujoco.mj_contactForce(m, scratch, i, f6)
        fw = c.frame.reshape(3, 3).T @ f6[:3]
        if fw[2] < 0:
            fw = -fw
        fn[s] += fw[2]
        ft[s] += fw[:2]
        R = scratch.site_xmat[site[s]].reshape(3, 3)
        loc = R.T @ (c.pos - scratch.site_xpos[site[s]])
        copw[s] += fw[2] * loc
        ext[s].append(loc)
        nc[s] += 1
    fn_s.append(fn)
    ft_s.append(ft)
    cop_s.append(np.where(fn[:, None] > 1e-9, copw / np.maximum(fn[:, None], 1e-12), np.nan))
    ext_s.append([np.ptp(np.array(e), axis=0) if len(e) else np.full(3, np.nan) for e in ext])
    nc_s.append(nc)
    serr.append([scratch.actuator_length[a] - scratch.ctrl[a] for a in watch])
    mujoco.mj_step(m, d)
    v1 = d.qvel.copy()
    errF.append(float(np.abs(v1 - vF).max()))
    e = np.abs(v1 - vP)
    errP.append(float(e.max()))
    errP_dof = np.maximum(errP_dof, e)
    W["constraint_v1"] += h * float(fc @ v1)
    W["constraint_v0"] += h * float(fc @ v0)
    W["constraint_mid"] += h * float(fc @ (0.5 * (v0 + v1)))
    W["damping_v1"] += -h * float(v1 @ D @ v1)
    vz.append(v1[2])
mujoco.mj_forward(m, d)
E1 = energy(d)
T = n * h
worst = int(np.argmax(errP_dof))
print(f"  realised v - [v + h (M+hD)^-1 (qfrc_smooth + qfrc_constraint)]: max {max(errF):.2e}")
print(f"  realised v - solver's v + h qacc: max {max(errP):.4f}, median {np.median(errP):.4f}; worst DOF {worst} "
      f"({m.joint(int(m.dof_jntid[worst])).name}) {errP_dof[worst]:.4f}")
print(f"  energy change {1e3 * (E1 - E0):+.4f} mJ over {T:.3f} s")
for key, val in W.items():
    print(f"  work {key:15s} {1e3 * val:+10.4f} mJ ({1e3 * val / T:+9.3f} mW)")
vz = np.array(vz)
if len(vz) >= 32:
    y = (vz - vz.mean()) * np.hanning(len(vz))
    sp = np.abs(np.fft.rfft(y))
    f = np.fft.rfftfreq(len(vz), h)
    ok = f > 1.0
    print(f"  root vz RMS {1e3 * np.sqrt(np.mean(vz ** 2)):.4f} mm/s; spectral peak {f[ok][np.argmax(sp[ok])]:.2f} Hz "
          f"(resolution {f[1]:.2f} Hz)")
fn_s, ft_s, cop_s, ext_s, nc_s = map(np.array, (fn_s, ft_s, cop_s, ext_s, nc_s))
for s, lab in enumerate(labels):
    fr = np.linalg.norm(ft_s[:, s], axis=1) / np.maximum(fn_s[:, s], 1e-12)
    print(f"  foot {lab}: normal mean {fn_s[:, s].mean():.4f} N (min {fn_s[:, s].min():.4f}, max {fn_s[:, s].max():.4f}); "
          f"tangential mean x {ft_s[:, s, 0].mean():+.4f} y {ft_s[:, s, 1].mean():+.4f} N; friction ratio mean "
          f"{np.nanmean(fr):.3f}; contacts mean {nc_s[:, s].mean():.2f}; load point in foot-site frame mean "
          f"({1e3 * np.nanmean(cop_s[:, s, 0]):+.2f}, {1e3 * np.nanmean(cop_s[:, s, 1]):+.2f}, {1e3 * np.nanmean(cop_s[:, s, 2]):+.2f}) mm; "
          f"contact extent ({1e3 * np.nanmean(ext_s[:, s, 0]):.2f}, {1e3 * np.nanmean(ext_s[:, s, 1]):.2f}, {1e3 * np.nanmean(ext_s[:, s, 2]):.2f}) mm")
serr = np.degrees(np.array(serr))
print("  servo error actuator_length - ctrl (deg; mean over stretch): " + ", ".join(
    f"{names[a]} {serr[:, i].mean():+.3f}" for i, a in enumerate(watch)))
env.close()
