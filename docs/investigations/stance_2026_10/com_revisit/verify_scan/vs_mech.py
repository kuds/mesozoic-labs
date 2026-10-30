"""Verifier's own mechanism probe (scan F5c, F7 and the mech.py control on robot 143).  Drive the env with env.step
(neutral action) to t_start, then mj_step directly for n substeps; before each, a scratch MjData copy runs mj_forward
(constraint forces, the solver's qacc).  Reports: max |v1 - (v0 + h qacc)| and its DOF, the work h * qfrc_constraint . v1
summed (and . v0), per-foot floor normal / world tangential force, friction ratio, contact count, contact extent in the
foot site's frame, the largest eigenvalue of h M^-1 D (D by finite differences of qfrc_actuator + qfrc_passive).
Usage: python vs_mech.py > out/vs_mech.txt"""
import mujoco
import numpy as np

import vs_lib as L

CASES = [("compy_anat", 313, 0.02, 0.02, 50), ("compy_anat", 313, 0.02, 10.0, 500), ("velo", 126, 0.05, 0.01, 50),
         ("velo", 126, 0.05, 5.0, 500), ("robot", 143, 0.02, 10.0, 1000), ("robot", 141, 0.02, 0.02, 50)]


def damping_fd(m, d, eps=1e-6):
    D = np.zeros((m.nv, m.nv))
    q0 = d.qvel.copy()
    for j in range(m.nv):
        f = []
        for s in (1, -1):
            d.qvel[:] = q0
            d.qvel[j] += s * eps
            mujoco.mj_forward(m, d)
            f.append((d.qfrc_actuator + d.qfrc_passive).copy())
        D[:, j] = -(f[0] - f[1]) / (2 * eps)
    d.qvel[:] = q0
    mujoco.mj_forward(m, d)
    return D


for model, seed, noise, t_start, n in CASES:
    CLS = L.CLASSES[model]
    env = CLS().make_env(reset_noise_scale=noise)
    u = env.unwrapped
    m, d = u.model, u.data
    lay = L.Layout(m, model)
    env.reset(seed=seed)
    a0 = np.zeros(env.action_space.shape, dtype=np.float32)
    for _ in range(int(round(t_start / u.dt))):
        env.step(a0)
    h = m.opt.timestep
    D = damping_fd(m, d)
    M = np.zeros((m.nv, m.nv))
    mujoco.mj_fullM(m, d, M)
    ev = np.linalg.eigvals(np.linalg.solve(M, D)).real
    scratch = mujoco.MjData(m)
    f6 = np.zeros(6)
    err = np.zeros(m.nv)
    errs = []
    W1 = W0 = 0.0
    fn_s, ft_s, nc_s, ext_s = [], [], [], []
    for _ in range(n):
        scratch.qpos[:] = d.qpos
        scratch.qvel[:] = d.qvel
        scratch.act[:] = d.act
        scratch.ctrl[:] = d.ctrl
        scratch.time = d.time
        scratch.qacc_warmstart[:] = d.qacc_warmstart
        mujoco.mj_forward(m, scratch)
        fc = scratch.qfrc_constraint.copy()
        v0 = d.qvel.copy()
        vP = v0 + h * scratch.qacc
        fn = np.zeros(2)
        ft = np.zeros((2, 2))
        nc = np.zeros(2)
        pts = [[], []]
        for i in range(scratch.ncon):
            c = scratch.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            if (g1 == lay.floor) == (g2 == lay.floor):
                continue
            g = g2 if g1 == lay.floor else g1
            k = lay.geom_leg[g]
            if not lay.geom_animal[g] or k < 0:
                continue
            mujoco.mj_contactForce(m, scratch, i, f6)
            fw = c.frame.reshape(3, 3).T @ f6[:3]
            if fw[2] < 0:
                fw = -fw
            fn[k] += fw[2]
            ft[k] += fw[:2]
            nc[k] += 1
            R = scratch.site_xmat[lay.sites[k]].reshape(3, 3)
            pts[k].append(R.T @ (c.pos - scratch.site_xpos[lay.sites[k]]))
        fn_s.append(fn)
        ft_s.append(ft)
        nc_s.append(nc)
        ext_s.append([np.ptp(np.array(p), axis=0)[:2] if len(p) else np.full(2, np.nan) for p in pts])
        mujoco.mj_step(m, d)
        v1 = d.qvel.copy()
        e = np.abs(v1 - vP)
        err = np.maximum(err, e)
        errs.append(e.max())
        W1 += h * float(fc @ v1)
        W0 += h * float(fc @ v0)
    fn_s, ft_s, nc_s, ext_s = map(np.array, (fn_s, ft_s, nc_s, ext_s))
    wd = int(np.argmax(err))
    print(f"{model} seed {seed} noise {noise}: t {t_start:.2f}-{t_start + n * h:.2f} s ({n} substeps); largest eig h M^-1 D "
          f"{h * ev.max():.3f}; max |v1 - (v0 + h qacc)| {err.max():.4f} on {m.joint(int(m.dof_jntid[wd])).name} (median of "
          f"per-substep max {np.median(errs):.4f}); contact work along v1 {1e3 * W1:+.4f} mJ, along v0 {1e3 * W0:+.4f} mJ")
    for k in range(2):
        fr = np.linalg.norm(ft_s[:, k], axis=1) / np.maximum(fn_s[:, k], 1e-12)
        print(f"   foot {k} ({m.site(lay.sites[k]).name}): normal mean {fn_s[:, k].mean():.3f} N (min {fn_s[:, k].min():.3f}); "
              f"tangential mean x {ft_s[:, k, 0].mean():+.3f} y {ft_s[:, k, 1].mean():+.3f} N; friction ratio mean "
              f"{np.nanmean(fr):.3f}; contacts mean {nc_s[:, k].mean():.2f}; contact extent in site frame "
              f"({1e3 * np.nanmean(ext_s[:, k, 0]):.2f}, {1e3 * np.nanmean(ext_s[:, k, 1]):.2f}) mm")
    env.close()
