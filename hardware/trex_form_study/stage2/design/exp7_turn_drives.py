"""E7: Stage 0 drive ratios of the prototype quick turn (cache npz from exp6 full) and of the periodic walk beside it."""
import sys, time
import numpy as np
from designlab import envelope, gait, sizing, statics, turning
from designlab.stage1 import refs, transitions as tr
from designlab.stage2 import screen
import proto_moves as pm

body, tag = sys.argv[1], sys.argv[2]
gait.set_accuracy("r2")
src = refs.source(body)
M, design = src["M"], src["design"]
z = np.load(pm.SCR / "cache" / f"{body}_{tag}_proto.npz")
q, qd, qdd, swing, t = z["q"], z["qd"], z["qdd"], z["swing"], z["t"]
ok = np.all(np.isfinite(qdd), axis=1)
t0 = time.perf_counter()
tau, cop, normal, fric = tr.idyn(M, q[ok], qd[ok], qdd[ok], swing[ok], "exact")
om = qd[ok][:, M.act_dofs]
bus = src["record"]["selection"]["bus"]
ms = sizing._motor_space(M.names, tau, om, src["layout"])
rat = envelope._ratios(design, bus, ms)
single = np.sum(swing[ok], axis=1) == 1
print(f"{body} {tag}: {int(ok.sum())} frames, idyn {time.perf_counter() - t0:.0f} s; single-support CoP margin min {1e3 * np.nanmin(cop[single]):.1f} mm, "
      f"all frames {1e3 * np.nanmin(cop):.1f} mm; friction ratio max {fric.max():.3f}")
for g, v in rat.items():
    print(f"   {g:10s} cont {v['cont_ratio']:.3f} peak {v['peak_ratio']:.3f} speed {v['speed_ratio']:.3f} worst {v['worst']:.3f}")
pas = screen.passive_loads(M, design, tau, om)
print("   passive: " + ", ".join(f"{k} cont {v['cont_ratio_total']:.3f} peak {v['peak_ratio_total']:.3f}" for k, v in pas.items()))
# torsional friction need in single support (turning.inverse_dynamics' formula, frame by frame)
M.forward(q[0])
r_eff = {s: turning.uniform_radius(*turning.pad_samples(M, s)) for s in statics.SIDES}
mu = []
for k in np.flatnonzero(single):
    kk = np.flatnonzero(ok)[k]
    side = statics.SIDES[int(np.flatnonzero(~swing[kk])[0])]
    tk, wr, _ = turning.contact_torques(M, q[kk], qd[kk], qdd[kk], (side,), "exact")
    f, mom = wr[side][:3], wr[side][3:]
    p, _ = M.foot_pose(side)
    c = statics.cop(M, side, wr[side])
    mz = float(mom[2] + np.cross(p - np.array([c[0], c[1], 0.0]), f)[2])
    mu.append(float(np.hypot(np.linalg.norm(f[:2]) / f[2], mz / (f[2] * r_eff[side]))))
print(f"   single-support mu_req max {max(mu):.3f} (turning's torsion formula), frames over 0.54: {int(np.sum(np.array(mu) > 0.54))} of {len(mu)}")
