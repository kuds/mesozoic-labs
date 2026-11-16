"""E4: the causal two-step stop from a set's 0.40 m/s walk, full solve (proto_causal)."""
import sys
import numpy as np
from designlab import gait
from designlab.stage1 import refs, transitions as tr
from designlab.stage2 import look
import proto_moves as pm
import proto_causal as pc
from proto_moves import const, _ramp

body, which = sys.argv[1], sys.argv[2]
p_cmd = float(sys.argv[3]) if len(sys.argv) > 3 else -0.5
gait.set_accuracy("r2")
src = refs.source(body)
M, q0, H = src["M"], src["q_home"], src["H"]
if which == "V0":
    spec = look.speed_spec(H)
else:
    v = 0.40
    fr = v * v / (gait.G * H)
    sr = round(min(0.94, max(0.62, 0.863 * ((max(fr, 0.025) / 0.05) ** 0.20))), 4)
    spec = gait.GaitSpec(froude=fr, duty=0.59, lift_frac=0.095, head_amp=0.05, tail_amp=-0.125, sway_gain=1.0,
                         stride_rel=sr, crouch=0.05, zmp_iters=6, zmp_gain=0.5, n=160)
a = pm.gait_from_spec("v0.4", spec, H)
stand = tr.Gait("stand")
ref_a = pm.periodic_ref(src, f"{which}_v0.4", spec)
geo = tr.Geometry.of(M, q0, a.crouch)
d, n = a.duty, 160
p_end = 1.0 - d
end = 1.0 - d + tr.SETTLE_STRIDES
legacy_plan = "legacy" in sys.argv
if legacy_plan:   # Stage 1's stop plan exactly (ramp about phi_c, amplitude from phase 0), only the command is earlier
    pc0 = tr.stop_command_phase(d)
    down = lambda p: 1.0 - _ramp((np.asarray(p, float) - pc0) / (2 * tr.PATH_HALF_WIDTH))
    amp = lambda p: 1.0 - _ramp(np.asarray(p, float) / end)
else:
    down = lambda p: 1.0 - _ramp((np.asarray(p, float) - p_cmd) / (p_end - p_cmd))
    amp = lambda p: 1.0 - _ramp((np.asarray(p, float) - p_cmd) / (end - p_cmd))
fn = dict(T=const(a.period), duty=const(d), lift=const(a.lift), v=lambda p: a.speed * down(p), w=const(0.0), amp=amp,
          head_amp=const(a.head_amp), tail_amp=const(a.tail_amp), crouch=const(a.crouch))
fn_pre = dict(T=const(a.period), v=const(a.speed), w=const(0.0))
fn_walk = dict(T=const(a.period), duty=const(d), lift=const(a.lift), v=const(a.speed), w=const(0.0), amp=const(1.0),
               head_amp=const(a.head_amp), tail_amp=const(a.tail_amp), crouch=const(a.crouch))
last = 3
pl = pm.gplan("stop2", "stop", a, stand, geo, fn, {}, (p_cmd, last), lift_max=0.0, cmd=None if legacy_plan else p_cmd, fn_pre=fn_pre, pre=1.0)
wl = pm.gplan("walk", "walk", a, a, geo, fn_walk, {}, (p_cmd, last + 6), pre=1.0)
delta = pm.walk_correction(ref_a, geo.omega)
g = pc.solve_causal(src, pl, geo, ref_a, wl, delta, amp, a.zmp_iters, a.zmp_gain, p_cmd)
o = pl.out
k_cmd = o.start
# the warm-up stride against the walk itself, and the join at the command frame
cw = pc.walk_frames(ref_a, pl.phase[:k_cmd + 1], pl.first, "com")
zw = pc.walk_frames(ref_a, pl.phase[:k_cmd + 1], pl.first, "zmp_mb")
kq = np.round(pl.phase[:k_cmd + 1] * n).astype(int) % n
dq = np.abs(g["q_all"][:k_cmd + 1, 7:] - ref_a.q[kq, 7:]).max(1)
dc = 1e3 * np.linalg.norm(g["com_all"][:k_cmd + 1, :2] - cw, axis=1)
dz = 1e3 * np.linalg.norm(g["zmb_all"][:k_cmd + 1] - zw, axis=1)
print(f"\n{body} {which} causal two-step stop, command phase {p_cmd:+.3f}: {g['seconds']:.0f} s, ik_ok {g['ik_ok']} ({g['ik_fail']}), "
      f"ZMP history mm {[round(1e3 * x, 3) for x in g['zmp_error_history_m']]}")
print(f"  warm-up stride against the walk: CoM max {dc.max():.4f} mm, joints max {dq.max():.2e} rad, ZMP max {np.nanmax(dz):.4f} mm")
print(f"  join at the command frame: CoM {dc[-1]:.4f} mm, ZMP {dz[-1]:.4f} mm, joints {dq[-1]:.2e} rad")
stand_f = pl.stand_point["final"]
jc = 1e3 * np.linalg.norm(g["com"][-n:, :2] - stand_f, axis=1)
jz = 1e3 * np.linalg.norm(g["zmp_mb"][-n:] - stand_f, axis=1)
t = pl.t[o] - pl.t[o][0]
tu, lp, raw = pm.lp_forward_acc(t, g["com"][:, :2], pl.heading[o])
v_f = np.gradient(g["com"][:, 0], t)
k_stop = next(k for k in range(len(t)) if np.all(np.abs(v_f[k:]) < 0.05 * a.speed))
e = 1e3 * np.linalg.norm(np.nan_to_num(g["zmp_mb"] - pl.zmp[o]), axis=1)
zcart = g["com_ref"] - np.gradient(np.gradient(g["com_ref"], t, axis=0), t, axis=0) / geo.omega**2
print(f"  join to the stand (last stride): CoM {jc.max():.4f} mm, ZMP {np.nanmax(jz):.4f} mm")
print(f"  ZMP error by part: first 0.1 stride {e[:16].max():.2f}, 0.1-0.5 {e[16:80].max():.2f}, 0.5-1.5 {e[80:240].max():.2f}, later {e[240:].max():.2f} mm")
print(f"  model CoM forward acc, 1 Hz low-passed: min {lp.min():.3f} max {lp.max():.3f} m/s^2 (raw min {raw[5:-5].min():.2f}); forward speed < 5 % from {t[k_stop]:.3f} s = {t[k_stop] / a.period:.3f} strides after the command")
tds = sorted((round(s[2], 3), s[0]) for s in pl.swings if s[2] > p_cmd + 1e-9)
print(f"  touchdowns after the command {tds}; last touchdown {max(x[0] for x in tds) - p_cmd:.3f} strides = {(max(x[0] for x in tds) - p_cmd) * a.period:.3f} s after it; steps {[(s[0], round(s[1], 3), round(float(np.linalg.norm(s[5][0] - s[4][0])), 4)) for s in pl.swings if s[1] >= p_cmd - 1e-9]}")
np.savez(pm.SCR / "cache" / f"{body}_{which}_stop2_proto.npz", t=t, q=g["q"], com=g["com"], zmp_mb=g["zmp_mb"], phase=pl.phase[o], swing=pl.swing[o])
