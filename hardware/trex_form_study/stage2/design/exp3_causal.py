"""E3: a causal stop: the preview controller starts AT the command frame from the periodic walk's own CoM state, so
nothing before the command knows of it.  Preview CoM only (no IK) unless "full"."""
import sys, time
import numpy as np
from designlab import gait
from designlab.stage1 import refs, transitions as tr
from designlab.stage2 import look
import proto_moves as pm
from proto_moves import const, _ramp

body, which = sys.argv[1], sys.argv[2]
full = "full" in sys.argv[3:]
gait.set_accuracy("r2")
src = refs.source(body)
M, q0, H = src["M"], src["q_home"], src["H"]
spec = look.speed_spec(H)
a = pm.gait_from_spec("v0.4", spec, H)
stand = tr.Gait("stand")
ref_a = pm.periodic_ref(src, "V0_v0.4", spec)
geo = tr.Geometry.of(M, q0, a.crouch)
d, n = a.duty, 160


def state_at(ref, phase0, first=None):
    """(c, c', c'') of the periodic walk at gait phase ``phase0`` (any multiple of 1/n), in the frame of a plan whose
    path origin is at phase ``first`` (default phase0): the straight walk's frame shifted by -first strides."""
    first = phase0 if first is None else first
    mot = tr._stride_motion(ref); inv = tr._inverse(mot)
    c, dt = ref.com_ref, ref.step
    k0 = int(round(phase0 * n))
    shift = np.array([-first * float(ref.stride_motion[3]), 0.0])

    def at(k):
        s, kk = divmod(k0 + k, n)
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


def walk_com(ref, phases):
    mot = tr._stride_motion(ref); inv = tr._inverse(mot)
    shift = np.array([-phases[0] * float(ref.stride_motion[3]), 0.0])
    out = []
    for p in phases:
        s, kk = divmod(int(round(p * n)), n)
        x = ref.com_ref[kk]
        for _ in range(abs(s)):
            x = tr._move(x, inv if s < 0 else mot)
        out.append(x + shift)
    return np.array(out)


def cart_zmp(t, com, omega):
    acc = np.gradient(np.gradient(com, t, axis=0), t, axis=0)
    return com - acc / omega**2


fn_pre = dict(T=const(a.period), v=const(a.speed), w=const(0.0))
fn_walk = dict(T=const(a.period), duty=const(d), lift=const(a.lift), v=const(a.speed), w=const(0.0), amp=const(1.0),
               head_amp=const(a.head_amp), tail_amp=const(a.tail_amp), crouch=const(a.crouch))
# (1) start-up transient of a controller started mid-stride on the continuing walk
for p0 in (-2.0, -0.5, -0.25):
    wk = pm.gplan("walk", "walk", a, a, geo, fn_walk, {}, (p0, 3), pre=0.0)
    com = tr.com_path(wk, wk.zmp, state_at(ref_a, p0), geo.omega)
    dev = 1e3 * np.linalg.norm(com - walk_com(ref_a, wk.phase), axis=1)
    print(f"walk continued from phase {p0:+.2f} with no warm-up stride: preview CoM against the periodic com_ref: max {dev.max():.4f} mm "
          f"(first stride {dev[:n].max():.4f} mm)")


def stop_fn(p_cmd, p_end):
    end = 1.0 - d + tr.SETTLE_STRIDES
    down = lambda p: 1.0 - _ramp((np.asarray(p, float) - p_cmd) / (p_end - p_cmd))
    amp = lambda p: 1.0 - _ramp((np.asarray(p, float) - p_cmd) / (end - p_cmd))
    fn = dict(T=const(a.period), duty=const(d), lift=const(a.lift), v=lambda p: a.speed * down(p), w=const(0.0), amp=amp,
              head_amp=const(a.head_amp), tail_amp=const(a.tail_amp), crouch=const(a.crouch))
    return fn, dict(path=[p_cmd, p_end], amplitude=[p_cmd, end], settle=[1.0 - d, end])


cases = {"causal two-step: command -0.5 (right lift-off), ramp to 1-d": (-0.5, 1.0 - d),
         "causal two-step: command -0.5, ramp to 0.2": (-0.5, 0.2),
         "causal one-step: command 0.0 (left lift-off), ramp 0.4": (0.0, 0.4),
         "causal one-step: command 0.0, ramp to 1-d": (0.0, 1.0 - d)}
keep = {}
for label, (p_cmd, p_end) in cases.items():
    fn, ramps = stop_fn(p_cmd, p_end)
    pl = pm.gplan("stop2", "stop", a, stand, geo, fn, ramps, (p_cmd, 3), lift_max=0.0, cmd=p_cmd, fn_pre=fn_pre, pre=0.0)
    x0 = state_at(ref_a, p_cmd)
    com = tr.com_path(pl, pl.zmp, x0, geo.omega)
    t, ph = pl.t, pl.phase
    zc = cart_zmp(t, com, geo.omega)
    ez = 1e3 * np.linalg.norm(zc - pl.zmp, axis=1)[3:-3]
    tu, lp, raw = pm.lp_forward_acc(t, com, pl.heading)
    v_f = np.gradient(com[:, 0], t)
    k_stop = next(k for k in range(len(t)) if np.all(np.abs(v_f[k:]) < 0.05 * a.speed))
    tds = sorted((round(s[2], 3), s[0]) for s in pl.swings if s[2] > p_cmd + 1e-9)
    vpath = np.gradient(pl.path[:, 0], t)
    print(f"\n{label}")
    print(f"  touchdowns after the command {tds}; steps {[(s[0], round(s[1], 3), round(float(np.linalg.norm(s[5][0] - s[4][0])), 4)) for s in pl.swings if s[1] >= p_cmd - 1e-9]}")
    print(f"  cart-table ZMP against the target: max {ez.max():.2f} mm at phase {ph[3:-3][int(np.argmax(ez))]:+.3f}; forward part max {1e3 * np.abs((zc - pl.zmp)[3:-3, 0]).max():.2f} mm")
    print(f"  path decel peak {np.gradient(vpath, t).min():.3f}; CoM forward acc raw min {raw.min():.3f}, 1 Hz low-passed min {lp.min():.3f} max {lp.max():.3f} m/s^2")
    print(f"  forward speed < 5 % of the command from {t[k_stop]:.3f} s = {t[k_stop] / a.period:.3f} strides after the command; CoM overshoot past the final stand {1e3 * (com[:, 0].max() - pl.stand_point['final'][0]):.2f} mm")
    keep[label] = (pl, x0)
if full:
    label = [k for k in keep if which in k][0] if which != "V0" else list(keep)[0]
    pl, x0 = keep[list(keep)[0]]
    # transitions.initial_state would refuse the mid-stride start, so solve with the state given
    delta = pm.walk_correction(ref_a, geo.omega)
    print("walk correction |delta| max mm", 1e3 * np.abs(delta).max(0))
    fn0, _ = stop_fn(-0.5, 1.0 - d)
    kk = np.round(pl.phase * n).astype(int) % n
    zref0 = pl.zmp + fn0["amp"](pl.phase)[:, None] * delta[kk] if "warm" in sys.argv else None
    g = pm.solve(src, pl, geo, ref_a, a.zmp_iters, a.zmp_gain, x0=x0, zref0=zref0)
    stand_f = pl.stand_point["final"]
    dc = 1e3 * np.linalg.norm(g["com"][-n:, :2] - stand_f, axis=1)
    dz = 1e3 * np.linalg.norm(g["zmp_mb"][-n:] - stand_f, axis=1)
    o = pl.out
    tu, lp, raw = pm.lp_forward_acc(pl.t[o], g["com"][:, :2], pl.heading[o])
    k0 = int(round((-0.5 % 1.0) * n))
    dq = np.abs(g["q"][0, 7:] - ref_a.q[k0, 7:]).max()
    print(f"\nFULL {list(keep)[0]}: {g['seconds']:.0f} s, ik_ok {g['ik_ok']} ({g['ik_fail']} frames), ZMP history mm {[round(1e3 * x, 3) for x in g['zmp_error_history_m']]}")
    print(f"  join at the command frame: joints max |dq| {dq:.2e} rad against the walk's frame {k0}; join to the stand (last stride): CoM {dc.max():.4f} mm, ZMP {np.nanmax(dz):.4f} mm")
    print(f"  model-CoM forward acc, 1 Hz: min {lp.min():.3f} max {lp.max():.3f} m/s^2; hip-yaw use {pm.hip_yaw_use(M, g['q'])}")
    e = 1e3 * np.linalg.norm(np.nan_to_num(g["zmp_mb"] - pl.zmp[o]), axis=1)
    print(f"  final ZMP error by phase: first 0.25 stride {e[:40].max():.2f} mm, 0.25-1 stride {e[40:160].max():.2f} mm, later {e[160:].max():.2f} mm")
