"""Shared helpers for the T-Rex Stage 1 investigation experiments (not repo code)."""
import sys
import numpy as np
import mujoco

REPO = "checkout"
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from environments.shared.config import build_env  # noqa: E402
from environments.shared.model_id import ModelId  # noqa: E402

MID = ModelId("tyrannosaurus_rex", "anatomical")


def make_env():
    env = build_env(MID, 1)
    return env, env.unwrapped


def names(u):
    m = u.model
    jn = [m.joint(i).name for i in range(m.njnt)]
    an = [m.actuator(i).name for i in range(m.nu)]
    return jn, an


def hinge_info(u):
    """Return list of (name, qposadr, dofadr, range, limited, stiffness, springref) for hinge joints."""
    m = u.model
    out = []
    for i in range(m.njnt):
        if m.jnt_type[i] != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        qa, da = m.jnt_qposadr[i], m.jnt_dofadr[i]
        out.append(
            dict(
                name=m.joint(i).name,
                qadr=int(qa),
                dadr=int(da),
                range=m.jnt_range[i].copy(),
                limited=bool(m.jnt_limited[i]),
                stiffness=float(m.jnt_stiffness[i]),
                springref=float(m.qpos_spring[qa]),
                damping=float(m.dof_damping[da]),
            )
        )
    return out


def actuator_info(u):
    m = u.model
    out = []
    for i in range(m.nu):
        jid = m.actuator_trnid[i, 0]
        out.append(
            dict(
                name=m.actuator(i).name,
                joint=m.joint(jid).name,
                qadr=int(m.jnt_qposadr[jid]),
                dadr=int(m.jnt_dofadr[jid]),
                kp=float(m.actuator_gainprm[i, 0]),
                kv=float(-m.actuator_biasprm[i, 2]),
                ctrlrange=m.actuator_ctrlrange[i].copy(),
                forcerange=m.actuator_forcerange[i].copy(),
                jrange=m.jnt_range[jid].copy(),
                home=float(u._home_ctrl[i]),
            )
        )
    return out


def kinetic_energy(u):
    m, d = u.model, u.data
    res = np.zeros(m.nv)
    mujoco.mj_mulM(m, d, res, d.qvel)
    return 0.5 * float(d.qvel @ res)


def floor_penetration(u):
    """Most negative floor-contact distance (m) and number of floor contacts."""
    d = u.data
    fl = u.floor_geom_id
    dmin, n = 0.0, 0
    for i in range(d.ncon):
        c = d.contact[i]
        if c.geom1 == fl or c.geom2 == fl:
            n += 1
            dmin = min(dmin, float(c.dist))
    return dmin, n


def term_keys(info):
    return sorted(k for k in info if k.startswith("reward_") and k != "reward_total")


def custom_reset(env, u, seed, qpos_noise=0.0, qvel_noise=0.0, root_vel_noise=True):
    """Noise-free env reset followed by a controlled re-injection of noise components."""
    saved = u.reset_noise_scale
    u.reset_noise_scale = 0.0
    obs, info = env.reset(seed=seed)
    u.reset_noise_scale = saved
    rng = np.random.default_rng(10_000 + seed)
    if qpos_noise > 0:
        u.data.qpos[7:] += rng.uniform(-qpos_noise, qpos_noise, size=u.data.qpos[7:].shape)
    if qvel_noise > 0:
        dq = rng.uniform(-qvel_noise, qvel_noise, size=u.data.qvel.shape)
        if not root_vel_noise:
            dq[:6] = 0.0
        u.data.qvel[:] += dq
    u._settle_root_on_ground()
    mujoco.mj_forward(u.model, u.data)
    return u._get_obs(), info


def rollout(env, u, policy, steps=1000, reset=None, record_obs=False):
    """Run one episode; policy(t, obs, u) -> action. Returns dict of per-step arrays."""
    obs, _ = reset() if reset is not None else env.reset()
    m, d = u.model, u.data
    H = hinge_info(u)
    qadr = np.array([h["qadr"] for h in H])
    dadr = np.array([h["dadr"] for h in H])
    rec = dict(
        z=[], vroot=[], wroot=[], q=[], qd=[], ctrl=[], afrc=[], ke=[], pen=[], ncon=[],
        terms={}, total=[], peak_bw=[], tilt=[], fz=[], act=[], filt=[], head_tip_z=[],
    )
    if record_obs:
        rec["obs"] = [obs.copy()]
    # state at reset (t = -1 marker stored separately)
    rec["reset_q"] = d.qpos[qadr].copy()
    rec["reset_qacc"] = d.qacc.copy()
    rec["reset_pen"] = floor_penetration(u)
    rec["reset_z"] = float(d.qpos[2])
    end_reason = "truncated"
    n = 0
    for t in range(steps):
        a = np.asarray(policy(t, obs, u), dtype=np.float32)
        obs, r, term, trunc, info = env.step(a)
        n += 1
        if record_obs:
            rec["obs"].append(obs.copy())
        rec["z"].append(float(d.qpos[2]))
        rec["vroot"].append(d.qvel[0:3].copy())
        rec["wroot"].append(d.qvel[3:6].copy())
        rec["q"].append(d.qpos[qadr].copy())
        rec["qd"].append(d.qvel[dadr].copy())
        rec["ctrl"].append(d.ctrl.copy())
        rec["afrc"].append(d.actuator_force.copy())
        rec["ke"].append(kinetic_energy(u))
        p, nc = floor_penetration(u)
        rec["pen"].append(p)
        rec["ncon"].append(nc)
        rec["total"].append(float(r))
        rec["peak_bw"].append(float(info.get("peak_foot_force_bw", np.nan)))
        rec["tilt"].append(float(info.get("tilt_angle", np.nan)))
        rec["fz"].append(float(info.get("forward_z", np.nan)))
        rec["head_tip_z"].append(float(info.get("head_tip_z", np.nan)))
        rec["act"].append(a.copy())
        rec["filt"].append(None if u._action_filter_state is None else u._action_filter_state.copy())
        for k in term_keys(info):
            rec["terms"].setdefault(k, []).append(float(info[k]))
        if term or trunc:
            end_reason = info.get("termination_reason", "truncated") if term else "truncated"
            break
    for k in list(rec):
        if k in ("terms", "reset_q", "reset_qacc", "reset_pen", "reset_z"):
            continue
        if k == "filt":
            continue
        rec[k] = np.asarray(rec[k])
    rec["terms"] = {k: np.asarray(v) for k, v in rec["terms"].items()}
    rec["n"] = n
    rec["end"] = end_reason
    return rec


def zero_policy(t, obs, u):
    return np.zeros(u.model.nu)
