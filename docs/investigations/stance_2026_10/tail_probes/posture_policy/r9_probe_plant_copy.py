"""CPU probe of the r9 T. rex tail plant (no checkpoints, no training).

Part A: actuator / joint parameters for the tail chain.
Part B: noise-free statue (home keyframe, home controls) settled 3 s; gravity
        torque about tail_1/2/3 pitch, spring, actuator, constraint torques.
Part C: tail_3 command sweep a in [-1, 1] on the standing statue (other
        commands at home), settled 4 s: q, torques, tail tip, tail_3 world pitch,
        COM fore-aft vs feet.
Part D: kinematic sweep of q3 over its joint range at the settled pose (all else
        frozen): gravity torque about tail_3, the actuator torque and the
        ctrl a static hold needs (ctrl = q + tau_req/kp), and the normalised
        action a = ctrl/0.2094.
Part E: same for tail_1 / tail_2 (minimum command that lifts them off the
        ventral stop).
"""

import sys

import mujoco
import numpy as np

XML = sys.argv[1]
m = mujoco.MjModel.from_xml_path(XML)
d = mujoco.MjData(m)
np.set_printoptions(precision=4, suppress=True)


def jid(n):
    return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)


def aid(n):
    return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n)


def bid(n):
    return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)


TJ = ["tail_1_pitch", "tail_1_yaw", "tail_2_pitch", "tail_3_pitch"]
TA = ["tail_1_pitch_act", "tail_1_yaw_act", "tail_2_pitch_act", "tail_3_pitch_act"]

print("== A. parameters ==")
print("timestep", m.opt.timestep, "integrator", m.opt.integrator)
for j, a in zip(TJ, TA):
    J, A = jid(j), aid(a)
    print(
        f"{j}: range {m.jnt_range[J]} stiffness {m.jnt_stiffness[J]} springref {m.qpos_spring[m.jnt_qposadr[J]]:.4f} "
        f"damping {m.dof_damping[m.jnt_dofadr[J]]} armature {m.dof_armature[m.jnt_dofadr[J]]} "
        f"margin {m.jnt_margin[J]} solref {m.jnt_solref[J]} solimp {m.jnt_solimp[J][:3]}"
    )
    print(
        f"   {a}: gain {m.actuator_gainprm[A][:3]} bias {m.actuator_biasprm[A][:3]} gear {m.actuator_gear[A][0]} "
        f"ctrlrange {m.actuator_ctrlrange[A]} ctrllimited {m.actuator_ctrllimited[A]} forcerange {m.actuator_forcerange[A]} "
        f"forcelimited {m.actuator_forcelimited[A]} dyntype {m.actuator_dyntype[A]} biastype {m.actuator_biastype[A]}"
    )
for b in ["tail_1", "tail_2", "tail_3", "tail_4", "tail_5", "pelvis"]:
    print(f"body {b}: mass {m.body_mass[bid(b)]:.3f}")
print("total mass", m.body_subtreemass[0] - m.body_mass[bid("prey")], "(excl. prey)")

key = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home")
home_ctrl = m.key_ctrl[key].copy()
print("home ctrl tail", home_ctrl[[aid(a) for a in TA]])


def reset_home():
    mujoco.mj_resetDataKeyframe(m, d, key)
    d.ctrl[:] = home_ctrl
    mujoco.mj_forward(m, d)


def tail_torques():
    """Per tail-joint generalized forces at the current state."""
    out = {}
    # gravity: qfrc_bias at zero velocity is -gravity force; compute separately
    qv = d.qvel.copy()
    d.qvel[:] = 0
    mujoco.mj_forward(m, d)
    grav = -d.qfrc_bias.copy()
    d.qvel[:] = qv
    mujoco.mj_forward(m, d)
    for j in TJ:
        dof = m.jnt_dofadr[jid(j)]
        out[j] = dict(
            q=d.qpos[m.jnt_qposadr[jid(j)]],
            grav=grav[dof],
            spring=-m.jnt_stiffness[jid(j)] * (d.qpos[m.jnt_qposadr[jid(j)]] - m.qpos_spring[m.jnt_qposadr[jid(j)]]),
            act=d.qfrc_actuator[dof],
            constraint=d.qfrc_constraint[dof],
        )
    return out


def tail3_world_pitch():
    # angle of the tail_3 segment axis (local -x) above horizontal, deg
    R = d.xmat[bid("tail_3")].reshape(3, 3)
    v = R @ np.array([-1.0, 0, 0])
    return np.degrees(np.arctan2(v[2], -v[0]))


def foot_mid_x():
    xs = []
    for s in ["r_foot", "l_foot"]:
        sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, s)
        if sid >= 0:
            xs.append(d.site_xpos[sid][0])
    return np.mean(xs) if xs else np.nan


sites = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SITE, i) for i in range(m.nsite)]
print("sites:", sites)

print("\n== B. noise-free statue, home controls, settled 3 s ==")
reset_home()
for _ in range(int(3.0 / m.opt.timestep)):
    mujoco.mj_step(m, d)
mujoco.mj_forward(m, d)
T = tail_torques()
for j, v in T.items():
    print(f"{j}: q {v['q']:+.4f} grav {v['grav']:+7.2f} spring {v['spring']:+7.2f} act {v['act']:+7.2f} constraint {v['constraint']:+7.2f}")
print(f"tail_3 segment world pitch {tail3_world_pitch():+.2f} deg; tail tip z {d.site_xpos[sites.index('tail_tip')][2]:.4f}")
settled_qpos = d.qpos.copy()
settled_qvel = d.qvel.copy()
pelvis_R = d.xmat[bid("pelvis")].reshape(3, 3)
print("pelvis forward_z", pelvis_R[2, 0])

print("\n== C. tail_3 command sweep on the standing statue (others at home), settle 4 s ==")
print(" a      ctrl    q1       q2       q3      act3   grav3  spring3  constr3  |act3|/50  t3_world_deg  tip_z   comx-footx(mm)")
for a in [-1.0, -0.5, 0.0, 0.5, 0.8, 0.9, 0.95, 1.0]:
    reset_home()
    c = a * 0.2094
    d.ctrl[aid("tail_3_pitch_act")] = c
    for _ in range(int(4.0 / m.opt.timestep)):
        mujoco.mj_step(m, d)
    mujoco.mj_forward(m, d)
    T = tail_torques()
    comx = d.subtree_com[bid("pelvis")][0]
    print(
        f"{a:+.2f} {c:+.4f} {T['tail_1_pitch']['q']:+.4f} {T['tail_2_pitch']['q']:+.4f} {T['tail_3_pitch']['q']:+.4f} "
        f"{T['tail_3_pitch']['act']:+6.2f} {T['tail_3_pitch']['grav']:+6.2f} {T['tail_3_pitch']['spring']:+6.2f} "
        f"{T['tail_3_pitch']['constraint']:+6.2f}   {abs(T['tail_3_pitch']['act'])/50:.3f}     {tail3_world_pitch():+6.2f}    "
        f"{d.site_xpos[sites.index('tail_tip')][2]:.4f}  {1000*(comx-foot_mid_x()):+.1f}"
    )

print("\n== D. kinematic q3 sweep at the settled pose (all else frozen) ==")
kp3 = m.actuator_gainprm[aid("tail_3_pitch_act")][0]
k3 = m.jnt_stiffness[jid("tail_3_pitch")]
qa3 = m.jnt_qposadr[jid("tail_3_pitch")]
dof3 = m.jnt_dofadr[jid("tail_3_pitch")]
cmax = m.actuator_ctrlrange[aid("tail_3_pitch_act")][1]
print(" q3      grav3   spring3  tau_act_req  ctrl_req  a_req   offset(ctrl-q)  world_deg")
for q3 in np.linspace(-0.2094, 0.2094, 13):
    d.qpos[:] = settled_qpos
    d.qvel[:] = 0
    d.qpos[qa3] = q3
    mujoco.mj_forward(m, d)
    grav = -d.qfrc_bias[dof3]
    spring = -k3 * q3
    tau_req = -(grav + spring)
    ctrl = q3 + tau_req / kp3
    print(f"{q3:+.4f} {grav:+7.2f} {spring:+7.2f}   {tau_req:+7.2f}    {ctrl:+.4f}  {ctrl/cmax:+.3f}   {ctrl-q3:+.4f}       {tail3_world_pitch():+6.2f}")

print("\n== E. tail_1 / tail_2: torque balance on the stop at the settled pose ==")
d.qpos[:] = settled_qpos
d.qvel[:] = 0
mujoco.mj_forward(m, d)
for j, a in [("tail_1_pitch", "tail_1_pitch_act"), ("tail_2_pitch", "tail_2_pitch_act")]:
    J, A = jid(j), aid(a)
    dof = m.jnt_dofadr[J]
    q = d.qpos[m.jnt_qposadr[J]]
    grav = -d.qfrc_bias[dof]
    spring = -m.jnt_stiffness[J] * q
    kp = m.actuator_gainprm[A][0]
    cm = m.actuator_ctrlrange[A][1]
    # actuator torque at ctrl = cm, q on the stop
    tmax = min(kp * (cm - q), m.actuator_forcerange[A][1])
    lo = m.jnt_range[J][0]
    tau_req_at_stop = -(grav - m.jnt_stiffness[J] * lo)
    ctrl_req = lo + tau_req_at_stop / kp
    print(
        f"{j}: q {q:+.4f} grav {grav:+7.2f} spring {spring:+7.2f}; max actuator torque at a=+1 on the stop {tmax:+.2f}; "
        f"ctrl needed just to lift off the stop {ctrl_req:+.4f} (a = {ctrl_req/cm:+.2f}); forcerange {m.actuator_forcerange[A]}"
    )
