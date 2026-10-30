"""Checks of the observation conventions used by tests 2 and 3.  Read only, no training.

PYTHONPATH=<policies/code_copy>:<this folder> python -B probe_obs.py > probe_obs.txt

1. The layout in pp_common (names, indices) against the live _get_obs at a settled statue state.
2. Projected gravity: obs[42:45] == -R[2,:] from the root quaternion; pitch sign (+ nose up = -g_x); the
   accelerometer at rest reads about -9.81 g (imu site has the pelvis frame).
3. pitch_illusion / roll_illusion against the env's own heading_free_root_segments on a quaternion rotated about the
   pelvis y / x axis, so the illusion is exactly what the policy would see were the pelvis pitched.
4. The normaliser scale (sqrt(var)) of the posture inputs for start_s42, one s42 end and one s80 end.
"""

import math

import mujoco
import numpy as np

from environments.shared.heading_observation import heading_free_root_segments
from pp_common import (I_ACC, I_GRAV, I_PREY, JOINTS, OBS_NAMES, TAIL3, Pol, make_env, pitch_deg_from_obs,
                       pitch_illusion, roll_illusion, stance_ctor)

ctor = stance_ctor()
env = make_env(ctor)
u = env.unwrapped
m, d = u.model, u.data
assert m.actuator(TAIL3).name == "tail_3_pitch_act"
obs, _ = env.reset(seed=3042)
for _ in range(500):
    obs, *_ = env.step(np.zeros(m.nu))
print("1. layout")
ok = True
for j, name in enumerate(JOINTS):
    jid = m.joint(name).id
    qa = int(m.jnt_qposadr[jid]) - 7
    va = (m.nq - 7) + int(m.jnt_dofadr[jid]) - 6
    good = qa == j and va == 21 + j and np.float32(d.qpos[qa + 7]) == obs[qa] and np.float32(d.qvel[va - 21 + 6]) == obs[va]
    ok &= bool(good)
print("   joint angle/velocity indices match the live obs:", ok)
R = d.xmat[u.pelvis_id].reshape(3, 3)
fwdz = R[2, 0]
print("2. gravity obs", obs[I_GRAV], " -R[2,:]", -R[2, :], " pelvis forward_z", fwdz,
      " pitch from obs (deg, + nose up)", float(pitch_deg_from_obs(obs)), " asin(fwdz)", math.degrees(math.asin(fwdz)))
print("   accelerometer obs", obs[I_ACC], " -9.81*g", -9.81 * obs[I_GRAV])
print("   prey direction body", obs[I_PREY], " prey distance", obs[59], " command", obs[60:63])
print("   foot contacts", obs[54:56])
q0 = d.qpos.copy()
quat = q0[3:7].copy()
for th in (+1.1, -1.1, +2.0):
    # rotate the root about its own y axis by alpha = -th (nose up by th)
    rq = np.zeros(4)
    mujoco.mju_axisAngle2Quat(rq, np.array([0.0, 1.0, 0.0]), -math.radians(th))
    nq = np.zeros(4)
    mujoco.mju_mulQuat(nq, quat, rq)
    q1 = q0.copy()
    q1[3:7] = nq
    prey_rel = d.mocap_pos[0] - d.xpos[u.pelvis_id]
    pdir = prey_rel / (np.linalg.norm(prey_rel) + 1e-8)
    g1, _, pb1 = heading_free_root_segments(q1, d.qvel[0:3], pdir)
    ill = pitch_illusion(obs, th)
    print(f"3. nose-up {th:+.1f} deg: env gravity {g1.round(6)} illusion {np.asarray(ill[I_GRAV], float).round(6)}"
          f" | env prey {pb1.round(6)} illusion {np.asarray(ill[I_PREY], float).round(6)}"
          f" | pitch from illusion {float(pitch_deg_from_obs(ill)):+.4f} (base {float(pitch_deg_from_obs(obs)):+.4f})")
for ph in (+1.0,):
    rq = np.zeros(4)
    mujoco.mju_axisAngle2Quat(rq, np.array([1.0, 0.0, 0.0]), math.radians(ph))
    nq = np.zeros(4)
    mujoco.mju_mulQuat(nq, quat, rq)
    q1 = q0.copy()
    q1[3:7] = nq
    g1, _, _ = heading_free_root_segments(q1, d.qvel[0:3], np.array([1.0, 0, 0]))
    ill = roll_illusion(obs, ph)
    print(f"3. roll +x {ph:+.1f} deg: env gravity {g1.round(6)} illusion {np.asarray(ill[I_GRAV], float).round(6)}")
env.close()
print("4. normaliser scale sqrt(var+eps) of selected inputs (and running mean)")
sel = [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 14, 16, 17, 42, 43, 44, 45, 46, 51, 52, 53, 54, 55, 56, 57, 58, 59]
for who in ("start_s42", "s42_A_1101", "s80_A_1301", "start_s80"):
    pol = Pol(who, lambda: make_env(ctor))
    print("  ", who, " ".join(f"{OBS_NAMES[i]}:{pol.std[i]:.4g}({pol.mu[i]:+.3g})" for i in sel))
