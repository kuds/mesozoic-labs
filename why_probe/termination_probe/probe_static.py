"""Static checks before any rollout (no rollout, no statistic of rollouts):
  - the model's joint order (obs index 6 = tail_3_pitch?), actuator 14 = tail_3_pitch_act, the tail geoms and site;
  - the code copy's stance env_kwargs against each run's recorded reward_weights;
  - each checkpoint pair: sha256, num_timesteps, obs/ret normaliser values, policy architecture and log_std[14];
  - forward kinematics at the home keyframe: d(tail tip z)/d q and d(lowest tail geom z)/d q for the tail pitch joints;
  - the capsule lowest-point formula against mujoco.mj_geomDistance to the floor plane.
PYTHONPATH not needed: tp_common puts the code copy first.  Run: PY -I -B probe_static.py > probe_static.txt
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tp_common as T  # noqa: E402

origin = T.setup_simulation("static")
print("environments origin:", origin)

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

factory = T.env_factory()
env = factory()
u = env.unwrapped
m, d = u.model, u.data
print("nq", m.nq, "nv", m.nv, "nu", m.nu, "obs dim", env.observation_space.shape)
jn = [m.joint(j).name for j in range(m.njnt)]
qadr = [int(m.jnt_qposadr[j]) for j in range(m.njnt)]
print("joints (name, qposadr, obs index = qposadr - 7):")
for n, a in zip(jn, qadr):
    print(f"   {n:16s} qpos {a:3d} obs {a - 7 if a >= 7 else '-'}")
print("obs[6] joint:", [n for n, a in zip(jn, qadr) if a - 7 == T.OBS_TAIL3])
print("actuator 14:", m.actuator(T.ACT_TAIL3).name, "ctrlrange", m.actuator_ctrlrange[T.ACT_TAIL3])
print("actuators:", [m.actuator(i).name for i in range(m.nu)])
tail_geoms = [u.tail_3_geom_id, u.tail_4_geom_id, u.tail_5_geom_id]
for g in tail_geoms:
    print("geom", m.geom(g).name, "type", int(m.geom_type[g]), "(capsule=3)", "size", m.geom_size[g])
print("tail_tip site", u.tail_tip_site_id, "pelvis body", u.pelvis_id)
print("healthy_z_range", u.healthy_z_range, "max_tilt_angle", u.max_tilt_angle, "fall_penalty", u.fall_penalty,
      "max_episode_steps", u.max_episode_steps, "action_filter_cutoff_hz", getattr(u, "action_filter_cutoff_hz", None))

print("\n== env kwargs vs each run's recorded reward_weights")
for lab in T.ORDER:
    ok, diffs = T.ctor_matches_run(T.CKPT[lab][2])
    print(f"   {lab:15s} match={ok} diffs={diffs}")

print("\n== checkpoint pairs")
for lab in T.ORDER:
    z, p, cfg, form = T.CKPT[lab]
    model, norm = T.load_pair(lab, factory)
    pol = model.policy
    rms = norm.obs_rms
    print(f"-- {lab}: zip {z}\n   zip sha256 {T.sha256_file(z)}\n   pkl {p}\n   pkl sha256 {T.sha256_file(p)}")
    print(f"   num_timesteps {model.num_timesteps}  critic form {form}  class {type(model).__name__}")
    print(f"   obs6 mean {math.degrees(float(rms.mean[6])):+.4f} deg  sd {math.degrees(math.sqrt(float(rms.var[6]) + norm.epsilon)):.4f} deg"
          f"  eps {norm.epsilon}  clip_obs {norm.clip_obs}")
    print(f"   ret_rms var {float(norm.ret_rms.var):.6g}  sqrt(var+eps) {math.sqrt(float(norm.ret_rms.var) + norm.epsilon):.6g}"
          f"  count {float(norm.ret_rms.count):.6g}  clip_reward {norm.clip_reward}  gamma {norm.gamma}"
          f"  norm_reward(now) {norm.norm_reward} training {norm.training}")
    print(f"   net_arch {pol.net_arch}  activation {pol.activation_fn}  share_features {pol.share_features_extractor}"
          f"  ortho {pol.ortho_init}  log_std[14] {float(pol.log_std[14]):+.4f} (std {math.exp(float(pol.log_std[14])):.4f})"
          f"  log_std mean {float(pol.log_std.mean()):+.4f}")
    print(f"   mlp_extractor class {type(pol.mlp_extractor).__name__}; value_net {pol.value_net}")

print("\n== forward kinematics at the home keyframe (scratch MjData; level pelvis as spawned)")
dk = mujoco.MjData(m)
dk.qpos[:] = m.key_qpos[u.home_keyframe_id]
mujoco.mj_kinematics(m, dk)
base_tip = float(dk.site_xpos[u.tail_tip_site_id][2])
base_low = T.tail_capsule_lowest(m, dk, tail_geoms)
print(f"   pelvis z {float(dk.xpos[u.pelvis_id][2]):.4f}  tail tip z {base_tip:.4f}  lowest tail_3/4/5 {np.round(base_low, 4)}")
for jname in ("tail_1_pitch", "tail_2_pitch", "tail_3_pitch"):
    a = int(m.jnt_qposadr[m.joint(jname).id])
    res = []
    for dd in (-1.0, +1.0):
        dk2 = mujoco.MjData(m)
        dk2.qpos[:] = m.key_qpos[u.home_keyframe_id]
        dk2.qpos[a] += math.radians(dd)
        mujoco.mj_kinematics(m, dk2)
        res.append((float(dk2.site_xpos[u.tail_tip_site_id][2]), min(T.tail_capsule_lowest(m, dk2, tail_geoms))))
    print(f"   {jname}: d tip_z/d q = {(res[1][0] - res[0][0]) / 2 * 1000:+.2f} mm/deg;"
          f" d lowest/d q = {(res[1][1] - res[0][1]) / 2 * 1000:+.2f} mm/deg (obs index {a - 7})")
# pelvis pitch sensitivity: rotate the root about the world y axis by +-1 deg (nose up is negative rotation about +y?)
q0 = m.key_qpos[u.home_keyframe_id].copy()
for dd in (-2.0, +2.0):
    dk3 = mujoco.MjData(m)
    dk3.qpos[:] = q0
    quat = np.zeros(4)
    mujoco.mju_axisAngle2Quat(quat, np.array([0.0, 1.0, 0.0]), math.radians(dd))
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, quat, q0[3:7])
    dk3.qpos[3:7] = out
    mujoco.mj_kinematics(m, dk3)
    print(f"   root rotated {dd:+.1f} deg about world +y: tail tip z {float(dk3.site_xpos[u.tail_tip_site_id][2]):.4f}"
          f" head tip z {float(dk3.site_xpos[u.head_tip_site_id][2]):.4f}")

print("\n== capsule lowest-point formula against mj_geomDistance(floor, geom) at the home keyframe and a pitched state")
for dd in (0.0, 20.0, 30.0):
    dk4 = mujoco.MjData(m)
    dk4.qpos[:] = q0
    quat = np.zeros(4)
    mujoco.mju_axisAngle2Quat(quat, np.array([0.0, 1.0, 0.0]), math.radians(-dd))
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, quat, q0[3:7])
    dk4.qpos[3:7] = out
    mujoco.mj_forward(m, dk4)
    f = T.tail_capsule_lowest(m, dk4, tail_geoms)
    gd = [float(mujoco.mj_geomDistance(m, dk4, u.floor_geom_id, g, 10.0, None)) for g in tail_geoms]
    print(f"   root rotated {-dd:+.1f} deg about +y: formula {np.round(f, 5)}  geomDistance {np.round(gd, 5)}"
          f"  tip z {float(dk4.site_xpos[u.tail_tip_site_id][2]):.4f}  pelvis z {float(dk4.xpos[u.pelvis_id][2]):.4f}")
env.close()
print("done")
