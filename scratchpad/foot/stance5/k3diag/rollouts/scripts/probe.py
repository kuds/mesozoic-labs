import json, sys, numpy as np, mujoco
import importlib.util
S = "/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad"
spec = importlib.util.spec_from_file_location("judge_stance5", S + "/foot/stance5/judge/judge_stance5.py")
J = importlib.util.module_from_spec(spec); sys.modules["judge_stance5"] = J; spec.loader.exec_module(J)
ctx = J.Ctx(sys.argv[1])
env = ctx.env_class(**ctx.env_kwargs)
u = env.unwrapped; m = u.model
nm = lambda t, i: mujoco.mj_id2name(m, t, i)
print("nu", m.nu, "nq", m.nq, "ngeom", m.ngeom, "frame_skip", u.frame_skip, "dt", m.opt.timestep)
for i in range(m.nu):
    j = int(m.actuator_trnid[i, 0])
    print(i, nm(mujoco.mjtObj.mjOBJ_ACTUATOR, i), nm(mujoco.mjtObj.mjOBJ_JOINT, j), "ctrl", m.actuator_ctrlrange[i].round(4).tolist(), "force", m.actuator_forcerange[i].round(1).tolist(), "kp", round(float(m.actuator_gainprm[i,0]),1), "jrange", m.jnt_range[j].round(4).tolist())
for attr in ("_tail_ground_geoms", "_head_ground_geoms", "_body_ground_geoms"):
    print(attr, sorted(nm(mujoco.mjtObj.mjOBJ_GEOM, g) for g in getattr(u, attr)))
print("torso", nm(mujoco.mjtObj.mjOBJ_GEOM, u.torso_geom_id), "floor", nm(mujoco.mjtObj.mjOBJ_GEOM, u.floor_geom_id))
print("healthy_z_range", u.healthy_z_range, "max_tilt", u.max_tilt_angle, "nosedive", u.nosedive_termination_threshold, "natural_fz", u._natural_forward_z)
print("action space", env.action_space.low, env.action_space.high)
print("filter", getattr(u, "action_filter_cutoff_hz", None))
from environments.shared.gait.morphology import Morphology
mo = Morphology.from_env(env, ctx.mid)
print("feet", [(f.label, f.sole_geom, nm(mujoco.mjtObj.mjOBJ_GEOM, f.sole_geom), f.ankle_joint, f.heel_line_m, f.toe_line_m, f.ankle_forward_m) for f in mo.feet])
print("animal geoms", [nm(mujoco.mjtObj.mjOBJ_GEOM, g) for g in mo.animal_geoms])
print("support", [nm(mujoco.mjtObj.mjOBJ_GEOM, g) for g in mo.support_geom_ids])
print("geom_foot", {nm(mujoco.mjtObj.mjOBJ_GEOM, g): int(mo.geom_foot[g]) for g in mo.animal_geoms})
obs, info = env.reset(seed=3042)
print("obs", obs.shape)
o, r, te, tr, inf = env.step(np.zeros(m.nu))
print("info keys", sorted(inf.keys()))
print("bodies", [nm(mujoco.mjtObj.mjOBJ_BODY, b) for b in range(m.nbody)])
print("sites", [nm(mujoco.mjtObj.mjOBJ_SITE, s) for s in range(m.nsite)])
