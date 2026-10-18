import numpy as np, mujoco, environments
assert "main_ro" in environments.__file__, environments.__file__
print(environments.__file__)
from environments.shared.model_id import ModelId
from environments.shared.config import load_stage_config
from environments.shared.species_registry import env_class as model_env_class
from environments.shared.gait.morphology import Morphology
mid = ModelId("tyrannosaurus_rex", "anatomical")
cfg = load_stage_config(mid, 1)
kw = dict(cfg["env_kwargs"])
print({k: kw[k] for k in ("timestep","frame_skip","max_episode_steps") if k in kw})
env = model_env_class(mid)(**kw)
m = env.unwrapped.model
print("nq nv nu nbody ngeom", m.nq, m.nv, m.nu, m.nbody, m.ngeom, "cone", m.opt.cone, "dt", m.opt.timestep, "frame_skip", env.unwrapped.frame_skip)
print("action filter", env.unwrapped.action_filter_cutoff_hz)
for i in range(m.nu):
    j = m.actuator_trnid[i,0]
    print(i, m.actuator(i).name, "joint", m.joint(j).name, "gain", m.actuator_gainprm[i,0], "bias", m.actuator_biasprm[i,:3], "ctrlrange", m.actuator_ctrlrange[i], "forcelimited", m.actuator_forcelimited[i], "forcerange", m.actuator_forcerange[i], "home", env.unwrapped._home_ctrl[i], "jnt_range", m.jnt_range[j], "actfrc", m.jnt_actfrclimited[j], m.jnt_actfrcrange[j])
for b in range(m.nbody):
    print("body", b, m.body(b).name, "parent", m.body_parentid[b], "mass", round(float(m.body_mass[b]),3))
morph = Morphology.from_env(env, mid)
import json
print(json.dumps(morph.describe(m), indent=1, default=str))
for g in morph.support_geom_ids:
    print("support geom", m.geom(g).name, "type", m.geom_type[g], "size", m.geom_size[g], "body", m.body(m.geom_bodyid[g]).name, "friction", m.geom_friction[g], "condim", m.geom_condim[g])
print("root body", m.body(morph.root_body).name, "subtreemass", m.body_subtreemass[morph.root_body], "BW", morph.body_weight_n)
print("feet ankle_forward", [f.ankle_forward_m for f in morph.feet], "span", [(f.heel_line_m, f.toe_line_m) for f in morph.feet])
print("sole axes", [(f.sole_normal_axis, f.sole_normal_sign, f.sole_forward_axis, f.sole_forward_sign, f.sole_lateral_axis, f.sole_lateral_sign) for f in morph.feet])
print("floor", morph.floor_geoms, [m.geom(g).name for g in morph.floor_geoms], m.geom_friction[morph.floor_geoms[0]])
print("opt", m.opt.impratio, m.opt.noslip_iterations, m.opt.gravity)
u = env.unwrapped
for name in ("pelvis_id","skull_body_id","head_tip_site_id","_natural_forward_z","nosedive_termination_threshold","healthy_z_range"):
    print(name, getattr(u, name, None))
print("obs dim", env.observation_space.shape)
env.close()
