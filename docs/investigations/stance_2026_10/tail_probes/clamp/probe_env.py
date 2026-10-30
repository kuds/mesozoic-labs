"""What tail_3_pitch_act's action means in the stance env (code_copy).  Read only; no training."""
import json, sys
import numpy as np
import mujoco
from environments.shared.config import load_stage_config
from environments.shared.model_id import ModelId
from environments.shared.species_registry import env_class

MID = ModelId("tyrannosaurus_rex", "anatomical")
ctor = dict(load_stage_config(MID, "stance")["env_kwargs"])
env = env_class(MID)(**ctor)
u = env.unwrapped
m = u.model
names = [m.actuator(i).name for i in range(m.nu)]
print("nu", m.nu, "action space", env.action_space.low.min(), env.action_space.high.max(), env.action_space.shape)
print("env class", type(u).__name__, "action_mapping", getattr(u, "action_mapping", None),
      "action_filter_cutoff_hz", u.action_filter_cutoff_hz, "dt", u.dt, "alpha", u._action_filter_alpha,
      "action_penalty_source", u.action_penalty_source)
for i, n in enumerate(names):
    j = int(m.actuator_trnid[i, 0])
    jn = m.joint(j).name
    print(f"{i:2d} {n:18s} joint {jn:16s} ctrlrange [{m.actuator_ctrlrange[i,0]:+.4f},{m.actuator_ctrlrange[i,1]:+.4f}] "
          f"home {u._home_ctrl[i]:+.5f} jnt_range_deg [{np.degrees(m.jnt_range[j,0]):+.2f},{np.degrees(m.jnt_range[j,1]):+.2f}] "
          f"gainprm0 {m.actuator_gainprm[i,0]:.1f} biasprm {m.actuator_biasprm[i,:3].round(2).tolist()} "
          f"forcerange {m.actuator_forcerange[i].round(1).tolist()} dyntype {int(m.actuator_dyntype[i])} gaintype {int(m.actuator_gaintype[i])}")
i3 = names.index("tail_3_pitch_act")
j3 = int(m.actuator_trnid[i3, 0])
print("tail_3 index", i3, "qpos adr", m.jnt_qposadr[j3], "spring stiffness", m.jnt_stiffness[j3], "springref", m.qpos_spring[m.jnt_qposadr[j3]])
a = np.zeros(m.nu); a[i3] = 1.0
print("scale(+1) tail_3 ctrl", u._scale_action(a)[i3], " scale(0)", u._scale_action(np.zeros(m.nu))[i3], " scale(-1)", u._scale_action(-a)[i3])
obs, _ = env.reset(seed=3042)
obs, r, term, trunc, info = env.step(np.zeros(m.nu, dtype=np.float32))
keys = sorted(k for k, v in info.items() if np.isscalar(v) or (isinstance(v, np.ndarray) and v.ndim == 0))
print("info scalar keys:", keys)
print("info nonscalar keys:", sorted(k for k in info if k not in keys))
